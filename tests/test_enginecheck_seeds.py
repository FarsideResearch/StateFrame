"""Seed registry and proof checks, run against throwaway git repos with a fake module (no engine involved).

Fake history: commit BAD has compute() returning 1 and no fakemod_extra; commit HEAD fixes both.
"""
from __future__ import annotations

import shutil
import subprocess
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
from enginecheck.seeds import Seed, SeedRegistryError, collect_seeds, prove_seed

ROOT = Path(__file__).resolve().parents[1]
_IDENT = ("-c", "user.name=enginecheck", "-c", "user.email=enginecheck@localhost", "-c", "commit.gpgsign=false")

SEED_FILE = textwrap.dedent(
    """\
    import os

    import pytest

    import fakemod


    @pytest.mark.race_seed("01", known_bad="BAD", expect="AssertionError")
    def test_known_bad_compute():
        assert fakemod.compute() == 2


    @pytest.mark.race_seed("02", mutant="flip", expect="AssertionError")
    def test_mutant_flip():
        if os.environ.get("ENGINECHECK_RACE_MUTANT") == "flip":
            fakemod.compute = lambda: 0
        assert fakemod.compute() == 2


    @pytest.mark.race_seed("03", known_bad="BAD", expect="AssertionError")
    def test_red_by_import_error():
        from fakemod_extra import EXTRA

        assert EXTRA == 2


    @pytest.mark.race_seed("04", known_bad="BAD", expect="AssertionError")
    def test_vacuous_on_bad_tree():
        assert fakemod.LABEL == "ok"
    """
)


VALID_SEED = 'import pytest\n\n\n@pytest.mark.race_seed("10", known_bad="abc1234", expect="AssertionError")\ndef test_ok():\n    pass\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *_IDENT, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _install_race_conftest(repo: Path) -> None:
    _write(repo, "tests/race_seeds/__init__.py", "")
    shutil.copyfile(ROOT / "tests/race_seeds/conftest.py", repo / "tests/race_seeds/conftest.py")


@pytest.fixture
def fake_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, "fakemod.py", "def compute():\n    return 1\n\nLABEL = 'ok'\n")
    bad = _commit(repo, "bug: compute returns 1")
    _write(repo, "fakemod.py", "def compute():\n    return 2\n\nLABEL = 'ok'\n")
    _write(repo, "fakemod_extra.py", "EXTRA = 2\n")
    _write(repo, "tests/enginecheck/__init__.py", "")
    _write(repo, "tests/race_seeds/test_fake_seeds.py", SEED_FILE.replace("BAD", bad))
    _install_race_conftest(repo)
    _commit(repo, "fix: compute returns 2")
    return repo, bad


def _seed(repo: Path, seed_id: str) -> Seed:
    matches = [seed for seed in collect_seeds(repo) if seed.seed_id == seed_id]
    assert len(matches) == 1, f"expected one seed {seed_id}, got {matches}"
    return matches[0]


def test_known_bad_seed_is_red_by_assertion_on_bad_tree_and_green_on_head(fake_repo, tmp_path):
    repo, _bad = fake_repo
    proof = prove_seed(_seed(repo, "01"), repo, tmp_path / "work")
    assert proof.green and proof.red_intended and proof.ok, proof.detail
    assert "assert 1 == 2" in proof.detail


def test_mutant_seed_is_green_normally_and_red_with_mutant_active(fake_repo, tmp_path):
    repo, _bad = fake_repo
    seed = _seed(repo, "02")
    assert seed.mutant == "flip" and seed.known_bad is None
    proof = prove_seed(seed, repo, tmp_path / "work")
    assert proof.green and proof.red_intended and proof.ok, proof.detail
    assert "assert 0 == 2" in proof.detail


def test_red_by_import_error_is_not_a_red_proof(fake_repo, tmp_path):
    repo, _bad = fake_repo
    proof = prove_seed(_seed(repo, "03"), repo, tmp_path / "work")
    assert proof.green and not proof.red_intended and not proof.ok
    assert "ModuleNotFoundError" in proof.detail


def test_vacuous_seed_that_is_also_green_on_bad_tree_is_not_ok(fake_repo, tmp_path):
    repo, _bad = fake_repo
    proof = prove_seed(_seed(repo, "04"), repo, tmp_path / "work")
    assert proof.green and not proof.red_intended and not proof.ok


def test_parent_mutant_env_does_not_leak_into_green_run(fake_repo, tmp_path, monkeypatch):
    """A mutant leaked from the parent env must not turn the GREEN run of an unmutated seed into a failure."""
    repo, _bad = fake_repo
    monkeypatch.setenv("ENGINECHECK_RACE_MUTANT", "flip")
    proof = prove_seed(_seed(repo, "02"), repo, tmp_path / "work")
    assert proof.green and proof.ok, proof.detail


def test_unknown_known_bad_sha_raises_with_git_stderr(fake_repo, tmp_path):
    repo, _bad = fake_repo
    missing = replace(_seed(repo, "01"), known_bad="0123456789abcdef0123456789abcdef01234567")
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(SeedRegistryError, match="git archive .* failed.*fatal"):
        prove_seed(missing, repo, work)
    assert list(work.iterdir()) == []


def test_known_bad_export_dir_is_removed_after_proof(fake_repo, tmp_path):
    repo, _bad = fake_repo
    work = tmp_path / "work"
    work.mkdir()
    proof = prove_seed(_seed(repo, "01"), repo, work)
    assert proof.ok, proof.detail
    assert list(work.iterdir()) == []


def test_known_bad_overlay_comes_from_committed_harness_not_working_tree(fake_repo, tmp_path):
    """Dirty the working-tree seed so it passes on the bad tree; the proof must still judge the committed seed."""
    repo, _bad = fake_repo
    seed_path = repo / "tests/race_seeds/test_fake_seeds.py"
    original = seed_path.read_text(encoding="utf-8")
    dirty = original.replace("    assert fakemod.compute() == 2\n", "    pass\n", 1)
    assert dirty != original
    seed_path.write_text(dirty, encoding="utf-8")
    proof = prove_seed(_seed(repo, "01"), repo, tmp_path / "work")
    assert proof.ok, proof.detail
    assert "assert 1 == 2" in proof.detail


def test_stray_race_seed_marker_outside_seed_dir_fails_collection(tmp_path):
    repo = tmp_path / "repo"
    _write(repo, "tests/race_seeds/test_ok.py", VALID_SEED)
    _write(repo, "tests/test_stray.py", VALID_SEED.replace('"10"', '"11"').replace("def test_ok", "def test_stray"))
    _install_race_conftest(repo)
    with pytest.raises(SeedRegistryError, match="outside tests/race_seeds.*test_stray.py"):
        collect_seeds(repo)


def test_unregistered_marker_in_seed_dir_fails_strict_collection(tmp_path):
    repo = tmp_path / "repo"
    _write(repo, "tests/race_seeds/test_ok.py", VALID_SEED)
    _write(repo, "tests/race_seeds/test_bogus.py", "import pytest\n\n\n@pytest.mark.bogus_unregistered\ndef test_b():\n    pass\n")
    _install_race_conftest(repo)
    with pytest.raises(SeedRegistryError, match="bogus_unregistered"):
        collect_seeds(repo)


_BAD_MARKERS = {
    "no expect": ('race_seed("01", known_bad="abc1234")', "expect is required"),
    "both known_bad and mutant": (
        'race_seed("01", known_bad="abc1234", mutant="m", expect="AssertionError")',
        "exactly one of known_bad",
    ),
    "neither known_bad nor mutant": ('race_seed("01", expect="AssertionError")', "exactly one of known_bad"),
}


@pytest.mark.parametrize("marker, reason", list(_BAD_MARKERS.values()), ids=list(_BAD_MARKERS))
def test_marker_misuse_fails_collection_with_clear_error(tmp_path, marker, reason):
    repo = tmp_path / "repo"
    _write(repo, "tests/race_seeds/test_bad.py", f"import pytest\n\n\n@pytest.mark.{marker}\ndef test_bad():\n    pass\n")
    _install_race_conftest(repo)
    with pytest.raises(SeedRegistryError, match=f"invalid race_seed marker.*{reason}"):
        collect_seeds(repo)


def test_duplicate_seed_id_is_an_error(tmp_path):
    repo = tmp_path / "repo"
    body = 'import pytest\n\n\n@pytest.mark.race_seed("07", known_bad="abc1234", expect="AssertionError")\n'
    _write(repo, "tests/race_seeds/test_one.py", body + "def test_one():\n    assert False\n")
    _write(repo, "tests/race_seeds/test_two.py", body + "def test_two():\n    assert False\n")
    _install_race_conftest(repo)
    with pytest.raises(SeedRegistryError, match="duplicate seed id '07'"):
        collect_seeds(repo)
