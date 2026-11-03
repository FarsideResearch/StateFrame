"""Tree digest encoding: unambiguous, versioned, and old pins fail closed."""

import hashlib
import json
import os
import random
import stat
from pathlib import Path

import pytest

from stateframe import powerup_state, powerups
from stateframe.powerup_manifest import tree_digest


def _tree(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        os.chmod(path, 0o644)
    return root


def _v1_digest(root: Path) -> str:
    """The retired ambiguous encoding, kept here only to prove old pins are refused."""
    digest = hashlib.sha256()
    for current, directories, files in os.walk(root):
        for name in sorted([*directories, *files]):
            item = Path(current) / name
            relative = item.relative_to(root).as_posix()
            if item.is_dir():
                digest.update(b"D" + relative.encode() + b"\0")
            else:
                digest.update(b"F" + relative.encode() + b"\0")
                digest.update(str(stat.S_IMODE(item.lstat().st_mode)).encode() + b"\0")
                digest.update(item.read_bytes())
    return digest.hexdigest()


def test_reported_collision_pair_has_distinct_digests(tmp_path):
    payload = b'{"schema":"stateframe-tools/1","tools":[]}'
    smuggled = _tree(tmp_path / "a", {"a": b"Fcommands.json\x00420\x00" + payload})
    honest = _tree(tmp_path / "b", {"a": b"", "commands.json": payload})
    assert _v1_digest(smuggled) == _v1_digest(honest)
    assert tree_digest(smuggled) != tree_digest(honest)


def test_random_splits_and_renames_never_collide(tmp_path):
    rng = random.Random(20261103)
    seen: dict[str, tuple] = {}
    for index in range(150):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))
        cuts = sorted(rng.sample(range(len(blob) + 1), k=min(len(blob) + 1, rng.randrange(1, 4))))
        pieces = [blob[a:b] for a, b in zip([0, *cuts], [*cuts, len(blob)])]
        names = sorted(rng.sample(["a", "b", "ab", "a\x00b", "c/d", "cd", "e"], k=len(pieces))) \
            if len(pieces) <= 7 else []
        files = {name.replace("\x00", "_"): piece for name, piece in zip(names, pieces)}
        key = tuple(sorted(files.items()))
        digest = tree_digest(_tree(tmp_path / f"t{index}", files))
        assert seen.setdefault(digest, key) == key


def test_file_dir_and_symlink_at_same_path_differ(tmp_path):
    as_file = _tree(tmp_path / "f", {"x": b""})
    as_dir = tmp_path / "d"
    (as_dir / "x").mkdir(parents=True)
    assert tree_digest(as_file) != tree_digest(as_dir)
    as_link = tmp_path / "l"
    as_link.mkdir()
    (as_link / "x").symlink_to(as_file / "x")
    with pytest.raises(powerups.PowerupError, match="symlinks"):
        tree_digest(as_link)


def test_merging_files_changes_digest(tmp_path):
    split = _tree(tmp_path / "s", {"a": b"12", "b": b"34"})
    merged = _tree(tmp_path / "m", {"a": b"1234", "b": b""})
    moved = _tree(tmp_path / "v", {"a": b"", "b": b"1234"})
    assert len({tree_digest(split), tree_digest(merged), tree_digest(moved)}) == 3


def test_corrupt_state_fails_closed_for_default_on_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    repo = tmp_path / "repo"
    source = repo / ".stateframe" / "powerups" / "sample"
    source.parent.mkdir(parents=True)
    _unit(source.parent)
    toml = source / "powerup.toml"
    toml.write_text(toml.read_text().replace('default = "off"', 'default = "on"'))
    powerups.trust(repo)
    assert [u.name for u in powerups.resolve_units(repo, "", "default", (), (), False).active] == ["sample"]
    (home / "powerups.json").write_text("{corrupt")
    notices = []
    selected = powerups.resolve_units(repo, "", "default", (), (), False, printer=notices.append)
    assert not selected.active
    assert selected.skipped[0]["code"] == "state_invalid"
    assert any("powerups disabled for this run" in notice for notice in notices)


def test_mode_and_empty_entries_change_digest(tmp_path):
    base = _tree(tmp_path / "x", {"a": b"1"})
    first = tree_digest(base)
    os.chmod(base / "a", 0o755)
    assert tree_digest(base) != first
    os.chmod(base / "a", 0o644)
    (base / "empty").write_bytes(b"")
    assert tree_digest(base) != first


def _unit(root: Path) -> Path:
    path = root / "sample"
    path.mkdir(parents=True)
    (path / "powerup.toml").write_text(
        'schema = "stateframe-powerup/1"\nname = "sample"\nversion = "1.0"\n'
        'summary = "Useful sample."\ndefault = "off"\n'
        "[activate]\nskill = true\ncommands = true\nhooks = true\n"
    )
    (path / "SKILL.md").write_text("---\nname: sample\ndescription: Useful sample.\n---\nRead.\n")
    return path


def test_enable_records_digest_version(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    installed = powerups.install(_unit(tmp_path))
    powerups.enable(installed.name)
    row = powerup_state.user_state()["global"]["sample"]
    assert row["digest_version"] == 2
    repo = tmp_path / "repo"
    repo.mkdir()
    assert [u.name for u in powerups.resolve_units(repo, "", "default", (), (), False).active] == ["sample"]


@pytest.mark.parametrize("old_digest", ["v1", "current"])
def test_old_format_pin_requires_retrust(tmp_path, monkeypatch, old_digest):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    installed = powerups.install(_unit(tmp_path))
    powerups.enable(installed.name)
    state_path = tmp_path / "home" / "powerups.json"
    state = json.loads(state_path.read_text())
    row = state["global"]["sample"]
    row.pop("digest_version")
    row["digest"] = _v1_digest(installed.path) if old_digest == "v1" else tree_digest(installed.path)
    state_path.write_text(json.dumps(state))
    repo = tmp_path / "repo"
    repo.mkdir()
    selected = powerups.resolve_units(repo, "", "default", (), (), False)
    assert not selected.active
    assert "re-trust required" in selected.skipped[0]["reason"]
    powerups.enable(installed.name)
    assert [u.name for u in powerups.resolve_units(repo, "", "default", (), (), False).active] == ["sample"]


def test_resume_refuses_record_without_digest_version(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _unit(tmp_path)
    selected = powerups.resolve_units(tmp_path, "read", "default", (str(source) + "=offer",), (), False)
    record = selected.active[0].record()
    assert record["tree_digest_version"] == 2
    powerups.verify_records({"active": [record], "skipped": []})
    old = dict(record)
    old.pop("tree_digest_version")
    with pytest.raises(powerups.PowerupError, match="POWERUP_CHANGED.*re-trust required"):
        powerups.verify_records({"active": [old], "skipped": []})


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "powerups_v1"


def test_upgrade_from_v1_pins_requires_retrust(tmp_path, monkeypatch):
    """The fixtures were written by the pre-fix code (tree ddd7e6a1): install, enable, trust."""
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / "home"
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    installed = powerups.install(FIXTURES / "sample")
    (home / "powerups.json").write_bytes((FIXTURES / "powerups.json").read_bytes())
    repo = tmp_path / "repo"
    (repo / ".stateframe" / "powerups").mkdir(parents=True)
    repo = repo.resolve()
    project_unit = repo / ".stateframe" / "powerups" / "sample"
    import shutil
    shutil.copytree(FIXTURES / "sample", project_unit)
    store = tmp_path / ".stateframe"
    store.mkdir(mode=0o700)
    (store / "trusted-hooks.json").write_text(
        (FIXTURES / "trusted-hooks.json").read_text().replace("@REPO@", str(repo)))
    os.chmod(store / "trusted-hooks.json", 0o600)

    notices: list[str] = []
    selected = powerups.resolve_units(repo, "", "default", (), (), False, printer=notices.append)
    assert not selected.active
    assert "re-trust required" in selected.skipped[0]["reason"]
    assert any(n.startswith("powerup sample disabled: re-trust required") for n in notices)

    shutil.rmtree(project_unit)
    notices.clear()
    selected = powerups.resolve_units(repo, "", "default", (), (), False, printer=notices.append)
    assert not selected.active
    assert notices == [
        "powerup sample disabled: re-trust required: pinned with an older digest format; "
        "run 'stateframe powerups enable sample' after review"
    ]
    powerups.enable(installed.name)
    assert [u.name for u in powerups.resolve_units(repo, "", "default", (), (), False).active] == ["sample"]


def test_unreadable_entry_fails_closed(tmp_path):
    root = _tree(tmp_path / "u", {"a": b"1", "sub/b": b"2"})
    os.chmod(root / "sub", 0)
    try:
        if os.access(root / "sub", os.R_OK):
            pytest.skip("running with privileges that ignore directory modes")
        with pytest.raises(powerups.PowerupError, match="unreadable"):
            tree_digest(root)
    finally:
        os.chmod(root / "sub", 0o755)


def test_materialize_refuses_copy_that_differs_from_pin(tmp_path, monkeypatch):
    from stateframe import powerup_materialize

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    source = _unit(tmp_path)
    real_copy = powerup_materialize.shutil.copytree

    def swapped(src, dst, **kwargs):
        result = real_copy(src, dst, **kwargs)
        (Path(dst) / "commands.json").write_text('{"schema":"stateframe-tools/1","tools":[]}')
        return result

    monkeypatch.setattr(powerup_materialize.shutil, "copytree", swapped)
    selected = powerups.resolve_units(tmp_path, "", "default", (str(source),), (), False)
    assert not selected.active
    assert "does not match its pin" in selected.skipped[0]["reason"]


def test_changed_trusted_project_state_is_refused_with_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    repo = (tmp_path / "repo")
    (repo / ".stateframe").mkdir(parents=True)
    repo = repo.resolve()
    state = repo / ".stateframe" / "powerups.json"
    state.write_text(json.dumps({"schema": "stateframe-powerup-state/1",
                                 "global": {"sample": {"mode": "off"}}, "profiles": {}}))
    powerups.trust(repo)
    state.write_text(json.dumps({"schema": "stateframe-powerup-state/1",
                                 "global": {}, "profiles": {}}))
    notices: list[str] = []
    selected = powerups.resolve_units(repo, "", "default", (), (), False, printer=notices.append)
    assert not selected.active
    assert selected.skipped[0]["code"] == "state_invalid"
    assert notices == [
        f"powerup project state disabled: re-trust required: {state} changed since it was "
        f"trusted; run 'stateframe powerups trust --repo {repo}' after review",
        "powerups disabled for this run: fix or remove the invalid state file",
    ]
