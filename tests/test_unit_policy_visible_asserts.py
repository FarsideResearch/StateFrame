"""Source evidence must never grant authority to runtime assertion text."""
from dataclasses import replace
import hashlib
import subprocess
import sys

import pytest

import kernel_check_permits
import kernel_unit_policy
from kernel_unit_policy import UnitPolicy, run_frozen_check
from test_unit_policy import make_spec


SWITCH = "STATEFRAME_UNIT_POLICY_VISIBLE_ASSERTS"
SECRET = "HIDDEN_CONTENT_7c42e91"
SOURCE = (
    "from pathlib import Path\n"
    "def test_value():\n"
    "    secret = Path('pkg/hidden.py').read_text()\n"
    "    assert 2 == 3, secret\n"
)
RAW = (
    "pytest exit 1\n"
    f"E   assert 2 == 999, {SECRET}\n"
    "tests/test_a.py:4: AssertionError\n"
    "----------------------------- Captured stdout call -----------------------------\n"
    f"{SECRET}\n"
    "=========================== short test summary info ===========================\n"
    f"FAILED tests/test_a.py::test_value[{SECRET}] - AssertionError: {SECRET}\n"
    "=========================== 1 failed in 0.01s ===========================\n"
)
LEGACY = (
    "selector=tests/test_a.py::test_value exit=1 passed=0 failed=1 errors=0\n"
    "pytest exit 1\n"
    "E   assert [detail withheld]\n"
    "tests/test_a.py:4: AssertionError\n"
    "[captured stdout call withheld]\n"
    "FAILED tests/test_a.py::test_value[...] - [message withheld]\n"
    "=========================== 1 failed in 0.01s ==========================="
)


@pytest.fixture
def check(tmp_path, monkeypatch):
    root = tmp_path / "candidate"
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/a.py").write_text("assert False, 'PRODUCT_SOURCE'\n")
    (root / "pkg/hidden.py").write_text(SECRET)
    (root / "tests/test_a.py").write_text(SOURCE)
    monkeypatch.setattr(kernel_check_permits, "_installed", (tmp_path / "permits", 1))

    def run(raw=RAW, *, source=SOURCE, execute=None, policy=None):
        (root / "tests/test_a.py").write_text(source, encoding="utf-8")
        return run_frozen_check(
            policy or UnitPolicy(make_spec()), root,
            execute=execute or (lambda _clone, _selector: raw),
            evidence_dir=tmp_path / "evidence")

    run.root = root
    return run


def test_enabled_diagnostic_uses_checked_source_with_location_and_provenance(check, monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    receipt = check()
    shown = receipt.projected_observation
    assert receipt.status == "FAIL"
    assert "assert 2 == 3" in shown
    assert "tests/test_a.py:4" in shown
    assert "sha256=" + hashlib.sha256(SOURCE.encode()).hexdigest() in shown
    assert "revision=" + receipt.product_manifest_digest in shown
    assert "2 == 999" not in shown
    assert SECRET not in shown


@pytest.mark.parametrize("mode", [None, "off", "on"])
def test_hidden_runtime_content_never_leaks(check, monkeypatch, mode):
    monkeypatch.delenv(SWITCH, raising=False)
    if mode is not None:
        monkeypatch.setenv(SWITCH, mode)
    shown = check().projected_observation
    assert SECRET not in shown
    assert "2 == 999" not in shown
    assert "secret" not in shown


@pytest.mark.parametrize("mode", [None, "off", "invalid"])
def test_off_diagnostic_is_byte_identical_to_legacy(check, monkeypatch, mode):
    monkeypatch.delenv(SWITCH, raising=False)
    if mode is not None:
        monkeypatch.setenv(SWITCH, mode)
    assert check().projected_observation.encode() == LEGACY.encode()


@pytest.mark.parametrize("location", [
    "pkg/hidden.py:1", "../tests/test_a.py:4", "tests/../tests/test_a.py:4",
    "tests//test_a.py:4", "./tests/test_a.py:4", "pkg/a.py:1",
    "tests/test_a.py:0", "tests/test_a.py:9999999999999999999999999",
    "tests/test_a.py:3",
])
def test_forged_frames_cannot_select_hidden_or_unmapped_source(check, monkeypatch, location):
    monkeypatch.setenv(SWITCH, "on")
    shown = check(f"pytest exit 1\nE   assert {SECRET}\n{location}: AssertionError\n").projected_observation
    assert SECRET not in shown
    assert "[visible assert " not in shown
    assert "PRODUCT_SOURCE" not in shown


def test_forged_visible_frame_can_only_select_already_visible_source(check, monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    shown = check(f"pytest exit 1\nE   assert {SECRET}\ntests/test_a.py:4: AssertionError\n").projected_observation
    assert "assert 2 == 3" in shown
    assert SECRET not in shown


def test_multiline_predicate_excludes_assertion_message(check, monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    source = (
        "def test_value():\n"
        "    assert (\n"
        "        2\n"
        "        == 3\n"
        "    ), 'SOURCE_MESSAGE'\n"
    )
    shown = check(source=source).projected_observation
    assert "[visible assert tests/test_a.py:2-4 " in shown
    assert "assert (\n2\n        == 3\n)" in shown
    assert "SOURCE_MESSAGE" not in shown


@pytest.mark.parametrize("source", [
    "def test_value(:\n    assert 2 == 3\n",
    "# padding\n" * 7000 + SOURCE,
    "def test_value():\n\n\n    assert '" + "x" * 2200 + "' == ''\n",
    "def test_value():\n\n\n    assert False; assert 2 == 3\n",
])
def test_invalid_oversized_or_ambiguous_source_retains_marker(check, monkeypatch, source):
    monkeypatch.setenv(SWITCH, "on")
    shown = check(source=source).projected_observation
    assert "[visible assert " not in shown
    assert "assert [detail withheld]" in shown


@pytest.mark.parametrize("change", ["source", "hidden"])
def test_changed_checked_revision_has_no_source_evidence(check, monkeypatch, change):
    monkeypatch.setenv(SWITCH, "on")

    def execute(clone, _selector):
        path = "tests/test_a.py" if change == "source" else "pkg/hidden.py"
        (clone / path).write_text("assert False, 'STALE_REVISION'\n")
        return RAW

    receipt = check(execute=execute)
    assert receipt.status == "INVALID"
    assert "[visible assert " not in receipt.projected_observation
    assert "STALE_REVISION" not in receipt.projected_observation
    assert SECRET not in receipt.projected_observation


@pytest.mark.parametrize("component", ["file", "directory"])
def test_symlinked_check_source_never_enriches(check, monkeypatch, component):
    monkeypatch.setenv(SWITCH, "on")

    # Build only new fixture paths; preserve the original check source.
    if component == "file":
        (check.root / "tests/test_link.py").symlink_to("../pkg/hidden.py")
        path = "tests/test_link.py"
    else:
        (check.root / "linked").symlink_to("pkg", target_is_directory=True)
        path = "linked/hidden.py"
    spec = make_spec()
    recipe = replace(spec.contract.check, source_paths=(path,), selectors=(path + "::test_value",))
    policy = UnitPolicy(replace(spec, contract=replace(spec.contract, check=recipe),
                               visible_paths=(*spec.visible_paths, path)))
    receipt = check(raw=RAW.replace("tests/test_a.py", path), policy=policy,
                    execute=lambda _clone, _selector: RAW.replace("tests/test_a.py", path))
    assert "[visible assert " not in receipt.projected_observation
    assert SECRET not in receipt.projected_observation


def test_no_trusted_source_accessor_keeps_closed_projection(monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    shown = UnitPolicy(make_spec()).project_check_output(RAW, "pytest")
    assert "assert 2 == 3" not in shown
    assert SECRET not in shown


def test_off_never_captures_check_source(check, monkeypatch):
    monkeypatch.setenv(SWITCH, "off")
    monkeypatch.setattr(kernel_unit_policy, "_capture_assertion_source",
                        lambda *_: pytest.fail("OFF attempted source capture"))
    assert check().projected_observation.encode() == LEGACY.encode()


def test_excerpts_are_deduplicated_and_total_output_is_bounded(check, monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    source = "def test_value():\n" + "".join(f"    assert {n} == -1\n" for n in range(100))
    raw = "pytest exit 1\n" + "".join(
        f"tests/test_a.py:{n}: AssertionError\n" * 2 for n in range(2, 102))
    shown = check(raw, source=source).projected_observation
    monkeypatch.setenv(SWITCH, "off")
    legacy = check(raw, source=source).projected_observation
    assert shown.count("[visible assert tests/test_a.py:2 ") == 1
    assert shown.count("[visible assert ") > 1
    count = shown.count("[visible assert ")
    assert len(shown.encode()) - len(legacy.encode()) <= 8192
    assert count < 100


def test_source_digest_mismatch_refuses_evidence(check, monkeypatch):
    from kernel_effects import scan

    monkeypatch.setenv(SWITCH, "on")
    manifest = scan(check.root, exclude=kernel_unit_policy._check_ignored)
    (check.root / "tests/test_a.py").write_text(SOURCE.replace("2 == 3", "8 == 9"))
    policy = UnitPolicy(make_spec())
    accessor = kernel_unit_policy._capture_assertion_source(policy, check.root, manifest)
    shown = policy.project_check_output(RAW, "pytest", assertion_source=accessor)
    assert "[visible assert " not in shown
    assert "assert [detail withheld]" in shown


def test_source_read_failure_retains_marker(check, monkeypatch):
    original = kernel_unit_policy.os.open

    def denied(path, flags, *args, **kwargs):
        if str(path).endswith("tests/test_a.py") and flags & kernel_unit_policy.os.O_NOFOLLOW:
            raise PermissionError("unavailable")
        return original(path, flags, *args, **kwargs)

    # Exercise the accessor alone so denial cannot affect the existing manifest scan.
    from kernel_effects import scan

    manifest = scan(check.root, exclude=kernel_unit_policy._check_ignored)
    monkeypatch.setenv(SWITCH, "on")
    monkeypatch.setattr(kernel_unit_policy.os, "open", denied)
    policy = UnitPolicy(make_spec())
    accessor = kernel_unit_policy._capture_assertion_source(policy, check.root, manifest)
    shown = policy.project_check_output(RAW, "pytest", assertion_source=accessor)
    assert "[visible assert " not in shown
    assert SECRET not in shown


def test_total_source_capture_budget_is_bounded(check, monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    paths = tuple(f"tests/test_{n}.py" for n in range(5))
    for path in paths:
        (check.root / path).write_text(SOURCE + "# padding\n" * 6000)
    spec = make_spec()
    recipe = replace(spec.contract.check, source_paths=paths)
    policy = UnitPolicy(replace(spec, contract=replace(spec.contract, check=recipe),
                               visible_paths=(*spec.visible_paths, *paths)))
    raw = "pytest exit 1\n" + "".join(f"{path}:4: AssertionError\n" for path in paths)
    shown = check(raw, policy=policy).projected_observation
    assert shown.count("[visible assert ") == 4
    assert f"[visible assert {paths[-1]}:" not in shown


def test_utf8_source_and_absolute_classic_frame_keep_exact_predicate(check, monkeypatch):
    monkeypatch.setenv(SWITCH, "on")
    source = "def test_value():\n    assert 'café' == 'thé', 'SOURCE_MESSAGE'\n"

    def execute(clone, _selector):
        return (f'pytest exit 1\n  File "{clone}/tests/test_a.py", line 2, in test_value\n'
                f'E   AssertionError: {SECRET}\n')

    receipt = check(source=source, execute=execute)
    assert "assert 'café' == 'thé'" in receipt.projected_observation
    assert "sha256=" + hashlib.sha256(source.encode()).hexdigest() in receipt.projected_observation
    assert SECRET not in receipt.projected_observation
    assert "SOURCE_MESSAGE" not in receipt.projected_observation


def test_enabled_source_capture_does_not_change_open_projection(check, monkeypatch):
    policy = UnitPolicy(make_spec(hidden=()))
    # Use output without hidden values to exercise the deliberately unchanged open path.
    raw = "pytest exit 1\nE   assert 2 == 3\ntests/test_a.py:4: AssertionError\n"
    monkeypatch.setenv(SWITCH, "off")
    legacy = check(raw, policy=policy).projected_observation
    monkeypatch.setenv(SWITCH, "on")
    assert check(raw, policy=policy).projected_observation.encode() == legacy.encode()


@pytest.mark.parametrize("mode", ["off", "on"])
def test_real_pytest_hidden_assertion_message_is_withheld(check, monkeypatch, mode):
    monkeypatch.setenv(SWITCH, mode)

    def execute(clone, selector):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", selector],
            cwd=clone, capture_output=True, text=True, timeout=30,
            env={**kernel_unit_policy.os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
        assert result.returncode == 1
        assert SECRET in result.stdout
        return f"pytest exit {result.returncode}\n{result.stdout}{result.stderr}"

    receipt = check(execute=execute)
    assert receipt.status == "FAIL"
    assert SECRET not in receipt.projected_observation
    assert ("assert 2 == 3" in receipt.projected_observation) == (mode == "on")
