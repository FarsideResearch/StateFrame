"""T8 fail-loud triage, batch 3: probes/ and governor / policy / guard modules.

Every broad ``except`` in the batch was classified (see the batch verdict table
in the test-gap notes).  No site was a CRASH-HIDER on a default path, so there
is no strict-xfail "silent today" test in this file.  What the triage did find
is a set of handlers whose *entire* safety argument is "the failure becomes a
typed outcome / a propagated error".  Those arguments are pinned here so a
future edit that turns one of them into a silent ``pass`` fails loudly.

SITE keys use the module, qualified scope, exception clause, and containing try AST shape; they contain no source lines.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

from _ast_site_keys import BroadExceptSite, find_broad_except_sites, rename_broad_except_site
import kernel_execution_policy as exec_policy
import kernel_guarded_effects as guarded_effects
import kernel_test_artifact_policy as artifact_policy

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
GOVERNOR_MODULES = ("kernel_guarded_effects", "kernel_execution_policy", "kernel_test_artifact_policy")
GOVERNOR_SCOPES = frozenset({
    ("kernel_guarded_effects", "GuardedEffectSequence._lookup"),
    ("kernel_guarded_effects", "GuardedEffectSequence.dispatch"),
    ("kernel_execution_policy", "_ResidentMemorySupervisor._record_error"),
    ("kernel_execution_policy", "supervised_process.cleanup"),
    ("kernel_test_artifact_policy", "normalize_public_test_bindings_for_mode"),
})
SITE_TAG = re.compile(
    r"SITE\[([a-z0-9_]+)\|([A-Za-z0-9_.<>]+)\|(Exception|BaseException|bare)\|([0-9a-f]{12})\]"
)
GOVERNOR_REVIEWED = frozenset({
    BroadExceptSite("kernel_guarded_effects", "GuardedEffectSequence._lookup", "Exception", "517e4dfa63b9"),
    BroadExceptSite("kernel_guarded_effects", "GuardedEffectSequence.dispatch", "Exception", "bb7509a57b8b"),
    BroadExceptSite("kernel_guarded_effects", "GuardedEffectSequence.dispatch", "BaseException", "6b8e894e6c21"),
    BroadExceptSite("kernel_execution_policy", "_ResidentMemorySupervisor._record_error", "BaseException", "fbf177513949"),
    BroadExceptSite("kernel_execution_policy", "supervised_process.cleanup", "BaseException", "e6d54340564a"),
    BroadExceptSite("kernel_execution_policy", "supervised_process.cleanup", "BaseException", "36934005d619"),
    BroadExceptSite("kernel_execution_policy", "supervised_process.cleanup", "BaseException", "e6d3865a570d"),
    BroadExceptSite("kernel_execution_policy", "supervised_process.cleanup", "BaseException", "742eacbbd39e"),
    BroadExceptSite("kernel_execution_policy", "supervised_process.cleanup", "BaseException", "4bb6ece3b0ef"),
    BroadExceptSite("kernel_execution_policy", "supervised_process.cleanup", "BaseException", "cff2d73448b0"),
    BroadExceptSite("kernel_test_artifact_policy", "normalize_public_test_bindings_for_mode", "Exception", "f4ef18b13b73"),
    BroadExceptSite("kernel_test_artifact_policy", "normalize_public_test_bindings_for_mode", "Exception", "fc0844881b3a"),
})


def _paths() -> dict[str, Path]:
    return {stem: ENGINE / f"{stem}.py" for stem in GOVERNOR_MODULES}


def _inventory_error(paths: dict[str, Path]) -> str | None:
    found = Counter(
        site
        for path in paths.values()
        for site in find_broad_except_sites(path)
        if (site.module, site.scope) in GOVERNOR_SCOPES
    )
    expected = Counter(GOVERNOR_REVIEWED)
    untriaged = sorted((found - expected).elements(), key=lambda site: site.label)
    stale = sorted((expected - found).elements(), key=lambda site: site.label)
    if not untriaged and not stale:
        return None
    details = []
    if untriaged:
        details.append("untriaged: " + "; ".join(site.label for site in untriaged))
    if stale:
        details.append("stale: " + "; ".join(site.label for site in stale))
    return " | ".join(details)


# --------------------------------------------------------------------------
# SITE[kernel_guarded_effects|GuardedEffectSequence._lookup|Exception|517e4dfa63b9]
# SITE[kernel_guarded_effects|GuardedEffectSequence.dispatch|Exception|bb7509a57b8b]
# SITE[kernel_guarded_effects|GuardedEffectSequence.dispatch|BaseException|6b8e894e6c21] preserves process-death re-raise
# --------------------------------------------------------------------------
class _Journal:
    """Minimal append-only journal with the surface GuardedEffectSequence reads."""

    run_id = "run-t8"

    def __init__(self) -> None:
        self.records: list[dict] = []

    def emit(self, kind: str, **fields: object) -> dict:
        event = {"kind": kind, "event_id": f"ev{len(self.records)}", **fields}
        self.records.append(event)
        return event


class _Destination:
    name = "t8-destination"

    def __init__(self, *, lookup_error: Exception | None = None,
                 submit_error: Exception | None = None) -> None:
        self._lookup_error = lookup_error
        self._submit_error = submit_error

    def result_for(self, key: tuple[str, str]) -> object:
        if self._lookup_error is not None:
            raise self._lookup_error
        return None

    def submit(self, key: tuple[str, str]) -> object:
        if self._submit_error is not None:
            raise self._submit_error
        return {"ok": True}


def test_guarded_effects_lookup_failure_is_typed_unknown() -> None:
    """A destination whose ``result_for`` raises yields UNKNOWN with the error class, no retry."""
    journal = _Journal()
    destination = _Destination(lookup_error=ConnectionError("boom"))
    sequence = guarded_effects.GuardedEffectSequence(
        journal, destination=destination, enabled_override=True)
    sequence.begin("effect-1", {"payload": 1})
    sequence.mark_dispatched("effect-1")

    outcome = sequence.reconcile("effect-1")

    assert outcome["status"] == "UNKNOWN"
    assert outcome["retry"] is False
    assert outcome["reason"] == "destination lookup failed: ConnectionError"


def test_guarded_effects_submit_failure_is_typed_unknown() -> None:
    """A destination whose ``submit`` raises yields UNKNOWN naming the class and forbids retry."""
    journal = _Journal()
    destination = _Destination(submit_error=TimeoutError("late"))
    sequence = guarded_effects.GuardedEffectSequence(
        journal, destination=destination, enabled_override=True)
    sequence.begin("effect-2", {"payload": 2})

    outcome = sequence.dispatch("effect-2")

    assert outcome["status"] == "UNKNOWN"
    assert outcome["retry"] is False
    assert outcome["reason"] == "destination submit failed: TimeoutError"
    # The durable dispatch marker stays in place for a later reconcile.
    assert outcome["has_dispatch"] is True
    assert outcome["has_receipt"] is False


# --------------------------------------------------------------------------
# SITE[kernel_execution_policy|_ResidentMemorySupervisor._record_error|BaseException|fbf177513949]
# The supervised_process.cleanup ladder is keyed by its guarded operation:
# SITE[kernel_execution_policy|supervised_process.cleanup|BaseException|e6d54340564a] poll at context exit
# SITE[kernel_execution_policy|supervised_process.cleanup|BaseException|36934005d619] terminate survivors
# SITE[kernel_execution_policy|supervised_process.cleanup|BaseException|e6d3865a570d] poll after termination
# SITE[kernel_execution_policy|supervised_process.cleanup|BaseException|742eacbbd39e] finish supervision
# SITE[kernel_execution_policy|supervised_process.cleanup|BaseException|4bb6ece3b0ef] close streams
# SITE[kernel_execution_policy|supervised_process.cleanup|BaseException|cff2d73448b0] bounded wait
# --------------------------------------------------------------------------
class _UnprintableError(RuntimeError):
    def __str__(self) -> str:
        raise ValueError("no message for you")


class _RecordingOwner:
    """Stand-in for the supervisor's error-recording fields."""

    def __init__(self) -> None:
        self._error_cause = None
        self.error = None
        self.stop_causes: list[str] = []

    def _record_stop_cause(self, reason: str) -> None:
        self.stop_causes.append(reason)


def test_kernel_execution_policy_unprintable_error_is_still_recorded() -> None:
    """``str(error)`` raising degrades the message, never the record of the failure."""
    owner = _RecordingOwner()
    error = _UnprintableError()

    exec_policy._ResidentMemorySupervisor._record_error(owner, error)

    assert owner._error_cause is error
    assert owner.error == (
        "resident-memory supervision failed (_UnprintableError: error message unavailable)")
    assert owner.stop_causes == [owner.error]


class _Stream:
    def __init__(self, error: BaseException | None = None) -> None:
        self._error = error
        self.closed = False

    def close(self) -> None:
        if self._error is not None:
            raise self._error
        self.closed = True


class _ExitedProcess:
    """An already-exited child with no memory supervisor attached."""

    pid = 4242

    def __init__(self, *, poll_error: BaseException | None = None,
                 stream_error: BaseException | None = None,
                 wait_error: BaseException | None = None) -> None:
        self.returncode = 0
        self._poll_error = poll_error
        self._wait_error = wait_error
        self.stdin = None
        self.stdout = _Stream(stream_error)
        self.stderr = _Stream()
        self.waited: list[float] = []

    def poll(self) -> int | None:
        if self._poll_error is not None:
            raise self._poll_error
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.waited.append(timeout if timeout is not None else -1.0)
        if self._wait_error is not None:
            raise self._wait_error
        return self.returncode


def test_supervised_process_stream_close_failure_propagates() -> None:
    """A pipe that cannot be closed is raised from the context, not swallowed."""
    process = _ExitedProcess(stream_error=OSError("pipe close failed"))

    with pytest.raises(OSError, match="pipe close failed"):
        with exec_policy.supervised_process(process, timeout=0.05):
            pass
    # The remaining cleanup steps still ran after the failing close.
    assert process.stderr.closed is True
    assert process.waited


def test_supervised_process_wait_failure_propagates() -> None:
    """A bounded wait that raises is the context's own failure on the clean path."""
    process = _ExitedProcess(wait_error=TimeoutError("wait stalled"))

    with pytest.raises(TimeoutError, match="wait stalled"):
        with exec_policy.supervised_process(process, timeout=0.05):
            pass


def test_supervised_process_cleanup_error_chains_to_original() -> None:
    """When the body raised, the first cleanup failure becomes the cause of the body error."""
    process = _ExitedProcess(stream_error=OSError("pipe close failed"))

    with pytest.raises(KeyError) as caught:
        with exec_policy.supervised_process(process, timeout=0.05):
            raise KeyError("body failed")

    assert isinstance(caught.value.__cause__, OSError)
    assert "pipe close failed" in str(caught.value.__cause__)


def test_t8_kernel_execution_policy_cleanup_keeps_only_the_first_error() -> None:
    """Later cleanup failures never mask the first one (the ladder is first-wins)."""
    process = _ExitedProcess(stream_error=OSError("first"), wait_error=TimeoutError("second"))

    with pytest.raises(OSError, match="first"):
        with exec_policy.supervised_process(process, timeout=0.05):
            pass


# --------------------------------------------------------------------------
# SITE[kernel_test_artifact_policy|normalize_public_test_bindings_for_mode|Exception|f4ef18b13b73]
# SITE[kernel_test_artifact_policy|normalize_public_test_bindings_for_mode|Exception|fc0844881b3a]
# --------------------------------------------------------------------------
def _bindings() -> tuple[dict[str, object], list[dict[str, object]]]:
    current = {"public_tests": {"tests/test_a.py": "def test_a(): pass\n"}}
    receipts = [{"public_tests": {"tests/test_b.py": "def test_b(): pass\n"}}]
    return current, receipts


def test_kernel_test_artifact_policy_optimized_failure_is_named(monkeypatch) -> None:
    """A crash in the optimized normalizer falls back to legacy AND says so."""
    def _broken(current, receipts):
        raise RuntimeError("optimized path exploded")

    monkeypatch.setattr(artifact_policy, "normalize_public_test_bindings", _broken)
    current, receipts = _bindings()
    legacy = artifact_policy._normalize_public_test_bindings_legacy(current, receipts)

    result = artifact_policy.normalize_public_test_bindings_for_mode(current, receipts, "shadow")

    assert result[0] == legacy[0]
    assert result[1] == legacy[1]
    assert result[2] == ()
    assert result[3] == "optimized_path_error"


def test_kernel_test_artifact_policy_mismatch_crash_is_a_mismatch(monkeypatch) -> None:
    """If the shadow comparison itself crashes, the output is reported as mismatched."""
    def _broken(optimized, legacy):
        raise ValueError("comparison exploded")

    monkeypatch.setattr(artifact_policy, "_normalization_mismatch_fields", _broken)
    current, receipts = _bindings()

    result = artifact_policy.normalize_public_test_bindings_for_mode(current, receipts, "shadow")

    assert result[2] == ("normalization_output",)
    assert result[3] is None


def test_t8_kernel_test_artifact_policy_mode_on_does_not_touch_legacy(monkeypatch) -> None:
    """Mode ``on`` never runs the legacy path, so a legacy crash cannot mask the optimized result."""
    def _broken(current, receipts):
        raise AssertionError("legacy must not run in mode on")

    monkeypatch.setattr(artifact_policy, "_normalize_public_test_bindings_legacy", _broken)
    current, receipts = _bindings()

    result = artifact_policy.normalize_public_test_bindings_for_mode(current, receipts, "on")

    assert result[2] == ()
    assert result[3] is None


def test_governor_site_keys_are_current() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    tags = {BroadExceptSite(*match) for match in SITE_TAG.findall(source)}
    assert tags == GOVERNOR_REVIEWED, (
        f"governor SITE tags changed: missing={GOVERNOR_REVIEWED - tags}, "
        f"extra={tags - GOVERNOR_REVIEWED}")
    error = _inventory_error(_paths())
    assert error is None, error


def _add_unreviewed_lookup_handler(source: str) -> str:
    tree = ast.parse(source)
    target: ast.FunctionDef | None = None

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        nonlocal target
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = (*scope, node.name)
        if isinstance(node, ast.FunctionDef) and scope == ("GuardedEffectSequence", "_lookup"):
            target = node
            return
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    if target is None or target.end_lineno is None or not target.body:
        raise AssertionError("GuardedEffectSequence._lookup: could not locate mutation target")
    lines = source.splitlines(keepends=True)
    indentation = " " * target.body[0].col_offset
    addition = (
        f"{indentation}try:\n"
        f"{indentation}    raise RuntimeError('unreviewed site')\n"
        f"{indentation}except Exception:\n"
        f"{indentation}    return {{}}\n"
    )
    lines.insert(target.end_lineno, addition)
    return "".join(lines)


def test_governor_site_locator_survives_shift_and_rejects_removed_or_new_sites(tmp_path: Path) -> None:
    original_paths = _paths()
    shifted_paths: dict[str, Path] = {}
    for stem, original_path in original_paths.items():
        copy_path = tmp_path / "shifted" / original_path.name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        copy_path.write_text("\n" * 5 + "# shifted copy\n" + original_path.read_text(encoding="utf-8"),
                             encoding="utf-8")
        shifted_paths[stem] = copy_path
    assert _inventory_error(shifted_paths) is None

    for site in sorted(GOVERNOR_REVIEWED, key=lambda item: item.label):
        original_path = original_paths[site.module]
        copy_path = tmp_path / "removed" / site.module / original_path.name
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        source = original_path.read_text(encoding="utf-8")
        copy_path.write_text(rename_broad_except_site(original_path, source, site), encoding="utf-8")
        changed_paths = dict(original_paths)
        changed_paths[site.module] = copy_path
        error = _inventory_error(changed_paths)
        assert error is not None and f"stale: {site.label}" in error, error

    module_path = original_paths["kernel_guarded_effects"]
    copy_path = tmp_path / "added" / module_path.name
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    source = _add_unreviewed_lookup_handler(module_path.read_text(encoding="utf-8"))
    copy_path.write_text(source, encoding="utf-8")
    changed_paths = dict(original_paths)
    changed_paths["kernel_guarded_effects"] = copy_path
    error = _inventory_error(changed_paths)
    assert error is not None and (
        "untriaged: kernel_guarded_effects.GuardedEffectSequence._lookup" in error), error
