"""T8 fail-loud triage, batch 3: probes/ and governor / policy / guard modules.

Every broad ``except`` in the batch was classified (see the batch verdict table
in the test-gap notes).  No site was a CRASH-HIDER on a default path, so there
is no strict-xfail "silent today" test in this file.  What the triage did find
is a set of handlers whose *entire* safety argument is "the failure becomes a
typed outcome / a propagated error".  Those arguments are pinned here so a
future edit that turns one of them into a silent ``pass`` fails loudly.

Site ids follow the census: ``T8-<MODULE>-<line>`` at the batch base.
"""
from __future__ import annotations

import pytest

import kernel_execution_policy as exec_policy
import kernel_guarded_effects as guarded_effects
import kernel_test_artifact_policy as artifact_policy


# --------------------------------------------------------------------------
# kernel_guarded_effects: T8-KERNEL_GUARDED_EFFECTS-329 / -382
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


def test_t8_kernel_guarded_effects_329_lookup_failure_is_typed_unknown() -> None:
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


def test_t8_kernel_guarded_effects_382_submit_failure_is_typed_unknown() -> None:
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
# kernel_execution_policy: T8-KERNEL_EXECUTION_POLICY-631 and the
# supervised_process cleanup ladder (-1139 / -1159 / -1168 / -1173 / -1182 / -1189)
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


def test_t8_kernel_execution_policy_631_unprintable_error_is_still_recorded() -> None:
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


def test_t8_kernel_execution_policy_1182_stream_close_failure_propagates() -> None:
    """A pipe that cannot be closed is raised from the context, not swallowed."""
    process = _ExitedProcess(stream_error=OSError("pipe close failed"))

    with pytest.raises(OSError, match="pipe close failed"):
        with exec_policy.supervised_process(process, timeout=0.05):
            pass
    # The remaining cleanup steps still ran after the failing close.
    assert process.stderr.closed is True
    assert process.waited


def test_t8_kernel_execution_policy_1189_wait_failure_propagates() -> None:
    """A bounded wait that raises is the context's own failure on the clean path."""
    process = _ExitedProcess(wait_error=TimeoutError("wait stalled"))

    with pytest.raises(TimeoutError, match="wait stalled"):
        with exec_policy.supervised_process(process, timeout=0.05):
            pass


def test_t8_kernel_execution_policy_1139_cleanup_error_chains_to_original() -> None:
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
# kernel_test_artifact_policy: T8-KERNEL_TEST_ARTIFACT_POLICY-701 / -711
# --------------------------------------------------------------------------
def _bindings() -> tuple[dict[str, object], list[dict[str, object]]]:
    current = {"public_tests": {"tests/test_a.py": "def test_a(): pass\n"}}
    receipts = [{"public_tests": {"tests/test_b.py": "def test_b(): pass\n"}}]
    return current, receipts


def test_t8_kernel_test_artifact_policy_701_optimized_failure_is_named(monkeypatch) -> None:
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


def test_t8_kernel_test_artifact_policy_711_mismatch_crash_is_a_mismatch(monkeypatch) -> None:
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
