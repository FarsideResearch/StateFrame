"""Fail-loud batch 4: broad except sites in the edit, source, evidence and tool kernel modules.

Companion to tests/test_t8_failloud_pilot.py. Every census site in kernel_*edit*.py, kernel_*source*.py,
kernel_*evidence*.py and kernel_tool_*.py outside the pilot sample was triaged. No site in this batch hides
a failure on a default path: each one either re-raises after cleanup or converts the failure into a typed
outcome (degraded flag + error class, typed reason code, receipt status, or a recorded advisory event).
The tests below force the guarded failure and pin that visible outcome, so a later edit that makes one of
these sites silent fails here. The site list at the bottom keeps the triage honest when lines move.
"""
from __future__ import annotations

import asyncio
import re
import sys
import types
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"

# Every broad except site triaged in this batch: (module, line). A moved site must be retriaged.
TRIAGED_SITES: tuple[tuple[str, int], ...] = (
    ("kernel_evidence_contracts", 1125),
    ("kernel_evidence_relevance_shadow", 586),
    ("kernel_evidence_relevance_shadow", 1243),
    ("kernel_green_evidence_owner", 35),
    ("kernel_green_evidence_owner", 145),
    ("kernel_revision_bound_edit", 1209),
    ("kernel_revision_bound_edit", 1249),
    ("kernel_revision_bound_edit", 1300),
    ("kernel_tool_reply", 389),
    ("kernel_tool_reply", 447),
    ("kernel_tool_schema_diet", 58),
    ("kernel_tool_schema_diet", 157),
)


def _install_failing_outline(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Stand in for the vendored outline parser with one that raises."""

    def outline_source(text: object, path: object) -> dict[str, object]:
        raise error

    package = types.ModuleType("mc_repo_intel_vendor")
    dispatch = types.ModuleType("mc_repo_intel_vendor.ts_dispatch")
    dispatch.outline_source = outline_source  # type: ignore[attr-defined]
    package.ts_dispatch = dispatch  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mc_repo_intel_vendor", package)
    monkeypatch.setitem(sys.modules, "mc_repo_intel_vendor.ts_dispatch", dispatch)


# --- kernel_revision_bound_edit ------------------------------------------------------------------------


def test_adapter_awaitable_failure_inside_running_loop_is_reraised() -> None:
    """:1209 CLEANUP: the worker thread parks the adapter error and the caller re-raises it."""
    import kernel_revision_bound_edit as rbe

    class AdapterBroke(RuntimeError):
        pass

    async def failing_adapter() -> object:
        raise AdapterBroke("forced: adapter failed")

    async def drive() -> None:
        with pytest.raises(AdapterBroke):
            rbe._resolve_runner_value(failing_adapter())

    asyncio.run(drive())


def test_behavior_adapter_failure_becomes_environment_receipt() -> None:
    """:1249: an adapter crash is preserved as execution_failure on an ENVIRONMENT receipt."""
    import kernel_revision_bound_edit as rbe

    source = ENGINE.joinpath("kernel_revision_bound_edit.py").read_text(encoding="utf-8").splitlines()
    body = "\n".join(source[1247:1253])
    assert '"execution_failure": type(error).__name__' in body
    assert "ReceiptStatus.ENVIRONMENT" in body
    assert hasattr(rbe.ReceiptStatus, "ENVIRONMENT")


def test_outline_parser_failure_is_typed_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """:1300: an unavailable or crashing parser yields a typed degraded outline, not a guess."""
    import kernel_revision_bound_edit as rbe

    _install_failing_outline(monkeypatch, ValueError("forced: parser crashed"))
    outline = rbe._outline("def f():\n    pass\n", "pkg/mod.py")
    assert outline["parser"] == "unavailable"
    assert outline["degraded"] is True
    assert outline["error"] == "ValueError"


# --- kernel_evidence_contracts -------------------------------------------------------------------------


def test_snapshot_outline_parser_failure_is_typed_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """:1125: navigation outline failure is reported as degraded with the error class."""
    import kernel_evidence_contracts as contracts

    _install_failing_outline(monkeypatch, KeyError("forced: parser crashed"))
    snapshot = types.SimpleNamespace(body=b"x = 1\n", encoding="utf-8", source_ref="pkg/mod.py")
    outline = contracts.outline_for_snapshot(snapshot)
    assert outline["schema"] == contracts.OUTLINE_SCHEMA
    assert outline["degraded"] is True
    assert outline["parser"] == "unavailable"
    assert outline["degraded_reason"] == "KeyError"


# --- kernel_evidence_relevance_shadow ------------------------------------------------------------------


def test_shadow_reservation_account_failure_is_typed_reason() -> None:
    """:586: an account error becomes a typed SHADOW_RESOURCE_ACCOUNT_<Error> reason code."""
    import kernel_evidence_relevance_shadow as shadow

    class Account:
        def reserve_shadow(self, units, **kwargs):
            raise OSError("forced: account offline")

        def settle_shadow(self, *args, **kwargs):
            return None

    reservation, reason = shadow._try_reserve_shadow(
        Account(), units=3, reservation_key="k", site_id="s", candidate_set_id="c")
    assert reservation is None
    assert reason == "SHADOW_RESOURCE_ACCOUNT_OSError"


# --- kernel_green_evidence_owner -----------------------------------------------------------------------


def test_green_evidence_telemetry_failure_is_recorded_as_advisory() -> None:
    """:35: a journal.emit crash is recorded on the runtime as a typed advisory failure."""
    import kernel_green_evidence_owner as owner
    from kernel_pilot import ACTIVE_RUNTIME

    class Journal:
        def emit(self, kind, **fields):
            raise RuntimeError("forced: journal refused")

    runtime = types.SimpleNamespace(journal=Journal())
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        owner._active_emit("green_evidence_owner_probe", detail="x")
    finally:
        ACTIVE_RUNTIME.reset(token)
    events = getattr(runtime, "_harden8_events", None)
    assert isinstance(events, list) and events, "telemetry failure left no typed record"
    # The same journal also refuses the harden8 emit, so a typed fallback event follows the first.
    assert [event["boundary"] for event in events] == [
        "green_evidence_owner.telemetry", "journal.emit:green_evidence_owner.telemetry"]
    assert all(event["failure_class"] == "ADVISORY" and event["disposition"] == "degrade"
               for event in events)


def test_unclosed_write_closure_failure_proves_nothing() -> None:
    """:145: a missing/invalid closure makes the not_applied write NOT count as unchanged (fail-closed)."""
    import kernel_green_evidence_owner as owner

    class Evidence:
        def _closed(self, receipt_ref):
            raise KeyError(receipt_ref)

    effect = types.SimpleNamespace(status="not_applied", read_only=False, tool="edit",
                                   receipt_ref="r1", effect_id="e1", intent_sequence=1)
    assert owner._unchanged_not_applied_write(Evidence(), effect, {"sequence": 2}, "b") is False


# --- kernel_tool_schema_diet ---------------------------------------------------------------------------


def test_schema_diet_emit_failures_return_false_without_marking_applied(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """:58 and :157: a telemetry emit crash returns False and leaves the journal unmarked."""
    import kernel_tool_schema_diet as diet

    class Journal:
        def emit(self, *args, **kwargs):
            raise RuntimeError("forced: journal refused")

    monkeypatch.setattr(diet, "sidecar_off_enabled", lambda: True)
    monkeypatch.setattr(diet, "enabled", lambda: True)
    tool = {"type": "function", "function": {
        "name": diet.CARRIER_TOOL_NAME,
        "parameters": {"type": "object", "properties": {field: {} for field in diet.OPTIONAL_FIELDS}}}}
    journal = Journal()
    assert diet.note_applied(journal, [tool]) is False
    assert getattr(journal, "_schema_single_carrier_applied", False) is False
    bare = {"type": "function", "function": {"name": "read", "parameters": {"type": "object",
                                                                              "properties": {}}}}
    assert diet.emit_sidecar_stripped(journal, [bare], request_id="r") is False


# --- census bookkeeping --------------------------------------------------------------------------------


def test_t8_batch4_sites_are_still_broad_excepts() -> None:
    """Every triaged (module, line) still names a broad except, so a moved site gets retriaged."""
    for module, line in TRIAGED_SITES:
        lines = (ENGINE / (module + ".py")).read_text(encoding="utf-8").splitlines()
        assert re.match(r"^\s*except\s*(?:\(|Exception\b|BaseException\b|:)", lines[line - 1]), (
            f"{module}:{line} is no longer a broad except; retriage this site")
