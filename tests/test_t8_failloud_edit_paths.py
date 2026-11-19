"""Fail-loud batch 4: broad except sites in the edit, source, evidence and tool kernel modules.

Companion to the original T8 pilot census. Every census site in kernel_*edit*.py, kernel_*source*.py,
kernel_*evidence*.py and kernel_tool_*.py outside the pilot sample was triaged. No site in this batch hides
a failure on a default path: each one either re-raises after cleanup or converts the failure into a typed
outcome (degraded flag + error class, typed reason code, receipt status, or a recorded advisory event).
The tests below force the guarded failure and pin that visible outcome, so a later edit that makes one of
these sites silent fails here. The inventory covers every broad handler in the modules named by the original pins.
"""
from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from _except_sites import (
    UNREVIEWED_HANDLER,
    ExceptSite,
    _broad_except_sites,
    assert_site_tags,
    assert_triaged,
    edited_module_paths,
    shifted_module_paths,
)

ENGINE = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"

_MODULE_PATHS = (
    ENGINE / "kernel_evidence_contracts.py",
    ENGINE / "kernel_evidence_relevance_shadow.py",
    ENGINE / "kernel_green_evidence_owner.py",
    ENGINE / "kernel_revision_bound_edit.py",
    ENGINE / "kernel_tool_reply.py",
    ENGINE / "kernel_tool_schema_diet.py",
)

_TRIAGED: set[ExceptSite] = {
    ("kernel_evidence_contracts", "outline_for_snapshot", "except Exception as exc:", 1),
    ("kernel_evidence_relevance_shadow", "_try_reserve_shadow", "except Exception as error:", 1),
    ("kernel_evidence_relevance_shadow", "score_evidence_relevance", "except Exception as error:", 2),
    ("kernel_green_evidence_owner", "_active_emit", "except Exception as error:", 1),
    ("kernel_green_evidence_owner", "_unchanged_not_applied_write", "except Exception:", 1),
    ("kernel_revision_bound_edit", "_resolve_runner_value.resolve_in_thread", "except BaseException as error:", 1),
    ("kernel_revision_bound_edit", "run_behavior_check", "except Exception as error:", 1),
    ("kernel_revision_bound_edit", "_outline", "except Exception as error:", 1),
    ("kernel_tool_reply", "copy_tool_reply_message", "except Exception:", 1),
    ("kernel_tool_reply", "record_and_strip_receipts", "except Exception:", 1),
    ("kernel_tool_schema_diet", "emit_sidecar_stripped", "except Exception:", 1),
    ("kernel_tool_schema_diet", "note_applied", "except Exception:", 1),
    # The rest of the broad handlers in these modules are advisory. Shadow journal and settlement
    # failures keep their error class in the evaluation; duplicate-reply telemetry stays best-effort.
    ("kernel_evidence_relevance_shadow", "score_evidence_relevance", "except Exception as error:", 1),
    ("kernel_evidence_relevance_shadow", "score_evidence_relevance", "except Exception as error:", 3),
    ("kernel_evidence_relevance_shadow", "score_evidence_relevance", "except Exception as error:", 4),
    ("kernel_evidence_relevance_shadow", "score_evidence_relevance", "except Exception as error:", 5),
    ("kernel_tool_reply", "compose_tool_replies", "except Exception:", 1),
}


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
    """Site T8-[kernel_revision_bound_edit | _resolve_runner_value.resolve_in_thread | except BaseException as error: | 1].
    CLEANUP: the worker thread parks the adapter error and the caller re-raises it."""
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
    """Site T8-[kernel_revision_bound_edit | run_behavior_check | except Exception as error: | 1].
    an adapter crash is preserved as execution_failure on an ENVIRONMENT receipt."""
    import kernel_revision_bound_edit as rbe

    def failing_adapter(_entrypoint: str) -> dict[str, object]:
        raise OSError("forced: behavior adapter unavailable")

    candidate = rbe.CandidateGeneration("candidate-1", "parent-revision", "candidate-revision", {})
    receipt = rbe.create_check_receipt(
        check_id="check-1", candidate=candidate, parent_revision=candidate.base_revision,
        observation_receipt="observation-1",
    )
    failed = rbe.run_behavior_check(
        receipt, candidate=candidate, entrypoint="pkg.mod:check", source_reachable=True, disposition="repair",
        parent_runner=lambda _entrypoint: {"reached": True}, candidate_runner=failing_adapter,
    )
    assert failed.execution_failure == "OSError"
    assert failed.outcome is rbe.ReceiptStatus.ENVIRONMENT


def test_outline_parser_failure_is_typed_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_revision_bound_edit | _outline | except Exception as error: | 1].
    an unavailable or crashing parser yields a typed degraded outline, not a guess."""
    import kernel_revision_bound_edit as rbe

    _install_failing_outline(monkeypatch, ValueError("forced: parser crashed"))
    outline = rbe._outline("def f():\n    pass\n", "pkg/mod.py")
    assert outline["parser"] == "unavailable"
    assert outline["degraded"] is True
    assert outline["error"] == "ValueError"


# --- kernel_evidence_contracts -------------------------------------------------------------------------


def test_snapshot_outline_parser_failure_is_typed_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Site T8-[kernel_evidence_contracts | outline_for_snapshot | except Exception as exc: | 1].
    navigation outline failure is reported as degraded with the error class."""
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
    """Site T8-[kernel_evidence_relevance_shadow | _try_reserve_shadow | except Exception as error: | 1].
    an account error becomes a typed SHADOW_RESOURCE_ACCOUNT_<Error> reason code."""
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
    """Site T8-[kernel_green_evidence_owner | _active_emit | except Exception as error: | 1].
    a journal.emit crash is recorded on the runtime as a typed advisory failure."""
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
    """Site T8-[kernel_green_evidence_owner | _unchanged_not_applied_write | except Exception: | 1].
    a missing/invalid closure makes the not_applied write NOT count as unchanged (fail-closed)."""
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
    """Sites T8-[kernel_tool_schema_diet | note_applied | except Exception: | 1]
    and T8-[kernel_tool_schema_diet | emit_sidecar_stripped | except Exception: | 1].
    a telemetry emit crash returns False and leaves the journal unmarked."""
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
    assert_triaged(_broad_except_sites(_MODULE_PATHS), _TRIAGED)


def test_edit_tags_point_at_broad_except_sites() -> None:
    assert_site_tags(Path(__file__), _broad_except_sites(_MODULE_PATHS))


def test_edit_inventory_survives_line_shift(tmp_path: Path) -> None:
    found = _broad_except_sites(shifted_module_paths(_MODULE_PATHS, tmp_path))
    assert_triaged(found, _TRIAGED)
    assert_site_tags(Path(__file__), found)


def test_edit_inventory_rejects_missing_site(tmp_path: Path) -> None:
    paths = edited_module_paths(
        _MODULE_PATHS, tmp_path, "kernel_revision_bound_edit",
        "def run_behavior_check(", "def _renamed_run_behavior_check(",
    )
    stale = ("kernel_revision_bound_edit", "run_behavior_check", "except Exception as error:", 1)
    found = _broad_except_sites(paths)
    with pytest.raises(AssertionError) as failure:
        assert_triaged(found, _TRIAGED)
    assert f"stale: [{stale!r}]" in str(failure.value)


def test_edit_inventory_rejects_untriaged_site(tmp_path: Path) -> None:
    paths = edited_module_paths(_MODULE_PATHS, tmp_path, "kernel_revision_bound_edit", None, UNREVIEWED_HANDLER)
    untriaged = ("kernel_revision_bound_edit", "_t8_unreviewed", "except Exception:", 1)
    with pytest.raises(AssertionError) as failure:
        assert_triaged(_broad_except_sites(paths), _TRIAGED)
    assert f"untriaged: [{untriaged!r}]; stale: []" in str(failure.value)
