import pytest
from types import SimpleNamespace

from kernel_bolt import KernelBolt
from kernel_continuation_evidence import HostEvidence, binding_versions
from kernel_continuation_runtime import _guarded_host_outcome
from kernel_continuation_state import ContinuationState, Effect
from kernel_measurement import digest
from kernel_step_ledger import StepLedger


def _shell_outcome(*, receipt_kind="shell_effect", result="APPLIED", effects=None,
                   receipt_fields=None, record_receipt_ids=None, tool="shell",
                   exit_code=0):
    run_id = "run-1"
    binding = {
        "product_files": {"pkg/__init__.py": "sha256:file"},
        "harness_files": {},
    }
    effect = Effect(
        "effect-1", "root", "action-hash", binding_versions(binding), False,
        tool=tool, arguments_hash="arguments-hash", dispatch_ref="dispatch-1",
        step=2, request_id="request-1", intent_sequence=2,
    )
    receipt = {
        "receipt_id": "receipt-1",
        "run_id": run_id,
        "request_id": "request-1",
        "turn": 2,
        "kind": receipt_kind,
        "result_class": result,
        "exit_code": exit_code,
        "continuation_id": "root",
        "continuation_effect_id": "effect-1",
        "continuation_dispatch_ref": "dispatch-1",
    }
    if effects is not None:
        receipt["effects"] = effects
    receipt.update(receipt_fields or {})
    receipt_event = {
        "event_id": "receipt-event-1",
        "sequence": 4,
        "run_id": run_id,
        "kind": "action_receipt",
        "receipt": receipt,
    }
    record = {
        "step": 2,
        "tool": tool,
        "arguments_hash": "arguments-hash",
        "result": result,
        "receipt_ids": ["receipt-1"] if record_receipt_ids is None else record_receipt_ids,
        "before_binding": binding,
        "after_binding": binding,
        "changes": [],
        "continuation_effect_id": "effect-1",
        "continuation_dispatch_ref": "dispatch-1",
        "continuation_id": "root",
    }
    record["step_record_id"] = digest(record)
    closed = {
        "event_id": "closed-1",
        "sequence": 5,
        "run_id": run_id,
        "kind": "action_outcome_closed",
        "record": record,
    }
    evidence = HostEvidence(run_id, source_refresh=True)
    evidence.observe(receipt_event)
    evidence.observe(closed)
    state = ContinuationState(effects={effect.effect_id: effect})
    data = {
        "effect_id": effect.effect_id,
        "receipt_ref": closed["event_id"],
        "current_binding": binding,
    }
    return evidence, state, data


def test_directory_only_shell_receipt_settles_applied():
    evidence, state, data = _shell_outcome(
        effects=[{"path": "pkg/batch", "change": "created"}],
    )

    assert evidence.effect_outcome(state, data) == "applied"


def test_directory_only_shell_receipt_after_nonzero_exit_still_settles_applied():
    evidence, state, data = _shell_outcome(
        effects=[{"path": "pkg/batch", "change": "created"}],
        exit_code=1,
        receipt_fields={
            "detail": "$ mkdir -p pkg/batch && false\n"
                      "exit 1 | 0.01s | bash | network loopback\n"
                      "files changed by this command: created: pkg/batch",
        },
    )

    assert evidence.effect_outcome(state, data) == "applied"


@pytest.mark.parametrize("exit_code", [0, 1])
def test_finished_shell_receipt_without_workspace_change_settles_applied(exit_code):
    evidence, state, data = _shell_outcome(effects=[], exit_code=exit_code)

    assert evidence.effect_outcome(state, data) == "applied"


def test_shell_receipt_without_recorded_exit_stays_unknown():
    evidence, state, data = _shell_outcome(effects=[], exit_code=None)

    assert evidence.effect_outcome(state, data) == "unknown"


def test_empty_shell_receipt_with_success_settles_as_noop():
    evidence, state, data = _shell_outcome(
        receipt_kind="shell", result="NO_OP", effects=[],
    )

    assert evidence.effect_outcome(state, data) == "not_applied"


def _produced_edit_outcome(*, result):
    """Exercise Bolt's receipt producer and the step ledger's close result."""
    run_id = "run-1"
    binding = {"product_files": {"pkg/value.py": "sha256:original"}, "harness_files": {}}
    events = []

    def emit(kind, **fields):
        event = {"event_id": f"event-{len(events) + 1}", "sequence": len(events) + 1,
                 "run_id": run_id, "kind": kind, **fields}
        events.append(event)
        return event

    journal = SimpleNamespace(run_id=run_id, emit=emit)
    emit("step_dispatch_binding", continuation_effect_id="effect-1")
    lineage = {"continuation_id": "root", "continuation_effect_id": "effect-1",
               "continuation_dispatch_ref": "dispatch-1"}
    ledger = StepLedger()
    arguments = {"path": "pkg/value.py", "old_str": "VALUE = 1", "new_str": "VALUE = 2"}
    ledger.start("str_replace", arguments, 2, binding)
    ledger.pending.update(lineage)
    continuation = SimpleNamespace(dispatch={"accepted": True, "step": 2,
                                             "request_id": "request-1"},
                                   lineage=lambda: lineage)
    measurement = SimpleNamespace(continuation=continuation,
                                  completed_receipt=lambda receipt, _material: ledger.receipt(receipt))
    bolt = KernelBolt.__new__(KernelBolt)
    bolt.run_id = run_id
    bolt.current_step = 2
    bolt.current_request_id = "request-1"
    bolt.audit_journal = journal
    bolt.measurement = measurement
    bolt._receipts = []
    # These are the exact producer calls in react_repair.py's str_replace and
    # pre-tool hook branches; the resulting receipt is never assembled by the test.
    if result == "APPLIED":
        bolt.receipt("edit", "str_replace value.py", "pkg/value.py|VALUE = 1|VALUE = 2",
                     result, "applied.")
    else:
        bolt.receipt("gate", "str_replace", "str_replace", result,
                     "REFUSED HOOK_DENIED: guard: blocked")
    record = ledger.close([], binding)
    closed = emit("action_outcome_closed", record=record)
    evidence = HostEvidence(run_id, source_refresh=True)
    for event in events:
        evidence.observe(event)
    effect = Effect("effect-1", "root", "action-hash", binding_versions(binding), False,
                    tool="str_replace", arguments_hash=digest(arguments),
                    dispatch_ref="dispatch-1", step=2, request_id="request-1",
                    intent_sequence=1)
    state = ContinuationState(effects={effect.effect_id: effect})
    return evidence, state, {"effect_id": effect.effect_id, "receipt_ref": closed["event_id"],
                             "current_binding": binding}, events


def test_hook_rewritten_edit_with_no_net_change_is_settled():
    evidence, state, data, events = _produced_edit_outcome(result="APPLIED")
    receipt = next(event["receipt"] for event in events if event["kind"] == "action_receipt")
    assert receipt["kind"] == "edit"
    assert evidence.effect_outcome(state, data) == "not_applied"


def test_denied_pre_tool_hook_edit_is_settled_without_execution():
    evidence, state, data, events = _produced_edit_outcome(
        result="REFUSED_PREDISPATCH")
    receipt = next(event["receipt"] for event in events if event["kind"] == "action_receipt")
    assert receipt["kind"] == "gate"
    assert _guarded_host_outcome(events[-1]["record"]["result"]) == "ABSENT"
    assert evidence.effect_outcome(state, data) == "not_applied"


def test_net_zero_edit_without_correlated_applied_receipt_stays_unknown():
    evidence, state, data = _shell_outcome(
        receipt_kind="str_replace", result="APPLIED", tool="str_replace",
        receipt_fields={"continuation_effect_id": "different-effect"},
    )

    assert evidence.effect_outcome(state, data) == "unknown"


def test_shell_effect_without_correlated_receipt_stays_unknown():
    evidence, state, data = _shell_outcome(
        effects=[{"path": "pkg/batch", "change": "created"}],
        receipt_fields={"request_id": "other-request"},
    )

    assert evidence.effect_outcome(state, data) == "unknown"


def test_shell_effect_with_missing_receipt_stays_unknown():
    evidence, state, data = _shell_outcome(
        effects=[{"path": "pkg/batch", "change": "created"}],
        record_receipt_ids=[],
    )

    assert evidence.effect_outcome(state, data) == "unknown"


def test_shell_effect_receipt_for_different_effect_id_stays_unknown():
    evidence, state, data = _shell_outcome(
        effects=[{"path": "pkg/batch", "change": "created"}],
        receipt_fields={"continuation_effect_id": "effect-other"},
    )

    assert evidence.effect_outcome(state, data) == "unknown"


@pytest.mark.parametrize(
    "effects",
    [
        pytest.param([{"path": "../outside", "change": "created"}], id="parent-segment"),
        pytest.param([{"path": "/outside", "change": "created"}], id="absolute"),
        pytest.param([{"path": "", "change": "created"}], id="empty"),
        pytest.param([{"path": "pkg/\x00bad", "change": "created"}], id="nul"),
        pytest.param(
            [
                {"path": "pkg/batch", "change": "created"},
                {"path": "../outside", "change": "modified"},
            ],
            id="mixed-valid-and-outside",
        ),
    ],
)
def test_shell_effect_receipt_with_invalid_paths_stays_unknown(effects):
    evidence, state, data = _shell_outcome(effects=effects)

    assert evidence.effect_outcome(state, data) == "unknown"


def test_non_shell_effect_cannot_settle_from_a_shell_effect_receipt():
    evidence, state, data = _shell_outcome(
        effects=[{"path": "pkg/batch", "change": "created"}],
        tool="custom",
    )

    assert evidence.effect_outcome(state, data) == "unknown"
