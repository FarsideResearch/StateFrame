"""Owner proof rebind refusals explain the cause without changing request authority."""
import copy
import hashlib
import io
from types import SimpleNamespace

import kernel_completion_owner as owner
import kernel_reason_log as rl
import pytest
from kernel_completion_owner import _digest
from kernel_request_seal import OwnerPresentation, RequestSealError, freeze


@pytest.fixture
def reason_log(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    stream = io.StringIO()
    events = []
    journal = SimpleNamespace(emit=lambda *args, **kwargs: events.append((args, kwargs)))
    rl.configure(level="DEBUG", stream=stream, journal=journal)
    yield stream, events
    rl.reset_for_tests()


def _presentation():
    contract = {
        "budget_wire_digest": _digest("saved limits"),
        "evidence_pin_wire_digest": _digest(["saved pin"]),
    }
    return contract, OwnerPresentation(
        hashlib.sha256(freeze(contract)).hexdigest(), "saved limits", "saved pin",
    )


@pytest.mark.parametrize("field,label", [
    ("budget_wire_digest", "limits"),
    ("evidence_pin_wire_digest", "evidence"),
])
@pytest.mark.parametrize("removed", [False, True])
def test_rebind_refusal_logs_once_without_journal_or_mutation(reason_log, field, label, removed):
    stream, events = reason_log
    previous, presentation = _presentation()
    current = dict(previous)
    if removed:
        del current[field]
    else:
        current[field] = _digest("different proof")
    before = copy.deepcopy((previous, current, presentation))
    change = "removed from" if removed else ("differ from" if label == "limits" else "differs from")
    message = f"owner presentation {label} {change} rebound contract"

    for _ in range(2):
        with pytest.raises(RequestSealError, match=message):
            presentation.rebind(previous, current)

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1 and lines[0].startswith("[stateframe] ERROR request_seal:")
    assert message in lines[0]
    cause = rl.run_summary()["by_cause"][f"invariant:{message}"]
    assert cause["count"] == 2 and cause["level"] == "ERROR"
    assert events == []
    assert (previous, current, presentation) == before
    presentation.validate(previous)


def test_healthy_rebind_preserves_owner_proof_and_stays_silent(reason_log):
    stream, events = reason_log
    previous, presentation = _presentation()
    current = {**previous, "tool_surface_digest": "new surface"}
    rebound = presentation.rebind(previous, current)

    rebound.validate(current)
    presentation.validate(previous)
    assert rebound.continuation_limits == presentation.continuation_limits
    assert rebound.evidence_pin == presentation.evidence_pin
    assert stream.getvalue() == "" and events == []
    assert rl.run_summary()["total"] == 0


def test_invalid_final_surface_presentation_logs_without_changing_contract(reason_log):
    stream, events = reason_log
    tool = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
    contract = {
        "contract_kind": owner.LEAN_REQUEST_CONTRACT_KIND, "route": "lean",
        "request_id": "request-1", "admitted_tool_schemas": [tool],
        "advertised_actions": ["read"], "owner_blocks": {},
    }
    contract["decision_digest"] = owner._contract_decision_digest(contract)
    capture = SimpleNamespace(request_action_contract=contract, owner_presentation=object())
    surface = {
        "request_id": "request-1", "tools": [tool], "names": ["read"],
        "tool_schema_digest": _digest([tool]),
    }
    runtime = SimpleNamespace(journal=SimpleNamespace(emit=lambda *args, **kwargs: events.append((args, kwargs))))
    before = copy.deepcopy(contract)
    presentation = capture.owner_presentation
    message = "invalid owner presentation at final surface binding"

    for _ in range(2):
        assert not owner.bind_final_tool_surface(runtime, capture, surface, messages=[])

    assert len(stream.getvalue().splitlines()) == 1
    assert message in stream.getvalue()
    assert rl.run_summary()["by_cause"][f"invariant:{message}"]["count"] == 2
    assert len(events) == 2
    assert all(args == ("completion_owner_unavailable",) and fields["error_type"] == "RequestSealError"
               for args, fields in events)
    assert capture.request_action_contract == contract == before
    assert capture.owner_presentation is presentation
