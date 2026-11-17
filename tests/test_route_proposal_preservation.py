"""MODEL-ROUTING-UNIFY MR-5: a role change keeps a completed proposal (routing v2 only)."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateframe import routes
from kernel_measurement import ACTIVE_REQUEST
from kernel_native_tools import NativeToolBatch
import kernel_frame_routes as frame_routes
from kernel_routing_schema import routing_v2

DEFAULT = {"endpoint": "http://127.0.0.1:8080/v1", "model": "base", "api_key_env": None,
           "temperature": 1.0, "max_tokens": 24000, "timeout_s": 1800.0,
           "slot": None, "context_tokens": 65536, "allow_remote": False}

READ = {"id": "r", "function": {"name": "read", "arguments": "{}"}}
EDIT = {"id": "w", "function": {"name": "edit", "arguments": '{"path":"a.py"}'}}


@pytest.fixture(autouse=True)
def clear_routes(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)
    frame_routes.install(None)
    yield
    frame_routes.install(None)


def install_split_table():
    # reader answers on its own model; edit resolves to the default model, so edit escalates.
    spec = routes.resolve(DEFAULT, cli=["read.model=reader"], environ={}).public()
    frame_routes.install(spec)


def committed_names(journal):
    received = [fields for kind, fields in journal.events if kind == "native_batch_received"]
    assert len(received) == 1
    return [call["function"]["name"] for call in received[0]["calls"]]


class Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))
        return fields


def test_routing_schema_default_is_v1_and_2_enables_v2(monkeypatch):
    assert routing_v2() is False
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    assert routing_v2() is True
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "1")
    assert routing_v2() is False


def test_v2_split_runs_reads_and_withholds_escalating_calls(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_split_table()
    answering = frame_routes.resolve("read")
    kept, withheld, target = frame_routes.route_split(answering, [READ, EDIT])
    assert kept == [READ]
    assert withheld == [EDIT]
    assert target == "edit"


def test_v2_pure_edit_is_fully_withheld(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_split_table()
    kept, withheld, target = frame_routes.route_split(frame_routes.resolve("read"), [EDIT])
    assert kept == [] and withheld == [EDIT] and target == "edit"


def test_v2_read_only_batch_is_untouched(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_split_table()
    kept, withheld, target = frame_routes.route_split(frame_routes.resolve("read"), [READ])
    assert kept == [READ] and withheld == [] and target is None


def test_v1_whole_batch_escalation_is_unchanged():
    install_split_table()
    assert frame_routes.escalation_target(frame_routes.resolve("read"), [READ, EDIT]) == "edit"


def test_v2_native_batch_commits_only_the_returned_subset(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_split_table()
    journal = Journal()
    capture = SimpleNamespace(journal=journal, request_id="request", snapshot_id="snapshot",
                              selected_response_event_id="response")
    token = ACTIVE_REQUEST.set(capture)
    batch = NativeToolBatch()
    seen = []

    def before_commit(calls):
        seen.append(list(calls))
        return [READ]  # the caller withheld the edit

    try:
        batch.next(lambda *_a, **_k: ("", [READ, EDIT]), [], before_commit=before_commit)
    finally:
        ACTIVE_REQUEST.reset(token)
    assert seen == [[READ, EDIT]]
    assert committed_names(journal) == ["read"]
    assert batch.size == 1


def test_v1_native_batch_commits_full_batch_when_before_commit_returns_none():
    # v1 before_commit only raises or returns None; a None return commits the full batch.
    install_split_table()
    journal = Journal()
    capture = SimpleNamespace(journal=journal, request_id="request", snapshot_id="snapshot",
                              selected_response_event_id="response")
    token = ACTIVE_REQUEST.set(capture)
    batch = NativeToolBatch()
    try:
        batch.next(lambda *_a, **_k: ("", [READ, EDIT]), [], before_commit=lambda calls: None)
    finally:
        ACTIVE_REQUEST.reset(token)
    assert committed_names(journal) == ["read", "edit"]
    assert batch.size == 2


def test_preserved_pure_proposal_is_carried_with_digest_and_journal():
    journal = Journal()
    frame_routes.preserve_proposal(journal, request_id="r1", target_role="edit",
                                   calls=[EDIT], revision="rev-A")
    kind, fields = journal.events[0]
    expected = hashlib.sha256(json.dumps([EDIT], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert kind == "route_proposal_preserved"
    assert fields["request_id"] == "r1" and fields["target_role"] == "edit"
    assert fields["revision"] == "rev-A" and fields["call_names"] == ["edit"]
    assert fields["proposal_digest"] == expected
    assert frame_routes.has_preserved_proposals() is True

    messages = frame_routes.take_preserved_messages("rev-A")
    assert len(messages) == 1 and messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert "against revision rev-A" in content
    assert "against revision rev-A; apply them if still correct, or revise:" in content
    assert json.dumps([EDIT], sort_keys=True, indent=2) in content
    assert "stale" not in content
    assert frame_routes.has_preserved_proposals() is False
    assert frame_routes.take_preserved_messages("rev-A") == []


def test_stale_revision_is_marked_stale_in_carried_message():
    frame_routes.preserve_proposal(Journal(), request_id="r1", target_role="edit",
                                   calls=[EDIT], revision="rev-A")
    content = frame_routes.take_preserved_messages("rev-B")[0]["content"]
    assert "stale" in content
    assert "rev-A" in content and "rev-B" in content


def test_install_clears_preserved_proposals():
    frame_routes.preserve_proposal(Journal(), request_id="r1", target_role="edit",
                                   calls=[EDIT], revision="rev-A")
    frame_routes.install(None)
    assert frame_routes.has_preserved_proposals() is False


def test_v1_escalation_branch_keeps_discard_event_and_v2_branch_preserves():
    from probes import react_repair

    tree = ast.parse(Path(react_repair.__file__).read_text())
    work = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "work")
    gates = [node for node in ast.walk(work) if isinstance(node, ast.If)
             and ast.unparse(node.test) == "routing_v2()"]
    assert len(gates) >= 1
    v2_gate = gates[0]
    body_src = "\n".join(ast.unparse(stmt) for stmt in v2_gate.body)
    orelse_src = "\n".join(ast.unparse(stmt) for stmt in v2_gate.orelse)
    assert "route_proposal_preserved" not in orelse_src
    assert "route_escalation_discarded" in orelse_src
    assert "route_escalation_discarded" not in body_src
    assert "route_proposal_preserved" in body_src or "preserve_proposal" in body_src


def test_v2_finish_call_is_withheld_when_its_route_differs(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_split_table()
    done = {"id": "d", "function": {"name": "done", "arguments": "{}"}}
    kept, withheld, target = frame_routes.route_split(frame_routes.resolve("read"), [READ, done])
    assert kept == [READ] and withheld == [done] and target == "finish"


class FakeMeasurement:
    def __init__(self, artifact):
        self.artifact = artifact

    def binding(self):
        return {"artifact": self.artifact}


def test_next_request_gets_withheld_proposal_once_and_v1_gets_identical_inputs():
    from probes import react_repair

    msgs = [{"role": "user", "content": "task"}]
    uncapped = [{"role": "user", "content": "task"}]
    # No preserved proposal (v1, or nothing withheld): the very same objects reach prepare().
    assert react_repair._preserved_request_inputs(FakeMeasurement("rev-A"), msgs, uncapped) == (msgs, uncapped)
    assert react_repair._preserved_request_inputs(FakeMeasurement("rev-A"), msgs, uncapped)[0] is msgs

    frame_routes.preserve_proposal(Journal(), request_id="r1", target_role="edit",
                                   calls=[EDIT], revision="rev-A")
    next_msgs, next_uncapped = react_repair._preserved_request_inputs(FakeMeasurement("rev-A"), msgs, uncapped)
    assert next_msgs[:1] == msgs and next_msgs[-1]["role"] == "user"
    assert "against revision rev-A" in next_msgs[-1]["content"]
    assert next_uncapped == next_msgs
    assert msgs == [{"role": "user", "content": "task"}]  # history itself is never mutated

    # Consumed: the following request carries nothing extra.
    again, _ = react_repair._preserved_request_inputs(FakeMeasurement("rev-A"), msgs, uncapped)
    assert again is msgs


def test_react_repair_gate_name_is_bound_to_the_schema_switch():
    # The loop gates read a bare routing_v2(); an unbound name would only fail once a route escalates.
    from probes import react_repair
    from kernel_routing_schema import routing_v2 as schema_switch

    assert react_repair.routing_v2 is schema_switch


def test_routing_v2_p1_journal_kinds_are_resume_safe_tail() -> None:
    from stateframe.engine import kernel_continuation_restart as restart

    for kind in ("route_proposal_preserved", "token_estimate_fallback"):
        assert kind in restart.SAFE_TAIL
        assert restart.resume_event_rejection_reason(kind) is None
def _install_temperature_split_table():
    # edit differs from read only in temperature: same endpoint and model.
    frame_routes.install(routes.resolve(DEFAULT, cli=["edit.temperature=0.2"], environ={}).public())


def test_v2_temperature_only_edit_is_withheld(monkeypatch):
    # MR-7b: under v2 the split compares full route identity, so a temperature-only difference withholds.
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    _install_temperature_split_table()
    kept, withheld, target = frame_routes.route_split(frame_routes.resolve("read"), [READ, EDIT])
    assert kept == [READ] and withheld == [EDIT] and target == "edit"


def test_v2_withheld_proposal_makes_next_request_resolve_to_target_role(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    _install_temperature_split_table()
    before = frame_routes.role_for_request("rev-A")
    _kept, withheld, target = frame_routes.route_split(frame_routes.resolve("read"), [READ, EDIT])
    frame_routes.preserve_proposal(Journal(), request_id="r1", target_role=target,
                                   calls=withheld, revision="rev-A")
    assert before != "edit"
    assert frame_routes.role_for_request("rev-A") == "edit"
    # The carried request runs on the target role's settings, not the answering route's.
    assert frame_routes.applied_settings(frame_routes.resolve("edit"))["temperature"] == 0.2
    assert frame_routes.applied_settings(frame_routes.resolve("read"))["temperature"] == 1.0


def test_v2_role_falls_back_after_preserved_proposal_is_consumed(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    _install_temperature_split_table()
    before = frame_routes.role_for_request("rev-A")
    frame_routes.preserve_proposal(Journal(), request_id="r1", target_role="edit",
                                   calls=[EDIT], revision="rev-A")
    assert frame_routes.role_for_request("rev-A") == "edit"
    frame_routes.take_preserved_messages("rev-A")
    # The carrying request itself still runs on the target role; the fallback starts once its action closes.
    assert frame_routes.role_for_request("rev-A") == "edit"
    frame_routes.close_action({"action": "edit", "status": "APPLIED"}, revision="rev-A")
    assert frame_routes.role_for_request("rev-A") != "edit" or frame_routes.CARRY_TARGET == {}
    assert frame_routes.CARRY_TARGET == {}


def test_v2_oldest_preserved_proposal_wins_and_decide2_is_not_overridden(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    _install_temperature_split_table()
    frame_routes.preserve_proposal(Journal(), request_id="r1", target_role="edit",
                                   calls=[EDIT], revision="rev-A")
    frame_routes.preserve_proposal(Journal(), request_id="r2", target_role="finish",
                                   calls=[EDIT], revision="rev-A")
    assert frame_routes.role_for_request("rev-A") == "edit"
    assert frame_routes.role_for_request("rev-A", request_kind="decide2") == "decide2"


def test_v1_split_and_role_ignore_temperature_and_preserved_proposals(monkeypatch):
    # Schema 1 keeps (endpoint, model) for the split and today's role logic, so nothing changes.
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "1")
    _install_temperature_split_table()
    kept, withheld, target = frame_routes.route_split(frame_routes.resolve("read"), [READ, EDIT])
    assert kept == [READ, EDIT] and withheld == [] and target is None
    before = frame_routes.role_for_request("rev-A")
    frame_routes.preserve_proposal(Journal(), request_id="r1", target_role="edit",
                                   calls=[EDIT], revision="rev-A")
    assert frame_routes.role_for_request("rev-A") == before


def test_v2_carrying_request_keeps_target_role_after_the_carry_is_consumed(monkeypatch):
    # Live order: the carry is consumed before prepare() asks for the role, so the role must survive it.
    import kernel_frame_routes as fr
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    fr.PRESERVED_PROPOSALS.clear(); fr.CARRY_TARGET.clear()
    fr.PRESERVED_PROPOSALS.append({"target_role": "edit", "revision": "r1", "calls": [], "digest": "d"})
    messages = fr.take_preserved_messages("r1")
    assert messages and not fr.PRESERVED_PROPOSALS
    assert fr.role_for_request("r1") == "edit"
    fr.close_action({"action": "edit", "status": "APPLIED"}, revision="r1")
    assert fr.role_for_request("r1") != "edit" or fr.CARRY_TARGET == {}
    assert fr.CARRY_TARGET == {}


def test_v1_take_preserved_messages_sets_no_carry_target(monkeypatch):
    import kernel_frame_routes as fr
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)
    fr.PRESERVED_PROPOSALS.clear(); fr.CARRY_TARGET.clear()
    fr.PRESERVED_PROPOSALS.append({"target_role": "edit", "revision": "r1", "calls": [], "digest": "d"})
    fr.take_preserved_messages("r1")
    assert fr.CARRY_TARGET == {}
