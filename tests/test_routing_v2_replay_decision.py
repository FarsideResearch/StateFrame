"""FRP: a cross-process replay reads back the journaled RouteDecision under STATEFRAME_ROUTING_SCHEMA=2.

The original decision is journaled as ``route_decision`` (kernel_route_decision._remember). A fresh
process that replays a prepared body must rebuild that decision instead of trusting the body alone:
same model, endpoint and role, credential resolved from the journaled key_env name, and a drifting
body model overridden by the journal. Schema 1 never reads the journal.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernel_route_decision as route_decision
from kernel_continuation_journal import ContinuationJournal
from kernel_route_decision import resolve_route_decision
from kernel_routing_schema import ROUTING_SCHEMA_ENV

CALLER_URL = "http://127.0.0.1:8000/v1"
ENDPOINT = "http://127.0.0.1:9100/v1"
ROUTER_KEY_VALUE = "router-secret"


class Journal:
    """Stands in for ContinuationJournal: emit appends, records(rebuild=False) reads back."""

    def __init__(self, rows: tuple[tuple[str, dict], ...] = ()) -> None:
        self.log: list[tuple[str, dict]] = list(rows)
        self.reads = 0

    def emit(self, kind: str, **fields: object) -> None:
        self.log.append((kind, dict(fields)))

    def records(self, *, rebuild: bool = True):
        self.reads += 1
        for kind, fields in list(self.log):
            yield {"kind": kind, **fields}

    def named(self, kind: str) -> list[dict]:
        return [fields for name, fields in self.log if name == kind]


class Meter:
    def __init__(self, journal: object, *, request_id: str = "req-1",
                 router: object | None = None) -> None:
        self.request_id = request_id
        self.journal = journal
        self.model_router = router
        self.frame_route = None


class Router:
    """Mimics ModelRouter.prepare_call for the first (decision) build only."""

    def __init__(self, endpoint: str = ENDPOINT) -> None:
        self.endpoint = endpoint

    def prepare_call(self, capture: object, body: dict, endpoint: str, key: str) -> SimpleNamespace:
        adapted = dict(body)
        adapted["model"] = "router-model"
        return SimpleNamespace(
            body=adapted, endpoint=self.endpoint, key=ROUTER_KEY_VALUE, applied=True,
            decision=SimpleNamespace(endpoint=SimpleNamespace(api_key_env="ROUTER_KEY")),
        )

    def validate_final_request(self, capture: object, body: dict) -> None:
        return None


class ForbiddenRouter(Router):
    """A replay must never consult the router for the destination."""

    def prepare_call(self, capture: object, body: dict, endpoint: str, key: str) -> SimpleNamespace:
        raise AssertionError("router consulted during replay")


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "2")
    monkeypatch.setenv("ROUTER_KEY", ROUTER_KEY_VALUE)


def _journaled(*, request_id: str = "req-1", model: str = "journaled-model",
               endpoint: str | None = ENDPOINT, key_env: str | None = "ROUTER_KEY",
               source: str = "model_router", role: str | None = None) -> tuple[str, dict]:
    fields: dict[str, object] = {
        "request_id": request_id, "model": model, "endpoint_host": "127.0.0.1",
        "key_env": key_env, "source": source, "role": role, "dispatch": False,
    }
    if endpoint is not None:
        fields["endpoint"] = endpoint
    return ("route_decision", fields)


def _body(model: str = "journaled-model") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def test_replay_in_new_process_rebuilds_journaled_decision(v2):
    journal = Journal((_journaled(),))
    meter = Meter(journal, router=ForbiddenRouter())

    decision, body, endpoint, key = resolve_route_decision(
        meter, _body(), ENDPOINT, "caller-key", recovery_prepared=True)

    assert decision.source == route_decision.SOURCE_REPLAYED == "replayed"
    assert (decision.model, decision.endpoint, decision.key_env) == (
        "journaled-model", ENDPOINT, "ROUTER_KEY")
    assert (endpoint, key) == (ENDPOINT, ROUTER_KEY_VALUE)
    assert body["model"] == "journaled-model"
    assert journal.named("route_drift_pinned") == []
    replayed = journal.named("route_decision")
    assert replayed[-1]["source"] == "replayed"
    assert ROUTER_KEY_VALUE not in repr(replayed)


def test_replay_without_journaled_decision_stays_recovered(v2):
    journal = Journal()
    meter = Meter(journal)

    decision, body, endpoint, key = resolve_route_decision(
        meter, _body("already-routed"), CALLER_URL, "caller-key", recovery_prepared=True)

    assert decision.source == route_decision.SOURCE_RECOVERED == "recovered"
    assert (endpoint, key, body["model"]) == (CALLER_URL, "caller-key", "already-routed")


def test_replay_ignores_journaled_decisions_for_other_requests(v2):
    journal = Journal((_journaled(request_id="req-9"),))
    meter = Meter(journal)

    decision, _, _, _ = resolve_route_decision(
        meter, _body("already-routed"), CALLER_URL, "caller-key", recovery_prepared=True)

    assert decision.source == "recovered"


def test_replay_drift_keeps_journaled_model_and_journals_pin_once(v2):
    journal = Journal((_journaled(),))
    meter = Meter(journal, router=ForbiddenRouter())

    for _ in range(2):
        decision, body, endpoint, _ = resolve_route_decision(
            meter, _body("wire-model"), ENDPOINT, "caller-key", recovery_prepared=True)

    assert decision.model == "journaled-model"
    assert body["model"] == "journaled-model"
    assert endpoint == ENDPOINT
    assert journal.named("route_drift_pinned") == [{
        "request_id": "req-1", "wanted_model": "wire-model", "kept_model": "journaled-model",
        "dispatch": False}]


def test_replay_with_no_key_env_uses_empty_credential_never_caller_key(v2):
    journal = Journal((_journaled(key_env=None, source="static_route"),))
    meter = Meter(journal)

    decision, _, endpoint, key = resolve_route_decision(
        meter, _body(), CALLER_URL, "caller-key", recovery_prepared=True)

    assert decision.source == "replayed"
    assert (endpoint, key, decision.key_env) == (ENDPOINT, "", None)


def test_ambiguous_journal_falls_back_to_recovered(v2):
    journal = Journal((_journaled(model="a"), _journaled(model="b")))
    meter = Meter(journal)

    decision, _, endpoint, _ = resolve_route_decision(
        meter, _body("already-routed"), CALLER_URL, "caller-key", recovery_prepared=True)

    assert decision.source == "recovered"
    assert endpoint == CALLER_URL


def test_replay_without_journaled_endpoint_falls_back_to_recovered(v2):
    journal = Journal((_journaled(endpoint=None),))
    meter = Meter(journal)

    decision, _, endpoint, _ = resolve_route_decision(
        meter, _body("already-routed"), CALLER_URL, "caller-key", recovery_prepared=True)

    assert decision.source == "recovered"
    assert endpoint == CALLER_URL


def test_credentialed_endpoint_is_not_journaled_for_replay(v2):
    journal = Journal()
    meter = Meter(journal, router=Router(endpoint="http://user:pw-secret@127.0.0.1:9100/v1?k=abc"))

    resolve_route_decision(meter, _body(), CALLER_URL, "caller-key", recovery_prepared=False)

    record = journal.named("route_decision")[0]
    assert "endpoint" not in record
    assert "pw-secret" not in repr(record) and "k=abc" not in repr(record)


def test_schema_1_never_reads_the_journal(monkeypatch):
    monkeypatch.setenv(ROUTING_SCHEMA_ENV, "1")
    journal = Journal((_journaled(),))
    meter = Meter(journal)

    decision, body, endpoint, key = resolve_route_decision(
        meter, _body("already-routed"), CALLER_URL, "caller-key", recovery_prepared=True)

    assert decision.source == "recovered"
    assert (endpoint, key, body["model"]) == (CALLER_URL, "caller-key", "already-routed")
    assert journal.reads == 0


def test_real_journal_round_trip_across_fresh_meter(v2, tmp_path):
    journal = ContinuationJournal(tmp_path, "frp-replay", {"origin": "routing-v2-replay"})
    first = Meter(journal, router=Router())
    original, _, _, _ = resolve_route_decision(
        first, _body(), CALLER_URL, "caller-key", recovery_prepared=False)
    assert original.source == "model_router"

    fresh = Meter(journal)  # a new meter with no cached decision, as after a process restart
    decision, body, endpoint, key = resolve_route_decision(
        fresh, dict(_body("router-model")), CALLER_URL, "caller-key", recovery_prepared=True)

    assert decision.source == "replayed"
    assert (decision.model, endpoint, key) == ("router-model", ENDPOINT, ROUTER_KEY_VALUE)
    assert body["model"] == "router-model"
    sources = [r["source"] for r in journal.records(rebuild=False) if r["kind"] == "route_decision"]
    assert sources == ["model_router", "replayed"]


WANTED_ENDPOINT = "http://127.0.0.1:9200/v1"


def test_endpoint_only_drift_journals_endpoint_fields_once(v2):
    journal = Journal((_journaled(),))
    meter = Meter(journal, router=ForbiddenRouter())

    for _ in range(2):
        decision, body, endpoint, _ = resolve_route_decision(
            meter, _body(), WANTED_ENDPOINT, "caller-key", recovery_prepared=True)

    assert (decision.model, endpoint, body["model"]) == (
        "journaled-model", ENDPOINT, "journaled-model")
    assert journal.named("route_drift_pinned") == [{
        "request_id": "req-1", "wanted_endpoint": WANTED_ENDPOINT, "kept_endpoint": ENDPOINT,
        "dispatch": False}]


def test_model_and_endpoint_drift_journal_one_event_with_both_pairs(v2):
    journal = Journal((_journaled(),))
    meter = Meter(journal, router=ForbiddenRouter())

    for _ in range(2):
        resolve_route_decision(meter, _body("wire-model"), WANTED_ENDPOINT, "caller-key",
                               recovery_prepared=True)

    assert journal.named("route_drift_pinned") == [{
        "request_id": "req-1", "wanted_model": "wire-model", "kept_model": "journaled-model",
        "wanted_endpoint": WANTED_ENDPOINT, "kept_endpoint": ENDPOINT, "dispatch": False}]


def test_credentialed_wanted_endpoint_is_stripped_in_drift_event(v2):
    journal = Journal((_journaled(),))
    meter = Meter(journal, router=ForbiddenRouter())

    resolve_route_decision(meter, _body(), "http://user:pw-secret@127.0.0.1:9200/v1?k=abc",
                           "caller-key", recovery_prepared=True)

    record = journal.named("route_drift_pinned")[0]
    assert record["wanted_endpoint"] == WANTED_ENDPOINT
    assert "pw-secret" not in repr(record) and "k=abc" not in repr(record)


def test_cross_process_reopened_journal_replays_decision(v2, tmp_path):
    """Write the decision, close, reopen the same events.jsonl as resume does, replay in a fresh meter."""
    from kernel_continuation_journal import ContinuationJournal as Journal2

    manifest = {"origin": "routing-v2-cross-process"}
    writer = ContinuationJournal(tmp_path, "frp-cross", manifest)
    resolve_route_decision(Meter(writer, router=Router()), _body(), CALLER_URL, "caller-key",
                           recovery_prepared=False)
    head = writer._head  # the trusted head resume is anchored to
    events_path = writer.path
    writer.close()

    reopened = Journal2.resume(events_path, task_id="frp-cross", manifest=manifest,
                               expected_head=head)
    try:
        fresh = Meter(reopened)
        decision, body, endpoint, key = resolve_route_decision(
            fresh, dict(_body("router-model")), CALLER_URL, "caller-key", recovery_prepared=True)
        assert decision.source == "replayed"
        assert (decision.model, endpoint, key) == ("router-model", ENDPOINT, ROUTER_KEY_VALUE)
        assert body["model"] == "router-model"
    finally:
        reopened.close()
