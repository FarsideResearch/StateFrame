"""MR-6(b): under STATEFRAME_ROUTING_SCHEMA=2 the selected endpoint's exact body counter is used when
available (method "tokenize"); otherwise the UTF-8 byte bound applies and one token_estimate_fallback
record is journaled per endpoint. Schema 1 never consults the exact counter and never journals."""
from __future__ import annotations

import json

import kernel_stateframe_model_router as router

BODY = {"messages": [{"role": "user", "content": "héllo ☃"}], "tools": []}


def _wire(body: object) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _endpoint(name: str = "selected", url: str | None = "http://127.0.0.1:9100/v1") -> router.EndpointConfig:
    return router.EndpointConfig(name=name, model_id="m", url=url, context_tokens=65536)


class FakeJournal:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict]] = []

    def emit(self, name: str, **fields: object) -> None:
        self.records.append((name, fields))

    def named(self, name: str) -> list[dict]:
        return [fields for event, fields in self.records if event == name]


def test_v2_uses_exact_count_when_available(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    seen: list[object] = []

    def exact(body, endpoint):
        seen.append(body)
        return 77

    monkeypatch.setattr(router, "exact_token_count", exact)
    journal = FakeJournal()
    journaled: set[str] = set()
    assert router.estimate_request_tokens(BODY, _endpoint(), emit=journal.emit, journaled=journaled) == 77
    assert seen == [BODY]
    assert journal.records == []
    assert journaled == set()


def test_v2_fallback_is_byte_bound_with_one_record_per_endpoint(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    monkeypatch.setattr(router, "exact_token_count", lambda body, endpoint: None)
    journal = FakeJournal()
    journaled: set[str] = set()
    endpoint = _endpoint()
    expected = len(_wire(BODY).encode("utf-8"))
    assert router.estimate_request_tokens(BODY, endpoint, emit=journal.emit, journaled=journaled) == expected
    assert router.estimate_request_tokens(BODY, endpoint, emit=journal.emit, journaled=journaled) == expected
    records = journal.named("token_estimate_fallback")
    assert len(records) == 1
    assert records[0]["endpoint"] == "selected"
    assert records[0]["tokenizer"] == "utf8-byte-conservative-v1"
    assert records[0]["reason"] == "exact_counter_unavailable"


def test_v2_fallback_journals_again_for_a_different_endpoint(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    monkeypatch.setattr(router, "exact_token_count", lambda body, endpoint: None)
    journal = FakeJournal()
    journaled: set[str] = set()
    router.estimate_request_tokens(BODY, _endpoint("a"), emit=journal.emit, journaled=journaled)
    router.estimate_request_tokens(BODY, _endpoint("b"), emit=journal.emit, journaled=journaled)
    assert [r["endpoint"] for r in journal.named("token_estimate_fallback")] == ["a", "b"]


def test_v2_counter_error_is_treated_as_unavailable(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")

    def broken(body, endpoint):
        raise TimeoutError("tokenizer request deadline expired")

    monkeypatch.setattr(router, "exact_token_count", broken)
    journal = FakeJournal()
    expected = len(_wire(BODY).encode("utf-8"))
    assert router.estimate_request_tokens(BODY, _endpoint(), emit=journal.emit,
                                          journaled=set()) == expected
    records = journal.named("token_estimate_fallback")
    assert len(records) == 1
    assert records[0]["reason"] == "exact_counter_error"


def test_v1_never_consults_exact_counter_and_never_journals(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)

    def forbidden(body, endpoint):
        raise AssertionError("schema 1 must not consult the exact counter")

    monkeypatch.setattr(router, "exact_token_count", forbidden)
    journal = FakeJournal()
    expected = len(_wire(BODY).encode("utf-8"))
    endpoint = _endpoint()
    assert router.estimate_request_tokens(BODY, endpoint, emit=journal.emit, journaled=set()) == expected
    assert router.estimate_request_tokens(BODY, endpoint, emit=journal.emit, journaled=set()) == expected
    assert journal.records == []
    assert router._estimate_request(BODY, endpoint=endpoint) == expected


def test_v2_endpoint_without_url_has_no_exact_count():
    assert router.exact_token_count(BODY, _endpoint(url=None)) is None
