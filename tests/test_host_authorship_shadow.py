"""Shadow comparison and the off-wire manifest of the host-row identity ledger."""
from __future__ import annotations

import hashlib
import json

import kernel_host_authorship as authorship
import kernel_host_authorship_shadow as shadow
from kernel_host_authorship import HostAuthorship


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))
        return fields


def test_unregistered_row_reports_the_mark_interval_it_appeared_in(monkeypatch):
    monkeypatch.setenv(authorship.MODE_ENV, "shadow")
    ledger, journal = HostAuthorship(3, "req-3"), _Journal()
    panel = authorship.author_row(ledger, "user", authorship.builder().host("[Panel]\nfine"),
                                  kind="panel", site="runtime.panel")
    messages = [{"role": "system", "content": "s"}, panel]
    shadow.mark(ledger, "prepare.start", messages)
    shadow.mark(ledger, "prepare.sources", messages)
    rewritten = {"role": "user", "content": "[Panel]\nfine (trimmed)"}  # a pass changed it off the ledger
    messages[1] = rewritten
    shadow.mark(ledger, "prepare.diet", messages)
    shadow.mark(ledger, "prepare.final", messages)
    shadow.compare(ledger, journal, "reader_a", True, False, rewritten)
    shadow.compare(ledger, journal, "reader_a", ["x"], [], panel)
    shadow.compare(ledger, journal, "reader_b", 1, 0, {"role": "user", "content": "never marked"})
    assert [kind for kind, _fields in journal.events] == ["host_identity_shadow"] * 3
    first, second, third = (fields for _kind, fields in journal.events)
    assert first["reader"] == "reader_a" and first["legacy"] is True and first["ledger"] is False
    assert first["site"] is None and first["mark"] == "prepare.sources..prepare.diet"
    assert second["site"] == "runtime.panel" and second["mark"] == "..prepare.start"
    assert third["site"] is None and third["mark"] is None
    assert all(fields["step"] == 3 and fields["dispatch"] is False for _kind, fields in journal.events)
    json.dumps([fields for _kind, fields in journal.events])
    # Marks are a shadow-only record.
    for value in ("on", "off"):
        monkeypatch.setenv(authorship.MODE_ENV, value)
        other = HostAuthorship(3)
        shadow.mark(other, "prepare.start", messages)
        assert other.marks == []
    shadow.mark(None, "prepare.start", messages)


def test_compare_is_silent_when_answers_match(monkeypatch):
    monkeypatch.setenv(authorship.MODE_ENV, "shadow")
    ledger, journal = HostAuthorship(1), _Journal()
    row = {"role": "user", "content": "anything"}
    assert shadow.compare(ledger, journal, "reader", ("a", 1), ("a", 1), row) is True
    assert shadow.compare(ledger, journal, "reader", None, None, row) is True
    assert journal.events == []
    assert shadow.compare(ledger, journal, "reader", 1, 2, row) is False
    assert len(journal.events) == 1
    # No journal in reach: still an answer, never an error.
    assert shadow.compare(ledger, None, "reader", 1, 2, row) is False
    # Values a journal cannot carry are described, not raised.
    shadow.compare(ledger, journal, "reader", {1, 2}, object(), row)
    json.dumps(journal.events[-1][1])


def test_manifest_spans_match_messages():
    ledger = HostAuthorship(6, "req-6")
    body = "x = 1\ny = 2\n"
    source = "[Current source: a.py]\n" + body
    messages = [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "the task"},
        authorship.author_row(ledger, "user", source, kind="plain_source", site="t.source",
                              data=[(len(source) - len(body), len(body))]),
        {"role": "user", "content": source + " "},
        authorship.author_row(ledger, "user", authorship.builder().host("[Files in frame]\n").data("a.py"),
                              kind="panel", site="t.panel"),
        {"role": "assistant", "content": source},
    ]
    wire = json.dumps({"messages": messages}).encode()
    payload = shadow.manifest(ledger, messages, wire=wire)
    assert payload["step"] == 6 and payload["request_id"] == "req-6"
    assert payload["body_sha256"] == hashlib.sha256(wire).hexdigest()
    assert payload["rows"] == [[2, "plain_source", [[23, len(body)]]], [4, "panel", [[17, 4]]]]
    for index, _kind, spans in payload["rows"]:
        content = messages[index]["content"]
        for start, length in spans:
            assert content[start:start + length] in (body, "a.py")
    json.dumps(payload)
    assert shadow.manifest(None, messages, wire=wire) is None
