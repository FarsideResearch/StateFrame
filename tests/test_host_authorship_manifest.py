"""ON journals which rows of each submitted wire the host authored; OFF and shadow add nothing."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap

import kernel_host_authorship as authorship
import kernel_measurement


def _capture(tmp_path, name):
    home = tmp_path / name
    home.mkdir()
    journal = kernel_measurement.AuditJournal(home, name, {})
    capture = kernel_measurement.RequestCapture(journal, 5, "snapshot")
    ledger = authorship.HostAuthorship(5, capture.request_id)
    body_text = "print('hi')\n"
    source = "[Current source: a.py]\n" + body_text
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task with [EVIDENCEPIN] text"},
        authorship.author_row(ledger, "user", source, kind="plain_source", site="t.source",
                              data=[(len(source) - len(body_text), len(body_text))]),
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": source},
    ]
    return journal, capture, ledger, {"model": "scripted", "messages": messages, "tools": []}


def _manifests(journal):
    from kernel_journal_store import open_journal
    with open_journal(journal.path) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return [row for row in rows if (row.get("kind") or row.get("event")) == "host_authorship_manifest"]


def test_manifest_rows_match_submitted_wire(tmp_path, monkeypatch):
    monkeypatch.setenv(authorship.MODE_ENV, "on")
    journal, capture, ledger, body = _capture(tmp_path, "manifest-on")
    capture._host_authorship = ledger
    wire = kernel_measurement._serialize_final_request_body(body, None, capture.request_id)
    capture._emit_host_authorship_manifest(body, wire)
    (event,) = _manifests(journal)
    assert event["body_sha256"] == hashlib.sha256(wire).hexdigest()
    assert event["step"] == 5 and event["request_id"] == capture.request_id
    sent = json.loads(wire)["messages"]
    assert event["rows"] == [[2, "plain_source", [[23, 12]]]]
    for index, _kind, spans in event["rows"]:
        for start, length in spans:
            assert sent[index]["content"][start:start + length] == "print('hi')\n"
    # The manifest is journal-only: the wire has no trace of the ledger.
    assert b"host_authorship" not in wire and b"plain_source" not in wire
    assert wire == json.dumps(body).encode()
    # It is emitted beside every final wire of the send path: the three legacy
    # serializations and the sealed qualified request (the default send path).
    tree = ast.parse(inspect.getsource(kernel_measurement))
    names = [getattr(node.func, "id", getattr(node.func, "attr", None))
             for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert names.count("_serialize_final_request_body") == 3
    assert names.count("_emit_host_authorship_manifest") == 4


def _is_call(statement, name):
    return (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
            and getattr(statement.value.func, "attr", None) == name)


def _assigns_wire_from(statement, source):
    """``wire, body = record.wire, ...`` or ``wire = _serialize_final_request_body(...)``."""
    if not isinstance(statement, ast.Assign):
        return False
    target = statement.targets[0]
    names = [target] if isinstance(target, ast.Name) else list(getattr(target, "elts", ()))
    if not any(isinstance(name, ast.Name) and name.id == "wire" for name in names):
        return False
    return any((isinstance(node, ast.Attribute) and node.attr == source
                and isinstance(node.value, ast.Name) and node.value.id == "record")
               or (isinstance(node, ast.Call) and getattr(node.func, "id", None) == source)
               for node in ast.walk(statement.value))


def test_every_final_wire_of_the_send_path_journals_its_manifest():
    """Each final wire of _send_once is followed at once by its manifest call.

    The default send path takes its wire from the sealed qualified request, not
    from a fresh serialization; a run with the switch on journaled no manifest
    while only the legacy serializations were covered.
    """
    source = textwrap.dedent(inspect.getsource(kernel_measurement.RequestCapture._send_once))
    sites = []
    blocks = [block for node in ast.walk(ast.parse(source))
              for block in (getattr(node, field, None) for field in ("body", "orelse", "finalbody"))
              if isinstance(block, list)]
    for statements in blocks:
        for index, statement in enumerate(statements):
            if (_assigns_wire_from(statement, "wire")
                    or _assigns_wire_from(statement, "_serialize_final_request_body")):
                following = statements[index + 1] if index + 1 < len(statements) else None
                sites.append((statement.lineno, following is not None
                               and _is_call(following, "_emit_host_authorship_manifest")))
    assert len(sites) == 4, sites
    assert all(covered for _line, covered in sites), sites


def test_off_and_shadow_emit_no_manifest_and_same_bytes(tmp_path, monkeypatch):
    wires = {}
    for mode in ("off", "shadow", "on"):
        monkeypatch.setenv(authorship.MODE_ENV, mode)
        journal, capture, ledger, body = _capture(tmp_path, f"manifest-{mode}")
        if mode != "off":
            capture._host_authorship = ledger
        before = json.dumps(body)
        wire = kernel_measurement._serialize_final_request_body(body, None, "request")
        capture._emit_host_authorship_manifest(body, wire)
        assert json.dumps(body) == before
        wires[mode] = wire
        assert len(_manifests(journal)) == (1 if mode == "on" else 0)
    assert wires["off"] == wires["shadow"] == wires["on"]
    # ON without a ledger in reach emits nothing rather than an empty claim.
    journal, capture, _ledger, body = _capture(tmp_path, "manifest-none")
    capture._emit_host_authorship_manifest(body, wires["on"])
    assert _manifests(journal) == []
