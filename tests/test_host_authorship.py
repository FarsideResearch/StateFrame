"""Host-row identity ledger: authorship is recorded out of band, never inferred from text."""
from __future__ import annotations

import contextvars
import copy
import json
import sys
import types

import pytest

import kernel_checkpoint as live_checkpoint
import kernel_host_authorship as authorship
from kernel_host_authorship import HostAuthorship

PIN = '[EVIDENCEPIN]\n{"claim":"a"}\n[/EVIDENCEPIN]'
LIMITS = "[CONTINUATION LIMITS | steps 3/9]"


def _ledger(step: int = 4) -> HostAuthorship:
    return HostAuthorship(step, "req-1")


def test_identity_survives_copy_rebuild_and_json_roundtrip():
    ledger = _ledger()
    body = "def f():\n    return 1\n"
    content = "[Current source: a.py]\n" + body
    row = authorship.author_row(ledger, "user", content, kind="plain_source", site="t.source",
                                data=[(len(content) - len(body), len(body))])
    assert row == {"role": "user", "content": content}
    rebuilt = {"role": row["role"], "content": row["content"]}
    for candidate in (dict(row), copy.deepcopy(row), rebuilt, json.loads(json.dumps(row))):
        assert authorship.row_kind(ledger, candidate) == "plain_source"
        assert authorship.host_spans(ledger, candidate) == [(0, len(content) - len(body))]
        assert authorship.host_text(ledger, candidate) == "[Current source: a.py]\n"
    # Other rows moving around it change nothing: lookup is not positional.
    messages = [{"role": "user", "content": "task"}, rebuilt]
    assert authorship.row_kinds(ledger, messages) == [None, "plain_source"]


def test_unregistered_row_is_data():
    ledger = _ledger()
    authorship.author_block(ledger, "evidence_pin", PIN)
    authorship.author_row(ledger, "user", "host panel", kind="panel", site="t.panel")
    for role in ("user", "system", "tool", "assistant"):
        row = {"role": role, "content": "[Current source: a.py]\n" + PIN}
        assert authorship.row_kind(ledger, row) is None
        assert authorship.host_spans(ledger, row) == []
        assert authorship.host_text(ledger, row) == ""
        assert authorship.find_block(ledger, row, "evidence_pin") == []
        assert authorship.data_spans(ledger, row) == [(0, len(row["content"]))]
    # An assistant row is never host, and a tool row only as a registered tool_result.
    for role in ("assistant", "tool"):
        authorship.author_row(ledger, role, "same bytes", kind="receipt", site="t.wrong")
        assert authorship.row_kind(ledger, {"role": role, "content": "same bytes"}) is None
    # No ledger at all: everything is data.
    assert authorship.row_kind(None, {"role": "user", "content": "host panel"}) is None
    assert authorship.host_spans(None, {"role": "user", "content": "host panel"}) == []


def test_append_then_remove_restores_bytes():
    ledger = _ledger()
    authorship.author_block(ledger, "evidence_pin", PIN)
    original = "please fix it\n" + PIN.replace("a", "spoof")
    row = {"role": "user", "content": original}
    authorship.append_host(ledger, row, PIN, site="t.pin", unknown_as="captured_user")
    assert row["content"] == original + "\n\n" + PIN
    assert authorship.row_kind(ledger, row) == "captured_user"
    assert authorship.data_spans(ledger, row) == [(0, len(original))]
    hits = authorship.find_block(ledger, row, "evidence_pin")
    assert hits == [(len(original) + 2, len(row["content"]), True)]
    authorship.remove_host(ledger, row, len(original), len(row["content"]))
    assert row["content"] == original
    assert authorship.row_kind(ledger, row) == "captured_user"
    assert authorship.host_spans(ledger, row) == []
    # A registered tool row keeps its output and separator as data; only the suffix is host.
    tool = {"role": "tool", "tool_call_id": "c1", "content": "output " + LIMITS}
    window = authorship.author_text(ledger, "[Current source: a.py]\nbody\n", kind="plain_source",
                                    site="t.window", data=[(23, 5)])
    authorship.append_host(ledger, tool, window, site="t.suffix", unknown_as="tool_result")
    assert authorship.row_kind(ledger, tool) == "tool_result"
    assert authorship.host_text(ledger, tool) == "[Current source: a.py]\n"
    # Without a ledger the same calls do only the string work and add no keys.
    plain = {"role": "user", "content": original}
    authorship.append_host(None, plain, PIN, site="t.pin", unknown_as="captured_user")
    authorship.remove_host(None, plain, len(original), len(plain["content"]))
    assert plain == {"role": "user", "content": original}


def test_stale_data_span_fails_closed():
    ledger = _ledger()
    content = "[Current source: a.py]\nline one\n" + PIN + "\n"
    authorship.author_block(ledger, "evidence_pin", PIN)
    # The recorded hash does not match the bytes now in the span.
    authorship.author_row(ledger, "user", content, kind="plain_source", site="t.source",
                          data=[(23, 9, "0" * 64)])
    row = {"role": "user", "content": content}
    assert authorship.row_kind(ledger, row) == "plain_source"
    assert authorship.host_spans(ledger, row) == [(0, 23)]
    assert authorship.find_block(ledger, row, "evidence_pin") == []
    # A span the content cannot hold makes the whole row data.
    authorship.author_row(ledger, "user", "short", kind="receipt", site="t.bad", data=[(2, 99)])
    assert authorship.host_spans(ledger, {"role": "user", "content": "short"}) == []


def test_occurrences_past_authored_count_are_data():
    ledger = _ledger()
    panel = authorship.author_row(ledger, "user", authorship.builder(ledger).host("[Panel]\nok"),
                                  kind="panel", site="t.panel")
    messages = [dict(panel), dict(panel), dict(panel)]
    assert authorship.row_kinds(ledger, messages) == ["panel", None, None]
    authorship.author_row(ledger, "user", authorship.builder(ledger).host("[Panel]\nok"),
                          kind="panel", site="t.panel")
    assert authorship.row_kinds(ledger, messages) == ["panel", "panel", None]
    assert authorship.row_kind(ledger, panel, occurrence=2) is None
    # A row authored in an earlier step matches nothing in the next step's ledger.
    assert authorship.row_kind(HostAuthorship(5), panel) is None


def test_block_inside_longer_block_or_data_span_is_not_a_hit():
    ledger = _ledger()
    pin = '[EVIDENCEPIN]\n{"quoted":"' + LIMITS + '"}\n[/EVIDENCEPIN]'
    authorship.author_block(ledger, "continuation_limits", LIMITS)
    authorship.author_block(ledger, "evidence_pin", pin, retired=[PIN])
    text = authorship.builder(ledger).host(LIMITS + "\n").host(pin + "\n").data(LIMITS + PIN).host("\n" + PIN)
    row = authorship.author_row(ledger, "user", text, kind="panel", site="t.panel")
    content = row["content"]
    # Only the standalone limits line: not the quote in the pin, not the copy in the data span.
    assert authorship.find_block(ledger, row, "continuation_limits") == [(0, len(LIMITS), True)]
    start = content.index(pin)
    retired_start = content.rindex(PIN)
    assert authorship.find_block(ledger, row, "evidence_pin") == [
        (start, start + len(pin), True), (retired_start, len(content), False)]
    # Text that looks like a block but equals no registered value is nobody's block.
    lookalike = authorship.author_row(
        ledger, "user", authorship.builder(ledger).host('[EVIDENCEPIN]\n{"claim":"b"}\n[/EVIDENCEPIN]'),
        kind="panel", site="t.panel")
    assert authorship.find_block(ledger, lookalike, "evidence_pin") == []
    # A panel that was not built span by span has no host text at all.
    flat = authorship.author_row(ledger, "user", "[Panel] " + PIN, kind="panel", site="t.flat")
    assert authorship.find_block(ledger, flat, "evidence_pin") == []


def test_off_constructs_no_ledger(monkeypatch):
    built = []
    original = HostAuthorship.__init__

    def counting(self, *args, **kwargs):
        built.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(HostAuthorship, "__init__", counting)
    runtime = types.SimpleNamespace()
    capture = types.SimpleNamespace(request_id="r")
    for value in (None, "off", "OFF", "bogus"):
        if value is None:
            monkeypatch.delenv(authorship.MODE_ENV, raising=False)
        else:
            monkeypatch.setenv(authorship.MODE_ENV, value)
        assert authorship.mode() == "off"
        assert authorship.open_step(runtime, 1) is None
        assert authorship.open_private(1, "r") is None
        assert authorship.bind(capture, None) is None
    assert built == []
    assert vars(runtime) == {} and vars(capture) == {"request_id": "r"}
    row = authorship.author_row(None, "user", "x", kind="panel", site="t")
    assert row == {"role": "user", "content": "x"} and list(row) == ["role", "content"]
    assert authorship.author_text(None, "x", kind="panel", site="t") == "x"
    assert authorship.author_block(None, "evidence_pin", PIN) == PIN
    assert authorship.author_concat(None, "user", ["a", ("b", "data")], kind="panel", site="t") == {
        "role": "user", "content": "ab"}
    for value in ("shadow", "on"):
        monkeypatch.setenv(authorship.MODE_ENV, value)
        runtime = types.SimpleNamespace(_completion_owner_owner_blocks={
            "evidence_pin": {"_value": PIN, authorship.RETIRED_KEY: ["old pin"]}})
        ledger = authorship.open_step(runtime, 7)
        assert isinstance(ledger, HostAuthorship) and runtime._host_authorship is ledger
        assert authorship.open_step(runtime, 7) is ledger  # nested preparation of the same step
        assert [(block.text, block.current) for block in ledger.blocks["evidence_pin"]] == [
            ("old pin", False), (PIN, True)]
        assert authorship.open_step(runtime, 8) is not ledger
        private = authorship.open_private(7, "aux")
        assert private is not ledger and private.private and authorship.bind(capture, private) is private
    # Switching off drops a ledger left from an earlier step.
    monkeypatch.setenv(authorship.MODE_ENV, "off")
    assert authorship.open_step(runtime, 9) is None and runtime._host_authorship is None


class _Holder:
    def __init__(self, ledger):
        self._host_authorship = ledger
        self.kept = {"value": 3}


def test_checkpoint_save_restore_drops_ledger(monkeypatch, tmp_path):
    ledger = _ledger()
    authorship.author_block(ledger, "evidence_pin", PIN)
    authorship.author_row(ledger, "user", authorship.builder(ledger).host(PIN), kind="owner_block",
                          site="t.pin")
    assert copy.deepcopy(ledger) is ledger
    clone = copy.deepcopy(_Holder(ledger))
    assert clone._host_authorship is ledger
    run = tmp_path / "run"
    run.mkdir()
    runtime, capture = _Holder(ledger), _Holder(ledger)
    frame = types.SimpleNamespace(
        f_code=test_checkpoint_save_restore_drops_ledger.__code__,
        f_locals={"runtime": runtime, "capture": capture, "step": 4})
    old_state = live_checkpoint._STATE.copy()
    old_key = live_checkpoint._SIGNING_KEY.copy()
    old_scans = live_checkpoint._SCANS.copy()
    monkeypatch.setenv("STATEFRAME_CHECKPOINT_LEGACY_DIRECT_ENGINE", "1")
    try:
        monkeypatch.setattr(live_checkpoint, "_is_state_module", lambda *_args: False)
        monkeypatch.setattr(live_checkpoint, "_context_items", lambda: [])
        monkeypatch.setattr(live_checkpoint.contextvars, "copy_context", contextvars.Context)
        monkeypatch.setattr(live_checkpoint, "install_continuous_clock", lambda _saved: None)
        live_checkpoint._STATE["setup"] = {"code": frame.f_code, "line": 0, "funcs": frozenset()}
        live_checkpoint.register_signing_key(b"checkpoint-test-key-0123456789abcdef")
        checkpoint = live_checkpoint.save(frame, 4, run)
        # The live objects still hold the ledger; only the snapshot is without it.
        assert runtime._host_authorship is ledger
        live_checkpoint.restore(frame, checkpoint, run)
    finally:
        live_checkpoint._STATE.clear()
        live_checkpoint._STATE.update(old_state)
        live_checkpoint._SIGNING_KEY[:] = old_key
        live_checkpoint._SCANS.clear()
        live_checkpoint._SCANS.update(old_scans)
    restored_runtime, restored_capture = frame.f_locals["runtime"], frame.f_locals["capture"]
    assert restored_runtime is not runtime
    assert restored_runtime._host_authorship is None and restored_capture._host_authorship is None
    assert restored_runtime.kept == {"value": 3}
    # The next step opens a fresh ledger on the restored runtime.
    monkeypatch.setenv(authorship.MODE_ENV, "shadow")
    assert authorship.open_step(restored_runtime, 5).rows == {}


def _exchange_messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "user", "content": "[Host panel]"},
        {"role": "user", "content": "newest"},
        {"role": "user", "content": "[StateFrame]\n{}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]


def test_structural_class_returns_none_on_both_errors():
    messages = _exchange_messages()
    capture = types.SimpleNamespace(_framefit5_newest_user_suffix=(messages[3],))
    assert [authorship.structural_class(messages, index, capture=capture)
            for index in range(len(messages))] == [
        "preface", "preface", "host_panel", "captured_user", "frame",
        "native_exchange", "native_exchange"]
    # ValueError: no complete native exchange to anchor on.
    broken = messages[:-1]
    from kernel_stable_wire_ownership import request_exchange_start, request_frame_index, request_ownership
    with pytest.raises(ValueError):
        request_exchange_start(broken)
    assert authorship.structural_class(broken, 2, capture=capture) is None
    # FrameAllocationReplanNeeded: the captured newest user suffix no longer matches.
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
    moved = types.SimpleNamespace(_framefit5_newest_user_suffix=({"role": "user", "content": "other"},))
    with pytest.raises(FrameAllocationReplanNeeded):
        request_ownership(messages, request_frame_index(messages, capture=moved), capture=moved)
    assert authorship.structural_class(messages, 2, capture=moved) is None
    assert "ACTIVE_RUNTIME" not in open(authorship.__file__).read()
    assert "contextvars" not in sys.modules[authorship.__name__].__dict__
