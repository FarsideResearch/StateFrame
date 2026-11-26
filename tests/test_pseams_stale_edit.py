"""A path edit must not use source that changed after the model last read it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from test_edit_protocol_switch_wiring import (
    RULES,
    RULES_PATH,
    SpanKind,
    _drive,
    _str_replace_step,
    _tool_replies,
)


EXTERNAL_RULES = "# external update after the read\n" + RULES
SECOND_PATH = "shop/pricing/extra.py"
SECOND_SOURCE = "quantity = 1\n"
SECOND_EXTERNAL = "# external update after the read\n" + SECOND_SOURCE
READ_RULES = {"tool": "read", "required": True,
              "arguments": {"path": RULES_PATH}}


def test_read_revision_snapshot_restores_stale_guard_state(tmp_path):
    from stateframe.engine.kernel_model_read_revisions import (
        ModelReadRevisionLedger,
        revision_of,
    )

    root = tmp_path / "repo"
    path = root / RULES_PATH
    path.parent.mkdir(parents=True)
    original = RULES.encode()
    path.write_bytes(original)
    ledger = ModelReadRevisionLedger()
    ledger.observe_read(root, path, revision_of(original))
    restored = ModelReadRevisionLedger(ledger.snapshot())

    path.write_bytes(EXTERNAL_RULES.encode())
    refusal = restored.refusal(root, path, operation="str_replace")
    assert refusal is not None
    assert RULES_PATH in refusal
    assert "changed since your last read" in refusal


def test_continuation_worker_checkpoint_restores_read_revisions():
    from stateframe.engine.kernel_continuation_restart import (
        WORKER_FIELDS,
        _valid_worker_state_fields,
    )
    from stateframe.engine.kernel_model_read_revisions import (
        ModelReadRevisionLedger,
        revision_of,
    )

    expected = {RULES_PATH: revision_of(RULES.encode())}
    worker = {field: None for field in WORKER_FIELDS}
    worker["model_read_revisions"] = expected

    assert _valid_worker_state_fields(worker)
    restored = ModelReadRevisionLedger(json.loads(json.dumps(worker))["model_read_revisions"])
    assert restored.snapshot() == expected

    malformed = dict(worker, model_read_revisions={"../escape.py": "sha256:" + "0" * 64})
    assert not _valid_worker_state_fields(malformed)


@pytest.fixture(autouse=True)
def _legacy_edit_mode(monkeypatch):
    for name in (
            "STATEFRAME_BOUND_EDIT", "STATEFRAME_EMPIRICAL_FIXES",
            "STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY",
            "STATEFRAME_TOKENLAB_COORDINATED_LEDGER", "STATEFRAME_EFFECT_BOUNDARY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REACT_TOOLCALL", "1")


def _engine_repo(tmp_path: Path) -> Path:
    roots = sorted((tmp_path / "worker-tmp").glob("react_*/repo"))
    assert len(roots) == 1, f"expected one engine workspace, found {roots}"
    return roots[0]


def _wrap_ask(monkeypatch, before_response, after_edit=None):
    from stateframe.engine.probes import react_repair

    original = react_repair.ask_tools
    original_reply = react_repair._tool_reply_for_call
    calls = 0

    def wrapped(messages, *args, **kwargs):
        nonlocal calls
        before_response(calls)
        calls += 1
        return original(messages, *args, **kwargs)

    monkeypatch.setattr(react_repair, "ask_tools", wrapped)

    if after_edit is not None:
        def capture_edit_result(messages, tool_call_id, content, *args, **kwargs):
            result = original_reply(messages, tool_call_id, content, *args, **kwargs)
            rendered = str(content)
            if ("changed since your last read" in rendered
                    or rendered.startswith(("str_replace applied.", "edit applied."))
                    or '"status": "APPLIED"' in rendered):
                after_edit(rendered)
            return result

        monkeypatch.setattr(react_repair, "_tool_reply_for_call", capture_edit_result)


def _file_edit_step(source: str) -> dict:
    return {"tool": "edit", "required": True,
            "arguments": {"path": RULES_PATH, "function_source": source}}


def test_bound_edit_refusal_survives_an_unread_target(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_BOUND_EDIT", "on")
    sends, root = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=2,
        steps=[_str_replace_step("cart.minimum_spend = 25",
                                 "cart.minimum_spend = 30")],
    )

    reply = _tool_replies(sends[-1][0])[-1]
    assert "bound edit target required" in reply
    assert (root / RULES_PATH).read_text(encoding="utf-8") == RULES


def test_empty_out_of_range_read_refreshes_target_revision(tmp_path, monkeypatch):
    target = "shop/pricing/__init__.py"
    read_target = {"tool": "read", "required": True,
                   "arguments": {"path": target}}
    edit_empty_target = {"tool": "str_replace", "required": True,
                         "arguments": {"path": target, "old_str": "missing",
                                       "new_str": "replacement"}}

    def truncate_before_reread(index):
        if index == 1:
            (_engine_repo(tmp_path) / target).write_bytes(b"")

    _wrap_ask(monkeypatch, truncate_before_reread)
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=4, target=target,
        steps=[read_target, read_target, edit_empty_target],
    )

    replies = _tool_replies(sends[-1][0])
    assert "has 0 lines" in replies[-2]
    assert "changed since your last read" not in replies[-1]
    assert "str_replace" in replies[-1]


def test_nonempty_out_of_range_read_does_not_refresh_revision(tmp_path, monkeypatch):
    out_of_range_read = {"tool": "read", "required": True,
                         "arguments": {"path": RULES_PATH, "line": 999}}
    after_edit = []

    def change_before_reread(index):
        if index == 1:
            (_engine_repo(tmp_path) / RULES_PATH).write_text(
                EXTERNAL_RULES, encoding="utf-8")

    _wrap_ask(monkeypatch, change_before_reread,
              after_edit=lambda _reply: after_edit.append(
                  (_engine_repo(tmp_path) / RULES_PATH).read_bytes()))
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=4,
        steps=[READ_RULES, out_of_range_read,
               _str_replace_step("cart.minimum_spend = 25",
                                 "cart.minimum_spend = 30")],
    )

    replies = _tool_replies(sends[-1][0])
    assert "Valid range is 1-4" in replies[-2]
    assert "changed since your last read" in replies[-1]
    assert after_edit == [EXTERNAL_RULES.encode()]


def test_collab_drift_after_explicit_read_merges_handle_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_COLLAB_DRIFT", "on")
    monkeypatch.delenv("STATEFRAME_STRICT_DRIFT", raising=False)
    edit_source = {"tool": "edit_source", "required": True, "arguments": {
        "edits": [{"handle": "q1h1", "old_text": "25", "new_text": "30"}],
    }}

    def drift_after_read_was_delivered(index):
        if index == 1:
            (_engine_repo(tmp_path) / RULES_PATH).write_text(
                EXTERNAL_RULES, encoding="utf-8")

    applied_sources = []
    _wrap_ask(monkeypatch, lambda _index: None,
              after_edit=lambda _reply: applied_sources.append(
                  (_engine_repo(tmp_path) / RULES_PATH).read_text(encoding="utf-8")))
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol="handles", stop_at=3,
        steps=[READ_RULES, edit_source], handle_kind=SpanKind.SPAN,
        after_model_response=drift_after_read_was_delivered,
    )

    reply = _tool_replies(sends[-1][0])[-1]
    result = json.loads(reply)
    assert result["status"] == "APPLIED", reply
    assert applied_sources == [EXTERNAL_RULES.replace(
        "cart.minimum_spend = 25", "cart.minimum_spend = 30", 1)]


def test_read_then_external_change_refuses_str_replace_without_writing(
        tmp_path, monkeypatch):
    after_edit = []

    def change_before_second_model_turn(index):
        if index == 1:
            (_engine_repo(tmp_path) / RULES_PATH).write_text(
                EXTERNAL_RULES, encoding="utf-8")

    _wrap_ask(monkeypatch, change_before_second_model_turn,
              after_edit=lambda _reply: after_edit.append(
                  (_engine_repo(tmp_path) / RULES_PATH).read_bytes()))
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=3,
        steps=[READ_RULES, _str_replace_step("cart.minimum_spend = 25",
                                             "cart.minimum_spend = 30")],
    )

    reply = _tool_replies(sends[-1][0])[-1]
    assert "shop/pricing/rules.py" in reply
    assert "changed since your last read" in reply
    assert "read shop/pricing/rules.py" in reply
    assert after_edit == [EXTERNAL_RULES.encode()]
    assert len(sends) == 3, "a stale edit refusal should return through the normal tool loop"


def test_stale_refusal_for_applied_filename_does_not_advance_revision(tmp_path, monkeypatch):
    from stateframe.engine.probes import react_repair

    target = "shop/pricing/discount_applied.py"
    read_target = {"tool": "read", "required": True,
                   "arguments": {"path": target}}
    stale_edit = _str_replace_step("cart.minimum_spend = 25",
                                  "cart.minimum_spend = 30")
    stale_edit["arguments"]["path"] = target
    applied_or_refused = []
    original_reply = react_repair._tool_reply_for_call

    def capture_edit_result(messages, tool_call_id, content, *args, **kwargs):
        result = original_reply(messages, tool_call_id, content, *args, **kwargs)
        rendered = str(content)
        if rendered.startswith("str_replace applied.") or "changed since your last read" in rendered:
            current = (_engine_repo(tmp_path) / target).read_text(encoding="utf-8")
            applied_or_refused.append((rendered, current))
        return result

    monkeypatch.setattr(react_repair, "_tool_reply_for_call", capture_edit_result)

    def change_before_stale_edit(index):
        if index == 1:
            (_engine_repo(tmp_path) / target).write_text(EXTERNAL_RULES, encoding="utf-8")

    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, target=target, stop_at=4,
        steps=[read_target, stale_edit, stale_edit],
        after_model_response=change_before_stale_edit,
    )

    replies = _tool_replies(sends[-1][0])
    assert "changed since your last read" in replies[-1]
    assert len(applied_or_refused) == 2
    assert all("changed since your last read" in reply for reply, _ in applied_or_refused)
    assert all(current == EXTERNAL_RULES for _, current in applied_or_refused)


def test_read_then_unchanged_source_applies_expected_bytes(tmp_path, monkeypatch):
    after_edit = []

    _wrap_ask(monkeypatch, lambda _index: None,
              after_edit=lambda _reply: after_edit.append(
                  (_engine_repo(tmp_path) / RULES_PATH).read_bytes()))
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=3,
        steps=[READ_RULES, _str_replace_step("cart.minimum_spend = 25",
                                             "cart.minimum_spend = 30")],
    )

    reply = _tool_replies(sends[-1][0])[-1]
    expected = RULES.replace("cart.minimum_spend = 25",
                              "cart.minimum_spend = 30", 1).encode()
    assert reply.startswith("str_replace applied.")
    assert after_edit == [expected]


def test_reread_after_external_change_allows_edit_on_fresh_revision(tmp_path, monkeypatch):
    after_edit = []

    def change_before_second_model_turn(index):
        if index == 1:
            (_engine_repo(tmp_path) / RULES_PATH).write_text(
                EXTERNAL_RULES, encoding="utf-8")

    _wrap_ask(monkeypatch, change_before_second_model_turn,
              after_edit=lambda _reply: after_edit.append(
                  (_engine_repo(tmp_path) / RULES_PATH).read_bytes()))
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=4,
        steps=[READ_RULES, READ_RULES,
               _str_replace_step("cart.minimum_spend = 25",
                                 "cart.minimum_spend = 30")],
    )

    reply = _tool_replies(sends[-1][0])[-1]
    expected = EXTERNAL_RULES.replace("cart.minimum_spend = 25",
                                      "cart.minimum_spend = 30", 1).encode()
    assert reply.startswith("str_replace applied.")
    assert after_edit == [expected]


def test_file_edit_refuses_stale_source_before_replacing_file(tmp_path, monkeypatch):
    replacement = RULES.replace("minimum_spend = 25", "minimum_spend = 30")
    after_edit = []

    def change_before_second_model_turn(index):
        if index == 1:
            (_engine_repo(tmp_path) / RULES_PATH).write_text(
                EXTERNAL_RULES, encoding="utf-8")

    _wrap_ask(monkeypatch, change_before_second_model_turn,
              after_edit=lambda _reply: after_edit.append(
                  (_engine_repo(tmp_path) / RULES_PATH).read_bytes()))
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=3,
        steps=[READ_RULES, _file_edit_step(replacement)],
    )

    reply = _tool_replies(sends[-1][0])[-1]
    assert "shop/pricing/rules.py" in reply
    assert "changed since your last read" in reply
    assert "read shop/pricing/rules.py" in reply
    assert after_edit == [EXTERNAL_RULES.encode()]


def test_edit_source_protocol_refuses_stale_source_without_stopping(tmp_path, monkeypatch):
    after_edit = []

    def change_before_edit(index):
        if index == 1:
            (_engine_repo(tmp_path) / RULES_PATH).write_text(
                EXTERNAL_RULES, encoding="utf-8")

    _wrap_ask(monkeypatch, change_before_edit,
              after_edit=lambda _reply: after_edit.append(
                  (_engine_repo(tmp_path) / RULES_PATH).read_bytes()))
    edit_source = {"tool": "edit_source", "required": True, "arguments": {
        "edits": [{"handle": "q1h1", "old_text": "25", "new_text": "30"}],
    }}
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol="handles", stop_at=3,
        steps=[READ_RULES, edit_source], handle_kind=SpanKind.SPAN,
    )

    reply = _tool_replies(sends[-1][0])[-1]
    result = json.loads(reply)
    assert result["status"] == "REJECTED"
    assert RULES_PATH in reply
    assert "changed since your last read" in reply
    assert after_edit == [EXTERNAL_RULES.encode()]
    assert len(sends) == 3


def test_batch_refuses_every_file_when_one_read_revision_is_stale(
        tmp_path, monkeypatch):
    from stateframe.engine.probes import react_repair

    original_tools = react_repair._tools_for_probe_modes

    def tools_with_batch(*args, **kwargs):
        tools = original_tools(*args, **kwargs)
        for tool in tools:
            function = tool.get("function", {})
            if function.get("name") != "str_replace":
                continue
            parameters = function.setdefault("parameters", {})
            properties = parameters.setdefault("properties", {})
            properties["edits"] = {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                }, "required": ["path", "old_str", "new_str"]},
            }
            parameters["required"] = list(parameters.get("required", ())) + ["edits"]
        return tools

    monkeypatch.setattr(react_repair, "_tools_for_probe_modes", tools_with_batch)
    monkeypatch.setenv("STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "on")
    read_second = {"tool": "read", "required": True,
                   "arguments": {"path": SECOND_PATH}}
    batch_step = {"tool": "str_replace", "required": True, "arguments": {
        "path": RULES_PATH,
        "old_str": "cart.minimum_spend = 25",
        "new_str": "cart.minimum_spend = 30",
        "edits": [
            {"path": RULES_PATH, "old_str": "cart.minimum_spend = 25",
             "new_str": "cart.minimum_spend = 30"},
            {"path": SECOND_PATH, "old_str": "quantity = 1", "new_str": "quantity = 2"},
        ],
    }}

    def add_then_change_second_file(index):
        if index == 1:
            extra = _engine_repo(tmp_path) / SECOND_PATH
            extra.parent.mkdir(parents=True, exist_ok=True)
            extra.write_text(SECOND_SOURCE, encoding="utf-8")
        elif index == 2:
            (_engine_repo(tmp_path) / SECOND_PATH).write_text(
                SECOND_EXTERNAL, encoding="utf-8")

    after_batch = []
    _wrap_ask(monkeypatch, add_then_change_second_file,
              after_edit=lambda _reply: after_batch.append((
                  (_engine_repo(tmp_path) / RULES_PATH).read_bytes(),
                  (_engine_repo(tmp_path) / SECOND_PATH).read_bytes())))
    sends, _ = _drive(
        tmp_path, monkeypatch, protocol=None, stop_at=4,
        steps=[READ_RULES, read_second, batch_step],
    )

    reply = _tool_replies(sends[-1][0])[-1]
    result = json.loads(reply)
    assert result["status"] == "REJECTED"
    assert "changed since your last read" in reply
    assert after_batch == [(RULES.encode(), SECOND_EXTERNAL.encode())]
