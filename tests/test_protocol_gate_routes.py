"""Protocol edit tools (edit_source/rewrite_source) must escalate like legacy edits.

The per-step route gate classifies a reply by tool name only (no arguments), so a
protocol edit is a workspace write and must reach the ``edit`` role exactly as a
legacy ``edit`` call does. Before the fix, ``classify_calls`` returned ``read`` for
protocol edits, so a read-model reply that edited through handles was never re-asked.
"""
from __future__ import annotations

import json

import pytest

from stateframe import routes
import kernel_frame_routes as frame_routes
from kernel_edit_tool_names import PROTOCOL_EDIT_TOOL_NAMES


DEFAULT = {"endpoint": "http://127.0.0.1:8080/v1", "model": "base", "api_key_env": None,
           "temperature": 1.0, "max_tokens": 24000, "timeout_s": 1800.0,
           "slot": None, "context_tokens": 65536, "allow_remote": False}


@pytest.fixture(autouse=True)
def clear_routes():
    frame_routes.install(None)
    yield
    frame_routes.install(None)


def table(*entries):
    return routes.resolve(DEFAULT, cli=list(entries), environ={}).public()


def call(name: str, arguments: dict | None = None) -> list[dict]:
    return [{"function": {"name": name, "arguments": json.dumps(arguments or {})}}]


EDIT_SOURCE_ARGS = {"edits": [{"handle": "h_1", "old": "a", "new": "b"}]}
REWRITE_SOURCE_ARGS = {"handle": "h_2", "text": "def f():\n    return 2\n"}


@pytest.mark.parametrize("name", sorted(PROTOCOL_EDIT_TOOL_NAMES))
def test_protocol_edit_classifies_as_edit(name):
    assert frame_routes.classify_calls(call(name, EDIT_SOURCE_ARGS)) == "edit"


def test_edit_source_escalates_read_reply_to_edit_role():
    frame_routes.install(table("read.model=reader"))
    reader = frame_routes.resolve("read")
    assert frame_routes.escalation_target(reader, call("edit_source", EDIT_SOURCE_ARGS)) == "edit"


def test_rewrite_source_escalates_read_reply_to_edit_role():
    frame_routes.install(table("read.model=reader"))
    reader = frame_routes.resolve("read")
    assert frame_routes.escalation_target(reader, call("rewrite_source", REWRITE_SOURCE_ARGS)) == "edit"


def test_protocol_edit_mixed_with_read_still_escalates():
    frame_routes.install(table("read.model=reader"))
    reader = frame_routes.resolve("read")
    batch = call("grep", {"pattern": "x"}) + call("edit_source", EDIT_SOURCE_ARGS)
    assert frame_routes.classify_calls(batch) == "edit"
    assert frame_routes.escalation_target(reader, batch) == "edit"


def test_legacy_edit_still_escalates_unchanged():
    frame_routes.install(table("read.model=reader"))
    reader = frame_routes.resolve("read")
    legacy = call("str_replace", {"path": "a.py", "old_str": "a", "new_str": "b"})
    assert frame_routes.classify_calls(legacy) == "edit"
    assert frame_routes.escalation_target(reader, legacy) == "edit"


def test_read_only_protocol_free_reply_does_not_escalate():
    frame_routes.install(table("read.model=reader"))
    reader = frame_routes.resolve("read")
    assert frame_routes.classify_calls(call("grep", {"pattern": "x"})) == "read"
    assert frame_routes.escalation_target(reader, call("grep", {"pattern": "x"})) is None
