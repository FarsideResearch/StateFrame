"""edit_source reaches the executor only under STATEFRAME_EDIT_PROTOCOL=handles."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

SOURCE = "def qualifies(cart):\n    cart.minimum_spend = 25\n    return cart.total >= cart.minimum_spend\n"


class _Stop(BaseException):
    """Stop the worker after the scripted tool reply has been captured."""


def _drive(tmp_path, monkeypatch, *, protocol: str | None, tool: str, arguments: dict,
           offer_anyway: bool = False):
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from stateframe.engine.probes import react_repair
    from stateframe.scripted_endpoint import Script, serve
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    repo = tmp_path / "repo"
    (repo / "shop/pricing").mkdir(parents=True)
    (repo / "shop/pricing/rules.py").write_text(SOURCE, encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests/test_pricing.py").write_text("def test_pricing():\n    assert False\n", encoding="utf-8")
    copy_root = tmp_path / "copy"
    row = build_row(create(repo, copy_root), task="Raise the minimum spend",
                    target="shop/pricing/rules.py", task_id="edit-proto")
    output = tmp_path / "run/worker/measurement-runs"
    output.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_IDEALPATH", "off")
    if protocol is None:
        monkeypatch.delenv("STATEFRAME_EDIT_PROTOCOL", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_EDIT_PROTOCOL", protocol)
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    original_init = StateFrameRuntime.__init__

    def scoped_init(self, owner, **kwargs):
        kwargs["output"] = output
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(StateFrameRuntime, "__init__", scoped_init)
    monkeypatch.setattr(react_repair, "MAX_STEPS", 3)
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": tool, "required": True, "arguments": arguments}]})
    server = serve(script)
    monkeypatch.setattr(react_repair, "API", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")
    original_ask = react_repair.ask_tools
    replies: list[str] = []
    menus: list[list[str]] = []
    sends = 0

    def capture(messages, *args, **kwargs):
        nonlocal sends
        sends += 1
        tools = kwargs.get("tools")
        if tools is None and len(args) >= 3:
            tools = args[2]
        if tools:
            menus.append([t["function"]["name"] for t in tools])
        replies.extend(str(m["content"]) for m in messages if m.get("role") == "tool")
        if sends == 2:
            raise _Stop
        if offer_anyway and tools is not None:
            # The scripted endpoint only emits tools the request offers. Offer the
            # protocol schema on the wire (the engine's own menu is unchanged) so a
            # model that calls an unadvertised tool can be simulated.
            from kernel_edit_source_tool_surface import EDIT_SOURCE_SCHEMA
            wire_tools = [*tools, EDIT_SOURCE_SCHEMA]
            if "tools" in kwargs or len(args) < 3:
                kwargs = {**kwargs, "tools": wire_tools}
            else:
                args = (*args[:2], wire_tools, *args[3:])
        return original_ask(messages, *args, **kwargs)

    monkeypatch.setattr(react_repair, "ask_tools", capture)
    contexts = (ACTIVE_PARENT, ACTIVE_RUNTIME, ACTIVE_REQUEST, _ACTIVE_PROVIDER)
    tokens = [c.set(None) for c in contexts]
    try:
        with pytest.raises(_Stop):
            react_repair.work(row)
    finally:
        server.shutdown()
        server.server_close()
        for c, t in zip(contexts, tokens):
            c.reset(t)
    return replies, menus, copy_root


def test_handles_mode_routes_edit_source_to_the_typed_executor(tmp_path, monkeypatch):
    replies, menus, copy_root = _drive(
        tmp_path, monkeypatch, protocol="handles", tool="edit_source",
        arguments={"edits": [{"handle": "q1h1", "old_text": "25", "new_text": "30"}]})
    assert menus and "edit_source" in menus[0] and "str_replace" not in menus[0]
    assert replies, "no tool reply captured"
    body = json.loads(replies[-1])
    assert body["tool"] == "edit_source" and body["status"] == "REJECTED"
    # No current-source map is attached yet (that wiring is a later unit): every handle is stale.
    assert [r["code"] for r in body["refusals"]] == ["STALE_HANDLE"]
    assert (copy_root / "shop/pricing/rules.py").read_text() == SOURCE  # nothing mutated


def test_legacy_safe_treats_edit_source_as_an_unknown_tool_and_keeps_str_replace(tmp_path, monkeypatch):
    replies, menus, copy_root = _drive(
        tmp_path, monkeypatch, protocol=None, tool="edit_source",
        arguments={"edits": [{"handle": "q1h1", "old_text": "25", "new_text": "30"}]},
        offer_anyway=True)
    assert menus and "str_replace" in menus[0] and "edit_source" not in menus[0]
    assert replies and replies[-1].startswith("Unknown tool `edit_source`")
    assert (copy_root / "shop/pricing/rules.py").read_text() == SOURCE
