"""Protocol edits (edit_source/rewrite_source) must pass the pytest collection guard, like legacy edits.

The collection guard in react_repair (_collection_refusal) runs only for legacy edit names. Protocol
edits compose their candidate bytes at publish time (_ep_write), so the same guard runs there.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PYTEST_INI = "[pytest]\naddopts = -q\n"
PRODUCT = "VALUE = 1\n"


class _Stop(BaseException):
    """Stop the worker once the scripted replies have been captured."""


def _drive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, protocol: str | None,
           steps: list[dict]) -> tuple[list[str], Path]:
    from kernel_measurement import ACTIVE_REQUEST
    from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
    from kernel_stateframe_runtime import StateFrameRuntime
    from kernel_verdict_identity import _ACTIVE_PROVIDER
    from stateframe.engine.probes import react_repair
    from stateframe.scripted_endpoint import Script, serve
    from stateframe.task_row import build_row
    from stateframe.workcopy import create

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pytest.ini").write_text(PYTEST_INI, encoding="utf-8")
    (repo / "product.py").write_text(PRODUCT, encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests/test_product.py").write_text(
        "from product import VALUE\n\ndef test_value():\n    assert VALUE == 1\n", encoding="utf-8")
    copy_root = tmp_path / "copy"
    row = build_row(create(repo, copy_root), task="Keep product value", target="product.py",
                    task_id="gate-collection-protocol")
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
    monkeypatch.setattr(react_repair, "MAX_STEPS", len(steps) + 1)
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": steps})
    server = serve(script)
    monkeypatch.setattr(react_repair, "API", f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
    monkeypatch.setattr(react_repair, "MODEL", "scripted")
    monkeypatch.setattr(react_repair, "KEY", "")
    original_ask = react_repair.ask_tools
    replies: list[str] = []
    sends = 0
    stop_after = len(steps) + 1

    def capture(messages, *args, **kwargs):
        nonlocal sends
        sends += 1
        replies.extend(str(m["content"]) for m in messages if m.get("role") == "tool")
        if sends == stop_after:
            raise _Stop
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
    return replies, copy_root


def _last_json(replies: list[str]) -> dict:
    return json.loads(replies[-1])


def test_protocol_rewrite_of_pytest_ini_is_refused_by_collection_guard(tmp_path, monkeypatch):
    # RED at base: rewrite_source publishes pytest.ini without the collection guard.
    # Handle q2h1 = REWRITE handle over the whole pytest.ini the read delivered (request ordinal 2).
    replies, copy_root = _drive(
        tmp_path, monkeypatch, protocol="handles",
        steps=[{"tool": "read", "required": True, "arguments": {"path": "pytest.ini"}},
               {"tool": "rewrite_source", "required": True,
                "arguments": {"handle": "q1h1", "new_text": "[pytest]\naddopts = -q -x\n"}}])
    body = _last_json(replies)
    assert body["status"] == "REJECTED", body
    assert "REFUSED PROTECTED_TEST" in json.dumps(body) and "pytest.ini" in json.dumps(body)
    assert (copy_root / "pytest.ini").read_text(encoding="utf-8") == PYTEST_INI


def test_legacy_str_replace_of_pytest_ini_stays_refused(tmp_path, monkeypatch):
    # Control: legacy collection refusal is unchanged.
    replies, copy_root = _drive(
        tmp_path, monkeypatch, protocol=None,
        steps=[{"tool": "str_replace", "required": True, "arguments": {
            "edits": [{"path": "pytest.ini", "old": "-q", "new": "-q -x"}]}}])
    assert "REFUSED PROTECTED_TEST" in replies[-1], replies[-1]
    assert (copy_root / "pytest.ini").read_text(encoding="utf-8") == PYTEST_INI
