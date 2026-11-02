"""Checkpoint serialization and retained offline runs after skill/MCP registration."""
from __future__ import annotations

import ctypes
import json
import pickle
import sys
from pathlib import Path

import pytest

from acceptance._support import run_case
from kernel_checkpoint import _class_state
from kernel_exec_tracker import _BSDInfo, _TaskInfo


def test_native_process_layout_is_rebuilt_by_import() -> None:
    for struct in (_BSDInfo, _TaskInfo):
        original_size = ctypes.sizeof(struct)
        state = _class_state(struct)
        assert "_fields_" not in state
        assert not any(name in state for name, *_ in struct._fields_)
        pickle.dumps(state)
        assert ctypes.sizeof(struct) == original_size

    class WithRuntimeData(ctypes.Structure):
        _fields_ = [("value", ctypes.c_int)]
        label = "retained"

    assert _class_state(WithRuntimeData) == {"label": "retained"}


def _script(tmp_path: Path, variant: str) -> Path:
    steps = []
    if variant == "skill":
        steps.append({"tool": "skill_load", "required": True,
                      "arguments": {"name": "inspect-app"}})
    elif variant == "mcp":
        steps.append({"tool": "mcp_fixture__echo", "required": True,
                      "arguments": {"value": "checkpoint roundtrip"}})
    steps.extend([
        {"tool": "read", "required": True, "arguments": {"path": "app.py"}},
        {"tool": "str_replace", "required": True,
         "arguments": {"path": "app.py", "old_str": 'VALUE = "before"',
                       "new_str": 'VALUE = "after"'}},
        {"tool": "run_tests", "required": True,
         "arguments": {"target": "tests/test_app.py"}},
        {"tool": "done", "required": True,
         "arguments": {"summary": "Changed the app value."}},
    ])
    script = tmp_path / "checkpoint.script.json"
    script.write_text(json.dumps({"model": "scripted", "steps": steps,
                                  "tokenize": {"mode": "ok"}}))
    return script


def _flags(tmp_path: Path, variant: str) -> tuple[str, ...]:
    flags = ("--shell", "off", "--keep-checkpoints")
    if variant == "skill":
        skill = tmp_path / "inspect-app"
        skill.mkdir()
        (skill / "SKILL.md").write_text(
            "---\nname: inspect-app\ndescription: Inspect and edit the example app.\n---\n"
            "Read app.py before changing its value.\n")
        return (*flags, "--skill", str(skill))
    if variant == "mcp":
        server = Path(__file__).parent / "fixtures" / "mcp_server.py"
        config = tmp_path / "mcp.json"
        config.write_text(json.dumps({"schema": "stateframe-mcp/1", "servers": [{
            "name": "fixture", "trust": "Reviewed local fixture for checkpoint coverage",
            "argv": [sys.executable, str(server), "normal"], "network": "none",
            "writes": "none", "replay": "observation", "processes": "single",
            "timeout_seconds": 15}]}))
        return (*flags, "--mcp", str(config))
    return flags


@pytest.mark.host_sandbox
@pytest.mark.parametrize("variant", ("default", "skill", "mcp"))
def test_offline_run_retains_default_checkpoint(tmp_path: Path, variant: str) -> None:
    case = run_case(tmp_path, str(_script(tmp_path, variant)), flags=_flags(tmp_path, variant))
    assert case.process.returncode == 0, (case.process.stdout, case.process.stderr)
    assert case.summary and case.summary["succeeded"], case.summary
    if variant == "skill":
        case.assert_offered("skill_load")
    elif variant == "mcp":
        case.assert_offered("mcp_fixture__echo")
        case.assert_reply("checkpoint roundtrip")
    root = case.run_dir / "checkpoints"
    events = root / "events.jsonl"
    assert events.is_file(), case.process.stdout + case.process.stderr
    records = [json.loads(line) for line in events.read_text().splitlines()]
    saved = [row for row in records if row.get("event") == "checkpoint_saved"]
    assert saved, records
    latest = root / (root / "LATEST").read_text().strip()
    assert (latest / "checkpoint.json").is_file()
    assert (latest / "state.pkl").is_file()
