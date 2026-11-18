"""The background checkpoint writer must pickle real-run state: per-thread runtime state stays out of it."""
from __future__ import annotations

import _thread
import io
import json
import sys
import threading
from pathlib import Path

import pytest

import ast

import fixer
import kernel_bounded_http as probe_http
import kernel_checkpoint as kc
import prompt_ledger
from acceptance._support import run_case
from stateframe.parallel_worker import TickProcessTable


def _ctx() -> dict:
    return {"framecells": {}, "framefuncs": {}, "cvars": kc._cvar_names(),
            "hostfns": kc._host_functions(), "participants": {}}


def _restore_ctx() -> dict:
    return {"framecells": {}, "framefuncs": {}}


def _script(tmp_path: Path) -> Path:
    steps = [
        {"tool": "read", "required": True, "arguments": {"path": "app.py"}},
        {"tool": "str_replace", "required": True,
         "arguments": {"path": "app.py", "old_str": 'VALUE = "before"',
                       "new_str": 'VALUE = "after"'}},
        {"tool": "run_tests", "required": True,
         "arguments": {"target": "tests/test_app.py"}},
        {"tool": "done", "required": True,
         "arguments": {"summary": "Changed the app value."}},
    ]
    script = tmp_path / "checkpoint.script.json"
    script.write_text(json.dumps({"model": "scripted", "steps": steps,
                                  "tokenize": {"mode": "ok"}}))
    return script


@pytest.mark.host_sandbox
def test_real_run_save_publishes_checkpoint(tmp_path: Path) -> None:
    # Red on 71c6e4e1: the background writer died on _PROBE_STATE (threading.local) and never published.
    case = run_case(tmp_path, str(_script(tmp_path)),
                    flags=("--shell", "off", "--keep-checkpoints"))
    assert case.process.returncode == 0, (case.process.stdout, case.process.stderr)
    assert case.summary and case.summary["succeeded"], case.summary
    events = case.run_dir / "checkpoints" / "events.jsonl"
    assert events.is_file(), case.process.stdout + case.process.stderr
    records = [json.loads(line) for line in events.read_text().splitlines()]
    failed = [row for row in records if row.get("event") == "checkpoint_failed"]
    assert not failed, failed
    saved = [row for row in records if row.get("event") == "checkpoint_saved"]
    assert saved, records


def test_payload_pickles_with_probe_state_populated() -> None:
    probe_http._PROBE_STATE.error = "TimeoutError"
    try:
        state = kc._module_state(probe_http.__name__, probe_http)
        assert "_PROBE_STATE" not in state
        kc.dumps({"modules": {probe_http.__name__: state}}, _ctx())
    finally:
        probe_http._PROBE_STATE.error = None


def test_restore_keeps_probe_state_fresh_and_readers_default() -> None:
    probe_http._PROBE_STATE.error = "ProbeNoResponse"
    try:
        raw = kc.dumps({"modules": {probe_http.__name__: kc._module_state(
            probe_http.__name__, probe_http)}}, _ctx())
    finally:
        probe_http._PROBE_STATE.error = None
    restored = kc._Unpickler(io.BytesIO(raw), _restore_ctx()).load()
    saved_keys = set(restored["modules"][probe_http.__name__])
    assert "_PROBE_STATE" not in saved_keys
    # Restore only sets the keys the payload carries, so the thread-local keeps its fresh state.
    seen: list[object] = []

    def reader() -> None:
        seen.append(probe_http.last_probe_error())

    worker = threading.Thread(target=reader)
    worker.start()
    worker.join()
    assert seen == [None]
    assert isinstance(probe_http._PROBE_STATE, _thread._local)


def test_unlisted_thread_local_still_fails_loudly_with_its_path() -> None:
    rogue = threading.local()
    with pytest.raises(TypeError, match="cannot pickle '_thread._local' object") as info:
        kc.dumps({"frame": {"rogue": rogue}}, _ctx())
    # The serializer records the object path as a note ("when serializing dict item 'rogue'").
    assert "rogue" in " ".join(getattr(info.value, "__notes__", []))


@pytest.mark.parametrize("module_name, global_name", [
    ("kernel_bounded_http", "_PROBE_STATE"),
    ("fixer", "_TRACE"),
    ("fixer", "_CALL_META"),
    ("fixer", "_USAGE_SINK"),
    ("fixer", "_TRUNCATION_SINK"),
    ("prompt_ledger", "_CONTEXT"),
])
def test_every_known_thread_local_global_is_excluded(module_name: str, global_name: str) -> None:
    module = __import__(module_name)
    assert isinstance(getattr(module, global_name), _thread._local)
    state = kc._module_state(module_name, module)
    assert global_name not in state
    kc.dumps({"modules": {module_name: state}}, _ctx())


def test_no_unreviewed_thread_local_module_global_in_engine_modules() -> None:
    # Guards future additions: every engine module global that is a threading.local must be reviewed
    # into DROP_GLOBALS, or it fails every real-run save as soon as its module is imported.
    for name in ("kernel_bounded_http", "fixer", "prompt_ledger"):
        __import__(name)
    unreviewed = []
    for name, module in list(sys.modules.items()):
        if module is None or not kc._is_state_module(name, module):
            continue
        for key, value in list(vars(module).items()):
            if type(value) is _thread._local and (name, key) not in kc.DROP_GLOBALS:
                unreviewed.append(f"{name}.{key}")
    assert unreviewed == []


def _enclosing_function_of_calls(source_path: str, names: set[str]) -> set[str]:
    tree = ast.parse(Path(source_path).read_text())
    found: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            callee = node.func
            name = callee.id if isinstance(callee, ast.Name) else (
                callee.attr if isinstance(callee, ast.Attribute) else None)
            if name in names and self.stack:
                found.add(self.stack[-1])
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def test_run_scoped_sinks_are_installed_only_by_fixer_run() -> None:
    # The fixer usage sink, truncation sink and prompt-ledger context are per-thread and never saved.
    # They are re-installed by every fixer.run call, so a resumed process always gets them back from
    # that one path. A new installer elsewhere must be reviewed for resume before it lands.
    installers = _enclosing_function_of_calls(
        fixer.__file__, {"set_usage_sink", "set_truncation_sink", "set_context"})
    assert installers == {"run"}


def test_sinks_are_empty_in_a_fresh_thread_until_run_installs_them() -> None:
    # A restored process starts with no sinks: accounting is a no-op until run() installs its own.
    seen: list[object] = []

    def fresh() -> None:
        totals = fixer._new_usage_totals()
        seen.append(totals["calls"])
        fixer._record_usage({"usage": {"completion_tokens": 3, "prompt_tokens": 5}}, 0.5)
        seen.append(totals["calls"])
        fixer.set_usage_sink(totals)
        fixer._record_usage({"usage": {"completion_tokens": 3, "prompt_tokens": 5}}, 0.5)
        seen.append(totals["calls"])
        seen.append(prompt_ledger.context())

    worker = threading.Thread(target=fresh)
    worker.start()
    worker.join()
    assert seen == [0, 0, 1, ("run-unbound", "task-unbound")]


def test_tick_process_table_round_trips_with_an_active_tick() -> None:
    table = TickProcessTable()
    with table.tick():
        table.tick_snapshot()
        raw = kc.dumps({"table": table}, _ctx())
    restored = kc._Unpickler(io.BytesIO(raw), {"framecells": {}, "framefuncs": {}}).load()["table"]
    assert isinstance(restored, TickProcessTable)
    assert restored._tick_state is not table._tick_state
    assert restored._state().depth == 0
    with restored.tick():
        assert isinstance(restored.tick_snapshot(), dict)


def test_tick_process_table_restore_never_reuses_a_saved_scan() -> None:
    table = TickProcessTable()
    table._running = True  # a scan in flight in the saving process
    table._result = {1: object()}  # type: ignore[dict-item]
    raw = kc.dumps({"table": table}, _ctx())
    restored = kc._Unpickler(io.BytesIO(raw), {"framecells": {}, "framefuncs": {}}).load()["table"]
    assert restored._running is False
    assert restored._result is None
