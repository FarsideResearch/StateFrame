"""A crash inside a real engine run is typed, persisted and reported, and never reads as success.

The public CLI runs the engine child as usual. A seeded crash replaces ONE product function on the run path
(probes.react_repair.ask_tools, the model request) from a child-only sitecustomize, installed through the
public --python flag (prior art: tests/acceptance/test_routes_host.py). The assertions read only the
persisted artifacts of the run directory: summary.json, the engine's measurement journal and problems.jsonl.
Nothing here spies on record_fault, outcomes or the runner.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import sys
from pathlib import Path

from _journal import measurement_events
from stateframe import cli
from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
SEEDED_CLASS = "SeededCrash"
SEEDED_MESSAGE = "seeded crash: receipt incomplete"
HANDLER_ID = "react_repair.run_exception"
FAULT_ID_PATTERN = re.compile(r"[0-9a-f]{16}")

# Imported by the engine child at start-up. The crash is armed lazily: once the product module has finished
# loading, its module-level ask_tools is replaced by a wrapper that raises. functools.wraps keeps any
# attributes the engine reads from ask_tools, so the run reaches the request path exactly as in production.
SITECUSTOMIZE = '''
import builtins
import functools
import sys

_import = builtins.__import__
_TARGETS = ("probes.react_repair", "stateframe.engine.probes.react_repair")


class SeededCrash(RuntimeError):
    pass


def _arm(module):
    original = getattr(module, "ask_tools", None)
    if original is None or getattr(original, "_seeded_crash", False):
        return

    @functools.wraps(original)
    def crashing(*_args, **_kwargs):
        raise SeededCrash("seeded crash: receipt incomplete")

    crashing._seeded_crash = True
    module.ask_tools = crashing


def _import_then_arm(name, *args, **kwargs):
    module = _import(name, *args, **kwargs)
    for target in _TARGETS:
        loaded = sys.modules.get(target)
        if loaded is not None:
            _arm(loaded)
    return module


builtins.__import__ = _import_then_arm
'''


def _seeded_python(tmp_path: Path) -> Path:
    """An interpreter wrapper whose only difference is the sitecustomize on PYTHONPATH."""
    hook_dir = tmp_path / "seeded-hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    wrapper = tmp_path / "seeded-python"
    wrapper.write_text(
        "#!/bin/sh\n"
        f"PYTHONPATH={shlex.quote(str(hook_dir))} exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8")
    wrapper.chmod(0o700)
    return wrapper


def _problem_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "problems.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_seeded_engine_crash_is_persisted_typed_and_not_success(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_CHECKPOINT", "off")
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples" / "discount_bug", repo)
    script = serve(Script(json.loads((ROOT / "examples" / "discount_bug.script.json").read_text())))
    run_dir = tmp_path / "run"
    try:
        status = cli.main([
            "run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
            "--endpoint", f"http://127.0.0.1:{script.server_port}/v1", "--model", "scripted",
            "--run-dir", str(run_dir), "--python", str(_seeded_python(tmp_path)),
            "--test-python", sys.executable, "--max-turns", "4", "--quiet",
        ])
    finally:
        script.shutdown()
        script.server_close()

    # Truthful outcome: the CLI exits non-zero and summary.json never claims success.
    assert status != 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["succeeded"] is False
    assert summary["engine_result"] == f"ERR:{SEEDED_CLASS}"
    assert summary["runtime_outcome"] == "engine_error"
    assert SEEDED_CLASS in summary["engine_error_reason"]
    assert SEEDED_MESSAGE in summary["engine_error_reason"]

    # Engine journal: the catch-all's run_exception event carries the class and message.
    run_events = [event for event in measurement_events(run_dir) if event.get("kind") == "run_exception"]
    assert len(run_events) == 1, run_events
    assert run_events[0]["error_type"] == SEEDED_CLASS
    assert run_events[0]["error_detail"] == SEEDED_MESSAGE

    # problems.jsonl: exactly one fault row for the catch-all site, with the full record schema.
    rows = [row["fields"] for row in _problem_rows(run_dir)
            if isinstance(row.get("fields"), dict) and row["fields"].get("site_id") == HANDLER_ID]
    assert len(rows) == 1, _problem_rows(run_dir)
    fault = rows[0]
    assert fault["exception_class"] == SEEDED_CLASS
    assert fault["exception_message"] == SEEDED_MESSAGE
    assert fault["handler_id"] == HANDLER_ID
    assert fault["required_work_id"] == "engine_run"
    assert FAULT_ID_PATTERN.fullmatch(fault["fault_id"])
    assert fault["primary_fault_id"] == fault["fault_id"]
