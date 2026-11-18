"""MODEL-ROUTING-UNIFY MR-5 end to end: a withheld edit rides the real react_repair loop to the edit route.

The scripted endpoint answers request 1 with one batch [read, str_replace]. The answering route runs
the read, the edit is withheld, and the carried request goes to the edit model, where the edit is
applied exactly once. Schema 1 runs the same script and discards the whole batch, as it always has.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from _journal import measurement_events
from _timeouts import scaled
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, serve
from test_hook_checkpoint_resume import _launcher, _repo

ROOT = Path(__file__).resolve().parents[1]
ANSWERING = "reader"  # request 1 is planned on the plan role (no action has closed yet)
EDITING = "editor"
PRESERVED_HEAD = "A previous step proposed these changes against revision"

READ_ARGS = {"path": "pkg/value.py"}
EDIT_ARGS = {"path": "pkg/value.py", "old_str": "VALUE = 1", "new_str": "VALUE = 3"}


def _steps() -> list[dict]:
    # Step 1 is one batch: the read runs on the answering route, the edit is the escalating call.
    return [
        {"tool": "read", "required": True, "arguments": READ_ARGS,
         "also": [{"tool": "str_replace", "arguments": EDIT_ARGS}]},
        {"tool": "str_replace", "required": True, "arguments": EDIT_ARGS},
        {"tool": "run_tests", "required": True, "arguments": {"target": "tests/test_value.py"}},
        {"tool": "done", "required": True, "arguments": {"summary": "Checks passed after the edit."}},
    ]


def _profile(tmp_path: Path, schema: str) -> Path:
    # The engine child rebuilds its environment from the run profile, so the schema travels as a switch.
    profile = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
    profile["switches"]["STATEFRAME_ROUTING_SCHEMA"] = schema
    path = tmp_path / f"routing-schema-{schema}.json"
    path.write_text(json.dumps(profile))
    return path


def _run(tmp_path: Path, schema: str) -> tuple[Script, Path]:
    script = Script({"model": "scripted", "models": [ANSWERING, EDITING], "steps": _steps(),
                     "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / "run"
    command = [sys.executable, str(_launcher(tmp_path)), "run", "--repo", str(_repo(tmp_path)),
               "--task", "Inspect and update VALUE.", "--endpoint",
               f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
               "--route", f"plan.model={ANSWERING}", "--route", f"edit.model={EDITING}",
               "--profile", str(_profile(tmp_path, schema)),
               "--run-dir", str(run), "--test-python", sys.executable,
               "--max-turns", "8", "--shell", "off", "--quiet"]
    try:
        result = run_monitored(command, run_dir=run, step=f"mr5 e2e schema {schema}",
                               capture_output=True, text=True, expected_seconds=scaled(240))
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    script.assert_consumed()
    return script, run


def _committed_batches(run: Path) -> list[list[str]]:
    """Tool names of each batch the engine committed, in order."""
    received = [event for event in measurement_events(run) if event.get("kind") == "native_batch_received"]
    return [[call["function"]["name"] for call in event["calls"]] for event in received]


def _kind_count(run: Path, kind: str) -> int:
    return sum(1 for event in measurement_events(run) if event.get("kind") == kind)


def _carried_calls(body: dict) -> list[dict]:
    """The withheld calls as the carrying request presents them: the preserved user message, verbatim JSON."""
    carried = [str(m.get("content")) for m in body["messages"]
               if str(m.get("content", "")).startswith(PRESERVED_HEAD)]
    assert len(carried) == 1
    return json.loads(carried[0].split("\n", 1)[1])


@pytest.mark.host_sandbox
def test_v2_withheld_edit_runs_on_edit_route_exactly_once(tmp_path):
    script, run = _run(tmp_path, "2")
    first, second = script.requests[0], script.requests[1]

    # Request 1 is answered on the answering route. The read ran; the edit was not committed there.
    assert first["model"] == ANSWERING
    assert _committed_batches(run)[0] == ["read"]
    assert _kind_count(run, "route_proposal_preserved") == 1
    assert _kind_count(run, "route_escalation_discarded") == 0

    # Request 2 goes to the edit route and carries the withheld edit verbatim.
    assert second["model"] == EDITING
    carried = _carried_calls(second)
    assert [call["function"]["name"] for call in carried] == ["str_replace"]
    assert json.loads(carried[0]["function"]["arguments"]) == EDIT_ARGS

    # The edit is applied once: one committed str_replace batch, and the file shows the single change.
    applied = [names for names in _committed_batches(run) if "str_replace" in names]
    assert applied == [["str_replace"]]
    # The run edits its own workspace copy; the original repo stays untouched.
    assert (run / "workspace" / "pkg" / "value.py").read_text() == "VALUE = 3\n"
    assert (tmp_path / "repo" / "pkg" / "value.py").read_text() == "VALUE = 1\n"


@pytest.mark.host_sandbox
def test_v1_whole_batch_is_discarded_on_escalation(tmp_path):
    script, run = _run(tmp_path, "1")

    # Schema 1 discards the whole batch on the answering route and journals it, as today.
    assert _kind_count(run, "route_escalation_discarded") == 1
    assert _kind_count(run, "route_proposal_preserved") == 0
    assert not any("read" in names for names in _committed_batches(run))
    assert script.requests[1]["model"] == EDITING
