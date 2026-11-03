"""Requirement recheck: the opt-in, once-per-run done refusal that lists the task's own requirements."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/stateframe/engine"))

import kernel_requirement_recheck as recheck  # noqa: E402

TASK = ("Add a deadline to the client. Reject zero, negative values and booleans before sending. "
        "Keep every existing call shape working.")


class _Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, kind: str, **payload: object) -> None:
        self.events.append((kind, payload))


class _Measurement:
    def __init__(self) -> None:
        self.journal = _Journal()


ON = {recheck.SWITCH: "on"}


def test_off_by_default_and_never_refuses():
    measurement = _Measurement()
    assert recheck.enabled({}) is False
    assert recheck.closing_refusal(measurement, {"msg": TASK}, 10, environ={}) is None
    assert recheck.closing_refusal(measurement, {"msg": TASK}, 10, environ={recheck.SWITCH: "off"}) is None
    assert not measurement.journal.events
    assert not hasattr(measurement, "_requirement_recheck_refusals")


def test_invalid_switch_value_is_an_error():
    with pytest.raises(ValueError):
        recheck.enabled({recheck.SWITCH: "yes"})


def test_refuses_once_with_the_task_sentences_numbered():
    measurement = _Measurement()
    first = recheck.closing_refusal(measurement, {"msg": TASK}, 10, environ=ON)
    assert first is not None
    assert first.startswith("REFUSED REQUIREMENT_RECHECK: before submitting, re-read the task")
    assert "1. Add a deadline to the client" in first
    assert "2. Reject zero, negative values and booleans before sending" in first
    assert "3. Keep every existing call shape working" in first
    assert measurement.journal.events == [("requirement_recheck_refused", {"clauses": 3, "dispatch": False})]
    assert recheck.closing_refusal(measurement, {"msg": TASK}, 9, environ=ON) is None


def test_never_refuses_on_the_last_steps():
    measurement = _Measurement()
    assert recheck.closing_refusal(measurement, {"msg": TASK}, 1, environ=ON) is None
    assert recheck.closing_refusal(measurement, {"msg": TASK}, 0, environ=ON) is None
    assert recheck.closing_refusal(measurement, {"msg": TASK}, None, environ=ON) is not None


def test_single_clause_tasks_are_not_refused():
    measurement = _Measurement()
    assert recheck.closing_refusal(measurement, {"msg": "Fix the failing test."}, 10, environ=ON) is None
    assert recheck.closing_refusal(measurement, {}, 10, environ=ON) is None
    assert not measurement.journal.events


def test_task_text_prefers_the_ordinary_brief():
    row = {"symptom": "symptom text here", "msg": "message text here", "_bridge_ordinary_brief": "brief text here"}
    assert recheck.task_text(row) == "brief text here"
    assert recheck.task_text({"msg": "  ", "brief": "brief only here"}) == "brief only here"


def test_checklist_is_bounded():
    long_task = " ".join(f"Requirement number {index} must hold." for index in range(40))
    items = recheck.checklist(long_task)
    assert len(items) == recheck.MAX_CLAUSES
    wide = recheck.checklist("word " * 200 + ". Second clause has words.")
    assert len(wide[0]) == recheck.MAX_CLAUSE_CHARS and wide[0].endswith("...")


# End to end through the CLI, the composed loop and done-is-submit (needs the host sandbox and a socket).

def _profile(tmp_path: Path, value: str | None) -> Path:
    document = json.loads((ROOT / "src/stateframe/profiles/r2hc3.json").read_text())
    if value is not None:
        document["switches"][recheck.SWITCH] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document))
    return path


def _run(tmp_path: Path, steps: list[dict], switch: str | None, max_turns: int = 8):
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("")
    (repo / "pkg/value.py").write_text("VALUE = 1\n")
    (repo / "tests/test_value.py").write_text("from pkg.value import VALUE\ndef test_value():\n    assert VALUE == 3\n")
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / "run"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Set VALUE to three in pkg/value.py. Keep the module importable. Make the tests pass.",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", str(max_turns),
             "--profile", str(_profile(tmp_path, switch)), "--quiet"],
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True, timeout=240)
        try:
            script.assert_consumed()
        except AssertionError as error:
            raise AssertionError(f"{error}\nCLI exit {result.returncode}: {result.stdout}{result.stderr}") from error
        wire = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        return result, run, wire
    finally:
        server.shutdown()
        server.server_close()


def _step(tool: str, **arguments: object) -> dict:
    return {"tool": tool, "required": True, "arguments": arguments}


_EDIT_AND_TEST = [_step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 3"),
                  _step("run_tests", target="tests/test_value.py")]


@pytest.mark.host_sandbox
def test_e2e_on_refuses_first_done_then_submits(tmp_path):
    result, run, wire = _run(tmp_path, [*_EDIT_AND_TEST, _step("done", summary="First."),
                                        _step("done", summary="Checked every requirement.")], "on")
    assert result.returncode == 0, result.stdout + result.stderr
    refusals = [request for request in wire if b"REFUSED REQUIREMENT_RECHECK" in request]
    assert len(refusals) == 1
    assert b"1. Set VALUE to three in pkg/value.py" in refusals[0]
    assert json.loads((run / "engine-result.json").read_text())["result"]["result"] == "SUBMITTED"
    assert json.loads((run / "spec.json").read_text())["environment"][recheck.SWITCH] == "on"


@pytest.mark.host_sandbox
def test_e2e_off_submits_on_first_done(tmp_path):
    result, run, wire = _run(tmp_path, [*_EDIT_AND_TEST, _step("done", summary="Done.")], None)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(b"REQUIREMENT_RECHECK" in request for request in wire)
    assert json.loads((run / "engine-result.json").read_text())["result"]["result"] == "SUBMITTED"


@pytest.mark.host_sandbox
def test_e2e_on_does_not_refuse_a_done_on_the_last_step(tmp_path):
    result, run, wire = _run(tmp_path, [*_EDIT_AND_TEST, _step("done", summary="Done.")], "on", max_turns=3)
    assert not any(b"REQUIREMENT_RECHECK" in request for request in wire)
    assert json.loads((run / "engine-result.json").read_text())["result"]["result"] == "SUBMITTED"


def test_shell_setting_reaches_the_engine_spec(monkeypatch):
    # The running guide says the switch can be set in the shell: the runner must forward it (it only forwards
    # allowlisted opt-in switches), and the shipped engine must declare it, or the shell setting is silently ignored.
    from stateframe import runner
    assert runner.OPT_IN_SWITCHES[recheck.SWITCH] == ("on", "off")
    assert recheck.SWITCH in runner.engine_switches()
    monkeypatch.setenv(recheck.SWITCH, "on")
    assert runner.resolve_opt_in_switches()[recheck.SWITCH] == "on"
    monkeypatch.setenv(recheck.SWITCH, "sometimes")
    with pytest.raises(runner.RunError):
        runner.resolve_opt_in_switches()
    monkeypatch.delenv(recheck.SWITCH)
    assert recheck.SWITCH not in runner.resolve_opt_in_switches()
