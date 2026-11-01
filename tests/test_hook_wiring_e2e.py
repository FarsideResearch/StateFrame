"""End-to-end child hook boundaries; these need a host OS sandbox and a socket."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("")
    (repo / "pkg/value.py").write_text("VALUE = 1\n")
    (repo / "tests/test_value.py").write_text(
        "from pkg.value import VALUE\ndef test_value():\n    assert VALUE in (1, 3)\n"
    )
    return repo


def _hook_file(tmp_path: Path, hooks: list[dict]) -> Path:
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": hooks}))
    return config


def _run(tmp_path: Path, steps: list[dict], hooks: list[dict] | None):
    repo = _repo(tmp_path)
    config = _hook_file(tmp_path, hooks) if hooks is not None else None
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / "run"
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
    }
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "stateframe",
                "run",
                "--repo",
                str(repo),
                "--task",
                "Inspect and update VALUE.",
                "--endpoint",
                f"http://127.0.0.1:{server.server_port}/v1",
                "--model",
                "scripted",
                "--run-dir",
                str(run),
                "--test-python",
                sys.executable,
                "--max-turns",
                "8",
                *(["--hooks", str(config)] if config is not None else []),
                "--quiet",
            ],
            env=environment,
            capture_output=True,
            text=True,
            timeout=240,
        )
        try:
            script.assert_consumed()
        except AssertionError as error:
            raise AssertionError(
                f"{error}\nCLI exit {result.returncode}: {result.stdout}{result.stderr}"
            ) from error
        records = (
            [json.loads(line) for line in (run / "hooks/records.jsonl").read_text().splitlines()]
            if config is not None else []
        )
        wire = [
            path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
        ]
        return result, run, records, wire
    finally:
        server.shutdown()
        server.server_close()


def _step(tool: str, **arguments: object) -> dict:
    return {"tool": tool, "required": True, "arguments": arguments}


def test_no_hooks_tool_call_run_tests_has_bound_route_arguments(tmp_path):
    result, run, records, wire = _run(
        tmp_path,
        [_step("run_tests", target="tests/test_value.py"),
         _step("done", summary="Checks passed.")],
        None,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not records
    assert "hooks" not in json.loads((run / "spec.json").read_text())
    assert any(b"pytest exit 0" in request for request in wire)


def test_pre_tool_deny_prevents_native_edit_and_post_tool_sees_denied(tmp_path):
    guard = {
        "name": "guard",
        "event": "pre_tool",
        "tools": ["str_replace"],
        "argv": [
            sys.executable,
            "-c",
            "import sys;sys.stderr.write('blocked');sys.exit(2)",
        ],
    }
    audit = {
        "name": "audit",
        "event": "post_tool",
        "tools": ["str_replace"],
        "argv": [sys.executable, "-c", "pass"],
    }
    result, run, records, wire = _run(
        tmp_path,
        [
            _step(
                "str_replace",
                path="pkg/value.py",
                old_str="VALUE = 1",
                new_str="VALUE = 2",
            ),
            _step("run_tests", target="tests/test_value.py"),
            _step("done", summary="The guard refused the edit and checks passed."),
        ],
        [guard, audit],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "workspace/pkg/value.py").read_text() == "VALUE = 1\n"
    assert [
        (record["event"], record["hook"])
        for record in records
        if record["event"] in {"pre_tool", "post_tool"}
    ] == [("pre_tool", "guard"), ("post_tool", "audit")]
    assert any(b"REFUSED HOOK_DENIED: guard: blocked" in request for request in wire)


def test_writer_receipt_and_note_follow_native_edit(tmp_path):
    writer = {
        "name": "fmt",
        "event": "post_tool",
        "tools": ["str_replace"],
        "changed": True,
        "writes": "changed",
        "argv": [
            sys.executable,
            "-c",
            "import pathlib,sys;p=pathlib.Path(sys.argv[1]);"
            "p.write_text(p.read_text().replace('VALUE = 2','VALUE = 3'))",
            "{paths}",
        ],
    }
    result, run, records, wire = _run(
        tmp_path,
        [
            _step(
                "str_replace",
                path="pkg/value.py",
                old_str="VALUE = 1",
                new_str="VALUE = 2",
            ),
            _step("run_tests", target="tests/test_value.py"),
            _step("done", summary="Formatted the edit and verified it."),
        ],
        [writer],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "workspace/pkg/value.py").read_text() == "VALUE = 3\n"
    assert any(
        record["event"] == "post_tool" and record["hook"] == "fmt" for record in records
    )
    assert any(b"fmt changed pkg/value.py" in request for request in wire)
    assert json.loads((run / "spec.json").read_text())["environment"]["STATEFRAME_F4"] == "on"
    panels = [
        message
        for request in wire
        for message in json.loads(request).get("messages", ())
        if str(message.get("content", "")).startswith("[USER HOOKS]")
    ]
    assert panels and all("_request_budget_section" not in panel for panel in panels)
    assert (run / "hooks/delivery.jsonl").is_file()


def test_stop_veto_recovers_once_then_blocks_without_seal(tmp_path):
    gate = {
        "name": "gate",
        "event": "stop",
        "max_blocks": 1,
        "argv": [
            sys.executable,
            "-c",
            "import sys;sys.stderr.write('run checks');sys.exit(2)",
        ],
    }
    result, run, records, wire = _run(
        tmp_path,
        [
            _step("str_replace", path="pkg/value.py", old_str="VALUE = 1",
                  new_str="VALUE = 3"),
            _step("run_tests", target="tests/test_value.py"),
            _step("done", summary="First submission."),
            _step("done", summary="Second submission."),
        ],
        [gate],
    )
    summary = json.loads((run / "summary.json").read_text())
    assert summary["engine_terminal"]["result"] == "HOOK_BLOCKED", summary
    assert [record["decision"] for record in records if record["event"] == "stop"] == [
        "deny",
        "deny",
    ]
    assert any(b"REFUSED HOOK_STOP: gate: run checks" in request for request in wire)
    assert not list((run / "worker/results/react_done_submissions").rglob("*.json"))
    assert result.returncode != 0


def test_six_event_lifecycle_survives_kill_and_resume(tmp_path):
    class InterruptAfterTwoActions(Script):
        def __init__(self, document):
            super().__init__(document)
            self.waiting = threading.Event()
            self.release = threading.Event()
            self.interrupted = threading.Event()

        def next_call(self, offered):
            if self.position == 2 and not self.interrupted.is_set():
                self.waiting.set()
                if not self.release.wait(60):
                    raise TimeoutError("interrupted request was not released")
                self.interrupted.set()
                raise ScriptStepUnavailable("request interrupted before a response")
            return super().next_call(offered)

    steps = [
        _step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 2"),
        _step("str_replace", path="pkg/value.py", old_str="VALUE = 1", new_str="VALUE = 4"),
        _step("run_tests", target="tests/test_value.py"),
        _step("done", summary="Checks passed after formatting."),
        _step("done", summary="Checks passed after formatting."),
    ]
    hooks = [
        {"name": "start", "event": "session_start", "argv": [
            sys.executable, "-c", "print('start marker')"]},
        {"name": "prompt", "event": "prompt_submit", "argv": [
            sys.executable, "-c", "print('prompt marker')"]},
        {"name": "guard", "event": "pre_tool", "tools": ["str_replace"],
         "argv": [sys.executable, "-c",
                  "import json,sys;d=json.load(sys.stdin)['data'];"
                  "deny=d['args'].get('new_str')=='VALUE = 2';"
                  "print(json.dumps({'decision':'deny' if deny else 'continue',"
                  "'reason':'blocked' if deny else ''}))"]},
        {"name": "audit", "event": "post_tool", "tools": ["str_replace"],
         "argv": [sys.executable, "-c", "print('post marker')"]},
        {"name": "fmt", "event": "post_tool", "tools": ["str_replace"],
         "changed": True, "writes": "changed", "argv": [
             sys.executable, "-c",
             "import pathlib,sys;p=pathlib.Path(sys.argv[1]);"
             "p.write_text(p.read_text().replace('VALUE = 4','VALUE = 3'))",
             "{paths}"]},
        {"name": "gate", "event": "stop", "max_blocks": 2,
         "argv": [sys.executable, "-c",
                  "import json,sys;d=json.load(sys.stdin)['data'];"
                  "deny=d['blocks_so_far']==0;"
                  "sys.stderr.write('run checks' if deny else '');"
                  "sys.exit(2 if deny else 0)"]},
        {"name": "end", "event": "session_end", "argv": [
            sys.executable, "-c", "print('end marker')"]},
    ]
    repo = _repo(tmp_path)
    config = _hook_file(tmp_path, hooks)
    script = InterruptAfterTwoActions({
        "model": "scripted", "steps": steps, "tokenize": {"mode": "ok"},
    })
    server = serve(script)
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(
        (str(ROOT / "src"), str(ROOT / "src/stateframe/engine")))}
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Inspect and update VALUE.", "--endpoint", endpoint,
        "--model", "scripted", "--run-dir", str(run), "--test-python",
        sys.executable, "--max-turns", "8", "--hooks", str(config), "--quiet",
    ]
    process = None
    try:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        assert script.waiting.wait(120), "first attempt did not reach the third request"
        prior_requests = {
            path.name: path.read_bytes()
            for path in (run / "wire").glob("*.request.json")
            if path.with_name(path.name.replace(".request.", ".response.")).exists()
        }
        assert len(prior_requests) == 2
        prior_tape = [json.loads(line) for line in (run / "hook-tape.jsonl").read_text().splitlines()]
        assert all(row["kind"] == "complete" for row in prior_tape[1::2])
        child_pids = subprocess.check_output(
            ["pgrep", "-P", str(process.pid)], text=True,
        ).split()
        assert len(child_pids) == 1, child_pids
        os.kill(int(child_pids[0]), signal.SIGKILL)
        script.release.set()
        assert script.interrupted.wait(10)
        process.communicate(timeout=90)
        resumed = subprocess.run(
            [sys.executable, "-m", "stateframe", "resume", str(run),
             "--endpoint", endpoint], env=environment, capture_output=True,
            text=True, timeout=240,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()
        for name, original in prior_requests.items():
            assert (run / "wire" / name).read_bytes() == original
        tape = [json.loads(line) for line in (run / "hook-tape.jsonl").read_text().splitlines()]
        old_ids = {row["record_id"] for row in prior_tape}
        assert [row for row in tape if row["record_id"] in old_ids] == prior_tape
        records = [json.loads(line) for line in (run / "hooks/records.jsonl").read_text().splitlines()]
        assert all(record["replayed"] for record in records
                   if record["event_id"] in old_ids)
        assert {record["event"] for record in records} >= {
            "pre_tool", "post_tool", "stop", "session_end",
        }
        first = json.loads((run / "wire/0001.request.json").read_text())
        assert any("start marker" in message.get("content", "")
                   and "prompt marker" in message.get("content", "")
                   for message in first["messages"])
        for path in (run / "wire").glob("*.request.json"):
            assert all("_request_budget_section" not in message
                       for message in json.loads(path.read_text())["messages"])
        wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        assert any(b"REFUSED HOOK_DENIED: guard: blocked" in wire for wire in wires)
        assert any(b"post marker" in wire for wire in wires)
        assert any(b"fmt changed pkg/value.py" in wire for wire in wires)
        assert any(b"REFUSED HOOK_STOP: gate: run checks" in wire for wire in wires)
        all_records = records + [json.loads(line) for line in
                                 (run / "attempts/1/hooks/records.jsonl").read_text().splitlines()]
        assert {record["event"] for record in all_records} == {
            "session_start", "prompt_submit", "pre_tool", "post_tool",
            "stop", "session_end",
        }
        assert any(record["event"] == "pre_tool" and record["hook"] == "guard"
                   and record["decision"] == "deny" for record in all_records)
        assert [record["decision"] for record in records if record["event"] == "stop"] == [
            "deny", "continue",
        ]
        assert any(row["kind"] == "complete" and
                   (row["result"].get("writer") or {}).get("authorized")
                   for row in tape)
        assert (run / "workspace/pkg/value.py").read_text() == "VALUE = 3\n"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        server.shutdown()
        server.server_close()
