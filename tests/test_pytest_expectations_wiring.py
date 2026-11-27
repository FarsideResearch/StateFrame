"""Native pytest predictions are optional evidence on the existing check."""
from __future__ import annotations

import json
import lzma
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

from _journal import measurement_events
from _timeouts import scaled
from stateframe import runner
from fixtures.process_signals import signal_run_owned
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve

ROOT = Path(__file__).resolve().parents[1]
SWITCH = "STATEFRAME_PYTEST_EXPECTATIONS"
NODE = "tests/test_pricing.py::test_quarter_off"
EVENT = "pytest_expectation_mismatch"


def test_switch_is_validated_and_off_is_absent(monkeypatch):
    for value in ("off", "on", "maybe"):
        monkeypatch.setenv(SWITCH, value)
        if value == "maybe":
            with pytest.raises(runner.RunError, match=SWITCH):
                runner.resolve_opt_in_switches()
        else:
            options = runner.RunOptions(repo=ROOT, task="Check pytest expectations", endpoint=None)
            assert runner._provider_environment(options).get(SWITCH) == (value if value == "on" else None)


class PauseBeforeSubmission(Script):
    """Leave the fourth request in flight without consuming its scripted reply."""

    def __init__(self, document):
        super().__init__(document)
        self.arrived = threading.Event()
        self.release = threading.Event()

    def pause(self):
        if self.position == 3 and not self.release.is_set():
            self.arrived.set()
            if not self.release.wait(scaled(120)):
                raise TimeoutError("test did not release the submission")
            raise ScriptStepUnavailable("interrupted before the submission reply")

    def next_call(self, offered):
        self.pause()
        return super().next_call(offered)

    def next_forced_call(self, offered, tool_name):
        self.pause()
        return super().next_forced_call(offered, tool_name)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("resume", [False, True], ids=["uninterrupted", "resume"])
def test_scripted_native_pass_then_submit(tmp_path, resume):
    document = json.loads((ROOT / "examples/discount_bug.script.json").read_text())
    document["steps"][2]["arguments"]["prediction"] = {"nodeid": NODE, "outcome": "failed"}
    document["steps"][2]["arguments"]["target"] = (
        "test_no_discount_keeps_price test_quarter_off")
    script = PauseBeforeSubmission(document) if resume else Script(document)
    server = serve(script)
    run, repo = tmp_path / "run", tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), SWITCH: "on"}
    cli = [sys.executable, "-m", "stateframe"]
    command = [*cli, "run", "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
               "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run),
               "--test-python", sys.executable, "--no-run-env", "--max-turns", "10", "--quiet"]
    process = None
    try:
        if resume:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + scaled(240)
            while not script.arrived.wait(0.2):
                assert process.poll() is None, process.communicate()
                assert time.monotonic() < deadline, "submission request did not arrive"
            signal_run_owned(run, process, signal.SIGTERM, deadline=time.monotonic() + scaled(120),
                             what="pytest expectation resume")
            process.communicate(timeout=scaled(30))
            script.release.set()
            command = [*cli, "resume", str(run), "--endpoint", endpoint, "--quiet"]
            env.pop(SWITCH)  # Resume must use the recorded run setting.
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=scaled(240))
        log = run / "engine.log"
        assert result.returncode == 0, (result.stdout, result.stderr, log.read_text()[-8000:] if log.exists() else "")
        script.assert_consumed()
        assert len(script.requests) == (5 if resume else 4)
        if resume:
            assert script.requests[3] == script.requests[4]
            assert not (run / "resume-divergence.json").exists()
        journal = measurement_events(run)
        events = [e for e in journal if e["kind"] == EVENT]
        assert len(events) == 1, [str(m)[:1600] for m in script.requests[-1]["messages"][-2:]]
        mismatch = {k: events[0][k] for k in ("execution_id", "nodeid", "expected", "observed")}
        assert (mismatch["nodeid"], mismatch["expected"], mismatch["observed"]) == (NODE, "failed", "passed")
        assert any(e["kind"] == "check_receipt" and e.get("execution_id") == mismatch["execution_id"]
                   and e.get("status") == "PASS" for e in journal)
        assert "[pytest expectation mismatch]" in json.dumps(script.requests[3]["messages"])
        summary = json.loads((run / "summary.json").read_text())
        assert summary["succeeded"] and summary["pytest_expectation_mismatches"] == [mismatch]
    finally:
        if resume:
            script.release.set()
        if process is not None and process.poll() is None:
            signal_run_owned(run, process, signal.SIGTERM, deadline=time.monotonic(), what="test cleanup",
                             require_checkpoint_saved=False)
            process.communicate(timeout=scaled(30))
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(("report_bytes", "code"), [
    (None, 0),
    (b'{"incomplete_reason":"tests_unfinished"}', 1),
    (b'{"exit_code":0,"outcomes":{"test_a.py::test_value[other]":"passed"}}', 0),
    (b'{"exit_code":0,"outcomes":{"test_a.py::test_value[x]":"passed"}}', 1),
    (b'{"exit_code":0,"outcomes":{"test_a.py::test_value[x]":"passed"}}', 0),
    (b'{"exit_code":1,"outcomes":{"test_a.py::test_value[x]":"failed"}}', 1),
])
def test_only_a_complete_matching_report_can_disagree(tmp_path, report_bytes, code):
    from kernel_pytest_expectations import Prediction, run_reported

    prediction = Prediction("test_a.py::test_value[x]", "failed")

    def run(command):
        path, = (tmp_path / ".test-tmp").glob(".stateframe-pytest-*")
        if report_bytes is None:
            path.unlink()
        else:
            path.write_bytes(report_bytes)
        return subprocess.CompletedProcess(command, code, "1 passed", "")

    _, report = run_reported([sys.executable, "-m", "pytest"], root=tmp_path, env={},
                             execution_id="execution-a", profiled=False, run=run)
    if report is not None:
        prediction.observe("execution-a", report)
    if report is not None and report.get("complete") is True and prediction.nodeid in report["outcomes"]:
        assert prediction.execution_id == "execution-a"
        actual = report["outcomes"][prediction.nodeid]
        expected_mismatch = (None if actual == prediction.expected else {
            "execution_id": "execution-a", "nodeid": prediction.nodeid,
            "expected": prediction.expected, "observed": actual,
        })
        assert prediction.mismatch == expected_mismatch
    else:
        assert prediction.execution_id is None
        prediction.observe("execution-b", {
            "complete": True, "outcomes": {prediction.nodeid: "passed"}})
        assert prediction.execution_id == "execution-b"
        assert prediction.mismatch == {
            "execution_id": "execution-b", "nodeid": prediction.nodeid,
            "expected": "failed", "observed": "passed",
        }
    assert not list((tmp_path / ".test-tmp").iterdir())


def test_incomplete_report_cannot_change_an_already_bound_prediction():
    from kernel_pytest_expectations import Prediction

    prediction = Prediction("test_a.py::test_value[x]", "failed")
    prediction.observe("execution-a", {
        "complete": True, "outcomes": {prediction.nodeid: "failed"}})
    prediction.observe("execution-a", {
        "complete": False, "outcomes": {prediction.nodeid: "passed"}})
    assert prediction.execution_id == "execution-a"
    assert prediction.mismatch is None


def test_off_native_requests_and_journal_match_0102(tmp_path, monkeypatch, isolated_pilot_context):
    """Pin the clock, UUID and subprocess observation before serialization, then compare raw bytes."""
    import importlib.util
    import io
    import uuid
    from types import SimpleNamespace

    import kernel_execution_policy as policy
    import kernel_redirect_policy
    from kernel_measurement import AuditJournal
    from kernel_pilot import ACTIVE_RUNTIME
    from probes import react_repair

    # Load only the released probe, from the exact 0.10.2 Git object. Keep its
    # module directory and package imports identical to the installed probe.
    source = subprocess.check_output(["git", "show",
        "6e3b096cf3059f759138a88f7fdf46acd3d75baa:src/stateframe/engine/probes/react_repair.py"], cwd=ROOT)
    spec = importlib.util.spec_from_file_location("probes.expect_baseline", react_repair.__file__)
    baseline = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, baseline)
    exec(compile(source, spec.origin, "exec"), baseline.__dict__)
    monkeypatch.setattr(time, "time_ns", lambda: 1700000000000000000)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=42))
    monkeypatch.setattr(policy, "run_sandboxed", lambda argv, *_args, **_kwargs: SimpleNamespace(
        completed=subprocess.CompletedProcess(argv, 0, "1 passed in 0.01s\n", ""), timed_out=False))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n")
    (repo / "test_a.py").write_text("def test_a(): pass\n")
    from stateframe.endpoint import Endpoint
    monkeypatch.setenv(SWITCH, "off")
    prepared_run, _, _ = runner.prepare(runner.RunOptions(
        repo=repo, task="Inspect app.py", target="app.py",
        endpoint=Endpoint("http://127.0.0.1:8080/v1", "scripted"),
        run_dir=tmp_path / "prepared-run", instructions="off"), printer=lambda *_args: None)
    child_environment = json.loads((prepared_run / "spec.json").read_text())["environment"]
    baseline_environment = {}  # 0.10.2 did not put this switch in its child spec.
    current_environment = ({SWITCH: child_environment[SWITCH]}
                           if SWITCH in child_environment else {})

    def capture(module, label, child_spec_environment):
        with monkeypatch.context() as env:
            # Use the generated spec's switch value without re-normalizing it here.
            env.delenv(SWITCH, raising=False)
            if SWITCH in child_spec_environment:
                env.setenv(SWITCH, child_spec_environment[SWITCH])
            journal = AuditJournal(tmp_path / label, "same-task", {})
            runtime = SimpleNamespace(journal=journal, binding=lambda: {},
                                      test_observation=lambda *args, **kwargs: None)
            requests = []

            def provider(request, **kwargs):
                requests.append(request.data)
                return io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}]}')

            env.setattr(module, "API", "http://127.0.0.1:1/v1/chat/completions")
            env.setattr(kernel_redirect_policy, "provider_urlopen", provider)
            env.setattr(module, "_effect_boundary_enabled", lambda: False)
            messages = [{"role": "user", "content": "Check test_a.py."}]
            module.ask_tools(messages)
            token = ACTIVE_RUNTIME.set(runtime)
            try:
                observation = module.run_tests(repo, "", "test_a.py", row={"runner_kind": "pytest"})
            finally:
                ACTIVE_RUNTIME.reset(token)
            messages.append({"role": "user", "content": observation})
            module.ask_tools(messages)
            assert len(requests) == 2 and b"ALL TESTS PASSED" in requests[1]
            journal_data = journal.path.read_bytes()
            assert b'"kind":"check_receipt"' in journal_data
            return requests, journal_data

    assert capture(baseline, "released", baseline_environment) == capture(
        react_repair, "explicit", current_environment)
    assert SWITCH not in child_environment


@pytest.mark.parametrize("profiled", [False, True])
def test_shared_pytest_driver_observes_one_session(tmp_path, profiled):
    from kernel_pytest_expectations import run_reported
    from kernel_test_run_spans import profiled_pytest_command

    (tmp_path / "test_example.py").write_text(
        "from pathlib import Path\n"
        "def test_example():\n"
        "    with Path('calls').open('a') as f: f.write('once\\n')\n")
    argv = [sys.executable, "-I", "-m", "pytest", "-q", "--", "test_example.py"]
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    if profiled:
        argv = profiled_pytest_command(argv, "check-42")
    result, report = run_reported(argv, root=tmp_path, env=env, execution_id="check-42",
                                  profiled=profiled, run=lambda cmd: subprocess.run(
                                      cmd, cwd=tmp_path, env=env, capture_output=True, text=True,
                                      timeout=scaled(60)))
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls").read_text() == "once\n"
    assert report == {"complete": True, "outcomes": {"test_example.py::test_example": "passed"}}
    assert not list((tmp_path / ".test-tmp").glob(".stateframe-pytest-*"))


def test_summary_projects_only_current_run_evidence_from_sealed_journal(tmp_path):
    mismatch = {"execution_id": "check-42", "nodeid": NODE, "expected": "failed", "observed": "passed"}
    journal_dir = tmp_path / "worker/measurement-runs/current"
    journal_dir.mkdir(parents=True)
    rows = [{"kind": EVENT, **mismatch}, {"kind": "provider_queue_wait", "seconds_waited": 1.25}]
    contents = "".join(json.dumps(row) + "\n" for row in rows)
    (journal_dir / "events.jsonl.xz").write_bytes(lzma.compress(contents.encode()))
    archived = tmp_path / "attempts/1/worker/measurement-runs/previous"
    archived.mkdir(parents=True)
    (archived / "events.jsonl").write_text(contents)

    current = runner._journal_advisories(tmp_path, pytest_expectations=True)
    assert current["pytest_expectation_mismatches"] == [mismatch]
    assert runner._journal_advisories(tmp_path) == {
        "provider_queue_wait_warning": current["provider_queue_wait_warning"]}
