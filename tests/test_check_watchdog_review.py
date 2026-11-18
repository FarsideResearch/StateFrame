"""Regression coverage for watchdog results, supervision, and replay boundaries."""
from __future__ import annotations

import json
import subprocess
import sys
import zlib
from types import SimpleNamespace

import pytest

import kernel_check_watchdog as watchdog
import kernel_execution_policy as policy


@pytest.fixture
def stop():
    return watchdog.CheckWatchdog(watchdog.CheckWatchdogConfig(2)).stop_payload(cleanup_ok=True)


@pytest.mark.parametrize("stopped_first", [False, True])
def test_terminal_receipt_wins_in_both_orders(stop, stopped_first):
    from kernel_check_receipts import record_check_output

    runtime = SimpleNamespace(test_receipts=[{"execution_id": "one", "outcome": "PASS", "status": "PASS"}])
    passed = dict(status="PASS", exit_code=0, counts={"passed": 2}, collected_tests=2,
                  assertion_reached=True, product_observed=True)
    stalled = dict(status="UNKNOWN", exit_code=-9, typed_observation={"execution_stop": stop})
    first, second = (stalled, passed) if stopped_first else (passed, stalled)
    for fields in (first, second):
        public = record_check_output(runtime, execution_id="one", selector="tests", **fields)
    private = next(iter(runtime._check_output_receipts.values()))
    for receipt in (public, runtime.host_check_receipts[0], private, runtime.test_receipts[0]):
        assert receipt["execution_complete"] is False
        assert receipt["termination_reason"] == "NO_PROGRESS"
        assert receipt["exit_code"] == -9
        assert receipt["reached_observation"] is False
        assert not receipt["counts"].get("passed")
        assert not receipt.get("assertion_reached")
        assert not receipt.get("product_observed")
    assert private["execution_stop"] == stop


def test_interrupted_bound_verification_is_unsuccessful_without_a_test_failure(stop):
    from stateframe import outcomes

    summary = {"engine_terminal": {"result": "SUBMITTED"}, "child_exit_code": 0,
               "candidate_sha256": "digest", "tests_after": {"candidate_sha256": "digest",
               "passed": False, "execution_stop": stop, "execution_complete": False}}
    summary.update(outcomes.project(summary))
    assert summary["check_outcome"] == "no_progress"
    assert outcomes.exit_code(summary) != 0


@pytest.mark.parametrize("route", ["witness", "authored_parent", "authored_candidate", "unittest"])
@pytest.mark.parametrize("stopped", [False, True])
def test_authored_adapters_supervise_before_interpreting_output(tmp_path, monkeypatch, route, stopped, stop):
    import kernel_agent_authored_check as agent
    import kernel_authored_check as authored
    import kernel_authored_witness as witness

    calls = []

    def sandbox(command, root, **settings):
        calls.append(settings)
        return SimpleNamespace(
            completed=subprocess.CompletedProcess(command, 0, "MARK\nRan 1 test\n", ""),
            timed_out=False, events=[{"kind": "target_entry"}, {"kind": "assertion"}],
            execution_stop=stop if stopped else None)

    monkeypatch.setenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", "41")
    monkeypatch.setattr(policy, "run_sandboxed", sandbox)
    monkeypatch.setattr(agent, "run_sandboxed", sandbox)
    snapshot = {"revision": "revision", "files": {}}
    if route == "witness":
        monkeypatch.setattr(witness, "build_parent_tree", lambda *a, **kw: None)
        result = witness.replay_on_parent(SimpleNamespace(), root=tmp_path, command=["pytest"],
                                          environment=None, test_path="test_a.py", parent={}, current={})
        if stopped:
            assert witness.classify_parent(result, ("MARK",)) is None
    elif route == "unittest":
        check = {"path": "test_a.py", "selector": "test_a.py", "source": b""}
        result = authored.PythonUnittestAdapter(runner=sandbox).run(
            check, snapshot, revision="revision", side="parent")
    else:
        result = agent._run_side(snapshot, "test_a.py", b"", "check", "MARK",
                                 route.removeprefix("authored_"), tmp_path)
    settings, = calls
    assert settings["timeout"] is None
    assert settings["cpu_seconds"] == 41
    assert settings["check_watchdog"].stall_seconds > 0
    if stopped:
        assert result["execution_stop"] == stop
        assert result.get("execution_complete") is False
        assert result.get("status", result.get("outcome")) != "PASS"
    else:
        assert result.get("status", result.get("outcome")) == "PASS"


def test_progress_diagnostics_never_enter_the_checkpoint_tail(stop):
    import kernel_pilot
    from kernel_continuation_restart import resume_tail_rejection_reason

    tail = []
    journal = SimpleNamespace(emit=lambda kind, **fields: tail.append({"kind": kind, **fields}))
    token = kernel_pilot.ACTIVE_RUNTIME.set(SimpleNamespace(journal=journal))
    try:
        watch = watchdog.CheckWatchdog(watchdog.CheckWatchdogConfig(2))
        for step in range(20):
            watch.observe(step / 2, watchdog.ProgressSnapshot({(42, (1, 0)): step}))
        assert watch.progress_sequence > 4
        watch.stopped(cleanup_ok=True)
    finally:
        kernel_pilot.ACTIVE_RUNTIME.reset(token)
    assert resume_tail_rejection_reason(tail) is None
    assert tail == []


def test_check_recorder_preserves_stop_and_detects_equal_signal_with_different_cause(tmp_path, monkeypatch, stop):
    from kernel_pilot import ACTIVE_RUNTIME
    from probes.react_repair import _effect_test_subprocess
    from stateframe import _child
    from stateframe.replay_tape import Replayer

    prior, faithful, retry = (tmp_path / name for name in ("prior", "faithful", "retry"))
    for run in (prior, faithful, retry):
        run.mkdir()
    terminations = iter((stop, {**stop, "progress_sequence": 12}, None))

    def execute(command, root, **kwargs):
        terminal = next(terminations)
        return policy.PolicyResult(subprocess.CompletedProcess(command, -9, "", ""), False,
                                   terminal is not None, False, policy.capability_report(),
                                   execution_stop=terminal)

    def run_check(run, replay=None):
        monkeypatch.setattr(policy, "run_sandboxed", execute)
        _child.install_check_recorder(run, replay, prior / "checks" if replay is not None else None)
        result, shadow = _effect_test_subprocess(
            ("check",), root=tmp_path, targets=("tests",), env={},
            runner=lambda command: policy.run_sandboxed(command, tmp_path))
        assert shadow is None
        return result

    def recorded(run):
        return json.loads(zlib.decompress((run / "checks/0001.json.z").read_bytes()))

    events = []
    monkeypatch.setenv("STATEFRAME_EFFECT_BOUNDARY", "enforce")
    monkeypatch.setattr(_child, "HOST_STATE", {})
    monkeypatch.setattr(_child, "emit", lambda kind, **fields: events.append((kind, fields)))
    runtime = SimpleNamespace(current_request_id="test-request")
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        assert run_check(prior).execution_stop == stop
        assert run_check(faithful, Replayer([{"send": 1}])).execution_stop == stop
        replayer = Replayer([{"send": 1}])
        assert run_check(retry, replayer).execution_stop is None
    finally:
        ACTIVE_RUNTIME.reset(token)

    assert {entry.status for entry in runtime._effect_application_receipts} == {"APPLIED"}
    assert len(runtime._effect_application_receipts) == 3
    assert recorded(prior)["execution_stop"] == stop
    assert recorded(faithful) == recorded(prior)
    assert recorded(retry) == {"check": 1, "argv": ["check"], "returncode": -9, "stdout": "", "stderr": ""}
    assert any(kind == "check_replay_diverged" for kind, _ in events)
    assert replayer.answer(b"same request") is None
    assert replayer.diverged["check"] == 1


@pytest.mark.parametrize("code, terminal", [(124, True), (-9, True), (137, False)])
def test_generic_completion_and_receipt_use_the_same_termination_evidence(stop, code, terminal):
    import kernel_generic_check as generic
    from kernel_check_receipts import record_check_output
    from kernel_kill_attribution import receipt_fields

    class Runtime:
        def test_observation(self, selectors, code, binding, **kwargs):
            self.fields = receipt_fields(code, memory_limit_killed=kwargs["memory_limit_killed"],
                                         evidence=kwargs.get("termination_evidence"))
            self.receipt = record_check_output(
                self, execution_id="run", selector=selectors[0], exit_code=code, status="UNKNOWN",
                typed_observation=kwargs["typed_observation"])

    runtime = Runtime()
    generic._record(runtime, {}, ("check",), code, "", outcome="RESOURCE_LIMIT",
                    memory_limit_killed=not terminal, execution_stop=stop if terminal else None)
    assert runtime.fields["likely_initiator"] == ("no_progress" if terminal else "aggregate_rss_supervisor")
    assert runtime.receipt["outcome"] == ("NO_PROGRESS" if terminal else "RESOURCE_LIMIT")


@pytest.mark.parametrize("settings", [{}, {"STATEFRAME_CHECK_STALL_SECONDS": "2",
                                           "STATEFRAME_PUBLIC_CHECK_CPU_SECONDS": "41"}])
def test_operator_check_settings_are_forwarded(monkeypatch, settings):
    from stateframe import runner

    for name in ("STATEFRAME_CHECK_STALL_SECONDS", "STATEFRAME_PUBLIC_CHECK_CPU_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    result = runner.resolve_opt_in_switches()
    assert {name: result[name] for name in result if name in settings} == settings


@pytest.mark.parametrize("missing", [False, True])
def test_launch_sampler_recovers_or_refuses_persistent_telemetry_gaps(monkeypatch, missing):
    import kernel_exec_tracker as tracker

    root = tracker.ProcessIdentity(42, 1, 42, 42, (10, 0), False)
    unrelated = tracker.ProcessIdentity(43, 1, 43, 43, (99, 0), False)
    process = SimpleNamespace(pid=42, returncode=None)
    table = SimpleNamespace(identity=lambda pid: (None if missing else root) if pid == 42 else unrelated,
                            refresh_members=lambda members: {42: root}, cpu_ticks=lambda row: 10)
    monkeypatch.setattr(watchdog, "ProcessTable", lambda: table)
    monkeypatch.setattr(watchdog, "recorded_launch_identity", lambda process:
                        tracker.LaunchIdentity(42, None if missing else root.start, None))
    sampler = watchdog.LaunchProgressSampler(process)
    if not missing:
        sampler.previous = watchdog.ProgressSnapshot({(42, root.start): 10, (43, (11, 0)): 1})
    watch = watchdog.CheckWatchdog(watchdog.CheckWatchdogConfig(2))
    for turn in range(80):
        if watch.observe(turn / 2, sampler.sample()):
            break
    else:
        pytest.fail("the check remained unsupervised")
    assert watch.stop_cause == ("telemetry_unavailable" if missing else "no_progress")


def test_waiting_for_shared_permit_can_outlast_the_old_admission_window(tmp_path, monkeypatch):
    import kernel_check_permits as permits

    clock = [0.0]
    waiting = []
    first = permits.acquire(tmp_path, 1, timeout=0)
    first.__enter__()

    def sleep(seconds):
        waiting.append(seconds)
        clock[0] += 100
        if clock[0] == 600:
            first.__exit__(None, None, None)

    monkeypatch.setattr(permits.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(permits.time, "sleep", sleep)
    try:
        with permits.acquire(tmp_path, 1, timeout=None) as number:
            assert number == 0 and clock[0] == 600
    finally:
        first.__exit__(None, None, None)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("stall,cpu", [("1.5", "40"), ("nan", "40"), ("1.5", "-1")])
def test_cli_reports_stalled_checks_without_selecting_partial_failures(tmp_path, monkeypatch, capsys, stall, cpu):
    from stateframe import cli, scripted_endpoint

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "value.py").write_text("VALUE = 1\n")
    (repo / "tests/test_idle.py").write_text(
        "import time\n"
        "def test_first():\n"
        "    print('FAILED tests/test_idle.py::test_first - AssertionError', flush=True)\n"
        "    assert False\n"
        "def test_idle():\n"
        "    time.sleep(30)\n"
    )
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STATEFRAME_HOME", str(home / "stateframe"))
    monkeypatch.setenv("STATEFRAME_CHECK_STALL_SECONDS", stall)
    monkeypatch.setenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", cpu)
    run_dir = tmp_path / "run"
    document = {"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "required": True, "arguments": {"path": "value.py"}},
        {"tool": "run_tests", "required": True, "arguments": {"target": "tests/test_idle.py::test_idle"}},
        {"tool": "done", "required": True, "arguments": {"summary": "Read the value."}},
    ]}
    script = scripted_endpoint.Script(document)
    invalid = stall == "nan" or cpu == "-1"
    server = scripted_endpoint.serve(script)
    try:
        code = cli.main(["run", "--repo", str(repo), "--task", "Read the value.",
                         "--run-dir", str(run_dir), "--model", "scripted", "--endpoint",
                         f"http://127.0.0.1:{server.server_port}/v1",
                         "--test-cmd", "python -m pytest -q -s tests/test_idle.py",
                         "--test-python", sys.executable, "--protect-tests", "task",
                         "--no-run-env", *(["--json"] if invalid else [])])
    finally:
        server.shutdown()
        server.server_close()
    output = capsys.readouterr()
    if invalid:
        result = json.loads(output.out)
        assert code == 2
        assert ("STATEFRAME_CHECK_STALL_SECONDS" if stall == "nan"
                else "STATEFRAME_PUBLIC_CHECK_CPU_SECONDS") in result["error"]
        return
    result = json.loads((run_dir / "summary.json").read_text())
    assert code == 1, result
    assert result["check_outcome"] == "no_progress", result
    for check in (result["tests_before"], result["tests_after"]):
        assert check["execution_stop"]["cause"] == "no_progress"
        assert check["execution_complete"] is False
        assert check["outcome"] == "NO_PROGRESS"
    assert "NO PROGRESS" in output.out
    assert " -> FAIL" not in output.out
    assert "FAILED tests/test_idle.py::test_first" in (run_dir / "tests-after.txt").read_text()
    spec = json.loads((run_dir / "spec.json").read_text())
    assert not spec["public_row"]["test_wrong_failed_nodes"]
    assert spec["environment"]["STATEFRAME_CHECK_STALL_SECONDS"] == stall
    assert spec["environment"]["STATEFRAME_PUBLIC_CHECK_CPU_SECONDS"] == cpu
    from _journal import measurement_events
    events = measurement_events(run_dir)
    stopped = [row for row in events if row["kind"] == "test_subprocess_completed"
               and row.get("likely_initiator") == "no_progress"]
    assert stopped and stopped[-1]["outcome"] == "NO_PROGRESS"
    event = stopped[-1]
    canonical = [row for row in events if row["kind"] == "check_receipt"
                 and row.get("execution_id") in event["canonical_check_execution_ids"]]
    assert canonical and canonical[-1]["outcome"] == "NO_PROGRESS"
    assert canonical[-1]["execution_stop"] == event["execution_stop"]
    assert canonical[-1]["exit_code"] == event["exit_code"]


def test_permit_wait_ends_for_explicit_budget_and_releases_on_cancellation(tmp_path, monkeypatch):
    import kernel_check_permits as permits

    clock = [0.0]
    monkeypatch.setattr(permits.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(permits.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with permits.acquire(tmp_path, 1):
        with pytest.raises(TimeoutError, match="run deadline"):
            with permits.acquire(tmp_path, 1, run_remaining=lambda: .12 - clock[0]):
                pytest.fail("expired worker acquired a permit")
    with pytest.raises(KeyboardInterrupt):
        with permits.acquire(tmp_path, 1):
            raise KeyboardInterrupt
    with permits.acquire(tmp_path, 1, timeout=0):
        pass


def test_unstamped_launch_recovers_only_while_its_child_is_pinned(monkeypatch):
    import threading
    import kernel_exec_tracker as tracker

    root = tracker.ProcessIdentity(42, 1, 42, 42, (10, 0), False)
    visible = False

    def identity(pid):
        return root if visible and pid == root.pid else None

    table = SimpleNamespace(identity=identity, refresh_members=lambda members: {42: root},
                            cpu_ticks=lambda row: 10)
    monkeypatch.setattr(watchdog, "ProcessTable", lambda: table)
    monkeypatch.setattr(watchdog, "recorded_launch_identity",
                        lambda process: tracker.LaunchIdentity(42, None, None))
    process = SimpleNamespace(pid=42, returncode=None, _waitpid_lock=threading.Lock())
    sampler = watchdog.LaunchProgressSampler(process)
    assert sampler.sample() is None
    visible = True
    assert sampler.sample() == watchdog.ProgressSnapshot({(42, root.start): 10})
