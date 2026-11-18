"""Check liveness is renewed by observed work, never by polling or lifetime."""
from __future__ import annotations

import ast
import io
import json
import resource
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_execution_policy as policy

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/stateframe/engine"


def watchdog_module():
    import kernel_check_watchdog
    return kernel_check_watchdog


def snapshot(ticks=0, *, child=False, start=10):
    mod = watchdog_module()
    rows = {(42, (start, 0)): ticks}
    if child:
        rows[(43, (11, 0))] = ticks
    return mod.ProgressSnapshot(rows)


@pytest.mark.parametrize("kind", ["cpu", "output", "completed_work"])
def test_work_renews_past_all_old_lifetimes(kind):
    mod = watchdog_module()
    watch = mod.CheckWatchdog(mod.CheckWatchdogConfig(300))
    for tick in range(8002):
        assert not watch.observe(tick / 2, snapshot(tick if kind == "cpu" else 0),
                                 output_bytes=tick if kind == "output" else 0,
                                 completed_work=tick if kind == "completed_work" else 0)
    assert watch.progress_sequence > 0


def test_idle_window_stops_once_and_boundary_progress_wins():
    mod = watchdog_module()
    events = []
    watch = mod.CheckWatchdog(mod.CheckWatchdogConfig(2, on_event=lambda *row: events.append(row)))
    for now in (0, .5, 1, 1.5):
        assert not watch.observe(now, snapshot())
    assert not watch.observe(2, snapshot(1))
    for now in (2.5, 3, 3.5):
        assert not watch.observe(now, snapshot(1))
    assert watch.observe(4, snapshot(1))
    stop = watch.stop_payload(cleanup_ok=True)
    assert stop["cause"] == "no_progress"
    assert stop["resource"] is None
    assert stop["inactivity_seconds"] == 2
    watch.stopped(cleanup_ok=True)
    watch.stopped(cleanup_ok=True)
    assert [name for name, _ in events].count("execution_stopped") == 1


@pytest.mark.parametrize("gap", ["missing", "pid_reuse", "counter_reset", "delayed"])
def test_unknown_or_discontinuous_samples_restart_full_window(gap):
    mod = watchdog_module()
    watch = mod.CheckWatchdog(mod.CheckWatchdogConfig(2))
    assert not watch.observe(0, snapshot(8))
    assert not watch.observe(.5, snapshot(8))
    value = None if gap == "missing" else snapshot(0 if gap == "counter_reset" else 8,
                                                 start=99 if gap == "pid_reuse" else 10)
    now = 20 if gap == "delayed" else 1
    assert not watch.observe(now, value)
    # A fresh valid baseline starts only after the gap.
    baseline = now + .5
    assert not watch.observe(baseline, snapshot(8))
    for offset in (.5, 1, 1.5):
        assert not watch.observe(baseline + offset, snapshot(8))
    assert watch.observe(baseline + 2, snapshot(8))


def test_exiting_child_cannot_erase_accumulated_cpu():
    mod = watchdog_module()
    watch = mod.CheckWatchdog(mod.CheckWatchdogConfig(2))
    watch.observe(0, snapshot(40, child=True))
    watch.observe(.5, snapshot(41, child=True))
    before = watch.cpu_ticks
    assert not watch.observe(1, snapshot(42))
    assert watch.cpu_ticks >= before
    assert watch.last_progress_kind in {"cpu", "members"}


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad", ""])
def test_bad_host_stall_window_is_refused(value):
    mod = watchdog_module()
    with pytest.raises(ValueError):
        mod.check_watchdog_config({"STATEFRAME_CHECK_STALL_SECONDS": value})


def test_host_window_and_load_scaling(monkeypatch):
    mod = watchdog_module()
    monkeypatch.setattr(mod.os, "getloadavg", lambda: (8, 8, 8))
    monkeypatch.setattr(mod.os, "cpu_count", lambda: 2)
    assert mod.check_watchdog_config({"STATEFRAME_CHECK_STALL_SECONDS": "17"}).stall_seconds == 68


@pytest.mark.parametrize("route,path", [
    ("composed", "probes/react_repair.py"),
    ("generic", "kernel_generic_check.py"),
    ("plain", "kernel_plain_host.py"),
    ("parallel_command", "kernel_unit_policy.py"),
    ("parallel_pytest", "kernel_parallel.py"),
    ("outer_inner", "../outer_checks.py"),
    ("parent_witness", "kernel_authored_witness.py"),
    ("agent_authored", "kernel_agent_authored_check.py"),
    ("unittest_adapter", "kernel_authored_check.py"),
])
def test_every_check_launch_explicitly_selects_unbounded_progress_mode(route, path):
    tree = ast.parse((ENGINE / path).read_text())
    aliases = {"runner"} if route == "unittest_adapter" else set()
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ((isinstance(node.func, ast.Name) and node.func.id in {"run_sandboxed", *aliases})
                  or isinstance(node.func, ast.Attribute) and node.func.attr == "run_sandboxed")]
    assert calls, route
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        assert isinstance(keywords.get("timeout"), ast.Constant) and keywords["timeout"].value is None, route
        assert "check_watchdog" in keywords, route
        assert not isinstance(keywords.get("cpu_seconds"), ast.Constant) or keywords["cpu_seconds"].value is None, route


class Clock:
    now = 0.0
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += .5


@pytest.mark.parametrize("mode", ["cpu_after_eof", "truncated_output", "idle", "boundary", "duplicate_samples", "inherited_writer"])
def test_collector_supervises_live_process_after_eof_and_full_output(monkeypatch, mode):
    mod = watchdog_module()
    clock = Clock()
    events = []
    chunks = 8002 if mode in {"cpu_after_eof", "truncated_output"} else 8
    state = {"turn": 0, "exited": False, "killed": 0}
    process = SimpleNamespace(pid=42, stdin=None, returncode=None,
                              stdout=io.BytesIO(), stderr=io.BytesIO())
    # File descriptors are real; selector and pipe reads are deterministic.
    import os
    pipes = [os.pipe(), os.pipe()]
    process.stdout = os.fdopen(pipes[0][0], "rb")
    process.stderr = os.fdopen(pipes[1][0], "rb")
    class Selector:
        def __init__(self): self.keys = {}
        def register(self, stream, mask, data):
            self.keys[stream] = SimpleNamespace(fileobj=stream, data=data)
        def unregister(self, stream): self.keys.pop(stream, None)
        def get_map(self): return self.keys
        def select(self, seconds):
            clock.now += .5
            state["turn"] += 1
            if mode == "duplicate_samples" and state["turn"] > 20:
                raise AssertionError("duplicate snapshot polls prevented a proven stall")
            if mode == "inherited_writer" and state["turn"] > 24:
                raise AssertionError("exited leader left the output collector blocked")
            if state["turn"] >= chunks and mode not in {"idle", "duplicate_samples"}: state["exited"] = True
            return [(key, 1) for key in list(self.keys.values())]
        def close(self): pass
    def read(fd, size):
        if mode == "inherited_writer":
            return b"x" if not state["killed"] else b""
        return b"xxxxxxxx" if mode in {"truncated_output", "boundary"} and not state["exited"] else b""
    def sample(proc, now):
        proc._stateframe_check_sample_pending = mode == "duplicate_samples" and state["turn"] % 2 == 0
        if proc._stateframe_check_sample_pending: return None
        ticks = state["turn"] if mode == "cpu_after_eof" else 0
        return snapshot(ticks)
    family = object()
    process._stateframe_check_sampler = SimpleNamespace(family=family)
    def kill(*args, **kwargs):
        if mode == "inherited_writer":
            assert args[1] is family
        state["killed"] += 1
        state["exited"] = True
        return True
    def wait(timeout):
        assert state["exited"]
        process.returncode = -9 if state["killed"] else 0
    process.wait = wait
    monkeypatch.setattr(policy.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(policy.time, "sleep", clock.sleep)
    monkeypatch.setattr(policy.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(policy.os, "read", read)
    monkeypatch.setattr(policy, "check_process_progress", sample)
    monkeypatch.setattr(policy, "child_exited_unreaped", lambda p: state["exited"])
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda p: SimpleNamespace(pid=42))
    monkeypatch.setattr(policy, "signal_launch_group", lambda *args: True)
    monkeypatch.setattr(policy, "ProcessTable", lambda: SimpleNamespace(snapshot=lambda: {}))
    monkeypatch.setattr(policy, "_kill_group", kill)
    monkeypatch.setattr(policy, "_exited_unit_cpu_seconds", lambda p: 0.0)
    try:
        result = policy._collect(process, None, None, 4,
                                 check_watchdog=mod.CheckWatchdogConfig(2, on_event=lambda *e: events.append(e)))
    finally:
        for _, fd in pipes: os.close(fd)
    assert state["killed"] == (1 if mode in {"idle", "duplicate_samples", "inherited_writer"} else 0)
    assert not result[2]  # no-progress is not a legacy elapsed timeout
    if mode in {"idle", "duplicate_samples"}:
        assert process.execution_stop["cause"] == "no_progress"
        assert process.execution_stop["cleanup_ok"] is True
        assert [name for name, _ in events].count("execution_stopped") == 1
    if mode == "truncated_output": assert result[3]


def test_default_check_preexec_does_not_set_or_clear_inherited_cpu(monkeypatch):
    applied = []
    monkeypatch.setattr(policy.resource, "getrlimit", lambda which: (7, 9))
    monkeypatch.setattr(policy.resource, "setrlimit", lambda which, limits: applied.append((which, limits)))
    monkeypatch.setattr(policy, "_cgroup_memory_limit_configured", lambda *args: False)
    policy._resource_preexec(None, 512 * 1024**2, 64 * 1024**2, 128, 64)()
    assert all(which != resource.RLIMIT_CPU for which, _ in applied)
    policy._resource_preexec(5, 512 * 1024**2, 64 * 1024**2, 128, 64)()
    assert (resource.RLIMIT_CPU, (4, 5)) in applied


def test_no_progress_receipt_round_trip_overrides_even_a_green_partial_report():
    from kernel_check_receipts import record_check_output
    from kernel_check_truth_adapter import record_native_observation
    mod = watchdog_module()
    stop = mod.CheckWatchdog(mod.CheckWatchdogConfig(2)).stop_payload(cleanup_ok=True)
    runtime = SimpleNamespace()
    row = record_native_observation(
        runtime, execution_id="check-stalled", selector="tests", status="PASS", exit_code=-9,
        typed_observation={"outcome": "PASS", "counts": {"failed": 0, "passed": 1},
                           "execution_stop": stop}, finalize=False)
    private = next(iter(runtime._check_output_receipts.values()))
    for receipt in (row, private):
        assert receipt["outcome"] == "NO_PROGRESS"
        assert receipt["termination_reason"] == "NO_PROGRESS"
        assert receipt["execution_complete"] is False
        assert receipt["reached_observation"] is False
    assert private["execution_stop"] == stop
    assert private["counts"]["failed"] == 0
    stop["cause"] = "injected"
    assert private["execution_stop"]["cause"] == "no_progress"
    legacy = record_check_output(SimpleNamespace(), execution_id="old", selector="tests", status="PASS", exit_code=0)
    assert legacy["outcome"] == "PASS"


def test_explicit_stop_precedes_numeric_exit_heuristics():
    from kernel_kill_attribution import attribute_termination
    mod = watchdog_module()
    stop = mod.CheckWatchdog(mod.CheckWatchdogConfig(2)).stop_payload(cleanup_ok=True)
    assert attribute_termination(124, execution_stop=stop).initiator == "no_progress"


def test_outer_progress_packets_have_separate_bounded_accounting():
    from stateframe.outer_checks import HelperPackets, OuterCheckError
    packets = HelperPackets()
    for sequence in range(1, 40001):
        raw = json.dumps({"progress": {"sequence": sequence, "kind": "copy", "completed": sequence}}).encode() + b"\n"
        packets.feed(raw)
    assert packets.completed_work == 40000
    assert packets.result_bytes == 0
    with pytest.raises(OuterCheckError): packets.feed(b"x" * (packets.packet_limit + 1))


@pytest.mark.parametrize("progress", [
    {"sequence": 1, "kind": "heartbeat", "completed": 1},
    {"sequence": True, "kind": "copy", "completed": 1},
    {"sequence": 1, "kind": "copy", "completed": 0},
])
def test_outer_rejects_fake_progress(progress):
    from stateframe.outer_checks import HelperPackets, OuterCheckError
    with pytest.raises(OuterCheckError):
        HelperPackets().feed(json.dumps({"progress": progress}).encode() + b"\n")


@pytest.mark.parametrize("route", ["generic", "composed", "plain", "parallel_command", "parallel_pytest"])
@pytest.mark.parametrize("stopped", [False, True])
def test_real_adapters_select_progress_mode_and_keep_incomplete_results(tmp_path, monkeypatch, route, stopped):
    import kernel_pilot
    import kernel_generic_check
    import kernel_plain_host
    import kernel_parallel
    import kernel_unit_policy
    from probes import react_repair
    mod = watchdog_module()
    calls = []
    events = []
    stop = mod.CheckWatchdog(mod.CheckWatchdogConfig(2)).stop_payload(cleanup_ok=True) if stopped else None
    def execute(command, root, **options):
        calls.append(options)
        assert options["timeout"] is None
        assert options["cpu_seconds"] is None
        watch = mod.CheckWatchdog(options["check_watchdog"])
        if not stopped:
            for tick in range(8002):
                assert not watch.observe(tick / 2, snapshot(tick))
        return SimpleNamespace(completed=subprocess.CompletedProcess(command, -9 if stopped else 0,
                                                                      "1 passed\n", ""),
                               timed_out=False, output_truncated=False, execution_stop=stop,
                               pid=None, cleanup_finding=None)
    monkeypatch.delenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", raising=False)
    monkeypatch.setattr(policy, "run_sandboxed", execute)
    monkeypatch.setattr(kernel_generic_check, "run_sandboxed", execute)
    monkeypatch.setattr(kernel_pilot, "charge", lambda *args: None)
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: None)
    monkeypatch.setattr(kernel_unit_policy, "reap_check_group", lambda *args, **kw: None)
    monkeypatch.setattr(kernel_unit_policy, "_matching_check_processes", lambda token: ())
    token = kernel_pilot.ACTIVE_RUNTIME.set(None)
    try:
        if route == "generic":
            result, code = kernel_generic_check.check_result(tmp_path, {"test_command": ["check"]})
            assert code == (-9 if stopped else 0)
        elif route == "composed":
            (tmp_path / "test_example.py").write_text("def test_example(): pass\n")
            monkeypatch.setattr(react_repair, "_effect_boundary_enabled", lambda: False)
            result = react_repair.run_tests(tmp_path, "", "test_example.py")
        elif route == "plain":
            runner = kernel_plain_host.sandboxed_check_runner(tmp_path / "lock", tmp_path / "audit")
            result = runner([sys.executable, "-m", "pytest"], tmp_path, {})
            assert result.execution_stop == stop
            assert result.exit_code == (None if stopped else 0)
            result = result.output
        elif route == "parallel_command":
            result = kernel_unit_policy.run_check_command(tmp_path, "check")
        else:
            spec = SimpleNamespace(contract=SimpleNamespace(check=SimpleNamespace(command_kind="pytest")))
            monkeypatch.setattr(kernel_unit_policy, "UnitPolicy", lambda spec: spec)
            monkeypatch.setattr(kernel_unit_policy, "run_frozen_check",
                                lambda policy, root, *, execute, evidence_dir: execute(root, "test_example.py"))
            result = kernel_parallel._check(spec, tmp_path, tmp_path, test_python=sys.executable)
        assert calls
        if stopped:
            assert "1 failed" not in result and "pytest exit 1" not in result
            if route != "plain": assert result.execution_stop == stop
    finally:
        kernel_pilot.ACTIVE_RUNTIME.reset(token)


@pytest.mark.parametrize("mode", ["copy", "cpu", "idle", "deadline", "lost_ack", "cancel", "inherited_writer"])
def test_outer_parent_liveness_protocol_deadline_and_cleanup(tmp_path, monkeypatch, mode):
    import os
    from stateframe import outer_checks
    from stateframe.engine import kernel_check_watchdog as outer_watch
    from stateframe.engine import kernel_exec_tracker as outer_tracker
    clock = Clock()
    state = {"turn": 0, "exited": False, "signals": [], "spec": None}
    target = {"copy": 40000, "cpu": 8002, "inherited_writer": 3}.get(mode, 20)
    read_ends = [os.pipe(), os.pipe()]
    stdout = os.fdopen(read_ends[0][0], "rb")
    stderr = os.fdopen(read_ends[1][0], "rb")
    child = SimpleNamespace(pid=42, stdout=stdout, stderr=stderr, returncode=None)
    def wait(timeout):
        assert state["exited"]
        child.returncode = 0
    def stop(signum):
        state["signals"].append(signum)
        state["exited"] = True
    child.wait = wait
    child.send_signal = stop
    child.kill = lambda: stop(9)
    def spawn(*args, **kwargs):
        state["spec"] = json.load(kwargs["stdin"])
        return child
    class Selector:
        def __init__(self): self.keys = {}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def register(self, stream, mask, data):
            self.keys[stream] = SimpleNamespace(fileobj=stream, data=data)
        def unregister(self, stream): self.keys.pop(stream, None)
        def get_map(self): return self.keys
        def select(self, seconds):
            clock.now += .5
            state["turn"] += 1
            assert mode != "inherited_writer" or state["turn"] <= 24
            if mode == "cancel" and state["turn"] == 3: raise KeyboardInterrupt
            if state["turn"] >= target: state["exited"] = True
            return [(key, 1) for key in list(self.keys.values())]
    sent_final = False
    def read(fd, count):
        nonlocal sent_final
        if fd == stderr.fileno(): return b""
        if state["exited"]:
            if sent_final:
                return b"x" if mode == "inherited_writer" else b""
            sent_final = True
            final = b'{"result":{"exit_code":0,"passed":true}}\n'
            if mode != "lost_ack": final += b'{"cleanup_ok":true}\n'
            return final
        if mode == "copy":
            turn = state["turn"]
            return json.dumps({"progress": {"sequence": turn, "kind": "copy", "completed": turn}}).encode() + b"\n"
        # A quiet pipe has no packet. Returning no selected key would be
        # equivalent; EOF additionally exercises post-EOF supervision.
        return b""
    # Copy mode keeps stdout open until its final records. Quiet mode closes
    # stdout early, so final acknowledgement is supplied on its last live turn.
    original_select = Selector.select
    def select(self, seconds):
        result = original_select(self, seconds)
        if mode != "copy" and not state["exited"]:
            return [(key, mask) for key, mask in result if key.data == "stderr"]
        return result
    Selector.select = select
    monkeypatch.setattr(outer_checks.subprocess, "Popen", spawn)
    monkeypatch.setattr(outer_checks.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(outer_checks.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(outer_checks.time, "sleep", clock.sleep)
    monkeypatch.setattr(outer_checks.os, "read", read)
    monkeypatch.setattr(outer_watch.os, "getloadavg", lambda: (0, 0, 0))
    family = object()
    monkeypatch.setattr(outer_watch, "LaunchProgressSampler", lambda child: SimpleNamespace(
        family=family, sample=lambda: snapshot(state["turn"] if mode == "cpu" else 0)))

    def sweep(proc, recorded_family, **kwargs):
        assert proc is child and state["exited"] and recorded_family is family
        return ()

    monkeypatch.setattr(outer_tracker, "kill_launch_unit", sweep)
    monkeypatch.setattr(outer_tracker, "capture_launch_identity", lambda *args, **kw: None)
    monkeypatch.setattr(outer_tracker, "child_exited_unreaped", lambda child: state["exited"])
    monkeypatch.delenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", raising=False)
    try:
        if mode == "inherited_writer":
            from stateframe.outer_checks import OuterCheckError
            with pytest.raises(OuterCheckError, match="helper pipes survived cleanup"):
                outer_checks.run("check", tmp_path, python=sys.executable, timeout_s=2, deadline=3)
        elif mode in {"lost_ack", "cancel"}:
            with pytest.raises(KeyboardInterrupt if mode == "cancel" else outer_checks.OuterCheckError):
                outer_checks.run("check", tmp_path, python=sys.executable, timeout_s=2)
        else:
            result = outer_checks.run("check", tmp_path, python=sys.executable, timeout_s=2,
                                      deadline=1.5 if mode == "deadline" else None)
            if mode in {"copy", "cpu"}:
                assert result["passed"] is True and state["signals"] == []
                assert result["execution_stop"] is None
                assert clock.now > 3600
            else:
                assert result["passed"] is False
                assert result["timed_out"] is True
                assert result["execution_stop"]["cause"] == ("user_deadline" if mode == "deadline" else "no_progress")
                assert result["execution_stop"]["cleanup_ok"] is True
        assert state["spec"]["deadline"] == {"deadline": 1.5, "inherited_writer": 3}.get(mode)
        assert state["spec"]["cpu_seconds"] is None
        assert state["spec"]["stall_seconds"] == 2
        if mode in {"idle", "deadline", "cancel", "lost_ack"}:
            assert state["signals"]
    finally:
        for stream in (stdout, stderr):
            if not stream.closed: stream.close()
        for _, fd in read_ends: os.close(fd)


@pytest.mark.host_sandbox
def test_real_idle_check_stops_and_cleans_its_descendant(tmp_path):
    mod = watchdog_module()
    source = "import subprocess,sys,time\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'])\nprint('child started',flush=True)\ntime.sleep(90)\n"
    result = policy.run_sandboxed([sys.executable, "-c", source], tmp_path, timeout=None,
                                  check_watchdog=mod.CheckWatchdogConfig(1.5), generic_check=True)
    assert result.execution_stop["cause"] == "no_progress"
    assert result.execution_stop["cleanup_ok"] is True
    assert result.process_group_killed
    assert not result.timed_out
    assert result.cpu_limit_seconds is None


@pytest.mark.host_sandbox
def test_real_cpu_check_keeps_running_after_both_pipes_close(tmp_path):
    mod = watchdog_module()
    source = "import os,time\nos.close(1);os.close(2)\nend=time.process_time()+2\nwhile time.process_time()<end: pass\n"
    result = policy.run_sandboxed([sys.executable, "-c", source], tmp_path, timeout=None,
                                  check_watchdog=mod.CheckWatchdogConfig(1), generic_check=True)
    assert result.completed.returncode == 0
    assert result.execution_stop is None
    assert result.cpu_limit_seconds is None


def test_progress_sink_and_receipt_preserve_terminal_evidence():
    import kernel_pilot
    from kernel_check_receipts import record_check_output
    mod = watchdog_module()
    events = []
    runtime = SimpleNamespace(journal=SimpleNamespace(emit=lambda name, **fields: events.append((name, fields))))
    token = kernel_pilot.ACTIVE_RUNTIME.set(runtime)
    try:
        watch = mod.CheckWatchdog(mod.CheckWatchdogConfig(1, on_event=lambda name, fields: events.append((name, fields))))
        for tick in range(20): watch.observe(tick / 2, snapshot(tick))
        stop = watch.stopped(cleanup_ok=True)
        assert [name for name, _ in events].count("execution_stopped") == 1
        assert 0 < [name for name, _ in events].count("execution_progress") < 20
        record_check_output(runtime, execution_id="idle", selector="test", status="PASS", exit_code=0,
                            typed_observation={"execution_stop": stop})
        packet = [fields for name, fields in events if name == "check_receipt"][-1]
        assert packet["execution_stop"] == stop
        # A late wrapper cannot turn this same unfinished execution green.
        late = record_check_output(runtime, execution_id="idle", selector="test", status="PASS", exit_code=0,
                                   outcome="PASS")
        assert late["outcome"] == "NO_PROGRESS" and late["execution_complete"] is False
        assert "execution_stop" not in late
    finally:
        kernel_pilot.ACTIVE_RUNTIME.reset(token)


def test_unknown_cpu_sample_and_repeated_supervisor_snapshot_are_not_stall_evidence(monkeypatch):
    from kernel_exec_tracker import ProcessIdentity
    mod = watchdog_module()
    row = ProcessIdentity(42, 1, 42, 42, (10, 0), False)
    assert mod.process_snapshot(SimpleNamespace(cpu_ticks=lambda row: None), [row]) is None
    value = snapshot(10)
    process = SimpleNamespace(_stateframe_memory_supervisor=object(), _stateframe_check_progress=(1, value))
    assert policy.check_process_progress(process, 1) is value
    assert policy.check_process_progress(process, 1.5) is None
    process._stateframe_check_progress = (2, value)
    assert policy.check_process_progress(process, 2) is value
    assert policy.check_process_progress(process, 20) is None


@pytest.mark.parametrize("field,value", [("cause", []), ("scope", {}), ("resource", []),
                                          ("evidence", []), ("cleanup_ok", 1),
                                          ("progress_sequence", True), ("inactivity_seconds", float("nan"))])
def test_receipts_reject_malformed_terminal_metadata(field, value):
    from kernel_check_receipts import record_check_output
    mod = watchdog_module()
    stop = mod.CheckWatchdog(mod.CheckWatchdogConfig(1)).stop_payload(cleanup_ok=True)
    stop[field] = value
    with pytest.raises(ValueError):
        record_check_output(SimpleNamespace(), execution_id="invalid", selector="test", status="UNKNOWN",
                            typed_observation={"execution_stop": stop})


@pytest.mark.parametrize("scratch_removal_fails", [False, True])
def test_helper_transports_selected_settings_and_reports_completed_copy_work(
        tmp_path, monkeypatch, scratch_removal_fails):
    import os
    import signal
    from stateframe import outer_checks
    source = tmp_path / "repo"
    source.mkdir()
    (source / "check.py").write_text("print('done')\n")
    captured = io.StringIO()
    remove_scratch = outer_checks.tempfile.TemporaryDirectory.cleanup

    def cleanup(scratch):
        assert json.loads(captured.getvalue().splitlines()[-1]) == {"cleanup_ok": True}
        remove_scratch(scratch)
        if scratch_removal_fails:
            raise OSError("scratch removal failed")

    spec = {"command": "check", "cwd": str(source), "deadline": None,
            "stall_seconds": 17.0, "cpu_seconds": 3.5, "test_env": {},
            "pytest_plugins": None, "test_python": None, "runner_kind": "generic"}
    seen = []
    def run(command, root, **options):
        assert options["timeout"] is None
        assert options["check_watchdog"].stall_seconds == 17
        assert options["cpu_seconds"] == 3.5
        assert (root / "check.py").read_text() == "print('done')\n"
        options["check_watchdog"].on_progress(1, "output")
        seen.append(root)
        return SimpleNamespace(completed=subprocess.CompletedProcess(command, 0, "done\n", ""),
                               timed_out=False, output_truncated=False, process_group_killed=True)
    monkeypatch.setattr(policy, "require_capabilities", lambda: None)
    monkeypatch.setattr(policy, "run_sandboxed", run)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(spec)))
    monkeypatch.setattr(sys, "stdout", captured)
    monkeypatch.setattr(outer_checks.tempfile.TemporaryDirectory, "cleanup", cleanup)
    original_spawn = policy.sandboxed_popen
    original_kill_group = policy._kill_group
    previous_term = signal.getsignal(signal.SIGTERM)
    old_env, old_path = dict(os.environ), list(sys.path)
    try:
        if scratch_removal_fails:
            with pytest.raises(OSError, match="scratch removal failed"):
                outer_checks._helper()
        else:
            assert outer_checks._helper() == 0
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        os.environ.clear()
        os.environ.update(old_env)
        sys.path[:] = old_path
    assert policy.sandboxed_popen is original_spawn
    assert policy._kill_group is original_kill_group
    packets = [json.loads(line) for line in captured.getvalue().splitlines()]
    assert any(packet.get("progress", {}).get("kind") == "copy" for packet in packets)
    assert packets[-1] == {"cleanup_ok": True}
    assert seen and not seen[0].exists()


def test_check_mode_omitted_limits_do_not_select_probe_defaults(tmp_path, monkeypatch):
    mod = watchdog_module()
    seen = []
    class Spawned(Exception): pass
    def spawn(command, root, **options):
        seen.append(options)
        raise Spawned
    monkeypatch.setattr(policy, "sandboxed_popen", spawn)
    monkeypatch.setattr(policy, "_command_cgroup", lambda *args, **kwargs: None)
    monkeypatch.delenv("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", raising=False)
    with pytest.raises(Spawned):
        policy.run_sandboxed(["check"], tmp_path, check_watchdog=mod.CheckWatchdogConfig())
    assert seen[0]["cpu_seconds"] is None
    with pytest.raises(Spawned):
        policy.run_sandboxed(["probe"], tmp_path)
    assert seen[1]["cpu_seconds"] == policy.DEFAULT_CPU_SECONDS


def test_bounded_probe_and_unbounded_check_modes_keep_distinct_validation(tmp_path):
    with pytest.raises(policy.ExecutionPolicyError):
        policy.run_sandboxed(["check"], tmp_path, timeout=None)
    assert policy._effective_cpu_limit_seconds(None) is None


def test_partial_family_sample_cannot_erase_a_still_live_child(monkeypatch):
    from kernel_exec_tracker import ProcessIdentity
    mod = watchdog_module()
    root = ProcessIdentity(42, 1, 42, 42, (10, 0), False)
    child = ProcessIdentity(43, 42, 42, 42, (11, 0), False)
    previous = mod.ProgressSnapshot({(42, root.start): 10, (43, child.start): 20})
    table = SimpleNamespace(cpu_ticks=lambda row: 10, identity=lambda pid: None)
    monkeypatch.setattr(mod.os, "kill", lambda pid, sig: None)
    assert mod.process_snapshot(table, [root], previous=previous) is None
    def gone(pid, sig): raise ProcessLookupError
    monkeypatch.setattr(mod.os, "kill", gone)
    assert mod.process_snapshot(table, [root], previous=previous) is not None
    table.identity = lambda pid: ProcessIdentity(43, 1, 43, 43, (99, 0), False)
    assert mod.process_snapshot(table, [root], previous=previous) is None


def test_no_progress_is_a_recorded_result_not_a_crashed_check_effect(tmp_path, monkeypatch):
    import kernel_pilot
    import kernel_unit_policy
    from probes import react_repair
    mod = watchdog_module()
    stop = mod.CheckWatchdog(mod.CheckWatchdogConfig(1)).stop_payload(cleanup_ok=True)
    crashed = []
    (tmp_path / "test_example.py").write_text("def test_example(): pass\n")
    def boundary(command, *, runner, **kwargs):
        try:
            return runner(command), None
        except Exception:
            crashed.append(True)
            raise
    monkeypatch.setattr(react_repair, "_effect_test_subprocess", boundary)
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: None)
    monkeypatch.setattr(kernel_pilot, "charge", lambda *args: None)
    monkeypatch.setattr(policy, "run_sandboxed", lambda command, root, **kw: SimpleNamespace(
        completed=subprocess.CompletedProcess(command, -9, "partial", ""), execution_stop=stop,
        output_truncated=False, timed_out=False))
    token = kernel_pilot.ACTIVE_RUNTIME.set(None)
    try:
        value = react_repair.run_tests(tmp_path, "", "test_example.py")
    finally:
        kernel_pilot.ACTIVE_RUNTIME.reset(token)
    assert value.execution_stop == stop
    assert crashed == []


def test_frozen_whole_command_keeps_no_progress_in_canonical_receipt_and_tape(tmp_path, monkeypatch):
    from dataclasses import replace
    import kernel_check_permits
    import kernel_pilot
    import kernel_unit_policy
    from kernel_check_receipts import record_check_output
    from kernel_parallel_records import CheckRecipe
    from test_unit_checks import candidate, policy as unit_policy
    from probes import react_repair
    mod = watchdog_module()
    stop = mod.CheckWatchdog(mod.CheckWatchdogConfig(1)).stop_payload(cleanup_ok=True)
    root = candidate(tmp_path)
    unit = unit_policy()
    recipe = CheckRecipe(("tests/test_a.py",), ("check",), "run_check_command")
    unit = kernel_unit_policy.UnitPolicy(replace(unit.spec, contract=replace(unit.spec.contract, check=recipe)))
    monkeypatch.setattr(kernel_check_permits, "_installed", (tmp_path / "permits", 1))
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: unit)
    monkeypatch.setattr(kernel_unit_policy, "_matching_check_processes", lambda token: ())
    monkeypatch.setattr(kernel_unit_policy, "reap_check_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy, "run_sandboxed", lambda command, root, **kw: SimpleNamespace(
        completed=subprocess.CompletedProcess(command, -9, "partial", ""), execution_stop=stop,
        output_truncated=False, timed_out=False, pid=None, cleanup_finding=None))
    runtime = SimpleNamespace(journal=SimpleNamespace(path=tmp_path / "events.jsonl", emit=lambda *a, **kw: None),
                              owner=SimpleNamespace(current_step=1),
                              binding=lambda **kwargs: {"product_revision": "revision"})
    def observe(selectors, code, binding, **fields):
        record_check_output(runtime, execution_id="frozen", selector="check", status="UNKNOWN",
                            exit_code=code, typed_observation=fields.get("typed_observation"))
    runtime.test_observation = observe
    token = kernel_pilot.ACTIVE_RUNTIME.set(runtime)
    try:
        react_repair.run_tests(root, "", "check")
    finally:
        kernel_pilot.ACTIVE_RUNTIME.reset(token)
    private = next(iter(runtime._check_output_receipts.values()))
    assert private["outcome"] == "NO_PROGRESS"
    assert private["termination_reason"] == "NO_PROGRESS"
    assert private["counts"]["failed"] == 0
    assert private["execution_stop"] == stop
    receipt = runtime._unit_check_receipt
    assert receipt.status == "INVALID" and receipt.invalid_reason == "CHECK_NO_PROGRESS"
    assert "execution_stopped" in Path(receipt.raw_evidence_ref).read_text()
    assert "inactivity_seconds" not in receipt.projected_observation


def test_macos_cpu_sample_is_identity_checked_and_never_synthesized(monkeypatch):
    import ctypes
    from kernel_exec_tracker import ProcessIdentity
    mod = watchdog_module()
    row = ProcessIdentity(42, 1, 42, 42, (10, 0), False)
    class Lib:
        def proc_pidinfo(self, pid, flavor, arg, pointer, size):
            info = ctypes.cast(pointer, ctypes.POINTER(mod._TaskInfo)).contents
            info.user, info.system = 20, 7
            return size
    table = SimpleNamespace(_lib=Lib(), cpu_ticks=lambda row: None, identity=lambda pid: row)
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    assert mod.process_snapshot(table, [row]).cpu[(42, row.start)] == 27
    table.identity = lambda pid: ProcessIdentity(42, 1, 42, 42, (99, 0), False)
    assert mod.process_snapshot(table, [row]) is None
