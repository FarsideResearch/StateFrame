"""Crash-boundary tests for mutation launch registration and phased broker calls."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

import pytest
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified

from mutate_leftovers import make_launch_spec, mutation_root, stage_test_unit, sweep_and_assert


pytestmark = pytest.mark.host_sandbox

LAUNCH_STAGES = ("PREPARED", "STARTED", "REGISTERED", "RELEASED", "EXITED", "REAPED")


def _kill_owner_at_stage(tmp_path, runtime_root, stage, source, *, kill_supervisor=False):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    staged = stage_test_unit(build, runtime_root, source)
    spec = make_launch_spec(execution, wire, staged, call=f"crash-{stage.lower()}")
    engine_dir = Path(__file__).resolve().parents[1] / "src" / "stateframe" / "engine"
    config = tmp_path / f"{stage.lower()}-spec.json"
    config.write_text(json.dumps({
        "engine_dir": str(engine_dir), "runtime_root": str(runtime_root),
        "unit_dir": str(spec.unit_dir), "call": spec.call, "launch": spec.launch,
        "stdin_doc": base64.b64encode(spec.stdin_doc).decode("ascii"),
        "role": spec.role,
        "limits": {"wall_s": spec.limits.wall_s, "cpu_s": spec.limits.cpu_s,
                   "mem_mb": spec.limits.mem_mb, "output_bytes": spec.limits.output_bytes},
        "target_stage": stage,
    }), encoding="utf-8")
    owner = tmp_path / f"{stage.lower()}-owner.py"
    owner.write_text(
        "import base64, json, sys, time\n"
        "from pathlib import Path\n"
        "cfg = json.loads(Path(sys.argv[1]).read_text())\n"
        "sys.path.insert(0, cfg['engine_dir'])\n"
        "from kernel_mutate_exec import LaunchSpec, MutationExecutor, MutationLimits\n"
        "root = Path(cfg['runtime_root'])\n"
        "spec = LaunchSpec(role=cfg['role'], call=cfg['call'], launch=cfg['launch'],\n"
        "    unit_dir=Path(cfg['unit_dir']), inputs={}, responses={},\n"
        "    stdin_doc=base64.b64decode(cfg['stdin_doc']), limits=MutationLimits(**cfg['limits']))\n"
        "def barrier(state):\n"
        "    if state == cfg['target_stage']:\n"
        "        print('AT_STAGE:' + state, flush=True)\n"
        "        while True: time.sleep(0.02)\n"
        "executor = MutationExecutor(root, network_envelope='none', launch_stage_hook=barrier)\n"
        "executor.run(spec)\n",
        encoding="utf-8",
    )
    launched_at = time.time()
    process = subprocess.Popen([sys.executable, "-I", str(owner), str(config)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               close_fds=True, start_new_session=True)
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    ready = ""
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            for key, _events in selector.select(timeout=0.1):
                ready = key.fileobj.readline().strip()
                if ready:
                    break
            if ready or process.poll() is not None:
                break
        assert ready == f"AT_STAGE:{stage}", (ready, process.stderr.read() if process.poll() is not None else "owner timed out")
        if kill_supervisor:
            from kernel_exec_tracker import ProcessTable, process_cmdline_digest

            ledger = [json.loads(line) for line in
                      (runtime_root / "launches.jsonl").read_text(encoding="utf-8").splitlines()]
            released = next(row for row in reversed(ledger) if row.get("state") == "RELEASED")
            identity = released["identity"]
            assert identity["cmdline_sha256"] == process_cmdline_digest(identity["pid"])
            assert identity["ns"] == ProcessTable().pid_namespace(identity["pid"])
            supervisor = json.loads((runtime_root / "owner.json").read_text())["pid"]
            supervisor_identity = record_identity(supervisor, launched_after=launched_at,
                                                  command_fragment="kernel_exec_supervisor")
            signal_run_verified(runtime_root, supervisor, supervisor_identity, signal.SIGKILL,
                                deadline=time.monotonic(), what="supervisor crash",
                                require_checkpoint_saved=False)
        signal_run_owned(runtime_root, process, signal.SIGKILL,
                         deadline=time.monotonic(), what="owner crash",
                         require_checkpoint_saved=False)
        process.wait(timeout=5)
    finally:
        selector.close()
        if process.poll() is None:
            signal_run_owned(runtime_root, process, signal.SIGKILL,
                             deadline=time.monotonic(), what="owner cleanup",
                             require_checkpoint_saved=False)
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
    sweep_and_assert(runtime_root)


@pytest.mark.parametrize("stage", LAUNCH_STAGES)
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux PID namespace and pidfd identity support orphan cleanup",
)
def test_mutation_launch_ledger_crash_matrix(mutation_root, tmp_path, stage):
    _kill_owner_at_stage(tmp_path, mutation_root, stage,
                         "def handle(input, ctx):\n    return {'ok': True}\n")
    assert not mutation_root.exists()


@pytest.mark.parametrize("stage", LAUNCH_STAGES)
@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux PID namespace and pidfd identity support orphan cleanup",
)
def test_mutation_engine_sigkill_at_each_launch_stage(mutation_root, tmp_path, stage):
    _kill_owner_at_stage(tmp_path, mutation_root, stage,
                         "def handle(input, ctx):\n    return {'ok': True}\n")
    assert not mutation_root.exists()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux PID namespace and pidfd identity support orphan cleanup",
)
def test_mutation_signal_engine_owner_and_supervisor_crash_cleanup(mutation_root, tmp_path):
    _kill_owner_at_stage(
        tmp_path, mutation_root, "RELEASED",
        "import time\ndef handle(input, ctx):\n    while True: time.sleep(0.02)\n",
        kill_supervisor=True,
    )
    assert not mutation_root.exists()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux final-exec identity and parent-death cleanup",
)
def test_post_exec_crash_before_identity_registration_cleans_orphan(mutation_root):
    import kernel_mutate_build as build
    import kernel_mutate_exec as execution
    import kernel_mutate_wire as wire

    staged = stage_test_unit(build, mutation_root,
                             "import time\ndef handle(input, ctx):\n    while True: time.sleep(0.02)\n")
    spec = make_launch_spec(execution, wire, staged, call="pre-identity-crash")
    executor = execution.MutationExecutor(mutation_root, network_envelope="none")
    original = executor._session._rpc

    def fault(request):
        if request.get("op") == "start":
            request = {**request, "_mutation_fault": "after_spawn_before_identity"}
        return original(request)

    executor._session._rpc = fault
    try:
        with pytest.raises(execution.MutationContainmentError):
            executor.run(spec)
        rows = [json.loads(line) for line in (mutation_root / "launches.jsonl").read_text().splitlines()]
        assert "RELEASED" not in {row["state"] for row in rows}
    finally:
        try:
            executor.close()
        except execution.MutationContainmentError:
            pass
        sweep_and_assert(mutation_root)


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux /proc identity after a delegated cgroup launcher exec",
)
def test_supervisor_records_identity_after_cgroup_launcher_exec(tmp_path, monkeypatch):
    import socket
    from kernel_exec_capability import ExecCapability, ExecLimits
    from kernel_exec_tracker import ProcessTable, process_cmdline_digest
    import kernel_exec_supervisor as supervisor_module

    directory = tmp_path / "supervisor"
    directory.mkdir()
    cgroup = tmp_path / "delegated-cgroup"
    cgroup.mkdir()
    (cgroup / "cgroup.procs").touch()
    final_command = [sys.executable, "-I", "-c", "import time; time.sleep(60)"]
    capability = ExecCapability("mutation", tmp_path, False, (tmp_path,), (), "none", {}, ExecLimits())
    control, peer = socket.socketpair()
    owner_read, owner_write = os.pipe()
    supervisor = supervisor_module.Supervisor(control, directory, "bwrap", "host", owner_read)
    monkeypatch.setattr(supervisor_module, "capability_from_dict", lambda _value: capability)
    monkeypatch.setattr(supervisor_module, "_command_cgroup",
                        lambda _ident, _nproc, memory_bytes=None: cgroup)
    monkeypatch.setattr(supervisor_module, "capability_command",
                        lambda *_args, **_kwargs: (final_command, {}))

    try:
        result = supervisor.start({"cap": {}, "argv": ["unused"], "mounts": {"boot": str(tmp_path)}})
        identity = result["identity"]
        assert identity["cmdline_sha256"] == process_cmdline_digest(identity["pid"])
        assert identity["ns"] == ProcessTable().pid_namespace(identity["pid"])
        assert identity["pid"] == supervisor.jobs[result["id"]].process.pid
    finally:
        for job in supervisor.jobs.values():
            if job.cleanup is None:
                job.stop(0)
                deadline = time.monotonic() + 3
                while job.cleanup is None and time.monotonic() < deadline:
                    job.tick(supervisor.table.snapshot())
                    time.sleep(0.01)
                if job.cleanup is None:
                    signal_run_owned(tmp_path, job.process, signal.SIGKILL,
                                     deadline=time.monotonic(), what="supervisor job cleanup",
                                     require_checkpoint_saved=False)
                    job.process.wait(timeout=5)
                    job.tick(supervisor.table.snapshot())
        supervisor.selector.close()
        os.close(owner_read)
        os.close(owner_write)
        control.close()
        peer.close()
        (cgroup / "cgroup.procs").unlink(missing_ok=True)
        cgroup.rmdir()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux pidfd sweeper verifies a real command-line mismatch",
)
def test_mutation_sweeper_reports_only_true_command_mismatch(mutation_root):
    from kernel_exec_tracker import ProcessTable, process_cmdline_digest
    from kernel_mutate_sweep import sweep

    process = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        table = ProcessTable()
        row = table.identity(process.pid)
        assert row is not None
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        identity = {"pid": row.pid, "start": list(row.start), "boot_id": boot_id,
                    "ns": table.pid_namespace(row.pid),
                    "cmdline_sha256": "0" * 64}
        assert identity["ns"]
        owner_identity = {"pid": 2**31, "start": [1, 0], "boot_id": boot_id,
                          "ns": identity["ns"], "cmdline_sha256": "0" * 64}
        (mutation_root / "owner.json").write_text(json.dumps({
            "boot_id": boot_id, "identity": owner_identity}), encoding="utf-8")
        (mutation_root / "launches.jsonl").write_text(json.dumps({
            "schema": "stateframe-mutation-launch/1", "launch": "a" * 32,
            "state": "RELEASED", "identity": identity}) + "\n", encoding="utf-8")

        report = sweep(mutation_root.parent, run_locks=lambda _run: False)

        assert report.cleaned == ()
        assert report.unresolved == (mutation_root.name,)
        unresolved = json.loads((mutation_root / "UNRESOLVED.json").read_text(encoding="utf-8"))
        assert unresolved == {"kind": "UNRESOLVED", "reason": "process command identity differs"}
        assert process.poll() is None
    finally:
        signal_run_owned(mutation_root, process, signal.SIGKILL,
                         deadline=time.monotonic(), what="sweeper fixture cleanup",
                         require_checkpoint_saved=False)
        process.wait(timeout=5)
        if mutation_root.exists():
            report = sweep(mutation_root.parent, run_locks=lambda _run: False)
            assert not report.unresolved


def test_mutation_full_duplex_pipe_deadlock_is_bounded():
    from kernel_mutate_wire import WireError, parse_stdout, request_key

    call = "phased-fetch"
    request = {"method": "GET", "url": "https://example.com/report", "accept": "text/html"}
    need = json.dumps({"wire": 1, "call": call, "kind": "need", "fetch": [request]},
                      separators=(",", ":")).encode()
    framed = b"SFMW1 " + str(len(need)).encode() + b"\n" + need
    parsed = parse_stdout(framed, call=call)
    assert parsed.kind == "need"
    key = request_key(request["url"], request["accept"])
    assert key.startswith("sha256:") and len(key) == 71
    # The accepted transport is one complete phase followed by a fresh phase, not duplex IO.
    with pytest.raises(WireError):
        parse_stdout(framed + b"broker_result", call=call)


def test_mutation_broker_bidirectional_transport():
    # §13 replaces the original duplex proposal with recorded, one-shot phases.
    from kernel_mutate_wire import WireError, build_stdin_doc, parse_stdout, request_key

    call = "broker-phase"
    url, accept = "https://example.com/a", "text/html"
    key = request_key(url, accept)
    body = json.dumps({"wire": 1, "call": call, "kind": "need", "fetch": [
        {"method": "GET", "url": url, "accept": accept}],
    }, separators=(",", ":")).encode("utf-8")
    framed = b"SFMW1 " + str(len(body)).encode("ascii") + b"\n" + body
    need = parse_stdout(framed, call=call)
    assert need.kind == "need"
    assert list(need.fetch) == [{"method": "GET", "url": url, "accept": accept}]
    response = {"key": key, "method": "GET", "url": url, "accept": accept,
                "status": 200, "headers": {"content-type": "text/html"},
                "final_url": url, "body": "/resp/" + "a" * 64, "bytes": 0, "error": None}
    next_phase = build_stdin_doc(wire=1, call=call, launch="c" * 32, phase=1, role="payload",
                                 input={}, inputs=[], responses=[response],
                                 grants={"network": True, "writes": False}, max_fetch=4)
    resumed = json.loads(next_phase)
    assert resumed["phase"] == 1
    assert resumed["responses"] == [response]
    result_body = json.dumps({"wire": 1, "call": call, "kind": "result",
                              "result": {"title": "Ledger", "text": "Q1 42"}},
                             separators=(",", ":")).encode("utf-8")
    result_frame = parse_stdout(
        b"SFMW1 " + str(len(result_body)).encode("ascii") + b"\n" + result_body,
        call=call,
    )
    assert result_frame.kind == "result"
    assert result_frame.result == {"title": "Ledger", "text": "Q1 42"}
    with pytest.raises(WireError):
        parse_stdout(framed, call="another-call")
