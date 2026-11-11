"""Deterministic hook startup and timeout lifecycle regressions."""
from __future__ import annotations

from pathlib import Path
import base64
import json
import shutil
import socket
from types import SimpleNamespace


def test_executor_waits_for_supervisor_readiness_before_returning(tmp_path, monkeypatch):
    import kernel_mutate_exec as execute

    root = tmp_path / "runtime"
    root.mkdir()
    sessions = []

    class Session:
        def __init__(self, _directory):
            self.ready = False
            self.requests = []
            sessions.append(self)

        def configure(self, _capability):
            pass

        def _start(self, _capability):
            pass  # Popen returns before the supervisor serves control requests.

        def _rpc(self, request):
            self.requests.append(request)
            self.ready = True
            return {"jobs": {}}

        def close(self):
            pass

    monkeypatch.setattr(execute, "ExecSession", Session)
    monkeypatch.setattr(execute, "_runtime_closure", lambda: execute.RuntimeClosure((), ()))
    monkeypatch.setattr(execute, "_copy_regular", lambda _source, _target: None)
    monkeypatch.setattr(execute, "_fixed_env_capability", lambda *_args: object())
    monkeypatch.setattr(execute.MutationExecutor, "_write_owner", lambda self: None)
    executor = execute.MutationExecutor(root, network_envelope="none")
    try:
        assert sessions[0].ready
        assert sessions[0].requests == [{"op": "all"}]
    finally:
        executor._ledger.close()


def test_hook_deadline_starts_after_executor_qualification(tmp_path, monkeypatch):
    import pytest
    import kernel_mutate_build
    import kernel_mutate_hooks
    import kernel_mutate_wire
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    clock = [0.0]
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "journal"))

    def qualify():
        clock[0] += 1.5
        return SimpleNamespace(root=tmp_path)

    def stage(*_args):
        clock[0] += 0.1
        return SimpleNamespace(dir=tmp_path)

    observed = []

    def launch(spec):
        observed.append((spec.limits.wall_s, spec.budget_after_ready,
                         spec.trigger_deadline))
        return SimpleNamespace(cleanup=SimpleNamespace(ok=True), timed_out=False,
                               stdout_overflow=False, exit_code=0, stdout=b"")

    runtime._ensure_executor = qualify
    runtime._launch = launch
    monkeypatch.setattr(kernel_mutate_hooks, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(kernel_mutate_build, "stage_unit", stage)
    monkeypatch.setattr(kernel_mutate_wire, "build_stdin_doc", lambda **_kwargs: b"{}")
    monkeypatch.setattr(kernel_mutate_wire, "parse_stdout", lambda *_args, **_kwargs:
                        SimpleNamespace(kind="result", result={"seen": True}))
    result = runtime._hook_manager()._run({"name": "observer", "entry": "main.py",
                                           "output_schema": {"type": "object"},
                                           "source": "def handle(input, ctx): return {}\n"},
                                          "trigger", {}, limit_s=1.0)
    assert result.enabled
    assert observed == [(pytest.approx(0.9), True, pytest.approx(2.5))]


def test_hook_waits_for_payload_marker_before_arming(monkeypatch, capsys):
    import kernel_mutate_exec as execute

    class Handle:
        id = "j1"
        replies = [b"", b"SFM", b"R1\n"]

        def _status(self, _consume):
            return {"output": base64.b64encode(self.replies.pop(0)).decode(),
                    "finished": False}

    class Session:
        def __init__(self):
            self.requests = []

        def _rpc(self, request):
            self.requests.append(request)
            return {"started": 123.0}

    session = Session()
    executor = object.__new__(execute.MutationExecutor)
    executor._session = session
    monkeypatch.setenv("STATEFRAME_DEBUG_HOOK_TIMING", "1")
    executor._await_hook_ready(Handle())
    assert session.requests == []
    assert "ready signal" in capsys.readouterr().err


def test_hook_readiness_rejects_unexpected_output(monkeypatch):
    import pytest
    import kernel_mutate_exec as execute

    class Handle:
        def _status(self, _consume):
            return {"output": base64.b64encode(b"SFMW1 error").decode(),
                    "finished": True}

    executor = object.__new__(execute.MutationExecutor)
    executor._session = SimpleNamespace(_rpc=lambda _request: pytest.fail("armed invalid payload"))
    with pytest.raises(execute.MutationContainmentError, match="readiness protocol"):
        executor._await_hook_ready(Handle())


def test_hook_startup_fields_cannot_be_supplied_by_unit():
    import pytest
    import kernel_mutate_exec as execute

    executor = object.__new__(execute.MutationExecutor)
    executor._closed = False
    executor._seen = set()
    executor.network_envelope = "none"
    spec = execute.LaunchSpec("payload", "call", "a" * 32, None, {}, {},
                              json.dumps({"call": "call", "launch": "a" * 32,
                                          "role": "payload", "hook_ready": True}).encode(),
                              execute.MutationLimits())
    with pytest.raises(ValueError, match="host-reserved"):
        executor.run(spec)


def test_hook_timing_is_opt_in_and_stderr_only(monkeypatch, capsys):
    import kernel_mutate_exec as execute

    monkeypatch.delenv("STATEFRAME_DEBUG_HOOK_TIMING", raising=False)
    execute._hook_timing("qualification begin", at=12.5)
    assert capsys.readouterr() == ("", "")
    monkeypatch.setenv("STATEFRAME_DEBUG_HOOK_TIMING", "1")
    execute._hook_timing("qualification begin", at=12.5)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "[hook-timing] 12.500000000 qualification begin\n"


def test_supervisor_arms_hook_wall_clock_once(monkeypatch):
    import pytest
    import kernel_exec_supervisor as supervisor

    monkeypatch.setattr(supervisor, "time", SimpleNamespace(monotonic=lambda: 123.0))
    job = SimpleNamespace(armed=False, cleanup=None, stop_at=None,
                          armable_wall_s=1.0, started=0.0, wall_seconds=10.0)
    session = object.__new__(supervisor.Supervisor)
    session.jobs = {"j1": job}
    reply = session.dispatch({"op": "arm", "id": "j1"})
    assert reply["started"] == job.started == 123.0
    assert job.armed and job.wall_seconds == 1.0
    with pytest.raises(ValueError, match="cannot be armed"):
        session.dispatch({"op": "arm", "id": "j1"})


def test_supervisor_hook_budget_uses_remaining_trigger_time(monkeypatch):
    import pytest
    import kernel_exec_supervisor as supervisor

    clock = [123.0]
    monkeypatch.setattr(supervisor, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    stopped = []
    job = SimpleNamespace(armed=False, cleanup=None, stop_at=None,
                          armable_wall_s=1.0, started=0.0, wall_seconds=10.0,
                          timed_out=False, timeout_at=None,
                          stop=lambda grace: stopped.append(grace))
    session = object.__new__(supervisor.Supervisor)
    session.jobs = {"j1": job}
    reply = session.dispatch({"op": "arm", "id": "j1", "deadline": 123.4})
    assert reply == {"started": 123.0, "expired": False}
    assert job.wall_seconds == pytest.approx(0.4)
    assert stopped == []

    job.armed = False
    clock[0] = 123.5
    reply = session.dispatch({"op": "arm", "id": "j1", "ready_at": 123.0,
                              "deadline": 124.0})
    assert reply == {"started": 123.0, "expired": False}
    assert job.wall_seconds == 1.0  # Half a second was already spent after payload readiness.

    job.armed = False
    clock[0] = 124.0
    reply = session.dispatch({"op": "arm", "id": "j1", "deadline": 123.4})
    assert reply == {"started": 124.0, "expired": True}
    assert job.wall_seconds == 0.0 and job.timed_out
    assert stopped == [0]

    job.armed = False
    job.timed_out = False
    clock[0] = 125.0
    reply = session.dispatch({"op": "arm", "id": "j1", "ready_at": 123.5,
                              "deadline": 130.0})
    assert reply == {"started": 123.5, "expired": True}
    assert job.wall_seconds == 1.0 and job.timed_out
    assert stopped == [0, 0]


def test_supervisor_timeout_uses_armed_wall_clock(monkeypatch):
    import kernel_exec_supervisor as supervisor

    monkeypatch.setattr(supervisor, "time", SimpleNamespace(monotonic=lambda: 101.1))
    job = object.__new__(supervisor.ManagedProcess)
    job.cleanup = None
    job.eof = True
    job.cap = SimpleNamespace(limits=SimpleNamespace(kill_on_output=False))
    job.family = SimpleNamespace(update=lambda _rows: [], leader_reaped=lambda: None,
                                 signal=lambda *_args: set(),
                                 table=SimpleNamespace(limit_breach=lambda *_args: "",
                                                       snapshot=lambda: {}))
    job.process = SimpleNamespace(poll=lambda: None)
    job.started = 100.0
    job.wall_seconds = 1.0
    job.stop_at = None
    job.kill_at = None
    job.timed_out = False
    job.timeout_at = None
    job.term_sent = set()
    job.killed = set()
    job.tick({})
    assert job.timed_out and job.timeout_at == 101.1
    assert job.stop_at == 101.1


def test_payload_boot_signals_ready_before_release_and_handler(monkeypatch):
    import importlib
    import os
    import sys
    import sf_launch
    import sf_mutate_sdk
    import sf_mutate_boot as boot

    events = []
    doc = {"wire": 1, "call": "hook", "launch": "a" * 32, "nonce": "nonce",
           "role": "payload", "hook_ready": True, "hook_cpu_s": 1.0,
           "input": {}, "grants": {"network": False, "writes": False}}
    monkeypatch.setattr(boot, "_read_doc", lambda: doc)
    monkeypatch.setattr(boot, "_unit_files", lambda _path: ({"entry": "main.py"}, []))
    monkeypatch.setattr(boot, "_close_payload_fds", lambda: None)
    monkeypatch.setattr(boot, "_write_frame", lambda frame: events.append(frame))
    monkeypatch.setattr(boot, "wait_for_release", lambda *_args: events.append("release") or {})
    monkeypatch.setattr(sf_launch, "install_syscall_filter", lambda **_kwargs: None)
    monkeypatch.setattr(importlib, "import_module", lambda _name:
                        SimpleNamespace(handle=lambda *_args: events.append("handler") or {}))
    monkeypatch.setattr(os, "listdir", lambda _path: ["1"])
    monkeypatch.setattr(os, "dup2", lambda *_args: None)
    monkeypatch.setattr(os, "write", lambda _fd, data: events.append(data) or len(data))
    monkeypatch.setattr(boot.resource, "getrusage", lambda _kind:
                        SimpleNamespace(ru_utime=0.1, ru_stime=0.1))
    monkeypatch.setattr(boot.resource, "getrlimit", lambda _kind: (10, 10))
    monkeypatch.setattr(boot.resource, "setrlimit", lambda _kind, bounds:
                        events.append(("cpu", bounds)))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    result = boot.main()
    assert result == 0, events
    assert events[:-1] == [b"SFMR1\n", "release", ("cpu", (2, 2)), "handler"]
    assert events[-1]["kind"] == "result"


def test_collect_reports_first_payload_byte_and_supervisor_timeout_time():
    from kernel_exec_session import ExecSession

    class Handle:
        def _status(self, _consume):
            return {"timed_out": True, "timeout_at": 42.5, "output": base64.b64encode(b"x").decode(),
                    "lost_bytes": 0, "finished": True, "exit_code": -15,
                    "output_total_bytes": 7, "wall_seconds": 1.0,
                    "cleanup": {"ok": True, "killed": 1, "survivors": [], "detail": ""}}

    events = []
    session = object.__new__(ExecSession)
    result = session._collect(Handle(), SimpleNamespace(limits=SimpleNamespace(output_bytes=4096)),
                              on_first_byte=lambda: events.append("first byte"),
                              on_timeout=lambda at: events.append(("timeout", at)))
    assert result.timed_out and result.output == b"x"
    assert events == [("timeout", 42.5), "first byte"]


def test_timed_out_hook_closes_executor_and_sweeps_runtime_root(tmp_path, monkeypatch):
    from kernel_exec_session import CleanupReport
    from kernel_mutate_hooks import HookResult
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    import kernel_mutate_sweep

    root = tmp_path / "runtime" / "run"
    root.mkdir(parents=True)
    ledger = (root / "launches.jsonl").open("wb")
    control, peer = socket.socketpair()

    class Executor:
        def __init__(self):
            self.root = root
            self.closed = False

        def close(self):
            if not self.closed:
                ledger.close()
                control.close()
                peer.close()
                self.closed = True
            return CleanupReport(True, 0, (), "")

    executor = Executor()
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "journal"),
                              runtime_root=root, executor_factory=lambda: executor)
    runtime.executor = executor
    runtime.attach_hook({"name": "observer", "requires": {"network": False, "writes": False},
                         "source": "def handle(input, ctx): return {}\n"}, event="post_tool")
    monkeypatch.setattr(runtime._hook_manager(), "_run", lambda *_args, **_kwargs:
                        HookResult(False, True, CleanupReport(True, 2, (), ""), 0, ""))

    def sweep(parent: Path, *, run_locks):
        assert parent == root.parent and not run_locks(root.name)
        assert executor.closed and ledger.closed and control.fileno() == -1
        shutil.rmtree(root)
        return SimpleNamespace(unresolved=())

    monkeypatch.setattr(kernel_mutate_sweep, "sweep", sweep)
    try:
        result = runtime.invoke_hook(event="post_tool", facts={"tool": "read_file"})
        assert result.timed_out
        assert executor.closed and not root.exists()
    finally:
        executor.close()


def test_hook_timeout_reopens_only_its_swept_root_on_next_use(tmp_path, monkeypatch):
    from kernel_exec_session import CleanupReport
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    import kernel_mutate_exec
    import kernel_mutate_sweep

    root = tmp_path / "runtime" / "run"
    root.mkdir(parents=True)
    created = []

    class Executor:
        def __init__(self, path, *, network_envelope):
            assert path == root and path.is_dir() and network_envelope == "none"
            self.root = path
            created.append(self)

        def close(self):
            return CleanupReport(True, 0, (), "")

    monkeypatch.setattr(kernel_mutate_exec, "MutationExecutor", Executor)

    def sweep(_parent, *, run_locks):
        assert not run_locks(root.name)
        root.rmdir()
        return SimpleNamespace(unresolved=())

    monkeypatch.setattr(kernel_mutate_sweep, "sweep", sweep)
    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path / "journal"),
                              runtime_root=root)
    first = runtime._ensure_executor()
    runtime.retire_hook_executor()
    assert not root.exists()
    second = runtime._ensure_executor()
    assert second is not first and len(created) == 2
    runtime.close()
