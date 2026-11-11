"""Execution supervisor lifetime controls for wrapped launch chains."""
from __future__ import annotations

import json
import base64
import os
from pathlib import Path
from dataclasses import replace
import select
import signal
import socket
import subprocess
import sys
import time

import pytest

from _timeouts import scaled
from kernel_exec_capability import shell_capability
from kernel_exec_tracker import ProcessTable
from fixtures.process_signals import (
    close_signal_binding,
    record_identity,
    retain_signal_binding,
    signal_bound,
    signal_owned,
)

pytestmark = pytest.mark.host_sandbox


def _read_reply(control: socket.socket, buffer: bytearray, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + scaled(timeout)
    while b"\n" not in buffer:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([control], [], [], max(0.0, remaining))[0]:
            raise AssertionError("execution supervisor did not answer before the deadline")
        chunk = control.recv(4096)
        if not chunk:
            raise AssertionError("execution supervisor exited while its intermediary was alive")
        buffer.extend(chunk)
    line, _, rest = buffer.partition(b"\n")
    buffer[:] = rest
    return json.loads(line)


def test_supervisor_stays_alive_when_started_through_intermediate_process(tmp_path):
    root = Path(__file__).resolve().parents[1]
    supervisor_script = root / "src/stateframe/engine/kernel_exec_supervisor.py"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    control, supervisor_input = socket.socketpair()
    owner_read, owner_write = os.pipe()
    intermediate_source = """
import subprocess, sys
fd, script, run_dir, owner_fd = sys.argv[1:]
child = subprocess.Popen(
    [sys.executable, "-I", script, run_dir, "sandbox-exec", "host", owner_fd],
    stdin=int(fd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
    start_new_session=True, pass_fds=(int(owner_fd),),
)
print(child.pid, flush=True)
child.wait()
"""
    supervisor_launched_after = time.time()
    intermediate = subprocess.Popen(
        [sys.executable, "-c", intermediate_source, str(supervisor_input.fileno()),
         str(supervisor_script), str(run_dir), str(owner_read)],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, close_fds=True, pass_fds=(supervisor_input.fileno(), owner_read), start_new_session=True,
    )
    supervisor_input.close()
    os.close(owner_read)
    supervisor_pid = None
    supervisor_identity = None
    supervisor_binding = None
    replies = bytearray()
    try:
        assert intermediate.stdout is not None
        if not select.select([intermediate.stdout], [], [], scaled(5))[0]:
            raise AssertionError("intermediate did not report the supervisor pid")
        supervisor_pid = int(intermediate.stdout.readline())
        supervisor_identity = record_identity(
            supervisor_pid, launched_after=supervisor_launched_after,
            command_fragment="kernel_exec_supervisor.py")
        supervisor_binding = retain_signal_binding(supervisor_identity)
        # The control and owner channels survive the wrapper boundary.
        assert intermediate.pid != os.getpid()
        control.settimeout(scaled(3))

        control.sendall(b'{"op":"all","rid":1}\n')
        assert _read_reply(control, replies) == {"jobs": {}, "rid": 1}

        control.sendall(b'{"op":"close","grace":0,"rid":2}\n')
        assert _read_reply(control, replies) == {"jobs": {}, "rid": 2}
        intermediate.wait(timeout=scaled(5))
    finally:
        control.close()
        os.close(owner_write)
        owner_write = -1
        if intermediate.poll() is None:
            try:
                intermediate.wait(timeout=scaled(3))
            except subprocess.TimeoutExpired:
                if supervisor_binding is not None:
                    try:
                        signal_bound(supervisor_binding, signal.SIGKILL)
                    except (ProcessLookupError, RuntimeError):
                        pass
                if intermediate.poll() is None:
                    signal_owned(intermediate, signal.SIGTERM)
                try:
                    intermediate.wait(timeout=scaled(3))
                except subprocess.TimeoutExpired:
                    signal_owned(intermediate, signal.SIGKILL)
                    intermediate.wait(timeout=scaled(3))
        if intermediate.stdout is not None:
            intermediate.stdout.close()
        if intermediate.stderr is not None:
            intermediate.stderr.close()
        if supervisor_binding is not None:
            close_signal_binding(supervisor_binding)


def _rpc(control, replies, rid, **request):
    control.sendall(json.dumps({**request, "rid": rid}, separators=(",", ":")).encode() + b"\n")
    return _read_reply(control, replies)


def _start_wrapped_supervisor(tmp_path: Path, *, exits_before_init: bool):
    run_dir = tmp_path / ("run-before" if exits_before_init else "run-after")
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    control, supervisor_input = socket.socketpair()
    owner_read, owner_write = os.pipe()
    release_read = release_write = None
    pid_file = run_dir / "supervisor.pid"
    command = [sys.executable, "-I", str(Path(__file__).resolve().parents[1] /
                                           "src/stateframe/engine/kernel_exec_supervisor.py"),
               str(run_dir), "sandbox-exec", "host", str(owner_read)]
    if exits_before_init:
        wrapper_source = """
import json, os, sys, time
command = json.loads(sys.argv[1])
pid_file = sys.argv[2]
pid = os.fork()
if pid == 0:
    time.sleep(0.35)
    os.execvpe(command[0], command, os.environ)
    os._exit(127)
with open(pid_file, "w") as stream: stream.write(str(pid))
os._exit(0)
"""
        wrapper_args = [sys.executable, "-c", wrapper_source, json.dumps(command), str(pid_file)]
    else:
        release_read, release_write = os.pipe()
        wrapper_source = """
import json, os, subprocess, sys
command = json.loads(sys.argv[1])
release_fd = int(sys.argv[2])
pid_file = sys.argv[3]
child = subprocess.Popen(command, stdin=0, stdout=subprocess.DEVNULL, stderr=None,
                         close_fds=True, pass_fds=(int(command[-1]),))
with open(pid_file, "w") as stream: stream.write(str(child.pid))
os.read(release_fd, 1)
os._exit(0)
"""
        wrapper_args = [sys.executable, "-c", wrapper_source, json.dumps(command),
                        str(release_read), str(pid_file)]
    passed = [supervisor_input.fileno(), owner_read]
    if release_read is not None:
        passed.append(release_read)
    wrapper = subprocess.Popen(wrapper_args, stdin=supervisor_input, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, close_fds=True, pass_fds=tuple(passed),
                               start_new_session=True)
    supervisor_input.close()
    os.close(owner_read)
    if release_read is not None:
        os.close(release_read)
    return wrapper, control, owner_write, release_write, workspace, pid_file


def _start_test_job(control, replies, workspace, *, first_rid: int = 1):
    (workspace.parent / "env").mkdir()
    (workspace.parent / "home").mkdir()
    cap = shell_capability(workspace=workspace, run_env_dir=workspace.parent / "env",
                           scratch_home=workspace.parent / "home", network="none", writable=False)
    cap = replace(cap, purpose="job", workspace_writable=False, sandboxed=False)
    started = _rpc(control, replies, first_rid, op="start",
                   argv=[sys.executable, "-c",
                         "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                         "print(os.getpid(), flush=True); time.sleep(60)"],
                   cap=cap.as_dict(), cwd=str(workspace))
    job_id = started["id"]
    output = bytearray()
    for rid in range(first_rid + 1, first_rid + 80):
        status = _rpc(control, replies, rid, op="status", id=job_id, consume=True)
        output.extend(base64.b64decode(status["output"]))
        if b"\n" in output:
            return job_id, int(output.partition(b"\n")[0]), rid + 1
        time.sleep(0.02)
    raise AssertionError("supervised job did not publish its pid")


def _wait_gone(pid: int, timeout: float = 5.0) -> None:
    table = ProcessTable()
    original = table.identity(pid)
    if original is None or original.zombie:
        return
    deadline = time.monotonic() + scaled(timeout)
    while time.monotonic() < deadline:
        current = table.identity(pid)
        if current is None or current.start != original.start or current.zombie:
            return
        time.sleep(0.02)
    raise AssertionError(f"process {pid} survived supervisor cleanup")


def test_short_lived_wrapper_exits_before_init_and_owner_eof_stops_job(tmp_path):
    wrapper, control, owner_write, _, workspace, pid_file = _start_wrapped_supervisor(
        tmp_path, exits_before_init=True)
    replies = bytearray()
    supervisor_pid = None
    job_pid = None
    try:
        assert wrapper.wait(timeout=scaled(5)) == 0
        supervisor_pid = int(pid_file.read_text(encoding="ascii"))
        _, job_pid, _ = _start_test_job(control, replies, workspace)
        os.close(owner_write)
        owner_write = -1
        _wait_gone(job_pid)
        _wait_gone(supervisor_pid)
    finally:
        control.close()
        if owner_write >= 0:
            os.close(owner_write)
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=scaled(5))
        if job_pid is not None:
            _wait_gone(job_pid)
        if supervisor_pid is not None:
            _wait_gone(supervisor_pid)


def test_wrapper_exit_after_init_does_not_stop_live_owner_session(tmp_path):
    wrapper, control, owner_write, release_write, workspace, pid_file = _start_wrapped_supervisor(
        tmp_path, exits_before_init=False)
    replies = bytearray()
    supervisor_pid = None
    job_pid = None
    try:
        deadline = time.monotonic() + scaled(5)
        while not pid_file.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        supervisor_pid = int(pid_file.read_text(encoding="ascii"))
        _, job_pid, next_rid = _start_test_job(control, replies, workspace)
        os.write(release_write, b"x")
        os.close(release_write)
        release_write = -1
        assert wrapper.wait(timeout=scaled(5)) == 0
        status = _rpc(control, replies, next_rid, op="status", id="j1")
        assert status["finished"] is False
        assert status["exit_code"] is None
        assert ProcessTable().identity(job_pid) is not None
        _rpc(control, replies, next_rid + 1, op="close", grace=0)
        _wait_gone(job_pid)
        _wait_gone(supervisor_pid)
        os.close(owner_write)
        owner_write = -1
    finally:
        control.close()
        if owner_write >= 0:
            os.close(owner_write)
        if release_write >= 0:
            os.close(release_write)
        if wrapper.poll() is None:
            wrapper.kill()
            wrapper.wait(timeout=scaled(5))
        if job_pid is not None:
            _wait_gone(job_pid)
        if supervisor_pid is not None:
            _wait_gone(supervisor_pid)
