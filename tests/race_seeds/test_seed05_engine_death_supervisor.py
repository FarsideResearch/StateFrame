"""Race seed 05: an execution supervisor must detect engine (owner) death; a live owner must never trigger cleanup.

Port of y4race seeds/seed05_engine_death_supervisor.py. Self-contained: no harness imports.

The real Supervisor.serve() loop runs in-process against real child processes. The owner (engine) holds the control
socket's peer and the owner pipe. A grandchild holder inherits only the control socket, so it outlives the owner.
- 05a (marked, known-bad f0568448): the owner dies as an unreaped zombie. Cleanup of the job family must start.
- 05b (companion): the owner is alive. Cleanup must not start.

Signals go only to processes this test spawned. The holder is matched by pid and start time before it is killed.
"""
from __future__ import annotations

import importlib
import inspect
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

KNOWN_BAD_SHA = "f0568448eb97ccc978a26d565090c6b67ef19f96"
POLL_WINDOW_SECONDS = 3.0  # supervisor polls every 20 ms, so this is about 150 polls
READY_WAIT_SECONDS = 5.0
THREAD_JOIN_SECONDS = 10.0
# Owner script: fork the holder first (the holder closes the owner pipe write end, as fork writers do), report its
# pid, then either exit (owner death, leaving a zombie) or stay alive.
OWNER_SCRIPT = r"""
import os, sys, time
control_fd, owner_fd, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
pid = os.fork()
if pid == 0:
    os.close(owner_fd)
    time.sleep(120)
    os._exit(0)
print(pid, flush=True)
if mode == "exit":
    os._exit(0)
time.sleep(120)
"""
JOB_SCRIPT = "import time; time.sleep(120)"


def _stat(pid: int) -> tuple[str, int] | None:
    """(state, start ticks) read straight from /proc, or None when the pid is gone. Independent of product probes."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rfind(")") + 2:].split()
    return fields[0], int(fields[19])


def _fd_targets(pid: int) -> set[str]:
    try:
        return {os.readlink(f"/proc/{pid}/fd/{name}") for name in os.listdir(f"/proc/{pid}/fd")}
    except OSError:
        return set()


def _kill_own_child(process: subprocess.Popen[Any]) -> None:
    process.kill()
    process.wait(timeout=THREAD_JOIN_SECONDS)
    if process.stdout is not None:
        process.stdout.close()


def _scenario(owner_mode: str) -> dict[str, Any]:
    supervisor_module = importlib.import_module("kernel_exec_supervisor")
    tracker = importlib.import_module("kernel_exec_tracker")
    capability = importlib.import_module("kernel_exec_capability")
    output = importlib.import_module("kernel_job_output")
    owner: subprocess.Popen[Any] | None = None
    job_process: subprocess.Popen[Any] | None = None
    holder: int | None = None
    holder_start: int | None = None
    thread: threading.Thread | None = None
    handed_owner_pipe = False
    errors: list[BaseException] = []
    with tempfile.TemporaryDirectory(prefix="race-seed05-") as tmp:
        srv, cli = socket.socketpair()
        owner_r, owner_w = os.pipe()
        try:
            owner = subprocess.Popen(
                [sys.executable, "-I", "-c", OWNER_SCRIPT, str(cli.fileno()), str(owner_w), owner_mode],
                pass_fds=(cli.fileno(), owner_w), stdout=subprocess.PIPE, close_fds=True,
            )
            cli.close()
            os.close(owner_w)  # harness copy must go, or the owner-pipe EOF can never reach the supervisor
            ready, _, _ = select.select([owner.stdout], [], [], READY_WAIT_SECONDS)
            if not ready:
                raise RuntimeError("owner never reported its holder pid")
            holder = int(owner.stdout.readline())  # type: ignore[union-attr]
            holder_stat = _stat(holder)
            if holder_stat is None:
                raise RuntimeError("holder vanished before the scenario started")
            holder_start = holder_stat[1]
            # Isolation: the holder keeps the control socket but never the owner-pipe write end.
            pipe_name = os.readlink(f"/proc/self/fd/{owner_r}")
            holder_fds = _fd_targets(holder)
            if pipe_name in holder_fds or not any(t.startswith("socket:") for t in holder_fds):
                raise RuntimeError("holder fd set is not the intended control-socket-only holder")

            supervisor_kwargs: dict[str, Any] = {}
            if "owner_fd" in inspect.signature(supervisor_module.Supervisor).parameters:
                supervisor_kwargs["owner_fd"] = owner_r
                handed_owner_pipe = True
            supervisor = supervisor_module.Supervisor(srv, Path(tmp), "bwrap", "host", **supervisor_kwargs)

            job_process = subprocess.Popen([sys.executable, "-I", "-c", JOB_SCRIPT], stdin=subprocess.DEVNULL,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
            cap = SimpleNamespace(limits=capability.ExecLimits(), purpose="job")
            collector = output.JobOutputCollector(Path(tmp) / "j1.output", capacity=1 << 20)
            job = supervisor_module.ManagedProcess("j1", job_process, collector, cap, tracker.ProcessTable())
            supervisor.jobs["j1"] = job

            def serve() -> None:
                try:
                    supervisor.serve()
                except BaseException as error:  # noqa: BLE001 - surfaced as a harness fault below
                    errors.append(error)

            thread = threading.Thread(target=serve, name="race-seed05-supervisor", daemon=True)
            thread.start()

            if owner_mode == "exit":
                deadline = time.monotonic() + READY_WAIT_SECONDS
                while (state := _stat(owner.pid)) is None or state[0] != "Z":
                    if time.monotonic() >= deadline:
                        raise RuntimeError("owner never became a zombie")
                    time.sleep(0.01)
                holder_now = _stat(holder)
                if holder_now is None or holder_now[0] == "Z":
                    raise RuntimeError("holder died with the owner; scenario does not isolate the control socket")

            started_at = time.monotonic()
            cleanup_started = False
            while time.monotonic() - started_at < POLL_WINDOW_SECONDS:
                if job.stop_at is not None or job.cleanup is not None:  # cleanup began (stop) or finished
                    cleanup_started = True
                    break
                if errors:
                    break
                time.sleep(0.01)
            owner_state = _stat(owner.pid)
            if owner_mode == "live" and (owner_state is None or owner_state[0] == "Z"):
                raise RuntimeError("live-owner scenario lost its owner before the window ended")
            if errors:
                raise RuntimeError(f"supervisor raised: {errors[0]!r}")
            job_state = _stat(job_process.pid)
            return {
                "cleanup_started": cleanup_started,
                "owner_state": owner_state[0] if owner_state else "gone",
                "job_live": job_state is not None and job_state[0] != "Z",
                "serving": thread.is_alive(),
            }
        finally:
            # Signal only what this test spawned. The holder is ours by pid, checked by start time first.
            if holder is not None and holder_start is not None:
                current = _stat(holder)
                if current is not None and current[1] == holder_start and current[0] != "Z":
                    os.kill(holder, signal.SIGKILL)
            if owner is not None:
                _kill_own_child(owner)
            if job_process is not None:
                _kill_own_child(job_process)
            if thread is not None:
                thread.join(THREAD_JOIN_SECONDS)
                if thread.is_alive():
                    raise RuntimeError("supervisor did not exit after its peers were killed")
            if not handed_owner_pipe:
                os.close(owner_r)  # on products with owner_fd the supervisor owns and closes this descriptor
            srv.close()


@pytest.mark.race_seed(
    "05",
    known_bad=KNOWN_BAD_SHA,
    expect="but no cleanup started within",
)
def test_05a_zombie_owner_starts_family_cleanup() -> None:
    result = _scenario("exit")
    if not result["cleanup_started"]:
        raise AssertionError(
            f"owner is a zombie (state {result['owner_state']}) but no cleanup started within "
            f"{POLL_WINDOW_SECONDS:g}s; job family live={result['job_live']}; control socket still held "
            f"by a surviving holder"
        )


def test_05b_live_owner_does_not_start_cleanup() -> None:
    result = _scenario("live")
    if result["cleanup_started"]:
        raise AssertionError(f"live owner (state {result['owner_state']}) triggered cleanup")
    # Non-vacuous: the supervisor must still be serving and the job family still alive at window end.
    if not result["serving"] or not result["job_live"]:
        raise AssertionError(
            f"live owner: supervisor serving={result['serving']} job_live={result['job_live']} "
            f"(stopped without cleanup)"
        )
