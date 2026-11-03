"""Fresh-interpreter adapter to the engine's bounded execution policy.

The CLI imports only this stdlib adapter. The execution policy is imported
only in the helper, after installing host test policy. The caller loads the
process-identity tracker lazily when it starts a check.
"""
from __future__ import annotations

import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CHECK_TIMEOUT = 900.0
HELPER_PROCESS_CLEANUP_S = 4.0
HELPER_CLEANUP_GRACE = HELPER_PROCESS_CLEANUP_S + 3.0
OUTPUT_LIMIT = 256 * 1024
SANDBOX_FORK_RETRIES = 2  # extra attempts after a transient bwrap fork EAGAIN
# JSON may escape each retained byte as six ASCII characters; task mode also
# sends a second copy of the final 4 KiB in the legacy tail field.
WIRE_LIMIT = (OUTPUT_LIMIT + 4000) * 6 + 16384
PASS_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT")


class OuterCheckError(RuntimeError):
    """A check could not be executed within the required boundary."""


def run(command: str, cwd: Path, *, python: str, timeout_s: float = CHECK_TIMEOUT,
        deadline: float | None = None, test_env: dict[str, str] | None = None,
        pytest_plugins: str | None = None, test_python: str | None = None,
        runner_kind: str = "pytest", capture_output: bool = False) -> dict[str, object]:
    """Copy and check in a fresh helper, bounding both IPC and execution time.

    ``test_python`` is the interpreter the check command runs. The engine's sandbox admits only the test
    interpreter it was told about (STATEFRAME_TEST_PYTHON), else its own; without this the sandbox refused a
    Homebrew python3 chosen for a repository with no venv (10-28: "execvp() ... Operation not permitted").
    """
    started = time.monotonic()
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise OuterCheckError("check timeout must be finite and positive")
    if deadline is not None and not math.isfinite(deadline):
        raise OuterCheckError("run deadline must be finite")
    end = min(started + min(timeout_s, CHECK_TIMEOUT), deadline if deadline is not None else math.inf)
    if end <= started:
        raise OuterCheckError("run deadline exhausted before outer check")
    if pytest_plugins not in (None, "installed"):
        raise OuterCheckError("--pytest-plugins only accepts 'installed'")
    if runner_kind not in {"pytest", "generic"}:
        raise OuterCheckError("check runner kind must be pytest or generic")
    spec = {"command": command, "cwd": str(cwd.resolve()), "deadline": end,
            "test_env": test_env or {}, "pytest_plugins": pytest_plugins, "test_python": test_python,
            "runner_kind": runner_kind, "capture_output": capture_output}
    env = {key: os.environ[key] for key in PASS_ENV if key in os.environ}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # A file-backed input avoids a pipe writer thread and cannot deadlock at startup.
    with tempfile.TemporaryFile(dir=cwd.parent) as request:
        request.write(json.dumps(spec).encode())
        request.seek(0)
        process = subprocess.Popen([python, "-I", "-B", str(Path(__file__).resolve()), "--helper"],
                                   cwd=Path(__file__).resolve().parent, env=env, stdin=request,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
    from .engine.kernel_exec_tracker import (capture_launch_identity, child_exited_unreaped,
                                             recorded_launch_identity, signal_launch_group)
    capture_launch_identity(process, session_leader=True)
    cleanup_ack = False
    helper_error = None
    helper_exit = None
    result = None
    pending = bytearray()
    received = 0
    diagnostic = bytearray()
    timed_out = False
    drain_deadline = end
    with selectors.DefaultSelector() as selector:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        try:
            while selector.get_map():
                remaining = drain_deadline - time.monotonic()
                if remaining <= 0:
                    if timed_out:
                        break
                    timed_out = True
                    # Ask the helper to unwind run_sandboxed so it cleans up its
                    # separate test process group and its throwaway directory.
                    if not child_exited_unreaped(process):
                        process.send_signal(signal.SIGTERM)
                    drain_deadline = time.monotonic() + HELPER_CLEANUP_GRACE
                    continue
                for key, _ in selector.select(min(remaining, 0.05)):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        diagnostic.extend(data)
                        del diagnostic[:-4000]
                        continue
                    pending.extend(data)
                    received += len(data)
                    if received > WIRE_LIMIT:
                        raise OuterCheckError("outer check helper exceeded its output bound")
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending[:] = rest
                        packet = json.loads(line)
                        if "cleanup_ok" in packet:
                            cleanup_ack = packet["cleanup_ok"] is True
                        elif "result" in packet:
                            result = packet["result"]
                        elif "error" in packet:
                            if not timed_out:
                                helper_error = packet["error"]
            if not timed_out:
                process.wait(timeout=max(0.001, end - time.monotonic()))
                if process.returncode:
                    helper_exit = process.returncode
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            raise OuterCheckError(f"outer check helper failed: {type(error).__name__}") from error
        finally:
            if not timed_out and not child_exited_unreaped(process):
                process.send_signal(signal.SIGTERM)
            cleanup_deadline = (drain_deadline if timed_out
                                else time.monotonic() + HELPER_CLEANUP_GRACE)
            while not child_exited_unreaped(process) and time.monotonic() < cleanup_deadline:
                time.sleep(.02)
            if not child_exited_unreaped(process):
                try:
                    signalled = signal_launch_group(process, recorded_launch_identity(process),
                                                    signal.SIGKILL)
                except PermissionError:
                    signalled = False
                if not signalled:
                    process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired as error:
                raise OuterCheckError("outer check helper survived cleanup") from error
            process.stdout.close()
            process.stderr.close()
    if not cleanup_ack:
        raise OuterCheckError("outer check containment unverified: helper did not acknowledge cleanup")
    if helper_error is not None:
        raise OuterCheckError(f"outer check refused: {helper_error}")
    if helper_exit is not None:
        raise OuterCheckError(f"outer check helper failed (exit {helper_exit})")
    if result is None:
        if not timed_out:
            detail = diagnostic.decode(errors="replace").strip()
            raise OuterCheckError(f"outer check helper failed (exit {process.returncode}): {detail}")
        result = {"exit_code": None, "passed": False, "tail": "check deadline exhausted",
                  "output_bytes": 0, "output_truncated": False}
    result.update(command=command, seconds=round(time.monotonic() - started, 2))
    if timed_out:
        result.update(exit_code=None, passed=False, timed_out=True)
    return result


def _helper() -> int:
    """Only this fresh process imports engine policy and runs repository code."""
    spec = json.load(sys.stdin)
    # Match _child.install_environment: no inherited engine switches; host test
    # values must be installed BEFORE importing the policy's environment snapshot.
    env = {key: os.environ[key] for key in PASS_ENV if key in os.environ}
    env["PYTHONHASHSEED"] = "0"
    env.update(spec["test_env"])
    env["STATEFRAME_TASK_ENV_KEYS"] = ",".join(sorted(spec["test_env"]))
    if spec.get("test_python"):
        # PASS_ENV deliberately drops STATEFRAME_* switches; this one is the runner's own contract with the engine.
        env["STATEFRAME_TEST_PYTHON"] = spec["test_python"]
    if spec["pytest_plugins"]:
        env["STATEFRAME_TASK_PYTEST_PLUGINS"] = spec["pytest_plugins"]
    os.environ.clear()
    os.environ.update(env)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
    import kernel_execution_policy as policy

    spawned = None
    scratch = None
    process_cleaned = False
    cleanup_ok = True
    original_spawn = policy.sandboxed_popen
    original_kill_group = policy._kill_group

    def tracked_spawn(*args, **kwargs):
        nonlocal spawned, process_cleaned
        spawned, capabilities = original_spawn(*args, **kwargs)
        process_cleaned = False
        return spawned, capabilities

    def tracked_kill_group(process):
        nonlocal process_cleaned
        cleaned = original_kill_group(process)
        if process is spawned and cleaned and getattr(process, "returncode", None) is not None:
            process_cleaned = True
        return cleaned

    policy.sandboxed_popen = tracked_spawn
    policy._kill_group = tracked_kill_group

    def stop(signum, frame):
        raise InterruptedError("outer check deadline exhausted")

    signal.signal(signal.SIGTERM, stop)
    try:
        policy.require_capabilities()
        scratch = tempfile.TemporaryDirectory(prefix="stateframe-tests-", dir=Path(spec["cwd"]).parent)
        copy = Path(scratch.name) / "repo"
        shutil.copytree(spec["cwd"], copy, symlinks=True)
        remaining = spec["deadline"] - time.monotonic()
        if remaining <= 0:
            raise OuterCheckError("run deadline exhausted copying outer check")
        # Reserve a small part of this SAME deadline for serializing the
        # bounded result, not an extra check allowance.
        check_budget = remaining - min(0.1, remaining / 2)
        for attempt in range(SANDBOX_FORK_RETRIES + 1):
            result = policy.run_sandboxed(shlex.split(spec["command"]), copy,
                                          env=env, timeout=min(CHECK_TIMEOUT, check_budget),
                                          cpu_seconds=min(CHECK_TIMEOUT, remaining),
                                          output_limit=OUTPUT_LIMIT,
                                          generic_check=spec["runner_kind"] == "generic")
            process_cleaned = process_cleaned or (result.process_group_killed and spawned is not None
                                                  and getattr(spawned, "returncode", None) is not None)
            # bwrap's own clone() can hit a transient EAGAIN on a loaded host; that is
            # "try again", not "no sandbox". Retry briefly within the same deadline only.
            stderr = result.completed.stderr
            transient = (result.completed.returncode and stderr.startswith("bwrap:")
                         and "Resource temporarily unavailable" in stderr)
            if not transient or attempt == SANDBOX_FORK_RETRIES:
                break
            pause = 0.5 * (attempt + 1)
            if spec["deadline"] - time.monotonic() <= pause + 1:
                break
            time.sleep(pause)
        sandbox_error = result.completed.stderr
        missing = re.match(r"bwrap: execvp (\S+): No such file or directory", sandbox_error)
        if result.completed.returncode and missing:
            # The sandbox started; the check program itself is absent (for example `make`
            # not installed). Say so instead of blaming the sandbox.
            raise OuterCheckError(f"check command not found: {missing.group(1)}; install it "
                                  "or change the test command")
        if result.completed.returncode and (
            sandbox_error.startswith("bwrap:") or
            (sandbox_error.startswith("sandbox-exec:") and "sandbox_apply" in sandbox_error)
        ):
            raise OuterCheckError("sandbox unavailable: " + result.completed.stderr[:1000])
        output = (result.completed.stdout + result.completed.stderr).encode("utf-8")
        payload = {"exit_code": None if result.timed_out else result.completed.returncode,
                   "passed": not result.timed_out and result.completed.returncode == 0,
                   "tail": output[-4000:].decode("utf-8", errors="ignore"),
                   "timed_out": result.timed_out, "output_truncated": result.output_truncated,
                   "output_bytes": len(output),
                   "process_group_killed": result.process_group_killed}
        if spec.get("capture_output"):
            payload["output"] = output.decode("utf-8", errors="replace")
        print(json.dumps({"result": payload}), flush=True)
    except (RuntimeError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), flush=True)
        return 2
    finally:
        policy._kill_group = original_kill_group
        def cleanup_timeout(_signum, _frame):
            raise TimeoutError("outer check process cleanup exceeded its budget")

        previous_alarm = signal.signal(signal.SIGALRM, cleanup_timeout)
        signal.setitimer(signal.ITIMER_REAL, HELPER_PROCESS_CLEANUP_S)
        if spawned is not None and not process_cleaned:
            try:
                cleanup_ok = policy._kill_group(spawned) and cleanup_ok
                spawned.wait(timeout=1)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                cleanup_ok = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        print(json.dumps({"cleanup_ok": cleanup_ok}), flush=True)
        if scratch is not None:
            scratch.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper())
