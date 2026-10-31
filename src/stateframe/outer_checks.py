"""Fresh-interpreter adapter to the engine's bounded execution policy.

The CLI imports only this stdlib adapter. Engine imports happen exclusively in
the helper, after installing host test policy, never in the long-lived caller.
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
OUTPUT_LIMIT = 256 * 1024
SANDBOX_FORK_RETRIES = 2  # extra attempts after a transient bwrap fork EAGAIN
# JSON may escape each retained byte as six ASCII characters.
WIRE_LIMIT = OUTPUT_LIMIT * 6 + 16384
PASS_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT")


class OuterCheckError(RuntimeError):
    """A check could not be executed within the required boundary."""


def _kill_group(pid: int | None) -> None:
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run(command: str, cwd: Path, *, python: str, timeout_s: float = CHECK_TIMEOUT,
        deadline: float | None = None, test_env: dict[str, str] | None = None,
        pytest_plugins: str | None = None, test_python: str | None = None) -> dict[str, object]:
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
    spec = {"command": command, "cwd": str(cwd.resolve()), "deadline": end,
            "test_env": test_env or {}, "pytest_plugins": pytest_plugins, "test_python": test_python}
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
    check_pid = None
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
                    if process.poll() is None:
                        process.send_signal(signal.SIGTERM)
                    _kill_group(check_pid)
                    drain_deadline = time.monotonic() + 0.5
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
                        if "check_pid" in packet:
                            check_pid = int(packet["check_pid"])
                        elif "result" in packet:
                            result = packet["result"]
                        elif "error" in packet:
                            if not timed_out:
                                raise OuterCheckError(f"outer check refused: {packet['error']}")
            if not timed_out:
                process.wait(timeout=max(0.001, end - time.monotonic()))
                if process.returncode:
                    raise OuterCheckError(f"outer check helper failed (exit {process.returncode})")
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            raise OuterCheckError(f"outer check helper failed: {type(error).__name__}") from error
        finally:
            # The engine starts checks in their own session. Killing only the
            # helper's group would leave those checks (and grandchildren) alive.
            _kill_group(check_pid)
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
            _kill_group(process.pid)
            process.wait(timeout=1)
            process.stdout.close()
            process.stderr.close()
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
    original_spawn = policy.sandboxed_popen

    def tracked_spawn(*args, **kwargs):
        nonlocal spawned
        spawned, capabilities = original_spawn(*args, **kwargs)
        print(json.dumps({"check_pid": spawned.pid}), flush=True)
        return spawned, capabilities

    policy.sandboxed_popen = tracked_spawn

    def stop(signum, frame):
        raise InterruptedError("outer check deadline exhausted")

    signal.signal(signal.SIGTERM, stop)
    try:
        policy.require_capabilities()
        with tempfile.TemporaryDirectory(prefix="stateframe-tests-", dir=Path(spec["cwd"]).parent) as scratch:
            copy = Path(scratch) / "repo"
            shutil.copytree(spec["cwd"], copy, symlinks=True)
            remaining = spec["deadline"] - time.monotonic()
            if remaining <= 0:
                raise OuterCheckError("run deadline exhausted copying outer check")
            # Reserve a small part of this SAME deadline for serializing the
            # bounded result and removing the copy, not an extra check allowance.
            check_budget = remaining - min(0.1, remaining / 2)
            try:
                for attempt in range(SANDBOX_FORK_RETRIES + 1):
                    result = policy.run_sandboxed(shlex.split(spec["command"]), copy,
                                                  env=env, timeout=min(CHECK_TIMEOUT, check_budget),
                                                  cpu_seconds=min(CHECK_TIMEOUT, remaining),
                                                  output_limit=OUTPUT_LIMIT)
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
            finally:
                # Reap surviving descendants before deleting their writable root,
                # including a check whose leader exited successfully.
                if spawned is not None:
                    policy._kill_group(spawned)
                    spawned.wait(timeout=1)
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
            print(json.dumps({"result": payload}), flush=True)
    except (RuntimeError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), flush=True)
        return 2
    finally:
        if spawned is not None:
            policy._kill_group(spawned)
            spawned.wait(timeout=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper())
