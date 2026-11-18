"""Fresh-interpreter adapter to the engine's bounded execution policy.

The CLI imports only this stdlib adapter. The execution policy is imported
only in the helper, after installing host test policy. The caller loads the
process-identity tracker lazily when it starts a check.
"""
from __future__ import annotations

from dataclasses import replace

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


CHECK_TIMEOUT = 300.0  # compatibility name: inactivity, never total lifetime
HELPER_PROCESS_CLEANUP_S = 4.0
HELPER_CLEANUP_GRACE = HELPER_PROCESS_CLEANUP_S + 3.0
HELPER_DRAIN_GRACE = 2.0
OUTPUT_LIMIT = 256 * 1024
SANDBOX_FORK_RETRIES = 2  # extra attempts after a transient bwrap fork EAGAIN
# JSON may escape each retained byte as six ASCII characters; task mode also
# sends a second copy of the final 4 KiB in the legacy tail field.
WIRE_LIMIT = (OUTPUT_LIMIT + 4000) * 6 + 16384
PASS_ENV = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT")


class OuterCheckError(RuntimeError):
    """A check could not be executed within the required boundary."""


class HelperPackets:
    """Bound individual IPC records and result volume independently of progress."""

    packet_limit = WIRE_LIMIT
    progress_limit = 1024

    def __init__(self):
        self.pending = bytearray()
        self.result_bytes = 0
        self.completed_work = 0
        self.sequence = 0
        self.counters: dict[str, int] = {}

    def feed(self, data: bytes) -> list[dict[str, object]]:
        self.pending.extend(data)
        packets = []
        while b"\n" in self.pending:
            line, _, rest = self.pending.partition(b"\n")
            self.pending[:] = rest
            if len(line) > self.packet_limit:
                raise OuterCheckError("outer check helper exceeded its packet bound")
            try:
                packet = json.loads(line)
            except (ValueError, UnicodeError) as error:
                raise OuterCheckError("malformed outer check helper packet") from error
            if not isinstance(packet, dict) or len(packet) != 1:
                raise OuterCheckError("malformed outer check helper packet")
            if "progress" in packet:
                work = packet["progress"]
                if (len(line) > self.progress_limit or not isinstance(work, dict)
                        or set(work) != {"sequence", "kind", "completed"}):
                    raise OuterCheckError("malformed outer check progress packet")
                seq, kind, completed = work["sequence"], work["kind"], work["completed"]
                if (type(seq) is not int or not self.sequence < seq <= 2**63 - 1
                        or not isinstance(kind, str) or kind not in {"copy", "check", "serialize"}
                        or type(completed) is not int
                        or not self.counters.get(kind, 0) < completed <= 2**63 - 1):
                    raise OuterCheckError("invalid outer check progress evidence")
                self.sequence = seq
                self.completed_work += completed - self.counters.get(kind, 0)
                self.counters[kind] = completed
            else:
                if not set(packet) <= {"result", "error", "cleanup_ok"}:
                    raise OuterCheckError("unknown outer check helper packet")
                self.result_bytes += len(line) + 1
                if self.result_bytes > WIRE_LIMIT:
                    raise OuterCheckError("outer check helper exceeded its output bound")
                if "result" in packet and not isinstance(packet["result"], dict):
                    raise OuterCheckError("invalid outer check result")
                packets.append(packet)
        if len(self.pending) > self.packet_limit:
            raise OuterCheckError("outer check helper exceeded its packet bound")
        return packets


class _ProgressEmitter:
    """Send at most two work packets per second; never send idle heartbeats."""

    def __init__(self):
        self.sequence = 0
        self.counts: dict[str, int] = {}
        self.sent: dict[str, int] = {}
        self.next_emit = 0.0

    def advance(self, kind: str, amount: int = 1) -> None:
        if amount <= 0:
            return
        self.counts[kind] = self.counts.get(kind, 0) + amount
        now = time.monotonic()
        if now < self.next_emit:
            return
        # One packet per turn; any remaining work is coalesced into a later turn.
        for name, completed in self.counts.items():
            if completed > self.sent.get(name, 0):
                self.sequence += 1
                print(json.dumps({"progress": {"sequence": self.sequence, "kind": name,
                                               "completed": completed}}), flush=True)
                self.sent[name] = completed
                self.next_emit = now + 0.5
                break


def run(command: str, cwd: Path, *, python: str, timeout_s: float | None = None,
        deadline: float | None = None, test_env: dict[str, str] | None = None,
        pytest_plugins: str | None = None, test_python: str | None = None,
        runner_kind: str = "pytest", capture_output: bool = False,
        check_settings: dict[str, str] | None = None) -> dict[str, object]:
    """Copy and check with progress supervision and bounded IPC.

    ``timeout_s`` selects inactivity, not total lifetime. Only an explicit
    ``deadline`` bounds the entire operation (including copying).

    ``test_python`` is the interpreter the check command runs. The engine's sandbox admits only the test
    interpreter it was told about (STATEFRAME_TEST_PYTHON), else its own; without this the sandbox refused a
    Homebrew python3 chosen for a repository with no venv (10-28: "execvp() ... Operation not permitted").
    """
    started = time.monotonic()
    from .engine.kernel_check_watchdog import (MAX_SAMPLE_GAP, STALL_ENV, CheckWatchdog, LaunchProgressSampler,
                                              check_watchdog_config, execution_stop_outcome,
                                              host_check_cpu_seconds, validate_execution_stop)
    try:
        settings = check_settings if check_settings is not None else os.environ
        if timeout_s is not None:
            settings = {**settings, STALL_ENV: str(timeout_s)}
        config = check_watchdog_config(settings)
        cpu_seconds = host_check_cpu_seconds(settings)
    except (ValueError, RuntimeError) as error:
        raise OuterCheckError(str(error)) from error
    if deadline is not None and not math.isfinite(deadline):
        raise OuterCheckError("run deadline must be finite")
    if deadline is not None and deadline <= started:
        raise OuterCheckError("run deadline exhausted before outer check")
    if pytest_plugins not in (None, "installed"):
        raise OuterCheckError("--pytest-plugins only accepts 'installed'")
    if runner_kind not in {"pytest", "generic"}:
        raise OuterCheckError("check runner kind must be pytest or generic")
    spec = {"command": command, "cwd": str(cwd.resolve()), "deadline": deadline,
            "stall_seconds": config.stall_seconds, "cpu_seconds": cpu_seconds,
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
    from .engine import kernel_reason_log as _reason_log
    from .engine.kernel_exec_tracker import (ContainmentUnverified, capture_launch_identity,
                                             child_exited_unreaped, kill_launch_unit,
                                             recorded_launch_identity, signal_launch_group)
    capture_launch_identity(process, session_leader=True)
    cleanup_ack = False
    helper_error = None
    helper_exit = None
    result = None
    packets = HelperPackets()
    # Give the inner check collector time to publish its terminal streams
    # and cleanup acknowledgement before the parent intervenes.
    watchdog = CheckWatchdog(replace(config, stall_seconds=config.stall_seconds + MAX_SAMPLE_GAP))
    sampler = LaunchProgressSampler(process)
    next_sample = started
    execution_stop = None
    diagnostic = bytearray()
    timed_out = False
    drain_deadline = None
    post_exit_cleanup = False
    with selectors.DefaultSelector() as selector:
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        try:
            while selector.get_map() or not child_exited_unreaped(process):
                now = time.monotonic()
                if child_exited_unreaped(process) and drain_deadline is None:
                    drain_deadline = now + HELPER_DRAIN_GRACE
                drain_until = drain_deadline
                if deadline is not None and not timed_out and not post_exit_cleanup:
                    drain_until = deadline if drain_until is None else min(drain_until, deadline)
                if drain_until is not None and now >= drain_until:
                    if child_exited_unreaped(process) and selector.get_map():
                        if post_exit_cleanup:
                            raise OuterCheckError("outer check containment unverified: helper pipes survived cleanup")
                        try:
                            survivors = kill_launch_unit(
                                process, sampler.family,
                                launch_identity=recorded_launch_identity(process),
                                reason="outer_helper_pipe_cleanup")
                        except ContainmentUnverified as error:
                            raise OuterCheckError(str(error)) from error
                        if survivors:
                            raise OuterCheckError("outer check containment unverified: helper descendants survived")
                        post_exit_cleanup = True
                        drain_deadline = time.monotonic() + HELPER_DRAIN_GRACE
                        continue
                    if timed_out:
                        break
                remaining = max(0.0, drain_until - now) if drain_until is not None else 0.05
                for key, _ in selector.select(min(remaining, 0.05)):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        diagnostic.extend(data)
                        del diagnostic[:-4000]
                        continue
                    for packet in packets.feed(data):
                        if "cleanup_ok" in packet:
                            cleanup_ack = packet["cleanup_ok"] is True
                        elif "result" in packet:
                            result = packet["result"]
                        elif "error" in packet:
                            if not timed_out:
                                helper_error = packet["error"]
                now = time.monotonic()
                if not timed_out and not child_exited_unreaped(process):
                    cause = None
                    if deadline is not None and now >= deadline:
                        cause = "user_deadline"
                    elif now >= next_sample:
                        next_sample = now + 0.5
                        if watchdog.observe(now, sampler.sample(), completed_work=packets.completed_work):
                            cause = watchdog.stop_cause
                    if cause is not None:
                        # Recheck completion at the boundary before signalling.
                        if not child_exited_unreaped(process):
                            timed_out = True
                            execution_stop = watchdog.stop_payload(cleanup_ok=False, cause=cause)
                            _reason_log.emit_kill(
                                "outer_checks", target="outer check helper", pid=process.pid,
                                why="check stopped: " + cause.replace("_", " "),
                                signal_name="SIGTERM", level="WARN", cause_key=cause)
                            process.send_signal(signal.SIGTERM)
                            drain_deadline = time.monotonic() + HELPER_CLEANUP_GRACE
            if packets.pending and not timed_out:
                raise OuterCheckError("incomplete outer check helper packet")
            if not timed_out:
                # The loop observes exit even after both IPC pipes close.
                process.wait(timeout=1)
                if process.returncode:
                    helper_exit = process.returncode
        except KeyboardInterrupt:
            execution_stop = watchdog.stop_payload(cleanup_ok=False, cause="cancelled")
            raise
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            if isinstance(error, subprocess.TimeoutExpired):
                _reason_log.emit_timeout(
                    "outer_checks", what="outer check helper", why="helper did not exit before the deadline",
                    elapsed_s=time.monotonic() - started,
                    level="WARN", cause_key="helper_wait_timeout")
            raise OuterCheckError(f"outer check helper failed: {type(error).__name__}") from error
        finally:
            if not timed_out and not child_exited_unreaped(process):
                _reason_log.emit_kill(
                    "outer_checks", target="outer check helper", pid=process.pid,
                    why="parent exited with the helper still running",
                    signal_name="SIGTERM", level="WARN", cause_key="helper_cleanup_term")
                process.send_signal(signal.SIGTERM)
            cleanup_deadline = (drain_deadline if timed_out
                                else time.monotonic() + HELPER_CLEANUP_GRACE)
            while not child_exited_unreaped(process) and time.monotonic() < cleanup_deadline:
                time.sleep(.02)
            if not child_exited_unreaped(process):
                _reason_log.emit_kill(
                    "outer_checks", target="outer check helper", pid=process.pid,
                    why="helper ignored SIGTERM through the cleanup grace",
                    signal_name="SIGKILL", level="WARN", cause_key="helper_survived_sigterm")
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
                _reason_log.emit_timeout(
                    "outer_checks", what="outer check helper", why="helper survived SIGKILL and wait",
                    deadline_s=1, level="WARN", cause_key="helper_survived_cleanup")
                raise OuterCheckError("outer check helper survived cleanup") from error
            process.stdout.close()
            process.stderr.close()
            if execution_stop is not None:
                execution_stop = watchdog.stopped(cleanup_ok=cleanup_ack,
                                                   cause=str(execution_stop["cause"]))
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
        result = {"exit_code": None, "passed": False, "tail": "check stopped before completion",
                  "output_bytes": 0, "output_truncated": False}
    result.update(command=command, seconds=round(time.monotonic() - started, 2))
    if execution_stop is not None:
        result["exit_code"] = None
    execution_stop = validate_execution_stop(execution_stop or result.get("execution_stop"))
    if execution_stop is not None:
        if not execution_stop["cleanup_ok"]:
            raise OuterCheckError("outer check containment unverified after execution stop")
        result.update(passed=False,
                      timed_out=execution_stop["cause"] == "user_deadline",
                      execution_stop=execution_stop, outcome=execution_stop_outcome(execution_stop),
                      termination_reason=execution_stop["cause"], execution_complete=False,
                      reached_observation=False)
    else:
        result.setdefault("execution_stop", None)
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
    import kernel_reason_log as _reason_log
    import kernel_execution_policy as policy
    from kernel_check_receipts import sandbox_open_denial

    spawned = None
    scratch = None
    process_cleaned = False
    cleanup_ok = True
    from kernel_check_watchdog import CheckWatchdogConfig, execution_stop_outcome, validate_execution_stop
    progress = _ProgressEmitter()
    watchdog_config = CheckWatchdogConfig(
        spec.get("stall_seconds", CHECK_TIMEOUT),
        on_progress=lambda sequence, kind: progress.advance("check"))
    original_spawn = policy.sandboxed_popen
    original_kill_group = policy._kill_group

    def tracked_spawn(*args, **kwargs):
        nonlocal spawned, process_cleaned
        spawned, capabilities = original_spawn(*args, **kwargs)
        process_cleaned = False
        return spawned, capabilities

    def tracked_kill_group(process, family=None, *, launch_identity=None, **log_fields):
        nonlocal process_cleaned
        cleaned = original_kill_group(
            process, family, launch_identity=launch_identity, **log_fields)
        if process is spawned and cleaned and getattr(process, "returncode", None) is not None:
            process_cleaned = True
        return cleaned

    def stop(signum, frame):
        raise InterruptedError("outer check cancelled by parent")

    signal.signal(signal.SIGTERM, stop)
    try:
        policy.sandboxed_popen = tracked_spawn
        policy._kill_group = tracked_kill_group
        policy.require_capabilities()
        scratch = tempfile.TemporaryDirectory(prefix="stateframe-tests-", dir=Path(spec["cwd"]).parent)
        copy = Path(scratch.name) / "repo"
        def copy_file(source, destination):
            copied = shutil.copy2(source, destination)
            progress.advance("copy", max(1, os.stat(destination).st_size))
            return copied

        shutil.copytree(spec["cwd"], copy, symlinks=True, copy_function=copy_file)
        progress.advance("copy")
        if spec.get("deadline") is not None and time.monotonic() >= spec["deadline"]:
            raise OuterCheckError("run deadline exhausted copying outer check")
        for attempt in range(SANDBOX_FORK_RETRIES + 1):
            result = policy.run_sandboxed(shlex.split(spec["command"]), copy,
                                          env=env, timeout=None, check_watchdog=watchdog_config,
                                          cpu_seconds=spec.get("cpu_seconds"),
                                          output_limit=OUTPUT_LIMIT,
                                          generic_check=spec["runner_kind"] == "generic")
            process_cleaned = process_cleaned or (result.process_group_killed and spawned is not None
                                                  and getattr(spawned, "returncode", None) is not None)
            execution_stop = validate_execution_stop(getattr(result, "execution_stop", None))
            if execution_stop is not None:
                break
            # bwrap's own clone() can hit a transient EAGAIN on a loaded host; that is
            # "try again", not "no sandbox". Retry briefly within the same deadline only.
            stderr = result.completed.stderr
            transient = (result.completed.returncode and stderr.startswith("bwrap:")
                         and "Resource temporarily unavailable" in stderr)
            if not transient or attempt == SANDBOX_FORK_RETRIES:
                break
            pause = 0.5 * (attempt + 1)
            if (spec.get("deadline") is not None
                    and spec["deadline"] - time.monotonic() <= pause + 1):
                break
            time.sleep(pause)
        if execution_stop is None:
            sandbox_error = result.completed.stderr
            denial = sandbox_open_denial(
                sandbox_error + "\n" + result.completed.stdout, policy.policy_read_grants(result))
            if denial is not None:
                raise OuterCheckError(denial)
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
                   "passed": not result.timed_out and execution_stop is None and result.completed.returncode == 0,
                   "tail": output[-4000:].decode("utf-8", errors="ignore"),
                   "timed_out": result.timed_out, "output_truncated": result.output_truncated,
                   "output_bytes": len(output),
                   "process_group_killed": result.process_group_killed,
                   "execution_stop": execution_stop}
        if execution_stop is not None:
            payload.update(outcome=execution_stop_outcome(execution_stop), termination_reason=execution_stop["cause"],
                           execution_complete=False, reached_observation=False)
        if spec.get("capture_output"):
            payload["output"] = output.decode("utf-8", errors="replace")
        encoded = json.dumps({"result": payload})
        progress.advance("serialize", len(encoded))
        print(encoded, flush=True)
    except (RuntimeError, OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}), flush=True)
        return 2
    finally:
        policy.sandboxed_popen = original_spawn
        policy._kill_group = original_kill_group
        def cleanup_timeout(_signum, _frame):
            raise TimeoutError("outer check process cleanup exceeded its budget")

        previous_alarm = signal.signal(signal.SIGALRM, cleanup_timeout)
        signal.setitimer(signal.ITIMER_REAL, HELPER_PROCESS_CLEANUP_S)
        if spawned is not None and not process_cleaned:
            try:
                cleanup_ok = policy._kill_group(
                    spawned, cause="helper_process_cleanup",
                    why="helper stopped; killing the check's own process group") and cleanup_ok
                spawned.wait(timeout=1)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                # TimeoutError (the SIGALRM budget above) is an OSError. Logged here, not in the
                # alarm handler: emit takes a lock the interrupted frame may already hold.
                _reason_log.emit_timeout(
                    "outer_check_helper", what="helper process cleanup",
                    why=f"process group cleanup failed: {type(error).__name__}",
                    deadline_s=HELPER_PROCESS_CLEANUP_S, level="WARN", cause_key="helper_cleanup")
                cleanup_ok = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if scratch is not None:
            try:
                scratch.cleanup()
            except OSError:
                cleanup_ok = False
        print(json.dumps({"cleanup_ok": cleanup_ok}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_helper())
