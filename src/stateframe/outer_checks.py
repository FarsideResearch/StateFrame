"""Fresh-interpreter adapter to the engine's bounded execution policy.

The CLI imports only this stdlib adapter. The execution policy is imported
only in the helper, after installing host test policy. The caller loads the
process-identity tracker lazily when it starts a check.
"""
from __future__ import annotations

from dataclasses import replace

import hashlib
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import stat
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


def _boundary_receipt(record: dict[str, object] | None) -> dict[str, object]:
    return {"execution_boundary": record} if record is not None else {}


def run(command: str, cwd: Path, *, python: str, timeout_s: float | None = None,
        deadline: float | None = None, test_env: dict[str, str] | None = None,
        pytest_plugins: str | None = None, test_python: str | None = None,
        runner_kind: str = "pytest", capture_output: bool = False,
        check_settings: dict[str, str] | None = None,
        required_test_ledger: bool = False,
        protected_test_hashes: dict[str, str] | None = None,
        execution_boundary: dict[str, object] | None = None) -> dict[str, object]:
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
    if required_test_ledger:
        spec["required_test_ledger"] = True
    from .engine.kernel_safe_paths import container_active
    boundary_record = execution_boundary
    if boundary_record is None and container_active():
        for name in ("stateframe.engine.kernel_execution_boundary", "kernel_execution_boundary"):
            boundary = sys.modules.get(name)
            active = getattr(boundary, "active", None)
            current = active() if callable(active) else None
            if getattr(current, "backend", None) == "container":
                boundary_record = current.record()
                break
    if boundary_record is not None and boundary_record.get("backend") == "container":
        spec["execution_boundary"] = boundary_record
        if protected_test_hashes:
            spec["protected_test_hashes"] = dict(protected_test_hashes)
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
                        if watchdog.observe(now, sampler.snapshot(), completed_work=packets.completed_work):
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
    result.update(command=command, seconds=round(time.monotonic() - started, 2),
                  **_boundary_receipt(boundary_record))
    if execution_stop is not None:
        result["exit_code"] = None
    execution_stop = validate_execution_stop(execution_stop or result.get("execution_stop"))
    if execution_stop is not None:
        if not execution_stop["cleanup_ok"]:
            raise OuterCheckError("outer check containment unverified after execution stop")
        # Keep the timeout flag for callers of the outer adapter. The typed
        # cause distinguishes an inactivity timeout from a user deadline.
        result.update(passed=False,
                      timed_out=execution_stop["cause"] in {"no_progress", "user_deadline"},
                      execution_stop=execution_stop, outcome=execution_stop_outcome(execution_stop),
                      termination_reason=execution_stop["cause"], execution_complete=False,
                      reached_observation=False)
    else:
        result.setdefault("execution_stop", None)
    return result


_PYTHON_NAME = re.compile(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?\Z")
_PYTHON_FLAGS = frozenset({"-B", "-E", "-I", "-P", "-s", "-u"})


def _pytest_console_script(program: str, copy: Path, env: dict[str, str]) -> tuple[str, str] | None:
    """Recognize the stock installed console script and retain its interpreter."""
    if "/" in program:
        path = str(copy / program)
    else:
        search_path = os.pathsep.join(str(copy / entry) for entry in os.get_exec_path(env))
        path = shutil.which(program, path=search_path)
    if path is None or not Path(path).is_file() or not os.access(path, os.X_OK):
        return None
    selected = Path(sys.executable)
    if not Path(path).resolve().is_relative_to(selected.parent.resolve()):
        return None
    with open(path, "rb") as stream:
        header = stream.read(4096).splitlines()
    if not header or not header[0].startswith(b"#!"):
        return None
    interpreter = Path(os.fsdecode(header[0][2:]).strip())
    if (interpreter.parent != selected.parent or interpreter.resolve() != selected.resolve()
            or b"from pytest import console_main" not in header
            or b"    sys.exit(console_main())" not in header):
        return None
    return str(interpreter), path


def _pytest_ledger_command(arguments: list[str], copy: Path, report_path: Path,
                          env: dict[str, str]) -> list[str] | None:
    """Attach a passive pytest plugin to a known launch without adding pytest flags."""
    from kernel_required_tests import MAX_PYTEST_REPORT_BYTES, PYTEST_LEDGER_SOURCE

    offset = 0
    managed = False
    command_env = dict(env)
    while offset < len(arguments):
        wrapper = Path(arguments[offset]).name
        if wrapper == "env":
            offset += 1
            while (offset < len(arguments)
                   and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", arguments[offset], re.DOTALL)):
                key, value = arguments[offset].split("=", 1)
                command_env[key] = value
                offset += 1
        elif wrapper in {"uv", "poetry", "pipenv"} and arguments[offset + 1:offset + 2] == ["run"]:
            offset += 2
            managed = True
        else:
            break
    if offset >= len(arguments):
        return None
    program = arguments[offset]
    name = Path(program).name
    console_script = None
    flags = []
    if _PYTHON_NAME.fullmatch(name):
        index = offset + 1
        while index < len(arguments) and arguments[index] in _PYTHON_FLAGS:
            index += 1
        if arguments[index:index + 2] != ["-m", "pytest"]:
            return None
        flags = arguments[offset + 1:index]
        pytest_args = arguments[index + 2:]
        original_args = [*flags, "-m", "pytest"]
    elif name in {"pytest", "py.test"} and not managed:
        console = _pytest_console_script(program, copy, command_env)
        if console is None:
            return None
        program, console_script = console
        pytest_args = arguments[offset + 1:]
        original_args = [console_script]
    else:
        return None
    settings = (str(copy), str(report_path), MAX_PYTEST_REPORT_BYTES, console_script, original_args)
    source = f"repo, report_path, report_limit, console_script, original_args = {settings!r}\n"
    wrapped = [*arguments[:offset], program, *flags, "-c", source + PYTEST_LEDGER_SOURCE, *pytest_args]
    # Keep the original check if the optional instrumentation would exceed argv policy.
    return wrapped if len(wrapped) <= 128 and sum(map(len, wrapped)) <= 64 * 1024 else None


def _read_required_test_report(path: Path, exit_code: int | None) -> tuple[dict[str, object], str | None]:
    from kernel_required_tests import (MAX_PYTEST_REPORT_BYTES, PYTEST_REPORT_NOT_PRODUCED,
                                       parse_pytest_report_with_reason)

    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    directory = os.open(path.parent, flags | os.O_DIRECTORY)
    try:
        descriptor = os.open(path.name, flags, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if stat.S_ISREG(metadata.st_mode) and metadata.st_size <= MAX_PYTEST_REPORT_BYTES:
            return parse_pytest_report_with_reason(stream.read(MAX_PYTEST_REPORT_BYTES + 1), exit_code)
    return {"complete": False, "outcomes": {}}, PYTEST_REPORT_NOT_PRODUCED


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
    if "execution_boundary" in spec:
        import kernel_execution_boundary as boundary
        record = spec["execution_boundary"]
        try:
            selected = boundary.Boundary.from_record(record)
            current = boundary.authorize(
                True, conflicts=(), detection=boundary.detect(),
                uid=getattr(os, "geteuid", lambda: -1)())
            boundary.check_resume(record, current)
            boundary.activate(selected)
        except boundary.BoundaryRefused as error:
            print(json.dumps({"error": error.code}), flush=True)
            print(json.dumps({"cleanup_ok": True}), flush=True)
            return 2
        except (TypeError, ValueError, KeyError) as error:
            raise OuterCheckError("CONTAINER_RESUME_MISMATCH") from error
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
    required_report_path = None
    required_report = {"complete": False, "outcomes": {}}
    required_report_reason = "pytest report was not produced"

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
        check_command = shlex.split(spec["command"])
        if spec.get("required_test_ledger") is True and spec.get("runner_kind") == "pytest":
            try:
                report_directory = copy / ".test-tmp"
                if report_directory.is_symlink():
                    raise ValueError("cannot place a ledger report in symlinked check scratch")
                report_directory.mkdir(exist_ok=True)
                with tempfile.NamedTemporaryFile(prefix=".stateframe-pytest-", dir=report_directory,
                                                 delete=False) as report_file:
                    report_path = Path(report_file.name)
                wrapped = _pytest_ledger_command(check_command, copy, report_path, env)
                if wrapped is not None:
                    check_command, required_report_path = wrapped, report_path
                    required_report_reason = "pytest report was not produced"
                else:
                    report_path.unlink()
                    required_report_reason = "pytest report was not produced"
            except Exception:  # noqa: BLE001 - report setup is advisory
                required_report_reason = "pytest report was not produced"
        if spec.get("deadline") is not None and time.monotonic() >= spec["deadline"]:
            raise OuterCheckError("run deadline exhausted copying outer check")
        for attempt in range(SANDBOX_FORK_RETRIES + 1):
            result = policy.run_sandboxed(check_command, copy,
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
        if required_report_path is not None and execution_stop is None and not result.timed_out:
            try:
                required_report, required_report_reason = _read_required_test_report(
                    required_report_path, result.completed.returncode)
            except Exception:  # noqa: BLE001 - unreadable reports remain unknown
                required_report_reason = "pytest report was not produced"
        elif spec.get("required_test_ledger") is True and spec.get("runner_kind") == "pytest":
            required_report_reason = "pytest report was not produced"
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
        tampered = _protected_test_tampering(copy, spec.get("protected_test_hashes", {}))
        output = (result.completed.stdout + result.completed.stderr).encode("utf-8")
        if tampered:
            output += ("\nDETECTED PROTECTED_TEST_TAMPERED: post-command hash changed for "
                       + ", ".join(tampered)).encode("utf-8")
        payload = {"exit_code": None if result.timed_out else result.completed.returncode,
                   "passed": not tampered and not result.timed_out and execution_stop is None and result.completed.returncode == 0,
                   "tail": output[-4000:].decode("utf-8", errors="ignore"),
                   "timed_out": result.timed_out, "output_truncated": result.output_truncated,
                   "output_bytes": len(output),
                   "process_group_killed": result.process_group_killed,
                   "execution_stop": execution_stop}
        if tampered:
            payload["protected_test_tampered"] = tampered
        if execution_stop is not None:
            payload.update(outcome=execution_stop_outcome(execution_stop), termination_reason=execution_stop["cause"],
                           execution_complete=False, reached_observation=False)
        if spec.get("capture_output"):
            payload["output"] = output.decode("utf-8", errors="replace")
        if spec.get("required_test_ledger") is True:
            payload["required_test_report"] = required_report
            if not required_report["complete"]:
                payload["required_test_report_reason"] = required_report_reason
        encoded = json.dumps({"result": payload})
        if required_report["complete"] and len(encoded) + len('{"cleanup_ok": false}\n') + 1 > WIRE_LIMIT:
            payload["required_test_report"] = {"complete": False, "outcomes": {}}
            payload["required_test_report_reason"] = "pytest report exceeded its size bound"
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
        # Acknowledge process containment before removing the scratch tree.
        print(json.dumps({"cleanup_ok": cleanup_ok}), flush=True)
        if scratch is not None:
            scratch.cleanup()
    return 0


def _protected_test_tampering(root: Path, expected: object) -> list[str]:
    if not isinstance(expected, dict):
        raise OuterCheckError("invalid protected test hashes")
    tampered = []
    for relative, digest in expected.items():
        path = Path(relative) if isinstance(relative, str) else Path("/")
        if (not isinstance(relative, str) or path.is_absolute() or ".." in path.parts
                or path.as_posix() != relative or not re.fullmatch(r"[0-9a-f]{64}", str(digest))):
            raise OuterCheckError("invalid protected test hashes")
        actual = None
        try:
            with os.fdopen(os.open(root / path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                                   | getattr(os, "O_NONBLOCK", 0)), "rb") as stream:
                info = os.fstat(stream.fileno())
                if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError:
            actual = None
        if actual != digest:
            tampered.append(relative)
    return sorted(tampered)


if __name__ == "__main__":
    raise SystemExit(_helper())
