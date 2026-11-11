"""Progress-aware subprocess capture for tests that exercise a real engine run."""
from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, IO, Sequence
from _timeouts import scaled

# Match the previous generic run_case limit when a caller has no measurement.
DEFAULT_EXPECTED_SECONDS = scaled(240.0)
# Ten polls per second catches durable events while keeping filesystem work modest.
POLL_INTERVAL_SECONDS = 0.1
# Cap load scaling so a pathological shared host cannot create an unbounded deadline.
MAX_LOAD_FACTOR = 4.0
# Allow three measured windows for a run that keeps progressing, while bounding it.
HARD_CEILING_MULTIPLIER = 3.0
# Give a child one second to shut down cleanly before escalating to SIGKILL.
TERMINATE_GRACE_SECONDS = scaled(1.0)
# Allow time for process reaping and then verify the whole process group is gone.
GROUP_VERIFY_SECONDS = scaled(3.0)
# Match the diagnostic tail requested by the lane brief.
OUTPUT_TAIL_LINES = 40
# Bound diagnostic reads even if a child emits one enormous line or a huge log.
OUTPUT_TAIL_BYTES = 256 * 1024
# Keep caller-probe evidence useful without embedding an unbounded object repr.
PROBE_VALUE_PREVIEW_CHARS = 200
_PROBE_UNSET = object()


class EngineRunTimeout(AssertionError):
    """A monitored child stopped progressing or exceeded its fixed hard ceiling."""

    def __init__(self, message: str, *, pid: int, process_group_id: int, group_gone: bool,
                 stage: str, elapsed: float, seconds_since_progress: float,
                 last_progress: str, idle_window: float, hard_ceiling: float,
                 load_factor: float, stdout_tail: str, stderr_tail: str,
                 last_probe_value: str | None = None,
                 progress_probe_error_type: str | None = None):
        super().__init__(message)
        self.pid = pid
        self.process_group_id = process_group_id
        self.group_gone = group_gone
        self.stage = stage
        self.elapsed = elapsed
        self.seconds_since_progress = seconds_since_progress
        self.last_progress = last_progress
        self.idle_window = idle_window
        self.hard_ceiling = hard_ceiling
        self.load_factor = load_factor
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail
        self.last_probe_value = last_probe_value
        self.progress_probe_error_type = progress_probe_error_type


class EngineRunMonitorError(RuntimeError):
    """Progress sampling failed, so the child was stopped instead of running unobserved."""

    def __init__(self, message: str, *, process_group_id: int, group_gone: bool):
        super().__init__(message)
        self.process_group_id = process_group_id
        self.group_gone = group_gone


def host_load_factor() -> float:
    """Normalize the five-minute load average and cap pathological host throttling."""
    try:
        load_average = os.getloadavg()[1]
    except (AttributeError, OSError) as error:
        raise RuntimeError("cannot read the host load average for a monitored subprocess") from error
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        raise RuntimeError("cannot determine the host CPU count for a monitored subprocess")
    if not math.isfinite(load_average) or load_average < 0:
        raise RuntimeError(f"invalid host five-minute load average: {load_average!r}")
    return max(1.0, min(load_average / cpu_count, MAX_LOAD_FACTOR))


def deadline_windows(expected_seconds: float, load_factor: float) -> tuple[float, float]:
    """Return an idle window and a fixed ceiling; later progress moves only the idle deadline."""
    if not math.isfinite(expected_seconds) or expected_seconds <= 0:
        raise ValueError("expected_seconds must be a finite positive number")
    if not math.isfinite(load_factor) or not 1 <= load_factor <= MAX_LOAD_FACTOR:
        raise ValueError(f"load_factor must be between 1 and {MAX_LOAD_FACTOR}")
    idle_window = max(expected_seconds, math.ceil(expected_seconds * load_factor))
    hard_ceiling = max(
        idle_window,
        math.ceil(expected_seconds * HARD_CEILING_MULTIPLIER * load_factor),
    )
    return float(idle_window), float(hard_ceiling)


class _LineCounter:
    """Count appended JSONL lines without rereading an entire growing journal."""

    def __init__(self) -> None:
        self._identity: tuple[int, int] | None = None
        self._offset = 0
        self._lines = 0

    def count(self, path: Path) -> int:
        try:
            info = path.stat()
        except FileNotFoundError:
            self._identity = None
            self._offset = 0
            self._lines = 0
            return -1
        identity = (info.st_dev, info.st_ino)
        if identity != self._identity or info.st_size < self._offset:
            self._identity = identity
            self._offset = 0
            self._lines = 0
        with path.open("rb") as stream:
            stream.seek(self._offset)
            appended = stream.read()
        self._offset += len(appended)
        self._lines += appended.count(b"\n")
        return self._lines


class _Progress:
    def __init__(self, run_dir: Path | None, output_paths: tuple[Path, Path],
                 watch_paths: Sequence[Path]):
        self.run_dir = run_dir
        self.output_paths = output_paths
        self.watch_paths = tuple(Path(path) for path in watch_paths)
        self._sends = _LineCounter()

    @staticmethod
    def _regular_files(root: Path) -> tuple[Path, ...]:
        try:
            if root.is_file():
                return (root,)
            if not root.is_dir():
                return ()
            return tuple(path for path in root.rglob("*") if path.is_file())
        except FileNotFoundError:
            return ()

    @classmethod
    def _add_file_set(cls, values: dict[str, int], label: str, paths: Sequence[Path]) -> None:
        for path in paths:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            values[f"{label}:{path}"] = size
        values[f"{label}:count"] = len(paths)
        values[f"{label}:bytes"] = sum(values[key] for key in values if key.startswith(f"{label}:" )
                                        and key not in {f"{label}:count", f"{label}:bytes"})

    def snapshot(self) -> dict[str, int]:
        values: dict[str, int] = {}
        if self.run_dir is not None:
            sends_path = self.run_dir / "sends.jsonl"
            values["sends.jsonl:lines"] = self._sends.count(sends_path)
            wire = self.run_dir / "wire"
            request_paths = tuple(sorted(wire.glob("*.request.json"))) if wire.is_dir() else ()
            response_paths = tuple(sorted(wire.glob("*.response.json"))) if wire.is_dir() else ()
            self._add_file_set(values, "wire request", request_paths)
            self._add_file_set(values, "wire response", response_paths)

            measurement = self.run_dir / "worker" / "measurement-runs"
            raw_events = tuple(path for path in self._regular_files(measurement)
                               if path.name == "events.jsonl")
            packed_events = tuple(path for path in self._regular_files(measurement)
                                  if path.name == "events.jsonl.xz")
            self._add_file_set(values, "measurement events", raw_events)
            self._add_file_set(values, "packed measurement events", packed_events)
            self._add_file_set(values, "checks", self._regular_files(self.run_dir / "checks"))

            summary = self.run_dir / "summary.json"
            try:
                values["summary.json:bytes"] = summary.stat().st_size
            except FileNotFoundError:
                values["summary.json:bytes"] = -1

        for label, path in zip(("stdout", "stderr"), self.output_paths):
            try:
                values[f"{label}:bytes"] = path.stat().st_size
            except FileNotFoundError:
                values[f"{label}:bytes"] = 0

        for index, path in enumerate(self.watch_paths):
            files = self._regular_files(path)
            self._add_file_set(values, f"watch[{index}]", files)
            if path.is_dir():
                values[f"watch[{index}]:present"] = 1
            else:
                values[f"watch[{index}]:present"] = int(path.is_file())
        return values


def _increases(previous: dict[str, int], current: dict[str, int]) -> list[str]:
    changes = []
    for key, value in current.items():
        old = previous.get(key)
        if old is None or value > old:
            changes.append(f"{key}={value}" if old is None else f"{key}={value} (+{value - old})")
    return changes


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            start = max(0, size - OUTPUT_TAIL_BYTES)
            stream.seek(start)
            content = stream.read().decode(errors="replace")
    except FileNotFoundError:
        return "<no output>"
    except OSError as error:
        return f"<cannot read output tail: {error}>"
    lines = content.splitlines()[-OUTPUT_TAIL_LINES:]
    prefix = "[older output omitted]\n" if start else ""
    return prefix + ("\n".join(lines) or "<empty>")


def _group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class EngineProcess:
    """A process-group watcher retaining Popen-style controls and captured text output."""

    def __init__(self, process: subprocess.Popen[bytes], args: Sequence[str | os.PathLike[str]],
                 expected_seconds: float, run_dir: Path | None, step: str,
                 output_dir: tempfile.TemporaryDirectory[str], output_paths: tuple[Path, Path],
                 progress: _Progress, initial_metrics: dict[str, int], load_factor: float, *,
                 progress_probe: Callable[[], object] | None = None):
        self._process = process
        self.args = args
        self.pid = process.pid
        self.process_group_id = process.pid
        self.expected_seconds = expected_seconds
        self.run_dir = run_dir
        self.step = step
        self._output_dir = output_dir
        self._output_paths = output_paths
        self._progress = progress
        self.load_factor = load_factor
        self.idle_window, self.hard_ceiling = deadline_windows(expected_seconds, load_factor)
        self.started_at = time.monotonic()
        self.last_progress_at = self.started_at
        self.last_progress = "no durable progress observed yet"
        self._metrics = initial_metrics
        self._progress_probe = progress_probe
        self._probe_value: object = _PROBE_UNSET
        self._last_probe_value: str | None = None
        self._probe_error_type: str | None = None
        self._closed = False

    @property
    def returncode(self) -> int | None:
        return self._process.returncode

    def _stage(self, metrics: dict[str, int]) -> str:
        if self.run_dir is None:
            return self.step
        if metrics.get("summary.json:bytes", -1) >= 0:
            return "finalizing"
        requests = metrics.get("wire request:count", 0)
        responses = metrics.get("wire response:count", 0)
        sends = max(metrics.get("sends.jsonl:lines", 0), 0)
        if requests > responses:
            return f"send {responses + 1} in flight"
        if sends > responses:
            return f"send {responses + 1} in flight"
        if metrics.get("checks:count", 0):
            return "checks"
        if sends or requests or responses:
            return "finalizing"
        return "starting"

    def _stop_group(self) -> bool:
        try:
            os.killpg(self.process_group_id, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        terminate_until = time.monotonic() + TERMINATE_GRACE_SECONDS
        while time.monotonic() < terminate_until and _group_exists(self.process_group_id):
            time.sleep(POLL_INTERVAL_SECONDS)
        if _group_exists(self.process_group_id):
            try:
                os.killpg(self.process_group_id, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            self._process.wait(timeout=GROUP_VERIFY_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process_group_id, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        verify_until = time.monotonic() + GROUP_VERIFY_SECONDS
        while time.monotonic() < verify_until:
            if not _group_exists(self.process_group_id):
                return True
            time.sleep(POLL_INTERVAL_SECONDS)
        return not _group_exists(self.process_group_id)

    def _timeout(self, now: float, metrics: dict[str, int], reason: str) -> EngineRunTimeout:
        elapsed = now - self.started_at
        idle_seconds = now - self.last_progress_at
        stage = "progress_probe" if self._probe_error_type is not None else self._stage(metrics)
        group_gone = self._stop_group()
        stderr_tail = _tail(self._output_paths[1])
        stdout_tail = _tail(self._output_paths[0])
        probe_evidence = ""
        if self._progress_probe is not None:
            probe_evidence = f"; last_probe_value={self._last_probe_value!r}"
            if self._probe_error_type is not None:
                probe_evidence += f"; progress_probe_error_type={self._probe_error_type}"
        message = (
            f"monitored subprocess expired during {stage!r} ({self.step}): {reason}; "
            f"elapsed={elapsed:.1f}s; seconds_since_progress={idle_seconds:.1f}s; "
            f"last_progress={self.last_progress!r}{probe_evidence}; idle_window={self.idle_window:.1f}s; "
            f"hard_ceiling={self.hard_ceiling:.1f}s; load_factor={self.load_factor:.2f}; "
            f"process_group={self.process_group_id}; group_gone={group_gone}\n"
            f"stderr tail (last {OUTPUT_TAIL_LINES} lines):\n{stderr_tail}\n"
            f"stdout tail (last {OUTPUT_TAIL_LINES} lines):\n{stdout_tail}"
        )
        return EngineRunTimeout(message, pid=self.pid, process_group_id=self.process_group_id,
                                group_gone=group_gone, stage=stage, elapsed=elapsed,
                                seconds_since_progress=idle_seconds,
                                last_progress=self.last_progress, idle_window=self.idle_window,
                                hard_ceiling=self.hard_ceiling, load_factor=self.load_factor,
                                stdout_tail=stdout_tail, stderr_tail=stderr_tail,
                                last_probe_value=self._last_probe_value,
                                progress_probe_error_type=self._probe_error_type)

    def poll(self) -> int | None:
        now = time.monotonic()
        returncode = self._process.poll()
        try:
            metrics = self._progress.snapshot()
        except OSError as error:
            group_gone = self._stop_group()
            raise EngineRunMonitorError(
                f"cannot sample subprocess progress: {error}; process_group={self.process_group_id}; "
                f"group_gone={group_gone}", process_group_id=self.process_group_id,
                group_gone=group_gone,
            ) from error
        changes = _increases(self._metrics, metrics)
        self._metrics = metrics
        if self._progress_probe is not None:
            try:
                probe_value = self._progress_probe()
                probe_preview = repr(probe_value)[:PROBE_VALUE_PREVIEW_CHARS]
                probe_changed = (self._probe_value is not _PROBE_UNSET and
                                 bool(probe_value != self._probe_value))
            except Exception as error:
                self._probe_error_type = type(error).__name__
                probe_changed = False
            else:
                self._probe_error_type = None
                self._probe_value = probe_value
                self._last_probe_value = probe_preview
            if probe_changed:
                changes.append(f"progress probe changed to {probe_preview}")
        if changes:
            self.last_progress = "; ".join(changes)
            self.last_progress_at = now
        elapsed = now - self.started_at
        idle_seconds = now - self.last_progress_at
        if elapsed >= self.hard_ceiling:
            raise self._timeout(now, metrics, "hard ceiling reached")
        if returncode is None and idle_seconds >= self.idle_window:
            raise self._timeout(now, metrics, "progress idle window expired")
        return returncode

    def wait(self, timeout: float | None = None) -> int:
        wait_started = time.monotonic()
        while True:
            returncode = self.poll()
            if returncode is not None:
                return returncode
            if timeout is not None and time.monotonic() - wait_started >= timeout:
                raise subprocess.TimeoutExpired(self.args, timeout, output=self.stdout_text,
                                                stderr=self.stderr_text)
            time.sleep(POLL_INTERVAL_SECONDS)

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.wait(timeout=timeout)
        return self.stdout_text, self.stderr_text

    @property
    def stdout_text(self) -> str:
        return self._output_paths[0].read_text(errors="replace")

    @property
    def stderr_text(self) -> str:
        return self._output_paths[1].read_text(errors="replace")

    def send_signal(self, sig: int) -> None:
        try:
            self._process.send_signal(sig)
        except ProcessLookupError:
            return

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def close(self) -> None:
        if self._closed:
            return
        if self._process.poll() is None:
            if not self._stop_group():
                raise RuntimeError(
                    f"process group {self.process_group_id} remained after monitored-process cleanup")
        self._closed = True
        self._output_dir.cleanup()

    def __enter__(self) -> "EngineProcess":
        return self

    def __exit__(self, _exc_type: object, exc_value: object, _traceback: object) -> None:
        if (isinstance(exc_value, (EngineRunTimeout, EngineRunMonitorError))
                and not exc_value.group_gone):
            # Cleanup was already attempted; preserve its evidence instead of masking it.
            self._closed = True
            self._output_dir.cleanup()
            return
        self.close()


def start_monitored(command: Sequence[str | os.PathLike[str]], *,
                    expected_seconds: float = DEFAULT_EXPECTED_SECONDS,
                    run_dir: Path | None = None, step: str = "subprocess",
                    cwd: str | os.PathLike[str] | None = None,
                    env: dict[str, str] | None = None,
                    watch_paths: Sequence[Path] = (),
                    progress_probe: Callable[[], object] | None = None,
                    stdout: object = subprocess.PIPE, stderr: object = subprocess.PIPE,
                    text: bool = True, start_new_session: bool = True) -> EngineProcess:
    """Start a captured child in its own group and monitor durable progress until it exits."""
    if not text:
        raise ValueError("monitored subprocess output must use text mode")
    if stdout not in (None, subprocess.PIPE) or stderr not in (None, subprocess.PIPE):
        raise ValueError("monitored subprocesses own stdout and stderr capture")
    if not start_new_session:
        raise ValueError("monitored subprocesses must start in their own process group")
    load_factor = host_load_factor()
    deadline_windows(expected_seconds, load_factor)
    output_dir = tempfile.TemporaryDirectory(prefix="stateframe-test-run-")
    stdout_path = Path(output_dir.name) / "stdout.log"
    stderr_path = Path(output_dir.name) / "stderr.log"
    actual_run_dir = Path(run_dir) if run_dir is not None else _infer_run_dir(command)
    progress = _Progress(actual_run_dir, (stdout_path, stderr_path), watch_paths)
    try:
        initial_metrics = progress.snapshot()
    except Exception as error:
        output_dir.cleanup()
        raise RuntimeError(f"cannot establish subprocess progress baseline: {error}") from error
    stdout_stream: IO[bytes] | None = None
    stderr_stream: IO[bytes] | None = None
    try:
        stdout_stream = stdout_path.open("wb")
        stderr_stream = stderr_path.open("wb")
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout_stream,
                                   stderr=stderr_stream, start_new_session=True)
        stdout_stream.close()
        stderr_stream.close()
        return EngineProcess(process, command, expected_seconds, actual_run_dir, step,
                             output_dir, (stdout_path, stderr_path), progress, initial_metrics,
                             load_factor, progress_probe=progress_probe)
    except BaseException:
        if stdout_stream is not None:
            stdout_stream.close()
        if stderr_stream is not None:
            stderr_stream.close()
        output_dir.cleanup()
        raise


def run_monitored(command: Sequence[str | os.PathLike[str]], *,
                  expected_seconds: float = DEFAULT_EXPECTED_SECONDS,
                  run_dir: Path | None = None, step: str = "subprocess",
                  cwd: str | os.PathLike[str] | None = None,
                  env: dict[str, str] | None = None,
                  watch_paths: Sequence[Path] = (),
                  progress_probe: Callable[[], object] | None = None,
                  check: bool = False,
                  capture_output: bool = True, text: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a monitored child and return the familiar CompletedProcess capture result."""
    if not capture_output:
        raise ValueError("monitored subprocess output must be captured")
    with start_monitored(command, expected_seconds=expected_seconds, run_dir=run_dir,
                         step=step, cwd=cwd, env=env, watch_paths=watch_paths,
                         progress_probe=progress_probe,
                         text=text) as process:
        returncode = process.wait()
        stdout = process.stdout_text
        stderr = process.stderr_text
        result = subprocess.CompletedProcess(command, returncode, stdout, stderr)
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, output=stdout, stderr=stderr)
    return result


def _infer_run_dir(command: Sequence[str | os.PathLike[str]]) -> Path | None:
    arguments = [os.fspath(argument) for argument in command]
    mode_index = next((index + 2 for index in range(len(arguments) - 1)
                       if arguments[index:index + 2] == ["-m", "stateframe"]), None)
    if mode_index is None and arguments and Path(arguments[0]).name == "stateframe":
        mode_index = 1
    if mode_index is None or mode_index >= len(arguments):
        return None
    mode = arguments[mode_index]
    if mode == "run":
        for argument in arguments:
            if argument.startswith("--run-dir="):
                return Path(argument.partition("=")[2])
        try:
            return Path(arguments[arguments.index("--run-dir") + 1])
        except (ValueError, IndexError):
            return None
    if mode == "resume" and mode_index + 1 < len(arguments):
        return Path(arguments[mode_index + 1])
    return None
