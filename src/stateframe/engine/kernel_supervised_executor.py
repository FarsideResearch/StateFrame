"""Bounded, snapshot-bound synchronous command execution for R-172.

The executor is deliberately separate from ``kernel_job_watch``.  That module
owns the model-visible background-job experiment; this module owns one host
supervised command at a time and is inert until its own switch is enabled.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence

from kernel_probe import workspace_revision


MODE_ENV = "STATEFRAME_SUPERVISED_EXECUTOR"
DEADLINE_ENV = "STATEFRAME_SUPERVISED_EXECUTOR_DEADLINE_S"
DEFAULT_DEADLINE_SECONDS = 300.0

EXECUTED = "EXECUTED"
ATTACHED = "ATTACHED"
REUSED = "REUSED"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _command_tuple(command: Sequence[object]) -> tuple[str, ...]:
    if isinstance(command, (str, bytes, bytearray)):
        raise TypeError("command must be a sequence of arguments")
    return tuple(str(item) for item in command)


def _environment_values(env: Mapping[object, object] | None) -> dict[str, str]:
    source = os.environ if env is None else env
    return {str(key): str(value) for key, value in source.items()}


@dataclass(frozen=True)
class CommandKey:
    """All inputs that make a supervised result reusable."""

    command: tuple[str, ...]
    cwd: str
    input_manifest: str
    test_digest: str
    environment_digest: str
    sample_id: str
    command_digest: str

    @property
    def fingerprint(self) -> str:
        return _digest({
            "command": self.command,
            "cwd": self.cwd,
            "input_manifest": self.input_manifest,
            "test_digest": self.test_digest,
            "environment_digest": self.environment_digest,
            "sample_id": self.sample_id,
        })


def make_key(command: Sequence[object], cwd: str | os.PathLike[str],
             env: Mapping[object, object] | None, *, input_manifest: str,
             test_digest: str, sample_id: str = "") -> CommandKey:
    arguments = _command_tuple(command)
    environment = _environment_values(env)
    return CommandKey(
        command=arguments,
        cwd=str(Path(cwd).resolve()),
        input_manifest=str(input_manifest),
        test_digest=str(test_digest),
        environment_digest=_digest(sorted(environment.items())),
        sample_id=str(sample_id or ""),
        command_digest=_digest(arguments),
    )


def _deadline_seconds(value: float | None) -> float:
    raw = os.environ.get(DEADLINE_ENV)
    candidate = value if raw is None else raw
    try:
        seconds = float(DEFAULT_DEADLINE_SECONDS if candidate is None else candidate)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{DEADLINE_ENV} must be finite and positive") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{DEADLINE_ENV} must be finite and positive")
    return seconds


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


@dataclass(frozen=True)
class ExecutionReceipt:
    disposition: str
    command_key: str
    snapshot_id: str
    final_snapshot_id: str
    accepted: bool
    timed_out: bool
    process_group_killed: bool
    deadline_seconds: float

    @property
    def note(self) -> str:
        return (
            "[supervised executor] "
            f"disposition={self.disposition} "
            f"snapshot={self.snapshot_id} "
            f"accepted={'true' if self.accepted else 'false'}"
        )


@dataclass(frozen=True)
class ExecutionResult:
    completed: subprocess.CompletedProcess
    receipt: ExecutionReceipt
    timed_out: bool = False

    def for_observation(self) -> subprocess.CompletedProcess:
        """Return the result shape consumed by the existing test interpreter."""
        stdout = _text(self.completed.stdout)
        stderr = _text(self.completed.stderr)
        note = self.receipt.note
        rendered_stdout = stdout + ("\n" if stdout else "") + note
        return subprocess.CompletedProcess(
            self.completed.args,
            1 if not self.receipt.accepted else self.completed.returncode,
            rendered_stdout,
            stderr,
        )

    def timeout_error(self) -> subprocess.TimeoutExpired:
        return subprocess.TimeoutExpired(
            self.completed.args,
            self.receipt.deadline_seconds,
            output=_text(self.completed.stdout),
            stderr=_text(self.completed.stderr),
        )


@dataclass
class _Entry:
    event: threading.Event
    result: ExecutionResult | None = None
    error: BaseException | None = None
    running: bool = True


def _run_process(command: tuple[str, ...], cwd: Path, env: Mapping[str, str] | None,
                 timeout: float, popen_factory: Callable[..., subprocess.Popen]) -> tuple[
                     subprocess.CompletedProcess, bool, bool]:
    del popen_factory  # compatibility parameter; raw subprocess injection is not supported
    from kernel_execution_policy import run_sandboxed
    result = run_sandboxed(command, cwd, env=env, timeout=timeout)
    return result.completed, result.timed_out, result.process_group_killed


class CommandRegistry:
    """Atomic get-or-create registry for one snapshot-keyed command."""

    def __init__(self, *, popen_factory: Callable[..., subprocess.Popen] | None = None):
        # Retained only for legacy test-call compatibility; _run_process ignores
        # it and always delegates production execution to the central policy.
        self._popen_factory = popen_factory
        self._lock = threading.RLock()
        self._entries: dict[CommandKey, _Entry] = {}

    def _receipt(self, key: CommandKey, *, disposition: str, final_snapshot: str,
                 accepted: bool, timed_out: bool, process_group_killed: bool,
                 deadline: float) -> ExecutionReceipt:
        return ExecutionReceipt(
            disposition=disposition,
            command_key=key.fingerprint,
            snapshot_id=key.input_manifest,
            final_snapshot_id=final_snapshot,
            accepted=accepted,
            timed_out=timed_out,
            process_group_killed=process_group_killed,
            deadline_seconds=deadline,
        )

    def _retag(self, result: ExecutionResult, key: CommandKey, disposition: str,
               *, revalidate: bool) -> ExecutionResult:
        final_snapshot = key.input_manifest
        if revalidate:
            try:
                final_snapshot = workspace_revision(Path(key.cwd))
            except (OSError, RuntimeError, ValueError):
                final_snapshot = "UNAVAILABLE"
        accepted = (result.receipt.accepted and not result.timed_out
                    and final_snapshot == key.input_manifest)
        receipt = replace(
            result.receipt,
            disposition=disposition,
            command_key=key.fingerprint,
            final_snapshot_id=final_snapshot,
            accepted=accepted,
        )
        return ExecutionResult(result.completed, receipt, result.timed_out)

    def run(self, command: Sequence[object], cwd: str | os.PathLike[str], *,
            env: Mapping[object, object] | None = None, timeout: float | None = None,
            test_digest: str = "", sample_id: str = "",
            popen_factory: Callable[..., subprocess.Popen] | None = None) -> ExecutionResult:
        root = Path(cwd).resolve()
        deadline = _deadline_seconds(timeout)
        environment = _environment_values(env)
        while True:
            snapshot = workspace_revision(root)
            key = make_key(command, root, environment, input_manifest=snapshot,
                           test_digest=test_digest or _digest(_command_tuple(command)),
                           sample_id=sample_id)
            with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    entry = _Entry(threading.Event())
                    self._entries[key] = entry
                    owner = True
                else:
                    owner = False

            if owner:
                try:
                    completed, timed_out, group_killed = _run_process(
                        key.command, root, environment, deadline,
                        popen_factory or self._popen_factory)
                    try:
                        final_snapshot = workspace_revision(root)
                    except (OSError, RuntimeError, ValueError):
                        final_snapshot = "UNAVAILABLE"
                    accepted = (not timed_out and final_snapshot == snapshot)
                    result = ExecutionResult(
                        completed,
                        self._receipt(
                            key, disposition=EXECUTED, final_snapshot=final_snapshot,
                            accepted=accepted, timed_out=timed_out,
                            process_group_killed=group_killed, deadline=deadline),
                        timed_out,
                    )
                    with self._lock:
                        entry.result = result
                        entry.running = False
                        if not accepted:
                            self._entries.pop(key, None)
                        entry.event.set()
                    return result
                except BaseException as error:
                    with self._lock:
                        entry.error = error
                        entry.running = False
                        self._entries.pop(key, None)
                        entry.event.set()
                    raise

            if entry.running:
                # The owner owns the process deadline.  A waiter has a small
                # grace period so it cannot outlive a failed owner forever.
                entry.event.wait(deadline + 1.0)
                if entry.running:
                    raise subprocess.TimeoutExpired(list(key.command), deadline)
                if entry.error is not None:
                    raise entry.error
                if entry.result is None:
                    raise RuntimeError("supervised command completed without a result")
                return self._retag(entry.result, key, ATTACHED, revalidate=True)

            if entry.error is not None:
                raise entry.error
            if entry.result is None:
                raise RuntimeError("supervised command cache entry has no result")
            # A cache hit is usable only if the host still sees the exact
            # snapshot that produced it.  If it moved while this call was
            # acquiring the lock, recompute the key and execute afresh.
            try:
                current = workspace_revision(root)
            except (OSError, RuntimeError, ValueError):
                current = "UNAVAILABLE"
            if current != snapshot:
                with self._lock:
                    if self._entries.get(key) is entry:
                        self._entries.pop(key, None)
                continue
            return self._retag(entry.result, key, REUSED, revalidate=True)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_GLOBAL_REGISTRY = CommandRegistry()


def reset_global_registry() -> None:
    _GLOBAL_REGISTRY.clear()


def _emit(journal: object, result: ExecutionResult) -> None:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    receipt = result.receipt
    emit(
        "supervised_command_completed",
        producer="kernel_supervised_executor",
        command_key=receipt.command_key,
        snapshot_id=receipt.snapshot_id,
        final_snapshot_id=receipt.final_snapshot_id,
        disposition=receipt.disposition,
        accepted=receipt.accepted,
        timed_out=receipt.timed_out,
        process_group_killed=receipt.process_group_killed,
        deadline_seconds=receipt.deadline_seconds,
        dispatch=False,
    )


def run_supervised(command: Sequence[object], cwd: str | os.PathLike[str], *,
                   env: Mapping[object, object] | None = None,
                   timeout: float | None = None, test_digest: str = "",
                   sample_id: str = "", journal: object = None,
                   registry: CommandRegistry | None = None) -> ExecutionResult:
    result = (registry or _GLOBAL_REGISTRY).run(
        command, cwd, env=env, timeout=timeout, test_digest=test_digest,
        sample_id=sample_id,
    )
    _emit(journal, result)
    return result


__all__ = [
    "ATTACHED", "CommandKey", "CommandRegistry", "DEFAULT_DEADLINE_SECONDS",
    "DEADLINE_ENV", "EXECUTED", "ExecutionReceipt", "ExecutionResult", "MODE_ENV",
    "REUSED", "enabled", "make_key", "mode", "reset_global_registry", "run_supervised",
]
