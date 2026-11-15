"""Bounded automatic diagnostics and provisioning, with no ambient authority."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import logging
import tempfile
from collections.abc import Callable, Mapping, Sequence

if __package__:
    from .kernel_exec_capability import executable_read_roots, shell_capability
    from .kernel_exec_session import ExecResult, ExecSession, ExecSessionError, ProbeFailure
    from .kernel_test_isolation import _interpreter_roots
else:
    from kernel_exec_capability import executable_read_roots, shell_capability
    from kernel_exec_session import ExecResult, ExecSession, ExecSessionError, ProbeFailure
    from kernel_test_isolation import _interpreter_roots


class ProbeCleanupError(BaseException):
    """Fatal containment failure, deliberately outside advisory Exception handlers."""


class ProbeEngineError(ExecSessionError):
    """A fresh second attempt also failed; retain both execution failures."""

    outcome = "ENGINE_ERROR"
    attempt_count = 2

    def __init__(self, failures: tuple[ProbeFailure, ...], error: ExecSessionError):
        self.probe_failures = failures
        messages = [f"attempt {item.attempt} {item.classification}: {item.error}"
                    for item in failures]
        if len(failures) < 2:
            messages.append(f"attempt 2: {error}")
        super().__init__("ENGINE_ERROR: " + "\n".join(messages))


def probe_evidence(value: object) -> dict[str, object] | None:
    failures = getattr(value, "probe_failures", ())
    if not failures:
        return None
    return {"attempt_count": getattr(value, "attempt_count", len(failures) + 1),
            "failures": [item.as_dict() for item in failures]}


def retry_probe(attempt: Callable[[], ExecResult]) -> ExecResult:
    """Retry only a lost execution channel whose launch unit close proved gone."""
    failures: list[ProbeFailure] = []
    for number in (1, 2):
        try:
            result = attempt()
        except ExecSessionError as error:
            if error.supervisor_lost:
                failures.append(ProbeFailure(number, "PROBE_SUPERVISOR_LOST", str(error)))
                # Invalid execution evidence must survive even when a prior
                # code outcome or unproven cleanup forbids another attempt.
                error.outcome = "ENGINE_ERROR"
                error.probe_failures = tuple(failures)
                error.attempt_count = number
                logging.getLogger(__name__).warning(
                    "PROBE_SUPERVISOR_LOST: attempt %d INVALID: %s", number, error)
            eligible = (error.supervisor_lost and error.probe_containment_verified
                        and not error.probe_retry_forbidden)
            if number == 2:
                raise ProbeEngineError(tuple(failures), error) from error
            if not eligible:
                raise
        else:
            return replace(result, probe_failures=tuple(failures)) if failures else result
    raise AssertionError("probe attempt bound exhausted")


def run_probe_session(session: ExecSession, execute: Callable[[], ExecResult], *,
                      cleanup_error: type[ProbeCleanupError] = ProbeCleanupError) -> ExecResult:
    """Close every attempt while preserving the original execution exception."""
    try:
        result = execute()
    except BaseException as original:
        close_error = None
        try:
            session.close()
        except BaseException as error:
            close_error = error
        if isinstance(original, ExecSessionError):
            original.probe_containment_verified = bool(
                getattr(session, "containment_verified", False))
        if close_error is not None:
            if isinstance(original, ExecSessionError):
                # Preserving the primary exception must not downgrade unsafe
                # cleanup into a baseline environment fact in import smoke.
                original.outcome = "ENGINE_ERROR"
            # Collection may already have attached a failed stop(). Preserve
            # that chain instead of replacing it with close()'s implicit cycle.
            previous = original.__cause__ or original.__context__
            if previous is not None and previous is not close_error:
                close_error.__context__ = previous
            raise original from close_error
        raise
    else:
        try:
            session.close()
        except (ExecSessionError, OSError) as error:
            raise cleanup_error("automatic probe cleanup failed: " + str(error)) from error
        if not result.cleanup.ok:
            raise cleanup_error("automatic probe cleanup failed: " + result.cleanup.detail)
        return result


def run_probe(argv: Sequence[str], workspace: Path, *, timeout: float = 10,
              stdin: bytes | None = None, read_roots: Sequence[Path] = (),
              write_roots: Sequence[Path] = (),
              env: Mapping[str, str] | None = None) -> ExecResult:
    """Execute repository-controlled code read-only, offline, with scratch HOME."""
    return retry_probe(lambda: _run_probe_once(
        argv, workspace, timeout=timeout, stdin=stdin,
        read_roots=read_roots, write_roots=write_roots, env=env))


def _run_probe_once(argv: Sequence[str], workspace: Path, *, timeout: float,
                    stdin: bytes | None, read_roots: Sequence[Path],
                    write_roots: Sequence[Path], env: Mapping[str, str] | None) -> ExecResult:
    executable = Path(argv[0]).absolute()
    with tempfile.TemporaryDirectory(prefix="stateframe-probe-") as directory:
        scratch = Path(directory)
        home, environment = scratch / "home", scratch / "env"
        home.mkdir()
        environment.mkdir()
        cap = shell_capability(
            workspace=workspace, run_env_dir=environment, scratch_home=home,
            network="none", writable=False,
            include_path_executable_roots=False,
            extra_read_roots=(*executable_read_roots(executable),
                              *map(Path, _interpreter_roots(executable)), *read_roots))
        child_env = dict(cap.env)
        child_env.update(env or {})
        cap = replace(cap, purpose="probe", env=child_env,
                      write_roots=(home, *write_roots),
                      limits=replace(cap.limits, wall_seconds=timeout, output_bytes=65536))
        session_dir = scratch / "session"
        session_dir.mkdir()
        session = ExecSession(session_dir)
        return run_probe_session(
            session, lambda: session.run(argv, cap, stdin=stdin, cwd=workspace))
