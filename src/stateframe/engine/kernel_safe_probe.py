"""Bounded automatic diagnostics and provisioning, with no ambient authority."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from collections.abc import Mapping, Sequence

if __package__:
    from .kernel_exec_capability import executable_read_roots, shell_capability
    from .kernel_exec_session import ExecResult, ExecSession, ExecSessionError
    from .kernel_test_isolation import _interpreter_roots
else:
    from kernel_exec_capability import executable_read_roots, shell_capability
    from kernel_exec_session import ExecResult, ExecSession, ExecSessionError
    from kernel_test_isolation import _interpreter_roots


class ProbeCleanupError(BaseException):
    """Fatal containment failure, deliberately outside advisory Exception handlers."""


def run_probe(argv: Sequence[str], workspace: Path, *, timeout: float = 10,
              stdin: bytes | None = None, read_roots: Sequence[Path] = (),
              write_roots: Sequence[Path] = (),
              env: Mapping[str, str] | None = None,
              diagnostics_on_failure: bool = False) -> ExecResult:
    """Execute repository-controlled code read-only, offline, with scratch HOME."""
    executable = Path(argv[0]).absolute()
    with tempfile.TemporaryDirectory(prefix="stateframe-probe-") as directory:
        scratch = Path(directory)
        home, environment = scratch / "home", scratch / "env"
        home.mkdir()
        environment.mkdir()
        cap = shell_capability(
            workspace=workspace, run_env_dir=environment, scratch_home=home,
            network="none", writable=False,
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
        primary_error: BaseException | None = None
        primary_traceback = None
        result = None
        try:
            result = session.run(argv, cap, stdin=stdin, cwd=workspace)
            if not result.cleanup.ok:
                primary_error = ProbeCleanupError(
                    "automatic probe cleanup failed: " + result.cleanup.detail)
        except BaseException as error:
            primary_error = error
            primary_traceback = error.__traceback__

        cleanup_error = None
        try:
            session.close()
        except (ExecSessionError, OSError) as error:
            cleanup_error = error

        if cleanup_error is not None:
            message = "automatic probe cleanup failed: " + str(cleanup_error)
            if primary_error is not None:
                message += "\nprimary error: " + str(primary_error)
            if diagnostics_on_failure:
                message += "\nstartup supervisor diagnostics:\n" + _session_diagnostics(session)
            raise ProbeCleanupError(message) from cleanup_error

        if primary_error is not None:
            if diagnostics_on_failure:
                _append_startup_diagnostics(primary_error, _session_diagnostics(session))
            raise primary_error.with_traceback(primary_traceback)

        if result is None:
            raise RuntimeError("execution session returned no probe result")
        if diagnostics_on_failure and (result.timed_out or result.exit_code != 0
                                       or result.output_truncated):
            diagnostic = _session_diagnostics(session)
            result = replace(
                result,
                output=(result.output + b"\nstartup supervisor diagnostics:\n"
                        + diagnostic.encode("utf-8")),
            )
        return result


def _session_diagnostics(session: ExecSession) -> str:
    reporter = getattr(session, "startup_diagnostics", None)
    if callable(reporter):
        return reporter()
    # Test and embedding boundaries without a retained supervisor cannot prove more.
    return "supervisor status unknown\nsupervisor.log absent"


def _append_startup_diagnostics(error: BaseException, diagnostic: str) -> None:
    error.args = (f"{error}\nstartup supervisor diagnostics:\n{diagnostic}",)
