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
              env: Mapping[str, str] | None = None) -> ExecResult:
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
        try:
            result = session.run(argv, cap, stdin=stdin, cwd=workspace)
            if not result.cleanup.ok:
                raise ProbeCleanupError("automatic probe cleanup failed: " + result.cleanup.detail)
        finally:
            try:
                session.close()
            except (ExecSessionError, OSError) as error:
                raise ProbeCleanupError("automatic probe cleanup failed: " + str(error)) from error
        return result
