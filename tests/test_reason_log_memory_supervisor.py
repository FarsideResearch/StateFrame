"""The resident-memory supervisor names which of its three stop paths killed a launch unit.

A measured breach, a supervisor error and post-exit descendant cleanup all end in
_stop_launch_unit, so each must log its own cause. Cleanup is never a memory kill.
"""
import errno
import io
from types import SimpleNamespace

import pytest

import kernel_execution_policy as policy
import kernel_reason_log as rl
from kernel_exec_tracker import ProcessIdentity

LIMIT = 64 << 20
CHILD = ProcessIdentity(200, 123, 123, 123, (20, 0), False)


@pytest.fixture
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    out = io.StringIO()
    rl.configure(level="INFO", stream=out)
    yield out
    rl.reset_for_tests()


def _supervisor(monkeypatch, *, returncode, breach="", rows_error=None, rss=0):
    class Table:
        def refresh_members(self, _members, **_kwargs):
            if rows_error is not None:
                raise rows_error
            return {}

        def snapshot(self):
            return {}

        def resident_memory_limit_breach(self, _live, _memory_bytes):
            return breach

        def accounted_memory_bytes(self, _row):
            return rss

    class Family:
        members = {}
        private_pid_namespace_inode = None

        def __init__(self, *_args):
            pass

        def update(self, _rows):
            return [CHILD]

        def leader_reaped(self):
            pass

    process = SimpleNamespace(pid=123, returncode=returncode)
    monkeypatch.setattr(policy, "ProcessTable", Table)
    monkeypatch.setattr(policy, "Descendants", Family)
    monkeypatch.setattr(policy, "recorded_launch_identity", lambda _process: object())
    supervisor = policy._ResidentMemorySupervisor(process, LIMIT)
    supervisor._stop_launch_unit = lambda: None
    return supervisor


def _causes():
    return {key for key in rl.run_summary()["by_cause"] if "memory_supervisor" in key}


def test_post_exit_cleanup_never_logs_as_a_memory_kill(monkeypatch, stream):
    supervisor = _supervisor(monkeypatch, returncode=0, rss=10 * LIMIT)

    supervisor._run()

    text = stream.getvalue()
    assert "INFO memory_supervisor:" in text
    assert "remaining descendant" in text
    assert "WARN" not in text and "ERROR" not in text
    assert "limit" not in text and "resident memory" not in text
    assert _causes() == {"kill:memory_supervisor:post_exit_cleanup"}
    assert supervisor._stop_cause == "post_exit_cleanup"


def test_measured_breach_logs_bytes_against_the_limit(monkeypatch, stream):
    measured = 3 * LIMIT
    supervisor = _supervisor(monkeypatch, returncode=None,
                             breach="aggregate resident-memory limit exceeded", rss=measured)

    supervisor._run()

    lines = [line for line in stream.getvalue().splitlines() if "memory_supervisor" in line]
    assert len(lines) == 1 and lines[0].startswith("[stateframe] WARN memory_supervisor:")
    assert f"resident memory {measured} bytes exceeded limit {LIMIT} bytes" in lines[0]
    assert "None" not in lines[0]
    assert _causes() == {"kill:memory_supervisor:rss_breach"}


def test_supervisor_error_logs_error_with_the_exception(monkeypatch, stream):
    supervisor = _supervisor(monkeypatch, returncode=None,
                             rows_error=PermissionError(errno.EACCES, "procfs denied"))

    supervisor._run()

    lines = [line for line in stream.getvalue().splitlines() if "memory_supervisor" in line]
    assert len(lines) == 1 and lines[0].startswith("[stateframe] ERROR memory_supervisor:")
    assert "PermissionError" in lines[0]
    assert "rss_breach" not in str(_causes())
    assert _causes() == {"kill:memory_supervisor:supervisor_error"}
