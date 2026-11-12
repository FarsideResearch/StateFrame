"""A started execution session in checkpointed state is saved without its process-bound parts."""
from __future__ import annotations

import io
from dataclasses import replace
import os
import sys

import pytest
from _timeouts import scaled

import kernel_checkpoint
import kernel_run_vitals
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession

pytestmark = pytest.mark.host_sandbox
PYTHON = os.path.realpath(sys.executable)


class ShellHolder:
    """Same shape as the frame path that failed: frame['_shell'].session."""

    def __init__(self, session):
        self.session = session


def _ctx():
    return {"framecells": {}, "framefuncs": {}, "cvars": {}, "hostfns": {}, "participants": {}}


def _setup(tmp_path):
    for name in ("workspace", "env", "home", "run"):
        (tmp_path / name).mkdir()
    cap = shell_capability(workspace=tmp_path / "workspace", run_env_dir=tmp_path / "env",
                           scratch_home=tmp_path / "home", network="none", writable=True)
    return tmp_path / "run", cap


def test_started_session_round_trips_and_runs_again(tmp_path):
    run, cap = _setup(tmp_path)
    session = ExecSession(run)
    try:
        session.configure(cap)
        assert session.run([PYTHON, "-I", "-c", "print('before')"], cap).output == b"before\n"
        assert session._control is not None  # the live socket that used to break saving
        payload = {"frame": {"_shell": ShellHolder(session)}}
        data = kernel_checkpoint.dumps(payload, _ctx())
    finally:
        session.close()

    restored = kernel_checkpoint._Unpickler(io.BytesIO(data), _ctx()).load()["frame"]["_shell"].session
    try:
        assert restored._control is None and restored._process is None
        assert restored._directory != session._directory  # fresh exec directory, old artifacts intact
        assert restored._initial_capability == cap
        assert restored.run([PYTHON, "-I", "-c", "print('after')"], cap).output == b"after\n"
    finally:
        restored.close()


def test_planted_session_without_lease_entry_fails_to_pickle(tmp_path, monkeypatch):
    run, cap = _setup(tmp_path)
    session = ExecSession(run)
    try:
        session.run([PYTHON, "-I", "-c", "pass"], cap)
        monkeypatch.delitem(kernel_checkpoint.LEASE_TYPES, ("kernel_exec_session", "ExecSession"))
        payload = {"frame": {"_shell": ShellHolder(session)}}
        with pytest.raises(TypeError, match="socket|lock") as caught:
            kernel_checkpoint.dumps(payload, _ctx())
        # The failure report names where the process-bound object sits.
        assert kernel_checkpoint._process_bound_path(payload).startswith(
            "payload['frame']['_shell']<ShellHolder>.session<ExecSession>.")
        detail = kernel_checkpoint._failure_detail(caught.value)
        assert detail.startswith("TypeError: ") and " at " in detail
    finally:
        session.close()


def test_finished_job_handle_restores_detached(tmp_path):
    run, cap = _setup(tmp_path)
    session = ExecSession(run)
    try:
        handle = session.spawn([PYTHON, "-I", "-c", "print('job')"], replace(cap, workspace_writable=False))
        assert handle.wait(scaled(30)) == 0
        data = kernel_checkpoint.dumps({"jobs": [handle], "session": session}, _ctx())
    finally:
        session.close()
    restored = kernel_checkpoint._Unpickler(io.BytesIO(data), _ctx()).load()
    job = restored["jobs"][0]
    try:
        assert type(job).__name__ == "RestoredJobHandle" and job.id == handle.id
        assert job.lost_bytes == handle.lost_bytes and job.read_available() == b""
        assert job.stop().ok
        with pytest.raises(Exception, match="finished before the checkpoint"):
            job.poll()
    finally:
        restored["session"].close()


def test_run_vitals_report_checkpoint_failures(monkeypatch):
    monkeypatch.setitem(kernel_checkpoint._STATE, "failed", 3)
    monkeypatch.setitem(kernel_checkpoint._STATE, "disabled", "3 consecutive checkpoint failures (x)")
    assert kernel_run_vitals._checkpoint_vitals() == {
        "checkpoint_failures": 3, "checkpoint_saving_disabled": True}
    monkeypatch.setitem(kernel_checkpoint._STATE, "failed", 0)
    monkeypatch.setitem(kernel_checkpoint._STATE, "disabled", None)
    assert kernel_run_vitals._checkpoint_vitals() == {
        "checkpoint_failures": 0, "checkpoint_saving_disabled": False}
