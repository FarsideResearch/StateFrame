from pathlib import Path

import pytest

import kernel_execution_boundary as child_boundary
import kernel_exec_session as execution_session
import kernel_exec_supervisor as supervisor
from kernel_exec_capability import ExecCapability, ExecLimits


def test_supervisor_refuses_a_loopback_only_custom_tool(monkeypatch):
    assert child_boundary.active().backend == "platform"
    root = Path.cwd()
    cap = ExecCapability("custom_tool", root, False, (), (), "loopback", {}, ExecLimits())
    service = object.__new__(supervisor.Supervisor)
    service.jobs = {}
    service.backend = "container"
    service.namespace_network = "host"
    service.next_id = 0
    service._capability_cache = {}
    service.directory = root

    def intercept_launch(_path, **_kwargs):
        pytest.fail("supervisor admitted loopback-only command to the launch path")

    monkeypatch.setattr(supervisor, "JobOutputCollector", intercept_launch)
    request = {"op": "start", "argv": ["true"], "cap": cap.as_dict(), "cwd": str(root)}
    response = service.reply(request)
    assert "CONTAINER_NETWORK" in response.get("error", ""), response


def test_exec_session_refuses_reduced_network_for_container_backend():
    root = Path.cwd()
    cap = ExecCapability("custom_tool", root, False, (), (), "loopback", {}, ExecLimits())
    session = object.__new__(execution_session.ExecSession)
    session._closed = False
    session.backend = "container"

    with pytest.raises(execution_session.ExecSessionError, match="CONTAINER_NETWORK"):
        session._start(cap)
