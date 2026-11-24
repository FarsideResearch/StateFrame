from types import SimpleNamespace

import json
import pytest

from stateframe import _child, cli, runner
from stateframe.engine import kernel_execution_boundary as boundary


DETECTED = boundary.Detection("container", ("mount:docker",), {"pid": "4026533000"}, 1)


def _container():
    return boundary.Boundary(boundary.BACKEND_CONTAINER, DETECTED)


@pytest.mark.parametrize("shell", ["read-only", "future-mode"])
def test_cli_refuses_shell_modes_the_container_cannot_enforce(monkeypatch, shell):
    network = "loopback" if shell == "read-only" else "full"
    args = SimpleNamespace(command="run", container_is_sandbox=True, shell=shell,
                           mcp=(), parallel="off", shell_sandbox="on", mutations=None,
                           profile="r2hc3", mutation_hook=(), yolo=False,
                           shell_network=network, shell_network_explicit=True)
    monkeypatch.setattr(boundary, "detect", lambda **_kwargs: DETECTED)
    monkeypatch.setattr(boundary, "authorize", lambda *_args, **_kwargs: _container())

    with pytest.raises(boundary.BoundaryRefused) as error:
        cli._container_boundary_for_command(args)

    assert error.value.code == "CONTAINER_CONFLICT"


@pytest.mark.parametrize("shell", ["read-only", "future-mode"])
def test_runner_refuses_shell_modes_the_container_cannot_enforce(tmp_path, shell):
    options = runner.RunOptions(repo=tmp_path, task="inspect", endpoint=None)
    options._execution_boundary = _container()
    network = "loopback" if shell == "read-only" else "full"
    options.tool_settings = runner.ToolSettings(shell="unrestricted", shell_network=network)
    object.__setattr__(options.tool_settings, "shell", shell)

    with pytest.raises(runner.RunError, match="CONTAINER_CONFLICT"):
        runner._validate_container_launch_options(options)


@pytest.mark.parametrize("shell", ["read-only", "future-mode"])
def test_child_refuses_recorded_shell_modes_the_container_cannot_enforce(monkeypatch, shell):
    monkeypatch.setattr(boundary, "_active_boundary", None)
    monkeypatch.setattr(boundary, "detect", lambda: pytest.fail("activation must not run"))
    monkeypatch.setattr(_child.os, "geteuid", lambda: 1000)
    network = "loopback" if shell == "read-only" else "full"
    spec = {"execution_boundary": _container().record(),
            "tool_settings": {"shell": shell, "shell_network": network}}

    with pytest.raises(boundary.BoundaryRefused) as error:
        _child._activate_execution_boundary(spec, selected=True)

    assert error.value.code == "CONTAINER_CONFLICT"


def test_child_refuses_missing_recorded_shell_before_activation(monkeypatch):
    monkeypatch.setattr(boundary, "_active_boundary", None)
    monkeypatch.setattr(boundary, "detect", lambda: pytest.fail("activation must not run"))
    monkeypatch.setattr(_child.os, "geteuid", lambda: 1000)
    spec = {"execution_boundary": _container().record(),
            "tool_settings": {"shell_network": "full"}}

    with pytest.raises(boundary.BoundaryRefused) as error:
        _child._activate_execution_boundary(spec, selected=True)

    assert error.value.code == "CONTAINER_CONFLICT"


@pytest.mark.parametrize("shell", ["read-only", "future-mode"])
def test_resume_refuses_recorded_shell_modes_before_activation(tmp_path, monkeypatch, shell):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    network = "loopback" if shell == "read-only" else "full"
    spec = {"execution_boundary": _container().record(), "parallel_mode": "off",
            "tool_settings": {"shell": shell, "shell_network": network}}
    spec_path = run_dir / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    original = spec_path.read_bytes()
    monkeypatch.setattr(boundary, "check_resume", lambda *_args: None)
    monkeypatch.setattr(boundary, "activate", lambda *_args: pytest.fail("activation must not run"))

    with pytest.raises(boundary.BoundaryRefused) as error:
        runner.resume_run(run_dir, _execution_boundary=_container())

    assert error.value.code == "CONTAINER_CONFLICT"
    assert spec_path.read_bytes() == original


def test_resume_refuses_missing_recorded_shell_before_activation(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {"execution_boundary": _container().record(), "parallel_mode": "off",
            "tool_settings": {"shell_network": "full"}}
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(boundary, "check_resume", lambda *_args: None)
    monkeypatch.setattr(boundary, "activate", lambda *_args: pytest.fail("activation must not run"))

    with pytest.raises(boundary.BoundaryRefused) as error:
        runner.resume_run(run_dir, _execution_boundary=_container())

    assert error.value.code == "CONTAINER_CONFLICT"


def test_container_accepts_every_enforced_shell_level(tmp_path):
    for shell in ("standard", "unrestricted", "off"):
        options = runner.RunOptions(repo=tmp_path, task="inspect", endpoint=None)
        options._execution_boundary = _container()
        options.tool_settings = runner.ToolSettings(shell=shell, shell_network="full")
        runner._validate_container_launch_options(options)


def test_resume_refuses_host_authorization_before_activation(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    spec = {"execution_boundary": _container().record(), "parallel_mode": "off",
            "tool_settings": {"shell": "unrestricted", "shell_network": "full"}}
    (run_dir / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(boundary, "check_resume", lambda *_args: None)
    monkeypatch.setattr(boundary, "activate", lambda *_args: pytest.fail("activation must not run"))

    with pytest.raises(boundary.BoundaryRefused) as error:
        runner.resume_run(run_dir, _execution_boundary=_container(), host_authorization=True)

    assert error.value.code == "CONTAINER_CONFLICT"
