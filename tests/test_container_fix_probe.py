from pathlib import Path

import sys

import pytest

import kernel_execution_boundary as child_boundary
import kernel_probe
from stateframe import runner
from stateframe.engine import kernel_execution_boundary as boundary


def selected(module):
    return module.Boundary(
        module.BACKEND_CONTAINER,
        module.Detection("container", ("mount:docker",), {"pid": "4026533000"}, 1),
    )


def test_model_supplied_python_probe_refuses_before_launch(monkeypatch):
    import kernel_execution_policy as policy

    monkeypatch.setattr(child_boundary, "_active_boundary", selected(child_boundary))

    def capture_spawn(*args, **kwargs):
        pytest.fail(f"model-supplied probe reached sandboxed_popen (backend={policy.capability_report().backend})")

    monkeypatch.setattr(policy, "sandboxed_popen", capture_spawn)
    with pytest.raises((child_boundary.BoundaryRefused, RuntimeError), match="CONTAINER_UNSUPPORTED:"):
        kernel_probe._run_child(Path.cwd(), "print(42)", executable=sys.executable,
                                wall_seconds=2, cpu_seconds=1, memory_bytes=128 * (1 << 20))


def test_capability_probe_does_not_require_a_writable_install(monkeypatch):
    from stateframe.engine import kernel_execution_policy as policy

    options = runner.RunOptions(repo=Path.cwd(), task="check imports", endpoint=None)
    options._execution_boundary = selected(boundary)
    monkeypatch.setattr(boundary, "_active_boundary", options._execution_boundary)
    calls = []
    original_mkdir = Path.mkdir

    def deny_install_write(path, *args, **kwargs):
        if path.is_relative_to(runner.PACKAGE):
            calls.append(path.relative_to(runner.PACKAGE).as_posix())
            raise PermissionError("installation directory is read-only")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_install_write)
    direct = runner._probe_capabilities(options)
    assert not calls, (calls, direct["schema_validation"], policy.capability_report().backend)
    assert all(type(direct[key]["enabled"]) is bool
               for key in ("schema_validation", "yaml", "repo_map_ast")), direct
