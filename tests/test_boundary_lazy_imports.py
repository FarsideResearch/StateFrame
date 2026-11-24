"""Container boundary checks stay lazy and recognize both import identities."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from stateframe.engine.kernel_safe_paths import container_active


BOUNDARY_NAMES = (
    "stateframe.engine.kernel_execution_boundary",
    "kernel_execution_boundary",
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("identity", BOUNDARY_NAMES, ids=["package", "flat"])
def test_container_active_checks_each_boundary_identity(monkeypatch, identity):
    for name in BOUNDARY_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    boundary = SimpleNamespace(active=lambda: SimpleNamespace(backend="container"))
    monkeypatch.setitem(sys.modules, identity, boundary)

    assert container_active()


def test_container_active_does_not_miss_container_in_second_identity(monkeypatch):
    for name in BOUNDARY_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, BOUNDARY_NAMES[0],
                        SimpleNamespace(active=lambda: SimpleNamespace(backend="platform")))
    monkeypatch.setitem(sys.modules, BOUNDARY_NAMES[1],
                        SimpleNamespace(active=lambda: SimpleNamespace(backend="container")))

    assert container_active()


@pytest.mark.parametrize("entrypoint", ("cli", "engine-child"))
def test_off_path_entrypoint_imports_no_boundary_identity(entrypoint):
    if entrypoint == "cli":
        startup = "import importlib; importlib.import_module('stateframe.cli')\n"
    else:
        startup = (f"sys.path.insert(0, {str(ROOT / 'src/stateframe')!r})\n"
                   "import runpy\n"
                   f"runpy.run_path({str(ROOT / 'src/stateframe/_child.py')!r}, "
                   "run_name='engine_child')\n")
    code = ("import sys\n" + startup + f"names = {BOUNDARY_NAMES!r}\n"
            "loaded = sorted(name for name in names if name in sys.modules)\n"
            "assert not loaded, loaded\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("STATEFRAME_", "REACT_", "SF_LAB_"))}
    env.update(PYTHONPATH=str(ROOT / "src"), PATH=os.defpath, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                            text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_resume_without_recorded_boundary_keeps_boundary_unimported(tmp_path, monkeypatch):
    from stateframe import runner

    for name in BOUNDARY_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    class StopResume(Exception):
        pass

    def stop_before_engine(*_args, **_kwargs):
        raise StopResume

    monkeypatch.setattr(runner, "_require_resume_host_authorization", stop_before_engine)
    (tmp_path / "spec.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StopResume):
        runner.resume_run(tmp_path)

    assert not any(name in sys.modules for name in BOUNDARY_NAMES)


def test_resume_with_malformed_boundary_record_imports_and_refuses(tmp_path, monkeypatch):
    from stateframe import runner

    for name in BOUNDARY_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # "from .engine import X" reuses an attribute an earlier import left on the package.
    monkeypatch.delattr("stateframe.engine.kernel_execution_boundary", raising=False)
    (tmp_path / "spec.json").write_text('{"execution_boundary": {}}', encoding="utf-8")

    with pytest.raises(Exception) as caught:
        runner.resume_run(tmp_path)

    assert getattr(caught.value, "code", None) == "CONTAINER_RESUME_MISMATCH"
    assert any(name in sys.modules for name in BOUNDARY_NAMES)
