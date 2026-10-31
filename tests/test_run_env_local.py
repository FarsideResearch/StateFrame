import json
import subprocess
import sys
import time
import zipfile
import pytest
from pathlib import Path

from kernel_run_env import create_run_env, env_identity, RunEnvError
from stateframe.runner import RunOptions, _tests_python, use_run_env_python


def command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=True, timeout=120)
    return result.stdout.strip()


def wheel(directory: Path, name: str) -> Path:
    version = "1.0"
    info = f"{name}-{version}.dist-info"
    path = directory / f"{name}-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{name}/__init__.py", f"IDENTITY = {name!r}\n")
        archive.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/top_level.txt", f"{name}\n")
        archive.writestr(f"{info}/RECORD", "")
    return path


def test_clone_relocates_pip_and_installed_packages(tmp_path, record_property):
    source = tmp_path / "source"
    command(sys.executable, "-m", "venv", str(source))
    first = wheel(tmp_path, "sf_local_one")
    command(str(source / "bin/python"), "-m", "pip", "install", "--no-index", "--no-deps", str(first))
    site = next(source.glob("lib/python*/site-packages"))
    (site / "internal.pth").write_text(str(site) + "\n")
    started = time.perf_counter()
    run_env = create_run_env(tmp_path / "run", source / "bin/python")
    assert run_env.kind == "clone"
    cloned_site = next(run_env.root.glob("lib/python*/site-packages"))
    assert (cloned_site / "internal.pth").read_text() == str(cloned_site) + "\n"
    assert str(run_env.root) in command(str(run_env.python), "-m", "pip", "--version")
    assert command(str(run_env.python), "-c", "import sf_local_one; print(sf_local_one.IDENTITY)") == "sf_local_one"
    initial = env_identity(run_env)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    record_property("clone_and_first_identity_ms", round(elapsed_ms, 2))
    second = wheel(tmp_path, "sf_local_two")
    command(str(run_env.python), "-m", "pip", "install", "--no-index", "--no-deps", str(second))
    assert env_identity(run_env) != initial
    assert command(str(run_env.python), "-c", "import sf_local_two; print(sf_local_two.IDENTITY)") == "sf_local_two"
    assert not (source / "lib").joinpath("sf_local_two").exists()
    assert all(str(run_env.root) in run_env.env[key] for key in
               ("PIP_CACHE_DIR", "npm_config_cache", "CARGO_HOME", "GOPATH", "GOMODCACHE"))


def test_system_site_environment_sees_base_distributions(tmp_path, record_property):
    base = Path(sys.base_prefix) / "bin" / "python3"
    assert base.is_file()
    run_env = create_run_env(tmp_path / "run", base)
    assert run_env.kind == "system-site"
    code = "import importlib.metadata as m,json; print(json.dumps(sorted(d.metadata['Name'] for d in m.distributions())))"
    base_packages = set(json.loads(command(str(base), "-c", code)))
    run_packages = set(json.loads(command(str(run_env.python), "-c", code)))
    assert base_packages <= run_packages
    started = time.perf_counter()
    identity = env_identity(run_env)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    record_property("env_identity_system_site_ms", round(elapsed_ms, 2))
    assert len(identity) == 64


def test_runner_hook_uses_same_python_for_host_and_child(tmp_path):
    source = tmp_path / "source"
    command(sys.executable, "-m", "venv", str(source))
    run_env = create_run_env(tmp_path / "run", source / "bin/python")
    options = RunOptions(repo=tmp_path, task="test", endpoint=None)
    options._capabilities = {}
    options._test_python = sys.executable
    use_run_env_python(options, run_env)
    assert _tests_python(options) == str(run_env.python)
    assert options._capabilities["test_python"] == str(run_env.python)


@pytest.mark.parametrize("name,content", [
    ("source.pth", "/original/repository\n"),
    ("source.pth", "../../../../original/repository\n"),
    ("__editable__.pth", "import __editable___finder; __editable___finder.install()\n"),
    ("source.egg-link", "/original/repository\n"),
])
def test_clone_refuses_external_source_imports_before_copy(tmp_path, name, content):
    source = tmp_path / "source"
    site = source / "lib/python3.11/site-packages"
    site.mkdir(parents=True)
    (source / "bin").mkdir()
    (source / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (site / name).write_text(content)
    with pytest.raises(RunEnvError, match="cannot be.*cloned"):
        create_run_env(tmp_path / "run", source / "bin/python")
    assert not (tmp_path / "run/env").exists()


def test_clone_never_relocates_through_source_pth_symlink(tmp_path):
    source = tmp_path / "source"
    site = source / "lib/python3.11/site-packages"
    site.mkdir(parents=True)
    (source / "bin").mkdir()
    (source / "pyvenv.cfg").write_text("home = /usr/bin\n")
    original = tmp_path / "original.pth"
    content = str(site) + "\n"
    original.write_text(content)
    (site / "linked.pth").symlink_to(original)
    with pytest.raises(RunEnvError, match="symlinked .pth cannot be cloned"):
        create_run_env(tmp_path / "run", source / "bin/python")
    assert original.read_text() == content
    assert not (tmp_path / "run/env").exists()
