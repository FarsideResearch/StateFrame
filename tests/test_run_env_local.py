import json
import os
import shutil
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


def _project(tmp_path):
    repository = tmp_path / "repository"
    package = repository / "src/sf_editable_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("ORIGIN = 'source'\n")
    (repository / "pyproject.toml").write_text(
        "[build-system]\nrequires = ['setuptools']\nbuild-backend = 'setuptools.build_meta'\n"
        "[project]\nname = 'sf-editable-pkg'\nversion = '1.0'\n")
    workspace = tmp_path / "run/workspace"
    cloned = workspace / "src/sf_editable_pkg"
    cloned.mkdir(parents=True)
    (cloned / "__init__.py").write_text("ORIGIN = 'workspace'\n")
    return repository, workspace


def _import_file(python: Path, script: Path) -> Path:
    script.write_text("import sf_editable_pkg; print(sf_editable_pkg.__file__)\n")
    return Path(command(str(python), str(script))).resolve()


def test_editable_install_imports_workspace_copy(tmp_path):
    repository, workspace = _project(tmp_path)
    base = Path(shutil.which("python3.11") or sys.executable)
    source = tmp_path / "source-env"
    command(str(base), "-m", "venv", "--system-site-packages", str(source))
    command(str(source / "bin/python"), "-m", "pip", "install", "-e", str(repository),
            "--no-build-isolation", "--no-deps", "--no-index")
    assert _import_file(source / "bin/python", tmp_path / "source_import.py").is_relative_to(repository)
    run_env = create_run_env(tmp_path / "run", source / "bin/python", source_repo=repository)
    assert _import_file(run_env.python, tmp_path / "clone_import.py") == (
        workspace / "src/sf_editable_pkg/__init__.py")
    assert not run_env.external_import_roots


def test_finder_mapping_and_namespace_paths_are_remapped(tmp_path):
    repository, workspace = _project(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    unicode_path = repository / "src/café"
    unicode_path.mkdir()
    (workspace / "src/café").mkdir()
    source = tmp_path / "source-env"
    command(sys.executable, "-m", "venv", str(source))
    site = next(source.glob("lib/python*/site-packages"))
    finder = site / "__editable___sf_editable_pkg_1_0_finder.py"
    finder.write_text("MAPPING = {'sf_editable_pkg': " + repr(str(repository / "src/sf_editable_pkg")) +
                      ", 'external_pkg': " + repr(str(external)) + "}\n"
                      "NAMESPACES = {'sf_editable_pkg': [" + repr(str(unicode_path)) + "]}\n"
                      "def install(): pass\n")
    (site / "__editable__.sf_editable_pkg.pth").write_text(
        "import __editable___sf_editable_pkg_1_0_finder; __editable___sf_editable_pkg_1_0_finder.install()\n")
    run_env = create_run_env(tmp_path / "run", source / "bin/python", source_repo=repository)
    copied = next(run_env.root.glob("lib/python*/site-packages/__editable___sf_editable_pkg_1_0_finder.py"))
    text = copied.read_text()
    assert str(workspace / "src/sf_editable_pkg") in text
    assert str(workspace / "src/café") in text
    assert str(repository) not in text
    assert str(external) in text
    assert run_env.external_import_roots == (external.resolve(),)


def test_legacy_egg_link_and_external_path(tmp_path):
    repository, workspace = _project(tmp_path)
    source = tmp_path / "source-env"
    command(sys.executable, "-m", "venv", str(source))
    site = next(source.glob("lib/python*/site-packages"))
    external = tmp_path / "external"
    external.mkdir()
    (site / "project.egg-link").write_text(str(repository) + "\n.\n")
    (site / "external.pth").write_text(str(external) + "\n")
    (site / "relative_external.pth").write_text(os.path.relpath(external, site) + "\n")
    run_env = create_run_env(tmp_path / "run", source / "bin/python", source_repo=repository)
    cloned_site = next(run_env.root.glob("lib/python*/site-packages"))
    assert (cloned_site / "project.egg-link").read_text().splitlines()[0] == str(workspace)
    assert (cloned_site / "external.pth").read_text() == str(external) + "\n"
    assert (cloned_site / "relative_external.pth").read_text() == str(external) + "\n"
    assert run_env.external_import_roots == (external.resolve(),)


def test_project_venv_internal_pth_stays_in_cloned_env(tmp_path):
    repository, _ = _project(tmp_path)
    source = repository / ".venv"
    command(sys.executable, "-m", "venv", str(source))
    site = next(source.glob("lib/python*/site-packages"))
    (site / "internal.pth").write_text(str(site) + "\n")
    run_env = create_run_env(tmp_path / "run", source / "bin/python", source_repo=repository)
    cloned_site = next(run_env.root.glob("lib/python*/site-packages"))
    assert (cloned_site / "internal.pth").read_text() == str(cloned_site) + "\n"


def test_host_provision_records_external_import_root(tmp_path):
    from types import SimpleNamespace
    from kernel_tool_settings import ToolSettings
    from stateframe.shell_host import provision

    repository, _ = _project(tmp_path)
    source = tmp_path / "source-env"
    command(sys.executable, "-m", "venv", str(source))
    external = tmp_path / "external"
    external.mkdir()
    next(source.glob("lib/python*/site-packages")).joinpath("external.pth").write_text(str(external) + "\n")
    run = tmp_path / "run"
    selected = str(source / "bin/python")
    settings = ToolSettings()
    options = SimpleNamespace(tool_settings=settings, _test_python=selected,
                              _capabilities={"test_python": selected})
    spec = {"repo": str(repository), "capabilities": options._capabilities,
            "tool_settings": settings.as_dict()}
    (run / "tool-settings.json").write_text(json.dumps(settings.as_dict()))
    provision(options, run, spec)
    saved = json.loads((run / "spec.json").read_text())
    assert saved["run_env"]["external_import_roots"] == [str(external.resolve())]
    assert str(external.resolve()) in saved["shell_read_roots"]


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
