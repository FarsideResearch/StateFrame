from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import dataclass
from email import message_from_bytes
from email.policy import default
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import Tag, sys_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import Version


def _builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_wheelhouse.py"
    spec = importlib.util.spec_from_file_location("offline_wheelhouse_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _Wheel:
    path: Path
    name: str
    version: Version
    tags: frozenset[Tag]
    requirements: tuple[str, ...]


def _read_manifest(path: Path, problems: list[str]) -> dict[str, str]:
    if not path.is_file():
        problems.append("wheelhouse/SHA256SUMS is missing")
        return {}

    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            problems.append(f"wheelhouse/SHA256SUMS line {line_number} is malformed")
            continue
        filename = parts[1].lstrip(" *")
        if filename in entries:
            problems.append(f"wheelhouse/SHA256SUMS lists {filename} more than once")
        entries[filename] = parts[0].lower()
    return entries


def _read_wheels(wheelhouse: Path, problems: list[str]) -> dict[str, list[_Wheel]]:
    wheels: dict[str, list[_Wheel]] = {}
    manifest = _read_manifest(wheelhouse / "SHA256SUMS", problems)

    for path in sorted(wheelhouse.glob("*.whl")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        recorded_digest = manifest.get(path.name)
        if recorded_digest is None:
            problems.append(f"{path.name} is missing from wheelhouse/SHA256SUMS")
        elif recorded_digest != digest:
            problems.append(f"{path.name} SHA256 does not match wheelhouse/SHA256SUMS")

        try:
            filename_name, filename_version, _, tags = parse_wheel_filename(path.name)
            with zipfile.ZipFile(path) as archive:
                metadata_paths = [
                    name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_paths) != 1:
                    raise ValueError("wheel must contain exactly one dist-info/METADATA")
                metadata = message_from_bytes(archive.read(metadata_paths[0]), policy=default)
            metadata_name = metadata.get("Name")
            metadata_version = metadata.get("Version")
            if not metadata_name or not metadata_version:
                raise ValueError("wheel metadata omits Name or Version")
            if canonicalize_name(metadata_name) != canonicalize_name(filename_name):
                raise ValueError("wheel filename and metadata names differ")
            if Version(metadata_version) != filename_version:
                raise ValueError("wheel filename and metadata versions differ")
            wheel = _Wheel(
                path=path,
                name=canonicalize_name(metadata_name),
                version=filename_version,
                tags=frozenset(tags),
                requirements=tuple(metadata.get_all("Requires-Dist", [])),
            )
        except (InvalidWheelFilename, OSError, ValueError, zipfile.BadZipFile) as exc:
            problems.append(f"{path.name} has unreadable wheel metadata: {exc}")
            continue
        wheels.setdefault(wheel.name, []).append(wheel)

    return wheels


def _active(requirement: Requirement, environment: dict[str, str]) -> bool:
    return requirement.marker is None or requirement.marker.evaluate(environment=environment)


def _closure_problems(project_root: Path) -> list[str]:
    problems: list[str] = []
    wheelhouse = project_root / "wheelhouse"
    wheels = _read_wheels(wheelhouse, problems)
    compatible_tags = set(sys_tags())
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    visited: set[tuple[str, Version, str]] = set()

    try:
        project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = [
            *project["project"]["dependencies"],
            *(requirement for extra in _builder().DEFAULT_INSTALL_EXTRAS
              for requirement in project["project"]["optional-dependencies"][extra]),
        ]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        return problems + [f"could not read project dependencies: {exc}"]

    def visit(raw_requirement: str, required_by: str) -> None:
        try:
            requirement = Requirement(raw_requirement)
            if not _active(requirement, marker_environment):
                return
        except (InvalidRequirement, ValueError) as exc:
            problems.append(f"invalid requirement from {required_by}: {raw_requirement!r} ({exc})")
            return

        name = canonicalize_name(requirement.name)
        candidates = [
            wheel
            for wheel in wheels.get(name, [])
            if wheel.version in requirement.specifier
            and not wheel.tags.isdisjoint(compatible_tags)
        ]
        if not candidates:
            requested = f"{name}{requirement.specifier}"
            problems.append(
                f"missing distribution {requested}, required by {required_by}; "
                "no compatible wheel is present in wheelhouse"
            )
            return

        wheel = max(candidates, key=lambda item: (item.version, item.path.name))
        key = (wheel.name, wheel.version, wheel.path.name)
        if key in visited:
            return
        visited.add(key)
        for dependency in wheel.requirements:
            visit(dependency, wheel.name)

    for raw_dependency in dependencies:
        visit(raw_dependency, "pyproject.toml")
    return problems


def test_wheelhouse_contains_runtime_dependency_closure() -> None:
    project_root = Path(__file__).resolve().parents[1]
    problems = _closure_problems(project_root)
    assert not problems, "Offline wheelhouse closure failed:\n- " + "\n- ".join(problems)


def test_default_extras_match_installer_and_pinned_requirements() -> None:
    root = Path(__file__).resolve().parents[1]
    installer = (root / "install.sh").read_text(encoding="utf-8")
    defaults = installer.split('for arg in "$@";', 1)[0]
    enabled = tuple(name.lower() for name in re.findall(r"^([A-Z_]+)=1$", defaults, re.MULTILINE))
    builder = _builder()
    assert builder.DEFAULT_INSTALL_EXTRAS == enabled
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    for extra in enabled:
        declared = {str(Requirement(text)) for text in project["optional-dependencies"][extra]}
        pinned = set(re.findall(r"^([\w-]+==[\w.]+)",
                               (root / f"requirements-{extra}.txt").read_text(), re.MULTILINE))
        assert declared == pinned
    assert set(builder._runtime_roots()) == {
        *project["dependencies"],
        *(text for extra in enabled for text in project["optional-dependencies"][extra]),
    }


@pytest.mark.slow
def test_offline_install_passes_pip_check_and_imports_default_extras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for module in ("venv", "ensurepip"):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            pytest.skip(f"{module} is unavailable in this Python installation")

    import venv

    home = tmp_path / "home"
    cache = tmp_path / "cache"
    pip_cache = tmp_path / "pip-cache"
    for directory in (home, cache, pip_cache):
        directory.mkdir()

    for key in tuple(os.environ):
        if key.startswith("PIP_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    environment = {
        "HOME": str(home),
        "XDG_CACHE_HOME": str(cache),
        "PIP_CACHE_DIR": str(pip_cache),
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    environment = os.environ.copy()

    environment_path = tmp_path / "environment"
    try:
        venv.EnvBuilder(with_pip=True).create(environment_path)
    except subprocess.CalledProcessError as exc:
        output = "\n".join(part for part in (exc.stdout, exc.stderr) if part)
        if "ensurepip is not available" in output or "No module named ensurepip" in output:
            pytest.skip("ensurepip is unavailable in this Python installation")
        pytest.fail(f"Could not create a clean virtual environment:\n{output}", pytrace=False)

    executable = environment_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    project_root = Path(__file__).resolve().parents[1]
    commands = (
        (
            "offline installation",
            [
                str(executable),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(project_root / "wheelhouse"),
                f"{project_root}[repomap]",
            ],
        ),
        ("pip check", [str(executable), "-m", "pip", "check"]),
        ("jsonschema import", [str(executable), "-c", "import jsonschema"]),
        (
            "repo map imports",
            [str(executable), "-c",
             "import tree_sitter, tree_sitter_python; print('tree_sitter and tree_sitter_python imports OK')"],
        ),
    )
    for label, command in commands:
        result = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
        if result.returncode != 0:
            pytest.fail(
                f"{label} failed with exit code {result.returncode}:\n"
                f"{result.stdout}{result.stderr}",
                pytrace=False,
            )
        print(f"{label}: exit {result.returncode}\n{result.stdout}{result.stderr}")


def _installer_environment(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("PIP_") and key not in {"PYTHONHOME", "PYTHONPATH"}}
    environment.update({
        "HOME": str(home), "SF_HOME": str(tmp_path / "install"), "PYTHON": sys.executable,
        "PIP_CONFIG_FILE": os.devnull, "PIP_NO_INDEX": "1", "PYTHONNOUSERSITE": "1",
        "XDG_CACHE_HOME": str(tmp_path / "cache"), "PIP_CACHE_DIR": str(tmp_path / "pip-cache"),
    })
    return environment


def test_offline_installer_missing_grammar_fails_before_creating_environment(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    for relative in ("install.sh", "scripts/build_wheelhouse.py", "pyproject.toml"):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    wheelhouse = source / "wheelhouse"
    wheelhouse.mkdir()
    for path in (root / "wheelhouse").iterdir():
        if path.is_file() and not path.name.startswith("tree_sitter_python-"):
            (wheelhouse / path.name).symlink_to(path)
    result = subprocess.run(
        ["bash", str(source / "install.sh"), "--offline"], cwd=source,
        env=_installer_environment(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "missing pinned wheel: tree-sitter-python==0.25.0" in result.stderr
    assert "offline wheelhouse is missing or incomplete" in result.stderr
    assert not (tmp_path / "install").exists()


@pytest.mark.slow
@pytest.mark.parametrize("repomap", [True, False])
def test_offline_installer_respects_repomap_default_and_opt_out(tmp_path: Path, repomap: bool) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = _installer_environment(tmp_path)
    command = ["bash", str(root / "install.sh"), "--offline"]
    if not repomap:
        command.append("--no-repomap")
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("repo map: on (17 languages:" if repomap else "repo map: OFF") in result.stdout
    executable = tmp_path / "install/venv/bin/python"
    check = subprocess.run([str(executable), "-m", "pip", "check"], env=environment,
                           capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stdout + check.stderr
    imports = subprocess.run([
        str(executable), "-c",
        "import importlib.util; "
        f"assert (importlib.util.find_spec('tree_sitter') is not None) is {repomap}; "
        f"assert (importlib.util.find_spec('tree_sitter_python') is not None) is {repomap}",
    ], env=environment, capture_output=True, text=True, timeout=30)
    assert imports.returncode == 0, imports.stdout + imports.stderr
    print(f"offline installer repomap={repomap}: exit {result.returncode}; "
          f"pip check exit {check.returncode}; imports/absence OK")
