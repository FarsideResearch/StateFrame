"""Per-run dependency environment and inexpensive installed-package identity."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


class RunEnvError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunEnv:
    root: Path
    python: Path
    bin_dir: Path
    env: Mapping[str, str]
    kind: Literal["clone", "system-site"]


def _run(argv: list[str], *, env: Mapping[str, str] | None = None) -> str:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False,
                                timeout=120, env=dict(env) if env is not None else None)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RunEnvError(f"run environment command failed: {argv[0]}: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise RunEnvError(f"run environment command failed ({result.returncode}): "
                          f"{argv[0]}: {detail}")
    return result.stdout.strip()


def _replace_root(data: bytes, old: Path, new: Path) -> bytes:
    return data.replace(os.fsencode(str(old)), os.fsencode(str(new)))


def _relocate(source: Path, destination: Path) -> None:
    config = destination / "pyvenv.cfg"
    if not config.is_file():
        raise RunEnvError(f"cloned environment has no pyvenv.cfg: {destination}")
    config.write_bytes(_replace_root(config.read_bytes(), source, destination))
    bin_dir = destination / "bin"
    for script in bin_dir.iterdir():
        if not script.is_file() or script.is_symlink():
            continue
        data = script.read_bytes()
        if b"\x00" in data:
            continue
        changed = _replace_root(data, source, destination)
        if changed != data:
            script.write_bytes(changed)
    # macOS framework venvs can carry activation scripts under alternate shells.
    for script in bin_dir.glob("activate*"):
        if script.is_file() and not script.is_symlink():
            data = script.read_bytes()
            changed = _replace_root(data, source, destination)
            if changed != data:
                script.write_bytes(changed)


def _import_probe(python: Path) -> str:
    code = (
        "import importlib.util, importlib.metadata as m, json, re\n"
        "c = []\n"
        "for d in m.distributions():\n"
        "    name = d.metadata.get('Name', '')\n"
        "    c.extend((name, x.strip()) for x in (d.read_text('top_level.txt') or '').splitlines())\n"
        "    c.append((name, name.replace('-', '_')))\n"
        "c.sort(key=lambda x: (x[0].lower() in ('pip', 'setuptools', 'wheel'), x))\n"
        "ok = next((module for _, module in c if re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', module) "
        "and importlib.util.find_spec(module) is not None), None)\n"
        "print(json.dumps(ok))\n"
    )
    return _run([str(python), "-c", code])


def create_run_env(run_dir: Path, test_python: Path) -> RunEnv:
    root = Path(run_dir).absolute() / "env"
    interpreter = Path(test_python).absolute()
    source = interpreter.parent.parent if interpreter.parent.name == "bin" else None
    clone = source is not None and (source / "pyvenv.cfg").is_file()
    if root.exists() or root.is_symlink():
        raise RunEnvError(f"run environment already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    if clone:
        assert source is not None
        if sys.platform == "darwin":
            command = ["cp", "-a", "-c", str(source), str(root)]
        elif sys.platform.startswith("linux"):
            command = ["cp", "-a", "--reflink=auto", str(source), str(root)]
        else:
            raise RunEnvError(f"unsupported run environment platform: {sys.platform}")
        _run(command)
        _relocate(source, root)
        kind: Literal["clone", "system-site"] = "clone"
    else:
        _run([str(interpreter), "-m", "venv", "--system-site-packages", str(root)])
        kind = "system-site"
    bin_dir = root / "bin"
    python = bin_dir / "python"
    environment = dict(os.environ)
    cache = root / "cache"
    overrides = {
        "VIRTUAL_ENV": str(root),
        "PATH": str(bin_dir) + os.pathsep + environment.get("PATH", ""),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "npm_config_cache": str(cache / "npm"),
        "CARGO_HOME": str(cache / "cargo"),
        "GOPATH": str(cache / "go"),
        "GOMODCACHE": str(cache / "go" / "pkg" / "mod"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment.update(overrides)
    run_env = RunEnv(root, python, bin_dir, overrides, kind)
    _run([str(python), "-m", "pip", "--version"], env=environment)
    if clone:
        package = json.loads(_import_probe(interpreter))
        if package is None:
            raise RunEnvError(f"source environment has no importable installed package: {source}")
        code = "import importlib,sys; importlib.import_module(sys.argv[1])"
        _run([str(python), "-c", code, package], env=environment)
    return run_env


def env_identity(run_env: RunEnv) -> str:
    """Hash installed distribution names/versions and run-local tool-cache metadata."""
    site_code = "import json,site; print(json.dumps(site.getsitepackages()))"
    sites = json.loads(_run([str(run_env.python), "-c", site_code]))
    distributions: list[tuple[str, str]] = []
    for directory in sorted(set(sites)):
        path = Path(directory)
        if not path.is_dir():
            continue
        for item in path.iterdir():
            if not item.name.endswith((".dist-info", ".egg-info")):
                continue
            metadata = item / "METADATA" if item.is_dir() else item
            if not metadata.is_file():
                metadata = item / "PKG-INFO"
            name = item.name
            version = ""
            if metadata.is_file():
                for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.startswith("Name: "):
                        name = line[6:].strip().lower()
                    elif line.startswith("Version: "):
                        version = line[9:].strip()
                    if version and not name.endswith((".dist-info", ".egg-info")):
                        break
            distributions.append((name, version))
    caches = []
    manifest_names = {"package.json", "package-lock.json", "npm-shrinkwrap.json",
                      "Cargo.toml", "Cargo.lock", "go.mod", "go.sum", "config.json"}
    for key in ("npm_config_cache", "CARGO_HOME", "GOPATH", "GOMODCACHE"):
        base = Path(run_env.env[key])
        if not base.exists():
            continue
        for directory, _, files in os.walk(base):
            for filename in files:
                path = Path(directory) / filename
                info = path.lstat()
                if path.is_symlink():
                    continue
                relative = path.relative_to(base).as_posix()
                if (filename in manifest_names or relative.startswith("registry/index/") or
                        relative.startswith("_cacache/index-v5/")):
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            digest.update(block)
                    caches.append((key, relative, digest.hexdigest()))
                else:
                    caches.append((key, relative, info.st_size, info.st_mtime_ns))
    payload = {"distributions": sorted(distributions), "caches": sorted(caches)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_env_test_python(run_env: RunEnv | None, selected_python: str) -> str:
    """One interpreter selection hook for engine and final host checks."""
    return str(run_env.python) if run_env is not None else selected_python
