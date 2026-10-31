"""Per-run dependency environment and inexpensive installed-package identity."""
from __future__ import annotations

import hashlib
import json
import os
import stat
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
    for path in destination.glob("lib/python*/site-packages/*.pth"):
        data = path.read_bytes()
        changed = _replace_root(data, source, destination)
        if data != changed:
            path.write_bytes(changed)
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


def _check_clone_paths(source: Path) -> None:
    """Reject source-tree imports before copying or executing the cloned Python."""
    if (source / "pyvenv.cfg").is_symlink() or (source / "bin").is_symlink():
        raise RunEnvError("symlinked environment configuration or bin directory cannot be cloned")
    for site in sorted(source.glob("lib/python*/site-packages")):
        if any(parent.is_symlink() for parent in (site, *site.parents)
               if parent != source and source in parent.parents):
            raise RunEnvError(f"symlinked site-packages directory cannot be cloned: {site}")
        for path in sorted(site.iterdir()):
            if path.suffix == ".egg-link":
                raise RunEnvError(f"editable install cannot be cloned: {path.name}")
            if path.suffix != ".pth":
                continue
            if path.is_symlink():
                raise RunEnvError(f"symlinked .pth cannot be cloned: {path.name}")
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("import ", "import\t")):
                    # The standard setuptools shim selects its installed distutils.
                    # Other executable .pth files can import editable finders whose
                    # source mappings cannot safely be relocated without executing.
                    setuptools_shim = (
                        "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = "
                        "os.environ.get(var, 'local') == 'local'; enabled and "
                        "__import__('_distutils_hack').add_shim();")
                    if line == setuptools_shim:
                        continue
                    raise RunEnvError(f"executable .pth cannot be safely cloned: {path.name}")
                target = (site / line).resolve()
                if not target.is_relative_to(source.resolve()):
                    raise RunEnvError(f"external import path cannot be cloned: {path.name}: {line}")


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
        _check_clone_paths(source)
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
    # Pin site directories while the host-created interpreter is still trusted.
    sites = json.loads(_run([str(python), "-c",
                             "import json,site; print(json.dumps(site.getsitepackages()))"]))
    (root.parent / "env-sites.json").write_text(json.dumps(sites))
    return run_env


def env_identity(run_env: RunEnv) -> str:
    """Hash installed distribution names/versions and run-local tool-cache metadata."""
    site_manifest = run_env.root.parent / "env-sites.json"
    sites = (json.loads(site_manifest.read_text()) if site_manifest.is_file() else
             [str(path) for path in run_env.root.glob("lib/python*/site-packages")])
    if not isinstance(sites, list) or any(not isinstance(p, str) for p in sites):
        raise RunEnvError("invalid host site-packages manifest")
    distributions: list[tuple[str, str]] = []
    for directory in sorted(set(sites)):
        path = Path(directory)
        mutable = path.is_relative_to(run_env.root)
        if mutable and any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(run_env.root)):
            raise RunEnvError(f"symlinked site-packages metadata: {path}")
        if not path.is_dir():
            continue
        for item in path.iterdir():
            if not item.name.endswith((".dist-info", ".egg-info")):
                continue
            if mutable and item.is_symlink():
                raise RunEnvError(f"symlinked distribution metadata: {item}")
            metadata = item / "METADATA" if item.is_dir() else item
            if not metadata.is_file():
                metadata = item / "PKG-INFO"
            if mutable and metadata.is_symlink():
                raise RunEnvError(f"symlinked package metadata: {metadata}")
            name = item.name
            version = ""
            if metadata.is_file():
                # Metadata descriptions may be huge; identity needs only headers.
                with metadata.open("rb") as source:
                    headers = source.read(65536).decode("utf-8", errors="replace")
                for line in headers.splitlines():
                    if not line.strip():
                        break
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
        if any(p.is_symlink() for p in (base, *base.parents) if p.is_relative_to(run_env.root)):
            raise RunEnvError(f"symlinked tool cache: {base}")
        if not base.exists():
            continue
        for directory, _, files in os.walk(base):
            for filename in files:
                path = Path(directory) / filename
                if not stat.S_ISREG(path.lstat().st_mode):
                    raise RunEnvError(f"non-regular tool cache metadata: {path}")
                relative = path.relative_to(base).as_posix()
                if (filename in manifest_names or relative.startswith("registry/index/") or
                        relative.startswith("_cacache/index-v5/")):
                    digest = hashlib.sha256()
                    with path.open("rb") as source:
                        while block := source.read(1024 * 1024):
                            digest.update(block)
                    caches.append((key, relative, digest.hexdigest()))

    payload = {"distributions": sorted(distributions), "caches": sorted(caches)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_env_test_python(run_env: RunEnv | None, selected_python: str) -> str:
    """One interpreter selection hook for engine and final host checks."""
    return str(run_env.python) if run_env is not None else selected_python
