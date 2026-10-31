"""Per-run dependency environment and inexpensive installed-package identity."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import shutil
import sys
import tempfile
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
    external_import_roots: tuple[Path, ...] = ()


def _run(argv: list[str], *, workspace: Path, destination: Path,
         env: Mapping[str, str] | None = None, read_roots: tuple[Path, ...] = ()) -> str:
    if __package__:
        from .kernel_safe_probe import run_probe
        from .kernel_exec_session import ExecSessionError
    else:
        from kernel_safe_probe import run_probe
        from kernel_exec_session import ExecSessionError
    try:
        with tempfile.TemporaryDirectory(prefix="stateframe-provision-") as directory:
            result = run_probe(argv, Path(directory), timeout=120, env=env,
                               write_roots=(destination,), read_roots=read_roots)
    except (OSError, ExecSessionError) as error:
        raise RunEnvError(f"run environment command failed: {argv[0]}: {error}") from error
    if result.timed_out or result.exit_code or result.output_truncated:
        detail = result.output.decode("utf-8", "replace").strip()[-1000:]
        raise RunEnvError(f"run environment command failed ({result.exit_code}, "
                          f"timed_out={result.timed_out}, truncated={result.output_truncated}): {argv[0]}: {detail}")
    return result.output.decode("utf-8", "strict").strip()


def _replace_root(data: bytes, old: Path, new: Path) -> bytes:
    return data.replace(os.fsencode(str(old)), os.fsencode(str(new)))


_SETUPTOOLS_SHIM = (
    "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = "
    "os.environ.get(var, 'local') == 'local'; enabled and "
    "__import__('_distutils_hack').add_shim();"
)
_FINDER_IMPORT = re.compile(r"import (__editable___[A-Za-z_0-9]+_finder); \1\.install\(\);?")


def _import_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RunEnvError(f"cannot read import metadata: {path.name}: {error}") from error


def _mapped_path(value: str, source_repo: Path | None, workspace: Path,
                 source_env: Path, destination: Path) -> str:
    if not Path(value).is_absolute():
        return value
    resolved = Path(value).resolve()
    if resolved.is_relative_to(source_env):
        return str(destination / resolved.relative_to(source_env))
    if source_repo is not None and resolved.is_relative_to(source_repo):
        return str(workspace / resolved.relative_to(source_repo))
    return value


def _external_root(value: str, source_repo: Path | None, source_env: Path,
                   authorized_roots: tuple[Path, ...] = ()) -> Path | None:
    path = Path(value).resolve()
    if path.is_relative_to(source_env) or (source_repo is not None and path.is_relative_to(source_repo)):
        return None
    if path == Path.home() or path in Path.home().parents:
        raise RunEnvError("import metadata cannot grant the home directory or its ancestors")
    # Metadata requests imports; only independent host policy grants reads.
    # Validate even missing paths so a later install cannot activate a grant.
    if not any(path == root or (root.is_dir() and path.is_relative_to(root))
               for root in authorized_roots):
        raise RunEnvError(f"external import path lacks host read authorization: {path}; "
                          "use --shell-read to authorize it")
    return path if path.exists() else None


def _rewrite_finder(path: Path, source_repo: Path | None, workspace: Path,
                    source_env: Path, destination: Path,
                    authorized_roots: tuple[Path, ...] = ()) -> set[Path]:
    if path.is_symlink() or not path.is_file():
        raise RunEnvError(f"editable finder cannot be safely cloned: {path.name}")
    original = _import_text(path)
    try:
        tree = ast.parse(original, filename=str(path))
    except SyntaxError as error:
        raise RunEnvError(f"invalid editable finder: {path.name}: {error.msg}") from error
    lines = original.encode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes: list[tuple[int, int, str]] = []
    roots: set[Path] = set()
    seen: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = {target.id for target in targets if isinstance(target, ast.Name)}
        if not names.intersection({"MAPPING", "NAMESPACES"}):
            continue
        value = node.value
        if value is None:
            raise RunEnvError(f"invalid editable finder mappings: {path.name}")
        try:
            mapping = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError, MemoryError) as error:
            raise RunEnvError(f"invalid editable finder mappings: {path.name}") from error
        if not isinstance(mapping, dict) or not isinstance(value, ast.Dict):
            raise RunEnvError(f"invalid editable finder mappings: {path.name}")
        seen.update(names)
        values = mapping.values()
        for entry in values:
            paths = entry if isinstance(entry, list) else [entry]
            if not all(isinstance(item, str) for item in paths):
                raise RunEnvError(f"invalid editable finder paths: {path.name}")
        for entry in value.values:
            nodes = entry.elts if isinstance(entry, ast.List) else [entry]
            for child in nodes:
                if not isinstance(child, ast.Constant) or not isinstance(child.value, str):
                    raise RunEnvError(f"invalid editable finder paths: {path.name}")
                if not Path(child.value).is_absolute():
                    continue
                external = _external_root(child.value, source_repo, source_env, authorized_roots)
                if external is not None:
                    roots.add(external)
                replacement = _mapped_path(child.value, source_repo, workspace,
                                           source_env, destination)
                if replacement != child.value:
                    start = offsets[child.lineno - 1] + child.col_offset
                    end = offsets[child.end_lineno - 1] + child.end_col_offset
                    changes.append((start, end, repr(replacement)))
    if not {"MAPPING", "NAMESPACES"}.issubset(seen):
        raise RunEnvError(f"invalid editable finder mappings: {path.name}")
    rewritten = original.encode("utf-8")
    for start, end, replacement in sorted(changes, reverse=True):
        rewritten = rewritten[:start] + replacement.encode("utf-8") + rewritten[end:]
    if changes:
        path.write_bytes(rewritten)
    return roots


def _relocate(source: Path, destination: Path, source_repo: Path | None,
              workspace: Path, authorized_roots: tuple[Path, ...] = ()) -> tuple[Path, ...]:
    config = destination / "pyvenv.cfg"
    if not config.is_file():
        raise RunEnvError(f"cloned environment has no pyvenv.cfg: {destination}")
    config.write_bytes(_replace_root(config.read_bytes(), source, destination))
    roots: set[Path] = set()
    for site in destination.glob("lib/python*/site-packages"):
        for path in sorted(site.iterdir()):
            if path.suffix not in (".pth", ".egg-link"):
                continue
            if path.is_symlink():
                raise RunEnvError(f"symlinked {path.suffix} cannot be cloned: {path.name}")
            lines = _import_text(path).splitlines(keepends=True)
            revised = []
            for line in lines:
                value = line.strip()
                match = _FINDER_IMPORT.fullmatch(value) if path.suffix == ".pth" else None
                if match:
                    roots.update(_rewrite_finder(site / (match.group(1) + ".py"),
                                                 source_repo, workspace, source, destination, authorized_roots))
                    revised.append(line)
                elif not value or value.startswith("#") or value == _SETUPTOOLS_SHIM:
                    revised.append(line)
                elif value.startswith(("import ", "import\t")):
                    raise RunEnvError(f"executable .pth cannot be safely cloned: {path.name}")
                else:
                    # Relative entries are interpreted from the source site before copying.
                    original_site = source / site.relative_to(destination)
                    old_target = (original_site / value).resolve()
                    external = _external_root(str(old_target), source_repo, source, authorized_roots)
                    if external is not None:
                        roots.add(external)
                    mapped = _mapped_path(str(old_target), source_repo, workspace,
                                          source, destination)
                    if mapped != str(old_target):
                        revised.append(line.replace(value, mapped, 1))
                    elif external is not None and not Path(value).is_absolute():
                        revised.append(line.replace(value, str(old_target), 1))
                    else:
                        revised.append(line)
            changed = "".join(revised)
            if changed != "".join(lines):
                path.write_text(changed, encoding="utf-8")
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
    return tuple(sorted(roots))


def _check_clone_paths(source: Path) -> None:
    """Reject unsafe import hooks before copying or executing the cloned Python."""
    if (source / "pyvenv.cfg").is_symlink() or (source / "bin").is_symlink():
        raise RunEnvError("symlinked environment configuration or bin directory cannot be cloned")
    for site in sorted(source.glob("lib/python*/site-packages")):
        if any(parent.is_symlink() for parent in (site, *site.parents)
               if parent != source and source in parent.parents):
            raise RunEnvError(f"symlinked site-packages directory cannot be cloned: {site}")
        for path in sorted(site.iterdir()):
            if path.suffix == ".egg-link" and path.is_symlink():
                raise RunEnvError(f"symlinked .egg-link cannot be cloned: {path.name}")
            if path.suffix != ".pth":
                continue
            if path.is_symlink():
                raise RunEnvError(f"symlinked .pth cannot be cloned: {path.name}")
            for line in _import_text(path).splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("import ", "import\t")):
                    # The standard setuptools shim selects its installed distutils.
                    # Other executable .pth files can import editable finders whose
                    # source mappings cannot safely be relocated without executing.
                    if line == _SETUPTOOLS_SHIM or _FINDER_IMPORT.fullmatch(line):
                        continue
                    raise RunEnvError(f"executable .pth cannot be safely cloned: {path.name}")


def create_run_env(run_dir: Path, test_python: Path, *, source_repo: Path | None = None,
                   authorized_read_roots: tuple[Path, ...] = ()) -> RunEnv:
    root = Path(run_dir).absolute() / "env"
    interpreter = Path(test_python).absolute()
    source = interpreter.parent.parent if interpreter.parent.name == "bin" else None
    source_repo = source_repo.resolve(strict=True) if source_repo is not None else None
    workspace = Path(run_dir).absolute() / "workspace"
    clone = source is not None and (source / "pyvenv.cfg").is_file()
    if root.exists() or root.is_symlink():
        raise RunEnvError(f"run environment already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    if clone:
        assert source is not None
        if __package__:
            from .kernel_test_isolation import _interpreter_roots
        else:
            from kernel_test_isolation import _interpreter_roots
        _check_clone_paths(source)
        # Copy links as links; no source interpreter or startup hook is run.
        shutil.copytree(source, root, symlinks=True)
        authorized = tuple(sorted({path.resolve(strict=True) for path in authorized_read_roots} |
                                  {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()} |
                                  {Path(path).resolve(strict=True) for path in _interpreter_roots(interpreter)} |
                                  {path.resolve(strict=True) for path in root.glob("lib/python*/site-packages")}))
        external_roots = _relocate(source.resolve(), root, source_repo, workspace, authorized)
        kind: Literal["clone", "system-site"] = "clone"
    else:
        root.mkdir()
        _run([str(interpreter), "-I", "-S", "-m", "venv", "--system-site-packages", str(root)],
             workspace=root, destination=root)
        kind = "system-site"
        external_roots = ()
    bin_dir = root / "bin"
    python = bin_dir / "python"
    cache = root / "cache"
    overrides = {
        "VIRTUAL_ENV": str(root),
        "PATH": str(bin_dir) + os.pathsep + os.defpath,
        "PIP_CACHE_DIR": str(cache / "pip"),
        "npm_config_cache": str(cache / "npm"),
        "CARGO_HOME": str(cache / "cargo"),
        "GOPATH": str(cache / "go"),
        "GOMODCACHE": str(cache / "go" / "pkg" / "mod"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    run_env = RunEnv(root, python, bin_dir, overrides, kind, external_roots)
    _run([str(python), "-I", "-B", "-m", "pip", "--version"], workspace=root,
         destination=root, env=overrides, read_roots=external_roots)
    # Site roots are host-derived metadata, never paths printed by package code.
    sites = {str(path.resolve()) for path in root.glob("lib/python*/site-packages")}
    config = dict(line.split("=", 1) for line in (root / "pyvenv.cfg").read_text().splitlines()
                  if "=" in line)
    config = {key.strip(): value.strip() for key, value in config.items()}
    if config.get("include-system-site-packages", "false").lower() == "true":
        prefix = interpreter.resolve(strict=True).parent.parent
        sites.update(str(path.resolve()) for path in prefix.glob("lib/python*/site-packages"))
    (root.parent / "env-sites.json").write_text(json.dumps(sorted(sites)))
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
