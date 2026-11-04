"""Rebuild a run-local Python environment from its sealed manifest."""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import storage_gc
from .engine.kernel_safe_paths import safe_expanduser, safe_home


class RehydrateError(RuntimeError):
    """A run environment cannot be rebuilt from its saved manifest."""


_DIST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_DIST_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$")


@dataclass(frozen=True)
class RehydrateResult:
    exact: bool | None
    used_seed: bool
    dry_run: bool
    mismatches: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return 1 if self.exact is False else 0


def _load_manifest(run_dir: Path) -> dict[str, object]:
    path = run_dir / "env-manifest.json"
    if not path.exists():
        # Runs made before environment manifests (0.3.16 and older) or that kept their env.
        if (run_dir / "env").is_dir():
            raise RehydrateError(f"{run_dir} has no env-manifest.json; it still has its environment "
                                 "in env/, so there is nothing to rehydrate")
        raise RehydrateError(f"{run_dir} has no env-manifest.json, so its environment cannot be "
                             "rebuilt; runs made before StateFrame recorded one have no manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RehydrateError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("distributions"), list):
        raise RehydrateError("env-manifest.json is not a supported environment manifest")
    identities = set()
    for row in value["distributions"]:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not _DIST_NAME.fullmatch(row["name"])
                or not isinstance(row.get("version"), str)
                or not _DIST_VERSION.fullmatch(row["version"])):
            raise RehydrateError("env-manifest.json has invalid distribution records")
        identity = row["name"].lower().replace("_", "-").replace(".", "-")
        identity = re.sub(r"-+", "-", identity)
        if identity in identities:
            raise RehydrateError("env-manifest.json has duplicate distribution names")
        identities.add(identity)
    if not isinstance(value.get("python_version"), str) or not value["python_version"]:
        raise RehydrateError("env-manifest.json has no recorded Python version")
    if not isinstance(value.get("source_interpreter_path"), str):
        raise RehydrateError("env-manifest.json has no recorded interpreter path")
    return value


def _seed_path(manifest: dict[str, object]) -> Path | None:
    key = manifest.get("seed_key")
    if not isinstance(key, str) or not key or Path(key).name != key or key.startswith("."):
        return None
    try:
        store = storage_gc.stateframe_home() / "env-seeds"
    except ValueError as error:
        raise RehydrateError(str(error)) from error
    if store.is_symlink() or not store.is_dir():
        return None
    seed = store / key
    return seed if seed.is_dir() and not seed.is_symlink() else None


@contextlib.contextmanager
def _seed_lock(seed: Path):
    """Coordinate seed verification/copy with the seed builder and garbage collector."""
    path = seed.parent / f".{seed.name}.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        locked = True
        yield True
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _same_distributions(expected: list[dict[str, object]], actual: list[dict[str, object]]) -> bool:
    def identity(rows):
        return sorted((row.get("name"), row.get("version"), row.get("record_sha256"))
                      for row in rows)
    return identity(expected) == identity(actual)


def _seed_matches(seed: Path, manifest: dict[str, object]) -> bool:
    key = str(manifest.get("seed_key") or "")
    try:
        from .engine import kernel_env_seed
        seed_json = json.loads((seed / "seed.json").read_text(encoding="utf-8"))
        interpreter = seed / "bin" / ("python.exe" if os.name == "nt" else "python")
        if (not (seed / "pyvenv.cfg").is_file() or not interpreter.is_file()
                or not os.access(interpreter, os.X_OK)):
            return False
        if seed_json.get("key") != key or not kernel_env_seed._valid(seed, key):
            return False
        config = storage_gc._parse_config(seed / "pyvenv.cfg")
        # venv writes "version", uv writes "version_info": the manifest reads them the same way.
        if ((config.get("version") or config.get("version_info")) != manifest.get("python_version")
                or config.get("implementation") != manifest.get("python_implementation")
                # sys.platform and the machine decide wheel compatibility; a kernel or OS
                # patch release changes platform.platform() without invalidating the seed.
                or manifest.get("platform") != sys.platform
                or manifest.get("machine", platform.machine()) != platform.machine()):
            return False
        actual = storage_gc.distribution_inventory(seed)
        expected = manifest["distributions"]
        return isinstance(expected, list) and _same_distributions(expected, actual)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def _make_writable(root: Path) -> None:
    for directory, dirs, names in os.walk(root, topdown=False, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
        for name in dirs:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
    root.chmod(stat.S_IMODE(root.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)


def _relocate_staging_paths(source: Path, destination: Path) -> None:
    """Rewrite venv launch paths after a staged environment is published at env/."""
    from .engine.kernel_run_env import _replace_root

    bin_dir = destination / "bin"
    if bin_dir.is_symlink() or not bin_dir.is_dir():
        raise RehydrateError("rehydrated environment has no regular bin directory")
    candidates = {destination / "pyvenv.cfg", *bin_dir.iterdir()}
    for site in destination.glob("lib/python*/site-packages"):
        if site.is_symlink() or not site.is_dir():
            continue
        candidates.update(site.glob("*.pth"))
        candidates.update(site.glob("*.egg-link"))
        candidates.update(site.glob("__editable___*_finder.py"))
    for path in sorted(candidates):
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            continue
        relocated = _replace_root(data, source, destination)
        if relocated != data:
            path.write_bytes(relocated)


def _copy_seed(seed: Path, destination: Path, run_dir: Path, manifest: dict[str, object]) -> None:
    shutil.copytree(seed, destination, symlinks=True,
                    ignore=shutil.ignore_patterns("seed.json"))
    _make_writable(destination)
    try:
        seed_data = json.loads((seed / "seed.json").read_text(encoding="utf-8"))
        inputs = seed_data.get("inputs") or {}
        relocation = inputs.get("relocation") or {}
        source_text = relocation.get("source")
        if not isinstance(source_text, str):
            return
        from .engine.kernel_run_env import _relocate, relocation_roots

        source_repo = relocation.get("source_repo")
        if not isinstance(source_repo, str):
            spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
            source_repo = spec.get("repo") if isinstance(spec, dict) else None
        authorized = relocation.get("authorized_roots")
        roots = tuple(Path(value) for value in authorized if isinstance(value, str)) \
            if isinstance(authorized, list) else ()
        interpreter = Path(str(manifest.get("source_interpreter_path") or ""))
        # The same grant run creation used, so a seed that cloned then relocates now.
        if interpreter.is_file():
            roots = relocation_roots(interpreter, destination, roots)
        _relocate(Path(source_text), destination,
                  Path(source_repo) if isinstance(source_repo, str) else None,
                  run_dir / "workspace", roots)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise RehydrateError(f"could not relocate the environment seed: {error}") from error


def _probe_python(interpreter: Path) -> tuple[str, str]:
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise RehydrateError(f"recorded Python interpreter is unavailable: {interpreter}")
    code = ("import json,sys; print(json.dumps({'version': '.'.join(map(str,sys.version_info[:3])), "
            "'implementation': sys.implementation.name}))")
    try:
        result = subprocess.run([str(interpreter), "-I", "-S", "-c", code], check=True,
                                capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise RehydrateError(f"recorded Python interpreter could not be checked: {error}") from error
    return str(data.get("version")), str(data.get("implementation"))


def _wheel_directories(run_dir: Path, manifest: dict[str, object]) -> list[Path]:
    try:
        candidates = [run_dir / "wheelhouse", storage_gc.stateframe_home() / "wheelhouse"]
    except ValueError as error:
        raise RehydrateError(str(error)) from error
    user_home = safe_home()
    if user_home is not None:
        candidates.append(user_home / ".cache" / "pip")
    try:
        spec = json.loads((run_dir / "spec.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        spec = {}
    repo = spec.get("repo") if isinstance(spec, dict) else None
    if isinstance(repo, str):
        candidates.append(Path(repo) / "wheelhouse")
    cache = os.environ.get("PIP_CACHE_DIR")
    if cache:
        candidates.append(safe_expanduser(Path(cache)))
    directories = set()
    for candidate in candidates:
        if not candidate.is_dir() or candidate.is_symlink():
            continue
        if candidate.name == "wheelhouse":
            directories.add(candidate)
        try:
            directories.update(path.parent for path in candidate.rglob("*.whl") if path.is_file())
        except OSError:
            continue
    return sorted(directories)


def _local_project(row: dict[str, object]) -> Path | None:
    """The source directory of a direct-URL (local or editable) distribution, if it still exists."""
    direct_url = row.get("direct_url")
    url = direct_url.get("url") if isinstance(direct_url, dict) else None
    if not isinstance(url, str) or not url.startswith("file://"):
        return None
    path = Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path))
    return path if path.is_dir() else None


def _build_environment(destination: Path, manifest: dict[str, object], *, run_dir: Path,
                       offline: bool, printer=print) -> None:
    interpreter = safe_expanduser(Path(str(manifest["source_interpreter_path"])))
    version, implementation = _probe_python(interpreter)
    if version != manifest["python_version"]:
        raise RehydrateError(f"recorded Python version mismatch: expected {manifest['python_version']}, got {version}")
    expected_implementation = manifest.get("python_implementation")
    if (expected_implementation and implementation.casefold()
            != str(expected_implementation).casefold()):
        raise RehydrateError("recorded Python implementation mismatch: "
                             f"expected {expected_implementation}, got {implementation}")
    command = [str(interpreter), "-I", "-m", "venv"]
    if manifest.get("system_site_packages") is True:
        command.append("--system-site-packages")
    command.append(str(destination))
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as error:
        raise RehydrateError(f"could not create the recorded Python environment: {error}") from error
    distributions = manifest["distributions"]
    # Index requirements only for plain distributions. A direct-URL project (a local checkout, an
    # editable install such as StateFrame itself) is not on the index at that version: install it
    # from its recorded directory when that still exists, else leave it for the mismatch report.
    requirements = sorted({f"{row['name']}=={row['version']}" for row in distributions
                           if not row.get("direct_url")})
    python = destination / "bin" / ("python.exe" if os.name == "nt" else "python")
    install = [str(python), "-I", "-m", "pip", "install"]
    if offline:
        install.append("--no-index")
        for directory in _wheel_directories(run_dir, manifest):
            install.extend(("--find-links", str(directory)))
    if requirements:
        try:
            subprocess.run([*install, *requirements], check=True, capture_output=True, text=True,
                           timeout=900)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RehydrateError(f"could not install the recorded distributions: {detail[-2000:]}") from error
        except (OSError, subprocess.SubprocessError) as error:
            raise RehydrateError(f"could not install the recorded distributions: {error}") from error
    for row in distributions:
        if not row.get("direct_url"):
            continue
        project = _local_project(row)
        if project is None:
            printer(f"warning: {row['name']} {row['version']} was installed from a direct URL that is "
                    "not available here; it is not restored")
            continue
        target = ["-e", str(project)] if row.get("editable") is True else [str(project)]
        try:
            subprocess.run([*install, "--no-deps", *target], check=True, capture_output=True,
                           text=True, timeout=900)
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            printer(f"warning: {row['name']} could not be installed from {project}: "
                    f"{str(detail).strip()[-500:]}")
    # venv seeds pip, which the run's environment may not have had (uv-made envs do not).
    if not any(str(row.get("name", "")).lower() == "pip" for row in distributions):
        try:
            subprocess.run([str(python), "-I", "-m", "pip", "uninstall", "-y", "pip"], check=True,
                           capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.SubprocessError) as error:
            printer(f"warning: pip was added by venv and could not be removed: {error}")


def _source_url(row: dict[str, object]) -> object:
    direct_url = row.get("direct_url")
    return direct_url.get("url") if isinstance(direct_url, dict) else None


def compare_distributions(expected: list[dict[str, object]],
                          actual: list[dict[str, object]]) -> list[str]:
    expected_by_name = {str(row["name"]): row for row in expected}
    actual_groups: dict[str, list[dict[str, object]]] = {}
    for row in actual:
        actual_groups.setdefault(str(row["name"]), []).append(row)
    mismatches = []
    for name in sorted(set(expected_by_name) | set(actual_groups)):
        wanted = expected_by_name.get(name)
        found_rows = actual_groups.get(name, [])
        if wanted is None:
            for found in found_rows:
                mismatches.append(f"unexpected distribution {name}=={found.get('version')}")
            continue
        if not found_rows:
            mismatches.append(f"missing distribution {name}=={wanted.get('version')}")
            continue
        if len(found_rows) > 1:
            mismatches.append(f"multiple installed distributions named {name}")
        found = found_rows[0]
        if wanted.get("version") != found.get("version"):
            mismatches.append(f"{name}: version expected {wanted.get('version')}, got {found.get('version')}")
        if wanted.get("editable") is True:
            # The code of an editable install lives in its project directory; its RECORD lists
            # only shim files the build backend regenerates. The source location identifies it.
            if not found.get("editable") or _source_url(wanted) != _source_url(found):
                mismatches.append(f"{name}: editable source expected {_source_url(wanted)}, "
                                  f"got {_source_url(found)}")
            continue
        # A reinstall rewrites pip's own RECORD rows; compare the payload when both sides have it.
        field = ("record_payload_sha256"
                 if wanted.get("record_payload_sha256") and found.get("record_payload_sha256")
                 else "record_sha256")
        if wanted.get(field) != found.get(field):
            mismatches.append(f"{name}: RECORD hash expected {wanted.get(field)}, got {found.get(field)}")
    return mismatches


def rehydrate_run(run_dir: str | os.PathLike[str], *, offline: bool = False,
                  dry_run: bool = False, printer=print) -> RehydrateResult:
    """Restore env/ without overwriting an existing environment; report exactness explicitly."""
    run = safe_expanduser(Path(run_dir)).resolve()
    if dry_run:
        return _rehydrate_plan(run, offline=offline, printer=printer)
    from .resume import ENGINE_LOCK, ResumeRefused, _lock_holder, hold_run_lock

    try:
        run_lock = hold_run_lock(run)
    except (OSError, ResumeRefused) as error:
        raise RehydrateError(f"run directory is in use or unavailable: {error}") from error
    try:
        holder = _lock_holder(run / ENGINE_LOCK)
        if holder is not None:
            raise RehydrateError(f"an engine is still running on this run directory (pid {holder})")
        return _rehydrate_locked(run, offline=offline, printer=printer)
    finally:
        run_lock.close()


def _rehydrate_plan(run: Path, *, offline: bool, printer) -> RehydrateResult:
    manifest = _load_manifest(run)
    env = run / "env"
    if env.exists() or env.is_symlink():
        raise RehydrateError(f"run environment already exists; refusing to overwrite: {env}")
    seed = _seed_path(manifest)
    use_seed = seed is not None and _seed_matches(seed, manifest)
    if use_seed:
        printer(f"would restore an exact environment from seed {seed}")
    else:
        printer(f"would rebuild with {manifest['source_interpreter_path']} and install "
                f"{len(manifest['distributions'])} distributions" + (" offline" if offline else ""))
    return RehydrateResult(exact=True if use_seed else None, used_seed=use_seed, dry_run=True)


def _rehydrate_locked(run: Path, *, offline: bool, printer) -> RehydrateResult:
    manifest = _load_manifest(run)
    seed = _seed_path(manifest)
    if seed is None:
        return _rehydrate_body(run, manifest, None, False, offline=offline, printer=printer)
    with _seed_lock(seed) as locked:
        if locked and seed.is_dir() and not seed.is_symlink() and _seed_matches(seed, manifest):
            return _rehydrate_body(run, manifest, seed, True, offline=offline, printer=printer)
    return _rehydrate_body(run, manifest, None, False, offline=offline, printer=printer)


def _rehydrate_body(run: Path, manifest: dict[str, object], seed: Path | None, use_seed: bool, *,
                    offline: bool, printer) -> RehydrateResult:
    env = run / "env"
    if env.exists() or env.is_symlink():
        raise RehydrateError(f"run environment already exists; refusing to overwrite: {env}")
    staging = run / f".env.rehydrate-{uuid.uuid4().hex}"
    published = False
    try:
        if use_seed:
            assert seed is not None
            try:
                _copy_seed(seed, staging, run, manifest)
                try:
                    os.utime(seed)
                except OSError:
                    pass  # A failed recency update must not undo a completed copy.
            except RehydrateError as error:
                printer(f"!!! environment seed could not be restored ({error}); "
                        "rebuilding from the recorded distributions instead !!!")
                shutil.rmtree(staging, ignore_errors=True)
                use_seed = False
        if not use_seed:
            _build_environment(staging, manifest, run_dir=run, offline=offline, printer=printer)
        os.replace(staging, env)
        published = True
        _relocate_staging_paths(staging, env)
        actual = storage_gc.distribution_inventory(env)
        mismatches = tuple(compare_distributions(manifest["distributions"], actual))
        if mismatches:
            printer("!!! REHYDRATED ENVIRONMENT IS INEXACT !!!")
            for mismatch in mismatches:
                printer(f"MISMATCH: {mismatch}")
        else:
            printer(f"rehydrated environment exactly from {'seed' if use_seed else 'recorded distributions'}")
        if not use_seed:
            try:
                (run / "env-seed.json").unlink(missing_ok=True)
            except OSError as error:
                printer(f"warning: stale environment seed reference was not removed: {error}")
        else:
            try:
                prior = json.loads((run / "env-seed.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prior = {}
            events = prior.get("events") if isinstance(prior, dict) else None
            reference: dict[str, object] = {"path": str(seed)}
            if isinstance(events, list):
                reference["events"] = events
            try:
                storage_gc._write_json_atomic(run / "env-seed.json", reference)
            except OSError as error:
                printer(f"warning: environment seed reference was not updated: {error}")
        return RehydrateResult(exact=not mismatches, used_seed=use_seed, dry_run=False,
                               mismatches=mismatches)
    except BaseException:
        if published and env.exists() and not env.is_symlink():
            try:
                os.replace(env, staging)
            except OSError:
                pass
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise
