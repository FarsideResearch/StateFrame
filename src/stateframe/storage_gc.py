"""Lossless run-storage retention, inventory and garbage collection."""
from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import json
import lzma
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .engine.kernel_safe_paths import safe_expanduser, safe_home
from .request_records import PACK_NAME as REQUEST_PACK_NAME
from .request_records import request_pack_size, seal_request_records


ARTIFACTS = ("env", "checkpoints", "measurement_journals", "blobs", "requests")
_DIST_NAME = re.compile(r"[-_.]+")
_BLOB_HEADER = struct.Struct(">4sI32s")
_BLOB_CRC = struct.Struct(">I")
_BLOB_PACK = "pack-0001.sfpk"
_BLOB_INDEX = "pack-0001.idx"
_BACKUP_NAME = "backups"


def default_runs_home() -> Path:
    """The same runs parent used by runner.default_run_dir(), including its no-home fallback."""
    user_home = safe_home()
    state_root = (user_home if user_home is not None else
                  Path(tempfile.gettempdir()) / f"stateframe-{getattr(os, 'getuid', lambda: 'unknown')()}")
    return state_root / ".stateframe" / "runs"


def stateframe_home() -> Path:
    configured = os.environ.get("STATEFRAME_HOME")
    if configured:
        return safe_expanduser(Path(configured))
    user_home = safe_home()
    if user_home is None:
        raise ValueError("cannot find the StateFrame home: set HOME or STATEFRAME_HOME")
    return safe_expanduser(user_home / ".stateframe")


def _normal_name(name: str) -> str:
    return _DIST_NAME.sub("-", name).strip("-").lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_config(path: Path) -> dict[str, str]:
    values = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key.strip().lower()] = value.strip()
    except OSError:
        pass
    return values


def _read_dist_metadata(path: Path) -> tuple[str, str]:
    name, version = path.name.removesuffix(".dist-info"), ""
    for metadata in (path / "METADATA", path / "PKG-INFO"):
        if not metadata.is_file() or metadata.is_symlink():
            continue
        for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Name: "):
                name = line[6:].strip()
            elif line.startswith("Version: "):
                version = line[9:].strip()
            elif not line:
                break
        break
    if not version and "-" in name:
        name, version = name.rsplit("-", 1)
    return _normal_name(name), version


def _probe_python_identity(interpreter: Path) -> tuple[str, str]:
    code = ("import json,sys; print(json.dumps({'version': '.'.join(map(str,sys.version_info[:3])), "
            "'implementation': sys.implementation.name}))")
    try:
        result = subprocess.run([str(interpreter), "-I", "-S", "-c", code], check=True,
                                capture_output=True, text=True, timeout=30)
        identity = json.loads(result.stdout)
        version = identity.get("version") if isinstance(identity, dict) else None
        implementation = identity.get("implementation") if isinstance(identity, dict) else None
        if not isinstance(version, str) or not isinstance(implementation, str):
            raise ValueError("interpreter identity output is invalid")
        return version, implementation
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        raise OSError(f"cannot verify run interpreter {interpreter}: {error}") from error


def _safe_direct_url(value: object) -> object:
    """Keep install provenance while removing URL credentials and secret query values."""
    if not isinstance(value, dict):
        return {"_unparsed": True}
    result = dict(value)
    raw_url = result.get("url")
    if not isinstance(raw_url, str):
        return result
    try:
        parts = urlsplit(raw_url)
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = parts.port
        netloc = f"{host}:{port}" if port is not None else host
        sensitive = ("token", "secret", "password", "passwd", "credential", "auth", "api_key",
                     "apikey", "access_key", "accesskey", "private_key", "signature", "sig", "sas",
                     "bearer", "authorization")
        query = [(key, "[redacted]" if any(word in key.lower() for word in sensitive) else item)
                 for key, item in parse_qsl(parts.query, keep_blank_values=True)]
        fragment = "[redacted]" if parts.fragment else ""
        result["url"] = urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), fragment))
    except ValueError:
        result["url"] = "[redacted invalid URL]"
    return result


# RECORD rows that pip writes differently on every install of the same wheel: bytecode it chose to
# compile, its own bookkeeping files, and console-script launchers that embed the env's path.
_INSTALLER_RECORD_FILES = {"INSTALLER", "REQUESTED", "RECORD", "direct_url.json"}


def record_payload_sha256(record: Path) -> str | None:
    """Digest of a RECORD's payload rows (path and hash), independent of installer artifacts."""
    try:
        rows = list(csv.reader(record.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeError, csv.Error):
        return None
    payload = set()
    for row in rows:
        if len(row) < 2 or not row[1]:
            continue
        path = row[0]
        parts = Path(path).parts
        if (path.endswith(".pyc") or "__pycache__" in parts or parts[:4] == ("..", "..", "..", "bin")
                or (len(parts) >= 2 and parts[-2].endswith(".dist-info")
                    and parts[-1] in _INSTALLER_RECORD_FILES)):
            continue
        payload.add((path, row[1]))
    encoded = "\n".join(f"{path},{digest}" for path, digest in sorted(payload)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def distribution_inventory(root: Path) -> list[dict[str, object]]:
    """Return installed dist-info identities and their raw RECORD hashes."""
    distributions = []
    for site in sorted(root.glob("lib/python*/site-packages")):
        if site.is_symlink() or not site.is_dir():
            continue
        for dist in sorted(site.glob("*.dist-info")):
            if dist.is_symlink() or not dist.is_dir():
                continue
            name, version = _read_dist_metadata(dist)
            record = dist / "RECORD"
            direct_url_path = dist / "direct_url.json"
            direct_url = None
            if direct_url_path.is_file() and not direct_url_path.is_symlink():
                raw = direct_url_path.read_bytes()
                try:
                    direct_url = _safe_direct_url(json.loads(raw))
                except (UnicodeError, ValueError):
                    direct_url = {"_unparsed": True}
            dir_info = direct_url.get("dir_info") if isinstance(direct_url, dict) else None
            distributions.append({
                "name": name,
                "version": version,
                "record_sha256": _sha256(record) if record.is_file() and not record.is_symlink() else None,
                "record_payload_sha256": (record_payload_sha256(record)
                                          if record.is_file() and not record.is_symlink() else None),
                "direct_url": direct_url,
                "editable": isinstance(dir_info, dict) and dir_info.get("editable") is True,
            })
    return sorted(distributions, key=lambda row: (str(row["name"]), str(row["version"])))


def _env_size(root: Path) -> tuple[int, int]:
    if root.is_symlink() or not root.is_dir():
        raise OSError(f"run environment is not a regular directory: {root}")
    total, count = 0, 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in names:
            path = Path(directory) / name
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                count += 1
                total += info.st_size
    return total, count


def _seed_key(run_dir: Path) -> str | None:
    path = run_dir / "env-seed.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = Path(data["path"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return seed.name or None


def build_env_manifest(run_dir: str | os.PathLike[str]) -> dict[str, object]:
    """Describe the run-local Python environment before it is discarded."""
    run = Path(run_dir)
    env = run / "env"
    if env.is_symlink() or not env.is_dir():
        raise OSError(f"run environment is not a regular directory: {env}")
    config = _parse_config(env / "pyvenv.cfg")
    spec = {}
    try:
        spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    run_env = spec.get("run_env") if isinstance(spec, dict) else None
    source_python = spec.get("run_env_source_python") if isinstance(spec, dict) else None
    if not source_python and isinstance(run_env, dict):
        source_python = run_env.get("python")
    python_version = config.get("version") or config.get("version_info")
    python_implementation = config.get("implementation")
    if not python_version or not python_implementation:
        interpreter = safe_expanduser(Path(str(source_python or (env / "bin" / "python"))))
        probed_version, probed_implementation = _probe_python_identity(interpreter)
        if python_version and python_version != probed_version:
            raise OSError("run interpreter version differs from env/pyvenv.cfg")
        if (python_implementation and python_implementation.casefold()
                != probed_implementation.casefold()):
            raise OSError("run interpreter implementation differs from env/pyvenv.cfg")
        python_version = python_version or probed_version
        python_implementation = python_implementation or probed_implementation
    byte_count, file_count = _env_size(env)
    return {
        "schema": "stateframe-env-manifest/1",
        "python_version": python_version,
        "python_implementation": python_implementation,
        "platform": sys.platform,
        "machine": platform.machine(),
        "platform_detail": platform.platform(),
        "interpreter_path": str(env / "bin" / ("python.exe" if os.name == "nt" else "python")),
        "source_interpreter_path": str(source_python or (env / "bin" / "python")),
        "seed_key": _seed_key(run),
        "system_site_packages": config.get("include-system-site-packages", "false").lower() == "true",
        "distributions": distribution_inventory(env),
        "byte_count": byte_count,
        "file_count": file_count,
    }


def _write_json_atomic(path: Path, value: object, *, mode: int | None = None) -> None:
    if mode is None:
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except FileNotFoundError:
            pass
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            if mode is not None:
                os.fchmod(stream.fileno(), mode)
            json.dump(value, stream, indent=1, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def remove_run_env(run_dir: str | os.PathLike[str], *, keep: bool = False,
                   require_finished: bool = True) -> int | None:
    """Write a recovery manifest, then drop env/. Return logical bytes removed."""
    run = Path(run_dir)
    if keep:
        return None
    if require_finished:
        from .resume import run_finished_for_good
        if not run_finished_for_good(run):
            return None
    env = run / "env"
    if not env.exists() and not env.is_symlink():
        return None
    manifest = build_env_manifest(run)
    _write_json_atomic(run / "env-manifest.json", manifest)
    size = int(manifest["byte_count"])
    remove_tree(env)
    return size


def _measurement_api():
    """Load the engine's established journal API lazily on the host."""
    import importlib

    engine_dir = str(Path(__file__).resolve().parent / "engine")
    inserted = engine_dir not in sys.path
    if inserted:
        sys.path.insert(0, engine_dir)
    try:
        return importlib.import_module("kernel_measurement")
    finally:
        if inserted:
            try:
                sys.path.remove(engine_dir)
            except ValueError:
                pass


def seal_measurements(run_dir: str | os.PathLike[str], *, printer=print) -> None:
    """Seal each completed measurement journal and pack its blob store, best effort."""
    run = Path(run_dir)
    worker = run / "worker"
    root = worker / "measurement-runs"
    if worker.is_symlink() or root.is_symlink() or not root.is_dir():
        return
    try:
        measurement = _measurement_api()
    except Exception as error:
        printer(f"warning: measurement storage could not load: {type(error).__name__}: {error}")
        return
    for directory in sorted(root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            result = measurement.seal_measurement_run(directory)
            if result.get("sealed") is False:
                printer(f"warning: measurement journal {directory.name} was not sealed: {result.get('reason')}")
        except Exception as error:
            printer(f"warning: measurement journal {directory.name} was not sealed: "
                    f"{type(error).__name__}: {error}")
        try:
            measurement.pack_blobs(directory)
        except Exception as error:
            printer(f"warning: measurement blobs for {directory.name} were not packed: "
                    f"{type(error).__name__}: {error}")


def seal_requests(run_dir: str | os.PathLike[str], *, printer=print) -> None:
    try:
        seal_request_records(run_dir)
    except Exception as error:
        printer(f"warning: preflight requests were not sealed: {type(error).__name__}: {error}")


def remove_checkpoints(run_dir: str | os.PathLike[str]) -> int:
    root = Path(run_dir) / "checkpoints"
    if not root.is_dir() or root.is_symlink():
        return 0
    size = _logical_tree_bytes(root)
    remove_tree(root)
    return size


def _logical_tree_bytes(root: Path) -> int:
    total = 0
    if root.is_file() and not root.is_symlink():
        return root.stat().st_size
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in names:
            path = Path(directory) / name
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
    return total


def _allocated_bytes(path: Path) -> int:
    try:
        info = path.stat()
        return getattr(info, "st_blocks", 0) * 512 or info.st_size
    except OSError:
        return 0


def _block_size(path: Path) -> int:
    try:
        size = os.statvfs(path).f_frsize
    except (OSError, AttributeError):
        size = 4096
    return max(512, int(size or 4096))


def _rounded_allocation(size: int, block: int) -> int:
    return ((size + block - 1) // block) * block if size else 0


def _compressed_size(path: Path) -> int:
    compressor = lzma.LZMACompressor(preset=6)
    total = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            total += len(compressor.compress(chunk))
    return total + len(compressor.flush())


def _measurement_inventory(run_dir: Path) -> tuple[int, int]:
    worker = run_dir / "worker"
    root = worker / "measurement-runs"
    journal_reclaim, blob_reclaim = 0, 0
    if worker.is_symlink() or root.is_symlink() or not root.is_dir():
        return journal_reclaim, blob_reclaim
    for measurement in root.iterdir():
        if measurement.is_symlink() or not measurement.is_dir():
            continue
        plain = measurement / "events.jsonl"
        if plain.is_file() and not plain.is_symlink():
            try:
                journal_reclaim += max(0, _allocated_bytes(plain) - _rounded_allocation(
                    _compressed_size(plain), _block_size(plain)))
            except OSError:
                pass
        blobs = measurement / "blobs"
        if blobs.is_symlink() or not blobs.is_dir():
            continue
        loose = [path for path in blobs.glob("*.zlib")
                 if path.is_file() and not path.is_symlink()
                 and re.fullmatch(r"[0-9a-f]{64}", path.stem)]
        if not loose:
            continue
        pack = blobs / _BLOB_PACK
        index = blobs / _BLOB_INDEX
        existing_pack_size = pack.stat().st_size if pack.is_file() and not pack.is_symlink() else 0
        existing_records: set[str] = set()
        index_data = {}
        if index.is_file() and not index.is_symlink():
            try:
                candidate = json.loads(index.read_text(encoding="utf-8"))
                candidate_records = candidate.get("records", {}) if isinstance(candidate, dict) else None
                if isinstance(candidate_records, dict):
                    index_data = candidate
                    existing_records = set(candidate_records)
            except (OSError, ValueError, TypeError):
                pass
        append_size = 0
        records = dict(index_data.get("records", {})) if isinstance(index_data, dict) else {}
        offset = existing_pack_size
        for path in loose:
            if path.stem in existing_records:
                continue
            payload_size = path.stat().st_size
            append_size += _BLOB_HEADER.size + payload_size + _BLOB_CRC.size
            records[path.stem] = [offset, payload_size]
            offset += _BLOB_HEADER.size + payload_size + _BLOB_CRC.size
        projected_index = json.dumps({"pack_size": offset, "records": records}, sort_keys=True).encode()
        block = _block_size(blobs)
        current_alloc = sum(_allocated_bytes(path) for path in loose)
        current_alloc += _allocated_bytes(index) if index.is_file() and not index.is_symlink() else 0
        current_alloc += _allocated_bytes(pack) if pack.is_file() and not pack.is_symlink() else 0
        projected_alloc = _rounded_allocation(existing_pack_size + append_size, block)
        projected_alloc += _rounded_allocation(len(projected_index), block)
        blob_reclaim += max(0, current_alloc - projected_alloc)
    return journal_reclaim, blob_reclaim


def _artifact_reclaimable(run_dir: Path) -> dict[str, int]:
    result = {name: 0 for name in ARTIFACTS}
    env = run_dir / "env"
    if env.is_dir() and not env.is_symlink():
        result["env"] = _logical_tree_bytes(env)
    checkpoints = run_dir / "checkpoints"
    if checkpoints.is_dir() and not checkpoints.is_symlink():
        result["checkpoints"] = _logical_tree_bytes(checkpoints)
    result["measurement_journals"], result["blobs"] = _measurement_inventory(run_dir)
    requests_dir = run_dir / "requests"
    if requests_dir.is_dir() and not requests_dir.is_symlink():
        loose_bytes = _logical_tree_bytes(requests_dir)
        try:
            packed_size = request_pack_size(run_dir)
            block = _block_size(requests_dir)
            allocated_loose = sum(_allocated_bytes(path) for path in requests_dir.glob("*.json")
                                  if path.is_file() and not path.is_symlink())
            pack = run_dir / REQUEST_PACK_NAME
            allocated_current_pack = _allocated_bytes(pack) if pack.is_file() and not pack.is_symlink() else 0
            result["requests"] = max(
                0, allocated_loose + allocated_current_pack - _rounded_allocation(packed_size, block))
        except (OSError, ValueError, tarfile.TarError):
            result["requests"] = loose_bytes
    return result


def _run_finished(run_dir: Path) -> bool:
    from .resume import run_finished_for_good
    return run_finished_for_good(run_dir)


def _run_mtime(run_dir: Path) -> float:
    for path in (run_dir / "summary.json", run_dir / "engine-result.json"):
        try:
            return path.stat().st_mtime
        except OSError:
            pass
    try:
        return run_dir.stat().st_mtime
    except OSError:
        return 0.0


def _discover_runs(root: Path) -> list[Path]:
    if _BACKUP_NAME in root.parts or not root.exists() or root.is_symlink():
        return []
    if root.is_file():
        return []
    if (root / "spec.json").is_file():
        try:
            spec = json.loads((root / "spec.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [root.resolve()] if isinstance(spec, dict) and spec.get("schema") == "stateframe-run/1" else []
    found = []
    prune_names = {"backups", "env", "workspace", "parent", "worker", "checkpoints", "attempts"}
    for directory, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        if _BACKUP_NAME in current.parts:
            dirs[:] = []
            continue
        if "spec.json" in files:
            try:
                spec = json.loads((current / "spec.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                spec = None
            if isinstance(spec, dict) and spec.get("schema") == "stateframe-run/1":
                found.append(current.resolve())
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if name not in prune_names
                   and not (current / name).is_symlink()]
    return sorted(set(found))


def _check_live_locks(run_dir: Path) -> str | None:
    from .resume import ENGINE_LOCK, RUN_LOCK

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    for name, label in ((ENGINE_LOCK, "engine"), (RUN_LOCK, "run")):
        path = run_dir / name
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            continue
        except OSError as error:
            return f"invalid {label} lock ({error.strerror or type(error).__name__})"
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return f"invalid {label} lock"
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.lseek(descriptor, 0, os.SEEK_SET)
                holder = os.read(descriptor, 128).decode("ascii", errors="replace").strip() or "unknown"
                return f"live {label} lock (pid {holder})"
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return None


@contextlib.contextmanager
def _acquire_gc_lock(run_dir: Path):
    """Acquire the run lock and verify no engine owns its lock before mutating artifacts."""
    from .resume import ENGINE_LOCK, RUN_LOCK

    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(run_dir / RUN_LOCK, flags, 0o600)
    except OSError as error:
        yield f"invalid run lock ({error.strerror or type(error).__name__})"
        return
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            yield "invalid run lock"
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield "live run lock"
            return
        locked = True
        engine_path = run_dir / ENGINE_LOCK
        try:
            engine_fd = os.open(engine_path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                                | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            engine_fd = None
        except OSError as error:
            yield f"invalid engine lock ({error.strerror or type(error).__name__})"
            return
        if engine_fd is not None:
            try:
                if not stat.S_ISREG(os.fstat(engine_fd).st_mode):
                    yield "invalid engine lock"
                    return
                try:
                    fcntl.flock(engine_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    yield "live engine lock"
                    return
                else:
                    fcntl.flock(engine_fd, fcntl.LOCK_UN)
            finally:
                os.close(engine_fd)
        yield None
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _seed_reference(path: Path, seed_store: Path) -> Path | None:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        raw = metadata.get("path")
        if not isinstance(raw, str) or not raw:
            return None
        seed = safe_expanduser(Path(raw))
        if not seed.is_absolute():
            seed = seed_store / seed
        return seed.resolve()
    except (OSError, ValueError, AttributeError):
        return None


def _seed_references(runs: list[Path], removed_runs: set[Path], seed_store: Path) -> set[Path]:
    references = set()
    for run in runs:
        if run in removed_runs:
            continue
        env = run / "env"
        record = run / "env-seed.json"
        if env.is_dir() and not env.is_symlink() and record.is_file():
            seed = _seed_reference(record, seed_store)
            if seed is not None:
                references.add(seed)
        checkpoint_root = run / "checkpoints"
        if checkpoint_root.is_dir() and not checkpoint_root.is_symlink():
            for record in checkpoint_root.rglob("env-seed.json"):
                if record.is_file() and not record.is_symlink():
                    seed = _seed_reference(record, seed_store)
                    if seed is not None:
                        references.add(seed)
    return references


def remove_tree(root: Path) -> None:
    """rmtree that also clears read-only directories (frozen candidate workspaces, read-only packages).

    Only directories are made writable: files may be hard-linked into a shared environment seed, and
    unlinking a file needs write permission on its directory, never on the file itself."""
    try:
        shutil.rmtree(root)
    except PermissionError:
        if root.is_symlink():
            raise
        root.chmod(stat.S_IMODE(root.lstat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
        for directory, dirs, _ in os.walk(root, topdown=True, followlinks=False):
            for name in dirs:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
        shutil.rmtree(root)


def _make_tree_writable(root: Path) -> None:
    if root.is_symlink():
        return
    for directory, dirs, names in os.walk(root, topdown=False, followlinks=False):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR)
            except OSError:
                pass
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                path.chmod(stat.S_IMODE(path.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
            except OSError:
                pass
    try:
        root.chmod(stat.S_IMODE(root.stat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
    except OSError:
        pass


def _seed_has_external_hardlinks(seed: Path) -> bool:
    """Keep a seed when read-only files still share inodes with another environment."""
    links: dict[tuple[int, int], tuple[int, int]] = {}

    def fail(error: OSError) -> None:
        raise error

    try:
        for directory, _, names in os.walk(seed, topdown=True, followlinks=False, onerror=fail):
            for name in names:
                path = Path(directory) / name
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                identity = (info.st_dev, info.st_ino)
                count, internal = links.get(identity, (info.st_nlink, 0))
                links[identity] = (max(count, info.st_nlink), internal + 1)
    except OSError:
        return True
    return any(count > internal for count, internal in links.values())


def _seed_storage_bytes(seed: Path, *, cache: bool = True) -> int:
    from .engine.kernel_env_seed import _SEED_KEY, _seed_bytes

    return _seed_bytes(seed, cache=cache) if _SEED_KEY.fullmatch(seed.name) else _logical_tree_bytes(seed)


def _remove_seed(seed: Path) -> int:
    if seed.is_symlink() or not seed.is_dir():
        return 0
    if _seed_has_external_hardlinks(seed):
        raise OSError("environment seed still has files hard-linked into another environment")
    size = _seed_storage_bytes(seed)
    _make_tree_writable(seed)
    shutil.rmtree(seed)
    from .engine.kernel_env_seed import _SEED_KEY, _fsync_directory

    if _SEED_KEY.fullmatch(seed.name):
        (seed.parent / f".{seed.name}.bytes").unlink(missing_ok=True)
        _fsync_directory(seed.parent)
    return size


def run_gc(roots: list[str | os.PathLike[str]] | None = None, *, apply: bool = False,
           older_than_days: float = 14, keep_last: int = 0, include_unfinished: bool = False,
           home: Path | None = None, now: float | None = None, prune_seeds: bool | None = None,
           printer=print) -> dict[str, object]:
    """Plan or apply safe, lossless retention for runs found under the requested roots.

    Environment seeds are pruned when gc scans the default runs home, or when ``prune_seeds`` is
    true; only seeds unused for ``older_than_days`` and unreferenced by the scanned runs qualify.
    """
    if older_than_days < 0:
        raise ValueError("--older-than must be zero or more days")
    if keep_last < 0:
        raise ValueError("--keep-last must be zero or more")
    state_home = safe_expanduser(home) if home is not None else stateframe_home()
    selected_roots = [safe_expanduser(Path(root)).resolve() for root in roots] if roots else [default_runs_home()]
    run_roots: dict[Path, Path] = {}
    for root in selected_roots:
        for run in _discover_runs(root):
            run_roots.setdefault(run, root)
    all_runs = sorted(run_roots)
    keep_newest: set[Path] = set()
    if keep_last:
        grouped: dict[Path, list[Path]] = {}
        for run, owner in run_roots.items():
            grouped.setdefault(owner, []).append(run)
        for candidates in grouped.values():
            keep_newest.update(sorted(candidates, key=_run_mtime, reverse=True)[:keep_last])

    timestamp = time.time() if now is None else now
    cutoff = timestamp - older_than_days * 86400
    rows: list[dict[str, object]] = []
    eligible: set[Path] = set()
    for run in all_runs:
        finished = _run_finished(run)
        lock_reason = _check_live_locks(run)
        if lock_reason:
            status = "locked"
        elif run in keep_newest:
            status = "kept_last"
        elif _run_mtime(run) > cutoff:
            status = "too_new"
        elif not finished and not include_unfinished:
            status = "unfinished"
        else:
            status = "eligible"
            eligible.add(run)
        if status == "eligible" and not finished and include_unfinished:
            printer(f"WARNING: resume will be lost for unfinished run {run} if GC is applied")
        reclaimable = _artifact_reclaimable(run) if status == "eligible" else {key: 0 for key in ARTIFACTS}
        rows.append({"run": str(run), "status": status, "finished_for_good": finished,
                     "reclaimable": reclaimable, "total": sum(reclaimable.values())})

    if apply:
        for row in rows:
            run = Path(str(row["run"]))
            if row["status"] != "eligible":
                continue
            with _acquire_gc_lock(run) as lock_reason:
                if lock_reason:
                    row["status"] = "locked"
                    row["reclaimable"] = {key: 0 for key in ARTIFACTS}
                    row["total"] = 0
                    continue
                finished = _run_finished(run)
                if not finished and not include_unfinished:
                    row["status"] = "unfinished"
                    row["reclaimable"] = {key: 0 for key in ARTIFACTS}
                    row["total"] = 0
                    continue
                if not finished and row.get("finished_for_good") is True:
                    printer(f"WARNING: resume will be lost for unfinished run {run}")
                try:
                    remove_checkpoints(run)
                except OSError as error:
                    printer(f"warning: checkpoints were not removed from {run}: {error}")
                try:
                    remove_run_env(run, require_finished=False)
                except (OSError, ValueError) as error:
                    printer(f"warning: run environment was not removed from {run}: {error}")
                seal_measurements(run, printer=printer)
                seal_requests(run, printer=printer)
                row["applied"] = True

    projected_release_runs = set() if apply else eligible
    seed_store = state_home / "env-seeds"
    # Seeds are shared by every run directory, and runs made with --run-dir can live anywhere. A gc
    # scoped to explicit roots cannot see all references, so it prunes seeds only when asked to.
    if prune_seeds is None:
        prune_seeds = not roots
        if roots:
            printer("age-based environment seed pruning was skipped: gc of explicit roots cannot see every "
                    "run that shares them; add --prune-seeds to include it")
    seed_references = _seed_references(all_runs, projected_release_runs, seed_store)
    seed_candidates = []
    if seed_store.is_dir() and not seed_store.is_symlink():
        for seed in sorted(seed_store.iterdir()):
            if seed.is_symlink() or not seed.is_dir():
                continue
            quarantine = seed.name.startswith("quarantine-")
            # A seed's mtime is its last use: every clone touches it. Unused for the retention
            # window and unreferenced by the scanned runs is the only way a live seed goes.
            old = seed.stat().st_mtime <= cutoff
            if (prune_seeds and old
                    and (quarantine or seed.resolve() not in seed_references)
                    and not _seed_has_external_hardlinks(seed)):
                seed_candidates.append(seed)
    seed_bytes = sum(_seed_storage_bytes(path, cache=apply) for path in seed_candidates)
    if apply:
        removed_seed_bytes = 0
        removed_seed_count = 0
        for seed in seed_candidates:
            descriptor = None
            try:
                descriptor = os.open(seed_store / f".{seed.name}.lock",
                                     os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
                                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Recheck references after waiting for the per-seed builder/clone lock.
                current_refs = _seed_references(all_runs, set(), seed_store)
                if seed.resolve() in current_refs and not seed.name.startswith("quarantine-"):
                    continue
                if _seed_has_external_hardlinks(seed):
                    printer(f"warning: environment seed is hard-linked into another environment and was retained: "
                            f"{seed.name}")
                    continue
                removed_seed_bytes += _remove_seed(seed)
                removed_seed_count += not seed.exists()
            except BlockingIOError:
                printer(f"warning: environment seed is locked and was retained: {seed.name}")
            except OSError as error:
                printer(f"warning: environment seed was retained ({seed.name}): {error}")
            finally:
                if descriptor is not None:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
                    os.close(descriptor)
        seed_bytes = removed_seed_bytes
        seed_count = removed_seed_count
    else:
        seed_count = len(seed_candidates)

    from .engine.kernel_env_seed import _cap_bytes, evict_to_cap

    cap = {"count": 0, "bytes": 0, "paths": []}
    if seed_store.is_dir() and not seed_store.is_symlink():
        def cap_event(event: dict[str, object]) -> None:
            if event["event"] == "env_seed_cap_invalid":
                printer("warning: invalid environment seed cap; using the 10 GB default")
            elif event["event"] == "env_seed_evict_failed":
                printer(f"warning: environment seed cap eviction failed: {event['error']}")

        cap = evict_to_cap(seed_store, _cap_bytes(cap_event), "", cap_event,
                           dry_run=not apply,
                           excluded=frozenset(seed_candidates) if not apply else frozenset())

    totals = {key: sum(int(row["reclaimable"][key]) for row in rows) for key in ARTIFACTS}
    totals["total"] = sum(totals.values())
    result = {"roots": [str(root) for root in selected_roots], "runs": rows, "totals": totals,
              "seeds": {"count": seed_count, "bytes": seed_bytes,
                        "paths": [str(path) for path in seed_candidates]},
              "seed_cap": cap}
    if not apply:
        _print_gc_table(result, printer)
    else:
        _print_gc_table(result, printer, applied=True)
    return result


def _print_gc_table(result: dict[str, object], printer, *, applied: bool = False) -> None:
    runs = result["runs"]
    printer("run | status | env | checkpoints | measurement journals | blobs | requests | total")
    for row in runs:
        values = row["reclaimable"]
        printer(f"{row['run']} | {row['status']} | {values['env']} | {values['checkpoints']} | "
                f"{values['measurement_journals']} | {values['blobs']} | {values['requests']} | {row['total']}")
    totals = result["totals"]
    printer(f"TOTAL | {('applied' if applied else 'dry run')} | {totals['env']} | {totals['checkpoints']} | "
            f"{totals['measurement_journals']} | {totals['blobs']} | {totals['requests']} | {totals['total']}")
    seeds = result["seeds"]
    printer(f"environment seeds | {'removed' if applied else 'unreferenced'} | count={seeds['count']} "
            f"bytes={seeds['bytes']}")
    cap = result["seed_cap"]
    printer(f"environment seed cap | {'removed' if applied else 'dry run'} | count={cap['count']} "
            f"bytes={cap['bytes']}")
