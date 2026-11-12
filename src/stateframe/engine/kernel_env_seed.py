"""Immutable source-environment seeds and private run-environment clones."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path
from typing import Callable

if __package__:
    from .kernel_pid_liveness import pid_alive
    from .kernel_safe_paths import safe_home
else:
    from kernel_pid_liveness import pid_alive
    from kernel_safe_paths import safe_home


FICLONE = 0x40049409
DEFAULT_MAX_GB = 10
_DECIMAL_GB = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)\Z")
_SEED_KEY = re.compile(r"[0-9a-f]{64}\Z")
_TEMPORARY = re.compile(r"\.[0-9a-f]{64}\.tmp-([0-9]+)-[0-9]+\Z")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(root: Path) -> str:
    """Framed digest of the source environment: every path with its type, mode and content or link."""
    if __package__:
        from .kernel_tree_digest import tree_digest
    else:
        from kernel_tree_digest import tree_digest
    entries = []
    for directory, dirs, names in os.walk(root):
        folder = Path(directory)
        dirs.sort()
        for name in [*dirs, *sorted(names)]:
            path = folder / name
            info = path.lstat()
            relative = path.relative_to(root).as_posix()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                entries.append(["symlink", relative, os.readlink(path), mode])
            elif stat.S_ISREG(info.st_mode):
                entries.append(["file", relative, _digest(path), mode])
            elif stat.S_ISDIR(info.st_mode):
                entries.append(["dir", relative, mode])
            else:
                raise OSError(f"unsupported environment file: {relative}")
    return tree_digest("stateframe.env_seed_source/1", entries)


def _tree_fingerprint(root: Path) -> list:
    """Cheap stat-only fingerprint: entry count, total size, newest mtime and ctime (no reads)."""
    count = total = newest_m = newest_c = 0
    for directory, dirs, names in os.walk(root):
        for name in [*dirs, *names]:
            info = (Path(directory) / name).lstat()
            count += 1
            total += info.st_size
            newest_m = max(newest_m, info.st_mtime_ns)
            newest_c = max(newest_c, info.st_ctime_ns)
        info = os.lstat(directory)
        newest_m = max(newest_m, info.st_mtime_ns)
        newest_c = max(newest_c, info.st_ctime_ns)
    return [count, total, newest_m, newest_c]


_DIGEST_CACHE_ENTRIES = 16


def _cached_source_digest(root: Path, cache_file: Path | None) -> str:
    """_source_digest memoized on disk, keyed by (root, file count, size, newest mtime/ctime)."""
    if cache_file is None:
        return _source_digest(root)
    key = str(root.resolve())
    fingerprint = _tree_fingerprint(root)
    try:
        entries = json.loads(cache_file.read_text())
        hit = entries[key]
        if hit["fingerprint"] == fingerprint and re.fullmatch(r"[0-9a-f]{64}", hit["digest"]):
            return hit["digest"]
    except (OSError, ValueError, KeyError, TypeError):
        entries = {}
    if not isinstance(entries, dict):
        entries = {}
    digest = _source_digest(root)
    # A file edited mid-walk must not be cached under the earlier fingerprint.
    if _tree_fingerprint(root) == fingerprint:
        entries.pop(key, None)
        entries[key] = {"fingerprint": fingerprint, "digest": digest}
        entries = dict(list(entries.items())[-_DIGEST_CACHE_ENTRIES:])
        temporary = cache_file.with_name(f".{cache_file.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            cache_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            temporary.write_text(json.dumps(entries, sort_keys=True))
            os.replace(temporary, cache_file)
        except OSError:
            temporary.unlink(missing_ok=True)  # Caching is best effort.
    return digest


def _key_inputs(source: Path, interpreter: Path, source_repo: Path | None,
                authorized_roots: tuple[Path, ...], digest_cache: Path | None = None) -> dict:
    config = {}
    for line in (source / "pyvenv.cfg").read_text().splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            config[name.strip().lower()] = value.strip()
    distributions = []
    for info in sorted(source.glob("lib/python*/site-packages/*.dist-info")):
        metadata = info / "METADATA"
        if metadata.is_symlink() or (info / "RECORD").is_symlink():
            raise OSError(f"symlinked distribution metadata: {info.name}")
        name, version = info.name, ""
        if metadata.is_file():
            for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("Name: "):
                    name = line[6:].strip()
                elif line.startswith("Version: "):
                    version = line[9:].strip()
                elif not line:
                    break
        record = info / "RECORD"
        distributions.append((name.lower(), version, _digest(record) if record.is_file() else None))
    return {
        "python": [config.get("implementation", sys.implementation.name),
                   config.get("version", ".".join(map(str, sys.version_info[:3])))],
        "platform": sys.platform,
        "interpreter": str(interpreter.resolve(strict=True)),
        "distributions": sorted(distributions),
        "relocation": {"source": str(source.resolve()),
                       "source_repo": str(source_repo) if source_repo is not None else None,
                       "authorized_roots": sorted(str(path) for path in authorized_roots),
                       "source_sha256": _cached_source_digest(source, digest_cache)},
    }


def _inventory(root: Path) -> tuple[dict[str, dict], list[str]]:
    files = {}
    directories = []
    for directory, dirs, names in os.walk(root):
        if Path(directory) != root:
            directories.append(Path(directory).relative_to(root).as_posix())
        linked_dirs = [name for name in dirs if (Path(directory) / name).is_symlink()]
        for name in names + linked_dirs:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                files[relative] = {"link": os.readlink(path)}
            elif stat.S_ISREG(info.st_mode):
                files[relative] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns,
                                   "sha256": _digest(path), "mode": stat.S_IMODE(info.st_mode)}
            else:
                raise OSError(f"unsupported environment file: {relative}")
    return files, directories


def _valid(seed: Path, key: str) -> bool:
    try:
        manifest = json.loads((seed / "seed.json").read_text())
        if manifest.get("key") != key or not isinstance(manifest.get("files"), dict):
            return False
        files = manifest["files"]
        actual_names = {path.relative_to(seed).as_posix() for path in seed.rglob("*")
                        if (path.is_symlink() or not path.is_dir()) and path.name != "seed.json"}
        if actual_names != set(files):
            return False
        if sorted(manifest.get("directories", ())) != sorted(
                path.relative_to(seed).as_posix() for path in seed.rglob("*")
                if path.is_dir() and not path.is_symlink()):
            return False
        for relative, expected in files.items():
            path = seed / relative
            info = path.lstat()
            if "link" in expected:
                if not stat.S_ISLNK(info.st_mode) or os.readlink(path) != expected["link"]:
                    return False
            elif (not stat.S_ISREG(info.st_mode)
                  or stat.S_IMODE(info.st_mode) != (expected["mode"] & ~0o222)
                  or ((info.st_size != expected["size"] or info.st_mtime_ns != expected["mtime_ns"])
                      and _digest(path) != expected["sha256"])):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _fsync_tree(root: Path) -> None:
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _reflink(source: Path, target: Path) -> bool:
    if sys.platform != "linux":
        return False
    try:
        with source.open("rb") as old, target.open("xb") as new:
            fcntl.ioctl(new.fileno(), FICLONE, old.fileno())
        shutil.copystat(source, target)
        return True
    except OSError:
        target.unlink(missing_ok=True)
        return False


def _private(relative: str) -> bool:
    path = Path(relative)
    return (path.parts[0] != "lib" or "cache" in path.parts
            or path.suffix in {".pth", ".egg-link"}
            or path.name.startswith("__editable___")
            or "__pycache__" in path.parts)


def _clone_files(seed: Path, destination: Path, files: dict[str, dict],
                 directories: list[str]) -> None:
    if sys.platform == "darwin":
        result = subprocess.run(["cp", "-c", "-R", str(seed), str(destination)],
                                capture_output=True, check=False)
        if result.returncode == 0:
            (destination / "seed.json").unlink()
            for relative, row in files.items():
                if "mode" in row:
                    (destination / relative).chmod(row["mode"])
            return
        if destination.exists():
            shutil.rmtree(destination)
    destination.mkdir()
    for relative in directories:
        (destination / relative).mkdir(parents=True, exist_ok=True)
    for relative, row in files.items():
        source, target = seed / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if "link" in row:
            target.symlink_to(row["link"])
        elif _reflink(source, target):
            target.chmod(row["mode"])
        elif _private(relative) or not _link(source, target):
            shutil.copy2(source, target)
            target.chmod(row["mode"])


def _link(source: Path, target: Path) -> bool:
    # A seed store on another filesystem (EXDEV) or one refusing links falls back to a private copy.
    try:
        os.link(source, target)
        return True
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EMLINK, errno.ENOTSUP):
            raise
        return False


def _tree_bytes(root: Path) -> int:
    # Keep the logical-size semantics of storage_gc._logical_tree_bytes in the engine child.
    total = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in names:
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def _write_seed_bytes(seed: Path, size: int) -> None:
    sidecar = seed.parent / f".{seed.name}.bytes"
    temporary = seed.parent / f".{seed.name}.bytes.tmp-{os.getpid()}-{time.time_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        with os.fdopen(os.open(temporary, flags, 0o600), "wb") as stream:
            stream.write(str(size).encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, sidecar)
        _fsync_directory(seed.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _seed_bytes(seed: Path, *, cache: bool = True) -> int:
    sidecar = seed.parent / f".{seed.name}.bytes"
    flags = (os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        with os.fdopen(os.open(sidecar, flags), "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("invalid seed size sidecar")
            raw = stream.read(65)
        if not raw or len(raw) > 64 or not raw.isdigit():
            raise ValueError("invalid seed size sidecar")
        return int(raw)
    except (OSError, ValueError):
        pass

    try:
        manifest = seed / "seed.json"
        if manifest.is_symlink():
            raise ValueError("symlinked seed manifest")
        files = json.loads(manifest.read_text())["files"]
        if not isinstance(files, dict):
            raise ValueError("invalid seed files")
        if any(not isinstance(row, dict) or ("size" not in row and not isinstance(row.get("link"), str))
               for row in files.values()):
            raise ValueError("invalid seed file")
        sizes = [row["size"] for row in files.values() if "size" in row]
        if any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in sizes):
            raise ValueError("invalid seed size")
        size = sum(sizes)
    except (OSError, ValueError, KeyError, TypeError):
        size = _tree_bytes(seed)
    if cache and seed.is_dir() and not seed.is_symlink():
        try:
            _write_seed_bytes(seed, size)
        except OSError:
            pass  # Cache repair must not prevent eviction or a clone.
    return size


def _external_hardlinks(seed: Path) -> bool:
    # Mirror storage_gc._seed_has_external_hardlinks; the engine cannot import product GC.
    links: dict[tuple[int, int], tuple[int, int]] = {}

    def fail(error: OSError) -> None:
        raise error

    try:
        for directory, _, names in os.walk(seed, followlinks=False, onerror=fail):
            for name in names:
                info = (Path(directory) / name).lstat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                identity = (info.st_dev, info.st_ino)
                count, internal = links.get(identity, (info.st_nlink, 0))
                links[identity] = (max(count, info.st_nlink), internal + 1)
    except OSError:
        return True
    return any(count > internal for count, internal in links.values())


def _make_tree_writable(root: Path) -> None:
    # Mirror storage_gc._make_tree_writable without changing hard-linked file modes.
    for directory, dirs, _ in os.walk(root, topdown=False, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)
    root.chmod(stat.S_IMODE(root.lstat().st_mode) | stat.S_IWUSR | stat.S_IXUSR)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_eviction_target(store: Path, target: Path) -> None:
    _make_tree_writable(target)
    shutil.rmtree(target)
    _fsync_directory(store)


def _evict_to_cap(store: Path, max_bytes: int, keep: str,
                  event: Callable[[dict], None] | None, result: dict[str, object], *, dry_run: bool = False,
                  excluded: frozenset[Path] = frozenset()) -> dict[str, object]:
    removed: list[str] = []
    removed_bytes = 0
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    lock = (nullcontext() if dry_run else
            os.fdopen(os.open(store / ".evict.lock", flags, 0o600), "a+b"))
    with lock as eviction_lock:
        if eviction_lock is not None:
            fcntl.flock(eviction_lock, fcntl.LOCK_EX)
        if not dry_run:
            for path in store.iterdir():
                try:
                    if path.name.startswith(".evicting-") and path.is_dir() and not path.is_symlink():
                        _remove_eviction_target(store, path)
                except FileNotFoundError:
                    continue

        candidates: list[tuple[int, int, Path, str, str, int]] = []
        total = 0
        for path in store.iterdir():
            try:
                if path in excluded or path.is_symlink() or not path.is_dir():
                    continue
                name = path.name
                if name.startswith("quarantine-"):
                    priority, reason, size = 0, "quarantine", _tree_bytes(path)
                elif _TEMPORARY.fullmatch(name):
                    priority, reason, size = 1, "orphan_tmp", _tree_bytes(path)
                elif _SEED_KEY.fullmatch(name):
                    priority, reason, size = 2, "cap", _seed_bytes(path, cache=not dry_run)
                else:
                    continue
                mtime = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            total += size
            candidates.append((priority, mtime, path, reason, name, size))

        for priority, listed_mtime, path, reason, name, listed_size in sorted(candidates):
            if total <= max_bytes:
                break
            if priority == 2 and name == keep:
                continue
            if priority == 1:
                pid = int(_TEMPORARY.fullmatch(name).group(1))
                try:
                    if pid_alive(pid):  # a Linux zombie owner counts as dead
                        continue
                except OSError:
                    continue
            try:
                if priority == 2:
                    try:
                        descriptor = os.open(store / f".{name}.lock",
                                             flags & ~os.O_CREAT if dry_run else flags, 0o600)
                    except FileNotFoundError:
                        if not dry_run:
                            raise
                        descriptor = None
                    try:
                        if descriptor is not None:
                            try:
                                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                continue
                        if (not path.is_dir() or path.is_symlink()
                                or path.stat().st_mtime_ns > listed_mtime
                                or _external_hardlinks(path)):
                            continue
                        size = _seed_bytes(path, cache=not dry_run)
                        if not dry_run:
                            _make_tree_writable(path)
                            target = store / f".evicting-{name}-{time.time_ns()}"
                            path.rename(target)
                            _remove_eviction_target(store, target)
                            (store / f".{name}.bytes").unlink(missing_ok=True)
                            _fsync_directory(store)
                    finally:
                        if descriptor is not None:
                            os.close(descriptor)
                else:
                    if (not path.is_dir() or path.is_symlink()
                            or path.stat().st_mtime_ns > listed_mtime
                            or _external_hardlinks(path)):
                        continue
                    size = _tree_bytes(path)
                    if not dry_run:
                        target = store / f".evicting-{name}-{time.time_ns()}"
                        path.rename(target)
                        _remove_eviction_target(store, target)
            except FileNotFoundError:
                continue
            total -= listed_size
            removed.append(str(path))
            removed_bytes += size
            result.update(count=len(removed), bytes=removed_bytes, paths=removed.copy())
            if event is not None and not dry_run:
                key = name
                if priority == 1:
                    key = name[1:65]
                elif priority == 0 and _SEED_KEY.fullmatch(name.rsplit("-", 1)[-1]):
                    key = name.rsplit("-", 1)[-1]
                event({"event": "env_seed_evicted", "key": key, "bytes": size, "reason": reason})
    return result


def evict_to_cap(store: Path, max_bytes: int, keep: str,
                 event: Callable[[dict], None] | None, *, dry_run: bool = False,
                 excluded: frozenset[Path] = frozenset()) -> dict[str, object]:
    """Best-effort serialized eviction after the clone's per-seed lock is released."""
    result: dict[str, object] = {"count": 0, "bytes": 0, "paths": []}
    if max_bytes == 0:
        return result
    try:
        return _evict_to_cap(store, max_bytes, keep, event, result,
                             dry_run=dry_run, excluded=excluded)
    except Exception as error:
        if event is not None:
            try:
                event({"event": "env_seed_evict_failed",
                       "error": f"{type(error).__name__}: eviction failed"})
            except Exception:
                pass
        return result


def _cap_bytes(event: Callable[[dict], None] | None) -> int:
    raw = os.environ.get("STATEFRAME_ENV_SEED_MAX_GB", str(DEFAULT_MAX_GB)).strip()
    try:
        if not _DECIMAL_GB.fullmatch(raw):
            raise ValueError("invalid cap")
        amount = Decimal(raw)
        size = int(amount * 10**9)
        return max(1, size) if amount else 0
    except (ValueError, ArithmeticError):
        if event is not None:
            try:
                event({"event": "env_seed_cap_invalid"})
            except Exception:
                pass
        return DEFAULT_MAX_GB * 10**9


def clone(source: Path, destination: Path, interpreter: Path, source_repo: Path | None,
          authorized_roots: tuple[Path, ...], *,
          event: Callable[[dict], None] | None = None) -> Path:
    """Build or verify one seed under a per-key lock, then clone it into a new run env."""
    configured = os.environ.get("STATEFRAME_HOME")
    if configured:
        home = Path(configured)
    else:
        user_home = safe_home()
        if user_home is None:
            raise OSError("cannot find the StateFrame home: set HOME or STATEFRAME_HOME")
        home = user_home / ".stateframe"
    inputs = _key_inputs(source, interpreter, source_repo, authorized_roots,
                         home / "cache" / "env-seed-source-digest.json")
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    store = home / "env-seeds"
    store.mkdir(mode=0o700, parents=True, exist_ok=True)
    seed = store / key
    with (store / f".{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if seed.exists() and not _valid(seed, key):
            quarantined = store / f"quarantine-{time.time_ns()}-{key}"
            seed.rename(quarantined)
            if event is not None:
                event({"event": "env_seed_quarantined", "key": key,
                       "quarantine": quarantined.name})
        if not seed.exists():
            temporary = store / f".{key}.tmp-{os.getpid()}-{time.time_ns()}"
            try:
                shutil.copytree(source, temporary, symlinks=True)
                if _source_digest(source) != inputs["relocation"]["source_sha256"]:
                    raise OSError("source environment changed while seeding")
                files, directories = _inventory(temporary)
                for relative, row in files.items():
                    if "mode" in row:
                        (temporary / relative).chmod(row["mode"] & ~0o222)
                (temporary / "seed.json").write_text(json.dumps(
                    {"key": key, "inputs": inputs, "files": files,
                     "directories": directories}, sort_keys=True))
                _fsync_tree(temporary)
                _write_seed_bytes(seed, sum(row["size"] for row in files.values() if "size" in row))
                temporary.rename(seed)
                fd = os.open(store, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                try:
                    (store / f".{key}.bytes").unlink(missing_ok=True)
                except OSError:
                    pass
                raise
        manifest = json.loads((seed / "seed.json").read_text())
        _clone_files(seed, destination, manifest["files"], manifest["directories"])
        # The seed directory's mtime records its last use; gc prunes only long-unused seeds.
        os.utime(seed)
    evict_to_cap(store, _cap_bytes(event), key, event)
    return seed
