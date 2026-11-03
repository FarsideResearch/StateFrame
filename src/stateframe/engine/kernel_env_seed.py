"""Immutable source-environment seeds and private run-environment clones."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


FICLONE = 0x40049409


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


def _key_inputs(source: Path, interpreter: Path, source_repo: Path | None,
                authorized_roots: tuple[Path, ...]) -> dict:
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
                       "source_sha256": _source_digest(source)},
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


def clone(source: Path, destination: Path, interpreter: Path, source_repo: Path | None,
          authorized_roots: tuple[Path, ...], *,
          event: Callable[[dict], None] | None = None) -> Path:
    """Build or verify one seed under a per-key lock, then clone it into a new run env."""
    inputs = _key_inputs(source, interpreter, source_repo, authorized_roots)
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    home = Path(os.environ.get("STATEFRAME_HOME") or Path.home() / ".stateframe")
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
                temporary.rename(seed)
                fd = os.open(store, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        manifest = json.loads((seed / "seed.json").read_text())
        _clone_files(seed, destination, manifest["files"], manifest["directories"])
        # The seed directory's mtime records its last use; gc prunes only long-unused seeds.
        os.utime(seed)
    return seed
