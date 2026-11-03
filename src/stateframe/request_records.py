"""Read and seal preflight records without changing the bytes written by the engine."""
from __future__ import annotations

import io
import os
import re
import stat
import tarfile
import tempfile
import uuid
from pathlib import Path


PACK_NAME = "requests.pack.xz"
_RECORD_NAME = re.compile(r"^[0-9]{4,}\.json$")


class RequestRecordsError(ValueError):
    """The loose or sealed preflight request records cannot be read safely."""


def _valid_name(name: str) -> bool:
    return bool(_RECORD_NAME.fullmatch(name))


def _loose_files(run_dir: Path) -> dict[str, Path]:
    root = run_dir / "requests"
    if not root.exists():
        return {}
    if root.is_symlink() or not root.is_dir():
        raise RequestRecordsError("requests path is not a regular directory")
    result = {}
    for path in root.iterdir():
        if not _valid_name(path.name):
            continue
        try:
            mode = path.lstat().st_mode
        except OSError:
            continue
        if stat.S_ISREG(mode):
            result[path.name] = path
    return result


# Readers list the records and then read each one. Decoding the small pack once per file
# identity keeps that linear instead of rescanning the whole xz stream for every record.
_PACK_CACHE: dict[str, tuple[tuple[int, int, int], dict[str, bytes]]] = {}


def _pack_records(pack: Path) -> dict[str, bytes]:
    """Every record in a sealed pack, decoded in one pass; empty when there is no pack."""
    try:
        info = pack.stat()
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise RequestRecordsError(f"sealed request pack is unreadable: {error}") from error
    identity = (info.st_ino, info.st_size, info.st_mtime_ns)
    cached = _PACK_CACHE.get(str(pack))
    if cached is not None and cached[0] == identity:
        return cached[1]
    records: dict[str, bytes] = {}
    try:
        with tarfile.open(pack, mode="r:xz") as archive:
            for member in archive:
                if (not member.isfile() or not _valid_name(member.name)
                        or Path(member.name).name != member.name):
                    continue
                source = archive.extractfile(member)
                if source is not None:
                    with source:
                        records[member.name] = source.read()
    except (OSError, tarfile.TarError, EOFError) as error:
        raise RequestRecordsError(f"sealed request pack is damaged: {error}") from error
    _PACK_CACHE.clear()
    _PACK_CACHE[str(pack)] = (identity, records)
    return records


def list_request_records(run_dir: str | os.PathLike[str]) -> list[str]:
    """Return canonical record names from the loose and/or sealed representation."""
    root = Path(run_dir)
    loose = _loose_files(root)
    try:
        packed = set(_pack_records(root / PACK_NAME))
    except RequestRecordsError:
        if not loose:
            raise
        packed = set()
    return sorted(set(loose) | packed)


def _read_packed(pack: Path, name: str) -> bytes:
    records = _pack_records(pack)
    if name not in records:
        raise FileNotFoundError(f"preflight request record not found: {name}")
    return records[name]


def read_request_record(run_dir: str | os.PathLike[str], name: str) -> bytes:
    """Read the exact bytes of one preflight record from either storage form."""
    if not isinstance(name, str) or not _valid_name(name) or Path(name).name != name:
        raise ValueError("request record name must be a numbered .json basename")
    root = Path(run_dir)
    loose = _loose_files(root).get(name)
    if loose is not None:
        return loose.read_bytes()
    return _read_packed(root / PACK_NAME, name)


def _all_records(run_dir: Path) -> dict[str, bytes]:
    return {name: read_request_record(run_dir, name) for name in list_request_records(run_dir)}


def _write_archive(stream, records: dict[str, bytes]) -> None:
    with tarfile.open(fileobj=stream, mode="w:xz", preset=6) as archive:
        for name, data in records.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))


def request_pack_size(run_dir: str | os.PathLike[str]) -> int:
    """Measure a prospective tar.xz pack without writing in the run directory."""
    records = _all_records(Path(run_dir))
    if not records:
        return 0
    with tempfile.TemporaryFile(mode="w+b") as output:
        _write_archive(output, records)
        output.flush()
        return output.tell()


def seal_request_records(run_dir: str | os.PathLike[str]) -> int:
    """Atomically pack preflight records, then remove loose files only after verification."""
    root = Path(run_dir)
    loose = _loose_files(root)
    pack = root / PACK_NAME
    if not loose:
        return 0

    # A prior successful publication can coexist with loose files if cleanup was interrupted.
    # Verify duplicates before deleting anything, and rebuild only when the loose set adds names.
    packed = _pack_records(pack)
    for name in set(loose) & set(packed):
        if loose[name].read_bytes() != packed[name]:
            raise RequestRecordsError(f"loose request record differs from sealed copy: {name}")
    if set(packed) >= set(loose):
        removed = sum(path.stat().st_size for path in loose.values())
        for path in loose.values():
            path.unlink()
        _fsync_directory(root / "requests")
        return removed

    temporary = root / f".{PACK_NAME}.{uuid.uuid4().hex}.tmp"
    try:
        records = _all_records(root)
        with temporary.open("xb") as stream:
            _write_archive(stream, records)
            stream.flush()
            os.fsync(stream.fileno())
        if _pack_records(temporary) != records:
            raise RequestRecordsError("sealed request pack differs from the loose records")
        os.replace(temporary, pack)
        _fsync_directory(root)
        removed = 0
        for path in loose.values():
            removed += path.stat().st_size
            path.unlink()
        _fsync_directory(root / "requests")
        return removed
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
