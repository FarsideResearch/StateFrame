"""Read and seal preflight records without changing the bytes written by the engine."""
from __future__ import annotations

import io
import json
import lzma
import os
import re
import stat
import tarfile
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

PACK_NAME = "requests.pack.xz"
_RECORD_NAME = re.compile(r"^[0-9]{4,}\.json$")


class RequestRecordsError(ValueError):
    """The loose or sealed preflight request records cannot be read safely."""


_SECRET_GUARD_REASON = "checkpoint refused: registered secret in state"
_STARTUP_HOME_IDENTITY_UNSET = object()


def _checkpoint_guard():
    from .engine import kernel_checkpoint

    return kernel_checkpoint


def _preserve_json_object_members(
        pairs: list[tuple[str, object]]) -> tuple[tuple[str, object], ...]:
    """Keep all members, including duplicate keys, while decoding JSON records."""
    return tuple(pairs)


def _iter_json_strings(value: object) -> Iterator[str]:
    """Yield keys and values, including JSON inside strings, within guard limits."""
    guard = _checkpoint_guard()
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if isinstance(current, str):
            yield current
            if (depth >= guard._SECRET_JSON_MAX_DEPTH
                    or len(current) > guard._SECRET_JSON_MAX_BYTES
                    or len(current.encode("utf-8", errors="surrogatepass")) > guard._SECRET_JSON_MAX_BYTES
                    or not current.lstrip().startswith(("{", "[", '"'))):
                continue
            try:
                decoded = json.loads(current, object_pairs_hook=_preserve_json_object_members)
            except (ValueError, RecursionError):
                continue  # Ordinary message text need not itself be JSON.
            pending.append((decoded, depth + 1))
        elif isinstance(current, (list, tuple)):
            pending.extend((item, depth) for item in current)


def _check_bundle_records(records: dict[str, bytes]) -> None:
    guard = _checkpoint_guard()
    try:
        for data in records.values():
            # Scan serialized bytes too: an object-shaped record may contain an
            # escaped key far larger than the structural traversal's parse budget.
            guard._check_registered_secrets(data)
            try:
                decoded = json.loads(data, object_pairs_hook=_preserve_json_object_members)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RequestRecordsError(
                    "preflight request record is not valid JSON") from error
            for value in _iter_json_strings(decoded):
                guard._check_registered_secrets(value.encode("utf-8", errors="surrogatepass"))
    except guard.CheckpointError as error:
        raise RequestRecordsError(_SECRET_GUARD_REASON) from error


def _check_bundle_pack_bytes(guard, pack: Path) -> None:
    if not pack.is_file():
        return
    try:
        guard.check_registered_secret_file(pack)
    except guard.CheckpointError as error:
        raise RequestRecordsError(_SECRET_GUARD_REASON) from error


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
# This is a reader optimization only: security verification always bypasses it.
_PACK_CACHE: dict[str, tuple[tuple[int, int, int], dict[str, bytes]]] = {}


class _RequestTarInfo(tarfile.TarInfo):
    """Validate physical headers before tarfile consumes hidden extensions."""

    def _proc_member(self, archive: tarfile.TarFile) -> _RequestTarInfo:
        # isfile() also accepts GNU sparse and contiguous-file extensions.
        # PAX/GNU pseudo-members must be refused before tarfile processes them.
        if (self.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                or not _valid_name(self.name)
                or Path(self.name).name != self.name):
            raise RequestRecordsError("sealed request pack contains an unexpected member")
        return super()._proc_member(archive)


def _check_decompressed_pack(pack: Path) -> None:
    """Scan headers, padding and trailers, not only the records tarfile exposes."""
    guard = _checkpoint_guard()
    scanner = None
    try:
        with lzma.open(pack, "rb") as stream:
            while block := stream.read(65536):
                scanner = guard._check_registered_secrets(block, scanner, final=False)
        guard._check_registered_secrets(b"", scanner)
    except guard.CheckpointError as error:
        raise RequestRecordsError(_SECRET_GUARD_REASON) from error


def _pack_records(pack: Path, *, use_cache: bool = True) -> dict[str, bytes]:
    """Read every sealed record; security callers also scan the raw tar stream."""
    try:
        info = pack.stat()
    except FileNotFoundError:
        return {}
    except OSError as error:
        raise RequestRecordsError(f"sealed request pack is unreadable: {error}") from error
    identity = (info.st_ino, info.st_size, info.st_mtime_ns)
    cached = _PACK_CACHE.get(str(pack))
    if use_cache and cached is not None and cached[0] == identity:
        return cached[1]
    records: dict[str, bytes] = {}
    try:
        if not use_cache:
            _check_decompressed_pack(pack)
        with tarfile.open(pack, mode="r:xz", tarinfo=_RequestTarInfo) as archive:
            for member in archive:
                if member.name in records:
                    raise RequestRecordsError("sealed request pack contains a duplicate member")
                source = archive.extractfile(member)
                if source is None:
                    raise RequestRecordsError("sealed request pack member is unreadable")
                with source:
                    data = source.read()
                # Security callers scan decompressed bytes before records enter the mapping.
                # Reader caching must never hide a member from bundle verification.
                if not use_cache:
                    _check_bundle_records({member.name: data})
                records[member.name] = data
    except (OSError, tarfile.TarError, lzma.LZMAError, EOFError) as error:
        raise RequestRecordsError(f"sealed request pack is damaged: {error}") from error
    _PACK_CACHE.clear()
    _PACK_CACHE[str(pack)] = (identity, records)
    return records


def _enumerate(root: Path) -> tuple[dict[str, Path], set[str]]:
    """One loose-directory listing plus the sealed name set; loose listing errors always raise."""
    loose = _loose_files(root)
    try:
        packed = set(_pack_records(root / PACK_NAME))
    except RequestRecordsError:
        if not loose:
            raise
        packed = set()
    return loose, packed


def list_request_records(run_dir: str | os.PathLike[str]) -> list[str]:
    """Return canonical record names from the loose and/or sealed representation."""
    loose, packed = _enumerate(Path(run_dir))
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


def _read_listed(pack: Path, name: str, loose: dict[str, Path]) -> bytes:
    """Read one record using the listing taken once for the whole pass (loose wins over sealed)."""
    path = loose.get(name)
    if path is not None:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            pass  # Unlinked after listing: seal publishes the pack before removing loose files.
    return _read_packed(pack, name)


def _all_records(run_dir: Path) -> dict[str, bytes]:
    """Read every record with a single directory listing instead of one listing per record."""
    loose, packed = _enumerate(run_dir)
    pack = run_dir / PACK_NAME
    return {name: _read_listed(pack, name, loose) for name in sorted(set(loose) | packed)}


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


def seal_request_records(
        run_dir: str | os.PathLike[str], *,
        startup_home_identity: object = _STARTUP_HOME_IDENTITY_UNSET,
) -> int:
    """Atomically pack preflight records, then remove loose files only after verification."""
    root = Path(run_dir)
    guard = _checkpoint_guard()

    def verify_home_identity() -> None:
        if startup_home_identity is _STARTUP_HOME_IDENTITY_UNSET:
            guard.verify_stateframe_home_identity(root, allow_legacy=True)
        else:
            guard.verify_stateframe_home_identity(
                root, expected_identity=startup_home_identity)

    try:
        verify_home_identity()
    except guard.CheckpointError as error:
        raise RequestRecordsError(guard._failure_reason(error)) from error
    captured_secret_generation = guard._SECRET_GENERATION
    loose = _loose_files(root)
    pack = root / PACK_NAME
    if not loose:
        records = _pack_records(pack, use_cache=False)
        _check_bundle_records(records)
        _check_bundle_pack_bytes(guard, pack)
        return 0

    # A prior successful publication can coexist with loose files if cleanup was interrupted.
    # Verify duplicates before deleting anything, and rebuild only when the loose set adds names.
    packed = _pack_records(pack, use_cache=False)
    _check_bundle_records(packed)
    _check_bundle_pack_bytes(guard, pack)
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
        _check_bundle_records(records)
        with temporary.open("xb") as stream:
            _write_archive(stream, records)
            stream.flush()
            os.fsync(stream.fileno())
        if _pack_records(temporary, use_cache=False) != records:
            raise RequestRecordsError("sealed request pack differs from the loose records")
        try:
            guard.check_registered_secret_file(temporary)
            verify_home_identity()
            if guard._SECRET_GENERATION != captured_secret_generation:
                raise guard.CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
        except guard.CheckpointError as error:
            raise RequestRecordsError(guard._failure_reason(error)) from error
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
