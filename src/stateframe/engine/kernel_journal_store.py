"""Standard-library storage helpers for measurement journals and blob packs."""
from __future__ import annotations

import functools
import hashlib
import json
import lzma
import os
import stat
import struct
import tempfile
import uuid
import zlib
from contextlib import contextmanager
from pathlib import Path


_JOURNAL_BLOB_MAX_BYTES = 256 * 1024 * 1024
_PACK_HEADER = ">4sI32s"
_PACK_CRC = ">I"
_PACK_NAME = "pack-0001.sfpk"
_PACK_INDEX = "pack-0001.idx"


def _journal_file(path_or_dir: str | os.PathLike[str]) -> Path:
    path = Path(path_or_dir)
    plain = path / "events.jsonl" if path.is_dir() else path
    if plain.suffix == ".xz":
        plain = plain.with_suffix("")
    return plain if plain.exists() else Path(str(plain) + ".xz")


@contextmanager
def open_journal(path_or_dir: str | os.PathLike[str], *, binary: bool = False):
    """Open live or sealed JSONL; binary mode retains uncompressed byte offsets."""
    path = _journal_file(path_or_dir)
    if path.suffix == ".xz":
        if binary:
            with lzma.open(path, "rb") as stream:
                yield stream
        else:
            with lzma.open(path, "rt", encoding="utf-8") as stream:
                yield stream
    else:
        with path.open("rb" if binary else "r",
                       **({} if binary else {"encoding": "utf-8"})) as stream:
            yield stream


def journal_paths(root_glob_base: str | os.PathLike[str]) -> list[Path]:
    """Find one measurement journal per child directory, preferring its live file."""
    root = Path(root_glob_base)
    return [path for child in sorted(root.iterdir()) if child.is_dir() and not child.is_symlink()
            if (path := _journal_file(child)).is_file()] if root.is_dir() else []


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _engine_lock_path(directory: Path) -> Path | None:
    if directory.parent.name != "measurement-runs" or directory.parent.parent.name != "worker":
        return None
    return directory.parents[2] / ".engine.lock"


@contextmanager
def _live_engine_lock_guard(directory: Path):
    """Hold an existing engine lock while sealing, or report unverifiable liveness."""
    lock_path = _engine_lock_path(directory)
    if lock_path is None:
        yield None
        return
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(lock_path, flags)
    except FileNotFoundError:
        # Legacy runs without the engine marker can only be sealed by a finished-run caller.
        yield None
        return
    except OSError:
        yield "cannot verify the engine writer lock"
        return

    import fcntl

    locked = False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            yield "cannot verify the engine writer lock"
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield "live engine holds .engine.lock"
            return
        except OSError:
            yield "cannot verify the engine writer lock"
            return
        locked = True
        yield None
    finally:
        try:
            if locked:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _write_sealed_journal(plain: Path, temporary: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with plain.open("rb") as source, temporary.open("xb") as raw:
        with lzma.LZMAFile(raw, "wb", preset=6) as target:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                target.write(chunk)
        raw.flush()
        os.fsync(raw.fileno())
    return size, digest.hexdigest()


def _verify_sealed_journal(path: Path, size: int, digest: str) -> None:
    actual = hashlib.sha256()
    actual_size = 0
    with lzma.open(path, "rb") as source:
        while chunk := source.read(1024 * 1024):
            actual.update(chunk)
            actual_size += len(chunk)
    if actual_size != size or actual.hexdigest() != digest:
        raise ValueError("sealed journal does not match its plain source")


def _publish_sealed_journal(temporary: Path, sealed: Path) -> None:
    os.replace(temporary, sealed)


def _unlink_plain_journal(plain: Path) -> None:
    plain.unlink()


def seal_measurement_run(directory: str | os.PathLike[str]) -> dict[str, object]:
    """Durably seal a finished journal; callers must finish runs without a live marker first.

    The engine holds ``.engine.lock`` for its lifetime. When that marker exists, sealing takes the
    same lock and keeps it through publication. Legacy layouts without the marker remain
    caller-enforced: only call this function after the measurement writer has finished.
    """
    root = Path(directory)
    with _live_engine_lock_guard(root) as reason:
        if reason is not None:
            return {"sealed": False, "reason": reason}
        return _seal_measurement_run(root)


def _seal_measurement_run(root: Path) -> dict[str, int]:
    plain, sealed = root / "events.jsonl", root / "events.jsonl.xz"
    if not plain.exists():
        if not sealed.is_file():
            raise FileNotFoundError(plain)
        with lzma.open(sealed, "rb") as source:
            size = sum(len(chunk) for chunk in iter(lambda: source.read(1024 * 1024), b""))
        return {"journal_bytes": size, "sealed_bytes": sealed.stat().st_size}
    temporary = root / ("events.jsonl.xz." + uuid.uuid4().hex + ".tmp")
    try:
        size, digest = _write_sealed_journal(plain, temporary)
        _verify_sealed_journal(temporary, size, digest)
        _publish_sealed_journal(temporary, sealed)
        _fsync_directory(root)
        _unlink_plain_journal(plain)
        _fsync_directory(root)
    finally:
        temporary.unlink(missing_ok=True)
    return {"journal_bytes": size, "sealed_bytes": sealed.stat().st_size}


def _scan_blob_pack(pack: Path) -> tuple[dict[str, tuple[int, int]], int]:
    records: dict[str, tuple[int, int]] = {}
    fd = os.open(pack, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("journal blob pack is not a regular file")
        while True:
            offset = stream.tell()
            header = stream.read(struct.calcsize(_PACK_HEADER))
            if not header:
                return records, offset
            if len(header) != struct.calcsize(_PACK_HEADER):
                return records, offset
            magic, length, digest = struct.unpack(_PACK_HEADER, header)
            if magic != b"SFPK" or length > _JOURNAL_BLOB_MAX_BYTES:
                return records, offset
            payload = stream.read(length)
            crc_size = struct.calcsize(_PACK_CRC)
            crc = stream.read(crc_size)
            if (len(payload) != length or len(crc) != crc_size
                    or struct.unpack(_PACK_CRC, crc)[0] != zlib.crc32(header + payload)):
                return records, offset
            records[digest.hex()] = (offset, length)


def _pack_index(blob_dir: Path) -> tuple[dict[str, tuple[int, int]], int]:
    pack = blob_dir / _PACK_NAME
    index_path = blob_dir / _PACK_INDEX
    pack_info = pack.stat()
    try:
        index_info = index_path.stat()
        index_key = (index_info.st_ino, index_info.st_size, index_info.st_mtime_ns)
    except FileNotFoundError:
        index_key = (0, 0, 0)
    return _cached_pack_index(
        blob_dir, pack_info.st_ino, pack_info.st_size, pack_info.st_mtime_ns, index_key)


@functools.lru_cache(maxsize=32)
def _cached_pack_index(blob_dir: Path, pack_inode: int, size: int, pack_mtime_ns: int,
                       index_key: tuple[int, int, int]) -> tuple[dict[str, tuple[int, int]], int]:
    index_path = blob_dir / _PACK_INDEX
    try:
        fd = os.open(index_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("journal blob index is not a regular file")
            stored = json.load(stream)
        if stored["pack_size"] == size:
            records = {key: (int(value[0]), int(value[1]))
                       for key, value in stored["records"].items()}
            if all(len(key) == 64 and 0 <= offset < size and
                   0 <= length <= _JOURNAL_BLOB_MAX_BYTES and
                   offset + struct.calcsize(_PACK_HEADER) + length + struct.calcsize(_PACK_CRC) <= size
                   for key, (offset, length) in records.items()):
                return records, size
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        pass
    return _rebuild_pack_index(blob_dir)


def _rebuild_pack_index(blob_dir: Path) -> tuple[dict[str, tuple[int, int]], int]:
    pack = blob_dir / _PACK_NAME
    records, valid_end = _scan_blob_pack(pack)
    if valid_end == pack.stat().st_size:
        try:
            _write_pack_index(blob_dir, records, valid_end)
        except OSError:
            pass
    return records, valid_end


def _write_pack_index(blob_dir: Path, records: dict[str, tuple[int, int]], size: int) -> None:
    fd, name = tempfile.mkstemp(prefix=".pack-index-", dir=blob_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump({"pack_size": size, "records": records}, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, blob_dir / _PACK_INDEX)
        _fsync_directory(blob_dir)
    finally:
        Path(name).unlink(missing_ok=True)


def _read_packed_blob(blob_dir: Path, blob_hash: str,
                      records: dict[str, tuple[int, int]] | None = None) -> bytes:
    pack = blob_dir / _PACK_NAME
    if records is None:
        records, _ = _pack_index(blob_dir)
    if blob_hash not in records:
        records, _ = _rebuild_pack_index(blob_dir)
        if blob_hash not in records:
            raise FileNotFoundError(f"journal blob {blob_hash} is missing")
    offset, length = records[blob_hash]
    fd = os.open(pack, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        stream.seek(offset)
        header_size = struct.calcsize(_PACK_HEADER)
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError("packed journal blob header is truncated")
        magic, actual_length, digest = struct.unpack(_PACK_HEADER, header)
        if magic != b"SFPK" or actual_length != length or digest.hex() != blob_hash:
            raise ValueError("packed journal blob header is invalid")
        payload = stream.read(length)
        crc = stream.read(struct.calcsize(_PACK_CRC))
    if (len(payload) != length or len(crc) != struct.calcsize(_PACK_CRC)
            or struct.unpack(_PACK_CRC, crc)[0] != zlib.crc32(header + payload)):
        raise ValueError("packed journal blob checksum is invalid")
    return payload


def pack_blobs(directory: str | os.PathLike[str]) -> dict[str, int]:
    """Append loose compressed blobs to a durable pack, then publish its index."""
    import fcntl

    run_dir = Path(directory)
    if (run_dir / "events.jsonl").exists() or not (run_dir / "events.jsonl.xz").is_file():
        raise ValueError("journal blobs can be packed only after the measurement journal is sealed")
    blob_dir = run_dir / "blobs"
    if blob_dir.is_symlink() or not blob_dir.is_dir():
        raise FileNotFoundError("journal blob store is missing")
    lock_fd = os.open(blob_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        pack = blob_dir / _PACK_NAME
        fd = os.open(pack, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        os.close(fd)
        records, valid_end = _scan_blob_pack(pack)
        pack_size = pack.stat().st_size
        if valid_end != pack_size:
            fd = os.open(pack, os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                os.ftruncate(fd, valid_end)
                os.fsync(fd)
            finally:
                os.close(fd)
            _cached_pack_index.cache_clear()
        loose = sorted(blob_dir.glob("*.zlib"))
        fd = os.open(pack, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "ab") as stream:
            for path in loose:
                digest = path.stem
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    continue
                if digest in records:
                    continue
                payload = path.read_bytes()
                if len(payload) > _JOURNAL_BLOB_MAX_BYTES:
                    raise ValueError("compressed journal blob exceeds the supported size")
                offset = stream.tell()
                header = struct.pack(_PACK_HEADER, b"SFPK", len(payload), bytes.fromhex(digest))
                stream.write(header + payload + struct.pack(_PACK_CRC, zlib.crc32(header + payload)))
                records[digest] = (offset, len(payload))
            stream.flush()
            os.fsync(stream.fileno())
            size = stream.tell()
        _write_pack_index(blob_dir, records, size)
        for path in loose:
            if path.stem in records:
                if path.read_bytes() != _read_packed_blob(blob_dir, path.stem, records):
                    raise ValueError("loose journal blob differs from its packed copy")
                path.unlink()
        _fsync_directory(blob_dir)
        return {"blobs": len(records), "pack_bytes": size}
    finally:
        os.close(lock_fd)
