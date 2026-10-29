"""Fail-open, content-addressed disk persistence for turn-zero indexes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MODE_ENV = "STATEFRAME_TURN0_PERSIST"
CACHE_FORMAT = "stateframe-turn0-index/1"
MAX_CACHE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_FILES = 100_000
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 512 * 1024 * 1024
_NAME_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")

_StatIdentity = tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class CacheContext:
    """Stable cache identity plus file signatures for a single repository walk."""

    name: str
    key: str
    path: Path
    root: Path
    files: tuple[str, ...]
    signatures: tuple[tuple[str, _StatIdentity], ...]


class _ContextFailure(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(MODE_ENV, "off")).strip().lower() == "on"


def cache_directory() -> Path:
    """Default shared cache root; callers may replace this function in tests."""
    return Path.home() / ".cache" / "stateframe" / "repo-intel"


def implementation_version(schema: str, module_path: Path) -> str | None:
    """Bind entries to both the consuming module and this serialization helper."""
    try:
        source_rows: list[tuple[str, str]] = []
        for path in sorted(
                {Path(module_path).resolve(), Path(__file__).resolve()},
                key=lambda item: item.as_posix()):
            source_rows.append((
                path.name,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            ))
        encoded = _canonical_json(source_rows)
        return f"{schema}:{hashlib.sha256(encoded).hexdigest()}"
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _stat_identity(value: os.stat_result) -> _StatIdentity:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _relative_path(value: object) -> str:
    if not isinstance(value, (str, Path)):
        raise _ContextFailure("invalid_file_set")
    raw = os.fspath(value).replace("\\", "/")
    parts = raw.split("/")
    if (not raw or raw.startswith("/") or any(part in {"", ".", ".."} for part in parts)
            or "\x00" in raw):
        raise _ContextFailure("invalid_file_set")
    return raw


def _file_digest(root: Path, relative: str) -> tuple[int, str, _StatIdentity]:
    target = root.joinpath(*relative.split("/"))
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
        before_path = target.stat()
        if not stat.S_ISREG(before_path.st_mode):
            raise _ContextFailure("source_unavailable")
        if before_path.st_size > MAX_SOURCE_FILE_BYTES:
            raise _ContextFailure("source_limits")
        descriptor = os.open(
            target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise _ContextFailure("source_unavailable")
            digest = hashlib.sha256()
            size = 0
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_SOURCE_FILE_BYTES:
                    raise _ContextFailure("source_limits")
                digest.update(block)
            after = os.fstat(stream.fileno())
        after_path = target.stat()
    except _ContextFailure:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise _ContextFailure("source_unavailable") from error
    before_identity = _stat_identity(before)
    if (_stat_identity(after) != before_identity
            or _stat_identity(before_path) != before_identity
            or _stat_identity(after_path) != before_identity
            or size != before.st_size):
        raise _ContextFailure("source_changed")
    return size, digest.hexdigest(), before_identity


def _snapshot_digest() -> str:
    snapshot = Path(__file__).resolve().with_name("SNAPSHOT.sha256")
    try:
        raw = snapshot.read_bytes()
    except OSError as error:
        raise _ContextFailure("snapshot_unavailable") from error
    if not raw:
        raise _ContextFailure("snapshot_unavailable")
    return hashlib.sha256(raw).hexdigest()


def make_context(
    root: Path,
    *,
    task_id: str,
    name: str,
    module_version: str,
    walked_paths: Sequence[str],
    extra_key: Mapping[str, object] | None = None,
) -> tuple[CacheContext | None, str | None]:
    """Create a cache key from the module's exact walk and stable source bytes."""
    try:
        if not _NAME_RE.fullmatch(name):
            raise _ContextFailure("invalid_cache_name")
        if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 512:
            raise _ContextFailure("task_id_unavailable")
        if not isinstance(module_version, str) or not module_version:
            raise _ContextFailure("module_version_unavailable")
        resolved_root = Path(root).resolve(strict=True)
        if not resolved_root.is_dir():
            raise _ContextFailure("root_unavailable")
        files = tuple(sorted({_relative_path(item) for item in walked_paths}))
        if len(files) > MAX_SOURCE_FILES:
            raise _ContextFailure("source_limits")
        file_rows: list[tuple[str, int, str]] = []
        signatures: list[tuple[str, _StatIdentity]] = []
        total_bytes = 0
        for relative in files:
            size, digest, identity = _file_digest(resolved_root, relative)
            total_bytes += size
            if total_bytes > MAX_SOURCE_BYTES:
                raise _ContextFailure("source_limits")
            file_rows.append((relative, size, digest))
            signatures.append((relative, identity))
        file_set_digest = hashlib.sha256(_canonical_json(file_rows)).hexdigest()
        key_body = {
            "format": CACHE_FORMAT,
            "task_id": task_id.strip(),
            "file_set_sha256": file_set_digest,
            "module_version": module_version,
            "snapshot_sha256": _snapshot_digest(),
            "extra": dict(extra_key or {}),
        }
        key = hashlib.sha256(_canonical_json(key_body)).hexdigest()
        base = cache_directory()
        repo_key = hashlib.sha256(resolved_root.as_posix().encode("utf-8")).hexdigest()[:16]
        path = base / repo_key / "turn0" / f"{name}-{key}.json"
        return CacheContext(name, key, path, resolved_root, files, tuple(signatures)), None
    except _ContextFailure as error:
        return None, error.reason
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return None, "key_build_failed"


def sources_unchanged(
    context: CacheContext,
    walked_paths: Sequence[str] | None = None,
) -> bool:
    """Reject a hit/store if the walk or any file changed during cache processing."""
    try:
        if walked_paths is not None:
            current_files = tuple(sorted({_relative_path(item) for item in walked_paths}))
            if current_files != context.files:
                return False
        for relative, expected in context.signatures:
            current = context.root.joinpath(*relative.split("/")).stat()
            if _stat_identity(current) != expected:
                return False
        return True
    except (OSError, RuntimeError, TypeError, ValueError, _ContextFailure):
        return False


def read_payload(context: CacheContext) -> tuple[object | None, str | None]:
    """Read and verify one JSON envelope. Missing entries are ordinary cold misses."""
    try:
        raw = context.path.read_bytes()
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "read_error"
    if len(raw) > MAX_CACHE_BYTES:
        return None, "oversized"
    try:
        envelope = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        if not isinstance(envelope, Mapping) or set(envelope) != {
                "format", "name", "key", "payload_sha256", "payload"}:
            return None, "corrupt"
        if (envelope.get("format") != CACHE_FORMAT
                or envelope.get("name") != context.name
                or envelope.get("key") != context.key):
            return None, "corrupt"
        payload = envelope.get("payload")
        actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if envelope.get("payload_sha256") != actual:
            return None, "corrupt"
        return payload, None
    except (UnicodeError, ValueError, TypeError, OverflowError, RuntimeError):
        return None, "corrupt"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_payload(context: CacheContext, payload: object) -> str | None:
    """Atomically publish a verified JSON envelope without pruning any cache."""
    temporary: Path | None = None
    try:
        payload_bytes = _canonical_json(payload)
        if len(payload_bytes) > MAX_CACHE_BYTES:
            return "payload_oversized"
        envelope = {
            "format": CACHE_FORMAT,
            "name": context.name,
            "key": context.key,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "payload": payload,
        }
        encoded = _canonical_json(envelope)
        if len(encoded) > MAX_CACHE_BYTES:
            return "payload_oversized"
        context.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{context.path.name}.", suffix=".tmp", dir=context.path.parent)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, context.path)
        temporary = None
        _fsync_directory(context.path.parent)
        return None
    except (OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return "write_failed"
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def emit_cache_outcome(journal: object | None, *, name: str, outcome: str) -> None:
    """Record one bounded turn-zero cache hit or miss without exposing paths."""
    if outcome not in {"hit", "miss"}:
        return
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit("turn0_cache", name=name, outcome=outcome, dispatch=False)
    except Exception:
        pass


def emit_cache_bypass(journal: object | None, *, name: str, reason: str) -> None:
    """Record bounded cache failure facts without exposing paths or exception data."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit("cache_bypass", name=name, reason=reason, dispatch=False)
    except Exception:
        pass


__all__ = [
    "CACHE_FORMAT",
    "MODE_ENV",
    "CacheContext",
    "cache_directory",
    "emit_cache_bypass",
    "emit_cache_outcome",
    "enabled",
    "implementation_version",
    "make_context",
    "read_payload",
    "sources_unchanged",
    "write_payload",
]
