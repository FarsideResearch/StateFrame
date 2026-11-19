"""Validated, atomic cross-process rows for the optional symbol registry."""

from __future__ import annotations

import hashlib
import json
import locale
import logging
import os
from pathlib import Path
import stat
import sys
import tempfile
from time import time_ns
from typing import Mapping


SCHEMA = "stateframe-symbol-cache/1"
_RACY_NS = 2_000_000_000
_MAX_CACHE_BYTES = 64 * 1024 * 1024
logger = logging.getLogger(__name__)


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def cache_path(root: Path, skip: tuple[str, ...]) -> tuple[Path, str]:
    """Keep each resolved repo and parser policy in its own versioned directory."""
    from mc_repo_intel_vendor.call_graph import repo_cache_file

    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    registry_source = hashlib.sha256((Path(__file__).with_name("symbol_registry.py")).read_bytes()).hexdigest()
    policy = _digest((SCHEMA, implementation, registry_source,
                      sys.version_info[:3], sys.flags.utf8_mode,
                      locale.getencoding(), skip))
    base = repo_cache_file(root, "symbols-v1.json").parent
    if base.resolve().is_relative_to(root):
        raise ValueError("symbol cache directory resolves inside the repository")
    directory = base / f"symbol-registry-v1-{policy[:16]}"
    directory.mkdir(mode=0o700, exist_ok=True)
    if directory.is_symlink() or directory.resolve().is_relative_to(root):
        raise ValueError("symbol cache directory is unsafe")
    return directory / "symbols-v1.json", policy


def signature(path: Path) -> tuple[list[int], int] | None:
    """A conservative POSIX identity; fresh timestamps are never stat-only hits."""
    try:
        observed = time_ns()
        info = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None
    return ([info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
             info.st_ctime_ns, info.st_mode, info.st_nlink], observed)


def reusable(row: object, signature_value: list[int], observed: int, relative: str) -> bool:
    if not isinstance(row, dict) or row.get("signature") != signature_value:
        return False
    verified = row.get("verified_at")
    definitions = row.get("defs")
    valid_rows = (isinstance(definitions, dict) and all(
        isinstance(name, str) and isinstance(items, list) and all(
            isinstance(item, dict) and item.get("file") == relative
            and isinstance(item.get("line"), int)
            and isinstance(item.get("signature"), str)
            and item.get("kind") in {"function", "method", "class", "constant"}
            for item in items)
        for name, items in definitions.items()))
    digest = row.get("digest")
    return (type(verified) is int and verified <= observed
            and max(signature_value[3:5]) < verified - _RACY_NS
            and isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            and valid_rows)


def load(path: Path, policy: str) -> dict[str, dict]:
    try:
        if path.is_symlink() or path.stat().st_size > _MAX_CACHE_BYTES:
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return {}
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA or payload.get("policy") != policy:
        return {}
    rows = payload.get("rows")
    try:
        valid_digest = isinstance(rows, dict) and payload.get("digest") == _digest(rows)
    except (RecursionError, TypeError, ValueError):
        return {}
    if not valid_digest:
        return {}
    try:
        root_digest = _digest(sorted((name, row["digest"]) for name, row in rows.items()))
    except (KeyError, TypeError, ValueError):
        return {}
    if payload.get("root_digest") != root_digest:
        return {}
    return {key: value for key, value in rows.items()
            if isinstance(key, str) and isinstance(value, dict)}


def publish(path: Path, policy: str, rows: Mapping[str, dict]) -> None:
    root_digest = _digest(sorted((name, row["digest"]) for name, row in rows.items()))
    payload = {"schema": SCHEMA, "policy": policy, "rows": rows,
               "root_digest": root_digest, "digest": _digest(rows)}
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=".symbols-", dir=path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_encoded(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        # The cache is advisory. A failed write leaves the previous complete
        # generation available; the caller still has its freshly built rows.
        logger.debug("Symbol cache publication failed: %s", type(error).__name__)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
