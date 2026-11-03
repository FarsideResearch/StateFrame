"""Canonical, domain-separated tree digests for engine revision families."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path


def file_entry(path: str, content: bytes) -> list[str]:
    return ["file", path, hashlib.sha256(content).hexdigest()]


def missing_entry(path: str) -> list[str]:
    return ["missing", path]


def unreadable_entry(path: str) -> list[str]:
    return ["unreadable", path]


def symlink_entry(path: str, target: str) -> list[str]:
    return ["symlink", path, target]


def dir_entry(path: str) -> list[str]:
    return ["dir", path]


def path_entry(path: Path, relative: str, reader: Callable[[Path], bytes] | None = None) -> list[str]:
    """Capture a path without following links or conflating read failures with bytes."""
    try:
        if path.is_symlink():
            return symlink_entry(relative, os.readlink(path))
        return file_entry(relative, (reader or Path.read_bytes)(path))
    except FileNotFoundError:
        return missing_entry(relative)
    except (OSError, UnicodeError):
        return unreadable_entry(relative)


def compatible_revision(domain: str, rows: Sequence[tuple[str, bytes | None, str]]) -> str:
    """Revision for sites whose value reaches model-visible text (request bytes).

    ``rows`` are (path, content, kind) in the site's own order, kind one of "file",
    "missing" or "unreadable" (content is None unless kind is "file").  When every row
    is a regular file with a non-empty path, the result is byte-identical to the
    historical ``path + NUL + content`` encoding, so normal runs keep their request
    bytes.  Any missing/unreadable row (the case that used to hash a sentinel such as
    "<missing>", which real content can equal) switches to a NUL-led canonical document.
    A legacy preimage always starts with a non-empty, NUL-free path, so the two
    encodings cannot produce the same preimage.  Returns hex only.
    """
    if not domain or "/" not in domain:
        raise ValueError("tree digest domain must include a version")
    if not rows:
        return hashlib.sha256().hexdigest()
    if all(kind == "file" and path and "\0" not in path and content is not None
           for path, content, kind in rows):
        digest = hashlib.sha256()
        for path, content, _kind in rows:
            digest.update(path.encode("utf-8") + b"\0")
            digest.update(content)
        return digest.hexdigest()
    entries = [file_entry(path, content) if kind == "file" and content is not None
               else missing_entry(path) if kind == "missing" else unreadable_entry(path)
               for path, content, kind in rows]
    document = {"format": domain, "entries": entries}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(b"\0" + encoded).hexdigest()


def read_row(path: Path, relative: str) -> tuple[str, bytes | None, str]:
    """(relative, bytes, kind), following symlinks as Path.read_bytes always did there."""
    try:
        return relative, path.read_bytes(), "file"
    except FileNotFoundError:
        return relative, None, "missing"
    except (OSError, UnicodeError):
        return relative, None, "unreadable"


def tree_digest(domain: str, entries: Iterable[Sequence[object]]) -> str:
    """Hash one framed document; no path, type, or content boundary is implicit."""
    if not domain or "/" not in domain:
        raise ValueError("tree digest domain must include a version")
    document = {"format": domain, "entries": sorted([list(entry) for entry in entries],
                                                     key=lambda entry: json.dumps(
                                                         entry, sort_keys=True, separators=(",", ":"),
                                                         ensure_ascii=True))}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
