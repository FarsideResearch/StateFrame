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
    "symlink", "missing" or "unreadable". Symlink content is its target as filesystem
    bytes. Any number of readable regular files with non-empty, NUL-free,
    surrogate-free paths and NUL-free content retain the historical
    ``path + NUL + content`` bytes. Other rows use a NUL-led typed document.
    Known residual: without a row terminator, DIFFERENT path lists can still
    collide (``a="bc", d="e"`` versus ``a="b", cd="e"``). Callers that compare
    revisions must compare the ordered path list as well. A legacy preimage
    starts with a non-empty path, so the two forms cannot meet.
    """
    if not domain or "/" not in domain:
        raise ValueError("tree digest domain must include a version")
    if not rows:
        return hashlib.sha256().hexdigest()
    if all(kind == "file" and path and "\0" not in path and content is not None
           and not any(0xD800 <= ord(character) <= 0xDFFF for character in path)
           and b"\0" not in content
           for path, content, kind in rows):
        digest = hashlib.sha256()
        for path, content, _kind in rows:
            digest.update(path.encode("utf-8") + b"\0")
            digest.update(content)
        return digest.hexdigest()
    entries = [file_entry(path, content) if kind == "file" and content is not None
               else symlink_entry(path, os.fsdecode(content)) if kind == "symlink" and content is not None
               else missing_entry(path) if kind == "missing" else unreadable_entry(path)
               for path, content, kind in rows]
    return _typed_compatible_revision(domain, entries)


def _typed_compatible_revision(domain: str, entries: Sequence[Sequence[str]]) -> str:
    document = {"format": domain, "entries": entries}
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(b"\0" + encoded).hexdigest()


def read_row(path: Path, relative: str) -> tuple[str, bytes | None, str]:
    """(relative, bytes, kind), preserving links as their own type."""
    try:
        if path.is_symlink():
            return relative, os.fsencode(os.readlink(path)), "symlink"
        return relative, path.read_bytes(), "file"
    except FileNotFoundError:
        return relative, None, "missing"
    except (OSError, UnicodeError):
        return relative, None, "unreadable"


def compatible_text_revision(domain: str, files: dict[str, str], *, terminator: bool) -> str:
    """Keep legacy bytes for NUL-free text; caller keys include paths if unterminated."""
    if not domain or "/" not in domain:
        raise ValueError("tree digest domain must include a version")
    safe = all(
               path and "\0" not in path and "\0" not in source
               and not any(0xD800 <= ord(character) <= 0xDFFF
                           for character in path + source)
               for path, source in files.items())
    if safe:
        digest = hashlib.sha256()
        for path in sorted(files):
            digest.update(path.encode("utf-8") + b"\0")
            digest.update(files[path].encode("utf-8"))
            if terminator:
                digest.update(b"\0")
        return digest.hexdigest()
    return _typed_compatible_revision(
        domain,
        [file_entry(path, files[path].encode("utf-8", "surrogatepass"))
         for path in sorted(files)],
    )


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
