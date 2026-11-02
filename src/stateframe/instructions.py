"""Project instructions: the repository's own AGENTS.md or CLAUDE.md, read once per run.

The host resolves and bounds the file before the engine starts, records its full-byte
digest in the run spec, and resume refuses when that file changed or disappeared. The
text is untrusted repository content: the engine frames it as such and it never changes
the tool menu, the sandbox or any engine rule.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

SCHEMA = "stateframe-project-instructions/1"
CANDIDATES = ("AGENTS.md", "CLAUDE.md")
MAX_SHOWN_BYTES = 6144       # bytes of instruction text placed in every request
MAX_FILE_BYTES = 1 << 20     # larger files are refused rather than hashed and truncated


class InstructionsError(ValueError):
    pass


def _read_regular(path: Path) -> bytes:
    """Read a regular file without following a final symlink; refuse anything else."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise InstructionsError(f"cannot read project instructions {path}: {error.strerror}") from error
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise InstructionsError(f"project instructions {path} is not a regular file")
        if info.st_size > MAX_FILE_BYTES:
            raise InstructionsError(f"project instructions {path} exceeds {MAX_FILE_BYTES} bytes")
        chunks = []
        remaining = MAX_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_FILE_BYTES:
            raise InstructionsError(f"project instructions {path} exceeds {MAX_FILE_BYTES} bytes")
        return data
    finally:
        os.close(fd)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _repo_file(repo: Path, relative: str) -> Path | None:
    """Return the in-repo file for ``relative``; a symlink must stay inside the repository."""
    root = repo.resolve()
    candidate = root / relative
    if not candidate.exists() and not candidate.is_symlink():
        return None
    target = candidate.resolve()
    if not _inside(target, root):
        raise InstructionsError(f"project instructions {relative} is a symlink outside the repository")
    return target


def _render(data: bytes) -> tuple[str, int, bool]:
    """Bounded UTF-8 text: whole lines up to MAX_SHOWN_BYTES, with a visible truncation note."""
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_SHOWN_BYTES:
        return text.rstrip("\n"), len(encoded), False
    head = encoded[:MAX_SHOWN_BYTES].decode("utf-8", errors="ignore")
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    shown = len(head.encode("utf-8"))
    note = (f"\n[Project instructions truncated: showing {shown} of {len(encoded)} bytes. "
            "Read the file for the rest.]")
    return head.rstrip("\n") + note, shown, True


def resolve(repo: Path, explicit: Path | None, *, discover: bool,
            readable=lambda relative: True) -> dict[str, object] | None:
    """Pick the instruction file for a run, or None.

    ``explicit`` wins; otherwise AGENTS.md, then CLAUDE.md at the repository root when
    ``discover`` is on. ``readable(relative)`` is the sandbox read policy: an in-repo file
    the working copy excludes is never shown.
    """
    root = repo.resolve()
    if explicit is not None:
        path = explicit.expanduser()
        if path.is_symlink():
            target = path.resolve()
            if _inside(path.absolute(), root) and not _inside(target, root):
                raise InstructionsError(f"project instructions {explicit} is a symlink outside the repository")
            path = target
        path = path.resolve()
        origin = "explicit"
        relative = path.relative_to(root).as_posix() if _inside(path, root) else None
        if relative is not None and not readable(relative):
            raise InstructionsError(f"project instructions {explicit} is excluded by the read policy")
        source = relative if relative is not None else str(path)
    else:
        if not discover:
            return None
        path = None
        for name in CANDIDATES:
            found = _repo_file(root, name)
            if found is not None:
                path = found
                break
        if path is None:
            return None
        relative = path.relative_to(root).as_posix()
        if not readable(relative):
            return None
        origin = "discovered"
        source = relative
    data = _read_regular(path)
    text, shown, truncated = _render(data)
    if not text.strip():
        return None
    return {"schema": SCHEMA, "origin": origin, "source": source, "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "shown_bytes": shown, "truncated": truncated, "text": text}


def validate(record: object) -> dict[str, object]:
    """Structural check of a recorded instruction record (run spec or settings file)."""
    keys = {"schema", "origin", "source", "path", "sha256", "bytes", "shown_bytes", "truncated", "text"}
    if not isinstance(record, dict) or set(record) != keys or record["schema"] != SCHEMA:
        raise InstructionsError("recorded project instructions are malformed")
    if record["origin"] not in {"explicit", "discovered"} or not isinstance(record["text"], str):
        raise InstructionsError("recorded project instructions are malformed")
    if len(record["text"].encode("utf-8")) > MAX_SHOWN_BYTES + 256:
        raise InstructionsError("recorded project instructions exceed the shown-byte bound")
    return record


def verify_unchanged(record: dict[str, object]) -> None:
    """Resume pin: the source file must still hash to the recorded digest."""
    validate(record)
    path = Path(str(record["path"]))
    try:
        data = _read_regular(path)
    except InstructionsError as error:
        raise InstructionsError(f"project instructions missing or unreadable: {record['source']} ({error})") from error
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise InstructionsError(f"project instructions changed since the run started: {record['source']}")


# Opt-in: a live A/B on the three bundled examples showed no measurable gain (see KNOWN_LIMITATIONS.md).
DEFAULT_MODE = "off"
