"""Unique resource lookup for legacy tool shortcuts; never pick the first hit."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable


class AmbiguousToolPath(ValueError):
    """The requested shortcut identifies multiple repository paths."""


class ToolPathEscape(AmbiguousToolPath):
    """The requested path (absolute, '..' or symlink) resolves outside the workspace root.

    Subclasses AmbiguousToolPath so every existing tool handler reports it as a refusal."""


def unique_tool_match(root: Path, requested: str, candidates: Iterable[Path]) -> Path | None:
    matches = sorted(set(candidates), key=lambda path: str(path))
    if len(matches) > 1:
        labels = [str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
                  for path in matches[:20]]
        raise AmbiguousToolPath(
            f"Ambiguous path {requested!r}: {len(matches)} matches ({', '.join(labels)}). "
            "Use an exact repository-relative path.")
    return matches[0] if matches else None


def _resolves_under(root: Path, path: Path) -> bool:
    """True when ``path`` (after following every symlink) stays inside ``root``."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def recover_by_basename(root: Path, requested: str) -> list[Path]:
    """Every path under ``root`` named like ``requested``'s basename that stays inside root.

    The basename is matched literally (no glob metacharacters), and a candidate that
    is, or sits under, a symlink resolving outside the workspace is skipped, so a
    repository-provided link can never become a recovered read/edit target.
    """
    name = Path(requested).name
    if name in ("", ".", ".."):
        return []
    return sorted(
        (candidate for candidate in root.rglob(glob.escape(name))
         if candidate.name == name and _resolves_under(root, candidate)),
        key=str)


def resolve_tool_path(root: Path, requested: str) -> Path:
    """Preserve exact paths; recover a missing path only when its basename is unique.

    Exact and recovered paths share one containment rule: they must resolve inside
    the workspace root. Access-policy checks remain the caller's responsibility. A
    missing result is returned unchanged so each tool can report its own
    missing-resource outcome.
    """
    exact = root / requested
    if not _resolves_under(root, exact):
        raise ToolPathEscape(f"Path {requested!r} is outside the workspace. Use a repository-relative path.")
    if exact.exists():
        return exact
    return unique_tool_match(root, requested, recover_by_basename(root, requested)) or exact


def receipt_target_path(root: Path, executed_path: Path) -> str | None:
    """Bind a receipt to the executor's resolved resource, never its raw argument."""
    try:
        relative = executed_path.resolve().relative_to(root.resolve())
    except (ValueError, OSError, RuntimeError):
        return None
    return relative.as_posix() if relative.parts else None
