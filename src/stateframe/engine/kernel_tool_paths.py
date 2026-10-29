"""Unique resource lookup for legacy tool shortcuts; never pick the first hit."""

from __future__ import annotations

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


def resolve_tool_path(root: Path, requested: str) -> Path:
    """Preserve exact paths; recover a missing path only when its basename is unique.

    Access-policy checks remain the caller's responsibility. A missing result is
    returned unchanged so each tool can report its own missing-resource outcome.
    """
    exact = root / requested
    try:
        contained = exact.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        contained = False
    if not contained:
        raise ToolPathEscape(f"Path {requested!r} is outside the workspace. Use a repository-relative path.")
    if exact.exists():
        return exact
    return unique_tool_match(root, requested, root.rglob(Path(requested).name)) or exact


def receipt_target_path(root: Path, executed_path: Path) -> str | None:
    """Bind a receipt to the executor's resolved resource, never its raw argument."""
    try:
        relative = executed_path.resolve().relative_to(root.resolve())
    except (ValueError, OSError, RuntimeError):
        return None
    return relative.as_posix() if relative.parts else None
