"""Build live entry hints only from explicit public ticket wording."""

from __future__ import annotations

import re
from pathlib import Path


_ENTRY_MARKER = re.compile(
    r"\b(?:starting\s+point|entry\s+(?:point|file)|"
    r"start(?:ing)?\s+(?:at|in|from)|begin(?:ning)?\s+(?:at|in|from))\b",
    re.IGNORECASE,
)
_PATH_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_./-])(?:\./)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py"
    r"(?![A-Za-z0-9_.-])"
)
_ENTRY_LINE = re.compile(
    r"^The entry file is (?P<path>.+?)\. It is where the call chain starts, "
    r"not the only file you may need to change\.$"
)
_TARGET_LINE = re.compile(
    r"^The target file is (?P<path>.+?); this is the actual workspace path\.$"
)


def entry_file_from_ticket(
    ticket_text: object,
    *,
    workspace_root: Path | str | None = None,
) -> str | None:
    """Return a Python file explicitly marked as a ticket entry or start point."""
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        return None

    lines = ticket_text.splitlines()
    for line_number, line in enumerate(lines):
        for marker in _ENTRY_MARKER.finditer(line):
            prefix = line[:marker.start()]
            suffix = line[marker.end():]
            if (re.search(r"\bnot\s+(?:(?:the|an?)\s+)?$", prefix, re.IGNORECASE)
                    or re.match(r"\s*(?:is|:)\s+not\b", suffix, re.IGNORECASE)):
                continue
            paths = _PATH_TOKEN.findall(suffix)
            if not paths and line_number + 1 < len(lines) and lines[line_number + 1].strip():
                paths = _PATH_TOKEN.findall(lines[line_number + 1])
            if not paths:
                continue

            path = paths[0].replace("\\", "/").removeprefix("./")
            parts = path.split("/")
            if (path.startswith("/") or any(part in {"", ".", ".."} for part in parts)
                    or not path.endswith(".py")):
                continue
            if path.startswith("app/") and workspace_root is not None:
                try:
                    if (Path(workspace_root) / "backend").is_dir():
                        path = f"backend/{path}"
                except OSError:
                    pass
            return path
    return None


def operating_instructions_for_ticket(
    ticket_text: object,
    steps: int | None,
    *,
    workspace_root: Path | str | None = None,
) -> str:
    """Render operating instructions with only a ticket-authorized file hint."""
    path = entry_file_from_ticket(ticket_text, workspace_root=workspace_root)
    from kernel_step_contract import operating_instructions

    return operating_instructions(path, "", steps, ticket_entry_file=path is not None)


def entry_prompt_is_authorized(
    prompt: str,
    ticket_text: object,
    *,
    workspace_root: Path | str | None = None,
) -> bool:
    """Check that an emitted file-only entry line matches the public ticket."""
    lines = [line for line in prompt.splitlines()
             if line.startswith(("The entry point is ", "The entry file is ", "The target file is "))]
    if not lines:
        return True
    if len(lines) != 1:
        return False
    match = _ENTRY_LINE.fullmatch(lines[0]) or _TARGET_LINE.fullmatch(lines[0])
    expected = entry_file_from_ticket(ticket_text, workspace_root=workspace_root)
    return bool(match and expected and match.group("path") == expected)
