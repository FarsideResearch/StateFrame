"""Correct line cardinality for complete, newline-terminated source reads."""
from __future__ import annotations

import os

MODE_ENV = "STATEFRAME_READ_NEWLINE_FIX"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def complete_line_count_matches(content: str, start_line: int, end_line: int) -> bool:
    """Return whether a complete read contains exactly its declared lines.

    ``str.split("\\n")`` includes the empty sentinel after a terminal newline;
    ``splitlines`` reflects source lines and matches the read executor/ledger.
    """
    if mode() != "on":
        return len(content.split("\n")) == end_line - start_line + 1
    return len(content.splitlines()) == end_line - start_line + 1


__all__ = ["complete_line_count_matches", "mode"]
