"""Resolve the public test path for a single-file synthetic build task.

Build rows have existed in several shapes over time.  The path is sometimes a
private/legacy field and sometimes part of the public task contract.  Keeping
the lookup here prevents the repair loop and the convergence oracle from
silently choosing different files.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Mapping, Sequence


_TASK_TEXT_FIELDS = (
    "msg",
    "message",
    "task",
    "brief",
    "symptom",
    "description",
    "prompt",
    "spec",
)
_TASK_TEST_PATH_RE = re.compile(
    r"(?<![\w./-])(?P<path>tests/[A-Za-z0-9_./-]*?\.py)"
    r"(?:::[A-Za-z_][A-Za-z0-9_]*(?:\[[^\s\]]+\])?)?"
    r"(?![\w-])"
)


def _safe_path(value: object) -> str | None:
    """Normalize one declared path, returning ``None`` for unsafe values."""

    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().split("::", 1)[0].strip("`'\".,);]}")
    raw = raw.replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    path = pathlib.PurePosixPath(raw)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:1] != ("tests",)
        or path.suffix != ".py"
    ):
        return None
    return path.as_posix()


def _path_values(value: object) -> list[object]:
    """Extract path-shaped values from legacy public-test declarations."""

    if isinstance(value, Mapping):
        return [value.get("path"), value.get("test_path")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values: list[object] = []
        for item in value:
            values.extend(_path_values(item))
        return values
    return [value]


def _declared_candidates(row: Mapping[str, object]) -> list[str]:
    """Return valid declared paths in the contract's fallback order."""

    values: list[object] = [row.get("test_path")]
    values.extend(_path_values(row.get("public_test_path")))
    values.extend(_path_values(row.get("public_test")))
    values.extend(_path_values(row.get("public_check_paths")))
    result: list[str] = []
    for value in values:
        path = _safe_path(value)
        if path and path not in result:
            result.append(path)
    return result


def _task_text(row: Mapping[str, object]) -> str:
    parts: list[str] = []
    for field in _TASK_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif field == "spec" and isinstance(value, Mapping):
            for key in ("text", "message", "task", "description"):
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    parts.append(nested)
    return "\n".join(parts)


def _text_candidates(row: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    for match in _TASK_TEST_PATH_RE.finditer(_task_text(row)):
        path = _safe_path(match.group("path"))
        if path and path not in result:
            result.append(path)
    return result


def resolve_build_test_path(row: Mapping[str, object], fn: str | None = None) -> str:
    """Return the one public test path for a non-multifile build row.

    The text path wins over declarations when both are present and disagree:
    it is the path the model was explicitly instructed to read.  Multiple
    conflicting text paths are rejected because choosing one would fabricate an
    oracle contract.  Legacy build rows that carry a ``test_src`` payload but
    predate an explicit public-path field retain their historical single-file
    selector.  Rows without that payload still fail closed: a function name
    alone is not enough to invent a public oracle path.
    """

    text_paths = _text_candidates(row)
    declared = _declared_candidates(row)
    if len(text_paths) > 1:
        raise ValueError(
            "build task names multiple conflicting test paths in its task text: "
            + ", ".join(text_paths)
        )
    if text_paths:
        return text_paths[0]
    if declared:
        return declared[0]
    legacy_fn = fn if isinstance(fn, str) else row.get("fn")
    if row.get("repo") == "build" and "test_src" in row and isinstance(legacy_fn, str):
        return f"tests/test_{legacy_fn.strip('_')}_build.py"
    raise ValueError(
        "build task has no public test path (no explicit row test_path); declare "
        "test_path, public_test_path, "
        "public_test, or public_check_paths, or name a tests/*.py path in the task "
        "message/spec text"
    )


__all__ = ["resolve_build_test_path"]
