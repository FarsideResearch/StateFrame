"""Opt-in integrity helpers for frame-arm edits, receipts, and recovery.

The frame arm historically had three presentation/dispatch seams which could
lose information while the history arm remained usable.  This module keeps
the repair switch in one place.  Importing it or calling any helper is a
no-op for the default ``off`` configuration; callers must explicitly gate
the new behavior with :func:`enabled`.
"""
from __future__ import annotations

import ast
import copy
import os
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_FRAME_EDIT_INTEGRITY"
_VALID_MODES = {"off", "on"}
_DEFINITION_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_TEST_KINDS = {"test", "run_tests", "pytest", "verification",
               "test_subprocess_completed", "test_subprocess_complete",
               "host_test_receipt", "test_receipt"}


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def definition_span(source: str, node: ast.AST) -> tuple[int, int]:
    """Return a definition's source span, including all decorators.

    ``ast.get_source_segment`` starts at ``def``/``class`` and therefore
    leaves an existing decorator in place when a complete decorated class is
    supplied as the replacement.  The frame integrity arm treats decorators
    as part of the definition's replaceable unit.
    """
    if not isinstance(node, _DEFINITION_TYPES):
        raise TypeError("definition span requires a function or class node")
    offsets = _line_offsets(source)
    decorators = list(getattr(node, "decorator_list", ()))
    starts = [node, *decorators]
    start_line = min(int(getattr(item, "lineno", 1)) for item in starts)
    start_col = min(int(getattr(item, "col_offset", 0)) for item in starts)
    end_line = int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))
    end_col = int(getattr(node, "end_col_offset", 0))
    start_index = offsets[max(0, start_line - 1)] + start_col
    end_index = offsets[max(0, end_line - 1)] + end_col
    return start_index, end_index


def definition_source(source: str, node: ast.AST) -> str:
    start, end = definition_span(source, node)
    return source[start:end]


def replace_definition(source: str, node: ast.AST, replacement: str) -> str:
    start, end = definition_span(source, node)
    return source[:start] + replacement + source[end:]


def _decorator_name(decorator: ast.AST) -> str:
    value = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        try:
            return ast.unparse(value)
        except (AttributeError, TypeError):
            return value.attr
    try:
        return ast.unparse(value)
    except (AttributeError, TypeError):
        return ast.dump(value, annotate_fields=False, include_attributes=False)


def duplicate_decorator_findings(source: str, *, file: str = "") -> list[dict[str, object]]:
    """Return deterministic structural findings for repeated decorators."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError, ValueError):
        return []
    findings: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, _DEFINITION_TYPES):
            continue
        counts: dict[str, int] = {}
        for decorator in getattr(node, "decorator_list", ()):
            name = _decorator_name(decorator)
            counts[name] = counts.get(name, 0) + 1
        for name, count in sorted(counts.items()):
            if count > 1:
                findings.append({
                    "file": file,
                    "line": int(getattr(node, "lineno", 1)),
                    "kind": "duplicate_decorator",
                    "symbol": str(getattr(node, "name", "<definition>")),
                    "decorator": name,
                    "count": count,
                })
    return sorted(findings, key=lambda row: (
        int(row["line"]), str(row["symbol"]), str(row["decorator"])))


def new_duplicate_decorator_findings(before: str | None, after: str,
                                    *, file: str = "") -> list[dict[str, object]]:
    before_rows = duplicate_decorator_findings(before or "", file=file)
    after_rows = duplicate_decorator_findings(after, file=file)
    before_keys = {
        (row["symbol"], row["decorator"], row["count"])
        for row in before_rows
    }
    return [row for row in after_rows
            if (row["symbol"], row["decorator"], row["count"]) not in before_keys]


def duplicate_decorator_rejection(before: str | None, candidate: str,
                                  *, file: str = "") -> str | None:
    rows = new_duplicate_decorator_findings(before, candidate, file=file)
    if not rows:
        return None
    details = ", ".join(
        f"{row['symbol']} has {row['count']}x @{row['decorator']}"
        for row in rows
    )
    return ("edit REJECTED: structural duplicate decorator detected (" + details
            + "); replace the target definition including its decorators")


def _rows(value: object) -> list[dict[str, object]]:
    if isinstance(value, Mapping):
        value = list(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _check_key(row: Mapping[str, object]) -> str | None:
    for key in ("check_id", "obligation_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    selectors = row.get("selectors", row.get("required_selectors"))
    if isinstance(selectors, str) and selectors.strip():
        return selectors.strip()
    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
        values = [str(item) for item in selectors if str(item).strip()]
        if values:
            return ", ".join(values)
    for key in ("selector", "suite", "test"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    kind = str(row.get("kind", row.get("tool", ""))).strip().lower()
    if kind in _TEST_KINDS:
        value = row.get("target", row.get("intent"))
        if value is not None and str(value).strip():
            return str(value)
    return None


def _is_check_row(row: Mapping[str, object]) -> bool:
    if _check_key(row) is not None:
        return True
    kind = str(row.get("kind", row.get("tool", ""))).strip().lower()
    return kind in _TEST_KINDS


def _order_key(row: Mapping[str, object], position: int) -> tuple[int, float, int]:
    for priority, key in enumerate(("sequence", "event_sequence", "timestamp_ns",
                                    "timestamp", "turn", "step"), start=6):
        value = row.get(key)
        if type(value) in {int, float}:
            return priority, float(value), position
    return 0, float(position), position


def _revision(row: Mapping[str, object]) -> str:
    for key in ("revision", "current_revision", "product_revision", "artifact_revision"):
        if row.get(key) is not None:
            return str(row[key])
    for key in ("after_binding", "binding", "current_binding"):
        binding = row.get(key)
        if isinstance(binding, Mapping):
            for nested in ("product_revision", "artifact", "revision", "current_revision"):
                if binding.get(nested) is not None:
                    return str(binding[nested])
    return "UNKNOWN"


def _status(row: Mapping[str, object]) -> str:
    for key in ("result_class", "result", "status", "outcome", "verdict"):
        if row.get(key) is not None:
            return str(row[key]).upper()
    exit_code = row.get("exit_code", row.get("exit"))
    if exit_code == 0:
        return "PASS"
    if exit_code == 1:
        return "FAIL"
    return "UNKNOWN"


def collapse_check_results(rows: object) -> list[dict[str, object]]:
    """Keep one current row per check and summarize older rows as superseded."""
    source = _rows(rows)
    groups: dict[str, list[tuple[int, dict[str, object]]]] = {}
    unkeyed: list[tuple[int, dict[str, object]]] = []
    for position, row in enumerate(source):
        if not _is_check_row(row):
            unkeyed.append((position, row))
            continue
        key = _check_key(row)
        if key is None:
            unkeyed.append((position, row))
        else:
            groups.setdefault(key, []).append((position, row))

    projected: list[tuple[int, dict[str, object]]] = list(unkeyed)
    for key, group in groups.items():
        winner_position, winner_source = max(
            group, key=lambda item: _order_key(item[1], item[0]))
        winner = copy.deepcopy(winner_source)
        winner.setdefault("check_id", key)
        superseded = []
        for position, row in sorted(group, key=lambda item: _order_key(item[1], item[0])):
            if position == winner_position and row is winner_source:
                continue
            superseded.append({
                "receipt_id": str(row.get("receipt_id", row.get("event_id", "UNKNOWN"))),
                "status": _status(row),
                "revision": _revision(row),
            })
        if superseded:
            winner["superseded"] = superseded
            winner["superseded_count"] = len(superseded)
            winner["superseded_revisions"] = list(dict.fromkeys(
                str(item["revision"]) for item in superseded))
        projected.append((winner_position, winner))
    projected.sort(key=lambda item: item[0])
    return [row for _position, row in projected]


def project_frame_results(frame: Mapping[str, object]) -> dict[str, object]:
    """Copy a frame and collapse all known result channels for presentation."""
    result = copy.deepcopy(dict(frame))
    for key in ("test_receipts", "recent_test_feedback", "probe_receipts", "receipts"):
        if key in result:
            result[key] = collapse_check_results(result[key])
    latest = result.get("latest_receipts")
    if isinstance(latest, Mapping):
        rows = collapse_check_results(latest)
        result["latest_receipts"] = {
            str(row.get("check_id", row.get("kind", index))): row
            for index, row in enumerate(rows)
        }
    for nested_key in ("completion_authority", "completion_status"):
        nested = result.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        nested_copy = dict(nested)
        if "receipts" in nested_copy:
            nested_copy["receipts"] = collapse_check_results(nested_copy["receipts"])
        result[nested_key] = nested_copy
    return result


__all__ = [
    "MODE_ENV", "collapse_check_results", "definition_source", "definition_span",
    "duplicate_decorator_findings", "duplicate_decorator_rejection", "enabled",
    "mode", "new_duplicate_decorator_findings", "project_frame_results",
    "replace_definition",
]
