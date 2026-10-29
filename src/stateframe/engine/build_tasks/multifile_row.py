"""Normalization and validation for the H10 multi-file repair-row contract.

The adapter is allowed to receive platform-specific path spellings while it is
reading a patch, but rows crossing the corpus boundary use repository-relative
POSIX paths and complete source files.  This module owns that boundary and does
not mutate the caller's mapping.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import PurePosixPath


class MultiFileRowError(ValueError):
    """The row cannot satisfy the normalized multi-file contract."""


RowValidationError = MultiFileRowError

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:($|/)")


def normalize_repo_path(value: object, *, field: str = "path") -> str:
    """Return a safe repository-relative POSIX path from a patch/row spelling."""
    if not isinstance(value, str) or not value:
        raise MultiFileRowError(f"{field} must be a non-empty string path")
    if "\x00" in value:
        raise MultiFileRowError(f"{field} contains NUL")
    slash_path = value.replace("\\", "/")
    if _WINDOWS_DRIVE_RE.match(slash_path):
        raise MultiFileRowError(f"{field} must be repository-relative: {value!r}")
    pure = PurePosixPath(slash_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise MultiFileRowError(f"{field} must be repository-relative: {value!r}")
    normalized = str(pure)
    if not normalized or normalized == ".":
        raise MultiFileRowError(f"{field} must not be empty or '.'")
    return normalized


def _validated_path(value: object, *, field: str) -> str:
    normalized = normalize_repo_path(value, field=field)
    if value != normalized:
        raise MultiFileRowError(
            f"{field} is not normalized: expected {normalized!r}, got {value!r}"
        )
    return normalized


def _complete_source(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise MultiFileRowError(f"{field} must contain a complete source string")
    try:
        ast.parse(value, filename=field)
        compile(value, field, "exec")
    except SyntaxError as exc:
        raise MultiFileRowError(
            f"{field} must be a complete parseable source file, not a fragment: {exc.msg}"
        ) from exc
    return value


def _validate_source_map(
    value: object,
    *,
    field: str,
    require_sources: bool,
    allow_empty: bool = True,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MultiFileRowError(f"{field} must be a mapping of repository paths")
    if not allow_empty and not value:
        raise MultiFileRowError(f"{field} must not be empty")
    validated: dict[str, object] = {}
    for raw_path, raw_source in value.items():
        path = _validated_path(raw_path, field=f"{field} path")
        if path in validated:
            raise MultiFileRowError(f"{field} has duplicate path: {path}")
        if require_sources:
            validated[path] = _complete_source(
                raw_source,
                field=f"{field}[{path}]",
            )
        else:
            validated[path] = raw_source
    return validated


def _validate_code_files(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MultiFileRowError("row.files must be a mapping of repository paths")
    if not value:
        raise MultiFileRowError("row.files must not be empty")
    validated: dict[str, object] = {}
    for raw_path, raw_payload in value.items():
        path = _validated_path(raw_path, field="row.files path")
        if path in validated:
            raise MultiFileRowError(f"row.files has duplicate path: {path}")
        if not isinstance(raw_payload, Mapping):
            raise MultiFileRowError(f"row.files[{path}] must be a mapping")
        payload = dict(raw_payload)
        if "parent_src" not in payload or "fixed_src" not in payload:
            raise MultiFileRowError(
                f"row.files[{path}] must contain complete parent_src and fixed_src"
            )
        payload["parent_src"] = _complete_source(
            payload["parent_src"],
            field=f"row.files[{path}].parent_src",
        )
        payload["fixed_src"] = _complete_source(
            payload["fixed_src"],
            field=f"row.files[{path}].fixed_src",
        )
        validated[path] = payload
    return validated


def _validate_selectors(value: object, *, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not value:
        raise MultiFileRowError(f"{field} must be a non-empty selector list")
    selectors: list[str] = []
    for index, selector in enumerate(value):
        if not isinstance(selector, str) or not selector:
            raise MultiFileRowError(f"{field}[{index}] must be a non-empty string")
        if any(char in selector for char in ("\x00", "\n", "\r")):
            raise MultiFileRowError(f"{field}[{index}] contains a newline/NUL")
        selectors.append(selector)
    return selectors


def validate_multifile_row(row: Mapping[str, object]) -> dict[str, object]:
    """Validate an already normalized H10 row and return a shallow copy.

    Validation rejects non-normalized paths rather than silently repairing them;
    callers that ingest external data should use :func:`normalize_multifile_row`.
    """
    if not isinstance(row, Mapping):
        raise MultiFileRowError("multi_file row must be a mapping")
    if row.get("shape") != "multi_file":
        raise MultiFileRowError("row.shape must be 'multi_file'")
    flat_fields = sorted(
        field for field in ("path", "rel", "fn", "parent_src", "fixed_src") if field in row
    )
    if flat_fields:
        raise MultiFileRowError(
            "multi_file rows must keep complete sources under files; flat fragments: "
            + ", ".join(flat_fields)
        )
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise MultiFileRowError("row.task_id must be a non-empty string")

    files = _validate_code_files(row.get("files"))

    oracle_value = row.get("oracle")
    if not isinstance(oracle_value, Mapping):
        raise MultiFileRowError("row.oracle is required and must be a mapping")
    oracle_id = oracle_value.get("id")
    if not isinstance(oracle_id, str) or not oracle_id:
        raise MultiFileRowError("row.oracle.id must be a non-empty string")
    selectors = _validate_selectors(
        oracle_value.get("selectors"),
        field="row.oracle.selectors",
    )
    test_files = _validate_source_map(
        oracle_value.get("test_files"),
        field="row.oracle.test_files",
        require_sources=True,
    )
    overlap = sorted(set(files) & set(test_files))
    if overlap:
        raise MultiFileRowError(
            "row.files and row.oracle.test_files overlap: " + ", ".join(overlap)
        )

    changed_value = row.get("changed_functions")
    if not isinstance(changed_value, Mapping):
        raise MultiFileRowError("row.changed_functions is required and must be a mapping")
    changed_functions: dict[str, list[str]] = {}
    for raw_path, raw_functions in changed_value.items():
        path = _validated_path(raw_path, field="row.changed_functions path")
        if path not in files:
            raise MultiFileRowError(
                f"row.changed_functions names undeclared file: {path}"
            )
        if not isinstance(raw_functions, (list, tuple)) or not raw_functions:
            raise MultiFileRowError(
                f"row.changed_functions[{path}] must be a non-empty list"
            )
        functions: list[str] = []
        for index, function in enumerate(raw_functions):
            if not isinstance(function, str) or not function:
                raise MultiFileRowError(
                    f"row.changed_functions[{path}][{index}] must be a non-empty string"
                )
            functions.append(function)
        if len(set(functions)) != len(functions):
            raise MultiFileRowError(
                f"row.changed_functions[{path}] contains duplicate functions"
            )
        changed_functions[path] = functions
    if not changed_functions:
        raise MultiFileRowError("row.changed_functions must not be empty")

    has_entry = "entry" in row
    entry_value = row.get("entry")
    if has_entry:
        if not isinstance(entry_value, Mapping):
            raise MultiFileRowError("row.entry must be a mapping when present")
        entry_path = _validated_path(entry_value.get("path"), field="row.entry.path")
        entry_fn = entry_value.get("fn")
        if not isinstance(entry_fn, str) or not entry_fn:
            raise MultiFileRowError("row.entry.fn must be a non-empty string")
        if entry_path not in changed_functions:
            raise MultiFileRowError(
                f"row.entry.path is not in row.changed_functions: {entry_path}"
            )
        if entry_fn not in changed_functions[entry_path]:
            raise MultiFileRowError(
                f"row.entry.fn is not listed for {entry_path}: {entry_fn}"
            )
        if sum(len(functions) for functions in changed_functions.values()) != 1:
            raise MultiFileRowError(
                "row.entry is allowed only when exactly one function is changed"
            )

    result = dict(row)
    result["files"] = files
    result["oracle"] = {
        **dict(oracle_value),
        "selectors": selectors,
        "test_files": test_files,
    }
    result["changed_functions"] = changed_functions
    if has_entry:
        result["entry"] = {
            **dict(entry_value),
            "path": _validated_path(entry_value.get("path"), field="row.entry.path"),
            "fn": entry_value["fn"],
        }
    return result


def normalize_multifile_row(row: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize external path spellings, then validate the resulting row."""
    if not isinstance(row, Mapping):
        raise MultiFileRowError("multi_file row must be a mapping")

    result = dict(row)
    raw_files = row.get("files")
    if not isinstance(raw_files, Mapping):
        raise MultiFileRowError("row.files must be a mapping of repository paths")
    normalized_files: dict[str, object] = {}
    for raw_path, raw_payload in raw_files.items():
        path = normalize_repo_path(raw_path, field="row.files path")
        if path in normalized_files:
            raise MultiFileRowError(f"row.files has duplicate normalized path: {path}")
        if not isinstance(raw_payload, Mapping):
            raise MultiFileRowError(f"row.files[{path}] must be a mapping")
        payload = dict(raw_payload)
        if "parent_src" not in payload or "fixed_src" not in payload:
            raise MultiFileRowError(
                f"row.files[{path}] must contain complete parent_src and fixed_src"
            )
        payload["parent_src"] = _complete_source(
            payload["parent_src"], field=f"row.files[{path}].parent_src"
        )
        payload["fixed_src"] = _complete_source(
            payload["fixed_src"], field=f"row.files[{path}].fixed_src"
        )
        normalized_files[path] = payload
    result["files"] = dict(sorted(normalized_files.items()))

    raw_oracle = row.get("oracle")
    if not isinstance(raw_oracle, Mapping):
        raise MultiFileRowError("row.oracle is required and must be a mapping")
    oracle = dict(raw_oracle)
    oracle["selectors"] = _validate_selectors(
        oracle.get("selectors"), field="row.oracle.selectors"
    )
    raw_test_files = oracle.get("test_files")
    if not isinstance(raw_test_files, Mapping):
        raise MultiFileRowError("row.oracle.test_files must be a mapping")
    normalized_test_files: dict[str, object] = {}
    for raw_path, raw_source in raw_test_files.items():
        path = normalize_repo_path(raw_path, field="row.oracle.test_files path")
        if path in normalized_test_files:
            raise MultiFileRowError(
                f"row.oracle.test_files has duplicate normalized path: {path}"
            )
        normalized_test_files[path] = _complete_source(
            raw_source,
            field=f"row.oracle.test_files[{path}]",
        )
    oracle["test_files"] = dict(sorted(normalized_test_files.items()))
    result["oracle"] = oracle

    raw_changed = row.get("changed_functions")
    if not isinstance(raw_changed, Mapping):
        raise MultiFileRowError("row.changed_functions is required and must be a mapping")
    normalized_changed: dict[str, object] = {}
    for raw_path, raw_functions in raw_changed.items():
        path = normalize_repo_path(raw_path, field="row.changed_functions path")
        if path in normalized_changed:
            raise MultiFileRowError(
                f"row.changed_functions has duplicate normalized path: {path}"
            )
        if not isinstance(raw_functions, (list, tuple)):
            raise MultiFileRowError(f"row.changed_functions[{path}] must be a list")
        normalized_changed[path] = list(raw_functions)
    result["changed_functions"] = dict(sorted(normalized_changed.items()))

    if "entry" in row:
        raw_entry = row["entry"]
        if not isinstance(raw_entry, Mapping):
            raise MultiFileRowError("row.entry must be a mapping when present")
        entry = dict(raw_entry)
        entry["path"] = normalize_repo_path(
            entry.get("path"), field="row.entry.path"
        )
        result["entry"] = entry

    return validate_multifile_row(result)


# Short aliases make the boundary convenient for callers that already use the
# generic row terminology, while the explicit names remain the public contract.
normalize_row = normalize_multifile_row
validate_row = validate_multifile_row


__all__ = [
    "MultiFileRowError",
    "RowValidationError",
    "normalize_multifile_row",
    "normalize_repo_path",
    "normalize_row",
    "validate_multifile_row",
    "validate_row",
]
