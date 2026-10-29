"""FLP1 partial premature-done frame for the v46 completion boundary.

This module computes host facts only.  ``kernel_completion_owner`` remains the
single renderer and menu projector; the Governor consumes this frame so a
challenge cannot acquire a second text or tool owner.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_FL_PREMATURE_RULE"
SCHEMA = "stateframe-flp1-premature-rule/1"
MENU = ("edit", "str_replace", "read", "done")
PROGRESS_TOOLS = frozenset(MENU[:3])
_TRUTHY = frozenset({"1", "on", "true", "yes"})
_GREEN = frozenset({"PASS", "PASSED", "SATISFIED", "GREEN", "SUCCESS", "OK"})
_CLOSING = frozenset({"CLOSED", "COMPLETE", "COMPLETED", "PASS", "PASSED", "SATISFIED", "VALIDATED"})
_ROLE_RE = re.compile(r"\b(model|service|api|cli|validation|worker|repository|handler|route)\b", re.IGNORECASE)


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict on/off mode for the FLP1 confirmation arm."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_fl_premature_rule_mode", None)
    if configured is not None:
        return str(configured).strip().lower() in _TRUTHY
    return mode() == "on"


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            result = converter()
        except (AttributeError, TypeError, ValueError):
            return None
        return result if isinstance(result, Mapping) else None
    return None


def _field(value: object, *names: str, default: object = None) -> object:
    row = _mapping(value)
    if row is not None:
        for name in names:
            if name in row:
                return row[name]
        return default
    for name in names:
        result = getattr(value, name, default)
        if result is not default:
            return result
    return default


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return ()


def _nonempty(value: object) -> bool:
    if value is None or value is False or value == "":
        return False
    if isinstance(value, (Sequence, Mapping)) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    return True


def _row_id(row: Mapping[str, object], index: int) -> str:
    for name in ("obligation_id", "clause_id", "requirement_id", "check_id", "id"):
        value = row.get(name)
        if value not in (None, "") and str(value).strip():
            return str(value)
    return f"clause-{index + 1}"


def _clause_text(row: Mapping[str, object], identifier: str) -> str:
    for name in ("clause", "clause_text", "requirement", "text"):
        value = row.get(name)
        if value is not None and str(value) != "":
            return str(value)
    return identifier


def _receipt_closes_clause(value: object, identifier: str) -> bool:
    row = _mapping(value)
    if row is None:
        return False
    if row.get("clause_closed") is True or row.get("closes_clause") is True:
        return True
    receipt_id = str(row.get("clause_id", row.get("obligation_id", row.get("requirement_id", ""))) or "")
    if receipt_id and receipt_id != identifier:
        return False
    kind = str(row.get("kind", row.get("type", row.get("scope", ""))) or "").strip().lower()
    result = str(row.get("result_class", row.get("result", row.get("status", row.get("state", "")))) or "").strip().upper()
    return bool(receipt_id and result in _CLOSING) or (
        "claus" in kind and result in _CLOSING
    )


def _has_clause_closing_receipt(row: Mapping[str, object], identifier: str) -> bool:
    if row.get("clause_closed") is True:
        return True
    # These names are clause-level by contract.  Aggregate public test
    # receipts are intentionally not inferred from verdict/status or green.
    for name in (
        "clause_closing_receipt", "closing_receipt", "clause_receipt",
        "closure_receipt", "closing_evidence", "clause_evidence",
    ):
        value = row.get(name)
        if _receipt_closes_clause(value, identifier) or (
                _nonempty(value) and name != "clause_evidence"):
            return True
    for receipt in _items(row.get("receipts")):
        if _receipt_closes_clause(receipt, identifier):
            return True
    return bool(row.get("closing_receipt_id"))


def has_clause_closing_receipt(value: object, identifier: str | None = None) -> bool:
    """Expose the clause-level closure test to other host-fact consumers."""
    row = _mapping(value)
    if row is None:
        return False
    resolved = identifier or _row_id(row, 0)
    return _has_clause_closing_receipt(row, resolved)


def _assessment_rows(runtime: object) -> tuple[dict[str, object], ...]:
    source = getattr(runtime, "host_closure_assessments", None)
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
        texts = _items(getattr(runtime, "open_clause_texts", ()))
        identifiers = _items(getattr(runtime, "open_requirement_ids", ()))
        receipts = getattr(runtime, "clause_closing_receipts", {})
        if not texts:
            return ()
        fallback: list[dict[str, object]] = []
        for index, text in enumerate(texts):
            identifier = str(identifiers[index]) if index < len(identifiers) else f"clause-{index + 1}"
            row = {"id": identifier, "clause": str(text), "source": {}}
            receipt = receipts.get(identifier) if isinstance(receipts, Mapping) else None
            if not _nonempty(receipt):
                fallback.append(row)
        return tuple(fallback)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(source):
        source_row = _mapping(item)
        if source_row is None:
            continue
        identifier = _row_id(source_row, index)
        if _has_clause_closing_receipt(source_row, identifier):
            continue
        rows.append({
            "id": identifier,
            "clause": _clause_text(source_row, identifier),
            "source": dict(source_row),
        })
    return tuple(rows)


def open_clause_rows(runtime: object) -> tuple[Mapping[str, object], ...]:
    """Return only clauses lacking explicit clause-level closing receipts."""
    return tuple(row for row in _assessment_rows(runtime))


def public_checks_green(runtime: object) -> bool:
    explicit = getattr(runtime, "public_green", None)
    if explicit is None:
        explicit = getattr(runtime, "public_checks_green", None)
    if type(explicit) is bool:
        return explicit
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    if callable(verdict_fn):
        try:
            verdict = verdict_fn()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        state = str(_field(verdict, "state", default="") or "").strip().upper()
        return state in _GREEN
    return False


def revision_unchanged(runtime: object) -> bool:
    explicit = getattr(runtime, "revision_unchanged", None)
    if type(explicit) is bool:
        return explicit
    state = str(getattr(runtime, "revision_state", "") or "").strip().lower()
    if state in {"unchanged", "current", "stable"}:
        return True
    verdict_fn = getattr(runtime, "public_test_verdict", None)
    binding_fn = getattr(runtime, "binding", None)
    if not callable(verdict_fn) or not callable(binding_fn):
        return False
    try:
        verdict = verdict_fn()
        current = binding_fn()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    if _field(verdict, "current_binding", default=None) != current:
        return False
    receipts = _field(verdict, "trusted_receipts", default=())
    values = _items(receipts)
    return bool(values) and all(
        _field(receipt, "binding", default=None) == current
        and _field(receipt, "after_binding", default=None) == current
        and _field(receipt, "exit_code", default=None) == 0
        and _field(receipt, "binding_stable", default=None) is True
        for receipt in values
    )


def _surface_text(value: object) -> tuple[str, str]:
    row = _mapping(value)
    if row is not None:
        role = row.get("role", row.get("surface_role", row.get("name", "")))
        path = row.get("path", row.get("file", row.get("target", "")))
        return str(role or "").strip(), str(path or "").strip().replace("\\", "/")
    text = str(value or "").strip().replace("\\", "/")
    if "::" in text:
        text = text.split("::", 1)[0]
    return "", text


def _role_for(value: object) -> tuple[str, str]:
    role, path = _surface_text(value)
    if role:
        return role, path
    if path:
        stem = Path(path).stem
        return stem or path, path
    return "", ""


def untouched_surface_roles(runtime: object, rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    changed: set[str] = set()
    for name in ("changed_paths", "changed_files", "edited_paths", "_changed_paths"):
        for item in _items(getattr(runtime, name, ())):
            _role, path = _role_for(item)
            if path:
                changed.add(path)
    values: list[str] = []

    def add(value: object) -> None:
        role, path = _role_for(value)
        if not role or path in changed:
            return
        if role not in values:
            values.append(role)

    for name in (
        "untouched_requirement_surfaces", "untouched_surface_roles",
        "host_untouched_surface_roles", "requirement_surface_roles",
    ):
        value = getattr(runtime, name, ())
        if isinstance(value, Mapping):
            for key, item in value.items():
                add(item if _mapping(item) is not None else {"role": item, "path": key})
        else:
            for item in _items(value):
                add(item)
        if values:
            return tuple(values)

    for row in rows:
        source = row.get("source", {})
        for name in ("untouched_surface_roles", "surface_roles", "surface_role", "role", "surfaces", "surface"):
            for item in _items(_field(source, name, default=())):
                add(item)
    if values:
        return tuple(values)

    binding = getattr(runtime, "binding", None)
    try:
        bound = binding() if callable(binding) else binding
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        bound = None
    bound_row = _mapping(bound)
    if bound_row is not None:
        for name in ("product_files", "files", "source_files"):
            for path in _items(bound_row.get(name, ())):
                add(path)
    for row in rows:
        clause = str(row.get("clause", ""))
        for match in _ROLE_RE.findall(clause):
            add(match.lower())
    return tuple(values) or ("not recorded",)


def _available_tool_names(runtime: object, tools: object) -> set[str]:
    surface = tools
    if not isinstance(surface, Sequence) or isinstance(surface, (str, bytes, bytearray)):
        surface = getattr(runtime, "_completion_owner_full_tools", ())
    def names_for(value: object) -> set[str]:
        names: set[str] = set()
        for tool in _items(value):
            if isinstance(tool, str):
                names.add(tool)
                continue
            row = _mapping(tool)
            function = _mapping(row.get("function")) if row is not None else None
            if function is not None and function.get("name"):
                names.add(str(function["name"]))
        return names

    names = names_for(surface)
    if not names.intersection(PROGRESS_TOOLS):
        names = names_for(getattr(runtime, "_completion_owner_full_tools", ()))
    if not names.intersection(PROGRESS_TOOLS):
        names = names_for(getattr(runtime, "available_tools", ()))
    return names


def frame(runtime: object, tools: object = ()) -> dict[str, object] | None:
    """Build the Variant-B frame, or ``None`` when its three-part predicate fails."""
    if not enabled(runtime) or not public_checks_green(runtime) or not revision_unchanged(runtime):
        return None
    rows = open_clause_rows(runtime)
    if not rows:
        return None
    surfaces = untouched_surface_roles(runtime, rows)
    names = _available_tool_names(runtime, tools)
    menu = tuple(tool for tool in MENU if tool in names)
    clauses = tuple(str(row.get("clause", "")) for row in rows)
    lines = tuple(
        [*(f"Open requirement clause (verbatim): {json.dumps(clause, ensure_ascii=False)}" for clause in clauses),
         "Requirement surfaces not yet touched by this revision: " + ", ".join(surfaces)]
    )
    return {
        "schema": SCHEMA,
        "kind": "partial-premature",
        "clause_ids": tuple(str(row.get("id", "")) for row in rows),
        "clause_texts": clauses,
        "clauses": clauses,
        "surface_roles": surfaces,
        "menu": menu,
        "lines": lines,
        "notice": "\n".join(lines),
    }


build_frame = frame


__all__ = [
    "MENU", "MODE_ENV", "PROGRESS_TOOLS", "SCHEMA", "build_frame", "enabled", "frame",
    "has_clause_closing_receipt", "mode", "open_clause_rows", "public_checks_green", "revision_unchanged",
    "untouched_surface_roles",
]
