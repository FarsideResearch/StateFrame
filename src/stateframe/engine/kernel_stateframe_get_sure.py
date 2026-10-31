"""Opt-in clause certainty for the model-facing StateFrame.

This module deliberately owns the get-sure vocabulary and all of its rendering.
The host may retain richer internal check classifications, but the provider
request gets only the three public verdicts and the invalid-check escape hatch.
The default path returns its inputs unchanged and does not inspect a workspace.
"""
from __future__ import annotations

import copy
import ast
import dataclasses
import difflib
import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import kernel_parse_cache as _parse_cache


MODE_ENV = "STATEFRAME_GET_SURE"
SCHEMA = "stateframe-get-sure/1"
BLOCK_HEADER = "[Get-sure state]"
GET_SURE_BLOCK_HEADER = BLOCK_HEADER
CLAUSE_VERDICTS = ("VIOLATED", "SATISFIED", "UNPROBED")
INVALID_VERDICT = "INVALID CHECK, IGNORE"
APPLIED_PREFIX = "APPLIED"
FORBIDDEN_LABELS = ("NOT_ENTAILED", "SHADOW", "ADVISORY", "tier", "admission")
REPAIR_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "file_delete", "file_move", "replace_module", "write", "replace",
    "run_tests", "test", "pytest", "predict",
})
_EDIT_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "file_delete", "file_move", "replace_module", "write", "replace",
})
_PASS = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "OK", "SATISFIED", "RESOLVED"})
_FAIL = frozenset({
    "FAIL", "FAILED", "RED", "ERROR", "COUNTEREXAMPLE", "UNSATISFIED", "VIOLATED",
    "PRODUCT_RED", "REGRESSION",
})
_INVALID = frozenset({"INVALID", "INVALID_CHECK", "NOT_APPLICABLE", "UNBOUND_CHECK"})


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower() or "off"
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except (AttributeError, TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return dict(result)
    return {}


def _first(mapping: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", (), [], {}):
            return value
    return None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "content", "detail", "message", "description", "value"):
            result = _text(value.get(key))
            if result:
                return result
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(item for item in (_text(item) for item in value) if item)
    return str(value).strip()


def _ticket_text(ticket: object) -> str:
    if isinstance(ticket, str):
        return ticket.strip()
    if isinstance(ticket, Mapping):
        for key in (
            "ticket", "task", "brief", "task_brief", "description", "request",
            "symptom", "requirements", "source",
        ):
            result = _ticket_text(ticket.get(key))
            if result:
                return result
        return ""
    if isinstance(ticket, Sequence) and not isinstance(ticket, (str, bytes, bytearray)):
        return "\n".join(item for item in (_ticket_text(item) for item in ticket) if item)
    return _text(ticket)


def _strip_clause_prefix(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*•]|\d+[.)]|\[[A-Za-z0-9_.-]+\])\s*", "", line)
    return line.strip()


def parse_ticket(ticket: object) -> tuple[dict[str, object], ...]:
    """Parse the ticket once into stable, numbered clauses."""
    text = _ticket_text(ticket)
    if os.environ.get("STATEFRAME_POST_GREEN_WORK_MENU", "off").strip().lower() == "on":
        # Rule 4: one clause per logical sentence/list item, never per soft-wrapped line
        # (r42 ldev_10: "[C5] ... A valid service" / "[C6] result includes approved=False.").
        from kernel_post_green_menu import wrap_invariant_clauses
        lines = [_strip_clause_prefix(item) for item in wrap_invariant_clauses(text)
                 if _strip_clause_prefix(item)]
    else:
        lines = [_strip_clause_prefix(line) for line in text.splitlines()
                 if _strip_clause_prefix(line)]
    if len(lines) <= 1:
        raw = lines[0] if lines else text
        pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", raw) if piece.strip()]
        lines = pieces or ([raw] if raw else [])
    clauses = []
    for index, clause in enumerate(lines, 1):
        clauses.append({"id": f"C{index}", "number": index, "text": clause})
    if not clauses:
        clauses.append({
            "id": "C1", "number": 1,
            "text": "The requested behavior is correct and verified.",
        })
    return tuple(clauses)


def extract_clauses(ticket: object) -> tuple[dict[str, object], ...]:
    """Compatibility name for callers that use the parser as a host seam."""
    return parse_ticket(ticket)


def _status(row: Mapping[str, object]) -> str:
    for key in (
        "verdict", "result_class", "status", "result", "outcome", "display_status",
        "state", "check_status",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper().replace("-", "_").replace(" ", "_")
    nested = row.get("receipt")
    if isinstance(nested, Mapping):
        return _status(nested)
    exit_code = row.get("exit_code", row.get("exit"))
    if type(exit_code) is int:
        return "PASS" if exit_code == 0 else "FAIL"
    return ""


def _invalid(row: Mapping[str, object]) -> tuple[bool, str]:
    for key in ("invalid_check", "check_invalid", "valid", "check_valid", "binding_valid"):
        if key in row and row[key] is False:
            return True, _text(_first(row, "reason", "detail", "message")) or "host marked the check invalid"
    value = _status(row)
    if value in _INVALID or "INVALID" in value or "NOT_APPLICABLE" in value:
        return True, _text(_first(row, "reason", "detail", "message")) or "host marked the check invalid"
    return False, ""


def _revision(row: Mapping[str, object], current: object = None) -> str | None:
    value = _first(
        row, "revision", "product_revision", "artifact_revision", "after_revision",
        "current_revision", "source_revision",
    )
    if value is None and isinstance(row.get("receipt"), Mapping):
        value = _revision(row["receipt"], None)
    if value is None:
        value = current
    return str(value) if value not in (None, "") else None


def _receipt_id(row: Mapping[str, object]) -> str | None:
    value = _first(row, "receipt_id", "event_id", "check_receipt_id", "probe_id")
    if value is None and isinstance(row.get("receipt"), Mapping):
        value = _receipt_id(row["receipt"])
    return str(value) if value not in (None, "") else None


def _target(row: Mapping[str, object]) -> str:
    file_name = _text(_first(row, "target_file", "target_path", "path", "file"))
    symbol = _text(_first(row, "target_symbol", "symbol", "function"))
    if file_name and symbol:
        return f"{file_name}:{symbol}"
    value = _first(
        row, "target", "target_symbol", "target_file", "target_path", "path",
        "file", "symbol", "binding",
    )
    if isinstance(value, Mapping):
        file_name = _text(_first(value, "file", "path", "target_file", "target_path"))
        symbol = _text(_first(value, "symbol", "target_symbol", "function"))
        if file_name and symbol:
            return f"{file_name}:{symbol}"
        value = file_name or symbol
    return _text(value) or "host check"


def _observed(row: Mapping[str, object]) -> str:
    return _text(_first(row, "observed", "actual", "actual_value", "got", "detail", "message"))


def _expected(row: Mapping[str, object]) -> str:
    return _text(_first(row, "expected", "expectation", "expected_value", "want", "requirement"))


def _clause_hint(row: Mapping[str, object]) -> str:
    return _text(_first(row, "clause", "exact_clause", "ticket_clause", "requirement", "description"))


def _clause_id(row: Mapping[str, object]) -> str:
    value = _first(row, "clause_id", "requirement_id", "ticket_id", "clause_ref")
    return str(value).strip() if value not in (None, "") else ""


def _tokens(value: object) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]{3,}", _text(value).lower())}


def _matches(row: Mapping[str, object], clause: Mapping[str, object], count: int) -> bool:
    ident = _clause_id(row)
    if ident:
        normalized = ident.upper().replace("CLAUSE-", "C").replace("CLAUSE_", "C")
        if normalized == str(clause.get("id", "")).upper():
            return True
        if normalized.isdigit() and normalized == str(clause.get("number")):
            return True
        return False
    if count == 1:
        return True
    hint = _tokens(_clause_hint(row))
    clause_tokens = _tokens(clause.get("text"))
    return bool(hint and clause_tokens and len(hint & clause_tokens) >= max(1, min(2, len(hint))))


def _evidence_rows(evidence: object) -> list[dict[str, object]]:
    if evidence is None:
        return []
    if dataclasses.is_dataclass(evidence):
        evidence = _as_dict(evidence)
    if isinstance(evidence, Mapping):
        for key in ("rows", "checks", "receipts", "counterexamples", "assessments", "items", "results"):
            if key in evidence and isinstance(evidence[key], (Mapping, Sequence)) and not isinstance(evidence[key], (str, bytes, bytearray)):
                nested = _evidence_rows(evidence[key])
                if nested:
                    return nested
        return [dict(evidence)]
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        result = []
        for item in evidence:
            if isinstance(item, Mapping) or dataclasses.is_dataclass(item):
                row = _as_dict(item)
                if row:
                    result.append(row)
        return result
    return []


def _classify(row: Mapping[str, object], current_revision: object) -> str:
    invalid, _ = _invalid(row)
    if invalid:
        return INVALID_VERDICT
    value = _status(row)
    if value in _FAIL or any(marker in value for marker in ("FAIL", "ERROR", "REGRESSION", "VIOLAT")):
        return "VIOLATED"
    if value in _PASS:
        # A green fact is only certain when it has both host identity and a
        # revision.  Do not silently turn an unlocated claim into certainty.
        return "SATISFIED" if _receipt_id(row) and _revision(row, current_revision) else "UNPROBED"
    observed = _observed(row)
    expected = _expected(row)
    if observed and expected and observed != expected:
        return "VIOLATED"
    if observed and expected and observed == expected:
        return "SATISFIED" if _receipt_id(row) and _revision(row, current_revision) else "UNPROBED"
    return "UNPROBED"


def _witness(row: Mapping[str, object], verdict: str) -> dict[str, str]:
    observed = _observed(row) or ("host reported a failed check" if verdict == "VIOLATED" else "")
    expected = _expected(row) or ("the ticket clause is satisfied" if verdict == "VIOLATED" else "")
    return {"observed": observed, "expected": expected, "target": _target(row)}


@dataclasses.dataclass(frozen=True)
class ClauseTable(Mapping[str, object]):
    ticket: tuple[dict[str, object], ...]
    rows: tuple[dict[str, object], ...]
    revision: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "revision": self.revision,
            "ticket": [copy.deepcopy(item) for item in self.ticket],
            "rows": [copy.deepcopy(item) for item in self.rows],
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(("schema", "revision", "ticket", "rows"))

    def __len__(self) -> int:
        return 4


def build_table(ticket: object, evidence: object = None, *, revision: object = None) -> ClauseTable:
    clauses = parse_ticket(ticket)
    current_revision = str(revision) if revision not in (None, "") else None
    evidence_rows = _evidence_rows(evidence)
    rows: list[dict[str, object]] = []
    for clause in clauses:
        matched = [item for item in evidence_rows if _matches(item, clause, len(clauses))]
        # A failed public result without a clause locator is a workspace fact;
        # in a multi-clause ticket it keeps every clause from being presented
        # as complete until a targeted check settles the mapping.
        if not matched and len(clauses) > 1:
            unmatched_failures = [item for item in evidence_rows if _classify(item, current_revision) == "VIOLATED"]
            if unmatched_failures:
                matched = unmatched_failures
        if not matched:
            rows.append({"id": clause["id"], "clause": clause["text"], "verdict": "UNPROBED"})
            continue
        classified = [_classify(item, current_revision) for item in matched]
        if INVALID_VERDICT in classified and not any(value == "VIOLATED" for value in classified):
            chosen = next(item for item, value in zip(matched, classified) if value == INVALID_VERDICT)
            rows.append({
                "id": clause["id"], "clause": clause["text"], "verdict": INVALID_VERDICT,
                "reason": _invalid(chosen)[1],
            })
            continue
        if "VIOLATED" in classified:
            chosen = next(item for item, value in zip(matched, classified) if value == "VIOLATED")
            rows.append({
                "id": clause["id"], "clause": clause["text"], "verdict": "VIOLATED",
                "witness": _witness(chosen, "VIOLATED"),
                "receipt_id": _receipt_id(chosen),
                "revision": _revision(chosen, current_revision),
            })
            continue
        if all(value == "SATISFIED" for value in classified):
            chosen = matched[-1]
            receipt_id = _receipt_id(chosen)
            row_revision = _revision(chosen, current_revision)
            rows.append({
                "id": clause["id"], "clause": clause["text"], "verdict": "SATISFIED",
                "receipt_id": receipt_id, "revision": row_revision,
            })
            continue
        rows.append({"id": clause["id"], "clause": clause["text"], "verdict": "UNPROBED"})
    return ClauseTable(clauses, tuple(rows), current_revision)


def table_from_dict(value: object) -> ClauseTable | None:
    data = _as_dict(value)
    rows = data.get("rows")
    ticket = data.get("ticket")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return None
    clauses = tuple(_as_dict(item) for item in ticket) if isinstance(ticket, Sequence) else ()
    if not clauses:
        clauses = tuple({"id": row.get("id", f"C{index}"), "number": index, "text": row.get("clause", "")}
                         for index, row in enumerate(rows, 1) if isinstance(row, Mapping))
    normalized_rows = tuple(dict(item) for item in rows if isinstance(item, Mapping))
    return ClauseTable(clauses, normalized_rows, str(data.get("revision")) if data.get("revision") is not None else None)


def violated_rows(table: ClauseTable | Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = table.rows if isinstance(table, ClauseTable) else _as_dict(table).get("rows", ())
    return tuple(row for row in rows if isinstance(row, Mapping) and row.get("verdict") == "VIOLATED")


def unprobed_rows(table: ClauseTable | Mapping[str, object]) -> tuple[dict[str, object], ...]:
    rows = table.rows if isinstance(table, ClauseTable) else _as_dict(table).get("rows", ())
    return tuple(row for row in rows if isinstance(row, Mapping) and row.get("verdict") == "UNPROBED")


def has_violations(table: ClauseTable | Mapping[str, object]) -> bool:
    return bool(violated_rows(table))


_HOST_LABELS = (
    (re.compile(r"\bNOT_ENTAILED\b"), "unproven"),
    (re.compile(r"\bSHADOW\b"), "unclassified"),
    (re.compile(r"\bADVISORY\b"), "nonblocking"),
)
_HOST_KEYS = {"admission": "host_classification", "admission_tier": "host_level",
              "tier": "level", "admitted": "host_accepted"}


def _replacement(value: str) -> str:
    # Case-sensitive host label tokens only: product source text is never rewritten.
    for pattern, replacement in _HOST_LABELS:
        value = pattern.sub(replacement, value)
    for key, replacement in _HOST_KEYS.items():
        value = value.replace(f'"{key}":', f'"{replacement}":')
    return value


def _replacement_key(key: str) -> str:
    return _HOST_KEYS.get(key, _replacement(key))


def sanitize_model_payload(value: object) -> object:
    """Remove host-only labels from every string and key in a provider payload."""
    if isinstance(value, str):
        return _replacement(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            rendered_key = _replacement_key(str(key))
            result[rendered_key] = sanitize_model_payload(item)
        return result
    if isinstance(value, list):
        return [sanitize_model_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_model_payload(item) for item in value)
    return value


def _row_text(row: Mapping[str, object]) -> str:
    verdict = str(row.get("verdict", "UNPROBED"))
    line = f"{row.get('id', 'C?')} | {verdict} | {row.get('clause', '')}"
    if verdict == "VIOLATED":
        witness = row.get("witness", {})
        if isinstance(witness, Mapping):
            line += (f" | observed: {witness.get('observed', '')}; expected: {witness.get('expected', '')};"
                     f" target: {witness.get('target', '')}")
    elif verdict == "SATISFIED":
        line += f" | receipt: {row.get('receipt_id', '')}; revision: {row.get('revision', '')}"
    elif verdict == INVALID_VERDICT:
        line += f" | reason: {row.get('reason', '')}"
    elif verdict == "UNPROBED" and row.get("layers"):   # STATEFRAME_NAMED_LAYER_REPAIR only
        from kernel_named_layer_repair import layer_text
        line += " | " + layer_text(row)
    return _replacement(line)


def _last_decision_text(last_conclusion: object, next_step: object) -> str:
    conclusion = _text(last_conclusion) or "no prior conclusion is recorded"
    step = _text(next_step) or "use the current clause table"
    return _replacement(f"Your last conclusion: {conclusion}; next step: {step}")


_GET_SURE_HINT_DROPPED: list = [0]  # rows dropped at the last render


def drain_get_sure_hint(journal: object) -> bool:
    rows, _GET_SURE_HINT_DROPPED[0] = _GET_SURE_HINT_DROPPED[0], 0
    if not rows or not hasattr(journal, "emit"):
        return False
    journal.emit("get_sure_hint_dropped", producer="kernel_tokenlab_v13", dispatch=False, rows=rows)
    return True


def render_model_block(table: ClauseTable | Mapping[str, object], *, source_path: str = "",
                      source_revision: object = None, last_conclusion: object = None,
                      next_step: object = None, exact_source: object = None,
                      extra: Mapping[str, object] | None = None) -> str:
    parsed = table if isinstance(table, ClauseTable) else table_from_dict(table)
    if parsed is None:
        parsed = build_table("", [], revision=source_revision)
    revision = source_revision if source_revision not in (None, "") else parsed.revision
    lines = [
        BLOCK_HEADER,
        "You already know the task; clause table below is current.",
        "Ticket clauses (parsed once; stable numbering):",
    ]
    for clause in parsed.ticket:
        lines.append(f"[{clause.get('id')}] {clause.get('text', '')}")
    source_label = source_path or "the current source block"
    lines.extend([
        f"EXACT-TEXT CERTAINTY: Source blocks are the exact current bytes at revision {revision or 'unknown'}; "
        f"edit against them directly; do not re-read. Source: {source_label}.",
        _last_decision_text(last_conclusion, next_step),
        "Clause verdict table (current):",
    ])
    for row in parsed.rows:
        lines.append(_row_text(row))
    unknown = unprobed_rows(parsed)
    from kernel_tokenlab_v13 import get_sure_hint_drop_enabled as _hint_drop_on
    if unknown and _hint_drop_on():
        # : the x N 'probe: predict the candidate edit' lines are dropped; the verdict table above
        # already carries every UNPROBED row.  Journaled by the runtime as get_sure_hint_dropped.
        _GET_SURE_HINT_DROPPED[0] = len(unknown)
        unknown = []
    if unknown:
        lines.append("Known unknowns (one settling probe each):")
        for row in unknown:
            lines.append(f"{row.get('id')}: UNPROBED; probe: predict the candidate edit against this clause.")
    if exact_source not in (None, ""):
        lines.append(f"Exact source witness: {_text(exact_source)}")
    if extra:
        lines.append("Host notes: " + _text(sanitize_model_payload(extra)))
    return _replacement("\n".join(lines))


def render_applied_edit(path: object, revision: object, changed_region: object) -> str:
    return _replacement(
        f"{APPLIED_PREFIX} at revision {revision}; changed region ({path}):\n{_text(changed_region)}"
    )


def done_refusal(table: ClauseTable | Mapping[str, object]) -> str | None:
    rows = violated_rows(table)
    if not rows:
        return None
    ids = ", ".join(str(row.get("id")) for row in rows)
    return f"DONE REFUSED: violated clauses remain open ({ids}); repair or run predict; the clause table is current."


def menu_with_repair_tools(selected: Sequence[dict], full_tools: Sequence[dict],
                           table: ClauseTable | Mapping[str, object]) -> list[dict]:
    result = [dict(item) for item in selected]
    available = {str(_as_dict(item.get("function")).get("name")) for item in full_tools
                 if isinstance(item, Mapping)}
    if has_violations(table):
        result = [item for item in result if _as_dict(item.get("function")).get("name") != "done"]
        present = {str(_as_dict(item.get("function")).get("name")) for item in result}
        for item in full_tools:
            fn = _as_dict(item.get("function")) if isinstance(item, Mapping) else {}
            name = str(fn.get("name", ""))
            if name in REPAIR_TOOLS and name in available and name not in present:
                result.append(item)
                present.add(name)
    elif unprobed_rows(table):
        present = {str(_as_dict(item.get("function")).get("name")) for item in result}
        for item in full_tools:
            fn = _as_dict(item.get("function")) if isinstance(item, Mapping) else {}
            if fn.get("name") == "predict" and "predict" not in present:
                result.append(item)
                break
    return result


def project_menu(selected: Sequence[dict], full_tools: Sequence[dict], table: ClauseTable | Mapping[str, object]) -> list[dict]:
    return menu_with_repair_tools(selected, full_tools, table)


def predict_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "predict",
            "description": "Run a candidate edit in a scratch copy and report each ticket clause.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "function_source": {"type": "string"},
                    "symbol": {"type": "string"},
                    "selectors": {"type": "array", "items": {"type": "string"}},
                    "why": {"type": "string"},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def tool_surface(tools: object) -> object:
    if not enabled():
        return tools
    if not isinstance(tools, list):
        return tools
    if any(_as_dict(item.get("function")).get("name") == "predict"
           for item in tools if isinstance(item, Mapping)):
        return tools
    return [*tools, predict_tool()]


def _candidate_path(root: Path, value: object) -> Path:
    raw = str(value or "").replace("\\", "/").lstrip("/")
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("candidate path escapes the scratch workspace")
    return path


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {".git", ".pytest_cache", ".mypy_cache", "__pycache__", ".venv", "node_modules"}
    return {name for name in names if name in ignored or name.startswith(".mutation")}


def _workspace_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in {".git", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        try:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        except OSError:
            continue
    return digest.hexdigest()


def _candidate_result_rows(table: ClauseTable, raw: object, public_status: str) -> list[dict[str, object]]:
    rows = _evidence_rows(raw)
    if not rows:
        rows = [{"clause_id": clause["id"], "status": public_status,
                 "receipt_id": "predict-public", "revision": table.revision or "scratch"}
                for clause in table.ticket]
    return rows


def _replace_definition(source: str, replacement: str, symbol: str = "") -> str:
    """Replace one top-level definition in scratch when a function candidate is supplied."""
    old_tree = _parse_cache.parse(source)  # shared read-only tree (L112)
    new_tree = _parse_cache.parse(replacement)
    new_nodes = [node for node in new_tree.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not new_nodes:
        raise ValueError("function_source must contain a top-level definition")
    new_node = next((node for node in new_nodes if not symbol or node.name == symbol), new_nodes[0])
    old_node = next((node for node in old_tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and node.name == new_node.name), None)
    if old_node is None:
        raise ValueError("candidate definition is not present in the scratch source")
    lines = source.splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement_lines and not replacement_lines[-1].endswith(("\n", "\r")):
        replacement_lines[-1] += "\n"
    return "".join(lines[:old_node.lineno - 1] + replacement_lines + lines[old_node.end_lineno:])


def predict_candidate(root: str | Path, candidate: Mapping[str, object], table: ClauseTable | Mapping[str, object], *,
                      public_runner: Callable[[Path, Sequence[str]], object] | None = None,
                      clause_runner: Callable[[Path, Mapping[str, object], Sequence[dict[str, object]]], object] | None = None,
                      selectors: Sequence[str] = ()) -> dict[str, object]:
    """Apply a candidate only in a temporary sibling copy and return typed results."""
    root_path = Path(root).resolve()
    parsed = table if isinstance(table, ClauseTable) else table_from_dict(table)
    if parsed is None:
        parsed = build_table("", [], revision="scratch")
    before = _workspace_fingerprint(root_path)
    public: dict[str, object] = {"status": "NOT_RUN"}
    clause_raw: object = None
    apply_error = ""
    with tempfile.TemporaryDirectory(prefix=".stateframe-get-sure-", dir=str(root_path.parent)) as scratch_name:
        scratch = Path(scratch_name)
        try:
            shutil.copytree(root_path, scratch, dirs_exist_ok=True, ignore=_copy_ignore)
            path = _candidate_path(scratch, candidate.get("path", "")) if candidate.get("path") else None
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                old = candidate.get("old_str")
                new = candidate.get("new_str")
                if isinstance(old, str) and isinstance(new, str):
                    current = path.read_text(encoding="utf-8")
                    if current.count(old) != 1:
                        raise ValueError("candidate old_str must match exactly once")
                    path.write_text(current.replace(old, new, 1), encoding="utf-8")
                elif isinstance(candidate.get("content"), str) or isinstance(candidate.get("function_source"), str):
                    source = str(candidate.get("content", candidate.get("function_source")))
                    if isinstance(candidate.get("function_source"), str):
                        source = _replace_definition(
                            path.read_text(encoding="utf-8"), source,
                            str(candidate.get("symbol") or ""))
                    path.write_text(source, encoding="utf-8")
                else:
                    raise ValueError("candidate needs old_str/new_str or source")
            if callable(clause_runner):
                clause_raw = clause_runner(scratch, candidate, parsed.ticket)
            if callable(public_runner):
                try:
                    public_raw = public_runner(scratch, tuple(selectors))
                except TypeError:
                    public_raw = public_runner(scratch)  # type: ignore[misc]
                public = _as_dict(public_raw) or {"status": _text(public_raw)}
            else:
                public = {"status": "NOT_RUN"}
        except (OSError, UnicodeError, TypeError, ValueError) as error:
            apply_error = str(error)
            public = {"status": "FAIL", "detail": apply_error}
    public_status = _status(public) or str(public.get("status", "NOT_RUN")).upper()
    result_rows = _candidate_result_rows(parsed, clause_raw, public_status)
    predicted = build_table(
        [clause.get("text", "") for clause in parsed.ticket],
        result_rows,
        revision=parsed.revision or "scratch",
    )
    after = _workspace_fingerprint(root_path)
    return sanitize_model_payload({
        "schema": SCHEMA,
        "status": "PREDICTED",
        "prediction_status": public_status,
        "workspace_unchanged": before == after,
        "applied_to_scratch": not bool(apply_error),
        "candidate": {key: value for key, value in candidate.items() if key not in {"new_str", "content", "function_source"}},
        "public_tests": {
            "status": public_status,
            "exit_code": public.get("exit_code"),
            "detail": _text(_first(public, "detail", "observation", "message")),
        },
        "clauses": list(predicted.rows),
    })


def _runtime_ticket(runtime: object, frame: Mapping[str, object] | None) -> object:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    if isinstance(row, Mapping):
        for key in ("_bridge_ordinary_brief", "brief", "ticket", "task", "description", "symptom"):
            if row.get(key):
                return row[key]
    if isinstance(frame, Mapping):
        for key in ("ticket", "task_brief", "requirements", "brief"):
            if frame.get(key):
                return frame[key]
        ledger = frame.get("requirements_ledger")
        if ledger:
            return ledger
    return ""


def _runtime_revision(runtime: object, frame: Mapping[str, object] | None = None) -> str | None:
    if isinstance(frame, Mapping):
        revisions = frame.get("artifact_revisions")
        if isinstance(revisions, Mapping) and revisions:
            return str(next(iter(revisions.values())))
    try:
        target = getattr(runtime, "target", None)
        if target is not None and Path(target).is_file():
            return "sha256:" + hashlib.sha256(Path(target).read_bytes()).hexdigest()
    except OSError:
        pass
    return None


def _runtime_evidence(runtime: object, frame: Mapping[str, object] | None) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    owner = getattr(runtime, "owner", None)
    for value in (
        getattr(owner, "_receipts", ()), getattr(runtime, "test_receipts", ()),
        getattr(runtime, "probe_receipts", ()), getattr(runtime, "host_check_receipts", ()),
        getattr(runtime, "host_repair_check_receipts", ()),
    ):
        evidence.extend(_evidence_rows(value))
    if isinstance(frame, Mapping):
        for key in (
            "test_receipts", "probe_receipts", "receipts", "counterexamples",
            "host_closure", "requirements_ledger", "latest_receipts", "verification",
            "last_observation", "public_tests",
        ):
            evidence.extend(_evidence_rows(frame.get(key)))
    return evidence


def authority_on(runtime: object) -> bool:
    return getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"


def _authority_evidence(runtime: object, frame: Mapping[str, object] | None):
    """Clause evidence projected from the ONE readiness snapshot (None when unavailable)."""
    try:
        from kernel_readiness_authority import runtime_snapshot
        decision = runtime_snapshot(runtime, frame)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None, None
    if decision is None:
        return None, None
    rows: list[dict[str, object]] = []
    actionable = [*list(decision.blockers), *list(decision.repairable_advisories)]
    actionable_ids = {str(item.get("check_id") or item.get("obligation_id") or "") for item in actionable}
    for item in actionable:
        row = dict(item)
        row["verdict"] = "VIOLATED"
        rows.append(row)
    for item in decision.advisory:
        ident = str(item.get("check_id") or item.get("obligation_id") or "")
        if ident in actionable_ids:
            continue
        row = dict(item)
        row["invalid_check"] = True
        row["valid"] = False
        row["reason"] = ("host classified this check as not a valid requirement probe ("
                         + str(item.get("demotion_reason") or "unverified") + ")")
        rows.append(row)
    if not decision.green:
        rows.append({"verdict": "FAIL", "observed": "public checks " + str(decision.public_check_status),
                     "expected": "public checks PASS", "target": "public tests",
                     "revision": decision.revision})
    return rows, decision.revision


def runtime_table(runtime: object, frame: Mapping[str, object] | None = None, *, revision: object = None) -> ClauseTable:
    table = _runtime_table_base(runtime, frame, revision=revision)
    if os.environ.get("STATEFRAME_NAMED_LAYER_REPAIR", "off").strip().lower() == "on":
        # autopsy-ldev01-cross-release #2: a clause naming a layer is never "INVALID CHECK, IGNORE"; it is
        # UNPROBED and bound to its named-layer row (kernel_named_layer_coverage).  Off: unchanged.
        provider = getattr(runtime, "layer_coverage_report", None)
        if callable(provider):
            try:
                from kernel_named_layer_repair import bind_invalid_named_clauses, emit as named_layer_emit
                bound = bind_invalid_named_clauses(table, provider())
                if bound is not table:
                    named_layer_emit(runtime, "get_sure_bind", stage="acted")
                table = bound
            except (ImportError, OSError, TypeError, ValueError):
                pass
    return table


def _runtime_table_base(runtime: object, frame: Mapping[str, object] | None = None, *,
                        revision: object = None) -> ClauseTable:
    supplied_revision = revision if revision not in (None, "") else _runtime_revision(runtime, frame)
    if authority_on(runtime):
        evidence, decision_revision = _authority_evidence(runtime, frame)
        if evidence is not None:
            return build_table(_runtime_ticket(runtime, frame), evidence,
                               revision=supplied_revision or decision_revision)
    return build_table(
        _runtime_ticket(runtime, frame),
        _runtime_evidence(runtime, frame),
        revision=supplied_revision,
    )


def _changed_region(runtime: object, receipt: Mapping[str, object]) -> str:
    existing = _first(receipt, "changed_region", "diff", "patch", "detail")
    if existing:
        return _text(existing)
    action = getattr(runtime, "_edit_before", None)
    path = receipt.get("target_path")
    if isinstance(action, tuple) and len(action) == 2 and isinstance(path, str):
        before = action[1] if isinstance(action[1], str) else ""
        try:
            root = Path(getattr(getattr(runtime, "owner", None), "root", Path.cwd()))
            after_path = _candidate_path(root, path)
            after = after_path.read_text(encoding="utf-8") if after_path.is_file() else ""
            return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), n=1, lineterm=""))
        except (OSError, UnicodeError, ValueError):
            pass
    return "edit result recorded by host"


def after_applied_edit(runtime: object, receipt: Mapping[str, object]) -> str | None:
    """Record a revision-bound APPLIED receipt after the normal host hooks run."""
    if not enabled() or str(receipt.get("result_class", "")).upper() != "APPLIED":
        return None
    revision = _runtime_revision(runtime)
    marker = render_applied_edit(receipt.get("target_path", "source"), revision or "unknown", _changed_region(runtime, receipt))
    setattr(runtime, "_get_sure_last_applied", marker)
    setattr(runtime, "_get_sure_last_applied_revision", revision)
    return marker


def configure_check_runner(runtime: object, runner: Callable[..., object]) -> None:
    if not callable(runner):
        raise TypeError("get-sure check runner must be callable")
    setattr(runtime, "get_sure_check_runner", runner)


def run_relevant_checks(runtime: object, receipt: Mapping[str, object]) -> object:
    """Use an explicitly installed host runner after an edit, if present."""
    runner = getattr(runtime, "get_sure_check_runner", None)
    if callable(runner):
        table = runtime_table(runtime)
        try:
            return runner(table, receipt)
        except TypeError:
            return runner(receipt)
    # The ordinary runner already has one host-owned implementation.  When
    # get-sure is the only opt-in rail, force that same runner once after an
    # applied product edit; the temporary mode change is local to this call
    # and leaves the legacy switch and its request bytes untouched.
    auto_runner = getattr(runtime, "auto_verify_runner", None)
    auto_verify = getattr(runtime, "_run_auto_verify", None)
    if (callable(auto_runner) and callable(auto_verify)
            and getattr(runtime, "auto_verify_mode", "off") != "on"):
        previous = getattr(runtime, "auto_verify_mode", "off")
        try:
            runtime.auto_verify_mode = "on"
            return auto_verify(receipt)
        finally:
            runtime.auto_verify_mode = previous
    return None


def predict_runtime(runtime: object, arguments: Mapping[str, object]) -> dict[str, object]:
    table = runtime_table(runtime)
    root = Path(getattr(runtime, "root", None) or getattr(getattr(runtime, "owner", None), "root", Path.cwd()))
    runner = getattr(runtime, "get_sure_public_runner", None)
    clause_runner = getattr(runtime, "get_sure_clause_runner", None)
    selectors = arguments.get("selectors", getattr(getattr(runtime, "owner", None), "f2p_ids", ()))
    result = predict_candidate(
        root, arguments, table, public_runner=runner if callable(runner) else None,
        clause_runner=clause_runner if callable(clause_runner) else None,
        selectors=selectors if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)) else (),
    )
    journal = getattr(runtime, "journal", None)
    if journal is not None and callable(getattr(journal, "emit", None)):
        journal.emit(
            "get_sure_prediction", producer="stateframe_get_sure",
            status=result.get("prediction_status"), clause_count=len(result.get("clauses", ())),
            workspace_unchanged=result.get("workspace_unchanged"), dispatch=False,
        )
    return result


def record_last_decision(runtime: object) -> tuple[str, str]:
    store = getattr(runtime, "decision_continuity", None)
    cache = getattr(store, "cache", None) if store is not None else None
    if not isinstance(cache, Mapping):
        cache = _as_dict(getattr(store, "snapshot", lambda: {})()) if store is not None else {}
    hypothesis = cache.get("hypothesis") if isinstance(cache, Mapping) else None
    next_action = cache.get("next_discriminating_action") if isinstance(cache, Mapping) else None
    return _text(hypothesis), _text(next_action)


def runtime_model_block(runtime: object, frame: Mapping[str, object] | None = None,
                        *, path: str = "", revision: object = None) -> str:
    table = runtime_table(runtime, frame, revision=revision)
    last, next_step = record_last_decision(runtime)
    return render_model_block(table, source_path=path, source_revision=revision,
                             last_conclusion=last, next_step=next_step,
                             exact_source=getattr(runtime, "_get_sure_last_applied", None))


def scrub_request(messages: Sequence[dict], tools: object) -> tuple[list[dict], object]:
    """Final boundary scrub; enabled callers use this after all other renderers."""
    if not enabled():
        return list(messages), tools
    clean_messages = sanitize_model_payload(list(messages))
    clean_tools = tools  # r39: tool schemas are a wire contract; never rewritten
    return list(clean_messages) if isinstance(clean_messages, list) else list(messages), clean_tools


__all__ = [
    "APPLIED_PREFIX", "BLOCK_HEADER", "CLAUSE_VERDICTS", "ClauseTable", "FORBIDDEN_LABELS",
    "GET_SURE_BLOCK_HEADER", "INVALID_VERDICT", "MODE_ENV", "REPAIR_TOOLS", "SCHEMA",
    "after_applied_edit", "build_table", "configure_check_runner", "done_refusal", "enabled",
    "extract_clauses", "has_violations", "menu_with_repair_tools", "mode", "parse_ticket",
    "predict_candidate", "predict_runtime", "predict_tool", "project_menu", "record_last_decision",
    "render_applied_edit", "render_model_block", "runtime_model_block", "runtime_table",
    "run_relevant_checks", "sanitize_model_payload", "scrub_request", "table_from_dict",
    "tool_surface", "unprobed_rows", "violated_rows",
]
