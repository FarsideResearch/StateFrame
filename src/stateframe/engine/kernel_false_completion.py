"""R-007 weak-public-coverage evidence for accepted completion.

This organ is deliberately advisory unless an existing completion arbiter owns
the challenge.  It measures only the public ticket, declared public test
sources, and host-owned product-file snapshots.  It never reads a test path or
any held-out material.

Switch: ``STATEFRAME_FALSE_COMPLETION=off|on`` (default ``off``).  The
``STATEFRAME_LOW_COVERAGE`` spelling is accepted as a compatibility alias for
local experiments; the runtime exposes one effective switch.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import kernel_named_layer_coverage as _layers


MODE_ENV = "STATEFRAME_FALSE_COMPLETION"
MODE_ALIAS = "STATEFRAME_LOW_COVERAGE"
SCHEMA = "stateframe-false-completion/1"
LOW_COVERAGE = "LOW_COVERAGE"
REASON_CODE = "LOW_COVERAGE_UNVERIFIED"
CHALLENGE_CLASS = "UNVERIFIED"
HEADER = "[False-completion coverage | UNVERIFIED]"
ATTRIBUTION_ENV = "STATEFRAME_COVERAGE_ATTRIBUTION_FIX"
RUNTIME_ENVIRONMENT = {
    "STATEFRAME_PROJECTOR": "rich",
    "STATEFRAME_VIEW": "operational",
    "KERNEL_MODE": "render_selection:ENFORCE",
    "STATEFRAME_F4": "off",
    "STATEFRAME_SOURCE_LABELS": "off",
}

_STATUS_RE = re.compile(r"(?<!\d)([45]\d{2})(?!\d)")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*|[45]\d{2}")
_QUOTED_RE = re.compile(r"\x60([^\x60\n]+)\x60|'([^'\n]+)'|\"([^\"\n]+)\"")
_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "must", "of", "on", "or", "the", "to", "through", "with",
    "this", "that", "when", "where", "value", "values", "new", "option",
})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    raw = source.get(MODE_ENV)
    if raw is None:
        raw = source.get(MODE_ALIAS, "off")
    value = (raw or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def snapshot_workspace(root: Path) -> dict[str, str | None]:
    return _layers.snapshot_workspace(Path(root))


def normalise_snapshot(snapshot: Mapping[str, object]) -> dict[str, str | None]:
    return _layers.normalise_snapshot(snapshot)


def _source_texts(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        direct: list[str] = []
        for key in ("source", "test_source", "public_test_source", "content", "text"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                direct.append(item)
        if direct:
            return tuple(direct)
        values: list[str] = []
        for item in value.values():
            values.extend(_source_texts(item))
        return tuple(values)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values: list[str] = []
        for item in value:
            values.extend(_source_texts(item))
        return tuple(values)
    return ()


def public_assertion_count(public_tests: object) -> int:
    """Count explicit public assertions without executing or resolving paths."""
    count = 0
    for source in _source_texts(public_tests):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, TypeError):
            count += len(re.findall(r"(?m)^\s*assert\b", source))
            count += len(re.findall(r"\b(?:assert|expect)\s*\.", source))
            count += len(re.findall(r"\b(?:assert|expect)\s*\(", source))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                count += 1
                continue
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = function.attr if isinstance(function, ast.Attribute) else (
                function.id if isinstance(function, ast.Name) else "")
            if name.startswith("assert") or name in {"expect", "raises", "raises_regex"}:
                count += 1
    return count


def _significant_tokens(clause: str) -> tuple[str, ...]:
    result: list[str] = []
    for token in _TOKEN_RE.findall(clause.lower()):
        if token in _STOP_WORDS or len(token) < 2:
            continue
        if token not in result:
            result.append(token)
    return tuple(result)


def _is_assertion_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    name = function.attr if isinstance(function, ast.Attribute) else (
        function.id if isinstance(function, ast.Name) else "")
    return name.startswith("assert") or name in {"expect", "raises", "raises_regex"}


def public_assertion_text(public_tests: object) -> str:
    """The text of public ASSERTION expressions only (r44 review F2).

    Coverage evidence comes from ``assert`` statements, assertion-style calls and the body of a
    ``with ...raises(...)`` block.  Comments, docstrings, bare layer mentions, imports and other
    prose never count.  Unparseable sources fall back to assertion lines with comments removed.
    """
    parts: list[str] = []
    for source in _source_texts(public_tests):
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, TypeError):
            for line in source.splitlines():
                code = line.split("#", 1)[0]
                if re.match(r"\s*(?:assert\b|expect\s*[.(])", code):
                    parts.append(code.strip())
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                parts.append(ast.unparse(node))
            elif isinstance(node, (ast.With, ast.AsyncWith)) and any(
                    _is_assertion_call(item.context_expr) for item in node.items):
                parts.append(ast.unparse(node))
            elif isinstance(node, ast.Expr) and _is_assertion_call(node.value):
                parts.append(ast.unparse(node.value))
    return "\n".join(parts)


def _clause_covered(clause: str, assertion_text: str, behavior_witnesses: object = ()) -> bool:
    """Conservative: a clause is covered only by a signal INSIDE an assertion expression
    (the whole clause phrase, a quoted/status contract literal, or >= 2 significant clause
    tokens).  Anything else keeps the clause on the uncovered list."""
    haystack = assertion_text.lower()
    normalized = " ".join(haystack.split())
    phrase = " ".join(clause.lower().split()).rstrip(".")
    if os.environ.get(ATTRIBUTION_ENV, "off").strip().lower() == "on":
        if phrase and phrase in normalized:
            return True
        for witness in behavior_witnesses if isinstance(behavior_witnesses, Sequence) else ():
            if not isinstance(witness, Mapping):
                continue
            if (witness.get("result") == "PASS"
                    and " ".join(str(witness.get("clause", "")).lower().split()).rstrip(".")
                    == phrase
                    and witness.get("partition")
                    and isinstance(witness.get("provenance"), Mapping)):
                return True
        # A field name, status literal, or two-token overlap is not a
        # behavioral discriminator when this attribution guard is enabled.
        return False
    if not haystack.strip():
        return False
    if phrase and phrase in normalized:
        return True
    literals: list[str] = []
    literals.extend(match.group(1) or match.group(2) or match.group(3)
                    for match in _QUOTED_RE.finditer(clause))
    literals.extend(match.group(1) for match in _STATUS_RE.finditer(clause))
    for literal in literals:
        if literal and literal.lower() in haystack:
            return True
    tokens = _significant_tokens(clause)
    if tokens:
        # identifiers count by their parts too: ``reject_invalid_input`` names 3 clause tokens
        split = re.sub(r"_+", " ", haystack)
        present = sum(1 for token in tokens
                      if re.search(rf"\b{re.escape(token)}\b", haystack)
                      or re.search(rf"\b{re.escape(token)}\b", split))
        if present >= 2:
            return True
    return False


def _ticket_clauses(ticket: str) -> tuple[str, ...]:
    return _layers._ticket_clauses(ticket)


def untouched_named_layers(report: Mapping[str, object] | None) -> tuple[dict[str, object], ...]:
    if not isinstance(report, Mapping):
        return ()
    rows = report.get("rows", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return ()
    return tuple(copy.deepcopy(dict(row)) for row in rows
                 if isinstance(row, Mapping)
                 and row.get("confidence") == "confident"
                 and row.get("diff_from_start") is False)


def coverage_report(ticket: object, root: Path, *,
                    start_snapshot: Mapping[str, object] | None = None,
                    current_snapshot: Mapping[str, object] | None = None,
                    public_tests: object = None,
                    behavior_witnesses: object = ()) -> dict[str, object]:
    text = _layers.public_ticket(ticket)
    root = Path(root).resolve()
    start = (snapshot_workspace(root) if start_snapshot is None
             else normalise_snapshot(start_snapshot))
    current = (snapshot_workspace(root) if current_snapshot is None
               else normalise_snapshot(current_snapshot))
    layer_report = _layers.build_report(
        text, root, start_snapshot=start, current_snapshot=current,
        public_tests=public_tests)
    assertion_text = public_assertion_text(public_tests)
    clauses = _ticket_clauses(text)
    assertion_count = public_assertion_count(public_tests)
    clause_rows: list[dict[str, object]] = []
    for index, clause in enumerate(clauses, start=1):
        covered = _clause_covered(clause, assertion_text, behavior_witnesses)
        clause_rows.append({
            "id": f"ticket-clause-{index}",
            "clause": clause,
            "covered": covered,
            "reason": "public assertion signal" if covered else "no public assertion signal",
        })
    few_assertions = bool(clauses) and (
        assertion_count < 2 or assertion_count < len(clauses))
    untouched = untouched_named_layers(layer_report)
    uncovered = [row for row in clause_rows if not row["covered"]]
    triggers: list[str] = []
    if few_assertions:
        triggers.append("few_public_assertions")
    if untouched:
        triggers.append("untouched_named_layers")
    low = bool(clauses) and bool(triggers)
    return {
        "schema": SCHEMA,
        "ticket_digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "ticket_clause_count": len(clause_rows),
        "ticket_clauses": clause_rows,
        "public_test_count": len(_source_texts(public_tests)),
        "assertion_count": assertion_count,
        "few_assertions": few_assertions,
        "named_layer_rows": copy.deepcopy(layer_report.get("rows", [])),
        "untouched_named_layers": list(untouched),
        "uncovered_ticket_clauses": uncovered,
        "triggers": triggers,
        "low_coverage": low,
        "coverage": LOW_COVERAGE if low else "ADEQUATE",
        "challenge_class": CHALLENGE_CLASS if low else None,
    }


build_report = coverage_report


def _report_for(runtime: object) -> dict[str, object] | None:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    if not isinstance(row, Mapping):
        row = {}
    root = Path(getattr(owner, "root", "."))
    ticket = _layers.row_ticket(row, root=root)
    public_tests = _layers.row_public_tests(row)
    start = getattr(runtime, "_false_completion_start_snapshot", {})
    current = snapshot_workspace(root)
    report = coverage_report(ticket, root, start_snapshot=start,
                             current_snapshot=current, public_tests=public_tests,
                             behavior_witnesses=getattr(runtime, "behavior_witness_records", ()))
    setattr(runtime, "_false_completion_report", report)
    return report


def report_for(runtime: object) -> dict[str, object] | None:
    if getattr(runtime, "stateframe_false_completion_mode", "off") != "on":
        return None
    provider = getattr(runtime, "false_completion_report", None)
    if callable(provider) and provider is not report_for:
        try:
            return provider()
        except TypeError:
            pass
    cached = getattr(runtime, "_false_completion_report", None)
    return cached if isinstance(cached, Mapping) else _report_for(runtime)


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_false_completion", dispatch=False, **fields)


def initialise(runtime: object, *, resume: object = None) -> None:
    owner = getattr(runtime, "owner", None)
    root = Path(getattr(owner, "root", "."))
    restored = None
    records = getattr(getattr(runtime, "journal", None), "records", None)
    if resume is not None and callable(records):
        for event in records():
            if (event.get("kind") == "false_completion_started"
                    and isinstance(event.get("start_snapshot"), Mapping)):
                restored = normalise_snapshot(event["start_snapshot"])
                break
    if restored is None:
        restored = snapshot_workspace(root)
        _emit(runtime, "false_completion_started",
              start_file_count=len(restored), start_snapshot=copy.deepcopy(restored))
    setattr(runtime, "_false_completion_start_snapshot", restored)


def _menu(runtime: object) -> tuple[str, ...]:
    value = getattr(runtime, "completion_menu", ())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item).strip().lower() for item in value if str(item).strip())
    return ()


def _arbiter_active(runtime: object) -> bool:
    if getattr(runtime, "stateframe_done_challenge_once_mode", "off") == "on":
        return True
    try:
        from kernel_first_green_disclosure import enabled as first_green_enabled
        return bool(first_green_enabled(runtime))
    except (ImportError, AttributeError, TypeError, ValueError):
        return getattr(runtime, "stateframe_first_green_disclosure_mode", "off") == "on"


def _repair_path(runtime: object, report: Mapping[str, object]) -> str:
    rows = report.get("untouched_named_layers", ())
    if isinstance(rows, Sequence):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            files = row.get("files", ())
            if isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
                for item in files:
                    if item and str(item) != _layers.UNKNOWN:
                        return str(item)
    owner = getattr(runtime, "owner", None)
    root = Path(getattr(owner, "root", ".")).resolve()
    target = getattr(runtime, "target", None)
    if target is not None:
        try:
            return Path(target).resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            pass
    return "."


def _challenge_text(report: Mapping[str, object]) -> str:
    clauses = report.get("uncovered_ticket_clauses", ())
    rendered = []
    if isinstance(clauses, Sequence):
        for row in clauses:
            if isinstance(row, Mapping):
                rendered.append(f"{row.get('id')}: {row.get('clause')}")
    if not rendered:
        rendered.append("no ticket clause has an assertion-backed public check")
    layers = report.get("untouched_named_layers", ())
    layer_names = []
    if isinstance(layers, Sequence):
        layer_names = [str(row.get("layer")) for row in layers if isinstance(row, Mapping)]
    suffix = (" Untouched named layers: " + ", ".join(layer_names) + ".") if layer_names else ""
    return ("UNVERIFIED: public coverage is weak; inspect these uncovered ticket clauses "
            "before submitting: " + "; ".join(rendered) + "." + suffix)


def merge_precommit(runtime: object, base: Mapping[str, object] | None, *,
                    admissible: bool, detail: str = "",
                    reason_codes: Sequence[str] = ()) -> dict[str, object] | None:
    if (getattr(runtime, "stateframe_false_completion_mode", "off") != "on"
            or not admissible):
        return dict(base) if isinstance(base, Mapping) else base
    report = report_for(runtime)
    if not isinstance(report, Mapping) or not report.get("low_coverage"):
        return dict(base) if isinstance(base, Mapping) else base
    challenge = _challenge_text(report)
    menu = _menu(runtime)
    if base is not None:
        assessment = copy.deepcopy(dict(base))
        codes = tuple(dict.fromkeys(
            str(item) for item in (*assessment.get("reason_codes", ()), REASON_CODE)
            if str(item)))
        assessment["reason_codes"] = codes
        old_detail = str(assessment.get("detail", "") or "")
        assessment["detail"] = (old_detail + "\n" if old_detail else "") + challenge
        assessment["false_completion_report"] = copy.deepcopy(dict(report))
        _emit(runtime, "false_completion_evidence", enforced=True, reason=REASON_CODE,
              trigger_count=len(report.get("triggers", ())),
              uncovered_count=len(report.get("uncovered_ticket_clauses", ())))
        return assessment
    reason = None
    if not _arbiter_active(runtime):
        reason = "no_challenge_arbiter"
    elif not menu or set(menu) <= {"done"}:
        reason = "sole_done_menu"
    elif "read" not in menu:
        reason = "no_read_on_menu"
    if reason is not None:
        _emit(runtime, "false_completion_evidence", enforced=False, reason=reason,
              trigger_count=len(report.get("triggers", ())),
              uncovered_count=len(report.get("uncovered_ticket_clauses", ())))
        return None
    _emit(runtime, "false_completion_evidence", enforced=True, reason=REASON_CODE,
          trigger_count=len(report.get("triggers", ())),
          uncovered_count=len(report.get("uncovered_ticket_clauses", ())))
    return {
        "reason_codes": tuple(dict.fromkeys((*reason_codes, REASON_CODE))),
        "detail": (detail + "\n" if detail else "") + challenge,
        "repair_action": {
            "tool": "read", "arguments": {"path": _repair_path(runtime, report)},
            "reason": "inspect public-ticket clauses before accepting done",
        },
        "false_completion_report": copy.deepcopy(dict(report)),
    }


def _uncovered_strings(report: Mapping[str, object]) -> list[str]:
    values = []
    rows = report.get("uncovered_ticket_clauses", ())
    if isinstance(rows, Sequence):
        values.extend(str(row.get("clause")) for row in rows if isinstance(row, Mapping))
    return list(dict.fromkeys(item for item in values if item))


def accepted_submission_fields(runtime: object) -> dict[str, object]:
    if getattr(runtime, "stateframe_false_completion_mode", "off") != "on":
        return {}
    report = report_for(runtime)
    if not isinstance(report, Mapping) or not report.get("low_coverage"):
        return {}
    # r44 review F1: coverage is a SEPARATE annotation.  It never overwrites the record's
    # ``assessment``/``challenge_class`` (a known violation must not become LOW_COVERAGE).
    return {
        "coverage": LOW_COVERAGE,
        "coverage_status": LOW_COVERAGE,
        "coverage_schema": SCHEMA,
        "coverage_challenge_class": CHALLENGE_CLASS,
        "uncovered_ticket_clauses": _uncovered_strings(report),
        "coverage_triggers": list(report.get("triggers", ())),
    }


def _revision(runtime: object, decision: object) -> str:
    try:
        from kernel_challenge_budget import artifact_revision
        return str(artifact_revision(runtime, basis=getattr(decision, "basis", None)))
    except (ImportError, OSError, TypeError, ValueError):
        basis = getattr(decision, "basis", None)
        return hashlib.sha256(repr(basis).encode("utf-8")).hexdigest()


def mark_accepted(runtime: object, decision: object) -> dict[str, object] | None:
    if (getattr(runtime, "stateframe_false_completion_mode", "off") != "on"
            or not bool(getattr(decision, "accepted", False))):
        return None
    fields = accepted_submission_fields(runtime)
    if not fields:
        return None
    existing = None
    for name in ("_done_submission_record", "_challenge_budget_submission_record",
                 "challenge_budget_submission_record"):
        value = getattr(runtime, name, None)
        if isinstance(value, Mapping):
            existing = dict(value)
            break
    created = existing is None
    if existing is None:
        existing = {
            "attempt_status": "submitted",
            "assessment": LOW_COVERAGE,
            "open_clause_ids": [],
            "revision": _revision(runtime, decision),
            "challenge_class": CHALLENGE_CLASS,
            "override_reason": "low_public_coverage",
            "run_challenge_count": 0,
            "run_challenge_cap": 0,
        }
    existing.update(copy.deepcopy(fields))
    existing["coverage"] = LOW_COVERAGE
    for name in ("_done_submission_record", "_challenge_budget_submission_record",
                 "challenge_budget_submission_record"):
        setattr(runtime, name, copy.deepcopy(existing))
    state = getattr(runtime, "_challenge_budget_state", None)
    if isinstance(state, dict) and isinstance(state.get("submission_record"), Mapping):
        state["submission_record"] = copy.deepcopy(existing)
    if created:
        _emit(runtime, "done_submission_record", **existing)
    return copy.deepcopy(existing)


def render_notice(report: Mapping[str, object] | None) -> str:
    if not isinstance(report, Mapping) or not report.get("low_coverage"):
        return ""
    lines = [HEADER,
             (f"Public assertions: {report.get('assertion_count', 0)}; "
              f"ticket clauses: {report.get('ticket_clause_count', 0)}; "
              "accepted done remains bounded evidence, not independent correctness.")]
    triggers = report.get("triggers", ())
    if isinstance(triggers, Sequence) and triggers:
        lines.append("Triggers: " + ", ".join(str(item) for item in triggers))
    rows = report.get("uncovered_ticket_clauses", ())
    if isinstance(rows, Sequence):
        lines.append("Uncovered ticket clauses:")
        for row in rows:
            if isinstance(row, Mapping):
                lines.append(f"- {row.get('id')}: {row.get('clause')}")
    for row in report.get("untouched_named_layers", ()) or ():
        if isinstance(row, Mapping):
            files = ", ".join(str(item) for item in row.get("files", ()))
            lines.append(f"- untouched named layer: {row.get('layer')} -> {files}")
    return "\n".join(lines)


__all__ = [
    "CHALLENGE_CLASS", "HEADER", "LOW_COVERAGE", "MODE_ALIAS", "MODE_ENV", "REASON_CODE",
    "RUNTIME_ENVIRONMENT", "SCHEMA", "accepted_submission_fields", "build_report",
    "coverage_report", "enabled", "initialise", "mark_accepted", "merge_precommit", "mode",
    "normalise_snapshot", "public_assertion_count", "render_notice", "report_for",
    "snapshot_workspace", "untouched_named_layers",
]
