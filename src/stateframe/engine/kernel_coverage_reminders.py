"""Opt-in, neutral reminders for coverage clauses filtered from verdicts.

``kernel_named_layer_coverage`` deliberately restricts completion evidence to
affirmative clauses.  This companion keeps the filtered public clauses visible
as quoted review material and adds small, clause-bound checks.  Reminder data
is never a verdict: it is marked REVIEW/UNVERIFIED, is not put in named-layer
rows, and can only join the existing bounded done challenge as UNVERIFIED.

Switch: ``STATEFRAME_COVERAGE_REMINDERS=off|on`` (default off).
"""
from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


MODE_ENV = "STATEFRAME_COVERAGE_REMINDERS"
SCHEMA = "stateframe-coverage-reminders/1"
REVIEW_STATUS = "REVIEW"
UNVERIFIED = "UNVERIFIED"
_PASS_STATUSES = frozenset({
    "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "SATISFIED", "RESOLVED",
    "CLOSED", "COMPLETE", "COMPLETED", "VALIDATED",
})
_CLAUSE_KEYS = ("clause", "covered_clause", "requirement", "obligation",
                "source_text", "quote", "text", "description")
_STATUS_KEYS = ("result_class", "status", "verdict", "result", "outcome", "state")


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _clauses(ticket: object) -> tuple[str, ...]:
    from kernel_named_layer_coverage import _ticket_clauses

    return _ticket_clauses(str(ticket or ""))


def _affirmative(clause: str) -> bool:
    from kernel_named_layer_coverage import affirmative_clause

    return affirmative_clause(clause)


def _guard_terms(clause: str) -> tuple[str, ...]:
    from kernel_named_layer_coverage import (
        _CONDITION_WORD_RE,
        _NEGATION_RE,
        _UNCERTAIN_SCOPE_RE,
    )

    terms: list[str] = []
    if _NEGATION_RE.search(clause):
        terms.append("negation")
    if _UNCERTAIN_SCOPE_RE.search(clause):
        terms.append("hedge_or_alternative")
    if len(_CONDITION_WORD_RE.findall(clause)) > 1:
        terms.append("multiple_conditions")
    return tuple(terms) or ("affirmative_ticket_guard",)


def _layers(clause: str) -> tuple[str, ...]:
    from kernel_named_layer_coverage import extract_named_layers

    return extract_named_layers(clause)


def _target_rows(clause: str, root: Path, start: Mapping[str, object],
                 current: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Resolve reminder targets without changing the affirmative row set."""
    from kernel_named_layer_coverage import (
        _path_lookup,
        _rows_for_layer,
        product_files,
    )

    lookup = _path_lookup(root, product_files(root))
    rows: list[dict[str, object]] = []
    for layer in _layers(clause):
        rows.extend(
            row.to_dict()
            for row in _rows_for_layer(layer, clause, root, lookup, start, current)
        )
    return tuple(rows)


def _covered(row: Mapping[str, object]) -> bool:
    return (row.get("confidence") == "confident"
            and row.get("diff_from_start") is True)


def _files(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    result: list[str] = []
    for row in rows:
        values = row.get("files", ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            continue
        for value in values:
            name = str(value)
            if name and name != "UNKNOWN" and name not in result:
                result.append(name)
    return tuple(result)


def _open(row: Mapping[str, object]) -> bool:
    return not _covered(row)


def _normalise_clause(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().rstrip(".")


def _status(value: object) -> str:
    text = str(value or "").strip().upper()
    return text.rsplit(".", 1)[-1]


def _record_stale(record: Mapping[str, object]) -> bool:
    return record.get("stale") is True or record.get("current") is False


def _check_records(value: object, inherited_stale: bool = False):
    """Yield (record, stale) public check records, without treating declared
    test source as a pass.  ``stale`` PROPAGATES to every nested record: a
    parent container marked stale/non-current (e.g. an old revision or a
    superseded snapshot) makes every record nested under it stale too, even
    if the nested record itself claims a pass (Review18 #1: "nested pass
    under a stale parent")."""
    if isinstance(value, Mapping):
        stale = inherited_stale or _record_stale(value)
        yield value, stale
        for child in value.values():
            yield from _check_records(child, stale)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _check_records(child, inherited_stale)


def _record_identity(record: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("check_id", "selector", "test_name", "check"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    selectors = record.get("selectors")
    if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes, bytearray)):
        values.extend(str(item).strip() for item in selectors if str(item).strip())
    return tuple(dict.fromkeys(values))


def _registry_clause(record: Mapping[str, object], registry: object) -> str:
    """Resolve a host check identity through an explicit public-check registry."""
    identities = _record_identity(record)
    if not identities:
        return ""
    entries: list[Mapping[str, object]] = []
    if isinstance(registry, Mapping):
        for identity in identities:
            value = registry.get(identity)
            if isinstance(value, Mapping):
                entries.append(value)
    elif isinstance(registry, Sequence) and not isinstance(registry, (str, bytes, bytearray)):
        for value in registry:
            if not isinstance(value, Mapping):
                continue
            declared = _record_identity(value)
            if set(identities) & set(declared):
                entries.append(value)
    for entry in entries:
        for key in _CLAUSE_KEYS:
            value = _normalise_clause(entry.get(key))
            if value:
                return value
        clauses = entry.get("clauses", entry.get("covered_clauses"))
        if isinstance(clauses, Sequence) and not isinstance(clauses, (str, bytes, bytearray)):
            for value in clauses:
                normalized = _normalise_clause(value)
                if normalized:
                    return normalized
    return ""


def _record_clause(record: Mapping[str, object], registry: object) -> str:
    mapped = _registry_clause(record, registry)
    if mapped:
        return mapped
    for key in _CLAUSE_KEYS:
        value = _normalise_clause(record.get(key))
        if value:
            return value
    covered = record.get("covered_clauses")
    if isinstance(covered, Sequence) and not isinstance(covered, (str, bytes, bytearray)):
        for value in covered:
            normalized = _normalise_clause(value)
            if normalized:
                return normalized
    return ""


def _record_is_current(record: Mapping[str, object], current_binding: object) -> bool:
    if _record_stale(record) or record.get("binding_current") is False:
        return False
    for container_key in ("binding", "identity", "current_binding", "after_binding"):
        container = record.get(container_key)
        if isinstance(container, Mapping) and (
                container.get("current") is False or container.get("valid") is False):
            return False
        if isinstance(container, Mapping) and isinstance(current_binding, Mapping):
            for key in ("revision", "product_revision", "code_hash", "release_id"):
                if (container.get(key) is not None and current_binding.get(key) is not None
                        and str(container[key]) != str(current_binding[key])):
                    return False
    if isinstance(current_binding, Mapping):
        for key in ("revision", "current_revision", "product_revision", "code_hash", "release_id"):
            value = record.get(key)
            expected = current_binding.get(key)
            if value is not None and expected is not None and str(value) != str(expected):
                return False
    return True


def _record_executed(record: Mapping[str, object]) -> bool:
    execution = record.get("execution")
    if execution is not None:
        return str(execution).upper() == "EXECUTED"
    if "executed" in record:
        return record.get("executed") is True
    # Clause-bound assessment records are already reduced by the public-check
    # adapter; host receipts always carry ``execution`` and take the strict path.
    return not bool(_record_identity(record))


def _clause_check_recorded(public_checks: object, clause: str, *,
                           registry: object = None,
                           current_binding: object = None) -> bool:
    """A clause counts as checked only if the LATEST fresh, current-binding
    record for that exact public clause passed.

    Review18 #1: matching on ANY passing record let (a) an old-revision pass
    that was never explicitly marked stale, (b) a PASS immediately followed
    by a FAIL for the same clause, and (c) a passing record nested under a
    stale/non-current parent all close a reminder that was never actually
    re-verified.  Records are walked in encounter order (the order the host
    appends them), so the LAST matching, non-stale record for a clause is
    the current truth -- a later FAIL supersedes an earlier PASS.
    """
    wanted = _normalise_clause(clause)
    if not wanted:
        return False
    last_pass: bool | None = None
    for record, stale in _check_records(public_checks):
        if stale or not isinstance(record, Mapping):
            continue
        if not _record_is_current(record, current_binding) or not _record_executed(record):
            continue
        if _record_clause(record, registry) != wanted:
            continue
        statuses = [_status(record.get(key)) for key in _STATUS_KEYS]
        last_pass = any(value in _PASS_STATUSES for value in statuses)
    return bool(last_pass)


def _has_cli(clause: str) -> bool:
    return bool(re.search(
        r"\b(?:CLI|command[- ]line|console\s+interface|terminal\s+interface)\b",
        clause, re.IGNORECASE))


def _has_api(clause: str) -> bool:
    return bool(re.search(
        r"\b(?:API|endpoint|web\s+service|HTTP\s+endpoint)\b", clause,
        re.IGNORECASE))


def suggested_check(clause: str, layer: str | None = None) -> str:
    """Return one bounded check recipe derived only from one public clause."""
    text = str(clause or "").strip()
    lower = text.lower()
    checks: list[str] = []
    if _has_cli(text) or str(layer or "").upper() == "CLI":
        if re.search(r"\b(?:optional|flag|accept|forward|speed|field)\b", lower):
            checks.append("call the CLI entry with the new optional flag and check it is forwarded")
        else:
            checks.append("call the CLI entry named in the clause and check the stated behavior")
        if re.search(r"\b(?:existing|without\s+breaking|call\s+shape|arity)\b", lower):
            checks.append("call the CLI entry with its existing arguments and confirm the old call shape still works")
    if (re.search(r"\b(?:payload|request\s+object|caller)\b", lower)
            and re.search(r"\b(?:logging|log|retry|reusable|remain\s+usable|preserv)", lower)):
        checks.append("call with the same payload twice and confirm the first call left it intact for logging or retry")
    if _has_api(text) or str(layer or "").upper() == "API":
        checks.append("call the API entry named in the clause with the ticket value and check the stated response")
    if re.search(r"\b(?:validation|validate|reject|invalid|keep\s+the\s+existing)\b", lower):
        checks.append("call the named validation path with the clause's stated value and confirm the stated validation remains")
    if not checks:
        name = str(layer or "named layer").replace("file:", "")
        checks.append(f"exercise the {name} named in the clause and check the stated behavior")
    # A check is deliberately one line: it is a suggestion, not a new receipt.
    return "; ".join(dict.fromkeys(checks)) + "."


def _clause_for_layer(layer: str, ticket: object) -> str:
    for clause in _clauses(ticket):
        if layer in _layers(clause):
            return clause
        if layer.startswith("file:") and layer[5:] in clause:
            return clause
        if layer.startswith("symbol:") and layer[7:] in clause:
            return clause
    return ""


def _open_rows(report: Mapping[str, object], ticket: object) -> list[dict[str, object]]:
    raw = report.get("rows", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    result: list[dict[str, object]] = []
    for value in raw:
        if not isinstance(value, Mapping) or not _open(value):
            continue
        row = copy.deepcopy(dict(value))
        layer = str(row.get("layer", "UNKNOWN"))
        clause = _clause_for_layer(layer, ticket)
        row.update({
            "clause": clause,
            "suggested_check": suggested_check(clause, layer),
            "status": UNVERIFIED,
            "classification": UNVERIFIED,
            "completion_evidence": False,
            "blocking": False,
        })
        result.append(row)
    return result


def _reminder(clause: str, root: Path, start: Mapping[str, object],
              current: Mapping[str, object], public_checks: object = None,
              *, public_check_registry: object = None,
              current_binding: object = None) -> dict[str, object]:
    target_rows = _target_rows(clause, root, start, current)
    layers = _layers(clause)
    target_files = _files(target_rows)
    # A file edit establishes only a candidate target.  Filtered behavioral
    # clauses stay REVIEW until that exact public clause has a passing check;
    # unknown/unbound targets remain REVIEW, never a pass or product verdict.
    clause_check_recorded = _clause_check_recorded(
        public_checks, clause, registry=public_check_registry,
        current_binding=current_binding)
    open_state = not clause_check_recorded
    layer = layers[0] if len(layers) == 1 else None
    check = suggested_check(clause, layer)
    return {
        "clause": clause,
        "quoted_clause": clause,
        "guard": list(_guard_terms(clause)),
        "layers": list(layers),
        "target_files": list(target_files),
        "status": REVIEW_STATUS,
        "classification": UNVERIFIED,
        "completion_evidence": False,
        "blocking": False,
        "open": open_state,
        "clause_check_recorded": clause_check_recorded,
        "suggested_check": check,
    }


def _reminder_challenge(reminder: Mapping[str, object]) -> str:
    return (f'UNVERIFIED: REVIEW reminder remains open; "{reminder.get("clause", "")}"; '
            f'suggested check: {reminder.get("suggested_check", "")}')


def augment_report(report: Mapping[str, object], ticket: object, root: Path, *,
                   start_snapshot: Mapping[str, object],
                   current_snapshot: Mapping[str, object],
                   public_checks: object = None,
                   public_check_registry: object = None,
                   current_binding: object = None) -> dict[str, object]:
    """Add reminder metadata while leaving the named-layer evidence rows intact."""
    result = copy.deepcopy(dict(report))
    open_rows = _open_rows(result, ticket)
    rejected = [clause for clause in _clauses(ticket) if not _affirmative(clause)]
    reminders = [_reminder(clause, Path(root).resolve(), start_snapshot, current_snapshot,
                           public_checks,
                           public_check_registry=public_check_registry,
                           current_binding=current_binding)
                 for clause in rejected]
    open_reminders = [item for item in reminders if item["open"]]
    result["open_rows"] = open_rows
    result["review_reminders"] = reminders
    # The alias keeps the field discoverable to callers that name the organ
    # rather than the presentation label; both contain the same public data.
    result["coverage_reminders"] = copy.deepcopy(reminders)
    result["open_review_reminders"] = open_reminders
    result["reminder_challenges"] = [_reminder_challenge(item) for item in open_reminders]
    result["reminder_challenge_class"] = UNVERIFIED if open_reminders else None
    if open_rows or open_reminders:
        result["challenge_class"] = UNVERIFIED
        result["done_challenges"] = [
            *[str(item) for item in result.get("done_challenges", ())],
            *result["reminder_challenges"],
        ]
    return result


def compact_state(report: Mapping[str, object] | None) -> dict[str, object]:
    """Return only currently open rows/reminders for the per-turn frame state."""
    if not isinstance(report, Mapping):
        return {"schema": SCHEMA, "status": "CLEAR", "open_rows": [],
                "open_review_reminders": []}
    rows = [copy.deepcopy(dict(item)) for item in report.get("open_rows", ())
            if isinstance(item, Mapping)]
    reminders = [copy.deepcopy(dict(item)) for item in report.get("open_review_reminders", ())
                 if isinstance(item, Mapping)]
    return {
        "schema": SCHEMA,
        "status": "OPEN" if rows or reminders else "CLEAR",
        "open_rows": rows,
        "open_review_reminders": reminders,
    }


def render_checklist(report: Mapping[str, object] | None) -> str:
    """Render the compact, repeatable review checklist; empty means all covered."""
    state = compact_state(report)
    rows = state["open_rows"]
    reminders = state["open_review_reminders"]
    if not rows and not reminders:
        return ""
    from kernel_named_layer_coverage import CHECKLIST_HEADER

    lines = [CHECKLIST_HEADER,
             "[Coverage reminders | open rows only; review material, not a verdict]"]
    for row in rows:
        files = ", ".join(str(item) for item in row.get("files", ())) or "UNKNOWN"
        lines.append(f"- {row.get('layer', 'UNKNOWN')} -> {files} -> "
                     f"diff-from-start: {row.get('diff_label', 'UNKNOWN')}")
        lines.append(f"  suggested check: {row.get('suggested_check', '')}")
    for reminder in reminders:
        clause = str(reminder.get("quoted_clause", reminder.get("clause", "")))
        lines.append(f'- REVIEW REMINDER: "{clause}"')
        lines.append(f"  suggested check: {reminder.get('suggested_check', '')}")
    return "\n".join(lines)


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_coverage_reminders", dispatch=False, **fields)


def _menu(runtime: object) -> tuple[str, ...]:
    value = getattr(runtime, "completion_menu", None)
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
        return False


def _executable_repair(repair: object, menu: Sequence[str], *, runtime: object = None) -> bool:
    if not isinstance(repair, Mapping):
        return False
    tool = str(repair.get("tool", "") or "").strip().lower()
    if not tool or (menu and tool not in set(menu)):
        return False
    # A live runtime may expose the same side-effect-free capability check used
    # by its runner.  When present, menu membership is only the first gate; an
    # unavailable argument/path cannot be advertised as a repair.
    checker = getattr(runtime, "repair_action_available", None)
    if callable(checker):
        try:
            return bool(checker(tool, repair.get("arguments", {})))
        except (AttributeError, OSError, TypeError, ValueError):
            return False
    return True


def merge_precommit(runtime: object, base: Mapping[str, object] | None, *,
                    admissible: bool, detail: str = "",
                    reason_codes: Sequence[str] = ()) -> dict[str, object] | None:
    """Join open reminders to the named-layer pre-commit assessment.

    The returned reason is the existing named-layer UNVERIFIED code so the
    established challenge-once arbiter remains the sole bounded authority.
    Sole-done menus and missing arbiters remain submittable.
    """
    if getattr(runtime, "stateframe_coverage_reminders_mode", "off") != "on" or not admissible:
        return base
    provider = getattr(runtime, "layer_coverage_report", None)
    report = provider() if callable(provider) else getattr(runtime, "_layer_coverage_report", None)
    if not isinstance(report, Mapping):
        return base
    reminders = tuple(item for item in report.get("open_review_reminders", ())
                      if isinstance(item, Mapping) and item.get("open"))
    if not reminders:
        return base
    from kernel_named_layer_coverage import CODE

    if base is None:
        menu = _menu(runtime)
        reason = None
        if not _arbiter_active(runtime):
            reason = "no_challenge_arbiter"
        elif not menu or set(menu) <= {"done"}:
            reason = "sole_done_menu"
        elif "read" not in menu:
            reason = "no_read_on_menu"
        if reason is not None:
            _emit(runtime, "coverage_reminders_evidence", enforced=False,
                  reason=reason, reminders=len(reminders), menu=list(menu))
            return None

    challenges = " ".join(_reminder_challenge(item) for item in reminders)
    codes = tuple(dict.fromkeys((*reason_codes, CODE,
                                 *(str(item) for item in (base or {}).get("reason_codes", ())))))
    prefix = str((base or {}).get("detail", "") or detail or "")
    merged_detail = (prefix + "\n" if prefix else "") + "[coverage reminders] " + challenges
    first_path = next(
        (str(path) for item in reminders for path in item.get("target_files", ())
         if str(path) and str(path) != "UNKNOWN"),
        "",
    )
    repair = (base or {}).get("repair_action")
    if not isinstance(repair, Mapping) and first_path:
        repair = {"tool": "read", "arguments": {"path": first_path},
                  "reason": "review the public clause reminder"}
    if not _executable_repair(repair, _menu(runtime), runtime=runtime):
        _emit(runtime, "coverage_reminders_evidence", enforced=False,
              reason="no_executable_repair", reminders=len(reminders),
              menu=list(_menu(runtime)))
        return None
    result = dict(base or {})
    result.update({
        "reason_codes": codes,
        "detail": merged_detail,
        "reminders": [copy.deepcopy(dict(item)) for item in reminders],
        "reminder_challenges": [_reminder_challenge(item) for item in reminders],
    })
    if repair is not None:
        result["repair_action"] = repair
    _emit(runtime, "coverage_reminders_evidence", enforced=True,
          reason=CODE, reminders=len(reminders), menu=list(_menu(runtime)))
    return result


__all__ = [
    "MODE_ENV", "REVIEW_STATUS", "SCHEMA", "UNVERIFIED", "augment_report",
    "compact_state", "enabled", "merge_precommit", "mode", "render_checklist",
    "suggested_check",
]
