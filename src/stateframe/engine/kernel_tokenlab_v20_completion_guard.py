"""Fail-closed completion facts for token-lab v20.

The v19 terminal paths can see a public green receipt without seeing every
ticket clause.  This module provides one conservative, duck-typed check for
both post-green quieting and the token-lab no-progress owner.  It does not
cap reasoning or change an off arm.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_TOKENLAB_V20_FAIL_CLOSED_COMPLETION"
COMPAT_MODE_ENV = "STATEFRAME_TOKENLAB_V20_COMPATIBILITY_WITNESS"
SCHEMA = "stateframe-v20-fail-closed-completion/1"
EVENT = "tokenlab_v20_completion_guard_blocked"
CLAUSE_EVENT = "tokenlab_v20_ticket_clause_surface"
PRODUCER = "kernel_tokenlab_v20_completion_guard"

_CLOSED = frozenset({
    "SATISFIED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK", "RESOLVED",
    "CLOSED", "COMPLETE", "COMPLETED", "VALIDATED", "SATISFIED",
})
_COMPATIBILITY = re.compile(
    r"compatib|call\s+shape|existing\s+(?:call|interface|invocation)|"
    r"two[- ]argument|backward[- ]compat|legacy\s+(?:call|interface)", re.I)
_NORMATIVE = re.compile(
    r"\b(?:must|should|shall|required|need(?:s)? to|preserve|accept|return|"
    r"reject|continue|remain|without breaking)\b", re.I)
_REVISION_KEYS = (
    "revision", "product_revision", "current_revision", "artifact_revision",
    "repository_revision", "snapshot_digest",
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_v20_fail_closed_completion_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return mode() == "on"


def compatibility_enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_v20_compatibility_witness_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    value = str(os.environ.get(COMPAT_MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{COMPAT_MODE_ENV} must be off or on")
    return value == "on"


def _row(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            result = converter()
        except (AttributeError, TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return dict(result)
    return {}


def _rows(value: object) -> list[dict[str, object]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [row for item in value if (row := _row(item))]
    return []


def _text(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _status(row: Mapping[str, object]) -> str:
    for key in ("status", "verdict", "result_class", "result", "state", "outcome"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().upper()
    return "UNKNOWN"


def _ident(row: Mapping[str, object]) -> str:
    for key in ("obligation_id", "requirement_id", "clause_id", "check_id", "id"):
        value = row.get(key)
        if _text(value):
            return _text(value)
    return ""


def _clause(row: Mapping[str, object]) -> str:
    for key in ("exact_clause", "clause", "scope", "text", "description", "basis"):
        value = row.get(key)
        if _text(value):
            return _text(value)
    return ""


def _assessment(row: Mapping[str, object]) -> dict[str, object]:
    nested = row.get("assessment")
    return _row(nested) if nested is not None else dict(row)


def _ledger(runtime: object) -> Mapping[str, object] | None:
    value = getattr(runtime, "requirements_ledger", None)
    return value if isinstance(value, Mapping) else None


def _ticket_text(runtime: object, ledger: Mapping[str, object] | None) -> str:
    if isinstance(ledger, Mapping):
        ticket_id = _text(ledger.get("ticket_source_id"))
        for source in _rows(ledger.get("sources")):
            if _text(source.get("id")) == ticket_id and _text(source.get("text")):
                return _text(source.get("text"))
    owner = getattr(runtime, "owner", None)
    task_row = getattr(owner, "task_row", None)
    containers = [task_row, getattr(runtime, "public_material", None), runtime]
    for container in containers:
        row = container if isinstance(container, Mapping) else _row(container)
        for key in ("ticket", "ticket_text", "public_ticket", "public_brief", "brief", "msg"):
            if _text(row.get(key)):
                return _text(row.get(key))
    return ""


def _current_revisions(runtime: object) -> tuple[str, ...]:
    values: list[str] = []
    for name in ("product_revision", "current_revision", "revision", "host_closure_snapshot"):
        value = getattr(runtime, name, None)
        if _text(value) and _text(value) not in values:
            values.append(_text(value))
    for name in ("artifact_hashes", "_empirical_current_revisions"):
        value = getattr(runtime, name, None)
        if isinstance(value, Mapping):
            for item in value.values():
                if _text(item) and _text(item) not in values:
                    values.append(_text(item))
    return tuple(values)


def _is_current(row: Mapping[str, object], revisions: Sequence[str]) -> bool:
    values: list[str] = []
    for key in _REVISION_KEYS:
        value = row.get(key)
        if _text(value):
            values.append(_text(value))
    for key in ("binding", "after_binding", "current_binding"):
        nested = _row(row.get(key))
        for child in _REVISION_KEYS:
            if _text(nested.get(child)):
                values.append(_text(nested.get(child)))
    return not values or not revisions or any(item in revisions for item in values)


def _open_clause_ids(runtime: object) -> tuple[str, ...]:
    try:
        from kernel_post_green_menu import open_clause_ids
        return tuple(str(item) for item in open_clause_ids(runtime) if _text(item))
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return ()


def _counterexamples(runtime: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        from kernel_live_counterexample_challenge import live_rows
        rows.extend(dict(item) for item in live_rows(runtime) if isinstance(item, Mapping))
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        from kernel_counterexample_visibility import collect_counterexamples
        rows.extend(dict(item) for item in collect_counterexamples(runtime)
                    if isinstance(item, Mapping))
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    for name in ("live_counterexamples", "open_counterexamples"):
        rows.extend(_rows(getattr(runtime, name, ())))
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    revisions = _current_revisions(runtime)
    for row in rows:
        if not _is_current(row, revisions):
            continue
        marker = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
        if marker not in seen:
            seen.add(marker)
            unique.append(row)
    return unique


def _requirements(ledger: Mapping[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(ledger, Mapping):
        return []
    return [row for row in _rows(ledger.get("requirements")) if _ident(row)]


def _check_refs(requirement: Mapping[str, object]) -> tuple[str, ...]:
    refs = requirement.get("check_refs", ())
    result: list[str] = []
    for item in _rows(refs):
        ident = _ident(item)
        if ident and ident not in result:
            result.append(ident)
    return tuple(result)


def _compatibility_clause(requirement: Mapping[str, object], text: str = "") -> bool:
    kind = _text(requirement.get("kind"))
    clause = _clause(requirement) or text
    return "compat" in kind.lower() or bool(_COMPATIBILITY.search(clause))


def _candidate_evidence(runtime: object, ledger: Mapping[str, object] | None
                        ) -> list[dict[str, object]]:
    values: list[object] = []
    if isinstance(ledger, Mapping):
        values.extend(_rows(ledger.get("receipts")))
        values.extend(_rows(ledger.get("evidence")))
    for name in ("compatibility_witnesses", "witness_receipts", "host_check_receipts",
                 "host_repair_check_receipts", "probe_receipts", "test_receipts",
                 "check_receipts", "receipts", "host_closure_assessments"):
        values.extend(_rows(getattr(runtime, name, ())))
    return values


def _witness_for(requirement: Mapping[str, object], evidence: Sequence[Mapping[str, object]],
                 checks: Mapping[str, Mapping[str, object]], revisions: Sequence[str]) -> bool:
    refs = set(_check_refs(requirement))
    for row in evidence:
        if not _is_current(row, revisions) or _status(row) not in _CLOSED:
            continue
        row_ids = {_ident(row)}
        raw_ids = row.get("check_ids", row.get("checks", ()))
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            row_ids.update(_text(item) for item in raw_ids if _text(item))
        if refs and not refs.intersection(row_ids):
            continue
        marker = json.dumps(row, sort_keys=True, default=str).upper()
        explicit = bool(row.get("compatibility_witness") or row.get("witness"))
        check_kind = any("COMPATIBILITY_WITNESS" in json.dumps(checks.get(ref, {}),
                                                               sort_keys=True).upper()
                         for ref in refs)
        if explicit or check_kind or "COMPATIBILITY_WITNESS" in marker:
            execution = _text(row.get("execution"))
            return execution in {"", "EXECUTED", "PASSED", "PASS"}
    return False


def _prepare_witnesses(
        evidence: Sequence[Mapping[str, object]], checks: Mapping[str, Mapping[str, object]],
        revisions: Sequence[str], check_refs: Sequence[str],
) -> tuple[tuple[dict[str, object], ...], dict[str, tuple[int, ...]],
           dict[str, tuple[int, ...]], tuple[int, ...], frozenset[str]]:
    rows: list[dict[str, object]] = []
    by_id: dict[str, list[int]] = {}
    marked_by_id: dict[str, list[int]] = {}
    marked_indices: list[int] = []
    for row in evidence:
        if not _is_current(row, revisions) or _status(row) not in _CLOSED:
            continue
        row_ids = {_ident(row)}
        raw_ids = row.get("check_ids", row.get("checks", ()))
        if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes, bytearray)):
            row_ids.update(_text(item) for item in raw_ids if _text(item))
        marker = "COMPATIBILITY_WITNESS" in json.dumps(
            row, sort_keys=True, default=str).upper()
        item = {
            "ids": frozenset(row_ids),
            "explicit": bool(row.get("compatibility_witness") or row.get("witness")),
            "marker": marker,
            "execution": _text(row.get("execution")),
        }
        index = len(rows)
        rows.append(item)
        for ident in row_ids:
            by_id.setdefault(ident, []).append(index)
            if marker or item["explicit"]:
                marked_by_id.setdefault(ident, []).append(index)
        if marker or item["explicit"]:
            marked_indices.append(index)
    check_witness_refs = frozenset(
        ref for ref in check_refs
        if "COMPATIBILITY_WITNESS" in json.dumps(
            checks.get(ref, {}), sort_keys=True).upper()
    )
    return (
        tuple(rows),
        {ident: tuple(indices) for ident, indices in by_id.items()},
        {ident: tuple(indices) for ident, indices in marked_by_id.items()},
        tuple(marked_indices),
        check_witness_refs,
    )


def _witness_for_prepared(requirement: Mapping[str, object],
                          prepared: tuple[tuple[dict[str, object], ...],
                                          dict[str, tuple[int, ...]],
                                          dict[str, tuple[int, ...]], tuple[int, ...],
                                          frozenset[str]]) -> bool:
    rows, by_id, marked_by_id, marked_indices, check_witness_refs = prepared
    refs = set(_check_refs(requirement))
    check_kind = bool(refs.intersection(check_witness_refs))
    if not refs:
        indices = marked_indices
    else:
        selected = by_id if check_kind else marked_by_id
        indices = tuple(sorted({index for ref in refs for index in selected.get(ref, ())}))
    for index in indices:
        row = rows[index]
        if row["explicit"] or check_kind or row["marker"]:
            return row["execution"] in {"", "EXECUTED", "PASSED", "PASS"}
    return False


def _ticket_clauses(text: str) -> tuple[str, ...]:
    clauses = [" ".join(part.split()) for part in re.split(r"(?<=[.!?])\s+|\n+", text)
               if _text(part) and _NORMATIVE.search(part)]
    return tuple(dict.fromkeys(clauses))


def _add(blockers: list[dict[str, object]], seen: set[str], *, ident: str,
         clause: str, reason: str, counterexample: bool = False) -> None:
    key = ident or clause or reason
    if key in seen:
        return
    seen.add(key)
    blockers.append({"id": ident or "TICKET_CLAUSE", "clause": clause or ident,
                     "reason": reason, "concrete_counterexample": counterexample})


def find_blockers(runtime: object, *, phase: str = "completion") -> tuple[dict[str, object], ...]:
    """Return current-revision counterexamples and ticket clauses without proof."""
    if not enabled(runtime):
        return ()
    blockers: list[dict[str, object]] = []
    seen: set[str] = set()
    ledger = _ledger(runtime)
    ticket = _ticket_text(runtime, ledger)
    requirements = _requirements(ledger)
    requirement_by_id = {_ident(row): row for row in requirements}
    host_assessments = _rows(getattr(runtime, "host_closure_assessments", ()))

    for ident in _open_clause_ids(runtime):
        row = requirement_by_id.get(ident, {})
        _add(blockers, seen, ident=ident, clause=_clause(row),
             reason="ticket clause is open or unverified")

    for requirement in requirements:
        assessment = _assessment(requirement)
        if _status(assessment).lower() not in {item.lower() for item in _CLOSED}:
            _add(blockers, seen, ident=_ident(requirement), clause=_clause(requirement),
                 reason="requirements ledger assessment is not closed")
    for row in host_assessments:
        if _status(row) not in _CLOSED:
            _add(blockers, seen, ident=_ident(row) or "HOST_ASSESSMENT_UNIDENTIFIED",
                 clause=_clause(row), reason="host assessment is not closed")

    if isinstance(ledger, Mapping):
        spans = {_text(row.get("id")): _clause(row) for row in _rows(ledger.get("spans"))}
        for row in _rows(ledger.get("coverage_dispositions")):
            if _text(row.get("disposition")).lower() in {"unresolved", "unmapped", "open"}:
                refs = _rows(row.get("requirement_refs"))
                ident = _ident(refs[0]) if refs else _text(row.get("span_id"))
                _add(blockers, seen, ident=ident,
                     clause=spans.get(_text(row.get("span_id")), _clause(row)),
                     reason="ticket clause has no admitted interpretation")

    evidence = _candidate_evidence(runtime, ledger)
    checks = {_ident(row): row for row in _rows(ledger.get("checks"))} if isinstance(ledger, Mapping) else {}
    revisions = _current_revisions(runtime)
    if compatibility_enabled(runtime):
        compatibility_requirements = [row for row in requirements
                                       if _compatibility_clause(row)]
        if not compatibility_requirements:
            compatibility_requirements = [
                {"id": f"ticket-compat-{index}", "exact_clause": clause,
                 "kind": "compatibility", "check_refs": ()}
                for index, clause in enumerate(_ticket_clauses(ticket), 1)
                if _COMPATIBILITY.search(clause)
            ]
        if compatibility_requirements:
            check_refs = tuple(dict.fromkeys(
                ref for requirement in compatibility_requirements for ref in _check_refs(requirement)
            ))
            prepared_witnesses = _prepare_witnesses(evidence, checks, revisions, check_refs)
            for requirement in compatibility_requirements:
                if not _witness_for_prepared(requirement, prepared_witnesses):
                    _add(blockers, seen, ident=_ident(requirement), clause=_clause(requirement),
                         reason="compatibility surface has no current executed witness")

    for row in _counterexamples(runtime):
        _add(blockers, seen, ident=_ident(row), clause=_clause(row),
             reason="concrete counterexample on current revision", counterexample=True)

    ticket_clauses = _ticket_clauses(ticket)
    if ticket and not ledger and not host_assessments:
        for index, clause in enumerate(ticket_clauses, 1):
            _add(blockers, seen, ident=f"ticket-clause-{index}", clause=clause,
                 reason="ticket clause has no host assessment")
    elif ticket and not ledger:
        assessed_text = " ".join(_clause(row) for row in host_assessments).lower()
        for index, clause in enumerate(ticket_clauses, 1):
            if _COMPATIBILITY.search(clause) or clause.lower() in assessed_text:
                continue
            _add(blockers, seen, ident=f"ticket-clause-{index}", clause=clause,
                 reason="ticket clause is not represented by a host assessment")
    elif ticket and isinstance(ledger, Mapping) and not requirements \
            and not _rows(ledger.get("coverage_dispositions")):
        for index, clause in enumerate(ticket_clauses, 1):
            _add(blockers, seen, ident=f"ticket-clause-{index}", clause=clause,
                 reason="ticket clause has no ledger requirement")
    return tuple(blockers)


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emitter = getattr(journal, "emit", None)
    if callable(emitter):
        try:
            emitter(kind, producer=PRODUCER, event_schema=SCHEMA, dispatch=False, **fields)
        except Exception:  # noqa: BLE001 - evidence must not break completion
            return


def guard(runtime: object, *, phase: str = "completion") -> tuple[dict[str, object], ...]:
    blockers = find_blockers(runtime, phase=phase)
    if not blockers:
        return ()
    state = getattr(runtime, "_tokenlab_v20_completion_guard_events", None)
    if not isinstance(state, set):
        state = set()
        try:
            setattr(runtime, "_tokenlab_v20_completion_guard_events", state)
        except (AttributeError, TypeError):
            pass
    digest = hashlib.sha256(json.dumps(blockers, sort_keys=True, default=str).encode()).hexdigest()
    key = (str(phase), digest)
    if key not in state:
        state.add(key)
        _emit(runtime, EVENT, phase=phase, allowed=False, blocker_count=len(blockers),
              blockers=[dict(row) for row in blockers],
              concrete_counterexample_ids=[row["id"] for row in blockers
                                           if row.get("concrete_counterexample")])
        clause_texts = [str(row.get("clause", "")) for row in blockers if row.get("clause")]
        if clause_texts:
            _emit(runtime, CLAUSE_EVENT, phase=phase, clause_texts=clause_texts,
                  clause_count=len(clause_texts), surfaced_once=True)
    return blockers


__all__ = ["CLAUSE_EVENT", "COMPAT_MODE_ENV", "EVENT", "MODE_ENV", "PRODUCER", "SCHEMA",
           "compatibility_enabled", "enabled", "find_blockers", "guard", "mode"]
