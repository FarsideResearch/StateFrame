"""Bounded host semantic admission for model-proposed S1 spines.

The model may describe a typed proposal, but only the host resolves its rows
against the ticket ledger and mints the review receipt.  This module is
language-neutral and deliberately does not decide whether prose is true.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence


DECISIONS = frozenset({"SUPPORTED", "CONTRADICTED", "UNKNOWN"})
SCHEMA = "stateframe-s1-semantic-review/1"


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                         separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text(value: object, label: str, limit: int = 500) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value.strip()[:limit]


def _ledger_rows(ledger: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(item for item in _items(ledger.get("requirements"))
                 if isinstance(item, Mapping))


def normalize_proposal(ledger: Mapping[str, object], proposal: object) -> tuple[dict[str, object], ...]:
    """Validate proposal shape and complete source-span coverage.

    Host-only fields are removed before review.  In particular, a model cannot
    smuggle ``ADMITTED`` or an admission receipt through the proposal.
    """
    if not isinstance(proposal, Mapping):
        raise ValueError("S1 proposal must be an object")
    raw_rows = proposal.get("rows")
    rows = tuple(item for item in _items(raw_rows) if isinstance(item, Mapping))
    if not rows or len(rows) != len(_items(raw_rows)):
        raise ValueError("S1 proposal.rows must be a non-empty sequence of objects")
    requirements = _ledger_rows(ledger)
    if not requirements:
        raise ValueError("host S1 ledger has no requirements")
    span_ids = {str(item.get("id")) for item in _items(ledger.get("spans"))
                if isinstance(item, Mapping)}
    expected = [frozenset(str(item) for item in _items(row.get("source_span_ids",
                                                              row.get("ticket_span_ids"))))
               for row in requirements]
    if any(not refs or not refs.issubset(span_ids) for refs in expected):
        raise ValueError("host S1 ledger contains invalid source-span coverage")
    normalized: list[dict[str, object]] = []
    matched: set[int] = set()
    for index, row in enumerate(rows):
        refs = frozenset(str(item) for item in _items(row.get("source_span_ids",
                                                        row.get("ticket_span_ids"))))
        if not refs or not refs.issubset(span_ids):
            raise ValueError(f"proposal row {index} has unknown source spans")
        matches = [position for position, requirement_refs in enumerate(expected)
                   if position not in matched and refs == requirement_refs]
        if len(matches) != 1:
            raise ValueError(f"proposal row {index} does not map to one host ledger row")
        matched_position = matches[0]
        matched.add(matched_position)
        requirement = requirements[matched_position]
        spine = row.get("spine")
        if not isinstance(spine, Mapping):
            raise ValueError(f"proposal row {index} must provide typed spine data")
        semantic = str(spine.get("semantic_admission", "")).upper()
        if semantic == "ADMITTED" or spine.get("admission_receipt"):
            raise ValueError("model cannot provide S1 admission status or receipt")
        clean_spine = {
            str(key): copy.deepcopy(value)
            for key, value in spine.items()
            if key not in {"semantic_admission", "admission_receipt"}
        }
        # Canonical row identity and role belong to the host ledger.  The
        # proposal may describe predicates, but it cannot relink them to a
        # different requirement or smuggle a role/receipt through admission.
        ledger_id = str(ledger.get("ticket_source_id", ""))
        row_ref = {
            "ledger_id": ledger_id,
            "id": str(requirement.get("id", "")),
            "version": requirement.get("version"),
        }
        derivation = clean_spine.get("derivation")
        if not isinstance(derivation, Mapping):
            raise ValueError(f"proposal row {index} has no typed derivation")
        clean_derivation = copy.deepcopy(dict(derivation))
        clean_derivation["premise_rows"] = [row_ref]
        clean_derivation["admission_receipt"] = None
        clean_spine["derivation"] = clean_derivation
        clean_spine["depends_on"] = []
        clean_spine["lineage"] = []
        host_role = str(requirement.get("role", "")).upper()
        if host_role not in {"REPAIR", "PRESERVE", "NEW_BEHAVIOR"}:
            existing_spine = requirement.get("spine")
            host_role = str(existing_spine.get("role", "")) if isinstance(existing_spine, Mapping) else ""
        clean_spine["role"] = host_role if host_role in {"REPAIR", "PRESERVE", "NEW_BEHAVIOR"} else "REPAIR"
        modality = requirement.get("contract", {}).get("modality") if isinstance(requirement.get("contract"), Mapping) else None
        clean_spine["obligation"] = "REQUIRED" if modality == "required" else "ADVISORY"
        normalized.append({"source_span_ids": sorted(refs), "spine": clean_spine})
    if len(matched) != len(requirements):
        raise ValueError("S1 proposal must cover every host ledger row")
    return tuple(normalized)


def host_review(
        ledger: Mapping[str, object], proposal: object, *, ticket_hash: str,
        base_revision: str, reviewer: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Run one bounded host review seam and mint its receipt.

    ``reviewer`` is an injected host/parser boundary.  Its return value is
    reduced to the three public decisions; model text and model receipt IDs are
    never trusted as admission authority.
    """
    ticket = _text(ticket_hash, "ticket_hash")
    base = _text(base_revision, "base_revision")
    rows = normalize_proposal(ledger, proposal)
    context = {
        "schema": SCHEMA,
        "ticket_hash": ticket,
        "base_revision": base,
        "ledger_id": str(ledger.get("ticket_source_id", "")),
        "row_ids": [str(row.get("id")) for row in _ledger_rows(ledger)],
        "proposal": copy.deepcopy(list(rows)),
    }
    raw: Mapping[str, object] = {}
    if reviewer is not None:
        if not callable(reviewer):
            raise TypeError("semantic reviewer must be callable")
        try:
            candidate = reviewer(copy.deepcopy(context))
            if isinstance(candidate, Mapping):
                raw = candidate
        except (OSError, RuntimeError, TypeError, ValueError):
            raw = {"decision": "UNKNOWN", "unresolved": ["host_review_error"]}
    decision = str(raw.get("decision", "UNKNOWN")).upper()
    if decision not in DECISIONS:
        decision = "UNKNOWN"
    receipt = {
        "schema": SCHEMA,
        "receipt_id": "s1-semantic-review:" + _digest({
            "ticket_hash": ticket,
            "base_revision": base,
            "decision": decision,
            "proposal": rows,
        })[7:31],
        "ticket_hash": ticket,
        "base_revision": base,
        "decision": decision,
        "producer": "HOST_REVIEW_SEAM",
        "unresolved": [str(item) for item in _items(raw.get("unresolved"))][:32],
        "contradictions": [str(item) for item in _items(raw.get("contradictions"))][:32],
    }
    return receipt, rows


def admit_reviewed_proposal(
        ledger: Mapping[str, object], proposal: object, *, ticket_hash: str,
        base_revision: str, request_id: str,
        reviewer: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Return an admitted ledger only for a host ``SUPPORTED`` review."""
    receipt, rows = host_review(
        ledger, proposal, ticket_hash=ticket_hash, base_revision=base_revision,
        reviewer=reviewer,
    )
    if receipt["decision"] != "SUPPORTED":
        return None, receipt
    from kernel_s1_spine import admit_spine_table

    existing = ledger.get("spine")
    expected_revision = None
    if isinstance(existing, Mapping) and isinstance(existing.get("header"), Mapping):
        expected_revision = existing["header"].get("table_revision")
    admitted = admit_spine_table(
        ledger,
        rows,
        expected_table_revision=expected_revision,
        request_id=request_id,
        semantic_review={"decision": "SUPPORTED", "receipt_id": receipt["receipt_id"]},
    )
    return admitted, receipt


__all__ = [
    "DECISIONS", "SCHEMA", "admit_reviewed_proposal", "host_review",
    "normalize_proposal",
]
