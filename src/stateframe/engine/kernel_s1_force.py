"""Host-owned S1 draft and generic stage-force contracts.

The draft is intentionally conservative: ticket spans and deterministic ledger
expectations become provisional rows, while the model may only confirm or edit
those rows through the typed S1 decision.  This module does not own stage state
or admit a completion; :mod:`kernel_idealpath` remains the reducer owner.
"""
from __future__ import annotations

import hashlib
import json
import re
import copy
from collections.abc import Mapping, Sequence


DRAFT_SCHEMA = "stateframe-s1-host-draft/1"
ABSTAIN_ACTION = "abstain"
ABSTAIN_SCHEMA = "stateframe-stage-abstain/1"
_FORBIDDEN = re.compile(
    r"(?:hidden|gold|backup|private|reference\s+fix|reference\s+patch|solution\s+file)",
    re.IGNORECASE,
)


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _safe_text(value: object, *, limit: int = 1800) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", " ").strip()
    if _FORBIDDEN.search(text):
        text = _FORBIDDEN.sub("[redacted]", text)
    return text[:limit]


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _expectation_text(requirement: Mapping[str, object], clause: str) -> str:
    contract = requirement.get("contract")
    if isinstance(contract, Mapping):
        expectations = contract.get("expectations")
        if _items(expectations):
            rendered = json.dumps(expectations, sort_keys=True, ensure_ascii=False, default=str)
            safe = _safe_text(rendered)
            if safe:
                return safe
    return f"The observable behavior described by ticket span must be confirmed: {_safe_text(clause)}"


def _row(
        requirement: Mapping[str, object],
        *,
        spans: Mapping[str, Mapping[str, object]],
        index: int,
) -> dict[str, object]:
    row_id = _safe_text(requirement.get("id"), limit=80) or f"R{index}"
    raw_refs = requirement.get("source_span_ids")
    refs = [str(item) for item in _items(raw_refs) if str(item) in spans]
    if not refs:
        refs = [next(iter(spans))]
    clause = _safe_text(requirement.get("exact_clause")) or _safe_text(
        spans[refs[0]].get("quote")
    )
    contract = requirement.get("contract")
    trigger = (
        _safe_text(contract.get("action"))
        if isinstance(contract, Mapping) else "exercise the public behavior described by the ticket span"
    )
    if not trigger:
        trigger = "exercise the public behavior described by the ticket span"
    oracle = requirement.get("oracle_provenance")
    oracle_provenance = {"kind": "ticket_span_unresolved", "refs": list(refs)}
    if isinstance(oracle, Mapping):
        kind = _safe_text(oracle.get("kind"), limit=120)
        oracle_refs = [str(item) for item in _items(oracle.get("refs")) if str(item) in spans]
        if kind and oracle_refs:
            oracle_provenance = {"kind": kind, "refs": oracle_refs}
    return {
        "row_id": row_id,
        "ticket_refs": refs,
        "trigger": trigger,
        "observed_symptom": clause or "ticket-described behavior requires confirmation",
        "expected_observable": _expectation_text(requirement, clause),
        "wrong_behavior": (
            "Observed behavior does not satisfy the expected observable from the cited ticket span; "
            "model confirmation is required."
        ),
        "constraints": [],
        "expectation_refs": refs,
        "scenario_provenance": {"kind": "ticket_span", "refs": list(refs)},
        "oracle_provenance": oracle_provenance,
        "ambiguity_status": "NONE",
    }


def _fallback_requirements(spans: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "id": f"R{index}",
            "source_span_ids": [str(span.get("id"))],
            "exact_clause": span.get("quote", ""),
            "contract": {},
            "oracle_provenance": {"kind": "ticket_span_unresolved", "refs": [str(span.get("id"))]},
        }
        for index, span in enumerate(spans, 1)
    ]


_SPINE_RELATIONS = {
    "equals": "EQ",
    "contains": "CONTAINS",
    "unchanged": "UNCHANGED",
    "absent": "ABSENT",
    "raises": "RAISES",
    "does_not_raise": "DOES_NOT_RAISE",
}


def _spine_family(subject: str) -> str:
    lowered = subject.lower()
    if any(token in lowered for token in ("timing", "deadline", "heartbeat", "clock")):
        return "TIMING_BOUND"
    if any(token in lowered for token in ("call", "invocation", "transport")):
        return "CALLS"
    if any(token in lowered for token in ("event", "order", "sequence")):
        return "EVENT_ORDER"
    if any(token in lowered for token in ("state", "message", "content", "pool")):
        return "PERSISTED_STATE"
    if any(token in lowered for token in ("config", "setting", "default")):
        return "CONFIG_EFFECT"
    return "RETURN"


def _spine_predicate(
        expectation: Mapping[str, object], *, span_id: str, predicate_id: str,
) -> dict[str, object] | None:
    operator = str(expectation.get("operator", "")).strip().lower()
    relation = _SPINE_RELATIONS.get(operator)
    expected = expectation.get("expected")
    if relation is None or not _json_safe(expected):
        return None
    return {
        "id": predicate_id,
        "subject": _safe_text(expectation.get("observable"), limit=240) or "public_observable",
        "relation": relation,
        "rhs": {
            "kind": "LITERAL",
            "value": copy.deepcopy(expected),
            "support": {"span_refs": [span_id], "rule": "requirements-ledger-contract/1"},
        },
        "support": {"span_refs": [span_id], "rule": "requirements-ledger-contract/1"},
        "relation_text": None,
    }


def _json_safe(value: object) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _typed_spine_proposal(
        requirement: Mapping[str, object], *, ledger_id: str,
) -> dict[str, object] | None:
    """Convert only deterministic public ledger contracts into spine rows.

    Unresolved contracts intentionally return ``None``.  S1 must not promote a
    prose sentence into an admitted literal oracle merely to make TestWriter
    appear available.
    """
    contract = requirement.get("contract")
    if not isinstance(contract, Mapping) or contract.get("modality") != "required":
        return None
    source_refs = [str(item) for item in _items(requirement.get("source_span_ids"))]
    if len(source_refs) != 1:
        return None
    span_id = source_refs[0]
    expectations = [item for item in _items(contract.get("expectations")) if isinstance(item, Mapping)]
    if not expectations:
        return None
    predicates: list[dict[str, object]] = []
    for index, expectation in enumerate(expectations):
        predicate = _spine_predicate(
            expectation, span_id=span_id,
            predicate_id=f"{requirement.get('id')}:predicate:{index + 1}",
        )
        if predicate is None:
            return None
        predicates.append(predicate)
    preconditions: list[dict[str, object]] = []
    for index, condition in enumerate(
            item for item in _items(contract.get("preconditions")) if isinstance(item, Mapping)):
        predicate = _spine_predicate(
            condition, span_id=span_id,
            predicate_id=f"{requirement.get('id')}:condition:{index + 1}",
        )
        if predicate is None:
            return None
        preconditions.append(predicate)
    first_subject = str(predicates[0]["subject"])
    requirement_ref = {"ledger_id": ledger_id, "id": str(requirement.get("id")), "version": requirement.get("version")}
    typed = {
        "schema": "requirement-spine/1",
        "role": "REPAIR",
        "obligation": "REQUIRED",
        "derivation": {
            "kind": "EXPLICIT",
            "premise_rows": [requirement_ref],
            "premise_spans": [span_id],
            "rule": str(contract.get("derivation_template") or "requirements-ledger-contract/1"),
            "admission_receipt": None,
        },
        "reported_symptom": {
            "text": _safe_text(requirement.get("exact_clause")) or "ticket-described behavior",
            "support": {"span_refs": [span_id], "rule": "requirements-ledger-contract/1"},
        },
        "trigger": {
            "action": _safe_text(contract.get("action")) or "exercise the public behavior",
            "support": {"span_refs": [span_id], "rule": "requirements-ledger-contract/1"},
            "conditions": preconditions,
            "slots": [],
            "quantifier": "EACH_CASE",
            "case_partition_refs": [],
            "exception_refs": [],
        },
        "observables": [{
            "id": f"{requirement.get('id')}:observable:1",
            "family": _spine_family(first_subject),
            "surface_hint": first_subject,
            "surface_support": {"span_refs": [span_id], "rule": "requirements-ledger-contract/1"},
            "predicates": predicates,
            "binding_status": "UNBOUND",
            "binding_receipt": None,
        }],
        "wrong_behaviors": [{
            "id": f"{requirement.get('id')}:wrong:1",
            "description": "Observed behavior violates the public ticket contract.",
            "under_trigger": "SAME",
            "extra_conditions": [],
            "violates_observation_ids": [f"{requirement.get('id')}:observable:1"],
            "rejection_predicate": copy.deepcopy(predicates[0]),
            "origin": "REPORTED_SYMPTOM",
            "support": {"span_refs": [span_id], "rule": "requirements-ledger-contract/1"},
        }],
        "ambiguities": [],
        "joint_scenario_group": None,
        "depends_on": [],
        "lineage": [],
        "semantic_admission": "PROPOSED",
        "admission_receipt": None,
    }
    return {"source_span_ids": source_refs, "spine": typed}


def build_s1_host_draft(
        ticket_id: str,
        ticket_text: str,
        *,
        ticket_path: str = "ticket.md",
        row_role: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Compile a deterministic provisional S1 table from public ticket text.

    ``row_role`` (Q3, ``kernel_row_role.classify``) stamps every row with
    ``role``/``witness_kind``; a NEW_BEHAVIOR role also becomes the typed spine
    role.  ``None`` leaves the draft byte-identical to the pre-Q3 shape.
    """
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        raise ValueError("host S1 draft requires non-empty public ticket text")
    from kernel_requirements_ledger import build_ledger

    ledger = build_ledger(str(ticket_id or "ticket"), ticket_text, path=ticket_path)
    raw_spans = [item for item in ledger.get("spans", ()) if isinstance(item, Mapping)]
    spans = {
        str(item.get("id")): {
            "id": str(item.get("id")),
            "start_byte": item.get("start_byte"),
            "end_byte": item.get("end_byte"),
            "quote": _safe_text(item.get("quote")),
        }
        for item in raw_spans
        if str(item.get("id", ""))
    }
    if not spans:
        raise ValueError("host S1 draft requires at least one ticket span")
    requirements = [item for item in ledger.get("requirements", ()) if isinstance(item, Mapping)]
    if not requirements:
        requirements = _fallback_requirements(tuple(spans.values()))
    proposals = [
        _typed_spine_proposal(item, ledger_id=str(ledger.get("ticket_source_id", ticket_id)))
        for item in requirements
    ]
    role = str(row_role.get("role") or "") if isinstance(row_role, Mapping) else ""
    if role == "NEW_BEHAVIOR":
        for proposal in proposals:
            if isinstance(proposal, dict) and isinstance(proposal.get("spine"), dict):
                proposal["spine"]["role"] = role
    admitted_ledger: dict[str, object] | None = None
    semantic_review_receipt: dict[str, object] | None = None
    spine_admission: dict[str, object] = {
        "status": "GAP",
        "code": "S1SPINE_UNSUPPORTED_CONTRACT",
        "reason": "Every requirement needs a deterministic public contract before admission.",
    }
    if requirements and all(item is not None for item in proposals):
        from kernel_s1_spine import admit_spine_table

        try:
            semantic_review_receipt = {
                "schema": "stateframe-s1-semantic-review/1",
                "receipt_id": "host-contract-parser:requirements-ledger/1",
                "ticket_hash": _digest(ticket_text),
                "base_revision": "host-deterministic-parser",
                "decision": "SUPPORTED",
                "producer": "HOST_DETERMINISTIC_PUBLIC_CONTRACT",
            }
            admitted_ledger = admit_spine_table(
                ledger,
                [item for item in proposals if item is not None],
                request_id=f"host-s1:{ticket_id}",
                semantic_review={
                    "decision": "SUPPORTED",
                    "receipt_id": semantic_review_receipt["receipt_id"],
                },
            )
        except (KeyError, TypeError, ValueError):
            admitted_ledger = None
        if admitted_ledger is not None:
            spine_admission = {
                "status": "ADMITTED",
                "code": "S1SPINE_ADMITTED",
                "provenance": "HOST_DETERMINISTIC_PUBLIC_CONTRACT",
            }
    if admitted_ledger is not None:
        requirement_by_id = {
            str(item.get("id")): item
            for item in admitted_ledger.get("requirements", ())
            if isinstance(item, Mapping)
        }
        rows = []
        for index, item in enumerate(requirements, 1):
            row = _row(item, spans=spans, index=index)
            admitted = requirement_by_id.get(str(item.get("id")))
            if isinstance(admitted, Mapping) and isinstance(admitted.get("spine"), Mapping):
                row["spine"] = copy.deepcopy(dict(admitted["spine"]))
            rows.append(row)
    else:
        rows = [_row(item, spans=spans, index=index) for index, item in enumerate(requirements, 1)]
    if role:
        for row in rows:
            row["role"] = role
            row["witness_kind"] = str(row_role.get("witness_kind") or "")
    refs = [str(item["id"]) for item in spans.values()]
    draft = {
        "schema": DRAFT_SCHEMA,
        "spine_schema": "requirement-spine/1",
        "ticket_id": str(ticket_id or "ticket"),
        "ticket_sha256": _digest(ticket_text),
        "spans": list(spans.values()),
        "rows": rows,
        "spine_admission": spine_admission,
        "entry_hint": {"kind": "TICKET_SPANS", "refs": refs[:1]},
        "assumptions": [{
            "claim": "host draft is provisional until a typed S1 decision",
            "status": "OPEN",
        }],
        "provenance": "HOST_DRAFT_FROM_PUBLIC_TICKET",
    }
    if admitted_ledger is not None:
        draft["requirement_ledger"] = copy.deepcopy(admitted_ledger)
    if semantic_review_receipt is not None:
        draft["semantic_review"] = copy.deepcopy(semantic_review_receipt)
    draft["draft_digest"] = _digest(draft)
    return draft


def abstain_tool_schema(stage: str, *, state_epoch: int, ticket_hash: str,
                        base_revision: str) -> dict[str, object]:
    """Return the typed, non-prose fallback offered by a forced menu."""
    return {
        "type": "function",
        "function": {
            "name": ABSTAIN_ACTION,
            "description": (
                f"Abstain from the forced {stage} exit decision. The host will "
                "apply only its admitted draft or retain the stage hold."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {"const": stage},
                    "state_epoch": {"const": state_epoch},
                    "ticket_hash": {"const": ticket_hash},
                    "base_revision": {"const": base_revision},
                    "reason": {"type": "string", "maxLength": 500},
                },
                "required": ["stage", "state_epoch", "ticket_hash", "base_revision"],
                "additionalProperties": False,
            },
        },
    }


__all__ = [
    "ABSTAIN_ACTION", "ABSTAIN_SCHEMA", "DRAFT_SCHEMA", "abstain_tool_schema",
    "build_s1_host_draft",
]
