"""Host-owned S2 localization fallback and bounded reject-loop receipts.

S2FORCE is deliberately a small adapter around the existing causal-localizer
contract.  It may summarize evidence already observed by the host, but it
never invents a source path or treats a model refusal as localization evidence.
The IDEALPATH reducer remains the owner of stage transitions.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence


SCHEMA = "stateframe-s2force/1"
DRAFT_SCHEMA = "stateframe-s2-localization-draft/1"
RECEIPT_SCHEMA = "stateframe-s2-localization-receipt/1"
MENU_SCHEMA = "stateframe-s2force-menu/1"
ALTERNATIVE_ACTION = "localization_alternative"
DEFAULT_STAGE_TURNS = 3
DEFAULT_REJECT_STREAK = 2
_TRUTHY = frozenset({"1", "on", "true", "yes"})
_FORBIDDEN = re.compile(
    r"(?:hidden|gold|backup|private|reference\s+fix|reference\s+patch|solution\s+file)",
    re.IGNORECASE,
)


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the v46 S2 fallback is active for this process."""
    source = os.environ if environ is None else environ
    return str(source.get("STATEFRAME_S2FORCE", "off")).strip().lower() in _TRUTHY


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _text(value: object, *, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", " ").strip()
    if not text or len(text) > limit or _FORBIDDEN.search(text):
        return ""
    return text


def _digest(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _refs(value: object, *, limit: int = 32) -> list[str]:
    result: list[str] = []
    for item in _items(value):
        ref = _text(item, limit=300)
        if ref and ref not in result:
            result.append(ref)
        if len(result) >= limit:
            break
    return result


def _candidate_ids(frontier: Mapping[str, object] | None, evidence: Sequence[object],
                   *, target_path: str) -> list[str]:
    candidates: list[str] = []
    bound = _text(target_path, limit=500)
    if bound:
        candidates.append(bound)
    if isinstance(frontier, Mapping):
        for item in _items(frontier.get("ranked_ids")):
            candidate = _text(item, limit=400)
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        for item in _items(frontier.get("candidates")):
            if not isinstance(item, Mapping):
                continue
            candidate = _text(item.get("candidate_id"), limit=400)
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        arguments = item.get("arguments")
        path = arguments.get("path") if isinstance(arguments, Mapping) else None
        candidate = _text(path, limit=500)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates[:5]


def _frontier_from_evidence(evidence: Sequence[object], *, target_path: str) -> dict[str, object]:
    """Lower existing model navigation into the causal-localizer contract."""
    from kernel_causal_localizer import build_frontier

    suspects: list[dict[str, object]] = []
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        arguments = item.get("arguments")
        path = _text(arguments.get("path"), limit=500) if isinstance(arguments, Mapping) else ""
        if not path:
            continue
        evidence_id = _text(item.get("evidence_id"), limit=300)
        suspects.append({
            "entity_id": "model-evidence:" + path,
            "path": path,
            "symbol_id": _text(arguments.get("symbol"), limit=300)
            if isinstance(arguments, Mapping) else "",
            "kind": "MODEL_NAVIGATION",
            "status": "OBSERVED",
            "score": 1.0 if path == _text(target_path, limit=500) else 0.5,
            "channels": ["model_read_or_grep"],
            "evidence": [{"ref_id": evidence_id}] if evidence_id else [],
        })
    bound = _text(target_path, limit=500)
    if bound and not any(item.get("path") == bound for item in suspects):
        suspects.insert(0, {
            "entity_id": "host-target:" + bound,
            "path": bound,
            "kind": "HOST_BOUND_TARGET",
            "status": "BOUND",
            "score": 1.0,
            "channels": ["host_target"],
            "evidence": [],
        })
    return build_frontier({
        "schema": "stateframe-localize/1",
        "status": "SEARCHED",
        "suspects": suspects,
        "source_receipt_ids": _refs(
            [item.get("evidence_id") for item in evidence if isinstance(item, Mapping)]
        ),
        "coverage": "S2FORCE_MODEL_EVIDENCE",
    })


def _evidence_refs(frontier: Mapping[str, object] | None,
                   evidence: Sequence[object]) -> list[str]:
    refs: list[str] = []
    if isinstance(frontier, Mapping):
        refs.extend(_refs(frontier.get("source_receipt_ids")))
        receipt = frontier.get("receipt")
        if isinstance(receipt, Mapping):
            refs.extend(_refs((receipt.get("receipt_id"),)))
        for item in _items(frontier.get("candidates")):
            if isinstance(item, Mapping):
                refs.extend(_refs(item.get("evidence_refs")))
    for item in evidence:
        if not isinstance(item, Mapping):
            continue
        refs.extend(_refs((item.get("evidence_id"), item.get("receipt_id"))))
    return list(dict.fromkeys(refs))[:32]


def build_s2_localization_draft(
        *,
        causal_frontier: Mapping[str, object] | None,
        model_evidence: Sequence[object],
        row_ids: Sequence[object],
        target_path: str = "",
        target_symbol: str = "",
        ticket_hash: str = "",
        base_revision: str = "",
) -> dict[str, object]:
    """Build a typed S2 answer from host frontier and bounded model evidence."""
    rows = _refs(row_ids, limit=64)
    evidence = [
        copy_item
        for item in model_evidence
        if isinstance(item, Mapping)
        for copy_item in ({
            "evidence_id": _text(item.get("evidence_id"), limit=300),
            "action": _text(item.get("action"), limit=80),
            "arguments": dict(item.get("arguments")) if isinstance(item.get("arguments"), Mapping) else {},
            "observation": _text(item.get("observation"), limit=900),
            "step": item.get("step"),
        },)
    ][:64]
    frontier = dict(causal_frontier) if isinstance(causal_frontier, Mapping) else {}
    if not frontier:
        frontier = _frontier_from_evidence(evidence, target_path=target_path)
    candidate_ids = _candidate_ids(frontier, evidence, target_path=target_path)
    evidence_refs = _evidence_refs(frontier, evidence)
    if not evidence_refs:
        evidence_refs = ["s2force:host-frontier"]
    edges = [
        {
            "row_id": row_id,
            "candidate_id": candidate_ids[0],
            "relation": "causal_localization",
            "evidence_refs": evidence_refs[:3],
        }
        for row_id in rows
    ] if candidate_ids else []
    unresolved_edges = [
        {
            "row_id": row_id,
            "reason": "NO_HOST_BOUNDARY",
            "evidence_refs": evidence_refs[:3],
            "next_route": "expand_lookup_scope",
        }
        for row_id in rows
    ] if not candidate_ids else []
    answer = {
        "row_ids": rows,
        "entry_refs": [
            "causal_localizer:localization_frontier",
            *evidence_refs[:3],
        ],
        "edges": edges,
        "decision_refs": evidence_refs[:8],
        "outcome_refs": candidate_ids[:5] or evidence_refs[:5],
        "unresolved_edges": unresolved_edges,
        "observation_boundary": (
            "HOST_DRAFT_FROM_CAUSAL_FRONTIER_AND_MODEL_NAVIGATION_EVIDENCE"
        ),
    }
    draft = {
        "schema": DRAFT_SCHEMA,
        "contract_schema": "stateframe-idealpath/1",
        "causal_localizer": "kernel_causal_localizer",
        "causal_tool": "localization_frontier",
        "ticket_hash": _text(ticket_hash, limit=300),
        "base_revision": _text(base_revision, limit=300),
        "target": {
            "path": _text(target_path, limit=500),
            "symbol": _text(target_symbol, limit=300),
        },
        "frontier": json.loads(json.dumps(frontier, ensure_ascii=False, default=str)),
        "model_evidence": evidence,
        "evidence_refs": evidence_refs,
        "answer": answer,
        "provenance": "HOST_DRAFT_FROM_CAUSAL_FRONTIER_AND_MODEL_EVIDENCE",
    }
    draft["draft_digest"] = _digest(draft)
    return draft


def draft_target(draft: Mapping[str, object]) -> dict[str, str] | None:
    """Extract one source target from the already-typed S2 draft evidence."""
    if not isinstance(draft, Mapping):
        return None

    def candidate_target(value: object) -> dict[str, str] | None:
        if not isinstance(value, Mapping):
            return None
        path = _text(value.get("path", value.get("source_path")), limit=500)
        if not path:
            return None
        symbol = _text(
            value.get("symbol", value.get("symbol_id", value.get("name"))),
            limit=300,
        )
        return {"path": path, "symbol": symbol}

    target = candidate_target(draft.get("target"))
    if target is not None:
        return target
    frontier = draft.get("frontier")
    if isinstance(frontier, Mapping):
        for candidate in _items(frontier.get("candidates")):
            target = candidate_target(candidate)
            if target is not None:
                return target
    for evidence in _items(draft.get("model_evidence")):
        if not isinstance(evidence, Mapping):
            continue
        target = candidate_target(evidence.get("arguments"))
        if target is not None:
            return target
    return None


def build_receipt(
        draft: Mapping[str, object], *, stage_epoch: int, ticket_hash: str,
        base_revision: str, forced: bool, override: bool,
) -> dict[str, object]:
    """Mint a compact receipt for a host or model-owned S2 localization."""
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "receipt_id": "s2force:" + _digest({
            "draft": draft.get("draft_digest"),
            "stage_epoch": stage_epoch,
            "ticket_hash": ticket_hash,
            "base_revision": base_revision,
            "forced": forced,
            "override": override,
        })[7:31],
        "stage": "S2",
        "next_stage": "S3",
        "state_epoch": stage_epoch,
        "ticket_hash": ticket_hash,
        "base_revision": base_revision,
        "draft_digest": draft.get("draft_digest"),
        "causal_localizer": "kernel_causal_localizer",
        "causal_tool": "localization_frontier",
        "model_evidence_refs": list(_refs(draft.get("evidence_refs"), limit=32)),
        "provenance": (
            "HOST_FORCED_S2_LOCALIZATION" if forced
            else "MODEL_TYPED_S2_LOCALIZATION_ALTERNATIVE"
        ),
        "forced": forced,
        "override": override,
    }
    receipt["receipt_digest"] = _digest(receipt)
    return receipt


def alternative_tool_schema(*, state_epoch: int, ticket_hash: str,
                            base_revision: str) -> dict[str, object]:
    """Return the explicit typed model override for an exhausted S2 menu."""
    from kernel_idealpath import _answer_schema

    return {
        "type": "function",
        "function": {
            "name": ALTERNATIVE_ACTION,
            "description": (
                "Submit a typed S2 localization alternative to the host draft. "
                "This is the only model override after the S2 budget is exhausted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "stage": {"const": "S2"},
                    "state_epoch": {"const": state_epoch},
                    "ticket_hash": {"const": ticket_hash},
                    "base_revision": {"const": base_revision},
                    "evidence_refs": {"type": "array", "items": {"type": "string"}, "maxItems": 64},
                    "answer": _answer_schema("S2"),
                },
                "required": [
                    "stage", "state_epoch", "ticket_hash", "base_revision",
                    "evidence_refs", "answer",
                ],
                "additionalProperties": False,
            },
        },
    }


def reject_loop_notice(*, tool: str, streak: int, admitted_actions: Sequence[object]) -> dict[str, object]:
    """Return one typed line for a repeated pre-dispatch refusal."""
    actions = ",".join(_refs(admitted_actions, limit=12))
    message = f"S2FORCE:{tool} refused x{streak}; admitted={actions}"
    return {
        "schema": MENU_SCHEMA,
        "code": "S2_REJECT_LOOP_RERENDER",
        "tool": _text(tool, limit=80),
        "streak": streak,
        "admitted_actions": list(_refs(admitted_actions, limit=12)),
        "message": message[:300],
    }


__all__ = [
    "ALTERNATIVE_ACTION", "DEFAULT_REJECT_STREAK", "DEFAULT_STAGE_TURNS",
    "DRAFT_SCHEMA", "MENU_SCHEMA", "RECEIPT_SCHEMA", "SCHEMA",
    "alternative_tool_schema", "build_receipt", "build_s2_localization_draft", "draft_target",
    "enabled", "reject_loop_notice",
]
