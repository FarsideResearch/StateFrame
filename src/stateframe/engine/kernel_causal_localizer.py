"""Bounded S2 causal-frontier projection over the existing LocalizeIndex."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence


SCHEMA = "stateframe-causal-localization/1"
TOOL_NAME = "localization_frontier"
MAX_RANKED = 5


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _text(value: object, limit: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    text = value.replace("\x00", " ").strip()
    return text if len(text) <= limit else ""


def build_frontier(search_result: Mapping[str, object], *, limit: int = MAX_RANKED) -> dict[str, object]:
    """Expose only bounded host-ranked candidates, preserving source evidence."""
    if search_result.get("schema") != "stateframe-localize/1":
        return {"schema": SCHEMA, "status": "REFUSED", "code": "LOCALIZATION_SCHEMA_REQUIRED"}
    suspects = [item for item in _items(search_result.get("suspects")) if isinstance(item, Mapping)]
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in suspects:
        ident = _text(item.get("entity_id"), 400)
        if not ident or ident in seen:
            continue
        seen.add(ident)
        evidence = [
            _text(ref.get("ref_id"), 500)
            for ref in _items(item.get("evidence"))
            if isinstance(ref, Mapping) and _text(ref.get("ref_id"), 500)
        ]
        candidates.append({
            "candidate_id": ident,
            "path": _text(item.get("path"), 500),
            "symbol_id": _text(item.get("symbol_id"), 400),
            "line": item.get("line"),
            "kind": _text(item.get("kind"), 80),
            "status": _text(item.get("status"), 80) or "UNRESOLVED",
            "score": item.get("score"),
            "channels": [str(value) for value in _items(item.get("channels"))][:8],
            "evidence_refs": evidence[:8],
        })
        if len(candidates) >= max(1, min(MAX_RANKED, int(limit))):
            break
    receipt = search_result.get("receipt")
    return {
        "schema": SCHEMA,
        "status": "SEARCHED",
        "snapshot_id": search_result.get("snapshot_id"),
        "scope_digest": search_result.get("scope_digest"),
        "source_receipt_ids": list(_items(search_result.get("source_receipt_ids")))[:64],
        "candidates": candidates,
        "ranked_ids": [str(item["candidate_id"]) for item in candidates],
        "receipt": copy.deepcopy(dict(receipt)) if isinstance(receipt, Mapping) else None,
        "completeness": {
            "scope": search_result.get("coverage"),
            "omitted_page_handle": next((item.get("next_cursor") for item in _items(search_result.get("frontier")) if isinstance(item, Mapping) and item.get("next_cursor")), None),
            "bounded": True,
        },
    }


def tool_schema(frontier: Mapping[str, object] | None = None) -> dict[str, object]:
    candidates = frontier.get("ranked_ids", ()) if isinstance(frontier, Mapping) else ()
    ranked_ids = [str(item) for item in _items(candidates) if str(item)]
    # Audit 2026-10-05 #1: an empty frontier must not emit ``enum: []`` (llama.cpp
    # HTTP 500).  With nothing to rank, the only valid ranking is the empty list.
    ranked_schema: dict[str, object] = (
        {"type": "array", "maxItems": MAX_RANKED, "items": {"type": "string", "enum": ranked_ids}}
        if ranked_ids else {"type": "array", "maxItems": 0})
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Rank a bounded host-generated S2 causal frontier and optionally select one candidate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ranked_ids": ranked_schema,
                    "selected_id": {"type": "string", "enum": [*ranked_ids, "NONE"]},
                    "answer": {"type": "string", "enum": ["SELECT", "ABSTAIN_FETCH", "INCOMPLETE"]},
                    "evidence_refs": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                    "missing_boundary": {"type": "string", "maxLength": 500},
                },
                "required": ["ranked_ids", "selected_id", "answer", "evidence_refs", "missing_boundary"],
                "additionalProperties": False,
            },
        },
    }


def validate_selection(frontier: Mapping[str, object], arguments: object) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not isinstance(arguments, Mapping):
        return None, {"code": "CAUSAL_FRONTIER_ARGUMENTS_REQUIRED", "detail": "arguments must be an object"}
    required = {"ranked_ids", "selected_id", "answer", "evidence_refs", "missing_boundary"}
    if set(arguments) != required:
        return None, {"code": "CAUSAL_FRONTIER_FIELDS_INVALID", "detail": "frontier fields are closed"}
    offered = {str(item) for item in _items(frontier.get("ranked_ids"))}
    ranked = arguments.get("ranked_ids")
    if not isinstance(ranked, list) or len(ranked) > MAX_RANKED or len(set(ranked)) != len(ranked) or any(item not in offered for item in ranked):
        return None, {"code": "CAUSAL_FRONTIER_RANKING_INVALID", "detail": "ranked_ids must be unique host frontier IDs"}
    selected = arguments.get("selected_id")
    answer = arguments.get("answer")
    refs = arguments.get("evidence_refs")
    missing = arguments.get("missing_boundary")
    if not isinstance(selected, str) or selected not in offered | {"NONE"}:
        return None, {"code": "CAUSAL_FRONTIER_SELECTION_INVALID", "detail": "selected_id is not host-generated"}
    if answer not in {"SELECT", "ABSTAIN_FETCH", "INCOMPLETE"}:
        return None, {"code": "CAUSAL_FRONTIER_ANSWER_INVALID", "detail": "answer is not typed"}
    if not isinstance(refs, list) or len(refs) > 3 or any(not isinstance(item, str) or not item.strip() for item in refs):
        return None, {"code": "CAUSAL_FRONTIER_EVIDENCE_INVALID", "detail": "evidence_refs are bounded strings"}
    if not isinstance(missing, str):
        return None, {"code": "CAUSAL_FRONTIER_BOUNDARY_INVALID", "detail": "missing_boundary must be a string"}
    if answer == "SELECT" and (selected == "NONE" or selected not in ranked):
        return None, {"code": "CAUSAL_FRONTIER_SELECTION_INVALID", "detail": "SELECT requires selected_id in ranked_ids"}
    if answer != "SELECT" and not missing.strip():
        return None, {"code": "CAUSAL_FRONTIER_BOUNDARY_REQUIRED", "detail": "non-SELECT answers require missing_boundary"}
    return {
        "ranked_ids": list(ranked), "selected_id": selected, "answer": answer,
        "evidence_refs": list(refs), "missing_boundary": missing.strip(),
    }, None


def selected_candidate(frontier: Mapping[str, object], candidate_id: str) -> Mapping[str, object] | None:
    return next((item for item in _items(frontier.get("candidates"))
                 if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id), None)


def serialized(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


__all__ = ["MAX_RANKED", "SCHEMA", "TOOL_NAME", "build_frontier", "selected_candidate", "serialized", "tool_schema", "validate_selection"]
