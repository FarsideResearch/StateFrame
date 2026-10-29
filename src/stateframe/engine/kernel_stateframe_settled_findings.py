"""Small, typed projections of findings reused by the visible history arm.

The catalog is deliberately static and evidence-linked.  It is an offline
replay corpus, not a hidden-test oracle and not a completion authority.  A
caller must opt in to both the mechanism and a named pair; without either
selection this module leaves the frame untouched.
"""
from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Mapping, Sequence

from kernel_context import token_estimate


MODE_ENV = "STATEFRAME_SETTLED_FINDINGS"
PAIR_ENV = "STATEFRAME_SETTLED_FINDINGS_PAIR"
SCHEMA = "stateframe-settled-findings/1"
FRAME_FIELD = "settled_findings"
SOURCE_REVISION = "r43:d1d1dd"
MIN_PAIR_TOKENS = 220
MAX_PAIR_TOKENS = 440
BLOCK_HEADER = f"[Settled findings | historical evidence only | schema={SCHEMA}]"

_KINDS = frozenset({
    "root_cause", "test_oracle", "failure", "decision", "semantic_contract",
})
_REQUIRED_FINDING_FIELDS = frozenset({
    "id", "kind", "claim", "scope", "status", "evidence_refs",
})


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict opt-in switch value."""
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _text(value: object, *, field: str, limit: int = 240) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"settled finding {field} must be a non-empty string")
    return " ".join(value.split())[:limit]


def _strings(value: object, *, field: str, limit: int = 8) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"settled finding {field} must be a list of strings")
    result: list[str] = []
    for item in value:
        text = _text(item, field=field, limit=240)
        if text not in result:
            result.append(text)
    return result[:limit]


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("settled finding values must be strict JSON")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(child) for child in value]
    raise ValueError("settled finding values must be JSON-compatible")


def _token_estimate(value: object) -> int:
    return token_estimate(value)


def _assert_no_free_form_reasoning(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"reasoning", "rationale"}:
                raise ValueError("settled findings cannot carry free-form reasoning")
            _assert_no_free_form_reasoning(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _assert_no_free_form_reasoning(child)


def _finding(*, ident: str, kind: str, claim: str, scope: str,
             evidence_refs: Sequence[str], **detail: object) -> dict[str, object]:
    # The pair-level source_revision binds every row, avoiding repeated budget cost.
    return {
        "id": ident,
        "kind": kind,
        "claim": claim,
        "scope": scope,
        "status": "SETTLED",
        "evidence_refs": list(evidence_refs),
        **detail,
    }



_PAIR_CATALOG: dict[str, dict[str, object]] = {}
PAIR_IDS: tuple[str, ...] = tuple(_PAIR_CATALOG)


def pair_for_owner(owner: object | None = None,
                   environ: Mapping[str, str] | None = None) -> str | None:
    """Select a named offline pair without guessing from free-form task text."""
    source = os.environ if environ is None else environ
    configured = (source.get(PAIR_ENV, "") or "").strip()
    if configured:
        return configured
    row = getattr(owner, "task_row", {}) if owner is not None else {}
    if isinstance(row, Mapping):
        for key in ("history_pair_id", "settled_findings_pair", "pair_id"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("history_pair_id", "settled_findings_pair"):
        value = getattr(owner, key, None) if owner is not None else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def replay_pair(pair_id: str) -> dict[str, object]:
    """Return a detached, validated projection for one visible history pair."""
    if pair_id not in _PAIR_CATALOG:
        raise KeyError(f"Unknown settled-findings pair: {pair_id}")
    payload = copy.deepcopy(_PAIR_CATALOG[pair_id])
    validate_projection(payload)
    return payload


def project_frame(frame: dict, *, pair_id: str | None = None) -> dict:
    """Attach one named pair only when the mechanism is explicitly enabled."""
    if mode() == "off" or not pair_id or pair_id not in _PAIR_CATALOG:
        return frame
    projected = dict(frame)
    projected[FRAME_FIELD] = replay_pair(pair_id)
    return projected


def render_block(payload: Mapping[str, object]) -> str:
    """Render the typed payload as one auditable model-facing block."""
    validate_projection(payload)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return BLOCK_HEADER + "\n" + body


def validate_projection(payload: Mapping[str, object]) -> None:
    """Validate the producer/consumer contract before it reaches a request."""
    if not isinstance(payload, Mapping):
        raise ValueError("settled findings payload must be a mapping")
    if payload.get("schema") != SCHEMA:
        raise ValueError("settled findings schema mismatch")
    if payload.get("source_revision") != SOURCE_REVISION:
        raise ValueError("settled findings source_revision mismatch")
    pair_id = payload.get("pair_id")
    if not isinstance(pair_id, str) or pair_id not in _PAIR_CATALOG:
        raise ValueError("settled findings pair_id is unknown")
    if payload.get("status") != "SETTLED":
        raise ValueError("settled findings status must be SETTLED")
    if payload.get("authority") != "historical_evidence_only":
        raise ValueError("settled findings authority mismatch")
    for outcome_field in ("history_outcome", "frame_outcome"):
        if payload.get(outcome_field) not in {"PASS", "FAIL"}:
            raise ValueError(f"settled findings {outcome_field} is invalid")
    source_refs = _strings(payload.get("source_refs"), field="source_refs")
    if not source_refs:
        raise ValueError("settled findings source_refs must not be empty")
    findings = payload.get("findings")
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes, bytearray)):
        raise ValueError("settled findings findings must be a list")
    if not findings:
        raise ValueError("settled findings findings must not be empty")
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise ValueError("settled finding row must be a mapping")
        _assert_no_free_form_reasoning(finding)
        missing = _REQUIRED_FINDING_FIELDS.difference(finding)
        if missing:
            raise ValueError("settled finding missing " + ", ".join(sorted(missing)))
        if finding.get("kind") not in _KINDS:
            raise ValueError("settled finding kind is unknown")
        if finding.get("status") != "SETTLED":
            raise ValueError("settled finding status must be SETTLED")
        if not _strings(finding.get("evidence_refs"), field="evidence_refs"):
            raise ValueError("settled finding evidence_refs must not be empty")
        for forbidden in ("reasoning", "rationale"):
            if forbidden in finding:
                raise ValueError("settled findings cannot carry free-form reasoning")
        kind = finding["kind"]
        if kind == "test_oracle" and not isinstance(finding.get("oracle"), Mapping):
            raise ValueError("test_oracle requires typed oracle detail")
        if kind == "failure" and not _strings(finding.get("invalidates"), field="invalidates"):
            raise ValueError("failure requires invalidates")
        if kind == "decision" and not isinstance(finding.get("decision"), Mapping):
            raise ValueError("decision requires typed decision detail")
        if kind == "semantic_contract" and not isinstance(finding.get("semantic_contract"), Mapping):
            raise ValueError("semantic_contract requires typed detail")
    json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    estimate = _token_estimate(payload)
    if not MIN_PAIR_TOKENS <= estimate <= MAX_PAIR_TOKENS:
        raise ValueError(
            f"settled findings pair {pair_id} is {estimate} tokens; "
            f"expected {MIN_PAIR_TOKENS}-{MAX_PAIR_TOKENS}"
        )


__all__ = [
    "BLOCK_HEADER", "FRAME_FIELD", "MAX_PAIR_TOKENS", "MIN_PAIR_TOKENS", "MODE_ENV",
    "PAIR_ENV", "PAIR_IDS", "SCHEMA", "SOURCE_REVISION", "mode", "pair_for_owner",
    "project_frame", "render_block", "replay_pair", "validate_projection",
]
