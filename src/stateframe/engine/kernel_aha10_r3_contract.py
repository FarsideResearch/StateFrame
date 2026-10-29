"""Pure integration contracts for the benchmark r3 release boundary.

The existing kernels remain owners for requests, witnesses, delivery, and
completion.  This module gives those owners one small, fail-closed vocabulary
for cross-owner identity and release evidence.  It deliberately performs no
model call, file write, process execution, or completion decision.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Callable


SCHEMA = "stateframe-aha10-r3-contract/1"
UNKNOWN = "UNKNOWN"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE"
HOST_COST_PROFILE_SECONDS = 312.0


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else str(value or "").strip()


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def removal_ledger(
        original: Sequence[object], provisional: Sequence[object], final: Sequence[object],
        *, reasons: Mapping[str, object] | None = None,
        section_reasons: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Account for every original route removed by either reducer.

    The first reducer sees the original menu; the final reducer must still be
    compared to that original, otherwise a later reducer can make a removal
    disappear from telemetry.
    """
    original_names = tuple(dict.fromkeys(_text(item) for item in original if _text(item)))
    provisional_names = set(_text(item) for item in provisional if _text(item))
    final_names = set(_text(item) for item in final if _text(item))
    configured = reasons if isinstance(reasons, Mapping) else {}
    configured_sections = section_reasons if isinstance(section_reasons, Mapping) else {}
    rows: list[dict[str, object]] = []
    for name in original_names:
        if name in final_names:
            continue
        phase = "provisional" if name not in provisional_names else "final"
        rows.append({
            "route": name,
            "phase": phase,
            "reason": _text(configured.get(name)) or (
                "provisional_reducer" if phase == "provisional" else "final_reducer"
            ),
        })
    sections = []
    for name, reason in configured_sections.items():
        if _text(name):
            sections.append({"section": _text(name), "reason": _text(reason) or "reducer"})
    return {
        "schema": SCHEMA,
        "original_routes": list(original_names),
        "provisional_routes": sorted(provisional_names),
        "final_routes": sorted(final_names),
        "removed": rows,
        "removed_count": len(rows),
        "sections": sections,
        "truthful": all(row.get("reason") for row in rows),
    }


def validate_host_receipt(
        receipt: Mapping[str, object], *, execution_store: Mapping[str, object],
        source_unit_handle: str, revision: str, input_value: object,
        observation: object, parent_identity: str, candidate_identity: str,
        clause_id: str, check_identity: str, oracle_identity: str,
        consumers: Sequence[str] = (),
) -> dict[str, object]:
    """Validate a receipt against the host's immutable execution store.

    A digest that a model can compute from its own payload is not an execution
    receipt.  The receipt ID must resolve to the host-stored canonical record,
    and every declared consumer must be reached independently.
    """
    receipt_id = _text(receipt.get("receipt_id"))
    stored = execution_store.get(receipt_id) if isinstance(execution_store, Mapping) else None
    expected_consumers = tuple(dict.fromkeys(_text(item) for item in consumers if _text(item)))
    reached = { _text(item) for item in _sequence(
        receipt.get("reached_consumers", receipt.get("reached_source_units", ()))
    ) if _text(item)}
    fields_match = (
        _text(receipt.get("source_unit_handle")) == source_unit_handle
        and _text(receipt.get("revision")) == revision
        and receipt.get("source_revision") in (None, "", revision)
        and receipt.get("input") == input_value
        and receipt.get("observation") == observation
        and _text(receipt.get("parent_identity")) == parent_identity
        and _text(receipt.get("candidate_identity")) == candidate_identity
        and _text(receipt.get("clause_id")) == clause_id
        and _text(receipt.get("check_identity", receipt.get("check_id"))) == check_identity
        and _text(receipt.get("oracle_identity", receipt.get("oracle_hash"))) == oracle_identity
        and receipt.get("immutable") is True
        and receipt.get("solver_authored") is True
    )
    host_identity_fields = (
        "receipt_id", "source_unit_handle", "revision", "parent_identity",
        "candidate_identity", "clause_id", "check_identity", "oracle_identity",
    )
    store_match = isinstance(stored, Mapping) and all(
        _text(stored.get(key)) and stored.get(key) == receipt.get(key)
        for key in host_identity_fields
    ) and _text(stored.get("status", stored.get("outcome"))).upper() in {
        "EXECUTED", "PASS", "PASSED", "FAIL", "FAILED",
    }
    reachability = bool(expected_consumers) and set(expected_consumers).issubset(reached)
    status = "VALID" if receipt_id and store_match and fields_match and reachability else UNKNOWN
    reason = (
        "host_stored_receipt_valid" if status == "VALID" else
        "receipt_not_in_host_execution_store" if not store_match else
        "declared_consumer_not_reached" if not reachability else
        "receipt_identity_incomplete"
    )
    return {
        "schema": SCHEMA, "status": status, "receipt_id": receipt_id,
        "reason": reason, "reached_consumers": sorted(reached),
        "expected_consumers": list(expected_consumers),
    }


def delivery_reducer(
        selected_units: Sequence[Mapping[str, object]],
        delivered_handles: Sequence[object],
        *, declared_paths: Sequence[object] = (),
) -> dict[str, object]:
    """Classify selected source units after the final provider reducer."""
    selected = tuple(item for item in selected_units if isinstance(item, Mapping))
    selected_handles = tuple(dict.fromkeys(
        _text(item.get("handle")) for item in selected if _text(item.get("handle"))
    ))
    delivered = { _text(item) for item in delivered_handles if _text(item) }
    missing = [handle for handle in selected_handles if handle not in delivered]
    paths = tuple(dict.fromkeys(_text(item) for item in declared_paths if _text(item)))
    selected_paths = {_text(item.get("path")) for item in selected if _text(item.get("path"))}
    missing_paths = sorted(set(paths) - selected_paths) if paths else []
    status = COMPLETE if selected_handles and not missing and not missing_paths else (
        PARTIAL if selected_handles or missing_paths else UNKNOWN
    )
    return {
        "schema": SCHEMA, "status": status,
        "selected_unit_handles": list(selected_handles),
        "delivered_unit_handles": sorted(delivered & set(selected_handles)),
        "uncovered_unit_handles": missing,
        "missing_paths": missing_paths,
        "continuation_required": status != COMPLETE,
    }


def advisory_projection(
        *, revision: str, menu_identity: str, route_identities: Sequence[str],
        prior: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Project ownerless DECIDE2 state without executing advice."""
    routes = tuple(dict.fromkeys(_text(item) for item in route_identities if _text(item)))
    state = {"revision": revision, "menu_identity": menu_identity, "routes": list(routes)}
    state_key = _digest(state)
    prior_key = _text(prior.get("state_key")) if isinstance(prior, Mapping) else ""
    if len(routes) <= 1:
        reason = "single_route_deterministic" if routes else "no_admitted_route"
        status = "SKIPPED"
    elif prior_key == state_key:
        reason, status = "unchanged_state_deduplicated", "DEDUPLICATED"
    else:
        reason, status = "ADVISORY_ONLY", "OPEN"
    return {
        "schema": SCHEMA, "status": status, "reason": reason,
        "state_key": state_key, "effective_manifest": state,
        "advice_executed": False,
    }


def host_cost_key(*, revision: str, content_digest: str, environment_identity: str) -> str:
    """Return the complete invalidation key for the bounded host-cost memo."""
    return _digest({"revision": revision, "content": content_digest,
                    "environment": environment_identity})


def memoized_host_cost(
        cache: dict[str, object], *, revision: str, content_digest: str,
        environment_identity: str, measure: Callable[[], object],
) -> dict[str, object]:
    key = host_cost_key(revision=revision, content_digest=content_digest,
                        environment_identity=environment_identity)
    if key in cache:
        return {"schema": SCHEMA, "status": "CACHED", "key": key, "value": cache[key]}
    value = measure()
    if isinstance(value, Mapping):
        value = dict(value)
        raw_seconds = value.get("seconds", value.get("elapsed_seconds"))
        try:
            seconds = float(raw_seconds)
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None:
            value["status"] = "PASS" if 0.0 <= seconds <= HOST_COST_PROFILE_SECONDS else "HOST_COST_EXCEEDED"
            value["budget_seconds"] = HOST_COST_PROFILE_SECONDS
    else:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None:
            value = {
                "seconds": seconds,
                "budget_seconds": HOST_COST_PROFILE_SECONDS,
                "status": "PASS" if 0.0 <= seconds <= HOST_COST_PROFILE_SECONDS else "HOST_COST_EXCEEDED",
            }
    cache[key] = value
    return {"schema": SCHEMA, "status": "MEASURED", "key": key, "value": value}


def provider_boundary(
        constructions: Sequence[Mapping[str, object]], *, reasoning: object,
        shared_policy: Mapping[str, object],
) -> dict[str, object]:
    """Prove effective provider settings across every shared client construction."""
    rows = [dict(item) for item in constructions if isinstance(item, Mapping)]
    policy_digest = _digest(dict(shared_policy))
    effective = str(reasoning).casefold() in {"off", "false", "0", "none", ""}
    consistent = effective and bool(rows) and all(
        bool(row.get("reasoning_off")) is True
        and _digest(row.get("shared_policy", {})) == policy_digest
        for row in rows
    )
    return {
        "schema": SCHEMA, "status": "VALID" if consistent else UNKNOWN,
        "reasoning_off": effective, "client_count": len(rows),
        "shared_policy_digest": policy_digest,
        "construction_ids": [_text(row.get("id")) for row in rows],
    }


def scope_trace(changes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compose typed boolean/enum/string/numeric authorization evidence."""
    rows = [dict(item) for item in changes if isinstance(item, Mapping)]
    types = {_text(row.get("type")).casefold() for row in rows}
    required = {"boolean", "enum", "string", "numeric"}
    authorized = all(row.get("authorized") is True for row in rows)
    return {
        "schema": SCHEMA, "status": "VALID" if required.issubset(types) and authorized else UNKNOWN,
        "types": sorted(types), "authorized": authorized,
        "changes": rows,
    }


def compose_release_proof(**evidence: object) -> dict[str, object]:
    """Reduce one coherent offline snapshot to a truthful release disposition."""
    required = (
        "budget", "witness", "authored_check", "delivery", "settlement", "smoke",
        "provider", "cost", "scope", "advisory", "source_config_hash",
    )
    missing = [name for name in required if name not in evidence]
    if not _text(evidence.get("source_config_hash")):
        missing.append("source_config_hash")
    if missing:
        return {"schema": SCHEMA, "status": UNKNOWN, "disposition": "OPEN",
                "reason": "proof_fields_missing", "missing": missing}
    statuses = {name: _text(value.get("status")) if isinstance(value, Mapping) else ""
                for name, value in evidence.items()}
    def positive_status(name: str) -> bool:
        row = evidence[name]
        if not isinstance(row, Mapping):
            return False
        status = _text(row.get("status")).upper()
        if name == "cost" and status in {"MEASURED", "CACHED"}:
            value = row.get("value")
            return isinstance(value, Mapping) and _text(value.get("status")).upper() == "PASS"
        return status in {"VALID", "COMPLETE", "READY", "MEASURED", "CACHED", "PASS", "SUPPORTED"}

    positive = all(positive_status(name)
                   for name in required if name not in {"settlement", "source_config_hash"})
    settlement = evidence["settlement"]
    settled = isinstance(settlement, Mapping) and settlement.get("certification") == "CERTIFIED"
    disposition = "PASS" if positive and settled else "UNCERTIFIED_SUBMIT" if (
        isinstance(settlement, Mapping) and settlement.get("artifact") == "STORED"
    ) else "OPEN"
    return {
        "schema": SCHEMA, "status": "VALID" if disposition == "PASS" else UNKNOWN,
        "disposition": disposition, "source_config_hash": _text(evidence["source_config_hash"]),
        "statuses": statuses,
    }


__all__ = [
    "COMPLETE", "HOST_COST_PROFILE_SECONDS", "PARTIAL", "SCHEMA", "UNKNOWN", "advisory_projection",
    "compose_release_proof", "delivery_reducer", "host_cost_key", "memoized_host_cost",
    "provider_boundary", "removal_ledger", "scope_trace", "validate_host_receipt",
]
