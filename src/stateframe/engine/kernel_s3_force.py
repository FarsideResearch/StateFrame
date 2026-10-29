"""Host-owned S3 TestWriter fallback for stalled IDEALPATH turns.

The fallback deliberately reuses the live ``kernel_testwriter`` validator and
runner.  It drafts only from the public ticket, the admitted S1 spine, and the
S2 localization target; it never fabricates a target or an oracle.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence


SCHEMA = "stateframe-s3force/1"
DEFAULT_STAGE_TURNS = 3
_TRUTHY = frozenset({"1", "on", "true", "yes"})


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether the host-owned S3 fallback is active."""
    source = os.environ if environ is None else environ
    return str(source.get("STATEFRAME_S3FORCE", "off")).strip().lower() in _TRUTHY


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _text(value: object, *, limit: int = 1200) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\x00", " ").strip()
    return value[:limit] if value else ""


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                         separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _path_matches(candidate: Mapping[str, object], path: str) -> bool:
    source_path = _text(candidate.get("source_path"), limit=500).replace("\\", "/").lstrip("./")
    wanted = path.replace("\\", "/").lstrip("./")
    return bool(source_path and wanted and (source_path == wanted
                                            or source_path.endswith("/" + wanted)
                                            or wanted.endswith("/" + source_path)))


def _localized_target(owner: object, packet: Mapping[str, object]) -> tuple[str, Mapping[str, object]]:
    from kernel_reprospec import symbol_target_menu

    targets = symbol_target_menu(packet)
    if not targets:
        raise ValueError("S3FORCE requires a non-empty host target menu")
    context = getattr(owner, "s2_force_target", {})
    context = context if isinstance(context, Mapping) else {}
    path = _text(context.get("path"), limit=500)
    symbol = _text(context.get("symbol"), limit=300)
    ranked: list[tuple[str, Mapping[str, object]]] = []
    for target_id, target in sorted(targets.items()):
        if not isinstance(target, Mapping):
            continue
        if path and not _path_matches(target, path):
            continue
        if symbol and target.get("name") != symbol:
            continue
        ranked.append((target_id, target))
    if not ranked and path:
        ranked = [
            (target_id, target)
            for target_id, target in sorted(targets.items())
            if isinstance(target, Mapping) and _path_matches(target, path)
        ]
    if not ranked:
        raise ValueError("S3FORCE localization does not identify a host target")
    return ranked[0]


def _localized_row(owner: object, span_ids: set[str]) -> tuple[str, Mapping[str, object]]:
    rows = getattr(owner, "rows", {})
    if not isinstance(rows, Mapping):
        raise ValueError("S3FORCE requires the admitted S1 requirement rows")
    candidates: list[tuple[str, Mapping[str, object]]] = []
    for row_id, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        refs = {str(item) for item in _items(row.get("ticket_refs"))}
        spine = row.get("spine")
        if refs & span_ids and isinstance(spine, Mapping) and spine.get("semantic_admission") == "ADMITTED":
            failures = getattr(owner, "admitted_failures", {})
            passes = getattr(owner, "admitted_parent_passes", {})
            settled = (isinstance(failures, Mapping) and str(row_id) in failures) or (
                isinstance(passes, Mapping) and str(row_id) in passes)
            if not settled:
                candidates.append((str(row_id), row))
    if not candidates:
        raise ValueError("S3FORCE has no unsatisfied admitted requirement row")
    cursor = getattr(owner, "s3_force_row_cursor", 0)
    index = int(cursor) % len(candidates)
    selected = candidates[index]
    try:
        setattr(owner, "s3_force_row_cursor", (index + 1) % len(candidates))
    except (AttributeError, TypeError):
        pass
    return selected


def attempt_key(*, stage: str, epoch: int, base_revision: str, row_id: str,
                target_packet: Mapping[str, object], admitted_contract: Mapping[str, object],
                bindings: Mapping[str, object] | None = None) -> str:
    """Identity for one host attempt; unchanged unsupported work is idempotent."""
    return _digest({
        "stage": stage, "epoch": epoch, "base_revision": base_revision,
        "row_id": row_id, "target_packet": target_packet,
        "admitted_contract": admitted_contract, "bindings": bindings or {},
    })


def cached_attempt(owner: object, key: str) -> dict[str, object] | None:
    attempts = getattr(owner, "s3_force_attempts", {})
    result = attempts.get(key) if isinstance(attempts, Mapping) else None
    return copy.deepcopy(dict(result)) if isinstance(result, Mapping) else None


def remember_attempt(owner: object, key: str, result: Mapping[str, object]) -> dict[str, object]:
    attempts = getattr(owner, "s3_force_attempts", None)
    if not isinstance(attempts, dict):
        attempts = {}
        setattr(owner, "s3_force_attempts", attempts)
    stored = copy.deepcopy(dict(result))
    stored["attempt_key"] = key
    attempts[key] = stored
    return copy.deepcopy(stored)


def capability_gap(key: str, *, reason: str, missing: Sequence[object] = ()) -> dict[str, object]:
    return {
        "schema": SCHEMA, "status": "INCOMPLETE", "outcome": "INCOMPLETE",
        "code": "HOST_CAPABILITY_GAP", "attempt_key": key,
        "missing": [str(item) for item in missing][:32],
        "reason": _text(reason, limit=500) or "host execution capability is unavailable",
        "reproduced": False,
    }


def _observation(row: Mapping[str, object]) -> str:
    spine = row.get("spine")
    observables = spine.get("observables") if isinstance(spine, Mapping) else ()
    first = next((item for item in _items(observables) if isinstance(item, Mapping)), None)
    predicates = first.get("predicates") if isinstance(first, Mapping) else ()
    predicate = next((item for item in _items(predicates) if isinstance(item, Mapping)), None)
    return "EXCEPTION" if isinstance(predicate, Mapping) and predicate.get("relation") == "RAISES" else "RETURN"


def select_unsatisfied_row(owner: object, span_ids: Sequence[object]) -> tuple[str, Mapping[str, object]]:
    return _localized_row(owner, {str(item) for item in span_ids})


def build_host_typed_spec(
        owner: object, packet: Mapping[str, object], *, row_id: str | None = None,
) -> dict[str, object]:
    """Build the smallest valid TestWriter proposal from host-owned evidence."""
    from kernel_testwriter import ticket_span_menu

    spans = ticket_span_menu(owner)
    if not spans:
        raise ValueError("S3FORCE requires an exact public ticket span")
    if row_id:
        row = getattr(owner, "rows", {}).get(row_id)
        if not isinstance(row, Mapping):
            raise ValueError("S3FORCE row selection is stale")
    else:
        row_id, row = _localized_row(owner, set(spans))
    target_id, target = _localized_target(owner, packet)
    expected = _text(row.get("expected_observable"), limit=180)
    assertion = ("The localized public behavior matches the ticket expectation: "
                 + (expected or "the expected observable"))[:240]
    recipe = "DIRECT_CALL"
    invocation = "CALL"
    span_id = next(ref for ref in _items(row.get("ticket_refs")) if str(ref) in spans)
    return {
        "target_id": target_id,
        "ticket_span": spans[str(span_id)],
        "fixture_recipe": recipe,
        "invocation": invocation,
        "observations": [_observation(row)],
        "setup_recipe_ids": [],
        "bindings": {
            "argument_bindings": {},
            "observation": {"source": "$1", "path": []},
        },
        "assertion": assertion,
        "_host_context": {
            "requirement_id": row_id,
            "localized_target": {"target_id": target_id, **dict(target)},
            "ticket_span_id": str(span_id),
        },
    }


def public_spec(spec: Mapping[str, object]) -> dict[str, object]:
    """Remove the private selection context before passing a spec to TestWriter."""
    return {key: copy.deepcopy(value) for key, value in spec.items() if not str(key).startswith("_")}


def stage_answer(owner: object, spec: Mapping[str, object], result: Mapping[str, object]) -> dict[str, object]:
    """Create the typed S3 answer that admits a host-observed parent failure."""
    context = spec.get("_host_context")
    context = context if isinstance(context, Mapping) else {}
    requirement_id = _text(result.get("lowered_requirement_ref"), limit=120) or _text(
        context.get("requirement_id"), limit=120)
    receipt = result.get("receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    receipt_id = _text(receipt.get("receipt_id"), limit=200)
    check_hash = _text(result.get("check_hash"), limit=200)
    row = getattr(owner, "rows", {}).get(requirement_id, {})
    wrong_behavior = _text(row.get("wrong_behavior"), limit=1200) if isinstance(row, Mapping) else ""
    refs = [value for value in (receipt_id, check_hash) if value]
    return {
        "row_ids": [requirement_id],
        "scenario": {
            "ticket_span": spec.get("ticket_span"),
            "localized_target": context.get("localized_target", {}),
            "invocation": spec.get("invocation"),
        },
        "oracle": {"assertion": spec.get("assertion"), "observation": spec.get("observations", [])},
        "product_refs": refs,
        "fixtures": [str(spec.get("fixture_recipe", "DIRECT_CALL"))],
        "check_artifact_ref": receipt_id or check_hash,
        "monitors": ["typed_test_writer", "parent_revision"],
        "sanity_control": {"base_revision": getattr(owner, "base_revision", "")},
        "failure_signature": (
            f"typed TestWriter receipt {receipt_id or 'unidentified'} observed BEHAVIOR_FAIL"
        ),
        "plausible_wrong_behavior": wrong_behavior or "the ticket expectation is not preserved",
    }


__all__ = [
    "DEFAULT_STAGE_TURNS", "SCHEMA", "attempt_key", "build_host_typed_spec",
    "cached_attempt", "capability_gap", "enabled", "public_spec",
    "remember_attempt", "select_unsatisfied_row", "stage_answer",
]
