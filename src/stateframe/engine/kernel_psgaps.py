"""Host-owned live adapters for the screened PSGAPS survivors.

The request qualification package is deliberately not imported from the release runtime.
This small port keeps the release self-contained while preserving the survivor
contracts: finite invariant challenges, failure-preserving typed reduction,
and evidence-keyed host routing.  Every result is advisory telemetry; this
module never changes a menu, executes a check, edits a workspace, or settles
completion.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence


SCHEMA = "stateframe-pskernel/1"
FLAGS = {
    "invariants": "STATEFRAME_PS_INVARIANTS",
    "minimize": "STATEFRAME_PS_MINIMIZE",
    "router": "STATEFRAME_PS_ROUTER",
}

_PROPERTY_RULES = {
    "invariant-idempotency": "IDEMPOTENT",
    "invariant-role-filter": "ROLE_FILTER",
    "invariant-order": "ORDER",
    "invariant-round-trip": "ROUND_TRIP",
    "invariant-resource-bound": "RESOURCE_BOUND",
    "invariant-custom-registered": "CUSTOM_REGISTERED",
}
_PROPERTY_SPECS = {
    "invariant-idempotency": {
        "req_ref": "public:preserve-repeat",
        "relation": "IDEMPOTENT",
        "domain_preconditions": ["repeatable operation"],
        "generator_ref": "public-replay-generator",
        "oracle_ref": "public-idempotency-oracle",
    },
    "invariant-role-filter": {
        "req_ref": "public:role-boundary",
        "relation": "ROLE_FILTER",
        "domain_preconditions": ["role-scoped output"],
        "generator_ref": "public-role-generator",
        "oracle_ref": "public-role-oracle",
    },
    "invariant-order": {
        "req_ref": "public:lifecycle-order",
        "relation": "ORDER",
        "domain_preconditions": ["ordered lifecycle or registration"],
        "generator_ref": "public-event-generator",
        "oracle_ref": "public-order-oracle",
    },
    "invariant-round-trip": {
        "req_ref": "public:serialized-boundary",
        "relation": "ROUND_TRIP",
        "domain_preconditions": ["serialized value boundary"],
        "generator_ref": "public-round-trip-generator",
        "oracle_ref": "public-round-trip-oracle",
    },
    "invariant-resource-bound": {
        "req_ref": "public:resource-bound",
        "relation": "RESOURCE_BOUND",
        "domain_preconditions": ["bounded time or capacity"],
        "generator_ref": "public-budget-generator",
        "oracle_ref": "public-resource-oracle",
    },
    "invariant-custom-registered": {
        "req_ref": "public:registered-predicate",
        "relation": "CUSTOM_REGISTERED",
        "domain_preconditions": ["public contract registers a predicate"],
        "generator_ref": "public-contract-generator",
        "oracle_ref": "public-registered-oracle",
    },
}
_METHOD_SPECS = (
    ("reobserve", "read_public_state"),
    ("clarify", "ask_clarification"),
    ("readiness", "run_registered_probe"),
    ("analogy", "read_verified_case"),
    ("experiment", "run_registered_probe"),
    ("localize", "read_public_graph"),
    ("hypothesis_test", "run_registered_probe"),
    ("minimize", "run_disposable_reducer"),
    ("reframe", "read_public_state"),
    ("information_value", "run_registered_probe"),
    ("invariants", "run_registered_probe"),
    ("integration", "run_registered_probe"),
    ("stop", ""),
)
# Benchmark task cases are not part of the engine source: a run supplies its case explicitly via
# runtime.ps_minimize_case; otherwise there is none.
_REPRO_CASES: dict[str, dict[str, object]] = {}
_TRUTHY = frozenset({"1", "on", "true", "yes"})


def enabled(name: str, environ: Mapping[str, str] | None = None) -> bool:
    """Return one strict opt-in switch; unknown values fail closed by raising."""
    flag = FLAGS.get(name, name)
    if flag not in FLAGS.values():
        raise ValueError(f"unknown PSGAPS mechanism {name!r}")
    source = os.environ if environ is None else environ
    return str(source.get(flag, "off")).strip().lower() in _TRUTHY


def _digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _text(value: object, limit: int = 240) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\x00", " ").replace("\n", " ").strip()
    return value[:limit]


def _owner(runtime: object) -> object | None:
    getter = getattr(runtime, "_idealpath_owner", None)
    if callable(getter):
        try:
            owner = getter()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            owner = None
        if owner is not None:
            return owner
    return getattr(runtime, "ideal_path_owner", None)


def _emit(runtime: object, kind: str, payload: Mapping[str, object]) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if not callable(emit):
        raise RuntimeError("PSKERNEL requires the live runtime journal")
    emit(kind, producer="kernel_psgaps", event_schema=SCHEMA,
         payload=dict(payload), dispatch=False)


def _ticket(runtime: object) -> str:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    if not isinstance(row, Mapping):
        row = {}
    for key in ("problem_statement", "ticket_text", "public_brief", "brief", "msg", "symptom"):
        value = _text(row.get(key), 1200)
        if value:
            return value
    return ""


def _frame_mapping(frame: object) -> Mapping[str, object]:
    return frame if isinstance(frame, Mapping) else {}


def _receipt_rows(runtime: object) -> tuple[Mapping[str, object], ...]:
    rows = getattr(runtime, "test_receipts", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in rows if isinstance(item, Mapping))


def _applicable_properties(ticket: str) -> tuple[str, ...]:
    words = ticket.casefold()
    selected: list[str] = []
    if any(token in words for token in ("unchanged", "duplicate", "retry", "same")):
        selected.append("invariant-idempotency")
    if any(token in words for token in ("role", "learner", "chat", "content")):
        selected.append("invariant-role-filter")
    if any(token in words for token in ("before", "immediately", "import", "order", "first")):
        selected.append("invariant-order")
    if any(token in words for token in ("transport", "content", "field")):
        selected.append("invariant-round-trip")
    if any(token in words for token in ("timeout", "lease", "pool", "budget", "connection")):
        selected.append("invariant-resource-bound")
    return tuple(dict.fromkeys(selected)) or ("invariant-custom-registered",)


def _holds(property_id: str, observation: Mapping[str, object]) -> bool:
    relation = _PROPERTY_RULES[property_id]
    if relation == "IDEMPOTENT":
        return observation.get("repeated_digest_equal") is True
    if relation == "ROLE_FILTER":
        roles = observation.get("roles")
        allowed = observation.get("allowed_roles")
        return (
            isinstance(roles, list)
            and isinstance(allowed, list)
            and set(roles).issubset(set(allowed))
            and observation.get("forbidden_role_present") is not True
        )
    if relation == "ORDER":
        events = observation.get("events")
        expected = observation.get("expected_prefix")
        return isinstance(events, list) and isinstance(expected, list) and events[:len(expected)] == expected
    if relation == "ROUND_TRIP":
        return observation.get("round_trip_equal") is True
    if relation == "RESOURCE_BOUND":
        duration = observation.get("duration_ms")
        limit = observation.get("max_duration_ms")
        inflight = observation.get("max_inflight")
        inflight_limit = observation.get("max_inflight_limit")
        duration_ok = isinstance(duration, (int, float)) and isinstance(limit, (int, float)) and duration <= limit
        inflight_ok = inflight is None or (
            isinstance(inflight, (int, float))
            and isinstance(inflight_limit, (int, float))
            and inflight <= inflight_limit
        )
        return duration_ok and inflight_ok
    return observation.get("oracle_pass") is True


def _wrong_observation(property_id: str) -> dict[str, object]:
    relation = _PROPERTY_RULES[property_id]
    if relation == "IDEMPOTENT":
        return {"repeated_digest_equal": False}
    if relation == "ROLE_FILTER":
        return {"roles": ["learner", "admin"], "allowed_roles": ["learner"], "forbidden_role_present": True}
    if relation == "ORDER":
        return {"events": ["construct", "configure"], "expected_prefix": ["configure", "construct"]}
    if relation == "ROUND_TRIP":
        return {"round_trip_equal": False}
    if relation == "RESOURCE_BOUND":
        return {"duration_ms": 30, "max_duration_ms": 20, "max_inflight": 3, "max_inflight_limit": 2}
    return {"oracle_pass": False}


def _candidate_observation(runtime: object, frame: Mapping[str, object], property_id: str) -> dict[str, object]:
    override = getattr(runtime, "ps_invariant_observation", None)
    if isinstance(override, Mapping):
        return dict(override)
    # No host observation means no claimed proof.  The empty values are
    # deliberately failing/unknown rather than a manufactured green result.
    relation = _PROPERTY_RULES[property_id]
    if relation == "IDEMPOTENT":
        rows = _receipt_rows(runtime)
        return {"repeated_digest_equal": len(rows) >= 2 and _digest(rows[-1]) == _digest(rows[-2])}
    if relation == "ROLE_FILTER":
        return {"roles": [], "allowed_roles": [], "forbidden_role_present": True}
    if relation == "ORDER":
        return {"events": [], "expected_prefix": ["configure", "construct"]}
    if relation == "ROUND_TRIP":
        return {"round_trip_equal": False}
    if relation == "RESOURCE_BOUND":
        return {"duration_ms": math.inf, "max_duration_ms": 0}
    return {"oracle_pass": False}


def invariant_checkpoint(runtime: object, frame: object) -> dict[str, object] | None:
    if not enabled("invariants"):
        return None
    frame_map = _frame_mapping(frame)
    ticket = _ticket(runtime)
    allowed = _applicable_properties(ticket)
    requested = getattr(runtime, "ps_invariant_property", allowed[0])
    property_id = requested if isinstance(requested, str) and requested in allowed else allowed[0]
    candidate = _candidate_observation(runtime, frame_map, property_id)
    wrong = _wrong_observation(property_id)
    candidate_passes = _holds(property_id, candidate)
    wrong_rejected = not _holds(property_id, wrong)
    challenge = {
        "property_id": property_id,
        "candidate_passes": int(candidate_passes),
        "wrong_controls": 1,
        "wrong_rejections": int(wrong_rejected),
        "false_alarms": int(not candidate_passes),
        "inapplicable": False,
        "gate_passed": candidate_passes and wrong_rejected,
    }
    payload = {
        "status": "FIRED",
        "authority": "host",
        "ticket_digest": _digest(ticket),
        "allowed_property_ids": list(allowed),
        "selected_property": property_id,
        "property_spec": dict(_PROPERTY_SPECS[property_id]),
        "challenge_plan": {
            "candidate_ref": "host:current-observation",
            "property_ref": _PROPERTY_SPECS[property_id]["req_ref"],
            "max_cases": 1,
            "wrong_control_refs": ["host:independent-wrong-control"],
        },
        "challenge": challenge,
        "observation_digest": _digest(candidate),
    }
    _emit(runtime, "ps_invariants_fired", payload)
    _emit(runtime, "ps_counterexamples_fired", {
        "status": "REJECTED_WRONG_CONTROL" if wrong_rejected else "WRONG_CONTROL_NOT_REJECTED",
        "property_id": property_id,
        "wrong_observation_digest": _digest(wrong),
        "independent_control_count": 1,
        "rejected_count": int(wrong_rejected),
    })
    runtime._ps_invariant_receipt = payload
    return payload


def _frozen_parent_authority(runtime: object, case: Mapping[str, object]) -> tuple[bool, str]:
    owner = _owner(runtime)
    if str(getattr(owner, "stage", "")) != "S6":
        return False, "stage_not_s6"
    requirement = str(case.get("requirement_ref", ""))
    frozen = getattr(owner, "frozen_repros", {})
    admitted = getattr(owner, "admitted_failures", {})
    if not isinstance(frozen, Mapping) or not isinstance(admitted, Mapping):
        return False, "frozen_parent_state_missing"
    record = frozen.get(requirement)
    parent_receipt = record.get("parent_receipt_id") if isinstance(record, Mapping) else None
    if not parent_receipt or admitted.get(requirement) != parent_receipt:
        return False, "parent_failure_receipt_not_admitted"
    return True, str(parent_receipt)


def _reduction_case(runtime: object) -> dict[str, object] | None:
    explicit = getattr(runtime, "ps_minimize_case", None)
    if isinstance(explicit, Mapping):
        case = dict(explicit)
    else:
        owner = _owner(runtime)
        task_id = str(getattr(getattr(owner, "task_row", {}), "get", lambda *_: "")("task_id", ""))
        source = _REPRO_CASES.get(task_id)
        if not source:
            return None
        case = dict(source)
        case["requirement_ref"] = str(getattr(owner, "current_requirement_id", ""))
    units = case.get("units")
    protected = case.get("protected", case.get("protected_preconditions"))
    signature = case.get("failure_signature")
    requirement = case.get("requirement_ref")
    if (
        not isinstance(units, Sequence) or isinstance(units, (str, bytes, bytearray))
        or not isinstance(protected, Sequence) or isinstance(protected, (str, bytes, bytearray))
        or not isinstance(signature, str) or not signature
        or not isinstance(requirement, str) or not requirement
    ):
        return None
    units_tuple = tuple(str(item) for item in units)
    protected_tuple = tuple(str(item) for item in protected)
    if (
        not units_tuple
        or not protected_tuple
        or len(set(units_tuple)) != len(units_tuple)
        or len(set(protected_tuple)) != len(protected_tuple)
        or not set(protected_tuple).issubset(units_tuple)
    ):
        return None
    try:
        max_runs = max(1, min(int(case.get("max_runs", 12)), 12))
    except (TypeError, ValueError):
        return None
    return {
        "requirement_ref": requirement,
        "boundary": _text(case.get("boundary"), 160),
        "failing_receipt": _text(case.get("failing_receipt"), 160),
        "reducible_dimension": _text(case.get("reducible_dimension"), 40) or "EVENTS",
        "max_runs": max_runs,
        "units": units_tuple,
        "protected": protected_tuple,
        "failure_signature": signature,
    }


def _observe_reduction(runtime: object, case: Mapping[str, object], units: tuple[str, ...]) -> tuple[str, str | None]:
    evaluator = getattr(runtime, "ps_minimize_evaluator", None)
    if callable(evaluator):
        result = evaluator(units)
        if isinstance(result, Mapping):
            return str(result.get("outcome", "INVALID_SETUP")), result.get("signature") if isinstance(result.get("signature"), str) else None
        if isinstance(result, Sequence) and len(result) >= 1:
            return str(result[0]), str(result[1]) if len(result) > 1 and result[1] is not None else None
        raise TypeError("PSKERNEL reducer evaluator must return a mapping or pair")
    if set(case["protected"]).issubset(units):
        return "SAME_FAILURE", str(case["failure_signature"])
    return "PASS", None


def _reduce(runtime: object, case: Mapping[str, object], max_runs: int = 32) -> dict[str, object]:
    original = tuple(case["units"])
    protected = frozenset(case["protected"])
    signature = str(case["failure_signature"])
    cache: dict[str, tuple[str, str | None]] = {}
    trials: list[dict[str, object]] = []

    def observe(units: tuple[str, ...]) -> tuple[str, str | None]:
        key = _digest(units)
        if key not in cache:
            if len(cache) >= max_runs:
                return "INVALID_SETUP", None
            cache[key] = _observe_reduction(runtime, case, units)
        return cache[key]

    baseline_outcome, baseline_signature = observe(original)
    trials.append({"trial_id": "baseline", "removed_count": 0, "remaining_count": len(original), "outcome": baseline_outcome, "signature": baseline_signature})
    if baseline_outcome != "SAME_FAILURE" or baseline_signature != signature:
        return {
            "status": "REJECTED_BASELINE",
            "same_failure": False,
            "original_units": list(original),
            "minimized_units": list(original),
            "trials": trials,
            "cache_hits": 0,
            "guarantee": "no reduction: frozen parent failure was not reproduced",
        }

    current = original
    granularity = 2
    cache_hits = 0
    while len(current) >= 2 and len(cache) < max_runs:
        chunk_size = max(1, math.ceil(len(current) / granularity))
        chunks = tuple(current[index:index + chunk_size] for index in range(0, len(current), chunk_size))
        reduced = False
        for index, chunk in enumerate(chunks):
            complement = tuple(item for part_index, part in enumerate(chunks) if part_index != index for item in part)
            if not protected.issubset(complement):
                continue
            before = len(cache)
            outcome, observed_signature = observe(complement)
            cache_hits += int(before == len(cache))
            trials.append({
                "trial_id": f"trial-{len(trials):03d}",
                "removed_count": len(current) - len(complement),
                "remaining_count": len(complement),
                "outcome": outcome,
                "signature": observed_signature,
            })
            if outcome == "SAME_FAILURE" and observed_signature == signature:
                current = complement
                granularity = max(granularity - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if granularity >= len(current):
            break
        granularity = min(len(current), granularity * 2)

    return {
        "status": "FIRED",
        "same_failure": True,
        "original_units": list(original),
        "minimized_units": list(current),
        "reduced_count": len(original) - len(current),
        "trials": trials,
        "cache_hits": cache_hits,
        "guarantee": "same declared failure signature retained; 1-minimal under tested complements",
    }


def minimize_checkpoint(runtime: object) -> dict[str, object] | None:
    if not enabled("minimize"):
        return None
    case = _reduction_case(runtime)
    if case is None:
        return None
    authority, parent_receipt = _frozen_parent_authority(runtime, case)
    if not authority:
        _emit(runtime, "ps_minimize_skipped", {
            "status": "SKIPPED",
            "reason": parent_receipt,
            "authority": "S6_frozen_parent_check",
        })
        return None
    result = _reduce(runtime, case, max_runs=int(case["max_runs"]))
    if not result["same_failure"]:
        # Never convert an S6 parent failure into a passing repro or replace its
        # frozen check; an unfaithful baseline is rejected and discarded.
        _emit(runtime, "ps_minimize_rejected", {
            "status": result["status"],
            "authority": "S6_frozen_parent_check",
            "parent_receipt_id": parent_receipt,
        })
        return None
    payload = {
        **result,
        "authority": "S6_frozen_parent_check",
        "parent_receipt_id": parent_receipt,
        "failing_receipt": parent_receipt,
        "parent_check_untouched": True,
        "boundary": case["boundary"],
        "reducible_dimension": case["reducible_dimension"],
        "max_runs": case["max_runs"],
        "failure_signature": case["failure_signature"],
        "requirement_ref": case["requirement_ref"],
    }
    _emit(runtime, "ps_minimize_fired", payload)
    runtime._ps_minimize_receipt = payload
    return payload


def _route(runtime: object, frame: Mapping[str, object]) -> dict[str, object]:
    ticket = _ticket(runtime)
    owner = _owner(runtime)
    stage = _text(getattr(owner, "stage", "S1"), 8) or "S1"
    try:
        target = _text(runtime._target_relative(), 160) if callable(getattr(runtime, "_target_relative", None)) else ""
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        target = ""
    refs = {"ticket:symptom", "ticket:focus"} if ticket else set()
    binding = {}
    binding_getter = getattr(runtime, "binding", None)
    if callable(binding_getter):
        try:
            candidate_binding = binding_getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            candidate_binding = {}
        if isinstance(candidate_binding, Mapping):
            binding = candidate_binding
    if isinstance(binding, Mapping) and any(
        binding.get(key) for key in ("product_revision", "artifact", "source_digest", "source_ref")
    ):
        refs.add("source:digest")
    configured_evidence = getattr(runtime, "ps_router_evidence_refs", refs)
    host_evidence = (
        {str(item) for item in configured_evidence}
        if isinstance(configured_evidence, (set, frozenset, list, tuple))
        else set(refs)
    )
    required = {"ticket:symptom", "ticket:focus", "source:digest"}
    missing = sorted(required - host_evidence)
    receipt_rows = _receipt_rows(runtime)
    latest = receipt_rows[-1] if receipt_rows else {}
    outcome = str(latest.get("outcome", latest.get("result_class", ""))).upper()
    if outcome in {"ENVIRONMENT", "ENV_BLOCKED", "RESOURCE_LIMIT"}:
        observation = "ENV_BLOCKED"
    elif outcome in {"BEHAVIOR_FAIL", "RED", "FAILED", "FAIL"}:
        observation = "BEHAVIOR_FAIL"
    elif outcome in {"PASS", "PASSED", "GREEN", "OK"}:
        observation = "PASS_NO_REPRO"
    else:
        observation = "NONE"
    invariant = getattr(runtime, "_ps_invariant_receipt", {})
    minimization = getattr(runtime, "_ps_minimize_receipt", {})
    violated = ("public:invariant",) if isinstance(invariant, Mapping) and not invariant.get("challenge", {}).get("gate_passed", False) else ()
    configured_capabilities = getattr(runtime, "ps_router_capabilities", {
        "read_public_state", "read_public_graph", "run_registered_probe", "run_disposable_reducer",
    })
    capabilities = (
        set(configured_capabilities)
        if isinstance(configured_capabilities, (set, frozenset, list, tuple))
        else set()
    )
    eligibility: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for method, required_capability in _METHOD_SPECS:
        eligible = not required_capability or required_capability in capabilities
        row = {"method": method, "version": "1", "eligible": eligible,
               "reason": "eligible" if eligible else "missing_capability"}
        (eligibility if eligible else rejected).append(row)
    stale = bool(frame.get("stale_binding", getattr(runtime, "ps_router_stale_binding", False)))
    unclear = any(token in ticket.casefold() for token in ("unclear", "ambiguous"))
    rivals = int(frame.get("live_hypothesis_count", getattr(runtime, "ps_router_hypothesis_count", 1)) or 0) >= 2
    capacity_conflict = "every" in ticket.casefold() and "shared" in ticket.casefold()
    desired = "abstain"
    reason = "no_eligible_evidence_procedure"
    if missing:
        reason = "forged_or_missing_evidence"
    elif stale:
        desired, reason = "reobserve", "stale_binding"
    elif unclear:
        desired, reason = "clarify", "goal_unclear_or_conflicting"
    elif observation == "ENV_BLOCKED":
        desired, reason = "readiness", "execution_unreachable"
    elif observation == "PASS_NO_REPRO":
        desired, reason = "experiment", "faithful_reproduction_missing"
    elif not target:
        desired, reason = "localize", "owner_not_exact"
    elif rivals and observation == "BEHAVIOR_FAIL":
        desired, reason = "hypothesis_test", "rival_hypotheses_with_probe"
    elif isinstance(minimization, Mapping) and minimization.get("same_failure"):
        desired, reason = "minimize", "stable_reducible_failure"
    elif frame.get("no_progress_receipt"):
        desired, reason = "reframe", "stable_no_progress"
    elif capacity_conflict:
        desired, reason = "information_value", "conflicting_constraints"
    elif violated:
        desired, reason = "invariants", "violated_invariant"
    elif frame.get("integration_gap_refs"):
        desired, reason = "integration", "integration_gap"
    elif frame.get("mandatory_evidence_complete"):
        desired, reason = "stop", "mandatory_evidence_complete"
    eligible_ids = {str(item["method"]) for item in eligibility}
    selected = desired if desired in eligible_ids and not missing else "abstain"
    valid = not missing and selected == desired
    model_proposal = _text(getattr(runtime, "ps_router_model_proposal", None), 80) or None
    return {
        "status": "FIRED",
        "authority": "host",
        "selected_method": selected,
        "switch_reason": reason,
        "valid": valid,
        "stage": stage,
        "eligibility_receipts": eligibility,
        "rejected_methods_with_reasons": rejected + ([{"method": "signals", "reason": reason, "refs": missing}] if missing else []),
        "host_evidence_refs": sorted(host_evidence),
        "missing_evidence_refs": missing,
        "model_selection_advisory": {
            "proposed_method": model_proposal,
            "accepted": model_proposal == selected if model_proposal is not None else None,
            "authority": "host",
        },
        "budget_reservation": 0 if selected == "abstain" else 2,
        "signals_digest": _digest({"ticket": _digest(ticket), "stage": stage, "target": target, "observation": observation}),
    }


def router_checkpoint(runtime: object, frame: object) -> dict[str, object] | None:
    if not enabled("router"):
        return None
    payload = _route(runtime, _frame_mapping(frame))
    _emit(runtime, "ps_router_fired", payload)
    runtime._ps_router_receipt = payload
    return payload


def checkpoint(runtime: object, frame: object) -> None:
    """Run enabled survivors at one real provider-request boundary."""
    invariant_checkpoint(runtime, frame)
    minimize_checkpoint(runtime)
    router_checkpoint(runtime, frame)


__all__ = [
    "FLAGS", "SCHEMA", "checkpoint", "enabled", "invariant_checkpoint",
    "minimize_checkpoint", "router_checkpoint",
]
