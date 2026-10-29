"""Host-owned, bounded adversarial pressure for the live IDEALPATH kernel.

The adversary is an experiment producer, not a second completion owner.  It
audits the current stage using host receipts, records a typed outcome, and
hands an admitted S3/S5 counterexample to the existing live-counterexample
consumer.  Model text cannot create a receipt, a violation, or a refusal.

A bounded adversarial calibration run supplies
the qualified public relation and control policy used by this adapter: parent
and candidate share one frozen case, viable controls must reach the target,
and environment failures are not mutant kills.  It is a calibration reference,
not solver-visible task or evaluator material.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping, Sequence


SCHEMA = "stateframe-adversary/1"
PROPOSAL_SCHEMA = "stateframe-adversary-proposal/1"
PLAN_SCHEMA = "stateframe-adversary-plan/1"
RECEIPT_SCHEMA = "stateframe-adversary-receipt/1"
MODE_ENV = "STATEFRAME_ADVERSARY"
STAGES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
MOVES = {
    "S1": "OMITTED_CLAUSE",
    "S2": "ALTERNATIVE_SUSPECT",
    "S3": "WRONG_REASON_REPRO",
    "S4": "COMPETING_CAUSE",
    "S5": "BREAK_CANDIDATE",
    "S6": "PLAUSIBLE_WRONG_FIX",
    "S7": "COMPLETION_CONTRADICTION",
}
MAX_PRESSURES_PER_STAGE_EPOCH = 1
FRAME_LAB_REFERENCE = {
    "schema": "frame-lab-adversary-s3-s5-s6/v1",
    "parent_cases": 6,
    "candidate_cases": 8,
    "viable_control_families": 3,
    "worker_calls": 0,
    "wrong_fix_rejection": {"numerator": 11, "denominator": 11},
    "candidate_false_alarm": {"numerator": 0, "denominator": 40},
}

_COUNTEREXAMPLE_OUTCOMES = frozenset({"COUNTEREXAMPLE", "VIOLATION", "VIOLATED"})
_PASS_OUTCOMES = frozenset({"PASS", "PASSED", "GREEN", "OK", "NO_COUNTEREXAMPLE_IN_SCOPE"})
_INVALID_OUTCOMES = frozenset({
    "INVALID_CHECK", "ENVIRONMENT", "RESOURCE_LIMIT", "NOT_RUN", "UNKNOWN",
    "CHECK_INCONCLUSIVE", "NO_EXECUTABLE_ORACLE",
})


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, *, limit: int = 400) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\x00", " ").strip()
    return value[:limit]


def _row(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _rows(value: object) -> tuple[dict[str, object], ...]:
    if isinstance(value, Mapping):
        value = tuple(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(row for item in value if (row := _row(item)))


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be on or off")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_adversary_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    return mode() == "on"


def _owner(runtime: object) -> object | None:
    getter = getattr(runtime, "_idealpath_owner", None)
    if callable(getter):
        try:
            result = getter()
        except (AttributeError, TypeError, ValueError):
            result = None
        if result is not None:
            return result
    return getattr(runtime, "ideal_path_owner", None)


def current_stage(runtime: object) -> str:
    owner = _owner(runtime)
    stage = _text(getattr(owner, "stage", "S1"), limit=8).upper()
    return stage if stage in STAGES else "S1"


def _binding(runtime: object) -> object:
    getter = getattr(runtime, "binding", None)
    if callable(getter):
        try:
            return getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {}
    return {}


def _revision(runtime: object, owner: object | None = None) -> str:
    binding = _binding(runtime)
    if isinstance(binding, Mapping):
        value = binding.get("product_revision") or binding.get("artifact")
        if value:
            return _text(value, limit=200)
    return _text(getattr(owner, "base_revision", "UNKNOWN"), limit=200) or "UNKNOWN"


def _common_terminal_sealed(runtime: object) -> bool:
    """True once the COMMON terminal owner minted a disposition receipt.

    ``kernel_completion_owner.seal_terminal_disposition`` is the only writer of
    ``_terminal_disposition_receipt``.  Post-green ``_postgreen_settlement`` is
    a non-authoritative progress observation and never seals a terminal (the
    old ``_postgreen_terminal_result`` attribute has no writer any more).
    """
    receipt = getattr(runtime, "_terminal_disposition_receipt", None)
    return isinstance(receipt, Mapping) and bool(receipt.get("receipt_digest"))


def _emit(runtime: object, kind: str, **fields: object) -> None:
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit(kind, producer="kernel_adversary", event_schema=SCHEMA,
                 dispatch=False, **fields)
        except Exception:  # noqa: BLE001 - telemetry cannot alter authority
            return


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_adversary_state", None)
    if not isinstance(state, dict):
        state = {
            "pressures": {},
            "receipts": [],
            "counterexample_receipts": [],
            "sealed": False,
            "effect_count": 0,
        }
        setattr(runtime, "_adversary_state", state)
    return state


def _requirements(owner: object | None) -> tuple[str, ...]:
    rows = getattr(owner, "rows", {})
    if isinstance(rows, Mapping):
        return tuple(str(key) for key in rows if str(key))[:2]
    return ()


def _evidence_ids(runtime: object, owner: object | None) -> tuple[str, ...]:
    values: list[str] = []
    receipts = getattr(owner, "receipts", {}) if owner is not None else {}
    if isinstance(receipts, Mapping):
        values.extend(str(key) for key in tuple(receipts)[-8:] if str(key))
    for item in _rows(getattr(runtime, "test_receipts", ())):
        ident = _text(item.get("receipt_id") or item.get("id"), limit=160)
        if ident:
            values.append(ident)
    return tuple(dict.fromkeys(values))[:16]


def _basis(runtime: object, owner: object | None, stage: str) -> str:
    return _digest({
        "stage": stage,
        "epoch": getattr(owner, "epoch", 1),
        "ticket": getattr(owner, "ticket_hash", ""),
        "base": getattr(owner, "base_revision", ""),
        "current": _revision(runtime, owner),
        "requirements": _requirements(owner),
        "evidence": _evidence_ids(runtime, owner),
    })


def _proposal(runtime: object, owner: object | None, stage: str, basis: str) -> dict[str, object]:
    requirements = _requirements(owner)
    return {
        "schema": PROPOSAL_SCHEMA,
        "proposal_id": _digest({"stage": stage, "basis": basis, "move": MOVES[stage]})[7:31],
        "stage": stage,
        "target_decision_id": f"idealpath:{getattr(owner, 'epoch', 1)}",
        "requirement_ids": list(requirements),
        "move": MOVES[stage],
        "branch_id": f"adversary:{stage}:{getattr(owner, 'epoch', 1)}",
        "entrypoint_id": _text(getattr(owner, "owner_id", "host-idealpath"), limit=120),
        "expected_relation_ref": "PUBLIC_REQUIREMENT" if requirements else "HYPOTHESIS_ONLY",
        "evidence_ids": list(_evidence_ids(runtime, owner)),
        "proposed_observation": "host receipt distinguishes the current stage claim",
    }


def _challenge_schema_check(runtime: object, required_selectors: Sequence[str]) -> dict[str, object]:
    try:
        from kernel_challenge_schema import evaluate
        result = evaluate(
            {"finding": "CHALLENGE_ONCE", "selectors": []},
            required_selectors=required_selectors,
        )
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        result = {
            "schema": "stateframe-sidekick-challenge/1",
            "status": "UNAVAILABLE",
            "error_type": type(error).__name__,
        }
    _emit(runtime, "adversary_challenge_schema_evaluated", result=result)
    return result


def _required_selectors(runtime: object) -> tuple[str, ...]:
    getter = getattr(runtime, "required_public_selectors", None)
    if not callable(getter):
        return ()
    try:
        values = getter()
    except (AttributeError, TypeError, ValueError):
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    return tuple(str(item) for item in values if str(item))


def _receipt_result(receipt: Mapping[str, object]) -> tuple[str, Mapping[str, object], Mapping[str, object]]:
    result = receipt.get("result")
    freshness = receipt.get("freshness")
    result_map = result if isinstance(result, Mapping) else receipt
    freshness_map = freshness if isinstance(freshness, Mapping) else receipt
    outcome = _text(result_map.get("outcome") or result_map.get("status") or receipt.get("outcome"), limit=80).upper()
    return outcome, result_map, freshness_map


def _typed_receipts(runtime: object, owner: object | None) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    source = getattr(owner, "receipts", {}) if owner is not None else {}
    if isinstance(source, Mapping):
        values.extend(_row(item) for item in source.values())
    values.extend(_rows(getattr(runtime, "adversary_execution_receipts", ())))
    values.extend(_rows(getattr(runtime, "test_receipts", ())))
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in values:
        ident = _text(item.get("receipt_id") or item.get("id"), limit=200)
        if ident and ident not in seen:
            seen.add(ident)
            result.append(item)
    return tuple(result)


def _contract_expected(owner: object | None, requirement_id: str) -> str:
    rows = getattr(owner, "rows", {})
    row = rows.get(requirement_id) if isinstance(rows, Mapping) else None
    if isinstance(row, Mapping):
        return _text(row.get("expected_observable"), limit=500)
    return ""


def _executed_outcome(runtime: object, owner: object | None, stage: str,
                      *, execution: Mapping[str, object] | None = None) -> dict[str, object]:
    """Classify host execution only; model prose and raw red exits do not qualify."""
    candidates = list(_typed_receipts(runtime, owner))
    if execution is not None:
        candidates.insert(0, dict(execution))
    base_revision = _text(getattr(owner, "base_revision", ""), limit=200)
    current_revision = _revision(runtime, owner)
    for receipt in candidates:
        outcome, result, freshness = _receipt_result(receipt)
        requirement_id = _text(receipt.get("requirement_id"), limit=160)
        reached = result.get("reached_observation") is True
        relevant = result.get("ticket_relevant") is True
        failure_phase = _text(result.get("failure_phase"), limit=80).upper()
        check_hash = _text(freshness.get("check_hash") or receipt.get("check_hash"), limit=200)
        if stage == "S3":
            eligible = (outcome == "BEHAVIOR_FAIL" and reached and relevant
                        and freshness.get("product_revision") == base_revision
                        and freshness.get("is_current") is True)
            target_role = "PARENT"
        elif stage == "S5":
            eligible = (outcome == "BEHAVIOR_FAIL" and reached and relevant
                        and freshness.get("product_revision") not in {None, "", base_revision}
                        and freshness.get("is_current") is True)
            target_role = "CANDIDATE"
        elif stage == "S6":
            eligible = bool(receipt.get("control_id") or receipt.get("artifact_role") == "CONTROL")
            target_role = "CONTROL"
        else:
            eligible = False
            target_role = "OBSERVATION"
        if stage not in {"S3", "S5", "S6"}:
            continue
        if not (eligible or outcome in _INVALID_OUTCOMES or outcome in _PASS_OUTCOMES):
            continue
        expected = _contract_expected(owner, requirement_id)
        receipt_id = _text(receipt.get("receipt_id") or receipt.get("id"), limit=200)
        if not receipt_id:
            continue
        if eligible and outcome == "BEHAVIOR_FAIL" and not expected:
            return {
                "status": "CHECK_INCONCLUSIVE",
                "outcome": "CHECK_INCONCLUSIVE",
                "receipt_id": receipt_id,
                "target_role": target_role,
                "reason": "ticket-bound expected observable is unavailable",
            }
        is_counterexample = bool(eligible and outcome == "BEHAVIOR_FAIL")
        return {
            "status": "COUNTEREXAMPLE" if is_counterexample else (
                "NO_COUNTEREXAMPLE_IN_SCOPE" if outcome in _PASS_OUTCOMES else outcome),
            "outcome": "COUNTEREXAMPLE" if is_counterexample else (
                "NO_COUNTEREXAMPLE_IN_SCOPE" if outcome in _PASS_OUTCOMES else outcome),
            "receipt_id": receipt_id,
            "target_role": target_role,
            "requirement_id": requirement_id,
            "check_hash": check_hash,
            "failure_phase": failure_phase,
            "reached_observation": reached,
            "ticket_relevant": relevant,
            "revision": _text(freshness.get("product_revision") or current_revision, limit=200),
            "expected": expected,
            "observed": _text(result.get("failure_signature") or receipt.get("observed"), limit=500),
        }
    return {
        "status": "NO_COUNTEREXAMPLE_IN_SCOPE",
        "outcome": "NO_COUNTEREXAMPLE_IN_SCOPE",
        "target_role": "OBSERVATION",
        "reason": "no host receipt with an admitted executable counterexample",
    }


def _host_stage_check(runtime: object, owner: object | None, stage: str) -> dict[str, object]:
    if stage in {"S3", "S5", "S6"}:
        return _executed_outcome(runtime, owner, stage)
    if stage == "S7":
        try:
            from kernel_live_counterexample_challenge import live_rows
            rows = live_rows(runtime)
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            rows = ()
        return {
            "status": "COUNTEREXAMPLE" if rows else "NO_COUNTEREXAMPLE_IN_SCOPE",
            "outcome": "COUNTEREXAMPLE" if rows else "NO_COUNTEREXAMPLE_IN_SCOPE",
            "check_ids": [_text(row.get("check_id"), limit=120) for row in rows if _row(row)],
            "reason": "reuses current host counterexample rows; no generic post-green review",
        }
    return {
        "status": "NO_COUNTEREXAMPLE_IN_SCOPE",
        "outcome": "NO_COUNTEREXAMPLE_IN_SCOPE",
        "reason": "stage pressure is a bounded host check; no executable oracle is required",
    }


def _live_receipt(result: Mapping[str, object], proposal: Mapping[str, object],
                  owner: object | None) -> dict[str, object] | None:
    if proposal.get("stage") not in {"S3", "S5"} or result.get("status") != "COUNTEREXAMPLE":
        return None
    ident = _text(result.get("receipt_id"), limit=200)
    expected = _text(result.get("expected"), limit=500)
    if not ident or not expected:
        return None
    check_id = "adversary:" + ident
    return {
        "schema": RECEIPT_SCHEMA,
        "check_id": check_id,
        "obligation_id": _text(result.get("requirement_id"), limit=160) or check_id,
        "receipt_id": ident,
        "status": "VIOLATED",
        "result_class": "VIOLATED",
        "outcome": "COUNTEREXAMPLE",
        "observed": result.get("observed") or "host observed a ticket-bound behavior failure",
        "expected": expected,
        "target_role": result.get("target_role"),
        "target_file": _text(result.get("target_file"), limit=300),
        "revision": result.get("revision") or getattr(owner, "base_revision", ""),
        "binding": result.get("revision") or getattr(owner, "base_revision", ""),
        "adversary_plan_id": proposal.get("proposal_id"),
        "adversary_executed": True,
    }


def _append_runtime_receipt(runtime: object, receipt: Mapping[str, object]) -> None:
    current = getattr(runtime, "_adversary_counterexample_receipts", None)
    if not isinstance(current, list):
        current = []
        setattr(runtime, "_adversary_counterexample_receipts", current)
    ident = _text(receipt.get("check_id") or receipt.get("receipt_id"), limit=200)
    if ident and any(_text(item.get("check_id") or item.get("receipt_id"), limit=200) == ident
                     for item in _rows(current)):
        return
    current.append(copy.deepcopy(dict(receipt)))


def pressure(runtime: object, *, stage: str | None = None, trigger: str = "stage",
             execution: Mapping[str, object] | None = None) -> dict[str, object]:
    """Run one deduplicated, host-only counterexample pressure check."""
    if not enabled(runtime):
        return {"schema": SCHEMA, "status": "OFF", "stage": stage or current_stage(runtime)}
    owner = _owner(runtime)
    stage_name = (stage or current_stage(runtime)).upper()
    if stage_name not in STAGES:
        raise ValueError(f"unknown adversary stage: {stage_name}")
    state = _state(runtime)
    basis = _basis(runtime, owner, stage_name)
    key = f"{stage_name}:{getattr(owner, 'epoch', 1)}:{basis}"
    previous = state["pressures"].get(key) if isinstance(state.get("pressures"), Mapping) else None
    if isinstance(previous, Mapping):
        reused = copy.deepcopy(dict(previous))
        reused["status"] = "REUSED"
        reused["trigger"] = trigger
        _emit(runtime, "adversary_pressure_reused", **reused)
        return reused
    proposal = _proposal(runtime, owner, stage_name, basis)
    state.setdefault("pressures", {})[key] = {"proposal_id": proposal["proposal_id"], "stage": stage_name}
    _emit(runtime, "adversary_attack_due", stage=stage_name, trigger=trigger,
          basis=basis, proposal=proposal, remaining_budget=STAGES.index(stage_name) + 1)
    selectors = _required_selectors(runtime)
    schema_check = _challenge_schema_check(runtime, selectors)
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": proposal["proposal_id"],
        "proposal_id": proposal["proposal_id"],
        "stage": stage_name,
        "basis": basis,
        "artifact_revision": _revision(runtime, owner),
        "expected_relation_ref": proposal["expected_relation_ref"],
        "idempotency_key": _digest({"basis": basis, "move": MOVES[stage_name]}),
        "budget": {"provider_calls": 0, "host_executions": 1},
        "schema_check": schema_check,
    }
    _emit(runtime, "adversary_attack_admitted", stage=stage_name, proposal=proposal, plan=plan)
    _emit(runtime, "adversary_attack_dispatched", stage=stage_name, plan_id=plan["plan_id"],
          executor="host_receipt_classifier", executed=True)
    if (_common_terminal_sealed(runtime)
            and (stage_name == "S7" or trigger == "completion")):
        result = {
            "status": "NO_ATTACK_DUE",
            "outcome": "NO_ATTACK_DUE",
            "terminal_sealed": True,
            "reason": "common terminal owner already sealed this terminal action",
        }
        state["sealed"] = True
    else:
        result = (_executed_outcome(runtime, owner, stage_name, execution=execution)
                  if stage_name in {"S3", "S5", "S6"} else
                  _host_stage_check(runtime, owner, stage_name))
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "receipt_id": "adversary-exec:" + _digest({"plan": plan["plan_id"], "result": result})[7:31],
        "plan_id": plan["plan_id"],
        "stage": stage_name,
        "status": result.get("status", "UNKNOWN"),
        "outcome": result.get("outcome", "UNKNOWN"),
        "target_reached": result.get("reached_observation") is True,
        "observations": {key: value for key, value in result.items()
                         if key not in {"status", "outcome"}},
        "artifact_revision": plan["artifact_revision"],
        "provider_calls": 0,
        "host_execution": True,
    }
    state.setdefault("receipts", []).append(copy.deepcopy(receipt))
    _emit(runtime, "adversary_attack_receipt", stage=stage_name, plan=plan, receipt=receipt)
    live = _live_receipt(result, proposal, owner)
    if live is not None:
        _append_runtime_receipt(runtime, live)
        effect = "LIVE_COUNTEREXAMPLE_SOURCE"
    else:
        effect = "NO_COUNTEREXAMPLE_IN_SCOPE" if result.get("status") == "NO_COUNTEREXAMPLE_IN_SCOPE" else "ADVISORY"
    state["effect_count"] = int(state.get("effect_count", 0) or 0) + 1
    effect_payload = {
        "stage": stage_name,
        "plan_id": plan["plan_id"],
        "receipt_id": receipt["receipt_id"],
        "effect": effect,
        "live_counterexample_check_id": live.get("check_id") if live else None,
        "consumed_by": "kernel_live_counterexample_challenge" if live else "stage_owner",
    }
    _emit(runtime, "adversary_decision_effect", **effect_payload)
    result_payload = {
        "schema": SCHEMA,
        "status": "EXECUTED",
        "stage": stage_name,
        "trigger": trigger,
        "basis": basis,
        "proposal": proposal,
        "plan": plan,
        "receipt": receipt,
        "result": result,
        "effect": effect_payload,
        "frame_lab_schema": FRAME_LAB_REFERENCE["schema"],
    }
    state["pressures"][key] = copy.deepcopy(result_payload)
    setattr(runtime, "_adversary_last_pressure", copy.deepcopy(result_payload))
    return result_payload


def attach(runtime: object) -> dict[str, object]:
    """Attach the durable projection and immediately pressure the current S1."""
    existing = getattr(runtime, "_adversary_state", None)
    if isinstance(existing, dict):
        return existing
    state = _state(runtime)
    owner = _owner(runtime)
    state["ticket_hash"] = getattr(owner, "ticket_hash", "")
    state["base_revision"] = getattr(owner, "base_revision", "")
    if not enabled(runtime):
        return state
    _emit(runtime, "adversary_attached", stage=current_stage(runtime),
          state_epoch=getattr(owner, "epoch", 1),
          frame_lab_schema=FRAME_LAB_REFERENCE["schema"],
          frame_lab_provenance="operator-calibration-only")
    try:
        pressure(runtime, trigger="attach")
    except (AttributeError, TypeError, ValueError, OSError) as error:
        _emit(runtime, "adversary_error", phase="attach", error_type=type(error).__name__)
    return state


__all__ = [
    "FRAME_LAB_REFERENCE", "MOVES", "PLAN_SCHEMA", "PROPOSAL_SCHEMA", "RECEIPT_SCHEMA",
    "SCHEMA", "STAGES", "attach", "current_stage", "enabled", "mode", "pressure",
]
