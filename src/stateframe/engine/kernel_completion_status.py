"""One reducer-owned completion status for the model-facing StateFrame.

This is deliberately a presentation seam.  It never evaluates public-test
receipts or green-lock facts: the only authority accepted here is the typed
decision returned by ``CompletionReducer`` (or its serialized decision).
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_BLOCKER_RENDER"
# The descriptive alias keeps fixtures written against the field name usable;
# the lane switch remains the primary, documented control.
ALIAS_MODE_ENV = "STATEFRAME_COMPLETION_STATUS"
VALID_MODES = frozenset({"off", "on"})

ACCEPTABLE = "ACCEPTABLE"
NEEDS_WORK = "NEEDS_WORK"
BLOCKED = "BLOCKED"
VALID_STATES = frozenset({ACCEPTABLE, NEEDS_WORK, BLOCKED})
_PASS_RESULTS = frozenset({
    "GREEN", "CERTIFIED", "PASS", "PASSED", "SUCCESS", "OK",
    "REUSED_VERIFICATION",
})

_MISSING = object()


def blocker_render_mode() -> str:
    """Return the strict opt-in mode for the completion-status projection."""
    value = os.environ.get(MODE_ENV)
    if value is None:
        value = os.environ.get(ALIAS_MODE_ENV, "off")
    value = str(value).strip().lower()
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return blocker_render_mode() == "on"


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except Exception:
            return None
        return result if isinstance(result, Mapping) else None
    return None


def _value(value: object, *names: str, default: object = _MISSING) -> object:
    mapping = _mapping(value)
    for name in names:
        if mapping is not None and name in mapping:
            return mapping[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _rows(value: object) -> list[object]:
    if isinstance(value, Mapping):
        value = tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _first(value: object) -> object:
    rows = _rows(value)
    return rows[0] if rows else None


def _copy(value: object) -> object:
    return copy.deepcopy(value)


def _first_present(containers: Sequence[object], names: Sequence[str]) -> object:
    for container in containers:
        value = _value(container, *names)
        if value is not _MISSING:
            return _copy(value)
    return None


def _counterexample(containers: Sequence[object]) -> object:
    raw = _first_present(containers, ("counterexample",))
    return raw if isinstance(raw, Mapping) else None


def _fixture_input(containers: Sequence[object], counterexample: Mapping[str, object] | None) -> object:
    value = _first_present(containers, (
        "fixture_input", "input_fixture", "fixture_args", "fixture",
    ))
    # ``fixture`` on a serialized CheckReceipt is a digest, never the input.
    if value is not None and not (isinstance(value, str) and value.startswith("sha256:")):
        return value
    if counterexample is not None:
        for key in ("fixture_input", "input", "args", "kwargs", "fixture"):
            if key in counterexample:
                return _copy(counterexample[key])
        if "input_before" in counterexample or "input_after" in counterexample:
            return {
                key: _copy(counterexample[key])
                for key in ("input_before", "input_after")
                if key in counterexample
            }
    return None


def _blocking_row(decision: Mapping[str, object]) -> dict[str, object] | None:
    payload = _mapping(decision.get("needs_work"))
    if payload is None:
        payload = _mapping(decision.get("needs_work_payload"))

    assessments = _rows(decision.get("obligation_assessments"))
    focus = _value(payload, "focus_obligation", "obligation_id", "id", default=None)
    assessment = next(
        (item for item in assessments
         if str(_value(item, "obligation_id", "id", default="")) == str(focus)),
        None,
    ) if focus else None
    if assessment is None:
        assessment = next((item for item in assessments
                           if str(_value(item, "verdict", "status", default="")).upper()
                           not in {"", "SATISFIED", "PASS", "GREEN"}), None)
    if assessment is None and assessments:
        assessment = assessments[0]

    assessment_receipts = _rows(_value(assessment, "receipts", default=()))
    receipt = next((item for item in assessment_receipts
                    if str(_value(item, "underlying_status", "status", "result",
                                   "result_class", default="")).upper()
                    in {"COUNTEREXAMPLE", "HARNESS_ERROR", "NOT_EXECUTED"}), None)
    if receipt is None and assessment_receipts:
        receipt = assessment_receipts[0]
    counterexample = _counterexample((payload, assessment, receipt))
    containers = (payload, assessment, receipt, counterexample)
    ids = _first_present(containers, ("check_id", "check"))
    if ids is None:
        ids = _first_present(containers, ("check_ids", "checks"))
        ids = _first(ids)
    obligation_id = _first_present(
        containers, ("focus_obligation", "obligation_id", "obligation", "id"))
    if obligation_id is None:
        ids_from_decision = _rows(decision.get("needs_work_ids"))
        obligation_id = _copy(ids_from_decision[0]) if ids_from_decision else None

    has_data = any(value is not None for value in (obligation_id, ids, counterexample))
    if not has_data:
        return None
    return {
        "obligation_id": obligation_id,
        "check_id": ids,
        "fixture_input": _fixture_input(containers, counterexample),
        "observed": _first_present(containers, ("observed", "actual")),
        "expected": _first_present(containers, ("expected", "required")),
    }


@dataclass(frozen=True)
class CompletionStatus:
    """The only completion-shaped object exposed by the blocker-render arm."""

    state: str
    blocking: tuple[dict[str, object], ...] = ()
    next_feasible_action: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        state = str(self.state).upper()
        if state not in VALID_STATES:
            raise ValueError(f"unknown completion status state: {self.state!r}")
        object.__setattr__(self, "state", state)
        rows = []
        for item in self.blocking or ():
            if not isinstance(item, Mapping):
                continue
            rows.append({
                "obligation_id": _copy(item.get("obligation_id")),
                "check_id": _copy(item.get("check_id")),
                "fixture_input": _copy(item.get("fixture_input")),
                "observed": _copy(item.get("observed")),
                "expected": _copy(item.get("expected")),
            })
        if state == ACCEPTABLE:
            rows = []
        object.__setattr__(self, "blocking", tuple(rows[:1]))
        action = self.next_feasible_action
        object.__setattr__(
            self, "next_feasible_action",
            _copy(dict(action)) if isinstance(action, Mapping) else None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "blocking": [_copy(item) for item in self.blocking],
            "next_feasible_action": _copy(self.next_feasible_action),
        }


def completion_status_from_decision(decision: object) -> CompletionStatus:
    """Project a reducer decision without consulting any evidence side-channel."""
    raw = _mapping(decision)
    if raw is None:
        return CompletionStatus(BLOCKED)
    if raw.get("accepted") is True or str(raw.get("outcome", "")).upper() == "ACCEPTED":
        state = ACCEPTABLE
    elif str(raw.get("outcome", "")).upper() in {"NEEDS_WORK", "REFUSED"}:
        state = NEEDS_WORK
    else:
        state = BLOCKED
    row = _blocking_row(raw) if state != ACCEPTABLE else None
    payload = _mapping(raw.get("needs_work")) or _mapping(raw.get("needs_work_payload"))
    action = raw.get("repair_action") if state == NEEDS_WORK else None
    if not isinstance(action, Mapping) and payload is not None:
        candidate = payload.get("next_action")
        action = candidate if isinstance(candidate, Mapping) else None
    return CompletionStatus(state, (row,) if row is not None else (), action)


def _status_from_mapping(value: Mapping[str, object]) -> CompletionStatus | None:
    state = str(value.get("state", "")).upper()
    if state not in VALID_STATES:
        return None
    return CompletionStatus(state, tuple(_rows(value.get("blocking"))),
                           value.get("next_feasible_action"))


def completion_status_from_frame(frame: Mapping[str, object]) -> CompletionStatus:
    """Read the typed block or its serialized reducer decision only."""
    if not isinstance(frame, Mapping):
        raise TypeError("completion status requires a frame mapping")
    for key in ("completion_reducer_decision", "completion_decision", "atomic_completion"):
        if frame.get(key) is not None:
            return completion_status_from_decision(frame[key])
    typed = frame.get("completion_status")
    if isinstance(typed, Mapping):
        parsed = _status_from_mapping(typed)
        if parsed is not None:
            return parsed
    # Opt-in rendering is fail-closed before the first reducer decision.
    return CompletionStatus(BLOCKED)


def render_completion_status(value: CompletionStatus | Mapping[str, object]) -> str:
    status = value if isinstance(value, CompletionStatus) else _status_from_mapping(value)
    if status is None:
        status = CompletionStatus(BLOCKED)
    return "completion_status: " + json.dumps(
        status.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )


def enrich_needs_work_payload(payload: object, *, assessments: Sequence[object] = (),
                              definitions: Sequence[object] = ()) -> dict[str, object] | None:
    """Carry the focused check's fixture input into the reducer decision."""
    if not isinstance(payload, Mapping):
        return None
    result = copy.deepcopy(dict(payload))
    focus = str(result.get("focus_obligation", ""))
    assessment = next((item for item in assessments
                       if str(_value(item, "obligation_id", "id", default="")) == focus), None)
    check_ids = _rows(_value(assessment, "check_ids", "checks", default=()))
    check_id = result.get("check_id") or (check_ids[0] if check_ids else None)
    definition = next((item for item in definitions
                       if str(_value(item, "check_id", "id", default="")) == str(check_id)), None)
    if check_id is not None:
        result.setdefault("check_id", copy.deepcopy(check_id))
    if definition is not None:
        fixture = _value(definition, "fixture", default=_MISSING)
        if fixture is not _MISSING:
            result.setdefault("fixture_input", _copy(fixture))
    return result


def suppress_conflicting_completion(frame: Mapping[str, object],
                                    status: CompletionStatus | None = None) -> dict[str, object]:
    """Project a frame while removing completion claims that disagree."""
    if not isinstance(frame, Mapping):
        raise TypeError("completion presentation requires a frame mapping")
    status = status or completion_status_from_frame(frame)
    result = copy.deepcopy(dict(frame))
    # The new block replaces duplicate reducer-shaped projections in the arm.
    for key in ("atomic_completion", "completion_decision", "completion_reducer_decision",
                "completion_verdict"):
        result.pop(key, None)
    if status.state != ACCEPTABLE:
        for key in ("valid_product_verification", "green_lock_message", "green_lock",
                    "terminal", "terminal_submission", "frame_series", "pattern_cards",
                    "portfolio", "test_receipts", "probe_receipts", "receipts",
                    "recent_test_feedback", "latest_receipts",
                    "recent_change_verification"):
            result.pop(key, None)
        contract = result.get("completion_contract")
        if isinstance(contract, Mapping):
            contract = dict(contract)
            for key in ("current_public_checks", "stop_action", "submission",
                        "admission", "certification", "completion_authority",
                        "atomic_completion"):
                contract.pop(key, None)
            result["completion_contract"] = contract
        verification = result.get("verification")
        if isinstance(verification, Mapping):
            verification = dict(verification)
            for key in ("status", "result", "result_class", "outcome", "verdict",
                        "certification", "certified"):
                verification.pop(key, None)
            result["verification"] = verification
        world = result.get("world")
        if isinstance(world, Mapping):
            world = dict(world)
            for key in ("certification", "valid_product_verification", "completion_verdict",
                        "last_test"):
                world.pop(key, None)
            result["world"] = world
        last_action = result.get("last_action")
        if isinstance(last_action, Mapping) and any(
                str(value).upper() in _PASS_RESULTS for value in last_action.values()):
            result.pop("last_action", None)
        for container_name in ("pending", "resume"):
            container = result.get(container_name)
            if not isinstance(container, Mapping):
                continue
            container = dict(container)
            for key in ("next_action", "recommended_next_action", "last_action_result",
                        "status", "result", "outcome", "verdict", "certification"):
                container.pop(key, None)
            result[container_name] = container
    result["completion_status"] = status.to_dict()
    return result


def presentation_frame(frame: Mapping[str, object]) -> dict[str, object]:
    """Return the opt-in model-facing projection; callers gate it explicitly."""
    return suppress_conflicting_completion(frame, completion_status_from_frame(frame))


__all__ = [
    "ACCEPTABLE", "ALIAS_MODE_ENV", "BLOCKED", "CompletionStatus", "MODE_ENV",
    "NEEDS_WORK", "VALID_MODES", "blocker_render_mode", "completion_status_from_decision",
    "completion_status_from_frame", "enabled", "enrich_needs_work_payload",
    "presentation_frame", "render_completion_status", "suppress_conflicting_completion",
]
