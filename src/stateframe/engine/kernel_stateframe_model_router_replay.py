"""Durable, opt-in replay state for the model-router release gate.

The router's ordinary assignment and call events are audit records.  This
module adds one separately versioned state record that is sufficient to resume
lease accounting without treating an audit projection as authoritative.  No
state is read or written unless ``STATEFRAME_MODEL_ROUTER_REPLAY=on``.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping


REPLAY_ENV = "STATEFRAME_MODEL_ROUTER_REPLAY"
REPLAY_SCHEMA = "stateframe-model-router-replay/1"
REPLAY_STATE_EVENT = "model_router_replay_state"
VALID_REPLAY_MODES = frozenset({"off", "on"})

# These events can appear after a worker checkpoint when replay is enabled.
# They are admitted dynamically by kernel_continuation_restart; the baseline
# SAFE_TAIL remains unchanged so an unrecognised router state is still refused.
MODEL_ROUTER_SAFE_TAIL = frozenset({
    "model_router_assignment",
    "model_router_call",
    REPLAY_STATE_EVENT,
})

_STATE_FIELDS = frozenset({
    "schema",
    "lease",
    "consumed_receipts",
    "route_count",
    "pending_reasons",
    "stable_actions",
    "nonprogress_streak",
})
_LEASE_FIELDS = frozenset({
    "lease_id",
    "endpoint",
    "model_id",
    "assigned_step",
    "actions_granted",
    "actions_remaining",
    "meaningful_actions_remaining",
    "reason",
    "why_escalated",
})


class RouterReplayError(ValueError):
    """A durable router state cannot safely authorize a resumed route."""


def _validate_event_identity(record: Mapping[str, object]) -> None:
    for name in ("assignment_id", "request_id"):
        value = record.get(name)
        if value is not None and (not isinstance(value, str) or not value):
            raise RouterReplayError(f"router replay {name} is malformed")
    checkpoint_sequence = record.get("checkpoint_sequence")
    if (checkpoint_sequence is not None
            and (type(checkpoint_sequence) is not int or checkpoint_sequence < 0)):
        raise RouterReplayError("router replay checkpoint identity is malformed")


def replay_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict replay switch, which is off unless explicitly enabled."""
    source = os.environ if environ is None else environ
    value = str(source.get(REPLAY_ENV, "off")).strip().lower() or "off"
    if value not in VALID_REPLAY_MODES:
        raise ValueError(f"{REPLAY_ENV} must be off or on")
    return value


def replay_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return replay_mode(environ) == "on"


def _plain_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RouterReplayError(f"invalid router replay {name}")
    return value


def _scalar(value: object, *, name: str) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return copy.deepcopy(value)
    raise RouterReplayError(f"invalid router replay {name}")


def _validate_lease(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != _LEASE_FIELDS:
        raise RouterReplayError("invalid router replay lease fields")
    lease_id = value.get("lease_id")
    endpoint = value.get("endpoint")
    model_id = value.get("model_id")
    reason = value.get("reason")
    if (not isinstance(lease_id, str) or not lease_id.strip()
            or not isinstance(endpoint, str) or not endpoint.strip()
            or not isinstance(model_id, str) or not model_id.strip()
            or not isinstance(reason, str)):
        raise RouterReplayError("invalid router replay lease identity")
    granted = _plain_integer(value.get("actions_granted"),
                             name="lease grant", minimum=2)
    remaining = _plain_integer(value.get("actions_remaining"),
                               name="lease remaining")
    meaningful_remaining = _plain_integer(
        value.get("meaningful_actions_remaining"),
        name="meaningful lease remaining")
    if remaining > granted or meaningful_remaining != remaining:
        raise RouterReplayError("invalid router replay lease accounting")
    why = value.get("why_escalated")
    if (not isinstance(why, list)
            or any(not isinstance(item, str) for item in why)):
        raise RouterReplayError("invalid router replay escalation reasons")
    return {
        "lease_id": lease_id,
        "endpoint": endpoint,
        "model_id": model_id,
        "assigned_step": _scalar(value.get("assigned_step"), name="assigned step"),
        "actions_granted": granted,
        "actions_remaining": remaining,
        "meaningful_actions_remaining": meaningful_remaining,
        "reason": reason,
        "why_escalated": list(why),
    }


def validate_state(value: object) -> dict[str, object]:
    """Validate and copy one complete replay state record."""
    if not isinstance(value, Mapping) or set(value) != _STATE_FIELDS:
        raise RouterReplayError("invalid router replay state fields")
    if value.get("schema") != REPLAY_SCHEMA:
        raise RouterReplayError("invalid router replay state schema")
    receipts = value.get("consumed_receipts")
    if (not isinstance(receipts, list)
            or any(not isinstance(item, str) or not item for item in receipts)
            or len(set(receipts)) != len(receipts)):
        raise RouterReplayError("invalid router replay receipt set")
    reasons = value.get("pending_reasons")
    if (not isinstance(reasons, list)
            or any(not isinstance(item, str) for item in reasons)):
        raise RouterReplayError("invalid router replay pending reasons")
    return {
        "schema": REPLAY_SCHEMA,
        "lease": _validate_lease(value.get("lease")),
        "consumed_receipts": list(receipts),
        "route_count": _plain_integer(value.get("route_count"), name="route count"),
        "pending_reasons": list(reasons),
        "stable_actions": _plain_integer(
            value.get("stable_actions"), name="stable action count"),
        "nonprogress_streak": _plain_integer(
            value.get("nonprogress_streak"), name="nonprogress streak"),
    }


def snapshot_state(*, lease: object | None, consumed_receipts: set[str] | frozenset[str],
                   route_count: int, pending_reasons: tuple[str, ...] | list[str],
                   stable_actions: int, nonprogress_streak: int) -> dict[str, object]:
    """Build the exact state shape written after each routed decision."""
    if lease is None:
        durable_lease = None
    elif callable(getattr(lease, "as_dict", None)):
        durable_lease = lease.as_dict()
    elif isinstance(lease, Mapping):
        durable_lease = dict(lease)
    else:
        raise RouterReplayError("router replay lease is not serializable")
    state = {
        "schema": REPLAY_SCHEMA,
        "lease": durable_lease,
        "consumed_receipts": sorted(str(item) for item in consumed_receipts),
        "route_count": route_count,
        "pending_reasons": [str(item) for item in pending_reasons],
        "stable_actions": stable_actions,
        "nonprogress_streak": nonprogress_streak,
    }
    return validate_state(state)


def latest_state_record(journal: object, *, through_sequence: int | None = None) -> dict[str, object] | None:
    """Return the last validated state event, retaining its resume identity.

    Staging the complete stream is intentional: an invalid earlier state must
    not be hidden by a later valid-looking record.
    """
    records = getattr(journal, "records", None)
    if not callable(records):
        return None
    latest = None
    replay_records: list[Mapping[str, object]] = []
    for record in records():
        if not isinstance(record, Mapping) or record.get("kind") != REPLAY_STATE_EVENT:
            if (isinstance(record, Mapping)
                    and record.get("kind") == "model_router_assignment"):
                sequence = record.get("sequence")
                if not (through_sequence is not None
                        and type(sequence) is int and sequence > through_sequence):
                    replay_records.append(record)
            continue
        sequence = record.get("sequence")
        if through_sequence is not None and type(sequence) is int and sequence > through_sequence:
            continue
        _validate_event_identity(record)
        replay_records.append(record)
        latest = dict(record)
        latest["state"] = validate_state(record.get("state"))
    validate_tail(replay_records)
    return latest


def latest_state(journal: object, *, through_sequence: int | None = None) -> dict[str, object] | None:
    """Replay every state event and return the last validated state."""
    record = latest_state_record(journal, through_sequence=through_sequence)
    return record.get("state") if record is not None else None


def validate_tail(records: object) -> None:
    """Reject a post-checkpoint assignment that was not durably snapshotted."""
    pending_assignments: list[tuple[str | None, str | None]] = []
    try:
        iterator = iter(records)
    except TypeError as error:
        raise RouterReplayError("router replay tail is not iterable") from error
    for record in iterator:
        if not isinstance(record, Mapping):
            raise RouterReplayError("invalid router replay tail record")
        kind = record.get("kind")
        if kind == "model_router_assignment":
            _validate_event_identity(record)
            assignment_id = record.get("assignment_id")
            pending_assignments.append((assignment_id, record.get("request_id")))
        elif kind == REPLAY_STATE_EVENT:
            _validate_event_identity(record)
            validate_state(record.get("state"))
            assignment_id = record.get("assignment_id")
            request_id = record.get("request_id")
            if assignment_id is not None:
                match = next((item for item in pending_assignments
                              if item[0] == assignment_id), None)
                if match is None:
                    raise RouterReplayError("router replay state is not bound to its assignment")
                if request_id != match[1]:
                    raise RouterReplayError("router replay state is not bound to its request")
                pending_assignments.remove(match)
            elif pending_assignments:
                match = pending_assignments.pop(0)
                if request_id != match[1]:
                    raise RouterReplayError("router replay state is not bound to its request")
            elif request_id is not None:
                raise RouterReplayError(
                    "router replay request identity is not bound to an assignment")
    if pending_assignments:
        raise RouterReplayError(
            "post-checkpoint router assignment has no durable replay state")


def public_projection(state: Mapping[str, object] | None) -> dict[str, object]:
    """Return bounded, non-secret replay status suitable for a handoff."""
    if state is None:
        return {"schema": REPLAY_SCHEMA, "status": "durable"}
    lease = state.get("lease")
    if isinstance(lease, Mapping):
        return {
            "schema": REPLAY_SCHEMA,
            "status": "replayed",
            "lease_id": str(lease.get("lease_id")),
            "endpoint": str(lease.get("endpoint")),
            "actions_remaining": int(lease.get("actions_remaining", 0)),
            "dedup_receipts": len(state.get("consumed_receipts", ())),
        }
    return {"schema": REPLAY_SCHEMA, "status": "replayed", "lease": "none"}


__all__ = [
    "MODEL_ROUTER_SAFE_TAIL",
    "REPLAY_ENV",
    "REPLAY_SCHEMA",
    "REPLAY_STATE_EVENT",
    "RouterReplayError",
    "latest_state",
    "latest_state_record",
    "public_projection",
    "replay_enabled",
    "replay_mode",
    "snapshot_state",
    "validate_tail",
    "validate_state",
]
