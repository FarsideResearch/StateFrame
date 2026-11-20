"""Fold committed journal deltas and serialize only selected panel inputs."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import NamedTuple

from kernel_stateframe_envelope import serialize_stateframe_payload
from kernel_turn_contracts import Corrupt, Event, Head, Key
from kernel_turn_panel_store import canonical, sha256, validate_ref


_KEY_FAMILIES = frozenset({
    "workspace-path", "query-membership", "receipt", "exchange", "step",
    "scope", "route", "policy", "hook", "budget", "authority",
})


class Delta(NamedTuple):
    changed: frozenset[Key]
    verify_dependencies: bool
    last_event: str | None


def _event_identity(event: Event) -> str:
    return sha256(canonical([event.before, event.after, event.kind,
                             [[key.family, key.selector.hex()] for key in event.changed],
                             event.payload]))


def reduce_events(after: Head | None, through: Head, events: Iterator[Event],
                  last_event: str | None) -> Delta:
    """Consume a verified suffix. With no manifest, facts can seed a snapshot at through."""
    cursor = after
    changed: set[Key] = set()
    verify_dependencies = False
    for event in events:
        validate_ref(event.payload, "turn/event/1", (1 << 63) - 1)
        identity = _event_identity(event)
        if cursor is None:
            if event.before.seq != 0:
                raise Corrupt("panel rebuild received a suffix without a starting head")
            cursor = event.before
        if event.after == cursor and identity == last_event:
            continue
        if (event.before != cursor or event.after.seq != cursor.seq + 1
                or event.after.seq > through.seq or not event.after.digest
                or event.after.digest == cursor.digest):
            raise Corrupt("gap, fork or conflicting duplicate in panel journal suffix")
        changed.update(event.changed)
        # An unclassified event is allowed, but only verified fact revisions
        # can prove that every dependent panel remains current.
        if not event.changed or any(key.family not in _KEY_FAMILIES for key in event.changed):
            verify_dependencies = True
        cursor, last_event = event.after, identity
    if cursor is not None and cursor != through:
        raise Corrupt("panel journal suffix did not reach the requested head")
    return Delta(frozenset(changed), verify_dependencies, last_event)


def ordered_value(value: object) -> object:
    """Carry legacy object insertion order through canonical cross-port JSON."""
    if isinstance(value, Mapping):
        return {"object": [[key, ordered_value(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [ordered_value(item) for item in value]
    return value


def _restore_order(value: object) -> object:
    if isinstance(value, list):
        return [_restore_order(item) for item in value]
    if not isinstance(value, dict):
        return value
    pairs = value.get("object")
    if list(value) != ["object"] or not isinstance(pairs, list):
        raise ValueError("panel objects require ordered field pairs")
    result = {}
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or not isinstance(pair[0], str) or pair[0] in result):
            raise ValueError("invalid or repeated panel object field")
        result[pair[0]] = _restore_order(pair[1])
    return result


def encode_fragment(encoding: str, value: object) -> bytes:
    """Encode one member of the legacy default-spaced JSON message array."""
    try:
        restored = _restore_order(value)
        if encoding == "stateframe-message":
            if not isinstance(restored, dict):
                raise ValueError("StateFrame panel payload must be an object")
            # The StateFrame inner JSON uses literal Unicode and compact sorted
            # keys. The outer message string uses the provider's ASCII escapes.
            restored = {"role": "user", "content": serialize_stateframe_payload(restored)}
        elif encoding != "json":
            raise ValueError("unknown panel encoding")
        return json.dumps(restored, allow_nan=False).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise Corrupt("invalid panel fragment input") from exc
