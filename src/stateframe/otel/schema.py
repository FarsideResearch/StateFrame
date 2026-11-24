"""Closed validation and canonical local framing for telemetry records."""
from __future__ import annotations

import json
import math
import re
from enum import Enum

from .levels import ATTRIBUTE_NAMES, valid_attribute
from .operations import (
    ARTIFACTS, CHECKS, CHECK_KINDS, DELIVERIES, EXECUTIONS, HOOK_EVENTS,
    OUTCOMES, RESUMES, RETRY_KINDS, ROLES, RUNTIMES, TOOLS,
)
from .types import (
    AttrKey,
    Coverage,
    Reason,
    SafeAttr,
    SafeEvent,
    SafeLink,
    SafeSpan,
    Signal,
    SignalBatch,
)


_MAX_COUNT = (1 << 53) - 1
_MAX_TOKEN = (1 << 31) - 1
_MAX_ORDINAL = (1 << 31) - 1
_MAX_EPOCH_NS = (1 << 63) - 1
_LOCAL_MAX_BYTES = 256 * 1024
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_CLEANUP = frozenset({"verified", "failed", "unknown"})
_MUTATION_STAGES = frozenset({"admission", "helper", "cleanup", "prepare", "commit", "broker", "watch", "hook"})
_PARALLEL_DISPOSITIONS = frozenset({
    "committed", "committed_unverified", "refused", "blocked", "exhausted", "error", "cancelled",
    "timed_out", "recovery_required", "unknown",
})
_SOURCES = frozenset({"record", "lifecycle", "historical_boundary"})
_TIMINGS = frozenset({"exact", "reconstructed", "publication", "registration", "observed_boundary"})
_EVENTS = frozenset({
    "retry_scheduled", "physical_send_observed", "queue_wait_observed", "check_completed",
    "operation_refused", "operation_blocked", "checkpoint_saved", "checkpoint_failed",
    "checkpoint_restored", "replay_started", "replay_completed", "replay_diverged", "effect_prepared",
    "effect_committed", "effect_indeterminate", "cleanup_verified", "delivery_refused",
    "coverage_incomplete",
})
_SPAN_NAMES = frozenset({
    "stateframe.run.context", "stateframe.attempt.context", "stateframe.parallel.context",
    "stateframe.subframe.context", "stateframe.invocation", "stateframe.request.prepare",
    "stateframe.model.request", "chat", "stateframe.action", "stateframe.mcp", "stateframe.check",
    "stateframe.checkpoint", "stateframe.hook", "stateframe.parallel", "stateframe.subframe",
    "stateframe.delivery", "stateframe.attempt.outcome", "stateframe.attempt.observation_end",
})
_ATTR_VALUES: dict[str, frozenset[str]] = {
    "stateframe.source": _SOURCES,
    "stateframe.timing": _TIMINGS,
    "stateframe.outcome": OUTCOMES | {"pass", "fail"},
    "stateframe.execution": EXECUTIONS,
    "stateframe.runtime": RUNTIMES,
    "stateframe.check": CHECKS,
    "stateframe.artifact": ARTIFACTS,
    "stateframe.delivery": DELIVERIES,
    "stateframe.resume": RESUMES | {"resumed"},
    "stateframe.role": ROLES,
    "stateframe.retry": RETRY_KINDS,
    "stateframe.usage": frozenset({"complete", "partial", "missing", "conflicting"}),
    "gen_ai.operation.name": frozenset({"chat"}),
    "stateframe.tool": TOOLS,
    "stateframe.cleanup": _CLEANUP,
    "stateframe.check.kind": CHECK_KINDS,
    "stateframe.hook.event": HOOK_EVENTS,
    "stateframe.mutation.stage": _MUTATION_STAGES,
    "stateframe.parallel.disposition": _PARALLEL_DISPOSITIONS,
    "stateframe.reason": frozenset(item.value for item in Reason),
    "gen_ai.token.type": frozenset({"input", "output"}),
    "stateframe.cache": frozenset({"cached", "uncached"}),
}
_BOOL_KEYS = frozenset({"stateframe.complete"})
_INT_KEYS = frozenset({
    "stateframe.duration_ns", "stateframe.attempt", "stateframe.exit_code", "stateframe.request.ordinal",
    "stateframe.http.status_code", "stateframe.cache.tokens", "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens", "stateframe.incarnation", "stateframe.count",
})
_FLOAT_KEYS = frozenset({"stateframe.duration_s"})
_ATTR_KEYS = frozenset(item.value for item in AttrKey)
def add_reason(reasons: dict[Reason, int | None], reason: Reason, count: int | None) -> None:
    previous = reasons.get(reason, 0)
    reasons[reason] = None if previous is None or count is None else min(_MAX_COUNT, previous + count)


def _fail() -> None:
    raise ValueError("invalid_field")


def _enum_text(value: object) -> str | None:
    if type(value) is str:
        return value
    if isinstance(value, Enum):
        return value.value if type(value.value) is str else None
    return None


def _bounded_int(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _bounded_number(value: object, low: float, high: float) -> bool:
    if type(value) is int:
        return low <= value <= high
    return type(value) is float and math.isfinite(value) and low <= value <= high


def _epoch(value: object) -> bool:
    return _bounded_int(value, 1, _MAX_EPOCH_NS)


def _id(value: object, length: int) -> bool:
    return type(value) is bytes and len(value) == length and any(value)


def _attr_key(value: object) -> str | None:
    text = _enum_text(value)
    return text if text in _ATTR_KEYS else None


def _validate_attr(attr: SafeAttr) -> None:
    if type(attr) is not SafeAttr or type(attr.key) is not AttrKey:
        _fail()
    key = _attr_key(attr.key)
    if key is None:
        _fail()
    value = attr.value
    if key in ATTRIBUTE_NAMES:
        if not valid_attribute(key, value):
            _fail()
        return
    if key in _BOOL_KEYS:
        if type(value) is not bool:
            _fail()
        return
    if key in _INT_KEYS:
        if key == "stateframe.attempt" or key == "stateframe.request.ordinal" or key == "stateframe.incarnation":
            if not _bounded_int(value, 1, _MAX_ORDINAL):
                _fail()
        elif key == "stateframe.exit_code":
            if not _bounded_int(value, 0, 255):
                _fail()
        elif key == "stateframe.http.status_code":
            if not _bounded_int(value, 100, 599):
                _fail()
        elif key in {"stateframe.cache.tokens", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"}:
            if not _bounded_int(value, 0, _MAX_TOKEN):
                _fail()
        elif not _bounded_int(value, 0, _MAX_COUNT):
            _fail()
        return
    if key in _FLOAT_KEYS:
        if not _bounded_number(value, 0.0, 604800.0):
            _fail()
        return
    if type(value) is not str or len(value) > 128:
        _fail()
    try:
        value.encode("utf-8", "strict")
    except UnicodeError:
        _fail()
    if key == "stateframe.run.id":
        if not _HEX32.fullmatch(value):
            _fail()
    elif value not in _ATTR_VALUES[key]:
        _fail()


def _validate_attrs(attrs: object, *, maximum: int, required: frozenset[str] = frozenset()) -> None:
    if type(attrs) is not tuple or len(attrs) > maximum:
        _fail()
    keys: list[str] = []
    for attr in attrs:
        _validate_attr(attr)
        key = _attr_key(attr.key)
        if key is None or key == AttrKey.CACHE:
            _fail()
        keys.append(key)
    if keys != sorted(keys) or len(set(keys)) != len(keys) or not required.issubset(keys):
        _fail()


def _span(span: SafeSpan) -> None:
    if type(span) is not SafeSpan or type(span.name) is not str or span.name not in _SPAN_NAMES:
        _fail()
    if not _id(span.trace_id, 16) or not _id(span.span_id, 8):
        _fail()
    if span.parent_id is not None and not _id(span.parent_id, 8):
        _fail()
    if not _epoch(span.start_ns) or not _epoch(span.end_ns) or span.start_ns > span.end_ns:
        _fail()
    if type(span.error) is not bool:
        _fail()
    if not all(_bounded_int(value, 0, _MAX_COUNT) for value in
               (span.dropped_attrs, span.dropped_events, span.dropped_links)):
        _fail()
    _validate_attrs(span.attrs, maximum=24, required=frozenset({
        "stateframe.run.id", "stateframe.source", "stateframe.timing", "stateframe.complete",
    }))
    if type(span.events) is not tuple or len(span.events) > 16:
        _fail()
    for event in span.events:
        if type(event) is not SafeEvent or type(event.name) is not str or event.name not in _EVENTS:
            _fail()
        if not _epoch(event.at_ns) or not span.start_ns <= event.at_ns <= span.end_ns:
            _fail()
        if not _bounded_int(event.dropped_attrs, 0, _MAX_COUNT):
            _fail()
        _validate_attrs(event.attrs, maximum=8)
    if type(span.links) is not tuple or len(span.links) > 16:
        _fail()
    for link in span.links:
        if type(link) is not SafeLink or not _id(link.trace_id, 16) or not _id(link.span_id, 8):
            _fail()
        if link.trace_id == span.trace_id and link.span_id == span.span_id:
            _fail()


def validate_batch(batch: SignalBatch) -> None:
    """Validate a closed signal batch, raising only a fixed public error."""
    try:
        if type(batch) is not SignalBatch or type(batch.signal) is not Signal:
            _fail()
        if not _id(batch.resource_id, 16):
            _fail()
        if type(batch.spans) is not tuple or not 1 <= len(batch.spans) <= 256:
            _fail()
        for span in batch.spans:
            _span(span)
    except ValueError as error:
        if str(error) == "invalid_field":
            raise
        _fail()
    except (AttributeError, OverflowError, TypeError):
        _fail()


def coverage_merge(a: Coverage, b: Coverage) -> Coverage:
    """Merge reason lower bounds without turning an unknown count into an exact one."""
    try:
        if type(a) is not Coverage or type(b) is not Coverage:
            _fail()
        if type(a.complete) is not bool or type(b.complete) is not bool:
            _fail()
        merged: dict[str, int | None] = {}
        saturated = False
        if type(a.reasons) is not tuple or type(b.reasons) is not tuple:
            _fail()
        for reason, count in (*a.reasons, *b.reasons):
            if type(reason) is not Reason:
                _fail()
            text = reason.value
            if count is not None and not _bounded_int(count, 0, _MAX_COUNT):
                _fail()
            if text not in merged:
                merged[text] = count
            elif merged[text] is None or count is None:
                merged[text] = None
            else:
                total = merged[text] + count
                merged[text] = min(total, _MAX_COUNT)
                if total > _MAX_COUNT:
                    saturated = True
        exact = all(count is not None for count in merged.values())
        return Coverage(a.complete and b.complete and exact and not saturated and not merged,
                        tuple((Reason(key), merged[key]) for key in sorted(merged)))
    except ValueError as error:
        if str(error) == "invalid_field":
            raise
        _fail()
    except (AttributeError, TypeError):
        _fail()


def pack_local(batch: SignalBatch) -> bytes:
    """Serialize the local IR without unknown keys or arbitrary Python values."""
    validate_batch(batch)
    payload = _batch_to_dict(batch)
    try:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError):
        _fail()
    if len(encoded) > _LOCAL_MAX_BYTES:
        _fail()
    return encoded


def unpack_local(frame: bytes) -> SignalBatch:
    """Decode one bounded canonical local frame and reject duplicate/unknown fields."""
    if type(frame) is not bytes or not frame or len(frame) > _LOCAL_MAX_BYTES:
        _fail()

    try:
        decoded = _load_bounded_json(frame)
        batch = _batch_from_dict(decoded)
        validate_batch(batch)
    except (UnicodeError, json.JSONDecodeError, RecursionError, TypeError, KeyError, IndexError):
        _fail()
    except ValueError as error:
        if str(error) == "invalid_field":
            raise
        _fail()
    if pack_local(batch) != frame:
        _fail()
    return batch


def _load_bounded_json(data: bytes, *, oversize=_fail) -> object:
    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail()
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        _fail()

    value = json.loads(data.decode("utf-8", "strict"), object_pairs_hook=unique_pairs,
                       parse_constant=reject_constant)
    remaining = 65_536

    def visit(node: object, depth: int) -> None:
        nonlocal remaining
        if depth > 32:
            _fail()
        remaining -= 1
        if remaining < 0:
            oversize()
        if type(node) is dict:
            for key, item in node.items():
                visit(key, depth + 1)
                visit(item, depth + 1)
        elif type(node) is list:
            for item in node:
                visit(item, depth + 1)
        elif type(node) is str:
            node.encode("utf-8", "strict")
        elif type(node) is float and not math.isfinite(node):
            _fail()

    visit(value, 0)
    return value


def _batch_to_dict(batch: SignalBatch) -> dict[str, object]:
    return {
        "signal": batch.signal.value,
        "resource_id": batch.resource_id.hex(),
        "spans": [_span_to_dict(span) for span in batch.spans],
    }


def _attr_to_dict(attr: SafeAttr) -> dict[str, object]:
    return {"key": _enum_text(attr.key), "value": attr.value}


def _span_to_dict(span: SafeSpan) -> dict[str, object]:
    return {
        "name": span.name, "trace_id": span.trace_id.hex(), "span_id": span.span_id.hex(),
        "parent_id": span.parent_id.hex() if span.parent_id is not None else None,
        "start_ns": span.start_ns, "end_ns": span.end_ns,
        "attrs": [_attr_to_dict(attr) for attr in span.attrs],
        "events": [{"name": event.name, "at_ns": event.at_ns,
                    "attrs": [_attr_to_dict(attr) for attr in event.attrs],
                    "dropped_attrs": event.dropped_attrs} for event in span.events],
        "links": [{"trace_id": link.trace_id.hex(), "span_id": link.span_id.hex()}
                  for link in span.links],
        "dropped_attrs": span.dropped_attrs, "dropped_events": span.dropped_events,
        "dropped_links": span.dropped_links, "error": span.error,
    }


def _expect_keys(value: object, expected: frozenset[str]) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        _fail()
    return value


def _parse_attrs(value: object) -> tuple[SafeAttr, ...]:
    if type(value) is not list:
        _fail()
    result: list[SafeAttr] = []
    for item in value:
        row = _expect_keys(item, frozenset({"key", "value"}))
        key = row["key"]
        if type(key) is not str or key not in _ATTR_KEYS:
            _fail()
        result.append(SafeAttr(AttrKey(key), row["value"]))
    return tuple(result)


def _parse_span(value: object) -> SafeSpan:
    row = _expect_keys(value, frozenset({
        "name", "trace_id", "span_id", "parent_id", "start_ns", "end_ns", "attrs", "events", "links",
        "dropped_attrs", "dropped_events", "dropped_links", "error",
    }))
    if type(row["name"]) is not str or type(row["trace_id"]) is not str or type(row["span_id"]) is not str:
        _fail()
    parent = row["parent_id"]
    if parent is not None and type(parent) is not str:
        _fail()
    events: list[SafeEvent] = []
    if type(row["events"]) is not list:
        _fail()
    for item in row["events"]:
        event = _expect_keys(item, frozenset({"name", "at_ns", "attrs", "dropped_attrs"}))
        events.append(SafeEvent(event["name"], event["at_ns"], _parse_attrs(event["attrs"]),
                               event["dropped_attrs"]))
    links: list[SafeLink] = []
    if type(row["links"]) is not list:
        _fail()
    for item in row["links"]:
        link = _expect_keys(item, frozenset({"trace_id", "span_id"}))
        if type(link["trace_id"]) is not str or type(link["span_id"]) is not str:
            _fail()
        links.append(SafeLink(bytes.fromhex(link["trace_id"]), bytes.fromhex(link["span_id"])))
    return SafeSpan(
        row["name"], bytes.fromhex(row["trace_id"]), bytes.fromhex(row["span_id"]),
        bytes.fromhex(parent) if parent is not None else None,
        row["start_ns"], row["end_ns"], _parse_attrs(row["attrs"]), tuple(events), tuple(links),
        row["dropped_attrs"], row["dropped_events"], row["dropped_links"], row["error"],
    )


def _batch_from_dict(value: object) -> SignalBatch:
    row = _expect_keys(value, frozenset({"signal", "resource_id", "spans"}))
    if type(row["signal"]) is not str or type(row["resource_id"]) is not str:
        _fail()
    if type(row["spans"]) is not list:
        _fail()
    if not re.fullmatch(r"[0-9a-f]{32}", row["resource_id"]):
        _fail()
    return SignalBatch(Signal(row["signal"]), bytes.fromhex(row["resource_id"]),
                       tuple(_parse_span(item) for item in row["spans"]))


__all__ = ["coverage_merge", "pack_local", "unpack_local", "validate_batch"]
