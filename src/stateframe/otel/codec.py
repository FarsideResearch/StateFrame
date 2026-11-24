"""Independent, stdlib-only OTLP/HTTP JSON and Protocol Buffer codecs."""
from __future__ import annotations

import json
import math
import struct
from typing import Literal

from .. import __version__
from .levels import valid_version
from .schema import validate_batch
from .types import ExportResult, ResultKind, Reason, SafeAttr, SafeEvent, SafeLink, SafeSpan, Signal, SignalBatch


Protocol = Literal["http/protobuf", "http/json"]
MAX_BATCH_ITEMS = 256
MAX_BATCH_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MAX_RESPONSE_FIELDS = 4096
MAX_PROTO_DEPTH = 16
_UINT64_MAX = (1 << 64) - 1


def encode(batch: SignalBatch, protocol: Protocol) -> bytes:
    """Encode one validated closed batch, rejecting oversized wire payloads."""
    validate_batch(batch)
    if protocol not in ("http/protobuf", "http/json") or not valid_version(__version__):
        raise ValueError("invalid_field")
    payload = _encode_json(batch) if protocol == "http/json" else _encode_protobuf(batch)
    if len(payload) > MAX_BATCH_BYTES:
        raise ValueError("payload_rejected")
    return payload


def decode_success(body: bytes, protocol: Protocol, signal: Signal, items: int) -> ExportResult:
    """Parse only the bounded OTLP partial-success fields needed for accounting."""
    if type(body) is not bytes or len(body) > MAX_RESPONSE_BYTES or type(signal) is not Signal:
        raise ValueError("bad_response")
    if type(items) is not int or not 1 <= items <= MAX_BATCH_ITEMS:
        raise ValueError("bad_response")
    try:
        if protocol == "http/json":
            rejected, warning = _decode_json_success(body or b"{}")
        elif protocol == "http/protobuf":
            rejected, warning = _decode_proto_success(body)
        else:
            raise ValueError("bad_response")
        if rejected < 0 or rejected > items:
            raise ValueError("bad_response")
        accepted = items - rejected
        if rejected:
            return ExportResult(ResultKind.PARTIAL, accepted, rejected, Reason.PARTIAL_REJECTION, None, False)
        if warning:
            return ExportResult(ResultKind.WARNING, items, 0, Reason.RECEIVER_WARNING, None, False)
        return ExportResult(ResultKind.ACCEPTED, items, 0, None, None, False)
    except ValueError as error:
        if str(error) == "bad_response":
            raise
        raise ValueError("bad_response") from None
    except (UnicodeError, RecursionError, OverflowError, TypeError):
        raise ValueError("bad_response") from None


def _encode_json(batch: SignalBatch) -> bytes:
    resource_spans = [{
        "resource": _json_resource(batch.resource_id),
        "scopeSpans": [{
            "scope": {"name": "stateframe.otel", "version": "1"},
            "spans": [_json_span(span) for span in batch.spans],
        }],
    }]
    value = {"resourceSpans": resource_spans}
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _json_resource(resource_id: bytes) -> dict[str, object]:
    attributes = (
        ("service.instance.id", resource_id.hex()),
        ("service.name", "stateframe"),
        ("service.version", __version__),
    )
    return {"attributes": [{"key": key, "value": {"stringValue": value}} for key, value in attributes]}


def _json_attr(attr: SafeAttr) -> dict[str, object]:
    value = attr.value
    if type(value) is bool:
        any_value: dict[str, object] = {"boolValue": value}
    elif type(value) is int:
        any_value = {"intValue": str(value)}
    elif type(value) is float:
        any_value = {"doubleValue": value}
    else:
        any_value = {"stringValue": value}
    return {"key": str(attr.key), "value": any_value}


def _json_span(span: SafeSpan) -> dict[str, object]:
    value: dict[str, object] = {
        "traceId": span.trace_id.hex(),
        "spanId": span.span_id.hex(),
        "name": span.name,
        "kind": 3 if span.name == "chat" else 1,
        "startTimeUnixNano": str(span.start_ns),
        "endTimeUnixNano": str(span.end_ns),
        "attributes": [_json_attr(attr) for attr in span.attrs],
        "flags": 1,
    }
    if span.parent_id is not None:
        value["parentSpanId"] = span.parent_id.hex()
    if span.dropped_attrs:
        value["droppedAttributesCount"] = span.dropped_attrs
    if span.events:
        value["events"] = [_json_event(event) for event in span.events]
    if span.dropped_events:
        value["droppedEventsCount"] = span.dropped_events
    if span.links:
        value["links"] = [_json_link(link) for link in span.links]
    if span.dropped_links:
        value["droppedLinksCount"] = span.dropped_links
    if span.error:
        value["status"] = {"code": 2}
    return value


def _json_event(event: SafeEvent) -> dict[str, object]:
    value: dict[str, object] = {
        "timeUnixNano": str(event.at_ns),
        "name": event.name,
        "attributes": [_json_attr(attr) for attr in event.attrs],
    }
    if event.dropped_attrs:
        value["droppedAttributesCount"] = event.dropped_attrs
    return value


def _json_link(link: SafeLink) -> dict[str, object]:
    return {"traceId": link.trace_id.hex(), "spanId": link.span_id.hex(), "flags": 1}


def _encode_protobuf(batch: SignalBatch) -> bytes:
    resource = _pb_resource(batch.resource_id)
    scope = _pb_scope()
    scope_spans = _pb_bytes(1, scope) + b"".join(_pb_bytes(2, _pb_span(span)) for span in batch.spans)
    resource_spans = _pb_bytes(1, resource) + _pb_bytes(2, scope_spans)
    return _pb_bytes(1, resource_spans)


def _pb_resource(resource_id: bytes) -> bytes:
    attributes = (
        ("service.instance.id", resource_id.hex()),
        ("service.name", "stateframe"),
        ("service.version", __version__),
    )
    return b"".join(_pb_bytes(1, _pb_key_value(key, value)) for key, value in attributes)


def _pb_scope() -> bytes:
    return _pb_bytes(1, b"stateframe.otel") + _pb_bytes(2, b"1")


def _pb_key_value(key: str, value: bool | int | float | str) -> bytes:
    if type(value) is bool:
        any_value = _pb_uint(2, int(value))
    elif type(value) is int:
        any_value = _pb_uint(3, value)
    elif type(value) is float:
        any_value = _pb_double(4, value)
    else:
        any_value = _pb_bytes(1, value.encode("utf-8"))
    return _pb_bytes(1, key.encode("utf-8")) + _pb_bytes(2, any_value)


def _pb_attrs(attrs: tuple[SafeAttr, ...], tag: int) -> bytes:
    return b"".join(_pb_bytes(tag, _pb_key_value(str(attr.key), attr.value)) for attr in attrs)


def _pb_span(span: SafeSpan) -> bytes:
    value = _pb_bytes(1, span.trace_id) + _pb_bytes(2, span.span_id)
    if span.parent_id is not None:
        value += _pb_bytes(4, span.parent_id)
    value += _pb_bytes(5, span.name.encode("utf-8"))
    value += _pb_uint(6, 3 if span.name == "chat" else 1)
    value += _pb_fixed64(7, span.start_ns) + _pb_fixed64(8, span.end_ns)
    value += _pb_attrs(span.attrs, 9)
    if span.dropped_attrs:
        value += _pb_uint(10, span.dropped_attrs)
    for event in span.events:
        value += _pb_bytes(11, _pb_event(event))
    if span.dropped_events:
        value += _pb_uint(12, span.dropped_events)
    for link in span.links:
        value += _pb_bytes(13, _pb_link(link))
    if span.dropped_links:
        value += _pb_uint(14, span.dropped_links)
    if span.error:
        value += _pb_bytes(15, _pb_uint(3, 2))
    value += _pb_fixed32(16, 1)
    return value


def _pb_event(event: SafeEvent) -> bytes:
    value = _pb_fixed64(1, event.at_ns) + _pb_bytes(2, event.name.encode("utf-8")) + _pb_attrs(event.attrs, 3)
    if event.dropped_attrs:
        value += _pb_uint(4, event.dropped_attrs)
    return value


def _pb_link(link: SafeLink) -> bytes:
    return _pb_bytes(1, link.trace_id) + _pb_bytes(2, link.span_id) + _pb_fixed32(6, 1)


def _pb_uint(tag: int, value: int) -> bytes:
    return _pb_key(tag, 0) + _varint(value)


def _pb_bytes(tag: int, value: bytes | str) -> bytes:
    data = value.encode("utf-8") if type(value) is str else value
    return _pb_key(tag, 2) + _varint(len(data)) + data


def _pb_fixed64(tag: int, value: int) -> bytes:
    return _pb_key(tag, 1) + struct.pack("<Q", value)


def _pb_double(tag: int, value: float) -> bytes:
    return _pb_key(tag, 1) + struct.pack("<d", value)


def _pb_fixed32(tag: int, value: int) -> bytes:
    return _pb_key(tag, 5) + struct.pack("<I", value)


def _pb_key(tag: int, wire: int) -> bytes:
    return _varint((tag << 3) | wire)


def _varint(value: int) -> bytes:
    if type(value) is not int:
        raise ValueError("invalid_field")
    value &= _UINT64_MAX
    encoded = bytearray()
    while value >= 0x80:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _decode_json_success(body: bytes) -> tuple[int, bool]:
    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_bad_constant)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError):
        raise ValueError("bad_response") from None
    if type(value) is not dict:
        raise ValueError("bad_response")
    _check_response_tree(value)
    if not value:
        return 0, False
    if "partialSuccess" not in value:
        raise ValueError("bad_response")
    partial = value["partialSuccess"]
    if type(partial) is not dict:
        raise ValueError("bad_response")
    rejected = _json_uint(partial.get("rejectedSpans", "0"))
    warning = False
    if "errorMessage" in partial:
        text = partial["errorMessage"]
        if type(text) is not str:
            raise ValueError("bad_response")
        warning = bool(text)
    return rejected, warning


def _decode_proto_success(body: bytes) -> tuple[int, bool]:
    partial_seen = False
    rejected = 0
    warning = False
    budget = [0]
    for tag, wire, value in _proto_fields(body, budget):
        if tag == 1:
            if wire != 2:
                raise ValueError("bad_response")
            inner_rejected, inner_warning = _decode_proto_partial(value, budget)
            if inner_rejected is not None:
                rejected = inner_rejected
            if inner_warning is not None:
                warning = inner_warning
            partial_seen = True
    return (rejected if partial_seen else 0), warning


def _decode_proto_partial(data: bytes, budget: list[int]) -> tuple[int | None, bool | None]:
    rejected: int | None = None
    warning = None
    for tag, wire, value in _proto_fields(data, budget):
        if tag == 1:
            if wire != 0 or type(value) is not int or value > _UINT64_MAX:
                raise ValueError("bad_response")
            rejected = value
        elif tag == 2:
            if wire != 2 or type(value) is not bytes:
                raise ValueError("bad_response")
            value.decode("utf-8")
            warning = bool(value)
    return rejected, warning


def _proto_fields(data: bytes, budget: list[int]):
    offset = 0
    while offset < len(data):
        budget[0] += 1
        if budget[0] > MAX_RESPONSE_FIELDS:
            raise ValueError("bad_response")
        key, offset = _read_varint(data, offset)
        tag, wire = key >> 3, key & 7
        if tag == 0 or tag > (1 << 29) - 1:
            raise ValueError("bad_response")
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 1:
            end = offset + 8
            if end > len(data):
                raise ValueError("bad_response")
            value = data[offset:end]
            offset = end
        elif wire == 2:
            length, offset = _read_varint(data, offset)
            if length > MAX_RESPONSE_BYTES or offset + length > len(data):
                raise ValueError("bad_response")
            value = data[offset:offset + length]
            offset += length
        elif wire == 5:
            end = offset + 4
            if end > len(data):
                raise ValueError("bad_response")
            value = data[offset:end]
            offset = end
        else:
            raise ValueError("bad_response")
        yield tag, wire, value


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for index in range(10):
        if offset >= len(data):
            raise ValueError("bad_response")
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise ValueError("bad_response")
        value |= (byte & 0x7F) << (index * 7)
        if byte < 0x80:
            return value, offset
    raise ValueError("bad_response")


def _json_uint(value: object) -> int:
    if type(value) is int:
        return value
    if type(value) is not str or not value or not value.isascii() or not value.isdecimal():
        raise ValueError("bad_response")
    if len(value) > 20 or (len(value) > 1 and value[0] == "0"):
        raise ValueError("bad_response")
    return int(value)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("bad_response")
        result[key] = value
    return result


def _bad_constant(_value: str):
    raise ValueError("bad_response")


def _check_response_tree(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    fields = 0
    while pending:
        item, depth = pending.pop()
        if depth > MAX_PROTO_DEPTH:
            raise ValueError("bad_response")
        if type(item) is dict:
            fields += len(item)
            if fields > MAX_RESPONSE_FIELDS:
                raise ValueError("bad_response")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            fields += len(item)
            if fields > MAX_RESPONSE_FIELDS:
                raise ValueError("bad_response")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("bad_response")
