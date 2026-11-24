from __future__ import annotations

import json
import hashlib
from dataclasses import replace

import pytest

from stateframe import __version__
from stateframe.otel import codec
from stateframe.otel.schema import validate_batch
from stateframe.otel.types import (
    AttrKey,
    SafeAttr,
    SafeEvent,
    SafeLink,
    SafeSpan,
    Signal,
    SignalBatch,
)
from tests.otlp_receiver import assert_valid_traces


TRACE_ID = bytes.fromhex("00112233445566778899aabbccddeeff")
RESOURCE_ID = bytes.fromhex("ffeeddccbbaa99887766554433221100")
SPAN_ID = bytes.fromhex("0102030405060708")
PARENT_ID = bytes.fromhex("1112131415161718")
START = 1_730_000_000_000_000_001
END = 1_730_000_000_123_456_789


def _attrs(*extra: SafeAttr) -> tuple[SafeAttr, ...]:
    return tuple(sorted((
        SafeAttr(AttrKey.RUN_ID, "10112233445566778899aabbccddeeff"),
        SafeAttr(AttrKey.ATTEMPT, 1),
        SafeAttr(AttrKey.SOURCE, "record"),
        SafeAttr(AttrKey.TIMING, "exact"),
        SafeAttr(AttrKey.COMPLETE, True),
        *extra,
    ), key=lambda item: str(item.key)))


def _span() -> SafeSpan:
    return SafeSpan(
        "stateframe.model.request",
        TRACE_ID,
        SPAN_ID,
        PARENT_ID,
        START,
        END,
        _attrs(
            SafeAttr(AttrKey.ROLE, "edit"),
            SafeAttr(AttrKey.HTTP_STATUS_CODE, 200),
            SafeAttr(AttrKey.GEN_AI_OPERATION_NAME, "chat"),
        ),
        (SafeEvent("retry_scheduled", END, (SafeAttr(AttrKey.RETRY, "transport"),), 0),),
        (SafeLink(TRACE_ID, b"\x02" * 8),),
        0,
        0,
        0,
        True,
    )


def _trace_batch() -> SignalBatch:
    return SignalBatch(Signal.TRACES, RESOURCE_ID, (_span(),))


def _field_map(data: bytes) -> list[tuple[int, int, bytes | int]]:
    result: list[tuple[int, int, bytes | int]] = []
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        tag, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 2:
            size, offset = _read_varint(data, offset)
            value = data[offset:offset + size]
            offset += size
        elif wire == 1:
            value = data[offset:offset + 8]
            offset += 8
        elif wire == 5:
            value = data[offset:offset + 4]
            offset += 4
        else:
            raise AssertionError(f"unexpected wire type {wire}")
        result.append((tag, wire, value))
    return result


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise AssertionError("malformed protobuf test vector")


def test_json_ids_enums_and_int64_encoding():
    trace_payload = json.loads(codec.encode(_trace_batch(), "http/json"))
    assert_valid_traces(trace_payload)
    resource = trace_payload["resourceSpans"][0]["resource"]
    resource_attrs = {item["key"]: item["value"] for item in resource["attributes"]}
    assert resource_attrs["service.name"] == {"stringValue": "stateframe"}
    assert resource_attrs["service.version"] == {"stringValue": __version__}
    assert resource_attrs["service.instance.id"] == {"stringValue": RESOURCE_ID.hex()}

    span = trace_payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == TRACE_ID.hex()
    assert span["spanId"] == SPAN_ID.hex()
    assert span["parentSpanId"] == PARENT_ID.hex()
    assert span["kind"] == 1
    assert span["startTimeUnixNano"] == str(START)
    assert span["endTimeUnixNano"] == str(END)
    assert span["status"]["code"] == 2
    assert span["flags"] == 1

def test_protobuf_golden_trace(monkeypatch):
    # Exercise the installed version separately, then freeze the golden fixture
    # so changes to the Release metadata do not require new Protobuf vectors.
    request = _field_map(codec.encode(_trace_batch(), "http/protobuf"))
    resource = _field_map(_field_map(request[0][2])[0][2])
    attributes = {_field_map(item)[0][2]: _field_map(_field_map(item)[1][2])[0][2]
                  for _, _, item in resource}
    assert attributes[b"service.version"] == __version__.encode("ascii")
    monkeypatch.setattr(codec, "__version__", "1.2.3")
    trace_wire = codec.encode(_trace_batch(), "http/protobuf")
    assert hashlib.sha256(trace_wire).hexdigest() == "566612d54d070d62657266741e00b7d7d252e61b04baaac4cf861d745482e848"

    # Independent decoding checks pinned field numbers and wire types, rather than
    # relying on the encoder's local JSON mapping or a runtime protobuf package.
    request = _field_map(trace_wire)
    assert [(tag, wire) for tag, wire, _ in request] == [(1, 2)]
    resource_spans = _field_map(request[0][2])
    assert [tag for tag, _wire, _value in resource_spans] == [1, 2]
    scope_spans = _field_map(resource_spans[1][2])
    assert [tag for tag, _wire, _value in scope_spans] == [1, 2]
    encoded_span = _field_map(scope_spans[1][2])
    span_by_tag = {tag: (wire, value) for tag, wire, value in encoded_span}
    assert span_by_tag[1] == (2, TRACE_ID)
    assert span_by_tag[2] == (2, SPAN_ID)
    assert span_by_tag[4] == (2, PARENT_ID)
    assert span_by_tag[6] == (0, 1)
    assert span_by_tag[7][0] == span_by_tag[8][0] == 1
    span_attributes = [value for tag, wire, value in encoded_span if tag == 9 and wire == 2]
    assert span_attributes
    encoded_event = _field_map(next(value for tag, _wire, value in encoded_span if tag == 11))
    assert any(tag == 3 and wire == 2 for tag, wire, _value in encoded_event)
    assert span_by_tag[15][0] == 2

def test_partial_success_never_retries_batch():
    response = b'{"partialSuccess":{"rejectedSpans":"1","errorMessage":"private-marker"}}'
    result = codec.decode_success(response, "http/json", Signal.TRACES, 3)
    assert result.kind.value == "partial"
    assert result.accepted == 2
    assert result.rejected == 1
    assert result.reason.value == "partial_rejection"
    assert "private-marker" not in repr(result)

    warning = codec.decode_success(
        b'{"partialSuccess":{"errorMessage":"warning-marker"}}',
        "http/json",
        Signal.TRACES,
        3,
    )
    assert warning.kind.value == "warning"
    assert warning.accepted == 3 and warning.rejected == 0
    assert "warning-marker" not in repr(warning)

    assert codec.decode_success(b"", "http/protobuf", Signal.TRACES, 1).kind.value == "accepted"
    assert codec.decode_success(b"", "http/json", Signal.TRACES, 1).kind.value == "accepted"


def test_json_partial_success_accepts_numeric_int64():
    response = json.dumps({"partialSuccess": {"rejectedSpans": 1}}).encode()
    result = codec.decode_success(response, "http/json", Signal.TRACES, 3)
    assert (result.kind.value, result.accepted, result.rejected) == ("partial", 2, 1)


@pytest.mark.parametrize("protocol,response", [
    ("http/json", b'{"partialSuccess":{"rejectedSpans":"0","errorMessage":""}}'),
    ("http/protobuf", b"\x0a\x04\x08\x00\x12\x00"),
])
def test_empty_partial_success_message_is_full_success(protocol, response):
    result = codec.decode_success(response, protocol, Signal.TRACES, 3)
    assert (result.kind.value, result.accepted, result.reason) == ("accepted", 3, None)


def test_wire_rejects_invalid_typed_values():
    batch = _trace_batch()
    invalid = replace(batch.spans[0], attrs=_attrs(SafeAttr(AttrKey.OUTCOME, "not-a-public-outcome")))
    bad_batch = replace(batch, spans=(invalid,))
    with pytest.raises(ValueError, match="^invalid_field$"):
        validate_batch(bad_batch)
    with pytest.raises(ValueError, match="^invalid_field$"):
        codec.encode(bad_batch, "http/json")


def test_decode_rejects_counts_outside_batch_and_ignores_bounded_unknown_fields():
    with pytest.raises(ValueError, match="^bad_response$"):
        codec.decode_success(
            b'{"partialSuccess":{"rejectedSpans":"2"}}',
            "http/json",
            Signal.TRACES,
            1,
        )

    # Protobuf field 99 is an unknown varint and must be ignored within bounds.
    body = bytes((0x98, 0x06, 0x01))
    assert codec.decode_success(body, "http/protobuf", Signal.TRACES, 1).kind.value == "accepted"
