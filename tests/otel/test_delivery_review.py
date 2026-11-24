from __future__ import annotations

import gzip
import json
import os
import socket
import ssl
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from stateframe import __version__
from stateframe.otel import codec, config, levels
from stateframe.otel.levels import model_id
from stateframe.otel.schema import validate_batch
from stateframe.otel.transport import TransportSettings, send
from stateframe.otel.types import (
    AttrKey,
    ResultKind,
    SafeAttr,
    SafeEvent,
    SafeLink,
    SafeSpan,
    Signal,
    SignalBatch,
)
from tests.otlp_receiver import OTLPReceiver, ResponseStep


_TRACE_ID = bytes.fromhex("102132435465768798a9bacbdcedfe0f")
_SPAN_ID = bytes.fromhex("1122334455667788")
_PARENT_ID = bytes.fromhex("8877665544332211")
_LINK_ID = bytes.fromhex("0102030405060708")
_RESOURCE_ID = bytes.fromhex("ffeeddccbbaa99887766554433221100")
_START = 1_730_000_000_000_000_001
_END = 1_730_000_000_123_456_789
_KEY = bytes(range(32))
_DEST = b"d" * 16
_RUN = (bytes.fromhex("10112233445566778899aabbccddeeff"))


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for index in range(10):
        if offset >= len(data):
            raise AssertionError("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        if index == 9 and byte > 1:
            raise AssertionError("protobuf varint exceeds uint64")
        value |= (byte & 0x7F) << (7 * index)
        if byte < 0x80:
            return value, offset
    raise AssertionError("protobuf varint exceeds ten bytes")


def _proto_fields(data: bytes) -> list[tuple[int, int, bytes | int]]:
    """Decode wire framing independently using public OTLP field numbers."""
    fields = []
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        tag, wire = key >> 3, key & 7
        if tag == 0:
            raise AssertionError("protobuf field zero")
        if wire == 0:
            value, offset = _read_varint(data, offset)
        elif wire == 1:
            value = data[offset:offset + 8]
            offset += 8
            if len(value) != 8:
                raise AssertionError("truncated fixed64")
        elif wire == 2:
            size, offset = _read_varint(data, offset)
            value = data[offset:offset + size]
            offset += size
            if len(value) != size:
                raise AssertionError("truncated bytes field")
        elif wire == 5:
            value = data[offset:offset + 4]
            offset += 4
            if len(value) != 4:
                raise AssertionError("truncated fixed32")
        else:
            raise AssertionError(f"unexpected protobuf wire type {wire}")
        fields.append((tag, wire, value))
    return fields


def _one(fields: list[tuple[int, int, bytes | int]], tag: int, wire: int):
    matches = [(actual_wire, value) for actual_tag, actual_wire, value in fields if actual_tag == tag]
    assert len(matches) == 1 and matches[0][0] == wire
    return matches[0][1]


def _many(fields: list[tuple[int, int, bytes | int]], tag: int, wire: int) -> list[bytes | int]:
    matches = [(actual_wire, value) for actual_tag, actual_wire, value in fields if actual_tag == tag]
    assert all(actual_wire == wire for actual_wire, _value in matches)
    return [value for _actual_wire, value in matches]


def _text(value: bytes | int) -> str:
    assert type(value) is bytes
    return value.decode("utf-8")


def _as_number(fields: list[tuple[int, int, bytes | int]], string_tag: int,
               bool_tag: int, int_tag: int, double_tag: int):
    candidates = ((string_tag, 2), (bool_tag, 0), (int_tag, 0), (double_tag, 1))
    found = [(tag, wire) for tag, wire in candidates if any(a == tag for a, _w, _v in fields)]
    assert len(found) == 1
    tag, wire = found[0]
    raw = _one(fields, tag, wire)
    if tag == string_tag:
        return _text(raw)
    if tag == bool_tag:
        return bool(raw)
    if tag == int_tag:
        assert type(raw) is int
        return raw
    assert type(raw) is bytes
    return struct.unpack("<d", raw)[0]


def _proto_attrs(values: list[bytes | int]) -> dict[str, object]:
    result = {}
    for raw in values:
        assert type(raw) is bytes
        key_value = _proto_fields(raw)
        key = _text(_one(key_value, 1, 2))
        any_value = _proto_fields(_one(key_value, 2, 2))
        result[key] = _as_number(any_value, 1, 2, 3, 4)
    return result


def _json_attrs(values: list[dict[str, object]]) -> dict[str, object]:
    result = {}
    for item in values:
        key = item["key"]
        value = item["value"]
        assert type(key) is str and type(value) is dict
        if "stringValue" in value:
            decoded = value["stringValue"]
        elif "boolValue" in value:
            decoded = value["boolValue"]
        elif "intValue" in value:
            decoded = int(value["intValue"])
        else:
            decoded = value["doubleValue"]
        result[key] = decoded
    return result


def _proto_resource(raw: bytes) -> dict[str, object]:
    fields = _proto_fields(raw)
    return _proto_attrs(_many(fields, 1, 2))


def _proto_trace(raw: bytes) -> dict[str, object]:
    request = _proto_fields(raw)
    resource_spans = _proto_fields(_one(request, 1, 2))
    resource = _proto_resource(_one(resource_spans, 1, 2))
    scope_spans = _proto_fields(_one(resource_spans, 2, 2))
    scope = _proto_fields(_one(scope_spans, 1, 2))
    spans = []
    for raw_span in _many(scope_spans, 2, 2):
        span = _proto_fields(raw_span)
        result: dict[str, object] = {
            "trace_id": _one(span, 1, 2),
            "span_id": _one(span, 2, 2),
            "name": _text(_one(span, 5, 2)),
            "kind": _one(span, 6, 0),
            "start": struct.unpack("<Q", _one(span, 7, 1))[0],
            "end": struct.unpack("<Q", _one(span, 8, 1))[0],
            "attributes": _proto_attrs(_many(span, 9, 2)),
            "flags": struct.unpack("<I", _one(span, 16, 5))[0],
        }
        if any(tag == 4 for tag, _wire, _value in span):
            result["parent_id"] = _one(span, 4, 2)
        for source, target in ((10, "dropped_attrs"), (12, "dropped_events"), (14, "dropped_links")):
            if any(tag == source for tag, _wire, _value in span):
                result[target] = _one(span, source, 0)
        events = []
        for raw_event in _many(span, 11, 2):
            event = _proto_fields(raw_event)
            item: dict[str, object] = {
                "time": struct.unpack("<Q", _one(event, 1, 1))[0],
                "name": _text(_one(event, 2, 2)),
                "attributes": _proto_attrs(_many(event, 3, 2)),
            }
            if any(tag == 4 for tag, _wire, _value in event):
                item["dropped_attrs"] = _one(event, 4, 0)
            events.append(item)
        if events:
            result["events"] = events
        links = []
        for raw_link in _many(span, 13, 2):
            link = _proto_fields(raw_link)
            links.append({
                "trace_id": _one(link, 1, 2),
                "span_id": _one(link, 2, 2),
                "flags": struct.unpack("<I", _one(link, 6, 5))[0],
            })
        if links:
            result["links"] = links
        if any(tag == 15 for tag, _wire, _value in span):
            status = _proto_fields(_one(span, 15, 2))
            result["status"] = _one(status, 3, 0)
        spans.append(result)
    return {
        "resource": resource,
        "scope": (_text(_one(scope, 1, 2)), _text(_one(scope, 2, 2))),
        "spans": spans,
    }


def _json_trace(raw: bytes) -> dict[str, object]:
    request = json.loads(raw)
    resource_spans = request["resourceSpans"][0]
    scope_spans = resource_spans["scopeSpans"][0]
    spans = []
    for span in scope_spans["spans"]:
        item: dict[str, object] = {
            "trace_id": bytes.fromhex(span["traceId"]),
            "span_id": bytes.fromhex(span["spanId"]),
            "name": span["name"],
            "kind": span["kind"],
            "start": int(span["startTimeUnixNano"]),
            "end": int(span["endTimeUnixNano"]),
            "attributes": _json_attrs(span["attributes"]),
            "flags": span["flags"],
        }
        if "parentSpanId" in span:
            item["parent_id"] = bytes.fromhex(span["parentSpanId"])
        for source, target in (("droppedAttributesCount", "dropped_attrs"),
                               ("droppedEventsCount", "dropped_events"),
                               ("droppedLinksCount", "dropped_links")):
            if source in span:
                item[target] = span[source]
        if "events" in span:
            item["events"] = [{
                "time": int(event["timeUnixNano"]),
                "name": event["name"],
                "attributes": _json_attrs(event["attributes"]),
                **({"dropped_attrs": event["droppedAttributesCount"]}
                   if "droppedAttributesCount" in event else {}),
            } for event in span["events"]]
        if "links" in span:
            item["links"] = [{
                "trace_id": bytes.fromhex(link["traceId"]),
                "span_id": bytes.fromhex(link["spanId"]),
                "flags": link["flags"],
            } for link in span["links"]]
        if "status" in span:
            item["status"] = span["status"]["code"]
        spans.append(item)
    resource = resource_spans["resource"]
    return {
        "resource": _json_attrs(resource["attributes"]),
        "scope": (scope_spans["scope"]["name"], scope_spans["scope"]["version"]),
        "spans": spans,
    }


def _trace_batch() -> SignalBatch:
    attrs = tuple(sorted((
        SafeAttr(AttrKey.RUN_ID, _RUN.hex()),
        SafeAttr(AttrKey.ATTEMPT, 2),
        SafeAttr(AttrKey.SOURCE, "record"),
        SafeAttr(AttrKey.TIMING, "exact"),
        SafeAttr(AttrKey.COMPLETE, True),
        SafeAttr(AttrKey.REQUEST_MODEL, model_id(b"k" * 32, "local-model-8b")),
        SafeAttr(AttrKey.PROVIDER_NAME, "local"),
        SafeAttr(AttrKey.HTTP_STATUS_CODE, 201),
    ), key=lambda attr: str(attr.key)))
    span = SafeSpan(
        "chat", _TRACE_ID, _SPAN_ID, _PARENT_ID, _START, _END, attrs,
        (SafeEvent("retry_scheduled", _END, (SafeAttr(AttrKey.RETRY, "transport"),), 2),),
        (SafeLink(bytes.fromhex("20112233445566778899aabbccddeeff"), _LINK_ID),),
        1, 1, 1, True,
    )
    return SignalBatch(Signal.TRACES, _RESOURCE_ID, (span,))


def test_protobuf_and_json_match_independent_public_field_decoder() -> None:
    traces = _trace_batch()
    assert _proto_trace(codec.encode(traces, "http/protobuf")) == _json_trace(codec.encode(traces, "http/json"))


def test_free_text_model_names_are_rejected_at_every_validation_boundary() -> None:
    value = "alpha-secret-canary"
    assert not levels.valid_attribute(AttrKey.REQUEST_MODEL.value, value)
    assert not levels.valid_attribute(AttrKey.RESPONSE_MODEL.value, value)
    original = _trace_batch().spans[0]
    attrs = tuple(sorted((
        *(attr for attr in original.attrs if attr.key is not AttrKey.REQUEST_MODEL),
        SafeAttr(AttrKey.REQUEST_MODEL, value),
    ), key=lambda attr: str(attr.key)))
    unsafe = SignalBatch(Signal.TRACES, _RESOURCE_ID, (SafeSpan(
        original.name, original.trace_id, original.span_id, original.parent_id,
        original.start_ns, original.end_ns, attrs, original.events, original.links,
        original.dropped_attrs, original.dropped_events, original.dropped_links, original.error,
    ),))
    with pytest.raises(ValueError, match="^invalid_field$"):
        validate_batch(unsafe)


@dataclass(frozen=True)
class _LevelRecord:
    request_model: str = "local-model-8b"
    response_model: str | None = None
    provider: str = "local"
    engine_version: str = __version__
    reported_cost: float = 0.25
    price_table: None = None
    usage: None = None


def test_export_defaults_keep_model_and_require_explicit_detail_opt_ins() -> None:
    resolved = config.resolve(config.Settings(), {})
    assert resolved.traces is not None
    assert resolved.levels == levels.Levels()
    baseline = dict(levels.extra_attributes(resolved.levels, _LevelRecord(), b"k" * 32))
    assert baseline[AttrKey.REQUEST_MODEL.value] == model_id(b"k" * 32, "local-model-8b")
    assert AttrKey.COST_VALUE.value not in baseline

    enabled = levels.Levels(include_cost=True)
    detailed = dict(levels.extra_attributes(enabled, _LevelRecord(), b"k" * 32))
    assert detailed[AttrKey.COST_VALUE.value] == 250_000
    assert "local-model-8b" not in repr(detailed)


def _transport_settings(endpoint: str, *, compression: str = "none",
                        headers: tuple[tuple[str, str], ...] = ()) -> TransportSettings:
    return TransportSettings(endpoint, "http/json", headers, compression, 2_000, None, None, None)


def _deadline(seconds: float = 3.0) -> int:
    return time.monotonic_ns() + int(seconds * 1_000_000_000)


@pytest.mark.parametrize("status", [429, 503])
def test_transient_status_honors_retry_after_without_exposing_response(monkeypatch, status: int) -> None:
    delays = []
    monkeypatch.setattr("stateframe.otel.transport.time.sleep", delays.append)
    name = f"http_{status}"
    with OTLPReceiver([ResponseStep(name, retry_after="0.01"), "accept"]) as receiver:
        result = send(_trace_batch(), _transport_settings(receiver.url + "/v1/traces"), _deadline())
        assert result.kind is ResultKind.ACCEPTED
        assert len(receiver.requests) == 2
    assert delays and delays[0] >= 0.01
    assert "collector rejected request" not in repr(result)


def test_partial_success_is_terminal_and_redirect_is_refused() -> None:
    with OTLPReceiver(["partial_success", "accept"]) as receiver:
        result = send(_trace_batch(), _transport_settings(receiver.url + "/v1/traces"), _deadline())
        assert result.kind is ResultKind.PARTIAL
        assert len(receiver.requests) == 1

    with OTLPReceiver(["redirect_same_origin"]) as receiver:
        result = send(_trace_batch(), _transport_settings(receiver.url + "/v1/traces"), _deadline())
        assert result.reason.value == "redirect"
        assert len(receiver.requests) == 1


def test_gzip_wire_and_header_error_secrecy(tmp_path: Path) -> None:
    with OTLPReceiver() as receiver:
        result = send(_trace_batch(), _transport_settings(receiver.url + "/v1/traces", compression="gzip"),
                      _deadline())
        assert result.kind is ResultKind.ACCEPTED
        request = receiver.requests[0]
        assert request.headers["content-encoding"] == "gzip"
        assert int(request.headers["content-length"]) == len(request.raw_body)
        assert json.loads(gzip.decompress(request.raw_body))["resourceSpans"]

    marker = "private-header-canary"
    settings = _transport_settings("http://127.0.0.1:4318/v1/traces?tenant=private-query-canary",
                                   headers=(("x-tenant", marker),))
    result = send(_trace_batch(), settings, _deadline())
    assert result.reason.value == "invalid_config"
    assert marker not in repr(settings) + repr(result)
    assert "private-query-canary" not in repr(settings) + repr(result)


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1:4318/v1/traces?token=private-query-canary",
    "http://127.0.0.1:4318/v1/traces#private-fragment-canary",
    "http://user@127.0.0.1:4318/v1/traces",
    "https://collector.example.test/v1/traces?tenant=private-query-canary",
    "https://collector.example.test/v1/traces#private-fragment-canary",
    "https://user@collector.example.test/v1/traces",
])
def test_transport_rejects_query_fragment_and_userinfo_for_every_scheme(endpoint: str) -> None:
    result = send(_trace_batch(), _transport_settings(endpoint), _deadline())
    assert result.reason.value == "invalid_config"

def test_tls_verification_and_client_files_are_applied_without_downgrade(tmp_path: Path, monkeypatch) -> None:
    original_resolve = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        if host == "collector.test":
            return original_resolve("127.0.0.1", port, *args, **kwargs)
        return original_resolve(host, port, *args, **kwargs)

    monkeypatch.setattr("stateframe.otel.transport.socket.getaddrinfo", resolve)
    ca, cert, key_file = (tmp_path / name for name in ("ca.pem", "client.pem", "client.key"))
    for path in (ca, cert, key_file):
        path.write_text("fixture", encoding="ascii")
        path.chmod(0o600)

    class _TLSContext:
        check_hostname = True
        verify_mode = ssl.CERT_REQUIRED

        def __init__(self):
            self.ca = None
            self.client_files = None
            self.server_name = None

        def load_verify_locations(self, *, cafile):
            self.ca = cafile

        def load_cert_chain(self, certfile, keyfile):
            self.client_files = (certfile, keyfile)

        def wrap_socket(self, sock, *, server_hostname):
            self.server_name = server_hostname
            return sock

    contexts = []

    def create_context(**_kwargs):
        context = _TLSContext()
        contexts.append(context)
        return context

    monkeypatch.setattr("stateframe.otel.transport.ssl.create_default_context", create_context)
    with OTLPReceiver() as receiver:
        port = urlsplit(receiver.url).port
        assert port is not None
        settings = TransportSettings(
            f"https://collector.test:{port}/v1/traces", "http/json", (), "none", 2_000,
            ca, cert, key_file,
        )
        result = send(_trace_batch(), settings, _deadline())

    assert result.kind is ResultKind.ACCEPTED
    assert contexts[0].check_hostname is True
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert contexts[0].ca == str(ca)
    assert contexts[0].client_files == (str(cert), str(key_file))
    assert contexts[0].server_name == "collector.test"
