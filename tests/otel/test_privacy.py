from __future__ import annotations

import dataclasses
import http.client
import json
from urllib.parse import urlsplit
from contextlib import closing

import pytest

from stateframe.otel.projection import Projector
from stateframe.otel.levels import model_id, valid_attribute
from stateframe.otel.schema import coverage_merge, pack_local, unpack_local, validate_batch
from stateframe.otel.store import Store
from stateframe.otel.types import (
    AttrKey,
    Coverage,
    Cursor,
    Namespace,
    RunRef,
    Reason,
    SafeAttr,
    SafeSpan,
    Signal,
    SignalBatch,
    SourceOccurrence,
)
from tests.otlp_receiver import OTLPReceiver, assert_capture_clean


KEY = bytes(range(32))
RUN = RunRef(bytes.fromhex("10112233445566778899aabbccddeeff"))
TRACE_ID = bytes.fromhex("20112233445566778899aabbccddeeff")
SPACE = Namespace(RUN, 1, None, None)
EPOCH = 1_730_000_000_000_000_000


def _occurrence(row: dict[str, object], family: str = "journal", *, index: int = 1,
                epoch_ns: int = EPOCH) -> SourceOccurrence:
    return SourceOccurrence(
        SPACE,
        family,
        index,
        epoch_ns,
        row,
        Cursor(b"s" * 16, b"g" * 32, index, index, b"h" * 32),
    )


def _projections(row: dict[str, object] | tuple[dict[str, object], ...], family: str = "journal"):
    projector = Projector(KEY, RUN, TRACE_ID)
    rows = row if isinstance(row, tuple) else (row,)
    result = []
    for index, item in enumerate(rows, start=1):
        elapsed = item.get("elapsed_ns")
        epoch = EPOCH + elapsed if type(elapsed) is int and elapsed > 0 else EPOCH
        result.extend(projector.consume(_occurrence(item, family, index=index, epoch_ns=epoch)))
    result.extend(projector.seal(SPACE, Coverage(True, ())))
    return tuple(result)


def _span() -> SafeSpan:
    return SafeSpan(
        "stateframe.invocation",
        TRACE_ID,
        b"\x01" * 8,
        None,
        EPOCH,
        EPOCH,
        tuple(sorted((
            SafeAttr(AttrKey.RUN_ID, RUN.value.hex()),
            SafeAttr(AttrKey.SOURCE, "record"),
            SafeAttr(AttrKey.TIMING, "publication"),
            SafeAttr(AttrKey.COMPLETE, True),
        ), key=lambda attr: str(attr.key))),
        (),
        (),
        0,
        0,
        0,
        False,
    )


def _post(receiver: OTLPReceiver, payload: bytes) -> None:
    parsed = urlsplit(receiver.url)
    assert parsed.hostname is not None and parsed.port is not None
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=1)
    try:
        connection.request("POST", "/v1/traces", body=payload,
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        response.read(1024)
        assert response.status == 200
    finally:
        connection.close()


def test_no_content_reaches_sender():
    canaries = (
        "synthetic-task-marker-DoNotExport-42",
        "synthetic-body-marker-DoNotExport-43",
        "synthetic-output-marker-DoNotExport-44",
    )
    rows = (
        {
            "kind": "request_submitted",
            "sequence": 1,
            "request_id": "opaque-request-id",
            "step": 1,
            "event_id": "opaque-event-id",
        },
        {
            "kind": "response_received",
            "sequence": 2,
            "request_id": "opaque-request-id",
            "step": 1,
            "transport_attempt_id": "opaque-event-id",
            "elapsed_ns": 50_000_000,
            "task": canaries[0],
            "request_body": canaries[1],
            "response_body": canaries[2],
        },
    )
    projections = _projections(rows)
    spans = tuple(span for projection in projections for span in projection.spans)
    assert spans
    payload = pack_local(SignalBatch(Signal.TRACES, b"r" * 16, spans))

    with OTLPReceiver(("accept",)) as receiver:
        _post(receiver, payload)
        assert_capture_clean(receiver.requests, canaries)


def test_names_and_ids_are_closed_and_opaque():
    projections = _projections((
        {"kind": "request_submitted", "sequence": 1, "event_id": "private-event-marker",
         "request_id": "/private/location/marker", "step": 1,
         "model": "private-model-marker", "run_dir": "/private/location/marker"},
        {"kind": "response_received", "sequence": 2, "transport_attempt_id": "private-event-marker",
         "request_id": "/private/location/marker", "step": 1, "elapsed_ns": 50_000_000},
    ))
    spans = tuple(span for projection in projections for span in projection.spans)
    assert spans
    payload = pack_local(SignalBatch(Signal.TRACES, b"r" * 16, spans))
    assert b"private-model-marker" not in payload
    assert b"/private/location/marker" not in payload
    assert all(span.name.startswith("stateframe.") or span.name == "chat" for span in spans)
    assert all(len(span.trace_id) == 16 and len(span.span_id) == 8 for span in spans)


def test_model_name_is_keyed_at_projection_store_and_wire_boundaries(tmp_path):
    model = "model-boundary-canary"
    identity = model_id(KEY, model)
    assert identity is not None and identity != model
    assert valid_attribute(AttrKey.REQUEST_MODEL.value, identity)
    assert not valid_attribute(AttrKey.REQUEST_MODEL.value, model)

    base = _span()
    attrs = tuple(sorted((*base.attrs, SafeAttr(AttrKey.REQUEST_MODEL, identity)),
                         key=lambda attr: str(attr.key)))
    safe = dataclasses.replace(base, attrs=attrs)
    frame = pack_local(SignalBatch(Signal.TRACES, b"r" * 16, (safe,)))
    assert model.encode() not in frame
    assert identity.encode() in frame

    unsafe_attrs = tuple(sorted((*base.attrs, SafeAttr(AttrKey.REQUEST_MODEL, model)),
                                key=lambda attr: str(attr.key)))
    unsafe = dataclasses.replace(base, attrs=unsafe_attrs)
    batch = SignalBatch(Signal.TRACES, b"r" * 16, (unsafe,))
    with pytest.raises(ValueError, match="^invalid_field$"):
        pack_local(batch)
    with closing(Store(tmp_path, KEY)) as store:
        with pytest.raises(ValueError, match="^invalid_field$"):
            store.record(b"d" * 16, base.trace_id, (unsafe,), runtime="completed", check="pass",
                         complete=True, reasons=(), sampled=True)
        store.record(b"d" * 16, base.trace_id, (safe,), runtime="completed", check="pass",
                     complete=True, reasons=(), sampled=True)
        stored = store.next_batch(b"d" * 16, base.trace_id)
        assert stored is not None
        assert dict((str(attr.key), attr.value) for attr in stored.spans[0].attrs)[
            AttrKey.REQUEST_MODEL.value
        ] == identity
    assert model.encode() not in (tmp_path / "state.sqlite3").read_bytes()


def test_exception_text_never_projected():
    marker = "synthetic-exception-marker-DoNotExport-45"
    projections = _projections({
        "kind": "transport_error",
        "sequence": 1,
        "error_type": "TimeoutError",
        "exception_text": marker,
        "detail": marker,
        "message": marker,
    })
    payload = b"".join(pack_local(SignalBatch(Signal.TRACES, b"r" * 16, (span,)))
                       for projection in projections for span in projection.spans)
    assert marker.encode() not in payload


def test_unknown_enum_and_bool_as_int_rejected():
    bad_enum = dataclasses.replace(
        _span(), attrs=tuple(sorted((*_span().attrs, SafeAttr(AttrKey.OUTCOME, "unlisted")),
                                    key=lambda attr: str(attr.key)))
    )
    bad_integer = dataclasses.replace(
        _span(), attrs=tuple(sorted((*_span().attrs, SafeAttr(AttrKey.ATTEMPT, True)),
                                    key=lambda attr: str(attr.key)))
    )
    for span in (bad_enum, bad_integer):
        with pytest.raises(ValueError, match="^invalid_field$"):
            validate_batch(SignalBatch(Signal.TRACES, b"r" * 16, (span,)))


def test_fixed_resources_ignore_secret_overrides():
    span = dataclasses.replace(
        _span(), attrs=(*_span().attrs, SafeAttr("service.name", "secret-override-marker"))
    )
    with pytest.raises(ValueError, match="^invalid_field$"):
        validate_batch(SignalBatch(Signal.TRACES, b"r" * 16, (span,)))


def test_covert_counts_are_not_content_fields():
    marker = "synthetic-covert-marker-DoNotExport-46"
    projections = _projections({
        "kind": "check_receipt",
        "sequence": 1,
        "check_kind": "unit",
        "execution_id": marker,
        "output": marker,
        "summary": marker,
        "count": 7,
    })
    safe = []
    for projection in projections:
        safe.extend(pack_local(SignalBatch(Signal.TRACES, b"r" * 16, (span,)))
                    for span in projection.spans)
    assert marker.encode() not in b"".join(safe)


def test_content_digests_never_leave_reader():
    digest = "a" * 64
    projections = _projections({
        "kind": "request_submitted",
        "sequence": 1,
        "request_id": "opaque-request",
        "task_sha256": digest,
        "body_digest": digest,
        "output_digest": digest,
    })
    payload = b"".join(pack_local(SignalBatch(Signal.TRACES, b"r" * 16, (span,)))
                       for projection in projections for span in projection.spans)
    assert digest.encode() not in payload


def test_local_frame_roundtrip_and_coverage_loss_are_canonical():
    batch = SignalBatch(Signal.TRACES, b"r" * 16, (_span(),))
    frame = pack_local(batch)
    assert frame == pack_local(unpack_local(frame))

    merged = coverage_merge(Coverage(True, ()), Coverage(False, ((Reason.SOURCE_MISSING, 2),)))
    assert not merged.complete
    assert merged.reasons == ((Reason.SOURCE_MISSING, 2),)
    with pytest.raises(ValueError, match="^invalid_field$"):
        coverage_merge(Coverage(True, (("source_missing", 1),)), Coverage(True, ()))


@pytest.mark.parametrize("name", [
    "gen_ai.request.model", "gen_ai.response.model", "gen_ai.provider.name",
    "server.address", "stateframe.cost.microunits", "stateframe.repo.path",
])
def test_detail_fields_cannot_extend_the_closed_attribute_table(name):
    batch = SignalBatch(Signal.TRACES, b"r" * 16, (_span(),))
    frame = json.loads(pack_local(batch))
    frame["spans"][0]["attrs"].append({"key": name, "value": "opaque"})
    encoded = json.dumps(frame, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="^invalid_field$"):
        unpack_local(encoded)
