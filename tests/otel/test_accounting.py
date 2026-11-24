from __future__ import annotations

import hashlib
import json

from stateframe.otel import accounting
from stateframe.otel.types import (
    AttrKey,
    Cursor,
    Family,
    Namespace,
    RunRef,
    SafeAttr,
    SourceOccurrence,
)


KEY = bytes(range(32))
RUN = RunRef(bytes.fromhex("10112233445566778899aabbccddeeff"))
TRACE = bytes.fromhex("20112233445566778899aabbccddeeff")
SPACE = Namespace(RUN, 1, None, None)
EPOCH = 1_730_000_000_000_000_000
SEED = bytes.fromhex("00112233445566778899aabbccddeeff")


def _record(
    sequence: int,
    *,
    row: dict[str, object] | None = None,
    family: Family = Family.JOURNAL,
    space: Namespace = SPACE,
    stream: bytes = b"s" * 16,
    generation: bytes = b"g" * 32,
    epoch: int | None = EPOCH,
) -> SourceOccurrence:
    content = row or {"kind": "event", "sequence": sequence}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return SourceOccurrence(
        space, family, sequence, epoch, content,
        Cursor(stream, generation, sequence, sequence, hashlib.sha256(encoded).digest()),
    )


def test_record_identity_uses_content_and_chain_position_not_reader_location():
    original = _record(7, row={"kind": "event", "value": 1})
    relocated = _record(
        7, row={"kind": "event", "value": 1}, stream=b"x" * 16, generation=b"y" * 32,
    )
    changed_content = _record(7, row={"kind": "event", "value": 2})
    changed_position = _record(8, row={"kind": "event", "value": 1})
    different_attempt = _record(
        7, row={"kind": "event", "value": 1},
        space=Namespace(RUN, 2, None, None), stream=b"z" * 16,
    )

    identity = accounting.record_identity(KEY, "span", "stateframe.event", original)
    assert identity == accounting.record_identity(KEY, "span", "stateframe.event", relocated)
    assert identity != accounting.record_identity(KEY, "span", "stateframe.event", different_attempt)
    assert identity != accounting.record_identity(KEY, "span", "stateframe.event", changed_content)
    assert identity != accounting.record_identity(KEY, "span", "stateframe.event", changed_position)


def test_pair_span_requires_one_stream_process_clock_and_bounded_forward_time():
    first = _record(1, row={"process_id": "p", "clock_id": "c"}, epoch=EPOCH)
    last = _record(2, row={"process_id": "p", "clock_id": "c"}, epoch=EPOCH + 50)
    span = accounting.paired_span(KEY, "chat", TRACE, first, last, None, ())
    assert span is not None and (span.start_ns, span.end_ns) == (EPOCH, EPOCH + 50)

    assert accounting.paired_span(
        KEY, "chat", TRACE, first,
        _record(2, row={"process_id": "other", "clock_id": "c"}, epoch=EPOCH + 50), None, (),
    ) is None
    assert accounting.paired_span(
        KEY, "chat", TRACE, first,
        _record(2, row={"process_id": "p", "clock_id": "other"}, epoch=EPOCH + 50), None, (),
    ) is None
    assert accounting.paired_span(
        KEY, "chat", TRACE, first,
        _record(2, row={"process_id": "p", "clock_id": "c"}, stream=b"x" * 16,
                epoch=EPOCH + 50), None, (),
    ) is None
    assert accounting.paired_span(
        KEY, "chat", TRACE, first,
        _record(2, row={"process_id": "p", "clock_id": "c"}, epoch=EPOCH - 1), None, (),
    ) is None
    assert accounting.paired_span(
        KEY, "chat", TRACE, first,
        _record(2, row={"process_id": "p", "clock_id": "c"}, epoch=None), None, (),
    ) is None
