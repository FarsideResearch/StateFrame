"""FU-070-ALLOC: the allocator emits a deterministic [Current source map] for its own windows."""
from __future__ import annotations

import hashlib
import json

from kernel_read_receipts import DecodedSpan, ReceiptIdentity, line_digests_with_numbers
from kernel_source_view import SOURCE_MAP_HEADER, parse_source_map, source_view
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_proof import verify_certificate
from kernel_working_set_render import render_assignment
from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot
from kernel_working_set_windows import render_frozen_span

import pytest

import kernel_unit_policy

PATH = "owned.py"
LINES = [f"VALUE_{n:04d} = {n}\n" for n in range(1, 31)]
SOURCE = "".join(LINES)
REVISION = "sha256:" + hashlib.sha256(SOURCE.encode()).hexdigest()


@pytest.fixture(autouse=True)
def _subframe_worker(monkeypatch):
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: object())


def window_slot(first: int, last: int, *, revision: str = REVISION, ordinal: int = 1) -> SourceSlot:
    payload = "".join(LINES[first - 1:last])
    page, start = render_frozen_span(
        {"path": PATH, "revision": revision, "total_lines": len(LINES)},
        {"start_line": first, "end_line": last}, payload)
    identity = ReceiptIdentity(PATH, revision, first, last, None, None,
                               hashlib.sha256(payload.encode()).hexdigest(),
                               DecodedSpan(0, len(LINES[last - 1]), "model-source-window/1"))
    digests, numbers = line_digests_with_numbers(payload, first)
    key = f"closure:{first}-{last}"
    return SourceSlot(key, (key,), identity, payload, page, digests, numbers,
                      json.dumps({"path": PATH, "start_line": first, "end_line": last}).encode(),
                      ordinal, 0, "CLOSURE", payload_start=len(page[:start].encode()))


def native_slot() -> SourceSlot:
    payload = "".join(LINES[:3])
    page = (f"[read page] path={PATH} revision={REVISION} lines=1-3/30\n"
            f"0 more; complete=true\n{payload}")
    identity = ReceiptIdentity(PATH, REVISION, 1, 3, 0, len(payload.encode()),
                               hashlib.sha256(payload.encode()).hexdigest(), None)
    digests, numbers = line_digests_with_numbers(payload, 1)
    return SourceSlot("native", ("native",), identity, payload, page, digests, numbers,
                      b'{"path":"owned.py","start_line":1,"end_line":3}', 1, 1, "DUE")


def snapshot(*slots: SourceSlot, current: tuple[tuple[str, str], ...] = ((PATH, REVISION),)):
    body = {"model": "local", "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "opaque newest suffix"}], "max_tokens": 100}
    return PreparationSnapshot(json.dumps(body).encode(), tuple(slots), "request", 1, 1_000_000,
                               current_revisions=current)


def messages_of(candidate) -> list[dict[str, object]]:
    return json.loads(candidate.wire)["messages"]


def maps_of(candidate) -> list[dict[str, tuple]]:
    return [parsed for message in messages_of(candidate)
            if (parsed := parse_source_map(message.get("content"))) is not None]


def test_three_windows_emit_a_map_the_verifier_accepts() -> None:
    slots = (window_slot(1, 10, ordinal=1), window_slot(11, 20, ordinal=2), window_slot(21, 30, ordinal=3))
    candidate = render_assignment(snapshot(*slots), {s.key: FULL for s in slots})
    assert maps_of(candidate) == [{PATH: (REVISION, ((1, 30),))}]
    view = source_view(messages_of(candidate), path=PATH, source=SOURCE.encode())
    assert view.complete and view.total_lines == 30
    assert verify_certificate(candidate.wire, candidate.certificate)
    assert sum(dict(candidate.section_breakdown).values()) == len(candidate.wire)


def test_map_lists_only_delivered_spans_so_a_removed_page_stays_incomplete() -> None:
    slots = (window_slot(1, 10, ordinal=1), window_slot(11, 20, ordinal=2), window_slot(21, 30, ordinal=3))
    modes = {slots[0].key: FULL, slots[1].key: DEFERRED, slots[2].key: FULL}
    candidate = render_assignment(snapshot(*slots), modes)
    assert maps_of(candidate) == [{PATH: (REVISION, ((1, 10), (21, 30)))}]
    assert not source_view(messages_of(candidate), path=PATH, source=SOURCE.encode()).complete


def test_map_is_deterministic_across_slot_order() -> None:
    slots = (window_slot(1, 10, ordinal=1), window_slot(11, 20, ordinal=2))
    first = render_assignment(snapshot(*slots), {s.key: FULL for s in slots})
    second = render_assignment(snapshot(*reversed(slots)), {s.key: FULL for s in slots})
    assert first.wire == second.wire


def test_nothing_is_emitted_without_a_verifiable_current_window() -> None:
    # Deferred only, a native read page, and a stale unobserved-current carry no map.
    only_deferred = window_slot(1, 10)
    assert maps_of(render_assignment(snapshot(only_deferred), {only_deferred.key: DEFERRED})) == []
    native = native_slot()
    assert maps_of(render_assignment(snapshot(native), {native.key: FULL})) == []
    stale = window_slot(1, 10)
    candidate = render_assignment(snapshot(stale, current=()), {stale.key: FULL})
    assert maps_of(candidate) == []
    assert SOURCE_MAP_HEADER not in candidate.wire.decode()


def test_no_map_outside_a_subframe_worker(monkeypatch) -> None:
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: None)
    slot = window_slot(1, 10)
    assert maps_of(render_assignment(snapshot(slot), {slot.key: FULL})) == []
