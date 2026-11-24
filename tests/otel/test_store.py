from contextlib import closing
from dataclasses import replace
import sqlite3

import pytest

from stateframe.otel.store import DB_FILENAME, Store
from tests.otel.support import synthetic_batch

KEY = b"k" * 32
DEST = b"d" * 16
OTHER_DEST = b"e" * 16


def _record(store, span, *, destination=DEST):
    store.record(destination, span.trace_id, (span,), runtime="completed", check="pass",
                 complete=True, reasons=(), sampled=True)


def test_delivery_is_durable_and_repeated_exports_do_not_queue_acknowledged_spans(private_root):
    span = synthetic_batch().spans[0]
    trace = span.trace_id
    with closing(Store(private_root, KEY)) as store:
        _record(store, span)
        original = store.next_batch(DEST, trace)
        assert original.spans == (span,)
        assert store.status(trace)["signals"]["traces"]["queue_items"] == 1

    with closing(Store(private_root, KEY)) as store:
        assert store.next_batch(DEST, trace) == original
        store.settle(DEST, trace, original, partial=False)
        _record(store, replace(span, end_ns=span.end_ns + 1))
        assert store.next_batch(DEST, trace) is None

    with closing(Store(private_root, KEY, read_only=True)) as store:
        assert store.known(trace)
        assert store.status(trace)["signals"]["traces"] == {
            "queue_items": 0, "acked": 1, "partial_items": 0,
        }


def test_each_destination_and_trace_has_its_own_queue(private_root):
    span = synthetic_batch().spans[0]
    other = replace(span, trace_id=b"x" * 16)
    with closing(Store(private_root, KEY)) as store:
        _record(store, span)
        _record(store, other)
        assert store.next_batch(OTHER_DEST, span.trace_id) is None
        assert store.next_batch(DEST, other.trace_id).spans == (other,)
        _record(store, span, destination=OTHER_DEST)
        assert store.next_batch(OTHER_DEST, span.trace_id).spans == (span,)


def test_partial_acknowledgement_stays_ambiguous(private_root):
    span = synthetic_batch().spans[0]
    with closing(Store(private_root, KEY)) as store:
        _record(store, span)
        batch = store.next_batch(DEST, span.trace_id)
        store.settle(DEST, span.trace_id, batch, partial=True)
        _record(store, span)
        assert store.next_batch(DEST, span.trace_id) is None
        assert store.partial(DEST, span.trace_id)


def test_modified_outbox_cannot_authorize_delivery(private_root):
    span = synthetic_batch().spans[0]
    with closing(Store(private_root, KEY)) as store:
        _record(store, span)
        with store.db:
            store.db.execute("UPDATE spans SET signature=?", (bytes(32),))
        with pytest.raises(ValueError, match="^state_unavailable$"):
            store.next_batch(DEST, span.trace_id)


def test_new_store_restricts_delivery_states(private_root):
    span = synthetic_batch().spans[0]
    with closing(Store(private_root, KEY)) as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.db.execute(
                "INSERT INTO spans VALUES (?, ?, ?, NULL, NULL, 'queuet')",
                (DEST, span.trace_id, span.span_id),
            )


def test_store_refuses_symlink_and_wrong_key(private_root):
    span = synthetic_batch().spans[0]
    with closing(Store(private_root, KEY)) as store:
        _record(store, span)
    with pytest.raises(ValueError, match="^state_unavailable$"):
        Store(private_root, b"x" * 32, read_only=True)
    (private_root / DB_FILENAME).rename(private_root / "target")
    (private_root / DB_FILENAME).symlink_to(private_root / "target")
    with pytest.raises(ValueError, match="^state_unavailable$"):
        Store(private_root, KEY)
