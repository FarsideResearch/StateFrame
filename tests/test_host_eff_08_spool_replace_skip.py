"""Identical history-row replacements do not append records or rewrite index paths.

Receipt history replaces a row on every reassignment, most of them byte-identical.
The spool must compare against the raw stored record (never the subclass's live
``_encoded`` view, which can already show mutated state) and skip only exact matches.
"""
from __future__ import annotations

import os
import pickle

import pytest

import kernel_history_spool
from kernel_history_spool import HistorySpool


def _encode(value) -> bytes:
    return pickle.dumps(value, protocol=5)


def _spool_with(tmp_path, values) -> HistorySpool:
    spool = HistorySpool(directory=tmp_path)
    for value in values:
        spool.append(value)
    return spool


@pytest.fixture
def write_calls(monkeypatch):
    calls = []
    original = HistorySpool._write

    def counting(self, payload: bytes):
        calls.append(len(payload))
        return original(self, payload)

    monkeypatch.setattr(HistorySpool, "_write", counting)
    return calls


def test_identical_replacement_appends_nothing(tmp_path, write_calls):
    values = [{"id": n, "status": "PASS", "body": "x" * 64} for n in range(40)]
    spool = _spool_with(tmp_path, values)
    root_before, end_before = spool._root, spool.spool_bytes
    write_calls.clear()

    spool._replace_encoded(17, _encode(values[17]))

    assert write_calls == []
    assert spool._root == root_before
    assert spool.spool_bytes == end_before
    assert spool[17] == values[17]
    spool.close()


def test_repeated_identical_reassignments_keep_spool_flat(tmp_path, write_calls):
    values = [{"row": n} for n in range(100)]
    spool = _spool_with(tmp_path, values)
    end_before = spool.spool_bytes
    write_calls.clear()

    for _ in range(100):
        for index, value in enumerate(values):
            spool._replace_encoded(index, _encode(value))

    assert write_calls == []
    assert spool.spool_bytes == end_before
    assert list(spool) == values
    spool.close()


def test_changed_replacement_is_written_and_reads_back(tmp_path, write_calls):
    values = [{"n": n} for n in range(10)]
    spool = _spool_with(tmp_path, values)
    root_before = spool._root
    write_calls.clear()

    spool._replace_encoded(3, _encode({"n": 3, "status": "FAIL"}))

    assert write_calls, "a changed row must be written"
    assert spool._root != root_before
    expected = list(values)
    expected[3] = {"n": 3, "status": "FAIL"}
    assert list(spool) == expected
    spool.close()


def test_skip_compares_raw_record_not_live_subclass_view(tmp_path, write_calls):
    """A subclass whose _encoded reports mutated live state must not suppress a real change."""

    class LiveView(HistorySpool):
        def _encoded(self, index):
            # Stands in for ReceiptHistory: live state already holds the new value.
            return _encode({"status": "PASS"}) if index == 0 else super()._encoded(index)

    spool = LiveView(directory=tmp_path)
    spool.append({"status": "PENDING"})
    write_calls.clear()

    spool._replace_encoded(0, _encode({"status": "PASS"}))

    assert write_calls, "stored PENDING differs from PASS, so the write must happen"
    assert spool[0] == {"status": "PASS"}
    spool.close()


def test_identical_replacement_still_refused_in_forked_child(tmp_path, monkeypatch):
    values = [{"n": 0}]
    spool = _spool_with(tmp_path, values)
    real_getpid = os.getpid
    monkeypatch.setattr(kernel_history_spool.os, "getpid", lambda: spool._writer_pid + 1)

    with pytest.raises(RuntimeError, match="forked history snapshots are read-only"):
        spool._replace_encoded(0, _encode(values[0]))

    monkeypatch.setattr(kernel_history_spool.os, "getpid", real_getpid)
    assert spool[0] == values[0]
    spool.close()


def test_mixed_replacements_match_list_model(tmp_path):
    model = [{"n": n, "v": 0} for n in range(70)]  # spans a second index level
    spool = _spool_with(tmp_path, model)
    for step in range(300):
        index = (step * 37) % len(model)
        value = model[index] if step % 3 else {"n": index, "v": step}
        model[index] = value
        spool._replace_encoded(index, _encode(value))
    assert list(spool) == model
    spool.close()
