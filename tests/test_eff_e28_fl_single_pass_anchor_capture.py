import hashlib
import io
import json
from contextlib import contextmanager

import pytest

import kernel_fl_checkpoint_capture as capture
import kernel_measurement as measurement


def digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def submission(sequence, payload):
    return dict(kind="request_submitted", request_id=f"r{sequence}", sequence=sequence,
                event_id=f"e{sequence}", payload_utf8=payload,
                payload_hash=digest(payload.encode()))


def line(row):
    return json.dumps(row).encode() + b"\n"


def test_latest_blob_backed_submission_beats_older_inline_anchor(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    old = submission(1, "inline")
    new = submission(2, "large" * 2000)
    blob = measurement.externalize_journal_fields({"payload_utf8": new["payload_utf8"]}, path)
    stored = line({**new, **blob})
    path.write_bytes(line(old) + b"\n" + stored)
    monkeypatch.setenv("STATEFRAME_FL_SINGLE_PASS_ANCHOR", "off")
    assert capture._journal_capture(path, 20000) == (old, digest(line(old)))
    monkeypatch.setenv("STATEFRAME_FL_SINGLE_PASS_ANCHOR", "on")
    assert capture._journal_capture(path, 20000) == (new, digest(line(old) + stored))


def test_single_pass_anchor_handles_retry_refs_blank_lines_and_corrupt_tail(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    base = submission(1, "wire")
    retry = dict(kind="request_submitted", request_id="r1", event_id="e2", sequence=2,
                 payload_ref="e1", payload_hash=base["payload_hash"],
                 payload_delta=dict(base_attempt=1, identical_wire=True, changed_keys=[]))
    path.write_bytes(line(base) + b"\n" + line(retry))
    monkeypatch.setenv("STATEFRAME_FL_SINGLE_PASS_ANCHOR", "on")
    assert capture._journal_capture(path, 1000) == (
        {**retry, "payload_utf8": "wire"}, digest(line(base) + line(retry)))
    for bad_tail in (b"{", line(dict(retry, sequence=3, payload_ref="missing"))):
        path.write_bytes(line(base) + line(retry) + bad_tail)
        with pytest.raises(capture.CaptureBoundaryError):
            capture._journal_capture(path, 1000)


def test_same_inline_anchor_with_half_the_journal_bytes_scanned(monkeypatch, tmp_path):
    rows = [submission(i, f"payload {i}") for i in range(1, 21)]
    journal = b"".join(map(line, rows))
    scanned = []

    @contextmanager
    def counting_open(*args, **kwargs):
        stream = io.BytesIO(journal)
        yield stream
        scanned.append(stream.tell())

    monkeypatch.setattr(measurement, "open_journal", counting_open)
    for mode, scans in (("off", 2), ("on", 1)):
        scanned.clear()
        monkeypatch.setenv("STATEFRAME_FL_SINGLE_PASS_ANCHOR", mode)
        assert capture._journal_capture(tmp_path / "events.jsonl", 8192) == (rows[-1], digest(journal))
        assert scanned == [len(journal)] * scans
