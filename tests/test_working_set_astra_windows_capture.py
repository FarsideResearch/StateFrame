"""Astra 5/8: historical pins and failed durable read capture."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import kernel_read_exec as reads
import kernel_read_receipts as receipts
import kernel_working_set_state as ws
import pytest
from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_snapshot import SnapshotConstructionError, freeze_preparation
from test_read_receipts import runtime


@pytest.fixture(autouse=True)
def allocator_on(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def frame_body(**frame):
    return {"messages": [{"role": "user", "content": "Task: fix"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps(
            {"state": {"framefit5": frame}})},
        {"role": "user", "content": "suffix"}]}


@pytest.mark.parametrize("current", ["new bytes\n", None])
def test_historical_pin_never_checks_current_disk(tmp_path, current):
    old = "historical bytes\n"
    revision = "sha256:" + hashlib.sha256(old.encode()).hexdigest()
    if current is not None:
        (tmp_path / "source.txt").write_text(current)
    body = frame_body(action_ledger={"reads": [{"path": "source.txt", "revision": revision,
        "lines": {"start": 1, "end": 1}, "pinned_range": old}]})
    rt = SimpleNamespace(owner=SimpleNamespace(root=tmp_path))
    frozen = freeze_preparation(body, rt, request_id="historical", byte_cap=100_000)
    assert not frozen.sources
    prepared = prepare_candidate(frozen)
    assert isinstance(prepared, PreparedCandidate)
    frame = json.loads(json.loads(prepared.wire)["messages"][1]["content"].split("\n", 1)[1])
    row, = frame["state"]["framefit5"]["action_ledger"]["reads"]
    assert row["revision"] == revision
    assert "pinned_range" not in row


def test_live_head_still_requires_current_revision(tmp_path):
    (tmp_path / "source.txt").write_text("new bytes\n")
    body = frame_body(headcarry={"heads": [{"path": "source.txt", "revision": "sha256:" + "a" * 64,
        "line_range": {"start": 1, "end": 1}, "content": "historical bytes\n"}]})
    with pytest.raises(SnapshotConstructionError, match="SOURCE_SNAPSHOT_UNSTABLE"):
        freeze_preparation(body, SimpleNamespace(owner=SimpleNamespace(root=tmp_path)),
                           request_id="head", byte_cap=100_000)


@pytest.mark.parametrize("previously_exposed", [False, True])
def test_new_revision_capture_supersedes_old_debt_and_historical_pin_authority(tmp_path, previously_exposed):
    rt = runtime(tmp_path)
    rt.owner = SimpleNamespace(root=tmp_path)
    path = tmp_path / "source.txt"
    path.write_text("old\n")
    first = reads.execute_read(tmp_path, path, path.name, path.name, 0,
                              args={"path": path.name}, measurement=rt, step=1, delivery={})
    assert first[0] == "OK"
    old, = rt.read_capture_ledger.all()
    if previously_exposed:
        prepared = prepare_candidate(freeze_preparation(frame_body(), rt,
            request_id="old", byte_cap=100_000))
        assert isinstance(prepared, PreparedCandidate)
        capture = SimpleNamespace(request_id="old", _working_set_certificate=prepared.rendered.certificate)
        ws.observe_submission(rt, capture, prepared.wire)
        ws.observe_response(rt, capture, disposition="accepted", response_id="accepted-old")
    path.write_text("new\n")
    second = reads.execute_read(tmp_path, path, path.name, path.name, 0,
                               args={"path": path.name}, measurement=rt, step=2, delivery={})
    assert second[0] == "OK"
    current = rt.read_capture_ledger.all()[-1]
    body = frame_body(action_ledger={"reads": [{"path": path.name,
        "revision": old.identity.revision, "lines": {"start": 1, "end": 1}, "pinned_range": "old\n"}]},
        read_facts=[{"path": path.name, "current_revision": current.identity.revision}])
    frozen = freeze_preparation(body, rt, request_id="current", byte_cap=100_000)
    # A new capture supersedes old debt even before exposure; both captures
    # remain in durable history, without turning the historical pin into source.
    assert {s.payload for s in frozen.sources} == {"new\n"}
    assert len(rt.read_capture_ledger.all()) == 2
    assert all(s.state == "DUE" for s in frozen.sources)
    assert dict(frozen.current_revisions)[path.name] == current.identity.revision
    prepared = prepare_candidate(frozen)
    assert isinstance(prepared, PreparedCandidate)
    assert all(mode == "FULL" for _, mode in prepared.rendered.certificate.assignment)


@pytest.mark.parametrize("cache", ["disk", "file_cache", "page_cache"])
@pytest.mark.parametrize("switch", ["on", "off"])
def test_capture_io_failure_cannot_return_an_unaccounted_ok_page(tmp_path, monkeypatch, cache, switch):
    payload = "source payload must not escape failed capture\n"
    (tmp_path / "source.txt").write_text(payload)
    rt = runtime(tmp_path)
    args = {"path": "source.txt"}
    if cache != "disk":
        # Populate the real execution cache without creating capture blobs.
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
        result = reads.execute_read(tmp_path, tmp_path / "source.txt", "source.txt", "source.txt", 0,
                                    args=args, measurement=rt, step=1, delivery={})
        assert result[0] == "OK"
        if cache == "page_cache":
            next(iter(rt._read_contract_source_cache.values())).content = None
    # A real filesystem conflict breaks the blob writer; capture itself is real.
    (rt.journal.path.parent / "blobs").write_text("not a directory")
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", switch)
    delivery = {}
    scope = receipts.begin_read_attempt(rt, args)
    result, observation, raw = reads.execute_read(
        tmp_path, tmp_path / "source.txt", "source.txt", "source.txt", 0,
        args=args, measurement=rt, step=2, delivery=delivery)
    if scope is not None:
        scope.finish(result)
    if switch == "off":
        assert result == "OK" and payload in observation
        assert not rt.journal.events
        return
    assert result == "ERROR"
    assert payload not in observation and raw is None
    assert "Retry" in observation
    assert delivery["result_class"] == "ERROR"
    assert delivery["failure_class"] == "capture_failed"
    assert delivery["effective_range"] is None
    assert delivery["requested_bytes_delivered"] == 0
    assert delivery["delivery_status"] == "WITHHELD"
    assert not getattr(rt, "working_set_state", None)
    assert not getattr(rt, "read_capture_ledger", ())
    failure, = [row for row in rt.journal.events if row["kind"] == "read_capture_failed"]
    assert failure["attempt_id"] == scope.attempt_id
    audit, = [row for row in rt.journal.events if row["kind"] == "read_result"]
    assert audit["status"] == "ERROR" and audit["reason"] == "capture_failed"
    assert "identity" not in audit
    # Once storage is available, the requested retry creates ordinary due debt.
    (rt.journal.path.parent / "blobs").rename(rt.journal.path.parent / "blob-failure-marker")
    retry = receipts.begin_read_attempt(rt, args)
    result, observation, _ = reads.execute_read(
        tmp_path, tmp_path / "source.txt", "source.txt", "source.txt", 0,
        args=args, measurement=rt, step=3, delivery={})
    retry.finish(result)
    assert result == "OK" and payload in observation
    obligation, = ws.state_for(rt).active_obligations()
    assert obligation.state == "DUE"
    assert rt.journal.events[-1]["status"] == "OK"
