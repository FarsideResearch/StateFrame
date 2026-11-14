"""Read admission must price the allocator's complete wire and close refused batches."""
from __future__ import annotations

from types import SimpleNamespace

import kernel_read_admission as admission
import pytest
from kernel_working_set_prepare import PreparedCandidate, prepare_candidate
from kernel_working_set_snapshot import freeze_preparation
from test_read_receipts_admission import read, runtime


def envelope(padding=4000):
    return {"model": "local", "messages": [
        {"role": "system", "content": "rules" + "x" * padding},
        {"role": "user", "content": "Task: inspect"},
        {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{}}}'},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"}], "tools": [], "max_tokens": 100}


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")


@pytest.mark.parametrize("line", ["λ" * 100, '"\\\t' * 100, "😀" * 100],
                         ids=["unicode", "json-escaping", "utf8-expansion"])
def test_admitted_page_has_feasible_real_serialized_candidate(tmp_path, monkeypatch, line):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 12_000)
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 12_000)
    rt = runtime(tmp_path)
    del rt.working_set_fixed_bytes
    rt.owner = SimpleNamespace(root=tmp_path)
    body = envelope()
    admission.note_sealed_request(rt, body["messages"], body["tools"])
    (tmp_path / "unicode.py").write_text((line + "\n") * 25)
    result, metadata = read(rt, tmp_path, "unicode.py", {
        "path": "unicode.py", "max_lines": 1000, "max_chars": 64000})
    assert result[0] == "OK"
    assert metadata["truncated"] and metadata["next_cursor"]
    snapshot = freeze_preparation(body, rt, request_id="next", byte_cap=12_000)
    prepared = prepare_candidate(snapshot)
    assert isinstance(prepared, PreparedCandidate), prepared
    assert prepared.total_bytes <= 12_000
    from kernel_working_set_live import read_admission_floor
    state = rt._read_admission_state
    pending_floor = read_admission_floor(state.snapshot, state.pages)
    assert state.pending_bytes == len(pending_floor.wire) - state.floor_bytes
    assert admission.remaining_allowance(rt, None) == 12_000 - len(pending_floor.wire)


def test_heterogeneous_batch_refuses_small_read_after_large_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    rt = runtime(tmp_path, fixed=700)
    (tmp_path / "large.py").write_text("x" * 2000 + "\n")
    (tmp_path / "small.py").write_text("a\n")
    results = [read(rt, tmp_path, name)[0][0] for name in ("large.py", "small.py")]
    assert results == [admission.REFUSED_ADMISSION] * 2
    events = [e for e in rt.journal.events if e["kind"] == "read_result"]
    assert [e["status"] for e in events] == results
    rt._frame_budget_request_count += 1
    result, metadata = read(rt, tmp_path, "small.py")
    assert result[0] == "OK" and metadata["effective_range"]


def test_protected_minimum_reduces_fresh_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 12_000)
    monkeypatch.setattr("kernel_request_budget.HARD_MAX_REQUEST_BYTES", 12_000)
    rt = runtime(tmp_path)
    rt.owner = SimpleNamespace(root=tmp_path)
    (tmp_path / "protected.py").write_text("# protected " + "p" * 5000 + "\n")
    assert read(rt, tmp_path, "protected.py")[0][0] == "OK"
    obligation, = rt.working_set_state.active_obligations()
    obligation.recovery_pin = True
    rt._frame_budget_request_count += 1
    del rt.working_set_fixed_bytes
    body = envelope(1000)
    admission.note_sealed_request(rt, body["messages"], body["tools"])
    (tmp_path / "fresh.py").write_text(("# " + "x" * 100 + "\n") * 60)
    result, metadata = read(rt, tmp_path, "fresh.py", {"path": "fresh.py", "max_lines": 1000})
    assert result[0] == "OK" and metadata["truncated"]
    prepared = prepare_candidate(freeze_preparation(body, rt, request_id="next", byte_cap=12_000))
    assert isinstance(prepared, PreparedCandidate), prepared
    assert prepared.total_bytes <= 12_000


def test_refusal_resets_on_new_origin_of_reused_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    rt = runtime(tmp_path, fixed=700)
    rt.native_batch = SimpleNamespace(origin="first", pending=[])
    (tmp_path / "large.py").write_text("x" * 2000 + "\n")
    (tmp_path / "tiny.py").write_text("a\n")
    assert read(rt, tmp_path, "large.py")[0][0] == admission.REFUSED_ADMISSION
    # Frame counters can be rewritten while draining a batch. That cannot open
    # admission again; the native origin is the batch authority.
    rt._frame_budget_request_count += 1
    assert read(rt, tmp_path, "tiny.py")[0][0] == admission.REFUSED_ADMISSION
    rt.native_batch.origin = "second"
    assert read(rt, tmp_path, "tiny.py")[0][0] == "OK"


def test_dormant_native_batch_still_resets_at_request_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "request_byte_cap", lambda: 1000)
    rt = runtime(tmp_path)
    rt.native_batch = SimpleNamespace(origin=None, pending=[])
    (tmp_path / "tiny.py").write_text("a\n")
    assert read(rt, tmp_path, "tiny.py")[0][0] == "OK"
    rt._frame_budget_request_count += 1
    assert admission.remaining_allowance(rt, None) == 1000
