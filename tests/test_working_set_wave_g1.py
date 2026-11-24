"""Wave G1 lifecycle: edits retire exposed obsolete revisions, overflow clears,
reserve retirement keeps protected residents and residents without a route."""
from __future__ import annotations

from dataclasses import replace
import json
import hashlib
from types import SimpleNamespace

import pytest

import kernel_working_set_state as ws
from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_prepare import prepare_candidate, PreparedCandidate
from kernel_working_set_render import _slot_modes, render_assignment
from kernel_working_set_proof import verify_certificate
from test_working_set_render import snapshot, source
from test_working_set_state import accept, runtime, submit
from test_working_set_state import capture as read_capture
from _stateframe_home import isolated_stateframe_home

REV1, REV2 = "sha256:" + "1" * 64, "sha256:" + "2" * 64


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")


def resident_chunks(count=3, *, previous_mode="F"):
    return [replace(source(f"a{i}", f"line {i}\n" * 120, first=1 + 120 * i, revision=REV1,
                           state="RESIDENT", ordinal=i + 1), previous_mode=previous_mode)
            for i in range(count)]


def test_edit_without_reread_retires_exposed_obsolete_residents():
    """A1 / S1: read -> accepted -> edit -> prepare, no reread in between."""
    chunks = resident_chunks()
    same = snapshot(*chunks, current_revisions=(("a.py", REV1),))
    edited = snapshot(*chunks, current_revisions=(("a.py", REV2),), protected_paths=("a.py",))
    assert [_slot_modes(same, s) for s in chunks] == [(FULL, DEFERRED)] * 3
    assert [_slot_modes(edited, s) for s in chunks] == [(DEFERRED,)] * 3
    floor = len(render_assignment(same, {s.key: FULL for s in chunks}).wire)
    result = prepare_candidate(replace(edited, byte_cap=floor - 1))
    assert isinstance(result, PreparedCandidate)
    assert len(result.wire) < floor
    wire = result.wire.decode()
    contents = [message["content"] for message in json.loads(result.wire)["messages"]]
    assert all(s.payload not in content for s in chunks for content in contents)
    assert "stale since your edit" in wire and "re-read current source" in wire
    assert 'path=\\"a.py\\"' in wire and "lines=1-120" in wire
    assert "read(" not in wire  # obsolete coordinates are not a proven recovery route
    assert "expected_revision" not in wire
    assert verify_certificate(result.wire, result.rendered.certificate)
    assert all(w.recovery_arguments == b"{}" for w in result.rendered.certificate.witnesses)
    # The stale notice needs no recovery route: an owner phase without read still retires it.
    narrowed = replace(edited, recovery_owner_allowed=False, recovery_allowed=False)
    assert [_slot_modes(narrowed, s) for s in chunks] == [(DEFERRED,)] * 3
    result = prepare_candidate(replace(narrowed, byte_cap=floor - 1))
    assert isinstance(result, PreparedCandidate)
    assert verify_certificate(result.wire, result.rendered.certificate)


@pytest.mark.parametrize("state,pin", [("DUE", False), ("DUE", True), ("RESIDENT", True)])
def test_unexposed_fresh_debt_at_an_edited_revision_stays_full(state, pin):
    """A1: genuinely unexposed DUE debt is preserved even after an edit."""
    fresh = replace(source("a", "line\n" * 10, revision=REV1, state=state), recovery_pin=pin)
    edited = snapshot(fresh, current_revisions=(("a.py", REV2),))
    assert _slot_modes(edited, fresh) == (FULL,)


def test_accepted_omission_discharges_the_obsolete_resident():
    """Lifecycle: the stale resident leaves the index on the accepted O carrier."""
    rt = runtime()
    ws.record_capture(rt, read_capture(1, revision="1"))
    submit(rt, "expose")
    accept(rt, "expose")
    state = ws.state_for(rt)
    old, = state.active_obligations()
    assert old.state == "RESIDENT" and old.previous_mode == "F"
    submit(rt, "retire", {old.obligation_id: "O"})
    accept(rt, "retire")
    assert not state.active_obligations()


def test_superseded_unexposed_debt_never_forces_a_stale_page():
    """S7: a second capture of the same path at a new revision retires the old DUE debt."""
    rt = runtime()
    ws.record_capture(rt, read_capture(1, revision="1"))
    ws.record_capture(rt, read_capture(2, revision="2"))
    state = ws.state_for(rt)
    current, = state.active_obligations()
    assert current.identity.revision == "sha256:" + "2" * 64 and current.state == "DUE"
    assert current.superseded_by is None
    assert ws.reconstruct(rt.journal.events).checkpoint() == state.checkpoint()


def test_overflow_never_drops_a_capture_and_clears_after_retirement():
    """S5: overflow is live overshoot, retired by the next accepted O carriers."""
    rt = runtime()
    total = ws.MAX_ACTIVE_OBLIGATIONS + 3
    for n in range(total):
        ws.record_capture(rt, read_capture(n, path=f"file{n}.py"))
    state = ws.state_for(rt)
    active = state.active_obligations()
    assert len(active) == total and state.overflow_count == 3 and not state.shadow_complete
    assert sum(o.state == "RETIRING" for o in active) == 0  # nothing exposed yet
    submit(rt, "expose")
    accept(rt, "expose")
    assert len(state.active_obligations()) == total
    retiring = {o.obligation_id for o in state.active_obligations() if o.state == "RETIRING"}
    assert len(retiring) == total - ws.MAX_RETAINED_RESIDENTS
    assert ws.WorkingSetState.from_checkpoint(state.checkpoint()).checkpoint() == state.checkpoint()
    # Restore requires the exact live overshoot, including no legacy missing captures.
    saved = state.checkpoint()
    saved["overflow_count"] = 2
    with pytest.raises(ValueError):
        ws.WorkingSetState.from_checkpoint(saved)
    saved["overflow_count"] = 4
    with pytest.raises(ValueError):
        ws.WorkingSetState.from_checkpoint(saved)
    submit(rt, "retire", {o.obligation_id: "O" if o.obligation_id in retiring else "F"
                          for o in state.active_obligations()})
    accept(rt, "retire")
    assert len(state.active_obligations()) == ws.MAX_RETAINED_RESIDENTS
    assert state.overflow_count == 0 and state.shadow_complete
    assert ws.reconstruct(rt.journal.events).checkpoint() == state.checkpoint()


def test_legacy_overflow_checkpoint_requires_journal_replay():
    from kernel_continuation_restart import checkpoint_runtime_fields_match, runtime_fields
    rt = runtime()
    ws.record_capture(rt, read_capture())
    saved = ws.state_for(rt).checkpoint()
    saved["overflow_count"] = 1  # old reducer dropped captures at the live limit
    with pytest.raises(ValueError, match="active working-set index"):
        ws.WorkingSetState.from_checkpoint(saved)
    saved_runtime = dict.fromkeys(runtime_fields(rt))
    saved_runtime["working_set_state"] = saved
    assert not checkpoint_runtime_fields_match(rt, saved_runtime)


def test_overshoot_index_is_still_frozen():
    """S5: an overshoot index is complete, so preparation is never refused for it."""
    from kernel_working_set_snapshot import freeze_preparation
    state = ws.WorkingSetState()
    state.overflow_count = 1
    frozen = freeze_preparation({"messages": []}, SimpleNamespace(working_set_state=state),
                                request_id="request", byte_cap=82026)
    assert frozen.sources == ()


def test_reserve_retirement_needs_a_valid_retirement_representation():
    """A7 / S10: RETIRING keeps protected residents and residents without a route."""
    retiring = replace(source("r", "line\n" * 10, revision=REV1, state="RETIRING", ordinal=1),
                       previous_mode="F")
    stale = snapshot(retiring, current_revisions=(("a.py", REV2),), protected_paths=("a.py",))
    assert _slot_modes(stale, retiring) == (DEFERRED,)
    assert "stale since your edit" in render_assignment(stale, {"r": DEFERRED}).wire.decode()
    protected = snapshot(retiring, current_revisions=(("a.py", REV1),), protected_paths=("a.py",))
    assert _slot_modes(protected, retiring) == (FULL,)
    assert json.dumps(retiring.payload)[1:-1] in render_assignment(protected, {"r": FULL}).wire.decode()
    # G2 adds an explicit UNAVAILABLE notice for missing read permission. A
    # malformed executable schema still has no valid retirement representation.
    no_route = snapshot(retiring, current_revisions=(("a.py", REV1),),
                        recovery_schema=b'{"type":"object","properties":{},"required":["missing"]}')
    assert _slot_modes(no_route, retiring) == (FULL,)
    routed = snapshot(retiring, current_revisions=(("a.py", REV1),))
    assert _slot_modes(routed, retiring) == (DEFERRED,)


def test_kept_retiring_resident_returns_and_requeues_the_next_oldest():
    rt = runtime()
    for n in range(ws.MAX_RETAINED_RESIDENTS + 1):
        ws.record_capture(rt, read_capture(n, path=f"file{n}.py"))
    submit(rt, "expose")
    accept(rt, "expose")
    state = ws.state_for(rt)
    first, = [o for o in state.active_obligations() if o.state == "RETIRING"]
    for disposition in ("failed", "discarded"):
        use_ordinal = state.use_ordinal
        submit(rt, disposition, {o.obligation_id: "F" for o in state.active_obligations()})
        accept(rt, disposition, disposition=disposition)
        assert state.use_ordinal == use_ordinal
        assert first.state == "RETIRING"
    # The allocator kept it FULL (protected / no retirement representation).
    submit(rt, "keep", {o.obligation_id: "F" for o in state.active_obligations()})
    accept(rt, "keep")
    kept = next(o for o in state.active_obligations() if o.obligation_id == first.obligation_id)
    assert kept.state == "RESIDENT"
    queued = [o for o in state.active_obligations() if o.state == "RETIRING"]
    assert len(queued) == 1 and queued[0].obligation_id != first.obligation_id
    with pytest.raises(ValueError):
        submit(rt, "bad", {o.obligation_id: "D" if o.state == "RETIRING" else "F"
                           for o in state.active_obligations()})
    # The newly queued resident can retire on the next accepted request.
    submit(rt, "progress", {o.obligation_id: "O" if o.state == "RETIRING" else "F"
                            for o in state.active_obligations()})
    accept(rt, "progress")
    assert len(state.active_obligations()) == ws.MAX_RETAINED_RESIDENTS
    assert first in state.active_obligations()


def test_real_capture_edit_prepare_and_adoption_without_reread(tmp_path):
    from kernel_working_set_snapshot import freeze_preparation
    from test_read_receipts import read, runtime as disk_runtime

    original = "original source line\n" * 300
    path = tmp_path / "source.txt"
    path.write_text(original)
    rt = disk_runtime(tmp_path)
    read(rt, tmp_path)
    captured, = rt.read_capture_ledger.all()

    def frozen(revision, request, cap):
        frame = {"read_facts": [{"path": "source.txt", "current_revision": revision}],
                 "next_site": {"path": "source.txt"},
                 "owner_protocol": {"admitted_actions": []}}
        body = {"messages": [{"role": "system", "content": "rules"},
            {"role": "user", "content": "Task: edit source"},
            {"role": "user", "content": "[StateFrame]\n" + json.dumps(
                {"state": {"framefit5": frame}})}]}
        return freeze_preparation(body, rt, request_id=request, byte_cap=cap)

    before = prepare_candidate(frozen(captured.identity.revision, "expose", 20_000))
    assert isinstance(before, PreparedCandidate)
    sent = SimpleNamespace(request_id="expose", _working_set_certificate=before.rendered.certificate)
    ws.observe_submission(rt, sent, before.wire)
    ws.observe_response(rt, sent, disposition="accepted", response_id="exposed")
    path.write_text("edited source line\n" * 300)
    revision = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    edited = frozen(revision, "retire", 2_000)
    assert edited.protected_paths == ("source.txt",)
    after = prepare_candidate(edited)
    assert isinstance(after, PreparedCandidate)
    assert len(rt.read_capture_ledger.all()) == 1  # no reread pays or supersedes this capture
    assert len(after.wire) < 2_000 < len(before.wire)
    contents = [message["content"] for message in json.loads(after.wire)["messages"]]
    assert all(slot.payload not in content for slot in before.rendered.certificate.snapshot.sources
               for content in contents)
    assert "stale since your edit" in after.wire.decode()
    assert verify_certificate(after.wire, after.rendered.certificate)
    sent = SimpleNamespace(request_id="retire", _working_set_certificate=after.rendered.certificate)
    ws.observe_submission(rt, sent, after.wire)
    ws.observe_response(rt, sent, disposition="failed", response_id="failed")
    assert len(ws.state_for(rt).active_obligations()) == 1
    ws.observe_submission(rt, sent, after.wire)
    ws.observe_response(rt, sent, disposition="accepted", response_id="retired")
    assert not ws.state_for(rt).active_obligations()
    assert ws.reconstruct(rt.journal.events).checkpoint() == ws.state_for(rt).checkpoint()


def test_real_overflow_freezes_every_capture_and_retires_on_acceptance(tmp_path, monkeypatch):
    from kernel_working_set_snapshot import freeze_preparation
    from test_read_receipts import runtime as disk_runtime
    from test_working_set_reserve_replenish import add_capture, candidate

    # Exercise real blobs and certificates with a small reserve; the full-size
    # reducer regression above covers the shipped 256-identity boundary.
    monkeypatch.setattr(ws, "MAX_ACTIVE_OBLIGATIONS", 4)
    monkeypatch.setattr(ws, "MAX_RETAINED_RESIDENTS", 2)
    rt = disk_runtime(tmp_path)
    for number in range(5):
        add_capture(rt, number)
    state = ws.state_for(rt)
    snap = freeze_preparation({"messages": []}, rt, request_id="overshoot", byte_cap=20_000)
    assert len(snap.sources) == 5 and state.overflow_count == 1
    exposed = render_assignment(snap, {slot.key: FULL for slot in snap.sources})
    sent = SimpleNamespace(request_id="overshoot", _working_set_certificate=exposed.certificate)
    ws.observe_submission(rt, sent, exposed.wire)
    ws.observe_response(rt, sent, disposition="accepted", response_id="exposed")
    sent, wire = candidate(rt, "retire")
    assert verify_certificate(wire, sent._working_set_certificate)
    ws.observe_submission(rt, sent, wire)
    ws.observe_response(rt, sent, disposition="discarded", response_id="discarded")
    assert len(state.active_obligations()) == 5 and state.overflow_count == 1
    ws.observe_submission(rt, sent, wire)
    ws.observe_response(rt, sent, disposition="accepted", response_id="retired")
    assert len(state.active_obligations()) == 2 and state.overflow_count == 0
    assert len(rt.read_capture_ledger.all()) == 5
    assert ws.reconstruct(rt.journal.events).checkpoint() == state.checkpoint()


def test_off_provider_bytes_match_group_base(tmp_path, tmp_path_factory):
    import difflib
    import os
    from pathlib import Path
    import subprocess
    import sys
    from tools.pinned_tree import pinned_tree

    baseline = pinned_tree("56af0dc6d6c7b632808070e137b9d95f66db882e", tmp_path_factory)
    current = Path(__file__).resolve().parents[1]
    driver = '''
import json, sys
from pathlib import Path
import pytest
from test_request_determinism import _run_scripted_example, EXAMPLES
from stateframe.scripted_endpoint import Script
with pytest.MonkeyPatch.context() as patch:
    original = Script.__init__
    def record(self, document):
        original(self, {**document, "request_log": sys.argv[3]})
    patch.setattr(Script, "__init__", record)
    patch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    from stateframe import runner
    runner.test_interpreter.choose = lambda *args, **kwargs: (sys.executable, "parity test interpreter")
    patch.setenv("SF_LAB_DETFIX3", "on")
    hashes, requests, _ = _run_scripted_example(
        patch, "discount_bug", EXAMPLES["discount_bug"], Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(hashes))
'''
    hashes, requests = [], []
    for label, tree in (("base", baseline), ("current", current)):
        destination = tmp_path / f"{label}.json"
        request_log = tmp_path / f"{label}-requests.jsonl"
        environment = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}
        environment["STATEFRAME_HOME"] = isolated_stateframe_home()
        environment["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
            ("src", "src/stateframe/engine", "tests"))
        completed = subprocess.run([sys.executable, "-c", driver, str(tmp_path / label),
            str(destination), str(request_log)], cwd=tree, env=environment,
            capture_output=True, text=True, timeout=900, check=False)
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
        hashes.append(json.loads(destination.read_text()))
        requests.append([json.loads(row) for row in request_log.read_text().splitlines()])
    difference = "\n".join(difflib.unified_diff(
        json.dumps(requests[0], indent=2).splitlines(),
        json.dumps(requests[1], indent=2).splitlines(), fromfile="base", tofile="current"))
    assert hashes[0] and hashes[0] == hashes[1], difference[:12_000]
