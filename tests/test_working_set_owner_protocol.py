"""S2: narrowing the read menu must not make retained source mandatory."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from kernel_working_set_allocator import DEFERRED, FULL
from kernel_working_set_prepare import Blocked, PreparedCandidate, prepare_candidate
from kernel_working_set_proof import CertificateError, verify_certificate
from kernel_working_set_render import domain, render_assignment
from kernel_working_set_stop import ReasonCode
from kernel_working_set_snapshot import freeze_preparation
from test_working_set_render import snapshot, source


@pytest.mark.parametrize("surface", ["recovery_owner_allowed", "recovery_allowed"])
def test_no_read_surface_does_not_make_resident_chunks_a_full_floor(surface):
    # Adapted from scout r6_stale_mandatory.py's owner-protocol tail.
    chunks = tuple(replace(source(f"a{i}", f"line {i}\n" * 120,
        first=1 + 120 * i, state="RESIDENT", ordinal=i + 1), previous_mode="F")
        for i in range(3))
    snap = snapshot(*chunks, current_revisions=(("a.py", chunks[0].identity.revision),),
                    **{surface: False})
    full = render_assignment(snap, {s.key: FULL for s in chunks})
    result = prepare_candidate(replace(snap, byte_cap=len(full.wire) - 1))
    assert isinstance(result, PreparedCandidate), result
    assert "DEFERRED" in dict(result.rendered.certificate.assignment).values()
    assert verify_certificate(result.wire, result.rendered.certificate)
    assert all(domain(snap, s) == (FULL, DEFERRED) for s in chunks)


@pytest.mark.parametrize("surface", ["recovery_owner_allowed", "recovery_allowed"])
@pytest.mark.parametrize("state,previous,plan", [
    ("RESIDENT", "F", False), ("DEGRADED", "D", False),
    ("RETIRING", "F", False), ("DISCHARGED", "O", False),
    ("CLOSURE", None, False), ("PLAN", None, True),
])
def test_nonrecoverable_deferral_is_truthful_and_certified(surface, state, previous, plan):
    slot = replace(source(state=state), previous_mode=previous, plan_only=plan)
    snap = snapshot(slot, **{surface: False})
    rendered = render_assignment(snap, {slot.key: DEFERRED})
    contents = "\n".join(m["content"] for m in json.loads(rendered.wire)["messages"])
    assert "recovery unavailable" in contents.lower()
    assert "read(" not in contents
    assert slot.payload not in contents
    assert verify_certificate(rendered.wire, rendered.certificate)
    for witness in rendered.certificate.witnesses:
        assert json.loads(witness.recovery_arguments)["status"] == "UNAVAILABLE"
        assert witness.payload_start is None and witness.payload_end is None
    if not plan:
        assert "a.py" in contents and slot.identity.revision in contents
        assert f"lines={slot.identity.first_line}-{slot.identity.last_line}" in contents
        forged = rendered.wire.replace(b"recovery unavailable", b"recovery available!!")
        with pytest.raises(CertificateError):
            verify_certificate(forged, rendered.certificate)


@pytest.mark.parametrize("requirement", ["DUE", "pin", "protected"])
def test_missing_read_preserves_genuine_full_obligations(requirement):
    slot = source(state="DUE" if requirement == "DUE" else "RESIDENT")
    slot = replace(slot, recovery_pin=requirement == "pin")
    snap = snapshot(slot, recovery_owner_allowed=False,
                    protected_paths=("a.py",) if requirement == "protected" else ())
    assert domain(snap, slot) == (FULL,)
    full = render_assignment(snap, {slot.key: FULL})
    result = prepare_candidate(replace(snap, byte_cap=len(full.wire) - 1))
    assert isinstance(result, Blocked)
    assert result.reason is ReasonCode.FLOOR_INFEASIBLE
    result.proof.validate()


def test_owner_transition_retires_real_capture_and_restores_callable_notices(
        tmp_path, monkeypatch):
    import kernel_working_set_state as ws
    from test_working_set_render_boundaries import envelope, retained

    rt, _, _, _ = retained(tmp_path, monkeypatch, "retained source\n" * 150, 1, 150)
    body = envelope()
    body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"owner_protocol": {"admitted_actions": ["edit"]}}}})
    body["tools"] = [{"type": "function", "function": {"name": "read",
        "parameters": {"type": "object"}}}]
    snap = freeze_preparation(body, rt, request_id="narrowed", byte_cap=100_000)
    slot, = snap.slots
    assert snap.recovery_allowed and not snap.recovery_owner_allowed
    full = render_assignment(snap, {slot.key: FULL})
    result = prepare_candidate(replace(snap, byte_cap=len(full.wire) - 1))
    assert isinstance(result, PreparedCandidate), result
    assert dict(result.rendered.certificate.assignment) == {slot.key: "DEFERRED"}
    capture = SimpleNamespace(request_id="narrowed",
                              _working_set_certificate=result.rendered.certificate)
    ws.observe_submission(rt, capture, result.wire)
    ws.observe_response(rt, capture, disposition="accepted", response_id="accepted-narrowed")
    # DEFERRED is the product's O retirement mode, paid only by acceptance.
    assert rt.working_set_state.active_obligations() == ()

    body["messages"][2]["content"] = "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"owner_protocol": {"admitted_actions": ["read", "edit"]}}}})
    reopened = freeze_preparation(body, rt, request_id="reopened", byte_cap=100_000)
    slot, = reopened.slots
    assert not slot.capture_ids  # A fresh closure proposal, not retained debt.
    from kernel_working_set_recovery import validate_recovery
    assert validate_recovery(reopened, slot)["path"] == "a.py"
    rendered = render_assignment(reopened, {slot.key: DEFERRED})
    assert b"UNAVAILABLE" not in rendered.wire
    assert verify_certificate(rendered.wire, rendered.certificate)


@pytest.mark.parametrize("actions", [[], ["edit"], ["done"]])
def test_embedded_window_with_narrowed_owner_can_be_deferred(tmp_path, actions):
    from kernel_stateframe_headcarry import _module_head, _panel

    raw = b"def important():\n    return 42\n"
    (tmp_path / "a.py").write_bytes(raw)
    head = _module_head(tmp_path, "a.py", raw)
    frame = {"headcarry": _panel([head], omitted_count=0, deduplicated_count=0,
                               byte_budget=6000),
             "owner_protocol": {"admitted_actions": actions}}
    body = {"messages": [{"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": frame}})}], "tools": [{"type": "function", "function": {
            "name": "read", "parameters": {"type": "object"}}}]}
    snap = freeze_preparation(body, SimpleNamespace(owner=SimpleNamespace(root=tmp_path)),
                              request_id="head", byte_cap=100_000)
    slot, = snap.slots
    assert not snap.recovery_owner_allowed
    assert domain(snap, slot) == (FULL, DEFERRED)
    rendered = render_assignment(snap, {slot.key: DEFERRED})
    assert raw.decode() not in rendered.wire.decode()
    assert b"Read recovery unavailable" in rendered.wire
    assert verify_certificate(rendered.wire, rendered.certificate)


def test_permission_transitions_do_not_reuse_a_cached_recovery_promise():
    from kernel_working_set_cache import transaction

    slot = source(state="RESIDENT")
    snap = snapshot(slot)

    @transaction
    def exercise():
        for owner_allowed in (True, False, True):
            current = replace(snap, recovery_owner_allowed=owner_allowed)
            rendered = render_assignment(current, {slot.key: DEFERRED})
            assert (b"Retrieve current bytes with: read(" in rendered.wire) == owner_allowed
            assert (b"Read recovery unavailable" in rendered.wire) != owner_allowed
            assert verify_certificate(rendered.wire, rendered.certificate)

    exercise()


@pytest.mark.parametrize("reference", [
    "56af0dc6d6",
    "1f6b7e9bc1",
])
def test_off_request_bytes_match_assigned_g2_base(tmp_path, tmp_path_factory, reference):
    from tools.pinned_tree import pinned_tree

    base = pinned_tree(reference, tmp_path_factory)
    current = Path(__file__).resolve().parents[1]
    # The scripted server is llama.cpp-compatible. Pin that same launch
    # authority on both sides: a loaded host can time out /props on one side
    # and otherwise silently compare two different provider dialects.
    driver = '''
import json, sys
from pathlib import Path
import pytest
from test_request_determinism import _run_scripted_example, EXAMPLES
with pytest.MonkeyPatch.context() as patch:
    patch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off")
    from stateframe import runner
    runner.test_interpreter.choose = lambda *args, **kwargs: (sys.executable, "parity test interpreter")
    patch.setenv("STATEFRAME_PROVIDER", "llama.cpp")
    patch.setenv("SF_LAB_DETFIX3", "on")
    work = Path(sys.argv[1])
    hashes, _, _ = _run_scripted_example(patch, "discount_bug", EXAMPLES["discount_bug"], work)
    spec = json.loads((work / "run/spec.json").read_bytes())
    assert spec["provider_capabilities"]
    assert all(row["dialect"] == "llama.cpp" for row in spec["provider_capabilities"].values())
    Path(sys.argv[2]).write_text(json.dumps(hashes))
'''
    results = []
    for label, tree in (("base", base), ("current", current)):
        destination = tmp_path / f"{label}.json"
        env = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}
        env["PYTHONPATH"] = os.pathsep.join(str(tree / path) for path in
            ("src", "src/stateframe/engine", "tests"))
        result = subprocess.run([sys.executable, "-c", driver, str(tmp_path / label),
            str(destination)], cwd=tree, env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        results.append(json.loads(destination.read_text()))
    assert results[0] and results[0] == results[1]
