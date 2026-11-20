"""Qualification reuses detached proofs while send authority stays live."""
from contextvars import copy_context
from dataclasses import replace
import hashlib
import json
import pickle
import sys
from types import ModuleType, SimpleNamespace

import pytest

import kernel_body_serialization_memo as serialization
import kernel_completion_owner
import kernel_final_wire_allocation as final_allocation
import kernel_frame_allocation_validation
import kernel_frame_request_allocation as allocation_module
import kernel_tool_tape
import kernel_working_set_proof as proof
from kernel_frame_optional_allocation import FrameAllocationReplanNeeded, identity
from kernel_frame_request_allocation import (
    FrameRequestAllocation,
    FrameRequestAllocationError,
    canonical_allocation_bytes,
    frame_request_allocation_digest,
)
from kernel_request_kind import RequestKindError, bind_primary, guard_physical_send
from kernel_request_measurement import RequestMeasurement
from kernel_request_seal import OwnerPresentation, QualifiedRequest, RequestSealError, freeze
from kernel_working_set_allocator import DEFERRED, FULL
from test_working_set_render import snapshot, source


@pytest.fixture
def pilot(monkeypatch):
    pilot = ModuleType("stateframe.send1_wave4")
    pilot.selected = False
    pilot.enabled = lambda: pilot.selected
    monkeypatch.setitem(sys.modules, pilot.__name__, pilot)
    return pilot


def _allocation(body=None, dependency_revisions=None):
    return FrameRequestAllocation.create(
        identity={"request_lineage": "request", "generation": 0,
                  "product_revision": "revision", "task_digest": "task",
                  "ledger_digest": "ledger", "newest_exchange_digest": "exchange",
                  "workspace_authorization": {"generation": "0"},
                  "journal_semantic_cutoff": "cutoff"},
        capacity={"byte_ceiling": 100_000, "token_ceiling": 1000, "frame_ceiling": 2000,
                  "measurement_identity": "tokenize", "template_identity": "template",
                  "tool_schema_digest": "tools"},
        source_obligations=[],
        source_layout={"task_message": {"role": "user", "content": "Task: résumé"},
                       "newest_user_suffix": [], "newest_user_suffix_start": 3,
                       "newest_tool_exchange": [], "newest_exchange_state": "NO_TOOL_CALLS",
                       "newest_exchange_start": 3, "current_user_facts": [], "carriers": []},
        optional_heads={"candidates": [], "retained_identities": [], "allowance": 1,
                        "dependency_revisions": dependency_revisions or {},
                        "omitted_count": 0, "deduplicated_count": 0,
                        "render_policy_identity": "policy"},
        site_inventory={"inventory_identity": [], "selected_identities": [],
                        "open_sites": [], "encoding_mode": "full_inventory"},
        outcome={"status": "COMPLETE", "final_verification_digest": "verification",
                 "body_digest": identity(body)},
    )


def test_section_access_skips_whole_document_decode(pilot, monkeypatch):
    eager = _allocation()
    pilot.selected = True
    fast = _allocation()
    assert fast.to_bytes() == eager.to_bytes()
    assert fast.digest == eager.digest

    decoded = []
    decode = allocation_module._decode_document

    def count(payload):
        decoded.append(payload)
        return decode(payload)

    monkeypatch.setattr(allocation_module, "_decode_document", count)
    changed = fast.identity
    changed["workspace_authorization"]["generation"] = "1"
    assert fast.identity == eager.identity
    assert len(decoded) == 1
    assert fast.source_layout == eager.source_layout
    assert len(decoded) == 2

    # Checkpoint restoration drops the index. Imported bytes rebuild it only
    # after the strict constructor checks.
    restored = pickle.loads(pickle.dumps(fast))
    assert restored.identity == fast.identity
    assert len(decoded) == 3
    imported = FrameRequestAllocation.from_bytes(fast.to_bytes())
    assert imported.source_layout == fast.source_layout
    assert len(decoded) == 6
    noncanonical = fast._payload.replace(b'"generation":0', b'"generation":0,"generation":0')
    with pytest.raises(FrameRequestAllocationError, match="not canonical"):
        FrameRequestAllocation(noncanonical, frame_request_allocation_digest(noncanonical))
    bad = fast.to_document()
    bad["source_layout"]["newest_exchange_start"] += 1
    bad.pop("digest")
    bad["digest"] = frame_request_allocation_digest(canonical_allocation_bytes(bad))
    with pytest.raises(FrameRequestAllocationError, match="directly precede"):
        FrameRequestAllocation.from_bytes(canonical_allocation_bytes(bad))

    monkeypatch.setattr(kernel_tool_tape, "_replay_active", True)
    replayed = _allocation()
    assert replayed.to_bytes() == fast.to_bytes()
    assert not serialization.qualification_reuse_enabled()


def test_fresh_wire_snapshot_reencodes_mutable_inputs(monkeypatch):
    body = {"messages": [{"role": "user", "content": "Task: open"}],
            "tools": [{"function": {"description": "look"}}]}
    original = serialization.plain_wire(body, snapshot=True)
    monkeypatch.setattr(serialization, "_entry", lambda _: pytest.fail("mutable memo used"))
    body["messages"][0]["content"] = "Task: read"
    body["tools"][0]["function"]["description"] = "find"
    body["messages"] = body.pop("messages")
    updated = serialization.plain_wire(body, snapshot=True)
    assert updated.wire == json.dumps(body).encode()
    assert updated.sha256 == hashlib.sha256(updated.wire).hexdigest()
    assert len(updated.wire) == len(original.wire)
    assert updated.sha256 != original.sha256


def test_independent_proof_rebuilds_once_and_later_proves_delivered_facts(monkeypatch):
    slot = source(state="RESIDENT")
    snap = snapshot(slot, recovery_allowed=False)
    first = proof.render_assignment(snap, {slot.key: DEFERRED})
    fresh = proof.render_assignment(replace(snap, request_id="next", due_ordinal=2),
                                    {slot.key: FULL})
    messages = json.loads(first.wire)["messages"]
    assert any("Read recovery unavailable" in row["content"] for row in messages)
    assert all(slot.payload not in row["content"] for row in messages)

    calls = []
    render = proof.render_assignment

    def count(*args):
        calls.append(args[0].request_id)
        return render(*args)

    monkeypatch.setattr(proof, "render_assignment", count)
    assert proof.verify_certificate(first.wire, first.certificate)
    assert proof.verify_messages(messages, first.certificate)
    assert calls == ["request"] * 3
    calls.clear()
    with proof.qualification_proof("request", "request", 0):
        assert proof.verify_certificate(first.wire, first.certificate)
        assert proof.verify_messages(messages, first.certificate)
        assert proof.verify_certificate(first.wire, first.certificate)
        assert calls == ["request"]
        messages[0]["content"] = "changed"
        with pytest.raises(proof.CertificateError, match="product source carriers"):
            proof.verify_messages(messages, first.certificate)

    # A new qualifier must validate and deliver the source it now includes.
    with proof.qualification_proof("next", "request", 1):
        delivered = json.loads(fresh.wire)["messages"]
        assert proof.verify_messages(delivered, fresh.certificate)
        assert any(slot.page == row["content"] for row in delivered)
    assert calls == ["request", "next", "next"]


def test_proof_cache_misses_changed_witness_limits_and_recovery(monkeypatch):
    slot = source()
    result = proof.render_assignment(snapshot(slot), {slot.key: FULL})
    certificate = result.certificate
    with proof.qualification_proof("request", "request", 0):
        assert proof.verify_certificate(result.wire, certificate)
        slot_witness = certificate.witnesses[0]
        monkeypatch.setattr(proof, "_unit_active", lambda: True)
        assert proof._proof_key(proof._QUALIFICATION_PROOF.get(), certificate) is None
        monkeypatch.setattr(proof, "_unit_active", lambda: False)
        # Check the actual nested fields even when the outer object and claimed
        # digests are identical.
        index = slot_witness.message_index
        try:
            object.__setattr__(slot_witness, "message_index", index + 1)
            with pytest.raises(proof.CertificateError):
                proof.verify_certificate(result.wire, certificate)
        finally:
            object.__setattr__(slot_witness, "message_index", index)
        for updated in (replace(certificate.snapshot, prompt_limit=10),
                        replace(certificate.snapshot, recovery_owner_allowed=False),
                        replace(certificate.snapshot, current_revisions=(("a.py", "other"),))):
            with pytest.raises(proof.CertificateError, match="recomputation"):
                proof.verify_certificate(result.wire, replace(certificate, snapshot=updated))
        with pytest.raises(proof.CertificateError, match="body digest"):
            proof.verify_certificate(result.wire.replace(b"Task: fix", b"Task: run"), certificate)
        assert proof.verify_certificate(result.wire, certificate)


def test_qualifier_discards_proof_on_failure_and_retry(pilot, monkeypatch):
    pilot.selected = True
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    rendered = proof.render_assignment(snapshot(), {})
    body = {"messages": [{"role": "user", "content": '[StateFrame]\n{"framefit5":{}}'}]}
    capture = SimpleNamespace(request_id="request", _replacement_allocation=_allocation())
    runtime = SimpleNamespace()
    contexts = []

    def fail(*args, **kwargs):
        scope = proof._QUALIFICATION_PROOF.get()
        assert (scope.request_id, scope.lineage, scope.generation) == ("request", "request", 1)
        assert proof.verify_certificate(rendered.wire, rendered.certificate)
        contexts.append(copy_context())
        raise TimeoutError("cancelled during qualification")

    monkeypatch.setattr(final_allocation, "_build_qualified_request", fail)
    with pytest.raises(TimeoutError, match="cancelled"):
        final_allocation.build_qualified_request(capture, runtime, body, endpoint="local", key="")
    inherited = contexts[0].run(proof._QUALIFICATION_PROOF.get)
    assert inherited.closed and inherited.verified is None

    def validate(*args):
        assert proof._QUALIFICATION_PROOF.get().verified is None
        assert proof.verify_certificate(rendered.wire, rendered.certificate)

    capture._working_set_certificate = rendered.certificate
    capture.qualified_request = SimpleNamespace(
        input_digest=hashlib.sha256(freeze({"body": body, "endpoint": "local"})).hexdigest(),
        validate=validate,
    )
    with proof.qualification_proof("request", "request", 1):
        assert proof.verify_certificate(rendered.wire, rendered.certificate)
        assert final_allocation.build_qualified_request(
            capture, runtime, body, endpoint="local", key="",
        ) is capture.qualified_request
        assert proof._QUALIFICATION_PROOF.get().verified is None


@pytest.mark.parametrize("enabled", [False, True])
def test_read_contract_replacement_preserves_disabled_qualification(pilot, tmp_path, monkeypatch, enabled):
    from kernel_release_wire_claims import READ_PREFIX
    from test_read_admission_qualification import seal_fixture

    pilot.selected = enabled
    capture, runtime, body, *_ = seal_fixture(tmp_path, monkeypatch)
    quoted = {"role": "user", "content": READ_PREFIX + "quoted in task evidence"}
    capsule = READ_PREFIX + '{"schema":"read-contract-wire/1"}'
    body["messages"].append(quoted)
    monkeypatch.setattr("kernel_wire_truth_gate.enabled", lambda: True)
    monkeypatch.setattr("kernel_wire_truth_gate.add_fact_messages", lambda body, *_args, **_kwargs: body)
    monkeypatch.setattr("kernel_release_wire_claims.read_contract_capsule", lambda _runtime: (capsule, {}))

    class ObservedMessages(Exception):
        pass

    def observe(_runtime, messages):
        assert (quoted in messages) is enabled
        assert messages[-1] == {"role": "user", "content": capsule}
        raise ObservedMessages

    monkeypatch.setattr("kernel_read_once.append_final_message", observe)
    with pytest.raises(ObservedMessages):
        final_allocation.build_qualified_request(capture, runtime, body, endpoint="http://local/v1", key="")


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_selection_moves_into_final_validation_only_when_enabled(pilot, tmp_path, monkeypatch, enabled):
    from kernel_request_seal import FrameSelectionProposal
    from test_read_admission_qualification import seal_case

    pilot.selected = enabled

    class ReachedValidation(Exception):
        pass

    def observe(*_args, frame_selection=None, optional_inputs=None, **_kwargs):
        if enabled:
            assert optional_inputs is not None
            assert frame_selection is None
        else:
            assert isinstance(frame_selection, FrameSelectionProposal)
            assert optional_inputs is None
        raise ReachedValidation

    monkeypatch.setattr("kernel_stateframe_framefit5.validate_final_product_boundary", observe)
    with pytest.raises(ReachedValidation):
        seal_case(tmp_path, monkeypatch, "K1")


def _sealed(tmp_path, monkeypatch):
    body = {"model": "local", "messages": [
        {"role": "user", "content": "Task: look"},
        {"role": "user", "content": '[StateFrame]\n{"state":{"framefit5":{}}}'},
    ], "tools": [{"type": "function", "function": {"name": "read", "description": "look"}}]}
    dependency = tmp_path / "dependency.py"
    dependency.write_bytes(b"old")
    allocation = _allocation(body, {dependency.name: "sha256:" + hashlib.sha256(b"old").hexdigest()})
    binding = {"product_revision": "revision"}
    runtime = SimpleNamespace(current_request_id="request", owner=SimpleNamespace(root=tmp_path),
                              binding=lambda: binding, checks=[])
    runtime._request_token_meter = SimpleNamespace(
        provider_identity="{}", tokenizer_identity=lambda body, endpoint: "test",
    )
    contract = {"route": "read"}
    measured = RequestMeasurement.bind(
        body, tokenizer_identity="test", request_id="request", logical_request_id="request",
        tokens=1, method="tokenize", template_overhead_tokens=0, max_prompt_tokens=1_000,
        max_request_bytes=100_000,
    )
    receipt = {"body_sha256": measured.body_sha256, "request_bytes": measured.request_bytes,
               **measured.receipt_fields()}
    owner = OwnerPresentation(hashlib.sha256(freeze(contract)).hexdigest(), None, None)
    authority = {"request_action_contract": contract, "_request_source_closure_override": None,
                 "request_source_final_delivery": {}, "request_budget_final_wire": receipt,
                 "final_tool_surface": {}, "framefit5_product_validation": {},
                 "host_contribution": None}
    wire = json.dumps(body).encode()
    record = QualifiedRequest(
        "request", wire, hashlib.sha256(wire).hexdigest(), allocation,
        freeze(authority), freeze(binding), "", owner, None, "request", measured,
    )
    capture = SimpleNamespace(
        **authority, request_id="request", frame_request_allocation=allocation,
        qualified_request=record, owner_presentation=owner, _request_measurement=measured,
        _qualified_adapter="test", _qualified_request_gate=lambda: None,
        _qualified_host_text_preparer=lambda: None, _qualified_request_gate_passed=True,
    )
    bind_primary(capture, runtime)
    monkeypatch.setattr(kernel_frame_allocation_validation, "validate_allocation_authority",
                        lambda *args: runtime.checks.append("allocation"))
    monkeypatch.setattr(kernel_completion_owner, "assert_request_action_contract",
                        lambda *args, **kwargs: runtime.checks.append("owner"))
    return body, record, capture, runtime, dependency


@pytest.mark.parametrize("enabled", [False, True])
def test_seal_rechecks_live_authority_before_transport(pilot, enabled, tmp_path, monkeypatch):
    pilot.selected = enabled
    body, record, capture, runtime, dependency = _sealed(tmp_path, monkeypatch)

    def send():
        guard_physical_send(capture, runtime, body, record.wire)

    send()
    send()
    assert runtime.checks == ["owner", "allocation"] * 2

    body["messages"][0]["content"] = "Task: edit"
    with pytest.raises(RequestSealError, match="body changed"):
        send()
    body["messages"][0]["content"] = "Task: look"
    body["tools"][0]["function"]["description"] = "edit"
    with pytest.raises(RequestSealError, match="body changed"):
        send()
    body["tools"][0]["function"]["description"] = "look"
    body["model"] = body.pop("model")
    with pytest.raises(RequestSealError, match="body changed"):
        send()
    body = record.body

    dependency.write_bytes(b"new")
    with pytest.raises(FrameAllocationReplanNeeded, match="optional dependency changed"):
        send()
    dependency.write_bytes(b"old")
    capture.request_action_contract["route"] = "edit"
    with pytest.raises(RequestSealError, match="authority changed"):
        send()
    capture.request_action_contract["route"] = "read"
    capture.request_budget_final_wire["max_prompt_tokens"] -= 1
    with pytest.raises(RequestSealError, match="authority changed"):
        send()
    capture.request_budget_final_wire["max_prompt_tokens"] += 1
    capture._qualified_request_gate_passed = False
    with pytest.raises(RequestKindError, match="primary_gate_binding_missing"):
        send()
    assert runtime.checks == ["owner", "allocation"] * 3
