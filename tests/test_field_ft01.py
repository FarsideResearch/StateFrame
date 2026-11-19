"""Before its first send, ft01 deferred a nonempty file whose units hit the 96-unit cap."""
import json
from types import SimpleNamespace

import pytest
from kernel_measurement import AuditJournal
from kernel_request_diet import RequestTokenMeter
from kernel_request_source_window import source_delivery_satisfied, source_revision
from kernel_working_set_live import prepare_live, read_admission_snapshot
from kernel_working_set_proof import verify_certificate

TARGET = "package/target.py"
ORPHAN = "package/dependency.py"
HANDLE = "expand:" + "a" * 64


def minimal_body() -> dict[str, object]:
    frame = {"mode": "FRAME_ONLY", "owner_protocol": {"admitted_actions": ["read"]}}
    return {"model": "scripted", "max_tokens": 100, "messages": [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Task: repair the target module"},
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({"state": {"framefit5": frame}})},
        {"role": "user", "content": "[WIRE VOLATILE TAIL]"},
    ], "tools": [{"type": "function", "function": {"name": "read", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
        "required": ["path"]}}}]}


def prepare(plan, body, *, workspace, scratch):
    journal = AuditJournal(scratch, "field-ft01", {})
    capture = SimpleNamespace(request_id="ft01-first-request", step=0,
                              journal=journal, request_action_contract={})
    runtime = SimpleNamespace(owner=SimpleNamespace(root=workspace), journal=journal,
                              _request_source_closure=plan, _active_capture=capture,
                              _request_token_meter=RequestTokenMeter(disabled=True))
    _, candidate = prepare_live(capture, runtime, body, dialect="llama.cpp", reasoning_mode="on")
    return candidate, capture.request_source_final_delivery


@pytest.fixture
def ft01(tmp_path):
    # Model the result of hitting the closure cap: a retained target span and
    # a deferred dependency with an omission handle and zero selected units.
    sources = {path: "".join(f"value_{line} = {line}\n" for line in range(1, 46))
               for path in (TARGET, ORPHAN)}
    workspace = tmp_path / "workspace"
    for path, source in sources.items():
        file = workspace / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source, encoding="utf-8")
    plan = {
        "paths": [TARGET, ORPHAN], "target_path": TARGET,
        "units": [{"path": TARGET, "start_line": 20, "end_line": 34, "role": "body",
                   "priority": 0, "unit_handle": "unit:target"}],
        "omissions": [{"path": ORPHAN, "status": "PARTIAL", "range": "27-44",
                       "reason": "semantic unit exceeds bounded closure", "continuation": HANDLE}],
        "deferred_paths": [ORPHAN], "source_contents": sources,
        "revisions": {path: source_revision(source) for path, source in sources.items()},
        "workspace_authorization": {"scope": str(workspace), "generation": "0"},
    }
    return plan, workspace, tmp_path


def test_first_request_qualifies_deferred_semantic_omission(ft01, monkeypatch):
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    monkeypatch.setenv("STATEFRAME_FINAL_CACHE_LAYOUT", "off")
    plan, workspace, scratch = ft01
    candidate, receipt = prepare(plan, minimal_body(), workspace=workspace, scratch=scratch)
    assert verify_certificate(candidate.wire, candidate.rendered.certificate)
    assert source_delivery_satisfied(receipt)
    row = receipt["paths"][ORPHAN]
    assert row["proof_status"] == "VERIFIED"
    if row["status"] == "COMPLETE":
        assert any(start <= 27 and end >= 44 for start, end in row["ranges"])
        assert receipt["delivery_proofs"][ORPHAN]
    else:
        assert row["status"] == "DEFERRED"
        route = row["retrieval_route"]
        assert receipt["deferred_notice_verified"] and receipt["read_tool_available"]
        assert route["path"] == ORPHAN and route["revision"] == plan["revisions"][ORPHAN]
        assert route["status"] == "EXECUTABLE_BOUNDED_READ" and HANDLE in route["unit_handles"]


@pytest.mark.parametrize("case", [
    "valid", "stale-revision", "out-of-bounds-range", "over-unit-limit",
    "not-deferred", "already-selected",
])
def test_read_admission_freezes_only_valid_deferred_omissions(ft01, monkeypatch, case):
    plan, workspace, _ = ft01
    if case == "stale-revision":
        plan["revisions"][ORPHAN] = source_revision("changed\n")
    elif case == "out-of-bounds-range":
        plan["omissions"][0]["range"] = "27-46"
    elif case == "over-unit-limit":
        omission = plan["omissions"][0]
        plan["omissions"] = [dict(omission, continuation=f"expand:{index:064x}")
                             for index in range(65)]
    elif case == "not-deferred":
        plan["deferred_paths"] = []
    elif case == "already-selected":
        plan["units"].append(dict(plan["units"][0], path=ORPHAN, unit_handle="unit:dependency"))
    original_units = plan["units"].copy()

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    runtime = SimpleNamespace(owner=SimpleNamespace(root=workspace), _request_source_closure=plan,
                              _read_admission_provider_body=minimal_body())
    snapshot = read_admission_snapshot(runtime, None, byte_cap=100_000)
    assert snapshot is not None
    ranges = [(slot.identity.first_line, slot.identity.last_line)
              for slot in snapshot.sources if slot.identity.path == ORPHAN]
    expected = {"valid": [(27, 44)], "already-selected": [(20, 34)]}
    assert ranges == expected.get(case, [])
    assert runtime._request_source_closure is plan
    assert plan["units"] == original_units
