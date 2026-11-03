"""A late optional-window fit may defer only pages it actually removes."""
import copy
import json

import pytest

from kernel_frame_request_allocation import FrameRequestAllocationError
from kernel_paged_source import source_revision
from kernel_request_source_window import parse_model_source_window, source_delivery_satisfied
from tests import test_narrow_stall as narrow
from tests.test_deferred_absent_pages import _route


def _wire_bytes(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _has_page(messages, path):
    return any(
        isinstance(message, dict)
        and (parsed := parse_model_source_window(message.get("content"))) is not None
        and parsed.get("path") == path
        for message in messages
    )


def _finalize_case(
        tmp_path, monkeypatch, *, drop_page, late_proof_status="FAILED",
        read_tool_available=True, authorization_status="VERIFIED",
        late_closure_omission=False,
):
    import kernel_completion_owner
    import kernel_frame_allocation_capture
    import kernel_frame_source_snapshot
    import kernel_request_source_window
    import kernel_step_contract

    root = tmp_path / ("drop" if drop_page else "keep")
    (root / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "pkg/__init__.py").write_text("VALUE = 1\n")
    (root / "tests/test_app.py").write_text("def test_app(): assert True\n")
    messages = [{"role": "user", "content": "Fix value."}]
    context = narrow.build_action_context(messages, workspace_root=root)
    context["journal_receipt_records"] = []
    runtime = narrow._runtime_with_product(root)
    runtime._framefit_action_context = context
    frame = {"verification": {"revision": "current", "status": "FAIL"}}
    fit_inputs = []
    fit_calls = 0
    dropped = False
    late_source_diagnostics = []

    def finalize_with_read(current_runtime, capture, body, **kwargs):
        body["tools"] = [{
            "type": "function",
            "function": {"name": "read", "parameters": {"type": "object", "properties": {}}},
        }]
        return kernel_frame_allocation_capture.finalize_frame_candidate(
            current_runtime, capture, body, **kwargs,
        )

    def fit(candidate, *_args, **_kwargs):
        nonlocal fit_calls, dropped
        fit_calls += 1
        fit_inputs.append(_wire_bytes(candidate))
        if drop_page and not dropped:
            path = next(unit["path"] for unit in capture_plan["units"])
            assert _has_page(candidate["messages"], path)
            candidate = copy.deepcopy(candidate)
            candidate["messages"] = [
                message for message in candidate["messages"]
                if not (
                    isinstance(message, dict)
                    and (parsed := parse_model_source_window(message.get("content"))) is not None
                    and parsed.get("path") == path
                )
            ]
            dropped = True
        return candidate

    def verify(plan, wire_messages, *, payload=None, sources=None):
        nonlocal capture_plan
        capture_plan = plan
        path = next(unit["path"] for unit in plan["units"])
        units = [unit for unit in plan["units"] if unit["path"] == path]
        handles = [
            str(unit.get("unit_handle") or unit.get("unit_id") or unit.get("handle") or "")
            for unit in units
        ]
        omitted_handles = [
            str(unit.get("handle", unit.get("unit_id", index)))
            for index, unit in enumerate(units)
        ]
        revision = str(plan.get("revisions", {}).get(path) or source_revision(sources[path]))
        route = _route(path, revision=revision, handles=handles)
        if path in plan.get("deferred_paths", ()):
            return {
                "status": "DEFERRED",
                "proof_status": "VERIFIED",
                "deferred_paths": [path],
                "deferred_notice_verified": True,
                "read_tool_available": True,
                "paths": {path: {
                    "status": "DEFERRED",
                    "proof_status": "VERIFIED",
                    "revision": revision,
                    "retrieval_route": route,
                }},
            }
        if _has_page(wire_messages, path):
            return {
                "status": "COMPLETE",
                "proof_status": "VERIFIED",
                "proof_failures": [],
                "paths": {path: {"status": "COMPLETE", "proof_status": "VERIFIED"}},
            }
        return {
            "status": "PARTIAL",
            "proof_status": late_proof_status,
            "read_tool_available": read_tool_available,
            "workspace_authorization": {"status": authorization_status},
            "proof_failures": ([{
                "kind": "final_wire_proof_missing",
                "path": path,
                "expected": 1,
                "actual": 0,
            }] if late_proof_status == "FAILED" else []),
            "paths": {path: {
                "status": "PARTIAL",
                "proof_status": late_proof_status,
                "selected_units": len(units),
                "delivered_units": 0,
                "omitted_units": len(units),
                "omitted_unit_handles": omitted_handles,
                "revision": revision,
                "retrieval_route": route,
            }},
        }

    capture_plan = {}

    def verify_with_omission(plan, wire_messages, *, payload=None, sources=None):
        receipt = verify(plan, wire_messages, payload=payload, sources=sources)
        if (late_closure_omission and receipt.get("status") == "PARTIAL"
                and receipt.get("proof_status") == "FAILED"):
            path = next(unit["path"] for unit in plan["units"])
            plan["omissions"] = [{
                "path": path,
                "status": "UNRECONCILED",
                "reason": "selected closure source omitted by late fit",
            }]
            late_source_diagnostics.append(
                kernel_frame_allocation_capture.source_refusal(
                    plan, receipt, sources,
                ).source_diagnostic
            )
        return receipt

    with monkeypatch.context() as patch:
        patch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
        patch.setattr(narrow, "finalize_frame_candidate", finalize_with_read)
        patch.setattr(kernel_completion_owner, "_current_tool_admission",
                      lambda *_args: (True, None))
        patch.setattr(kernel_frame_source_snapshot, "recent_editable_read_windows",
                      lambda *_args: [])
        patch.setattr(kernel_frame_source_snapshot, "editable_product_windows",
                      lambda *_args: ([], []))
        patch.setattr(kernel_frame_source_snapshot, "public_check_source_windows",
                      lambda *_args: [])
        patch.setattr(kernel_frame_source_snapshot, "fit_recent_read_windows", fit)
        patch.setattr(kernel_request_source_window, "verify_final_source_delivery", verify_with_omission)
        patch.setattr(kernel_step_contract, "reconcile_final_source_access",
                      lambda *_args: False)
        result, capture = narrow._qualified_fixture_request(
            runtime, frame, messages, context, baseline=False,
            monkeypatch=patch, validate=False, return_capture=True,
        )

    return result, capture, fit_inputs, fit_calls, dropped, late_source_diagnostics


def test_late_fit_defers_missing_page_and_noop_fit_preserves_body_bytes(tmp_path, monkeypatch):
    dropped, dropped_capture, _, drop_fit_calls, did_drop, _ = _finalize_case(
        tmp_path, monkeypatch, drop_page=True,
    )
    path = "pkg/__init__.py"
    assert did_drop
    assert drop_fit_calls == 2
    assert dropped_capture._frame_candidate_inputs["source_plan"]["deferred_paths"] == [path]
    assert any(
        "[Deferred required source paths]" in str(message.get("content", ""))
        and path in str(message.get("content", ""))
        for message in dropped["messages"]
    )
    receipt = dropped_capture._frame_candidate_inputs["receipt"]
    assert source_delivery_satisfied(receipt)
    assert receipt["status"] == "DEFERRED"

    kept, kept_capture, fit_inputs, keep_fit_calls, did_keep_drop, _ = _finalize_case(
        tmp_path, monkeypatch, drop_page=False,
    )
    assert not did_keep_drop
    assert keep_fit_calls == 1
    assert kept_capture._frame_candidate_inputs["source_plan"].get("deferred_paths", []) == []
    assert fit_inputs == [_wire_bytes(kept)]
    assert source_delivery_satisfied(kept_capture._frame_candidate_inputs["receipt"])


def test_late_fit_defers_verified_partial_receipt_with_authorized_read(tmp_path, monkeypatch):
    result, capture, _, fit_calls, did_drop, _ = _finalize_case(
        tmp_path, monkeypatch, drop_page=True, late_proof_status="VERIFIED",
    )

    assert did_drop
    assert fit_calls == 2
    assert capture._frame_candidate_inputs["source_plan"]["deferred_paths"] == ["pkg/__init__.py"]
    assert any(
        "[Deferred required source paths]" in str(message.get("content", ""))
        and "pkg/__init__.py" in str(message.get("content", ""))
        for message in result["messages"]
    )
    receipt = capture._frame_candidate_inputs["receipt"]
    assert source_delivery_satisfied(receipt)
    assert receipt["status"] == "DEFERRED"


@pytest.mark.parametrize(
    ("read_tool_available", "authorization_status"),
    ((False, "VERIFIED"), (True, "UNVERIFIED")),
)
def test_late_fit_refuses_verified_partial_without_read_authority(
        tmp_path, monkeypatch, read_tool_available, authorization_status,
):
    with pytest.raises(FrameRequestAllocationError, match="proof=VERIFIED"):
        _finalize_case(
            tmp_path, monkeypatch, drop_page=True, late_proof_status="VERIFIED",
            read_tool_available=read_tool_available,
            authorization_status=authorization_status,
        )


def test_late_fit_defers_partial_delivery_with_closure_omission(tmp_path, monkeypatch):
    result, capture, _, fit_calls, did_drop, diagnostics = _finalize_case(
        tmp_path, monkeypatch, drop_page=True, late_closure_omission=True,
    )

    assert did_drop
    assert fit_calls == 2
    assert diagnostics[0]["failures"] == [
        "final_wire_proof_missing",
        "closure_units_undelivered",
        "closure_omissions_unreconciled",
    ]
    assert capture._frame_candidate_inputs["source_plan"]["deferred_paths"] == [
        "pkg/__init__.py",
    ]
    receipt = capture._frame_candidate_inputs["receipt"]
    assert source_delivery_satisfied(receipt)
    assert receipt["status"] == "DEFERRED"
    assert any(
        "[Deferred required source paths]" in str(message.get("content", ""))
        and "pkg/__init__.py" in str(message.get("content", ""))
        for message in result["messages"]
    )
