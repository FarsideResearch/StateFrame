"""A deferred path the model has since re-read must not keep the receipt PARTIAL.

Regression (mid-run ``source_delivery_partial``): page pressure deferred a file, the
model read it again, and the next request carried it COMPLETE and VERIFIED. The old
deferral notice stays in the stable prefix, so the notice-verified branch skipped the
already-complete path, found nothing to defer and returned before the completion
promotion. The receipt stayed PARTIAL with no failures and the request was refused.
"""
import pytest

import kernel_request_source_window as window

DEFERRED = "shop/inventory/models.py"
OTHER = "shop/pricing.py"


def _receipt(status_by_path):
    return {
        "status": "PARTIAL", "proof_status": "VERIFIED", "proof_failures": [],
        "paths": {path: {"status": status, "proof_status": "VERIFIED"}
                  for path, status in status_by_path.items()},
    }


def _plan(omission_paths=()):
    return {
        "deferred_paths": [DEFERRED], "units": [], "delivery": {},
        "omissions": [{"path": path, "status": "PARTIAL"} for path in omission_paths],
        "coverage": {"delivered_span": {"status": "PARTIAL", "deferred_paths": [DEFERRED]}},
    }


@pytest.fixture
def wire_verified(monkeypatch):
    """Paths COMPLETE in the receipt are proven on the final wire (the exact-wire proof is tested elsewhere)."""
    def verified(receipt, selected):
        return {path for path in selected
                if receipt["paths"].get(path, {}).get("status") == "COMPLETE"}
    monkeypatch.setattr(window, "_verified_complete_final_wire_paths", verified)


@pytest.mark.parametrize("notice_verified", [True, False])
def test_reread_deferred_path_promotes_receipt_to_complete(wire_verified, notice_verified):
    plan, receipt = _plan(omission_paths=(DEFERRED,)), _receipt({DEFERRED: "COMPLETE", OTHER: "COMPLETE"})
    assert window._apply_deferred_delivery_status(
        plan, receipt, [DEFERRED], {DEFERRED, OTHER}, notice_verified=notice_verified) == ()
    assert receipt["status"] == "COMPLETE"
    assert plan["final_delivery_status"] == "COMPLETE"
    assert plan["coverage"]["delivered_span"]["deferred_paths"] == []
    assert plan["omissions"] == []


def test_partial_selected_path_still_not_promoted(wire_verified):
    # Control: a selected path that is not complete on the wire keeps the receipt PARTIAL.
    plan, receipt = _plan(), _receipt({DEFERRED: "COMPLETE", OTHER: "PARTIAL"})
    assert window._apply_deferred_delivery_status(
        plan, receipt, [DEFERRED], {DEFERRED, OTHER}, notice_verified=True) == ()
    assert receipt["status"] == "PARTIAL"
    assert "final_delivery_status" not in plan


def test_relevant_omission_on_other_path_still_not_promoted(wire_verified):
    # Control: an unresolved omission on a path that is not proven complete blocks promotion.
    plan, receipt = _plan(omission_paths=("shop/orders.py",)), _receipt({DEFERRED: "COMPLETE", OTHER: "COMPLETE"})
    window._apply_deferred_delivery_status(plan, receipt, [DEFERRED], {DEFERRED, OTHER}, notice_verified=True)
    assert receipt["status"] == "PARTIAL"


def test_failed_proof_never_promoted(wire_verified):
    plan, receipt = _plan(), _receipt({DEFERRED: "COMPLETE", OTHER: "COMPLETE"})
    receipt["proof_status"] = "FAILED"
    window._apply_deferred_delivery_status(plan, receipt, [DEFERRED], {DEFERRED, OTHER}, notice_verified=True)
    assert receipt["status"] == "PARTIAL"
