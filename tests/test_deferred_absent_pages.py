"""Only selected pages omitted from the final wire can use bounded-read deferral."""
import copy

from kernel_request_source_window import partial_source_paths_for_deferred_delivery


REVISION = "sha256:" + "0" * 64
AWARDS = "shop/loyalty/awards.py"
FULLY_DELIVERED = "shop/loyalty/policy.py"
ALREADY_DEFERRED = "shop/inventory/models.py"


def _route(path, *, revision=REVISION, handles=None):
    return {
        "status": "EXECUTABLE_BOUNDED_READ",
        "tool": "read",
        "path": path,
        "revision": revision,
        "unit_handles": list(handles or (f"unit:{path}",)),
        "ranges": [[1, 63]],
    }


def _row(path, *, delivered, omitted):
    return {
        "status": "PARTIAL",
        "proof_status": "FAILED",
        "selected_units": delivered + omitted,
        "delivered_units": delivered,
        "omitted_units": omitted,
        "revision": REVISION,
        "retrieval_route": _route(path),
    }


def _plan():
    paths = [AWARDS, FULLY_DELIVERED, ALREADY_DEFERRED]
    return {
        "paths": paths,
        "deferred_paths": [ALREADY_DEFERRED],
        "revisions": {path: REVISION for path in paths},
        "units": [
            {"path": path, "unit_handle": f"unit:{path}"}
            for path in paths
        ],
    }


def _receipt():
    return {
        "status": "PARTIAL",
        "proof_status": "FAILED",
        "read_tool_available": True,
        "workspace_authorization": {"status": "VERIFIED"},
        "proof_failures": [{
            "kind": "final_wire_proof_missing",
            "path": AWARDS,
            "expected": 1,
            "actual": 0,
        }],
        "paths": {
            AWARDS: _row(AWARDS, delivered=0, omitted=1),
            FULLY_DELIVERED: _row(FULLY_DELIVERED, delivered=1, omitted=0),
            ALREADY_DEFERRED: _row(ALREADY_DEFERRED, delivered=0, omitted=4),
        },
    }


def _defer(plan, receipt, **kwargs):
    return partial_source_paths_for_deferred_delivery(
        copy.deepcopy(plan), copy.deepcopy(receipt), **kwargs,
    )


def test_absent_pages_defer_only_omitted_undeferred_paths():
    plan, receipt = _plan(), _receipt()
    assert _defer(plan, receipt, absent_pages_deferrable=True) == (AWARDS,)

    verified = copy.deepcopy(receipt)
    verified["proof_status"] = "VERIFIED"
    verified["proof_failures"] = []
    for row in verified["paths"].values():
        row["proof_status"] = "VERIFIED"
    assert _defer(plan, verified) == (AWARDS, FULLY_DELIVERED, ALREADY_DEFERRED)


def test_absent_page_deferral_rejects_other_failures_and_unverified_context():
    plan, receipt = _plan(), _receipt()
    for failure in (
        {"kind": "source_window_bytes_mismatch", "path": FULLY_DELIVERED},
        {"kind": "source_revision_mismatch", "path": FULLY_DELIVERED},
        {"kind": "delivery_range_invalid", "path": FULLY_DELIVERED},
        {"kind": "final_message_location_unavailable", "path": FULLY_DELIVERED},
        {"kind": "workspace_authorization_unverified", "status": "UNVERIFIED"},
        {"kind": "source_body_blank", "path": FULLY_DELIVERED},
        {"kind": "final_wire_proof_incomplete", "path": FULLY_DELIVERED},
        {"kind": "final_wire_proof_missing"},
    ):
        mixed = copy.deepcopy(receipt)
        mixed["proof_failures"].append(failure)
        assert _defer(plan, mixed, absent_pages_deferrable=True) == ()

    without_read = copy.deepcopy(receipt)
    without_read["read_tool_available"] = False
    assert _defer(plan, without_read, absent_pages_deferrable=True) == ()

    unauthorized = copy.deepcopy(receipt)
    unauthorized["workspace_authorization"]["status"] = "UNVERIFIED"
    assert _defer(plan, unauthorized, absent_pages_deferrable=True) == ()
