"""DECIDE2 sees host check receipts as their journaled rows, never as live objects.

0.5.9 handed ``CheckReceipt`` dataclasses to DECIDE2 as ``public_evidence``. Its evidence
digest serializes with ``default=str``, so the digest silently covered each receipt's
``repr`` instead of its recorded fields. Host rows now arrive in their journaled
``to_dict()`` form with audit-only fields removed, like test receipt rows.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace

from kernel_stateframe_runtime import _model_facing_check_evidence
from kernel_v46_mechanism_wiring import _digest
from kernel_wire_identity import WIRE_AUDIT_ONLY_FIELDS, WIRE_ENV_IDENTITY_FIELDS
from test_finish_on_green_receipt_json import _receipt

TEST_ROW = {"id": "test-1", "request_id": "request-1", "description": "tests passed"}


def _runtime(*host_rows):
    return SimpleNamespace(test_receipts=[dict(TEST_ROW)], host_check_receipts=tuple(host_rows),
                           _scope_lock_last_result=None)


def _without_audit(row: Mapping[str, object]) -> dict[str, object]:
    excluded = WIRE_AUDIT_ONLY_FIELDS | WIRE_ENV_IDENTITY_FIELDS
    return {key: value for key, value in row.items() if key not in excluded}


def test_host_receipts_reach_decide2_as_journaled_rows():
    receipt = _receipt()
    evidence = _model_facing_check_evidence(_runtime(receipt))
    rows = evidence["public_evidence"]
    assert [row["id"] for row in rows[:1]] == ["test-1"]
    host = rows[1]
    assert isinstance(host, Mapping)
    expected = json.loads(json.dumps(_without_audit(receipt.to_dict())))
    assert json.loads(json.dumps(host)) == expected


def test_decide2_evidence_serializes_without_the_str_fallback():
    evidence = _model_facing_check_evidence(_runtime(_receipt()))
    # _digest falls back to default=str; the rows must never need it.
    json.dumps(evidence["public_evidence"], sort_keys=True)
    json.dumps(evidence["dependency_identity"])


def test_equal_receipts_give_one_decide2_evidence_digest():
    first = _model_facing_check_evidence(_runtime(_receipt()))
    second = _model_facing_check_evidence(_runtime(_receipt()))
    assert first == second
    assert _digest(first["public_evidence"]) == _digest(second["public_evidence"])


def test_a_changed_receipt_field_changes_the_decide2_evidence_digest():
    passing = _model_facing_check_evidence(_runtime(_receipt()))
    failing = _model_facing_check_evidence(
        _runtime(_receipt(status="COUNTEREXAMPLE", reason="no")))
    assert _digest(passing["public_evidence"]) != _digest(failing["public_evidence"])
    assert passing["dependency_identity"] != failing["dependency_identity"]


def test_rows_already_json_native_are_unchanged():
    evidence = _model_facing_check_evidence(_runtime())
    assert evidence["public_evidence"] == (TEST_ROW,)
