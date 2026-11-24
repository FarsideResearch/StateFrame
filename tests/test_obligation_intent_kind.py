"""Receipt intent is selector evidence only for test rows."""
from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path

import pytest

from kernel_bolt import KernelBolt
import kernel_history_receipts as receipt_history
import kernel_obligations as obligations


SELECTOR = "tests/test_a.py::test_x"


def _bolt(root: Path, receipts: Iterable[object] | None = None) -> KernelBolt:
    bolt = KernelBolt.__new__(KernelBolt)
    bolt.root = root
    bolt._receipts = receipts if receipts is not None else []
    bolt.run_id = "obligation-intent-test"
    bolt.current_step = 1
    bolt.workspace_revision = 1
    bolt.current_request_id = None
    bolt.audit_journal = None
    bolt.measurement = None
    return bolt


def _oracle_state(rows: Iterable[object]) -> obligations.ObligationState:
    graph = obligations.compile_obligation_graph(
        {"task_id": "intent-kind"}, oracle_ids=[SELECTOR]
    )
    updated = obligations.update_obligation_graph(graph, rows)
    return next(item.state for item in updated.obligations if item.key == f"oracle:{SELECTOR}")


@pytest.mark.parametrize(
    ("kind", "result"),
    [("done", "ACCEPTED"), ("grep", "ERROR"), ("probe", "PASS")],
)
def test_non_test_intent_does_not_change_oracle(tmp_path, kind, result):
    bolt = _bolt(tmp_path)
    bolt.receipt(kind, SELECTOR, SELECTOR, result)

    assert _oracle_state(bolt._receipts) is obligations.ObligationState.UNKNOWN


def test_done_claim_does_not_set_oracle_obligation(tmp_path):
    bolt = _bolt(tmp_path)
    bolt.receipt("done", "done summary", SELECTOR, "CHALLENGED")
    bolt._receipts[-1]["claim"] = {
        f"oracle:{SELECTOR}": {"result": "ACCEPTED"},
        "contract:copy": {"result": "ACCEPTED"},
    }
    graph = obligations.ObligationGraph(
        task_id="intent-kind",
        obligations=(
            obligations.Obligation(f"oracle:{SELECTOR}", obligations.ObligationKind.ORACLE),
            obligations.Obligation("contract:copy", obligations.ObligationKind.CONTRACT),
        ),
    )
    updated = obligations.update_obligation_graph(graph, bolt._receipts)
    states = {item.key: item.state for item in updated.obligations}

    assert states[f"oracle:{SELECTOR}"] is obligations.ObligationState.UNKNOWN
    assert states["contract:copy"] is obligations.ObligationState.SATISFIED


@pytest.mark.parametrize(
    ("result", "expected"),
    [("GREEN", obligations.ObligationState.SATISFIED),
     ("RED", obligations.ObligationState.MISSING)],
)
def test_test_intent_still_updates_oracle(tmp_path, result, expected):
    bolt = _bolt(tmp_path)
    bolt.receipt("test", SELECTOR, SELECTOR, result)

    assert _oracle_state(bolt._receipts) is expected


def test_restored_done_receipt_does_not_pass_real_inworld_gate(tmp_path):
    from probes import react_repair

    stored = receipt_history.receipts(directory=tmp_path)
    bolt = _bolt(tmp_path, stored)
    bolt.receipt("done", SELECTOR, SELECTOR, "ACCEPTED", "edits=1 last_exit=0")
    saved = [pickle.dumps(receipt_history._raw(row), protocol=5) for row in bolt._receipts]
    restored = receipt_history._restore_receipts(saved)

    accepted, reason = react_repair.inworld_obligation_gate(
        {"task_id": "intent-kind"}, [SELECTOR], restored
    )

    assert not accepted
    assert "oracle:" + SELECTOR in reason
    assert "UNKNOWN" in reason
