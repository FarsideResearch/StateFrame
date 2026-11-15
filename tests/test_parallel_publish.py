"""Committed subframe bytes use the parent receipt and binding path."""
from __future__ import annotations

import json

import pytest

import stateframe.engine.kernel_parallel as parallel
from stateframe.engine.kernel_parallel import execute_parallel
from stateframe.engine.kernel_parallel_records import CheckReceipt
from stateframe.engine.kernel_parallel_txn import TxnStore
from stateframe.engine.kernel_parallel_contracts import canonical_signature
from stateframe.engine.kernel_live_projection import workspace_binding

from test_parallel_dispatch import ScriptedHost, restore_snapshot_permissions, setup_context, unit  # noqa: F401


def test_publish_records_exact_paths_and_retires_green(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    original_binding = workspace_binding(root, "disabled")
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})
    result = execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert result.txn_order == (1, 2)
    assert root.joinpath("a.py").read_text() == "VALUE = 2\n"
    assert root.joinpath("b.py").read_text() == "OTHER = 2\n"
    assert context.bolt.product_edits == 2
    assert context.bolt.last_test_exit is None
    assert context.bolt._last_green_revision is None
    effects = [kwargs for args, kwargs in context.bolt.receipts if args[0] == "shell_effect"]
    assert [item["changed_paths"] for item in effects] == [("a.py",), ("b.py",)]
    published = [record for record, state in TxnStore(run / "parallel" / "parallel_1" / "txn").records()
                 if state == "PUBLISHED"]
    assert [record.txn_id for record in published] == [1, 2]
    final_binding = workspace_binding(root, "disabled")
    assert published[0].pre_binding.product_revision == original_binding["product_revision"]
    assert published[0].pre_binding.harness_revision == original_binding["test_harness_revision"]
    assert published[0].post_binding == published[1].pre_binding
    assert published[1].post_binding.product_revision == final_binding["product_revision"]
    assert published[1].post_binding.harness_revision == final_binding["test_harness_revision"]
    saved = json.loads((run / "parallel" / "parallel_1" / "result.json").read_text())
    assert saved["observation"] == result.observation
    assert saved["measured"]["peak_active_workers"] == 2


def test_producer_failure_does_not_publish_its_bytes(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}},
                        terminal={"a": "BLOCKED"})
    result = execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert {item.unit_id: item.status for item in result.outcomes} == {
        "a": "BLOCKED", "b": "COMMITTED_UNVERIFIED"}
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"
    assert root.joinpath("b.py").read_text() == "OTHER = 2\n"
    assert result.txn_order == (1,)
    assert (result.final_check, result.integration) == ("NOT_RUN", "FAILED")


@pytest.mark.parametrize("check_status,expected_commits", [("FAIL", 1), ("PASS", 2)])
def test_composition_check_controls_publication_with_planted_failure(
        tmp_path, monkeypatch, check_status, expected_commits):
    run, root, context = setup_context(
        tmp_path, monkeypatch,
        files={"a.py": "VALUE = 1\n", "b.py": "OTHER = 1\n",
               "tests/test_a.py": "assert False, 'planted failing control'\n"})
    checked_candidates = []

    def frozen_check(spec, candidate, evidence_dir, *, test_python):
        checked_candidates.append((candidate.joinpath("a.py").read_text(),
                                   spec.contract.check.selectors))
        return CheckReceipt(spec.contract.check.digest(), "product", "harness", "environment",
                            spec.policy_digest, check_status,
                            "planted failing control" if check_status == "FAIL" else None,
                            (), str(evidence_dir), check_status)

    monkeypatch.setattr(parallel, "_check", frozen_check)
    checked = unit("a", "a.py")
    checked["check"] = {"tests": ["tests/test_a.py"]}
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})
    result = execute_parallel(context, {"subframes": [checked, unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")
    assert len(result.txn_order) == expected_commits
    assert ("VALUE = 2\n", ("tests/test_a.py",)) in checked_candidates
    assert root.joinpath("a.py").read_text() == ("VALUE = 1\n" if check_status == "FAIL"
                                                 else "VALUE = 2\n")
    assert root.joinpath("b.py").read_text() == "OTHER = 2\n"


def test_final_verifier_rechecks_surviving_export_on_final_manifest(tmp_path, monkeypatch):
    run, root, context = setup_context(
        tmp_path, monkeypatch,
        files={"a.py": "def source(x):\n    return x\n", "b.py": "OTHER = 1\n"})
    expected = canonical_signature("def source(x):\n    return x\n", "source")
    exported = unit("a", "a.py", exports=[{"id": "source", "path": "a.py",
                                            "symbol": "source", "signature": expected.to_json()}])
    request = parallel._request({"subframes": [exported, unit("b", "b.py")]}, "parallel_1", "2")
    # A later composition can invalidate an earlier export even if its own check passed.
    root.joinpath("a.py").write_text("def source(x, y):\n    return x + y\n")
    final_check, integration = parallel._final_check(
        context, request, run / "parallel" / "parallel_1", (), {"a", "b"})
    report = json.loads((run / "parallel" / "parallel_1" / "final-contracts.json").read_text())
    assert (final_check, integration) == ("NOT_RUN", "FAILED")
    assert report["a"]["verdicts"][0]["reason"] == "SIGNATURE_MISMATCH"
    assert report["b"]["ok"] is True


def test_refused_contract_report_is_recorded_and_rewritable(tmp_path):
    # final-contracts.json lists surviving commits only; a refused subframe keeps its verdicts here.
    class Report:
        def __init__(self, status):
            self.status = status

        def to_json(self):
            return {"ok": False, "verdicts": [{"status": self.status}]}

    parallel._record_refused_contracts(tmp_path, "api", Report("FAIL"))
    parallel._record_refused_contracts(tmp_path, "api", Report("UNKNOWN"))
    saved = json.loads((tmp_path / "refused-contracts" / "api.json").read_text())
    assert saved["verdicts"] == [{"status": "UNKNOWN"}]
