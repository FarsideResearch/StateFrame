"""The parallel join carries the final-check receipt and labels NOT_RUN only when nothing ran."""
from __future__ import annotations

import json

import stateframe.engine.kernel_parallel as parallel
from stateframe.engine.kernel_parallel import execute_parallel, render_parallel_disposition
from stateframe.engine.kernel_parallel_records import CheckReceipt, ParallelResult

from test_parallel_dispatch import ScriptedHost, restore_snapshot_permissions, setup_context, unit  # noqa: F401


def _receipt(spec, evidence_dir, status, reason=None):
    return CheckReceipt(spec.contract.check.digest(), "product", "harness", "environment",
                        spec.policy_digest, status, reason, (), str(evidence_dir), status)


def test_final_check_receipt_survives_join_into_written_result_and_disposition(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    parent_spec = dict(parallel._service[1], test_cmd="python -m pytest -q")
    monkeypatch.setattr(parallel, "_service", (run, parent_spec))
    monkeypatch.setattr(parallel, "_check", lambda spec, candidate, evidence_dir, *, test_python:
                        _receipt(spec, evidence_dir, "PASS"))
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})
    call_root = run / "parallel" / "parallel_1"

    result = execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=call_root)

    assert (result.final_check, result.integration) == ("PASS", "VERIFIED")
    receipt = result.final_check_receipt
    assert receipt is not None and receipt.status == "PASS"
    written = ParallelResult.from_json(json.loads((call_root / "result.json").read_bytes()))
    assert written.final_check_receipt == receipt
    request = parallel._request({"subframes": [unit("a", "a.py"), unit("b", "b.py")]}, "parallel_1", "2")
    section = render_parallel_disposition(result, request)["sections"]["disposition"]
    assert section["final_check"] == "PASS"
    assert section["final_check_receipt"]["status"] == "PASS"
    assert section["final_check_receipt"]["receipt_digest"] == receipt.digest()


def test_failed_composition_check_keeps_its_status_on_the_unit_outcome(tmp_path, monkeypatch):
    run, root, context = setup_context(
        tmp_path, monkeypatch,
        files={"a.py": "VALUE = 1\n", "b.py": "OTHER = 1\n",
               "tests/test_a.py": "assert False, 'planted failing control'\n"})
    monkeypatch.setattr(parallel, "_check", lambda spec, candidate, evidence_dir, *, test_python:
                        _receipt(spec, evidence_dir, "FAIL", "planted failing control"))
    checked = unit("a", "a.py")
    checked["check"] = {"tests": ["tests/test_a.py"]}
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})

    result = execute_parallel(context, {"subframes": [checked, unit("b", "b.py")]}, host=host,
                              store_root=run / "parallel" / "parallel_1")

    outcome = next(item for item in result.outcomes if item.unit_id == "a")
    assert (outcome.status, outcome.reason, outcome.check) == (
        "REFUSED", "COMPOSITION_CHECK_FAILED", "FAIL")
    assert result.integration == "FAILED"
    assert root.joinpath("a.py").read_text() == "VALUE = 1\n"


def test_committed_work_without_any_final_check_is_unverified_not_not_run(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    parent_spec = dict(parallel._service[1], test_cmd=None)
    monkeypatch.setattr(parallel, "_service", (run, parent_spec))
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})

    result = execute_parallel(context, {"subframes": [unit("a", "a.py"), unit("b", "b.py")]},
                              host=host, store_root=run / "parallel" / "parallel_1")

    assert all(item.status == "COMMITTED_UNVERIFIED" for item in result.outcomes)
    assert (result.final_check, result.integration) == ("NOT_RUN", "UNVERIFIED")
    assert "Final check: NOT_RUN; integration: UNVERIFIED" in result.observation
    assert result.final_check_receipt is None


def test_legacy_parallel_result_without_receipt_still_reads(tmp_path, monkeypatch):
    run, root, context = setup_context(tmp_path, monkeypatch)
    parent_spec = dict(parallel._service[1], test_cmd=None)
    monkeypatch.setattr(parallel, "_service", (run, parent_spec))
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}})
    result = execute_parallel(context, {"subframes": [unit("a", "a.py")]}, host=host,
                              store_root=run / "parallel" / "parallel_1")
    legacy = result.to_json()
    legacy.pop("final_check_receipt", None)
    legacy["schema"] = "stateframe-parallel-result/1"

    restored = ParallelResult.from_json(legacy)

    assert restored.final_check_receipt is None
    assert restored.final_check == result.final_check
