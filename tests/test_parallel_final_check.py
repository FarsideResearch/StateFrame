"""Integration labels keep unknown coupling separate from the check result."""
from __future__ import annotations

from types import SimpleNamespace

from stateframe.engine import kernel_parallel
from stateframe.engine.kernel_parallel_contracts import canonical_signature
from stateframe.engine.kernel_parallel_records import (
    CheckReceipt, CheckRecipe, CouplingPair, ParallelRequest, UnitContract,
)
from test_parallel_dispatch import ScriptedHost, restore_snapshot_permissions, setup_context, unit  # noqa: F401


def test_unknown_coupling_without_a_final_check_is_unverified(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("A = 1\n")
    (tmp_path / "b.py").write_text("B = 1\n")
    units = tuple(UnitContract(unit_id, "update", (f"{unit_id}.py",), (), (), (), (), (),
                               CheckRecipe((), (), "none")) for unit_id in ("a", "b"))
    request = ParallelRequest("call", units, "2")
    call_root = tmp_path / "call"
    call_root.mkdir()
    monkeypatch.setattr(kernel_parallel, "_service", (object(), {"test_cmd": None}))
    context = SimpleNamespace(root=tmp_path)
    unknown = (CouplingPair("a", "b", "UNKNOWN", "dynamic config", ()),)

    assert kernel_parallel._final_check(context, request, call_root, unknown) == (
        "NOT_RUN", "UNVERIFIED")
    assert kernel_parallel._final_check(context, request, call_root, ()) == (
        "NOT_RUN", "NOT_RUN")


def test_cancelled_first_subframe_and_refused_producer_return_observation(tmp_path, monkeypatch):
    run, root, context = setup_context(
        tmp_path, monkeypatch,
        files={"a.py": "VALUE = 1\n", "b.py": "def source(value):\n    return value\n"},
    )
    signature = canonical_signature(
        "def source(value):\n    return value\n", "source")
    producer = unit("b", "b.py", exports=[{
        "id": "source", "path": "b.py", "symbol": "source",
        "signature": signature.to_json(),
    }])
    consumer = unit("a", "a.py", consumes=[{"export_id": "source"}])
    host = ScriptedHost({}, terminal={"b": "REFUSED"})
    parent_spec = dict(kernel_parallel._service[1], test_cmd="python -m pytest -q")
    monkeypatch.setattr(kernel_parallel, "_service", (run, parent_spec))

    result = kernel_parallel.execute_parallel(
        context, {"subframes": [consumer, producer]}, host=host,
        store_root=run / "parallel" / "parallel_1",
    )

    assert result.admission.status == "ADMITTED"
    assert [(item.unit_id, item.status, item.reason) for item in result.outcomes] == [
        ("a", "CANCELLED", "PRODUCER_FAILED"), ("b", "REFUSED", "REFUSED"),
    ]
    assert (result.final_check, result.integration, result.txn_order) == (
        "NOT_RUN", "FAILED", ())
    assert "Final check: NOT_RUN; integration: FAILED" in result.observation
    assert not (run / "parallel" / "parallel_1" / "units" / "a" / "unit-spec.json").exists()
    assert root.joinpath("b.py").read_text() == "def source(value):\n    return value\n"


def test_final_check_uses_started_survivor_when_first_subframe_never_started(tmp_path, monkeypatch):
    run, root, context = setup_context(
        tmp_path, monkeypatch,
        files={"a.py": "VALUE = 1\n", "b.py": "def source(value):\n    return value\n",
               "c.py": "OTHER = 1\n"},
    )
    signature = canonical_signature("def source(value):\n    return value\n", "source")
    producer = unit("b", "b.py", exports=[{
        "id": "source", "path": "b.py", "symbol": "source",
        "signature": signature.to_json(),
    }])
    consumer = unit("a", "a.py", consumes=[{"export_id": "source"}])
    independent = unit("c", "c.py")
    parent_spec = dict(kernel_parallel._service[1], test_cmd="python -m pytest -q")
    monkeypatch.setattr(kernel_parallel, "_service", (run, parent_spec))
    checked = []

    def check(spec, candidate, evidence_dir, *, test_python):
        checked.append((spec.unit_id, spec.contract.check.command_kind,
                        spec.contract.check.selectors, candidate.joinpath("c.py").read_text()))
        return CheckReceipt(spec.contract.check.digest(), "product", "harness", "environment",
                            spec.policy_digest, "PASS", None, (), str(evidence_dir), "1 passed")

    monkeypatch.setattr(kernel_parallel, "_check", check)
    host = ScriptedHost({"c": {"c.py": "OTHER = 2\n"}}, terminal={"b": "REFUSED"})
    result = kernel_parallel.execute_parallel(
        context, {"subframes": [consumer, producer, independent]}, host=host,
        store_root=run / "parallel" / "parallel_1",
    )

    assert [(item.unit_id, item.status) for item in result.outcomes] == [
        ("a", "CANCELLED"), ("b", "REFUSED"), ("c", "COMMITTED_UNVERIFIED")]
    assert checked == [("c", "run_check_command", ("python -m pytest -q",), "OTHER = 2\n")]
    assert (result.final_check, result.integration, result.txn_order) == ("PASS", "FAILED", (1,))
    assert "Final check: PASS; integration: FAILED" in result.observation
    assert root.joinpath("c.py").read_text() == "OTHER = 2\n"
