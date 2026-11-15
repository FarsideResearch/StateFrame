"""Regression proofs from the three recorded live subframe calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateframe.engine.kernel_parallel import (_planner_clauses, _request, _resource_admission,
                                               execute_parallel)
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot, admit
from stateframe.engine.kernel_parallel_contracts import verify_contracts
from stateframe.engine.kernel_parallel_projection import project_unit
from stateframe.engine.kernel_parallel_records import ParallelLimits
from test_parallel_dispatch import ScriptedHost, setup_context, unit


FIXTURES = Path(__file__).parent / "fixtures/parallel"
EXAMPLE = Path(__file__).parents[1] / "examples/parallel_fixes"


@pytest.fixture(autouse=True)
def restore_generated_snapshot_permissions(tmp_path):
    yield
    for manifest in tmp_path.rglob("snapshot-manifest.json"):
        snapshot = manifest.parent
        if "snapshots" not in snapshot.parts:
            continue
        for entry in snapshot.rglob("*"):
            entry.chmod(0o755 if entry.is_dir() else 0o644)
        snapshot.chmod(0o755)


def recorded(number: int) -> dict:
    return json.loads((FIXTURES / f"live2_{number:04d}.json").read_bytes())


@pytest.mark.parametrize("number", (2, 3))
def test_recorded_call_admits_and_projects_tax_dependency(tmp_path, number):
    import shutil

    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = recorded(number)
    request = _request(args, f"recorded-{number}", "2")
    snapshot = DirectorySnapshot(tmp_path / "repo", 0)
    clauses = _planner_clauses(args, ())
    admission = admit(request, snapshot, clauses, ParallelLimits(),
                      protected=lambda _: False, case_insensitive=False)
    assert admission.status == "ADMITTED", admission.refusal
    assert [(edge.src_unit, edge.dst_unit, edge.source) for edge in admission.edges] == [
        ("fix_tax", "fix_checkout", "declared")]
    assert admission.order.index("fix_tax") < admission.order.index("fix_checkout")
    spec = project_unit(admission, request, "fix_checkout", snapshot, parent_seed="seed",
                        policy_digest="policy", failure_projection="", requirements=clauses)
    assert [(interface.export_id, interface.path, interface.symbol) for interface in spec.interfaces] == [
        ("tax_export", "pricing/tax.py", "calculate_tax")]


def test_recorded_third_call_resolves_tax_export_to_existing_definition(tmp_path):
    import shutil

    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = recorded(5)
    assert "fix_tax" not in {item["id"] for item in args["subframes"]}
    from stateframe.engine.kernel_parallel import _complete_arguments

    snapshot = DirectorySnapshot(tmp_path / "repo", 0)
    completed, actions = _complete_arguments(args, snapshot)
    request = _request(completed, "recorded-5", "2")
    admission = admit(request, snapshot, _planner_clauses(completed, ()),
                      ParallelLimits(), protected=lambda _: False, case_insensitive=False)
    assert (admission.status, admission.edges) == ("ADMITTED", ())
    assert next(unit for unit in request.units if unit.id == "fix_checkout").consumes[0].to_json() == {
        "export_id": None, "path": "pricing/tax.py", "symbol": "calculate_tax"}
    assert any(action["reason_code"] == "UNIQUE_EXISTING_DEFINITION" for action in actions)


def test_duplicate_export_identity_names_both_claimants(tmp_path):
    (tmp_path / "a.py").write_text("def f(): return 1\n")
    (tmp_path / "b.py").write_text("VALUE = 1\n")
    args = {"subframes": [
        {"id": "a", "outcome": "first", "owns": ["a.py"], "creates": [],
         "exports": [{"id": "one", "path": "a.py", "symbol": "f"},
                     {"id": "two", "path": "a.py", "symbol": "f"}]},
        {"id": "b", "outcome": "second", "owns": ["b.py"], "creates": [],
         "consumes": [{"path": "a.py", "symbol": "f"}]},
    ]}
    request = _request(args, "call", "2")
    admission = admit(request, DirectorySnapshot(tmp_path, 0), (), ParallelLimits(),
                      protected=lambda _: False, case_insensitive=False)
    assert admission.status == "REFUSED"
    assert admission.refusal.reason == "DUPLICATE_EXPORT"
    assert "one" in admission.refusal.detail and "two" in admission.refusal.detail


@pytest.mark.parametrize("number", (2, 3))
def test_recorded_float_annotation_is_unverified_but_structure_is_checked(tmp_path, number):
    import shutil

    shutil.copytree(EXAMPLE, tmp_path / "repo")
    request = _request(recorded(number), "call", "2")
    snapshot = DirectorySnapshot(tmp_path / "repo", 0)
    tax = verify_contracts(request.units, snapshot, unit_id="fix_tax").verdicts[0]
    assert (tax.status, tax.reason, tax.expected.returns, tax.actual.returns) == (
        "PASS", "annotation_unverified", "float", None)
    source = tmp_path / "repo/pricing/tax.py"
    source.write_text("def calculate_tax(subtotal, rate, extra=0):\n    return subtotal * rate / 100\n")
    wrong = verify_contracts(request.units, DirectorySnapshot(tmp_path / "repo", 1),
                             unit_id="fix_tax").verdicts[0]
    assert (wrong.status, wrong.reason) == ("FAIL", "SIGNATURE_MISMATCH")


def test_guessed_parameter_annotation_is_ignored_only_when_source_is_unannotated(tmp_path):
    import copy
    import shutil

    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = copy.deepcopy(recorded(2))
    tax = next(unit for unit in args["subframes"] if unit["id"] == "fix_tax")
    tax["exports"][0]["signature"]["params"][0]["annotation"] = "float"
    request = _request(args, "call", "2")
    snapshot = DirectorySnapshot(tmp_path / "repo", 0)
    verdict, = verify_contracts(request.units, snapshot, unit_id="fix_tax").verdicts
    assert (verdict.status, verdict.reason, verdict.actual.params[0].annotation) == (
        "PASS", "annotation_unverified", None)

    (tmp_path / "repo/pricing/tax.py").write_text(
        "def calculate_tax(subtotal: str, rate):\n    return subtotal * rate\n")
    verdict, = verify_contracts(request.units, DirectorySnapshot(tmp_path / "repo", 1),
                                unit_id="fix_tax").verdicts
    assert (verdict.status, verdict.reason, verdict.actual.params[0].annotation) == (
        "FAIL", "SIGNATURE_MISMATCH", "str")


def test_default_cap_uses_available_parent_requests_and_preserves_explicit_setting(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel

    names = ("a", "b", "c", "d", "e")
    run, _, context = setup_context(tmp_path, monkeypatch,
                                    files={f"{name}.py": f"VALUE_{name} = 1\n" for name in names})
    context.limits = ParallelLimits()
    parent_spec = dict(parallel._service[1], max_turns=40,
                       parallel_limits=context.limits.to_json())
    monkeypatch.setattr(parallel, "_service", (run, parent_spec))
    admitted = admit(_request({"subframes": [
            {"id": name, "outcome": name, "owns": [f"{name}.py"], "creates": []}
            for name in names
        ]}, "call", "2"), DirectorySnapshot(context.root, 0), (), context.limits,
            protected=lambda _: False, case_insensitive=False)
    assert admitted.status == "ADMITTED"
    chosen = _resource_admission(admitted, context, 5)
    assert (chosen.status, chosen.limits.unit_max_requests) == ("ADMITTED", 7)
    state = json.loads((run / "parallel/budget.json").read_bytes())
    assert (state["free"], state["parent_reserve"], state["parent_steps_spent"]) == (35, 4, 1)

    parent_spec["parallel_subframe_max_requests_explicit"] = True
    strict = _resource_admission(admitted, context, 5)
    assert (strict.status, strict.refusal.reason, strict.limits.unit_max_requests) == (
        "REFUSED", "RESOURCES_UNAVAILABLE", 12)
    assert "--parallel-subframe-max-requests" in strict.refusal.detail


def test_default_cap_below_three_refuses(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel

    names = ("a", "b", "c", "d", "e")
    run, _, context = setup_context(tmp_path, monkeypatch,
                                    files={f"{name}.py": f"VALUE_{name} = 1\n" for name in names})
    context.limits = ParallelLimits()
    monkeypatch.setattr(parallel, "_service", (run, dict(parallel._service[1], max_turns=14)))
    request = _request({"subframes": [
        {"id": name, "outcome": name, "owns": [f"{name}.py"], "creates": []}
        for name in names
    ]}, "call", "2")
    admission = admit(request, DirectorySnapshot(context.root, 0), (), context.limits,
                      protected=lambda _: False, case_insensitive=False)
    assert admission.status == "ADMITTED"
    result = _resource_admission(admission, context, 5)
    assert (result.status, result.refusal.reason) == ("REFUSED", "RESOURCES_UNAVAILABLE")
    assert "minimum 3" in result.refusal.detail


def test_chosen_cap_reaches_intent_spec_observation_and_journal(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel

    names = ("a", "b", "c", "d", "e")
    run, _, context = setup_context(tmp_path, monkeypatch,
                                    files={f"{name}.py": "VALUE = 0\n" for name in names})
    context.limits = ParallelLimits()
    monkeypatch.setattr(parallel, "_service", (run, dict(parallel._service[1], max_turns=40)))
    host = ScriptedHost({name: {f"{name}.py": "VALUE = 1\n"} for name in names})
    result = execute_parallel(context, {"subframes": [unit(name, f"{name}.py") for name in names]},
                              host=host, store_root=run / "parallel/parallel_1")
    assert result.admission.status == "ADMITTED"
    assert result.admission.limits.unit_max_requests == 7
    assert result.txn_order == (1, 2, 3, 4, 5)
    assert "subframe_request_cap=7" in result.observation
    intent = json.loads((run / "parallel/parallel_1/intent.json").read_bytes())
    assert intent["reservations"] == {name: 7 for name in names}
    assert [(event[1]["status"], event[1]["subframe_request_cap"])
            for event in context.runtime.journal.events if event[0] == "parallel_admission"] == [
                ("ADMITTED", 7)]
    for name in names:
        spec = json.loads((run / "parallel/parallel_1/units" / name / "unit-spec.json").read_bytes())
        assert spec["limits"]["unit_max_requests"] == 7


def test_existing_create_collision_is_pinned_and_journaled(tmp_path, monkeypatch):
    import stateframe.engine.kernel_parallel as parallel

    run, root, context = setup_context(
        tmp_path, monkeypatch, files={"a.py": "VALUE = 0\n", "b.py": "OTHER = 0\n"})
    context.limits = ParallelLimits()
    monkeypatch.setattr(parallel, "_service", (run, dict(parallel._service[1], max_turns=40)))
    create_existing = unit("a", "a.py")
    create_existing["owns"] = []
    create_existing["creates"] = ["a.py"]
    host = ScriptedHost({"a": {"a.py": "VALUE = 1\n"},
                         "b": {"b.py": "OTHER = 1\n"}})

    result = execute_parallel(
        context, {"subframes": [create_existing, unit("b", "b.py")]}, host=host,
        store_root=run / "parallel/parallel_1")

    assert result.admission.status == "REFUSED"
    assert result.admission.refusal.reason == "CREATE_EXISTS"
    assert result.admission.refusal.paths == ("a.py",)
    assert result.admission.adjustments == ()
    intent = json.loads((run / "parallel/parallel_1/intent.json").read_bytes())
    assert intent["admission"]["refusal"]["reason"] == "CREATE_EXISTS"
    event, = [fields for kind, fields in context.runtime.journal.events
              if kind == "parallel_admission"]
    assert event["status"] == "REFUSED"
    assert event["reason"] == "CREATE_EXISTS"
