"""Every future live proof's recorded parallel plans belong in this table.

Fixtures preserve the model's tool arguments. Mutations exercise equivalent
forms that the host can complete without asking the model to plan again.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from stateframe.engine.kernel_parallel import (_bind_implicit_signatures, _complete_arguments,
                                               _observation, _planner_clauses, _request,
                                               ParallelInputError)
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot, admit
from stateframe.engine.kernel_parallel_records import ParallelLimits, ParallelMeasured
from stateframe.engine.kernel_request_diet import RequestTokenMeter


FIXTURES = Path(__file__).parent / "fixtures/parallel/live_plans"
EXAMPLE = Path(__file__).parents[1] / "examples/parallel_fixes"
PLANS = (
    ("live1_A_0002", "ADMITTED"),
    ("live1_B_0002", "ADMITTED"),
    ("live2_A_0002", "ADMITTED"),
    ("live2_A_0003", "ADMITTED"),
    ("live2_A_0005", "ADMITTED"),
    ("live2_B_0002", "ADMITTED"),
    ("live2_B_0003", "ADMITTED"),
    ("live2_D_0002", "ADMITTED"),
)


def _plan(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_bytes())


def _admission(args: dict, repo: Path):
    snapshot = DirectorySnapshot(repo, 0)
    completed, actions = _complete_arguments(args, snapshot)
    request = _request(completed, "live-plan", "2")
    admission = admit(request, snapshot, _planner_clauses(completed, ()), ParallelLimits(),
                      protected=lambda _: False, case_insensitive=False)
    return completed, actions, request, admission


@pytest.mark.parametrize(("name", "expected"), PLANS)
def test_recorded_a3b_plan(tmp_path, name, expected):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan(name)
    completed, actions, _, admission = _admission(args, tmp_path / "repo")
    assert admission.status == expected, admission.refusal
    assert [(edge.src_unit, edge.dst_unit) for edge in admission.edges] == (
        [("fix_tax", "fix_checkout")] if {"fix_tax", "fix_checkout"} <= {
            unit["id"] for unit in args["subframes"]}
        else [])
    if name == "live1_B_0002":
        assert any(action["reason_code"] == "REDUNDANT_OWNED_READ" for action in actions)
    if name == "live2_B_0002":
        assert any(action["reason_code"] == "OPTIONAL_FIELD_DEFAULTED" for action in actions)
    if name == "live2_A_0005":
        checkout = next(unit for unit in completed["subframes"] if unit["id"] == "fix_checkout")
        assert checkout["consumes"] == [{"path": "pricing/tax.py", "symbol": "calculate_tax"}]
        assert any(action["subframe"] == "fix_checkout" and
                   action["reason_code"] == "UNIQUE_EXISTING_DEFINITION" for action in actions)
    if name == "live2_D_0002":
        checkout = next(unit for unit in completed["subframes"] if unit["id"] == "fix_checkout")
        assert checkout["consumes"] == [{"export_id": "calculate_tax", "path": "pricing/tax.py",
                                         "symbol": "calculate_tax"}]
        assert any(action["subframe"] == "fix_checkout" and action["producer"] == "fix_tax"
                   and action["action"] == "linked_coupling" and
                   action["reason_code"] == "UNIQUE_COUPLING_OWNER" for action in actions)


def test_ambiguous_imported_file_owners_name_both_claimants(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan("live2_D_0002")
    discount = next(unit for unit in args["subframes"] if unit["id"] == "fix_discount")
    discount["owns"].append("pricing/tax.py")
    with pytest.raises(ParallelInputError) as error:
        _admission(args, tmp_path / "repo")
    assert all(text in str(error.value) for text in (
        "fix_checkout", "pricing/tax.py", "fix_tax", "fix_discount", "owns", "fix"))


def test_undeclared_import_projects_existing_interface_without_declared_export(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan("live2_D_0002")
    tax = next(unit for unit in args["subframes"] if unit["id"] == "fix_tax")
    tax["exports"] = []
    completed, actions, _, admission = _admission(args, tmp_path / "repo")
    assert admission.status == "ADMITTED", admission.refusal
    export, = next(unit for unit in completed["subframes"] if unit["id"] == "fix_tax")["exports"]
    assert (export["path"], export["symbol"]) == ("pricing/tax.py", "calculate_tax")
    checkout = next(unit for unit in completed["subframes"] if unit["id"] == "fix_checkout")
    assert checkout["consumes"] == [{"export_id": export["id"], "path": "pricing/tax.py",
                                     "symbol": "calculate_tax"}]
    assert [(edge.src_unit, edge.dst_unit) for edge in admission.edges] == [
        ("fix_tax", "fix_checkout")]
    assert any(action["reason_code"] == "OWNED_SYMBOL_EXPORTED" for action in actions)


def test_all_imported_symbols_from_one_owner_are_linked(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    repo = tmp_path / "repo"
    (repo / "pricing/tax.py").write_text(
        (repo / "pricing/tax.py").read_text() + "\ndef tax_helper(value):\n    return value\n")
    (repo / "pricing/checkout.py").write_text(
        (repo / "pricing/checkout.py").read_text() +
        "\nfrom pricing.tax import tax_helper\n")
    completed, _, _, admission = _admission(_plan("live2_D_0002"), repo)
    assert admission.status == "ADMITTED", admission.refusal
    checkout = next(unit for unit in completed["subframes"] if unit["id"] == "fix_checkout")
    assert {(item["path"], item["symbol"]) for item in checkout["consumes"]} == {
        ("pricing/tax.py", "calculate_tax"), ("pricing/tax.py", "tax_helper")}


def test_coupling_completion_covers_every_owned_consumer_file(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    repo = tmp_path / "repo"
    (repo / "pricing/tax.py").write_text(
        (repo / "pricing/tax.py").read_text() + "\ndef tax_helper(value):\n    return value\n")
    (repo / "pricing/other_checkout.py").write_text("from .tax import tax_helper\n")
    args = _plan("live2_D_0002")
    checkout = next(unit for unit in args["subframes"] if unit["id"] == "fix_checkout")
    checkout["owns"].append("pricing/other_checkout.py")
    completed, _, _, admission = _admission(args, repo)
    assert admission.status == "ADMITTED", admission.refusal
    consumer = next(unit for unit in completed["subframes"] if unit["id"] == "fix_checkout")
    assert {(item["path"], item["symbol"]) for item in consumer["consumes"]} == {
        ("pricing/tax.py", "calculate_tax"), ("pricing/tax.py", "tax_helper")}


@pytest.mark.parametrize("import_line", ("import pricing.tax", "from . import tax"))
def test_module_import_projects_current_public_interface(tmp_path, import_line):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    repo = tmp_path / "repo"
    checkout = repo / "pricing/checkout.py"
    checkout.write_text(checkout.read_text().replace(
        "from .tax import calculate_tax", import_line))
    completed, _, _, admission = _admission(_plan("live2_D_0002"), repo)
    assert admission.status == "ADMITTED", admission.refusal
    consumer = next(unit for unit in completed["subframes"] if unit["id"] == "fix_checkout")
    assert {(item["path"], item["symbol"]) for item in consumer["consumes"]} == {
        ("pricing/tax.py", "calculate_tax")}


def test_unprojectable_import_names_exact_contract_fix(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    repo = tmp_path / "repo"
    (repo / "pricing/tax.py").write_text("calculate_tax = factory()\n")
    with pytest.raises(ParallelInputError) as error:
        _admission(_plan("live2_D_0002"), repo)
    assert all(text in str(error.value) for text in (
        "fix_checkout", "pricing/tax.py::calculate_tax", "fix_tax", "fix"))


@pytest.mark.parametrize("field", ("creates", "consumes", "reads", "exports", "requirements", "check"))
def test_optional_fields_default_to_empty(tmp_path, field):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan("live2_A_0002")
    for unit in args["subframes"]:
        if field == "consumes" and unit["id"] == "fix_checkout":
            continue
        unit.pop(field, None)
    _, actions, _, admission = _admission(args, tmp_path / "repo")
    assert admission.status == "ADMITTED", admission.refusal
    assert {action["field"] for action in actions if action["reason_code"] == "OPTIONAL_FIELD_DEFAULTED"} == {field}


@pytest.mark.parametrize("form", ("path_symbol", "export_id", "all_three", "path_only", "symbol_only", "owned_symbol"))
def test_consume_forms_complete_to_same_edge(tmp_path, form):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan("live2_A_0002")
    checkout = next(unit for unit in args["subframes"] if unit["id"] == "fix_checkout")
    forms = {
        "path_symbol": {"path": "pricing/tax.py", "symbol": "calculate_tax"},
        "export_id": {"export_id": "tax_export"},
        "all_three": {"export_id": "tax_export", "path": "pricing/tax.py", "symbol": "calculate_tax"},
        "path_only": {"path": "pricing/tax.py"},
        "symbol_only": {"symbol": "calculate_tax"},
        "owned_symbol": {"path": "pricing/tax.py", "symbol": "calculate_tax"},
    }
    checkout["consumes"] = [forms[form]]
    if form == "owned_symbol":
        next(unit for unit in args["subframes"] if unit["id"] == "fix_tax")["exports"] = []
    completed, actions, _, admission = _admission(args, tmp_path / "repo")
    assert admission.status == "ADMITTED", admission.refusal
    assert [(edge.src_unit, edge.dst_unit) for edge in admission.edges] == [("fix_tax", "fix_checkout")]
    consume, = next(unit for unit in completed["subframes"] if unit["id"] == "fix_checkout")["consumes"]
    assert consume["export_id"]
    assert consume["path"] == "pricing/tax.py"
    assert consume["symbol"] == "calculate_tax"
    assert any(action["action"] == "linked_consume" and action["producer"] == "fix_tax"
               for action in actions)


def test_reorder_requirements_missing_signature_and_duplicate_paths(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan("live2_A_0002")
    args["subframes"].reverse()
    tax = next(unit for unit in args["subframes"] if unit["id"] == "fix_tax")
    tax["exports"][0].pop("signature")
    tax["owns"].append("pricing/tax.py")
    tax["reads"].extend(("pricing/tax.py", "tests/test_tax.py"))
    tax["requirements"] = ["Tax on 100 at 10 percent must be 10."]
    completed, actions, request, admission = _admission(args, tmp_path / "repo")
    assert admission.status == "ADMITTED", admission.refusal
    assert admission.order.index("fix_tax") < admission.order.index("fix_checkout")
    assert next(unit for unit in request.units if unit.id == "fix_tax").requirement_ids
    assert next(unit for unit in completed["subframes"] if unit["id"] == "fix_tax")["owns"] == ["pricing/tax.py"]
    assert {action["reason_code"] for action in actions} >= {
        "SIGNATURE_INFERRED", "DUPLICATE_PATH", "REDUNDANT_OWNED_READ"}


def test_ambiguous_and_unresolved_consume_name_field_and_corrected_json(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan("live2_A_0002")
    args["subframes"][1]["exports"].append({
        "id": "other_tax", "path": "pricing/discount.py", "symbol": "calculate_tax"})
    checkout = next(unit for unit in args["subframes"] if unit["id"] == "fix_checkout")
    checkout["consumes"] = [{"symbol": "calculate_tax"}]
    with pytest.raises(ParallelInputError) as error:
        _admission(args, tmp_path / "repo")
    assert all(text in str(error.value) for text in (
        "consumes[0]", "tax_export", "other_tax", '"export_id"', "fix"))

    missing = copy.deepcopy(args)
    missing["subframes"][1]["exports"].pop()
    checkout = next(unit for unit in missing["subframes"] if unit["id"] == "fix_checkout")
    checkout["consumes"] = [{"symbol": "not_a_symbol"}]
    with pytest.raises(ParallelInputError) as error:
        _admission(missing, tmp_path / "repo")
    assert all(text in str(error.value) for text in (
        "consumes[0]", "not_a_symbol", '"consumes": []', "fix"))


def test_existing_definition_partial_consume_and_duplicate_consume(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    (tmp_path / "repo/helpers.py").write_text("def helper(value):\n    return value\n")
    args = _plan("live2_B_0003")
    discount = next(unit for unit in args["subframes"] if unit["id"] == "fix_discount")
    discount["consumes"] = [{"path": "helpers.py"}, {"path": "helpers.py", "symbol": "helper"}]
    completed, actions, _, admission = _admission(args, tmp_path / "repo")
    assert admission.status == "ADMITTED", admission.refusal
    assert next(unit for unit in completed["subframes"] if unit["id"] == "fix_discount")["consumes"] == [
        {"path": "helpers.py", "symbol": "helper"}]
    assert {action["reason_code"] for action in actions} >= {
        "UNIQUE_EXISTING_DEFINITION", "DUPLICATE_CONSUME"}


def test_inconsistent_redundant_consume_refuses_with_exact_correction(tmp_path):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    args = _plan("live2_A_0002")
    checkout = next(unit for unit in args["subframes"] if unit["id"] == "fix_checkout")
    checkout["consumes"] = [{"export_id": "tax_export", "path": "pricing/quantity.py",
                             "symbol": "calculate_tax"}]
    _, _, request, admission = _admission(args, tmp_path / "repo")
    assert (admission.status, admission.refusal.reason) == ("REFUSED", "UNRESOLVED_CONSUME")
    message = _observation(admission, (), "NOT_RUN", "NOT_RUN",
                           ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0), request)
    assert all(text in message for text in (
        "fix_checkout", "consumes[0]", "tax_export", '"path": "pricing/tax.py"', "fix"))
    assert "Plan fix_checkout: owns=pricing/checkout.py; check=tests/test_checkout.py" in message


@pytest.mark.parametrize("created", (False, True))
def test_unique_owner_can_supply_a_new_symbol(tmp_path, created):
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_a.py").write_text(
        "from a import future\n\ndef test_future():\n    assert future(2) == 3\n")
    if not created:
        (tmp_path / "a.py").write_text("VALUE = 1\n")
    args = {"subframes": [
        {"id": "a", "outcome": "Provide future", "owns": [] if created else ["a.py"],
         "creates": ["a.py"] if created else [], "check": {"tests": ["tests/test_a.py"]}},
        {"id": "b", "outcome": "Use future", "owns": ["b.py"],
         "consumes": [{"path": "a.py", "symbol": "future"}]},
    ]}
    completed, actions, _, admission = _admission(args, tmp_path)
    assert admission.status == "ADMITTED", admission.refusal
    assert [(edge.src_unit, edge.dst_unit) for edge in admission.edges] == [("a", "b")]
    export, = completed["subframes"][0]["exports"]
    assert (export["path"], export["symbol"]) == ("a.py", "future")
    assert completed["subframes"][1]["consumes"] == [
        {"export_id": export["id"], "path": "a.py", "symbol": "future"}]
    assert {action["reason_code"] for action in actions} >= {
        "OWNED_SYMBOL_EXPORTED", "UNIQUE_PRODUCER"}


def test_export_style_name_links_to_unique_owner_when_export_is_omitted(tmp_path):
    (tmp_path / "a.py").write_text("def future(value):\n    return value + 1\n")
    (tmp_path / "b.py").write_text("VALUE = 1\n")
    args = {"subframes": [
        {"id": "a", "outcome": "Update future", "owns": ["a.py"]},
        {"id": "b", "outcome": "Use future", "owns": ["b.py"],
         "consumes": [{"export_id": "a_export"}]},
    ]}
    completed, actions, _, admission = _admission(args, tmp_path)
    assert admission.status == "ADMITTED", admission.refusal
    assert [(edge.src_unit, edge.dst_unit) for edge in admission.edges] == [("a", "b")]
    export, = completed["subframes"][0]["exports"]
    assert completed["subframes"][1]["consumes"] == [
        {"export_id": export["id"], "path": "a.py", "symbol": "future"}]
    assert any(action["action"] == "linked_consume" and action["producer"] == "a"
               for action in actions)


def _terse_plan(name: str) -> dict:
    full = _plan(name)
    return {"subframes": [
        {"id": unit["id"], "goal": unit["outcome"], "owns": unit["owns"]}
        for unit in full["subframes"]]}


@pytest.mark.parametrize(("name", "expected"), PLANS)
def test_terse_live_plan_keeps_checks_ownership_and_edges(tmp_path, name, expected):
    shutil.copytree(EXAMPLE, tmp_path / "repo")
    full, _, _, full_admission = _admission(_plan(name), tmp_path / "repo")
    terse, actions, request, admission = _admission(_terse_plan(name), tmp_path / "repo")
    assert (admission.status, full_admission.status) == (expected, expected)
    assert [(edge.src_unit, edge.dst_unit) for edge in admission.edges] == [
        (edge.src_unit, edge.dst_unit) for edge in full_admission.edges]
    assert [(unit["id"], unit["owns"], unit["check"]["tests"])
            for unit in terse["subframes"]] == [
        (unit["id"], unit["owns"], unit["check"]["tests"])
        for unit in full["subframes"]]
    assert all(unit["reads"] == unit["check"]["tests"] for unit in terse["subframes"])
    assert all(unit["outcome"] == unit["goal"] for unit in terse["subframes"])
    assert all(unit.requirement_ids for unit in request.units)
    expanded = {(action["subframe"], action["field"], action["source"], action["reason_code"])
                for action in actions if action["action"] == "expanded_field"}
    if any(unit.id == "fix_tax" for unit in request.units):
        assert ("fix_tax", "check", "tests", "STATIC_TEST_IMPORT") in expanded
    for unit in request.units:
        assert (unit.id, "outcome", "goal", "GOAL_SENTENCE") in expanded
        assert (unit.id, "requirements", "goal", "GOAL_PLANNER_CLAUSE") in expanded
        assert (unit.id, "reads", "check", "CHECK_SOURCE_PATHS") in expanded
    frozen = _bind_implicit_signatures(request, DirectorySnapshot(tmp_path / "repo", 0))
    tax = next((unit for unit in frozen.units if unit.id == "fix_tax"), None)
    if tax is not None:
        assert any(export.symbol == "calculate_tax" and export.signature is not None
                   for export in tax.exports)
    observation = _observation(admission, (), "NOT_RUN", "NOT_RUN",
                               ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0), request)
    for unit in request.units:
        assert (f"Plan {unit.id}: owns={','.join(unit.owns)}; "
                f"check={','.join(unit.check.selectors)}") in observation


def test_five_subframe_terse_plan_fits_completion_budget():
    plan = _terse_plan("live2_A_0002")
    body = {"messages": [{"role": "assistant", "content": "",
                          "tool_calls": [{"id": "plan", "type": "function", "function": {
                              "name": "parallel", "arguments": json.dumps(plan, separators=(",", ":"))}}]}],
            "tools": []}
    tokens, method = RequestTokenMeter(template_overhead_tokens=0).measure(body)
    assert method != "unavailable"
    assert tokens < 600


def test_terse_ambiguous_check_refuses_with_exact_fragment(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    (tmp_path / "b.py").write_text("def g():\n    return 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_first.py").write_text("from a import f\n")
    (tmp_path / "tests/test_second.py").write_text("from a import f\n")
    args = {"subframes": [{"id": "a", "goal": "Fix f", "owns": ["a.py"]},
                          {"id": "b", "goal": "Fix g", "owns": ["b.py"]}]}
    with pytest.raises(ParallelInputError) as error:
        _admission(args, tmp_path)
    assert error.value.field == "check"
    assert '"check": {"tests": ["tests/test_first.py"]}' in error.value.hint


def test_terse_no_matching_check_is_honest_not_run(tmp_path):
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    completed, _, request, admission = _admission({"subframes": [
        {"id": "a", "goal": "Change a", "owns": ["a.py"]},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]}]}, tmp_path)
    assert admission.status == "ADMITTED"
    assert [unit["check"] for unit in completed["subframes"]] == [{"tests": []}, {"tests": []}]
    assert [unit.check.command_kind for unit in request.units] == ["none", "none"]


def test_terse_named_check_fallback_when_no_test_imports_module(tmp_path):
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_a.py").write_text("def test_value():\n    assert 1 == 1\n")
    completed, actions, _, admission = _admission({"subframes": [
        {"id": "a", "goal": "Change a", "owns": ["a.py"]},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]}]}, tmp_path)
    assert admission.status == "ADMITTED"
    assert completed["subframes"][0]["check"] == {"tests": ["tests/test_a.py"]}
    assert completed["subframes"][1]["check"] == {"tests": []}
    assert any(action["subframe"] == "a" and action["field"] == "check"
               and action["reason_code"] == "MATCHING_TEST_NAME" for action in actions)


def test_importing_check_wins_over_matching_name_with_failing_control(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_a.py").write_text(
        "def test_planted_failure():\n    assert False\n")
    (tmp_path / "tests/test_feature.py").write_text(
        "from a import f\n\ndef test_f():\n    assert f() == 1\n")
    completed, _, _, admission = _admission({"subframes": [
        {"id": "a", "goal": "Fix f", "owns": ["a.py"]},
        {"id": "b", "goal": "Fix b", "owns": ["b.py"]}]}, tmp_path)
    assert admission.status == "ADMITTED"
    assert completed["subframes"][0]["check"] == {"tests": ["tests/test_feature.py"]}
    assert completed["subframes"][1]["check"] == {"tests": []}


@pytest.mark.parametrize("statement", ("import a", "from pkg import a", "from pkg.a import *"))
def test_terse_module_import_freezes_public_definitions(tmp_path, statement):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/__init__.py").write_text("")
    (tmp_path / "pkg/a.py").write_text("def public(value):\n    return value\n\ndef _private():\n    pass\n")
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_feature.py").write_text(statement + "\n")
    if statement == "import a":
        (tmp_path / "a.py").write_text("def public(value):\n    return value\n")
        owned = "a.py"
    else:
        owned = "pkg/a.py"
    completed, _, request, admission = _admission({"subframes": [
        {"id": "a", "goal": "Maintain public", "owns": [owned]},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]}]}, tmp_path)
    assert admission.status == "ADMITTED"
    assert completed["subframes"][0]["check"] == {"tests": ["tests/test_feature.py"]}
    assert [export["symbol"] for export in completed["subframes"][0]["exports"]] == ["public"]
    frozen = _bind_implicit_signatures(request, DirectorySnapshot(tmp_path, 0))
    assert frozen.units[0].exports[0].signature is not None
