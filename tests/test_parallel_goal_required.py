"""The parallel schema requires goal while admission preserves the documented outcome alias."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from stateframe.engine.kernel_parallel import (
    _UNIT,
    PARALLEL_TOOL,
    ParallelInputError,
    _complete_arguments,
    _request,
    execute_parallel,
    normalize_goal_aliases,
    schema_input_error,
)
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot
from stateframe.engine.probes.react_repair import _contract_errors


def _repo(tmp_path):
    (tmp_path / "a.py").write_text("VALUE = 1\n")
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    return DirectorySnapshot(tmp_path, 0)


def _parse_expanded(args, snapshot):
    completed, _ = _complete_arguments(args, snapshot)
    return _request(completed, "call", "2")


def test_schema_advertises_goal_short_form_without_nested_combinators():
    # No anyOf/oneOf at the unit level: nested combinators broke llama.cpp grammar (0 parallel calls live).
    assert _UNIT["required"] == ["id", "goal", "owns"]
    assert "anyOf" not in _UNIT and "oneOf" not in _UNIT
    assert "outcome" in _UNIT["properties"]
    assert "{id, goal, owns}" in PARALLEL_TOOL["function"]["description"]
    assert PARALLEL_TOOL["function"]["parameters"]["properties"]["subframes"]["items"] is _UNIT


def test_goal_only_units_complete_to_outcome(tmp_path):
    args = {"subframes": [
        {"id": "a", "goal": "Change a", "owns": ["a.py"]},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]},
    ]}
    request = _parse_expanded(args, _repo(tmp_path))
    outcomes = {unit.id: unit.outcome for unit in request.units}
    assert outcomes == {"a": "Change a", "b": "Change b"}


def test_legacy_full_contract_is_accepted_at_runtime_without_expansion():
    # The schema no longer advertises outcome; the runtime validator stays lenient for old contracts.
    args = {"subframes": [
        {"id": "a", "outcome": "Change a", "owns": ["a.py"], "creates": []},
        {"id": "b", "outcome": "Change b", "owns": ["b.py"], "creates": ["c.py"]},
    ]}
    request = _request(args, "call", "2")
    assert {unit.id: unit.outcome for unit in request.units} == {"a": "Change a", "b": "Change b"}


def test_legacy_full_contract_is_accepted_at_runtime_through_completion(tmp_path):
    args = {"subframes": [
        {"id": "a", "outcome": "Change a", "owns": ["a.py"], "creates": []},
        {"id": "b", "outcome": "Change b", "owns": ["b.py"], "creates": []},
    ]}
    request = _parse_expanded(args, _repo(tmp_path))
    assert {unit.id: unit.outcome for unit in request.units} == {"a": "Change a", "b": "Change b"}


def test_outcome_alias_passes_schema_and_missing_goal_is_refused(tmp_path):
    args = {"subframes": [
        {"id": "a", "outcome": "Change a", "owns": ["a.py"], "creates": []},
        {"id": "b", "outcome": "Change b", "owns": ["b.py"], "creates": []},
    ]}
    schema_errors = _contract_errors((PARALLEL_TOOL,), "parallel", args)
    assert not schema_errors, "documented outcome alias was rejected: " + "; ".join(schema_errors)

    from stateframe.engine.kernel_parallel import normalize_goal_aliases
    normalized, expansions = normalize_goal_aliases(args)
    assert [unit["goal"] for unit in normalized["subframes"]] == ["Change a", "Change b"]
    assert all("outcome" not in unit for unit in normalized["subframes"])
    assert all("goal" not in unit for unit in args["subframes"])
    assert [(item["field"], item["action"], item["source"], item["reason_code"])
            for item in expansions] == [
                ("goal", "expanded_field", "outcome", "GOAL_SENTENCE"),
                ("goal", "expanded_field", "outcome", "GOAL_SENTENCE"),
            ]
    goal_args = {"subframes": [{"id": "a", "goal": "Change a", "owns": ["a.py"]}]}
    normalized_goal_args, goal_expansions = normalize_goal_aliases(goal_args)
    assert normalized_goal_args == goal_args
    assert not goal_expansions
    snapshot = _repo(tmp_path)
    completed, recorded_expansions = _complete_arguments(
        normalized, snapshot, input_expansions=expansions)
    assert recorded_expansions[:len(expansions)] == expansions
    request = _request(completed, "call", "2")
    assert {unit.id: unit.outcome for unit in request.units} == {
        "a": "Change a", "b": "Change b"}

    missing = {"subframes": [
        {"id": "a", "owns": ["a.py"], "creates": []},
        {"id": "b", "goal": "Change b", "owns": ["b.py"], "creates": []},
    ]}
    missing_errors = _contract_errors((PARALLEL_TOOL,), "parallel", missing)
    assert any("goal" in error for error in missing_errors)
    with pytest.raises(ParallelInputError) as refusal:
        _request(missing, "call", "2")
    assert refusal.value.field == "goal"


@pytest.mark.parametrize("invalid", ("", 3, None))
def test_invalid_outcome_alias_keeps_outcome_refusal_through_schema(invalid):
    args = {"subframes": [
        {"id": "a", "outcome": invalid, "owns": ["a.py"]},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]},
    ]}
    normalized, expansions = normalize_goal_aliases(args)
    assert normalized == args
    assert not expansions

    errors = _contract_errors((PARALLEL_TOOL,), "parallel", args)
    assert errors
    refusal = schema_input_error(args, errors)
    assert refusal.unit_id == "a"
    assert refusal.field == "outcome"
    assert refusal.hint == "provide outcome as text"


def test_mixed_plan_names_the_subframe_that_is_missing_its_goal():
    args = {"subframes": [
        {"id": "a", "outcome": "Change a", "owns": ["a.py"]},
        {"id": "b", "owns": ["b.py"]},
    ]}

    errors = _contract_errors((PARALLEL_TOOL,), "parallel", args)
    assert errors
    refusal = schema_input_error(args, errors)

    assert (refusal.unit_id, refusal.field) == ("b", "goal")
    assert refusal.hint == 'add "goal": ... and resend the whole parallel call'
    args["subframes"][1]["goal"] = "Change b"
    assert not _contract_errors((PARALLEL_TOOL,), "parallel", args)


def test_oversized_outcome_is_refused_by_the_executor(tmp_path, monkeypatch):
    from test_parallel_dispatch import ScriptedHost, setup_context

    run, _, context = setup_context(tmp_path, monkeypatch)
    args = {"subframes": [
        {"id": "a", "outcome": "x" * 301, "owns": ["a.py"], "creates": []},
        {"id": "b", "outcome": "Change b", "owns": ["b.py"], "creates": []},
    ]}
    host = ScriptedHost({"a": {"a.py": "VALUE = 2\n"}, "b": {"b.py": "OTHER = 2\n"}})

    result = execute_parallel(context, args, host=host, store_root=run / "parallel" / "parallel_1")

    assert result.admission.status == "REFUSED"
    assert result.admission.refusal.reason == "INVALID_ARGUMENTS"
    assert "subframe a outcome" in result.observation
    assert host.calls == []


def test_oversized_outcome_schema_refusal_names_the_supplied_field():
    args = {"subframes": [
        {"id": "a", "outcome": "x" * 301, "owns": ["a.py"]},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]},
    ]}

    errors = _contract_errors((PARALLEL_TOOL,), "parallel", args)
    assert errors
    refusal = schema_input_error(args, errors)

    assert (refusal.unit_id, refusal.field) == ("a", "outcome")
    assert "300 characters" in refusal.hint
    args["subframes"][0]["outcome"] = "x" * 300
    assert not _contract_errors((PARALLEL_TOOL,), "parallel", args)


def test_shipped_parallel_example_uses_goal_for_every_subframe():
    root = Path(__file__).resolve().parents[1]
    script = json.loads((root / "examples" / "parallel_fixes.script.json").read_text())
    parent_plan = next(step["arguments"] for step in script["parent"]
                       if step.get("tool") == "parallel")
    subframes = parent_plan["subframes"]
    assert subframes
    assert all("goal" in unit and "outcome" not in unit for unit in subframes)


def test_unit_without_goal_or_outcome_is_refused_on_goal(tmp_path):
    # Shape from the live run: the model sent id, owns, check and model_class only.
    args = {"subframes": [
        {"id": "tax_fix", "owns": ["a.py"], "model_class": "default"},
        {"id": "b", "goal": "Change b", "owns": ["b.py"]},
    ]}
    with pytest.raises(ParallelInputError) as refusal:
        _parse_expanded(args, _repo(tmp_path))
    assert refusal.value.unit_id == "tax_fix"
    assert refusal.value.field == "goal"
    assert refusal.value.hint == 'add "goal": "<one sentence>"'


def test_unit_without_goal_or_outcome_is_refused_on_goal_at_request_level():
    args = {"subframes": [{"id": "tax_fix", "owns": ["a.py"], "creates": []}]}
    with pytest.raises(ParallelInputError) as refusal:
        _request(args, "call", "2")
    assert refusal.value.field == "goal"
    assert refusal.value.hint == 'add "goal": "<one sentence>"'


def test_blank_outcome_without_goal_is_still_refused_on_outcome():
    args = {"subframes": [{"id": "a", "outcome": "   ", "owns": ["a.py"], "creates": []}]}
    with pytest.raises(ParallelInputError) as refusal:
        _request(args, "call", "2")
    assert refusal.value.field == "outcome"
    assert refusal.value.hint == "provide outcome as text"


def test_missing_id_is_still_refused_on_id():
    args = {"subframes": [{"goal": "Change a", "owns": ["a.py"]}]}
    with pytest.raises(ParallelInputError) as refusal:
        _request(args, "call", "2")
    assert refusal.value.field == "id"
    assert refusal.value.hint == 'add "id": "subframe_0"'


# Exact call captured from the live run: no goal and no outcome, only id/owns/check/model_class.
_CAPTURED_LIVE_CALL = {"subframes": [
    {"id": "tax_fix", "owns": ["pricing/tax.py", "pricing/checkout.py"],
     "check": ["tests/test_tax.py", "tests/test_checkout.py"], "model_class": "bug_fixer"},
    {"id": "other_fix", "owns": ["pricing/discount.py", "pricing/quantity.py", "pricing/rounding.py"],
     "check": ["tests/test_discount.py", "tests/test_quantity.py", "tests/test_rounding.py"],
     "model_class": "bug_fixer"},
]}


def _captured_repo(tmp_path):
    for path in ("pricing/tax.py", "pricing/checkout.py", "pricing/discount.py",
                 "pricing/quantity.py", "pricing/rounding.py"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("VALUE = 1\n")
    for path in ("tests/test_tax.py", "tests/test_checkout.py", "tests/test_discount.py",
                 "tests/test_quantity.py", "tests/test_rounding.py"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("def test_placeholder():\n    assert True\n")
    return DirectorySnapshot(tmp_path, 0)


def test_captured_goalless_call_is_never_refused_on_outcome(tmp_path):
    snapshot = _captured_repo(tmp_path)
    try:
        _parse_expanded(copy.deepcopy(_CAPTURED_LIVE_CALL), snapshot)
    except ParallelInputError as refusal:
        # Either outcome is acceptable here: the captured call must never be told to add outcome.
        assert refusal.field != "outcome", refusal.hint
        assert (refusal.field, refusal.hint) == ("goal", 'add "goal": "<one sentence>"')


def test_captured_goalless_call_refused_on_goal_at_request_level():
    with pytest.raises(ParallelInputError) as refusal:
        _request(copy.deepcopy(_CAPTURED_LIVE_CALL), "call", "2")
    assert refusal.value.unit_id == "tax_fix"
    assert refusal.value.field == "goal"
    assert refusal.value.hint == 'add "goal": "<one sentence>"'


def test_schema_required_equals_description_short_form():
    # Guard: the schema's top-level required list is exactly the short form the description tells the model to send.
    description = PARALLEL_TOOL["function"]["description"]
    short_form = re.search(r"\{([a-z_, ]+)\}", description)
    assert short_form is not None
    described_fields = [field.strip() for field in short_form.group(1).split(",")]
    assert described_fields == ["id", "goal", "owns"]
    assert _UNIT["required"] == described_fields
    assert set(_UNIT["required"]) <= set(_UNIT["properties"])
