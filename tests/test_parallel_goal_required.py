"""The parallel schema advertises the goal short form; the runtime validator still accepts legacy outcome contracts."""
from __future__ import annotations

import copy
import re

import pytest

from stateframe.engine.kernel_parallel import (PARALLEL_TOOL, ParallelInputError, _UNIT,
                                               _complete_arguments, _request)
from stateframe.engine.kernel_parallel_admission import DirectorySnapshot


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
