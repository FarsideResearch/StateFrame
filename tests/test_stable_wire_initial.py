"""Initial selection records stay bounded, content-free and detached."""
from __future__ import annotations

import copy

import pytest

from kernel_stable_wire_record import validate_reserves_record
from test_stable_wire_trim_selection import allocator as allocator


@pytest.fixture
def initial_record(allocator):
    prepare, render = allocator
    prepared = prepare()
    render(prepared)
    return prepared[1]._stable_wire_reserves_record


@pytest.mark.parametrize("corruption", [
    "content", "boolean_index", "duplicate_index", "decision_kind", "decision_count",
    "method", "ratio", "pair", "bound", "wrong_result_shape",
])
def test_initial_record_rejects_non_identifiers_and_malformed_numbers(initial_record, corruption):
    record = copy.deepcopy(initial_record)
    initial = record["initial_selection"]
    if corruption == "content":
        initial["removed"] = [[0, "private source content"]]
    elif corruption == "boolean_index":
        initial["kept"][0][0] = True
    elif corruption == "duplicate_index":
        initial["kept"].append(copy.deepcopy(initial["kept"][0]))
    elif corruption == "decision_kind":
        initial["decisions"][0][0] = True
    elif corruption == "decision_count":
        initial["decisions"][0][2][0] = False
    elif corruption == "method":
        initial["decisions"][0][2][1] = 3
    elif corruption == "ratio":
        counter = next(row[3] for row in initial["decisions"] if row[3])
        counter[3] = float("inf")
    elif corruption == "pair":
        initial["decisions"][0][4] = [False, 1]
    elif corruption == "bound":
        initial["decisions"] = [initial["decisions"][0]] * 20_001
    elif corruption == "wrong_result_shape":
        initial["decisions"][0][2] = [1]
    with pytest.raises(ValueError, match="stable wire initial"):
        validate_reserves_record(record)


def test_initial_record_only_contains_numbers_and_candidate_identifiers(initial_record):
    initial = initial_record["initial_selection"]

    def inspect(value):
        if isinstance(value, dict):
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
        else:
            assert type(value) in (int, float) or (
                isinstance(value, str) and value.startswith("sha256:") and len(value) == 71)

    inspect(initial)
    detached = validate_reserves_record(initial_record)
    detached["initial_selection"]["decisions"][0][2][0] += 1
    assert detached != initial_record
    assert validate_reserves_record(initial_record) == initial_record
