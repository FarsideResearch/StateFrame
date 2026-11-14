"""Durable parent allowance across successive subframe calls."""
from __future__ import annotations

import json

from stateframe.parallel_resources import SendLedger
from stateframe.engine.kernel_parallel_records import UnitOutcome, UnitUsage


def test_two_calls_charge_prior_authorizations_and_release_terminal_holds(tmp_path):
    from stateframe.engine.kernel_parallel_budget import BudgetStore

    run = tmp_path / "run"
    store = BudgetStore(run, limit=13, parent_reserve=2)
    assert store.reserve("first", {"a": 4, "b": 4}, parent_steps_spent=1)
    state = json.loads((run / "parallel/budget.json").read_bytes())
    assert (state["parent_steps_spent"], state["subframe_sends_authorized"],
            state["held_reservations"], state["free"], state["parent_reserve"]) == (1, 0, 8, 2, 2)

    for unit_id in ("a", "b"):
        directory = run / "parallel" / "first" / "units" / unit_id
        directory.mkdir(parents=True)
        ledger = SendLedger(directory / "sends-ledger.jsonl")
        for attempt in range(2):
            send_id = f"{unit_id}:{attempt}"
            assert ledger.authorize_with_cap(send_id, 4)
            ledger.settle(send_id, {})
        store.terminal("first", unit_id, parent_steps_spent=1)
        state = json.loads((run / "parallel/budget.json").read_bytes())
        assert min(state[key] for key in ("parent_steps_spent", "subframe_sends_authorized",
                                          "held_reservations", "free", "parent_reserve")) >= 0
        assert sum(state[key] for key in ("parent_steps_spent", "subframe_sends_authorized",
                                             "held_reservations", "free", "parent_reserve")) == 13
    assert (state["parent_steps_spent"], state["subframe_sends_authorized"],
            state["held_reservations"], state["free"], state["parent_reserve"]) == (1, 4, 0, 6, 2)
    assert not store.can_reserve({"a": 4, "b": 4}, parent_steps_spent=2)
    assert not BudgetStore(run, limit=13, parent_reserve=2).can_reserve(
        {"a": 4, "b": 4}, parent_steps_spent=2)


def test_recovery_reconciles_authorized_send_and_durable_terminal(tmp_path):
    from stateframe.engine.kernel_parallel_budget import BudgetStore

    run = tmp_path / "run"
    budget = BudgetStore(run, limit=13, parent_reserve=2)
    assert budget.reserve("first", {"a": 4, "b": 4}, parent_steps_spent=1)
    directory = run / "parallel" / "first" / "units" / "a"
    directory.mkdir(parents=True)
    ledger = SendLedger(directory / "sends-ledger.jsonl")
    assert ledger.authorize_with_cap("a:ambiguous", 4)
    outcomes = run / "parallel" / "first" / "outcomes.jsonl"
    outcome = UnitOutcome("a", "ENGINE_ERROR", "ENGINE_ERROR", "stopped", None, (), None,
                          "NOT_RUN", UnitUsage(0, 1, 0, 0, None), 0.0, 0.0)
    outcomes.write_text(json.dumps(outcome.to_json()) + "\n")

    BudgetStore(run, limit=13, parent_reserve=2).recover(parent_steps_spent=1)
    state = json.loads((run / "parallel/budget.json").read_bytes())
    assert state["calls"]["first"]["terminal"] == ["a"]
    assert (state["parent_steps_spent"], state["subframe_sends_authorized"],
            state["held_reservations"], state["free"], state["parent_reserve"]) == (
                1, 1, 4, 5, 2)
    assert sum(state[key] for key in ("parent_steps_spent", "subframe_sends_authorized",
                                      "held_reservations", "free", "parent_reserve")) == 13
