"""A response funded by a terminal reserve may make its one tool call even at the tool-attempt cap.

Before the fix, ContinuationLimits charged the terminal reserve in charge_response() but
can_start_tool_call() only checked the tool-attempt cap, so the funded response's write (for
example the final answer file) was refused. Each funded terminal response now buys exactly one
extra tool attempt, recorded as unit "terminal_tool_attempt", and that count survives restore.

Budget arithmetic used throughout: mode defaults to "decision", so tool_call_cap is None and
tool_attempt_limit == limit. limit=1 therefore means 1 model decision and 1 tool attempt.
"""
from __future__ import annotations

import pytest

from kernel_read_fixes import ContinuationLimits


class _Journal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, kind: str, **fields: object) -> None:
        self.events.append((kind, fields))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]

    def units(self) -> list[object]:
        return [fields.get("unit") for kind, fields in self.events if kind == "continuation_unit_charged"]


@pytest.fixture(autouse=True)
def _no_resumed_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    import kernel_read_fixes

    monkeypatch.setattr(kernel_read_fixes, "_RESUMED_DEADLINE_MONOTONIC", None)


def _budget(journal: _Journal) -> ContinuationLimits:
    return ContinuationLimits(limit=1, tool_multiplier=1, journal=journal)


def test_funded_terminal_response_can_make_its_one_tool_call() -> None:
    journal = _Journal()
    budget = _budget(journal)

    assert budget.charge_response() is True
    assert budget.register_tool_call(tool="search") is True
    assert budget.tool_calls_used == 1  # tool attempt cap (1) now reached

    assert budget.can_start_response() is False  # decisions spent, no reserve yet
    assert budget.reserve_terminal() is True
    assert budget.charge_response() is True  # funded by the terminal reserve

    assert budget.can_start_tool_call() is True
    assert budget.register_tool_call(tool="write") is True
    assert "terminal_tool_attempt" in journal.units()
    assert budget.terminal_tool_attempts_used == 1
    assert budget.tool_calls_used == 2


def test_terminal_tool_attempt_is_one_per_funded_response() -> None:
    journal = _Journal()
    budget = _budget(journal)

    budget.charge_response()
    budget.register_tool_call(tool="search")
    budget.reserve_terminal()
    budget.charge_response()
    assert budget.register_tool_call(tool="write") is True

    assert budget.can_start_tool_call() is False
    assert budget.register_tool_call(tool="write_again") is False
    assert "tool_attempt_cap_hit" in journal.kinds()
    assert budget.terminal_tool_attempts_used == 1


def test_no_reserve_means_cap_still_refuses() -> None:
    journal = _Journal()
    budget = _budget(journal)

    budget.charge_response()
    assert budget.register_tool_call(tool="search") is True

    assert budget.can_start_tool_call() is False
    assert budget.register_tool_call(tool="write") is False
    assert "tool_attempt_cap_hit" in journal.kinds()


def test_sandbox_deadline_still_refuses_terminal_tool() -> None:
    journal = _Journal()
    budget = _budget(journal)

    budget.charge_response()
    budget.register_tool_call(tool="search")
    budget.reserve_terminal()
    assert budget.charge_response() is True

    # Set the deadline after funding: reserve_terminal and charge_response both refuse past deadlines.
    budget.sandbox_deadline_monotonic = 1.0
    assert budget.can_start_tool_call() is False
    assert budget.register_tool_call(tool="write") is False
    assert "continuation_deadline_exhausted" in journal.kinds()


def test_restore_usage_folds_terminal_tool_attempts() -> None:
    journal = _Journal()
    budget = _budget(journal)

    budget.restore_usage(
        [
            {"kind": "continuation_unit_charged", "unit": "model_decision"},
            {"kind": "continuation_unit_charged", "unit": "terminal_reserve"},
            {"kind": "continuation_unit_charged", "unit": "terminal_response"},
            {"kind": "continuation_unit_charged", "unit": "terminal_tool_attempt"},
        ],
        tool_calls_used=2,
    )

    assert budget.terminal_reserves_consumed == 1
    assert budget.terminal_tool_attempts_used == 1
    assert budget._terminal_tool_attempt_available is False
    assert budget.continuation_snapshot()["terminal_tool_attempts_used"] == 1


def test_cap_not_reached_path_unchanged() -> None:
    journal = _Journal()
    budget = _budget(journal)

    assert budget.register_tool_call(tool="read") is True

    assert budget.tool_calls_used == 1
    assert budget.terminal_tool_attempts_used == 0
    charged = [fields for kind, fields in journal.events if kind == "continuation_unit_charged"]
    assert [(fields["unit"], fields["tool"]) for fields in charged] == [("tool_attempt", "read")]
