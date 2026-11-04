"""Measured subprocess budgets stay tied to their measurements instead of drifting into bare literals."""
from __future__ import annotations

import ast
from pathlib import Path

from ._support import HEADROOM, MEASURED_SECONDS, budget

ACCEPTANCE = Path(__file__).resolve().parent
BUDGETED = {
    "test_resume_completion.py": "test_e19_volatile_shell_observation_is_byte_identical_on_resume",
    "test_no_skills_wire_compat.py": "_run_tree",
    "../test_refusal_names_live_action.py": "recorded_blocked_run",
}


def _budget_cases(tree: ast.AST) -> set[str]:
    return {node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "budget"
            and node.args and isinstance(node.args[0], ast.Constant)}


def _literal_timeouts(function: ast.FunctionDef) -> list[int]:
    return [keyword.value.lineno for node in ast.walk(function) if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "timeout" and isinstance(keyword.value, ast.Constant)
            and keyword.value.value > 60]


def test_every_measured_case_is_used_and_budgeted_functions_have_no_bare_timeout():
    used = set()
    for name, function_name in BUDGETED.items():
        tree = ast.parse((ACCEPTANCE / name).read_text())
        used |= _budget_cases(tree)
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == function_name)
        assert not _literal_timeouts(function), f"{name}:{function_name} uses a bare timeout literal"
    assert used == set(MEASURED_SECONDS)


def test_budget_is_headroom_over_the_measurement():
    assert HEADROOM >= 1.5
    for case, seconds in MEASURED_SECONDS.items():
        assert budget(case) >= HEADROOM * seconds
        assert budget(case) - HEADROOM * seconds < 1
