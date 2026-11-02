"""Hook budget sections are present only in a wired request diet."""

import sys
from types import SimpleNamespace

import kernel_request_diet as diet


def test_no_hooks_diet_breakdown_keeps_021_sections_and_allowlist(monkeypatch):
    monkeypatch.delitem(sys.modules, "kernel_hook_wiring", raising=False)
    body = {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Task: work"}], "tools": []}
    result = diet.breakdown(body, "S1")
    assert "user_hooks" not in result["sections"]
    assert "user_hooks" not in result["allowlist"]


def test_wired_diet_admits_typed_hook_panel(monkeypatch):
    monkeypatch.setitem(sys.modules, "kernel_hook_wiring", SimpleNamespace(_ACTIVE=object()))
    body = {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": "Task: work"},
                         {"role": "user", "content": "[USER HOOKS] note",
                          "_request_budget_section": "user_hooks"}], "tools": []}
    result = diet.breakdown(body, "S1")
    assert result["sections"]["user_hooks"]["messages"] == 1
    assert "user_hooks" in result["allowlist"]
