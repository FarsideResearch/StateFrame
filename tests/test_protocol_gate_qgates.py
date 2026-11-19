"""Protocol edit tools must pass through the qgate before-first-product-edit gate.

edit_source / rewrite_source are real product edits. guard_action keys off
kernel_qgates.EDIT_KINDS, which previously held only legacy names, so a protocol
edit before the gate was answered dispatched without a typed gate.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernel_qgates as qgates
from kernel_edit_tool_names import PROTOCOL_EDIT_TOOL_NAMES


@pytest.fixture
def gates_on(monkeypatch):
    monkeypatch.setenv(qgates.FLAG, "on")


def _runtime():
    return SimpleNamespace(journal=None)


@pytest.mark.parametrize("tool", sorted(PROTOCOL_EDIT_TOOL_NAMES))
def test_protocol_edit_before_first_product_edit_is_blocked(gates_on, tool):
    runtime = _runtime()
    feedback = qgates.guard_action(runtime, tool, {"edits": []})
    assert feedback is not None
    assert "before_first_product_edit" in feedback


@pytest.mark.parametrize("tool", sorted(PROTOCOL_EDIT_TOOL_NAMES))
def test_protocol_edit_passes_once_first_product_edit_gate_is_answered(gates_on, tool):
    runtime = _runtime()
    qgates.state(runtime)["completed"] = ["before_first_product_edit"]
    assert qgates.guard_action(runtime, tool, {"edits": []}) is None


def test_legacy_edit_before_first_product_edit_is_still_blocked(gates_on):
    runtime = _runtime()
    feedback = qgates.guard_action(runtime, "str_replace", {"path": "a.py"})
    assert feedback is not None
    assert "before_first_product_edit" in feedback


def test_read_action_is_not_blocked_by_first_product_edit_gate(gates_on):
    runtime = _runtime()
    assert qgates.guard_action(runtime, "read", {"path": "a.py"}) is None


def test_gate_off_never_blocks_protocol_edit(monkeypatch):
    monkeypatch.setenv(qgates.FLAG, "off")
    assert qgates.guard_action(_runtime(), "edit_source", {"edits": []}) is None
