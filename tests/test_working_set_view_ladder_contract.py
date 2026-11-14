"""I-6 merge contract only: this tree has no production view-ladder hook."""
from __future__ import annotations

import json

import pytest


def test_allocator_on_bypasses_both_ladder_and_pre_trim(monkeypatch):
    from _view_ladder_contract import apply_view_ladder_contract

    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "on")
    body = {"messages": [{"role": "user", "content": "source stays intact"}]}

    def forbidden(_body):
        pytest.fail("allocator ON must bypass both pre-trim and apply_views")

    wires = []
    for ladder_enabled in (False, True):
        result = apply_view_ladder_contract(
            body, ladder_enabled=ladder_enabled, pre_trim=forbidden, apply_views=forbidden)
        assert result is body
        wires.append(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode())
    assert wires[0] == wires[1]


@pytest.mark.parametrize("allocator", [None, "off", "ON", "true", "1"])
@pytest.mark.parametrize("ladder_enabled", [False, True])
def test_non_on_values_preserve_ladder_switch_and_order(monkeypatch, allocator, ladder_enabled):
    from _view_ladder_contract import apply_view_ladder_contract

    if allocator is None:
        monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", allocator)
    calls = []

    def pre_trim(wire):
        calls.append("pre_trim")
        return wire + b"/trimmed"

    def apply_views(wire):
        calls.append("apply_views")
        return wire + b"/views"

    result = apply_view_ladder_contract(
        b"request", ladder_enabled=ladder_enabled, pre_trim=pre_trim, apply_views=apply_views)
    assert result == (b"request/trimmed/views" if ladder_enabled else b"request")
    assert calls == (["pre_trim", "apply_views"] if ladder_enabled else [])


@pytest.mark.parametrize("enabled", [False, True])
def test_contract_delegates_to_live_read_receipts_authority(monkeypatch, enabled):
    import kernel_read_receipts
    from _view_ladder_contract import apply_view_ladder_contract

    # Contradict the environment to catch a copied switch check or cached value.
    monkeypatch.setenv("STATEFRAME_WORKING_SET_ALLOCATOR", "off" if enabled else "on")
    calls = []

    def authority():
        calls.append("enabled")
        return enabled

    monkeypatch.setattr(kernel_read_receipts, "enabled", authority)
    result = apply_view_ladder_contract(
        b"request", ladder_enabled=True, pre_trim=lambda wire: wire + b"/trimmed",
        apply_views=lambda wire: wire + b"/views")
    assert result == (b"request" if enabled else b"request/trimmed/views")
    assert calls == ["enabled"]
