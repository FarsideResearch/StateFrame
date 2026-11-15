"""Pins the legacy source-branch gate in assemble_framefit5_request (UNITVIS-073 scope).

Allocator-ON parent requests must keep the pre-UNITVIS-073 gate (legacy branches off), so
their wire bytes match the default parallel-off baseline. Only subframe workers (unit policy
active) run the legacy branches with the allocator on.
"""
import sys
import types

import pytest

from kernel_stateframe_framefit5 import _legacy_source_path


@pytest.fixture
def no_unit_policy(monkeypatch):
    monkeypatch.delitem(sys.modules, "kernel_unit_policy", raising=False)


def _policy_module(active_value):
    module = types.ModuleType("kernel_unit_policy")
    module.active = lambda: active_value
    return module


def test_allocator_off_unallocated_runs_legacy_branches(no_unit_policy):
    assert _legacy_source_path(None, False) is True


def test_allocator_on_parent_unallocated_keeps_pre_unitvis_gate(no_unit_policy):
    # Pre-9ad377c2 gate was `not allocator_on and unallocated`: False here, so the parent
    # request bytes stay identical to the default parallel-off baseline.
    assert _legacy_source_path(None, True) is False


def test_allocated_request_never_runs_legacy_branches(no_unit_policy):
    assert _legacy_source_path(object(), False) is False
    assert _legacy_source_path(object(), True) is False


def test_allocator_on_subframe_worker_unallocated_runs_legacy_branches(monkeypatch):
    monkeypatch.setitem(sys.modules, "kernel_unit_policy", _policy_module(object()))
    assert _legacy_source_path(None, True) is True


def test_allocator_on_unit_policy_present_but_inactive_stays_off(monkeypatch):
    monkeypatch.setitem(sys.modules, "kernel_unit_policy", _policy_module(None))
    assert _legacy_source_path(None, True) is False


def test_runtime_bound_unit_policy_counts_as_subframe_worker(monkeypatch):
    from types import SimpleNamespace
    import kernel_unit_policy
    monkeypatch.setattr(kernel_unit_policy, "active", lambda: None)
    assert _legacy_source_path(None, True, SimpleNamespace(_unit_policy=object())) is True
    assert _legacy_source_path(None, True, SimpleNamespace(_unit_policy=None)) is False
    assert _legacy_source_path(object(), True, SimpleNamespace(_unit_policy=object())) is False
