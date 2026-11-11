"""The execution session retains both launch identity and memory diagnostics."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import kernel_exec_session as execution


def test_mutation_spawn_keeps_identity_and_memory_diagnostics(monkeypatch):
    session = object.__new__(execution.ExecSession)
    session._lock = threading.RLock()
    session._initial_capability = None
    session._handles = []
    monkeypatch.setattr(execution.ExecSession, "_start", lambda self, cap: None)
    monkeypatch.setattr(execution, "_host_mutation_identity", lambda identity: identity)
    response = {
        "id": "j1",
        "identity": {"pid": 123},
        "memory_limit_mechanism": "rlimit_data+rss_supervisor",
        "cgroup_leaf_pattern": "none",
    }
    monkeypatch.setattr(execution.ExecSession, "_rpc", lambda self, request: response)
    capability = SimpleNamespace(purpose="mutation", as_dict=lambda: {"purpose": "mutation"})

    handle = session._spawn(["python", "unit.py"], capability)

    assert handle.identity == {"pid": 123}
    assert handle.memory_limit_mechanism == "rlimit_data+rss_supervisor"
    assert handle.cgroup_leaf_pattern == "none"
    assert session._handles == [handle]
