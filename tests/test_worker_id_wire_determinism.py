"""Guard request bytes against the memory-isolation worker identity fallback.

``worker_identity_from_row`` falls back to a derived identity when neither the
task row, the owner, nor ``STATEFRAME_WORKER_ID`` names a worker.  That
identity is part of the task namespace, and the namespace label is rendered
into every admitted carry capsule, which ``prepare`` appends to a user message.
If the fallback carried process or thread identity, two runs with identical
inputs would send different request bytes.
"""
from __future__ import annotations

import json
import os
import threading
from types import SimpleNamespace

import pytest

import kernel_memory_isolation as isolation


ROW = {"task_id": "task-alpha", "fn": "fix_pricing", "sha": "0123abcd", "repo": "pricing"}
RUN_ID = "run-fixed"
MESSAGES = (
    {"role": "system", "content": "You repair the repository until its tests pass."},
    {"role": "user", "content": "Fix the pricing code so the tests pass."},
)


@pytest.fixture
def no_configured_worker(monkeypatch):
    monkeypatch.delenv("STATEFRAME_WORKER_ID", raising=False)
    monkeypatch.setattr(isolation, "_WORKER_REGISTRIES", {})


def _impersonate(monkeypatch, *, pid: int, thread_ident: int) -> None:
    monkeypatch.setattr(os, "getpid", lambda: pid)
    monkeypatch.setattr(threading, "get_ident", lambda: thread_ident)


def _request_bytes() -> bytes:
    """Mirror the runtime: bind a producer envelope to the owner namespace, then prepare."""
    owner = SimpleNamespace(task_row=dict(ROW))
    namespace = isolation.namespace_for_owner(owner, run_id=RUN_ID)
    boundary = isolation.MemoryIsolationBoundary(
        namespace, registry=isolation.registry_for_worker(namespace.worker_id))
    # The production producer names its source worker the same way the runtime does.
    source = isolation.TaskNamespace(
        worker_id=isolation.worker_identity_from_row(ROW), task_id=ROW["task_id"],
        run_id="prior-history")
    envelope = {
        "schema": isolation.CARRY_SCHEMA,
        "kind": "prior_attempt",
        "source": source.to_dict(),
        "records": [isolation.MemoryRecord(
            "prior_attempt", "summary", "attempt 1 rounded before applying the discount").to_dict()],
        "reason": "same-task prior agent-world history",
    }
    assert boundary.admit_envelope(envelope), boundary.rejections
    prepared, uncapped = boundary.prepare([dict(m) for m in MESSAGES], [dict(m) for m in MESSAGES])
    return json.dumps({"messages": prepared, "uncapped": uncapped},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_worker_fallback_ignores_process_and_thread_identity(monkeypatch, no_configured_worker):
    _impersonate(monkeypatch, pid=31337, thread_ident=140000000001)
    first = isolation.worker_identity_from_row(ROW)
    _impersonate(monkeypatch, pid=4242, thread_ident=140000000002)
    second = isolation.worker_identity_from_row(ROW)

    assert first == second == isolation.DEFAULT_WORKER_ID
    for leaked in ("31337", "4242", "140000000001", "140000000002"):
        assert leaked not in first


def test_request_bytes_identical_across_pids_when_worker_fallback_fires(monkeypatch, no_configured_worker):
    _impersonate(monkeypatch, pid=31337, thread_ident=140000000001)
    first = _request_bytes()
    monkeypatch.setattr(isolation, "_WORKER_REGISTRIES", {})
    _impersonate(monkeypatch, pid=4242, thread_ident=140000000002)
    second = _request_bytes()

    # The guard is only meaningful if the rendered capsule reached the request.
    assert isolation.CAPSULE_HEADER.encode("utf-8") in first
    assert b"attempt 1 rounded before applying the discount" in first
    assert first == second


def test_explicit_worker_identities_still_take_precedence(monkeypatch, no_configured_worker):
    assert isolation.worker_identity_from_row({**ROW, "worker_id": "worker-07"}) == "worker-07"
    owner = SimpleNamespace(worker_id="owner-worker")
    assert isolation.worker_identity_from_row(ROW, owner) == "owner-worker"
    monkeypatch.setenv("STATEFRAME_WORKER_ID", "configured-worker")
    assert isolation.worker_identity_from_row(ROW) == "configured-worker"
