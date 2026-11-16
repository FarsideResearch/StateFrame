"""Guard request bytes against the per-run journal id leaking through carry labels.

The runtime builds the active memory namespace with ``run_id`` set to the
continuation journal's random run id.  The namespace label is rendered into
every admitted carry capsule, which ``prepare`` appends to a user message, so
two runs with identical inputs must still send identical bytes.  The run id
stays in the namespace itself: cross-run reads and carries are still refused.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import kernel_memory_isolation as isolation


ROW = {"task_id": "task-alpha", "worker_id": "worker-07", "fn": "fix_pricing", "sha": "0123abcd"}
MESSAGES = (
    {"role": "system", "content": "You repair the repository until its tests pass."},
    {"role": "user", "content": "Fix the pricing code so the tests pass."},
)
RUN_A = "0b5f7c1e-3d0a-4c5e-9f61-6d2a8c1b7e01"
RUN_B = "f93a12d4-77be-4f0c-8a2b-1c5e9d03a6f2"


@pytest.fixture(autouse=True)
def fresh_registries(monkeypatch):
    monkeypatch.setattr(isolation, "_WORKER_REGISTRIES", {})


def _envelope() -> dict:
    source = isolation.TaskNamespace(worker_id=ROW["worker_id"], task_id=ROW["task_id"],
                                     run_id="prior-history")
    return {
        "schema": isolation.CARRY_SCHEMA,
        "kind": "prior_attempt",
        "source": source.to_dict(),
        "records": [isolation.MemoryRecord(
            "prior_attempt", "summary", "attempt 1 rounded before applying the discount").to_dict()],
        "reason": "same-task prior agent-world history",
    }


def _boundary(run_id: str) -> isolation.MemoryIsolationBoundary:
    namespace = isolation.namespace_for_owner(SimpleNamespace(task_row=dict(ROW)), run_id=run_id)
    return isolation.MemoryIsolationBoundary(namespace, registry=isolation.WorkerMemoryRegistry(
        namespace.worker_id))


def _request_bytes(run_id: str) -> bytes:
    boundary = _boundary(run_id)
    assert boundary.admit_envelope(_envelope()), boundary.rejections
    prepared, uncapped = boundary.prepare([dict(m) for m in MESSAGES], [dict(m) for m in MESSAGES])
    return json.dumps({"messages": prepared, "uncapped": uncapped},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_request_bytes_identical_across_journal_run_ids():
    first, second = _request_bytes(RUN_A), _request_bytes(RUN_B)

    # The guard is only meaningful if the rendered capsule reached the request.
    assert isolation.CAPSULE_HEADER.encode("utf-8") in first
    assert b"attempt 1 rounded before applying the discount" in first
    assert first == second
    for run_id in (RUN_A, RUN_B):
        assert run_id.encode("utf-8") not in first


def test_rendered_label_still_separates_tasks_and_workers():
    base = isolation.TaskNamespace("worker-07", "task-alpha", RUN_A)
    assert base.render_label == isolation.TaskNamespace("worker-07", "task-alpha", RUN_B).render_label
    assert base.render_label != isolation.TaskNamespace("worker-07", "task-beta", RUN_A).render_label
    assert base.render_label != isolation.TaskNamespace("worker-08", "task-alpha", RUN_A).render_label


def test_namespace_identity_still_includes_run_id():
    a = isolation.TaskNamespace("worker-07", "task-alpha", RUN_A)
    b = isolation.TaskNamespace("worker-07", "task-alpha", RUN_B)
    assert a != b
    assert a.digest != b.digest
    assert a.to_dict()["run_id"] == RUN_A


def test_cross_run_reads_and_carries_still_refused():
    registry = isolation.WorkerMemoryRegistry("worker-07")
    run_a = isolation.TaskNamespace("worker-07", "task-alpha", RUN_A)
    run_b = isolation.TaskNamespace("worker-07", "task-alpha", RUN_B)
    registry.put(run_a, isolation.MemoryRecord("prior_attempt", "summary", "run a only"))

    with pytest.raises(isolation.NamespaceViolation):
        registry.read(run_b, run_a, "summary")
    assert registry.snapshot(run_b) == ()

    # A capsule sealed for run A cannot be admitted by run B's boundary.
    capsule = registry.issue_capsule(run_a, run_a, reason="carry within run a")
    boundary_b = isolation.MemoryIsolationBoundary(run_b, registry=registry)
    boundary_b.prepare([{"role": "user", "content": "go", "memory_capsule": capsule.to_dict()}], [])
    assert boundary_b.accepted_count == 0
    assert "carry target is not the active namespace" in boundary_b.rejections
    with pytest.raises(isolation.NamespaceViolation):
        registry.accept_capsule(run_b, capsule)
