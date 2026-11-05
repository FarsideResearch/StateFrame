"""Unified completion snapshots must not expose host environment identity in request IDs."""

from types import SimpleNamespace

import pytest

import kernel_completion_policy as policy
from kernel_stateframe_runtime import StateFrameRuntime


def _binding(environment: str, *, rich: bool) -> dict[str, object]:
    binding: dict[str, object] = {
        "artifact": "product-a",
        "environment": environment,
        "test_harness_revision": "harness-a",
        "policy_identity": "policy-a",
        "authorization_scope": "scope-a",
        "tenant_identity": "tenant-a",
    }
    if rich:
        binding.update({
            "product_revision": "product-a",
            "environment_revision": environment,
            "product_files": {"src/app.py": "file-a"},
            "harness_files": {"tests/test_app.py": "test-a"},
        })
    return binding


def _runtime(binding: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(binding=lambda: binding)


@pytest.mark.parametrize("rich", [False, True])
def test_snapshot_id_ignores_environment_binding(rich: bool) -> None:
    clauses = (policy.ClauseAssessment("tests.public", policy.SATISFIED),)
    first = _runtime(_binding("env-a", rich=rich))
    second = _runtime(_binding("env-b", rich=rich))

    assert policy.snapshot_id(first, clauses) == policy.snapshot_id(second, clauses)


def test_snapshot_id_changes_for_product_or_clause_state() -> None:
    binding = _binding("env-a", rich=True)
    clauses = (policy.ClauseAssessment("tests.public", policy.SATISFIED),)
    original = policy.snapshot_id(_runtime(binding), clauses)
    changed_product = {**binding, "product_revision": "product-b", "artifact": "product-b"}
    changed_clauses = (policy.ClauseAssessment("tests.public", policy.VIOLATED),)

    assert policy.snapshot_id(_runtime(changed_product), clauses) != original
    assert policy.snapshot_id(_runtime(binding), changed_clauses) != original


def test_dispatch_audit_keeps_environment_revision() -> None:
    binding = _binding("env-a", rich=True)
    events: list[tuple[str, dict[str, object]]] = []
    journal = SimpleNamespace(manifest={"environment_revision": "env-a"},
                              emit=lambda kind, **fields: events.append((kind, fields)))
    runtime = SimpleNamespace(binding=lambda: binding, prepared_binding=binding,
                              snapshot_id="frame-a", journal=journal)

    assert StateFrameRuntime.dispatch_binding_current(runtime) is True
    assert journal.manifest["environment_revision"] == "env-a"
    assert events == [("step_dispatch_binding", {
        "snapshot_id": "frame-a", "accepted": True, "expected": binding, "current": binding,
    })]
    current = events[0][1]["current"]
    assert isinstance(current, dict)
    assert current["environment_revision"] == "env-a"
