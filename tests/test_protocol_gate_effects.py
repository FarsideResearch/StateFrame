"""Protocol edits must grant the same completion effects and repair signal as legacy edits.

Site: kernel_completion_authority.ACTION_RESULT_EFFECTS / _observed_effects, and
kernel_completion_policy.REPAIR_TOOLS (consumed by evaluate()).  An applied
edit_source / rewrite_source is a product edit: it must resolve product_fixed
and candidate_revision exactly like the legacy edit tool, and a menu offering
only protocol edits must report repair=True.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import kernel_completion_authority as authority
import kernel_completion_policy as policy

PROTOCOL_EDITS = ("edit_source", "rewrite_source")
PRODUCT_EFFECTS = frozenset({"product_fixed", "candidate_revision"})


@pytest.mark.parametrize("action", PROTOCOL_EDITS)
def test_applied_protocol_edit_grants_product_effects(action: str) -> None:
    assert authority._observed_effects(action, "", "APPLIED", {}) == PRODUCT_EFFECTS


def test_applied_legacy_edit_still_grants_product_effects() -> None:
    assert authority._observed_effects("edit", "", "APPLIED", {}) == PRODUCT_EFFECTS


@pytest.mark.parametrize("action", PROTOCOL_EDITS)
def test_failed_protocol_edit_grants_no_effects(action: str) -> None:
    assert authority._observed_effects(action, "", "FAILED", {}) == frozenset()


@pytest.mark.parametrize("action", PROTOCOL_EDITS)
def test_protocol_edit_result_feeds_product_contract(action: str) -> None:
    contract = authority.ProgressContract(
        reason_code="PRODUCT_VIOLATION",
        cause_type=authority.ProductViolation,
        resolution_predicate="product_fixed",
        recovery_owner="product",
        permitted_action=action,
    )
    assert contract.result_feeds_resolution({"action": action, "status": "APPLIED"})


@pytest.mark.parametrize("action", PROTOCOL_EDITS)
def test_protocol_edit_table_entries_exist(action: str) -> None:
    assert authority.ACTION_RESULT_EFFECTS.get(action) == PRODUCT_EFFECTS


def _binding() -> dict[str, object]:
    return {"artifact": "product-a", "environment": "env-a", "policy_identity": "policy-a"}


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(binding=_binding)


@pytest.mark.parametrize("menu", [("edit_source",), ("rewrite_source", "read"), ("edit_source", "read")])
def test_protocol_only_menu_reports_repair(monkeypatch: pytest.MonkeyPatch, menu: tuple[str, ...]) -> None:
    monkeypatch.setattr(policy, "collect_clauses",
                        lambda *a, **k: (policy.ClauseAssessment("tests.public", policy.VIOLATED),))
    monkeypatch.setattr(policy, "_menu", lambda runtime: menu)
    decision = policy.evaluate(_runtime())
    assert decision.repair_on_menu is True
    assert decision.action == "challenge"


def test_read_only_menu_does_not_report_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "collect_clauses",
                        lambda *a, **k: (policy.ClauseAssessment("tests.public", policy.VIOLATED),))
    monkeypatch.setattr(policy, "_menu", lambda runtime: ("read", "run_tests"))
    decision = policy.evaluate(_runtime())
    assert decision.repair_on_menu is False
    assert decision.action == "stop"
