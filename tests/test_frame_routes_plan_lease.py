"""MR-4: the plan lease keeps planner ownership through read-kind actions (routing schema 2 only)."""
from __future__ import annotations

import pytest

from stateframe import routes
import kernel_frame_routes as frame_routes


DEFAULT = {"endpoint": "http://127.0.0.1:8080/v1", "model": "base", "api_key_env": None,
           "temperature": 1.0, "max_tokens": 24000, "timeout_s": 1800.0,
           "slot": None, "context_tokens": 65536, "allow_remote": False}

CURRENT = [{"receipt_id": "r1", "revision": "rev"}, {"receipt_id": "r2", "revision": "rev"}]
FAIL_CHECK = {"execution_ids": ["a", "b"], "check_receipt_ids": ["r1", "r2"], "status": "FAIL"}


@pytest.fixture(autouse=True)
def clear_routes(monkeypatch):
    monkeypatch.delenv("STATEFRAME_ROUTING_SCHEMA", raising=False)
    frame_routes.install(None)
    yield
    frame_routes.install(None)


def install_routes():
    spec = routes.resolve(DEFAULT, cli=["read.model=reader"], environ={}).public()
    frame_routes.install(spec)


def fail_tests_at_rev():
    frame_routes.close_action({"tool": "run_tests", "tool_call_id": "t"}, revision="rev",
                              canonical=FAIL_CHECK, receipts=CURRENT)


def test_v2_start_read_read_keeps_plan(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    assert frame_routes.role_for_request("rev") == "plan"
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "plan"
    frame_routes.close_action({"tool": "grep"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "plan"


def test_v2_fail_tests_then_read_keeps_plan(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    fail_tests_at_rev()
    assert frame_routes.role_for_request("rev") == "plan"
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "plan"


def test_v2_plan_then_edit_is_not_plan(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    assert frame_routes.role_for_request("rev") == "plan"
    frame_routes.close_action({"tool": "edit"}, revision="rev")
    assert frame_routes.role_for_request("rev") != "plan"
    assert frame_routes.role_for_request("rev") == "read"


def test_v2_read_after_edit_does_not_reopen_plan(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    frame_routes.close_action({"tool": "edit"}, revision="rev")
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "read"


def test_v2_fail_lease_closed_by_non_read_action(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    fail_tests_at_rev()
    frame_routes.close_action({"tool": "edit"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "read"


def test_v2_fail_lease_does_not_survive_revision_change(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    fail_tests_at_rev()
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("new-rev") == "read"


def test_v2_non_qualifying_fail_does_not_open_lease(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    frame_routes.close_action({"tool": "edit"}, revision="rev")
    incomplete = {**FAIL_CHECK, "check_receipt_ids": ["r1"]}
    frame_routes.close_action({"tool": "run_tests"}, revision="rev", canonical=incomplete,
                              receipts=CURRENT[:1])
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "read"


def test_v2_passing_tests_close_lease(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    frame_routes.close_action({"tool": "run_tests"}, revision="rev",
                              canonical={**FAIL_CHECK, "status": "PASS"},
                              receipts=[{**row, "status": "PASS"} for row in CURRENT])
    assert frame_routes.role_for_request("rev") == "read"


def test_v2_install_reset_reopens_start_lease(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "2")
    install_routes()
    frame_routes.close_action({"tool": "edit"}, revision="rev")
    install_routes()
    assert frame_routes.role_for_request("rev") == "plan"


def test_v1_start_read_is_read(monkeypatch):
    monkeypatch.setenv("STATEFRAME_ROUTING_SCHEMA", "1")
    install_routes()
    assert frame_routes.role_for_request("rev") == "plan"
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "read"


def test_v1_unset_fail_then_read_is_read():
    install_routes()
    fail_tests_at_rev()
    assert frame_routes.role_for_request("rev") == "plan"
    frame_routes.close_action({"tool": "read"}, revision="rev")
    assert frame_routes.role_for_request("rev") == "read"


def test_local_read_set_matches_router_read_set():
    # The router keys read actions by uppercase router names; frame tools are lowercase.
    from kernel_stateframe_model_router import _READ_ACTIONS as router_reads
    assert frame_routes.LEASE_READ_ACTIONS == frozenset(name.lower() for name in router_reads)
