"""DECIDE2 check evidence keeps v0.4.13 bytes outside audit-field removal."""
from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace

import pytest

import kernel_stateframe_runtime as runtime_module
import kernel_v46_mechanism_wiring as wiring


_EXCLUDED_FIELDS = {
    "memory_limit_mechanism", "memory_limit_killed", "memory_limit_reason",
    "cgroup_leaf_pattern", "environment", "environment_revision",
}


def _base_model_facing_check_evidence(runtime):
    """Golden v0.4.13 inline implementation from 6003aba."""
    return {
        "dependency_identity": runtime_module.digest({
            "test_receipts": runtime.test_receipts,
            "host_checks": getattr(runtime, "host_check_receipts", ()),
            "scope": getattr(runtime, "_scope_lock_last_result", None),
        }),
        "public_evidence": tuple(
            runtime.test_receipts + list(getattr(runtime, "host_check_receipts", ()))
        ),
    }


def _without_audit_fields(value, parent_key=""):
    if isinstance(value, Mapping):
        return {
            key: (None if str(key) == "likely_initiator"
                  and item == "aggregate_rss_supervisor"
                  else _without_audit_fields(item, str(key)))
            for key, item in value.items()
            if str(key) not in _EXCLUDED_FIELDS
            and not (parent_key == "observation_certificate" and str(key) == "binding")
        }
    if isinstance(value, list):
        return [_without_audit_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_audit_fields(item) for item in value)
    return value


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


class _Runtime:
    def __init__(self, test_receipts, host_check_receipts, scope):
        self.test_receipts = test_receipts
        self.host_check_receipts = host_check_receipts
        self._scope_lock_last_result = scope
        self.journal = SimpleNamespace(run_id="decide2-wire-compat")
        self.decide2_provider = lambda *_args, **_kwargs: None
        self.owner = SimpleNamespace(
            _capability_snapshot=SimpleNamespace(
                final_actions=("localize", "localization_frontier"),
                capability_digest="capability-digest",
                product_revision="product-revision",
            ),
            menu_admissible_actions=lambda: ("localize", "localization_frontier"),
        )

    def binding(self):
        return {"product_revision": "product-revision"}

    def _idealpath_owner(self):
        return self.owner

    def _target_relative(self):
        return ""


class _Journal:
    run_id = "decide2-wire-compat"

    def emit(self, *_args, **_kwargs):
        return None


def _runtime(test_receipts, host_checks, scope):
    runtime = _Runtime(test_receipts, host_checks, scope)
    runtime.journal = _Journal()
    return runtime


def _checkpoint(runtime, evidence):
    frame = {
        "receipt_count": 0,
        "revision": "product-revision",
        "dependency_identity": evidence["dependency_identity"],
        "public_evidence": evidence["public_evidence"],
    }
    receipt = wiring.decide2_checkpoint(runtime, frame)
    return receipt, runtime._v46_mechanism_notice


def _run_against_golden(monkeypatch, runtime, evidence):
    captured_requests = []
    captured_bodies = []

    def capture_request(_runtime, request, _provider, **_kwargs):
        from kernel_decide2 import build_frame, build_micro_request

        captured_requests.append(request)
        captured_bodies.append(build_micro_request(
            build_frame(request), rung="L1",
            tool_surface=({"type": "function", "function": {"name": "read"}},),
        ))
        return {
            "status": "selected", "route": "localize", "arm": "micro",
            "rung": "D1", "provider_calls": 1,
        }

    monkeypatch.setattr(wiring, "host_trigger_decide2", capture_request)
    receipt, notice = _checkpoint(runtime, evidence)
    assert len(captured_requests) == 1
    return captured_requests[0], captured_bodies[0], receipt, notice


@pytest.fixture(autouse=True)
def enable_decide2(monkeypatch):
    monkeypatch.setenv("STATEFRAME_DECIDE2", "on")


def test_decide2_evidence_and_advisory_match_base_when_no_audit_fields(monkeypatch):
    runtime = _runtime(
        [{"id": "test-1", "request_id": "request-1", "description": "tests passed"}],
        ({"id": "host-1", "kind": "lint", "state": "pass"},),
        {"scope": "workspace", "request_id": "scope-request"},
    )

    actual = runtime_module._model_facing_check_evidence(runtime)
    expected = _base_model_facing_check_evidence(runtime)
    assert _bytes(actual) == _bytes(expected)

    actual_request, actual_body, actual_receipt, actual_notice = _run_against_golden(
        monkeypatch, runtime, actual)
    expected_runtime = _runtime(
        runtime.test_receipts, runtime.host_check_receipts, runtime._scope_lock_last_result)
    expected_request, expected_body, expected_receipt, expected_notice = _run_against_golden(
        monkeypatch, expected_runtime, expected)
    assert actual_request == expected_request
    assert _bytes(actual_body) == _bytes(expected_body)
    assert actual_receipt == expected_receipt
    assert actual_notice.encode("utf-8") == expected_notice.encode("utf-8")


def test_decide2_evidence_matches_base_after_only_audit_fields_are_removed(monkeypatch):
    runtime = _runtime(
        [{
            "id": "test-1", "request_id": "request-1", "description": "tests passed",
            "memory_limit_mechanism": "rlimit_data+rss_supervisor",
            "memory_limit_killed": False,
            "likely_initiator": "aggregate_rss_supervisor",
            "environment": {"packages": ["pytest"]},
            "observation_certificate": {
                "binding": {"revision": "host-only"}, "digest": "certificate",
            },
        }],
        [{"id": "host-1", "kind": "lint", "state": "pass",
          "cgroup_leaf_pattern": "private", "environment_revision": "host-revision"}],
        {"scope": "workspace", "memory_limit_reason": "audit-only"},
    )
    filtered_runtime = _runtime(
        _without_audit_fields(runtime.test_receipts),
        _without_audit_fields(runtime.host_check_receipts),
        _without_audit_fields(runtime._scope_lock_last_result),
    )

    actual = runtime_module._model_facing_check_evidence(runtime)
    expected = _base_model_facing_check_evidence(filtered_runtime)
    assert _bytes(actual) == _bytes(expected)

    actual_request, actual_body, actual_receipt, actual_notice = _run_against_golden(
        monkeypatch, runtime, actual)
    expected_request, expected_body, expected_receipt, expected_notice = _run_against_golden(
        monkeypatch, filtered_runtime, expected)
    assert actual_request == expected_request
    assert _bytes(actual_body) == _bytes(expected_body)
    assert actual_receipt == expected_receipt
    assert actual_notice.encode("utf-8") == expected_notice.encode("utf-8")
