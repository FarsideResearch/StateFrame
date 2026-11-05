"""Terminal receipts reject unavailable authority and unproven completion facts."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from types import SimpleNamespace

import pytest

import kernel_completion_owner as completion_owner
import kernel_done_is_submit_authority as submit_authority
import kernel_done_readiness_authority as readiness_authority
import kernel_terminal_disposition as terminal_disposition
import kernel_terminal_lattice as terminal_lattice
from kernel_terminal_disposition import mint


# This reviewed inventory is independent of the production exception and fact handling.
_AUTHORITY_ERRORS = (
    ("ImportError", ImportError),
    ("AttributeError", AttributeError),
    ("TypeError", TypeError),
    ("ValueError", ValueError),
)
_FRESHNESS_STATES = ("fresh", "stale", "unknown", "missing")
_ACCEPTANCE_STATES = ("accepted", "rejected", "unknown", "missing")
_FACT_SCENARIOS = frozenset(product(_FRESHNESS_STATES, _ACCEPTANCE_STATES))
_EXPECTED_SCENARIOS = frozenset(
    {
        ("mint", error_name, freshness, acceptance, "off")
        for (error_name, _error_type), (freshness, acceptance)
        in product(_AUTHORITY_ERRORS, _FACT_SCENARIOS)
    }
    | {
        ("mint", error_name, freshness, acceptance, "prior-accepted")
        for (error_name, _error_type), (freshness, acceptance)
        in product(_AUTHORITY_ERRORS, _FACT_SCENARIOS)
    }
    | {
        ("mint", "success", freshness, acceptance, "off")
        for freshness, acceptance in _FACT_SCENARIOS
    }
    | {
        ("mint", "success", freshness, acceptance, "prior-accepted")
        for freshness, acceptance in _FACT_SCENARIOS
    }
    | {("owner", "success", "unknown", "unknown", "off")}
)
_EXPECTED_COUNT = 161
_RECEIPT_SCHEMA = "stateframe-terminal-disposition/1"


def _legacy_result(freshness: str, acceptance: str) -> dict[str, object]:
    result: dict[str, object] = {
        "result": "DONE_ACCEPTED",
        "accepted": True,
        "artifact_digest": "artifact-1",
    }
    facts: dict[str, object] = {}
    if acceptance != "missing":
        facts["accepted"] = {
            "accepted": True,
            "rejected": False,
            "unknown": None,
        }[acceptance]
    if freshness == "missing":
        facts["last_check_result"] = {}
    else:
        facts["last_check_result"] = {
            "fresh": {"current": True},
            "stale": {"current": False},
            "unknown": {"current": None},
        }[freshness]
    if freshness != "missing" or acceptance != "missing":
        result["done_facts"] = facts
    return result


def _raising_authority(error_type: type[Exception]):
    def raise_error(_runtime: object) -> None:
        raise error_type("planted authority lookup failure")

    return raise_error


def _seed_prior_acceptance(runtime: object) -> None:
    prior_receipt: dict[str, object] = {
        "schema": _RECEIPT_SCHEMA,
        "artifact_digest": "artifact-1",
        "stop_reason": "SUBMITTED",
        "delivery": "STORED",
        "artifact": "STORED",
        "verification": "VERIFIED",
        "acceptance": "ACCEPTED",
        "certification": "UNCERTIFIED",
        "repair": "VERIFIED_FIXED",
        "accepted": True,
    }
    runtime._terminal_lattice_state = {
        "accepted": True,
        "delivery": "STORED",
        "artifact": "STORED",
        "stop_reason": "SUBMITTED",
        "stop_rank": terminal_lattice.STOP_REASON_RANKS["SUBMITTED"],
        "status_values": {},
        "artifact_digest": "artifact-1",
        "artifact_digests": ["artifact-1"],
        "receipt": prior_receipt,
    }
    runtime._terminal_disposition_receipt = prior_receipt


def _invoke_scenario(runtime: object, result: dict[str, object], scenario):
    path, _authority_name, _freshness, _acceptance, _lattice_state = scenario
    if path == "mint":
        return mint(runtime, result)

    prior_result = _legacy_result("fresh", "accepted")
    prior_facts = prior_result["done_facts"]
    assert isinstance(prior_facts, dict)
    prior_facts["evidence_digest"] = "prior-evidence"
    runtime._terminal_disposition_receipt = mint(runtime, prior_result)
    result["certified"] = False
    return completion_owner.seal_terminal_disposition(runtime, result)


def _as_receipt(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if value.get("schema") == _RECEIPT_SCHEMA:
        return value
    receipt = value.get("terminal_disposition")
    return receipt if isinstance(receipt, Mapping) else None


def _expected_unresolved(scenario) -> set[str]:
    _path, authority_name, freshness, acceptance, _lattice_state = scenario
    expected: set[str] = set()
    if authority_name != "success":
        expected.add("DONE_READINESS_AUTHORITY_UNAVAILABLE")
    if freshness == "stale":
        expected.add("DONE_FACTS_NOT_CURRENT")
    elif freshness in {"unknown", "missing"}:
        expected.add("DONE_FACTS_CURRENTNESS_UNKNOWN")
    if acceptance == "rejected":
        expected.add("DONE_FACTS_REJECTED")
    elif acceptance in {"unknown", "missing"}:
        expected.add("DONE_FACTS_ACCEPTANCE_UNKNOWN")
    return expected


def _scan_matrix(monkeypatch, invoke_override=None):
    scanned = []
    violations = []
    error_types = dict(_AUTHORITY_ERRORS)

    for scenario in sorted(_EXPECTED_SCENARIOS):
        path, authority_name, freshness, acceptance, lattice_state = scenario
        with monkeypatch.context() as patch:
            patch.setattr(submit_authority, "terminal_from_seal", lambda _runtime: None)
            if authority_name == "success":
                patch.setattr(
                    readiness_authority,
                    "cached_decision",
                    lambda _runtime: SimpleNamespace(to_dict=lambda: {}),
                )
                patch.setattr(
                    readiness_authority,
                    "apply_terminal_result",
                    lambda _runtime, result, _decision: result,
                )
            else:
                patch.setattr(
                    readiness_authority,
                    "cached_decision",
                    _raising_authority(error_types[authority_name]),
                )
            runtime = SimpleNamespace(
                stateframe_terminal_lattice_mode=(
                    "on" if lattice_state == "prior-accepted" else "off"
                ),
            )
            if lattice_state == "prior-accepted":
                _seed_prior_acceptance(runtime)
            result = _legacy_result(freshness, acceptance)
            scanned.append(scenario)
            try:
                if invoke_override is None:
                    returned = _invoke_scenario(runtime, result, scenario)
                else:
                    returned = invoke_override(runtime, result, scenario, _invoke_scenario)
            except Exception as exc:
                violations.append((scenario, f"raised {type(exc).__name__}: {exc}"))
                continue

            receipt = _as_receipt(returned)
            problems = []
            expected_accepted = (
                authority_name == "success"
                and freshness == "fresh"
                and acceptance == "accepted"
            )
            if receipt is None:
                problems.append("missing terminal receipt")
            else:
                if receipt.get("accepted") is not expected_accepted:
                    problems.append(f"accepted={receipt.get('accepted')!r}")
                expected_acceptance = "ACCEPTED" if expected_accepted else "UNACCEPTED"
                if receipt.get("acceptance") != expected_acceptance:
                    problems.append(f"acceptance={receipt.get('acceptance')!r}")
                expected_verification = "VERIFIED" if expected_accepted else "UNVERIFIED"
                if receipt.get("verification") != expected_verification:
                    problems.append(f"verification={receipt.get('verification')!r}")
                unresolved = receipt.get("unresolved_ids", ())
                actual_unresolved = (
                    {str(item) for item in unresolved}
                    if isinstance(unresolved, Sequence)
                    and not isinstance(unresolved, (str, bytes, bytearray))
                    else set()
                )
                if actual_unresolved != _expected_unresolved(scenario):
                    problems.append(f"unresolved_ids={sorted(actual_unresolved)!r}")
            if lattice_state == "prior-accepted":
                lattice_state_value = getattr(runtime, "_terminal_lattice_state", None)
                if (not isinstance(lattice_state_value, Mapping)
                        or lattice_state_value.get("accepted") is not expected_accepted):
                    problems.append("prior lattice acceptance was not reconciled")
            if path == "owner" and isinstance(returned, Mapping):
                if returned.get("accepted") is not expected_accepted:
                    problems.append(f"projected accepted={returned.get('accepted')!r}")
            if problems:
                violations.append((scenario, "; ".join(problems)))

    return scanned, violations


def _assert_guard(scanned, violations) -> str:
    summary = (
        f"GUARD e8_terminal_authority: expected={_EXPECTED_COUNT} "
        f"scanned={len(scanned)} violated={len(violations)}"
    )
    print(summary)
    assert _EXPECTED_COUNT == len(_EXPECTED_SCENARIOS)
    assert len(scanned) == _EXPECTED_COUNT and set(scanned) == _EXPECTED_SCENARIOS, (
        f"{summary}; expected and scanned scenario inventories differ"
    )
    assert not violations, f"{summary}; violations={violations!r}"
    return summary


def _assert_planted_failure(monkeypatch, capsys, invoke_override=None) -> str:
    scanned, violations = _scan_matrix(monkeypatch, invoke_override)
    with pytest.raises(AssertionError, match="violated=1"):
        _assert_guard(scanned, violations)
    summary = (
        f"GUARD e8_terminal_authority: expected={_EXPECTED_COUNT} "
        f"scanned={_EXPECTED_COUNT} violated=1"
    )
    assert capsys.readouterr().out.strip() == summary
    return summary


def test_terminal_authority_freshness_guard(monkeypatch, capsys):
    scanned, violations = _scan_matrix(monkeypatch)
    summary = _assert_guard(scanned, violations)
    assert summary == "GUARD e8_terminal_authority: expected=161 scanned=161 violated=0"
    assert capsys.readouterr().out.strip() == summary


def test_terminal_authority_guard_catches_planted_value_error_fallback(monkeypatch, capsys):
    control_scenario = ("mint", "ValueError", "fresh", "accepted", "off")
    control_observed: list[bool] = []

    def planted_fallback(runtime, result, scenario, default_invoke):
        if scenario != control_scenario:
            return default_invoke(runtime, result, scenario)
        try:
            readiness_authority.cached_decision(runtime)
        except ValueError:
            control_observed.append(True)
            return {
                "schema": _RECEIPT_SCHEMA,
                "accepted": result.get("accepted") is True,
                "acceptance": "ACCEPTED",
                "verification": "VERIFIED",
                "unresolved_ids": [],
            }
        raise AssertionError("the ValueError control did not reach the authority boundary")

    _assert_planted_failure(monkeypatch, capsys, planted_fallback)
    assert control_observed == [True]


def test_terminal_authority_guard_catches_successful_lookup_fact_fallback(monkeypatch, capsys):
    control_scenario = ("mint", "success", "unknown", "accepted", "off")
    control_observed: list[bool] = []

    def planted_fact_fallback(runtime, result, scenario, default_invoke):
        if scenario != control_scenario:
            return default_invoke(runtime, result, scenario)
        decision = readiness_authority.cached_decision(runtime)
        control_observed.append(decision is not None)
        return {
            "schema": _RECEIPT_SCHEMA,
            "accepted": True,
            "acceptance": "ACCEPTED",
            "verification": "VERIFIED",
            "unresolved_ids": [],
        }

    _assert_planted_failure(monkeypatch, capsys, planted_fact_fallback)
    assert control_observed == [True]


def test_terminal_authority_guard_catches_planted_lattice_acceptance(monkeypatch, capsys):
    control_scenario = ("mint", "ValueError", "fresh", "accepted", "prior-accepted")
    control_observed: list[bool] = []
    control_active = False
    original_join_fields = terminal_lattice._join_fields

    def planted_monotone_acceptance(candidate, previous):
        joined, refusals = original_join_fields(candidate, previous)
        if (control_active
                and previous.get("accepted") is True
                and (candidate.get("accepted") is False
                     or candidate.get("acceptance") == "UNACCEPTED")):
            control_observed.append(True)
            joined["accepted"] = True
            joined["acceptance"] = "ACCEPTED"
        return joined, refusals

    def invoke_with_planted_lattice(runtime, result, scenario, default_invoke):
        nonlocal control_active
        if scenario != control_scenario:
            return default_invoke(runtime, result, scenario)
        control_active = True
        try:
            return default_invoke(runtime, result, scenario)
        finally:
            control_active = False

    monkeypatch.setattr(terminal_lattice, "_join_fields", planted_monotone_acceptance)
    _assert_planted_failure(monkeypatch, capsys, invoke_with_planted_lattice)
    assert control_observed


def test_terminal_authority_guard_catches_planted_completion_receipt_reuse(monkeypatch, capsys):
    control_observed: list[bool] = []

    def planted_stale_reuse(runtime, result):
        prior = getattr(runtime, "_terminal_disposition_receipt", None)
        if not isinstance(prior, Mapping):
            raise AssertionError("the completion owner control had no prior receipt")
        control_observed.append(True)
        return terminal_disposition.project(result, prior, runtime=runtime)

    monkeypatch.setattr(completion_owner, "seal_terminal_disposition", planted_stale_reuse)
    _assert_planted_failure(monkeypatch, capsys)
    assert control_observed == [True]
