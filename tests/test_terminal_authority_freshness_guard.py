"""Freeze the terminal authority/freshness acceptance boundary."""
from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import kernel_done_is_submit_authority as submit_authority
import kernel_done_readiness_authority as readiness_authority
import kernel_terminal_disposition as terminal_disposition


_AUTHORITY_ERRORS = (
    ("ImportError", ImportError),
    ("AttributeError", AttributeError),
    ("TypeError", TypeError),
    ("ValueError", ValueError),
)
_AUTHORITY_CASES = (*_AUTHORITY_ERRORS, ("none", None))
_FACT_STATES = ("fresh", "stale", "none", "missing")
_SCENARIOS = tuple(product(
    (name for name, _error in _AUTHORITY_CASES),
    _FACT_STATES,
    _FACT_STATES,
))
_EXPECTED_SCENARIO_COUNT = 80


def _legacy_result(freshness: str, acceptance: str) -> dict[str, object]:
    result: dict[str, object] = {
        "result": "DONE_ACCEPTED",
        "accepted": True,
        "artifact_digest": "candidate-1",
    }
    facts: dict[str, object] = {}
    if freshness != "missing":
        facts["last_check_result"] = {
            "current": {"fresh": True, "stale": False, "none": None}[freshness],
        }
    if acceptance != "missing":
        facts["accepted"] = {"fresh": True, "stale": False, "none": None}[acceptance]
    result["done_facts"] = facts
    return result


def _raise_authority(error_type: type[Exception]):
    def raise_error(_runtime: object) -> None:
        raise error_type("planted readiness authority failure")

    return raise_error


def _is_green(receipt: dict[str, object]) -> bool:
    return (
        receipt.get("accepted") is True
        or receipt.get("acceptance") == "ACCEPTED"
        or receipt.get("verification") == "VERIFIED"
    )


def _is_explicitly_safe(receipt: dict[str, object]) -> bool:
    return (
        receipt.get("accepted") is False
        and receipt.get("acceptance") == "UNACCEPTED"
        and receipt.get("verification") == "UNVERIFIED"
        and receipt.get("repair") in {
            terminal_disposition.UNRESOLVED,
            terminal_disposition.ENVIRONMENT_BLOCKED,
        }
    )


def test_terminal_authority_freshness_matrix(monkeypatch):
    """Authority errors or any stale/unknown fact must never yield a green receipt."""
    scanned = []
    violations = []
    error_types = dict(_AUTHORITY_ERRORS)

    for scenario in _SCENARIOS:
        authority_name, freshness, acceptance = scenario
        with monkeypatch.context() as patch:
            patch.setattr(submit_authority, "terminal_from_seal", lambda _runtime: None)
            if authority_name == "none":
                # No exception and no cached authority decision is the legacy-mode control.
                patch.setattr(readiness_authority, "cached_decision", lambda _runtime: None)
            else:
                patch.setattr(
                    readiness_authority,
                    "cached_decision",
                    _raise_authority(error_types[authority_name]),
                )

            runtime = SimpleNamespace(stateframe_terminal_lattice_mode="off")
            result = _legacy_result(freshness, acceptance)
            scanned.append(scenario)
            try:
                receipt = terminal_disposition.mint(runtime, result)
            except Exception as error:  # report unexpected authority behavior as a matrix violation
                violations.append((scenario, f"raised {type(error).__name__}: {error}"))
                continue

            authority_errored = authority_name != "none"
            facts_proven = freshness == "fresh" and acceptance == "fresh"
            expected_accepted = not authority_errored and facts_proven
            if receipt.get("schema") != terminal_disposition.SCHEMA:
                violations.append((scenario, "missing terminal disposition receipt"))
            elif expected_accepted:
                if not (
                    receipt.get("accepted") is True
                    and receipt.get("acceptance") == "ACCEPTED"
                    and receipt.get("verification") == "VERIFIED"
                ):
                    violations.append((scenario, f"valid control was not accepted: {receipt!r}"))
            elif _is_green(receipt) or not _is_explicitly_safe(receipt):
                violations.append((scenario, f"unsafe disposition: {receipt!r}"))

    assert _EXPECTED_SCENARIO_COUNT == len(_SCENARIOS)
    assert len(scanned) == _EXPECTED_SCENARIO_COUNT and len(set(scanned)) == _EXPECTED_SCENARIO_COUNT
    assert not violations, (
        f"terminal authority/freshness guard scanned={len(scanned)} "
        f"violated={len(violations)}; violations={violations!r}"
    )


def test_terminal_authority_value_error_control(monkeypatch):
    """A ValueError cannot fall back to a legacy accepted result."""
    monkeypatch.setattr(submit_authority, "terminal_from_seal", lambda _runtime: None)
    monkeypatch.setattr(
        readiness_authority,
        "cached_decision",
        _raise_authority(ValueError),
    )
    result = _legacy_result("fresh", "fresh")
    assert result["accepted"] is True

    receipt = terminal_disposition.mint(
        SimpleNamespace(stateframe_terminal_lattice_mode="off"), result,
    )

    assert receipt["accepted"] is False
    assert receipt["acceptance"] == "UNACCEPTED"
    assert receipt["verification"] == "UNVERIFIED"
    assert receipt["repair"] == terminal_disposition.UNRESOLVED
    assert "DONE_READINESS_AUTHORITY_UNAVAILABLE" in receipt["unresolved_ids"]
