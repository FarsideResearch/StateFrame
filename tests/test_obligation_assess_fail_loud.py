"""High-risk host obligation evidence failures must stop request construction."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import kernel_obligation_assess as obligations
import kernel_green_lock_repair
import kernel_oracle_grounding
import pytest


def _journal() -> tuple[SimpleNamespace, list[tuple[str, dict[str, object]]]]:
    events: list[tuple[str, dict[str, object]]] = []
    return SimpleNamespace(emit=lambda kind, **data: events.append((kind, data))), events


def _assert_failure(events: list[tuple[str, dict[str, object]]], code: str) -> None:
    assert events == [("host_obligation_evidence_failure", {
        "producer": "kernel_obligation_assess", "dispatch": False,
        "reason_code": code,
    })]


def _check() -> obligations.CheckDefinition:
    return obligations.CheckDefinition(
        "check-1", "obligation-1", "OUTPUT_EQUALS",
        entry_point="app:value", expected=7,
    )


@pytest.mark.parametrize("with_journal", [False, True])
def test_oracle_grounding_failure_cannot_certify_obligation(
    monkeypatch: pytest.MonkeyPatch, with_journal: bool,
) -> None:
    journal, events = _journal()
    check = _check()
    spec = obligations.ObligationSpec("obligation-1", acceptance_policy=(check.check_id,))

    def broken_grounding(*args: object, **kwargs: object) -> None:
        raise ValueError("grounding unavailable")

    monkeypatch.setenv("STATEFRAME_ORACLE_GROUNDING", "on")
    monkeypatch.setattr(kernel_oracle_grounding, "ground_expectation", broken_grounding)
    with pytest.raises(RuntimeError, match="ORACLE_GROUNDING_UNAVAILABLE") as caught:
        obligations.assess_obligation(
            spec, checks=(check,), **({"journal": journal} if with_journal else {}))
    assert isinstance(caught.value.__cause__, ValueError)
    if with_journal:
        _assert_failure(events, "ORACLE_GROUNDING_UNAVAILABLE")


@pytest.mark.parametrize("with_journal", [False, True])
def test_green_lock_classification_failure_cannot_run_stronger_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_journal: bool,
) -> None:
    journal, events = _journal()

    def broken_classification(*args: object, **kwargs: object) -> None:
        raise TypeError("policy unavailable")

    monkeypatch.setenv("STATEFRAME_GREEN_LOCK_REPAIR", "on")
    monkeypatch.setattr(kernel_green_lock_repair, "classify_check", broken_classification)
    with pytest.raises(RuntimeError, match="GREEN_LOCK_CLASSIFICATION_UNAVAILABLE") as caught:
        obligations.run_check(
            tmp_path, _check(), snapshot="fixed-snapshot",
            **({"journal": journal} if with_journal else {}))
    assert isinstance(caught.value.__cause__, TypeError)
    if with_journal:
        _assert_failure(events, "GREEN_LOCK_CLASSIFICATION_UNAVAILABLE")


def test_registered_checks_forward_one_classification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal, events = _journal()

    def broken_classification(*args: object, **kwargs: object) -> None:
        raise ValueError("policy unavailable")

    monkeypatch.setenv("STATEFRAME_GREEN_LOCK_REPAIR", "on")
    monkeypatch.setattr(kernel_green_lock_repair, "classify_check", broken_classification)
    with pytest.raises(RuntimeError, match="GREEN_LOCK_CLASSIFICATION_UNAVAILABLE"):
        obligations.run_registered_checks(
            tmp_path, (_check(),), snapshot="fixed-snapshot", journal=journal)
    _assert_failure(events, "GREEN_LOCK_CLASSIFICATION_UNAVAILABLE")


def test_unexpected_classification_error_propagates_without_evidence_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal, events = _journal()

    def unexpected(*args: object, **kwargs: object) -> None:
        raise LookupError("unexpected classifier fault")

    monkeypatch.setenv("STATEFRAME_GREEN_LOCK_REPAIR", "on")
    monkeypatch.setattr(kernel_green_lock_repair, "classify_check", unexpected)
    with pytest.raises(LookupError, match="unexpected classifier fault"):
        obligations.run_check(tmp_path, _check(), snapshot="fixed-snapshot", journal=journal)
    assert events == []
