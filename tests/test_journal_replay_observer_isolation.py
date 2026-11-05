"""Replay ordinary observer failures retain the 15de58e isolate-and-continue contract."""
from __future__ import annotations

import copy

import pytest

from kernel_continuation_journal import ContinuationJournal


def _replay_observer_failure(tmp_path, name: str) -> dict[str, object]:
    journal = ContinuationJournal(
        tmp_path, f"observer-isolation-{name}", {"origin": "observer-isolation"},
    )
    try:
        for ordinal in (1, 2, 3):
            journal.emit("observer_probe", ordinal=ordinal)

        delivered: list[tuple[int, str, int | None]] = []

        def observe(record: dict[str, object]) -> None:
            kind = str(record.get("kind"))
            ordinal_value = record.get("ordinal")
            ordinal = ordinal_value if type(ordinal_value) is int else None
            delivered.append((int(record["sequence"]), kind, ordinal))
            if kind == "observer_probe" and ordinal == 2:
                raise RuntimeError("observer rejected row")

        raised: tuple[type[Exception], str] | None = None
        try:
            journal.subscribe(name, observe)
        except Exception as error:
            raised = (type(error), str(error))

        failure_events = [
            {
                key: row.get(key)
                for key in ("kind", "subscriber", "event_kind", "event_sequence", "error_type", "error")
            }
            for row in journal.records()
            if row.get("kind") == "journal_observer_failed"
        ]
        stats = journal.publication_stats()
        return {
            "raised": raised,
            "delivered": delivered,
            "delivered_sequences": [sequence for sequence, _kind, _ordinal in delivered],
            "probe_ordinals": [ordinal for _sequence, kind, ordinal in delivered
                               if kind == "observer_probe"],
            "failure_events": failure_events,
            "observer_failures": stats.get("observer_failures", 0),
            "replayed": stats["replayed"],
        }
    finally:
        journal.close()


def _assert_15de58e_contract(result: dict[str, object]) -> None:
    assert result["raised"] is None, "ordinary observer exception escaped replay"
    assert result["delivered_sequences"] == [1, 2, 3, 4, 5, 6]
    assert result["probe_ordinals"] == [1, 2, 3], "replay did not continue after the raising callback"
    assert result["delivered"][-1][1] == "journal_observer_failed"
    assert result["observer_failures"] == 1
    assert result["replayed"] == 5
    assert result["failure_events"] == [{
        "kind": "journal_observer_failed",
        "subscriber": "observer-isolation",
        "event_kind": "observer_probe",
        "event_sequence": 4,
        "error_type": "RuntimeError",
        "error": "observer rejected row",
    }]


def test_replay_observer_exception_matches_15de58e(tmp_path):
    result = _replay_observer_failure(tmp_path, "observer-isolation")

    _assert_15de58e_contract(result)


def test_propagating_control_fails_the_15de58e_contract(tmp_path, monkeypatch):
    def propagate(_journal, _name, observer, record, *, on_unobserved=None):
        observer(copy.deepcopy(record))

    monkeypatch.setattr(ContinuationJournal, "_notify_isolated", propagate)
    result = _replay_observer_failure(tmp_path, "observer-isolation")

    with pytest.raises(AssertionError):
        _assert_15de58e_contract(result)
