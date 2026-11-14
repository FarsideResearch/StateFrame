"""Stable-wire publication refusals explain epoch drift without changing request authority."""
import copy
import io
from types import SimpleNamespace

import pytest

import kernel_reason_log as rl
import kernel_stable_wire as stable
from kernel_final_wire_allocation import _stable_wire_record_for_publication
from kernel_request_seal import RequestSealError


@pytest.fixture
def reason_log(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    stream = io.StringIO()
    journal = SimpleNamespace(events=[])
    journal.emit = lambda *args, **kwargs: journal.events.append((args, kwargs))
    rl.configure(level="DEBUG", stream=stream, journal=journal)
    yield stream, journal
    rl.reset_for_tests()


def test_publication_epoch_refusal_logs_once_without_journal_or_mutation(monkeypatch, reason_log):
    stream, journal = reason_log
    body = {"messages": [{"role": "user", "content": "unchanged"}]}
    before = copy.deepcopy(body)
    record = {"reserves": [], "trim_count": 0, "epoch": 1}
    candidate = SimpleNamespace(_stable_wire_reserves_record=record)
    monkeypatch.setattr(stable, "recorded_window_layout", lambda _messages: ([], 2))

    for _ in range(2):
        with pytest.raises(RequestSealError, match="stable wire trim epoch changed before publication"):
            _stable_wire_record_for_publication(candidate, body)

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("[stateframe] ERROR request_seal:")
    assert "trim record epoch 1, qualified layout epoch 2" in lines[0]
    cause = rl.run_summary()["by_cause"]["invariant:stable wire trim epoch changed before publication"]
    assert cause["count"] == 2 and cause["level"] == "ERROR"
    assert journal.events == []
    assert body == before
    assert record == {"reserves": [], "trim_count": 0, "epoch": 1}


@pytest.mark.parametrize("has_record,has_layout", [(False, True), (True, False), (True, True)])
def test_publication_without_epoch_drift_stays_silent(monkeypatch, reason_log, has_record, has_layout):
    stream, journal = reason_log
    record = {"reserves": [], "trim_count": 0, "epoch": 1}
    candidate = SimpleNamespace(_stable_wire_reserves_record=record if has_record else None)
    monkeypatch.setattr(stable, "recorded_window_layout",
                        lambda _messages: ([], 1) if has_layout else None)

    result = _stable_wire_record_for_publication(candidate, {"messages": []})

    assert result == (record if has_record and has_layout else None)
    if result is not None:
        assert result is not record
    assert stream.getvalue() == "" and journal.events == []
    assert rl.run_summary()["total"] == 0
