"""Independent context-alarm gates for the real K1/K2 qualification cases."""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
from test_read_admission_qualification import seal_case


def _alarm(monkeypatch):
    reference = os.environ.get("SF_TEST_CONTEXT_ALARM_SOURCE")
    if reference:
        spec = importlib.util.spec_from_file_location("wsa_admission_context_alarm", reference)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        return module
    return pytest.importorskip("kernel_context_alarm", reason="context alarm is unavailable")


@pytest.mark.parametrize("scenario", ["K1", "K2"])
def test_k1_k2_context_health(tmp_path, monkeypatch, scenario):
    # Resolve the independent oracle first. Missing coordination artifacts are
    # explicit skips, never an untested successful invariant assertion.
    alarm = _alarm(monkeypatch)
    record, capture, runtime = seal_case(tmp_path, monkeypatch, scenario)
    receipts = [c.receipt() for c in runtime.read_capture_ledger.all()]
    index = alarm.index_request(record.wire)
    assert index.valid
    prepared = next(row for row in _events(capture.journal.path)
                    if row.get("kind") == "working_set_prepared")
    health = alarm.assess(index, receipts, {r.path: r.revision for r in receipts},
                         alarm.AlarmState(), max(r.due_request for r in receipts),
                         fits_without_degrade=prepared["fits_without_degrade"])
    for invariant in ("I2", "I4", "I6"):
        assert health.invariants[invariant] == {"violations": 0, "unverifiable": 0}


def _events(path):
    import json
    return [json.loads(line) for line in path.read_bytes().splitlines()]
