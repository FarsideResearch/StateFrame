import pytest

from test_host_eff_09_shared_check_cache import (
    _check_events, _closure, _repair, _runtime, probe,  # noqa: F401
)


@pytest.mark.parametrize("passed", [True, False])
def test_only_a_passing_proof_skips_the_second_probe(tmp_path, monkeypatch, probe, passed):
    probe.ok = passed
    probe.observed = 7 if passed else 8
    results = []
    for mode in ("off", "on"):
        monkeypatch.setenv("STATEFRAME_SHARED_CHECK_PASS_ONLY", mode)
        runtime, events = _runtime(tmp_path / mode, monkeypatch, "off")
        probe.runs.clear()
        _repair(runtime)
        _closure(runtime)
        reused = mode == "on" and passed
        assert len(probe.runs) == (1 if reused else 2)
        assert _check_events(events)[-1] == ("host_obligation_check", reused)
        results.append([(row.status, row.observed) for row in runtime.host_check_receipts])
    assert results[0] == results[1]


@pytest.mark.parametrize("passed", [True, False])
def test_rehydrated_receipt_retains_passing_only_policy(tmp_path, monkeypatch, probe, passed):
    from kernel_measurement import AuditJournal

    monkeypatch.setenv("STATEFRAME_SHARED_CHECK_PASS_ONLY", "on")
    probe.ok = passed
    probe.observed = 7 if passed else 8
    initial, events = _runtime(tmp_path / "initial", monkeypatch, "off")
    _repair(initial)
    rows = [{"kind": kind, **fields} for kind, fields in events]
    monkeypatch.setattr(AuditJournal, "records", lambda _: iter(rows), raising=False)

    resumed, closure_events = _runtime(tmp_path / "resumed", monkeypatch, "off")
    _closure(resumed)
    assert _check_events(closure_events)[-1] == ("host_obligation_check", passed)
    assert len(probe.runs) == (1 if passed else 2)
