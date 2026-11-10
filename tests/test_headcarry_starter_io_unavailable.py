"""Planted live failures must leave a journaled, reason-coded evidence gap."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import kernel_starter_conflicts as starter
import kernel_stateframe_headcarry as headcarry
import kernel_stateframe_framefit5 as framefit5
from kernel_obligation_assess import UnavailableEvidence, _unavailable_evidence
import pytest


def _journal():
    events = []
    return SimpleNamespace(emit=lambda kind, **fields: events.append((kind, fields))), events


def _gap(value, events, code, producer):
    assert isinstance(value, UnavailableEvidence)
    assert value.reason_code == code
    assert events == [("host_obligation_evidence_failure", {
        "producer": producer, "dispatch": False, "reason_code": code,
    })]


@pytest.mark.parametrize("failure,code", [
    ("read", "HEADCARRY_SOURCE_UNAVAILABLE"),
    ("stat", "HEADCARRY_STAT_UNAVAILABLE"),
    ("missing", "HEADCARRY_STAT_UNAVAILABLE"),
    ("root", "HEADCARRY_ROOT_UNAVAILABLE"),
])
def test_planted_live_headcarry_gap(monkeypatch, tmp_path, failure, code):
    journal, events = _journal()
    (tmp_path / "a.py").write_text("x = 1\n")
    root = tmp_path
    path = "a.py"
    if failure == "read":
        import kernel_stateframe_action_ledger as ledger
        monkeypatch.setattr(ledger, "_read_open_site_file_bytes",
                            lambda *_args: (_ for _ in ()).throw(OSError("read failed")))
    elif failure == "stat":
        original = Path.lstat
        def broken_stat(path, *args, **kwargs):
            if path.name == "a.py":
                raise OSError("stat failed")
            return original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "lstat", broken_stat)
    elif failure == "missing":
        path = "missing.py"
    else:
        root = tmp_path / "missing"
    result = headcarry.render_headcarry(root, [{"path": path}], (), journal=journal)
    _gap(result, events, code, "kernel_stateframe_headcarry")
    events.clear()
    monkeypatch.setenv("STATEFRAME_HEADCARRY", "on")
    runtime = SimpleNamespace(journal=journal)
    frame = {}
    framefit5._update_headcarry(frame, runtime=runtime, root=root,
                                open_sites=[{"path": path}], applied_edits=(),
                                messages=())
    _gap(runtime._framefit5_headcarry_unavailable, events, code,
         "kernel_stateframe_headcarry")
    assert "headcarry" not in frame


def test_planted_live_starter_snapshot_gap(monkeypatch, tmp_path):
    journal, events = _journal()
    (tmp_path / "a.py").write_text("x = 1\n")
    original = Path.read_bytes
    def broken_read(path):
        if path.name == "a.py":
            raise OSError("read failed")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", broken_read)
    _gap(starter.workspace_snapshot(tmp_path, journal=journal), events,
         "STARTER_SNAPSHOT_UNAVAILABLE", "kernel_starter_conflicts")
    events.clear()
    tracker = starter.StarterConflictTracker("ticket", [object()], starter_snapshot="old",
                                            starter_facts=(), root=tmp_path, emit=journal)
    _gap(tracker.refresh_from_root(force=True, journal=journal), events,
         "STARTER_SNAPSHOT_UNAVAILABLE", "kernel_starter_conflicts")


def test_planted_live_runner_type_error_gap(tmp_path):
    journal, events = _journal()
    def broken(_root, _checks, **_kwargs):
        raise TypeError("runner bug")
    _gap(starter._call_runner(broken, tmp_path, (), journal=journal, snapshot="r1"),
         events, "STARTER_RUNNER_UNAVAILABLE", "kernel_starter_conflicts")


def test_planted_live_refresh_import_gap(monkeypatch, tmp_path):
    journal, events = _journal()
    tracker = starter.StarterConflictTracker("ticket", [object()], starter_snapshot="old",
                                            starter_facts=(), root=tmp_path, emit=journal)
    original_import = __import__
    def broken_import(name, *args, **kwargs):
        if name == "kernel_obligation_assess":
            raise ImportError("checks unavailable")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", broken_import)
    _gap(tracker.refresh_from_root(force=True, journal=journal), events,
         "STARTER_CHECKS_UNAVAILABLE", "kernel_starter_conflicts")


def test_unexpected_headcarry_error_propagates(monkeypatch, tmp_path):
    import kernel_stateframe_action_ledger as ledger
    (tmp_path / "a.py").write_text("x = 1\n")
    journal, events = _journal()
    monkeypatch.setattr(ledger, "_read_open_site_file_bytes",
                        lambda *_args: (_ for _ in ()).throw(LookupError("unexpected")))
    with pytest.raises(LookupError, match="unexpected"):
        headcarry.render_headcarry(tmp_path, [{"path": "a.py"}], (), journal=journal)
    assert events == []


def test_two_argument_runner_is_called_once(tmp_path):
    calls = []
    def runner(root, checks):
        calls.append((root, checks))
        return ()
    assert starter._call_runner(runner, tmp_path, (), snapshot="r1") == ()
    assert calls == [(tmp_path, ())]


def test_registered_runner_gap_keeps_its_journal_reason(tmp_path):
    journal, events = _journal()
    def runner(_root, _checks, *, snapshot, environment, cache, journal):
        return _unavailable_evidence(journal, "REGISTERED_CHECK_UNAVAILABLE",
                                     producer="kernel_starter_conflicts")
    tracker = starter.StarterConflictTracker("ticket", [object()], starter_snapshot="old",
                                            starter_facts=(), root=tmp_path, emit=journal)
    _gap(tracker.refresh_from_root(runner=runner, force=True, journal=journal), events,
         "REGISTERED_CHECK_UNAVAILABLE", "kernel_starter_conflicts")
