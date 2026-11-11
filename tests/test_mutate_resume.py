"""Mutation resume reconciliation runs before generic pending-intent refusal."""

from pathlib import Path
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_resume_reconciles_mutation_pending_before_refusal(tmp_path, monkeypatch):
    from stateframe import resume

    run = tmp_path / "run"
    run.mkdir()
    calls = []
    monkeypatch.setattr(resume, "_reconcile_mutations", lambda path, **_kw: calls.append(path))

    class Tape:
        def __init__(self, path):
            assert path == run

        def pending(self):
            assert calls == [run]
            return []

        def records(self):
            return []

    monkeypatch.setattr(resume, "ToolTape", Tape)
    assert resume._tool_safety(run, {"mutations": {"level": "prebuilt"}},
                               replay_external=True) == []


def test_reconcile_receives_launch_ledger_rows(tmp_path, monkeypatch):
    from stateframe import resume

    run = tmp_path / "run"
    run.mkdir()
    home = tmp_path / "home"
    seed = "a" * 32
    ledger = home / "rt" / "mutations" / seed
    ledger.mkdir(parents=True)
    (ledger / "launches.jsonl").write_text(json.dumps({"launch": "l1", "state": "REAPED"}) + "\n")
    (run / "spec.json").write_text(json.dumps({"run_identity_seed": seed,
                                                "environment": {"STATEFRAME_HOME": str(home)}}))
    monkeypatch.setattr(resume, "ToolTape", lambda _run: SimpleNamespace(records=lambda: []))
    captured = []
    module = ModuleType("stateframe.engine.kernel_mutate_records")
    module.Journal = lambda _run: object()
    module.reconcile = lambda _rows, _journal, ledger_rows, **_kw: (
        captured.extend(ledger_rows) or SimpleNamespace(refuse=None, actions=[]))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    resume._reconcile_mutations(run)
    assert captured == [{"launch": "l1", "state": "REAPED"}]


def test_corrupt_mutation_evidence_refuses_resume(tmp_path, monkeypatch):
    from stateframe import resume

    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({"run_identity_seed": "a" * 32,
                                                "environment": {"STATEFRAME_HOME": str(tmp_path)}}))
    monkeypatch.setattr(resume, "ToolTape", lambda _run: SimpleNamespace(records=lambda: []))
    module = ModuleType("stateframe.engine.kernel_mutate_records")
    module.Journal = lambda _run: object()
    module.reconcile = lambda *_args, **_kw: (_ for _ in ()).throw(ValueError("bad journal hash"))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(resume.ResumeRefused, match="bad journal hash"):
        resume._reconcile_mutations(run)


def test_effect_free_pending_mutation_is_completed_as_interrupted(tmp_path):
    from stateframe import resume
    from stateframe.engine.kernel_tool_tape import ToolTape

    run = tmp_path / "run"
    run.mkdir()
    (run / "spec.json").write_text(json.dumps({
        "run_identity_seed": "a" * 32,
        "environment": {"STATEFRAME_HOME": str(tmp_path / "home")}}))
    tape = ToolTape(run)
    tape.intent(call_id="call-pending", tool="mutate", args={"op": "catalog"},
                replay_class="observation", capability_digest="cap", definition_digest=None,
                env_identity="mutation-runtime/1", pre_manifest_digest="mutation-runtime/1")
    resume._reconcile_mutations(run)
    completed = ToolTape(run).lookup("call-pending")
    assert completed.completed
    assert completed.observation.startswith("REFUSED MUTATION_INTERRUPTED")


def test_resume_sweeps_interrupted_launch_before_completing_intent(tmp_path, monkeypatch):
    from stateframe import resume

    run = tmp_path / "run"
    run.mkdir()
    home = tmp_path / "home"
    seed = "a" * 32
    root = home / "rt" / "mutations" / seed
    root.mkdir(parents=True)
    (root / "launches.jsonl").write_text(json.dumps({"launch": "launch-1",
                                                       "state": "STARTED"}) + "\n")
    (run / "spec.json").write_text(json.dumps({"run_identity_seed": seed,
                                                "environment": {"STATEFRAME_HOME": str(home)}}))
    completed = []
    events = []

    class Tape:
        def __init__(self, path):
            assert path == run

        def lookup(self, call):
            assert call == "pending-call"
            return SimpleNamespace(completed=False)

        def complete(self, **fields):
            completed.append(fields)

    monkeypatch.setattr(resume, "ToolTape", Tape)
    records = ModuleType("stateframe.engine.kernel_mutate_records")
    records.Journal = lambda _run: SimpleNamespace(append=lambda *args, **kw: events.append((args, kw)))
    calls = []

    def reconcile(_rows, _journal, ledger_rows, **_kw):
        calls.append(ledger_rows)
        action = "cleanup" if ledger_rows else "interrupt"
        return SimpleNamespace(refuse=None, actions=[SimpleNamespace(
            kind=action, call_id="pending-call", detail="", evidence=None)])

    records.reconcile = reconcile
    monkeypatch.setitem(sys.modules, records.__name__, records)
    sweep_module = ModuleType("stateframe.engine.kernel_mutate_sweep")

    def sweep(parent, *, run_locks):
        assert parent == root.parent
        assert not run_locks(seed)
        assert run_locks("other-run")
        (root / "launches.jsonl").unlink()
        root.rmdir()
        return SimpleNamespace(cleaned=(seed,), unresolved=())

    sweep_module.sweep = sweep
    monkeypatch.setitem(sys.modules, sweep_module.__name__, sweep_module)
    resume._reconcile_mutations(run)
    assert len(calls) == 2 and calls[0] and calls[1] == []
    assert events[0][0] == ("INTERRUPTED",)
    assert completed[0]["call_id"] == "pending-call"
