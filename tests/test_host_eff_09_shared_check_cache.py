"""STATEFRAME_SHARED_CHECK_CACHE: one exact-instance proof cache for repair and closure checks.

Off must keep the two caches separate, so each path runs its own check. On must let
one path reuse a proof the other path ran for the same instance key. Both journal
event kinds keep their shape either way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_obligation_assess as obligations
import kernel_safe_probe
from kernel_counterexample_visibility import collect_counterexamples
from kernel_measurement import AuditJournal
from kernel_stateframe_runtime import StateFrameRuntime

SNAPSHOT = "sha256:" + "a" * 64  # workspace snapshots are sha256 identities in the engine
CHECK_EVENTS = ("host_repair_check", "host_obligation_check")


@dataclass
class Probe:
    """Fake of the inner operation every fresh check run performs (the sandboxed probe)."""

    ok: bool = True
    observed: object = 7
    runs: list[list[str]] = field(default_factory=list)


def _check() -> obligations.CheckDefinition:
    return obligations.CheckDefinition(
        "check-1", "obligation-1", "OUTPUT_EQUALS", entry_point="app:value", expected=7)


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        checks=(_check(),), specs=(SimpleNamespace(obligation_id="obligation-1"),),
        path="<test-bundle>", provenance={},
    )


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Probe:
    fake = Probe()

    def run_probe(argv, workspace, *, timeout=10, **_kwargs):
        fake.runs.append(list(argv))
        return SimpleNamespace(
            timed_out=False, exit_code=0,
            output=json.dumps({"ok": fake.ok, "observed": fake.observed}).encode())

    monkeypatch.setattr(kernel_safe_probe, "run_probe", run_probe)
    monkeypatch.setattr(obligations, "workspace_snapshot", lambda *_a, **_k: SNAPSHOT)
    monkeypatch.setattr(obligations, "assess_obligations", lambda *_a, **_k: ())
    monkeypatch.setattr(obligations, "needs_work_payload", lambda *_a, **_k: None)
    return fake


def _runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str | None,
             ) -> tuple[StateFrameRuntime, list[tuple[str, dict]]]:
    """Build a real runtime through its constructor; only the check bundle and binding are faked.

    The run-start switch registry preflight refuses a switch the tree does not register
    (tests/test_retired_switches.py covers that refusal), so it is stubbed here to keep
    these tests on the cache behaviour, on head and on a tree that predates the switch.
    """
    import kernel_release_switch_registry as switch_registry
    monkeypatch.setattr(switch_registry, "enforce_stateframe_switch_registry",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setenv("STATEFRAME_HOST_CLOSURE", "on")
    if mode is None:
        monkeypatch.delenv("STATEFRAME_SHARED_CHECK_CACHE", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_SHARED_CHECK_CACHE", mode)
    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text("def value():\n    return 7\n")
    owner = SimpleNamespace(root=root, task_id="task-09", current_step=1, task_row={},
                            authorize=lambda *_args, **_kwargs: None)
    runtime = StateFrameRuntime(
        owner, output=tmp_path / "journal", target=root / "app.py", model="scripted",
        tools=[], temperature=0, source_resolver=lambda _path, _symbol: None)
    runtime.stateframe_host_closure_mode = "on"
    runtime.host_closure_payload = {
        "focus_obligation": "obligation-1", "coupled_preserve_constraints": []}
    runtime._host_check_bundle = lambda: _bundle()
    runtime.binding = lambda *_args, **_kwargs: {}
    events: list[tuple[str, dict]] = []
    emit = runtime.journal.emit

    def record(kind: str, **fields: object) -> object:
        events.append((kind, fields))
        return emit(kind, **fields)

    runtime.journal.emit = record
    return runtime, events


def _repair(runtime: StateFrameRuntime) -> None:
    runtime._run_host_repair_checks({"receipt_id": "edit-1", "kind": "edit"})


def _closure(runtime: StateFrameRuntime) -> None:
    runtime._host_closure_for_done()


def _check_events(events: list[tuple[str, dict]]) -> list[tuple[str, bool]]:
    return [(kind, bool(fields["receipt"]["reused"]))
            for kind, fields in events if kind in CHECK_EVENTS]


def test_off_runs_each_path_fresh_with_separate_caches(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe) -> None:
    runtime, events = _runtime(tmp_path, monkeypatch, "off")
    _repair(runtime)
    _closure(runtime)

    assert len(probe.runs) == 2
    assert _check_events(events) == [
        ("host_repair_check", False), ("host_obligation_check", False)]
    assert runtime.host_repair_check_cache is not runtime.host_check_cache


def test_on_one_fresh_run_when_closure_follows_repair(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe) -> None:
    runtime, events = _runtime(tmp_path, monkeypatch, "on")
    _repair(runtime)
    _closure(runtime)

    assert len(probe.runs) == 1
    assert _check_events(events) == [
        ("host_repair_check", False), ("host_obligation_check", True)]
    assert runtime.host_repair_check_cache is runtime.host_check_cache


def test_on_one_fresh_run_when_repair_follows_closure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe) -> None:
    runtime, events = _runtime(tmp_path, monkeypatch, "on")
    _closure(runtime)
    # The closure stub clears the repair payload (no assessed gap); restore it so the
    # repair path is entered the way a real edit after a gap would enter it.
    runtime.host_closure_payload = {
        "focus_obligation": "obligation-1", "coupled_preserve_constraints": []}
    _repair(runtime)

    assert len(probe.runs) == 1
    assert _check_events(events) == [
        ("host_obligation_check", False), ("host_repair_check", True)]


def test_on_different_snapshot_is_never_reused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe) -> None:
    runtime, _events = _runtime(tmp_path, monkeypatch, "on")
    _repair(runtime)
    monkeypatch.setattr(obligations, "workspace_snapshot", lambda *_a, **_k: "sha256:" + "b" * 64)
    _closure(runtime)

    assert len(probe.runs) == 2


@pytest.mark.parametrize("mode", ["off", "on"])
def test_rehydration_on_resume_fills_caches_from_both_event_kinds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe, mode: str) -> None:
    """Resume rebuilds the caches from journal events; both kinds must land in the cache."""
    runtime, events = _runtime(tmp_path / "first", monkeypatch, mode)
    _repair(runtime)
    _closure(runtime)
    assert len(probe.runs) == (1 if mode == "on" else 2)
    # Resume reads the journal through records(); serve the events this run emitted.
    journal_events = [{"kind": kind, **fields} for kind, fields in events]
    monkeypatch.setattr(AuditJournal, "records", lambda _self: iter(journal_events), raising=False)

    resumed, _ = _runtime(tmp_path / "resumed", monkeypatch, mode)

    assert len(resumed.host_check_receipts) == 1
    assert len(resumed.host_repair_check_receipts) == 1
    assert len(resumed.host_check_cache) == 1
    assert ("check-1" in {r.check_id for r in resumed.host_check_receipts})
    if mode == "on":
        assert resumed.host_repair_check_cache is resumed.host_check_cache
    else:
        assert resumed.host_repair_check_cache is not resumed.host_check_cache
        assert len(resumed.host_repair_check_cache) == 1
    # Construction itself runs no check: the resumed caches come from the journal alone.
    assert len(probe.runs) == (1 if mode == "on" else 2)


def test_counterexample_visibility_matches_off(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe) -> None:
    probe.ok, probe.observed = False, 8

    def visible(mode: str) -> list[dict]:
        runtime, _events = _runtime(tmp_path / mode, monkeypatch, mode)
        _repair(runtime)
        _closure(runtime)
        return collect_counterexamples(runtime)

    off_rows = visible("off")
    on_rows = visible("on")

    assert off_rows, "visibility must surface the failed check (vacuous otherwise)"
    assert on_rows == off_rows


@pytest.mark.parametrize("raw", ["", "maybe", "1", "ON "])
def test_invalid_switch_value_fails_closed(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("STATEFRAME_SHARED_CHECK_CACHE", raw)
    expected = "on" if raw.strip().lower() == "on" else "off"
    assert obligations.shared_check_cache_mode() == expected
