"""STATEFRAME_SHARED_CHECK_CACHE release proof rows: T1 pairs, T2 off identity, T3 multi-turn, T7 wiring.

T1 crosses the switch with STATEFRAME_HISTORY_DIGEST_IDS over the shipped r2hc3 profile and checks both
the resolved mode values and the check paths. T2 pins the off path (explicit and unset) to one journal and
receipt stream. T3 replays a turn and a resume with the switch on and checks the stream is stable and adds
no fields that the off stream lacks. T7 proves profile, env, runtime attribute and child inheritance.

Prior art: test_host_authorship_switch_wiring (pairs and child inheritance) and
test_host_eff_09_shared_check_cache (check fakes and resume rehydration).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_obligation_assess as obligations
from kernel_framefit_receipt_history import HISTORY_DIGEST_IDS_SWITCH, history_digest_ids_enabled
from kernel_measurement import AuditJournal
from stateframe import _child, profiles, runner
from stateframe.parallel_worker import prepare_worker
from test_host_eff_09_shared_check_cache import (  # noqa: F401 (probe is a pytest fixture)
    CHECK_EVENTS, Probe, _check_events, _closure, _repair, _runtime, probe)
from test_parallel_worker_prep import _unit_fixture

ROOT = Path(__file__).resolve().parents[1]
SWITCH = obligations.SHARED_CHECK_CACHE_ENV
OTHER = HISTORY_DIGEST_IDS_SWITCH
PAIRS = ["00", "01", "10", "11"]
REGISTRY = ROOT / "src/stateframe/engine/switch-registry.json"
CHECK_KINDS = ("host_repair_check", "host_obligation_check")
TASK = "Check shared proof cache wiring."



def _set_pair(monkeypatch: pytest.MonkeyPatch, this_on: bool, other_on: bool) -> None:
    monkeypatch.setenv(SWITCH, "on" if this_on else "off")
    monkeypatch.setenv(OTHER, "on" if other_on else "off")


def _normalized(events: list[tuple[str, dict]], root: Path) -> str:
    """Canonical journal bytes with the run's own root replaced, so two runs compare on content only."""
    raw = json.dumps([[kind, fields] for kind, fields in events], sort_keys=True, default=repr)
    return raw.replace(str(root), "<ROOT>")


def _field_keys(events: list[tuple[str, dict]]) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {}
    for kind, fields in events:
        keys.setdefault(kind, set()).update(fields.keys())
    return keys


def _serve_journal(monkeypatch: pytest.MonkeyPatch, events: list[tuple[str, dict]]) -> None:
    """Serve these events to the next runtime's rehydration at construction (same as the cache suite)."""
    journal_events = [{"kind": kind, **fields} for kind, fields in events]
    monkeypatch.setattr(AuditJournal, "records", lambda _self: iter(journal_events), raising=False)


def _normalized_runs(runs: list[list[str]], root: Path) -> list[list[str]]:
    """Probe argv carries the run's workspace path, so compare argv with that root replaced."""
    return [[arg.replace(str(root), "<ROOT>") for arg in argv] for argv in runs]


# --- T1: shipped profile x STATEFRAME_SHARED_CHECK_CACHE x STATEFRAME_HISTORY_DIGEST_IDS -----------------

@pytest.mark.parametrize("pair", PAIRS)
def test_t1_pair_resolves_expected_mode_values(shipped_runtime_profile, monkeypatch, pair):
    this_on, other_on = pair[0] == "1", pair[1] == "1"
    # The shipped profile ships both switches off, so every pair is an explicit override of it.
    assert shipped_runtime_profile[SWITCH] == "off"
    assert shipped_runtime_profile[OTHER] == "off"

    _set_pair(monkeypatch, this_on, other_on)
    assert obligations.shared_check_cache_mode() == ("on" if this_on else "off")
    assert history_digest_ids_enabled() is other_on

    effective = runner._provider_environment(runner.RunOptions(repo=ROOT, task=TASK, endpoint=None))
    assert effective[SWITCH] == ("on" if this_on else "off")
    assert effective[OTHER] == ("on" if other_on else "off")


@pytest.mark.parametrize("pair", PAIRS)
def test_t1_pair_check_paths_reuse_only_when_this_switch_on(monkeypatch,
                                                            tmp_path, probe: Probe, pair):
    this_on, other_on = pair[0] == "1", pair[1] == "1"
    _set_pair(monkeypatch, this_on, other_on)
    runtime, events = _runtime(tmp_path, monkeypatch, "on" if this_on else "off")
    assert runtime.stateframe_shared_check_cache_mode == ("on" if this_on else "off")

    _repair(runtime)
    _closure(runtime)

    # The other switch must not change the reuse decision: the outcome depends on this switch alone.
    assert len(probe.runs) == (1 if this_on else 2)
    assert _check_events(events) == [
        ("host_repair_check", False),
        ("host_obligation_check", this_on),
    ]


# --- T2: marginal off identity in the shipped context ----------------------------------------------------

def _off_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe, value: str | None):
    """One repair then closure run with the switch off (explicit value or unset)."""
    probe.runs.clear()
    runtime, events = _runtime(tmp_path, monkeypatch, value)
    _repair(runtime)
    _closure(runtime)
    return runtime, events, _normalized_runs(probe.runs, tmp_path)


def test_t2_off_explicit_and_unset_are_byte_identical(monkeypatch,
                                                     tmp_path, probe: Probe):
    unset_root, explicit_root = tmp_path / "unset", tmp_path / "explicit"
    unset_rt, unset_events, unset_runs = _off_stream(unset_root, monkeypatch, probe, None)
    explicit_rt, explicit_events, explicit_runs = _off_stream(explicit_root, monkeypatch, probe, "off")

    assert _normalized(unset_events, unset_root) == _normalized(explicit_events, explicit_root)
    assert len(unset_runs) == len(explicit_runs) == 2
    assert unset_runs == explicit_runs
    assert unset_rt.stateframe_shared_check_cache_mode == explicit_rt.stateframe_shared_check_cache_mode == "off"
    for rt in (unset_rt, explicit_rt):
        assert rt.host_repair_check_cache is not rt.host_check_cache


def test_t2_off_stream_shape_is_the_separate_cache_path(monkeypatch,
                                                       tmp_path, probe: Probe):
    runtime, events, runs = _off_stream(tmp_path, monkeypatch, probe, None)

    check_events = [(kind, fields) for kind, fields in events if kind in CHECK_EVENTS]
    assert [kind for kind, _ in check_events] == list(CHECK_KINDS)
    assert all(fields["receipt"]["reused"] is False for _, fields in check_events)
    assert len(runs) == 2, "off must run each path's own check"
    assert runtime.host_repair_check_cache is not runtime.host_check_cache


# --- T3: multi-turn stability with the switch on ---------------------------------------------------------

def _two_turns_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: Probe):
    """Turn 1 runs repair and closure. Turn 2 is a resume that only closes again."""
    probe.runs.clear()
    # Turn 1 starts from an empty journal even when an earlier call in this test served one.
    monkeypatch.setattr(AuditJournal, "records", lambda _self: iter(()), raising=False)
    turn1, turn1_events = _runtime(tmp_path / "turn1", monkeypatch, "on")
    _repair(turn1)
    _closure(turn1)
    turn1_runs = len(probe.runs)

    _serve_journal(monkeypatch, turn1_events)
    turn2, turn2_events = _runtime(tmp_path / "turn2", monkeypatch, "on")
    _closure(turn2)
    return turn1_events, turn2_events, turn1_runs, len(probe.runs)


def test_t3_two_turns_on_are_stable_and_add_no_fields(monkeypatch,
                                                     tmp_path, probe: Probe):
    turn1_events, turn2_events, turn1_runs, total_runs = _two_turns_on(tmp_path / "a", monkeypatch, probe)

    # Turn 2 is served from turn 1's proof: no new probe run and a reused closure receipt.
    assert turn1_runs == 1 and total_runs == 1
    assert [kind for kind, _ in turn2_events if kind in CHECK_EVENTS] == ["host_obligation_check"]
    assert _check_events(turn2_events) == [("host_obligation_check", True)]

    # Same event kinds as off: the switch never adds a kind to either turn.
    probe.runs.clear()
    off_turn1, off_turn1_events = _runtime(tmp_path / "off", monkeypatch, "off")
    _repair(off_turn1)
    _closure(off_turn1)
    off_kinds = [kind for kind, _ in off_turn1_events if kind in CHECK_EVENTS]
    on_kinds = [kind for kind, _ in turn1_events if kind in CHECK_EVENTS]
    assert on_kinds == off_kinds == list(CHECK_KINDS)

    # No field may appear on the on-path journal that the off-path journal lacks.
    off_keys = _field_keys(off_turn1_events)
    on_keys = _field_keys(turn1_events)
    for kind in CHECK_EVENTS:
        assert on_keys.get(kind, set()) <= off_keys.get(kind, set())


def test_t3_repeated_turns_produce_identical_bytes_in_separate_roots(
                                                                    monkeypatch, tmp_path, probe: Probe):
    """Two independent runs of the same two turns must match once their root is normalized."""
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first1, first2, _, _ = _two_turns_on(first_root, monkeypatch, probe)
    second1, second2, _, _ = _two_turns_on(second_root, monkeypatch, probe)

    assert _normalized(first1, first_root) == _normalized(second1, second_root)
    assert _normalized(first2, first_root) == _normalized(second2, second_root)

    # Turn 1 closure reuses the repair proof of the same turn, and turn 2 reuses it from the journal.
    # Both closure receipts must therefore carry the same proof content.
    turn1_closure = [f["receipt"] for k, f in first1 if k == "host_obligation_check"][-1]
    turn2_closure = [f["receipt"] for k, f in first2 if k == "host_obligation_check"][-1]
    assert turn1_closure["reused"] is True and turn2_closure["reused"] is True
    # duplicate_count is the intended per-instance sighting counter: a resume adds one sighting.
    assert turn2_closure["duplicate_count"] == turn1_closure["duplicate_count"] + 1
    strip = lambda receipt: {k: v for k, v in receipt.items() if k not in {"reused", "duplicate_count"}}  # noqa: E731
    assert strip(turn1_closure) == strip(turn2_closure)


# --- T7: profile, env, runtime attribute and child inheritance --------------------------------------------

def test_t7_switch_is_registered_and_ships_off_in_profile():
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("on", "off")
    assert profiles.load("r2hc3")[SWITCH] == "off"
    registered = json.loads(REGISTRY.read_text())["switches"]
    assert SWITCH in registered


@pytest.mark.parametrize("value", [None, "off", "on"])
def test_t7_env_reaches_effective_profile(monkeypatch, value):
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    options = runner.RunOptions(repo=ROOT, task=TASK, endpoint=None)
    effective = runner._provider_environment(options)
    # Unset resolves from the shipped profile entry; an env value overrides that entry.
    assert effective[SWITCH] == (value or "off")


@pytest.mark.parametrize("env,attr,expected", [
    ("off", "on", "on"),
    ("on", "off", "off"),
    ("on", None, "on"),
    (None, "on", "on"),
    ("on", "OFF ", "off"),
    ("on", "maybe", "off"),
])
def test_t7_runtime_attribute_override_wins_over_env(monkeypatch, env, attr, expected):
    if env is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, env)
    runtime = SimpleNamespace(stateframe_shared_check_cache_mode=attr)
    assert obligations.shared_check_cache_mode(runtime) == expected


@pytest.mark.parametrize("value", [None, "off", "on"])
def test_t7_child_process_inherits_switch(tmp_path, monkeypatch, value):
    # Child install mutates os.environ wholesale; swap in a copy so monkeypatch restores the real env.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in runner.OPT_IN_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    expected = value or "off"

    options = runner.RunOptions(repo=ROOT, task=TASK, endpoint=None)
    effective = runner._provider_environment(options)
    assert effective.get(SWITCH, "off") == expected

    snapshot, run_dir, spec_path, spec, parent_spec, _parent_run = _unit_fixture(tmp_path)
    parent_spec["environment"][SWITCH] = effective.get(SWITCH, "off")
    prepare_worker(spec, spec_path, run_dir, snapshot, parent_spec=parent_spec)
    prepared = json.loads((run_dir / "spec.json").read_text())
    assert prepared["environment"].get(SWITCH, "off") == expected

    os.environ[SWITCH] = ""
    _child.install_environment(prepared)
    assert os.environ.get(SWITCH, "off") == expected
