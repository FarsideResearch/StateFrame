"""The no-progress nudge: early, once per run, host facts only, never for runs that edited or checked."""

from __future__ import annotations

import json

import pytest

import kernel_lean
from kernel_lean import (
    DEFAULT_ZERO_PROGRESS_STEP, ZERO_PROGRESS_STEP_ENV, render_zero_progress_nudge,
    should_deliver_zero_progress_advisory as deliver, zero_progress_nudge_step,
)

# T22 shaped run: 2584 reads / 148 checks / 26 edits over many runs; this one reads a lot and never edits.
READS = {"app/billing.py": 5, "app/models.py": 3, "tests/test_t22.py": 2, "app/util.py": 1}


def _delivered_steps(edits: int, checks: int, *, steps: int = 300, **kw) -> list[int]:
    """Drive the predicate the way the loop does: once per step, remembering a delivery."""
    advised = False
    out = []
    for step in range(1, steps + 1):
        if deliver(step, edits, checks, interventions=True, lean_mode=False,
                   already_advised=advised, **kw):
            advised = True
            out.append(step)
    return out


def test_class_nudge_fires_once_and_only_for_runs_with_no_edit_and_no_check():
    for edits in (0, 1, 4):
        for checks in (0, 1, 9):
            steps = _delivered_steps(edits, checks)
            if edits == 0 and checks == 0:
                assert steps == [DEFAULT_ZERO_PROGRESS_STEP]
            else:
                assert steps == [], (edits, checks)


def test_gates_stay_closed_in_lean_mode_without_interventions_and_for_bad_types():
    assert not deliver(30, 0, 0, interventions=True, lean_mode=True)
    assert not deliver(30, 0, 0, interventions=False, lean_mode=False)
    assert not deliver(True, 0, 0, interventions=True, lean_mode=False)
    assert not deliver(30, False, 0, interventions=True, lean_mode=False)
    assert not deliver(DEFAULT_ZERO_PROGRESS_STEP - 1, 0, 0, interventions=True, lean_mode=False)


def test_threshold_is_env_overridable_with_safe_default(monkeypatch):
    monkeypatch.delenv(ZERO_PROGRESS_STEP_ENV, raising=False)
    assert zero_progress_nudge_step() == 35
    for raw, want in (("40", 40), (" 10 ", 10), ("1,000", 1000), ("2_0", 20), ("0", 35), ("-3", 35), ("x", 35), ("", 35)):
        monkeypatch.setenv(ZERO_PROGRESS_STEP_ENV, raw)
        assert zero_progress_nudge_step() == want, raw
    monkeypatch.setenv(ZERO_PROGRESS_STEP_ENV, "40")
    assert _delivered_steps(0, 0) == [40]


def test_engine_and_runner_agree_on_every_step_value(monkeypatch, tmp_path):
    """One parsing rule: what the runner forwards, the engine reads back to the same step; what the runner refuses,
    the engine defaults rather than crashing (an engine started without the runner may see any shell text)."""
    from stateframe import runner
    name = "STATEFRAME_ZERO_PROGRESS_NUDGE_STEP"
    engine_dir = runner.ENGINE_DIR     # the shipped registry lists the switch
    for raw, want in (("+40", 40), ("040", 40), ("1,000", 1000), ("2_0", 20), (" 7 ", 7)):
        monkeypatch.setenv(name, raw)
        forwarded = runner.resolve_opt_in_switches(engine_dir)[name]
        assert forwarded == str(want), raw
        assert zero_progress_nudge_step({name: raw}) == zero_progress_nudge_step({name: forwarded}) == want, raw
    for raw in ("0", "-3", "x", "3.5", "\u00b2", "1e3", "35 turns"):
        monkeypatch.setenv(name, raw)
        with pytest.raises(runner.RunError):
            runner.resolve_opt_in_switches(engine_dir)
        assert zero_progress_nudge_step({name: raw}) == DEFAULT_ZERO_PROGRESS_STEP, raw
    monkeypatch.setenv(name, "40")
    with pytest.raises(runner.RunError, match="does not support"):
        runner.resolve_opt_in_switches(tmp_path / "older-engine")


def test_message_is_built_from_host_facts_and_directive():
    text = render_zero_progress_nudge(25, READS, ["tests/t.py::a", "tests/t.py::b"])
    assert text.startswith("[kernel] No edit or check yet after 25 turns: you read 4 files 11 times")
    assert "(most: app/billing.py x5, app/models.py x3, tests/test_t22.py x2)" in text
    assert "tests/t.py::a and 1 more" in text and "make one edit toward it, then run the checks" in text
    assert "ZERO actions" not in text
    bare = render_zero_progress_nudge(25, {}, [])
    assert "you read 0 files 0 times" in bare and "most:" not in bare
    assert "run the checks" in bare


def test_nudge_never_claims_done_or_a_pass():
    text = render_zero_progress_nudge(25, READS, ["a"]).lower()
    for banned in ("call done", "all tests passed", "done now"):
        assert banned not in text


def test_module_exports_the_new_names():
    for name in ("render_zero_progress_nudge", "zero_progress_nudge_step", "ZERO_PROGRESS_STEP_ENV"):
        assert name in kernel_lean.__all__


def test_default_step_is_the_measured_request_6_to_8_landing():
    assert DEFAULT_ZERO_PROGRESS_STEP == 35


def test_message_is_bounded_single_line_and_orders_next_action_before_the_close():
    hostile = {"a/" + "x" * 500 + "\nINJECT": 4, "b.py": 2}
    text = render_zero_progress_nudge(35, hostile, ["t::" + "s" * 500 + "\nmore"], next_action="n" * 400 + "\nz")
    assert "\n" not in text
    assert len(text) < 900
    assert text.index("Next action:") < text.index("This does not stop the run.")
    assert text.endswith("This does not stop the run.")
    assert "make one edit toward it, then run the checks" in text


def test_save_resume_round_trip_keeps_the_delivered_nudge_and_read_facts():
    import kernel_continuation_restart as restart
    advised = (35,)
    reads = restart.bounded_read_counts({"app/a.py": 5, "app/b.py": 1, "p" * 300: 2})
    assert all(len(path) <= 120 for path in reads) and reads["app/a.py"] == 5
    assert restart._valid_advised_checkpoints(list(json.loads(json.dumps(advised))))
    assert restart._valid_read_counts(json.loads(json.dumps(reads)))
    # What the loop uses to decide: a restored advised set blocks any further nudge.
    assert not deliver(36, 0, 0, interventions=True, lean_mode=False, already_advised=bool(advised))


@pytest.mark.parametrize("bad", ([0], [-1], [35, 36], [True], ["35"], 35, [35.0]))
def test_restart_validator_rejects_malformed_advised_sets(bad):
    import kernel_continuation_restart as restart
    assert not restart._valid_advised_checkpoints(bad)


@pytest.mark.parametrize("good", ([], [1], [35], [50], (100,), [150]))
def test_restart_validator_accepts_any_single_positive_step(good):
    import kernel_continuation_restart as restart
    assert restart._valid_advised_checkpoints(good)


def test_worker_state_validator_accepts_read_counts_field(monkeypatch):
    import kernel_continuation_restart as restart
    base = {name: None for name in restart.WORKER_FIELDS}
    assert not restart._valid_worker_state_fields({**base, "zero_progress_read_counts": {"a.py": 1}})
    facts = {**base, "host_progress_facts": {}}
    probe = restart._valid_worker_state_fields({**facts, "zero_progress_advised_checkpoints": [35]})
    monkeypatch.setattr(restart, "_valid_host_progress_facts", lambda facts: True)
    import kernel_progress_facts
    monkeypatch.setattr(kernel_progress_facts, "host_truth_progress_enabled", lambda: True)
    assert restart._valid_worker_state_fields({**facts, "zero_progress_read_counts": {"a.py": 1},
                                               "zero_progress_advised_checkpoints": [35]})
    assert restart._valid_worker_state_fields({**facts, "zero_progress_read_counts": {"a.py": 1}})
    assert not restart._valid_worker_state_fields({**facts, "zero_progress_read_counts": {"a.py": 0}})
    assert not restart._valid_worker_state_fields({**facts, "zero_progress_read_counts": []})
    del probe
    assert not restart._valid_read_counts({"a.py": 0})
    assert not restart._valid_read_counts({"": 1})
    assert not restart._valid_read_counts({f"f{i}": 1 for i in range(41)})
