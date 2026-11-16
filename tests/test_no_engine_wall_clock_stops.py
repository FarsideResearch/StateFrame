"""The engine never refuses useful work because of its own clock. A limit the user sets stays.

Covers the plain loop (wall_s default None, user-set wall kept), the host limits builder, the
job watch (no total-job-wall cap) and the edit churn budget (no time deadline, request count kept).
"""
import inspect
import json
import subprocess
import sys
import time

import pytest

import kernel_edit_churn_budget as churn
import kernel_execution_policy as policy
import kernel_job_watch as jobs
import kernel_plain_host as host
import kernel_plain_loop as plain


class _Journal:
    def __init__(self):
        self.events = []

    def write(self, kind, **fields):
        self.events.append((kind, fields))
        return len(self.events)


class _JumpClock:
    """Each call advances 10_000 s, so any wall check fires on the first loop pass."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 10_000.0
        return self.now


def _tool_turn(call_id, name, arguments):
    return {"choices": [{"message": {"content": "", "tool_calls": [{
        "id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)},
    }]}, "finish_reason": "tool_calls"}]}


def _final_on_third(calls):
    """Provider stub: two tool turns, then a done call on the third request."""
    def provider(body, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            return _tool_turn(f"c{len(calls)}", "list_files", {})
        return _tool_turn("c3", "done", {"summary": "finished"})
    return provider


# ---- plain loop -------------------------------------------------------------------------------

def test_plain_limits_default_has_no_engine_wall():
    assert plain.Limits(model="m").wall_s is None


def test_plain_limits_wall_validation_allows_none_and_rejects_bad_numbers():
    assert plain.Limits(model="m", wall_s=None).wall_s is None
    assert plain.Limits(model="m", wall_s=5.0).wall_s == 5.0
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            plain.Limits(model="m", wall_s=bad)


def test_plain_loop_without_wall_ignores_engine_clock_and_finishes():
    calls, journal = [], _Journal()
    limits = plain.Limits(model="m")
    result = plain.run_loop("task", provider=_final_on_third(calls), execute=lambda name, args: "",
                            limits=limits, journal=journal, clock=_JumpClock(), sleep=lambda s: None)
    assert result.end == "done"
    assert result.requests == 3
    # Per-request timeout is the request cap itself: no clamp against a remaining wall budget.
    assert calls == [limits.request_timeout_s] * 3


def test_plain_loop_with_user_wall_still_ends_time_limit():
    calls, journal = [], _Journal()
    result = plain.run_loop("task", provider=_final_on_third(calls), execute=lambda name, args: "",
                            limits=plain.Limits(model="m", wall_s=10.0), journal=journal,
                            clock=_JumpClock(), sleep=lambda s: None)
    assert result.end == "time_limit"


def test_transport_retry_is_not_blocked_by_engine_clock_when_no_user_wall():
    result = plain.LoopResult()
    assert plain._may_retry(result, plain.Limits(model="m"), 10 ** 6, _Journal(), "http") is True


def test_transport_retry_still_bounded_by_user_wall():
    result = plain.LoopResult()
    assert plain._may_retry(result, plain.Limits(model="m", wall_s=100.0), 99.0, _Journal(), "http") is False


# ---- host limits construction -------------------------------------------------------------------

def test_host_without_plain_wall_env_builds_no_wall(monkeypatch):
    monkeypatch.delenv("CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC", raising=False)
    assert host.limits_from_env({"DS_MODEL": "m"}).wall_s is None


def test_host_empty_plain_wall_env_is_unset(monkeypatch):
    monkeypatch.delenv("CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC", raising=False)
    assert host.limits_from_env({"DS_MODEL": "m", "STATEFRAME_PLAIN_WALL_S": ""}).wall_s is None


def test_host_user_plain_wall_env_is_honoured(monkeypatch):
    monkeypatch.delenv("CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC", raising=False)
    assert host.limits_from_env({"DS_MODEL": "m", "STATEFRAME_PLAIN_WALL_S": "120"}).wall_s == 120.0


def test_host_supervisor_deadline_still_caps_an_unset_wall(monkeypatch):
    deadline = time.monotonic() + 1000.0
    wall = host.limits_from_env({"DS_MODEL": "m",
                                 "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC": repr(deadline)}).wall_s
    assert wall is not None and 900.0 < wall <= 940.0


def test_host_supervisor_deadline_takes_min_with_user_wall(monkeypatch):
    deadline = time.monotonic() + 1000.0
    env = {"DS_MODEL": "m", "STATEFRAME_PLAIN_WALL_S": "120", "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC": repr(deadline)}
    assert host.limits_from_env(env).wall_s == 120.0
    env["STATEFRAME_PLAIN_WALL_S"] = "5000"
    assert 900.0 < host.limits_from_env(env).wall_s <= 940.0


# ---- job watch ---------------------------------------------------------------------------------

def test_job_watch_configuration_has_no_total_wall_cap(monkeypatch):
    monkeypatch.setenv("STATEFRAME_JOB_WATCH_MAX_WALL_S", "0")  # a dead env var must not be read
    config = jobs.configuration()
    assert "max_total_job_wall_s" not in config
    assert config["max_concurrent_jobs"] == jobs.MAX_CONCURRENT_JOBS


def test_job_watch_constructor_has_no_total_wall_param():
    assert "max_total_wall_s" not in inspect.signature(jobs.JobWatch.__init__).parameters


def test_job_watch_dispatches_when_summed_job_wall_exceeds_old_cap(tmp_path, monkeypatch):
    now = [1000.0]  # fixed monotonic clock so reconcile keeps each fixture's 600 s elapsed
    owner = jobs.JobWatch(tmp_path, clock=lambda: now[0])
    for name in ("J1", "J2"):  # finished jobs: 1200 s summed, above the removed 900 s cap
        done = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        done.wait(timeout=30)
        record = jobs.JobRecord(name, time.time(), elapsed=600.0, expected_s=30, state="exited", exit_code=0)
        managed = jobs._ManagedJob(record, done, now[0] - 600.0, b"", lambda: now[0])
        managed.reader_done = True  # the output reader already drained this finished job
        owner._jobs[name] = managed
    owner._next_id = 3  # ids J1/J2 belong to the fixtures; the next launch is J3
    launched = []

    def fake_popen(argv, root, **kwargs):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        launched.append(process)
        return process, None

    monkeypatch.setattr(policy, "sandboxed_popen", fake_popen)
    try:
        result = owner.launch_job("sleep 60", 30, "DONE")
        assert result["job_id"] == "J3"
        assert "total_job_wall" not in json.dumps(result)
    finally:
        for process in launched:
            process.kill()
            process.wait(timeout=10)
            if process.stdout is not None:
                process.stdout.close()


# ---- edit churn budget ---------------------------------------------------------------------------

def test_churn_budget_has_no_deadline_parameter():
    assert "deadline_seconds" not in inspect.signature(churn.EditChurnBudget.__init__).parameters
    assert not hasattr(churn, "DEFAULT_DEADLINE_SECONDS")


def test_churn_elapsed_over_old_deadline_still_admits():
    budget = churn.EditChurnBudget("run", configured_limit=None)
    budget.charge("e1", True, {"k": "v"}, 4000.0, transaction_id="t1", action="edit")
    assert budget.state.elapsed_seconds == 4000.0
    assert budget.admit("edit", {"settled": True}, {}) == churn.ALLOW


def test_churn_time_reserve_no_longer_stops_public_green_candidate():
    budget = churn.EditChurnBudget("run", configured_limit=None)
    budget.mark_public_green_candidate()
    reserves = {"response_slots_remaining": 50, "remaining_seconds": 0.0, "p95_verification_seconds": 500.0}
    assert budget.admit("edit", {"settled": True}, reserves) == churn.ALLOW


def test_churn_request_count_limit_still_stops():
    budget = churn.EditChurnBudget("run", configured_limit=2)
    budget.note_request(2)
    assert budget.admit("edit", {"settled": True}, {}) == churn.STOP_INCOMPLETE


def test_churn_reserve_slots_still_stop_public_green_candidate():
    budget = churn.EditChurnBudget("run", configured_limit=None)
    budget.mark_public_green_candidate()
    assert budget.admit("edit", {"settled": True}, {"response_slots_remaining": 0}) == churn.STOP_INCOMPLETE


def test_churn_saved_state_with_old_deadline_key_loads_and_admits():
    snapshot = churn.EditChurnBudget("run", configured_limit=None).snapshot()
    snapshot["state"]["deadline_seconds"] = 5.0
    snapshot["state"]["elapsed_seconds"] = 4000.0
    restored = churn.EditChurnBudget.from_snapshot(snapshot)
    assert restored.state.elapsed_seconds == 4000.0
    assert restored.admit("edit", {"settled": True}, {}) == churn.ALLOW
