"""ERR:* engine outcomes name the exception class and message, and the catch-all records a reason row.

Incident (subframes benchmark L4, 0.7.3): the run ended ERR:FrameFit5Error. engine-result.json carried only the result
name, summary.json had no engine_error_reason, and problems.jsonl had no row.
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

import kernel_reason_log as rl
from _fault_contract import isolated_reason_log
from stateframe import _child, outcomes, runner
from stateframe.engine.probes import react_repair
from test_intake_orientation_fail_loud import _fixture_row

INCIDENT_MESSAGE = "typed next-site receipt is incomplete: current_bytes"
HANDLER_ID = "react_repair.run_exception"


class FrameFit5Error(Exception):
    """Stands in for the typed frame-fit fault from the L4 incident."""


class HarnessTransportFault(Exception):
    outcome = "HARNESS_TRANSPORT"


# ---- outcomes.engine_error_reason ----------------------------------------------------------------

def test_err_outcome_reason_names_class_and_message():
    reason = outcomes.engine_error_reason({"requests": 7, "engine_terminal": {
        "result": "ERR:FrameFit5Error", "error_type": "FrameFit5Error", "error_message": INCIDENT_MESSAGE}})
    assert reason == f"engine error: FrameFit5Error: {INCIDENT_MESSAGE}"


def test_err_outcome_without_message_falls_back_to_result_name():
    reason = outcomes.engine_error_reason({"engine_terminal": {"result": "ERR:ValueError", "error_type": "ValueError"}})
    assert reason == "engine error: ERR:ValueError"


def test_seal_failure_reason_is_unchanged():
    reason = outcomes.engine_error_reason({"requests": 7, "engine_terminal": {
        "result": "HARNESS_CONTRACT", "error_message": "bad seal"}})
    assert reason == "engine could not seal request 8: bad seal (HARNESS_CONTRACT)"


def test_successful_terminal_has_no_engine_error_reason():
    assert outcomes.engine_error_reason({"engine_terminal": {"result": "SUBMITTED"}}) is None


# ---- react_repair catch-all: returned outcome + problems.jsonl row --------------------------------

@pytest.fixture()
def reason_log(monkeypatch, tmp_path):
    with isolated_reason_log(monkeypatch, tmp_path) as harness:
        yield harness


def test_catch_all_returns_class_and_message_and_records_reason_row(tmp_path, monkeypatch, reason_log):
    # Same worker setup as test_intake_orientation_fail_loud._run_to_first_request, but the first model
    # request raises a typed engine fault instead of stopping, so the run reaches the catch-all.
    run_dir = tmp_path / "run"
    journal_parent = run_dir / "worker" / "measurement-runs"
    journal_parent.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    runtime_type = importlib.import_module("kernel_stateframe_runtime").StateFrameRuntime
    original_init = runtime_type.__init__

    def scoped_init(self: object, owner: object, **kwargs: object) -> None:
        kwargs["output"] = journal_parent
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(runtime_type, "__init__", scoped_init)

    def raise_engine_fault(*_args: object, **_kwargs: object) -> object:
        raise FrameFit5Error(INCIDENT_MESSAGE)

    monkeypatch.setattr(react_repair, "ask_tools", raise_engine_fault)
    row = _fixture_row(tmp_path / "fixture", "engine-error-reason-1")
    # Spy on the reason-log call the catch-all makes: the fault row must be written with journal=False so no
    # reason_logged event shifts the identity-bearing journal positions.
    seen_journal_args: list[object] = []
    original_emit_reason = rl.emit_reason

    def spy_emit_reason(category, cause_key, *args, **kwargs):
        if str(cause_key).startswith(HANDLER_ID):
            seen_journal_args.append(kwargs.get("journal", "<absent>"))
        return original_emit_reason(category, cause_key, *args, **kwargs)

    monkeypatch.setattr(rl, "emit_reason", spy_emit_reason)

    result = react_repair.work(row)

    assert result["result"] == "ERR:FrameFit5Error"
    assert result["error_type"] == "FrameFit5Error"
    assert result["error_message"] == INCIDENT_MESSAGE
    assert outcomes.engine_error_reason({"engine_terminal": result}) == (
        f"engine error: FrameFit5Error: {INCIDENT_MESSAGE}")

    assert seen_journal_args == [False]
    assert not [kind for kind, _fields in reason_log.journal.events if kind == "reason_logged"]

    rows = [json.loads(line) for line in Path(reason_log.problems).read_text().splitlines() if line.strip()]
    matching = [item for item in rows if isinstance(item.get("fields"), dict)
                and item["fields"].get("handler_id") == HANDLER_ID]
    assert len(matching) == 1, rows
    fields = matching[0]["fields"]
    assert fields["cause_code"] == "FrameFit5Error"
    assert fields["disposition"] == "boundary"
    assert fields["expectation"] == "unexpected"
    assert INCIDENT_MESSAGE in matching[0]["why"]
    # Fault record schema: the catch-all row names its site, class, message, work item and a stable id.
    assert fields["site_id"] == HANDLER_ID
    assert fields["kind"] == "D"
    assert fields["exception_class"] == "FrameFit5Error"
    assert fields["exception_message"] == INCIDENT_MESSAGE
    assert fields["required_work_id"] == "engine_run"
    assert len(fields["fault_id"]) == 16
    assert fields["cause_chain"] == "[]"
    assert fields["primary_fault_id"] == fields["fault_id"]


def test_worker_transport_terminal_records_request_identity(tmp_path, monkeypatch, reason_log):
    run_dir = tmp_path / "run"
    journal_parent = run_dir / "worker" / "measurement-runs"
    journal_parent.mkdir(parents=True)
    private_temp = tmp_path / "worker-tmp"
    private_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_temp))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])
    runtime_type = importlib.import_module("kernel_stateframe_runtime").StateFrameRuntime
    original_init = runtime_type.__init__

    def scoped_init(self: object, owner: object, **kwargs: object) -> None:
        kwargs["output"] = journal_parent
        original_init(self, owner, **kwargs)

    monkeypatch.setattr(runtime_type, "__init__", scoped_init)

    def raise_transport(*_args: object, **_kwargs: object) -> object:
        raise HarnessTransportFault("provider request failed")

    monkeypatch.setattr(react_repair, "ask_tools", raise_transport)
    result = react_repair.work(_fixture_row(tmp_path / "fixture", "transport-request-id"))

    assert result["result"] == "HARNESS_TRANSPORT"
    assert isinstance(result.get("request_id"), str) and result["request_id"]


def test_catch_all_fault_id_is_stable_for_the_same_incident(tmp_path):
    # Same inputs in a fresh reason-log state give the same id (no clock, uuid or random input).
    import kernel_fault_disposition as disposition

    error = FrameFit5Error(INCIDENT_MESSAGE)
    ids = []
    for index in range(2):
        run_root = tmp_path / str(index)
        run_root.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            with isolated_reason_log(patch, run_root) as harness:
                assert disposition.record_fault(HANDLER_ID, error, disposition="boundary",
                                                observed_at="react_repair.run", journal=False) is True
                row = json.loads(Path(harness.problems).read_text().splitlines()[0])
                ids.append(row["fields"]["fault_id"])
    assert ids[0] == ids[1]


REPLAY_REASON = "replay diverged at send 4: prompt bytes differ"


# Truth table for the resume hint. Rule: ERR:* keeps the hint; everything else is the pre-2c334afb rule
# (no engine_error_reason on a stopped run). ReplayUnfaithful/CheckpointUnfaithful arrive as ENGINE_ERROR with a
# reason copied from engine.log, so they must stay suppressed.
@pytest.mark.parametrize("summary, expected", [
    ({"runtime_outcome": "engine_error", "engine_error_reason": "engine error: X: y",
      "engine_terminal": {"result": "ERR:X"}}, True),
    ({"runtime_outcome": "engine_error",
      "engine_terminal": {"result": "ERR:X", "error": {"message": "crashed"}}}, True),
    ({"runtime_outcome": "engine_error", "engine_error_reason": REPLAY_REASON,
      "engine_terminal": {"result": "ENGINE_ERROR", "error": "ReplayUnfaithful"}}, False),
    ({"runtime_outcome": "engine_error", "engine_terminal": {"result": "ENGINE_ERROR", "error": "ValueError"}}, True),
    ({"runtime_outcome": "engine_error", "engine_error_reason": "engine could not seal request 2: z",
      "engine_terminal": {"result": "FINALIZATION_ERROR"}}, False),
    ({"runtime_outcome": "engine_error", "engine_error_reason": "engine could not seal request 2: z",
      "engine_terminal": {"result": "HARNESS_CONTRACT"}}, False),
    ({"runtime_outcome": "budget_stop",
      "engine_terminal": {"result": "BUDGET_EXHAUSTED",
                          "error": {"message": "step limit"}}}, True),
    ({"runtime_outcome": "budget_stop", "engine_error_reason": REPLAY_REASON, "engine_terminal": {}}, False),
    ({"runtime_outcome": "cancelled",
      "engine_terminal": {"result": "CANCELLED", "error": {"message": "interrupted"}}}, True),
    ({"runtime_outcome": "missing_result"}, True),
    ({"runtime_outcome": "recovery_required",
      "engine_terminal": {"result": "RECOVERY_REQUIRED"}}, True),
    ({"runtime_outcome": "engine_error", "engine_result": "DELIVERY_REFUSED",
      "candidate_error": "sealed candidate artifact missing",
      "engine_terminal": {"result": "SUBMITTED"}}, False),
    ({"runtime_outcome": "completed", "engine_terminal": {"result": "ERR:X"}}, False),
    ({"runtime_outcome": "completed"}, False),
])
def test_resume_hint_truth_table(summary, expected):
    assert outcomes.resume_hint_applies(summary) is expected


@pytest.mark.parametrize("malformed_terminal", [None, False, 0, 1, "", "bad", [], ["bad"], True])
def test_non_object_terminal_matches_missing_resume_hint(malformed_terminal):
    runtime_outcomes = (
        "hook_containment", "hook_blocked", "continuation_blocked", "suspended", "cancelled", "completed",
        "budget_stop", "recovery_required", "engine_error", "context_too_small", "incomplete", "missing_result",
    )
    refusal_states = (
        (None, None),
        ("DELIVERY_REFUSED", "bad"),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "CHECKS_STALE"}),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "CANDIDATE_NOT_DELIVERED"}),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "CANDIDATE_OUT_OF_SCOPE"}),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "OTHER"}),
        ("OTHER", {"engine_result": "SUBMITTED", "reason_code": "CHECKS_STALE"}),
    )
    for runtime_outcome in runtime_outcomes:
        for engine_result, delivery_refusal in refusal_states:
            summary = {"runtime_outcome": runtime_outcome}
            if engine_result is not None:
                summary["engine_result"] = engine_result
            if delivery_refusal is not None:
                summary["delivery_refusal"] = delivery_refusal
            expected = outcomes.resume_hint_applies(summary)

            summary["engine_terminal"] = malformed_terminal
            assert outcomes.resume_hint_applies(summary) is expected


def test_resume_hint_wellformed_truth_table():
    runtime_outcomes = (
        "hook_containment", "hook_blocked", "continuation_blocked", "suspended", "cancelled", "completed",
        "budget_stop", "recovery_required", "engine_error", "context_too_small", "incomplete", "missing_result",
    )
    resumable = {"cancelled", "engine_error", "missing_result", "budget_stop", "recovery_required"}
    absent = object()
    terminals = (
        absent,
        {},
        {"result": "ERR:ValueError"},
        {"result": "ENGINE_ERROR", "error": "ValueError", "error_message": "boom"},
        {"result": "HARNESS_CONTRACT"},
        {"result": "SUBMITTED"},
    )
    refusal_states = (
        (None, None, False),
        ("DELIVERY_REFUSED", "bad", False),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "CHECKS_STALE"}, True),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "CANDIDATE_NOT_DELIVERED"}, True),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "CANDIDATE_OUT_OF_SCOPE"}, True),
        ("DELIVERY_REFUSED", {"engine_result": "SUBMITTED", "reason_code": "OTHER"}, False),
        ("DELIVERY_REFUSED", {"engine_result": "OTHER", "reason_code": "CHECKS_STALE"}, False),
        ("OTHER", {"engine_result": "SUBMITTED", "reason_code": "CHECKS_STALE"}, False),
    )
    for runtime_outcome in runtime_outcomes:
        for terminal in terminals:
            for engine_result, delivery_refusal, refused in refusal_states:
                summary = {"runtime_outcome": runtime_outcome}
                if terminal is not absent:
                    summary["engine_terminal"] = terminal
                if engine_result is not None:
                    summary["engine_result"] = engine_result
                if delivery_refusal is not None:
                    summary["delivery_refusal"] = delivery_refusal
                submitted = isinstance(terminal, dict) and terminal.get("result") == "SUBMITTED"
                assert outcomes.resume_hint_applies(summary) is (
                    runtime_outcome in resumable and not refused and not submitted)


@pytest.mark.parametrize("reason_code", ["CHECKS_STALE", "CANDIDATE_NOT_DELIVERED"])
def test_submitted_delivery_refusal_does_not_offer_resume_hint(reason_code):
    summary = {
        "runtime_outcome": "engine_error",
        "engine_result": "DELIVERY_REFUSED",
        "delivery_refusal": {"engine_result": "SUBMITTED", "reason_code": reason_code},
        "engine_terminal": {"result": "SUBMITTED"},
    }
    before = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()

    assert not outcomes.resume_hint_applies(summary)
    assert json.dumps(summary, sort_keys=True, separators=(",", ":")).encode() == before


def test_submitted_out_of_scope_delivery_refusal_does_not_offer_resume_hint():
    summary = {
        "runtime_outcome": "engine_error",
        "engine_result": "DELIVERY_REFUSED",
        "delivery_refusal": {
            "engine_result": "SUBMITTED", "reason_code": "CANDIDATE_OUT_OF_SCOPE",
        },
        "engine_terminal": {"result": "SUBMITTED"},
    }
    before = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()

    assert not outcomes.resume_hint_applies(summary)
    assert json.dumps(summary, sort_keys=True, separators=(",", ":")).encode() == before


# ---- fixture envelope -> terminal receipt -> runner summary -> outcome projection ------------------

def test_fixture_envelope_names_class_and_message_through_summary_and_projection(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    envelope = {"result": {"result": "ERR:FrameFit5Error", "error_type": "FrameFit5Error",
                           "error_message": INCIDENT_MESSAGE, "steps": 3, "edits": 0, "tests": 0},
                "seconds": 1.2}
    (run / "engine-result.json").write_text(json.dumps(envelope), encoding="utf-8")

    receipt = outcomes.terminal_receipt(run)
    assert receipt["error_type"] == "FrameFit5Error"
    assert receipt["error_message"] == INCIDENT_MESSAGE

    # runner.summarize copies error_type/error_message into summary engine_terminal (its key tuple).
    summary = runner.summarize(run, 0.0)
    assert summary["engine_terminal"]["error_type"] == "FrameFit5Error"
    assert summary["engine_terminal"]["error_message"] == INCIDENT_MESSAGE

    projected = outcomes.project(summary)
    assert projected["engine_error_reason"] == f"engine error: FrameFit5Error: {INCIDENT_MESSAGE}"


@pytest.mark.parametrize(("http_status", "submission_state", "request_sent"), [
    (503, "unknown", True), (None, "sent", True),
    (None, "not_sent", False), (None, "unknown", "unknown"),
])
def test_harness_transport_projection_uses_recorded_cause_and_keeps_exit_policy(
        tmp_path, http_status, submission_state, request_sent):
    run = tmp_path / "run"
    run.mkdir()
    (run / "engine-result.json").write_text(json.dumps({"result": {
        "result": "HARNESS_TRANSPORT", "error_type": "HarnessTransportError",
        "failure_origin": "harness", "request_id": "terminal-request",
    }}), encoding="utf-8")
    journal = run / "worker" / "measurement-runs" / "one" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    events = [
        {"kind": "transport_error", "outcome": None, "request_id": "terminal-request", "http_status": 502,
         "transport_reason": "upstream_timeout", "error_type": "TimeoutError",
         "submission_state": "sent", "retry_scheduled": True},
        {"kind": "transport_error", "outcome": "HARNESS_TRANSPORT", "request_id": "terminal-request",
         "http_status": http_status,
         "transport_reason": "upstream_unavailable", "error_type": "ConnectionError",
         "submission_state": submission_state, "retry_scheduled": False},
    ]
    journal.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    summary = runner.summarize(run, 0.0)
    summary["child_exit_code"] = 0
    projected = outcomes.project(summary)
    summary.update(projected)
    previous = {"runtime_outcome": "missing_result", "engine_terminal": summary["engine_terminal"]}

    assert summary["runtime_outcome"] == "provider_transport_failure"
    assert summary["provider_transport_failure"] == {
        "status": http_status, "last_error": "upstream_unavailable", "request_sent": request_sent,
    }
    assert outcomes.exit_code(summary) == outcomes.exit_code(previous) == 2
    assert outcomes.resume_hint_applies(summary) == outcomes.resume_hint_applies(previous) is True


def test_harness_transport_retry_abort_projects_last_recorded_attempt(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "engine-result.json").write_text(json.dumps({"result": {
        "result": "HARNESS_TRANSPORT", "failure_origin": "harness", "request_id": "terminal-request",
    }}), encoding="utf-8")
    journal = run / "worker" / "measurement-runs" / "one" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({
        "kind": "provider_transport_retry_aborted", "outcome": "HARNESS_TRANSPORT",
        "request_id": "terminal-request",
        "reason": "response_already_received",
        "attempt_reasons": [{"http_status": 503, "reason": "upstream_timeout",
                             "error_type": "TimeoutError"}],
    }) + "\n", encoding="utf-8")

    terminal = runner.summarize(run, 0.0)["engine_terminal"]

    assert terminal["provider_transport_failure"] == {
        "status": 503, "last_error": "upstream_timeout", "request_sent": True,
    }


def test_harness_transport_ignores_journal_evidence_from_an_earlier_request(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "engine-result.json").write_text(json.dumps({"result": {
        "result": "HARNESS_TRANSPORT", "error_type": "TimeoutError",
        "request_id": "terminal-request", "failure_origin": "harness",
    }}), encoding="utf-8")
    journal = run / "worker" / "measurement-runs" / "one" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({
        "kind": "transport_error", "outcome": "HARNESS_TRANSPORT",
        "request_id": "earlier-request", "transport_attempt": 1,
        "http_status": 503, "transport_reason": "upstream_unavailable",
        "error_type": "ConnectionError", "submission_state": "sent",
    }) + "\n", encoding="utf-8")

    summary = runner.summarize(run, 0.0)

    assert summary["engine_terminal"]["provider_transport_failure"] == {
        "status": None, "last_error": "TimeoutError", "request_sent": "unknown",
    }


def test_harness_transport_retry_abort_keeps_submission_state_for_same_attempt(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "engine-result.json").write_text(json.dumps({"result": {
        "result": "HARNESS_TRANSPORT", "error_type": "TimeoutError",
        "request_id": "terminal-request", "failure_origin": "harness",
    }}), encoding="utf-8")
    journal = run / "worker" / "measurement-runs" / "one" / "events.jsonl"
    journal.parent.mkdir(parents=True)
    events = [
        {"kind": "transport_error", "request_id": "terminal-request",
         "transport_attempt": 1, "submission_state": "not_sent",
         "error_type": "TimeoutError", "http_status": None},
        {"kind": "provider_transport_retry_aborted", "outcome": "HARNESS_TRANSPORT",
         "request_id": "terminal-request", "attempt": 2,
         "reason": "total_deadline_expired",
         "attempt_reasons": [{"attempt": 1, "error_type": "TimeoutError",
                              "http_status": None}]},
    ]
    journal.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    terminal = runner.summarize(run, 0.0)["engine_terminal"]

    assert terminal["provider_transport_failure"] == {
        "status": None, "last_error": "TimeoutError", "request_sent": False,
    }


@pytest.mark.parametrize(("child_exit_code", "candidate_error"), [
    (1, None), (0, {"reason": "candidate conflict"}),
])
def test_harness_transport_does_not_override_existing_engine_errors(child_exit_code, candidate_error):
    summary = {
        "child_exit_code": child_exit_code,
        "candidate_error": candidate_error,
        "engine_terminal": {"result": "HARNESS_TRANSPORT"},
    }
    summary.update(outcomes.project(summary))

    assert summary["runtime_outcome"] == "engine_error"
    assert outcomes.exit_code(summary) == 2
    assert outcomes.resume_hint_applies(summary) is True


@pytest.mark.parametrize(("child_exit_code", "cancelled", "timed_out", "expected"), [
    (130, True, False, "cancelled"), (124, False, True, "budget_stop"),
])
def test_non_transport_runtime_does_not_keep_transport_cause(
        child_exit_code, cancelled, timed_out, expected):
    summary = {
        "child_exit_code": child_exit_code,
        "engine_terminal": {"result": "HARNESS_TRANSPORT", "provider_transport_failure": {
            "status": 503, "last_error": "upstream_timeout", "request_sent": True,
        }},
    }

    runner._project_final_outcome(summary, {"cancelled": cancelled, "timed_out": timed_out})

    assert summary["runtime_outcome"] == expected
    assert "provider_transport_failure" not in summary["engine_terminal"]
    assert "provider_transport_failure" not in summary


def test_non_transport_terminal_projection_does_not_gain_transport_details():
    projected = outcomes.project({"child_exit_code": 0, "engine_terminal": {
        "result": "HARNESS_CONTRACT", "error_message": "bad seal"}})

    assert projected["runtime_outcome"] == "engine_error"
    assert "provider_transport_failure" not in projected


def test_cli_summary_reports_provider_transport_cause(tmp_path, capsys):
    from stateframe import cli

    (tmp_path / "final.diff").write_text("", encoding="utf-8")
    cli._print_summary({
        "run_dir": str(tmp_path),
        "runtime_outcome": "provider_transport_failure",
        "check_outcome": "not_run",
        "artifact": "none",
        "tests_before": {"command": None, "passed": None},
        "tests_after": {"command": None},
        "engine_result": "HARNESS_TRANSPORT",
        "engine_terminal": {},
        "provider_transport_failure": {
            "status": 503, "last_error": "upstream_unavailable", "request_sent": True,
        },
        "files_changed": [],
    })

    assert "provider transport: HTTP 503; last error=upstream_unavailable; request sent=yes" in capsys.readouterr().out


# ---- _child terminal construction -----------------------------------------------------------------

def test_child_engine_error_terminal_carries_message_capped_at_600():
    result = _child._engine_error_terminal("ENGINE_ERROR", ValueError("boom\n  second line"))
    assert result == {"result": "ENGINE_ERROR", "error": "ValueError", "error_type": "ValueError",
                      "error_message": "boom second line"}
    capped = _child._engine_error_terminal("ENGINE_ERROR", ValueError("x" * 1000))
    assert len(capped["error_message"]) == 600


def test_child_engine_error_terminal_without_message_keeps_bare_shape():
    assert _child._engine_error_terminal("ENGINE_ERROR", ValueError("")) == {
        "result": "ENGINE_ERROR", "error": "ValueError"}


class _RefusingMutationRuntime:
    def invoke_hook(self, **_kwargs: object) -> None:
        raise RuntimeError("hook refused\nlater")

    def close(self) -> None:
        return None


def test_mutation_containment_terminal_carries_class_and_message():
    events: list[tuple[str, dict]] = []
    result, status = _child._finish_mutation_runtime(
        _RefusingMutationRuntime(), {"result": "FINISHED"}, 0,
        lambda kind, **fields: events.append((kind, fields)))
    assert status == 1
    assert result == {"result": "MUTATION_CONTAINMENT", "error": "hook refused\nlater",
                      "error_type": "RuntimeError", "error_message": "hook refused later"}
    assert events and events[0][0] == "engine_error"


def test_outcome_reason_for_engine_error_and_mutation_containment_with_message():
    engine = outcomes.engine_error_reason({"engine_terminal": {
        "result": "ENGINE_ERROR", "error": "ValueError", "error_type": "ValueError", "error_message": "boom"}})
    assert engine == "engine error: ValueError: boom"
    containment = outcomes.engine_error_reason({"engine_terminal": {
        "result": "MUTATION_CONTAINMENT", "error_type": "RuntimeError", "error_message": "hook refused"}})
    assert containment == "engine error: RuntimeError: hook refused"


def test_outcome_reason_unchanged_for_engine_error_without_message():
    assert outcomes.engine_error_reason({"engine_terminal": {"result": "ENGINE_ERROR", "error": "X"}}) is None


def test_replay_guidance_from_engine_log_is_kept_over_terminal_message():
    # ReplayUnfaithful carries actionable guidance in summary engine_error_reason (copied from engine.log).
    summary = {"engine_error_reason": REPLAY_REASON, "engine_terminal": {
        "result": "ENGINE_ERROR", "error": "ReplayUnfaithful", "error_type": "ReplayUnfaithful",
        "error_message": "terminal detail"}}
    assert outcomes.engine_error_reason(summary) == REPLAY_REASON


def test_resume_hint_kept_for_child_engine_error_with_message():
    from stateframe import outcomes
    crash = {"runtime_outcome": "engine_error",
             "engine_terminal": {"result": "ENGINE_ERROR", "error": "FrameFit5Error",
                                 "error_type": "FrameFit5Error", "error_message": "receipt incomplete"}}
    crash["engine_error_reason"] = outcomes.engine_error_reason(crash)
    assert crash["engine_error_reason"]
    assert outcomes.resume_hint_applies(crash)
    replay = {"runtime_outcome": "engine_error", "engine_error_reason": "replay diverged at request 3",
              "engine_terminal": {"result": "ENGINE_ERROR", "error": "ReplayUnfaithful",
                                  "error_type": "ReplayUnfaithful", "error_message": "diverged"}}
    assert not outcomes.resume_hint_applies(replay)
