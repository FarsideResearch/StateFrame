"""A proved host stop settles only the check effect, so the model can continue."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from http.server import ThreadingHTTPServer
from threading import Thread

import pytest

from kernel_check_watchdog import (
    CheckWatchdog, CheckWatchdogConfig, execution_stop_outcome,
)
from kernel_continuation_evidence import HostEvidence, binding_versions
from kernel_continuation_state import ContinuationError, ContinuationState, Effect
from kernel_measurement import digest
from _journal import measurement_events
from stateframe.scripted_endpoint import Script, make_handler
from tools.immediate_baseline import baseline_checkout  # noqa: F401 (pytest fixture)
from test_final_cache_layout_host import (
    ROOT, RUN_TIMEOUT_SECONDS, _parity_launchers, _scripted_environment,
)


def _stalled_check_document():
    return {
        "model": "scripted",
        "steps": [
            {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
            {"tool": "run_tests", "arguments": {"target": "tests/test_slow.py"}},
            {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
            {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
            {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        ],
        "tokenize": {"mode": "ok"},
    }


def _run_cli_case(tmp_path, steps, *, checkout=ROOT, max_turns=6,
                  stall_seconds="3", run_dir=None, repo_dir=None, add_slow=True,
                  profile_path=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = run_dir or tmp_path / "run"
    repo = repo_dir or tmp_path / "repo"
    shutil.copytree(checkout / "examples" / "discount_bug", repo)
    if add_slow:
        (repo / "tests" / "test_slow.py").write_text(
            "import time\n\ndef test_slow():\n    time.sleep(900)\n", encoding="utf-8")
    script = Script({"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}})
    launcher, wrapper = _parity_launchers(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(script))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = {
        **_scripted_environment(checkout),
        "STATEFRAME_CHECK_STALL_SECONDS": str(stall_seconds),
    }

    try:
        command = [
            sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
            "--task", "Fix the discount in shop/pricing.py.",
            "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
            "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
        ]
        if profile_path is not None:
            command.extend(("--profile", str(profile_path)))
        if max_turns is not None:
            command.extend(("--max-turns", str(max_turns)))
        command.extend(("--no-run-env", "--quiet"))
        result = subprocess.run(
            command,
            env=environment, capture_output=True, text=True, timeout=RUN_TIMEOUT_SECONDS,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    requests = [json.loads(path.read_bytes())
                for path in sorted((run / "wire").glob("*.request.json"))]
    terminal = json.loads((run / "engine-result.json").read_bytes())["result"]
    return run, repo, result, script, requests, terminal, measurement_events(run)


@pytest.mark.host_sandbox
def test_stopped_check_leaves_next_request_with_check_and_edit_tools(tmp_path):
    _run, _repo, result, _script, requests, _terminal, _events = _run_cli_case(
        tmp_path, _stalled_check_document()["steps"])
    assert len(requests) >= 3, result.stdout + result.stderr
    after_stop = requests[2]
    offered = {
        tool.get("function", {}).get("name")
        for tool in after_stop.get("tools", ())
        if isinstance(tool, dict)
    }
    from kernel_continuation_menu_authority import EDIT_TOOLS

    assert "run_tests" in offered
    assert offered & EDIT_TOOLS
    model_text = "\n".join(
        str(message.get("content") or "")
        for message in after_stop.get("messages", ())
        if isinstance(message, dict)
    )
    assert "UNRESOLVED" in model_text
    assert "CHECK_STOPPED:no_progress" in model_text


@pytest.mark.host_sandbox
def test_stop_then_real_failure_is_visible_with_the_stop_reason(tmp_path):
    steps = [
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_slow.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
    ]
    _run, _repo, result, _script, requests, terminal, events = _run_cli_case(tmp_path, steps)

    assert len(requests) >= 4, result.stdout + result.stderr
    after_failure = "\n".join(
        str(message.get("content") or "") for message in requests[3].get("messages", ())
    )
    assert "FAILED" in after_failure
    assert "CHECK_STOPPED:no_progress" in after_failure
    assert any(event.get("kind") == "test_subprocess_completed"
               and event.get("execution_stop", {}).get("cause") == "no_progress"
               for event in events)
    assert terminal.get("done_facts", {}).get("accepted") is False


@pytest.mark.host_sandbox
def test_failure_then_stop_keeps_failure_and_stop_visible(tmp_path):
    steps = [
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_slow.py"}},
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
    ]
    _run, _repo, result, _script, requests, _terminal, _events = _run_cli_case(tmp_path, steps)

    assert len(requests) >= 4, result.stdout + result.stderr
    after_stop = "\n".join(
        str(message.get("content") or "") for message in requests[3].get("messages", ())
    )
    assert "FAILED" in after_stop
    assert "CHECK_STOPPED:no_progress" in after_stop


@pytest.mark.host_sandbox
def test_stop_then_edit_then_pass_keeps_the_stop_unresolved(tmp_path):
    steps = [
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_slow.py"}},
        {"tool": "str_replace", "arguments": {
            "path": "shop/pricing.py",
            "old_str": "    return round(price * percent / 100, 2)",
            "new_str": "    return round(price * (100 - percent) / 100, 2)",
        }},
        {"tool": "run_tests", "arguments": {"target": "tests/test_pricing.py"}},
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
    ]
    _run, _repo, result, _script, requests, _terminal, events = _run_cli_case(tmp_path, steps)

    assert len(requests) >= 4, result.stdout + result.stderr
    after_stop = {
        tool.get("function", {}).get("name")
        for tool in requests[2].get("tools", ()) if isinstance(tool, dict)
    }
    assert "str_replace" in after_stop
    assert any(event.get("kind") == "test_subprocess_completed"
               and event.get("execution_stop", {}).get("cause") == "no_progress"
               for event in events)
    assert any(event.get("kind") == "test_subprocess_completed"
               and event.get("exit_code") == 0
               and event.get("execution_stop") is None for event in events)


@pytest.mark.host_sandbox
def test_stop_then_done_is_refused(tmp_path):
    steps = [
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        {"tool": "run_tests", "arguments": {"target": "tests/test_slow.py"}},
        {"tool": "done", "arguments": {"summary": "The stopped check is unresolved."}},
    ]
    _run, _repo, result, _script, requests, terminal, events = _run_cli_case(
        tmp_path, steps, max_turns=4)

    assert len(requests) >= 3, result.stdout + result.stderr
    assert terminal.get("done_facts", {}).get("accepted") is False
    assert "CHECK_STOPPED:no_progress" in terminal.get("done_facts", {}).get(
        "rejection_reasons", ())
    last = terminal["done_facts"].get("last_check_result") or {}
    assert last.get("status") != "PASS" and last.get("result") != "PASS"


@pytest.mark.host_sandbox
def test_repeated_stopped_checks_end_under_the_existing_no_progress_watch(tmp_path):
    steps = [
        {"tool": "read", "arguments": {"path": "shop/pricing.py"}},
        {"tool": "run_tests", "required": True, "repeat": True,
         "arguments": {"target": "tests/test_slow.py"}},
    ]
    profile = json.loads((ROOT / "src" / "stateframe" / "profiles" / "r2hc3.json").read_bytes())
    profile["switches"]["STATEFRAME_CODE_BOOKMARKS"] = "off"
    profile_path = tmp_path / "no-bookmarks-profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    run, _repo, result, _script, requests, terminal, events = _run_cli_case(
        tmp_path, steps, max_turns=None, profile_path=profile_path)

    assert len(requests) > 3, result.stdout + result.stderr
    assert terminal.get("result") == "STALLED"
    assert any(event.get("kind") == "unbounded_no_progress_terminal"
               for event in events)
    stops = [event for event in events
             if event.get("kind") == "test_subprocess_completed"
             and event.get("execution_stop", {}).get("cause") == "no_progress"
             and event.get("continuation_effect_id")]
    assert len({event["continuation_effect_id"] for event in stops}) > 1
    assert len({json.dumps(event["binding"], sort_keys=True) for event in stops}) == 1
    assert terminal.get("done_facts", {}).get("accepted") is not True
    assert (run / "engine-result.json").is_file()


def _resume_journal(tmp_path, *, closure):
    from kernel_continuation_executive import ContinuationExecutive
    from kernel_continuation_journal import ContinuationJournal

    journal = ContinuationJournal(tmp_path, "run-stopped-check", {"origin": "test"})
    engine = ContinuationExecutive(journal)
    dependencies = binding_versions(_EFFECT_BINDING)
    engine.apply(
        "create", new_id="root", objective="verify the candidate",
        pending_decision="run the selected check", dependencies=dependencies,
        turns=8, acceptance={"kind": "no_public_checks", "selectors": []},
    )
    assert engine.choose_next(_EFFECT_BINDING) == "root"
    arguments_hash = digest({"target": "tests/test_slow.py"})
    dispatch = journal.emit(
        "step_dispatch_binding", producer="stateframe_runtime", accepted=True,
        expected=_EFFECT_BINDING, current=_EFFECT_BINDING,
        continuation_id="root", continuation_effect_id="effect-1",
        request_id="request-1", step=1, tool="run_tests", snapshot_id="snapshot",
        arguments_hash=arguments_hash,
    )
    engine.apply(
        "begin_effect", intention_id="root", effect_id="effect-1",
        action_hash=digest({"tool": "run_tests", "arguments_hash": arguments_hash}),
        arguments_hash=arguments_hash, tool="run_tests", read_only=False,
        step=1, request_id="request-1", snapshot_id="snapshot",
        dispatch_ref=dispatch["event_id"], current_binding=_EFFECT_BINDING,
        current_dependencies=dependencies,
    )
    if closure:
        journal.emit(
            "test_subprocess_completed", producer="test_subprocess",
            request_id="request-1", continuation_id="root",
            continuation_effect_id="effect-1", continuation_dispatch_ref=dispatch["event_id"],
            selector_scope=["tests/test_slow.py"], selectors=["tests/test_slow.py"],
            exit_code=-9, outcome="NO_PROGRESS", execution_complete=False,
            execution_stop=_STOP, binding=_EFFECT_BINDING, after_binding=_EFFECT_BINDING,
            binding_stable=True,
        )
        receipt = {
            "receipt_id": "receipt-1", "run_id": journal.run_id,
            "request_id": "request-1", "turn": 1, "kind": "test",
            "result_class": "NO_VERDICT", "continuation_id": "root",
            "continuation_effect_id": "effect-1",
            "continuation_dispatch_ref": dispatch["event_id"],
        }
        journal.emit("action_receipt", receipt=receipt)
        record = {
            "step": 1, "tool": "run_tests", "target": "run_tests",
            "action_parameters": {"target": "tests/test_slow.py"},
            "arguments_hash": arguments_hash, "result": "NO_VERDICT",
            "receipt_ids": ["receipt-1"], "before_binding": _EFFECT_BINDING,
            "after_binding": _EFFECT_BINDING, "changes": [],
            "continuation_id": "root", "continuation_effect_id": "effect-1",
            "continuation_dispatch_ref": dispatch["event_id"],
        }
        record["step_record_id"] = digest(record)
        journal.emit("action_outcome_closed", record=record)
    path = journal.path
    journal.close()
    return path, journal.head


def _resumed_live(journal):
    from types import SimpleNamespace

    from kernel_continuation_executive import ContinuationExecutive
    from kernel_continuation_runtime import LiveContinuation, _ContinuationJournalView

    live = LiveContinuation.__new__(LiveContinuation)
    live.journal = journal
    live.engine = ContinuationExecutive(journal)
    live.runtime = SimpleNamespace(binding=lambda: _EFFECT_BINDING)
    live._journal_view = _ContinuationJournalView(journal)
    live.guarded_effects = None
    return live


def test_resume_after_closed_host_stop_replays_the_normal_check_settlement(tmp_path):
    from kernel_continuation_journal import ContinuationJournal

    path, head = _resume_journal(tmp_path, closure=True)
    journal = ContinuationJournal.resume(
        path, task_id="run-stopped-check", manifest={"origin": "test"}, expected_head=head)
    try:
        live = _resumed_live(journal)
        assert live.engine.state.effects["effect-1"].status == "pending"
        live._reconcile_unresolved_effects()
        assert live.engine.state.effects["effect-1"].status == "not_applied"
    finally:
        journal.close()


def test_resume_after_process_died_during_check_keeps_effect_pending(tmp_path):
    from kernel_continuation_journal import ContinuationJournal

    path, head = _resume_journal(tmp_path, closure=False)
    journal = ContinuationJournal.resume(
        path, task_id="run-stopped-check", manifest={"origin": "test"}, expected_head=head)
    try:
        live = _resumed_live(journal)
        assert live.engine.state.effects["effect-1"].status == "pending"
        assert live._reconcile_unresolved_effects() is None
        assert live.engine.state.effects["effect-1"].status == "pending"
        assert live.engine.choose_next(_EFFECT_BINDING) is None
    finally:
        journal.close()


@pytest.mark.host_sandbox
def test_no_stopped_check_keeps_request_bytes_equal_to_head(tmp_path, baseline_checkout):
    document = json.loads((ROOT / "examples" / "discount_bug.script.json").read_bytes())
    base_repo, current_repo = tmp_path / "repo", tmp_path / "repo"
    run = tmp_path / "run"
    _base_run, base_repo, _result, _script, base_requests, _terminal, base_events = _run_cli_case(
        tmp_path / "baseline", document["steps"], checkout=baseline_checkout,
        run_dir=run, repo_dir=base_repo, add_slow=False)
    base_wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
    run.rename(tmp_path / "baseline-run")
    base_repo.rename(tmp_path / "baseline-repo")

    _current_run, current_repo, _result, _script, current_requests, _terminal, current_events = (
        _run_cli_case(tmp_path / "current", document["steps"], checkout=ROOT,
                      run_dir=run, repo_dir=current_repo, add_slow=False))
    current_wires = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]

    assert base_requests and current_requests
    assert not any(event.get("execution_stop") for event in (*base_events, *current_events)
                   if event.get("kind") == "test_subprocess_completed")
    assert base_wires == current_wires


_EFFECT_BINDING = {
    "product_files": {"pkg/value.py": "sha256:" + "a1" * 32},
    "harness_files": {},
}
_STOP = CheckWatchdog(CheckWatchdogConfig(3)).stop_payload(
    cleanup_ok=True, cause="no_progress")


def _effect_outcome(*, effect_tool="run_tests", stop=_STOP, test_effect_id="effect-1",
                    test_binding=None, result="NO_VERDICT", closure=True,
                    typed_incomplete=True):
    run_id = "run-stopped-check"
    binding = dict(_EFFECT_BINDING)
    arguments = {"target": "tests/test_slow.py"}
    arguments_hash = digest(arguments)
    dispatch_ref = "dispatch-1"
    effect = Effect(
        "effect-1", "root", digest({"tool": effect_tool, "arguments_hash": arguments_hash}),
        binding_versions(binding), False, tool=effect_tool, arguments_hash=arguments_hash,
        dispatch_ref=dispatch_ref, step=1, request_id="request-1", intent_sequence=2,
    )
    receipt = {
        "receipt_id": "receipt-1", "run_id": run_id, "request_id": "request-1",
        "turn": 1, "kind": "test", "result_class": result,
        "continuation_id": "root", "continuation_effect_id": "effect-1",
        "continuation_dispatch_ref": dispatch_ref,
    }
    receipt_event = {
        "event_id": "receipt-event-1", "sequence": 4, "run_id": run_id,
        "kind": "action_receipt", "receipt": receipt,
    }
    check = {
        "event_id": "check-event-1", "sequence": 3, "run_id": run_id,
        "kind": "test_subprocess_completed", "producer": "test_subprocess",
        "request_id": "request-1", "continuation_id": "root",
        "continuation_effect_id": test_effect_id,
        "continuation_dispatch_ref": dispatch_ref,
        "selector_scope": ["tests/test_slow.py"], "selectors": ["tests/test_slow.py"],
        "exit_code": -9, "outcome": execution_stop_outcome(stop) if stop is not None else "NO_PROGRESS",
        "binding": test_binding or binding,
        "after_binding": test_binding or binding, "binding_stable": True,
    }
    if typed_incomplete:
        check.update(execution_complete=False, outcome="NO_PROGRESS")
    if stop is not None:
        check["execution_stop"] = stop
    record = {
        "step": 1, "tool": effect_tool, "arguments_hash": arguments_hash,
        "result": result, "receipt_ids": ["receipt-1"],
        "before_binding": binding, "after_binding": binding, "changes": [],
        "continuation_effect_id": "effect-1",
        "continuation_dispatch_ref": dispatch_ref, "continuation_id": "root",
    }
    record["step_record_id"] = digest(record)
    closed = {
        "event_id": "closed-1", "sequence": 5, "run_id": run_id,
        "kind": "action_outcome_closed", "record": record,
    }
    evidence = HostEvidence(run_id, source_refresh=True)
    evidence.observe(check)
    evidence.observe(receipt_event)
    if closure:
        evidence.observe(closed)
    state = ContinuationState(effects={effect.effect_id: effect})
    data = {"effect_id": effect.effect_id, "receipt_ref": closed["event_id"],
            "current_binding": binding}
    return evidence, state, data


def _settled_outcome(**kwargs):
    evidence, state, data = _effect_outcome(**kwargs)
    try:
        return evidence.effect_outcome(state, data)
    except ContinuationError:
        return "unknown"


def test_proved_stopped_run_tests_effect_settles_as_no_verdict():
    assert _settled_outcome() == "not_applied"


def test_stop_with_cleanup_failure_does_not_settle():
    stop = {**_STOP, "cleanup_ok": False}
    assert _settled_outcome(stop=stop) == "unknown"


def test_missing_execution_stop_does_not_settle_typed_incomplete_check():
    assert _settled_outcome(stop=None) == "unknown"


def test_malformed_execution_stop_does_not_settle():
    malformed = {key: value for key, value in _STOP.items() if key != "cleanup_ok"}
    assert _settled_outcome(stop=malformed) == "unknown"


def test_typed_incomplete_outcome_without_stop_does_not_settle():
    assert _settled_outcome(stop=None, typed_incomplete=True) == "unknown"


def test_stop_correlated_to_another_effect_does_not_settle():
    assert _settled_outcome(test_effect_id="another-effect") == "unknown"


def test_stop_bound_to_another_revision_does_not_settle():
    changed = {"product_files": {"pkg/value.py": "sha256:" + "b2" * 32},
               "harness_files": {}}
    assert _settled_outcome(test_binding=changed) == "unknown"


def test_missing_action_closure_does_not_settle():
    assert _settled_outcome(closure=False) == "unknown"


def test_kill_without_a_known_stop_reason_does_not_settle():
    assert _settled_outcome(stop=None, typed_incomplete=False) == "unknown"


def test_merely_unknown_check_outcome_does_not_settle_even_with_stop_payload():
    assert _settled_outcome(result="UNKNOWN") == "unknown"


def test_stopped_write_effect_does_not_settle():
    assert _settled_outcome(effect_tool="file_create") == "unknown"


def test_stopped_shell_effect_does_not_settle():
    assert _settled_outcome(effect_tool="shell") == "unknown"


def test_stopped_job_effect_does_not_settle():
    assert _settled_outcome(effect_tool="job_start") == "unknown"


def test_stopped_custom_effect_does_not_settle():
    assert _settled_outcome(effect_tool="custom_fixture") == "unknown"


def test_check_catalog_treats_proved_stop_as_a_settled_no_verdict():
    from kernel_catalog import _check_settled

    assert _check_settled({"status": "NO_VERDICT", "outcome": "NO_PROGRESS",
                           "execution_complete": False, "execution_stop": _STOP})


def test_check_catalog_keeps_unproved_stop_unsettled():
    from kernel_catalog import _check_settled

    assert not _check_settled({"status": "NO_VERDICT", "outcome": "NO_PROGRESS",
                               "execution_complete": False})


def test_no_pass_consumer_accepts_a_proved_stop():
    from kernel_catalog import _canonical_check_outcome, _check_settled
    from kernel_frame_continuation import _check_rows

    stopped = {"check_id": "slow", "selector": "tests/test_slow.py",
               "predicate": "check_result", "status": "NO_VERDICT",
               "result_class": "NO_VERDICT", "outcome": "NO_PROGRESS", "exit_code": -9,
               "execution_complete": False, "execution_stop": _STOP}
    assert _check_settled(stopped)
    for exit_code in (-9, 0):
        row = {**stopped, "exit_code": exit_code}
        assert _canonical_check_outcome(row) == "NO_PROGRESS"
        assert _check_rows({"check_receipts": [row]}) == []
    passed = {"check_id": "slow", "selector": "tests/test_slow.py",
              "predicate": "check_result", "status": "PASS", "exit_code": 0}
    assert _canonical_check_outcome(passed) == "PASS"
    assert len(_check_rows({"check_receipts": [passed]})) == 1
