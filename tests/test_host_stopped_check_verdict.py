"""A host-stopped public check is unresolved evidence, never a test verdict."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_check_watchdog import CheckWatchdog, CheckWatchdogConfig
from kernel_stateframe_f4 import project_f4_state
from kernel_verdict import SATISFIED, UNRESOLVED, VIOLATED, evaluate_required_tests


SELECTOR = "tests/test_idle.py"
BINDING = {"product_revision": "sha256:" + "a1" * 32}
CAUSES = (
    "no_progress", "resource_limit", "user_deadline", "cancelled",
    "external_signal", "supervision_error", "telemetry_unavailable",
)
EXIT_CODES = (None, -9, -15, 124, 137, 0)
ROOT = Path(__file__).resolve().parents[1]


def _stop(cause: str) -> dict[str, object]:
    return CheckWatchdog(CheckWatchdogConfig(3)).stop_payload(cleanup_ok=True, cause=cause)


def _event(*, sequence: int = 1, selector: str = SELECTOR, exit_code: int | None = 0,
           stop: dict[str, object] | None = None, binding: dict[str, object] | None = None,
           after_binding: dict[str, object] | None = None,
           binding_stable: bool = True) -> dict[str, object]:
    selected_binding = BINDING if binding is None else binding
    selected_after = selected_binding if after_binding is None else after_binding
    result: dict[str, object] = {
        "producer": "test_subprocess", "kind": "test_subprocess_completed",
        "receipt_id": f"check-{sequence}-{selector}", "sequence": sequence,
        "selector": selector, "selectors": [selector], "exit_code": exit_code,
        "binding": selected_binding, "after_binding": selected_after,
        "binding_stable": binding_stable,
    }
    if stop is not None:
        result.update(execution_stop=stop, execution_complete=False,
                      outcome="NO_PROGRESS" if stop["cause"] == "no_progress" else "INCOMPLETE")
    return result


def _verdict(events: list[dict[str, object]], selectors: object = SELECTOR):
    return evaluate_required_tests({"binding": BINDING}, selectors, events)


def _f4_public_checks(events: list[dict[str, object]], selectors: list[str]) -> dict[str, object]:
    receipts = []
    for event in events:
        status = ("UNKNOWN" if "execution_stop" in event else
                  "FAIL" if event["exit_code"] not in (None, 0) else
                  "PASS" if event["exit_code"] == 0 else "UNKNOWN")
        receipts.append({**event, "revision": BINDING["product_revision"],
                         "status": status, "result": status,
                         "execution_id": event["receipt_id"]})
    projection = project_f4_state({
        "revision": BINDING["product_revision"],
        "required_public_checks": selectors,
        "test_receipts": receipts,
    }, current_product_revision=BINDING["product_revision"])
    return projection["state"]["public_checks"]


_A, _B, _C = "tests/a.py", "tests/b.py", "tests/c.py"
_STOP = _stop("no_progress")
_BOTH_FAIL = _event(sequence=2, selector=_A, exit_code=1)
_BOTH_FAIL["selectors"] = [_A, _B]
_BOTH_FAIL["selector"] = _A + ", " + _B


def _child_launchers(tmp_path: Path, trace_path: Path) -> tuple[Path, Path]:
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child.py"
    launcher.write_text(
        "import ast, os, re, subprocess, sys\n"
        "from stateframe import cli\n"
        "wrapper, *args = sys.argv[1:]\n"
        "popen = subprocess.Popen\n"
        "def child(command, *positional, **keywords):\n"
        "    if len(command) == 5 and list(command[1:3]) == ['-B', '-c'] and command[4] == 'stateframe._child':\n"
        "        child_path, run_path = ast.literal_eval(re.search(r'sys\\.argv=(\\[.*?\\]);', command[3]).group(1))\n"
        "        command = [os.path.join(keywords.get('cwd') or '.', command[0]), '-B', child_path, run_path]\n"
        "        keywords['cwd'] = run_path\n"
        "    if len(command) >= 4 and str(command[2]).endswith('_child.py'):\n"
        "        command = [command[0], command[1], wrapper, command[2], command[3]]\n"
        "    return popen(command, *positional, **keywords)\n"
        "subprocess.Popen = child\n"
        "raise SystemExit(cli.main(args))\n",
        encoding="utf-8",
    )
    wrapper.write_text(
        "import os, runpy, sys\n"
        "child, run = sys.argv[1:]\n"
        "sys.path.insert(0, os.path.dirname(child))\n"
        "sys.path.insert(0, os.path.join(os.path.dirname(child), 'engine'))\n"
        "import json\n"
        "import kernel_stateframe_f4 as f4\n"
        "_reduce = f4._reduce_status\n"
        "_project = f4.project_f4_state\n"
        "def traced_project(*args, **kwargs):\n"
        "    projected = _project(*args, **kwargs)\n"
        f"    with open({str(trace_path)!r}, 'a', encoding='utf-8') as stream:\n"
        "        stream.write(json.dumps({'projected_public_checks': projected.get('state', {}).get('public_checks')}) + '\\n')\n"
        "    return projected\n"
        "f4.project_f4_state = traced_project\n"
        "def traced(candidates, revision):\n"
        "    projected = _reduce(candidates, revision)\n"
        "    if any('public-check' in str(row.get('path')) or 'receipt' in str(row.get('path')) for row in candidates):\n"
        f"        with open({str(trace_path)!r}, 'a', encoding='utf-8') as stream:\n"
        "            stream.write(json.dumps([projected[0], [{key: str(value)[:160] for key, value in row.items()} for row in candidates]]) + '\\n')\n"
        "    return projected\n"
        "f4._reduce_status = traced\n"
        "sys.argv = [child, run]\n"
        "runpy.run_path(child, run_name='__main__')\n",
        encoding="utf-8",
    )
    return launcher, wrapper


@pytest.mark.parametrize("cause", CAUSES)
@pytest.mark.parametrize("exit_code", EXIT_CODES)
def test_host_stop_is_unresolved_for_every_typed_cause_and_exit(cause, exit_code):
    verdict = _verdict([_event(exit_code=exit_code, stop=_stop(cause))])

    assert verdict.state == UNRESOLVED
    assert f"CHECK_STOPPED:{cause}" in verdict.reason_codes


def test_newer_stop_replaces_pass():
    verdict = _verdict([
        _event(sequence=1, exit_code=0),
        _event(sequence=2, exit_code=-9, stop=_stop("no_progress")),
    ])

    assert verdict.state == UNRESOLVED
    assert "CHECK_STOPPED:no_progress" in verdict.reason_codes


def test_newer_pass_replaces_stop():
    verdict = _verdict([
        _event(sequence=1, exit_code=-9, stop=_stop("no_progress")),
        _event(sequence=2, exit_code=0),
    ])

    assert verdict.state == SATISFIED
    assert "CHECK_STOPPED:no_progress" not in verdict.reason_codes


def test_newer_failure_replaces_stop():
    verdict = _verdict([
        _event(sequence=1, exit_code=-9, stop=_stop("no_progress")),
        _event(sequence=2, exit_code=1),
    ])

    assert verdict.state == VIOLATED


def test_newer_stop_replaces_failure():
    verdict = _verdict([
        _event(sequence=1, exit_code=1),
        _event(sequence=2, exit_code=-9, stop=_stop("no_progress")),
    ])

    assert verdict.state == UNRESOLVED
    assert "CHECK_STOPPED:no_progress" in verdict.reason_codes


def test_real_failure_and_pass_keep_their_existing_verdicts():
    failed = _verdict([_event(exit_code=1)])
    passed = _verdict([_event(exit_code=0)])

    assert failed.state == VIOLATED
    assert passed.state == SATISFIED


def test_stop_remains_unresolved_when_binding_is_unstable():
    changed = {"product_revision": "sha256:" + "b2" * 32}
    verdict = _verdict([_event(exit_code=-9, stop=_stop("cancelled"),
                               after_binding=changed, binding_stable=False)])

    assert verdict.state == UNRESOLVED
    assert "CHECK_STOPPED:cancelled" in verdict.reason_codes


def test_typed_incomplete_stop_outcome_without_payload_is_unresolved():
    row = _event(exit_code=0)
    row.update(execution_complete=False, outcome="NO_PROGRESS")
    verdict = _verdict([row])

    assert verdict.state == UNRESOLVED
    assert "CHECK_STOPPED:no_progress" in verdict.reason_codes


def test_incomplete_observation_certificate_preserves_typed_stop_cause():
    row = _event(exit_code=0)
    row["observation_certificate"] = {
        "execution_completeness": [False], "outcomes": ["NO_PROGRESS"],
    }
    verdict = _verdict([row])

    assert verdict.state == UNRESOLVED
    assert "CHECK_STOPPED:no_progress" in verdict.reason_codes


def test_one_stopped_selector_makes_multi_selector_verdict_unresolved():
    verdict = _verdict([
        _event(sequence=1, selector="tests/test_first.py", exit_code=0),
        _event(sequence=2, selector="tests/test_second.py", exit_code=0,
               stop=_stop("resource_limit")),
    ], ["tests/test_first.py", "tests/test_second.py"])

    assert verdict.state == UNRESOLVED
    assert "CHECK_STOPPED:resource_limit" in verdict.reason_codes


def test_untyped_kill_keeps_legacy_exit_code_verdict():
    verdict = _verdict([_event(exit_code=-9)])

    assert verdict.state == VIOLATED
    assert "CHECK_STOPPED:no_progress" not in verdict.reason_codes


def test_no_check_yet_keeps_unbacked_not_green_status_not_run():
    frame = {
        "revision": BINDING["product_revision"],
        "required_public_checks": [SELECTOR],
        "verification": {"revision": BINDING["product_revision"], "status": "FAILED"},
        "readiness": {"revision": BINDING["product_revision"],
                      "public_check_status": "FAILED"},
    }

    projection = project_f4_state(frame)

    assert projection["state"]["public_checks"]["state"] == "NOT_RUN"


def test_f4_carries_stop_from_latest_execution_into_its_summary_row():
    execution_id = "check-execution:stopped"
    frame = {
        "revision": BINDING["product_revision"],
        "required_public_checks": [SELECTOR],
        "test_receipts": [
            {**_event(sequence=1, exit_code=-9, stop=_stop("no_progress")),
             "execution_id": execution_id},
            {"kind": "test_subprocess_completed", "sequence": 2,
             "selector": SELECTOR, "selectors": [SELECTOR],
             "revision": BINDING["product_revision"], "execution_id": execution_id,
             "receipt_id": "check-receipt:stopped-summary", "status": "UNKNOWN"},
        ],
    }

    projection = project_f4_state(frame)

    assert projection["state"]["public_checks"]["state"] == "UNRESOLVED"
    assert projection["state"]["public_checks"]["reason_code"] == "CHECK_STOPPED:no_progress"
    assert any(
        row.get("reason") == "CHECK_STOPPED:no_progress"
        for row in projection["state"]["public_checks"]["evidence"]
    )


def test_host_done_facts_report_stop_reason_without_check_failed(tmp_path):
    from kernel_host_done_facts import compute_done_facts

    facts = compute_done_facts(
        tmp_path, {}, current_revision=BINDING["product_revision"],
        test_receipts=[_event(exit_code=0, stop=_stop("no_progress"))],
        required_public_tests=[SELECTOR],
    )

    assert facts.last_check_result is not None
    assert facts.last_check_result.status == "UNRESOLVED"
    assert "CHECK_STOPPED:no_progress" in facts.rejection_reasons
    assert "CHECK_FAILED" not in facts.rejection_reasons


@pytest.mark.parametrize(
    ("events", "selectors", "verdict_state", "f4_state", "stop_reason"),
    [
        ([_event(sequence=1, selector=_A, exit_code=-9, stop=_STOP),
          _event(sequence=2, selector=_B, exit_code=1)], [_A, _B], VIOLATED,
         "FAILED", "CHECK_STOPPED:no_progress"),
        ([_event(sequence=1, selector=_A, exit_code=1),
          _event(sequence=2, selector=_B, exit_code=-9, stop=_STOP)], [_A, _B], VIOLATED,
         "FAILED", "CHECK_STOPPED:no_progress"),
        ([_event(sequence=1, selector=_A, exit_code=-9, stop=_STOP),
          _event(sequence=2, selector=_B, exit_code=0)], [_A, _B], UNRESOLVED,
         "UNRESOLVED", "CHECK_STOPPED:no_progress"),
        ([_event(sequence=1, selector=_A, exit_code=-9, stop=_STOP),
          _event(sequence=2, selector=_A, exit_code=0)], [_A], SATISFIED,
         "PASSED", None),
        ([_event(sequence=1, selector=_A, exit_code=-9, stop=_STOP),
          _event(sequence=2, selector=_A, exit_code=1)], [_A], VIOLATED,
         "FAILED", None),
        ([_event(sequence=1, selector=_A, exit_code=0),
          _event(sequence=2, selector=_A, exit_code=-9, stop=_STOP)], [_A], UNRESOLVED,
         "UNRESOLVED", "CHECK_STOPPED:no_progress"),
        ([_event(sequence=1, selector=_A, exit_code=-9, stop=_STOP), _BOTH_FAIL],
         [_A, _B], VIOLATED, "FAILED", None),
    ],
)
def test_mixed_selectors_keep_each_newest_result_in_verdict_and_f4(
        events, selectors, verdict_state, f4_state, stop_reason):
    assert _verdict(events, selectors).state == verdict_state

    public_checks = _f4_public_checks(events, selectors)

    assert public_checks["state"] == f4_state
    if stop_reason is None:
        assert "reason_code" not in public_checks
    else:
        assert public_checks["reason_code"] == stop_reason


def test_three_selectors_keep_failure_visible_next_to_pass_and_stop():
    events = [
        _event(sequence=1, selector=_A, exit_code=0),
        _event(sequence=2, selector=_B, exit_code=1),
        _event(sequence=3, selector=_C, exit_code=-9, stop=_STOP),
    ]
    selectors = [_A, _B, _C]

    assert _verdict(events, selectors).state == VIOLATED
    public_checks = _f4_public_checks(events, selectors)
    assert public_checks["state"] == "FAILED"
    assert public_checks["reason_code"] == "CHECK_STOPPED:no_progress"


def test_failure_superseded_by_stop_stays_unresolved_with_other_selector_pass():
    events = [
        _event(sequence=1, selector=_A, exit_code=1),
        _event(sequence=2, selector=_A, exit_code=-9, stop=_STOP),
        _event(sequence=3, selector=_B, exit_code=0),
    ]
    selectors = [_A, _B]

    assert _verdict(events, selectors).state == UNRESOLVED
    public_checks = _f4_public_checks(events, selectors)
    assert public_checks["state"] == "UNRESOLVED"
    assert public_checks["reason_code"] == "CHECK_STOPPED:no_progress"


def test_untrusted_not_green_status_does_not_make_stopped_frame_failed():
    stopped = _event(sequence=1, selector=_A, exit_code=-9, stop=_STOP)
    not_green = {**_event(sequence=2, selector=_B, exit_code=0),
                 "status": "FAILED", "result": "FAILED"}
    selectors = [_A, _B]

    assert _verdict([stopped, not_green], selectors).state == UNRESOLVED
    public_checks = _f4_public_checks([stopped, not_green], selectors)
    assert public_checks["state"] == "UNRESOLVED"
    assert public_checks["reason_code"] == "CHECK_STOPPED:no_progress"


@pytest.mark.parametrize("events", [
    [_event(sequence=1, selector=_A, exit_code=-9, stop=_STOP),
     _event(sequence=2, selector=_B, exit_code=1)],
    [_event(sequence=1, selector=_A, exit_code=1),
     _event(sequence=2, selector=_B, exit_code=-9, stop=_STOP)],
])
def test_lean_keeps_failure_status_and_certification_with_another_stopped_selector(events):
    from kernel_stateframe_lean import (
        _authority_certification, _authority_receipt, _completion_authority,
    )

    frame = {
        "revision": BINDING["product_revision"], "binding": BINDING,
        "required_public_checks": [_A, _B], "test_receipts": events,
        "completion_contract": {
            "required_public_checks": [_A, _B],
            "current_public_checks": {"state": "FAILED", "status": "FAILED",
                                       "revision": BINDING["product_revision"],
                                       "selectors": [_A, _B]},
        },
    }
    authority = _completion_authority(frame)
    receipt = _authority_receipt(frame, authority)

    assert receipt["status"] == "FAILED"
    assert receipt["reason_code"] == "CHECK_STOPPED:no_progress"
    assert _authority_certification(frame, authority, receipt) == "RED_CURRENT"


def test_mixed_readers_keep_the_failure_and_stop_reason(tmp_path):
    from kernel_green_coherence import terminal_status
    from kernel_host_done_facts import compute_done_facts
    from kernel_live_projection import LiveProjection

    events = [
        _event(sequence=1, selector=_A, exit_code=-9, stop=_STOP),
        _event(sequence=2, selector=_B, exit_code=1),
    ]
    facts = compute_done_facts(
        tmp_path, {}, current_revision=BINDING["product_revision"],
        test_receipts=[{**row, "revision": BINDING["product_revision"]} for row in events],
        required_public_tests=[_A, _B],
    )
    assert facts.last_check_result is not None
    assert facts.last_check_result.status == "FAIL"

    facts_with_latest_stop = compute_done_facts(
        tmp_path, {}, current_revision=BINDING["product_revision"],
        test_receipts=[
            {**_event(sequence=1, selector=_A, exit_code=1),
             "revision": BINDING["product_revision"]},
            {**_event(sequence=2, selector=_B, exit_code=-9, stop=_STOP),
             "revision": BINDING["product_revision"]},
        ],
        required_public_tests=[_A, _B],
    )
    assert facts_with_latest_stop.last_check_result is not None
    assert facts_with_latest_stop.last_check_result.status == "UNRESOLVED"
    assert facts_with_latest_stop.last_check_result.reason_code == "CHECK_STOPPED:no_progress"
    assert "CHECK_STOPPED:no_progress" in facts_with_latest_stop.rejection_reasons

    coherent = terminal_status(frame={
        "binding": BINDING, "required_public_checks": [_A, _B], "test_receipts": events,
    })
    assert coherent.verdict_state == VIOLATED
    assert coherent.public_check_status == "FAIL"

    binding = {**BINDING, "test_harness_revision": "harness",
               "environment_revision": "environment", "harness_files": {},
               "product_files": {"value.py": BINDING["product_revision"]}}
    live_events = [{**row, "binding": binding, "after_binding": binding} for row in events]
    live = LiveProjection(task_id="mixed", symbol="", path="value.py", brief="mixed checks",
                          selectors=[_A, _B], binding=binding)
    projection = live.update([], [], 1, binding, live_events)
    live_verdict = projection["world"]["completion_verdict"]
    assert live_verdict["state"] == VIOLATED
    assert "CHECK_STOPPED:no_progress" in live_verdict["reason_codes"]
    assert projection["world"]["certification"]["status"] == "failed"


def test_green_coherence_preserves_stop_as_unresolved():
    from kernel_green_coherence import collapse_test_rows

    row = {**_event(exit_code=0, stop=_stop("cancelled")),
           "status": "FAILED", "result_class": "RED", "verdict": "VIOLATED"}
    collapsed = collapse_test_rows(
        {"test_receipts": [row]}, current_revision=BINDING["product_revision"])

    projected = collapsed["test_receipts"][0]
    assert projected["status"] == "UNRESOLVED"
    assert projected["result_class"] == "UNRESOLVED"
    assert projected["verdict"] == "UNRESOLVED"
    assert projected["reason_code"] == "CHECK_STOPPED:cancelled"


def test_live_lean_and_unified_readers_keep_stops_unresolved():
    from kernel_live_projection import LiveProjection
    from kernel_stateframe_lean import _status as lean_status
    from kernel_stateframe_unified import _status as unified_status
    from kernel_v33_lean import QUIET, difficulty

    row = {**_event(exit_code=0, stop=_stop("no_progress")),
           "status": "FAILED", "result_class": "RED"}
    binding = {**BINDING, "test_harness_revision": "harness",
               "environment_revision": "environment", "harness_files": {},
               "product_files": {"value.py": BINDING["product_revision"]}}
    live = LiveProjection(task_id="task", symbol="", path="value.py", brief="read value",
                          selectors=[SELECTOR], binding=binding)
    projection = live.update([], [], 1, binding, [
        {**row, "binding": binding, "after_binding": binding},
    ])

    assert projection["facts"]["verdicts"][-1]["status"] == "UNRESOLVED"
    assert projection["facts"]["verdicts"][-1]["reason_code"] == "CHECK_STOPPED:no_progress"
    assert lean_status(row) == "UNRESOLVED"
    assert unified_status(row) == "UNRESOLVED"
    assert difficulty({"test_receipts": [row]}).level == QUIET


@pytest.mark.host_sandbox
def test_stalled_check_is_model_visible_as_unresolved(tmp_path, monkeypatch):
    from stateframe import scripted_endpoint

    repo = tmp_path / "repo"
    examples = ROOT / "examples" / "discount_bug"
    shutil.copytree(examples, repo)
    for test_file in repo.rglob("test_*.py"):
        test_file.write_text("import time\ntime.sleep(30)\n" + test_file.read_text())
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STATEFRAME_HOME", str(home / "stateframe"))
    monkeypatch.setenv("STATEFRAME_CHECK_STALL_SECONDS", "3")
    run_dir = tmp_path / "run"
    document = {
        "model": "scripted",
        "steps": [
            {"tool": "read", "required": True,
             "arguments": {"path": "shop/pricing.py"}},
            {"tool": "str_replace", "required": True,
             "arguments": {"path": "shop/pricing.py",
                            "old_str": "    return round(price * percent / 100, 2)",
                            "new_str": "    return round(price * (100 - percent) / 100, 2)"}},
            {"tool": "done", "required": True,
             "arguments": {"summary": "The stopped check is unresolved."}},
        ],
        "tokenize": {"mode": "ok"},
    }
    script = scripted_endpoint.Script(document)
    server = scripted_endpoint.serve(script)
    trace_path = tmp_path / "f4-trace.jsonl"
    launcher, wrapper = _child_launchers(tmp_path, trace_path)
    try:
        result = subprocess.run(
            [sys.executable, str(launcher), str(wrapper), "run", "--repo", str(repo),
             "--task", "Fix the pricing code so the tests pass.", "--run-dir", str(run_dir),
             "--model", "scripted", "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--test-python", sys.executable, "--max-turns", "8", "--no-run-env", "--quiet"],
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True, text=True, check=False,
        )
    finally:
        server.shutdown()
        server.server_close()

    terminal = json.loads((run_dir / "engine-result.json").read_bytes())["result"]
    done_facts = terminal["done_facts"]
    assert done_facts["accepted"] is False
    assert done_facts["last_check_result"]["status"] == "UNRESOLVED"
    assert "CHECK_STOPPED:no_progress" in done_facts["rejection_reasons"]
    assert "CHECK_FAILED" not in done_facts["rejection_reasons"]
    requests = [json.loads(path.read_bytes()) for path in sorted((run_dir / "wire").glob("*.request.json"))]
    assert len(requests) >= 3, result.stdout + result.stderr
    messages = "\n".join(
        str(message.get("content") or "")
        for request in requests[2:]
        for message in request.get("messages", ())
    )
    trace = trace_path.read_text(encoding="utf-8") if trace_path.is_file() else "no F4 trace"
    assert '"public_checks"' in messages and '"state":"UNRESOLVED"' in messages, trace
    assert '"reason_code":"CHECK_STOPPED:no_progress"' in messages, trace
    assert not ('"public_checks"' in messages and '"state":"FAILED"' in messages), trace
