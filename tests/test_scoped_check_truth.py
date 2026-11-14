"""A partial public-check run must never read as a full pass or invite `done`.

The scoped truth is one shared definition (kernel_public_evidence): the denominator is the
task's declared selectors, "covers" is done admission's own rule, refused selectors never
count, and every producer (model-facing result, action receipt, frame, automatic verification,
restart) reports the same ratio.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from kernel_bolt import KernelBolt

# The scoped helpers are imported lazily so the behavioral tests below (worker, frame,
# restart) fail by assertion, not by import, on a tree without the fix.


def _scoped_check_truth_for_run(*args, **kwargs):
    from probes.react_repair import _scoped_check_truth_for_run as impl
    return impl(*args, **kwargs)

DECLARED = tuple(f"tests/test_{n}.py" for n in (
    "inventory", "pricing", "shipping", "tax", "promo", "orders"))
REV = "sha256:" + "a" * 64


def _obs(codes: dict[str, int]) -> str:
    blocks = [f"[{s}]\npytest exit {c} — {'ALL TESTS PASSED.' if c == 0 else 'TESTS FAILED'}"
              for s, c in codes.items()]
    agg = 0 if all(c == 0 for c in codes.values()) else 1
    return f"pytest exit {agg} — multi-selector oracle ({len(codes)} selectors)\n" + "\n\n".join(blocks)


def _measurement(declared=DECLARED):
    return SimpleNamespace(required_public_selectors=lambda: list(declared))


def _bolt(tmp_path: pathlib.Path, declared=DECLARED) -> KernelBolt:
    bolt = KernelBolt(tmp_path, "fn", "t")
    bolt.f2p_ids = tuple(declared)
    return bolt


@pytest.mark.parametrize("failure_site", ["selector_getter", "coverage_check"])
def test_coverage_projection_failure_withholds_hint_and_reports_type(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure_site: str,
) -> None:
    import kernel_public_evidence
    from probes.react_repair import _run_covers_declared_selectors

    def broken(*_args: object) -> list[str]:
        raise RuntimeError("private selector details")

    runtime = SimpleNamespace(required_public_selectors=lambda: list(DECLARED))
    if failure_site == "selector_getter":
        runtime.required_public_selectors = broken
    else:
        monkeypatch.setattr(kernel_public_evidence, "scoped_check_complete", broken)

    with caplog.at_level("WARNING", logger="probes.react_repair"):
        assert _run_covers_declared_selectors(runtime, {}, DECLARED) is False
    assert caplog.messages == ["Declared selector coverage unavailable: RuntimeError"]
    assert "private selector details" not in caplog.text


def test_coverage_projection_preserves_complete_partial_and_generic_results() -> None:
    from probes.react_repair import _run_covers_declared_selectors

    runtime = SimpleNamespace(required_public_selectors=lambda: list(DECLARED))
    assert _run_covers_declared_selectors(runtime, {}, DECLARED) is True
    assert _run_covers_declared_selectors(runtime, {}, DECLARED[:1]) is False
    assert _run_covers_declared_selectors(runtime, {"runner_kind": "generic"}, ()) is True


# --- shared definition ------------------------------------------------------------------

def test_every_proper_subset_is_rendered_as_partial_and_never_nudges_done(tmp_path):
    bolt = _bolt(tmp_path)
    for size in range(1, len(DECLARED)):
        for ran in itertools.combinations(DECLARED, size):
            for failing in (None, ran[0]):
                codes = {s: (1 if s == failing else 0) for s in ran}
                line, complete = _scoped_check_truth_for_run(
                    _measurement(), bolt, _obs(codes), ran, tuple(codes.values()))
                assert complete is False and line
                assert "ALL TESTS PASSED" not in line
                assert "call done" not in line.lower()
                assert f"{size} of 6 required selectors" in line
                assert ("FAIL" in line) == (failing is not None)
                bolt.last_test_exit = 0
                bolt._challenged = {}
                assert bolt.finish_nudge(1, scope_complete=complete) is None
                assert bolt._challenged.get("finish_gen") is None


def test_full_six_selector_run_is_complete_and_keeps_todays_behavior(tmp_path):
    bolt = _bolt(tmp_path)
    codes = {s: 0 for s in DECLARED}
    assert _scoped_check_truth_for_run(
        _measurement(), bolt, _obs(codes), DECLARED, tuple(codes.values())) == (None, True)
    bolt.last_test_exit = 0
    assert "call `done` now" in bolt.finish_nudge(1)


def test_coverage_reuses_done_admissions_selector_matching():
    from kernel_public_evidence import scoped_check_complete, uncovered_public_selectors
    # Item 2: a file never stands in for a declared node, a node never for its file.
    from kernel_verdict import _covers
    declared_nodes = ("tests/test_a.py::test_x", "tests/test_a.py::test_y")
    assert not _covers("tests/test_a.py", declared_nodes[0])
    assert not scoped_check_complete(declared_nodes, ("tests/test_a.py",))
    assert uncovered_public_selectors(declared_nodes, ("tests/test_a.py::test_x",)) == [
        "tests/test_a.py::test_y"]
    assert not scoped_check_complete(("tests/test_a.py",), ("tests/test_a.py::test_x",))
    assert scoped_check_complete(declared_nodes, declared_nodes)


def test_refused_selectors_never_complete_the_scope(tmp_path):
    from kernel_public_evidence import executed_selectors
    # Item 3: a passing selector followed by a refused one is not a complete run.
    bolt = _bolt(tmp_path, DECLARED[:2])
    obs = ("HOST_REFUSED: host could not run tests tests/test_pricing.py\n"
           f"[{DECLARED[0]}]\npytest exit 0 — ALL TESTS PASSED.\n\n[{DECLARED[1]}]\nHOST_REFUSED")
    line, complete = _scoped_check_truth_for_run(
        _measurement(DECLARED[:2]), bolt, obs, DECLARED[:2], (0, -1))
    assert complete is False and "1 of 2 required selectors" in line
    # The single-runner refusal path: nothing executed.
    line, complete = _scoped_check_truth_for_run(
        _measurement(DECLARED[:2]), bolt, "[selector] REFUSED_PREDISPATCH — nope", DECLARED[:1])
    assert complete is False and "0 of 2 required selectors" in line
    assert executed_selectors(DECLARED[:2], (None, -1)) == []


def test_denominator_is_the_task_declaration_not_the_active_subset(tmp_path):
    from kernel_public_evidence import declared_task_selectors
    # Item 6: six declared, one active acceptance selector, one ran -> 1 of 6.
    bolt = _bolt(tmp_path)
    active = SimpleNamespace(required_public_selectors=lambda: [DECLARED[0]])
    line, complete = _scoped_check_truth_for_run(
        active, bolt, _obs({DECLARED[0]: 0}), DECLARED[:1], (0,))
    assert complete is False and "(1 of 6 required selectors)" in line
    # Only an undeclared task falls back to the active scope.
    assert declared_task_selectors(None, (), [DECLARED[0]]) == [DECLARED[0]]
    assert declared_task_selectors(None, DECLARED, [DECLARED[0]]) == list(DECLARED)


def test_generic_runner_identity_is_its_check_command(tmp_path):
    from kernel_public_evidence import declared_task_selectors, executed_selectors, project_scoped_run
    # Item 10: a generic runner executes @check whatever target text was requested.
    row = {"runner_kind": "generic", "test_command": ["make", "check"]}
    assert declared_task_selectors(row, ()) == ["@check"]
    assert executed_selectors(("tests/test_x.py",), (0,), row) == ["@check"]
    assert project_scoped_run(["@check"], ("tests/test_x.py",), (0,), row) == (None, True)
    assert project_scoped_run(["@check"], ("tests/test_x.py",), (None,), row)[1] is False
    bolt = _bolt(tmp_path, ())
    assert _scoped_check_truth_for_run(
        SimpleNamespace(required_public_selectors=lambda: ["@check"]), bolt,
        "check exit 0 — ok", ("tests/test_x.py",), (0,), row) == (None, True)


def test_parametrized_selectors_use_structured_outcomes(tmp_path):
    # Item 12: a selector containing ']' keeps its own exit code.
    declared = ("tests/test_a.py::test_x[1]", "tests/test_a.py::test_x[2]", "tests/test_b.py")
    bolt = _bolt(tmp_path, declared)
    ran, codes = declared[:2], (0, 1)
    line, complete = _scoped_check_truth_for_run(
        _measurement(declared), bolt, _obs(dict(zip(ran, codes))), ran, codes)
    assert complete is False
    assert line.startswith("[kernel] FAIL tests/test_a.py::test_x[2]; PASS 1 (2 of 3 required selectors)")


def test_partial_pass_is_neutral_for_pivot_and_governor():
    from kernel_public_evidence import scoped_result_class
    # Item 11: an exit-0 scoped run is neither a product failure nor a green.
    from kernel_pivot import _FAILURE_RESULTS, _GREEN_RESULTS
    from kernel_continuation_evidence import RUN_TEST_RESULT_CLASSES
    assert scoped_result_class("GREEN", False) == "PARTIAL_PASS"
    assert scoped_result_class("GREEN", True) == "GREEN"
    assert scoped_result_class("RED", False) == "RED"
    assert "PARTIAL_PASS" not in _FAILURE_RESULTS and "PARTIAL_PASS" not in _GREEN_RESULTS
    assert RUN_TEST_RESULT_CLASSES["PARTIAL_PASS"] == "settled_applied"


def test_scoped_line_fits_the_result_budget(monkeypatch):
    from kernel_public_evidence import render_scoped_check_truth
    # Item 13: the truth line is reserved inside the bounded result, never appended after it.
    import kernel_frame_tool_result_budget as budget
    from kernel_public_evidence import render_scoped_delivery
    monkeypatch.setattr(budget, "frame_only_page_byte_limit", lambda _messages: 400)
    line = render_scoped_check_truth(DECLARED, DECLARED[:1], {DECLARED[0]: 0})
    obs = render_scoped_delivery("pytest exit 0 — ALL TESTS PASSED.\n" + "x" * 2000, "").rstrip("\n")
    delivered = budget.bound_frame_tool_result(obs, [], kind="test", trailer="\n" + line)
    assert len(delivered.encode("utf-8")) <= 400
    assert delivered.endswith("\n" + line) and "ALL TESTS PASSED" not in delivered
    assert budget.bound_frame_tool_result("short", [], kind="test", trailer="\n" + line) == "short\n" + line


# --- action receipt, ledger fact, frame ---------------------------------------------------

def test_scoped_failure_stays_red_in_ledger(tmp_path):
    from kernel_stateframe_action_ledger import _test_fact
    from probes.react_repair import _render_scoped_delivery
    obs = _obs({DECLARED[0]: 0, DECLARED[1]: 1})
    line, _ = _scoped_check_truth_for_run(
        _measurement(), _bolt(tmp_path), obs, DECLARED[:2], (0, 1))
    fact = _test_fact({"target": DECLARED[0]}, _render_scoped_delivery(obs, line), "c", REV)
    assert fact["verdict"] == "RED" and fact["required_selectors"] == "2/6"


def _frame_runtime(tmp_path, records):
    from kernel_stateframe_action_ledger import build_action_context
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg/app.py").write_text("VALUE = 1\n")
    task = {"role": "user", "content": "Task: Change the product value."}
    messages = [task]
    context = build_action_context(messages, workspace_root=tmp_path)
    context["journal_receipt_records"] = records
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=tmp_path, task_row={}, f2p_ids=DECLARED),
        required_public_selectors=lambda: [DECLARED[0]],
        edit_contract={"edit_authority": {"editable_paths": ["pkg/app.py"]}},
        protected_tests={}, frame_ceiling=10_000, _framefit_action_context=context,
    )
    return messages, context, runtime


def _check(selector, status="PASS"):
    return {"kind": "check_receipt", "status": status, "revision": REV, "tested_rev": REV,
            "receipt_id": f"check:{selector}", "selector": selector}


def test_frame_aggregates_the_runs_receipts_and_agrees_with_itself(tmp_path):
    # Items 1 and 7 via the real frame builder: six passing receipts read as a full pass,
    # one receipt reads as 1/6, and journal_verdict never disagrees with last_verdict.
    from kernel_stateframe_framefit5 import build_typed_state
    messages, context, runtime = _frame_runtime(tmp_path, [_check(s) for s in DECLARED])
    full = build_typed_state(messages, context, runtime)
    assert full["action_ledger"]["last_test"]["verdict"] == "GREEN"
    assert "required_selectors" not in full["action_ledger"]["last_test"]
    assert full["action_ledger"]["journal_verdict"]["status"] == full["last_verdict"]["status"]

    messages, context, runtime = _frame_runtime(tmp_path, [_check(DECLARED[0])])
    scoped = build_typed_state(messages, context, runtime)
    assert scoped["action_ledger"]["last_test"]["verdict"] == "PARTIAL_PASS"
    assert scoped["action_ledger"]["last_test"]["required_selectors"] == "1/6"
    assert scoped["last_verdict"]["status"] == "PARTIAL_PASS"
    assert scoped["action_ledger"]["journal_verdict"]["status"] == "PARTIAL_PASS"
    assert scoped["action_ledger"]["journal_verdict"] == scoped["last_verdict"]

    # A run with one failing selector among its receipts is red, not green.
    messages, context, runtime = _frame_runtime(
        tmp_path, [_check(DECLARED[0], "FAIL")] + [_check(s) for s in DECLARED[1:]])
    red = build_typed_state(messages, context, runtime)
    assert red["action_ledger"]["last_test"]["verdict"] == "RED"


def test_frame_counts_only_executed_receipts(tmp_path):
    from kernel_stateframe_framefit5 import _scoped_ratio
    runtime = SimpleNamespace(owner=SimpleNamespace(task_row={}, f2p_ids=DECLARED),
                              required_public_selectors=lambda: list(DECLARED))
    checks = [_check(DECLARED[0]), _check(DECLARED[1], "HOST_REFUSED")]
    assert _scoped_ratio(runtime, checks) == (1, 6)
    assert _scoped_ratio(runtime, [_check(s) for s in DECLARED]) is None


# --- restart ----------------------------------------------------------------------------

def test_continuation_checkpoint_persists_scope_flag():
    # Item 8: the flag is an owner checkpoint field and restores with last_test_exit.
    from kernel_continuation_restart import OWNER_FIELDS
    assert "last_test_scope_complete" in OWNER_FIELDS
    assert OWNER_FIELDS.index("last_test_scope_complete") == OWNER_FIELDS.index("last_test_exit") + 1


def test_scoped_run_is_not_a_host_green_fact(tmp_path):
    bolt = _bolt(tmp_path)
    bolt.last_test_exit = 0
    bolt.last_test_scope_complete = False
    assert bolt.last_test_green is False
    bolt.last_test_scope_complete = True
    assert bolt.last_test_green is True


# --- automatic verification --------------------------------------------------------------

def test_automatic_verification_takes_the_scoped_path():
    # Item 9: the runtime helper reports the task-wide ratio from canonical receipts.
    from kernel_stateframe_runtime import StateFrameRuntime
    runtime = SimpleNamespace(owner=SimpleNamespace(task_row={}, f2p_ids=DECLARED),
                              required_public_selectors=lambda: [DECLARED[0]])
    receipts = [{"selector": DECLARED[0], "outcome": "PASS", "execution_complete": True}]
    line, complete = StateFrameRuntime._scoped_auto_verify_truth(
        runtime, (DECLARED[0],), 0, receipts)
    assert complete is False and "(1 of 6 required selectors)" in line
    everything = [{"selector": s, "outcome": "PASS", "execution_complete": True} for s in DECLARED]
    assert StateFrameRuntime._scoped_auto_verify_truth(
        runtime, DECLARED, 0, everything) == (None, True)
    # An incomplete execution never counts as covered.
    partial = [{"selector": s, "outcome": "PASS", "execution_complete": s != DECLARED[-1]}
               for s in DECLARED]
    assert StateFrameRuntime._scoped_auto_verify_truth(runtime, DECLARED, 0, partial)[1] is False


# --- the real worker: run_tests on one of two declared checks ----------------------------

def test_work_delivers_scoped_truth_for_one_of_two_declared_checks(tmp_path):
    # Item 14: drive the real worker end to end.  The model runs one of the two discovered
    # public checks; the delivered tool result says 1 of 2 with no all-pass claim, the test
    # action receipt is PARTIAL_PASS (never GREEN) and `done` is never nudged.
    import os
    import subprocess
    from stateframe.scripted_endpoint import Script, serve
    from _journal import measurement_events

    # Same repo shape as the hook e2e runs (two public checks instead of one).  Host
    # closure checks are left off: the untouched base tree fails on them for an unrelated
    # reason (mixed receipt rows in the check-evidence digest), which would hide this
    # test's own red->green evidence.
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("")
    (repo / "pkg/value.py").write_text("VALUE = 1\n")
    for name in ("alpha", "beta"):
        (repo / "tests" / f"test_{name}.py").write_text(
            "from pkg.value import VALUE\ndef test_value():\n    assert VALUE in (1, 3)\n")
    script = Script({"model": "scripted", "steps": [
        {"tool": "run_tests", "required": True, "arguments": {"target": "tests/test_alpha.py"}},
        {"tool": "done", "required": False, "arguments": {"summary": "checked"}},
    ], "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / "run"
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", "Inspect and update VALUE.", "--endpoint",
             f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--run-dir", str(run), "--test-python", sys.executable,
             "--max-turns", "3", "--quiet"],
            env={**os.environ, "PYTHONPATH": str(root / "src"), "STATEFRAME_HOST_CLOSURE": "off",
                 "STATEFRAME_BOUNDARY_CHECKS": "off"},
            capture_output=True, text=True, timeout=600)
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    assert len(script.requests) >= 2, result.stdout + result.stderr
    wire = json.dumps(script.requests[1], sort_keys=True)
    assert "(1 of 2 required selectors)" in wire, result.stdout + result.stderr
    assert "ALL TESTS PASSED" not in wire and "call `done` now" not in wire
    events = measurement_events(run)
    receipts = [e["receipt"] for e in events if e.get("kind") == "action_receipt"
                and isinstance(e.get("receipt"), dict) and e["receipt"].get("kind") == "test"]
    assert receipts, "the run_tests action must leave a test action receipt"
    assert receipts[0]["result_class"] == "PARTIAL_PASS"
    assert all(r["result_class"] != "GREEN" for r in receipts)
