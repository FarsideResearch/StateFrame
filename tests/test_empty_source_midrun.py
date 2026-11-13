"""A zero-byte source must stay proven COMPLETE for the whole run, not only at the start.

Regression (0.5.1 mid-run ``final cache layout source proof failed``): the final
allocator adopts ``frame_source_snapshot`` as the plan's ``source_contents``. The
snapshot only visited paths that carry a selected unit, so an empty file (no unit)
lost its empty bytes on the first request that deferred a page. Without those bytes
``zero_byte_source_path`` was False, the empty target went PARTIAL, and the run died
partway through while its other files were deferred.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, serve

from kernel_frame_source_snapshot import FrameAllocationReplanNeeded, frame_source_snapshot
from kernel_request_evidence_closure import (
    build_source_closure, record_final_source_delivery, source_revision,
    zero_byte_source_path,
)
import kernel_request_source_window as window
from kernel_request_source_window import (
    render_deferred_source_notice, source_delivery_satisfied, verify_final_source_delivery,
)

ROOT = Path(__file__).resolve().parents[1]
BODY_A = "".join(f"def f{i}():\n    return {i}\n" for i in range(30))
BODY_B = "".join(f"def g{i}():\n    return {i}\n" for i in range(30))
EMPTY = "app/__init__.py"


def _plan(tmp_path: Path, *, deferred=("app/a.py", "app/b.py")):
    root = tmp_path / "repo"
    files = {EMPTY: "", "app/a.py": BODY_A, "app/b.py": BODY_B}
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    plan = build_source_closure(
        root, tuple(files), "Implement the item.", target_path=EMPTY).as_dict()
    plan["workspace_authorization"] = {"scope": str(root), "generation": "0"}
    for path in ("app/a.py", "app/b.py"):
        plan["units"].append({
            "path": path, "start_line": 1, "end_line": 4, "role": "target_reservation",
            "reason": "source_reservation", "priority": 1, "handle": f"unit:{path}:1-4"})
    plan["deferred_paths"] = list(deferred)
    return plan, root


def _runtime(root: Path):
    return SimpleNamespace(owner=SimpleNamespace(root=root), _empirical_frame_sources={})


def _deferred_receipt(plan, notice_paths=None):
    paths = list(plan["deferred_paths"]) if notice_paths is None else list(notice_paths)
    messages = ([{"role": "user", "content": render_deferred_source_notice(paths)}]
                if paths else [{"role": "user", "content": "Implement the item."}])
    payload = {"messages": messages,
               "tools": [{"type": "function", "function": {"name": "read"}}]}
    return verify_final_source_delivery(plan, messages, payload=payload, sources={})


def _selected_only(plan):
    """The pre-fix allocator state: source_contents limited to paths with a unit."""
    trimmed = copy.deepcopy(plan)
    selected = {unit["path"] for unit in trimmed["units"]}
    trimmed["source_contents"] = {
        path: text for path, text in trimmed["source_contents"].items() if path in selected}
    return trimmed


def test_empty_file_beside_deferred_paths_is_complete_and_receipt_deferred(tmp_path):
    plan, _root = _plan(tmp_path)
    receipt = _deferred_receipt(plan)
    assert receipt["status"] == "DEFERRED" and source_delivery_satisfied(receipt)
    row = receipt["paths"][EMPTY]
    assert row["status"] == "COMPLETE" and row["empty_source"] is True
    assert "delivery_proofs" in receipt and EMPTY not in receipt["delivery_proofs"]  # no wire row needed


def test_snapshot_keeps_the_empty_bytes_the_allocator_adopts_into_the_plan(tmp_path):
    plan, root = _plan(tmp_path)
    sources = frame_source_snapshot(_runtime(root), plan)
    assert sources[EMPTY] == ""
    assert set(sources) == {EMPTY, "app/a.py", "app/b.py"}
    adopted = copy.deepcopy(plan)
    adopted["source_contents"] = sources  # kernel_frame_allocation_capture does exactly this
    assert zero_byte_source_path(adopted, EMPTY) is True
    receipt = _deferred_receipt(adopted)
    assert receipt["status"] == "DEFERRED" and source_delivery_satisfied(receipt)
    assert receipt["paths"][EMPTY]["status"] == "COMPLETE"


def test_pre_fix_allocator_state_loses_the_proof(tmp_path):
    """Control: dropping the empty bytes is what turned the empty target PARTIAL."""
    plan, _root = _plan(tmp_path)
    trimmed = _selected_only(plan)
    assert zero_byte_source_path(trimmed, EMPTY) is False
    receipt = _deferred_receipt(trimmed)
    assert receipt["paths"][EMPTY]["status"] != "COMPLETE"


def test_complete_receipt_with_empty_sibling_after_every_page_is_delivered(tmp_path):
    """No deferral left (promote path): complete pages plus the empty file are COMPLETE."""
    plan, root = _plan(tmp_path, deferred=())
    adopted = copy.deepcopy(plan)
    adopted["source_contents"] = frame_source_snapshot(_runtime(root), plan)
    ranges = {"app/a.py": [(1, 4)], "app/b.py": [(1, 4)]}
    receipt = record_final_source_delivery(adopted, ranges, wire_proof_verified=True)
    assert receipt["status"] == "COMPLETE"
    assert receipt["paths"][EMPTY]["status"] == "COMPLETE"
    # Control: the same delivery with the empty bytes dropped is not COMPLETE.
    assert record_final_source_delivery(
        _selected_only(plan), ranges, wire_proof_verified=True)["status"] != "COMPLETE"


def test_not_selected_omission_row_for_the_empty_sibling_does_not_block(tmp_path):
    plan, root = _plan(tmp_path)
    plan["omissions"] = [{"path": "app/other.py", "status": "NOT_SELECTED"}]
    plan["source_contents"] = frame_source_snapshot(_runtime(root), plan)
    assert source_delivery_satisfied(_deferred_receipt(plan))


def test_non_empty_unselected_path_is_not_added_to_the_snapshot(tmp_path):
    plan, root = _plan(tmp_path)
    (root / "app/c.py").write_text("x = 1\n")
    plan["source_contents"]["app/c.py"] = "x = 1\n"
    plan["revisions"]["app/c.py"] = source_revision("x = 1\n")
    assert "app/c.py" not in frame_source_snapshot(_runtime(root), plan)


def test_forged_zero_byte_revision_is_not_snapshotted(tmp_path):
    plan, root = _plan(tmp_path)
    plan["revisions"][EMPTY] = source_revision("x = 1\n")
    assert EMPTY not in frame_source_snapshot(_runtime(root), plan)


def test_empty_file_that_gained_bytes_mid_run_forces_a_replan_not_a_stale_proof(tmp_path):
    plan, root = _plan(tmp_path)
    (root / EMPTY).write_text("VALUE = 1\n")
    with pytest.raises(FrameAllocationReplanNeeded):
        frame_source_snapshot(_runtime(root), plan)


def test_empty_file_deleted_mid_run_forces_a_replan(tmp_path):
    plan, root = _plan(tmp_path)
    (root / EMPTY).unlink()
    with pytest.raises(FrameAllocationReplanNeeded):
        frame_source_snapshot(_runtime(root), plan)


# ---------------------------------------------------------------------------
# End to end: the exact lab run that died on request 8, then more requests.
# ---------------------------------------------------------------------------

RUN_TIMEOUT_SECONDS = 1200
FIXTURE = ROOT / "tests/fixtures/empty_source_midrun.json"


def _scripted_run(tmp_path, *, env_overrides=None, slots=None):
    """Empty target app/__init__.py; the run defers pages, creates app/audit.py, truncates
    app/metrics.py to empty and deletes app/emailer.py, and must send to the end."""
    fixture = json.loads(FIXTURE.read_text())
    repo = tmp_path / "repo"
    for relative, text in fixture["files"].items():
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text(text, encoding="utf-8")
    assert (repo / EMPTY).read_text() == ""
    for command in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"]):
        subprocess.run(command, cwd=repo, check=True)
    document = {"model": "scripted", "steps": fixture["steps"]}
    if slots:
        document["slots"] = slots  # an explicit prompt budget requires a reported server slot
    script = Script(document)
    server = serve(script)
    run = tmp_path / "run"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update({"PYTHONPATH": str(ROOT / "src"),
                        "STATEFRAME_FINAL_CACHE_LAYOUT": "on", "STATEFRAME_STABLE_WIRE": "on"})
    environment["STATEFRAME_HOME"] = isolated_stateframe_home()
    environment.update(env_overrides or {})
    try:
        result = run_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
             "--task", fixture["task"],
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run), "--target", fixture["target"],
             "--test-python", sys.executable, "--max-turns", "30", "--quiet"],
            env=environment, capture_output=True, text=True,
            expected_seconds=RUN_TIMEOUT_SECONDS, run_dir=run)
    finally:
        server.shutdown()
        server.server_close()
    log = (run / "engine.log").read_text() if (run / "engine.log").exists() else ""
    text = log + result.stdout + result.stderr
    assert "source proof" not in text, text[-3000:]
    assert "ERR:ValueError" not in text, text[-3000:]
    sent = len(list((run / "wire").glob("*.request.json")))
    # The recorded run died on its 8th request; the script has 13 model steps.
    assert sent >= 12, (sent, text[-3000:])
    script.assert_consumed()
    assert (repo / "app/audit.py").exists() or (run / "workspace/app/audit.py").exists()
    return run, text


@pytest.mark.host_sandbox
def test_scripted_run_keeps_sending_through_defer_create_empty_and_delete(tmp_path):
    _scripted_run(tmp_path)


@pytest.mark.host_sandbox
def test_scripted_run_survives_outgrowing_a_low_request_cap(tmp_path):
    """T7: the recorded death needed a request over the byte cap while the target is
    empty. A low prompt budget forces that on an early send; the run must still finish
    with no proof failure and no ValueError, and some request must have been deferred."""
    run, text = _scripted_run(
        tmp_path, slots=[65536], env_overrides={"STATEFRAME_PROMPT_BUDGET": "29000"})
    assert "final cache layout source proof failed" not in text
    assert "ValueError" not in text, text[-3000:]
    requests = [json.loads(path.read_bytes())
                for path in sorted((run / "wire").glob("*.request.json"))]
    notices = [text for request in requests for text in _messages(request)
               if window.DEFERRED_SOURCE_NOTICE_HEADER in text]
    assert notices, "the low cap never deferred a page; the class was not exercised"
    assert all(EMPTY not in notice for notice in notices)


def _messages(request):
    body = request.get("body", request) if isinstance(request, dict) else request
    messages = body.get("messages", []) if isinstance(body, dict) else []
    return [item.get("content") for item in messages
            if isinstance(item, dict) and isinstance(item.get("content"), str)]


# ---------------------------------------------------------------------------
# Class killer: a zero-byte path is never deferred and never fails the proof.
# ---------------------------------------------------------------------------

PAD = [{"role": "user", "content": "x" * 3000}]


def _bound_runtime(root: Path, target: str, *, files=("app/__init__.py", "app/a.py")):
    return SimpleNamespace(target=root / target, journal=None, owner=SimpleNamespace(
        root=root, task_row={"ticket_text": "Implement the item in app/a.py.",
                             "files": list(files)}))


def _write_tree(tmp_path: Path, files):
    root = tmp_path / "repo"
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    return root


def _bound(root: Path, target: str, *, max_request_bytes=3400, files=None):
    runtime = _bound_runtime(root, target, **({"files": files} if files else {}))
    result = window.bound_runtime_sources(list(PAD), runtime, max_request_bytes=max_request_bytes)
    assert isinstance(result, dict), result
    return runtime, runtime._request_source_closure


def test_t1_empty_target_on_deferred_paths_no_longer_fails_the_final_proof(tmp_path):
    """The recorded death: deferred_paths = [empty target, a.py], notice + read tool."""
    plan, _root = _plan(tmp_path, deferred=(EMPTY, "app/a.py", "app/b.py"))
    receipt = _deferred_receipt(plan)
    assert receipt["proof_failures"] == []
    assert source_delivery_satisfied(receipt)
    assert receipt["deferred_paths"] == ["app/a.py", "app/b.py"]
    assert plan["deferred_paths"] == ["app/a.py", "app/b.py"]
    assert receipt["paths"][EMPTY]["status"] == "COMPLETE"
    assert receipt["paths"]["app/a.py"]["status"] == "DEFERRED"


def test_t1_control_the_same_plan_fails_without_the_zero_byte_proof(tmp_path, monkeypatch):
    """Proves T1 exercises the guard: a non-proven path on deferred_paths still fails."""
    plan, _root = _plan(tmp_path, deferred=(EMPTY, "app/a.py", "app/b.py"))
    monkeypatch.setattr(window, "zero_byte_source_path", lambda *_args: False)
    receipt = _deferred_receipt(plan)
    assert [row["kind"] for row in receipt["proof_failures"]] == [
        "deferred_source_path_unselected"]
    assert not source_delivery_satisfied(receipt)


def test_t2_empty_target_over_the_request_cap_is_not_deferred(tmp_path):
    root = _write_tree(tmp_path, {EMPTY: "", "app/a.py": BODY_A})
    runtime, plan = _bound(root, EMPTY)
    assert plan["deferred_paths"] == []
    assert getattr(runtime, "_request_deferred_source_paths", ()) == ()
    assert plan["source_contents"][EMPTY] == "" and zero_byte_source_path(plan, EMPTY)
    # Control for the harness: a non-empty target over the same cap is deferred.
    runtime, plan = _bound(root, "app/a.py")
    assert plan["deferred_paths"] == ["app/a.py"]
    assert runtime._request_deferred_source_paths == ("app/a.py",)


def test_t2_control_the_guard_is_what_keeps_the_empty_target_off_deferred_paths(
        tmp_path, monkeypatch):
    root = _write_tree(tmp_path, {EMPTY: "", "app/a.py": BODY_A})
    monkeypatch.setattr(window, "zero_byte_source_path", lambda *_args: False)
    runtime, plan = _bound(root, EMPTY)
    assert plan["deferred_paths"] == [EMPTY]
    assert runtime._request_deferred_source_paths == (EMPTY,)


def test_t3_empty_required_dependency_beside_a_non_empty_target(tmp_path):
    root = _write_tree(tmp_path, {EMPTY: "", "app/a.py": BODY_A, "app/b.py": BODY_B})
    runtime, plan = _bound(root, "app/a.py", files=(EMPTY, "app/a.py", "app/b.py"))
    assert EMPTY in plan["paths"] and zero_byte_source_path(plan, EMPTY)
    assert EMPTY not in plan["deferred_paths"]
    assert EMPTY not in getattr(runtime, "_request_deferred_source_paths", ())
    # The over-cap target itself is still deferred; the empty file never grows a window.
    assert "app/a.py" in plan["deferred_paths"]
    receipt = _deferred_receipt(plan)
    assert source_delivery_satisfied(receipt), receipt["proof_failures"]
    assert EMPTY not in receipt["deferred_paths"]


def test_t4_file_emptied_mid_run_leaves_the_deferral_set_on_the_next_request(tmp_path):
    """a.py is deferred while non-empty, then an edit empties it; the rebuilt closure
    must drop it from deferred_paths, from the runtime carry, and from the proof."""
    root = _write_tree(tmp_path, {EMPTY: "", "app/a.py": BODY_A})
    runtime, plan = _bound(root, "app/a.py")
    assert plan["deferred_paths"] == ["app/a.py"]
    (root / "app/a.py").write_text("", encoding="utf-8")
    result = window.bound_runtime_sources(list(PAD), runtime, max_request_bytes=3400)
    assert isinstance(result, dict), result
    plan = runtime._request_source_closure
    assert plan["revisions"]["app/a.py"] == source_revision("")
    assert plan["units"] == []
    assert plan["deferred_paths"] == []
    assert runtime._request_deferred_source_paths == ()
    # Stale carry from a checkpoint of the earlier request still cannot fail the proof.
    plan["deferred_paths"] = ["app/a.py"]
    plan["workspace_authorization"] = {"scope": str(root), "generation": "0"}
    receipt = _deferred_receipt(plan)
    assert receipt["proof_failures"] == [] and source_delivery_satisfied(receipt)
    assert plan["deferred_paths"] == [] and receipt["deferred_paths"] == []


def test_t5_stale_deferral_from_an_older_plan_is_pruned_by_the_verifier(tmp_path):
    """Checkpoint/resume: the plan already carries the empty path; the wire notice
    (rendered by the older request) still names it beside the live deferrals."""
    plan, _root = _plan(tmp_path, deferred=(EMPTY, "app/a.py", "app/b.py"))
    receipt = _deferred_receipt(plan, notice_paths=(EMPTY, "app/a.py", "app/b.py"))
    assert receipt["proof_failures"] == []
    assert plan["deferred_paths"] == ["app/a.py", "app/b.py"]
    assert receipt["deferred_paths"] == ["app/a.py", "app/b.py"]
    assert receipt["deferred_notice_verified"] is True
    assert source_delivery_satisfied(receipt)
    assert receipt["paths"][EMPTY]["status"] == "COMPLETE"
    # Only the stale entry on the plan, no live deferral: no notice is required at all.
    plan, _root = _plan(tmp_path / "second", deferred=(EMPTY,))
    plan["units"] = []
    receipt = _deferred_receipt(plan, notice_paths=(EMPTY,))
    assert receipt["proof_failures"] == []
    assert plan["deferred_paths"] == [] and receipt["deferred_paths"] == []


def test_t6_snapshot_keeps_the_empty_bytes_so_the_zero_byte_proof_survives_adoption(tmp_path):
    plan, root = _plan(tmp_path)
    assert not any(unit["path"] == EMPTY for unit in plan["units"])
    sources = frame_source_snapshot(_runtime(root), plan)
    assert sources[EMPTY] == ""
    plan["source_contents"] = sources
    assert zero_byte_source_path(plan, EMPTY) is True


def test_t8_no_deferral_producer_records_a_zero_byte_path(tmp_path):
    from kernel_read_reuse_visibility import _retain_capture_deferred_paths
    plan, _root = _plan(tmp_path, deferred=())
    plan["units"] = [unit for unit in plan["units"] if unit["path"] == "app/a.py"]
    runtime = SimpleNamespace()
    deferred: list[str] = []
    window._record_required_source_deferral(
        EMPTY, required_paths={EMPTY, "app/a.py"}, deferred_paths=deferred,
        closure_plan=plan, runtime=runtime)
    assert deferred == [] and plan["deferred_paths"] == []
    assert not hasattr(runtime, "_request_deferred_source_paths")
    window._record_required_source_deferral(
        "app/a.py", required_paths={EMPTY, "app/a.py"}, deferred_paths=deferred,
        closure_plan=plan, runtime=runtime)
    assert deferred == ["app/a.py"]
    plan["deferred_paths"] = []
    messages: list[dict[str, object]] = []
    assert window.record_deferred_source_paths(plan, messages, [EMPTY, "app/a.py"]) == ("app/a.py",)
    assert plan["deferred_paths"] == ["app/a.py"]
    plan["deferred_paths"] = []
    assert window.settle_blank_source_windows(plan, [
        {"role": "user", "content": f"[Current source: {EMPTY}]\n"}]) == ()
    assert plan["deferred_paths"] == []
    receipt = _deferred_receipt(plan)
    assert EMPTY not in window.partial_source_paths_for_deferred_delivery(plan, receipt)
    # The capture carry re-unions an older override verbatim; the verifier settles it.
    capture = SimpleNamespace(_request_source_closure_override={"deferred_paths": [EMPTY]})
    plan["deferred_paths"] = ["app/a.py"]
    _retain_capture_deferred_paths(capture, plan)
    assert plan["deferred_paths"] == [EMPTY, "app/a.py"]
    receipt = _deferred_receipt(plan)
    assert receipt["proof_failures"] == [] and plan["deferred_paths"] == ["app/a.py"]
