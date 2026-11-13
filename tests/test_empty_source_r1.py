"""0.5.2 review findings for zero-byte sources: non-Python, LEAN, unsafe path, stale cache."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_frame_source_snapshot import FrameAllocationReplanNeeded, frame_source_snapshot
from kernel_request_evidence_closure import build_source_closure, source_revision
from kernel_request_source_window import source_delivery_satisfied, verify_final_source_delivery
from kernel_request_source_window import render_deferred_source_notice

BODY = "".join(f"def f{i}():\n    return {i}\n" for i in range(30))


def _plan(tmp_path: Path, empty: str):
    root = tmp_path / "repo"
    files = {empty: "", "app/a.py": BODY}
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text, encoding="utf-8")
    plan = build_source_closure(root, tuple(files), "Implement.", target_path=empty).as_dict()
    plan["workspace_authorization"] = {"scope": str(root), "generation": "0"}
    plan["units"].append({
        "path": "app/a.py", "start_line": 1, "end_line": 4, "role": "target_reservation",
        "reason": "source_reservation", "priority": 1, "handle": "unit:app/a.py:1-4"})
    plan["deferred_paths"] = ["app/a.py"]
    return plan, root


def _receipt(plan):
    messages = [{"role": "user", "content": render_deferred_source_notice(
        list(plan["deferred_paths"]))}]
    payload = {"messages": messages, "tools": [{"type": "function", "function": {"name": "read"}}]}
    return verify_final_source_delivery(plan, messages, payload=payload, sources={})


def _runtime(root):
    return SimpleNamespace(owner=SimpleNamespace(root=root), _empirical_frame_sources={})


@pytest.mark.parametrize("empty", ["web/index.js", "notes.txt", "cmd/main.go"])
def test_empty_non_python_target_is_complete(tmp_path, empty):
    plan, _ = _plan(tmp_path, empty)
    receipt = _receipt(plan)
    assert receipt["paths"][empty]["status"] == "COMPLETE"
    assert receipt["paths"][empty]["empty_source"] is True
    assert source_delivery_satisfied(receipt)


@pytest.mark.parametrize("empty", ["app/__init__.py", "web/index.js"])
def test_empty_target_complete_when_parser_unavailable(tmp_path, empty):
    plan, _ = _plan(tmp_path, empty)
    plan["adapters"][empty] = {"status": "UNKNOWN", "completeness": "UNKNOWN",
                               "reason": "ModuleNotFoundError: tree_sitter"}
    receipt = _receipt(plan)
    assert receipt["paths"][empty]["status"] == "COMPLETE"
    assert source_delivery_satisfied(receipt)


def test_lean_gate_does_not_demand_a_unit_for_an_empty_target(tmp_path):
    from kernel_wire_truth_gate import _lean_source_plan_view, _source_failures
    empty = "app/__init__.py"
    plan, _ = _plan(tmp_path, empty)
    required = (empty, "app/a.py")
    view = _lean_source_plan_view(plan, required)
    assert empty in view["paths"] and view["source_contents"][empty] == ""
    context = {"source_plan": plan, "lean_required_source_paths": required}
    failures = _source_failures([{"role": "user", "content": ""}], context, b"")
    assert f"source_plan_missing_required_path:{empty}" not in failures


def test_lean_budget_keeps_the_empty_target_required_and_planned(tmp_path, monkeypatch):
    import kernel_request_budget as budget
    empty = "app/__init__.py"
    plan, _ = _plan(tmp_path, empty)
    monkeypatch.setenv("STATEFRAME_LEAN", "on")
    monkeypatch.setattr(budget, "_required_source_paths", lambda runtime: ("app/a.py", empty))
    capture = SimpleNamespace(_request_source_closure_override=plan)
    budget._record_lean_required_source_paths(capture, SimpleNamespace())
    assert capture._lean_required_source_paths == ("app/a.py", empty)


def test_disallowed_zero_byte_path_raises(tmp_path):
    plan, root = _plan(tmp_path, "app/__init__.py")
    bad = ".env"
    (root / bad).write_text("")
    plan["source_contents"][bad] = ""
    plan["revisions"][bad] = source_revision("")
    with pytest.raises(FrameAllocationReplanNeeded):
        frame_source_snapshot(_runtime(root), plan)


def test_stale_cache_for_now_empty_path_is_not_silently_overwritten(tmp_path):
    empty = "app/__init__.py"
    plan, root = _plan(tmp_path, empty)
    runtime = _runtime(root)
    runtime._empirical_frame_sources[empty] = {"content": "OLD = 1\n"}
    with pytest.raises(FrameAllocationReplanNeeded):
        frame_source_snapshot(runtime, plan)
    # During a fresh preparation the optional cache is dropped, not trusted.
    runtime._source_replan_in_progress = True
    assert frame_source_snapshot(runtime, copy.deepcopy(plan))[empty] == ""
