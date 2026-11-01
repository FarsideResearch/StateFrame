"""Recorded request 20: whole product source wins over duplicated frame bookkeeping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from kernel_context import token_estimate
from kernel_frame_source_snapshot import (
    _current_window, _prune_redundant_source_metadata, editable_product_windows,
    fit_recent_read_windows,
)
from kernel_request_budget import HARD_MAX_REQUEST_BYTES
from kernel_request_source_window import parse_model_source_window, verify_model_source_window
from kernel_reread_result_truth import refresh_messages, validate_delivery
from kernel_stateframe_framefit5 import _state_payload


FIXTURE = Path(__file__).parent / "fixtures" / "pin_bm10_b"


def _frame(body):
    return next(_state_payload(message)[1]["framefit5"] for message in body["messages"]
                if str(message.get("content", "")).startswith("[StateFrame]\n"))


def _runtime(root: Path, editable: list[str]):
    return SimpleNamespace(
        owner=SimpleNamespace(root=root),
        edit_contract={"edit_authority": {"editable_paths": editable}},
    )


def _recorded_rebuild():
    before = json.loads((FIXTURE / "0020.request.json").read_text())
    frame = _frame(before)
    editable = [path for path in frame["read_facts"]
                if path["path"].startswith("pkg/")]
    runtime = _runtime(FIXTURE / "source", [row["path"] for row in editable])
    windows, paths = editable_product_windows(runtime, frame, frame["code_bookmarks"])
    checks = [{"full": message["content"], "excerpt": message["content"],
               "assertions": message["content"], "full_chars": 0,
               "excerpt_chars": 0, "assertions_chars": 0}
              for message in before["messages"]
              if (meta := parse_model_source_window(message.get("content")))
              and meta.get("owner") == "required_public_check"]
    after = fit_recent_read_windows(
        before, windows, public_checks=checks, editable_paths=paths,
        fits=lambda body: (len(json.dumps(body).encode()) <= HARD_MAX_REQUEST_BYTES
                           and token_estimate(body) <= 33_344),
    )
    refresh_messages(after["messages"])
    _prune_redundant_source_metadata(after["messages"])
    return before, after, paths


def test_recorded_request_20_shows_every_product_file_and_keeps_authority(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    before, after, paths = _recorded_rebuild()
    assert len(paths) == 11
    assert len(json.dumps(after).encode()) <= HARD_MAX_REQUEST_BYTES
    assert token_estimate(after) <= 33_344
    state = _frame(after)
    validate_delivery(state, after["messages"])
    visible = {}
    for message in after["messages"]:
        metadata = parse_model_source_window(message.get("content"))
        if metadata and metadata.get("path") in paths and metadata.get("status") == "COMPLETE":
            path = metadata["path"]
            assert verify_model_source_window(
                message["content"], path=path,
                source=(FIXTURE / "source" / path).read_text(),
            )
            visible[path] = metadata
    assert set(visible) == set(paths)
    public_checks = {meta["path"] for message in after["messages"]
                     if (meta := parse_model_source_window(message.get("content")))
                     and meta.get("owner") == "required_public_check"
                     and meta.get("status") == "COMPLETE"}
    assert public_checks == {
        "tests/test_batch_contract.py", "tests/test_consumers.py",
        "tests/test_preservation.py", "tests/test_registration_serialization.py",
    }
    coverage = next(message["content"] for message in after["messages"]
                    if str(message.get("content", "")).startswith("[Source coverage]\n"))
    assert all(f"{path} (" in coverage and "not shown" not in
               next(line for line in coverage.splitlines() if line.startswith(path + " ("))
               for path in paths)
    assert _frame(before)["last_verdict"] == state["last_verdict"]
    assert _frame(before)["owner_protocol"] == state["owner_protocol"]
    assert _frame(before).get("completion_contract") == state.get("completion_contract")
    original_state = next(_state_payload(message)[1] for message in before["messages"]
                          if str(message.get("content", "")).startswith("[StateFrame]\n"))
    rebuilt_state = next(_state_payload(message)[1] for message in after["messages"]
                         if str(message.get("content", "")).startswith("[StateFrame]\n"))
    for key in ("edit_authority", "public_checks", "completion", "done_permission"):
        assert original_state.get(key) == rebuilt_state.get(key)
    assert not any(row["path"] in paths for row in state.get("code_bookmarks", ()))
    assert len(state["action_ledger"]["applied_edits"]) == 3
    assert state["action_ledger"]["historical_applied_edits"]["count"] == 5
    for fact in state["read_facts"]:
        if fact["path"] in paths:
            assert len(fact.get("receipts", ())) <= 1
            assert "content_sha256" not in fact and "lines" not in fact
            original = next(row for row in _frame(before)["read_facts"]
                            if row["path"] == fact["path"])
            assert fact["current_revision"] == original["current_revision"]
    assert json.dumps(_recorded_rebuild()[1], sort_keys=True) == json.dumps(after, sort_keys=True)


def test_large_repo_prioritizes_failure_and_bookmarks_with_complete_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    paths = [f"pkg/module_{number:02}.py" for number in range(20)]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("value = '" + "x" * 900 + "'\n")
    failure, bookmark = paths[-1], paths[-2]
    frame = {"mode": "FRAME_ONLY", "read_facts": [], "applied_edits": []}
    runtime = _runtime(tmp_path, paths)
    bookmarks = [{"path": failure, "rank": 1, "origin": "failed_check"},
                 {"path": bookmark, "rank": 2, "origin": "targeted_read"},
                 {"path": failure, "rank": 2, "origin": "direct_neighbor"}]
    windows, editable = editable_product_windows(runtime, frame, bookmarks)
    assert len(windows) < len(paths)
    assert [row["path"] for row in windows[:2]] == [failure, bookmark]
    body = {"messages": [{"role": "user", "content": "[StateFrame]\n" +
                         json.dumps({"state": {"framefit5": frame}})}], "tools": []}
    minimum = fit_recent_read_windows(body, windows[:2], editable_paths=editable,
                                      fits=lambda _body: True)
    ceiling = len(json.dumps(minimum).encode()) + 50
    fitted = fit_recent_read_windows(body, windows, editable_paths=editable,
                                     fits=lambda candidate: len(json.dumps(candidate).encode()) <= ceiling)
    visible = [meta["path"] for message in fitted["messages"]
               if (meta := parse_model_source_window(message.get("content")))
               and meta.get("path") in paths]
    assert visible[:2] == [failure, bookmark]
    assert len(json.dumps(fitted).encode()) <= ceiling
    coverage = next(message["content"] for message in fitted["messages"]
                    if str(message.get("content", "")).startswith("[Source coverage]\n"))
    assert all(path in coverage for path in paths)
    assert any("not shown" in line for line in coverage.splitlines())


def test_recorded_budget_bound_is_monotone_for_source_selection(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    before = json.loads((FIXTURE / "0020.request.json").read_text())
    frame = _frame(before)
    paths = [row["path"] for row in frame["read_facts"] if row["path"].startswith("pkg/")]
    windows, editable = editable_product_windows(_runtime(FIXTURE / "source", paths),
                                                 frame, frame["code_bookmarks"])
    previous = 0
    for ceiling in (76_000, 78_000, 80_000, HARD_MAX_REQUEST_BYTES):
        fitted = fit_recent_read_windows(
            before, windows, editable_paths=editable,
            fits=lambda body: len(json.dumps(body).encode()) <= ceiling,
        )
        assert len(json.dumps(fitted).encode()) <= ceiling
        visible = {meta["path"] for message in fitted["messages"]
                   if (meta := parse_model_source_window(message.get("content")))
                   and meta.get("path") in editable and meta.get("status") == "COMPLETE"}
        assert len(visible) >= previous
        previous = len(visible)


def test_coverage_lists_unshown_files_without_any_source_carrier(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    body = {"messages": [{"role": "user", "content": "[StateFrame]\n" + json.dumps({
        "state": {"framefit5": {"mode": "FRAME_ONLY", "read_facts": []}},
    })}], "tools": []}
    fitted = fit_recent_read_windows(body, [], editable_paths=["pkg/one.py", "pkg/two.py"],
                                     fits=lambda _body: True)
    coverage = next(message["content"] for message in fitted["messages"]
                    if str(message.get("content", "")).startswith("[Source coverage]\n"))
    assert "pkg/one.py: not shown" in coverage
    assert "pkg/two.py: not shown" in coverage


def test_stale_whole_window_cannot_erase_current_read_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir()
    target.write_text("VALUE = 1\n")
    revision = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
    window = _current_window(tmp_path, "pkg/module.py", 1, 1, revision)
    assert window is not None
    fact = {"path": "pkg/module.py", "current_revision": "sha256:changed",
            "read_revision": "sha256:changed", "receipts": [{"receipt_id": "last"}],
            "body_delivery": "not_shown"}
    body = {"messages": [
        {"role": "user", "content": "[StateFrame]\n" + json.dumps({
            "state": {"framefit5": {"mode": "FRAME_ONLY", "read_facts": [fact]}},
        })},
        {"role": "user", "content": window[0]},
    ]}
    _prune_redundant_source_metadata(body["messages"])
    assert _frame(body)["read_facts"][0]["receipts"] == [{"receipt_id": "last"}]
