"""Recorded source presentation from the bm9_a request stream."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from kernel_code_bookmarks import collect, resolve, windows
from kernel_context import token_estimate
from kernel_frame_source_snapshot import (
    _current_window, fit_recent_read_windows, recent_editable_read_windows,
)
from kernel_request_source_window import parse_model_source_window, verify_model_source_window
from kernel_stateframe_framefit5 import _serialize_state_message, _state_payload


FIXTURE = Path(__file__).parent / "fixtures" / "pin_bm9_a"


def _recorded_request(number: int):
    return json.loads((FIXTURE / f"{number:04}.request.json").read_text())


def _state_message(body):
    return next(message for message in body["messages"]
                if str(message.get("content", "")).startswith("[StateFrame]\n"))


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for fixture in (FIXTURE / "source").rglob("*.py.txt"):
        target = root / fixture.relative_to(FIXTURE / "source").with_suffix("")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(fixture.read_bytes())
    return root


def _rebuild(number: int, root: Path):
    body = _recorded_request(number)
    message = _state_message(body)
    payload, state = _state_payload(message)
    frame = state["framefit5"]
    current = [row for row in frame["code_bookmarks"]
               if resolve(root, row)["status"] == "current"]
    frame["code_bookmarks"] = current
    message["content"] = _serialize_state_message(payload)
    selected = windows(root, current)
    fitted = fit_recent_read_windows(
        body, selected, fits=lambda candidate: token_estimate(candidate) <= 33_344,
    )
    return body, fitted, selected


def test_recorded_requests_show_whole_small_files_and_final_coverage(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root = _source_root(tmp_path)
    for number, required in ((5, "pkg/cli.py"), (21, "pkg/registry.py")):
        before, after, selected = _rebuild(number, root)
        assert selected
        assert token_estimate(after) <= 33_344
        contents = [str(message.get("content", "")) for message in after["messages"]]
        coverage = next(content for content in contents
                        if content.startswith("[Source coverage]\n"))
        assert required in coverage
        assert f"{required} (" in coverage
        expected_lines = 11 if number == 5 else 12
        assert f"all {expected_lines} lines are shown below" in coverage
        assert "not resolved in current source" not in "\n".join(contents)
        for row in selected:
            assert "read the symbol or range for more" not in row["content"]
            metadata = parse_model_source_window(row["content"])
            assert metadata is not None and metadata["status"] == "COMPLETE"
            assert verify_model_source_window(
                row["content"], path=row["path"],
                source=(root / row["path"]).read_text(),
            )
        assert len(after["messages"]) <= len(before["messages"]) + 1


def test_recorded_unresolved_bookmark_frees_a_slot_with_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root = _source_root(tmp_path)
    body = _recorded_request(5)
    frame = _state_payload(_state_message(body))[1]["framefit5"]
    paths = [row["path"] for row in frame["read_facts"]]
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=root, task_row={}),
        edit_contract={"edit_authority": {"editable_paths": paths}},
        _code_bookmark_carry=frame["code_bookmarks"],
        _framefit_action_context={"ledger": {"rows": []}},
    )
    decision = {}
    current = collect(runtime, frame, decision=decision)
    assert current
    assert all(resolve(root, row)["status"] == "current"
               for row in current)
    assert {"path": "pkg/__init__.py", "symbol": "worker.Worker",
            "reason": "unresolved_symbol"} in decision["drops"]


def test_recent_small_file_prefers_complete_with_window_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    root = tmp_path / "source"
    root.mkdir()
    target = root / "tiny.py"
    target.write_text("def value():\n    return 1\n")
    raw = target.read_bytes()
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    original = "    return 1\n"
    frame = {"action_ledger": {"reads": [{
        "kind": "read", "path": "tiny.py", "revision": revision,
        "lines": {"start": 2, "end": 2},
        "content": original, "content_sha256": hashlib.sha256(original.encode()).hexdigest(),
    }]}}
    runtime = SimpleNamespace(
        owner=SimpleNamespace(root=root, task_row={"editable_paths": ["tiny.py"]}),
        edit_contract={"edit_authority": {"editable_paths": ["tiny.py"]}},
        _framefit_action_context={"ledger": {"rows": []}},
    )
    row = recent_editable_read_windows(runtime, frame)[0]
    assert (row["start_line"], row["end_line"]) == (1, 2)
    assert parse_model_source_window(row["content"])["status"] == "COMPLETE"
    assert parse_model_source_window(row["compact_content"])["status"] == "CONTINUATION_AVAILABLE"


def test_partial_coverage_reports_changed_revision_and_omitted_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    target = tmp_path / "large.py"
    target.write_text("".join(f"LINE_{index} = {index}\n" for index in range(100)))
    revision = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
    projected = _current_window(tmp_path, "large.py", 10, 20, revision)
    assert projected is not None
    frame = {"mode": "FRAME_ONLY", "read_facts": [{"path": "large.py", "read_revision": "sha256:old",
                             "current_revision": revision}]}
    body = {"messages": [{"role": "user", "content": _serialize_state_message(
        {"state": {"framefit5": frame}})}], "tools": []}
    row = {"path": "large.py", "start_line": 10, "end_line": 20,
           "content": projected[0], "source_chars": len(projected[1])}
    fitted = fit_recent_read_windows(body, [row], fits=lambda _candidate: True)
    coverage = next(message["content"] for message in fitted["messages"]
                    if str(message.get("content", "")).startswith("[Source coverage]\n"))
    assert ("large.py (100 lines, changed since last read): lines 10-20 shown below; "
            "other lines not shown.") in coverage


def test_bookmarks_off_fitter_matches_recorded_baseline_bytes(monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    body = _recorded_request(21)
    fitted = fit_recent_read_windows(body, [], fits=lambda _candidate: True)
    raw = json.dumps(fitted, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode()
    assert len(raw) == 76164
    assert hashlib.sha256(raw).hexdigest() == (
        "a3c4ca85015c2726e8d46753d038646f7e7bb30fa2fd740ee12c6bec48432a92"
    )
