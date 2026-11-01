"""Release blockers: source retention, byte identity, and late secret paths."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_code_bookmarks import _source as bookmark_source
from kernel_frame_source_snapshot import (
    _current_window, _source_coverage_line_items, editable_product_windows,
    fit_recent_read_windows, frame_source_snapshot, public_check_source_windows,
    recent_editable_read_windows,
)
from kernel_request_source_window import parse_model_source_window
from stateframe.workcopy import frame_source_path_allowed


def _window(root: Path, path: str, start: int, end: int) -> dict[str, object]:
    raw = (root / path).read_bytes()
    rendered = _current_window(root, path, start, end,
                               "sha256:" + hashlib.sha256(raw).hexdigest())
    assert rendered is not None
    return {"path": path, "start_line": start, "end_line": end,
            "content": rendered[0], "source_chars": len(rendered[1])}


def _visible(body: dict) -> set[tuple[str, int, int]]:
    return {(meta["path"], span["start_line"], span["end_line"])
            for message in body["messages"]
            if (meta := parse_model_source_window(message.get("content")))
            for span in meta["ranges"]}


def test_whole_file_upgrades_never_displace_baseline_windows(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    rng = random.Random(14)
    for case in range(80):
        root = tmp_path / str(case)
        root.mkdir()
        paths = [f"module_{index}.py" for index in range(rng.randint(2, 7))]
        for path in paths:
            (root / path).write_text("value = 1\n" + "#" + "x" * rng.randint(0, 3500) + "\n")
        recent = [_window(root, path, 1, 1) for path in paths[-3:]]
        whole = [_window(root, path, 1, 2) for path in paths]
        for row in whole:
            row["product_source"] = True
            row["bookmark"] = True
        body = {"messages": [{"role": "user", "content": "task"}], "tools": []}
        monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
        minimum = fit_recent_read_windows(body, recent, fits=lambda _body: True)
        budget = len(json.dumps(minimum).encode()) + rng.randint(0, 3000)
        def fits(candidate):
            return len(json.dumps(candidate).encode()) <= budget

        baseline_021 = fit_recent_read_windows(body, recent, fits=fits)
        monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
        upgraded = fit_recent_read_windows(body, recent, product_windows=whole,
                                           editable_paths=paths, fits=fits)
        assert _visible(baseline_021) <= _visible(upgraded), case
        assert len(json.dumps(upgraded).encode()) <= budget


def test_oversized_whole_file_does_not_block_later_small_upgrade(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    (tmp_path / "huge.py").write_text("value = 1\n#" + "x" * 5000 + "\n")
    (tmp_path / "small.py").write_text("value = 2\n# small\n")
    recent = [_window(tmp_path, "huge.py", 1, 1)]
    whole = [_window(tmp_path, path, 1, 2) for path in ("huge.py", "small.py")]
    body = {"messages": [{"role": "user", "content": "task"}], "tools": []}
    baseline = fit_recent_read_windows(body, recent, fits=lambda _body: True)
    budget = len(json.dumps(baseline).encode()) + 1500
    fitted = fit_recent_read_windows(body, recent, product_windows=whole,
                                     fits=lambda candidate: len(json.dumps(candidate).encode()) <= budget)
    assert ("huge.py", 1, 1) in _visible(fitted)
    assert ("small.py", 1, 2) in _visible(fitted)
    assert ("huge.py", 1, 2) not in _visible(fitted)


def test_switches_off_matches_021_protected_window_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "off")
    monkeypatch.setenv("STATEFRAME_READ_LOCKOUT", "off")
    (tmp_path / "module.py").write_text("value = 1\n")
    window = _window(tmp_path, "module.py", 1, 1)
    protected = {"role": "user", "content": window["content"]}
    body = {"messages": [protected], "tools": []}
    actual = fit_recent_read_windows(body, [window], protected_messages=[protected],
                                     fits=lambda _body: True)
    expected_021 = {"messages": [protected, {"role": "user", "content":
                    "kept 1 recent reads; about 10 chars of source fit"}], "tools": []}
    assert json.dumps(actual).encode() == json.dumps(expected_021).encode()


@pytest.mark.parametrize("secret", ["pkg/credentials.py", "pkg/.env.py"])
def test_late_secret_path_excluded_from_every_frame_source_route(tmp_path, monkeypatch, secret):
    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    target = tmp_path / secret
    target.parent.mkdir()
    target.write_text("password = 'TOP_SECRET_RELEASE14'\n")
    raw = target.read_bytes()
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    fact = {"path": secret, "revision": revision, "lines": {"start": 1, "end": 1},
            "content_sha256": hashlib.sha256(raw).hexdigest()}
    runtime = SimpleNamespace(owner=SimpleNamespace(root=tmp_path, task_row={
        "editable_paths": [secret], "public_check_paths": [secret]}),
        edit_contract={"edit_authority": {"editable_paths": [secret]}})
    frame = {"read_facts": [], "action_ledger": {"reads": [fact]},
             "completion_contract": {"required_public_checks": [secret]},
             "applied_edits": []}
    assert _current_window(tmp_path, secret, 1, 1, revision) is None
    assert bookmark_source(tmp_path, secret) is None
    assert recent_editable_read_windows(runtime, frame) == []
    assert editable_product_windows(runtime, frame, ())[0] == []
    assert public_check_source_windows(runtime, frame) == []
    assert _source_coverage_line_items([], [secret]) == []
    assert frame_source_snapshot(SimpleNamespace(owner=runtime.owner,
           _empirical_frame_sources={secret: {"content": target.read_text()}}), None) == {}


def test_frame_source_paths_all_use_shared_workcopy_predicate(monkeypatch):
    checked = []
    original = frame_source_path_allowed

    def observe(path):
        checked.append(path)
        return original(path)

    import kernel_frame_source_snapshot as snapshot
    import kernel_code_bookmarks as bookmarks
    monkeypatch.setattr(snapshot, "frame_source_path_allowed", observe)
    monkeypatch.setattr(bookmarks, "frame_source_path_allowed", observe, raising=False)
    assert not original("pkg/.env.py") and not original("pkg/credentials.py")
    for path in ("pkg/.env.py", "pkg/credentials.py"):
        assert snapshot._current_window(Path("/tmp"), path, 1, 1, "sha256:x") is None
        assert snapshot._source_coverage_line_items([], [path]) == []
        assert bookmarks._source(Path("/tmp"), path) is None
    assert set(checked) >= {"pkg/.env.py", "pkg/credentials.py"}
    for entrypoint in (_current_window, recent_editable_read_windows,
                       editable_product_windows, public_check_source_windows,
                       _source_coverage_line_items, fit_recent_read_windows,
                       frame_source_snapshot, bookmark_source):
        assert "frame_source_path_allowed" in inspect.getsource(entrypoint), entrypoint


def test_read_free_checkpoint_kill_and_resume_preserves_later_notice(tmp_path, monkeypatch):
    from kernel_continuation_restart import WorkerResumeError, restore_provider_request_count
    from kernel_read_exec import execute_read
    from kernel_stateframe_runtime import note_provider_request

    monkeypatch.setenv("STATEFRAME_CODE_BOOKMARKS", "on")
    monkeypatch.setenv("STATEFRAME_DELIVER_REQUESTED_BYTES", "on")
    monkeypatch.setenv("KERNEL_CONTINUATION_RESTART", "1")
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n")
    marker = tmp_path / "checkpoint.json"
    child = """
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path[:0] = sys.argv[1:3]
import kernel_continuation_restart as restart
import kernel_frame_series_replay, kernel_native_restart, kernel_working_notes_restart
from kernel_stateframe_runtime import note_provider_request
marker = Path(sys.argv[3])
restart._valid_worker_state_fields = lambda _value: True
restart.runtime_fields = lambda _value: ()
restart._checkpoint_ledger = lambda _value: {}
restart.checkpoint_conversation = lambda _value: {}
restart.capture_owner_restore = lambda _value: {}
restart._register_checkpoint = lambda *_args: None
restart.first_green_checkpoint_section = lambda _value: None
restart.post_green_checkpoint_section = lambda _value: None
restart.resume_work_checkpoint_section = lambda _value: None
kernel_native_restart.snapshot = lambda _value: None
kernel_frame_series_replay.snapshot_series = lambda _value: None
kernel_working_notes_restart.snapshot = lambda _value: None
def emit(_kind, **fields):
    marker.write_text(json.dumps(fields['checkpoint']))
    return fields
runtime = SimpleNamespace(owner=SimpleNamespace(current_step=4),
    step_ledger=SimpleNamespace(pending=None), live_projection=None,
    frame_series=None, decision_state=None, native_batch=None,
    generated=None, retained=None, actions=SimpleNamespace(actions=[]),
    _trimmed=set(), binding=lambda: {'product_revision': 'rev1'},
    journal=SimpleNamespace(emit=emit))
for _ in range(4):
    note_provider_request(runtime, [])
restart.checkpoint(runtime, [], {})
os._exit(0)
"""
    root = Path(__file__).resolve().parents[1] / "src"
    subprocess.run([sys.executable, "-c", child, str(root),
                    str(root / "stateframe/engine"), str(marker)], check=True,
                   timeout=20, env=os.environ.copy())
    saved = json.loads(marker.read_text())
    assert saved["provider_request_count"] == 4
    assert "code_bookmark_replay" not in saved
    with pytest.raises(WorkerResumeError, match="invalid provider request count"):
        restore_provider_request_count(SimpleNamespace(),
                                       {"provider_request_count": None})

    def measurement(count):
        return SimpleNamespace(owner=SimpleNamespace(workspace_revision=1),
                               retained=None, step_ledger=None,
                               _frame_budget_request_count=count)

    uninterrupted = measurement(4)
    resumed = measurement(0)
    restore_provider_request_count(resumed, saved)
    assert resumed._frame_budget_request_count == 4

    def two_reads(runtime):
        status, first, _ = execute_read(tmp_path, source, "module.py", "module.py", 0,
                                        args={"path": "module.py"}, measurement=runtime,
                                        step=5, delivery={})
        assert status == "OK"
        note_provider_request(runtime, [{"role": "tool", "content": first}])
        note_provider_request(runtime, [])
        status, repeated, _ = execute_read(tmp_path, source, "module.py", "module.py", 0,
                                           args={"path": "module.py"}, measurement=runtime,
                                           step=6, delivery={})
        assert status == "OK"
        return repeated.encode()

    actual = two_reads(resumed)
    assert actual == two_reads(uninterrupted)
    assert b"unchanged since request 5" in actual
