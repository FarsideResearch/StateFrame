import lzma
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import kernel_fl_checkpoint_capture as capture
from kernel_continuation_journal import ContinuationJournal


def test_writer_reuses_prefix_then_rebuilds_after_reopen_and_seal(tmp_path, monkeypatch):
    journal = ContinuationJournal(tmp_path, "task", {})
    path = journal.path
    scan = Mock(wraps=capture._scan_anchor)
    monkeypatch.setattr(capture, "_scan_anchor", scan)
    try:
        for i in range(4):
            text = f"wire {i}"
            journal.emit("request_submitted", request_id=str(i), payload_utf8=text)
            monkeypatch.setenv("STATEFRAME_FL_WRITER_ANCHOR", "off")
            baseline = capture._journal_capture(path, 1 << 20)
            monkeypatch.setenv("STATEFRAME_FL_WRITER_ANCHOR", "on")
            assert capture._journal_capture(path, 1 << 20, journal=journal) == baseline
            assert scan.call_count == 1
    finally:
        journal.close()
    reopened = ContinuationJournal.resume(path, task_id="task", manifest={}, expected_head=journal.head)
    try:
        assert capture._journal_capture(path, 1 << 20, journal=reopened) == baseline
        assert scan.call_count == 2
    finally:
        reopened.close()
    path.with_suffix(".jsonl.xz").write_bytes(lzma.compress(path.read_bytes()))
    path.unlink()
    assert capture._journal_capture(path, 1 << 20, journal=reopened) == baseline
    assert scan.call_count == 3


def test_truncated_journal_cannot_reuse_writer_anchor(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_FL_WRITER_ANCHOR", "on")
    journal = ContinuationJournal(tmp_path, "task", {})
    try:
        journal.emit("request_submitted", request_id="r", payload_utf8="wire")
        capture._journal_capture(journal.path, 1 << 20, journal=journal)
        journal.path.write_bytes(b"")
        with pytest.raises(capture.CaptureBoundaryError):
            capture._journal_capture(journal.path, 1 << 20, journal=journal)
    finally:
        journal.close()


def test_capture_bound_failure_detaches_writer_anchor_from_live_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_FL_CHECKPOINT_CAPTURE", "on")
    monkeypatch.setenv("STATEFRAME_FL_WRITER_ANCHOR", "on")
    monkeypatch.setenv("STATEFRAME_FL_CAPTURE_EVERY_K", "1")
    monkeypatch.setenv("STATEFRAME_FL_CAPTURE_MAX_BYTES", "4096")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "too-large.bin").write_bytes(b"x" * 8192)
    journal = ContinuationJournal(tmp_path, "task", {})
    runtime = SimpleNamespace(journal=journal, owner=SimpleNamespace(root=workspace))
    try:
        journal.emit("request_submitted", request_id="r", payload_utf8="wire")
        capture._journal_capture(journal.path, 4096, journal=journal)
        assert journal._fl_writer_anchor is not None
        capture.observe_settled_boundary(runtime, "done", "finish")

        assert runtime._fl_checkpoint_controller.disabled
        assert getattr(journal, "_fl_writer_anchor", None) is None

        stamp_calls = []
        original_stamp = capture._file_stamp

        def count_stamp(info):
            stamp_calls.append(info)
            return original_stamp(info)

        monkeypatch.setattr(capture, "_file_stamp", count_stamp)
        journal.emit("after_capture_disabled", dispatch=False)
        assert stamp_calls == []
    finally:
        journal.close()


@pytest.mark.parametrize("workspace_limit", ("files", "bytes"))
@pytest.mark.parametrize("writer_anchor_enabled", (False, True), ids=("anchor-off", "anchor-on"))
def test_capture_limit_failure_journal_body_tracks_writer_anchor_switch(
        tmp_path, monkeypatch, workspace_limit, writer_anchor_enabled):
    monkeypatch.setenv("STATEFRAME_FL_CHECKPOINT_CAPTURE", "on")
    monkeypatch.setenv("STATEFRAME_FL_CAPTURE_EVERY_K", "1")
    monkeypatch.setenv("STATEFRAME_FL_WRITER_ANCHOR", "on" if writer_anchor_enabled else "off")
    monkeypatch.delenv("STATEFRAME_FL_CAPTURE_MAX_FILES", raising=False)
    monkeypatch.delenv("STATEFRAME_FL_CAPTURE_MAX_BYTES", raising=False)
    assert capture._limits()[2:] == (64, 4 * 1024 * 1024)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    if workspace_limit == "files":
        for index in range(65):
            (workspace / f"file-{index:02d}.txt").write_text("x", encoding="utf-8")
    else:
        (workspace / "too-large.bin").write_bytes(b"x" * (4 * 1024 * 1024 + 1))

    emitted = []
    journal = SimpleNamespace(
        path=tmp_path / "run" / "events.jsonl",
        _fl_writer_anchor=object() if writer_anchor_enabled else None,
        emit=lambda kind, **fields: emitted.append({"kind": kind, **fields}),
    )
    runtime = SimpleNamespace(journal=journal, owner=SimpleNamespace(root=workspace))
    detach = Mock(wraps=capture._detach_writer_anchor)
    monkeypatch.setattr(capture, "_fl04_api", lambda: SimpleNamespace(
        SnapshotExporter=lambda _path: object(), SnapshotHook=lambda **_kwargs: object(),
        bytes_digest=lambda _data: "sha256:" + "0" * 64,
        WorkspaceFile=lambda path, _digest, size, _mode, _kind: SimpleNamespace(
            path=path, size=size)))
    monkeypatch.setattr(capture, "_journal_capture", lambda *_args, **_kwargs: (
        {"sequence": 1, "payload_utf8": "wire"}, "sha256:" + "0" * 64))
    monkeypatch.setattr(capture, "_detach_writer_anchor", detach)

    capture.observe_settled_boundary(runtime, "tool", "completed")

    legacy_event_body = {
        "kind": "fl_checkpoint_capture_disabled",
        "producer": "kernel_fl_checkpoint_capture",
        "event_schema": capture.SCHEMA,
        "reason": "capture_boundary_error",
        "error": "CaptureBoundaryError: solver workspace exceeds capture bound",
        "dispatch": False,
    }
    expected_event_body = dict(legacy_event_body)
    if writer_anchor_enabled:
        expected_event_body["writer_anchor_detached"] = True
    assert emitted == [expected_event_body]
    assert runtime._fl_checkpoint_controller.disabled
    assert detach.call_count == int(writer_anchor_enabled)
    assert getattr(journal, "_fl_writer_anchor", None) is None


@pytest.mark.host_sandbox
def test_explicitly_disabled_e28_switches_preserve_every_base_request(tmp_path, tmp_path_factory):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from _stateframe_home import isolated_stateframe_home
    from tools.pinned_tree import pinned_tree

    base = pinned_tree("1886f919", tmp_path_factory)
    candidate = Path(__file__).resolve().parents[1]
    e28_switches = (
        "STATEFRAME_ALLOCATOR_SKIP_NORMALIZATION",
        "STATEFRAME_CATALOG_COMPARE_BYTES",
        "STATEFRAME_CATALOG_ENTRY_BLOBS",
        "STATEFRAME_CATALOG_SKIP_CLEAN_WRITE",
        "STATEFRAME_CHECKPOINT_AFTER_CHECK",
        "STATEFRAME_DEGENERATION_GREEDY_REPLAN",
        "STATEFRAME_DEGENERATION_REPLAN",
        "STATEFRAME_FL_SINGLE_PASS_ANCHOR",
        "STATEFRAME_FL_WRITER_ANCHOR",
        "STATEFRAME_SHARED_CHECK_PASS_ONLY",
        "STATEFRAME_STABLE_WIRE_CURRENT_ONLY",
    )
    driver = """
import json, sys
from pathlib import Path
import pytest
from stateframe import runner
from test_request_determinism import EXAMPLES, _run_scripted_example
with pytest.MonkeyPatch.context() as patch:
    patch.setenv("SF_LAB_DETFIX3", "on")
    patch.setattr(runner.test_interpreter, "choose", lambda *a, **kw: (sys.executable, "parity"))
    hashes, _, _ = _run_scripted_example(patch, "discount_bug", EXAMPLES["discount_bug"],
                                          Path(sys.argv[1]))
print(json.dumps(hashes))
"""
    hashes = []
    for tree, label in ((base, "base"), (candidate, "candidate")):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STATEFRAME_")}
        env["STATEFRAME_HOME"] = isolated_stateframe_home()
        env["PYTHONPATH"] = os.pathsep.join(str(tree / subdir)
                                            for subdir in ("tests", "src", "src/stateframe/engine"))
        if label == "candidate":
            env.update(dict.fromkeys(e28_switches, "off"))
        result = subprocess.run([sys.executable, "-c", driver, str(tmp_path / label)],
                                cwd=tree, env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-4000:]
        hashes.append(json.loads(result.stdout.splitlines()[-1]))
    assert hashes[0] and hashes[0] == hashes[1]
