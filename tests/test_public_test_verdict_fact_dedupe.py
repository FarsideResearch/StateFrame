"""The normalized public verdict row is content-deduplicated per runtime."""
from __future__ import annotations

import copy
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernel_continuation_restart as restart
from kernel_stateframe_runtime import StateFrameRuntime
from test_rp3_completion_evidence_work import _runtime, _switches


KIND = "public_test_verdict_fact_normalized"
pytestmark = pytest.mark.usefixtures("isolated_pilot_context")


def _run_python(source_root, code, *args, env=None, timeout=180):
    command = [
        "bash", "-c", 'ulimit -v 16000000; exec nice -n 15 "$@"',
        "bash", sys.executable, "-c", code, str(source_root), *map(str, args),
    ]
    return subprocess.run(command, cwd=source_root, env=env, capture_output=True,
                          text=True, timeout=timeout)


def _rows(runtime):
    reader = getattr(runtime.journal, "records", None)
    if callable(reader):
        return [row for row in reader() if row.get("kind") == KIND]
    path = getattr(runtime.journal, "path", None)
    if path is not None and Path(path).is_file():
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("kind") == KIND:
                rows.append(row)
        return rows
    return [row for row in getattr(runtime.journal, "events", ())
            if row.get("kind") == KIND]


def _kind_rows(runtime, kind):
    reader = getattr(runtime.journal, "records", None)
    if callable(reader):
        return [row for row in reader() if row.get("kind") == kind]
    path = getattr(runtime.journal, "path", None)
    if path is not None and Path(path).is_file():
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line and json.loads(line).get("kind") == kind]
    return [row for row in getattr(runtime.journal, "events", ())
            if row.get("kind") == kind]


def _rebind_receipts(runtime, binding):
    for receipt in runtime.test_receipts:
        receipt["binding"] = copy.deepcopy(binding)
        receipt["after_binding"] = copy.deepcopy(binding)
        receipt["binding_stable"] = True


def _durable_runtime(tmp_path, monkeypatch):
    from kernel_continuation_journal import ContinuationJournal

    runtime = _runtime(tmp_path, monkeypatch)
    parent = tmp_path / "continuation-journal"
    parent.mkdir()
    runtime.journal = ContinuationJournal(
        parent, "completion-test", {"task": "public-verdict-dedupe-test"})
    return runtime


def test_unchanged_verdict_and_binding_emit_one_normalized_row(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)

    verdicts = [runtime.public_test_verdict() for _ in range(9)]

    assert {verdict.state for verdict in verdicts} == {"SATISFIED"}
    rows = _rows(runtime)
    assert len(rows) == 1
    assert {key: value for key, value in rows[0].items()
            if key not in {"schema", "kind", "run_id", "task_id", "event_id", "sequence",
                           "timestamp_ns", "journal_format", "previous_hash", "record_hash"}} == {
        "dispatch": False, "producer": "kernel_stateframe_runtime",
    }


@pytest.mark.parametrize("telemetry_kind", ["turn_timing", "action_envelope_metrics"])
def test_suppressed_fact_keeps_base_fsync_durability(
        tmp_path, tmp_path_factory, telemetry_kind):
    from tools.pinned_tree import pinned_tree

    baseline = pinned_tree("014bb2aded3f592c1490205d51a8854597700727",
                           tmp_path_factory)
    child = r'''import json, os, sys
from pathlib import Path
import pytest
source, case = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
kind = sys.argv[3]
sys.path[:0] = [str(source / "src/stateframe/engine"), str(source / "src"),
                str(source / "tests")]
from kernel_continuation_journal import ContinuationJournal
from test_rp3_completion_evidence_work import _runtime, _switches
patch = pytest.MonkeyPatch()
_switches(patch, "on")
case.mkdir(parents=True, exist_ok=True)
runtime = _runtime(case, patch)
parent = case / "continuation-journal"
parent.mkdir()
runtime.journal = ContinuationJournal(parent, "durability-probe", {})
assert runtime.public_test_verdict().state == "SATISFIED"
journal = runtime.journal
journal.emit(kind, step=1)
telemetry_end = journal.path.stat().st_size
synced_sizes = []
real_fsync = os.fsync
def tracked(fd):
    if fd == journal._fd:
        synced_sizes.append(os.fstat(fd).st_size)
    return real_fsync(fd)
os.fsync = tracked
assert runtime.public_test_verdict().state == "SATISFIED"
print(json.dumps({"fsyncs": len(synced_sizes),
                  "telemetry_durable": bool(synced_sizes and synced_sizes[-1] >= telemetry_end)}))
'''
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    current = Path(__file__).resolve().parents[1]
    results = []
    for label, root in (("candidate", current), ("baseline", baseline)):
        case = tmp_path / f"durability-{telemetry_kind}-{label}"
        process = _run_python(root, child, case, telemetry_kind, env=env)
        assert process.returncode == 0, process.stdout + process.stderr
        results.append(json.loads(process.stdout.strip().splitlines()[-1]))
    assert results == [{"fsyncs": 1, "telemetry_durable": True}] * 2


def test_suppressed_fact_kill_keeps_pending_telemetry(tmp_path, monkeypatch):
    child = r'''import json, os, signal, sys
from pathlib import Path
import pytest
source, case = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path[:0] = [str(source / "src/stateframe/engine"), str(source / "src"),
                str(source / "tests")]
from kernel_continuation_journal import ContinuationJournal
from test_rp3_completion_evidence_work import _runtime, _switches
patch = pytest.MonkeyPatch()
_switches(patch, "on")
case.mkdir(parents=True, exist_ok=True)
runtime = _runtime(case, patch)
parent = case / "continuation-journal"
parent.mkdir()
runtime.journal = ContinuationJournal(parent, "suppression-kill", {})
runtime.public_test_verdict()
journal = runtime.journal
metadata = case / "crash.json"
synced_size = journal.path.stat().st_size
journal.emit("turn_timing", step=1)
telemetry_end = journal.path.stat().st_size
real_fsync = os.fsync
def tracked(fd):
    global synced_size
    if fd == journal._fd:
        synced_size = os.fstat(fd).st_size
    return real_fsync(fd)
os.fsync = tracked
assert runtime.public_test_verdict().state == "SATISFIED"
metadata.write_text(json.dumps({"path": str(journal.path),
    "synced_size": synced_size, "telemetry_end": telemetry_end}), encoding="utf-8")
os.kill(os.getpid(), signal.SIGKILL)
'''
    case = tmp_path / "suppression-kill"
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    process = _run_python(Path(__file__).resolve().parents[1], child, case, env=env)
    assert process.returncode == -signal.SIGKILL, process.stdout + process.stderr
    metadata = json.loads((case / "crash.json").read_text(encoding="utf-8"))
    path = Path(metadata["path"])
    with path.open("r+b") as journal_file:
        journal_file.truncate(metadata["synced_size"])
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert metadata["synced_size"] >= metadata["telemetry_end"]
    assert any(row.get("kind") == "turn_timing" for row in rows)


def test_suppressed_fact_without_pending_rows_does_not_fsync(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    assert runtime.public_test_verdict().state == "SATISFIED"
    calls = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1])

    assert runtime.public_test_verdict().state == "SATISFIED"
    assert calls == []


def test_failed_suppressed_fact_flush_poisons_without_changing_verdict(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    assert runtime.public_test_verdict().state == "SATISFIED"
    runtime.journal.emit("turn_timing", step=1)

    def fail_fsync(_fd):
        raise OSError("injected pending telemetry fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert runtime.journal._poisoned and runtime.journal._unsynced


def test_marker_tracks_verdict_binding_change_back_and_strict_types(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    binding = runtime.binding()
    runtime.binding = lambda **_kwargs: copy.deepcopy(binding)
    _rebind_receipts(runtime, binding)

    assert runtime.public_test_verdict().state == "SATISFIED"
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 1

    runtime.test_receipts[-1]["exit_code"] = 1
    assert runtime.public_test_verdict().state == "VIOLATED"
    assert len(_rows(runtime)) == 2
    runtime.test_receipts[-1]["exit_code"] = 0
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 3

    binding["environment_revision"] = "environment-two"
    binding["environment"] = "environment-two"
    _rebind_receipts(runtime, binding)
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 4
    binding["environment_revision"] = "environment"
    binding["environment"] = "environment"
    _rebind_receipts(runtime, binding)
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 5

    # Python considers True == 1. A strict content marker must still see this
    # full binding-content change even though the verdict remains satisfied.
    binding["marker_type_probe"] = 1
    _rebind_receipts(runtime, binding)
    assert runtime.public_test_verdict().state == "SATISFIED"
    binding["marker_type_probe"] = True
    _rebind_receipts(runtime, binding)
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 7


def test_marker_uses_full_detached_verdict_and_binding_content():
    import threading

    import kernel_stateframe_runtime as runtime_module
    from kernel_verdict import Verdict

    class Journal:
        def __init__(self):
            self.events = []
            self.sequence = 0
            self._last_durable_checkpoint_sequence = -1

        def emit(self, kind, **fields):
            self.sequence += 1
            row = {"kind": kind, "sequence": self.sequence, **fields}
            self.events.append(row)
            return row

    runtime = SimpleNamespace(
        journal=Journal(),
        _public_test_verdict_fact_marker=None,
        _public_test_verdict_fact_lock=threading.RLock(),
    )
    binding = {"nested": {"values": [1]}, "typed": True}
    verdict = Verdict("SATISFIED", ["receipt-ref"], [], binding=binding)

    runtime_module._emit_changed_public_test_verdict_fact(runtime, verdict, binding)
    runtime_module._emit_changed_public_test_verdict_fact(runtime, verdict, binding)
    assert len(runtime.journal.events) == 1

    # A verdict-only change emits even when its binding is byte-for-byte equal.
    changed_verdict = Verdict("UNRESOLVED", ["receipt-ref"], ["MISSING_SELECTOR"],
                              binding=binding)
    runtime_module._emit_changed_public_test_verdict_fact(
        runtime, changed_verdict, binding)
    assert len(runtime.journal.events) == 2

    # The stored key must be a detached immutable copy of nested binding data.
    binding["nested"]["values"].append(2)
    runtime_module._emit_changed_public_test_verdict_fact(runtime, verdict, binding)
    assert len(runtime.journal.events) == 3

    # A binding-only change emits even when the verdict content stays fixed.
    changed_binding = {"nested": {"values": [1]}, "typed": 1}
    runtime_module._emit_changed_public_test_verdict_fact(
        runtime, verdict, changed_binding)
    assert len(runtime.journal.events) == 4


def test_failed_emit_does_not_advance_marker_and_retries(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    emit = runtime.journal.emit
    attempts = []

    def fail_first(kind, **fields):
        if kind == KIND:
            attempts.append(kind)
            if len(attempts) == 1:
                raise OSError("injected pre-write failure")
        return emit(kind, **fields)

    monkeypatch.setattr(runtime.journal, "emit", fail_first)
    runtime.public_test_verdict()
    runtime.public_test_verdict()

    assert attempts == [KIND, KIND]
    assert len(_rows(runtime)) == 1


def test_two_runtimes_and_concurrent_calls_keep_markers_independent(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = _durable_runtime(first_dir, monkeypatch)
    second = _durable_runtime(second_dir, monkeypatch)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: first.public_test_verdict(), range(24)))
    assert all(result.state == "SATISFIED" for result in results)
    first_rows = _rows(first)
    assert 1 <= len(first_rows) <= 24
    assert first._public_test_verdict_fact_marker[1] in {
        row["sequence"] for row in first_rows
    }
    assert first._public_test_verdict_fact_marker[2] is first.journal
    assert second.public_test_verdict().state == "SATISFIED"
    assert len(_rows(second)) == 1


def test_marker_and_lock_are_excluded_from_runtime_pickle_state():
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime._public_test_verdict_fact_marker = ("prior", "content")
    import threading
    runtime._public_test_verdict_fact_lock = threading.RLock()

    state = runtime.__getstate__()

    assert "_public_test_verdict_fact_marker" not in state
    assert "_public_test_verdict_fact_lock" not in state
    restored = StateFrameRuntime.__new__(StateFrameRuntime)
    restored.__setstate__(state)
    assert getattr(restored, "_public_test_verdict_fact_marker", None) is None


def test_resume_clears_marker_before_reconciliation(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 1
    import kernel_candidate_restart_reconcile as candidate_reconcile
    import kernel_continuation_state as continuation_state
    import kernel_true_resume

    class StopRestore(Exception):
        pass

    monkeypatch.setattr(kernel_true_resume, "require_enabled", lambda: None)
    monkeypatch.setattr(continuation_state, "unresolved", lambda *_args: False)
    monkeypatch.setattr(candidate_reconcile, "enabled", lambda: False)
    monkeypatch.setattr(restart, "_reconcile_external_effects",
                        lambda _runtime: (_ for _ in ()).throw(StopRestore()))
    with pytest.raises(StopRestore):
        restart.restore_runtime(runtime)

    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 2


def _prepare_checkpoint_runtime(runtime, monkeypatch):
    import kernel_code_bookmarks
    import kernel_frame_series_replay
    import kernel_native_restart
    import kernel_working_notes_restart

    monkeypatch.setattr(restart, "enabled", lambda: True)
    monkeypatch.setattr(restart, "_valid_worker_state_fields", lambda _value: True)
    monkeypatch.setattr(restart, "runtime_fields", lambda _runtime: ())
    monkeypatch.setattr(restart, "_checkpoint_ledger", lambda _ledger: {})
    monkeypatch.setattr(restart, "capture_owner_restore", lambda _runtime: {})
    monkeypatch.setattr(restart, "checkpoint_conversation", lambda _messages: {})
    monkeypatch.setattr(restart, "_register_checkpoint", lambda *_args: None)
    monkeypatch.setattr(restart, "first_green_checkpoint_section", lambda _runtime: None)
    monkeypatch.setattr(restart, "post_green_checkpoint_section", lambda _runtime: None)
    monkeypatch.setattr(restart, "resume_work_checkpoint_section", lambda _runtime: None)
    monkeypatch.setattr(kernel_code_bookmarks, "enabled", lambda: False)
    monkeypatch.setattr(kernel_native_restart, "snapshot", lambda _batch: None)
    monkeypatch.setattr(kernel_frame_series_replay, "snapshot_series", lambda _series: None)
    monkeypatch.setattr(kernel_working_notes_restart, "snapshot", lambda _state: None)
    runtime.step_ledger = SimpleNamespace(pending=None)
    runtime.live_projection = None
    runtime.frame_series = None
    runtime.decision_state = None
    runtime.native_batch = None
    runtime.generated = None
    runtime.retained = None
    runtime.actions = SimpleNamespace(actions=[])
    runtime._trimmed = set()
    runtime.owner.current_step = 1


def test_checkpoint_sequence_invalidates_marker_without_forcing_fact_rows(
        tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    _prepare_checkpoint_runtime(runtime, monkeypatch)

    # A checkpoint must not compute a verdict merely to emit telemetry.
    restart.checkpoint(runtime, [], {})
    assert not _rows(runtime)
    assert runtime._public_test_verdict_fact_marker is None
    assert runtime.journal._last_durable_checkpoint_sequence == max(
        row["sequence"] for row in runtime.journal.records()
        if row["kind"] == "continuation_worker_checkpoint")

    # A normal query emits a row after the checkpoint sequence.
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 1
    first_fact = _rows(runtime)[0]
    restart.checkpoint(runtime, [], {})
    assert len(_rows(runtime)) == 1
    assert runtime._public_test_verdict_fact_marker[1] == first_fact["sequence"]
    assert (runtime._public_test_verdict_fact_marker[1]
            < runtime.journal._last_durable_checkpoint_sequence)
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 2
    assert (_rows(runtime)[-1]["sequence"]
            > runtime.journal._last_durable_checkpoint_sequence)
    restart.checkpoint(runtime, [], {})
    assert len(_rows(runtime)) == 2
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 3

    records = list(runtime.journal.records())
    checkpoints = [row for row in records
                   if row.get("kind") == "continuation_worker_checkpoint"]
    assert len(checkpoints) == 3
    first_checkpoint = checkpoints[0]
    assert not any(row.get("kind") == KIND
                   and row.get("sequence", 0) < first_checkpoint["sequence"]
                   for row in records)
    assert "_public_test_verdict_fact_marker" not in first_checkpoint["checkpoint"]
    assert not any("public_test_verdict_fact" in key
                   for key in first_checkpoint["checkpoint"].get("runtime", {}))


def test_failed_checkpoint_emit_preserves_verdict_marker(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    _prepare_checkpoint_runtime(runtime, monkeypatch)
    assert runtime.public_test_verdict().state == "SATISFIED"
    marker = runtime._public_test_verdict_fact_marker
    emit = runtime.journal.emit

    def fail_checkpoint(kind, **fields):
        if kind == "continuation_worker_checkpoint":
            raise OSError("injected checkpoint emit failure")
        return emit(kind, **fields)

    monkeypatch.setattr(runtime.journal, "emit", fail_checkpoint)
    with pytest.raises(OSError, match="injected checkpoint emit failure"):
        restart.checkpoint(runtime, [], {})

    assert runtime._public_test_verdict_fact_marker is marker
    assert runtime._public_test_verdict_fact_marker[2] is runtime.journal
    assert len(_rows(runtime)) == 1


def test_checkpoint_sequence_invalidates_marker_during_concurrent_query(
        tmp_path, monkeypatch):
    import threading

    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    _prepare_checkpoint_runtime(runtime, monkeypatch)
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 1

    durable, release = threading.Event(), threading.Event()
    emit = runtime.journal.emit
    errors = []

    def pause_after_durable_checkpoint(kind, **fields):
        row = emit(kind, **fields)
        if kind == "continuation_worker_checkpoint":
            assert runtime.journal._last_durable_checkpoint_sequence == row["sequence"]
            durable.set()
            if not release.wait(10):
                raise TimeoutError("test did not release checkpoint emit barrier")
        return row

    monkeypatch.setattr(runtime.journal, "emit", pause_after_durable_checkpoint)

    def save_checkpoint():
        try:
            restart.checkpoint(runtime, [], {})
        except BaseException as error:
            errors.append(error)

    query_done = threading.Event()

    def query_verdict():
        try:
            assert runtime.public_test_verdict().state == "SATISFIED"
        except BaseException as error:
            errors.append(error)
        finally:
            query_done.set()

    checkpoint_thread = threading.Thread(target=save_checkpoint, daemon=True)
    query_thread = None
    checkpoint_thread.start()
    query_prompt = False
    try:
        assert durable.wait(10), "checkpoint did not reach its durable emit"
        query_thread = threading.Thread(target=query_verdict, daemon=True)
        query_thread.start()
        query_prompt = query_done.wait(1)
    finally:
        release.set()
        checkpoint_thread.join(10)
        if query_thread is not None:
            query_thread.join(10)

    assert not checkpoint_thread.is_alive(), "checkpoint deadlocked"
    assert query_thread is not None and not query_thread.is_alive(), "verdict query deadlocked"
    assert query_prompt, "verdict query blocked after the durable checkpoint"
    assert query_done.is_set()
    assert not errors
    assert len(_rows(runtime)) == 2
    assert (runtime._public_test_verdict_fact_marker[1]
            > runtime.journal._last_durable_checkpoint_sequence)


def test_fact_observer_query_does_not_deadlock_changed_verdict_emit(
        tmp_path, monkeypatch):
    import threading
    import kernel_stateframe_runtime as runtime_module
    from kernel_verdict import Verdict

    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    initial = runtime.public_test_verdict()
    binding = initial.binding
    changed = Verdict("UNRESOLVED", ["receipt-ref"], ["MISSING_SELECTOR"],
                      binding=binding)

    def query_unchanged_verdict():
        runtime_module._emit_changed_public_test_verdict_fact(
            runtime, initial, binding)
        return initial

    runtime._public_test_verdict_uncached = query_unchanged_verdict
    query_done = threading.Event()
    query_results = []
    query_errors = []
    query_threads = []
    stalled = []
    observer_called = threading.Event()

    def query_verdict():
        try:
            query_results.append(runtime.public_test_verdict().state)
        except BaseException as error:
            query_errors.append(error)
        finally:
            query_done.set()

    def observe_fact(row):
        if row.get("kind") != KIND or observer_called.is_set():
            return
        observer_called.set()
        query = threading.Thread(target=query_verdict, daemon=True)
        query_threads.append(query)
        query.start()
        query.join(2)
        stalled.append(query.is_alive())

    runtime.journal.subscribe(
        "changed-verdict-query-reentry", observe_fact, kinds=(KIND,),
    )
    emitter_errors = []

    def emit_changed_verdict():
        try:
            runtime_module._emit_changed_public_test_verdict_fact(
                runtime, changed, binding)
        except BaseException as error:
            emitter_errors.append(error)

    emitter = threading.Thread(target=emit_changed_verdict, daemon=True)
    emitter.start()
    emitter.join(10)
    assert not emitter.is_alive(), "changed verdict emission deadlocked"
    assert query_done.wait(5), "observer query did not finish after fact emit"
    for query in query_threads:
        query.join(5)

    assert stalled == [False], "observer query blocked on the verdict marker lock"
    assert not query_errors and not emitter_errors
    assert query_results == ["SATISFIED"]
    fact_rows = _rows(runtime)
    assert len(fact_rows) == 2
    assert runtime._public_test_verdict_fact_marker[1] == fact_rows[-1]["sequence"]
    assert runtime._public_test_verdict_fact_marker[2] is runtime.journal


def test_late_older_fact_emit_cannot_overwrite_newer_marker(tmp_path, monkeypatch):
    import threading
    import kernel_stateframe_runtime as runtime_module
    from kernel_verdict import Verdict

    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    runtime.public_test_verdict()
    binding = runtime.binding()
    older = Verdict("UNRESOLVED", ["receipt-ref"], ["MISSING_SELECTOR"], binding=binding)
    newer = Verdict("VIOLATED", ["receipt-ref"], ["TEST_FAILED"], binding=binding)
    old_appended, release_old = threading.Event(), threading.Event()
    emit = runtime.journal.emit

    def interleaved_emit(kind, **fields):
        row = emit(kind, **fields)
        if kind == KIND and threading.current_thread().name == "older-fact":
            old_appended.set()
            assert release_old.wait(5), "test did not release older emit"
        return row

    monkeypatch.setattr(runtime.journal, "emit", interleaved_emit)
    errors = []

    def emit_fact(verdict):
        try:
            runtime_module._emit_changed_public_test_verdict_fact(runtime, verdict, binding)
        except BaseException as error:
            errors.append(error)

    old_thread = threading.Thread(target=emit_fact, args=(older,), name="older-fact")
    new_thread = threading.Thread(target=emit_fact, args=(newer,), name="newer-fact")
    old_thread.start()
    try:
        assert old_appended.wait(5), "older fact was not appended"
        new_thread.start()
        new_thread.join(5)
        assert not new_thread.is_alive(), "newer fact did not finish"
        newer_sequence = runtime._public_test_verdict_fact_marker[1]
    finally:
        release_old.set()
        old_thread.join(5)
        if new_thread.ident is not None:
            new_thread.join(5)

    assert not old_thread.is_alive(), "older fact did not finish"
    assert not errors
    marker = runtime._public_test_verdict_fact_marker
    assert marker[0] == runtime_module._public_test_verdict_fact_marker(newer, binding)
    assert marker[1] == newer_sequence
    assert marker[1] == _rows(runtime)[-1]["sequence"]


def test_checkpoint_observer_query_thread_does_not_stall_checkpoint_emit(
        tmp_path, monkeypatch):
    import threading
    import kernel_stateframe_runtime as runtime_module

    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    _prepare_checkpoint_runtime(runtime, monkeypatch)
    verdict = runtime.public_test_verdict()
    assert verdict.state == "SATISFIED"
    binding = verdict.binding

    def quick_verdict_query():
        runtime_module._emit_changed_public_test_verdict_fact(runtime, verdict, binding)
        return verdict

    runtime._public_test_verdict_uncached = quick_verdict_query
    query_started = threading.Event()
    query_done = threading.Event()
    observed = []
    errors = []

    def query_verdict():
        try:
            query_started.set()
            observed.append(runtime.public_test_verdict().state)
        except BaseException as error:
            errors.append(error)
        finally:
            query_done.set()

    def reenter(row):
        if row.get("kind") == "continuation_worker_checkpoint":
            query = threading.Thread(target=query_verdict, daemon=True)
            query.start()
            assert query_started.wait(5), "observer query did not start"
            assert not query_done.is_set(), "query finished before observer returned"

    runtime.journal.subscribe(
        "checkpoint-verdict-reentry", reenter,
        kinds=("continuation_worker_checkpoint",),
    )

    def save_checkpoint():
        try:
            restart.checkpoint(runtime, [], {})
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=save_checkpoint, daemon=True)
    worker.start()
    worker.join(10)

    assert not worker.is_alive(), "observer re-entry deadlocked checkpoint emit"
    assert not errors
    assert query_done.wait(5), "observer query did not finish after checkpoint emit"
    assert observed == ["SATISFIED"]
    records = list(runtime.journal.records())
    checkpoint = next(row for row in records
                      if row["kind"] == "continuation_worker_checkpoint")
    fact_rows = _rows(runtime)
    assert len(fact_rows) == 2
    assert fact_rows[0]["sequence"] < checkpoint["sequence"] < fact_rows[1]["sequence"]


def test_raising_checkpoint_observer_does_not_block_later_verdict_fact(
        tmp_path, monkeypatch):
    import threading

    _switches(monkeypatch, "on")
    runtime = _durable_runtime(tmp_path, monkeypatch)
    _prepare_checkpoint_runtime(runtime, monkeypatch)
    assert runtime.public_test_verdict().state == "SATISFIED"

    def fail(_row):
        raise RuntimeError("observer failure")

    runtime.journal.subscribe(
        "checkpoint-raises", fail, kinds=("continuation_worker_checkpoint",),
    )
    errors = []

    def save_checkpoint():
        try:
            restart.checkpoint(runtime, [], {})
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=save_checkpoint, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "raising observer blocked checkpoint emit"
    assert not errors
    checkpoint = next(row for row in runtime.journal.records()
                      if row["kind"] == "continuation_worker_checkpoint")
    assert runtime.journal._last_durable_checkpoint_sequence == checkpoint["sequence"]
    assert runtime.public_test_verdict().state == "SATISFIED"
    assert len(_rows(runtime)) == 2
    assert _rows(runtime)[-1]["sequence"] > checkpoint["sequence"]
    assert any(row["kind"] == "journal_observer_failed"
               for row in runtime.journal.records())


def test_checkpoint_window_kill_resumes_like_7540148c_archive(tmp_path):
    current_root = Path(__file__).resolve().parents[1]
    baseline_root = tmp_path / "baseline-7540148c"
    baseline_root.mkdir()
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "7540148c", "src"],
        cwd=current_root, capture_output=True, check=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as source:
        source.extractall(baseline_root)

    child = r'''import json, os, signal, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
test_root = Path(sys.argv[2]).resolve()
case = Path(sys.argv[3]).resolve()
sys.path[:0] = [str(source / "src/stateframe/engine"),
                str(source / "src"), str(test_root / "tests")]
import pytest
from test_public_test_verdict_fact_dedupe import (
    KIND, _durable_runtime, _prepare_checkpoint_runtime, _switches,
)
import kernel_continuation_restart as restart
patch = pytest.MonkeyPatch()
_switches(patch, "on")
case.mkdir(parents=True, exist_ok=True)
runtime = _durable_runtime(case, patch)
_prepare_checkpoint_runtime(runtime, patch)
metadata_path = case / "crash.json"
emit = runtime.journal.emit
def interrupt_after_boundary(kind, **fields):
    result = emit(kind, **fields)
    if kind in {KIND, "continuation_worker_checkpoint"}:
        metadata_path.write_text(json.dumps({
            "journal_path": str(runtime.journal.path),
            "task_id": runtime.journal.task_id,
            "manifest": runtime.journal.manifest,
            "head": runtime.journal._head,
            "killed_on": kind,
        }), encoding="utf-8")
        os.kill(os.getpid(), signal.SIGKILL)
    return result
runtime.journal.emit = interrupt_after_boundary
restart.checkpoint(runtime, [], {})
raise SystemExit("checkpoint boundary kill injection did not fire")
'''
    reader = r'''import json, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(source / "src/stateframe/engine"), str(source / "src")]
from kernel_continuation_journal import ContinuationJournal
from kernel_stateframe_resume import latest_checkpoint
metadata = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
journal = ContinuationJournal.resume(
    Path(metadata["journal_path"]), task_id=metadata["task_id"],
    manifest=metadata["manifest"], expected_head=metadata["head"])
try:
    records = list(journal.records())
    checkpoint = latest_checkpoint(records)
    event = next(row for row in reversed(records)
                 if row.get("kind") == "continuation_worker_checkpoint"
                 and row.get("checkpoint") == checkpoint)
    tail = [row["kind"] for row in records
            if (row.get("sequence", 0) > event["sequence"]
                and row.get("kind") != "continuation_writer_resumed")]
    print(json.dumps({"checkpoint_selected": True,
                      "checkpoint_step": checkpoint["step"],
                      "tail": tail}, sort_keys=True))
finally:
    journal.close()
'''

    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    outcomes = []
    for name, source_root in (("candidate", current_root), ("baseline", baseline_root)):
        case = tmp_path / f"checkpoint-window-{name}"
        process = _run_python(source_root, child, current_root, case, env=env, timeout=180)
        assert process.returncode == -signal.SIGKILL, process.stdout + process.stderr
        metadata_path = case / "crash.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["killed_on"] == "continuation_worker_checkpoint"
        resumed = _run_python(source_root, reader, metadata_path, env=env, timeout=120)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        outcomes.append(json.loads(resumed.stdout.strip().splitlines()[-1]))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0] == {"checkpoint_selected": True, "checkpoint_step": 1, "tail": []}


def test_shadow_mismatch_and_bypass_kinds_remain_per_call(tmp_path, monkeypatch):
    import kernel_test_artifact_policy as artifact_policy

    _switches(monkeypatch, "on")
    monkeypatch.setenv("STATEFRAME_PUBLIC_TEST_VERDICT_FACT", "shadow")
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    shadow = _durable_runtime(shadow_dir, monkeypatch)
    normalize = artifact_policy.normalize_public_test_bindings_for_mode

    def report_mismatch(current, receipts, mode):
        normalized = normalize(current, receipts, mode)
        return normalized[0], normalized[1], ("normalization_output",), None

    monkeypatch.setattr(artifact_policy, "normalize_public_test_bindings_for_mode", report_mismatch)
    shadow.public_test_verdict()
    shadow.public_test_verdict()
    assert len(_kind_rows(shadow, "public_test_verdict_fact_shadow_checked")) == 2
    assert len(_kind_rows(shadow, "public_test_verdict_shadow_mismatch")) == 2
    assert not _rows(shadow)

    _switches(monkeypatch, "on")
    bypass_dir = tmp_path / "bypass"
    bypass_dir.mkdir()
    bypass = _durable_runtime(bypass_dir, monkeypatch)
    monkeypatch.setattr(
        artifact_policy, "normalize_public_test_bindings_for_mode",
        lambda current, receipts, _mode: (dict(current), tuple(receipts), (), "injected_bypass"),
    )
    bypass.public_test_verdict()
    bypass.public_test_verdict()
    assert len(_kind_rows(bypass, "public_test_verdict_fact_bypass")) == 2
    assert not _rows(bypass)


def test_scripted_session_parity_within_same_build(tmp_path, monkeypatch):
    current_root = Path(__file__).resolve().parents[1]

    import json
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "README.md").write_text("Journal telemetry must not alter requests.\n")
    (repo / "tests" / "test_public_fact_fixture.py").write_text(
        "from pathlib import Path\n"
        "def test_readme():\n"
        "    assert 'alter requests' in Path('README.md').read_text()\n"
    )
    script_data = {
        "model": "scripted",
        "steps": [
            {"tool": "read", "required": True, "arguments": {"path": "README.md"}},
            {"tool": "run_tests", "required": True,
             "arguments": {"target": "tests/test_public_fact_fixture.py"}},
            {"tool": "done", "required": True,
             "arguments": {"summary": "Read the fixture and ran its test."}},
        ],
        "tokenize": {"mode": "ok"},
    }

    def run_tree(source_root, name):
        case = tmp_path / name
        case.mkdir()
        run_dir = case / "run"
        script = Script(copy.deepcopy(script_data))
        endpoint = serve(script)
        test_tmp = case / "tmp"
        test_tmp.mkdir()
        env = os.environ.copy()
        env.update({
            "STATEFRAME_HOME": str(case / "stateframe-home"),
            "STATEFRAME_PUBLIC_TEST_VERDICT_FACT": "on",
            "STATEFRAME_HOST_DONE_FACTS": "on",
            "TMPDIR": str(test_tmp),
        })
        bootstrap = (
            "import secrets,sys; secrets.token_hex=lambda n=16: ('ab'*n)[:2*n]; "
            "sys.path.insert(0, sys.argv[1]); from stateframe import cli; "
            "raise SystemExit(cli.main(sys.argv[2:]))"
        )
        args = [
            "run", "--repo", str(repo), "--task", "Read the fixture and validate it.",
            "--endpoint", f"http://127.0.0.1:{endpoint.server_port}/v1",
            "--model", "scripted", "--run-dir", str(run_dir), "--quiet", "--json",
            "--max-turns", "8", "--runner-kind", "generic",
            "--test-cmd",
            "bash -c 'ulimit -v 16000000; nice -n 15 "
            "~/work/venv-sf/bin/python -m pytest -n 2 -q "
            "--basetemp=\"$TMPDIR/pytest-public-fact\" tests/test_public_fact_fixture.py'",
            "--test-python", sys.executable,
        ]
        try:
            process = subprocess.run(
                ["bash", "-c", 'ulimit -v 16000000; exec nice -n 15 "$@"',
                 "bash", sys.executable, "-c", bootstrap, str(source_root / "src"), *args],
                cwd=source_root, env=env, capture_output=True, text=True, timeout=900,
            )
        finally:
            endpoint.shutdown()
            endpoint.server_close()
        assert process.returncode in {0, 1}, process.stdout[-1000:] + process.stderr[-1000:]
        script.assert_consumed()
        assert (run_dir / "wire").is_dir()
        from tests._journal import measurement_events
        events = measurement_events(run_dir)
        requests = {path.relative_to(run_dir).as_posix(): path.read_bytes()
                    for path in sorted(run_dir.rglob("*.request.json"))}
        assert requests
        return events, requests

    first_events, first_requests = run_tree(current_root, "first")
    second_events, second_requests = run_tree(current_root, "second")

    # These envelope fields are stable across repeated runs; journal identity,
    # timing, workspace and content-blob fields are run-specific.
    stable_fields = ("schema", "kind", "producer", "dispatch")
    first_rows = [{key: row[key] for key in stable_fields if key in row}
                  for row in first_events]
    second_rows = [{key: row[key] for key in stable_fields if key in row}
                   for row in second_events]
    assert first_rows == second_rows
    first_fact_count = sum(row.get("kind") == KIND for row in first_events)
    second_fact_count = sum(row.get("kind") == KIND for row in second_events)
    assert first_fact_count == second_fact_count > 0
    assert first_requests == second_requests


@pytest.mark.parametrize("crash_point", ["before_emit", "before_fsync"])
def test_crash_recovers_current_verdict_and_valid_chain(tmp_path, monkeypatch, crash_point):
    _switches(monkeypatch, "on")
    source_root = Path(__file__).resolve().parents[1]
    case = tmp_path / crash_point
    case.mkdir()
    metadata_path = case / "crash.json"
    child = r'''import json, os, signal, sys
from pathlib import Path
import pytest
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "tests"))
sys.path.insert(0, str(source / "src" / "stateframe" / "engine"))
sys.path.insert(0, str(source / "src"))
from test_public_test_verdict_fact_dedupe import _durable_runtime, _switches
runtime_root = Path(sys.argv[2])
phase, metadata_path = sys.argv[3], Path(sys.argv[4])
patch = pytest.MonkeyPatch()
_switches(patch, "on")
runtime = _durable_runtime(runtime_root, patch)
runtime.public_test_verdict()
before_size = runtime.journal.path.stat().st_size
previous_head = runtime.journal._head
runtime.test_receipts[-1]["exit_code"] = 1
metadata_path.write_text(json.dumps({
    "journal_path": str(runtime.journal.path), "task_id": runtime.journal.task_id,
    "manifest": runtime.journal.manifest, "previous_head": previous_head,
    "before_size": before_size,
}), encoding="utf-8")
if phase == "before_emit":
    import kernel_stateframe_runtime as runtime_module
    def stop_before_emit(*_args, **_kwargs):
        os.kill(os.getpid(), signal.SIGKILL)
    runtime_module._emit_public_test_verdict_fact_event = stop_before_emit
else:
    original_fsync = os.fsync
    journal_stat = runtime.journal.path.stat()
    def stop_after_append(fd):
        descriptor = os.fstat(fd)
        if (descriptor.st_dev == journal_stat.st_dev and descriptor.st_ino == journal_stat.st_ino
                and descriptor.st_size > before_size):
            os.kill(os.getpid(), signal.SIGKILL)
        return original_fsync(fd)
    os.fsync = stop_after_append
runtime.public_test_verdict()
raise SystemExit("crash injection did not fire")
'''
    child_runtime = case / "runtime"
    child_runtime.mkdir()
    env = os.environ.copy()
    env["STATEFRAME_HOME"] = str(case / "stateframe-home")
    process = _run_python(source_root, child, child_runtime, crash_point, metadata_path,
                          env=env, timeout=120)
    assert process.returncode == -signal.SIGKILL, process.stdout + process.stderr
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    from kernel_continuation_journal import ContinuationJournal
    journal_path = Path(metadata["journal_path"])
    if crash_point == "before_emit":
        resumed = ContinuationJournal.resume(
            journal_path, task_id=metadata["task_id"], manifest=metadata["manifest"],
            expected_head=metadata["previous_head"],
        )
    else:
        # A complete row may have reached the page cache without its fsync.
        # The documented recovery path validates the whole chain before appending.
        resumed = ContinuationJournal.resume(
            journal_path, task_id=metadata["task_id"], manifest=metadata["manifest"],
            allow_unanchored_recovery=True,
        )
    try:
        assert [row for row in resumed.records() if row.get("kind") == KIND]
        second_case = case / "resumed-runtime"
        second_case.mkdir()
        second_patch = pytest.MonkeyPatch()
        _switches(second_patch, "on")
        runtime = _durable_runtime(second_case, second_patch)
        runtime.journal = resumed
        runtime.test_receipts[-1]["exit_code"] = 1
        assert runtime.public_test_verdict().state == "VIOLATED"
        rows = [row for row in resumed.records() if row.get("kind") == KIND]
        assert len(rows) >= 2
        assert rows[-1]["sequence"] > rows[0]["sequence"]
        # Full validation exercises contiguous sequence and previous/hash links.
        assert list(resumed.records())
    finally:
        resumed.close()


def test_old_and_new_journal_writers_read_each_others_normalized_rows(
        tmp_path, tmp_path_factory):
    from tools.pinned_tree import pinned_tree

    baseline_root = pinned_tree("014bb2aded3f592c1490205d51a8854597700727",
                                tmp_path_factory)
    current_root = Path(__file__).resolve().parents[1]
    roots = [current_root] if current_root == baseline_root else [current_root, baseline_root]
    artifacts = []

    create = r'''import json, sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "src" / "stateframe" / "engine"))
sys.path.insert(0, str(source / "src"))
from kernel_continuation_journal import ContinuationJournal
parent = Path(sys.argv[2])
manifest = {"task": "public-verdict-cross-version"}
journal = ContinuationJournal(parent, "public-verdict-cross-version", manifest)
journal.emit("public_test_verdict_fact_normalized", producer="kernel_stateframe_runtime",
             dispatch=False)
result = {"path": str(journal.path), "task_id": journal.task_id,
          "manifest": journal.manifest, "head": journal._head}
journal.close()
print(json.dumps(result))
'''
    resume = r'''import sys
from pathlib import Path
source = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source / "src" / "stateframe" / "engine"))
sys.path.insert(0, str(source / "src"))
from kernel_continuation_journal import ContinuationJournal
journal = ContinuationJournal.resume(
    Path(sys.argv[2]), task_id=sys.argv[3], manifest=__import__("json").loads(sys.argv[4]),
    expected_head=sys.argv[5])
assert any(row.get("kind") == "public_test_verdict_fact_normalized"
           for row in journal.records())
journal.close()
'''

    for index, writer_root in enumerate(roots):
        parent = tmp_path / f"writer-{index}"
        parent.mkdir()
        made = _run_python(writer_root, create, parent, timeout=120)
        assert made.returncode == 0, made.stdout + made.stderr
        artifact = json.loads(made.stdout.strip().splitlines()[-1])
        artifacts.append((artifact, writer_root))

    for artifact, writer_root in artifacts:
        reader_root = (baseline_root if writer_root == current_root else current_root)
        if reader_root == writer_root and len(roots) == 1:
            reader_root = writer_root
        read = _run_python(
            reader_root, resume, artifact["path"], artifact["task_id"],
            json.dumps(artifact["manifest"], separators=(",", ":")), artifact["head"],
            timeout=120,
        )
        assert read.returncode == 0, read.stdout + read.stderr
