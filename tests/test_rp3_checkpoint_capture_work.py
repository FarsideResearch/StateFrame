"""Checkpoint work and publication invariants over a large owned run."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager

import kernel_workspace_index as index
import pytest
from kernel_secret_scan import SecretScanner
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_feed import WorkspaceFeed
from kernel_workspace_refs import bind_run_store
from kernel_workspace_state import WorkspaceFence, WorkspaceState, _PayloadSpool

from stateframe.engine import kernel_checkpoint as checkpoint
from stateframe.engine.kernel_boundary_recovery import BoundaryRecoveryStop
from stateframe import resume


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    run = tmp_path / "run"
    run.mkdir(mode=0o700)
    workspace = run / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("STATEFRAME_HOME", str(home))
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    monkeypatch.setenv(checkpoint.LEGACY_DIRECT_ENGINE_ENV, "1")
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [b"checkpoint-test-key-0123456789"])
    monkeypatch.setattr(checkpoint, "_SECRETS", [b"registered-test-pattern"])
    monkeypatch.setattr(checkpoint, "_STATE", checkpoint._STATE.copy())
    monkeypatch.setattr(checkpoint, "_SECRET_GENERATION", checkpoint._SECRET_GENERATION)
    bind_run_store(run, "1" * 64)
    state = WorkspaceState(WorkspaceFence.create(workspace, tmp_path / "fence"),
                           feed=WorkspaceFeed(workspace, enabled=True))
    consumers = WorkspaceConsumers(state)
    bind_consumers(workspace, consumers)
    context = {name: {} for name in ("hostfns", "participants", "framecells", "framefuncs", "cvars")}
    monkeypatch.setattr(checkpoint, "collect", lambda _: ({"frame": {"value": 17}}, context))
    yield run, workspace, state
    state._feed.close()


def _save(run, step):
    return checkpoint.save(sys._getframe(), step, run)


def test_capture_diagnostics_accept_only_known_stage_metadata():
    error = checkpoint.ObjectCaptureError("objects", "redacted")
    assert checkpoint._capture_stage(error) == {"failure_stage": "objects"}
    assert checkpoint._capture_stage(checkpoint.ObjectCaptureError("unknown", "redacted")) == {}
    assert checkpoint._capture_stage(checkpoint.ObjectCaptureError.__new__(checkpoint.ObjectCaptureError)) == {}


def _populate(root, directories, count, data):
    # Fixed timestamps keep this test focused on observation and secret scanning.
    timestamp = 1_700_000_000_000_000_000
    for group in range(directories):
        directory = root / f"group-{group:02d}"
        directory.mkdir()
        for number in range(count):
            path = directory / f"record-{number:04d}"
            path.write_bytes(data)
            os.utime(path, ns=(timestamp, timestamp))


@pytest.mark.skipif(sys.platform != "linux", reason="sparse observation requires a Linux change feed")
def test_dense_workspace_and_growing_run_have_bounded_checkpoint_work(boundary, monkeypatch):
    run, workspace, state = boundary
    data = b"ordinary checkpoint payload\n" * 16
    _populate(workspace, 50, 100, data)
    committed = run / "committed"
    committed.mkdir()
    _populate(committed, 16, 64, data)
    tail = run / "tail"
    tail.mkdir()
    (tail / "0000").write_bytes(b"initial")
    counts = Counter()
    refresh = index.WorkspaceIndex.refresh
    feed = SecretScanner.feed
    scandir = index.os.scandir

    def count_refresh(self, changed=None, **kwargs):
        result = refresh(self, changed, **kwargs)
        counts["sweeps"] += int(self.full)
        counts["reads"] += self.work.content_reads
        return result

    def count_feed(self, block):
        caller = sys._getframe(1).f_locals.get("self")
        if isinstance(caller, _PayloadSpool):
            counts["scan_bytes"] += len(block)
        return feed(self, block)

    def count_scandir(path):
        counts["scandirs"] += 1
        return scandir(path)

    monkeypatch.setattr(index.WorkspaceIndex, "refresh", count_refresh)
    monkeypatch.setattr(SecretScanner, "feed", count_feed)
    monkeypatch.setattr(index.os, "scandir", count_scandir)
    # The time budget covers checkpoint work without shared-device flush
    # contention. Publication and crash tests below use the actual barriers.
    monkeypatch.setattr(os, "fsync", lambda _: None)

    def sample(step):
        counts.clear()
        started = time.perf_counter()
        saved = _save(run, step)
        seconds = time.perf_counter() - started
        print(f"checkpoint {step}: {seconds:.6f}s {dict(sorted(counts.items()))}")
        assert checkpoint.latest(run) == saved
        return counts.copy(), seconds

    sample(1)
    unchanged, unchanged_seconds = sample(2)
    with state.begin_write(("group-00/record-0000",)):
        (workspace / "group-00/record-0000").write_bytes(b"changed")
    for number in range(1, 5):
        (tail / f"{number:04d}").write_bytes(b"new record")
    changed, changed_seconds = sample(3)

    assert unchanged["sweeps"] == unchanged["reads"] == unchanged["scan_bytes"] == 0
    # The legacy publication selector still enumerates retained checkpoints.
    assert unchanged["scandirs"] <= 8
    assert changed["sweeps"] == 0
    assert changed["reads"] <= 5
    assert changed["scan_bytes"] <= len(b"changed") + 4 * len(b"new record")
    assert changed["scandirs"] <= 8
    assert max(unchanged_seconds, changed_seconds) < 10


def test_known_refusal_stays_visible_without_disabling_saving(boundary, monkeypatch):
    run, _, _ = boundary
    monkeypatch.setattr(checkpoint, "_SIGNING_KEY", [])
    monkeypatch.setenv(checkpoint.MODE_ENV, "on")
    monkeypatch.setenv(checkpoint.EVERY_ENV, "1")
    sends = {"count": 0}
    monkeypatch.setattr(checkpoint, "_sends", lambda _: sends["count"])

    def must_not_collect(_):
        pytest.fail("an unsigned boundary started semantic capture")

    monkeypatch.setattr(checkpoint, "collect", must_not_collect)
    # A new send must remain eligible after repeated refusals; the old ceiling blocked recovery.
    for step in range(1, checkpoint.FAILURE_WARNING_THRESHOLD + 2):
        sends["count"] = step
        checkpoint.save_point(sys._getframe(), step, run / "worker")

    assert checkpoint.latest(run) is None
    assert checkpoint.failure_summary() == {
        "checkpoint_failures": checkpoint.FAILURE_WARNING_THRESHOLD + 1,
        "checkpoint_saving_disabled": False,
    }
    events = [json.loads(line) for line in (run / "checkpoints/events.jsonl").read_text().splitlines()]
    failures = [event for event in events if event["event"] == "checkpoint_failed"]
    assert len(failures) == checkpoint.FAILURE_WARNING_THRESHOLD + 1
    assert not any(event["event"] == "checkpoint_disabled" for event in events)
    assert {event["failure_stage"] for event in failures} == {"admission"}
    assert {event["failure_reason"] for event in failures} == {"checkpoint refused"}

    class UntrustedReason(str):
        pass

    error = checkpoint.ObjectCaptureError("unlisted", UntrustedReason("checkpoint refused"))
    assert checkpoint._failure_reason(error) == "unexpected writer error"
    assert checkpoint._capture_stage(error) == {}


def test_checkpoint_failure_recovery_waits_for_new_send_and_keeps_resume_candidate(
        boundary, monkeypatch):
    run, _, _ = boundary
    monkeypatch.setenv(checkpoint.MODE_ENV, "on")
    monkeypatch.setenv(checkpoint.EVERY_ENV, "1")
    monkeypatch.setenv(checkpoint.KEEP_ENV, "2")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setattr(checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(checkpoint, "_low_disk_space", lambda _run: None)
    sends = {"count": 1}
    monkeypatch.setattr(checkpoint, "_sends", lambda _run: sends["count"])
    checkpoint._STATE.update(
        failed=0, consecutive_failed=0, disabled=None, last_sends=None,
        restored_step=None, saved=0, warned=set(),
    )

    attempts = []
    real_save = checkpoint.save

    def observe_save(frame, step, run_dir, *, background=False):
        attempts.append((sends["count"], step))
        return real_save(frame, step, run_dir, background=background)

    monkeypatch.setattr(checkpoint, "save", observe_save)
    real_collect = checkpoint.collect

    def fail_three_captures(frame):
        if sends["count"] in {2, 3, 4}:
            raise checkpoint.ObjectCaptureError("admission", "injected capture failure")
        return real_collect(frame)

    monkeypatch.setattr(checkpoint, "collect", fail_three_captures)
    worker = run / "worker"
    checkpoint.save_point(None, 1, worker)
    first = checkpoint.latest(run)
    assert first is not None

    sends["count"] = 2
    checkpoint.save_point(None, 2, worker)
    assert len(attempts) == 2
    checkpoint.save_point(None, 3, worker)
    assert len(attempts) == 2  # the failed send count is not attempted twice

    for send, step in ((3, 4), (4, 5)):
        sends["count"] = send
        checkpoint.save_point(None, step, worker)
    events = [json.loads(line) for line in
              (run / "checkpoints" / "events.jsonl").read_text().splitlines()]
    failures = [event for event in events if event.get("event") == "checkpoint_failed"]
    assert len(failures) == 3
    assert checkpoint.latest(run) == first
    selected = resume.select_checkpoint(run, [{"send": 2}], run / "wire")
    assert selected is not None and selected["dir"] == str(first)

    sends["count"] = 5  # the capture fault has cleared; a new send makes saving reachable
    checkpoint.save_point(None, 6, worker)
    recovered = checkpoint.latest(run)
    assert recovered is not None and recovered != first
    selected = resume.select_checkpoint(run, [{"send": 2}], run / "wire")
    assert selected is not None and selected["dir"] == str(first)
    assert attempts == [(1, 1), (2, 2), (3, 4), (4, 5), (5, 6)]


@pytest.mark.parametrize("stop_type", [KeyboardInterrupt, BoundaryRecoveryStop])
def test_fourth_checkpoint_attempt_propagates_stop_after_transaction_cleanup(
        boundary, monkeypatch, stop_type):
    run, _, _ = boundary
    monkeypatch.setenv(checkpoint.MODE_ENV, "on")
    monkeypatch.setenv(checkpoint.EVERY_ENV, "1")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setattr(checkpoint, "_async_enabled", lambda: False)
    monkeypatch.setattr(checkpoint, "_low_disk_space", lambda _run: None)
    sends = {"count": 0}
    monkeypatch.setattr(checkpoint, "_sends", lambda _run: sends["count"])
    checkpoint._STATE.update(
        failed=0, consecutive_failed=0, disabled=None, last_sends=None,
        restored_step=None, saved=0, warned=set(),
    )
    real_collect = checkpoint.collect

    def fail_first_three_captures(frame):
        if sends["count"] <= 3:
            raise checkpoint.ObjectCaptureError("admission", "injected capture failure")
        return real_collect(frame)

    def stop_in_transaction(*_args, **_kwargs):
        raise stop_type("engine stop")

    monkeypatch.setattr(checkpoint, "collect", fail_first_three_captures)
    monkeypatch.setattr(checkpoint, "_save_into", stop_in_transaction)
    worker = run / "worker"
    for sends["count"] in range(1, 4):
        checkpoint.save_point(None, sends["count"], worker)

    sends["count"] = 4
    with pytest.raises(stop_type, match="engine stop"):
        checkpoint.save_point(None, 4, worker)

    root = run / "checkpoints"
    assert not list(root.glob(".0004-*.tmp"))
    rows = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    assert len([row for row in rows if row.get("event") == "checkpoint_failed"]) == 3


def test_object_capture_preserves_boundary_stop_after_stack_cleanup(boundary, monkeypatch):
    run, _, state = boundary
    monkeypatch.setenv(checkpoint.MODE_ENV, "on")
    monkeypatch.setenv(checkpoint.EVERY_ENV, "1")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    monkeypatch.setattr(checkpoint, "_sends", lambda _run: 1)
    checkpoint._STATE.update(
        failed=0, consecutive_failed=0, disabled=None, last_sends=None,
        restored_step=None, warned=set(),
    )
    exited = []
    pin = state.pin

    @contextmanager
    def track_pin(*args, **kwargs):
        try:
            with pin(*args, **kwargs) as value:
                yield value
        finally:
            exited.append(True)

    def stop(_frame):
        raise BoundaryRecoveryStop("engine stop")

    monkeypatch.setattr(state, "pin", track_pin)
    monkeypatch.setattr(checkpoint, "collect", stop)
    with pytest.raises(BoundaryRecoveryStop, match="engine stop"):
        checkpoint.save_point(sys._getframe(), 1, run / "worker")

    assert exited == [True]


def test_workspace_writer_refuses_capture_before_run_tail(boundary, monkeypatch):
    run, _, state = boundary
    codec = checkpoint._workspace_checkpoint_codec()

    def must_not_open(*_):
        pytest.fail("a blocked workspace started run-side capture")

    monkeypatch.setattr(codec, "_run_side", must_not_open)
    with state.begin_write(), pytest.raises(checkpoint.ObjectCaptureError) as failure:
        _save(run, 1)

    assert failure.value.stage == "workspace"
    assert checkpoint.latest(run) is None


@pytest.mark.skipif(sys.platform != "linux", reason="race detector requires a Linux change feed")
def test_run_writer_race_between_pin_and_publication(boundary, monkeypatch):
    run, _, _ = boundary
    tail = run / "journal.jsonl"
    tail.write_bytes(b"first record\n")
    first = _save(run, 1)
    codec = checkpoint._workspace_checkpoint_codec()
    save_state = codec._save_state

    def write_after_pin(*args):
        with tail.open("ab") as stream:
            stream.write(b"raced record\n")
        return save_state(*args)

    with monkeypatch.context() as patch:
        patch.setattr(codec, "_save_state", write_after_pin)
        with pytest.raises(checkpoint.ObjectCaptureError) as failure:
            _save(run, 2)

    assert failure.value.stage == "run_tail"
    assert checkpoint.latest(run) == first
    assert checkpoint.validate(first)["step"] == 1
    assert checkpoint.validate(_save(run, 3))["step"] == 3


@pytest.mark.skipif(sys.platform != "linux", reason="warm rejection requires a Linux change feed")
def test_secret_rotation_revokes_run_attestation_and_repeat_refuses_cheaply(boundary, monkeypatch):
    run, _, _ = boundary
    (run / "ledger").write_bytes(b"newly-registered-marker")
    first = _save(run, 1)
    checkpoint.register_secret("newly-registered-marker")

    codec = checkpoint._workspace_checkpoint_codec()

    def must_not_write(*_):
        pytest.fail("a prohibited blob reached the state writer")

    monkeypatch.setattr(codec, "_save_state", must_not_write)
    with pytest.raises(checkpoint.ObjectCaptureError) as failure:
        _save(run, 2)
    assert failure.value.stage == "run_tail"

    def must_not_read(*_):
        pytest.fail("a known refusal reread an unchanged blob")

    monkeypatch.setattr(index.WorkspaceIndex, "_file_digest", must_not_read)
    with pytest.raises(checkpoint.ObjectCaptureError):
        _save(run, 3)
    assert checkpoint.latest(run) == first


def test_external_change_and_link_policy_are_reflected_in_restore(boundary, monkeypatch):
    run, workspace, _ = boundary
    leaf = workspace / "value"
    leaf.write_bytes(b"1111")
    tail = run / "ledger"
    tail.write_bytes(b"aaaa")
    tail.chmod(0o640)
    original = tail.stat()
    redirect = run.parent / "redirect"
    redirect.symlink_to(run.parent / "elsewhere")
    (run / "link").symlink_to(redirect)
    _save(run, 1)

    tail.write_bytes(b"bbbb")
    os.utime(tail, ns=(original.st_atime_ns, original.st_mtime_ns))
    redirect.unlink()
    redirect.symlink_to(run.parent / "home")
    second = _save(run, 2)

    tail.write_bytes(b"cccc")
    leaf.write_bytes(b"2222")
    restored = {}
    monkeypatch.setattr(checkpoint, "_apply_restored_state", lambda _, value: restored.update(value))
    monkeypatch.setattr(checkpoint, "install_continuous_clock", lambda _: None)
    checkpoint.restore(sys._getframe(), second, run)

    assert tail.read_bytes() == b"bbbb"
    assert tail.stat().st_mtime_ns == original.st_mtime_ns
    assert tail.stat().st_mode & 0o777 == 0o640
    assert leaf.read_bytes() == b"1111"
    assert not (run / "link").is_symlink()
    assert restored["frame"] == {"value": 17, "step": 1}


def test_corrupt_retained_state_blob_refuses_to_advance_latest(boundary, monkeypatch):
    from hashlib import sha256

    from kernel_workspace_refs import open_run_store

    run, _, _ = boundary
    value = b"retained safe state" * 512
    collect = checkpoint.collect

    def with_blob(frame):
        payload, ctx = collect(frame)
        payload["frame"]["blob"] = value
        return payload, ctx

    monkeypatch.setattr(checkpoint, "collect", with_blob)
    first = _save(run, 1)
    store = open_run_store(run)
    stored = store._path / "objects" / sha256(value).hexdigest()
    before = stored.stat()
    stored.chmod(0o600)
    stored.write_bytes(b"!" * len(value))
    stored.chmod(0o400)
    os.utime(stored, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(checkpoint.ObjectCaptureError) as failure:
        _save(run, 2)

    assert failure.value.stage == "semantic_state"
    assert checkpoint.latest(run) == first


def test_batched_text_leaf_is_stamped_and_reused(boundary, monkeypatch):
    run, _, _ = boundary
    codec = checkpoint._workspace_checkpoint_codec()
    text = ("long text history with an isolated surrogate \ud800\n" * 256)
    encoded = text.encode("utf-8", "surrogatepass")
    collect = checkpoint.collect
    scan = checkpoint._check_registered_secrets
    text_scans = 0

    def with_text(frame):
        payload, ctx = collect(frame)
        payload["frame"]["history"] = text
        return payload, ctx

    def count_scan(data, *args, **kwargs):
        nonlocal text_scans
        text_scans += data == encoded
        return scan(data, *args, **kwargs)

    monkeypatch.setattr(checkpoint, "collect", with_text)
    monkeypatch.setattr(checkpoint, "_check_registered_secrets", count_scan)
    _save(run, 1)
    assert text_scans == 1
    memo = next(iter(codec._state_memo.values()))
    assert any(leaf[0] is text and leaf[2] is not None for leaf in memo.values())
    second = _save(run, 2)
    assert text_scans == 1
    assert checkpoint.validate(second)["step"] == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="crash injection requires fork")
def test_crash_after_signed_record_keeps_previous_latest(boundary, monkeypatch):
    run, workspace, _ = boundary
    (workspace / "value").write_bytes(b"before crash")
    first = _save(run, 1)
    pid = os.fork()
    if pid == 0:
        try:
            state = WorkspaceState(WorkspaceFence.create(workspace, run.parent / "crash-fence"))
            consumers = WorkspaceConsumers(state)
            bind_consumers(workspace, consumers)
            # save_checkpoint has already committed the signed object record when
            # it calls this function to expose the boundary.
            checkpoint._publish_checkpoint = lambda *_: os._exit(41)
            _save(run, 2)
        finally:
            os._exit(42)

    assert os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]) == 41
    assert checkpoint.latest(run) == first
    assert checkpoint.validate(first)["step"] == 1
    assert checkpoint.validate(_save(run, 3))["step"] == 3
