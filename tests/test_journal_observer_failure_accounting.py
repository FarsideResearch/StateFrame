"""Observer failure accounting survives batching and journal replay."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import kernel_continuation_journal as journal_module
import pytest
from kernel_continuation_executive import ContinuationExecutive
from kernel_continuation_journal import ContinuationJournal
from kernel_generated_context import GeneratedContext

_EVENT = "observer_probe"
_FAILURE = "journal_observer_failed"


def _reject(_row):
    raise RuntimeError("observer rejected row")


def _failures_on_disk(journal):
    return [row for row in map(json.loads, journal.path.read_bytes().splitlines())
            if row["kind"] == _FAILURE]


def _accounted(failures):
    return sum(row.get("dropped_count", 1) for row in failures)


def _write_probes(journal, *, batch):
    events = [(_EVENT, {"index": index}) for index in range(32)]
    if batch:
        return journal.emit_batch(events, atomic=True)
    return [journal.emit(kind, **fields) for kind, fields in events]


def test_all_32_batched_observer_failures_are_accounted_for(tmp_path):
    with ContinuationJournal(tmp_path, "observer-accounting", {}) as journal:
        journal.subscribe("rejecting", _reject, kinds=(_EVENT,))
        _write_probes(journal, batch=True)

        assert _accounted(_failures_on_disk(journal)) == 32


def test_batch_and_single_append_persist_the_same_failure_count(tmp_path):
    counts = {}
    for batch in (True, False):
        with ContinuationJournal(tmp_path, "observer-accounting", {}) as journal:
            journal.subscribe("rejecting", _reject, kinds=(_EVENT,))
            _write_probes(journal, batch=batch)
            counts[batch] = _accounted(_failures_on_disk(journal))

    assert counts[True] == counts[False] == 32


@pytest.mark.parametrize("source", ["batch", "single", "replay"])
def test_overflow_reports_exact_count_and_sequence_bounds(tmp_path, source):
    with ContinuationJournal(tmp_path, "observer-overflow", {}) as journal:
        if source == "single":
            for index in range(32):
                journal.subscribe(f"rejecting-{index}", _reject, kinds=(_EVENT,))
            row = journal.emit(_EVENT)
            failed_sequences = [row["sequence"]] * 32
        else:
            def reject_even(row):
                if row["index"] % 2 == 0:
                    _reject(row)

            events = [(_EVENT, {"index": index}) for index in range(64)]
            if source == "batch":
                journal.subscribe("rejecting", reject_even, kinds=(_EVENT,))
                rows = journal.emit_batch(events, atomic=True)
            else:
                rows = [journal.emit(kind, **fields) for kind, fields in events]
                journal.subscribe("rejecting", reject_even, kinds=(_EVENT,))
            failed_sequences = [row["sequence"] for row in rows[::2]]

        failures = _failures_on_disk(journal)
        details = [row["event_sequence"] for row in failures if "dropped_count" not in row]
        overflow, = [row for row in failures if "dropped_count" in row]

        assert details == failed_sequences[:20]
        assert overflow["first_event_sequence"] == failed_sequences[20]
        assert overflow["last_event_sequence"] == failed_sequences[-1]
        assert overflow["dropped_count"] == 12

        # Counts describe failures, even for sparse ranges or many failures on one row.
        assert _accounted(failures) == len(failed_sequences)


def test_failed_delivery_of_failure_report_is_accounted_without_recursing(tmp_path):
    with ContinuationJournal(tmp_path, "observer-recursion", {}) as journal:
        called = Counter()

        def reject(row):
            called[row["kind"]] += 1
            _reject(row)

        journal.subscribe("rejecting", reject, kinds=(_EVENT, _FAILURE))
        probe = journal.emit(_EVENT)
        detail, overflow = _failures_on_disk(journal)

        assert detail["event_sequence"] == probe["sequence"]
        assert overflow["dropped_count"] == 1
        assert overflow["first_event_sequence"] == overflow["last_event_sequence"] == detail["sequence"]
        assert called == {_EVENT: 1, _FAILURE: 1}
        assert journal.subscriber_cursor("rejecting") == overflow["sequence"]

        # A later subscription sees the ordinary records and can advance past the summary.
        replayed = []
        before_replay = journal.path.read_bytes()
        journal.subscribe("replay", replayed.append, kinds=(_EVENT, _FAILURE))
        assert [row["sequence"] for row in replayed] == [probe["sequence"], detail["sequence"]]
        assert journal.subscriber_cursor("replay") == overflow["sequence"]
        assert journal.path.read_bytes() == before_replay


def test_overflow_keeps_the_live_artifact_index_in_step_with_single_appends(tmp_path):
    with ContinuationJournal(tmp_path, "observer-artifact-sequence", {}) as journal:
        context = GeneratedContext(journal)
        journal.subscribe("rejecting", _reject, kinds=(_EVENT,))

        first = journal.emit_batch([(_EVENT, {})] * 20, atomic=True)
        second = journal.emit_batch([(_EVENT, {})] * 21, atomic=True)
        artifact = journal.emit(
            "rejected_candidate_artifact",
            reference="stateframe://artifacts/rejected/probe", artifact_id="probe", text="candidate",
        )
        last = journal.emit(_EVENT)

        assert context.index.sequence == journal.cached_records()[-1]["sequence"]
        assert context.index.completeness == "COMPLETE"
        assert any(item["text"] == "candidate" for item in context.index.records.values())
        failures = _failures_on_disk(journal)
        assert [row["event_sequence"] for row in failures if "dropped_count" not in row] == [
            row["sequence"] for row in [*first, *second[:-1], last]
        ]
        overflow, = [row for row in failures if "dropped_count" in row]
        assert overflow["dropped_count"] == 1
        assert overflow["first_event_sequence"] == overflow["last_event_sequence"] == second[-1]["sequence"]
        assert overflow["sequence"] + 1 == artifact["sequence"]


def test_replay_observer_failure_storage_stays_bounded_past_the_detail_limit(tmp_path):
    with ContinuationJournal(tmp_path, "observer-storage-bound", {}) as journal:
        rows = journal.emit_batch([(_EVENT, {"index": index}) for index in range(1025)], atomic=True)
        samples = {}

        def reject(row):
            index = row["index"]
            if index in (20, 1024):
                pending = journal._pending_observer_failure_report
                samples[index] = len(pending.details), pending.overflow
            if index != 1024:
                raise RuntimeError(f"failure {index:04}: " + "é" * 170)

        journal.subscribe("rejecting", reject, kinds=(_EVENT,))
        assert samples == {20: (20, None), 1024: (20, (rows[20]["sequence"], rows[1023]["sequence"], 1004))}
        *details, summary = _failures_on_disk(journal)
        assert [row["event_sequence"] for row in details] == [row["sequence"] for row in rows[:20]]
        assert summary["dropped_count"] == 1004
        assert summary["first_event_sequence"] == rows[20]["sequence"]
        assert summary["last_event_sequence"] == rows[1023]["sequence"]
        assert journal._pending_observer_failure_report is None


@pytest.mark.parametrize("attach_after_overflow", [False, True])
def test_overflow_preserves_continuation_transitions(tmp_path, attach_after_overflow):
    task = "observer-continuation-sequence"
    with ContinuationJournal(tmp_path, task, {}) as journal:
        if not attach_after_overflow:
            executive = ContinuationExecutive(journal)
        journal.subscribe("rejecting", _reject, kinds=(_EVENT,))
        journal.emit_batch([(_EVENT, {})] * 21, atomic=True)
        path, head = journal.path, journal.head
        if not attach_after_overflow:
            _create_intention(executive)

    if attach_after_overflow:
        with ContinuationJournal.resume(path, task_id=task, manifest={}, expected_head=head) as resumed:
            _create_intention(ContinuationExecutive(resumed))


def _create_intention(executive):
    executive.apply("create", new_id="root", objective="inspect", pending_decision="inspect",
                    dependencies={"workspace": "sha256:" + "a" * 64}, turns=2,
                    acceptance={"kind": "no_public_checks", "selectors": []})
    assert "root" in executive.state.intentions


_STOP_DURING_FAILURE_REPORT = """
import os
import signal
import sys
from pathlib import Path

from kernel_continuation_journal import ContinuationJournal

def reject(_row):
    raise RuntimeError("observer rejected row")

journal = ContinuationJournal(Path(sys.argv[1]), "observer-crash-recovery", {})
print(journal.path, flush=True)
subscriber = sys.argv[4] if len(sys.argv) > 4 else "rejecting"
journal.subscribe(subscriber, reject, kinds=("observer_probe", "journal_observer_failed"))
emit = journal.emit
stop_after = int(sys.argv[2])
stop_point = sys.argv[3]
written = 0

if stop_point == "during_delivery":
    def interrupt_delivery(_row):
        if written == stop_after - 1:
            os.kill(os.getpid(), signal.SIGKILL)

    journal.subscribe("interrupting", interrupt_delivery, kinds=("journal_observer_failed",))

def interrupt(kind, **fields):
    global written
    if kind == "journal_observer_failed" and "dropped_count" in fields and stop_point == "after_summary_fsync":
        fsync = os.fsync

        def stop_after_summary_fsync(fd):
            fsync(fd)
            if fd == journal._fd:
                os.kill(os.getpid(), signal.SIGKILL)

        os.fsync = stop_after_summary_fsync
    if kind == "journal_observer_failed" and written == stop_after:
        if stop_point == "during_summary_write":
            assert "dropped_count" in fields
            write = os.write

            def tear_summary(fd, data):
                if fd == journal._fd:
                    write(fd, data[:len(data) // 2])
                    os.kill(os.getpid(), signal.SIGKILL)
                return write(fd, data)

            os.write = tear_summary
        else:
            os.kill(os.getpid(), signal.SIGKILL)
    row = emit(kind, **fields)
    if kind == "journal_observer_failed":
        written += 1
        if written == stop_after == 21:
            os.kill(os.getpid(), signal.SIGKILL)
    return row

journal.emit = interrupt
journal.emit_batch([("observer_probe", {})] * 32, atomic=True)
"""


@pytest.mark.parametrize(("stop_after", "stop_point"), [
    (1, "between_rows"), (20, "between_rows"), (20, "during_delivery"),
    (20, "during_summary_write"), (21, "after_summary_fsync"), (21, "between_rows"),
])
def test_resume_finishes_an_interrupted_failure_report_exactly_once(tmp_path, stop_after, stop_point):
    crashed = subprocess.run([sys.executable, "-c", _STOP_DURING_FAILURE_REPORT,
                              str(tmp_path), str(stop_after), stop_point],
                             capture_output=True, text=True, timeout=45, check=False)
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    path = Path(crashed.stdout.strip())
    lines = path.read_bytes().splitlines(keepends=True)
    if stop_point == "during_summary_write":
        assert not lines.pop().endswith(b"\n")
    before = [json.loads(line) for line in lines]
    probes = [row["sequence"] for row in before if row["kind"] == _EVENT]
    written_failures = [row for row in before if row["kind"] == _FAILURE]
    assert len(probes) == 32
    assert len(written_failures) == stop_after
    observed_details = written_failures[:20]

    with ContinuationJournal.resume(path, task_id="observer-crash-recovery", manifest={},
                                    expected_head=before[-1]["record_hash"]) as journal:
        head = journal.head
        reports = _failures_on_disk(journal)
        assert [row["event_sequence"] for row in reports if "dropped_count" not in row] == probes[:20]
        overflow, = [row for row in reports if "dropped_count" in row]
        assert _accounted(reports) == len(probes) + len(observed_details)
        assert overflow["first_event_sequence"] == probes[20]
        assert overflow["last_event_sequence"] == observed_details[-1]["sequence"]
        assert journal.cached_records()[-1]["previous_hash"] == overflow["record_hash"]
        assert journal.cached_records()[-1]["kind"] == "continuation_writer_resumed"
        assert path.read_bytes().startswith(b"".join(lines))

    with ContinuationJournal.resume(path, task_id="observer-crash-recovery", manifest={},
                                    expected_head=head) as journal:
        assert _failures_on_disk(journal) == reports


_STOP_AT_REPORT_IO_BOUNDARY = """
import os
import signal
import sys
from pathlib import Path

import kernel_continuation_journal as journal_module
import kernel_turn_publication as publication

point = sys.argv[2]

def stop():
    os.kill(os.getpid(), signal.SIGKILL)

def arm(journal_path):
    info = journal_path.stat()
    intent = journal_path.parent / ".journal-publication"
    report = journal_path.parent / ".journal-observer-report"
    replace, fsync, write, truncate = os.replace, os.fsync, os.write, os.ftruncate
    unlink, sync_dir, sync_intent_dir = Path.unlink, journal_module._sync_directory, publication.sync_directory

    def journal_fd(fd):
        opened = os.fstat(fd)
        return (opened.st_dev, opened.st_ino) == (info.st_dev, info.st_ino)

    def replace_file(source, target, **kwargs):
        if target == intent and point == "before_row_intent_replace":
            stop()
        replace(source, target, **kwargs)
        if target == intent and point == "after_row_intent_replace":
            stop()

    def write_row(fd, data):
        if journal_fd(fd) and point == "during_detail_write":
            write(fd, data[:len(data) // 2])
            stop()
        written = write(fd, data)
        if journal_fd(fd) and point == "after_detail_write":
            stop()
        return written

    def truncate_row(fd, size):
        truncate(fd, size)
        if journal_fd(fd) and point == "after_recovery_truncate":
            stop()

    def sync_file(fd):
        fsync(fd)
        if journal_fd(fd) and point in ("after_detail_fsync", "after_recovery_fsync"):
            stop()

    def unlink_file(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if path == intent and point in ("after_row_intent_unlink", "after_recovery_intent_unlink"):
            stop()
        if path == report and point == "after_report_unlink":
            stop()

    def sync_directory(path):
        sync_dir(path)
        if path == journal_path.parent:
            if not report.exists() and point == "after_report_cleanup_sync":
                stop()
            if not intent.exists() and point in ("after_row_cleanup_sync", "after_recovery_cleanup_sync"):
                stop()

    def sync_intent_directory(path):
        sync_intent_dir(path)
        if path == journal_path.parent and intent.exists() and point == "after_row_intent_sync":
            stop()

    os.replace, os.fsync, os.write, os.ftruncate = replace_file, sync_file, write_row, truncate_row
    Path.unlink = unlink_file
    journal_module._sync_directory = sync_directory
    publication.sync_directory = sync_intent_directory

if point.startswith("after_recovery_") or point.startswith("after_report_"):
    path = Path(sys.argv[1])
    arm(path)
    with journal_module.ContinuationJournal.resume(path, task_id="observer-crash-recovery", manifest={},
                                                   expected_head=sys.argv[3]):
        pass
else:
    journal = journal_module.ContinuationJournal(Path(sys.argv[1]), "observer-io-boundary", {})
    print(journal.path, flush=True)

    def reject(_row):
        raise RuntimeError("observer rejected row")

    journal.subscribe("rejecting", reject, kinds=("observer_probe",))
    emit = journal.emit

    def emit_with_stop(kind, **fields):
        if kind == "journal_observer_failed":
            arm(journal.path)
        return emit(kind, **fields)

    journal.emit = emit_with_stop
    journal.emit_batch([("observer_probe", {})] * 22, atomic=True)
"""


def _complete_prefix(path):
    data = path.read_bytes()
    return data[:data.rfind(b"\n") + 1]


@pytest.mark.parametrize(("point", "complete_details", "torn"), [
    ("before_row_intent_replace", 0, False),
    ("after_row_intent_replace", 0, False),
    ("after_row_intent_sync", 0, False),
    ("during_detail_write", 0, True),
    ("after_detail_write", 1, False),
    ("after_detail_fsync", 1, False),
    ("after_row_intent_unlink", 1, False),
    ("after_row_cleanup_sync", 1, False),
])
def test_report_recovery_preserves_complete_rows_at_write_boundaries(tmp_path, point, complete_details, torn):
    stopped = subprocess.run([sys.executable, "-c", _STOP_AT_REPORT_IO_BOUNDARY, str(tmp_path), point],
                             capture_output=True, text=True, timeout=45, check=False)
    assert stopped.returncode == -signal.SIGKILL, stopped.stderr
    path = Path(stopped.stdout.strip())
    assert path.read_bytes().endswith(b"\n") is not torn
    prefix = _complete_prefix(path)
    rows = list(map(json.loads, prefix.splitlines()))
    assert sum(row["kind"] == _FAILURE for row in rows) == complete_details
    probes = [row["sequence"] for row in rows if row["kind"] == _EVENT]
    assert len(probes) == 22

    with ContinuationJournal.resume(path, task_id="observer-io-boundary", manifest={},
                                    expected_head=rows[-1]["record_hash"]) as journal:
        assert path.read_bytes().startswith(prefix)
        *details, summary = _failures_on_disk(journal)
        assert [row["event_sequence"] for row in details] == probes[:20]
        assert summary["dropped_count"] == 2
        assert summary["first_event_sequence"] == probes[20]
        assert summary["last_event_sequence"] == probes[21]
        assert journal.cached_records()[-1]["previous_hash"] == summary["record_hash"]


@pytest.mark.parametrize("point", [
    "after_recovery_truncate", "after_recovery_fsync", "after_recovery_intent_unlink",
    "after_recovery_cleanup_sync", "after_report_unlink", "after_report_cleanup_sync",
])
def test_repeated_interruption_during_report_recovery_keeps_original_rows(tmp_path, point):
    stopped = subprocess.run([sys.executable, "-c", _STOP_DURING_FAILURE_REPORT,
                              str(tmp_path), "20", "during_summary_write"],
                             capture_output=True, text=True, timeout=45, check=False)
    assert stopped.returncode == -signal.SIGKILL, stopped.stderr
    path = Path(stopped.stdout.strip())
    prefix = _complete_prefix(path)
    rows = list(map(json.loads, prefix.splitlines()))

    retry = subprocess.run([sys.executable, "-c", _STOP_AT_REPORT_IO_BOUNDARY, str(path), point,
                            rows[-1]["record_hash"]], capture_output=True, text=True, timeout=45, check=False)
    assert retry.returncode == -signal.SIGKILL, retry.stderr
    recovered_prefix = _complete_prefix(path)
    recovered_head = json.loads(recovered_prefix.splitlines()[-1])["record_hash"]
    assert recovered_prefix.startswith(prefix)

    with ContinuationJournal.resume(path, task_id="observer-crash-recovery", manifest={},
                                    expected_head=recovered_head) as journal:
        assert path.read_bytes().startswith(recovered_prefix)
        *details, summary = _failures_on_disk(journal)
        assert details == [row for row in rows if row["kind"] == _FAILURE]
        assert len(details) == 20
        assert summary["dropped_count"] == 32
        assert summary["previous_hash"] == details[-1]["record_hash"]
        assert journal.cached_records()[-1]["previous_hash"] == summary["record_hash"]


@pytest.mark.parametrize("stop_point", ["between_rows", "during_summary_write"])
def test_replay_resume_archives_the_completed_observer_failure_report(tmp_path, stop_point):
    from stateframe.resume import archive_attempt

    measurements = tmp_path / "worker" / "measurement-runs"
    measurements.mkdir(parents=True)
    crashed = subprocess.run([sys.executable, "-c", _STOP_DURING_FAILURE_REPORT,
                              str(measurements), "20", stop_point],
                             capture_output=True, text=True, timeout=45, check=False)
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    path = Path(crashed.stdout.strip())
    lines = path.read_bytes().splitlines(keepends=True)
    if stop_point == "during_summary_write":
        assert not lines.pop().endswith(b"\n")
    before = [json.loads(line) for line in lines]
    probes = [row for row in before if row["kind"] == _EVENT]
    details = [row for row in before if row["kind"] == _FAILURE]
    assert len(probes) == 32
    assert len(details) == 20

    spec = {"schema": "stateframe-run/1", "environment": {}, "max_turns": 2}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    state = {"run": tmp_path, "spec": spec, "attempt": 1, "max_turns": 2}
    archived = archive_attempt(state, tmp_path / "engine", use_checkpoint=False)

    archived_journal = archived / path.relative_to(tmp_path)
    rows = [json.loads(line) for line in archived_journal.read_bytes().splitlines()]
    assert rows[:-1] == before
    overflow = rows[-1]
    assert overflow["kind"] == _FAILURE
    assert overflow["previous_hash"] == before[-1]["record_hash"]
    assert overflow["dropped_count"] == 32
    assert overflow["first_event_sequence"] == probes[20]["sequence"]
    assert overflow["last_event_sequence"] == details[-1]["sequence"]


_STOP_AFTER_REPORT_RECOVERY_BEFORE_ARCHIVE = """
import json
import os
import signal
import sys
from pathlib import Path

from stateframe.engine import kernel_continuation_journal as journal_module
from stateframe.resume import archive_attempt

run = Path(sys.argv[1])
spec = json.loads((run / "spec.json").read_bytes())
recover = journal_module.recover_publication_mirror

def stop_after_recover(path):
    recover(path)
    os.kill(os.getpid(), signal.SIGKILL)

journal_module.recover_publication_mirror = stop_after_recover
archive_attempt({"run": run, "spec": spec, "attempt": 1, "max_turns": 2}, run / "engine")
"""


def test_retry_archive_after_report_recovery_preserves_the_completed_report(tmp_path):
    from stateframe.resume import archive_attempt

    measurements = tmp_path / "worker" / "measurement-runs"
    measurements.mkdir(parents=True)
    stopped = subprocess.run([sys.executable, "-c", _STOP_DURING_FAILURE_REPORT,
                              str(measurements), "20", "during_summary_write"],
                             capture_output=True, text=True, timeout=45, check=False)
    assert stopped.returncode == -signal.SIGKILL, stopped.stderr
    path = Path(stopped.stdout.strip())
    prefix = _complete_prefix(path)
    spec = {"schema": "stateframe-run/1", "environment": {}, "max_turns": 2}
    (tmp_path / "spec.json").write_text(json.dumps(spec))

    archive_stop = subprocess.run([sys.executable, "-c", _STOP_AFTER_REPORT_RECOVERY_BEFORE_ARCHIVE,
                                   str(tmp_path)], capture_output=True, text=True, timeout=45, check=False)
    assert archive_stop.returncode == -signal.SIGKILL, archive_stop.stderr
    assert not (tmp_path / "attempts").exists()
    repaired = path.read_bytes()
    assert repaired.startswith(prefix)
    summary = json.loads(repaired[len(prefix):])
    assert summary["kind"] == _FAILURE
    assert summary["dropped_count"] == 32

    state = {"run": tmp_path, "spec": spec, "attempt": 1, "max_turns": 2}
    archived = archive_attempt(state, tmp_path / "engine")
    assert (archived / path.relative_to(tmp_path)).read_bytes() == repaired


def test_archive_refuses_an_unreadable_pending_observer_detail(tmp_path):
    from stateframe.resume import ResumeRefused, archive_attempt

    measurements = tmp_path / "worker" / "measurement-runs"
    measurements.mkdir(parents=True)
    crashed = subprocess.run([sys.executable, "-c", _STOP_DURING_FAILURE_REPORT,
                              str(measurements), "0", "between_rows", "π" * 3000],
                             capture_output=True, text=True, timeout=45, check=False)
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    path = Path(crashed.stdout.strip())
    prefix = path.read_bytes()
    assert [row["kind"] for row in map(json.loads, prefix.splitlines())].count(_EVENT) == 32
    blob, = (path.parent / "blobs").glob("*.zlib")
    blob.write_bytes(b"unreadable compressed data")

    spec = {"schema": "stateframe-run/1", "environment": {}, "max_turns": 2}
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    state = {"run": tmp_path, "spec": spec, "attempt": 1, "max_turns": 2}
    with pytest.raises(ResumeRefused, match="journal publication could not be recovered"):
        archive_attempt(state, tmp_path / "engine", use_checkpoint=False)

    assert path.read_bytes() == prefix
    assert not (tmp_path / "attempts").exists()


def test_interrupted_replay_reports_its_failures_before_a_later_single_append(tmp_path):
    single_kind = "single_observer_probe"
    with ContinuationJournal(tmp_path, "observer-replay-interrupted", {}) as journal:
        journal.subscribe("live-rejecting", _reject, kinds=(single_kind,))
        rows = journal.emit_batch([(_EVENT, {"index": index}) for index in range(23)], atomic=True)

        def interrupt_replay(row):
            if row["index"] == 22:
                raise KeyboardInterrupt("stop replay")
            _reject(row)

        with pytest.raises(KeyboardInterrupt, match="stop replay"):
            journal.subscribe("interrupted", interrupt_replay, kinds=(_EVENT,))

        single = journal.emit(single_kind)
        failures = _failures_on_disk(journal)
        overflow, = [row for row in failures if "dropped_count" in row]
        assert overflow["first_event_sequence"] == rows[20]["sequence"]
        assert overflow["last_event_sequence"] == rows[21]["sequence"]
        assert overflow["dropped_count"] == 2
        assert overflow["sequence"] < single["sequence"]
        assert failures[-1]["event_sequence"] == single["sequence"]
        assert failures[-1]["subscriber"] == "live-rejecting"
        assert _accounted(failures) == 23


def test_interrupted_audit_cursor_preserves_prior_replay_failures_on_resume(tmp_path):
    task = "observer-audit-cursor-interrupted"
    with ContinuationJournal(tmp_path, task, {}) as journal:
        journal.subscribe("seed", _reject, kinds=(_EVENT,))
        probes = journal.emit_batch([(_EVENT, {})] * 22, atomic=True)
        previous_reports = _failures_on_disk(journal)
        delivered = []

        def reject(row):
            delivered.append(row["sequence"])
            _reject(row)

        def interrupt_cursor(_sequence):
            raise RuntimeError("audit cursor could not advance")

        with pytest.raises(RuntimeError, match="audit cursor could not advance"):
            journal.subscribe("replay", reject, kinds=(_EVENT,), advance_audit=interrupt_cursor)
        assert delivered == [row["sequence"] for row in probes]
        path, head = journal.path, journal.head

    with ContinuationJournal.resume(path, task_id=task, manifest={}, expected_head=head) as journal:
        reports = _failures_on_disk(journal)
        assert reports[:len(previous_reports)] == previous_reports
        details = reports[len(previous_reports):-1]
        assert [row["event_sequence"] for row in details] == delivered[:20]
        assert all(row["subscriber"] == "replay" for row in details)
        overflow = reports[-1]
        assert overflow["first_event_sequence"] == delivered[20]
        assert overflow["last_event_sequence"] == delivered[21]
        assert overflow["dropped_count"] == 2
        assert journal.cached_records()[-1]["previous_hash"] == overflow["record_hash"]


_STOP_DURING_OBSERVER_FANOUT = """
import os
import signal
import sys
from pathlib import Path

from kernel_continuation_journal import ContinuationJournal

def reject(_row):
    raise RuntimeError("observer rejected row")

def stop(_row):
    os.kill(os.getpid(), signal.SIGKILL)

def reject_then_stop(row):
    if row["index"] == 22:
        stop(row)
    reject(row)

journal = ContinuationJournal(Path(sys.argv[1]), "interrupted-fanout", {})
print(journal.path, flush=True)
source = sys.argv[2]
events = [("observer_probe", {"index": index}) for index in range(24)]
if source == "batch":
    journal.subscribe("rejecting", reject_then_stop, kinds=("observer_probe",))
    journal.emit_batch(events, atomic=True)
elif source == "replay":
    journal.emit_batch(events, atomic=True)
    journal.subscribe("rejecting", reject_then_stop, kinds=("observer_probe",))
else:
    for index in range(22):
        journal.subscribe(f"rejecting-{index}", reject, kinds=("observer_probe",))
    journal.subscribe("interrupting", stop, kinds=("observer_probe",))
    journal.emit("observer_probe")
"""


@pytest.mark.parametrize("source", ["batch", "replay", "single"])
def test_resume_keeps_failures_recorded_before_fanout_is_killed(tmp_path, source):
    crashed = subprocess.run(
        [sys.executable, "-c", _STOP_DURING_OBSERVER_FANOUT, str(tmp_path), source],
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert crashed.returncode == -signal.SIGKILL, crashed.stderr
    path = Path(crashed.stdout.strip())
    prefix = path.read_bytes()
    rows = [json.loads(line) for line in prefix.splitlines()]
    probes = [row["sequence"] for row in rows if row["kind"] == _EVENT]
    assert len(probes) == (1 if source == "single" else 24)
    failed = probes * 22 if source == "single" else probes[:22]
    assert not any(row["kind"] == _FAILURE for row in rows)

    with ContinuationJournal.resume(path, task_id="interrupted-fanout", manifest={},
                                    expected_head=rows[-1]["record_hash"]) as journal:
        assert path.read_bytes().startswith(prefix)
        *details, summary = _failures_on_disk(journal)
        assert [row["event_sequence"] for row in details] == failed[:20]
        assert summary["dropped_count"] == 2
        assert summary["first_event_sequence"] == failed[20]
        assert summary["last_event_sequence"] == failed[21]
        assert journal.cached_records()[-1]["previous_hash"] == summary["record_hash"]
        appended = journal.emit("after_recovery")
        assert appended["sequence"] == summary["sequence"] + 2
        assert _failures_on_disk(journal) == [*details, summary]


_STOP_WHILE_SAVING_OBSERVER_FAILURE = """
import os
import signal
import sys
from pathlib import Path

import kernel_turn_publication as publication
from kernel_continuation_journal import ContinuationJournal

journal = ContinuationJournal(Path(sys.argv[1]), "observer-report-save", {})
print(journal.path, flush=True)
point, target = sys.argv[2], int(sys.argv[3])
intent = journal.path.parent / ".journal-observer-report"
replace, sync, fsync = os.replace, publication.sync_directory, os.fsync
saves = 0
syncing_report = False

def stop():
    os.kill(os.getpid(), signal.SIGKILL)

def replace_file(source, destination, **kwargs):
    global saves, syncing_report
    if destination == intent:
        saves += 1
        syncing_report = True
        if saves == target and point == "before_replace":
            stop()
    replace(source, destination, **kwargs)
    if destination == intent and saves == target and point == "after_replace":
        stop()

def sync_directory(path):
    global syncing_report
    sync(path)
    if syncing_report and saves == target and point == "after_directory_sync":
        stop()
    syncing_report = False

def sync_file(fd):
    fsync(fd)
    if point == "after_temporary_file_sync" and not syncing_report and saves + 1 == target:
        # The targeted replacement is the next metadata write during an observer callback.
        stop()

def reject(_row):
    os.replace, publication.sync_directory, os.fsync = replace_file, sync_directory, sync_file
    raise RuntimeError("observer rejected row")

journal.subscribe("rejecting", reject, kinds=("observer_probe",))
journal.emit_batch([("observer_probe", {})] * 24, atomic=True)
"""


@pytest.mark.parametrize(("point", "target", "durable"), [
    ("after_temporary_file_sync", 1, 0),
    ("after_replace", 1, 1),
    ("before_replace", 20, 19),
    ("after_directory_sync", 20, 20),
    ("before_replace", 21, 20),
    ("after_replace", 21, 21),
    ("after_temporary_file_sync", 22, 21),
    ("after_directory_sync", 22, 22),
])
def test_interrupted_report_save_recovers_exactly_the_published_failures(tmp_path, point, target, durable):
    stopped = subprocess.run([sys.executable, "-c", _STOP_WHILE_SAVING_OBSERVER_FAILURE,
                              str(tmp_path), point, str(target)],
                             capture_output=True, text=True, timeout=45, check=False)
    assert stopped.returncode == -signal.SIGKILL, stopped.stderr
    path = Path(stopped.stdout.strip())
    prefix = path.read_bytes()
    rows = list(map(json.loads, prefix.splitlines()))
    probes = [row["sequence"] for row in rows if row["kind"] == _EVENT]
    assert len(probes) == 24
    assert all(row["kind"] != _FAILURE for row in rows)

    with ContinuationJournal.resume(path, task_id="observer-report-save", manifest={},
                                    expected_head=rows[-1]["record_hash"]) as journal:
        assert path.read_bytes().startswith(prefix)
        reports = _failures_on_disk(journal)
        assert _accounted(reports) == durable
        detail_count = min(20, durable)
        assert [row["event_sequence"] for row in reports[:detail_count]] == probes[:detail_count]
        if durable > 20:
            summary, = reports[20:]
            assert summary["first_event_sequence"] == probes[20]
            assert summary["last_event_sequence"] == probes[durable - 1]
            assert summary["dropped_count"] == durable - 20
        else:
            assert len(reports) == durable


def test_no_failure_batch_and_single_append_are_byte_identical_after_replay_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(journal_module, "uuid", SimpleNamespace(
        uuid4=lambda: SimpleNamespace(hex="a" * 32)))
    monkeypatch.setattr(journal_module, "time", SimpleNamespace(time_ns=lambda: 123))
    journals = {}
    for batch in (True, False):
        parent = tmp_path / str(batch)
        parent.mkdir()
        with ContinuationJournal(parent, "observer-parity", {}) as journal:
            observed = []
            journal.subscribe("live", observed.append, kinds=(_EVENT,))
            rows = _write_probes(journal, batch=batch)
            assert observed == rows
            path, head = journal.path, journal.head

        before_resume = path.read_bytes()
        with ContinuationJournal.resume(path, task_id="observer-parity", manifest={},
                                        expected_head=head) as resumed:
            before_replay = path.read_bytes()
            assert before_replay.startswith(before_resume)
            replayed = []
            resumed.subscribe("replay", replayed.append, kinds=(_EVENT,))
            assert replayed == rows
            assert path.read_bytes() == before_replay
        journals[batch] = (before_resume, path.read_bytes())

    assert journals[True] == journals[False]
