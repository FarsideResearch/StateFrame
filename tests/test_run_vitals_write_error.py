from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from detreason.world_knowledge import WorldKnowledgeStore
from kernel_bolt import KernelBolt
from kernel_continuation_journal import (
    ContinuationJournal, JournalError, recover_publication_mirror,
)
from kernel_run_vitals import RunVitalsAccumulator
from kernel_stateframe_runtime import StateFrameRuntime
from probes import react_repair


def _runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> tuple[StateFrameRuntime, Path]:
    assert shipped_runtime_profile["STATEFRAME_F4"] == "on"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))
    monkeypatch.setenv("KERNEL_CONTINUATION_EXECUTIVE", "1")
    monkeypatch.setenv("STATEFRAME_RUN_VITALS", "on")

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "app.py"
    target.write_text("VALUE = 7\n", encoding="utf-8")
    task_id = "run-vitals-eio"
    owner = KernelBolt(repo, "", task_id)
    owner.world = WorldKnowledgeStore(sandbox_id=task_id)
    owner.task_row = {"msg": "Inspect app.py."}
    owner.current_step = 0
    tools = react_repair._tools_for_probe_modes(
        probe_gate_on=False, observation_probe_on=False,
    )
    run_dir = tmp_path / "run"
    runtime = StateFrameRuntime(
        owner,
        output=run_dir,
        target=target,
        model="scripted",
        tools=tools,
        temperature=0,
        source_resolver=lambda _path, _symbol: None,
        multi_file=True,
    )
    assert isinstance(runtime.journal, ContinuationJournal)
    # A direct finish has no wire event to initialize the optional accumulator.
    runtime.journal._run_vitals_accumulator = RunVitalsAccumulator()
    runtime.journal._run_vitals_emitted = False
    runtime.journal._run_vitals_observer_failed = False
    return runtime, run_dir


def _finish(
    runtime: StateFrameRuntime, run_dir: Path,
) -> tuple[list[dict[str, object]], Exception | None]:
    journal = runtime.journal
    assert Path(journal.path).parent.parent == run_dir
    error: Exception | None = None
    try:
        runtime.finish({"result": "FAIL", "steps": 0})
    except Exception as caught:
        error = caught
    finally:
        if journal._fd is not None:
            try:
                journal.close()
            except Exception as caught:
                if error is None:
                    error = caught
    events = [
        json.loads(line)
        for line in Path(journal.path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return events, error


def test_control_finishes_with_terminal_vitals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> None:
    runtime, run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)

    events, error = _finish(runtime, run_dir)
    kinds = [str(event["kind"]) for event in events]

    assert error is None, repr(error)
    assert "run_vitals" in kinds
    assert kinds.index("run_vitals") < kinds.index("attempt_terminated")


def test_vitals_eio_does_not_abort_finish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> None:
    runtime, run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)
    original_write = os.write
    vitals_writes = 0

    def write_with_terminal_vitals_eio(
        fd: int, data: bytes | bytearray | memoryview,
    ) -> int:
        nonlocal vitals_writes
        if fd == runtime.journal._fd and b'"kind":"run_vitals"' in bytes(data):
            vitals_writes += 1
            if vitals_writes == 1:
                raise OSError(errno.EIO, "injected terminal vitals write failure")
        return original_write(fd, data)

    monkeypatch.setattr(os, "write", write_with_terminal_vitals_eio)
    events, error = _finish(runtime, run_dir)
    kinds = [str(event["kind"]) for event in events]

    assert vitals_writes == 1
    assert "run_vitals" not in kinds
    assert "attempt_terminated" in kinds, f"finish error: {error!r}"
    assert error is None, repr(error)


@pytest.mark.parametrize("error_number", [errno.EIO, errno.ENOSPC], ids=["eio", "enospc"])
def test_zero_byte_vitals_write_error_allows_required_append_and_chain_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
    error_number: int,
) -> None:
    runtime, _run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)
    journal = runtime.journal
    original_write = os.write
    sequence_before = journal._sequence
    head_before = journal._head
    size_before = os.fstat(journal._fd).st_size

    def fail_without_writing(
        fd: int, data: bytes | bytearray | memoryview,
    ) -> int:
        if fd == journal._fd and b'"kind":"run_vitals"' in bytes(data):
            raise OSError(error_number, "injected zero-byte vitals write failure")
        return original_write(fd, data)

    monkeypatch.setattr(os, "write", fail_without_writing)
    try:
        with pytest.raises(OSError) as caught:
            journal.emit("run_vitals", producer="test")
        assert caught.value.errno == error_number
        assert os.fstat(journal._fd).st_size == size_before
        assert (journal._sequence, journal._head) == (sequence_before, head_before)
        assert not journal._poisoned

        monkeypatch.setattr(os, "write", original_write)
        required = journal.emit("attempt_terminated", outcome="test")
        rows = list(journal.records())

        assert required["sequence"] == sequence_before + 1
        assert rows[-1]["kind"] == "attempt_terminated"
        assert rows[-1]["record_hash"] == journal.head
        assert not any(row["kind"] == "run_vitals" for row in rows)
    finally:
        journal.close()


def test_partial_vitals_write_keeps_journal_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> None:
    runtime, _run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)
    journal = runtime.journal
    original_write = os.write
    size_before = os.fstat(journal._fd).st_size

    def write_part_then_fail(
        fd: int, data: bytes | bytearray | memoryview,
    ) -> int:
        if fd == journal._fd and b'"kind":"run_vitals"' in bytes(data):
            original_write(fd, memoryview(data)[:4])
            raise OSError(errno.EIO, "injected partial vitals write failure")
        return original_write(fd, data)

    monkeypatch.setattr(os, "write", write_part_then_fail)
    try:
        with pytest.raises(OSError, match="partial vitals"):
            journal.emit("run_vitals", producer="test")
        assert os.fstat(journal._fd).st_size > size_before
        assert journal._poisoned
        with pytest.raises(JournalError, match="poisoned"):
            journal.emit("attempt_terminated", outcome="test")
    finally:
        journal.close()


def test_vitals_fsync_failure_keeps_journal_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> None:
    runtime, _run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)
    journal = runtime.journal
    original_fsync = os.fsync

    def fail_journal_fsync(fd: int) -> None:
        if fd == journal._fd:
            raise OSError(errno.EIO, "injected journal fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_journal_fsync)
    try:
        with pytest.raises(OSError, match="journal fsync"):
            journal.emit("attempt_terminated", outcome="test")
        assert journal._poisoned
        with pytest.raises(JournalError, match="poisoned"):
            journal.emit("attempt_terminated", outcome="again")
    finally:
        monkeypatch.setattr(os, "fsync", original_fsync)
        journal.close()


def test_journal_close_failure_keeps_writer_poisoned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> None:
    runtime, _run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)
    journal = runtime.journal
    original_close = os.close
    journal_fd = journal._fd
    close_calls = 0

    def close_then_fail(fd: int) -> None:
        nonlocal close_calls
        if fd == journal_fd:
            close_calls += 1
            original_close(fd)
            raise OSError(errno.EIO, "injected journal close failure")
        original_close(fd)

    monkeypatch.setattr(os, "close", close_then_fail)
    with pytest.raises(OSError, match="journal close failure"):
        journal.close()

    assert close_calls == 1
    assert journal._fd is None
    assert journal._poisoned


def test_child_forked_while_the_journal_closes_can_still_close_its_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> None:
    """A child that keeps the handle keeps the parent's journal lock."""
    runtime, _run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)
    journal = runtime.journal
    original_close = os.close
    journal_fd = journal._fd
    seen: list[int | None] = []

    def observe(fd: int) -> None:
        if fd == journal_fd:
            seen.append(journal._fd)
        original_close(fd)

    monkeypatch.setattr(os, "close", observe)
    journal.close()

    assert seen == [journal_fd]
    assert journal._fd is None


def test_external_process_change_after_zero_byte_error_poisoned_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shipped_runtime_profile: dict[str, str],
) -> None:
    runtime, _run_dir = _runtime(tmp_path, monkeypatch, shipped_runtime_profile)
    journal = runtime.journal
    original_write = os.write

    def fail_without_writing(
        fd: int, data: bytes | bytearray | memoryview,
    ) -> int:
        if fd == journal._fd and b'"kind":"run_vitals"' in bytes(data):
            raise OSError(errno.EIO, "injected zero-byte vitals write failure")
        return original_write(fd, data)

    monkeypatch.setattr(os, "write", fail_without_writing)
    try:
        with pytest.raises(OSError):
            journal.emit("run_vitals", producer="test")
        assert not journal._poisoned

        mutation = (
            "import os,sys; fd=os.open(sys.argv[1], os.O_WRONLY|os.O_APPEND); "
            "os.write(fd,b'x'); os.close(fd)"
        )
        subprocess.run(
            [sys.executable, "-c", mutation, str(journal.path)],
            check=True, timeout=10, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        with pytest.raises(JournalError, match="changed outside its writer"):
            journal.emit("attempt_terminated", outcome="test")
        assert journal._poisoned
    finally:
        monkeypatch.setattr(os, "write", original_write)
        journal.close()


def test_zero_byte_report_write_with_open_publication_intent_recovers_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = ContinuationJournal(tmp_path, "intent-task", {})
    path = Path(journal.path)
    original_write = os.write
    intent = path.parent / ".journal-publication"
    report = path.parent / ".journal-observer-report"

    def fail_for_trigger(row: dict[str, object]) -> None:
        if row.get("kind") == "trigger_probe":
            raise RuntimeError("injected observer failure")

    def fail_report_row(fd: int, data: bytes | bytearray | memoryview) -> int:
        if fd == journal._fd and b'"kind":"journal_observer_failed"' in bytes(data):
            raise OSError(errno.EIO, "injected zero-byte report write")
        return original_write(fd, data)

    journal.subscribe("failing-observer", fail_for_trigger)
    monkeypatch.setattr(os, "write", fail_report_row)
    try:
        with pytest.raises(OSError, match="zero-byte report"):
            journal.emit("trigger_probe")
        assert intent.exists()
        # The report must be finished by recovery, so this writer takes no more rows.
        with pytest.raises(JournalError):
            journal.emit("attempt_terminated", outcome="test")
    finally:
        monkeypatch.setattr(os, "write", original_write)
        journal.close()

    for _ in range(2):  # a second reopen must add nothing
        recover_publication_mirror(path)
        resumed = ContinuationJournal.resume(
            path, task_id="intent-task", manifest={}, allow_unanchored_recovery=True)
        try:
            rows = list(resumed.records())
        finally:
            resumed.close()
        kinds = [row["kind"] for row in rows]
        assert kinds.count("trigger_probe") == 1
        assert kinds.count("journal_observer_failed") == 1
        assert [row["sequence"] for row in rows] == list(range(1, len(rows) + 1))
        assert not intent.exists() and not report.exists()
