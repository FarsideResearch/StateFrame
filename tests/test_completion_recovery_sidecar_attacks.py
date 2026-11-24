"""Attacks against completion recovery persistence and its public loopback path."""
from __future__ import annotations

import errno
import inspect
import json
import os
import signal
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

import kernel_completion_authority as authority_module
from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from fixtures.process_signals import signal_run_owned
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "run_tests", "parameters": {"type": "object"}}},
]
INTERNAL_RECOVERY_FIELDS = {
    "recovery_progress", "recovery_progress_delta", "recovery_progress_pending",
    "recovery_progress_seen", "action_result_progress",
    "action_result_progress_by_reason", "blocker_state", "history_dropped",
}


@pytest.fixture(autouse=True)
def _private_stateframe_home(monkeypatch):
    monkeypatch.setenv("STATEFRAME_HOME", isolated_stateframe_home())


class _MemoryJournal:
    def __init__(self, path: Path | None, events: list[dict[str, object]] | None = None):
        self.path = path
        self.task_id = "sidecar-attack-task"
        self.manifest = {"environment_revision": "env-sidecar-attack"}
        self.events = events if events is not None else []

    def emit(self, kind: str, **fields: object) -> dict[str, object]:
        event = {"kind": kind, **fields}
        self.events.append(event)
        return event

    def records(self):
        return iter(self.events)


class _CrashJournal(_MemoryJournal):
    def __init__(self, path: Path, ready, hold, *, pause_after_delta: bool):
        super().__init__(path)
        self.ready = ready
        self.hold = hold
        self.pause_after_delta = pause_after_delta

    def emit(self, kind: str, **fields: object) -> dict[str, object]:
        event = super().emit(kind, **fields)
        assert self.path is not None
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        state = fields.get("state")
        delta = state.get("recovery_progress_delta") if isinstance(state, dict) else None
        if self.pause_after_delta and kind == "completion_authority_state" and delta:
            self.ready.set()
            self.hold.wait()
        return event


class _FileEvent:
    """Small cross-process event for the owned crash-helper subprocesses."""

    def __init__(self, path: Path):
        self.path = path

    def set(self) -> None:
        self.path.touch()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.path.exists():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(scaled(0.01))
        return True


def _runtime_for_journal(journal):
    from kernel_stateframe_runtime import StateFrameRuntime

    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.journal = journal
    runtime.binding = lambda: {"product_revision": "sha256:sidecar-attack-candidate"}
    runtime.required_public_selectors = lambda: ()
    runtime.completion_contract_version = "completion-authority/1"
    runtime.verdict_identity_provider = None
    return runtime, journal


def _runtime_shell(tmp_path: Path, *, events=None, with_path: bool = True):
    journal_path = (tmp_path / "worker" / "measurement-runs" / "run-a" / "events.jsonl"
                    if with_path else None)
    return _runtime_for_journal(_MemoryJournal(journal_path, events))


def _start_runtime(tmp_path: Path, *, events=None, with_path: bool = True):
    runtime, journal = _runtime_shell(tmp_path, events=events, with_path=with_path)
    runtime._init_completion_authority(
        step_limit=None, tools=TOOLS, resume=object() if events is not None else None,
    )
    return runtime, journal


def _legacy_sidecar_path(journal: _MemoryJournal) -> Path:
    from kernel_wire_identity import completion_root_budget_wire_id

    assert journal.path is not None
    return journal.path.parent / (
        f"completion-progress-{completion_root_budget_wire_id(journal)}.sqlite3"
    )


def _saved_state_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [event for event in events if event.get("kind") == "completion_authority_state"]


def _journal_progress(events: list[dict[str, object]]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for event in _saved_state_events(events):
        state = event.get("state")
        for reason, digests in authority_module.recovery_progress_from_snapshot(state).items():
            found.setdefault(reason, set()).update(digests)
    return found


def _nested_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nested_keys(child)


def _assert_request_has_no_internal_recovery_fields(request: dict[str, object]) -> None:
    leaked = INTERNAL_RECOVERY_FIELDS & set(_nested_keys(request))
    assert not leaked, f"internal recovery progress leaked into model request: {sorted(leaked)}"


def _authority(*, progress_store_path: Path | None = None):
    arguments = {
        "candidate": {"revision": "state-0"},
        "available_actions": ("edit", "run_tests"),
        "budget": authority_module.RootFuelBudget(unbounded=True),
    }
    if (progress_store_path is not None
            and "progress_store_path" in inspect.signature(
                authority_module.CompletionAuthority).parameters):
        arguments["progress_store_path"] = progress_store_path
    authority = authority_module.CompletionAuthority(**arguments)
    authority.refuse("PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a")
    return authority


def _lock_sqlite(path: str, ready, release) -> None:
    connection = sqlite3.connect(path, timeout=2)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS recovery_progress ("
        "reason TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(reason, digest))"
    )
    connection.commit()
    connection.execute("BEGIN EXCLUSIVE")
    ready.set()
    release.wait(scaled(15))
    connection.rollback()
    connection.close()


def _crash_before_journal_append(path: str, ready, hold) -> None:
    journal = _CrashJournal(Path(path), ready, hold, pause_after_delta=False)
    runtime, _journal = _runtime_for_journal(journal)
    runtime._init_completion_authority(step_limit=None, tools=TOOLS)
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    runtime._persist_completion_authority_state()
    runtime.completion_authority.reduce()
    ready.set()
    hold.wait()
    runtime._persist_completion_authority_state()


def _crash_after_journal_append(path: str, ready, hold) -> None:
    journal = _CrashJournal(Path(path), ready, hold, pause_after_delta=True)
    runtime, _journal = _runtime_for_journal(journal)
    runtime._init_completion_authority(step_limit=None, tools=TOOLS)
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    runtime._persist_completion_authority_state()
    runtime.completion_authority.reduce()
    # emit() fsyncs the delta row, then blocks before returning to the authority ack.
    runtime._persist_completion_authority_state()
    ready.set()
    hold.wait()


def _spawn_crash_helper(tmp_path: Path, target: str, journal_path: Path):
    ready = _FileEvent(tmp_path / f"{target}-ready")
    hold = _FileEvent(tmp_path / f"{target}-hold")
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), target, str(journal_path),
         str(ready.path), str(hold.path)],
        start_new_session=True,
    )
    return process, ready


def _signal_owned_helper(tmp_path: Path, process: subprocess.Popen, what: str) -> None:
    signal_run_owned(
        tmp_path, process, signal.SIGKILL, deadline=time.monotonic() + scaled(5),
        what=what, require_checkpoint_saved=False, group=True,
    )


def test_missing_sidecar_is_not_created_and_recovery_starts(tmp_path):
    runtime, journal = _start_runtime(tmp_path)
    sidecar = _legacy_sidecar_path(journal)

    assert not sidecar.exists()
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    decision = runtime.completion_authority.reduce()

    assert decision.action is not None and decision.action["tool"] == "edit"
    assert not sidecar.exists()


def test_truncated_sidecar_artifact_does_not_affect_journal_restore(tmp_path):
    runtime, journal = _runtime_shell(tmp_path)
    sidecar = _legacy_sidecar_path(journal)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"SQLite format 3\x00truncated")
    events: list[dict[str, object]] = []
    runtime, journal = _start_runtime(tmp_path, events=events)

    assert journal.path is not None
    assert sidecar.read_bytes() == b"SQLite format 3\x00truncated"
    assert runtime.completion_authority.state == authority_module.ACTIVE


def test_garbage_sidecar_artifact_is_ignored(tmp_path):
    runtime, journal = _runtime_shell(tmp_path)
    sidecar = _legacy_sidecar_path(journal)
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"not a database\x00\xff")
    runtime, _journal = _start_runtime(tmp_path)
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )

    decision = runtime.completion_authority.reduce()

    assert decision.action is not None
    assert sidecar.read_bytes() == b"not a database\x00\xff"


def test_read_only_sidecar_directory_cannot_stop_runtime(tmp_path, monkeypatch):
    runtime, journal = _runtime_shell(tmp_path)
    sidecar = _legacy_sidecar_path(journal)
    sidecar.parent.mkdir(parents=True)
    sidecar.parent.chmod(0o555)
    original_connect = sqlite3.connect

    def denied_connect(*args, **kwargs):
        raise PermissionError(errno.EACCES, "read-only sidecar directory")

    monkeypatch.setattr(sqlite3, "connect", denied_connect)
    try:
        runtime, _journal = _start_runtime(tmp_path)
    finally:
        sidecar.parent.chmod(0o755)
        monkeypatch.setattr(sqlite3, "connect", original_connect)

    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    assert runtime.completion_authority.reduce().action is not None


def test_mocked_disk_full_on_sidecar_write_does_not_stop_runtime(tmp_path, monkeypatch):
    original_connect = sqlite3.connect

    def full_disk(*args, **kwargs):
        raise OSError(errno.ENOSPC, "mocked sidecar disk full")

    monkeypatch.setattr(sqlite3, "connect", full_disk)
    runtime, _journal = _start_runtime(tmp_path)
    monkeypatch.setattr(sqlite3, "connect", original_connect)

    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    assert runtime.completion_authority.reduce().action is not None


def test_sidecar_locked_by_another_process_does_not_block_runtime(tmp_path, monkeypatch):
    shell, journal = _runtime_shell(tmp_path)
    sidecar = _legacy_sidecar_path(journal)
    sidecar.parent.mkdir(parents=True)
    process, ready = _spawn_crash_helper(tmp_path, "_lock_sqlite", sidecar)
    release = _FileEvent(tmp_path / "_lock_sqlite-hold")
    original_connect = sqlite3.connect

    def fast_locked_connect(path, *args, **kwargs):
        kwargs["timeout"] = 0.01
        return original_connect(path, *args, **kwargs)

    try:
        assert ready.wait(scaled(5)), "the sidecar lock holder did not become ready"
        monkeypatch.setattr(sqlite3, "connect", fast_locked_connect)
        runtime, _journal = _start_runtime(tmp_path)
    finally:
        release.set()
        try:
            process.wait(timeout=scaled(5))
        except subprocess.TimeoutExpired:
            _signal_owned_helper(tmp_path, process, "sidecar lock cleanup")
            process.wait(timeout=scaled(5))
        monkeypatch.setattr(sqlite3, "connect", original_connect)

    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    assert runtime.completion_authority.reduce().action is not None


def test_two_runs_in_one_stateframe_home_keep_progress_independent(tmp_path):
    first, first_journal = _start_runtime(tmp_path / "first")
    second, second_journal = _start_runtime(tmp_path / "second")
    assert _legacy_sidecar_path(first_journal).parent != _legacy_sidecar_path(second_journal).parent

    first.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="first run", obligation_id="item-a",
    )
    second.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="second run", obligation_id="item-a",
    )
    first_decision = first.completion_authority.reduce()
    second_decision = second.completion_authority.reduce()

    assert first_decision.action is not None and second_decision.action is not None
    assert not _legacy_sidecar_path(first_journal).exists()
    assert not _legacy_sidecar_path(second_journal).exists()


def test_forked_progress_does_not_poison_parent_branch(tmp_path, monkeypatch):
    store = tmp_path / "fork-progress.sqlite3"
    authority = _authority(progress_store_path=store)
    first = authority.reduce()
    assert first.action is not None
    child = authority.fork()
    if hasattr(authority_module, "_RECOVERY_PROGRESS_BATCH"):
        monkeypatch.setattr(authority_module, "_RECOVERY_PROGRESS_BATCH", 2)

    child.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    child.set_candidate({"revision": "state-1"})
    assert child.reduce().action is not None

    authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    authority.set_candidate({"revision": "state-1"})
    parent_decision = authority.reduce()

    assert not parent_decision.terminal
    assert parent_decision.action is not None and parent_decision.action["tool"] == "edit"


def test_sigkill_before_journal_append_resumes_without_false_stop(tmp_path):
    journal_path = tmp_path / "worker" / "measurement-runs" / "run-a" / "events.jsonl"
    journal_path.parent.mkdir(parents=True)
    process, ready = _spawn_crash_helper(
        tmp_path, "_crash_before_journal_append", journal_path,
    )
    try:
        assert ready.wait(scaled(10)), "worker did not reach the pre-append crash boundary"
        _signal_owned_helper(tmp_path, process, "pre-append crash boundary")
        process.wait(timeout=scaled(5))
    finally:
        if process.poll() is None:
            _signal_owned_helper(tmp_path, process, "pre-append crash cleanup")
            process.wait(timeout=scaled(5))

    assert process.returncode == -signal.SIGKILL
    events = measurement_events(tmp_path)
    resumed, journal = _start_runtime(tmp_path, events=events)
    assert not _legacy_sidecar_path(journal).exists()
    retry = resumed.completion_authority.reduce()
    assert retry.action is not None and not retry.terminal
    resumed._persist_completion_authority_state()
    resumed.completion_authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    stopped = resumed.completion_authority.reduce()
    assert stopped.terminal and stopped.reason_codes == ("NO_FEASIBLE_RECOVERY",)


def test_sigkill_after_journal_append_before_ack_replays_progress(tmp_path):
    journal_path = tmp_path / "worker" / "measurement-runs" / "run-a" / "events.jsonl"
    journal_path.parent.mkdir(parents=True)
    process, ready = _spawn_crash_helper(
        tmp_path, "_crash_after_journal_append", journal_path,
    )
    try:
        assert ready.wait(scaled(10)), "worker did not reach the post-append crash boundary"
        _signal_owned_helper(tmp_path, process, "post-append crash boundary")
        process.wait(timeout=scaled(5))
    finally:
        if process.poll() is None:
            _signal_owned_helper(tmp_path, process, "post-append crash cleanup")
            process.wait(timeout=scaled(5))

    assert process.returncode == -signal.SIGKILL
    events = measurement_events(tmp_path)
    resumed, journal = _start_runtime(tmp_path, events=events)
    assert not _legacy_sidecar_path(journal).exists()
    waiting = resumed.completion_authority.reduce()
    assert not waiting.terminal and waiting.action is None
    resumed.completion_authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    stopped = resumed.completion_authority.reduce()

    assert stopped.terminal and stopped.reason_codes == ("NO_FEASIBLE_RECOVERY",)
    assert len(resumed.completion_authority._seen_recovery_progress["PRODUCT_VIOLATION"]) == 1


def test_0827_style_resume_without_sidecar_preserves_active_state(tmp_path):
    runtime, journal = _start_runtime(tmp_path)
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="legacy blocker", obligation_id="item-a",
    )
    assert runtime.completion_authority.reduce().action is not None
    runtime._persist_completion_authority_state()
    legacy_events = list(journal.events)
    state_event = next(
        event for event in reversed(legacy_events)
        if event.get("kind") == "completion_authority_state"
    )
    legacy_state = dict(state_event["state"])
    for key in (
        "recovery_progress", "recovery_progress_seen", "recovery_progress_delta",
        "recovery_progress_pending", "progress_store_path", "action_result_progress",
        "action_result_progress_by_reason", "blocker_state", "history_dropped",
    ):
        legacy_state.pop(key, None)
    legacy_state["history"] = [
        {key: value for key, value in item.items() if key != "progress_digest"}
        for item in legacy_state.get("history", ())
    ]
    legacy_events = [{"kind": "completion_authority_state", "state": legacy_state}]
    resumed, resumed_journal = _start_runtime(tmp_path, events=legacy_events)
    resumed.completion_authority.record_action_result(
        {"action": "edit", "status": "NO_CHANGE"},
    )

    decision = resumed.completion_authority.reduce()

    assert decision.action is not None and decision.action["tool"] == "edit"
    assert not decision.terminal
    assert not _legacy_sidecar_path(resumed_journal).exists()


def test_sidecar_copied_from_another_run_cannot_false_stop_progress(tmp_path):
    donor, donor_journal = _start_runtime(tmp_path / "donor")
    donor.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    donor_digest = donor.completion_authority._recovery_progress_digest("PRODUCT_VIOLATION")
    assert donor_digest
    donor_sidecar = _legacy_sidecar_path(donor_journal)
    donor_sidecar.parent.mkdir(parents=True)
    with sqlite3.connect(donor_sidecar) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS recovery_progress (reason TEXT NOT NULL, digest TEXT NOT NULL, "
            "PRIMARY KEY(reason, digest))"
        )
        connection.execute(
            "INSERT INTO recovery_progress(reason, digest) VALUES (?, ?)",
            ("PRODUCT_VIOLATION", donor_digest),
        )
    copied_sidecar = _legacy_sidecar_path(donor_journal)
    copied_sidecar = tmp_path / "recipient" / copied_sidecar.name
    copied_sidecar.parent.mkdir(parents=True)
    shutil.copyfile(donor_sidecar, copied_sidecar)
    recipient, recipient_journal = _runtime_shell(tmp_path / "recipient")
    target_sidecar = _legacy_sidecar_path(recipient_journal)
    target_sidecar.parent.mkdir(parents=True)
    shutil.copyfile(copied_sidecar, target_sidecar)
    recipient, recipient_journal = _start_runtime(tmp_path / "recipient")
    recipient.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )

    decision = recipient.completion_authority.reduce()

    assert decision.action is not None and not decision.terminal
    assert copied_sidecar.is_file()
    assert target_sidecar.is_file()


def test_renamed_run_replays_journal_progress_without_path_binding(tmp_path):
    original_root = tmp_path / "original"
    runtime, journal = _start_runtime(original_root)
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    runtime._persist_completion_authority_state()
    runtime.completion_authority.reduce()
    runtime._persist_completion_authority_state()
    moved_root = tmp_path / "renamed-run"
    assert journal.path is not None
    journal.path.parent.mkdir(parents=True)
    journal.path.write_text("\n".join(json.dumps(event) for event in journal.events) + "\n")
    shutil.move(original_root, moved_root)
    events = list(journal.events)
    resumed, moved_journal = _runtime_shell(moved_root, events=events)
    moved_journal.path = moved_root / "measurement-runs" / "run-a" / "events.jsonl"
    resumed._init_completion_authority(step_limit=None, tools=TOOLS, resume=object())
    resumed.completion_authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    resumed.completion_authority.set_candidate({"revision": "state-1"}, revision="state-1")
    next_decision = resumed.completion_authority.reduce()

    assert next_decision.action is not None and not next_decision.terminal
    assert not _legacy_sidecar_path(moved_journal).exists()


def test_journal_rows_preserve_all_digests_after_audit_history_cap(tmp_path):
    proofs = [f"sha256:{index:064x}" for index in range(4100)]
    events = [
        {"kind": "completion_authority_state", "state": {
            "recovery_progress_delta": {"PRODUCT_VIOLATION": [proof]},
        }}
        for proof in proofs
    ]
    seen = _journal_progress(events)
    authority = _authority()
    authority._seen_recovery_progress = seen
    authority._history = [{"kind": "audit"}] * 4096

    assert len(authority._seen_recovery_progress["PRODUCT_VIOLATION"]) == 4100
    assert len(authority.history) == 4096
    assert authority._has_recovery_progress("PRODUCT_VIOLATION", proofs[0])


def test_sidecar_release_replay_recovers_proofs_flushed_out_of_pending_batch(tmp_path):
    proofs = [f"sha256:{index:064x}" for index in range(4100)]
    first_state = {
        "recovery_progress_seen": {},
        "recovery_progress_pending": {},
        "recovery_progress": {"PRODUCT_VIOLATION": proofs[3]},
        "history": [
            {"kind": "action_dispatched", "reason_code": "PRODUCT_VIOLATION",
             "progress_digest": proof}
            for proof in proofs[:4]
        ],
    }
    later_state = {
        "recovery_progress_seen": {},
        "recovery_progress_pending": {"PRODUCT_VIOLATION": proofs[-128:]},
        "recovery_progress": {"PRODUCT_VIOLATION": proofs[-1]},
        "history": [
            {"kind": "action_dispatched", "reason_code": "PRODUCT_VIOLATION",
             "progress_digest": proof}
            for proof in proofs[4:]
        ],
    }

    replayed = _journal_progress([
        {"kind": "completion_authority_state", "state": first_state},
        {"kind": "completion_authority_state", "state": later_state},
    ])

    assert replayed["PRODUCT_VIOLATION"] == set(proofs)


def test_runtime_resume_merges_set_aggregates_from_earlier_journal_rows(tmp_path):
    earlier_proofs = [f"sha256:{index:064x}" for index in range(4100)]
    latest_proof = "sha256:" + "2" * 64
    latest_state = _authority().snapshot(include_recovery_progress_seen=False)
    latest_state["recovery_progress_seen"] = {}
    latest_state["recovery_progress_delta"] = {"PRODUCT_VIOLATION": [latest_proof]}
    latest_state["history"] = []
    events = [
        {"kind": "completion_authority_state", "state": {
            "recovery_progress_delta": {"PRODUCT_VIOLATION": earlier_proofs},
        }},
        {"kind": "completion_authority_state", "state": latest_state},
    ]

    resumed, _journal = _start_runtime(tmp_path, events=events)

    restored = resumed.completion_authority._seen_recovery_progress["PRODUCT_VIOLATION"]
    assert len(restored) == 4101
    assert earlier_proofs[0] in restored
    assert earlier_proofs[-1] in restored
    assert resumed.completion_authority._has_recovery_progress(
        "PRODUCT_VIOLATION", latest_proof,
    )


def test_per_reason_progress_remains_consistent_after_journal_replay(tmp_path):
    runtime, journal = _start_runtime(tmp_path)
    authority = runtime.completion_authority
    authority.refuse("PRODUCT_VIOLATION", detail="product blocker", obligation_id="item-a")
    authority.refuse("REQUIRED_EVIDENCE_GAP", detail="evidence blocker", obligation_id="item-b")
    assert authority.dispatch_action("edit", reason_code="PRODUCT_VIOLATION").action is not None
    runtime._persist_completion_authority_state()
    authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    assert authority.dispatch_action(
        "run_tests", reason_code="REQUIRED_EVIDENCE_GAP",
    ).action is not None
    product_proof, evidence_proof = [
        row["progress_digest"] for row in authority.history
        if row["kind"] == "action_dispatched"
    ]
    runtime._persist_completion_authority_state()
    authority.record_action_result({"action": "run_tests", "status": "PARTIAL"})
    runtime._persist_completion_authority_state()
    resumed, _journal = _start_runtime(tmp_path, events=list(journal.events))
    replayed = resumed.completion_authority._seen_recovery_progress

    assert replayed["PRODUCT_VIOLATION"] == {product_proof}
    assert replayed["REQUIRED_EVIDENCE_GAP"] == {evidence_proof}
    assert not resumed.completion_authority._has_recovery_progress(
        "PRODUCT_VIOLATION", evidence_proof,
    )
    assert not resumed.completion_authority._has_recovery_progress(
        "REQUIRED_EVIDENCE_GAP", product_proof,
    )

    product_retry = resumed.completion_authority.dispatch_action(
        "edit", reason_code="PRODUCT_VIOLATION",
    )

    assert product_retry.terminal
    assert product_retry.reason_codes == ("NO_FEASIBLE_RECOVERY",)


def test_real_progress_before_resume_allows_next_recovery(tmp_path):
    runtime, journal = _start_runtime(tmp_path)
    authority = runtime.completion_authority
    authority.refuse("PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a")
    assert authority.reduce().action is not None
    runtime._persist_completion_authority_state()
    authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    authority.set_candidate({"revision": "state-1"}, revision="state-1")
    runtime._persist_completion_authority_state()
    resumed, _journal = _start_runtime(tmp_path, events=list(journal.events))

    second = resumed.completion_authority.reduce()

    assert second.action is not None and second.action["tool"] == "edit"
    assert not second.terminal


def test_progress_proof_persists_without_changing_public_request_projection(tmp_path):
    runtime, journal = _start_runtime(tmp_path)
    authority = runtime.completion_authority
    authority.refuse("PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a")
    assert authority.reduce().action is not None
    runtime._persist_completion_authority_state()
    before = authority.render()
    decision_version = authority.state_version

    authority.record_action_result({"action": "edit", "status": "PARTIAL"})
    runtime._persist_completion_authority_state()
    after = authority.render()

    assert authority.state_version == decision_version
    expected_projection = dict(before)
    expected_projection["inflight"] = None
    assert after == expected_projection
    assert "recovery_progress" not in after
    states = _saved_state_events(journal.events)
    assert len(states) >= 2
    assert states[-1]["state"]["action_result_progress_by_reason"]["PRODUCT_VIOLATION"]

    resumed, _ = _start_runtime(tmp_path, events=list(journal.events))
    second = resumed.completion_authority.reduce()
    assert not second.terminal
    assert second.action is not None and second.action["tool"] == "edit"


def test_no_progress_after_resume_terminalizes_once_without_recovery_loop(tmp_path):
    runtime, journal = _start_runtime(tmp_path)
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    assert runtime.completion_authority.reduce().action is not None
    runtime._persist_completion_authority_state()
    resumed, _journal = _start_runtime(tmp_path, events=list(journal.events))
    resumed.completion_authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})

    first = resumed.completion_authority.reduce()
    later = [resumed.completion_authority.reduce() for _ in range(5)]

    assert first.terminal and first.reason_codes == ("NO_FEASIBLE_RECOVERY",)
    assert all(decision == first for decision in later)


def test_no_journal_file_uses_in_memory_cycle_detection_only(tmp_path):
    runtime, journal = _start_runtime(tmp_path, with_path=False)
    runtime.completion_authority.refuse(
        "PRODUCT_VIOLATION", detail="synthetic blocker", obligation_id="item-a",
    )
    first = runtime.completion_authority.reduce()
    assert first.action is not None
    runtime.completion_authority.record_action_result({"action": "edit", "status": "NO_CHANGE"})
    second = runtime.completion_authority.reduce()

    assert second.terminal and second.reason_codes == ("NO_FEASIBLE_RECOVERY",)
    assert journal.path is None
    assert not hasattr(runtime.completion_authority, "_progress_store")


@pytest.mark.host_sandbox
def test_public_loopback_completion_run_needs_no_sidecar(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("A small local loopback fixture.\n")
    run = tmp_path / "run"
    script = Script({"steps": [{"tool": "done", "arguments": {"summary": "finished"}}]})
    server = serve(script)
    assert server.server_address[0] == "127.0.0.1"
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "STATEFRAME_COMPLETION_V2": "on",
        # This test covers the live completion path; checkpoint I/O is unrelated.
        "STATEFRAME_CHECKPOINT": "off",
    }
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Review the small repository and finish.", "--endpoint", endpoint,
        "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
        "--max-turns", "3", "--quiet",
    ]
    try:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=scaled(180),
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((run / "summary.json").read_text())
    assert summary["succeeded"] is True
    assert len(script.requests) >= 1
    assert not list(run.rglob("completion-progress-*.sqlite3"))


@pytest.mark.host_sandbox
@pytest.mark.parametrize("resume_before_second_recovery", [False, True],
                         ids=("plain", "resumed"))
def test_public_loopback_second_recovery_request_hides_internal_progress(
        tmp_path, resume_before_second_recovery):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n")
    (repo / "check.sh").write_text(
        "count=0\n"
        "if [ -f .check-count ]; then count=$(cat .check-count); fi\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > .check-count\n"
        "printf 'intentional check failure %s\\n' \"$count\"\n"
        "exit 1\n"
    )
    run = tmp_path / "run"
    script = Script({"steps": [
        {"tool": "done", "arguments": {"summary": "finished"}},
        {"tool": "run_tests", "arguments": {"target": "sample.py"}},
        {"tool": "run_tests", "arguments": {"target": "sample.py"}},
    ]})
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "STATEFRAME_HOME": isolated_stateframe_home(),
        "STATEFRAME_COMPLETION_V2": "on",
        "STATEFRAME_CHECKPOINT": "off",
    }

    def run_command(max_turns: int) -> list[str]:
        return [
            sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
            "--task", "Run the declared check, repair any failure, and finish.",
            "--target", "sample.py", "--endpoint", endpoint, "--model", "scripted",
            "--run-dir", str(run), "--test-python", sys.executable,
            "--check-cmd", "bash check.sh",
            "--max-turns", str(max_turns), "--quiet",
        ]

    try:
        if resume_before_second_recovery:
            initial = subprocess.run(
                run_command(2), env=env, capture_output=True, text=True,
                timeout=scaled(180),
            )
            assert len(script.requests) == 2, (
                "initial attempt should stop after the first recovery result",
                initial.stdout, initial.stderr,
            )
            resumed = subprocess.run(
                [sys.executable, "-m", "stateframe", "resume", str(run),
                 "--endpoint", endpoint, "--model", "scripted", "--max-turns", "3",
                 "--quiet"],
                env=env, capture_output=True, text=True, timeout=scaled(180),
            )
            result = resumed
        else:
            result = subprocess.run(
                run_command(3), env=env, capture_output=True, text=True,
                timeout=scaled(180),
            )
    finally:
        server.shutdown()
        server.server_close()

    assert len(script.requests) >= 3, result.stdout + result.stderr
    script.assert_consumed()
    second_recovery_request = script.requests[2]
    assert second_recovery_request.get("messages")
    _assert_request_has_no_internal_recovery_fields(second_recovery_request)
    assert not list(run.rglob("completion-progress-*.sqlite3"))


if __name__ == "__main__":
    target, journal_path, ready_path, hold_path = sys.argv[1:]
    if target not in {
            "_lock_sqlite", "_crash_before_journal_append", "_crash_after_journal_append"}:
        raise SystemExit(f"unknown crash helper {target!r}")
    globals()[target](
        journal_path, _FileEvent(Path(ready_path)), _FileEvent(Path(hold_path)),
    )
