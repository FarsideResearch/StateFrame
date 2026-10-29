"""Opt-in retention for the host continuation ledger.

The continuation journal remains the authoritative append-only history.  This
module owns only the hot projection's reclamation boundary: closed objects are
written as compact tombstones to a cold sidecar before a hash-chained journal
event publishes the GC.  The journal event is the commit point.  A sidecar
batch without that event is deliberately ignored on replay.

The default path does not construct this owner, touch the filesystem, or add a
journal event.  Tombstones retain identities needed to reject late/replayed
effects, intentions, wake tokens, and signal events after their rich records
leave the resident projection.
"""
from __future__ import annotations

import copy
import json
import os
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from kernel_continuation_state import (TERMINAL, ContinuationError, ContinuationState,
                                       Effect, Intention)
from kernel_measurement import canonical_bytes, digest

SWITCH = "KERNEL_CONTINUATION_RETENTION_GC"
FORMAT = "continuation-retention/1"
ARCHIVE_NAME = "retention.jsonl"
MAX_RETURN_RECEIPTS = 4
OBJECT_TYPES = ("intention", "effect", "wake_token", "signal", "return_receipt")
MAX_ARCHIVE_RECORD_BYTES = 1024 * 1024


class RetentionError(RuntimeError):
    """A retention archive or its journal watermark cannot be trusted."""


def mode() -> str:
    value = os.environ.get(SWITCH, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{SWITCH} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetentionError(f"{name} must be a nonempty string")
    return value


def _head(value: object, name: str) -> str:
    value = _id(value, name)
    if len(value) != 71 or not value.startswith("sha256:") \
            or any(char not in "0123456789abcdef" for char in value[7:]):
        raise RetentionError(f"{name} must be a journal digest")
    return value


def _ids(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise RetentionError(f"{name} must be a list of identities")
    result = tuple(value)
    if any(not isinstance(item, str) or not item or len(item) > 256 for item in result):
        raise RetentionError(f"{name} contains an invalid identity")
    if result != tuple(sorted(set(result))):
        raise RetentionError(f"{name} must be sorted and unique")
    return result


@dataclass(frozen=True)
class RetentionPlan:
    """A deterministic set of records safe to evict at one verified boundary."""

    source_revision: int
    intention_ids: tuple[str, ...] = ()
    effect_ids: tuple[str, ...] = ()
    wake_tokens: tuple[str, ...] = ()
    signal_ids: tuple[str, ...] = ()
    return_receipts: tuple[str, ...] = ()
    tombstones: dict = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not any((self.intention_ids, self.effect_ids, self.wake_tokens,
                        self.signal_ids, self.return_receipts))

    def ids_digest(self) -> str:
        return digest({
            "intentions": self.intention_ids,
            "effects": self.effect_ids,
            "wake_tokens": self.wake_tokens,
            "signals": self.signal_ids,
            "return_receipts": self.return_receipts,
        })


@dataclass(frozen=True)
class RetentionBatch:
    batch_id: str
    start: int
    end: int
    tail_hash: str
    batch_digest: str
    checkpoint_sequence: int
    checkpoint_head: str
    rows: tuple[dict, ...] = ()


class _RetentionArchive:
    """Small append-only sidecar with its own integrity chain."""

    def __init__(self, journal):
        self.journal = journal
        self.path = Path(journal.path).parent / ARCHIVE_NAME
        self._rows: list[dict] = []
        self._tail: str | None = None
        self._load()

    @property
    def archive_path(self) -> Path:
        return self.path

    def _load(self) -> None:
        self._rows = []
        self._tail = None
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return
        try:
            if not stat.S_ISREG(info.st_mode):
                raise RetentionError("retention archive must be a regular file")
            lines = self.path.read_bytes().splitlines(keepends=True)
        except OSError as error:
            raise RetentionError("retention archive cannot be read") from error
        if not lines:
            raise RetentionError("retention archive is empty")
        previous = None
        for sequence, line in enumerate(lines, 1):
            if len(line) > MAX_ARCHIVE_RECORD_BYTES:
                raise RetentionError("retention archive record exceeds limit")
            if not line.endswith(b"\n"):
                raise RetentionError("retention archive has a torn record")
            try:
                row = json.loads(line)
                canonical = canonical_bytes(row) + b"\n"
            except (TypeError, ValueError, UnicodeError) as error:
                raise RetentionError("retention archive has invalid JSON") from error
            if not isinstance(row, dict) or canonical != line:
                raise RetentionError("retention archive is not canonical JSON")
            if (row.get("retention_format") != FORMAT
                    or row.get("run_id") != self.journal.run_id
                    or row.get("task_id") != self.journal.task_id
                    or type(row.get("archive_sequence")) is not int
                    or row.get("archive_sequence") != sequence
                    or row.get("previous_hash") != previous):
                raise RetentionError("retention archive lineage mismatch")
            supplied = row.get("record_hash")
            body = {key: value for key, value in row.items() if key != "record_hash"}
            if _head(supplied, "record hash") != digest(body):
                raise RetentionError("retention archive hash mismatch")
            if sequence == 1 and row.get("kind") != "retention_archive_started":
                raise RetentionError("retention archive is missing its header")
            if sequence > 1 and row.get("kind") != "tombstone":
                raise RetentionError("retention archive has an unknown record")
            if sequence > 1:
                if (type(row.get("checkpoint_sequence")) is not int
                        or row.get("checkpoint_sequence") < 1
                        or _head(row.get("checkpoint_head"), "checkpoint head")
                        != row.get("checkpoint_head")
                        or row.get("object_type") not in OBJECT_TYPES
                        or not isinstance(row.get("object_id"), str)
                        or not isinstance(row.get("tombstone"), dict)):
                    raise RetentionError("retention archive tombstone is invalid")
            self._rows.append(row)
            previous = supplied
        self._tail = previous

    def verify(self) -> None:
        """Re-read the sidecar before a commit so external edits fail closed."""
        had_rows = bool(self._rows)
        self._load()
        if had_rows and not self._rows:
            raise RetentionError("retention archive disappeared")

    def _append(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return []
        self.path.parent.mkdir(mode=0o700, exist_ok=True)
        self.verify()
        if not self._rows:
            header = {
                "retention_format": FORMAT,
                "kind": "retention_archive_started",
                "run_id": self.journal.run_id,
                "task_id": self.journal.task_id,
                "archive_sequence": 1,
                "previous_hash": None,
            }
            header["record_hash"] = digest(header)
            data = canonical_bytes(header) + b"\n"
            try:
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            except FileExistsError:
                self._load()
                return self._append(rows)
            try:
                view = memoryview(data)
                while view:
                    count = os.write(fd, view)
                    if count <= 0:
                        raise OSError("retention archive write made no progress")
                    view = view[count:]
                os.fsync(fd)
            finally:
                os.close(fd)
            self._rows.append(header)
            self._tail = header["record_hash"]
        start = len(self._rows) + 1
        encoded: list[bytes] = []
        prepared: list[dict] = []
        previous = self._tail
        for offset, raw in enumerate(rows):
            row = {
                **copy.deepcopy(raw),
                "retention_format": FORMAT,
                "kind": "tombstone",
                "run_id": self.journal.run_id,
                "task_id": self.journal.task_id,
                "archive_sequence": start + offset,
                "previous_hash": previous,
            }
            row["record_hash"] = digest(row)
            prepared.append(row)
            encoded.append(canonical_bytes(row) + b"\n")
            previous = row["record_hash"]
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            for data in encoded:
                view = memoryview(data)
                while view:
                    count = os.write(fd, view)
                    if count <= 0:
                        raise OSError("retention archive write made no progress")
                    view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        self._rows.extend(prepared)
        self._tail = previous
        return prepared


class ContinuationRetention:
    """Hot-state owner used only when ``KERNEL_CONTINUATION_RETENTION_GC=on``."""

    def __init__(self, journal):
        self.journal = journal
        self._archive = _RetentionArchive(journal)
        self._committed_batches: set[str] = set()
        self._tombstones: dict[tuple[str, str], dict] = {}
        self._last_reclaimed: dict[str, int] = {}
        self._last_watermark: dict[str, object] | None = None

    @property
    def archive_path(self) -> Path:
        return self._archive.archive_path

    def plan(self, state: ContinuationState) -> RetentionPlan:
        if not isinstance(state, ContinuationState):
            raise RetentionError("retention requires a continuation state")
        keep = {item.intention_id for item in state.intentions.values()
                if item.phase not in TERMINAL}
        for item in state.intentions.values():
            if item.phase not in TERMINAL:
                continue
            parent = state.intentions.get(item.parent_id) if item.parent_id else None
            if (parent is not None and parent.return_pending
                    and item.terminal_receipt
                    and item.terminal_receipt in parent.return_receipts[-MAX_RETURN_RECEIPTS:]):
                keep.add(item.intention_id)
        # A malformed or externally assembled projection must not let a
        # terminal parent disappear while a live child still points at it.
        changed = True
        while changed:
            changed = False
            for item in state.intentions.values():
                if item.parent_id and item.intention_id in keep and item.parent_id not in keep:
                    keep.add(item.parent_id)
                    changed = True

        candidates = {
            item.intention_id for item in state.intentions.values()
            if item.phase in TERMINAL and item.intention_id not in keep
        }
        # Any non-applied effect is still a reopen/reconciliation obligation.
        # Keep its intention too; otherwise compaction would orphan a retryable
        # effect in the resident projection.
        for effect in state.effects.values():
            if effect.intention_id in candidates and effect.status != "applied":
                candidates.discard(effect.intention_id)
                keep.add(effect.intention_id)
        changed = True
        while changed:
            changed = False
            for item in state.intentions.values():
                if item.parent_id and item.intention_id in keep and item.parent_id not in keep:
                    keep.add(item.parent_id)
                    changed = True
        candidates = {
            item.intention_id for item in state.intentions.values()
            if item.phase in TERMINAL and item.intention_id not in keep
        }

        # Applied effects are closed and have no live reducer reference once
        # their outcome is durable.  ``not_applied`` remains reopenable under
        # its original identity, while pending/unknown outcomes remain live.
        effect_ids = tuple(sorted(
            effect.effect_id for effect in state.effects.values()
            if effect.status == "applied"
        ))
        current_wakes = {
            item.wake.token for item in state.intentions.values()
            if item.wake is not None and item.intention_id in keep
        }
        wake_tokens = tuple(sorted(token for token in state.used_wake_tokens
                                  if token not in current_wakes))
        signal_ids = tuple(sorted(state.signals))
        return_receipts = tuple(sorted({
            receipt for item in state.intentions.values()
            for receipt in item.return_receipts[:-MAX_RETURN_RECEIPTS]
            if receipt
        }))

        tombstones = {
            "intentions": {
                key: self._intention_tombstone(state.intentions[key])
                for key in sorted(candidates)
            },
            "effects": {
                effect_id: self._effect_tombstone(state.effects[effect_id])
                for effect_id in effect_ids
            },
            "wake_tokens": {token: {"token": token} for token in wake_tokens},
            "signals": {event_id: {"event_hash": state.signals[event_id]}
                        for event_id in signal_ids},
            "return_receipts": {receipt: {"receipt_ref": receipt}
                                for receipt in return_receipts},
        }
        return RetentionPlan(
            source_revision=state.revision,
            intention_ids=tuple(sorted(candidates)), effect_ids=effect_ids,
            wake_tokens=wake_tokens, signal_ids=signal_ids,
            return_receipts=return_receipts, tombstones=tombstones,
        )

    @staticmethod
    def _intention_tombstone(item: Intention) -> dict:
        return {
            "parent_id": item.parent_id, "phase": item.phase,
            "revision": item.revision, "terminal_receipt": item.terminal_receipt,
            "terminal_reason": item.terminal_reason,
        }

    @staticmethod
    def _effect_tombstone(effect: Effect) -> dict:
        return {
            "intention_id": effect.intention_id, "action_hash": effect.action_hash,
            "dependencies": copy.deepcopy(effect.dependencies),
            "status": effect.status, "attempts": effect.attempts,
            "receipt_ref": effect.receipt_ref,
        }

    @staticmethod
    def compact(state: ContinuationState, plan: RetentionPlan) -> ContinuationState:
        if state.revision != plan.source_revision:
            raise RetentionError("retention plan is stale")
        if state.active_id in set(plan.intention_ids):
            raise RetentionError("active intention is not reclaimable")
        compacted = copy.deepcopy(state)
        for identity in plan.intention_ids:
            compacted.intentions.pop(identity, None)
        for effect_id in plan.effect_ids:
            compacted.effects.pop(effect_id, None)
        remove_wakes = set(plan.wake_tokens)
        compacted.used_wake_tokens = [token for token in compacted.used_wake_tokens
                                     if token not in remove_wakes]
        for event_id in plan.signal_ids:
            compacted.signals.pop(event_id, None)
        old_receipts = set(plan.return_receipts)
        for item in compacted.intentions.values():
            item.return_receipts = [receipt for receipt in item.return_receipts
                                    if receipt not in old_receipts][-MAX_RETURN_RECEIPTS:]
        return compacted

    @staticmethod
    def _rows_for(plan: RetentionPlan, *, batch_id: str,
                  checkpoint_sequence: int, checkpoint_head: str) -> list[dict]:
        rows: list[dict] = []
        for object_type, values in (
                ("intention", plan.tombstones["intentions"]),
                ("effect", plan.tombstones["effects"]),
                ("wake_token", plan.tombstones["wake_tokens"]),
                ("signal", plan.tombstones["signals"]),
                ("return_receipt", plan.tombstones["return_receipts"]),
        ):
            for object_id in sorted(values):
                rows.append({
                    "batch_id": batch_id, "checkpoint_sequence": checkpoint_sequence,
                    "checkpoint_head": checkpoint_head, "object_type": object_type,
                    "object_id": object_id, "tombstone": values[object_id],
                })
        return rows

    def archive(self, plan: RetentionPlan, *, checkpoint_sequence: int,
                checkpoint_head: str | None) -> RetentionBatch:
        if type(checkpoint_sequence) is not int or checkpoint_sequence < 1:
            raise RetentionError("checkpoint sequence is invalid")
        checkpoint_head = _head(checkpoint_head, "checkpoint head")
        if plan.empty:
            raise RetentionError("empty retention plan")
        batch_id = uuid.uuid4().hex
        try:
            raw_rows = self._rows_for(plan, batch_id=batch_id,
                                      checkpoint_sequence=checkpoint_sequence,
                                      checkpoint_head=checkpoint_head)
        except (KeyError, TypeError) as error:
            raise RetentionError("retention plan tombstones are incomplete") from error
        rows = tuple(self._archive._append(raw_rows))
        if not rows:
            raise RetentionError("retention archive did not commit rows")
        return RetentionBatch(
            batch_id=batch_id, start=rows[0]["archive_sequence"],
            end=rows[-1]["archive_sequence"], tail_hash=rows[-1]["record_hash"],
            batch_digest=digest([row["record_hash"] for row in rows]),
            checkpoint_sequence=checkpoint_sequence, checkpoint_head=checkpoint_head,
            rows=rows,
        )

    def _batch_rows(self, batch: RetentionBatch) -> list[dict]:
        if (not isinstance(batch, RetentionBatch)
                or type(batch.start) is not int or type(batch.end) is not int
                or batch.start < 2 or batch.end < batch.start
                or batch.end - batch.start + 1 != len(batch.rows)):
            raise RetentionError("retention archive batch bounds are invalid")
        self._archive.verify()
        rows = [row for row in self._archive._rows
                if row.get("batch_id") == batch.batch_id]
        if (len(rows) != len(batch.rows) or not rows
                or rows[0]["archive_sequence"] != batch.start
                or rows[-1]["archive_sequence"] != batch.end
                or rows[-1]["record_hash"] != batch.tail_hash
                or digest([row["record_hash"] for row in rows]) != batch.batch_digest):
            raise RetentionError("retention archive batch is not verified")
        seen: set[tuple[str, str]] = set()
        for offset, row in enumerate(rows):
            if (row.get("archive_sequence") != batch.start + offset
                    or row.get("batch_id") != batch.batch_id
                    or row.get("checkpoint_sequence") != batch.checkpoint_sequence
                    or row.get("checkpoint_head") != batch.checkpoint_head
                    or row.get("object_type") not in OBJECT_TYPES):
                raise RetentionError("retention archive batch lineage mismatch")
            object_id = row.get("object_id")
            if not isinstance(object_id, str) or not object_id:
                raise RetentionError("retention archive batch identity is invalid")
            identity = (row["object_type"], object_id)
            if identity in seen:
                raise RetentionError("retention archive batch has duplicate identities")
            seen.add(identity)
        if tuple(rows) != tuple(batch.rows):
            raise RetentionError("retention archive batch changed")
        return rows

    def commit(self, batch: RetentionBatch, *, checkpoint_sequence: int,
               checkpoint_head: str | None) -> None:
        checkpoint_head = _head(checkpoint_head, "checkpoint head")
        if (checkpoint_sequence != batch.checkpoint_sequence
                or checkpoint_head != batch.checkpoint_head):
            raise RetentionError("retention GC checkpoint does not match archive watermark")
        rows = self._batch_rows(batch)
        if batch.batch_id in self._committed_batches:
            return
        for row in rows:
            self._tombstones[(row["object_type"], row["object_id"])] = copy.deepcopy(
                row["tombstone"])
        self._committed_batches.add(batch.batch_id)
        self._last_reclaimed = {
            "intentions": sum(row["object_type"] == "intention" for row in rows),
            "effects": sum(row["object_type"] == "effect" for row in rows),
            "wake_tokens": sum(row["object_type"] == "wake_token" for row in rows),
            "signals": sum(row["object_type"] == "signal" for row in rows),
            "return_receipts": sum(row["object_type"] == "return_receipt" for row in rows),
        }
        self._last_watermark = {
            "checkpoint_sequence": checkpoint_sequence,
            "checkpoint_head": checkpoint_head,
            "archive_sequence": batch.end,
            "archive_head": batch.tail_hash,
        }

    def batch_from_event(self, event: Mapping[str, object]) -> RetentionBatch:
        batch_id = _id(event.get("batch_id"), "batch_id")
        raw_rows = [row for row in self._archive._rows
                    if row.get("batch_id") == batch_id]
        if not raw_rows:
            raise RetentionError("retention GC references an absent archive batch")
        start = event.get("archive_start")
        end = event.get("archive_end")
        checkpoint_sequence = event.get("checkpoint_sequence")
        if (type(start) is not int or type(end) is not int
                or type(checkpoint_sequence) is not int):
            raise RetentionError("retention GC archive bounds are invalid")
        batch = RetentionBatch(
            batch_id=batch_id,
            start=start, end=end,
            tail_hash=_head(event.get("archive_head"), "archive head"),
            batch_digest=_head(event.get("archive_digest"), "archive digest"),
            checkpoint_sequence=checkpoint_sequence,
            checkpoint_head=_head(event.get("checkpoint_head"), "checkpoint head"),
            rows=tuple(raw_rows),
        )
        self._batch_rows(batch)
        return batch

    def is_tombstoned(self, object_type: str, object_id: str) -> bool:
        return (object_type, object_id) in self._tombstones

    @staticmethod
    def _event_plan(state: ContinuationState,
                    event: Mapping[str, object]) -> RetentionPlan:
        fields = {
            "intention_ids": _ids(event.get("intention_ids"), "intention_ids"),
            "effect_ids": _ids(event.get("effect_ids"), "effect_ids"),
            "wake_tokens": _ids(event.get("wake_tokens"), "wake_tokens"),
            "signal_ids": _ids(event.get("signal_ids"), "signal_ids"),
            "return_receipts": _ids(event.get("return_receipts"), "return_receipts"),
        }
        return RetentionPlan(source_revision=state.revision, **fields)

    @staticmethod
    def _archive_identities(batch: RetentionBatch) -> dict[str, tuple[str, ...]]:
        values: dict[str, list[str]] = {kind: [] for kind in OBJECT_TYPES}
        for row in batch.rows:
            object_type = row.get("object_type")
            object_id = row.get("object_id")
            if object_type not in OBJECT_TYPES or not isinstance(object_id, str):
                raise RetentionError("retention archive identity is invalid")
            values[object_type].append(object_id)
        return {kind: tuple(sorted(values[kind])) for kind in OBJECT_TYPES}

    def validate_operation(self, operation: str, data: Mapping[str, object]) -> None:
        checks: list[tuple[str, object]] = []
        if operation == "create":
            checks.append(("intention", data.get("new_id")))
        elif operation in {"enqueue", "detour"}:
            child = data.get("child")
            if isinstance(child, Mapping):
                checks.append(("intention", child.get("new_id")))
            if operation == "detour":
                checks.append(("wake_token", data.get("wake_token")))
        elif operation == "begin_effect":
            checks.append(("effect", data.get("effect_id")))
        elif operation == "suspend":
            checks.append(("wake_token", data.get("wake_token")))
        elif operation == "wake":
            checks.append(("wake_token", data.get("wake_token")))
            checks.append(("signal", data.get("event_id")))
            checks.append(("return_receipt", data.get("receipt_ref")))
        elif operation in {"complete", "cancel"}:
            checks.append(("return_receipt", data.get("receipt_ref")))
        for object_type, object_id in checks:
            if isinstance(object_id, str) and self.is_tombstoned(object_type, object_id):
                raise ContinuationError(
                    f"{object_type} identity was reclaimed and is reserved")

    def apply_event(self, state: ContinuationState, event: Mapping[str, object],
                    *, expected_sequence: int) -> ContinuationState:
        if event.get("retention_schema") != FORMAT:
            raise RetentionError("invalid continuation retention schema")
        if (type(expected_sequence) is not int
                or type(event.get("checkpoint_sequence")) is not int
                or event.get("checkpoint_sequence") != expected_sequence):
            raise RetentionError("retention GC checkpoint sequence is not the replay watermark")
        if event.get("checkpoint_head") != event.get("previous_hash"):
            raise RetentionError("retention GC checkpoint head is not journal-bound")
        if event.get("checkpoint_state_hash") != digest(state.document()):
            raise RetentionError("retention GC checkpoint state is not verified")
        batch = self.batch_from_event(event)
        plan = self._event_plan(state, event)
        expected = {
            "intention": plan.intention_ids,
            "effect": plan.effect_ids,
            "wake_token": plan.wake_tokens,
            "signal": plan.signal_ids,
            "return_receipt": plan.return_receipts,
        }
        if self._archive_identities(batch) != expected:
            raise RetentionError("retention GC identity set does not match archive")
        if event.get("reclaimed_digest") != plan.ids_digest():
            raise RetentionError("retention GC identity set changed")
        self.commit(batch, checkpoint_sequence=event["checkpoint_sequence"],
                    checkpoint_head=event["checkpoint_head"])
        compacted = self.compact(state, plan)
        if event.get("after_state_hash") != digest(compacted.document()):
            raise RetentionError("retention GC compacted state hash mismatch")
        return compacted

    def summary(self, state: ContinuationState) -> dict:
        return {
            "schema": FORMAT,
            "resident_intentions": len(state.intentions),
            "resident_effects": len(state.effects),
            "resident_wake_tokens": len(state.used_wake_tokens),
            "resident_signals": len(state.signals),
            "tombstones": len(self._tombstones),
            "last_reclaimed": copy.deepcopy(self._last_reclaimed),
            "archive_watermark": copy.deepcopy(self._last_watermark),
        }
