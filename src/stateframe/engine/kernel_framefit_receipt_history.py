"""Incremental semantic frontier over the durable continuation receipt stream.

Repeated receipts cost constant space. Distinct read (path, workspace revision)
keys and edited paths remain explicit authority, not an arbitrary eviction cap.
Every superseded/opaque receipt enters the existing capacity hash-chain format.
The journal's validated replay reconstructs this cache after restart.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from kernel_stateframe_capacity import _fold, _sha
from kernel_stateframe_call_coverage import (
    CONTEXT_KEY, CoverageSession, bind_index, context_index,
)
from kernel_vacuity_truth import _applied_edit_receipt, journal_applied_edit_rows


def read_receipt(event: Mapping[str, object]) -> tuple[str, dict[str, object]] | None:
    """Use exactly the typed read-fact admission/identity contract."""
    receipt = event.get("receipt")
    if (event.get("kind") != "action_receipt" or not isinstance(receipt, Mapping)
            or receipt.get("kind") != "read"):
        return None
    if str(receipt.get("result_class") or "").strip().upper() in {
        "RECALLED", "RECALL_REJECTED", "REJECTED", "FAILED", "ERROR",
        "INVALID_ARGUMENT", "MISSING", "NOT_FOUND", "OUT_OF_RANGE",
    }:
        return None
    from kernel_stateframe_framefit5 import FrameFit5Error, _pathless_source_path, _safe_relative_path

    # A typed read receipt without a target path is not a source-file read (e.g. a check-output read
    # whose handle was not found). It creates no file read fact; a present-but-unsafe path still raises.
    target = receipt.get("target_path")
    if not isinstance(target, str) or _pathless_source_path(target):
        return None
    try:
        path = _safe_relative_path(receipt.get("target_path"))
    except FrameFit5Error as error:
        raise FrameFit5Error("FRAMEFIT5 journal read receipt contains an unsafe path") from error
    revision = receipt.get("workspace_revision")
    return path, {
        "receipt_id": str(receipt.get("receipt_id") or "")[:160],
        "workspace_revision": str(revision)[:512] if revision is not None else "",
    }


def fold_summary(summary: dict[str, object], row: Mapping[str, object]) -> None:
    summary["count"] += 1
    summary["sha256"] = _sha([summary["sha256"], row])


def empty_summary() -> dict[str, object]:
    return {"count": 0, "sha256": _sha([])}


def coverage_session_for(journal: object, digest: Mapping[str, object] | None) -> CoverageSession | None:
    """Reuse the journal-owned CI index for every receipt-history fold batch."""
    index = context_index(bind_index({}, journal))
    return CoverageSession(index, digest) if index is not None else None


class ReceiptHistory:
    """Host-owned reducer. Ordering follows append order, never replay source IDs."""

    def __init__(self, *, public_run_id: str | None = None) -> None:
        self.public_run_id = public_run_id
        self.count = 0
        self.sha256 = _sha([])
        self.frontier: dict[tuple[str, ...], tuple[int, dict[str, object]]] = {}
        self.ledger: dict[str, object] = {}
        self.reads: dict[str, dict[str, object]] = {}
        self.historical_reads = empty_summary()
        self.historical_edits = empty_summary()

    def _public_event(self, event: Mapping[str, object]) -> dict[str, object]:
        """Keep durable host identity in the witness, not its model-visible hash."""
        result = copy.deepcopy(dict(event))
        if self.public_run_id is None:
            return result  # Preserve historical/synthetic digest semantics.
        if "run_id" in result:
            result["run_id"] = self.public_run_id
        result.pop("timestamp_ns", None)
        sequence = result.get("sequence")
        if type(sequence) is int:
            result["event_id"] = f"{self.public_run_id}:{sequence}"
        return result

    def coverage_session(self, journal: object) -> CoverageSession | None:
        digest = self.ledger.get("history_digest")
        return coverage_session_for(journal, digest if isinstance(digest, Mapping) else None)

    def ensure_coverage(self, coverage: CoverageSession) -> None:
        """Promote a replay-built legacy digest into CI's proven host index."""
        digest = self.ledger.get("history_digest")
        if not isinstance(digest, dict) or "coverage_contract" in digest:
            return
        coverage.fold(digest, set())

    def _retire(self, event: Mapping[str, object], coverage: CoverageSession | None) -> None:
        _fold(self.ledger, self._public_event(event), kind="journal_" + str(event["kind"]),
              exact_ids=set(), coverage=coverage)
        read = read_receipt(event)
        if read is not None:
            path, receipt = read
            fold_summary(self.reads[path]["historical_receipts"], receipt)
            fold_summary(self.historical_reads, {"path": path, **receipt})
        edits = journal_applied_edit_rows((event,))
        if edits:
            fold_summary(self.historical_edits, edits[0])

    def observe(self, event: Mapping[str, object], *,
                coverage: CoverageSession | None = None) -> None:
        event = copy.deepcopy(dict(event))
        key: tuple[str, ...] | None = None
        read = read_receipt(event)  # Reject unsafe paths before they can be folded away.
        if read is not None:
            path, receipt = read
            key = ("read", path, str(receipt["workspace_revision"]))
            summary = self.reads.setdefault(path, {
                "receipt_count": 0, "historical_receipts": empty_summary(),
            })
            summary["receipt_count"] += 1
        elif event.get("kind") == "check_receipt":
            key = ("check",)
        elif (edit := _applied_edit_receipt(event)) is not None:
            key = ("edit", _sha(edit.get("target_path")))

        self.count += 1
        self.sha256 = _sha([self.sha256, self._public_event(event)])
        if key is None:
            self._retire(event, coverage)
        else:
            previous = self.frontier.get(key)
            if previous is not None:
                self._retire(previous[1], coverage)
            self.frontier[key] = (self.count, event)

    def context(self) -> dict[str, object]:
        records = [event for _, event in sorted(self.frontier.values(), key=lambda pair: pair[0])]
        history = {
            "schema": "framefit-receipt-history/1",
            "receipt_count": self.count,
            "receipts_sha256": self.sha256,
            "history_digest": self.ledger.get("history_digest"),
            "reads": self.reads,
            "historical_read_facts": self.historical_reads,
            "historical_applied_edits": self.historical_edits,
        }
        context = copy.deepcopy({"journal_receipt_count": self.count,
                                 "journal_receipt_records": records,
                                 "journal_receipt_history": history})
        context["journal_receipt_history"]["projection_sha256"] = _sha(context)
        validate_context(context)
        return context


def validate_context(context: Mapping[str, object]) -> Mapping[str, object] | None:
    """Fail closed on detached/missing witnesses, counts, or damaged summaries."""
    history = context.get("journal_receipt_history")
    from kernel_stateframe_framefit5 import FrameFit5Error

    if history is None:
        count = context.get("journal_receipt_count")
        if count is not None and count != len(context.get("journal_receipt_records", ())):
            raise FrameFit5Error("journal receipt history accounting is missing")
        return None  # Legacy complete, unreduced callers remain supported.

    records = context.get("journal_receipt_records")
    if (not isinstance(history, Mapping) or history.get("schema") != "framefit-receipt-history/1"
            or not isinstance(records, Sequence) or isinstance(records, (str, bytes))):
        raise FrameFit5Error("journal receipt history is malformed")
    unsigned = dict(history)
    signature = unsigned.pop("projection_sha256", None)
    projection = {
        "journal_receipt_count": context.get("journal_receipt_count"),
        "journal_receipt_records": records,
        "journal_receipt_history": unsigned,
    }
    if CONTEXT_KEY in context:
        projection[CONTEXT_KEY] = context[CONTEXT_KEY]
    if signature != _sha(projection):
        raise FrameFit5Error("journal receipt history projection hash mismatch")
    digest = history.get("history_digest")
    folded = digest.get("row_count", 0) if isinstance(digest, Mapping) else 0
    count = history.get("receipt_count")
    if (type(count) is not int or count != context.get("journal_receipt_count")
            or type(folded) is not int or folded < 0 or count != len(records) + folded):
        raise FrameFit5Error("journal receipt history accounting is incomplete")
    return history


def attach_history(state: dict[str, object], context: Mapping[str, object]) -> None:
    history = validate_context(context)
    if history is None:
        return
    digest = history.get("history_digest")
    if isinstance(digest, Mapping):
        state["action_ledger"]["history_digest"] = copy.deepcopy(dict(digest))
    for name in ("historical_read_facts", "historical_applied_edits"):
        if history[name]["count"]:
            state[name] = copy.deepcopy(history[name])
    state["receipt_projection"] = {
        "receipt_count": history["receipt_count"],
        "receipts_sha256": history["receipts_sha256"],
        "frontier_count": len(context["journal_receipt_records"]),
        "historical_count": digest["row_count"] if isinstance(digest, Mapping) else 0,
    }
