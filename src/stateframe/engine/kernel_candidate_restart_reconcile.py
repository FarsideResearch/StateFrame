"""R-042: reconcile one durable local edit candidate after a worker restart.

The opt-in seam is intentionally narrow.  It records an exact ``str_replace``
candidate before dispatch, then accepts only the two artifact states that can
be proved from that record: the original bytes (not applied) or the expected
bytes (applied).  Any other artifact is left unresolved for the existing
fail-closed restart path.
"""
from __future__ import annotations

import copy
import hashlib
import os
from bisect import bisect_right
from collections.abc import Mapping
from pathlib import Path

from kernel_continuation_evidence import binding_versions
from kernel_measurement import digest

SWITCH = "STATEFRAME_CANDIDATE_RESTART_RECONCILE"
EDIT_TOOL = "str_replace"
MAX_CANDIDATE_TEXT = 65536


def enabled() -> bool:
    value = os.environ.get(SWITCH, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{SWITCH} must be off or on")
    return value == "on"


def _artifact_hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _workspace_file(root: Path, raw: object) -> tuple[str, Path] | None:
    try:
        if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
            return None
        parts = raw.replace("\\", "/").split("/")
        if any(part in {"", ".", ".."} for part in parts):
            return None
        path = root / raw
        if path.is_symlink():
            return None
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            return None
        return resolved.relative_to(root).as_posix(), resolved
    except (OSError, TypeError, ValueError):
        return None


def _intent(runtime, *, effect_id: str, arguments: Mapping[str, object],
            current: Mapping[str, object]) -> dict | None:
    root = Path(runtime.owner.root).resolve()
    located = _workspace_file(root, arguments.get("path"))
    old = arguments.get("old_str")
    new = arguments.get("new_str")
    if (located is None or not isinstance(old, str) or not isinstance(new, str)
            or not old or len(old) > MAX_CANDIDATE_TEXT or len(new) > MAX_CANDIDATE_TEXT):
        return None
    relative, path = located
    try:
        before_bytes = path.read_bytes()
        before = before_bytes.decode("utf-8")
    except (OSError, UnicodeError):
        return None
    if before.count(old) != 1:
        return None
    after = before.replace(old, new, 1)
    after_bytes = after.encode("utf-8")
    before_hash = _artifact_hash(before_bytes)
    expected_hash = _artifact_hash(after_bytes)
    if before_hash == expected_hash:
        return None
    state = runtime.continuation.engine.state
    item = state.intentions.get(state.active_id)
    if item is None or effect_id not in state.effects:
        return None
    candidate = {
        "tool": EDIT_TOOL, "path": relative, "old_str": old, "new_str": new,
    }
    candidate_digest = digest({
        "candidate": candidate, "before_hash": before_hash,
        "expected_hash": expected_hash,
    })
    acceptance = getattr(item, "acceptance", {})
    if not isinstance(acceptance, dict):
        acceptance = {}
    return {
        "intent_schema": "candidate-restart-intent/1",
        "effect_id": effect_id,
        "intention_id": item.intention_id,
        "dispatch_ref": str(state.effects[effect_id].dispatch_ref),
        "request_id": str(state.effects[effect_id].request_id),
        "step": int(state.effects[effect_id].step),
        "tool": EDIT_TOOL,
        "path": relative,
        "before_hash": before_hash,
        "expected_hash": expected_hash,
        "candidate_digest": candidate_digest,
        "candidate": candidate,
        "pending_decision": str(getattr(item, "pending_decision", ""))[:4096],
        "obligations": copy.deepcopy(acceptance),
        "dispatch_binding": copy.deepcopy(dict(current)),
    }


def persist_intent(runtime, *, effect_id: str, arguments: Mapping[str, object],
                   current: Mapping[str, object]) -> dict | None:
    """Persist the bounded candidate after the executive intent is durable."""
    if not enabled():
        return None
    intent = _intent(runtime, effect_id=effect_id, arguments=arguments, current=current)
    if intent is None:
        return None
    return runtime.journal.emit(
        "candidate_restart_intent",
        producer="candidate_restart_reconcile",
        effect_id=intent["effect_id"], intention_id=intent["intention_id"],
        dispatch_ref=intent["dispatch_ref"], request_id=intent["request_id"],
        step=intent["step"], tool=intent["tool"], path=intent["path"],
        before_hash=intent["before_hash"], expected_hash=intent["expected_hash"],
        candidate_digest=intent["candidate_digest"], candidate=intent["candidate"],
        pending_decision=intent["pending_decision"], obligations=intent["obligations"],
        dispatch_binding=intent["dispatch_binding"],
    )


class _RestartIndex:
    """Latest and first-match joins over the validated restart journal."""

    def __init__(self, records: list[dict]):
        self.intents: dict[str, dict] = {}
        self.begins: dict[str, dict] = {}
        self.fences: dict[str, dict] = {}
        self.fences_by_sequence: dict[int, dict] = {}
        self.receipts: dict[tuple[str, str, str], dict] = {}
        self.closures: dict[tuple[str, str], dict] = {}
        self.unknown: set[tuple[str, str | None]] = set()
        self.dispatches: dict[tuple[str, int], list[dict]] = {}
        self.parent_budgets: list[dict] = []
        self.predispatches: dict[int, list[dict]] = {}
        self.last_sequence = 0
        for event in records:
            self.add(event)

    def add(self, event: dict) -> None:
        kind = event.get("kind")
        sequence = event.get("sequence", 0)
        self.last_sequence = max(self.last_sequence, sequence)
        effect_id = event.get("effect_id")
        if kind == "candidate_restart_intent" and isinstance(effect_id, str):
            self.intents[effect_id] = event
        elif kind == "continuation_transition" and event.get("operation") == "begin_effect":
            data = event.get("data")
            if isinstance(data, Mapping) and isinstance(data.get("effect_id"), str):
                self.begins[data["effect_id"]] = event
        elif kind == "step_dispatch_binding":
            ref = event.get("event_id")
            if isinstance(ref, str):
                self.fences.setdefault(ref, event)
            self.fences_by_sequence.setdefault(sequence, event)
            request_id, step = event.get("request_id"), event.get("step")
            if isinstance(request_id, str) and type(step) is int:
                self.dispatches.setdefault((request_id, step), []).append(event)
        elif kind == "action_receipt":
            receipt = event.get("receipt")
            if isinstance(receipt, Mapping):
                key = (receipt.get("continuation_effect_id"),
                       receipt.get("continuation_dispatch_ref"),
                       receipt.get("continuation_id"))
                if all(isinstance(part, str) for part in key):
                    self.receipts[key] = event
        elif kind == "action_outcome_closed":
            record = event.get("record")
            if isinstance(record, Mapping):
                key = (record.get("continuation_effect_id"),
                       record.get("continuation_dispatch_ref"))
                if all(isinstance(part, str) for part in key):
                    self.closures[key] = event
        elif kind == "candidate_restart_reconciled" and event.get("outcome") == "UNKNOWN":
            actual_hash = event.get("actual_hash")
            if isinstance(effect_id, str) and (actual_hash is None or isinstance(actual_hash, str)):
                self.unknown.add((effect_id, actual_hash))
        elif (kind == "parent_budget" and event.get("category") == "tool_work"
              and event.get("reason") == "worker dispatch"):
            self.parent_budgets.append(event)
        elif kind == "predispatch_decision" and type(event.get("step")) is int:
            self.predispatches.setdefault(event["step"], []).append(event)

    @staticmethod
    def between(rows: list[dict], begin: int, end: int) -> list[dict]:
        return rows[bisect_right(rows, begin, key=lambda row: row["sequence"]):
                    bisect_right(rows, end - 1, key=lambda row: row["sequence"])]


def _fence(index: _RestartIndex, intent: Mapping[str, object], effect) -> dict | None:
    reference = intent.get("dispatch_ref") or getattr(effect, "dispatch_ref", "")
    if not isinstance(reference, str):
        return None
    fence = index.fences.get(reference)
    if not isinstance(fence, dict):
        return None
    if (fence.get("continuation_effect_id") != effect.effect_id
            or fence.get("continuation_id") != effect.intention_id
            or fence.get("tool") != EDIT_TOOL
            or not isinstance(fence.get("current"), Mapping)):
        return None
    return fence


def _recovery_payload(intent: Mapping[str, object], *, outcome: str,
                      actual_hash: str | None, portfolio_candidate_id: str | None = None) -> dict:
    payload = {
        "outcome": outcome,
        "effect_id": intent.get("effect_id"),
        "intention_id": intent.get("intention_id"),
        "tool": intent.get("tool"),
        "path": intent.get("path"),
        "candidate_digest": intent.get("candidate_digest"),
        "candidate": copy.deepcopy(intent.get("candidate", {})),
        "decision": intent.get("pending_decision", ""),
        "obligations": copy.deepcopy(intent.get("obligations", {})),
        "before_hash": intent.get("before_hash"),
        "expected_hash": intent.get("expected_hash"),
        "actual_hash": actual_hash,
        "next_action": "run_tests" if outcome == "APPLIED" else "new_str_replace",
    }
    if portfolio_candidate_id is not None:
        payload["portfolio_candidate_id"] = portfolio_candidate_id
    return payload


def _recovery_from_event(event: Mapping[str, object], records: list[dict]) -> dict | None:
    """Rehydrate the latest recovery projection after a later worker restart."""
    outcome = event.get("outcome")
    if outcome not in {"APPLIED", "NOT_APPLIED", "UNKNOWN"}:
        return None
    candidate_digest = event.get("candidate_digest")
    linked_id = None
    if isinstance(candidate_digest, str):
        linked = [row for row in records
                  if row.get("kind") == "candidate_restart_portfolio_linked"
                  and row.get("candidate_digest") == candidate_digest]
        if linked:
            value = linked[-1].get("candidate_id")
            linked_id = value if isinstance(value, str) and value else None
    intent = {
        key: copy.deepcopy(event.get(key))
        for key in ("effect_id", "intention_id", "tool", "path", "candidate_digest",
                    "candidate", "before_hash", "expected_hash", "pending_decision",
                    "obligations")
    }
    return _recovery_payload(
        intent, outcome=outcome, actual_hash=event.get("actual_hash"),
        portfolio_candidate_id=linked_id)


def _restore_latest_recovery(runtime, records: list[dict]) -> None:
    events = [event for event in records
              if event.get("kind") == "candidate_restart_reconciled"]
    if not events:
        return
    recovery = _recovery_from_event(events[-1], records)
    if recovery is not None:
        runtime._candidate_restart_recovery = recovery


def _emit_recovery(runtime, intent: Mapping[str, object], *, outcome: str,
                   actual_hash: str | None, candidate: Mapping[str, object],
                   ids: set[str]) -> None:
    event = runtime.journal.emit(
        "candidate_restart_reconciled",
        producer="candidate_restart_reconcile",
        effect_id=intent.get("effect_id"), intention_id=intent.get("intention_id"),
        outcome=outcome, path=intent.get("path"),
        before_hash=intent.get("before_hash"), expected_hash=intent.get("expected_hash"),
        actual_hash=actual_hash, candidate_digest=intent.get("candidate_digest"),
        candidate=copy.deepcopy(candidate),
        pending_decision=intent.get("pending_decision", ""),
        obligations=copy.deepcopy(intent.get("obligations", {})),
    )
    ids.add(event["event_id"])
    runtime._candidate_restart_recovery = _recovery_payload(
        intent, outcome=outcome, actual_hash=actual_hash)


def _reconcile_effect(runtime, effect, intent_event: Mapping[str, object],
                      index: _RestartIndex) -> set[str]:
    intent = intent_event
    ids: set[str] = {str(intent_event["event_id"])}
    begin = index.begins.get(effect.effect_id)
    if begin is not None:
        ids.add(str(begin["event_id"]))
    fence = _fence(index, intent, effect)
    if fence is not None:
        ids.add(str(fence["event_id"]))
        prior_fence = index.fences_by_sequence.get(fence.get("sequence", 0) - 1)
        if prior_fence is not None and (prior_fence.get("current") != fence.get("current")
                                        or prior_fence.get("expected") != fence.get("expected")):
            prior_fence = None
        if prior_fence is not None:
            ids.add(str(prior_fence["event_id"]))
    begin_sequence = begin.get("sequence", 0) if begin is not None else 0
    receipt_event = index.receipts.get((effect.effect_id, effect.dispatch_ref, effect.intention_id))
    upper_sequence = (receipt_event.get("sequence") if receipt_event is not None
                      else index.last_sequence + 1)
    window_rows = (
        index.dispatches.get((effect.request_id, effect.step), []),
        index.parent_budgets,
        index.predispatches.get(effect.step, []),
    )
    for rows in window_rows:
        for event in index.between(rows, begin_sequence, upper_sequence):
            ids.add(str(event["event_id"]))
    located = _workspace_file(Path(runtime.owner.root).resolve(), intent.get("path"))
    actual_hash = None
    path = located[1] if located is not None else None
    if path is not None:
        try:
            actual_hash = _artifact_hash(path.read_bytes())
        except OSError:
            actual_hash = None
    before_hash = intent.get("before_hash")
    expected_hash = intent.get("expected_hash")
    existing_receipt = receipt_event
    existing_result = (existing_receipt or {}).get("receipt", {}).get("result_class")
    if (fence is None or not isinstance(before_hash, str)
            or not isinstance(expected_hash, str)
            or (existing_result == "APPLIED" and actual_hash != expected_hash)):
        outcome = "UNKNOWN"
    elif actual_hash == expected_hash and expected_hash != before_hash:
        outcome = "APPLIED"
    elif actual_hash == before_hash and existing_receipt is None:
        outcome = "NOT_APPLIED"
    else:
        outcome = "UNKNOWN"
    if outcome == "UNKNOWN":
        if (effect.effect_id, actual_hash) not in index.unknown:
            _emit_recovery(runtime, intent, outcome=outcome, actual_hash=actual_hash,
                           candidate=intent.get("candidate", {}), ids=ids)
        else:
            runtime._candidate_restart_recovery = _recovery_payload(
                intent, outcome=outcome, actual_hash=actual_hash)
        return ids

    before_binding = fence.get("current")
    current_binding = runtime.binding()
    if (not isinstance(before_binding, Mapping)
            or binding_versions(dict(before_binding)) != effect.dependencies):
        _emit_recovery(runtime, intent, outcome="UNKNOWN", actual_hash=actual_hash,
                       candidate=intent.get("candidate", {}), ids=ids)
        return ids
    if outcome == "NOT_APPLIED" and current_binding != before_binding:
        _emit_recovery(runtime, intent, outcome="UNKNOWN", actual_hash=actual_hash,
                       candidate=intent.get("candidate", {}), ids=ids)
        return ids
    changes = []
    if outcome == "APPLIED":
        before_version = before_binding.get("product_files", {}).get(intent.get("path"))
        after_version = current_binding.get("product_files", {}).get(intent.get("path"))
        if before_version == after_version:
            _emit_recovery(runtime, intent, outcome="UNKNOWN", actual_hash=actual_hash,
                           candidate=intent.get("candidate", {}), ids=ids)
            return ids
        changes = [{"path": intent.get("path"), "role": "product",
                    "before": before_version, "after": after_version}]

    receipt_id = None
    if existing_receipt is not None:
        receipt = existing_receipt["receipt"]
        receipt_id = receipt.get("receipt_id")
        ids.add(str(existing_receipt["event_id"]))
    if not isinstance(receipt_id, str) or not receipt_id:
        receipt_id = "candidate-restart:" + str(effect.effect_id)
        receipt = {
            "receipt_id": receipt_id, "kind": "edit", "turn": effect.step,
            "result_class": "APPLIED" if outcome == "APPLIED" else "NO_OP",
            "detail": "candidate/restart reconciliation matched the persisted artifact hash",
            "intent": "candidate_restart_reconcile",
            "producer": "candidate_restart_reconcile", "run_id": runtime.journal.run_id,
            "request_id": effect.request_id,
            "continuation_id": effect.intention_id,
            "continuation_effect_id": effect.effect_id,
            "continuation_dispatch_ref": effect.dispatch_ref,
            "target_path": intent.get("path"), "artifact_hash": actual_hash,
        }
        receipt_event = runtime.journal.emit("action_receipt", receipt=receipt)
        ids.add(str(receipt_event["event_id"]))

    closed = index.closures.get((effect.effect_id, effect.dispatch_ref))
    if closed is None:
        record = {
            "step": effect.step, "tool": EDIT_TOOL, "target": intent.get("path", ""),
            "arguments_hash": effect.arguments_hash,
            "action_parameters": {"path": intent.get("path", "")},
            "intent": {"text": intent.get("pending_decision", ""),
                       "status": "RECOVERED_HOST_INTENT"},
            "result": "APPLIED" if outcome == "APPLIED" else "NO_OP",
            "receipt_ids": [receipt_id],
            "observation": "candidate/restart reconciled the actual artifact hash",
            "observation_hash": digest("candidate/restart reconciled the actual artifact hash"),
            "observation_omitted_chars": 0, "changes": changes,
            "before_binding": copy.deepcopy(dict(before_binding)),
            "after_binding": copy.deepcopy(current_binding),
            "continuation_id": effect.intention_id,
            "continuation_effect_id": effect.effect_id,
            "continuation_dispatch_ref": effect.dispatch_ref,
            "artifact_before_hash": before_hash, "artifact_after_hash": actual_hash,
        }
        record["step_record_id"] = digest(record)
        closed = runtime.journal.emit("action_outcome_closed", record=record)
    ids.add(str(closed["event_id"]))
    settled = runtime.continuation.engine.settle_host_effect(
        effect_id=effect.effect_id, receipt_ref=closed["event_id"],
        current_binding=current_binding)
    ids.add(str(settled["event_id"]))
    _emit_recovery(runtime, intent, outcome=outcome, actual_hash=actual_hash,
                   candidate=intent.get("candidate", {}), ids=ids)
    return ids


def reconcile(runtime) -> set[str]:
    """Reconcile supported unresolved effects and return consumed tail events."""
    if not enabled():
        return set()
    records = list(runtime.journal.records())
    cursor = records[-1]["sequence"] if records else 0
    index = _RestartIndex(records)
    state = runtime.continuation.engine.state
    consumed: set[str] = set()
    for effect in state.effects.values():
        if effect.tool != EDIT_TOOL or effect.status not in {"pending", "unknown"}:
            continue
        intent_event = index.intents.get(effect.effect_id)
        if intent_event is None:
            continue
        consumed.update(_reconcile_effect(runtime, effect, intent_event, index))
        appended = runtime.journal.cached_records_since(cursor)
        records.extend(appended)
        for event in appended:
            index.add(event)
        if appended:
            cursor = appended[-1]["sequence"]
    _restore_latest_recovery(runtime, records)
    return consumed


def context_payload(runtime) -> dict | None:
    if not enabled():
        return None
    recovery = getattr(runtime, "_candidate_restart_recovery", None)
    if not isinstance(recovery, Mapping):
        return None
    candidate = recovery.get("candidate")
    if isinstance(candidate, Mapping):
        candidate = {"tool": candidate.get("tool"), "path": candidate.get("path"),
                     "candidate_digest": recovery.get("candidate_digest")}
    payload = {key: copy.deepcopy(value) for key, value in recovery.items()
               if key != "candidate"}
    payload["candidate"] = candidate
    return payload


def maybe_promote_portfolio(runtime) -> None:
    """Use existing portfolio admission after a recovered edit reaches green."""
    if not enabled():
        return
    recovery = getattr(runtime, "_candidate_restart_recovery", None)
    if (not isinstance(recovery, dict) or recovery.get("outcome") != "APPLIED"
            or recovery.get("portfolio_candidate_id") is not None
            or getattr(getattr(runtime, "owner", None), "last_test_exit", None) != 0):
        return
    preserve = getattr(runtime, "portfolio_preserve_candidate", None)
    if not callable(preserve) or getattr(runtime, "portfolio", None) is None:
        return
    # _restore_latest_recovery folds durable links into this projection on resume.
    # A live promotion also sets the ID immediately after its link is appended.
    candidate = preserve(reason="candidate/restart reconciliation reached current public green")
    if candidate is None:
        return
    recovery["portfolio_candidate_id"] = candidate.candidate_id
    runtime.journal.emit(
        "candidate_restart_portfolio_linked",
        producer="candidate_restart_reconcile",
        candidate_id=candidate.candidate_id,
        candidate_digest=recovery.get("candidate_digest"),
        effect_id=recovery.get("effect_id"),
    )
