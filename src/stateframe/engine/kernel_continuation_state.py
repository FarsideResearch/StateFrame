"""Typed continuation projection; only the host executive may submit transitions.

This module is a pure reducer, not a tool surface or an evidence authority.
The journal adapter validates host receipt references before submitting receipt
transitions. Model-authored text is a pending decision, never completion proof.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Literal

from kernel_stale_edit_policy import allows_stale_repair

Phase = Literal["eligible", "active", "suspended", "completed", "cancelled", "blocked"]
EffectStatus = Literal["pending", "applied", "not_applied", "unknown"]
TERMINAL = frozenset({"completed", "cancelled"})


class ContinuationError(ValueError):
    """A transition would lose work, reuse stale evidence or duplicate an effect."""


@dataclass
class Wake:
    token: str
    kind: Literal["child", "signal", "timer"]
    deadline_us: int
    child_id: str | None = None


@dataclass
class Intention:
    intention_id: str
    parent_id: str | None
    objective: str
    pending_decision: str
    dependencies: dict[str, str]
    created_us: int
    last_progress_us: int
    turns_left: int | None
    acceptance: dict = field(default_factory=dict)
    phase: Phase = "eligible"
    revision: int = 1
    suspension_reason: str = ""
    wake: Wake | None = None
    stale_dependencies: list[str] = field(default_factory=list)
    return_receipts: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    quanta: int = 0
    resumptions: int = 0
    return_pending: bool = False
    terminal_receipt: str | None = None
    terminal_reason: str = ""


@dataclass
class Effect:
    effect_id: str
    intention_id: str
    action_hash: str
    dependencies: dict[str, str]
    read_only: bool
    tool: str = ""
    arguments_hash: str = ""
    dispatch_ref: str = ""
    step: int = 0
    request_id: str = ""
    intent_sequence: int = 0
    status: EffectStatus = "pending"
    attempts: int = 1
    receipt_ref: str | None = None


@dataclass
class ContinuationState:
    revision: int = 0
    last_time_us: int = 0
    used_wake_tokens: list[str] = field(default_factory=list)
    active_id: str | None = None
    intentions: dict[str, Intention] = field(default_factory=dict)
    effects: dict[str, Effect] = field(default_factory=dict)
    signals: dict[str, str] = field(default_factory=dict)

    def document(self) -> dict:
        return asdict(self)


def text(value: object, name: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ContinuationError(f"{name} must be a nonempty string of at most {limit} characters")
    return value


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ContinuationError(f"{name} must be an integer >= {minimum}")
    return value


def dependencies(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or len(value) > 256:
        raise ContinuationError("dependencies must be a bounded resource/version map")
    return {text(key, "resource", 1024): text(version, "version", 256)
            for key, version in value.items()}


def intention(state: ContinuationState, identity: object) -> Intention:
    key = text(identity, "intention_id", 256)
    if key not in state.intentions:
        raise ContinuationError("unknown intention")
    return state.intentions[key]


def unresolved(state: ContinuationState, key: str) -> list[str]:
    return [e.effect_id for e in state.effects.values()
            if e.intention_id == key and e.status in {"pending", "unknown"}]


def children_open(state: ContinuationState, key: str) -> list[str]:
    # Cancelled required work is not silently counted as successful completion.
    return [i.intention_id for i in state.intentions.values()
            if i.parent_id == key and i.phase != "completed"]


def _active(state: ContinuationState, key: object) -> Intention:
    item = intention(state, key)
    if item.phase != "active" or state.active_id != item.intention_id:
        raise ContinuationError("intention is not the active continuation")
    return item


def _settled(state: ContinuationState, item: Intention) -> None:
    if unresolved(state, item.intention_id):
        raise ContinuationError("effect outcome unresolved; reconcile before progressing")


def _new(state: ContinuationState, data: dict, now: int, parent: str | None) -> Intention:
    key = text(data.get("new_id"), "new_id", 256)
    if key in state.intentions or sum(i.phase not in TERMINAL for i in state.intentions.values()) >= 128:
        raise ContinuationError("duplicate intention or capacity exhausted")
    item = Intention(key, parent, text(data.get("objective"), "objective"),
                     text(data.get("pending_decision"), "pending_decision"),
                     dependencies(data.get("dependencies", {})), now, now,
                     (None if "turns" in data and data["turns"] is None
                      else integer(data.get("turns"), "turns", 1)))
    if item.turns_left is not None and item.turns_left > 10000:
        raise ContinuationError("turn budget exceeds limit")
    acceptance = data.get("acceptance", {})
    if not isinstance(acceptance, dict):
        raise ContinuationError("acceptance must be a host contract")
    item.acceptance = copy.deepcopy(acceptance)
    state.intentions[key] = item
    return item


def _suspend(state: ContinuationState, item: Intention, data: dict, now: int,
             kind: Literal["child", "signal", "timer"], child: str | None = None) -> None:
    _settled(state, item)
    token = text(data.get("wake_token"), "wake_token", 256)
    if token in state.used_wake_tokens:
        raise ContinuationError("wake token already in use")
    due = integer(data.get("deadline_us"), "deadline_us", now + 1)
    item.phase, item.suspension_reason = "suspended", text(data.get("reason"), "reason")
    item.wake = Wake(token, kind, due, child)
    state.used_wake_tokens.append(token)
    item.revision += 1
    state.active_id = None


def transition(previous: ContinuationState, operation: str, data: dict) -> ContinuationState:
    """Validate against a copy; a rejected transition never mutates prior state."""
    if not isinstance(data, dict):
        raise ContinuationError("transition data must be an object")
    state = copy.deepcopy(previous)
    now = integer(data.get("now_us"), "now_us", previous.last_time_us)
    state.last_time_us = now
    if operation == "create":
        if state.intentions:
            raise ContinuationError("root already exists; use a scoped detour")
        _new(state, data, now, None)
    elif operation == "enqueue":
        parent = intention(state, data.get("parent_id"))
        if parent.phase in TERMINAL:
            raise ContinuationError("cannot add work to a terminal parent")
        child_data = data.get("child")
        if not isinstance(child_data, dict):
            raise ContinuationError("child must be a scoped intention")
        _new(state, child_data, now, parent.intention_id)
        parent.revision += 1
    elif operation == "select":
        item = intention(state, data.get("intention_id"))
        continuing = state.active_id == item.intention_id and item.phase == "active"
        if (not continuing and (state.active_id is not None or item.phase != "eligible")) or item.turns_left == 0:
            raise ContinuationError("continuation is not eligible")
        _settled(state, item)
        current = dependencies(data.get("current_dependencies"))
        item.stale_dependencies = sorted(k for k, v in item.dependencies.items() if current.get(k) != v)
        item.phase, state.active_id = "active", item.intention_id
        item.quanta += 1
        if item.turns_left is not None:
            item.turns_left -= 1
        item.revision += 1
        if item.return_pending:
            item.resumptions += 1
            item.return_pending = False
    elif operation == "yield":
        item = _active(state, data.get("intention_id"))
        _settled(state, item)
        item.phase = "blocked" if item.turns_left == 0 else "eligible"
        if item.turns_left == 0:
            item.suspension_reason = "turn budget exhausted"
        item.revision += 1
        state.active_id = None
    elif operation == "exhaust":
        item = intention(state, data.get("intention_id"))
        _settled(state, item)
        if item.phase not in {"active", "eligible"} or item.turns_left != 0:
            raise ContinuationError("continuation budget is not exhausted")
        item.phase, item.suspension_reason = "blocked", "turn budget exhausted; intervention required"
        item.revision += 1
        if state.active_id == item.intention_id:
            state.active_id = None
    elif operation == "detour":
        item = _active(state, data.get("intention_id"))
        child_data = data.get("child")
        if not isinstance(child_data, dict):
            raise ContinuationError("child must be a scoped intention")
        child = _new(state, child_data, now, item.intention_id)
        _suspend(state, item, data, now, "child", child.intention_id)
    elif operation == "suspend":
        item = _active(state, data.get("intention_id"))
        kind = data.get("wake_kind")
        if kind not in {"signal", "timer"}:
            raise ContinuationError("external suspension requires signal or timer")
        _suspend(state, item, data, now, kind)
    elif operation == "wake":
        item = intention(state, data.get("intention_id"))
        event_id = text(data.get("event_id"), "event_id", 256)
        event_hash = text(data.get("event_hash"), "event_hash", 256)
        if event_id in state.signals:
            raise ContinuationError("wake event was already consumed")
        if (item.phase != "suspended" or item.wake is None or item.wake.kind != "signal"
                or item.wake.token != data.get("wake_token") or now >= item.wake.deadline_us):
            raise ContinuationError("wake does not match a current unexpired suspension")
        state.signals[event_id] = event_hash
        item.return_receipts.append(text(data.get("receipt_ref"), "receipt_ref", 256))
        item.phase, item.wake = "eligible", None
        item.return_pending = True
        item.revision += 1
    elif operation == "expire":
        item = intention(state, data.get("intention_id"))
        if item.phase != "suspended" or item.wake is None or now < item.wake.deadline_us:
            raise ContinuationError("suspension has not reached its deadline")
        timer = item.wake.kind == "timer"
        item.phase, item.wake = ("eligible" if timer else "blocked"), None
        item.suspension_reason = "timer elapsed" if timer else "wake deadline expired; intervention required"
        item.return_pending = timer
        item.revision += 1
    elif operation == "revalidate":
        item = _active(state, data.get("intention_id"))
        _settled(state, item)
        item.dependencies = dependencies(data.get("current_dependencies"))
        item.stale_dependencies = []
        item.evidence_refs = [text(data.get("receipt_ref"), "receipt_ref", 256)]
        item.revision += 1
    elif operation == "begin_effect":
        item = _active(state, data.get("intention_id"))
        _settled(state, item)
        key = text(data.get("effect_id"), "effect_id", 256)
        action_hash = text(data.get("action_hash"), "action_hash", 256)
        current = dependencies(data.get("current_dependencies"))
        read_only = data.get("read_only")
        if type(read_only) is not bool:
            raise ContinuationError("read_only must be host classified")
        stale = any(current.get(k) != v for k, v in item.dependencies.items())
        # Verification is an observation opportunity, even when an earlier edit
        # invalidated evidence. Its outcome still requires exact host receipts.
        # HostEvidence separately proves an exact native prefix. This grants only
        # this effect; stale evidence and completion requirements stay unchanged.
        if ((item.stale_dependencies or stale) and not read_only and data.get("tool") != "run_tests"
                and not isinstance(data.get("native_selected_ref"), str)
                and not allows_stale_repair(data.get("tool"))):
            raise ContinuationError("stale evidence cannot authorize an effect")
        prior = state.effects.get(key)
        if prior is not None and (
                prior.status != "not_applied" or prior.intention_id != item.intention_id
                or prior.action_hash != action_hash or prior.dependencies != current
                or prior.read_only != read_only or prior.attempts >= 3):
            raise ContinuationError("effect is applied, unresolved, changed or retry-exhausted")
        state.effects[key] = Effect(
            key, item.intention_id, action_hash, current, read_only,
            tool=str(data.get("tool", "")), arguments_hash=str(data.get("arguments_hash", "")),
            dispatch_ref=str(data.get("dispatch_ref", "")),
            step=integer(data.get("step", 0), "step"),
            request_id=str(data.get("request_id", "")),
            intent_sequence=integer(data.get("intent_sequence", 0), "intent_sequence"),
            attempts=(prior.attempts + 1) if prior else 1)
        item.revision += 1
    elif operation == "settle_effect":
        key = text(data.get("effect_id"), "effect_id", 256)
        effect = state.effects.get(key)
        status = data.get("status")
        if effect is None or effect.status not in {"pending", "unknown"}:
            raise ContinuationError("no unresolved effect to settle")
        if status not in {"applied", "not_applied", "unknown"}:
            raise ContinuationError("invalid effect outcome")
        effect.status = status
        effect.receipt_ref = text(data.get("receipt_ref"), "receipt_ref", 256)
        item = state.intentions[effect.intention_id]
        item.revision += 1
        if status == "applied":
            item.last_progress_us = now
            # A write receipt does not refresh old evidence or prove completion.
            if not effect.read_only and data.get("binding_changed", True):
                item.stale_dependencies = sorted(item.dependencies)
                item.evidence_refs = []
    elif operation == "repair_effect":
        # A retry is a host reconciliation, never a model assertion.  It is
        # restricted to read-only observations and test execution, whose
        # unknown outcome cannot have mutated the workspace.  The durable
        # continuation_control receipt is validated by HostEvidence before this
        # reducer is allowed to close the old effect.
        key = text(data.get("effect_id"), "effect_id", 256)
        effect = state.effects.get(key)
        if effect is None or effect.status not in {"pending", "unknown"}:
            raise ContinuationError("no unresolved effect to repair")
        if not (effect.read_only or effect.tool == "run_tests"):
            raise ContinuationError("effect is not safely retryable")
        if effect.intention_id != data.get("intention_id"):
            raise ContinuationError("repair must stay within the active intention")
        item = _active(state, data.get("intention_id"))
        effect.status = "not_applied"
        effect.receipt_ref = text(data.get("receipt_ref"), "receipt_ref", 256)
        item.revision += 1
    elif operation == "complete":
        item = _active(state, data.get("intention_id"))
        _settled(state, item)
        current = dependencies(data.get("current_dependencies"))
        if item.stale_dependencies or current != item.dependencies or children_open(state, item.intention_id):
            raise ContinuationError("stale evidence or unfinished required children")
        receipt = text(data.get("receipt_ref"), "receipt_ref", 256)
        item.phase, item.terminal_receipt = "completed", receipt
        item.revision += 1
        state.active_id = None
        if item.parent_id is not None:
            parent = state.intentions[item.parent_id]
            parent.return_receipts.append(receipt)
            parent.revision += 1
            if (parent.phase == "suspended" and parent.wake is not None
                    and parent.wake.kind == "child" and parent.wake.child_id == item.intention_id
                    and now < parent.wake.deadline_us):
                parent.phase, parent.wake = "eligible", None
                parent.return_pending = True
    elif operation == "cancel":
        item = intention(state, data.get("intention_id"))
        _settled(state, item)
        if item.phase in TERMINAL:
            raise ContinuationError("intention already terminal")
        if any(i.parent_id == item.intention_id and i.phase not in TERMINAL
               for i in state.intentions.values()):
            raise ContinuationError("cancel required children before their parent")
        item.phase, item.wake = "cancelled", None
        item.terminal_reason = text(data.get("reason"), "reason")
        item.terminal_receipt = text(data.get("receipt_ref"), "receipt_ref", 256)
        item.revision += 1
        if state.active_id == item.intention_id:
            state.active_id = None
        if item.parent_id is not None:
            parent = state.intentions[item.parent_id]
            if parent.phase not in TERMINAL:
                parent.phase, parent.wake = "blocked", None
                parent.suspension_reason = "required child cancelled; intervention required"
                parent.revision += 1
                if state.active_id == parent.intention_id:
                    state.active_id = None
    else:
        raise ContinuationError("unknown continuation operation")
    state.revision += 1
    return state
