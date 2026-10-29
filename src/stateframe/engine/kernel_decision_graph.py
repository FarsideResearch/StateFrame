"""Bounded decision truth-maintenance for the v46 state frame.

The graph stores immutable decision records and derives current health from the
same evidence identifiers used by ``LedgerIndex``.  It is intentionally a
projection owner, not a completion owner: existing action/completion owners
still decide whether an admitted action can execute.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum


SCHEMA = "stateframe-decision-graph/1"
MAX_KEY_ITEMS = 8
MAX_ALTERNATIVES = 2
MAX_REASON_CHARS = 256
MAX_REASON_WORDS = 25
MAX_ALTERNATIVE_CHARS = 192
MAX_EVENTS = 512


class DecisionGraphError(ValueError):
    """Malformed decision data or an unsafe truth-maintenance transition."""


class BasisState(StrEnum):
    SUPPORTED = "SUPPORTED"
    SUSPECT = "SUSPECT"
    UNSUPPORTED = "UNSUPPORTED"


class ItemState(StrEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    FALSIFIED = "FALSIFIED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


class ActionKind(StrEnum):
    DECIDE = "decide"
    WHY = "why"
    REVISIT = "revisit"
    CHECKPOINT = "checkpoint"
    GATE = "gate"


class NoticeKind(StrEnum):
    BASIS_BROKEN = "BASIS_BROKEN"
    ALTERNATIVE_NEWLY_VIABLE = "ALTERNATIVE_NEWLY_VIABLE"
    ALTERNATIVE_REJECTION_UNSUPPORTED = "ALTERNATIVE_REJECTION_UNSUPPORTED"
    SUSPECT_TRANSITIVE = "SUSPECT_TRANSITIVE"
    SWITCH_COMMITTED = "SWITCH_COMMITTED"
    OSCILLATION_GUARD = "OSCILLATION_GUARD"


def _text(value: object, name: str, limit: int = MAX_REASON_CHARS, *, required: bool = False) -> str:
    if not isinstance(value, str):
        if required:
            raise DecisionGraphError(f"{name} must be a string")
        return ""
    result = " ".join(value.split())
    if required and not result:
        raise DecisionGraphError(f"{name} is required")
    if len(result) > limit:
        raise DecisionGraphError(f"{name} exceeds {limit} characters")
    return result


def _ids(value: object, name: str, limit: int) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Sequence[object]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise DecisionGraphError(f"{name} must be a string or sequence")
    result: list[str] = []
    for item in values:
        if isinstance(item, Mapping):
            item = item.get("id", item.get("ref", item.get("node_id", "")))
        ident = _text(item, f"{name} item", 96, required=True)
        if ident in result:
            raise DecisionGraphError(f"{name} contains duplicate ids")
        result.append(ident)
    if len(result) > limit:
        raise DecisionGraphError(f"{name} exceeds {limit} items")
    return tuple(result)


def _condition(value: object, name: str, role: str) -> "ConditionRef | None":
    if value in (None, ""):
        return None
    if isinstance(value, Mapping):
        ident = value.get("predicate_id", value.get("id", value.get("ref", "")))
        condition_role = value.get("role", role)
    else:
        ident = value
        condition_role = role
    condition_id = _text(ident, name, 96, required=True)
    condition_role_text = _text(condition_role, f"{name}.role", 48, required=True).upper()
    allowed = {"CURRENT_FALSIFIER", "ALTERNATIVE_VIABILITY", "COMPARISON_REVIEW"}
    if condition_role_text not in allowed:
        raise DecisionGraphError(f"{name}.role is unsupported")
    return ConditionRef(condition_id, condition_role_text)


def _word_count(value: str) -> int:
    return len(value.split())


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:16]


def _safe_float(value: object, name: str, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DecisionGraphError(f"{name} must be finite")
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise DecisionGraphError(f"{name} must be between 0 and 1")
    return numeric


@dataclass(frozen=True)
class ConditionRef:
    predicate_id: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return {"predicate_id": self.predicate_id, "role": self.role}


@dataclass(frozen=True)
class AlternativeNode:
    option: str
    why_rejected: str
    evidence_ids: tuple[str, ...] = ()
    would_choose_if: ConditionRef | None = None
    rejection_state: ItemState = ItemState.CURRENT

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "ALTERNATIVE",
            "option": self.option,
            "why_rejected": self.why_rejected,
            "evidence_ids": list(self.evidence_ids),
            "would_choose_if": self.would_choose_if.to_dict() if self.would_choose_if else None,
            "rejection_state": self.rejection_state.value,
        }


@dataclass(frozen=True)
class ItemRecord:
    item_id: str
    state: ItemState = ItemState.CURRENT
    reason: str = ""
    changed_epoch: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "state": self.state.value,
            "reason": self.reason,
            "changed_epoch": self.changed_epoch,
        }


@dataclass(frozen=True)
class DecisionNode:
    decision_id: str
    question: str
    answer: str
    reason: str
    key_item_ids: tuple[str, ...]
    current_basis_ids: tuple[str, ...]
    would_flip_if: ConditionRef | None
    confidence: float | None
    alternatives: tuple[AlternativeNode, ...]
    gate_kind: str = "NONE"
    gating: bool = False
    parent_decision_ids: tuple[str, ...] = ()
    course_id: str = ""
    option: str = ""
    basis_state: BasisState = BasisState.SUPPORTED
    suspect_reason: str = ""
    suspect_item_id: str = ""
    created_epoch: int = 0
    supersedes_id: str | None = None
    basis_snapshot: tuple[tuple[str, ItemState], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "kind": "DECISION",
            "decision_id": self.decision_id,
            "question": self.question,
            "answer": self.answer,
            "option": self.option,
            "reason": self.reason,
            "key_item_ids": list(self.key_item_ids),
            "current_basis_ids": list(self.current_basis_ids),
            "would_flip_if": self.would_flip_if.to_dict() if self.would_flip_if else None,
            "confidence": self.confidence,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "gate_kind": self.gate_kind,
            "gating": self.gating,
            "parent_decision_ids": list(self.parent_decision_ids),
            "course_id": self.course_id,
            "basis_state": self.basis_state.value,
            "suspect_reason": self.suspect_reason,
            "suspect_item_id": self.suspect_item_id,
            "created_epoch": self.created_epoch,
            "supersedes_id": self.supersedes_id,
            "basis_snapshot": [
                {"item_id": item_id, "state": state.value}
                for item_id, state in self.basis_snapshot
            ],
        }


@dataclass(frozen=True)
class GraphEdge:
    """Selector-compatible edge projection from the canonical decision graph."""

    source_id: str
    relation: str
    target_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "relation": self.relation,
            "target_id": self.target_id,
        }


class DecisionGraph:
    """One bounded decision graph attached to the live dependency index."""

    def __init__(
        self,
        *,
        ledger: object | None = None,
        recall: object | None = None,
        micrograph: object | None = None,
    ) -> None:
        self._ledger: object | None = None
        self._recall = recall
        self._micrograph = micrograph
        self._decisions: dict[str, DecisionNode] = {}
        self._history: dict[str, tuple[DecisionNode, ...]] = {}
        self._items: dict[str, ItemRecord] = {}
        self._conditions: dict[str, bool | None] = {}
        self._edges: set[GraphEdge] = set()
        self._events: list[dict[str, object]] = []
        self._option_notes: list[dict[str, object]] = []
        self._pins: set[str] = set()
        self._dependents: dict[str, set[str]] = defaultdict(set)
        self._action_to_decision: dict[str, str] = {}
        self._course_current: dict[str, str] = {}
        self._departed: dict[str, set[str]] = defaultdict(set)
        self._transitions: list[dict[str, str]] = []
        self._revisit_counts: dict[str, int] = defaultdict(int)
        self._epoch = 0
        if ledger is not None:
            self.attach_ledger(ledger)
        if micrograph is not None:
            self.bind_micrograph(micrograph)

    @property
    def ledger(self) -> object | None:
        return self._ledger

    @property
    def graph_epoch(self) -> int:
        return self._epoch

    @property
    def decisions(self) -> tuple[DecisionNode, ...]:
        return tuple(self._decisions[key] for key in sorted(self._decisions))

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._events)

    @property
    def option_notes(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(note) for note in self._option_notes)

    @property
    def pins(self) -> tuple[str, ...]:
        return tuple(sorted(self._pins))

    @property
    def transitions(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(item) for item in self._transitions)

    @property
    def oscillation_count(self) -> int:
        seen: dict[str, set[str]] = defaultdict(set)
        count = 0
        for transition in self._transitions:
            course_id = transition["course_id"]
            source = transition["from"]
            target = transition["to"]
            if target in seen[course_id]:
                count += 1
            seen[course_id].update((source, target))
        return count

    @property
    def graph_nodes(self) -> tuple[Mapping[str, object], ...]:
        rows: list[Mapping[str, object]] = []
        for decision in self.decisions:
            rows.append(decision.to_dict())
            for index, alternative in enumerate(decision.alternatives):
                rows.append({
                    "schema": SCHEMA,
                    "kind": "ALTERNATIVE",
                    "node_id": f"{decision.decision_id}:alternative:{index}",
                    "parent_decision_id": decision.decision_id,
                    **alternative.to_dict(),
                })
        return tuple(rows)

    @property
    def graph_edges(self) -> tuple[GraphEdge, ...]:
        return tuple(sorted(self._edges, key=lambda edge: (edge.source_id, edge.relation, edge.target_id)))

    def attach_ledger(self, ledger: object) -> None:
        self._ledger = ledger
        attach = getattr(ledger, "attach_decision_graph", None)
        if callable(attach):
            attach(self)
        else:
            setattr(ledger, "_decision_graph", self)
        self.sync_ledger()

    def bind_micrograph(self, tracker: object) -> None:
        self._micrograph = tracker
        project = getattr(self._ledger, "project_micrograph", None)
        if callable(project):
            project(tracker)

    def attach_recall(self, recall: object) -> None:
        self._recall = recall

    def sync_ledger(self) -> None:
        """Project DEPGRAPH stale refs into decision health without new authority."""
        ledger = self._ledger
        if ledger is None:
            return
        dirty_refs = getattr(ledger, "dirty_evidence_refs", ())
        if isinstance(dirty_refs, Sequence):
            for ref in dirty_refs:
                self.mark_item(ref, state=ItemState.STALE, reason="DEPGRAPH evidence stale")
        if self._micrograph is not None:
            project = getattr(ledger, "project_micrograph", None)
            if callable(project):
                project(self._micrograph)

    def _event(self, kind: str, **fields: object) -> None:
        self._epoch += 1
        event = {"schema": SCHEMA, "seq": self._epoch, "kind": kind, **fields}
        self._events.append(event)
        if len(self._events) > MAX_EVENTS:
            del self._events[: len(self._events) - MAX_EVENTS]

    def _edge(self, source_id: str, relation: str, target_id: str) -> None:
        if source_id and target_id:
            self._edges.add(GraphEdge(source_id, relation, target_id))

    def _ensure_item(self, item_id: str) -> ItemRecord:
        current = self._items.get(item_id)
        if current is None:
            current = ItemRecord(item_id)
            self._items[item_id] = current
        return current

    def item_state(self, item_id: str) -> ItemState:
        return self._ensure_item(_text(item_id, "item_id", 96, required=True)).state

    def _normalise_alternatives(self, value: object) -> tuple[AlternativeNode, ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise DecisionGraphError("alternatives must be a sequence")
        if len(value) > MAX_ALTERNATIVES:
            raise DecisionGraphError("at most two alternatives are allowed")
        result: list[AlternativeNode] = []
        for raw in value:
            if isinstance(raw, AlternativeNode):
                alternative = raw
            elif isinstance(raw, Mapping):
                if "alternatives" in raw:
                    raise DecisionGraphError("alternatives cannot contain alternatives")
                alternative = AlternativeNode(
                    option=_text(raw.get("option"), "alternative.option", 96, required=True),
                    why_rejected=_text(
                        raw.get("why_rejected", ""),
                        "alternative.why_rejected",
                        MAX_ALTERNATIVE_CHARS,
                        required=True,
                    ),
                    evidence_ids=_ids(raw.get("evidence_ids", ()), "alternative.evidence_ids", 4),
                    would_choose_if=_condition(
                        raw.get("would_choose_if"),
                        "alternative.would_choose_if",
                        "ALTERNATIVE_VIABILITY",
                    ),
                )
            else:
                raise DecisionGraphError("alternative must be an object")
            if _word_count(alternative.why_rejected) > 15:
                raise DecisionGraphError("alternative.why_rejected exceeds 15 words")
            if any(item.option == alternative.option for item in result):
                raise DecisionGraphError("alternatives must have distinct options")
            result.append(alternative)
        return tuple(result)

    def decide(
        self,
        decision_id: str | None = None,
        *,
        question: str = "decision",
        answer: str,
        reason: str,
        key_item_ids: object = (),
        current_basis_ids: object | None = None,
        would_flip_if: object = None,
        confidence: object = None,
        alternatives: object = (),
        gate_kind: str = "NONE",
        gating: bool = False,
        parent_decision_ids: object = (),
        course_id: str = "",
        option: str | None = None,
        supersedes_id: str | None = None,
    ) -> dict[str, object]:
        ident = _text(decision_id, "decision_id", 96) if decision_id else f"d:{_digest((self._epoch, answer, reason))}"
        if ident in self._decisions:
            raise DecisionGraphError(f"decision already exists: {ident}")
        question_text = _text(question, "question", 96, required=True)
        answer_text = _text(answer, "answer", 96, required=True)
        reason_text = _text(reason, "reason", MAX_REASON_CHARS, required=True)
        if _word_count(reason_text) > MAX_REASON_WORDS:
            raise DecisionGraphError("reason exceeds 25 words")
        keys = _ids(key_item_ids, "key_item_ids", MAX_KEY_ITEMS)
        current_basis = _ids(
            key_item_ids if current_basis_ids is None else current_basis_ids,
            "current_basis_ids",
            MAX_KEY_ITEMS,
        )
        if not set(current_basis).issubset(set(keys)):
            raise DecisionGraphError("current_basis_ids must be key_item_ids")
        parents = _ids(parent_decision_ids, "parent_decision_ids", MAX_KEY_ITEMS)
        if ident in parents:
            raise DecisionGraphError("decision cannot depend on itself")
        flip = _condition(would_flip_if, "would_flip_if", "CURRENT_FALSIFIER")
        gate_kind_text = _text(gate_kind, "gate_kind", 48).upper() or "NONE"
        gate = bool(gating or gate_kind_text != "NONE")
        course = _text(course_id, "course_id", 96) or ident
        selected_option = _text(option, "option", 96) if option is not None else answer_text
        if not selected_option:
            raise DecisionGraphError("option is required")
        confidence_value = _safe_float(confidence, "confidence")
        alternative_nodes = self._normalise_alternatives(alternatives)
        supersedes_ident = _text(supersedes_id, "supersedes_id", 96) if supersedes_id else None
        superseded = self._decisions.get(supersedes_ident) if supersedes_ident else None
        if supersedes_ident and superseded is None:
            raise DecisionGraphError(f"unknown superseded decision: {supersedes_ident}")
        if superseded is not None and superseded.course_id != course:
            raise DecisionGraphError("superseding decision must stay in the same course")
        for parent in parents:
            if parent not in self._decisions:
                raise DecisionGraphError(f"unknown parent decision: {parent}")
        snapshot = tuple((item_id, self.item_state(item_id)) for item_id in keys)
        broken_basis = next(
            ((item_id, state) for item_id, state in snapshot
             if item_id in current_basis and state is not ItemState.CURRENT),
            None,
        )
        node = DecisionNode(
            decision_id=ident,
            question=question_text,
            answer=answer_text,
            reason=reason_text,
            key_item_ids=keys,
            current_basis_ids=current_basis,
            would_flip_if=flip,
            confidence=confidence_value,
            alternatives=alternative_nodes,
            gate_kind=gate_kind_text,
            gating=gate,
            parent_decision_ids=parents,
            course_id=course,
            option=selected_option,
            basis_state=BasisState.SUSPECT if broken_basis else BasisState.SUPPORTED,
            suspect_reason=(
                f"{broken_basis[0]} is {broken_basis[1].value.lower()} at decision time"
                if broken_basis else ""
            ),
            suspect_item_id=broken_basis[0] if broken_basis else "",
            created_epoch=self._epoch + 1,
            supersedes_id=supersedes_ident,
            basis_snapshot=snapshot,
        )
        self._decisions[ident] = node
        self._history.setdefault(ident, tuple())
        if superseded is not None:
            self._course_current[course] = selected_option
        else:
            self._course_current.setdefault(course, selected_option)
        if broken_basis:
            self._pins.add(ident)
        for item_id in keys:
            self._edge(ident, "relies_on", item_id)
        for parent in parents:
            self._dependents[parent].add(ident)
            self._edge(ident, "decision_depends_on", parent)
        if flip is not None:
            self._edge(ident, "watches", f"condition:{flip.predicate_id}")
        for index, alternative in enumerate(alternative_nodes):
            alternative_id = f"{ident}:alternative:{index}"
            self._edge(ident, "has_alternative", alternative_id)
            for evidence_id in alternative.evidence_ids:
                self._edge(alternative_id, "rejected_because", evidence_id)
            if alternative.would_choose_if is not None:
                self._edge(alternative_id, "watches", f"condition:{alternative.would_choose_if.predicate_id}")
        self._event(
            "DECISION_RECORDED",
            decision_id=ident,
            answer=answer_text,
            gating=gate,
            basis_state=node.basis_state.value,
        )
        if broken_basis:
            self._event(
                NoticeKind.BASIS_BROKEN.value,
                decision_id=ident,
                item_id=broken_basis[0],
                reason=node.suspect_reason,
            )
        self._publish_recall(node)
        return {"schema": SCHEMA, "action": ActionKind.DECIDE.value, "status": "RECORDED", "decision": node.to_dict()}

    def record_decision(self, *args: object, **kwargs: object) -> dict[str, object]:
        return self.decide(*args, **kwargs)  # type: ignore[arg-type]

    def _publish_recall(self, node: DecisionNode) -> None:
        append = getattr(self._recall, "append", None)
        if not callable(append):
            return
        payload = {
            "decision_id": node.decision_id,
            "answer": node.answer,
            "reason": node.reason,
            "key_item_ids": list(node.key_item_ids),
            "basis_state": node.basis_state.value,
            "links": {"hypotheses": list(node.key_item_ids)},
        }
        try:
            append(
                "decision",
                payload,
                item_id=f"decision-{_digest(node.decision_id)}",
                source_key=f"decision:{node.decision_id}",
                gist=f"{node.decision_id}: {node.answer} because {node.reason}",
                status=node.basis_state.value,
            )
        except (TypeError, ValueError):
            # A compatibility recall adapter cannot become a second authority.
            self._event("RECALL_PROJECTION_SKIPPED", decision_id=node.decision_id)

    def _suspect(self, decision_id: str, *, reason: str, item_id: str, transitive: bool = False) -> None:
        node = self._decisions.get(decision_id)
        if node is None:
            return
        if node.basis_state is not BasisState.SUSPECT:
            updated = replace(
                node,
                basis_state=BasisState.SUSPECT,
                suspect_reason=_text(reason, "suspect_reason", MAX_REASON_CHARS),
                suspect_item_id=item_id,
            )
            self._history[decision_id] = self._history.get(decision_id, ()) + (node,)
            self._decisions[decision_id] = updated
            self._pins.add(decision_id)
            kind = NoticeKind.SUSPECT_TRANSITIVE.value if transitive else NoticeKind.BASIS_BROKEN.value
            self._event(kind, decision_id=decision_id, item_id=item_id, reason=reason)
        for dependent in sorted(self._dependents.get(decision_id, ())):
            self._suspect(dependent, reason=f"{decision_id} is SUSPECT", item_id=decision_id, transitive=True)

    def mark_item(self, item_id: object, *, state: ItemState | str = ItemState.STALE, reason: str = "") -> dict[str, object]:
        ident = _text(item_id, "item_id", 96, required=True)
        try:
            item_state = state if isinstance(state, ItemState) else ItemState(str(state).upper())
        except ValueError as exc:
            raise DecisionGraphError("unsupported item state") from exc
        previous = self._ensure_item(ident)
        if previous.state is item_state and previous.reason == reason:
            return {"schema": SCHEMA, "status": "UNCHANGED", "item_id": ident, "state": item_state.value}
        self._epoch += 1
        self._items[ident] = ItemRecord(ident, item_state, _text(reason, "reason", 160), self._epoch)
        if item_state is not ItemState.CURRENT:
            for decision in tuple(self.decisions):
                if ident in decision.current_basis_ids:
                    self._suspect(
                        decision.decision_id,
                        reason=f"{ident} is {item_state.value.lower()}: {reason or 'support changed'}",
                        item_id=ident,
                    )
                for alternative in decision.alternatives:
                    if ident in alternative.evidence_ids:
                        self._record_option_note(
                            decision,
                            alternative,
                            NoticeKind.ALTERNATIVE_REJECTION_UNSUPPORTED.value,
                            f"{ident} is no longer current",
                        )
        self._event("ITEM_STATUS_CHANGED", item_id=ident, state=item_state.value, reason=reason)
        return {"schema": SCHEMA, "status": "UPDATED", "item_id": ident, "state": item_state.value}

    def invalidate(self, item_id: object, *, reason: str = "evidence stale") -> dict[str, object]:
        return self.mark_item(item_id, state=ItemState.STALE, reason=reason)

    def set_condition(self, predicate_id: object, value: bool | None) -> dict[str, object]:
        ident = _text(predicate_id, "predicate_id", 96, required=True)
        if value is not None and not isinstance(value, bool):
            raise DecisionGraphError("condition value must be bool or None")
        self._conditions[ident] = value
        self._event("CONDITION_CHANGED", predicate_id=ident, value=value)
        if value is True:
            for decision in tuple(self.decisions):
                if decision.would_flip_if and decision.would_flip_if.predicate_id == ident:
                    self._suspect(
                        decision.decision_id,
                        reason=f"flip condition {ident} became true",
                        item_id=f"condition:{ident}",
                    )
                for alternative in decision.alternatives:
                    if alternative.would_choose_if and alternative.would_choose_if.predicate_id == ident:
                        self._record_option_note(
                            decision,
                            alternative,
                            NoticeKind.ALTERNATIVE_NEWLY_VIABLE.value,
                            f"{ident} became true; incumbent support remains unchanged",
                        )
        return {"schema": SCHEMA, "status": "UPDATED", "predicate_id": ident, "value": value}

    def _record_option_note(
        self,
        decision: DecisionNode,
        alternative: AlternativeNode,
        kind: str,
        reason: str,
    ) -> None:
        signature = (decision.decision_id, alternative.option, kind)
        if any((note.get("decision_id"), note.get("option"), note.get("kind")) == signature for note in self._option_notes):
            return
        note = {
            "schema": SCHEMA,
            "kind": kind,
            "decision_id": decision.decision_id,
            "option": alternative.option,
            "reason": reason,
            "priority": "LOW",
            "surfaced": False,
            "pinned": False,
        }
        self._option_notes.append(note)
        self._event(kind, decision_id=decision.decision_id, option=alternative.option, reason=reason)

    def checkpoint(
        self,
        decision_id: str,
        *,
        stalled: bool = False,
        natural_checkpoint: bool = False,
        progress_steps: int = 0,
    ) -> dict[str, object]:
        node = self._decisions.get(_text(decision_id, "decision_id", 96, required=True))
        if node is None:
            raise DecisionGraphError(f"unknown decision: {decision_id}")
        due = bool(stalled or natural_checkpoint)
        notes: list[dict[str, object]] = []
        if due and node.basis_state is BasisState.SUPPORTED:
            for note in self._option_notes:
                if note.get("decision_id") == node.decision_id:
                    note["surfaced"] = True
                    notes.append(dict(note))
        self._event("CHECKPOINT_REVIEW", decision_id=node.decision_id, stalled=stalled, progress_steps=progress_steps)
        return {
            "schema": SCHEMA,
            "action": ActionKind.CHECKPOINT.value,
            "status": "ALTERNATIVES_SURFACED" if notes else "NO_SWITCH",
            "decision_id": node.decision_id,
            "notes": notes,
            "switch_required": False,
        }

    @staticmethod
    def _loss(value: object, name: str) -> float:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise DecisionGraphError(f"{name} must be a non-negative finite number")
        return float(value)

    def switch_option(
        self,
        decision_id: str,
        challenger: str,
        *,
        stalled: bool = False,
        natural_checkpoint: bool = False,
        current_loss_low: object,
        challenger_loss_high: object,
        switching_cost_high: object,
        margin_absolute: object = 15.0,
        margin_relative: object = 0.10,
        admitted: bool = True,
        challenger_supported: bool = True,
    ) -> dict[str, object]:
        ident = _text(decision_id, "decision_id", 96, required=True)
        node = self._decisions.get(ident)
        if node is None:
            raise DecisionGraphError(f"unknown decision: {ident}")
        target = _text(challenger, "challenger", 96, required=True)
        current = self._course_current.get(node.course_id, node.option)
        review_due = bool(stalled or natural_checkpoint)
        if node.basis_state is not BasisState.SUPPORTED:
            return self._switch_receipt("REVISIT_REQUIRED", ident, current, target, "current basis is SUSPECT")
        if not review_due:
            return self._switch_receipt("KEEP", ident, current, target, "review not due; viability is only an option note")
        if not admitted or not challenger_supported:
            return self._switch_receipt("KEEP", ident, current, target, "challenger is not admitted and supported")
        low = self._loss(current_loss_low, "current_loss_low")
        high = self._loss(challenger_loss_high, "challenger_loss_high")
        cost = self._loss(switching_cost_high, "switching_cost_high")
        absolute = self._loss(margin_absolute, "margin_absolute")
        relative = self._loss(margin_relative, "margin_relative")
        if relative > 1.0:
            raise DecisionGraphError("margin_relative must be at most 1")
        gain = low - high - cost
        margin = max(absolute, relative * low)
        if gain <= margin:
            return self._switch_receipt("KEEP", ident, current, target, "hysteresis margin not met", gain=gain, margin=margin)
        if target in self._departed[node.course_id]:
            self._event(NoticeKind.OSCILLATION_GUARD.value, decision_id=ident, current=current, challenger=target)
            return self._switch_receipt("OSCILLATION_GUARD", ident, current, target, "destination already departed", gain=gain, margin=margin)
        self._departed[node.course_id].add(current)
        self._course_current[node.course_id] = target
        transition = {"course_id": node.course_id, "from": current, "to": target}
        self._transitions.append(transition)
        self._event(NoticeKind.SWITCH_COMMITTED.value, decision_id=ident, **transition, gain=gain, margin=margin)
        return self._switch_receipt("SWITCH", ident, current, target, "comparative margin exceeded", gain=gain, margin=margin)

    def _switch_receipt(self, status: str, decision_id: str, current: str, target: str, reason: str, **fields: object) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "action": "switch",
            "status": status,
            "decision_id": decision_id,
            "current_option": current,
            "challenger": target,
            "reason": reason,
            "switched": status == "SWITCH",
            **fields,
        }

    def why(self, decision_id: str, *, action_id: str | None = None) -> dict[str, object]:
        ident = _text(decision_id, "decision_id", 96, required=True)
        if action_id:
            ident = self._action_to_decision.get(action_id, ident)
        node = self._decisions.get(ident)
        if node is None:
            raise DecisionGraphError(f"unknown decision: {ident}")
        basis = []
        for item_id, state_at_decision in node.basis_snapshot:
            current_state = self.item_state(item_id)
            basis.append({
                "item_id": item_id,
                "state_at_decision": state_at_decision.value,
                "current_state": current_state.value,
                "changed": state_at_decision is not current_state,
            })
        changes = [
            dict(event)
            for event in self._events
            if int(event.get("seq", 0)) > node.created_epoch
            and (event.get("decision_id") == ident or event.get("item_id") in node.key_item_ids)
        ]
        return {
            "schema": SCHEMA,
            "action": ActionKind.WHY.value,
            "read_only": True,
            "decision_id": ident,
            "decision": node.to_dict(),
            "reason": node.reason,
            "key_items": basis,
            "alternatives": [item.to_dict() for item in node.alternatives],
            "status": node.basis_state.value,
            "pins": [ident] if ident in self._pins else [],
            "changed_since": changes[-32:],
            "graph_epoch": self._epoch,
        }

    def register_action(self, action_id: str, decision_id: str) -> dict[str, object]:
        action = _text(action_id, "action_id", 96, required=True)
        ident = _text(decision_id, "decision_id", 96, required=True)
        if ident not in self._decisions:
            raise DecisionGraphError(f"unknown decision: {ident}")
        self._action_to_decision[action] = ident
        self._edge(action, "governed_by", ident)
        self._event("ACTION_GOVERNED", action_id=action, decision_id=ident)
        return {"schema": SCHEMA, "action": ActionKind.GATE.value, "status": "REGISTERED", "action_id": action, "decision_id": ident}

    def action_gate(self, action: object, *, decision_id: str | None = None) -> dict[str, object]:
        action_name = action.get("kind", action.get("action", "")) if isinstance(action, Mapping) else action
        action_text = _text(action_name, "action", 64).lower()
        ident = decision_id
        if isinstance(action, Mapping) and ident is None:
            raw_id = action.get("decision_id", action.get("decision"))
            ident = _text(raw_id, "decision_id", 96) if raw_id else None
        superseded = {
            node.supersedes_id
            for node in self.decisions
            if node.supersedes_id
        }
        candidates = [self._decisions.get(ident)] if ident else [
            node for node in self.decisions
            if node.gating and node.decision_id not in superseded
        ]
        candidates = [node for node in candidates if node is not None]
        blocked = next((node for node in candidates if node.basis_state is not BasisState.SUPPORTED), None)
        gate_action = action_text in {"done", "certify", "stage", "advance", "submit", "complete"}
        if blocked is not None and (blocked.gating or gate_action):
            return {
                "schema": SCHEMA,
                "action": ActionKind.GATE.value,
                "status": "BLOCKED",
                "reason_code": "SUSPECT_DECISION",
                "decision_id": blocked.decision_id,
                "message": f"{blocked.decision_id} is SUSPECT; revisit before {action_text or 'action'}",
                "revisit": {"action": ActionKind.REVISIT.value, "decision_id": blocked.decision_id},
            }
        return {"schema": SCHEMA, "action": ActionKind.GATE.value, "status": "ALLOWED", "decision_id": ident}

    def revisit(
        self,
        decision_id: str,
        *,
        answer: str | None = None,
        reason: str | None = None,
        key_item_ids: object | None = None,
        alternatives: object = (),
        would_flip_if: object = None,
        confidence: object = None,
    ) -> dict[str, object]:
        ident = _text(decision_id, "decision_id", 96, required=True)
        old = self._decisions.get(ident)
        if old is None:
            raise DecisionGraphError(f"unknown decision: {ident}")
        if answer is None:
            options = [old.option, *(item.option for item in old.alternatives)]
            return {
                "schema": SCHEMA,
                "action": ActionKind.REVISIT.value,
                "status": "REDECISION_REQUIRED",
                "decision_id": ident,
                "basis_state": old.basis_state.value,
                "proposed_options": tuple(dict.fromkeys(options)),
                "fair_comparison": True,
                "pinned": ident in self._pins,
            }
        if key_item_ids is None:
            raise DecisionGraphError("revisit requires fresh key_item_ids")
        self._revisit_counts[ident] += 1
        result = self.decide(
            f"{ident}:r{self._revisit_counts[ident]}",
            question=old.question,
            answer=answer,
            reason=reason or "fresh evidence requires a new decision",
            key_item_ids=key_item_ids,
            current_basis_ids=key_item_ids,
            would_flip_if=would_flip_if,
            confidence=confidence,
            alternatives=alternatives,
            gate_kind=old.gate_kind,
            gating=old.gating,
            parent_decision_ids=old.parent_decision_ids,
            course_id=old.course_id,
            option=answer,
            supersedes_id=ident,
        )
        replacement = result["decision"]
        replacement_id = replacement.get("decision_id") if isinstance(replacement, Mapping) else ""
        self._event("DECISION_REVISITED", decision_id=ident, replacement=replacement_id)
        return {
            "schema": SCHEMA,
            "action": ActionKind.REVISIT.value,
            "status": "REDECIDED",
            "supersedes_id": ident,
            "decision": result["decision"],
            "fair_comparison": True,
        }

    def typed_action(self, kind: str, decision_id: str, **payload: object) -> dict[str, object]:
        action = _text(kind, "action kind", 32, required=True).lower()
        if action == ActionKind.WHY.value:
            return self.why(decision_id)
        if action == ActionKind.REVISIT.value:
            return self.revisit(decision_id, **payload)  # type: ignore[arg-type]
        raise DecisionGraphError(f"unsupported decision graph action: {kind}")

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "graph_epoch": self._epoch,
            "decision_count": len(self._decisions),
            "item_count": len(self._items),
            "suspect_decisions": list(self.pins),
            "option_notes": len(self._option_notes),
            "oscillation_count": self.oscillation_count,
        }


__all__ = [
    "ActionKind",
    "AlternativeNode",
    "BasisState",
    "ConditionRef",
    "DecisionGraph",
    "DecisionGraphError",
    "DecisionNode",
    "GraphEdge",
    "ItemRecord",
    "ItemState",
    "NoticeKind",
    "SCHEMA",
]
