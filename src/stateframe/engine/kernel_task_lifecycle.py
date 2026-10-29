"""Pure lifecycle/task-state contract for the ORPHAN_C4 lane.

This module is an adapter and evaluator, not a second runtime owner.  A host
supplies a snapshot from its authoritative continuation, requirements,
evidence, and submission owners.  The functions here validate that snapshot,
derive one consistent set of lifecycle predicates, and describe the impact of
a request/source change for the host to commit.

The source binding deliberately accepts the shared ``mc-outline/1`` result as
opaque structured data.  It does not parse Python (or any other language), so
the same contract covers Go, Rust, TypeScript/JavaScript, Java, PHP, Vue, and
other languages supported by OUTLINEGENERIC.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


SCHEMA = "stateframe-task-lifecycle/1"
POLICY_VERSION = "task-lifecycle-unification/1"
OUTLINE_SCHEMA = "mc-outline/1"


class LifecycleError(ValueError):
    """Raised when a host snapshot or transition violates this contract."""


class Phase(str, Enum):
    PLANNED = "planned"
    ACTIVE = "active"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


class NodeKind(str, Enum):
    GOAL = "goal"
    WORKSTREAM = "workstream"
    INTEGRATION = "integration"
    SOURCE_DETOUR = "source_detour"


class EvidenceState(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


SATISFIED = EvidenceState.SATISFIED.value
VIOLATED = EvidenceState.VIOLATED.value
UNKNOWN = EvidenceState.UNKNOWN.value


def _text(value: object, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise LifecycleError(f"{name} is required")
    return result


def _optional_text(value: object, name: str) -> Optional[str]:
    if value in (None, ""):
        return None
    return _text(value, name)


def _refs(value: object, name: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = tuple(value)
    else:
        values = ()
    result = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if any(not item for item in result):
        raise LifecycleError(f"{name} contains an empty reference")
    return result


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class SourceBinding:
    """A revision-bound result from the shared language-neutral outline API."""

    path: str
    language: str
    revision: str
    outline_digest: str
    schema: str = OUTLINE_SCHEMA

    def __post_init__(self) -> None:
        path = _text(self.path, "source path").replace("\\", "/")
        language = _text(self.language, "source language").lower()
        revision = _text(self.revision, "source revision")
        digest = _text(self.outline_digest, "outline digest")
        schema = _text(self.schema, "outline schema")
        if schema != OUTLINE_SCHEMA:
            raise LifecycleError(f"unsupported outline schema: {schema}")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "outline_digest", digest)
        object.__setattr__(self, "schema", schema)

    @classmethod
    def from_outline(cls, outline: Mapping[str, object], *, revision: object) -> "SourceBinding":
        """Bind an ``outline_source`` result without language-specific parsing."""

        if not isinstance(outline, Mapping):
            raise LifecycleError("outline must be a mapping")
        schema = outline.get("schema")
        if schema != OUTLINE_SCHEMA:
            raise LifecycleError(f"outline must use {OUTLINE_SCHEMA}")
        path = outline.get("file") or outline.get("path")
        language = outline.get("language")
        digest = outline.get("digest") or _digest(outline)
        return cls(path=str(path or ""), language=str(language or ""),
                   revision=_text(revision, "source revision"),
                   outline_digest=str(digest), schema=str(schema))

    def to_dict(self) -> Dict[str, object]:
        return {"schema": self.schema, "path": self.path, "language": self.language,
                "revision": self.revision, "outline_digest": self.outline_digest}


@dataclass(frozen=True)
class TaskNode:
    """One plan node; lifecycle is host-owned and never inferred from evidence."""

    task_id: str
    goal_id: str
    kind: NodeKind
    contract_revision: str
    parent_id: Optional[str] = None
    generation: int = 1
    phase: Phase = Phase.PLANNED
    requirement_refs: Tuple[str, ...] = ()
    evidence_refs: Tuple[str, ...] = ()
    source_bindings: Tuple[SourceBinding, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        task_id = _text(self.task_id, "task_id")
        goal_id = _text(self.goal_id, "goal_id")
        try:
            kind = NodeKind(self.kind)
            phase = Phase(self.phase)
        except ValueError as error:
            raise LifecycleError(str(error)) from error
        contract = _text(self.contract_revision, "contract_revision")
        parent = _optional_text(self.parent_id, "parent_id")
        if int(self.generation) < 1:
            raise LifecycleError("generation must be positive")
        requirements = _refs(self.requirement_refs, "requirement_refs")
        evidence = _refs(self.evidence_refs, "evidence_refs")
        bindings = tuple(self.source_bindings)
        if len({item.path for item in bindings}) != len(bindings):
            raise LifecycleError(f"duplicate source binding for task {task_id}")
        if kind is NodeKind.GOAL and parent is not None:
            raise LifecycleError("goal node cannot have a parent")
        if kind is not NodeKind.GOAL and parent is None:
            raise LifecycleError(f"{kind.value} node requires a parent")
        if kind is not NodeKind.GOAL and kind is not NodeKind.SOURCE_DETOUR and not requirements:
            raise LifecycleError(f"required {kind.value} must own a requirement")
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "contract_revision", contract)
        object.__setattr__(self, "parent_id", parent)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "requirement_refs", requirements)
        object.__setattr__(self, "evidence_refs", evidence)
        object.__setattr__(self, "source_bindings", bindings)
        object.__setattr__(self, "required", bool(self.required))

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id, "goal_id": self.goal_id, "kind": self.kind.value,
            "contract_revision": self.contract_revision, "parent_id": self.parent_id,
            "generation": self.generation, "phase": self.phase.value,
            "requirement_refs": list(self.requirement_refs), "evidence_refs": list(self.evidence_refs),
            "source_bindings": [item.to_dict() for item in self.source_bindings],
            "required": self.required,
        }


@dataclass(frozen=True)
class DependencyEdge:
    producer_id: str
    consumer_id: str
    relation: str = "requires_acceptance"

    def __post_init__(self) -> None:
        producer = _text(self.producer_id, "dependency producer")
        consumer = _text(self.consumer_id, "dependency consumer")
        relation = _text(self.relation, "dependency relation")
        if producer == consumer:
            raise LifecycleError("a task cannot depend on itself")
        object.__setattr__(self, "producer_id", producer)
        object.__setattr__(self, "consumer_id", consumer)
        object.__setattr__(self, "relation", relation)

    def to_dict(self) -> Dict[str, str]:
        return {"producer_id": self.producer_id, "consumer_id": self.consumer_id,
                "relation": self.relation}


@dataclass(frozen=True)
class LifecyclePlan:
    goal_id: str
    nodes: Tuple[TaskNode, ...]
    edges: Tuple[DependencyEdge, ...] = ()
    revision: int = 1

    def __post_init__(self) -> None:
        goal_id = _text(self.goal_id, "plan goal_id")
        nodes = tuple(self.nodes)
        if int(self.revision) < 1:
            raise LifecycleError("plan revision must be positive")
        by_id = {node.task_id: node for node in nodes}
        if len(by_id) != len(nodes):
            raise LifecycleError("task IDs must be unique")
        goals = tuple(node for node in nodes if node.kind is NodeKind.GOAL)
        if len(goals) != 1 or goals[0].task_id != goal_id:
            raise LifecycleError("plan must contain exactly one goal matching goal_id")
        if any(node.goal_id != goal_id for node in nodes):
            raise LifecycleError("every node must belong to the plan goal")
        for node in nodes:
            if node.parent_id is not None and node.parent_id not in by_id:
                raise LifecycleError(f"unknown parent for {node.task_id}: {node.parent_id}")
            if node.parent_id == node.task_id:
                raise LifecycleError(f"task cannot parent itself: {node.task_id}")
        edges = tuple(self.edges)
        for edge in edges:
            if edge.producer_id not in by_id or edge.consumer_id not in by_id:
                raise LifecycleError("dependency edge references an unknown task")
        self._assert_acyclic(edges, "dependency")
        parent_edges = tuple(
            DependencyEdge(node.parent_id, node.task_id, "parent")
            for node in nodes if node.parent_id is not None
        )
        self._assert_acyclic(parent_edges, "parent")
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "revision", int(self.revision))

    @staticmethod
    def _assert_acyclic(edges: Sequence[DependencyEdge], label: str) -> None:
        adjacency: Dict[str, List[str]] = {}
        for edge in edges:
            adjacency.setdefault(edge.producer_id, []).append(edge.consumer_id)
        visiting: Set[str] = set()
        visited: Set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise LifecycleError(f"{label} graph contains a cycle at {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for child in adjacency.get(node_id, ()):
                visit(child)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in tuple(adjacency):
            visit(node_id)

    def node(self, task_id: str) -> TaskNode:
        ident = _text(task_id, "task_id")
        for node in self.nodes:
            if node.task_id == ident:
                return node
        raise LifecycleError(f"unknown task: {ident}")

    def incoming(self, task_id: str) -> Tuple[DependencyEdge, ...]:
        return tuple(edge for edge in self.edges if edge.consumer_id == task_id)

    def replace_node(self, node: TaskNode) -> "LifecyclePlan":
        self.node(node.task_id)
        return replace(self, nodes=tuple(node if item.task_id == node.task_id else item
                                         for item in self.nodes), revision=self.revision + 1)

    def to_dict(self) -> Dict[str, object]:
        return {"schema": SCHEMA, "goal_id": self.goal_id, "revision": self.revision,
                "nodes": [item.to_dict() for item in self.nodes],
                "edges": [item.to_dict() for item in self.edges]}


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    task_id: str
    generation: int
    contract_revision: str
    state: EvidenceState
    producer: str
    source_refs: Tuple[str, ...]
    source_bindings: Tuple[SourceBinding, ...] = ()

    def __post_init__(self) -> None:
        evidence_id = _text(self.evidence_id, "evidence_id")
        task_id = _text(self.task_id, "evidence task_id")
        contract = _text(self.contract_revision, "evidence contract_revision")
        producer = _text(self.producer, "evidence producer")
        try:
            state = EvidenceState(self.state)
        except ValueError as error:
            raise LifecycleError(f"evidence state must be SATISFIED, VIOLATED, or UNKNOWN: {self.state}") from error
        source_refs = _refs(self.source_refs, "source_refs")
        if not source_refs:
            raise LifecycleError("evidence must retain at least one source reference")
        bindings = tuple(self.source_bindings)
        if len({item.path for item in bindings}) != len(bindings):
            raise LifecycleError(f"duplicate evidence source binding: {evidence_id}")
        if int(self.generation) < 1:
            raise LifecycleError("evidence generation must be positive")
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "contract_revision", contract)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "source_bindings", bindings)

    def binds(self, node: TaskNode) -> bool:
        if self.task_id != node.task_id or self.generation != node.generation:
            return False
        if self.contract_revision != node.contract_revision:
            return False
        actual = {item.path: item for item in self.source_bindings}
        return all(actual.get(item.path) == item for item in node.source_bindings)

    def to_dict(self) -> Dict[str, object]:
        return {"evidence_id": self.evidence_id, "task_id": self.task_id,
                "generation": self.generation, "contract_revision": self.contract_revision,
                "state": self.state.value, "producer": self.producer,
                "source_refs": list(self.source_refs),
                "source_bindings": [item.to_dict() for item in self.source_bindings]}


@dataclass(frozen=True)
class SubmissionReceipt:
    receipt_id: str
    task_id: str
    generation: int
    contract_revision: str
    status: str
    artifact_ref: str

    def __post_init__(self) -> None:
        receipt = _text(self.receipt_id, "submission receipt_id")
        task_id = _text(self.task_id, "submission task_id")
        contract = _text(self.contract_revision, "submission contract_revision")
        status = _text(self.status, "submission status").upper()
        if status not in {"ACCEPTED", "REJECTED", "UNKNOWN"}:
            raise LifecycleError(f"unknown submission status: {status}")
        artifact = _text(self.artifact_ref, "submission artifact_ref")
        if int(self.generation) < 1:
            raise LifecycleError("submission generation must be positive")
        object.__setattr__(self, "receipt_id", receipt)
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "contract_revision", contract)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "artifact_ref", artifact)

    def binds(self, node: TaskNode) -> bool:
        return (self.task_id == node.task_id and self.generation == node.generation
                and self.contract_revision == node.contract_revision
                and self.status == "ACCEPTED")

    def to_dict(self) -> Dict[str, object]:
        return {"receipt_id": self.receipt_id, "task_id": self.task_id,
                "generation": self.generation, "contract_revision": self.contract_revision,
                "status": self.status, "artifact_ref": self.artifact_ref}


@dataclass(frozen=True)
class TaskPredicates:
    task_id: str
    generation: int
    phase: Phase
    completed: bool
    submitted: bool
    correct: bool
    dependencies_satisfied: bool
    accepted: bool
    state: str
    reasons: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {"task_id": self.task_id, "generation": self.generation,
                "phase": self.phase.value, "completed": self.completed,
                "submitted": self.submitted, "correct": self.correct,
                "dependencies_satisfied": self.dependencies_satisfied,
                "accepted": self.accepted, "state": self.state,
                "reasons": list(self.reasons)}


@dataclass(frozen=True)
class PlanReadiness:
    goal_id: str
    ready: bool
    accepted_task_ids: Tuple[str, ...]
    blockers: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return {"goal_id": self.goal_id, "ready": self.ready,
                "accepted_task_ids": list(self.accepted_task_ids),
                "blockers": list(self.blockers)}


@dataclass(frozen=True)
class LifecycleSnapshot:
    plan: LifecyclePlan
    evidence: Tuple[EvidenceRecord, ...] = ()
    submissions: Tuple[SubmissionReceipt, ...] = ()
    request_revision: str = "request:1"
    sequence: int = 0
    applied_events: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        request = _text(self.request_revision, "request_revision")
        if int(self.sequence) < 0:
            raise LifecycleError("snapshot sequence cannot be negative")
        evidence = tuple(self.evidence)
        submissions = tuple(self.submissions)
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise LifecycleError("evidence IDs must be unique")
        if len({item.receipt_id for item in submissions}) != len(submissions):
            raise LifecycleError("submission receipt IDs must be unique")
        events = tuple((str(event_id), str(digest)) for event_id, digest in self.applied_events)
        if len({event_id for event_id, _digest_value in events}) != len(events):
            raise LifecycleError("applied event IDs must be unique")
        for record in (*evidence, *submissions):
            if record.task_id not in {node.task_id for node in self.plan.nodes}:
                raise LifecycleError(f"record references unknown task: {record.task_id}")
        object.__setattr__(self, "request_revision", request)
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "submissions", submissions)
        object.__setattr__(self, "applied_events", events)

    def _evidence_by_id(self) -> Dict[str, EvidenceRecord]:
        return {item.evidence_id: item for item in self.evidence}

    def _submissions_for(self, task_id: str) -> Tuple[SubmissionReceipt, ...]:
        return tuple(item for item in self.submissions if item.task_id == task_id)

    def predicates(self, task_id: str) -> TaskPredicates:
        memo: Dict[str, TaskPredicates] = {}

        def evaluate(ident: str, stack: Set[str]) -> TaskPredicates:
            if ident in memo:
                return memo[ident]
            if ident in stack:
                raise LifecycleError(f"dependency cycle while evaluating {ident}")
            node = self.plan.node(ident)
            stack.add(ident)
            records = self._evidence_by_id()
            reasons: List[str] = []
            correct = bool(node.evidence_refs)
            for evidence_id in node.evidence_refs:
                record = records.get(evidence_id)
                if record is None:
                    correct = False
                    reasons.append(f"missing_evidence:{evidence_id}")
                elif not record.binds(node):
                    correct = False
                    reasons.append(f"stale_evidence:{evidence_id}")
                elif record.state is not EvidenceState.SATISFIED:
                    correct = False
                    reasons.append(f"evidence_{record.state.value.lower()}:{evidence_id}")
            incoming = self.plan.incoming(ident)
            dependencies_satisfied = True
            for edge in incoming:
                dependency = evaluate(edge.producer_id, stack)
                if not dependency.accepted:
                    dependencies_satisfied = False
                    reasons.append(f"dependency_not_accepted:{edge.producer_id}")
            stack.remove(ident)
            completed = node.phase is Phase.COMPLETED
            submitted = any(item.binds(node) for item in self._submissions_for(ident))
            if not submitted:
                reasons.append("submission_missing")
            if node.phase in {Phase.CANCELLED, Phase.SUPERSEDED}:
                reasons.append(f"phase_{node.phase.value}")
            accepted = completed and submitted and correct and dependencies_satisfied
            if not completed:
                reasons.append("work_not_completed")
            state = node.phase.value
            if accepted:
                state = "accepted"
            elif submitted and correct:
                state = "submitted_correct_unaccepted"
            elif submitted:
                state = "submitted_unverified"
            elif correct:
                state = "correct_unsubmitted"
            memo[ident] = TaskPredicates(
                task_id=ident, generation=node.generation, phase=node.phase,
                completed=completed, submitted=submitted, correct=correct,
                dependencies_satisfied=dependencies_satisfied, accepted=accepted,
                state=state, reasons=tuple(dict.fromkeys(reasons)),
            )
            return memo[ident]

        return evaluate(_text(task_id, "task_id"), set())

    def readiness(self) -> PlanReadiness:
        accepted: List[str] = []
        blockers: List[str] = []
        for node in self.plan.nodes:
            if node.kind is NodeKind.GOAL or not node.required:
                continue
            result = self.predicates(node.task_id)
            if result.accepted:
                accepted.append(node.task_id)
            else:
                blockers.extend(f"{node.task_id}:{reason}" for reason in result.reasons)
        return PlanReadiness(self.plan.goal_id, not blockers, tuple(accepted),
                             tuple(dict.fromkeys(blockers)))

    def to_dict(self) -> Dict[str, object]:
        return {"schema": SCHEMA, "policy_version": POLICY_VERSION,
                "request_revision": self.request_revision, "sequence": self.sequence,
                "plan": self.plan.to_dict(), "evidence": [item.to_dict() for item in self.evidence],
                "submissions": [item.to_dict() for item in self.submissions],
                "applied_events": [list(item) for item in self.applied_events],
                "readiness": self.readiness().to_dict()}


@dataclass(frozen=True)
class TaskTransition:
    event_id: str
    task_id: str
    expected_generation: int
    to_phase: Phase
    expected_plan_revision: Optional[int] = None
    replacement_task_ids: Tuple[str, ...] = ()
    withdrawn_requirement_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _text(self.event_id, "transition event_id"))
        object.__setattr__(self, "task_id", _text(self.task_id, "transition task_id"))
        try:
            phase = Phase(self.to_phase)
        except ValueError as error:
            raise LifecycleError(str(error)) from error
        if int(self.expected_generation) < 1:
            raise LifecycleError("expected_generation must be positive")
        if self.expected_plan_revision is not None and int(self.expected_plan_revision) < 1:
            raise LifecycleError("expected_plan_revision must be positive")
        object.__setattr__(self, "expected_generation", int(self.expected_generation))
        if self.expected_plan_revision is not None:
            object.__setattr__(self, "expected_plan_revision", int(self.expected_plan_revision))
        object.__setattr__(self, "to_phase", phase)
        object.__setattr__(self, "replacement_task_ids", _refs(self.replacement_task_ids, "replacement_task_ids"))
        object.__setattr__(self, "withdrawn_requirement_refs",
                           _refs(self.withdrawn_requirement_refs, "withdrawn_requirement_refs"))

    def digest(self) -> str:
        return _digest({"event_id": self.event_id, "task_id": self.task_id,
                        "expected_generation": self.expected_generation,
                        "expected_plan_revision": self.expected_plan_revision,
                        "to_phase": self.to_phase.value,
                        "replacement_task_ids": self.replacement_task_ids,
                        "withdrawn_requirement_refs": self.withdrawn_requirement_refs})


def apply_transition(snapshot: LifecycleSnapshot, transition: TaskTransition) -> LifecycleSnapshot:
    """Validate one host transition and return a new snapshot.

    This is deliberately not a journal or persistence implementation.  The
    live owner must commit the returned delta through its own sequencer.
    """

    prior = dict(snapshot.applied_events)
    digest = transition.digest()
    if transition.event_id in prior:
        if prior[transition.event_id] != digest:
            raise LifecycleError("event ID was reused with a different transition")
        return snapshot
    node = snapshot.plan.node(transition.task_id)
    if (transition.expected_plan_revision is not None
            and snapshot.plan.revision != transition.expected_plan_revision):
        raise LifecycleError("transition plan revision is stale")
    if node.generation != transition.expected_generation:
        raise LifecycleError("transition generation is stale")
    if node.phase in {Phase.CANCELLED, Phase.SUPERSEDED, Phase.COMPLETED}:
        raise LifecycleError(f"terminal task cannot transition: {node.task_id}")
    allowed = {
        Phase.PLANNED: {Phase.ACTIVE, Phase.WAITING, Phase.BLOCKED, Phase.CANCELLED, Phase.SUPERSEDED},
        Phase.ACTIVE: {Phase.WAITING, Phase.BLOCKED, Phase.COMPLETED, Phase.CANCELLED, Phase.SUPERSEDED},
        Phase.WAITING: {Phase.ACTIVE, Phase.BLOCKED, Phase.COMPLETED, Phase.CANCELLED, Phase.SUPERSEDED},
        Phase.BLOCKED: {Phase.ACTIVE, Phase.WAITING, Phase.COMPLETED, Phase.CANCELLED, Phase.SUPERSEDED},
    }
    if transition.to_phase not in allowed.get(node.phase, set()):
        raise LifecycleError(f"invalid transition {node.phase.value}->{transition.to_phase.value}")
    if transition.to_phase in {Phase.CANCELLED, Phase.SUPERSEDED}:
        withdrawn = set(transition.withdrawn_requirement_refs)
        if not withdrawn.issubset(node.requirement_refs):
            raise LifecycleError("withdrawn requirement is not owned by the task")
        retained = set(node.requirement_refs) - withdrawn
        replacements = set(transition.replacement_task_ids)
        if node.task_id in replacements:
            raise LifecycleError("a task cannot replace itself")
        for replacement_id in replacements:
            replacement = snapshot.plan.node(replacement_id)
            if not retained.intersection(replacement.requirement_refs):
                raise LifecycleError(f"replacement does not cover {node.task_id}: {replacement_id}")
        if retained and not replacements:
            raise LifecycleError("cancellation must preserve requirement coverage or withdraw it explicitly")
    plan = snapshot.plan.replace_node(replace(node, phase=transition.to_phase))
    return replace(snapshot, plan=plan, sequence=snapshot.sequence + 1,
                   applied_events=(*snapshot.applied_events, (transition.event_id, digest)))


@dataclass(frozen=True)
class TaskChange:
    change_id: str
    previous_request_revision: str
    next_request_revision: str
    changed_refs: Tuple[str, ...] = ()
    affected_task_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        change_id = _text(self.change_id, "change_id")
        previous = _text(self.previous_request_revision, "previous_request_revision")
        next_revision = _text(self.next_request_revision, "next_request_revision")
        if previous == next_revision:
            raise LifecycleError("task change must advance request revision")
        refs = _refs(self.changed_refs, "changed_refs")
        tasks = _refs(self.affected_task_ids, "affected_task_ids")
        if not refs and not tasks:
            raise LifecycleError("task change must identify changed refs or affected tasks")
        object.__setattr__(self, "change_id", change_id)
        object.__setattr__(self, "previous_request_revision", previous)
        object.__setattr__(self, "next_request_revision", next_revision)
        object.__setattr__(self, "changed_refs", refs)
        object.__setattr__(self, "affected_task_ids", tasks)


@dataclass(frozen=True)
class TaskChangeImpact:
    change_id: str
    next_request_revision: str
    stale_task_ids: Tuple[str, ...]
    stale_evidence_ids: Tuple[str, ...]
    dispositions: Tuple[Tuple[str, str], ...]
    readiness_reopened: bool

    def to_dict(self) -> Dict[str, object]:
        return {"change_id": self.change_id, "next_request_revision": self.next_request_revision,
                "stale_task_ids": list(self.stale_task_ids),
                "stale_evidence_ids": list(self.stale_evidence_ids),
                "dispositions": {key: value for key, value in self.dispositions},
                "readiness_reopened": self.readiness_reopened}


def track_task_change(snapshot: LifecycleSnapshot, change: TaskChange) -> TaskChangeImpact:
    """Compute request-change invalidation without mutating any live owner."""

    if change.previous_request_revision != snapshot.request_revision:
        raise LifecycleError("task change starts from a stale request revision")
    refs = set(change.changed_refs)
    explicit = set(change.affected_task_ids)
    known_task_ids = {node.task_id for node in snapshot.plan.nodes}
    unknown_task_ids = explicit - known_task_ids
    if unknown_task_ids:
        raise LifecycleError("task change references an unknown task: " + sorted(unknown_task_ids)[0])
    evidence_owned_tasks = {
        item.task_id for item in snapshot.evidence if refs.intersection(item.source_refs)
    }
    stale: List[str] = []
    for node in snapshot.plan.nodes:
        binding_paths = {item.path for item in node.source_bindings}
        if (node.task_id in explicit or node.task_id in evidence_owned_tasks
                or refs.intersection(node.requirement_refs)
                or refs.intersection(binding_paths)):
            stale.append(node.task_id)
    # A producer change invalidates every transitive consumer.  This is a
    # projection only; BUILDLEAD commits the corresponding owner deltas.
    changed = set(stale)
    while True:
        dependents = {edge.consumer_id for edge in snapshot.plan.edges
                      if edge.producer_id in changed}
        additions = dependents - changed
        if not additions:
            break
        stale.extend(sorted(additions))
        changed.update(additions)
    stale_ids = tuple(stale)
    stale_evidence = tuple(item.evidence_id for item in snapshot.evidence
                           if item.task_id in stale_ids or refs.intersection(item.source_refs))
    dispositions: List[Tuple[str, str]] = []
    for task_id in stale_ids:
        phase = snapshot.plan.node(task_id).phase
        disposition = "STALE_REPAIR_GENERATION" if phase is Phase.COMPLETED else (
            "HISTORY_ONLY" if phase in {Phase.CANCELLED, Phase.SUPERSEDED} else "REPLAN_REQUIRED")
        dispositions.append((task_id, disposition))
    return TaskChangeImpact(
        change_id=change.change_id, next_request_revision=change.next_request_revision,
        stale_task_ids=stale_ids, stale_evidence_ids=stale_evidence,
        dispositions=tuple(dispositions), readiness_reopened=bool(stale_ids),
    )


__all__ = [
    "DependencyEdge", "EvidenceRecord", "EvidenceState", "LifecycleError", "LifecyclePlan",
    "LifecycleSnapshot", "NodeKind", "OUTLINE_SCHEMA", "POLICY_VERSION", "Phase",
    "PlanReadiness", "SATISFIED", "SCHEMA", "SourceBinding", "SubmissionReceipt",
    "TaskChange", "TaskChangeImpact", "TaskNode", "TaskPredicates", "TaskTransition",
    "UNKNOWN", "VIOLATED", "apply_transition", "track_task_change",
]
