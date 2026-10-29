"""The smallest live composition of the kernel's cognition organs.

This module is deliberately an adapter, not a second implementation of any organ.
It folds typed executor receipts into a bounded state, asks the existing micrograph
selector for one advisory action, and persists only deterministic SNITCH records.
No method here edits code, executes an action, licenses completion, or accepts model
prose as evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from detreason import micrograph as mg
from detreason.attempt_memory import AttemptMemoryV1, validate_agent_world_payload
from detreason.uledger import STANDARD, ULedger
from detreason.world_knowledge import WorldKnowledgeStore
from expectation_ledger import ExpectationLedger
from judgment_gap_schemas import (
    AdvisoryKind,
    CertaintyDemand,
    CertaintyPreAnswer,
    CertaintyStatus,
    CompletionState,
    ContractSummary,
    CurrentState,
    DependencySummary,
    EditPermission,
    Frontier,
    HypothesisAssessment,
    HypothesisStatus,
    NegativeWork,
    Objective,
    PersistenceState,
    Provenance,
    Receipt as JudgmentReceipt,
    ResumeState,
    RunScope,
    SituationCard,
    UnknownReason,
)
from kernel_close_look import CloseLookKernel
from kernel_fact_compiler import compile_facts, render_contract_facts
from kernel_liveness import Evidence, OrganCensusSpec, TriState, run_census
from kernel_nitpicker import ContractAtom
from kernel_semantic_ir import SourceSpan


SCHEMA = "ComposedCognition/1"
ATTENTION_BUDGET_E394 = 8
REPLAN_REPEAT_THRESHOLD_E423 = 3
REPLAN_WINDOW_E394 = 8
MAX_REPLAN_CARDS = 3
MAX_VISIBLE_WORLD_FACTS = 8
_NO_OP_RESULTS = frozenset({"NOOP", "NO_OP", "UNCHANGED", "SKIPPED", "REJECTED"})
_INFRA_RESULTS = frozenset({"INFRA_ERROR", "ENVIRONMENT_ERROR", "SELECTOR_ERROR", "UNKNOWN"})
_CANDIDATE_FIELDS = frozenset(
    {"action_id", "id", "text", "target", "predictions", "expected_outcomes", "cost", "risk",
     "reliability", "reachability", "implemented", "tier", "dependencies", "source_id", "ordinal",
     "priority", "contract_keys"}
)


class CandidateSource(str, Enum):
    """The closed-world origins allowed to create licensed action candidates."""

    SEAM = "seam"
    ORACLE_ATOM = "oracle_atom"
    CONTRACT_CARD = "contract_card"
    TEST_RECOVERY = "test_recovery"
    GUARD_ANCHOR = "guard_anchor"
    KNOWN_DEPENDENCY = "known_dependency"


@dataclass(frozen=True, slots=True)
class TrustedReceipt:
    """A minimum typed receipt accepted by the composition boundary."""

    receipt_id: str
    kind: str
    result_class: str
    turn: int
    workspace_revision: int
    fingerprint: str
    action_id: str
    intent: str = ""
    anchor: str = ""
    detail: str = ""
    dependency_generation: int = 0
    raw: Mapping[str, object] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "TrustedReceipt":
        if not isinstance(value, Mapping):
            raise ValueError("trusted receipt must be a mapping")
        kind = str(value.get("kind") or "").strip().lower()
        result = str(value.get("result_class", value.get("result", "")) or "").strip().upper()
        fingerprint = str(value.get("fingerprint") or "").strip()
        if not kind or not result or not fingerprint:
            raise ValueError("trusted receipt needs kind, result_class, and fingerprint")
        try:
            turn = int(value.get("turn", 0))
            revision = int(value.get("workspace_revision", 0))
            generation = int(value.get("dependency_generation", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("receipt turn, workspace_revision, and dependency_generation must be integers") from exc
        if turn < 0 or revision < 0 or generation < 0:
            raise ValueError("receipt counters must be non-negative")
        action_id = str(value.get("action_id") or f"observed:{fingerprint}").strip()
        stable = {
            "kind": kind,
            "result_class": result,
            "turn": turn,
            "workspace_revision": revision,
            "fingerprint": fingerprint,
            "action_id": action_id,
            "intent": str(value.get("intent") or "")[:500],
            "anchor": str(value.get("anchor", value.get("primary_anchor", "")) or "")[:300],
            "detail": str(value.get("detail") or "")[:500],
            "dependency_generation": generation,
        }
        receipt_id = "receipt:" + _digest(stable).removeprefix("sha256:")
        return cls(receipt_id=receipt_id, raw=dict(value), **stable)

    def schema_record(self, scope: RunScope) -> JudgmentReceipt:
        """Validate this boundary through the landed slice-0 receipt schema."""

        return JudgmentReceipt(
            receipt_id=self.receipt_id,
            action_id=self.action_id,
            kind=self.kind,
            result_class=self.result_class,
            turn=self.turn,
            source_revision=scope.source_revision,
            env_id=scope.env_id,
            oracle_id=scope.oracle_id,
            selector=self.intent,
            target=self.anchor,
            provenance=Provenance(receipt_ids=(self.receipt_id,)),
        )


@dataclass(frozen=True, slots=True)
class AuthorizedAction:
    """One deterministic action source projected into the micrograph vocabulary."""

    candidate_id: str
    source: CandidateSource | str
    text: str
    predictions: Mapping[str, object]
    target: Mapping[str, str] = field(default_factory=dict)
    cost: float | None = 0.0
    risk: float | None = 0.0
    reliability: float | None = 1.0
    reachability: float | None = 1.0
    implemented: bool = True
    tier: int = 0
    dependencies: tuple[str, ...] = ()
    contract_keys: tuple[str, ...] = ()
    source_id: str = ""
    ordinal: int = 0
    priority: int = 0

    def __post_init__(self) -> None:
        source = self.source if isinstance(self.source, CandidateSource) else CandidateSource(str(self.source))
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "candidate_id", str(self.candidate_id).strip())
        if not self.candidate_id:
            raise ValueError("authorized action candidate_id is required")
        object.__setattr__(self, "text", " ".join(str(self.text).split())[:1000])
        object.__setattr__(self, "predictions", dict(self.predictions))
        object.__setattr__(self, "target", {str(k): str(v) for k, v in self.target.items()})
        object.__setattr__(self, "dependencies", tuple(sorted({str(v) for v in self.dependencies if str(v).strip()})))
        object.__setattr__(self, "contract_keys", tuple(sorted({str(v) for v in self.contract_keys if str(v).strip()})))
        try:
            tier = int(self.tier)
            ordinal = int(self.ordinal)
        except (TypeError, ValueError) as exc:
            raise ValueError("action tier and ordinal must be integers") from exc
        if tier < 0 or ordinal < 0:
            raise ValueError("action tier and ordinal must be non-negative")
        object.__setattr__(self, "tier", tier)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "priority", int(self.priority))
        for name in ("cost", "risk", "reliability", "reachability"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                object.__setattr__(self, name, None)
            elif value is not None:
                object.__setattr__(self, name, float(value))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], source: CandidateSource) -> "AuthorizedAction":
        predictions = value.get("predictions", value.get("expected_outcomes", {}))
        if not isinstance(predictions, Mapping):
            predictions = {}
        source_id = str(value.get("source_id", value.get("id", "")) or "")
        target_value = value.get("target", {})
        target = target_value if isinstance(target_value, Mapping) else {}
        raw_id = str(value.get("action_id", value.get("id", "")) or "")
        candidate_id = raw_id or "candidate:" + _digest({"source": source.value, "source_id": source_id, "target": target})
        deps = value.get("dependencies", ())
        if isinstance(deps, str):
            deps = (deps,)
        return cls(
            candidate_id=candidate_id,
            source=source,
            source_id=source_id or candidate_id,
            text=str(value.get("text") or value.get("intent") or candidate_id),
            predictions=predictions,
            target=target,
            cost=value.get("cost", 0.0),
            risk=value.get("risk", 0.0),
            reliability=value.get("reliability", 1.0),
            reachability=value.get("reachability", 1.0),
            implemented=bool(value.get("implemented", True)),
            tier=int(value.get("tier", 0)),
            dependencies=tuple(deps) if isinstance(deps, Sequence) else (),
            contract_keys=tuple(value.get("contract_keys", ())) if isinstance(value.get("contract_keys", ()), Sequence) else (),
            ordinal=int(value.get("ordinal", 0)),
            priority=int(value.get("priority", 0)),
        )

    def micrograph_value(self) -> mg.ActionCandidate:
        return mg.ActionCandidate(
            self.candidate_id,
            self.predictions,
            self.cost,
            self.risk,
            self.reliability,
            authorized=self.implemented,
            reachability=self.reachability,
        )

    def advisory(self, scope: RunScope, selected: bool = False) -> object:
        return _candidate_advisory(self, scope, selected=selected)


@dataclass(frozen=True, slots=True)
class ReplanTrigger:
    should_fire: bool
    reason: str
    receipt_id: str | None = None
    challenge_once: bool = False
    evidence_required: bool = False
    handoff: str | None = None


@dataclass(frozen=True, slots=True)
class ComposedAdvance:
    receipt: TrustedReceipt
    trigger: ReplanTrigger
    selection: mg.GetSureSelection
    card: SituationCard
    rendered_card: str
    organ_output: str | None
    stage_receipts: tuple[Mapping[str, object], ...]


def _canonical(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def _json(value: object) -> str:
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _action_fingerprint(action: AuthorizedAction) -> str:
    return _digest({
        "candidate_id": action.candidate_id,
        "source": action.source.value,
        "text": action.text,
        "predictions": action.predictions,
        "target": action.target,
        "cost": action.cost,
        "risk": action.risk,
        "reliability": action.reliability,
        "reachability": action.reachability,
        "implemented": action.implemented,
        "tier": action.tier,
        "dependencies": action.dependencies,
        "contract_keys": action.contract_keys,
        "source_id": action.source_id,
        "ordinal": action.ordinal,
        "priority": action.priority,
    })


def _clip(value: object, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _span_for(value: object, fallback_path: str = "kernel") -> SourceSpan:
    text = str(value or "")
    match = re.match(r"(.+?)(?::(\d+))?(?::[^:]*)?$", text)
    path = (match.group(1) if match else text) or fallback_path
    line = int(match.group(2) or 1) if match else 1
    return SourceSpan(path=path, start_line=max(1, line), symbol="", role="composition-source")


def _provenance(*, receipt_ids: Iterable[str] = (), detail: str = "") -> Provenance:
    refs = tuple(receipt_ids)
    # Schema provenance cannot be an ungrounded prose label. The adapter itself
    # is the deterministic source when a still-unknown hypothesis has no receipt.
    spans = (_span_for("composed_cognition.py:1"),) if not refs else ()
    return Provenance(source_spans=spans, receipt_ids=refs, detail=detail)


def _candidate_advisory(action: AuthorizedAction, scope: RunScope, *, selected: bool) -> object:
    from judgment_gap_schemas import CandidateAdvisory

    return CandidateAdvisory(
        candidate_id=action.candidate_id,
        kind=AdvisoryKind.FRONTIER if selected else AdvisoryKind.DEPENDENCY,
        text=("PROPOSED (advisory): " if selected else "Available seam: ") + action.text,
        provenance=Provenance(
            source_spans=(_span_for(action.source_id or action.candidate_id, action.source.value),),
            detail=f"source={action.source.value}:{action.source_id}",
        ),
        mandatory=selected,
        blocks_frontier=selected,
        unlock_value=max(0, int(action.priority)),
        risk=max(0, int(round((action.risk or 0.0) * 100))),
        freshness=max(0, int(action.ordinal)),
        priority=max(0, int(action.priority)),
        action_unlocking=True,
        scope=scope,
    )


class _ExpectationWorldAdapter:
    """Map expectation-ledger facts onto the world store's closed predicate set."""

    def __init__(self, store: WorldKnowledgeStore) -> None:
        self._store = store

    def observe_fact(
        self,
        *,
        path: str,
        predicate: str,
        subject_ref: str,
        value: Mapping[str, object],
        revision: int,
        evidence_ref: str,
        spans: Sequence[tuple[int, int]] = (),
        dependencies: Sequence[str] = (),
        dependencies_complete: bool = True,
        content_digest: str = "",
    ) -> object:
        del predicate
        return self._store.observe_fact(
            path=path,
            predicate="oracle_selector",
            subject_ref=subject_ref,
            value=value,
            revision=revision,
            evidence_ref=evidence_ref,
            spans=spans,
            dependencies=dependencies,
            dependencies_complete=dependencies_complete,
            content_digest=content_digest,
        )


def order_then_cap(values: Iterable[AuthorizedAction], limit: int = 100) -> tuple[AuthorizedAction, ...]:
    """The planted control: establish order before applying a cardinality cap."""

    if limit < 1:
        return ()
    return tuple(sorted(values, key=lambda item: (-item.ordinal, -item.priority, item.candidate_id))[:limit])


def render_range_direction(start: int, stop: int, step: int) -> str:
    """Render a range direction without claiming ASCENDING for a descending range."""

    if step == 0:
        return "UNKNOWN"
    if step < 0:
        return "DESCENDING"
    if step > 0:
        return "ASCENDING"
    return "UNKNOWN"


def select_discriminating_action(
    *,
    graph: mg.MicroGraph | mg.HypothesisGraphConsumer | None,
    hypotheses: Sequence[mg.Hypothesis | Mapping[str, object] | str],
    authorized_actions: Sequence[AuthorizedAction],
    baseline_action_id: str | None,
    graph_snapshot_id: str | None = None,
    feature_enabled: bool = True,
) -> mg.GetSureSelection:
    """Call the existing selector with an explicit graph/baseline contract."""

    if graph is None or baseline_action_id is None:
        return mg.select_get_sure_action(
            {"graph_snapshot_id": graph_snapshot_id or ""},
            graph=graph,
            hypotheses=hypotheses,
            authorized_actions=(),
            baseline_action_id=baseline_action_id,
            feature_enabled=feature_enabled,
        )
    actual_snapshot_id = graph_snapshot_id
    if actual_snapshot_id is None:
        actual_snapshot_id = graph.graph.snapshot_id if isinstance(graph, mg.HypothesisGraphConsumer) else graph.snapshot_id
    return mg.select_get_sure_action(
        {"graph_snapshot_id": actual_snapshot_id},
        graph=graph,
        hypotheses=hypotheses,
        authorized_actions=[item.micrograph_value() for item in authorized_actions],
        baseline_action_id=baseline_action_id,
        feature_enabled=feature_enabled,
    )


def _trigger_selector(evidence: Evidence) -> TriState:
    event = str(evidence.event.get("event", "") or "").upper()
    return TriState.YES if event.startswith("SNITCH_SELECTOR_") else TriState.NO


def _trigger_replanner(evidence: Evidence) -> TriState:
    event = str(evidence.event.get("event", "") or "").upper()
    return TriState.YES if event.startswith("SNITCH_REPLANNER_") else TriState.NO


CENSUS_SPECS = (
    OrganCensusSpec(
        organ_id="composed.selector",
        trigger=_trigger_selector,
        fire_events=frozenset({"SNITCH_SELECTOR_SELECTED", "SNITCH_SELECTOR_FALLBACK"}),
        suppression_events=frozenset({"SNITCH_SELECTOR_SUPPRESSED", "SNITCH_SELECTOR_DEDUPED"}),
        dedup_events=frozenset({"SNITCH_SELECTOR_DEDUPED"}),
        challenge_once_events=frozenset({"SNITCH_SELECTOR_CHALLENGE_ONCE"}),
    ),
    OrganCensusSpec(
        organ_id="composed.replanner",
        trigger=_trigger_replanner,
        fire_events=frozenset({"SNITCH_REPLANNER_TRIGGERED", "SNITCH_REPLANNER_SUPPRESSED"}),
        suppression_events=frozenset({"SNITCH_REPLANNER_SUPPRESSED"}),
        dedup_events=frozenset({"SNITCH_REPLANNER_DEDUPED"}),
        challenge_once_events=frozenset({"SNITCH_REPLANNER_CHALLENGE_ONCE"}),
    ),
)


class ComposedCognition:
    """Deterministic, advisory-only vertical slice."""

    def __init__(
        self,
        *,
        scope: RunScope,
        acceptance_atoms: Sequence[str],
        baseline_ledger: ULedger,
        hypotheses: Sequence[mg.Hypothesis | Mapping[str, object]],
        seams: Sequence[Mapping[str, object]] = (),
        oracle_atoms: Sequence[Mapping[str, object]] = (),
        contract_cards: Sequence[Mapping[str, object] | ContractAtom] = (),
        test_recovery: Sequence[Mapping[str, object]] = (),
        guard_anchors: Sequence[Mapping[str, object]] = (),
        known_dependencies: Sequence[Mapping[str, object]] = (),
        dependency_status: Mapping[str, str] | None = None,
        close_look: CloseLookKernel | None = None,
        fact_view: object | None = None,
        receipt_sink: Callable[[Mapping[str, object]], object] | None = None,
        receipt_path: str | Path | None = None,
        organ_out: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.scope = scope
        self.acceptance_atoms = tuple(str(item) for item in acceptance_atoms if str(item).strip())
        self.baseline_ledger = baseline_ledger
        baseline = baseline_ledger.next_action(STANDARD)
        if baseline is None:
            raise ValueError("composed cognition requires a real U-ledger baseline action")
        self.baseline_row_id, baseline_move = baseline
        self.baseline_action_id = baseline_move.action
        self.queue: list[str] = [self.baseline_action_id]
        self.dependency_status = {str(k): str(v).upper() for k, v in (dependency_status or {}).items()}
        self._graph = mg.MicroGraph(scope.workspace_id or "workspace", scope.task_id or "task")
        self._consumer = mg.HypothesisGraphConsumer(self._graph)
        self._hypotheses: dict[str, mg.Hypothesis] = {}
        for value in hypotheses:
            item = value if isinstance(value, mg.Hypothesis) else mg.Hypothesis(
                str(value.get("hypothesis_id", value.get("id", ""))),
                str(value.get("claim", "")),
                tuple(value.get("required", ())),
                tuple(value.get("soft_predictions", ())),
                value.get("prior_weight", 1.0),
                tuple(value.get("discriminators", ())),
                tuple(value.get("provenance", ())),
            )
            self._consumer.add_hypothesis(item)
            self._hypotheses[item.hypothesis_id] = item
        self._candidate_inputs = {
            CandidateSource.SEAM: seams,
            CandidateSource.ORACLE_ATOM: oracle_atoms,
            CandidateSource.CONTRACT_CARD: contract_cards,
            CandidateSource.TEST_RECOVERY: test_recovery,
            CandidateSource.GUARD_ANCHOR: guard_anchors,
            CandidateSource.KNOWN_DEPENDENCY: known_dependencies,
        }
        self.close_look = close_look
        self.fact_view = fact_view
        self._fact_text = self._compile_fact_text(fact_view)
        self.world_store = WorldKnowledgeStore(
            {"sandbox_id": scope.sandbox_id or "sandbox", "workspace_id": scope.workspace_id,
             "mission_id": scope.mission_id, "unit_id": scope.unit_id, "task_id": scope.task_id},
        )
        self.expectations = ExpectationLedger(world_store=_ExpectationWorldAdapter(self.world_store))
        self._hypothesis_tracker = self._make_hypothesis_tracker()
        self._receipt_sink = receipt_sink
        self._receipt_path = Path(receipt_path) if receipt_path else None
        self._organ_out = organ_out
        self._trusted: list[TrustedReceipt] = []
        self._stage_receipts: list[dict[str, object]] = []
        self._seen_receipts: set[tuple[str, str, str, str, int]] = set()
        self._seen_semantics: set[str] = set()
        self._challenged: set[str] = set()
        self._negative: list[NegativeWork] = []
        self._dependency_generation = 0
        self._replan_count = 0
        self._replan_card_count = 0
        self._last_result = ""
        self._last_action = ""
        self._model_proposals: list[dict[str, str]] = []
        self._explored_approaches: set[str] = set()
        self._evidence_fingerprints: set[str] = set()
        self._advised_action_evidence: dict[str, frozenset[str]] = {}
        self._handoff_emitted = False
        self._test_fingerprint_streak = ""
        self._test_fingerprint_streak_count = 0
        self._pending_applied_edit = False
        self._edit_cycle_fingerprint = ""
        self._edit_cycle_count = 0
        self._last_repeat_replan_turn: int | None = None
        # The continuity adapter reads the last host selector result after a
        # receipt boundary.  Keeping the typed result here avoids re-running the
        # selector while rendering every subsequent frame.
        self._last_selection: mg.GetSureSelection | None = None

    def _make_hypothesis_tracker(self) -> object:
        from kernel_hypotheses import HypothesisTracker

        return HypothesisTracker(self.scope.task_id or "task", workspace_id=self.scope.workspace_id or "workspace")

    def _compile_fact_text(self, view: object | None) -> str:
        if view is None:
            return ""
        try:
            facts = compile_facts(view)
            return render_contract_facts(facts) or ""
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            return f"FACT_COMPILER_UNAVAILABLE:{type(exc).__name__}"

    @property
    def graph(self) -> mg.MicroGraph:
        return self._consumer.graph

    @property
    def stage_receipts(self) -> tuple[Mapping[str, object], ...]:
        return tuple(dict(item) for item in self._stage_receipts)

    @property
    def trusted_receipts(self) -> tuple[TrustedReceipt, ...]:
        return tuple(self._trusted)

    @property
    def queue_head(self) -> str | None:
        return self.queue[0] if self.queue else None

    @property
    def replan_card_count(self) -> int:
        return self._replan_card_count

    @property
    def advised_action_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._advised_action_evidence))

    def add_model_proposal(self, text: str) -> str:
        """Keep model intent visible, permanently outside the authorized catalog."""

        body = _clip(text, 400)
        proposal_id = "proposal:" + _digest(body).removeprefix("sha256:")[:24]
        if body and proposal_id not in {item["proposal_id"] for item in self._model_proposals}:
            self._model_proposals.append({"proposal_id": proposal_id, "text": body, "status": "UNVERIFIED"})
            tracker = self._hypothesis_tracker
            tracker.open_from_prose(body)
        return proposal_id

    def remember_explored(self, memory: AttemptMemoryV1 | Mapping[str, object]) -> tuple[str, ...]:
        """Import prior attempts as explored history, never as contradiction."""

        payload = memory.to_dict() if isinstance(memory, AttemptMemoryV1) else dict(memory)
        validate_agent_world_payload(payload)
        values: list[str] = []
        for section in (payload.get("edits", ()), payload.get("failed_legs", ()), payload.get("frontier", ())):
            if isinstance(section, Sequence) and not isinstance(section, (str, bytes)):
                for item in section:
                    if isinstance(item, Mapping):
                        approach = str(item.get("action_id", item.get("action", item.get("path", ""))) or "")
                        if approach:
                            values.append(approach)
        self._explored_approaches.update(values)
        return tuple(sorted(set(values)))

    def _source_actions(self) -> tuple[AuthorizedAction, ...]:
        actions: list[AuthorizedAction] = []
        for source, values in self._candidate_inputs.items():
            iterable: Iterable[object]
            if isinstance(values, Mapping):
                iterable = tuple(values.values())
            else:
                iterable = values
            for value in iterable:
                if isinstance(value, AuthorizedAction):
                    action = value
                elif isinstance(value, Mapping):
                    action = AuthorizedAction.from_mapping(value, source)
                else:
                    action = AuthorizedAction.from_mapping({"id": str(value), "text": str(value)}, source)
                actions.append(action)
        by_id: dict[str, AuthorizedAction] = {}
        for action in actions:
            prior = by_id.get(action.candidate_id)
            if prior is None:
                by_id[action.candidate_id] = action
            elif _action_fingerprint(prior) != _action_fingerprint(action):
                raise ValueError(f"conflicting authorized candidate {action.candidate_id!r}")
        return order_then_cap(by_id.values())

    def _licensed_actions(self) -> tuple[AuthorizedAction, ...]:
        actions = [item for item in self._source_actions() if item.implemented]
        unblocked = [item for item in actions if all(self.dependency_status.get(dep, "READY") != "BLOCKED" for dep in item.dependencies)]
        if not unblocked:
            return ()
        # A live lower-tier alternative is the only tier visible to the selector.
        lowest_tier = min(item.tier for item in unblocked)
        return tuple(item for item in unblocked if item.tier == lowest_tier)

    def _selector_actions(self, actions: Sequence[AuthorizedAction]) -> tuple[AuthorizedAction, ...]:
        """Exclude stale advice and prefer candidates never advised for this task."""

        if not actions or not self._advised_action_evidence:
            return tuple(actions)
        unadvised = tuple(item for item in actions if item.candidate_id not in self._advised_action_evidence)
        if unadvised:
            return unadvised
        return tuple(
            item
            for item in actions
            if self._evidence_fingerprints.difference(
                self._advised_action_evidence.get(item.candidate_id, frozenset())
            )
        )

    def _finalize_replan_trigger(
        self,
        trigger: ReplanTrigger,
        *,
        actions: Sequence[AuthorizedAction],
        selection: mg.GetSureSelection,
    ) -> ReplanTrigger:
        """Turn a raw trigger into one emitted card or a safe suppression."""

        if not trigger.should_fire:
            return trigger
        if self._replan_card_count >= MAX_REPLAN_CARDS:
            return ReplanTrigger(
                False,
                "replan_card_cap_reached",
                trigger.receipt_id,
                trigger.challenge_once,
                trigger.evidence_required,
                "frame_breaker_escalation",
            )
        if not actions:
            # The kernel wire still renders a baseline replan notice when no
            # authorized candidate exists, so those notices consume the same
            # per-task card budget even though they do not advise an action.
            self._replan_card_count += 1
            return trigger
        if selection.status != "selected" or not selection.selected_action_id:
            return ReplanTrigger(
                False,
                "no_new_discriminating_action",
                trigger.receipt_id,
                trigger.challenge_once,
                trigger.evidence_required,
            )
        if selection.status == "selected" and selection.selected_action_id:
            self._replan_card_count += 1
            self._advised_action_evidence[selection.selected_action_id] = frozenset(self._evidence_fingerprints)
        return trigger

    def _emit_replan_handoff(self, receipt: TrustedReceipt, trigger: ReplanTrigger) -> None:
        if not trigger.handoff or self._handoff_emitted:
            return
        self._handoff_emitted = True
        detail = {
            "reason": trigger.reason,
            "replan_cards": self._replan_card_count,
            "next": trigger.handoff,
        }
        self._emit(
            "SNITCH_FRAME_BREAKER_HANDOFF",
            stage="handoff",
            organ_id="composed.framebreaker",
            receipt=receipt,
            emitted=True,
            detail=detail,
        )
        self._emit(
            "SNITCH_ESCALATION_HANDOFF",
            stage="handoff",
            organ_id="composed.escalation",
            receipt=receipt,
            emitted=True,
            detail=detail,
        )

    def _observe_expectation(self, receipt: TrustedReceipt) -> None:
        match = re.search(r"(?:^|\s)n=(\d+)", receipt.detail)
        if receipt.kind == "grep" and match:
            self.expectations.observe_grep_count(
                int(match.group(1)), scope=self.scope.task_id or "task", environment=self.scope.env_id or "env",
                runner="receipt", subject=receipt.intent or "grep", selector=receipt.intent,
                turn=receipt.turn, workspace_revision=receipt.workspace_revision,
                action_id=receipt.action_id, receipt_id=receipt.receipt_id,
                command=receipt.intent, output_digest=receipt.fingerprint,
            )

    def _fold_world(self, receipt: TrustedReceipt) -> None:
        if receipt.kind == "test":
            path = receipt.anchor.split(":", 1)[0] if receipt.anchor else None
            self.world_store.observe_test_selector(
                selector=receipt.intent or receipt.action_id,
                result_class=receipt.result_class,
                revision=receipt.workspace_revision,
                evidence_ref=receipt.receipt_id,
                path=path,
            )
        self._observe_expectation(receipt)

    def _fold_hypotheses(self, receipt: TrustedReceipt) -> tuple[str, ...]:
        tracker = self._hypothesis_tracker
        tracker.observe(dict(receipt.raw, fingerprint=receipt.fingerprint, result_class=receipt.result_class))
        evidence = mg.make_node(
            mg.EntityKind.EVIDENCE,
            receipt.receipt_id,
            {"validity": "valid", "predicate": "receipt", "value": receipt.result_class, "path": receipt.anchor},
        )
        self._consumer.add_evidence(evidence)
        contradicted = str(receipt.raw.get("contradicts_hypothesis_id") or "")
        supported = str(receipt.raw.get("supports_hypothesis_id") or "")
        if contradicted in self._hypotheses and receipt.result_class in {"RED", "CONTRADICTION"}:
            self._consumer.contradict(contradicted, evidence, validity="valid", soft=False, assessment="contradiction")
        if supported in self._hypotheses and receipt.result_class in {"GREEN", "SUPPORTED", "OK"}:
            self._consumer.support(supported, evidence, validity="valid", predicate="receipt")
        postures = self._consumer.update_postures()
        return tuple(sorted(f"{key}:{value.posture}" for key, value in postures.items()))

    def _semantic_key(self, receipt: TrustedReceipt) -> str:
        return _digest({"kind": receipt.kind, "result": receipt.result_class, "intent": receipt.intent,
                        "anchor": receipt.anchor, "detail": receipt.detail, "revision": receipt.workspace_revision})

    def _track_repeat_pattern(self, receipt: TrustedReceipt) -> None:
        """Track RED tests and applied-edit cycles without treating reads as tests."""

        if receipt.kind == "edit":
            if receipt.result_class == "APPLIED":
                self._pending_applied_edit = True
            return
        if receipt.kind != "test":
            return
        if receipt.result_class != "RED":
            self._test_fingerprint_streak = ""
            self._test_fingerprint_streak_count = 0
            self._pending_applied_edit = False
            self._edit_cycle_fingerprint = ""
            self._edit_cycle_count = 0
            return

        if receipt.fingerprint == self._test_fingerprint_streak:
            self._test_fingerprint_streak_count += 1
        else:
            self._test_fingerprint_streak = receipt.fingerprint
            self._test_fingerprint_streak_count = 1

        if self._pending_applied_edit:
            if receipt.fingerprint == self._edit_cycle_fingerprint:
                self._edit_cycle_count += 1
            else:
                self._edit_cycle_fingerprint = receipt.fingerprint
                self._edit_cycle_count = 1
        else:
            self._edit_cycle_fingerprint = ""
            self._edit_cycle_count = 0
        self._pending_applied_edit = False

    @staticmethod
    def _stable_hypothesis_context(
        before_posture: tuple[str, ...], after_posture: tuple[str, ...]
    ) -> bool:
        """A repeated failure is actionable when posture is stable or already eliminated."""

        if not after_posture:
            return False
        if before_posture == after_posture:
            return True
        return any(item.rsplit(":", 1)[-1] == mg.ELIMINATED for item in after_posture)

    def _repeat_trigger(
        self,
        receipt: TrustedReceipt,
        before_posture: tuple[str, ...],
        after_posture: tuple[str, ...],
    ) -> ReplanTrigger | None:
        if receipt.kind != "test" or receipt.result_class != "RED":
            return None
        stable = self._stable_hypothesis_context(before_posture, after_posture)
        reason: str | None = None
        evidence_required = True
        if self._edit_cycle_count >= REPLAN_REPEAT_THRESHOLD_E423:
            reason = "edit_test_identical_fingerprint_threshold"
        elif stable and self._test_fingerprint_streak_count >= REPLAN_REPEAT_THRESHOLD_E423:
            reason = "same_red_fingerprint_threshold"
        if reason is None:
            return None

        last_turn = self._last_repeat_replan_turn
        if last_turn is not None and 0 <= receipt.turn - last_turn < REPLAN_WINDOW_E394:
            return ReplanTrigger(False, "replan_window", receipt.receipt_id, evidence_required=evidence_required)
        self._last_repeat_replan_turn = receipt.turn
        return ReplanTrigger(True, reason, receipt.receipt_id, evidence_required=evidence_required)

    def _trigger(
        self,
        receipt: TrustedReceipt,
        before_posture: tuple[str, ...],
        after_posture: tuple[str, ...],
        previous_result: str,
    ) -> ReplanTrigger:
        key = self._semantic_key(receipt)
        self._track_repeat_pattern(receipt)
        semantic_duplicate = key in self._seen_semantics
        if semantic_duplicate:
            repeat = self._repeat_trigger(receipt, before_posture, after_posture)
            if repeat is not None:
                return repeat
            return ReplanTrigger(False, "identical_observation", receipt.receipt_id)
        self._seen_semantics.add(key)
        raw = receipt.raw
        if receipt.kind == "edit" and receipt.result_class == "APPLIED" and previous_result == "GREEN":
            challenge = key not in self._challenged
            self._challenged.add(key)
            return ReplanTrigger(False, "post_green_edit_without_red", receipt.receipt_id, challenge_once=challenge)
        if bool(raw.get("e399_checkpoint")):
            if key in self._challenged:
                return ReplanTrigger(False, "e399_checkpoint_deduped", receipt.receipt_id)
            self._challenged.add(key)
            return ReplanTrigger(True, "e399_pre_abort", receipt.receipt_id, challenge_once=True)
        generation = receipt.dependency_generation
        if bool(raw.get("dependency_changed")) or generation > self._dependency_generation:
            self._dependency_generation = max(self._dependency_generation, generation)
            return ReplanTrigger(True, "dependency_changed", receipt.receipt_id, evidence_required=True)
        if before_posture != after_posture:
            return ReplanTrigger(True, "hypothesis_posture_transition", receipt.receipt_id, evidence_required=True)
        repeat = self._repeat_trigger(receipt, before_posture, after_posture)
        if repeat is not None:
            return repeat
        if receipt.result_class in _NO_OP_RESULTS or receipt.result_class in _INFRA_RESULTS:
            return ReplanTrigger(False, "non_product_observation", receipt.receipt_id)
        return ReplanTrigger(False, "no_replan_trigger", receipt.receipt_id)

    def _assessment_rows(self, receipt_ids: tuple[str, ...]) -> tuple[HypothesisAssessment, ...]:
        postures = self._consumer.update_postures()
        rows = []
        for hypothesis_id in sorted(self._hypotheses):
            hypothesis = self._hypotheses[hypothesis_id]
            posture = postures[hypothesis_id]
            if posture.posture == mg.ELIMINATED:
                status = HypothesisStatus.CONTRADICTED
                reason = "valid hard contradiction"
            elif not posture.support_evidence_ids:
                status = HypothesisStatus.UNKNOWN
                reason = "no measured evidence yet"
            elif posture.posture == mg.WEAKENED:
                status = HypothesisStatus.LIVE
                reason = "; ".join(posture.reasons)
            else:
                status = HypothesisStatus.SUPPORTED
                reason = "; ".join(posture.reasons)
            refs = tuple(sorted(set(posture.support_evidence_ids + posture.contradiction_evidence_ids)))
            rows.append(HypothesisAssessment(
                scope=self.scope, hypothesis_id=hypothesis_id, claim=hypothesis.claim, status=status,
                evidence_weights={ref: (-1 if ref in posture.contradiction_evidence_ids else 1) for ref in refs},
                evidence_refs=refs, kill_signal=reason if status is HypothesisStatus.CONTRADICTED else None,
                unknown_reason=reason if status is HypothesisStatus.UNKNOWN else None,
                retention_reason_code="cross-run-explored-is-not-contradiction",
                new_prediction="; ".join(hypothesis.discriminators),
                next_test_or_observation="; ".join(hypothesis.discriminators),
                provenance=_provenance(receipt_ids=(*receipt_ids, *refs), detail="micrograph posture"),
            ))
        return tuple(rows)

    def _contract_atoms(self) -> tuple[ContractAtom, ...]:
        atoms: list[ContractAtom] = []
        for value in self._candidate_inputs[CandidateSource.CONTRACT_CARD]:
            if isinstance(value, ContractAtom):
                atoms.append(value)
            elif isinstance(value, Mapping):
                span = _span_for(value.get("source", value.get("path", "contract")))
                atoms.append(ContractAtom(
                    kind=str(value.get("kind", "policy")), subject=str(value.get("subject", value.get("id", "contract"))),
                    expected=value.get("expected"), observed=value.get("observed"), state=str(value.get("state", "UNKNOWN")),
                    basis=str(value.get("basis", "TEST_AST")), provenance=(span,), task_id=self.scope.task_id,
                    target_symbol=str(value.get("target_symbol", "")), notes=(str(value.get("notes", "")),),
                ))
        return tuple(sorted(atoms, key=lambda atom: atom.contract_key)[:3])

    def _visible_world_facts(self) -> tuple[object, ...]:
        """Keep the model-facing card bounded while retaining newest facts."""

        return tuple(sorted(
            self.world_store.facts,
            key=lambda fact: (fact.workspace_revision, fact.fact_id),
            reverse=True,
        )[:MAX_VISIBLE_WORLD_FACTS])

    def _card(self, receipt: TrustedReceipt, trigger: ReplanTrigger, selection: mg.GetSureSelection) -> SituationCard:
        assessments = self._assessment_rows((receipt.receipt_id,))
        atoms = self._contract_atoms()
        violated = tuple(atom for atom in atoms if atom.is_contradicted)
        supported = tuple(atom for atom in atoms if not atom.is_contradicted and not atom.is_unknown)
        unresolved = tuple(atom for atom in atoms if atom.is_unknown)
        actions = self._licensed_actions()
        selected = next((item for item in actions if item.candidate_id == selection.selected_action_id), None)
        unknown: UnknownReason | None = None
        if selected is None:
            unknown = UnknownReason(
                unknown_id="unknown:selector", subject="next action", reason=selection.fallback_reason or "no authorized discriminating action",
                required_action="provide two live predictions from an authorized deterministic seam",
                provenance=_provenance(receipt_ids=(receipt.receipt_id,), detail="selector"),
            )
        advisory_values = [item.advisory(self.scope, selected=item is selected) for item in actions]
        if unknown is not None:
            from judgment_gap_schemas import CandidateAdvisory

            advisory_values.append(CandidateAdvisory(
                candidate_id="unknown:selector", kind=AdvisoryKind.UNKNOWN, text="Selector did not license a next action",
                provenance=_provenance(receipt_ids=(receipt.receipt_id,), detail="selector"),
                action_unlocking=True, unknown_reason=unknown.reason, scope=self.scope,
            ))
        safe_action = (f"PROPOSED (advisory): {selected.text}" if selected else
                       "PROPOSED (advisory): retain baseline and emit a truthful validation/recovery packet")
        frontier = Frontier(
            highest_value_question=(self._hypotheses[next(iter(sorted(self._hypotheses)))].claim
                                    if self._hypotheses else "Which deterministic observation separates the live theories?"),
            recommended_action=safe_action,
            why_now=(f"trigger={trigger.reason}; selector={selection.source}; baseline={self.baseline_action_id}"),
            unlocks=tuple(item.candidate_id for item in actions if item is selected),
            safe_alternatives=tuple(item.candidate_id for item in actions if item is not selected)[:3],
            off_frontier_policy="Advisory only; executor and completion gates remain authoritative.",
            provenance=_provenance(receipt_ids=(receipt.receipt_id,), detail="frontier"),
        )
        demand = CertaintyDemand(
            scope=self.scope, demand_point="PLAN_REVISION" if trigger.should_fire else "CHECKPOINT",
            subject=receipt.intent or receipt.action_id, source_revision=self.scope.source_revision,
            dependency_generation=self._dependency_generation,
            pre_answer=CertaintyPreAnswer(
                certainty_status=CertaintyStatus.NOT_YET,
                what_is_proven=tuple(sorted({receipt.result_class, "receipt accepted"})),
                what_is_not_proven=("selected action outcome", "completion"),
                evidence_refs=(receipt.receipt_id,),
                required_next_evidence=(selected.text if selected else "validation recovery or a new authorized probe",),
                safe_next_action=safe_action, can_edit=EditPermission.NO,
                stop_allowed=False, success_certified=False,
                unknown_reasons=(unknown,) if unknown is not None else (),
                provenance=_provenance(receipt_ids=(receipt.receipt_id,), detail="E394 attention demand"),
            ),
        )
        dependencies = DependencySummary(
            blocking=tuple(sorted(key for key, value in self.dependency_status.items() if value == "BLOCKED")),
            ready=tuple(sorted(key for key, value in self.dependency_status.items() if value != "BLOCKED")),
            reopened=(), unknown_count=0,
            cross_dependency_summary=(self._fact_text or "No additional contract fact text supplied."),
            provenance=_provenance(receipt_ids=(receipt.receipt_id,), detail="dependency fold"),
        )
        state = CurrentState(
            world_revision=receipt.workspace_revision, last_product_result=self._last_result,
            last_action=self._last_action, progress_state="REPLAN_PROPOSED" if trigger.should_fire else "OBSERVING",
            persistence_state=PersistenceState.DURABLE if self._receipt_path or self._receipt_sink else PersistenceState.DEGRADED,
            provenance=_provenance(receipt_ids=(receipt.receipt_id,), detail="folded situation state"),
            world_facts=self._visible_world_facts(),
        )
        return SituationCard(
            run_scope=self.scope,
            objective=Objective(self.acceptance_atoms, (), CompletionState.IN_PROGRESS,
                                _provenance(receipt_ids=(receipt.receipt_id,), detail="objective")),
            current_state=state, frontier=frontier, dependencies=dependencies,
            hypotheses=assessments,
            contracts=ContractSummary(violated, unresolved, supported,
                                      action_unlocking_unknown=next((item for item in advisory_values
                                                                     if getattr(item, "kind", None) is AdvisoryKind.UNKNOWN), None)),
            certainty=demand, negative_work=tuple(self._negative[-16:]),
            resume=ResumeState(safe_action, self._last_result, self._logical_digest(), len(self._stage_receipts),
                               _provenance(receipt_ids=(receipt.receipt_id,), detail="resume")),
            advisories=tuple(advisory_values),
        )

    def _logical_digest(self) -> str:
        return _digest({
            "schema": SCHEMA, "scope": self.scope.to_dict(), "receipts": [item.receipt_id for item in self._trusted],
            "graph": self.graph.snapshot_id, "queue": self.queue, "baseline": self.baseline_action_id,
            "dependency_generation": self._dependency_generation, "replans": self._replan_count,
            "replan_cards": self._replan_card_count,
            "evidence_fingerprints": sorted(self._evidence_fingerprints),
            "advised_action_evidence": {
                key: sorted(value) for key, value in sorted(self._advised_action_evidence.items())
            },
            "postures": [item.to_dict() for item in self._assessment_rows(())],
        })

    def _emit(self, event: str, *, stage: str, organ_id: str = "", receipt: TrustedReceipt | None = None,
              emitted: bool = False, detail: object = None) -> dict[str, object]:
        payload = {
            "schema": SCHEMA, "kind": "composed_cognition", "event": event, "stage": stage,
            "organ_id": organ_id, "run_id": self.scope.mission_id or self.scope.task_id,
            "receipt_id": receipt.receipt_id if receipt else "", "state_digest": self._logical_digest(),
            "emitted": bool(emitted), "detail": _canonical(detail if detail is not None else {}),
        }
        payload["digest"] = _digest(payload)
        payload["receipt_id"] = "snitch:" + str(payload["digest"]).removeprefix("sha256:")[:24]
        self._stage_receipts.append(payload)
        if self._receipt_sink is not None:
            self._receipt_sink(dict(payload))
        if self._receipt_path is not None:
            self._receipt_path.parent.mkdir(parents=True, exist_ok=True)
            with self._receipt_path.open("a", encoding="utf-8") as handle:
                handle.write(_json(payload) + "\n")
        return payload

    def advance(self, value: Mapping[str, object]) -> ComposedAdvance:
        """Fold one receipt through the complete composed slice."""

        trusted = TrustedReceipt.from_mapping(value)
        trusted.schema_record(self.scope)
        dedup_key = (trusted.kind, trusted.result_class, trusted.fingerprint, trusted.action_id, trusted.turn)
        if dedup_key in self._seen_receipts:
            trigger = ReplanTrigger(False, "duplicate_receipt", trusted.receipt_id)
            selection = select_discriminating_action(
                graph=self._consumer, hypotheses=self._consumer.live_hypotheses(), authorized_actions=(),
                baseline_action_id=self.baseline_action_id, graph_snapshot_id=self.graph.snapshot_id,
            )
            self._last_selection = selection
            self._emit("SNITCH_REPLANNER_DEDUPED", stage="trigger", organ_id="composed.replanner", receipt=trusted)
            card = self._card(trusted, trigger, selection)
            rendered = card.render()
            output = self._organ_out("composed_cognition", rendered) if self._organ_out else rendered
            return ComposedAdvance(trusted, trigger, selection, card, rendered, output, self.stage_receipts)
        self._seen_receipts.add(dedup_key)
        self._evidence_fingerprints.add(trusted.fingerprint)
        before = tuple(f"{key}:{value.posture}" for key, value in self._consumer.update_postures().items())
        previous_result = self._last_result
        self._trusted.append(trusted)
        self._emit("SNITCH_TRUSTED_RECEIPT", stage="trust", receipt=trusted, detail={"kind": trusted.kind})
        self._fold_world(trusted)
        self._emit("SNITCH_STATE_FOLDED", stage="fold", receipt=trusted,
                   detail={"world_revision": trusted.workspace_revision, "world_facts": len(self.world_store.facts)})
        after = self._fold_hypotheses(trusted)
        self._last_result = trusted.result_class
        self._last_action = trusted.action_id
        self._emit("SNITCH_HYPOTHESIS_POSTURE_UPDATED", stage="posture", receipt=trusted, detail={"postures": after})
        raw_trigger = self._trigger(trusted, before, after, previous_result)
        actions = self._licensed_actions() if raw_trigger.should_fire else ()
        selector_actions = self._selector_actions(actions)
        if self._replan_card_count >= MAX_REPLAN_CARDS:
            selector_actions = ()
        selection = select_discriminating_action(
            graph=self._consumer, hypotheses=self._consumer.live_hypotheses(), authorized_actions=selector_actions,
            baseline_action_id=self.baseline_action_id, graph_snapshot_id=self.graph.snapshot_id,
        )
        self._last_selection = selection
        trigger = self._finalize_replan_trigger(raw_trigger, actions=actions, selection=selection)
        trigger_event = "SNITCH_REPLANNER_TRIGGERED" if trigger.should_fire else "SNITCH_REPLANNER_SUPPRESSED"
        self._emit(trigger_event, stage="trigger", organ_id="composed.replanner", receipt=trusted,
                   emitted=trigger.should_fire, detail={
                       "reason": trigger.reason,
                       "challenge_once": trigger.challenge_once,
                       "replan_cards": self._replan_card_count,
                       "handoff": trigger.handoff,
                   })
        self._emit("SNITCH_AUTHORIZED_ACTIONS_GENERATED" if actions else "SNITCH_AUTHORIZED_ACTIONS_EMPTY",
                   stage="catalog", organ_id="composed.selector", receipt=trusted, detail={
                       "candidate_ids": [item.candidate_id for item in selector_actions],
                       "sources": sorted({item.source.value for item in selector_actions}),
                       "advised_action_ids": sorted(self._advised_action_evidence),
                   })
        selector_event = "SNITCH_SELECTOR_SELECTED" if selection.status == "selected" else "SNITCH_SELECTOR_FALLBACK"
        self._emit(selector_event, stage="selector", organ_id="composed.selector", receipt=trusted,
                   emitted=selection.status == "selected", detail={
                       **selection.to_dict(),
                       "selector_candidate_ids": [item.candidate_id for item in selector_actions],
                       "evidence_fingerprints": sorted(self._evidence_fingerprints),
                   })
        self._emit_replan_handoff(trusted, trigger)
        if trigger.should_fire and selection.status == "selected" and selection.selected_action_id != self.queue_head:
            self.queue = [selection.selected_action_id or self.baseline_action_id] + [
                item for item in self.queue if item != selection.selected_action_id
            ]
            self._replan_count += 1
            self._emit("SNITCH_QUEUE_HEAD_REPLACED", stage="queue", organ_id="composed.replanner", receipt=trusted,
                       emitted=True, detail={"old": self.queue[1] if len(self.queue) > 1 else self.baseline_action_id,
                                            "new": self.queue_head})
        else:
            self._emit("SNITCH_QUEUE_HEAD_RETAINED", stage="queue", organ_id="composed.replanner", receipt=trusted,
                       detail={"reason": trigger.reason if not trigger.should_fire else selection.fallback_reason})
        if trusted.result_class in _NO_OP_RESULTS or trigger.reason in {"identical_observation", "post_green_edit_without_red"}:
            self._negative.append(NegativeWork(
                kind="NO_OP_OBSERVATION" if trusted.result_class in _NO_OP_RESULTS else "CHALLENGE_ONCE",
                action_signature=trusted.action_id, turn=trusted.turn,
                detail=trigger.reason, ruled_out=False, provenance=_provenance(receipt_ids=(trusted.receipt_id,)),
            ))
        if self.close_look is not None:
            try:
                self.close_look.process_receipt(dict(value))
            except Exception as exc:  # close-look is advisory; receipt fold remains durable
                self._emit("SNITCH_CLOSE_LOOK_DEGRADED", stage="fold", organ_id="close-look", receipt=trusted,
                           detail={"error": type(exc).__name__})
        card = self._card(trusted, trigger, selection)
        rendered = card.render()
        self._emit("SNITCH_SITUATION_CARD_RENDERED", stage="card", organ_id="composed.selector", receipt=trusted,
                   emitted=True, detail={
                       "card_digest": card.card_digest,
                        "attention_budget": ATTENTION_BUDGET_E394,
                        "replan_card_emitted": bool(trigger.should_fire),
                   })
        if trigger.handoff:
            handoff_text = (
                "FRAME-BREAKER/ESCALATION HANDOFF: replan card cap reached; "
                "question frame requires a new bounded investigation."
            )
            output = self._organ_out("framebreak", handoff_text) if self._organ_out else handoff_text
        else:
            output = self._organ_out("composed_cognition", rendered) if self._organ_out else rendered
        self._emit("SNITCH_DIGEST_PERSISTED", stage="persist", organ_id="composed.selector", receipt=trusted,
                   emitted=output is not None, detail={"card_digest": card.card_digest, "delivered": output is not None})
        return ComposedAdvance(trusted, trigger, selection, card, rendered, output, self.stage_receipts)

    def replay(self, receipts: Iterable[Mapping[str, object]], *, trim: bool = False) -> ComposedAdvance | None:
        last: ComposedAdvance | None = None
        for value in receipts:
            if trim:
                value = {key: value[key] for key in ("kind", "result_class", "fingerprint", "turn", "workspace_revision",
                                                     "intent", "anchor", "detail", "action_id", "dependency_generation",
                                                     "e399_checkpoint") if key in value}
            last = self.advance(value)
        return last

    def census(self) -> object:
        return run_census(
            specs=CENSUS_SPECS,
            receipts=self._stage_receipts,
            run_id=self.scope.mission_id or self.scope.task_id,
            complete=True,
        )


__all__ = [
    "ATTENTION_BUDGET_E394", "MAX_REPLAN_CARDS", "MAX_VISIBLE_WORLD_FACTS", "REPLAN_REPEAT_THRESHOLD_E423", "REPLAN_WINDOW_E394", "AuthorizedAction", "CENSUS_SPECS", "CandidateSource", "ComposedAdvance",
    "ComposedCognition", "ReplanTrigger", "SCHEMA", "TrustedReceipt", "order_then_cap",
    "render_range_direction", "select_discriminating_action",
]
