"""Host-owned BUILD MODE foundations for multi-package StateFrame work.

This module is a small, deterministic qualification kernel rather than a new
completion owner.  It projects typed design/build records onto the existing
``kernel_invalidation.LedgerIndex`` and ``kernel_timeline.TimelineIndex``.
The requirements identity remains S1's ``(ledger_id, requirement_id,
version)`` shape; dependency edges remain consumer -> premise.

The implementation follows the authored BUILD MODE reports (2026-10-04):
``build_state_REPORT.md`` (typed design graph and five-facet ledger),
``design_validation_REPORT.md`` (G0-G6 readiness),
``change_mgmt_REPORT.md`` (versioned impact/rework), and
``build_orchestration_REPORT.md`` (package boundaries and live+fired done).
``model_agnostic_REPORT.md`` and ``genius_process_REPORT.md`` require that
these are CORE host invariants: a frontier model with a complete typed answer
must be accepted without Qwen-specific syntax, model-name branches, or forced
micro-steps.  Model output remains a proposal; only host-emitted receipts
can advance a facet.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import TypeAlias

from kernel_invalidation import LedgerIndex
from kernel_timeline import TimelineIndex


SCHEMA = "stateframe-build-mode/1"
POLICY_VERSION = "build-mode-policy/1"


class BuildModeError(ValueError):
    """Malformed, stale, out-of-scope, or unauthorised build operation."""


class IntegrityConflict(BuildModeError):
    """A host identity was reused with a different immutable payload."""


class StaleBasis(BuildModeError):
    """An operation was proposed against an obsolete baseline or watermark."""


class Facet(StrEnum):
    DESIGNED = "DESIGNED"
    BUILT = "BUILT"
    VERIFIED = "VERIFIED"
    WIRED_IN = "WIRED_IN"
    LIVE = "LIVE"


class AssessmentValue(StrEnum):
    SATISFIED = "SATISFIED"
    UNSATISFIED = "UNSATISFIED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Freshness(StrEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNASSESSED = "UNASSESSED"


class Gate(StrEnum):
    G0 = "G0_SCOPE_INTENT"
    G1 = "G1_GRAPH_CONSISTENCY"
    G2 = "G2_REUSE_BINDING"
    G3 = "G3_SPEC_READINESS"
    G4 = "G4_ARCHITECTURE_FEASIBILITY"
    G5 = "G5_ADVERSARIAL_DISPOSITION"
    G6 = "G6_BUILD_READY"


class GateResult(StrEnum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNKNOWN = "UNKNOWN"


class ReceiptOutcome(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"
    UNKNOWN = "UNKNOWN"


class PackageWorkflow(StrEnum):
    PLANNED = "PLANNED"
    READY = "READY"
    ACTIVE = "ACTIVE"
    BLOCKED = "BLOCKED"
    IN_REVIEW = "IN_REVIEW"
    ACCEPTED = "ACCEPTED"
    REWORK = "REWORK"
    SUPERSEDED = "SUPERSEDED"


class ChangeKind(StrEnum):
    REQUIREMENT_AMENDMENT = "REQUIREMENT_AMENDMENT"
    DESIGN_CORRECTION = "DESIGN_CORRECTION"
    CONTRACT_CHANGE = "CONTRACT_CHANGE"
    IMPLEMENTATION_DEVIATION = "IMPLEMENTATION_DEVIATION"
    ORACLE_CORRECTION = "ORACLE_CORRECTION"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    COMPOSITION_FAILURE = "COMPOSITION_FAILURE"
    OPPORTUNITY = "OPPORTUNITY"


class ImpactDisposition(StrEnum):
    REWORK_REQUIRED = "REWORK_REQUIRED"
    REVERIFY_REQUIRED = "REVERIFY_REQUIRED"
    REVISIT_REQUIRED = "REVISIT_REQUIRED"
    PRESERVED = "PRESERVED"
    UNKNOWN_IMPACT = "UNKNOWN_IMPACT"


class DecisionStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    SUSPECT = "SUSPECT"
    REJECTED = "REJECTED"


class EdgeRelation(StrEnum):
    DEPENDS_ON = "DEPENDS_ON"
    IMPLEMENTS = "IMPLEMENTS"
    PROVIDES = "PROVIDES"
    CONSUMES = "CONSUMES"
    VERIFIES = "VERIFIES"
    REALIZED_BY = "REALIZED_BY"
    CHECKED_BY = "CHECKED_BY"
    OBSERVED_BY = "OBSERVED_BY"
    COMPOSES = "COMPOSES"
    GOVERNED_BY = "GOVERNED_BY"


def _text(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    value_text = str(value or "").strip()
    if not value_text and not allow_empty:
        raise BuildModeError(f"{field_name} is required")
    return value_text


def _strings(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _refs(values: Iterable["Ref"]) -> tuple["Ref", ...]:
    result: list[Ref] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Ref):
            raise BuildModeError(
                f"typed reference collection contains {type(value).__name__}, expected Ref"
            )
        if value.key not in seen:
            seen.add(value.key)
            result.append(value)
    return tuple(result)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict") and not isinstance(value, type):
        return value.to_dict()  # type: ignore[no-any-return]
    return value


@dataclass(frozen=True, slots=True)
class Scope:
    workspace_id: str
    project_id: str
    work_id: str
    repository_id: str

    def __post_init__(self) -> None:
        for name in ("workspace_id", "project_id", "work_id", "repository_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))

    @property
    def key(self) -> str:
        return "/".join((self.workspace_id, self.project_id, self.work_id, self.repository_id))

    def to_dict(self) -> dict[str, object]:
        return _plain(asdict(self))  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Ref:
    scope: Scope
    node_id: str
    revision: int = 1
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.scope, Scope):
            raise BuildModeError("Ref.scope must be Scope")
        object.__setattr__(self, "node_id", _text(self.node_id, "node_id"))
        if type(self.revision) is not int or self.revision < 1:
            raise BuildModeError("Ref.revision must be a positive integer")
        if self.digest and not self.digest.startswith("sha256:"):
            raise BuildModeError("Ref.digest must be sha256-prefixed")

    @property
    def key(self) -> str:
        return f"{self.scope.key}:{self.node_id}@{self.revision}"

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "node_id": self.node_id,
            "revision": self.revision,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class ReqRef:
    """S1-compatible requirement identity; never replace with a string ID."""

    ledger_id: str
    requirement_id: str
    version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "ledger_id", _text(self.ledger_id, "ledger_id"))
        object.__setattr__(self, "requirement_id", _text(self.requirement_id, "requirement_id"))
        if type(self.version) is not int or self.version < 1:
            raise BuildModeError("ReqRef.version must be a positive integer")

    @property
    def key(self) -> str:
        return f"{self.ledger_id}:{self.requirement_id}@{self.version}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    digest: str
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "digest", _text(self.digest, "artifact.digest"))
        object.__setattr__(self, "media_type", _text(self.media_type, "artifact.media_type"))


class _RecordMixin:
    """Closed-record serialization shared by graph payload dataclasses."""

    def to_dict(self) -> dict[str, object]:
        value = _plain(asdict(self))
        if not isinstance(value, dict):
            raise BuildModeError("record serialization did not produce an object")
        return value


@dataclass(frozen=True, slots=True)
class RequirementNode(_RecordMixin):
    ref: Ref
    requirement: ReqRef
    intent: str
    source_refs: tuple[str, ...]
    observation_ids: tuple[str, ...]
    acceptance_refs: tuple[Ref, ...] = ()
    package_refs: tuple[Ref, ...] = ()
    admitted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _text(self.intent, "requirement.intent"))
        object.__setattr__(self, "source_refs", _strings(self.source_refs))
        object.__setattr__(self, "observation_ids", _strings(self.observation_ids))
        object.__setattr__(self, "acceptance_refs", _refs(self.acceptance_refs))
        object.__setattr__(self, "package_refs", _refs(self.package_refs))
        if not self.source_refs or not self.observation_ids:
            raise BuildModeError("a requirement needs source refs and observations")


@dataclass(frozen=True, slots=True)
class Component(_RecordMixin):
    ref: Ref
    name: str
    responsibility: str
    component_kind: str
    requirement_refs: tuple[Ref, ...]
    provides_contract_refs: tuple[Ref, ...]
    consumes_contract_refs: tuple[Ref, ...]
    work_package_ref: Ref
    reuse_disposition: str
    reuse_evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("name", "responsibility", "component_kind", "reuse_disposition"):
            object.__setattr__(self, name, _text(getattr(self, name), f"component.{name}"))
        object.__setattr__(self, "requirement_refs", _refs(self.requirement_refs))
        object.__setattr__(self, "provides_contract_refs", _refs(self.provides_contract_refs))
        object.__setattr__(self, "consumes_contract_refs", _refs(self.consumes_contract_refs))
        object.__setattr__(self, "reuse_evidence_refs", _strings(self.reuse_evidence_refs))


@dataclass(frozen=True, slots=True)
class Contract(_RecordMixin):
    ref: Ref
    contract_id: str
    kind: str
    producer_ref: Ref
    consumer_refs: tuple[Ref, ...]
    schema_digest: str
    required_fields: tuple[str, ...]
    compatibility_policy: str
    version: str

    def __post_init__(self) -> None:
        for name in ("contract_id", "kind", "schema_digest", "compatibility_policy", "version"):
            object.__setattr__(self, name, _text(getattr(self, name), f"contract.{name}"))
        object.__setattr__(self, "consumer_refs", _refs(self.consumer_refs))
        object.__setattr__(self, "required_fields", _strings(self.required_fields))
        if not self.consumer_refs or not self.required_fields:
            raise BuildModeError("a contract needs consumers and required fields")


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion(_RecordMixin):
    ref: Ref
    requirement_ref: Ref
    package_ref: Ref
    scenario_id: str
    predicate: str
    check_id: str
    negative_control_id: str
    required: bool = True

    def __post_init__(self) -> None:
        for name in ("scenario_id", "predicate", "check_id", "negative_control_id"):
            object.__setattr__(self, name, _text(getattr(self, name), f"acceptance.{name}"))


@dataclass(frozen=True, slots=True)
class Journey(_RecordMixin):
    ref: Ref
    actor: str
    entrypoint: str
    package_refs: tuple[Ref, ...]
    acceptance_refs: tuple[Ref, ...]
    target: str

    def __post_init__(self) -> None:
        for name in ("actor", "entrypoint", "target"):
            object.__setattr__(self, name, _text(getattr(self, name), f"journey.{name}"))
        object.__setattr__(self, "package_refs", _refs(self.package_refs))
        object.__setattr__(self, "acceptance_refs", _refs(self.acceptance_refs))
        if not self.package_refs or not self.acceptance_refs:
            raise BuildModeError("a journey needs packages and acceptance criteria")


@dataclass(frozen=True, slots=True)
class WorkPackage(_RecordMixin):
    ref: Ref
    name: str
    requirement_refs: tuple[Ref, ...]
    component_refs: tuple[Ref, ...]
    contract_refs: tuple[Ref, ...]
    acceptance_refs: tuple[Ref, ...]
    dependency_refs: tuple[Ref, ...]
    integration_refs: tuple[Ref, ...]
    output_paths: tuple[str, ...]
    replaces_ref: Ref | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "package.name"))
        for name in (
            "requirement_refs", "component_refs", "contract_refs", "acceptance_refs",
            "dependency_refs", "integration_refs",
        ):
            object.__setattr__(self, name, _refs(getattr(self, name)))
        object.__setattr__(self, "output_paths", _strings(self.output_paths))
        if not self.requirement_refs or not self.acceptance_refs or not self.output_paths:
            raise BuildModeError("a package needs requirements, acceptance, and outputs")


@dataclass(frozen=True, slots=True)
class IntegrationObligation(_RecordMixin):
    ref: Ref
    producer_ref: Ref
    consumer_ref: Ref
    contract_ref: Ref
    package_ref: Ref
    boundary_kind: str
    journey_refs: tuple[Ref, ...]
    effect_ids: tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "boundary_kind", _text(self.boundary_kind, "integration.boundary_kind"))
        object.__setattr__(self, "journey_refs", _refs(self.journey_refs))
        object.__setattr__(self, "effect_ids", _strings(self.effect_ids))
        if not self.effect_ids:
            raise BuildModeError("an integration obligation needs an effect atom")


@dataclass(frozen=True, slots=True)
class DesignBaseline(_RecordMixin):
    ref: Ref
    requirement_refs: tuple[Ref, ...]
    component_refs: tuple[Ref, ...]
    contract_refs: tuple[Ref, ...]
    acceptance_refs: tuple[Ref, ...]
    journey_refs: tuple[Ref, ...]
    package_refs: tuple[Ref, ...]
    integration_refs: tuple[Ref, ...]
    target_refs: tuple[str, ...]
    source_dispositions_complete: bool = True
    check_specs_complete: bool = True
    replaces_ref: Ref | None = None

    def __post_init__(self) -> None:
        for name in (
            "requirement_refs", "component_refs", "contract_refs", "acceptance_refs",
            "journey_refs", "package_refs", "integration_refs",
        ):
            object.__setattr__(self, name, _refs(getattr(self, name)))
        object.__setattr__(self, "target_refs", _strings(self.target_refs))
        if not self.requirement_refs or not self.package_refs or not self.target_refs:
            raise BuildModeError("a design baseline needs requirements, packages, and targets")


@dataclass(frozen=True, slots=True)
class ImplementationBinding(_RecordMixin):
    ref: Ref
    design_ref: Ref
    package_ref: Ref
    candidate_ref: Ref
    path: str
    content_digest: str
    role: str

    def __post_init__(self) -> None:
        for name in ("path", "content_digest", "role"):
            object.__setattr__(self, name, _text(getattr(self, name), f"binding.{name}"))


@dataclass(frozen=True, slots=True)
class ExecutionReceipt(_RecordMixin):
    ref: Ref
    receipt_id: str
    kind: str
    subject_ref: Ref
    package_ref: Ref | None
    facet: Facet | None
    baseline_ref: Ref
    candidate_ref: Ref
    target_ref: str
    environment_digest: str
    config_digest: str
    input_digest: str
    outcome: ReceiptOutcome
    assertion_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    reached_boundaries: tuple[str, ...]
    source_paths: tuple[str, ...]
    coverage: str
    host_issued: bool
    authenticity_digest: str
    fired: bool = False

    def __post_init__(self) -> None:
        for name in (
            "receipt_id", "kind", "target_ref", "environment_digest", "config_digest",
            "input_digest", "coverage", "authenticity_digest",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), f"receipt.{name}"))
        object.__setattr__(self, "assertion_ids", _strings(self.assertion_ids))
        object.__setattr__(self, "event_ids", _strings(self.event_ids))
        object.__setattr__(self, "reached_boundaries", _strings(self.reached_boundaries))
        object.__setattr__(self, "source_paths", _strings(self.source_paths))
        if self.outcome is ReceiptOutcome.PASS and not self.host_issued:
            raise BuildModeError("a passing receipt must be host-issued")


@dataclass(frozen=True, slots=True)
class GateEvaluation(_RecordMixin):
    ref: Ref
    gate: Gate
    baseline_ref: Ref
    result: GateResult
    reason_codes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    required_count: int
    covered_count: int
    input_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs))
        if self.required_count < 0 or self.covered_count < 0:
            raise BuildModeError("gate counts must be non-negative")


@dataclass(frozen=True, slots=True)
class BuildAdmission(_RecordMixin):
    ref: Ref
    baseline_ref: Ref
    allowed_package_refs: tuple[Ref, ...]
    basis_digest: str
    dependency_watermark: int
    owner_epoch: int
    expires_at: str


@dataclass(frozen=True, slots=True)
class Assessment(_RecordMixin):
    value: AssessmentValue
    freshness: Freshness
    receipt_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_refs", _strings(self.receipt_refs))
        object.__setattr__(self, "reason_codes", _strings(self.reason_codes))


@dataclass(frozen=True, slots=True)
class BuildAssessment(_RecordMixin):
    package_ref: Ref
    baseline_ref: Ref
    candidate_ref: Ref
    target_ref: str
    facets: Mapping[str, Assessment]
    fired_receipt_refs: tuple[str, ...]
    missing_predicates: tuple[str, ...]

    @property
    def live_and_fired(self) -> bool:
        return (
            all(
                self.facets.get(facet.value, Assessment(AssessmentValue.UNKNOWN, Freshness.UNASSESSED, (), ())).value
                is AssessmentValue.SATISFIED
                for facet in Facet
            )
            and bool(self.fired_receipt_refs)
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_ref", _text(self.target_ref, "assessment.target_ref"))
        object.__setattr__(self, "missing_predicates", _strings(self.missing_predicates))
        object.__setattr__(self, "fired_receipt_refs", _strings(self.fired_receipt_refs))


@dataclass(frozen=True, slots=True)
class BuildLedger(_RecordMixin):
    work_id: str
    baseline_ref: Ref
    candidate_ref: Ref
    target_ref: str
    assessments: tuple[BuildAssessment, ...]
    required_total: int
    satisfied: int
    stale: int
    unknown: int
    failed: int
    done: bool
    done_reason: str


@dataclass(frozen=True, slots=True)
class TraceLink(_RecordMixin):
    ref: Ref
    source_ref: Ref
    relation: str
    target_ref: Ref
    admission_receipt_id: str


@dataclass(frozen=True, slots=True)
class ChangeProposal(_RecordMixin):
    ref: Ref
    kind: ChangeKind
    expected_baseline_ref: Ref
    old_refs: tuple[Ref, ...]
    new_refs: tuple[Ref, ...]
    replacement_package_refs: tuple[Ref, ...]
    replacement_acceptance_refs: tuple[Ref, ...]
    replacement_integration_refs: tuple[Ref, ...]
    replacement_journey_refs: tuple[Ref, ...]
    source_message_ref: str
    source_paths: tuple[str, ...]
    rationale: str
    opportunity_only: bool = False
    replacement_requirement_refs: tuple[Ref, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "old_refs", _refs(self.old_refs))
        object.__setattr__(self, "new_refs", _refs(self.new_refs))
        object.__setattr__(self, "replacement_package_refs", _refs(self.replacement_package_refs))
        object.__setattr__(self, "replacement_acceptance_refs", _refs(self.replacement_acceptance_refs))
        object.__setattr__(self, "replacement_integration_refs", _refs(self.replacement_integration_refs))
        object.__setattr__(self, "replacement_journey_refs", _refs(self.replacement_journey_refs))
        object.__setattr__(self, "replacement_requirement_refs", _refs(self.replacement_requirement_refs))
        object.__setattr__(self, "source_message_ref", _text(self.source_message_ref, "change.source_message_ref"))
        object.__setattr__(self, "source_paths", _strings(self.source_paths))
        object.__setattr__(self, "rationale", _text(self.rationale, "change.rationale"))


@dataclass(frozen=True, slots=True)
class ImpactRow(_RecordMixin):
    subject_ref: Ref
    facets: tuple[Facet, ...]
    disposition: ImpactDisposition
    cause_refs: tuple[Ref, ...]
    reason_code: str


@dataclass(frozen=True, slots=True)
class ImpactAnalysis(_RecordMixin):
    ref: Ref
    proposal_ref: Ref
    graph_watermark: int
    coverage: str
    affected: tuple[ImpactRow, ...]
    preserved: tuple[ImpactRow, ...]
    unknown_boundaries: tuple[str, ...]
    checks_to_run: tuple[Ref, ...]
    admissions_to_revoke: tuple[Ref, ...]
    result_digest: str


@dataclass(frozen=True, slots=True)
class DecisionRecord(_RecordMixin):
    ref: Ref
    question: str
    proposal_ref: Ref | None
    baseline_ref: Ref
    answer: str
    reason: str
    basis_refs: tuple[Ref, ...]
    alternatives: tuple[str, ...]
    status: DecisionStatus
    basis_broken: bool = False

    def __post_init__(self) -> None:
        for name in ("question", "answer", "reason"):
            object.__setattr__(self, name, _text(getattr(self, name), f"decision.{name}"))
        object.__setattr__(self, "basis_refs", _refs(self.basis_refs))
        object.__setattr__(self, "alternatives", _strings(self.alternatives))


@dataclass(frozen=True, slots=True)
class ChangeEnactment(_RecordMixin):
    proposal_ref: Ref
    old_baseline_ref: Ref
    new_baseline_ref: Ref
    impact_ref: Ref
    revoked_admission_refs: tuple[Ref, ...]
    rework_package_refs: tuple[Ref, ...]
    preserved_package_refs: tuple[Ref, ...]


Record: TypeAlias = (
    RequirementNode | Component | Contract | AcceptanceCriterion | Journey | WorkPackage |
    IntegrationObligation | DesignBaseline | ImplementationBinding
)


class BuildGraph:
    """One typed build graph plus projections owned by existing v46 organs."""

    def __init__(self, scope: Scope, *, timeline: TimelineIndex | None = None,
                 dependency_graph: LedgerIndex | None = None) -> None:
        self.scope = scope
        self.timeline = timeline or TimelineIndex(session_id=scope.work_id)
        self.dependency_graph = dependency_graph or LedgerIndex()
        self.nodes: dict[str, Record] = {}
        self.edges: set[tuple[str, str, str]] = set()
        self.trace_links: dict[str, TraceLink] = {}
        self.receipts: dict[str, ExecutionReceipt] = {}
        self.gates: dict[str, GateEvaluation] = {}
        self.decisions: dict[str, DecisionRecord] = {}
        self.changes: dict[str, ChangeProposal] = {}
        self.impacts: dict[str, ImpactAnalysis] = {}
        self.baselines: dict[str, DesignBaseline] = {}
        self.current_baseline_ref: Ref | None = None
        self.current_candidate_ref: Ref | None = None
        self.current_target_ref: str | None = None
        self.owner_epoch = 1
        self.watermark = 0
        self._stale_receipts: dict[str, str] = {}
        self._design_invalid: set[str] = set()
        self._preserved_package_keys: set[str] = set()
        self._receipt_counter = 0
        self._decision_counter = 0
        self._current_admission: BuildAdmission | None = None
        self._refs_by_id: dict[tuple[str, int], Ref] = {}

    def ref(self, node_id: str, revision: int = 1) -> Ref:
        node_id = _text(node_id, "node_id")
        key = (node_id, revision)
        existing = self._refs_by_id.get(key)
        if existing is not None:
            return existing
        reference = Ref(self.scope, node_id, revision, _digest({"scope": self.scope.key, "node_id": node_id, "revision": revision}))
        self._refs_by_id[key] = reference
        return reference

    def _check_scope(self, reference: Ref) -> None:
        if reference.scope != self.scope:
            raise BuildModeError(f"cross-scope reference rejected: {reference.key}")

    def _record(self, record: Record) -> None:
        self._check_scope(record.ref)
        key = record.ref.key
        previous = self.nodes.get(key)
        if previous is not None and _digest(previous.to_dict()) != _digest(record.to_dict()):
            raise IntegrityConflict(f"node identity reused with a different payload: {key}")
        self.nodes[key] = record
        if isinstance(record, DesignBaseline):
            self.baselines[key] = record
        self.watermark += 1

    def add(self, record: Record) -> Record:
        """Admit one host-validated typed graph node."""
        self._record(record)
        self.timeline.append(
            "design_node", {"node_id": record.ref.node_id, "revision": record.ref.revision},
            actor="host", entities=(record.ref.node_id,), turn=self.watermark,
            content=f"admitted {type(record).__name__} {record.ref.node_id}",
            persist=False,
        )
        return record

    def get(self, reference: Ref) -> Record | ExecutionReceipt | GateEvaluation | DecisionRecord | None:
        self._check_scope(reference)
        return self.nodes.get(reference.key) or self.receipts.get(reference.key) or self.gates.get(reference.key) or self.decisions.get(reference.key)

    def add_edge(self, consumer: Ref, premise: Ref, relation: EdgeRelation | str) -> None:
        self._check_scope(consumer)
        self._check_scope(premise)
        if consumer.key not in self.nodes or premise.key not in self.nodes:
            raise BuildModeError("graph edge endpoints must be admitted first")
        relation_text = str(relation.value if isinstance(relation, EdgeRelation) else relation)
        edge = (consumer.key, relation_text, premise.key)
        if edge in self.edges:
            return
        if relation_text in {EdgeRelation.DEPENDS_ON.value, EdgeRelation.GOVERNED_BY.value}:
            if self._would_cycle(consumer.key, premise.key):
                raise BuildModeError("support/build dependency cycle rejected")
        self.edges.add(edge)
        self.watermark += 1

    def _would_cycle(self, consumer: str, premise: str) -> bool:
        stack = [premise]
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current == consumer:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(right for left, relation, right in self.edges
                         if relation in {EdgeRelation.DEPENDS_ON.value, EdgeRelation.GOVERNED_BY.value}
                         and left == current)
        return False

    def add_trace(self, source: Ref, relation: EdgeRelation | str, target: Ref, admission_receipt_id: str) -> TraceLink:
        self._check_scope(source)
        self._check_scope(target)
        if source.key not in self.nodes and source.key not in self.receipts:
            raise BuildModeError("trace source is not admitted")
        if target.key not in self.nodes and target.key not in self.receipts:
            raise BuildModeError("trace target is not admitted")
        relation_text = relation.value if isinstance(relation, EdgeRelation) else _text(relation, "trace.relation")
        receipt_id = _text(admission_receipt_id, "trace.admission_receipt_id")
        link = TraceLink(
            self.ref(f"trace:{source.node_id}:{relation_text}:{target.node_id}:{receipt_id}", 1),
            source, relation_text, target, receipt_id,
        )
        self.trace_links[link.ref.key] = link
        self.watermark += 1
        return link

    def admit_design(self, baseline: DesignBaseline) -> ExecutionReceipt:
        if self.current_baseline_ref is not None and baseline.replaces_ref != self.current_baseline_ref:
            raise StaleBasis("design baseline must replace the current baseline")
        self.add(baseline)
        self.current_baseline_ref = baseline.ref
        receipt = self.emit_receipt(
            kind="DESIGN_ADMISSION", subject_ref=baseline.ref, package_ref=None,
            facet=Facet.DESIGNED, baseline_ref=baseline.ref,
            candidate_ref=baseline.ref, target_ref=baseline.target_refs[0],
            outcome=ReceiptOutcome.PASS, assertion_ids=("design:admitted",),
            event_ids=("design_admitted",), reached_boundaries=("design_graph",),
            coverage="COMPLETE", source_paths=(),
        )
        return receipt

    def _baseline(self) -> DesignBaseline:
        if self.current_baseline_ref is None:
            raise BuildModeError("no current design baseline")
        baseline = self.baselines.get(self.current_baseline_ref.key)
        if baseline is None:
            raise BuildModeError("current baseline is not admitted")
        return baseline

    def _packages(self, baseline: DesignBaseline | None = None) -> tuple[WorkPackage, ...]:
        selected = baseline or self._baseline()
        result: list[WorkPackage] = []
        for reference in selected.package_refs:
            package = self.nodes.get(reference.key)
            if isinstance(package, WorkPackage):
                result.append(package)
        return tuple(result)

    def emit_receipt(
        self,
        *,
        kind: str,
        subject_ref: Ref,
        package_ref: Ref | None,
        facet: Facet | None,
        baseline_ref: Ref,
        candidate_ref: Ref,
        target_ref: str,
        outcome: ReceiptOutcome,
        assertion_ids: Sequence[str],
        event_ids: Sequence[str],
        reached_boundaries: Sequence[str],
        coverage: str,
        source_paths: Sequence[str],
        environment_digest: str = "env:synthetic",
        config_digest: str = "config:default",
        input_digest: str = "input:synthetic",
        fired: bool = False,
    ) -> ExecutionReceipt:
        self._check_scope(subject_ref)
        self._check_scope(baseline_ref)
        self._check_scope(candidate_ref)
        if package_ref is not None:
            self._check_scope(package_ref)
        self._receipt_counter += 1
        receipt_id = f"receipt:{self.scope.work_id}:{self._receipt_counter}:{kind.lower()}"
        payload = {
            "receipt_id": receipt_id, "kind": kind, "subject": subject_ref.key,
            "package": package_ref.key if package_ref else None, "facet": facet.value if facet else None,
            "baseline": baseline_ref.key, "candidate": candidate_ref.key, "target": target_ref,
            "environment": environment_digest, "config": config_digest, "input": input_digest,
            "outcome": outcome.value, "assertions": list(assertion_ids), "events": list(event_ids),
            "boundaries": list(reached_boundaries), "paths": list(source_paths), "coverage": coverage,
            "fired": fired,
        }
        authenticity = _digest(payload)
        receipt = ExecutionReceipt(
            ref=self.ref(receipt_id), receipt_id=receipt_id, kind=_text(kind, "receipt.kind"),
            subject_ref=subject_ref, package_ref=package_ref, facet=facet,
            baseline_ref=baseline_ref, candidate_ref=candidate_ref, target_ref=_text(target_ref, "receipt.target_ref"),
            environment_digest=environment_digest, config_digest=config_digest, input_digest=input_digest,
            outcome=outcome, assertion_ids=tuple(assertion_ids), event_ids=tuple(event_ids),
            reached_boundaries=tuple(reached_boundaries), source_paths=tuple(source_paths),
            coverage=_text(coverage, "receipt.coverage"), host_issued=True,
            authenticity_digest=authenticity, fired=fired,
        )
        self.accept_receipt(receipt)
        self.timeline.append(
            "receipt", {"receipt_id": receipt_id, "kind": kind, "outcome": outcome.value,
                         "package": package_ref.node_id if package_ref else None},
            actor="host", turn=self.watermark, entities=tuple(
                item for item in (subject_ref.node_id, package_ref.node_id if package_ref else "") if item
            ), content=f"{kind} {outcome.value} {receipt_id}", persist=False,
        )
        return receipt

    @staticmethod
    def _receipt_payload(receipt: ExecutionReceipt) -> dict[str, object]:
        return {
            "receipt_id": receipt.receipt_id, "kind": receipt.kind,
            "subject": receipt.subject_ref.key,
            "package": receipt.package_ref.key if receipt.package_ref else None,
            "facet": receipt.facet.value if receipt.facet else None,
            "baseline": receipt.baseline_ref.key, "candidate": receipt.candidate_ref.key,
            "target": receipt.target_ref, "environment": receipt.environment_digest,
            "config": receipt.config_digest, "input": receipt.input_digest,
            "outcome": receipt.outcome.value, "assertions": list(receipt.assertion_ids),
            "events": list(receipt.event_ids), "boundaries": list(receipt.reached_boundaries),
            "paths": list(receipt.source_paths), "coverage": receipt.coverage,
            "fired": receipt.fired,
        }

    def accept_receipt(self, receipt: ExecutionReceipt) -> ExecutionReceipt:
        """Accept only an intact host receipt; product output cannot mint evidence."""
        self._check_scope(receipt.ref)
        expected = _digest(self._receipt_payload(receipt))
        if not receipt.host_issued or receipt.authenticity_digest != expected:
            raise BuildModeError("receipt authenticity or host provenance is invalid")
        previous = self.receipts.get(receipt.ref.key)
        if previous is not None and previous != receipt:
            raise IntegrityConflict("receipt identity reused with a different payload")
        self.receipts[receipt.ref.key] = receipt
        self.watermark += 1
        if receipt.package_ref is not None:
            self.add_trace(receipt.package_ref, EdgeRelation.OBSERVED_BY, receipt.ref, receipt.receipt_id)
            package = self.nodes.get(receipt.package_ref.key)
            if isinstance(package, WorkPackage):
                for acceptance_ref in package.acceptance_refs:
                    if acceptance_ref.key in self.nodes:
                        self.add_trace(acceptance_ref, EdgeRelation.CHECKED_BY, receipt.ref, receipt.receipt_id)
        self._index_receipt(receipt)
        return receipt

    def _index_receipt(self, receipt: ExecutionReceipt) -> None:
        if receipt.package_ref is None or not receipt.source_paths:
            return
        if receipt.outcome not in {ReceiptOutcome.PASS, ReceiptOutcome.FAIL}:
            return
        kind = "test" if receipt.facet in {Facet.VERIFIED, Facet.WIRED_IN, Facet.LIVE} else "read"
        result = "GREEN" if receipt.outcome is ReceiptOutcome.PASS else "RED"
        self.dependency_graph.observe_receipt({
            "receipt_id": receipt.receipt_id,
            "kind": kind,
            "result_class": result,
            "target_path": receipt.source_paths[0],
            "workspace_revision": receipt.candidate_ref.revision,
            "dependencies": receipt.source_paths[1:],
            "turn": self.watermark,
        })

    def _receipt_matches(self, receipt: ExecutionReceipt, package: WorkPackage,
                         facet: Facet, candidate_ref: Ref, baseline_ref: Ref,
                         target_ref: str) -> bool:
        baseline_compatible = (
            receipt.baseline_ref == baseline_ref
            or (
                receipt.baseline_ref.scope == baseline_ref.scope
                and receipt.baseline_ref.revision < baseline_ref.revision
                and package.ref.node_id not in self._design_invalid
            )
        )
        candidate_compatible = (
            receipt.candidate_ref == candidate_ref
            or (
                package.ref.key in self._preserved_package_keys
                and receipt.baseline_ref != baseline_ref
                and receipt.baseline_ref.revision < baseline_ref.revision
            )
        )
        return (
            receipt.package_ref == package.ref and receipt.facet is facet
            and receipt.outcome is ReceiptOutcome.PASS and candidate_compatible
            and baseline_compatible and receipt.target_ref == target_ref
            and receipt.ref.key not in self._stale_receipts
            and receipt.host_issued and receipt.coverage in {"COMPLETE", "CONSERVATIVE"}
        )

    def _receipts_for(self, package: WorkPackage, facet: Facet, candidate_ref: Ref,
                      baseline_ref: Ref, target_ref: str) -> tuple[ExecutionReceipt, ...]:
        return tuple(receipt for receipt in self.receipts.values()
                     if self._receipt_matches(receipt, package, facet, candidate_ref, baseline_ref, target_ref))

    def assess_package(self, package_ref: Ref, *, candidate_ref: Ref | None = None,
                       target_ref: str | None = None) -> BuildAssessment:
        baseline = self._baseline()
        candidate = candidate_ref or self.current_candidate_ref
        target = target_ref or self.current_target_ref
        if candidate is None or target is None:
            raise BuildModeError("assessment needs a current candidate and target")
        package = self.nodes.get(package_ref.key)
        if not isinstance(package, WorkPackage):
            raise BuildModeError("assessment package is not admitted")
        facets: dict[str, Assessment] = {}
        for facet in Facet:
            if facet is Facet.DESIGNED:
                if package.ref not in baseline.package_refs or package.ref.node_id in self._design_invalid:
                    facets[facet.value] = Assessment(AssessmentValue.UNKNOWN, Freshness.STALE, (), ("DESIGN_REWORK_REQUIRED",))
                else:
                    facets[facet.value] = Assessment(AssessmentValue.SATISFIED, Freshness.CURRENT, ("design:admitted",), ())
                continue
            matching = self._receipts_for(package, facet, candidate, baseline.ref, target)
            if matching:
                facets[facet.value] = Assessment(
                    AssessmentValue.SATISFIED, Freshness.CURRENT,
                    tuple(item.receipt_id for item in matching), (),
                )
            else:
                historical = tuple(receipt.receipt_id for receipt in self.receipts.values()
                                   if receipt.package_ref == package.ref and receipt.facet is facet
                                   and receipt.outcome is ReceiptOutcome.PASS)
                if historical:
                    facets[facet.value] = Assessment(
                        AssessmentValue.UNKNOWN, Freshness.STALE, historical, ("RECEIPT_BINDING_STALE",),
                    )
                else:
                    facets[facet.value] = Assessment(
                        AssessmentValue.UNKNOWN, Freshness.UNASSESSED, (), (f"MISSING_{facet.value}",),
                    )
        fire_refs = tuple(receipt.receipt_id for receipt in self.receipts.values()
                           if receipt.package_ref == package.ref and receipt.fired
                           and receipt.outcome is ReceiptOutcome.PASS
                           and (
                               receipt.candidate_ref == candidate
                               or (
                                   package.ref.key in self._preserved_package_keys
                                   and receipt.baseline_ref != baseline.ref
                                   and receipt.baseline_ref.revision < baseline.ref.revision
                               )
                           )
                           and (
                               receipt.baseline_ref == baseline.ref
                               or (
                                   package.ref.key in self._preserved_package_keys
                                   and receipt.baseline_ref.revision < baseline.ref.revision
                               )
                           )
                           and receipt.target_ref == target and receipt.ref.key not in self._stale_receipts)
        missing = tuple(
            f"{package.name}:{facet.value}"
            for facet in Facet
            if facets[facet.value].value is not AssessmentValue.SATISFIED
        )
        if not fire_refs:
            missing += (f"{package.name}:FIRED",)
        return BuildAssessment(package.ref, baseline.ref, candidate, target, facets, fire_refs, missing)

    def ledger(self, *, candidate_ref: Ref | None = None, target_ref: str | None = None) -> BuildLedger:
        baseline = self._baseline()
        candidate = candidate_ref or self.current_candidate_ref
        target = target_ref or self.current_target_ref
        if candidate is None or target is None:
            raise BuildModeError("ledger needs a current candidate and target")
        assessments = tuple(self.assess_package(package.ref, candidate_ref=candidate, target_ref=target)
                            for package in self._packages(baseline))
        all_assessments = [assessment.facets[facet.value] for assessment in assessments for facet in Facet]
        satisfied = sum(item.value is AssessmentValue.SATISFIED for item in all_assessments)
        stale = sum(item.freshness is Freshness.STALE for item in all_assessments)
        failed = sum(item.value is AssessmentValue.UNSATISFIED for item in all_assessments)
        unknown = len(all_assessments) - satisfied - failed
        done = bool(assessments) and all(item.live_and_fired for item in assessments)
        reason = "LIVE_AND_FIRED" if done else "REQUIRED_LIVE_AND_FIRED_EVIDENCE_MISSING"
        return BuildLedger(
            self.scope.work_id, baseline.ref, candidate, target, assessments, len(all_assessments),
            satisfied, stale, unknown, failed, done, reason,
        )

    def evaluate_gate(self, gate: Gate, baseline_ref: Ref | None = None) -> GateEvaluation:
        baseline = self.baselines.get((baseline_ref or self._baseline().ref).key)
        if baseline is None:
            raise BuildModeError("gate baseline is not admitted")
        missing: list[str] = []
        evidence: list[str] = []
        required = 1
        if gate is Gate.G0:
            required = len(baseline.requirement_refs)
            for reference in baseline.requirement_refs:
                node = self.nodes.get(reference.key)
                if not isinstance(node, RequirementNode) or not node.admitted:
                    missing.append(f"REQUIREMENT_UNADMITTED:{reference.node_id}")
                else:
                    evidence.extend(node.source_refs)
            if not baseline.source_dispositions_complete:
                missing.append("SOURCE_DISPOSITION_INCOMPLETE")
        elif gate is Gate.G1:
            required = len(baseline.requirement_refs) + len(baseline.package_refs) + len(baseline.integration_refs)
            missing.extend(self._consistency_findings(baseline))
        elif gate is Gate.G2:
            components = [self.nodes.get(reference.key) for reference in baseline.component_refs]
            required = len(components)
            for component in components:
                if not isinstance(component, Component) or not component.reuse_evidence_refs:
                    missing.append("REUSE_UNASSESSED")
                else:
                    evidence.extend(component.reuse_evidence_refs)
        elif gate is Gate.G3:
            criteria = [self.nodes.get(reference.key) for reference in baseline.acceptance_refs]
            required = len(criteria)
            for criterion in criteria:
                if not isinstance(criterion, AcceptanceCriterion):
                    missing.append("ACCEPTANCE_MISSING")
                elif not criterion.check_id or not criterion.negative_control_id:
                    missing.append(f"VACUOUS_CHECK:{criterion.ref.node_id}")
                else:
                    evidence.append(criterion.check_id)
            if not baseline.check_specs_complete:
                missing.append("CHECK_SPEC_INCOMPLETE")
        elif gate is Gate.G4:
            validation_receipts = self._design_receipts(baseline.ref, {"DESIGN_VALIDATION", "RISK_SPIKE"})
            required = max(1, len(baseline.integration_refs))
            integration_ids = {reference.node_id for reference in baseline.integration_refs}
            covered_boundaries = {
                boundary for receipt in validation_receipts
                for boundary in receipt.reached_boundaries
                if boundary in integration_ids
            }
            missing_boundaries = integration_ids - covered_boundaries
            if missing_boundaries:
                missing.append(
                    f"BOUNDARY_UNREACHED:{len(covered_boundaries)}/{required}:"
                    + ",".join(sorted(missing_boundaries))
                )
            else:
                evidence.extend(item.receipt_id for item in validation_receipts)
        elif gate is Gate.G5:
            review_receipts = self._design_receipts(baseline.ref, {"ADVERSARIAL_REVIEW"})
            required = 1
            if not review_receipts:
                missing.append("REVIEW_INCOMPLETE")
            else:
                evidence.extend(item.receipt_id for item in review_receipts)
        else:
            prior = [self.evaluate_gate(item, baseline.ref) for item in Gate if item is not Gate.G6]
            required = len(prior)
            missing.extend(code for item in prior if item.result is not GateResult.SATISFIED for code in item.reason_codes)
            evidence.extend(item.ref.node_id for item in prior if item.result is GateResult.SATISFIED)
            current_basis = set(baseline.contract_refs) | set(baseline.requirement_refs)
            if any(
                item.basis_broken and item.baseline_ref == baseline.ref
                and bool(set(item.basis_refs).intersection(current_basis))
                for item in self.decisions.values()
            ):
                missing.append("BASIS_BROKEN")
        result = GateResult.SATISFIED if not missing else GateResult.UNKNOWN
        if any(code.startswith(("TYPE_", "SUPPORT_CYCLE", "CROSS_SCOPE")) for code in missing):
            result = GateResult.VIOLATED
        evaluation = GateEvaluation(
            self.ref(f"gate:{baseline.ref.node_id}:{gate.value}"), gate, baseline.ref, result,
            tuple(dict.fromkeys(missing)), tuple(dict.fromkeys(evidence)), required,
            max(0, required - len(missing)), _digest({"baseline": baseline.ref.key, "gate": gate.value, "missing": missing}),
        )
        self.gates[evaluation.ref.key] = evaluation
        return evaluation

    def _design_receipts(self, baseline_ref: Ref, kinds: set[str]) -> tuple[ExecutionReceipt, ...]:
        return tuple(receipt for receipt in self.receipts.values()
                     if receipt.baseline_ref == baseline_ref and receipt.kind in kinds
                     and receipt.outcome is ReceiptOutcome.PASS and receipt.ref.key not in self._stale_receipts)

    def _consistency_findings(self, baseline: DesignBaseline) -> list[str]:
        findings: list[str] = []
        requirement_set = set(baseline.requirement_refs)
        packages_by_key: dict[str, WorkPackage] = {}
        for reference in baseline.package_refs:
            package = self.nodes.get(reference.key)
            if not isinstance(package, WorkPackage):
                findings.append(f"UNCOVERED_PACKAGE:{reference.node_id}")
                continue
            packages_by_key[package.ref.key] = package
            if not set(package.requirement_refs).intersection(requirement_set):
                findings.append(f"ORPHAN_PACKAGE:{package.ref.node_id}")
            if any(dependency == package.ref for dependency in package.dependency_refs):
                findings.append(f"SUPPORT_CYCLE:{package.ref.node_id}")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(package: WorkPackage) -> None:
            if package.ref.key in visiting:
                findings.append(f"SUPPORT_CYCLE:{package.ref.node_id}")
                return
            if package.ref.key in visited:
                return
            visiting.add(package.ref.key)
            for dependency_ref in package.dependency_refs:
                dependency = packages_by_key.get(dependency_ref.key)
                if dependency is not None:
                    visit(dependency)
            visiting.remove(package.ref.key)
            visited.add(package.ref.key)

        for package in packages_by_key.values():
            visit(package)
        for reference in baseline.requirement_refs:
            requirement = self.nodes.get(reference.key)
            if not isinstance(requirement, RequirementNode):
                findings.append(f"UNCOVERED_REQUIREMENT:{reference.node_id}")
            elif not requirement.acceptance_refs or not requirement.package_refs:
                findings.append(f"UNCOVERED_OBSERVATION:{reference.node_id}")
        for reference in baseline.integration_refs:
            integration = self.nodes.get(reference.key)
            if not isinstance(integration, IntegrationObligation):
                findings.append(f"INTEGRATION_MISSING:{reference.node_id}")
            elif not integration.required or not integration.effect_ids:
                findings.append(f"INTEGRATION_UNRESOLVED:{reference.node_id}")
        return findings

    def definition_of_ready(self, baseline_ref: Ref | None = None) -> tuple[GateEvaluation, ...]:
        baseline = baseline_ref or self._baseline().ref
        return tuple(self.evaluate_gate(gate, baseline) for gate in Gate)

    def admit_build(self, *, candidate_ref: Ref, target_ref: str, owner_epoch: int | None = None) -> BuildAdmission:
        baseline = self._baseline()
        self._check_scope(candidate_ref)
        if owner_epoch is not None and owner_epoch != self.owner_epoch:
            raise StaleBasis("owner epoch is stale")
        evaluations = self.definition_of_ready(baseline.ref)
        if any(item.result is not GateResult.SATISFIED for item in evaluations):
            missing = sorted({code for item in evaluations for code in item.reason_codes})
            raise BuildModeError("BUILD_READY refused: " + ",".join(missing))
        basis = _digest({
            "baseline": baseline.ref.to_dict(),
            "gates": [item.input_digest for item in evaluations],
            "watermark": self.watermark,
        })
        self.current_candidate_ref = candidate_ref
        self.current_target_ref = _text(target_ref, "target_ref")
        admission = BuildAdmission(
            self.ref(f"admission:{baseline.ref.node_id}:{candidate_ref.node_id}"), baseline.ref,
            baseline.package_refs, basis, self.watermark, self.owner_epoch, "session-bound",
        )
        self.timeline.append("stage_change", {"stage": "BUILD_READY", "admission": admission.ref.node_id},
                             actor="host", turn=self.watermark, entities=(baseline.ref.node_id,),
                              content="BUILD_READY admitted", persist=False)
        self._current_admission = admission
        return admission

    def bind_implementation(self, binding: ImplementationBinding) -> ImplementationBinding:
        self.add(binding)
        self.add_trace(binding.design_ref, "IMPLEMENTED_BY", binding.ref, "binding-admission")
        return binding

    def package_state(self, package_ref: Ref) -> tuple[PackageWorkflow, BuildAssessment]:
        assessment = self.assess_package(package_ref)
        if any(value.freshness is Freshness.STALE for value in assessment.facets.values()):
            return PackageWorkflow.REWORK, assessment
        if assessment.live_and_fired:
            return PackageWorkflow.ACCEPTED, assessment
        package = self.nodes.get(package_ref.key)
        if not isinstance(package, WorkPackage):
            raise BuildModeError("package is not admitted")
        for dependency in package.dependency_refs:
            dependency_state, dependency_assessment = self.package_state(dependency)
            if dependency_state not in {PackageWorkflow.ACCEPTED, PackageWorkflow.ACTIVE}:
                return PackageWorkflow.BLOCKED, assessment
            if dependency_assessment.facets[Facet.BUILT.value].value is not AssessmentValue.SATISFIED:
                return PackageWorkflow.BLOCKED, assessment
        if assessment.facets[Facet.DESIGNED.value].value is AssessmentValue.SATISFIED:
            return PackageWorkflow.READY, assessment
        return PackageWorkflow.BLOCKED, assessment

    def start_package(self, package_ref: Ref) -> str:
        if self.current_candidate_ref is None or self.current_target_ref is None:
            raise BuildModeError("package start needs an admitted build admission")
        admission = self._current_admission
        if (
            admission is None
            or admission.baseline_ref != self._baseline().ref
            or package_ref not in admission.allowed_package_refs
            or admission.owner_epoch != self.owner_epoch
        ):
            raise StaleBasis("package start is outside the current build admission")
        workflow, _assessment = self.package_state(package_ref)
        if workflow not in {PackageWorkflow.READY, PackageWorkflow.ACTIVE}:
            raise BuildModeError(f"package is not ready: {package_ref.node_id}")
        self.timeline.append("stage_change", {"stage": "B4_IMPLEMENT", "package": package_ref.node_id},
                             actor="host", turn=self.watermark, entities=(package_ref.node_id,),
                             content=f"started package {package_ref.node_id}", persist=False)
        return f"attempt:{package_ref.node_id}:{self.current_candidate_ref.node_id}"

    def fire(self, *, package_ref: Ref, journey_ref: Ref, entrypoint: str,
             boundaries: Sequence[str], observable: str) -> ExecutionReceipt:
        baseline = self._baseline()
        candidate = self.current_candidate_ref
        target = self.current_target_ref
        if candidate is None or target is None:
            raise BuildModeError("fire needs an admitted build candidate")
        assessment = self.assess_package(package_ref)
        if assessment.facets[Facet.LIVE.value].value is not AssessmentValue.SATISFIED:
            receipt_bindings = tuple(
                (receipt.receipt_id, receipt.package_ref.key if receipt.package_ref else None,
                 receipt.candidate_ref.key, receipt.baseline_ref.key, receipt.facet.value if receipt.facet else None)
                for receipt in self.receipts.values() if receipt.package_ref == package_ref
            )
            raise BuildModeError(
                "LIVE receipt is required before fired behavior: "
                + ",".join(assessment.missing_predicates)
                + f" (candidate={assessment.candidate_ref.key}, baseline={assessment.baseline_ref.key})"
                + f" receipts={receipt_bindings}"
            )
        journey = self.nodes.get(journey_ref.key)
        if not isinstance(journey, Journey) or package_ref not in journey.package_refs:
            raise BuildModeError("journey does not cover package")
        required_boundaries = (entrypoint, *tuple(boundaries), observable)
        if len(set(required_boundaries)) != len(required_boundaries):
            raise BuildModeError("fire trace must name distinct boundaries")
        return self.emit_receipt(
            kind="FIRE", subject_ref=journey_ref, package_ref=package_ref,
            facet=Facet.LIVE, baseline_ref=baseline.ref, candidate_ref=candidate,
            target_ref=target, outcome=ReceiptOutcome.PASS,
            assertion_ids=tuple(criterion.node_id for criterion in journey.acceptance_refs),
            event_ids=(f"fired:{journey_ref.node_id}",), reached_boundaries=required_boundaries,
            coverage="COMPLETE", source_paths=(), fired=True,
        )

    def propose_change(
        self, *, kind: ChangeKind, old_refs: Sequence[Ref], new_refs: Sequence[Ref],
        replacement_package_refs: Sequence[Ref], replacement_acceptance_refs: Sequence[Ref],
        replacement_integration_refs: Sequence[Ref] = (),
        replacement_journey_refs: Sequence[Ref] = (),
        source_message_ref: str, source_paths: Sequence[str], rationale: str,
        opportunity_only: bool = False,
        replacement_requirement_refs: Sequence[Ref] = (),
    ) -> ChangeProposal:
        baseline = self._baseline()
        for reference in (*old_refs, *new_refs, *replacement_package_refs,
                          *replacement_acceptance_refs, *replacement_integration_refs,
                          *replacement_journey_refs, *replacement_requirement_refs):
            self._check_scope(reference)
        proposal = ChangeProposal(
            self.ref(f"change:{self.watermark + 1}"), kind, baseline.ref,
            tuple(old_refs), tuple(new_refs), tuple(replacement_package_refs),
            tuple(replacement_acceptance_refs), tuple(replacement_integration_refs),
            tuple(replacement_journey_refs), source_message_ref, tuple(source_paths), rationale, opportunity_only,
            tuple(replacement_requirement_refs),
        )
        self.changes[proposal.ref.key] = proposal
        self.timeline.append("change_proposal", proposal.to_dict(), actor="user", turn=self.watermark,
                             entities=tuple(reference.node_id for reference in (*proposal.old_refs, *proposal.new_refs)),
                             content=rationale, persist=False)
        return proposal

    def analyze_change(self, proposal_ref: Ref) -> ImpactAnalysis:
        proposal = self.changes.get(proposal_ref.key)
        if proposal is None:
            raise BuildModeError("unknown change proposal")
        if proposal.expected_baseline_ref != self._baseline().ref:
            raise StaleBasis("change preview baseline is stale")
        roots = {reference.key for reference in (*proposal.old_refs, *proposal.new_refs)}
        affected_keys: set[str] = set(roots)
        queue: deque[str] = deque(roots)
        while queue:
            premise = queue.popleft()
            for consumer, relation, edge_premise in self.edges:
                if edge_premise == premise and relation in {
                    EdgeRelation.DEPENDS_ON.value, EdgeRelation.CONSUMES.value,
                    EdgeRelation.IMPLEMENTS.value, EdgeRelation.VERIFIES.value,
                    EdgeRelation.GOVERNED_BY.value, EdgeRelation.COMPOSES.value,
                } and consumer not in affected_keys:
                    affected_keys.add(consumer)
                    queue.append(consumer)
        affected_packages = [node for key, node in self.nodes.items()
                             if key in affected_keys and isinstance(node, WorkPackage)]
        affected_package_keys = {package.ref.key for package in affected_packages}
        affected: list[ImpactRow] = []
        preserved: list[ImpactRow] = []
        for package in self._packages():
            if package.ref.key in affected_package_keys:
                disposition = ImpactDisposition.REWORK_REQUIRED if proposal.kind is not ChangeKind.ENVIRONMENT_CHANGE else ImpactDisposition.REVERIFY_REQUIRED
                affected.append(ImpactRow(package.ref, tuple(Facet), disposition, tuple(proposal.old_refs), "DEPGRAPH_REVERSE_CLOSURE"))
            else:
                preserved.append(ImpactRow(package.ref, tuple(Facet), ImpactDisposition.PRESERVED, (), "NO_MATERIAL_PATH"))
        if proposal.opportunity_only or proposal.kind is ChangeKind.OPPORTUNITY:
            affected = []
        unknown = () if self.edges else ("DEPENDENCY_COVERAGE_UNKNOWN",)
        checks = tuple(
            criterion.ref for package in affected_packages
            for criterion in (self.nodes.get(reference.key) for reference in package.acceptance_refs)
            if isinstance(criterion, AcceptanceCriterion)
        )
        admissions = (self.ref(f"admission:{self._baseline().ref.node_id}:{self.current_candidate_ref.node_id}"),) if self.current_candidate_ref else ()
        payload = {
            "proposal": proposal.ref.key, "watermark": self.watermark,
            "affected": [item.subject_ref.key for item in affected],
            "preserved": [item.subject_ref.key for item in preserved], "unknown": unknown,
        }
        impact = ImpactAnalysis(
            self.ref(f"impact:{proposal.ref.node_id}"), proposal.ref, self.watermark,
            "COMPLETE_DECLARED" if not unknown else "UNKNOWN", tuple(affected), tuple(preserved),
            unknown, checks, admissions, _digest(payload),
        )
        self.impacts[impact.ref.key] = impact
        self.timeline.append("impact_analysis", payload, actor="host", turn=self.watermark,
                             entities=tuple(item.subject_ref.node_id for item in affected),
                             content=f"impact {proposal.ref.node_id}: {len(affected)} packages rework", persist=False)
        return impact

    def record_decision(self, *, question: str, proposal_ref: Ref | None, answer: str,
                        reason: str, basis_refs: Sequence[Ref], alternatives: Sequence[str],
                        status: DecisionStatus = DecisionStatus.SUPPORTED) -> DecisionRecord:
        self._decision_counter += 1
        decision = DecisionRecord(
            self.ref(f"decision:{self._decision_counter}"), question, proposal_ref,
            self._baseline().ref, answer, reason, tuple(basis_refs), tuple(alternatives), status,
        )
        self.decisions[decision.ref.key] = decision
        self.timeline.record_decision(decision.ref.node_id, reason, turn=self.watermark,
                                      entities=tuple(reference.node_id for reference in basis_refs),
                                      prompted_by=(), persist=False)
        return decision

    def enact_change(self, *, proposal_ref: Ref, impact_ref: Ref,
                     decision_ref: Ref) -> ChangeEnactment:
        proposal = self.changes.get(proposal_ref.key)
        impact = self.impacts.get(impact_ref.key)
        decision = self.decisions.get(decision_ref.key)
        old_baseline = self._baseline()
        if proposal is None or impact is None or decision is None:
            raise BuildModeError("change enactment needs proposal, impact, and decision")
        if proposal.expected_baseline_ref != old_baseline.ref or impact.proposal_ref != proposal.ref:
            raise StaleBasis("change enactment basis changed")
        if decision.status is not DecisionStatus.SUPPORTED:
            raise BuildModeError("unsupported decision cannot enact a change")
        affected_old = {item.subject_ref.key for item in impact.affected}
        if proposal.opportunity_only:
            return ChangeEnactment(proposal.ref, old_baseline.ref, old_baseline.ref, impact.ref, (), (),
                                   tuple(item.subject_ref for item in impact.preserved))
        for existing in self.decisions.values():
            if set(existing.basis_refs).intersection(proposal.old_refs):
                self.decisions[existing.ref.key] = replace(existing, status=DecisionStatus.SUSPECT, basis_broken=True)
        for receipt in tuple(self.receipts.values()):
            if receipt.package_ref is not None and receipt.package_ref.key in affected_old:
                self._stale_receipts[receipt.ref.key] = "MATERIAL_CHANGE"
        for path in proposal.source_paths:
            self.dependency_graph.observe_receipt({
                "receipt_id": f"change:{proposal.ref.node_id}:{path}", "kind": "edit",
                "result_class": "APPLIED", "target_path": path,
                "workspace_revision": old_baseline.ref.revision, "turn": self.watermark,
            })
        replacement_packages = tuple(
            package for reference in proposal.replacement_package_refs
            if isinstance(package := self.nodes.get(reference.key), WorkPackage)
        )
        old_to_new = {package.replaces_ref.key: package.ref for package in replacement_packages
                      if package.replaces_ref is not None}
        package_refs = tuple(old_to_new.get(reference.key, reference) for reference in old_baseline.package_refs)
        contract_replacements = tuple(
            reference for reference in proposal.new_refs
            if isinstance(self.nodes.get(reference.key), Contract)
        )
        replaced_contract_ids = {
            reference.node_id.rsplit("@", 1)[0]
            for reference in proposal.old_refs if reference in old_baseline.contract_refs
        }
        contract_refs = tuple(
            reference for reference in old_baseline.contract_refs
            if reference.node_id.rsplit("@", 1)[0] not in replaced_contract_ids
        ) + contract_replacements
        requirement_refs = tuple(proposal.replacement_requirement_refs) or old_baseline.requirement_refs
        acceptance_refs = tuple(proposal.replacement_acceptance_refs) or old_baseline.acceptance_refs
        integration_refs = tuple(proposal.replacement_integration_refs) or old_baseline.integration_refs
        journey_refs = tuple(proposal.replacement_journey_refs) or old_baseline.journey_refs
        new_baseline = replace(
            old_baseline, ref=self.ref(f"baseline:{old_baseline.ref.revision + 1}", old_baseline.ref.revision + 1),
            requirement_refs=requirement_refs, contract_refs=contract_refs, acceptance_refs=acceptance_refs,
            integration_refs=integration_refs, journey_refs=journey_refs, package_refs=package_refs,
            replaces_ref=old_baseline.ref,
        )
        self.add(new_baseline)
        self.current_baseline_ref = new_baseline.ref
        self.current_candidate_ref = None
        self.current_target_ref = None
        self._current_admission = None
        self.owner_epoch += 1
        for package in replacement_packages:
            self._design_invalid.discard(package.ref.node_id)
        for item in impact.affected:
            self._design_invalid.add(item.subject_ref.node_id)
        self._preserved_package_keys = {item.subject_ref.key for item in impact.preserved}
        self.timeline.append("change_enacted", {
            "proposal": proposal.ref.node_id, "old_baseline": old_baseline.ref.node_id,
            "new_baseline": new_baseline.ref.node_id, "rework": [item.subject_ref.node_id for item in impact.affected],
        }, actor="host", turn=self.watermark, entities=(new_baseline.ref.node_id,),
                             content="change enacted; affected packages fenced", persist=False)
        return ChangeEnactment(
            proposal.ref, old_baseline.ref, new_baseline.ref, impact.ref,
            impact.admissions_to_revoke, tuple(item.subject_ref for item in impact.affected),
            tuple(item.subject_ref for item in impact.preserved),
        )

    def trace(self, start: Ref, *, direction: str = "out", limit: int = 64) -> tuple[Ref, ...]:
        self._check_scope(start)
        if limit < 1:
            raise BuildModeError("trace limit must be positive")
        links = list(self.trace_links.values())
        result: list[Ref] = [start]
        queue: deque[Ref] = deque([start])
        seen = {start.key}
        while queue and len(result) < limit:
            current = queue.popleft()
            for link in links:
                candidate: Ref | None = None
                if direction == "out" and link.source_ref == current:
                    candidate = link.target_ref
                elif direction == "in" and link.target_ref == current:
                    candidate = link.source_ref
                if candidate is not None and candidate.key not in seen:
                    seen.add(candidate.key)
                    result.append(candidate)
                    queue.append(candidate)
        return tuple(result)

    def fired(self, journey_ref: Ref, *, candidate_ref: Ref | None = None) -> tuple[ExecutionReceipt, ...]:
        candidate = candidate_ref or self.current_candidate_ref
        baseline = self._baseline().ref
        return tuple(receipt for receipt in self.receipts.values()
                     if receipt.subject_ref == journey_ref and receipt.fired
                     and receipt.outcome is ReceiptOutcome.PASS and receipt.baseline_ref == baseline
                     and receipt.candidate_ref == candidate and receipt.ref.key not in self._stale_receipts)


def record_payload(record: object) -> dict[str, object]:
    """Public bounded serialization used by receipts and operator views."""
    if hasattr(record, "to_dict"):
        value = record.to_dict()  # type: ignore[union-attr]
        if isinstance(value, dict):
            return value
    value = _plain(asdict(record)) if hasattr(record, "__dataclass_fields__") else _plain(record)
    return value if isinstance(value, dict) else {"value": value}


__all__ = [
    "AcceptanceCriterion", "Assessment", "AssessmentValue", "ArtifactRef", "BuildAssessment",
    "BuildAdmission", "BuildGraph", "BuildGraphError", "BuildLedger", "BuildModeError", "ChangeEnactment",
    "ChangeKind", "ChangeProposal", "Component", "Contract", "DecisionRecord", "DecisionStatus",
    "DesignBaseline", "EdgeRelation", "ExecutionReceipt", "Facet", "Freshness", "Gate", "GateEvaluation",
    "GateResult", "ImpactAnalysis", "ImpactDisposition", "ImpactRow", "ImplementationBinding", "IntegrationObligation",
    "Journey", "POLICY_VERSION", "PackageWorkflow", "ReceiptOutcome", "Ref", "ReqRef", "RequirementNode",
    "SCHEMA", "Scope", "TraceLink", "WorkPackage", "record_payload",
]


# Backward-compatible spelling for callers that use the report's generic name.
BuildGraphError = BuildModeError
