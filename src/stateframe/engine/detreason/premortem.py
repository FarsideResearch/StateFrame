"""Enumerate possible failure modes before applying a proposed change.

Hazard catalogs provide deterministic candidates and checks. Their entries
remain hypotheses until supported by recorded evidence.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_POC = os.path.dirname(_HERE)
if _POC not in sys.path:
    sys.path.insert(0, _POC)

from detreason.snapshot import cse1_digest  # noqa: E402 — the one canonical-hash function
from detreason import obligations as OBL  # noqa: E402 — compose target (Obligation(kind=PREMORTEM))

GENERATOR_VERSION = "premortem/v1"
SCHEMA = "premortem.feature/1"

# ---------------------------------------------------------------------------
# Component taxonomy (w53 doc, verbatim list).
# ---------------------------------------------------------------------------

COMPONENT_TYPES: Tuple[str, ...] = (
    "api_handler", "queue_publisher", "queue_consumer", "worker", "cron", "scheduler",
    "db_writer", "db_reader", "migration", "external_client", "webhook_receiver",
    "event_publisher", "event_consumer", "cache", "frontend_effect", "unknown",
)

# Procedure stages (w53 doc's `procedure.py` integration section).
STAGE_DESIGN = "DESIGN"
STAGE_VERIFY = "VERIFY"
STAGE_TEST = "TEST"
STAGE_PROVE = "PROVE"
STAGE_REVIEW = "REVIEW"
STAGES: Tuple[str, ...] = (STAGE_DESIGN, STAGE_VERIFY, STAGE_TEST, STAGE_PROVE, STAGE_REVIEW)

SEVERITIES: Tuple[str, ...] = ("critical", "high", "medium", "low")
_SEVERITY_RANK: Dict[str, int] = {"critical": 3, "high": 2, "medium": 1, "low": 0}

# The eight measured taxonomy classes (canonical, closed — never extended
# ad hoc). Each is triggered ONLY by an explicit, author-declared marker in
# ``FeatureSpec.constraints`` — never by fuzzy keyword sniffing over prose,
# per the design doc: "No class is assigned from a failed result alone."
TAXONOMY_CLASSES: Tuple[str, ...] = (
    "position_lock", "sign_error", "emission_starvation", "slot_context_wall",
    "vocabulary_gap", "structural_gap", "attractor_lock", "sandbox_containment",
)

TAXONOMY_TRIGGER_MARKERS: Dict[str, str] = {
    "position_lock": "positional_selection",
    "sign_error": "binary_polarity_score",
    "emission_starvation": "broad_model_output_channel",
    "slot_context_wall": "context_plus_output_over_budget",
    "vocabulary_gap": "retrieval_or_vocabulary_disjoint",
    "structural_gap": "required_shape_absent_from_typed_language",
    "attractor_lock": "shared_high_fanin_utility_competes_with_target",
    "sandbox_containment": "witness_lacks_runtime_or_fixtures",
}

TAXONOMY_REQUIRED_OBLIGATION: Dict[str, str] = {
    "position_lock": "stable opaque IDs, permutation replay, held-out swap",
    "sign_error": "known-good/no-op/known-bad sentinels and token mapping",
    "emission_starvation": "typed holes/templates, exact envelope, compile gate",
    "slot_context_wall": "budget inequality, truncation receipt, explicit abstention",
    "vocabulary_gap": "additive retrieval union and recall measurement",
    "structural_gap": "closed enumeration and SEARCH_EXHAUSTED distinction",
    "attractor_lock": "ownership/fan-in evidence and bounded anti-hub replay",
    "sandbox_containment": "baseline preflight, hermetic fixture, target-reach marker",
}


# ---------------------------------------------------------------------------
# Typed input.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    id: str
    type: str
    name: str = ""
    attrs: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Component.id is required")
        if self.type not in COMPONENT_TYPES:
            raise ValueError(f"Component.type must be one of {COMPONENT_TYPES}, got {self.type!r}")

    def attr(self, key: str) -> Optional[str]:
        for k, v in self.attrs:
            if k == key:
                return v
        return None


# Migration change kinds (w53 doc's migration hazard table, verbatim).
MIGRATION_KINDS: Tuple[str, ...] = (
    "add_nullable_field", "add_required_field", "rename", "remove", "change_type",
    "add_enum", "add_fk_unique_index", "backfill",
)


@dataclass(frozen=True)
class MigrationChange:
    id: str
    change_kind: str
    target_ref: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("MigrationChange.id is required")
        if self.change_kind not in MIGRATION_KINDS:
            raise ValueError(f"MigrationChange.change_kind must be one of {MIGRATION_KINDS}, got {self.change_kind!r}")


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    revision: int = 1
    repository_id: str = ""
    base_snapshot: str = ""
    objective: str = ""
    components: Tuple[Component, ...] = ()
    migrations: Tuple[MigrationChange, ...] = ()
    flows: Tuple[str, ...] = ()
    integrations: Tuple[str, ...] = ()
    vocabularies: Tuple[str, ...] = ()
    constraints: Tuple[str, ...] = ()
    provenance: Tuple[str, ...] = ()
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if not self.feature_id:
            raise ValueError("FeatureSpec.feature_id is required")
        ids = [c.id for c in self.components] + [m.id for m in self.migrations]
        if len(ids) != len(set(ids)):
            raise ValueError("FeatureSpec component/migration ids must be unique")

    def digest(self) -> str:
        return cse1_digest(
            {
                "schema": self.schema,
                "feature_id": self.feature_id,
                "revision": self.revision,
                "repository_id": self.repository_id,
                "base_snapshot": self.base_snapshot,
                "objective": self.objective,
                "components": sorted((c.id, c.type, c.name, c.attrs) for c in self.components),
                "migrations": sorted((m.id, m.change_kind, m.target_ref) for m in self.migrations),
                "flows": sorted(self.flows),
                "integrations": sorted(self.integrations),
                "vocabularies": sorted(self.vocabularies),
                "constraints": sorted(self.constraints),
                "provenance": sorted(self.provenance),
            }
        )


def front_door_contract_to_feature_spec(contract: Mapping[str, Any], *, feature_id: str) -> FeatureSpec:
    """Adapter for a ``detreason.front_door`` contract dict (or a Chef goal
    tree flattened to the same shallow shape): pulls the subset of fields
    premortem needs and leaves everything else untouched. Never invents a
    component/migration the contract does not name.
    """
    components = tuple(
        Component(id=str(c["id"]), type=str(c["type"]), name=str(c.get("name", "")),
                  attrs=tuple(sorted((str(k), str(v)) for k, v in dict(c.get("attrs", {})).items())))
        for c in contract.get("components", ())
    )
    migrations = tuple(
        MigrationChange(id=str(m["id"]), change_kind=str(m["change_kind"]), target_ref=str(m.get("target_ref", "")))
        for m in contract.get("migrations", ())
    )
    return FeatureSpec(
        feature_id=feature_id,
        revision=int(contract.get("revision", 1)),
        repository_id=str(contract.get("repository_id", "")),
        base_snapshot=str(contract.get("base_snapshot", "")),
        objective=str(contract.get("objective", "")),
        components=components,
        migrations=migrations,
        flows=tuple(str(x) for x in contract.get("flows", ())),
        integrations=tuple(str(x) for x in contract.get("integrations", ())),
        vocabularies=tuple(str(x) for x in contract.get("vocabularies", ())),
        constraints=tuple(str(x) for x in contract.get("constraints", ())),
        provenance=tuple(str(x) for x in contract.get("provenance", ())),
    )


# ---------------------------------------------------------------------------
# Rule corpus — measured ids, hand-authored executable subset.
# ---------------------------------------------------------------------------

EXECUTABLE = "executable"
DIAGNOSTIC = "diagnostic"
UNCOMPILED = "uncompiled"
RULE_STATUSES: Tuple[str, ...] = (EXECUTABLE, DIAGNOSTIC, UNCOMPILED)

_RULE_HEADER_RE = re.compile(r"^### ([A-Za-z0-9]+)\.", re.MULTILINE)


def _rule_sort_key(rule_id: str) -> Tuple[int, str]:
    m = re.match(r"^(\d+)([a-zA-Z]*)$", rule_id)
    if not m:
        return (10 ** 9, rule_id)
    return (int(m.group(1)), m.group(2))


def discover_rule_ids(claude_md_path: str) -> Tuple[str, ...]:
    """MEASURE the rule ids actually present in ``project rules`` — a scan, not an
    assumption (rule #207/#208: never trust a compressed or remembered
    count). Pure I/O helper; ``generate_premortem`` itself never touches the
    filesystem — callers freeze this result before generation.
    """
    with open(claude_md_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    ids = sorted(set(_RULE_HEADER_RE.findall(text)), key=_rule_sort_key)
    return tuple(ids)


@dataclass(frozen=True)
class RuleSpec:
    id: str
    version: str
    scope: str
    trigger_query: str
    finite_domain: Tuple[str, ...]
    failure_mode: str
    required_producers: Tuple[str, ...]
    forbidden_states: Tuple[str, ...]
    witness_plan: str
    safety_level: str
    status: str
    source_refs: Tuple[str, ...] = ()
    measured_support: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in RULE_STATUSES:
            raise ValueError(f"RuleSpec.status must be one of {RULE_STATUSES}, got {self.status!r}")
        if self.safety_level not in ("hard", "advisory"):
            raise ValueError(f"RuleSpec.safety_level must be hard|advisory, got {self.safety_level!r}")
        if self.status == EXECUTABLE:
            bad = [t for t in self.finite_domain if t not in COMPONENT_TYPES]
            if bad:
                raise ValueError(f"RuleSpec.finite_domain has unknown component types: {bad}")

    def applies_to(self, component_type: str) -> bool:
        return self.status == EXECUTABLE and component_type in self.finite_domain


def _rule(id_: str, domain: Tuple[str, ...], failure_mode: str, trigger: str, safety: str = "hard") -> RuleSpec:
    return RuleSpec(
        id=id_,
        version="1",
        scope="component",
        trigger_query=trigger,
        finite_domain=domain,
        failure_mode=failure_mode,
        required_producers=(f"rules#{id_}",),
        forbidden_states=(),
        witness_plan=f"add/extend a regression test proving rule #{id_} still holds",
        safety_level=safety,
        status=EXECUTABLE,
        source_refs=(f"rules#{id_}",),
    )


#: Hand-authored EXECUTABLE subset. Each id here MUST be a real project rule
#: rule header — enforced by ``build_rule_corpus`` (a stale/typo'd id would
#: otherwise silently vanish into "uncompiled", which is exactly the
#: silent-drop failure family rule #223/#181c exist to prevent).
_EXECUTABLE_RULE_SPECS: Tuple[RuleSpec, ...] = (
    _rule("18", ("queue_consumer", "worker"), "duplicate AgentTask creation without dedup/dispatch guard",
          "a creation site for recurring/queued work with no existing-task query"),
    _rule("28", ("migration",), "table-name list drifts from model.__tablename__",
          "an existence-guarded migration naming a table"),
    _rule("29", ("migration",), "raw SQL executed with SQLAlchemy text() bind style inside a psycopg2 runner",
          "a migration executing parameterized SQL"),
    _rule("29b", ("migration",), "migration entrypoint missing def upgrade(db)",
          "any migration module"),
    _rule("30", ("migration",), "startup seed existence lookup does not match the table's unique constraint",
          "a seed/backfill step guarded by an existence check"),
    _rule("31b", ("migration", "db_writer"), "workspace_id added without converting global UNIQUE to composite",
          "a table gaining a workspace_id/tenant column"),
    _rule("31c", ("migration", "db_writer"), "Postgres text[] column mapped as base ARRAY instead of the postgresql dialect ARRAY(Text)",
          "a migration or model declaring an array column with .contains()/@>/&&"),
    _rule("11b", ("worker", "db_reader"), "free-shape JSON column iterated without isinstance guards",
          "a JSON/JSONB column read and iterated as list/dict"),
    _rule("11c", ("external_client",), "LLM call that json.loads()s output without response_format=json_object",
          "an LLM call whose result is parsed as JSON"),
    _rule("11d", ("external_client",), "finish_reason=='length' or empty content parsed/persisted as a real completion",
          "an LLM call whose completion is persisted"),
    _rule("45b", ("external_client",), "high-effort reasoning model content parsed without a reasoning_content fallback",
          "a direct structured-JSON read of message.content from a reasoning-capable provider"),
    _rule("48", ("db_writer",), "'review' treated as terminal kanban status for deliverable/phase completion",
          "a status-based completion check over kanban/deliverable state"),
    _rule("90", ("db_writer", "worker"), "a service that commits internally shares the caller's SQLAlchemy session",
          "a service function that calls db.commit() internally"),
    _rule("121c", ("db_writer", "migration"), "SQLAlchemy text() uses :param::type cast syntax the bind regex cannot see",
          "a raw text() query casting a bound parameter"),
    _rule("137", ("external_client", "worker"), "a transient provider/infra failure is terminalized instead of self-healing",
          "a failure classifier for an external provider or infra call"),
    _rule("144", ("api_handler", "db_reader"), "a /runner-style endpoint loads a resource by id without filtering workspace_id",
          "an endpoint or service-token-authenticated reader keyed by id/name/prefix"),
    _rule("145", ("api_handler",), "a caller-supplied workspace header outranks the task/agent-authoritative workspace",
          "a workspace-scoped mutating endpoint reachable from a shared service token"),
    _rule("163", ("api_handler",), "an undeclared query/filter param is silently dropped instead of applied",
          "a filtered list/query endpoint"),
    _rule("181c", ("api_handler", "event_consumer"), "the endpoint/tool-name/two-shape-contract silent-drop family",
          "a caller and a contract that must agree on an exact field or tool name"),
    _rule("194", ("api_handler",), "client already sends a scoping header the server query discards",
          "a client-supplied identity/scope header consumed by a server query"),
    _rule("206", ("worker",), "per-task state written to a process-global (os.environ) inside a concurrent worker",
          "concurrent per-task state in a worker process"),
    _rule("217", ("queue_publisher", "queue_consumer", "cron", "scheduler"), "scheduled work lacks one durable occurrence identity across claim/redelivery/reconciliation",
          "a scheduler dispatch, queue message, or child-task creation for recurring/scheduled work"),
    _rule("219", ("db_writer", "event_publisher"), "an audit/telemetry write invents or defaults a tenant/workspace identity",
          "an audit/telemetry insert at an activity boundary"),
    _rule("220", ("queue_consumer", "worker", "api_handler"), "shared service authentication substitutes for signed per-task authority",
          "an agent-task mutation authorized only by shared service credentials"),
    _rule("222", ("event_publisher", "event_consumer", "external_client"), "a cross-service structured-content contract changes on only one side, or permits an unsafe media URL",
          "a shared structured-content block (e.g. hosted media) crossing a service boundary"),
    _rule("223", ("event_consumer", "api_handler", "db_reader"), "a reader accepts N candidate field-name spellings, or has no producer at all",
          "a field crossing a service boundary read by more than one consumer"),
    _rule("9c", ("frontend_effect",), "a SortableJS/vuedraggable drag surface omits touch-action: none",
          "a new drag-and-drop UI surface"),
    _rule("13b", ("frontend_effect",), "a Vue <script setup> block contains a named export",
          "shared constants/functions defined inside <script setup>"),
)


def build_rule_corpus(all_rule_ids: Sequence[str]) -> Tuple[RuleSpec, ...]:
    """Fold the hand-authored executable subset into the full measured id
    list. Every measured id not hand-authored becomes an explicit,
    VISIBLE ``uncompiled`` stub — never silently dropped. Raises if a
    hand-authored id is not present in ``all_rule_ids`` (a typo'd rule number
    must fail loudly, not vanish).
    """
    executable_by_id = {r.id: r for r in _EXECUTABLE_RULE_SPECS}
    stale = sorted(set(executable_by_id) - set(all_rule_ids))
    if stale:
        raise ValueError(f"hand-authored rule ids not found in the measured corpus: {stale}")

    out: List[RuleSpec] = []
    for rid in sorted(set(all_rule_ids), key=_rule_sort_key):
        if rid in executable_by_id:
            out.append(executable_by_id[rid])
        else:
            out.append(
                RuleSpec(
                    id=rid, version="1", scope="uncompiled",
                    trigger_query="", finite_domain=(), failure_mode="",
                    required_producers=(), forbidden_states=(), witness_plan="",
                    safety_level="advisory", status=UNCOMPILED,
                    source_refs=(f"rules#{rid}",),
                )
            )
    return tuple(out)


# ---------------------------------------------------------------------------
# Component-type risk registry (concurrency/idempotency catalog + rule-derived).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskTemplate:
    template_id: str
    component_type: str
    category: str  # "concurrency_idempotency" | "rule" | "migration" | "taxonomy"
    failure_class: str
    failure_statement: str
    consequence: str
    severity: str
    prevention: str
    detection: str
    recovery: str
    stage: str
    required_tests: Tuple[str, ...]
    source_refs: Tuple[str, ...]
    rule_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"RiskTemplate.severity must be one of {SEVERITIES}, got {self.severity!r}")
        if self.stage not in STAGES:
            raise ValueError(f"RiskTemplate.stage must be one of {STAGES}, got {self.stage!r}")


#: w53 doc's concurrency/idempotency catalog, hand-encoded verbatim (component
#: -> exposure/required-facts/required-tests). Not tied to a specific rule id
#: — sourced from the measured design doc directly.
_CONCURRENCY_CATALOG: Tuple[RiskTemplate, ...] = (
    RiskTemplate("conc.api_handler", "api_handler", "concurrency_idempotency", "duplicate_or_concurrent_write",
                 "duplicate requests or concurrent writes are not distinguished",
                 "a retried or double-submitted request applies its effect twice",
                 "high", "idempotency key, transaction boundary, unique constraint, version policy",
                 "replay-same-request test", "reject/no-op on replay",
                 STAGE_TEST, ("replay same request", "concurrent update race", "duplicate-effect assertion"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.queue_publisher", "queue_publisher", "concurrency_idempotency", "ambiguous_send_retry",
                 "a retry after an ambiguous send may double-publish",
                 "a downstream consumer sees the same logical message twice with two different message ids",
                 "high", "stable message id, transactional outbox boundary, FIFO/dedup policy",
                 "duplicate-message assertion on replay", "outbox reconciliation",
                 STAGE_TEST, ("retry and crash-window replay",),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.queue_consumer", "queue_consumer", "concurrency_idempotency", "duplicate_delivery",
                 "duplicate delivery, crash-after-side-effect, or visibility expiry re-delivers a message",
                 "a side effect runs twice for one logical message",
                 "critical", "idempotency store, ack-after-commit ordering, visibility budget, DLQ",
                 "deliver-twice and crash-before-ack fixtures", "DLQ inspection + replay",
                 STAGE_TEST, ("deliver twice", "crash before ack", "slow handler", "poison message"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.worker", "worker", "concurrency_idempotency", "retry_lease_partial_progress",
                 "a retried or lease-lost worker resumes without a run identity or checkpoint",
                 "partial work is silently duplicated or lost on restart",
                 "high", "run identity, explicit state machine, checkpointing",
                 "restart-at-each-effect-boundary fixture", "resume from checkpoint",
                 STAGE_TEST, ("restart at each effect boundary", "duplicate worker", "resume"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.cron", "cron", "concurrency_idempotency", "overlap_or_missed_run",
                 "overlapping runs, a missed schedule, or DST ambiguity is not handled",
                 "two replicas run the same job concurrently, or a run is silently skipped",
                 "high", "distributed lease, deterministic run key, overlap policy",
                 "two-replica + DST-boundary fixture", "lease expiry recovery",
                 STAGE_TEST, ("runtime longer than interval", "two replicas", "restart", "DST boundary"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.scheduler", "scheduler", "concurrency_idempotency", "split_brain_duplicate_enqueue",
                 "two scheduler instances both believe they own the schedule",
                 "the same recurring job is enqueued twice",
                 "critical", "single-owner lease, schedule identity, enqueue idempotency",
                 "lease-expiry-during-enqueue fixture", "duplicate-enqueue detection",
                 STAGE_TEST, ("lease expiry during enqueue", "duplicate scheduler"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.db_writer", "db_writer", "concurrency_idempotency", "lost_update_or_deadlock",
                 "a lost update, check-then-act race, or deadlock is not classified/retried",
                 "a concurrent write silently overwrites another write",
                 "high", "explicit isolation level, constraints, retry classification",
                 "concurrent-write race fixture", "deadlock retry with backoff",
                 STAGE_TEST, ("concurrent writes", "deadlock retry", "rollback"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.db_reader", "db_reader", "concurrency_idempotency", "stale_read_during_deploy",
                 "a reader observes a schema window it was not built for",
                 "old code reads new-shape data or vice versa during a rolling deploy",
                 "medium", "compatible schema window, explicit isolation policy",
                 "old/new reader against old/new schema matrix", "compatibility window shortening",
                 STAGE_TEST, ("old/new reader against old/new schema",),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.migration", "migration", "concurrency_idempotency", "partial_execution_lock_contention",
                 "a migration partially executes or blocks on lock contention",
                 "the schema is left in a half-migrated state with no safe resume",
                 "critical", "expand-contract order, resumable backfill, timeout policy",
                 "interruption + restart fixture", "resumable backfill",
                 STAGE_TEST, ("interruption", "restart", "old binary", "new binary"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.external_client", "external_client", "concurrency_idempotency", "timeout_after_remote_success",
                 "a client times out after the remote call actually succeeded",
                 "a retry re-executes an already-applied remote mutation",
                 "high", "provider idempotency key, reconciliation path",
                 "timeout-after-success fixture", "reconciliation query",
                 STAGE_TEST, ("timeout-after-success", "429/5xx", "partial response"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.webhook_receiver", "webhook_receiver", "concurrency_idempotency", "duplicate_or_reordered_event",
                 "a duplicate or out-of-order webhook event is not deduped/ordered",
                 "an event is applied twice, or applied out of causal order",
                 "high", "signature verification, event id dedupe, ordering policy",
                 "duplicate + reordered event fixture", "signature-verified replay",
                 STAGE_TEST, ("duplicate event", "reordered event", "invalid signature"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.event_consumer", "event_consumer", "concurrency_idempotency", "duplicate_reorder_unknown_version",
                 "a duplicate, reordered, or unknown-schema-version event is mishandled",
                 "state transitions out of order or on an unrecognized event shape",
                 "high", "event id, version policy, ordered or commutative transition",
                 "duplicate + reorder + unknown-version fixture", "version-gated rejection",
                 STAGE_TEST, ("duplicate", "reorder", "unknown version"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.cache", "cache", "concurrency_idempotency", "stale_read_or_stampede",
                 "a cache read is stale or a cache miss stampede is not single-flighted",
                 "many concurrent misses hammer the origin at once, or serve stale data past invalidation",
                 "medium", "versioned key, invalidation owner, single-flight policy",
                 "concurrent-miss + stale-after-write fixture", "single-flight lock",
                 STAGE_TEST, ("concurrent miss", "stale-after-write"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
    RiskTemplate("conc.frontend_effect", "frontend_effect", "concurrency_idempotency", "stale_async_or_double_submit",
                 "a stale async response, double submit, or unmount race is not guarded",
                 "an out-of-order response overwrites a newer one, or a form double-submits",
                 "medium", "request identity, cancellation, replay policy",
                 "out-of-order response + unmount-during-request fixture", "request-identity guard",
                 STAGE_TEST, ("out-of-order response", "unmount during request"),
                 ("design-reference/w53#concurrency-idempotency-catalog",)),
)


def _risk_template_from_rule(rule: RuleSpec, component_type: str) -> RiskTemplate:
    return RiskTemplate(
        template_id=f"rule.{rule.id}.{component_type}",
        component_type=component_type,
        category="rule",
        failure_class=f"rule_{rule.id}",
        failure_statement=rule.failure_mode,
        consequence=f"{rule.failure_mode} (rule #{rule.id})",
        severity="high" if rule.safety_level == "hard" else "medium",
        prevention=f"apply rule #{rule.id} at design time",
        detection=rule.trigger_query,
        recovery=rule.witness_plan,
        stage=STAGE_VERIFY,
        required_tests=(rule.witness_plan,),
        source_refs=rule.source_refs,
        rule_id=rule.id,
    )


def component_risk_registry(rule_corpus: Sequence[RuleSpec]) -> Dict[str, Tuple[RiskTemplate, ...]]:
    """Per-component-type risk catalog: the concurrency/idempotency catalog
    plus every EXECUTABLE rule whose finite_domain names that component type.
    Deterministic given the same rule corpus.
    """
    registry: Dict[str, List[RiskTemplate]] = {t: [] for t in COMPONENT_TYPES}
    for rt in _CONCURRENCY_CATALOG:
        registry[rt.component_type].append(rt)
    for rule in rule_corpus:
        if rule.status != EXECUTABLE:
            continue
        for ctype in rule.finite_domain:
            registry.setdefault(ctype, []).append(_risk_template_from_rule(rule, ctype))
    return {k: tuple(sorted(v, key=lambda r: r.template_id)) for k, v in registry.items()}


# ---------------------------------------------------------------------------
# Migration hazard catalog (w53 doc's migration table, verbatim).
# ---------------------------------------------------------------------------

_MIGRATION_CATALOG: Dict[str, RiskTemplate] = {
    "add_nullable_field": RiskTemplate(
        "mig.add_nullable_field", "migration", "migration", "old_reader_compatibility",
        "an added nullable field/index/table breaks an old reader's assumptions",
        "an old-code reader crashes or silently ignores the new field",
        "low", "keep the field genuinely optional in every reader", "old/new reader matrix",
        "no rollback needed if truly additive", STAGE_TEST, ("old app + new schema",),
        ("design-reference/w53#migration-hazards",),
    ),
    "add_required_field": RiskTemplate(
        "mig.add_required_field", "migration", "migration", "default_backfill_ordering",
        "a required field/default is added without a safe default/backfill order",
        "existing rows violate the new constraint at write time",
        "high", "expand-contract: default first, backfill, then enforce", "backfill-then-enforce ordering test",
        "resumable backfill", STAGE_TEST, ("partial backfill + new app", "restart at every batch boundary"),
        ("design-reference/w53#migration-hazards",),
    ),
    "rename": RiskTemplate(
        "mig.rename", "migration", "migration", "dual_read_write_window",
        "a rename has no dual-read/write compatibility window",
        "an in-flight deploy reads or writes the old name after it stops existing",
        "high", "dual-read/write until every consumer is migrated", "old app + new schema / new app + old schema",
        "keep both names live until confirmed unused", STAGE_TEST, ("old app + new schema", "new app + old schema"),
        ("design-reference/w53#migration-hazards",),
    ),
    "remove": RiskTemplate(
        "mig.remove", "migration", "migration", "stale_reader_or_consumer",
        "a field/value removal leaves a stale reader or consumer referencing it",
        "a reader silently gets None/undefined where it expected a value",
        "high", "confirm zero live readers before removal", "stale-consumer sweep",
        "temporary compatibility shim", STAGE_VERIFY, ("stale reader/consumer sweep",),
        ("design-reference/w53#migration-hazards",),
    ),
    "change_type": RiskTemplate(
        "mig.change_type", "migration", "migration", "conversion_and_rollback",
        "a type/meaning change has no conversion or rollback plan",
        "existing data is misinterpreted under the new type",
        "critical", "explicit conversion function + rollback plan", "conversion round-trip test",
        "documented rollback is never assumed safe after new data is written", STAGE_PROVE, ("conversion round-trip",),
        ("design-reference/w53#migration-hazards",),
    ),
    "add_enum": RiskTemplate(
        "mig.add_enum", "migration", "migration", "vocabulary_compatibility",
        "an added enum/status value has no unknown-value handling on the consumer side",
        "a consumer's exhaustive switch/if-chain silently drops the new value",
        "high", "unknown-value branch in every consumer", "unknown-value matrix",
        "fail-closed unknown-value handling", STAGE_TEST, ("exhaustive value matrix", "unknown-value behavior"),
        ("design-reference/w53#migration-hazards",),
    ),
    "add_fk_unique_index": RiskTemplate(
        "mig.add_fk_unique_index", "migration", "migration", "existing_data_and_lock_risk",
        "an added FK/unique/index is applied without checking existing-data violations or lock duration",
        "the migration fails mid-run against live data, or locks the table for writers",
        "high", "pre-check existing data + concurrent index creation", "existing-data violation check",
        "abort before lock, not during", STAGE_VERIFY, ("existing-data pre-check",),
        ("design-reference/w53#migration-hazards",),
    ),
    "backfill": RiskTemplate(
        "mig.backfill", "migration", "migration", "restart_and_idempotency_risk",
        "a backfill/recompute has no restart or idempotency guarantee",
        "a restarted backfill double-applies its effect on already-processed rows",
        "high", "idempotent batch processing + resumable cursor", "restart-mid-batch fixture",
        "resumable cursor + idempotent writes", STAGE_TEST, ("restart mid-batch",),
        ("design-reference/w53#migration-hazards",),
    ),
}


# ---------------------------------------------------------------------------
# Output records.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskItem:
    risk_id: str
    category: str
    failure_class: str
    affected_ids: Tuple[str, ...]
    trigger_receipt: str
    failure_statement: str
    consequence: str
    severity: str
    exposure: str
    confidence: str
    priority_key: str
    source_refs: Tuple[str, ...]
    required_obligation_ids: Tuple[str, ...] = ()
    status: str = "open"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"RiskItem.severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass(frozen=True)
class ProcedureObligation:
    obligation_id: str
    risk_id: str
    rule_id: Optional[str]
    stage: str
    kind: str
    blocking: bool
    trigger_receipt: str
    setup_ir: Tuple[Tuple[str, Any], ...]
    invocation_ir: Tuple[Tuple[str, Any], ...]
    expected_observation: str
    forbidden_observations: Tuple[str, ...]
    required_edges: Tuple[str, ...]
    required_effects: Tuple[str, ...]
    mutation_kinds: Tuple[str, ...]
    owner: str
    evidence_refs: Tuple[str, ...]
    generator_version: str = GENERATOR_VERSION
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"ProcedureObligation.stage must be one of {STAGES}, got {self.stage!r}")

    def to_ledger_kwargs(
        self, *, workspace_id: str, session_id: str, origin_event_id: str, now: int,
    ) -> Dict[str, Any]:
        """Compose with ``detreason.obligations.Obligation(kind=PREMORTEM)``.

        Returns kwargs that construct a valid ``Obligation`` — the caller
        (a real kernel stage with a live workspace/session/clock) decides
        when to actually register it into an ``ObligationLedger``. This
        module never opens a ledger itself: registration is a session-scoped
        side effect and premortem.py is deliberately pure.
        """
        contract = OBL.ActionContract(
            action_type=f"premortem_{self.kind}",
            target_ref=self.risk_id,
            expected_actor="agent" if self.owner in ("agent", "system") else "human",
            postconditions=(self.expected_observation,),
            authorization_scope=self.owner,
        )
        return dict(
            id=self.obligation_id,
            workspace_id=workspace_id,
            session_id=session_id,
            kind=OBL.PREMORTEM,
            title=self.expected_observation[:200],
            contract=contract,
            origin_event_id=origin_event_id,
            owner="agent" if self.owner in ("agent", "system") else "human",
            priority_class=OBL.P0 if self.blocking else OBL.P2,
            normalized_intent=self.trigger_receipt,
            target_ref=self.risk_id,
            first_seen_at=now,
            created_at=now,
            last_progress_at=now,
        )


@dataclass(frozen=True)
class CoverageReceipt:
    expected_rule_count: int
    observed_rule_count: int
    unique_rule_ids: int
    rule_counts: Tuple[Tuple[str, int], ...]
    taxonomy_classes_seen: Tuple[str, ...]
    taxonomy_classes_with_fixtures: Tuple[str, ...]
    graph: Tuple[Tuple[str, int], ...]
    truncation: Tuple[str, ...]
    hard_unknowns: int
    complete: bool


@dataclass(frozen=True)
class PreMortemReport:
    schema: str
    generator_version: str
    feature_digest: str
    corpus_digest: str
    taxonomy_digest: str
    baseline_snapshot_digest: str
    config_digest: str
    coverage: CoverageReceipt
    assumptions: Tuple[str, ...]
    seams: Tuple[str, ...]
    vocabularies: Tuple[str, ...]
    concurrency: Tuple[str, ...]
    migrations: Tuple[str, ...]
    taxonomy_findings: Tuple[str, ...]
    risks: Tuple[RiskItem, ...]
    obligations: Tuple[ProcedureObligation, ...]
    unknowns: Tuple[str, ...]
    report_digest: str = ""


# ---------------------------------------------------------------------------
# Deterministic algorithm.
# ---------------------------------------------------------------------------


def _risk_from_template(rt: RiskTemplate, affected_id: str, trigger_receipt: str, exposure: str) -> RiskItem:
    risk_id = cse1_digest({"template": rt.template_id, "affected": affected_id, "exposure": exposure})
    return RiskItem(
        risk_id=risk_id,
        category=rt.category,
        failure_class=rt.failure_class,
        affected_ids=(affected_id,),
        trigger_receipt=trigger_receipt,
        failure_statement=rt.failure_statement,
        consequence=rt.consequence,
        severity=rt.severity,
        exposure=exposure,
        confidence="certain",
        priority_key=f"{_SEVERITY_RANK[rt.severity]:02d}:{risk_id}",
        source_refs=rt.source_refs,
    )


def _apply_taxonomy(spec: FeatureSpec) -> List[RiskItem]:
    out: List[RiskItem] = []
    constraints = frozenset(spec.constraints)
    for failure_class in TAXONOMY_CLASSES:
        marker = TAXONOMY_TRIGGER_MARKERS[failure_class]
        if marker not in constraints:
            continue
        trigger_receipt = f"FeatureSpec.constraints contains {marker!r}"
        risk_id = cse1_digest({"taxonomy": failure_class, "feature": spec.feature_id})
        out.append(
            RiskItem(
                risk_id=risk_id,
                category="taxonomy",
                failure_class=failure_class,
                affected_ids=(spec.feature_id,),
                trigger_receipt=trigger_receipt,
                failure_statement=f"measured failure class {failure_class} is exposed",
                consequence=TAXONOMY_REQUIRED_OBLIGATION[failure_class],
                severity="high",
                exposure="taxonomy",
                confidence="certain",
                priority_key=f"{_SEVERITY_RANK['high']:02d}:{risk_id}",
                source_refs=("design-reference/w46/a_failure_taxonomy.md", "design-reference/w53#measured-taxonomy-application"),
            )
        )
    return out


def _deduplicate_and_rank(risks: Sequence[RiskItem]) -> Tuple[RiskItem, ...]:
    by_id: Dict[str, RiskItem] = {}
    for r in risks:
        by_id.setdefault(r.risk_id, r)
    return tuple(sorted(by_id.values(), key=lambda r: r.risk_id))


def _materialize_obligations(risks: Sequence[RiskItem]) -> Tuple[ProcedureObligation, ...]:
    out: List[ProcedureObligation] = []
    for r in risks:
        blocking = r.severity in ("critical", "high")
        stage = STAGE_VERIFY if blocking else STAGE_DESIGN
        kind = "test" if blocking else "review"
        rule_id = r.failure_class[len("rule_"):] if r.category == "rule" else None
        obligation_id = cse1_digest({"risk_id": r.risk_id, "kind": kind, "stage": stage})
        out.append(
            ProcedureObligation(
                obligation_id=obligation_id,
                risk_id=r.risk_id,
                rule_id=rule_id,
                stage=stage,
                kind=kind,
                blocking=blocking,
                trigger_receipt=r.trigger_receipt,
                setup_ir=(("affected_ids", r.affected_ids), ("category", r.category)),
                invocation_ir=(("failure_class", r.failure_class),),
                expected_observation=r.consequence,
                forbidden_observations=(r.failure_statement,),
                required_edges=(),
                required_effects=(),
                mutation_kinds=(),
                owner="agent",
                evidence_refs=r.source_refs,
            )
        )
    return tuple(sorted(out, key=lambda o: o.obligation_id))


def generate_premortem(
    spec: FeatureSpec,
    rule_corpus: Sequence[RuleSpec],
    *,
    baseline_snapshot_digest: str = "",
    config: Optional[Mapping[str, Any]] = None,
) -> PreMortemReport:
    """The pure compiler. No clock, no randomness, no I/O, no model call.

    ``rule_corpus`` must be produced by :func:`build_rule_corpus` over a
    FROZEN ``discover_rule_ids(...)`` result — the caller freezes inputs
    before calling this function, per the design's ``validate_and_freeze_inputs``
    step.
    """
    cfg = dict(config or {})
    registry = component_risk_registry(rule_corpus)

    risks: List[RiskItem] = []
    for comp in spec.components:
        for rt in registry.get(comp.type, ()):
            trigger = f"component {comp.id!r} has type {comp.type!r}"
            risks.append(_risk_from_template(rt, comp.id, trigger, exposure="component"))
    for mig in spec.migrations:
        rt = _MIGRATION_CATALOG.get(mig.change_kind)
        if rt is not None:
            trigger = f"migration {mig.id!r} has change_kind {mig.change_kind!r}"
            risks.append(_risk_from_template(rt, mig.id, trigger, exposure="migration"))
    risks.extend(_apply_taxonomy(spec))

    ranked = _deduplicate_and_rank(risks)
    obligations = _materialize_obligations(ranked)

    executable = tuple(r for r in rule_corpus if r.status == EXECUTABLE)
    diagnostic = tuple(r for r in rule_corpus if r.status == DIAGNOSTIC)
    uncompiled = tuple(r for r in rule_corpus if r.status == UNCOMPILED)
    applicable_component_types = {c.type for c in spec.components}
    applicable = tuple(
        r for r in executable if set(r.finite_domain) & applicable_component_types
    )
    triggered_rule_ids = sorted({r.failure_class[len("rule_"):] for r in ranked if r.category == "rule"})

    unique_ids = {r.id for r in rule_corpus}
    complete = (
        len(rule_corpus) > 0
        and len(unique_ids) == len(rule_corpus)
        and all(r.status == EXECUTABLE for r in rule_corpus)
    )

    coverage = CoverageReceipt(
        expected_rule_count=cfg.get("expected_rule_count", len(rule_corpus)),
        observed_rule_count=len(rule_corpus),
        unique_rule_ids=len(unique_ids),
        rule_counts=(
            ("executable", len(executable)),
            ("diagnostic", len(diagnostic)),
            ("uncompiled", len(uncompiled)),
            ("applicable", len(applicable)),
            ("triggered", len(triggered_rule_ids)),
        ),
        taxonomy_classes_seen=tuple(sorted({r.failure_class for r in ranked if r.category == "taxonomy"})),
        taxonomy_classes_with_fixtures=(),
        graph=(("nodes", 0), ("edges", 0), ("exact", 0), ("partial", 0), ("unknown", 0)),
        truncation=(),
        hard_unknowns=0,
        complete=complete,
    )

    unknowns: List[str] = list(sorted(r.id for r in uncompiled))

    def _cat(cat: str) -> Tuple[str, ...]:
        return tuple(sorted(r.risk_id for r in ranked if r.category == cat))

    report = PreMortemReport(
        schema=SCHEMA,
        generator_version=GENERATOR_VERSION,
        feature_digest=spec.digest(),
        corpus_digest=cse1_digest([r.id for r in rule_corpus]),
        taxonomy_digest=cse1_digest(TAXONOMY_CLASSES),
        baseline_snapshot_digest=baseline_snapshot_digest,
        config_digest=cse1_digest(sorted(cfg.items())),
        coverage=coverage,
        assumptions=(
            "layer/component identity comes only from the caller-supplied FeatureSpec; nothing is inferred from prose",
            "the rule corpus is a measured rules scan, not an assumed 223-rule manifest",
        ),
        seams=_cat("rule"),
        vocabularies=(),
        concurrency=_cat("concurrency_idempotency"),
        migrations=_cat("migration"),
        taxonomy_findings=_cat("taxonomy"),
        risks=ranked,
        obligations=obligations,
        unknowns=tuple(unknowns),
    )
    digest = cse1_digest(
        {
            "feature_digest": report.feature_digest,
            "corpus_digest": report.corpus_digest,
            "taxonomy_digest": report.taxonomy_digest,
            "baseline_snapshot_digest": report.baseline_snapshot_digest,
            "config_digest": report.config_digest,
            "risks": [r.risk_id for r in ranked],
            "obligations": [o.obligation_id for o in obligations],
        }
    )
    return PreMortemReport(**{**report.__dict__, "report_digest": digest})


__all__ = [
    "GENERATOR_VERSION", "SCHEMA",
    "COMPONENT_TYPES", "STAGE_DESIGN", "STAGE_VERIFY", "STAGE_TEST", "STAGE_PROVE", "STAGE_REVIEW", "STAGES",
    "SEVERITIES", "TAXONOMY_CLASSES", "TAXONOMY_TRIGGER_MARKERS", "TAXONOMY_REQUIRED_OBLIGATION",
    "Component", "MIGRATION_KINDS", "MigrationChange", "FeatureSpec", "front_door_contract_to_feature_spec",
    "EXECUTABLE", "DIAGNOSTIC", "UNCOMPILED", "RULE_STATUSES",
    "discover_rule_ids", "RuleSpec", "build_rule_corpus",
    "RiskTemplate", "component_risk_registry",
    "RiskItem", "ProcedureObligation", "CoverageReceipt", "PreMortemReport",
    "generate_premortem",
]
