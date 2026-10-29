"""Build bounded context slices for model execution and work resumption.

Slices retain the state and evidence needed for the next decision while enforcing
the configured context budget. Presentation does not change durable work state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from detreason.layer_registry import CanonicalLayer

# ---------------------------------------------------------------------------
# Level taxonomy
# ---------------------------------------------------------------------------


class ZoomLevel(enum.Enum):
    """ZoomSliceV1's level taxonomy.

    PLATFORM | APP | SUBSYSTEM come directly from `ZoomSliceV1.level` in
    hierarchical-working-model.md ("level: PLATFORM | APP | SUBSYSTEM").

    OVERLAY is NOT in the base ZoomSliceV1 spec. It is named in
    `w57/a_goal_converge.md` (B22 in BURN_LIST: "a GoalDeclaration states
    ... over the typed PLATFORM/APP/SUBSYSTEM/OVERLAY graph") as a fourth
    scope representing continuously-fresh deployed-bindings/runtime-
    observation facts layered over the other three. BURN_LIST flags this as
    an unscoped prerequisite ("No such materialization pipeline exists" --
    B22 landmine) -- OVERLAY is accepted here as a valid `ZoomLevel` for W0-6
    vocabulary compatibility (goal-converge already builds on the named
    scheme) but `validate_zoom_slice` never assumes an OVERLAY slice has a
    real backing materializer. A caller emitting OVERLAY slices today is
    describing an aspiration, not a built pipeline; that is a build-order
    fact, not a schema fact, and stays out of this module's job.
    """

    PLATFORM = "PLATFORM"
    APP = "APP"
    SUBSYSTEM = "SUBSYSTEM"
    OVERLAY = "OVERLAY"


# Levels the base ZoomSliceV1 doc actually defines a per-portion budget for.
# OVERLAY is excluded -- see the ZoomLevel docstring.
_BUDGETED_LEVELS = (ZoomLevel.PLATFORM, ZoomLevel.APP, ZoomLevel.SUBSYSTEM)


# ---------------------------------------------------------------------------
# Budget profiles -- every number below is transcribed from a source doc, not
# invented. See the citation in each profile's `source` field.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetProfile:
    """One named token/row budget. This is the "profile" BURN_LIST W0-1 means
    by "budget profiles of ZoomSliceV1" -- narrower consumers get a profile,
    never a new artifact type.
    """

    name: str
    target_tokens: int
    hard_max_tokens: int
    hard_max_nodes: Optional[int]
    hard_max_edges: Optional[int]
    hard_max_rows: Optional[int]
    source: str


# --- ZoomSliceV1 native per-portion table (hierarchical-working-model.md,
#     "Context and storage budgets" > "Per-slice 1B envelope") -------------

PROFILE_TASK_ENVELOPE = BudgetProfile(
    name="task_policy_evidence_envelope",
    target_tokens=500, hard_max_tokens=800,
    hard_max_nodes=None, hard_max_edges=None, hard_max_rows=20,
    source="hierarchical-working-model.md: task/policy/evidence envelope row",
)

PROFILE_PLATFORM = BudgetProfile(
    name="platform_slice",
    target_tokens=700, hard_max_tokens=1_100,
    hard_max_nodes=32, hard_max_edges=64, hard_max_rows=None,
    source="hierarchical-working-model.md: platform slice row (32 nodes / 64 edges)",
)

PROFILE_APP = BudgetProfile(
    name="app_slice",
    target_tokens=1_200, hard_max_tokens=1_900,
    hard_max_nodes=48, hard_max_edges=96, hard_max_rows=None,
    source="hierarchical-working-model.md: app slice row (48 nodes / 96 edges)",
)

PROFILE_SUBSYSTEM = BudgetProfile(
    name="subsystem_slice",
    target_tokens=2_400, hard_max_tokens=3_800,
    hard_max_nodes=96, hard_max_edges=192, hard_max_rows=None,
    source="hierarchical-working-model.md: subsystem slice row (96 nodes / 192 edges)",
)

PROFILE_TESTS_WITNESSES = BudgetProfile(
    name="tests_witnesses_unknowns",
    target_tokens=400, hard_max_tokens=700,
    hard_max_nodes=None, hard_max_edges=None, hard_max_rows=32,
    source="hierarchical-working-model.md: tests/witnesses/unknowns row",
)

# The doc's own "total request" row -- 5,200 target / 8,300 hard max. This is
# the sum-of-portions ceiling for one ZoomSliceV1 response, distinct from the
# whole-session ceiling below.
PROFILE_TOTAL_REQUEST = BudgetProfile(
    name="total_request",
    target_tokens=5_200, hard_max_tokens=8_300,
    hard_max_nodes=None, hard_max_edges=None, hard_max_rows=None,
    source="hierarchical-working-model.md: 'total request' row, per-slice 1B envelope table",
)

# "One ZOOM session also has a default 4-slice/20K-token target and a hard
# 32K-token ceiling." -- hierarchical-working-model.md, "Per-slice 1B envelope"
# section, paragraph following the table.
PROFILE_SESSION = BudgetProfile(
    name="zoom_session",
    target_tokens=20_000, hard_max_tokens=32_000,
    hard_max_nodes=None, hard_max_edges=None, hard_max_rows=4,  # "4-slice" target
    source="hierarchical-working-model.md: session budget paragraph (4-slice/20K target, 32K hard ceiling)",
)

# --- Juggler-lane-resume profiles (narrower than ZoomSliceV1's own slices;
#     these are what W0-1 means by "budget profiles" for the pamphlet/stub) --

PROFILE_JUGGLER_PAMPHLET = BudgetProfile(
    name="juggler_rehydration_pamphlet",
    target_tokens=900, hard_max_tokens=1_024,
    hard_max_nodes=None, hard_max_edges=None, hard_max_rows=None,
    source=(
        "a_juggler_core.md section 5 'Pamphlet sizing': "
        "'Target: 700 to 900 tokens / Hard ceiling: 1,024 tokens'. "
        "This module uses the top of the target range (900) as the single "
        "target_tokens value; the doc's own context-relative formula "
        "(pamphlet_cap = min(1024, max(512, floor(context_window * 0.20)))) "
        "is a sizing FORMULA a caller may use to derive a tighter cap, not a "
        "second budget -- it never exceeds this profile's hard_max_tokens."
    ),
)

PROFILE_JUGGLER_DECISION_STUB = BudgetProfile(
    name="juggler_decision_stub",
    target_tokens=384, hard_max_tokens=768,
    hard_max_nodes=None, hard_max_edges=None, hard_max_rows=None,
    source=(
        "a_juggler_switch.md section 1: "
        "'I would start with a hard target of approximately 384 tokens for "
        "the normal decision stub, with an absolute ceiling around 768 tokens.'"
    ),
)

ALL_PROFILES: Tuple[BudgetProfile, ...] = (
    PROFILE_TASK_ENVELOPE, PROFILE_PLATFORM, PROFILE_APP, PROFILE_SUBSYSTEM,
    PROFILE_TESTS_WITNESSES, PROFILE_TOTAL_REQUEST, PROFILE_SESSION,
    PROFILE_JUGGLER_PAMPHLET, PROFILE_JUGGLER_DECISION_STUB,
)

# Which profile a ZoomLevel's slice-portion is normally checked against, for
# the ZoomSliceV1-native (non-juggler) use case. A caller building a juggler
# lane-resume slice passes PROFILE_JUGGLER_PAMPHLET/_DECISION_STUB explicitly
# instead -- that is a different consumer, not a different level.
LEVEL_DEFAULT_PROFILE: Dict[ZoomLevel, BudgetProfile] = {
    ZoomLevel.PLATFORM: PROFILE_PLATFORM,
    ZoomLevel.APP: PROFILE_APP,
    ZoomLevel.SUBSYSTEM: PROFILE_SUBSYSTEM,
    # OVERLAY has no native budget row in hierarchical-working-model.md (see
    # the ZoomLevel docstring) -- default it to the same envelope as PLATFORM
    # (the coarsest defined profile) rather than leaving it unbudgeted, since
    # "no budget stated" is exactly the W0-1 failure mode this module exists
    # to close. A caller with a real OVERLAY materializer should pass an
    # explicit profile once one is designed.
    ZoomLevel.OVERLAY: PROFILE_PLATFORM,
}


# ---------------------------------------------------------------------------
# Supporting record types (all directly transcribed field-for-field from
# ZoomSliceV1's spec in hierarchical-working-model.md, "ZOOM protocol" >
# "Request and response")
# ---------------------------------------------------------------------------


class ResolutionState(enum.Enum):
    """"Every edge has source evidence and a resolution state: resolved,
    partial, unknown, or conflict." -- hierarchical-working-model.md, Model
    invariant 3 ("Evidence or unknown").
    """

    RESOLVED = "resolved"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class NextActionKind(enum.Enum):
    """ZoomActionV1.action -- hierarchical-working-model.md, ZOOM protocol."""

    SELECT = "SELECT"
    DRILL = "DRILL"
    MATERIALIZE = "MATERIALIZE"
    FOLLOW_EDGE = "FOLLOW_EDGE"
    REQUEST_PAGE = "REQUEST_PAGE"
    ABSTAIN = "ABSTAIN"
    ZOOM_OUT = "ZOOM_OUT"


@dataclass(frozen=True)
class SliceFact:
    """One typed fact record. `layer` is the narration-rehydration hook (see
    module docstring "Narration rehydration") -- when present it MUST resolve
    against `layer_registry.CanonicalLayer`; the validator enforces this.
    `layer` is Optional because not every ZoomSlice consumer (e.g. a juggler
    decision stub built from lane-state fields, not graph facts) has a graph
    layer to cite.
    """

    fact_id: str
    text: str
    source_ref: str
    layer: Optional[CanonicalLayer] = None


@dataclass(frozen=True)
class BoundaryEdge:
    """A `cut_edge` or ordinary boundary edge with a resolution state.

    hierarchical-working-model.md: "A hard boundary is returned as
    `cut_edge`; a missing fact is returned as `unknown`; neither is treated
    as a negative result."
    """

    src_id: str
    dst_id: str
    kind: str
    resolution_state: ResolutionState


@dataclass(frozen=True)
class BoundedExcerpt:
    """A redacted, digest-bearing source excerpt, per the "Source-excerpt
    gate" section: excerpts are placed only after an allowed ID is selected,
    and secret-like literals are redacted before storage.
    """

    source_ref: str
    content: str
    digest: str


@dataclass(frozen=True)
class Omitted:
    """ZoomSliceV1.omitted -- "count, reasons, continuation_token".

    This is the paging/continuation-token mechanism BURN_LIST W0-1 calls out
    as one of the reasons ZoomSliceV1 wins: "A slice over budget is paged by
    deterministic continuation token; the model never gets the overflow
    automatically. At most one page is active per layer."
    """

    count: int = 0
    reasons: Tuple[str, ...] = ()
    continuation_token: Optional[str] = None


@dataclass(frozen=True)
class NextAction:
    """One entry of ZoomSliceV1.next_actions (a restricted view of
    ZoomActionV1 -- the full request/response envelope is a ZOOM-protocol
    concern; a ZoomSlice only needs to state what the model MAY do next).
    """

    action: NextActionKind
    candidate_id: Optional[str] = None
    reason_code: Optional[str] = None


@dataclass(frozen=True)
class BudgetReceipt:
    """"The budget receipt reports rendered tokens, bytes, rows, graph hops,
    and elapsed materialization time." -- hierarchical-working-model.md,
    ZOOM protocol, "Request and response".
    """

    rendered_tokens: int
    rendered_bytes: int
    rendered_rows: int
    graph_hops: int
    elapsed_ms: float


# ---------------------------------------------------------------------------
# The canonical artifact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoomSlice:
    """THE compact-resumption / working-model artifact. See module docstring.

    Field-for-field against ZoomSliceV1 in hierarchical-working-model.md
    ("ZOOM protocol" > "Request and response"):

        ZoomSliceV1 {
          session_id, zoom_id, parent_zoom_id, snapshot_id,
          level: PLATFORM | APP | SUBSYSTEM,
          slice_digest, candidate_ids, selected_ids,
          facts, boundary_edges, unknowns, source_refs, bounded_excerpts,
          omitted: { count, reasons, continuation_token },
          budget_receipt, evidence_refs, next_actions
        }

    Two fields are additions beyond the literal doc quote, both required by
    W0-1's reduction of the other four docs into this one schema:

      * `profile`   -- which BudgetProfile this slice was rendered against.
                       Required so `validate_zoom_slice` can enforce the
                       right ceiling for a juggler pamphlet vs. a native
                       platform/app/subsystem slice vs. a decision stub,
                       without a second dataclass per consumer.
      * `token_count` -- the actual rendered token count, measured by the
                       caller's pinned tokenizer ("Budgets are enforced after
                       rendering with the pinned tokenizer used by the
                       kernel" -- hierarchical-working-model.md, "Context and
                       storage budgets"). This module does not tokenize
                       anything itself; it only enforces the ceiling once the
                       caller supplies a real count.
    """

    session_id: str
    zoom_id: str
    snapshot_id: str
    level: ZoomLevel
    slice_digest: str
    profile: BudgetProfile
    token_count: int

    parent_zoom_id: Optional[str] = None
    candidate_ids: Tuple[str, ...] = ()
    selected_ids: Tuple[str, ...] = ()
    facts: Tuple[SliceFact, ...] = ()
    boundary_edges: Tuple[BoundaryEdge, ...] = ()
    unknowns: Tuple[str, ...] = ()
    source_refs: Tuple[str, ...] = ()
    bounded_excerpts: Tuple[BoundedExcerpt, ...] = ()
    omitted: Omitted = field(default_factory=Omitted)
    budget_receipt: Optional[BudgetReceipt] = None
    evidence_refs: Tuple[str, ...] = ()
    next_actions: Tuple[NextAction, ...] = ()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationResult:
    """Pass/fail plus every reason found, so a test can assert on the reason
    text rather than only a boolean (per the task's requirement for a
    validator whose failures are diagnosable, not just a bool).
    """

    ok: bool
    reasons: Tuple[str, ...]

    def __bool__(self) -> bool:  # convenience: `if validate_zoom_slice(s):`
        return self.ok


def _ok() -> ValidationResult:
    return ValidationResult(ok=True, reasons=())


def _fail(reasons: Sequence[str]) -> ValidationResult:
    return ValidationResult(ok=False, reasons=tuple(reasons))


def validate_zoom_slice(zs: ZoomSlice) -> ValidationResult:
    """Validate a ZoomSlice against the reconciled schema invariants.

    Checks, in order (all run; every failure reason is collected, not just
    the first):

      1. required identity fields are non-empty strings
         (session_id, zoom_id, snapshot_id, slice_digest).
      2. `level` is a real ZoomLevel member.
      3. `selected_ids` is a subset of `candidate_ids` -- model invariant 5
         ("Closed expansion"): "A ZOOM step can select only opaque candidate
         IDs present in its parent slice."
      4. `token_count` does not exceed `profile.hard_max_tokens`. Meeting the
         target is not required (targets are soft); exceeding the HARD max
         is always a failure. Exactly at the hard max is allowed (a ceiling
         is inclusive, not exclusive, per every "X to Y tokens" / "hard
         ceiling: N" phrasing in the source docs -- none of them say
         "strictly less than").
      5. If `profile.hard_max_nodes`/`hard_max_edges`/`hard_max_rows` is set,
         the corresponding count on the slice (facts+boundary_edges treated
         as "nodes"/"edges" for the platform/app/subsystem profiles; the
         total row count via facts+boundary_edges+unknowns for row-budgeted
         profiles) does not exceed it.
      6. Every `SliceFact.layer` that is set resolves against
         `layer_registry.CanonicalLayer` -- this is what makes narration
         rehydration (see module docstring) safe to cite. Since `layer` is
         typed `Optional[CanonicalLayer]`, a caller can only violate this by
         constructing with a raw non-enum value; the check still runs
         defensively (an isinstance check) so a dict-shaped/duck-typed slice
         built outside this module's constructor is still caught, not just
         a real ZoomSlice instance.
      7. `level` is a real, recognized taxonomy member -- OVERLAY is
         accepted (W0-6 compatibility) but flagged with a reason string when
         it is used, since no source doc defines a real OVERLAY
         materializer yet (see ZoomLevel docstring). This is a WARNING-class
         reason: it does not flip `ok` to False by itself.
      8. `omitted.count > 0` requires at least one of `reasons` or
         `continuation_token` to be set -- an omission with neither is
         exactly the silent-drop failure family (#181c/#202 in the project
         rules): "not shown" must always be auditable.
      9. If `parent_zoom_id` is set, it must not equal `zoom_id` (a slice
         cannot be its own parent -- guards against a trivial cycle).
     10. `budget_receipt` is present. hierarchical-working-model.md's ZOOM
         protocol always renders one; a slice without it cannot prove its
         own budget compliance independent of the caller's claimed
         `token_count`.
    """
    reasons: List[str] = []
    warnings: List[str] = []

    # 1. required identity fields
    for name, value in (
        ("session_id", zs.session_id),
        ("zoom_id", zs.zoom_id),
        ("snapshot_id", zs.snapshot_id),
        ("slice_digest", zs.slice_digest),
    ):
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"required field '{name}' is missing or empty")

    # 2. level taxonomy
    if not isinstance(zs.level, ZoomLevel):
        reasons.append(
            f"'level' must be a ZoomLevel member, got {zs.level!r}"
        )
    elif zs.level is ZoomLevel.OVERLAY:
        warnings.append(
            "level=OVERLAY has no defined budget/materializer in "
            "hierarchical-working-model.md; accepted for W0-6 vocabulary "
            "compatibility only (see ZoomLevel docstring)"
        )

    # 3. closed expansion: selected subset of candidate
    candidate_set = set(zs.candidate_ids)
    stray_selections = [sid for sid in zs.selected_ids if sid not in candidate_set]
    if stray_selections:
        reasons.append(
            "closed-expansion violation (model invariant 5): selected_ids "
            f"not present in candidate_ids: {stray_selections}"
        )

    # 4. token budget
    if zs.token_count < 0:
        reasons.append(f"token_count cannot be negative: {zs.token_count}")
    elif zs.token_count > zs.profile.hard_max_tokens:
        reasons.append(
            f"token_count {zs.token_count} exceeds profile "
            f"'{zs.profile.name}' hard_max_tokens {zs.profile.hard_max_tokens} "
            f"(source: {zs.profile.source})"
        )

    # 5. node/edge/row budgets
    node_count = len(zs.facts)
    edge_count = len(zs.boundary_edges)
    row_count = len(zs.facts) + len(zs.boundary_edges) + len(zs.unknowns)

    if zs.profile.hard_max_nodes is not None and node_count > zs.profile.hard_max_nodes:
        reasons.append(
            f"fact ('node') count {node_count} exceeds profile "
            f"'{zs.profile.name}' hard_max_nodes {zs.profile.hard_max_nodes}"
        )
    if zs.profile.hard_max_edges is not None and edge_count > zs.profile.hard_max_edges:
        reasons.append(
            f"boundary_edge count {edge_count} exceeds profile "
            f"'{zs.profile.name}' hard_max_edges {zs.profile.hard_max_edges}"
        )
    if zs.profile.hard_max_rows is not None and row_count > zs.profile.hard_max_rows:
        reasons.append(
            f"row count {row_count} exceeds profile '{zs.profile.name}' "
            f"hard_max_rows {zs.profile.hard_max_rows}"
        )

    # 6. layer refs valid against the canonical registry
    for f in zs.facts:
        if f.layer is not None and not isinstance(f.layer, CanonicalLayer):
            reasons.append(
                f"fact '{f.fact_id}' has an invalid layer ref {f.layer!r}; "
                "must be a detreason.layer_registry.CanonicalLayer member"
            )

    # 8. omission auditability
    if zs.omitted.count > 0 and not zs.omitted.reasons and not zs.omitted.continuation_token:
        reasons.append(
            "omitted.count > 0 but neither reasons nor continuation_token "
            "is set -- an unauditable omission (silent-drop family, project "
            "rules #181c/#202)"
        )

    # 9. no self-parenting
    if zs.parent_zoom_id is not None and zs.parent_zoom_id == zs.zoom_id:
        reasons.append("parent_zoom_id equals zoom_id -- a slice cannot be its own parent")

    # 10. budget receipt present
    if zs.budget_receipt is None:
        reasons.append("budget_receipt is required and was not provided")

    all_reasons = tuple(reasons) + tuple(f"WARNING: {w}" for w in warnings)
    if reasons:
        return _fail(all_reasons)
    if warnings:
        return ValidationResult(ok=True, reasons=all_reasons)
    return _ok()


def validate_session_budget(slices: Sequence[ZoomSlice]) -> ValidationResult:
    """Validate a whole ZOOM session against the session-level ceiling.

    "One ZOOM session also has a default 4-slice/20K-token target and a hard
    32K-token ceiling." -- hierarchical-working-model.md. This checks the
    HARD ceiling only (32K tokens); the 4-slice/20K figure is a soft target
    and is reported as a warning when exceeded, mirroring the per-slice
    target/hard-max distinction in `validate_zoom_slice`.
    """
    total_tokens = sum(s.token_count for s in slices)
    reasons: List[str] = []
    warnings: List[str] = []

    if total_tokens > PROFILE_SESSION.hard_max_tokens:
        reasons.append(
            f"session total token_count {total_tokens} exceeds hard ceiling "
            f"{PROFILE_SESSION.hard_max_tokens} (source: {PROFILE_SESSION.source})"
        )
    elif total_tokens > PROFILE_SESSION.target_tokens:
        warnings.append(
            f"session total token_count {total_tokens} exceeds the soft "
            f"target {PROFILE_SESSION.target_tokens}"
        )
    if len(slices) > 4:
        warnings.append(
            f"session has {len(slices)} slices, exceeding the soft "
            "4-slice target (not a hard failure)"
        )

    all_reasons = tuple(reasons) + tuple(f"WARNING: {w}" for w in warnings)
    if reasons:
        return _fail(all_reasons)
    if warnings:
        return ValidationResult(ok=True, reasons=all_reasons)
    return _ok()
