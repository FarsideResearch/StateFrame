"""Schedule durable work units on transient model execution slots.

The scheduler tracks work state, classifies interrupt events, and preserves
receipts across slot changes. Logical work identity is independent of physical
slot ownership, so suspended work can resume without losing its history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence,
                    Tuple)

from detreason import obligations as OB
from detreason import receipt_contract as RC
from detreason import zoom_slice as ZS
from detreason.juggler_vocab import (LANE_BLOCKED, LANE_CHECKPOINTED,
                                     LANE_DONE, LANE_FAILED, LANE_LEASED,
                                     LANE_QUARANTINED, LANE_REGISTERED,
                                     LANE_SUPERSEDED, LANE_TERMINAL,
                                     LANE_UNBLOCKED, PRIORITY_P0_HUMAN_SAFETY,
                                     PRIORITY_P1_UNBLOCK,
                                     PRIORITY_P2_WORK,
                                     PRIORITY_P3_MAINTENANCE, PRIORITY_TIERS,
                                     Lane, lane_transition_allowed)

MODULE = "detreason.juggler"
VERSION = "1"

MICROS_PER_SECOND = OB.MICROS_PER_SECOND


class JugglerError(RuntimeError):
    """Base for control-plane programming errors (never a model's fault)."""


# ======================================================================
# COMPONENT CHECKLIST — a_juggler_interrupts.md "Genuinely missing wiring"
# ======================================================================
#
# The source doc enumerates eight components. Each is either IMPLEMENTED here
# or PUNTED with a reason. A test asserts this table stays in sync with the
# module's actual exports, so a punt cannot silently become a lie.

COMPONENT_CHECKLIST: Tuple[Tuple[str, str, str], ...] = (
    ("juggler_orchestration_loop", "implemented",
     "Juggler.step()/run_until_idle(): owns lane leases, the single model "
     "seat, inference completion and scheduling."),
    ("normalized_event_bus", "implemented",
     "EventBus: one EventEnvelope shape for every adapter, with dedup by "
     "event_id, per-source ordering detection, coalescing by coalesce_key "
     "(newest + every severity transition), a monotonic cursor and replay."),
    ("checkpoint_transaction", "implemented",
     "Juggler._commit_checkpoint(): receipt + lane state + consumed event "
     "cursor are staged and applied in one all-or-nothing swap; a fault at "
     "any staged point leaves zero partial mutations (TestCrashAtomicity)."),
    ("preemption_hook", "implemented",
     "preempt_requested is set at ingress; the action boundary refuses to "
     "execute unless a WORK_INTENT is durable, and refuses to START a new "
     "inference on a preempted lane. Generation is never killed mid-token."),
    ("lane_lifecycle_projection", "implemented",
     "LaneRegistry projects juggler_vocab's LANE_* statuses onto the same "
     "ledger vocabulary; no second ledger is created."),
    ("front_door_correlation", "partial",
     "Routing semantics (new_lane/continue_lane/answer_blocked_question/"
     "cancel_lane/supersede_lane) are implemented as HumanRouting + "
     "Juggler.submit_human_message. PUNT: the deterministic speech-act "
     "PARSER that derives instruction_mode from raw prose lives in "
     "front_door.py (B16's seam), not here — this module consumes an "
     "already-classified instruction_mode and refuses to guess one."),
    ("fairness_policy_and_telemetry", "partial",
     "Scheduler tiers + aging + deficit + monitor ceiling + source token "
     "buckets are implemented; Telemetry keeps MEASURED raw samples for human "
     "latency, dispatch wait and event lag. PUNT: checkpoint latency and "
     "replay divergence are not reported — this single-seat loop takes one "
     "`now` per tick, so both would be a constant 0, and a fabricated metric "
     "is worse than an absent one. They belong with B15's real clock."),
    ("crash_and_concurrency_tests", "implemented",
     "test_juggler.py: preemption at every boundary, duplicate/out-of-order "
     "events, human arrival during inference, monitor storms, child failure, "
     "lease loss, crash after intent before action, crash after action "
     "before receipt, and two lanes competing for one resource."),
)

#: Things the source docs describe that this module deliberately does NOT do.
#: Stated here so a reader never mistakes an absence for an oversight.
EXPLICIT_PUNTS: Tuple[Tuple[str, str], ...] = (
    ("llama_server_slot_pool",
     "B15 owns SlotPool/id_slot/cache_prompt. This module schedules onto an "
     "abstract seat count (Juggler(seats=N)); it never speaks the native API."),
    ("kv_cache_affinity",
     "cache_family affinity is a B15 concern. The switch-cost model here "
     "(SwitchCosts) takes resume cost as an injected number, so B15 can feed "
     "measured H_j without this module growing a cache."),
    ("speech_act_parser",
     "front_door.py owns turning prose into DIRECTIVE_NOW/MUSING. Guessing "
     "it here would give ambiguity preemption authority — banned by "
     "obligations.ParkedItem.preemption_allowed."),
    ("durable_storage",
     "Everything here is in-process append-only structures with explicit "
     "cursors. Persisting them is a storage-adapter concern; the atomic-swap "
     "commit is written so an outbox adapter can wrap it unchanged."),
)


# ======================================================================
# 1. EVENT ENVELOPE + DETERMINISTIC CLASSIFICATION
# ======================================================================

SEV_INFO = "info"
SEV_LOW = "low"
SEV_MEDIUM = "medium"
SEV_HIGH = "high"
SEV_CRITICAL = "critical"
SEVERITIES: Tuple[str, ...] = (SEV_INFO, SEV_LOW, SEV_MEDIUM, SEV_HIGH,
                               SEV_CRITICAL)
_SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}

# Event types, grouped by the doc paragraph that names them.
EV_HUMAN_MESSAGE = "human_message"
EV_CONTROL_COMMAND = "control_command"
EV_SAFETY_ALARM = "safety_alarm"
EV_LEASE_LOST = "lease_lost"
EV_WORKSPACE_DRIFT = "workspace_drift"
EV_BENCH_COMPLETED = "bench_completed"
EV_CHILD_COMPLETED = "child_completed"
EV_CHILD_FAILED = "child_failed"
EV_MONITOR_SAMPLE = "monitor_sample"
EV_QUESTION_ANSWERED = "question_answered"
EV_TIMER = "timer"
EV_TELEMETRY = "telemetry"
EV_MAINTENANCE = "maintenance"

EVENT_TYPES: Tuple[str, ...] = (
    EV_HUMAN_MESSAGE, EV_CONTROL_COMMAND, EV_SAFETY_ALARM, EV_LEASE_LOST,
    EV_WORKSPACE_DRIFT, EV_BENCH_COMPLETED, EV_CHILD_COMPLETED,
    EV_CHILD_FAILED, EV_MONITOR_SAMPLE, EV_QUESTION_ANSWERED, EV_TIMER,
    EV_TELEMETRY, EV_MAINTENANCE,
)

#: "a lightweight deterministic control parser should recognize stop, cancel,
#: approve, and redirect without waiting for a model slot."
CONTROL_COMMANDS: Tuple[str, ...] = ("stop", "cancel", "approve", "redirect")

DELIVER_INTERRUPT = "interrupt_now"
DELIVER_QUEUE = "queue"
DELIVER_MERGE = "merge"
DELIVERY_MODES: Tuple[str, ...] = (DELIVER_INTERRUPT, DELIVER_QUEUE,
                                   DELIVER_MERGE)


@dataclass(frozen=True)
class EventEnvelope:
    """``EventEnvelope`` from a_juggler_interrupts.md, field-for-field.

    ``payload`` is an addition: the doc carries only ``payload_digest`` +
    ``payload_ref`` because it assumes object storage. In-process we keep the
    payload itself, and ``payload_digest`` is still computed so a storage
    adapter can drop the inline copy without changing any consumer.
    """

    event_id: str
    event_type: str
    source: str
    source_sequence: int
    occurred_at: int
    workspace_id: str
    run_id: Optional[str] = None
    lane_id: Optional[str] = None
    parent_id: Optional[str] = None
    causation_id: Optional[str] = None
    correlation_id: str = ""
    severity: str = SEV_INFO
    coalesce_key: Optional[str] = None
    payload_ref: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise JugglerError("event_id is required")
        if self.event_type not in EVENT_TYPES:
            raise JugglerError(f"unknown event_type {self.event_type!r}")
        if self.severity not in SEVERITIES:
            raise JugglerError(f"unknown severity {self.severity!r}")
        if self.source_sequence < 0:
            raise JugglerError("source_sequence must be >= 0")
        if self.occurred_at < 0:
            raise JugglerError("occurred_at must be >= 0 (micros)")

    @property
    def payload_digest(self) -> str:
        return RC.digest(dict(self.payload))


@dataclass(frozen=True)
class Classification:
    """The classifier's verdict. ``reason_code`` is machine-readable on
    purpose: an interrupt that cannot say WHY it interrupted is not auditable
    (same discipline as zoom_slice's omission reasons)."""

    mode: str
    reason_code: str
    target_lane_id: Optional[str] = None


def classify_event(env: EventEnvelope, *,
                   active_lane_id: Optional[str] = None) -> Classification:
    """Deterministic code, never a model. Returns exactly one delivery mode.

    Order of the rules, and the doc sentence each one comes from:

      1. MERGE reducers first for the four event types the doc says "should
         never consume a model inference" — but only when they have a target
         lane to fold into. A bench completion with no lane to wake is not a
         reducer input, it is background noise -> QUEUE ("unrelated bench
         completions").
      2. ``child_failed`` splits on evidence, not on vibes: the doc puts it in
         BOTH lists. It is an interrupt only when the payload states it blocks
         a required parent obligation AND that parent is the active lane
         ("a child failure that blocks a required parent obligation"); the
         plain case is the reducer ("opens a failure obligation or moves the
         parent to blocked").
      3. A bench completion becomes an interrupt only when the payload says it
         invalidates the active lane's evidence ("A normal bench completion is
         not automatically an interrupt").
      4. Human/control/safety/lease/drift are the doc's interrupt list.
      5. Everything else queues.
    """
    lane = env.lane_id
    p = env.payload

    # --- 4a. human control is absolute and comes before every reducer ------
    if env.event_type == EV_HUMAN_MESSAGE:
        return Classification(DELIVER_INTERRUPT, "human_message", lane)
    if env.event_type == EV_CONTROL_COMMAND:
        cmd = str(p.get("command", "")).lower()
        if cmd in CONTROL_COMMANDS:
            return Classification(DELIVER_INTERRUPT, f"control:{cmd}", lane)
        # An unrecognized control word is NOT given interrupt authority.
        return Classification(DELIVER_QUEUE, "control:unrecognized", lane)
    if env.event_type in (EV_LEASE_LOST, EV_WORKSPACE_DRIFT):
        return Classification(DELIVER_INTERRUPT, env.event_type, lane)
    if env.event_type == EV_SAFETY_ALARM:
        if _SEV_RANK[env.severity] >= _SEV_RANK[SEV_HIGH]:
            return Classification(DELIVER_INTERRUPT, "safety_alarm", lane)
        return Classification(DELIVER_QUEUE, "alarm_below_threshold", lane)

    # --- 2. child failure: evidence-gated split ---------------------------
    if env.event_type == EV_CHILD_FAILED:
        blocks = bool(p.get("blocks_required_obligation"))
        if blocks and lane is not None and lane == active_lane_id:
            return Classification(DELIVER_INTERRUPT,
                                  "child_failed_blocks_parent", lane)
        if lane is None:
            return Classification(DELIVER_QUEUE, "child_failed_unrouted", None)
        return Classification(DELIVER_MERGE, "child_failed", lane)

    # --- 3. bench completion: interrupt only if it invalidates evidence ----
    if env.event_type == EV_BENCH_COMPLETED:
        if lane is None:
            return Classification(DELIVER_QUEUE, "bench_unrelated", None)
        if bool(p.get("invalidates_active_evidence")) and lane == active_lane_id:
            return Classification(DELIVER_INTERRUPT,
                                  "bench_invalidates_evidence", lane)
        return Classification(DELIVER_MERGE, "bench_completed", lane)

    # --- 1. remaining reducers -------------------------------------------
    if env.event_type in (EV_CHILD_COMPLETED, EV_MONITOR_SAMPLE,
                          EV_QUESTION_ANSWERED):
        if lane is None:
            return Classification(DELIVER_QUEUE, f"{env.event_type}_unrouted",
                                  None)
        return Classification(DELIVER_MERGE, env.event_type, lane)

    # --- 5. everything else ----------------------------------------------
    return Classification(DELIVER_QUEUE, env.event_type, lane)


# ======================================================================
# 2. EVENT BUS — durable append-only, dedup, ordering, coalescing, replay
# ======================================================================

@dataclass(frozen=True)
class BusEntry:
    cursor: int
    envelope: EventEnvelope
    duplicate: bool = False
    out_of_order: bool = False


class EventBus:
    """One normalized ingress for every adapter (bench, monitor, child-task,
    human). Append-only: a duplicate or out-of-order event is still RECORDED
    (as an audit row) — it just has no effect. That is the doc's "Late or
    duplicate completions become no-op audit records", generalized.
    """

    def __init__(self) -> None:
        self._log: List[BusEntry] = []
        self._seen_ids: Dict[str, int] = {}
        self._source_high_water: Dict[str, int] = {}

    # -- ingress ------------------------------------------------------
    def append(self, env: EventEnvelope) -> BusEntry:
        cursor = len(self._log) + 1
        if env.event_id in self._seen_ids:
            entry = BusEntry(cursor=cursor, envelope=env, duplicate=True)
            self._log.append(entry)
            return entry
        high = self._source_high_water.get(env.source)
        ooo = high is not None and env.source_sequence <= high
        entry = BusEntry(cursor=cursor, envelope=env, out_of_order=ooo)
        self._log.append(entry)
        self._seen_ids[env.event_id] = cursor
        if high is None or env.source_sequence > high:
            self._source_high_water[env.source] = env.source_sequence
        return entry

    # -- reads --------------------------------------------------------
    @property
    def head(self) -> int:
        return len(self._log)

    def since(self, cursor: int) -> Tuple[BusEntry, ...]:
        """Every entry strictly after ``cursor``, in append order."""
        return tuple(e for e in self._log if e.cursor > cursor)

    def effective_since(self, cursor: int) -> Tuple[BusEntry, ...]:
        """``since`` minus the no-op audit rows (duplicates, stale replays)."""
        return tuple(e for e in self.since(cursor)
                     if not e.duplicate and not e.out_of_order)

    def replay(self, from_cursor: int = 0) -> Tuple[EventEnvelope, ...]:
        """Deterministic replay of the effective event stream."""
        return tuple(e.envelope for e in self.effective_since(from_cursor))

    def all_entries(self) -> Tuple[BusEntry, ...]:
        return tuple(self._log)


def coalesce(entries: Sequence[BusEntry]) -> Tuple[BusEntry, ...]:
    """"Coalesce by (source, lane, alarm_type), keeping only the newest sample
    plus every severity transition."

    Implementation: group by ``coalesce_key`` (the adapter is responsible for
    composing it from source/lane/alarm_type — one key, one spelling, #223).
    Entries with no key are never coalesced. Within a group we keep the LAST
    entry of every severity RUN (so each transition is represented by its
    freshest sample) plus the final entry (the newest sample overall). Keeping
    the last of a run rather than the first is deliberate: a severity spike
    followed by quiet must not be represented by a stale sample, and a uniform
    storm must collapse to exactly one entry. Order of the surviving entries
    is the original append order, so replay stays deterministic.
    """
    keep: Dict[int, BusEntry] = {}
    groups: Dict[str, List[BusEntry]] = {}
    for e in entries:
        key = e.envelope.coalesce_key
        if key is None:
            keep[e.cursor] = e
            continue
        groups.setdefault(key, []).append(e)
    for _key, group in groups.items():
        for i, e in enumerate(group):
            nxt = group[i + 1] if i + 1 < len(group) else None
            if nxt is None or nxt.envelope.severity != e.envelope.severity:
                keep[e.cursor] = e          # last sample of this severity run
    return tuple(sorted(keep.values(), key=lambda e: e.cursor))


# ======================================================================
# 3. LANE RUNTIME + REGISTRY  (lane lifecycle projection)
# ======================================================================

#: A lane may receive an inference in these statuses. LEASED is excluded (it
#: is already running), BLOCKED/QUARANTINED/terminal are excluded by name.
SCHEDULABLE_STATUSES: Tuple[str, ...] = (LANE_REGISTERED, LANE_CHECKPOINTED,
                                         LANE_UNBLOCKED)

LANE_KIND_MONITOR = "monitor"


@dataclass
class LaneRuntime:
    """Scheduling state that hangs off the durable ``Lane`` record.

    The immutable ``Lane`` (juggler_vocab) stays the source of truth for
    status/lease_epoch/event_cursor/open_obligations. Everything here is
    scheduler bookkeeping that a_juggler_core.md's ``LaneRecord`` names but
    ``LaneState`` (the vocab shape) does not carry.
    """

    lane: Lane
    kind: str = "code"
    weight: int = 1
    base_priority: int = 50
    deadline_at: Optional[int] = None
    created_at: int = 0
    last_progress_at: int = 0
    last_scheduled_at: Optional[int] = None
    next_eligible_at: int = 0
    consecutive_quanta: int = 0
    failure_count: int = 0
    turns_remaining: Optional[int] = 8
    unblock_bonus_until: int = 0
    quarantine_reason: str = ""
    preempt_requested_by: Optional[str] = None
    pending_question_ids: Tuple[str, ...] = ()
    resources: Tuple[str, ...] = ()
    merged_deltas: Tuple[Mapping[str, Any], ...] = ()
    checkpoints: Tuple[RC.LaneCheckpoint, ...] = ()
    error_receipts: Tuple[Mapping[str, Any], ...] = ()
    work_intents: Tuple[Mapping[str, Any], ...] = ()

    @property
    def lane_id(self) -> str:
        return self.lane.lane_id

    @property
    def status(self) -> str:
        return self.lane.status

    @property
    def preempt_requested(self) -> bool:
        return self.preempt_requested_by is not None

    def state_digest(self) -> str:
        """Everything durable about this lane, hashed. Used both as the
        checkpoint's ``state_digest_before/after`` and as the isolation
        assertion in the tests: a sibling lane's digest must be byte-identical
        before and after another lane fails."""
        return RC.digest({
            "lane": {
                "lane_id": self.lane.lane_id,
                "status": self.lane.status,
                "event_cursor": self.lane.event_cursor,
                "lease_epoch": self.lane.lease_epoch,
                "receipt_head": self.lane.receipt_head,
                "open_obligations": list(self.lane.open_obligations),
                "priority_class": self.lane.priority_class,
                "service_debt": self.lane.service_debt,
                "pending_question_id": self.lane.pending_question_id,
            },
            "failure_count": self.failure_count,
            "turns_remaining": self.turns_remaining,
            "merged_deltas": [dict(d) for d in self.merged_deltas],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "error_receipts": [dict(r) for r in self.error_receipts],
            "work_intents": [dict(i) for i in self.work_intents],
        })


class LaneRegistry:
    """Owns every lane and the ONLY sanctioned mutation paths. No caller
    reaches into ``LaneRuntime.lane`` to set a status — that would bypass
    ``lane_transition_allowed`` and re-open the illegal-transition hole
    juggler_vocab closes."""

    def __init__(self) -> None:
        self._lanes: Dict[str, LaneRuntime] = {}

    # -- registration --------------------------------------------------
    def register(self, *, lane_id: str, contract_id: str,
                 parent_request_id: str = "", now: int = 0,
                 priority_class: str = PRIORITY_P2_WORK, kind: str = "code",
                 weight: int = 1, base_priority: int = 50,
                 deadline_at: Optional[int] = None, turns: Optional[int] = 8,
                 resources: Sequence[str] = (),
                 open_obligations: Sequence[str] = ()) -> LaneRuntime:
        if lane_id in self._lanes:
            raise JugglerError(f"work stream {lane_id!r} already registered")
        if priority_class not in PRIORITY_TIERS:
            raise JugglerError(f"unknown priority_class {priority_class!r}")
        lane = Lane(lane_id=lane_id, contract_id=contract_id,
                    parent_request_id=parent_request_id,
                    priority_class=priority_class,
                    open_obligations=tuple(open_obligations))
        rt = LaneRuntime(lane=lane, kind=kind, weight=max(1, weight),
                         base_priority=base_priority, deadline_at=deadline_at,
                         created_at=now, last_progress_at=now,
                         next_eligible_at=now, turns_remaining=turns,
                         resources=tuple(resources))
        self._lanes[lane_id] = rt
        return rt

    # -- reads ---------------------------------------------------------
    def get(self, lane_id: str) -> LaneRuntime:
        try:
            return self._lanes[lane_id]
        except KeyError:
            raise JugglerError(f"unknown work stream {lane_id!r}") from None

    def has(self, lane_id: str) -> bool:
        return lane_id in self._lanes

    def all(self) -> Tuple[LaneRuntime, ...]:
        return tuple(self._lanes[k] for k in sorted(self._lanes))

    def digests(self) -> Dict[str, str]:
        return {lid: rt.state_digest() for lid, rt in self._lanes.items()}

    # -- transitions ---------------------------------------------------
    def set_status(self, lane_id: str, target: str) -> LaneRuntime:
        rt = self.get(lane_id)
        if not lane_transition_allowed(rt.lane.status, target):
            raise JugglerError(
                f"illegal lane transition {rt.lane.status!r} -> {target!r} "
                f"for lane {lane_id!r}")
        rt.lane = rt.lane.with_status(target)
        return rt

    def update_lane(self, lane_id: str, **changes: Any) -> LaneRuntime:
        """Non-status field updates on the immutable Lane record."""
        if "status" in changes:
            raise JugglerError("use set_status() for status changes")
        rt = self.get(lane_id)
        rt.lane = replace(rt.lane, **changes)
        return rt


# ======================================================================
# 4. SCHEDULER — tiers, score, fairness, anti-starvation
# ======================================================================

@dataclass(frozen=True)
class SchedulerPolicy:
    """Every knob the two docs name, with its source. No hidden constants.

    The numeric defaults are the doc's own illustrative values where it gives
    one ("a starvation threshold, for example 30 seconds"; "for example 25% of
    background quanta"; "maximum consecutive turns per lane, normally two").
    They are policy, not physics — a caller overrides them per deployment.
    """

    max_consecutive_quanta: int = 2                 # core.md "normally two"
    starvation_threshold_micros: int = 30 * MICROS_PER_SECOND
    monitor_capacity_fraction_num: int = 1          # 25% == 1/4, exact ints
    monitor_capacity_fraction_den: int = 4
    unblock_bonus: int = 120
    unblock_bonus_window_micros: int = 5 * MICROS_PER_SECOND
    age_cap_micros: int = 60 * MICROS_PER_SECOND
    age_weight: int = 150
    deadline_weight: int = 300
    service_debt_weight: int = 40
    retry_backoff_base_micros: int = 1 * MICROS_PER_SECOND
    quarantine_after_failures: int = 3
    source_bucket_capacity: int = 64                # per-source token bucket

    def __post_init__(self) -> None:
        if self.max_consecutive_quanta < 1:
            raise JugglerError("max_consecutive_quanta must be >= 1")
        if self.monitor_capacity_fraction_den < 1:
            raise JugglerError("monitor capacity denominator must be >= 1")
        if self.quarantine_after_failures < 1:
            raise JugglerError("quarantine_after_failures must be >= 1")


@dataclass(frozen=True)
class Candidate:
    lane_id: str
    tier: str
    score: int
    last_progress_at: int
    starved: bool


def _tier_index(priority_class: str) -> int:
    return PRIORITY_TIERS.index(priority_class)


class Scheduler:
    """Hard priority tiers first, fair scheduling within each tier.

    Two rules interact and the order matters:

      * P0 (human/safety) is ABSOLUTE for dispatch ordering — the interrupts
        doc says so in as many words ("Human priority remains absolute for
        dispatch ordering").
      * Aging is the starvation guard for everything BELOW P0. A starved P2
        lane is promoted above P1/P3, never above P0. If aging could jump P0
        we would have re-invented the priority inversion the tiers exist to
        prevent.
    """

    def __init__(self, policy: Optional[SchedulerPolicy] = None) -> None:
        self.policy = policy or SchedulerPolicy()
        self.background_quanta_served = 0
        self.monitor_quanta_served = 0

    # -- eligibility ---------------------------------------------------
    def eligible(self, registry: LaneRegistry, *, now: int,
                 held_resources: Mapping[str, str] = {},
                 relax_consecutive: bool = False) -> Tuple[LaneRuntime, ...]:
        out: List[LaneRuntime] = []
        for rt in registry.all():
            if rt.status not in SCHEDULABLE_STATUSES:
                continue
            if rt.status in LANE_TERMINAL:
                continue
            if rt.preempt_requested:
                continue                       # no NEW inference on a preempted lane
            if rt.next_eligible_at > now:
                continue
            if rt.turns_remaining is not None and rt.turns_remaining <= 0:
                continue
            if not relax_consecutive and \
                    rt.consecutive_quanta >= self.policy.max_consecutive_quanta:
                continue
            if any(held_resources.get(r) not in (None, rt.lane_id)
                   for r in rt.resources):
                continue
            out.append(rt)
        return tuple(out)

    # -- scoring -------------------------------------------------------
    def score(self, rt: LaneRuntime, *, now: int) -> int:
        """``priority + bounded_age + unblock_bonus + deadline_pressure
        - weighted_service_time`` (a_juggler_interrupts.md, "Within a tier").

        Integer arithmetic throughout: a float score makes tie-breaking
        depend on rounding, and a scheduler whose order depends on rounding is
        not testably deterministic.
        """
        p = self.policy
        score = rt.base_priority

        # staleness — measured from last SUCCESSFUL progress, never from last
        # scheduling ("Merely giving a lane model turns should not make it
        # appear fresh." — core.md).
        age = max(0, now - rt.last_progress_at)
        score += (min(age, p.age_cap_micros) * p.age_weight) // max(1, p.age_cap_micros)

        # bounded unblock boost
        if now < rt.unblock_bonus_until:
            score += p.unblock_bonus

        # deadline pressure rises as the deadline approaches
        if rt.deadline_at is not None:
            remaining = rt.deadline_at - now
            if remaining <= 0:
                score += p.deadline_weight
            else:
                window = max(1, p.age_cap_micros)
                near = max(0, window - min(remaining, window))
                score += (near * p.deadline_weight) // window

        # weighted service time (deficit accounting)
        score -= int((rt.lane.service_debt * p.service_debt_weight)
                     // max(1, rt.weight))
        return score

    def starved(self, rt: LaneRuntime, *, now: int) -> bool:
        return (now - rt.last_progress_at) >= self.policy.starvation_threshold_micros

    def monitor_over_ceiling(self) -> bool:
        """A monitor lane gets at most ``fraction`` of BACKGROUND quanta.
        Background == everything served below P0, which is the population the
        doc's "25% of background quanta" is a fraction of."""
        p = self.policy
        served = self.background_quanta_served
        if served == 0:
            return False
        allowed = (served * p.monitor_capacity_fraction_num) \
            // p.monitor_capacity_fraction_den
        return self.monitor_quanta_served >= max(1, allowed)

    # -- selection -----------------------------------------------------
    def rank(self, registry: LaneRegistry, *, now: int,
             held_resources: Mapping[str, str] = {}) -> Tuple[Candidate, ...]:
        cands = self.eligible(registry, now=now, held_resources=held_resources)
        if not cands:
            # Anti-starvation release valve: if the ONLY thing blocking every
            # lane is the consecutive-quanta cap, relax it rather than idle.
            cands = self.eligible(registry, now=now,
                                  held_resources=held_resources,
                                  relax_consecutive=True)
        rows: List[Candidate] = []
        for rt in cands:
            if rt.kind == LANE_KIND_MONITOR and self.monitor_over_ceiling() \
                    and rt.lane.priority_class != PRIORITY_P0_HUMAN_SAFETY:
                continue
            rows.append(Candidate(
                lane_id=rt.lane_id,
                tier=rt.lane.priority_class,
                score=self.score(rt, now=now),
                last_progress_at=rt.last_progress_at,
                starved=self.starved(rt, now=now),
            ))

        def sort_key(c: Candidate) -> Tuple[int, int, int, int, str]:
            p0 = 0 if c.tier == PRIORITY_P0_HUMAN_SAFETY else 1
            # starvation override lifts a lane above its tier, never above P0
            tier_rank = 0 if (p0 == 1 and c.starved) else _tier_index(c.tier)
            return (p0, tier_rank, -c.score, c.last_progress_at, c.lane_id)

        rows.sort(key=sort_key)
        return tuple(rows)

    def choose(self, registry: LaneRegistry, *, now: int,
               held_resources: Mapping[str, str] = {}) -> Optional[str]:
        ranked = self.rank(registry, now=now, held_resources=held_resources)
        return ranked[0].lane_id if ranked else None

    def note_served(self, rt: LaneRuntime) -> None:
        if rt.lane.priority_class != PRIORITY_P0_HUMAN_SAFETY:
            self.background_quanta_served += 1
            if rt.kind == LANE_KIND_MONITOR:
                self.monitor_quanta_served += 1


# -- switch-rate policy (a_juggler_switch.md section 3) -------------------

def min_quantum_tokens(*, decode_tokens_per_second: float,
                       switch_overhead_seconds: float,
                       target_overhead_fraction: float) -> int:
    """``Q_min = ceil(d * H * (1 - a) / a)``.

    Verbatim from the switch doc. Its own worked example (d=40, H=0.4,
    a=0.15 -> ~91) is a test case, not a constant baked in here — every one of
    those numbers is deployment-measured.
    """
    a = target_overhead_fraction
    if not 0 < a < 1:
        raise JugglerError("target_overhead_fraction must be in (0, 1)")
    if decode_tokens_per_second <= 0 or switch_overhead_seconds < 0:
        raise JugglerError("decoding rate must be > 0 and overhead >= 0")
    return int(math.ceil(decode_tokens_per_second * switch_overhead_seconds
                         * (1 - a) / a))


@dataclass(frozen=True)
class SwitchCosts:
    """The switch doc's cost variables. Every one is INJECTED — this module
    measures none of them, and inventing a number here would be exactly the
    "illustrative throughput treated as fact" landmine B15 warns about."""

    finish_current_micros: int          # F_i
    resume_candidate_micros: int        # H_j
    displaced_resume_micros: int        # R_i
    interrupt_coherence_cost: int       # L_i
    candidate_staleness_penalty: int    # A_j(F_i)


def should_switch(costs: SwitchCosts) -> bool:
    """``Switch to j when A_j(F_i) > H_j + R_i + L_i``."""
    return costs.candidate_staleness_penalty > (
        costs.resume_candidate_micros + costs.displaced_resume_micros
        + costs.interrupt_coherence_cost)


# ======================================================================
# 5. MODEL SEAT CONTRACT (modelless: injected callable)
# ======================================================================

ACT_YIELD = "yield"
ACT_RETRIEVE = "retrieve"
ACT_START_IO = "start_io"
ACT_ASK_QUESTION = "ask_question"
ACT_COMPLETE = "complete"
ACTION_KINDS: Tuple[str, ...] = (ACT_YIELD, ACT_RETRIEVE, ACT_START_IO,
                                 ACT_ASK_QUESTION, ACT_COMPLETE)

# Validator error codes — a_juggler_switch.md "Useful validator outputs".
ERR_WRONG_LANE = "WRONG_LANE"
ERR_CHECKPOINT_MISMATCH = "CHECKPOINT_MISMATCH"
ERR_MISSING_NEXT_ACTION = "MISSING_NEXT_ACTION"
ERR_UNKNOWN_REQUEST = "UNKNOWN_REQUEST"
ERR_CONSTRAINT_CONFLICT = "CONSTRAINT_CONFLICT"
ERR_DUPLICATE_SIDE_EFFECT = "DUPLICATE_SIDE_EFFECT"
ERR_WRONG_PHASE = "WRONG_PHASE"
ERR_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
VALIDATOR_ERROR_CODES: Tuple[str, ...] = (
    ERR_WRONG_LANE, ERR_CHECKPOINT_MISMATCH, ERR_MISSING_NEXT_ACTION,
    ERR_UNKNOWN_REQUEST, ERR_CONSTRAINT_CONFLICT, ERR_DUPLICATE_SIDE_EFFECT,
    ERR_WRONG_PHASE, ERR_BUDGET_EXHAUSTED,
)


@dataclass(frozen=True)
class SeatRequest:
    """What the model seat receives. Exactly one lane, ever.

    This is a REQUEST envelope, not a state schema of its own: the ``pamphlet``
    field simply carries a ``zoom_slice.ZoomSlice`` (the wave-0 W0-1 winner),
    and the juggler validates that slice before handing it over — an
    over-budget slice is a control-plane bug, not something to discover in the
    model's output.
    """

    lane_id: str
    base_revision: int
    pamphlet: ZS.ZoomSlice
    latest_event: Optional[EventEnvelope] = None
    merged_deltas: Tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ModelStep:
    """The seat's typed result — an INTENT, never a mutation."""

    lane_id: str
    base_revision: int
    step_id: str
    action_kind: str
    action: Mapping[str, Any] = field(default_factory=dict)
    procedure_receipt: Optional[Mapping[str, Any]] = None
    obligations_opened: Tuple[str, ...] = ()
    obligations_closed: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action_kind not in ACTION_KINDS:
            raise JugglerError(f"unknown action_kind {self.action_kind!r}")

    def output_digest(self) -> str:
        return RC.digest({
            "lane_id": self.lane_id, "base_revision": self.base_revision,
            "step_id": self.step_id, "action_kind": self.action_kind,
            "action": dict(self.action),
            "obligations_opened": list(self.obligations_opened),
            "obligations_closed": list(self.obligations_closed),
        })


ModelSeat = Callable[[SeatRequest], ModelStep]


# ======================================================================
# 6. PAMPHLET COMPILATION (ZoomSlice, never a second schema)
# ======================================================================

def compile_pamphlet(rt: LaneRuntime, *, session_id: str,
                     profile: ZS.BudgetProfile = ZS.PROFILE_JUGGLER_DECISION_STUB,
                     token_estimator: Optional[Callable[[Sequence[str]], int]] = None,
                     ) -> ZS.ZoomSlice:
    """Build the lane's rehydration payload as a ``ZoomSlice``.

    Profile choice follows zoom_slice.py's own note: the DECISION_STUB is the
    default for a routine scheduler switch; a caller passes
    ``PROFILE_JUGGLER_PAMPHLET`` for cold start / post-compaction. Compression
    is state PROJECTION (deterministic code reading lane state), never
    free-form summarization — the switch doc is explicit that a summarizer may
    never delete the completion condition, a constraint, an active blocker, an
    in-flight request, or the next action.

    ``token_estimator`` is injected because this module owns no tokenizer;
    the default is a deliberately crude word count, and callers with the
    pinned kernel tokenizer pass theirs.
    """
    lane = rt.lane
    facts: List[ZS.SliceFact] = []
    facts.append(ZS.SliceFact(fact_id="lane", text=f"lane={lane.lane_id} "
                              f"status={lane.status} rev={lane.lease_epoch}",
                              source_ref=f"lane://{lane.lane_id}"))
    facts.append(ZS.SliceFact(fact_id="contract",
                              text=f"contract={lane.contract_id}",
                              source_ref=f"lane://{lane.lane_id}/contract"))
    for i, ob_id in enumerate(lane.open_obligations):
        facts.append(ZS.SliceFact(fact_id=f"obligation-{i}",
                                  text=f"open obligation {ob_id}",
                                  source_ref=f"obligation://{ob_id}"))
    for i, delta in enumerate(rt.merged_deltas):
        facts.append(ZS.SliceFact(
            fact_id=f"delta-{i}",
            text=f"{delta.get('kind', 'delta')}: {delta.get('summary', '')}",
            source_ref=str(delta.get("source_ref", "event://merged"))))

    unknowns: List[str] = []
    if lane.pending_question_id:
        unknowns.append(f"awaiting answer to {lane.pending_question_id}")

    next_actions = (ZS.NextAction(action=ZS.NextActionKind.SELECT,
                                  reason_code="scheduler_dispatch"),)

    texts = [f.text for f in facts] + unknowns
    if token_estimator is not None:
        token_count = token_estimator(texts)
    else:
        token_count = sum(len(t.split()) for t in texts)

    body = {
        "lane_id": lane.lane_id,
        "facts": [(f.fact_id, f.text, f.source_ref) for f in facts],
        "unknowns": unknowns,
        "cursor": lane.event_cursor,
        "receipt_head": lane.receipt_head,
    }
    slice_digest = RC.digest(body)

    return ZS.ZoomSlice(
        session_id=session_id,
        zoom_id=f"{lane.lane_id}@{lane.lease_epoch}",
        snapshot_id=lane.snapshot_digest or "snapshot://none",
        level=ZS.ZoomLevel.SUBSYSTEM,
        slice_digest=slice_digest,
        profile=profile,
        token_count=token_count,
        facts=tuple(facts),
        unknowns=tuple(unknowns),
        next_actions=next_actions,
        budget_receipt=ZS.BudgetReceipt(rendered_tokens=token_count,
                                        rendered_bytes=len(str(body)),
                                        rendered_rows=len(facts) + len(unknowns),
                                        graph_hops=0, elapsed_ms=0.0),
    )


# ======================================================================
# 7. TELEMETRY
# ======================================================================

@dataclass
class Telemetry:
    """Item 7 of the missing-wiring list. Counters only — no sampling, no
    averaging into a lifetime cumulative rate (project rule #160: a displayed
    rate comes from a window over raw events, so we keep the raw events)."""

    ticks: int = 0
    inferences: int = 0
    events_ingested: int = 0
    events_duplicate: int = 0
    events_out_of_order: int = 0
    events_merged: int = 0
    events_queued: int = 0
    events_interrupted: int = 0
    events_coalesced_away: int = 0
    events_dropped_by_bucket: int = 0
    preemptions: int = 0
    checkpoints: int = 0
    invalid_model_outputs: int = 0
    lane_failures: int = 0
    quarantines: int = 0
    starvation_overrides: int = 0
    resource_contentions: int = 0
    switches: int = 0
    human_messages: int = 0
    parks: int = 0
    #: raw (lane_id, micros) samples — the reader windows them (#160). Every
    #: sample here is MEASURED; there is deliberately no sample recorded for a
    #: quantity this single-seat loop cannot observe (in-step checkpoint
    #: latency is 0 by construction, so it is not reported as if it were data).
    #: human_latency: dispatch time minus human-lane creation time, recorded
    #: once, when the human lane first gets the seat.
    human_latency_samples: List[Tuple[str, int]] = field(default_factory=list)
    #: dispatch_wait: how long the served lane had gone without progress.
    dispatch_wait_samples: List[Tuple[str, int]] = field(default_factory=list)
    event_lag_samples: List[Tuple[str, int]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v)
                for k, v in self.__dict__.items()}


# ======================================================================
# 8. HUMAN ROUTING (front-door correlation — consumes, never guesses)
# ======================================================================

ROUTE_NEW_LANE = "new_lane"
ROUTE_CONTINUE_LANE = "continue_lane"
ROUTE_ANSWER_QUESTION = "answer_blocked_question"
ROUTE_CANCEL_LANE = "cancel_lane"
ROUTE_SUPERSEDE_LANE = "supersede_lane"
HUMAN_ROUTES: Tuple[str, ...] = (ROUTE_NEW_LANE, ROUTE_CONTINUE_LANE,
                                 ROUTE_ANSWER_QUESTION, ROUTE_CANCEL_LANE,
                                 ROUTE_SUPERSEDE_LANE)


@dataclass(frozen=True)
class HumanRouting:
    """front_door output, as this module needs it. ``instruction_mode`` comes
    from ``obligations.INSTRUCTION_MODES`` — we consume the classification,
    we never derive it (see EXPLICIT_PUNTS/speech_act_parser)."""

    route: str
    instruction_mode: str
    target_lane_id: Optional[str] = None
    question_id: Optional[str] = None
    contract_id: str = "human"
    park_kind: str = "question"

    def __post_init__(self) -> None:
        if self.route not in HUMAN_ROUTES:
            raise JugglerError(f"unknown route {self.route!r}")
        if self.instruction_mode not in OB.INSTRUCTION_MODES:
            raise JugglerError(
                f"unknown instruction_mode {self.instruction_mode!r}")
        if self.route == ROUTE_ANSWER_QUESTION and not self.question_id:
            raise JugglerError(
                "answer_blocked_question requires an explicit question_id "
                "correlation (a_juggler_interrupts.md: 'A response to a "
                "blocked question requires an explicit correlation')")
        if self.route in (ROUTE_CONTINUE_LANE, ROUTE_CANCEL_LANE,
                          ROUTE_SUPERSEDE_LANE) and not self.target_lane_id:
            raise JugglerError(f"{self.route} requires target_lane_id")


# ======================================================================
# 9. THE JUGGLER
# ======================================================================

@dataclass(frozen=True)
class Tick:
    """What one ``step()`` did. Every field is observable evidence, so a test
    (and an operator) can tell a no-op apart from a served quantum."""

    served_lane_id: Optional[str]
    action_kind: Optional[str] = None
    checkpoint: Optional[RC.LaneCheckpoint] = None
    validation_errors: Tuple[str, ...] = ()
    preempted_by_event_id: Optional[str] = None
    merged_event_ids: Tuple[str, ...] = ()
    interrupt_event_ids: Tuple[str, ...] = ()
    queued_event_ids: Tuple[str, ...] = ()
    idle_reason: str = ""


class Juggler:
    """The orchestration loop.

    SINGLE-SEAT ONLY, on purpose. The interrupts doc's "reserve one slot for
    foreground control" needs concurrent dispatch, which needs the slot pool
    B15 owns. Rather than accept a ``seats`` argument that silently does
    nothing (the #223 reader-with-no-writer failure, inverted), ``seats > 1``
    is REFUSED with a message naming its owner.
    """

    def __init__(self, *, session_id: str, workspace_id: str, seat: ModelSeat,
                 policy: Optional[SchedulerPolicy] = None,
                 ledger: Optional[OB.ObligationLedger] = None,
                 seats: int = 1,
                 pamphlet_profile: ZS.BudgetProfile = ZS.PROFILE_JUGGLER_DECISION_STUB,
                 action_executor: Optional[Callable[[str, ModelStep], Mapping[str, Any]]] = None,
                 fault_hook: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
                 planned_artifact_root: Optional[str] = None,
                 ) -> None:
        if seats != 1:
            raise JugglerError(
                "this control plane dispatches ONE seat per tick; concurrent "
                "seats (and the reserved foreground slot) require the B15 "
                "slot pool — see EXPLICIT_PUNTS['llama_server_slot_pool']")
        self.session_id = session_id
        self.workspace_id = workspace_id
        self.seat = seat
        self.seats = seats
        self.registry = LaneRegistry()
        self.bus = EventBus()
        self.scheduler = Scheduler(policy)
        self.telemetry = Telemetry()
        self.ledger = ledger or OB.ObligationLedger(workspace_id=workspace_id,
                                                    session_id=session_id)
        self.pamphlet_profile = pamphlet_profile
        #: mechanical action layer. Default is a no-op that reports success —
        #: real deployments inject the chain runner. It is called ONLY after
        #: WORK_INTENT is durable.
        self.action_executor = action_executor or (
            lambda lane_id, step: {"status": "ok"})
        #: test seam for crash injection: called at each named commit point.
        self.fault_hook = fault_hook or (lambda point, ctx: None)

        self._cursor = 0                        # scheduler-owned event cursor
        self._deferred: List[BusEntry] = []     # queue-mode backlog
        self._held_resources: Dict[str, str] = {}
        self._source_buckets: Dict[str, int] = {}
        self._last_served: Optional[str] = None
        self._question_index: Dict[str, str] = {}   # question_id -> lane_id
        self._idempotency_seen: Dict[str, str] = {}  # key -> lane_id
        # planned artifacts remains lazy so the legacy path performs no extra imports or
        # scheduling work while STATEFRAME_PLANNED_ARTIFACTS is off.
        self._planned_artifacts = None
        self._return_ledger = None
        if planned_artifact_root is not None:
            self.enable_planned_artifacts(planned_artifact_root)

    # --------------------------------------------------------- planned artifacts seam
    def enable_planned_artifacts(self, root: str, *, package_roots: Sequence[str] = ()):
        """Attach planned inputs to this juggler; dispatch still uses one scheduler."""
        from kernel_planned_artifacts import PlannedArtifactLedger, ReturnLedger, enabled
        if not enabled():
            return None
        if self._planned_artifacts is None:
            self._planned_artifacts = PlannedArtifactLedger(
                root, workspace_id=self.workspace_id, session_id=self.session_id,
                obligation_ledger=self.ledger, package_roots=package_roots)
            self._return_ledger = ReturnLedger()
        return self._planned_artifacts

    @property
    def planned_artifacts(self):
        return self._planned_artifacts

    @property
    def return_ledger(self):
        return self._return_ledger

    def declare_artifact(self, *args, **kwargs):
        if self._planned_artifacts is None:
            raise JugglerError("planned artifacts are not enabled for this juggler")
        return self._planned_artifacts.declare_artifact(*args, **kwargs)

    def infer_planned_candidates(self, source_path: str, source: Optional[str] = None):
        if self._planned_artifacts is None:
            return ()
        return self._planned_artifacts.infer_candidates(source_path, source)

    def planned_readiness(self, check_id: str, inputs=(), **kwargs):
        if self._planned_artifacts is None:
            return None
        return self._planned_artifacts.readiness(check_id, inputs, **kwargs)

    def planned_completion_blockers(self, *, now: int = 0):
        if self._planned_artifacts is None:
            return ()
        return self._planned_artifacts.completion_blockers(now=now)

    def capture_resumption_cue(self, parent_id: str, *, interrupted_by: str,
                               focus: str, last_evidence=(), return_target: str = "",
                               trigger: str, owed_items=(), waiting_checks=(),
                               priority: int = 0, affinity: int = 0, now: int = 0,
                               proposed_next_action: str = "", snapshot_refs=()):
        if self._return_ledger is None:
            return None
        return self._return_ledger.interrupt(
            parent_id, interrupted_by=interrupted_by, focus=focus,
            last_evidence=last_evidence, return_target=return_target,
            trigger=trigger, owed_items=owed_items, waiting_checks=waiting_checks,
            priority=priority, affinity=affinity, now=now,
            proposed_next_action=proposed_next_action, snapshot_refs=snapshot_refs)

    def mark_continuation_return(self, child_id: str, *, now: int = 0,
                                 evidence=(), receipt: str = ""):
        if self._return_ledger is None:
            return ()
        return self._return_ledger.mark_complete(
            child_id, now=now, evidence=evidence, receipt=receipt)

    def reconcile_continuation(self, parent_id: str, *, evidence=(), now: int = 0):
        if self._return_ledger is None:
            return None
        return self._return_ledger.reconcile(parent_id, evidence=evidence, now=now)

    def continuation_projection(self, parent_id: str):
        if self._return_ledger is None:
            return ""
        return self._return_ledger.projection(parent_id)

    def continuation_projection_dict(self, parent_id: str):
        if self._return_ledger is None:
            return {}
        return self._return_ledger.projection_dict(parent_id)

    # ------------------------------------------------------------------
    # ingress
    # ------------------------------------------------------------------
    def submit(self, env: EventEnvelope) -> BusEntry:
        """Durably append FIRST, classify second. Never the other way round."""
        entry = self.bus.append(env)
        self.telemetry.events_ingested += 1
        if entry.duplicate:
            self.telemetry.events_duplicate += 1
        if entry.out_of_order:
            self.telemetry.events_out_of_order += 1
        return entry

    def submit_human_message(self, *, event_id: str, text: str,
                             routing: HumanRouting, now: int,
                             source: str = "human",
                             source_sequence: int = 0) -> BusEntry:
        """Human requests take priority under the scheduling protocol.

        Persisted immediately; outranks every pending lane; the CURRENT
        inference is allowed to finish its bounded quantum; no new inference
        or side effect begins on the interrupted lane.

        A message whose ``instruction_mode`` is not DIRECTIVE_NOW is PARKED
        (obligations.ObligationLedger.park) rather than given preemption
        authority. That is the park integration: a park mints exactly one
        return obligation, and return obligations are schedulable work
        (``obligation_work_items``).
        """
        env = EventEnvelope(
            event_id=event_id, event_type=EV_HUMAN_MESSAGE, source=source,
            source_sequence=source_sequence, occurred_at=now,
            workspace_id=self.workspace_id, lane_id=routing.target_lane_id,
            severity=SEV_HIGH, correlation_id=routing.question_id or "",
            payload={"text": text, "route": routing.route,
                     "instruction_mode": routing.instruction_mode,
                     "question_id": routing.question_id,
                     "contract_id": routing.contract_id})
        entry = self.submit(env)
        self.telemetry.human_messages += 1

        # Capture the return cue at ingress, before the next dispatch can
        # change the active lane.  This is scheduling evidence, not a verdict.
        if self._return_ledger is not None and self._last_served:
            checkpoint_ref = self._active_checkpoint_ref()
            owed = (tuple(self.registry.get(self._last_served).lane.open_obligations)
                    if self.registry.has(self._last_served) else ())
            self.capture_resumption_cue(
                self._last_served, interrupted_by=event_id,
                focus=text[:240], last_evidence=(checkpoint_ref,) if checkpoint_ref else (),
                return_target=self._last_served, trigger="human interruption",
                owed_items=owed, waiting_checks=(), priority=0, affinity=1, now=now,
                proposed_next_action="reconcile the parent and resume its owed work",
                snapshot_refs=(checkpoint_ref,) if checkpoint_ref else ())

        if routing.instruction_mode not in ("DIRECTIVE_NOW",):
            item = OB.ParkedItem(
                park_id=f"park:{event_id}", session_id=self.session_id,
                workspace_id=self.workspace_id, source_event_id=event_id,
                raw_message=text, captured_at=now, kind=routing.park_kind,
                instruction_mode=routing.instruction_mode,
                priority_class=OB.P1 if routing.route == ROUTE_ANSWER_QUESTION
                else OB.P2,
                active_lane_id=self._last_served or "",
                active_checkpoint_ref=self._active_checkpoint_ref(),
                context_capsule={"route": routing.route,
                                 "target_lane_id": routing.target_lane_id},
            )
            self.ledger.park(item, now=now)
            self.telemetry.parks += 1
        return entry

    def _active_checkpoint_ref(self) -> str:
        if self._last_served and self.registry.has(self._last_served):
            rt = self.registry.get(self._last_served)
            if rt.checkpoints:
                return RC.digest(rt.checkpoints[-1].to_dict())
        return ""

    # ------------------------------------------------------------------
    # drain: classify + apply merges + set preempt flags
    # ------------------------------------------------------------------
    def drain(self, *, now: int, active_lane_id: Optional[str] = None) -> Tick:
        """Consume every effective event past the scheduler cursor.

        Per-source token buckets bound a storm: once a source exceeds its
        bucket in one drain, its further QUEUE-mode events are coalesced away
        rather than each minting scheduler work. INTERRUPT and MERGE events are
        never dropped by a bucket — dropping a safety alarm to protect the
        scheduler would be the cure killing the patient.
        """
        merged: List[str] = []
        interrupts: List[str] = []
        queued: List[str] = []

        raw = self.bus.effective_since(self._cursor)
        entries = coalesce(raw)
        self.telemetry.events_coalesced_away += len(raw) - len(entries)
        self._source_buckets = {}

        for entry in entries:
            env = entry.envelope
            self.telemetry.event_lag_samples.append(
                (env.event_id, max(0, now - env.occurred_at)))
            cls = classify_event(env, active_lane_id=active_lane_id)

            if cls.mode == DELIVER_MERGE:
                self._apply_merge(env, cls, now=now)
                merged.append(env.event_id)
                self.telemetry.events_merged += 1
            elif cls.mode == DELIVER_INTERRUPT:
                self._apply_interrupt(env, cls, now=now,
                                      active_lane_id=active_lane_id)
                interrupts.append(env.event_id)
                self.telemetry.events_interrupted += 1
            else:
                used = self._source_buckets.get(env.source, 0)
                if used >= self.scheduler.policy.source_bucket_capacity:
                    self.telemetry.events_dropped_by_bucket += 1
                else:
                    self._source_buckets[env.source] = used + 1
                    self._deferred.append(entry)
                    queued.append(env.event_id)
                    self.telemetry.events_queued += 1

        self._cursor = self.bus.head
        return Tick(served_lane_id=None, merged_event_ids=tuple(merged),
                    interrupt_event_ids=tuple(interrupts),
                    queued_event_ids=tuple(queued), idle_reason="drain_only")

    def _apply_merge(self, env: EventEnvelope, cls: Classification, *,
                     now: int) -> None:
        """The deterministic reducer. Never consumes an inference; the model
        sees only the resulting delta in its next pamphlet."""
        lane_id = cls.target_lane_id
        if lane_id is None or not self.registry.has(lane_id):
            return
        rt = self.registry.get(lane_id)
        delta: Dict[str, Any] = {
            "kind": env.event_type, "event_id": env.event_id,
            "summary": str(env.payload.get("summary", "")),
            "source_ref": env.payload_ref or f"event://{env.event_id}",
            "at": now,
        }

        if env.event_type == EV_QUESTION_ANSWERED:
            qid = str(env.payload.get("question_id", ""))
            # "Human answers late -> apply only if question is still pending"
            if rt.lane.pending_question_id != qid:
                delta["ignored"] = "question_not_pending"
                rt.merged_deltas += (delta,)
                return
            self.registry.update_lane(lane_id, pending_question_id=None)
            self._unblock(rt, now=now, reason="question_answered")
        elif env.event_type == EV_CHILD_COMPLETED:
            closed = str(env.payload.get("closes_obligation", ""))
            if closed and closed in rt.lane.open_obligations:
                remaining = tuple(o for o in rt.lane.open_obligations
                                  if o != closed)
                self.registry.update_lane(lane_id, open_obligations=remaining)
        elif env.event_type == EV_CHILD_FAILED:
            opened = str(env.payload.get("opens_obligation", ""))
            if opened:
                self.registry.update_lane(
                    lane_id, open_obligations=rt.lane.open_obligations + (opened,))
            if rt.status in SCHEDULABLE_STATUSES:
                self._block(rt, now=now, reason="child_failed")
        elif env.event_type == EV_BENCH_COMPLETED:
            if rt.status == LANE_BLOCKED:
                self._unblock(rt, now=now, reason="bench_completed")
        elif env.event_type == EV_MONITOR_SAMPLE:
            delta["freshness"] = env.payload.get("freshness", "current")

        rt.merged_deltas += (delta,)

    def _apply_interrupt(self, env: EventEnvelope, cls: Classification, *,
                         now: int, active_lane_id: Optional[str]) -> None:
        """Set ``preempt_requested`` — do NOT kill anything.

        "This does not mean killing generation mid-token. It means setting
        preempt_requested on the active lane and preventing its next inference
        or side effect from starting."
        """
        target = cls.target_lane_id or active_lane_id
        if target is not None and self.registry.has(target):
            rt = self.registry.get(target)
            if rt.preempt_requested_by is None:
                rt.preempt_requested_by = env.event_id
                self.telemetry.preemptions += 1
        elif active_lane_id is not None and self.registry.has(active_lane_id):
            rt = self.registry.get(active_lane_id)
            if rt.preempt_requested_by is None:
                rt.preempt_requested_by = env.event_id
                self.telemetry.preemptions += 1

        if env.event_type == EV_HUMAN_MESSAGE:
            self._route_human(env, now=now)
        elif env.event_type == EV_CONTROL_COMMAND:
            self._apply_control(env, now=now)
        elif env.event_type == EV_LEASE_LOST and cls.target_lane_id:
            self._on_lease_lost(cls.target_lane_id, now=now)

    def _route_human(self, env: EventEnvelope, *, now: int) -> None:
        route = str(env.payload.get("route", ROUTE_NEW_LANE))
        mode = str(env.payload.get("instruction_mode", "MUSING"))
        target = env.lane_id
        if route == ROUTE_NEW_LANE:
            lane_id = f"human:{env.event_id}"
            if not self.registry.has(lane_id):
                self.registry.register(
                    lane_id=lane_id,
                    contract_id=str(env.payload.get("contract_id", "human")),
                    parent_request_id=env.event_id,
                    # created_at is the MESSAGE time, not the drain time, so
                    # human latency measures the actual user wait time.
                    now=env.occurred_at,
                    priority_class=PRIORITY_P0_HUMAN_SAFETY,
                    kind="question", base_priority=100)
        elif route == ROUTE_CANCEL_LANE and target and self.registry.has(target):
            rt = self.registry.get(target)
            if lane_transition_allowed(rt.status, LANE_FAILED):
                self.registry.set_status(target, LANE_FAILED)
        elif route == ROUTE_SUPERSEDE_LANE and target and self.registry.has(target):
            rt = self.registry.get(target)
            if lane_transition_allowed(rt.status, LANE_SUPERSEDED):
                self.registry.set_status(target, LANE_SUPERSEDED)
        elif route == ROUTE_CONTINUE_LANE and target and self.registry.has(target):
            rt = self.registry.get(target)
            if mode == "DIRECTIVE_NOW":
                self.registry.update_lane(
                    target, priority_class=PRIORITY_P0_HUMAN_SAFETY)
        # ROUTE_ANSWER_QUESTION is correlated by question_id and handled by
        # the question_answered reducer — a human message never silently
        # mutates whichever lane happened to be active.

    def _apply_control(self, env: EventEnvelope, *, now: int) -> None:
        cmd = str(env.payload.get("command", "")).lower()
        target = env.lane_id
        if not target or not self.registry.has(target):
            return
        rt = self.registry.get(target)
        if cmd in ("stop", "cancel") and lane_transition_allowed(rt.status,
                                                                 LANE_FAILED):
            self.registry.set_status(target, LANE_FAILED)
        elif cmd == "redirect" and lane_transition_allowed(rt.status,
                                                           LANE_SUPERSEDED):
            self.registry.set_status(target, LANE_SUPERSEDED)

    def _on_lease_lost(self, lane_id: str, *, now: int) -> None:
        """Lease loss returns the lane to schedulable state with a bumped
        epoch — a stale worker's later result is then rejected by CAS."""
        rt = self.registry.get(lane_id)
        if rt.status == LANE_LEASED:
            self.registry.set_status(lane_id, LANE_CHECKPOINTED)
        self.registry.update_lane(lane_id, lease_epoch=rt.lane.lease_epoch + 1)
        rt.preempt_requested_by = None

    # ------------------------------------------------------------------
    # blocking helpers
    # ------------------------------------------------------------------
    def _block(self, rt: LaneRuntime, *, now: int, reason: str) -> None:
        """Blocking is a no-op on a lane that has never run: juggler_vocab has
        no REGISTERED -> BLOCKED edge, deliberately (a lane with no lease and
        no checkpoint has nothing to block ON). The opened obligation still
        lands, and the lane's next pamphlet carries it — the work is not lost,
        it just is not modelled as a block."""
        if lane_transition_allowed(rt.status, LANE_BLOCKED):
            self.registry.set_status(rt.lane_id, LANE_BLOCKED)

    def _unblock(self, rt: LaneRuntime, *, now: int, reason: str) -> None:
        if rt.status == LANE_BLOCKED:
            self.registry.set_status(rt.lane_id, LANE_UNBLOCKED)
        rt.unblock_bonus_until = now + \
            self.scheduler.policy.unblock_bonus_window_micros
        rt.next_eligible_at = min(rt.next_eligible_at, now)
        if rt.lane.priority_class == PRIORITY_P3_MAINTENANCE:
            return
        # unblock events are the P1 tier's whole purpose
        if rt.lane.priority_class == PRIORITY_P2_WORK:
            self.registry.update_lane(rt.lane_id,
                                      priority_class=PRIORITY_P1_UNBLOCK)

    # ------------------------------------------------------------------
    # obligations as schedulable work
    # ------------------------------------------------------------------
    def obligation_work_items(self, *, now: int) -> Tuple[Tuple[str, int], ...]:
        """(obligation_id, service_score) for every OPEN obligation, highest
        first. Obligations are schedulable work — the ledger owns the formula
        (``obligations.service_score``); the juggler never re-derives one."""
        rows = [(o.id, OB.service_score(o, now=now))
                for o in self.ledger.open_obligations()]
        rows.sort(key=lambda r: (-r[1], r[0]))
        return tuple(rows)

    def promote_obligation_to_lane(self, obligation_id: str, *, now: int,
                                   kind: str = "general") -> LaneRuntime:
        """A park's return obligation becomes a real lane. This is the
        ``promote_to_lane`` disposition of the park lifecycle, wired: the
        ledger records the disposition, the juggler creates the work unit."""
        ob = self.ledger.get(obligation_id)
        lane_id = f"lane:{obligation_id}"
        if self.registry.has(lane_id):
            return self.registry.get(lane_id)
        rt = self.registry.register(
            lane_id=lane_id, contract_id=ob.kind,
            parent_request_id=ob.origin_event_id, now=now,
            priority_class=ob.priority_class, kind=kind,
            base_priority={OB.P0: 100, OB.P1: 80, OB.P2: 50,
                           OB.P3: 20}[ob.priority_class],
            open_obligations=(obligation_id,))
        if ob.park_id:
            self.ledger.service_park(ob.park_id, disposition="promote_to_lane",
                                     now=now)
        return rt

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------
    def step(self, *, now: int) -> Tick:
        """One scheduling quantum: drain -> select -> lease -> infer ->
        validate -> commit -> release.

        The preemption protocol is visible in the ORDER of these lines and
        nowhere else. A lane that acquires the lease finishes its inference
        and gets a checkpoint even if an interrupt lands mid-inference; the
        interrupt is honored by refusing to START the next one.
        """
        self.telemetry.ticks += 1
        drained = self.drain(now=now, active_lane_id=None)

        lane_id = self.scheduler.choose(self.registry, now=now,
                                        held_resources=self._held_resources)
        if lane_id is None:
            self._clear_stale_preempts()
            return replace(drained, served_lane_id=None,
                           idle_reason="no_eligible_lane")

        rt = self.registry.get(lane_id)

        # --- measured latency, recorded at the moment of dispatch --------
        self.telemetry.dispatch_wait_samples.append(
            (lane_id, max(0, now - rt.last_progress_at)))
        if rt.kind == "question" and rt.last_scheduled_at is None:
            self.telemetry.human_latency_samples.append(
                (lane_id, max(0, now - rt.created_at)))

        # --- lease (CAS on lease epoch) ---------------------------------
        expected_epoch = rt.lane.lease_epoch
        self.registry.set_status(lane_id, LANE_LEASED)
        self.registry.update_lane(lane_id, lease_epoch=expected_epoch + 1)
        rt = self.registry.get(lane_id)
        lease_epoch = rt.lane.lease_epoch
        digest_before = rt.state_digest()

        for r in rt.resources:
            self._held_resources[r] = lane_id

        # --- rehydrate ---------------------------------------------------
        pamphlet = compile_pamphlet(rt, session_id=self.session_id,
                                    profile=self.pamphlet_profile)
        verdict = ZS.validate_zoom_slice(pamphlet)
        if not verdict.ok:
            # Fail FIRST, release second: LANE_LEASED -> LANE_QUARANTINED is a
            # legal edge and LANE_CHECKPOINTED -> LANE_QUARANTINED is not
            # (juggler_vocab.LANE_TRANSITIONS). Releasing before failing would
            # silently swallow every quarantine — the exact class of bug the
            # conservative transition map exists to surface.
            return self._fail_lane(rt, now=now,
                                   errors=(ERR_CONSTRAINT_CONFLICT,) +
                                   tuple(verdict.reasons),
                                   digest_before=digest_before,
                                   drained=drained)

        latest = self._latest_relevant_event(lane_id)
        request = SeatRequest(lane_id=lane_id, base_revision=lease_epoch,
                              pamphlet=pamphlet, latest_event=latest,
                              merged_deltas=rt.merged_deltas)

        # --- ONE bounded inference. Never interrupted mid-token. ---------
        self.telemetry.inferences += 1
        if self._last_served is not None and self._last_served != lane_id:
            self.telemetry.switches += 1
        step_result = self.seat(request)

        # --- validate -----------------------------------------------------
        errors = self.validate(step_result, rt, lease_epoch=lease_epoch)
        if errors:
            self.telemetry.invalid_model_outputs += 1
            # Fail FIRST, release second: LANE_LEASED -> LANE_QUARANTINED is a
            # legal edge and LANE_CHECKPOINTED -> LANE_QUARANTINED is not
            # (juggler_vocab.LANE_TRANSITIONS). Releasing before failing would
            # silently swallow every quarantine — the exact class of bug the
            # conservative transition map exists to surface.
            return self._fail_lane(rt, now=now, errors=errors,
                                   digest_before=digest_before,
                                   drained=drained,
                                   output_digest=step_result.output_digest())

        # --- WORK_INTENT (durable) BEFORE the mechanical action -----------
        intent = {
            "lane_id": lane_id, "step_id": step_result.step_id,
            "action_kind": step_result.action_kind,
            "idempotency_key": str(step_result.action.get("idempotency_key", "")),
            "base_revision": lease_epoch, "at": now,
        }
        self.fault_hook("after_intent", intent)
        rt.work_intents += (intent,)

        # --- mechanical action (only now, and only once) ------------------
        observed = self._execute(rt, step_result, now=now)
        self.fault_hook("after_action", {"lane_id": lane_id,
                                         "observed": observed})

        # --- checkpoint (atomic) ------------------------------------------
        checkpoint = self._commit_checkpoint(
            rt, step_result, observed=observed, now=now,
            digest_before=digest_before, lease_epoch=lease_epoch)

        self._release(rt, now=now)
        self.scheduler.note_served(rt)
        rt.consecutive_quanta = (rt.consecutive_quanta + 1
                                 if self._last_served == lane_id else 1)
        self._last_served = lane_id
        rt.last_scheduled_at = now
        # The preemption has now been EFFECTED: some other lane got the seat.
        # Clear the flags of the lanes that yielded so they are not frozen
        # forever (the flag stops exactly one inference, by design).
        self._clear_preempts_except(lane_id)
        if rt.turns_remaining is not None:
            rt.turns_remaining -= 1
        self.registry.update_lane(
            lane_id, service_debt=rt.lane.service_debt + 1.0)

        return replace(drained, served_lane_id=lane_id,
                       action_kind=step_result.action_kind,
                       checkpoint=checkpoint,
                       preempted_by_event_id=rt.preempt_requested_by,
                       idle_reason="")

    def run_until_idle(self, *, now: int, max_steps: int = 64,
                       tick_micros: int = MICROS_PER_SECOND
                       ) -> Tuple[Tick, ...]:
        """Bounded drive loop for tests and batch operation. The timer arm of
        the doc's "event-driven, but also has a bounded timer" is the caller's
        ``tick_micros`` — this module never sleeps."""
        ticks: List[Tick] = []
        t = now
        for _ in range(max_steps):
            tick = self.step(now=t)
            ticks.append(tick)
            t += tick_micros
            if tick.served_lane_id is None and not tick.merged_event_ids \
                    and not tick.interrupt_event_ids:
                break
        return tuple(ticks)

    # ------------------------------------------------------------------
    # validation / execution / commit
    # ------------------------------------------------------------------
    def validate(self, step_result: ModelStep, rt: LaneRuntime, *,
                 lease_epoch: int) -> Tuple[str, ...]:
        """The resume handshake validator. Machine-readable codes only — "it
        is much more effective than adding another page of explanatory prose
        to the pamphlet"."""
        errors: List[str] = []
        if step_result.lane_id != rt.lane_id:
            errors.append(ERR_WRONG_LANE)
        if step_result.base_revision != lease_epoch:
            errors.append(ERR_CHECKPOINT_MISMATCH)
        if not step_result.step_id:
            errors.append(ERR_MISSING_NEXT_ACTION)
        if rt.turns_remaining is not None and rt.turns_remaining <= 0:
            errors.append(ERR_BUDGET_EXHAUSTED)
        if step_result.action_kind == ACT_START_IO:
            key = str(step_result.action.get("idempotency_key", ""))
            if not key:
                errors.append(ERR_MISSING_NEXT_ACTION)
            elif key in self._idempotency_seen:
                errors.append(ERR_DUPLICATE_SIDE_EFFECT)
        if step_result.action_kind == ACT_COMPLETE:
            if not step_result.action.get("proof"):
                # "Work is complete only when acceptance criteria have
                # verification evidence." A completion claim with no proof is
                # rejected, never trusted.
                errors.append(ERR_CONSTRAINT_CONFLICT)
            if rt.lane.open_obligations:
                errors.append(ERR_WRONG_PHASE)
        for closed in step_result.obligations_closed:
            if closed not in rt.lane.open_obligations:
                errors.append(ERR_UNKNOWN_REQUEST)
        return tuple(errors)

    def _execute(self, rt: LaneRuntime, step_result: ModelStep, *,
                 now: int) -> Mapping[str, Any]:
        """Mechanical action layer. Idempotency keys are recorded BEFORE the
        call so a crash-then-retry is detected as a duplicate side effect."""
        kind = step_result.action_kind
        if kind == ACT_START_IO:
            key = str(step_result.action.get("idempotency_key", ""))
            self._idempotency_seen[key] = rt.lane_id
            observed = self.action_executor(rt.lane_id, step_result)
            self._block(rt, now=now, reason="io")
            return observed
        if kind == ACT_ASK_QUESTION:
            qid = str(step_result.action.get("question_id",
                                             f"q:{step_result.step_id}"))
            self.registry.update_lane(rt.lane_id, pending_question_id=qid)
            self._question_index[qid] = rt.lane_id
            self._block(rt, now=now, reason="question")
            return {"status": "asked", "question_id": qid}
        if kind == ACT_COMPLETE:
            return self.action_executor(rt.lane_id, step_result)
        return self.action_executor(rt.lane_id, step_result)

    def _commit_checkpoint(self, rt: LaneRuntime, step_result: ModelStep, *,
                           observed: Mapping[str, Any], now: int,
                           digest_before: str,
                           lease_epoch: int) -> RC.LaneCheckpoint:
        """ONE atomic swap: receipt + lane state + consumed event cursor.

        Everything is staged into locals first; the fault hook fires between
        staging and application. If it raises, ZERO mutations have landed —
        that is what makes "crash after action, before receipt" recoverable
        rather than a lane with a receipt its ledger never saw.
        """
        pr = step_result.procedure_receipt
        overrides: Dict[str, Any] = {
            "model_output_digest": step_result.output_digest(),
            "next_action": step_result.action_kind,
            "open_obligations": tuple(rt.lane.open_obligations),
            "preempted_by_event_id": rt.preempt_requested_by,
            "world_digest": RC.digest(dict(observed)),
            "state_digest_before": digest_before,
        }
        if pr is not None:
            staged = RC.checkpoint_from_procedure_receipt(
                lane_id=rt.lane_id, procedure_receipt=pr,
                predecessor_receipt=rt.lane.receipt_head,
                consumed_event_cursor=self._cursor,
                deferred_event_cursor=len(self._deferred), **overrides)
        else:
            staged = RC.LaneCheckpoint(
                lane_id=rt.lane_id,
                predecessor_receipt=rt.lane.receipt_head,
                consumed_event_cursor=self._cursor,
                deferred_event_cursor=len(self._deferred),
                status=str(observed.get("status", "ok")),
                rehydration_digest="", **overrides)

        obligations_after = tuple(
            o for o in rt.lane.open_obligations
            if o not in set(step_result.obligations_closed)
        ) + tuple(o for o in step_result.obligations_opened
                  if o not in rt.lane.open_obligations)

        self.fault_hook("before_receipt",
                        {"lane_id": rt.lane_id, "checkpoint": staged})

        # ---- APPLY (no observable partial state above this line) --------
        # Order matters and is not arbitrary: the lane's post-step state is
        # digested BEFORE the checkpoint is appended, otherwise
        # `state_digest_after` would have to contain a hash of itself. Then
        # the finalized checkpoint is hashed once, and THAT hash is the
        # receipt head — so `receipt_head == digest(stored_checkpoint)` always
        # holds, which is what makes the chain verifiable on resume.
        self.registry.update_lane(rt.lane_id, event_cursor=self._cursor,
                                  open_obligations=obligations_after)
        rt.merged_deltas = ()
        rt.last_progress_at = now
        staged = replace(staged, state_digest_after=rt.state_digest())
        head = RC.digest(staged.to_dict())
        rt.checkpoints += (staged,)
        self.registry.update_lane(rt.lane_id, receipt_head=head)
        self.telemetry.checkpoints += 1
        return staged

    def _release(self, rt: LaneRuntime, *, now: int) -> None:
        """Release lease + resources. Status lands on CHECKPOINTED unless the
        action already moved the lane (blocked/terminal)."""
        for r in rt.resources:
            if self._held_resources.get(r) == rt.lane_id:
                del self._held_resources[r]
        if rt.status == LANE_LEASED:
            self.registry.set_status(rt.lane_id, LANE_CHECKPOINTED)

    def _fail_lane(self, rt: LaneRuntime, *, now: int,
                   errors: Sequence[str], digest_before: str, drained: Tick,
                   output_digest: str = "") -> Tick:
        """Lane-local failure: error receipt, this lane's counter, this lane's
        backoff, this lane's quarantine. Siblings are untouched — asserted by
        TestLaneIsolation, not merely intended."""
        rt.failure_count += 1
        self.telemetry.lane_failures += 1
        rt.error_receipts += ({
            "lane_id": rt.lane_id, "errors": list(errors),
            "output_digest": output_digest,
            "state_digest_before": digest_before,
            "status": "failed", "next_action": "retry", "at": now,
        },)
        backoff = self.scheduler.policy.retry_backoff_base_micros * \
            (2 ** min(rt.failure_count - 1, 10))
        rt.next_eligible_at = now + backoff
        if rt.failure_count >= self.scheduler.policy.quarantine_after_failures:
            if lane_transition_allowed(rt.status, LANE_QUARANTINED):
                self.registry.set_status(rt.lane_id, LANE_QUARANTINED)
                rt.quarantine_reason = ",".join(errors)
                self.telemetry.quarantines += 1
        self._release(rt, now=now)
        return replace(drained, served_lane_id=rt.lane_id,
                       validation_errors=tuple(errors),
                       idle_reason="lane_failed")

    def clear_quarantine(self, lane_id: str, *, now: int,
                         cleared_by: str) -> LaneRuntime:
        """Quarantine is cleared by a human, never by the scheduler
        (juggler_vocab: QUARANTINED -> LEASED is the only outgoing edge, and a
        human clears it)."""
        if not cleared_by:
            raise JugglerError("clearing a quarantine requires an authority")
        rt = self.registry.get(lane_id)
        if rt.status != LANE_QUARANTINED:
            raise JugglerError(f"work stream {lane_id!r} is not quarantined")
        rt.quarantine_reason = ""
        rt.failure_count = 0
        rt.next_eligible_at = now
        # QUARANTINED's only edge is -> LEASED; go through it and land back on
        # CHECKPOINTED so the scheduler can pick the lane up normally.
        self.registry.set_status(lane_id, LANE_LEASED)
        self.registry.set_status(lane_id, LANE_CHECKPOINTED)
        return rt

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _latest_relevant_event(self, lane_id: str) -> Optional[EventEnvelope]:
        for entry in reversed(self.bus.all_entries()):
            if entry.duplicate or entry.out_of_order:
                continue
            if entry.envelope.lane_id == lane_id:
                return entry.envelope
        return None

    def _clear_stale_preempts(self) -> None:
        """A preempt flag is honored exactly once: it stops the NEXT inference
        on that lane. When nothing else is eligible, the preemption has been
        fully served (there is no competing work left), so the flag clears and
        the lane is not frozen forever."""
        for rt in self.registry.all():
            rt.preempt_requested_by = None

    def _clear_preempts_except(self, served_lane_id: str) -> None:
        for rt in self.registry.all():
            if rt.lane_id != served_lane_id:
                rt.preempt_requested_by = None

    def deferred_events(self) -> Tuple[EventEnvelope, ...]:
        return tuple(e.envelope for e in self._deferred)

    @property
    def cursor(self) -> int:
        return self._cursor
