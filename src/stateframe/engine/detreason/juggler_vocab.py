"""Shared scheduling vocabulary for logical work units and physical slots.

A slot is a transient execution resource. A Lane is durable work state; ModeRun
records a particular execution. State transitions and priority tiers are explicit
so scheduling and replay apply the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Tuple

MODULE = "detreason.juggler_vocab"
VERSION = "1"


# ======================================================================
# 1. SLOT — physical llama-server sequence slot
# Physical slot fields: SlotPool, id_slot, cache_prompt.
# ======================================================================

#: A slot is FREE (unowned) or BUSY (currently generating for some lane).
#: The source doc's SlotPool tracks this as membership in a `_busy` set, not
#: a named enum — we name it for callers who need to reason about it.
SLOT_FREE = "free"
SLOT_BUSY = "busy"
SLOT_STATES: Tuple[str, ...] = (SLOT_FREE, SLOT_BUSY)

#: Slots never transition through any state but these two, and only ever
#: directly between them (acquire: free->busy, release: busy->free). There is
#: no "starting" or "draining" slot state in the source doc.
SLOT_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    SLOT_FREE: frozenset({SLOT_BUSY}),
    SLOT_BUSY: frozenset({SLOT_FREE}),
}


@dataclass(frozen=True)
class Slot:
    """A single physical llama-server sequence slot.

    Field shape per ``a_juggler_slots.md``'s ``SlotPool``: slots are addressed
    by integer id (``id_slot`` on the wire), track the ``cache_family`` of the
    prompt prefix currently warm in them (for sticky affinity — see
    ``SlotPool.acquire``'s sort key), and a logical clock of last use for
    LRU eviction when no family match is free.
    """

    slot_id: int
    state: str = SLOT_FREE
    cache_family: Optional[str] = None
    last_used: int = 0

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("slot_id must be >= 0")
        if self.state not in SLOT_STATES:
            raise ValueError(f"unknown slot state {self.state!r}")


# ======================================================================
# 2. LANE — durable logical work unit (``LaneState``)
# Logical work-state vocabulary.
# ======================================================================

#: Lifecycle events/statuses named explicitly in "Genuinely missing wiring"
#: item 5 of a_juggler_interrupts.md: "Add lane registration, lease,
#: checkpoint, blocked, unblocked, superseded, failed, quarantined, and done
#: events to the existing ledger/event vocabulary."
LANE_REGISTERED = "registered"
LANE_LEASED = "leased"
LANE_CHECKPOINTED = "checkpointed"
LANE_BLOCKED = "blocked"
LANE_UNBLOCKED = "unblocked"
LANE_SUPERSEDED = "superseded"
LANE_FAILED = "failed"
LANE_QUARANTINED = "quarantined"
LANE_DONE = "done"

LANE_STATUSES: Tuple[str, ...] = (
    LANE_REGISTERED, LANE_LEASED, LANE_CHECKPOINTED, LANE_BLOCKED,
    LANE_UNBLOCKED, LANE_SUPERSEDED, LANE_FAILED, LANE_QUARANTINED, LANE_DONE,
)

#: Terminal statuses: a lane in one of these never re-enters ``LANE_STATUSES``
#: again as an active worker. Quarantine is deliberately NOT terminal here —
#: a human can clear it and re-lease the same lane (see LANE_TRANSITIONS
#: below); only FAILED/SUPERSEDED/DONE have zero outgoing edges.
LANE_TERMINAL: FrozenSet[str] = frozenset({LANE_SUPERSEDED, LANE_FAILED,
                                            LANE_DONE})

#: Deliberately conservative transition map — only the edges the source doc's
#: prose actually describes. Anything not listed here is a drift signal, not
#: an oversight: extend this map (and cite the new edge to a doc) rather than
#: guessing.
LANE_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    LANE_REGISTERED: frozenset({LANE_LEASED, LANE_SUPERSEDED, LANE_FAILED}),
    LANE_LEASED: frozenset({LANE_CHECKPOINTED, LANE_BLOCKED, LANE_FAILED,
                             LANE_QUARANTINED}),
    LANE_CHECKPOINTED: frozenset({LANE_LEASED, LANE_BLOCKED, LANE_DONE,
                                   LANE_SUPERSEDED, LANE_FAILED}),
    LANE_BLOCKED: frozenset({LANE_UNBLOCKED, LANE_SUPERSEDED,
                              LANE_QUARANTINED, LANE_FAILED}),
    LANE_UNBLOCKED: frozenset({LANE_LEASED, LANE_CHECKPOINTED}),
    LANE_SUPERSEDED: frozenset(),
    LANE_FAILED: frozenset(),
    LANE_QUARANTINED: frozenset({LANE_LEASED}),  # human clears the quarantine
    LANE_DONE: frozenset(),
}


def lane_transition_allowed(current: str, target: str) -> bool:
    """Pure check against LANE_TRANSITIONS. Never mutates, never raises for an
    unknown pair — callers decide what an illegal transition means to them."""
    return target in LANE_TRANSITIONS.get(current, frozenset())


@dataclass(frozen=True)
class Lane:
    """A durable logical work unit — ``LaneState`` in the source doc, renamed
    ``Lane`` here per the W0-2 resolution. Field shape is a direct transcription
    of ``a_juggler_interrupts.md``'s ``LaneState {}`` block; nothing invented.

    Note the fields this type does NOT have: no ``slot_id`` (a lane does not
    own a physical slot — it is granted one transiently by the scheduler) and
    no ``mode`` / ``output_contract`` (that belongs to ``ModeRun`` below, which
    references this lane by id).
    """

    lane_id: str
    contract_id: str
    parent_request_id: str
    status: str = LANE_REGISTERED
    snapshot_digest: str = ""
    receipt_head: Optional[str] = None
    ledger_head: Optional[str] = None
    event_cursor: int = 0
    open_obligations: Tuple[str, ...] = ()
    pending_question_id: Optional[str] = None
    budget: Optional[dict] = None
    priority_class: str = "P2"
    last_started_at: Optional[str] = None
    service_debt: float = 0.0
    lease_epoch: int = 0

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("lane_id is required")
        if self.status not in LANE_STATUSES:
            raise ValueError(f"unknown work stream status {self.status!r}")
        if self.priority_class not in PRIORITY_TIERS:
            raise ValueError(f"unknown priority_class {self.priority_class!r}")

    @property
    def terminal(self) -> bool:
        return self.status in LANE_TERMINAL

    def with_status(self, target: str) -> "Lane":
        """Fix-forward status change — returns a NEW Lane, never mutates.
        Raises if the transition is not in LANE_TRANSITIONS (fail closed, per
        the same discipline `procedure.py` uses for attempts)."""
        if not lane_transition_allowed(self.status, target):
            raise ValueError(
                f"illegal lane transition {self.status!r} -> {target!r}")
        from dataclasses import replace
        return replace(self, status=target)


#: Priority tiers from a_juggler_interrupts.md's "Scheduler and starvation
#: guards": hard tiers first, fair scheduling within each tier.
PRIORITY_P0_HUMAN_SAFETY = "P0"
PRIORITY_P1_UNBLOCK = "P1"
PRIORITY_P2_WORK = "P2"
PRIORITY_P3_MAINTENANCE = "P3"
PRIORITY_TIERS: Tuple[str, ...] = (
    PRIORITY_P0_HUMAN_SAFETY, PRIORITY_P1_UNBLOCK, PRIORITY_P2_WORK,
    PRIORITY_P3_MAINTENANCE,
)


# ======================================================================
# 3. MODE_RUN — mode-execution unit (mode_selector.md §7's "Lane")
#    Source: /tmp/solq/w58/a_mode_selector.md §7 (renamed per W0-2)
# ======================================================================

#: mode_selector.md §7.1 "Lane states" verbatim, renamed to MODE_RUN_* here.
MODE_RUN_RUNNABLE = "RUNNABLE"
MODE_RUN_WAITING = "WAITING"
MODE_RUN_BLOCKED = "BLOCKED"
MODE_RUN_EXECUTING = "EXECUTING"
MODE_RUN_WITNESSING = "WITNESSING"
MODE_RUN_ACCEPTED = "ACCEPTED"
MODE_RUN_PARTIAL = "PARTIAL"
MODE_RUN_ABSTAINED = "ABSTAINED"
MODE_RUN_FAILED = "FAILED"

MODE_RUN_STATUSES: Tuple[str, ...] = (
    MODE_RUN_RUNNABLE, MODE_RUN_WAITING, MODE_RUN_BLOCKED, MODE_RUN_EXECUTING,
    MODE_RUN_WITNESSING, MODE_RUN_ACCEPTED, MODE_RUN_PARTIAL,
    MODE_RUN_ABSTAINED, MODE_RUN_FAILED,
)

MODE_RUN_TERMINAL: FrozenSet[str] = frozenset({
    MODE_RUN_ACCEPTED, MODE_RUN_PARTIAL, MODE_RUN_ABSTAINED, MODE_RUN_FAILED,
})

#: Transition edges implied by §7's narrative: a run becomes RUNNABLE once
#: dependencies clear, waits/blocks while they don't, executes, then goes
#: through witnessing before landing on a terminal outcome. Not exhaustive of
#: every doc's phrasing — conservative and citeable, same policy as LANE_*.
MODE_RUN_TRANSITIONS: Dict[str, FrozenSet[str]] = {
    MODE_RUN_WAITING: frozenset({MODE_RUN_RUNNABLE, MODE_RUN_BLOCKED}),
    MODE_RUN_BLOCKED: frozenset({MODE_RUN_WAITING, MODE_RUN_RUNNABLE,
                                  MODE_RUN_ABSTAINED, MODE_RUN_FAILED}),
    MODE_RUN_RUNNABLE: frozenset({MODE_RUN_EXECUTING, MODE_RUN_BLOCKED}),
    MODE_RUN_EXECUTING: frozenset({MODE_RUN_WITNESSING, MODE_RUN_FAILED,
                                    MODE_RUN_BLOCKED}),
    MODE_RUN_WITNESSING: frozenset({MODE_RUN_ACCEPTED, MODE_RUN_PARTIAL,
                                     MODE_RUN_ABSTAINED, MODE_RUN_FAILED}),
    MODE_RUN_ACCEPTED: frozenset(),
    MODE_RUN_PARTIAL: frozenset(),
    MODE_RUN_ABSTAINED: frozenset(),
    MODE_RUN_FAILED: frozenset(),
}


def mode_run_transition_allowed(current: str, target: str) -> bool:
    return target in MODE_RUN_TRANSITIONS.get(current, frozenset())


@dataclass(frozen=True)
class ModeRun:
    """A mode-execution unit — mode_selector.md §7's "Lane", renamed
    ``ModeRun`` here per W0-2 so it stops colliding with the durable work-unit
    ``Lane`` above. Field shape is a direct transcription of that doc's
    ``Lane {}`` block, PLUS ``lane_id`` (renamed from the doc's own overloaded
    ``lane_id`` field, which in that doc pointed at itself — here it points at
    the durable ``Lane`` this run executes within, resolving the W0-2 clash
    structurally: N mode runs can reference the same lane over its lifetime).
    """

    mode_run_id: str
    lane_id: str                      # FK -> Lane.lane_id, never self-referential
    objective: str
    mode_or_stack: Tuple[str, ...] = ()
    priority: str = PRIORITY_P2_WORK
    dependencies: Tuple[str, ...] = ()      # other mode_run_ids
    required_inputs: Tuple[str, ...] = ()
    output_contract: Optional[dict] = None
    budget: Optional[dict] = None
    authority: Tuple[str, ...] = ()
    status: str = MODE_RUN_WAITING
    receipt_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.mode_run_id:
            raise ValueError("mode_run_id is required")
        if not self.lane_id:
            raise ValueError("lane_id is required (a ModeRun always executes "
                              "within a durable Lane)")
        if self.status not in MODE_RUN_STATUSES:
            raise ValueError(f"unknown mode_run status {self.status!r}")

    @property
    def terminal(self) -> bool:
        return self.status in MODE_RUN_TERMINAL

    def with_status(self, target: str) -> "ModeRun":
        if not mode_run_transition_allowed(self.status, target):
            raise ValueError(
                f"illegal mode_run transition {self.status!r} -> {target!r}")
        from dataclasses import replace
        return replace(self, status=target)


# ======================================================================
# Admission: N logical lanes over M physical slots (never 1:1)
# ======================================================================

@dataclass(frozen=True)
class AdmissionRequest:
    """A lane asking the scheduler for a physical slot. This is the object
    that makes the N-lanes-vs-M-slots relationship explicit and queueable,
    per B15's landmine note: "N logical lanes > 4 physical slots — needs an
    admission queue (W0-2)." Deliberately minimal — full admission-control
    scheduling policy is B15's job, not this vocabulary module's."""

    lane_id: str
    priority_class: str = PRIORITY_P2_WORK
    cache_family: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.lane_id:
            raise ValueError("lane_id is required")
        if self.priority_class not in PRIORITY_TIERS:
            raise ValueError(f"unknown priority_class {self.priority_class!r}")
