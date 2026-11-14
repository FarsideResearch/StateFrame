"""Pure working-set allocator with floor-first feasibility (allocator review r1 finding 2, amendment A3).

The allocator decides, for every pre-serialized slot of a request, which rendering mode
goes on the wire. It is a pure function over immutable inputs: no runtime, no clock, no
environment, no engine imports. The same inputs always give the same answer.

Algorithm (Astra G1):
1. Floor first. ``M = fixed + sum(min-cost mode per slot)``. If ``M > cap`` the request is
   infeasible before any search, and ``Blocked(reason="FLOOR_INFEASIBLE")`` carries the
   floor broken down by priority tier and by slot kind.
2. Greedy domain restriction in an explicit priority order (fresh, protected, carried
   previous modes until the first mismatch, then optional slots by byte-LRU). Each slot
   takes the richest mode whose cost, together with the min-cost completion of every
   undecided slot, keeps ``M <= cap``. Because the floor was feasible, the min-cost mode
   always qualifies, so the greedy pass never dead-ends. Fresh slots are FIFO (G1, A1 "in
   batch order"): the oldest admitted page is decided first. A CARRIED slot is never
   promoted above its previous mode, before or after the break (plan: capacity increases do
   not auto-promote; a new read or recovery creates the promotion obligation).
3. The result is the lexicographic optimum under that order, which an exhaustive reference
   solver in the test suite confirms for small instances.

The bytes cap is not a token proof: the caller tokenizes the final templated request.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# Modes


@dataclass(frozen=True, order=False)
class Mode:
    """One fidelity level. Higher ``rank`` is richer. Only exposure modes count as source exposure."""

    name: str
    rank: int
    is_exposure: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Mode.name must be non-empty")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise ValueError(f"Mode.rank must be an int, got {self.rank!r}")

    def richer_than(self, other: "Mode") -> bool:
        return self.rank > other.rank


FULL = Mode("FULL", 3, is_exposure=True)
DEGRADED = Mode("DEGRADED", 2)
DEFERRED = Mode("DEFERRED", 1)

DEFAULT_MODES: tuple[Mode, ...] = (FULL, DEGRADED, DEFERRED)


# ---------------------------------------------------------------------------
# Slots


class PriorityTier(IntEnum):
    """Allocation priority. Lower value is decided first."""

    FRESH = 0        # due reads from the previous request; product before tests/context, FIFO
    PROTECTED = 1    # edit surface, contracts, pins: must stay, decided before everything carried
    CARRIED = 2      # slots that were on the previous wire; keep their mode until the first mismatch
    OPTIONAL = 3     # everything else, byte-LRU


# Fresh slots order by kind: product sources before tests before context before anything else.
KIND_ORDER: Mapping[str, int] = {"product": 0, "tests": 1, "context": 2}
_UNRANKED_KIND = len(KIND_ORDER)


@dataclass(frozen=True)
class Slot:
    """One allocatable wire element with its pre-serialized rendering domain.

    ``domain`` lists ``(mode, exact_cost_bytes)`` pairs. The costs are exact additive wire
    bytes supplied by the candidate graph; the allocator never renders anything.
    ``previous_mode`` is the mode this slot had on the previous wire (required for CARRIED).
    For CARRIED slots only the entries no richer than ``previous_mode`` are allocatable
    (``allowed``); a richer entry is ignored by the floor and the greedy pass alike.
    """

    key: str
    kind: str
    priority_tier: PriorityTier
    admission_ordinal: int
    last_use_ordinal: int
    domain: tuple[tuple[Mode, int], ...]
    previous_mode: Mode | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Slot.key must be non-empty")
        if not self.domain:
            raise ValueError(f"slot {self.key!r}: domain must be non-empty")
        ranks: set[int] = set()
        for entry in self.domain:
            if len(entry) != 2 or not isinstance(entry[0], Mode):
                raise ValueError(f"slot {self.key!r}: domain entries are (Mode, cost_bytes)")
            mode, cost = entry
            if not isinstance(cost, int) or isinstance(cost, bool) or cost < 0:
                raise ValueError(f"slot {self.key!r}: cost for {mode.name} must be a non-negative int")
            if mode.rank in ranks:
                raise ValueError(f"slot {self.key!r}: duplicate mode rank {mode.rank}")
            ranks.add(mode.rank)
        try:
            tier = PriorityTier(self.priority_tier)
        except ValueError as exc:
            raise ValueError(f"slot {self.key!r}: unknown priority_tier {self.priority_tier!r}") from exc
        object.__setattr__(self, "priority_tier", tier)  # store the enum, not a bare int
        if tier is PriorityTier.CARRIED:
            if self.previous_mode is None:
                raise ValueError(f"slot {self.key!r}: CARRIED slots need previous_mode")
            if not self.allowed:
                raise ValueError(f"slot {self.key!r}: CARRIED domain has no mode at or below "
                                 f"previous_mode {self.previous_mode.name}; a richer-only slot is PROTECTED")

    # -- domain views -------------------------------------------------------
    @property
    def allowed(self) -> tuple[tuple[Mode, int], ...]:
        """The allocatable domain: everything for FRESH/PROTECTED/OPTIONAL, ``<= previous_mode`` for CARRIED."""
        if self.priority_tier is PriorityTier.CARRIED and self.previous_mode is not None:
            ceiling = self.previous_mode.rank
            return tuple(entry for entry in self.domain if entry[0].rank <= ceiling)
        return self.domain

    def cost_of(self, mode: Mode) -> int | None:
        for candidate, cost in self.allowed:
            if candidate == mode:
                return cost
        return None

    @property
    def min_cost(self) -> int:
        return min(cost for _, cost in self.allowed)

    @property
    def richest_first(self) -> tuple[tuple[Mode, int], ...]:
        return tuple(sorted(self.allowed, key=lambda entry: entry[0].rank, reverse=True))

    @property
    def richest_cost(self) -> int:
        return self.richest_first[0][1]


# ---------------------------------------------------------------------------
# Results


@dataclass(frozen=True)
class FloorBreakdown:
    """Where the floor bytes come from, for the BLOCKED report."""

    fixed_bytes: int
    by_tier: Mapping[str, int]
    by_kind: Mapping[str, int]

    @property
    def total(self) -> int:
        return self.fixed_bytes + sum(self.by_tier.values())


@dataclass(frozen=True)
class Blocked:
    """Typed infeasibility. ``floor_bytes > cap`` before any search happened."""

    reason: str
    floor_bytes: int
    cap: int
    breakdown: FloorBreakdown

    @property
    def infeasible(self) -> bool:
        return True


@dataclass(frozen=True)
class Assignment:
    """Chosen mode per slot plus the exact byte accounting behind it."""

    modes: Mapping[str, Mode]
    total_bytes: int
    floor_bytes: int
    fixed_bytes: int
    cap: int
    order: tuple[str, ...] = field(default=())          # decision order, for diagnostics
    break_key: str | None = None                        # first carried slot that lost its mode

    @property
    def infeasible(self) -> bool:
        return False

    def exposure_bytes(self, slots: Iterable[Slot]) -> int:
        """Bytes of slots assigned an exposure mode (FULL in the default ladder)."""
        total = 0
        for slot in slots:
            mode = self.modes[slot.key]
            if mode.is_exposure:
                cost = slot.cost_of(mode)
                total += cost if cost is not None else 0
        return total


# ---------------------------------------------------------------------------
# Ordering


def _kind_rank(kind: str) -> tuple[int, str]:
    return (KIND_ORDER.get(kind, _UNRANKED_KIND), kind)


def fresh_order_key(slot: Slot) -> tuple[object, ...]:
    """Product before tests/context, then FIFO (oldest admission first, as the read tool admitted
    the batch), then the stable key (path+range)."""
    return (_kind_rank(slot.kind), slot.admission_ordinal, slot.key)


def protected_order_key(slot: Slot) -> tuple[object, ...]:
    return (slot.key,)


def carried_order_key(slot: Slot) -> tuple[object, ...]:
    """Previous wire order: admission first, then key. This is the prefix the cache keeps."""
    return (slot.admission_ordinal, slot.key)


def optional_order_key(slot: Slot) -> tuple[object, ...]:
    """Byte-LRU: most recently used first, cheaper richest allocatable form first, then the stable key."""
    return (-slot.last_use_ordinal, slot.richest_cost, slot.key)


def priority_groups(slots: Sequence[Slot]) -> tuple[list[Slot], list[Slot], list[Slot], list[Slot]]:
    """Slots split by tier, each list already in its decision order."""
    fresh = sorted((s for s in slots if s.priority_tier is PriorityTier.FRESH), key=fresh_order_key)
    protected = sorted((s for s in slots if s.priority_tier is PriorityTier.PROTECTED), key=protected_order_key)
    carried = sorted((s for s in slots if s.priority_tier is PriorityTier.CARRIED), key=carried_order_key)
    optional = sorted((s for s in slots if s.priority_tier is PriorityTier.OPTIONAL), key=optional_order_key)
    return fresh, protected, carried, optional


# ---------------------------------------------------------------------------
# Floor


def floor_breakdown(slots: Sequence[Slot], fixed_bytes: int) -> FloorBreakdown:
    by_tier: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    for slot in slots:
        by_tier[slot.priority_tier.name] = by_tier.get(slot.priority_tier.name, 0) + slot.min_cost
        by_kind[slot.kind] = by_kind.get(slot.kind, 0) + slot.min_cost
    return FloorBreakdown(fixed_bytes=fixed_bytes,
                          by_tier=dict(sorted(by_tier.items())),
                          by_kind=dict(sorted(by_kind.items())))


def _validate(slots: Sequence[Slot], fixed_bytes: int, cap: int) -> None:
    if not isinstance(fixed_bytes, int) or isinstance(fixed_bytes, bool) or fixed_bytes < 0:
        raise ValueError("fixed_bytes must be a non-negative int")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("cap must be a non-negative int")
    seen: set[str] = set()
    for slot in slots:
        if not isinstance(slot, Slot):
            raise ValueError(f"expected Slot, got {type(slot).__name__}")
        if slot.key in seen:
            raise ValueError(f"duplicate slot key {slot.key!r}")
        seen.add(slot.key)


# ---------------------------------------------------------------------------
# Solver


def solve(slots: Sequence[Slot], fixed_bytes: int, cap: int) -> Assignment | Blocked:
    """Floor-first, then greedy domain restriction in priority order. Pure and deterministic.

    Guarantees (checked against the exhaustive reference in the tests):
    - total_bytes <= cap for every Assignment;
    - if every fresh slot fits at its richest mode alongside the floor of the rest, every
      fresh slot gets its richest mode;
    - the Assignment is the lexicographic optimum under the priority order;
    - FLOOR_INFEASIBLE is returned without any search when fixed + sum(min_cost) > cap.
    """
    slots = tuple(slots)
    _validate(slots, fixed_bytes, cap)

    breakdown = floor_breakdown(slots, fixed_bytes)
    floor = breakdown.total
    if floor > cap:
        return Blocked(reason="FLOOR_INFEASIBLE", floor_bytes=floor, cap=cap, breakdown=breakdown)

    fresh, protected, carried, optional = priority_groups(slots)

    chosen: dict[str, Mode] = {}
    order: list[str] = []
    # Running M: committed bytes + min-cost of everything still undecided.
    committed = fixed_bytes
    remaining_floor = floor - fixed_bytes

    def decide(slot: Slot, mode: Mode, cost: int) -> None:
        nonlocal committed, remaining_floor
        chosen[slot.key] = mode
        order.append(slot.key)
        committed += cost
        remaining_floor -= slot.min_cost

    def richest_feasible(slot: Slot) -> None:
        for mode, cost in slot.richest_first:
            if committed + cost + (remaining_floor - slot.min_cost) <= cap:
                decide(slot, mode, cost)
                return
        raise AssertionError(f"slot {slot.key!r}: min-cost completion lost feasibility")  # unreachable after the floor check

    for slot in fresh:
        richest_feasible(slot)
    for slot in protected:
        richest_feasible(slot)

    # Carried slots keep their previous mode until the first one that cannot. From that slot on
    # the wire prefix changes anyway, so the rest compete with the optional slots by byte-LRU,
    # still capped at their previous mode (Slot.allowed): no auto-promotion.
    break_key: str | None = None
    tail: list[Slot] = list(optional)
    for index, slot in enumerate(carried):
        previous = slot.previous_mode
        cost = slot.cost_of(previous) if previous is not None else None
        if cost is not None and committed + cost + (remaining_floor - slot.min_cost) <= cap:
            decide(slot, previous, cost)  # type: ignore[arg-type]  # previous is not None when cost is
            continue
        break_key = slot.key
        tail.extend(carried[index:])
        break

    for slot in sorted(tail, key=optional_order_key):
        richest_feasible(slot)

    assert remaining_floor == 0 and len(chosen) == len(slots)
    return Assignment(modes=dict(chosen), total_bytes=committed, floor_bytes=floor,
                      fixed_bytes=fixed_bytes, cap=cap, order=tuple(order), break_key=break_key)


# ---------------------------------------------------------------------------
# Read admission arithmetic (A1b)


ADMIT_FULL = "FULL"
ADMIT_TRUNCATE = "TRUNCATE"
ADMIT_REFUSED = "REFUSED_ADMISSION"


@dataclass(frozen=True)
class AdmissionDecision:
    """Per-read admission: FULL (all bytes), TRUNCATE (first ``admitted_bytes``), or REFUSED_ADMISSION."""

    kind: str
    admitted_bytes: int

    @property
    def admitted(self) -> bool:
        return self.kind != ADMIT_REFUSED


def admit_read_bytes(allowance: int, page_bytes_list: Sequence[int]) -> tuple[AdmissionDecision, ...]:
    """Spend the fresh-exposure allowance over a batch's pages in batch order.

    A page that fits is FULL. The first page that overflows takes whatever is left as a
    shorter page (TRUNCATE, n) so a giant single read is paginated, not blocked (K1). Once
    the allowance is spent, every later page is REFUSED_ADMISSION and nothing is dropped:
    the caller resumes those with cursors (K2).

    ``admitted_bytes`` of a TRUNCATE is an upper bound: the delivered page must end on a line
    boundary so the existing next_cursor stays valid. Use ``admit_read_lines`` when the
    per-line sizes are known; it refuses instead of delivering an empty page.
    """
    if not isinstance(allowance, int) or isinstance(allowance, bool) or allowance < 0:
        raise ValueError("allowance must be a non-negative int; a negative allowance is FLOOR_INFEASIBLE")
    remaining = allowance
    decisions: list[AdmissionDecision] = []
    for page_bytes in page_bytes_list:
        if not isinstance(page_bytes, int) or isinstance(page_bytes, bool) or page_bytes < 0:
            raise ValueError("page sizes must be non-negative ints")
        if page_bytes <= remaining:
            decisions.append(AdmissionDecision(ADMIT_FULL, page_bytes))
            remaining -= page_bytes
        elif remaining > 0:
            decisions.append(AdmissionDecision(ADMIT_TRUNCATE, remaining))
            remaining = 0
        else:
            decisions.append(AdmissionDecision(ADMIT_REFUSED, 0))
    return tuple(decisions)


@dataclass(frozen=True)
class LineAdmissionDecision:
    """Line-granular admission: ``admitted_lines`` whole lines totalling ``admitted_bytes``."""

    kind: str
    admitted_bytes: int
    admitted_lines: int

    @property
    def admitted(self) -> bool:
        return self.kind != ADMIT_REFUSED


def admit_read_lines(allowance: int, pages_line_bytes: Sequence[Sequence[int]]) -> tuple[LineAdmissionDecision, ...]:
    """``admit_read_bytes`` on whole lines, so a shorter page always ends on a valid cursor.

    Each page is its per-line byte lengths in order. A page that fits is FULL. The first page
    that overflows keeps the longest prefix of whole lines that fits; if not even its first
    line fits it is REFUSED_ADMISSION (the cursor stays at the page start) rather than an
    empty TRUNCATE. Every page after the overflow is REFUSED_ADMISSION.
    """
    if not isinstance(allowance, int) or isinstance(allowance, bool) or allowance < 0:
        raise ValueError("allowance must be a non-negative int; a negative allowance is FLOOR_INFEASIBLE")
    remaining = allowance
    decisions: list[LineAdmissionDecision] = []
    for line_bytes in pages_line_bytes:
        lines = tuple(line_bytes)
        for size in lines:
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError("line sizes must be non-negative ints")
        page_bytes = sum(lines)
        if page_bytes <= remaining:  # includes an empty page after the overflow: FULL with zero lines
            decisions.append(LineAdmissionDecision(ADMIT_FULL, page_bytes, len(lines)))
            remaining -= page_bytes
            continue
        # First overflow: longest prefix of whole lines that fits. The batch is spent from here on,
        # so a line-cut leftover is not handed to later pages (they are REFUSED, as in admit_read_bytes).
        taken_bytes = taken_lines = 0
        for size in lines:
            if taken_bytes + size > remaining:
                break
            taken_bytes += size
            taken_lines += 1
        remaining = 0
        if taken_lines > 0:
            decisions.append(LineAdmissionDecision(ADMIT_TRUNCATE, taken_bytes, taken_lines))
        else:
            decisions.append(LineAdmissionDecision(ADMIT_REFUSED, 0, 0))
    return tuple(decisions)


def fresh_exposure_allowance(cap: int, fixed_bytes: int, protected_slots: Iterable[Slot]) -> int:
    """A1b allowance: ``cap - fixed - min-cost(protected)``, floored at zero.

    Spec-exact (A1b). The next request's floor also carries the min-cost of carried and
    optional slots (deferred notices are not free), so the caller folds those into
    ``fixed_bytes`` or FLOOR_INFEASIBLE can still fire after a successful admission.
    """
    floor = fixed_bytes + sum(slot.min_cost for slot in protected_slots)
    return max(0, cap - floor)


__all__ = [
    "ADMIT_FULL", "ADMIT_REFUSED", "ADMIT_TRUNCATE", "AdmissionDecision", "Assignment", "Blocked",
    "DEFAULT_MODES", "DEFERRED", "DEGRADED", "FULL", "FloorBreakdown", "KIND_ORDER", "LineAdmissionDecision",
    "Mode", "PriorityTier", "Slot", "admit_read_bytes", "admit_read_lines", "carried_order_key", "floor_breakdown",
    "fresh_exposure_allowance", "fresh_order_key", "optional_order_key", "priority_groups",
    "protected_order_key", "solve",
]
