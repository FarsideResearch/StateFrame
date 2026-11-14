"""Independent exhaustive reference for the working-set allocator priority order.

This module deliberately does NOT import kernel_working_set_allocator. It works on plain
dicts and enumerates every assignment, so a disagreement with the greedy allocator is a
real finding, not a shared bug.

Slot dict shape:
    {"key": str, "kind": str, "tier": "FRESH"|"PROTECTED"|"CARRIED"|"OPTIONAL",
     "admission": int, "last_use": int, "previous": mode_name | None,
     "previous_rank": int | None, "domain": [(mode_name, rank, cost_bytes), ...]}

A CARRIED slot may only take modes with rank <= previous_rank (no auto-promotion); the
feasible floor and the enumeration both use that restricted domain.

Objective (lexicographic maximum, decided in this order):
  1. fresh slots, ordered (kind rank product<tests<context<other by name, FIFO = oldest admission
     first, key): the chosen rank;
  2. protected slots, ordered by key: the chosen rank;
  3. carried slots in previous wire order (admission, key): 1 if the previous mode is kept,
     else 0, and the sequence stops right after the first 0;
  4. the tail (optional slots plus carried slots from the first mismatch onward), ordered by
     byte-LRU (most recent last_use first, cheapest richest allocatable form first, key): the
     chosen rank.
"""
from __future__ import annotations

from itertools import product

KIND_RANK = {"product": 0, "tests": 1, "context": 2}


def _kind_rank(kind: str) -> tuple[int, str]:
    return (KIND_RANK.get(kind, 3), kind)


def _richest_cost(slot: dict) -> int:
    """Cost of the richest ALLOCATABLE form (a carried slot's ceiling is its previous mode)."""
    return max(_choices(slot), key=lambda entry: entry[1])[2]


def _lru_key(slot: dict) -> tuple:
    return (-slot["last_use"], _richest_cost(slot), slot["key"])


def _rank_of(slot: dict, mode_name: str) -> int:
    for name, rank, _ in slot["domain"]:
        if name == mode_name:
            return rank
    raise KeyError(mode_name)


def _cost_of(slot: dict, mode_name: str) -> int:
    for name, _, cost in slot["domain"]:
        if name == mode_name:
            return cost
    raise KeyError(mode_name)


def _choices(slot: dict) -> list[tuple[str, int, int]]:
    if slot["tier"] == "CARRIED":
        return [entry for entry in slot["domain"] if entry[1] <= slot["previous_rank"]]
    return list(slot["domain"])


def floor_bytes(slots: list[dict], fixed: int) -> int:
    return fixed + sum(min(cost for _, _, cost in _choices(slot)) for slot in slots)


def score(slots: list[dict], assignment: dict[str, str]) -> tuple:
    """Score vector of an assignment; larger is better (lexicographic)."""
    fresh = sorted((s for s in slots if s["tier"] == "FRESH"),
                   key=lambda s: (_kind_rank(s["kind"]), s["admission"], s["key"]))
    protected = sorted((s for s in slots if s["tier"] == "PROTECTED"), key=lambda s: s["key"])
    carried = sorted((s for s in slots if s["tier"] == "CARRIED"), key=lambda s: (s["admission"], s["key"]))
    optional = [s for s in slots if s["tier"] == "OPTIONAL"]

    vector: list[int] = [_rank_of(s, assignment[s["key"]]) for s in fresh]
    vector += [_rank_of(s, assignment[s["key"]]) for s in protected]

    tail = list(optional)
    for index, slot in enumerate(carried):
        if assignment[slot["key"]] == slot["previous"]:
            vector.append(1)
        else:
            vector.append(0)
            tail.extend(carried[index:])
            break
    vector += [_rank_of(s, assignment[s["key"]]) for s in sorted(tail, key=_lru_key)]
    return tuple(vector)


def solve_exhaustive(slots: list[dict], fixed: int, cap: int) -> dict | None:
    """Best feasible assignment as {"modes": {key: mode_name}, "bytes": int}, or None when infeasible."""
    keys = [slot["key"] for slot in slots]
    choices = [[name for name, _, _ in _choices(slot)] for slot in slots]
    best: tuple | None = None
    best_assignment: dict[str, str] | None = None
    best_bytes = 0
    for combo in product(*choices):
        assignment = dict(zip(keys, combo))
        total = fixed + sum(_cost_of(slot, assignment[slot["key"]]) for slot in slots)
        if total > cap:
            continue
        vector = score(slots, assignment)
        if best is None or vector > best:
            best, best_assignment, best_bytes = vector, assignment, total
        elif vector == best:
            raise AssertionError(f"objective is not a total order: tie between {best_assignment} and {assignment}")
    if best_assignment is None:
        return None
    return {"modes": best_assignment, "bytes": best_bytes}
