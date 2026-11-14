"""Unit auto-merge (coalesce): dynamic producer+consumer merge after a failed unit gate.

``unit_planner_v2`` already coalesces STATIC cycles (Tarjan SCC) at plan time.  This module
is the DYNAMIC half: when earlier units have passed their gate and the current unit's gate
fails at a cross-unit surface (a seam violation naming a passed unit's definition, or a
shared gate test that regressed), the two units become ONE joint unit at the earliest
position that keeps providers first, with the union of files, members and gate tests.

Rules the mechanism keeps:
- budget pools leftover gate attempts with a minimum of two edited repair attempts;
- a merged unit that fails its gate again with no leftover ESCALATES (pointer holds, code-owned
  notice, record for the escalation packet); the loop stops after closing its native exchange;
- at most ``MAX_MERGE_MEMBERS`` symbols per merged unit and ``MAX_COALESCE_EVENTS`` merges per
  plan, so a plan never degenerates into "no split";
- every merge is a pure function of (plan, trigger, failures, surfaces): the new plan carries the
  previous digest in ``lineage`` and a resume that replays the gate records reproduces it.

Opt-in with ``STATEFRAME_UNIT_COALESCE=on`` (strict: anything else but ``off`` raises) and only
effective under a v2 plan.  When off this module is never imported by the engine.  No model call.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import chains
from unit_coalesce_contract import MODE_ENV, enabled, mode
from unit_planner import _bounded, _digest, render_plan_block
from unit_planner_v2 import GateVerdict, PlannedUnitV2, UnitPlanV2

MAX_MERGE_MEMBERS: Final = 4          # symbols per merged unit; above -> escalate
MAX_COALESCE_EVENTS: Final = 2        # merges per plan (lab cap)
DEFAULT_GATE_ATTEMPTS: Final = 2      # per-unit gate allowance when the unit carries none
_GEN: Final = re.compile(r":g(\d+)$")
_EXPOSES: Final = re.compile(r"^class (\w+) exposes ")


@dataclass(frozen=True)
class CoalesceTrigger:
    kind: Literal["integration", "unmapped"]
    failing_unit: str
    partner_unit: str | None
    surface: str                      # "path:Name" from a seam row or "test:name"


@dataclass(frozen=True)
class CoalesceOutcome:
    """unit_index is the earliest merged position that keeps providers first."""
    plan: UnitPlanV2
    unit_index: int
    merged_unit_id: str | None
    escalate: bool
    reason: str


# ---------------------------------------------------------------- helpers
def allowance(unit: PlannedUnitV2) -> int:
    return int(getattr(unit, "gate_attempts", 0) or 0) or DEFAULT_GATE_ATTEMPTS


def _leftover(unit: PlannedUnitV2, failures: Mapping[str, int]) -> int:
    return max(0, allowance(unit) - int(failures.get(unit.unit_id, 0)))


def _unit_at(plan: UnitPlanV2, unit_id: str) -> int:
    for i, unit in enumerate(plan.units):
        if unit.unit_id == unit_id:
            return i
    raise KeyError(f"unknown unit {unit_id!r}")


def _defining_paths(manifest: Mapping[str, object], name: str, kind: str | None = None) -> tuple[str, ...]:
    hits = sorted({d["file"] for d in manifest.get("definitions", ())
                   if d.get("name") == name and (kind is None or d.get("kind") == kind)})
    return tuple(hits)


def _pick(candidates: Sequence[PlannedUnitV2], unit: PlannedUnitV2, match) -> PlannedUnitV2 | None:
    """First matching passed unit, those the failing unit depends on first."""
    for pool in (tuple(c for c in candidates if c.unit_id in unit.depends_on), tuple(candidates)):
        for cand in pool:
            if match(cand):
                return cand
    return None


# ---------------------------------------------------------------- locate / reseam
def locate_partner(plan: UnitPlanV2, unit: PlannedUnitV2, verdict: GateVerdict,
                   manifest: Mapping[str, object], *, skipped: list[str] | None = None) -> CoalesceTrigger:
    """Map the failing surface to an earlier PASSED unit, in a fixed order of evidence.

    (1) a seam violation row resolved through the manifest to its defining path -> the unit whose
    files and members own it; (2) a failing gate test shared with a passed unit (regression: the producer broke).
    A test's symbol refs are already folded into v2 ``gate_tests`` (a test belongs to every unit it
    touches), so the ref-overlap rule is rule (2) by construction.  Nothing found -> "unmapped" with
    the immediate predecessor as the labelled fallback partner.
    """
    idx = _unit_at(plan, unit.unit_id)
    passed = plan.units[:idx]
    for violation in verdict.seam_violations:
        name = str(violation.get("name", ""))
        exposes = _EXPOSES.match(str(violation.get("detail", "")))
        cls = exposes.group(1) if exposes else None
        paths = _defining_paths(manifest, cls, "class") if cls else _defining_paths(manifest, name)
        if len(paths) != 1:
            if len(paths) > 1 and skipped is not None:
                skipped.append(f"ambiguous_definition name={cls or name} paths={','.join(paths)}")
            continue
        path = paths[0]
        owner = _pick(passed, unit, lambda c, path=path, cls=cls, name=name: path in c.files and (
            (cls in c.members or f"{cls}.{name}" in c.members) if cls else
            name in c.members))
        if owner is not None:
            surface = f"{path}:{cls}.{name}" if cls else f"{path}:{name}"
            return CoalesceTrigger("integration", unit.unit_id, owner.unit_id, surface)
    for test in verdict.failing_tests:
        owner = _pick(passed, unit, lambda c, test=test: test in c.gate_tests)
        if owner is not None:
            return CoalesceTrigger("integration", unit.unit_id, owner.unit_id, f"test:{test}")
    if verdict.failing_tests:
        surface = f"test:{verdict.failing_tests[0]}"
    elif verdict.seam_violations:
        row = verdict.seam_violations[0]
        surface = f"{row.get('file', '?')}:{row.get('name', '?')}"
    else:
        surface = "none"
    predecessor = passed[-1].unit_id if passed else None
    return CoalesceTrigger("unmapped", unit.unit_id, predecessor, surface)


def reseam(files: Mapping[str, str], unit_files: Sequence[str], partner_files: Sequence[str]) -> tuple[str, ...]:
    """Cross-file seam surfaces ("path:qualname") between the two file sets on the current tree."""
    left, right = set(unit_files), set(partner_files)
    if not left or not right or not files:
        return ()
    out: set[str] = set()
    for seam in chains.cross_file_seams(chains.seam_manifest(dict(files))):
        definition = seam["definition"]
        home, refs = definition["file"], set(seam["ref_files"])
        if (home in left and refs & right) or (home in right and refs & left):
            out.add(f"{home}:{definition['qualname']}")
    return tuple(sorted(out))


# ---------------------------------------------------------------- order invariants
def _stable_topological(slots: Sequence[tuple[int, PlannedUnitV2]]) -> list[PlannedUnitV2]:
    """Kahn's order over depends_on; among ready units the lowest original slot goes first."""
    ids = {u.unit_id for _, u in slots}
    unknown = sorted({d for _, u in slots for d in u.depends_on} - ids)
    if unknown:
        raise ValueError(f"coalesce dangling dependencies: {','.join(unknown)}")
    pending = list(slots)
    out: list[PlannedUnitV2] = []
    done: set[str] = set()
    while pending:
        ready = [(i, u) for i, u in pending if all(d in done for d in u.depends_on)]
        if not ready:
            raise ValueError("coalesce produced a dependency cycle among units: " +
                             ",".join(u.unit_id for _, u in pending))
        slot, unit = min(ready, key=lambda item: item[0])
        out.append(unit)
        done.add(unit.unit_id)
        pending.remove((slot, unit))
    return out


def assert_providers_first(units: Sequence[PlannedUnitV2]) -> None:
    """Invariant: every depends_on names an earlier unit of the plan (raise on violation)."""
    seen: set[str] = set()
    for unit in units:
        for dep in unit.depends_on:
            if dep not in seen:
                raise ValueError(f"{unit.unit_id} depends on {dep!r} which is not an earlier unit")
        seen.add(unit.unit_id)


def merge_closure(plan: UnitPlanV2, seed: set[str]) -> set[str]:
    """``seed`` plus every unit that both (transitively) consumes a seed unit and feeds one."""
    consumers: dict[str, set[str]] = {u.unit_id: set() for u in plan.units}
    providers: dict[str, set[str]] = {u.unit_id: set(u.depends_on) for u in plan.units}
    for unit in plan.units:
        for dep in unit.depends_on:
            consumers.setdefault(dep, set()).add(unit.unit_id)

    def reach(start: set[str], step: Mapping[str, set[str]]) -> set[str]:
        seen, queue = set(start), list(start)
        while queue:
            for nxt in step.get(queue.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    return reach(seed, consumers) & reach(seed, providers)


# ---------------------------------------------------------------- pure merge
def _refuse(plan: UnitPlanV2, index: int, reason: str) -> CoalesceOutcome:
    return CoalesceOutcome(plan, index, None, True, reason)


def coalesce(plan: UnitPlanV2, trigger: CoalesceTrigger | None, failures: Mapping[str, int] | None = None,
             surfaces: Sequence[str] = ()) -> CoalesceOutcome:
    """Merge at the earliest position that keeps providers first; pure and deterministic.

    ``failures`` is unit_id -> edited repair failures; leftovers pool with a two-attempt floor.
    Returns ``escalate=True`` with the plan unchanged when a cap or the second-failure rule applies.
    """
    if trigger is None:
        raise ValueError("coalesce needs a trigger with a partner unit")
    failures = dict(failures or {})
    ids = {u.unit_id for u in plan.units}
    unknown = sorted({d for u in plan.units for d in u.depends_on} - ids)
    if unknown:
        raise ValueError(f"coalesce dangling dependencies: {','.join(unknown)}")
    f_idx = _unit_at(plan, trigger.failing_unit)
    failing = plan.units[f_idx]
    left_f = _leftover(failing, failures)
    if failing.coalesced_from and left_f <= 0:
        return _refuse(plan, f_idx, f"{failing.unit_id} already coalesced and out of gate attempts")
    if trigger.partner_unit is None:
        raise ValueError("coalesce needs a trigger with a partner unit")
    p_idx = _unit_at(plan, trigger.partner_unit)
    if f_idx == p_idx:
        raise ValueError("coalesce needs two distinct units")
    if trigger.kind == "unmapped" and not surfaces:
        return CoalesceOutcome(plan, f_idx, None, False, "no_reseam_evidence")
    partner = plan.units[p_idx]
    if plan.coalesce_events >= MAX_COALESCE_EVENTS:
        return _refuse(plan, f_idx, f"coalesce events cap {MAX_COALESCE_EVENTS} reached")
    # Units strictly between the pair that both consume the partner and feed the failing unit would
    # form a unit-level cycle with the merged unit; they join the merge (SCC closure, like v2's static
    # coalesce) so the merged unit never sits before one of its providers.
    merge_set = merge_closure(plan, {partner.unit_id, failing.unit_id})
    group = [u for u in plan.units if u.unit_id in merge_set]        # plan order
    members: tuple[str, ...] = ()
    for unit in group:
        members += tuple(m for m in unit.members if m not in members)
    if len(members) > MAX_MERGE_MEMBERS:
        return _refuse(plan, f_idx, f"merge would hold {len(members)} members (max {MAX_MERGE_MEMBERS})")

    pair = merge_set
    gens = [int(m.group(1)) for m in (_GEN.search(u.unit_id) for u in group) if m]
    symbol = "+".join(members)
    new_id = f"unit:{symbol}:g{(max(gens) if gens else 0) + 1}"
    files = tuple(sorted({f for u in group for f in u.files}))
    wanted = {t for u in group for t in u.gate_tests}
    gate_tests: list[str] = []
    for unit in plan.units:                                    # union in plan order, de-duplicated
        gate_tests.extend(t for t in unit.gate_tests if t in wanted and t not in gate_tests)
    outer = {d for u in group for d in u.depends_on} - pair
    depends_on = tuple(u.unit_id for u in plan.units if u.unit_id in outer)
    gate = ", ".join(plan.public_gate_paths or gate_tests)
    seams = f"; seams: {', '.join(surfaces)}" if surfaces else ""
    subcontract = (f"Implement {', '.join(members)} together in {', '.join(files)} (coalesced after "
                   f"{failing.symbol} failed its gate at {trigger.surface}{seams}); gate = [{gate}].")
    budget = max(DEFAULT_GATE_ATTEMPTS, sum(_leftover(u, failures) for u in group))
    merged = PlannedUnitV2(new_id, symbol, "joint", tuple(gate_tests), depends_on, subcontract, "",
                           files, members, tuple(u.unit_id for u in group), budget)

    # Rewrite depends_on (old ids -> merged id), then re-order by a stable topological sort with the
    # earliest original group position as tiebreak: a merged unit
    # never sits before one of its providers nor after its first consumer.
    rewritten: list[tuple[int, PlannedUnitV2]] = []
    first = min(i for i, u in enumerate(plan.units) if u.unit_id in pair)
    for i, unit in enumerate(plan.units):
        if i == first:
            rewritten.append((i, merged))
        elif unit.unit_id in pair:
            continue
        else:
            deps: list[str] = []
            for dep in unit.depends_on:
                dep = new_id if dep in pair else dep
                if dep not in deps and dep != unit.unit_id:
                    deps.append(dep)
            rewritten.append((i, PlannedUnitV2(**{**unit.__dict__, "depends_on": tuple(deps)})))
    ordered = _stable_topological(rewritten)
    position = next(i for i, u in enumerate(ordered) if u.unit_id == new_id)
    units: list[PlannedUnitV2] = []
    for i, unit in enumerate(ordered):
        nxt = ordered[i + 1].symbol if i + 1 < len(ordered) else None   # handoff re-rendered for new neighbours
        handoff = (f"Handoff: do not start {nxt} until {unit.symbol}'s gate tests and seam check pass."
                   if nxt else f"Handoff: {unit.symbol} is the final unit; its gate must pass for plan completion.")
        units.append(PlannedUnitV2(**{**unit.__dict__, "handoff_rule": handoff}))
    assert_providers_first(units)
    draft = UnitPlanV2(**{**plan.__dict__, "units": tuple(units), "plan_digest": "",
                          "lineage": plan.lineage + (plan.plan_digest,),
                          "coalesce_events": plan.coalesce_events + 1})
    new_plan = UnitPlanV2(**{**draft.__dict__, "plan_digest": _digest(draft._basis())})
    return CoalesceOutcome(new_plan, position, new_id, False,
                           f"{trigger.kind} {partner.unit_id}+{failing.unit_id} at {trigger.surface}")


# ---------------------------------------------------------------- hook
def _runtime_of(bolt) -> object | None:
    return getattr(bolt, "measurement", None)


def _say(bolt, line: str) -> str | None:
    organ_out = getattr(bolt, "organ_out", None)
    if callable(organ_out):
        return organ_out("unit_plan", line)
    return None


def _queue_notice(bolt, text: str) -> None:
    approved = _say(bolt, text)
    if approved:
        pending = getattr(bolt, "unit_plan_refresh", None)
        bolt.unit_plan_refresh = pending + "\n" + approved if pending else approved
    else:
        bolt._rec("unit_plan", "unit_plan_refresh_skipped", "unit_plan organ suppressed notice")


def _escalate(bolt, unit_id: str, reason: str) -> None:
    bolt.unit_escalate = unit_id
    bolt._rec("unit_plan", "unit_coalesce_escalate", _bounded(f"{unit_id} {reason}", 240, "…"))
    _queue_notice(bolt, f"[planner] {unit_id} failed its gate after coalesce; stop retrying this unit and "
                       f"report the failing surface ({reason})")
    runtime = _runtime_of(bolt)
    if runtime is not None:
        try:
            runtime._unit_coalesce_escalated = unit_id
        except AttributeError as error:
            bolt._rec("unit_plan", "unit_coalesce_error", f"escalation runtime: {error}")


def tree_digest(plan: UnitPlanV2, files: Mapping[str, str]) -> str:
    """Include missing paths and path names as well as contents in the repair identity."""
    return _digest([(p, files.get(p)) for p in plan.index_paths])


def observe_tree(bolt, plan: UnitPlanV2, files: Mapping[str, str]) -> None:
    """Seed all units before the first edit; merged units are seeded at merge time."""
    if not hasattr(bolt, "unit_gate_snapshots"):
        digest = tree_digest(plan, files)
        bolt.unit_gate_snapshots = {u.unit_id: digest for u in plan.units}


def advance_passed(bolt, plan: UnitPlanV2, pointer: int) -> None:
    """Retain completion of untouched units even when a merge moves them past the pointer."""
    passed = set(getattr(bolt, "unit_gate_passed", ()))
    passed.add(plan.units[pointer].unit_id)
    bolt.unit_gate_passed = passed
    rechecks = set(getattr(bolt, "unit_gate_rechecks", ())) - {plan.units[pointer].unit_id}
    bolt.unit_gate_rechecks = rechecks
    nxt = min([pointer + 1] + [i for i, u in enumerate(plan.units) if u.unit_id in rechecks])
    while (nxt < len(plan.units) and plan.units[nxt].unit_id in passed
           and plan.units[nxt].unit_id not in rechecks):
        nxt += 1
    bolt.unit_index = nxt


def on_gate_failure(bolt, plan: UnitPlanV2, unit: PlannedUnitV2, verdict: GateVerdict,
                    files: Mapping[str, str] | None, root) -> CoalesceOutcome | None:
    """The gate_and_advance fail-branch hook: count, then merge or escalate; None = plain retry."""
    files = dict(files or {})
    digest = tree_digest(plan, files)
    observe_tree(bolt, plan, files)
    snapshots = bolt.unit_gate_snapshots
    if snapshots.get(unit.unit_id) == digest:
        return None  # pre-edit or repeated tests on the last counted repair
    snapshots[unit.unit_id] = digest
    failures = getattr(bolt, "unit_gate_failures", None)
    if not isinstance(failures, dict):
        failures = bolt.unit_gate_failures = {}
    failures[unit.unit_id] = failures.get(unit.unit_id, 0) + 1
    if failures[unit.unit_id] < allowance(unit):
        return None
    if getattr(bolt, "unit_escalate", None) == unit.unit_id:
        return None                                            # already escalated: no third path
    if unit.coalesced_from:
        # Exhaustion is a retry stop, independent of whether another merge
        # partner exists or a local failure happens to share a cross-file seam.
        outcome = coalesce(plan, CoalesceTrigger("unmapped", unit.unit_id, None, "none"), failures)
        _escalate(bolt, unit.unit_id, outcome.reason)
        return outcome
    skipped: list[str] = []
    trigger = locate_partner(plan, unit, verdict, chains.seam_manifest(files) if files else {}, skipped=skipped)
    for reason in skipped:
        bolt._rec("unit_plan", "unit_coalesce_skip", _bounded(f"{unit.unit_id} {reason}", 240, "…"))
    if trigger.partner_unit is None:
        reason = f"no earlier unit to merge with at {trigger.surface}"
        _escalate(bolt, unit.unit_id, reason)
        return CoalesceOutcome(plan, _unit_at(plan, unit.unit_id), None, True, reason)
    partner = plan.units[_unit_at(plan, trigger.partner_unit)]
    surfaces = reseam(files, unit.files, partner.files)
    if trigger.kind == "unmapped" and not surfaces:
        bolt._rec("unit_plan", "unit_coalesce_skip", _bounded(
            f"{unit.unit_id} partner={partner.unit_id} no_reseam_evidence surface={trigger.surface}", 240, "…"))
        return None
    outcome = coalesce(plan, trigger, failures=failures, surfaces=surfaces)
    if outcome.escalate:
        _escalate(bolt, unit.unit_id, outcome.reason)
        return outcome
    # Resolve/render/deliver all fallible advice before committing the new plan.
    # An advisory failure must not leave a new plan paired with the old pointer.
    passed = set(getattr(bolt, "unit_gate_passed", set()))
    from unit_planner_v2 import invalidate_dependents
    merged = outcome.plan.units[outcome.unit_index]
    replaced = set(merged.coalesced_from)
    invalid = invalidate_dependents(plan, replaced)
    rechecks = set(getattr(bolt, "unit_gate_rechecks", ())) | (passed & invalid)
    if rechecks & replaced:
        rechecks.add(merged.unit_id)
    current_ids = {u.unit_id for u in outcome.plan.units}
    rechecks.intersection_update(current_ids)
    passed.difference_update(invalid)
    passed.intersection_update(current_ids)
    passed.difference_update(rechecks)
    merged_digest = tree_digest(outcome.plan, files)
    notice = _bounded((f"[planner] supersedes digest {plan.plan_digest}; current digest {outcome.plan.plan_digest}; "
              f"coalesced {partner.symbol}+{unit.symbol} into one unit ({outcome.reason}); "
              f"id={merged.unit_id}; gate=[{', '.join(merged.gate_tests)}]; files={', '.join(merged.files)}"), 1200, "…")
    _queue_notice(bolt, notice + "\n" + render_plan_block(outcome.plan))
    bolt.unit_gate_passed = passed
    bolt.unit_gate_rechecks = rechecks
    bolt.unit_gate_pass_sources = {key: value for key, value in getattr(bolt, "unit_gate_pass_sources", {}).items()
                                   if key in passed}
    bolt.unit_plan, bolt.unit_index = outcome.plan, outcome.unit_index
    snapshots[outcome.merged_unit_id] = merged_digest
    bolt.unit_plan_refresh_from = plan.plan_digest
    fallback = " fallback=predecessor" if trigger.kind == "unmapped" else ""
    bolt._rec("unit_plan", "unit_coalesce", _bounded(
        f"{plan.plan_digest}->{outcome.plan.plan_digest} {partner.unit_id}+{unit.unit_id} "
        f"kind={trigger.kind} surface={trigger.surface} seams={len(surfaces)}{fallback}", 240, "…"))
    return outcome


__all__ = [
    "DEFAULT_GATE_ATTEMPTS",
    "MAX_COALESCE_EVENTS",
    "MAX_MERGE_MEMBERS",
    "MODE_ENV",
    "CoalesceOutcome",
    "CoalesceTrigger",
    "allowance",
    "assert_providers_first",
    "coalesce",
    "enabled",
    "locate_partner",
    "merge_closure",
    "mode",
    "on_gate_failure",
    "reseam",
]
