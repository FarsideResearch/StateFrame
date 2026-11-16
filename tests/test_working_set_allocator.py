"""kernel_working_set_allocator: floor-first feasibility, greedy lexicographic allocation, A1b admission."""
from __future__ import annotations

import random

import pytest

import kernel_working_set_allocator as wsa
from working_set_allocator_reference import floor_bytes as ref_floor, solve_exhaustive

FULL, D, O = wsa.FULL, wsa.DEGRADED, wsa.DEFERRED
T = wsa.PriorityTier

# A synthetic 4-mode ladder: the allocator must not assume three modes.
SYN_MODES = (wsa.Mode("RICH", 40, is_exposure=True), wsa.Mode("MEDIUM", 30), wsa.Mode("THIN", 20),
             wsa.Mode("STUB", 10))

EXHAUSTIVE_INSTANCES = 1500


def slot(key, tier, domain, *, kind="product", admission=0, last_use=0, previous=None):
    return wsa.Slot(key=key, kind=kind, priority_tier=tier, admission_ordinal=admission,
                    last_use_ordinal=last_use, domain=tuple(domain), previous_mode=previous)


# ---------------------------------------------------------------------------
# Floor first


def test_floor_infeasible_is_immediate_with_breakdown():
    slots = [slot("a", T.FRESH, [(FULL, 100), (D, 30)], kind="product"),
             slot("b", T.PROTECTED, [(FULL, 50)], kind="tests"),
             slot("c", T.OPTIONAL, [(FULL, 80), (O, 5)], kind="context")]
    result = wsa.solve(slots, fixed_bytes=20, cap=100)
    assert isinstance(result, wsa.Blocked)
    assert result.infeasible
    assert result.reason == "FLOOR_INFEASIBLE"
    assert result.floor_bytes == 20 + 30 + 50 + 5 == 105
    assert result.cap == 100
    assert result.breakdown.fixed_bytes == 20
    assert result.breakdown.by_tier == {"FRESH": 30, "OPTIONAL": 5, "PROTECTED": 50}
    assert result.breakdown.by_kind == {"context": 5, "product": 30, "tests": 50}
    assert result.breakdown.total == 105


def test_floor_exactly_at_cap_is_feasible_at_min_cost():
    slots = [slot("a", T.FRESH, [(FULL, 100), (D, 30)]), slot("b", T.OPTIONAL, [(FULL, 80), (O, 5)])]
    result = wsa.solve(slots, fixed_bytes=20, cap=55)
    assert isinstance(result, wsa.Assignment)
    assert result.modes == {"a": D, "b": O}
    assert result.total_bytes == 55 == result.floor_bytes


def test_floor_infeasible_does_not_depend_on_rich_modes(monkeypatch):
    """No search happens: the richest_first view is never consulted when the floor is over the cap."""
    calls = []
    original = wsa.Slot.richest_first.fget

    def spy(self):
        calls.append(self.key)
        return original(self)

    monkeypatch.setattr(wsa.Slot, "richest_first", property(spy))
    slots = [slot("a", T.FRESH, [(FULL, 100), (D, 60)]), slot("b", T.OPTIONAL, [(FULL, 80), (O, 50)])]
    assert isinstance(wsa.solve(slots, fixed_bytes=0, cap=100), wsa.Blocked)
    assert calls == []


# ---------------------------------------------------------------------------
# Priority order


def test_all_fresh_full_when_they_fit():
    slots = [slot(f"f{i}", T.FRESH, [(FULL, 10), (D, 2)], admission=i) for i in range(5)]
    slots += [slot("opt", T.OPTIONAL, [(FULL, 100), (O, 1)])]
    result = wsa.solve(slots, fixed_bytes=0, cap=51)
    assert isinstance(result, wsa.Assignment)
    assert all(result.modes[f"f{i}"] is FULL for i in range(5))
    assert result.modes["opt"] is O
    assert result.total_bytes == 51 <= result.cap


def test_fresh_order_product_then_tests_then_context_then_fifo():
    slots = [slot("ctx", T.FRESH, [(FULL, 10), (D, 1)], kind="context", admission=9),
             slot("test_old", T.FRESH, [(FULL, 10), (D, 1)], kind="tests", admission=1),
             slot("test_new", T.FRESH, [(FULL, 10), (D, 1)], kind="tests", admission=2),
             slot("prod", T.FRESH, [(FULL, 10), (D, 1)], kind="product", admission=0)]
    result = wsa.solve(slots, fixed_bytes=0, cap=22)  # room for two FULL and two D
    assert isinstance(result, wsa.Assignment)
    assert result.order == ("prod", "test_old", "test_new", "ctx")  # kind first, then FIFO
    assert result.modes == {"prod": FULL, "test_old": FULL, "test_new": D, "ctx": D}


def test_fresh_beats_protected_beats_carried_beats_optional():
    slots = [slot("opt", T.OPTIONAL, [(FULL, 10), (O, 1)], last_use=99),
             slot("car", T.CARRIED, [(FULL, 10), (D, 1)], previous=FULL),
             slot("pro", T.PROTECTED, [(FULL, 10), (D, 1)]),
             slot("fre", T.FRESH, [(FULL, 10), (D, 1)])]
    for cap, expected in [(13, {"fre": FULL, "pro": D, "car": D, "opt": O}),
                          (22, {"fre": FULL, "pro": FULL, "car": D, "opt": O}),
                          (31, {"fre": FULL, "pro": FULL, "car": FULL, "opt": O}),
                          (40, {"fre": FULL, "pro": FULL, "car": FULL, "opt": FULL})]:
        result = wsa.solve(slots, fixed_bytes=0, cap=cap)
        assert isinstance(result, wsa.Assignment)
        assert result.modes == expected, cap


def test_carried_keeps_previous_mode_even_when_richer_fits():
    """Keeping the previous rendering preserves the cache prefix; it is not auto-promoted."""
    slots = [slot("car", T.CARRIED, [(FULL, 10), (D, 4)], previous=D)]
    result = wsa.solve(slots, fixed_bytes=0, cap=100)
    assert isinstance(result, wsa.Assignment)
    assert result.modes == {"car": D}
    assert result.break_key is None


def test_carried_breaks_at_first_mismatch_then_tail_is_lru():
    slots = [slot("c1", T.CARRIED, [(FULL, 10), (D, 1)], previous=FULL, admission=1, last_use=1),
             slot("c2", T.CARRIED, [(FULL, 20), (D, 5)], previous=FULL, admission=2, last_use=1),
             slot("c3", T.CARRIED, [(FULL, 10), (D, 1)], previous=FULL, admission=3, last_use=50),
             slot("opt", T.OPTIONAL, [(FULL, 10), (O, 1)], last_use=40)]
    # c1 keeps FULL (10). c2 cannot keep FULL (10+20 + floors 1+1 = 32 > 26) -> break at c2.
    # Tail by LRU: c3 (last_use 50) gets FULL (10+10+5+1 = 26), opt (40) only fits O, c2 (1) only D.
    result = wsa.solve(slots, fixed_bytes=0, cap=26)
    assert isinstance(result, wsa.Assignment)
    assert result.break_key == "c2"
    assert result.modes == {"c1": FULL, "c2": D, "c3": FULL, "opt": O}
    assert result.order == ("c1", "c3", "opt", "c2")
    assert result.total_bytes == 10 + 10 + 1 + 5 == 26


def test_carried_previous_mode_no_longer_in_domain_is_a_mismatch():
    """Monotone retirement: the domain shrank below the previous mode, so the slot breaks the prefix."""
    slots = [slot("car", T.CARRIED, [(D, 4), (O, 1)], previous=FULL)]
    result = wsa.solve(slots, fixed_bytes=0, cap=100)
    assert isinstance(result, wsa.Assignment)
    assert result.break_key == "car"
    assert result.modes == {"car": D}


def test_optional_byte_lru_prefers_recent_then_cheaper_then_key():
    slots = [slot("old_big", T.OPTIONAL, [(FULL, 30), (O, 1)], last_use=1),
             slot("new_big", T.OPTIONAL, [(FULL, 30), (O, 1)], last_use=5),
             slot("new_small", T.OPTIONAL, [(FULL, 10), (O, 1)], last_use=5)]
    result = wsa.solve(slots, fixed_bytes=0, cap=41)
    assert isinstance(result, wsa.Assignment)
    assert result.order == ("new_small", "new_big", "old_big")
    assert result.modes == {"new_small": FULL, "new_big": FULL, "old_big": O}


# ---------------------------------------------------------------------------
# Synthetic 4-mode domain


def test_four_mode_domain_steps_down_one_rung_at_a_time():
    rich, medium, thin, stub = SYN_MODES
    domain = [(rich, 100), (medium, 60), (thin, 30), (stub, 5)]
    for cap, expected in [(100, rich), (99, medium), (60, medium), (59, thin), (30, thin), (29, stub), (5, stub)]:
        result = wsa.solve([slot("s", T.FRESH, domain)], fixed_bytes=0, cap=cap)
        assert isinstance(result, wsa.Assignment)
        assert result.modes["s"] is expected, cap
    assert isinstance(wsa.solve([slot("s", T.FRESH, domain)], fixed_bytes=0, cap=4), wsa.Blocked)


def test_four_mode_exposure_bytes_counts_only_exposure_modes():
    rich, medium, thin, stub = SYN_MODES
    slots = [slot("a", T.FRESH, [(rich, 40), (stub, 1)]), slot("b", T.OPTIONAL, [(rich, 40), (medium, 20), (stub, 1)])]
    result = wsa.solve(slots, fixed_bytes=0, cap=60)
    assert isinstance(result, wsa.Assignment)
    assert result.modes == {"a": rich, "b": medium}
    assert result.exposure_bytes(slots) == 40


# ---------------------------------------------------------------------------
# Validation


@pytest.mark.parametrize("bad", [
    dict(domain=()),
    dict(domain=((FULL, -1),)),
    dict(domain=((FULL, 1), (wsa.Mode("X", 3), 2))),
    dict(tier=T.CARRIED),
])
def test_slot_validation(bad):
    kwargs = dict(key="k", tier=T.FRESH, domain=((FULL, 1), (D, 0)))
    kwargs.update(bad)
    with pytest.raises(ValueError):
        slot(kwargs["key"], kwargs["tier"], kwargs["domain"])


def test_duplicate_keys_and_bad_caps_are_rejected():
    s = slot("k", T.FRESH, [(FULL, 1)])
    with pytest.raises(ValueError):
        wsa.solve([s, s], fixed_bytes=0, cap=10)
    with pytest.raises(ValueError):
        wsa.solve([s], fixed_bytes=-1, cap=10)
    with pytest.raises(ValueError):
        wsa.solve([s], fixed_bytes=0, cap=-1)


def test_empty_slots_only_fixed_bytes():
    assert wsa.solve([], fixed_bytes=10, cap=10) == wsa.Assignment(modes={}, total_bytes=10, floor_bytes=10,
                                                                   fixed_bytes=10, cap=10)
    assert isinstance(wsa.solve([], fixed_bytes=11, cap=10), wsa.Blocked)


# ---------------------------------------------------------------------------
# Random instances: properties and the exhaustive reference


def random_instance(rng: random.Random, modes: tuple[wsa.Mode, ...], max_slots: int = 8):
    count = rng.randint(1, max_slots)
    slots = []
    for index in range(count):
        tier = rng.choice(list(T))
        kind = rng.choice(["product", "tests", "context", "frame", "history"])
        size = rng.randint(1, len(modes))
        chosen = sorted(rng.sample(modes, size), key=lambda m: m.rank, reverse=True)
        costs = sorted((rng.randint(0, 40) for _ in chosen), reverse=True)
        if rng.random() < 0.2:  # not always monotone in rank: a richer form can be cheaper
            rng.shuffle(costs)
        domain = tuple(zip(chosen, costs))
        previous = rng.choice(modes) if (tier is T.CARRIED or rng.random() < 0.3) else None
        if tier is T.CARRIED and all(m.rank > previous.rank for m in chosen):
            previous = chosen[0]  # the generator must build valid CARRIED slots (ceiling >= some entry)
        slots.append(wsa.Slot(key=f"s{index}", kind=kind, priority_tier=tier,
                              admission_ordinal=rng.randint(0, 4), last_use_ordinal=rng.randint(0, 4),
                              domain=domain, previous_mode=previous))
    fixed = rng.randint(0, 20)
    total_richest = fixed + sum(s.richest_cost for s in slots)
    cap = rng.randint(0, total_richest + 5)
    return slots, fixed, cap


def to_reference(slots):
    return [{"key": s.key, "kind": s.kind, "tier": s.priority_tier.name, "admission": s.admission_ordinal,
             "last_use": s.last_use_ordinal, "previous": s.previous_mode.name if s.previous_mode else None,
             "previous_rank": s.previous_mode.rank if s.previous_mode else None,
             "domain": [(m.name, m.rank, c) for m, c in s.domain]} for s in slots]


@pytest.mark.parametrize("modes,seed", [(wsa.DEFAULT_MODES, 11), (SYN_MODES, 12)])
def test_matches_exhaustive_reference(modes, seed):
    rng = random.Random(seed)
    feasible = infeasible = 0
    for _ in range(EXHAUSTIVE_INSTANCES):
        slots, fixed, cap = random_instance(rng, modes)
        ours = wsa.solve(slots, fixed, cap)
        reference = solve_exhaustive(to_reference(slots), fixed, cap)
        if isinstance(ours, wsa.Blocked):
            infeasible += 1
            assert reference is None, (slots, fixed, cap)
            assert ours.floor_bytes == ref_floor(to_reference(slots), fixed)
            continue
        feasible += 1
        assert reference is not None, (slots, fixed, cap)
        assert {k: m.name for k, m in ours.modes.items()} == reference["modes"], (slots, fixed, cap)
        assert ours.total_bytes == reference["bytes"]
    assert feasible > EXHAUSTIVE_INSTANCES // 4 and infeasible > EXHAUSTIVE_INSTANCES // 20


def test_properties_cap_determinism_floor_and_fresh_full():
    rng = random.Random(7)
    for _ in range(3000):
        slots, fixed, cap = random_instance(rng, wsa.DEFAULT_MODES, max_slots=24)
        first = wsa.solve(slots, fixed, cap)
        shuffled = list(slots)
        rng.shuffle(shuffled)
        again = wsa.solve(shuffled, fixed, cap)
        assert first == again  # input order never matters
        floor = fixed + sum(s.min_cost for s in slots)
        if floor > cap:
            assert isinstance(first, wsa.Blocked) and first.floor_bytes == floor
            continue
        assert isinstance(first, wsa.Assignment)
        assert first.total_bytes <= cap
        assert first.total_bytes == fixed + sum(s.cost_of(first.modes[s.key]) for s in slots)
        assert set(first.modes) == {s.key for s in slots}
        fresh = [s for s in slots if s.priority_tier is T.FRESH]
        rest_floor = sum(s.min_cost for s in slots if s.priority_tier is not T.FRESH)
        if fixed + sum(s.richest_cost for s in fresh) + rest_floor <= cap:
            assert all(first.modes[s.key] == s.richest_first[0][0] for s in fresh)


# ---------------------------------------------------------------------------
# A1b admission arithmetic


def test_k1_giant_single_read_is_truncated_not_blocked():
    decisions = wsa.admit_read_bytes(78_000, [384_000])
    assert decisions == (wsa.AdmissionDecision(wsa.ADMIT_TRUNCATE, 78_000),)
    assert decisions[0].admitted


def test_k2_batch_overflow_admits_in_order_then_refuses_rest():
    decisions = wsa.admit_read_bytes(78_000, [24_000] * 16)
    kinds = [d.kind for d in decisions]
    assert kinds == ["FULL"] * 3 + ["TRUNCATE"] + ["REFUSED_ADMISSION"] * 12
    assert decisions[3].admitted_bytes == 78_000 - 3 * 24_000 == 6_000
    assert all(d.admitted_bytes == 0 for d in decisions[4:])
    assert sum(d.admitted_bytes for d in decisions) == 78_000
    assert len(decisions) == 16  # nothing dropped


def test_admission_exact_fit_and_zero_remaining_refuses_rather_than_truncating_to_zero():
    assert wsa.admit_read_bytes(10, [4, 6, 1]) == (wsa.AdmissionDecision("FULL", 4), wsa.AdmissionDecision("FULL", 6),
                                                   wsa.AdmissionDecision("REFUSED_ADMISSION", 0))
    assert wsa.admit_read_bytes(0, [0, 5]) == (wsa.AdmissionDecision("FULL", 0), wsa.AdmissionDecision("REFUSED_ADMISSION", 0))
    assert wsa.admit_read_bytes(5, []) == ()
    with pytest.raises(ValueError):
        wsa.admit_read_bytes(-1, [1])
    with pytest.raises(ValueError):
        wsa.admit_read_bytes(1, [-1])


def test_fresh_exposure_allowance_uses_protected_min_cost():
    protected = [slot("p1", T.PROTECTED, [(FULL, 50)]), slot("p2", T.PROTECTED, [(FULL, 40), (D, 10)])]
    assert wsa.fresh_exposure_allowance(cap=100, fixed_bytes=20, protected_slots=protected) == 20
    assert wsa.fresh_exposure_allowance(cap=50, fixed_bytes=20, protected_slots=protected) == 0


def test_admission_property_never_exceeds_allowance_and_preserves_order(seed=3):
    rng = random.Random(seed)
    for _ in range(2000):
        allowance = rng.randint(0, 200)
        pages = [rng.randint(0, 120) for _ in range(rng.randint(0, 8))]
        decisions = wsa.admit_read_bytes(allowance, pages)
        assert len(decisions) == len(pages)
        assert sum(d.admitted_bytes for d in decisions) <= allowance
        seen_partial = False
        for page, decision in zip(pages, decisions):
            if decision.kind == "FULL":
                assert (not seen_partial or page == 0) and decision.admitted_bytes == page
            elif decision.kind == "TRUNCATE":
                assert not seen_partial and 0 < decision.admitted_bytes < page
                seen_partial = True
            else:
                assert decision.admitted_bytes == 0
                seen_partial = True


# ---------------------------------------------------------------------------
# Round-2 review (design review): F1 FIFO, F2 no post-break promotion, F3 tier coercion, F4 line admission


def test_fresh_is_fifo_within_a_kind_like_admission():
    """G1 'fresh FIFO' / A1 'in batch order': the oldest admitted page keeps FULL, the newest degrades.

    The read tool admitted these pages in batch order, so the newest is the one that already
    carries a continuation cursor; degrading the oldest would contradict what the model was told.
    """
    slots = [slot("new", T.FRESH, [(FULL, 10), (D, 1)], admission=2),
             slot("mid", T.FRESH, [(FULL, 10), (D, 1)], admission=1),
             slot("old", T.FRESH, [(FULL, 10), (D, 1)], admission=0)]
    result = wsa.solve(slots, fixed_bytes=0, cap=21)
    assert isinstance(result, wsa.Assignment)
    assert result.order == ("old", "mid", "new")
    assert result.modes == {"old": FULL, "mid": FULL, "new": D}


def test_carried_slot_is_never_promoted_above_previous_mode_after_the_break():
    """Plan L398: capacity increases never auto-promote. The tail competes by LRU but stays <= previous."""
    slots = [slot("c1", T.CARRIED, [(FULL, 50), (D, 1)], previous=FULL, admission=1),
             slot("c2", T.CARRIED, [(FULL, 10), (D, 4), (O, 1)], previous=D, admission=2, last_use=99)]
    # c1 cannot keep FULL (50 + 4 > 30) -> break at c1; c2 joins the tail with plenty of room.
    result = wsa.solve(slots, fixed_bytes=0, cap=30)
    assert isinstance(result, wsa.Assignment)
    assert result.break_key == "c1"
    assert result.modes["c2"] is D
    assert result.modes["c1"] is D
    # The floor uses the effective (<= previous) domain too: a cheap richer mode does not lower it.
    cheap_rich = slot("c3", T.CARRIED, [(FULL, 0), (D, 5)], previous=D)
    assert cheap_rich.min_cost == 5
    assert isinstance(wsa.solve([cheap_rich], fixed_bytes=0, cap=4), wsa.Blocked)


def test_carried_domain_entirely_richer_than_previous_is_rejected():
    with pytest.raises(ValueError):
        slot("c", T.CARRIED, [(FULL, 10)], previous=D)


def test_int_priority_tier_is_coerced_to_the_enum():
    s = wsa.Slot(key="k", kind="product", priority_tier=0, admission_ordinal=0, last_use_ordinal=0,
                 domain=((FULL, 1),))
    assert s.priority_tier is T.FRESH
    result = wsa.solve([s], fixed_bytes=0, cap=1)
    assert isinstance(result, wsa.Assignment) and result.modes == {"k": FULL}
    with pytest.raises(ValueError):
        wsa.Slot(key="k", kind="product", priority_tier=9, admission_ordinal=0, last_use_ordinal=0,
                 domain=((FULL, 1),))


def test_admit_read_lines_cuts_on_a_line_boundary_and_refuses_empty_pages():
    pages = [[10, 10, 10], [7, 7, 7], [3, 3]]
    decisions = wsa.admit_read_lines(45, pages)
    assert [d.kind for d in decisions] == ["FULL", "TRUNCATE", "REFUSED_ADMISSION"]
    assert decisions[0] == wsa.LineAdmissionDecision("FULL", 30, 3)
    assert decisions[1] == wsa.LineAdmissionDecision("TRUNCATE", 14, 2)  # 15 bytes left -> two whole lines
    assert decisions[2] == wsa.LineAdmissionDecision("REFUSED_ADMISSION", 0, 0)
    # Not even the first line fits: that is a refusal with a cursor at the page start, not an empty TRUNCATE.
    assert wsa.admit_read_lines(5, [[10, 10]]) == (wsa.LineAdmissionDecision("REFUSED_ADMISSION", 0, 0),)
    # An empty page is FULL with zero lines, like admit_read_bytes.
    assert wsa.admit_read_lines(0, [[]]) == (wsa.LineAdmissionDecision("FULL", 0, 0),)
    with pytest.raises(ValueError):
        wsa.admit_read_lines(1, [[-1]])


def test_admit_read_lines_never_exceeds_allowance_and_agrees_with_byte_arithmetic(seed=5):
    rng = random.Random(seed)
    for _ in range(1500):
        allowance = rng.randint(0, 200)
        pages = [[rng.randint(1, 30) for _ in range(rng.randint(0, 6))] for _ in range(rng.randint(0, 6))]
        by_lines = wsa.admit_read_lines(allowance, pages)
        by_bytes = wsa.admit_read_bytes(allowance, [sum(p) for p in pages])
        assert len(by_lines) == len(pages)
        assert sum(d.admitted_bytes for d in by_lines) <= allowance
        for page, line_decision, byte_decision in zip(pages, by_lines, by_bytes):
            assert line_decision.admitted_bytes == sum(page[:line_decision.admitted_lines])
            assert line_decision.admitted_bytes <= byte_decision.admitted_bytes
            if byte_decision.kind == "FULL":
                assert line_decision.kind == "FULL" and line_decision.admitted_lines == len(page)
            elif byte_decision.kind == "REFUSED_ADMISSION":
                assert line_decision.kind == "REFUSED_ADMISSION"
            else:  # TRUNCATE by bytes: whole lines, or refused when not even one line fits
                assert line_decision.kind in {"TRUNCATE", "REFUSED_ADMISSION"}
                assert (line_decision.kind == "TRUNCATE") == (line_decision.admitted_lines > 0)
                assert 0 <= line_decision.admitted_lines < len(page)
