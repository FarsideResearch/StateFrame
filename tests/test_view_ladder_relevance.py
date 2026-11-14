"""B6: relevance selection for the ``relevant`` rung (spec 1.5, build plan row B6).

Priority order, budget never exceeded, merge rule, determinism, frozen
selection, seeds absent -> fallback, model-supplied ranges win.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
from kernel_view_ladder import relevance as rel
from kernel_view_ladder.relevance import Seed

SOURCE = """\
import json
from typing import Sequence

DEFAULT_SCALE = 2


class Calc:
    \"\"\"Scaled accumulator.\"\"\"

    def __init__(self, scale: int = DEFAULT_SCALE) -> None:
        self.total = 0
        self.scale = scale

    def add(self, value: int) -> int:
        self.total += value * self.scale
        return self.total

    def divide(self, value: int) -> float:
        return self.total / value


def mean(values: Sequence[int]) -> float:
    return sum(values) / len(values)


def dump(calc: Calc) -> str:
    return json.dumps({"total": calc.total, "mode": "HTTP/2"})


def run(values):
    calc = Calc()
    for value in values:
        calc.add(value)
    return dump(calc) + str(mean(values)) + str(helper(values))
"""
LINES = SOURCE.splitlines(keepends=True)
HELD_END = len(LINES)
BIG = 1 << 20


def _runtime(**attrs):
    owner = SimpleNamespace(task_row=attrs.pop("task_row", {}), fn=attrs.pop("fn", None),
                            information_policy=None)
    return SimpleNamespace(owner=owner, **attrs)


def _select(seeds, budget=BIG, text=SOURCE, start=1, end=None):
    return rel.select(seeds, held_text=text, held_start=start,
                      held_end=end or len(text.splitlines(keepends=True)), budget_bytes=budget)


# --------------------------------------------------------------------------- outline


def test_outline_python_ast_offsets_to_held_start():
    defs = rel.outline(SOURCE, 1)
    names = {d.name: (d.start, d.end) for d in defs}
    assert names["Calc"] == (7, 19)
    assert names["add"] == (14, 16)
    assert names["mean"] == (22, 23)
    shifted = rel.outline(SOURCE, 101)
    assert {d.name: (d.start, d.end) for d in shifted}["mean"] == (122, 123)


def test_outline_dedent_retry_and_generic_scan():
    body = "".join(LINES[13:19])  # the indented add/divide methods only
    defs = rel.outline(body, 14)
    assert [(d.name, d.start) for d in defs] == [("add", 14), ("divide", 18)]
    js = "export function one() {\n  return 1;\n}\n\nconst two = 2;\nfunction three() {\n}\n"
    generic = rel.outline(js, 1, language="javascript")
    assert [(d.name, d.start, d.end) for d in generic] == [("one", 1, 3), ("three", 6, 7)]


# --------------------------------------------------------------------------- seeds


def test_seed_validation():
    with pytest.raises(ValueError):
        Seed("bogus", 1, 2, 0, rel.EDGE_OBSERVED)
    with pytest.raises(ValueError):
        Seed("frame", 1, 2, 0, "maybe")
    with pytest.raises(ValueError):
        Seed("frame", 3, 2, 0, rel.EDGE_OBSERVED)


def test_call_site_regex_matches_spec():
    lines = ["mean(values)", "x.mean(1)", "  mean (2)", "meanwhile(3)", "mean"]
    assert rel.call_sites("mean", lines, 10) == [10, 12]
    assert rel.call_sites("not an ident", lines, 1) == []


def test_seeds_priority_order_and_edges():
    runtime = _runtime(
        task_row={"fn": "mean", "problem_statement": "mean() breaks on 'HTTP/2' input"},
        _request_source_closure={"units": [
            {"path": "pkg/calc.py", "start_line": 26, "end_line": 27, "role": "body"},
            {"path": "other.py", "start_line": 1, "end_line": 5, "role": "body"}]},
        _latest_check_receipt={"status": "FAIL", "runner_tail":
                               "pkg/calc.py:23: ZeroDivisionError\nFAILED tests/test_calc.py::test_mean"},
        _framefit_action_context={"framefit5": {"applied_edits": [
            {"path": "pkg/calc.py", "lines": {"start": 15}, "result_class": "APPLIED"}]}},
    )
    seeds = rel.seeds_from_runtime(runtime, path="pkg/calc.py", held_text=SOURCE, held_start=1,
                                   held_end=HELD_END, override_ranges=[(30, 31)],
                                   requested_ranges=[(1, 4)])
    priorities = [s.priority for s in seeds]
    assert priorities == sorted(priorities), "seeds come out in priority order"
    by_kind = {}
    for seed in seeds:
        by_kind.setdefault(seed.kind, []).append(seed)
    assert by_kind["model"][0] == Seed("model", 30, 31, rel.PRIORITY_MODEL, rel.EDGE_RESOLVED, "model:30-31")
    assert (1, 4) in {(s.start, s.end) for s in by_kind["requested"]}
    assert by_kind["unit"] == [Seed("unit", 26, 27, rel.PRIORITY_REQUESTED, rel.EDGE_RESOLVED, "unit:body")]
    frame = by_kind["frame"][0]
    assert (frame.start, frame.end, frame.edge) == (11, HELD_END, rel.EDGE_OBSERVED)  # 23 +- 12, clipped
    changed = by_kind["changed"][0]
    assert (changed.start, changed.end, changed.edge) == (14, 16, rel.EDGE_RESOLVED)  # Calc.add
    symbol = by_kind["symbol"][0]
    assert (symbol.start, symbol.end, symbol.edge) == (22, 23, rel.EDGE_RESOLVED)
    assert any(s.label == "lit:HTTP/2" and s.start == 27 for s in by_kind["lit"]), "original spelling matched"
    callers = [s for s in by_kind["call"] if s.label == "call:mean"]
    assert callers and callers[0].edge == rel.EDGE_RESOLVED
    assert callers[0].start == 32  # line 34 +- 2, clipped to the held range


def test_unresolved_call_site_for_symbol_not_defined_in_range():
    runtime = _runtime(task_row={"fn": "helper"})
    seeds = rel.seeds_from_runtime(runtime, path="pkg/calc.py", held_text=SOURCE, held_start=1,
                                   held_end=HELD_END)
    expected = Seed("call", 32, HELD_END, rel.PRIORITY_NEIGHBOUR, rel.EDGE_UNRESOLVED, "call:helper")
    assert seeds == (expected,)


def test_failing_test_seed_resolves_test_definition():
    text = "def test_mean_empty():\n    assert mean([]) == 0\n\n\ndef test_other():\n    pass\n"
    tail = "FAILED tests/test_calc.py::test_mean_empty - x"
    runtime = _runtime(_latest_check_receipt={"runner_tail": tail})
    seeds = rel.seeds_from_runtime(runtime, path="tests/test_calc.py", held_text=text,
                                   held_start=1, held_end=6)
    assert seeds[0] == Seed("test", 1, 2, rel.PRIORITY_FAILURE, rel.EDGE_OBSERVED,
                            "test:test_mean_empty")


def test_seeds_outside_held_range_are_clipped_or_dropped():
    seeds = rel.seeds_from_runtime(_runtime(), path="p.py", held_text="".join(LINES[9:19]),
                                   held_start=10, held_end=19, override_ranges=[(5, 12), (40, 50)],
                                   requested_ranges=[(18, 30)])
    assert [(s.kind, s.start, s.end) for s in seeds] == [("model", 10, 12), ("requested", 18, 19)]


def test_neighbours_capped_at_eight_per_seed():
    body = "def hub():\n    pass\n\n" + "".join(f"def f{i}():\n    hub()\n\n" for i in range(12))
    runtime = _runtime(task_row={"fn": "hub"})
    seeds = rel.seeds_from_runtime(runtime, path="p.py", held_text=body, held_start=1,
                                   held_end=len(body.splitlines()))
    assert sum(1 for s in seeds if s.kind == "call") == rel.MAX_NEIGHBOURS


def test_missing_runtime_seams_degrade_to_no_seeds():
    assert rel.seeds_from_runtime(object(), path="p.py", held_text=SOURCE, held_start=1,
                                  held_end=HELD_END) == ()
    assert rel.ticket_text(object()) == ""
    assert rel.closure_units(object(), "p.py") == ()
    assert rel.evidence_lines(object(), "p.py", "") == ()
    assert rel.applied_edit_lines(object(), "p.py") == ()


# --------------------------------------------------------------------------- selection


def test_budget_never_exceeded_and_lower_priority_skipped_first():
    seeds = [Seed("frame", 20, 23, rel.PRIORITY_FAILURE, rel.EDGE_OBSERVED),
             Seed("symbol", 7, 12, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),
             Seed("lit", 1, 1, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED)]
    full = _select(seeds)
    assert full == ((1, 1, "lit", "resolved"), (7, 12, "symbol", "resolved"), (20, 23, "frame", "observed"))
    frame_only = rel.selection_bytes(((20, 23, "frame", "observed"),), SOURCE, 1)
    for budget in range(0, rel.selection_bytes(full, SOURCE, 1) + 40):
        chosen = _select(seeds, budget=budget)
        if chosen is None:
            assert budget < min(frame_only, rel.selection_bytes(((1, 1, "lit", "resolved"),), SOURCE, 1))
            continue
        assert rel.selection_bytes(chosen, SOURCE, 1) <= budget
    tight = _select(seeds, budget=frame_only)
    assert tight == ((20, 23, "frame", "observed"),), "the priority-2 seed wins over both priority-4 seeds"


def test_smaller_lower_priority_seed_still_fits_after_a_skip():
    seeds = [Seed("symbol", 7, 19, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),
             Seed("lit", 4, 4, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED)]
    budget = rel.selection_bytes(((4, 4, "lit", "resolved"),), SOURCE, 1)
    assert _select(seeds, budget=budget) == ((4, 4, "lit", "resolved"),)


def test_merge_within_two_lines_and_disjoint_output():
    seeds = [Seed("frame", 1, 3, rel.PRIORITY_FAILURE, rel.EDGE_OBSERVED),
             Seed("lit", 6, 6, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),   # 2 omitted lines -> merge
             Seed("lit", 10, 11, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),  # 3 omitted lines -> separate
             Seed("symbol", 10, 12, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED)]  # overlaps -> merge
    chosen = _select(seeds)
    # equal priority: the earlier-sorted seed (start, end, kind) names the merged range
    assert chosen == ((1, 6, "frame", "observed"), (10, 12, "lit", "resolved"))
    for (_, end_a, _, _), (start_b, _, _, _) in zip(chosen, chosen[1:]):
        assert start_b > end_a + rel.MERGE_GAP + 1
    assert rel.omitted_ranges(chosen, 1, HELD_END) == ((7, 9), (13, HELD_END))


def test_merged_range_takes_highest_priority_label():
    seeds = [Seed("lit", 5, 5, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),
             Seed("frame", 6, 8, rel.PRIORITY_FAILURE, rel.EDGE_OBSERVED)]
    assert _select(seeds) == ((5, 8, "frame", "observed"),)


def test_merge_counts_gap_lines_against_budget():
    seeds = [Seed("lit", 1, 1, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),
             Seed("lit", 3, 3, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED)]
    merged_cost = rel.selection_bytes(((1, 3, "lit", "resolved"),), SOURCE, 1)
    assert _select(seeds, budget=merged_cost) == ((1, 3, "lit", "resolved"),)
    assert _select(seeds, budget=merged_cost - 1) == ((1, 1, "lit", "resolved"),)


def test_empty_seeds_or_zero_budget_means_rung_not_applicable():
    assert _select([]) is None
    assert _select([Seed("lit", 1, 1, 4, rel.EDGE_RESOLVED)], budget=0) is None


def test_select_rejects_mismatched_held_text():
    with pytest.raises(ValueError):
        rel.select([], held_text=SOURCE, held_start=1, held_end=HELD_END + 1, budget_bytes=BIG)


def test_shuffle_invariance_and_determinism():
    seeds = [Seed("frame", 20, 23, rel.PRIORITY_FAILURE, rel.EDGE_OBSERVED),
             Seed("symbol", 7, 12, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),
             Seed("lit", 1, 1, rel.PRIORITY_IDENTIFIER, rel.EDGE_RESOLVED),
             Seed("call", 30, 33, rel.PRIORITY_NEIGHBOUR, rel.EDGE_UNRESOLVED),
             Seed("unit", 26, 27, rel.PRIORITY_REQUESTED, rel.EDGE_RESOLVED)]
    budget = 600
    reference = _select(seeds, budget=budget)
    signature = rel.selection_signature(reference)
    rng = random.Random(1234)
    for _ in range(25):
        rng.shuffle(seeds)
        again = _select(list(seeds), budget=budget)
        assert again == reference
        assert rel.selection_signature(again) == signature
    assert rel.selection_signature(None) != signature


def test_crlf_and_non_ascii_bytes_counted_exactly():
    text = "a = 'é'\r\nb = 2\r\nc = 3\r\n"
    cost = rel.selection_bytes(((1, 1, "lit", "resolved"),), text, 1)
    assert cost == len("--- lines 1-1 (lit, resolved) ---\n") + len("a = 'é'\r\n".encode())
    assert rel.select([Seed("lit", 1, 1, 4, rel.EDGE_RESOLVED)], held_text=text, held_start=1,
                      held_end=3, budget_bytes=cost) == ((1, 1, "lit", "resolved"),)
    assert rel.select([Seed("lit", 1, 1, 4, rel.EDGE_RESOLVED)], held_text=text, held_start=1,
                      held_end=3, budget_bytes=cost - 1) is None


# --------------------------------------------------------------------------- model override


def test_model_ranges_are_top_priority_under_the_same_budget():
    runtime = _runtime(task_row={"fn": "mean"},
                       _latest_check_receipt={"runner_tail": "pkg/calc.py:23: ZeroDivisionError"})
    kwargs = dict(path="pkg/calc.py", held_text=SOURCE, held_start=1, held_end=HELD_END)
    model_cost = rel.selection_bytes(((10, 12, "model", "resolved"),), SOURCE, 1)
    seeds = rel.seeds_from_runtime(runtime, override_ranges=[(10, 12)], **kwargs)
    assert _select(seeds, budget=model_cost) == ((10, 12, "model", "resolved"),)
    seeds_without = rel.seeds_from_runtime(runtime, **kwargs)
    assert _select(seeds_without, budget=model_cost) != ((10, 12, "model", "resolved"),)
    generous = _select(seeds)
    assert generous is not None
    assert any(s <= 10 and 12 <= e for s, e, _, _ in generous), "model range kept when the budget allows"


def test_model_range_merges_with_code_seeds_under_merge_rule():
    seeds = [Seed("model", 10, 12, rel.PRIORITY_MODEL, rel.EDGE_RESOLVED),
             Seed("frame", 14, 16, rel.PRIORITY_FAILURE, rel.EDGE_OBSERVED)]
    assert _select(seeds) == ((10, 16, "model", "resolved"),)


# --------------------------------------------------------------------------- frozen selection


def test_frozen_selection_is_computed_once_and_never_changes():
    state: dict = {}
    key = ("pkg/calc.py", "sha256:abc", 1, HELD_END)
    calls = []

    def first():
        calls.append("first")
        return ((20, 23, "frame", "observed"),)

    def second():
        calls.append("second")
        return ((1, 5, "model", "resolved"),)

    assert rel.frozen_selection(state, key, first) == ((20, 23, "frame", "observed"),)
    assert rel.frozen_selection(state, key, second) == ((20, 23, "frame", "observed"),)
    assert calls == ["first"], "a frozen selection is re-emitted without recomputation"
    assert state[key]["frozen_selection"] == ((20, 23, "frame", "observed"),)
    other = ("pkg/calc.py", "sha256:def", 1, HELD_END)
    assert rel.frozen_selection(state, other, second) == ((1, 5, "model", "resolved"),)


def test_frozen_selection_survives_json_round_trip():
    state: dict = {"k": {"frozen_selection": ((3, 4, "lit", "resolved"),)}}
    thawed = {"k": json.loads(json.dumps(state["k"]))}
    assert thawed["k"]["frozen_selection"] == [[3, 4, "lit", "resolved"]]
    assert rel.frozen_selection(thawed, "k", lambda: None) == ((3, 4, "lit", "resolved"),)


def test_not_applicable_is_not_frozen():
    state: dict = {}
    assert rel.frozen_selection(state, "k", lambda: None) is None
    assert "frozen_selection" not in state["k"]
    chosen = ((1, 1, "lit", "resolved"),)
    assert rel.frozen_selection(state, "k", lambda: chosen) == chosen


def test_relevant_selection_end_to_end_freezes_across_requests():
    state: dict = {}
    key = ("pkg/calc.py", "sha256:abc", 1, HELD_END)
    runtime = _runtime(task_row={"fn": "mean"})
    kwargs = dict(path="pkg/calc.py", held_text=SOURCE, held_start=1, held_end=HELD_END)
    first = rel.relevant_selection(runtime, state, key, budget_bytes=200, **kwargs)
    assert first is not None and any(kind == "symbol" for _, _, kind, _ in first)
    runtime.owner.task_row = {"fn": "dump"}  # new anchors arrive on a later request
    second = rel.relevant_selection(runtime, state, key, budget_bytes=BIG, override_ranges=[(1, 3)], **kwargs)
    assert second == first


# --------------------------------------------------------------------------- fallback


def test_fallback_head_plus_def_lines_when_no_seeds():
    chosen = rel.fallback_selection(SOURCE, 1, HELD_END, BIG)
    assert chosen is not None
    assert chosen[0][:2] == (1, 14), "head of range merged with the Calc/__init__/add def lines"
    starts = {start for start, _, _, _ in chosen}
    assert {18, 22, 26, 30} <= starts
    assert all(rel.selection_bytes((row,), SOURCE, 1) > 0 for row in chosen)
    head_only = rel.selection_bytes(((1, 3, "requested", "resolved"),), SOURCE, 1)
    head_three = rel.fallback_selection(SOURCE, 1, HELD_END, head_only, head_lines=3)
    assert head_three == ((1, 3, "requested", "resolved"),)
    assert rel.fallback_selection("x\n", 1, 1, 0) is None
