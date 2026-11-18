"""Table and seeded property tests for the edit merge core (stateframe.engine.kernel_edit_merge).

Every case drives merge_span or reanchor, the entry points the apply paths will call. Ranges are byte offsets, so
_br computes them from the bytes of a unique needle.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from stateframe.engine.kernel_edit_merge import (
    AppliedHunk,
    MergeAmbiguous,
    MergeApplied,
    MergeConflict,
    ConflictHunk,
    merge_span,
    reanchor,
)


def _br(text: str, needle: str) -> tuple[int, int]:
    """Byte range of the single occurrence of needle in text."""
    assert text.count(needle) == 1, needle
    start = len(text[: text.index(needle)].encode("utf-8"))
    return start, start + len(needle.encode("utf-8"))


# --- reanchor -----------------------------------------------------------------------------------------------------------


def test_reanchor_unique_span_returns_byte_range() -> None:
    cur = "x = 1\nprint(x)\n"
    assert reanchor("print(x)\n", cur, None) == _br(cur, "print(x)\n")


def test_reanchor_absent_span_is_none() -> None:
    assert reanchor("missing\n", "a\nb\n", 1) is None


def test_reanchor_duplicate_without_hint_is_none() -> None:
    assert reanchor("foo()\n", "foo()\nx\nfoo()\n", None) is None


def test_reanchor_duplicate_with_strictly_nearest_hint_picks_it() -> None:
    cur = "foo()\nx\ny\nz\nfoo()\n"
    second = len(cur[: cur.rindex("foo()")].encode("utf-8"))
    assert reanchor("foo()\n", cur, 5) == (second, second + len("foo()\n"))


def test_reanchor_duplicate_with_tie_is_none() -> None:
    cur = "foo()\nx\nfoo()\n"  # occurrences on lines 1 and 3; hint 2 is equidistant
    assert reanchor("foo()\n", cur, 2) is None


def test_reanchor_counts_overlapping_occurrences() -> None:
    assert reanchor("aa", "aaa", None) is None


def test_reanchor_empty_span_is_none() -> None:
    assert reanchor("", "anything", 1) is None


# --- merge_span: direct, reanchored, duplicate --------------------------------------------------------------------------


def test_unchanged_file_applies_directly() -> None:
    base = "a\nb\nc\n"
    res = merge_span(base, _br(base, "b\n"), "B\n", base)
    assert res == MergeApplied("a\nB\nc\n", "direct", _br(base, "b\n"), (2, 4), ())


def test_edit_above_shifts_span_and_reanchors() -> None:
    base = "a\nb\nc\nd\n"
    cur = "new\na\nb\nc\nd\n"
    res = merge_span(base, _br(base, "c\n"), "C\n", cur)
    assert isinstance(res, MergeApplied)
    assert (res.mode, res.text) == ("reanchored", "new\na\nb\nC\nd\n")
    assert res.current_range == _br(res.text, "C\n")


def test_edit_below_reanchors() -> None:
    base = "a\nb\nc\nd\n"
    res = merge_span(base, _br(base, "c\n"), "C\n", "a\nb\nc\nd\ne\n")
    assert isinstance(res, MergeApplied)
    assert res.text == "a\nb\nC\nd\ne\n"


@pytest.mark.parametrize(
    "cur",
    [
        "head\nfoo()\ntail\nfoo()\n",  # S03-style duplicate placed elsewhere
        "head\nfoo()\nfoo()\ntail\n",  # duplicate placed right next to the original
    ],
)
def test_duplicate_span_never_silently_picks_a_location(cur: str) -> None:
    base = "head\nfoo()\ntail\n"
    res = merge_span(base, _br(base, "foo()\n"), "bar()\n", cur)
    assert isinstance(res, MergeAmbiguous)


# --- merge_span: three-way line merge -----------------------------------------------------------------------------------


def test_same_lines_changed_differently_conflicts() -> None:
    base = "a\nb\nc\n"
    res = merge_span(base, _br(base, "b\n"), "B1\n", "a\nB2\nc\n")
    assert res == MergeConflict(
        (
            ConflictHunk(
                base_lines=(1, 2),
                ours_lines=(1, 2),
                theirs_lines=(1, 2),
                base_text="b\n",
                ours_text="B1\n",
                theirs_text="B2\n",
            ),
        )
    )


def test_same_change_on_both_sides_merges() -> None:
    base = "a\nb\nc\n"
    res = merge_span(base, _br(base, "b\n"), "B\n", "a\nB\nc\n")
    assert res == MergeApplied("a\nB\nc\n", "merged", _br(base, "b\n"), (2, 4), (AppliedHunk("both", (1, 2)),))


def test_theirs_deleting_span_conflicts() -> None:
    base = "a\nb\nc\n"
    res = merge_span(base, _br(base, "b\n"), "B\n", "a\nc\n")
    assert isinstance(res, MergeConflict)
    assert res.hunks[0].theirs_text == ""
    assert res.hunks[0].ours_text == "B\n"


def test_insert_next_to_theirs_change_merges_in_order() -> None:
    base = "a\nb\nc\n"
    res = merge_span(base, (2, 2), "X\n", "a\nB\nc\n")
    assert isinstance(res, MergeApplied)
    assert res.text == "a\nX\nB\nc\n"
    assert res.mode == "merged"
    assert res.hunks == (AppliedHunk("ours", (1, 1)), AppliedHunk("theirs", (1, 2)))
    assert res.current_range == (2, 4)


def test_insert_far_from_theirs_change_merges() -> None:
    base = "1\n2\n3\n4\n5\n6\n"
    res = merge_span(base, (4, 4), "X\n", "1\n2\n3\n4\nF\n6\n")
    assert isinstance(res, MergeApplied)
    assert res.text == "1\n2\nX\n3\n4\nF\n6\n"


def test_theirs_rewrites_everything_is_ambiguous() -> None:
    base = "a\nb\nc\n"
    res = merge_span(base, _br(base, "b\n"), "B\n", "X\nY\nZ\n")
    assert isinstance(res, MergeAmbiguous)


def test_ours_deletes_all_lines_conflicts_with_theirs_edit() -> None:
    base = "a\nb\n"
    res = merge_span(base, (0, 4), "", "a\nX\n")
    assert isinstance(res, MergeConflict)
    assert res.hunks[0].base_lines == (0, 2)


def test_ours_deletes_all_lines_untouched_file_applies_directly() -> None:
    base = "a\nb\n"
    assert merge_span(base, (0, 4), "", base) == MergeApplied("", "direct", (0, 4), (0, 0), ())


def test_ours_edit_equal_to_base_span_takes_theirs_without_conflict() -> None:
    base = "a\nb\nc\n"
    cur = "a\nB\nc\n"
    res = merge_span(base, _br(base, "b\n"), "b\n", cur)
    assert res == MergeApplied(cur, "merged", _br(base, "b\n"), None, ())


# --- merge_span: line endings, BOM, unicode, empty, no trailing newline -------------------------------------------------


def test_crlf_reanchored_keeps_line_endings() -> None:
    base = "a\r\nb\r\nc\r\n"
    res = merge_span(base, _br(base, "b\r\n"), "B\r\n", "top\r\na\r\nb\r\nc\r\n")
    assert isinstance(res, MergeApplied)
    assert res.text == "top\r\na\r\nB\r\nc\r\n"


def test_crlf_three_way_insert_keeps_line_endings() -> None:
    base = "a\r\nb\r\nc\r\n"
    res = merge_span(base, (3, 3), "X\r\n", "a\r\nb\r\nC\r\n")
    assert isinstance(res, MergeApplied)
    assert res.text == "a\r\nX\r\nb\r\nC\r\n"


def test_no_trailing_newline_reanchors() -> None:
    base = "a\nb\nc"
    res = merge_span(base, _br(base, "c"), "C", "z\na\nb\nc")
    assert isinstance(res, MergeApplied)
    assert res.text == "z\na\nb\nC"


def test_no_trailing_newline_three_way_append_merges() -> None:
    base = "a\nb"
    res = merge_span(base, (3, 3), "\nd", "A\nb")
    assert isinstance(res, MergeApplied)
    assert res.text == "A\nb\nd"


def test_bom_is_preserved_and_ranges_count_its_bytes() -> None:
    base = "﻿a\nb\n"
    res = merge_span(base, _br(base, "b\n"), "B\n", "﻿A\nb\n")
    assert isinstance(res, MergeApplied)
    assert res.text == "﻿A\nB\n"
    assert res.current_range == _br(res.text, "B\n")


def test_non_ascii_bytes_are_counted_correctly() -> None:
    base = "café\nnaïve\n"
    res = merge_span(base, _br(base, "naïve\n"), "naive\n", "top\ncafé\nnaïve\n")
    assert isinstance(res, MergeApplied)
    assert res.text == "top\ncafé\nnaive\n"
    assert res.current_range == _br(res.text, "naive\n")


def test_empty_base_and_current_applies_directly() -> None:
    assert merge_span("", (0, 0), "hello\n", "") == MergeApplied("hello\n", "direct", (0, 0), (0, 6), ())


def test_empty_base_with_different_current_conflicts() -> None:
    res = merge_span("", (0, 0), "hello\n", "other\n")
    assert res == MergeConflict(
        (
            ConflictHunk(
                base_lines=(0, 0),
                ours_lines=(0, 1),
                theirs_lines=(0, 1),
                base_text="",
                ours_text="hello\n",
                theirs_text="other\n",
            ),
        )
    )


@pytest.mark.parametrize("base_range", [(1, 2), (4, 2)], ids=["splits-character", "reversed"])
def test_invalid_base_range_raises(base_range: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        merge_span("é\n", base_range, "x", "é\n!\n")


def test_merge_is_deterministic() -> None:
    base = "a\nb\nc\n"
    args = (base, _br(base, "b\n"), "B1\n", "a\nB2\nc\n")
    assert merge_span(*args) == merge_span(*args)


# --- seeded property tests ----------------------------------------------------------------------------------------------

CASES = 300


def _base_lines(rng: random.Random, n: int) -> list[str]:
    """Unique lines, so any base span is unique in the file unless a test makes it otherwise."""
    return [f"r{k}-{rng.getrandbits(32):08x}\n" for k in range(n)]


def _new_lines(rng: random.Random, tag: str, count: int) -> list[str]:
    return [f"{tag}{k}-{rng.getrandbits(32):08x}\n" for k in range(count)]


def _apply(lines: list[str], edits: list[tuple[int, int, list[str]]]) -> str:
    out = list(lines)
    for start, end, new in sorted(edits, key=lambda e: e[0], reverse=True):
        out[start:end] = new
    return "".join(out)


def _byte_pos(lines: list[str], line: int) -> int:
    return len("".join(lines[:line]).encode("utf-8"))


def _disjoint_edits(rng: random.Random, n: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Ours and theirs ranges with at least one untouched line between them (either order)."""
    while True:
        a = rng.randint(0, n)
        b = min(n, a + rng.randint(0, 3))
        c = rng.randint(0, n)
        d = min(n, c + rng.randint(0, 3))
        if d < a or c > b:
            return (a, b), (c, d)


def test_property_disjoint_edits_always_merge_to_both_edits() -> None:
    modes: Counter[str] = Counter()
    for seed in range(CASES):
        rng = random.Random(seed)
        lines = _base_lines(rng, rng.randint(20, 200))
        (a, b), (c, d) = _disjoint_edits(rng, len(lines))
        ours_new = _new_lines(rng, "o", rng.randint(0, 3))
        theirs_new = _new_lines(rng, "t", rng.randint(0, 3))
        base = "".join(lines)
        cur = _apply(lines, [(c, d, theirs_new)])
        expected = _apply(lines, [(a, b, ours_new), (c, d, theirs_new)])
        bs = _byte_pos(lines, a)
        be = _byte_pos(lines, b)
        ours_text = "".join(ours_new)
        res = merge_span(base, (bs, be), ours_text, cur)
        assert isinstance(res, MergeApplied), f"seed {seed}: {res}"
        assert res.text == expected, f"seed {seed}"
        modes[res.mode] += 1
    assert modes["reanchored"] > 0 and modes["merged"] > 0, modes


def _overlapping_edits(rng: random.Random, n: int) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        a = rng.randint(0, n - 1)
        b = min(n, a + rng.randint(1, 3))
        c = rng.randint(0, n - 1)
        d = min(n, c + rng.randint(1, 3))
        if c < b and a < d:
            return (a, b), (c, d)


def test_property_overlapping_differing_edits_never_silently_merge() -> None:
    outcomes: Counter[str] = Counter()
    for seed in range(CASES):
        rng = random.Random(10_000 + seed)
        lines = _base_lines(rng, rng.randint(20, 200))
        (a, b), (c, d) = _overlapping_edits(rng, len(lines))
        ours_new = _new_lines(rng, "o", rng.randint(0, 3))
        theirs_new = _new_lines(rng, "t", rng.randint(1, 3))  # non-empty, so the two sides always differ
        base = "".join(lines)
        cur = _apply(lines, [(c, d, theirs_new)])
        res = merge_span(base, (_byte_pos(lines, a), _byte_pos(lines, b)), "".join(ours_new), cur)
        assert not isinstance(res, MergeApplied), f"seed {seed} silently merged overlapping edits"
        outcomes[type(res).__name__] += 1
        if isinstance(res, MergeConflict):
            hunk = res.hunks[0]
            assert "".join(ours_new) in hunk.ours_text, f"seed {seed} lost ours"
            assert "".join(theirs_new) in hunk.theirs_text, f"seed {seed} lost theirs"
    assert outcomes["MergeConflict"] > 0, outcomes
