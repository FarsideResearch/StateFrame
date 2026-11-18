"""Pure three-way merge for an edit whose base span may have drifted on disk.

An edit names the span it was shown (base bytes) and offers replacement text. If the file moved since the model read it,
the span is re-located by its exact bytes when that location is unique. Otherwise the change is merged line by line
(diff3 semantics): non-overlapping changes combine, overlapping changes that differ conflict, and any edit whose location
cannot be proven is ambiguous. A repeated span is never placed by position, because line alignment cannot prove which
copy the model meant.

Contract: no I/O, no globals, deterministic. Ranges given to and returned by merge_span are UTF-8 byte offsets. Line
ranges are zero-based half-open indices. Lines split on "\\n" only, so CRLF, bare CR, a BOM and an unterminated last line
survive byte-for-byte. Replacement text is inserted exactly as given; callers normalize line endings before calling.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

ByteRange = tuple[int, int]
LineRange = tuple[int, int]

_ENC = "utf-8"
_ERR = "surrogatepass"  # lone surrogates must not crash offset math


@dataclass(frozen=True)
class AppliedHunk:
    """One change that landed in a merged result."""

    side: Literal["ours", "theirs", "both"]
    base_lines: LineRange


@dataclass(frozen=True)
class MergeApplied:
    text: str
    mode: Literal["direct", "reanchored", "merged"]
    base_range: ByteRange
    current_range: ByteRange | None  # where the replacement sits in text; None when the edit was a no-op
    hunks: tuple[AppliedHunk, ...]


@dataclass(frozen=True)
class ConflictHunk:
    base_lines: LineRange
    ours_lines: LineRange  # lines of base with ours applied
    theirs_lines: LineRange  # lines of the current file
    base_text: str
    ours_text: str
    theirs_text: str


@dataclass(frozen=True)
class MergeConflict:
    hunks: tuple[ConflictHunk, ...]


@dataclass(frozen=True)
class MergeAmbiguous:
    reason: str


MergeOutcome = MergeApplied | MergeConflict | MergeAmbiguous


@dataclass(frozen=True)
class _Change:
    start: int
    end: int
    text: str
    side: Literal["ours", "theirs", "both"]


@dataclass(frozen=True)
class _Text:
    text: str
    lines: list[str]
    starts: list[int]

    @classmethod
    def of(cls, text: str) -> _Text:
        return cls(text, _split_lines(text), _line_starts(text))

    def pos(self, line: int) -> int:
        """Char offset where line begins; the end of text for the index one past the last line."""
        return self.starts[line] if line < len(self.starts) else len(self.text)


def _nbytes(text: str) -> int:
    return len(text.encode(_ENC, _ERR))


def _char_index(text: str, byte_off: int) -> int:
    raw = text.encode(_ENC, _ERR)
    if not 0 <= byte_off <= len(raw):
        raise ValueError(f"byte offset {byte_off} is outside a {len(raw)}-byte file")
    try:
        return len(raw[:byte_off].decode(_ENC, _ERR))
    except UnicodeDecodeError as exc:
        raise ValueError(f"byte offset {byte_off} splits a UTF-8 character") from exc


def _split_lines(text: str) -> list[str]:
    """Split on newline only. Joining the result reproduces text exactly."""
    parts = text.split("\n")
    lines = [part + "\n" for part in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def _line_starts(text: str) -> list[int]:
    return [0, *(i + 1 for i, ch in enumerate(text) if ch == "\n")]


def _line_bounds(text: str, cs: int, ce: int) -> LineRange:
    """Lines touched by char span [cs, ce). A span ending mid-line claims that whole line."""
    first = text.count("\n", 0, cs)
    if ce == 0 or text[ce - 1] == "\n":
        return first, text.count("\n", 0, ce)
    return first, text.count("\n", 0, ce) + 1


def _occurrences(haystack: str, needle: str) -> list[int]:
    """Every start index of needle, overlapping matches included ("aa" occurs twice in "aaa")."""
    hits: list[int] = []
    idx = haystack.find(needle)
    while idx >= 0:
        hits.append(idx)
        idx = haystack.find(needle, idx + 1)
    return hits


def reanchor(base_span: str, current_file: str, hint_line: int | None) -> ByteRange | None:
    """Byte range of base_span in current_file, or None when absent or not provably the right copy.

    A repeated span resolves only when hint_line (1-based) is strictly nearer to exactly one occurrence.
    """
    if not base_span:
        return None
    hits = _occurrences(current_file, base_span)
    if len(hits) > 1:
        if hint_line is None:
            return None
        dists = [abs(current_file.count("\n", 0, h) + 1 - hint_line) for h in hits]
        best = min(dists)
        if dists.count(best) != 1:
            return None
        idx = hits[dists.index(best)]
    elif hits:
        idx = hits[0]
    else:
        return None
    start = _nbytes(current_file[:idx])
    return start, start + _nbytes(base_span)


def merge_span(base_file: str, base_range: ByteRange, ours_span: str, current_file: str) -> MergeOutcome:
    """Apply the edit base_file[base_range] -> ours_span onto current_file.

    Order: identical file -> direct. Span bytes unique in current -> reanchored. Otherwise three-way line merge,
    unless the span occurs more than once, which is ambiguous by definition.
    """
    bs, be = base_range
    if be < bs:
        raise ValueError("base_range ends before it starts")
    cs, ce = _char_index(base_file, bs), _char_index(base_file, be)
    base_span = base_file[cs:ce]
    if current_file == base_file:
        text = base_file[:cs] + ours_span + base_file[ce:]
        return MergeApplied(text, "direct", (bs, be), (bs, bs + _nbytes(ours_span)), ())
    if base_span:
        hits = _occurrences(current_file, base_span)
        if len(hits) == 1:
            idx = hits[0]
            start = _nbytes(current_file[:idx])
            text = current_file[:idx] + ours_span + current_file[idx + len(base_span) :]
            return MergeApplied(text, "reanchored", (bs, be), (start, start + _nbytes(ours_span)), ())
        if hits:
            return MergeAmbiguous(f"base span occurs {len(hits)} times in the current file; its location is unprovable")
    if ours_span == base_span:
        return MergeApplied(current_file, "merged", (bs, be), None, ())
    return _three_way(base_file, (bs, be), (cs, ce), ours_span, current_file)


def _overlaps(i1: int, i2: int, a: int, b: int) -> bool:
    """Shared lines, or two pure insertions at one point. Touching ranges do not overlap."""
    if i1 < b and a < i2:
        return True
    return i1 == i2 == a == b


def _anchored(x: int, ops: list[tuple[str, int, int, int, int]], nb: int) -> bool:
    """A base boundary is located when it is file start/end or sits on a line both sides kept."""
    return x == 0 or x == nb or any(tag == "equal" and i1 <= x <= i2 for tag, i1, i2, _j1, _j2 in ops)


def _equal_bound(
    ops: list[tuple[str, int, int, int, int]], x: int, cur_len: int, *, prefer_last: bool
) -> int:
    """Current-file line matching base boundary x. Last match puts insertions at x before the bound, first after."""
    found: int | None = None
    for tag, i1, i2, j1, _j2 in ops:
        if tag == "equal" and i1 <= x <= i2:
            found = j1 + (x - i1)
            if not prefer_last:
                return found
    if found is not None:
        return found
    return 0 if x == 0 else cur_len


def _three_way(
    base_file: str, base_range: ByteRange, span: tuple[int, int], ours_span: str, cur_file: str
) -> MergeOutcome:
    cs, ce = span
    base, cur = _Text.of(base_file), _Text.of(cur_file)
    a, b = _line_bounds(base_file, cs, ce)
    ops = SequenceMatcher(None, base.lines, cur.lines, autojunk=False).get_opcodes()
    nb = len(base.lines)
    if not (_anchored(a, ops, nb) and _anchored(b, ops, nb)):
        return MergeAmbiguous("no stable anchor lines around the edited span; both sides rewrote its surroundings")

    changed = [op for op in ops if op[0] != "equal"]
    overlapping = [op for op in changed if _overlaps(op[1], op[2], a, b)]
    changes: list[_Change] = []
    lo, hi = a, b
    if overlapping:
        g1 = min(a, overlapping[0][1])
        g2 = max(b, overlapping[-1][2])
        lo, hi = g1, g2
        t1 = overlapping[0][3] if overlapping[0][1] == g1 else _equal_bound(ops, g1, len(cur.lines), prefer_last=True)
        t2 = overlapping[-1][4] if overlapping[-1][2] == g2 else _equal_bound(ops, g2, len(cur.lines), prefer_last=False)
        base_group = base_file[base.pos(g1) : base.pos(g2)]
        ours_group = base_file[base.pos(g1) : cs] + ours_span + base_file[ce : base.pos(g2)]
        theirs_group = cur_file[cur.pos(t1) : cur.pos(t2)]
        if ours_group != theirs_group:
            hunk = ConflictHunk(
                base_lines=(g1, g2),
                ours_lines=(g1, g1 + len(_split_lines(ours_group))),
                theirs_lines=(t1, t2),
                base_text=base_group,
                ours_text=ours_group,
                theirs_text=theirs_group,
            )
            return MergeConflict((hunk,))
        changes.append(_Change(g1, g2, ours_group, "both"))  # identical change on both sides: not a conflict
    else:
        ours_region = base_file[base.pos(a) : cs] + ours_span + base_file[ce : base.pos(b)]
        changes.append(_Change(a, b, ours_region, "ours"))

    for tag, i1, i2, j1, j2 in changed:
        if not _overlaps(i1, i2, lo, hi):
            changes.append(_Change(i1, i2, cur_file[cur.pos(j1) : cur.pos(j2)], "theirs"))

    changes.sort(key=lambda c: (c.start, c.end))
    out: list[str] = []
    hunks: list[AppliedHunk] = []
    cursor = 0
    length = 0
    ours_at = 0
    for change in changes:
        head = base_file[base.pos(cursor) : base.pos(change.start)]
        out.append(head)
        length += len(head)
        if change.side != "theirs":
            ours_at = length + (cs - base.pos(change.start))
        out.append(change.text)
        length += len(change.text)
        hunks.append(AppliedHunk(change.side, (change.start, change.end)))
        cursor = change.end
    out.append(base_file[base.pos(cursor) :])
    merged = "".join(out)
    start = _nbytes(merged[:ours_at])
    return MergeApplied(merged, "merged", base_range, (start, start + _nbytes(ours_span)), tuple(hunks))
