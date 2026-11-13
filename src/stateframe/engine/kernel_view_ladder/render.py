"""Pure renderer for the view ladder (`[read view]` carriers).

A delivered read is an *artifact*: `(display path, revision, held_start, held_end)` plus the
decoded text the model received for those lines. This module renders any rung of that
artifact on demand (full is the original page; range/relevant, symbols/outline and stub are
degraded views) as canonical text whose bytes depend only on the inputs. Nothing here
reads the runtime, the clock, the request or the environment: wiring into the diet, the
finalizer and the dispatch loop is a separate item.

Ground truth this follows: reads are delivered as decoded text and numbered with
`str.splitlines(keepends=True)`; every verbatim row is a slice of that same text at the
stated line numbers, so `lines=a-b` in a view means the same lines as in the page it
replaces. Omissions are explicit records, never an ellipsis inside exact source.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from kernel_view_ladder import structure

RENDERER_VERSION = 1
VIEW_LADDER_ENV = "STATEFRAME_VIEW_LADDER"

# Content order, best first. `range`/`relevant` share a level, as do `symbols`/`outline`.
RUNGS: tuple[str, ...] = ("full", "range", "relevant", "symbols", "outline", "stub")
DEGRADED_RUNGS: tuple[str, ...] = RUNGS[1:]
REASONS: tuple[str, ...] = ("context_limit", "model_keep", "edit_target_refresh")
_REASON_TEXT = {
    "context_limit": "context limit",
    "model_keep": "model keep",
    "edit_target_refresh": "edit target refresh",
}
_RUNG_LEVEL = {"full": 0, "range": 1, "relevant": 1, "symbols": 2, "outline": 2, "stub": 3}

VIEW_PREFIX = "[read view] "
END_VIEW = "[end view]"
STUB_TOKEN = "READ_RESULT_DEGRADED"
MERGE_GAP_LINES = 2

# Header grammar: fixed key order, one grammar for every degraded rung. `lines=a-b/N` is the
# HELD range so the diet's `_source_identity` / `_source_range_record` still classify it.
_READ_VIEW_HEADER = re.compile(
    r"^\[read view\] view=(?P<view>range|relevant|symbols|outline|stub) path=(?P<path>.+?) "
    r"revision=(?P<revision>sha256:[0-9a-f]{64}) lines=(?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+) "
    r"degraded=true reason=(?P<reason>context_limit|model_keep|edit_target_refresh) "
    r"vid=(?P<vid>[0-9a-f]{12}) body_sha256=(?P<body_sha256>[0-9a-f]{64})$"
)
READ_VIEW_HEADER = _READ_VIEW_HEADER
_RANGE_MARKER = re.compile(r"^--- lines (?P<start>\d+)-(?P<end>\d+)(?: \((?P<label>[^)]*)\))? ---$")
_REVISION = re.compile(r"^sha256:[0-9a-f]{64}$")


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Lenient switch: default off; nothing imports the renderer on the shipped path yet."""
    if environ is None:
        environ = os.environ
    return str(environ.get(VIEW_LADDER_ENV, "")).strip().lower() == "on"


def is_degraded_view(content: object) -> bool:
    """True for a `[read view]` carrier; coverage readers return before counting any range."""
    return isinstance(content, str) and content.startswith(VIEW_PREFIX)


def rung_level(rung: str) -> int:
    try:
        return _RUNG_LEVEL[rung]
    except KeyError:
        raise ValueError(f"unknown rung {rung!r}") from None


def split_delivered_lines(text: str) -> list[str]:
    """The one line rule: identical to `deliver_source`, so view and page line numbers agree."""
    return text.splitlines(keepends=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def short_revision(revision: str, width: int = 8) -> str:
    return revision.split(":", 1)[-1][:width]


def view_id(*, display: str, revision: str, held_start: int, held_end: int, rung: str,
            reason: str, frozen_selection: Sequence[Sequence[object]] | None) -> str:
    """vid = sha256(RENDERER_VERSION|path|revision|a|b|rung|reason|json(selection))[:12]."""
    normalized = normalize_ranges(frozen_selection or (), held_start, held_end)
    selection = json.dumps(_selection_json(normalized), sort_keys=True, separators=(",", ":"))
    material = "|".join([str(RENDERER_VERSION), display, revision, str(held_start), str(held_end),
                         rung, reason, selection])
    return hashlib.sha256(material.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]


def _selection_json(selection: Sequence[Sequence[object]] | None) -> list[list[object]]:
    return [[int(item[0]), int(item[1]), *(str(part) for part in item[2:] if str(part))] for item in selection or ()]


@dataclass(frozen=True)
class Artifact:
    """One delivered read: what the model actually received for `lines=a-b/N`."""

    display: str
    revision: str
    held_start: int
    held_end: int
    total_lines: int
    held_text: str

    def __post_init__(self) -> None:
        if not _REVISION.match(self.revision):
            raise ValueError("revision must be sha256:<64hex>")
        if self.held_start < 1 and not (self.held_start == 1 and self.held_end == 0):
            raise ValueError("held_start must be >= 1")
        expected = max(0, self.held_end - self.held_start + 1)
        got = len(split_delivered_lines(self.held_text))
        if got != expected:
            raise ValueError(f"held_text has {got} lines, header says {expected}")
        if self.total_lines < self.held_end:
            raise ValueError("total_lines must cover held_end")

    @property
    def lines(self) -> list[str]:
        return split_delivered_lines(self.held_text)

    @property
    def is_empty(self) -> bool:
        return self.held_end < self.held_start

    def line(self, number: int) -> str:
        """Exact delivered line `number` (1-based file numbering) including its terminator."""
        if not self.held_start <= number <= self.held_end:
            raise IndexError(number)
        return self.lines[number - self.held_start]

    def slice(self, start: int, end: int) -> str:
        if not (self.held_start <= start <= end <= self.held_end):
            raise IndexError((start, end))
        return "".join(self.lines[start - self.held_start:end - self.held_start + 1])


@dataclass(frozen=True)
class Rendered:
    rung: str
    vid: str
    body_sha256: str
    text: str


# --- range helpers -----------------------------------------------------------------------

def normalize_ranges(ranges: Iterable[Sequence[object]], held_start: int, held_end: int,
                     ) -> tuple[tuple[int, int, str], ...]:
    """Clamp to the held range, sort, merge ranges within MERGE_GAP_LINES; keep labels."""
    items: list[tuple[int, int, str]] = []
    for item in ranges:
        start, end = int(item[0]), int(item[1])
        if end < start:
            start, end = end, start
        start, end = max(start, held_start), min(end, held_end)
        if end < start:
            continue
        label = ", ".join(str(part) for part in item[2:] if str(part))  # spec 1.5: "(frame, observed)"
        items.append((start, end, label))
    items.sort(key=lambda row: (row[0], row[1], row[2]))
    merged: list[tuple[int, int, str]] = []
    for start, end, label in items:
        if merged and start - merged[-1][1] - 1 <= MERGE_GAP_LINES:
            p_start, p_end, p_label = merged[-1]
            labels = sorted({part for joined in (p_label, label) for part in joined.split("+") if part})
            merged[-1] = (p_start, max(p_end, end), "+".join(labels))
        else:
            merged.append((start, end, label))
    return tuple(merged)


def complement(ranges: Sequence[tuple[int, int, str]], held_start: int, held_end: int) -> list[tuple[int, int]]:
    """Exact omitted complement of `ranges` inside the held range."""
    out: list[tuple[int, int]] = []
    cursor = held_start
    for start, end, _label in ranges:
        if start > cursor:
            out.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= held_end:
        out.append((cursor, held_end))
    return out


def format_ranges(ranges: Sequence[tuple[int, int]]) -> str:
    if not ranges:
        return "none"
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)


def range_marker(start: int, end: int, label: str = "") -> str:
    """The marker row above each kept range (`_RANGE_MARKER` parses it; relevance budgets with it)."""
    suffix = f" ({label})" if label else ""
    return f"--- lines {start}-{end}{suffix} ---"


def _ensure_row_break(segment: str) -> str:
    """Verbatim rows are emitted exactly; a single LF separates an unterminated row from the next marker."""
    return segment if segment.endswith("\n") else segment + "\n"


# --- bodies ------------------------------------------------------------------------------

def reread_instruction(artifact: Artifact) -> str:
    return (f'read(path="{artifact.display}", start_line={artifact.held_start}, '
            f'end_line={artifact.held_end})')


def render_stub_line(*, display: str, revision: str, held_start: int, held_end: int,
                     total_lines: int, reason: str) -> str:
    """The one-line stub statement shared with the retention notice (sha=/lines= fields)."""
    if reason not in _REASON_TEXT:
        raise ValueError(f"unknown reason {reason!r}")
    return (f"{STUB_TOKEN} path={display} sha=sha256:{short_revision(revision, 12)} "
            f"lines={held_start}-{held_end}/{total_lines} full content omitted ({_REASON_TEXT[reason]}); "
            f're-read for full: read(path="{display}", start_line={held_start}, end_line={held_end})')


def _stub_body(artifact: Artifact, reason: str) -> list[str]:
    return [render_stub_line(display=artifact.display, revision=artifact.revision,
                             held_start=artifact.held_start, held_end=artifact.held_end,
                             total_lines=artifact.total_lines, reason=reason)]


def _range_body(artifact: Artifact, ranges: tuple[tuple[int, int, str], ...]) -> list[str]:
    if not ranges:
        raise ValueError("range view needs at least one sub-range inside the held range")
    held_count = artifact.held_end - artifact.held_start + 1
    kept = sum(end - start + 1 for start, end, _ in ranges)
    omitted = complement(ranges, artifact.held_start, artifact.held_end)
    out = [f"kept {kept} of {held_count} lines in {len(ranges)} ranges; omitted: {format_ranges(omitted)}"]
    for start, end, label in ranges:
        out.append(range_marker(start, end, label))
        # Rows are joined with LF below, so the row's own (or added) final LF is the separator.
        out.append(_ensure_row_break(artifact.slice(start, end))[:-1])
    out.append(f"{END_VIEW} re-read for full: {reread_instruction(artifact)}")
    return out


def _structure_body(artifact: Artifact, outline: structure.Outline, rung: str) -> list[str]:
    out = [f"parser={outline.parser}"]
    rows = outline.rows if rung == "symbols" else [row for row in outline.rows
                                                   if row.kind not in structure.SYMBOLS_ONLY_KINDS]
    for row in rows:
        out.append(_format_row(row))
    if rung == "symbols":
        if outline.facts:
            out.append("syntactic: " + "; ".join(outline.facts[:structure.FACT_CAP]))
        out.append("omitted: " + outline.omitted_record)
    else:
        out.append("omitted: bodies")
    symbol = outline.reread_symbol
    if symbol:
        out.append(f'{END_VIEW} bodies omitted; re-read a symbol with read(path="{artifact.display}", '
                   f'symbol="{symbol}") or a range with start_line/end_line')
    else:
        out.append(f"{END_VIEW} re-read for full: {reread_instruction(artifact)}")
    return out


def _format_row(row: structure.Row) -> str:
    if row.enclosing:
        label = f"L{row.start}"
        tail = f" (enclosing, starts L{row.start})"
    elif row.continues:
        label = f"L{row.start}-..."
        tail = ""
    elif row.end is None or row.end == row.start:
        label = f"L{row.start}"
        tail = ""
    else:
        label = f"L{row.start}-{row.end}"
        tail = ""
    return f"{label:<7} " + "  " * row.depth + row.text + tail


# --- change notice ------------------------------------------------------------------------

def change_notice(artifact: Artifact, *, disk_revision: str | None, disk_text: str | None) -> str | None:
    """One line, no hunk bodies, no edit authority; None when disk is still at the held revision."""
    if disk_revision is None or disk_text is None or disk_revision == artifact.revision:
        return None
    if not _REVISION.match(disk_revision):
        raise ValueError("disk_revision must be sha256:<64hex>")
    old = artifact.lines
    new = split_delivered_lines(disk_text)
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    ops = [op for op in matcher.get_opcodes()]
    # Leading/trailing pure inserts are file content outside the held range, not changes inside it.
    while ops and ops[0][0] == "insert":
        ops.pop(0)
    while ops and ops[-1][0] == "insert":
        ops.pop()
    added = removed = 0
    if not any(tag == "equal" for tag, *_ in ops):
        ops = [("delete", 0, len(old), 0, 0)]  # nothing of the held range survives; new text is outside it
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("insert", "replace"):
            added += j2 - j1
        if tag in ("delete", "replace"):
            removed += i2 - i1
    if ops and any(tag == "equal" for tag, *_ in ops):
        mapped = f"{ops[0][3] + 1}-{ops[-1][4]}"
    else:
        mapped = "none"
    return (f"changed: file is now @{short_revision(disk_revision)}; held "
            f"{artifact.held_start}-{artifact.held_end} maps to {mapped} "
            f"(+{added} -{removed} lines inside range); re-read before relying on this view")


# --- entry points -------------------------------------------------------------------------

def applicable_rungs(artifact: Artifact, *, language: str | None = None,
                     file_text: str | None = None) -> tuple[str, ...]:
    """Rungs whose 'applies when' column holds for this artifact (content rule, not size)."""
    if artifact.is_empty:
        return ("stub",)
    out: list[str] = ["range", "relevant"]
    outline = structure.outline_for(artifact, language=language, file_text=file_text)
    if outline is not None and outline.applicable:
        out.extend(["symbols", "outline"])
    out.append("stub")
    return tuple(out)


def render_view(artifact: Artifact, rung: str, *, reason: str,
                ranges: Iterable[Sequence[object]] | None = None,
                frozen_selection: Sequence[Sequence[object]] | None = None,
                language: str | None = None, file_text: str | None = None,
                disk_revision: str | None = None, disk_text: str | None = None) -> Rendered:
    """Render any degraded rung of `artifact` as canonical text (LF header, exact rows).

    `ranges` are the model's sub-ranges for `range`; `frozen_selection` is code's selection
    for `relevant` (`(start, end, anchor_kind, edge)` rows). Both are normalized (clamped,
    sorted, merged within 2 lines) so input order never changes the bytes.
    """
    if rung not in DEGRADED_RUNGS:
        raise ValueError(f"render_view renders degraded rungs only, not {rung!r}")
    if reason not in REASONS:
        raise ValueError(f"unknown reason {reason!r}")
    selection: tuple[tuple[int, int, str], ...] = ()
    if rung == "range":
        selection = normalize_ranges(ranges or (), artifact.held_start, artifact.held_end)
        body = _range_body(artifact, selection)
    elif rung == "relevant":
        selection = normalize_ranges(frozen_selection or (), artifact.held_start, artifact.held_end)
        body = _range_body(artifact, selection)
    elif rung in ("symbols", "outline"):
        outline = structure.outline_for(artifact, language=language, file_text=file_text)
        if outline is None or not outline.applicable:
            raise ValueError(f"{rung} does not apply: held range has no structure")
        body = _structure_body(artifact, outline, rung)
    else:
        body = _stub_body(artifact, reason)
    notice = change_notice(artifact, disk_revision=disk_revision, disk_text=disk_text)
    if notice:
        body.insert(len(body) - 1, notice)
    body_text = "\n".join(body)
    vid = view_id(display=artifact.display, revision=artifact.revision, held_start=artifact.held_start,
                  held_end=artifact.held_end, rung=rung, reason=reason, frozen_selection=selection)
    digest = sha256_text(body_text)
    header = (f"{VIEW_PREFIX}view={rung} path={artifact.display} revision={artifact.revision} "
              f"lines={artifact.held_start}-{artifact.held_end}/{artifact.total_lines} "
              f"degraded=true reason={reason} vid={vid} body_sha256={digest}")
    return Rendered(rung=rung, vid=vid, body_sha256=digest, text=header + "\n" + body_text)


def render_stub(artifact: Artifact, *, reason: str, **kwargs: object) -> Rendered:
    return render_view(artifact, "stub", reason=reason, **kwargs)  # type: ignore[arg-type]


def parse_view_header(content: str) -> dict[str, object] | None:
    """Parsed first line of a `[read view]` carrier, or None when it is not one."""
    if not is_degraded_view(content):
        return None
    first = content.split("\n", 1)[0]
    match = _READ_VIEW_HEADER.match(first)
    if match is None:
        return None
    row = match.groupdict()
    for key in ("start", "end", "total"):
        row[key] = int(row[key])
    return row


def verify_body_sha256(content: str) -> bool:
    header = parse_view_header(content)
    if header is None:
        return False
    body = content.split("\n", 1)[1] if "\n" in content else ""
    return sha256_text(body) == header["body_sha256"]


def iter_verbatim_rows(content: str) -> Iterator[tuple[int, int, str]]:
    """Yield `(start, end, text)` for every verbatim block of a range/relevant view.

    Blocks are consumed by line COUNT (the marker states a-b), so a source line that looks
    like a marker cannot confuse the walk. The text is the exact slice, except that an
    unterminated final row carries the single LF the renderer added as its separator.
    """
    header = parse_view_header(content)
    if header is None or header["view"] not in ("range", "relevant"):
        return
    body = content.split("\n", 1)[1]
    units = split_delivered_lines(body)
    index = 0
    while index < len(units):
        marker = _RANGE_MARKER.match(units[index].rstrip("\n"))
        index += 1
        if marker is None:
            continue
        start, end = int(marker["start"]), int(marker["end"])
        count = end - start + 1
        segment = "".join(units[index:index + count])
        index += count
        if not segment.endswith("\n") and index < len(units) and units[index] == "\n":
            index += 1  # the separator LF after an unterminated row (e.g. a lone \x0c)
        yield start, end, segment


def row_matches_slice(row_text: str, slice_text: str) -> bool:
    """Exactness contract for verbatim rows (see `iter_verbatim_rows`)."""
    if row_text == slice_text:
        return True
    return (not slice_text.endswith("\n")) and row_text == slice_text + "\n"
