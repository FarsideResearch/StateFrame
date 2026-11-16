"""Request-scoped edit handles and the positive [CURRENT SOURCE] map.

Pure module: no I/O, no environment reads, no engine wiring. It turns the source spans
a request delivered into handles named ``q{request_ordinal}h{n}`` and renders one map
block that says, for every delivered file, which handle covers which lines and which
lines were NOT shown. The map is an address book: it never contains source bodies, so
there are no gutters or omission markers for a model to copy into old_text.

Determinism: handles are a pure function of (request_ordinal, span order). Journal
replay and resume re-mint the same names from the same inputs; nothing here reads a
clock or a counter outside its arguments.

Site inventory: handles are derived FROM open sites (path, revision, lines) and never
written back into them. The map lives only in the final user message of a request, so
the open-site inventory identity is unaffected by its presence (the per-request
delivery annotations excluded there are a separate carrier; this module adds none).
"""
from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from kernel_paged_source import source_revision

SCHEMA = "stateframe-edit-handle-map/1"
CURRENT_SOURCE_PREFIX = "[CURRENT SOURCE]"
# Same spelling as the refusal catalog HANDLE_PATTERN; kept local so the catalog
# stays import-free from the engine side until a later unit wires both in.
HANDLE_PATTERN = re.compile(r"\bq\d+h\d+\b")
_HANDLE_RE = re.compile(r"^q(\d+)h(\d+)$")
_REVISION_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SpanKind(str, enum.Enum):
    """SPAN covers a window of lines; REWRITE covers the whole file and licenses rewrite."""

    SPAN = "SPAN"
    REWRITE = "REWRITE"


def handle_name(request_ordinal: int, ordinal: int) -> str:
    """``q{request_ordinal}h{ordinal}``; the only place a handle is spelled."""
    _check_ordinal(request_ordinal, "request_ordinal")
    if type(ordinal) is not int or ordinal < 1:
        raise ValueError("handle ordinal must be a positive integer")
    name = f"q{request_ordinal}h{ordinal}"
    assert HANDLE_PATTERN.fullmatch(name)
    return name


def parse_handle(handle: str) -> tuple[int, int] | None:
    """(request_ordinal, ordinal) for a well-formed handle, else None."""
    if not isinstance(handle, str):
        return None
    match = _HANDLE_RE.match(handle)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _check_ordinal(value: object, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _lines(source: str) -> list[str]:
    # Same split as kernel_paged_source so line numbers agree with existing descriptors.
    return source.splitlines(keepends=True)


def line_byte_range(source: str, start_line: int, end_line: int) -> tuple[int, int, int]:
    """UTF-8 byte offsets [start, end) of inclusive 1-based lines, plus the file's line count."""
    if not isinstance(source, str):
        raise TypeError("source must be text")
    lines = _lines(source)
    total = len(lines)
    if type(start_line) is not int or type(end_line) is not int:
        raise ValueError("line numbers must be integers")
    if start_line < 1 or end_line < start_line or end_line > total:
        raise ValueError(
            f"line span {start_line}-{end_line} is outside the file (1-{total})")
    byte_start = sum(len(line.encode("utf-8")) for line in lines[: start_line - 1])
    byte_end = byte_start + sum(
        len(line.encode("utf-8")) for line in lines[start_line - 1 : end_line])
    return byte_start, byte_end, total


@dataclass(frozen=True)
class DeliveredSpan:
    """One span of source a request showed the model, before it has a handle."""

    path: str
    revision: str
    start_line: int
    end_line: int
    total_lines: int
    byte_start: int
    byte_end: int
    kind: SpanKind = SpanKind.SPAN

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("path must be a non-empty string")
        if not isinstance(self.revision, str) or not _REVISION_RE.match(self.revision):
            raise ValueError("revision must be a sha256: digest")
        for name in ("start_line", "end_line", "total_lines", "byte_start", "byte_end"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.start_line < 1 or self.end_line < self.start_line or self.end_line > self.total_lines:
            raise ValueError(
                f"line span {self.start_line}-{self.end_line} is outside the file "
                f"(1-{self.total_lines})")
        if self.byte_start < 0 or self.byte_end < self.byte_start:
            raise ValueError("byte range must be non-negative and ordered")
        if not isinstance(self.kind, SpanKind):
            raise ValueError(f"unknown span kind: {self.kind!r}")
        if self.kind is SpanKind.REWRITE and (
                self.start_line != 1 or self.end_line != self.total_lines or self.byte_start != 0):
            raise ValueError("a REWRITE span must cover the whole file")

    @classmethod
    def from_source(cls, path: str, source: str, *, start_line: int | None = None,
                    end_line: int | None = None,
                    kind: SpanKind = SpanKind.SPAN) -> "DeliveredSpan":
        """Build a span from the full current source; omitted lines mean the whole file."""
        total = len(_lines(source))
        if total == 0:
            raise ValueError("cannot deliver a span of an empty file")
        first = 1 if start_line is None else start_line
        last = total if end_line is None else end_line
        byte_start, byte_end, total = line_byte_range(source, first, last)
        return cls(path=path, revision=source_revision(source), start_line=first,
                   end_line=last, total_lines=total, byte_start=byte_start,
                   byte_end=byte_end, kind=kind)

    def span_key(self) -> tuple[str, str, int, int, str]:
        return (self.path, self.revision, self.byte_start, self.byte_end, self.kind.value)


def spans_from_open_sites(sites: Iterable[Mapping[str, object]],
                          sources: Mapping[str, str]) -> tuple[DeliveredSpan, ...]:
    """Read delivered windows off open-site rows without touching the rows.

    ``sources`` maps path -> full current text; a site whose ``revision`` disagrees with
    its source is stale and raises rather than minting a handle to bytes the model never
    saw. Sites the request did not show (``read_this_request`` is False) are skipped.
    """
    spans: list[DeliveredSpan] = []
    for site in sites:
        if site.get("read_this_request") is False:
            continue
        path = site.get("path")
        lines = site.get("lines")
        if not isinstance(path, str) or not isinstance(lines, Mapping):
            raise ValueError("open site needs a path and a lines mapping")
        source = sources.get(path)
        if not isinstance(source, str):
            raise ValueError(f"no current source for delivered site {path!r}")
        revision = site.get("revision")
        if revision is not None and revision != source_revision(source):
            raise ValueError(f"open site revision for {path!r} does not match current source")
        start, end = lines.get("start"), lines.get("end")
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"open site {path!r} has no integer line span")
        spans.append(DeliveredSpan.from_source(path, source, start_line=start, end_line=end))
    return tuple(spans)


@dataclass(frozen=True)
class HandleRecord:
    """A minted handle: the span plus its request-scoped name."""

    handle: str
    request_ordinal: int
    ordinal: int
    span: DeliveredSpan

    @property
    def path(self) -> str:
        return self.span.path

    @property
    def revision(self) -> str:
        return self.span.revision

    @property
    def kind(self) -> SpanKind:
        return self.span.kind

    def covers_bytes(self, byte_start: int, byte_end: int) -> bool:
        """True when [byte_start, byte_end) lies inside this handle's span."""
        return (self.span.byte_start <= byte_start <= byte_end <= self.span.byte_end)

    def text(self, source: str) -> str:
        """The covered bytes of the current source; raises when the source moved."""
        if source_revision(source) != self.span.revision:
            raise ValueError("source revision does not match this handle")
        return source.encode("utf-8")[self.span.byte_start:self.span.byte_end].decode("utf-8")

    def as_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "request_ordinal": self.request_ordinal,
            "ordinal": self.ordinal,
            "path": self.span.path,
            "revision": self.span.revision,
            "kind": self.span.kind.value,
            "start_line": self.span.start_line,
            "end_line": self.span.end_line,
            "total_lines": self.span.total_lines,
            "byte_start": self.span.byte_start,
            "byte_end": self.span.byte_end,
        }


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def line_gaps(spans: Sequence[DeliveredSpan]) -> list[tuple[int, int]]:
    """Inclusive line ranges of one file that no span in ``spans`` covers."""
    if not spans:
        return []
    total = spans[0].total_lines
    if any(span.total_lines != total for span in spans):
        raise ValueError("spans of one file must agree on total_lines")
    gaps: list[tuple[int, int]] = []
    cursor = 1
    for start, end in _merge_ranges((span.start_line, span.end_line) for span in spans):
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = end + 1
    if cursor <= total:
        gaps.append((cursor, total))
    return gaps


def _format_ranges(ranges: Sequence[tuple[int, int]]) -> str:
    return ", ".join(f"{start}-{end}" if start != end else str(start) for start, end in ranges)


@dataclass(frozen=True)
class HandleMap:
    """All handles of one request, in minting order."""

    request_ordinal: int
    records: tuple[HandleRecord, ...]

    def lookup(self, handle: str) -> HandleRecord | None:
        for record in self.records:
            if record.handle == handle:
                return record
        return None

    def paths(self) -> tuple[str, ...]:
        seen: list[str] = []
        for record in self.records:
            if record.path not in seen:
                seen.append(record.path)
        return tuple(seen)

    def for_path(self, path: str) -> tuple[HandleRecord, ...]:
        return tuple(record for record in self.records if record.path == path)

    def gaps(self, path: str) -> list[tuple[int, int]]:
        return line_gaps([record.span for record in self.for_path(path)])

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "request_ordinal": self.request_ordinal,
            "handles": [record.as_dict() for record in self.records],
        }

    def render(self) -> str:
        """The positive map block appended to the final user message.

        Lists every handle and states the lines not shown per file. No bodies.
        """
        head = (f"{CURRENT_SOURCE_PREFIX}\nrequest q{self.request_ordinal}: "
                f"{len(self.records)} handle{'s' if len(self.records) != 1 else ''} over "
                f"{len(self.paths())} file{'s' if len(self.paths()) != 1 else ''}. ")
        if not self.records:
            return head + "No source was delivered in this request; read before editing."
        lines = [head + ("Each handle covers exactly the lines listed; lines marked not "
                         "shown were not delivered in this request and cannot be edited "
                         "through these handles.")]
        for path in self.paths():
            records = self.for_path(path)
            lines.append(f"file {path} revision {records[0].revision} "
                         f"({records[0].span.total_lines} lines)")
            for record in records:
                kind = ("rewrite, whole file" if record.kind is SpanKind.REWRITE else "span")
                lines.append(f"  {record.handle} lines {record.span.start_line}-"
                             f"{record.span.end_line} ({kind})")
            gaps = self.gaps(path)
            if gaps:
                lines.append(f"  not shown: lines {_format_ranges(gaps)}")
            else:
                lines.append("  not shown: none; the whole file is delivered")
        return "\n".join(lines)


def mint_handles(request_ordinal: int, spans: Sequence[DeliveredSpan]) -> HandleMap:
    """Mint ``q{request_ordinal}h1..hN`` in span order. Same inputs, same handles.

    Spans of one file must share a revision (one request sees one version of a file)
    and an identical span may appear only once, so each handle names distinct bytes.
    """
    _check_ordinal(request_ordinal, "request_ordinal")
    seen: set[tuple[str, str, int, int, str]] = set()
    revisions: dict[str, str] = {}
    records: list[HandleRecord] = []
    for index, span in enumerate(spans, start=1):
        if not isinstance(span, DeliveredSpan):
            raise ValueError("spans must be DeliveredSpan records")
        key = span.span_key()
        if key in seen:
            raise ValueError(f"duplicate delivered span for {span.path!r}")
        seen.add(key)
        if revisions.setdefault(span.path, span.revision) != span.revision:
            raise ValueError(f"spans of {span.path!r} disagree on revision within one request")
        records.append(HandleRecord(handle=handle_name(request_ordinal, index),
                                    request_ordinal=request_ordinal, ordinal=index, span=span))
    return HandleMap(request_ordinal=request_ordinal, records=tuple(records))


def append_map_to_final_user_message(messages: Sequence[Mapping[str, object]],
                                     handle_map: HandleMap) -> list[dict[str, object]]:
    """Return a new message list with the map appended to the final user message only.

    Earlier messages are returned as the same objects, so the retained prefix of request
    N and N+1 stays byte-identical and cache-safe. The final message must be a user
    message with string content and must not already carry a map.
    """
    if not messages:
        raise ValueError("no messages to place the map in")
    final = messages[-1]
    content = final.get("content")
    if final.get("role") != "user" or not isinstance(content, str):
        raise ValueError("the final message must be a user message with text content")
    if CURRENT_SOURCE_PREFIX in content:
        raise ValueError("the final user message already carries a current source map")
    separator = "" if not content else ("\n\n" if not content.endswith("\n") else "\n")
    placed = {**final, "content": content + separator + handle_map.render()}
    return [*messages[:-1], placed]


_FILE_LINE_RE = re.compile(r"^file (?P<path>.+) revision (?P<revision>sha256:[0-9a-f]{64}) "
                           r"\((?P<total>\d+) lines\)$")
_HANDLE_LINE_RE = re.compile(r"^  (?P<handle>q\d+h\d+) lines (?P<start>\d+)-(?P<end>\d+) "
                             r"\((?P<kind>span|rewrite, whole file)\)$")
_GAP_LINE_RE = re.compile(r"^  not shown: (?P<body>.+)$")


def parse_map(text: str) -> dict[str, object] | None:
    """Parse a rendered map back into rows; None when ``text`` holds no map block.

    Returns {"request_ordinal", "files": [{"path", "revision", "total_lines",
    "handles": [{"handle", "start_line", "end_line", "kind"}], "not_shown": [(a, b)]}]}.
    """
    if not isinstance(text, str):
        return None
    index = text.find(CURRENT_SOURCE_PREFIX)
    if index < 0:
        return None
    lines = text[index:].split("\n")
    header = re.match(r"^request q(\d+): ", lines[1]) if len(lines) > 1 else None
    if header is None:
        return None
    files: list[dict[str, object]] = []
    for line in lines[2:]:
        file_match = _FILE_LINE_RE.match(line)
        if file_match:
            files.append({"path": file_match.group("path"),
                          "revision": file_match.group("revision"),
                          "total_lines": int(file_match.group("total")),
                          "handles": [], "not_shown": []})
            continue
        handle_match = _HANDLE_LINE_RE.match(line)
        if handle_match and files:
            handles = files[-1]["handles"]
            assert isinstance(handles, list)
            handles.append({
                "handle": handle_match.group("handle"),
                "start_line": int(handle_match.group("start")),
                "end_line": int(handle_match.group("end")),
                "kind": SpanKind.REWRITE.value if handle_match.group("kind").startswith("rewrite")
                else SpanKind.SPAN.value,
            })
            continue
        gap_match = _GAP_LINE_RE.match(line)
        if gap_match and files:
            body = gap_match.group("body")
            if body.startswith("lines "):
                not_shown = files[-1]["not_shown"]
                assert isinstance(not_shown, list)
                for piece in body[len("lines "):].split(", "):
                    start, _, end = piece.partition("-")
                    not_shown.append((int(start), int(end or start)))
            continue
        break
    return {"request_ordinal": int(header.group(1)), "files": files}
