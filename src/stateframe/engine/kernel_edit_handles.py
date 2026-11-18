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
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
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


# ---------------------------------------------------------------------------
# Request map: one minter over two span providers.
#
# Provider A (frame / allocator): open-site rows of the request's [StateFrame]
# block, read through ``spans_from_open_sites``; the rows are never written.
# Provider B (legacy source blocks): exact runs of current file lines that the
# request text carries verbatim ([Current source:]/[Working source:] windows,
# read pages, exact clipped reads). A run counts only when the bytes the model
# saw are byte-equal to the current file, so a handle never names unseen bytes.
# A window that a read receipt names at another revision (or another path) is a
# stale delivery: its text is not scanned for this file (see _delivery_text_and_segments).
# Both providers feed ``request_spans``, which unions covered lines per file and
# mints one handle per maximal run (whole file -> REWRITE). Paged descriptors and
# allocator source-range handles stay paging addresses; edit handles are only
# ever spelled ``q{request_ordinal}h{n}``.
# ---------------------------------------------------------------------------

_STATE_HEADER = "[StateFrame]\n"
_SOURCE_HEADER_RE = re.compile(
    r"\[(?:Current source|Working source|Retained source|Observed source|Repair source"
    r"|Paged source|Exact clipped read): (?P<path>[^\]\n]+)\]")
_READ_PAGE_RE = re.compile(r"\[read page\] path=(?P<path>\S+) ")
_PATH_REVISION_SUFFIX_RE = re.compile(r" revision sha256:[0-9a-f]{64}.*$")
# A read receipt names the revision of the bytes its window was delivered from.
_RECEIPT_RE = re.compile(r"\[read page\] path=(?P<path>.+?) revision=(?P<revision>sha256:[0-9a-f]{64})(?= )")
# Frame-rendered window markers (kernel_request_source_window); a test pins these spellings.
_MODEL_WINDOW_MARKER = "[Model source window]"
_WINDOW_CONTENT_MARKER = "[Source window content]"
_WINDOW_END_MARKER = "[End source window]"
# Delivery text a request carries for the model; assistant turns are model output.
_DELIVERY_ROLES = frozenset({"user", "tool", "system"})
MAX_MAP_FILES = 16
MAX_MAP_FILE_BYTES = 512 * 1024
# A run of fewer non-blank lines than this is coincidence, not delivery
# (a lone "return None" in the task text), unless it is the whole file.
MIN_RUN_NONBLANK_LINES = 2
# A start line that occurs this often in the request text is boilerplate.
MAX_RUN_START_OCCURRENCES = 256

def _delivery_text(messages: Sequence[Mapping[str, object]]) -> str:
    return "\n".join(str(message.get("content")) for message in messages
                     if isinstance(message, Mapping) and message.get("role") in _DELIVERY_ROLES
                     and isinstance(message.get("content"), str))


def _clean_path(raw: str) -> str | None:
    path = _PATH_REVISION_SUFFIX_RE.sub("", raw.strip()).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if not path or path.startswith("/") or any(part == ".." for part in path.split("/")):
        return None
    return path


def _frame_open_sites(messages: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    # Only the host frame row is read: authorship comes from its position in the request,
    # so model, file or tool text that starts with the state header is never parsed.
    from kernel_host_authorship import structural_class

    sites: list[Mapping[str, object]] = []
    for index, message in enumerate(messages):
        if structural_class(messages, index) != "frame":
            continue
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.startswith(_STATE_HEADER):
            continue
        try:
            payload = json.loads(content[len(_STATE_HEADER):])
        except (TypeError, ValueError, RecursionError):
            continue
        state = payload.get("state") if isinstance(payload, Mapping) else None
        for holder in ((state.get("framefit5"), state) if isinstance(state, Mapping) else ()):
            rows = holder.get("open_sites") if isinstance(holder, Mapping) else None
            if isinstance(rows, list):
                sites.extend(row for row in rows if isinstance(row, Mapping))
                break
    return sites


def candidate_paths(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Workspace-relative paths the request names as delivered source, first mention first."""
    seen: list[str] = []
    text = _delivery_text(messages)
    found = [(match.start(), match.group("path"))
             for pattern in (_SOURCE_HEADER_RE, _READ_PAGE_RE, _RECEIPT_RE)
             for match in pattern.finditer(text)]
    for _, raw in sorted(found):
        path = _clean_path(raw)
        if path is not None and path not in seen:
            seen.append(path)
    for site in _frame_open_sites(messages):
        path = _clean_path(site.get("path")) if isinstance(site.get("path"), str) else None
        if path is not None and path not in seen:
            seen.append(path)
    return tuple(seen)


def verbatim_line_runs(source: str, text: str) -> list[tuple[int, int]]:
    """Inclusive 1-based maximal runs of ``source`` lines that ``text`` carries verbatim."""
    lines = _lines(source)
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        first = lines[index]
        if not first.strip():
            index += 1
            continue
        positions: list[int] = []
        cursor = text.find(first)
        while cursor >= 0 and len(positions) <= MAX_RUN_START_OCCURRENCES:
            positions.append(cursor + len(first))
            cursor = text.find(first, cursor + 1)
        if not positions or len(positions) > MAX_RUN_START_OCCURRENCES:
            index += 1
            continue
        end = index
        while end + 1 < len(lines):
            following = lines[end + 1]
            extended = [p + len(following) for p in positions if text.startswith(following, p)]
            if not extended:
                break
            positions = extended
            end += 1
        nonblank = sum(1 for line in lines[index:end + 1] if line.strip())
        if nonblank >= MIN_RUN_NONBLANK_LINES or (index == 0 and end == len(lines) - 1):
            runs.append((index + 1, end + 1))
        index = end + 1
    return runs


Segment = tuple[int, int, str | None, str | None]
# A segment the engine cannot attribute to a current delivery: masked for every file.
_UNTRUSTED: tuple[str | None, str | None] = ("", "")


def _read_call_ids(messages: Sequence[Mapping[str, object]]) -> frozenset[str]:
    """Ids of assistant tool calls that invoke the read tool; only their results carry receipts."""
    ids: set[str] = set()
    for message in messages:
        calls = message.get("tool_calls") if isinstance(message, Mapping) else None
        for call in calls if isinstance(calls, list) else ():
            function = call.get("function") if isinstance(call, Mapping) else None
            name = function.get("name") if isinstance(function, Mapping) else None
            if name == "read" and isinstance(call.get("id"), str):
                ids.add(call["id"])
    return frozenset(ids)


def _in_frame_window_header(content: str, position: int) -> bool:
    """True when ``position`` sits in the engine's header part of a frame-rendered window.

    The engine renders a window as header, model marker, projection, receipt, content marker,
    body, end marker. A receipt-looking line in the body is file text: it comes after the
    content marker with no model marker between.
    """
    last_model = content.rfind(_MODEL_WINDOW_MARKER, 0, position)
    if last_model < 0:
        return False
    closed = max(content.rfind(_WINDOW_CONTENT_MARKER, 0, position),
                 content.rfind(_WINDOW_END_MARKER, 0, position))
    return last_model > closed


def _message_segments(content: str, offset: int, *, read_result: bool) -> list[Segment]:
    """Windows of one message, in absolute text positions (``offset`` + local index).

    A window opens at a source header or a read page line. A receipt is trusted only where the
    engine writes one: at the start of a read tool result, or in a frame window's header part.
    Any other read page line is untrusted and masks its window: file or model text can quote a
    receipt, and a quoted one must not vouch for the bytes after it.
    """
    starts = sorted({match.start() for pattern in (_SOURCE_HEADER_RE, _READ_PAGE_RE)
                     for match in pattern.finditer(content)})
    segments: list[Segment] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(content)
        path: str | None = None
        revision: str | None = None
        if content.startswith("[read page]", start):
            receipt = _RECEIPT_RE.match(content, start)
            trusted = receipt is not None and (
                (read_result and start == 0) or _in_frame_window_header(content, start))
            if trusted and receipt is not None:
                path, revision = _clean_path(receipt.group("path")), receipt.group("revision")
            else:
                path, revision = _UNTRUSTED
        segments.append((offset + start, offset + end, path, revision))
    return segments


def _delivery_text_and_segments(messages: Sequence[Mapping[str, object]]) -> tuple[str, list[Segment]]:
    """The delivery text (same join as ``_delivery_text``) and its trusted/untrusted windows."""
    read_ids = _read_call_ids(messages)
    parts: list[str] = []
    segments: list[Segment] = []
    offset = 0
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in _DELIVERY_ROLES:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if parts:
            offset += 1                      # the "\n" join between delivery messages
        read_result = message.get("role") == "tool" and message.get("tool_call_id") in read_ids
        segments.extend(_message_segments(content, offset, read_result=read_result))
        parts.append(content)
        offset += len(content)
    return "\n".join(parts), segments


def _current_delivery_pieces(text: str, segments: Sequence[Segment], path: str, revision: str) -> list[str]:
    """The maximal kept pieces of ``text`` for one file at one revision, in text order.

    A window governed by a read receipt is dropped unless the receipt names this path at
    ``revision``: text the model saw at an older revision never mints a handle at the
    current one. Receipt-less windows are kept. Pieces are split only where a dropped window
    sits, so with nothing dropped the single piece is the whole text, as before.
    """
    pieces: list[str] = []
    cursor = 0
    for start, end, receipt_path, receipt_revision in segments:
        if receipt_revision is None or (receipt_path == path and receipt_revision == revision):
            continue
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return pieces


def request_spans(messages: Sequence[Mapping[str, object]],
                  read_source: Callable[[str], "str | None"]) -> tuple[DeliveredSpan, ...]:
    """Delivered spans of one request from both providers, unioned into maximal runs per file.

    ``read_source(path)`` returns the current full text of a workspace-relative path, or
    None when it is unreadable, missing, binary or too large. Nothing here writes.
    """
    text, segments = _delivery_text_and_segments(messages)
    sites = _frame_open_sites(messages)
    spans: list[DeliveredSpan] = []
    for path in candidate_paths(messages)[:MAX_MAP_FILES]:
        source = read_source(path)
        if not isinstance(source, str) or not source:
            continue
        try:
            if len(source.encode("utf-8")) > MAX_MAP_FILE_BYTES:
                continue
        except UnicodeError:
            continue                         # undecodable bytes cannot be addressed by text
        revision = source_revision(source)
        covered: list[tuple[int, int]] = [
            run for piece in _current_delivery_pieces(text, segments, path, revision)
            for run in verbatim_line_runs(source, piece)]
        for site in sites:
            if _clean_path(str(site.get("path", ""))) != path:
                continue
            try:
                row = {**site, "path": path}
                covered.extend((span.start_line, span.end_line)
                               for span in spans_from_open_sites([row], {path: source}))
            except ValueError:
                continue                     # stale or malformed row: its bytes are not current
        total = len(_lines(source))
        for start, end in _merge_ranges(covered):
            kind = SpanKind.REWRITE if (start, end) == (1, total) else SpanKind.SPAN
            spans.append(DeliveredSpan.from_source(path, source, start_line=start,
                                                   end_line=end, kind=kind))
    return tuple(spans)


# A rendered map header at a line start; file text merely quoting the prefix is not a map.
_MAP_HEADER_RE = re.compile(r"(?:^|\n)" + re.escape(CURRENT_SOURCE_PREFIX) + r"\nrequest q\d+: ")


def current_turn_has_map(messages: Sequence[Mapping[str, object]]) -> bool:
    """True when a message after the last assistant turn already carries a map.

    That is a repair request: its map and handles are this request's, and a second map
    would contradict it. Maps in older turns are history and do not count.
    """
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "assistant":
            return False
        content = message.get("content")
        if isinstance(content, str) and _MAP_HEADER_RE.search(content):
            return True
    return False


def compose_request_map(messages: Sequence[Mapping[str, object]], *, request_ordinal: int,
                        read_source: Callable[[str], "str | None"],
                        ) -> tuple[list[dict[str, object]], HandleMap | None]:
    """Mint this request's handles and place the map at the end of the final user message.

    Returns (messages, map). Earlier messages are returned as the same objects, so the
    retained prefix of request N and N+1 is untouched; the caller sends the result and
    keeps its own transcript map-free. When this turn already carries a map (a repair
    request) or nothing was delivered, the messages come back unchanged with
    map None. A final message that is not user text gets a NEW user message after it.
    """
    messages = list(messages)
    if not messages or current_turn_has_map(messages):
        return messages, None
    final = messages[-1]
    final_content = final.get("content") if isinstance(final, Mapping) else None
    spans = request_spans(messages, read_source)
    if not spans:
        return messages, None
    handle_map = mint_handles(request_ordinal, spans)
    if final.get("role") == "user" and isinstance(final_content, str):
        return append_map_to_final_user_message(messages, handle_map), handle_map
    return [*messages, {"role": "user", "content": handle_map.render()}], handle_map


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
