"""Typed source-view contract for verifying source carried by a frozen request.

``complete`` means the current-source map in the request covers lines 1..N at the
current source sha256, and every map span has a byte-verified payload somewhere in
the final frozen request. Stale-revision windows, receipts, and history never
count; overlapping spans cannot fill holes.

Two payload carriers are recognised: raw ``[Current|Working source: P]`` blocks
and model/legacy source windows (``[Model source window]`` / ``[Source window]``).
Every segment is compared byte-for-byte with the current source lines, so a
page whose text was edited, or whose revision is stale, contributes nothing.

The current-source map is an index, never payload::

    [Current source map]
    - <path> source=sha256:<64 hex> spans=1-1000,1001-2000
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

SOURCE_MAP_HEADER = "[Current source map]"
_MAP_LINE = re.compile(
    r"^- (?P<path>\S.*?) source=sha256:(?P<sha>[0-9a-f]{64}) spans=(?P<spans>-|\d+-\d+(?:,\d+-\d+)*)$")
_DEFERRED_HEADER = "[Deferred required source paths]"
_RAW_HEADER = re.compile(r"(?m)^\[(?:Current|Working) source: (.+)\]$")


@dataclass(frozen=True)
class VerifiedSpan:
    """A 1-based inclusive full-line span verified against source bytes."""

    start: int
    end: int
    sha256: str


@dataclass(frozen=True)
class SourceView:
    """Current source metadata and the verified line spans in a frozen request."""

    path: str
    source_sha256: str
    total_lines: int
    eof_newline: bool
    spans: tuple[VerifiedSpan, ...]
    complete: bool


class SourceIncomplete(Exception):
    """The frozen request does not prove complete current source coverage."""

    def __init__(self, path: str, missing_spans: tuple[tuple[int, int], ...]) -> None:
        self.path = path
        self.missing_spans = missing_spans
        super().__init__(f"incomplete source view for {path}: {missing_spans}")


# --------------------------------------------------------------------------- map

def _merge(spans: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Union of inclusive line spans; overlapping and adjacent spans coalesce."""
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _subtract(whole: tuple[int, int], covered: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Spans of ``whole`` that ``covered`` leaves uncovered."""
    gaps: list[tuple[int, int]] = []
    cursor = whole[0]
    for start, end in _merge(covered):
        if end < cursor or start > whole[1]:
            continue
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= whole[1]:
        gaps.append((cursor, whole[1]))
    return tuple(gaps)


def render_source_map(entries: Mapping[str, tuple[str, Iterable[tuple[int, int]]]]) -> str | None:
    """Deterministic map block: sorted paths, merged spans, ``None`` when empty.

    ``entries`` maps path -> (``sha256:<hex>`` revision, delivered spans).
    """
    lines: list[str] = []
    for path in sorted(entries):
        revision, spans = entries[path]
        if "\n" in path or "\r" in path or not re.fullmatch(r"sha256:[0-9a-f]{64}", revision):
            continue
        merged = _merge(spans)
        lines.append(f"- {path} source={revision} spans="
                     + (",".join(f"{a}-{b}" for a, b in merged) if merged else "-"))
    return SOURCE_MAP_HEADER + "\n" + "\n".join(lines) if lines else None


def parse_source_map(content: object) -> dict[str, tuple[str, tuple[tuple[int, int], ...]]] | None:
    """Parse one map block into path -> (revision, spans); ``None`` if not a map."""
    if not isinstance(content, str) or not content.startswith(SOURCE_MAP_HEADER + "\n"):
        return None
    entries: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {}
    for line in content.split("\n")[1:]:
        match = _MAP_LINE.match(line)
        if match is None:
            continue
        spans: tuple[tuple[int, int], ...] = ()
        if match["spans"] != "-":
            parsed = [tuple(int(n) for n in item.split("-")) for item in match["spans"].split(",")]
            if any(a < 1 or b < a for a, b in parsed):
                continue
            spans = _merge(parsed)  # type: ignore[arg-type]
        entries[match["path"]] = ("sha256:" + match["sha"], spans)
    return entries


# ----------------------------------------------------------------------- payload

def source_lines(source: bytes) -> list[bytes]:
    """Line bytes by the engine's line grammar (``str.splitlines``).

    The window renderers number lines this way too, so edit admission must use this
    helper: a ``\n``-only split disagrees on files with form feeds, ``\x85`` and similar.
    """
    return _source_lines(source)


def _source_lines(source: bytes) -> list[bytes]:
    """Line bytes by the engine's line grammar (``str.splitlines``)."""
    text = source.decode("utf-8", errors="surrogateescape")
    return [line.encode("utf-8", errors="surrogateescape")
            for line in text.splitlines(keepends=True)]


def _eof_newline(lines: Sequence[bytes]) -> bool:
    if not lines:
        return False
    text = lines[-1].decode("utf-8", errors="surrogateescape")
    return text.splitlines()[0] != text


def window_claims(content: object, *, path: str, revision: str,
                  total_lines: int) -> list[tuple[int, int, str]]:
    """Claimed (start, end, text) segments of a current-revision window for ``path``.

    Nothing is compared with source here; a window for another path, another
    revision, or a different total length yields no claims.
    """
    if not isinstance(content, str):
        return []
    if __package__:
        from .kernel_request_source_window import (
            _model_window_segments, _window_segments, parse_model_source_window, parse_window,
        )
    else:
        from kernel_request_source_window import (
            _model_window_segments, _window_segments, parse_model_source_window, parse_window,
        )
    metadata = parse_model_source_window(content)
    segments_of = _model_window_segments
    if metadata is None:
        metadata = parse_window(content)
        segments_of = _window_segments
    if (not isinstance(metadata, Mapping)
            or str(metadata.get("path", "")).replace("\\", "/") != path.replace("\\", "/")
            or metadata.get("revision") != revision
            or metadata.get("sha256") not in (None, revision.removeprefix("sha256:"))
            or metadata.get("total_lines") != total_lines
            or not isinstance(metadata.get("ranges"), list)):
        return []
    claims: list[tuple[int, int, str]] = []
    for row, text in segments_of(content, metadata):
        start, end = row.get("start_line"), row.get("end_line")
        trailing = row.get("trailing_newline")
        if (type(start) is not int or type(end) is not int or not 1 <= start <= end
                or type(trailing) is not bool):
            continue
        if not trailing and text.endswith("\n"):
            text = text[:-1]  # framing newline added after an unterminated last line
        elif trailing and not text.endswith("\n"):
            continue
        claims.append((start, end, text))
    return claims


_READ_PAGE_HEADER = re.compile(
    r"\[read page\] path=(?P<path>\S+) revision=(?P<revision>sha256:[0-9a-f]{64}) "
    r"lines=(?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+)[^\n]*\n[^\n]*\n")


def _read_page_claim(content: object, *, path: str, revision: str,
                     total_lines: int) -> tuple[int, int, str] | None:
    """(start, end, body) of a host read page for ``path`` at ``revision``; None otherwise.

    A subframe's own read result is the only copy of a file once the host retires that
    file's source window, so it is source evidence. Nothing is compared with source here.
    """
    if not isinstance(content, str) or not content.startswith("[read page] "):
        return None
    match = _READ_PAGE_HEADER.match(content)
    if (match is None
            or match.group("path").replace("\\", "/") != path.replace("\\", "/")
            or match.group("revision") != revision
            or int(match.group("total")) != total_lines):
        return None
    start, end = int(match.group("start")), int(match.group("end"))
    if not 1 <= start <= end <= total_lines:
        return None
    return start, end, content[match.end():]


def page_carries_span(page: object, *, path: str, revision: str, first: int, last: int,
                      payload: str) -> bool:
    """True when ``page`` is a window whose exact text for ``first..last`` is ``payload``.

    The allocator uses this before listing a span in the current-source map, so the
    map never claims a span the verifier could not byte-verify from the same page.
    """
    if __package__:
        from .kernel_request_source_window import parse_model_source_window, parse_window
    else:
        from kernel_request_source_window import parse_model_source_window, parse_window
    metadata = parse_model_source_window(page) or parse_window(page)
    total = metadata.get("total_lines") if isinstance(metadata, Mapping) else None
    if type(total) is not int:
        return False
    return (first, last, payload) in window_claims(page, path=path, revision=revision,
                                                   total_lines=total)


def _raw_block_is_full(content: str, *, path: str, source: bytes) -> bool:
    """True when a raw ``[Current|Working source: P]`` block carries ALL source bytes."""
    if not source:
        return False
    text = source.decode("utf-8", errors="surrogateescape")
    for match in _RAW_HEADER.finditer(content):
        if (match.group(1).replace("\\", "/") == path.replace("\\", "/")
                and content[match.end():].startswith("\n" + text)):
            return True
    return False


def _texts(messages: Sequence[Mapping[str, object]]) -> Iterable[str]:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            yield content
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    yield part["text"]


def _map_entry(messages: Sequence[Mapping[str, object]], path: str,
               ) -> tuple[str, tuple[tuple[int, int], ...]] | None:
    """The path's entry in the LAST map block of the request (that block is authoritative)."""
    entry = None
    for content in _texts(messages):
        parsed = parse_source_map(content)
        if parsed is not None:
            entry = parsed.get(path.replace("\\", "/"))
    return entry


def source_view(final_messages: Sequence[Mapping[str, object]], *, path: str,
                source: bytes) -> SourceView:
    """Build a byte-verified view from the final frozen request messages."""
    digest = "sha256:" + hashlib.sha256(source).hexdigest()
    lines = _source_lines(source)
    total = len(lines)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    verified: list[tuple[int, int]] = []
    full_block = False
    for content in _texts(final_messages):
        if parse_source_map(content) is not None:
            continue
        if _raw_block_is_full(content, path=path, source=source):
            verified.append((1, total))
            full_block = True
            continue
        own: list[tuple[int, int]] = []
        for start, end, text in window_claims(content, path=path, revision=digest,
                                              total_lines=total):
            if end <= total and (text.encode("utf-8", errors="surrogateescape")
                                 == source[offsets[start - 1]:offsets[end]]):
                own.append((start, end))
        page = _read_page_claim(content, path=path, revision=digest, total_lines=total)
        if page is not None:
            start, end, body = page
            expected = source[offsets[start - 1]:offsets[end]]
            # The page body is the exact line bytes, optionally followed by a host notice.
            if body.encode("utf-8", errors="surrogateescape").startswith(expected):
                own.append((start, end))
        verified.extend(own)
        if total and _merge(own) == ((1, total),):
            full_block = True  # one block carrying every line: the allocator-OFF shape
    merged = _merge(verified)
    spans = tuple(
        VerifiedSpan(a, b, "sha256:" + hashlib.sha256(source[offsets[a - 1]:offsets[b]]).hexdigest())
        for a, b in merged)
    entry = _map_entry(final_messages, path)
    map_complete = False
    if entry is not None and entry[0] == digest:
        claimed = entry[1]
        map_complete = total == 0 or (
            claimed == ((1, total),) and not any(_subtract(span, merged) for span in claimed))
    return SourceView(path=path, source_sha256=digest, total_lines=total,
                      eof_newline=_eof_newline(lines), spans=spans,
                      complete=bool(full_block or map_complete))


def _missing_spans(view: SourceView, final_messages: Sequence[Mapping[str, object]],
                   ) -> tuple[tuple[int, int], ...]:
    """Lines not proven: map-claimed AND payload-verified lines count as proven."""
    if view.total_lines == 0:
        return ()
    entry = _map_entry(final_messages, view.path)
    claimed = entry[1] if entry is not None and entry[0] == view.source_sha256 else ()
    proven = [(max(span.start, a), min(span.end, b)) for span in view.spans for a, b in claimed
              if max(span.start, a) <= min(span.end, b)]
    return _subtract((1, view.total_lines), proven)


def require_complete_source(final_messages: Sequence[Mapping[str, object]], *, path: str,
                            source: bytes, journal: object) -> SourceView:
    """Require complete current-source coverage, recording refusal evidence."""
    view = source_view(final_messages, path=path, source=source)
    if view.complete:
        return view
    missing = _missing_spans(view, final_messages)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("source_view_incomplete", producer="kernel_source_view", path=path,
             source_sha256=view.source_sha256,
             missing_spans=[list(span) for span in missing], dispatch=False)
    raise SourceIncomplete(path, missing)


# ------------------------------------------------------------------------ floor

def file_floor_messages(path: str, source: str) -> list[dict[str, str]]:
    """The cheapest request payload that proves ``path`` in full: every page plus its map line.

    Pages come from the allocator's own page planner (``page_spans``: the line ceiling
    and the character budget net of the page header reserve), so pricing these
    messages prices the FILE floor the allocator can never go below.
    """
    if __package__:
        from .kernel_working_set_windows import page_spans, render_frozen_span
    else:
        from kernel_working_set_windows import page_spans, render_frozen_span

    lines = source.splitlines(keepends=True)
    revision = "sha256:" + hashlib.sha256(source.encode("utf-8", errors="surrogateescape")).hexdigest()
    spans = page_spans(lines, 1, len(lines), path, revision) if lines else []
    messages: list[dict[str, str]] = []
    for first, last in spans:
        page, _ = render_frozen_span(
            {"path": path, "revision": revision, "total_lines": len(lines)},
            {"start_line": first, "end_line": last}, "".join(lines[first - 1:last]))
        messages.append({"role": "user", "content": page})
    block = render_source_map({path: (revision, spans)})
    if block is not None:
        messages.append({"role": "user", "content": block})
    return messages


def file_floor_bytes(path: str, source: str, *, fixed_messages: Sequence[Mapping[str, object]],
                     tools: object = None,
                     wire: Callable[[dict[str, object]], Mapping[str, object]] | None = None) -> int:
    """Serialized request bytes of the FILE floor: fixed rows, every page and the map line.

    The allocator's byte cap (``serialized_bytes``) is measured on the whole wire,
    so the floor is priced the same way: ``json.dumps`` of the body, UTF-8 encoded.
    ``wire`` maps the assembled body to the body the transport serializes (private
    keys stripped, provider transform applied); the caller supplies it so this
    module stays free of engine dependencies.
    """
    body: dict[str, object] = {"messages": [*map(dict, fixed_messages), *file_floor_messages(path, source)]}
    if tools is not None:
        body["tools"] = tools
    if wire is not None:
        body = dict(wire(body))
    return len(json.dumps(body).encode())


# -------------------------------------------------------------------- deferral

def _normalized(value: object) -> str:
    return str(value).replace("\\", "/").strip().removeprefix("./")


def deferred_read_obligation_valid(final_messages: Sequence[Mapping[str, object]], *,
                                   path: str, source: bytes, runtime: object,
                                   capture: object = None) -> bool:
    """True when the frozen request defers ``path`` under a live, current-revision obligation.

    Two independent facts are required: the FINAL request carries the host's
    ``[Deferred required source paths]`` notice naming ``path``, and the source
    plan records ``path`` as deferred at exactly the current source revision.
    The plan is the request capture's frozen plan (where the allocator records the
    deferral) or, failing that, the runtime's. A stale or missing revision fails closed.
    """
    target = _normalized(path)
    noticed = any(
        content.startswith(_DEFERRED_HEADER)
        and any(_normalized(line[2:]) == target for line in content.split("\n")
                if line.startswith("- "))
        for message in final_messages
        if isinstance(message, Mapping) and message.get("role") == "user"
        for content in _texts([message]))
    if not noticed:
        return False
    plans = [plan for plan in (
        getattr(capture, "_request_source_closure_override", None),
        getattr(runtime, "_request_source_closure", None)) if isinstance(plan, Mapping)]
    deferred: set[str] = set()
    for plan in plans:
        raw = plan.get("deferred_paths", ())
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            deferred.update(_normalized(item) for item in raw)
    if target not in deferred:
        return False
    current = "sha256:" + hashlib.sha256(source).hexdigest()
    recorded = [revisions.get(path, revisions.get(target)) for plan in plans
                if isinstance(revisions := plan.get("revisions"), Mapping)]
    recorded = [value for value in recorded if value is not None]
    return bool(recorded) and all(value == current for value in recorded)


__all__ = [
    "SOURCE_MAP_HEADER", "SourceIncomplete", "SourceView", "VerifiedSpan",
    "deferred_read_obligation_valid", "file_floor_bytes", "file_floor_messages", "page_carries_span", "parse_source_map", "render_source_map",
    "require_complete_source", "source_view", "window_claims",
]
