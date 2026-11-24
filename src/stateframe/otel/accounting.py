"""Construct record identities and source-backed spans in one place."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from .identity import derive, namespace_bytes
from .types import (
    AttrKey,
    Cursor,
    Namespace,
    SafeAttr,
    SafeEvent,
    SafeLink,
    SafeSpan,
    SourceOccurrence,
)


_MAX_EPOCH = (1 << 63) - 1
_MAX_DURATION_NS = 604_800_000_000_000


def keyed_identity(key: bytes, tag: str, parts: tuple[bytes, ...], size: int) -> bytes:
    return derive(key, tag, parts, size)


def namespace_material(space: Namespace) -> bytes:
    return namespace_bytes(space)


def record_identity(
    key: bytes, tag: str, name: str, record: SourceOccurrence, *, size: int = 8,
) -> bytes:
    return derive(key, tag, (*_record_parts(record), name.encode("ascii", "strict")), size)


def paired_identity(
    key: bytes, tag: str, name: str, first: SourceOccurrence, last: SourceOccurrence,
) -> bytes:
    if not same_record_pair(first, last):
        raise ValueError("invalid_field")
    return derive(key, tag, (*_record_parts(first), *_record_parts(last), name.encode("ascii", "strict")), 8)


def context_identity(key: bytes, tag: str, space: Namespace, extra: bytes | None = None) -> bytes:
    domain = tag if tag.endswith("-context") else "context-" + tag
    parts = (namespace_bytes(space),)
    if extra is not None:
        parts += (extra,)
    return derive(key, domain, parts, 8)


def same_record_pair(first: SourceOccurrence, last: SourceOccurrence) -> bool:
    if (type(first) is not SourceOccurrence or type(last) is not SourceOccurrence
            or type(first.cursor) is not Cursor or type(last.cursor) is not Cursor):
        return False
    first_sequence = _position(first)
    last_sequence = _position(last)
    if (first_sequence is None or last_sequence is None
            or first.space != last.space or first.family != last.family
            or first.cursor.stream != last.cursor.stream
            or first_sequence >= last_sequence):
        return False
    if not _same_domain(first.row, last.row, ("process_id", "process", "pid")):
        return False
    return _same_domain(first.row, last.row, ("clock_id", "clock", "clock_source"))


def point_span(
    key: bytes,
    name: str,
    trace_id: bytes,
    record: SourceOccurrence,
    parent_id: bytes | None,
    attrs: Iterable[SafeAttr],
    *,
    at_ns: int | None = None,
    error: bool = False,
    events: tuple[SafeEvent, ...] = (),
    links: tuple[SafeLink, ...] = (),
    context_tag: str | None = None,
    identity_space: Namespace | None = None,
    identity_extra: bytes | None = None,
) -> SafeSpan | None:
    at = record.epoch_ns if at_ns is None else at_ns
    if not _valid_epoch(at):
        return None
    span_id = (context_identity(key, context_tag, identity_space or record.space, identity_extra)
               if context_tag is not None
               else record_identity(key, "span", name, record))
    return _span(name, trace_id, span_id, parent_id, at, at, attrs, error, events, links)


def anchored_point_span(
    key: bytes,
    name: str,
    trace_id: bytes,
    identity_record: SourceOccurrence | None,
    anchor_record: SourceOccurrence,
    parent_id: bytes | None,
    attrs: Iterable[SafeAttr],
    *,
    error: bool = False,
    events: tuple[SafeEvent, ...] = (),
) -> SafeSpan | None:
    at = anchor_record.epoch_ns
    if not _valid_epoch(at) or (identity_record is not None and identity_record.space != anchor_record.space):
        return None
    parts = _record_parts(anchor_record)
    if identity_record is not None:
        parts += _record_parts(identity_record)
    span_id = derive(key, "span", (*parts, name.encode("ascii", "strict")), 8)
    return _span(name, trace_id, span_id, parent_id, at, at, attrs, error, events, ())


def paired_span(
    key: bytes,
    name: str,
    trace_id: bytes,
    first: SourceOccurrence,
    last: SourceOccurrence,
    parent_id: bytes | None,
    attrs: Iterable[SafeAttr],
    *,
    error: bool = False,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> SafeSpan | None:
    bounds = paired_bounds(first, last, start_ns=start_ns, end_ns=end_ns)
    if bounds is None:
        return None
    span_id = paired_identity(key, "span", name, first, last)
    return _span(name, trace_id, span_id, parent_id, *bounds, attrs, error, (), ())


def recorded_span(
    key: bytes,
    name: str,
    trace_id: bytes,
    record: SourceOccurrence,
    parent_id: bytes | None,
    start_ns: object,
    end_ns: object,
    attrs: Iterable[SafeAttr],
    *,
    error: bool = False,
) -> SafeSpan | None:
    if not _ordered_bounds(start_ns, end_ns):
        return None
    span_id = record_identity(key, "span", name, record)
    return _span(name, trace_id, span_id, parent_id, start_ns, end_ns, attrs, error, (), ())


def paired_bounds(
    first: SourceOccurrence,
    last: SourceOccurrence,
    *,
    start_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[int, int] | None:
    if not same_record_pair(first, last):
        return None
    start = first.epoch_ns if start_ns is None else start_ns
    end = last.epoch_ns if end_ns is None else end_ns
    return (start, end) if _ordered_bounds(start, end) else None


def seconds_bounds(start: object, end: object) -> tuple[int, int] | None:
    from .operations import epoch_seconds_ns

    first, last = epoch_seconds_ns(start), epoch_seconds_ns(end)
    return (first, last) if _ordered_bounds(first, last) else None


def event(name: str, record: SourceOccurrence, attrs: Iterable[SafeAttr]) -> SafeEvent | None:
    at = record.epoch_ns
    if not _valid_epoch(at):
        return None
    return SafeEvent(name, at, tuple(sorted(attrs, key=lambda item: item.key.value)), 0)


def private_cursor_stream(
    key: bytes, family: str, space: Namespace, path: tuple[str, ...], run_id: object,
) -> bytes:
    """Private Reader cursor key; never use this path-derived value for export identity."""
    logical = path
    if len(path) >= 2 and path[0] == "attempts" and path[1] == str(space.attempt):
        logical = path[2:]
    if logical and logical[-1].endswith(".xz"):
        logical = (*logical[:-1], logical[-1][:-3])
    source_id = run_id.encode("utf-8", "strict") if type(run_id) is str else b""
    if not 0 < len(source_id) <= 256:
        source_id = b""
    path_id = hashlib.sha256("/".join(logical).encode("utf-8", "strict")).digest()
    return derive(key, "stream", (
        namespace_bytes(space), family.encode("ascii"), path_id, source_id,
    ), 16)


def _record_parts(record: SourceOccurrence) -> tuple[bytes, ...]:
    if type(record) is not SourceOccurrence or type(record.cursor) is not Cursor:
        raise ValueError("invalid_field")
    cursor = record.cursor
    sequence = _position(record)
    if sequence is None:
        raise ValueError("invalid_field")
    content = cursor.head if type(cursor.head) is bytes and len(cursor.head) == 32 else _row_digest(record.row)
    return (
        namespace_bytes(record.space), str(record.family).encode("ascii"),
        sequence.to_bytes(8, "big"), content,
    )


def _position(record: SourceOccurrence) -> int | None:
    value = record.cursor.sequence if type(record.cursor.sequence) is int else None
    if value == 0:
        value = record.index if type(record.index) is int else None
    return value if value is not None and value > 0 else None


def _row_digest(row: dict[str, object]) -> bytes:
    try:
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("invalid_field") from error
    return hashlib.sha256(encoded).digest()


def _same_domain(first: dict[str, object], last: dict[str, object], keys: tuple[str, ...]) -> bool:
    left = next((first[name] for name in keys if name in first), None)
    right = next((last[name] for name in keys if name in last), None)
    return left == right


def _valid_epoch(value: object) -> bool:
    return type(value) is int and 1 <= value <= _MAX_EPOCH


def _ordered_bounds(start: object, end: object) -> bool:
    return (_valid_epoch(start) and _valid_epoch(end) and start <= end
            and end - start <= _MAX_DURATION_NS)


def _span(name, trace_id, span_id, parent_id, start, end, attrs, error, events, links) -> SafeSpan:
    if (type(trace_id) is not bytes or len(trace_id) != 16 or not any(trace_id)
            or type(span_id) is not bytes or len(span_id) != 8 or not any(span_id)):
        raise ValueError("invalid_field")
    return SafeSpan(name, trace_id, span_id, parent_id, start, end,
                    tuple(sorted(attrs, key=lambda item: item.key.value)), events, links, 0, 0, 0, error)
