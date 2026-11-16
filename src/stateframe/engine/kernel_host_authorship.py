"""Out-of-band record of the request rows and spans the host authored in one step.

Host markers are plain text, so a file, a tool result or a user message can
carry the same bytes as a host row. This module records authorship where the
host writes a row and lets readers ask for it instead of inferring it from text.

* One ``HostAuthorship`` ledger per step, owned by the runtime and shared by
  reference with every capture of that step. Nothing is added to message dicts.
* A row is keyed by role and content hash. Each record carries its data spans
  (file bytes, tool output, captured user text, untrusted values in panels);
  host text is the complement.
* Typed host blocks are located by exact registered value inside host spans.
* Unknown row, stale record or no ledger all mean "data".

Switch: STATEFRAME_HOST_ROW_IDENTITY = off (default) | shadow | on. OFF builds
no ledger object; every write call then only does its string work.
Stdlib only; no engine import at module level and no ContextVar.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, replace

MODE_ENV = "STATEFRAME_HOST_ROW_IDENTITY"
MODES = ("off", "shadow", "on")
SLOT = "_host_authorship"
ROW_KINDS = frozenset({
    "state", "task", "panel", "plain_source", "model_window", "raw_window", "receipt",
    "captured_user", "hook_output", "owner_block", "tool_result",
})
BLOCK_KINDS = frozenset({
    "evidence_pin", "continuation_limits", "terminal_line", "authority_overlay",
    "stable_wire_order",
})
# Kinds whose host text surrounds untrusted values; they need explicit data spans.
_BUILDER_KINDS = frozenset({"panel", "owner_block"})
# Typed owner-block registry entries may keep the values they replaced under this key.
RETIRED_KEY = "_retired_values"
OWNER_BLOCK_REGISTRY = "_completion_owner_owner_blocks"
ANY_ROLE = "*"

Span = tuple[int, int, str]


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    return value if value in MODES else "off"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class RowRecord:
    kind: str
    role: str
    content_sha256: str
    data: tuple[Span, ...]
    step: int
    request_id: str | None
    site: str
    count: int = 1


@dataclass(frozen=True)
class BlockRecord:
    kind: str
    text: str
    sha256: str
    current: bool
    step: int
    request_id: str | None


class HostText:
    """Builds host text while recording which pieces are untrusted values."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._spans: list[tuple[int, int]] = []
        self._length = 0

    def host(self, text: str) -> "HostText":
        self._parts.append(text)
        self._length += len(text)
        return self

    def data(self, text: str) -> "HostText":
        if text:
            self._spans.append((self._length, len(text)))
        self._parts.append(text)
        self._length += len(text)
        return self

    @property
    def text(self) -> str:
        return "".join(self._parts)

    @property
    def data_spans(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._spans)

    def __str__(self) -> str:
        return self.text


class HostAuthorship:
    """Rows and typed blocks the host authored in one step."""

    def __init__(self, step: int, request_id: str | None = None, *, private: bool = False) -> None:
        self.step = step
        self.request_id = request_id
        self.private = private
        self.rows: dict[tuple[str, str], RowRecord] = {}
        self.blocks: dict[str, list[BlockRecord]] = {}
        self.marks: list[tuple[str, frozenset[tuple[str, str]]]] = []

    # A snapshot never contains a ledger: a restored runtime and its captures
    # hold None until the next step opens one.
    def __reduce__(self):
        return (type(None), ())

    def __deepcopy__(self, memo: dict) -> "HostAuthorship":
        return self

    def __copy__(self) -> "HostAuthorship":
        return self

    # -- records ---------------------------------------------------------
    def _spans(self, content: str, data: Iterable[Sequence[object]]) -> tuple[Span, ...]:
        spans: list[Span] = []
        for item in sorted(data, key=lambda value: (int(value[0]), int(value[1]))):
            start, length = int(item[0]), int(item[1])
            if length <= 0:
                continue
            if start < 0 or start + length > len(content) or (spans and start < spans[-1][0] + spans[-1][1]):
                # A span the content cannot hold says nothing reliable: whole row is data.
                return ((0, len(content), _sha(content)),) if content else ()
            digest = str(item[2]) if len(item) > 2 else _sha(content[start:start + length])
            spans.append((start, length, digest))
        return tuple(spans)

    def _put(self, role: str, content: str, *, kind: str, site: str,
             data: Iterable[Sequence[object]]) -> RowRecord:
        if kind not in ROW_KINDS:
            raise ValueError(f"unknown host row kind: {kind}")
        key = (role, _sha(content))
        spans = self._spans(content, data)
        existing = self.rows.get(key)
        count = existing.count + 1 if existing is not None else 1
        record = RowRecord(kind, role, key[1], spans, self.step, self.request_id, site, count)
        self.rows[key] = record
        return record

    def _record(self, role: object, content: object) -> RowRecord | None:
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        digest = _sha(content)
        return self.rows.get((role, digest)) or self.rows.get((ANY_ROLE, digest))

    def _forget(self, record: RowRecord | None) -> None:
        """One authored copy of a row was mutated into something else."""
        if record is None:
            return
        key = (record.role, record.content_sha256)
        if self.rows.get(key) is not record:
            return
        if record.count > 1:
            self.rows[key] = replace(record, count=record.count - 1)
        else:
            del self.rows[key]


def ledger_of(owner: object) -> HostAuthorship | None:
    """The ledger a runtime or capture holds, or None."""
    value = getattr(owner, SLOT, None)
    return value if isinstance(value, HostAuthorship) else None


def _seed_owner_blocks(ledger: HostAuthorship, runtime: object) -> None:
    registry = getattr(runtime, OWNER_BLOCK_REGISTRY, None)
    if not isinstance(registry, Mapping):
        return
    for kind, entry in registry.items():
        if kind not in BLOCK_KINDS or not isinstance(entry, Mapping):
            continue
        value = entry.get("_value")
        retired = entry.get(RETIRED_KEY, ())
        if isinstance(value, str) and value:
            author_block(ledger, kind, value,
                         retired=[item for item in retired if isinstance(item, str)]
                         if isinstance(retired, (list, tuple)) else ())


def open_step(runtime: object, step: int) -> HostAuthorship | None:
    """Open the runtime's ledger for a step; re-entrant within the step.

    OFF constructs nothing. A nested or repeated preparation of the same step
    gets the object already open; a new step replaces it.
    """
    if mode() == "off":
        if getattr(runtime, SLOT, None) is not None:
            setattr(runtime, SLOT, None)
        return None
    current = ledger_of(runtime)
    if current is not None and current.step == step and not current.private:
        return current
    ledger = HostAuthorship(step)
    _seed_owner_blocks(ledger, runtime)
    setattr(runtime, SLOT, ledger)
    return ledger


def open_private(step: int, request_id: str | None = None) -> HostAuthorship | None:
    """A ledger for an auxiliary capture, never shared with the parent step."""
    if mode() == "off":
        return None
    return HostAuthorship(step, request_id, private=True)


def bind(capture: object, ledger: HostAuthorship | None) -> HostAuthorship | None:
    """Give a capture its reference. A None ledger leaves the capture untouched."""
    if ledger is None:
        return None
    setattr(capture, SLOT, ledger)
    request_id = getattr(capture, "request_id", None)
    if isinstance(request_id, str) and (ledger.request_id is None or not ledger.private):
        ledger.request_id = request_id
    return ledger


# -- write API ------------------------------------------------------------

def builder(ledger: HostAuthorship | None = None) -> HostText:
    del ledger
    return HostText()


def _content_and_data(content: object, data: Iterable[Sequence[object]],
                      kind: str) -> tuple[str, tuple[Sequence[object], ...]]:
    if isinstance(content, HostText):
        return content.text, tuple(content.data_spans)
    text = str(content)
    spans = tuple(data)
    if kind in _BUILDER_KINDS and not spans and text:
        # Not built span by span: nothing in it can be told apart from a value.
        spans = ((0, len(text)),)
    return text, spans


def author_row(ledger: HostAuthorship | None, role: str, content: object, *, kind: str,
               site: str, data: Iterable[Sequence[object]] = ()) -> dict[str, object]:
    """Return exactly ``{"role": role, "content": content}`` and record who wrote it."""
    if ledger is None:
        return {"role": role, "content": content.text if isinstance(content, HostText) else content}
    text, spans = _content_and_data(content, data, kind)
    ledger._put(role, text, kind=kind, site=site, data=spans)
    return {"role": role, "content": text}


def author_text(ledger: HostAuthorship | None, text: object, *, kind: str, site: str,
                data: Iterable[Sequence[object]] = ()) -> str:
    """Record host text before its row exists; a row whose whole content equals it resolves."""
    if ledger is None:
        return text.text if isinstance(text, HostText) else text
    value, spans = _content_and_data(text, data, kind)
    ledger._put(ANY_ROLE, value, kind=kind, site=site, data=spans)
    return value


def author_block(ledger: HostAuthorship | None, kind: str, text: str, *,
                 retired: Iterable[str] = ()) -> str:
    """Register the exact value of a typed host block; ``retired`` are values it replaced."""
    if ledger is None:
        return text
    if kind not in BLOCK_KINDS:
        raise ValueError(f"unknown host block kind: {kind}")
    records = ledger.blocks.setdefault(kind, [])
    known = {record.text: index for index, record in enumerate(records)}
    for value, current in (*((item, False) for item in retired), (text, True)):
        if not isinstance(value, str) or not value:
            continue
        record = BlockRecord(kind, value, _sha(value), current, ledger.step, ledger.request_id)
        if value in known:
            if current:
                records[known[value]] = record
        else:
            known[value] = len(records)
            records.append(record)
    if text:
        for index, record in enumerate(records):
            if record.current and record.text != text:
                records[index] = replace(record, current=False)
    return text


def _shift(spans: Iterable[Span], offset: int) -> tuple[Span, ...]:
    return tuple((start + offset, length, digest) for start, length, digest in spans)


def append_host(ledger: HostAuthorship | None, row: MutableMapping[str, object], text: str, *,
                sep: str = "\n\n", site: str, unknown_as: str) -> None:
    """``row["content"] = content + sep + text``; the appended text is host text.

    An unregistered target becomes ``unknown_as`` with its whole prior content
    recorded as one data span. Data spans of a registered ``text`` are kept.
    """
    prior = row.get("content")
    prior = prior if isinstance(prior, str) else ""
    content = prior + sep + text
    row["content"] = content
    if ledger is None:
        return
    role = str(row.get("role", ""))
    record = ledger._record(role, prior)
    if record is None:
        if unknown_as not in ROW_KINDS:
            raise ValueError(f"unknown host row kind: {unknown_as}")
        kind = unknown_as
        covered = len(prior) + (len(sep) if kind == "tool_result" else 0)
        spans: tuple[Span, ...] = ((0, covered, _sha(content[:covered])),) if covered else ()
    else:
        kind, spans = record.kind, record.data
        ledger._forget(record)
    suffix = ledger._record(ANY_ROLE, text)
    if suffix is not None:
        spans = (*spans, *_shift(suffix.data, len(prior) + len(sep)))
    ledger._put(role, content, kind=kind, site=site, data=spans)


def remove_host(ledger: HostAuthorship | None, row: MutableMapping[str, object],
                start: int, end: int) -> None:
    """Remove a host span and shift the data spans behind it."""
    prior = row.get("content")
    prior = prior if isinstance(prior, str) else ""
    start, end = max(0, start), min(len(prior), end)
    if end <= start:
        return
    content = prior[:start] + prior[end:]
    row["content"] = content
    if ledger is None:
        return
    role = str(row.get("role", ""))
    record = ledger._record(role, prior)
    if record is None:
        return
    ledger._forget(record)
    spans: list[Span] = []
    for span_start, length, digest in record.data:
        if span_start + length <= start:
            spans.append((span_start, length, digest))
        elif span_start >= end:
            spans.append((span_start - (end - start), length, digest))
        else:
            # The cut touched recorded data: the row is no longer what was authored.
            return
    ledger._put(role, content, kind=record.kind, site=record.site, data=spans)


def rewrite_row(ledger: HostAuthorship | None, row: MutableMapping[str, object], new_content: str,
                *, data: Iterable[Sequence[object]], site: str) -> None:
    """A compaction pass replaced a registered row's content; ``data`` is required."""
    prior = row.get("content")
    row["content"] = new_content
    if ledger is None:
        return
    role = str(row.get("role", ""))
    record = ledger._record(role, prior)
    if record is None:
        return
    ledger._forget(record)
    ledger._put(role, new_content, kind=record.kind, site=site, data=tuple(data))


def author_concat(ledger: HostAuthorship | None, role: str, parts: Iterable[object], *,
                  kind: str, site: str) -> dict[str, object]:
    """Build a row from registered texts and ``(text, "data")`` / ``(text, "host")`` parts.

    A bare string that was never registered is recorded as data.
    """
    pieces: list[str] = []
    spans: list[Span] = []
    offset = 0
    for part in parts:
        label = None
        if isinstance(part, tuple):
            part, label = part
        text = str(part)
        if ledger is not None and text:
            record = None if label is not None else (
                ledger._record(role, text) or ledger._record(ANY_ROLE, text))
            if label == "host":
                pass
            elif record is not None:
                spans.extend(_shift(record.data, offset))
            else:
                spans.append((offset, len(text), _sha(text)))
        pieces.append(text)
        offset += len(text)
    content = "".join(pieces)
    if ledger is not None:
        ledger._put(role, content, kind=kind, site=site, data=spans)
    return {"role": role, "content": content}


# -- read API -------------------------------------------------------------

def _resolved(ledger: HostAuthorship | None, message: object, occurrence: int = 0) -> RowRecord | None:
    if ledger is None or not isinstance(message, Mapping):
        return None
    role = message.get("role")
    if role == "assistant":
        return None
    record = ledger._record(role, message.get("content"))
    if record is None or record.step != ledger.step or occurrence >= record.count:
        return None
    if role == "tool" and record.kind != "tool_result":
        return None
    return record


def row_kind(ledger: HostAuthorship | None, message: object, *, occurrence: int = 0) -> str | None:
    """The kind the host authored this row as, or None: the row is data."""
    record = _resolved(ledger, message, occurrence)
    return record.kind if record is not None else None


def row_site(ledger: HostAuthorship | None, message: object) -> str | None:
    record = _resolved(ledger, message)
    return record.site if record is not None else None


def occurrences(ledger: HostAuthorship | None, messages: Sequence[object], *,
                capture: object = None) -> list[int]:
    """Per row, which authored copy it is: host-owned interval first, then wire order."""
    result = [0] * len(messages)
    if ledger is None:
        return result
    order = list(range(len(messages)))
    owned = {index for index in order if structural_class(messages, index, capture=capture)
             in {"host_panel", "frame", "preface"}} if messages else set()
    order.sort(key=lambda index: (index not in owned, index))
    seen: dict[tuple[object, str], int] = {}
    for index in order:
        message = messages[index]
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            continue
        key = (message.get("role"), _sha(message["content"]))
        result[index] = seen.get(key, 0)
        seen[key] = result[index] + 1
    return result


def row_kinds(ledger: HostAuthorship | None, messages: Sequence[object], *,
              capture: object = None) -> list[str | None]:
    """``row_kind`` for every row, with copies past the authored count resolved as data."""
    order = occurrences(ledger, messages, capture=capture)
    return [row_kind(ledger, message, occurrence=order[index])
            for index, message in enumerate(messages)]


def _data_spans(record: RowRecord, content: str) -> list[tuple[int, int]]:
    for start, length, digest in record.data:
        if _sha(content[start:start + length]) != digest:
            # Stale record: kind stays known, everything after the first line is data.
            first = content.find("\n")
            first = len(content) if first < 0 else first + 1
            return [(first, len(content))] if first < len(content) else []
    return [(start, start + length) for start, length, _digest in record.data]


def data_spans(ledger: HostAuthorship | None, message: object, *,
               occurrence: int = 0) -> list[tuple[int, int]]:
    """Recorded data spans as (start, end); the whole content for an unknown row."""
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str):
        return []
    record = _resolved(ledger, message, occurrence)
    if record is None:
        return [(0, len(content))] if content else []
    return _data_spans(record, content)


def host_spans(ledger: HostAuthorship | None, message: object, *,
               occurrence: int = 0) -> list[tuple[int, int]]:
    """The complement of the data spans; empty for an unknown row."""
    record = _resolved(ledger, message, occurrence)
    if record is None:
        return []
    content = message["content"]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in _data_spans(record, content):
        if start > cursor:
            spans.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(content):
        spans.append((cursor, len(content)))
    return spans


def host_text(ledger: HostAuthorship | None, message: object, *, occurrence: int = 0) -> str:
    """Host spans joined by a newline; empty for an unknown row."""
    spans = host_spans(ledger, message, occurrence=occurrence)
    if not spans:
        return ""
    content = message["content"]
    return "\n".join(content[start:end] for start, end in spans)


def find_block(ledger: HostAuthorship | None, message: object, kind: str, *,
               occurrence: int = 0) -> list[tuple[int, int, bool]]:
    """Exact registered values of ``kind`` inside host spans, as (start, end, current).

    A hit inside a data span, or inside the occurrence of a longer registered
    block, is not a hit. There is no marker scan.
    """
    spans = host_spans(ledger, message, occurrence=occurrence)
    if not spans or ledger is None:
        return []
    content = message["content"]
    hits: list[tuple[int, int, str, bool]] = []
    for block_kind, records in ledger.blocks.items():
        for record in records:
            for span_start, span_end in spans:
                at = content.find(record.text, span_start, span_end)
                while at >= 0:
                    hits.append((at, at + len(record.text), block_kind, record.current))
                    at = content.find(record.text, at + 1, span_end)
    found = []
    for start, end, block_kind, current in hits:
        if block_kind != kind:
            continue
        if any(other_start <= start and end <= other_end
               and other_end - other_start > end - start
               for other_start, other_end, _kind, _current in hits):
            continue
        found.append((start, end, current))
    return sorted(set(found))


def structural_class(messages: Sequence[object], index: int, *,
                     capture: object = None) -> str | None:
    """Positional row class from the stable-wire ownership rules, or None.

    None means "no structural answer", never "host".
    """
    from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
    from kernel_stable_wire_ownership import request_frame_index, request_ownership
    try:
        frame_index = request_frame_index(messages, capture=capture)
        ownership = request_ownership(messages, frame_index, capture=capture)
    except (ValueError, FrameAllocationReplanNeeded):
        return None
    if not 0 <= index < len(messages):
        return None
    if index >= ownership.exchange_start:
        return "native_exchange"
    if index == ownership.frame_index:
        return "frame"
    if ownership.newest_start <= index < ownership.newest_end:
        return "captured_user"
    if index < ownership.prefix_end:
        return "preface"
    return "host_panel"


def note_missing(journal: object, reader: str, *, site: str = "",
                 request_id: str | None = None) -> None:
    """ON with no ledger in reach: say so once per request and reader."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    seen = getattr(journal, "_host_identity_missing_seen", None)
    if not isinstance(seen, set):
        seen = set()
        try:
            journal._host_identity_missing_seen = seen
        except AttributeError:
            pass
    key = (request_id, reader)
    if key in seen:
        return
    seen.add(key)
    emit("host_identity_missing", reader=reader, site=site, request_id=request_id, dispatch=False)
