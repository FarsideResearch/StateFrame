"""A1b read-time admission: shorten or refuse a fresh page before it is captured.

Gated by ``kernel_read_receipts.enabled()``.  OFF leaves the read path byte-identical.

With a provider envelope, admission uses the allocator's complete renderer and
minimum-completion domain. Allowance is cap minus the rendered floor (fixed and
protected minimum); pending pages are charged by re-rendering that floor with
FULL-only proposals. JSON escaping, source facts and carrier sharing are priced
by the same oracle used at preparation. Older callers without a composed body
retain the conservative envelope estimate, with serialized string costs.

A refusal closes the current batch. All subsequent reads return the existing
continuation cursor grammar until the next batch/request begins.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot

from kernel_source_delivery import NormalizedReadRequest, SourceDelivery, deliver_source
from kernel_working_set_allocator import ADMIT_REFUSED, fresh_exposure_allowance
from kernel_working_set_cache import transaction

REFUSED_ADMISSION = ADMIT_REFUSED
FIXED_RESERVE_BYTES = 8_192
# Two frame-only protected pages: what a read may add when nothing about the request is known.
UNMEASURED_FRESH_ALLOWANCE_BYTES = 16_384
EVENT = "read_admission"
_STATE_ATTR = "_read_admission_state"
_SEALED_ATTR = "_read_admission_sealed_fixed_bytes"


def request_byte_cap() -> int:
    """The existing request byte ceiling the next request must fit under."""
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES
    return int(HARD_MAX_REQUEST_BYTES)


def _budget_wire_bytes(value: object) -> int:
    """Same serializer as kernel_request_budget._wire_bytes (escaped Unicode counts fully)."""
    return len(json.dumps(value).encode("utf-8"))


def note_sealed_request(runtime: object, messages: Sequence[Mapping[str, object]],
                        tools: Sequence[object] | None) -> int | None:
    """Record the measured non-fresh bytes of a sealed provider request on the runtime.

    Everything except tool-role replies is treated as mandatory envelope: schemas, system
    prompt, StateFrame, source windows and host facts are composed at request time and are
    invisible to the read boundary.  Returns the recorded value (None when unmeasurable).
    """
    try:
        total = _budget_wire_bytes({"messages": list(messages), "tools": list(tools or ())})
        fresh = sum(_budget_wire_bytes(message) for message in messages
                    if isinstance(message, Mapping) and message.get("role") == "tool")
    except (TypeError, ValueError):
        return None
    fixed = max(0, total - fresh)
    try:
        setattr(runtime, _SEALED_ATTR, fixed)
        body = {"messages": list(messages), "tools": list(tools or ())}
        active = getattr(runtime, "_active_capture", None)
        qualified = getattr(active, "qualified_request", None)
        if qualified is not None:
            body = qualified.body
        runtime._read_admission_provider_body = json.loads(json.dumps(body))
        certificate = getattr(active, "_working_set_certificate", None)
        authority = getattr(certificate, "snapshot", None)
        if authority is not None:
            runtime._read_admission_provider_authority = authority
    except (AttributeError, TypeError):
        return None
    return fixed


def sealed_fixed_bytes(runtime: object) -> int | None:
    value = getattr(runtime, _SEALED_ATTR, None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def fixed_envelope_estimate(runtime: object,
                            visible_messages: Sequence[Mapping[str, object]] | None) -> int:
    """Conservative mandatory-envelope estimate; see the module docstring."""
    pinned = getattr(runtime, "working_set_fixed_bytes", None)
    if isinstance(pinned, int) and not isinstance(pinned, bool) and pinned >= 0:
        return pinned
    candidates: list[int] = []
    if visible_messages:
        try:
            serialized = _budget_wire_bytes(list(visible_messages))
            candidates.append(serialized + FIXED_RESERVE_BYTES)
        except (TypeError, ValueError):
            pass
    sealed = sealed_fixed_bytes(runtime)
    if sealed is not None:
        candidates.append(sealed)
    if not candidates:
        return max(0, request_byte_cap() - UNMEASURED_FRESH_ALLOWANCE_BYTES)
    return max(candidates)


@dataclass
class _RequestState:
    request_index: int
    allowance: int
    pending_bytes: int = 0
    admitted_count: int = 0
    truncated_count: int = 0
    refused_count: int = 0
    closed: bool = False
    batch_identity: object = None
    snapshot: PreparationSnapshot | None = None
    floor_bytes: int = 0
    pages: list[tuple[SourceSlot, str | None]] = field(default_factory=list)
    # Cache exact floor results, including infeasibility, only for this request.
    # Whole immutable slots bind ranges, payloads and native carrier call IDs.
    floor_results: dict[tuple[str, tuple[tuple[SourceSlot, str | None], ...]], int | None] = field(
        default_factory=dict)


def _request_index(runtime: object) -> int:
    return int(getattr(runtime, "_frame_budget_request_count", 0))


def _state(runtime: object, visible_messages: Sequence[Mapping[str, object]] | None) -> _RequestState:
    index = _request_index(runtime)
    state = getattr(runtime, _STATE_ATTR, None)
    batch = getattr(runtime, "native_batch", None)
    origin = (getattr(batch, "origin_response_event_id", None), getattr(batch, "origin", None))
    batch_identity = (batch, *origin) if any(value is not None for value in origin) else None
    if (not isinstance(state, _RequestState)
            or (batch_identity is None and state.request_index != index)
            or state.batch_identity != batch_identity):
        from kernel_request_seal import RequestSealError
        from kernel_working_set_live import (
            read_admission_floor,
            read_admission_snapshot,
        )
        try:
            snapshot = read_admission_snapshot(runtime, visible_messages, byte_cap=request_byte_cap())
            floor_bytes = len(read_admission_floor(snapshot).wire) if snapshot is not None else 0
            allowance = (max(0, snapshot.byte_cap - floor_bytes) if snapshot is not None else
                fresh_exposure_allowance(request_byte_cap(),
                                         fixed_envelope_estimate(runtime, visible_messages), ()))
        except (RequestSealError, ValueError, TypeError, KeyError, OSError, RuntimeError):
            snapshot, floor_bytes, allowance = None, 0, 0
        state = _RequestState(request_index=index, allowance=allowance,
                              batch_identity=batch_identity, snapshot=snapshot, floor_bytes=floor_bytes,
                              closed=allowance == 0)
        try:
            setattr(runtime, _STATE_ATTR, state)
        except (AttributeError, TypeError):
            pass
    return state


def remaining_allowance(runtime: object,
                        visible_messages: Sequence[Mapping[str, object]] | None) -> int:
    """Fresh bytes still admissible in the current request (pending reads already spent)."""
    state = _state(runtime, visible_messages)
    return max(0, state.allowance - state.pending_bytes)


def _wire_bytes(text: str) -> int:
    # Transport uses json.dumps(...).encode(), including ASCII escapes and
    # surrogate pairs for non-BMP characters. Exclude only the string quotes.
    return len(json.dumps(text).encode("utf-8")) - 2


def _journal(runtime: object, state: _RequestState, *, decision: str, page_bytes: int,
             admitted_bytes: int, admitted_lines: int | None, path: str) -> None:
    journal = getattr(runtime, "journal", None)
    if journal is None:
        return
    try:
        journal.emit(
            EVENT, request_index=state.request_index, path=path, decision=decision,
            allowance=state.allowance, pending_before=state.pending_bytes - (
                admitted_bytes if decision != REFUSED_ADMISSION else 0),
            page_bytes=page_bytes, admitted_bytes=admitted_bytes, admitted_lines=admitted_lines,
            admitted_count=state.admitted_count, truncated_count=state.truncated_count,
            refused_count=state.refused_count)
    except Exception:  # audit only; never fail a delivery
        pass


def _resume_cursor(raw: bytes, metadata: Mapping[str, object]) -> str:
    from kernel_source_delivery import _cursor
    scope_end = metadata.get("scope_end_line")
    return _cursor(raw, int(metadata.get("start_line", 1)), int(metadata.get("start_column", 0)),
                   int(scope_end) if isinstance(scope_end, int) else None)


def _refusal(raw: bytes, source: SourceDelivery) -> SourceDelivery:
    """REFUSED_ADMISSION result whose text reuses the pager's own continuation wording."""
    metadata = dict(source.metadata)
    cursor = _resume_cursor(raw, metadata)
    total = int(metadata.get("total_lines", 0))
    start_line = int(metadata.get("start_line", 1))
    remaining_lines = max(0, total - start_line + 1)
    text = f"{remaining_lines} more lines; page with cursor={cursor}"
    metadata.update({
        "result_class": REFUSED_ADMISSION, "shown_end_line": start_line,
        "shown_end_column": int(metadata.get("start_column", 0)), "shown_chars": 0,
        "truncated": True, "more_content_available": True, "more_lines_available": remaining_lines,
        "continuation": cursor, "next_cursor": cursor, "clip_reason": "admission",
        "effective_range": None, "byte_length": 0, "token_count": 1,
    })
    for key in ("byte_start", "byte_end", "page_bytes_sha256"):
        metadata.pop(key, None)
    return SourceDelivery(result_class=REFUSED_ADMISSION, text=text, content="", metadata=metadata)


def fits(runtime: object, visible_messages: Sequence[Mapping[str, object]] | None,
         text: str) -> bool:
    """True when ``text`` fits the remaining allowance (used by the page-cache branch)."""
    state = _state(runtime, visible_messages)
    # A replay without metadata cannot establish a new complete-candidate
    # proposal. Re-render from source through admit(), including capture facts.
    return (not state.closed and state.snapshot is None
            and _wire_bytes(text) <= remaining_allowance(runtime, visible_messages))


def spend(runtime: object, visible_messages: Sequence[Mapping[str, object]] | None,
          text: str, *, path: str) -> None:
    """Charge a replayed cached page against the batch allowance (A1 accumulation).

    The page-cache branch replays without ``admit``; without this charge the next fresh read
    in the same request would be admitted against bytes the replay already spent.  The full
    observation is charged even when a shorter UNCHANGED_VISIBLE stub ends up on the wire.
    """
    state = _state(runtime, visible_messages)
    page_bytes = _wire_bytes(text)
    state.pending_bytes += page_bytes
    state.admitted_count += 1
    _journal(runtime, state, decision="CACHED", page_bytes=page_bytes,
             admitted_bytes=page_bytes, admitted_lines=None, path=path)


def _shared_prefix_counts(state: _RequestState, source: SourceDelivery,
                          path: str | None) -> tuple[int, ...]:
    """Known carrier-sharing dips are separate from growing prefix costs.

    An exact older page can share its FULL carrier with the new page. That
    one prefix can therefore be cheaper than either neighbor. Inspect frozen
    ranges, never every source line, to retain linear-search equivalence.
    """
    if state.snapshot is None:
        return ()
    bounds = source.metadata.get("effective_range")
    if not isinstance(bounds, Mapping):
        return ()
    first, last = bounds["start_line"], bounds["end_line"]
    sources = (*state.snapshot.sources, *(slot for slot, _ in state.pages))
    return tuple(sorted({slot.identity.last_line - first + 1 for slot in sources
        if slot.identity.path == path
        and slot.identity.revision == source.metadata.get("revision")
        and slot.identity.first_line == first and first <= slot.identity.last_line < last},
        reverse=True))


@transaction
def admit(runtime: object, *, raw: bytes, display: str, start: int,
          request: NormalizedReadRequest, source: SourceDelivery,
          visible_messages: Sequence[Mapping[str, object]] | None) -> SourceDelivery:
    """Admit, shorten or refuse an OK page before capture.  Non-OK pages pass through."""
    if source.result_class != "OK":
        return source
    state = _state(runtime, visible_messages)
    remaining = max(0, state.allowance - state.pending_bytes)
    page_bytes = _wire_bytes(source.text)
    path = display
    if state.snapshot is not None:
        from kernel_read_receipts import relative_path
        root = getattr(getattr(runtime, "owner", None), "root", None)
        path = relative_path(root, Path(root) / display) if root is not None else display

    def cost(candidate):
        if state.snapshot is None:
            return _wire_bytes(candidate.text)
        from kernel_working_set_live import read_admission_floor, read_admission_slot
        if path is None:
            return state.allowance + 1
        ordinal = max((s.admission_ordinal for s in state.snapshot.sources), default=0) + len(state.pages) + 1
        slot = read_admission_slot(candidate, path=path, arguments=request.arguments, ordinal=ordinal)
        call_id = getattr(getattr(runtime, "_read_attempt_scope", None), "call_id", None)
        pages = (*state.pages, (slot, call_id))
        key = (state.snapshot.snapshot_digest, pages)
        if key not in state.floor_results:
            floor = read_admission_floor(state.snapshot, pages, require_feasible=True)
            state.floor_results[key] = len(floor.wire) if floor is not None else None
        floor_bytes = state.floor_results[key]
        if floor_bytes is None:
            return remaining + 1
        return max(0, floor_bytes - state.floor_bytes - state.pending_bytes)

    def accept(candidate, charged):
        if state.snapshot is not None:
            from kernel_working_set_live import read_admission_slot
            ordinal = max((s.admission_ordinal for s in state.snapshot.sources), default=0) + len(state.pages) + 1
            slot = read_admission_slot(candidate, path=path, arguments=request.arguments, ordinal=ordinal)
            call_id = getattr(getattr(runtime, "_read_attempt_scope", None), "call_id", None)
            state.pages.append((slot, call_id))
        state.pending_bytes += charged

    # The original page can be cheaper than a shorter one: it may remove the
    # continuation footer or reuse an exact frozen native carrier. Check it
    # separately before searching prefixes that all retain continuation text.
    full_cost = cost(source) if not state.closed else remaining + 1
    if not state.closed and full_cost <= remaining:
        accept(source, full_cost)
        state.admitted_count += 1
        _journal(runtime, state, decision="FULL", page_bytes=page_bytes,
                 admitted_bytes=full_cost, admitted_lines=None, path=display)
        return source
    shorter: SourceDelivery | None = None
    lines = 0
    admitted = 0
    total_lines = len(source.content.splitlines())
    # A fixed start/revision and increasing whole-line prefixes grow the fresh
    # FULL carrier. Price complete floors at logarithmically many boundaries,
    # retaining the largest feasible prefix. A 120-line page needs at most
    # one full-page evaluation plus seven prefix evaluations.
    if not state.closed:
        low, high = 1, total_lines - 1
        while low <= high:
            count = (low + high) // 2
            candidate = deliver_source(raw, display, start, args=replace(request, max_lines=count))
            charged = cost(candidate) if candidate.result_class == "OK" else remaining + 1
            if charged <= remaining:
                shorter, lines, admitted = candidate, count, charged
                low = count + 1
            else:
                high = count - 1
        # Exact sharing creates isolated cost dips in the otherwise growing
        # prefix family. Only frozen carrier boundaries need additional probes;
        # extra probes depend on matching frozen ranges, not the line count.
        for count in _shared_prefix_counts(state, source, path):
            if count <= lines:
                break
            candidate = deliver_source(raw, display, start, args=replace(request, max_lines=count))
            carriers = (*state.snapshot.sources, *(slot for slot, _ in state.pages))
            # Model source windows use a different carrier grammar and cannot
            # share the pager's native read text, even at identical ranges.
            if candidate.result_class == "OK" and any(s.page == candidate.text for s in carriers):
                charged = cost(candidate)
                if charged <= remaining:
                    shorter, lines, admitted = candidate, count, charged
                    break
    if shorter is None:
        state.closed = True
        state.refused_count += 1
        _journal(runtime, state, decision=REFUSED_ADMISSION, page_bytes=page_bytes,
                 admitted_bytes=0, admitted_lines=0, path=display)
        return _refusal(raw, source)
    accept(shorter, admitted)
    state.truncated_count += 1
    _journal(runtime, state, decision="TRUNCATE", page_bytes=page_bytes,
             admitted_bytes=admitted, admitted_lines=lines, path=display)
    return shorter


__all__ = ["EVENT", "FIXED_RESERVE_BYTES", "REFUSED_ADMISSION", "UNMEASURED_FRESH_ALLOWANCE_BYTES",
           "admit", "fits", "fixed_envelope_estimate", "note_sealed_request",
           "remaining_allowance", "request_byte_cap", "sealed_fixed_bytes", "spend"]
