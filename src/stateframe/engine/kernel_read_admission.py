"""A1b read-time admission: shorten or refuse a fresh page before it is captured.

Gated by ``kernel_read_receipts.enabled()``.  OFF leaves the read path byte-identical.

Per request, the fresh-exposure allowance is ``fresh_exposure_allowance(cap, fixed, protected)``
from the pure allocator core, where:

* ``cap`` is the existing request byte ceiling, ``HARD_MAX_REQUEST_BYTES`` (the ceiling the
  frame-only page limit already reserves against);
* ``fixed`` is a conservative estimate of the mandatory envelope, the larger of two components:
  the compact-JSON bytes of the messages visible to this read (the same estimator
  ``frame_only_page_byte_limit`` uses) plus a ``FIXED_RESERVE_BYTES`` reserve, and the MEASURED
  non-fresh bytes of the last sealed provider request (``note_sealed_request``: messages + tools
  in the request budget's own serializer, minus that request's tool-role replies).  The measured
  component exists because the read-time messages miss what request composition adds later
  (tool schemas ~17.5 KB, the StateFrame 12-30 KB, source windows, volatile tail; r38g wire 0020
  has a 72 KB envelope under an 82 KB cap).  With neither messages nor a sealed request the
  allowance is ``UNMEASURED_FRESH_ALLOWANCE_BYTES``.  History fitting can only free space later,
  never add, so the estimate errs towards a smaller allowance.  A runtime may pin the estimate
  with ``working_set_fixed_bytes`` (item 2 floor plumbing, tests);
* protected slots are not known at the read boundary yet, so the min-cost term is zero here and
  the caller folds any protected minimum into ``working_set_fixed_bytes`` (fable_core_r2 F5).

Pending fresh bytes admitted earlier in the same request are accumulated on the runtime and
spent in batch order (A1).  A page that overflows is re-rendered on a whole-line boundary
(``admit_read_lines``) so the existing ``next_cursor`` grammar carries the remainder; when no
line fits the read returns ``REFUSED_ADMISSION`` with a resume cursor.  BLOCKED is never raised
from here: a zero allowance only produces refusals with cursors.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import json

from kernel_source_delivery import NormalizedReadRequest, SourceDelivery, deliver_source
from kernel_working_set_allocator import ADMIT_REFUSED, admit_read_lines, fresh_exposure_allowance

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
            serialized = len(json.dumps(
                list(visible_messages), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
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


def _request_index(runtime: object) -> int:
    return int(getattr(runtime, "_frame_budget_request_count", 0))


def _state(runtime: object, visible_messages: Sequence[Mapping[str, object]] | None) -> _RequestState:
    index = _request_index(runtime)
    state = getattr(runtime, _STATE_ATTR, None)
    if not isinstance(state, _RequestState) or state.request_index != index:
        allowance = fresh_exposure_allowance(
            request_byte_cap(), fixed_envelope_estimate(runtime, visible_messages), ())
        state = _RequestState(request_index=index, allowance=allowance)
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
    return len(text.encode("utf-8", errors="replace"))


def _line_byte_lengths(body: str) -> list[int]:
    return [len(piece.encode("utf-8", errors="replace")) for piece in body.splitlines(keepends=True)]


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
    return _wire_bytes(text) <= remaining_allowance(runtime, visible_messages)


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


def admit(runtime: object, *, raw: bytes, display: str, start: int,
          request: NormalizedReadRequest, source: SourceDelivery,
          visible_messages: Sequence[Mapping[str, object]] | None) -> SourceDelivery:
    """Admit, shorten or refuse an OK page before capture.  Non-OK pages pass through."""
    if source.result_class != "OK":
        return source
    state = _state(runtime, visible_messages)
    remaining = max(0, state.allowance - state.pending_bytes)
    page_bytes = _wire_bytes(source.text)
    if page_bytes <= remaining:
        state.pending_bytes += page_bytes
        state.admitted_count += 1
        _journal(runtime, state, decision="FULL", page_bytes=page_bytes,
                 admitted_bytes=page_bytes, admitted_lines=None, path=display)
        return source
    overhead = page_bytes - _wire_bytes(source.content)
    line_bytes = _line_byte_lengths(source.content)
    (decision,) = admit_read_lines(max(0, remaining - overhead), [line_bytes])
    lines = decision.admitted_lines if decision.kind != REFUSED_ADMISSION else 0
    shorter: SourceDelivery | None = None
    while lines > 0:
        candidate = deliver_source(raw, display, start, args=replace(request, max_lines=lines))
        if candidate.result_class == "OK" and _wire_bytes(candidate.text) <= remaining:
            shorter = candidate
            break
        lines -= 1  # the header grew; the loop ends on a line boundary or refuses
    if shorter is None:
        state.refused_count += 1
        _journal(runtime, state, decision=REFUSED_ADMISSION, page_bytes=page_bytes,
                 admitted_bytes=0, admitted_lines=0, path=display)
        return _refusal(raw, source)
    admitted = _wire_bytes(shorter.text)
    state.pending_bytes += admitted
    state.truncated_count += 1
    _journal(runtime, state, decision="TRUNCATE", page_bytes=page_bytes,
             admitted_bytes=admitted, admitted_lines=lines, path=display)
    return shorter


__all__ = ["EVENT", "FIXED_RESERVE_BYTES", "REFUSED_ADMISSION", "UNMEASURED_FRESH_ALLOWANCE_BYTES",
           "admit", "fits", "fixed_envelope_estimate", "note_sealed_request",
           "remaining_allowance", "request_byte_cap", "sealed_fixed_bytes", "spend"]
