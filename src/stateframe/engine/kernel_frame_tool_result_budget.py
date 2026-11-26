"""Bound newest FRAME_ONLY tool results before they become protected history."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from kernel_context import token_estimate
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, HARD_MAX_REQUEST_BYTES

if TYPE_CHECKING:
    from kernel_generic_check import CauseFrame


_TEST_CLIP_NOTICE = ("\n[TEST OUTPUT CLIPPED: full result remains in the check receipt; "
                     "use read_check_receipt for details or rerun a narrower test target.]\n")


def frame_only_page_byte_limit(messages: Sequence[Mapping[str, object]]) -> int:
    """Reserve request and token headroom for the next protected result.

    History fitting may free more space later. This conservative limit is for
    the one result it cannot evict, and the read pager supplies the cursor.
    """
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()
    byte_room = HARD_MAX_REQUEST_BYTES - len(serialized) - 8_192
    token_room = (ENDPOINT_PROMPT_TOKEN_BUDGET - token_estimate(messages) - 2_048) * 2
    return max(1_024, min(8_192, byte_room, token_room))


def read_page_byte_limit(
    messages: Sequence[Mapping[str, object]], lean_limit: int | None = None,
) -> int | None:
    """Apply the protected-result cap even when lean presentation is active."""
    from kernel_stateframe_framefit5 import has_framefit5_state

    if not has_framefit5_state(messages):
        return lean_limit
    frame_limit = frame_only_page_byte_limit(messages)
    return min(lean_limit, frame_limit) if lean_limit is not None else frame_limit


def bound_frame_tool_result(
    content: str, messages: Sequence[Mapping[str, object]], *, kind: str,
    trailer: str = "",
) -> str:
    """Keep test output useful while giving the model a truthful retrieval route.

    ``trailer`` is host truth that must survive verbatim (the scoped-check line); its bytes
    are reserved inside the budget so the delivered result never exceeds the cap.
    """
    limit = frame_only_page_byte_limit(messages)
    if trailer:
        limit = max(0, limit - len(trailer.encode("utf-8")))
    raw = content.encode("utf-8")
    if len(raw) <= limit:
        return content + trailer
    if kind != "test":
        return content + trailer
    notice = _TEST_CLIP_NOTICE
    room = max(0, limit - len(notice.encode("utf-8")))
    head = raw[:room // 2].decode("utf-8", errors="ignore")
    tail = raw[-(room - len(head.encode("utf-8"))):].decode("utf-8", errors="ignore") if room else ""
    return head + notice + tail + trailer


def fit_check_cause_excerpt(ordinary: str, full: str, observation: str,
                            frames: Sequence[CauseFrame], *,
                            source_line_ends: frozenset[int],
                            diagnostic_starts_midline: bool = False) -> str:
    """Use a clipped traceback source line's byte budget for the complete chain."""
    if (not source_line_ends or not full.startswith(observation) or not frames[-1].visible_in(ordinary)
            or all(frame.visible_in(ordinary) for frame in frames[:-1])):
        return ordinary

    if ordinary == full:
        if not diagnostic_starts_midline:
            return ordinary
        head, separator, tail = ordinary.partition("\n")
    else:
        head, separator, tail = ordinary.partition(_TEST_CLIP_NOTICE)
        if not separator or not full.startswith(head) or not full.endswith(tail):
            return ordinary
        tail_start = len(full) - len(tail)
        if tail_start <= len(head) or full[tail_start - 1] == "\n":
            return ordinary

    fragment, newline, remainder = tail.partition("\n")
    suffix = full[len(observation):]
    prefix = head + separator
    fragment_end = len(full) - len(remainder)
    # A clipped exception message can look like source padding. Use the line
    # boundary identified in the capture, counted backward from the observation end.
    if (not newline or fragment_end > len(observation)
            or observation.count("\n", fragment_end) not in source_line_ends
            or not remainder.endswith(suffix)
            or not frames[-1].visible_in(prefix + remainder)):
        return ordinary

    rows = "\n  directly caused\n".join(
        "  ".join(filter(None, (frame.exception, frame.location, frame.function)))
        for frame in frames)
    excerpt = f"[Direct exception causes]\n{rows}\n"
    excerpt_bytes = len(excerpt.encode("utf-8"))
    fragment_bytes = (fragment + newline).encode("utf-8")
    if excerpt_bytes > len(fragment_bytes):
        return ordinary
    residual_fragment = fragment_bytes[excerpt_bytes:].decode("utf-8", errors="ignore")
    return prefix + excerpt + residual_fragment + remainder
