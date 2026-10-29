"""Bound newest FRAME_ONLY tool results before they become protected history."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from kernel_context import token_estimate
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, HARD_MAX_REQUEST_BYTES


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
) -> str:
    """Keep test output useful while giving the model a truthful retrieval route."""
    limit = frame_only_page_byte_limit(messages)
    raw = content.encode("utf-8")
    if len(raw) <= limit:
        return content
    if kind != "test":
        return content
    notice = ("\n[TEST OUTPUT CLIPPED: full result remains in the check receipt; "
              "use read_check_receipt for details or rerun a narrower test target.]\n")
    room = max(0, limit - len(notice.encode("utf-8")))
    head = raw[:room // 2].decode("utf-8", errors="ignore")
    tail = raw[-(room - len(head.encode("utf-8"))):].decode("utf-8", errors="ignore") if room else ""
    return head + notice + tail
