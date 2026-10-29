"""A done call is never answered with an empty tool reply.

b1427 benchmark: three `done` calls on a green, unchanged, correct candidate were answered with len-0 tool replies.
The model had no refusal to inspect, so it re-read, re-ran tests and called done again until the wall.
Every request passes through RequestCapture.send, which fills such replies with one plain sentence BEFORE
preparation (so wire truth and request hashes describe the final bytes). Non-empty replies are untouched.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

EMPTY_DONE_REPLY = ("done was not accepted on this turn, and the host gave no reason. "
                    "Your change is kept. Call done again to submit it.")


def _done_call_ids(messages: Sequence[object]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else call.get("name")
            if str(name or "").strip() == "done" and call.get("id"):
                ids.add(str(call["id"]))
    return ids


def fill_empty_done_replies(messages: object) -> int:
    """Fill empty tool replies to done calls in place; return how many were filled."""
    if not isinstance(messages, list):
        return 0
    done_ids = _done_call_ids(messages)
    if not done_ids:
        return 0
    filled = 0
    for message in messages:
        if (isinstance(message, dict) and message.get("role") == "tool"
                and str(message.get("tool_call_id") or "") in done_ids
                and not str(message.get("content") or "").strip()):
            message["content"] = EMPTY_DONE_REPLY
            filled += 1
    return filled


__all__ = ["EMPTY_DONE_REPLY", "fill_empty_done_replies"]
