"""Shared, layout-aware message-order contract for FrameAppend and PAGE_LOG.

Plain FrameAppend defaults to leading system messages, the original user task,
then one StateFrame base. PAGE_LOG explicitly opts into its extended
`[S]|[L]|A+t|[V]` layout: optional user context before the base, PageLog records
and tail after it, the newest exchange, then the final user message. FrameAppend
stores only the StateFrame payload; assistant/tool transcript messages stay outside it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast


STATEFRAME_PREFIX = "[StateFrame]\n"
TASK_PREFIX = "Task:"


class StateFrameMessageLayoutError(ValueError):
    """Messages do not follow the shared FrameAppend/PAGE_LOG layout."""


@dataclass(frozen=True)
class StateFrameMessageLayout:
    system_end: int
    task_index: int
    state_index: int
    exchange_start: int | None = None
    exchange_end: int | None = None


def _validated_messages(messages: object) -> Sequence[Mapping[str, object]]:
    if (not isinstance(messages, Sequence)
            or isinstance(messages, (str, bytes, bytearray))):
        raise StateFrameMessageLayoutError("frame append request messages are malformed")
    if any(not isinstance(message, Mapping) for message in messages):
        raise StateFrameMessageLayoutError("frame append request messages are malformed")
    return cast(Sequence[Mapping[str, object]], messages)


def stateframe_message_index(messages: object) -> int:
    """Return the unique typed StateFrame base index."""
    validated = _validated_messages(messages)
    indexes = [
        index for index, message in enumerate(validated)
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and str(message.get("content")).startswith(STATEFRAME_PREFIX)
    ]
    if len(indexes) != 1:
        raise StateFrameMessageLayoutError("frame append requires exactly one StateFrame")
    return indexes[0]


def validate_stateframe_message_order(
        messages: object,
        *,
        require_task_label: bool,
        allow_native_exchange: bool = False,
        allow_user_context_before_stateframe: bool = False,
) -> StateFrameMessageLayout:
    """Validate FrameAppend order, with optional PAGE_LOG context and native exchange."""
    validated = _validated_messages(messages)
    state_index = stateframe_message_index(validated)
    system_end = next(
        (index for index, message in enumerate(validated)
         if message.get("role") != "system"),
        len(validated),
    )
    if (system_end < 1 or system_end >= state_index
            or any(message.get("role") == "system" for message in validated[system_end:])):
        raise StateFrameMessageLayoutError(
            "frame append message order is not system, task, StateFrame",
        )

    task_index = system_end
    task = validated[task_index]
    task_content = task.get("content")
    if (task.get("role") != "user" or not isinstance(task_content, str)
            or not task_content.strip()
            or (require_task_label and not task_content.startswith(TASK_PREFIX))):
        raise StateFrameMessageLayoutError(
            "frame append requires the original task before StateFrame",
        )

    if (not allow_user_context_before_stateframe
            and state_index != task_index + 1):
        raise StateFrameMessageLayoutError(
            "frame append message order is not system, task, StateFrame",
        )
    # PAGE_LOG may retain user context between Task and StateFrame. These rows
    # remain in [L]; only the task itself anchors FrameAppend message identity.
    if any(message.get("role") != "user"
           for message in validated[task_index:state_index + 1]):
        raise StateFrameMessageLayoutError(
            "frame append message order is not system, task, StateFrame",
        )

    tail = validated[state_index + 1:]
    action_indexes = [
        index for index, message in enumerate(tail)
        if message.get("role") in {"assistant", "tool"}
    ]
    if not action_indexes:
        if any(message.get("role") != "user" for message in tail):
            raise StateFrameMessageLayoutError(
                "frame append cannot carry prior assistant/tool messages",
            )
        return StateFrameMessageLayout(system_end, task_index, state_index)

    if not allow_native_exchange:
        raise StateFrameMessageLayoutError(
            "frame append cannot carry prior assistant/tool messages",
        )

    first_action = action_indexes[0]
    assistant = tail[first_action]
    calls = assistant.get("tool_calls")
    if (assistant.get("role") != "assistant" or not isinstance(calls, list) or not calls
            or any(message.get("role") != "user" for message in tail[:first_action])):
        raise StateFrameMessageLayoutError("PAGE_LOG A+t exchange is malformed")

    call_ids: list[str] = []
    for call in calls:
        call_id = call.get("id") if isinstance(call, Mapping) else None
        if not isinstance(call_id, str) or not call_id or call_id in call_ids:
            raise StateFrameMessageLayoutError("PAGE_LOG A+t call identity is malformed")
        call_ids.append(call_id)

    exchange_end = first_action + 1 + len(call_ids)
    if exchange_end >= len(tail):
        raise StateFrameMessageLayoutError("PAGE_LOG A+t exchange has no final [V] user message")
    replies = tail[first_action + 1:exchange_end]
    if (len(replies) != len(call_ids)
            or any(reply.get("role") != "tool" for reply in replies)):
        raise StateFrameMessageLayoutError("PAGE_LOG A+t tool replies are malformed")
    reply_ids = [reply.get("tool_call_id") for reply in replies]
    if (any(not isinstance(reply_id, str) or not reply_id for reply_id in reply_ids)
            or len(set(reply_ids)) != len(reply_ids)
            or set(reply_ids) != set(call_ids)):
        raise StateFrameMessageLayoutError("PAGE_LOG A+t tool reply identity is malformed")
    if (action_indexes != list(range(first_action, exchange_end))
            or any(message.get("role") != "user" for message in tail[exchange_end:])
            or len(tail[exchange_end:]) != 1):
        raise StateFrameMessageLayoutError(
            "PAGE_LOG message order is not [S]|[L]|A+t|[V]",
        )

    return StateFrameMessageLayout(
        system_end=system_end,
        task_index=task_index,
        state_index=state_index,
        exchange_start=state_index + 1 + first_action,
        exchange_end=state_index + 1 + exchange_end,
    )
