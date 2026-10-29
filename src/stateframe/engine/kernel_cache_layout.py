"""Deterministic, permutation-only layout for StateFrame request prefixes.

The cache-layout arm changes presentation order only.  It does not enable a
provider cache, alter the authorized tool menu, redact fields, or synthesize a
new model-facing block.  The structured ``tools`` value stays a separate
request member; its existing schemas are sorted as one stable schema group so
the chat template can consume them deterministically.

The module is intentionally independent of the StateFrame runtime.  That keeps
the hot integration point small and gives the offline journal measurement a
pure function to exercise.
"""
from __future__ import annotations

import collections
import dataclasses
import hashlib
import glob
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from kernel_read_fixes import SOURCE_BLOCK_PREFIXES as _LABELLED_SOURCE_PREFIXES


MODE_ENV = "STATEFRAME_CACHE_LAYOUT"
TRAILING_PANELS_MODE_ENV = "STATEFRAME_TRAILING_PANELS"
SCHEMA = "stateframe-cache-layout/1"
_VALID_MODES = frozenset({"off", "on"})
# These tools are host-gated and may enter the menu only after an observed
# condition.  Keep them out of the stable tool prefix and append them in a
# deterministic order when prompt-cache serialization is active.
GATED_TOOL_NAMES = frozenset({"job"})

# These are presentation classes, not authorization classes.  Unknown and
# future blocks remain in the dynamic suffix in their original order.
_MESSAGE_RANK = {
    "protocol": 0,
    "ticket": 1,
    "ticket_facts": 2,
    "source": 10,
    "checks": 20,
    "receipts": 21,
    "open_work": 22,
    "jobs": 23,
    "step_metadata": 24,
    "dynamic": 25,
    "history": 26,
}

# Label contract: every source-block label (plain "[Current source:" included) is a
# source block; a missing label would demote plain-label sources to the dynamic tail.
_SOURCE_PREFIXES = (
    *_LABELLED_SOURCE_PREFIXES,
    "[Source content:",
    "CURRENT_SOURCE ",
)
_CHECK_PREFIXES = (
    "[StateFrame]\n",
    "<current_state ",
    "[WORKING_VIEW ",
    "<settled_state>\n",
)
_RECEIPT_PREFIXES = (
    "[Receipts]",
    "[Receipt]",
    "[TEST RECEIPT]",
    "<receipts>",
)
_OPEN_WORK_PREFIXES = (
    "[Open work]",
    "[OPEN WORK]",
    "<open_work>",
)
_JOB_PREFIXES = (
    "[Jobs]",
    "[JOBS]",
    "<jobs>",
)
_STEP_PREFIXES = (
    "[Continuation]",
    "FRAME_CLARITY_CONTINUATION:",
    "[Step metadata]",
    "[STEP METADATA]",
)
_TICKET_FACT_PREFIXES = (
    "[Repository layout;",
    "[Immutable repository facts]",
    "[REPOSITORY FACTS]",
)
_VOLATILE_FIELD_RE = re.compile(
    r"(?i)(?:[\"']?(?:step(?:_number)?|timestamp|revision(?:_hash)?|"
    r"(?:git_)?(?:sha|hash)|budget(?:_tokens)?)[\"']?\s*[:=])"
)


def _has_volatile_field(content: str) -> bool:
    return (bool(_VOLATILE_FIELD_RE.search(content))
            or "WORKING_SOURCE_DELIVERED:" in content)


def cache_layout_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strictly validated cache-layout switch."""
    source = os.environ if environ is None else environ
    value = source.get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Short alias used by switch manifests and callers."""
    return cache_layout_mode(environ)


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return cache_layout_mode(environ) == "on"


def configuration(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    return {"cache_layout": cache_layout_mode(environ)}


def _canonical(value: object) -> bytes:
    """Serialize JSON request values with a stable key and separator policy."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")

def trailing_panels_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict default-off Stage 1 switch."""
    source = os.environ if environ is None else environ
    value = source.get(TRAILING_PANELS_MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{TRAILING_PANELS_MODE_ENV} must be off or on")
    return value


def trailing_panels_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return trailing_panels_mode(environ) == "on"


def history_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return assistant/tool transcript messages in their original order."""
    return [
        dict(message)
        for message in messages
        if message.get("role") in {"assistant", "tool"}
    ]


def _messages_digest(messages: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(_canonical(list(messages))).hexdigest()


def history_prefix_audit(
    previous: Sequence[Mapping[str, object]] | None,
    current: Sequence[Mapping[str, object]],
    *,
    previous_request_id: object = None,
) -> dict[str, object]:
    """Prove old assistant/tool history is an exact prefix, or return a pointer."""
    current_rows = [dict(message) for message in current]
    previous_rows = [dict(message) for message in previous] if previous is not None else []
    prefix_unchanged = previous is None or (
        len(current_rows) >= len(previous_rows)
        and current_rows[:len(previous_rows)] == previous_rows
    )
    changed_index = None
    if previous is not None and not prefix_unchanged:
        changed_index = next(
            (
                index for index, (expected, actual) in enumerate(
                    zip(previous_rows, current_rows)
                )
                if expected != actual
            ),
            min(len(previous_rows), len(current_rows)),
        )
    previous_digest = _messages_digest(previous_rows) if previous is not None else None
    current_digest = _messages_digest(current_rows)
    pointer = None
    if previous is not None and not prefix_unchanged:
        request_label = str(previous_request_id or "unknown")
        pointer = (
            "[History fold pointer] Earlier assistant/tool history was folded or changed. "
            f"Previous request {request_label} has {len(previous_rows)} messages "
            f"(sha256={previous_digest}); first changed index={changed_index}."
        )
    return {
        "previous_history_count": len(previous_rows) if previous is not None else None,
        "current_history_count": len(current_rows),
        "previous_history_sha256": previous_digest,
        "current_history_sha256": current_digest,
        "prefix_unchanged": prefix_unchanged,
        "first_changed_index": changed_index,
        "fold_pointer": pointer,
    }


def observe_history_prefix(
    owner: object,
    messages: Sequence[Mapping[str, object]],
    *,
    request_id: object = None,
) -> dict[str, object]:
    """Journal each turn's append-only history check and retain its next prefix."""
    current = history_messages(messages)
    previous = getattr(owner, "_trailing_panels_previous_history", None)
    previous_request_id = getattr(owner, "_trailing_panels_previous_request_id", None)
    previous_rows = previous if isinstance(previous, list) else None
    audit = history_prefix_audit(
        previous_rows, current, previous_request_id=previous_request_id,
    )
    setattr(owner, "_trailing_panels_previous_history", current)
    setattr(owner, "_trailing_panels_previous_request_id", request_id)
    journal = getattr(owner, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        fields = dict(audit)
        pointer = fields.pop("fold_pointer")
        emit(
            "trailing_panels_history_prefix",
            producer="kernel_cache_layout",
            request_id=request_id,
            status="fold_pointer_appended" if pointer else (
                "append_only" if previous_rows is not None else "baseline"
            ),
            dispatch=False,
            **fields,
        )
    return audit


def track_tool_menu(owner: object, tools: object, *, request_id: object = None) -> dict[str, object]:
    """Journal menu changes while leaving every authorization decision intact."""
    digest = hashlib.sha256(_canonical(tools)).hexdigest()
    names = tuple(sorted({
        _tool_name(tool)
        for tool in tools
        if isinstance(tool, Mapping) and _tool_name(tool)
    })) if isinstance(tools, Sequence) and not isinstance(
        tools, (str, bytes, bytearray)
    ) else ()
    previous_digest = getattr(owner, "_trailing_panels_tool_menu_digest", None)
    previous_names = getattr(owner, "_trailing_panels_tool_menu_names", ())
    event: dict[str, object] = {
        "status": "baseline" if previous_digest is None else "unchanged",
        "menu_sha256": digest,
        "tool_names": list(names),
    }
    if previous_digest is not None and previous_digest != digest:
        previous = set(previous_names) if isinstance(previous_names, (set, tuple, list)) else set()
        changed_names = previous.symmetric_difference(names)
        break_count = int(getattr(owner, "_trailing_panels_tool_menu_break_count", 0))
        expected = changed_names == {"job"} and break_count == 0
        break_count += 1
        setattr(owner, "_trailing_panels_tool_menu_break_count", break_count)
        event.update({
            "status": "expected_job_gate" if expected else "unexpected_menu_change",
            "previous_menu_sha256": previous_digest,
            "added_tools": sorted(set(names) - previous),
            "removed_tools": sorted(previous - set(names)),
            "menu_break_ordinal": break_count,
        })
        journal = getattr(owner, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit(
                "trailing_panels_tool_menu_break",
                producer="kernel_cache_layout",
                request_id=request_id,
                expected=expected,
                changed_tool_names=sorted(changed_names),
                dispatch=False,
                **event,
            )
    setattr(owner, "_trailing_panels_tool_menu_digest", digest)
    setattr(owner, "_trailing_panels_tool_menu_names", names)
    return event


def _panel_content(panels: Sequence[Mapping[str, object]]) -> object:
    structured = any(
        not isinstance(message.get("content"), str)
        for message in panels
    )
    if not structured:
        return "\n\n".join(
            content for message in panels
            if isinstance((content := message.get("content")), str)
        )

    parts: list[object] = []
    for index, message in enumerate(panels):
        if index:
            parts.append({"type": "text", "text": "\n\n"})
        content = message.get("content")
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, Sequence) and not isinstance(
            content, (str, bytes, bytearray)
        ):
            parts.extend(content)
        elif content is not None:
            parts.append(content)
    return parts


def layout_trailing_panels(
    messages: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """Match Stage 0: system, task, target pin, intact history, one panel message."""
    original = list(messages)
    if any(not isinstance(message, Mapping) for message in original):
        raise TypeError("trailing-panel layout requires mapping messages")

    system: list[Mapping[str, object]] = []
    developer: list[Mapping[str, object]] = []
    history: list[Mapping[str, object]] = []
    user_messages: list[Mapping[str, object]] = []
    for message in original:
        role = message.get("role")
        if role == "system":
            system.append(message)
        elif role == "developer":
            developer.append(message)
        elif role == "user":
            user_messages.append(message)
        else:
            history.append(message)

    task_index = next(
        (
            index for index, message in enumerate(user_messages)
            if isinstance(message.get("content"), str)
            and message["content"].startswith("Task:")
        ),
        None,
    )
    target_index = next(
        (
            index for index, message in enumerate(user_messages)
            if isinstance(message.get("content"), str)
            and message["content"].startswith("[Current source:")
        ),
        None,
    )
    if task_index is None:
        task_index = next(
            (
                index for index, message in enumerate(user_messages)
                if index != target_index
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ),
            None,
        )

    task = user_messages[task_index] if task_index is not None else None
    target = user_messages[target_index] if target_index is not None else None
    panels = [
        message for index, message in enumerate(user_messages)
        if index not in {task_index, target_index}
    ]

    arranged: list[Mapping[str, object]] = [*system, *developer]
    if task is not None:
        arranged.append(task)
    if target is not None:
        arranged.append(target)
    arranged.extend(history)
    if panels:
        arranged.append({
            "role": "user",
            "content": _panel_content(panels),
        })
    return arranged


def trailing_panels_audit(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Describe layout and history fingerprints without logging panel contents."""
    original_history = history_messages(before)
    final_history = history_messages(after)
    return {
        "trailing_panels_schema": "stateframe-trailing-panels/1",  # never "schema": journal-RESERVED (r2EY ALL9 crash)
        "changed": list(before) != list(after),
        "message_count_before": len(before),
        "message_count_after": len(after),
        "history_message_count": len(final_history),
        "history_sha256": _messages_digest(final_history),
        "history_preserved": original_history == final_history,
        "trailing_user_message_count": sum(
            1 for message in after
            if message.get("role") == "user"
        ),
        "order": ["system", "task", "target_pin", "history", "panels"],
    }





def canonical_request_bytes(messages: Sequence[Mapping[str, object]],
                            tools: object) -> bytes:
    """Return the deterministic prompt portion used by the offline metric.

    Tools are a separate structured request member at the adapter boundary,
    but they belong to the stable prefix group.  Keep that member first in the
    metric envelope, then preserve the rendered message order exactly.
    """
    return (b'{"tools":' + _canonical(tools) + b',"messages":' +
            _canonical(list(messages)) + b'}')


def _content(message: Mapping[str, object]) -> str:
    value = message.get("content", "")
    return value if isinstance(value, str) else ""


def _tool_name(tool: Mapping[str, object]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping):
        value = function.get("name")
        if isinstance(value, str):
            return value
    value = tool.get("name")
    return value if isinstance(value, str) else ""


def _source_sort_key(message: Mapping[str, object]) -> tuple[str, str, bytes]:
    """Sort source blocks by stable path before considering their bytes."""
    content = _content(message)
    first_line = content.splitlines()[0] if content.splitlines() else ""
    path = first_line
    if ": " in first_line:
        path = first_line.split(": ", 1)[1]
    if path.endswith("]"):
        path = path[:-1]
    # Revision aliases belong to the changing source body, not the episode
    # stable path used for ordering.
    path = path.split("@", 1)[0].strip()
    return path, first_line, _canonical(message)


def layout_message_kind(message: Mapping[str, object]) -> str:
    """Classify one already-rendered message without inspecting its status.

    Classification is marker-based and deliberately conservative.  A block
    that is not known to be stable is dynamic, so a future feature cannot
    accidentally move volatile data into the reusable prefix.
    """
    role = message.get("role")
    content = _content(message)
    if (role == "system" or role == "developer"):
        return "dynamic" if _has_volatile_field(content) else "protocol"
    if role in {"assistant", "tool"}:
        return "history"
    if content.startswith("Task:"):
        return "dynamic" if _has_volatile_field(content) else "ticket"
    if content.startswith("[FRAME_CLARITY_BOOTSTRAP]"):
        # The capsule is stable ticket context only when it has no delivered
        # source revision or other explicitly volatile field.
        return "dynamic" if _has_volatile_field(content) else "ticket_facts"
    if content.startswith(_TICKET_FACT_PREFIXES):
        return "dynamic" if _has_volatile_field(content) else "ticket_facts"
    if content.startswith(_SOURCE_PREFIXES):
        return "source"
    if content.startswith(_CHECK_PREFIXES):
        return "checks"
    if content.startswith(_RECEIPT_PREFIXES):
        return "receipts"
    if content.startswith(_OPEN_WORK_PREFIXES):
        return "open_work"
    if content.startswith(_JOB_PREFIXES):
        return "jobs"
    if content.startswith(_STEP_PREFIXES):
        return "step_metadata"
    return "dynamic"


def block_kind(message: Mapping[str, object]) -> str:
    """Alias for callers that use the shorter vocabulary."""
    return layout_message_kind(message)


def _tool_values(tools: object) -> list[object] | None:
    if tools is None:
        return None
    if isinstance(tools, (str, bytes, bytearray)):
        return None
    if not isinstance(tools, Sequence):
        return None
    return list(tools)


def layout_tools(tools: object, *, mode: str = "on", prompt_cache: bool = False,
                 gated_tool_names: Iterable[str] = GATED_TOOL_NAMES) -> object:
    """Sort the existing schema list without changing its membership.

    This function never applies a permission rule.  Host authorization and
    menu construction happen before this presentation-only operation.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    values = _tool_values(tools)
    if (mode == "off" and not prompt_cache) or values is None:
        return tools
    ordered = sorted(values, key=lambda item: (
        _tool_name(item) if isinstance(item, Mapping) else "",
        _canonical(item),
    ))
    if not prompt_cache:
        return ordered
    gated = set(gated_tool_names)
    return [item for item in ordered
            if not (isinstance(item, Mapping) and _tool_name(item) in gated)] + [
                item for item in ordered
                if isinstance(item, Mapping) and _tool_name(item) in gated]


def _contains_history(messages: Sequence[Mapping[str, object]]) -> bool:
    return any(message.get("role") in {"assistant", "tool"}
               for message in messages)


def _open_prefill(messages: Sequence[Mapping[str, object]]) \
        -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]] | None:
    """Split a frame's final open assistant continuation from its user prefix."""
    if not messages:
        return None
    final = messages[-1]
    if (final.get("role") != "assistant"
            or final.get("content") is not None
            or not isinstance(final.get("reasoning_content"), str)):
        return None
    return list(messages[:-1]), [final]


def layout_messages(messages: Sequence[Mapping[str, object]], *,
                    mode: str = "on", state_placement: str = "off",
                    prompt_cache: bool = False,
                    trailing_panels: bool = False) \
        -> list[Mapping[str, object]]:
    """Put reusable message classes before the dynamic suffix.

    A history transcript is left intact by the ordinary cache-layout arm.
    The separate trailing-panels arm keeps each assistant/tool exchange intact
    while moving task and target pins before history and coalescing user panels.
    """
    original = list(messages)
    if mode not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    if state_placement not in {"off", "end"}:
        raise ValueError("STATEFRAME_STATE_PLACEMENT must be off or end")
    if mode == "off" and not prompt_cache and not trailing_panels:
        return original
    if trailing_panels:
        # The Stage 1 order takes precedence over open-prefill placement: every
        # assistant message stays in history before the single trailing panel.
        return layout_trailing_panels(original)
    open_prefill = _open_prefill(original) if prompt_cache else None
    if open_prefill is not None:
        original, prefill = open_prefill
    else:
        prefill = []
    if prompt_cache:
        # State is the only authoritative per-turn block.  It is deliberately
        # kept as its existing user message and moved after all stable and
        # frequently-changing material, so the chat template sees a separate
        # final user message at the reusable-prefix boundary.
        state = [message for message in original
                 if _content(message).startswith("[StateFrame]\n")]
        non_state = [message for message in original
                     if not _content(message).startswith("[StateFrame]\n")]
        if _contains_history(original):
            # A frame can carry a bounded latest exchange.  Keep that
            # assistant/tool transcript byte-for-byte ordered, but still move
            # the authoritative StateFrame after its user-side material.  The
            # next model turn then sees one final user state message without
            # splitting a tool-call exchange.
            if (state and all(message.get("role") == "user" for message in state)):
                return [*non_state, *state, *prefill]
            return [*original, *prefill]
        rank = {
            "protocol": 0,
            "ticket": 1,
            "ticket_facts": 1,
            "source": 2,
            "checks": 3,
            "receipts": 3,
            "open_work": 3,
            "jobs": 3,
            "step_metadata": 3,
            "dynamic": 3,
            "history": 3,
        }
        decorated = list(enumerate(non_state))
        ordered = [message for _index, message in sorted(
            decorated,
            key=lambda item: (
                rank[layout_message_kind(item[1])],
                _source_sort_key(item[1]) if layout_message_kind(item[1]) == "source"
                else ("", "", item[0].to_bytes(8, "big")),
            ),
        )]
        # A malformed/future request with multiple state blocks remains a
        # permutation and keeps every authoritative block at the end.  The
        # normal runtime emits exactly one.
        return [*ordered, *state, *prefill]
    if _contains_history(original):
        return [*original, *prefill]
    arranged = [message for _index, message in sorted(
        enumerate(original),
        key=lambda item: (_MESSAGE_RANK[layout_message_kind(item[1])], item[0]),
    )]
    if state_placement == "end":
        state_indexes = [index for index, message in enumerate(arranged)
                         if _content(message).startswith("[StateFrame]\n")]
        if len(state_indexes) == 1 and state_indexes[0] != len(arranged) - 1:
            arranged.append(arranged.pop(state_indexes[0]))
    return arranged


def _block_bytes(value: object) -> bytes:
    return _canonical(value)


def prove_permutation(before: Sequence[object], after: Sequence[object]) -> bool:
    """Prove equal block membership as a multiset, not merely as a set."""
    return collections.Counter(_block_bytes(item) for item in before) == \
        collections.Counter(_block_bytes(item) for item in after)


@dataclasses.dataclass(frozen=True)
class CacheLayoutResult:
    """Pure layout output plus an audit trail suitable for a journal event."""

    messages: tuple[Mapping[str, object], ...]
    tools: object
    mode: str
    block_order: tuple[str, ...]
    tool_order: tuple[str, ...]
    permutation_verified: bool
    state_placement: str = "off"
    prompt_cache: bool = False
    expected_prefix_breaks: tuple[str, ...] = ()

    def audit(self) -> dict[str, object]:
        result = {
            "schema": SCHEMA,
            "mode": self.mode,
            "block_order": list(self.block_order),
            "tool_order": list(self.tool_order),
            "permutation_verified": self.permutation_verified,
            "state_placement": self.state_placement,
            "cache_enabled": False,
        }
        if self.prompt_cache:
            result.update({
                "prompt_cache": "on",
                "expected_prefix_breaks": list(self.expected_prefix_breaks),
            })
        return result


def apply_layout(messages: Sequence[Mapping[str, object]], tools: object,
                 *, mode: str | None = None,
                 state_placement: str = "off", prompt_cache: bool = False,
                 gated_tool_names: Iterable[str] = GATED_TOOL_NAMES) -> CacheLayoutResult:
    """Apply the presentation permutation, defaulting to the environment."""
    selected = cache_layout_mode() if mode is None else mode
    gated_tool_names = frozenset(gated_tool_names)
    before_messages = list(messages)
    before_tools = tools
    arranged_messages = layout_messages(
        before_messages, mode=selected, state_placement=state_placement,
        prompt_cache=prompt_cache)
    arranged_tools = layout_tools(
        before_tools, mode=selected, prompt_cache=prompt_cache,
        gated_tool_names=gated_tool_names)
    after_tools = list(arranged_tools) if isinstance(arranged_tools, Sequence) \
        and not isinstance(arranged_tools, (str, bytes, bytearray)) else arranged_tools
    verified = prove_permutation(before_messages, arranged_messages)
    if isinstance(before_tools, Sequence) and not isinstance(before_tools, (str, bytes, bytearray)) \
            and isinstance(after_tools, Sequence):
        verified = verified and prove_permutation(list(before_tools), list(after_tools))
    gated = set(gated_tool_names)
    expected_prefix_breaks = tuple(
        sorted({
            _tool_name(tool) for tool in after_tools
            if isinstance(tool, Mapping) and _tool_name(tool) in gated
        })
    ) if prompt_cache else ()
    return CacheLayoutResult(
        messages=tuple(arranged_messages), tools=after_tools, mode=selected,
        block_order=tuple(layout_message_kind(message) for message in arranged_messages),
        tool_order=tuple(
            _tool_name(tool) if isinstance(tool, Mapping) else ""
            for tool in (after_tools if isinstance(after_tools, Sequence)
                         and not isinstance(after_tools, (str, bytes, bytearray)) else ())
        ),
        permutation_verified=verified, state_placement=state_placement,
        prompt_cache=prompt_cache, expected_prefix_breaks=expected_prefix_breaks,
    )


def layout_request(messages: Sequence[Mapping[str, object]], tools: object,
                   *, mode: str | None = None,
                   state_placement: str = "off", prompt_cache: bool = False,
                   gated_tool_names: Iterable[str] = GATED_TOOL_NAMES) -> CacheLayoutResult:
    return apply_layout(
        messages, tools, mode=mode, state_placement=state_placement,
        prompt_cache=prompt_cache, gated_tool_names=gated_tool_names)


def apply(messages: Sequence[Mapping[str, object]], tools: object,
          *, mode: str | None = None,
          state_placement: str = "off", prompt_cache: bool = False,
          gated_tool_names: Iterable[str] = GATED_TOOL_NAMES) -> CacheLayoutResult:
    return apply_layout(
        messages, tools, mode=mode, state_placement=state_placement,
        prompt_cache=prompt_cache, gated_tool_names=gated_tool_names)


def _prefix_bytes(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    index = 0
    # Byte comparison is intentional: it measures exactly the deterministic
    # serialized prefix without pretending to know the provider tokenizer.
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _proxy_tokens(byte_count: int) -> int:
    return (byte_count + 3) // 4


def _pair_metric(left: bytes, right: bytes) -> dict[str, int | float]:
    common_bytes = _prefix_bytes(left, right)
    common_tokens = _proxy_tokens(common_bytes)
    total_tokens = _proxy_tokens(len(right))
    return {
        "longest_common_prefix_bytes": common_bytes,
        "longest_common_prefix_tokens": common_tokens,
        "request_tokens": total_tokens,
        "reusable_fraction": (common_tokens / total_tokens if total_tokens else 1.0),
    }


def _aggregate(metrics: Sequence[Mapping[str, int | float]]) -> dict[str, int | float]:
    common_bytes = sum(int(row["longest_common_prefix_bytes"]) for row in metrics)
    common_tokens = sum(int(row["longest_common_prefix_tokens"]) for row in metrics)
    request_tokens = sum(int(row["request_tokens"]) for row in metrics)
    fraction = common_tokens / request_tokens if request_tokens else 1.0
    return {
        "longest_common_prefix_bytes": common_bytes,
        "longest_common_prefix_tokens": common_tokens,
        "request_tokens": request_tokens,
        "token_weighted_reusable_fraction": fraction,
        # Friendly alias for reports and callers that use the shorter name.
        "reusable_fraction": fraction,
    }


def _event_request(event: Mapping[str, object]) -> tuple[list[Mapping[str, object]], object]:
    raw_messages = event.get("messages", ())
    messages = list(raw_messages) if isinstance(raw_messages, Sequence) \
        and not isinstance(raw_messages, (str, bytes, bytearray)) else []
    return messages, event.get("tools")


def measure_events(events: Sequence[Mapping[str, object]], *,
                   prompt_cache: bool = False) -> dict[str, object]:
    """Compare consecutive ``request_prepared`` events before and after layout.

    The denominator is the serialized proxy-token weight of the later request
    in each pair.  Aggregating numerator and denominator first gives a
    token-weighted reusable fraction rather than an unweighted mean of pairs.
    """
    requests = [event for event in events
                if isinstance(event, Mapping) and event.get("kind") == "request_prepared"]
    before_metrics: list[dict[str, int | float]] = []
    after_metrics: list[dict[str, int | float]] = []
    permutation_ok = True
    for previous, current in zip(requests, requests[1:]):
        previous_messages, previous_tools = _event_request(previous)
        current_messages, current_tools = _event_request(current)
        before_metrics.append(_pair_metric(
            canonical_request_bytes(previous_messages, previous_tools),
            canonical_request_bytes(current_messages, current_tools)))
        previous_layout = apply_layout(
            previous_messages, previous_tools, mode="on", prompt_cache=prompt_cache)
        current_layout = apply_layout(
            current_messages, current_tools, mode="on", prompt_cache=prompt_cache)
        permutation_ok = permutation_ok and previous_layout.permutation_verified \
            and current_layout.permutation_verified
        after_metrics.append(_pair_metric(
            canonical_request_bytes(previous_layout.messages, previous_layout.tools),
            canonical_request_bytes(current_layout.messages, current_layout.tools)))
    result = {
        "schema": SCHEMA,
        "pair_count": len(before_metrics),
        "permutation_verified": permutation_ok,
        "before": _aggregate(before_metrics),
        "after": _aggregate(after_metrics),
        "pairs": [
            {"before": before, "after": after}
            for before, after in zip(before_metrics, after_metrics)
        ],
    }
    if prompt_cache:
        result["prompt_cache"] = "on"
    return result


def measure_event_file(path: str | os.PathLike[str], *,
                       prompt_cache: bool = False) -> dict[str, object]:
    events = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    result = measure_events(events, prompt_cache=prompt_cache)
    return {"path": str(path), **result}


def measure_event_files(paths: Sequence[str | os.PathLike[str]] | str | os.PathLike[str], *,
                        prompt_cache: bool = False) -> dict[str, object]:
    """Measure one or more JSONL journal paths, accepting a shell-like glob."""
    if isinstance(paths, (str, os.PathLike)):
        raw = str(paths)
        expanded = sorted(glob.glob(raw)) if any(char in raw for char in "*?[") else [raw]
    else:
        expanded = [str(path) for path in paths]
    reports = [measure_event_file(path, prompt_cache=prompt_cache) for path in expanded]
    before = _aggregate([report["before"] for report in reports])
    after = _aggregate([report["after"] for report in reports])
    result = {
        "schema": SCHEMA,
        "files": reports,
        "file_count": len(reports),
        "pair_count": sum(int(report["pair_count"]) for report in reports),
        "permutation_verified": all(report["permutation_verified"] for report in reports),
        "before": before,
        "after": after,
    }
    if prompt_cache:
        result["prompt_cache"] = "on"
    return result


def measure(paths: Sequence[str | os.PathLike[str]] | str | os.PathLike[str], *,
            prompt_cache: bool = False) -> dict[str, object]:
    return measure_event_files(paths, prompt_cache=prompt_cache)


__all__ = [
    "CacheLayoutResult", "GATED_TOOL_NAMES", "MODE_ENV", "SCHEMA", "TRAILING_PANELS_MODE_ENV",
    "apply", "apply_layout", "block_kind", "cache_layout_mode", "canonical_request_bytes",
    "configuration", "enabled", "history_messages", "history_prefix_audit",
    "layout_message_kind", "layout_messages", "layout_request", "layout_trailing_panels",
    "layout_tools", "measure", "measure_event_file", "measure_event_files", "measure_events",
    "mode", "observe_history_prefix", "prove_permutation", "track_tool_menu",
    "trailing_panels_audit", "trailing_panels_enabled", "trailing_panels_mode",
]
