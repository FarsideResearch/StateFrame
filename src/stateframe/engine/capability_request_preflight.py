"""Study-only payload checks for history and StateFrame comparisons."""
from __future__ import annotations

import copy
import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path

from kernel_context import token_estimate
from kernel_frame_request_allocation import FrameRequestAllocation
from kernel_paged_source import PAGED_SOURCE_PREFIX, validate_descriptor
from kernel_read_fixes import SOURCE_BLOCK_PREFIXES, plain_source_labels_enabled, source_header
from kernel_request_source_window import (
    bound_runtime_sources,
    is_source_window,
    verify_model_source_window,
    verify_window,
    verify_window_projection,
)
from kernel_trajectory_frame import trajectory_frame_mode
from kernel_lean_presentation import (
    _closed_exchanges,
    _prefix_count,
    exchange_limit,
    select_recent_exchanges,
    source_token_limit,
    token_limit,
    truth_tail_enabled,
)

PRECOMPOSED_HOST_TEXT_SCHEMA = "release-precomposed-host-text/1"


def _tool_name(tool: object) -> str:
    if not isinstance(tool, dict):
        return ""
    direct = tool.get("name")
    if direct:
        return str(direct)
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    return ""


def _lean_menu(tail: str) -> tuple[str, ...]:
    first_line = next(
        (line for line in tail.splitlines()
         if line.startswith("[LEAN TRUTH TAIL] check_status=")),
        "",
    )
    marker = "; menu="
    if marker not in first_line:
        raise ValueError("lean presentation truth tail requires the final tool menu")
    status, raw_menu = first_line.split(marker, 1)
    if not status.removeprefix("[LEAN TRUTH TAIL] check_status=").strip():
        raise ValueError("lean presentation truth tail requires check status")
    names = tuple(name.strip() for name in raw_menu.split(",") if name.strip())
    if not names or len(names) != len(set(names)):
        raise ValueError("lean presentation truth tail requires a unique owner menu")
    return names


def _validate_lean_window(messages: list[dict], environment: dict,
                          ignored_prefixes: tuple[str, ...] = ()) -> tuple[int, int]:
    selected_messages, selected_exchanges = _selected_lean_window(
        messages, environment, ignored_prefixes,
    )
    return selected_exchanges, len(selected_messages)


def _selected_lean_window(
        messages: list[dict], environment: dict,
        ignored_prefixes: tuple[str, ...] = (),
) -> tuple[tuple[dict, ...], int]:
    max_exchanges = exchange_limit(environment)
    max_tokens = token_limit(environment)
    window_messages = [
        message for message in messages
        if not any(str(message.get("content") or "").startswith(prefix)
                   for prefix in ignored_prefixes)
    ]
    prefix_count = _prefix_count(window_messages)
    candidates = _closed_exchanges(window_messages, prefix_count)
    selected = select_recent_exchanges(
        window_messages,
        max_exchanges=max_exchanges,
        max_tokens=max_tokens,
    )
    selected_messages = tuple(
        message
        for exchange in selected
        for message in exchange.messages
    )
    actual_window = tuple(window_messages[prefix_count:])
    if actual_window != selected_messages:
        raise ValueError(
            "lean presentation recent-exchange window is not the newest complete bounded window"
        )
    if len(candidates) > max_exchanges:
        raise ValueError("lean presentation recent-exchange window exceeds K")
    return selected_messages, len(selected)


def _presentation_parts(messages: list[dict]) -> dict[str, list[object]]:
    """Classify presentation-bearing messages for checking and trimming."""
    contents = [message.get("content") or "" for message in messages]
    frames = [text for text in contents if text.startswith("[StateFrame]\n")]
    working_prefix = ("[Current source: " if plain_source_labels_enabled()
                      else "[Working source: ")
    working = [text for text in contents if text.startswith(working_prefix)]
    paged_working = [text for text in contents if text.startswith(PAGED_SOURCE_PREFIX)]
    working.extend(paged_working)
    source_section = [text for text in contents if text.startswith("[WORKING SOURCE]")]
    source_section.extend(paged_working)
    source_blocks = [text for text in contents
                     if text.startswith(SOURCE_BLOCK_PREFIXES)
                     or text.startswith("CURRENT_SOURCE ")
                     or text.startswith(PAGED_SOURCE_PREFIX)]
    source_windows = [text for text in working if is_source_window(text)]
    return {
        "contents": contents,
        "frames": frames,
        "working": working,
        "paged_working": paged_working,
        "source_section": source_section,
        "source_blocks": source_blocks,
        "source_windows": source_windows,
    }


def _framefit_target_source_deferred(
        messages: list[dict], *, path: str, source: str, tools: object,
        runtime: object, environment: Mapping[str, object],
) -> bool:
    if getattr(runtime, "stateframe_framefit_mode", "off") != "on":
        from kernel_stateframe_f4_fit import framefit_mode
        if framefit_mode(environment) != "on":
            return False
    from kernel_stateframe_f4_fit import is_valid_deferred_target_source_message
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        if is_valid_deferred_target_source_message(
                message.get("content"), path=path, source=source, tools=tools):
            return True
    return False


def _expected_working_source(runtime: object, environment: Mapping[str, object]) -> str:
    target = getattr(runtime, "target", None)
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    if target is None or root is None:
        return ""
    path = Path(target).resolve().relative_to(Path(root).resolve()).as_posix()
    source = Path(target).read_text()
    expected = source_header(path, role="working") + "\n"
    if environment.get("STATEFRAME_WORKING_SOURCE_SCOPE", "0") == "1":
        lines = len(source.splitlines())
        expected += f"Coverage: COMPLETE FILE; lines {1 if lines else 0}-{lines} of {lines}; nothing omitted.\n"
    return expected + source


def _history_target_source_is_current(
        messages: list[dict], runtime: object, path: str,
        environment: Mapping[str, object],
) -> bool:
    """Accept one history-arm target only when its bytes match the current workspace."""
    target = getattr(runtime, "target", None)
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    if target is None or root is None or not path:
        return False
    try:
        target_file = Path(target)
        target_path = target_file.resolve().relative_to(Path(root).resolve()).as_posix()
        if target_path != path:
            return False
        source = target_file.read_text(encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeError):
        return False

    parts = _presentation_parts(messages)
    if parts["frames"] or len(parts["source_blocks"]) != 1:
        return False
    content = parts["source_blocks"][0]
    if not isinstance(content, str):
        return False
    header = content.splitlines()[0] if content.splitlines() else ""
    if header not in {f"[Current source: {path}]", f"[Working source: {path}]"}:
        return False
    matching_messages = [message for message in messages
                         if message.get("content") == content]
    if len(matching_messages) != 1 or matching_messages[0].get("role") != "user":
        return False
    try:
        from kernel_lean import wide_target_window_stack_enabled
        from kernel_request_source_window import C5_TARGET_CONTEXT_LINES, bound_source_message

        expected = bound_source_message(
            header, path=path, source=source, runtime=runtime,
            max_chars=max(512, 12000 - len((header + "\n").encode("utf-8"))),
            is_target=True,
            context_lines=(C5_TARGET_CONTEXT_LINES
                           if wide_target_window_stack_enabled() else None),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, UnicodeError):
        return False
    return content == expected


def _unique_message_objects(messages: list[dict]) -> tuple[dict, ...]:
    unique: list[dict] = []
    seen: set[int] = set()
    for message in messages:
        identity = id(message)
        if identity not in seen:
            seen.add(identity)
            unique.append(message)
    return tuple(unique)


def presentation_protected_messages(
        messages: list[dict], spec: dict, runtime: object,
        tools: list[dict] | tuple[dict, ...] | None = None,
) -> tuple[dict, ...]:
    """Return messages shaping must retain for ``check_presentation`` validity."""
    del tools  # Menu equality is checked after shaping; menu text itself is protected below.
    arm = spec.get("presentation_arm", "frame")
    environment = spec.get("environment", {})
    environment = environment if isinstance(environment, dict) else {}
    if (arm == "history" and getattr(runtime, "stateframe_hybrid_mode", "off") == "on"
            and getattr(runtime, "_hybrid_crossed", False)):
        arm = "frame"
    parts = _presentation_parts(messages)
    protected: list[dict] = []

    def add_matching(predicate) -> None:
        protected.extend(message for message in messages if predicate(message))

    from kernel_stateframe_framefit5 import has_framefit5_state
    if (getattr(runtime, "stateframe_framefit_mode", "off") == "on"
            or has_framefit5_state(messages)):
        protected.extend(messages[:2])
        add_matching(lambda message: str(message.get("content") or "").startswith(
            "[StateFrame]\n"))
        return _unique_message_objects(protected)

    lean_enabled = (
        getattr(runtime, "stateframe_lean_presentation_mode", "off") == "on"
        or str(environment.get("STATEFRAME_LEAN_PRESENTATION", "off")).strip().lower() == "on"
    )
    if lean_enabled:
        add_matching(lambda message: "[LEAN TRUTH TAIL] check_status=" in str(
            message.get("content") or ""))
        try:
            selected, _ = _selected_lean_window(
                messages,
                environment,
                ignored_prefixes=("[StateFrame]\n", "[WORKING SOURCE]", PAGED_SOURCE_PREFIX,
                                  *SOURCE_BLOCK_PREFIXES, "CURRENT_SOURCE "),
            )
        except ValueError:
            selected = ()
        protected.extend(selected)
        return _unique_message_objects(protected)
    if arm == "history":
        return ()

    trajectory_environment = dict(environment)
    if "stateframe_trajectory_frame" in spec:
        trajectory_environment["STATEFRAME_TRAJECTORY_FRAME"] = spec[
            "stateframe_trajectory_frame"]
    trajectory_frame = trajectory_frame_mode(trajectory_environment)
    if trajectory_frame != "off":
        protected.extend(messages[:2])
        state_indexes = [index for index, text in enumerate(parts["contents"])
                         if text.startswith("[StateFrame]\n")]
        if state_indexes and trajectory_frame in {"handoff_host", "handoff_model"}:
            state_index = state_indexes[0]
            protected.extend(
                message for message in messages[2:state_index]
                if message.get("role") == "assistant"
                and isinstance(message.get("reasoning_content"), str)
                and message.get("reasoning_content")
            )

    expected = _expected_working_source(runtime, environment)
    paged_working = parts["paged_working"]
    source_windows = parts["source_windows"]
    if paged_working:
        add_matching(lambda message: str(message.get("content") or "") in paged_working)
    elif source_windows:
        add_matching(lambda message: str(message.get("content") or "") in source_windows)
    elif expected:
        add_matching(lambda message: str(message.get("content") or "") == expected)
    else:
        # Keep malformed source visible until final preflight reports its exact failure.
        add_matching(lambda message: str(message.get("content") or "") in parts["working"])

    prefill_mode = spec.get(
        "stateframe_think_prefill",
        environment.get("STATEFRAME_THINK_PREFILL", "off"),
    )
    if prefill_mode == "on":
        add_matching(lambda message: message.get("role") == "assistant"
                     and isinstance(message.get("reasoning_content"), str)
                     and bool(message.get("reasoning_content")))
    return _unique_message_objects(protected)


def check_presentation(messages: list[dict], spec: dict, runtime,
                       tools: list[dict] | tuple[dict, ...] | None = None, *,
                       history_source_runtime: object | None = None,
                       frame_request_allocation: FrameRequestAllocation | None = None) -> dict:
    arm = spec.get("presentation_arm", "frame")
    if arm not in {"history", "frame"}:
        raise ValueError("unknown study presentation arm")
    from kernel_stateframe_framefit5 import has_framefit5_state, validate_product_request
    if (getattr(runtime, "stateframe_framefit_mode", "off") == "on"
            or has_framefit5_state(messages)):
        return validate_product_request(
            messages, runtime, tools=tools,
            refresh_headcarry=frame_request_allocation is None,
            frame_request_allocation=frame_request_allocation,
        )
    if (arm == "history" and getattr(runtime, "stateframe_hybrid_mode", "off") == "on"
            and getattr(runtime, "_hybrid_crossed", False)):
        # STATEFRAME_HYBRID_COMPACTION: a history arm that crossed its context threshold is
        # latched to the F4 frame for the rest of the episode; check it as a frame request.
        arm = "frame"
    environment = spec.get("environment", {})
    environment = environment if isinstance(environment, dict) else {}
    lean_enabled = (
        getattr(runtime, "stateframe_lean_presentation_mode", "off") == "on"
        or str(environment.get("STATEFRAME_LEAN_PRESENTATION", "off")).strip().lower() == "on"
    )
    parts = _presentation_parts(messages)
    contents = parts["contents"]
    frames = parts["frames"]
    working = parts["working"]
    paged_working = parts["paged_working"]
    source_windows = parts["source_windows"]
    source_section = parts["source_section"]
    source_blocks = parts["source_blocks"]
    roles = [message["role"] for message in messages]
    lean_contract: dict[str, object] | None = None
    if lean_enabled:
        if not truth_tail_enabled(environment):
            raise ValueError("lean presentation requires the truth tail contract")
        if len(frames) > 1:
            raise ValueError("lean presentation permits at most one StateFrame")
        tails = [text for text in contents if "[LEAN TRUTH TAIL] check_status=" in text]
        if len(tails) != 1:
            raise ValueError("lean presentation requires exactly one truth tail")
        tail = tails[0]
        menu = _lean_menu(tail)
        if tools is not None:
            owner_menu = tuple(name for name in (_tool_name(tool) for tool in tools) if name)
            if menu != owner_menu:
                raise ValueError("lean presentation truth tail menu differs from owner menu")
        recent_exchanges, recent_messages = _validate_lean_window(
            messages,
            environment,
            ignored_prefixes=("[StateFrame]\n", "[WORKING SOURCE]", PAGED_SOURCE_PREFIX,
                              *SOURCE_BLOCK_PREFIXES,
                              "CURRENT_SOURCE "),
        )
        if source_blocks and not source_section:
            raise ValueError("lean presentation source blocks require a WORKING SOURCE section")
        if source_section:
            source_messages = [
                message for message in messages
                if str(message.get("content") or "").startswith(
                    ("[WORKING SOURCE]", PAGED_SOURCE_PREFIX, *SOURCE_BLOCK_PREFIXES,
                     "CURRENT_SOURCE "))
            ]
            source_tokens = token_estimate({"messages": source_messages})
            if source_tokens > source_token_limit(environment):
                raise ValueError("lean presentation WORKING SOURCE section exceeds source budget")
        lean_contract = {
            "checked_before_provider": True,
            "presentation_arm": "lean",
            "message_count": len(messages),
            "history_roles": roles,
            "automatic_frame_present": bool(frames),
            "automatic_working_source_present": bool(source_section and source_blocks),
            "lean_working_source_required": bool(source_section),
            "lean_truth_tail_present": True,
            "lean_menu": ",".join(menu),
            "lean_recent_exchange_window_valid": True,
            "lean_recent_exchange_limit": exchange_limit(environment),
            "lean_recent_exchange_token_limit": token_limit(environment),
            "lean_recent_exchanges": recent_exchanges,
            "lean_recent_exchange_messages": recent_messages,
        }
        # Lean owns its bounded source section; it is not required to carry the
        # ordinary StateFrame arm's separate frame/source pairing.
        return lean_contract
    if arm == "history":
        if frames:
            raise ValueError("history arm received automatic StateFrame delivery")
        history_source_path = spec.get("history_current_target_source_path")
        history_has_source = bool(source_blocks or working)
        if (history_has_source or history_source_path is not None) and (
                not isinstance(history_source_path, str)
                or not _history_target_source_is_current(
                    messages, history_source_runtime, history_source_path, environment,
                )):
            raise ValueError(
                "history arm source delivery requires byte-proven current target bytes"
            )
        history_environment = spec.get("environment", {})
        history_environment = history_environment if isinstance(history_environment, dict) else {}
        if spec.get("stateframe_trajectory_frame",
                   history_environment.get("STATEFRAME_TRAJECTORY_FRAME", "off")) != "off":
            raise ValueError("trajectory frame requires the frame presentation arm")
        return {"checked_before_provider": True, "presentation_arm": arm,
                "message_count": len(messages), "history_roles": roles,
                "automatic_frame_present": False,
                "automatic_working_source_present": history_has_source,
                "history_target_source_byte_proven": history_has_source}
    trajectory_environment = dict(environment)
    if "stateframe_trajectory_frame" in spec:
        trajectory_environment["STATEFRAME_TRAJECTORY_FRAME"] = spec[
            "stateframe_trajectory_frame"]
    trajectory_frame = trajectory_frame_mode(trajectory_environment)
    if trajectory_frame != "off":
        # A caller may append the exact restored source while probing the
        # already-composed screen messages.  Treat identical copies as one
        # effective source, but still reject divergent or missing copies.
        distinct_working = list(dict.fromkeys(working))
        working = distinct_working
    # With STATEFRAME_SOURCE_LABELS=plain every source block (working, retained, observed) shares the
    # "[Current source: " header, so several are legitimate once the model has read more than one file (r35c full
    # smoke died at step 2 on the old exactly-one rule). Legacy labels keep the exactly-one working source rule.
    working_ok = len(working) >= 1 if plain_source_labels_enabled() else len(working) == 1
    if len(frames) != 1 or not working_ok:
        raise ValueError("frame arm requires one StateFrame and working source "
                         f"(frames={len(frames)}, source_blocks={len(working)}, "
                         f"plain_labels={plain_source_labels_enabled()})")
    if trajectory_frame != "off" and arm != "frame":
        raise ValueError("trajectory frame requires the frame presentation arm")
    state_indexes = [index for index, text in enumerate(contents)
                     if text.startswith("[StateFrame]\n")]
    trajectory_reasoning_ids: set[int] = set()
    if trajectory_frame != "off":
        if len(state_indexes) != 1:
            raise ValueError("trajectory frame requires exactly one StateFrame")
        state_index = state_indexes[0]
        if state_index < 2 or [message.get("role") for message in messages[:2]] != [
                "assistant", "tool"]:
            raise ValueError("trajectory frame must begin with the recorded assistant/tool pair")
        assistant = messages[0]
        if assistant.get("reasoning_content"):
            trajectory_reasoning_ids.add(id(assistant))
        if trajectory_frame == "replay" and assistant.get("reasoning_content"):
            raise ValueError("replay trajectory must not contain forged reasoning")
        if trajectory_frame == "pre_thought" and not assistant.get("reasoning_content"):
            raise ValueError("pre_thought trajectory requires host reasoning on the historical call")
        if trajectory_frame in {"handoff_host", "handoff_model"}:
            handoffs = [message for message in messages[2:state_index]
                        if message.get("role") == "assistant"
                        and isinstance(message.get("reasoning_content"), str)
                        and message.get("reasoning_content")]
            if len(handoffs) != 1:
                raise ValueError("handoff trajectory requires one synthetic host reasoning turn")
            trajectory_reasoning_ids.add(id(handoffs[0]))
    else:
        state_index = state_indexes[0] if len(state_indexes) == 1 else -1
    placement = spec.get("stateframe_state_placement",
                         environment.get("STATEFRAME_STATE_PLACEMENT", "off"))
    if placement not in {"off", "end"}:
        raise ValueError("unknown STATEFRAME_STATE_PLACEMENT arm")
    prefill_mode = spec.get("stateframe_think_prefill",
                            environment.get("STATEFRAME_THINK_PREFILL", "off"))
    if prefill_mode not in {"off", "on", "user", "self_plan"}:
        raise ValueError("unknown STATEFRAME_THINK_PREFILL arm")
    assistant_prefills = [message for message in messages
                          if message.get("role") == "assistant"
                          and isinstance(message.get("reasoning_content"), str)
                          and message.get("reasoning_content")
                          and id(message) not in trajectory_reasoning_ids]
    if prefill_mode == "on":
        if len(assistant_prefills) != 1 or messages[-1] is not assistant_prefills[0]:
            raise ValueError("true think prefill requires one final assistant reasoning continuation")
        if "</think>" in assistant_prefills[0]["reasoning_content"]:
            raise ValueError("true think prefill must not close the assistant think block")
    elif assistant_prefills:
        raise ValueError("assistant reasoning prefill is present outside the true-prefill arm")
    expected_state_index = len(contents) - 2 if prefill_mode == "on" else len(contents) - 1
    if placement == "end" and state_indexes != [expected_state_index]:
        raise ValueError("end state placement requires StateFrame immediately before generation")
    path = runtime.target.resolve().relative_to(Path(runtime.owner.root).resolve()).as_posix()
    source = runtime.target.read_text()
    enabled = environment.get("STATEFRAME_WORKING_SOURCE_SCOPE", "0") == "1"
    lines = len(source.splitlines())
    expected = source_header(path, role="working") + "\n"
    if enabled:
        expected += f"Coverage: COMPLETE FILE; lines {1 if lines else 0}-{lines} of {lines}; nothing omitted.\n"
    expected += source
    # Plain labels: the target's block is one of several "[Current source: ...]" blocks, not necessarily the first.
    target_present = False
    if paged_working:
        if len(paged_working) != 1:
            raise ValueError("frame arm requires exactly one paged working source descriptor")
        validate_descriptor(paged_working[0], path=path, source=source)
        target_present = True
    else:
        target_present = expected in working if plain_source_labels_enabled() else working[0] == expected
    bounded_present = False
    if source_windows and getattr(runtime, "target", None) is not None:
        try:
            bounded_present = any(
                verify_window(text, path=path, source=source)
                or (source.startswith(f"{text.splitlines()[1]}\n")
                    and verify_window_projection(text, path=path))
                for text in source_windows
            )
        except (OSError, TypeError, ValueError):
            bounded_present = False
    if not target_present and not bounded_present and getattr(runtime, "target", None) is not None:
        # The final provider boundary projects verified windows into the
        # compact model form; its displayed ranges are checked against the
        # current workspace bytes here, never accepted by marker alone.
        bounded_present = any(
            verify_model_source_window(text, path=path, source=source) for text in working)
    deferred_target = False
    if not target_present and not bounded_present:
        deferred_target = _framefit_target_source_deferred(
            messages, path=path, source=source, tools=tools, runtime=runtime,
            environment=environment,
        )
        target_present = deferred_target
    if (not target_present
            and not bounded_present
            and getattr(runtime, "target", None) is not None
            and os.environ.get("STATEFRAME_REAL04_WATCHDOG_SPAN", "off").strip().lower() != "off"):
        from kernel_real04_bugs import switch_enabled, verify_bounded_window
        # Fix round (rv_real04_bugs_REPORT.md finding 2): a bounded window is
        # only accepted once its declared span digest is verified against
        # CURRENT workspace bytes -- a marker/path/reference substring alone
        # let a caller splice in stale or arbitrary text.
        bounded_present = switch_enabled("STATEFRAME_REAL04_WATCHDOG_SPAN") and any(
            verify_bounded_window(text, path=path, source=source) for text in working)
    if not target_present and not bounded_present:
        raise ValueError("working source differs from current workspace bytes")
    result = {"checked_before_provider": True, "presentation_arm": arm, "scope_enabled": enabled,
            "state_block_is_last": state_indexes == [len(contents) - 1],
            "state_block_before_generation": state_indexes == [expected_state_index],
            "think_prefill_mode": prefill_mode,
            "self_plan_pending": prefill_mode == "self_plan",
            "assistant_prefill_present": bool(assistant_prefills),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "source_lines": lines,
            "message_sha256": hashlib.sha256(working[0].encode()).hexdigest()}
    if deferred_target:
        result["target_source_deferred"] = True
    if trajectory_frame != "off":
        result.update({
            "trajectory_frame": trajectory_frame,
            "trajectory_pair_before_state": state_indexes == [state_index]
            and messages[0].get("role") == "assistant"
            and messages[1].get("role") == "tool",
            "trajectory_host_reasoning_present": bool(trajectory_reasoning_ids),
        })
    if lean_contract is not None:
        result.update(lean_contract)
    return result


def _hostfact_truthful_enabled(runtime: object | None = None) -> bool:
    value = getattr(runtime, "stateframe_hostfact_truthful_mode", None)
    if value is None:
        value = os.environ.get("STATEFRAME_HOSTFACT_TRUTHFUL", "off")
    if value not in {"off", "on"}:
        raise ValueError("STATEFRAME_HOSTFACT_TRUTHFUL must be off or on")
    return value == "on"


def _frame_degradation_reason(error: Exception, runtime: object) -> str:
    """Prefer the host's measured frame-fit failure over a secondary preflight error."""
    truthful = _hostfact_truthful_enabled(runtime)
    manifest = getattr(runtime, "_f4_last_fit_manifest", None)
    if isinstance(manifest, Mapping) and manifest.get("fits") is False:
        if truthful:
            return "the composed F4 StateFrame did not fit its configured request ceiling"
        final_tokens = manifest.get("final_tokens")
        ceiling_tokens = manifest.get("ceiling_tokens")
        if type(final_tokens) is int and type(ceiling_tokens) is int:
            return (
                "the composed F4 StateFrame exceeded its request ceiling after fitting "
                f"({final_tokens:,} tokens > {ceiling_tokens:,} tokens)"
            )
        return "the composed F4 StateFrame did not fit its configured request ceiling"
    if truthful:
        return "presentation preflight rejected the composed frame"

    detail = " ".join(str(error).split())[:240]
    if detail:
        return f"presentation preflight rejected the composed frame ({detail})"
    return f"presentation preflight rejected the composed frame ({type(error).__name__})"


def _refresh_current_target_source(
    messages: list[dict], runtime: object, *, append_if_missing: bool = False,
) -> str | None:
    """Refresh a target block and return its path only after byte proof succeeds."""
    target = getattr(runtime, "target", None)
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    if target is None or root is None:
        return None
    try:
        target_path = Path(target).resolve().relative_to(Path(root).resolve()).as_posix()
        source = Path(target).read_text(encoding="utf-8")
        from kernel_lean import wide_target_window_stack_enabled
        from kernel_request_source_window import (
            C5_TARGET_CONTEXT_LINES, bound_source_message, source_revision,
            verify_current_source_message,
        )
        from kernel_read_fixes import source_header

        source_revision_value = source_revision(source)
        current_matches = [
            index for index, message in enumerate(messages)
            if isinstance(message, Mapping)
            and isinstance(message.get("content"), str)
            and (f"[Current source: {target_path}]" in str(message["content"])
                 or f"[Working source: {target_path}]" in str(message["content"])
                 or (str(message["content"]).startswith(PAGED_SOURCE_PREFIX)
                     and re.search(
                         r'"path"\s*:\s*"' + re.escape(target_path) + r'"',
                         str(message["content"]),
                     )))
        ]
        if not current_matches and not append_if_missing:
            return None
        index = current_matches[-1] if current_matches else None
        first_line = source_header(target_path, role="working")
        refreshed = bound_source_message(
            first_line, path=target_path, source=source, runtime=runtime,
            max_chars=max(512, 12000 - len((first_line + "\n").encode("utf-8"))),
            is_target=True,
            context_lines=(C5_TARGET_CONTEXT_LINES
                           if wide_target_window_stack_enabled() else None),
        )
        if not verify_current_source_message(
                refreshed, path=target_path, source=source):
            return None
        if index is None:
            insertion_index = next(
                (row_index + 1 for row_index, message in enumerate(messages)
                 if message.get("role") == "user"),
                len(messages),
            )
            messages.insert(insertion_index, {"role": "user", "content": refreshed})
        else:
            messages[index] = {**dict(messages[index]), "content": refreshed}
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit(
                "preflight_source_refreshed", producer="capability_request_preflight",
                path=target_path, revision=source_revision_value,
                byte_proof=True, replaced_messages=1 if index is not None else 0,
                appended_messages=1 if index is None else 0, dispatch=False,
            )
        return target_path
    except (OSError, RuntimeError, TypeError, ValueError, UnicodeError):
        return None


def _assemble_framefit5_host_text(
        prepared: list[dict], source_messages: list[dict], runtime: object,
        *, include_editready_fact: bool,
        frame_request_allocation: FrameRequestAllocation | None = None,
        framefit5_inputs: Mapping[str, object] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    """Move generated host facts to [V], then use the shared frame assembler."""
    from kernel_stateframe_framefit5 import (
        FrameFit5Error, VOLATILE_TAIL_MARKER,
        _is_framefit5_host_context_panel, assemble_framefit5_request,
    )
    from kernel_editready_fact import _is_generated_editready_fact

    prepared = [copy.deepcopy(dict(message)) for message in prepared]
    editready_facts = [
        message for message in prepared
        if include_editready_fact and _is_generated_editready_fact(message)
    ]
    if len(editready_facts) > 1:
        raise FrameFit5Error(
            "host text transform produced multiple edit-ready facts"
        )
    if editready_facts:
        host_fact = str(editready_facts[0]["content"])
        prepared = [
            message for message in prepared
            if not _is_generated_editready_fact(message)
        ]
        tail_indexes = [
            index for index, message in enumerate(prepared)
            if message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and str(message["content"]).startswith(VOLATILE_TAIL_MARKER)
        ]
        if len(tail_indexes) > 1:
            raise FrameFit5Error(
                "host text transform found multiple volatile tails"
            )
        if tail_indexes:
            index = tail_indexes[0]
            tail_content = prepared[index].get("content")
            if not isinstance(tail_content, str):
                raise FrameFit5Error(
                    "host text transform found an invalid volatile tail"
                )
            separator = "" if tail_content.endswith(("\n", "\r")) else "\n"
            prepared[index]["content"] = tail_content + separator + host_fact
        else:
            prepared.append({
                "role": "user",
                "content": f"{VOLATILE_TAIL_MARKER}\n{host_fact}",
            })

    host_text_messages = [
        copy.deepcopy(dict(message)) for message in prepared
        if isinstance(message, Mapping)
        and _is_framefit5_host_context_panel(message)
    ]
    # Pre-allocation host composition must retain the same source authority and
    # semantic suffix used by the runtime's immediately preceding assembly.
    # Rediscovering them from the projected body can lose deferred-source proof.
    inputs = dict(framefit5_inputs or {})
    semantic_messages = inputs.pop("source_messages", source_messages)
    assembled = assemble_framefit5_request(
        prepared, semantic_messages, runtime=runtime,
        host_text_messages=host_text_messages,
        frame_request_allocation=frame_request_allocation,
        **inputs,
    )
    return assembled, {
        "framefit5_assembly": {
            "owner": "assemble_framefit5_request",
            "host_text_slot": "volatile_tail",
        },
    }


def _apply_request_host_text(
        messages: list[dict], runtime: object,
        *, tools: list[dict] | tuple[dict, ...] | None = None,
        frame_request_allocation: FrameRequestAllocation | None = None,
        framefit5_inputs: Mapping[str, object] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    """Apply enabled host text, then route FrameFit5 requests through its assembler."""
    from kernel_stateframe_framefit5 import has_framefit5_state

    framefit5 = (
        getattr(runtime, "stateframe_framefit_mode", "off") == "on"
        or has_framefit5_state(messages)
    )
    prepared = messages
    receipt: dict[str, object] = {}

    from kernel_editready_fact import apply_editready_fact, enabled as editready_enabled
    editready_active = editready_enabled()
    if editready_active:
        prepared, editready_receipt = apply_editready_fact(prepared, runtime)
        receipt["editready_fact"] = editready_receipt

    from kernel_cache_layout import (
        layout_messages, trailing_panels_mode, trailing_panels_audit,
    )
    if trailing_panels_mode() == "on":
        if framefit5:
            # Generic panel layout can merge the StateFrame JSON with user panels.
            # FrameFit5 owns this placement and keeps any host fact inside [V].
            receipt["trailing_panels"] = {
                "status": "delegated_to_framefit5_assembler",
            }
        else:
            before = list(prepared)
            prepared = list(layout_messages(
                before, mode="off", trailing_panels=True,
            ))
            receipt["trailing_panels"] = trailing_panels_audit(before, prepared)

    if framefit5:
        prepared = [copy.deepcopy(dict(message)) for message in prepared]
        tool_names = tuple(
            name for tool in tools or ()
            if (name := _tool_name(tool))
        )
        if tool_names:
            from kernel_stateframe_framefit5 import (
                FrameFit5Error, _replace_stale_host_menu,
            )

            route = (
                "submit"
                if "done" in tool_names and set(tool_names) <= {"done", "read"}
                else "repair"
            )
            current_menu = (
                f"[Current admitted host menu: route={route}; "
                f"available actions: {', '.join(tool_names)}.]"
            )
            for message in prepared:
                content = message.get("content")
                if (message.get("role") == "user"
                        and isinstance(content, str)
                        and content.startswith("[COMPLETION OWNER]")):
                    updated = _replace_stale_host_menu(
                        content, current_menu,
                    )
                    if updated == content:
                        menu_start = content.rfind("[Current admitted host menu:")
                        if menu_start >= 0:
                            menu_end = content.find("]", menu_start)
                            if menu_end < 0:
                                raise FrameFit5Error(
                                    "completion owner host menu is malformed"
                                )
                            updated = (
                                content[:menu_start] + current_menu
                                + content[menu_end + 1:]
                            )
                    message["content"] = updated
        prepared, assembly_receipt = _assemble_framefit5_host_text(
            prepared, messages, runtime,
            include_editready_fact=editready_active,
            frame_request_allocation=frame_request_allocation,
            framefit5_inputs=framefit5_inputs,
        )
        receipt.update(assembly_receipt)
    return prepared, receipt


def prepare_release_host_text(
        messages: list[dict], runtime: object,
        *, tools: list[dict] | tuple[dict, ...] | None = None,
        frame_request_allocation: FrameRequestAllocation | None = None,
        framefit5_inputs: Mapping[str, object] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    """Compose release host text at the runtime's pre-allocation boundary."""
    return _apply_request_host_text(
        messages, runtime, tools=tools,
        frame_request_allocation=frame_request_allocation,
        framefit5_inputs=framefit5_inputs,
    )


def _degraded_history_request(
        messages: object, spec: Mapping[str, object], reason: str,
        runtime: object = None,
) -> tuple[list[dict], dict, dict[str, object]]:
    """Build an auditable history request after a frame presentation fails preflight."""
    usable_messages: list[dict] = []
    removed_frames = 0
    removed_sources = 0
    removed_file_indexes = 0
    source_prefixes = (
        "[WORKING SOURCE]", PAGED_SOURCE_PREFIX,
        "CURRENT_SOURCE ", *SOURCE_BLOCK_PREFIXES,
    )
    allowed_roles = {"system", "developer", "user", "assistant", "tool"}
    candidates = messages if isinstance(messages, (list, tuple)) else ()
    for message in candidates:
        if not isinstance(message, Mapping) or message.get("role") not in allowed_roles:
            continue
        copied = copy.deepcopy(dict(message))
        content = copied.get("content")
        if not isinstance(content, str):
            content = ""
            copied["content"] = content
        if content.startswith("[StateFrame]"):
            removed_frames += 1
            continue
        if content.startswith("[Files in frame]"):
            removed_file_indexes += 1
            continue
        if content.startswith(source_prefixes):
            removed_sources += 1
            continue
        usable_messages.append(copied)

    environment_value = spec.get("environment", {})
    environment = dict(environment_value) if isinstance(environment_value, Mapping) else {}
    environment["STATEFRAME_LEAN_PRESENTATION"] = "off"
    environment["STATEFRAME_TRAJECTORY_FRAME"] = "off"
    degraded_spec = dict(spec)
    degraded_spec.update({
        "presentation_arm": "history",
        "stateframe_trajectory_frame": "off",
        "environment": environment,
    })
    host_fact = (
        "[Host fact: StateFrame unavailable]\n"
        f"Reason: {reason}.\n"
        "Automatic StateFrame and carried source sections are omitted; a current target "
        "section is included only after its bytes are verified. "
        "Continue from recorded conversation history."
    )
    insertion_index = next(
        (index + 1 for index, message in enumerate(usable_messages)
         if message.get("role") == "user"),
        len(usable_messages),
    )
    usable_messages.insert(insertion_index, {"role": "user", "content": host_fact})
    usable_messages, host_text_receipt = _apply_request_host_text(
        usable_messages, runtime,
    )
    degradation = {
        "presentation_degraded": True,
        "degradation_reason": reason,
        "preflight_error_type": "",
        "removed_stateframe_messages": removed_frames,
        "removed_source_messages": removed_sources,
        "removed_file_index_messages": removed_file_indexes,
        **({"host_text": host_text_receipt} if host_text_receipt else {}),
    }
    return usable_messages, degraded_spec, degradation


def _journal_framefit5_preflight_error(runtime: object, error: Exception) -> None:
    """Record a bounded FrameFit5 reason without adding request or source text."""
    from kernel_stateframe_framefit5 import FrameFit5Error

    if not isinstance(error, FrameFit5Error):
        return
    try:
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if not callable(emit):
            return
        from kernel_v46_mechanism_wiring import _sanitize_error_text

        reason = _sanitize_error_text(str(error), limit=240)
        error_type = _sanitize_error_text(type(error).__name__, limit=80)
        emit(
            "framefit5_request_preflight_failed",
            producer="capability_request_preflight",
            request_id=getattr(runtime, "current_request_id", None),
            error_type=error_type or "FrameFit5Error",
            reason=reason or "<empty reason>",
            dispatch=False,
        )
    except Exception:
        # Diagnostics must not replace the FrameFit5Error that stopped preflight.
        return


def preflight_presentation_or_degrade(
        messages: list[dict], spec: dict, runtime,
        tools: list[dict] | tuple[dict, ...] | None = None,
        *, frame_request_allocation: FrameRequestAllocation | None = None,
        precomposed_host_text: Mapping[str, object] | None = None,
) -> tuple[list[dict], dict[str, object]]:
    """Keep valid frame checks strict; continue with checked history if one fails."""
    failure: Exception | None = None
    try:
        active_capture = getattr(runtime, "_active_capture", None)
        sealed = getattr(active_capture, "qualified_request", None)
        if sealed is not None:
            from kernel_request_seal import RequestSealError
            if (frame_request_allocation is not sealed.allocation
                    or not isinstance(precomposed_host_text, Mapping)
                    or precomposed_host_text.get("body_digest") != sealed.body_digest
                    or precomposed_host_text.get("allocation_digest") != sealed.allocation.digest):
                raise RequestSealError("sealed host preflight binding missing or mismatched")
            if messages != sealed.body["messages"] or list(tools or ()) != sealed.body.get("tools", []):
                raise RequestSealError("host preflight body differs from sealed input")
            sealed.validate(active_capture, runtime)
        if precomposed_host_text is None:
            prepared_messages, host_text_receipt = _apply_request_host_text(
                messages, runtime, tools=tools,
                frame_request_allocation=frame_request_allocation,
            )
        else:
            if precomposed_host_text.get("schema") != PRECOMPOSED_HOST_TEXT_SCHEMA:
                raise ValueError("release host-text precomposition receipt is invalid")
            request_id = getattr(runtime, "current_request_id", None)
            receipt_request_id = precomposed_host_text.get("request_id")
            if request_id is None or receipt_request_id != request_id:
                raise ValueError("release host-text precomposition belongs to another request")
            prepared_messages = copy.deepcopy(list(messages))
            raw_receipt = precomposed_host_text.get("receipt", {})
            if not isinstance(raw_receipt, Mapping):
                raise ValueError("release host-text precomposition receipt is malformed")
            host_text_receipt = copy.deepcopy(dict(raw_receipt))
            host_text_receipt["composed_before_allocation"] = True
        presentation = check_presentation(
            prepared_messages, spec, runtime, tools=tools,
            frame_request_allocation=frame_request_allocation,
        )
        presentation.update(host_text_receipt)
        return prepared_messages, presentation
    except Exception as initial_error:
        failure = initial_error
        from kernel_stateframe_framefit5 import has_framefit5_state
        if (getattr(runtime, "stateframe_framefit_mode", "off") == "on"
                or has_framefit5_state(messages)):
            _journal_framefit5_preflight_error(runtime, initial_error)
            from kernel_request_seal import RequestSealError
            raise RequestSealError("frame-only request preflight failed") from initial_error
        refreshed_messages = copy.deepcopy(list(messages))
        if _refresh_current_target_source(
                refreshed_messages, runtime, append_if_missing=True):
            try:
                prepared_messages, host_text_receipt = _apply_request_host_text(
                    refreshed_messages, runtime, tools=tools,
                )
                presentation = check_presentation(
                    prepared_messages, spec, runtime, tools=tools,
                )
                presentation.update(host_text_receipt)
                presentation["source_refreshed_from_current_bytes"] = True
                return prepared_messages, presentation
            except Exception as refreshed_error:
                failure = refreshed_error
        if failure is None:
            raise RuntimeError("presentation preflight failed without a captured error")
        reason = _frame_degradation_reason(failure, runtime)
        history_messages, history_spec, degradation = _degraded_history_request(
            messages, spec, reason, runtime,
        )
        history_source_path = _refresh_current_target_source(
            history_messages, runtime, append_if_missing=True,
        )
        if getattr(runtime, "target", None) is not None and not history_source_path:
            raise ValueError(
                "preflight cannot send a history fallback without byte-proven current target source"
            )
        if history_source_path:
            history_spec["history_current_target_source_path"] = history_source_path
        history_messages, host_text_receipt = _apply_request_host_text(
            history_messages, runtime, tools=tools,
        )
        # Use a neutral runtime for the explicitly degraded history check. A
        # frame/lean latch on the live runtime must not silently re-arm a frame
        # that this request has already declared unavailable.
        presentation = check_presentation(
            history_messages, history_spec, None, tools=tools,
            history_source_runtime=runtime,
        )
        presentation.update(degradation)
        presentation.update(host_text_receipt)
        presentation["source_refreshed_from_current_bytes"] = bool(history_source_path)
        presentation["preflight_error_type"] = type(failure).__name__
        error_detail = " ".join(str(failure).split())[:240]
        if error_detail:
            presentation["preflight_error"] = error_detail
        return history_messages, presentation


__all__ = [
    "PRECOMPOSED_HOST_TEXT_SCHEMA", "bound_runtime_sources", "check_presentation",
    "prepare_release_host_text", "preflight_presentation_or_degrade",
    "presentation_protected_messages",
]
