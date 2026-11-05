"""C4c integration of the measured r2IC cache renderer, before the sole seal.

Separate from the pre-existing permutation-only kernel_cache_layout APIs.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import replace
import json
import os
import threading

from kernel_stateframe_page_log import PageLogPending, mode

_COMMIT_LOCK = threading.RLock()

FINAL_MODE_ENV = "STATEFRAME_FINAL_CACHE_LAYOUT"
SOURCE_COVERAGE_PREFIX = "[Source coverage]"
FINAL_LAYOUT_TOKEN_MARGIN = 16
FINAL_LAYOUT_BYTE_MARGIN = 256
_REPLAY_LAYOUT_DECISION_PROVIDER: Callable[[], str | None] | None = None
_REPLAY_STABLE_WIRE_DECISION_PROVIDER: Callable[[], str | None] | None = None


class FinalCacheLayoutProjectionError(ValueError):
    """The default layout cannot represent this request without changing its meaning."""


def final_layout_mode(environ=None):
    source = os.environ if environ is None else environ
    selected = str(source.get(FINAL_MODE_ENV, "on")).strip().lower()
    if selected not in {"off", "on"}:
        raise ValueError("invalid final cache layout mode")
    return selected


def final_layout_requested(environ=None) -> bool:
    """Whether the layout was asked for explicitly, rather than being on by default."""
    source = os.environ if environ is None else environ
    return bool(str(source.get(FINAL_MODE_ENV, "")).strip())


def install_replay_layout_decision_provider(provider: Callable[[], str | None] | None) -> None:
    """Install the next recorded wire decision inside the replay-only child path."""
    global _REPLAY_LAYOUT_DECISION_PROVIDER
    if provider is not None and not callable(provider):
        raise TypeError("final layout replay decision provider must be callable")
    _REPLAY_LAYOUT_DECISION_PROVIDER = provider


def replay_final_layout_decision() -> str | None:
    provider = _REPLAY_LAYOUT_DECISION_PROVIDER
    if provider is None:
        return None
    decision = provider()
    if decision not in {None, "layout", "plain"}:
        raise ValueError("invalid recorded final layout decision")
    return decision


def install_replay_stable_wire_decision_provider(provider: Callable[[], str | None] | None) -> None:
    """Install the next recorded stable-wire decision inside the replay-only child path."""
    global _REPLAY_STABLE_WIRE_DECISION_PROVIDER
    if provider is not None and not callable(provider):
        raise TypeError("stable wire replay decision provider must be callable")
    _REPLAY_STABLE_WIRE_DECISION_PROVIDER = provider


def replay_stable_wire_decision() -> str | None:
    provider = _REPLAY_STABLE_WIRE_DECISION_PROVIDER
    if provider is None:
        return None
    decision = provider()
    if decision not in {None, "applied", "plain"}:
        raise ValueError("invalid recorded stable wire decision")
    return decision


def _record_final_layout(capture, disposition: str, reason: str | None = None) -> None:
    if capture is None:
        return
    current = getattr(capture, "_final_cache_layout_record", None)
    record = dict(current) if isinstance(current, Mapping) else {}
    record["final_layout"] = disposition
    record.pop("final_layout_reason", None)
    if reason:
        record["final_layout_reason"] = reason
    try:
        capture._final_cache_layout_record = record
    except (AttributeError, TypeError):
        return


def _record_stable_wire(capture, disposition: str, reason: str | None = None) -> None:
    if capture is None:
        return
    current = getattr(capture, "_final_cache_layout_record", None)
    record = dict(current) if isinstance(current, Mapping) else {}
    record["stable_wire"] = disposition
    record.pop("stable_wire_reason", None)
    if reason:
        record["stable_wire_reason"] = reason
    try:
        capture._final_cache_layout_record = record
    except (AttributeError, TypeError):
        return


def _failed_open(capture, body, reason, **fields):
    from kernel_stateframe_page_log import _emit
    journal = getattr(capture, "journal", None) if capture is not None else None
    if journal is not None:
        _emit(journal, "final_cache_layout_failed_open", request_id=getattr(capture, "request_id", None),
              reason=reason, vital=True, **fields)
    _record_final_layout(capture, "failed_open", reason)
    return body


def render_final_cache_candidate(capture, runtime, body, *, meter, inputs):
    """Place current authority at the end of a final, fully fitted request.

    On by default. When it was asked for explicitly, a request it cannot project is an error; when it
    is only on by default, that request is sent in the normal (already fitted) layout instead, with a
    final_cache_layout_failed_open event, so the default never ends a run that did not opt in."""
    from kernel_stable_wire import mode as stable_wire_mode

    try:
        capture._final_cache_layout_record = {}
    except (AttributeError, TypeError):
        pass
    decision = replay_final_layout_decision()
    stable_decision = replay_stable_wire_decision()
    stable_switch_enabled = stable_wire_mode() == "on"
    selected_layout = final_layout_mode()
    if stable_switch_enabled and selected_layout != "on":
        raise ValueError(
            "STATEFRAME_STABLE_WIRE=on requires STATEFRAME_FINAL_CACHE_LAYOUT=on"
        )
    stable_wire_enabled = stable_switch_enabled or stable_decision in {"applied", "plain"}
    trim_callback = inputs.pop("_final_layout_trim", None) if isinstance(inputs, dict) else None
    if decision == "plain":
        _record_final_layout(capture, "plain")
        if stable_wire_enabled or stable_decision == "plain":
            _record_stable_wire(capture, "plain", "recorded_plain_layout")
        return body
    if decision is None and selected_layout == "off":
        _record_final_layout(capture, "plain")
        if stable_wire_enabled:
            _record_stable_wire(capture, "off")
        return body
    requested = final_layout_requested()
    if mode() != "off":
        if requested or decision == "layout":
            raise ValueError("final cache layout and page log cannot share one request")
        fallback = _failed_open(capture, body, "page_log_on")
        if stable_wire_enabled:
            _record_stable_wire(capture, "plain", "final_layout_failed_open")
        return fallback
    try:
        projected = _render_final_cache_layout(
            capture, runtime, body, meter=meter, inputs=inputs,
            replay_decision=decision, trim_callback=trim_callback,
            stable_wire_enabled=stable_wire_enabled,
            stable_replay_decision=stable_decision)
    except FinalCacheLayoutProjectionError as error:
        if requested or decision == "layout":
            raise
        fallback = _failed_open(capture, body, "cannot_project", error_type=type(error).__name__)
        if stable_wire_enabled:
            _record_stable_wire(capture, "plain", "final_layout_failed_open")
        return fallback
    if projected is not body:
        _record_final_layout(capture, "applied")
    return projected


def _wire_size(body) -> int:
    return len(json.dumps(body).encode("utf-8"))


def _final_layout_reserve(projected_tokens: int, token_ceiling: int,
                          projected_bytes: int, byte_ceiling: int) -> tuple[int, int]:
    return (
        max(0, projected_tokens - token_ceiling) + FINAL_LAYOUT_TOKEN_MARGIN,
        max(0, projected_bytes - byte_ceiling) + FINAL_LAYOUT_BYTE_MARGIN,
    )


def _cacheable_frame_content(original: str, cacheable: object) -> str:
    """Remove authority fields while preserving every untouched rendered JSON span."""
    header, separator, source = original.partition("\n")
    if not separator or header != "[StateFrame]":
        raise ValueError("final cache layout requires a rendered StateFrame")
    decoder = json.JSONDecoder()

    def rewrite(text: str, target: object, start: int = 0) -> tuple[str, int]:
        source_value, end = decoder.raw_decode(text, start)
        if source_value == target:
            return text[start:end], end
        if isinstance(source_value, Mapping) and isinstance(target, Mapping):
            if not set(target).issubset(source_value):
                raise ValueError("cacheable frame added a field during authority projection")
            cursor = start + 1
            leading_start = cursor
            while cursor < end and text[cursor].isspace():
                cursor += 1
            leading = text[leading_start:cursor]
            members = []
            while cursor < end and text[cursor] != "}":
                key_start = cursor
                key, key_end = decoder.raw_decode(text, cursor)
                if not isinstance(key, str):
                    raise ValueError("StateFrame object keys must be strings")
                cursor = key_end
                while cursor < end and text[cursor].isspace():
                    cursor += 1
                if cursor >= end or text[cursor] != ":":
                    raise ValueError("malformed rendered StateFrame object")
                cursor += 1
                while cursor < end and text[cursor].isspace():
                    cursor += 1
                value_start = cursor
                _source_child, value_end = decoder.raw_decode(text, value_start)
                if key in target:
                    rewritten_child, _ = rewrite(text, target[key], value_start)
                    members.append(text[key_start:value_start] + rewritten_child)
                cursor = value_end
                while cursor < end and text[cursor].isspace():
                    cursor += 1
                if cursor < end and text[cursor] == ",":
                    cursor += 1
                    while cursor < end and text[cursor].isspace():
                        cursor += 1
                elif cursor < end and text[cursor] == "}":
                    break
                else:
                    raise ValueError("malformed rendered StateFrame object")
            trailing_start = cursor
            while trailing_start < end and text[trailing_start].isspace():
                trailing_start += 1
            trailing = text[trailing_start:end - 1]
            return "{" + leading + ",".join(members) + trailing + "}", end
        if isinstance(source_value, list) and isinstance(target, list) and len(source_value) == len(target):
            cursor = start + 1
            leading_start = cursor
            while cursor < end and text[cursor].isspace():
                cursor += 1
            leading = text[leading_start:cursor]
            values = []
            for index, target_child in enumerate(target):
                rewritten_child, cursor = rewrite(text, target_child, cursor)
                values.append(rewritten_child)
                while cursor < end and text[cursor].isspace():
                    cursor += 1
                if index + 1 < len(target):
                    if cursor >= end or text[cursor] != ",":
                        raise ValueError("malformed rendered StateFrame array")
                    cursor += 1
                    while cursor < end and text[cursor].isspace():
                        cursor += 1
            trailing_start = cursor
            while trailing_start < end and text[trailing_start].isspace():
                trailing_start += 1
            trailing = text[trailing_start:end - 1]
            return "[" + leading + ",".join(values) + trailing + "]", end
        return json.dumps(target, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False, sort_keys=False), end

    rendered, end = rewrite(source, cacheable)
    return header + "\n" + rendered + source[end:]


def _project_final_cache_layout(body):
    from kernel_cache_layout_view import FINAL_LAYOUT_NAME, semantic_messages
    from kernel_greenfact import GREENFACT_TEXT
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER, _validated_exchange_and_volatile_tail
    from kernel_stateframe_page_log import _partition_authority, _payload_has_authority_keys
    from kernel_stateframe_final_message import AUTHORITY_OVERLAY_HEADER

    projected = copy.deepcopy(body)
    messages = projected.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("final cache layout requires a message array")
    _exchange, exchange_start, _tail_index = _validated_exchange_and_volatile_tail(messages)
    system_end = next((i for i, row in enumerate(messages) if row.get("role") != "system"), len(messages))
    frame_indexes = [index for index, row in enumerate(messages[system_end:exchange_start], start=system_end)
                     if str(row.get("content", "")).startswith("[StateFrame]\n")]
    if (len(frame_indexes) != 1 or system_end >= len(messages)
            or not str(messages[system_end].get("content", "")).startswith("Task:")):
        raise ValueError("final cache layout requires the canonical frame input")
    state_index = frame_indexes[0]
    if state_index <= system_end:
        raise ValueError("final cache layout requires the canonical frame input")
    payload = json.loads(messages[state_index]["content"].split("\n", 1)[1])
    cacheable, fields = _partition_authority(payload, final=True)
    if _payload_has_authority_keys(cacheable, final=True):
        raise ValueError("final cache layout retained frame authority")
    if json.dumps(fields, ensure_ascii=False).count("[CONTINUATION LIMITS |") != 1:
        raise ValueError("final cache layout requires one current continuation limit")
    authority_prefixes = (
        "[CONTINUATION LIMITS |", "[Unified stop authority]", "[COMPLETION OWNER]",
        "[WIRE_TRUTH_FACT ", "[EVIDENCEPIN]", "[DONE AUTHORITY]", "[MENU AUTHORITY]",
        "[Read lockout]", "[WIRE_READ_CONTRACT]", "[OPEN UNKNOWNS]",
        "[NATIVE BATCH RECEIPTS]", "[LITERAL SITE INVENTORY |",
    )
    stable = []
    moved: list[tuple[int, str, dict[str, object]]] = []
    for index, row in enumerate(messages[system_end:exchange_start], start=system_end):
        if index == state_index:
            continue
        content = row.get("content")
        if row.get("role") != "user" or not isinstance(content, str):
            raise ValueError("final cache layout has a non-user stable row")
        if content.startswith(authority_prefixes) or content.startswith(SOURCE_COVERAGE_PREFIX):
            extras = {key: copy.deepcopy(value) for key, value in row.items()
                      if key not in {"role", "content"}}
            moved.append((index, content, extras))
        else:
            stable.append(copy.deepcopy(dict(row)))
    # Preserve whole source carriers, with their path and revision attached.
    frame = messages[state_index]
    original_frame_content = frame["content"]
    frame["content"] = _cacheable_frame_content(original_frame_content, cacheable)
    final = messages[-1]
    original = final.get("content")
    if not isinstance(original, str) or not original.startswith(VOLATILE_TAIL_MARKER):
        raise ValueError("final cache layout lost the volatile tail")
    green = GREENFACT_TEXT if original.endswith(GREENFACT_TEXT) else ""
    prefix = original[:-len(green)] if green else original
    if AUTHORITY_OVERLAY_HEADER in prefix or "[CONTINUATION LIMITS |" in prefix:
        raise ValueError("final cache layout has duplicate authority text")
    final_prefix = prefix
    moved_wire: list[dict[str, object]] = []
    for index, content, extras in moved:
        final_prefix += "\n\n"
        start = len(final_prefix)
        final_prefix += content
        entry: dict[str, object] = {"index": index, "start": start, "end": len(final_prefix)}
        if extras:
            entry["extra"] = extras
        moved_wire.append(entry)
    overlay_fields = []
    for field in fields:
        value = field.get("value")
        exact_reference = next((index for index, content, _extras in moved
                                if isinstance(value, str) and value and content == value), None)
        substring_references = []
        if isinstance(value, str) and value and exact_reference is None:
            for index, content, _extras in moved:
                start = content.find(value)
                if start >= 0 and content.find(value, start + len(value)) < 0:
                    substring_references.append((index, start))
        if exact_reference is not None:
            overlay_fields.append({"path": field["path"], "message": exact_reference})
        elif len(substring_references) == 1:
            referenced, start = substring_references[0]
            overlay_fields.append({"path": field["path"], "message": referenced,
                                  "span": [start, start + len(value)]})
        else:
            overlay_fields.append(field)
    overlay = json.dumps(
        {"schema": "stateframe-authority-overlay/1", "layout": FINAL_LAYOUT_NAME,
          "fields": overlay_fields,
          "order": {"prefix_length": len(prefix), "original_prefix_count": exchange_start,
                    "frame_index": state_index, "moved": moved_wire}},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    final["content"] = final_prefix + "\n\n" + AUTHORITY_OVERLAY_HEADER + "\n" + overlay + green
    final["name"] = FINAL_LAYOUT_NAME
    projected["messages"] = [*messages[:system_end], *stable, frame,
                             *messages[exchange_start:-1], final]
    restored = semantic_messages(projected["messages"])
    if restored != list(body.get("messages", [])):
        raise ValueError("final cache layout message reconstruction differs")
    restored_frame = next((row for row in restored
                           if str(row.get("content", "")).startswith("[StateFrame]\n")), None)
    if not isinstance(restored_frame, Mapping) or restored_frame.get("content") != original_frame_content:
        raise ValueError("final cache layout frame reconstruction differs")
    return projected


def _checked_final_cache_projection(body):
    try:
        return _project_final_cache_layout(body)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        raise FinalCacheLayoutProjectionError(str(error)) from error


def _render_final_cache_layout(capture, runtime, body, *, meter, inputs,
                               replay_decision=None, trim_callback=None,
                               stable_wire_enabled=False, stable_replay_decision=None):
    from kernel_request_source_window import verify_final_source_delivery, source_delivery_satisfied
    from kernel_frame_allocation_capture import final_request_tokens, carry_final_request_tokens

    projected = _checked_final_cache_projection(body)
    tokens, method = final_request_tokens(projected, meter, capture)
    size = _wire_size(projected)
    if not stable_wire_enabled and replay_decision != "layout" and (
            tokens > inputs["token_ceiling"] or size > inputs["byte_ceiling"]):
        token_reserve, byte_reserve = _final_layout_reserve(
            tokens, inputs["token_ceiling"], size, inputs["byte_ceiling"])
        if callable(trim_callback):
            fitted_body = copy.deepcopy(body)
            trimmed = trim_callback(token_reserve, byte_reserve)
            if trimmed is False:
                if isinstance(body, dict):
                    body.clear()
                    body.update(fitted_body)
                return _failed_open(
                    capture, body, "exceeds_measured_envelope", meter=method,
                    prompt_tokens=tokens, token_ceiling=inputs["token_ceiling"],
                    bytes=size, byte_ceiling=inputs["byte_ceiling"],
                    token_reserve=token_reserve, byte_reserve=byte_reserve,
                )
            projected = _checked_final_cache_projection(body)
            tokens, method = final_request_tokens(projected, meter, capture)
            size = _wire_size(projected)
        if tokens > inputs["token_ceiling"] or size > inputs["byte_ceiling"]:
            return _failed_open(
                capture, body, "exceeds_measured_envelope", meter=method,
                prompt_tokens=tokens, token_ceiling=inputs["token_ceiling"],
                bytes=size, byte_ceiling=inputs["byte_ceiling"],
                token_reserve=token_reserve, byte_reserve=byte_reserve,
            )
    receipt = verify_final_source_delivery(
        inputs["source_plan"], projected["messages"], payload=projected,
        sources=inputs["sources"])
    if not source_delivery_satisfied(receipt):
        raise ValueError("final cache layout source proof failed")
    plain_receipt = copy.deepcopy(inputs.get("receipt"))
    inputs["receipt"] = receipt
    if capture is not None:
        capture.request_source_final_delivery = receipt
    if stable_wire_enabled and stable_replay_decision != "plain":
        from kernel_stable_wire import STATE_ATTR, project_stable_wire
        stable_state_present = hasattr(runtime, STATE_ATTR)
        stable_state_before = copy.deepcopy(getattr(runtime, STATE_ATTR, None))

        def restore_plain():
            inputs["receipt"] = plain_receipt
            if capture is not None:
                capture.request_source_final_delivery = plain_receipt
            if stable_state_present:
                setattr(runtime, STATE_ATTR, stable_state_before)
            elif hasattr(runtime, STATE_ATTR):
                delattr(runtime, STATE_ATTR)
            _record_stable_wire(capture, "plain", "final_layout_failed_open")

        def stable_trim(token_reserve, byte_reserve):
            if not callable(trim_callback):
                return False
            fitted_body = copy.deepcopy(body)
            trimmed = trim_callback(token_reserve, byte_reserve)
            if trimmed is False and isinstance(body, dict):
                body.clear()
                body.update(fitted_body)
            return trimmed

        result = project_stable_wire(
            capture, runtime, projected, meter=meter, inputs=inputs,
            trim_callback=stable_trim if callable(trim_callback) else None,
            replay_decision=stable_replay_decision)
        if inputs.pop("_stable_wire_retry_after_trim", False):
            projected = _checked_final_cache_projection(body)
            receipt = verify_final_source_delivery(
                inputs["source_plan"], projected["messages"], payload=projected,
                sources=inputs["sources"])
            if not source_delivery_satisfied(receipt):
                raise ValueError("final cache layout source proof failed")
            plain_receipt = copy.deepcopy(inputs.get("receipt"))
            inputs["receipt"] = receipt
            if capture is not None:
                capture.request_source_final_delivery = receipt
            result = project_stable_wire(
                capture, runtime, projected, meter=meter, inputs=inputs,
                replay_decision=stable_replay_decision)
        if result is projected:
            # Stable wire failed open to the final-layout body. The layout
            # fallback still owns any overflow in that body.
            tokens, method = final_request_tokens(projected, meter, capture)
            size = _wire_size(projected)
            if tokens > inputs["token_ceiling"] or size > inputs["byte_ceiling"]:
                restore_plain()
                return _failed_open(
                    capture, body, "exceeds_measured_envelope", meter=method,
                    prompt_tokens=tokens, token_ceiling=inputs["token_ceiling"],
                    bytes=size, byte_ceiling=inputs["byte_ceiling"],
                )
        result_tokens, result_method = final_request_tokens(result, meter, capture)
        if (result_tokens > inputs["token_ceiling"]
                or _wire_size(result) > inputs["byte_ceiling"]):
            restore_plain()
            return _failed_open(
                capture, body, "exceeds_measured_envelope", meter=result_method,
                prompt_tokens=result_tokens, token_ceiling=inputs["token_ceiling"],
                bytes=_wire_size(result), byte_ceiling=inputs["byte_ceiling"],
            )
        carry_final_request_tokens(result, capture, result_tokens)
        return result
    if stable_wire_enabled:
        _record_stable_wire(capture, "plain", "recorded_plain_wire")
    carry_final_request_tokens(projected, capture, tokens)
    return projected


def registered_done_for_wire(capture, registered, tools):
    """Resolve a description projection from the current bound final surface."""
    from kernel_request_tool_surface import SURFACE_SCHEMA
    from kernel_stateframe_tool_stable import _semantic_schema

    # Recovery can replace a request after its provisional receipt is made.
    # The bound final surface is current; parameter changes remain inadmissible.
    surface = getattr(capture, "final_tool_surface", None)
    if (not isinstance(surface, Mapping)
            or surface.get("schema") != SURFACE_SCHEMA
            or surface.get("tools") is not tools
            or surface.get("request_id") != getattr(capture, "request_id", None)
            or not isinstance(tools, list)):
        return registered
    projected = next((tool for tool in tools
                      if isinstance(tool, Mapping)
                      and isinstance(tool.get("function"), Mapping)
                      and tool["function"].get("name") == "done"), None)
    if (isinstance(projected, Mapping)
            and _semantic_schema(projected) == _semantic_schema(registered)):
        return copy.deepcopy(projected)
    return registered


def render_cache_candidate(capture, runtime, body, *, meter, inputs):
    """Project a fitted canonical candidate and prove the physical cache body.

    Source selection belongs to C4c. A cache epoch cannot resurrect a page
    removed by that selection, trim a source, or publish request authority.
    """
    selected = mode()
    if selected == "off":
        return body
    from kernel_cache_layout_view import inspect_layout
    from kernel_context import token_estimate
    from kernel_frame_optional_allocation import identity
    from kernel_request_source_window import verify_final_source_delivery, source_delivery_satisfied
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER
    from kernel_stateframe_page_log import _project_request_once, _newest_complete_exchange
    from kernel_stateframe_page_log import _body_without_exchange, _emit
    from kernel_stateframe_page_log import PageLogState

    journal = capture.journal
    capture._stateframe_page_log_pending = None
    stage = "input_validation"
    try:
        if getattr(capture, "_framefit5_newest_user_suffix", ()):
            # Original user input is an opaque allocation-owned suffix. Do not
            # relabel it as a host page or flatten its metadata into [V].
            raise ValueError("opaque_user_suffix")
        run_id = journal.run_id
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("missing_run_identity")
        previous = getattr(runtime, "_stateframe_page_log_state", None)
        if not isinstance(previous, PageLogState) or previous.run_id != run_id:
            previous = None
        exchange = _newest_complete_exchange(body)
        source = _body_without_exchange(body) if exchange else body
        kwargs = dict(
            token_meter=meter, selected_mode=selected, require_product=True,
            run_id=run_id, runtime=runtime, request_id=capture.request_id, supersede_source_pages=True,
            window_key_sources=True, budget_mode="on",
            host_prompt_cap=inputs["token_ceiling"],
            _exchange_messages=exchange[1] if exchange else (),
            token_input_body=body,
        )
        stage = "page_projection"
        projection = _project_request_once(source, previous, **kwargs)

        def finish(projected):
            result = copy.deepcopy(projected.body)
            final = result["messages"][-1]
            # C4c's marker belongs to the sole final user. Preserve all host
            # fields while positioning that existing marker at its beginning.
            text = final["content"]
            parts = text.split(VOLATILE_TAIL_MARKER)
            if len(parts) != 2:
                raise ValueError("volatile_marker_not_unique")
            final["content"] = VOLATILE_TAIL_MARKER + "\n\n" + parts[0] + parts[1]
            return result

        stage = "volatile_composition"
        projected = finish(projection)
        if (len(json.dumps(projected).encode()) > inputs["byte_ceiling"]
                or token_estimate(projected) > runtime.frame_ceiling):
            projection = _project_request_once(source, None, **kwargs)
            projection = replace(projection, previous_state=previous, action="rebase",
                                 rebase_reason="host_envelope_capacity")
            if previous is not None:
                projection = replace(projection, state=replace(
                    projection.state, generation=previous.generation + 1,
                    turn_index=previous.turn_index + 1, epoch=previous.epoch + 1))
            projected = finish(projection)
        from kernel_frame_allocation_capture import final_request_tokens
        tokens, method = final_request_tokens(projected, meter, capture)
        if (method != "tokenize" or getattr(meter, "measure_method", method) != "tokenize"
                or tokens > inputs["token_ceiling"]
                or len(json.dumps(projected).encode()) > inputs["byte_ceiling"]
                or token_estimate(projected) > runtime.frame_ceiling):
            raise ValueError("host_envelope_capacity")
        stage = "frame_reconstruction"
        _index, materialized, _live, _end = inspect_layout(projected["messages"])
        if identity(materialized) != identity(projection.state.frame_state.current_payload):
            raise ValueError("frame_reconstruction_mismatch")
        stage = "source_delivery"
        receipt = verify_final_source_delivery(
            inputs["source_plan"], projected["messages"], payload=projected,
            sources=inputs["sources"])
        if not source_delivery_satisfied(receipt):
            raise ValueError("source_proof_failed")
        pending = PageLogPending(runtime, previous, projection.state, selected,
                                 capture.request_id, capture.step)
        capture._stateframe_page_log_pending = pending
        _emit(journal, "stateframe_page_log_prepared", request_id=capture.request_id,
              mode=selected, action=projection.action, rebase_reason=projection.rebase_reason,
              prompt_tokens=tokens, cap=inputs["token_ceiling"],
              epoch=projection.state.epoch, page_count=len(projection.state.pages),
              prefix_preserved=projection.prefix_preserved)
        if selected == "shadow":
            _emit(journal, "stateframe_page_log_shadow_mismatch", request_id=capture.request_id,
                  field="messages", changed=projected != body)
            return body
        inputs["receipt"] = receipt
        capture.request_source_final_delivery = receipt
        from kernel_frame_allocation_capture import carry_final_request_tokens
        carry_final_request_tokens(projected, capture, tokens)
        return projected
    except Exception as error:
        from kernel_boundary_recovery import safe_error_class
        _emit(journal, "stateframe_page_log_failed_open", request_id=capture.request_id,
              mode=selected, reason="candidate_projection_or_proof", vital=True,
              failure_stage=stage, error_type=safe_error_class(error))
        return body


def prepare_stable_tools(capture, runtime, tools, body):
    """N4 calls this before binding its final surface; membership is unchanged."""
    from kernel_stateframe_tool_stable import (
        tool_mode, project_tools, make_pending, _emit,
    )

    selected = tool_mode()
    if selected == "off":
        return tools
    capture._stateframe_tool_surface_pending = None
    capture._stateframe_tool_guidance_pending = ()
    try:
        projection = project_tools(runtime, capture, tools, selected_tool_mode=selected,
                                   selected_menu_mode="off", selected_rebase_mode="off")
        pending = make_pending(runtime, projection, request_id=capture.request_id)
    except Exception as error:
        from kernel_boundary_recovery import safe_error_class
        _emit(capture.journal, "stateframe_tool_stable_failed_open",
              request_id=capture.request_id, mode=selected,
              reason="tool_projection_or_guidance", error_type=safe_error_class(error))
        return tools
    capture._stateframe_tool_surface_pending = pending
    _emit(capture.journal, "stateframe_tool_stable_prepared", request_id=capture.request_id,
          mode=selected, tool_count=len(projection.admitted_names))
    if selected == "shadow":
        _emit(capture.journal, "stateframe_tool_stable_shadow_mismatch",
              request_id=capture.request_id, field="tools", changed=projection.wire_tools != tools)
        return tools
    capture._stateframe_tool_guidance_pending = projection.guidance
    return projection.wire_tools


def compose_stable_guidance(capture, body):
    """Compose after the canonical assembler, before GREENFACT, fit and seal."""
    guidance = getattr(capture, "_stateframe_tool_guidance_pending", ())
    if not guidance:
        return body
    from kernel_stateframe_final_message import compose_final_user_message
    from kernel_greenfact import GREENFACT_TEXT
    return compose_final_user_message(body, tail_start=len(body["messages"]) - 1,
                                      additions=guidance, greenfact_text=GREENFACT_TEXT)


def pending_publication(candidate, runtime):
    """Rebind detached pending state to the live run only at atomic publication."""
    return {name: replace(pending, runtime=runtime) if pending is not None else None
            for name in ("_stateframe_page_log_pending", "_stateframe_tool_surface_pending")
            for pending in (getattr(candidate, name, None),)}


def commit_after_response(capture):
    from kernel_stateframe_page_log import commit_after_response as commit_pages
    from kernel_stateframe_tool_stable import commit_after_response as commit_tools
    with _COMMIT_LOCK:
        commit_pages(capture, capture.journal)
        commit_tools(capture, capture.journal)
