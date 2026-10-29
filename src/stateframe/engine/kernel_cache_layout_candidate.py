"""C4c integration of the measured r2IC cache renderer, before the sole seal.

Separate from the pre-existing permutation-only kernel_cache_layout APIs.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import replace
import json
import threading

from kernel_stateframe_page_log import PageLogPending, mode

_COMMIT_LOCK = threading.RLock()


def registered_done_for_wire(capture, registered, tools):
    """Resolve only the current N4 description projection of registered done."""
    from kernel_stateframe_tool_stable import ToolSurfacePending, _semantic_schema

    pending = getattr(capture, "_stateframe_tool_surface_pending", None)
    if (not isinstance(pending, ToolSurfacePending) or pending.stable_mode != "on"
            or pending.request_id != getattr(capture, "request_id", None)
            or pending.candidate_state.run_id != getattr(getattr(capture, "journal", None), "run_id", None)
            or not isinstance(tools, list)):
        return registered
    projected = dict(pending.candidate_state.schemas).get("done")
    if (isinstance(projected, Mapping) and projected in tools
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
        tokens, method = meter.measure(projected)
        if (method != "tokenize" or tokens > inputs["token_ceiling"]
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
