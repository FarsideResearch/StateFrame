"""Detached state and canonical rendering before request qualification."""
from __future__ import annotations

from collections.abc import Mapping
import copy


def runtime_authority_fields(runtime=None):
    from kernel_request_tool_surface import _RUNTIME_FIELDS

    dynamic = tuple(name for name in vars(runtime) if name.startswith(
        ("_completion_owner_", "_greenlock_"))) if runtime is not None else ()
    return tuple(dict.fromkeys((*_RUNTIME_FIELDS, *dynamic, "_completion_owner_request_contract_assertions",
            "_greenfact_rendered_request_revisions", "_greenfact_rendered_revisions",
            "_greenfact_skipped_revisions")))


def detached_context(capture, runtime):
    """Detach mutable request authority without cloning transports or journals."""
    candidate_runtime = copy.copy(runtime)
    for name in dict.fromkeys((*runtime_authority_fields(runtime),
            "_framefit_action_context", "_request_source_closure",
            "_empirical_frame_sources", "_request_deferred_source_paths",
            "prepared_binding", "_framefit5_owner_presentation_contract")):
        if hasattr(runtime, name):
            setattr(candidate_runtime, name, copy.deepcopy(getattr(runtime, name)))
    candidate_runtime.owner = copy.copy(runtime.owner)
    for name, value in vars(runtime.owner).items():
        if isinstance(value, (dict, list, set)):
            setattr(candidate_runtime.owner, name, copy.deepcopy(value))
    candidate_capture = copy.copy(capture)
    # Only request-owned mutable containers are detached. Shared services stay
    # read-only; the candidate carries its own source/head/owner state.
    for name, value in vars(capture).items():
        if isinstance(value, (dict, list, set)):
            setattr(candidate_capture, name, copy.deepcopy(value))
    candidate_capture.completion_owner_runtime = candidate_runtime
    candidate_capture.qualified_request = None
    candidate_capture.frame_request_allocation = None
    candidate_runtime._active_capture = candidate_capture
    return candidate_capture, candidate_runtime


def candidate_source_plan(capture, runtime):
    """The one source-authority lookup for a candidate: the capture's frozen override, else the runtime's bound closure, else a
    typed seal refusal (never an AttributeError/None deref; f4e, crash on 72beeb96 at render_visibility_sources)."""
    plan = getattr(capture, "_request_source_closure_override", None)
    if not isinstance(plan, Mapping):
        plan = getattr(runtime, "_request_source_closure", None)
    if not isinstance(plan, Mapping):
        from kernel_request_seal import RequestSealError
        raise RequestSealError("candidate lacks source authority")
    return plan


def render_candidate(capture, runtime, body):
    from kernel_frame_source_snapshot import frame_source_snapshot
    from kernel_stateframe_framefit5 import assemble_framefit5_request
    from kernel_frame_optional_allocation import render_optional_selection
    from kernel_stateframe_framefit5 import _state_payload, _workspace_root, _serialize_state_message
    from kernel_stateframe_envelope import STATE_HEADER

    proposal = getattr(capture, "_frame_request_allocation_candidate", None)
    if proposal is not None:
        for message in body["messages"]:
            if str(message.get("content") or "").startswith(STATE_HEADER + "\n"):
                payload, state = _state_payload(message)
                state["framefit5"] = render_optional_selection(
                    state["framefit5"], proposal.optional_heads,
                    proposal.site_inventory, root=_workspace_root(runtime))
                message["content"] = _serialize_state_message(payload)

    plan = candidate_source_plan(capture, runtime)
    sources = frame_source_snapshot(runtime, plan)
    return {**body, "messages": assemble_framefit5_request(
        body["messages"], getattr(capture, "_framefit5_semantic_messages", body["messages"]),
        task_message=capture._framefit5_task_message,
        source_plan=plan, sources=sources, runtime=runtime,
        owner_contract=capture.request_action_contract,
        owner_presentation=capture.owner_presentation,
        host_text_messages=body["messages"],
        frame_request_allocation=getattr(capture, "_replacement_allocation", None),
    )}


def stabilize_host_fields(body, *, semantic_messages=()):
    """Opaque semantic/source/tool payloads never enter the marker parser."""
    from kernel_stateframe_framefit5 import VOLATILE_TAIL_MARKER
    from kernel_wire_stability import project

    result = copy.deepcopy(body)
    indexes = [i for i, message in enumerate(result["messages"])
               if message.get("role") in {"system", "developer"}
               or (message.get("role") == "user" and message not in semantic_messages and
                   str(message.get("content", "")).startswith(VOLATILE_TAIL_MARKER))]
    for index in indexes:
        host = project({"messages": [result["messages"][index]]}).body["messages"]
        # Stability may split host lines. Rejoin their text in the original
        # host-owned slot; only the canonical assembler positions carriers.
        result["messages"][index]["content"] = "\n".join(str(row.get("content", "")) for row in host)
    return result


def render_visibility_sources(capture, runtime, body, check):
    """Carry verified read ranges through canonical assembly before fitting.

    Visibility's legacy prose pin has no revision identity. Render its complete
    lines from the verified snapshot; preserve every original tool/user byte.
    The finite source selection may subsequently defer these host copies.
    """
    from kernel_frame_source_snapshot import frame_source_snapshot
    from kernel_request_source_window import _render_ranges, project_source_window, source_revision
    from kernel_request_seal import RequestSealError
    from kernel_stateframe_framefit5 import _framefit_source_identity
    from kernel_read_reuse_visibility import _source_message, editable_read_paths

    plan = candidate_source_plan(capture, runtime)
    sources = frame_source_snapshot(runtime, plan)
    deferred = set(plan.get("deferred_paths", ()))
    editable = editable_read_paths(runtime)
    closure_paths = set(plan.get("paths", ()))
    missing = check.get("missing_ranges", ())
    result = copy.deepcopy(body)
    retained_paths = set()
    retained_candidates = []
    for candidate in check.get("candidates", ()):
        path = candidate["path"]
        if (path not in editable and path not in closure_paths) or (
                path in deferred and path not in editable) or any(
                all(row.get(key) == candidate.get(key) for key in
                    ("path", "revision", "start_line", "end_line")) for row in missing):
            continue
        retained_paths.add(path)
        retained_candidates.append(candidate)
        if any(message.get("role") == "user"
               and (identity := _framefit_source_identity(message)) is not None
               and identity[:3] == ("source", path, candidate["revision"])
               and _source_message(message.get("content"), candidate)
               for message in result["messages"]):
            continue
        source = sources.get(path)
        if not isinstance(source, str) or source_revision(source) != candidate["revision"]:
            raise RequestSealError("visibility source snapshot differs from verified revision")
        lines = source.splitlines(keepends=True)
        start, end = candidate["start_line"], candidate["end_line"]
        if not 1 <= start <= end <= len(lines):
            raise RequestSealError("visibility source range exceeds verified snapshot")
        rendered = project_source_window(_render_ranges(
            lines, [{"start_line": start, "end_line": end,
                     "anchors": ["read_reuse_visibility"]}],
            path=path, revision=candidate["revision"], total_lines=len(lines)),
            owner="read_reuse_visibility")
        if rendered is None:
            raise RequestSealError("visibility source could not be canonically rendered")
        message = {"role": "user", "content": rendered}
        if message not in result["messages"]:
            result["messages"].append(message)
    from kernel_visibility_requirements import register_visibility_requirements
    register_visibility_requirements(plan, retained_candidates)
    capture._visibility_required_paths = tuple(sorted(retained_paths))
    return result


def selected_visibility(check, delivery, *, wire=None, sources=None):
    """Choose delivery requirements before sealing; source proof owns deferral."""
    from kernel_request_source_window import source_delivery_satisfied
    from kernel_request_seal import RequestSealError

    if not source_delivery_satisfied(delivery):
        raise RequestSealError("visibility selection lacks a verified source proof")
    deferred = set(delivery.get("deferred_paths", ()))
    result = copy.deepcopy(check)
    result["deferred_paths"] = sorted(deferred)
    for name in ("candidates", "missing_ranges"):
        result[name] = [row for row in result.get(name, ())
                        if not isinstance(row, Mapping) or row.get("path") not in deferred]
    if wire is not None:
        from kernel_visibility_requirements import qualify_visibility_requirements
        result = qualify_visibility_requirements(check, result, delivery, wire, sources or {})
    return result
