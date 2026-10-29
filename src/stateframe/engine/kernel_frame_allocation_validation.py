"""Read-only freshness checks for the host's FRAME_ONLY allocation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from kernel_frame_optional_allocation import FrameAllocationReplanNeeded, identity
from kernel_frame_request_allocation import FrameRequestAllocation


def template_environment() -> dict[str, str]:
    return {
        key: os.environ.get(key, "")
        for key in (
            "REACT_CHAT_TEMPLATE",
            "REACT_TOKENIZER",
            "REACT_SERVER_BUILD",
            "STATEFRAME_HEADCARRY",
            "STATEFRAME_HEADCARRY_MAX_BYTES",
        )
    }


def validate_allocation_authority(
    allocation: FrameRequestAllocation,
    runtime: object,
    messages: Sequence[Mapping[str, object]],
    tools: object = None,
    *,
    check_layout: bool = True,
) -> None:
    """Check live authority without changing selection, messages, or runtime."""
    from kernel_frame_source_snapshot import frame_source_snapshot
    from kernel_request_budget import (
        ENDPOINT_PROMPT_TOKEN_BUDGET,
        HARD_MAX_REQUEST_BYTES,
    )
    from kernel_stateframe_envelope import STATE_HEADER
    from kernel_stateframe_framefit5 import _state_payload, _workspace_root
    from kernel_frame_optional_allocation import render_optional_selection

    allocation.validate()
    binding = runtime.binding()
    revision = str(binding.get("product_revision") or binding.get("artifact"))
    if revision != allocation.identity["product_revision"]:
        raise FrameAllocationReplanNeeded("allocation product revision changed")
    plan = allocation.source_layout.get("source_plan")
    if not isinstance(plan, Mapping):
        raise FrameAllocationReplanNeeded("allocation source authority missing")
    current_plan = getattr(runtime, "_request_source_closure", None)
    current_authorization = (
        current_plan.get("workspace_authorization")
        if isinstance(current_plan, Mapping)
        else None
    )
    allocated_authorization = allocation.identity["workspace_authorization"]
    # The verifier appends status=VERIFIED to its receipt. Scope/generation are
    # the input authority; receipt status is not a field of the source plan.
    if (
        not isinstance(current_authorization, Mapping)
        or not isinstance(allocated_authorization, Mapping)
        or any(
            str(current_authorization.get(key, "")).strip()
            != allocated_authorization.get(key)
            for key in ("scope", "generation")
        )
    ):
        raise FrameAllocationReplanNeeded("allocation workspace authorization changed")
    frame_source_snapshot(runtime, plan)
    capacity = allocation.capacity
    if (
        runtime.frame_ceiling != capacity["frame_ceiling"]
        or capacity["token_ceiling"] != ENDPOINT_PROMPT_TOKEN_BUDGET
        or capacity.get("host_byte_ceiling", HARD_MAX_REQUEST_BYTES)
        != HARD_MAX_REQUEST_BYTES
    ):
        raise FrameAllocationReplanNeeded("allocation capacity authority changed")
    if tools is not None and identity(tools) != capacity["tool_schema_digest"]:
        raise FrameAllocationReplanNeeded("allocation tool schema changed")
    if capacity.get("template_environment") != template_environment():
        raise FrameAllocationReplanNeeded("allocation template authority changed")
    systems = [dict(msg) for msg in messages if msg.get("role") == "system"]
    if identity(systems) != capacity.get("system_messages_digest"):
        raise FrameAllocationReplanNeeded("allocation system inputs changed")
    from kernel_request_source_window import (
        _validate_frame_request_allocation_placement,
    )
    from kernel_stateframe_framefit5 import FrameFit5Error

    if check_layout:
        try:
            _validate_frame_request_allocation_placement(messages, allocation)
        except FrameFit5Error as error:
            raise FrameAllocationReplanNeeded(
                f"allocation source layout changed: {error}"
            ) from error
    from kernel_cache_layout_view import semantic_messages
    frames = [
        msg
        for msg in semantic_messages(messages)
        if str(msg.get("content") or "").startswith(STATE_HEADER + "\n")
    ]
    if len(frames) != 1:
        raise FrameAllocationReplanNeeded("allocation frame carrier changed")
    frame = _state_payload(frames[0])[1]["framefit5"]
    context = getattr(runtime, "_framefit_action_context", None)
    if isinstance(context, Mapping):
        from kernel_stateframe_framefit5 import _next_sites

        if [
            identity(site) for site in _next_sites(runtime, context)
        ] != allocation.site_inventory["inventory_identity"]:
            raise FrameAllocationReplanNeeded(
                "allocation semantic site inventory changed"
            )
    if identity(frame.get("action_ledger", {})) != allocation.identity["ledger_digest"]:
        raise FrameAllocationReplanNeeded("allocation ledger authority changed")
    expected = render_optional_selection(
        frame,
        allocation.optional_heads,
        allocation.site_inventory,
        root=_workspace_root(runtime),
    )
    for key in ("headcarry", "open_sites"):
        if expected.get(key) != frame.get(key) or (key in expected) != (key in frame):
            raise FrameAllocationReplanNeeded("allocation optional selection changed")


def require_active_allocation(
    runtime: object, allocation: FrameRequestAllocation | None
) -> None:
    """A missing explicit argument must not reactivate unconstrained rendering."""
    capture = getattr(runtime, "_active_capture", None)
    if getattr(capture, "request_id", None) != getattr(
        runtime, "current_request_id", None
    ):
        return
    active = getattr(capture, "frame_request_allocation", None)
    if active is not None and (
        allocation is None or active.digest != allocation.digest
    ):
        raise FrameAllocationReplanNeeded(
            "active host allocation binding missing or mismatched"
        )
