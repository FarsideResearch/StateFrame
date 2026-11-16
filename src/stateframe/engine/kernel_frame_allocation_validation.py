"""Read-only freshness checks for the host's FRAME_ONLY allocation."""

from __future__ import annotations

import copy
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_frame_optional_allocation import FrameAllocationReplanNeeded, identity, site_identity
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


def _frame_site_rows(frame: Mapping[str, object], root: Path) -> list[object]:
    from kernel_open_site_capacity import LOCATION_MODE, expand_location_inventory

    if frame.get("site_inventory_encoding") == LOCATION_MODE:
        return expand_location_inventory(frame, root)
    sites = frame.get("open_sites")
    if isinstance(sites, list):
        return sites
    next_site = frame.get("next_site")
    return [next_site] if isinstance(next_site, Mapping) else []


def _with_unit_sites(
    frame: Mapping[str, object],
    sites: list[dict[str, object]],
    site_inventory: Mapping[str, object],
) -> dict[str, object]:
    from kernel_open_site_capacity import LOCATION_MODE

    result = dict(frame)
    result.pop("site_inventory_encoding", None)
    result.pop("open_sites", None)
    next_site_id = site_inventory.get("next_site")
    result["next_site"] = next(
        (site for site in sites if site_identity(site) == next_site_id),
        sites[0] if sites else None,
    )
    encoding = site_inventory.get("encoding_mode")
    if encoding == "full_inventory":
        result["open_sites"] = sites
    elif encoding == "location_inventory":
        result["open_sites"] = sites
        result["site_inventory_encoding"] = LOCATION_MODE
    elif encoding != "singleton_next_site":
        raise FrameAllocationReplanNeeded("unit allocation site encoding is invalid")
    return result


# read_facts keys that describe how a read body was delivered on the wire. The
# working-set renderer and source-metadata pruning own them per request, so the
# host-derived unit authority cannot carry them; the non-subframe path never
# compares them either. Every other read_facts key stays authority-strict.
READ_FACT_DELIVERY_KEYS = frozenset({
    "body_delivery", "delivery_notice", "visible_symbols", "receipts",
    "historical_receipts", "journal_workspace_revisions",
    "historical_workspace_revisions", "receipt_count", "revision_scope",
    # Also removed by _prune_redundant_source_metadata once a whole-file window
    # is on the wire (the host view prunes, the allocator's frame is pre-window).
    "content_sha256", "lines",
})


def _authority_view(frame: Mapping[str, object]) -> dict[str, object]:
    """Return the frame with per-request delivery metadata removed from read_facts."""
    view = dict(frame)
    for key in ("open_sites",):
        rows = view.get(key)
        if isinstance(rows, list):
            view[key] = [_site_view(row) if isinstance(row, Mapping) else row for row in rows]
    if isinstance(view.get("next_site"), Mapping):
        view["next_site"] = _site_view(view["next_site"])
    facts = view.get("read_facts")
    if isinstance(facts, list):
        view["read_facts"] = [
            {key: value for key, value in fact.items() if key not in READ_FACT_DELIVERY_KEYS}
            if isinstance(fact, Mapping) else fact
            for fact in facts
        ]
    return view


def _owner_protocol_mismatch_keys(
    frame_view: Mapping[str, object], projection_view: Mapping[str, object]
) -> list[str] | None:
    """Return owner_protocol subkeys that differ, or None when either side is not a mapping.

    Key names only: values are never included so the message stays free of payload data.
    """
    frame_owner = frame_view.get("owner_protocol")
    projection_owner = projection_view.get("owner_protocol")
    if not isinstance(frame_owner, Mapping) or not isinstance(projection_owner, Mapping):
        return None
    keys = set(frame_owner) | set(projection_owner)
    differing = [key for key in keys if frame_owner.get(key) != projection_owner.get(key)]
    return [str(key) for key in sorted(differing, key=str)]


def _authority_mismatch_message(
    changed: list[str], frame_view: Mapping[str, object], projection_view: Mapping[str, object]
) -> str:
    """Build the replan reason for a scoped frame whose authority differs from its unit projection."""
    message = f"scoped allocation frame differs from its unit authority (fields={changed!r}"
    if "owner_protocol" in changed:
        owner_keys = _owner_protocol_mismatch_keys(frame_view, projection_view)
        if owner_keys is not None:
            message += f", owner_protocol_keys={owner_keys!r}"
    return message + ")"


def _site_view(site: Mapping[str, object]) -> dict[str, object]:
    """Identity of an open site independent of how its bytes ride the wire.

    Allocation snapshots a site before the working-set renderer adds its source
    window, so the bytes are inline; the final request may carry the same bytes by
    ``source_ref``. Only that representation is dropped, and only when the digest
    proves the bytes (or the reference) are exactly the site's own content.
    """
    view = dict(site)
    digest = view.get("content_sha256")
    inline, ref = view.get("current_bytes"), view.get("source_ref")
    if isinstance(digest, str):
        if isinstance(inline, str) and hashlib.sha256(inline.encode("utf-8")).hexdigest() == digest:
            view.pop("current_bytes", None)
            view.pop("source_ref", None)
        elif (inline is None and isinstance(ref, Mapping)
                and ref.get("content_sha256") == digest):
            view.pop("current_bytes", None)
            view.pop("source_ref", None)
    return view


def site_authority_identity(site: Mapping[str, object]) -> str:
    """Digest of ``_site_view``: the site's authority fields only."""
    return identity(_site_view(site))


def unit_allocation_projection(
    frame: Mapping[str, object], unit_spec: object, *, root: Path,
    delivered_windows: Sequence[Mapping[str, object]] = (),
    continuation_limits: str | None = None,
) -> dict[str, object]:
    """Build the canonical host-derived view used by every child allocation phase."""
    from kernel_stateframe_framefit5 import scope_typed_state_for_unit

    projection = scope_typed_state_for_unit(
        frame, unit_spec, root=root, delivered_windows=delivered_windows,
        continuation_limits=continuation_limits,
    )
    if not isinstance(projection.get("action_ledger"), Mapping):
        raise FrameAllocationReplanNeeded("unit allocation ledger is unavailable")
    return projection


def allocation_ledger_authority(runtime: object, frame: Mapping[str, object]) -> object:
    """Return the host-derived ledger bound to a child, or the parent frame ledger."""
    if not isinstance(frame.get("subframe_scope"), Mapping):
        return frame.get("action_ledger", {})
    unit_module = sys.modules.get("kernel_unit_policy")
    active_unit = unit_module.active() if unit_module is not None else None
    context = getattr(runtime, "_framefit_action_context", None)
    context_frame = context.get("framefit5") if isinstance(context, Mapping) else None
    unit_spec = getattr(active_unit, "spec", None)
    if (unit_spec is None or not isinstance(context_frame, Mapping)):
        raise FrameAllocationReplanNeeded("scoped allocation has no host ledger authority")
    from kernel_stateframe_framefit5 import _workspace_root

    projection = unit_allocation_projection(
        context_frame, unit_spec, root=_workspace_root(runtime),
    )
    return projection["action_ledger"]


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
    retry_limits = getattr(getattr(runtime, "_active_capture", None),
                           "_capacity_retry_limits", None)
    expected_token_ceiling = (
        min(ENDPOINT_PROMPT_TOKEN_BUDGET, retry_limits.max_prompt_tokens)
        if retry_limits is not None else ENDPOINT_PROMPT_TOKEN_BUDGET
    )
    if (
        runtime.frame_ceiling != capacity["frame_ceiling"]
        or capacity["token_ceiling"] != expected_token_ceiling
        or capacity.get("host_byte_ceiling", HARD_MAX_REQUEST_BYTES)
        != HARD_MAX_REQUEST_BYTES
        or (retry_limits is not None
            and capacity["byte_ceiling"] > retry_limits.max_request_bytes)
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
            from kernel_cache_layout_view import FINAL_LAYOUT_NAME, semantic_messages
            placement_messages = (semantic_messages(messages)
                                  if messages and messages[-1].get("name") == FINAL_LAYOUT_NAME
                                  else messages)
            _validate_frame_request_allocation_placement(placement_messages, allocation)
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
    context_frame = context.get("framefit5") if isinstance(context, Mapping) else None
    unit_module = sys.modules.get("kernel_unit_policy")
    active_unit = unit_module.active() if unit_module is not None else None
    scoped = isinstance(frame.get("subframe_scope"), Mapping)
    unit_projection: dict[str, object] | None = None
    unit_sites: list[dict[str, object]] | None = None
    site_inventory: Mapping[str, object] = allocation.site_inventory
    if active_unit is not None:
        if not isinstance(context, Mapping) or not isinstance(context_frame, Mapping):
            raise FrameAllocationReplanNeeded("unit allocation has no host action authority")
        from kernel_stateframe_framefit5 import _next_sites

        unit_spec = getattr(active_unit, "spec", None)
        if unit_spec is None:
            raise FrameAllocationReplanNeeded("unit allocation specification is unavailable")
        root = _workspace_root(runtime)
        unit_projection = unit_allocation_projection(context_frame, unit_spec, root=root)
        host_frame = {**context_frame, "open_sites": _next_sites(runtime, context)}
        host_frame.pop("site_inventory_encoding", None)
        host_projection = unit_allocation_projection(
            host_frame, unit_spec, root=root,
        )
        host_sites = host_projection["open_sites"]
        if not isinstance(host_sites, list):
            raise FrameAllocationReplanNeeded("unit allocation site projection is invalid")
        if [site_identity(site) for site in unit_projection["open_sites"]] != [
                site_identity(site) for site in host_sites]:
            raise FrameAllocationReplanNeeded("unit allocation host sites changed")
        # The allocation records the child wire representation. Build it from
        # fresh full-byte host sites and this request's delivered windows.
        wire_host_projection = unit_allocation_projection(
            host_frame, unit_spec, root=root,
            delivered_windows=semantic_messages(messages),
        )
        unit_sites = wire_host_projection["open_sites"]
        inventory = allocation.site_inventory
        site_ids = [site_identity(site) for site in unit_sites]
        allocated_ids = [site_authority_identity(site) for site in inventory.get("sites", ())]
        if ([site_authority_identity(site) for site in unit_sites] != allocated_ids
                or len(allocated_ids) != len(inventory["inventory_identity"])):
            raise FrameAllocationReplanNeeded(
                "unit-scoped allocation inventory differs from host authority "
                f"(allocated={inventory['inventory_identity']!r}, host={site_ids!r})"
            )
        recorded_next = inventory.get("next_site")
        if recorded_next is not None and recorded_next not in inventory["inventory_identity"]:
            raise FrameAllocationReplanNeeded("unit-scoped allocation frontier differs from host authority")
        # Same sites, now in the representation this request's windows give them.
        host_next = (site_ids[inventory["inventory_identity"].index(recorded_next)]
                     if recorded_next is not None else None)
        site_inventory = {**inventory, "sites": copy.deepcopy(unit_sites),
                          "inventory_identity": list(site_ids), "selected_identities": list(site_ids),
                          "open_sites": list(site_ids), "next_site": host_next}
        if not scoped:
            serialized_frame = {
                **frame, "open_sites": _frame_site_rows(frame, root),
            }
            serialized_frame.pop("site_inventory_encoding", None)
            serialized_projection = unit_allocation_projection(
                serialized_frame, unit_spec, root=root,
            )
            serialized_sites = serialized_projection["open_sites"]
            if [site_identity(site) for site in serialized_sites] != site_ids:
                raise FrameAllocationReplanNeeded("unit-scoped serialized site inventory changed")
    elif scoped:
        raise FrameAllocationReplanNeeded("scoped allocation has no active unit authority")
    elif isinstance(context, Mapping):
        from kernel_stateframe_framefit5 import _next_sites

        if [
            site_identity(site) for site in _next_sites(runtime, context)
        ] != allocation.site_inventory["inventory_identity"]:
            raise FrameAllocationReplanNeeded(
                "allocation semantic site inventory changed"
            )
    authority_frame = context_frame if scoped else frame
    if not isinstance(authority_frame, Mapping):
        raise FrameAllocationReplanNeeded("allocation frame authority is unavailable")
    if active_unit is not None:
        if unit_sites is None or unit_projection is None:
            raise FrameAllocationReplanNeeded("unit allocation projection is unavailable")
        authority_frame = _with_unit_sites(
            authority_frame, unit_sites, site_inventory,
        )
        if scoped:
            projected_ledger = unit_projection.get("action_ledger")
            if not isinstance(projected_ledger, Mapping):
                raise FrameAllocationReplanNeeded("unit allocation ledger is unavailable")
            authority_frame["action_ledger"] = projected_ledger
    if identity(authority_frame.get("action_ledger", {})) != allocation.identity["ledger_digest"]:
        raise FrameAllocationReplanNeeded("allocation ledger authority changed")
    expected = render_optional_selection(
        authority_frame,
        allocation.optional_heads,
        site_inventory,
        root=_workspace_root(runtime),
    )
    for key in ("headcarry", "open_sites"):
        if (expected.get(key) != authority_frame.get(key)
                or (key in expected) != (key in authority_frame)):
            raise FrameAllocationReplanNeeded("allocation optional selection changed")
    if scoped:
        if unit_projection is None:
            raise FrameAllocationReplanNeeded("scoped allocation has no active unit authority")
        from kernel_stateframe_framefit5 import (
            _journal_read_facts, _serialize_state_message,
        )
        from kernel_lean_presentation import _runtime_continuation_limits

        host_ledger = context_frame.get("action_ledger")
        if not isinstance(host_ledger, Mapping):
            raise FrameAllocationReplanNeeded("scoped allocation host read ledger is unavailable")
        records = context.get("journal_receipt_records")
        fresh_host_frame = {
            **context_frame,
            "read_facts": (
                _journal_read_facts(context, _workspace_root(runtime), host_ledger)
                if records is not None else context_frame.get("read_facts", [])
            ),
        }
        wire_projection = unit_allocation_projection(
            fresh_host_frame, active_unit.spec, root=_workspace_root(runtime),
            delivered_windows=semantic_messages(messages),
            continuation_limits=_runtime_continuation_limits(runtime),
        )
        from kernel_reread_result_truth import reconcile_frame

        reconcile_frame(wire_projection, semantic_messages(messages))
        # Whole-file source windows replace redundant receipt metadata on the
        # actual candidate. Apply that same existing projection to the fresh
        # host view before comparing it with the child wire.
        from kernel_frame_source_snapshot import _prune_redundant_source_metadata

        projected_messages = copy.deepcopy([dict(row) for row in semantic_messages(messages)])
        for projected in projected_messages:
            if str(projected.get("content") or "").startswith(STATE_HEADER + "\n"):
                payload, state = _state_payload(projected)
                state["framefit5"] = wire_projection
                payload["state"] = state
                projected["content"] = _serialize_state_message(payload)
                break
        _prune_redundant_source_metadata(projected_messages)
        projected_frame = next(
            row for row in projected_messages
            if str(row.get("content") or "").startswith(STATE_HEADER + "\n")
        )
        wire_projection = _state_payload(projected_frame)[1]["framefit5"]
        # The host projection re-derives exact range pins from the journal. Apply the
        # fit's delivery rule so a pin this request already carries is compared as the
        # same-request representation the fitted frame holds.
        from kernel_stateframe_capacity import drop_delivered_range_pins, drop_unbacked_range_pins

        projected_ledger = copy.deepcopy(wire_projection.get("action_ledger"))
        if isinstance(projected_ledger, dict) and isinstance(projected_ledger.get("reads"), list):
            drop_unbacked_range_pins(projected_ledger, semantic_messages(messages))
            drop_delivered_range_pins(projected_ledger, semantic_messages(messages), frame)
            wire_projection = {**wire_projection, "action_ledger": projected_ledger}
        if frame.get("action_ledger") != wire_projection.get("action_ledger"):
            raise FrameAllocationReplanNeeded(
                "unit-scoped serialized ledger differs from host projection"
            )
        if ("open_sites" not in frame and site_inventory.get("encoding_mode") == "singleton_next_site"
                and isinstance(frame.get("next_site"), Mapping)
                and [site_authority_identity(site) for site in wire_projection.get("open_sites", ())]
                == [site_authority_identity(frame["next_site"])]):
            # The singleton alias omits open_sites; the host projection always
            # lists it. Drop it only when it is exactly the aliased next_site.
            wire_projection = {k: v for k, v in wire_projection.items() if k != "open_sites"}
        if _authority_view(frame) != _authority_view(wire_projection):
            authority_frame_view = _authority_view(frame)
            authority_projection_view = _authority_view(wire_projection)
            changed = sorted(
                key for key in set(authority_frame_view) | set(authority_projection_view)
                if authority_frame_view.get(key) != authority_projection_view.get(key)
            )
            raise FrameAllocationReplanNeeded(_authority_mismatch_message(
                changed, authority_frame_view, authority_projection_view,
            ))


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
