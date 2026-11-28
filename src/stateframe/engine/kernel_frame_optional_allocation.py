"""Pure optional FRAME_ONLY selections, rendered from host-owned snapshots."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_frame_request_allocation import (
    FrameRequestAllocationError,
    canonical_allocation_bytes,
)
from kernel_stateframe_headcarry import _seal_byte_count


class FrameAllocationReplanNeeded(FrameRequestAllocationError):
    """An allocated input changed; the caller must use bounded preparation recovery."""

    code = "FRAME_ALLOCATION_REPLAN_NEEDED"


def identity(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_allocation_bytes(value)).hexdigest()


# Per-request delivery annotations written by rebuild_delivery_truth. They describe
# what this request showed, not what the site is, so they stay out of inventory
# identity. Source pins (path, revision, lines, content_sha256) remain in it.
_SITE_DELIVERY_KEYS = frozenset({"read_this_request", "current_bytes"})
# source_ref is the other delivery carrier: site_ref_or_inline swaps current_bytes for
# source_ref per request when the bytes are proven present in that request's windows.
# Which carrier a request uses is a delivery choice, so source_ref is excluded too.
_SITE_CARRIER_KEYS = frozenset({"source_ref"})


def site_identity(site: Mapping[str, object]) -> str:
    """Identity of an open site, independent of how this request delivered its bytes."""
    excluded = _SITE_DELIVERY_KEYS | _SITE_CARRIER_KEYS
    return identity({key: value for key, value in site.items() if key not in excluded})


def head_identity(head: Mapping[str, object]) -> str:
    return identity(
        {
            key: head.get(key)
            for key in (
                "path",
                "revision",
                "line_range",
                "content_sha256",
                "source_kind",
            )
        }
    )


def snapshot_optional_inputs(
    frame: Mapping[str, object],
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Bind even inputs omitted by the panel's own configured byte allowance."""
    from kernel_stateframe_headcarry import (
        _import_targets,
        _module_head,
        _read_optional_source,
        _requested_paths,
        headcarry_owed_reads_enabled,
        owed_read_rows,
    )
    from kernel_obligation_assess import UnavailableEvidence

    sites = frame.get("open_sites")
    if not isinstance(sites, list):
        site = frame.get("next_site")
        sites = [site] if isinstance(site, Mapping) else []
    from kernel_open_site_capacity import LOCATION_MODE, expand_location_inventory
    compact_inventory = frame.get("site_inventory_encoding") == LOCATION_MODE
    if compact_inventory:
        sites = expand_location_inventory(frame, root)
    edits = frame.get("applied_edits", [])
    # T2FIX: owed reads are head inputs too (same rule as render_headcarry), so their
    # heads stay inside the dependency inventory instead of forcing a replan.
    owed_reads = (owed_read_rows(frame.get("read_facts"))
                  if headcarry_owed_reads_enabled() else [])
    primary, roles = _requested_paths(sites, edits, owed_reads)
    dependencies: dict[str, str] = {}
    for path in primary:
        raw = _read_optional_source(root, path)
        dependencies[path] = (
            "sha256:" + hashlib.sha256(raw).hexdigest()
            if raw is not None
            else "UNAVAILABLE"
        )
        if raw is not None:
            targets = _import_targets(root, path, raw)
            if isinstance(targets, UnavailableEvidence):
                raise FrameAllocationReplanNeeded("optional dependency unavailable")
            for dependency in targets:
                source = _read_optional_source(root, dependency)
                dependencies[dependency] = (
                    "sha256:" + hashlib.sha256(source).hexdigest()
                    if source is not None
                    else "UNAVAILABLE"
                )
    panel = frame.get("headcarry")
    panel = copy.deepcopy(dict(panel)) if isinstance(panel, Mapping) else None
    rows = panel.get("heads", []) if panel is not None else []
    for row in rows:
        path = row.get("path")
        if path not in dependencies:
            raise FrameAllocationReplanNeeded(
                "head is outside the current dependency inventory"
            )
        raw = _read_optional_source(root, path)
        whole_file = (roles.get(path) == "owed_read" and row.get("source_kind") == "owed_read")
        minimum_end = len(raw.splitlines(keepends=True)) if (raw is not None and whole_file) else 0
        fresh = (_module_head(root, path, raw, minimum_end=minimum_end)
                 if raw is not None else None)
        if fresh is None or any(row.get(key) != value for key, value in fresh.items()):
            raise FrameAllocationReplanNeeded("head source changed before allocation")
    candidates = [
        {
            "candidate_id": head_identity(row),
            "source_revision": row["revision"],
            "dependency_revisions": dict(dependencies),
            "head": copy.deepcopy(row),
        }
        for row in rows
    ]
    ids = [row["candidate_id"] for row in candidates]
    if len(set(ids)) != len(ids):
        raise FrameRequestAllocationError("duplicate optional head identity")
    optional = {
        "candidates": candidates,
        "retained_identities": ids,
        "allowance": panel.get("byte_budget", 0) if panel is not None else 0,
        "omitted_count": panel.get("omitted_count", 0) if panel is not None else 0,
        "deduplicated_count": panel.get("deduplicated_count", 0)
        if panel is not None
        else 0,
        "render_policy_identity": "stateframe-framefit5-headcarry/1",
        "dependency_revisions": dependencies,
        "panel": panel,
    }
    site_ids = [site_identity(site) for site in sites]
    inventory = {
        "inventory_identity": site_ids,
        "selected_identities": list(site_ids),
        "open_sites": list(site_ids),
        "next_site": site_identity(frame["next_site"])
        if isinstance(frame.get("next_site"), Mapping)
        else None,
        "encoding_mode": LOCATION_MODE if compact_inventory else "full_inventory",
        "sites": copy.deepcopy(sites),
    }
    return optional, inventory


def selection_from_frame(
    optional: Mapping[str, object],
    inventory: Mapping[str, object],
    frame: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Capture a measured subset without making the wire its own authority."""
    heads = copy.deepcopy(dict(optional))
    sites = copy.deepcopy(dict(inventory))
    panel = frame.get("headcarry")
    if isinstance(panel, Mapping):
        if heads["panel"] is None:
            raise FrameAllocationReplanNeeded("head panel appeared during allocation")
        selected = [head_identity(row) for row in panel.get("heads", [])]
        allowed = [row["candidate_id"] for row in heads["candidates"]]
        if selected != [item for item in allowed if item in set(selected)]:
            raise FrameAllocationReplanNeeded(
                "head selection changed during allocation"
            )
        expected_omitted = (
            heads["panel"]["omitted_count"] + len(allowed) - len(selected)
        )
        if panel.get("omitted_count") != expected_omitted:
            raise FrameAllocationReplanNeeded(
                "head omission count changed during allocation"
            )
        heads["retained_identities"] = selected
        heads["omitted_count"] = expected_omitted
    elif heads["panel"] is not None:
        raise FrameAllocationReplanNeeded("head panel disappeared during allocation")
    from kernel_open_site_capacity import LOCATION_MODE, location_inventory_candidate
    if frame.get("site_inventory_encoding") == LOCATION_MODE:
        if not sites["sites"]:
            raise FrameAllocationReplanNeeded("location inventory has no host sites")
        expected = location_inventory_candidate({
            "open_sites": sites["sites"], "next_site": sites["sites"][0],
        })
        if (expected is None or expected["open_sites"] != frame.get("open_sites")
                or frame.get("next_site") != sites["sites"][0]):
            raise FrameAllocationReplanNeeded("location inventory changed during allocation")
        sites["encoding_mode"] = LOCATION_MODE
    elif "open_sites" not in frame:
        if sites["open_sites"] != [sites["next_site"]]:
            if sites["open_sites"]:
                raise FrameAllocationReplanNeeded("non-singleton inventory disappeared")
        elif frame.get("next_site") == sites["sites"][0]:
            sites["encoding_mode"] = "singleton_next_site"
    elif frame["open_sites"] != sites["sites"]:
        raise FrameAllocationReplanNeeded(
            "semantic inventory changed during allocation"
        )
    rendered = render_optional_selection(frame, heads, sites)
    if rendered.get("headcarry") != frame.get("headcarry"):
        raise FrameAllocationReplanNeeded(
            "head bytes differ from the allocated selection"
        )
    return heads, sites


def validate_optional_dependencies(optional: Mapping[str, object], *, root: Path | None) -> None:
    """Check retained and omitted dependency revisions without rendering."""
    if root is not None:
        from kernel_stateframe_headcarry import _read_optional_source

        for path, revision in optional.get("dependency_revisions", {}).items():
            raw = _read_optional_source(root, path)
            current = (
                "sha256:" + hashlib.sha256(raw).hexdigest()
                if raw is not None
                else "UNAVAILABLE"
            )
            if current != revision:
                raise FrameAllocationReplanNeeded("optional dependency changed")


def render_optional_selection(
    frame: Mapping[str, object],
    optional: Mapping[str, object],
    inventory: Mapping[str, object],
    *,
    root: Path | None = None,
) -> dict[str, object]:
    """Render a detached exact selection; never write it into action authority."""
    validate_optional_dependencies(optional, root=root)
    result = copy.deepcopy(dict(frame))
    panel = optional.get("panel")
    if isinstance(panel, Mapping):
        selected = optional["retained_identities"]
        candidates = optional["candidates"]
        ids = [row["candidate_id"] for row in candidates]
        if len(set(ids)) != len(ids) or list(selected) != [
            key for key in ids if key in selected
        ]:
            raise FrameRequestAllocationError("invalid retained head selection")
        rendered = copy.deepcopy(dict(panel))
        rendered["heads"] = [
            copy.deepcopy(row["head"])
            for row in candidates
            if row["candidate_id"] in selected
        ]
        rendered["omitted_count"] = optional["omitted_count"]
        rendered["omitted_line"] = f"omitted {optional['omitted_count']} heads"
        result["headcarry"] = _seal_byte_count(rendered)
    sites = inventory.get("sites")
    if not isinstance(sites, Sequence):
        raise FrameRequestAllocationError("allocation has no semantic inventory")
    from kernel_open_site_capacity import LOCATION_MODE, location_inventory_candidate
    if inventory["encoding_mode"] == LOCATION_MODE:
        if not sites or frame.get("next_site") != sites[0]:
            raise FrameAllocationReplanNeeded("location inventory frontier changed")
        full = {**result, "open_sites": copy.deepcopy(list(sites))}
        full.pop("site_inventory_encoding", None)
        compact = location_inventory_candidate(full)
        if compact is None:
            raise FrameAllocationReplanNeeded("location inventory cannot be rendered")
        return compact
    result.pop("site_inventory_encoding", None)
    if inventory["encoding_mode"] == "singleton_next_site":
        if list(sites) != [frame.get("next_site")]:
            raise FrameAllocationReplanNeeded("singleton alias changed")
        result.pop("open_sites", None)
    else:
        result["open_sites"] = copy.deepcopy(list(sites))
    return result


def singleton_alias_candidate(frame: Mapping[str, object]) -> dict[str, object] | None:
    """The sole permitted inventory compression preserves every semantic site."""
    site = frame.get("next_site")
    if not isinstance(site, Mapping) or frame.get("open_sites") != [site]:
        return None
    key = site_identity(site)
    return render_optional_selection(
        frame,
        {},
        {
            "sites": [copy.deepcopy(dict(site))],
            "open_sites": [key],
            "inventory_identity": [key],
            "selected_identities": [key],
            "next_site": key,
            "encoding_mode": "singleton_next_site",
        },
    )


def _effect_consumer_fact(runtime, capture, body):
    from kernel_host_authorship import ledger_of, row_kinds, structural_class
    from kernel_r200_f046_remember_product_consumer import enabled, select
    from kernel_stateframe_framefit5 import _state_payload
    from kernel_stable_wire_ownership import request_frame_index

    if not enabled() or getattr(runtime, "_unit_policy", None) is not None:
        return None
    messages = body["messages"]
    try:
        index = request_frame_index(messages, capture=capture)
    except (FrameAllocationReplanNeeded, ValueError):
        return None
    if structural_class(messages, index, capture=capture) != "frame":
        return None
    ledger = ledger_of(capture) or ledger_of(runtime)
    kinds = row_kinds(ledger, messages, capture=capture)
    # The frame can be re-rendered after the ledger recorded its original bytes.
    if kinds[index] not in (None, "state") or any(
            kind == "state" and position != index for position, kind in enumerate(kinds)):
        return None
    return select(runtime, _state_payload(messages[index])[1]["framefit5"])


def _with_effect_consumer_fact(body, fact):
    from kernel_stateframe_framefit5 import _is_volatile_tail_message, effect_consumer_tail

    tails = [i for i, row in enumerate(body["messages"]) if _is_volatile_tail_message(row)]
    if len(tails) != 1:
        return None
    index = tails[0]
    row = body["messages"][index]
    content = effect_consumer_tail(row["content"], fact)
    if content is None:
        return None
    messages = list(body["messages"])
    messages[index] = {**row, "content": content}
    return {**body, "messages": messages}


def _with_allocated_consumer(prepared, fact, measure):
    import json
    from dataclasses import replace
    from kernel_cache_layout_view import semantic_messages
    from kernel_r200_f046_remember_product_consumer import MAX_ROW_TOKENS
    from kernel_working_set_allocator import DEFAULT_MODES
    from kernel_working_set_prepare import (
        MAX_MEASUREMENTS, Measurement, PreparedCandidate, Trial, _violations,
    )
    from kernel_working_set_proof import verify_certificate
    from kernel_working_set_render import render_assignment

    if (not fact or measure is None or not isinstance(prepared, PreparedCandidate)
            or len(prepared.trials) >= MAX_MEASUREMENTS):
        return prepared
    certificate = prepared.rendered.certificate
    snapshot = certificate.snapshot
    fixed = _with_effect_consumer_fact(json.loads(snapshot.fixed_body), fact)
    expected = _with_effect_consumer_fact(json.loads(prepared.wire), fact)
    if fixed is None or expected is None:
        return prepared
    if len(json.dumps(expected).encode()) > snapshot.byte_cap:
        return prepared
    try:
        modes = {mode.name: mode for mode in DEFAULT_MODES}
        rendered = render_assignment(
            replace(snapshot, fixed_body=json.dumps(fixed).encode()),
            {key: modes[name] for key, name in certificate.assignment}, certificate.layout,
        )
        # Cached layouts rebase the volatile overlay's offsets. Compare the
        # reconstructed messages so only the added fact can change evidence.
        baseline = json.loads(prepared.wire)
        decoded = json.loads(rendered.wire)
        baseline["messages"] = semantic_messages(baseline["messages"])
        decoded["messages"] = semantic_messages(decoded["messages"])
        stripped = _with_effect_consumer_fact(decoded, None)
        if (stripped is None or sum(str(row.get("content")).count(fact)
                                    for row in decoded["messages"]) != 1
                or stripped != baseline):
            return prepared
        raw = measure(rendered.wire)
        metered = raw if isinstance(raw, Measurement) else Measurement(*raw)
        if (not metered.contract_valid or metered.method != prepared.measurement.method
                or metered.tokenizer_identity != prepared.measurement.tokenizer_identity
                or metered.prompt_tokens - prepared.measurement.prompt_tokens > MAX_ROW_TOKENS
                or _violations(snapshot, rendered, metered)):
            return prepared
        verify_certificate(rendered.wire, rendered.certificate)
    except (ValueError, TypeError, KeyError, OSError):
        return prepared
    trial = Trial(rendered.certificate.assignment_digest, rendered.certificate.body_sha256,
                  len(rendered.wire), metered, ())
    return replace(prepared, rendered=rendered, measurement=metered,
                   trials=(*prepared.trials, trial))


def _with_legacy_consumer(runtime, capture, body):
    import json
    from kernel_cache_layout_candidate import _checked_final_cache_projection
    from kernel_cache_layout_view import FINAL_LAYOUT_NAME, is_cache_layout, semantic_messages
    from kernel_context import token_estimate
    from kernel_context_capacity import _OUTPUT_KEYS, check_capacity, resolve_n_ctx
    from kernel_endpoint_size import ENDPOINT_SIZE
    from kernel_frame_allocation_capture import final_request_tokens, rebind_final_layout_trim
    from kernel_measurement import _prompt_token_estimate
    from kernel_r200_f046_remember_product_consumer import MAX_ROW_TOKENS
    from kernel_request_budget import _request_token_meter
    from kernel_request_diet import request_tokenizer_credentials
    from kernel_request_source_window import source_delivery_satisfied, verify_final_source_delivery
    from kernel_stateframe_capacity import within_protected_capacity

    semantic = body
    projected = is_cache_layout(body["messages"])
    if projected:
        if body["messages"][-1].get("name") != FINAL_LAYOUT_NAME:
            return body
        semantic = {**body, "messages": semantic_messages(body["messages"])}
    fact = _effect_consumer_fact(runtime, capture, semantic)
    if not fact:
        return body
    trial = _with_effect_consumer_fact(semantic, fact)
    if trial is None or trial == semantic:
        return body
    if projected:
        try:
            if _checked_final_cache_projection(semantic) != body:
                return body
            trial = _checked_final_cache_projection(trial)
        except ValueError:
            return body
    inputs = capture._frame_candidate_inputs
    byte_limit, token_limit = inputs["byte_ceiling"], inputs["token_ceiling"]
    wire = json.dumps(trial).encode()
    if len(wire) > byte_limit or token_estimate(trial) > runtime.frame_ceiling:
        return body
    key, endpoint = request_tokenizer_credentials(capture)
    n_ctx = (resolve_n_ctx(endpoint, key or "") if endpoint else None) or ENDPOINT_SIZE.context_tokens
    route = getattr(capture, "frame_route", None)
    route_limit = route.settings.get("context_tokens") if route is not None else None
    if type(route_limit) is int and route_limit > 0:
        n_ctx = min(n_ctx, route_limit)
    output = next((trial[name] for name in _OUTPUT_KEYS
                   if type(trial.get(name)) is int and trial[name] > 0), 0)
    if check_capacity(_prompt_token_estimate(trial), output, n_ctx)[0] != "ok":
        return body
    meter = _request_token_meter(runtime, capture)
    baseline, method = final_request_tokens(body, meter, capture)
    metered, trial_method = final_request_tokens(trial, meter, capture, wire_bytes=len(wire))
    if (trial_method != method or metered > token_limit or metered - baseline > MAX_ROW_TOKENS
            or not within_protected_capacity(trial, exact_tokenizer=method == "tokenize")):
        return body
    plan = copy.deepcopy(inputs["source_plan"])
    receipt = verify_final_source_delivery(plan, trial["messages"], payload=wire,
                                           sources=inputs["sources"])
    if not source_delivery_satisfied(receipt):
        return body
    rebind_final_layout_trim(trial, meter, capture, byte_limit=byte_limit,
                            token_limit=token_limit, wire=wire)
    inputs["receipt"] = copy.deepcopy(receipt)
    capture.request_source_final_delivery = receipt
    return trial


def install_effect_consumer_allocation():
    """Attach the optional selection to the two complete-envelope fitters."""
    from contextvars import ContextVar
    from functools import wraps
    import kernel_final_wire_allocation as final_wire
    import kernel_working_set_live as live
    import kernel_working_set_prepare as solver

    active_fact = ContextVar("effect_consumer_fact", default=None)
    prepare_live = live.prepare_live
    prepare_candidate = solver.prepare_candidate
    prepare_hook_body = final_wire._prepare_qualified_hook_body

    @wraps(prepare_live)
    def live_fit(capture, runtime, body, **kwargs):
        token = active_fact.set(_effect_consumer_fact(runtime, capture, body))
        try:
            return prepare_live(capture, runtime, body, **kwargs)
        finally:
            active_fact.reset(token)

    @wraps(prepare_candidate)
    def allocated_fit(snapshot, *, measure=None):
        prepared = prepare_candidate(snapshot, measure=measure)
        return _with_allocated_consumer(prepared, active_fact.get(), measure)

    @wraps(prepare_hook_body)
    def legacy_fit(runtime, body, request_id):
        result = prepare_hook_body(runtime, body, request_id)
        capture = getattr(runtime, "_active_capture", None)
        if (getattr(capture, "request_id", None) != request_id
                or getattr(capture, "_working_set_certificate", None) is not None
                or not hasattr(capture, "_frame_candidate_inputs")):
            return result
        # The legacy cache projection and its source fitting are complete at
        # this seam. Price their exact body with the optional fact added.
        return _with_legacy_consumer(runtime, capture, result)

    live.prepare_live = live_fit
    solver.prepare_candidate = allocated_fit
    final_wire._prepare_qualified_hook_body = legacy_fit
