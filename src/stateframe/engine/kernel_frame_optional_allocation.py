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
        _read_source,
        _requested_paths,
        headcarry_owed_reads_enabled,
        owed_read_rows,
    )

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
        raw = _read_source(root, path)
        dependencies[path] = (
            "sha256:" + hashlib.sha256(raw).hexdigest()
            if raw is not None
            else "UNAVAILABLE"
        )
        if raw is not None:
            for dependency in _import_targets(root, path, raw):
                source = _read_source(root, dependency)
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
        raw = _read_source(root, path)
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
    site_ids = [identity(site) for site in sites]
    inventory = {
        "inventory_identity": site_ids,
        "selected_identities": list(site_ids),
        "open_sites": list(site_ids),
        "next_site": identity(frame["next_site"])
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
        from kernel_stateframe_headcarry import _read_source

        for path, revision in optional.get("dependency_revisions", {}).items():
            raw = _read_source(root, path)
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
    key = identity(site)
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
