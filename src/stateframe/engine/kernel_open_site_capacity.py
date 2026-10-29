"""Lossless location inventory with one resident, editable source frontier.

Full windows remain host allocation inputs. On the wire, secondary locations
are exact range receipts or bounded read routes, never implied edit authority.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path

LOCATION_MODE = "location_inventory"
_LOCATION_FIELDS = ("source", "path", "line", "literal", "revision", "lines", "content_sha256")


def location_inventory_candidate(frame: Mapping[str, object]) -> dict[str, object] | None:
    """Extend the singleton alias to a complete, source-independent inventory."""
    from kernel_stateframe_capacity import _delivered, _segments

    sites = frame.get("open_sites")
    if (not isinstance(sites, list) or len(sites) < 2
            or not isinstance(sites[0], Mapping)
            or sites[0] != frame.get("next_site")
            or not isinstance(sites[0].get("current_bytes"), str)):
        return None
    if frame.get("site_inventory_encoding") == LOCATION_MODE:
        return None
    segments = _segments((), {"open_sites": [sites[0]]}, ())
    rows = []
    for site in sites:
        if not isinstance(site, Mapping) or any(key not in site for key in _LOCATION_FIELDS):
            return None
        lines = site["lines"]
        if (not isinstance(lines, Mapping) or type(lines.get("start")) is not int
                or type(lines.get("end")) is not int
                or not 1 <= lines["start"] <= lines["end"]):
            return None
        row = {key: copy.deepcopy(site[key]) for key in _LOCATION_FIELDS}
        if _delivered(site, segments):
            row["range_delivery"] = "same_request"
        else:
            row["range_delivery"] = "not_carried"
            row["retrieval_route"] = {
                "tool": "read",
                "arguments": {"path": site["path"],
                              "start_line": site["lines"]["start"],
                              "end_line": site["lines"]["end"]},
            }
        rows.append(row)
    result = copy.deepcopy(dict(frame))
    result["open_sites"] = rows
    result["site_inventory_encoding"] = LOCATION_MODE
    return result


def expand_location_inventory(frame: Mapping[str, object], root: Path) -> list[dict[str, object]]:
    """Verify omitted ranges against disk, returning detached host-only windows."""
    from kernel_stateframe_framefit5 import (
        FrameFit5Error, _CandidateSourceChanged, _render_next_site,
        _safe_relative_path, _verify_next_site_bytes,
    )

    sites = frame.get("open_sites")
    if frame.get("site_inventory_encoding") != LOCATION_MODE:
        return copy.deepcopy(sites if isinstance(sites, list) else [])
    if (not isinstance(sites, list) or len(sites) < 2
            or any(not isinstance(site, Mapping) for site in sites)
            or not isinstance(frame.get("next_site"), Mapping)):
        raise FrameFit5Error("location inventory lacks its exact resident frontier")
    frontier = frame["next_site"]
    _verify_next_site_bytes(frontier, root)
    if any(sites[0].get(key) != frontier.get(key) for key in _LOCATION_FIELDS):
        raise FrameFit5Error("location inventory frontier identity changed")
    full = [copy.deepcopy(dict(frontier))]
    for row in sites[1:]:
        path = _safe_relative_path(row.get("path"))
        line = row.get("line")
        if type(line) is not int or line < 1:
            raise FrameFit5Error("location inventory has an invalid line")
        fresh = _render_next_site(root, path, line, source=str(row.get("source") or ""),
                                  literal=str(row.get("literal") or ""))
        if any(row.get(key) != fresh.get(key) for key in _LOCATION_FIELDS):
            raise _CandidateSourceChanged("location inventory source range changed")
        _verify_next_site_bytes(fresh, root)
        full.append(fresh)
    expected = location_inventory_candidate({"open_sites": full, "next_site": full[0]})
    if expected is None or expected["open_sites"] != sites:
        raise FrameFit5Error("location inventory receipt or retrieval route changed")
    return full


def validate_location_routes(frame: Mapping[str, object], runtime: object, tools: object) -> None:
    """Every omitted window must have a read action admitted on this exact wire."""
    from kernel_completion_owner import _current_tool_admission
    from kernel_request_source_window import _wire_deferred_notice_state
    from kernel_stateframe_framefit5 import FrameFit5Error
    from kernel_tool_surface import runner_dispatchable

    if frame.get("site_inventory_encoding") != LOCATION_MODE:
        return
    if not any(row.get("range_delivery") == "not_carried" for row in frame["open_sites"]):
        return
    admitted, _reason = _current_tool_admission(runtime, "read")
    wire_tools = (list(tools) if isinstance(tools, Sequence)
                  and not isinstance(tools, (str, bytes, bytearray)) else [])
    _notice, on_wire = _wire_deferred_notice_state({"tools": wire_tools, "messages": []}, ())
    if not admitted or not on_wire or not runner_dispatchable("read", runtime=runtime):
        raise FrameFit5Error("location inventory has no executable read route")
