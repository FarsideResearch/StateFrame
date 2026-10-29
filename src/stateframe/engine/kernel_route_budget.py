"""Model-facing route projection bound to the live observation probe runner."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_ROUTE_BUDGET"
START = "[STATEFRAME recovery view]"
END = "[/STATEFRAME recovery view]"


def enabled() -> bool:
    return mode() == "on"


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _text(value: object, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def _tool_names(tools: object, runtime: object) -> tuple[str, ...]:
    if isinstance(tools, Mapping):
        tools = (tools,)
    names: list[str] = []
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        for tool in tools:
            function = tool.get("function") if isinstance(tool, Mapping) else None
            name = _text(function.get("name"), 100) if isinstance(function, Mapping) else ""
            if name and name not in names:
                names.append(name)
    if not names:
        names = [_text(item, 100) for item in getattr(runtime, "completion_menu", ()) or ()]
    return tuple(name for name in names if name)


def _budget(runtime: object) -> dict[str, object]:
    # The live runner is measurement.observation_probe_runner.  Runtime-level
    # counters remain a compatibility fallback for older replay fixtures.
    owners = (getattr(runtime, "measurement", None), runtime)
    for owner in owners:
        runner = getattr(owner, "observation_probe_runner", None)
        budget = getattr(runner, "budget", None)
        count = getattr(budget, "count", None)
        limit = getattr(budget, "limit", None)
        if type(count) is int and type(limit) is int:
            return {"count": max(0, count), "limit": max(0, limit),
                    "remaining": max(0, limit - count)}
    budget = getattr(runtime, "probe_budget", None)
    count = getattr(budget, "count", None)
    limit = getattr(budget, "limit", None)
    if type(count) is int and type(limit) is int:
        return {"count": max(0, count), "limit": max(0, limit),
                "remaining": max(0, limit - count)}
    remaining = getattr(runtime, "probe_remaining", None)
    return {"count": None, "limit": None,
            "remaining": max(0, remaining) if type(remaining) is int else None}


def _receipts(runtime: object) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for name in ("probe_receipts", "observation_probe_receipts"):
        value = getattr(runtime, name, ())
        if isinstance(value, Mapping):
            value = value.values()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows.extend(row for row in value if isinstance(row, Mapping))
    return rows[-12:]


def _next_action(offered: set[str], remaining: object) -> str:
    for candidate in ("run_tests", "read", "grep", "str_replace", "probe"):
        if candidate in offered and not (candidate == "probe" and remaining == 0):
            return candidate
    return "none currently offered"


def render(runtime: object, decision: object, tools: object) -> str:
    if not enabled():
        return ""
    budget = _budget(runtime)
    offered = set(_tool_names(tools, runtime))
    failed: list[dict[str, object]] = []
    denied: list[dict[str, object]] = []
    for row in _receipts(runtime):
        status = _text(row.get("status", row.get("result_class", row.get("result", "UNKNOWN"))), 80).upper()
        item = {"route": _text(row.get("tool", row.get("route", "probe")), 80),
                "status": status,
                "detail": _text(row.get("reason", row.get("detail", row.get("message", ""))))}
        if status in {"POLICY_DENIED", "DENIED", "REFUSED", "NOT_RUN"}:
            denied.append(item)
        elif status not in {"COMPLETED", "HOST_OBSERVED", "PASS", "GREEN", "OK"}:
            failed.append(item)
    payload = {"probe_budget": budget, "failed_routes": failed[-8:],
               "denied_routes": denied[-8:], "offered_routes": sorted(offered),
               "next_action": _next_action(offered, budget.get("remaining")),
               "completion_outcome": _text(getattr(decision, "outcome", ""), 80)}
    return START + "\n" + json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), default=str) + "\n" + END


def filter_menu(runtime: object, tools: list[dict]) -> list[dict]:
    """Remove an exhausted probe from the actual dispatch menu."""
    if not enabled() or _budget(runtime).get("remaining") != 0:
        return tools
    filtered = [tool for tool in tools
                if tool.get("function", {}).get("name") != "probe"]
    if len(filtered) == len(tools):
        return tools
    names = [tool.get("function", {}).get("name") for tool in filtered]
    runtime.completion_menu = tuple(name for name in names if name)
    runtime._question_done_available = "done" in runtime.completion_menu
    runtime.journal.emit("route_budget_menu_projected", producer="kernel_route_budget",
                         budget=_budget(runtime), removed=["probe"], offered=names,
                         dispatch=False)
    return filtered


__all__ = ["END", "MODE_ENV", "START", "enabled", "filter_menu", "mode", "render"]
