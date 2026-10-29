"""Opt-in phased StateFrame rendering with a hard request budget.

The module is intentionally presentation-only.  The durable frame remains the
source of truth; this layer selects a small, typed projection and compact tool
surface for the model-facing request.  ``STATEFRAME_FRAME_BUDGET=off`` is the
legacy arm and callers do not import or invoke this module on that path.
"""
from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence

from kernel_stateframe_gears import GEAR_ORDER, GearController, token_accounting


MODE_ENV = "STATEFRAME_FRAME_BUDGET"
SCHEMA = "stateframe-frame-budget/1"
TRUTH_SCHEMA = "stateframe-truth/1"
BOOT_TOKEN_LIMIT = 2500
FIRST_REQUEST_TOKEN_LIMIT = 5000
STEADY_REQUEST_TOKEN_LIMIT = 6000
SOURCE_CHAR_LIMIT = 3200
DIFF_CHAR_LIMIT = 4800
TRUTH_CHAR_LIMIT = 5000

_PASS = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "OK", "SATISFIED", "CERTIFIED"})
_FAIL = frozenset({"FAIL", "FAILED", "RED", "ERROR", "TIMEOUT", "VIOLATED", "CONFLICT"})
_TRIGGER_TARGETS = {
    "model_requested_read": "g1",
    "read_requested": "g1",
    "read_completed": "g2",
    "edit_applied": "g2",
    "check_run": "g3",
    "green": "g3",
}
_EDIT_TOOLS = frozenset({"edit", "str_replace", "file_create", "replace_module", "write", "replace"})
_READ_TOOLS = frozenset({
    "read", "grep", "list", "ls", "recall_evidence", "recall",
    "recall_generated_artifact", "read_ticket", "probe",
    "where_defined", "who_calls", "references",
})
_CHECK_TOOLS = frozenset({"run_tests", "test", "pytest"})
PHASE_BY_GEAR = {
    "g0": "BOOT",
    "g1": "LOCATE_READ",
    "g2": "EDIT",
    "g3": "VERIFY",
}
_LEGACY_ASYNC_TOOLS = frozenset({
    "job_status", "job_output", "job_kill", "launch_job", "kill_job", "dispute_check",
})


def mode(value: object = None) -> str:
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return raw.strip().lower()


def enabled(value: object = None) -> bool:
    return mode(value) == "on"


def _deep(value: object) -> object:
    return copy.deepcopy(value)


def _seq(value: object) -> list:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    if value is None:
        return []
    return [value]


def _text(value: object, limit: int | None = None) -> str:
    result = value if isinstance(value, str) else str(value or "")
    if limit is not None and len(result) > limit:
        return result[:max(0, limit - 28)].rstrip() + "\n…[truncated]"
    return result


def _tokens(value: object) -> int:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, default=str).encode("utf-8")
    return int(math.ceil(len(encoded) / 4)) if encoded else 0


def _request_tokens(value: object) -> int:
    """Use the runtime estimate as a floor while retaining byte-based accounting."""
    try:
        from kernel_context import token_estimate
        return max(_tokens(value), int(token_estimate(value)))
    except (ImportError, TypeError, ValueError):
        return _tokens(value)


def _status(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("verdict", "result_class", "status", "state", "result", "outcome"):
            if value.get(key) not in (None, ""):
                return str(value[key]).strip().upper()
    if value not in (None, ""):
        return str(value).strip().upper()
    return "UNKNOWN"


def _pick(frame: Mapping[str, object], *keys: str, default: object = None) -> object:
    for key in keys:
        value = frame.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _status_from_frame(frame: Mapping[str, object]) -> str:
    for key in ("verification", "completion_verdict", "terminal_status", "test_verdict",
                "public_check_status", "last_test", "test_receipts", "applicable_check_statuses"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            statuses = [_status(item) for item in value.values()]
            if any(item in _FAIL for item in statuses):
                return next(item for item in statuses if item in _FAIL)
            if any(item in _PASS for item in statuses):
                return next(item for item in statuses if item in _PASS)
            result = _status(value)
            if result != "UNKNOWN":
                return result
        elif value not in (None, ""):
            result = _status(value)
            if result != "UNKNOWN":
                return result
    return "UNKNOWN"


def _green_lock(frame: Mapping[str, object]) -> object:
    value = _pick(frame, "green_lock", "green_lock_state", "lock_state", "green_lock_message",
                  default="UNKNOWN")
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if str(key) not in {"state", "status", "reason", "revision", "bound_revision"}:
                continue
            result[str(key)] = _text(item, 160) if isinstance(item, (str, bytes)) else _deep(item)
        return result
    return {"state": _text(value, 160)}


def _compact_receipt(row: object) -> dict[str, object]:
    if not isinstance(row, Mapping):
        return {"value": _text(row, 180)}
    keys = ("receipt_id", "id", "kind", "tool", "result_class", "status", "verdict",
            "path", "target_path", "revision", "freshness", "selector", "reason")
    result = {}
    for key in keys:
        value = row.get(key)
        if value in (None, "", [], {}):
            continue
        result[key] = _text(value, 100) if isinstance(value, (str, bytes)) else _deep(value)
    return result


def _compact_obligations(frame: Mapping[str, object]) -> list[dict[str, object]]:
    raw = _pick(frame, "open_obligations", "open_work", "obligations", "requirements_ledger",
                default=())
    result: list[dict[str, object]] = []
    for index, row in enumerate(_seq(raw), 1):
        if isinstance(row, Mapping):
            item = {
                "id": _text(row.get("id", row.get("obligation_id", f"open-{index}")), 100),
                "status": _status(row.get("status", row.get("state", "OPEN"))),
            }
            for key in ("summary", "reason", "selector", "path"):
                if row.get(key) not in (None, ""):
                    item[key] = _text(row[key], 120)
            result.append(item)
        elif row not in (None, ""):
            result.append({"id": _text(row, 100), "status": "OPEN"})
    return result[:32]


def _compact_receipts(frame: Mapping[str, object]) -> list[dict[str, object]]:
    raw = _pick(frame, "edit_receipts", "relevant_receipts", "receipts", "test_receipts",
                "recent_test_feedback", default=())
    return [_compact_receipt(row) for row in _seq(raw)[:12]]


def _compact_statuses(frame: Mapping[str, object]) -> dict[str, str]:
    raw = _pick(frame, "applicable_check_statuses", "check_statuses", "public_check_statuses",
                "current_public_checks", default={})
    if isinstance(raw, Mapping):
        return {_text(key, 120): _status(value) for key, value in list(raw.items())[:32]}
    if raw not in (None, ""):
        return {"current": _status(raw)}
    return {}


def _green_value(status: str, frame: Mapping[str, object]) -> bool:
    if status in _FAIL:
        return False
    explicit = _pick(frame, "green", "is_green", "green_locked", default=None)
    if explicit is not None:
        if isinstance(explicit, str):
            return explicit.strip().lower() in {"1", "true", "yes", "green", "pass"}
        return bool(explicit)
    return status in _PASS


def truth_projection(frame: Mapping[str, object]) -> dict[str, object]:
    """Return the compact host-verified facts that no phase may omit."""
    if not isinstance(frame, Mapping):
        raise TypeError("truth projection requires a mapping frame")
    status = _status_from_frame(frame)
    lock = _green_lock(frame)
    revisions = _pick(frame, "current_revision", "product_revision", "revision", default=None)
    truth = {
        "schema": TRUTH_SCHEMA,
        "authority": "host",
        "verdict": status,
        "green": _green_value(status, frame),
        "green_lock": lock,
        "lock_state": lock.get("state", lock.get("status", "UNKNOWN"))
            if isinstance(lock, Mapping) else _text(lock, 120),
        "check_statuses": _compact_statuses(frame),
        "open_obligations": _compact_obligations(frame),
        "edit_receipts": _compact_receipts(frame),
    }
    test_rows = _pick(frame, "test_receipts", "recent_test_feedback", "verification_receipts", default=())
    if test_rows not in (None, "", [], {}):
        truth["test_receipts"] = [_compact_receipt(row) for row in _seq(test_rows)[:12]]
    completion = frame.get("completion_status")
    if isinstance(completion, Mapping):
        truth["completion_status"] = {
            key: _deep(completion[key]) for key in ("state", "status", "reason", "open_ids")
            if key in completion
        }
    if revisions not in (None, ""):
        truth["current_revision"] = _deep(revisions)
    # A host-provided reason is useful to decide whether a red state is retryable,
    # but it is bounded and remains a fact rather than a model-authored claim.
    reason = _pick(frame, "verification_reason", "completion_reason", "green_lock_message", default=None)
    if reason not in (None, ""):
        truth["reason"] = _text(reason, 360)
    return truth


def _assumptions(frame: Mapping[str, object]) -> list[dict[str, object]]:
    raw = _pick(frame, "assumptions", "model_assumptions", "stated_assumptions", default=())
    observations = _pick(frame, "assumption_observations", "public_assumption_observations",
                         default={})
    observations = observations if isinstance(observations, Mapping) else {}
    result = []
    for index, row in enumerate(_seq(raw), 1):
        if isinstance(row, Mapping):
            item = {"id": str(row.get("id", f"assumption-{index}")),
                    "status": str(row.get("status", "UNVERIFIED")).upper()}
            observed = observations.get(item["id"])
            if isinstance(observed, Mapping):
                item["status"] = str(observed.get("status", observed.get("result", "UNVERIFIED"))).upper()
                if observed.get("evidence") not in (None, ""):
                    item["evidence"] = _text(observed["evidence"], 180)
            elif observed not in (None, ""):
                item["status"] = str(observed).upper()
            value = row.get("text", row.get("assumption", row.get("value")))
            if value not in (None, ""):
                item["text"] = _text(value, 240)
            result.append(item)
        elif row not in (None, ""):
            result.append({"id": f"assumption-{index}", "status": "UNVERIFIED",
                           "text": _text(row, 240)})
    return result[:16]


def _source_rows(frame: Mapping[str, object], context: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    raw = _pick(frame, "current_source_spans", "relevant_source_spans", "source_spans",
                "recent_reads", "current_sources", "observed_sources", default=())
    rows = [row for row in _seq(raw) if isinstance(row, Mapping)]
    if not rows and isinstance(context, Mapping):
        rows = [row for row in _seq(context.get("source_spans")) if isinstance(row, Mapping)]
        if not rows and context.get("path") and isinstance(context.get("source"), str):
            rows = [{"path": context["path"], "content": context["source"],
                     "start_line": 1, "end_line": len(context["source"].splitlines()),
                     "coverage": "complete"}]
    # A read/edit phase may expose only host-selected paths: the model's typed
    # request, the current host read, or a where-used site.  A full source list
    # is retained in the durable frame, but is not a reason to render every
    # file in the next request.
    requested = _pick(frame, "requested_files", "model_requested_files", "needed_files",
                      default=())
    if isinstance(context, Mapping):
        requested = [*list(_seq(requested)), *list(_seq(context.get("requested_files")))]
        if context.get("path") not in (None, ""):
            requested.append(context["path"])
    allowed: set[str] = set()
    for item in _seq(requested):
        value = item.get("path") if isinstance(item, Mapping) else item
        if value not in (None, ""):
            allowed.add(str(value).replace("\\", "/").lstrip("./"))
    for item in _where_used(frame):
        value = item.get("path")
        if value not in (None, ""):
            allowed.add(str(value).replace("\\", "/").lstrip("./"))
    if allowed:
        rows = [row for row in rows
                if str(row.get("path", row.get("requested_path", "")))
                .replace("\\", "/").lstrip("./") in allowed]
    else:
        rows = []
    return rows


def _slice_source(row: Mapping[str, object], limit: int = SOURCE_CHAR_LIMIT) -> dict[str, object]:
    path = _text(row.get("path", row.get("requested_path", "")), 240)
    content = _text(row.get("content", row.get("source", "")))
    start = row.get("start_line", row.get("line", 1))
    end = row.get("end_line", row.get("selected_end_line", start))
    try:
        start_i, end_i = int(start), int(end)
    except (TypeError, ValueError):
        start_i, end_i = 1, 1
    if len(content) <= limit:
        selected = content
        truncated = bool(row.get("more_lines_available", False) or row.get("omitted_chars"))
    else:
        lines = content.splitlines()
        center = max(0, min(len(lines) - 1, start_i - 1))
        first = max(0, center - 12)
        selected_lines: list[str] = []
        for line in lines[first:]:
            candidate = "\n".join(selected_lines + [line])
            if len(candidate) > limit:
                break
            selected_lines.append(line)
        selected = "\n".join(selected_lines)
        start_i = first + 1
        end_i = start_i + len(selected_lines) - 1
        truncated = True
    item: dict[str, object] = {
        "path": path,
        "start_line": start_i,
        "end_line": end_i,
        "revision": _deep(row.get("revision", row.get("file_revision"))),
        "coverage": _text(row.get("coverage", "partial"), 40),
        "content": selected,
    }
    if truncated:
        item["truncated"] = True
        item["pull_more"] = {
            "type": "read_range",
            "tool": "read",
            "path": path,
            "line": end_i + 1,
            "max_lines": 120,
            "host_verified": False,
        }
    return {key: value for key, value in item.items() if value not in (None, "")}


def _where_used(frame: Mapping[str, object]) -> list[dict[str, object]]:
    raw = _pick(frame, "where_used_closure", "where_used", "callers", "caller_sites", default=())
    if isinstance(raw, Mapping):
        raw = raw.get("sites", raw.get("callers", raw.get("references", ())))
    result = []
    for row in _seq(raw):
        if isinstance(row, Mapping):
            item = {}
            for key in ("path", "line", "symbol", "kind", "reason", "reference"):
                if row.get(key) not in (None, ""):
                    item[key] = _text(row[key], 180)
            if item:
                result.append(item)
        elif row not in (None, ""):
            result.append({"reference": _text(row, 180)})
    return result[:24]


def _compact_repo_map(value: object) -> object:
    if isinstance(value, Mapping):
        result = {}
        for key, item in list(value.items())[:16]:
            name = _text(key, 80)
            if isinstance(item, (str, bytes)):
                result[name] = _text(item, 160)
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                result[name] = [_text(row, 120) for row in list(item)[:24]]
            else:
                result[name] = _deep(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_text(item, 120) for item in list(value)[:24]]
    return _text(value, 800)


def _minimal_action(frame: Mapping[str, object]) -> object:
    value = _pick(frame, "last_action", "action", "latest_action", default=None)
    if not isinstance(value, Mapping):
        return _text(value, 320) if value not in (None, "") else None
    return {key: _text(value[key], 240) for key in
            ("tool", "path", "result_class", "status", "outcome", "receipt_id", "reason")
            if value.get(key) not in (None, "", [], {})}


def phase_projection(frame: Mapping[str, object], gear: str,
                     *, context: Mapping[str, object] | None = None) -> dict[str, object]:
    """Build the phase projection without mutating the durable frame."""
    if not isinstance(frame, Mapping):
        raise TypeError("phase projection requires a mapping frame")
    if gear not in GEAR_ORDER:
        raise ValueError(f"unknown frame-budget gear: {gear}")
    context = context if isinstance(context, Mapping) else {}
    truth = truth_projection(frame)
    ticket = _text(_pick(frame, "ticket", "task_brief", "brief", "msg", "objective", default=""), 2400)
    projection: dict[str, object] = {
        "schema": SCHEMA,
        "gear": gear,
        "phase": PHASE_BY_GEAR[gear],
        "ticket": ticket,
        "truth": truth,
        "assumptions": _assumptions(frame),
    }
    if gear == "g0":
        targets = _pick(frame, "target_files", "files", "target_paths", default=())
        repo_map = _pick(frame, "repo_map", "repository_map", "repo", default=None)
        boot_packet = context.get("boot_packet")
        if isinstance(boot_packet, Mapping):
            facts = boot_packet.get("facts")
            if isinstance(facts, Mapping):
                if not targets:
                    targets = facts.get("target_files", ())
                if repo_map in (None, "", [], {}):
                    repo_map = {"target_files": _deep(facts.get("target_files", ())),
                                "interpreter": _text(facts.get("interpreter", ""), 120)}
        if targets:
            projection["needed_files"] = [_text(row.get("path", row), 240)
                                           if isinstance(row, Mapping) else _text(row, 240)
                                           for row in _seq(targets)][:16]
        if repo_map not in (None, "", [], {}):
            projection["repo_map"] = _compact_repo_map(repo_map)
        projection["request"] = {
            "type": "assumptions_plan_files",
            "fields": ["assumptions", "plan", "needed_files"],
            "host_verified": False,
        }
        return projection

    if gear == "g3":
        receipts = _compact_receipts(frame)
        if receipts:
            projection["verification_receipts"] = receipts
        projection["done_allowed"] = bool(truth["green"])
        return projection

    spans = [_slice_source(row) for row in _source_rows(frame, context)]
    if spans:
        projection["source_slices"] = spans[:8]
    callers = _where_used(frame)
    if callers:
        projection["where_used"] = callers
    diff = _pick(frame, "current_diff", "diff", "current_changes", "changes", default="")
    if diff not in (None, "", [], {}):
        projection["current_diff"] = _text(diff, DIFF_CHAR_LIMIT if gear != "g2" else 7600)
    action = _minimal_action(frame)
    if action not in (None, "", {}, []):
        projection["last_action"] = action
    if gear == "g1":
        needed = _pick(frame, "requested_files", "needed_files", default=())
        if needed:
            projection["requested_files"] = [_text(item, 240) for item in _seq(needed)][:24]
    elif gear == "g2":
        working = _pick(frame, "working_set_manifest", "working_set", default=None)
        if working not in (None, "", [], {}):
            projection["working_set"] = _deep(working)
        receipts = _compact_receipts(frame)
        if receipts:
            projection["edit_receipts"] = receipts
    return projection


def render_projection(frame: Mapping[str, object], gear: str,
                      *, context: Mapping[str, object] | None = None) -> str:
    return "[StateFrame]\n" + json.dumps(
        phase_projection(frame, gear, context=context), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False, default=str)


def account_projection(projection: Mapping[str, object]) -> dict[str, object]:
    """Use the installed gear accountant for the budget projection."""
    return token_accounting(projection)


def _tool_name(tool: Mapping[str, object]) -> str:
    function = tool.get("function") if isinstance(tool, Mapping) else None
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(tool.get("name", "")) if isinstance(tool, Mapping) else ""


def _compact_parameters(parameters: object, *, add_read_range: bool = False) -> object:
    if not isinstance(parameters, Mapping):
        return parameters
    result = {key: _deep(value) for key, value in parameters.items()
              if key not in {"title", "examples"}
              and (add_read_range or key != "description")}
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        compacted = {}
        for name, value in properties.items():
            if isinstance(value, Mapping):
                compacted[name] = {key: _deep(item) for key, item in value.items()
                                   if key not in {"title", "examples"}
                                   and (add_read_range or key != "description")}
            else:
                compacted[name] = _deep(value)
        if add_read_range:
            compacted.setdefault("max_lines", {
                "type": "integer", "minimum": 1, "maximum": 1000, "default": 120,
                "description": "Lines per cursor page; default 120, maximum 1000. "
                               "For an inclusive end_line range, omit max_lines.",
            })
            compacted.setdefault("max_chars", {
                "type": "integer", "minimum": 512, "maximum": 64000, "default": 12000,
                "description": "Character budget; default 12000, range 512-64000. "
                               "A clipped page returns a revision-bound next cursor.",
            })
        result["properties"] = compacted
    elif add_read_range:
        result["properties"] = {
            "max_lines": {
                "type": "integer", "minimum": 1, "maximum": 1000, "default": 120,
                "description": "Lines per cursor page; default 120, maximum 1000. "
                               "For an inclusive end_line range, omit max_lines.",
            },
            "max_chars": {
                "type": "integer", "minimum": 512, "maximum": 64000, "default": 12000,
                "description": "Character budget; default 12000, range 512-64000. "
                               "A clipped page returns a revision-bound next cursor.",
            },
        }
    return result


# Tools whose description/parameters ARE a host contract (the single EditAuthority
# description the release runner asserts on).  Compaction must never rewrite them.
CONTRACT_TOOLS = frozenset({"edit", "str_replace", "grep"})
PROBE_DESCRIPTION = (
    "Host probe action: execute one bounded Python observation-only snippet. "
    "No shell commands, pytest or unittest runners, writes, network, or workspace mutation."
)
_TOOL_DESCRIPTIONS = {
    "read": (
        "Read source from a path and optional range. max_chars defaults to 12000 "
        "(maximum 64000) and can clip the page; the result reports the effective "
        "delivered span and a revision-bound next cursor when more source remains. "
        "The source body has no line-number gutter. Use where_defined or references "
        "to locate a symbol instead of paging by guesswork."
    ),
    "recall": (
        "Retrieve one indexed run item by its id. The item is queued into the next "
        "frame; check next_frame in the result. Use where_defined or references "
        "for repository symbol locations."
    ),
    "recall_evidence": (
        "Retrieve retained evidence by its host-issued reference. The result carries "
        "the stored evidence reference and provenance; it is not a fresh source read."
    ),
    "recall_generated_artifact": (
        "Retrieve a historical generated proposal by its listed reference. It is an "
        "unverified proposal, not current source. Use offset and limit to page it, "
        "then follow next_offset when present."
    ),
    "where_defined": (
        "Find the repository definition for a symbol. Returns its file location and "
        "coverage status; use this before reading a large file to locate the symbol."
    ),
    "references": (
        "Find a symbol's repository definition plus its bounded caller and callee "
        "references. The result includes coverage and counts."
    ),
    "who_calls": (
        "Find direct callers of a symbol; set transitive=true for a bounded caller "
        "search. Partial results report their lower-bound status."
    ),
}


def _tool_description(name: str) -> str:
    return _TOOL_DESCRIPTIONS.get(name, f"Host {name} action.")


def _compact_tool(tool: Mapping[str, object], name: str) -> dict[str, object]:
    result = copy.deepcopy(dict(tool))
    if name in CONTRACT_TOOLS:
        return result
    function = result.get("function")
    if isinstance(function, Mapping):
        function = dict(function)
        function["description"] = PROBE_DESCRIPTION if name == "probe" else _tool_description(name)
        function["parameters"] = _compact_parameters(
            function.get("parameters", {}), add_read_range=name == "read")
        result["function"] = function
    else:
        result["description"] = PROBE_DESCRIPTION if name == "probe" else _tool_description(name)
    return result


def _truth_from_value(value: object) -> dict[str, object]:
    if isinstance(value, Mapping) and "green" in value:
        return dict(value)
    status = _status(value)
    return {"verdict": status, "green": status in _PASS}


def compact_tool_surface(tools: Sequence[Mapping[str, object]], gear: str,
                         *, truth: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    """Gate rare tools by host phase and shorten schemas without changing names."""
    if gear not in GEAR_ORDER:
        raise ValueError(f"unknown frame-budget gear: {gear}")
    truth = _truth_from_value(truth or {})
    verdict = _status(truth.get("verdict"))
    green = bool(truth.get("green", verdict in _PASS))
    if gear == "g0":
        allowed = _READ_TOOLS
    elif gear == "g1":
        allowed = _READ_TOOLS
    elif gear == "g2":
        allowed = _READ_TOOLS | _EDIT_TOOLS | _CHECK_TOOLS | {"job", "read_ticket", "dispute"}
    else:
        allowed = _EDIT_TOOLS | _READ_TOOLS | _CHECK_TOOLS
        if green:
            allowed = frozenset({"run_tests", "test", "pytest", "done"}) | _READ_TOOLS
        else:
            allowed |= {"job", "read_ticket", "dispute"}
    result = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        name = _tool_name(tool)
        if name in _LEGACY_ASYNC_TOOLS:
            continue
        if name == "done" and (gear != "g3" or not green):
            continue
        if name not in allowed:
            continue
        result.append(_compact_tool(tool, name))
    return result


def compact_system_prompt(gear: str) -> str:
    phase = PHASE_BY_GEAR[gear].replace("_", "/")
    generic = ("You are a careful workspace editor. Host StateFrame facts are authoritative: "
               "never turn prose into a verdict, never claim green without a host PASS, and "
               "keep open obligations and receipts in view. Current host phase: " + phase + ". "
               "Use only offered tools. State assumptions and the files/plan you need; "
               "promotions happen only after host-observed actions.")
    from kernel_harness_card import enabled as harness_card_enabled, system_prefix
    return system_prefix(generic) if harness_card_enabled() else generic


def prompt_guidance(variant: str = "v3") -> str:
    """Small system-prompt guidance used only by the budget arm."""
    base = compact_system_prompt("g0")
    from kernel_intent_carry import f4_enabled as _intent_carry_f4_enabled
    if _intent_carry_f4_enabled():
        # r46 STATEFRAME_INTENT_CARRY_F4 (tt_output_gap_v2 item 2): the system prompt is built
        # once, so "Current host phase: BOOT" stayed on the wire for the whole run.  The frame
        # carries the real phase; drop the static sentence.
        base = base.replace("Current host phase: " + PHASE_BY_GEAR["g0"].replace("_", "/") + ". ", "")
    return (base + "\n"
            "Boot asks for assumptions, a plan, and needed files. Read only the "
            "host-requested range; edits and checks advance the host phase.")


def _message_content(message: object) -> str:
    return str(message.get("content", "")) if isinstance(message, Mapping) else ""


def _is_optional(content: str) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in ("job table", "background job", "dispute", "optional source",
                                                 "worked example", "method card", "tool contract"))


def _request_blocks(messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]]) -> dict[str, int]:
    blocks: dict[str, int] = {}
    for message in messages:
        content = _message_content(message)
        lowered = content.lower()
        if content.startswith("[StateFrame]"):
            key = "state"
        elif "boot" in lowered and "stateframe" in lowered:
            key = "boot"
        elif "source" in lowered or "current source" in lowered:
            key = "source"
        elif "receipt" in lowered or "verdict" in lowered or "green" in lowered:
            key = "truth"
        elif message.get("role") == "system":
            key = "system"
        elif message.get("role") == "user":
            key = "ticket"
        else:
            key = "history"
        blocks[key] = blocks.get(key, 0) + _tokens(message)
    blocks["tools"] = _tokens(tools)
    return blocks


def _fit_projection(projection: dict[str, object], *, source_limit: int,
                    diff_limit: int) -> dict[str, object]:
    result = copy.deepcopy(projection)
    spans = result.get("source_slices")
    if isinstance(spans, list):
        for row in spans:
            if isinstance(row, Mapping):
                original = str(row.get("content", ""))
                row["content"] = _text(original, source_limit)
                if len(str(row.get("content", ""))) < len(original):
                    row["truncated"] = True
                    row.setdefault("pull_more", {
                        "type": "read_range", "tool": "read",
                        "path": row.get("path", ""),
                        "line": int(row.get("end_line", 0) or 0) + 1,
                        "max_lines": 120, "host_verified": False,
                    })
    if "current_diff" in result:
        result["current_diff"] = _text(result["current_diff"], diff_limit)
    # Truth is a required block.  Bound only explanatory reason text, never the
    # verdict, lock, obligations, or receipts.
    truth = result.get("truth")
    if isinstance(truth, Mapping) and len(json.dumps(truth, default=str)) > TRUTH_CHAR_LIMIT:
        truth = dict(truth)
        truth["reason"] = _text(truth.get("reason", ""), 360)
        result["truth"] = truth
    return result


def _minimal_messages(messages: Sequence[Mapping[str, object]], projection: Mapping[str, object],
                      *, gear: str, first_request: bool) -> list[dict[str, object]]:
    ticket = ""
    for message in messages:
        if message.get("role") == "user" and _message_content(message).strip() and not _is_optional(_message_content(message)):
            ticket = _message_content(message)
            if not ticket.startswith("["):
                break
    if not ticket:
        ticket = str(projection.get("ticket", ""))
    ticket = _text(ticket, 3000)
    state = "[StateFrame]\n" + json.dumps(projection, sort_keys=True, separators=(",", ":"),
                                               ensure_ascii=False, default=str)
    result = [{"role": "system", "content": compact_system_prompt(gear)},
              {"role": "user", "content": ticket},
              {"role": "user", "content": state}]
    return result


def enforce_request_budget(messages: Sequence[Mapping[str, object]],
                           tools: Sequence[Mapping[str, object]], *, frame: Mapping[str, object],
                           gear: str, first_request: bool,
                           context: Mapping[str, object] | None = None) -> dict[str, object]:
    """Compose the bounded frame arm and return deterministic accounting."""
    projection = phase_projection(frame, gear, context=context)
    bounded_tools = compact_tool_surface(tools, gear, truth=projection["truth"])
    source_limit = 3600 if first_request else SOURCE_CHAR_LIMIT
    diff_limit = 2600 if first_request else DIFF_CHAR_LIMIT
    limit = (BOOT_TOKEN_LIMIT if gear == "g0" and first_request
             else FIRST_REQUEST_TOKEN_LIMIT if first_request
             else STEADY_REQUEST_TOKEN_LIMIT)
    bounded_projection = _fit_projection(projection, source_limit=source_limit, diff_limit=diff_limit)
    bounded_messages = _minimal_messages(messages, bounded_projection, gear=gear,
                                          first_request=first_request)
    estimate = _request_tokens({"messages": bounded_messages, "tools": bounded_tools})
    # Trim optional edit context in deterministic passes.  The truth block is
    # never removed and no model reasoning/output cap is introduced here.
    for source_size, diff_size in ((2200, 1400), (1200, 700), (500, 300)):
        if estimate <= limit:
            break
        bounded_projection = _fit_projection(projection, source_limit=source_size, diff_limit=diff_size)
        bounded_messages = _minimal_messages(messages, bounded_projection, gear=gear,
                                              first_request=first_request)
        estimate = _request_tokens({"messages": bounded_messages, "tools": bounded_tools})
    if estimate > limit:
        # A pathological ticket/tool contract must not allow an unbounded
        # source block to evict truth.  Keep the ticket bounded and compact the
        # system text one final time; the typed truth remains in StateFrame.
        bounded_messages[0]["content"] = "Host facts are authoritative. Use offered tools; do not claim green without PASS."
        bounded_messages[1]["content"] = _text(bounded_messages[1]["content"], 1800)
        estimate = _request_tokens({"messages": bounded_messages, "tools": bounded_tools})
    blocks = _request_blocks(bounded_messages, bounded_tools)
    return {
        "schema": SCHEMA,
        "messages": bounded_messages,
        "tools": bounded_tools,
        "projection": bounded_projection,
        "estimate": {"total_tokens": estimate, "limit": limit,
                      "first_request": first_request, "blocks": blocks,
                      "fits": estimate <= limit},
        "gear_accounting": account_projection(bounded_projection),
        "dropped": ["history", "optional_sidecars", "job_and_dispute_tools"],
    }


class _CallableJournal:
    def __init__(self, callback: Callable[[dict], object]):
        self.callback = callback

    def emit(self, kind: str, **fields: object) -> object:
        return self.callback({"kind": kind, **fields})


class FrameBudgetController:
    """Host-driven monotone phase controller backed by the installed gears."""

    def __init__(self, *, journal: object = None, context: Mapping[str, object] | None = None,
                 gears: GearController | None = None):
        self._journal = (_CallableJournal(journal) if callable(journal) and not hasattr(journal, "emit")
                         else journal)
        # Do not record an extra entry here: the runtime's existing gear entry
        # is the durable start event.  Promotions still use GearController's
        # lossless snapshot/replay path.
        self.gears = gears or GearController("g0", context=context, journal=None)
        self.gears.journal = self._journal
        self._pending: tuple[str, dict[str, object]] | None = None
        self._next_pending: tuple[str, dict[str, object]] | None = None

    @property
    def gear(self) -> str:
        return self.gears.current

    @property
    def phase(self) -> str:
        return PHASE_BY_GEAR[self.gear]

    def trigger(self, trigger: str, *, state: Mapping[str, object] | None = None,
                step: int = 0, **detail: object) -> dict[str, object] | None:
        """Apply a host trigger immediately; unknown/model prose is ignored."""
        target = _TRIGGER_TARGETS.get(str(trigger).strip().lower())
        if target is None or GEAR_ORDER[target] <= GEAR_ORDER[self.gear]:
            return None
        return self.gears.promote(target, reason=str(trigger), step=step,
                                  state=state or {}, atomic_boundary=True)

    def request(self, trigger: str, **detail: object) -> bool:
        """Queue a typed host trigger until the next frame boundary."""
        target = _TRIGGER_TARGETS.get(str(trigger).strip().lower())
        if target is None or GEAR_ORDER[target] <= GEAR_ORDER[self.gear]:
            return False
        if self._pending is None:
            self._pending = (target, {"trigger": str(trigger), **detail})
            return True
        if GEAR_ORDER[target] > GEAR_ORDER[self._pending[0]]:
            if (self._next_pending is None
                    or GEAR_ORDER[target] > GEAR_ORDER[self._next_pending[0]]):
                self._next_pending = (target, {"trigger": str(trigger), **detail})
                return True
            return False
        return False

    def commit(self, frame: Mapping[str, object], *, step: int = 0) -> dict[str, object] | None:
        if self._pending is None:
            return None
        target, detail = self._pending
        self._pending = self._next_pending
        self._next_pending = None
        return self.gears.promote(target, reason=str(detail.get("trigger", "host_trigger")),
                                  step=step, state=frame, atomic_boundary=True)

    def project(self, frame: Mapping[str, object], *, context: Mapping[str, object] | None = None) -> dict[str, object]:
        return phase_projection(frame, self.gear, context=context)

    def account(self, projection: Mapping[str, object]) -> dict[str, object]:
        return account_projection(projection)

    def snapshot(self) -> dict[str, object]:
        return {"payload_schema": SCHEMA, "gear": self.gear, "phase": self.phase,
                "pending": copy.deepcopy(self._pending[1]) if self._pending else None,
                "next_pending": copy.deepcopy(self._next_pending[1]) if self._next_pending else None}


def block_reference_table(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Measure generic next-action references to rendered blocks.

    This is deliberately an offline audit helper.  It uses only recorded
    action names/arguments and block names; it never reads hidden labels or
    task-specific fixtures.
    """
    names = ("system", "ticket", "boot", "state", "truth", "source", "receipts",
             "checks", "tools", "optional")
    counts = {name: {"exposed": 0, "referenced": 0} for name in names}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        blocks = record.get("blocks", record.get("exposed_blocks", ()))
        action = record.get("next_action", record.get("action", record.get("tool", "")))
        args = record.get("arguments", record.get("parameters", {}))
        haystack = (str(action) + " " + json.dumps(args, sort_keys=True, default=str)).lower()
        exposed = list(blocks) if isinstance(blocks, Mapping) else blocks
        if not isinstance(exposed, Sequence) or isinstance(exposed, (str, bytes, bytearray)):
            exposed = ()
        for name in exposed:
            key = str(name)
            if key not in counts:
                continue
            counts[key]["exposed"] += 1
            cues = {"system": ("system",), "ticket": ("ticket", "task", "brief"),
                    "boot": ("assumption", "plan", "file"), "state": ("state",),
                    "truth": ("test", "green", "fail", "pass", "verdict"),
                    "source": ("read", "grep", "edit", "replace"),
                    "receipts": ("receipt",), "checks": ("test", "probe"),
                    "tools": ("tool",), "optional": ("job", "dispute")}[key]
            if any(cue in haystack for cue in cues):
                counts[key]["referenced"] += 1
    return [{"block": key, **value,
             "reference_rate": (value["referenced"] / value["exposed"] if value["exposed"] else 0.0)}
            for key, value in counts.items()]


__all__ = [
    "BOOT_TOKEN_LIMIT", "FIRST_REQUEST_TOKEN_LIMIT", "MODE_ENV", "SCHEMA",
    "PHASE_BY_GEAR", "STEADY_REQUEST_TOKEN_LIMIT", "FrameBudgetController", "account_projection",
    "block_reference_table", "compact_system_prompt", "compact_tool_surface", "PROBE_DESCRIPTION",
    "enabled", "enforce_request_budget", "mode", "phase_projection", "render_projection",
    "prompt_guidance", "truth_projection",
]
