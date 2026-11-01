"""Bounded menu restriction for consecutive, evidence-free source re-reads."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path


_BUILTIN_SOURCE_READERS = frozenset({
    "read", "grep", "shell", "list_files", "where_defined", "who_calls", "references",
    "git",
})
_MAX_LOCKED_REQUESTS = 2
_MAX_LOCKED_PATHS = 64
_MAX_PATH_TEXT = 4096
_SWITCH = "STATEFRAME_READ_LOCKOUT"


def enabled() -> bool:
    value = os.environ.get(_SWITCH, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{_SWITCH} must be on or off")
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    return value == "on" and bookmarks_enabled()


def _failure(runtime: object) -> tuple[str, ...] | None:
    from kernel_reread_result_truth import _current_failure

    row = _current_failure(runtime)
    if row is None:
        return None
    return tuple(row[key] for key in (
        "receipt_id", "revision", "status", "outcome", "failure_phase",
        "runner_tail_digest", "assertion"))


def _fresh(runtime: object, paths: Mapping[str, str]) -> bool:
    from kernel_code_bookmarks import _confined_source_bytes
    from kernel_public_localization import public_path

    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is None or not paths:
        return False
    for path, revision in paths.items():
        if public_path(path) != path or not revision.startswith("sha256:"):
            return False
        data = _confined_source_bytes(Path(root), path)
        if data is None or "sha256:" + hashlib.sha256(data).hexdigest() != revision:
            return False
    return True


def _request_receipts(runtime: object, request_id: str) -> list[Mapping[str, object]]:
    receipts = getattr(getattr(runtime, "owner", None), "_receipts", ())
    return [row for row in receipts if isinstance(row, Mapping)
            and row.get("request_id") == request_id]


def _request_reads(runtime: object, rows: Sequence[Mapping[str, object]]) -> dict[str, str] | None:
    if not rows or any(row.get("kind") != "read" for row in rows):
        return None
    paths: dict[str, str] = {}
    for row in rows:
        delivery = row.get("delivery")
        path = row.get("target_path")
        if (row.get("result_class") != "OK" or not isinstance(delivery, Mapping)
                or delivery.get("unchanged_since_request") is not True
                or delivery.get("requested_bytes_delivered", 0) != 0
                or not isinstance(delivery.get("effective_range"), Mapping)
                or not isinstance(path, str) or not isinstance(delivery.get("revision"), str)):
            return None
        if path in paths and paths[path] != delivery["revision"]:
            return None
        paths[path] = delivery["revision"]
    if len(paths) > _MAX_LOCKED_PATHS or sum(map(len, paths)) > _MAX_PATH_TEXT:
        return None
    return paths if _fresh(runtime, paths) else None


def _new_evidence(rows: Sequence[Mapping[str, object]]) -> bool:
    for row in rows:
        kind, result = row.get("kind"), row.get("result_class")
        if kind == "edit" and result == "APPLIED":
            return True
        if kind in {"test", "check"} and result in {
                "PASS", "FAIL", "RED", "GREEN", "PASSED", "FAILED", "FAIL_WITH_WITNESS"}:
            return True
        delivery = row.get("delivery")
        if (kind == "read" and result == "OK" and isinstance(delivery, Mapping)
                and delivery.get("unchanged_since_request") is False
                and (type(delivery.get("requested_bytes_delivered")) is int
                     and delivery["requested_bytes_delivered"] > 0)):
            return True
    return False


def _completed_request(runtime: object, previous: object,
                       current: object) -> tuple[str | None, list[Mapping[str, object]]]:
    """Use the provider batch, not its individual action request IDs."""
    batch = getattr(runtime, "native_batch", None)
    origin = getattr(batch, "origin", None)
    size = getattr(batch, "size", 0)
    if (isinstance(origin, str) and type(size) is int and size > 1
            and getattr(batch, "index", 0) == size
            and not getattr(batch, "pending", ())
            and getattr(batch, "inflight", None) is None):
        receipts = getattr(getattr(runtime, "owner", None), "_receipts", ())
        first = next((index for index in range(len(receipts) - 1, -1, -1)
                      if isinstance(receipts[index], Mapping)
                      and receipts[index].get("request_id") == origin), None)
        if first is not None:
            rows = list(receipts[first:])
            if len(rows) == size and all(isinstance(row, Mapping) for row in rows):
                return origin, rows
        return None, []
    completed = (current if isinstance(current, str) and _request_receipts(runtime, current)
                 else previous if isinstance(previous, str)
                 and _request_receipts(runtime, previous) else None)
    return completed, _request_receipts(runtime, completed) if completed else []


def _repair_route(runtime: object, tools: Sequence[dict], paths: Mapping[str, str]) -> bool:
    """Lock only with visible, editable source and an admitted edit/check pair."""
    if (not paths or getattr(runtime, "stateframe_framefit_mode", "off") != "on"
            or getattr(runtime, "_code_bookmarks_active", False) is not True):
        return False
    # The carry contains only bookmarks actually delivered by the last
    # qualified request; the pre-allocation frame can still say not_shown.
    carry = getattr(runtime, "_code_bookmark_carry", ())
    if not isinstance(carry, list):
        return False
    from kernel_read_reuse_visibility import editable_read_paths
    editable = editable_read_paths(runtime)
    targets = {row.get("path") for row in carry if isinstance(row, Mapping)
               and row.get("read_only") is False}
    # The model can be circling over test/dependency files while the repair
    # source is already visible in a separate editable bookmark.
    if not (targets & editable):
        return False
    names = {tool.get("function", {}).get("name") for tool in tools
             if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)}
    return (bool(names & {"edit", "str_replace"})
            and bool(names & {"run_tests", "authored_check"}))


@lru_cache(maxsize=1)
def _builtin_tool_names() -> frozenset[str]:
    from kernel_agent_schemas import BUILTIN_SCHEMAS
    from kernel_completion_owner import TOOL_SPECS
    from kernel_lean import _KNOWN_TOOL_NAMES
    return frozenset(set(TOOL_SPECS) | _KNOWN_TOOL_NAMES
                     | {row["function"]["name"] for row in BUILTIN_SCHEMAS}
                     | {"file_create", "file_delete", "file_move", "read_check_receipt",
                        "authored_check", "run_tests", "done"})


def _source_access(runtime: object, tool: Mapping[str, object]) -> str | None:
    """Return source-reader, other, or unknown for a request tool."""
    function = tool.get("function")
    name = function.get("name") if isinstance(function, Mapping) else None
    if not isinstance(name, str):
        return None
    if name in _BUILTIN_SOURCE_READERS:
        return "source_reader"
    declared = getattr(runtime, "_workspace_tool_capabilities", {})
    if isinstance(declared, Mapping) and name in declared:
        value = declared[name]
        return value if isinstance(value, str) and value in {"source_reader", "other"} else None
    if name in _builtin_tool_names():
        return "other"
    # Dynamic tools must declare capability out of band. Unknown tools can
    # read workspace source, so a lockout cannot truthfully engage with them.
    return None


def _classifiable(runtime: object, tools: Sequence[dict]) -> bool:
    return all(isinstance(tool, Mapping) and _source_access(runtime, tool) is not None
               for tool in tools)


def final_source_visible(runtime: object, visible_bookmarks: Sequence[Mapping[str, object]],
                         tools: Sequence[dict]) -> bool:
    """Validate that a locked request actually carries its repair source."""
    if not enabled():
        return True
    state = getattr(runtime, "_read_lockout_state", None)
    if not isinstance(state, Mapping) or not state.get("remaining"):
        return True
    paths = state.get("paths")
    if not isinstance(paths, Mapping) or not paths:
        return False
    from kernel_read_reuse_visibility import editable_read_paths
    editable = editable_read_paths(runtime)
    repairable = {row.get("path") for row in visible_bookmarks
                  if isinstance(row, Mapping) and row.get("read_only") is False}
    names = {tool.get("function", {}).get("name") for tool in tools
             if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)}
    return (bool(repairable & editable)
            and bool(names & {"edit", "str_replace"})
            and bool(names & {"run_tests", "authored_check"}))


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_read_lockout_state", None)
    if isinstance(state, dict):
        return state
    state = {"evaluated": None, "counted": None, "streak": 0, "failure": None, "paths": {},
             "remaining": 0, "projected_after": None}
    runtime._read_lockout_state = state
    return state


def _emit(runtime: object, reason: str, state: Mapping[str, object]) -> None:
    runtime.journal.emit(
        "read_lockout_decision", producer="kernel_read_lockout",
        reason_code=reason, request_id=getattr(runtime, "current_request_id", None),
        remaining=state["remaining"], paths=sorted(state["paths"]),
        failure_key=state["failure"], dispatch=False)


def project(runtime: object, tools: Sequence[dict]) -> list[dict]:
    """Project the next request once, even if its menu is recomposed."""
    if not enabled():
        return list(tools)
    state = _state(runtime)
    batch = getattr(runtime, "native_batch", None)
    if getattr(batch, "pending", ()) or getattr(batch, "inflight", None) is not None:
        return filter_tools(runtime, tools) if state["remaining"] else list(tools)
    if getattr(runtime, "_request_deferred_source_paths", ()):
        release(runtime, "FRESH_READ_REQUIRED")
        state["streak"] = 0
        return list(tools)
    current_request = getattr(runtime, "current_request_id", None)
    previous = state["evaluated"]
    if isinstance(current_request, str) and current_request != previous:
        capped = False
        if state["remaining"] and current_request != state["projected_after"]:
            state["remaining"] -= 1
            _completed_id, completed_rows = _completed_request(runtime, previous, current_request)
            if _new_evidence(completed_rows):
                state["remaining"] = 0
                _emit(runtime, "NEW_EVIDENCE", state)
                state["streak"] = 0
                state["paths"] = {}
                state["failure"] = None
            elif not state["remaining"]:
                capped = True
                _emit(runtime, "REQUEST_CAP", state)
                state["streak"] = 0
                state["paths"] = {}
                state["failure"] = None
        current = _failure(runtime)
        # Count each dispatched request once. A post-dispatch projection can
        # already contain this request's receipts; normal sends see the prior
        # request's receipts instead.
        completed, completed_rows = _completed_request(runtime, previous, current_request)
        if isinstance(completed, str) and completed != state["counted"]:
            reads = _request_reads(runtime, completed_rows) if current is not None else None
            if reads is not None and current == state["failure"]:
                state["streak"] = min(2, state["streak"] + 1)
                state["paths"].update(reads)
            elif reads is not None:
                state["streak"] = 1
                state["failure"] = current
                state["paths"] = reads
            else:
                state["streak"] = 0
                if not state["remaining"]:
                    state["paths"] = {}
                    state["failure"] = None
            state["counted"] = completed
        state["evaluated"] = current_request
        if (len(state["paths"]) > _MAX_LOCKED_PATHS
                or sum(map(len, state["paths"])) > _MAX_PATH_TEXT):
            state["streak"] = 0
            state["paths"] = {}
        if (state["streak"] >= 1 and not state["remaining"] and not capped
                 and _repair_route(runtime, tools, state["paths"])
                 and _classifiable(runtime, tools)):
            state["remaining"] = _MAX_LOCKED_REQUESTS
            _emit(runtime, "UNCHANGED_READ_REQUEST", state)
    if (not state["remaining"] or _failure(runtime) != state["failure"]
            or not _fresh(runtime, state["paths"])
            or getattr(runtime, "_request_deferred_source_paths", ())
             or (state["remaining"] and (not _repair_route(runtime, tools, state["paths"])
                                          or not _classifiable(runtime, tools)))):
        if state["remaining"]:
            state["remaining"] = 0
            _emit(runtime, "FRESH_READ_REQUIRED", state)
        runtime._read_lockout_notice = None
        return list(tools)
    state["projected_after"] = current_request
    paths = ", ".join(sorted(state["paths"]))
    runtime._read_lockout_notice = (
        f"Read lockout: {paths} were re-read unchanged; failing assertion: "
        f"{state['failure'][-1]}; read is unavailable for this request; "
        "grep, shell, and source navigation are also unavailable.")
    runtime._read_lockout_last_notice = runtime._read_lockout_notice
    return filter_tools(runtime, tools)


def filter_tools(runtime: object, tools: Sequence[dict]) -> list[dict]:
    """Keep the restriction through later request menu authorities."""
    if not enabled() or not getattr(runtime, "_read_lockout_notice", None):
        return list(tools)
    if getattr(runtime, "_request_deferred_source_paths", ()):
        release(runtime, "FRESH_READ_REQUIRED")
        return list(tools)
    if not _classifiable(runtime, tools):
        release(runtime, "FRESH_READ_REQUIRED")
        return list(tools)
    selected = [tool for tool in tools if _source_access(runtime, tool) != "source_reader"]
    state = getattr(runtime, "_read_lockout_state", None)
    paths = state.get("paths", {}) if isinstance(state, Mapping) else {}
    if not selected or not _repair_route(runtime, tools, paths):
        release(runtime, "FRESH_READ_REQUIRED")
        return list(tools)
    return selected


def append_frame_line(runtime: object, messages: list[dict]) -> None:
    if not enabled():
        return
    notice = getattr(runtime, "_read_lockout_notice", None)
    previous = getattr(runtime, "_read_lockout_last_notice", None)
    if isinstance(previous, str) and previous != notice:
        messages[:] = [message for message in messages
                       if message.get("content") != "[Read lockout] " + previous]
    if not notice:
        return
    if any(message.get("content") == "[Read lockout] " + notice
           for message in messages):
        return
    # Keep the canonical StateFrame JSON valid; this adjacent host line is part
    # of the current frame presentation, not model-authored transcript history.
    messages.append({"role": "user", "content": "[Read lockout] " + notice,
                     "_request_budget_section": "stage_evidence"})


def release(runtime: object, reason: str) -> None:
    if not enabled():
        return
    state = getattr(runtime, "_read_lockout_state", None)
    if isinstance(state, dict) and state["remaining"]:
        state["remaining"] = 0
        state["streak"] = 0
        _emit(runtime, reason, state)
    runtime._read_lockout_notice = None


def checkpoint(runtime: object) -> dict[str, object] | None:
    if not enabled():
        return None
    state = getattr(runtime, "_read_lockout_state", None)
    if not isinstance(state, dict):
        return None
    return {"schema": "stateframe-read-lockout/2", **state}


def restore(runtime: object, snapshot: object) -> None:
    if snapshot is None or not enabled():
        return
    if (isinstance(snapshot, Mapping)
            and snapshot.get("schema") == "stateframe-read-lockout/1"):
        legacy_fields = {"schema", "evaluated", "streak", "failure", "paths",
                         "remaining", "projected_after"}
        if (set(snapshot) != legacy_fields
                or type(snapshot["streak"]) is not int or not 0 <= snapshot["streak"] <= 2
                or type(snapshot["remaining"]) is not int
                or not 0 <= snapshot["remaining"] <= _MAX_LOCKED_REQUESTS
                or any(value is not None and not isinstance(value, str)
                       for value in (snapshot["evaluated"], snapshot["projected_after"]))
                or not isinstance(snapshot["paths"], Mapping)
                or len(snapshot["paths"]) > _MAX_LOCKED_PATHS
                or sum(len(path) for path in snapshot["paths"] if isinstance(path, str))
                > _MAX_PATH_TEXT
                or any(not isinstance(path, str) or len(path) > 2048
                       or not isinstance(revision, str) or len(revision) > 128
                       for path, revision in snapshot["paths"].items())
                or (snapshot["failure"] is not None and
                    (not isinstance(snapshot["failure"], (tuple, list))
                     or len(snapshot["failure"]) != 7
                     or any(not isinstance(value, str) or len(value) > 200
                            for value in snapshot["failure"])))):
            raise ValueError("invalid read lockout replay")
        # The old state counted a completed request twice and cannot identify
        # which request supplied its streak. Resume it unlocked, preserving the
        # last menu boundary but never carrying an unproved restriction.
        snapshot = {**snapshot, "schema": "stateframe-read-lockout/2",
                    "counted": snapshot["evaluated"], "streak": 0,
                    "failure": None, "paths": {}, "remaining": 0,
                    "projected_after": None}
    if (not isinstance(snapshot, Mapping) or snapshot.get("schema") != "stateframe-read-lockout/2"
            or set(snapshot) != {"schema", "evaluated", "counted", "streak", "failure", "paths",
                                     "remaining", "projected_after"}
            or type(snapshot["streak"]) is not int or not 0 <= snapshot["streak"] <= 2
            or type(snapshot["remaining"]) is not int
            or not 0 <= snapshot["remaining"] <= _MAX_LOCKED_REQUESTS
            or any(value is not None and not isinstance(value, str)
                   for value in (snapshot["evaluated"], snapshot["counted"],
                                 snapshot["projected_after"]))
            or (snapshot["failure"] is not None and
                (not isinstance(snapshot["failure"], (tuple, list))
                 or len(snapshot["failure"]) != 7
                 or any(not isinstance(value, str) or len(value) > 200
                        for value in snapshot["failure"])))
            or not isinstance(snapshot["paths"], Mapping)
            or len(snapshot["paths"]) > _MAX_LOCKED_PATHS
            or sum(len(path) for path in snapshot["paths"] if isinstance(path, str))
            > _MAX_PATH_TEXT
            or any(not isinstance(path, str) or len(path) > 2048
                   or not isinstance(revision, str) or len(revision) > 128
                   for path, revision in snapshot["paths"].items())):
        raise ValueError("invalid read lockout replay")
    runtime._read_lockout_state = {
        "evaluated": snapshot["evaluated"], "counted": snapshot["counted"],
        "streak": snapshot["streak"],
        "failure": tuple(snapshot["failure"]) if snapshot["failure"] is not None else None,
        "paths": dict(snapshot["paths"]), "remaining": snapshot["remaining"],
        "projected_after": snapshot["projected_after"],
    }
