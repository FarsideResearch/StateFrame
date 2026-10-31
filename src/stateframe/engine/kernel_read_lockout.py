"""Bounded menu restriction for consecutive, evidence-free source re-reads."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path


_PURE_SOURCE_TOOLS = frozenset({"read"})
_MAX_LOCKED_REQUESTS = 2
_MAX_LOCKED_PATHS = 64
_MAX_PATH_TEXT = 4096


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


def _request_reads(runtime: object, request_id: str) -> dict[str, str] | None:
    receipts = getattr(getattr(runtime, "owner", None), "_receipts", ())
    rows = [row for row in receipts if isinstance(row, Mapping)
            and row.get("request_id") == request_id]
    if not rows or any(row.get("kind") != "read" for row in rows):
        return None
    paths: dict[str, str] = {}
    for row in rows:
        delivery = row.get("delivery")
        path = row.get("target_path")
        if (row.get("result_class") != "OK" or not isinstance(delivery, Mapping)
                or delivery.get("unchanged_since_request") is not True
                or not isinstance(path, str) or not isinstance(delivery.get("revision"), str)):
            return None
        paths[path] = delivery["revision"]
    if len(paths) > _MAX_LOCKED_PATHS or sum(map(len, paths)) > _MAX_PATH_TEXT:
        return None
    return paths if _fresh(runtime, paths) else None


def _new_evidence(runtime: object, request_id: str) -> bool:
    receipts = getattr(getattr(runtime, "owner", None), "_receipts", ())
    return any(isinstance(row, Mapping) and row.get("request_id") == request_id
               and (row.get("kind") not in {None, "read"}
                    or (isinstance(row.get("delivery"), Mapping)
                        and row["delivery"].get("unchanged_since_request") is not True))
               for row in receipts)


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_read_lockout_state", None)
    if isinstance(state, dict):
        return state
    state = {"evaluated": None, "streak": 0, "failure": None, "paths": {},
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
    from kernel_code_bookmarks import enabled as bookmarks_enabled

    if not bookmarks_enabled() or getattr(runtime, "_code_bookmarks_active", False) is not True:
        return list(tools)
    state = _state(runtime)
    if getattr(runtime, "_request_deferred_source_paths", ()):
        release(runtime, "FRESH_READ_REQUIRED")
        state["streak"] = 0
        return list(tools)
    previous = getattr(runtime, "current_request_id", None)
    if isinstance(previous, str) and previous != state["evaluated"]:
        capped = False
        if state["remaining"] and previous != state["projected_after"]:
            state["remaining"] -= 1
            if _new_evidence(runtime, previous):
                state["remaining"] = 0
                _emit(runtime, "NEW_EVIDENCE", state)
            elif not state["remaining"]:
                capped = True
                _emit(runtime, "REQUEST_CAP", state)
        current = _failure(runtime)
        reads = _request_reads(runtime, previous) if current is not None else None
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
        state["evaluated"] = previous
        if (len(state["paths"]) > _MAX_LOCKED_PATHS
                or sum(map(len, state["paths"])) > _MAX_PATH_TEXT):
            state["streak"] = 0
            state["paths"] = {}
        if state["streak"] == 2 and not state["remaining"] and not capped:
            state["remaining"] = _MAX_LOCKED_REQUESTS
            _emit(runtime, "TWO_UNCHANGED_READ_REQUESTS", state)
    if (not state["remaining"] or _failure(runtime) != state["failure"]
            or not _fresh(runtime, state["paths"])
            or getattr(runtime, "_request_deferred_source_paths", ())):
        if state["remaining"]:
            state["remaining"] = 0
            _emit(runtime, "FRESH_READ_REQUIRED", state)
        runtime._read_lockout_notice = None
        return list(tools)
    state["projected_after"] = previous
    paths = ", ".join(sorted(state["paths"]))
    runtime._read_lockout_notice = (
        f"Read lockout: {paths} were re-read unchanged; failing assertion: "
        f"{state['failure'][-1]}; read is unavailable for this request.")
    runtime._read_lockout_last_notice = runtime._read_lockout_notice
    return filter_tools(runtime, tools)


def filter_tools(runtime: object, tools: Sequence[dict]) -> list[dict]:
    """Keep the restriction through later request menu authorities."""
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    if not bookmarks_enabled() or not getattr(runtime, "_read_lockout_notice", None):
        return list(tools)
    if getattr(runtime, "_request_deferred_source_paths", ()):
        release(runtime, "FRESH_READ_REQUIRED")
        return list(tools)
    selected = [tool for tool in tools
                if tool.get("function", {}).get("name") not in _PURE_SOURCE_TOOLS]
    if not selected:
        release(runtime, "FRESH_READ_REQUIRED")
        return list(tools)
    return selected


def append_frame_line(runtime: object, messages: list[dict]) -> None:
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    if not bookmarks_enabled():
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
    state = getattr(runtime, "_read_lockout_state", None)
    if isinstance(state, dict) and state["remaining"]:
        state["remaining"] = 0
        state["streak"] = 0
        _emit(runtime, reason, state)
    runtime._read_lockout_notice = None


def checkpoint(runtime: object) -> dict[str, object] | None:
    state = getattr(runtime, "_read_lockout_state", None)
    if not isinstance(state, dict):
        return None
    return {"schema": "stateframe-read-lockout/1", **state}


def restore(runtime: object, snapshot: object) -> None:
    if snapshot is None:
        return
    if (not isinstance(snapshot, Mapping) or snapshot.get("schema") != "stateframe-read-lockout/1"
            or set(snapshot) != {"schema", "evaluated", "streak", "failure", "paths",
                                     "remaining", "projected_after"}
            or type(snapshot["streak"]) is not int or not 0 <= snapshot["streak"] <= 2
            or type(snapshot["remaining"]) is not int
            or not 0 <= snapshot["remaining"] <= _MAX_LOCKED_REQUESTS
            or any(value is not None and not isinstance(value, str)
                   for value in (snapshot["evaluated"], snapshot["projected_after"]))
            or (snapshot["failure"] is not None and
                (not isinstance(snapshot["failure"], (tuple, list))
                 or len(snapshot["failure"]) != 7
                 or any(not isinstance(value, str) or len(value) > 200
                        for value in snapshot["failure"])))
            or not isinstance(snapshot["paths"], Mapping)
            or len(snapshot["paths"]) > _MAX_LOCKED_PATHS
            or sum(map(len, snapshot["paths"])) > _MAX_PATH_TEXT
            or any(not isinstance(path, str) or len(path) > 2048
                   or not isinstance(revision, str) or len(revision) > 128
                   for path, revision in snapshot["paths"].items())):
        raise ValueError("invalid read lockout replay")
    runtime._read_lockout_state = {
        "evaluated": snapshot["evaluated"], "streak": snapshot["streak"],
        "failure": tuple(snapshot["failure"]) if snapshot["failure"] is not None else None,
        "paths": dict(snapshot["paths"]), "remaining": snapshot["remaining"],
        "projected_after": snapshot["projected_after"],
    }
