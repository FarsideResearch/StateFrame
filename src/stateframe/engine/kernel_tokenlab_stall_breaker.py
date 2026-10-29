"""v17 no-progress guard for repeated unchanged reads.

The guard is deliberately narrow: it observes only a delivered
``READ_UNCHANGED`` receipt for one path/revision and becomes active on the
second unchanged receipt.  A changed revision clears that path's block.
"""
from __future__ import annotations

import os
import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import re


MODE_ENV = "STATEFRAME_TOKENLAB_STALL_BREAKER"
EVENT = "tokenlab_v17_no_progress_stall"
MENU_EVENT = "tokenlab_v17_stall_menu_projection"
SCHEMA = "stateframe-v17-no-progress/1"
V20_MODE_ENV = "STATEFRAME_TOKENLAB_V20_STALL_BREAKER"
V20_EVENT = "tokenlab_v20_no_progress_stall"
V20_MENU_EVENT = "tokenlab_v20_stall_menu_projection"
V20_RESTORE_EVENT = "tokenlab_v20_stall_breaker_restored"
V20_SCHEMA = "stateframe-v20-stall-breaker/1"
V22_PROGRESS_EVENT = "tokenlab_v22_stall_progress"
V22_SCHEMA = "stateframe-v22-stall-breaker/1"

# STALLWIRE is the single v46 owner for settled-action stall recovery.  The
# older constants and functions below remain as compatibility projections for
# the v17/v20 receipts; STALLWIRE never delegates authority to those reducers.
STALLWIRE_MODE_ENV = "STATEFRAME_TOKENLAB_STALLWIRE"
STALLWIRE_K_ENV = "STATEFRAME_TOKENLAB_STALLWIRE_K"
STALLWIRE_SCHEMA = "stateframe-stallwire/1"
STALLWIRE_EVENT = "tokenlab_stallwire_detected"
STALLWIRE_MENU_EVENT = "tokenlab_stallwire_menu_projected"
STALLWIRE_PROBE_EVENT = "tokenlab_stallwire_probe_suppressed"
_STALLWIRE_TRUTHY = frozenset({"1", "on", "true", "yes"})
_STALLWIRE_EDIT_TOOLS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_STALLWIRE_READ_TOOLS = frozenset({"read", "grep", "search", "locate"})
_STALLWIRE_TEST_TOOLS = frozenset({"run_tests", "test", "pytest"})
_STALLWIRE_PROBE_TOOLS = frozenset({"probe"})
_STALLWIRE_RED = frozenset({"RED", "FAIL", "FAILED", "REJECTED", "ERROR"})
_STALLWIRE_GREEN = frozenset({"GREEN", "PASS", "PASSED", "SUCCESS", "OK"})
_STALLWIRE_REFUSED = frozenset({
    "REFUSED", "REFUSED_PREDISPATCH", "REJECTED", "NO_OP", "NOOP", "BLOCKED",
})
_STALLWIRE_NO_PROGRESS_EDIT_RESULTS = frozenset({
    "NO_OP", "NOOP", "EDIT_NOOP", "IDENTICAL", "NO_EXECUTION_RECEIPT",
    "REJECTED", "REJECTED_CONTRACT", "REFUSED", "REFUSED_PREDISPATCH",
})
# NOOPFIX is keyed by the settled effect, not by a tool name or edit identity.
# The old constant names remain as compatibility aliases for callers that still
# describe this state as an "edit" streak.
_STALLWIRE_NO_PROGRESS_NOTICE_STREAK = 2
_STALLWIRE_NO_PROGRESS_CAP = 3
_STALLWIRE_NO_PROGRESS_EFFECT_KEY = "observable-no-progress"
_STALLWIRE_NO_PROGRESS_EDIT_NOTICE_STREAK = _STALLWIRE_NO_PROGRESS_NOTICE_STREAK
_STALLWIRE_NO_PROGRESS_EDIT_CAP = _STALLWIRE_NO_PROGRESS_CAP
_STALLWIRE_NO_PROGRESS_EDIT_EFFECT_KEY = _STALLWIRE_NO_PROGRESS_EFFECT_KEY


def stallwire_enabled(runtime: object | None = None) -> bool:
    """Return whether the explicit v46 unified owner is enabled."""
    configured = getattr(runtime, "stateframe_tokenlab_stallwire_mode", None)
    if configured is not None:
        return str(configured).strip().lower() in _STALLWIRE_TRUTHY
    return stallwire_mode() == "on"


def stallwire_mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(STALLWIRE_MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{STALLWIRE_MODE_ENV} must be off or on")
    return value


def stallwire_threshold(runtime: object | None = None) -> int:
    """Read the bounded no-progress K used by the typed owner."""
    configured = getattr(runtime, "stateframe_tokenlab_stallwire_k", None)
    raw = configured if configured is not None else os.environ.get(STALLWIRE_K_ENV, "5")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{STALLWIRE_K_ENV} must be an integer from 3 to 32") from error
    if not 3 <= value <= 32:
        raise ValueError(f"{STALLWIRE_K_ENV} must be an integer from 3 to 32")
    return value


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_stall_breaker_mode", None)
    return configured == "on" if configured is not None else mode() == "on"


def v20_enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_v20_stall_breaker_mode", None)
    if configured is not None:
        return str(configured).strip().lower() == "on"
    value = os.environ.get(V20_MODE_ENV, "off") or "off"
    value = str(value).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{V20_MODE_ENV} must be off or on")
    return value == "on"


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_tokenlab_stall_breaker", None)
    if not isinstance(state, dict):
        state = {"reads": {}, "blocked": {}, "menu_revisions": set()}
        if v20_enabled(runtime):
            state.update({"facts": {}, "v20_menu_markers": set(), "v20_change_marker": None})
        try:
            setattr(runtime, "_tokenlab_stall_breaker", state)
        except (AttributeError, TypeError):
            pass
    defaults = [("reads", {}), ("blocked", {}), ("menu_revisions", set())]
    if v20_enabled(runtime):
        defaults.extend((("facts", {}), ("v20_menu_markers", set())))
    for key, default in defaults:
        if not isinstance(state.get(key), type(default)):
            state[key] = default
    return state


def _current_revision(runtime: object, path: str) -> str | None:
    for name in ("_empirical_current_revisions", "artifact_hashes"):
        values = getattr(runtime, name, None)
        if isinstance(values, Mapping) and values.get(path) not in (None, ""):
            return str(values[path])
    return None


def _open_obligation_ids(runtime: object) -> tuple[str, ...]:
    try:
        from kernel_post_green_menu import open_clause_ids
        return tuple(str(item) for item in open_clause_ids(runtime) if str(item).strip())
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return ()


def _emit(runtime: object, kind: str, *, event_schema: str = SCHEMA,
          **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(kind, producer="kernel_tokenlab_stall_breaker", event_schema=event_schema,
             dispatch=False, **fields)


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _state_change_marker(runtime: object) -> tuple[object, ...]:
    """Return only host state that can make an old read fact stale.

    Read receipts are deliberately excluded.  Revision maps cover edits and the
    result collections cover test outcomes; the explicit aliases make this work
    with both the runtime and small replay doubles.
    """
    marker: list[object] = []
    for name in ("_empirical_current_revisions", "artifact_hashes"):
        value = getattr(runtime, name, None)
        if isinstance(value, Mapping):
            marker.append((name, _fingerprint(sorted((str(key), str(item))
                                                       for key, item in value.items()))))
    for name in ("current_revision", "product_revision", "revision"):
        value = getattr(runtime, name, None)
        if value not in (None, "", "UNKNOWN"):
            marker.append((name, str(value)))
    for name in ("test_receipts", "test_results", "recent_test_feedback",
                 "last_test_result", "latest_test_result", "test_result"):
        value = getattr(runtime, name, None)
        if value not in (None, "", (), [], {}):
            marker.append((name, _fingerprint(value)))
    for name in ("edit_receipts", "edit_history", "last_edit", "latest_edit",
                 "file_changes", "changed_files", "applied_edits"):
        value = getattr(runtime, name, None)
        if value not in (None, "", (), [], {}):
            marker.append((name, _fingerprint(value)))
    return tuple(marker)


def _clear_v20_blocks(runtime: object, state: dict[str, object], *, reason: str) -> None:
    blocked = state.get("blocked")
    reads = state.get("reads")
    cleared_paths = sorted(str(path) for path in blocked) if isinstance(blocked, Mapping) else []
    if isinstance(blocked, dict):
        blocked.clear()
    if isinstance(reads, dict):
        reads.clear()
    state["stall_events"] = set()
    state["facts"] = {}
    state["v20_menu_markers"] = set()
    if cleared_paths:
        _emit(runtime, V20_RESTORE_EVENT, event_schema=V20_SCHEMA,
              reason=reason, restored_paths=cleared_paths)


def note_state_change(runtime: object, *, kind: str = "state_change") -> None:
    """Clear v20 path blocks after an explicit edit, test, or revision change."""
    if not enabled(runtime) or not v20_enabled(runtime):
        return
    state = _state(runtime)
    _clear_v20_blocks(runtime, state, reason=str(kind))
    state["v20_change_marker"] = _state_change_marker(runtime)


def note_coordinated_progress(runtime: object, change_set: str,
                              changed_paths: Sequence[object]) -> bool:
    """Count one distinct applied batch as progress, never one receipt per file.

    A duplicate parent receipt, a no-op, or an unchanged stage does not clear a
    stall.  The batch capability owns truth; this function only supplies the
    existing stall breaker with an aggregate progress observation.
    """
    if not enabled(runtime):
        return False
    state = _state(runtime)
    markers = state.setdefault("coordinated_progress", set())
    if not isinstance(markers, set):
        markers = set()
        state["coordinated_progress"] = markers
    marker = (str(change_set), tuple(sorted(str(path) for path in changed_paths)))
    if not marker[1] or marker in markers:
        return False
    markers.add(marker)
    note_state_change(runtime, kind="coordinated_batch")
    _emit(runtime, V22_PROGRESS_EVENT, event_schema=V22_SCHEMA,
          change_set=str(change_set), changed_paths=list(marker[1]),
          distinct_progress=True)
    return True


def _fact(reads: int, obligation_ids: Sequence[str]) -> str:
    open_ids = ", ".join(str(item) for item in obligation_ids if str(item).strip()) or "none"
    return (f"NO_PROGRESS reads={reads} revision unchanged; open: {open_ids}; "
            "to close: edit | probe | done")


def observe_read_unchanged(runtime: object, path: str, revision: object,
                           receipt: dict | None = None) -> dict | None:
    """Record a same-revision unchanged read and decorate the second receipt."""
    if not enabled(runtime):
        return receipt
    try:
        from kernel_tokenlab_postgreen_no_progress import owned
        if owned(runtime):
            return receipt
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    relative = str(path).replace("\\", "/").lstrip("./")
    revision_key = str(revision)
    if revision in (None, "", "UNKNOWN"):
        return receipt
    state = _state(runtime)
    reads = state["reads"]
    previous = reads.get(relative)
    if not isinstance(previous, Mapping) or str(previous.get("revision")) != revision_key:
        previous = {"revision": revision_key, "reads": 0}
    count = int(previous.get("reads", 0) or 0) + 1
    reads[relative] = {"revision": revision_key, "reads": count}
    blocked = state["blocked"]
    if blocked.get(relative) != revision_key:
        blocked.pop(relative, None)
    if count < 2:
        return receipt
    obligation_ids = _open_obligation_ids(runtime)
    fact = _fact(count, obligation_ids)
    if isinstance(receipt, dict):
        receipt["typed_fact"] = fact
        receipt["no_progress"] = {
            "reads": count,
            "revision": revision_key,
            "revision_unchanged": True,
            "open_obligation_ids": list(obligation_ids),
            "next_actions": ["edit", "probe", "done"],
        }
        receipt["message"] = str(receipt.get("message", "")) + "\n" + fact
    blocked[relative] = revision_key
    if v20_enabled(runtime):
        state.setdefault("facts", {})[relative] = fact
    event_key = (relative, revision_key)
    emitted = state.setdefault("stall_events", set())
    if event_key not in emitted:
        emitted.add(event_key)
        _emit(runtime, EVENT, path=relative, revision=revision_key, reads=count,
              open_obligation_ids=list(obligation_ids), next_actions=["edit", "probe", "done"],
              fact=fact)
        if v20_enabled(runtime):
            _emit(runtime, V20_EVENT, event_schema=V20_SCHEMA, path=relative,
                  revision=revision_key, reads=count,
                  open_obligation_ids=list(obligation_ids), next_actions=["edit", "probe", "done"],
                  typed_fact="NO_PROGRESS", fact=fact)
    return receipt


def _clip_text(value: object, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.lstrip().startswith(("{", "[")):
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _first_value(row: Mapping[str, object], *names: str) -> object:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def _row_detail(row: Mapping[str, object]) -> dict[str, object]:
    detail = _json_mapping(row.get("detail"))
    observation = _mapping(row.get("observation"))
    if observation:
        nested = _json_mapping(observation.get("observation"))
        if nested:
            detail = {**detail, **nested}
    return detail


def _row_text(row: Mapping[str, object], fallback: object = "") -> str:
    values: list[str] = []
    for source in (row, _row_detail(row)):
        for name in ("observation", "output", "message", "detail", "failure_tail"):
            value = source.get(name)
            if isinstance(value, str) and value.strip():
                values.append(value)
    nested = _mapping(row.get("observation"))
    if nested:
        value = nested.get("observation", nested.get("output", ""))
        if isinstance(value, str) and value.strip():
            values.append(value)
    if isinstance(fallback, str) and fallback.strip():
        values.append(fallback)
    return "\n".join(dict.fromkeys(values))


def _row_tool(row: Mapping[str, object], fallback: object = "") -> str:
    action = _mapping(row.get("action"))
    value = _first_value(row, "tool", "action_kind", "operation")
    if value is None:
        value = _first_value(action, "action_kind", "tool", "operation")
    text = str(value or fallback or "unknown").strip().lower()
    return text.split(None, 1)[0] if text else "unknown"


def _row_result(row: Mapping[str, object], fallback: object = "") -> str:
    action = _mapping(row.get("action"))
    value = _first_value(row, "result_class", "result_kind", "result", "status", "disposition")
    if value is None:
        value = _first_value(action, "result_kind", "result_class", "result", "status")
    text = str(value or fallback or "UNKNOWN").strip().upper().replace("-", "_")
    return text.split(None, 1)[0] if text else "UNKNOWN"


def _row_arguments(row: Mapping[str, object]) -> dict[str, object]:
    action = _mapping(row.get("action"))
    for source in (row, action):
        for name in ("action_parameters", "canonical_arguments", "arguments", "parameters"):
            value = source.get(name)
            if isinstance(value, Mapping):
                return dict(value)
    return {}


def _row_target(row: Mapping[str, object], tool: str) -> str:
    arguments = _row_arguments(row)
    for name in ("target_path", "path", "file", "target", "selector", "test_selector"):
        value = _first_value(row, name) or _first_value(arguments, name)
        if value is not None and str(value).strip():
            return str(value).replace("\\", "/").strip()
    text = _row_text(row)
    match = re.search(r"(?:path|file|target)\s*[=:]\s*([A-Za-z0-9_.@/-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    if tool in _STALLWIRE_TEST_TOOLS:
        match = re.search(r"(?:pytest|test)\s+([A-Za-z0-9_./-]+\.py(?:::[A-Za-z0-9_]+)?)", text)
        if match:
            return match.group(1)
    return ""


def _normalize_edit_value(value: object) -> object:
    """Canonicalize edit arguments without changing their semantic content."""
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_edit_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key).lower() not in {"why", "reason"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_normalize_edit_value(item) for item in value)
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return value


def _edit_identity(row: Mapping[str, object], target: str) -> str:
    """Retain the legacy green/open-work identity for non-RED presentation."""
    arguments = _row_arguments(row)
    span = {
        name: arguments.get(name)
        for name in ("line", "start_line", "end_line", "column", "start_column", "end_column",
                     "symbol", "selector")
        if arguments.get(name) not in (None, "")
    }
    content = {
        key: value for key, value in arguments.items()
        if str(key).lower() not in {
            "path", "target", "file", "file_path", "target_path", "line", "start_line",
            "end_line", "column", "start_column", "end_column", "symbol", "selector",
        }
    }
    if not content:
        content = {"arguments_hash": row.get("arguments_hash", "")}
    if not content.get("arguments_hash") and row.get("candidate_digest"):
        content = {"candidate_digest": row.get("candidate_digest")}
    return _fingerprint({"target": target, "span": _normalize_edit_value(span),
                         "content": _normalize_edit_value(content)})


def _edit_byte_change(row: Mapping[str, object]) -> bool | None:
    """Return observed byte-change state; None means the receipt omitted it."""
    changes = row.get("changes")
    if isinstance(changes, Sequence) and not isinstance(changes, (str, bytes, bytearray)):
        return any(
            isinstance(change, Mapping)
            and change.get("role", "product") == "product"
            and change.get("before") != change.get("after")
            for change in changes
        )
    for name in ("byte_changed", "file_changed", "artifact_change"):
        if name in row:
            return bool(row.get(name))
    before = _mapping(row.get("before_binding"))
    after = _mapping(row.get("after_binding"))
    before_files = before.get("product_files")
    after_files = after.get("product_files")
    if isinstance(before_files, Mapping) and isinstance(after_files, Mapping):
        return dict(before_files) != dict(after_files)
    return None


def _no_progress_edit_detail(row: Mapping[str, object], fallback: object) -> str:
    """Keep one bounded host explanation in the next frame.

    A provider response with assistant text but no structured call is not a
    schema-valid proposal.  Older receipts only carried the generic closure
    sentence, so turn that specific case into an actionable validation error
    while preserving any richer host validator detail verbatim.
    """
    tool = _row_tool(row)
    result = _row_result(row, fallback)
    action = _mapping(row.get("action"))
    sources = (
        row,
        _mapping(row.get("validation")),
        _row_detail(row),
        action,
        _mapping(action.get("validation")),
    )
    for source in sources:
        for name in (
                "validation_error", "validator_error", "schema_error", "argument_error",
                "error_detail", "errors", "current_content", "rejection_reason",
                "refusal_reason", "error", "reason"):
            value = source.get(name)
            if value not in (None, ""):
                return _clip_text(value, 360)
    text = _row_text(row, fallback)
    if (tool == "invalid_proposal" and result == "NO_EXECUTION_RECEIPT"
            and "no tool was executed" in text.lower()):
        return (
            "expected one structured function tool call with a tool name and a JSON object "
            "of arguments; the provider returned assistant text/prose, so no call was "
            "dispatchable"
        )
    return _clip_text(text, 360)


def _product_file_revision(row: Mapping[str, object], path: str) -> str:
    detail = _row_detail(row)
    candidates = [
        row.get("file_revision"), row.get("source_revision"),
        detail.get("file_revision"), detail.get("source_revision"),
    ]
    for source in (row.get("read_evidence"), detail.get("read_evidence"),
                   row.get("delivery"), detail.get("delivery")):
        source_row = _mapping(source)
        candidates.extend((source_row.get("file_revision"), source_row.get("revision"),
                           source_row.get("content_hash")))
    for source in (row.get("after_binding"), row.get("before_binding"),
                   detail.get("after_binding"), detail.get("before_binding")):
        binding = _mapping(source)
        files = _mapping(binding.get("product_files"))
        if path in files:
            candidates.append(files[path])
    for value in candidates:
        if value not in (None, "") and not isinstance(value, Mapping):
            return str(value)
    return str(_first_value(row, "workspace_revision", "product_revision") or "UNKNOWN")


def _source_identity(row: Mapping[str, object], path: str) -> str:
    detail = _row_detail(row)
    read_evidence = _mapping(row.get("read_evidence"))
    if not read_evidence:
        read_evidence = _mapping(detail.get("read_evidence"))
    delivery = _mapping(row.get("delivery"))
    if not delivery:
        delivery = _mapping(detail.get("delivery"))
    delivery_status = _first_value(
        read_evidence, "delivery_status",
    ) or _first_value(delivery, "delivery_status")
    if delivery_status not in (None, "", "DELIVERED"):
        # A cache lookup or a reuse stub is not new information unless this
        # request actually returned the requested source bytes.
        return ""
    delivered_bytes = _first_value(
        read_evidence, "delivered_bytes",
    )
    if (delivery_status == "DELIVERED" and type(delivered_bytes) is int
            and delivered_bytes <= 0):
        return ""
    content_hash = _first_value(
        read_evidence, "content_hash", "content_revision",
    ) or _first_value(delivery, "content_hash", "content_revision")
    revision = _product_file_revision(row, path)
    if content_hash not in (None, ""):
        version = ("revision", revision, "content", str(content_hash))
    else:
        if revision == "UNKNOWN":
            # Grep/search receipts often omit read_evidence.  Their observed
            # result is the only host-backed content version available.
            revision = _fingerprint(_row_text(row))
        version = ("revision", revision)
    return _fingerprint({
        "path": path, "version": version,
    })


def _read_span(row: Mapping[str, object]) -> dict[str, object]:
    detail = _row_detail(row)
    read_evidence = _mapping(row.get("read_evidence")) or _mapping(detail.get("read_evidence"))
    delivery = _mapping(row.get("delivery")) or _mapping(detail.get("delivery"))
    return {
        key: read_evidence.get(key, delivery.get(key))
        for key in ("start_line", "end_line", "line_start", "line_end", "span_id")
        if read_evidence.get(key, delivery.get(key)) not in (None, "")
    }


def _source_restoration(row: Mapping[str, object], source_identity: str) -> tuple[bool, str]:
    """Read only explicit host delivery metadata as restoration evidence."""
    detail = _row_detail(row)
    delivery = _mapping(row.get("delivery")) or _mapping(detail.get("delivery"))
    retention = str(delivery.get("source_retention", "")).strip().upper()
    restored = (
        retention in {"RESTORED_AFTER_HOST_EVICTION", "RESTORED_AFTER_EVICTION"}
        or delivery.get("restored_after_eviction") is True
        or delivery.get("host_evicted_source_restored") is True
    )
    if not restored or not source_identity:
        return False, ""
    eviction_id = _first_value(delivery, "eviction_id", "source_eviction_id", "eviction_event_id")
    return True, _fingerprint({
        "source_identity": source_identity,
        "span": _read_span(row),  # 10-08: span identifies WHICH evicted range is restored; never plain-read novelty
        "eviction_id": str(eviction_id) if eviction_id not in (None, "") else "host-eviction",
    })


def _source_delivery_key(row: Mapping[str, object], path: str) -> str:
    """Build the same strong identity used by final-wire source visibility."""
    detail = _row_detail(row)
    read_evidence = _mapping(row.get("read_evidence")) or _mapping(detail.get("read_evidence"))
    delivery = _mapping(row.get("delivery")) or _mapping(detail.get("delivery"))
    fields = {**read_evidence, **delivery, "path": path}
    from kernel_read_reuse_visibility import source_delivery_identity
    identity = source_delivery_identity(fields)
    return json.dumps(identity, sort_keys=True, separators=(",", ":")) if identity else ""


def _check_identity(row: Mapping[str, object]) -> str:
    """Semantic check identity includes current product/environment binding."""
    detail = _row_detail(row)
    binding = _mapping(row.get("after_binding")) or _mapping(detail.get("after_binding"))
    return _fingerprint({
        "check": _first_value(row, "check_identity", "check_id", "selector", "command")
        or _first_value(detail, "check_identity", "check_id", "selector", "command"),
        "revision": _first_value(row, "product_revision", "revision", "source_revision")
        or _first_value(binding, "product_revision", "revision", "artifact"),
        "environment": _first_value(row, "environment_revision", "environment")
        or _first_value(binding, "environment_revision", "environment"),
        "observation": _clip_text(_row_text(row), 360),
    })


def _failure_detail(text: str) -> str:
    rows = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    lower_rows = [(line, line.lower()) for line in rows]
    assertion = [line for line, lower in lower_rows if "assertionerror" in lower or "assert " in lower]
    mismatch = [
        line for line, lower in lower_rows
        if any(marker in lower for marker in
               ("differing items", "expected", " != ", "_cents", "error"))
    ]
    failed = [line for line, lower in lower_rows if "failed" in lower]
    selected: list[str] = []
    for line in (assertion[-1:] + mismatch[-4:] + failed[-1:]):
        if line not in selected:
            selected.append(line)
    return _clip_text("; ".join(selected or rows[-3:]), 420)


def _source_symbol(source: str, path: str) -> str:
    match = re.search(r"(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", source)
    if match:
        return match.group(1)
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem or "<unknown>"


def _refusal_reason(text: str) -> str:
    match = re.search(r"(?:reason|refusal_code|code)\s*[\"':= ]+([A-Za-z][A-Za-z0-9_-]+)", text,
                      re.IGNORECASE)
    return match.group(1) if match else _clip_text(text, 120)


@dataclass(frozen=True)
class StallWireSignal:
    """A host-receipt-backed, one-shot recovery signal."""

    trigger: str
    step: int
    no_progress_decisions: int
    cycle: tuple[str, ...]
    failure_excerpt: str
    implicated_path: str
    implicated_symbol: str
    next_actions: tuple[str, ...] = ("edit", "str_replace")

    @property
    def repair_line(self) -> str:
        failure = self.failure_excerpt or "the latest public check remains RED"
        target = f"{self.implicated_path}::{self.implicated_symbol}" \
            if self.implicated_path else "the implicated source symbol"
        return (
            f"[STALLWIRE] last failing assertion/output: {failure}; "
            f"implicates {target}; edit is the next legal action. "
            "Diagnose the root cause, then use edit or str_replace now."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "trigger": self.trigger, "step": self.step,
            "no_progress_decisions": self.no_progress_decisions,
            "cycle": list(self.cycle), "failure_excerpt": self.failure_excerpt,
            "implicated_path": self.implicated_path,
            "implicated_symbol": self.implicated_symbol,
            "next_actions": list(self.next_actions), "repair_line": self.repair_line,
        }


@dataclass(frozen=True)
class _ClosedAction:
    step: int
    tool: str
    result: str
    target: str
    receipt_id: str
    observation: str
    source_identity: str = ""
    source_delivery_key: str = ""
    edit_identity: str = ""
    check_identity: str = ""
    revision_identity: str = ""
    source_restoration: bool = False
    restoration_identity: str = ""
    byte_changed: bool | None = None
    no_progress_edit: bool = False
    no_progress_detail: str = ""

    @property
    def product_edit(self) -> bool:
        row_result = self.result == "APPLIED"
        return row_result and self.tool in _STALLWIRE_EDIT_TOOLS and self.byte_changed is not False

    @property
    def fingerprint(self) -> str:
        return _fingerprint({
            "tool": self.tool, "result": self.result, "target": self.target,
            "observation": _failure_detail(self.observation) if self.tool in _STALLWIRE_TEST_TOOLS
            else _clip_text(self.observation, 180),
        })


def _closed_action(row: Mapping[str, object], *, action: object = "",
                   observation: object = "") -> _ClosedAction:
    tool = _row_tool(row, action)
    result = _row_result(row, observation)
    target = _row_target(row, tool)
    byte_changed = _edit_byte_change(row) if tool in _STALLWIRE_EDIT_TOOLS else None
    edit_identity = _edit_identity(row, target) if tool in _STALLWIRE_EDIT_TOOLS else ""
    no_progress_edit = bool(
        tool in _STALLWIRE_EDIT_TOOLS
        and (result in _STALLWIRE_NO_PROGRESS_EDIT_RESULTS or byte_changed is False)
    )
    source_identity = (
        _source_identity(row, target) if tool in _STALLWIRE_READ_TOOLS and target else ""
    )
    revision_identity = _fingerprint({
        key: row.get(key) for key in (
            "product_revision", "revision", "after_revision", "source_revision",
            "dependency_identity", "dependency_revision", "environment_revision",
        ) if row.get(key) not in (None, "")
    })
    if source_identity and revision_identity:
        source_identity = _fingerprint({
            "source": source_identity, "revision": revision_identity,
        })
    source_restoration, restoration_identity = _source_restoration(row, source_identity)
    source_delivery_key = (
        _source_delivery_key(row, target) if tool in _STALLWIRE_READ_TOOLS else ""
    )
    step_value = _first_value(row, "step", "turn", "sequence")
    try:
        step = int(step_value)
    except (TypeError, ValueError):
        step = 0
    receipt_id = str(_first_value(row, "receipt_id", "step_record_id", "event_id") or
                     _fingerprint({"step": step, "tool": tool, "observation": observation}))
    return _ClosedAction(
        step=step, tool=tool, result=result, target=target, receipt_id=receipt_id,
        observation=_row_text(row, observation), source_identity=source_identity,
        source_delivery_key=source_delivery_key,
        edit_identity=edit_identity,
        check_identity=_check_identity(row) if tool in _STALLWIRE_TEST_TOOLS else "",
        revision_identity=revision_identity,
        source_restoration=source_restoration,
        restoration_identity=restoration_identity,
        byte_changed=byte_changed,
        no_progress_edit=no_progress_edit,
        no_progress_detail=_no_progress_edit_detail(row, observation),
    )


def _host_failure(row: Mapping[str, object]) -> bool:
    """Host admission/transport/bookkeeping failures are not model attempts."""
    value = row.get("failure_class", row.get("outcome_class", row.get("host_outcome", "")))
    normalized = str(value or "").strip().casefold().replace("-", "_")
    if normalized in {
        "host_admission", "transport", "bookkeeping", "unsupported_capability",
        "environment", "host_contract_limit", "host_stop",
    }:
        return True
    return row.get("model_opportunity") is False


def _cycle_signature(history: Sequence[_ClosedAction]) -> tuple[str, ...]:
    signatures = tuple(f"{item.tool}({item.result})" for item in history[-3:])
    return signatures


class StallWireOwner:
    """One receipt reducer for v43H interrogation + rd3 recovery semantics."""

    def __init__(self, *, threshold: int = 5) -> None:
        if threshold < 3:
            raise ValueError("STALLWIRE K must be at least 3")
        self.threshold = threshold
        self.history: list[_ClosedAction] = []
        self.seen_receipts: set[str] = set()
        self.seen_information: set[str] = set()
        self.seen_tests: set[str] = set()
        self.seen_source_restorations: set[str] = set()
        self.pending_source_deliveries: dict[str, dict[str, _ClosedAction]] = {}
        self.episode_information: set[str] = set()
        self.episode_tests: set[str] = set()
        self.episode_source_restorations: set[str] = set()
        self.source_symbols: dict[str, str] = {}
        self.source_order: list[str] = []
        self.red_active = False
        self.no_progress_decisions = 0
        self.host_failure_count = 0
        self.last_failure = ""
        self.probe_refused = False
        self.probe_refusal_reason = ""
        self.signal: StallWireSignal | None = None
        self.notice_emitted = False
        # This is the one bounded streak for settled actions that produce
        # neither product change nor new host-backed information.
        self.no_progress_edit_key: str | None = None
        self.no_progress_edit_streak = 0
        self.no_progress_edit_target = ""
        self.no_progress_edit_detail = ""
        self.no_progress_tool = ""
        self.no_progress_result = ""
        self.no_progress_edit_capped = False

    def _clear_no_progress_edit(self) -> None:
        self.no_progress_edit_key = None
        self.no_progress_edit_streak = 0
        self.no_progress_edit_target = ""
        self.no_progress_edit_detail = ""
        self.no_progress_tool = ""
        self.no_progress_result = ""
        self.no_progress_edit_capped = False

    def _queue_source_restoration(self, action: _ClosedAction) -> bool:
        if (not action.source_restoration
                or action.restoration_identity in self.seen_source_restorations):
            return False
        if action.result != "OK" or not action.source_delivery_key:
            return False
        pending = self.pending_source_deliveries
        restorations = pending.get(action.source_delivery_key)
        if restorations is None:
            if len(pending) >= 64:
                return False
            restorations = {}
            pending[action.source_delivery_key] = restorations
        if (action.restoration_identity not in restorations and len(restorations) >= 32):
            return False
        restorations.setdefault(action.restoration_identity, action)
        return True

    def _observe_no_progress_edit(self, action: _ClosedAction, *,
                                  information_progress: bool,
                                  defer_no_progress: bool = False) -> None:
        if information_progress:
            self._clear_no_progress_edit()
            return
        if defer_no_progress:
            return
        # Every settled action without a product change or a new host-backed
        # fact is one unit of the same no-progress effect.  This deliberately
        # includes invalid proposals, refused calls, no-op edits, unchanged
        # reads/greps, and unchanged check receipts.
        self.no_progress_edit_key = _STALLWIRE_NO_PROGRESS_EDIT_EFFECT_KEY
        self.no_progress_edit_streak = min(
            self.no_progress_edit_streak + 1,
            _STALLWIRE_NO_PROGRESS_EDIT_CAP,
        )
        self.no_progress_edit_target = action.target
        self.no_progress_edit_detail = action.no_progress_detail
        self.no_progress_tool = action.tool
        self.no_progress_result = action.result
        if self.no_progress_edit_streak >= _STALLWIRE_NO_PROGRESS_EDIT_CAP:
            self.no_progress_edit_capped = True

    def _observe_no_progress_if_enabled(
            self, action: _ClosedAction, *, information_progress: bool,
            defer_no_progress: bool, enabled_for_runtime: bool,
    ) -> None:
        if enabled_for_runtime:
            self._observe_no_progress_edit(
                action, information_progress=information_progress,
                defer_no_progress=defer_no_progress,
            )

    def _record_delivered_restoration(self, action: _ClosedAction) -> bool:
        information_progress = False
        if action.source_identity and action.source_identity not in self.seen_information:
            self.seen_information.add(action.source_identity)
            self.episode_information.add(action.source_identity)
            self.source_order.append(action.target)
            self.source_symbols.setdefault(
                action.target, _source_symbol(action.observation, action.target),
            )
            information_progress = True
        restoration = action.restoration_identity or action.source_identity
        if restoration and restoration not in self.seen_source_restorations:
            self.seen_source_restorations.add(restoration)
            self.episode_source_restorations.add(restoration)
            information_progress = True
        return information_progress

    def note_provider_source_delivery(self, sources: Sequence[object]) -> int:
        """Count pending restorations only after their exact bytes reach a request."""
        from kernel_read_reuse_visibility import source_delivery_identity

        confirmed = 0
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            identity = source_delivery_identity(source)
            if identity is None:
                continue
            key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
            pending = self.pending_source_deliveries.pop(key, None)
            if not pending:
                continue
            confirmed += 1
            information_progress = False
            for action in pending.values():
                information_progress = (
                    self._record_delivered_restoration(action) or information_progress
                )
            if information_progress:
                self.no_progress_decisions = 0
                self._clear_no_progress_edit()
        return confirmed

    def _reset_after_product_progress(self) -> None:
        self.history.clear()
        self.pending_source_deliveries.clear()
        self.episode_information.clear()
        self.episode_tests.clear()
        self.episode_source_restorations.clear()
        self.red_active = False
        self.no_progress_decisions = 0
        self.last_failure = ""
        self.probe_refused = False
        self.probe_refusal_reason = ""
        self.signal = None
        self.notice_emitted = False
        self._probe_event_emitted = False
        self._clear_no_progress_edit()

    def _implicated_source(self) -> tuple[str, str]:
        for path in reversed(self.source_order):
            if not path.startswith("tests/"):
                return path, self.source_symbols.get(path, "<unknown>")
        return "", "<unknown>"

    def _information_identity(self, action: _ClosedAction) -> tuple[str, str] | None:
        if (action.tool in {"requirement_table", "question_gate"}
                and action.result in {"ADMITTED", "ACCEPTED", "CERTIFIED"}):
            # Stage admission is semantic progress only when its settled
            # record is new.  Receipt IDs are transport metadata, so duplicate
            # tables and repeated renders retain the same identity.
            return ("stage", _fingerprint({
                "tool": action.tool,
                "result": action.result,
                "target": action.target,
                "observation": _clip_text(action.observation, 720),
            }))
        if action.tool in _STALLWIRE_READ_TOOLS:
            if action.source_restoration:
                return None
            return ("read", action.source_identity) if action.source_identity else None
        if (action.tool in _STALLWIRE_PROBE_TOOLS
                and action.result in _STALLWIRE_GREEN.union({"COMPLETED"})):
            return ("check", _fingerprint({
                "result": action.result,
                # A new probe receipt is transport metadata, not a new
                # observation.  Only semantic scope/output may clear the
                # settled no-progress identity.
                "target": action.target,
                "observation": _clip_text(action.observation, 360),
                "check_identity": action.check_identity,
                "revision": action.revision_identity,
            }))
        if action.tool in _STALLWIRE_TEST_TOOLS:
            return ("check", _fingerprint({
                "result": action.result,
                "failure": _failure_detail(action.observation),
                "check_identity": action.check_identity,
                "revision": action.revision_identity,
            }))
        return None

    def _is_information_progress(self, action: _ClosedAction, *, episode: bool = False) -> bool:
        identity = self._information_identity(action)
        if identity is None:
            return False
        kind, value = identity
        seen = (
            self.episode_information if kind in {"read", "stage"} and episode else
            self.episode_tests if kind == "check" and episode else
            self.seen_information if kind in {"read", "stage"} else self.seen_tests
        )
        restorations = (self.episode_source_restorations if episode
                        else self.seen_source_restorations)
        if value in seen:
            if kind == "read" and action.source_restoration:
                restoration = action.restoration_identity or value
                if restoration not in restorations:
                    restorations.add(restoration)
                    return True
            return False
        seen.add(value)
        if kind == "read" and action.source_restoration:
            restorations.add(action.restoration_identity or value)
        if kind == "read" and not episode:
            self.source_order.append(action.target)
            self.source_symbols.setdefault(action.target, _source_symbol(action.observation, action.target))
        return True

    def observe(self, row: Mapping[str, object], *, action: object = "",
                observation: object = "",
                count_no_progress: bool = True) -> StallWireSignal | None:
        """Consume exactly one settled host receipt; model prose is not progress."""
        closed = _closed_action(row, action=action, observation=observation)
        if closed.receipt_id in self.seen_receipts:
            return None
        self.seen_receipts.add(closed.receipt_id)
        self.history.append(closed)
        self.history = self.history[-16:]

        # Host admission/transport/bookkeeping outcomes are settled once but
        # never become model no-progress or a product-red observation.
        if _host_failure(row):
            self.host_failure_count += 1
            return None

        if closed.product_edit:
            self._reset_after_product_progress()
            return None
        green_check = closed.tool in _STALLWIRE_TEST_TOOLS and closed.result in _STALLWIRE_GREEN
        if green_check:
            if self.red_active:
                self.episode_information.clear()
                self.episode_tests.clear()
                self.episode_source_restorations.clear()
            self.pending_source_deliveries.clear()
            self.red_active = False
        if closed.result in _STALLWIRE_RED:
            if not self.red_active:
                self.episode_information.clear()
                self.episode_tests.clear()
                self.episode_source_restorations.clear()
                self._clear_no_progress_edit()
            self.red_active = True
            self.last_failure = _failure_detail(closed.observation)

        restoration_deferred = self._queue_source_restoration(closed)

        if closed.tool in _STALLWIRE_PROBE_TOOLS and closed.result in _STALLWIRE_REFUSED:
            self.probe_refused = True
            self.probe_refusal_reason = _refusal_reason(closed.observation)

        legacy_information_progress = self._is_information_progress(closed)
        episode_information_progress = self._is_information_progress(closed, episode=True)
        self._observe_no_progress_if_enabled(
            closed, information_progress=episode_information_progress,
            defer_no_progress=restoration_deferred,
            enabled_for_runtime=count_no_progress,
        )
        if green_check:
            self.no_progress_decisions = 0
            self.signal = None
            self.notice_emitted = False
            self._probe_event_emitted = False
            if episode_information_progress:
                self.history.clear()
                self.episode_information.clear()
                self.episode_tests.clear()
                self.episode_source_restorations.clear()
            return None
        if restoration_deferred:
            return None
        if not self.red_active:
            self.no_progress_decisions = 0
            return None
        self.no_progress_decisions = (
            0 if legacy_information_progress else self.no_progress_decisions + 1
        )
        if self.signal is not None or self.no_progress_decisions < self.threshold:
            return None

        path, symbol = self._implicated_source()
        trigger = "NO_PROGRESS_K"
        signal = StallWireSignal(
            trigger=trigger, step=closed.step,
            no_progress_decisions=self.no_progress_decisions,
            cycle=_cycle_signature(self.history), failure_excerpt=self.last_failure,
            implicated_path=path, implicated_symbol=symbol,
        )
        self.signal = signal
        return signal

    @property
    def no_progress_edit_notice(self) -> str:
        """Return one concrete host-backed line for the settled no-progress streak."""
        target = self.no_progress_edit_target or "the repeated edit target"
        detail = self.no_progress_edit_detail or "the host reported no byte change"
        if self.no_progress_tool == "invalid_proposal":
            return (
                "[NOOPFIX] invalid_proposal: host validation error: "
                f"{detail}; no structured proposal was executed (streak="
                f"{self.no_progress_edit_streak})"
            )
        action = self.no_progress_tool or "settled action"
        if action in _STALLWIRE_EDIT_TOOLS:
            description = "repeated settled edit made no progress"
        else:
            description = f"repeated settled {action} made no product/information progress"
        return (
            f"[NOOPFIX] {target}: {description}; host detail: {detail} "
            f"(streak={self.no_progress_edit_streak})"
        )


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""


def _fallback_tool(name: str) -> dict[str, object]:
    properties: dict[str, object] = {"path": {"type": "string"}}
    if name == "str_replace":
        properties.update({"old_str": {"type": "string"}, "new_str": {"type": "string"}})
    if name == "edit":
        properties["function_source"] = {"type": "string"}
    return {"type": "function", "function": {
        "name": name, "parameters": {"type": "object", "properties": properties},
    }}


def _stallwire_owner(runtime: object) -> StallWireOwner:
    owner = getattr(runtime, "_stallwire_owner", None)
    if isinstance(owner, StallWireOwner):
        return owner
    owner = StallWireOwner(threshold=stallwire_threshold(runtime))
    try:
        setattr(runtime, "_stallwire_owner", owner)
    except (AttributeError, TypeError):
        pass
    return owner


def stallwire_signal(runtime: object) -> StallWireSignal | None:
    """Expose the settled current-red signal to the single live frame owner."""
    owner = getattr(runtime, "_stallwire_owner", None)
    return owner.signal if isinstance(owner, StallWireOwner) else None


def stallwire_probe_refused(runtime: object) -> bool:
    """Expose probe refusal without letting STALLWIRE render a second menu."""
    owner = getattr(runtime, "_stallwire_owner", None)
    return bool(isinstance(owner, StallWireOwner) and owner.probe_refused)


def stallwire_no_progress_edit_state(runtime: object) -> dict[str, object] | None:
    """Expose the current settled no-product/no-information streak."""
    owner = getattr(runtime, "_stallwire_owner", None)
    if not isinstance(owner, StallWireOwner) or owner.no_progress_edit_streak < 1:
        return None
    return {
        "streak": owner.no_progress_edit_streak,
        "notice_streak": _STALLWIRE_NO_PROGRESS_EDIT_NOTICE_STREAK,
        "target": owner.no_progress_edit_target,
        "detail": owner.no_progress_edit_detail,
        "tool": owner.no_progress_tool,
        "result": owner.no_progress_result,
        "notice": owner.no_progress_edit_notice,
        "capped": owner.no_progress_edit_capped,
        "pending_delivery": bool(owner.pending_source_deliveries),
    }


def note_provider_source_delivery(
        runtime: object, sources: Sequence[object], *, request_id: object = None,
) -> int:
    """Confirm eligible restorations against exact source blocks on a submitted wire."""
    owner = getattr(runtime, "_stallwire_owner", None)
    if not isinstance(owner, StallWireOwner):
        return 0
    confirmed = owner.note_provider_source_delivery(sources)
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if confirmed and callable(emit):
        emit(
            "tokenlab_stallwire_source_delivery_confirmed",
            producer="kernel_tokenlab_stall_breaker",
            request_id=request_id,
            delivered_count=confirmed,
            pending_delivery_count=len(owner.pending_source_deliveries),
            dispatch=False,
        )
    return confirmed


def stallwire_no_progress_edit_capped(runtime: object) -> bool:
    state = stallwire_no_progress_edit_state(runtime)
    return bool(state and state.get("capped"))


def observe_closed_action(runtime: object, action: object, observation: object,
                          *, receipt: Mapping[str, object] | None = None
                          ) -> StallWireSignal | None:
    """Join the runtime's settled receipt to the one STALLWIRE reducer."""
    # CONVERGE is a separate opt-in observer.  It must see the same settled
    # receipt even when the older STALLWIRE switch is off, and its state is
    # advisory: a reducer failure cannot block the action stream.
    try:
        from kernel_convergence_observer import observe_closed_action as observe_convergence
        observe_convergence(runtime, action, observation, receipt=receipt)
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    if not stallwire_enabled(runtime):
        return None
    row = dict(receipt) if isinstance(receipt, Mapping) else {}
    row.setdefault("action", action)
    row.setdefault("observation", observation)
    host_failure = row.get("host_failure") or row.get("failure_class") in {
        "HOST_CAPABILITY", "HOST_ERROR", "UNRUNNABLE", "TRANSPORT_ERROR",
    }
    if host_failure:
        from kernel_aha10_honest_loop import HostFailureRecovery
        recovery = getattr(runtime, "_aha10_host_failure_recovery", None)
        if not isinstance(recovery, HostFailureRecovery):
            recovery = HostFailureRecovery()
            setattr(runtime, "_aha10_host_failure_recovery", recovery)
        route = recovery.observe(
            request_id=row.get("request_id", getattr(runtime, "request_id", "")),
            stage=row.get("stage", getattr(runtime, "stage", "FREE")),
            revision=row.get("revision", row.get("source_revision", getattr(runtime, "current_revision", "unknown"))),
            error=row.get("host_failure", row.get("observation", "host capability failure")),
        )
        row["recovery_route"] = route.as_dict()
        setattr(runtime, "_aha10_recovery_route", route.as_dict())
        if route.replayed:
            emit = getattr(getattr(runtime, "journal", None), "emit", None)
            if callable(emit):
                emit("host_failure_replayed", producer="kernel_tokenlab_stall_breaker",
                     route=route.as_dict(), dispatch=False)
            return None
    owner = _stallwire_owner(runtime)
    from kernel_capability_floor import enabled as capability_floor_enabled
    count_no_progress = not capability_floor_enabled(runtime)
    signal = owner.observe(
        row, action=action, observation=observation,
        count_no_progress=count_no_progress,
    )
    try:
        from kernel_completion_owner import note_stall_open_work_attempt
        note_stall_open_work_attempt(runtime, action, receipt=row)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if owner.probe_refused and callable(emit) and not getattr(owner, "_probe_event_emitted", False):
        owner._probe_event_emitted = True
        emit(STALLWIRE_PROBE_EVENT, producer="kernel_tokenlab_stall_breaker",
             event_schema=STALLWIRE_SCHEMA, reason=owner.probe_refusal_reason,
             next_actions=["edit", "str_replace"], dispatch=False)
    if signal is None:
        return None
    if callable(emit):
        from kernel_continuation_journal import journal_safe_fields  # emit boundary (reserved keys)
        emit(STALLWIRE_EVENT, producer="kernel_tokenlab_stall_breaker",
             event_schema=STALLWIRE_SCHEMA, dispatch=False, **journal_safe_fields(signal.as_dict()))
    if not owner.notice_emitted:
        runtime._v46_mechanism_notice = signal.repair_line
        owner.notice_emitted = True
    return signal


def _project_stallwire_menu(runtime: object, tools: Sequence[dict]) -> list[dict]:
    try:
        from kernel_completion_owner import _no_host_checks
        no_host_checks = _no_host_checks(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        no_host_checks = False
    owner = getattr(runtime, "_stallwire_owner", None)
    if not isinstance(owner, StallWireOwner):
        return [tool for tool in tools
                if not (no_host_checks and _tool_name(tool) == "recall")]
    result = [tool for tool in tools if not (owner.probe_refused and _tool_name(tool) in _STALLWIRE_PROBE_TOOLS)]
    # The no-host terminal handoff has a frozen public schema.  ``recall`` is
    # admitted on normal and host-backed frames, but cannot be added to this
    # compatibility-only projection before a host-backed frame exists.
    if no_host_checks:
        result = [tool for tool in result if _tool_name(tool) != "recall"]
    if owner.signal is None:
        return result
    available = {_tool_name(tool): tool for tool in result if _tool_name(tool)}
    candidates: list[dict] = []
    for source in (
        getattr(runtime, "_completion_owner_full_tools", ()),
        getattr(runtime, "available_tools", ()),
        tools,
    ):
        if isinstance(source, Mapping):
            source = tuple(source.values())
        if not isinstance(source, Sequence) or isinstance(source, (str, bytes, bytearray)):
            continue
        candidates.extend(item for item in source if isinstance(item, Mapping))
    for name in ("edit", "str_replace"):
        candidate = next((dict(item) for item in candidates if _tool_name(item) == name), None)
        if candidate is None:
            candidate = _fallback_tool(name)
        available[name] = candidate
    projected = list(available.values())
    names = [_tool_name(tool) for tool in projected]
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit) and not getattr(owner, "_menu_event_emitted", False):
        owner._menu_event_emitted = True
        emit(STALLWIRE_MENU_EVENT, producer="kernel_tokenlab_stall_breaker",
             event_schema=STALLWIRE_SCHEMA, offered=names,
             next_actions=["edit", "str_replace"], dispatch=False)
    return projected


def _apply_menu_legacy(runtime: object, tools: Sequence[dict]) -> list[dict]:
    """Remove the read capability once a path has hit the no-progress threshold."""
    if not enabled(runtime):
        return list(tools)
    try:
        from kernel_tokenlab_postgreen_no_progress import owned
        if owned(runtime):
            return list(tools)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    state = _state(runtime)
    blocked = dict(state["blocked"])
    for path, revision in list(blocked.items()):
        current = _current_revision(runtime, path)
        if current is not None and current != revision:
            blocked.pop(path, None)
    state["blocked"] = blocked
    if not blocked:
        return list(tools)
    result: list[dict] = []
    removed: list[str] = []
    constrained = False
    for tool in tools:
        if _tool_name(tool) != "read":
            result.append(tool)
            continue
        function = tool.get("function") if isinstance(tool, Mapping) else None
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        path_schema = properties.get("path") if isinstance(properties, Mapping) else None
        enum = path_schema.get("enum") if isinstance(path_schema, Mapping) else None
        if isinstance(enum, list):
            remaining = [item for item in enum if str(item) not in blocked]
            constrained = constrained or len(remaining) != len(enum)
            if remaining:
                import copy
                constrained_tool = copy.deepcopy(dict(tool))
                constrained_tool["function"]["parameters"]["properties"]["path"]["enum"] = remaining
                result.append(constrained_tool)
                continue
        removed.append("read")
    if removed or constrained:
        revision_key = tuple(sorted((str(path), str(revision)) for path, revision in blocked.items()))
        menu_revisions = state["menu_revisions"]
        if revision_key not in menu_revisions:
            menu_revisions.add(revision_key)
            _emit(runtime, MENU_EVENT, blocked_paths=sorted(blocked),
                  removed_tools=sorted(set(removed)),
                  constrained_tools=["read"] if constrained else [])
    return result


def _apply_menu_v20(runtime: object, tools: Sequence[dict]) -> list[dict]:
    state = _state(runtime)
    marker = _state_change_marker(runtime)
    previous_marker = state.get("v20_change_marker")
    if previous_marker is not None and marker != previous_marker:
        _clear_v20_blocks(runtime, state, reason="edit_or_test_or_revision")
    state["v20_change_marker"] = marker

    blocked = dict(state["blocked"])
    for path, revision in list(blocked.items()):
        current = _current_revision(runtime, path)
        if current is not None and current != revision:
            blocked.pop(path, None)
            state.get("facts", {}).pop(path, None)
    state["blocked"] = blocked
    if not blocked:
        return list(tools)

    result: list[dict] = []
    removed_paths: list[str] = []
    constrained_paths: list[str] = []
    retained_generic = False
    for tool in tools:
        if _tool_name(tool) != "read":
            result.append(tool)
            continue
        function = tool.get("function") if isinstance(tool, Mapping) else None
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        path_schema = properties.get("path") if isinstance(properties, Mapping) else None
        enum = path_schema.get("enum") if isinstance(path_schema, Mapping) else None
        if not isinstance(enum, list) or not enum:
            # A generic read has no safe way to express one blocked path.  Keep
            # it callable so unrelated and not-yet-read files remain reachable.
            result.append(tool)
            retained_generic = True
            continue
        remaining = [item for item in enum if str(item) not in blocked]
        if len(remaining) == len(enum):
            result.append(tool)
            continue
        if remaining:
            constrained = deepcopy(dict(tool))
            constrained["function"]["parameters"]["properties"]["path"]["enum"] = remaining
            result.append(constrained)
            constrained_paths.extend(str(item) for item in enum if str(item) in blocked)
            continue
        # Every addressable path in this path-enumerated read is blocked.  It is
        # safe to remove this particular read tool, but not a generic read.
        removed_paths.extend(str(item) for item in enum)

    changed = bool(removed_paths or constrained_paths or retained_generic)
    if changed:
        marker = ("v20", tuple(sorted((str(path), str(revision))
                                       for path, revision in blocked.items())),
                  tuple(sorted(removed_paths)), tuple(sorted(constrained_paths)), retained_generic)
        markers = state["v20_menu_markers"]
        if marker not in markers:
            markers.add(marker)
            facts = state.get("facts", {})
            if isinstance(facts, dict):
                reads = state.get("reads", {})
                for path, revision in blocked.items():
                    if path not in facts:
                        prior = reads.get(path, {}) if isinstance(reads, Mapping) else {}
                        facts[path] = _fact(int(prior.get("reads", 2) or 2),
                                            _open_obligation_ids(runtime))
            _emit(runtime, V20_MENU_EVENT, event_schema=V20_SCHEMA,
                  blocked_paths=sorted(blocked), removed_paths=sorted(set(removed_paths)),
                  constrained_paths=sorted(set(constrained_paths)),
                  removed_tools=["read"] if removed_paths else [],
                  retained_generic_read=retained_generic,
                  global_tool_removed=bool(removed_paths and not retained_generic),
                  no_progress_facts=[facts[path] for path in sorted(blocked)
                                    if isinstance(facts, Mapping) and path in facts],
                  typed_fact="NO_PROGRESS")
    return result


def _apply_capability_floor(runtime: object, selected: Sequence[dict],
                            current_tools: object) -> list[dict]:
    from kernel_capability_floor import apply_floor, enabled as capability_floor_enabled

    if not capability_floor_enabled(runtime):
        return selected if isinstance(selected, list) else list(selected)
    sources = (
        current_tools,
        getattr(runtime, "_readiness_full_tools", ()),
        getattr(runtime, "_completion_owner_full_tools", ()),
        getattr(runtime, "available_tools", ()),
    )
    by_name: dict[str, Mapping[str, object]] = {}
    for source in sources:
        if isinstance(source, Mapping):
            items = source.values()
        elif isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
            items = source
        else:
            continue
        for item in items:
            if isinstance(item, Mapping):
                name = _tool_name(item)
                if name:
                    by_name.setdefault(name, item)
    try:
        from kernel_completion_owner import capability_floor_action_admissible
    except ImportError:
        # Import uncertainty must not turn a recovery-menu helper into an
        # admission bypass; the capability floor fails closed here.
        def admissible(_name: str) -> bool:
            return False
    else:
        def admissible(name: str) -> bool:
            return capability_floor_action_admissible(runtime, name)
    result = apply_floor(
        selected, list(by_name.values()), admissible=admissible,
        runtime=runtime,
    )
    names = tuple(_tool_name(tool) for tool in result)
    setattr(runtime, "completion_menu", names)
    setattr(runtime, "offered_tools", names)
    setattr(runtime, "_question_done_available", "done" in names)
    setattr(runtime, "_completion_owner_dispatch_tools", deepcopy(result))
    setattr(runtime, "_completion_owner_actionable", bool(result))
    if isinstance(current_tools, list):
        current_tools[:] = deepcopy(result)
    return result


def apply_menu(runtime: object, tools: Sequence[dict]) -> list[dict]:
    """Project a path-scoped stall without hiding unrelated read paths."""
    floor_available_tools = deepcopy(list(tools))
    baseline_names = {_tool_name(tool) for tool in tools if _tool_name(tool)}

    def finish(projected: Sequence[dict]) -> list[dict]:
        from kernel_capability_floor import enabled as capability_floor_enabled

        if capability_floor_enabled(runtime):
            try:
                from kernel_completion_owner import capability_floor_action_admissible
            except ImportError:
                capability_floor_action_admissible = None

            def admitted_projection_tool(tool: dict) -> bool:
                name = _tool_name(tool)
                if name in baseline_names:
                    return True
                if capability_floor_action_admissible is None:
                    return False
                try:
                    return bool(capability_floor_action_admissible(runtime, name))
                except Exception:  # noqa: BLE001 - synthetic routes fail closed
                    return False

            projected = [tool for tool in projected if admitted_projection_tool(tool)]

        result = _apply_capability_floor(runtime, projected, floor_available_tools)
        if capability_floor_enabled(runtime) and isinstance(tools, list):
            tools[:] = deepcopy(result)
        return result

    try:
        from kernel_completion_owner import owner_done_selection
        owner_selection = owner_done_selection(runtime)
    except (ImportError, AttributeError, TypeError, ValueError):
        owner_selection = None
    if owner_selection is not None:
        # Completion-owner terminal selection is authoritative for this
        # request: no stall/recovery/convergence projection may drop ``done``.
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            emit("completion_owner_terminal_menu_preserved",
                 producer="kernel_tokenlab_stall_breaker",
                 offered=["done"], reason=owner_selection.get("reason", ""),
                 displaced=[_tool_name(tool) for tool in tools],
                 dispatch=False)
        return finish(list(owner_selection["tools"]))
    projected = list(tools)
    recovery = getattr(runtime, "_aha10_recovery_route", None)
    if isinstance(recovery, Mapping) and str(recovery.get("route", "")):
        route_name = str(recovery["route"])
        if not any(_tool_name(tool) == route_name for tool in projected):
            projected.insert(0, _fallback_tool(route_name))
    if stallwire_enabled(runtime):
        try:
            from kernel_completion_owner import project_stall_open_work, stall_rule_enabled
            if stall_rule_enabled(runtime):
                projected = project_stall_open_work(runtime, projected)
            else:
                projected = _project_stallwire_menu(runtime, projected)
        except (ImportError, AttributeError, TypeError, ValueError):
            projected = _project_stallwire_menu(runtime, projected)
    elif enabled(runtime):
        if not v20_enabled(runtime):
            projected = _apply_menu_legacy(runtime, projected)
        else:
            try:
                from kernel_tokenlab_postgreen_no_progress import owned
                if not owned(runtime):
                    projected = _apply_menu_v20(runtime, projected)
            except (ImportError, AttributeError, TypeError, ValueError):
                projected = _apply_menu_v20(runtime, projected)
        projected = _project_stallwire_menu(runtime, projected)
    try:
        from kernel_convergence_observer import apply_menu as apply_convergence_menu
        projected = apply_convergence_menu(runtime, projected)
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    return finish(projected)


__all__ = [
    "EVENT", "MENU_EVENT", "MODE_ENV", "SCHEMA", "V20_EVENT", "V20_MENU_EVENT",
    "V20_MODE_ENV", "V20_RESTORE_EVENT", "V20_SCHEMA", "STALLWIRE_EVENT",
    "STALLWIRE_K_ENV", "STALLWIRE_MENU_EVENT", "STALLWIRE_MODE_ENV",
    "STALLWIRE_PROBE_EVENT", "STALLWIRE_SCHEMA", "StallWireOwner", "StallWireSignal",
    "apply_menu", "enabled", "mode", "note_state_change", "observe_closed_action",
    "observe_read_unchanged", "stallwire_enabled", "stallwire_mode", "stallwire_probe_refused",
    "stallwire_no_progress_edit_capped", "stallwire_no_progress_edit_state",
    "stallwire_signal", "stallwire_threshold", "v20_enabled",
]
