"""Opt-in per-step model routing for StateFrame requests.

The host owns this policy.  A model response is never inspected to decide
where the next request goes: routing consumes only typed step state, host
receipts, and bounded escalation counters.  The default is deliberately
inactive so the existing request bytes and journal shape stay unchanged.

``STATEFRAME_MODEL_ROUTER=on`` enables adaptive routing.  ``shadow`` records
the same assignments but leaves the submitted endpoint and body untouched.
The optional ``STATEFRAME_MODEL_ROUTER_CONFIG=moe_only`` arm is the control
configuration from the model-hopping study: one MoE engine, the same host
state/verification/stop policy, and no frontier handoff.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import islice
from urllib.parse import urlparse


MODE_ENV = "STATEFRAME_MODEL_ROUTER"
CONFIG_ENV = "STATEFRAME_MODEL_ROUTER_CONFIG"
ENDPOINTS_ENV = "STATEFRAME_MODEL_ROUTER_ENDPOINTS"
LEASE_ENV = "STATEFRAME_MODEL_ROUTER_LEASE_ACTIONS"
SCHEMA = "stateframe-model-router/1"
CALL_EVENT = "model_router_call"
ASSIGNMENT_EVENT = "model_router_assignment"
HANDOFF_FIELD = "stateframe_handoff"
ADAPTER_FIELD = "stateframe_adapter"
TOOLS_FIELD = "stateframe_tools"
MANIFEST_MODE_FIELD = "stateframe_model_router"
MANIFEST_SCHEMA_FIELD = "model_router_schema"
MANIFEST_CONFIG_FIELD = "model_router_config"
VALID_MODES = frozenset({"off", "shadow", "on"})
SUPPORTED_CHAT_TEMPLATES = frozenset({"openai", "openai-compatible"})
SUPPORTED_TOOL_FORMATS = frozenset({"openai"})
SUPPORTED_TOKENIZERS = frozenset({"utf8-byte-conservative-v1"})
HANDOFF_CHAR_BUDGET = 2400
HANDOFF_TOKEN_BUDGET = 600
NONPROGRESS_OVERRIDE_DEFAULT = 2
_MISSING = object()


class RouterBudgetError(ValueError):
    """The selected endpoint cannot accept the final prepared request."""

    def __init__(self, required_tokens: int, available_tokens: int):
        super().__init__(
            f"model-router request estimate {required_tokens} exceeds endpoint capacity "
            f"{available_tokens}")
        self.required_tokens = int(required_tokens)
        self.available_tokens = int(available_tokens)


class StepType(str, Enum):
    """Host-visible kinds used by the routing policy."""

    ORDINARY = "ORDINARY"
    DECISION = "DECISION"
    MECHANICAL_VERIFY = "MECHANICAL_VERIFY"
    MECHANICAL_READ = "MECHANICAL_READ"
    TERMINAL_SUBMIT = "TERMINAL_SUBMIT"


def model_router_mode() -> str:
    """Return the validated router switch, which is OFF by default."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off, shadow or on")
    return value


def _token(value: object) -> str:
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _step_type(value: object) -> StepType:
    if isinstance(value, StepType):
        return value
    normalized = _token(value)
    aliases = {
        "ACTION": StepType.DECISION,
        "CODING": StepType.DECISION,
        "REASONING": StepType.DECISION,
        "MECHANICAL": StepType.MECHANICAL_VERIFY,
        "VERIFY": StepType.MECHANICAL_VERIFY,
        "READ": StepType.MECHANICAL_READ,
        "SUBMIT": StepType.TERMINAL_SUBMIT,
        "TERMINAL": StepType.TERMINAL_SUBMIT,
    }
    try:
        return StepType[normalized]
    except KeyError:
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(f"unknown StateFrame model-router step type: {value!r}") from error


def classify_step(frame: Mapping[str, object]) -> StepType:
    """Classify a frame from host-owned fields, never from model prose."""
    if not isinstance(frame, Mapping):
        raise TypeError("model-router step classification requires a mapping")
    for key in ("step_type", "host_step_type", "step_class"):
        value = frame.get(key)
        if isinstance(value, (str, StepType)) and str(value).strip():
            return _step_type(value)
    typed_step = frame.get("step")
    if isinstance(typed_step, Mapping):
        for key in ("type", "step_type", "class"):
            value = typed_step.get(key)
            if isinstance(value, (str, StepType)) and str(value).strip():
                return _step_type(value)

    for key in ("green_lock", "terminal", "terminal_submission"):
        value = frame.get(key)
        if isinstance(value, Mapping) and (
                value.get("engaged") is True or value.get("terminal_engaged") is True):
            return StepType.TERMINAL_SUBMIT
    if isinstance(frame.get("green_lock_message"), str) and frame["green_lock_message"].strip():
        return StepType.TERMINAL_SUBMIT

    actions: list[object] = []
    for container in (frame, frame.get("pending"), frame.get("resume"),
                      frame.get("thinking_router")):
        if not isinstance(container, Mapping):
            continue
        for key in ("next_action", "recommended_next_action", "next_step", "action", "tool"):
            if container.get(key) is not None:
                actions.append(container[key])
    normalized = {_token(value) for value in actions if isinstance(value, str)}
    if normalized & {"SUBMIT", "FINISH", "DONE", "TERMINAL_SUBMIT"}:
        return StepType.TERMINAL_SUBMIT
    if normalized & {"VERIFY", "RUN_TESTS", "MECHANICAL_VERIFY", "TEST_CHANGED_SOURCE"}:
        return StepType.MECHANICAL_VERIFY
    if normalized & {"READ", "READ_FILE", "READ_REQUESTED_FILE", "MECHANICAL_READ"}:
        return StepType.MECHANICAL_READ
    return StepType.ORDINARY


def _as_count(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) == value:
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(value.strip()))
        except ValueError:
            return 1 if value.strip().lower() in {
                "true", "yes", "on", "exhausted", "mismatch", "out_of_scope",
                "conflict", "repeated",
            } else 0
    return 0


_READ_ACTIONS = frozenset({
    "READ", "READ_FILE", "READ_REQUESTED_FILE", "REFRESH_READ", "RECALL_EVIDENCE",
    "MECHANICAL_READ", "GREP",
})
_DISCRIMINATING_ACTIONS = frozenset({
    "EDIT", "STR_REPLACE", "FILE_CREATE", "REPLACE_MODULE", "WRITE", "REPLACE",
    "RUN_TESTS", "TEST", "PYTEST", "PROBE", "VERIFY", "VERIFY_CURRENT_SOURCE",
    "MECHANICAL_VERIFY", "SUBMIT", "DONE",
})
_REJECTED_STATUSES = frozenset({"REJECTED", "FAIL", "FAILED", "RED", "ERROR", "CONFLICT"})
_SUCCESS_STATUSES = frozenset({
    "ACCEPTED", "APPLIED", "COMMITTED", "COMPLETED", "DONE", "GREEN", "OK",
    "PASS", "PASSED", "SUCCESS", "SUCCEEDED",
})


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _row_action(row: Mapping[str, object]) -> str:
    for key in ("tool", "kind", "action", "intent", "operation"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            action = _token(value)
            if action in {"TEST_SUBPROCESS_COMPLETED", "AUTO_VERIFY_COMPLETED",
                          "TEST_OBSERVATION"}:
                return "TEST"
            return action
    if "exit_code" in row:
        return "TEST"
    return ""


def _row_status(row: Mapping[str, object]) -> str:
    for key in ("result_class", "status", "verdict", "result", "outcome", "state"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return _token(value)
    exit_code = row.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return "PASSED" if exit_code == 0 else "FAILED"
    return ""


def _row_receipt_ids(row: Mapping[str, object]) -> tuple[str, ...]:
    nested_ids = row.get("receipt_ids")
    if isinstance(nested_ids, Sequence) and not isinstance(nested_ids, (str, bytes, bytearray)):
        values = list(nested_ids)
    else:
        values = []
    for key in ("receipt_id", "action_receipt_id", "event_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            values.append(value)
            break
    if not values and row.get("attempt_id") not in (None, ""):
        values.append(row["attempt_id"])
    if not values and row.get("step_record_id") not in (None, ""):
        values.append(row["step_record_id"])
    result: list[str] = []
    for value in values:
        if value in (None, ""):
            continue
        identity = str(value)
        if identity and identity not in result:
            result.append(identity)
    return tuple(result)


def _row_receipt_id(row: Mapping[str, object], index: int = 0) -> str | None:
    del index
    # Normalized ledger rows may retain a batched ``receipt_ids`` list while
    # carrying one scalar receipt selected for this row.  The scalar identity
    # is the row's canonical receipt and must win over the first batch member.
    for key in ("_receipt_id", "receipt_id", "action_receipt_id", "event_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    identities = _row_receipt_ids(row)
    return identities[0] if identities else None


def _row_field(row: Mapping[str, object], names: Sequence[str]) -> object:
    nested = (row, _mapping(row.get("observation")), _mapping(row.get("verification")),
              _mapping(row.get("witness")))
    for container in nested:
        if container is None:
            continue
        for name in names:
            value = container.get(name)
            if value not in (None, ""):
                return value
    return None


def _stable_token(value: object) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, default=str).encode()
    except (TypeError, ValueError):
        encoded = repr(value).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()[:24]


def _source_rows(runtime: object | None, frame: Mapping[str, object]) -> list[tuple[str, int, Mapping[str, object]]]:
    """Collect host producers with a source/order tag before normalization."""
    sources: list[tuple[str, object]] = []
    if runtime is not None:
        ledger = getattr(runtime, "step_ledger", None)
        records = getattr(ledger, "records", None)
        if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
            sources.append(("ledger", records))
        for name in ("test_receipts", "probe_receipts", "observation_probe_receipts"):
            value = getattr(runtime, name, None)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                sources.append((name, value))
        owner = getattr(runtime, "owner", None)
        value = getattr(owner, "_receipts", None)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sources.append(("owner", value))
    for key in ("committed_receipts", "test_receipts", "verification_receipts",
                "probe_receipts", "action_receipts"):
        value = frame.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            sources.append((key, value))
    latest = frame.get("latest_receipts")
    if isinstance(latest, Mapping):
        sources.append(("latest", tuple(value for value in latest.values()
                                         if isinstance(value, Mapping))))
    rows: list[tuple[str, int, Mapping[str, object]]] = []
    for source_name, source in sources:
        for index, row in enumerate(source):
            if isinstance(row, Mapping):
                rows.append((source_name, index, row))
    return rows


def _source_collection_token(value: object) -> tuple[object, ...]:
    from kernel_history_spool import HistorySpool
    if isinstance(value, HistorySpool):
        return (id(value), len(value), 0 if value else None, len(value) - 1 if value else None)
    if isinstance(value, Mapping):
        values = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
    else:
        return (type(value).__name__, id(value))
    return (id(value), len(values), id(values[0]) if values else None,
            id(values[-1]) if values else None)


def _projection_input_token(runtime: object | None, frame: Mapping[str, object]) -> tuple[object, ...]:
    values: list[tuple[str, object]] = []
    if runtime is not None:
        ledger = getattr(runtime, "step_ledger", None)
        values.append(("ledger", getattr(ledger, "records", None)))
        for name in ("test_receipts", "probe_receipts", "observation_probe_receipts"):
            values.append((name, getattr(runtime, name, None)))
        values.append(("owner", getattr(getattr(runtime, "owner", None), "_receipts", None)))
    for key in ("committed_receipts", "test_receipts", "verification_receipts",
                "probe_receipts", "action_receipts", "latest_receipts"):
        values.append((key, frame.get(key)))
    scalar_inputs = tuple(
        (key, _stable_token(frame.get(key)))
        for key in ("scope_mismatch", "out_of_scope", "scope", "host_closure",
                    "verification", "verification_conflict", "host_escalation_signals",
                    "budget", "workspace_revision")
    )
    return (tuple((name, _source_collection_token(value)) for name, value in values),
            scalar_inputs,
            getattr(runtime, "workspace_revision", None) if runtime is not None else None)


def _row_binding(row: Mapping[str, object]) -> Mapping[str, object] | None:
    for key in ("after_binding", "binding", "current_binding", "workspace_binding",
                "scope"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _current_binding(runtime: object | None, frame: Mapping[str, object]) -> Mapping[str, object]:
    if runtime is not None:
        cached = getattr(runtime, "_model_router_binding_cache", None)
        if isinstance(cached, Mapping) and cached.get("frame") is frame:
            value = cached.get("value")
            if isinstance(value, Mapping):
                return value
    if runtime is not None:
        binding = getattr(runtime, "binding", None)
        if callable(binding):
            try:
                value = binding()
                if isinstance(value, Mapping):
                    if runtime is not None:
                        setattr(runtime, "_model_router_binding_cache",
                                {"frame": frame, "value": value})
                    return value
            except (OSError, TypeError, ValueError):
                pass
    for key in ("current_binding", "binding", "workspace_binding"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            if runtime is not None:
                setattr(runtime, "_model_router_binding_cache",
                        {"frame": frame, "value": value})
            return value
    artifacts = frame.get("artifact_revisions")
    if isinstance(artifacts, Mapping):
        value = {"product_files": artifacts,
                 "environment_revision": frame.get("environment_revision")}
        if runtime is not None:
            setattr(runtime, "_model_router_binding_cache",
                    {"frame": frame, "value": value})
        return value
    if runtime is not None:
        setattr(runtime, "_model_router_binding_cache", {"frame": frame, "value": {}})
    return {}


def _current_workspace_revision(runtime: object | None, frame: Mapping[str, object]) -> object:
    candidates: list[object] = [frame.get("workspace_revision")]
    if runtime is not None:
        candidates.extend((getattr(runtime, "workspace_revision", None),
                           getattr(getattr(runtime, "owner", None),
                                   "workspace_revision", None)))
    for value in candidates:
        if value not in (None, ""):
            return value
    return None


def _same_revision(left: object, right: object) -> bool:
    if left in (None, "") or right in (None, ""):
        return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in ("product_revision", "artifact", "revision", "environment_revision"):
            if left.get(key) not in (None, "") and right.get(key) not in (None, ""):
                if left[key] != right[key]:
                    return False
        for key in ("product_files", "harness_files"):
            left_files, right_files = left.get(key), right.get(key)
            if isinstance(left_files, Mapping) and isinstance(right_files, Mapping):
                for path in set(left_files) & set(right_files):
                    if left_files[path] != right_files[path]:
                        return False
        return True
    return left == right


def _slim_after_matches(row, runtime, current) -> bool:
    from kernel_measurement import digest
    if row.get("after_binding_digest") == digest(current):
        return True
    ledger = getattr(runtime, "step_ledger", None)
    index = row.get("_ledger_index")
    return (type(index) is int and ledger is not None
            and ledger.after_binding_matches(index, current))


def _revision_relevant(row: Mapping[str, object], runtime: object | None,
                       frame: Mapping[str, object]) -> bool:
    current_workspace = _current_workspace_revision(runtime, frame)
    row_workspace = row.get("workspace_revision")
    if current_workspace not in (None, "") and row_workspace not in (None, ""):
        if not _same_revision(row_workspace, current_workspace):
            return False
    current = _current_binding(runtime, frame)
    binding = _row_binding(row)
    if (current and not isinstance(row.get("after_binding"), Mapping)
            and isinstance(row.get("after_binding_digest"), str)):
        # A slim after-binding has the same precedence as a full ledger row,
        # even when a joined producer receipt carries its older binding.
        if not _slim_after_matches(row, runtime, current):
            return False
        binding = None
    if binding is not None and current and not _same_revision(binding, current):
        return False
    source_revision = row.get("source_revision")
    current_product = current.get("product_revision", current.get("revision"))
    if (source_revision not in (None, "") and current_product not in (None, "")
            and source_revision != current_product):
        return False
    return True


def _row_order(row: Mapping[str, object], source: str, index: int) -> tuple[int, float, int, int]:
    for key in ("step", "turn"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            source_rank = {"ledger": 0, "owner": 1}.get(source, 2)
            return (0, float(value), source_rank, index)
    for key in ("sequence", "event_sequence", "journal_sequence"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (1, float(value), 0, index)
    for key in ("timestamp_ns", "ts"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (2, float(value), 0, index)
    return (3, float(index), 0, index)


_ACTION_ID_FIELDS = (
    "host_action_id", "verification_id", "action_identity", "completion_id",
    "action_id", "test_action_id", "test_receipt_id", "parent_action_id",
)


def _completion_identity(row: Mapping[str, object]) -> str | None:
    """Return the host identity shared by all records for one action.

    A receipt/event id is only the fallback.  Producers that emit an owner
    receipt plus a subprocess observation must carry one of the explicit
    action/verification ids above; step numbers are deliberately not an
    identity because one host step may contain a batch of distinct actions.
    """
    nested_ids = row.get("receipt_ids")
    scalar_receipt = row.get("receipt_id")
    if (isinstance(nested_ids, Sequence)
            and not isinstance(nested_ids, (str, bytes, bytearray))
            and len(tuple(nested_ids)) > 1
            and scalar_receipt not in (None, "")
            and not any(row.get(key) not in (None, "") for key in _ACTION_ID_FIELDS)):
        # A single ledger step can carry a genuine batch.  Once normalization
        # selected one scalar receipt, its identity stays distinct from the
        # other members even if the enclosing row has a common step.
        return str(scalar_receipt)
    for key in _ACTION_ID_FIELDS:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    receipt_id = _row_receipt_id(row)
    return str(receipt_id) if receipt_id is not None else None


def _normalize_receipts(runtime: object | None, frame: Mapping[str, object]) -> list[dict[str, object]]:
    """Join producer rows into an ordered stream without freshness filtering.

    The returned rows are completion facts.  Revision/stability filtering is
    intentionally a later evidence projection so a receipt stamped just before
    a successful edit can still consume its lease exactly once.
    """
    source_rows = _source_rows(runtime, frame)
    raw_by_id: dict[str, tuple[str, int, Mapping[str, object]]] = {}
    for source, index, row in source_rows:
        for receipt_id in _row_receipt_ids(row):
            prior = raw_by_id.get(receipt_id)
            if prior is None or (prior[0] == "ledger" and source != "ledger"):
                raw_by_id[receipt_id] = (source, index, row)

    normalized: list[dict[str, object]] = []
    emitted: set[str] = set()
    for source, index, row in source_rows:
        if source == "ledger" and row.get("receipt_ids"):
            identities = _row_receipt_ids(row)
            for receipt_id in identities:
                matched = raw_by_id.get(receipt_id)
                if matched is not None and matched[2] is not row:
                    merged = {**dict(row), **dict(matched[2])}
                    merged["_ledger_index"] = index
                    for key in ("before_binding", "after_binding", "step", "step_record_id"):
                        if key not in merged and key in row:
                            merged[key] = row[key]
                    normalized.append({
                        **merged, "receipt_id": receipt_id,
                        "_order": _row_order(row, source, index),
                    })
                else:
                    normalized.append({**dict(row), "_ledger_index": index,
                                       "receipt_id": receipt_id,
                                       "_order": _row_order(row, source, index)})
                emitted.add(receipt_id)
            continue
        identities = _row_receipt_ids(row)
        if not identities:
            continue
        for receipt_id in identities:
            if receipt_id in emitted:
                continue
            normalized.append({**dict(row), "receipt_id": receipt_id,
                               "_order": _row_order(row, source, index)})
            emitted.add(receipt_id)
    normalized.sort(key=lambda row: row["_order"])
    for row in normalized:
        action_id = _completion_identity(row)
        if action_id is not None:
            row["_action_id"] = action_id
    return normalized


def _component_rows(row: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    components = row.get("_components")
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes, bytearray)):
        result = tuple(item for item in components if isinstance(item, Mapping))
        if result:
            return result
    return (row,)


def _collapse_action_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Collapse duplicate producer records, retaining distinct batched actions."""
    collapsed: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        action_id = str(row.get("_action_id") or _completion_identity(row) or "")
        if not action_id:
            continue
        existing = collapsed.get(action_id)
        if existing is None:
            merged = dict(row)
            merged["_components"] = [row]
            merged["_action_id"] = action_id
            collapsed[action_id] = merged
            order.append(action_id)
            continue
        components = existing.setdefault("_components", [])
        if isinstance(components, list):
            components.append(row)
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if key not in existing or existing[key] in (None, ""):
                existing[key] = value
        # An explicit producer flag is authoritative across the group.  This
        # prevents a missing flag on the subprocess event from changing the
        # owner receipt's completion classification (and vice versa).
        existing["meaningful_action"] = any(
            _receipt_is_meaningful(component)
            for component in _component_rows(existing))
    return [collapsed[action_id] for action_id in order]


def _is_subprocess_test_row(row: Mapping[str, object]) -> bool:
    action = _row_action(row)
    kind = _token(row.get("kind", ""))
    producer = _token(row.get("producer", ""))
    return action == "TEST" and (
        producer == "TEST_SUBPROCESS" or kind in {"TEST_SUBPROCESS_COMPLETED", "TEST_SUBPROCESS"})


def _is_test_row(row: Mapping[str, object]) -> bool:
    action = _row_action(row)
    kind = _token(row.get("kind", ""))
    return action == "TEST" or kind in {"TEST", "VERIFICATION", "RUN_TESTS", "PYTEST"}


def _test_evidence_is_current(row: Mapping[str, object],
                              current: Mapping[str, object]) -> bool:
    """Accept test evidence only after a stable test of the current binding."""
    components = _component_rows(row)
    subprocess_rows = tuple(item for item in components if _is_subprocess_test_row(item))
    test_rows = subprocess_rows or tuple(item for item in components if _is_test_row(item))
    if not test_rows:
        return True
    for test_row in test_rows:
        if test_row.get("binding_stable") is not True:
            return False
        tested_binding = test_row.get("binding")
        after_binding = test_row.get("after_binding")
        if not isinstance(tested_binding, Mapping):
            return False
        if current and not _same_revision(tested_binding, current):
            return False
        if isinstance(after_binding, Mapping) and current and not _same_revision(after_binding, current):
            return False
    return True


def _evidence_relevant(row: Mapping[str, object], runtime: object | None,
                       frame: Mapping[str, object]) -> bool:
    components = _component_rows(row)
    subprocess_rows = tuple(item for item in components if _is_subprocess_test_row(item))
    revision_rows = subprocess_rows or (row,)
    current = _current_binding(runtime, frame)
    if not _test_evidence_is_current(row, current):
        return False
    if any(_revision_relevant(item, runtime, frame) for item in revision_rows):
        return True
    # The action producer records an applied edit before incrementing the
    # owner's revision.  Its successful after-binding is still valid for
    # resetting a failure streak; only stale failure evidence is discarded.
    if _receipt_is_successful(row):
        if (current and not isinstance(row.get("after_binding"), Mapping)
            and isinstance(row.get("after_binding_digest"), str)):
            return _slim_after_matches(row, runtime, current)
        after_binding = row.get("after_binding")
        return (isinstance(after_binding, Mapping) and bool(current)
                and _same_revision(after_binding, current))
    return False


def _receipt_is_meaningful(row: Mapping[str, object]) -> bool:
    explicit = row.get("meaningful_action")
    if isinstance(explicit, bool):
        return explicit
    action = _row_action(row)
    if (not action or action in _READ_ACTIONS
            or action in {"NO_ACTION", "PREPARE", "VERDICT", "OBSERVATION", "TELEMETRY"}):
        return False
    return True


def _receipt_is_successful(row: Mapping[str, object]) -> bool:
    """Return true for an explicitly successful meaningful host action."""
    return (_row_action(row) not in _READ_ACTIONS
            and _receipt_is_meaningful(row)
            and _row_status(row) in _SUCCESS_STATUSES)


def _failure_token(row: Mapping[str, object], *, hypothesis: bool = False) -> str | None:
    names = (
        ("hypothesis_id", "hypothesis_digest", "rejected_hypothesis_id")
        if hypothesis else
        ("counterexample_digest", "failure_signature_digest", "witness_digest",
         "counterexample_id", "same_valid_counterexample_id")
    )
    value = _row_field(row, names)
    if value not in (None, ""):
        return str(value)
    # The real owner receipt has a stable fingerprint rather than the older
    # counterexample/hypothesis aliases.  It is the host's generic identity
    # for a repeated rejected action; model prose is never hashed here.
    value = _row_field(row, ("fingerprint", "attempt_id"))
    if value not in (None, ""):
        return str(value)
    if not hypothesis and _row_action(row) in {"TEST", "RUN_TESTS", "PYTEST", "PROBE"}:
        selectors = row.get("selectors", row.get("target", row.get("kind")))
        if selectors not in (None, ""):
            return _stable_token(selectors)
    return None


def _budget_projection(runtime: object | None, frame: Mapping[str, object]) -> dict[str, object]:
    values: list[Mapping[str, object]] = []
    budget = frame.get("budget")
    if isinstance(budget, Mapping):
        values.append(budget)
    for candidate in (
        getattr(runtime, "decision_budget", None) if runtime is not None else None,
        getattr(runtime, "parent", None) if runtime is not None else None,
    ):
        if candidate is None:
            continue
        snapshot = candidate.snapshot() if callable(getattr(candidate, "snapshot", None)) else candidate
        if isinstance(snapshot, Mapping):
            values.append(snapshot)
    result: dict[str, object] = {}
    for value in values:
        for key in ("remaining", "remaining_steps", "decisions_remaining", "responses_used",
                    "limit", "attempts_remaining", "model_calls_remaining", "exhausted",
                    "fuel_exhausted"):
            if key in value:
                result[key] = value[key]
    exhausted = any(value.get(key) is True for value in values
                    for key in ("exhausted", "fuel_exhausted", "budget_exhausted"))
    for value in values:
        for key in ("remaining", "remaining_steps", "decisions_remaining",
                    "attempts_remaining", "model_calls_remaining"):
            raw = value.get(key)
            if isinstance(raw, (int, float)) and raw <= 0:
                exhausted = True
    result["exhausted"] = exhausted
    return result


def _host_signal_projection_uncached(runtime: object | None,
                                     frame: Mapping[str, object]) -> dict[str, object]:
    """Project revision-bound host receipts into the six router counters.

    This is deliberately conservative: repeated failures require an explicit
    host digest/id, reads require committed receipt rows, and budget/scope
    signals come from typed host fields.  Model text and arbitrary prose never
    contributes to this projection.
    """
    completion_rows = _collapse_action_rows(_normalize_receipts(runtime, frame))
    evidence_rows = [row for row in completion_rows
                     if _evidence_relevant(row, runtime, frame)]
    counterexample_counts: dict[str, int] = {}
    hypothesis_counts: dict[str, int] = {}
    reads_since_action = 0
    for row in evidence_rows:
        action = _row_action(row)
        status = _row_status(row)
        if action in _READ_ACTIONS:
            reads_since_action += 1
        elif action in _DISCRIMINATING_ACTIONS or _receipt_is_meaningful(row):
            reads_since_action = 0
        if _receipt_is_successful(row):
            # A successful host action makes earlier rejected attempts stale
            # for escalation.  Keep only the current consecutive failure
            # streak in this relevant, revision-bound stream.
            counterexample_counts.clear()
            hypothesis_counts.clear()
        failure_token = _failure_token(row)
        if failure_token not in (None, "") and status in _REJECTED_STATUSES:
            key = str(failure_token)
            counterexample_counts[key] = counterexample_counts.get(key, 0) + 1
        hypothesis_token = _failure_token(row, hypothesis=True)
        if hypothesis_token not in (None, "") and status in _REJECTED_STATUSES:
            key = str(hypothesis_token)
            hypothesis_counts[key] = hypothesis_counts.get(key, 0) + 1

    explicit_scope = max(
        _as_count(frame.get("scope_mismatch")),
        _as_count(frame.get("out_of_scope")),
    )
    scope = _mapping(frame.get("scope")) or _mapping(frame.get("host_closure"))
    if scope is not None:
        explicit_scope = max(explicit_scope, _as_count(scope.get("scope_mismatch")))
        if _token(scope.get("status", "")) in {"MISMATCH", "OUT_OF_SCOPE", "SCOPE_MISMATCH"}:
            explicit_scope = max(1, explicit_scope)
    verification = _mapping(frame.get("verification"))
    verification_conflict = _as_count(frame.get("verification_conflict"))
    if verification is not None:
        verification_conflict = max(verification_conflict,
                                    _as_count(verification.get("conflict")))
        if _token(verification.get("status", "")) in {"CONFLICT", "MISMATCH"}:
            verification_conflict = max(1, verification_conflict)
    economic = _budget_projection(runtime, frame)
    explicit = _mapping(frame.get("host_escalation_signals")) or {}
    signals = {
        "same_valid_counterexample": max(
            counterexample_counts.values(), default=0,
            key=lambda value: value),
        "rejected_hypothesis_repeats": max(hypothesis_counts.values(), default=0),
        "reads_without_discriminating_action": reads_since_action,
        "scope_mismatch": explicit_scope,
        "verification_conflict": verification_conflict,
        "economic_exhaustion": int(bool(economic.get("exhausted"))),
    }
    for name in tuple(signals):
        signals[name] = max(signals[name], _as_count(explicit.get(name)))
    latest_meaningful = next((row for row in reversed(evidence_rows)
                              if _row_receipt_id(row) is not None
                              and _receipt_is_meaningful(row)), None)
    committed_id = (_row_receipt_id(latest_meaningful, 0)
                    if latest_meaningful is not None else None)
    def public_receipt(row: Mapping[str, object], *, evidence_valid: bool) -> dict[str, object] | None:
        receipt_id = _row_receipt_id(row)
        if receipt_id is None:
            return None
        action_id = str(row.get("_action_id") or receipt_id)
        return {
            "receipt_id": receipt_id,
            "action_id": action_id,
            "action": _row_action(row),
            "status": _row_status(row),
            "meaningful_action": _receipt_is_meaningful(row),
            "evidence_valid": evidence_valid,
            **({"workspace_revision": row["workspace_revision"]}
               if row.get("workspace_revision") not in (None, "") else {}),
            **({"binding": copy.deepcopy(dict(_row_binding(row)))}
               if _row_binding(row) is not None else {}),
        }

    completion_stream = [item for row in completion_rows
                         if (item := public_receipt(row, evidence_valid=_evidence_relevant(
                             row, runtime, frame))) is not None]
    receipt_stream = [item for row in evidence_rows
                      if (item := public_receipt(row, evidence_valid=True)) is not None]
    completed_meaningful = [row for row in completion_stream
                            if row["meaningful_action"]]
    committed_meaningful = [row for row in receipt_stream
                            if row["meaningful_action"]]
    completed_id = (completed_meaningful[-1]["action_id"]
                    if completed_meaningful else None)
    return {
        "schema": "stateframe-model-router-host-projection/1",
        "signals": signals,
        "receipt_count": len(completion_stream),
        "evidence_receipt_count": len(receipt_stream),
        "receipts": receipt_stream,
        "completed_receipts": completion_stream,
        "committed_action_receipt_ids": [row["action_id"] for row in committed_meaningful],
        "committed_action_receipt_id": committed_id,
        "committed_action_meaningful": latest_meaningful is not None,
        "completed_action_receipt_ids": [row["action_id"] for row in completed_meaningful],
        "completed_action_receipt_id": completed_id,
        "budget": economic,
    }


def host_signal_projection(runtime: object | None, frame: Mapping[str, object]) -> dict[str, object]:
    """Return one stable projection for one frame/source revision.

    Several request-boundary consumers ask for the same projection.  Cache
    hits are limited to the identical frame and append-only source identities;
    a new receipt, replacement collection, or new frame recomputes the full
    safety projection.
    """
    token = (id(frame), _projection_input_token(runtime, frame))
    cached = getattr(runtime, "_model_router_projection_cache", None) if runtime is not None else None
    if isinstance(cached, Mapping) and cached.get("token") == token:
        result = cached.get("result")
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    result = _host_signal_projection_uncached(runtime, frame)
    if runtime is not None:
        setattr(runtime, "_model_router_projection_cache",
                {"token": token, "result": copy.deepcopy(result)})
    return result


@dataclass(frozen=True)
class EscalationSignals:
    """Bounded host-observed counters used by the generic policy."""

    same_valid_counterexample: int = 0
    rejected_hypothesis_repeats: int = 0
    reads_without_discriminating_action: int = 0
    scope_mismatch: int = 0
    verification_conflict: int = 0
    economic_exhaustion: int = 0

    @classmethod
    def from_frame(cls, frame: Mapping[str, object]) -> "EscalationSignals":
        host_projection = frame.get("model_router_host_projection")
        if isinstance(host_projection, Mapping):
            projected = host_projection.get("signals")
            if isinstance(projected, Mapping):
                return cls(
                    same_valid_counterexample=_as_count(
                        projected.get("same_valid_counterexample")),
                    rejected_hypothesis_repeats=_as_count(
                        projected.get("rejected_hypothesis_repeats")),
                    reads_without_discriminating_action=_as_count(
                        projected.get("reads_without_discriminating_action")),
                    scope_mismatch=_as_count(projected.get("scope_mismatch")),
                    verification_conflict=_as_count(projected.get("verification_conflict")),
                    economic_exhaustion=_as_count(projected.get("economic_exhaustion")),
                )
        # No untrusted frame alias is a routing input.  A direct caller that
        # owns a typed counter projection must mark it explicitly; production
        # runtime calls use host_signal_projection above.
        return cls()

    def as_dict(self) -> dict[str, int]:
        return {
            "same_valid_counterexample": self.same_valid_counterexample,
            "rejected_hypothesis_repeats": self.rejected_hypothesis_repeats,
            "reads_without_discriminating_action": self.reads_without_discriminating_action,
            "scope_mismatch": self.scope_mismatch,
            "verification_conflict": self.verification_conflict,
            "economic_exhaustion": self.economic_exhaustion,
        }

    def reasons(self, thresholds: "RouterThresholds") -> tuple[str, ...]:
        checks = (
            ("same_valid_counterexample", self.same_valid_counterexample,
             thresholds.same_counterexample),
            ("rejected_hypothesis_repeats", self.rejected_hypothesis_repeats,
             thresholds.rejected_hypothesis),
            ("reads_without_discriminating_action", self.reads_without_discriminating_action,
             thresholds.nondiscriminating_reads),
            ("scope_mismatch", self.scope_mismatch, thresholds.scope_mismatch),
            ("verification_conflict", self.verification_conflict,
             thresholds.verification_conflict),
            ("economic_exhaustion", self.economic_exhaustion,
             thresholds.economic_exhaustion),
        )
        return tuple(name for name, value, threshold in checks if value >= threshold)


@dataclass(frozen=True)
class RouterThresholds:
    """Asymmetric escalation/de-escalation thresholds."""

    same_counterexample: int = 2
    rejected_hypothesis: int = 2
    nondiscriminating_reads: int = 3
    scope_mismatch: int = 1
    verification_conflict: int = 1
    economic_exhaustion: int = 1
    stable_actions_to_deescalate: int = 3

    def __post_init__(self) -> None:
        for name in (
            "same_counterexample", "rejected_hypothesis", "nondiscriminating_reads",
            "scope_mismatch", "verification_conflict", "economic_exhaustion",
            "stable_actions_to_deescalate",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"router threshold {name} must be a positive integer")

    def as_dict(self) -> dict[str, int]:
        return {
            "same_counterexample": self.same_counterexample,
            "rejected_hypothesis": self.rejected_hypothesis,
            "nondiscriminating_reads": self.nondiscriminating_reads,
            "scope_mismatch": self.scope_mismatch,
            "verification_conflict": self.verification_conflict,
            "economic_exhaustion": self.economic_exhaustion,
            "stable_actions_to_deescalate": self.stable_actions_to_deescalate,
        }


def _set_path(mapping: dict[str, object], path: str, value: object) -> None:
    parts = [part for part in path.split(".") if part]
    if not parts:
        return
    cursor = mapping
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, Mapping):
            child = {}
            cursor[part] = child
        else:
            child = copy.deepcopy(dict(child))
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


_PRIVATE_HANDOFF_KEYS = frozenset({
    "analysis", "chain_of_thought", "cot", "raw_thinking", "reasoning",
    "reasoning_content", "thinking", "thoughts", "internal_monologue",
    "messages", "prompt", "completion",
})


def _public_value(value: object, *, depth: int = 0,
                  budget: list[int] | None = None) -> object:
    """Copy bounded public decision/evidence without private reasoning.

    ``islice`` keeps hostile/unbounded containers from being materialized before
    their field and aggregate limits are applied.
    """
    if budget is None:
        budget = [HANDOFF_CHAR_BUDGET]
    if depth > 4 or budget[0] <= 0:
        return "[bounded]"
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in islice(value.items(), 32):
            if not isinstance(key, str) or key.lower() in _PRIVATE_HANDOFF_KEYS:
                continue
            key_cost = len(json.dumps(key, separators=(",", ":"))) + 1
            if budget[0] <= key_cost:
                break
            budget[0] -= key_cost
            lowered = key.lower()
            if any(token in lowered for token in (
                    "reasoning", "thinking", "analysis", "chain_of_thought")):
                continue
            result[key] = _public_value(child, depth=depth + 1, budget=budget)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for child in islice(iter(value), 32):
            if budget[0] <= 2:
                break
            budget[0] -= 1
            result.append(_public_value(child, depth=depth + 1, budget=budget))
        return result
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            limit = min(1200, max(0, budget[0]))
            result = value[:limit]
            budget[0] -= len(result)
            return result
        budget[0] -= min(32, budget[0])
        return value
    result = str(value)[:min(240, max(0, budget[0]))]
    budget[0] -= len(result)
    return result


def build_handoff(frame: Mapping[str, object], *, why_escalated: Sequence[str] = (),
                  max_tokens: int = HANDOFF_TOKEN_BUDGET) -> dict[str, object]:
    """Build the cross-engine context from decisions and host evidence only."""
    step = classify_step(frame)
    handoff: dict[str, object] = {"step_type": step.value}
    for output_key, frame_keys in (
        ("decision", ("decision", "host_decision", "current_decision", "next_action")),
        ("evidence", ("evidence", "evidence_refs", "receipts", "verification")),
        ("last_action", ("last_action", "last_receipt")),
        ("scope", ("scope", "scope_binding")),
    ):
        for key in frame_keys:
            if key in frame and frame[key] is not None:
                handoff[output_key] = _public_value(frame[key])
                break
    signals = EscalationSignals.from_frame(frame)
    if any(signals.as_dict().values()):
        handoff["escalation_signals"] = signals.as_dict()
    if why_escalated:
        handoff["why_escalated"] = list(dict.fromkeys(str(item) for item in why_escalated))
    try:
        aggregate_budget = max(1, int(max_tokens)) * 4
    except (TypeError, ValueError):
        aggregate_budget = HANDOFF_CHAR_BUDGET
    return _public_value(handoff, budget=[min(HANDOFF_CHAR_BUDGET, aggregate_budget)])


@dataclass(frozen=True)
class EngineAdapter:
    """Validated OpenAI-compatible wire policy for one endpoint.

    Provider-specific JSON/tool envelopes are intentionally out of scope for
    this minimal lane.  A label is not an adapter: unsupported formats fail at
    configuration time instead of being moved into an unconsumed top-level
    field.
    """

    chat_template: str = "openai"
    tool_format: str = "openai"
    thinking_field: str = "reasoning_effort"
    thinking: str = "passthrough"
    tokenizer: str = "utf8-byte-conservative-v1"

    def __post_init__(self) -> None:
        chat_template = str(self.chat_template).strip().lower()
        tool_format = str(self.tool_format).strip().lower()
        tokenizer = str(self.tokenizer).strip().lower()
        if chat_template not in SUPPORTED_CHAT_TEMPLATES:
            raise ValueError(
                "adapter chat_template must be openai-compatible; unsupported format: "
                + chat_template)
        if tool_format not in SUPPORTED_TOOL_FORMATS:
            raise ValueError(
                "adapter tool_format must be openai; unsupported format: "
                + tool_format)
        if tokenizer not in SUPPORTED_TOKENIZERS:
            raise ValueError(
                "adapter tokenizer must use a selected conservative counter; unsupported tokenizer: "
                + tokenizer)
        thinking = ("on" if self.thinking else "off") if isinstance(self.thinking, bool) \
            else str(self.thinking).strip().lower()
        if thinking not in {"on", "off", "passthrough"}:
            raise ValueError("adapter thinking must be on, off or passthrough")
        object.__setattr__(self, "chat_template", "openai")
        object.__setattr__(self, "tool_format", tool_format)
        object.__setattr__(self, "thinking", thinking)
        object.__setattr__(self, "tokenizer", tokenizer)

    def as_dict(self) -> dict[str, str]:
        return {
            "chat_template": self.chat_template,
            "tool_format": self.tool_format,
            "thinking_field": self.thinking_field,
            "thinking": self.thinking,
            "tokenizer": self.tokenizer,
        }

    def estimate_tokens(self, value: object) -> int:
        """Use the selected endpoint's conservative UTF-8 token bound.

        The router has no provider tokenizer offline.  One token per serialized
        UTF-8 byte is intentionally an overestimate for the JSON wire body;
        it is safe for admission and is recorded as a bound, not a measured
        provider usage value.
        """
        if self.tokenizer != "utf8-byte-conservative-v1":
            raise ValueError("unsupported selected tokenizer")
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, default=str).encode("utf-8")
        return max(1, len(encoded))

    def render_handoff(self, handoff: Mapping[str, object]) -> str:
        """Render only the public handoff into a provider message."""
        public = _public_value(handoff, budget=[HANDOFF_CHAR_BUDGET])
        rendered = json.dumps({HANDOFF_FIELD: public}, sort_keys=True, separators=(",", ":"))
        if len(rendered) > HANDOFF_CHAR_BUDGET:
            rendered = json.dumps({HANDOFF_FIELD: {"truncated": True}},
                                  sort_keys=True, separators=(",", ":"))
        return rendered

    def handoff_message(self, handoff: Mapping[str, object]) -> dict[str, str]:
        return {
            "role": "user",
            "content": "[StateFrame model handoff]\n" + self.render_handoff(handoff),
        }

    @staticmethod
    def _insert_handoff(messages: Sequence[Mapping[str, object]],
                        handoff_message: Mapping[str, object]) -> list[dict[str, object]]:
        result = [copy.deepcopy(dict(message)) for message in messages]
        if any(str(message.get("content", "")).startswith("[StateFrame model handoff]")
               for message in result if isinstance(message, Mapping)):
            return result
        insert_at = len(result)
        state_indices: list[int] = []
        for index, message in enumerate(result):
            content = str(message.get("content", "")) if isinstance(message, Mapping) else ""
            if content.startswith(("[StateFrame]", "[StateFrame Current State]",
                                   "[H+state+ checklist]", "[StateFrame v2 S1",
                                   "[StateFrame advisory sidecar]")):
                state_indices.append(index)
        if result:
            last = result[-1]
            content = str(last.get("content", "")) if isinstance(last, Mapping) else ""
            if (last.get("role") == "assistant" and
                    ("reasoning_content" in last or last.get("continue_final_message"))):
                insert_at = len(result) - 1
            elif state_indices and state_indices[-1] == len(result) - 1:
                insert_at = state_indices[-1]
        if state_indices:
            # Keep the handoff before the authoritative state tail even when
            # an assistant reasoning prefill follows that tail.
            insert_at = min(insert_at, state_indices[-1])
        result.insert(insert_at, copy.deepcopy(dict(handoff_message)))
        return result

    def prepare_messages(self, messages: Sequence[Mapping[str, object]], tools: object,
                         *, handoff: Mapping[str, object], capacity_tokens: int | None = None,
                         host_thinking_off: bool = False) -> tuple[list[dict[str, object]], object]:
        """Compose one bounded handoff before final budget/prefill/layout passes."""
        prepared = self._insert_handoff(messages, self.handoff_message(handoff))
        prepared_tools = copy.deepcopy(tools)
        if capacity_tokens is not None:
            after = self.estimate_tokens({"messages": prepared, "tools": prepared_tools})
            if after > capacity_tokens:
                raise RouterBudgetError(after, capacity_tokens)
        return prepared, prepared_tools

    def adapt_request_fields(self, body: Mapping[str, object], *, model_id: str,
                             host_thinking_off: bool = False) -> dict[str, object]:
        """Apply only scalar/request-field policy to an already prepared body."""
        result = copy.deepcopy(dict(body))
        result["model"] = model_id
        result.pop(TOOLS_FIELD, None)
        existing_kwargs = result.get("chat_template_kwargs")
        thinking_already_off = (
            isinstance(existing_kwargs, Mapping)
            and existing_kwargs.get("enable_thinking") is False
        ) or result.get("reasoning_effort") == "none"
        if (self.thinking != "passthrough" and self.thinking_field
                and not host_thinking_off and not thinking_already_off):
            if self.thinking_field.endswith("enable_thinking"):
                value: object = self.thinking == "on"
            elif self.thinking_field.endswith("reasoning_effort"):
                value = "high" if self.thinking == "on" else "none"
            else:
                value = self.thinking == "on"
            _set_path(result, self.thinking_field, value)
        return result

    def adapt_request(self, body: Mapping[str, object], *, handoff: Mapping[str, object],
                      model_id: str) -> dict[str, object]:
        """Return an engine-shaped request with a public-only handoff."""
        if not isinstance(body, Mapping):
            raise TypeError("model-router request body must be a mapping")
        result = self.adapt_request_fields(body, model_id=model_id)
        messages = result.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes, bytearray)):
            result["messages"], result["tools"] = self.prepare_messages(
                messages, result.get("tools"), handoff=handoff,
                host_thinking_off=False)
        return result

    adapt = adapt_request


def _normalize_allowed(values: object) -> frozenset[StepType]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, StepType)):
        values = [values]
    elif isinstance(values, (set, frozenset)):
        values = tuple(values)
    if not isinstance(values, Sequence):
        raise ValueError("endpoint allowed_step_types must be a sequence")
    return frozenset(_step_type(item) for item in values)


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    model_id: str
    role: str = "moe"
    url: str | None = None
    api_key_env: str | None = None
    allowed_step_types: frozenset[StepType] = field(default_factory=frozenset)
    adapter: EngineAdapter = field(default_factory=EngineAdapter)
    inherit_api_key: bool = False
    allow_empty_key: bool = False
    context_tokens: int | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.model_id).strip():
            raise ValueError("router endpoint name and model_id must be nonempty")
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "model_id", str(self.model_id).strip())
        object.__setattr__(self, "role", _token(self.role).lower())
        object.__setattr__(self, "allowed_step_types", _normalize_allowed(self.allowed_step_types))
        if not isinstance(self.adapter, EngineAdapter):
            raise TypeError("endpoint adapter must be an EngineAdapter")
        if self.narrow and not self.allowed_step_types:
            raise ValueError("narrow endpoint requires an explicit step allowlist")
        if (self.context_tokens is not None
                and (isinstance(self.context_tokens, bool)
                     or not isinstance(self.context_tokens, int)
                     or self.context_tokens < 1)):
            raise ValueError("endpoint context_tokens must be a positive integer")

    @property
    def narrow(self) -> bool:
        return self.role in {"narrow", "small", "2b", "3b", "a3b", "small_reasoner"}

    @property
    def frontier(self) -> bool:
        return self.role in {"frontier", "large", "frontier_model"}

    @property
    def moe(self) -> bool:
        return self.role in {"moe", "mixture", "default"}

    def supports(self, step_type: StepType) -> bool:
        return not self.allowed_step_types or step_type in self.allowed_step_types

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "role": self.role,
            "configured_url": bool(self.url),
            "api_key_env": self.api_key_env,
            "inherit_api_key": self.inherit_api_key,
            "allow_empty_key": self.allow_empty_key,
            "context_tokens": self.context_tokens,
            "allowed_step_types": sorted(item.value for item in self.allowed_step_types),
            "adapter": self.adapter.as_dict(),
        }


@dataclass(frozen=True)
class ModelRouterConfig:
    endpoints: tuple[EndpointConfig, ...]
    default_endpoint: str | None = None
    lease_actions: int = 2
    thresholds: RouterThresholds = field(default_factory=RouterThresholds)
    moe_only: bool = False
    stop_policy: str = "host"
    control_name: str = "adaptive"
    nonprogress_override: int = NONPROGRESS_OVERRIDE_DEFAULT
    handoff_token_budget: int = HANDOFF_TOKEN_BUDGET

    def __post_init__(self) -> None:
        endpoints = tuple(self.endpoints)
        if not endpoints:
            raise ValueError("model router requires at least one endpoint")
        names = [item.name for item in endpoints]
        if len(set(names)) != len(names):
            raise ValueError("model router endpoint names must be unique")
        if isinstance(self.lease_actions, bool) or not isinstance(self.lease_actions, int):
            raise ValueError("router lease must be an integer")
        if self.lease_actions < 2:
            raise ValueError("router lease must grant at least two meaningful actions")
        default = self.default_endpoint or next((item.name for item in endpoints if item.moe), names[0])
        if default not in names:
            raise ValueError(f"unknown model router default endpoint: {default}")
        default_config = next(item for item in endpoints if item.name == default)
        if not default_config.moe:
            raise ValueError("model router default endpoint must be a MoE endpoint")
        if not any(item.moe for item in endpoints):
            raise ValueError("model router requires a real MoE endpoint")
        if (isinstance(self.nonprogress_override, bool)
                or not isinstance(self.nonprogress_override, int)
                or self.nonprogress_override < 1):
            raise ValueError("router nonprogress_override must be a positive integer")
        if (isinstance(self.handoff_token_budget, bool)
                or not isinstance(self.handoff_token_budget, int)
                or self.handoff_token_budget < 1):
            raise ValueError("router handoff_token_budget must be a positive integer")
        if not isinstance(self.thresholds, RouterThresholds):
            raise TypeError("router thresholds must be RouterThresholds")
        if self.stop_policy != "host":
            raise ValueError("model router stop_policy must be host")
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "default_endpoint", default)

    @classmethod
    def from_env(cls, base_model: str, *, fallback_endpoint: str | None = None) -> "ModelRouterConfig":
        return config_from_env(base_model, fallback_endpoint=fallback_endpoint)

    @classmethod
    def moe_only_control(cls, model_id: str = "") -> "ModelRouterConfig":
        return moe_only_control_config(model_id)

    def endpoint(self, name: str) -> EndpointConfig:
        for item in self.endpoints:
            if item.name == name:
                return item
        raise KeyError(name)

    def default(self) -> EndpointConfig:
        return self.endpoint(str(self.default_endpoint))

    def moe(self) -> EndpointConfig:
        return next(item for item in self.endpoints if item.moe)

    def frontier(self) -> EndpointConfig | None:
        return next((item for item in self.endpoints if item.frontier), None)

    def narrow_for(self, step_type: StepType) -> EndpointConfig | None:
        return next((item for item in self.endpoints
                     if item.narrow and item.allowed_step_types and item.supports(step_type)), None)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "control": self.control_name,
            "default_endpoint": self.default_endpoint,
            "lease_actions": self.lease_actions,
            "thresholds": self.thresholds.as_dict(),
            "moe_only": self.moe_only,
            "stop_policy": self.stop_policy,
            "nonprogress_override": self.nonprogress_override,
            "handoff_token_budget": self.handoff_token_budget,
            "endpoints": [item.as_dict() for item in self.endpoints],
        }


def moe_only_control_config(model_id: str = "") -> ModelRouterConfig:
    """Build the MoE-only comparison arm with host-owned stopping."""
    if not str(model_id).strip():
        model_id = os.environ.get("DS_MODEL", "moe")
    return ModelRouterConfig(
        endpoints=(EndpointConfig("moe", str(model_id), role="moe"),),
        default_endpoint="moe", lease_actions=2, moe_only=True,
        stop_policy="host", control_name="moe_only",
    )


def _bool(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"})


def _adapter_from_mapping(value: object) -> EngineAdapter:
    if not isinstance(value, Mapping):
        return EngineAdapter()
    return EngineAdapter(
        chat_template=str(value.get("chat_template", "openai")),
        tool_format=str(value.get("tool_format", "openai")),
        thinking_field=str(value.get("thinking_field", "reasoning_effort")),
        thinking=value.get("thinking", "passthrough"),
        tokenizer=str(value.get("tokenizer", value.get(
            "tokenizer_name", "utf8-byte-conservative-v1"))),
    )


def _endpoint_from_mapping(value: Mapping[str, object], fallback_name: str = "moe") -> EndpointConfig:
    adapter_source = value.get("adapter", value)
    name = str(value.get("name", fallback_name))
    return EndpointConfig(
        name=name,
        model_id=str(value.get("model_id", value.get("model", name))),
        role=str(value.get("role", name)),
        url=(str(value["url"]) if value.get("url") is not None else
             str(value["endpoint"]) if value.get("endpoint") is not None else None),
        api_key_env=(str(value["api_key_env"]) if value.get("api_key_env") is not None else None),
        inherit_api_key=_bool(value.get("inherit_api_key", False)),
        allow_empty_key=_bool(value.get("allow_empty_key", False)),
        context_tokens=(int(value["context_tokens"])
                        if value.get("context_tokens") is not None else None),
        allowed_step_types=_normalize_allowed(value.get("allowed_step_types", value.get("step_types"))),
        adapter=_adapter_from_mapping(adapter_source),
    )


def _parse_endpoints(value: object) -> tuple[EndpointConfig, ...]:
    if isinstance(value, Mapping):
        value = value.get("endpoints", value)
    if isinstance(value, Mapping):
        return tuple(_endpoint_from_mapping(item, fallback_name=str(name))
                     for name, item in value.items() if isinstance(item, Mapping))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_endpoint_from_mapping(item, fallback_name=f"engine-{index}")
                     for index, item in enumerate(value) if isinstance(item, Mapping))
    return ()


def _thresholds_from_mapping(value: object) -> RouterThresholds:
    if not isinstance(value, Mapping):
        return RouterThresholds()
    aliases = {
        "same_counterexample": ("same_counterexample", "same_valid_counterexample"),
        "rejected_hypothesis": ("rejected_hypothesis", "rejected_hypothesis_repeats"),
        "nondiscriminating_reads": (
            "nondiscriminating_reads", "reads_without_discriminating_action"),
        "scope_mismatch": ("scope_mismatch",),
        "verification_conflict": ("verification_conflict",),
        "economic_exhaustion": ("economic_exhaustion",),
        "stable_actions_to_deescalate": (
            "stable_actions_to_deescalate", "deescalation_threshold"),
    }
    values = {}
    for name, candidates in aliases.items():
        found = next((candidate for candidate in candidates if candidate in value), None)
        if found is not None:
            values[name] = int(value[found])
    return RouterThresholds(**values)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def config_from_env(base_model: str, *, fallback_endpoint: str | None = None) -> ModelRouterConfig:
    """Parse only inline router configuration; no run data is copied or read."""
    raw_config = os.environ.get(CONFIG_ENV,
                                os.environ.get("STATEFRAME_ROUTER_CONFIG", "")).strip()
    if raw_config.lower() in {"moe_only", "moe-only", "control"}:
        config = moe_only_control_config(base_model)
        if fallback_endpoint:
            config = replace(config, endpoints=(replace(config.moe(), url=fallback_endpoint),))
        return config

    parsed: object = {}
    if raw_config:
        try:
            parsed = json.loads(raw_config)
        except json.JSONDecodeError as error:
            raise ValueError(f"{CONFIG_ENV} must be moe_only or JSON") from error
    endpoint_raw = os.environ.get(
        ENDPOINTS_ENV,
        os.environ.get("STATEFRAME_MODEL_ENDPOINTS",
                       os.environ.get("STATEFRAME_ROUTER_ENDPOINTS")))
    if endpoint_raw:
        try:
            parsed_endpoints = json.loads(endpoint_raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{ENDPOINTS_ENV} must be JSON") from error
    elif isinstance(parsed, Mapping):
        parsed_endpoints = parsed.get("endpoints", parsed)
    elif isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
        parsed_endpoints = parsed
    else:
        parsed_endpoints = ()
    endpoints = _parse_endpoints(parsed_endpoints)
    if not endpoints:
        endpoints = (EndpointConfig("moe", base_model or "moe", role="moe",
                                    url=fallback_endpoint),)
    options = parsed if isinstance(parsed, Mapping) else {}
    thresholds = _thresholds_from_mapping(
        options.get("thresholds", options.get("escalation_thresholds")))
    lease_default = options.get("lease_actions", options.get("lease", 2))
    lease = int(lease_default)
    lease = _env_int(LEASE_ENV, lease)
    if os.environ.get("STATEFRAME_MODEL_ROUTER_MOE_ONLY", "").strip():
        moe_only = _bool(os.environ["STATEFRAME_MODEL_ROUTER_MOE_ONLY"])
    else:
        moe_only = _bool(options.get("moe_only", False))
    default_name = options.get("default_endpoint", options.get("default"))
    return ModelRouterConfig(
        endpoints=endpoints,
        default_endpoint=str(default_name) if default_name else None,
        lease_actions=lease,
        thresholds=thresholds,
        moe_only=moe_only,
        stop_policy=str(options.get("stop_policy", "host")),
        control_name="moe_only" if moe_only else str(options.get("control", "adaptive")),
        nonprogress_override=int(options.get("nonprogress_override", NONPROGRESS_OVERRIDE_DEFAULT)),
        handoff_token_budget=int(options.get("handoff_token_budget", HANDOFF_TOKEN_BUDGET)),
    )


@dataclass(frozen=True)
class AssignmentLease:
    lease_id: str
    endpoint_name: str
    model_id: str
    assigned_step: object
    actions_granted: int
    actions_remaining: int
    reason: str
    why_escalated: tuple[str, ...] = ()

    def consume(self) -> "AssignmentLease":
        return replace(self, actions_remaining=max(0, self.actions_remaining - 1))

    @property
    def minimum_actions(self) -> int:
        return self.actions_granted

    @property
    def remaining_actions(self) -> int:
        return self.actions_remaining

    @property
    def remaining(self) -> int:
        return self.actions_remaining

    @property
    def min_actions(self) -> int:
        return self.actions_granted

    def as_dict(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "endpoint": self.endpoint_name,
            "model_id": self.model_id,
            "assigned_step": self.assigned_step,
            "actions_granted": self.actions_granted,
            "actions_remaining": self.actions_remaining,
            "meaningful_actions_remaining": self.actions_remaining,
            "reason": self.reason,
            "why_escalated": list(self.why_escalated),
        }


@dataclass(frozen=True)
class RouteDecision:
    endpoint: EndpointConfig
    step_type: StepType
    signals: EscalationSignals
    reason: str
    why_escalated: tuple[str, ...]
    lease: AssignmentLease

    @property
    def model_id(self) -> str:
        return self.endpoint.model_id

    @property
    def model(self) -> str:
        return self.endpoint.model_id

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint.name,
            "model_id": self.endpoint.model_id,
            "step_type": self.step_type.value,
            "signals": self.signals.as_dict(),
            "reason": self.reason,
            "why_escalated": list(self.why_escalated),
            "lease": self.lease.as_dict(),
        }


@dataclass(frozen=True)
class PreparedCall:
    body: dict[str, object]
    endpoint: str
    key: str
    decision: RouteDecision
    applied: bool


def _output_reservation(body: Mapping[str, object]) -> int:
    reservations = []
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        value = body.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            reservations.append(value)
    return max(reservations, default=0)


def _request_input_capacity(endpoint: EndpointConfig, body: Mapping[str, object],
                            host_capacity: int | None = None) -> int | None:
    capacity = endpoint.context_tokens
    if host_capacity is not None:
        capacity = min(host_capacity, capacity) if capacity is not None else host_capacity
    if capacity is None:
        return None
    return max(0, capacity - _output_reservation(body))


def _estimate_request(body: Mapping[str, object], *, endpoint: EndpointConfig | None = None) -> int:
    """Count with the selected endpoint adapter, never the global host estimator."""
    adapter = endpoint.adapter if endpoint is not None else EngineAdapter()
    return adapter.estimate_tokens(body)


def _completed_receipt(frame: Mapping[str, object]) -> tuple[str | None, bool]:
    events = _receipt_events(frame)
    for receipt_id, meaningful, _evidence_valid in reversed(events):
        if meaningful:
            return receipt_id, True
    if events:
        return events[-1][0], events[-1][1]
    return None, False


def _receipt_events(frame: Mapping[str, object]) -> tuple[tuple[str, bool, bool], ...]:
    projection = frame.get("model_router_host_projection")
    if isinstance(projection, Mapping):
        # Lease accounting consumes immutable completion identities.  The
        # evidence stream remains available for escalation, but stale evidence
        # must not erase a completed edit from the action lease.
        for stream_name in ("completed_receipts", "receipts"):
            receipts = projection.get(stream_name)
            if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
                continue
            result: list[tuple[str, bool, bool]] = []
            for row in receipts:
                if not isinstance(row, Mapping):
                    continue
                receipt_id = row.get("action_id") or _row_receipt_id(row)
                if receipt_id is not None:
                    result.append((
                        str(receipt_id),
                        bool(row.get("meaningful_action", _receipt_is_meaningful(row))),
                        row.get("evidence_valid") is not False,
                    ))
            if result:
                return tuple(result)
        receipt_id = projection.get("committed_action_receipt_id")
        if receipt_id not in (None, ""):
            return ((str(receipt_id),
                     projection.get("committed_action_meaningful") is True, True),)
    for key in ("committed_action_receipt_id", "action_receipt_id", "receipt_id"):
        receipt_id = frame.get(key)
        if receipt_id not in (None, ""):
            return ((str(receipt_id), frame.get("meaningful_action") is not False, True),)
    return ()


def _is_meaningful(frame: Mapping[str, object], step_type: StepType) -> bool:
    """Return true only for an identified, host-committed action receipt."""
    _, meaningful = _completed_receipt(frame)
    return meaningful


def validate_restart_compatibility(restart_enabled: bool, *, mode: str | None = None,
                                   replay: bool | None = None) -> None:
    """Reject modes whose mutable assignment state is not checkpointed/replayed."""
    effective_mode = model_router_mode() if mode is None else str(mode).strip().lower()
    if restart_enabled and effective_mode != "off":
        if replay is None:
            from kernel_stateframe_model_router_replay import replay_enabled
            replay = replay_enabled()
        if not isinstance(replay, bool):
            raise TypeError("router replay compatibility requires a boolean replay flag")
        if replay:
            return
        raise ValueError(
            "continuation restart cannot run with STATEFRAME_MODEL_ROUTER on or shadow; "
            "router assignments and action receipts are not replayable")


class ModelRouter:
    """Stateful per-step assignment and request adapter."""

    def __init__(self, mode: str | None = None, *, config: ModelRouterConfig | None = None,
                 journal: object | None = None, base_model: str = "",
                 replay_enabled: bool | None = None, resume: bool = False):
        self.mode = model_router_mode() if mode is None else str(mode).strip().lower()
        if self.mode not in VALID_MODES:
            raise ValueError("invalid StateFrame model router mode")
        self.config = config or config_from_env(base_model)
        self.base_model = str(base_model or "").strip()
        self.allowed_model_ids = frozenset(
            endpoint.model_id for endpoint in self.config.endpoints
        ) | ({self.base_model} if self.base_model else set())
        self.journal = journal
        if self.mode == "off":
            # The replay switch belongs to the router's enabled profile.  Do
            # not import or validate it on the byte-identical OFF path.
            requested_replay = False
        elif replay_enabled is None:
            from kernel_stateframe_model_router_replay import replay_enabled as replay_switch_enabled
            requested_replay = replay_switch_enabled()
        else:
            requested_replay = replay_enabled
        if not isinstance(requested_replay, bool):
            raise TypeError("router replay_enabled must be boolean")
        self.replay_enabled = self.mode != "off" and requested_replay
        if self.replay_enabled and (
                self.journal is None
                or not callable(getattr(self.journal, "emit", None))):
            from kernel_stateframe_model_router_replay import RouterReplayError
            raise RouterReplayError(
                "model-router replay requires a journal with durable emit")
        self._lease: AssignmentLease | None = None
        self._last_step: object = _MISSING
        self._last_frame: Mapping[str, object] | None = None
        self._last_decision: RouteDecision | None = None
        self._pending_reasons: tuple[str, ...] = ()
        self._stable_actions = 0
        self._nonprogress_streak = 0
        self._consumed_receipts: set[str] = set()
        self._route_count = 0
        self._replay_restored = False
        if self.replay_enabled:
            from kernel_stateframe_model_router_replay import (
                RouterReplayError, latest_state,
            )
            if resume and not callable(getattr(self.journal, "records", None)):
                raise RouterReplayError(
                    "model-router replay resume requires a journal with records")
            checkpoint_sequence = None
            if resume and self.journal is not None:
                checkpoint_sequence = max(
                    (record.get("sequence", 0) for record in self.journal.records()
                     if isinstance(record, Mapping)
                     and record.get("kind") == "continuation_worker_checkpoint"
                     and type(record.get("sequence")) is int),
                    default=None,
                )
            state = (latest_state(self.journal, through_sequence=checkpoint_sequence)
                     if self.journal is not None else None)
            if state is None and resume and checkpoint_sequence is None:
                state = latest_state(self.journal)
            if resume and state is None:
                raise RouterReplayError("durable model-router replay state is missing")
            if state is not None:
                self.restore_replay_state(state)
            elif not resume:
                self._persist_replay_state()

    @property
    def lease(self) -> AssignmentLease | None:
        return self._lease

    @property
    def last_decision(self) -> RouteDecision | None:
        return self._last_decision

    def replay_snapshot(self) -> dict[str, object]:
        from kernel_stateframe_model_router_replay import snapshot_state
        return snapshot_state(
            lease=self._lease,
            consumed_receipts=self._consumed_receipts,
            route_count=self._route_count,
            pending_reasons=self._pending_reasons,
            stable_actions=self._stable_actions,
            nonprogress_streak=self._nonprogress_streak,
        )

    def replay_projection(self) -> dict[str, object]:
        from kernel_stateframe_model_router_replay import public_projection
        return public_projection(self.replay_snapshot())

    def restore_replay_state(self, state: Mapping[str, object]) -> None:
        """Restore only the typed mutable state needed for another route."""
        from kernel_stateframe_model_router_replay import RouterReplayError, validate_state
        state = validate_state(state)
        raw_lease = state["lease"]
        if raw_lease is None:
            lease = None
        else:
            assert isinstance(raw_lease, Mapping)
            try:
                endpoint = self._endpoint_for_name(str(raw_lease["endpoint"]))
            except (KeyError, ValueError) as error:
                raise RouterReplayError("replayed router endpoint is unavailable") from error
            if (endpoint.model_id != raw_lease["model_id"]
                    or raw_lease["actions_granted"] != self.config.lease_actions):
                raise RouterReplayError("replayed router lease does not match configuration")
            lease = AssignmentLease(
                lease_id=str(raw_lease["lease_id"]),
                endpoint_name=endpoint.name,
                model_id=endpoint.model_id,
                assigned_step=raw_lease["assigned_step"],
                actions_granted=int(raw_lease["actions_granted"]),
                actions_remaining=int(raw_lease["actions_remaining"]),
                reason=str(raw_lease["reason"]),
                why_escalated=tuple(str(item) for item in raw_lease["why_escalated"]),
            )
        self._lease = lease
        self._consumed_receipts = set(state["consumed_receipts"])
        self._route_count = int(state["route_count"])
        self._pending_reasons = tuple(str(item) for item in state["pending_reasons"])
        self._stable_actions = int(state["stable_actions"])
        self._nonprogress_streak = int(state["nonprogress_streak"])
        # A resumed request must be routed afresh; the previous in-memory
        # frame is not a durable authority and must not trigger the same-step
        # memoization path.
        self._last_step = _MISSING
        self._last_frame = None
        self._last_decision = None
        self._replay_restored = True

    def _checkpoint_sequence(self) -> int | None:
        records = getattr(self.journal, "records", None)
        if not callable(records):
            return None
        return max(
            (record.get("sequence", 0) for record in records()
             if isinstance(record, Mapping)
             and record.get("kind") == "continuation_worker_checkpoint"
             and type(record.get("sequence")) is int),
            default=None,
        )

    def _persist_replay_state(self, *, assignment_id: str | None = None,
                              request_id: str | None = None) -> None:
        if self.replay_enabled:
            from kernel_stateframe_model_router_replay import REPLAY_STATE_EVENT
            self._emit(
                REPLAY_STATE_EVENT,
                state=self.replay_snapshot(),
                assignment_id=assignment_id,
                request_id=request_id,
                checkpoint_sequence=self._checkpoint_sequence(),
            )

    def _emit(self, kind: str, **fields: object) -> None:
        if (self.mode != "off" and self.journal is not None
                and callable(getattr(self.journal, "emit", None))):
            self.journal.emit(kind, **fields)

    def _execution_policy(self, runtime: object | None, endpoint: EndpointConfig):
        from kernel_model_policy import execution_policy_for

        if runtime is not None:
            setattr(runtime, "model_router_allowed_models", self.allowed_model_ids)
        actual_model = (
            (getattr(runtime, "model_name", None)
             or getattr(runtime, "model", None)
             or self.base_model)
            if self.mode == "shadow" and runtime is not None
            else endpoint.model_id
        )
        provider = None
        if self.mode != "shadow" and endpoint.url:
            provider = urlparse(endpoint.url).hostname
        return execution_policy_for(runtime, actual_model, provider=provider)

    def apply_current_execution_policy(self, runtime: object):
        endpoint = (self._endpoint_for_name(self._lease.endpoint_name)
                    if self._lease is not None else self.config.default())
        return self._execution_policy(runtime, endpoint)

    def _new_lease(self, endpoint: EndpointConfig, *, step: object, reason: str,
                   why: Sequence[str]) -> AssignmentLease:
        lease = AssignmentLease(
            lease_id=uuid.uuid4().hex,
            endpoint_name=endpoint.name,
            model_id=endpoint.model_id,
            assigned_step=step,
            actions_granted=self.config.lease_actions,
            actions_remaining=self.config.lease_actions,
            reason=reason,
            why_escalated=tuple(dict.fromkeys(str(item) for item in why)),
        )
        self._lease = lease
        return lease

    def _endpoint_for_name(self, name: str) -> EndpointConfig:
        return self.config.endpoint(name)

    def _target_for(self, step_type: StepType, reasons: tuple[str, ...]) -> tuple[EndpointConfig, str]:
        if self.config.moe_only:
            endpoint = self.config.moe()
            if not endpoint.supports(step_type):
                raise ValueError(
                    f"MoE-only endpoint {endpoint.name} does not support {step_type.value}")
            return endpoint, "moe_only_control"
        if reasons:
            frontier = self.config.frontier()
            if frontier is not None and frontier.supports(step_type):
                return frontier, "escalate:" + ",".join(reasons)
            default = self.config.default()
            if not default.supports(step_type):
                raise ValueError(
                    f"no eligible escalation endpoint supports {step_type.value}")
            return default, "escalation_unavailable:" + ",".join(reasons)
        narrow = self.config.narrow_for(step_type)
        if narrow is not None:
            return narrow, "narrow_proven:" + step_type.value
        default = self.config.default()
        if default.supports(step_type):
            return default, "default_moe_assignment"
        moe = next((item for item in self.config.endpoints
                    if item.moe and item.supports(step_type)), None)
        if moe is not None:
            return moe, "eligible_moe_assignment"
        raise ValueError(f"no eligible MoE endpoint supports {step_type.value}")

    def _project_frame(self, runtime: object | None,
                       frame: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(frame, Mapping):
            raise TypeError("model-router frame must be a mapping")
        projected = copy.deepcopy(dict(frame))
        existing = frame.get("model_router_host_projection")
        if runtime is None and isinstance(existing, Mapping):
            projected["model_router_host_projection"] = copy.deepcopy(dict(existing))
        elif runtime is not None or any(key in frame for key in (
                "committed_receipts", "latest_receipts", "test_receipts",
                "probe_receipts", "verification_receipts", "action_receipts")):
            projected["model_router_host_projection"] = host_signal_projection(runtime, frame)
        return projected

    def route(self, frame: Mapping[str, object], *, step: object = None,
              runtime: object | None = None) -> RouteDecision:
        frame = self._project_frame(runtime, frame)
        if step is None:
            self._route_count += 1
            effective_step = self._route_count
            same_step = False
        else:
            effective_step = step
            same_step = self._last_step is not _MISSING and step == self._last_step
        step_type = classify_step(frame)
        if (same_step and self._last_decision is not None
                and self._last_frame == frame):
            return self._last_decision
        signals = EscalationSignals.from_frame(frame)
        current_reasons = signals.reasons(self.config.thresholds)
        reasons = tuple(dict.fromkeys((*self._pending_reasons, *current_reasons)))
        receipt_events = _receipt_events(frame)
        new_events = tuple(event for event in receipt_events
                           if event[0] not in self._consumed_receipts)
        for receipt_id, _meaningful, _evidence_valid in new_events:
            self._consumed_receipts.add(receipt_id)
        meaningful_events = tuple(item for item in new_events if item[1])
        new_receipt = bool(new_events)
        meaningful = bool(meaningful_events)
        receipt_id = (meaningful_events[-1][0] if meaningful_events
                      else new_events[-1][0] if new_events else None)

        previous = self._lease
        if previous is not None and meaningful_events:
            previous = replace(
                previous,
                actions_remaining=max(0, previous.actions_remaining - len(meaningful_events)),
            )
            self._lease = previous
            self._nonprogress_streak = 0
        elif reasons and new_receipt:
            self._nonprogress_streak += 1
        elif new_receipt:
            self._nonprogress_streak = 0

        if (previous is not None and self._endpoint_for_name(previous.endpoint_name).frontier
                and not reasons and meaningful_events):
            # Stability is a property of each completed frontier action, not
            # of the instant at which a lease happens to expire.
            for _receipt_id, _meaningful, evidence_valid in meaningful_events:
                if evidence_valid:
                    self._stable_actions += 1
                else:
                    # A completion still consumes its lease, but an invalid
                    # meaningful action breaks the consecutive current-evidence
                    # streak and cannot justify de-escalation.
                    self._stable_actions = 0
        elif reasons:
            self._stable_actions = 0

        nonprogress_override = bool(
            reasons and self._nonprogress_streak >= self.config.nonprogress_override)
        scope_override = "scope_mismatch" in reasons

        endpoint: EndpointConfig
        reason: str
        why: tuple[str, ...]
        if previous is None:
            endpoint, reason = self._target_for(step_type, reasons)
            why = reasons
        elif (reasons and (not previous.actions_remaining or nonprogress_override or scope_override)):
            endpoint, reason = self._target_for(step_type, reasons)
            why = reasons or previous.why_escalated
        elif previous.actions_remaining and not self._endpoint_for_name(previous.endpoint_name).supports(step_type):
            # A narrow assignment is scoped to proven categories only.  A new
            # step outside that scope returns to the ordinary engine without
            # making the small engine own the wrong kind of work.
            endpoint, reason = self._target_for(step_type, ())
            why = previous.why_escalated
        elif previous.actions_remaining:
            endpoint = self._endpoint_for_name(previous.endpoint_name)
            reason = "lease_hold"
            why = tuple(dict.fromkeys((*previous.why_escalated, *reasons)))
            if not endpoint.frontier:
                self._stable_actions = 0
        else:
            current = self._endpoint_for_name(previous.endpoint_name)
            if not current.supports(step_type):
                endpoint, reason = self._target_for(step_type, reasons)
                why = reasons or previous.why_escalated
                self._stable_actions = 0
            elif current.frontier:
                if reasons:
                    endpoint, reason, why = (
                        current, "frontier_retain:" + ",".join(reasons),
                        tuple(dict.fromkeys((*previous.why_escalated, *reasons))))
                    self._stable_actions = 0
                else:
                    if (self._stable_actions >=
                            self.config.thresholds.stable_actions_to_deescalate
                            and previous.actions_remaining == 0):
                        endpoint, reason = self._target_for(step_type, ())
                        why = ()
                        self._stable_actions = 0
                    else:
                        endpoint, reason, why = (
                            current, "frontier_stability_lease", previous.why_escalated)
            else:
                endpoint, reason = self._target_for(step_type, reasons)
                why = reasons or previous.why_escalated

        # Renew only at an actual endpoint transition or after a completed
        # non-frontier lease.  Repeated preparation/read calls never mint a
        # fresh lease and therefore cannot hold an assignment forever.
        renew_completed_same_endpoint = bool(
            self._lease is not None and self._lease.actions_remaining == 0
            and endpoint.name == self._lease.endpoint_name
            and meaningful_events
            and not (endpoint.frontier and reason == "frontier_stability_lease"))
        if (previous is None or self._lease is None or
                self._lease.endpoint_name != endpoint.name or
                renew_completed_same_endpoint):
            lease = self._new_lease(endpoint, step=effective_step, reason=reason, why=why)
        else:
            lease = replace(
                self._lease,
                why_escalated=tuple(dict.fromkeys((*self._lease.why_escalated, *why))),
            )
            self._lease = lease

        if endpoint.frontier and (reason.startswith("escalate:") or
                                  (previous is not None and previous.endpoint_name != endpoint.name)):
            # The reason is carried by the frontier lease itself.  It should be
            # remembered for later inspection, but must not keep the frontier
            # sticky forever after the lease has completed.
            self._pending_reasons = ()
        elif current_reasons and not endpoint.frontier:
            self._pending_reasons = tuple(dict.fromkeys((*self._pending_reasons, *current_reasons)))
        elif endpoint.frontier and lease.why_escalated:
            self._pending_reasons = ()
        elif not current_reasons and not endpoint.frontier:
            self._pending_reasons = ()

        decision = RouteDecision(endpoint, step_type, signals, reason, lease.why_escalated, lease)
        execution_policy = self._execution_policy(runtime, endpoint)
        self._last_step = effective_step
        self._last_frame = copy.deepcopy(dict(frame))
        self._last_decision = decision
        assignment_fields = {
            "model_id": decision.model_id,
            "endpoint": decision.endpoint.name,
            "step": effective_step,
            "step_type": decision.step_type.value,
            "reason": decision.reason,
            "why_escalated": list(decision.why_escalated),
            "signals": decision.signals.as_dict(),
            "lease": decision.lease.as_dict(),
            "committed_receipt_id": receipt_id,
            "completed_action": bool(new_receipt and meaningful),
            "assignment_id": decision.lease.lease_id,
            "request_id": frame.get("request_id"),
            "execution_policy": execution_policy.to_dict(),
            "mode": self.mode,
        }
        if self.replay_enabled and callable(getattr(self.journal, "emit_batch", None)):
            from kernel_stateframe_model_router_replay import REPLAY_STATE_EVENT
            self.journal.emit_batch([
                (ASSIGNMENT_EVENT, assignment_fields),
                (REPLAY_STATE_EVENT, {
                    "state": self.replay_snapshot(),
                    "assignment_id": decision.lease.lease_id,
                    "request_id": frame.get("request_id"),
                    "checkpoint_sequence": self._checkpoint_sequence(),
                }),
            ])
        else:
            self._emit(ASSIGNMENT_EVENT, **assignment_fields)
            self._persist_replay_state(
                assignment_id=decision.lease.lease_id,
                request_id=frame.get("request_id"),
            )
        return decision

    assign = route
    choose = route
    route_step = route

    def bind(self, capture: object, frame: Mapping[str, object], *, step: object = None,
             runtime: object | None = None) -> RouteDecision:
        decision = self.route(frame, step=step, runtime=runtime)
        projected_frame = self._last_frame or self._project_frame(runtime, frame)
        setattr(capture, "model_router", self)
        setattr(capture, "model_router_decision", decision)
        setattr(capture, "model_router_frame", projected_frame)
        setattr(capture, "model_router_assignment_id", decision.lease.lease_id)
        setattr(capture, "model_router_parent_assignment_id", None)
        setattr(capture, "model_router_call_purpose", str(
            frame.get("call_purpose", frame.get("step_type", decision.step_type.value))))
        thinking_router = getattr(capture, "thinking_router", None)
        classification = getattr(capture, "thinking_router_classification", None)
        setattr(capture, "model_router_host_thinking_off", bool(
            thinking_router is not None and classification is not None and
            callable(getattr(thinking_router, "should_disable", None)) and
            thinking_router.should_disable(classification)))
        handoff = build_handoff(
            projected_frame, why_escalated=decision.why_escalated,
            max_tokens=self.config.handoff_token_budget)
        if self.replay_enabled:
            handoff["model_router_replay"] = self.replay_projection()
        setattr(capture, "model_router_handoff", handoff)
        return decision

    def _current_child_decision(self, capture: object) -> RouteDecision:
        child_step = getattr(capture, "model_router_child_step_type", None)
        if child_step is None:
            child_step = StepType.ORDINARY
        else:
            child_step = _step_type(child_step)
        child_frame = getattr(capture, "model_router_frame", None)
        child_frame = child_frame if isinstance(child_frame, Mapping) else {
            "step_type": child_step.value,
        }
        signals = EscalationSignals.from_frame(child_frame)
        reasons = signals.reasons(self.config.thresholds)
        parent_id = getattr(capture, "model_router_parent_assignment_id", None)
        parent = self._last_decision
        if (parent is not None and parent_id == parent.lease.lease_id
                and parent.endpoint.supports(child_step)):
            endpoint, reason = parent.endpoint, "child_inherited_assignment"
            why = parent.why_escalated
        else:
            endpoint, reason = self._target_for(child_step, reasons)
            why = reasons
        lease = AssignmentLease(
            lease_id=uuid.uuid4().hex,
            endpoint_name=endpoint.name,
            model_id=endpoint.model_id,
            assigned_step=getattr(capture, "step", None),
            actions_granted=self.config.lease_actions,
            actions_remaining=self.config.lease_actions,
            reason=reason,
            why_escalated=tuple(why),
        )
        return RouteDecision(endpoint, child_step, signals, reason, lease.why_escalated, lease)

    @staticmethod
    def _bounded_capacity(endpoint: EndpointConfig,
                          host_capacity: int | None = None) -> int:
        if endpoint.context_tokens is None:
            raise ValueError(
                f"selected endpoint {endpoint.name} requires explicit context_tokens "
                "when STATEFRAME_MODEL_ROUTER=on")
        if host_capacity is None:
            return endpoint.context_tokens
        return min(host_capacity, endpoint.context_tokens)

    def prepare_messages(self, capture: object, messages: Sequence[Mapping[str, object]],
                         tools: object, *, capacity_tokens: int | None = None) -> tuple[list[dict[str, object]], object]:
        decision = getattr(capture, "model_router_decision", None)
        if not isinstance(decision, RouteDecision):
            decision = self._current_child_decision(capture)
            setattr(capture, "model_router_decision", decision)
            setattr(capture, "model_router_assignment_id", decision.lease.lease_id)
            existing_purpose = getattr(capture, "model_router_call_purpose", None)
            setattr(capture, "model_router_call_purpose", str(
                existing_purpose or ("child:" + decision.step_type.value)))
        handoff = getattr(capture, "model_router_handoff", None)
        if not isinstance(handoff, Mapping):
            child_frame = getattr(capture, "model_router_frame", None)
            if not isinstance(child_frame, Mapping):
                child_frame = {"step_type": decision.step_type.value}
            handoff = build_handoff(
                child_frame,
                why_escalated=decision.why_escalated,
                max_tokens=self.config.handoff_token_budget)
            if self.replay_enabled:
                handoff["model_router_replay"] = self.replay_projection()
            setattr(capture, "model_router_handoff", handoff)
        if self.mode != "on":
            prepared = copy.deepcopy(list(messages))
            prepared_tools = copy.deepcopy(tools)
        else:
            selected_capacity = self._bounded_capacity(decision.endpoint, capacity_tokens)
            prepared, prepared_tools = decision.endpoint.adapter.prepare_messages(
                messages, tools, handoff=handoff,
                capacity_tokens=selected_capacity,
                host_thinking_off=bool(getattr(capture, "model_router_host_thinking_off", False)))
        setattr(capture, "model_router_prepared", self.mode == "on")
        return prepared, prepared_tools

    def validate_final_preparation(self, capture: object,
                                   messages: Sequence[Mapping[str, object]], tools: object,
                                   *, capacity_tokens: int | None = None) -> None:
        """Check the exact final body after state-tail/prefill/layout composition."""
        if self.mode != "on":
            return
        decision = getattr(capture, "model_router_decision", None)
        if not isinstance(decision, RouteDecision):
            return
        selected_capacity = self._bounded_capacity(decision.endpoint, capacity_tokens)
        estimate = _estimate_request(
            {"messages": list(messages), "tools": tools}, endpoint=decision.endpoint)
        if estimate > selected_capacity:
            raise RouterBudgetError(estimate, selected_capacity)

    def validate_final_request(self, capture: object, body: Mapping[str, object]) -> None:
        """Validate the exact adapted wire body, including child calls."""
        if self.mode != "on":
            return
        decision = getattr(capture, "model_router_decision", None)
        if not isinstance(decision, RouteDecision):
            return
        self._bounded_capacity(decision.endpoint)
        capacity = _request_input_capacity(decision.endpoint, body)
        if capacity is None:  # defensive: _bounded_capacity above is authoritative
            raise ValueError("selected endpoint input capacity is unavailable")
        estimate = _estimate_request(body, endpoint=decision.endpoint)
        setattr(capture, "model_router_input_capacity", capacity)
        setattr(capture, "model_router_request_estimate", estimate)
        if estimate > capacity:
            raise RouterBudgetError(estimate, capacity)

    def prepare_call(self, capture: object, body: Mapping[str, object], endpoint: str,
                     key: str) -> PreparedCall:
        decision = getattr(capture, "model_router_decision", None)
        handoff = getattr(capture, "model_router_handoff", None)
        if not isinstance(decision, RouteDecision):
            decision = self._current_child_decision(capture)
            setattr(capture, "model_router_decision", decision)
            setattr(capture, "model_router_assignment_id", decision.lease.lease_id)
            existing_purpose = getattr(capture, "model_router_call_purpose", None)
            setattr(capture, "model_router_call_purpose", str(
                existing_purpose or ("child:" + decision.step_type.value)))
        if not isinstance(handoff, Mapping):
            child_frame = getattr(capture, "model_router_frame", None)
            if not isinstance(child_frame, Mapping):
                child_frame = {"step_type": decision.step_type.value}
            handoff = build_handoff(
                child_frame,
                why_escalated=decision.why_escalated,
                max_tokens=self.config.handoff_token_budget)
            if self.replay_enabled:
                handoff["model_router_replay"] = self.replay_projection()
        applied = self.mode == "on"
        selected = decision.endpoint
        if applied:
            if getattr(capture, "model_router_prepared", False):
                adapted = selected.adapter.adapt_request_fields(
                    body, model_id=selected.model_id,
                    host_thinking_off=bool(getattr(
                        capture, "model_router_host_thinking_off", False)))
            else:
                adapted = selected.adapter.adapt_request(
                    body, handoff=handoff, model_id=selected.model_id)
            effective_endpoint = selected.url or endpoint
            effective_key = self._resolve_key(selected, endpoint, key)
            actual_model = selected.model_id
        else:
            adapted = copy.deepcopy(dict(body))
            effective_endpoint = endpoint
            effective_key = key
            manifest = getattr(self.journal, "manifest", {})
            manifest_model = manifest.get("model") if isinstance(manifest, Mapping) else None
            actual_model = str(adapted.get("model") or manifest_model or selected.model_id)
        setattr(capture, "model_router_effective_endpoint", effective_endpoint)
        setattr(capture, "model_router_effective_model_id", actual_model)
        setattr(capture, "model_router_selected_endpoint", selected.name)
        setattr(capture, "model_router_applied", applied)
        self._emit(
            CALL_EVENT,
            request_id=getattr(capture, "request_id", None),
            step=getattr(capture, "step", None),
            model_id=actual_model,
            selected_model_id=selected.model_id,
            endpoint=selected.name,
            configured_endpoint=selected.url,
            effective_endpoint=effective_endpoint,
            effective_model_id=actual_model,
            parent_assignment_id=getattr(capture, "model_router_parent_assignment_id", None),
            call_purpose=getattr(capture, "model_router_call_purpose", None),
            reason=decision.reason,
            why_escalated=list(decision.why_escalated),
            lease=decision.lease.as_dict(),
            adapter=selected.adapter.as_dict(),
            applied=applied,
            handoff=_public_value(handoff),
        )
        return PreparedCall(adapted, effective_endpoint, effective_key, decision, applied)

    def _resolve_key(self, selected: EndpointConfig, incoming_endpoint: str,
                     incoming_key: str) -> str:
        if selected.api_key_env:
            if selected.api_key_env not in os.environ:
                if selected.allow_empty_key:
                    return ""
                raise ValueError(
                    f"credential {selected.api_key_env} for endpoint {selected.name} is not set")
            value = os.environ[selected.api_key_env]
            if not value and not selected.allow_empty_key:
                raise ValueError(
                    f"credential {selected.api_key_env} for endpoint {selected.name} is empty")
            return value
        effective_endpoint = selected.url or incoming_endpoint
        if effective_endpoint != incoming_endpoint:
            if selected.inherit_api_key:
                return incoming_key
            if selected.allow_empty_key:
                return ""
            raise ValueError(
                f"endpoint {selected.name} changes transport without an explicit credential")
        return incoming_key

    prepare_request = prepare_call


__all__ = [
    "ASSIGNMENT_EVENT", "CALL_EVENT", "AssignmentLease", "EndpointConfig",
    "EngineAdapter", "EscalationSignals", "ModelRouter", "ModelRouterConfig",
    "ADAPTER_FIELD", "HANDOFF_FIELD", "PreparedCall", "RouteDecision", "RouterThresholds",
    "RouterBudgetError", "SCHEMA", "StepType", "TOOLS_FIELD",
    "build_handoff", "classify_step", "config_from_env", "host_signal_projection",
    "model_router_mode", "validate_restart_compatibility",
    "moe_only_control_config",
]
