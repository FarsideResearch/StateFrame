"""Strict, offline-safe primitives for opt-in StateFrame session resume.

The live runtime owns effect authorization.  This module owns durable
checkpoint validation, conversation sanitisation, typed reducer replay, and the
owner restore manifest.  A missing owner serializer is recorded explicitly as
non-restorable; it is never replaced with a fresh default without evidence.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import time
import uuid
from collections.abc import Mapping, Sequence

from kernel_continuation_state import ContinuationError, ContinuationState, transition
from kernel_measurement import digest

SCHEMA = "stateframe-true-resume/1"
CHECKPOINT_SCHEMA = "continuation-worker-checkpoint/4"
OWNER_NAMES = (
    "continuation_state", "obligations", "requirements", "coordinated_ledger",
    "findings", "environment", "green_lock", "stall_counters", "model_policy",
)


class ResumeError(RuntimeError):
    """A session cannot be resumed without risking a false continuation."""


def _json_value(value: object) -> object | None:
    if value is None:
        return None
    for _ in range(3):
        for method_name in ("to_dict", "snapshot", "document", "checkpoint_snapshot"):
            method = getattr(value, method_name, None)
            if callable(method):
                try:
                    value = method()
                except (AttributeError, TypeError, ValueError):
                    return None
                break
        else:
            break
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _message_copy(message: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(message, Mapping):
        raise ResumeError("conversation contains a non-object message")
    if message.get("role") not in {"system", "user", "assistant", "tool"}:
        raise ResumeError("conversation contains an invalid message role")
    return copy.deepcopy(dict(message))


def sanitize_conversation(messages: object) -> list[dict[str, object]]:
    """Preserve committed messages while removing orphaned tool exchanges."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise ResumeError("committed conversation must be a message sequence")
    result: list[dict[str, object]] = []
    pending: dict[str, int] = {}

    def close_unanswered() -> None:
        if not pending:
            return
        for index in sorted(set(pending.values())):
            assistant = result[index]
            calls = assistant.get("tool_calls")
            if not isinstance(calls, list):
                continue
            kept = [call for call in calls
                    if isinstance(call, Mapping) and call.get("id") not in pending]
            if kept:
                assistant["tool_calls"] = kept
            else:
                assistant.pop("tool_calls", None)
        pending.clear()

    for raw in messages:
        message = _message_copy(raw) if isinstance(raw, Mapping) else None
        if message is None:
            raise ResumeError("conversation contains a non-object message")
        role = message["role"]
        if role == "assistant":
            calls = message.get("tool_calls")
            if calls is not None:
                if not isinstance(calls, list):
                    raise ResumeError("assistant tool_calls must be a list")
                valid_calls: list[dict[str, object]] = []
                for call in calls:
                    if not isinstance(call, Mapping) or not isinstance(call.get("id"), str):
                        raise ResumeError("assistant tool call has no stable id")
                    call_id = str(call["id"])
                    if call_id in pending:
                        raise ResumeError("conversation reuses a pending tool call id")
                    valid_calls.append(copy.deepcopy(dict(call)))
                message["tool_calls"] = valid_calls
                result.append(message)
                assistant_index = len(result) - 1
                for call in valid_calls:
                    pending[str(call["id"])] = assistant_index
                continue
            close_unanswered()
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                continue
            pending.pop(call_id, None)
            result.append(message)
            continue
        else:
            close_unanswered()
        result.append(message)
    close_unanswered()
    return result


def conversation_hash(messages: Sequence[Mapping[str, object]]) -> str:
    return digest(sanitize_conversation(messages))


def checkpoint_conversation(messages: object) -> dict[str, object]:
    clean = sanitize_conversation(messages)
    return {
        "schema": SCHEMA,
        "messages": clean,
        "messages_hash": digest(clean),
        "prompt_cache_prefix": copy.deepcopy(clean[:2]),
        "prompt_cache_prefix_hash": digest(clean[:2]),
    }


def restore_conversation(checkpoint: Mapping[str, object]) -> list[dict[str, object]]:
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != SCHEMA:
        raise ResumeError("missing true-resume conversation checkpoint")
    messages = sanitize_conversation(checkpoint.get("messages"))
    if checkpoint.get("messages_hash") != digest(messages):
        raise ResumeError("conversation checkpoint hash mismatch")
    prefix = checkpoint.get("prompt_cache_prefix")
    if prefix is not None:
        clean_prefix = sanitize_conversation(prefix)
        if (clean_prefix != messages[:len(clean_prefix)]
                or checkpoint.get("prompt_cache_prefix_hash") != digest(clean_prefix)):
            raise ResumeError("prompt-cache prefix checkpoint mismatch")
    return messages


def latest_checkpoint(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    checkpoints = [record for record in records
                   if record.get("kind") == "continuation_worker_checkpoint"]
    if not checkpoints:
        raise ResumeError("no committed worker checkpoint")
    event = checkpoints[-1]
    state = event.get("checkpoint")
    if not isinstance(state, Mapping) or event.get("checkpoint_hash") != digest(state):
        raise ResumeError("worker checkpoint integrity failure")
    if state.get("schema") not in {
            CHECKPOINT_SCHEMA, "continuation-worker-checkpoint/3",
            "continuation-worker-checkpoint/2", "continuation-worker-checkpoint/1"}:
        raise ResumeError("unsupported worker checkpoint schema")
    return state


def require_snapshot(records: Sequence[Mapping[str, object]], checkpoint: Mapping[str, object]) -> Mapping[str, object]:
    runtime = checkpoint.get("runtime")
    snapshot_id = runtime.get("snapshot_id") if isinstance(runtime, Mapping) else None
    if not isinstance(snapshot_id, str) or not snapshot_id:
        snapshot_id = checkpoint.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ResumeError("committed checkpoint has no snapshot identity")
    snapshots = [record for record in records
                 if record.get("kind") == "snapshot_committed"
                 and record.get("snapshot_id") == snapshot_id]
    if not snapshots or any(digest(record.get("frame")) != snapshot_id for record in snapshots[-1:]):
        raise ResumeError("committed snapshot is missing or corrupted")
    return snapshots[-1]


def replay_continuation_state(records: Sequence[Mapping[str, object]]) -> ContinuationState:
    state = ContinuationState()
    for event in records:
        if event.get("kind") != "continuation_transition":
            continue
        if (event.get("previous_revision") != state.revision
                or event.get("before_hash") != digest(state.document())):
            raise ResumeError("typed continuation state replay diverged")
        operation, data = event.get("operation"), event.get("data")
        if not isinstance(operation, str) or not isinstance(data, Mapping):
            raise ResumeError("typed continuation transition is malformed")
        try:
            staged = transition(state, operation, copy.deepcopy(dict(data)))
        except (ContinuationError, TypeError, ValueError) as error:
            raise ResumeError("typed continuation reducer rejected journal replay") from error
        if event.get("after_hash") != digest(staged.document()):
            raise ResumeError("typed continuation resulting state hash mismatch")
        state = staged
    return state


_OWNER_SOURCES: dict[str, tuple[str, ...]] = {
    "obligations": ("obligation_graph", "obgraph", "obligation_ledger"),
    "requirements": ("requirements_ledger", "requirements_admissions"),
    "coordinated_ledger": ("coordinated_change_ledger",),
    "findings": ("history_findings", "host_closure_payload", "_host_facts"),
    "environment": ("environment_state",),
    "green_lock": ("green_lock",),
    "stall_counters": ("_empirical_stall_counter", "same_streak", "_consecutive_refusals"),
    "model_policy": ("model_policy",),
}
_GREEN_LOCK_FIELDS = (
    "mode", "_starting_binding", "_starting_revision", "_starting_product_files",
    "_starting_product_sources", "_green", "_green_revision", "_product_edit_applied",
    "_refused_targets", "_overridden_targets", "_first_action_refused",
    "_first_action_overridden", "_refusal_count", "_consecutive_refusals",
    "_refused_action_kind", "_override_action_kind", "_host_refusal_count",
    "_post_green_refusal_count", "_post_green_refusal_state_version",
    "_repair_blocks_done_non_progress", "_terminal_view_fingerprint",
    "_terminal_view_repeats", "_terminal_fallback_revision",
)


def _green_lock_snapshot(lock: object) -> dict[str, object] | None:
    if lock is None or not hasattr(lock, "_green") or not hasattr(lock, "mode"):
        return None
    state: dict[str, object] = {}
    for name in _GREEN_LOCK_FIELDS:
        value = getattr(lock, name, None)
        if isinstance(value, bytes):
            value = {"encoding": "hex", "value": value.hex()}
        elif isinstance(value, (set, frozenset, tuple)):
            value = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        state[name] = copy.deepcopy(value)
    return {"schema": "stateframe-green-lock/1", "state": state}


def _owner_snapshot(name: str, value: object) -> object | None:
    if name == "green_lock":
        return _json_value(_green_lock_snapshot(value))
    return _json_value(value)


def _owner_feature_enabled(runtime: object, name: str) -> bool:
    flags = {
        "obligations": ("stateframe_host_closure_mode",),
        "requirements": ("stateframe_requirements_ledger_mode", "stateframe_requirement_alignment_mode"),
        "coordinated_ledger": ("stateframe_coordinated_change_mode", "stateframe_coordination_guidance_mode"),
        "findings": ("stateframe_history_findings_mode", "stateframe_settled_findings_mode", "stateframe_host_closure_mode", "stateframe_host_facts_mode"),
        "environment": ("stateframe_environment_state_mode",),
        "green_lock": ("green_lock_mode",),
        "model_policy": ("model_policy",),
    }.get(name, ())
    return any(str(getattr(runtime, flag, "off")).strip().lower()
               not in {"", "0", "false", "off", "none"} for flag in flags)


def _restore_green_lock(lock: object, expected: Mapping[str, object]) -> None:
    if expected.get("schema") != "stateframe-green-lock/1":
        raise ResumeError("green-lock owner snapshot has an unsupported schema")
    state = expected.get("state")
    if not isinstance(state, Mapping) or set(state) != set(_GREEN_LOCK_FIELDS):
        raise ResumeError("green-lock owner snapshot is incomplete")
    for name in _GREEN_LOCK_FIELDS:
        value = state[name]
        if (isinstance(value, Mapping) and value.get("encoding") == "hex"
                and isinstance(value.get("value"), str)):
            try:
                value = bytes.fromhex(value["value"])
            except ValueError as error:
                raise ResumeError("green-lock fingerprint is malformed") from error
        if name in {"_refused_targets", "_overridden_targets"}:
            value = set(value) if isinstance(value, list) else value
        elif name in {"_starting_product_files", "_starting_product_sources"}:
            value = dict(value) if isinstance(value, Mapping) else value
        setattr(lock, name, copy.deepcopy(value))


def _restore_coordinated_ledger(runtime: object, expected: Mapping[str, object]) -> object:
    from kernel_coordinated_change_ledger import CoordinatedChangeLedger
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    ticket = next((row.get(key) for key in
                   ("_bridge_ordinary_brief", "public_brief", "ticket_text", "brief", "msg")
                   if isinstance(row, Mapping) and isinstance(row.get(key), str)
                   and row.get(key).strip()), "")
    root = getattr(owner, "root", None)
    if not ticket or root is None:
        raise ResumeError("coordinated ledger owner has no reconstruction identity")
    requirements = expected.get("requirements_ledger")
    ledger = CoordinatedChangeLedger.from_ticket(
        root, ticket,
        requirements_ledger=requirements if isinstance(requirements, Mapping) else None,
        journal=getattr(runtime, "journal", None),
    )
    if _json_value(ledger) != expected:
        raise ResumeError("coordinated ledger restoration postcondition mismatch")
    return ledger


def capture_owner_restore(runtime: object) -> dict[str, object]:
    owners: dict[str, object] = {
        "continuation_state": {"status": "replayed", "source": "continuation_transition.reducer"},
    }
    for name, candidates in _OWNER_SOURCES.items():
        attached = [candidate for candidate in candidates if hasattr(runtime, candidate)]
        if not attached:
            owners[name] = {"status": "absent", "reason": "owner is not enabled in this runtime profile"}
            continue
        saw_empty = False
        for selected in attached:
            raw = getattr(runtime, selected)
            if raw is None:
                saw_empty = True
                continue
            value = _owner_snapshot(name, raw)
            if value is not None:
                owners[name] = {"status": "restored", "source": selected, "state": value}
                break
        else:
            enabled = _owner_feature_enabled(runtime, name)
            owners[name] = {
                "status": "non_restorable" if enabled and not saw_empty else "absent",
                "source": attached[0],
                "reason": ("owner has no safe durable serializer"
                           if enabled and not saw_empty else
                           "owner is not enabled in this runtime profile"),
            }
    return {"schema": SCHEMA, "owners": owners}


def validate_owner_restore(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or value.get("schema") != SCHEMA:
        raise ResumeError("missing owner restore manifest")
    owners = value.get("owners")
    if not isinstance(owners, Mapping) or set(owners) != set(OWNER_NAMES):
        raise ResumeError("owner restore manifest is incomplete")
    result: dict[str, object] = {}
    for name in OWNER_NAMES:
        row = owners[name]
        if not isinstance(row, Mapping) or row.get("status") not in {"replayed", "restored", "absent", "non_restorable"}:
            raise ResumeError(f"owner {name} has no restore status")
        if row["status"] in {"absent", "non_restorable"} and not isinstance(row.get("reason"), str):
            raise ResumeError(f"owner {name} lacks a restore reason")
        if row["status"] == "restored" and "state" not in row:
            raise ResumeError(f"owner {name} lacks restored state")
        result[name] = copy.deepcopy(dict(row))
    return {"schema": SCHEMA, "owners": result}


def restore_owner_state(runtime: object, manifest: Mapping[str, object]) -> None:
    owners = manifest.get("owners")
    if not isinstance(owners, Mapping):
        raise ResumeError("owner restore manifest is malformed")
    for name in OWNER_NAMES:
        row = owners[name]
        if not isinstance(row, Mapping):
            raise ResumeError(f"owner {name} has no restore row")
        status = row.get("status")
        if status in {"replayed", "absent"}:
            continue
        if status == "non_restorable":
            raise ResumeError(f"owner {name} is enabled but not restorable")
        if status != "restored":
            raise ResumeError(f"owner {name} has an invalid restore status")
        source = row.get("source")
        if not isinstance(source, str) or not hasattr(runtime, source):
            raise ResumeError(f"restored owner {name} has no live source")
        expected = row.get("state")
        current = getattr(runtime, source)
        if name == "green_lock":
            if not isinstance(expected, Mapping):
                raise ResumeError("green-lock owner snapshot is malformed")
            _restore_green_lock(current, expected)
            if _owner_snapshot(name, current) != expected:
                raise ResumeError("green-lock restoration postcondition mismatch")
            continue
        if name == "coordinated_ledger" and current is None:
            if not isinstance(expected, Mapping):
                raise ResumeError("coordinated ledger owner snapshot is malformed")
            setattr(runtime, source, _restore_coordinated_ledger(runtime, expected))
            continue
        if _json_value(current) == expected:
            continue
        if current is None or isinstance(current, (Mapping, list, tuple, set, frozenset, str, int, float, bool)):
            setattr(runtime, source, copy.deepcopy(expected))
            continue
        raise ResumeError(f"restored owner {name} differs from its checkpoint")


def workspace_revision_matches(expected: Mapping[str, object], actual: Mapping[str, object]) -> bool:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return False
    keys = set(expected)
    return bool(keys) and keys <= set(actual) and all(expected[key] == actual[key] for key in keys)


def user_request_event(journal: object, text: str, *, request_id: str | None = None,
                       source_message_id: str | None = None,
                       actor_kind: str = "user",
                       authority_scope: str = "product_contract",
                       received_contract_revision: int | None = None) -> dict:
    if not isinstance(text, str) or not text:
        raise ResumeError("follow-up user message must be nonempty text")
    request_id = request_id or uuid.uuid4().hex
    if not isinstance(request_id, str) or not request_id:
        raise ResumeError("follow-up request_id must be nonempty")
    return journal.emit(
        "user_request", user_request_schema="stateframe-user-request/1",
        request_id=request_id, text=text, verbatim_text=text,
        source_message_id=source_message_id or request_id, actor_kind=actor_kind,
        authority_scope=authority_scope, received_contract_revision=received_contract_revision,
        recorded_at_ns=time.time_ns())


def append_follow_up(messages: object, text: str) -> list[dict[str, object]]:
    restored = sanitize_conversation(messages)
    if not isinstance(text, str) or not text:
        raise ResumeError("follow-up user message must be nonempty text")
    return [*restored, {"role": "user", "content": text}]


def restore_tail_follow_up(messages: object, records: Sequence[Mapping[str, object]],
                           checkpoint_sequence: int, requested: str | None) -> tuple[list[dict[str, object]], Mapping[str, object] | None]:
    pending = [record for record in records
               if record.get("kind") == "user_request"
               and type(record.get("sequence")) is int
               and record["sequence"] > checkpoint_sequence]
    pending.sort(key=lambda record: int(record["sequence"]))
    if not pending:
        return sanitize_conversation(messages), None
    restored = sanitize_conversation(messages)
    for event in pending:
        if (event.get("user_request_schema") != "stateframe-user-request/1"
                or not isinstance(event.get("request_id"), str)
                or type(event.get("recorded_at_ns")) is not int
                or event["recorded_at_ns"] <= 0
                or not isinstance(event.get("text"), str)
                or event.get("verbatim_text") != event.get("text")
                or not event["text"]):
            raise ResumeError("uncheckpointed user request is malformed")
        if requested is not None and requested != event["text"]:
            raise ResumeError("follow-up conflicts with an already committed user request")
        restored = append_follow_up(restored, event["text"])
    return restored, pending[-1]


def restore_tail_follow_ups(messages: object, records: Sequence[Mapping[str, object]],
                            checkpoint_sequence: int, requested: str | None = None) -> tuple[list[dict[str, object]], tuple[Mapping[str, object], ...]]:
    restored, _last = restore_tail_follow_up(messages, records, checkpoint_sequence, requested)
    pending = tuple(record for record in records
                    if record.get("kind") == "user_request"
                    and type(record.get("sequence")) is int
                    and record["sequence"] > checkpoint_sequence)
    return restored, pending
