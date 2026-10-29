"""v17 revision-aware delivery for automatic test results.

Automatic verification is host evidence, not provider history.  This guard
keeps a newer result in the final request and labels an older failing result
when a newer GREEN supersedes it.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence

from kernel_tool_reply import Outcome, tool_reply

MODE_ENV = "STATEFRAME_TOKENLAB_HOST_RESULT_GUARD"
EVENT = "tokenlab_v17_host_result_guard"
SCHEMA = "stateframe-v17-host-result/1"


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_tokenlab_host_result_guard_mode", None)
    return configured == "on" if configured is not None else mode() == "on"


def _state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "_tokenlab_host_result_guard", None)
    if not isinstance(state, dict):
        state = {"deliveries": {}, "latest": None, "pending": False, "events": set()}
        try:
            setattr(runtime, "_tokenlab_host_result_guard", state)
        except (AttributeError, TypeError):
            pass
    for key, default in (("deliveries", {}), ("events", set())):
        if not isinstance(state.get(key), type(default)):
            state[key] = default
    return state


def _revision(value: object) -> str:
    if value not in (None, "") and not isinstance(value, Mapping):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("revision", "product_revision", "artifact_revision", "artifact", "current_revision"):
            if value.get(key) not in (None, ""):
                return str(value[key])
        for key in ("after_binding", "binding", "current_binding"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                result = _revision(nested)
                if result != "UNKNOWN":
                    return result
    return "UNKNOWN"


def _green(row: Mapping[str, object]) -> bool:
    return row.get("exit_code") == 0 or str(
        row.get("result_class", row.get("status", ""))).upper() in {
            "GREEN", "PASSED", "PASS", "SUCCESS",
        }


def _emit(runtime: object, action: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(EVENT, producer="kernel_tokenlab_host_result_guard", event_schema=SCHEMA,
             action=action, dispatch=False, **fields)


def remember_delivery(runtime: object, delivery: Mapping[str, object], *,
                      receipt_index: int | None = None) -> None:
    """Retain the latest automatic result until final request composition proves delivery."""
    if not enabled(runtime):
        return
    row = copy.deepcopy(dict(delivery))
    row["revision"] = _revision(row.get("revision", row))
    if receipt_index is not None:
        row["receipt_index"] = receipt_index
    state = _state(runtime)
    base_call_id = row.get("tool_call_id")
    call_id = f"{base_call_id}:auto_verify" if base_call_id else None
    if call_id:
        state["deliveries"][str(call_id)] = row
    state["latest"] = row
    state["pending"] = True


def _newer_green(runtime: object, row: Mapping[str, object]) -> tuple[str, int] | None:
    index = row.get("receipt_index")
    if not isinstance(index, int):
        return None
    receipts = getattr(runtime, "test_receipts", ())
    for current_index, receipt in enumerate(receipts):
        if current_index <= index or not isinstance(receipt, Mapping) or not _green(receipt):
            continue
        return _revision(receipt), current_index
    return None


def _label_stale(runtime: object, row: dict[str, object], sequence: list[dict]) -> int:
    newer = _newer_green(runtime, row)
    if newer is None or _green(row):
        return 0
    stale_revision = str(row.get("revision", "UNKNOWN"))
    green_revision, green_index = newer
    prefix = (f"[STALE TEST RESULT revision={stale_revision}; superseded by GREEN "
              f"revision={green_revision}]\n")
    changed = 0
    base_call_id = row.get("tool_call_id")
    call_id = f"{base_call_id}:auto_verify" if base_call_id else None
    for item in sequence:
        if (isinstance(item, dict) and item.get("role") == "tool"
                and item.get("tool_call_id") == call_id):
            content = str(item.get("content", ""))
            if not content.startswith(prefix):
                item["content"] = prefix + content
                changed += 1
    if not row.get("stale_labeled"):
        row["stale_labeled"] = True
        _emit(runtime, "label_stale", call_id=call_id, stale_revision=stale_revision,
              green_revision=green_revision, green_receipt_index=green_index)
    return changed


def _tool_call_names(sequence: Sequence[Mapping[str, object]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in sequence:
        if not isinstance(item, Mapping) or item.get("role") != "assistant":
            continue
        for call in item.get("tool_calls", ()):
            if not isinstance(call, Mapping) or not call.get("id"):
                continue
            function = call.get("function")
            if isinstance(function, Mapping):
                names[str(call["id"])] = str(function.get("name", ""))
    return names


def _failure_text(content: object) -> bool:
    text = str(content or "").lower()
    return any(marker in text for marker in (
        "traceback", "pytest exit 1", "tests failed", "assertionerror", "failed",
    ))


def _label_explicit_stale_replies(runtime: object, sequence: list[dict]) -> int:
    """Annotate retained explicit run_tests failures superseded by a newer GREEN."""
    receipts = [row for row in getattr(runtime, "test_receipts", ()) if isinstance(row, Mapping)]
    failing = [(index, row) for index, row in enumerate(receipts) if not _green(row)]
    if not failing:
        return 0
    names = _tool_call_names(sequence)
    changed = 0
    failure_cursor = 0
    for item in sequence:
        if not isinstance(item, dict) or item.get("role") != "tool":
            continue
        call_id = str(item.get("tool_call_id", ""))
        if not call_id or call_id.endswith(":auto_verify") or names.get(call_id) != "run_tests":
            continue
        if not _failure_text(item.get("content")) or failure_cursor >= len(failing):
            continue
        receipt_index, receipt = failing[failure_cursor]
        newer = _newer_green(runtime, {**receipt, "receipt_index": receipt_index})
        failure_cursor += 1
        if newer is None:
            continue
        stale_revision = _revision(receipt)
        green_revision, green_index = newer
        prefix = (f"[STALE TEST RESULT revision={stale_revision}; superseded by GREEN "
                  f"revision={green_revision}]\n")
        content = str(item.get("content", ""))
        if not content.startswith(prefix):
            item["content"] = prefix + content
            changed += 1
        _emit(runtime, "label_explicit_stale", call_id=call_id,
              stale_revision=stale_revision, green_revision=green_revision,
              stale_receipt_index=receipt_index, green_receipt_index=green_index)
    return changed


def _pair(row: Mapping[str, object]) -> tuple[dict, dict] | tuple[None, dict]:
    content = str(row.get("content", ""))
    call_id = row.get("tool_call_id")
    if not call_id:
        return None, {"role": "user", "content": content}
    auto_call_id = f"{call_id}:auto_verify"
    assistant = {"role": "assistant", "content": None, "tool_calls": [{
        "id": auto_call_id, "type": "function", "function": {
            "name": "run_tests", "arguments": json.dumps(
                {"target": row.get("target", "")}, separators=(",", ":"))}}]}
    return assistant, tool_reply(auto_call_id, "run_tests", Outcome.ACCEPTED, content)


def ensure_sequence(runtime: object, sequence: list[dict], *, final: bool = False) -> dict[str, int]:
    """Label superseded failures and ensure the latest result is represented once."""
    if not enabled(runtime):
        return {"stale_labeled": 0, "latest_inserted": 0}
    state = _state(runtime)
    stale_labeled = 0
    for row in state["deliveries"].values():
        if isinstance(row, dict):
            stale_labeled += _label_stale(runtime, row, sequence)
    stale_labeled += _label_explicit_stale_replies(runtime, sequence)
    latest = state.get("latest")
    inserted = 0
    if isinstance(latest, dict) and state.get("pending"):
        assistant, message = _pair(latest)
        if assistant is None:
            exists = any(item.get("role") == "user" and item.get("content") == message["content"]
                         for item in sequence if isinstance(item, Mapping))
            if not exists:
                sequence.append(message)
                inserted = 1
        else:
            call_id = message["tool_call_id"]
            existing = next((item for item in sequence
                             if isinstance(item, dict) and item.get("tool_call_id") == call_id), None)
            if existing is None:
                sequence.extend((copy.deepcopy(assistant), copy.deepcopy(message)))
                inserted = 1
            else:
                if existing.get("content") != message["content"]:
                    existing["content"] = message["content"]
                    inserted = 1
                if not any(
                        isinstance(item, Mapping)
                        and item.get("role") == "assistant"
                        and any(call.get("id") == call_id
                                for call in item.get("tool_calls", ())
                                if isinstance(call, Mapping))
                        for item in sequence):
                    position = sequence.index(existing)
                    sequence.insert(position, copy.deepcopy(assistant))
        if final:
            state["pending"] = False
            _emit(runtime, "ensure_latest", call_id=latest.get("tool_call_id"),
                  revision=latest.get("revision", "UNKNOWN"),
                  result_class=latest.get("result_class"), inserted=bool(inserted))
    return {"stale_labeled": stale_labeled, "latest_inserted": inserted}


__all__ = ["EVENT", "MODE_ENV", "SCHEMA", "enabled", "ensure_sequence", "mode",
           "remember_delivery"]
