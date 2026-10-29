"""Opt-in F4 request fitting with an explicit, loss-aware shed order.

The F4 renderer deliberately builds one required projection. This module provides
a presentation-only token fit and a separate opt-in serialized-byte fit. Neither
pass edits the durable frame or tool menu. With ``STATEFRAME_F4_FIT=off`` callers
receive the original message sequence without even copying it; the byte pass is
independently disabled by default.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_context import token_estimate


MODE_ENV = "STATEFRAME_F4_FIT"
FRAMEFIT_MODE_ENV = "STATEFRAME_FRAMEFIT"
FITTED_FRAME_RETENTION_ENV = "STATEFRAME_F4_KEEP_FITTED_FRAME"
FIT_SCHEMA = "stateframe-f4-fit/1"
SCHEMA = FIT_SCHEMA
FRAMEFIT_SCHEMA = "stateframe-framefit/1"
SHED_PHASES = (
    "older_real_exchanges",
    "verbose_notice_bodies",
    "lower_priority_state_rows",
)
_CONTINUITY_OMISSION_FIELDS = {
    "hypotheses": "hypotheses_omitted",
    "model_claims": "model_claims_omitted",
    "read_memory": "read_memory_omitted",
    "negative_work": "negative_work_omitted",
}

_STATE_HEADER = "[StateFrame]"
_HANDOFF_HEADER = "[Decision handoff]"
_SOURCE_PREFIXES = (
    "[Working source:", "[Current source:", "[Observed source:",
    "[Retained source:", "[Source content:", "CURRENT_SOURCE ",
)
_SOURCE_BLOCK_HEADER = re.compile(
    r"^\[(?:Current|Working) source:\s*([^\]\r\n]+)\]$",
    re.IGNORECASE,
)
_NOTICE_PREFIXES = (
    "GREEN LOCK /", "GREEN LOCK:", "[Named-layer coverage", "[Named notice]",
    "[Live verbose notice]", "[Notice]", "NOTICE:", "[Render notice]",
    "[False-completion coverage",  # r44 F4 item 4: now an F4 notice carrier
    "[Working sets | host-retained]", "<settled_state>",  # r44 F4 item 6
    "[Source page evicted]", "[Source page deferred]",
)
_OPEN_WORDS = frozenset({
    "OPEN", "BLOCKED", "FAILED", "FAIL", "NEEDS_WORK", "NEEDS-WORK", "VIOLATED",
})
_NEVER_DROP_NAMES = (
    "obligation", "counterexample", "blocker", "ticket", "handoff", "decision", "progress",
)
_NOTICE_KEEP_RE = re.compile(
    r"(?:open\s+counterexamples?|counterexamples?|open\s+obligations?|"
    r"obligations?|ticket|never\s+drop|must\s+remain|revisit|reopen|"
    r"retrieve|read\s*\()",
    re.IGNORECASE,
)


def mode(value: object = None) -> str:
    if value is None:
        raw = os.environ.get(MODE_ENV, "off")
    elif isinstance(value, Mapping):
        raw = value.get(MODE_ENV, "off")
    else:
        raw = value
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return raw.strip().lower()


def enabled(value: object = None) -> bool:
    return mode(value) == "on"


def framefit_mode(value: object = None) -> str:
    """Return the strict FRAMEFIT switch; it is independent of legacy F4 fitting."""
    if value is None:
        raw = os.environ.get(FRAMEFIT_MODE_ENV, "off")
    elif isinstance(value, Mapping):
        raw = value.get(FRAMEFIT_MODE_ENV, "off")
    else:
        raw = value
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{FRAMEFIT_MODE_ENV} must be off or on")
    return raw.strip().lower()


def framefit_enabled(value: object = None) -> bool:
    return framefit_mode(value) == "on"


def fitted_frame_retention_mode(value: object = None) -> str:
    """Return the strict TOKEN F1 switch; the legacy behavior remains the default."""
    if value is None:
        raw = os.environ.get(FITTED_FRAME_RETENTION_ENV, "off")
    elif isinstance(value, Mapping):
        raw = value.get(FITTED_FRAME_RETENTION_ENV, "off")
    else:
        raw = value
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{FITTED_FRAME_RETENTION_ENV} must be off or on")
    return raw.strip().lower()


def fitted_frame_retention_enabled(value: object = None) -> bool:
    return fitted_frame_retention_mode(value) == "on"


def render_deferred_source_receipt(route: Mapping[str, object]) -> str:
    """Render a bounded, exact read route for a deferred current source file."""
    path = route.get("path")
    revision = route.get("revision")
    start_line = route.get("start_line")
    end_line = route.get("end_line")
    if not isinstance(path, str):
        raise ValueError("deferred source route path must be a string")
    normalized_path = path.replace("\\", "/")
    parts = normalized_path.split("/")
    if (not normalized_path or normalized_path.startswith("/")
            or re.match(r"^[A-Za-z]:", normalized_path)
            or any(part in {"", ".", ".."} or part.casefold() == "backups"
                   for part in parts)
            or "\x00" in normalized_path or "\n" in normalized_path
            or "\r" in normalized_path):
        raise ValueError("deferred source route path is unsafe")
    if (not isinstance(revision, str)
            or re.fullmatch(r"[A-Za-z0-9:._-]{1,128}", revision) is None):
        raise ValueError("deferred source route revision is invalid")
    if (type(start_line) is not int or type(end_line) is not int
            or start_line < 1 or end_line < start_line):
        raise ValueError("deferred source route range is invalid")
    return (
        "[Source page deferred]\n"
        f"path={json.dumps(normalized_path, ensure_ascii=False)} "
        f"revision={revision} lines={start_line}-{end_line}\n"
        "Retrieve current bytes with: read("
        f"path={json.dumps(normalized_path, ensure_ascii=False)}, "
        f"start_line={start_line}, end_line={end_line})."
    )


def parse_deferred_source_receipt(value: object) -> dict[str, object] | None:
    """Parse only the exact FRAMEFIT receipt format and its matching read call."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    lines = value.splitlines()
    if len(lines) != 3 or lines[0] != "[Source page deferred]":
        return None
    match = re.fullmatch(
        r"path=(.+) revision=(\S+) lines=(\d+)-(\d+)", lines[1])
    route_match = re.fullmatch(
        r"Retrieve current bytes with: read\(path=(.+), start_line=(\d+), "
        r"end_line=(\d+)\)\.", lines[2])
    if match is None or route_match is None:
        return None
    revision = match.group(2)
    try:
        path = json.loads(match.group(1))
        route_path = json.loads(route_match.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    start_line, end_line = int(match.group(3)), int(match.group(4))
    if (not isinstance(path, str) or not isinstance(route_path, str) or route_path != path
            or int(route_match.group(2)) != start_line
            or int(route_match.group(3)) != end_line):
        return None
    try:
        canonical = render_deferred_source_receipt({
            "path": path, "revision": revision,
            "start_line": start_line, "end_line": end_line,
        })
    except ValueError:
        return None
    if canonical != value:
        return None
    return {"path": path, "revision": revision,
            "start_line": start_line, "end_line": end_line}


def is_valid_deferred_target_source_message(
        content: object, *, path: str, source: str, tools: object) -> bool:
    """Verify a deferred target receipt against current bytes and the admitted read tool."""
    if not isinstance(content, str):
        return False
    lines = content.splitlines()
    if not lines:
        return False
    header = _SOURCE_BLOCK_HEADER.fullmatch(lines[0])
    if header is None or header.group(1).strip().replace("\\", "/") != path:
        return False
    receipt = parse_deferred_source_receipt("\n".join(lines[1:]))
    if not isinstance(receipt, Mapping):
        return False
    line_count = len(source.splitlines())
    if line_count < 1:
        return False
    from kernel_request_source_window import source_revision
    from kernel_stateframe_m3 import _read_tool_admitted
    return bool(
        receipt.get("path") == path
        and receipt.get("revision") == source_revision(source)
        and receipt.get("start_line") == 1
        and receipt.get("end_line") == line_count
        and _read_tool_admitted(tools)
    )


@dataclass(frozen=True)
class _Exchange:
    start: int
    end: int
    call_ids: tuple[str, ...]
    real: bool


def _estimate(messages: Sequence[Mapping[str, object]], tools: object) -> int:
    return token_estimate({"messages": list(messages), "tools": tools})


def _message_text(message: object) -> str:
    return str(message.get("content") or "") if isinstance(message, Mapping) else ""


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _exchange_spans(messages: Sequence[Mapping[str, object]]) -> tuple[_Exchange, ...]:
    """Find only closed assistant/tool groups; a group is removed atomically."""
    result: list[_Exchange] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            index += 1
            continue
        raw_calls = message.get("tool_calls")
        if not _is_sequence(raw_calls) or not raw_calls:
            index += 1
            continue
        call_ids = tuple(
            str(call.get("id")) for call in raw_calls
            if isinstance(call, Mapping) and isinstance(call.get("id"), str)
            and str(call.get("id"))
        )
        if len(call_ids) != len(raw_calls) or len(set(call_ids)) != len(call_ids):
            index += 1
            continue
        cursor = index + 1
        results: list[Mapping[str, object]] = []
        while cursor < len(messages):
            candidate = messages[cursor]
            if not isinstance(candidate, Mapping) or candidate.get("role") != "tool":
                break
            results.append(candidate)
            cursor += 1
        result_ids = [str(item.get("tool_call_id")) for item in results]
        if (len(results) != len(call_ids) or set(result_ids) != set(call_ids)
                or any(not item.get("tool_call_id") for item in results)):
            index += 1
            continue
        real = any(":auto_verify" not in call_id for call_id in call_ids)
        result.append(_Exchange(index, cursor - 1, call_ids, real))
        index = cursor
    return tuple(result)


def _ledger_covered_call_ids(messages: Sequence[Mapping[str, object]]) -> set[str]:
    """Read only call IDs already represented by the typed F3 action ledger."""
    from kernel_stateframe_action_ledger import SCHEMA as ACTION_LEDGER_SCHEMA

    covered: set[str] = set()
    for message in messages:
        text = _message_text(message)
        if not text.startswith(_STATE_HEADER + "\n"):
            continue
        try:
            payload = json.loads(text.split("\n", 1)[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        state = payload.get("state") if isinstance(payload, Mapping) else None
        ledger = state.get("framefit_action_ledger") if isinstance(state, Mapping) else None
        if (not isinstance(ledger, Mapping)
                or ledger.get("schema") != ACTION_LEDGER_SCHEMA):
            continue
        rows = ledger.get("rows")
        if not _is_sequence(rows):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            call_ids = row.get("call_ids")
            if _is_sequence(call_ids):
                covered.update(
                    call_id for call_id in call_ids
                    if isinstance(call_id, str) and call_id
                )
    return covered


def _remove_covered_exchanges(
        messages: Sequence[Mapping[str, object]], covered_ids: set[str],
) -> list[dict]:
    """Remove typed exchanges, preserving uncovered calls in mixed provider batches."""
    if not covered_ids:
        return [dict(message) for message in messages]
    removed_indexes: set[int] = set()
    replacements: dict[int, dict] = {}
    for exchange in _exchange_spans(messages):
        covered = covered_ids.intersection(exchange.call_ids)
        if not covered:
            continue
        if covered == set(exchange.call_ids):
            removed_indexes.update(range(exchange.start, exchange.end + 1))
            continue
        assistant = dict(messages[exchange.start])
        calls = assistant.get("tool_calls")
        if not _is_sequence(calls):
            continue
        retained_calls = [
            copy.deepcopy(dict(call)) for call in calls
            if isinstance(call, Mapping) and str(call.get("id") or "") not in covered
        ]
        if not retained_calls:
            removed_indexes.update(range(exchange.start, exchange.end + 1))
            continue
        assistant["tool_calls"] = retained_calls
        replacements[exchange.start] = assistant
        for index in range(exchange.start + 1, exchange.end + 1):
            message = messages[index]
            if (message.get("role") == "tool"
                    and str(message.get("tool_call_id") or "") in covered):
                removed_indexes.add(index)

    if not removed_indexes and not replacements:
        return [dict(message) for message in messages]
    return [
        replacements.get(index, copy.deepcopy(dict(message)))
        for index, message in enumerate(messages)
        if index not in removed_indexes
    ]


def _remove_spans(messages: Sequence[dict], spans: Sequence[_Exchange]) -> list[dict]:
    removed = {index for span in spans for index in range(span.start, span.end + 1)}
    return [message for index, message in enumerate(messages) if index not in removed]


def _empty_phase() -> dict[str, object]:
    return {"count": 0, "tokens": 0, "items": []}


def _record_phase(phase: dict[str, object], *, item: object,
                  tokens: int, count: int = 1) -> None:
    phase["count"] = int(phase.get("count", 0)) + count
    phase["tokens"] = int(phase.get("tokens", 0)) + max(0, int(tokens))
    items = phase.setdefault("items", [])
    if isinstance(items, list):
        items.append(item)


def _manifest(*, initial: int, final: int, ceiling: int, fits: bool,
              changed: bool, enabled_value: bool,
              protect_latest_batch: bool = False) -> dict[str, object]:
    never_drop = [
        "ticket", "open_obligations", "counterexamples", "decision_handoff",
        "progress_ledger",
        "latest_native_batch" if protect_latest_batch else "last_2_real_exchanges",
        "tool_call_result_pairing",
    ]
    return {
        "fit_schema": FIT_SCHEMA,
        "enabled": enabled_value,
        "phase_order": list(SHED_PHASES),
        "initial_tokens": initial,
        "final_tokens": final,
        "ceiling_tokens": ceiling,
        "fits": fits,
        "changed": changed,
        "shed": {phase: _empty_phase() for phase in SHED_PHASES},
        "never_drop": never_drop,
    }


def _is_notice(text: str) -> bool:
    if not text or text.startswith((_STATE_HEADER, _HANDOFF_HEADER, *_SOURCE_PREFIXES)):
        return False
    return text.startswith(_NOTICE_PREFIXES)


def _compact_notice(text: str) -> str:
    """Keep the notice identity and obligation/CE references in full, shed its prose body.

    review11 fix #2: a line that matches the never-drop keep pattern (obligations,
    counterexamples, tickets, ...) is preserved byte-for-byte, including the first/
    header line -- the real first-green disclosure line carries its complete open-
    counterexample ID list and action hint on that single line, so truncating it at
    480 chars silently drops IDs the model must see.
    """
    lines = text.splitlines()
    if not lines:
        return text
    kept: list[str] = []
    first = lines[0].strip()
    if first:
        if _NOTICE_KEEP_RE.search(first):
            kept.append(first)
        else:
            kept.append(first[:480] + ("…" if len(first) > 480 else ""))
    for line in lines[1:]:
        line = line.strip()
        if line and _NOTICE_KEEP_RE.search(line):
            kept.append(line)
    marker = "[F4 budget fit: verbose notice body omitted]"
    if marker not in kept:
        kept.append(marker)
    return "\n".join(kept)


def _contains_never_drop(value: object, *, parent_key: str = "") -> bool:
    key_text = parent_key.lower()
    if any(name in key_text for name in _NEVER_DROP_NAMES):
        return True
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(name in key_text for name in _NEVER_DROP_NAMES):
                return True
            if _contains_never_drop(item, parent_key=key_text):
                return True
        return False
    if _is_sequence(value):
        return any(_contains_never_drop(item, parent_key=parent_key) for item in value)
    text = str(value or "")
    return bool(_NOTICE_KEEP_RE.search(text))


def _status_is_open(value: object) -> bool:
    if isinstance(value, Mapping):
        for key in ("status", "state", "verdict", "result", "outcome"):
            if key in value:
                return _status_is_open(value[key])
        return False
    return str(value or "").strip().upper() in _OPEN_WORDS


def _protected_state_row(row: object, *, field: str) -> bool:
    field_name = field.lower()
    if field_name in {"blockers", "open_obligations", "open_work", "counterexamples",
                      "open_counterexamples", "advisory_counterexamples"}:
        return True
    if _contains_never_drop(row, parent_key=field_name):
        return True
    if isinstance(row, Mapping):
        return any(
            _status_is_open(row.get(key))
            for key in ("status", "state", "verdict", "result", "outcome")
        ) and any(name in field_name for name in ("obligation", "counterexample", "block"))
    return False


def _state_message_parts(message: Mapping[str, object]) -> tuple[dict, dict] | None:
    text = _message_text(message)
    if not text.startswith(_STATE_HEADER + "\n"):
        return None
    try:
        payload = json.loads(text.split("\n", 1)[1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    if not isinstance(state, dict):
        return None
    return payload, state


def _render_state_message(message: Mapping[str, object], payload: Mapping[str, object]) -> dict:
    return {**dict(message), "content": _STATE_HEADER + "\n" + json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)}


def _legacy_turn_messages(
    turns: Sequence[Sequence[Mapping[str, object]]],
) -> list[dict]:
    copied: list[dict] = []
    for turn in turns:
        if not isinstance(turn, Sequence) or isinstance(turn, (str, bytes, bytearray)):
            raise TypeError("legacy verbatim turns must be message sequences")
        if any(not isinstance(message, Mapping) for message in turn):
            raise TypeError("legacy verbatim turns must contain message mappings")
        copied.extend(copy.deepcopy(dict(message)) for message in turn)
    return copied


def _remove_legacy_tool_pairs(
    messages: list[dict],
    call_ids: Sequence[str],
) -> list[dict]:
    if isinstance(call_ids, (str, bytes, bytearray)):
        raise TypeError("legacy call ids must be a sequence of ids")
    removed_call_ids = {
        call_id for call_id in call_ids if isinstance(call_id, str) and call_id
    }
    removed_message_indexes: set[int] = set()
    for index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, Sequence):
            continue
        call_row_ids = {
            call.get("id") for call in calls if isinstance(call, Mapping)
            and isinstance(call.get("id"), str) and call.get("id")
        }
        if call_row_ids & removed_call_ids:
            removed_message_indexes.add(index)
            removed_call_ids.update(call_row_ids)
    return [
        message for index, message in enumerate(messages)
        if index not in removed_message_indexes
        and not (
            message.get("role") == "tool"
            and message.get("tool_call_id") in removed_call_ids
        )
    ]


def _strip_legacy_framefit_rows(messages: list[dict]) -> list[dict]:
    cleaned = list(messages)
    for index, message in enumerate(cleaned):
        parts = _state_message_parts(message)
        if parts is None:
            continue
        payload, state = parts
        retained_state = dict(state)
        retained_state.pop("framefit_action_ledger", None)
        retained_state.pop("framefit_open_edit_site", None)
        if retained_state != state:
            cleaned[index] = _render_state_message(
                message, {**payload, "state": retained_state}
            )
    return cleaned


def remove_framefit_action_context(
    messages: Sequence[Mapping[str, object]],
    call_ids: Sequence[str] = (),
    verbatim_turns: Sequence[Sequence[Mapping[str, object]]] = (),
) -> list[dict]:
    """Remove explicitly appended legacy turns and F4-owned action rows for fallback."""
    cleaned = [copy.deepcopy(dict(message)) for message in messages if isinstance(message, Mapping)]
    explicit_turn = _legacy_turn_messages(verbatim_turns)
    if explicit_turn:
        suffix_len = len(explicit_turn)
        if suffix_len > len(cleaned) or cleaned[-suffix_len:] != explicit_turn:
            raise ValueError("legacy verbatim turns are not the exact appended suffix")
        cleaned = cleaned[:-suffix_len]
    return _strip_legacy_framefit_rows(
        _remove_legacy_tool_pairs(cleaned, call_ids)
    )


def _first_continuity_row(
        rows: Sequence[object], field_name: str, omission_key: str | None,
) -> int | None:
    if omission_key is None:
        return None
    return next((index for index, row in enumerate(rows)
                 if not _protected_state_row(row, field=field_name)), None)


def _record_continuity_omission(
        state: dict[str, object], field_name: str, omission_key: str,
) -> None:
    current = state.get(omission_key)
    count = current if type(current) is int and current >= 0 else 0
    state[omission_key] = count + 1
    continuity = state.get("frame_continuity")
    if not isinstance(continuity, Mapping):
        return
    manifest = copy.deepcopy(dict(continuity))
    current_counts = manifest.get("omitted_counts")
    counts = copy.deepcopy(dict(current_counts)) if isinstance(current_counts, Mapping) else {}
    counts[field_name] = count + 1
    manifest["omitted_counts"] = counts
    state["frame_continuity"] = manifest


def _shed_state_rows(messages: list[dict], tools: object, ceiling: int,
                     phase: dict[str, object]) -> list[dict]:
    """Remove optional rows one at a time, retaining every protected row."""
    optional_fields = {"step", "advisories", "advisory", "notes", "history",
                       "receipts", "metadata"}
    while _estimate(messages, tools) > ceiling:
        candidates: list[tuple[int, int, str, int, dict]] = []
        for index, message in enumerate(messages):
            parts = _state_message_parts(message)
            if parts is None:
                continue
            payload, state = parts
            continuity = state.get("frame_continuity")
            continuity_active = (
                isinstance(continuity, Mapping)
                and continuity.get("schema") == "stateframe-frame-continuity/1"
            )
            for field, value in list(state.items()):
                field_name = str(field)
                if field_name in {"revision", "public_checks", "done_permission", "completion",
                                  "submission"}:
                    continue
                if continuity_active and field_name == "required_public_checks":
                    continue
                if _is_sequence(value):
                    rows = list(value)
                    continuity_count_key = (
                        _CONTINUITY_OMISSION_FIELDS.get(field_name)
                        if continuity_active else None
                    )
                    first_continuity_row = _first_continuity_row(
                        rows, field_name, continuity_count_key)
                    for row_index, row in enumerate(rows):
                        if (continuity_count_key is not None
                                and row_index != first_continuity_row):
                            continue
                        if _protected_state_row(row, field=field_name):
                            continue
                        candidate_payload = copy.deepcopy(payload)
                        candidate_state = candidate_payload["state"]
                        candidate_rows = list(candidate_state[field_name])
                        candidate_rows.pop(row_index)
                        if candidate_rows or continuity_count_key is not None:
                            candidate_state[field_name] = candidate_rows
                        else:
                            candidate_state.pop(field_name, None)
                        if continuity_count_key is not None:
                            _record_continuity_omission(
                                candidate_state, field_name, continuity_count_key)
                        candidate = _render_state_message(message, candidate_payload)
                        saving = token_estimate(message) - token_estimate(candidate)
                        candidates.append((saving, index, field_name, row_index, candidate))
                elif field_name in optional_fields and not _protected_state_row(
                        value, field=field_name):
                    candidate_payload = copy.deepcopy(payload)
                    candidate_payload["state"].pop(field_name, None)
                    candidate = _render_state_message(message, candidate_payload)
                    saving = token_estimate(message) - token_estimate(candidate)
                    candidates.append((saving, index, field_name, -1, candidate))
        if not candidates:
            break
        # Prefer the largest row, but keep input order deterministic on ties.
        _saving, index, field_name, row_index, candidate = max(
            candidates, key=lambda item: (item[0], -item[1], -item[3]))
        before = token_estimate(messages[index])
        messages[index] = candidate
        after = token_estimate(candidate)
        _record_phase(
            phase,
            item={"message_index": index, "field": field_name,
                  **({"row_index": row_index} if row_index >= 0 else {})},
            tokens=max(0, before - after),
        )
    return messages


def _valid_tool_pairs(messages: Sequence[Mapping[str, object]]) -> bool:
    pending: set[str] = set()
    for message in messages:
        role = message.get("role") if isinstance(message, Mapping) else None
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                return False
            pending.remove(call_id)
        elif role == "assistant":
            if pending:
                return False
            calls = message.get("tool_calls") or []
            if not _is_sequence(calls):
                return False
            for call in calls:
                call_id = call.get("id") if isinstance(call, Mapping) else None
                if not isinstance(call_id, str) or not call_id or call_id in pending:
                    return False
                pending.add(call_id)
    return not pending


def _m3_over_budget_sections(manifest: Mapping[str, object] | None) -> list[str]:
    if not isinstance(manifest, Mapping):
        return []
    sections: list[str] = []
    for section, token_key, budget_key in (
            ("metadata", "metadata_tokens", "metadata_budget_tokens"),
            ("source", "source_tokens", "source_budget_tokens")):
        tokens = manifest.get(token_key)
        budget = manifest.get(budget_key)
        if type(tokens) is int and type(budget) is int and tokens > budget:
            sections.append(section)
    return sections


def _attach_framefit_manifest(manifest: dict[str, object], *,
                              m3_manifest: Mapping[str, object] | None,
                              m3_fits: bool,
                              initial_over_budget_sections: Sequence[str] = (),
                              carried_source_phase: Mapping[str, object] | None = None,
                              ) -> None:
    manifest["phase_order"] = [*SHED_PHASES, "carried_source"]
    shed = manifest.get("shed")
    if isinstance(shed, dict):
        shed["carried_source"] = (
            copy.deepcopy(dict(carried_source_phase))
            if isinstance(carried_source_phase, Mapping) else _empty_phase()
        )
    manifest["framefit"] = {
        "schema": FRAMEFIT_SCHEMA,
        "m3_subbudget_advisory": not m3_fits,
        "m3_initial_over_budget_sections": list(initial_over_budget_sections),
        "m3_over_budget_sections": _m3_over_budget_sections(m3_manifest),
    }


def _defer_target_source(working: list[dict], tools: object,
                         ceiling_tokens: int,
                         route: Mapping[str, object] | None,
                         phase: dict[str, object], *, force: bool = False,
                         reason: str = "request_ceiling") -> bool:
    if not isinstance(route, Mapping):
        return False
    from kernel_stateframe_m3 import _read_tool_admitted
    if not _read_tool_admitted(tools):
        return False
    try:
        receipt = render_deferred_source_receipt(route)
    except ValueError:
        return False
    path = str(route["path"]).replace("\\", "/")
    changed = False
    for index, message in enumerate(working):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = _message_text(message)
        first_line = content.splitlines()[0] if content.splitlines() else ""
        header = _SOURCE_BLOCK_HEADER.fullmatch(first_line)
        if header is None or header.group(1).strip().replace("\\", "/") != path:
            continue
        replacement = {**dict(message), "content": first_line + "\n" + receipt}
        before = _estimate(working, tools)
        if token_estimate(message) <= token_estimate(replacement):
            continue
        working[index] = replacement
        changed = True
        after = _estimate(working, tools)
        _record_phase(
            phase,
            item={"message_index": index, "start_line": route["start_line"],
                  "end_line": route["end_line"], "reason": reason},
            tokens=max(0, before - after),
        )
        if not force and after <= ceiling_tokens:
            return changed
    return changed


# Room for the non-message envelope (model, max_tokens, sampling fields) on top of messages + tools.
RETAIN_BYTE_HEADROOM = 2048


def _retained_request_fits_bytes(messages: object, tools: object) -> bool:
    """The kept frame must also fit the endpoint BYTE cap, not only the token ceiling.

    b1441 (10-19): F1 kept 84,149/96,432-byte requests under the token ceiling; final revalidation
    refused them (request_bytes_exceed_max_request_bytes > 82,026), recovery could not shrink them,
    and the run died RUN_INCOMPLETE. Unmeasurable -> not retained (historical fallback).
    """
    try:
        from kernel_request_budget import HARD_MAX_REQUEST_BYTES
        size = len(json.dumps({"messages": messages, "tools": tools}).encode("utf-8"))
        return size + RETAIN_BYTE_HEADROOM <= int(HARD_MAX_REQUEST_BYTES)
    except Exception:
        return False


BYTE_FIT_MODE_ENV = "STATEFRAME_F4_FIT_BYTES"
# Lowest value first: stale findings, advisory context, receipts, metadata, then
# free-form notes. Task, open-edit bytes, and verdict rows remain protected.
BYTE_FIT_STATE_SECTION_PRIORITY = (
    "history_findings", "advisories", "receipts", "metadata", "notes",
)
_BYTE_FIT_NEVER_DROP_NAMES = (
    "task", "last_verdict", "verdict", "open_edit", "edit_site",
)
_BYTE_SOURCE_HEADER = re.compile(
    r"^\[(?:Current|Working) source:\s*([^\r\n\]]+)\]$",
)
_BYTE_COVERAGE = re.compile(
    r"(?m)^Coverage: lines (\d+)-(\d+), columns \d+-\d+; "
    r"current bytes are delivered for this range\.$",
)


def byte_fit_mode(value: object = None) -> str:
    if value is None:
        raw = os.environ.get(BYTE_FIT_MODE_ENV, "off")
    elif isinstance(value, Mapping):
        raw = value.get(BYTE_FIT_MODE_ENV, "off")
    else:
        raw = value
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{BYTE_FIT_MODE_ENV} must be off or on")
    return raw.strip().lower()


def byte_fit_enabled(value: object = None) -> bool:
    return byte_fit_mode(value) == "on"


def _safe_byte_fit_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.replace("\\", "/").strip()
    parts = path.split("/")
    if (not path or path.startswith("/") or re.match(r"^[A-Za-z]:", path)
            or any(part in {"", ".", ".."} or part.casefold() == "backups" for part in parts)
            or any(char in path for char in "\x00\r\n")):
        return None
    return path


def _source_section(message: object) -> dict[str, object] | None:
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    first_line = content.splitlines()[0] if content.splitlines() else ""
    match = _BYTE_SOURCE_HEADER.fullmatch(first_line)
    if match is None:
        return None
    path = _safe_byte_fit_path(match.group(1))
    if path is None:
        return None
    metadata = None
    try:
        from kernel_request_source_window import parse_model_source_window, parse_window
        metadata = parse_model_source_window(content) or parse_window(content)
    except ImportError:
        pass
    ranges: list[tuple[int, int]] = []
    raw_ranges = metadata.get("ranges") if isinstance(metadata, Mapping) else None
    if isinstance(raw_ranges, Sequence) and not isinstance(raw_ranges, (str, bytes, bytearray)):
        for row in raw_ranges:
            if (isinstance(row, Mapping) and type(row.get("start_line")) is int
                    and type(row.get("end_line")) is int
                    and row["start_line"] >= 1 and row["end_line"] >= row["start_line"]):
                ranges.append((row["start_line"], row["end_line"]))
    if not ranges:
        coverage = _BYTE_COVERAGE.search(content)
        if coverage is not None:
            ranges.append((int(coverage.group(1)), int(coverage.group(2))))
    if not ranges:
        return None
    return {"path": path, "ranges": tuple(ranges)}


def _read_tool_available(tools: object) -> bool:
    if not _is_sequence(tools):
        return False
    for tool in tools:
        if isinstance(tool, str) and tool == "read":
            return True
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
        if name == "read":
            return True
    return False


def _serialized_request_bytes(body: Mapping[str, object]) -> int | None:
    try:
        return len(json.dumps(body).encode("utf-8"))
    except (TypeError, ValueError, OverflowError):
        return None


def _byte_fit_required(
        body: Mapping[str, object], *, max_request_bytes: int,
        headroom: int = RETAIN_BYTE_HEADROOM,
) -> bool:
    size = _serialized_request_bytes(body)
    return size is None or size > max_request_bytes - headroom


def _state_section_is_empty(value: object) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return True
    if isinstance(value, Mapping):
        return all(
            str(key) in {"schema", "version"}
            or _state_section_is_empty(item)
            for key, item in value.items()
        )
    if _is_sequence(value):
        return all(_state_section_is_empty(item) for item in value)
    return False


def _contains_byte_fit_never_drop(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).casefold()
            if any(name in key_text for name in _BYTE_FIT_NEVER_DROP_NAMES):
                return True
            if _contains_byte_fit_never_drop(item):
                return True
        return False
    if _is_sequence(value):
        return any(_contains_byte_fit_never_drop(item) for item in value)
    return bool(re.search(
        r"\b(?:task|last\s+verdict|verdict|open\s+edit|edit\s+site)\b",
        str(value or ""), re.IGNORECASE,
    ))


def _byte_fit_protected_state_row(row: object, *, field: str) -> bool:
    return (
        _protected_state_row(row, field=field)
        or any(name in field.casefold() for name in _BYTE_FIT_NEVER_DROP_NAMES)
        or _contains_byte_fit_never_drop(row)
    )


def _state_byte_drop_candidate(
        message: Mapping[str, object], field: str, row_index: int | None,
) -> dict[str, object] | None:
    parts = _state_message_parts(message)
    if parts is None:
        return None
    payload, state = parts
    value = state.get(field)
    candidate_payload = copy.deepcopy(payload)
    if _is_sequence(value):
        rows = list(value)
        if row_index is None or not 0 <= row_index < len(rows):
            return None
        if _byte_fit_protected_state_row(rows[row_index], field=field):
            return None
        rows.pop(row_index)
        if rows:
            candidate_payload["state"][field] = rows
        else:
            candidate_payload["state"].pop(field, None)
    elif row_index is None and (
            _state_section_is_empty(value)
            or (isinstance(value, Mapping)
                and not _byte_fit_protected_state_row(value, field=field))):
        candidate_payload["state"].pop(field, None)
    else:
        return None
    return _render_state_message(message, candidate_payload)


def _shed_state_sections_for_bytes(
        messages: list[dict[str, object]], body: dict[str, object], target: int,
        dropped: list[dict[str, object]],
) -> None:
    """Shed only unprotected rows in the declared state-field priority."""
    for field in BYTE_FIT_STATE_SECTION_PRIORITY:
        while (current := _serialized_request_bytes(body)) is not None and current > target:
            applied = False
            for index, message in enumerate(messages):
                parts = _state_message_parts(message)
                if parts is None or field not in parts[1]:
                    continue
                value = parts[1][field]
                row_indices = (
                    [row_index for row_index, row in enumerate(value)
                     if not _byte_fit_protected_state_row(row, field=field)]
                    if _is_sequence(value) else [None]
                )
                for row_index in row_indices:
                    candidate = _state_byte_drop_candidate(message, field, row_index)
                    if candidate is None:
                        continue
                    before = _serialized_request_bytes(body)
                    messages[index] = candidate
                    body["messages"] = messages
                    after = _serialized_request_bytes(body)
                    if before is None or after is None or after >= before:
                        messages[index] = message
                        body["messages"] = messages
                        continue
                    dropped.append({"section": f"state.{field}", "bytes": before - after})
                    applied = True
                    break
                if applied:
                    break
            if not applied:
                break


def _omitted_source_rows(messages: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue
        marker = re.search(r"(?m)^\[Host fact\] omitted to fit: (.*)$", content)
        if marker is None:
            continue
        for path, start, end in re.findall(r"([^ ,;]+) lines (\d+)-(\d+)", marker.group(1)):
            safe_path = _safe_byte_fit_path(path)
            if safe_path is not None:
                rows.append({"path": safe_path, "ranges": ((int(start), int(end)),)})
    return rows


def _omitted_source_line(rows: Sequence[Mapping[str, object]]) -> str:
    unique: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        path = _safe_byte_fit_path(row.get("path"))
        ranges = row.get("ranges")
        if path is None or not _is_sequence(ranges):
            continue
        for span in ranges:
            if (not _is_sequence(span) or len(span) != 2
                    or type(span[0]) is not int or type(span[1]) is not int
                    or span[0] < 1 or span[1] < span[0]):
                continue
            key = (path, span[0], span[1])
            if key not in seen:
                seen.add(key)
                unique.append(key)
    sections = ", ".join(f"{path} lines {start}-{end}" for path, start, end in unique)
    routes = ", ".join(
        f"read(path={json.dumps(path, ensure_ascii=False)}, start_line={start}, end_line={end})"
        for path, start, end in unique
    )
    if not sections or not routes:
        return ""
    return f"[Host fact] omitted to fit: {sections}; {routes} loads them."


def _put_omitted_source_line(
        messages: list[dict[str, object]], rows: Sequence[Mapping[str, object]],
) -> bool:
    line = _omitted_source_line(rows)
    if not line:
        return False
    for index, message in enumerate(messages):
        parts = _state_message_parts(message)
        if parts is None:
            continue
        payload, _state = parts
        payload["omitted_state"] = line
        messages[index] = _render_state_message(message, payload)
        return True
    return False


def _remove_prior_exchanges(
        messages: list[dict[str, object]],
        protected_source_paths: Sequence[str] = (),
) -> tuple[list[dict[str, object]], int]:
    """Keep task/frame, protected edit/verdict context, and only the latest user turn."""
    last_user = max(
        (index for index, message in enumerate(messages)
         if message.get("role") == "user"),
        default=-1,
    )
    if last_user < 0:
        return messages, 0
    task_indices = {
        index for index, message in enumerate(messages)
        if message.get("role") == "user"
        and (
            _message_text(message).startswith("Task: # ")
            or str(message.get("_request_budget_section", "")).casefold()
            in {"task", "ticket", "active_task"}
        )
    }
    protected_paths = {
        path for raw in protected_source_paths
        if (path := _safe_byte_fit_path(raw)) is not None
    }
    protected_markers = (
        "[Last verdict]", "[VERIFICATION CERTIFICATE]", "[Unified stop authority]",
        "[Open edit]", "[OPEN EDIT]", "open_edit_bytes",
    )
    drop_indices: set[int] = set()
    for index, message in enumerate(messages[:last_user]):
        role = message.get("role")
        text = _message_text(message)
        lowered = text.casefold()
        source = _source_section(message)
        protected = (
            role in {"system", "developer"}
            or index in task_indices
            or _state_message_parts(message) is not None
            or any(marker.casefold() in lowered for marker in protected_markers)
            or (source is not None and source["path"] in protected_paths)
        )
        if not protected:
            drop_indices.add(index)
    if not drop_indices:
        return messages, 0
    dropped_bytes = sum(
        len(json.dumps(messages[index]).encode("utf-8")) for index in drop_indices
    )
    candidate = [
        message for index, message in enumerate(messages) if index not in drop_indices
    ]
    if not _valid_tool_pairs(candidate):
        return messages, 0
    return candidate, dropped_bytes


def fit_request_to_byte_cap(
        body: Mapping[str, object], *, max_request_bytes: int,
        protected_source_paths: Sequence[str] = (),
        headroom: int = RETAIN_BYTE_HEADROOM,
        allow_source_omission: bool = True,
) -> dict[str, object]:
    """Trim safe, reloadable request sections in fixed priority without losing the live turn."""
    if type(max_request_bytes) is not int or max_request_bytes <= 0:
        raise ValueError("maximum request bytes must be a positive integer")
    if type(headroom) is not int or headroom < 0:
        raise ValueError("byte fit headroom must be a non-negative integer")
    original = copy.deepcopy(dict(body))
    messages = original.get("messages")
    tools = original.get("tools")
    if not isinstance(messages, list) or any(not isinstance(row, dict) for row in messages):
        before = _serialized_request_bytes(original)
        return {
            "body": original, "fits": False, "changed": False,
            "before": before, "after": before, "dropped": [],
            "omitted_sources": [], "headroom": headroom,
        }
    before = _serialized_request_bytes(original)
    if before is None:
        return {
            "body": original, "fits": False, "changed": False,
            "before": None, "after": None, "dropped": [],
            "omitted_sources": [], "headroom": headroom,
        }
    target = max_request_bytes - headroom
    dropped: list[dict[str, object]] = []
    omitted = _omitted_source_rows(messages)
    changed = False

    def size() -> int | None:
        return _serialized_request_bytes(original)

    current = size()
    if current is not None and current <= target:
        return {
            "body": original, "fits": True, "changed": False,
            "before": before, "after": current, "dropped": [],
            "omitted_sources": omitted, "headroom": headroom,
        }

    messages, exchange_bytes = _remove_prior_exchanges(messages, protected_source_paths)
    if exchange_bytes:
        original["messages"] = messages
        dropped.append({"section": "prior_turn_history", "bytes": exchange_bytes})
        changed = True
        current = size()

    _shed_state_sections_for_bytes(messages, original, target, dropped)
    changed = changed or bool(dropped)
    current = size()

    if (allow_source_omission and _read_tool_available(tools)
            and (current is None or current > target)):
        protected = {
            path for raw in protected_source_paths
            if (path := _safe_byte_fit_path(raw)) is not None
        }
        last_user = max(
            (index for index, message in enumerate(messages)
             if message.get("role") == "user"),
            default=-1,
        )
        candidates = []
        for index, message in enumerate(messages):
            source = _source_section(message)
            content = message.get("content") or ""
            if (source is None or index == last_user or source["path"] in protected
                    or any(marker in content for marker in (
                        "[VERIFICATION CERTIFICATE]", "[Unified stop authority]",
                        "[Last verdict]", "[Host fact] omitted to fit:",
                    ))):
                continue
            candidates.append((index, source))
        for index, source in reversed(candidates):
            if current is not None and current <= target:
                break
            path = str(source["path"])
            ranges = source["ranges"]
            candidate_messages = list(messages)
            removed_bytes = len(json.dumps(candidate_messages[index]).encode("utf-8"))
            candidate_messages.pop(index)
            row = {"path": path, "ranges": ranges}
            candidate_omissions = [*omitted, row]
            if not _put_omitted_source_line(candidate_messages, candidate_omissions):
                continue
            original["messages"] = candidate_messages
            candidate_size = size()
            # A projection is only useful if it reduces the actual serialized body.
            if candidate_size is None or (current is not None and candidate_size >= current):
                original["messages"] = messages
                continue
            messages = candidate_messages
            omitted = candidate_omissions
            dropped.append({"section": "source_window", "bytes": removed_bytes})
            changed = True
            current = candidate_size

    after = size()
    fits = after is not None and after <= target
    return {
        "body": original, "fits": fits, "changed": changed,
        "before": before, "after": after, "dropped": dropped,
        "omitted_sources": omitted, "headroom": headroom,
    }


def _should_retain_ceiling_fitting_frame(
        *, m3_fits: bool, framefit_active: bool,
        fitted_tokens: int, ceiling_tokens: int,
        messages: object = None, tools: object = None,
) -> bool:
    if m3_fits or framefit_active or fitted_tokens > ceiling_tokens:
        return False
    try:
        if not fitted_frame_retention_enabled():
            return False
    except Exception:
        # Invalid configuration or a feature error keeps the historical fallback.
        return False
    return _retained_request_fits_bytes(messages, tools)


def fit_f4_request(messages: Sequence[Mapping[str, object]], tools: object,
                   ceiling_tokens: int, *, enabled_value: bool | None = None,
                   protected_call_ids: Sequence[str] = (),
                   covered_call_ids: Sequence[str] | None = None,
                   protected_source_keys: Sequence[Sequence[object]] = (),
                   protected_source_paths: Sequence[str] = (),
                   framefit_enabled_value: bool = False,
                   target_source_route: Mapping[str, object] | None = None) -> dict[str, object]:
    """Fit one rendered F4 request and return its auditable shed manifest.

    The result is a mapping to keep this presentation helper easy to consume
    from the old runtime without changing ``PreparedRequest``.  ``messages``
    is returned by identity on the disabled arm.
    """
    if type(ceiling_tokens) is not int or ceiling_tokens <= 0:
        raise ValueError("F4 fit ceiling must be a positive integer")
    active = enabled() if enabled_value is None else bool(enabled_value)
    framefit_active = type(framefit_enabled_value) is bool and framefit_enabled_value
    protected_ids = {
        value for value in protected_call_ids
        if isinstance(value, str) and value
    } if _is_sequence(protected_call_ids) else set()
    initial = _estimate(messages, tools)
    if not active:
        manifest = _manifest(initial=initial, final=initial, ceiling=ceiling_tokens,
                             fits=initial <= ceiling_tokens, changed=False,
                             enabled_value=False)
        return {"messages": messages, "manifest": manifest, "fits": initial <= ceiling_tokens,
                "changed": False, "initial_tokens": initial, "final_tokens": initial,
                "ceiling_tokens": ceiling_tokens}

    working_messages: Sequence[Mapping[str, object]] = messages
    m3_manifest: dict[str, object] | None = None
    m3_fits = True
    m3_changed = False
    initial_over_budget_sections: list[str] = []
    carried_source_phase = _empty_phase()
    from kernel_stateframe_m3 import (
        apply_m3_working_context_budget, m3_working_context_enabled,
    )
    # F5 replaces prior exchanges with typed ledger state. Remove only exchanges
    # already covered there, while the newest provider batch always wins.
    if framefit_active:
        covered_ids = (
            {
                value for value in covered_call_ids
                if isinstance(value, str) and value
            }
            if _is_sequence(covered_call_ids)
            else _ledger_covered_call_ids(messages)
        )
        covered_ids.difference_update(protected_ids)
        if covered_ids:
            covered_messages = _remove_covered_exchanges(messages, covered_ids)
            if covered_messages != list(messages):
                working_messages = covered_messages
                m3_changed = True
    if m3_working_context_enabled():
        m3_result = apply_m3_working_context_budget(
            working_messages, tools, protected_source_keys=protected_source_keys,
            protected_source_paths=protected_source_paths,
            enabled_value=True)
        working_messages = m3_result["messages"]
        m3_manifest = copy.deepcopy(m3_result["manifest"])
        m3_fits = bool(m3_result["fits"])
        m3_changed = bool(m3_result["changed"]) or m3_changed
        initial_over_budget_sections = _m3_over_budget_sections(m3_manifest)
        if framefit_active and "source" in initial_over_budget_sections:
            source_candidates = [copy.deepcopy(dict(message)) for message in working_messages]
            if _defer_target_source(
                    source_candidates, tools, ceiling_tokens, target_source_route,
                    carried_source_phase, force=True, reason="m3_source_subbudget"):
                previous_deferrals = m3_manifest.get("deferrals", [])
                rechecked = apply_m3_working_context_budget(
                    source_candidates, tools, protected_source_keys=protected_source_keys,
                    protected_source_paths=protected_source_paths, enabled_value=True)
                refreshed_manifest = copy.deepcopy(rechecked["manifest"])
                refreshed_deferrals = refreshed_manifest.get("deferrals", [])
                refreshed_manifest["deferrals"] = [
                    *([copy.deepcopy(item) for item in previous_deferrals]
                      if _is_sequence(previous_deferrals) else []),
                    *([copy.deepcopy(item) for item in refreshed_deferrals]
                      if _is_sequence(refreshed_deferrals) else []),
                ]
                m3_manifest = refreshed_manifest
                working_messages = rechecked["messages"]
                m3_fits = bool(rechecked["fits"])
                m3_changed = True
    byte_fit_manifest: dict[str, object] | None = None
    try:
        byte_fit_enabled_value = byte_fit_enabled()
    except ValueError:
        byte_fit_enabled_value = False
    from kernel_request_budget import HARD_MAX_REQUEST_BYTES
    byte_fit_body = {"messages": working_messages, "tools": tools}
    byte_fit_required = _byte_fit_required(
        byte_fit_body, max_request_bytes=int(HARD_MAX_REQUEST_BYTES),
    )
    byte_fit_requested = (
        not framefit_active and byte_fit_enabled_value
        and (not m3_fits or byte_fit_required)
    )
    if byte_fit_requested:
        byte_body = {
            "messages": [copy.deepcopy(dict(message)) for message in working_messages],
            "tools": copy.deepcopy(tools),
        }
        byte_result = fit_request_to_byte_cap(
            byte_body,
            max_request_bytes=int(HARD_MAX_REQUEST_BYTES),
            protected_source_paths=protected_source_paths,
            allow_source_omission=False,
        )
        byte_fit_manifest = {
            "before": byte_result["before"],
            "after": byte_result["after"],
            "headroom": byte_result["headroom"],
            "dropped": copy.deepcopy(byte_result["dropped"]),
            "fits": bool(byte_result["fits"]),
            "changed": bool(byte_result["changed"]),
        }
        if byte_result["fits"] and byte_result["changed"]:
            fitted_body = byte_result["body"]
            fitted_messages = fitted_body.get("messages")
            if isinstance(fitted_messages, list):
                working_messages = fitted_messages
                m3_changed = True
            else:
                byte_fit_manifest["fits"] = False
    fitted_initial = _estimate(working_messages, tools)

    byte_fit_retained_frame = bool(
        byte_fit_requested and not m3_fits
        and isinstance(byte_fit_manifest, Mapping)
        and byte_fit_manifest.get("fits") is True
    )
    byte_fit_failed = bool(
        byte_fit_requested
        and (not isinstance(byte_fit_manifest, Mapping)
             or byte_fit_manifest.get("fits") is not True)
    )
    legacy_retained_frame = _should_retain_ceiling_fitting_frame(
        m3_fits=m3_fits,
        framefit_active=framefit_active,
        fitted_tokens=fitted_initial,
        ceiling_tokens=ceiling_tokens,
        messages=working_messages,
        tools=tools,
    )
    retain_ceiling_fitting_frame = byte_fit_retained_frame or legacy_retained_frame

    if byte_fit_failed or (
            not m3_fits and not framefit_active and not retain_ceiling_fitting_frame):
        manifest = _manifest(initial=initial, final=fitted_initial, ceiling=ceiling_tokens,
                             fits=False, changed=m3_changed, enabled_value=True,
                             protect_latest_batch=bool(protected_ids))
        if m3_manifest is not None:
            manifest["m3_working_context"] = m3_manifest
        if byte_fit_manifest is not None:
            manifest["f4_byte_fit"] = copy.deepcopy(byte_fit_manifest)
            if not byte_fit_manifest["fits"]:
                manifest["fit_false"] = {
                    "reason": "minimum_frame_byte_cap",
                    "before": byte_fit_manifest["before"],
                    "after": byte_fit_manifest["after"],
                }
        manifest["fits"] = False
        manifest["changed"] = m3_changed
        return {
            "messages": working_messages, "manifest": manifest, "fits": False,
            "changed": m3_changed, "initial_tokens": initial,
            "final_tokens": fitted_initial, "ceiling_tokens": ceiling_tokens,
        }

    if fitted_initial <= ceiling_tokens:
        manifest = _manifest(initial=initial, final=initial, ceiling=ceiling_tokens,
                             fits=True, changed=m3_changed, enabled_value=True,
                             protect_latest_batch=bool(protected_ids))
        manifest["final_tokens"] = fitted_initial
        manifest["changed"] = m3_changed
        if m3_manifest is not None:
            manifest["m3_working_context"] = m3_manifest
        if byte_fit_manifest is not None:
            manifest["f4_byte_fit"] = copy.deepcopy(byte_fit_manifest)
        if legacy_retained_frame:
            manifest["f1_fit_retained_over_m3_subbudget"] = True
        if byte_fit_retained_frame:
            manifest["f4_byte_fit_retained_over_m3_subbudget"] = True
        if framefit_active:
            _attach_framefit_manifest(
                manifest, m3_manifest=m3_manifest, m3_fits=m3_fits,
                initial_over_budget_sections=initial_over_budget_sections,
                carried_source_phase=carried_source_phase)
        return {"messages": working_messages, "manifest": manifest, "fits": True,
                "changed": m3_changed, "initial_tokens": initial, "final_tokens": fitted_initial,
                "ceiling_tokens": ceiling_tokens}

    working = [copy.deepcopy(dict(message)) for message in working_messages]
    manifest = _manifest(initial=initial, final=fitted_initial, ceiling=ceiling_tokens,
                         fits=False, changed=m3_changed, enabled_value=True,
                         protect_latest_batch=bool(protected_ids))
    if m3_manifest is not None:
        manifest["m3_working_context"] = m3_manifest
    if framefit_active:
        _attach_framefit_manifest(
            manifest, m3_manifest=m3_manifest, m3_fits=m3_fits,
            initial_over_budget_sections=initial_over_budget_sections,
            carried_source_phase=carried_source_phase)
    shed = manifest["shed"]

    # Phase 1: older real exchanges, oldest first; each assistant/tool group
    # is atomic. When the caller identifies a provider batch, retain every pair
    # in that response even when the batch has more than two calls.
    while _estimate(working, tools) > ceiling_tokens:
        exchanges = _exchange_spans(working)
        real = [exchange for exchange in exchanges if exchange.real]
        if protected_ids:
            removable = [exchange for exchange in real
                         if not protected_ids.intersection(exchange.call_ids)]
            if not removable:
                break
            exchange = removable[0]
        else:
            if len(real) <= 2:
                break
            exchange = real[0]
        before = _estimate(working, tools)
        working = _remove_spans(working, (exchange,))
        after = _estimate(working, tools)
        _record_phase(
            shed["older_real_exchanges"],
            item={"call_ids": list(exchange.call_ids), "message_count": exchange.end - exchange.start + 1},
            tokens=max(0, before - after),
        )

    # Phase 2: notice bodies.  The identity and any open obligation/CE line stay.
    # review11 fix #2: only standalone host notices are eligible -- never a retained
    # exchange message (tool result / assistant call), whose content can legitimately
    # begin with the same textual prefix as a real host notice but which phase 1 (and
    # the never-drop pairing invariant) already governs.
    if _estimate(working, tools) > ceiling_tokens:
        notice_excluded = {
            index for exchange in _exchange_spans(working)
            for index in range(exchange.start, exchange.end + 1)
        }
        for index, message in enumerate(working):
            if index in notice_excluded:
                continue
            if not isinstance(message, Mapping) or message.get("role") != "user":
                continue
            text = _message_text(message)
            if not _is_notice(text):
                continue
            compacted = _compact_notice(text)
            if compacted == text:
                continue
            before = token_estimate(message)
            working[index] = {**message, "content": compacted}
            after = token_estimate(working[index])
            _record_phase(
                shed["verbose_notice_bodies"],
                item={"message_index": index, "header": compacted.split("\n", 1)[0]},
                tokens=max(0, before - after),
            )
            if _estimate(working, tools) <= ceiling_tokens:
                break

    # Phase 3: lower-priority rows in the canonical state block.
    if _estimate(working, tools) > ceiling_tokens:
        _shed_state_rows(working, tools, ceiling_tokens, shed["lower_priority_state_rows"])

    if framefit_active and _estimate(working, tools) > ceiling_tokens:
        source_phase = shed.get("carried_source")
        if isinstance(source_phase, dict):
            _defer_target_source(
                working, tools, ceiling_tokens, target_source_route, source_phase)

    final = _estimate(working, tools)
    changed = working != list(working_messages) or m3_changed
    fits = final <= ceiling_tokens
    manifest["final_tokens"] = final
    manifest["fits"] = fits
    manifest["changed"] = changed
    result = {
        "messages": working if changed else messages,
        "manifest": manifest,
        "fits": fits,
        "changed": changed,
        "initial_tokens": initial,
        "final_tokens": final,
        "ceiling_tokens": ceiling_tokens,
    }
    if not _valid_tool_pairs(result["messages"]):
        raise AssertionError("F4 budget fitting split a tool call/result pair")
    return result


fit_request = fit_f4_request
fit_messages = fit_f4_request


__all__ = [
    "BYTE_FIT_MODE_ENV", "FIT_SCHEMA", "FITTED_FRAME_RETENTION_ENV",
    "FRAMEFIT_MODE_ENV", "FRAMEFIT_SCHEMA", "MODE_ENV", "SCHEMA", "SHED_PHASES",
    "byte_fit_enabled", "byte_fit_mode", "enabled", "fit_f4_request",
    "fit_messages", "fit_request", "fit_request_to_byte_cap", "fitted_frame_retention_enabled",
    "remove_framefit_action_context",
    "fitted_frame_retention_mode", "framefit_enabled", "framefit_mode",
    "is_valid_deferred_target_source_message", "mode",
    "parse_deferred_source_receipt", "render_deferred_source_receipt",
]
