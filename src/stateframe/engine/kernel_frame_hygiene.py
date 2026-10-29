"""Presentation-only hygiene for model-facing StateFrame messages.

The host keeps admission, reducer, receipt, and completion authority unchanged.
This module only projects the final wire messages so stale or uncertifiable
metadata cannot masquerade as an actionable blocker.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence

from kernel_stateframe_envelope import STATE_HEADER, serialize_stateframe_payload


MODE_ENV = "STATEFRAME_FRAME_HYGIENE"
SCHEMA = "stateframe-frame-hygiene/1"
_STATE_PREFIX = STATE_HEADER + "\n"
_ID_RE = re.compile(r"(?<![A-Za-z0-9_])(C[1-5]|obl-\d{3}|obligation:unidentified:\d+)(?![A-Za-z0-9_])")
_CLAUSE_RE = re.compile(
    r"^\s*(?P<id>C[1-5]|obl-\d{3}|obligation:unidentified:\d+)\s*"
    r"(?:\||:)\s*(?P<status>[A-Za-z_]+)\s*(?:\||:)\s*(?P<meaning>.*)$"
)
_TARGET_BLOCKS = (
    "[VERIFICATION CERTIFICATE]",
    "[Unified stop authority]",
    "[DELIVERY_MANIFEST",
    "[Get-sure state]",
    "[Decision handoff]",
    "[TOKENS row6 continuity]",
    "[TOKENS provability advisory]",
    "[ENVIRONMENT STATE]",
    "<settled_state>",
    "[OPERATOR",
)
_STALE_LINES = (
    "checklist:",
    "normative clause coverage is incomplete",
    "check_inconclusive:",
    "provability notice already delivered; disposition:",
    "open requirement ids:",
    "pending checks:",
    "action dispositions:",
    "last closed action:",
    "current check/refusal:",
    "pending model intention",
    "hypothesis/approach:",
    "decisive evidence:",
    "next decision:",
    "work:",
)
_PROTECTED_MARKERS = (
    "COUNTEREXAMPLE",
    "NON_WAIVABLE",
    "NON-WAIVABLE",
    "VIOLATED",
    "PUBLIC CHECKS ON THE CURRENT REVISION ARE RED",
)
_MEANING_KEYS = (
    "meaning", "description", "clause", "missing", "what_missing", "detail", "message", "text",
)
_ACTION_KEYS = (
    "action", "next_action", "repair", "repair_action", "resolution", "available_action", "tool",
)
_BLOCKER_KEYS = (
    "blockers", "blocking", "blocking_items", "needs_work", "needs_work_items", "open_blockers",
)
_OPEN_KEYS = (
    "open_items", "open_obligations", "open_obligation", "pending_checks", "open_requirement_ids",
)
_FINAL_SUBMIT_RE = re.compile(
    r"Final step: call done now to submit this revision "
    r"\(uncertified; open: \[[^\]]*\]\)\. No other action is available\."
)
_EXHAUSTED_ROUTE_RE = re.compile(
    r"(?:probe|refresh)[\s\S]{0,500}(?:remaining|budget)[\s\S]*?[:=]\s*0"
    r"|(?:remaining|budget)[\s\S]{0,500}(?:probe|refresh)[\s\S]*?[:=]\s*0",
    re.IGNORECASE,
)
_SOURCE_BLOCK_PREFIXES = (
    "[Working source:",
    "[Paged source:",
    "[Current source:",
    "[Retained source:",
    "[Observed source:",
    "[Repair source:",
    "[Source content:",
    "CURRENT_SOURCE ",
)


def mode() -> str:
    """Return strict opt-in mode; an unset or invalid value is off."""
    return "on" if os.environ.get(MODE_ENV, "off").strip().lower() in {"1", "on", "true", "yes"} else "off"


def enabled() -> bool:
    return mode() == "on"


def _is_source_block(content: str) -> bool:
    """Return whether content is a source block whose bytes belong to the host contract."""
    if not isinstance(content, str):
        return False
    header = content.split("\n", 1)[0]
    return header.startswith(_SOURCE_BLOCK_PREFIXES)


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _public_meaning(value: object) -> str | None:
    meaning = _text(value)
    if not meaning:
        return None
    upper = meaning.upper()
    if upper in {"UNKNOWN", "UNPROBED", "NO_BOUND_CHECK", "MISSING_CHECKS", "AMBIGUOUS_CONTRACT"}:
        return None
    if "NO AVAILABLE HOST TOOL CAN ESTABLISH" in upper:
        return None
    return meaning


def _mapping_rows(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        if any(key in value for key in (*_MEANING_KEYS, *_ACTION_KEYS, "id", "obligation_id", "check_id")):
            return [value]
        return [item for item in value.values() if isinstance(item, Mapping)]
    if _is_sequence(value):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _row_id(row: Mapping[str, object]) -> str | None:
    for key in ("id", "obligation_id", "check_id", "clause_id", "name"):
        value = _text(row.get(key))
        if value and _ID_RE.fullmatch(value):
            return value
    return None


def _row_meaning(row: Mapping[str, object]) -> str | None:
    for key in _MEANING_KEYS:
        meaning = _public_meaning(row.get(key))
        if meaning:
            return meaning
    return None


def _row_action(row: Mapping[str, object]) -> str | None:
    for key in _ACTION_KEYS:
        action = _public_meaning(row.get(key))
        if action:
            return action
    return None


def _formatted_row(row: Mapping[str, object]) -> dict[str, str] | None:
    identifier = _row_id(row)
    meaning = _row_meaning(row)
    action = _row_action(row)
    if not identifier or not meaning or not action:
        return None
    return {"id": identifier, "missing": meaning, "action": action}


def _structured_entries(frame: Mapping[str, object], keys: Sequence[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in keys:
        for row in _mapping_rows(frame.get(key)):
            item = _formatted_row(row)
            if item and item not in rows:
                rows.append(item)
    return rows


def _structured_open(frame: Mapping[str, object]) -> list[str]:
    values: list[str] = []
    for key in _OPEN_KEYS:
        raw = frame.get(key)
        rows = _mapping_rows(raw)
        if rows:
            for row in rows:
                identifier = _row_id(row)
                meaning = _row_meaning(row)
                if identifier and meaning:
                    values.append(f"{identifier} — {meaning}")
            continue
        if _is_sequence(raw):
            for value in raw:
                if isinstance(value, str) and _ID_RE.fullmatch(value.strip()):
                    # An identifier with no public meaning is deliberately dropped.
                    continue
                meaning = _public_meaning(value)
                if meaning and "unverifiable_claim" not in meaning:
                    values.append(meaning)
    return list(dict.fromkeys(values))


def _remove_bare_ids(value: object) -> object:
    """Drop identifier-only JSON values unless their public meaning travels with them."""
    if isinstance(value, Mapping):
        identifier = _row_id(value)
        if identifier and _row_meaning(value) is None:
            return None
        result: dict[str, object] = {}
        for key, item in value.items():
            cleaned = _remove_bare_ids(item)
            if cleaned is not None:
                result[str(key)] = cleaned
        return result
    if _is_sequence(value):
        result_list: list[object] = []
        for item in value:
            cleaned = _remove_bare_ids(item)
            if cleaned is not None:
                result_list.append(cleaned)
        return result_list
    if isinstance(value, str) and _ID_RE.search(value):
        if " — " not in value:
            return None
    return value


def _remove_exhausted_routes(value: object) -> object:
    """Do not expose a refresh/probe route after its host budget reaches zero."""
    if isinstance(value, Mapping):
        result = {str(key): _remove_exhausted_routes(item) for key, item in value.items()}
        exhausted = any(
            isinstance(item, Mapping)
            and type(item.get("remaining")) is int
            and item["remaining"] <= 0
            for key, item in value.items()
            if str(key).lower() in {"probe_budget", "refresh_budget", "route_budget", "probe", "refresh"}
        )
        if exhausted:
            for key in (
                    "offered_routes", "available_routes", "refresh_routes", "probe_routes",
                    "refresh_action", "probe_action", "next_action"):
                result.pop(key, None)
        return result
    if _is_sequence(value):
        return [_remove_exhausted_routes(item) for item in value]
    return value


def _line_entries(text: str) -> list[str]:
    values: list[str] = []
    for line in text.splitlines():
        match = _CLAUSE_RE.match(line)
        if not match:
            continue
        meaning = _public_meaning(match.group("meaning"))
        if meaning:
            values.append(f"{match.group('id')} — {meaning}")
    return list(dict.fromkeys(values))


def _public_green(frame: Mapping[str, object], text: str) -> bool:
    checks = frame.get("public_checks")
    if isinstance(checks, Mapping):
        status = _text(checks.get("state", checks.get("status", ""))).upper()
        if status in {"PASSED", "PASS", "GREEN", "SATISFIED", "CERTIFIED"}:
            return True
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping) and bool(terminal.get("green")):
        return True
    return bool(re.search(r"public (?:tests?|checks?) (?:passed|green)\b", text, re.IGNORECASE))


def _contains_protected(frame: Mapping[str, object], text: str) -> bool:
    upper = text.upper()
    for line in upper.splitlines():
        if "OPEN COUNTEREXAMPLES" in line and "[]" in line:
            continue
        if any(marker in line for marker in _PROTECTED_MARKERS):
            if "COUNTEREXAMPLE" in line:
                tail = line.split("COUNTEREXAMPLE", 1)[1].lstrip("S :")
                if tail.startswith(("[]", "{}", "NONE", "0")):
                    continue
            return True

    def walk(value: object, key: str = "") -> bool:
        key_upper = key.upper()
        if key_upper in {"COUNTEREXAMPLE", "COUNTEREXAMPLES", "NON_WAIVABLE", "NON-WAIVABLE", "VIOLATIONS"}:
            return value not in (None, False, "", [], {})
        if isinstance(value, Mapping):
            return any(walk(item, str(name)) for name, item in value.items())
        if _is_sequence(value):
            return any(walk(item, key) for item in value)
        return False

    return walk(frame)


def _effects_settled(frame: Mapping[str, object], text: str) -> bool:
    negative_keys = {"open_effects", "unsettled_effects", "pending_effects", "unknown_effects", "unresolved_effects"}

    def walk(value: object, key: str = "") -> bool:
        key_lower = key.lower()
        if key_lower in negative_keys and value not in (None, False, "", [], {}):
            return False
        if isinstance(value, Mapping):
            return all(walk(item, str(name)) for name, item in value.items())
        if _is_sequence(value):
            return all(walk(item, key) for item in value)
        return True

    if not walk(frame):
        return False
    return not bool(re.search(r"(?:unresolved|unknown) effect", text, re.IGNORECASE))


def _split_stateframe(content: str) -> tuple[dict[str, object] | None, str]:
    if not content.startswith(_STATE_PREFIX):
        return None, content
    boundary = content.find("\n[", len(_STATE_PREFIX))
    body = content[len(_STATE_PREFIX):] if boundary == -1 else content[len(_STATE_PREFIX):boundary]
    suffix = "" if boundary == -1 else content[boundary + 1:]
    try:
        value = json.loads(body)
    except (TypeError, ValueError):
        return None, content
    return (value if isinstance(value, dict) else None), suffix


def _frame_state(payload: Mapping[str, object]) -> Mapping[str, object]:
    state = payload.get("state")
    if payload.get("schema") == "stateframe-f4/1" and isinstance(state, Mapping):
        return state
    return payload


def _frame_state_without_framefit5(
        payload: Mapping[str, object],
) -> tuple[dict[str, object], bool, object]:
    state = dict(_frame_state(payload))
    has_framefit5 = "framefit5" in state
    framefit5 = state.pop("framefit5", None)
    return state, has_framefit5, framefit5


def _drop_block(lines: list[str], index: int) -> int:
    index += 1
    while index < len(lines):
        line = lines[index]
        if line.startswith(("[", "<settled_state>", "[Current source:", "[Paged source:")):
            break
        if line.startswith("[/TOKENS") or line == "</settled_state>":
            return index + 1
        index += 1
    return index


def _sanitize_text(text: str, *, terminal: bool, open_items: Sequence[str]) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    index = 0
    exhausted_route_budget = bool(_EXHAUSTED_ROUTE_RE.search(text))
    while index < len(lines):
        line = lines[index]
        if line.strip().lower() == "[stateframe recovery view]":
            end = next((candidate for candidate in range(index + 1, len(lines))
                        if lines[candidate].strip().lower() == "[/stateframe recovery view]"), len(lines))
            block = "\n".join(lines[index:end + int(end < len(lines))])
            if _EXHAUSTED_ROUTE_RE.search(block):
                index = end + int(end < len(lines))
                continue
        if terminal and line.startswith(_TARGET_BLOCKS):
            index = _drop_block(lines, index)
            continue
        normalized = line.strip().lower()
        if exhausted_route_budget and any(marker in normalized for marker in (
                "available refresh action", "refresh action=", "probe action=",
                "offered_routes", "available_routes", "refresh_routes", "probe_routes",
                "next_action", "next action:")):
            index += 1
            continue
        if normalized.startswith(_STALE_LINES):
            index += 1
            continue
        # Completion-owner notices are host-authored clearing instructions.  A
        # typed probe may carry an obligation id, but it is still actionable
        # owner output and must not be discarded by the generic bare-id filter.
        if _owner_gate_actionable(line):
            kept.append(line)
            index += 1
            continue
        if "no available host tool can establish" in normalized or "no_bound_check" in normalized:
            index += 1
            continue
        if "ambiguous_contract" in normalized and "what is missing" not in normalized:
            index += 1
            continue
        if _ID_RE.search(line):
            clause = _CLAUSE_RE.match(line)
            if clause is None or _public_meaning(clause.group("meaning")) is None:
                # Do not leave a bare obligation/check identifier in the
                # model view; without its public meaning it is not actionable.
                index += 1
                continue
        if any(marker.lower() in normalized for marker in ("structural refactor", "move the calculation", "move correct logic", "inline the calculation")):
            index += 1
            continue
        kept.append(line)
        index += 1
    result = "\n".join(kept).strip()
    if terminal:
        return result
    if open_items and "Submit now: call done" not in result and "open, uncertifiable" not in result:
        result = (result + "\n" if result else "") + "Open, uncertifiable: [" + "; ".join(open_items) + "]"
    return result


def _sanitize_with_owned_evidence_pin(
        text: str, *, terminal: bool, open_items: Sequence[str], owned_evidence_pin: str | None,
) -> str:
    """Sanitize advisory text while carrying one typed host block byte-for-byte."""
    if (not isinstance(owned_evidence_pin, str)
            or not owned_evidence_pin.startswith("[EVIDENCEPIN]")
            or not owned_evidence_pin.endswith("[/EVIDENCEPIN]")
            or owned_evidence_pin not in text):
        return _sanitize_text(text, terminal=terminal, open_items=open_items)
    # Only the exact block registered by the completion owner is opaque.  A
    # model/tool-authored lookalike still flows through ordinary hygiene and is
    # rejected by the final exact-one-pin contract if it reaches the wire.
    return owned_evidence_pin.join(
        _sanitize_text(part, terminal=terminal, open_items=open_items)
        for part in text.split(owned_evidence_pin)
    )


def _open_label(open_items: Sequence[str]) -> str:
    return "; ".join(open_items) if open_items else "uncertifiable host coverage"


def _owner_gate_actionable(text: str) -> bool:
    """Keep an owner refusal in the live frame instead of projecting done-only."""
    lowered = text.lower()
    return any(marker in lowered for marker in (
        "[completion owner] done refused:",
        "[completion owner] completion held:",
        "[completion owner] next legal action:",
    ))


def _strip_terminal_offer(text: str) -> str:
    return _FINAL_SUBMIT_RE.sub("", text).strip()


def _project_frame(
        frame: Mapping[str, object], *, text: str,
        contract: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], bool, list[str]]:
    source = copy.deepcopy(dict(frame))
    actionable = _structured_entries(source, _BLOCKER_KEYS)
    open_items = list(dict.fromkeys((*_structured_open(source), *_line_entries(text))))
    raw_result = _remove_exhausted_routes(_remove_bare_ids(source))
    result = raw_result if isinstance(raw_result, dict) else {}
    protected = _contains_protected(result, text)
    green = _public_green(result, text)
    settled = _effects_settled(result, text)
    if isinstance(contract, Mapping):
        terminal = str(contract.get("route", "")) == "submit"
    else:
        terminal = (green and settled and not actionable and not protected
                    and not _owner_gate_actionable(text))

    retained_blockers = result.get("blockers")
    for key in _OPEN_KEYS:
        result.pop(key, None)
    result.pop("blockers", None)
    result.pop("blocking_items", None)
    if protected:
        if actionable:
            result["blockers"] = actionable
        elif retained_blockers is not None:
            result["blockers"] = retained_blockers
    result["frame_hygiene"] = {
        "schema": SCHEMA,
        "actionable_blockers": actionable,
        "open_uncertifiable": list(open_items),
        "terminal_submit": terminal,
        "host_authority": "unchanged",
    }
    if terminal:
        for key in (
                "done_permission", "completion", "completion_status", "completion_verdict",
                "completion_decision", "submission", "submission_status", "advisories"):
            result.pop(key, None)
        result["submit_option"] = {
            "state": "OPEN",
            "uncertified": True,
            "open": list(open_items),
            "authority": "host gate unchanged",
        }
    elif not actionable and not protected:
        result.pop("done_permission", None)
        result["submit_option"] = {
            "state": "NOT_READY",
            "open": list(open_items),
            "authority": "host gate unchanged",
        }
    return result, terminal, open_items


def project_messages(
        messages: Sequence[Mapping[str, object]],
        *,
        contract: Mapping[str, object] | None = None,
        owned_evidence_pin: str | None = None,
) -> list[dict[str, object]]:
    """Project final model-facing content; never mutates the input list."""
    if not enabled():
        return [dict(message) for message in messages]
    if owned_evidence_pin is None and isinstance(contract, Mapping):
        candidate_pin = contract.get("evidence_pin")
        if isinstance(candidate_pin, str):
            owned_evidence_pin = candidate_pin

    from kernel_harness_card import ensure_system_prefix
    messages = ensure_system_prefix(messages)
    raw_text = "\n".join(
        str(message.get("content") or "") for message in messages if isinstance(message, Mapping)
    )
    state_frames: list[tuple[int, dict[str, object], str]] = []
    for index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            frame, suffix = _split_stateframe(content)
            if frame is not None:
                state_frames.append((index, frame, suffix))
    combined_frame: dict[str, object] = {}
    for _, frame, _ in state_frames:
        projected_input, _, _ = _frame_state_without_framefit5(frame)
        combined_frame.update(projected_input)
    _, terminal, open_items = _project_frame(combined_frame, text=raw_text, contract=contract)
    open_label = _open_label(open_items)
    projected: list[dict[str, object]] = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if not isinstance(content, str):
            projected.append(item)
            continue
        # Source blocks are host-delivered evidence, not advisory panels.  They
        # must remain byte-exact for the request preflight to bind them to the
        # current workspace bytes; hygiene is limited to advisory content.
        if _is_source_block(content):
            projected.append(item)
            continue
        frame, suffix = _split_stateframe(content)
        if frame is not None:
            projection_input, has_framefit5, framefit5 = _frame_state_without_framefit5(frame)
            projected_state, _, _ = _project_frame(
                projection_input, text=raw_text, contract=contract,
            )
            if has_framefit5:
                projected_state["framefit5"] = copy.deepcopy(framefit5)
            if frame.get("schema") == "stateframe-f4/1" and isinstance(frame.get("state"), Mapping):
                hygiene = projected_state.pop("frame_hygiene", None)
                projected_frame = copy.deepcopy(frame)
                projected_frame["state"] = projected_state
                if hygiene is not None:
                    projected_frame["frame_hygiene"] = hygiene
            else:
                projected_frame = projected_state
            item["content"] = serialize_stateframe_payload(projected_frame)
            if suffix:
                item["content"] += "\n" + _sanitize_with_owned_evidence_pin(
                    suffix,
                    terminal=terminal,
                    open_items=(open_items if not terminal else ()),
                    owned_evidence_pin=owned_evidence_pin,
                )
            if (not terminal
                    and (isinstance(contract, Mapping)
                         or _owner_gate_actionable(raw_text))):
                item["content"] = _strip_terminal_offer(str(item["content"]))
        else:
            item["content"] = _sanitize_with_owned_evidence_pin(
                content,
                terminal=terminal,
                open_items=(open_items if not terminal else ()),
                owned_evidence_pin=owned_evidence_pin,
            )
            if (not terminal
                    and (isinstance(contract, Mapping)
                         or _owner_gate_actionable(raw_text))):
                item["content"] = _strip_terminal_offer(str(item["content"]))
        projected.append(item)

    if terminal:
        if isinstance(contract, Mapping):
            line = str(contract.get("terminal_line") or "").strip()
            if not line:
                raise ValueError("submit contract has no terminal line")
        else:
            offer = _FINAL_SUBMIT_RE.search(raw_text)
            line = offer.group(0) if offer is not None else (
                "Final step: call done now to submit this revision "
                f"(uncertified; open: [{open_label}]). No other action is available."
            )
        # The terminal owner has already made the host decision.  Keeping the
        # prior denial, stale binding, or refresh narrative creates a second
        # instruction and spends the reserved response on the wrong route.
        system_messages = [
            item for item in projected if item.get("role") == "system"
        ]
        state_messages = [
            item for item in projected
            if str(item.get("content") or "").startswith(_STATE_PREFIX)
        ]
        source_messages = [
            item for item in projected
            if _is_source_block(str(item.get("content") or ""))
        ]
        # Terminal projection is still a frame-arm provider request.  Keep one
        # projected StateFrame in the shared request assembler; dropping it
        # here turns the truthful terminal offer into an invalid side-channel
        # request and makes release preflight reject exhaustion/diagnosis.
        state_message = state_messages[:1]
        return [*system_messages, *state_message, *source_messages,
                {"role": "user", "content": line}]
    return projected


__all__ = ["MODE_ENV", "SCHEMA", "enabled", "mode", "project_messages"]
