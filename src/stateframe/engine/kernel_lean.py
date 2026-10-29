"""Plain-loop request shaping with a small, opt-in stack of verified levers."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kernel_tool_reply import (
    Outcome, copy_tool_reply_message, enabled as tool_reply_enabled,
    outcome_from_receipt, tool_reply,
)


MODE_ENV = "STATEFRAME_LEAN"
STALL_ENV = "STATEFRAME_STALL_OBSERVATION"
PLAIN_SYSTEM_PROMPT = (
    "You are a software engineer fixing a bug in a repository.\n"
    "The repository is at /workspace. Use the tools to inspect the code, make the fix, and run tests.\n"
    "Make the smallest durable production-quality change that repairs the described behavior.\n"
    "When you are finished, call done with a short summary."
)
PLAIN_TOOL_NAMES = frozenset({
    "list_files", "read", "grep", "str_replace", "edit", "file_create",
    "write", "run_tests", "done",
})
MAX_TOOL_RESULT_BYTES = 24_000
MAX_TOOL_RESULT_CHARS = 12_000
MAX_STACK_NOTICE_TOKENS = 100
MAX_READ_PAGE_CHARS = 12_000
# Source excerpts need room for the immutable transcript, final wire facts, and
# JSON/tool envelope bytes not represented by source-window character counts.
LEAN_SOURCE_WINDOW_HEADROOM_BYTES = 4_096
# The plain harness bounds source output by characters (baseline_harness.py:96-99).
# Keep that same character ceiling while allowing focused reads to span 400 lines.
MAX_READ_PAGE_LINES = 400
MAX_GREP_PAGE_SIZE = 120
MAX_LISTING_ROWS = 120
_KNOWN_TOOL_NAMES = frozenset({
    *PLAIN_TOOL_NAMES, "insert_before", "authored_check", "read_check_receipt",
    "recall", "work", "work_update", "note_update", "probe", "reproduce",
    "test", "replace_module", "decide", "where_defined", "who_calls",
    "references", "outline", "dependency_fact", "read_pointer", "abstain",
})
_EVIDENCE_PIN_RE = re.compile(
    r"\[EVIDENCEPIN\].*?(?:\[/EVIDENCEPIN\]|$)", re.DOTALL,
)


def _switch(name: str, environ: Mapping[str, str] | None, default: str = "off") -> bool:
    source = os.environ if environ is None else environ
    value = str(source.get(name, default) or default).strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{name} must be on or off, got {value!r}")
    return value == "on"


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return strict STATEFRAME_LEAN mode; malformed values never silently disable it."""
    return _switch(MODE_ENV, environ)


def history_compaction_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """History summaries are incompatible with LEAN's append-only transcript."""
    source = os.environ if environ is None else environ
    if enabled(source):
        return False
    return str(source.get("REACT_HISTORY_COMPACTION", "off")).strip().lower() in {
        "on", "1", "true", "yes",
    }


def should_deliver_zero_progress_advisory(
        step: object, edits: object, checks: object, *,
        interventions: object, lean_mode: bool,
) -> bool:
    """Select legacy zero-progress checkpoints for an advisory only."""
    return (
        not lean_mode
        and bool(interventions)
        and type(step) is int
        and step in {50, 100, 150}
        and type(edits) is int
        and edits == 0
        and type(checks) is int
        and checks == 0
    )


def stall_observation_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _switch(STALL_ENV, environ)


@dataclass(frozen=True)
class LeanStackSlot:
    name: str
    environment: str
    default: str
    renderer: str
    experimental: bool = False


# Add a lever here once; readers, snapshots, and tests share this registry.
LEAN_STACK: tuple[LeanStackSlot, ...] = (
    LeanStackSlot(
        "restate_requirement", "STATEFRAME_RESTATE_REQUIREMENT",
        "auto", "restate_requirement",
    ),
    LeanStackSlot(
        "wide_target_window", "STATEFRAME_LEAN_STACK_WIDE_TARGET_WINDOW",
        "on", "wide_target_window",
    ),
    LeanStackSlot(
        "sibling_site_facts", "STATEFRAME_LEAN_STACK_SIBLING_SITE_FACTS",
        "off", "sibling_site_facts", experimental=True,
    ),
)
_STACK_BY_NAME = {slot.name: slot for slot in LEAN_STACK}
_STACK_BY_ENV = {slot.environment: slot for slot in LEAN_STACK}


def stack_enabled(
        name: str, environ: Mapping[str, str] | None = None, *,
        runtime: object | None = None,
) -> bool:
    slot = _STACK_BY_NAME.get(name)
    if slot is None:
        raise ValueError(f"unknown LEAN stack slot: {name}")
    # Validate both controls even when the parent feature is off.
    if slot.renderer == "restate_requirement":
        if runtime is None:
            from kernel_restate_requirement import mode as restate_mode
            return restate_mode(environ) == "on"
        from kernel_restate_requirement import enabled as restate_enabled
        return restate_enabled(runtime, environ)
    lean_on = enabled(environ)
    slot_on = _switch(slot.environment, environ, slot.default)
    return lean_on and slot_on


def restate_requirement_stack_enabled(
        environ: Mapping[str, str] | None = None, *, runtime: object | None = None,
) -> bool:
    return stack_enabled("restate_requirement", environ, runtime=runtime)


def wide_target_window_stack_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return stack_enabled("wide_target_window", environ)


def sibling_site_facts_stack_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return stack_enabled("sibling_site_facts", environ)


def stack_snapshot(
        environ: Mapping[str, str] | None = None, *,
        runtime: object | None = None,
) -> dict[str, dict[str, object]]:
    source = os.environ if environ is None else environ
    lean_on = enabled(source)
    snapshot: dict[str, dict[str, object]] = {}
    for slot in LEAN_STACK:
        if slot.renderer == "restate_requirement":
            from kernel_restate_requirement import enabled as restate_enabled, mode as restate_mode
            configured = restate_mode(source)
            if runtime is not None:
                effective = "on" if restate_enabled(runtime, source) else "off"
            elif configured == "auto" and lean_on:
                effective = "scope_dependent"
            else:
                effective = "on" if configured == "on" else "off"
            snapshot[slot.name] = {
                "environment": slot.environment,
                "configured": configured,
                "effective": effective,
                "default": slot.default,
                "trigger": "LEAN and at least two declared public files",
                "experimental": slot.experimental,
            }
            continue
        configured_on = _switch(slot.environment, source, slot.default)
        snapshot[slot.name] = {
            "environment": slot.environment,
            "configured": "on" if configured_on else "off",
            "effective": "on" if lean_on and configured_on else "off",
            "default": slot.default,
            "experimental": slot.experimental,
        }
    return snapshot


def runtime_lean_snapshot(
        environ: Mapping[str, str] | None = None, *,
        runtime: object | None = None,
) -> dict[str, object]:
    lean_on = enabled(environ)
    disabled = (
        "advisory_panels", "host_ledgers", "owner_menu_routes", "idle_governor",
        "repeat_gate", "done_only_collapse", "kernel_tool_surface",
        "kernel_vacuity_sentinel", "action_receipt_amended", "request_frame_organs",
        "memory_isolation", "ticket_identity", "notice_dedup", "frame_edit_integrity",
        "f4_and_hybrid_presentation", "greenlock_empty_menu_authority",
        "tool_call_loop_breaker_notice", "read_once_prompt_note", "request_budget_guidance",
        "wire_read_contract_model_capsule", "unbounded_tool_output",
    ) if lean_on else ()
    return {
        "enabled": lean_on,
        "mode": "on" if lean_on else "off",
        "disabled_subsystems": list(disabled),
        "stack": stack_snapshot(environ, runtime=runtime),
        "stall_observation": "on" if stall_observation_enabled(environ) else "off",
    }


def tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(tool.get("name", ""))


def plain_tool_surface(tools: Sequence[object]) -> list[dict[str, object]]:
    """Project onto plain coding tools while preserving compatible dispatch schemas."""
    selected = [copy.deepcopy(dict(tool)) for tool in tools
                if isinstance(tool, Mapping) and tool_name(tool) in PLAIN_TOOL_NAMES]
    for tool in selected:
        function = tool.get("function")
        parameters = function.get("parameters") if isinstance(function, Mapping) else None
        properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
        if not isinstance(function, dict) or not isinstance(properties, dict):
            continue
        name = str(function.get("name", ""))
        if name == "read":
            if isinstance(properties.get("max_chars"), dict):
                properties["max_chars"].update(
                    {"maximum": MAX_READ_PAGE_CHARS, "default": MAX_READ_PAGE_CHARS}
                )
            if isinstance(properties.get("max_lines"), dict):
                properties["max_lines"].update(
                    {"maximum": MAX_READ_PAGE_LINES, "default": MAX_READ_PAGE_LINES}
                )
        elif name == "grep" and isinstance(properties.get("page_size"), dict):
            properties["page_size"].update({"maximum": MAX_GREP_PAGE_SIZE,
                                             "default": MAX_GREP_PAGE_SIZE})
        elif name == "list_files" and isinstance(properties.get("limit"), dict):
            properties["limit"].update({"maximum": MAX_LISTING_ROWS,
                                         "default": MAX_LISTING_ROWS})
        if name in {"read", "grep", "list_files", "run_tests"}:
            function["description"] = (
                str(function.get("description", "")).rstrip()
                + " LEAN caps each result page so the next request stays bounded."
            )
    return selected


def bound_plain_tool_arguments(name: str, arguments: object) -> dict[str, object]:
    """Clamp page arguments even when an endpoint ignores the advertised schema."""
    bounded = copy.deepcopy(dict(arguments)) if isinstance(arguments, Mapping) else {}
    if name == "read":
        max_chars = bounded.get("max_chars", MAX_READ_PAGE_CHARS)
        bounded["max_chars"] = min(
            MAX_READ_PAGE_CHARS, max(512, int(max_chars))
        ) if str(max_chars).lstrip("+-").isdigit() else MAX_READ_PAGE_CHARS
        explicit_end = bounded.get("end_line") not in (None, "")
        explicit_max_lines = bounded.get("max_lines") not in (None, "")
        if explicit_end and not explicit_max_lines:
            # The line range is the model's request; the dispatcher applies the
            # lean page budget internally so these public coordinates stay intact.
            bounded.pop("max_lines", None)
        else:
            max_lines = bounded.get("max_lines", MAX_READ_PAGE_LINES)
            bounded["max_lines"] = min(
                MAX_READ_PAGE_LINES, max(1, int(max_lines))
            ) if str(max_lines).lstrip("+-").isdigit() else MAX_READ_PAGE_LINES
    elif name == "grep":
        page_size = bounded.get("page_size", MAX_GREP_PAGE_SIZE)
        bounded["page_size"] = min(MAX_GREP_PAGE_SIZE, max(1, int(page_size))) \
            if str(page_size).isdigit() else MAX_GREP_PAGE_SIZE
    elif name == "list_files":
        limit = bounded.get("limit", MAX_LISTING_ROWS)
        bounded["limit"] = min(MAX_LISTING_ROWS, max(1, int(limit))) \
            if str(limit).isdigit() else MAX_LISTING_ROWS
    return bounded


def bound_tool_result(content: object) -> str:
    """Cap a host result without presenting clipped source as an exact source proof."""
    text = str(content or "")
    data = text.encode("utf-8")
    if len(text) <= MAX_TOOL_RESULT_CHARS and len(data) <= MAX_TOOL_RESULT_BYTES:
        return text
    if any(marker in text for marker in _SOURCE_MARKERS):
        return (
            "[LEAN read result exceeded the bounded page size. Request a narrower "
            "line range; the host retained the original source receipt.]"
        )
    marker = f"\n[LEAN tool output bounded; {len(data)} UTF-8 bytes, middle omitted]\n"
    remaining_chars = MAX_TOOL_RESULT_CHARS - len(marker)
    remaining_bytes = MAX_TOOL_RESULT_BYTES - len(marker.encode("utf-8"))
    head_chars = min(remaining_chars // 2, remaining_bytes // 2)
    tail_chars = min(remaining_chars - head_chars, remaining_bytes - head_chars)
    head = text[:head_chars]
    tail = text[-tail_chars:] if tail_chars else ""
    # UTF-8-heavy tails can exceed the byte ceiling despite fitting the char cap.
    while len((head + marker + tail).encode("utf-8")) > MAX_TOOL_RESULT_BYTES and tail:
        tail = tail[1:]
    while len((head + marker + tail).encode("utf-8")) > MAX_TOOL_RESULT_BYTES and head:
        head = head[:-1]
    return head + marker + tail


class BoundedToolMessages(list[dict[str, object]]):
    """List-compatible transcript that bounds tool output at the append boundary."""

    def append(self, message: object) -> None:
        if isinstance(message, Mapping) and message.get("role") == "tool":
            copied = copy_tool_reply_message(message)
            copied["content"] = bound_tool_result(copied.get("content"))
            super().append(copied)
            return
        super().append(message)

    def extend(self, messages: Sequence[object]) -> None:
        for message in messages:
            self.append(message)


def reconcile_tool_call_results(
        messages: Sequence[Mapping[str, object]],
        journal_records: Sequence[Mapping[str, object]] = (),
        *, settled_results: Sequence[Mapping[str, object]] = (),
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """Return an ordered transcript, restoring only captured or journaled results."""
    n3_reply_on = tool_reply_enabled()
    copied: list[dict[str, object]] = []
    calls: dict[str, Mapping[str, object]] = {}
    existing_results: dict[str, dict[str, object]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"tool transcript message {index} is not an object")
        row = copy.deepcopy(message)
        if not isinstance(row, dict):
            row = copy.deepcopy(dict(message))
        role = row.get("role")
        if role == "assistant" and "tool_calls" in row:
            raw_calls = row.get("tool_calls")
            if not isinstance(raw_calls, list) or not raw_calls:
                raise ValueError(f"assistant message {index} has invalid tool_calls")
            for call in raw_calls:
                if not isinstance(call, Mapping):
                    raise ValueError(f"assistant message {index} has an invalid tool call")
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id or call_id in calls:
                    raise ValueError("tool call ids must be non-empty and unique")
                calls[call_id] = call
        elif role == "tool":
            result_id = row.get("tool_call_id")
            if not isinstance(result_id, str) or not result_id:
                raise ValueError("tool result ids must be non-empty and unique")
            previous = existing_results.get(result_id)
            if previous is not None:
                if not n3_reply_on:
                    raise ValueError("tool result ids must be non-empty and unique")
                previous_content = previous.get("content")
                current_content = row.get("content")
                if (not isinstance(previous_content, str) or not previous_content.strip()) and (
                        isinstance(current_content, str) and current_content.strip()):
                    existing_results[result_id] = row
            else:
                existing_results[result_id] = row
        elif "tool_call_id" in row or "tool_calls" in row:
            raise ValueError(f"message {index} has tool fields on an invalid role")
        copied.append(row)

    orphan_results = set(existing_results) - set(calls)
    if orphan_results:
        raise ValueError("tool results have no matching assistant call")

    prepared_results: dict[str, dict[str, object]] = {}
    for index, result in enumerate(settled_results):
        if not isinstance(result, Mapping) or result.get("role") != "tool":
            raise ValueError(f"prepared tool result {index} is not a tool message")
        result_id = result.get("tool_call_id")
        if (not isinstance(result_id, str) or not result_id
                or result_id in prepared_results or result_id not in calls):
            raise ValueError("prepared tool results must have unique matching call ids")
        prepared_copy = copy.deepcopy(result)
        prepared_results[result_id] = (
            prepared_copy if isinstance(prepared_copy, dict) else copy.deepcopy(dict(result))
        )

    closed_by_step: dict[str, Mapping[str, object]] = {}
    native_call_steps: dict[str, str] = {}
    closed_by_call: dict[str, Mapping[str, object]] = {}
    for event in journal_records:
        if not isinstance(event, Mapping):
            continue
        kind = event.get("kind")
        if kind == "native_batch_action_closed":
            call_id = event.get("tool_call_id")
            step_id = event.get("step_record_id")
            if isinstance(call_id, str) and isinstance(step_id, str):
                native_call_steps[call_id] = step_id
        elif kind == "action_outcome_closed":
            record = event.get("record")
            if not isinstance(record, Mapping):
                continue
            step_id = record.get("step_record_id")
            if isinstance(step_id, str):
                closed_by_step[step_id] = record
            call_id = record.get("tool_call_id")
            if isinstance(call_id, str):
                closed_by_call[call_id] = record
    for call_id, step_id in native_call_steps.items():
        record = closed_by_step.get(step_id)
        if record is not None:
            closed_by_call.setdefault(call_id, record)

    recovered_results: dict[str, dict[str, object]] = {}
    for call_id, call in calls.items():
        existing = existing_results.get(call_id)
        prepared = prepared_results.get(call_id)
        existing_content = existing.get("content") if existing is not None else None
        prepared_content = prepared.get("content") if prepared is not None else None
        existing_nonempty = isinstance(existing_content, str) and bool(existing_content.strip())
        prepared_nonempty = isinstance(prepared_content, str) and bool(prepared_content.strip())
        prior_result = (
            existing if existing_nonempty else prepared if prepared_nonempty
            else existing if existing is not None else prepared
        )
        if prior_result is not None:
            content = prior_result.get("content")
            if not (n3_reply_on and (not isinstance(content, str) or not content.strip())):
                continue
        record = closed_by_call.get(call_id)
        function = call.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        observation = record.get("observation") if isinstance(record, Mapping) else None
        if (record is None or not isinstance(name, str) or record.get("tool") != name
                or not isinstance(observation, str)):
            if tool_reply_enabled():
                reply_name = name if isinstance(name, str) and name else "tool"
                recovered_results[call_id] = tool_reply(
                    call_id, reply_name, Outcome.ERROR, "",
                    receipt={"cause": "empty_reply" if prior_result is not None else "missing_reply"},
                )
            continue
        outcome, receipt = outcome_from_receipt(record, call)
        recovered_results[call_id] = tool_reply(
            call_id, name, outcome, observation, receipt=receipt,
        )

    ordered: list[dict[str, object]] = []
    consumed_results: set[str] = set()
    recovered_ids: list[str] = []
    for message in copied:
        if message.get("role") == "tool":
            continue
        ordered.append(message)
        if message.get("role") != "assistant" or "tool_calls" not in message:
            continue
        raw_calls = message.get("tool_calls")
        for call in raw_calls if isinstance(raw_calls, list) else ():
            call_id = str(call.get("id"))
            result = existing_results.get(call_id)
            if n3_reply_on and result is not None:
                content = result.get("content")
                if not isinstance(content, str) or not content.strip():
                    prepared = prepared_results.get(call_id)
                    prepared_content = prepared.get("content") if prepared is not None else None
                    if isinstance(prepared_content, str) and prepared_content.strip():
                        result = prepared
                    else:
                        result = recovered_results.get(call_id, result)
            if result is None:
                result = prepared_results.get(call_id)
            if n3_reply_on and result is not None:
                content = result.get("content")
                if not isinstance(content, str) or not content.strip():
                    result = recovered_results.get(call_id, result)
            if result is None:
                result = recovered_results.get(call_id)
                marker = getattr(result, "_stateframe_tool_reply_receipt", None)
                if result is not None and not (
                        isinstance(marker, Mapping) and marker.get("cause") == "missing_reply"):
                    recovered_ids.append(call_id)
            elif call_id not in existing_results or result is recovered_results.get(call_id):
                marker = getattr(result, "_stateframe_tool_reply_receipt", None)
                if not (isinstance(marker, Mapping) and marker.get("cause") == "missing_reply"):
                    recovered_ids.append(call_id)
            if result is None:
                raise ValueError(f"tool call {call_id!r} has no settled result")
            ordered.append(copy.deepcopy(result))
            consumed_results.add(call_id)
    if consumed_results != set(existing_results) | set(prepared_results) | set(recovered_results):
        raise ValueError("tool transcript contains an unpaired result")
    return ordered, tuple(recovered_ids)


_SOURCE_PATH_PATTERNS = (
    re.compile(r"(?m)^\[(?:Current|Working) source: ([^\]\n]+)\]"),
    re.compile(r'"path"\s*:\s*"([^"\n]+)".*?"revision"\s*:', re.S),
)
_SOURCE_MARKERS = (
    "[Current source:", "[Working source:", "[MODEL_SOURCE_WINDOW]",
    "[Model source window]", "[kernel] current source", "[Source window",
)
_TRUTH_HEADERS = (
    "[Deferred required source paths]", "[Host fact:", "[WIRE_TRUTH_FACT ",
    "[Host observation]", "[kernel-governor] REFUSED:", "[Host source proof]",
)
_ADVISORY_HEADERS = (
    "[StateFrame]", "[Files in frame]", "[StateFrame advisory sidecar]",
    "[BOOT CAPSULE", "[Decisive factor]", "[Unified stop authority]",
    "[VERIFICATION CERTIFICATE]", "[False-completion coverage",
    "[Passive conclusions", "[Settled findings", "[WORKING_VIEW",
    "[Repository signals v1", "[YOUR NOTES |", "[COORDINATED CHANGE LEDGER]",
    "[unified ledger]", "[REASONING LEDGER]", "[Pattern cards]",
    "[Decision cache]", "[Unknowns", "[Read once]", "[Read reuse]",
    "[WIRE_READ_CONTRACT]", "[Capability manifest]", "[r44",
)


def source_paths_in_content(content: object) -> tuple[str, ...]:
    text = str(content or "")
    found: list[str] = []
    for pattern in _SOURCE_PATH_PATTERNS:
        for match in pattern.finditer(text):
            path = match.group(1).replace("\\", "/").strip()
            if path and path not in found:
                found.append(path)
    return tuple(found)


def is_source_content(content: object, path: str | None = None) -> bool:
    text = str(content or "")
    if not any(marker in text for marker in _SOURCE_MARKERS):
        return False
    return path is None or path.replace("\\", "/") in source_paths_in_content(text)


def _source_span_key(content: object) -> str:
    text = str(content or "")
    try:
        from kernel_request_source_window import parse_model_source_window, parse_window

        parsed = parse_window(text) or parse_model_source_window(text)
    except (ImportError, TypeError, ValueError):
        parsed = None
    if isinstance(parsed, Mapping):
        metadata = parsed.get("metadata", parsed)
        if isinstance(metadata, Mapping):
            path = str(metadata.get("path") or "")
            revision = str(metadata.get("revision") or metadata.get("sha256") or "")
            raw_ranges = metadata.get("ranges")
            try:
                ranges = tuple(
                    (int(item.get("start_line", 0)), int(item.get("end_line", 0)))
                    for item in raw_ranges if isinstance(item, Mapping)
                ) if isinstance(raw_ranges, Sequence) else ()
            except (TypeError, ValueError):
                ranges = ()
            if path and revision and ranges:
                return f"{path}\0{revision}\0{ranges!r}"
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def _is_truth_notice(content: object) -> bool:
    text = str(content or "")
    return text.startswith(_TRUTH_HEADERS)


def is_truth_content(content: object) -> bool:
    """Identify non-evictable host truth separately from editable source text."""
    return _is_truth_notice(content)


def advisory_kind(content: object) -> str | None:
    text = str(content or "")
    for header in _ADVISORY_HEADERS:
        if text.startswith(header):
            return header.strip("[] ").split(maxsplit=1)[0].lower()
    return None


def drop_advisory_messages(
    messages: Sequence[Mapping[str, object]], *, declared_source_paths: Sequence[str] = (),
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """Drop known host panels, keeping the initial task, history, truth, and declared source."""
    declared = {str(path).replace("\\", "/").strip("/") for path in declared_source_paths}
    kept: list[dict[str, object]] = []
    dropped: list[str] = []
    for index, message in enumerate(messages):
        copied = copy_tool_reply_message(message)
        content = copied.get("content")
        if index < 2 or copied.get("role") in {"assistant", "tool"}:
            kept.append(copied)
            continue
        kind = advisory_kind(content)
        source_paths = source_paths_in_content(content)
        if source_paths and any(path in declared for path in source_paths):
            kept.append(copied)
        elif _is_truth_notice(content):
            kept.append(copied)
        elif kind is not None:
            dropped.append(kind)
        else:
            kept.append(copied)
    return kept, tuple(dropped)


def plain_loop_messages(
    messages: Sequence[Mapping[str, object]], *, declared_source_paths: Sequence[str] = (),
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """Keep the task and append-only assistant/tool transcript; fold current host facts."""
    from kernel_restate_requirement import MESSAGE as RESTATE_MESSAGE

    source_paths = {str(path).replace("\\", "/").strip("/")
                    for path in declared_source_paths}
    task_index = next(
        (index for index, message in enumerate(messages)
         if isinstance(message, Mapping)
         and message.get("role") == "user"
         and message.get("content") != RESTATE_MESSAGE),
        None,
    )
    system_prefix_indices = {
        index for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "system"
        and (task_index is None or index < task_index)
    }
    kept: list[dict[str, object]] = []
    dropped: list[str] = []
    seen_user_source_spans: set[str] = set()
    for index, original in enumerate(messages):
        if not isinstance(original, Mapping):
            dropped.append("invalid_message")
            continue
        message = copy.deepcopy(dict(original))
        role = message.get("role")
        if (index in system_prefix_indices or index == task_index
                or (role == "user" and message.get("content") == RESTATE_MESSAGE)
                or role in {"assistant", "tool"}):
            kept.append(message)
            continue
        content = message.get("content")
        paths = source_paths_in_content(content)
        source_is_declared = bool(paths and any(path in source_paths for path in paths))
        if source_is_declared and role == "user":
            span_key = _source_span_key(content)
            if span_key in seen_user_source_spans:
                dropped.append("duplicate_source_span")
                continue
            seen_user_source_spans.add(span_key)
        if source_is_declared or _is_truth_notice(content):
            kept.append(message)
            continue
        kind = advisory_kind(content)
        if kind is not None:
            dropped.append(kind)
        else:
            dropped.append("host_user_message")

    last_tool_index = next(
        (index for index in range(len(kept) - 1, -1, -1)
         if kept[index].get("role") == "tool"),
        -1,
    )
    if last_tool_index >= 0:
        tail_users = [message for message in kept[last_tool_index + 1:]
                      if message.get("role") == "user"]
        if tail_users:
            for message in tail_users:
                content = str(message.get("content") or "")
                if _is_truth_notice(content) or is_source_content(content):
                    _append_latest_tool_text(kept[:last_tool_index + 1], content)
                dropped.append(advisory_kind(content) or "trailing_host_user")
            kept = [message for index, message in enumerate(kept)
                    if index <= last_tool_index or message.get("role") != "user"]
    return kept, tuple(dict.fromkeys(dropped))


def _append_latest_tool_text(messages: list[dict[str, object]], text: str) -> bool:
    if not text.strip():
        return False
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = str(message.get("content") or "")
            if text in content:
                return False
            observation_marker = "\n\n[Host observation] "
            if observation_marker in content:
                before, observation = content.split(observation_marker, 1)
                message["content"] = (
                    before.rstrip() + "\n\n" + text + observation_marker + observation
                ).lstrip()
            else:
                message["content"] = (content.rstrip() + "\n\n" + text).lstrip()
            return True
    return False


def _append_latest_tool_source(messages: list[dict[str, object]], content: str) -> bool:
    """Keep proof-carrying source inside the existing tool result message."""
    if not content.strip():
        return False
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        current = str(message.get("content") or "")
        if content in current:
            return False
        message["content"] = (current.rstrip() + "\n\n" + content).lstrip()
        return True
    return False


def _notice_words(text: str, max_words: int = 100) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words - 1]) + " …"


def _bounded_stack_notice(text: str) -> str:
    """Keep each optional stack note within its measured token budget."""
    from kernel_context import token_estimate

    words = text.split()
    while words:
        candidate = " ".join(words)
        if token_estimate({"content": candidate}) <= MAX_STACK_NOTICE_TOKENS:
            return candidate
        words.pop()
    return ""


def render_sibling_site_facts(runtime: object) -> str | None:
    """Render at most 100 tokens of source-backed references in declared product files."""
    try:
        from kernel_request_source_window import (
            editable_product_paths, entry_symbol, literal_site_inventory,
        )
        root = Path(getattr(getattr(runtime, "owner", None), "root"))
        paths = editable_product_paths(runtime)
        symbol = entry_symbol(runtime).rsplit(".", 1)[-1]
        if not paths or not symbol:
            return None
        sites = literal_site_inventory(root, paths, (symbol,), max_sites=12)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    lines = [
        f"{site.path}:L{site.line} {site.text.strip()}"
        for site in sites if getattr(site, "role", "") == "executable"
    ]
    if not lines:
        return None
    return _bounded_stack_notice("[Sibling site facts] " + " | ".join(lines))


def stack_tail_blocks(
    runtime: object, environ: Mapping[str, str] | None = None, *,
    include_restate_requirement: bool = True,
) -> tuple[str, ...]:
    blocks: list[str] = []
    for slot in LEAN_STACK:
        if slot.renderer == "restate_requirement" and not include_restate_requirement:
            continue
        if not stack_enabled(slot.name, environ, runtime=runtime):
            continue
        if slot.renderer == "restate_requirement":
            from kernel_restate_requirement import MESSAGE
            blocks.append(_bounded_stack_notice(MESSAGE))
        elif slot.renderer == "sibling_site_facts":
            block = render_sibling_site_facts(runtime)
            if block:
                blocks.append(_bounded_stack_notice(block))
    return tuple(block for block in blocks if block)


def append_stack_notice(
    messages: Sequence[Mapping[str, object]], blocks: Sequence[str],
) -> list[dict[str, object]]:
    """Append optional stack notes inside the latest tool result, preserving its id."""
    copied = [copy.deepcopy(dict(message)) for message in messages]
    bounded = [_bounded_stack_notice(str(block)) for block in blocks if str(block).strip()]
    bounded = [block for block in bounded if block]
    if not bounded:
        return copied
    latest_tool = next((message for message in reversed(copied)
                        if message.get("role") == "tool"), None)
    if latest_tool is None:
        return copied
    notice = "[Host stack] " + " | ".join(bounded)
    content = str(latest_tool.get("content") or "")
    if notice not in content:
        latest_tool["content"] = content.rstrip() + "\n\n" + notice
    return copied


def sanitize_tool_references(
    messages: Sequence[Mapping[str, object]], tools: Sequence[object],
) -> tuple[list[dict[str, object]], int]:
    """Rewrite host prose that instructs use of a tool absent from this request."""
    offered = {tool_name(tool) for tool in tools}
    rendered: list[dict[str, object]] = []
    replacements = 0
    reference = re.compile(
        r"(?i)(?:use|call|invoke)\s+(?:the\s+)?`?([a-z][a-z0-9_]*)`?(?:\s+tool)?"
    )
    for index, message in enumerate(messages):
        if (isinstance(message, dict) and message.get("role") == "tool"
                and isinstance(getattr(message, "_stateframe_tool_reply_receipt", None), Mapping)):
            copied = copy_tool_reply_message(message)
        else:
            copied = copy.deepcopy(dict(message))
        if (index == 1 or copied.get("role") in {"assistant", "tool"}
                or copied.get("_stateframe_owner_block") is not None
                or not isinstance(copied.get("content"), str)
                 # Serialized StateFrame is typed data with revision-bound source bytes.
                 # Host-prose rewriting must never change it, in either flag mode.
                 or copied["content"].startswith("[StateFrame]\n")
                or is_source_content(copied.get("content"))):
            rendered.append(copied)
            continue
        content = str(copied["content"])

        def rewrite(match: re.Match[str]) -> str:
            nonlocal replacements
            name = match.group(1)
            if name in _KNOWN_TOOL_NAMES and name not in offered:
                replacements += 1
                return "use only a tool listed in this request"
            return match.group(0)

        def sanitize_segment(segment: str) -> str:
            segment = reference.sub(rewrite, segment)
            for name in sorted(_KNOWN_TOOL_NAMES - offered, key=len, reverse=True):
                mention = re.compile(
                    rf"(?i)(?<![A-Za-z0-9_])(?:`{re.escape(name)}`|"
                    rf"{re.escape(name)}(?=\s+tool\b))(?![A-Za-z0-9_])"
                )
                segment = mention.sub("available tools", segment)
            return segment

        parts: list[str] = []
        cursor = 0
        for match in _EVIDENCE_PIN_RE.finditer(content):
            parts.append(sanitize_segment(content[cursor:match.start()]))
            parts.append(match.group())
            cursor = match.end()
        parts.append(sanitize_segment(content[cursor:]))
        content = "".join(parts)
        copied["content"] = content
        rendered.append(copied)
    return rendered, replacements


def deferred_read_paths(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    paths: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("[Deferred required source paths]"):
            continue
        for line in content.splitlines():
            if line.startswith("- "):
                path = line[2:].strip().replace("\\", "/")
                if path and path not in paths:
                    paths.append(path)
    return tuple(paths)


def validate_deferred_reads(
    messages: Sequence[Mapping[str, object]], tools: Sequence[object],
) -> tuple[str, ...]:
    """A missing-source notice requires an offered read tool or delivered source bytes."""
    missing = []
    offered_read = "read" in {tool_name(tool) for tool in tools}
    for path in deferred_read_paths(messages):
        delivered = any(is_source_content(message.get("content"), path)
                        for message in messages)
        if not delivered and not offered_read:
            missing.append(path)
    if missing:
        raise ValueError(
            "deferred required source notice has no admitted read route or source: "
            + ", ".join(missing)
        )
    return ()


def _drop_unknown_wire_facts(content: str) -> str:
    kept = []
    for line in content.splitlines():
        if line.startswith("[WIRE_TRUTH_FACT") and re.search(r"(?i)(?::|=)\s*UNKNOWN(?:\b|\s|$)", line):
            continue
        kept.append(line)
    return "\n".join(kept)


def _preserve_qwen_reasoning(
    messages: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], int]:
    """Keep host user notices from becoming the last user turn after prior reasoning."""
    copied = [copy_tool_reply_message(message) for message in messages
              if isinstance(message, Mapping)]
    last_user = max(
        (index for index, message in enumerate(copied)
         if message.get("role") == "user"),
        default=-1,
    )
    reasoning_indexes = [
        index for index, message in enumerate(copied)
        if message.get("role") == "assistant"
        and (
            any(key in message for key in ("reasoning_content", "reasoning", "thinking"))
            or any(marker in str(message.get("content") or "")
                   for marker in ("<think>", "<analysis>"))
        )
    ]
    if not reasoning_indexes:
        return copied, 0
    last_reasoning = max(reasoning_indexes)
    if last_user <= last_reasoning:
        return copied, 0
    trailing_host_users = [
        index for index in range(max(2, last_reasoning + 1), len(copied))
        if copied[index].get("role") == "user"
    ]
    if not trailing_host_users:
        return copied, 0
    latest_tool_index = next(
        (index for index in range(len(copied) - 1, -1, -1)
         if copied[index].get("role") == "tool"),
        -1,
    )
    if latest_tool_index >= 0:
        for index in trailing_host_users:
            _append_latest_tool_text(copied[:latest_tool_index + 1],
                                     str(copied[index].get("content") or ""))
        removed = set(trailing_host_users)
        return [message for index, message in enumerate(copied) if index not in removed], len(removed)
    insert_at = last_reasoning
    notices = [copied[index] for index in trailing_host_users]
    removed = set(trailing_host_users)
    kept = [message for index, message in enumerate(copied) if index not in removed]
    kept[insert_at:insert_at] = notices
    return kept, len(removed)


def _prune_deferred_paths_with_current_evidence(
        messages: list[dict[str, object]],
) -> bool:
    """Drop deferred paths already represented by same-revision current source evidence."""
    try:
        from kernel_request_diet import (
            _current_source_duplicate_location,
            _current_source_segments,
            _source_identity,
            _source_delivery_ranges,
            duplicated_source_eviction_enabled,
        )

        if not duplicated_source_eviction_enabled():
            return False
        current_sources = _current_source_segments(messages)
        if not current_sources:
            return False
        current_revisions: dict[str, set[str]] = {}
        for source in current_sources:
            path = str(source.get("path", "")).replace("\\", "/").removeprefix("./")
            revision = str(source.get("revision", ""))
            if path and revision:
                current_revisions.setdefault(path, set()).add(revision)
        if not current_revisions:
            return False
        delivered = _source_delivery_ranges(messages)
        evidenced_paths = {
            path for path, revision in delivered
            if revision and revision in current_revisions.get(path, set())
        }
        for message in messages:
            if message.get("role") != "tool":
                continue
            identity = _source_identity(message)
            if (identity is None or identity[0] != "read"
                    or not _current_source_duplicate_location(
                        messages, message, current_sources,
                        allow_unversioned_current_source=True,
                    )):
                continue
            evidenced_paths.add(identity[1].replace("\\", "/").removeprefix("./"))
        if not evidenced_paths:
            return False

        revised: list[dict[str, object]] = []
        changed = False
        for message in messages:
            content = message.get("content")
            if (message.get("role") != "user" or not isinstance(content, str)
                    or not content.startswith("[Deferred required source paths]")):
                revised.append(message)
                continue
            lines = content.splitlines()
            kept_lines = [lines[0]]
            kept_paths = 0
            for line in lines[1:]:
                if line.startswith("- "):
                    path = line[2:].strip().replace("\\", "/").removeprefix("./")
                    if path in evidenced_paths:
                        changed = True
                        continue
                    if path:
                        kept_paths += 1
                kept_lines.append(line)
            if kept_paths == 0:
                changed = True
                continue
            if len(kept_lines) != len(lines):
                message = dict(message)
                message["content"] = "\n".join(kept_lines)
            revised.append(message)
        if changed:
            messages[:] = revised
        return changed
    except Exception:
        # This wire cleanup is optional; parser uncertainty preserves the notice.
        return False


def finalize_wire_body(body: Mapping[str, object], *, lean: bool) -> tuple[dict[str, object], tuple[str, ...]]:
    """Preserve wire roles while removing nonessential host tails in LEAN mode."""
    result = copy.deepcopy(dict(body))
    raw_messages = result.get("messages")
    if not isinstance(raw_messages, list):
        return result, ()
    messages = [copy_tool_reply_message(message) for message in raw_messages
                if isinstance(message, Mapping)]
    deferred_paths_pruned = _prune_deferred_paths_with_current_evidence(messages)
    if not lean:
        messages, moved = _preserve_qwen_reasoning(messages)
        result["messages"] = messages
        notices = []
        if deferred_paths_pruned:
            notices.append("deferred_source_paths_already_delivered")
        if moved:
            notices.append("qwen_host_notice_repositioned")
        return result, tuple(notices)

    from kernel_restate_requirement import MESSAGE as RESTATE_MESSAGE
    dropped: list[str] = []
    if deferred_paths_pruned:
        dropped.append("deferred_source_paths_already_delivered")
    latest_tool_index = next(
        (index for index in range(len(messages) - 1, -1, -1)
         if messages[index].get("role") == "tool"),
        -1,
    )
    task_index = next(
        (index for index, message in enumerate(messages)
         if message.get("role") == "user"
         and message.get("content") != RESTATE_MESSAGE),
        None,
    )
    task_end = task_index + 1 if task_index is not None else 2
    current_tail_start = max(
        task_end, latest_tool_index + 1 if latest_tool_index >= 0 else 2,
    )
    known_fact_lines: list[str] = []
    foldable_notices: list[str] = []
    source_tail_blocks: list[str] = []
    first_turn_task_additions: list[str] = []
    first_turn_deferred_notices: list[str] = []
    removed_indexes: set[int] = set()
    for index in range(current_tail_start, len(messages)):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith("[WIRE_READ_CONTRACT]"):
            removed_indexes.add(index)
            dropped.append("wire_read_contract")
            continue
        if is_source_content(content):
            if latest_tool_index >= 0:
                # The exact block remains byte-verifiable inside the latest
                # tool result and does not displace prior assistant reasoning.
                source_tail_blocks.append(content)
                removed_indexes.add(index)
                dropped.append("source_block_folded_into_latest_tool")
            continue
        cleaned = _drop_unknown_wire_facts(content)
        if cleaned != content:
            message["content"] = cleaned
            dropped.append("unknown_wire_facts")
        if cleaned.startswith("[WIRE_TRUTH_FACT "):
            known_fact_lines.extend(
                line for line in cleaned.splitlines()
                if line.startswith("[WIRE_TRUTH_FACT ")
            )
            removed_indexes.add(index)
            dropped.append("wire_truth_facts_folded")
            continue
        if _is_truth_notice(cleaned):
            if latest_tool_index >= 0:
                foldable_notices.append(cleaned)
                dropped.append("trailing_notice_folded")
            elif cleaned.startswith("[Deferred required source paths]"):
                # Keep the exact notice as a separate user message before any
                # assistant reasoning so the source gate can prove its bytes.
                first_turn_deferred_notices.append(cleaned)
                dropped.append("first_turn_deferred_notice_preserved")
            elif not cleaned.startswith("[WIRE_TRUTH_FACT "):
                # The first request has no tool result yet. Keep missing-source
                # truth inside the task after exact source blocks are preserved.
                first_turn_task_additions.append(cleaned)
                dropped.append("first_turn_notice_folded")
            removed_indexes.add(index)
            continue
        removed_indexes.add(index)
        dropped.append(advisory_kind(cleaned) or "trailing_host_user")

    if known_fact_lines:
        dropped.append("wire_truth_facts_folded")
    if latest_tool_index >= 0 and (foldable_notices or known_fact_lines):
        if known_fact_lines:
            foldable_notices.append("\n".join(dict.fromkeys(known_fact_lines)))
        current_tool_content = str(messages[latest_tool_index].get("content") or "")
        unique_notices = list(dict.fromkeys(
            text for text in foldable_notices if text and text not in current_tool_content
        ))
        joined_notices = (
            "[Host truth notice]\n" + "\n\n".join(unique_notices)
            if unique_notices else ""
        )
        if joined_notices and not _append_latest_tool_text(
                messages[:latest_tool_index + 1], joined_notices):
            dropped.append("duplicate_trailing_notice")
    elif known_fact_lines:
        # Keep required facts on a first-turn request without adding a new tail.
        first_turn_task_additions.append("\n".join(dict.fromkeys(known_fact_lines)))
        dropped.append("wire_truth_facts_folded_into_task")

    if latest_tool_index >= 0:
        for source_block in dict.fromkeys(source_tail_blocks):
            if not _append_latest_tool_source(
                    messages[:latest_tool_index + 1], source_block):
                dropped.append("duplicate_trailing_source")

    if first_turn_task_additions:
        task = next((message for message in messages
                     if message.get("role") == "user"
                     and message.get("content") != RESTATE_MESSAGE), None)
        if task is not None:
            task_content = str(task.get("content") or "")
            for addition in dict.fromkeys(first_turn_task_additions):
                if addition not in task_content:
                    task_content = task_content.rstrip() + "\n\n" + addition
            task["content"] = task_content

    if removed_indexes:
        messages = [message for index, message in enumerate(messages)
                    if index not in removed_indexes]
    if first_turn_deferred_notices:
        task_index = next(
            (index for index, message in enumerate(messages)
             if message.get("role") == "user"
             and message.get("content") != RESTATE_MESSAGE),
            None,
        )
        if task_index is not None:
            insert_at = task_index + 1
        else:
            insert_at = next(
                (index for index, message in enumerate(messages)
                 if message.get("role") == "assistant"),
                len(messages),
            )
        for notice in dict.fromkeys(first_turn_deferred_notices):
            messages.insert(insert_at, {
                "role": "user",
                "content": notice,
                "_request_budget_section": "stage_evidence",
            })
            insert_at += 1
    messages, qwen_moved = _preserve_qwen_reasoning(messages)
    if qwen_moved:
        dropped.append("qwen_host_notice_repositioned")
    result["messages"] = messages
    return result, tuple(dict.fromkeys(dropped))


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def request_size_bytes(messages: Sequence[Mapping[str, object]], tools: Sequence[object]) -> int:
    return len(canonical_bytes({"messages": list(messages), "tools": list(tools)}))


def shared_prefix_ratio(
    previous: Sequence[Mapping[str, object]] | None,
    current: Sequence[Mapping[str, object]],
) -> float | None:
    if previous is None:
        return None
    previous_bytes = sum(len(canonical_bytes(message)) + 1 for message in previous)
    if previous_bytes == 0:
        return 1.0
    current_rows = list(current)
    prefix_bytes = 0
    for index, message in enumerate(previous):
        if index >= len(current_rows) or canonical_bytes(message) != canonical_bytes(current_rows[index]):
            break
        prefix_bytes += len(canonical_bytes(message)) + 1
    return min(1.0, prefix_bytes / previous_bytes)


def message_digest(messages: Sequence[Mapping[str, object]]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(list(messages))).hexdigest()


__all__ = [
    "BoundedToolMessages", "LEAN_STACK", "MAX_GREP_PAGE_SIZE", "MAX_LISTING_ROWS",
    "MAX_READ_PAGE_CHARS", "MAX_READ_PAGE_LINES", "MAX_TOOL_RESULT_BYTES",
    "MODE_ENV", "PLAIN_SYSTEM_PROMPT", "PLAIN_TOOL_NAMES", "STALL_ENV",
    "MAX_STACK_NOTICE_TOKENS", "MAX_TOOL_RESULT_CHARS", "advisory_kind",
    "append_stack_notice", "bound_plain_tool_arguments", "bound_tool_result", "canonical_bytes",
    "deferred_read_paths", "drop_advisory_messages", "enabled", "finalize_wire_body",
    "history_compaction_enabled", "is_source_content", "is_truth_content", "message_digest",
    "plain_loop_messages", "plain_tool_surface",
    "render_sibling_site_facts", "request_size_bytes", "should_deliver_zero_progress_advisory",
    "runtime_lean_snapshot", "sanitize_tool_references", "shared_prefix_ratio",
    "stack_enabled", "stack_snapshot", "stack_tail_blocks", "stall_observation_enabled",
    "validate_deferred_reads", "restate_requirement_stack_enabled",
    "wide_target_window_stack_enabled", "sibling_site_facts_stack_enabled",
]
