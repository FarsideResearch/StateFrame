"""Host-owned trajectory representations for the checkpoint replay screen.

This module is deliberately independent from the live StateFrame worker.  It
accepts a public checkpoint and one already-closed assistant/tool exchange and
returns JSON-shaped prompt messages.  It never discovers a result, executes a
tool call, or treats model prose as a host fact.

The ``replay`` arm is the important control: it carries the real native call
and the real native result, with no reasoning field.  The other arms add
explicitly host-authored material and mark its accounting boundary so a
screen can keep prompt input separate from model-generated output.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence


TRAJECTORY_FRAME_ENV = "STATEFRAME_TRAJECTORY_FRAME"
TRAJECTORY_MODES = frozenset({
    "off", "replay", "pre_thought", "handoff_host", "handoff_model",
})

_STALE = frozenset({"STALE", "CONFLICT", "INVALID", "REJECTED", "SUPERSEDED"})
_CURRENT = frozenset({"CURRENT", "VALID", "VERIFIED", "OPEN", "PASS", "ACCEPTED"})
_INTENTION = frozenset({"INTENTION", "INTENDED", "PLAN", "PLANNED", "PROPOSED"})
_EXECUTED = frozenset({"EXECUTED", "APPLIED", "PASS", "PASSED", "SUCCESS", "COMPLETED"})

# Public synthetic pairs for the E3 revocation/stale-intent screen.  They are
# fixtures, not evidence from a real run and contain no executable action.
E3_STALE_INTENT_CHECKPOINT_PAIRS = (
    {
        "pair_id": "E3-revoked-revision",
        "stale_intent": {
            "revision": "r16", "fact_id": "I-r16-c3", "status": "INTENDED",
            "value": "run_tests(C3) was planned after the earlier edit",
        },
        "current": {
            "revision": "r17", "fact_id": "F-r17-c3", "status": "OPEN",
            "value": "C3 still requires a current verification",
        },
    },
    {
        "pair_id": "E3-intention-is-not-execution",
        "stale_intent": {
            "revision": "r21", "fact_id": "I-r21-edit", "status": "PROPOSED",
            "value": "edit module.py would address C7",
        },
        "current": {
            "revision": "r22", "fact_id": "F-r22-edit", "status": "NOT_EXECUTED",
            "value": "no edit receipt exists for C7",
        },
    },
)


def e3_stale_intent_checkpoint_pairs() -> tuple[dict[str, object], ...]:
    """Return defensive copies of the public E3 checkpoint-pair fixtures."""
    return tuple(copy.deepcopy(pair) for pair in E3_STALE_INTENT_CHECKPOINT_PAIRS)


def _source(environ: Mapping[str, object] | None) -> Mapping[str, object]:
    return os.environ if environ is None else environ


def trajectory_frame_mode(environ: Mapping[str, object] | None = None) -> str:
    """Read and strictly validate the independent trajectory-frame switch."""
    value = _source(environ).get(TRAJECTORY_FRAME_ENV, "off")
    if not isinstance(value, str):
        value = str(value)
    if value not in TRAJECTORY_MODES:
        choices = "|".join(("off", "replay", "pre_thought", "handoff_host", "handoff_model"))
        raise ValueError(f"{TRAJECTORY_FRAME_ENV} must be {choices}")
    return value


def trajectory_enabled(environ: Mapping[str, object] | None = None) -> bool:
    return trajectory_frame_mode(environ) != "off"


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return " ".join(str(value).split())


def _first(mapping: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _rows(value: object, *, depth: int = 0) -> list[dict[str, object]]:
    if depth > 7 or value is None:
        return []
    if isinstance(value, Mapping):
        rows = [dict(value)]
        for key in ("value", "fact", "facts", "receipts", "checks", "items",
                    "open_work", "open_items", "assessments", "verification"):
            if key in value:
                rows.extend(_rows(value[key], depth=depth + 1))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows: list[dict[str, object]] = []
        for item in value:
            rows.extend(_rows(item, depth=depth + 1))
        return rows
    return []


def _row_status(row: Mapping[str, object]) -> str:
    nested = row.get("value")
    sources = (nested, row) if isinstance(nested, Mapping) else (row,)
    for source in sources:
        for key in ("status", "result", "result_class", "verdict", "state"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip().upper()
    return ""


def _row_id(row: Mapping[str, object]) -> str:
    nested = row.get("value")
    sources = (row, nested) if isinstance(nested, Mapping) else (row,)
    for source in sources:
        for key in ("id", "fact_id", "check_id", "obligation_id", "work_id", "subject", "name"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""


def _fact_value(row: Mapping[str, object]) -> str:
    nested = row.get("value")
    if isinstance(nested, Mapping):
        value = _first(nested, "value", "observation", "observed", "detail", "text", "assertion")
        if value is not None:
            return _text(value)
    value = _first(row, "observation", "observed", "detail", "text", "fact", "value")
    return _text(value)


def _current_revision(frame: Mapping[str, object]) -> str:
    for key in ("current_revision", "revision", "product_revision", "workspace_revision"):
        value = frame.get(key)
        if value is not None and not isinstance(value, Mapping):
            return _text(value)
    binding = frame.get("binding")
    if isinstance(binding, Mapping):
        return _text(_first(binding, "product_revision", "revision"))
    return "current"


def _revision_forms(value: object) -> set[str]:
    text = _text(value)
    if not text:
        return set()
    if text.startswith("sha256:"):
        return {text, text[7:]}
    return {text, "sha256:" + text}


def _same_revision(candidate: object, current: str) -> bool:
    if not candidate or not current or current == "current":
        return True
    candidate_forms = _revision_forms(candidate)
    current_forms = _revision_forms(current)
    return bool(candidate_forms & current_forms) or any(
        len(left) >= 8 and len(right) >= 8 and (left.startswith(right) or right.startswith(left))
        for left in candidate_forms for right in current_forms
    )


def _fact_rows(frame: Mapping[str, object], exchange: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in ("host_facts", "host_check_receipts", "check_receipts", "receipts",
                "verification", "current_public_checks"):
        rows.extend(_rows(frame.get(key)))
    if isinstance(exchange, Mapping):
        rows.extend(_rows(exchange.get("pre_action_facts")))
        rows.extend(_rows(exchange.get("post_result_facts")))
        rows.extend(_rows(exchange.get("verified_facts")))
    return rows


def _stale_fact_ids(frame: Mapping[str, object], exchange: Mapping[str, object] | None = None) -> set[str]:
    current = _current_revision(frame)
    result: set[str] = set()
    for row in _fact_rows(frame, exchange):
        identifier = _row_id(row)
        if not identifier:
            continue
        revision = _first(row, "revision", "snapshot_digest", "artifact_revision", "product_revision")
        if _row_status(row) in _STALE or (revision is not None and not _same_revision(revision, current)):
            result.add(identifier)
    return result


def _fact_catalog(frame: Mapping[str, object], exchange: Mapping[str, object] | None = None) -> dict[str, dict[str, object]]:
    """Return facts the host is willing to expose to the model selector."""
    current = _current_revision(frame)
    allowed: dict[str, dict[str, object]] = {}
    explicitly_verified = set()
    if isinstance(exchange, Mapping):
        values = exchange.get("verified_fact_ids")
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            explicitly_verified.update(str(value) for value in values)
    for row in _fact_rows(frame, exchange):
        identifier = _row_id(row)
        if not identifier or identifier in allowed:
            continue
        status = _row_status(row)
        revision = _first(row, "revision", "snapshot_digest", "artifact_revision", "product_revision")
        verified = row.get("verified") is True or identifier in explicitly_verified
        if not verified or status in _STALE or (revision is not None and not _same_revision(revision, current)):
            continue
        value = _fact_value(row)
        allowed[identifier] = {
            "id": identifier, "status": status or "CURRENT", "revision": _text(revision) or current,
            "value": value or "host-verified fact",
        }
    return allowed


def _exchange_pair(exchange: Mapping[str, object] | None) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(exchange, Mapping):
        raise ValueError(f"{TRAJECTORY_FRAME_ENV} requires a recorded assistant/tool exchange")
    raw_assistant = _mapping(exchange.get("assistant"))
    raw_tool = _mapping(exchange.get("tool"))
    if raw_assistant is None:
        raw_assistant = _mapping(exchange.get("last_tool_call"))
    if raw_tool is None:
        raw_tool = _mapping(exchange.get("last_tool_result"))

    call_id = _first(exchange, "call_id", "tool_call_id")
    tool_name = _first(exchange, "tool_name", "tool")
    arguments = _first(exchange, "arguments", "args", "parameters", "action_parameters")
    if raw_assistant is None and tool_name:
        call_id = str(call_id or "trajectory-call")
        argument_text = arguments if isinstance(arguments, str) else json.dumps(
            arguments if isinstance(arguments, Mapping) else {}, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False)
        raw_assistant = {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": str(tool_name), "arguments": argument_text}}],
        }
    elif raw_assistant is not None and raw_assistant.get("role") != "assistant":
        # Snapshot records often store only the function object under
        # ``last_tool_call`` rather than the surrounding assistant message.
        function = raw_assistant.get("function")
        if isinstance(function, Mapping):
            call_id = str(raw_assistant.get("id") or call_id or "trajectory-call")
            raw_assistant = {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": call_id, "type": "function",
                                "function": dict(function)}],
            }
    if raw_assistant is None or raw_tool is None:
        raise ValueError(f"{TRAJECTORY_FRAME_ENV} requires both real call and real result")
    assistant = copy.deepcopy(dict(raw_assistant))
    tool_result = copy.deepcopy(dict(raw_tool))
    if assistant.get("role") != "assistant":
        raise ValueError("trajectory exchange assistant record has the wrong role")
    calls = assistant.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or not calls:
        raise ValueError("trajectory exchange has no native tool call")
    # The screen replays exactly one closed call.  Narrowing a provider batch
    # is safe only when the checkpoint identifies the selected call.
    selected = next((call for call in calls if isinstance(call, Mapping)
                     and (call.get("id") == call_id or call.get("id") == tool_result.get("tool_call_id"))), None)
    selected = selected or (calls[0] if isinstance(calls[0], Mapping) else None)
    if selected is None:
        raise ValueError("trajectory exchange call is not an object")
    selected = copy.deepcopy(dict(selected))
    selected_id = str(selected.get("id") or call_id or "trajectory-call")
    selected["id"] = selected_id
    selected["type"] = "function"
    function = _mapping(selected.get("function"))
    if function is None:
        function = {"name": str(tool_name or "unknown"), "arguments": "{}"}
    else:
        function = dict(function)
    function["name"] = str(function.get("name") or tool_name or "unknown")
    raw_args = function.get("arguments", arguments if arguments is not None else {})
    if isinstance(raw_args, Mapping):
        raw_args = json.dumps(dict(raw_args), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    elif not isinstance(raw_args, str):
        raw_args = _text(raw_args)
    function["arguments"] = raw_args or "{}"
    selected["function"] = function
    assistant["tool_calls"] = [selected]
    # A replay is intentionally not a reasoning transcript.  Any historical
    # reasoning is a separate private channel and is removed from the control.
    assistant.pop("reasoning_content", None)
    assistant.pop("response_reasoning_content", None)
    assistant.setdefault("content", "")
    if tool_result.get("role") != "tool":
        tool_result["role"] = "tool"
    tool_result["tool_call_id"] = selected_id
    if "content" not in tool_result:
        result = _first(exchange, "result", "result_status", "result_value", "observation")
        tool_result["content"] = _text(result) or "{}"
    return assistant, tool_result


def _call_label(assistant: Mapping[str, object]) -> tuple[str, str, str]:
    calls = assistant.get("tool_calls")
    call = calls[0] if isinstance(calls, Sequence) and calls else {}
    function = call.get("function") if isinstance(call, Mapping) else {}
    function = function if isinstance(function, Mapping) else {}
    name = str(function.get("name") or "unknown")
    raw = function.get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        parsed = raw
    compact = _text(parsed) or "{}"
    return name, compact, str(call.get("id") or "") if isinstance(call, Mapping) else ""


def _pre_action_facts(exchange: Mapping[str, object], frame: Mapping[str, object]) -> list[dict[str, object]]:
    current = _current_revision(frame)
    post_ids = {
        _row_id(row) for row in _rows(exchange.get("post_result_facts")) if _row_id(row)
    }
    facts: list[dict[str, object]] = []
    for row in _rows(exchange.get("pre_action_facts")):
        identifier = _row_id(row)
        if not identifier or identifier in post_ids:
            continue
        if row.get("available_before") is not True:
            continue
        if _row_status(row) in _STALE:
            continue
        revision = _first(row, "revision", "snapshot_digest", "artifact_revision", "product_revision")
        if revision is not None and not _same_revision(revision, current):
            continue
        facts.append({"id": identifier, "value": _fact_value(row) or "recorded before the call"})
    return facts


def build_pre_action_thought(frame: Mapping[str, object], exchange: Mapping[str, object]) -> str:
    """Compile a bounded thought from facts explicitly available before the call."""
    assistant, _tool = _exchange_pair(exchange)
    name, arguments, _call_id = _call_label(assistant)
    facts = _pre_action_facts(exchange, frame)
    if facts:
        rendered = "; ".join(f"{fact['id']}: {fact['value']}" for fact in facts[:4])
        premise = f"Host pre-action facts: {rendered}."
    else:
        premise = "No verified pre-action fact was recorded."
    # "proposed" and "not yet known" are deliberate causal guards: this arm
    # must never turn a post-result observation into an earlier thought.
    return (f"{premise} Proposed action only: {name}({arguments}). "
            "The result was not yet known at the time of this call.")


def _open_ids(frame: Mapping[str, object]) -> list[str]:
    found: list[str] = []
    for key in ("open_work", "open_items", "pending", "obligations", "completion_contract"):
        for row in _rows(frame.get(key)):
            status = _row_status(row)
            identifier = _row_id(row)
            if identifier and status not in {"PASS", "PASSED", "GREEN", "SATISFIED", "CLOSED", "DONE", "RESOLVED"}:
                found.append(identifier)
    return list(dict.fromkeys(found))


def _current_value(frame: Mapping[str, object], *keys: str, default: str = "") -> str:
    for key in keys:
        value = frame.get(key)
        if value is not None:
            return _text(value)
    return default


def _result_ref(exchange: Mapping[str, object]) -> str:
    return _text(_first(exchange, "result_ref", "result_id", "receipt_ref", "record_ref")) or "result:recorded"


def _result_scope(exchange: Mapping[str, object], frame: Mapping[str, object]) -> str:
    return (_text(_first(exchange, "result_scope", "scope", "completed_scope"))
            or ", ".join(_open_ids(frame)) or "the recorded action only")


def _cutoff_receipt(exchange: Mapping[str, object], frame: Mapping[str, object]) -> str:
    return (_text(_first(exchange, "cutoff_receipt", "receipt", "cutoff"))
            or _current_value(frame, "cutoff_receipt", "receipt", default="receipt:unavailable"))


def build_handoff(
    frame: Mapping[str, object], exchange: Mapping[str, object],
    *, selected_fact_ids: Sequence[str] = (), selection_author: str = "host",
) -> str:
    """Render the canonical 80--160 word post-result host handoff."""
    assistant, _tool = _exchange_pair(exchange)
    name, arguments, _call_id = _call_label(assistant)
    revision = _current_revision(frame)
    cutoff = _cutoff_receipt(exchange, frame)
    result_ref = _result_ref(exchange)
    scope = _result_scope(exchange, frame)
    remaining = ", ".join(_open_ids(frame)) or "none recorded"
    disposition = (_text(_first(exchange, "disposition", "current_disposition"))
                   or _current_value(frame, "current_disposition", "disposition", default="continue with the next verified requirement"))
    selected = ", ".join(str(value) for value in selected_fact_ids) or "none"
    facts = _fact_catalog(frame, exchange)
    selected_details = "; ".join(
        f"{identifier}: {facts[identifier]['value']}"
        for identifier in selected_fact_ids if identifier in facts
    ) or "no additional selected fact"
    return (
        f"Revision {revision} is the host-accepted revision. Cutoff receipt {cutoff} "
        f"bounds this replay. The completed action was {name}({arguments}), and the "
        f"recorded result reference is {result_ref}. The result scope is {scope}; it "
        f"does not establish any other requirement. Remaining requirement IDs are "
        f"{remaining}. The current disposition is {disposition}. This {selection_author}-selected "
        f"handoff is host-authored from the closed exchange and verified frame facts. "
        f"Verified fact IDs selected for the handoff: {selected}. Their admissible values: "
        f"{selected_details}. The packet records execution, not intention, and does not "
        "promote stale evidence. The next model action must re-check the open requirement "
        "against the current frame before claiming completion. No prompt-parsed call is executable."
    )


def trajectory_template_kwargs(template: str) -> dict[str, object]:
    """Return only the template flags needed to retain the historical thought."""
    normalized = str(template).strip().casefold().replace(" ", "")
    if normalized in {"minicpm5", "minicpm", "minicpm5-2b"}:
        return {"enable_thinking": True}
    if normalized in {"qwen3.6", "qwen36", "qwen3_6", "qwen3-coder"}:
        return {"preserve_thinking": True, "enable_thinking": True}
    raise ValueError("trajectory template must be MiniCPM5 or Qwen3.6")


def _native_call(call: Mapping[str, object], *, template: str) -> str:
    function = call.get("function") if isinstance(call.get("function"), Mapping) else {}
    name = str(function.get("name") or "unknown")
    raw_arguments = function.get("arguments", "{}")
    try:
        arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except (TypeError, ValueError):
        arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    normalized = str(template).strip().casefold().replace(" ", "")
    if normalized in {"qwen3.6", "qwen36", "qwen3_6", "qwen3-coder"}:
        parts = [f"<function={name}>"]
        for key, value in arguments.items():
            parts.append(f"<parameter={key}>{_text(value)}</parameter>")
        parts.append("</function>")
        return "".join(parts)
    parts = [f'<function name="{name}">']
    for key, value in arguments.items():
        parts.append(f'<param name="{key}"><![CDATA[{_text(value)}]]></param>')
    parts.append("</function>")
    return "".join(parts)


def apply_template_equivalent(
    messages: Sequence[Mapping[str, object]], template: str,
    chat_template_kwargs: Mapping[str, object] | None = None,
) -> str:
    """Render the public prompt shape used by the two supported templates.

    This is an offline ``/apply-template-equivalent`` proof, not a substitute
    for a live tokenizer.  It makes the two safety facts testable: native tool
    syntax is template-specific, and Qwen3.6 only retains earlier reasoning
    when both required kwargs are enabled.
    """
    kwargs = dict(chat_template_kwargs or {})
    required = trajectory_template_kwargs(template)
    normalized = str(template).strip().casefold().replace(" ", "")
    is_qwen = normalized in {"qwen3.6", "qwen36", "qwen3_6", "qwen3-coder"}
    if is_qwen:
        preserve = kwargs.get("preserve_thinking") is True and kwargs.get("enable_thinking") is True
    else:
        preserve = kwargs.get("enable_thinking", required.get("enable_thinking", True)) is not False
    rendered: list[str] = []
    last_user = max((index for index, message in enumerate(messages)
                     if isinstance(message, Mapping) and message.get("role") == "user"), default=-1)
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "user"))
        body: list[str] = []
        thought = message.get("reasoning_content")
        if isinstance(thought, str) and thought:
            # The Qwen3.6 rule is intentionally visible here: after a new user
            # turn, preserve_thinking=true is what keeps this historical block.
            keep = (not is_qwen or preserve or index > last_user)
            if keep:
                body.append("<think>" + thought + "</think>")
        content = message.get("content")
        if isinstance(content, str) and content:
            body.append(content)
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes)):
                body.extend(_native_call(call, template=template)
                            for call in calls if isinstance(call, Mapping))
        rendered.append(f"<|im_start|>{role}\n" + "\n".join(body) + "<|im_end|>\n")
    return "".join(rendered)


def _response_message(response: Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(response.get("message"), Mapping):
        return response["message"]
    choices = response.get("choices")
    if isinstance(choices, Sequence) and not isinstance(choices, (str, bytes)) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping) and isinstance(choice.get("message"), Mapping):
            return choice["message"]
        if isinstance(choice, Mapping):
            return choice
    return response


def select_verified_fact_ids(
    response: Mapping[str, object], frame: Mapping[str, object],
    exchange: Mapping[str, object] | None = None,
) -> tuple[list[str], list[str]]:
    """Parse only a JSON ID selection and intersect it with host verification."""
    message = _response_message(response)
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    candidate: object = None
    try:
        candidate = json.loads(content)
    except (TypeError, ValueError):
        stripped = content.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            try:
                candidate = json.loads("\n".join(stripped.splitlines()[1:-1]))
            except (TypeError, ValueError):
                candidate = None
    if isinstance(candidate, Mapping):
        candidate = candidate.get("fact_ids", candidate.get("verified_fact_ids", ()))
    if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes, bytearray)):
        candidate = ()
    valid = _fact_catalog(frame, exchange)
    selected: list[str] = []
    rejected: list[str] = []
    for value in candidate:
        identifier = str(value)
        if identifier in valid and identifier not in selected:
            selected.append(identifier)
        elif identifier not in rejected:
            rejected.append(identifier)
    return selected, rejected


def fact_misuse_metrics(
    response: Mapping[str, object], action: Mapping[str, object] | None,
    frame: Mapping[str, object], exchange: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Classify the two E3 misuse outcomes without treating prose as evidence."""
    message = _response_message(response)
    reasoning = " ".join(str(message.get(key) or "") for key in ("reasoning_content", "content"))
    action_text = json.dumps(action or {}, sort_keys=True, ensure_ascii=False)
    text = (reasoning + " " + action_text).casefold()
    stale_ids = _stale_fact_ids(frame, exchange)
    stale_hits = [identifier for identifier in stale_ids if identifier.casefold() in text]
    stale_language = bool(re.search(r"\b(?:current|use|rely|treat|valid|passed|green|should)\b", text))
    stale_misuse = bool(stale_hits and (stale_language or action is not None))

    intent_rows = []
    for row in _fact_rows(frame, exchange):
        identifier = _row_id(row)
        status = _row_status(row)
        if identifier and (status in _INTENTION or str(row.get("kind", "")).upper() in _INTENTION):
            intent_rows.append((identifier, _fact_value(row)))
    intent_hits = [identifier for identifier, value in intent_rows
                   if identifier.casefold() in text or (value and value.casefold() in text)]
    execution_language = bool(re.search(r"\b(?:executed|applied|completed|done|finished|ran|passed)\b", text))
    intention_misuse = bool(intent_hits and execution_language)
    return {
        "stale_evidence_treated_as_current": stale_misuse,
        "intention_treated_as_execution": intention_misuse,
        "stale_fact_ids": stale_hits,
        "intention_fact_ids": intent_hits,
    }


def trajectory_journal_fields(
    *, host_input_tokens: int, helper_generated_tokens: int = 0,
    selected_fact_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Return non-reserved fields for the continuation journal event."""
    return {
        "author": "host", "transport": "reasoning_content",
        "host_input_tokens": int(max(0, host_input_tokens)),
        "helper_generated_tokens": int(max(0, helper_generated_tokens)),
        "selected_fact_ids": list(selected_fact_ids),
    }


def build_trajectory_prefix(
    frame: Mapping[str, object], exchange: Mapping[str, object], mode: str,
    *, selected_fact_ids: Sequence[str] = (),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build the synthetic prefix that must precede the current StateFrame."""
    if mode not in TRAJECTORY_MODES or mode == "off":
        raise ValueError("trajectory prefix requires replay, pre_thought, handoff_host or handoff_model")
    assistant, tool_result = _exchange_pair(exchange)
    messages: list[dict[str, object]] = [assistant, tool_result]
    host_input_tokens = 0
    meta: dict[str, object] = {
        "trajectory_frame": mode,
        "trajectory_causality_ok": True,
        "trajectory_selected_fact_ids": list(selected_fact_ids),
    }
    if mode == "pre_thought":
        thought = build_pre_action_thought(frame, exchange)
        messages[0]["reasoning_content"] = thought
        host_input_tokens = len(thought.split())
        meta["trajectory_pre_thought"] = thought
        meta["trajectory_author"] = "host"
    elif mode in {"handoff_host", "handoff_model"}:
        author = "model_selected_ids" if mode == "handoff_model" else "host"
        handoff = build_handoff(frame, exchange, selected_fact_ids=selected_fact_ids,
                                selection_author=author)
        messages.append({"role": "assistant", "content": "", "reasoning_content": handoff})
        host_input_tokens = len(handoff.split())
        meta["trajectory_handoff"] = handoff
        meta["trajectory_author"] = "host"
    meta["trajectory_host_input_tokens"] = host_input_tokens
    meta["trajectory_journal"] = trajectory_journal_fields(
        host_input_tokens=host_input_tokens, selected_fact_ids=selected_fact_ids)
    # A tool call in this prefix is display-only.  The screen's execution path
    # consumes only a newly returned model message.
    meta["prompt_tool_calls_executable"] = False
    return messages, meta


def build_trajectory_messages(
    frame: Mapping[str, object], exchange: Mapping[str, object], mode: str,
    *, selected_fact_ids: Sequence[str] = (),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compatibility name for callers that treat the replay prefix as messages."""
    return build_trajectory_prefix(
        frame, exchange, mode, selected_fact_ids=selected_fact_ids)


__all__ = [
    "E3_STALE_INTENT_CHECKPOINT_PAIRS", "TRAJECTORY_FRAME_ENV", "TRAJECTORY_MODES",
    "apply_template_equivalent", "build_handoff", "build_pre_action_thought",
    "build_trajectory_messages", "build_trajectory_prefix", "e3_stale_intent_checkpoint_pairs",
    "fact_misuse_metrics",
    "select_verified_fact_ids", "trajectory_enabled", "trajectory_frame_mode",
    "trajectory_journal_fields", "trajectory_template_kwargs",
]
