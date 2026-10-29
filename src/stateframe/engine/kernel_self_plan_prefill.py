"""The opt-in two-call self-plan continuation arm.

This module owns the small, deterministic seam around a StateFrame request:
the first request is a no-thinking plan call and the second request continues
an open thinking block with the host bridge and that plan.  Transport callers
provide the callback; this module never discovers files, runs checks, or
interprets free-form model prose as host state.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from kernel_frame_continuation import build_prefill_seed


MODE_ENV = "STATEFRAME_THINK_PREFILL"
TRIGGER_ENV = "STATEFRAME_SELF_PLAN_TRIGGER"
CACHE_ENV = "STATEFRAME_SELF_PLAN_CACHE"
PAIR_CACHE_ENV = "STATEFRAME_SELF_PLAN_PAIR_CACHE"
SLOT_ENV = "STATEFRAME_SELF_PLAN_ID_SLOT"
REACT_SLOT_ENV = "REACT_SLOT_ID"
MAX_TOKENS_ENV = "STATEFRAME_SELF_PLAN_MAX_TOKENS"
SELF_PLAN_MODE = "self_plan"
VALID_TRIGGERS = frozenset({"first", "spiral", "always"})
PLAN_INSTRUCTION = (
    "For the next decision only, output a numbered plan of at most 8 short lines. "
    "No code."
)
_PASS = frozenset({"PASS", "PASSED", "GREEN", "SATISFIED", "ACCEPTED", "CLOSED", "DONE"})
_CLOSED = _PASS | frozenset({"RESOLVED"})
_OPEN = frozenset({"OPEN", "UNRESOLVED", "NEEDS_WORK", "PENDING", "UNSATISFIED",
                   "COUNTEREXAMPLE", "FAIL", "FAILED", "RED"})
_ID_KEYS = ("id", "check_id", "obligation_id", "work_id", "subject", "name")
_STATUS_KEYS = ("status", "result", "result_class", "verdict", "state")
_RE_NUMBER = re.compile(r"^\s*(?:[-*]\s*|\d+\s*[.)]\s*)")
_RE_CODE = re.compile(
    r"(?:```|^\s*(?:def\s|class\s|import\s|from\s+\S+\s+import\s|return\b|print\s*\()|[{}])"
)


def _source(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def self_plan_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _source(environ).get(MODE_ENV, "off").strip().lower() == SELF_PLAN_MODE


def self_plan_trigger(environ: Mapping[str, str] | None = None) -> str:
    value = _source(environ).get(TRIGGER_ENV, "first").strip().lower()
    if value not in VALID_TRIGGERS:
        raise ValueError(f"{TRIGGER_ENV} must be first, spiral or always")
    return value


def _switch_bool(name: str, value: object) -> bool:
    token = str(value).strip().lower()
    if token not in {"on", "off"}:
        raise ValueError(f"{name} must be on or off")
    return token == "on"


def pair_cache_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = _source(environ)
    from kernel_prompt_cache import pair_cache_enabled_for_prompt_cache
    if pair_cache_enabled_for_prompt_cache(source):
        return True
    # The longer alias is useful in launch manifests; the short name is the
    # documented operator switch.  Both are strict and the alias wins when
    # explicitly supplied.
    if PAIR_CACHE_ENV in source:
        return _switch_bool(PAIR_CACHE_ENV, source[PAIR_CACHE_ENV])
    return _switch_bool(CACHE_ENV, source.get(CACHE_ENV, "on"))


def self_plan_id_slot(environ: Mapping[str, str] | None = None) -> int:
    source = _source(environ)
    raw = source.get(SLOT_ENV, source.get(REACT_SLOT_ENV, "0"))
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{SLOT_ENV} must be a nonnegative integer") from error
    if value < 0:
        raise ValueError(f"{SLOT_ENV} must be a nonnegative integer")
    return value


def self_plan_max_tokens(environ: Mapping[str, str] | None = None) -> int:
    raw = _source(environ).get(MAX_TOKENS_ENV, "128")
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{MAX_TOKENS_ENV} must be a positive integer") from error
    if value < 1:
        raise ValueError(f"{MAX_TOKENS_ENV} must be a positive integer")
    return value


def _walk_rows(value: object, *, depth: int = 0) -> list[dict[str, object]]:
    if depth > 7 or value is None:
        return []
    if isinstance(value, Mapping):
        rows = [dict(value)]
        for key in ("value", "fact", "facts", "checks", "receipts", "items",
                    "open_work", "open_items", "assessments", "verification"):
            if key in value:
                rows.extend(_walk_rows(value[key], depth=depth + 1))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows: list[dict[str, object]] = []
        for item in value:
            rows.extend(_walk_rows(item, depth=depth + 1))
        return rows
    return []


def _row_id(row: Mapping[str, object]) -> str:
    for key in _ID_KEYS:
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return ""


def _row_status(row: Mapping[str, object]) -> str:
    nested = row.get("value")
    sources = (nested, row) if isinstance(nested, Mapping) else (row,)
    for source in sources:
        for key in _STATUS_KEYS:
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip().upper()
    return ""


def status_map(frame: Mapping[str, object]) -> dict[str, str]:
    """Return stable host-owned check/work statuses for trigger comparisons."""
    if not isinstance(frame, Mapping):
        raise TypeError("StateFrame must be a mapping")
    result: dict[str, str] = {}
    # Host facts are authoritative.  A later open-work row must not erase a
    # richer PASS/COUNTEREXAMPLE result for the same check.
    for key in ("host_facts", "host_check_receipts", "check_receipts", "receipts",
                "verification", "current_public_checks", "checks"):
        for row in _walk_rows(frame.get(key)):
            identifier, status = _row_id(row), _row_status(row)
            if identifier and status:
                result.setdefault(identifier, status)
    for key in ("open_work", "open_items", "obligations", "pending"):
        for row in _walk_rows(frame.get(key)):
            identifier, status = _row_id(row), _row_status(row)
            if identifier and status:
                result.setdefault(identifier, status)
                # Keep a second host-owned namespace so a change from OPEN to
                # PASS is visible even when a receipt for the same id remains.
                result[f"{key}:{identifier}"] = status
    return result


def _restart_hit(reasoning: object) -> bool:
    try:
        from kernel_frame_continuation import RESTART_LANGUAGE_RE
    except ImportError:  # pragma: no cover - dependency is part of this lane
        RESTART_LANGUAGE_RE = re.compile(r"(?i)\bunderstand\s+the\s+(?:task|problem)\b")
    text = " ".join(str(reasoning or "").split()[:512])
    return bool(RESTART_LANGUAGE_RE.search(text))


def _frame_marks_check_change(frame: Mapping[str, object]) -> bool:
    for key in ("check_status_changed", "checks_changed", "edit_changed_check_status",
                "edit_changes_check_status"):
        if frame.get(key) is True:
            return True
    for key in ("recent_changes", "edits", "actions"):
        for row in _walk_rows(frame.get(key)):
            if row.get("check_status_changed") is True or row.get("changes_check_status") is True:
                return True
    return False


def should_trigger(trigger: str, step: int, frame: Mapping[str, object], *,
                   previous_frame: Mapping[str, object] | None = None,
                   previous_reasoning: object = "") -> bool:
    trigger = str(trigger).strip().lower()
    if trigger not in VALID_TRIGGERS:
        raise ValueError(f"{TRIGGER_ENV} must be first, spiral or always")
    if trigger == "always":
        return True
    if int(step) == 1:
        return True
    if trigger == "first":
        return False
    return (_frame_marks_check_change(frame)
            or (previous_frame is not None and status_map(previous_frame) != status_map(frame))
            or _restart_hit(previous_reasoning))


@dataclass
class SelfPlanState:
    trigger: str = "first"
    previous_frame: Mapping[str, object] | None = None
    previous_reasoning: str = ""
    last_plan: str = ""
    last_plan_tokens: int = 0
    _observed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.trigger not in VALID_TRIGGERS:
            raise ValueError(f"{TRIGGER_ENV} must be first, spiral or always")

    def wants_plan(self, step: int, frame: Mapping[str, object]) -> bool:
        return should_trigger(self.trigger, step, frame,
                              previous_frame=self.previous_frame,
                              previous_reasoning=self.previous_reasoning)

    def observe(self, frame: Mapping[str, object], reasoning: object = "") -> None:
        self.previous_frame = copy.deepcopy(dict(frame))
        self.previous_reasoning = str(reasoning or "")
        self._observed = True


def normalize_plan(value: object) -> str:
    """Keep the plan small, numbered, and free of fenced/code-shaped lines."""
    raw = str(value or "").replace("\r", "")
    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("``") or _RE_CODE.search(line):
            continue
        line = _RE_NUMBER.sub("", line).strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) == 8:
            break
    if not lines:
        lines = ["Check the current StateFrame before choosing the next action"]
    return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, 1))


def _response_message(response: Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(response.get("message"), Mapping):
        return response["message"]
    choices = response.get("choices")
    if isinstance(choices, Sequence) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        return message if isinstance(message, Mapping) else choices[0]
    return response


def extract_plan(response: Mapping[str, object]) -> str:
    message = _response_message(response)
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        content = message.get("reasoning_content", "")
    return normalize_plan(content)


def _completion_tokens(response: Mapping[str, object], fallback: str) -> int:
    usage = response.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    for candidate in (usage.get("completion_tokens"), details.get("total_tokens"),
                      details.get("reasoning_tokens")):
        if isinstance(candidate, int) and candidate >= 0:
            return candidate
    return len(str(fallback).split())


def build_thinking_prefill(frame: Mapping[str, object], plan: str) -> str:
    bridge = build_prefill_seed(frame).strip()
    return f"{bridge}\nDraft plan (unverified):\n{normalize_plan(plan)}\nChecking this plan against the frame:"


def _template_kwargs(body: Mapping[str, object], *, thinking: bool) -> dict[str, object]:
    raw = body.get("chat_template_kwargs")
    kwargs = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {}
    kwargs["enable_thinking"] = thinking
    return kwargs


def _pair_controls(body: dict[str, object], *, id_slot: int, cache: bool) -> None:
    body["id_slot"] = id_slot
    body["cache_prompt"] = bool(cache)


def build_pair_requests(base_body: Mapping[str, object], frame: Mapping[str, object],
                        plan: str, *, id_slot: int, cache: bool) -> tuple[dict, dict, dict]:
    """Build the no-thinking plan call and the open-think continuation call."""
    if not isinstance(base_body, Mapping):
        raise TypeError("base request must be a mapping")
    prefix = copy.deepcopy(list(base_body.get("messages", ())))
    plan_body = copy.deepcopy(dict(base_body))
    plan_body["messages"] = [*prefix, {"role": "user", "content": PLAN_INSTRUCTION}]
    # Keep the tool schema in both requests: llama.cpp/OpenAI templates often
    # render it into the prompt prefix, and dropping it would make cache_n
    # incomparable even though the StateFrame prefix is identical.
    plan_body.pop("parallel_tool_calls", None)
    plan_body.pop("continue_final_message", None)
    plan_body["chat_template_kwargs"] = _template_kwargs(plan_body, thinking=False)
    plan_body["reasoning_effort"] = "none"
    plan_body["max_tokens"] = min(int(plan_body.get("max_tokens", 128)), self_plan_max_tokens())
    _pair_controls(plan_body, id_slot=id_slot, cache=cache)

    thinking_body = copy.deepcopy(dict(base_body))
    prefill = build_thinking_prefill(frame, plan)
    thinking_body["messages"] = [*prefix, {
        "role": "assistant", "content": None, "reasoning_content": prefill,
    }]
    thinking_body.pop("reasoning_effort", None)
    thinking_body["chat_template_kwargs"] = _template_kwargs(thinking_body, thinking=True)
    thinking_body["continue_final_message"] = "reasoning_content"
    _pair_controls(thinking_body, id_slot=id_slot, cache=cache)
    metadata = {
        "shared_prompt_prefix": thinking_body["messages"][:-1] == prefix
        and plan_body["messages"][:-1] == prefix,
        "cache_scope": "self_plan_pair_only",
        "prefill_text": prefill,
        "prefill_tokens": len(prefill.split()),
        "plan_tokens": len(normalize_plan(plan).split()),
        "id_slot": id_slot,
        "cache_prompt": bool(cache),
    }
    return plan_body, thinking_body, metadata


def _render_open_think(request: Mapping[str, object], *, template: str) -> str:
    messages = request.get("messages", ())
    rendered: list[str] = []
    last = len(messages) - 1 if isinstance(messages, Sequence) else -1
    for index, message in enumerate(messages if isinstance(messages, Sequence) else ()):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "user"))
        if index == last and role == "assistant" and request.get("continue_final_message"):
            rendered.append("<|im_start|>assistant\n<think>\n"
                            + str(message.get("reasoning_content") or ""))
        else:
            content = message.get("content")
            rendered.append(f"<|im_start|>{role}\n{content or ''}<|im_end|>\n")
    return "".join(rendered)


def render_self_plan_prefill(request: Mapping[str, object], *, template: str = "MiniCPM5") -> str:
    if str(template).strip().casefold() not in {"minicpm5", "minicpm", "qwen3", "qwen"}:
        raise ValueError("template must be MiniCPM5 or Qwen3")
    return _render_open_think(request, template=str(template))


def render_qwen3_prefill(request: Mapping[str, object]) -> str:
    return _render_open_think(request, template="Qwen3")


def run_pair(base_body: Mapping[str, object], frame: Mapping[str, object],
             call: Callable[[Mapping[str, object]], Mapping[str, object]], *,
             id_slot: int, cache: bool) -> dict[str, object]:
    """Run exactly one plan call followed by exactly one thinking call."""
    plan_body, _unused_thinking, _unused_meta = build_pair_requests(
        base_body, frame, "1. Check the current StateFrame.", id_slot=id_slot, cache=cache)
    plan_response = call(plan_body)
    plan = extract_plan(plan_response)
    _plan_body, thinking_body, metadata = build_pair_requests(
        base_body, frame, plan, id_slot=id_slot, cache=cache)
    thinking_response = call(thinking_body)
    plan_tokens = _completion_tokens(plan_response, plan)
    thinking_message = _response_message(thinking_response)
    thinking_fallback = " ".join(
        str(thinking_message.get(key) or "") for key in ("reasoning_content", "content"))
    thinking_tokens = _completion_tokens(thinking_response, thinking_fallback)
    return {
        "plan": plan,
        "plan_body": plan_body,
        "thinking_body": thinking_body,
        "plan_response": plan_response,
        "thinking_response": thinking_response,
        "plan_tokens": plan_tokens,
        "thinking_tokens": thinking_tokens,
        "prefill_tokens": metadata["prefill_tokens"],
        "metadata": metadata,
        "total_generated_including_plan": plan_tokens + thinking_tokens,
    }


def journal_plan(journal: object, *, request_id: str, plan_request_id: str,
                 plan: str, plan_tokens: int, metadata: Mapping[str, object]) -> object:
    """Write explicit accounting labels without colliding with journal-owned fields."""
    return journal.emit(
        "self_plan",
        request_id=request_id,
        plan_request_id=plan_request_id,
        plan_text=plan,
        plan_tokens=plan_tokens,
        generated_cost_tokens=plan_tokens,
        host_authored_for_accounting=True,
        prefill_region_not_model_reasoning=True,
        prefill_token_count=metadata.get("prefill_tokens", 0),
        generated_token_exclusion="prompt_prefill",
        plan_cache_n=metadata.get("plan_cache_n"),
        plan_prompt_n=metadata.get("plan_prompt_n"),
        cache_n=metadata.get("plan_cache_n"),
        prompt_n=metadata.get("plan_prompt_n"),
        id_slot=metadata.get("id_slot"),
        cache_prompt=metadata.get("cache_prompt", False),
        cache_scope=metadata.get("cache_scope", "self_plan_pair_only"),
    )


def _action_text(action: Mapping[str, object] | None) -> str:
    return json.dumps(action or {}, sort_keys=True, ensure_ascii=False).casefold()


def _open_ids(frame: Mapping[str, object]) -> set[str]:
    result: set[str] = set()
    for identifier, status in status_map(frame).items():
        if identifier.startswith(("open_work:", "open_items:", "obligations:", "pending:")):
            if status not in _CLOSED:
                result.add(identifier.split(":", 1)[1])
        elif status in _OPEN:
            result.add(identifier)
    return result


def acceptable_next_action(action: Mapping[str, object] | None,
                           frame: Mapping[str, object]) -> bool:
    if not isinstance(action, Mapping):
        return False
    tool = str(action.get("tool", action.get("name", ""))).casefold()
    args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
    target = str(args.get("target", args.get("path", ""))).casefold()
    if tool not in {"run_tests", "read"} or not target:
        return False
    return any(identifier.casefold() in target for identifier in _open_ids(frame))


def followed_plan_step_contradicted_by_frame(plan: str, frame: Mapping[str, object],
                                              action: Mapping[str, object] | None) -> bool:
    """Detect the narrow bad-plan anchor: an executed planned item is closed."""
    action_text = _action_text(action)
    if not action_text:
        return False
    statuses = status_map(frame)
    for identifier, status in statuses.items():
        if ":" in identifier:
            continue
        if status not in _CLOSED:
            continue
        lowered = identifier.casefold()
        if lowered in action_text and re.search(rf"\b{re.escape(lowered)}\b", plan.casefold()):
            return True
    return False


__all__ = [
    "CACHE_ENV", "PAIR_CACHE_ENV", "PLAN_INSTRUCTION", "SELF_PLAN_MODE", "SelfPlanState",
    "acceptable_next_action", "build_pair_requests", "build_thinking_prefill",
    "extract_plan", "followed_plan_step_contradicted_by_frame", "journal_plan",
    "normalize_plan", "pair_cache_enabled", "render_qwen3_prefill",
    "render_self_plan_prefill", "run_pair", "self_plan_enabled", "self_plan_id_slot",
    "self_plan_max_tokens", "self_plan_trigger", "should_trigger", "status_map",
]
