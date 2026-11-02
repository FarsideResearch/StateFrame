"""Host-owned request surface budgets for provider-bound StateFrame requests.

The budget is deliberately applied after presentation/menu projection and before
the completion owner freezes its request contract.  The transport seam repeats
the hard-ceiling check, so a later adapter cannot submit an oversized request.
"""
from __future__ import annotations

import kernel_frame_routes as _frame_routes

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

from kernel_measurement import profiled_function_span
from kernel_endpoint_size import ENDPOINT_SIZE

if TYPE_CHECKING:
    from kernel_request_diet import RequestTokenMeter


MODE_ENV = "STATEFRAME_REQUEST_BUDGET"
SCHEMA = "stateframe-request-budget/1"
STAGES = ("NATIVE", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
FREE_STAGE = "FREE"
DEFAULT_MAX_TOOLS = 5
DEFAULT_MAX_SCHEMA_BYTES = 24 * 1024
# L126E 922 capture: effective-config.json records context_per_slot=65,536.
# Measure prefill throughput before choosing the request budget.
# GET /props default_generation_settings.n_ctx = 65,536 and GET /slots = 4 slots x 65,536 (build b11044-2ec6bf846,
# Qwen3.6-35B-A3B-UD-Q4_K_XL). Per-slot context is the real limit for one request.
ENDPOINT_CONTEXT_TOKENS = ENDPOINT_SIZE.context_tokens
# All 11 captured 922 requests sent max_tokens=24,000. Reserve a separate
# 4,096 tokens for reasoning and another 4,096 for template/tokenizer variance.
ENDPOINT_OUTPUT_BUDGET_TOKENS = ENDPOINT_SIZE.output_tokens
ENDPOINT_THINKING_BUDGET_TOKENS = ENDPOINT_SIZE.thinking_tokens
ENDPOINT_SAFETY_HEADROOM_TOKENS = ENDPOINT_SIZE.safety_tokens
# Fallback calibration starts at the conservative 922 minimum. Exact request
# admission uses the llama.cpp tokenizer; this byte ratio is only the offline
# estimate until this run has response measurements.
ENDPOINT_REQUEST_BYTES_PER_TOKEN = 2.46
# Verified tokenizer measurements own the token ceiling; this finite serialized
# byte envelope covers ASCII JSON plus escaped Unicode representation variance.
ENDPOINT_VERIFIED_BYTES_PER_TOKEN = 16
ENDPOINT_VERIFIED_MAX_REQUEST_BYTES = 16 * 1024 * 1024
ENDPOINT_PROMPT_TOKEN_BUDGET = ENDPOINT_SIZE.prompt_budget
ENDPOINT_MAX_REQUEST_BYTES = int(
    ENDPOINT_PROMPT_TOKEN_BUDGET * ENDPOINT_REQUEST_BYTES_PER_TOKEN
)

# All request admission paths share the endpoint-derived byte ceiling.
DEFAULT_MAX_REQUEST_BYTES = ENDPOINT_MAX_REQUEST_BYTES
S3_MAX_REQUEST_BYTES = ENDPOINT_MAX_REQUEST_BYTES
HARD_MAX_REQUEST_BYTES = ENDPOINT_MAX_REQUEST_BYTES
HOST_CONTRACT_LIMIT = "HOST_CONTRACT_LIMIT"

# INVARIANT: a live budget may trim optional tools/sections, but it never
# rejects or removes a protected tool/section; overflow is receipt-marked.

# Static fallbacks are used only when a reduced standalone capture has no live
# IDEALPATH owner.  Live runtime calls add the owner's satisfying exit action.
DEFAULT_STAGE_EXIT_ACTIONS = {
    "NATIVE": frozenset(),
    FREE_STAGE: frozenset(),
    "S1": frozenset({"requirement_table"}),
    "S2": frozenset({"question_gate", "localize", "localization_frontier"}),
    "S3": frozenset({"question_gate"}),
    "S4": frozenset({"question_gate"}),
    "S5": frozenset({"question_gate"}),
    "S6": frozenset({"question_gate"}),
    "S7": frozenset({"question_gate"}),
}

_TRUTHY = frozenset({"on", "1", "true", "yes"})
_OPTIONAL_SECTIONS = frozenset({
    "optional", "user_hooks", "advice", "playbook", "history", "history_ledger", "diagnostic", "scratch",
    "skill_bodies",
})
_PROTECTED_SECTIONS = frozenset({"required", "completion", "contract", "terminal"})
_LOW_RELEVANCE_TOOLS = frozenset({
    "help", "status", "job_status", "job_output", "job_kill", "abstain",
})
_CORE_TOOLS = frozenset({
    "read", "grep", "skill_load", "edit", "str_replace", "file_create", "file_delete", "file_move", "probe", "run_tests", "typed_test_writer", "authored_check", "localize",
    "localization_frontier", "question_gate", "requirement_table", "done",
})
TRIM_ORDER = ("least_relevant_tools", "optional_sections", "defer")
_PROTECTED_SURFACE_MARKER = "__protected_completion_surface__"
_COMPLETION_ONLY_MARKER = "__completion_only_surface__"
RESERVATION_ORDER = (
    "legal_schemas", "ticket", "required_source", "required_truth",
    "latest_failure", "tool_result_pairs",
)


@dataclass(frozen=True)
class StageRequestBudget:
    stage: str
    max_tools: int = DEFAULT_MAX_TOOLS
    max_schema_bytes: int = DEFAULT_MAX_SCHEMA_BYTES
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES


STAGE_BUDGETS = {
    stage: StageRequestBudget(
        stage=stage,
        max_request_bytes=S3_MAX_REQUEST_BYTES if stage == "S3" else DEFAULT_MAX_REQUEST_BYTES,
    )
    for stage in STAGES
}


class RequestBudgetDeferred(RuntimeError):
    """Compatibility marker for older callers; live budgeting never raises it."""

    def __init__(self, detail: Mapping[str, object]):
        self.detail = dict(detail)
        super().__init__(str(self.detail.get("reason", "request budget deferred")))


class LeanHistoryCapacityError(ValueError):
    """Append-only LEAN history cannot be trimmed to fit the request capacity."""

    outcome = HOST_CONTRACT_LIMIT
    failure_origin = "harness"
    harness_contract = True
    transport_retryable = False
    transport_class = "capacity"

    def __init__(self, detail: Mapping[str, object]):
        self.capacity = dict(detail)
        super().__init__(str(self.capacity.get("reason", "LEAN history exceeds capacity")))


class LeanHistoryMutationError(ValueError):
    """A downstream request stage changed the prepared LEAN transcript."""

    outcome = HOST_CONTRACT_LIMIT
    failure_origin = "harness"
    harness_contract = True
    transport_retryable = False
    transport_class = "contract"


@dataclass(frozen=True)
class BudgetDecision:
    stage: str
    changed: bool
    before_request_bytes: int
    after_request_bytes: int
    before_tool_count: int
    after_tool_count: int
    before_schema_bytes: int
    after_schema_bytes: int
    dropped_tools: tuple[str, ...]
    dropped_sections: tuple[str, ...]
    preserved_tools: tuple[str, ...]
    reason: str


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Resolve the opt-in flag strictly; misspellings fail closed loudly."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value == "on"


def _runtime_enabled(runtime: object | None) -> bool:
    declared = getattr(runtime, "declared_environment", None)
    if isinstance(declared, Mapping) and MODE_ENV in declared:
        return enabled({MODE_ENV: str(declared[MODE_ENV])})
    return enabled()


def stage_budget(stage: str) -> StageRequestBudget:
    if stage == FREE_STAGE:
        return StageRequestBudget(stage=FREE_STAGE)
    if stage not in STAGE_BUDGETS:
        raise ValueError(f"unknown request-budget stage: {stage}")
    return STAGE_BUDGETS[stage]


def _lean_request_limits(
        body: Mapping[str, object], stage: str, *, exact_tokenizer: bool = False,
) -> tuple[int, int]:
    """Use serialized output allocation to reserve matching prompt room."""
    stage_limit = min(stage_budget(stage).max_request_bytes, HARD_MAX_REQUEST_BYTES)
    output_tokens = body.get("max_tokens")
    if type(output_tokens) is not int or output_tokens <= 0:
        prompt_tokens = ENDPOINT_PROMPT_TOKEN_BUDGET
    else:
        prompt_tokens = max(
            0,
            ENDPOINT_CONTEXT_TOKENS
            - output_tokens
            - ENDPOINT_THINKING_BUDGET_TOKENS
            - ENDPOINT_SAFETY_HEADROOM_TOKENS,
        )
        if ENDPOINT_SIZE.configured:
            prompt_tokens = min(prompt_tokens, ENDPOINT_PROMPT_TOKEN_BUDGET)
    if exact_tokenizer:
        # JSON's default ensure_ascii encoding can multiply Unicode source bytes.
        # Exact token admission below is authoritative; keep a finite wire bound
        # that cannot reject a prompt solely because of serialized escaping.
        return min(
            ENDPOINT_VERIFIED_MAX_REQUEST_BYTES,
            prompt_tokens * ENDPOINT_VERIFIED_BYTES_PER_TOKEN,
        ), prompt_tokens
    if type(output_tokens) is not int or output_tokens <= 0:
        return stage_limit, prompt_tokens
    output_reserve_delta = ENDPOINT_OUTPUT_BUDGET_TOKENS - output_tokens
    stage_adjusted_limit = stage_limit + int(
        output_reserve_delta * ENDPOINT_REQUEST_BYTES_PER_TOKEN
    )
    endpoint_prompt_bytes = int(prompt_tokens * ENDPOINT_REQUEST_BYTES_PER_TOKEN)
    return max(0, min(stage_adjusted_limit, endpoint_prompt_bytes)), prompt_tokens


def _wire_bytes(value: object) -> int:
    return len(json.dumps(value).encode())


def _schema_bytes(tools: Sequence[object]) -> int:
    # Canonical bytes make the cap deterministic across dict insertion order;
    # the provider wire byte count remains recorded separately.
    return len(json.dumps(
        list(tools), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode())


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(tool.get("name", ""))


def final_tool_names(body: Mapping[str, object]) -> frozenset[str]:
    """Names present in the serialized, post-diet tool schema."""
    tools = body.get("tools", ())
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        return frozenset()
    return frozenset(name for name in (_tool_name(tool) for tool in tools) if name)


def validate_final_schema(body: Mapping[str, object], offered_actions: Sequence[object], *,
                         dispatchable: Callable[[str], bool] | None = None) -> dict[str, object]:
    """Validate recovery/menu actions against the actual final wire schema."""
    available = final_tool_names(body)
    offered = tuple(dict.fromkeys(str(item) for item in offered_actions if str(item)))
    missing = tuple(item for item in offered if item not in available)
    refused = tuple(item for item in offered if dispatchable is not None
                    and item in available and not dispatchable(item))
    return {"valid": not missing and not refused, "available": sorted(available),
            "offered": list(offered), "missing": list(missing), "dispatch_refused": list(refused),
            "status": "VALID" if not missing and not refused else HOST_CONTRACT_LIMIT}


def _section_kind(message: Mapping[str, object]) -> str:
    explicit = message.get("_request_budget_section")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    if message.get("_stateframe_owner_block"):
        return "completion"
    content = str(message.get("content", "")).lower()
    if content.startswith("edit turn ") and ". continue the edit." in content:
        return "contract"
    if any(marker in content for marker in ("completion contract", "stage exit", "call done now")):
        return "completion"
    for marker in _OPTIONAL_SECTIONS:
        if f"[{marker}]" in content or content.startswith(f"{marker}:"):
            return marker
    return "required"


def _is_framefit5_stateframe(message: object) -> bool:
    if not isinstance(message, Mapping) or message.get("role") != "user":
        return False
    content = message.get("content")
    return (
        isinstance(content, str)
        and content.startswith("[StateFrame]\n")
        and ('"framefit5"' in content or "stateframe-framefit5/1" in content)
    )


def _is_stateframe_message(message: object) -> bool:
    if not isinstance(message, Mapping) or message.get("role") != "user":
        return False
    content = message.get("content")
    return isinstance(content, str) and content.startswith("[StateFrame]\n")


def _contains_framefit5_stateframe(messages: object) -> bool:
    return isinstance(messages, Sequence) and not isinstance(
        messages, (str, bytes, bytearray)
    ) and any(_is_framefit5_stateframe(message) for message in messages)


def _protected_message(message: Mapping[str, object]) -> bool:
    if _is_framefit5_stateframe(message):
        return True
    role = str(message.get("role", ""))
    section = _section_kind(message)
    if section in _OPTIONAL_SECTIONS:
        return False
    return role not in {"system", "developer"} or section in _PROTECTED_SECTIONS


def _reservation_kind(message: Mapping[str, object]) -> str | None:
    explicit = message.get("_request_budget_reservation")
    if isinstance(explicit, str) and explicit.strip():
        value = explicit.strip().lower()
        if value in RESERVATION_ORDER:
            return value
    section = _section_kind(message)
    content = str(message.get("content", ""))
    if section == "failure_packet" or content.lstrip().lower().startswith(
            ("[raising-h", "[latest failure", "failure evidence:")):
        return "latest_failure"
    if section == "task" or content.lstrip().lower().startswith(("task:", "ticket:", "brief:")):
        return "ticket"
    from kernel_lean import is_truth_content
    if is_truth_content(content):
        return "required_truth"
    if section == "file_excerpts" or any(marker in content for marker in (
            "[Current source:", "[Working source:", "[Model source window]",
            "[Request budget source window]",
    )):
        return "required_source"
    return None


def _ordered_reserved_messages(
        messages: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Reserve required evidence in one stable order before optional diet."""
    groups: dict[str, list[Mapping[str, object]]] = {name: [] for name in RESERVATION_ORDER}
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        kind = _reservation_kind(message)
        if kind is not None:
            groups[kind].append(message)
    failure_rows = groups["latest_failure"]
    groups["latest_failure"] = failure_rows[-1:] if failure_rows else []
    try:
        from kernel_request_diet import _paired_tool_exchange_messages
        groups["tool_result_pairs"].extend(
            _paired_tool_exchange_messages(messages)
        )
    except (ImportError, TypeError, ValueError):
        pass
    ordered: list[Mapping[str, object]] = []
    seen: set[int] = set()
    for kind in RESERVATION_ORDER:
        for message in groups[kind]:
            marker = id(message)
            if marker not in seen:
                seen.add(marker)
                ordered.append(message)
    return tuple(ordered)


def _reserved_tool_names(tools: Sequence[object]) -> tuple[str, ...]:
    """Return explicitly legal tool schemas without guessing from prose."""
    names: list[str] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        function_map = function if isinstance(function, Mapping) else {}
        required = (
            tool.get("_request_budget_reservation") == "legal_schemas"
            or tool.get("_request_budget_required") is True
            or tool.get("_legal_schema") is True
            or function_map.get("x-stateframe-required") is True
        )
        name = _tool_name(tool)
        if required and name and name not in names:
            names.append(name)
    return tuple(names)


def _essential_owner_messages(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        message for message in messages
        if isinstance(message, Mapping)
        and (
            bool(message.get("_stateframe_owner_block"))
            or message.get("_request_budget_reservation") == "owner_body"
        )
    )


def _reexpand_owner_messages(
        body: dict[str, object],
        owner_messages: Sequence[Mapping[str, object]],
        *,
        max_request_bytes: int,
) -> tuple[str, ...]:
    """Re-add omitted owner bodies only when the post-diet budget can carry them."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return ()
    present = {
        str(message.get("_stateframe_owner_block"))
        for message in messages
        if isinstance(message, Mapping) and message.get("_stateframe_owner_block")
    }
    reexpanded: list[str] = []
    for message in owner_messages:
        block_id = str(message.get("_stateframe_owner_block", "")).strip()
        if not block_id or block_id in present:
            continue
        candidate = [*messages, copy.deepcopy(dict(message))]
        if _wire_bytes({**body, "messages": candidate}) > max_request_bytes:
            continue
        messages.append(copy.deepcopy(dict(message)))
        present.add(block_id)
        reexpanded.append(block_id)
    return tuple(reexpanded)


def _tool_priority(name: str, stage: str, protected: set[str], index: int) -> tuple[int, int]:
    if name in protected:
        return (10_000, index)
    if name in _LOW_RELEVANCE_TOOLS:
        return (100, index)
    if name in _CORE_TOOLS:
        return (900, index)
    # Existing host menu order is the final relevance signal for an untyped
    # tool.  Stage-specific menus therefore remain stable without guessing
    # from tool descriptions or model prose.
    return (700 if stage in STAGES else 500, index)


def _protected_surface_exceeds_cap(
        protected_tools: Sequence[object], budget: StageRequestBudget,
) -> bool:
    """Apply the protected-wins invariant to both tool and schema caps."""
    return (
        len(protected_tools) > budget.max_tools
        or _schema_bytes(protected_tools) > budget.max_schema_bytes
    )


def _emit_budget_receipt(
        emit: Callable[..., object] | None, *,
        kind: str,
        stage: str,
        budget: StageRequestBudget,
        action: str,
        reason: str,
        request_bytes_before: int,
        request_bytes_after: int,
        before_tool_count: int,
        after_tool_count: int,
        before_schema_bytes: int,
        after_schema_bytes: int,
        dropped_tools: Sequence[str],
        dropped_sections: Sequence[str],
        preserved_tools: Sequence[str],
        request_id: str | None,
        step: int | None,
        **extra: object,
) -> dict[str, object]:
    """Record a degradation without turning provider-bound work into a throw."""
    detail: dict[str, object] = {
        "event_schema": SCHEMA,
        "stage": stage,
        "action": action,
        "reason": reason,
        "max_tools": budget.max_tools,
        "max_schema_bytes": budget.max_schema_bytes,
        "max_request_bytes": budget.max_request_bytes,
        "request_bytes_before": request_bytes_before,
        "request_bytes_after": request_bytes_after,
        "before_tool_count": before_tool_count,
        "after_tool_count": after_tool_count,
        "before_schema_bytes": before_schema_bytes,
        "after_schema_bytes": after_schema_bytes,
        "dropped_tools": list(dropped_tools),
        "dropped_sections": list(dropped_sections),
        "preserved_tools": list(preserved_tools),
        "request_id": request_id,
        "step": step,
        "producer": "kernel_request_budget",
        "trim_order": list(TRIM_ORDER),
    }
    detail.update(extra)
    if emit is not None:
        # Emit boundary: receipts forwarded via **extra may carry journal-RESERVED keys
        # (a wire receipt's own ``schema``); rename them, never raise, never drop the receipt.
        from kernel_continuation_journal import journal_safe_fields
        emit(kind, **journal_safe_fields(detail))
    return detail


def apply_budget(
        body: dict[str, object], *, stage: str,
        stage_exit_actions: Sequence[str] | None = None,
        completion_tool_names: Sequence[str] | None = None,
        reserved_tool_names: Sequence[str] = (),
        protected_completion_surface: bool = False,
        completion_only: bool = False,
        budget: StageRequestBudget | None = None,
        emit: Callable[..., object] | None = None,
        request_id: str | None = None,
        step: int | None = None,
        protected_messages: Sequence[Mapping[str, object]] = (),
) -> BudgetDecision:
    """Trim typed optional surface while preserving required action schemas."""
    if not isinstance(body, dict):
        raise TypeError("request budget requires a mutable request dictionary")
    selected_budget = budget or stage_budget(stage)
    if selected_budget.stage != stage:
        raise ValueError("budget stage does not match request stage")
    tools_value = body.get("tools", [])
    if not isinstance(tools_value, list):
        raise TypeError("request budget requires tools to be a list")
    messages_value = body.get("messages", [])
    if not isinstance(messages_value, list):
        raise TypeError("request budget requires messages to be a list")
    before_tools = list(tools_value)
    before_messages = list(messages_value)
    protected_ids = {id(message) for message in protected_messages}
    protected_stateframe = any(
        _is_stateframe_message(message) for message in protected_messages
    )

    def is_budget_protected(message: object) -> bool:
        if not isinstance(message, Mapping):
            return False
        return (
            id(message) in protected_ids
            or _is_framefit5_stateframe(message)
            or (protected_stateframe and _is_stateframe_message(message))
            or _protected_message(message)
            or any(dict(message) == dict(item) for item in protected_messages)
        )

    before_request_bytes = _wire_bytes(body)
    before_schema = _schema_bytes(before_tools)
    stage_required = set(
        DEFAULT_STAGE_EXIT_ACTIONS[stage]
        if stage_exit_actions is None else stage_exit_actions
    )
    completion_required = {
        str(name) for name in completion_tool_names or ()
        if str(name) not in {_PROTECTED_SURFACE_MARKER, _COMPLETION_ONLY_MARKER}
    }
    reserved = {str(name) for name in reserved_tool_names if str(name)}
    required = completion_required if completion_only and completion_required else stage_required | completion_required
    required.update(reserved)
    from kernel_agent_tools import is_registered
    required.update(_tool_name(tool) for tool in before_tools if is_registered(_tool_name(tool)))
    protected_tools: list[object] = []
    protected_seen: set[str] = set()
    for tool in before_tools:
        name = _tool_name(tool)
        if name in required and name not in protected_seen:
            protected_tools.append(copy.deepcopy(tool))
            protected_seen.add(name)
    protected_names = tuple(_tool_name(tool) for tool in protected_tools)
    protected_schema = _schema_bytes(protected_tools)
    protected_over_cap = _protected_surface_exceeds_cap(protected_tools, selected_budget)
    if protected_over_cap and protected_completion_surface:
        _emit_budget_receipt(
            emit,
            kind="request_budget_applied",
            stage=stage,
            budget=selected_budget,
            action="preserve_completion_contract",
            reason="completion_contract_exempt_from_ordinary_tool_cap",
            request_bytes_before=before_request_bytes,
            request_bytes_after=before_request_bytes,
            before_tool_count=len(before_tools),
            after_tool_count=len(protected_tools),
            before_schema_bytes=before_schema,
            after_schema_bytes=protected_schema,
            dropped_tools=(),
            dropped_sections=(),
            preserved_tools=protected_names,
            request_id=request_id,
            step=step,
            protected_completion_surface=True,
            protected_tool_count=len(protected_tools),
        )
    if protected_over_cap and not protected_completion_surface:
        reason = (
            "protected_tool_set_exceeds_max_tools"
            if len(protected_tools) > selected_budget.max_tools
            else "protected_tool_schema_exceeds_max_schema_bytes"
        )
        _emit_budget_receipt(
            emit,
            kind="request_budget_exceeded",
            stage=stage,
            budget=selected_budget,
            action="budget_exceeded",
            reason=reason,
            request_bytes_before=before_request_bytes,
            request_bytes_after=before_request_bytes,
            before_tool_count=len(before_tools),
            after_tool_count=len(protected_tools),
            before_schema_bytes=before_schema,
            after_schema_bytes=protected_schema,
            dropped_tools=(),
            dropped_sections=(),
            preserved_tools=protected_names,
            request_id=request_id,
            step=step,
            protected_tool_count=len(protected_tools),
        )
    candidates = [
        (index, tool) for index, tool in enumerate(before_tools)
        if _tool_name(tool) not in protected_seen
    ]
    candidates.sort(key=lambda pair: (
        -_tool_priority(_tool_name(pair[1]), stage, required, pair[0])[0], pair[0]))
    selected = list(protected_tools)
    dropped_tools: list[str] = []
    candidate_limit = (
        0 if completion_only or protected_over_cap
        else len(candidates)
    )
    for index, tool in candidates[:candidate_limit]:
        name = _tool_name(tool)
        proposed = [*selected, copy.deepcopy(tool)]
        if len(proposed) > selected_budget.max_tools or _schema_bytes(proposed) > selected_budget.max_schema_bytes:
            dropped_tools.append(name or f"<unnamed:{index}>")
            continue
        selected = proposed
    # Preserve source order after selection, while keeping the host-selected
    # relevance set deterministic.
    selected_names_set = {_tool_name(tool) for tool in selected}
    selected = [copy.deepcopy(tool) for tool in before_tools
                if _tool_name(tool) in selected_names_set]
    dropped_tools = [name for name in (_tool_name(tool) for tool in before_tools)
                     if name and name not in selected_names_set]
    after_messages = list(before_messages)
    dropped_sections: list[str] = []
    # A protected completion surface may exceed ordinary tool/schema caps by
    # contract.  Only an actual request-byte/hard-ceiling failure is blocking.
    request_budget_exceeded = protected_over_cap and not protected_completion_surface
    while _wire_bytes({**body, "messages": after_messages, "tools": selected}) > selected_budget.max_request_bytes:
        removable = [
            (index, _section_kind(message), _wire_bytes(message))
            for index, message in enumerate(after_messages)
            if isinstance(message, Mapping)
            and not is_budget_protected(message)
            and _section_kind(message) in _OPTIONAL_SECTIONS
        ]
        if not removable:
            request_budget_exceeded = True
            _emit_budget_receipt(
                emit,
                kind="request_budget_exceeded",
                stage=stage,
                budget=selected_budget,
                action="budget_exceeded",
                reason="required_request_sections_exceed_max_request_bytes",
                request_bytes_before=before_request_bytes,
                request_bytes_after=_wire_bytes({
                    **body, "messages": after_messages, "tools": selected,
                }),
                before_tool_count=len(before_tools),
                after_tool_count=len(selected),
                before_schema_bytes=before_schema,
                after_schema_bytes=_schema_bytes(selected),
                dropped_tools=dropped_tools,
                dropped_sections=dropped_sections,
                preserved_tools=sorted(selected_names_set & required),
                request_id=request_id,
                step=step,
                protected_sections_only=True,
            )
            break
        index, section, _ = sorted(removable, key=lambda row: (row[1] != "optional", -row[2]))[0]
        dropped_sections.append(section)
        removed = after_messages.pop(index)
        if section == "skill_bodies":
            from kernel_skills import note_budget_omission
            note_budget_omission(after_messages, removed)
    body["tools"] = selected
    body["messages"] = after_messages
    after_request_bytes = _wire_bytes(body)
    after_schema = _schema_bytes(selected)
    if after_request_bytes > selected_budget.max_request_bytes and not request_budget_exceeded:
        request_budget_exceeded = True
        _emit_budget_receipt(
            emit,
            kind="request_budget_exceeded",
            stage=stage,
            budget=selected_budget,
            action="budget_exceeded",
            reason="selected_request_surface_exceeds_budget",
            request_bytes_before=before_request_bytes,
            request_bytes_after=after_request_bytes,
            before_tool_count=len(before_tools),
            after_tool_count=len(selected),
            before_schema_bytes=before_schema,
            after_schema_bytes=after_schema,
            dropped_tools=dropped_tools,
            dropped_sections=dropped_sections,
            preserved_tools=sorted(selected_names_set & required),
            request_id=request_id,
            step=step,
            protected_sections_only=True,
        )
    # The only schemas permitted above the ordinary cap are the protected
    # ones; optional candidates are rejected while building `selected`.
    if after_schema > selected_budget.max_schema_bytes and not protected_over_cap:
        request_budget_exceeded = True
        _emit_budget_receipt(
            emit,
            kind="request_budget_exceeded",
            stage=stage,
            budget=selected_budget,
            action="budget_exceeded",
            reason="selected_request_schema_exceeds_budget",
            request_bytes_before=before_request_bytes,
            request_bytes_after=after_request_bytes,
            before_tool_count=len(before_tools),
            after_tool_count=len(selected),
            before_schema_bytes=before_schema,
            after_schema_bytes=after_schema,
            dropped_tools=dropped_tools,
            dropped_sections=dropped_sections,
            preserved_tools=sorted(selected_names_set & required),
            request_id=request_id,
            step=step,
            protected_sections_only=True,
        )
    changed = (selected != before_tools or after_messages != before_messages)
    decision = BudgetDecision(
        stage=stage, changed=changed,
        before_request_bytes=before_request_bytes, after_request_bytes=after_request_bytes,
        before_tool_count=len(before_tools), after_tool_count=len(selected),
        before_schema_bytes=before_schema, after_schema_bytes=after_schema,
        dropped_tools=tuple(dropped_tools), dropped_sections=tuple(dropped_sections),
        preserved_tools=tuple(sorted(selected_names_set & required)),
        reason=("budget_exceeded" if request_budget_exceeded
                else ("trimmed" if changed else "within_budget")),
    )
    if emit is not None:
        action = (
            "preserve_completion_contract"
            if protected_completion_surface and len(selected) > selected_budget.max_tools
            else ("trim" if changed else "noop")
        )
        reason = (
            "completion_contract_exempt_from_ordinary_tool_cap"
            if action == "preserve_completion_contract"
            else decision.reason
        )
        emit(
            "request_budget_applied", event_schema=SCHEMA, stage=stage,
            action=action, reason=reason,
            max_tools=selected_budget.max_tools,
            max_schema_bytes=selected_budget.max_schema_bytes,
            max_request_bytes=selected_budget.max_request_bytes,
            request_bytes_before=before_request_bytes,
            request_bytes_after=after_request_bytes,
            before_tool_count=len(before_tools), after_tool_count=len(selected),
            before_schema_bytes=before_schema, after_schema_bytes=after_schema,
            dropped_tools=list(decision.dropped_tools),
            dropped_sections=list(decision.dropped_sections),
            preserved_tools=list(decision.preserved_tools), request_id=request_id,
            step=step, producer="kernel_request_budget", trim_order=list(TRIM_ORDER),
            protected_completion_surface=protected_completion_surface,
            completion_only=completion_only,
            protected_tool_count=len(selected) if protected_completion_surface else 0,
            before_tool_names=[_tool_name(tool) for tool in before_tools],
            after_tool_names=[_tool_name(tool) for tool in selected],
        )
    return decision


_DIET_PROTECTED_SECTIONS = frozenset({
    "system_prompt", "task", "stateframe", "skills", "settled_state", "menus",
    "completion_contract", "continuation_limits", "stage_evidence",
})


OBSERVATION_EXCERPT_MARKER = "[observation excerpt]"
_EXCERPT_HEAD_LINES = 3
_EXCERPT_TAIL_LINES = 12
_RETRIEVAL_ROUTE = re.compile(
    r"\b(?:cursor=\S+|full_output_handle=\S+|receipt_id=\S+|expansion_handle\"?:\"?expand:[0-9a-f]+)")


def _required_source_paths(runtime: object | None) -> tuple[str, ...] | None:
    """Target plus editable product paths whose source bodies are reserved.

    ``None`` (no runtime, or no resolvable target) keeps the legacy rule that
    every bounded source window is protected.
    """
    if runtime is None:
        return None
    try:
        from kernel_request_source_window import _path_from_runtime, editable_product_paths
        target = _path_from_runtime(runtime)
        editable = editable_product_paths(runtime)
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if not target:
        return None
    paths = [str(target).replace("\\", "/")]
    paths.extend(str(path).replace("\\", "/") for path in editable if str(path))
    return tuple(dict.fromkeys(paths))


def _record_lean_required_source_paths(capture: object | None, runtime: object | None) -> None:
    """Freeze the task source scope before any capture-budget fast path returns."""
    if capture is None:
        return
    from kernel_lean import enabled as lean_enabled

    required_paths = _required_source_paths(runtime)
    if lean_enabled() and isinstance(required_paths, tuple):
        plan = _request_source_plan_for_request(runtime, capture)
        units = plan.get("units") if isinstance(plan, Mapping) else None
        if isinstance(units, Sequence) and not isinstance(units, (str, bytes, bytearray)):
            planned_paths = {
                str(unit.get("path", "")).replace("\\", "/").strip()
                for unit in units if isinstance(unit, Mapping)
                and str(unit.get("path", "")).strip()
            }
            # Keep the target fail-closed. Other editable paths become required
            # only when this request's source closure actually selects them.
            if required_paths and required_paths[0] in planned_paths:
                required_paths = tuple(path for path in required_paths if path in planned_paths)
    setattr(
        capture,
        "_lean_required_source_paths",
        tuple(required_paths or ()) if lean_enabled() else (),
    )


def _lean_append_only_receipt(
        capture: object, body: Mapping[str, object], *, boundary: str,
) -> dict[str, object]:
    """Prove prepared history remains an exact prefix and account source/truth separately."""
    reference = getattr(capture, "lean_prepared_history", None)
    messages = body.get("messages")
    journal = getattr(capture, "journal", None)
    emit = getattr(journal, "emit", None)
    request_id = getattr(capture, "request_id", None)
    if (not isinstance(reference, list)
            or not isinstance(messages, list)
            or len(messages) < len(reference)
            or messages[:len(reference)] != reference):
        changed_index = (
            next((index for index, (expected, actual) in enumerate(zip(reference, messages))
                  if expected != actual),
                 min(len(reference), len(messages)) if len(reference) != len(messages) else None)
            if isinstance(reference, list) and isinstance(messages, list) else None
        )
        changed_fields: tuple[str, ...] = ()
        if (changed_index is not None and changed_index < len(reference)
                and changed_index < len(messages)
                and isinstance(reference[changed_index], Mapping)
                and isinstance(messages[changed_index], Mapping)):
            expected_message = reference[changed_index]
            actual_message = messages[changed_index]
            changed_fields = tuple(sorted(
                key for key in set(expected_message) | set(actual_message)
                if expected_message.get(key) != actual_message.get(key)
            ))
        if callable(emit):
            emit(
                "lean_history_append_only_refused", producer="kernel_request_budget",
                request_id=request_id, boundary=boundary,
                prepared_count=len(reference) if isinstance(reference, list) else None,
                actual_count=len(messages) if isinstance(messages, list) else None,
                first_changed_index=changed_index,
                first_changed_fields=changed_fields,
                status=HOST_CONTRACT_LIMIT, dispatch=False,
            )
        raise LeanHistoryMutationError(
            "LEAN downstream request stage changed the prepared transcript "
            f"at message index {changed_index} (fields={changed_fields})")
    from kernel_lean import is_source_content, is_truth_content, message_digest
    source_rows = [message for message in messages
                   if isinstance(message, Mapping)
                   and is_source_content(message.get("content"))]
    truth_rows = [message for message in messages
                  if isinstance(message, Mapping)
                  and (
                      is_truth_content(message.get("content"))
                      or any(is_truth_content(line)
                             for line in str(message.get("content", "")).splitlines())
                  )]
    appended = messages[len(reference):]
    receipt = {
        "boundary": boundary,
        "history_message_count": len(reference),
        "history_sha256": message_digest(reference),
        "appended_message_count": len(appended),
        "source_message_count": len(source_rows),
        "source_sha256": message_digest(source_rows),
        "truth_message_count": len(truth_rows),
        "truth_sha256": message_digest(truth_rows),
    }
    setattr(capture, "lean_append_only_receipt", receipt)
    if callable(emit):
        emit("lean_history_append_only_preserved", producer="kernel_request_budget",
             request_id=request_id, dispatch=False, **receipt)
    return receipt


def _request_source_plan_for_request(
        runtime: object | None, capture: object | None,
) -> Mapping[str, object] | None:
    override = getattr(capture, "_request_source_closure_override", None)
    if isinstance(override, Mapping):
        return override
    plan = getattr(runtime, "_request_source_closure", None) if runtime is not None else None
    return plan if isinstance(plan, Mapping) else None


def _refresh_request_source_plan_after_rebind(
        runtime: object | None, *, source_plan: dict[str, object] | None,
        source_plan_before: dict[str, object] | None,
        has_capture_override: bool,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Follow a newly bound closure without mutating the runtime authority."""
    if has_capture_override or runtime is None:
        return source_plan, source_plan_before
    rebound = getattr(runtime, "_request_source_closure", None)
    if not isinstance(rebound, Mapping):
        return source_plan, source_plan_before
    request_plan = copy.deepcopy(dict(rebound))
    baseline = source_plan_before
    if baseline is None:
        baseline = copy.deepcopy(request_plan)
    return request_plan, baseline


def _initial_request_deferred_paths(
        capture: object | None, source_plan: Mapping[str, object] | None,
) -> tuple[str, ...]:
    if not isinstance(getattr(capture, "_request_source_closure_override", None), Mapping):
        return ()
    raw_paths = source_plan.get("deferred_paths", ()) if isinstance(source_plan, Mapping) else ()
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
        return ()
    return tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in raw_paths if isinstance(path, str) and path.strip()
    ))


def _required_source_plan_paths(
        runtime: object | None, base_paths: Sequence[str] | None,
        extra_paths: Sequence[str] = (),
        *, source_plan: Mapping[str, object] | None = None,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    """Protect every selected closure path and preserve its relevance order."""
    plan = source_plan
    if not isinstance(plan, Mapping):
        plan = getattr(runtime, "_request_source_closure", None) if runtime is not None else None
    units = plan.get("units", ()) if isinstance(plan, Mapping) else ()
    unit_paths = tuple(dict.fromkeys(
        str(item.get("path", "")).replace("\\", "/").strip()
        for item in units if isinstance(item, Mapping) and str(item.get("path", "")).strip()
    ))
    unit_path_set = set(unit_paths)
    paths: list[str] = []
    paths.extend(str(path).replace("\\", "/").strip()
                 for path in (base_paths or ()) if str(path).strip())
    paths.extend(unit_paths)
    paths.extend(str(path).replace("\\", "/").strip()
                 for path in extra_paths if str(path).strip())
    required = tuple(dict.fromkeys(paths))
    if not required:
        return None, ()
    ordered: list[str] = []
    if isinstance(plan, Mapping):
        ordered.extend(
            str(path).replace("\\", "/").strip()
            for path in plan.get("paths", ())
            if str(path).replace("\\", "/").strip() in unit_path_set
        )
    ordered.extend(path for path in unit_paths if path not in ordered)
    return required, tuple(ordered)


def _request_scoped_source_plan(
        source_plan: Mapping[str, object],
        deferred_paths: Sequence[str],
) -> dict[str, object]:
    """Copy the reusable closure with only this serialized request's deferrals."""
    verification_plan = copy.deepcopy(dict(source_plan))
    units = verification_plan.get("units")
    if isinstance(units, list):
        verification_plan["units"] = [
            unit for unit in units
            if not (
                isinstance(unit, Mapping)
                and unit.get("role") == "bounded_closure_omission"
            )
        ]
    current_paths = tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in deferred_paths if str(path).strip()
    ))
    verification_plan["deferred_paths"] = list(current_paths)
    return verification_plan


def _settle_blank_source_paths(
        plan: dict[str, object], messages: list[dict[str, object]],
        deferred_paths: Sequence[object], *, runtime: object | None = None,
) -> tuple[str, ...]:
    """Settle blank-only source messages and keep request deferrals in sync."""
    from kernel_request_source_window import settle_blank_source_windows

    settled = settle_blank_source_windows(plan, messages)
    current = plan.get("deferred_paths", ())
    existing = (
        current if isinstance(current, Sequence)
        and not isinstance(current, (str, bytes, bytearray)) else ()
    )
    paths = tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in (*deferred_paths, *existing, *settled) if str(path).strip()
    ))
    plan["deferred_paths"] = list(paths)
    if runtime is not None:
        setattr(runtime, "_request_deferred_source_paths", paths)
    return paths


def _remove_deferred_source_excerpts(
        messages: list[dict[str, object]], deferred_paths: Sequence[str], *,
        preserve_stateframe: bool = False,
        protected_messages: Sequence[Mapping[str, object]] = (),
) -> int:
    """Remove source excerpts for paths replaced by a verified read route."""
    normalized_paths = {
        str(path).replace("\\", "/").strip()
        for path in deferred_paths if str(path).strip()
    }
    if not normalized_paths:
        return 0
    from kernel_request_diet import _required_source_excerpt_path

    retained: list[dict[str, object]] = []
    removed = 0
    from kernel_stateframe_f4 import latest_complete_tool_exchange
    newest_exchange = latest_complete_tool_exchange(messages) if preserve_stateframe else ()
    for index, message in enumerate(messages):
        if message in protected_messages:
            retained.append(message)
            continue
        if preserve_stateframe and (message in (newest_exchange or ()) or _is_stateframe_message(message)):
            retained.append(message)
            continue
        source_path = (
            _required_source_excerpt_path(message, index)
            if isinstance(message, Mapping) else None
        )
        if (source_path is not None
                and source_path.replace("\\", "/").strip() in normalized_paths):
            removed += 1
            continue
        retained.append(message)
    messages[:] = retained
    return removed


def _remember_dropped_ranges_notice(runtime: object | None, audit: Mapping[str, object]) -> None:
    if runtime is None:
        return
    notice = audit.get("dropped_ranges_notice_text")
    if isinstance(notice, str) and "[Request diet: dropped source ranges]" in notice:
        setattr(runtime, "_request_dropped_ranges_notice", notice)


def _request_token_meter(
        runtime: object | None, capture: object | None, *,
        tokenizer: Callable[[str], object] | None = None,
        tokenizer_endpoint: str | None = None,
        request_id: str | None = None,
) -> RequestTokenMeter:
    from kernel_request_diet import RequestTokenMeter, resolve_tokenize_url, tokenizer_contract

    resolved_endpoint = resolve_tokenize_url(
        runtime, capture, endpoint=(tokenizer_endpoint or getattr(capture, "_tokenize_endpoint", None)),
    )
    routed = _frame_routes.ROUTES_ACTIVE and getattr(capture, "frame_route", None) is not None
    holders = (capture,) if routed else (runtime, capture)
    meter = None
    for owner in holders:
        if owner is None:
            continue
        try:
            candidate = getattr(owner, "_request_token_meter", None)
        except Exception:
            continue
        if isinstance(candidate, RequestTokenMeter):
            meter = candidate
            break
    if meter is None:
        meter = RequestTokenMeter(
            tokenizer=tokenizer,
            endpoint=resolved_endpoint,
            **({"route_allow_remote": bool(capture.frame_route.settings.get("allow_remote"))}
               if routed else {}),
        )
        for owner in holders:
            if owner is None:
                continue
            try:
                setattr(owner, "_request_token_meter", meter)
                break
            except (AttributeError, TypeError):
                continue
    elif resolved_endpoint is not None:
        # Route changes invalidate cache/calibration at the next measurement,
        # but cannot reset failure state within the logical request.
        meter.endpoint = resolved_endpoint
    meter.provider_identity = tokenizer_contract(runtime, capture)
    generation = getattr(capture, "request_id", None) or request_id
    logical = (getattr(capture, "_tokenize_logical_request_id", None)
               or getattr(capture, "_replacement_lineage", None) or generation)
    if isinstance(generation, str) and isinstance(logical, str):
        meter.bind_request(generation, logical, deadline=getattr(capture, "_tokenize_deadline", None))
    meter.attach_journal(getattr(capture, "journal", None) if capture is not None else None)
    return meter


def _is_tool_exchange(message: Mapping[str, object]) -> bool:
    role = str(message.get("role", ""))
    return role == "tool" or (role == "assistant" and bool(message.get("tool_calls")))


def _latest_exchange_call_ids(messages: Sequence[object]) -> frozenset[str]:
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "assistant" \
                and message.get("tool_calls"):
            return frozenset(
                str(call.get("id", "")) for call in message.get("tool_calls") or ()
                if isinstance(call, Mapping) and str(call.get("id", "")))
    return frozenset()


def observation_excerpt(content: str) -> str | None:
    """Bound one settled observation to its head and tail, never silently.

    The head keeps the observation's identity line(s) (path/revision/cursor or
    action) and the tail keeps the verdict lines (test summary, assertion).
    The omission is explicit: line/byte counts, the full-text digest, and the
    retrieval route the host recorded in the observation, or ``UNAVAILABLE``.
    """
    lines = content.splitlines(keepends=True)
    if len(lines) <= _EXCERPT_HEAD_LINES + _EXCERPT_TAIL_LINES:
        return None
    head = lines[:_EXCERPT_HEAD_LINES]
    tail = lines[-_EXCERPT_TAIL_LINES:]
    middle = lines[_EXCERPT_HEAD_LINES:-_EXCERPT_TAIL_LINES]
    route = _RETRIEVAL_ROUTE.search(content)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    marker = (
        f"{OBSERVATION_EXCERPT_MARKER} omitted_lines={len(middle)} "
        f"omitted_bytes={len(''.join(middle).encode('utf-8'))} "
        f"full_sha256={digest} retrieval={route.group(0) if route else 'UNAVAILABLE'}\n"
    )
    joined_head = "".join(head)
    if not joined_head.endswith("\n"):
        joined_head += "\n"
    return joined_head + marker + "".join(tail)


def _fit_framefit5_headcarry_to_request_budget(
        body: Mapping[str, object], messages: list[object], *,
        max_request_bytes: int,
        max_request_tokens: int | None = None,
        measure_request_tokens: Callable[[], int] | None = None,
) -> int:
    request = {**body, "messages": messages}
    if (_wire_bytes(request) <= max_request_bytes
            and (max_request_tokens is None or measure_request_tokens is None
                 or measure_request_tokens() <= max_request_tokens)):
        return 0
    from kernel_stateframe_framefit5 import fit_headcarry_to_request_budget

    return fit_headcarry_to_request_budget(
        messages,
        max_request_bytes=max_request_bytes,
        measure_request_bytes=lambda: _wire_bytes({**body, "messages": messages}),
        max_request_tokens=max_request_tokens,
        measure_request_tokens=measure_request_tokens,
    )


def _graceful_trim_optional_messages(
        body: dict[str, object], *,
        protected_contents: Sequence[str],
        protected_messages: Sequence[Mapping[str, object]],
        max_request_bytes: int,
        required_source_paths: Sequence[str] | None = None,
        max_request_tokens: int | None = None,
        measure_request_tokens: Callable[[], int] | None = None,
        token_meter: RequestTokenMeter | None = None,
) -> tuple[str, ...]:
    """Trim only optional request material when REQDIET cannot return a body.

    Native tool exchanges and volatile [V] tails are never removed. When
    optional messages are exhausted, older tool results are bounded explicitly.
    """
    if (max_request_tokens is None) != (measure_request_tokens is None):
        raise ValueError("request token ceiling and measurement must be provided together")
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return ()
    from kernel_request_diet import RequestTokenMeter, section_name, source_window_path
    required_paths = (
        frozenset(str(path).replace("\\", "/") for path in required_source_paths)
        if required_source_paths is not None else None
    )

    def over_budget() -> bool:
        return (
            _wire_bytes({**body, "messages": messages}) > max_request_bytes
            or (max_request_tokens is not None
                and measure_request_tokens is not None
                and measure_request_tokens() > max_request_tokens)
        )

    from kernel_execution_reply_budget import fit_execution_replies
    fitted_execution = fit_execution_replies(
        body, max_request_bytes=max_request_bytes,
        measure_bytes=lambda: _wire_bytes(body),
        max_request_tokens=max_request_tokens, measure_tokens=measure_request_tokens,
    )

    def is_protected(index: int, message: object) -> bool:
        if not isinstance(message, Mapping):
            return False
        from kernel_stateframe_framefit5 import _is_volatile_tail_message
        if _is_volatile_tail_message(message):
            return True
        if _is_stateframe_message(message):
            return True
        if message.get("_stateframe_owner_block"):
            return True
        if _is_tool_exchange(message):
            return True
        content = str(message.get("content", ""))
        window_path = source_window_path(content)
        if window_path is not None and required_paths is not None:
            return window_path in required_paths
        if any(value and (content == value or value in content)
               for value in protected_contents):
            return True
        if any(dict(message) == dict(item) for item in protected_messages):
            return True
        return section_name(message, index) in _DIET_PROTECTED_SECTIONS

    from kernel_diet_accounting import RemovalAccounting

    original_messages = list(messages)
    dropped: list[str] = [f"execution_reply_fit:{call_id}" for call_id in fitted_execution]
    accounting = None
    # Only this removal phase owns immutable retained messages. Excerpt and
    # headcarry rewrites below deliberately return to whole-body measurement.
    from kernel_request_diet import diet_incr_enabled

    for exact in ((False, True) if diet_incr_enabled() else (True,)):
        while True:
            size = (_wire_bytes(body) if exact or accounting is None
                    else accounting.total)
            over = size > max_request_bytes
            # Both are required, as in over_budget(): a meter without a token
            # cap must not compare against None (DIET_INCR item 4).
            if (not over and measure_request_tokens is not None
                    and max_request_tokens is not None):
                if (not exact and accounting is not None
                        and type(token_meter) is RequestTokenMeter
                        and "measure" not in token_meter.__dict__
                        and token_meter._availability == "unavailable"):
                    tokens = token_meter._measure_with_wire_bytes(body, size)[0]
                else:
                    tokens = measure_request_tokens()
                over = tokens > max_request_tokens
            if not over:
                break
            if not exact and accounting is None:
                accounting = RemovalAccounting(body, _wire_bytes)
            candidates = [
                (index, _wire_bytes(message) if exact else accounting.item_size(message),
                 section_name(message, index))
                for index, message in enumerate(messages)
                if not is_protected(index, message)
            ]
            if not candidates:
                break
            index, _size, section = max(
                candidates, key=lambda row: (row[2] == "skill_bodies", row[1], -row[0]))
            dropped.append(section)
            removed = messages.pop(index)
            if section == "skill_bodies":
                from kernel_skills import note_budget_omission
                note_budget_omission(messages, removed)
            if not exact:
                accounting.remove(removed)
        if exact or accounting is None or accounting.verify(body):
            break
        # A failed additivity guard invalidates the entire speculative pass,
        # including earlier stop decisions. Replay from its original selection.
        messages[:] = original_messages
        dropped.clear()
    latest_ids = _latest_exchange_call_ids(messages)
    for index, message in enumerate(messages):
        if not over_budget():
            break
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if str(message.get("tool_call_id", "")) in latest_ids:
            continue
        content = message.get("content")
        if not isinstance(content, str) or OBSERVATION_EXCERPT_MARKER in content:
            continue
        excerpt = observation_excerpt(content)
        if excerpt is None or len(excerpt) >= len(content):
            continue
        message["content"] = excerpt
        dropped.append(f"observation_excerpt:{section_name(message, index)}")
    removed_heads = _fit_framefit5_headcarry_to_request_budget(
        body,
        messages,
        max_request_bytes=max_request_bytes,
        max_request_tokens=max_request_tokens,
        measure_request_tokens=measure_request_tokens,
    )
    if removed_heads:
        dropped.append(f"headcarry_fit:omitted={removed_heads}")
    if over_budget():
        from kernel_frame_optional_allocation import singleton_alias_candidate
        from kernel_stateframe_framefit5 import _state_payload, _serialize_state_message
        for message in messages:
            if not isinstance(message, dict) or not _is_stateframe_message(message):
                continue
            if not _contains_framefit5_stateframe([message]):
                continue
            payload, state = _state_payload(message)
            candidate = singleton_alias_candidate(state["framefit5"])
            if candidate is not None:
                state["framefit5"] = candidate
                message["content"] = _serialize_state_message(payload)
                dropped.append("framefit5_inventory:singleton_next_site")
            break
    dropped.extend(_bound_oversized_panels(
        body,
        messages,
        max_request_bytes,
        required_paths,
        latest_ids,
        max_request_tokens=max_request_tokens,
        measure_request_tokens=measure_request_tokens,
    ))
    body["messages"] = messages
    return tuple(dropped)


PANEL_EXCERPT_MARKER = "[panel excerpt]"
_PANEL_HEAD_BYTES = 4096


def _bound_oversized_panels(
        body: Mapping[str, object], messages: list[object], max_request_bytes: int,
        required_paths: frozenset[str] | None, latest_ids: frozenset[str], *,
        max_request_tokens: int | None = None,
        measure_request_tokens: Callable[[], int] | None = None,
) -> list[str]:
    """Last resort before the hard ceiling: bound oversized protected panels.

    L125b oracles sent 160 KB against a 49 KB hard max: a single protected
    panel line carried a 124 KB binding dump, and "unachievable" was sent
    anyway (relay dropped it 3x). Panels are bounded largest-first to a head
    excerpt with an explicit omission marker (bytes, digest). Never touched:
    system prompt, StateFrame or marked owner JSON, the authorized ticket/task,
    required source windows, or any tool exchange of the latest call.
    """
    from kernel_request_diet import source_window_path
    bounded: list[str] = []

    def eligible(message: object) -> bool:
        if not isinstance(message, dict):
            return False
        from kernel_stateframe_framefit5 import _is_volatile_tail_message
        if _is_volatile_tail_message(message):
            # The owned tail carries structured tool guidance and GREENFACT.
            # Byte excerpts can split JSON or discard required action semantics.
            return False
        content = message.get("content")
        if ((isinstance(content, str) and content.startswith("[StateFrame]\n"))
                or message.get("_stateframe_owner_block")):
            return False
        if message.get("role") in {"system", "developer"}:
            return False
        if message.get("role") == "tool" and str(message.get("tool_call_id", "")) in latest_ids:
            return False
        if message.get("tool_calls"):
            return False
        if not isinstance(content, str) or len(content.encode("utf-8")) <= 2 * _PANEL_HEAD_BYTES:
            return False
        head = content.lstrip()[:512]
        if head.startswith(("[AUTHORIZED TICKET", "Task:", "Brief:")) or PANEL_EXCERPT_MARKER in content:
            return False
        window_path = source_window_path(content)
        if window_path is not None and (required_paths is None or window_path in required_paths):
            return False
        return True

    def over_budget() -> bool:
        return (
            _wire_bytes({**body, "messages": messages}) > max_request_bytes
            or (max_request_tokens is not None
                and measure_request_tokens is not None
                and measure_request_tokens() > max_request_tokens)
        )

    while over_budget():
        candidates = [message for message in messages if eligible(message)]
        if not candidates:
            break
        largest = max(candidates, key=lambda message: len(str(message.get("content")).encode("utf-8")))
        content = str(largest["content"])
        raw = content.encode("utf-8")
        head = raw[:_PANEL_HEAD_BYTES].decode("utf-8", errors="ignore")
        largest["content"] = (
            head + f"\n{PANEL_EXCERPT_MARKER} omitted_bytes={len(raw) - len(head.encode('utf-8'))} "
            f"full_sha256={hashlib.sha256(raw).hexdigest()[:16]}"
        )
        bounded.append(f"panel_excerpt:{head.split(chr(10), 1)[0][:40]}")
    return bounded


def _deliver_authored_context(body: dict[str, object], runtime: object) -> None:
    public_context = getattr(runtime, "_authored_check_public_context", None)
    if public_context is None:
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        raise TypeError("authored-check context requires list request messages")
    from kernel_authored_check_context import deliver_public_check_context
    delivery = deliver_public_check_context(messages, public_context)
    setattr(runtime, "_authored_check_context_delivery", delivery)


_MAX_BOUNDED_OMISSION_UNITS = 128
_MAX_BOUNDED_OMISSION_UNITS_PER_PATH = 64


def _bounded_semantic_omission_unit(
        omission: Mapping[str, object], declared: set[str],
        revisions: Mapping[str, object], sources: Mapping[str, object],
) -> tuple[str, dict[str, object]] | None:
    if (
        omission.get("status") != "PARTIAL"
        or omission.get("reason") != "semantic unit exceeds bounded closure"
    ):
        return None
    path = str(omission.get("path", "")).replace("\\", "/").strip()
    range_match = re.fullmatch(r"([1-9][0-9]*)-([1-9][0-9]*)", str(omission.get("range", "")))
    continuation = omission.get("continuation")
    if (
        path not in declared or range_match is None
        or not isinstance(continuation, str)
        or re.fullmatch(r"expand:[0-9a-f]{64}", continuation) is None
    ):
        return None
    source = sources.get(path)
    revision = revisions.get(path)
    if not isinstance(source, str) or not source or not isinstance(revision, str):
        return None
    from kernel_request_evidence_closure import source_revision

    if source_revision(source) != revision:
        return None
    start_line, end_line = map(int, range_match.groups())
    line_count = len(source.splitlines(keepends=True))
    if not 1 <= start_line <= end_line <= line_count:
        return None
    return path, {
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "role": "bounded_closure_omission",
        "reason": "semantic unit exceeds bounded closure",
        "priority": 0,
        "handle": continuation,
    }


def _materialize_bounded_omission_units(plan: dict[str, object]) -> tuple[str, ...]:
    """Expose exact omitted semantic spans to the revision-bound read gate."""
    units = plan.get("units")
    omissions = plan.get("omissions")
    revisions = plan.get("revisions")
    sources = plan.get("source_contents")
    paths = plan.get("paths")
    if (
        not isinstance(units, list) or not isinstance(omissions, Sequence)
        or isinstance(omissions, (str, bytes, bytearray))
        or not isinstance(revisions, Mapping) or not isinstance(sources, Mapping)
        or not isinstance(paths, Sequence)
        or isinstance(paths, (str, bytes, bytearray))
    ):
        return ()
    declared = {str(path) for path in paths}
    existing_handles = {
        str(unit.get("unit_handle", unit.get("unit_id", unit.get("handle", ""))))
        for unit in units if isinstance(unit, Mapping)
    }
    candidates: dict[str, dict[str, dict[str, object]]] = {}
    invalid_paths: set[str] = set()
    for omission in omissions:
        if not isinstance(omission, Mapping):
            continue
        path = str(omission.get("path", "")).replace("\\", "/").strip()
        if (
            omission.get("status") != "PARTIAL"
            or omission.get("reason") != "semantic unit exceeds bounded closure"
        ):
            continue
        candidate = _bounded_semantic_omission_unit(
            omission, declared, revisions, sources,
        )
        if candidate is None:
            if path in declared:
                invalid_paths.add(path)
            continue
        candidate_path, unit = candidate
        handle = str(unit["handle"])
        prior = candidates.setdefault(candidate_path, {}).get(handle)
        if prior is not None and (
            prior.get("start_line") != unit.get("start_line")
            or prior.get("end_line") != unit.get("end_line")
        ):
            invalid_paths.add(candidate_path)
            continue
        candidates[candidate_path][handle] = unit

    materialized: list[str] = []
    added_count = 0
    for path, path_units in candidates.items():
        if (
            path in invalid_paths
            or len(path_units) > _MAX_BOUNDED_OMISSION_UNITS_PER_PATH
            or added_count + len(path_units) > _MAX_BOUNDED_OMISSION_UNITS
        ):
            continue
        for handle, unit in path_units.items():
            if handle not in existing_handles:
                units.append(unit)
                existing_handles.add(handle)
                added_count += 1
        materialized.append(path)
    return tuple(materialized)


def _bounded_omitted_handle_values(
        value: object, *, unit_count: int,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        return ()
    handles = tuple(value)
    if (
        not handles or len(handles) > unit_count
        or any(not isinstance(handle, str) or not handle.strip() for handle in handles)
    ):
        return ()
    normalized = tuple(handle.strip() for handle in handles)
    return normalized if len(set(normalized)) == len(normalized) else ()


def _declared_bounded_source_paths(value: object) -> frozenset[str] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        return None
    paths: set[str] = set()
    for raw_path in value:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        paths.add(raw_path.replace("\\", "/").strip())
    return frozenset(paths)


def _bounded_delivery_omitted_unit_paths(
        plan: Mapping[str, object],
) -> tuple[str, ...]:
    """Return omitted unit paths only when every handle is revision-bound."""
    bounded = plan.get("bounded_delivery")
    raw_handles = bounded.get("omitted_handles") if isinstance(bounded, Mapping) else None
    raw_units = plan.get("units")
    raw_paths = plan.get("paths")
    if (
        not isinstance(raw_units, Sequence)
        or isinstance(raw_units, (str, bytes, bytearray))
    ):
        return ()
    omitted_handles = _bounded_omitted_handle_values(
        raw_handles, unit_count=len(raw_units),
    )
    declared_paths = _declared_bounded_source_paths(plan.get("paths"))
    if not omitted_handles or declared_paths is None:
        return ()

    units_by_handle: dict[str, list[Mapping[str, object]]] = {}
    for unit in raw_units:
        if not isinstance(unit, Mapping):
            continue
        raw_handle = (
            unit.get("unit_handle") or unit.get("unit_id") or unit.get("handle") or ""
        )
        if not isinstance(raw_handle, str):
            continue
        handle = raw_handle.strip()
        if handle:
            units_by_handle.setdefault(handle, []).append(unit)

    revisions = plan.get("revisions")
    revisions = revisions if isinstance(revisions, Mapping) else {}
    sources = plan.get("source_contents")
    sources = sources if isinstance(sources, Mapping) else {}
    from kernel_request_evidence_closure import source_revision

    paths_with_omissions: set[str] = set()
    invalid_paths: set[str] = set()
    for handle in omitted_handles:
        matches = units_by_handle.get(handle, ())
        if len(matches) != 1:
            return ()
        unit = matches[0]
        raw_path = unit.get("path")
        if not isinstance(raw_path, str):
            return ()
        path = raw_path.replace("\\", "/").strip()
        if not path or path not in declared_paths:
            return ()
        revision = revisions.get(path)
        source = sources.get(path)
        start = unit.get("start_line")
        end = unit.get("end_line")
        if (
            not isinstance(revision, str) or not revision.strip()
            or not isinstance(source, str) or not source
            or source_revision(source) != revision
            or type(start) is not int or type(end) is not int
        ):
            invalid_paths.add(path)
            continue
        line_count = len(source.splitlines(keepends=True))
        if not 1 <= start <= end <= line_count:
            invalid_paths.add(path)
            continue
        unit_revision = str(unit.get("revision", "") or "").strip()
        if unit_revision and unit_revision != revision:
            invalid_paths.add(path)
            continue
        paths_with_omissions.add(path)

    ordered_paths = tuple(dict.fromkeys(
        str(path).replace("\\", "/").strip()
        for path in raw_paths if str(path).strip()
    ))
    return tuple(
        path for path in ordered_paths
        if path in paths_with_omissions and path not in invalid_paths
    )


def _failed_source_delivery_paths(receipt: Mapping[str, object]) -> tuple[str, ...]:
    failures = receipt.get("proof_failures", ())
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes, bytearray)):
        return ()
    paths: list[str] = []
    for failure in failures:
        if not isinstance(failure, Mapping):
            continue
        path = str(failure.get("path", "")).replace("\\", "/").strip()
        if path and path not in paths:
            paths.append(path)
    return tuple(paths)


def _candidate_path_facts(receipt: Mapping[str, object]) -> list[str]:
    rows = receipt.get("paths")
    if not isinstance(rows, Mapping):
        return []
    facts: list[str] = []
    for path, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        route = row.get("retrieval_route")
        route = route if isinstance(route, Mapping) else {}
        handles = route.get("unit_handles")
        revision = str(row.get("revision", ""))
        ranges = row.get("ranges")
        delivered_ranges = ",".join(
            f"{span[0]}-{span[1]}"
            for span in ranges
            if isinstance(span, Sequence) and not isinstance(span, (str, bytes))
            and len(span) == 2 and all(type(line) is int for line in span)
        ) if isinstance(ranges, Sequence) else ""
        facts.append(
            f"{path}:{row.get('status', 'UNKNOWN')}:{row.get('proof_status', 'UNKNOWN')}:"
            f"{route.get('status', 'NONE')}:{route.get('tool', 'NONE')}:"
            f"{str(bool(revision) and route.get('revision') == revision).lower()}:"
            f"{len(handles) if isinstance(handles, Sequence) else 0}:ranges={delivered_ranges}"
        )
        if len(facts) >= 40:
            break
    return facts


def _record_source_delivery_candidate(
        emit: Callable[..., object] | None, request_id: str | None, step: int | None,
        deferred_paths: Sequence[str], candidate_plan: Mapping[str, object],
        candidate_receipt: Mapping[str, object], satisfied: bool,
) -> bool:
    """Journal a bounded reason when an exact-wire deferral candidate is held."""
    unresolved_paths: list[str] = []
    rows = candidate_receipt.get("paths")
    if isinstance(rows, Mapping):
        for path, row in rows.items():
            if (isinstance(row, Mapping)
                    and str(row.get("status", "")).upper() not in {"COMPLETE", "DEFERRED"}):
                unresolved_paths.append(f"{path}:{row.get('status', 'UNKNOWN')}")
    proof_failures = candidate_receipt.get("proof_failures")
    proof_failure_kinds = [
        f"{row.get('kind', 'UNKNOWN')}:{row.get('path', 'UNKNOWN')}"
        for row in proof_failures
        if isinstance(row, Mapping)
    ][:40] if isinstance(proof_failures, Sequence) else []
    omissions: list[str] = []
    for key in ("omissions", "skipped_files"):
        values = candidate_plan.get(key, ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            continue
        for row in values:
            if isinstance(row, Mapping):
                span = row.get("range")
                start = row.get("start_line")
                end = row.get("end_line")
                if isinstance(span, Sequence) and not isinstance(span, (str, bytes)):
                    if len(span) == 2 and all(type(line) is int for line in span):
                        start, end = span
                range_label = (
                    f"{start}-{end}"
                    if type(start) is int and type(end) is int else ""
                )
                kind = row.get("kind", row.get("reason", "UNKNOWN"))
                handle = row.get("unit_handle", row.get("unit_id", row.get("handle", "")))
                omissions.append(
                    f"{row.get('path', 'UNKNOWN')}:{row.get('status', 'UNKNOWN')}:"
                    f"{kind}:{range_label}:{str(handle)[:120]}"
                )
                if len(omissions) >= 40:
                    break
    if emit is not None:
        emit(
            "request_source_delivery_candidate",
            producer="kernel_request_budget",
            request_id=request_id,
            step=step,
            candidate_paths=list(deferred_paths),
            candidate_path_facts=_candidate_path_facts(candidate_receipt),
            source_status=candidate_receipt.get("status"),
            proof_status=candidate_receipt.get("proof_status"),
            deferred_paths=candidate_receipt.get("deferred_paths", ()),
            proof_failure_kinds=proof_failure_kinds,
            notice_verified=candidate_receipt.get("deferred_notice_verified"),
            read_tool_available=candidate_receipt.get("read_tool_available"),
            unresolved_paths=unresolved_paths[:40],
            plan_omissions=omissions,
            blocked=not satisfied,
            dispatch=False,
        )
    return satisfied


def _frame_only_runtime(runtime: object | None) -> bool:
    return (
        runtime is not None
        and getattr(runtime, "stateframe_framefit_mode", "off") == "on"
        and any(base.__module__ == "kernel_stateframe_runtime"
                and base.__name__ == "StateFrameRuntime" for base in type(runtime).__mro__)
    )


def _live_framefit5_product_request(runtime: object | None, messages: object) -> bool:
    if runtime is None or not isinstance(messages, list):
        return False
    if not any(
            base.__module__ == "kernel_stateframe_runtime"
            and base.__name__ == "StateFrameRuntime"
            for base in type(runtime).__mro__):
        return False
    from kernel_stateframe_f4 import HANDOFF_HEADER, latest_complete_tool_exchange
    from kernel_stateframe_framefit5 import has_framefit5_state, is_frame_only
    from kernel_wire_truth_facts import FACT_SCHEMA

    if is_frame_only(messages):
        return True
    if (not has_framefit5_state(messages)
            or latest_complete_tool_exchange(messages) is None
            or any(
                str(message.get("content") or "").startswith(HANDOFF_HEADER)
                for message in messages
            )):
        return False
    fact_prefix = f"[WIRE_TRUTH_FACT {FACT_SCHEMA}]"
    return any(
        message.get("role") == "user"
        and str(message.get("content") or "").startswith(fact_prefix)
        for message in messages
    )


def _project_or_assemble_request_messages(
        messages: list[dict[str, object]], plan: Mapping[str, object], *,
        runtime: object, capture: object | None,
        allocation_plan: dict[str, object] | None,
        sources: Mapping[str, str],
        fit_selection: bool = True,
) -> int:
    """Keep every live FrameFit5 budget projection on the shared request assembler."""
    frame_request_allocation = getattr(capture, "frame_request_allocation", None)
    if frame_request_allocation is not None and fit_selection:
        # Only this host fitting pass may propose a smaller selection. Its
        # incoming allocation was checked before any diet mutation. Keep the
        # active capture unchanged until final source/capacity qualification.
        from kernel_frame_optional_allocation import selection_from_frame
        from kernel_frame_request_allocation import FrameRequestAllocation
        from kernel_stateframe_framefit5 import _state_payload
        current = next(_state_payload(msg)[1]["framefit5"] for msg in messages
                       if _is_stateframe_message(msg))
        active = frame_request_allocation
        optional, inventory = selection_from_frame(active.optional_heads, active.site_inventory, current)
        frame_request_allocation = FrameRequestAllocation.create(
            identity=active.identity, capacity=active.capacity,
            source_obligations=active.source_obligations, source_layout=active.source_layout,
            optional_heads=optional, site_inventory=inventory, outcome=active.outcome,
        )
        if list(plan.get("deferred_paths", ())) != list(active.source_layout["source_plan"].get("deferred_paths", ())):
            # A changed source decision is a new allocation, with newly proved
            # positions; it cannot reuse the previous source-placement record.
            frame_request_allocation = None
    if (frame_request_allocation is not None
            or _frame_only_runtime(runtime)
            or _live_framefit5_product_request(runtime, messages)):
        from kernel_stateframe_framefit5 import assemble_framefit5_request

        audit: dict[str, object] = {}
        assembled = assemble_framefit5_request(
            messages,
            getattr(capture, "_framefit5_semantic_messages", messages),
            source_plan=plan,
            sources=sources,
            allocation_plan=allocation_plan,
            task_message=getattr(capture, "_framefit5_task_message", None),
            runtime=runtime,
            owner_contract=(getattr(capture, "request_action_contract", None)
                            or getattr(capture, "_framefit5_owner_presentation_contract", None)),
            owner_source_messages=messages,
            projection_audit=audit,
            host_text_messages=messages,
            frame_request_allocation=frame_request_allocation,
        )
        messages[:] = assembled
        projected = audit.get("projected_count", 0)
        return projected if type(projected) is int else 0

    from kernel_request_source_window import project_model_source_messages

    return project_model_source_messages(
        messages,
        plan,
        allocation_plan=allocation_plan,
        materialize_framefit5_closure_units=False,
        sources=sources,
    )


@profiled_function_span("request_budget")
def _apply_budget_with_diet(
        body: dict[str, object], *, stage: str, stage_exit_actions: Sequence[str],
        completion_tool_names: Sequence[str], protected_completion_surface: bool,
        completion_only: bool, emit: Callable[..., object] | None,
        request_id: str | None, step: int | None,
        protected_contents: Sequence[str] = (),
        protected_messages: Sequence[Mapping[str, object]] = (),
        runtime: object | None = None,
        capture: object | None = None,
        tokenizer: Callable[[str], object] | None = None,
        tokenizer_endpoint: str | None = None,
) -> dict[str, object]:
    """Trim tools first, diet messages second, then validate the stage ceiling."""
    destination_body = body
    frame_candidate = _frame_only_runtime(runtime)
    if frame_candidate:
        # A rejected draft must not leak its source projections or notices into
        # the unallocated request. Failure receipts still hold dispatch closed.
        body = copy.deepcopy(body)
        allocation = getattr(capture, "frame_request_allocation", None)
        if allocation is not None:
            from kernel_frame_allocation_validation import validate_allocation_authority
            validate_allocation_authority(allocation, runtime, body["messages"], check_layout=False)
    if runtime is not None:
        _deliver_authored_context(body, runtime)
    ensure_ticket = getattr(capture, "_ensure_final_wire_ticket", None)
    if callable(ensure_ticket):
        ensured_body = ensure_ticket(body)
        if isinstance(ensured_body, dict):
            body = ensured_body
    ticket_identity = getattr(capture, "ticket_identity", None)
    if isinstance(ticket_identity, Mapping):
        ticket_text = ticket_identity.get("text")
        if isinstance(ticket_text, str) and ticket_text:
            protected_contents = (*protected_contents, ticket_text)
    original_tools = copy.deepcopy(body.get("tools", []))
    original_messages = copy.deepcopy(body.get("messages", []))
    request_source_plan_override = isinstance(
        getattr(capture, "_request_source_closure_override", None), Mapping,
    )
    if (frame_candidate and not request_source_plan_override
            and getattr(capture, "frame_request_allocation", None) is None):
        # A new request may follow an edit. Bind this request's closure before
        # taking its immutable snapshot, rather than consuming the prior turn's
        # closure alongside freshly rendered source bytes. Recovery overrides
        # remain frozen and still fail the ordinary freshness checks.
        from capability_request_preflight import bound_runtime_sources
        bound_runtime_sources(
            body["messages"], runtime,
            max_request_bytes=stage_budget(stage).max_request_bytes,
            tools=body["tools"],
        )
    request_source_plan = _request_source_plan_for_request(runtime, capture)
    request_source_plan_before = (
        copy.deepcopy(dict(request_source_plan))
        if isinstance(request_source_plan, Mapping) else None
    )
    if isinstance(request_source_plan, Mapping) and (frame_candidate or not request_source_plan_override):
        # Deferrals and final projection belong to this request. Preserve the
        # runtime closure as the immutable source authority for later recovery.
        request_source_plan = copy.deepcopy(dict(request_source_plan))
    optional_inputs = None
    if _frame_only_runtime(runtime):
        from kernel_stateframe_framefit5 import _state_payload, _workspace_root
        from kernel_frame_optional_allocation import snapshot_optional_inputs
        from kernel_frame_source_snapshot import frame_source_snapshot
        initial_sources = frame_source_snapshot(runtime, request_source_plan)
        _project_or_assemble_request_messages(
            body["messages"], request_source_plan,
            runtime=runtime, capture=capture, allocation_plan=None, sources=initial_sources,
            fit_selection=False,
        )
        for message in body["messages"]:
            if _is_stateframe_message(message):
                _payload, initial_state = _state_payload(message)
                active = getattr(capture, "frame_request_allocation", None)
                optional_inputs = (
                    (active.optional_heads, active.site_inventory) if active is not None
                    else snapshot_optional_inputs(initial_state["framefit5"], _workspace_root(runtime))
                )
                break
    if optional_inputs is not None:
        from kernel_stateframe_f4 import latest_complete_tool_exchange
        semantic_messages = getattr(capture, "_framefit5_semantic_messages", body["messages"])
        newest_exchange = latest_complete_tool_exchange(semantic_messages)
        task_message = getattr(capture, "_framefit5_task_message", None)
        protected_messages = (
            *protected_messages,
            *(getattr(capture, "_framefit5_newest_user_suffix", ()) or ()),
            *(newest_exchange or ()),
            *((task_message,) if isinstance(task_message, Mapping) else ()),
        )
    base_required_source_paths = _required_source_paths(runtime)
    _record_lean_required_source_paths(capture, runtime)
    required_source_paths = base_required_source_paths
    required_source_order: tuple[str, ...] = ()
    deferred_source_paths = _initial_request_deferred_paths(capture, request_source_plan)
    if runtime is not None:
        setattr(runtime, "_request_deferred_source_paths", deferred_source_paths)
        setattr(runtime, "_request_dropped_ranges_notice", None)
    reservation_messages = _ordered_reserved_messages(
        original_messages if isinstance(original_messages, list) else []
    )
    if required_source_paths is not None:
        # Keep only the editable/target source bodies in the initial reservation;
        # connected-file excerpts stay eligible for the ordinary source LRU.
        from kernel_request_diet import _source_identity, source_window_path

        def _is_required_source_reservation(message: Mapping[str, object]) -> bool:
            identity = _source_identity(message)
            source_path = (
                identity[1] if identity is not None and identity[0] in {"source", "certified"}
                else source_window_path(str(message.get("content", "")))
            )
            return source_path is None or source_path.replace("\\", "/") in required_source_paths

        reservation_messages = tuple(
            message for message in reservation_messages
            if _is_required_source_reservation(message)
        )
    owner_messages = _essential_owner_messages(
        original_messages if isinstance(original_messages, list) else []
    )
    reserved_tool_names = _reserved_tool_names(
        original_tools if isinstance(original_tools, list) else []
    )
    protected_messages = tuple(protected_messages) + reservation_messages
    if capture is not None:
        setattr(capture, "request_budget_reservation_order", RESERVATION_ORDER)
        setattr(capture, "request_budget_reserved_tool_names", reserved_tool_names)
    selected_budget = stage_budget(stage)
    diet_stage = "FREE" if stage in {"NATIVE", FREE_STAGE} else stage
    # Tool selection is a separate pass.  Do not let the final request ceiling
    # reject a large pre-diet body before message sections can be removed.
    provisional = StageRequestBudget(
        stage=stage,
        max_tools=selected_budget.max_tools,
        max_schema_bytes=selected_budget.max_schema_bytes,
        max_request_bytes=16 * 1024 * 1024,
    )
    apply_budget(
        body,
        stage=stage,
        stage_exit_actions=stage_exit_actions,
        completion_tool_names=completion_tool_names,
        reserved_tool_names=reserved_tool_names,
        protected_completion_surface=protected_completion_surface,
        completion_only=completion_only,
        budget=provisional,
        emit=None,
        request_id=request_id,
        step=step,
        protected_messages=protected_messages,
    )
    provisional_tools = copy.deepcopy(body.get("tools", []))
    owner_first_plan: dict[str, object] | None = None
    owner_requested_paths: tuple[str, ...] = ()
    if runtime is not None:
        from capability_request_preflight import bound_runtime_sources
        bound_runtime_sources(
            body["messages"], runtime,
            max_request_bytes=selected_budget.max_request_bytes,
            tools=body["tools"],
        )
        required_source_paths, required_source_order = _required_source_plan_paths(
            runtime, base_required_source_paths, source_plan=request_source_plan,
        )
    token_meter = _request_token_meter(
        runtime, capture, tokenizer=tokenizer,
        tokenizer_endpoint=tokenizer_endpoint,
        request_id=request_id,
    )
    from kernel_owner_first import allocation_enabled
    if allocation_enabled():
        from kernel_owner_first import build_priority_plan, protected_priority_messages
        owner_first_plan = build_priority_plan(body, runtime)
        protected_messages = tuple(protected_messages) + protected_priority_messages(
            body, owner_first_plan)
        owner_requested_paths = tuple(str(path) for path in owner_first_plan.get("requested_paths", ())
                                       if str(path))
        required_source_paths, required_source_order = _required_source_plan_paths(
            runtime, base_required_source_paths, owner_requested_paths,
            source_plan=request_source_plan,
        )
        if capture is not None:
            setattr(capture, "request_owner_first_allocation_plan", owner_first_plan)
    try:
        from kernel_request_diet import RequestDietDeferred, apply_diet

        def _run_diet(*, max_request_bytes: int) -> dict[str, object]:
            nonlocal deferred_source_paths
            pretrimmed = _graceful_trim_optional_messages(
                body, protected_contents=protected_contents,
                protected_messages=protected_messages, required_source_paths=required_source_paths,
                max_request_bytes=max_request_bytes,
                max_request_tokens=ENDPOINT_PROMPT_TOKEN_BUDGET,
                measure_request_tokens=lambda: token_meter.measure(body)[0],
                token_meter=token_meter,
            ) if optional_inputs is not None else ()
            audit = apply_diet(
                body,
                stage=diet_stage,
                max_request_bytes=max_request_bytes,
                emit=emit,
                request_id=request_id,
                step=step,
                protected_contents=protected_contents,
                protected_messages=protected_messages,
                required_source_paths=required_source_paths,
                required_source_order=required_source_order,
                deferred_source_paths=deferred_source_paths,
                token_budget=ENDPOINT_PROMPT_TOKEN_BUDGET,
                token_meter=token_meter,
                allocation_plan=owner_first_plan,
            )
            audit["dropped_sections"] = list(pretrimmed) + list(audit.get("dropped_sections", ()))
            raw_paths = audit.get("deferred_source_paths", ())
            if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (str, bytes, bytearray)):
                diet_paths = tuple(dict.fromkeys(
                    str(path).replace("\\", "/").strip()
                    for path in raw_paths if isinstance(path, str) and path.strip()
                ))
                deferred_source_paths = tuple(dict.fromkeys(
                    (*deferred_source_paths, *diet_paths),
                ))
            if runtime is not None:
                setattr(runtime, "_request_deferred_source_paths", deferred_source_paths)
                _remember_dropped_ranges_notice(runtime, audit)
                plan = request_source_plan
                if isinstance(plan, dict):
                    plan["deferred_paths"] = list(deferred_source_paths)
            return audit

        def _run_hard_budget_diet() -> dict[str, object]:
            nonlocal request_source_plan, request_source_plan_before
            nonlocal required_source_paths, required_source_order
            audit = _run_diet(max_request_bytes=HARD_MAX_REQUEST_BYTES)
            _reexpanded = _reexpand_owner_messages(
                body, owner_messages, max_request_bytes=HARD_MAX_REQUEST_BYTES,
            )
            if capture is not None:
                setattr(capture, "request_budget_owner_reexpanded", _reexpanded)
            if runtime is not None:
                from capability_request_preflight import bound_runtime_sources
                rebound = bound_runtime_sources(
                    body["messages"], runtime,
                    max_request_bytes=HARD_MAX_REQUEST_BYTES,
                    tools=body["tools"],
                )
                request_source_plan, request_source_plan_before = (
                    _refresh_request_source_plan_after_rebind(
                        runtime,
                        source_plan=request_source_plan,
                        source_plan_before=request_source_plan_before,
                        has_capture_override=request_source_plan_override,
                    )
                )
                required_source_paths, required_source_order = _required_source_plan_paths(
                    runtime, base_required_source_paths, owner_requested_paths,
                    source_plan=request_source_plan,
                )
                if isinstance(rebound, Mapping) and rebound.get("changed"):
                    audit = _run_diet(max_request_bytes=HARD_MAX_REQUEST_BYTES)
                    _reexpanded = _reexpand_owner_messages(
                        body, owner_messages, max_request_bytes=HARD_MAX_REQUEST_BYTES,
                    )
                    if capture is not None:
                        setattr(capture, "request_budget_owner_reexpanded", _reexpanded)
            return audit

        _run_diet(max_request_bytes=selected_budget.max_request_bytes)
        _reexpanded_owner_blocks = _reexpand_owner_messages(
            body, owner_messages, max_request_bytes=selected_budget.max_request_bytes,
        )
        if capture is not None:
            setattr(capture, "request_budget_owner_reexpanded", _reexpanded_owner_blocks)
        if runtime is not None:
            from capability_request_preflight import bound_runtime_sources
            # Menu/prose compaction changes available bytes.  Re-fit source
            # windows from authoritative files so an old small page is not
            # treated as complete merely because it fit an earlier budget.
            rebound = bound_runtime_sources(
                body["messages"], runtime,
                max_request_bytes=selected_budget.max_request_bytes,
                tools=body["tools"],
            )
            request_source_plan, request_source_plan_before = (
                _refresh_request_source_plan_after_rebind(
                    runtime,
                    source_plan=request_source_plan,
                    source_plan_before=request_source_plan_before,
                    has_capture_override=request_source_plan_override,
                )
            )
            required_source_paths, required_source_order = _required_source_plan_paths(
                runtime, base_required_source_paths, owner_requested_paths,
                source_plan=request_source_plan,
            )
            if isinstance(rebound, Mapping) and rebound.get("changed"):
                _run_diet(max_request_bytes=selected_budget.max_request_bytes)
                _reexpanded_owner_blocks = _reexpand_owner_messages(
                    body, owner_messages, max_request_bytes=selected_budget.max_request_bytes,
                )
                if capture is not None:
                    setattr(capture, "request_budget_owner_reexpanded", _reexpanded_owner_blocks)
    except RequestDietDeferred as error:
        detail = dict(error.detail)
        tools_after = body.get("tools", [])
        tool_count = len(tools_after) if isinstance(tools_after, list) else 0
        schema_bytes = _schema_bytes(tools_after) if isinstance(tools_after, list) else 0
        _emit_budget_receipt(
            emit,
            kind="request_budget_applied",
            stage=stage,
            budget=selected_budget,
            action="degraded",
            reason=str(detail.get("reason", "request diet deferred")),
            request_bytes_before=_wire_bytes(body),
            request_bytes_after=_wire_bytes(body),
            before_tool_count=tool_count,
            after_tool_count=tool_count,
            before_schema_bytes=schema_bytes,
            after_schema_bytes=schema_bytes,
            dropped_tools=(),
            dropped_sections=(
                detail.get("dropped_sections", ())
                if isinstance(detail.get("dropped_sections"), Sequence) else ()
            ),
            preserved_tools=tuple(_tool_name(tool) for tool in tools_after)
            if isinstance(tools_after, list) else (),
            request_id=request_id,
            step=step,
            source_receipt_action="request_diet_deferred",
        )
        # The first request must remain deliverable. Rebind the current source
        # window against the hard ceiling, then let REQDIET remove optional
        # sections a second time. The exceptional path is journaled below.
        hard_budget = StageRequestBudget(
            stage=stage,
            max_tools=selected_budget.max_tools,
            max_schema_bytes=selected_budget.max_schema_bytes,
            max_request_bytes=HARD_MAX_REQUEST_BYTES,
        )
        if runtime is not None:
            from capability_request_preflight import bound_runtime_sources
            bound_runtime_sources(
                body["messages"], runtime,
                max_request_bytes=HARD_MAX_REQUEST_BYTES,
                tools=body["tools"],
            )
            request_source_plan, request_source_plan_before = (
                _refresh_request_source_plan_after_rebind(
                    runtime,
                    source_plan=request_source_plan,
                    source_plan_before=request_source_plan_before,
                    has_capture_override=request_source_plan_override,
                )
            )
            required_source_paths, required_source_order = _required_source_plan_paths(
                runtime, base_required_source_paths, owner_requested_paths,
                source_plan=request_source_plan,
            )
        try:
            _run_hard_budget_diet()
        except RequestDietDeferred as hard_error:
            hard_detail = dict(hard_error.detail)

            def measure_current_tokens() -> int:
                return token_meter.measure(body)[0]

            dropped = _graceful_trim_optional_messages(
                body,
                protected_contents=protected_contents,
                protected_messages=protected_messages,
                required_source_paths=required_source_paths,
                max_request_bytes=HARD_MAX_REQUEST_BYTES,
                max_request_tokens=ENDPOINT_PROMPT_TOKEN_BUDGET,
                measure_request_tokens=measure_current_tokens,
                token_meter=token_meter,
            )
            recovered_after_shedding = False
            if dropped:
                try:
                    _run_hard_budget_diet()
                except RequestDietDeferred as retry_error:
                    hard_detail = dict(retry_error.detail)
                else:
                    recovered_after_shedding = True
                    if capture is not None:
                        setattr(capture, "request_budget_exceeded", False)
            if not recovered_after_shedding:
                hard_tools = body.get("tools", [])
                hard_tool_count = len(hard_tools) if isinstance(hard_tools, list) else 0
                hard_schema = _schema_bytes(hard_tools) if isinstance(hard_tools, list) else 0
                if capture is not None:
                    setattr(capture, "request_budget_exceeded", True)
                    setattr(capture, "request_budget_hard_max", HARD_MAX_REQUEST_BYTES)
                _emit_budget_receipt(
                    emit,
                    kind="request_budget_exceeded",
                    stage=stage,
                    budget=hard_budget,
                    action="budget_exceeded",
                    reason="budget_exceeded_hard_max",
                    request_bytes_before=_wire_bytes(body),
                    request_bytes_after=_wire_bytes(body),
                    before_tool_count=hard_tool_count,
                    after_tool_count=hard_tool_count,
                    before_schema_bytes=hard_schema,
                    after_schema_bytes=hard_schema,
                    dropped_tools=(),
                    dropped_sections=dropped,
                    preserved_tools=tuple(_tool_name(tool) for tool in hard_tools)
                    if isinstance(hard_tools, list) else (),
                    request_id=request_id,
                    step=step,
                    hard_max_request_bytes=HARD_MAX_REQUEST_BYTES,
                    hard_max_unachievable=(
                        _wire_bytes(body) > HARD_MAX_REQUEST_BYTES
                        or (
                            type(hard_detail.get("request_tokens_after")) is int
                            and hard_detail["request_tokens_after"] > ENDPOINT_PROMPT_TOKEN_BUDGET
                        )
                    ),
                    source_receipt_reason=str(hard_detail.get("reason", "request diet deferred")),
                )
        else:
            if capture is not None:
                setattr(capture, "request_budget_exceeded", True)
                setattr(capture, "request_budget_hard_max", HARD_MAX_REQUEST_BYTES)
            hard_tools = body.get("tools", [])
            hard_tool_count = len(hard_tools) if isinstance(hard_tools, list) else 0
            hard_schema = _schema_bytes(hard_tools) if isinstance(hard_tools, list) else 0
            _emit_budget_receipt(
                emit,
                kind="request_budget_exceeded",
                stage=stage,
                budget=hard_budget,
                action="budget_exceeded",
                reason="stage_ceiling_relaxed_to_hard_max",
                request_bytes_before=_wire_bytes(body),
                request_bytes_after=_wire_bytes(body),
                before_tool_count=hard_tool_count,
                after_tool_count=hard_tool_count,
                before_schema_bytes=hard_schema,
                after_schema_bytes=hard_schema,
                dropped_tools=(),
                dropped_sections=(),
                preserved_tools=tuple(_tool_name(tool) for tool in hard_tools)
                if isinstance(hard_tools, list) else (),
                request_id=request_id,
                step=step,
                stage_max_request_bytes=selected_budget.max_request_bytes,
                hard_max_request_bytes=HARD_MAX_REQUEST_BYTES,
            )
        selected_budget = hard_budget
    apply_budget(
        body,
        stage=stage,
        stage_exit_actions=stage_exit_actions,
        completion_tool_names=completion_tool_names,
        reserved_tool_names=reserved_tool_names,
        protected_completion_surface=protected_completion_surface,
        completion_only=completion_only,
        budget=selected_budget,
        emit=emit,
        request_id=request_id,
        step=step,
        protected_messages=protected_messages,
    )
    final_tools = body.get("tools") if isinstance(body.get("tools"), list) else []
    listed_paths: list[str] = []
    try:
        from kernel_serve_reads import enabled as serve_reads_enabled, files_in_frame_notice
        if serve_reads_enabled():
            final_messages, listed_paths = files_in_frame_notice(
                body.get("messages", []) if isinstance(body.get("messages"), list) else []
            )
            body["messages"] = final_messages
    except (ImportError, TypeError, ValueError):
        pass
    from kernel_aha10_honest_loop import request_boundary_receipt
    original_names = {_tool_name(tool) for tool in original_tools if _tool_name(tool)}
    provisional_names = {_tool_name(tool) for tool in provisional_tools if _tool_name(tool)}
    final_names = {_tool_name(tool) for tool in final_tools if _tool_name(tool)}
    removal_reasons = {
        name: ("provisional_tool_or_schema_limit" if name not in provisional_names
               else "final_request_or_schema_limit")
        for name in original_names if name not in final_names
    }
    contract = getattr(capture, "request_action_contract", None) if capture is not None else None
    offered_actions = contract.get("final_actions", contract.get("admitted_actions", ())) \
        if isinstance(contract, Mapping) else ()
    boundary = request_boundary_receipt(
        original_tools if isinstance(original_tools, Sequence) else (),
        provisional_tools if isinstance(provisional_tools, Sequence) else (),
        final_tools,
        protected_surface=tuple(stage_exit_actions) + tuple(completion_tool_names),
        removal_reasons=removal_reasons,
        hard_limit={
            "blocked": bool(getattr(capture, "request_budget_exceeded", False)),
            "status": "BLOCKED" if bool(getattr(capture, "request_budget_exceeded", False)) else "READY",
        },
        offered_actions=offered_actions if isinstance(offered_actions, Sequence) else (),
    )
    if capture is not None:
        setattr(capture, "request_budget_boundary_receipt", boundary)
    source_delivery: dict[str, object] | None = None
    source_plan: Mapping[str, object] | None = None
    source_contents: dict[str, str] = {}
    verify_final_source_delivery = None
    if runtime is not None:
        from kernel_request_source_window import (
            partial_source_paths_for_deferred_delivery,
            source_delivery_satisfied,
            upsert_deferred_source_notice, verify_final_source_delivery,
        )

        source_plan = request_source_plan
        final_messages = body.get("messages")
        if isinstance(source_plan, Mapping) and isinstance(final_messages, list):
            # Deferred paths belong to a serialized request, not the reusable
            # source plan. Re-qualify the current pages without carrying an old
            # notice into the next request, then rebuild a fresh exact notice
            # from the paths that are still partial on this wire.
            verification_plan = _request_scoped_source_plan(
                source_plan, deferred_source_paths,
            )
            from kernel_public_localization import public_path
            source_contents = {
                str(path): str(row.get("content"))
                for path, row in getattr(runtime, "_empirical_frame_sources", {}).items()
                if isinstance(row, Mapping) and isinstance(row.get("content"), str)
                and public_path(path) == str(path)
            }
            if _frame_only_runtime(runtime):
                from kernel_frame_source_snapshot import frame_source_snapshot
                source_contents = frame_source_snapshot(runtime, verification_plan)
            # Projection happens only after source-first fitting and the last
            # diet pass.  The host closure/manifest stays full-fidelity while
            # the verifier below validates the exact bytes actually submitted.
            projected_count = _project_or_assemble_request_messages(
                final_messages,
                verification_plan,
                runtime=runtime,
                capture=capture,
                allocation_plan=owner_first_plan,
                sources=source_contents,
            )
            blank_paths = _settle_blank_source_paths(
                verification_plan, final_messages, deferred_source_paths,
                runtime=runtime,
            )
            new_blank_paths = tuple(
                path for path in blank_paths if path not in deferred_source_paths
            )
            if new_blank_paths:
                deferred_source_paths = blank_paths
                if isinstance(source_plan, dict):
                    source_plan["deferred_paths"] = list(deferred_source_paths)
                _run_diet(max_request_bytes=selected_budget.max_request_bytes)
                deferred_source_paths = tuple(dict.fromkeys(
                    (*blank_paths, *deferred_source_paths),
                ))
                setattr(runtime, "_request_deferred_source_paths", deferred_source_paths)
                if isinstance(source_plan, dict):
                    source_plan["deferred_paths"] = list(deferred_source_paths)
                verification_plan = _request_scoped_source_plan(
                    source_plan, deferred_source_paths,
                )
                final_messages = body.get("messages")
                if isinstance(final_messages, list):
                    projected_count = _project_or_assemble_request_messages(
                        final_messages, verification_plan,
                        runtime=runtime,
                        capture=capture,
                        allocation_plan=owner_first_plan,
                        sources=source_contents,
                    )
                final_tools = body.get("tools", [])
            if capture is not None:
                setattr(capture, "request_source_model_projection_count", projected_count)
            verify_kwargs = {
                "payload": {"messages": final_messages, "tools": final_tools},
            }
            if source_contents:
                verify_kwargs["sources"] = source_contents
            deferred_source_paths = _settle_blank_source_paths(
                verification_plan, final_messages, deferred_source_paths,
                runtime=runtime,
            )
            if isinstance(source_plan, dict):
                source_plan["deferred_paths"] = list(deferred_source_paths)
            source_delivery = verify_final_source_delivery(
                verification_plan, final_messages, **verify_kwargs)
            if isinstance(source_plan, dict):
                source_plan.clear()
                source_plan.update(verification_plan)
            if isinstance(source_plan, dict):
                partial_paths = partial_source_paths_for_deferred_delivery(
                    verification_plan, source_delivery,
                )
                failed_paths = _failed_source_delivery_paths(source_delivery)
                candidate_plan = copy.deepcopy(verification_plan)
                omission_paths = tuple(dict.fromkeys((
                    *_materialize_bounded_omission_units(candidate_plan),
                    *_bounded_delivery_omitted_unit_paths(candidate_plan),
                )))
                read_available = any(
                    isinstance(tool, Mapping)
                    and isinstance(tool.get("function"), Mapping)
                    and tool["function"].get("name") == "read"
                    for tool in final_tools
                )
                if read_available:
                    from kernel_completion_owner import _current_tool_admission
                    read_available = _current_tool_admission(runtime, "read")[0]
                capacity_paths = ()
                if optional_inputs is not None and read_available and (
                    _wire_bytes(body) > selected_budget.max_request_bytes
                    or token_meter.measure(body)[0] > ENDPOINT_PROMPT_TOKEN_BUDGET
                ):
                    capacity_paths = tuple(dict.fromkeys(
                        str(unit["path"]) for unit in candidate_plan.get("units", ())
                        if isinstance(unit, Mapping) and unit.get("path")
                    ))
                if read_available and (partial_paths or omission_paths or failed_paths or capacity_paths):
                    deferred_paths = tuple(dict.fromkeys((
                        *deferred_source_paths, *partial_paths, *omission_paths, *failed_paths,
                        *capacity_paths[:1],
                    )))
                    candidate_plan["deferred_paths"] = list(deferred_paths)
                    try:
                        from kernel_request_diet import RequestDietDeferred, apply_diet

                        candidate_limit = len(candidate_plan.get("units", ())) + 2
                        seen_candidates: set[tuple[str, ...]] = set()
                        for _candidate_pass in range(candidate_limit):
                            signature = tuple(deferred_paths)
                            if signature in seen_candidates:
                                break
                            seen_candidates.add(signature)
                            # Rebuild from the same bounded request each time; a
                            # verified partial receipt can reveal paths eligible
                            # for deferral only after the first missing ranges are
                            # removed from the wire.
                            candidate_body = copy.deepcopy(body)
                            candidate_messages = candidate_body.get("messages")
                            if not isinstance(candidate_messages, list):
                                raise TypeError("source deferral candidate requires messages")
                            _remove_deferred_source_excerpts(
                                candidate_messages, deferred_paths,
                                preserve_stateframe=_contains_framefit5_stateframe(candidate_messages),
                                protected_messages=getattr(capture, "_framefit5_newest_user_suffix", ()),
                            )
                            upsert_deferred_source_notice(candidate_messages, deferred_paths)
                            before_blank_settlement = deferred_paths
                            deferred_paths = _settle_blank_source_paths(
                                candidate_plan, candidate_messages, deferred_paths,
                            )
                            if deferred_paths != before_blank_settlement:
                                _remove_deferred_source_excerpts(
                                    candidate_messages, deferred_paths,
                                    preserve_stateframe=_contains_framefit5_stateframe(candidate_messages),
                                    protected_messages=getattr(capture, "_framefit5_newest_user_suffix", ()),
                                )
                                upsert_deferred_source_notice(
                                    candidate_messages, deferred_paths,
                                )
                            candidate_pretrimmed = _graceful_trim_optional_messages(
                                candidate_body, protected_contents=protected_contents,
                                protected_messages=protected_messages,
                                required_source_paths=required_source_paths,
                                max_request_bytes=selected_budget.max_request_bytes,
                                max_request_tokens=ENDPOINT_PROMPT_TOKEN_BUDGET,
                                measure_request_tokens=lambda: token_meter.measure(candidate_body)[0],
                                token_meter=token_meter,
                            )
                            try:
                                candidate_audit = apply_diet(
                                    candidate_body,
                                    stage=diet_stage,
                                    max_request_bytes=selected_budget.max_request_bytes,
                                    protected_contents=protected_contents,
                                    protected_messages=protected_messages,
                                    required_source_paths=required_source_paths,
                                    required_source_order=required_source_order,
                                    deferred_source_paths=deferred_paths,
                                    token_budget=ENDPOINT_PROMPT_TOKEN_BUDGET,
                                    token_meter=token_meter,
                                    allocation_plan=(
                                        copy.deepcopy(owner_first_plan)
                                        if isinstance(owner_first_plan, dict) else None
                                    ),
                                )
                            except RequestDietDeferred as candidate_error:
                                candidate_audit = dict(candidate_error.detail)
                                candidate_audit["candidate_diet_deferred"] = True
                            candidate_audit["dropped_sections"] = (
                                list(candidate_pretrimmed)
                                + list(candidate_audit.get("dropped_sections", ()))
                            )
                            candidate_messages = candidate_body.get("messages")
                            candidate_tools = candidate_body.get("tools", [])
                            if not isinstance(candidate_messages, list):
                                raise TypeError("source deferral candidate requires messages")
                            candidate_projection_count = _project_or_assemble_request_messages(
                                candidate_messages,
                                candidate_plan,
                                runtime=runtime,
                                capture=capture,
                                allocation_plan=(
                                    copy.deepcopy(owner_first_plan)
                                    if isinstance(owner_first_plan, dict) else None
                                ),
                                sources=source_contents,
                            )
                            candidate_tokens, _candidate_measure_method = token_meter.measure(
                                candidate_body,
                            )
                            candidate_over_budget = (
                                _wire_bytes(candidate_body) > selected_budget.max_request_bytes
                                or candidate_tokens > ENDPOINT_PROMPT_TOKEN_BUDGET
                            )
                            if candidate_over_budget:
                                candidate_trimmed = _graceful_trim_optional_messages(
                                    candidate_body,
                                    protected_contents=protected_contents,
                                    protected_messages=protected_messages,
                                    required_source_paths=required_source_paths,
                                    max_request_bytes=selected_budget.max_request_bytes,
                                    max_request_tokens=ENDPOINT_PROMPT_TOKEN_BUDGET,
                                    measure_request_tokens=lambda: token_meter.measure(
                                        candidate_body,
                                    )[0],
                                    token_meter=token_meter,
                                )
                                if candidate_trimmed:
                                    candidate_audit = apply_diet(
                                        candidate_body,
                                        stage=diet_stage,
                                        max_request_bytes=selected_budget.max_request_bytes,
                                        protected_contents=protected_contents,
                                        protected_messages=protected_messages,
                                        required_source_paths=required_source_paths,
                                        required_source_order=required_source_order,
                                        deferred_source_paths=deferred_paths,
                                        token_budget=ENDPOINT_PROMPT_TOKEN_BUDGET,
                                        token_meter=token_meter,
                                        allocation_plan=(
                                            copy.deepcopy(owner_first_plan)
                                            if isinstance(owner_first_plan, dict) else None
                                        ),
                                    )
                                    candidate_messages = candidate_body.get("messages")
                                    if not isinstance(candidate_messages, list):
                                        raise TypeError(
                                            "trimmed source candidate requires messages",
                                        )
                                    candidate_projection_count = (
                                        _project_or_assemble_request_messages(
                                            candidate_messages,
                                            candidate_plan,
                                            runtime=runtime,
                                            capture=capture,
                                            allocation_plan=(
                                                copy.deepcopy(owner_first_plan)
                                                if isinstance(owner_first_plan, dict) else None
                                            ),
                                            sources=source_contents,
                                        )
                                    )
                            deferred_paths = _settle_blank_source_paths(
                                candidate_plan, candidate_messages, deferred_paths,
                            )
                            candidate_receipt = verify_final_source_delivery(
                                candidate_plan,
                                candidate_messages,
                                payload={
                                    "messages": candidate_messages,
                                    "tools": candidate_tools,
                                },
                                sources=source_contents or None,
                            )
                            candidate_tokens, _candidate_measure_method = token_meter.measure(
                                candidate_body,
                            )
                            candidate_satisfied = (
                                source_delivery_satisfied(candidate_receipt)
                                and _wire_bytes(candidate_body) <= selected_budget.max_request_bytes
                                and candidate_tokens <= ENDPOINT_PROMPT_TOKEN_BUDGET
                            )
                            if candidate_satisfied:
                                break
                            newly_deferable = tuple(
                                path for path in partial_source_paths_for_deferred_delivery(
                                    candidate_plan, candidate_receipt,
                                )
                                if path not in deferred_paths
                            )
                            if not newly_deferable:
                                newly_deferable = tuple(
                                    path for path in capacity_paths if path not in deferred_paths
                                )[:1]
                            if not newly_deferable:
                                break
                            deferred_paths = tuple(dict.fromkeys((
                                *deferred_paths, *newly_deferable,
                            )))
                            candidate_plan["deferred_paths"] = list(deferred_paths)

                        if _record_source_delivery_candidate(
                            emit,
                            request_id,
                            step,
                            deferred_paths,
                            candidate_plan,
                            candidate_receipt,
                            candidate_satisfied,
                        ):
                            body.clear()
                            body.update(candidate_body)
                            source_plan.clear()
                            source_plan.update(candidate_plan)
                            deferred_source_paths = deferred_paths
                            setattr(runtime, "_request_deferred_source_paths", deferred_paths)
                            _remember_dropped_ranges_notice(runtime, candidate_audit)
                            source_delivery = candidate_receipt
                            final_messages = candidate_messages
                            final_tools = candidate_tools
                            if capture is not None:
                                setattr(capture, "request_budget_exceeded", False)
                                setattr(capture, "request_budget_contract_status", "VALID")
                                setattr(
                                    capture,
                                    "request_source_model_projection_count",
                                    candidate_projection_count,
                                )
                            boundary["hard_limit"] = {
                                "blocked": False,
                                "status": "READY",
                            }
                            if capture is not None:
                                setattr(capture, "request_budget_boundary_receipt", boundary)
                    except (ImportError, OSError, RequestDietDeferred, TypeError, ValueError) as error:
                        # Keep the exact partial receipt. The downstream wire gate
                        # will hold it unless its deferred notice is fully proven.
                        # Preserve only a bounded failure class/reason so an
                        # unavailable deferral is diagnosable without journaling
                        # source excerpts or provider payloads.
                        detail = getattr(error, "detail", None)
                        reason = detail.get("reason") if isinstance(detail, Mapping) else None
                        if emit is not None:
                            emit(
                                "request_source_delivery_candidate_failed",
                                producer="kernel_request_budget",
                                request_id=request_id,
                                step=step,
                                failure_class=type(error).__name__,
                                reason_code=(reason[:80] if isinstance(reason, str) else None),
                                candidate_path_count=len(deferred_paths),
                                dispatch=False,
                            )
    allocation_host_limit: dict[str, object] | None = None
    measured_prompt_tokens: int | None = None
    measure_method: str | None = None
    if owner_first_plan is not None:
        from kernel_owner_first import finalize_range_plan, host_limit_outcome
        from kernel_request_diet import RequestDietDeferred, apply_diet

        final_messages = body.get("messages")
        if isinstance(final_messages, list):
            finalize_range_plan(owner_first_plan, final_messages)
        for attempt in range(2):
            measured_prompt_tokens, measure_method = token_meter.measure(body)
            over_bytes = _wire_bytes(body) > selected_budget.max_request_bytes
            over_tokens = measured_prompt_tokens > ENDPOINT_PROMPT_TOKEN_BUDGET
            if not over_bytes and not over_tokens:
                break
            if attempt == 0:
                try:
                    _run_diet(max_request_bytes=selected_budget.max_request_bytes)
                except RequestDietDeferred:
                    allocation_host_limit = host_limit_outcome(
                        owner_first_plan, request_bytes=_wire_bytes(body),
                        max_request_bytes=selected_budget.max_request_bytes,
                        prompt_tokens=measured_prompt_tokens,
                        max_prompt_tokens=ENDPOINT_PROMPT_TOKEN_BUDGET,
                    )
                    break
                final_messages = body.get("messages")
                if (runtime is not None and source_plan is not None
                        and isinstance(final_messages, list)):
                    _project_or_assemble_request_messages(
                        final_messages,
                        source_plan,
                        runtime=runtime,
                        capture=capture,
                        allocation_plan=owner_first_plan,
                        sources=source_contents,
                    )
                continue
            allocation_host_limit = host_limit_outcome(
                owner_first_plan, request_bytes=_wire_bytes(body),
                max_request_bytes=selected_budget.max_request_bytes,
                prompt_tokens=measured_prompt_tokens,
                max_prompt_tokens=ENDPOINT_PROMPT_TOKEN_BUDGET,
            )
        if allocation_host_limit is not None:
            allocation_host_limit["measure_method"] = measure_method
            if capture is not None:
                setattr(capture, "request_budget_exceeded", True)
                setattr(capture, "request_budget_contract_status", HOST_CONTRACT_LIMIT)
                setattr(capture, "request_budget_host_limit", allocation_host_limit)
        final_messages = body.get("messages")
        if (runtime is not None and source_plan is not None
                and isinstance(final_messages, list)
                and verify_final_source_delivery is not None):
            verify_kwargs = {"payload": {"messages": final_messages, "tools": body.get("tools", [])}}
            if source_contents:
                verify_kwargs["sources"] = source_contents
            final_plan = _request_scoped_source_plan(
                source_plan, deferred_source_paths,
            )
            deferred_source_paths = _settle_blank_source_paths(
                final_plan, final_messages, deferred_source_paths, runtime=runtime,
            )
            if isinstance(source_plan, dict):
                source_plan["deferred_paths"] = list(deferred_source_paths)
            source_delivery = verify_final_source_delivery(
                final_plan, final_messages, **verify_kwargs)
    final_wire = {
        "schema": "stateframe-request-wire/1",
        "request_bytes": _wire_bytes(body),
        "max_request_bytes": selected_budget.max_request_bytes,
        "schema_bytes": _schema_bytes(final_tools),
        "tool_names": tuple(_tool_name(tool) for tool in final_tools),
        "tool_schema_digest": "sha256:" + hashlib.sha256(json.dumps(
            final_tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "messages": isinstance(body.get("messages"), list),
        "tools": isinstance(body.get("tools"), list),
        "boundary": boundary,
        "reservation_order": list(RESERVATION_ORDER),
        "reserved_tool_names": list(reserved_tool_names),
        "owner_reexpanded": list(
            getattr(capture, "request_budget_owner_reexpanded", ())
        ),
    }
    if owner_first_plan is not None:
        from kernel_owner_first import allocation_summary, finalize_range_plan
        final_messages = body.get("messages")
        if isinstance(final_messages, list):
            final_wire["owner_first_allocation"] = finalize_range_plan(
                owner_first_plan, final_messages)
        else:
            final_wire["owner_first_allocation"] = allocation_summary(owner_first_plan)
        final_wire["measured_prompt_tokens"] = measured_prompt_tokens
        final_wire["measure_method"] = measure_method
        if allocation_host_limit is not None:
            final_wire["host_limit"] = allocation_host_limit
    if source_delivery is not None:
        final_wire["source_delivery"] = source_delivery
        final_wire["context_digest"] = source_delivery.get("context_digest", "UNKNOWN")
        final_wire["submitted_http_bytes_digest"] = source_delivery.get(
            "submitted_http_bytes_digest", "UNVERIFIED"
        )
    final_wire["files_in_frame"] = list(listed_paths)
    if runtime is not None:
        # Reconcile the immutable gate vector and exact dispatch schemas after
        # the last diet/tool pass.  Earlier menu admission is not evidence that
        # this serialized surface still carries the same recovery route.
        try:
            vector_factory = getattr(runtime, "pass_gates_vector", None)
            if callable(vector_factory):
                from kernel_completion_owner import dispatch_tool_call
                from kernel_pass_gates_contract import validate_final_surface
                vector = vector_factory(final_tools)

                def dispatchable(name: str, arguments: Mapping[str, object]) -> bool:
                    try:
                        dispatch_tool_call(final_tools, name, arguments)
                        return True
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        return False

                final_wire["pass_gates"] = validate_final_surface(
                    final_tools, vector, dispatchable=dispatchable)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
            # Host-only receipt (journal), never solver-visible: a swallowed TypeError here made
            # every wire HOST_BLOCKED and done uncertifiable for 40 requests (batch 836) with no
            # way to see why.  Carry the message and the innermost frames.
            import traceback as _traceback
            frames = _traceback.extract_tb(error.__traceback__)[-4:]
            final_wire["pass_gates"] = {
                "status": "HOST_UNSUPPORTED", "reason": "final gate surface unavailable",
                "error_type": type(error).__name__, "error": str(error)[:400],
                "error_frames": [f"{os.path.basename(frame.filename)}:{frame.lineno}:{frame.name}"
                                 for frame in frames],
            }
    final_wire["status"] = (
        "READY" if final_wire["messages"] and final_wire["tools"]
        else "HOST_BLOCKED")
    if (isinstance(final_wire.get("source_delivery"), Mapping)
            and not source_delivery_satisfied(final_wire["source_delivery"])):
        final_wire["status"] = "HOST_BLOCKED"
    if isinstance(final_wire.get("pass_gates"), Mapping) and final_wire["pass_gates"].get(
            "status") != "VALID":
        final_wire["status"] = "HOST_BLOCKED"
    if allocation_host_limit is not None:
        final_wire["status"] = HOST_CONTRACT_LIMIT
    if optional_inputs is not None:
        measured_prompt_tokens, measure_method = token_meter.measure(body)
        final_wire["measured_prompt_tokens"] = measured_prompt_tokens
        final_wire["measure_method"] = measure_method
        if (_wire_bytes(body) > selected_budget.max_request_bytes
                or measured_prompt_tokens > ENDPOINT_PROMPT_TOKEN_BUDGET):
            final_wire["status"] = HOST_CONTRACT_LIMIT
            if capture is not None:
                setattr(capture, "request_budget_exceeded", True)
        if final_wire["status"] == "READY" and capture is not None:
            from kernel_request_seal import FrameSelectionProposal, freeze
            from kernel_frame_optional_allocation import selection_from_frame
            from kernel_stateframe_framefit5 import _state_payload
            selected_frame = next(_state_payload(message)[1]["framefit5"]
                                  for message in body["messages"]
                                  if _is_stateframe_message(message))
            selected_optional, selected_inventory = selection_from_frame(
                *optional_inputs, selected_frame)
            # A budget proposal owns selection inputs, never provider authority.
            capture._frame_request_allocation_candidate = FrameSelectionProposal(
                freeze(selected_optional), freeze(selected_inventory),
                freeze({"byte_ceiling": selected_budget.max_request_bytes,
                        "token_ceiling": ENDPOINT_PROMPT_TOKEN_BUDGET}),
            )
            setattr(capture, "request_budget_exceeded", False)
            boundary["hard_limit"] = {"blocked": False, "status": "READY"}
    if capture is not None:
        setattr(capture, "request_budget_final_wire", dict(final_wire))
        published_delivery = source_delivery
        if frame_candidate and final_wire["status"] != "READY":
            # The failed draft remains only in the budget receipt. Its source
            # proof cannot certify the untouched body returned to the caller.
            published_delivery = verify_final_source_delivery(
                source_plan, destination_body["messages"],
                payload=destination_body, sources=source_contents,
            )
        setattr(capture, "request_source_final_delivery", published_delivery)
    if owner_first_plan is not None and emit is not None:
        from kernel_owner_first import ALLOC_ENV, allocation_summary
        emit(
            "organ_wiring_state", producer="kernel_owner_first", organ=ALLOC_ENV,
            state="on", boundary="request", request_id=request_id, step=step,
            allocation_plan=allocation_summary(owner_first_plan),
            host_limit=allocation_host_limit, dispatch=False,
        )
    _emit_budget_receipt(
        emit,
        kind="request_budget_final_wire",
        stage=stage,
        budget=selected_budget,
        action="final_wire",
        reason=str(final_wire["status"]),
        request_bytes_before=_wire_bytes(body),
        request_bytes_after=int(final_wire["request_bytes"]),
        before_tool_count=len(final_tools),
        after_tool_count=len(final_tools),
        before_schema_bytes=int(final_wire["schema_bytes"]),
        after_schema_bytes=int(final_wire["schema_bytes"]),
        dropped_tools=tuple(boundary["dropped_tools"]),
        dropped_sections=(),
        preserved_tools=tuple(final_wire["tool_names"]),
        request_id=request_id,
        step=step,
        # final_wire carries "schema" (journal-RESERVED): rename to receipt_schema.
        **__import__("kernel_continuation_journal").journal_safe_fields(final_wire),
    )
    _emit_budget_receipt(
        emit,
        kind="request_budget_boundary",
        stage=stage,
        budget=selected_budget,
        action="boundary_reconciled",
        reason=str(boundary["status"]),
        request_bytes_before=_wire_bytes({"tools": original_tools}),
        request_bytes_after=_wire_bytes({"tools": final_tools}),
        before_tool_count=len(original_tools) if isinstance(original_tools, list) else 0,
        after_tool_count=len(final_tools),
        before_schema_bytes=_schema_bytes(original_tools) if isinstance(original_tools, list) else 0,
        after_schema_bytes=_schema_bytes(final_tools),
        dropped_tools=tuple(boundary["dropped_tools"]),
        dropped_sections=(),
        preserved_tools=tuple(boundary["final_tools"]),
        request_id=request_id,
        step=step,
        boundary_receipt=boundary,
    )
    if (capture is not None and (frame_candidate or not request_source_plan_override)
            and isinstance(request_source_plan, dict)
            and (not frame_candidate or final_wire["status"] == "READY")
            and (optional_inputs is not None or request_source_plan_before != request_source_plan)):
        setattr(capture, "_request_source_closure_override", copy.deepcopy(request_source_plan))
    if frame_candidate:
        if final_wire["status"] == "READY":
            destination_body.clear()
            destination_body.update(body)
        return destination_body
    return body


def _runtime_context(
        runtime: object, capture: object, tools: Sequence[object],
        messages: Sequence[object] | None = None,
) -> tuple[str, set[str], set[str]]:
    available_tool_names = {
        _tool_name(tool) for tool in tools if _tool_name(tool)
    }
    owner = getattr(runtime, "ideal_path_owner", None)
    if owner is None:
        resolver = getattr(runtime, "_idealpath_owner", None)
        if callable(resolver):
            try:
                owner = resolver()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                owner = None
    # A capture can retain the last staged value after IDEALPATH is detached.
    # It is not an authority for a new request: ownerless work is a native
    # free loop.  Continue through completion dispatch below so the returned
    # surface is one contract, rather than an early, incomplete projection.
    stage = (
        (getattr(owner, "stage", None) or getattr(capture, "request_budget_stage", None) or "NATIVE")
        if owner is not None else FREE_STAGE
    )
    exits = set(DEFAULT_STAGE_EXIT_ACTIONS.get(str(stage), ()))
    configured_exits = getattr(capture, "request_budget_exits", ())
    if owner is not None and isinstance(configured_exits, Sequence) \
            and not isinstance(configured_exits, (str, bytes, bytearray)):
        exits.update(str(name) for name in configured_exits if str(name))
    if owner is not None:
        try:
            live = owner.stage_exit_liveness()
            exits.update(str(name) for name in live.get("satisfying_action_ids", ()))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    else:
        # Owner-free IDEALPATH-off requests are native work, not S1.  Protect
        # one currently available route for each irreducible contract: editor,
        # source acquisition, current check/witness, and terminal recovery.
        names = {_tool_name(tool) for tool in tools}
        for choices in (("edit", "str_replace", "file_create", "apply_patch", "write_file", "patch"),
                        ("read", "grep", "list_files", "references", "where_defined", "who_calls"),
                        ("authored_check", "typed_test_writer", "run_tests", "probe", "check", "execute"),
                        ("done", "submit", "complete", "question_gate", "recover")):
            selected = next((name for name in choices if name in names), None)
            if selected:
                exits.add(selected)
        # There is no owner in this branch.  In particular, do not map the
        # native loop to S1 or call an owner method that cannot exist.
    # A requirement absent from the current provider schemas is not a dropped
    # route and must not be fabricated into this request's protected surface.
    exits.intersection_update(available_tool_names)
    completion: set[str] = set()
    protected_surface = False
    dispatch = getattr(runtime, "_completion_owner_dispatch_tools", None)
    # The completion owner may remain authoritative for the current FREE
    # dispatch surface even when IDEALPATH has no stage owner.  Consume its
    # frozen route instead of silently dropping recovery in free mode.
    if isinstance(dispatch, list):
        dispatch_names = {
            _tool_name(tool) for tool in dispatch
            if _tool_name(tool) and _tool_name(tool) in available_tool_names
        }
        if dispatch_names:
            completion.update(dispatch_names)
            protected_surface = len(dispatch_names) > DEFAULT_MAX_TOOLS
    current_menu = getattr(runtime, "completion_menu", ())
    if isinstance(current_menu, Sequence) and not isinstance(
            current_menu, (str, bytes, bytearray)):
        completion.update(
            str(name).strip() for name in current_menu
            if str(name).strip() in available_tool_names
        )
    owner_state = getattr(runtime, "_completion_owner_state", None)
    terminal_owner_active = (
        isinstance(owner_state, Mapping)
        and any(bool(owner_state.get(key)) for key in (
            "terminal_choice_offered", "final_submit_reserved", "final_submit_request_pending",
        ))
    )
    auto_verified_terminal = str(os.environ.get("STATEFRAME_AUTO_VERIFY", "off")).strip().lower() == "on"
    if terminal_owner_active or auto_verified_terminal:
        # The owner has already selected a terminal/recovery contract.  The
        # budget may validate it, but cannot reorder or replace its actions.
        # Auto-verify is the host's terminal-only completion rail; it is set
        # before the owner state is promoted in the synthetic release smoke.
        completion.update(_tool_name(tool) for tool in tools if _tool_name(tool))
        protected_surface = True
    terminal_offer = any(
        isinstance(message, Mapping)
        and "Final step: call done now to submit this revision" in str(message.get("content", ""))
        for message in messages or ()
    )
    if terminal_offer and any(_tool_name(tool) == "done" for tool in tools):
        completion.add("done")
        completion.add(_COMPLETION_ONLY_MARKER)
    contract = getattr(capture, "request_action_contract", None)
    if isinstance(contract, Mapping) and str(contract.get("route", "")) in {"submit", "host-stop", "terminal"}:
        completion.update(name for name in ("done", "submit", "complete")
                          if any(_tool_name(tool) == name for tool in tools))
    if isinstance(owner_state, Mapping) and owner_state.get("final_submit_request_pending"):
        if any(_tool_name(tool) == "done" for tool in tools):
            completion.add("done")
    completion.intersection_update(available_tool_names)
    protected_surface = protected_surface or len(completion) > DEFAULT_MAX_TOOLS
    return str(stage), exits, completion | ({_PROTECTED_SURFACE_MARKER} if protected_surface else set())


def _presentation_spec(runtime: object, capture: object) -> dict[str, object]:
    declared = getattr(runtime, "declared_environment", {})
    environment = dict(declared) if isinstance(declared, Mapping) else {}
    arm = getattr(capture, "presentation", None)
    if arm not in {"history", "frame"}:
        arm = "frame" if getattr(capture, "stateframe_f4_mode", "off") == "on" else "history"
    prefill = getattr(capture, "stateframe_think_prefill_mode", None)
    if prefill in {"off", "on", "user", "self_plan"}:
        environment.setdefault("STATEFRAME_THINK_PREFILL", prefill)
    return {"presentation_arm": arm, "environment": environment}


@profiled_function_span("request_budget")
def apply_runtime_budget(runtime: object, capture: object,
                         prepared: list[dict[str, object]],
                         tools: list[dict[str, object]], *,
                         tokenizer: Callable[[str], object] | None = None,
                         tokenizer_endpoint: str | None = None,
                         ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Apply the budget before completion-owner contract finalization."""
    from kernel_owner_first import allocation_enabled
    if not _runtime_enabled(runtime) and not allocation_enabled():
        return prepared, tools
    # Captures are reused by some transport paths.  Ownerless FREE requests
    # must not inherit a prior staged hard-block bit.
    setattr(capture, "request_budget_exceeded", False)
    stage, exits, completion = _runtime_context(runtime, capture, tools, prepared)
    protected_surface = _PROTECTED_SURFACE_MARKER in completion
    completion_only = _COMPLETION_ONLY_MARKER in completion
    completion.difference_update({_PROTECTED_SURFACE_MARKER, _COMPLETION_ONLY_MARKER})
    journal = getattr(capture, "journal", None)
    emit = getattr(journal, "emit", None)
    body: dict[str, object] = {"messages": prepared, "tools": tools}
    protected_contents: list[str] = []
    owner_blocks = getattr(runtime, "_completion_owner_owner_blocks", None)
    if isinstance(owner_blocks, Mapping):
        for record in owner_blocks.values():
            if not isinstance(record, Mapping):
                continue
            value = record.get("_value")
            if isinstance(value, str) and value:
                protected_contents.append(value)
    from capability_request_preflight import presentation_protected_messages
    protected_messages = presentation_protected_messages(
        prepared, _presentation_spec(runtime, capture), runtime, tools=tools,
    )
    _apply_budget_with_diet(
        body,
        stage=stage,
        stage_exit_actions=sorted(exits),
        completion_tool_names=sorted(completion),
        protected_completion_surface=protected_surface,
        completion_only=completion_only,
        emit=emit,
        request_id=getattr(capture, "request_id", None),
        step=getattr(capture, "step", None),
        protected_contents=protected_contents,
        protected_messages=protected_messages,
        runtime=runtime,
        capture=capture,
        tokenizer=tokenizer,
        tokenizer_endpoint=tokenizer_endpoint,
    )
    setattr(capture, "request_budget_prepared", True)
    setattr(capture, "request_budget_stage", stage)
    setattr(capture, "request_budget_exits", exits)
    setattr(capture, "request_budget_completion", completion)
    setattr(capture, "request_budget_protected_surface", protected_surface)
    setattr(capture, "request_budget_final_tools", tuple(_tool_name(tool) for tool in body["tools"]))
    final_schema = validate_final_schema(body, tuple(exits) + tuple(completion))
    setattr(capture, "request_budget_final_schema", final_schema)
    setattr(capture, "request_budget_contract_status", (
        HOST_CONTRACT_LIMIT
        if bool(getattr(capture, "request_budget_exceeded", False))
        or not bool(final_schema["valid"])
        else "VALID"
    ))
    return list(body["messages"]), list(body["tools"])


def _owner_first_transport_check(capture: object, runtime: object,
                                 body: Mapping[str, object], stage: str, *,
                                 emit_event: bool) -> dict[str, object] | None:
    """Measure an already-frozen request and return a typed paging outcome if needed."""
    from kernel_owner_first import (
        ALLOC_ENV, allocation_enabled, allocation_summary, build_priority_plan,
        host_limit_outcome,
    )
    if not allocation_enabled():
        return None
    plan = getattr(capture, "request_owner_first_allocation_plan", None)
    if not isinstance(plan, dict):
        plan = build_priority_plan(body, runtime)
        setattr(capture, "request_owner_first_allocation_plan", plan)
        emit_event = True
    meter = _request_token_meter(runtime, capture)
    measured_tokens, method = meter.measure(body)
    request_action_contract = getattr(capture, "request_action_contract", None)
    lean_contract = (
        isinstance(request_action_contract, Mapping)
        and request_action_contract.get("contract_kind") == "lean_stable_surface"
    )
    if lean_contract:
        max_request_bytes, max_prompt_tokens = _lean_request_limits(
            body, stage, exact_tokenizer=method == "tokenize",
        )
    else:
        selected_budget = stage_budget(stage)
        max_request_bytes = min(selected_budget.max_request_bytes, HARD_MAX_REQUEST_BYTES)
        max_prompt_tokens = ENDPOINT_PROMPT_TOKEN_BUDGET
    request_bytes = _wire_bytes(body)
    exceeds = request_bytes > max_request_bytes or measured_tokens > max_prompt_tokens
    outcome = (host_limit_outcome(
        plan, request_bytes=request_bytes,
        max_request_bytes=max_request_bytes,
        prompt_tokens=measured_tokens,
        max_prompt_tokens=max_prompt_tokens,
    ) if exceeds else None)
    if outcome is not None:
        outcome["measure_method"] = method
        setattr(capture, "request_budget_exceeded", True)
        setattr(capture, "request_budget_contract_status", HOST_CONTRACT_LIMIT)
        setattr(capture, "request_budget_host_limit", outcome)
    existing = getattr(capture, "request_budget_final_wire", None)
    final_wire = dict(existing) if isinstance(existing, Mapping) else {
        "schema": "stateframe-request-wire/1", "request_bytes": request_bytes,
    }
    final_wire.update({
        "request_bytes": request_bytes,
        "max_request_bytes": max_request_bytes,
        "max_prompt_tokens": max_prompt_tokens,
        "measured_prompt_tokens": measured_tokens,
        "measure_method": method,
        "owner_first_allocation": allocation_summary(plan),
        "status": HOST_CONTRACT_LIMIT if outcome is not None else final_wire.get("status", "READY"),
    })
    if outcome is not None:
        final_wire["host_limit"] = outcome
    setattr(capture, "request_budget_final_wire", final_wire)
    if emit_event:
        emit = getattr(getattr(capture, "journal", None), "emit", None)
        if callable(emit):
            emit("organ_wiring_state", producer="kernel_owner_first", organ=ALLOC_ENV,
                 state="on", boundary="request",
                 request_id=getattr(capture, "request_id", None),
                 step=getattr(capture, "step", None),
                 allocation_plan=allocation_summary(plan), host_limit=outcome,
                 dispatch=False)
    return outcome


def apply_capture_budget(
        capture: object, body: dict[str, object], *,
        tokenizer: Callable[[str], object] | None = None,
        tokenizer_endpoint: str | None = None,
) -> dict[str, object]:
    """Apply/validate at the transport seam for standalone captures."""
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    if runtime is None:
        runtime = getattr(capture, "presentation_runtime", None)
    if runtime is None:
        runtime = getattr(capture, "completion_owner_runtime", None)
    if runtime is None:
        # Standalone/auxiliary fitting records request-local selection state.
        # It must be mutable without borrowing an active primary owner.
        runtime = SimpleNamespace()
    _record_lean_required_source_paths(capture, runtime)
    from kernel_owner_first import allocation_enabled
    if not _runtime_enabled(runtime) and not allocation_enabled():
        return body
    setattr(capture, "request_budget_exceeded", False)
    request_action_contract = getattr(capture, "request_action_contract", None)
    lean_contract = (
        isinstance(request_action_contract, Mapping)
        and request_action_contract.get("contract_kind") == "lean_stable_surface"
    )
    if (isinstance(getattr(capture, "lean_prepared_history", None), list)
            and not lean_contract):
        journal = getattr(capture, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit("lean_history_append_only_refused", producer="kernel_request_budget",
                 request_id=getattr(capture, "request_id", None),
                 boundary="capture_budget_input", reason="lean_action_contract_missing",
                 status=HOST_CONTRACT_LIMIT, dispatch=False)
        raise LeanHistoryMutationError(
            "LEAN prepared history reached request budget without its action contract")
    if isinstance(request_action_contract, Mapping):
        # The completion owner has already frozen an admitted surface.  A
        # transport fallback must validate its bytes, never replace that
        # contract after the owner boundary.
        if lean_contract:
            _lean_append_only_receipt(capture, body, boundary="capture_budget_input")
            setattr(capture, "request_budget_reservation_order", RESERVATION_ORDER)
        _deliver_authored_context(body, runtime)
        setattr(capture, "request_budget_prepared", True)
        setattr(capture, "request_budget_protected_surface", True)
        owner = getattr(runtime, "ideal_path_owner", None)
        stage = str(getattr(owner, "stage", None) or FREE_STAGE)
        setattr(capture, "request_budget_stage", stage)
        from kernel_aha10_honest_loop import request_boundary_receipt
        final_tools = body.get("tools", [])
        offered = getattr(capture, "request_action_contract", {}).get("final_actions", ())
        setattr(capture, "request_budget_boundary_receipt", request_boundary_receipt(
            final_tools if isinstance(final_tools, Sequence) else (),
            final_tools if isinstance(final_tools, Sequence) else (),
            final_tools if isinstance(final_tools, Sequence) else (),
            offered_actions=offered if isinstance(offered, Sequence) else (),
        ))
        stage = str(getattr(getattr(runtime, "ideal_path_owner", None), "stage", None)
                    or FREE_STAGE)
        allocation_outcome = _owner_first_transport_check(
            capture, runtime, body, stage, emit_event=True)
        if lean_contract:
            measured_prompt_tokens, measure_method = _request_token_meter(
                runtime, capture,
            ).measure(body)
            request_limit, prompt_limit = _lean_request_limits(
                body, stage, exact_tokenizer=measure_method == "tokenize",
            )
            final_wire = dict(getattr(capture, "request_budget_final_wire", {}) or {})
            final_wire.update({
                "max_request_bytes": request_limit,
                "max_prompt_tokens": prompt_limit,
                "measured_prompt_tokens": measured_prompt_tokens,
                "measure_method": measure_method,
            })
            setattr(capture, "request_budget_final_wire", final_wire)
        _validate_wire_ceiling(capture, body)
        if lean_contract:
            _lean_append_only_receipt(capture, body, boundary="capture_budget_output")
            request_bytes = _wire_bytes(body)
            token_capacity_exceeded = measured_prompt_tokens > prompt_limit
            if (allocation_outcome is not None or request_bytes > request_limit
                    or token_capacity_exceeded):
                reason = (
                    "lean_append_only_history_exceeds_owner_capacity"
                    if allocation_outcome is not None
                    else "lean_append_only_history_exceeds_request_capacity"
                    if request_bytes > request_limit
                    else "lean_append_only_history_exceeds_prompt_token_capacity"
                )
                outcome = {
                    "status": HOST_CONTRACT_LIMIT,
                    "reason": reason,
                    "request_bytes": request_bytes,
                    "max_request_bytes": request_limit,
                    "measured_prompt_tokens": measured_prompt_tokens,
                    "max_prompt_tokens": prompt_limit,
                    "measure_method": measure_method,
                    "request_id": getattr(capture, "request_id", None),
                    "stage": stage,
                }
                setattr(capture, "request_budget_exceeded", True)
                setattr(capture, "request_budget_contract_status", HOST_CONTRACT_LIMIT)
                setattr(capture, "request_budget_host_limit", outcome)
                setattr(capture, "lean_capacity_outcome", outcome)
                setattr(capture, "request_budget_final_wire", {
                    "schema": "stateframe-request-wire/1",
                    "request_bytes": request_bytes,
                    "max_request_bytes": request_limit,
                    "measured_prompt_tokens": measured_prompt_tokens,
                    "max_prompt_tokens": prompt_limit,
                    "measure_method": measure_method,
                    "status": HOST_CONTRACT_LIMIT,
                    "host_limit": outcome,
                })
                emit = getattr(getattr(capture, "journal", None), "emit", None)
                if callable(emit):
                    emit("lean_history_capacity_outcome",
                         producer="kernel_request_budget", dispatch=False, **outcome)
                raise LeanHistoryCapacityError(outcome)
        from kernel_final_wire_budget import reconcile_frozen_budget
        reconcile_frozen_budget(capture, runtime, body)
        return body
    if getattr(capture, "request_budget_prepared", False):
        stage = str(getattr(capture, "request_budget_stage", None) or FREE_STAGE)
        _owner_first_transport_check(
            capture, runtime, body, stage,
            emit_event=not isinstance(
                getattr(capture, "request_owner_first_allocation_plan", None), Mapping),
        )
        _validate_wire_ceiling(capture, body)
        return body
    body_tools = body.get("tools", [])
    body_messages = body.get("messages", [])
    stage, exits, completion = _runtime_context(runtime, capture, body_tools,
                                                body_messages)
    protected_surface = _PROTECTED_SURFACE_MARKER in completion
    completion_only = _COMPLETION_ONLY_MARKER in completion
    completion.difference_update({_PROTECTED_SURFACE_MARKER, _COMPLETION_ONLY_MARKER})
    emit = getattr(getattr(capture, "journal", None), "emit", None)
    from capability_request_preflight import presentation_protected_messages
    protected_messages = presentation_protected_messages(
        body_messages if isinstance(body_messages, list) else [],
        _presentation_spec(runtime, capture), runtime,
        tools=body_tools if isinstance(body_tools, list) else [],
    )
    _apply_budget_with_diet(
        body,
        stage=stage,
        stage_exit_actions=sorted(exits),
        completion_tool_names=sorted(completion),
        protected_completion_surface=protected_surface,
        completion_only=completion_only,
        emit=emit,
        request_id=getattr(capture, "request_id", None),
        step=getattr(capture, "step", None),
        protected_messages=protected_messages,
        runtime=runtime,
        capture=capture,
        tokenizer=tokenizer,
        tokenizer_endpoint=tokenizer_endpoint,
    )
    setattr(capture, "request_budget_prepared", True)
    setattr(capture, "request_budget_stage", stage)
    setattr(capture, "request_budget_exits", exits)
    setattr(capture, "request_budget_completion", completion)
    setattr(capture, "request_budget_protected_surface", protected_surface)
    _validate_wire_ceiling(capture, body)
    final_schema = validate_final_schema(body, tuple(exits) + tuple(completion))
    setattr(capture, "request_budget_final_schema", final_schema)
    if not bool(final_schema["valid"]):
        setattr(capture, "request_budget_contract_status", HOST_CONTRACT_LIMIT)
    return body


def _validate_wire_ceiling(capture: object, body: Mapping[str, object]) -> None:
    stage = str(getattr(capture, "request_budget_stage", None) or FREE_STAGE)
    budget = stage_budget(stage)
    tools = body.get("tools", [])
    if not isinstance(tools, list):
        emit = getattr(getattr(capture, "journal", None), "emit", None)
        if emit is not None:
            emit(
                "request_budget_exceeded",
                event_schema=SCHEMA,
                stage=stage,
                action="budget_exceeded",
                reason="wire_tools_not_list",
                request_id=getattr(capture, "request_id", None),
                step=getattr(capture, "step", None),
                producer="kernel_request_budget",
                trim_order=list(TRIM_ORDER),
            )
        setattr(capture, "request_budget_exceeded", True)
        return
    request_bytes = _wire_bytes(body)
    schema = _schema_bytes(tools)
    protected_surface = bool(getattr(capture, "request_budget_protected_surface", False))
    budget_exceeded = bool(getattr(capture, "request_budget_exceeded", False))
    request_limit = HARD_MAX_REQUEST_BYTES if budget_exceeded else budget.max_request_bytes
    if ((len(tools) > budget.max_tools and not protected_surface)
            or (schema > budget.max_schema_bytes and not protected_surface)):
        emit = getattr(getattr(capture, "journal", None), "emit", None)
        setattr(capture, "request_budget_exceeded", True)
        if emit is not None:
            emit(
                "request_budget_exceeded",
                event_schema=SCHEMA,
                stage=stage,
                action="budget_exceeded",
                reason="wire_request_exceeds_hard_ceiling",
                max_tools=budget.max_tools,
                max_schema_bytes=budget.max_schema_bytes,
                max_request_bytes=request_limit,
                request_bytes_before=request_bytes,
                request_bytes_after=request_bytes,
                before_tool_count=len(tools),
                after_tool_count=len(tools),
                before_schema_bytes=schema,
                after_schema_bytes=schema,
                dropped_tools=[],
                dropped_sections=[],
                preserved_tools=sorted(_tool_name(tool) for tool in tools),
                request_id=getattr(capture, "request_id", None),
                step=getattr(capture, "step", None),
                producer="kernel_request_budget",
                trim_order=list(TRIM_ORDER),
                protected_completion_surface=protected_surface,
                budget_exceeded=budget_exceeded,
                hard_max_unachievable=request_bytes > HARD_MAX_REQUEST_BYTES,
            )


__all__ = [
    "BudgetDecision", "DEFAULT_MAX_REQUEST_BYTES", "DEFAULT_MAX_SCHEMA_BYTES",
    "DEFAULT_MAX_TOOLS", "ENDPOINT_CONTEXT_TOKENS", "ENDPOINT_MAX_REQUEST_BYTES",
    "ENDPOINT_OUTPUT_BUDGET_TOKENS", "ENDPOINT_PROMPT_TOKEN_BUDGET",
    "ENDPOINT_REQUEST_BYTES_PER_TOKEN", "ENDPOINT_SAFETY_HEADROOM_TOKENS",
    "ENDPOINT_THINKING_BUDGET_TOKENS", "HARD_MAX_REQUEST_BYTES", "MODE_ENV",
    "LeanHistoryCapacityError", "LeanHistoryMutationError", "RequestBudgetDeferred",
    "SCHEMA", "S3_MAX_REQUEST_BYTES", "STAGE_BUDGETS",
    "StageRequestBudget", "apply_budget", "apply_capture_budget",
    "apply_runtime_budget", "enabled", "final_tool_names", "stage_budget", "TRIM_ORDER",
    "RESERVATION_ORDER",
    "validate_final_schema", "HOST_CONTRACT_LIMIT",
]
