"""Fixed-vocabulary mapping helpers for source-backed operation records."""
from __future__ import annotations

import math


RUNTIMES = frozenset({
    "completed", "hook_blocked", "continuation_blocked", "suspended", "blocked", "cancelled", "budget_stop",
    "recovery_required", "hook_containment", "engine_error", "context_too_small", "incomplete",
    "missing_result", "startup_error", "mutation_containment", "unknown", "interrupted", "wrong_test",
})
CHECKS = frozenset({"pass", "fail", "not_run", "incomplete", "no_progress", "unknown"})
ARTIFACTS = frozenset({"saved", "partial", "none", "unknown"})
DELIVERIES = frozenset({"saved", "applied", "refused", "apply_failed", "none", "unknown"})
RESUMES = frozenset({"initial", "checkpoint", "replay", "restart_live", "fallback", "unknown"})
OUTCOMES = frozenset({
    "ok", "error", "refused", "blocked", "cancelled", "timed_out", "exhausted", "recovery_required",
    "incomplete", "interrupted", "unknown",
})
EXECUTIONS = frozenset({"executed", "replayed", "reexecuted", "not_dispatched", "unknown"})
TOOLS = frozenset({
    "read", "edit", "create", "delete", "move", "shell", "job", "git", "web", "custom", "mcp",
    "mutation", "check", "parallel", "other",
})
CHECK_KINDS = frozenset({"baseline", "inner", "authored", "unit", "final", "other"})
ROLES = frozenset({"read", "plan", "edit", "finish", "decide2", "unit", "unknown"})
HOOK_EVENTS = frozenset({"session_start", "prompt_submit", "pre_tool", "post_tool", "stop", "session_end"})
RETRY_KINDS = frozenset({"capacity", "transport", "semantic", "other"})
ERROR_RESULTS = frozenset({
    "ENGINE_ERROR", "HARNESS_CONTRACT", "FINALIZATION_ERROR", "MUTATION_CONTAINMENT", "HOOK_CONTAINMENT",
    "RECOVERY_REQUIRED", "CONTEXT_TOO_SMALL", "HARNESS_ERROR", "ENVIRONMENT_BROKEN", "RUN_FAILED",
    "AUTHORITY_FAILURE",
})
BUDGET_RESULTS = frozenset({
    "RESPONSE_BUDGET_EXHAUSTED", "ATTEMPT_BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED", "TOOL_CALL_CAP",
    "CONTEXT_OVERFLOW", "DEADLINE_EXCEEDED",
})


def terminal_fields(summary: dict[str, object]) -> tuple[str, str, str, str, str, int | None, bool]:
    """Return runtime/check/artifact/delivery/resume/exit/error using fixed source facts only."""
    terminal = summary.get("engine_terminal")
    terminal = terminal if type(terminal) is dict else {}
    result = terminal.get("result")
    if type(result) is not str:
        result = summary.get("engine_result")
    result = result if type(result) is str else None
    runtime = _runtime(summary.get("runtime_outcome"), result, terminal)
    if result == "TEST_REPORTED_WRONG":
        runtime = "wrong_test"
    check = _fixed(summary.get("check_outcome"), CHECKS, "unknown")
    artifact = _fixed(summary.get("artifact"), ARTIFACTS, "unknown")
    delivery = _delivery(summary, result, artifact)
    resume = _resume(summary.get("resume"))
    exit_code = summary.get("exit_code")
    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        exit_code = summary.get("child_exit_code")
    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        exit_code = None
    error = (_is_error(runtime, check, delivery, result, terminal)
             or summary.get("uncaught_exception") is True
             or terminal.get("uncaught_exception") is True)
    return runtime, check, artifact, delivery, resume, exit_code, error


def outcome_for(runtime: str, check: str, delivery: str, result: str | None) -> str:
    if result in ERROR_RESULTS or (result is not None and result.startswith("ERR:")):
        return "error"
    if delivery == "refused" or result == "REFUSED":
        return "refused"
    if delivery == "apply_failed" or check == "fail" or runtime in {
        "engine_error", "hook_containment", "mutation_containment", "recovery_required",
    }:
        return "error"
    if runtime in {"suspended", "blocked", "hook_blocked", "continuation_blocked"}:
        return "blocked"
    if runtime == "cancelled":
        return "cancelled"
    if runtime == "budget_stop":
        return "exhausted"
    if runtime == "interrupted":
        return "interrupted"
    if runtime in {"incomplete", "wrong_test"}:
        return "incomplete"
    if runtime == "completed" and check in {"pass", "not_run", "unknown"} and delivery not in {
        "refused", "apply_failed",
    }:
        return "ok"
    return "unknown"


def fixed_tool(value: object) -> str:
    if type(value) is str and value in TOOLS:
        return value
    return "other"


def fixed_check_kind(value: object) -> str:
    return value if type(value) is str and value in CHECK_KINDS else "other"


def fixed_outcome(value: object) -> str:
    if type(value) is str and value in OUTCOMES:
        return value
    aliases = {
        "COMMITTED": "ok", "COMMITTED_UNVERIFIED": "ok", "SUBMITTED": "ok", "PASS": "ok", "DONE": "ok",
        "REFUSED": "refused", "BLOCKED": "blocked", "CANCELLED": "cancelled", "CANCELED": "cancelled",
        "TIMED_OUT": "timed_out", "BUDGET_EXHAUSTED": "exhausted", "RECOVERY_REQUIRED": "recovery_required",
        "ENGINE_ERROR": "error", "HARNESS_CONTRACT": "error", "FINALIZATION_ERROR": "error",
        "MUTATION_CONTAINMENT": "error", "HOOK_CONTAINMENT": "error",
        "CONTEXT_TOO_SMALL": "error", "HARNESS_ERROR": "error", "ENVIRONMENT_BROKEN": "error",
        "RUN_FAILED": "error", "AUTHORITY_FAILURE": "error", "ERROR": "error",
        "ATTEMPT_BUDGET_EXHAUSTED": "exhausted",
        "RESPONSE_BUDGET_EXHAUSTED": "exhausted", "TOOL_CALL_CAP": "exhausted",
        "INDETERMINATE": "unknown",
    }
    return aliases.get(value, "unknown") if type(value) is str else "unknown"


def finite_duration(value: object) -> float | None:
    if type(value) is int:
        return float(value) if 0 <= value <= 604800 else None
    if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 604800:
        return None
    return float(value)


def epoch_seconds_ns(value: object) -> int | None:
    if type(value) is int:
        if not 0 < value <= ((1 << 63) - 1) // 1_000_000_000:
            return None
        return value * 1_000_000_000
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        return None
    scaled = value * 1_000_000_000
    if not math.isfinite(scaled) or scaled > (1 << 63) - 1:
        return None
    ns = int(scaled)
    return ns if 1 <= ns <= (1 << 63) - 1 else None


def _runtime(value: object, result: str | None, terminal: dict[str, object]) -> str:
    if result == "BLOCKED":
        return "suspended" if terminal.get("resumable") is True else "blocked"
    if type(value) is str and value in RUNTIMES:
        return value
    if result in {"HOOK_BLOCKED", "CONTINUATION_BLOCKED"}:
        return "hook_blocked" if result == "HOOK_BLOCKED" else "continuation_blocked"
    if result in ERROR_RESULTS or (result is not None and result.startswith("ERR:")):
        return {
            "MUTATION_CONTAINMENT": "mutation_containment",
            "HOOK_CONTAINMENT": "hook_containment",
            "RECOVERY_REQUIRED": "recovery_required",
            "CONTEXT_TOO_SMALL": "context_too_small",
        }.get(result, "engine_error")
    if result in {"CANCELLED", "CANCELED"}:
        return "cancelled"
    if result in BUDGET_RESULTS:
        return "budget_stop"
    if result in {"TOOL_CALL_LOOP_BREAKER", "STOPPED_INCOMPLETE", "RUN_INCOMPLETE"}:
        return "context_too_small" if terminal.get("reason_code") == "CONTEXT_TOO_SMALL" else "incomplete"
    if result == "TEST_REPORTED_WRONG":
        return "wrong_test"
    if type(value) is str and value in RUNTIMES:
        return value
    if result == "SUBMITTED" or result == "DONE_ACCEPTED":
        return "completed"
    if result == "REFUSED":
        return "blocked"
    if result is None:
        return "missing_result"
    return "unknown"


def _delivery(summary: dict[str, object], result: str | None, artifact: str) -> str:
    if result == "DELIVERY_REFUSED" or type(summary.get("delivery_refusal")) is dict:
        return "refused"
    if type(summary.get("apply_error")) is str and summary.get("apply_error"):
        return "apply_failed"
    if type(summary.get("apply-receipt")) is dict or type(summary.get("apply_receipt")) is dict:
        return "applied"
    if type(summary.get("applied_in_place")) is list and bool(summary.get("applied_in_place")):
        return "applied"
    if artifact == "saved":
        return "saved"
    if result in {"SUBMITTED", "DONE_ACCEPTED"} or summary.get("candidate_captured") is False:
        return "none"
    return "unknown"


def _resume(value: object) -> str:
    if value is None:
        return "initial"
    if type(value) is not dict:
        return "unknown"
    for key in ("method", "mode", "source", "kind"):
        item = value.get(key)
        if type(item) is str and item in RESUMES:
            return item
    if value.get("checkpoint_restored") is True:
        return "checkpoint"
    if value.get("restart_live") is True:
        return "restart_live"
    return "unknown"


def _is_error(runtime: str, check: str, delivery: str, result: str | None,
              terminal: dict[str, object]) -> bool:
    if result in ERROR_RESULTS or (result is not None and result.startswith("ERR:")):
        return True
    if runtime in {"engine_error", "hook_containment", "mutation_containment", "recovery_required"}:
        return True
    if check == "fail" or delivery in {"refused", "apply_failed"}:
        return True
    if runtime == "interrupted" or terminal.get("interrupted") is True:
        return True
    return False


def _fixed(value: object, values: frozenset[str], fallback: str) -> str:
    return value if type(value) is str and value in values else fallback
