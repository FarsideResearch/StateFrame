"""Opt-in perpetual-worker soak harness and prepare-path sidecar.

Fix round: the in-process simulator was removed.  The soak DRIVER lives in
``probes/perpetual_soak_harness.py`` and runs N task epochs through the
existing commit-idempotency fake-provider child harness (real react_repair
worker, fresh processes, kills after observed done-path journal writes, resume
from the trusted anchor) under one durable worker identity.  This module keeps
the switch, the prepare-path sidecar, and pure accounting over the real
journals against an independent admitted-task list.  The live arm is a
specification only and is intentionally not invoked here.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Callable, Mapping, Sequence


MODE_ENV = "STATEFRAME_PERPETUAL_SOAK"
SCHEMA = "stateframe-perpetual-soak/1"
PREPARE_SCHEMA = SCHEMA
EFFECT_SCHEMA = "stateframe-perpetual-soak-effects/1"
SOAK_HEADER = "[Perpetual soak]"
PRODUCER = "kernel_perpetual_soak"

# The soak driver passes the durable worker identity / task epoch to every
# fresh child through these variables; the prepare hook journals them.
WORKER_ID_ENV = "STATEFRAME_PERPETUAL_SOAK_WORKER_ID"
TASK_ID_ENV = "STATEFRAME_PERPETUAL_SOAK_TASK_ID"
RESTART_EPOCH_ENV = "STATEFRAME_PERPETUAL_SOAK_RESTART_EPOCH"


class PerpetualSoakError(RuntimeError):
    """Invalid harness state or an unsafe recovery configuration."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None,
            environ: Mapping[str, str] | None = None) -> bool:
    if runtime is not None:
        configured = getattr(runtime, "stateframe_perpetual_soak_mode", None)
        if configured is not None:
            return mode({MODE_ENV: str(configured)}) == "on"
    return mode(environ) == "on"


def _as_nonnegative_int(value: object, default: int = 0) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _prepare_payload(value: Mapping[str, object] | None) -> dict[str, object]:
    value = value if isinstance(value, Mapping) else {}
    pending = value.get("pending_obligations", ())
    if isinstance(pending, (str, bytes, bytearray)):
        pending = (pending,)
    if not isinstance(pending, Sequence):
        pending = ()
    return {
        "schema": PREPARE_SCHEMA,
        "worker_identity": str(value.get("worker_identity") or "unbound-worker"),
        "active_task_id": str(value.get("active_task_id") or "none"),
        "task_index": _as_nonnegative_int(value.get("task_index")),
        "pending_obligations": [str(item) for item in pending if str(item).strip()],
        "restart_epoch": _as_nonnegative_int(value.get("restart_epoch")),
    }


def render_prepare_block(value: Mapping[str, object] | None) -> str:
    """Render the single model-visible prepare sidecar.

    ``SOAK_HEADER`` and ``parse_prepare_block`` are the shared producer/
    consumer contract.  The block deliberately does not borrow source labels;
    source-label mode therefore cannot create a second spelling of this header.
    """
    payload = _prepare_payload(value)
    return SOAK_HEADER + "\n" + json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def parse_prepare_block(value: str) -> dict[str, object]:
    """Parse and validate a sidecar produced by :func:`render_prepare_block`."""
    if not isinstance(value, str) or not value.startswith(SOAK_HEADER + "\n"):
        raise ValueError("not a long-run test prepare block")
    try:
        payload = json.loads(value[len(SOAK_HEADER) + 1:])
    except (TypeError, ValueError, UnicodeError) as error:
        raise ValueError("invalid long-run test prepare JSON") from error
    if not isinstance(payload, dict) or payload.get("schema") != PREPARE_SCHEMA:
        raise ValueError("invalid long-run test prepare schema")
    expected = {
        "schema", "worker_identity", "active_task_id", "task_index",
        "pending_obligations", "restart_epoch",
    }
    if set(payload) != expected:
        raise ValueError("incomplete long-run test prepare block")
    if (not isinstance(payload["worker_identity"], str)
            or not isinstance(payload["active_task_id"], str)
            or type(payload["task_index"]) is not int
            or payload["task_index"] < 0
            or not isinstance(payload["pending_obligations"], list)
            or any(not isinstance(item, str) for item in payload["pending_obligations"])
            or type(payload["restart_epoch"]) is not int
            or payload["restart_epoch"] < 0):
        raise ValueError("invalid long-run test prepare fields")
    return payload


def _runtime_prepare_state(runtime: object) -> dict[str, object]:
    state = getattr(runtime, "perpetual_soak_state", None)
    if isinstance(state, Mapping):
        return dict(state)
    owner = getattr(runtime, "owner", None)
    return {
        "worker_identity": os.environ.get(WORKER_ID_ENV, "prepare-worker"),
        "active_task_id": os.environ.get(TASK_ID_ENV) or getattr(owner, "task_id", "none"),
        "task_index": getattr(owner, "current_step", 0),
        "pending_obligations": (),
        "restart_epoch": os.environ.get(RESTART_EPOCH_ENV)
        or getattr(runtime, "perpetual_soak_restart_epoch", 0),
    }


def _record_prepare(runtime: object, payload: Mapping[str, object], *,
                    rendered: bool, reason: str | None = None) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    fields: dict[str, object] = {
        "producer": PRODUCER,
        "soak_schema": SCHEMA,
        "header": SOAK_HEADER,
        "worker_identity": payload["worker_identity"],
        "active_task_id": payload["active_task_id"],
        "task_index": payload["task_index"],
        "pending_obligations": list(payload["pending_obligations"]),
        "restart_epoch": payload["restart_epoch"],
        "rendered": bool(rendered),
        "dispatch": False,
    }
    if reason:
        fields["reason"] = reason
    capture = getattr(runtime, "_active_capture", None)
    request_id = getattr(capture, "request_id", None)
    if request_id:
        fields["request_id"] = request_id
    emit("perpetual_soak_prepare", **fields)


def append_prepare_block(
        runtime: object,
        messages: Sequence[dict],
        *,
        fit: Callable[[Sequence[dict]], int] | None = None,
        ceiling: int | None = None,
) -> tuple[list[dict], bool]:
    """Append the bounded sidecar only when the explicit switch is enabled.

    ``fit`` is the runtime's existing serialized-request estimator.  If the
    sidecar would consume the remaining ceiling, it is omitted and the reason
    is journaled; the hook never turns an already-fitting request into an
    oversized provider payload.
    """
    prepared = [copy.deepcopy(dict(message)) for message in messages]
    if not enabled(runtime):
        return prepared, False
    if any(str(message.get("content", "")).startswith(SOAK_HEADER)
           for message in prepared if isinstance(message, Mapping)):
        return prepared, True
    payload = _prepare_payload(_runtime_prepare_state(runtime))
    block = {"role": "user", "content": render_prepare_block(payload)}
    candidate = [*prepared, block]
    if fit is not None and ceiling is not None:
        estimate = int(fit(candidate))
        if estimate > int(ceiling):
            _record_prepare(runtime, payload, rendered=False, reason="ceiling")
            return prepared, False
    _record_prepare(runtime, payload, rendered=True)
    return candidate, True




# --------------------------------------------------------------------------
# Soak accounting over REAL worker journals (fix round, review #1/#2).
#
# The driver (probes/perpetual_soak_harness.py) runs every task epoch through
# the existing commit-idempotency fake-provider child harness: fresh processes,
# the real react_repair worker, kills after observed done-path journal writes,
# and resume from the trusted anchor.  This module only ACCOUNTS for what the
# real journals say, against an independent list of admitted task ids.
# --------------------------------------------------------------------------

EVENT_ROOT_COMPLETE = "continuation_transition"
EVENT_BUDGET_STATE = "challenge_budget_state"


def root_completes(tape: Sequence[Mapping[str, object]]) -> int:
    """Keyed root completions (the task's external effect) in one task journal."""
    return sum(
        1 for event in tape
        if event.get("kind") == EVENT_ROOT_COMPLETE
        and event.get("operation") == "complete"
        and isinstance(event.get("data"), Mapping)
        and event["data"].get("intention_id") == "root")


def terminal_keyed_submission(tape: Sequence[Mapping[str, object]]) -> bool:
    """The durable obligation: a terminal submission record whose effect is keyed."""
    for event in tape:
        if event.get("kind") != EVENT_BUDGET_STATE:
            continue
        state = event.get("budget_state")
        if not isinstance(state, Mapping) or not state.get("terminal"):
            continue
        record = state.get("submission_record")
        effect = record.get("submission_effect") if isinstance(record, Mapping) else None
        if isinstance(effect, Mapping) and effect.get("kind") == "keyed":
            return True
    return False


def _estimated_tokens(value: object) -> int:
    """Use the runtime's documented serialized-request estimate for CI cost."""
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return 0
    return (len(encoded) + 3) // 4


def _nonnegative_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return value


def _provider_usage(tape: Sequence[Mapping[str, object]]) -> tuple[int, int, int]:
    prompt_tokens = 0
    output_tokens = 0
    measured_calls = 0
    for event in tape:
        if event.get("kind") != "response_received":
            continue
        accounting = event.get("accounting")
        if not isinstance(accounting, Mapping):
            continue
        from kernel_prompt_accounting import accounting_tokens
        prompt = _nonnegative_number(accounting_tokens(accounting, event.get("response"))["total"])
        raw_usage = accounting.get("raw_usage")
        raw_usage = raw_usage if isinstance(raw_usage, Mapping) else {}
        output = _nonnegative_number(raw_usage.get("completion_tokens"))
        if prompt is None and output is None:
            continue
        prompt_tokens += int(prompt or 0)
        output_tokens += int(output or 0)
        measured_calls += 1
    return prompt_tokens, output_tokens, measured_calls


def cost_metrics(tape: Sequence[Mapping[str, object]], *,
                 model_usage: Sequence[Mapping[str, object]] = (),
                 wall_time_ms: float = 0.0) -> dict[str, object]:
    """Measure one real task's inference cost without inventing provider dollars.

    The fake-provider worker emits the same prepared request it gives the real
    runner, so CI records a serialized-request estimate plus the deterministic
    tool-call estimate.  A live provider tape with response accounting is
    preferred automatically.  ``source`` makes that distinction explicit.
    """
    if not isinstance(tape, Sequence) or isinstance(tape, (str, bytes, bytearray)):
        raise PerpetualSoakError("task tape is malformed")
    request_events = [
        event for event in tape
        if isinstance(event, Mapping) and event.get("kind") == "request_submitted"
    ]
    estimated_input = sum(
        _estimated_tokens(event.get("payload"))
        for event in request_events
        if isinstance(event.get("payload"), Mapping)
    )
    usage_input = 0
    usage_output = 0
    usage_count = 0
    for usage in model_usage:
        if not isinstance(usage, Mapping):
            continue
        input_tokens = _nonnegative_number(usage.get("input_tokens"))
        output_tokens = _nonnegative_number(usage.get("output_tokens"))
        if input_tokens is None and output_tokens is None:
            continue
        usage_input += int(input_tokens or 0)
        usage_output += int(output_tokens or 0)
        usage_count += 1

    provider_input, provider_output, provider_calls = _provider_usage(tape)
    if provider_calls:
        input_tokens = provider_input
        output_tokens = provider_output
        source = "provider_usage"
    elif usage_count:
        input_tokens = usage_input
        output_tokens = usage_output
        source = "runtime_request_estimate"
    else:
        input_tokens = estimated_input
        output_tokens = 0
        source = "runtime_request_estimate"
    model_calls = max(len(request_events), usage_count, provider_calls)
    wall_time = _nonnegative_number(wall_time_ms)
    if wall_time is None:
        raise PerpetualSoakError("wall time must be a non-negative number")
    return {
        "model_calls": model_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        # Token units are comparable in the deterministic child and remain
        # honest for live runs; a provider-specific USD price is not assumed.
        "cost_tokens": input_tokens + output_tokens,
        "wall_time_ms": round(float(wall_time), 3),
        "source": source,
    }


def account(expected_task_ids: Sequence[str],
            observed: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Loss/duplicate/drift against the INDEPENDENT admitted-task list.

    ``observed`` maps task id -> {"tape": [...], "correct": bool}.  A task that
    is missing, malformed, or lacks its durable keyed terminal submission is a
    LOST obligation; the count never comes from surviving entries alone.
    """
    if (not isinstance(expected_task_ids, Sequence) or isinstance(expected_task_ids, str)
            or len(set(expected_task_ids)) != len(expected_task_ids)):
        raise PerpetualSoakError("expected task ids must be a unique sequence")
    if not isinstance(observed, Mapping):
        raise PerpetualSoakError("observed task state is malformed")
    unknown = set(observed) - set(expected_task_ids)
    if unknown:
        raise PerpetualSoakError(f"observed tasks were never admitted: {sorted(unknown)}")
    lost: list[str] = []
    duplicates = 0
    incorrect: list[str] = []
    for task_id in expected_task_ids:
        entry = observed.get(task_id)
        tape = entry.get("tape") if isinstance(entry, Mapping) else None
        if not isinstance(tape, list) or any(not isinstance(e, Mapping) for e in tape):
            lost.append(task_id)
            incorrect.append(task_id)
            continue
        completes = root_completes(tape)
        duplicates += max(0, completes - 1)
        if completes < 1 or not terminal_keyed_submission(tape):
            lost.append(task_id)
        if entry.get("correct") is not True:
            incorrect.append(task_id)
    return {
        "expected_tasks": len(expected_task_ids),
        "obligations_lost": len(lost),
        "lost_task_ids": lost,
        "duplicate_effects": duplicates,
        "correctness_drift": len(incorrect),
        "incorrect_task_ids": incorrect,
    }


def live_mode_spec() -> dict[str, object]:
    """Return the provider-run contract without starting a live run."""
    return {
        "schema": "stateframe-perpetual-soak-live-spec/1",
        "run": "spec-only; not run in CI",
        "identity": "one stable worker identity, fresh runtime process per task/restart",
        "environment": {
            MODE_ENV: "on",
            WORKER_ID_ENV: "stable worker identity",
            "STATEFRAME_SUBMIT_IDEMPOTENT": "on",
            "STATEFRAME_CHALLENGE_BUDGET": "durable positive integer",
            "KERNEL_CONTINUATION_RESTART": "1",
        },
        "fault_boundaries": ("every journal write after the done dispatch "
                             "(probes/test_commit_idempotency.DONE_WRITES), seeded per task"),
        "restart": (
            "kill the real child after a selected done-path journal write, retain the trusted "
            "journal anchor and effect destination, then start a fresh child with the same worker id"
        ),
        "model": "real provider configured by the release runner; no live credentials in CI",
        "metrics": [
            "rss_bytes", "journal_bytes", "checkpoint_bytes", "restart_time_ms",
            "host_state_bytes", "state_growth_bytes", "cost_per_task",
            "input_tokens", "output_tokens", "obligations_lost", "duplicate_effects",
            "correctness_drift",
        ],
        "acceptance": {
            "lost_obligations": 0,
            "duplicate_effects": 0,
            "correctness_drift": 0,
            "new_legitimate_task_after_recovery": True,
        },
    }


__all__ = [
    "MODE_ENV", "PREPARE_SCHEMA", "PerpetualSoakError", "SOAK_HEADER", "WORKER_ID_ENV",
    "TASK_ID_ENV", "RESTART_EPOCH_ENV", "account", "append_prepare_block", "enabled",
    "live_mode_spec", "mode", "parse_prepare_block", "render_prepare_block", "root_completes",
    "terminal_keyed_submission", "cost_metrics",
]
