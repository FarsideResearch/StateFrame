"""Pre-edit requirement restatement (soak CONFIRMED lead C1, restate_req).

Paired qualification (same endpoint): treatment 6/12 vs control
0/12 vs token-matched placebo 0/12. The measured lever is ONE user message, with
exactly this text, placed immediately after the system prompt, while no product
edit has been applied. Text and position are the lever: do not reword or move it.

Language-generic: the instruction names no language, file type or task.
The default ``auto`` policy is intentionally narrow: it applies only in LEAN
when the declared public scope contains at least two files. Explicit ``on`` and
``off`` remain available for controlled replays and release claims.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os

ENV = "STATEFRAME_RESTATE_REQUIREMENT"
LEGACY_ENV = "STATEFRAME_LEAN_STACK_RESTATE_REQUIREMENT"
MESSAGE = ("Begin by restating the requirement in one sentence together with the "
           "observable behaviour that will prove it is fixed.")
MESSAGE_SHA256 = hashlib.sha256(MESSAGE.encode()).hexdigest()
CLAIM_FIELD = "restate_requirement_claim"
STATUS_FIELD = "restate_requirement_status"


def legacy_mode(environ: Mapping[str, str] | None = None) -> str:
    """Read the retired stack switch for protocol documentation and migration."""
    values = os.environ if environ is None else environ
    value = str(values.get(LEGACY_ENV, "off") or "off").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{LEGACY_ENV} must be on or off")
    return value


def mode(environ: Mapping[str, str] | None = None) -> str:
    values = os.environ if environ is None else environ
    raw = values.get(ENV)
    if raw is None and legacy_mode(values) == "on":
        raw = "on"
    value = str(raw or "auto").strip().lower()
    if value not in {"auto", "on", "off"}:
        raise ValueError(f"{ENV} must be auto, on, or off")
    return value


def enabled(runtime: object, environ: Mapping[str, str] | None = None) -> bool:
    """Resolve explicit controls or the LEAN multi-file public-scope default."""
    values = os.environ if environ is None else environ
    configured = mode(values)
    if configured == "on":
        return True
    if configured == "off":
        return False
    from kernel_lean import enabled as lean_enabled
    if not lean_enabled(values):
        return False
    from kernel_completion_policy import _safe_declared_paths
    return len(_safe_declared_paths(runtime)) >= 2


def _insert_index(messages: Sequence[Mapping[str, object]]) -> int:
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1
    return index


def _is_lever(row: object) -> bool:
    return isinstance(row, Mapping) and row.get("role") == "user" and row.get("content") == MESSAGE


def prepare(runtime: object, messages: list[dict[str, object]], tools: object, capture: object,
            *, token_counter, environ: Mapping[str, str] | None = None) -> list[dict[str, object]]:
    """Return the request messages with the lever placed after the system prefix.

    Every turn first removes any carried copy, so the lever appears at most once
    and disappears as soon as a product edit is applied.
    """
    cleaned = [row for row in messages if not _is_lever(row)]
    if hasattr(capture, CLAIM_FIELD):
        delattr(capture, CLAIM_FIELD)
    try:
        configured = mode(environ)
        if not enabled(runtime, environ):
            setattr(capture, STATUS_FIELD, {"status": "OFF", "mode": configured})
            return cleaned
        from kernel_preedit_coverage_runtime import _zero_edits
        if not _zero_edits(runtime):
            setattr(capture, STATUS_FIELD, {"status": "NOOP", "reason": "product_edit_applied"})
            return cleaned
        index = _insert_index(cleaned)
        if index == 0:
            setattr(capture, STATUS_FIELD, {"status": "NOOP", "reason": "no_system_prefix"})
            return cleaned
        candidate = cleaned[:index] + [{"role": "user", "content": MESSAGE}] + cleaned[index:]
        tokens = token_counter({"messages": candidate, "tools": tools})
        ceiling = getattr(runtime, "frame_ceiling", None)
        if type(tokens) is not int or tokens < 0 or (type(ceiling) is int and tokens > ceiling):
            setattr(capture, STATUS_FIELD, {"status": "NOOP", "reason": "request_budget_exceeded"})
            return cleaned
    except (AttributeError, ImportError, KeyError, LookupError, OSError, TypeError, ValueError) as error:
        setattr(capture, STATUS_FIELD, {"status": "NOOP", "reason": "host_input_unavailable",
                                        "error_type": type(error).__name__})
        return cleaned
    setattr(capture, STATUS_FIELD, {"status": "READY", "index": index})
    setattr(capture, CLAIM_FIELD, {"message_sha256": MESSAGE_SHA256, "index": index})
    return candidate


def submission(capture: object, wire: bytes) -> dict[str, object] | None:
    """Authenticate the lever in the exact serialized bytes about to be sent."""
    claim = getattr(capture, CLAIM_FIELD, None)
    if not isinstance(claim, Mapping):
        return None
    try:
        messages = json.loads(wire).get("messages")
    except (TypeError, ValueError, AttributeError):
        return None
    if not isinstance(messages, list) or not all(isinstance(row, Mapping) for row in messages):
        return None
    positions = [index for index, row in enumerate(messages) if _is_lever(row)]
    index = _insert_index(messages)
    if positions != [index] or index == 0:
        return None
    return {"message_sha256": MESSAGE_SHA256, "index": index,
            "body_sha256": "sha256:" + hashlib.sha256(wire).hexdigest()}
