"""Opt-in stable-prefix KV chunk reuse for frame requests.

This is deliberately a small serving prototype.  llama.cpp can save and
restore a slot's prompt cache, but its slot file is a snapshot of the prompt
prefix it was given; it is not an arbitrary slice operation.  The benchmark
therefore warms a stable-only request, saves that slot, and restores the
digest-addressed artifact before the full frame request.  The runtime arm
only consumes an existing artifact and asks the server to reuse the prompt.

The default path does not import or call this module from request preparation,
and ``prepare_request(..., enabled_value=False)`` returns the original
objects.  That makes the opt-in boundary easy to audit against the lane
baseline.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import urllib.parse
import urllib.request
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path

from kernel_cache_layout import apply_layout, canonical_request_bytes, layout_message_kind
from kernel_context import token_estimate


MODE_ENV = "STATEFRAME_KV_CHUNK_REUSE"
SLOT_ENV = "STATEFRAME_KV_CHUNK_SLOT"
CACHE_DIR_ENV = "STATEFRAME_KV_CHUNK_CACHE_DIR"
RESTORE_REQUIRED_ENV = "STATEFRAME_KV_CHUNK_RESTORE_REQUIRED"
TIMEOUT_ENV = "STATEFRAME_KV_CHUNK_HTTP_TIMEOUT"
SCHEMA = "stateframe-kv-chunk-reuse/1"
_VALID_MODES = frozenset({"off", "on"})
_STABLE_KINDS = frozenset({"protocol", "ticket", "ticket_facts", "source"})
_SLOT_ACTIONS = frozenset({"save", "restore", "erase"})


def _source(environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def mode(environ: Mapping[str, str] | None = None) -> str:
    value = _source(environ).get(MODE_ENV, "off")
    if value not in _VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _binary_switch(name: str, environ: Mapping[str, str], default: str = "off") -> bool:
    value = environ.get(name, default)
    if value not in _VALID_MODES:
        raise ValueError(f"{name} must be off or on")
    return value == "on"


def _content(message: Mapping[str, object]) -> str:
    value = message.get("content", "")
    return value if isinstance(value, str) else ""


def _has_stateframe(messages: Sequence[Mapping[str, object]]) -> bool:
    return any(_content(message).startswith("[StateFrame]\n") for message in messages)


def _stable_prefix(messages: Sequence[Mapping[str, object]]) \
        -> tuple[Mapping[str, object], ...]:
    stable: list[Mapping[str, object]] = []
    for message in messages:
        if layout_message_kind(message) not in _STABLE_KINDS:
            break
        stable.append(message)
    return tuple(stable)


def _digest_for(stable_messages: Sequence[Mapping[str, object]], tools: object) -> str:
    payload = canonical_request_bytes(stable_messages, tools)
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class ChunkPlan:
    """A prepared presentation plus the stable prefix used for slot identity."""

    messages: object
    tools: object
    enabled: bool
    eligible: bool
    reason: str
    stable_messages: tuple[Mapping[str, object], ...] = ()
    chunk_digest: str | None = None
    cache_filename: str | None = None
    stable_message_count: int = 0
    stable_chars: int = 0
    stable_estimated_tokens: int = 0
    layout_applied: bool = False
    permutation_verified: bool = True

    def audit(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "mode": "on" if self.enabled else "off",
            "eligible": self.eligible,
            "reason": self.reason,
            "chunk_digest": self.chunk_digest,
            "cache_filename": self.cache_filename,
            "stable_message_count": self.stable_message_count,
            "stable_chars": self.stable_chars,
            "stable_estimated_tokens": self.stable_estimated_tokens,
            "layout_applied": self.layout_applied,
            "permutation_verified": self.permutation_verified,
        }


def _plan(messages: object, tools: object, *, selected: str, layout: bool) -> ChunkPlan:
    if selected == "off":
        return ChunkPlan(messages=messages, tools=tools, enabled=False,
                         eligible=False, reason="off")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise TypeError("messages must be a sequence")
    message_values = [message for message in messages if isinstance(message, Mapping)]
    if len(message_values) != len(messages):
        raise TypeError("every message must be a mapping")
    if not _has_stateframe(message_values):
        return ChunkPlan(messages=messages, tools=tools, enabled=True,
                         eligible=False, reason="no_stateframe")
    arranged = message_values
    arranged_tools = tools
    verified = True
    layout_applied = False
    if layout:
        result = apply_layout(message_values, tools, mode="off",
                              prompt_cache=True, state_placement="end")
        arranged = list(result.messages)
        arranged_tools = result.tools
        verified = result.permutation_verified
        layout_applied = True
    stable = _stable_prefix(arranged)
    if not stable:
        return ChunkPlan(messages=tuple(arranged) if layout else messages,
                         tools=arranged_tools, enabled=True, eligible=False,
                         reason="no_stable_prefix", layout_applied=layout_applied,
                         permutation_verified=verified)
    chunk_digest = _digest_for(stable, arranged_tools)
    stable_chars = sum(len(_content(message)) for message in stable)
    stable_tokens = token_estimate({"messages": list(stable), "tools": arranged_tools})
    return ChunkPlan(
        messages=tuple(arranged) if layout else messages,
        tools=arranged_tools,
        enabled=True,
        eligible=True,
        reason="stable_prefix",
        stable_messages=stable,
        chunk_digest=chunk_digest,
        cache_filename=f"kvchunk-{chunk_digest}.bin",
        stable_message_count=len(stable),
        stable_chars=stable_chars,
        stable_estimated_tokens=stable_tokens,
        layout_applied=layout_applied,
        permutation_verified=verified,
    )


def prepare_request(messages: Sequence[Mapping[str, object]], tools: object, *,
                    enabled_value: bool | None = None,
                    environ: Mapping[str, str] | None = None) -> ChunkPlan:
    """Compose the stable-prefix presentation for the real runtime prepare path."""
    selected = mode(environ) if enabled_value is None else ("on" if enabled_value else "off")
    return _plan(messages, tools, selected=selected, layout=True)


def describe_request(messages: Sequence[Mapping[str, object]], tools: object, *,
                     environ: Mapping[str, str] | None = None) -> ChunkPlan:
    """Describe the exact body about to be sent without reordering it."""
    return _plan(messages, tools, selected=mode(environ), layout=False)


def _slot_id(body: Mapping[str, object], environ: Mapping[str, str]) -> int:
    raw = environ.get(SLOT_ENV)
    if raw is None and type(body.get("id_slot")) is int:
        value = body["id_slot"]
    elif raw is None:
        raw = environ.get("REACT_SLOT_ID", "0")
        try:
            value = int(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{SLOT_ENV} must be a nonnegative integer") from error
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{SLOT_ENV} must be a nonnegative integer") from error
    if type(value) is not int or value < 0:
        raise ValueError(f"{SLOT_ENV} must be a nonnegative integer")
    return value


def _timeout(environ: Mapping[str, str]) -> float:
    raw = environ.get(TIMEOUT_ENV, "30")
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{TIMEOUT_ENV} must be positive") from error
    if value <= 0:
        raise ValueError(f"{TIMEOUT_ENV} must be positive")
    return value


def _cache_path(plan: ChunkPlan, environ: Mapping[str, str]) -> Path | None:
    directory = environ.get(CACHE_DIR_ENV, "").strip()
    if not directory or not plan.cache_filename:
        return None
    return Path(directory).expanduser() / plan.cache_filename


def _origin(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("llama.cpp endpoint must include an HTTP origin")
    return f"{parsed.scheme}://{parsed.netloc}"


def slot_action(endpoint: str, slot: int, action: str, filename: str | None,
                key: str = "", *, timeout: float = 30,
                opener=None) -> dict[str, object]:
    """Call llama.cpp's ``POST /slots/{slot}?action=...`` endpoint."""
    if type(slot) is not int or slot < 0:
        raise ValueError("slot must be a nonnegative integer")
    if action not in _SLOT_ACTIONS:
        raise ValueError("action must be save, restore, or erase")
    if filename is None and action != "erase":
        raise ValueError(f"{action} requires a filename")
    safe_filename = Path(filename).name if filename is not None else None
    if action != "erase" and not safe_filename:
        raise ValueError(f"{action} requires a filename")
    payload = {} if safe_filename is None else {"filename": safe_filename}
    url = f"{_origin(endpoint)}/slots/{slot}?action={action}"
    request = urllib.request.Request(
        url, data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
        method="POST")
    open_url = opener or urllib.request.urlopen
    handle = open_url(request, timeout=timeout)
    try:
        raw = handle.read()
    finally:
        close = getattr(handle, "close", None)
        if callable(close):
            close()
    result = json.loads(raw) if raw else {}
    if not isinstance(result, dict):
        raise ValueError("llama.cpp slot endpoint returned a non-object JSON value")
    return result


def configure_transport(body: MutableMapping[str, object], endpoint: str, key: str,
                        *, environ: Mapping[str, str] | None = None,
                        opener=None) -> dict[str, object] | None:
    """Apply the on-only transport flags and restore a known stable slot cache.

    A missing artifact is a normal cold start.  Restore failures are fail-open
    by default so the experiment cannot strand a repair run; set
    ``STATEFRAME_KV_CHUNK_RESTORE_REQUIRED=on`` when a benchmark wants a hard
    failure instead.
    """
    if not isinstance(body, MutableMapping):
        raise TypeError("request body must be mutable")
    env = _source(environ)
    if mode(env) == "off":
        return None
    messages = body.get("messages", ())
    tools = body.get("tools")
    plan = describe_request(messages, tools, environ=env)
    audit = plan.audit()
    if not plan.eligible:
        audit.update({"transport_applied": False, "restore_status": "not_eligible"})
        return audit
    slot = _slot_id(body, env)
    body["id_slot"] = slot
    body["cache_prompt"] = True
    audit.update({"transport_applied": True, "slot_id": slot})
    cache_path = _cache_path(plan, env)
    if cache_path is None:
        audit["restore_status"] = "not_configured"
        return audit
    audit["cache_path"] = str(cache_path)
    if not cache_path.is_file():
        audit["restore_status"] = "cache_miss"
        return audit
    try:
        restored = slot_action(endpoint, slot, "restore", plan.cache_filename, key,
                               timeout=_timeout(env), opener=opener)
    except Exception as error:  # noqa: BLE001 - fail-open is part of this prototype contract
        audit["restore_status"] = "restore_error"
        audit["restore_error_type"] = type(error).__name__
        if _binary_switch(RESTORE_REQUIRED_ENV, env):
            raise
        return audit
    audit["restore_status"] = "restored"
    if type(restored.get("n_restored")) is int and restored["n_restored"] >= 0:
        audit["n_restored"] = restored["n_restored"]
    return audit


__all__ = [
    "CACHE_DIR_ENV", "ChunkPlan", "MODE_ENV", "RESTORE_REQUIRED_ENV", "SCHEMA",
    "SLOT_ENV", "TIMEOUT_ENV", "configure_transport", "describe_request", "enabled",
    "mode", "prepare_request", "slot_action",
]
