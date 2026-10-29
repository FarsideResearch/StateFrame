"""Provider-neutral wire and diagnostic receipts for switch-policy studies.

The contract closes the specific micro-rollout gap identified by the switch
cluster without owning a transport, frame builder, editor, or evidence pin.
It binds a pairing seed to the request that is actually prepared, requires the
wire completion cap to equal the reserved output budget, and preserves finish
and diagnostic metadata before a caller parses actions.

No provider-specific fields are invented.  The seed label stays in the
receipt; only the provider-compatible ``seed`` and ``max_tokens`` fields are
placed in the request payload.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


SCHEMA = "stateframe-switch-wire/1"


class WireContractError(ValueError):
    """Raised when a request/receipt cannot preserve wire identity."""


def _text(value: object, field: str, *, limit: int = 512, required: bool = True) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        requirement = "non-empty " if required else ""
        raise WireContractError(f"{field} must be a {requirement}string of at most {limit} characters")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WireContractError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WireContractError(f"{field} must be a non-negative integer")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WireProfile:
    """One immutable execution profile used by preparation and validation."""

    input_token_ceiling: int
    output_reservation: int
    wire_output_cap: int
    context_token_ceiling: int | None = None
    wire_byte_ceiling: int | None = None

    def __post_init__(self) -> None:
        input_limit = _positive_int(self.input_token_ceiling, "input_token_ceiling")
        reserve = _positive_int(self.output_reservation, "output_reservation")
        cap = _positive_int(self.wire_output_cap, "wire_output_cap")
        if reserve != cap:
            raise WireContractError("wire output cap must equal output reservation")
        object.__setattr__(self, "input_token_ceiling", input_limit)
        object.__setattr__(self, "output_reservation", reserve)
        object.__setattr__(self, "wire_output_cap", cap)
        if self.context_token_ceiling is not None:
            object.__setattr__(self, "context_token_ceiling", _positive_int(self.context_token_ceiling, "context_token_ceiling"))
            if input_limit + reserve > self.context_token_ceiling:
                raise WireContractError("input plus output reservation exceeds context token ceiling")
        if self.wire_byte_ceiling is not None:
            object.__setattr__(self, "wire_byte_ceiling", _positive_int(self.wire_byte_ceiling, "wire_byte_ceiling"))

    def to_dict(self) -> dict[str, object]:
        return {
            "input_token_ceiling": self.input_token_ceiling,
            "output_reservation": self.output_reservation,
            "wire_output_cap": self.wire_output_cap,
            "context_token_ceiling": self.context_token_ceiling,
            "wire_byte_ceiling": self.wire_byte_ceiling,
        }


@dataclass(frozen=True)
class SeedBinding:
    """The canonical pairing identity carried beside the provider request."""

    pairing_id: str
    seed: int
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairing_id", _text(self.pairing_id, "pairing_id"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise WireContractError("seed must be a non-negative integer")
        object.__setattr__(self, "label", _text(self.label, "seed label", limit=128))

    def to_dict(self) -> dict[str, object]:
        return {"pairing_id": self.pairing_id, "seed": self.seed, "label": self.label}


@dataclass(frozen=True)
class WireValidation:
    accepted: bool
    reasons: tuple[str, ...] = ()
    payload_sha256: str | None = None
    input_tokens: int | None = None
    output_reservation: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "payload_sha256": self.payload_sha256,
            "input_tokens": self.input_tokens,
            "output_reservation": self.output_reservation,
        }


@dataclass(frozen=True)
class PreparedWireRequest:
    payload: Mapping[str, object] | None
    pairing: SeedBinding
    profile: WireProfile
    validation: WireValidation

    @property
    def accepted(self) -> bool:
        return self.validation.accepted

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "payload": dict(self.payload) if self.payload is not None else None,
            "pairing": self.pairing.to_dict(),
            "profile": self.profile.to_dict(),
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True)
class DiagnosticReceipt:
    """Bounded check/provider diagnostics with explicit omission accounting."""

    summary: str
    omitted_chars: int
    full_output_handle: str | None
    source: str = "provider"

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", _text(self.summary, "diagnostic summary", limit=4096, required=False))
        object.__setattr__(self, "omitted_chars", _nonnegative_int(self.omitted_chars, "omitted_chars"))
        if self.omitted_chars and not self.full_output_handle:
            raise WireContractError("truncated diagnostics require a full output handle")
        if self.full_output_handle is not None:
            object.__setattr__(self, "full_output_handle", _text(self.full_output_handle, "full_output_handle", limit=512))
        object.__setattr__(self, "source", _text(self.source, "diagnostic source", limit=128))

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "omitted_chars": self.omitted_chars,
            "full_output_handle": self.full_output_handle,
            "source": self.source,
        }


@dataclass(frozen=True)
class ResponseReceipt:
    """Metadata retained before action parsing or settlement classification."""

    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    content_present: bool
    reasoning_present: bool
    actions: tuple[str, ...]
    diagnostic: DiagnosticReceipt | None
    response_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "content_present": self.content_present,
            "reasoning_present": self.reasoning_present,
            "actions": list(self.actions),
            "diagnostic": self.diagnostic.to_dict() if self.diagnostic else None,
            "response_sha256": self.response_sha256,
        }


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise WireContractError(f"{field} must be an object")
    return value


def _optional_count(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, field)


def validate_wire_request(
    payload: Mapping[str, object],
    pairing: SeedBinding,
    profile: WireProfile,
) -> WireValidation:
    """Validate final payload values without changing the supplied mapping."""
    reasons: list[str] = []
    seed = payload.get("seed")
    if seed != pairing.seed or isinstance(seed, bool):
        reasons.append("WIRE_SEED_MISMATCH" if seed is not None else "WIRE_SEED_MISSING")
    max_tokens = payload.get("max_tokens")
    if max_tokens != profile.output_reservation or isinstance(max_tokens, bool):
        reasons.append(
            "WIRE_OUTPUT_RESERVATION_MISMATCH" if max_tokens is not None else "WIRE_OUTPUT_RESERVATION_MISSING"
        )
    input_tokens = payload.get("input_tokens")
    if input_tokens is not None:
        try:
            input_tokens = _nonnegative_int(input_tokens, "input_tokens")
        except WireContractError:
            reasons.append("WIRE_INPUT_TOKENS_INVALID")
            input_tokens = None
        else:
            if input_tokens > profile.input_token_ceiling:
                reasons.append("WIRE_INPUT_CEILING_EXCEEDED")
            if profile.context_token_ceiling is not None and input_tokens + profile.output_reservation > profile.context_token_ceiling:
                reasons.append("WIRE_CONTEXT_CEILING_EXCEEDED")
    if profile.wire_byte_ceiling is not None:
        wire_bytes = len(_canonical(dict(payload)).encode("utf-8"))
        if wire_bytes > profile.wire_byte_ceiling:
            reasons.append("WIRE_BYTE_CEILING_EXCEEDED")
    payload_sha256 = _digest(dict(payload))
    return WireValidation(
        accepted=not reasons,
        reasons=tuple(reasons),
        payload_sha256=payload_sha256,
        input_tokens=input_tokens,
        output_reservation=profile.output_reservation if not reasons or "WIRE_OUTPUT_RESERVATION_MISMATCH" not in reasons else (
            max_tokens if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) else None
        ),
    )


def prepare_wire_request(
    payload: Mapping[str, object],
    pairing: SeedBinding,
    profile: WireProfile,
) -> PreparedWireRequest:
    """Bind missing provider fields while rejecting conflicting caller values."""
    source = dict(_mapping(payload, "payload"))
    conflicts: list[str] = []
    if "seed" in source and source["seed"] != pairing.seed:
        conflicts.append("WIRE_SEED_ALREADY_BOUND_DIFFERENTLY")
    if "max_tokens" in source and source["max_tokens"] != profile.output_reservation:
        conflicts.append("WIRE_OUTPUT_RESERVATION_ALREADY_BOUND_DIFFERENTLY")
    if conflicts:
        validation = WireValidation(False, tuple(conflicts), _digest(source))
        return PreparedWireRequest(None, pairing, profile, validation)
    source["seed"] = pairing.seed
    source["max_tokens"] = profile.output_reservation
    validation = validate_wire_request(source, pairing, profile)
    return PreparedWireRequest(source if validation.accepted else None, pairing, profile, validation)


def capture_diagnostic(
    text: object,
    *,
    max_chars: int = 1_800,
    full_output_handle: str | None = None,
    source: str = "check",
) -> DiagnosticReceipt:
    """Create an honest bounded view; never turn missing detail into an empty result."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise WireContractError("max_chars must be a positive integer")
    value = str(text) if text is not None else ""
    omitted = max(0, len(value) - max_chars)
    summary = value if omitted == 0 else value[:max_chars]
    return DiagnosticReceipt(summary, omitted, full_output_handle, source=source)


def _action_label(value: object) -> str:
    if isinstance(value, str):
        return _text(value, "action", limit=128)
    if isinstance(value, Mapping):
        function = value.get("function")
        if isinstance(function, Mapping):
            name = function.get("name")
            if isinstance(name, str) and name.strip():
                return _text(name, "action", limit=128)
        for key in ("name", "tool", "type"):
            name = value.get(key)
            if isinstance(name, str) and name.strip():
                return _text(name, "action", limit=128)
    raise WireContractError("response action must be a name or provider action object")


def capture_response(response: Mapping[str, object], *, diagnostic_limit: int = 1_800) -> ResponseReceipt:
    """Retain finish/usage/actions/diagnostic facts before response parsing."""
    body = _mapping(response, "response")
    finish_value = body.get("finish_reason")
    finish_reason = None if finish_value is None else _text(finish_value, "finish_reason", limit=128)
    usage = body.get("usage")
    usage_map = usage if isinstance(usage, Mapping) else {}
    input_tokens = _optional_count(usage_map.get("prompt_tokens"), "prompt_tokens")
    output_tokens = _optional_count(usage_map.get("completion_tokens"), "completion_tokens")
    content = body.get("content")
    reasoning = body.get("reasoning_content", body.get("reasoning"))
    raw_actions = body.get("actions", ())
    if not isinstance(raw_actions, Sequence) or isinstance(raw_actions, (str, bytes, bytearray)):
        raise WireContractError("response actions must be a sequence")
    actions = tuple(_action_label(item) for item in raw_actions)
    diagnostic_value = body.get("diagnostic")
    if diagnostic_value is None:
        diagnostic = None
    else:
        diagnostic = capture_diagnostic(
            diagnostic_value,
            max_chars=diagnostic_limit,
            full_output_handle=body.get("full_output_handle") if isinstance(body.get("full_output_handle"), str) else None,
            source=str(body.get("diagnostic_source", "provider")),
        )
    digest_input = {
        "finish_reason": finish_reason,
        "usage": dict(usage_map),
        "content": content,
        "reasoning": reasoning,
        "actions": list(actions),
        "diagnostic": diagnostic.to_dict() if diagnostic else None,
    }
    return ResponseReceipt(
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        content_present=content is not None and str(content) != "",
        reasoning_present=reasoning is not None and str(reasoning) != "",
        actions=actions,
        diagnostic=diagnostic,
        response_sha256=_digest(digest_input),
    )


__all__ = [
    "DiagnosticReceipt", "PreparedWireRequest", "ResponseReceipt", "SCHEMA",
    "SeedBinding", "WireContractError", "WireProfile", "WireValidation",
    "capture_diagnostic", "capture_response", "prepare_wire_request",
    "validate_wire_request",
]
