"""Transport-side request and outcome types for System One scoring.

This module is deliberately narrower than the System One decision API.  It
describes the qualified token/template inputs that a provider adapter needs and
the wire-level domain/outcome vocabulary.  It does not select an option or
interpret a sampled label.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
SCHEMA = "sys1-score-transport/1"
DEFAULT_TOP_LOGPROBS = 20
DEFAULT_DIAGNOSTIC_BYTES = 8 * 1024


class ProtocolError(ValueError):
    """Raised when a host-owned request or qualification binding is invalid."""


class TransportKind(str, Enum):
    CHAT = "chat"
    NATIVE_N_PROBS = "native_n_probs"


class ScoreDomain(str, Enum):
    """The provider operation that produced a score value.

    Pre-sampler logprobs and post-sampling probabilities are intentionally
    different domains.  They must never be inferred from the emitted label or
    silently mixed in one vector.
    """

    PRE_SAMPLER_LOGPROB = "pre_sampler_logprob"
    RAW_PRE_SAMPLER_LOGPROB = "pre_sampler_logprob"
    PRE_SAMPLER_LOGIT = "pre_sampler_logit"
    RAW_PRE_SAMPLER_LOGIT = "pre_sampler_logit"
    POST_SAMPLING_PROB = "post_sampling_probability"
    POST_SAMPLING_PROBABILITY = "post_sampling_probability"


class OutcomeStatus(str, Enum):
    ADMITTED = "ADMITTED"
    SCORES_INCOMPLETE = "SCORES_INCOMPLETE"
    UNAVAILABLE = "UNAVAILABLE"


class UsageStatus(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"


class DiagnosticCode(str, Enum):
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"
    RESPONSE_SHAPE_INVALID = "RESPONSE_SHAPE_INVALID"
    WRONG_POSITION = "WRONG_POSITION"
    ONE_TOKEN_REQUIRED = "ONE_TOKEN_REQUIRED"
    DOMAIN_AMBIGUOUS = "DOMAIN_AMBIGUOUS"
    DOMAIN_MISSING = "DOMAIN_MISSING"
    DUPLICATE_ID = "DUPLICATE_ID"
    INVALID_SCORE = "INVALID_SCORE"
    MISSING_ID = "MISSING_ID"
    MISSING_BYTES = "MISSING_BYTES"
    TOKEN_ID_MISMATCH = "TOKEN_ID_MISMATCH"
    SCORE_DOMAIN_MISMATCH = "SCORE_DOMAIN_MISMATCH"
    SUBSET_SOFTMAX_FORBIDDEN = "SUBSET_SOFTMAX_FORBIDDEN"
    LEGAL_MASS_INVALID = "LEGAL_MASS_INVALID"
    LENGTH_WITHOUT_COMPLETE_LABEL = "LENGTH_WITHOUT_COMPLETE_LABEL"
    USAGE_UNKNOWN = "USAGE_UNKNOWN"
    MISSING_LABEL_IDS = "MISSING_LABEL_IDS"
    SCORES_INCOMPLETE = "SCORES_INCOMPLETE"


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ProtocolError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True, slots=True)
class TokenBinding:
    """One tokenizer-qualified, single-token output label.

    ``bytes`` are supplied by qualification, never reconstructed from the
    label text at response time.  The eventual SYS1_API labels module can
    provide objects with this same structural contract.
    """

    semantic_id: str
    label: str
    token_id: int
    bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_id, str) or not self.semantic_id:
            raise ProtocolError("semantic_id is required")
        if not isinstance(self.label, str) or len(self.label) != 1:
            raise ProtocolError("a transport label must be exactly one character")
        if self.label.isspace() or any(ord(char) < 0x20 for char in self.label):
            raise ProtocolError("a transport label cannot contain whitespace/control text")
        if isinstance(self.token_id, bool) or not isinstance(self.token_id, int) or self.token_id < 0:
            raise ProtocolError("token_id must be a non-negative integer")
        try:
            token_bytes = tuple(self.bytes)
        except TypeError as exc:
            raise ProtocolError("bytes must be an integer sequence") from exc
        if not token_bytes:
            raise ProtocolError("bytes are required for a qualified token")
        if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in token_bytes):
            raise ProtocolError("bytes must contain only octets")
        object.__setattr__(self, "bytes", token_bytes)


# Short alias for callers that use the report's terminology.
LabelToken = TokenBinding


@dataclass(frozen=True, slots=True)
class QualifiedTemplate:
    """A host-qualified chat/native template identity.

    The rendered native prompt is optional for chat and required by the native
    ``/completion`` request builder.  ``enable_thinking`` is fixed off for this
    one-token protocol.
    """

    template_id: str
    fingerprint: str
    rendered_prompt: str | None = None
    enable_thinking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id.strip():
            raise ProtocolError("template_id is required")
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise ProtocolError("template fingerprint is required")
        if self.enable_thinking:
            raise ProtocolError("SYS1 score transport requires thinking disabled")
        if self.rendered_prompt is not None and not isinstance(self.rendered_prompt, str):
            raise ProtocolError("rendered_prompt must be text when supplied")
        if self.rendered_prompt == "":
            raise ProtocolError("rendered_prompt cannot be empty when supplied")


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    """One bounded provider request; the adapter never retries it."""

    prompt: str
    labels: tuple[TokenBinding, ...]
    template: QualifiedTemplate
    model: str = "decision-tier"
    seed: int = 17
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    top_logprobs: int = DEFAULT_TOP_LOGPROBS

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ProtocolError("prompt is required")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ProtocolError("model is required")
        try:
            labels = tuple(self.labels)
        except TypeError as exc:
            raise ProtocolError("labels must be a sequence") from exc
        if len(labels) < 2:
            raise ProtocolError("at least two labels are required")
        if any(not isinstance(label, TokenBinding) for label in labels):
            raise ProtocolError("labels must be tokenizer-qualified TokenBinding values")
        semantic_ids = [label.semantic_id for label in labels]
        token_ids = [label.token_id for label in labels]
        byte_values = [label.bytes for label in labels]
        if len(set(semantic_ids)) != len(labels):
            raise ProtocolError("duplicate semantic label IDs are not allowed")
        if len(set(token_ids)) != len(labels):
            raise ProtocolError("duplicate token IDs are not allowed")
        if len(set(byte_values)) != len(labels):
            raise ProtocolError("duplicate token byte spellings are not allowed")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ProtocolError("seed must be an integer")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ProtocolError("top_k must be a non-negative integer")
        if isinstance(self.top_logprobs, bool) or not isinstance(self.top_logprobs, int) or self.top_logprobs <= 0:
            raise ProtocolError("top_logprobs must be a positive integer")
        for name in (
            "temperature",
            "top_p",
            "min_p",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            numeric = _finite_positive(getattr(self, name), name) if name in {"temperature", "repeat_penalty"} else getattr(self, name)
            if isinstance(numeric, bool) or not isinstance(numeric, (int, float)) or not math.isfinite(float(numeric)):
                raise ProtocolError(f"{name} must be finite and numeric")
        if not 0 <= float(self.top_p) <= 1:
            raise ProtocolError("top_p must be between zero and one")
        if not 0 <= float(self.min_p) <= 1:
            raise ProtocolError("min_p must be between zero and one")
        if not isinstance(self.template, QualifiedTemplate):
            raise ProtocolError("template must be QualifiedTemplate")
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True, slots=True)
class UsageInfo:
    """Provider usage is evidence, not something the adapter estimates."""

    status: UsageStatus
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, UsageStatus):
            raise ProtocolError("usage status must be UsageStatus")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ProtocolError(f"{name} must be a non-negative integer when present")
        if self.status is UsageStatus.KNOWN and all(
            getattr(self, name) is None
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise ProtocolError("known usage must contain at least one token count")

    @classmethod
    def unknown(cls) -> "UsageInfo":
        return cls(UsageStatus.UNKNOWN)

    @classmethod
    def from_mapping(cls, value: object) -> "UsageInfo":
        if not isinstance(value, Mapping):
            return cls.unknown()
        values: list[int | None] = []
        seen = False
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            raw = value.get(name)
            if raw is None:
                values.append(None)
            elif isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                return cls.unknown()
            else:
                seen = True
                values.append(raw)
        return cls(UsageStatus.KNOWN, *values) if seen else cls.unknown()


@dataclass(frozen=True, slots=True)
class TransportDiagnostics:
    """Bounded, non-authoritative diagnostics for one adapter attempt."""

    codes: tuple[str, ...] = ()
    message: str | None = None
    missing_ids: tuple[int, ...] = ()
    missing_semantic_ids: tuple[str, ...] = ()
    generated_token_id: int | None = None
    generated_token: str | None = None
    domain: ScoreDomain | None = None
    transport: TransportKind | None = None
    finish_reason: str | None = None
    usage_status: UsageStatus = UsageStatus.UNKNOWN
    raw_envelope: str | None = None
    raw_envelope_truncated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.usage_status, UsageStatus):
            raise ProtocolError("diagnostic usage status must be UsageStatus")
        if self.domain is not None and not isinstance(self.domain, ScoreDomain):
            raise ProtocolError("diagnostic domain must be ScoreDomain when supplied")
        if self.transport is not None and not isinstance(self.transport, TransportKind):
            raise ProtocolError("diagnostic transport must be TransportKind when supplied")
        codes = tuple(str(code) for code in self.codes)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.missing_ids):
            raise ProtocolError("missing token IDs must be integers")
        if any(not isinstance(value, str) or not value for value in self.missing_semantic_ids):
            raise ProtocolError("missing semantic IDs must be non-empty text")
        missing_ids = tuple(self.missing_ids)
        missing_semantic_ids = tuple(self.missing_semantic_ids)
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "missing_ids", missing_ids)
        object.__setattr__(self, "missing_semantic_ids", missing_semantic_ids)
        if len(self.codes) > 16:
            raise ProtocolError("diagnostic code list is unbounded")
        if len(self.missing_ids) > 128 or len(self.missing_semantic_ids) > 128:
            raise ProtocolError("missing-ID diagnostics exceed bound")
        if any(value < 0 for value in self.missing_ids):
            raise ProtocolError("missing token IDs must be non-negative")
        if self.message is not None and not isinstance(self.message, str):
            raise ProtocolError("diagnostic message must be text when supplied")
        if not isinstance(self.raw_envelope_truncated, bool):
            raise ProtocolError("raw envelope truncation flag must be boolean")
        if self.raw_envelope is not None and not isinstance(self.raw_envelope, str):
            raise ProtocolError("raw envelope must be text when supplied")
        if self.raw_envelope is not None and len(self.raw_envelope.encode("utf-8")) > DEFAULT_DIAGNOSTIC_BYTES:
            raise ProtocolError("raw envelope exceeds diagnostic bound")


__all__ = [
    "DEFAULT_DIAGNOSTIC_BYTES",
    "DEFAULT_TOP_LOGPROBS",
    "DiagnosticCode",
    "LabelToken",
    "OutcomeStatus",
    "ProtocolError",
    "QualifiedTemplate",
    "SCHEMA",
    "ScoreDomain",
    "ScoreRequest",
    "TokenBinding",
    "TransportDiagnostics",
    "TransportKind",
    "UsageInfo",
    "UsageStatus",
]
