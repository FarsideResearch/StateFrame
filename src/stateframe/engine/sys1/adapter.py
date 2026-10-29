"""Strict chat/native score transport adapters for System One.

The adapter builds one bounded request and parses one recorded response.  It
does not own an HTTP client, retry a forced label, choose an action, or turn a
sampled letter/top-two margin into confidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .protocol import (
    DEFAULT_DIAGNOSTIC_BYTES,
    DiagnosticCode,
    OutcomeStatus,
    ProtocolError,
    SCHEMA,
    ScoreDomain,
    ScoreRequest,
    TokenBinding,
    TransportDiagnostics,
    TransportKind,
    UsageInfo,
    UsageStatus,
)
from .scoring import (
    DuplicateScoreError,
    IncompleteScoresError,
    LegalMassError,
    NormalizedScoreVector,
    NormalizationError,
    ScoreEntry,
    ScoreValidationError,
    ScoreVector,
    SubsetNormalizationError,
    TokenIdentityError,
    admit_complete_vector,
    normalize_score_vector,
)


class _AdapterFailure(Exception):
    def __init__(
        self,
        code: DiagnosticCode,
        message: str,
        *,
        missing: Sequence[TokenBinding] = (),
    ) -> None:
        self.code = code
        self.message = message
        self.missing = tuple(missing)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class TransportOutcome:
    """Typed result of one adapter attempt, with no unbounded response copy."""

    status: OutcomeStatus
    score_vector: ScoreVector | None
    normalized: NormalizedScoreVector | None
    usage: UsageInfo
    diagnostics: TransportDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.status, OutcomeStatus):
            raise ProtocolError("outcome status must be OutcomeStatus")
        if not isinstance(self.usage, UsageInfo):
            raise ProtocolError("outcome usage must be UsageInfo")
        if not isinstance(self.diagnostics, TransportDiagnostics):
            raise ProtocolError("outcome diagnostics must be TransportDiagnostics")
        if self.diagnostics.usage_status is not self.usage.status:
            raise ProtocolError("outcome diagnostics and usage status disagree")
        if self.status is OutcomeStatus.ADMITTED:
            if not isinstance(self.score_vector, ScoreVector) or not isinstance(self.normalized, NormalizedScoreVector):
                raise ProtocolError("admitted outcome must carry vector and normalization")
            if self.normalized.vector != self.score_vector:
                raise ProtocolError("normalized outcome must reference its admitted vector")
        elif self.score_vector is not None or self.normalized is not None:
            raise ProtocolError("non-admitted outcome cannot carry score values")

    @property
    def admitted(self) -> bool:
        return self.status == OutcomeStatus.ADMITTED

    @property
    def usage_unknown(self) -> bool:
        return self.usage.status == UsageStatus.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        normalized = self.normalized
        normalized_mapping: dict[str, object] | None = None
        if normalized is not None:
            normalized_mapping = {
                "domain": normalized.vector.domain.value,
                "source": normalized.vector.source,
                "probabilities": list(normalized.probabilities),
                "legal_mass": normalized.legal_mass,
            }
        return {
            "schema": SCHEMA,
            "status": self.status.value,
            "score_vector": (
                {
                    "domain": self.score_vector.domain.value,
                    "source": self.score_vector.source,
                    "token_ids": list(self.score_vector.token_ids),
                    "raw_scores": list(self.score_vector.raw_scores),
                }
                if self.score_vector is not None
                else None
            ),
            "normalized": normalized_mapping,
            "usage": {
                "status": self.usage.status.value,
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "diagnostics": {
                "codes": list(self.diagnostics.codes),
                "message": self.diagnostics.message,
                "missing_ids": list(self.diagnostics.missing_ids),
                "missing_semantic_ids": list(self.diagnostics.missing_semantic_ids),
                "generated_token_id": self.diagnostics.generated_token_id,
                "generated_token": self.diagnostics.generated_token,
                "domain": self.diagnostics.domain.value if self.diagnostics.domain else None,
                "transport": self.diagnostics.transport.value if self.diagnostics.transport else None,
                "finish_reason": self.diagnostics.finish_reason,
                "usage_status": self.diagnostics.usage_status.value,
                "raw_envelope": self.diagnostics.raw_envelope,
                "raw_envelope_truncated": self.diagnostics.raw_envelope_truncated,
            },
        }


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _AdapterFailure(DiagnosticCode.RESPONSE_SHAPE_INVALID, f"{name} must be an object")
    return value


def _as_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _AdapterFailure(DiagnosticCode.RESPONSE_SHAPE_INVALID, f"{name} must be an array")
    return value


def _bounded_envelope(value: object, limit: int) -> tuple[str | None, bool]:
    if limit <= 0:
        return None, False
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=repr, separators=(",", ":"))
    except (TypeError, ValueError):
        serialized = repr(value)
    raw = serialized.encode("utf-8")
    if len(raw) <= limit:
        return serialized, False
    clipped = raw[:limit].decode("utf-8", errors="ignore")
    return clipped, True


def _bounded_message(value: str, limit: int = 512) -> str:
    """Keep provider/error text useful without creating an unbounded receipt."""

    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return value
    return encoded[:limit].decode("utf-8", errors="ignore")


def _grammar(labels: Sequence[TokenBinding]) -> str:
    return "root ::= " + " | ".join(json.dumps(label.label, ensure_ascii=False) for label in labels)


def _common_sampling(request: ScoreRequest) -> dict[str, object]:
    return {
        "model": request.model,
        "temperature": request.temperature,
        "top_k": request.top_k,
        "top_p": request.top_p,
        "min_p": request.min_p,
        "repeat_penalty": request.repeat_penalty,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "samplers": ["temperature"],
        "seed": request.seed,
        "cache_prompt": True,
        "stream": False,
        "post_sampling_probs": False,
        "grammar": _grammar(request.labels),
    }


def build_chat_request(request: ScoreRequest) -> dict[str, object]:
    """Build the one-token chat request with pre-sampling scores by default."""

    if not isinstance(request, ScoreRequest):
        raise ProtocolError("request must be ScoreRequest")
    body = _common_sampling(request)
    body.update(
        {
            "messages": [{"role": "user", "content": request.prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": request.top_logprobs,
        }
    )
    return body


def build_native_request(request: ScoreRequest) -> dict[str, object]:
    """Build the declared native ``n_probs`` alternative.

    ``n_probs`` changes the transport field, not the admission promise:
    missing requested IDs still produce ``SCORES_INCOMPLETE``.
    """

    if not isinstance(request, ScoreRequest):
        raise ProtocolError("request must be ScoreRequest")
    rendered_prompt = request.template.rendered_prompt
    if not rendered_prompt:
        raise ProtocolError("native scoring requires a qualified rendered template prompt")
    body = _common_sampling(request)
    body.update(
        {
            "prompt": rendered_prompt,
            "n_predict": 1,
            "max_tokens": 1,
            "n_probs": request.top_logprobs,
            "return_tokens": True,
        }
    )
    return body


def _explicit_domain(value: object) -> ScoreDomain | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, "score domain tag must be text")
    normalized = value.lower().replace("-", "_")
    mappings = {
        "pre": ScoreDomain.PRE_SAMPLER_LOGPROB,
        "pre_sampler": ScoreDomain.PRE_SAMPLER_LOGPROB,
        "pre_sampler_logprob": ScoreDomain.PRE_SAMPLER_LOGPROB,
        "raw_pre_sampler_logprob": ScoreDomain.PRE_SAMPLER_LOGPROB,
        "raw_pre_sampler_logits": ScoreDomain.PRE_SAMPLER_LOGIT,
        "pre_sampler_logit": ScoreDomain.PRE_SAMPLER_LOGIT,
        "post": ScoreDomain.POST_SAMPLING_PROB,
        "post_sampling": ScoreDomain.POST_SAMPLING_PROB,
        "post_sampling_probability": ScoreDomain.POST_SAMPLING_PROB,
        "post_sampling_prob": ScoreDomain.POST_SAMPLING_PROB,
    }
    result = mappings.get(normalized)
    if result is None:
        raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, f"unsupported score domain tag {value!r}")
    return result


def _position_domain(
    position: Mapping[str, object], inherited_domain: ScoreDomain | None = None
) -> tuple[ScoreDomain, str, str]:
    explicit = _explicit_domain(position.get("domain"))
    if inherited_domain is not None and explicit is not None and inherited_domain != explicit:
        raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, "position and envelope score domains disagree")
    explicit = explicit or inherited_domain
    pre = any(key in position for key in ("logprob", "top_logprobs", "logit", "top_logits"))
    post = any(key in position for key in ("prob", "top_probs"))
    inferred: ScoreDomain | None
    if pre and post:
        raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, "pre and post score fields are mixed")
    if pre:
        inferred = ScoreDomain.PRE_SAMPLER_LOGIT if "logit" in position or "top_logits" in position else ScoreDomain.PRE_SAMPLER_LOGPROB
    elif post:
        inferred = ScoreDomain.POST_SAMPLING_PROB
    else:
        inferred = None
    if explicit is not None and inferred is not None and explicit != inferred:
        raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, "explicit and wire score domains disagree")
    domain = explicit or inferred
    if domain is None:
        raise _AdapterFailure(DiagnosticCode.DOMAIN_MISSING, "score domain is not declared by the response")
    if domain == ScoreDomain.PRE_SAMPLER_LOGIT:
        return domain, "logit", "top_logits"
    if domain == ScoreDomain.POST_SAMPLING_PROB:
        return domain, "prob", "top_probs"
    return domain, "logprob", "top_logprobs"


def _entry(value: object, *, domain: ScoreDomain, score_key: str) -> ScoreEntry:
    mapping = _as_mapping(value, "score entry")
    if "id" not in mapping or mapping["id"] is None:
        raise _AdapterFailure(DiagnosticCode.MISSING_ID, "score entry has no token ID")
    if "bytes" not in mapping or mapping["bytes"] is None:
        raise _AdapterFailure(DiagnosticCode.MISSING_BYTES, "score entry has no token bytes")
    raw_bytes = mapping["bytes"]
    if isinstance(raw_bytes, (str, bytes, bytearray, Sequence)) and not raw_bytes:
        raise _AdapterFailure(DiagnosticCode.MISSING_BYTES, "score entry has no token bytes")
    if "token" not in mapping:
        raise _AdapterFailure(DiagnosticCode.RESPONSE_SHAPE_INVALID, "score entry has no token text")
    if score_key not in mapping:
        raise _AdapterFailure(DiagnosticCode.INVALID_SCORE, f"score entry has no {score_key}")
    if any(key in mapping for key in ("logprob", "top_logprobs", "logit", "top_logits")) and domain == ScoreDomain.POST_SAMPLING_PROB:
        raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, "post score entry contains pre-sampler fields")
    if any(key in mapping for key in ("prob", "top_probs")) and domain != ScoreDomain.POST_SAMPLING_PROB:
        raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, "pre score entry contains post-sampling fields")
    try:
        position = mapping.get("position", 0)
        return ScoreEntry(
            token_id=mapping["id"],
            token=mapping["token"],
            bytes=tuple(mapping["bytes"]),
            value=mapping[score_key],
            domain=domain,
            position=position,
        )
    except _AdapterFailure:
        raise
    except ScoreValidationError as exc:
        message = str(exc)
        code = DiagnosticCode.WRONG_POSITION if "position" in message else DiagnosticCode.INVALID_SCORE
        raise _AdapterFailure(code, message) from exc
    except (TypeError, ValueError, KeyError) as exc:
        raise _AdapterFailure(DiagnosticCode.RESPONSE_SHAPE_INVALID, "score entry fields have invalid types") from exc


def _extract_position(
    response: Mapping[str, object], kind: TransportKind
) -> tuple[Mapping[str, object], Mapping[str, object], str | None, ScoreDomain | None]:
    inherited_domain = _explicit_domain(response.get("score_domain"))
    if "domain" in response:
        response_domain = _explicit_domain(response.get("domain"))
        if inherited_domain is not None and response_domain is not None and inherited_domain != response_domain:
            raise _AdapterFailure(DiagnosticCode.DOMAIN_AMBIGUOUS, "response score domains disagree")
        inherited_domain = inherited_domain or response_domain
    if kind == TransportKind.CHAT:
        choices = _as_sequence(response.get("choices"), "choices")
        if len(choices) != 1:
            raise _AdapterFailure(DiagnosticCode.ONE_TOKEN_REQUIRED, "response must contain exactly one choice")
        choice = _as_mapping(choices[0], "choice")
        logprobs = _as_mapping(choice.get("logprobs"), "choice.logprobs")
        positions = _as_sequence(logprobs.get("content"), "choices[0].logprobs.content")
        finish = choice.get("finish_reason")
    else:
        choices = response.get("choices")
        if choices is not None:
            rows = _as_sequence(choices, "choices")
            if len(rows) != 1:
                raise _AdapterFailure(DiagnosticCode.ONE_TOKEN_REQUIRED, "response must contain exactly one choice")
            choice = _as_mapping(rows[0], "choice")
        else:
            choice = response
        raw_positions = choice.get("completion_probabilities", response.get("completion_probabilities"))
        positions = _as_sequence(raw_positions, "completion_probabilities")
        finish_value = choice.get("finish_reason", response.get("finish_reason"))
        finish = finish_value if isinstance(finish_value, str) else None
    if len(positions) != 1:
        raise _AdapterFailure(DiagnosticCode.ONE_TOKEN_REQUIRED, "response must contain exactly one generated position")
    position = _as_mapping(positions[0], "generated position")
    position_number = position.get("position", 0)
    if isinstance(position_number, bool) or not isinstance(position_number, int) or position_number != 0:
        raise _AdapterFailure(DiagnosticCode.WRONG_POSITION, "scores are not from position zero")
    return position, choice, finish, inherited_domain


def _parse_candidates(
    position: Mapping[str, object],
    inherited_domain: ScoreDomain | None = None,
) -> tuple[ScoreDomain, tuple[ScoreEntry, ...], ScoreEntry]:
    domain, score_key, top_key = _position_domain(position, inherited_domain)
    selected = _entry(position, domain=domain, score_key=score_key)
    raw_top = position.get(top_key, ())
    top_values = _as_sequence(raw_top, top_key)
    candidates: list[ScoreEntry] = [selected]
    seen_top_ids: set[int] = set()
    for raw in top_values:
        candidate = _entry(raw, domain=domain, score_key=score_key)
        if candidate.token_id in seen_top_ids:
            raise _AdapterFailure(DiagnosticCode.DUPLICATE_ID, f"top score list repeats token ID {candidate.token_id}")
        seen_top_ids.add(candidate.token_id)
        if candidate.token_id == selected.token_id:
            if candidate != selected:
                raise _AdapterFailure(DiagnosticCode.DUPLICATE_ID, f"selected token ID {candidate.token_id} conflicts with top score")
            # Providers commonly repeat the selected token in top-k.  It is one
            # wire fact, not two masses; only this exact duplicate is merged.
            continue
        candidates.append(candidate)
    return domain, tuple(candidates), selected


def _diagnostics(
    *,
    kind: TransportKind,
    usage: UsageInfo,
    response: object,
    retain_diagnostics: bool,
    diagnostic_bytes: int,
    codes: Sequence[str] = (),
    message: str | None = None,
    missing: Sequence[TokenBinding] = (),
    generated: ScoreEntry | None = None,
    domain: ScoreDomain | None = None,
    finish: str | None = None,
) -> TransportDiagnostics:
    raw, truncated = _bounded_envelope(response, diagnostic_bytes) if retain_diagnostics else (None, False)
    return TransportDiagnostics(
        codes=tuple(dict.fromkeys(codes)),
        message=message,
        missing_ids=tuple(item.token_id for item in missing),
        missing_semantic_ids=tuple(item.semantic_id for item in missing),
        generated_token_id=generated.token_id if generated is not None else None,
        generated_token=generated.token if generated is not None else None,
        domain=domain,
        transport=kind,
        finish_reason=finish,
        usage_status=usage.status,
        raw_envelope=raw,
        raw_envelope_truncated=truncated,
    )


class ScoreTransportAdapter:
    """Parse one recorded response for one request; no retry ladder exists."""

    def __init__(self, *, retain_diagnostics: bool = False, diagnostic_bytes: int = DEFAULT_DIAGNOSTIC_BYTES) -> None:
        if (
            isinstance(diagnostic_bytes, bool)
            or not isinstance(diagnostic_bytes, int)
            or diagnostic_bytes < 0
            or diagnostic_bytes > DEFAULT_DIAGNOSTIC_BYTES
        ):
            raise ProtocolError(f"diagnostic_bytes must be between zero and {DEFAULT_DIAGNOSTIC_BYTES}")
        self.retain_diagnostics = retain_diagnostics
        self.diagnostic_bytes = diagnostic_bytes

    def adapt_chat(self, request: ScoreRequest, response: object, *, error: BaseException | None = None) -> TransportOutcome:
        return self.adapt(request, response, kind=TransportKind.CHAT, error=error)

    def adapt_native(self, request: ScoreRequest, response: object, *, error: BaseException | None = None) -> TransportOutcome:
        return self.adapt(request, response, kind=TransportKind.NATIVE_N_PROBS, error=error)

    def adapt(
        self,
        request: ScoreRequest,
        response: object,
        *,
        kind: TransportKind,
        error: BaseException | None = None,
    ) -> TransportOutcome:
        if not isinstance(request, ScoreRequest):
            raise ProtocolError("request must be ScoreRequest")
        if not isinstance(kind, TransportKind):
            raise ProtocolError("kind must be TransportKind")
        usage = UsageInfo.from_mapping(response.get("usage") if isinstance(response, Mapping) else None)
        if error is not None:
            usage_codes = (DiagnosticCode.USAGE_UNKNOWN.value,) if usage.status == UsageStatus.UNKNOWN else ()
            diagnostics = _diagnostics(
                kind=kind,
                usage=usage,
                response=response,
                retain_diagnostics=self.retain_diagnostics,
                diagnostic_bytes=self.diagnostic_bytes,
                codes=(DiagnosticCode.TRANSPORT_ERROR.value,) + usage_codes,
                message=_bounded_message(f"{type(error).__name__}: transport did not return a score envelope"),
            )
            return TransportOutcome(OutcomeStatus.UNAVAILABLE, None, None, usage, diagnostics)
        if not isinstance(response, Mapping):
            usage_codes = (DiagnosticCode.USAGE_UNKNOWN.value,) if usage.status == UsageStatus.UNKNOWN else ()
            diagnostics = _diagnostics(
                kind=kind,
                usage=usage,
                response=response,
                retain_diagnostics=self.retain_diagnostics,
                diagnostic_bytes=self.diagnostic_bytes,
                codes=(DiagnosticCode.EMPTY_RESPONSE.value,) + usage_codes,
                message="response envelope is absent or not an object",
            )
            return TransportOutcome(OutcomeStatus.UNAVAILABLE, None, None, usage, diagnostics)

        try:
            position, choice, finish, inherited_domain = _extract_position(response, kind)
            domain, candidates, generated = _parse_candidates(position, inherited_domain)
            if kind == TransportKind.CHAT:
                message = choice.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str) and content != generated.token:
                        raise _AdapterFailure(
                            DiagnosticCode.TOKEN_ID_MISMATCH,
                            "message content does not match the generated score token",
                        )
            if finish == "length":
                generated_label = next(
                    (label for label in request.labels if label.token_id == generated.token_id and label.bytes == generated.bytes and label.label == generated.token),
                    None,
                )
                if generated_label is None:
                    raise _AdapterFailure(
                        DiagnosticCode.LENGTH_WITHOUT_COMPLETE_LABEL,
                        "finish_reason=length is accepted only for a complete permitted label",
                    )
            vector = admit_complete_vector(
                request.labels,
                candidates,
                domain=domain,
                source=kind.value,
            )
            normalized = normalize_score_vector(vector)
        except IncompleteScoresError as exc:
            codes = [DiagnosticCode.SCORES_INCOMPLETE.value, DiagnosticCode.MISSING_LABEL_IDS.value]
            diagnostics = _diagnostics(
                kind=kind,
                usage=usage,
                response=response,
                retain_diagnostics=self.retain_diagnostics,
                diagnostic_bytes=self.diagnostic_bytes,
                codes=codes + ([DiagnosticCode.USAGE_UNKNOWN.value] if usage.status == UsageStatus.UNKNOWN else []),
                message=str(exc),
                missing=exc.missing,
                generated=locals().get("generated"),
                domain=locals().get("domain"),
                finish=locals().get("finish"),
            )
            return TransportOutcome(OutcomeStatus.SCORES_INCOMPLETE, None, None, usage, diagnostics)
        except SubsetNormalizationError as exc:
            diagnostics = _diagnostics(
                kind=kind,
                usage=usage,
                response=response,
                retain_diagnostics=self.retain_diagnostics,
                diagnostic_bytes=self.diagnostic_bytes,
                codes=(DiagnosticCode.SUBSET_SOFTMAX_FORBIDDEN.value,) + ((DiagnosticCode.USAGE_UNKNOWN.value,) if usage.status == UsageStatus.UNKNOWN else ()),
                message=str(exc),
                generated=locals().get("generated"),
                domain=locals().get("domain"),
                finish=locals().get("finish"),
            )
            return TransportOutcome(OutcomeStatus.UNAVAILABLE, None, None, usage, diagnostics)
        except _AdapterFailure as exc:
            diagnostics = _diagnostics(
                kind=kind,
                usage=usage,
                response=response,
                retain_diagnostics=self.retain_diagnostics,
                diagnostic_bytes=self.diagnostic_bytes,
                codes=(exc.code.value,) + ((DiagnosticCode.USAGE_UNKNOWN.value,) if usage.status == UsageStatus.UNKNOWN else ()),
                message=_bounded_message(exc.message),
                missing=exc.missing,
                generated=locals().get("generated"),
                domain=locals().get("domain"),
                finish=locals().get("finish"),
            )
            return TransportOutcome(OutcomeStatus.UNAVAILABLE, None, None, usage, diagnostics)
        except LegalMassError as exc:
            diagnostics = _diagnostics(
                kind=kind,
                usage=usage,
                response=response,
                retain_diagnostics=self.retain_diagnostics,
                diagnostic_bytes=self.diagnostic_bytes,
                codes=(DiagnosticCode.LEGAL_MASS_INVALID.value,)
                + ((DiagnosticCode.USAGE_UNKNOWN.value,) if usage.status == UsageStatus.UNKNOWN else ()),
                message=_bounded_message(str(exc)),
                generated=locals().get("generated"),
                domain=locals().get("domain"),
                finish=locals().get("finish"),
            )
            return TransportOutcome(OutcomeStatus.UNAVAILABLE, None, None, usage, diagnostics)
        except (DuplicateScoreError, TokenIdentityError, NormalizationError, ScoreValidationError) as exc:
            if isinstance(exc, DuplicateScoreError):
                code = DiagnosticCode.DUPLICATE_ID
            elif isinstance(exc, TokenIdentityError):
                code = DiagnosticCode.TOKEN_ID_MISMATCH
            else:
                code = DiagnosticCode.INVALID_SCORE
            diagnostics = _diagnostics(
                kind=kind,
                usage=usage,
                response=response,
                retain_diagnostics=self.retain_diagnostics,
                diagnostic_bytes=self.diagnostic_bytes,
                codes=(code.value,) + ((DiagnosticCode.USAGE_UNKNOWN.value,) if usage.status == UsageStatus.UNKNOWN else ()),
                message=_bounded_message(str(exc)),
                generated=locals().get("generated"),
                domain=locals().get("domain"),
                finish=locals().get("finish"),
            )
            return TransportOutcome(OutcomeStatus.UNAVAILABLE, None, None, usage, diagnostics)

        codes = (DiagnosticCode.USAGE_UNKNOWN.value,) if usage.status == UsageStatus.UNKNOWN else ()
        diagnostics = _diagnostics(
            kind=kind,
            usage=usage,
            response=response,
            retain_diagnostics=self.retain_diagnostics,
            diagnostic_bytes=self.diagnostic_bytes,
            codes=codes,
            message=None,
            generated=generated,
            domain=domain,
            finish=finish,
        )
        return TransportOutcome(OutcomeStatus.ADMITTED, vector, normalized, usage, diagnostics)


def parse_chat_response(
    request: ScoreRequest,
    response: object,
    *,
    retain_diagnostics: bool = False,
    error: BaseException | None = None,
) -> TransportOutcome:
    return ScoreTransportAdapter(retain_diagnostics=retain_diagnostics).adapt_chat(request, response, error=error)


def parse_native_response(
    request: ScoreRequest,
    response: object,
    *,
    retain_diagnostics: bool = False,
    error: BaseException | None = None,
) -> TransportOutcome:
    return ScoreTransportAdapter(retain_diagnostics=retain_diagnostics).adapt_native(request, response, error=error)


__all__ = [
    "ScoreTransportAdapter",
    "TransportOutcome",
    "build_chat_request",
    "build_native_request",
    "parse_chat_response",
    "parse_native_response",
]
