"""Strict, provider-neutral score admission and normalization.

Only a complete same-position vector for the host-qualified labels can become
a :class:`ScoreVector`.  Unknown top-k tokens remain useful for diagnostics but
are never treated as zero and never enter a subset softmax.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .protocol import ProtocolError, ScoreDomain, TokenBinding


_SENTINEL_VALUES = frozenset({-999.0, -9999.0, -1_000_000_000.0, 999.0, 9999.0, 1_000_000_000.0})
_PROBABILITY_TOLERANCE = 1e-6


class ScoreValidationError(ProtocolError):
    """A score entry violates its declared wire contract."""


class DuplicateScoreError(ScoreValidationError):
    """The response contains two distinct rows for one token ID."""


class TokenIdentityError(ScoreValidationError):
    """A token ID was paired with bytes/text different from qualification."""


class IncompleteScoresError(ScoreValidationError):
    """At least one requested label is absent from the provider vector."""

    def __init__(self, missing: Sequence[TokenBinding]):
        self.missing = tuple(missing)
        super().__init__("missing requested token IDs: " + ", ".join(str(item.token_id) for item in self.missing))


class NormalizationError(ScoreValidationError):
    """The admitted vector cannot produce the requested probability semantics."""


class LegalMassError(NormalizationError):
    """A pre-sampler logprob vector reports impossible legal mass."""


class SubsetNormalizationError(NormalizationError):
    """Post-sampling values do not constitute a complete probability domain."""


def _normal_domain(value: ScoreDomain | str) -> ScoreDomain:
    try:
        return value if isinstance(value, ScoreDomain) else ScoreDomain(value)
    except ValueError as exc:
        raise ScoreValidationError(f"unknown score domain: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ScoreEntry:
    """One finite provider score with exact token identity and position."""

    token_id: int
    token: str
    bytes: tuple[int, ...]
    value: float
    domain: ScoreDomain
    position: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.token_id, bool) or not isinstance(self.token_id, int) or self.token_id < 0:
            raise ScoreValidationError("score token_id must be a non-negative integer")
        if not isinstance(self.token, str) or not self.token:
            raise ScoreValidationError("score token text is required")
        try:
            token_bytes = tuple(self.bytes)
        except TypeError as exc:
            raise ScoreValidationError("score bytes must be an integer sequence") from exc
        if not token_bytes or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
            for item in token_bytes
        ):
            raise ScoreValidationError("score bytes must be non-empty octets")
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position != 0:
            raise ScoreValidationError("scores must come from answer position zero")
        domain = _normal_domain(self.domain)
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ScoreValidationError("score value must be numeric")
        value = float(self.value)
        if not math.isfinite(value) or value in _SENTINEL_VALUES or abs(value) >= 1e30:
            raise ScoreValidationError("score value is non-finite or a sentinel")
        if domain in (ScoreDomain.POST_SAMPLING_PROB, ScoreDomain.POST_SAMPLING_PROBABILITY) and not 0 <= value <= 1:
            raise ScoreValidationError("post-sampling probability must be between zero and one")
        object.__setattr__(self, "bytes", token_bytes)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "domain", domain)


@dataclass(frozen=True, slots=True)
class ScoreVector:
    """A complete vector ordered by the host's qualified labels."""

    labels: tuple[TokenBinding, ...]
    entries: tuple[ScoreEntry, ...]
    domain: ScoreDomain
    source: str

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        entries = tuple(self.entries)
        domain = _normal_domain(self.domain)
        if len(labels) != len(entries) or not labels:
            raise ScoreValidationError("score vector labels and entries must have equal non-zero length")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ScoreValidationError("score vector source is required")
        if any(not isinstance(label, TokenBinding) for label in labels):
            raise ScoreValidationError("score vector labels must be TokenBinding values")
        if any(not isinstance(entry, ScoreEntry) for entry in entries):
            raise ScoreValidationError("score vector entries must be ScoreEntry values")
        semantic_ids = [label.semantic_id for label in labels]
        token_ids = [label.token_id for label in labels]
        byte_spellings = [label.bytes for label in labels]
        if len(set(semantic_ids)) != len(labels):
            raise DuplicateScoreError("score vector contains duplicate semantic IDs")
        if len(set(token_ids)) != len(labels):
            raise DuplicateScoreError("score vector contains duplicate token IDs")
        if len(set(byte_spellings)) != len(labels):
            raise DuplicateScoreError("score vector contains duplicate token byte spellings")
        if len(set(entry.token_id for entry in entries)) != len(entries):
            raise DuplicateScoreError("score vector contains duplicate entry token IDs")
        for label, entry in zip(labels, entries):
            if entry.domain != domain:
                raise ScoreValidationError("score vector contains mixed domains")
            if entry.token_id != label.token_id or entry.token != label.label or entry.bytes != label.bytes:
                raise TokenIdentityError(f"score does not exactly match qualified label {label.semantic_id!r}")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "domain", domain)

    @property
    def raw_scores(self) -> tuple[float, ...]:
        return tuple(entry.value for entry in self.entries)

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(entry.token_id for entry in self.entries)

    @classmethod
    def from_candidates(
        cls,
        labels: Sequence[TokenBinding],
        candidates: Iterable[ScoreEntry],
        *,
        domain: ScoreDomain,
        source: str,
    ) -> "ScoreVector":
        qualified = tuple(labels)
        expected_by_id = {label.token_id: label for label in qualified}
        by_id: dict[int, ScoreEntry] = {}
        for candidate in candidates:
            if candidate.domain != domain:
                raise ScoreValidationError("candidate score domain does not match declared domain")
            if candidate.token_id in by_id:
                raise DuplicateScoreError(f"duplicate score token ID {candidate.token_id}")
            label = expected_by_id.get(candidate.token_id)
            if label is not None and (
                candidate.token != label.label or candidate.bytes != label.bytes
            ):
                raise TokenIdentityError(f"token ID {label.token_id} has an unqualified spelling")
            by_id[candidate.token_id] = candidate
        missing = tuple(label for label in qualified if label.token_id not in by_id)
        if missing:
            raise IncompleteScoresError(missing)
        return cls(
            labels=qualified,
            entries=tuple(by_id[label.token_id] for label in qualified),
            domain=domain,
            source=source,
        )


@dataclass(frozen=True, slots=True)
class NormalizedScoreVector:
    """Stable probabilities plus legal mass in the provider's declared domain."""

    vector: ScoreVector
    probabilities: tuple[float, ...]
    legal_mass: float | None

    def __post_init__(self) -> None:
        probabilities = tuple(float(value) for value in self.probabilities)
        if len(probabilities) != len(self.vector.entries):
            raise NormalizationError("probability/vector lengths differ")
        if any(not math.isfinite(value) or value < 0 for value in probabilities):
            raise NormalizationError("normalized probabilities must be finite and non-negative")
        if self.legal_mass is not None and (
            not math.isfinite(float(self.legal_mass)) or float(self.legal_mass) < 0
        ):
            raise NormalizationError("legal mass must be finite and non-negative")
        probability_sum = math.fsum(probabilities)
        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=_PROBABILITY_TOLERANCE,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise NormalizationError("normalized probabilities must sum to one")
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def by_semantic_id(self) -> dict[str, float]:
        return {label.semantic_id: probability for label, probability in zip(self.vector.labels, self.probabilities)}


def admit_complete_vector(
    labels: Sequence[TokenBinding],
    candidates: Iterable[ScoreEntry],
    *,
    domain: ScoreDomain,
    source: str,
) -> ScoreVector:
    """Admit exactly one complete, same-position vector; never fill a hole."""

    return ScoreVector.from_candidates(labels, candidates, domain=domain, source=source)


def normalize_score_vector(vector: ScoreVector) -> NormalizedScoreVector:
    """Normalize without subset softmax and without losing legal-mass evidence."""

    values = vector.raw_scores
    if vector.domain in (ScoreDomain.POST_SAMPLING_PROB, ScoreDomain.POST_SAMPLING_PROBABILITY):
        legal_mass = math.fsum(values)
        if not math.isclose(legal_mass, 1.0, rel_tol=_PROBABILITY_TOLERANCE, abs_tol=_PROBABILITY_TOLERANCE):
            raise SubsetNormalizationError(
                "post-sampling legal probabilities do not sum to one; subset softmax is forbidden"
            )
        # Preserve provider probabilities.  Dividing by legal_mass would hide
        # a constrained/partial domain and turn it into false confidence.
        return NormalizedScoreVector(vector, tuple(values), legal_mass)

    maximum = max(values)
    weights = tuple(math.exp(value - maximum) for value in values)
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise NormalizationError("raw score vector has no finite normalization")
    probabilities = tuple(weight / total for weight in weights)
    legal_mass: float | None = None
    if vector.domain == ScoreDomain.PRE_SAMPLER_LOGPROB:
        log_mass = maximum + math.log(total)
        try:
            legal_mass = math.exp(log_mass)
        except OverflowError as exc:
            raise LegalMassError("legal mass overflowed") from exc
        if not math.isfinite(legal_mass) or legal_mass > 1.0 + _PROBABILITY_TOLERANCE:
            raise LegalMassError("pre-sampler logprobs report impossible legal mass")
    return NormalizedScoreVector(vector, probabilities, legal_mass)


# Readable aliases for provider-neutral callers.
RawScore = ScoreEntry
normalize_scores = normalize_score_vector


__all__ = [
    "DuplicateScoreError",
    "IncompleteScoresError",
    "LegalMassError",
    "NormalizedScoreVector",
    "NormalizationError",
    "RawScore",
    "ScoreEntry",
    "ScoreValidationError",
    "ScoreVector",
    "SubsetNormalizationError",
    "TokenIdentityError",
    "admit_complete_vector",
    "normalize_score_vector",
    "normalize_scores",
]
