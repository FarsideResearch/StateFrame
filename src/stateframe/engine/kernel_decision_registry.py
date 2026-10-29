"""Declarative System One decision-site registry and journal adapters.

This module owns the contract for asking a typed decision question.  It does
not select candidates, call a model, calibrate probabilities, apply a result,
or infer an outcome label.  Those responsibilities stay with their respective
host boundaries.

The event helpers deliberately sanitize every caller-owned mapping at the
journal boundary.  In particular, a decision record may contain ``run_id`` or
``task_id`` as evidence metadata; those keys must never bind the journal's
reserved envelope fields.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

try:  # Supports the v46 flat-module loader and package imports.
    from kernel_continuation_journal import journal_safe_fields
except ImportError:  # pragma: no cover - exercised by package importers.
    from .kernel_continuation_journal import journal_safe_fields


SCHEMA = "stateframe-decision-registry/1"
DECISION_RECORDED = "decision_recorded"
DECISION_LABELLED = "decision_labelled"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:/-]{0,255}$")
_MAX_CHOICE_OPTIONS = 7  # six options plus an optional ABSTAIN option.
_MAX_SCORE_LEVELS = 5


class DecisionRegistryError(ValueError):
    """Base error for a site that cannot enter the decision contract."""


class DuplicateDecisionSiteError(DecisionRegistryError):
    """Raised when an id is reused with a different site definition."""


class UnknownDecisionSiteError(DecisionRegistryError):
    """Raised when a caller asks for an unregistered site."""


class DecisionEventError(DecisionRegistryError):
    """Raised when a decision journal payload is incomplete or malformed."""


class QuestionType(str, Enum):
    """The three generic question shapes supported by System One."""

    CHOICE = "CHOICE"
    SCORE = "SCORE"
    BOOLEAN = "BOOLEAN"

    def __str__(self) -> str:
        return self.value


def _text(value: object, label: str, *, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DecisionRegistryError(f"{label} must be a non-empty string")
    if len(value) > max_length or "\x00" in value:
        raise DecisionRegistryError(f"{label} is invalid or too long")
    return value.strip()


def _identifier(value: object, label: str) -> str:
    text = _text(value, label, max_length=256)
    if _IDENTIFIER_RE.fullmatch(text) is None:
        raise DecisionRegistryError(f"{label} must be an identifier")
    return text


def _question_type(value: object) -> QuestionType:
    if isinstance(value, QuestionType):
        return value
    if isinstance(value, str):
        try:
            return QuestionType(value.upper())
        except ValueError:
            pass
    allowed = ", ".join(member.value for member in QuestionType)
    raise DecisionRegistryError(f"question_type must be one of: {allowed}")


def _finite_probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionRegistryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise DecisionRegistryError(f"{label} must be between 0 and 1")
    return result


def _finite_cost(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionRegistryError(f"{label} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise DecisionRegistryError(f"{label} must be a finite non-negative number")
    return result


@dataclass(frozen=True, slots=True)
class OptionSpec:
    """One admissible CHOICE or BOOLEAN option and its answer token."""

    option_id: str
    label_token: str
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", _identifier(self.option_id, "option_id"))
        object.__setattr__(self, "label_token", _text(self.label_token, "label_token", max_length=128))
        object.__setattr__(self, "description", self.description.strip() if isinstance(self.description, str) else "")

    @property
    def id(self) -> str:
        """Compatibility alias for callers that use the contract's ``id`` wording."""
        return self.option_id

    def to_dict(self) -> dict[str, str]:
        return {"id": self.option_id, "label_token": self.label_token, "description": self.description}


@dataclass(frozen=True, slots=True)
class LevelSpec:
    """One ordered SCORE level and its answer token."""

    level: int
    label_token: str
    description: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 1:
            raise DecisionRegistryError("level must be a positive integer")
        object.__setattr__(self, "label_token", _text(self.label_token, "level.label_token", max_length=128))
        object.__setattr__(self, "description", self.description.strip() if isinstance(self.description, str) else "")

    def to_dict(self) -> dict[str, object]:
        return {"level": self.level, "label_token": self.label_token, "description": self.description}


@dataclass(frozen=True, slots=True)
class LabelSource:
    """Declarative owner of the later outcome label; no label is computed here."""

    source_id: str
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "label_source.source_id"))
        if not isinstance(self.description, str):
            raise DecisionRegistryError("label_source.description must be a string")
        object.__setattr__(self, "description", self.description.strip())

    def to_dict(self) -> dict[str, str]:
        return {"source_id": self.source_id, "description": self.description}


@dataclass(frozen=True, slots=True)
class CostPair:
    """Cost of taking the wrong action versus using the declared fallback."""

    wrong_action: float
    fallback: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "wrong_action", _finite_cost(self.wrong_action, "cost_pair.wrong_action"))
        object.__setattr__(self, "fallback", _finite_cost(self.fallback, "cost_pair.fallback"))

    @property
    def cost_of_wrong_action(self) -> float:
        return self.wrong_action

    @property
    def cost_of_fallback(self) -> float:
        return self.fallback

    def to_dict(self) -> dict[str, float]:
        return {"wrong_action": self.wrong_action, "fallback": self.fallback}


@dataclass(frozen=True, slots=True)
class AuthorityBoundary:
    """Who owns interpretation and application of a site's answer."""

    decision_owner: str
    application_owner: str = "host"
    advisory_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_owner", _text(self.decision_owner, "authority_boundary.decision_owner"))
        object.__setattr__(self, "application_owner", _text(self.application_owner, "authority_boundary.application_owner"))
        if not isinstance(self.advisory_only, bool):
            raise DecisionRegistryError("authority_boundary.advisory_only must be boolean")

    @property
    def owner(self) -> str:
        return self.decision_owner

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_owner": self.decision_owner,
            "application_owner": self.application_owner,
            "advisory_only": self.advisory_only,
        }


def _option(value: object, index: int) -> OptionSpec:
    if isinstance(value, OptionSpec):
        return value
    if isinstance(value, str):
        return OptionSpec(value, value)
    if isinstance(value, Mapping):
        option_id = value.get("id", value.get("option_id"))
        label_token = value.get("label_token", value.get("token", option_id))
        return OptionSpec(
            _identifier(option_id, f"options[{index}].id"),
            _text(label_token, f"options[{index}].label_token", max_length=128),
            value.get("description", ""),
        )
    raise DecisionRegistryError(f"options[{index}] must be an OptionSpec, id string, or mapping")


def _level(value: object, index: int) -> LevelSpec:
    if isinstance(value, LevelSpec):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return LevelSpec(value, str(value))
    if isinstance(value, str):
        return LevelSpec(index + 1, value)
    if isinstance(value, Mapping):
        level = value.get("level", index + 1)
        label_token = value.get("label_token", value.get("token", str(level)))
        return LevelSpec(level, _text(label_token, f"levels[{index}].label_token", max_length=128), value.get("description", ""))
    raise DecisionRegistryError(f"levels[{index}] must be a LevelSpec, integer, string, or mapping")


def _label_source(value: object) -> LabelSource:
    if isinstance(value, LabelSource):
        return value
    if isinstance(value, str):
        return LabelSource(value)
    if isinstance(value, Mapping):
        return LabelSource(value.get("source_id", value.get("id")), value.get("description", ""))
    if callable(value):
        name = getattr(value, "__qualname__", getattr(value, "__name__", "label_source"))
        name = "label_source_" + re.sub(r"[^A-Za-z0-9._:/-]", "_", str(name))
        return LabelSource(_identifier(name, "label_source"))
    raise DecisionRegistryError("label_source is required and must identify a label owner")


def _cost_pair(value: object) -> CostPair:
    if isinstance(value, CostPair):
        return value
    if isinstance(value, Mapping):
        wrong = value.get("wrong_action", value.get("cost_wrong_action"))
        fallback = value.get("fallback", value.get("cost_fallback"))
        return CostPair(wrong, fallback)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 2:
        return CostPair(value[0], value[1])
    raise DecisionRegistryError("cost_pair must contain wrong_action and fallback costs")


def _authority(value: object) -> AuthorityBoundary:
    if isinstance(value, AuthorityBoundary):
        return value
    if isinstance(value, str):
        return AuthorityBoundary(value)
    if isinstance(value, Mapping):
        owner = value.get("decision_owner", value.get("owner"))
        application = value.get("application_owner", value.get("application", "host"))
        advisory = value.get("advisory_only", True)
        return AuthorityBoundary(owner, application, advisory)
    raise DecisionRegistryError("authority_boundary is required")


def _named_callable(value: object, label: str) -> object:
    if callable(value):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise DecisionRegistryError(f"{label} must be a callable or non-empty identifier")


def _callable_name(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if callable(value):
        return str(getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__)))
    return None


@dataclass(frozen=True, slots=True)
class DecisionSite:
    """Complete declarative contract for one generic decision site."""

    site_id: str
    question_type: QuestionType | str
    options: Sequence[OptionSpec | str | Mapping[str, object]] = ()
    levels: Sequence[LevelSpec | int | str | Mapping[str, object]] = ()
    label_source: LabelSource | str | Mapping[str, object] | Callable[..., object] | None = None
    cost_pair: CostPair | Sequence[object] | Mapping[str, object] | None = None
    authority_boundary: AuthorityBoundary | str | Mapping[str, object] | None = None
    option_builder: Callable[..., object] | str | None = None
    code_filters: Sequence[Callable[..., object] | str] = ()
    fallback: Callable[..., object] | str = "human_review"
    question_version: str = "1"
    candidate_top_k: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "site_id", _identifier(self.site_id, "site_id"))
        question_type = _question_type(self.question_type)
        object.__setattr__(self, "question_type", question_type)

        # Validate this first so an incomplete site fails for the missing label
        # source, rather than exposing a secondary option/cost error.
        object.__setattr__(self, "label_source", _label_source(self.label_source))
        object.__setattr__(self, "cost_pair", _cost_pair(self.cost_pair))
        object.__setattr__(self, "authority_boundary", _authority(self.authority_boundary))
        object.__setattr__(self, "question_version", _text(self.question_version, "question_version", max_length=128))
        if type(self.candidate_top_k) is not int or not 1 <= self.candidate_top_k <= 64:
            raise DecisionRegistryError("candidate_top_k must be an integer within 1..64")

        raw_options = tuple(self.options) if not isinstance(self.options, (str, bytes, bytearray)) else (self.options,)
        raw_levels = tuple(self.levels) if not isinstance(self.levels, (str, bytes, bytearray)) else (self.levels,)
        if question_type is QuestionType.BOOLEAN and not raw_options:
            raw_options = ("yes", "no")
        if question_type in {QuestionType.CHOICE, QuestionType.BOOLEAN}:
            if raw_levels:
                raise DecisionRegistryError(f"{question_type.value} sites cannot declare SCORE levels")
            if question_type is QuestionType.BOOLEAN and len(raw_options) != 2:
                raise DecisionRegistryError("boolean sites require exactly 2 options")
            if question_type is QuestionType.CHOICE and (len(raw_options) < 2 or len(raw_options) > _MAX_CHOICE_OPTIONS):
                raise DecisionRegistryError("choice sites require 2 to 7 options")
            options = tuple(_option(value, index) for index, value in enumerate(raw_options))
            if len({item.option_id for item in options}) != len(options):
                raise DecisionRegistryError("options must have unique ids")
            if len(options) == _MAX_CHOICE_OPTIONS and not any(
                    item.option_id.upper() == "ABSTAIN" or item.label_token.upper() == "ABSTAIN"
                    for item in options):
                raise DecisionRegistryError("the seventh choice option must be ABSTAIN")
            object.__setattr__(self, "options", options)
            object.__setattr__(self, "levels", ())
        else:
            if raw_options:
                raise DecisionRegistryError("SCORE sites cannot declare CHOICE options")
            if len(raw_levels) < 2 or len(raw_levels) > _MAX_SCORE_LEVELS:
                raise DecisionRegistryError("score sites require 2 to 5 levels")
            levels = tuple(_level(value, index) for index, value in enumerate(raw_levels))
            if tuple(item.level for item in levels) != tuple(range(1, len(levels) + 1)):
                raise DecisionRegistryError("score levels must be ordered consecutively from 1")
            object.__setattr__(self, "levels", levels)
            object.__setattr__(self, "options", ())

        if self.option_builder is not None:
            object.__setattr__(self, "option_builder", _named_callable(self.option_builder, "option_builder"))
        filters = tuple(self.code_filters)
        object.__setattr__(self, "code_filters", tuple(_named_callable(value, "code_filters[]") for value in filters))
        object.__setattr__(self, "fallback", _named_callable(self.fallback, "fallback"))

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(option.option_id for option in self.options)

    @property
    def type(self) -> QuestionType:
        """Compatibility alias for the journal record's ``type`` field."""
        return self.question_type

    @property
    def level_values(self) -> tuple[int, ...]:
        return tuple(level.level for level in self.levels)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "site_id": self.site_id,
            "question_type": self.question_type.value,
            "question_version": self.question_version,
            "options": [option.to_dict() for option in self.options],
            "levels": [level.to_dict() for level in self.levels],
            "label_source": self.label_source.to_dict(),
            "cost_pair": self.cost_pair.to_dict(),
            "authority_boundary": self.authority_boundary.to_dict(),
            "option_builder": _callable_name(self.option_builder),
            "code_filters": [_callable_name(value) for value in self.code_filters],
            "fallback": _callable_name(self.fallback),
            "candidate_top_k": self.candidate_top_k,
        }


class DecisionSiteRegistry:
    """In-memory, immutable-definition registry with idempotent registration."""

    def __init__(self, sites: Sequence[DecisionSite] = ()) -> None:
        self._sites: dict[str, DecisionSite] = {}
        for site in sites:
            self.register(site)

    def register(self, site: DecisionSite) -> DecisionSite:
        if not isinstance(site, DecisionSite):
            raise DecisionRegistryError("registry accepts DecisionSite instances only")
        previous = self._sites.get(site.site_id)
        if previous is not None and previous != site:
            raise DuplicateDecisionSiteError(f"site id reused with a different definition: {site.site_id}")
        self._sites[site.site_id] = site
        return site

    register_site = register

    def get(self, site_id: str) -> DecisionSite | None:
        return self._sites.get(site_id)

    def require(self, site_id: str) -> DecisionSite:
        try:
            return self._sites[site_id]
        except KeyError as error:
            raise UnknownDecisionSiteError(f"unregistered decision site: {site_id}") from error

    def all(self) -> tuple[DecisionSite, ...]:
        return tuple(self._sites.values())

    def __contains__(self, site_id: object) -> bool:
        return site_id in self._sites

    def __len__(self) -> int:
        return len(self._sites)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Typed journal payload for one scorer observation.

    The scorer may populate optional evidence/probability fields later.  This
    contract records them without interpreting or evaluating their contents.
    """

    decision_id: str
    site_id: str
    question_version: str
    question_type: QuestionType | str
    options: Sequence[Mapping[str, object]] = ()
    evidence_handles: Sequence[Mapping[str, object]] = ()
    served_model_identity: Mapping[str, object] | str | None = None
    server_build: str | None = None
    raw_top_logprobs: Mapping[str, object] = field(default_factory=dict)
    legal_mass: float | None = None
    raw_p: Mapping[str, object] = field(default_factory=dict)
    calibrated_p: Mapping[str, object] = field(default_factory=dict)
    temperature: float | None = None
    threshold: float | None = None
    disposition: str = "UNCALIBRATED_PROPOSE"
    applied: bool = False
    cost: Mapping[str, object] = field(default_factory=dict)
    run_id: str | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "site_id", _identifier(self.site_id, "site_id"))
        object.__setattr__(self, "question_version", _text(self.question_version, "question_version", max_length=128))
        object.__setattr__(self, "question_type", _question_type(self.question_type))
        if not isinstance(self.applied, bool):
            raise DecisionEventError("applied must be boolean")
        object.__setattr__(self, "disposition", _text(self.disposition, "disposition", max_length=128))
        if self.legal_mass is not None:
            object.__setattr__(self, "legal_mass", _finite_probability(self.legal_mass, "legal_mass"))
        if self.temperature is not None:
            object.__setattr__(self, "temperature", _finite_cost(self.temperature, "temperature"))
        if self.threshold is not None:
            object.__setattr__(self, "threshold", _finite_probability(self.threshold, "threshold"))
        if self.run_id is not None:
            object.__setattr__(self, "run_id", _text(self.run_id, "run_id", max_length=256))
        if self.step is not None and (isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0):
            raise DecisionEventError("step must be an integer >= 0")
        object.__setattr__(self, "options", tuple(dict(value) for value in self.options))
        object.__setattr__(self, "evidence_handles", tuple(dict(value) for value in self.evidence_handles))
        for label, value in (("raw_top_logprobs", self.raw_top_logprobs), ("raw_p", self.raw_p),
                             ("calibrated_p", self.calibrated_p), ("cost", self.cost)):
            if not isinstance(value, Mapping):
                raise DecisionEventError(f"{label} must be a mapping")
            object.__setattr__(self, label, dict(value))

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "site_id": self.site_id,
            "question_version": self.question_version,
            "type": self.question_type.value,
            "options": [dict(value) for value in self.options],
            "evidence_handles": [dict(value) for value in self.evidence_handles],
            "served_model_identity": self.served_model_identity,
            "server_build": self.server_build,
            "raw_top_logprobs": dict(self.raw_top_logprobs),
            "legal_mass": self.legal_mass,
            "raw_p": dict(self.raw_p),
            "calibrated_p": dict(self.calibrated_p),
            "T_site": self.temperature,
            "tau_site": self.threshold,
            "disposition": self.disposition,
            "applied": self.applied,
            "cost": dict(self.cost),
            "run_id": self.run_id,
            "step": self.step,
        }

    @property
    def type(self) -> QuestionType:
        return self.question_type


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """Later host/grader/human label for a previously recorded decision."""

    decision_id: str
    label: str
    label_source: LabelSource | str
    labelled_at: str | int | float

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "label", _text(self.label, "label", max_length=256))
        object.__setattr__(self, "label_source", _label_source(self.label_source))
        if isinstance(self.labelled_at, bool) or not isinstance(self.labelled_at, (str, int, float)):
            raise DecisionEventError("labelled_at must be a timestamp string or finite number")
        if isinstance(self.labelled_at, str) and not self.labelled_at.strip():
            raise DecisionEventError("labelled_at must not be empty")
        if isinstance(self.labelled_at, (int, float)) and not math.isfinite(float(self.labelled_at)):
            raise DecisionEventError("labelled_at must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "label": self.label,
            "label_source": self.label_source.source_id,
            "labelled_at": self.labelled_at,
        }


class JournalLike(Protocol):
    def emit(self, kind: str, **fields: object) -> dict[str, object]: ...


def _payload(value: DecisionRecord | DecisionOutcome | Mapping[str, object], required: tuple[str, ...]) -> dict[str, object]:
    if isinstance(value, (DecisionRecord, DecisionOutcome)):
        result = value.to_dict()
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        raise DecisionEventError("decision event must be a DecisionRecord, DecisionOutcome, or mapping")
    missing = [key for key in required if key not in result]
    if missing:
        raise DecisionEventError(f"decision event missing required fields: {', '.join(missing)}")
    return result


def _emit_decision_event(journal: JournalLike, kind: str, payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(kind, str) or kind not in {DECISION_RECORDED, DECISION_LABELLED}:
        raise DecisionEventError("unsupported decision journal event kind")
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise DecisionEventError("journal must provide an emit(kind, **fields) method")
    return journal.emit(kind, **journal_safe_fields(payload))


def emit_decision_recorded(
    journal: JournalLike,
    record: DecisionRecord | Mapping[str, object],
) -> dict[str, object]:
    """Append one ``decision_recorded`` event through the safe-field boundary."""
    return _emit_decision_event(
        journal,
        DECISION_RECORDED,
        _payload(record, ("decision_id", "site_id", "question_version", "type")),
    )


def emit_decision_labelled(
    journal: JournalLike,
    outcome: DecisionOutcome | Mapping[str, object],
) -> dict[str, object]:
    """Append one ``decision_labelled`` event through the safe-field boundary."""
    return _emit_decision_event(
        journal,
        DECISION_LABELLED,
        _payload(outcome, ("decision_id", "label", "label_source", "labelled_at")),
    )


record_decision = emit_decision_recorded
label_decision = emit_decision_labelled


__all__ = [
    "AuthorityBoundary", "CostPair", "DECISION_LABELLED", "DECISION_RECORDED",
    "DecisionEventError", "DecisionOutcome", "DecisionRecord", "DecisionRegistryError",
    "DecisionSite", "DecisionSiteRegistry", "DuplicateDecisionSiteError", "LabelSource",
    "LevelSpec", "OptionSpec", "QuestionType", "SCHEMA", "UnknownDecisionSiteError",
    "emit_decision_labelled", "emit_decision_recorded", "label_decision", "record_decision",
]
