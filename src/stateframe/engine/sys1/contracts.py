"""Immutable, provider-neutral contracts for System One decision sites.

This module describes what may be asked and how a later outcome may be labelled.
It does not call a model, evaluate code filters, read receipts, grant permission,
cross a terminal gate, or execute an effect.  Those decisions remain with the
existing API, policy, receipt, and owner lanes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


SCHEMA_ID = "sys1.decision-site"
SCHEMA_VERSION = "1"
CONTRACT_SCHEMA = f"{SCHEMA_ID}/{SCHEMA_VERSION}"
DECISION_RECORDED = "decision_recorded"
DECISION_LABELLED = "decision_labelled"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_UNKNOWN_WORDS = frozenset({"unknown", "insufficient", "undetermined"})
_NON_BOOLEAN_WORDS = _UNKNOWN_WORDS | frozenset({"abstain"})


class ContractError(ValueError):
    """Base error for a contract that cannot safely enter the registry."""


class AmbiguousOutcomeError(ContractError):
    """Raised when a set-valued outcome is incorrectly made one-hot."""


class QuestionType(str, Enum):
    CHOICE = "CHOICE"
    BOOLEAN = "BOOLEAN"
    SCORE = "SCORE"

    def __str__(self) -> str:
        return self.value


class Answerability(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT = "INSUFFICIENT"

    def __str__(self) -> str:
        return self.value


class ResolutionStatus(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    MODEL_REQUIRED = "MODEL_REQUIRED"


def _text(value: object, name: str, *, limit: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ContractError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > limit:
        raise ContractError(f"{name} exceeds {limit} characters")
    return result


def _identifier(value: object, name: str) -> str:
    result = _text(value, name, limit=256)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ContractError(f"{name} must be a stable identifier")
    return result


def _version(value: object, name: str) -> str:
    return _identifier(value, name)


def _question_type(value: object) -> QuestionType:
    if isinstance(value, QuestionType):
        return value
    if isinstance(value, str):
        try:
            return QuestionType(value.strip().upper())
        except ValueError:
            pass
    raise ContractError("question_type must be CHOICE, BOOLEAN, or SCORE")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be a finite number")
    return result


def _freeze(value: object, name: str = "mapping") -> object:
    """Recursively freeze JSON-shaped values used by public contracts."""
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{name} keys must be strings")
            frozen[key] = _freeze(item, f"{name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{name}[]") for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{name} must contain only finite numbers")
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise ContractError(f"{name} must contain JSON-shaped values")


def _metadata(value: Mapping[str, object] | None, name: str) -> Mapping[str, object]:
    """Validate and recursively freeze optional JSON metadata mappings."""
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a mapping")
    frozen = _freeze(value, name)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded above
        raise ContractError(f"{name} must be a mapping")
    return frozen


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _sequence(value: object, name: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ContractError(f"{name} must be a sequence")
    return tuple(value)


def _unique(values: Sequence[str], name: str, *, folded: bool = False) -> None:
    keys = [value.casefold() if folded else value for value in values]
    if len(set(keys)) != len(keys):
        raise ContractError(f"{name} must not contain duplicates")


def _coerce_label(value: object, name: str) -> str:
    return _text(value, name, limit=256)


_MISSING = object()


def _alias_value(
    value: Mapping[str, object],
    names: tuple[str, ...],
    name: str,
    default: object = _MISSING,
) -> object:
    """Read compatibility aliases without silently accepting contradictions."""
    found = [value[key] for key in names if key in value]
    if not found:
        if default is _MISSING:
            raise ContractError(f"{name} is required")
        return default
    first = found[0]
    if any(item != first for item in found[1:]):
        raise ContractError(f"{name} aliases contradict each other")
    return first


@dataclass(frozen=True, slots=True, init=False)
class OptionSpec:
    """One semantic option and its neutral answer label."""

    option_id: str
    label: str
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __init__(
        self,
        option_id: str,
        label: str | None = None,
        description: str = "",
        metadata: Mapping[str, object] | None = None,
        *,
        label_token: str | None = None,
        definition: str | None = None,
    ) -> None:
        if label is not None and label_token is not None and label != label_token:
            raise ContractError("option label and label_token contradict each other")
        if definition is not None:
            if description and description != definition:
                raise ContractError("option description and definition contradict each other")
            description = definition
        object.__setattr__(self, "option_id", _identifier(option_id, "option_id"))
        object.__setattr__(self, "label", _coerce_label(label if label is not None else label_token, "option.label"))
        if not isinstance(description, str):
            raise ContractError("option.description must be a string")
        object.__setattr__(self, "description", description.strip())
        object.__setattr__(self, "metadata", _metadata(metadata, "option.metadata"))

    @property
    def id(self) -> str:
        return self.option_id

    @property
    def label_token(self) -> str:
        """Compatibility name used by the scored API lane."""
        return self.label

    @property
    def definition(self) -> str:
        return self.description

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.option_id,
            "label": self.label,
            "label_token": self.label,
            "description": self.description,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True, init=False)
class LevelSpec:
    """One ordered SCORE level with an explicit finite numeric value."""

    level_id: str
    value: float
    label: str
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __init__(
        self,
        level_id: str | int | float | None = None,
        value: float | int | None = None,
        label: str | None = None,
        description: str = "",
        metadata: Mapping[str, object] | None = None,
        *,
        level: str | int | float | None = None,
        label_token: str | None = None,
        definition: str | None = None,
    ) -> None:
        if level_id is None:
            level_id = level
        elif level is not None and level_id != level:
            raise ContractError("level_id and level contradict each other")
        if level_id is None:
            raise ContractError("level_id is required")
        # Support the earlier positional shape LevelSpec(level, label_token, definition)
        # while retaining the explicit LevelSpec(level_id, value, label) form.
        if isinstance(level_id, (int, float)) and not isinstance(level_id, bool) and isinstance(value, str):
            legacy_label = value
            legacy_definition = label
            if legacy_definition is not None and not isinstance(legacy_definition, str):
                raise ContractError("legacy level definition must be a string")
            if legacy_definition is not None:
                if description and description != legacy_definition:
                    raise ContractError("level description and definition contradict each other")
                description = legacy_definition
            label = legacy_label
            value = level_id
        if value is None and isinstance(level_id, (int, float)) and not isinstance(level_id, bool):
            value = level_id
        if label is not None and label_token is not None and label != label_token:
            raise ContractError("level label and label_token contradict each other")
        if definition is not None:
            if description and description != definition:
                raise ContractError("level description and definition contradict each other")
            description = definition
        object.__setattr__(self, "level_id", _identifier(str(level_id), "level_id"))
        object.__setattr__(self, "value", _finite_number(value, "level.value"))
        object.__setattr__(self, "label", _coerce_label(label if label is not None else label_token, "level.label"))
        if not isinstance(description, str):
            raise ContractError("level.description must be a string")
        object.__setattr__(self, "description", description.strip())
        object.__setattr__(self, "metadata", _metadata(metadata, "level.metadata"))

    @property
    def id(self) -> str:
        return self.level_id

    @property
    def level(self) -> str:
        """Compatibility alias for callers that refer to the semantic level id."""
        return self.level_id

    @property
    def label_token(self) -> str:
        return self.label

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.level_id,
            "level": self.level_id,
            "value": self.value,
            "label": self.label,
            "label_token": self.label,
            "description": self.description,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CodeFilter:
    """Named host-side admissibility rule; the registry never executes it."""

    filter_id: str
    version: str = "1"
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "filter_id", _identifier(self.filter_id, "code_filter.filter_id"))
        object.__setattr__(self, "version", _version(self.version, "code_filter.version"))
        if not isinstance(self.description, str):
            raise ContractError("code_filter.description must be a string")
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "metadata", _metadata(self.metadata, "code_filter.metadata"))

    @property
    def id(self) -> str:
        return self.filter_id

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.filter_id,
            "version": self.version,
            "description": self.description,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class LabelSource:
    """Explicit owner/rubric for a later outcome label."""

    source_id: str
    version: str = "1"
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _identifier(self.source_id, "label_source.source_id"))
        object.__setattr__(self, "version", _version(self.version, "label_source.version"))
        if not isinstance(self.description, str):
            raise ContractError("label_source.description must be a string")
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "metadata", _metadata(self.metadata, "label_source.metadata"))

    @property
    def id(self) -> str:
        return self.source_id

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.source_id,
            "version": self.version,
            "description": self.description,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AcceptableSetPolicy:
    """Policy for labels that may name more than one acceptable action."""

    mode: str = "single"
    allow_multiple: bool = False
    tie_policy: str = "explicit_target"
    version: str = "1"

    def __post_init__(self) -> None:
        mode = _text(self.mode, "acceptable_set_policy.mode", limit=32).casefold()
        tie = _text(self.tie_policy, "acceptable_set_policy.tie_policy", limit=64).casefold()
        if mode not in {"single", "set"}:
            raise ContractError("acceptable_set_policy.mode must be single or set")
        if not isinstance(self.allow_multiple, bool):
            raise ContractError("acceptable_set_policy.allow_multiple must be boolean")
        if (mode == "single") == self.allow_multiple:
            raise ContractError("acceptable-set mode and allow_multiple contradict each other")
        if tie in {"one_hot", "argmax", "first", "arbitrary"}:
            raise ContractError("acceptable-set policy cannot choose an arbitrary one-hot target")
        if tie not in {"explicit_target", "preserve_set", "fallback", "reject"}:
            raise ContractError("acceptable_set_policy.tie_policy is unsupported")
        if not self.allow_multiple and tie == "preserve_set":
            raise ContractError("preserve_set requires allow_multiple")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "tie_policy", tie)
        object.__setattr__(self, "version", _version(self.version, "acceptable_set_policy.version"))

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "allow_multiple": self.allow_multiple,
            "tie_policy": self.tie_policy,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class AnswerabilityPolicy:
    """Allowed outcome-row states; unknown is never a Boolean false."""

    allow_unknown: bool = True
    allow_insufficient: bool = True
    version: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.allow_unknown, bool) or not isinstance(self.allow_insufficient, bool):
            raise ContractError("answerability policy flags must be boolean")
        object.__setattr__(self, "version", _version(self.version, "answerability_policy.version"))

    def allows(self, answerability: Answerability) -> bool:
        return {
            Answerability.ANSWERABLE: True,
            Answerability.UNKNOWN: self.allow_unknown,
            Answerability.INSUFFICIENT: self.allow_insufficient,
        }[answerability]

    def to_dict(self) -> dict[str, object]:
        return {
            "allow_unknown": self.allow_unknown,
            "allow_insufficient": self.allow_insufficient,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class LossSpec:
    """Finite, versioned policy inputs; not an action authorization."""

    wrong_action: float
    fallback: float
    unit: str = "loss"
    horizon: str = "site"
    version: str = "1"

    def __post_init__(self) -> None:
        wrong = _finite_number(self.wrong_action, "loss.wrong_action")
        fallback = _finite_number(self.fallback, "loss.fallback")
        if wrong < 0 or fallback < 0:
            raise ContractError("loss values must be non-negative")
        object.__setattr__(self, "wrong_action", wrong)
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "unit", _text(self.unit, "loss.unit", limit=128))
        object.__setattr__(self, "horizon", _text(self.horizon, "loss.horizon", limit=128))
        object.__setattr__(self, "version", _version(self.version, "loss.version"))

    @property
    def wrong_action_cost(self) -> float:
        return self.wrong_action

    @property
    def fallback_cost(self) -> float:
        return self.fallback

    def to_dict(self) -> dict[str, object]:
        return {
            "wrong_action": self.wrong_action,
            "fallback": self.fallback,
            "unit": self.unit,
            "horizon": self.horizon,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class FallbackBoundary:
    """Named fallback consumed by a host owner, never executed here."""

    fallback_id: str
    description: str = ""
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fallback_id", _identifier(self.fallback_id, "fallback.fallback_id"))
        if not isinstance(self.description, str):
            raise ContractError("fallback.description must be a string")
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "version", _version(self.version, "fallback.version"))

    @property
    def id(self) -> str:
        return self.fallback_id

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.fallback_id,
            "description": self.description,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class AuthorityBoundary:
    """Ownership metadata that deliberately stops before permission/effects."""

    proposal_owner: str
    consumer_owner: str = "host"
    advisory_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "proposal_owner", _identifier(self.proposal_owner, "authority.proposal_owner"))
        object.__setattr__(self, "consumer_owner", _identifier(self.consumer_owner, "authority.consumer_owner"))
        if self.advisory_only is not True:
            raise ContractError("System One registry entries must remain advisory_only")

    @property
    def decision_owner(self) -> str:
        return self.proposal_owner

    @property
    def application_owner(self) -> str:
        return self.consumer_owner

    def to_dict(self) -> dict[str, object]:
        return {
            "proposal_owner": self.proposal_owner,
            "consumer_owner": self.consumer_owner,
            "advisory_only": self.advisory_only,
        }


def _option(value: object, index: int) -> OptionSpec:
    if isinstance(value, OptionSpec):
        return value
    if isinstance(value, str):
        return OptionSpec(value, value)
    if isinstance(value, Mapping):
        option_id = _alias_value(value, ("option_id", "id"), f"options[{index}].id")
        label = _alias_value(value, ("label", "label_token", "token"), f"options[{index}].label")
        description = _alias_value(value, ("description", "definition"), f"options[{index}].description", "")
        return OptionSpec(
            _identifier(option_id, f"options[{index}].id"),
            _coerce_label(label, f"options[{index}].label"),
            description,
            value.get("metadata", {}),
        )
    raise ContractError(f"options[{index}] must be an OptionSpec, id string, or mapping")


def _level(value: object, index: int) -> LevelSpec:
    if isinstance(value, LevelSpec):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return LevelSpec(str(index + 1), value, str(index + 1))
    if isinstance(value, str):
        return LevelSpec(value, float(index + 1), value)
    if isinstance(value, Mapping):
        raw_id = _alias_value(value, ("level_id", "id", "level"), f"levels[{index}].id")
        raw_value = value.get("value", raw_id)
        raw_label = _alias_value(value, ("label", "label_token", "token"), f"levels[{index}].label", None)
        if raw_label is None:
            raw_label = str(raw_id)
        if raw_id is None:
            raise ContractError(f"levels[{index}].id is required")
        return LevelSpec(
            raw_id,
            raw_value,
            _coerce_label(raw_label, f"levels[{index}].label"),
            _alias_value(value, ("description", "definition"), f"levels[{index}].description", ""),
            value.get("metadata", {}),
        )
    raise ContractError(f"levels[{index}] must be a LevelSpec, number, string, or mapping")


def _code_filter(value: object, index: int) -> CodeFilter:
    if isinstance(value, CodeFilter):
        return value
    if isinstance(value, str):
        return CodeFilter(value)
    if isinstance(value, Mapping):
        return CodeFilter(
            _alias_value(value, ("filter_id", "id"), f"code_filters[{index}].id"),
            value.get("version", "1"),
            _alias_value(value, ("description", "definition"), f"code_filters[{index}].description", ""),
            value.get("metadata", {}),
        )
    raise ContractError(f"code_filters[{index}] must be a named declarative filter")


def _label_source(value: object) -> LabelSource:
    if isinstance(value, LabelSource):
        return value
    if isinstance(value, str):
        return LabelSource(value)
    if isinstance(value, Mapping):
        return LabelSource(
            _alias_value(value, ("source_id", "id"), "label_source.id"),
            value.get("version", "1"),
            _alias_value(value, ("description", "definition"), "label_source.description", ""),
            value.get("metadata", {}),
        )
    raise ContractError("label_source is required and must be explicit")


def _loss(value: object) -> LossSpec:
    if isinstance(value, LossSpec):
        return value
    if isinstance(value, Mapping):
        return LossSpec(
            _alias_value(value, ("wrong_action", "wrong_action_cost"), "loss.wrong_action"),
            _alias_value(value, ("fallback", "fallback_cost"), "loss.fallback"),
            value.get("unit", "loss"),
            value.get("horizon", "site"),
            value.get("version", "1"),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 2:
        return LossSpec(value[0], value[1])
    raise ContractError("loss_spec is required and must define wrong_action and fallback costs")


def _fallback(value: object) -> FallbackBoundary:
    if isinstance(value, FallbackBoundary):
        return value
    if isinstance(value, str):
        return FallbackBoundary(value)
    if isinstance(value, Mapping):
        return FallbackBoundary(
            _alias_value(value, ("fallback_id", "id"), "fallback.id"),
            value.get("description", ""),
            value.get("version", "1"),
        )
    raise ContractError("fallback must identify a host-owned boundary")


def _authority(value: object) -> AuthorityBoundary:
    if isinstance(value, AuthorityBoundary):
        return value
    if isinstance(value, str):
        return AuthorityBoundary(value)
    if isinstance(value, Mapping):
        return AuthorityBoundary(
            _alias_value(value, ("proposal_owner", "decision_owner", "owner"), "authority.proposal_owner"),
            _alias_value(value, ("consumer_owner", "application_owner"), "authority.consumer_owner", "host"),
            value.get("advisory_only", True),
        )
    raise ContractError("authority_boundary is required")


def _acceptable_policy(value: object) -> AcceptableSetPolicy:
    if value is None:
        return AcceptableSetPolicy()
    if isinstance(value, AcceptableSetPolicy):
        return value
    if isinstance(value, Mapping):
        raw_mode = value.get("mode", "single")
        return AcceptableSetPolicy(
            raw_mode,
            value.get("allow_multiple", isinstance(raw_mode, str) and raw_mode.casefold() == "set"),
            value.get("tie_policy", "explicit_target"),
            value.get("version", "1"),
        )
    raise ContractError("acceptable_set_policy must be declarative metadata")


def _answerability_policy(value: object) -> AnswerabilityPolicy:
    if value is None:
        return AnswerabilityPolicy()
    if isinstance(value, AnswerabilityPolicy):
        return value
    if isinstance(value, Mapping):
        return AnswerabilityPolicy(
            value.get("allow_unknown", True),
            value.get("allow_insufficient", True),
            value.get("version", "1"),
        )
    raise ContractError("answerability_policy must be declarative metadata")


@dataclass(frozen=True, slots=True, init=False)
class DecisionSite:
    """Complete versioned contract for one non-authoritative decision site."""

    site_id: str
    site_version: str
    question_id: str
    question_version: str
    question_type: QuestionType
    question: str
    schema_id: str
    schema_version: str
    options: tuple[OptionSpec, ...]
    levels: tuple[LevelSpec, ...]
    code_filters: tuple[CodeFilter, ...]
    label_source: LabelSource
    acceptable_set_policy: AcceptableSetPolicy
    answerability_policy: AnswerabilityPolicy
    loss_spec: LossSpec
    fallback: FallbackBoundary
    authority_boundary: AuthorityBoundary
    evidence_version: str
    procedure_version: str
    policy_version: str
    open_menu: bool
    menu_complete: bool
    menu_expansion_option_id: str | None
    deterministic_option_id: str | None
    deterministic_reason: str | None
    metadata: Mapping[str, object]

    def __init__(
        self,
        site_id: str,
        question_type: QuestionType | str,
        *,
        question_id: str | None = None,
        question: str | None = None,
        site_version: str = "1",
        question_version: str = "1",
        schema_id: str = SCHEMA_ID,
        schema_version: str = SCHEMA_VERSION,
        options: Sequence[OptionSpec | str | Mapping[str, object]] = (),
        levels: Sequence[LevelSpec | int | float | str | Mapping[str, object]] = (),
        code_filters: Sequence[CodeFilter | str | Mapping[str, object]] = (),
        label_source: LabelSource | str | Mapping[str, object] | None = None,
        acceptable_set_policy: AcceptableSetPolicy | Mapping[str, object] | None = None,
        answerability_policy: AnswerabilityPolicy | Mapping[str, object] | None = None,
        loss_spec: LossSpec | Sequence[object] | Mapping[str, object] | None = None,
        cost_pair: LossSpec | Sequence[object] | Mapping[str, object] | None = None,
        fallback: FallbackBoundary | str | Mapping[str, object] = "human_review",
        authority_boundary: AuthorityBoundary | str | Mapping[str, object] = "host",
        evidence_version: str = "1",
        procedure_version: str = "1",
        policy_version: str = "1",
        open_menu: bool = False,
        menu_complete: bool = True,
        menu_expansion_option_id: str | None = None,
        deterministic_option_id: str | None = None,
        deterministic_target_id: str | None = None,
        deterministic_reason: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        site = _identifier(site_id, "site_id")
        qid = _identifier(question_id, "question_id")
        qtext = _text(question, "question", limit=4096)
        kind = _question_type(question_type)
        if deterministic_option_id is not None and deterministic_target_id is not None and deterministic_option_id != deterministic_target_id:
            raise ContractError("deterministic option aliases contradict each other")
        deterministic = deterministic_option_id if deterministic_option_id is not None else deterministic_target_id
        raw_options = tuple(_option(item, index) for index, item in enumerate(_sequence(options, "options")))
        raw_levels = tuple(_level(item, index) for index, item in enumerate(_sequence(levels, "levels")))
        raw_filters = tuple(_code_filter(item, index) for index, item in enumerate(_sequence(code_filters, "code_filters")))
        _unique(tuple(item.option_id for item in raw_options), "option ids")
        _unique(tuple(item.label.casefold() for item in raw_options), "option labels")
        _unique(tuple(item.level_id for item in raw_levels), "level ids")
        _unique(tuple(item.label.casefold() for item in raw_levels), "level labels")
        _unique(tuple(item.filter_id for item in raw_filters), "code filter ids")
        if kind is QuestionType.BOOLEAN:
            if raw_levels:
                raise ContractError("BOOLEAN cannot declare SCORE levels")
            if len(raw_options) != 2:
                raise ContractError("BOOLEAN requires exactly two options")
            if any(item.option_id.casefold() in _NON_BOOLEAN_WORDS or item.label.casefold() in _NON_BOOLEAN_WORDS for item in raw_options):
                raise ContractError("BOOLEAN cannot conflate unknown, insufficient, or abstain with truth")
        elif kind is QuestionType.CHOICE:
            if raw_levels:
                raise ContractError("CHOICE cannot declare SCORE levels")
            if len(raw_options) < 2:
                raise ContractError("CHOICE requires a non-empty menu of at least two options")
        else:
            if raw_options or len(raw_levels) < 2:
                raise ContractError("SCORE requires at least two ordered levels and no CHOICE options")
            values = tuple(item.value for item in raw_levels)
            if any(left >= right for left, right in zip(values, values[1:])):
                raise ContractError("SCORE levels must have strictly increasing values")
        if kind is not QuestionType.SCORE and any(item.value != float(index + 1) for index, item in enumerate(raw_levels)):
            raise ContractError("non-SCORE contracts cannot carry level metadata")
        if not isinstance(open_menu, bool) or not isinstance(menu_complete, bool):
            raise ContractError("menu metadata flags must be boolean")
        if not menu_complete:
            if open_menu:
                raise ContractError("incomplete open menus are not admissible")
            raise ContractError("incomplete menus are not admissible")
        expansion = None if menu_expansion_option_id is None else _identifier(menu_expansion_option_id, "menu_expansion_option_id")
        if expansion is not None and not open_menu:
            raise ContractError("menu expansion metadata requires an open menu")
        if expansion is not None and expansion not in {item.option_id for item in raw_options}:
            raise ContractError("menu expansion option must be in the declared menu")
        target = None if deterministic is None else _identifier(deterministic, "deterministic_option_id")
        all_ids = {item.option_id for item in raw_options} | {item.level_id for item in raw_levels}
        if target is not None and target not in all_ids:
            raise ContractError("deterministic option must be declared by the site")
        reason = None if deterministic_reason is None else _text(deterministic_reason, "deterministic_reason", limit=512)
        if (target is None) != (reason is None):
            raise ContractError("deterministic sites require both a target and a reason")
        if label_source is None:
            raise ContractError("label_source is required")
        if loss_spec is not None and cost_pair is not None:
            raise ContractError("loss_spec and cost_pair are duplicate contract fields")
        loss = _loss(loss_spec if loss_spec is not None else cost_pair)
        policy = _acceptable_policy(acceptable_set_policy)
        if kind is QuestionType.BOOLEAN and policy.allow_multiple:
            raise ContractError("BOOLEAN cannot use a set-valued acceptable outcome policy")
        authority = _authority(authority_boundary)
        object.__setattr__(self, "site_id", site)
        object.__setattr__(self, "site_version", _version(site_version, "site_version"))
        object.__setattr__(self, "question_id", qid)
        object.__setattr__(self, "question_version", _version(question_version, "question_version"))
        object.__setattr__(self, "question_type", kind)
        object.__setattr__(self, "question", qtext)
        object.__setattr__(self, "schema_id", _identifier(schema_id, "schema_id"))
        object.__setattr__(self, "schema_version", _version(schema_version, "schema_version"))
        object.__setattr__(self, "options", raw_options)
        object.__setattr__(self, "levels", raw_levels)
        object.__setattr__(self, "code_filters", raw_filters)
        object.__setattr__(self, "label_source", _label_source(label_source))
        object.__setattr__(self, "acceptable_set_policy", policy)
        object.__setattr__(self, "answerability_policy", _answerability_policy(answerability_policy))
        object.__setattr__(self, "loss_spec", loss)
        object.__setattr__(self, "fallback", _fallback(fallback))
        object.__setattr__(self, "authority_boundary", authority)
        object.__setattr__(self, "evidence_version", _version(evidence_version, "evidence_version"))
        object.__setattr__(self, "procedure_version", _version(procedure_version, "procedure_version"))
        object.__setattr__(self, "policy_version", _version(policy_version, "policy_version"))
        object.__setattr__(self, "open_menu", open_menu)
        object.__setattr__(self, "menu_complete", menu_complete)
        object.__setattr__(self, "menu_expansion_option_id", expansion)
        object.__setattr__(self, "deterministic_option_id", target)
        object.__setattr__(self, "deterministic_reason", reason)
        object.__setattr__(self, "metadata", _metadata(metadata, "site.metadata"))

    @property
    def type(self) -> QuestionType:
        return self.question_type

    @property
    def schema(self) -> str:
        return f"{self.schema_id}/{self.schema_version}"

    @property
    def option_ids(self) -> tuple[str, ...]:
        return tuple(item.option_id for item in self.options)

    @property
    def level_ids(self) -> tuple[str, ...]:
        return tuple(item.level_id for item in self.levels)

    @property
    def level_values(self) -> tuple[float, ...]:
        return tuple(item.value for item in self.levels)

    @property
    def cost_pair(self) -> LossSpec:
        """Compatibility alias for the earlier cost-pair record lane."""
        return self.loss_spec

    @property
    def loss(self) -> LossSpec:
        return self.loss_spec

    @property
    def fallback_boundary(self) -> FallbackBoundary:
        return self.fallback

    @property
    def authority(self) -> AuthorityBoundary:
        return self.authority_boundary

    @property
    def code_filter_ids(self) -> tuple[str, ...]:
        return tuple(item.filter_id for item in self.code_filters)

    @property
    def version_identity(self) -> str:
        encoded = json.dumps(self._identity_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return f"{self.site_id}@{self.site_version}#sha256:{hashlib.sha256(encoded).hexdigest()}"

    @property
    def site_version_identity(self) -> str:
        return self.version_identity

    def _identity_payload(self) -> dict[str, object]:
        return self.to_dict(include_identity=False)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "site_id": self.site_id,
            "site_version": self.site_version,
            "question_id": self.question_id,
            "question_version": self.question_version,
            "type": self.question_type.value,
            "question": self.question,
            "options": [item.to_dict() for item in self.options],
            "levels": [item.to_dict() for item in self.levels],
            "code_filters": [item.to_dict() for item in self.code_filters],
            "label_source": self.label_source.to_dict(),
            "acceptable_set_policy": self.acceptable_set_policy.to_dict(),
            "answerability_policy": self.answerability_policy.to_dict(),
            "loss": self.loss_spec.to_dict(),
            "fallback": self.fallback.to_dict(),
            "authority_boundary": self.authority_boundary.to_dict(),
            "evidence_version": self.evidence_version,
            "procedure_version": self.procedure_version,
            "policy_version": self.policy_version,
            "open_menu": self.open_menu,
            "menu_complete": self.menu_complete,
            "menu_expansion_option_id": self.menu_expansion_option_id,
            "deterministic_option_id": self.deterministic_option_id,
            "deterministic_reason": self.deterministic_reason,
            "metadata": _thaw(self.metadata),
        }
        if include_identity:
            result["site_version_identity"] = self.version_identity
        return result

    def validate_outcome(self, outcome: "OutcomeLabel") -> None:
        if not isinstance(outcome, OutcomeLabel):
            raise ContractError("outcome must be an OutcomeLabel")
        if not self.answerability_policy.allows(outcome.answerability):
            raise ContractError(f"{outcome.answerability.value} rows are not allowed by this site")
        declared = set(self.option_ids) | set(self.level_ids)
        if any(value not in declared for value in outcome.acceptable_option_ids):
            raise ContractError("outcome contains an undeclared option or level")
        if outcome.target_id is not None and outcome.target_id not in declared:
            raise ContractError("outcome target is not declared by the site")
        if outcome.answerability is not Answerability.ANSWERABLE:
            if outcome.target_id is not None or outcome.acceptable_option_ids:
                raise ContractError("unknown/insufficient rows cannot carry a truth or action label")
            return
        if outcome.target_id is None and not outcome.acceptable_option_ids:
            raise ContractError("answerable rows require an explicit target or acceptable set")
        if outcome.target_id is None and not self.acceptable_set_policy.allow_multiple:
            raise ContractError("single-target policy cannot consume a set without an explicit target")
        if outcome.target_id is not None and outcome.acceptable_option_ids and outcome.target_id not in outcome.acceptable_option_ids:
            raise ContractError("target must belong to the acceptable set when both are present")


@dataclass(frozen=True, slots=True, init=False)
class OutcomeLabel:
    """Later label; unknown and insufficient are explicit non-truth rows."""

    decision_id: str
    answerability: Answerability
    target_id: str | None
    acceptable_option_ids: tuple[str, ...]
    label_source: LabelSource | None
    observed_at: str | int | float | None
    metadata: Mapping[str, object]

    def __init__(
        self,
        decision_id: str,
        answerability: Answerability | str = Answerability.ANSWERABLE,
        *,
        target_id: str | None = None,
        label: str | None = None,
        acceptable_option_ids: Sequence[str] = (),
        label_source: LabelSource | str | Mapping[str, object] | None = None,
        observed_at: str | int | float | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(answerability, Answerability):
            status = answerability
        else:
            try:
                status = Answerability(str(answerability).strip().upper())
            except ValueError as error:
                raise ContractError("unknown answerability state") from error
        if target_id is not None and label is not None and target_id != label:
            raise ContractError("target_id and label contradict each other")
        target = target_id if target_id is not None else label
        target_value = None if target is None else _identifier(target, "outcome.target_id")
        accepted = tuple(_identifier(item, "outcome.acceptable_option_ids[]") for item in _sequence(acceptable_option_ids, "acceptable_option_ids"))
        _unique(accepted, "acceptable_option_ids")
        if status is not Answerability.ANSWERABLE and (target_value is not None or accepted):
            raise ContractError("UNKNOWN and INSUFFICIENT rows cannot be coerced into false or a choice")
        source = None if label_source is None else _label_source(label_source)
        if observed_at is not None and not isinstance(observed_at, str) and (isinstance(observed_at, bool) or not isinstance(observed_at, (int, float)) or not math.isfinite(float(observed_at))):
            raise ContractError("outcome.observed_at must be a finite timestamp")
        if isinstance(observed_at, str) and not observed_at.strip():
            raise ContractError("outcome.observed_at must not be empty")
        object.__setattr__(self, "decision_id", _identifier(decision_id, "outcome.decision_id"))
        object.__setattr__(self, "answerability", status)
        object.__setattr__(self, "target_id", target_value)
        object.__setattr__(self, "acceptable_option_ids", accepted)
        object.__setattr__(self, "label_source", source)
        object.__setattr__(self, "observed_at", observed_at.strip() if isinstance(observed_at, str) else observed_at)
        object.__setattr__(self, "metadata", _metadata(metadata, "outcome.metadata"))

    @property
    def label(self) -> str | None:
        return self.target_id

    @property
    def acceptable_set(self) -> tuple[str, ...]:
        return self.acceptable_option_ids

    def to_one_hot(self, declared_ids: Sequence[str] | None = None) -> dict[str, float]:
        if self.answerability is not Answerability.ANSWERABLE or self.target_id is None:
            raise AmbiguousOutcomeError("only an explicit ANSWERABLE target can become one-hot")
        if declared_ids is None:
            ids = (self.target_id,)
        else:
            ids = tuple(_identifier(item, "declared_ids[]") for item in _sequence(declared_ids, "declared_ids"))
            _unique(ids, "declared_ids")
            if self.target_id not in ids:
                raise AmbiguousOutcomeError("explicit target is not in the declared domain")
        return {item: 1.0 if item == self.target_id else 0.0 for item in ids}

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "answerability": self.answerability.value,
            "target_id": self.target_id,
            "label": self.target_id,
            "acceptable_option_ids": list(self.acceptable_option_ids),
            "label_source": self.label_source.to_dict() if self.label_source else None,
            "observed_at": self.observed_at,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Minimal immutable identity envelope shared with record/journal lanes."""

    decision_id: str
    site_id: str
    site_version_identity: str
    question_id: str
    question_version: str
    question_type: QuestionType
    schema_id: str
    schema_version: str
    evidence_version: str
    procedure_version: str
    payload: Mapping[str, object] = field(default_factory=dict)
    run_id: str | None = None
    task_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_id", _identifier(self.decision_id, "decision_id"))
        object.__setattr__(self, "site_id", _identifier(self.site_id, "site_id"))
        object.__setattr__(self, "site_version_identity", _text(self.site_version_identity, "site_version_identity", limit=512))
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(self, "question_version", _version(self.question_version, "question_version"))
        object.__setattr__(self, "question_type", _question_type(self.question_type))
        object.__setattr__(self, "schema_id", _identifier(self.schema_id, "schema_id"))
        object.__setattr__(self, "schema_version", _version(self.schema_version, "schema_version"))
        object.__setattr__(self, "evidence_version", _version(self.evidence_version, "evidence_version"))
        object.__setattr__(self, "procedure_version", _version(self.procedure_version, "procedure_version"))
        object.__setattr__(self, "payload", _freeze(self.payload, "record.payload"))
        if self.run_id is not None:
            object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        if self.task_id is not None:
            object.__setattr__(self, "task_id", _identifier(self.task_id, "task_id"))

    @property
    def type(self) -> QuestionType:
        return self.question_type

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "site_id": self.site_id,
            "site_version_identity": self.site_version_identity,
            "question_id": self.question_id,
            "question_version": self.question_version,
            "type": self.question_type.value,
            "evidence_version": self.evidence_version,
            "procedure_version": self.procedure_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "payload": _thaw(self.payload),
        }


# Readable aliases for consumers that prefer type-specific terminology.
ChoiceContract = DecisionSite
BooleanContract = DecisionSite
ScoreContract = DecisionSite
DecisionOutcome = OutcomeLabel
CostPair = LossSpec


__all__ = [
    "AcceptableSetPolicy",
    "AmbiguousOutcomeError",
    "Answerability",
    "AnswerabilityPolicy",
    "AuthorityBoundary",
    "BooleanContract",
    "ChoiceContract",
    "CodeFilter",
    "CostPair",
    "CONTRACT_SCHEMA",
    "ContractError",
    "DECISION_LABELLED",
    "DECISION_RECORDED",
    "DecisionOutcome",
    "DecisionRecord",
    "DecisionSite",
    "FallbackBoundary",
    "LabelSource",
    "LevelSpec",
    "LossSpec",
    "OptionSpec",
    "OutcomeLabel",
    "QuestionType",
    "ResolutionStatus",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "ScoreContract",
]
