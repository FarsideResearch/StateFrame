"""The single in-memory owner of registered System One site definitions.

Registration is explicit and versioned.  Lookup and deterministic resolution
are pure operations; this module never evaluates filters, calls a provider,
looks up a receipt, or applies an effect.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

try:  # Flat v46 module loading.
    from kernel_continuation_journal import RESERVED, journal_safe_fields
except ImportError:  # pragma: no cover - package import path.
    from ..kernel_continuation_journal import RESERVED, journal_safe_fields

from .contracts import (
    DECISION_LABELLED,
    DECISION_RECORDED,
    Answerability,
    ContractError,
    DecisionRecord,
    DecisionSite,
    OutcomeLabel,
    QuestionType,
    ResolutionStatus,
)


class RegistryError(ContractError):
    """Base registry error."""


class DuplicateSiteVersionError(RegistryError):
    """A site/version key was registered with a different definition."""


class UnknownSiteError(RegistryError):
    """A requested site or site version is not registered."""


class AmbiguousSiteVersionError(RegistryError):
    """A site lookup omitted a version while multiple versions exist."""


class JournalEventError(RegistryError):
    """A decision event is incomplete or unsafe for journal emission."""


_RECORD_SCHEMA = "sys1.records/1"
_API_DECISION_SCHEMA = "stateframe-decision-api/1"


@dataclass(frozen=True, slots=True)
class DeterministicResolution:
    """A proposal produced without a model call or authority change."""

    site_id: str
    site_version_identity: str
    option_id: str | None
    status: ResolutionStatus
    reason: str
    model_calls: int = 0
    authoritative: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResolutionStatus):
            try:
                object.__setattr__(self, "status", ResolutionStatus(str(self.status).upper()))
            except ValueError as error:
                raise RegistryError("unknown deterministic resolution status") from error
        if self.model_calls != 0:
            raise RegistryError("registry resolution cannot claim model calls")
        if self.authoritative is not False:
            raise RegistryError("registry resolution is never authoritative")

    @property
    def calls(self) -> int:
        return self.model_calls

    @property
    def call_count(self) -> int:
        return self.model_calls

    def to_dict(self) -> dict[str, object]:
        return {
            "site_id": self.site_id,
            "site_version_identity": self.site_version_identity,
            "option_id": self.option_id,
            "status": self.status.value,
            "reason": self.reason,
            "model_calls": self.model_calls,
            "authoritative": self.authoritative,
        }


class JournalLike(Protocol):
    def emit(self, kind: str, **fields: object) -> dict[str, object]: ...


def _payload(value: DecisionRecord | OutcomeLabel | Mapping[str, object]) -> dict[str, object]:
    if isinstance(value, (DecisionRecord, OutcomeLabel)):
        return value.to_dict()
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if isinstance(value, Mapping):
        return dict(value)
    raise JournalEventError("event payload must be a supported immutable record or mapping")


def validate_journal_event_fields(
    kind: str,
    payload: DecisionRecord | OutcomeLabel | Mapping[str, object],
) -> Mapping[str, object]:
    """Validate required identity, then apply the existing journal safe-field rule.

    ``journal_safe_fields`` renames envelope keys such as ``run_id`` rather than
    dropping them.  A pre-existing ``receipt_<reserved>`` key would collide with
    that rename, so it is refused instead of silently overwriting caller data.
    """
    if kind not in {DECISION_RECORDED, DECISION_LABELLED}:
        raise JournalEventError("unsupported decision journal event kind")
    raw = _payload(payload)
    rich_record = raw.get("record_schema") == _RECORD_SCHEMA
    api_record = raw.get("schema") == _API_DECISION_SCHEMA
    if kind == DECISION_RECORDED:
        if rich_record:
            required = (
                "decision_id", "task_id", "site_id", "question_version",
                "prompt_digest", "evidence_digest",
            )
        elif api_record:
            # The provider-neutral API owns its complete scoring payload; this
            # seam validates its identity without importing or executing it.
            required = (
                "decision_id", "site_id", "question_version", "type",
                "options", "option_order", "permutation",
            )
        else:
            required = (
                "decision_id", "site_id", "site_version_identity",
                "question_version", "type",
            )
    else:
        required = ("decision_id", "answerability")
    missing = tuple(field for field in required if field not in raw)
    if missing:
        raise JournalEventError(f"event missing required fields: {', '.join(missing)}")
    if kind == DECISION_RECORDED:
        event_type = raw.get("question_type") if rich_record else raw.get("type")
        if isinstance(event_type, QuestionType):
            event_type = event_type.value
        if event_type not in {item.value for item in QuestionType}:
            raise JournalEventError("event type must be CHOICE, BOOLEAN, or SCORE")
    else:
        answerability = raw["answerability"]
        if isinstance(answerability, Answerability):
            answerability = answerability.value
        if answerability not in {item.value for item in Answerability}:
            raise JournalEventError("event answerability is unsupported")
    for key in raw:
        if key.startswith("receipt_") and key[8:] in RESERVED:
            raise JournalEventError(f"event field collides with journal-safe rename: {key}")
    safe = journal_safe_fields(raw)
    if any(key in RESERVED for key in safe):
        raise JournalEventError("journal-safe event still contains a reserved field")
    return MappingProxyType(dict(safe))


def emit_decision_recorded(
    journal: JournalLike,
    record: DecisionRecord | Mapping[str, object],
) -> dict[str, object]:
    """Emit one safe ``decision_recorded`` event through the host journal."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise JournalEventError("journal must provide emit(kind, **fields)")
    fields = validate_journal_event_fields(DECISION_RECORDED, record)
    return emit(DECISION_RECORDED, **dict(fields))


def emit_decision_labelled(
    journal: JournalLike,
    outcome: OutcomeLabel | Mapping[str, object],
) -> dict[str, object]:
    """Emit one safe ``decision_labelled`` event through the host journal."""
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        raise JournalEventError("journal must provide emit(kind, **fields)")
    fields = validate_journal_event_fields(DECISION_LABELLED, outcome)
    return emit(DECISION_LABELLED, **dict(fields))


class DecisionSiteRegistry:
    """Mutable registration boundary with immutable definitions and snapshots."""

    def __init__(self, sites: Sequence[DecisionSite] = ()) -> None:
        self._sites: dict[tuple[str, str], DecisionSite] = {}
        for site in sites:
            self.register(site)

    def register(self, site: DecisionSite) -> DecisionSite:
        if not isinstance(site, DecisionSite):
            raise RegistryError("registry accepts DecisionSite instances only")
        key = (site.site_id, site.site_version)
        previous = self._sites.get(key)
        if previous is not None and previous.version_identity != site.version_identity:
            raise DuplicateSiteVersionError(f"site version already registered: {site.site_id}@{site.site_version}")
        self._sites[key] = site
        return site

    register_site = register

    def get(self, site_id: str, site_version: str | None = None) -> DecisionSite | None:
        if site_version is not None:
            return self._sites.get((site_id, site_version))
        matches = tuple(site for (ident, _), site in self._sites.items() if ident == site_id)
        if len(matches) > 1:
            raise AmbiguousSiteVersionError(f"site version is required: {site_id}")
        return matches[0] if matches else None

    def require(self, site_id: str, site_version: str | None = None) -> DecisionSite:
        site = self.get(site_id, site_version)
        if site is None:
            suffix = "" if site_version is None else f"@{site_version}"
            raise UnknownSiteError(f"unregistered decision site: {site_id}{suffix}")
        return site

    def all(self) -> tuple[DecisionSite, ...]:
        return tuple(self._sites[key] for key in sorted(self._sites))

    def snapshot(self) -> Mapping[tuple[str, str], DecisionSite]:
        return MappingProxyType(dict(self._sites))

    def resolve(self, site_id: str, site_version: str | None = None) -> DeterministicResolution:
        """Resolve only a declared deterministic answer, with exactly zero calls."""
        site = self.require(site_id, site_version)
        if site.deterministic_option_id is None:
            return DeterministicResolution(
                site_id=site.site_id,
                site_version_identity=site.version_identity,
                option_id=None,
                status=ResolutionStatus.MODEL_REQUIRED,
                reason="site has no deterministic answer; provider/policy lane remains separate",
            )
        return DeterministicResolution(
            site_id=site.site_id,
            site_version_identity=site.version_identity,
            option_id=site.deterministic_option_id,
            status=ResolutionStatus.DETERMINISTIC,
            reason=site.deterministic_reason or "declared deterministic site rule",
        )

    resolve_deterministic = resolve

    def validate_outcome(self, site_id: str, outcome: OutcomeLabel, site_version: str | None = None) -> None:
        self.require(site_id, site_version).validate_outcome(outcome)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            return key in self._sites
        if isinstance(key, str):
            return any(site_id == key for site_id, _ in self._sites)
        return False

    def __len__(self) -> int:
        return len(self._sites)


# Concise aliases for API consumers and prior registry naming.
Registry = DecisionSiteRegistry
DecisionRegistryError = RegistryError
DuplicateDecisionSiteError = DuplicateSiteVersionError
UnknownDecisionSiteError = UnknownSiteError
DecisionEventError = JournalEventError


__all__ = [
    "AmbiguousSiteVersionError",
    "DecisionEventError",
    "DecisionRegistryError",
    "DecisionSiteRegistry",
    "DeterministicResolution",
    "DuplicateDecisionSiteError",
    "DuplicateSiteVersionError",
    "JournalEventError",
    "Registry",
    "RegistryError",
    "UnknownDecisionSiteError",
    "UnknownSiteError",
    "emit_decision_labelled",
    "emit_decision_recorded",
    "validate_journal_event_fields",
]
