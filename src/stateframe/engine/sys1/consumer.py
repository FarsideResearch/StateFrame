"""Shared host-owned seam for System One consumer adapters.

The seam deliberately knows nothing about Python syntax, schedulers, model
providers, files, or evaluators.  Adapters offer bounded host-created options;
this module validates the offer, optionally asks an injected provider for an
option identity, and binds any resulting host receipt to the exact packet and
handle.  It never certifies completion or applies an effect.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


SCHEMA = "sys1.consumer/1"
MAX_MENU_OPTIONS = 8
MAX_PREDICATES = 32
MAX_EVIDENCE_REFS = 64
MAX_TRIED_IDENTITIES = 128
_PUBLIC_LIMIT = 16_384
_EVALUATOR_KEYS = frozenset(
    {
        "answer_key",
        "canonical_solution",
        "expected_patch",
        "gold",
        "grader",
        "ground_truth",
        "hidden_test",
        "oracle",
        "reference_fix",
        "reference_solution",
        "evaluator",
    }
)
_EVALUATOR_TEXT = (
    "canonical solution",
    "expected patch",
    "ground truth",
    "hidden test",
    "reference fix",
    "reference solution",
    "grader-only",
    "evaluator-only",
)


class ConsumerError(ValueError):
    """A host contract was incomplete, stale, illegal, or unsafe."""


class ConsumerMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    ACTIVE = "ACTIVE"


class DecisionStatus(str, Enum):
    DISABLED = "DISABLED"
    NO_OPPORTUNITY = "NO_OPPORTUNITY"
    FORCED = "FORCED"
    SHADOWED = "SHADOWED"
    DISPATCHED = "DISPATCHED"
    UNKNOWN_EFFECT = "UNKNOWN_EFFECT"
    ABSTAIN = "ABSTAIN"
    BLOCKED = "BLOCKED"
    REJECTED = "REJECTED"
    STALE = "STALE"
    RECEIPT_MISSING = "RECEIPT_MISSING"
    DISPATCH_FAILED = "DISPATCH_FAILED"


class CapabilityState(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"
    STALE = "STALE"


class PredicateState(str, Enum):
    UNRESOLVED = "UNRESOLVED"
    CONTRADICTED = "CONTRADICTED"
    RESOLVED = "RESOLVED"
    UNKNOWN = "UNKNOWN"


def _text(value: object, field_name: str, *, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConsumerError(f"{field_name} must be non-empty text")
    value = value.replace("\x00", " ").strip()
    if len(value) > limit:
        raise ConsumerError(f"{field_name} exceeds its bounded length")
    return value


def _optional_text(value: object, field_name: str, *, limit: int = 512) -> str | None:
    if value is None:
        return None
    return _text(value, field_name, limit=limit)


def _unique_text(values: Sequence[object], field_name: str, *, limit: int = 128) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ConsumerError(f"{field_name} must be a sequence")
    result: list[str] = []
    for value in values:
        item = _text(value, field_name)
        if item not in result:
            result.append(item)
        if len(result) > limit:
            raise ConsumerError(f"{field_name} exceeds its bounded length")
    return tuple(result)


def _public_copy(value: object, *, path: str = "value", depth: int = 0) -> object:
    """Copy only bounded JSON-like public data and reject evaluator material."""

    if depth > 8:
        raise ConsumerError(f"{path} is too deeply nested")
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise ConsumerError(f"{path} has too many fields")
        result: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _text(raw_key, f"{path}.key", limit=128).casefold()
            if key in _EVALUATOR_KEYS or any(marker in key for marker in ("grader", "evaluator", "ground_truth", "hidden_test")):
                raise ConsumerError(f"{path} contains evaluator material")
            if isinstance(raw_value, str) and any(marker in raw_value.casefold() for marker in _EVALUATOR_TEXT):
                raise ConsumerError(f"{path}.{key} contains evaluator material")
            result[key] = _public_copy(raw_value, path=f"{path}.{key}", depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 128:
            raise ConsumerError(f"{path} has too many items")
        return [_public_copy(item, path=f"{path}[{index}]", depth=depth + 1) for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConsumerError(f"{path} contains a non-finite number")
        return value
    raise ConsumerError(f"{path} contains a non-public value")


def reject_evaluator_material(value: object) -> None:
    """Fail closed at the solver boundary when evaluator-only data is present."""

    _public_copy(value)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def public_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(_public_copy(value)).encode("utf-8")).hexdigest()


def _enum_value(value: object, enum_type: type[Enum], field_name: str) -> str:
    if isinstance(value, enum_type):
        return str(value.value)
    if not isinstance(value, str):
        raise ConsumerError(f"{field_name} must be text")
    try:
        return str(enum_type(value.upper()).value)
    except ValueError as exc:
        raise ConsumerError(f"{field_name} is unsupported") from exc


@dataclass(frozen=True, slots=True)
class UnresolvedPredicate:
    predicate_id: str
    statement: str
    state: PredicateState = PredicateState.UNRESOLVED
    required_roles: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate_id", _text(self.predicate_id, "predicate_id"))
        object.__setattr__(self, "statement", _text(self.statement, "predicate statement", limit=1000))
        object.__setattr__(self, "state", PredicateState(_enum_value(self.state, PredicateState, "predicate state")))
        object.__setattr__(self, "required_roles", _unique_text(self.required_roles, "predicate required_roles"))
        object.__setattr__(self, "evidence_refs", _unique_text(self.evidence_refs, "predicate evidence_refs"))

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "statement": self.statement,
            "state": self.state.value,
            "required_roles": list(self.required_roles),
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class Capability:
    name: str
    state: CapabilityState
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "capability name", limit=128))
        object.__setattr__(self, "state", CapabilityState(_enum_value(self.state, CapabilityState, "capability state")))
        object.__setattr__(self, "detail", self.detail if isinstance(self.detail, str) else str(self.detail))

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "state": self.state.value, "detail": self.detail[:512]}


@dataclass(frozen=True, slots=True)
class CandidateCoverage:
    candidate_ids: tuple[str, ...]
    total_candidates: int
    complete: bool
    omitted_count: int = 0
    next_page_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = _unique_text(self.candidate_ids, "candidate_ids", limit=256)
        if self.total_candidates < len(ids) or self.total_candidates < 0:
            raise ConsumerError("candidate coverage total is inconsistent")
        if self.omitted_count != self.total_candidates - len(ids) or self.omitted_count < 0:
            raise ConsumerError("candidate coverage omitted_count is inconsistent")
        if self.complete and self.omitted_count:
            raise ConsumerError("complete candidate coverage cannot omit candidates")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "next_page_handles", _unique_text(self.next_page_handles, "next_page_handles", limit=16))

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_ids": list(self.candidate_ids),
            "total_candidates": self.total_candidates,
            "complete": self.complete,
            "omitted_count": self.omitted_count,
            "next_page_handles": list(self.next_page_handles),
        }


@dataclass(frozen=True, slots=True)
class ActionHandle:
    """Opaque, revision-bound host action; adapters cannot mutate files."""

    handle_id: str
    action_id: str
    kind: str
    language: str
    revision: str
    required_capabilities: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    mutates_files: bool = False
    expires_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "handle_id", _text(self.handle_id, "handle_id", limit=256))
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id", limit=256))
        object.__setattr__(self, "kind", _text(self.kind, "handle kind", limit=80).upper())
        object.__setattr__(self, "language", _text(self.language, "handle language", limit=80).casefold())
        object.__setattr__(self, "revision", _text(self.revision, "handle revision", limit=256))
        object.__setattr__(self, "required_capabilities", _unique_text(self.required_capabilities, "required_capabilities"))
        object.__setattr__(self, "required_roles", _unique_text(self.required_roles, "handle required_roles"))
        object.__setattr__(self, "metadata", _public_copy(self.metadata, path="handle metadata"))
        if self.mutates_files:
            raise ConsumerError("consumer handles cannot mutate files")
        object.__setattr__(self, "expires_at", _optional_text(self.expires_at, "handle expires_at", limit=128))

    def to_dict(self) -> dict[str, object]:
        return {
            "handle_id": self.handle_id,
            "action_id": self.action_id,
            "kind": self.kind,
            "language": self.language,
            "revision": self.revision,
            "required_capabilities": list(self.required_capabilities),
            "required_roles": list(self.required_roles),
            "metadata": copy.deepcopy(self.metadata),
            "mutates_files": self.mutates_files,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class MenuOption:
    option_id: str
    label: str
    handle: ActionHandle | None = None
    legal: bool = True
    reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", _text(self.option_id, "option_id", limit=256))
        object.__setattr__(self, "label", _text(self.label, "option label", limit=256))
        if self.handle is not None and not isinstance(self.handle, ActionHandle):
            raise ConsumerError("menu handle must be an ActionHandle")
        object.__setattr__(self, "reason", self.reason if isinstance(self.reason, str) else str(self.reason))
        object.__setattr__(self, "metadata", _public_copy(self.metadata, path="menu metadata"))

    def to_dict(self) -> dict[str, object]:
        return {
            "option_id": self.option_id,
            "label": self.label,
            "handle": self.handle.to_dict() if self.handle else None,
            "legal": self.legal,
            "reason": self.reason[:512],
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ConsumerPacket:
    consumer_id: str
    site_id: str
    question: str
    question_type: str
    language: str
    revision: str
    predicates: tuple[UnresolvedPredicate, ...]
    candidate_coverage: CandidateCoverage
    already_tried_ids: tuple[str, ...]
    required_roles: tuple[str, ...]
    bound_roles: tuple[str, ...]
    capabilities: tuple[Capability, ...]
    menu: tuple[MenuOption, ...]
    menu_complete: bool
    evidence_refs: tuple[str, ...] = ()
    contradictions: tuple[str, ...] = ()
    hard_stop: bool = False
    unknown_effects: tuple[str, ...] = ()
    baseline_frame: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "consumer_id", _text(self.consumer_id, "consumer_id", limit=128))
        object.__setattr__(self, "site_id", _text(self.site_id, "site_id", limit=256))
        object.__setattr__(self, "question", _text(self.question, "question", limit=2000))
        object.__setattr__(self, "question_type", _text(self.question_type, "question_type", limit=64).upper())
        object.__setattr__(self, "language", _text(self.language, "language", limit=80).casefold())
        object.__setattr__(self, "revision", _text(self.revision, "revision", limit=256))
        predicates = tuple(self.predicates)
        if len(predicates) > MAX_PREDICATES or any(not isinstance(item, UnresolvedPredicate) for item in predicates):
            raise ConsumerError("predicates are invalid or unbounded")
        object.__setattr__(self, "predicates", predicates)
        if not isinstance(self.candidate_coverage, CandidateCoverage):
            raise ConsumerError("candidate coverage is required")
        object.__setattr__(self, "already_tried_ids", _unique_text(self.already_tried_ids, "already_tried_ids", limit=MAX_TRIED_IDENTITIES))
        object.__setattr__(self, "required_roles", _unique_text(self.required_roles, "required_roles"))
        object.__setattr__(self, "bound_roles", _unique_text(self.bound_roles, "bound_roles"))
        capabilities = tuple(self.capabilities)
        if any(not isinstance(item, Capability) for item in capabilities):
            raise ConsumerError("capabilities are invalid")
        if len({item.name for item in capabilities}) != len(capabilities):
            raise ConsumerError("capability names must be unique")
        object.__setattr__(self, "capabilities", capabilities)
        menu = tuple(self.menu)
        if len(menu) > MAX_MENU_OPTIONS or len({item.option_id for item in menu}) != len(menu):
            raise ConsumerError("menu is unbounded or contains duplicate options")
        if any(not isinstance(item, MenuOption) for item in menu):
            raise ConsumerError("menu contains an invalid option")
        object.__setattr__(self, "menu", menu)
        object.__setattr__(self, "evidence_refs", _unique_text(self.evidence_refs, "evidence_refs", limit=MAX_EVIDENCE_REFS))
        object.__setattr__(self, "contradictions", _unique_text(self.contradictions, "contradictions", limit=32))
        object.__setattr__(self, "unknown_effects", _unique_text(self.unknown_effects, "unknown_effects", limit=32))
        object.__setattr__(self, "baseline_frame", _public_copy(self.baseline_frame, path="baseline_frame"))
        object.__setattr__(self, "metadata", _public_copy(self.metadata, path="packet metadata"))

    @property
    def packet_id(self) -> str:
        return public_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "consumer_id": self.consumer_id,
            "site_id": self.site_id,
            "question": self.question,
            "question_type": self.question_type,
            "language": self.language,
            "revision": self.revision,
            "predicates": [item.to_dict() for item in self.predicates],
            "candidate_coverage": self.candidate_coverage.to_dict(),
            "already_tried_ids": list(self.already_tried_ids),
            "required_roles": list(self.required_roles),
            "bound_roles": list(self.bound_roles),
            "capabilities": [item.to_dict() for item in self.capabilities],
            "menu": [item.to_dict() for item in self.menu],
            "menu_complete": self.menu_complete,
            "evidence_refs": list(self.evidence_refs),
            "contradictions": list(self.contradictions),
            "hard_stop": self.hard_stop,
            "unknown_effects": list(self.unknown_effects),
            "baseline_frame": copy.deepcopy(self.baseline_frame),
            "metadata": copy.deepcopy(self.metadata),
        }

    def provider_view(self) -> dict[str, object]:
        payload = self.to_dict()
        payload.pop("baseline_frame", None)
        return payload


@dataclass(frozen=True, slots=True)
class DispatchReceipt:
    receipt_id: str
    decision_id: str
    handle_id: str
    action_id: str
    revision: str
    status: str
    outcome_status: str
    labels: Mapping[str, str] = field(default_factory=dict)
    unknown_effect: bool = False
    public_observations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "receipt_id": self.receipt_id,
            "decision_id": self.decision_id,
            "handle_id": self.handle_id,
            "action_id": self.action_id,
            "revision": self.revision,
            "status": self.status,
            "outcome_status": self.outcome_status,
            "labels": dict(self.labels),
            "unknown_effect": self.unknown_effect,
            "public_observations": list(self.public_observations),
        }

    @classmethod
    def from_host(cls, decision_id: str, handle: ActionHandle, raw: object) -> "DispatchReceipt":
        if not isinstance(raw, Mapping):
            raise ConsumerError("host dispatch did not return a receipt object")
        reject_evaluator_material(raw)
        receipt_id = raw.get("receipt_id")
        if not isinstance(receipt_id, str) or not receipt_id.strip():
            raise ConsumerError("host dispatch receipt_id is required")
        for key, expected in (("decision_id", decision_id), ("handle_id", handle.handle_id), ("revision", handle.revision)):
            if key in raw and raw[key] != expected:
                raise ConsumerError(f"host receipt is stale for {key}")
        status = str(raw.get("status", "UNKNOWN")).upper()
        outcome = str(raw.get("outcome_status", raw.get("outcome", "UNKNOWN"))).upper()
        labels_raw = raw.get("labels", {})
        if not isinstance(labels_raw, Mapping):
            raise ConsumerError("host receipt labels must be an object")
        labels: dict[str, str] = {}
        for key, value in labels_raw.items():
            labels[_text(key, "receipt label", limit=80)] = _text(value, "receipt label value", limit=80).upper()
        observations_raw = raw.get("public_observations", ())
        observations = _unique_text(observations_raw, "public_observations", limit=16) if isinstance(observations_raw, Sequence) and not isinstance(observations_raw, (str, bytes, bytearray)) else ()
        return cls(
            receipt_id=receipt_id,
            decision_id=decision_id,
            handle_id=handle.handle_id,
            action_id=handle.action_id,
            revision=handle.revision,
            status=status,
            outcome_status=outcome,
            labels=labels,
            unknown_effect=bool(raw.get("unknown_effect", False)) or outcome in {"UNKNOWN", "UNKNOWN_EFFECT"},
            public_observations=observations,
        )


@dataclass(frozen=True, slots=True)
class AdapterResult:
    status: DecisionStatus
    decision_id: str
    packet_id: str
    selected_option_id: str | None = None
    reason: str = ""
    receipt: DispatchReceipt | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    shadow_frame: Mapping[str, object] | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "status": self.status.value,
            "decision_id": self.decision_id,
            "packet_id": self.packet_id,
            "selected_option_id": self.selected_option_id,
            "reason": self.reason,
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "labels": dict(self.labels),
            "shadow_frame": copy.deepcopy(self.shadow_frame) if self.shadow_frame is not None else None,
            "details": copy.deepcopy(self.details),
        }


Provider = Callable[[Mapping[str, object]], object]
Dispatcher = Callable[[ActionHandle], object]


def _capability_map(packet: ConsumerPacket) -> dict[str, CapabilityState]:
    return {item.name: item.state for item in packet.capabilities}


def _blocked_reason(packet: ConsumerPacket, option: MenuOption) -> str | None:
    option_roles = option.handle.required_roles if option.handle else ()
    missing_roles = tuple(role for role in packet.required_roles + option_roles if role not in packet.bound_roles)
    if missing_roles:
        return "missing_required_role:" + ",".join(dict.fromkeys(missing_roles))
    if option.handle is None:
        return None
    states = _capability_map(packet)
    unavailable = tuple(
        f"{name}:{states.get(name, CapabilityState.MISSING).value}"
        for name in option.handle.required_capabilities
        if states.get(name, CapabilityState.MISSING) is not CapabilityState.AVAILABLE
    )
    return "capability_unavailable:" + ",".join(unavailable) if unavailable else None


def _provider_option(provider: Provider, packet: ConsumerPacket) -> str:
    raw = provider(packet.provider_view())
    reject_evaluator_material(raw)
    if isinstance(raw, str):
        return _text(raw, "provider option", limit=256)
    if isinstance(raw, Mapping):
        value = raw.get("option_id", raw.get("selected_option_id"))
        return _text(value, "provider option", limit=256)
    raise ConsumerError("provider must return an option identity")


def _result(
    packet: ConsumerPacket,
    status: DecisionStatus,
    decision_id: str,
    *,
    selected: str | None = None,
    reason: str = "",
    receipt: DispatchReceipt | None = None,
    labels: Mapping[str, str] | None = None,
    shadow_frame: Mapping[str, object] | None = None,
    details: Mapping[str, object] | None = None,
) -> AdapterResult:
    return AdapterResult(
        status=status,
        decision_id=decision_id,
        packet_id=packet.packet_id,
        selected_option_id=selected,
        reason=reason,
        receipt=receipt,
        labels=dict(labels or (receipt.labels if receipt else {})),
        shadow_frame=copy.deepcopy(shadow_frame) if shadow_frame is not None else None,
        details=dict(details or {}),
    )


def run_menu(
    packet: ConsumerPacket,
    *,
    decision_id: str,
    mode: ConsumerMode | str = ConsumerMode.OFF,
    qualified: bool = False,
    provider: Provider | None = None,
    dispatcher: Dispatcher | None = None,
) -> AdapterResult:
    """Run one bounded consumer decision through the existing host dispatcher."""

    try:
        current_mode = ConsumerMode(_enum_value(mode, ConsumerMode, "consumer mode"))
        decision_id = _text(decision_id, "decision_id", limit=256)
        reject_evaluator_material(packet.to_dict())
    except ConsumerError as exc:
        return _result(packet, DecisionStatus.REJECTED, str(decision_id), reason=str(exc))
    if current_mode is ConsumerMode.OFF:
        return _result(packet, DecisionStatus.DISABLED, decision_id, reason="consumer mode is OFF")
    if packet.hard_stop:
        return _result(packet, DecisionStatus.BLOCKED, decision_id, reason="host hard stop is active")
    missing_roles = tuple(role for role in packet.required_roles if role not in packet.bound_roles)
    if missing_roles:
        return _result(packet, DecisionStatus.BLOCKED, decision_id, reason="missing_required_role:" + ",".join(missing_roles))
    language_state = _capability_map(packet).get("language", CapabilityState.UNKNOWN)
    if language_state is not CapabilityState.AVAILABLE:
        return _result(packet, DecisionStatus.BLOCKED, decision_id, reason=f"language_capability:{language_state.value}")
    legal = tuple(option for option in packet.menu if option.legal and _blocked_reason(packet, option) is None)
    if packet.unknown_effects:
        legal = tuple(
            option for option in legal
            if str(option.metadata.get("kind", "")).upper() in {"RECONCILE", "REFRESH", "INSPECT"}
            or (option.handle is not None and option.handle.kind in {"RECONCILE", "REFRESH", "INSPECT"})
        )
        if not legal:
            return _result(packet, DecisionStatus.BLOCKED, decision_id, reason="unknown effects require host reconciliation", details={"unknown_effects": list(packet.unknown_effects)})
    if not legal:
        reason = "no legal host route" if packet.menu else "zero host routes"
        return _result(packet, DecisionStatus.NO_OPPORTUNITY, decision_id, reason=reason)
    if not packet.menu_complete:
        expansion = tuple(option for option in legal if option.handle and option.handle.kind == "EXPAND")
        if not expansion:
            return _result(packet, DecisionStatus.BLOCKED, decision_id, reason="candidate menu is incomplete and has no expansion handle")
        # Incomplete coverage may only dispatch the host's expansion handle;
        # selecting a visible candidate before expansion would make coverage
        # look complete to the provider.
        legal = expansion
    if not qualified and current_mode is ConsumerMode.ACTIVE:
        return _result(packet, DecisionStatus.BLOCKED, decision_id, reason="active consumer contract is not qualified")
    if len(legal) == 1:
        selected = legal[0]
    else:
        if provider is None:
            return _result(packet, DecisionStatus.ABSTAIN, decision_id, reason="provider is not installed")
        try:
            selected_id = _provider_option(provider, packet)
        except ConsumerError as exc:
            return _result(packet, DecisionStatus.REJECTED, decision_id, reason=str(exc))
        selected = next((option for option in legal if option.option_id == selected_id), None)
        if selected is None:
            return _result(packet, DecisionStatus.REJECTED, decision_id, reason="provider selected an option outside the host menu")
    if selected.handle is None:
        return _result(packet, DecisionStatus.FORCED, decision_id, selected=selected.option_id, reason="host supplied a non-dispatch decision")
    if current_mode is ConsumerMode.SHADOW:
        return _result(
            packet,
            DecisionStatus.SHADOWED,
            decision_id,
            selected=selected.option_id,
            reason="shadow proposal was not dispatched",
            shadow_frame=packet.baseline_frame,
            details={"action_id": selected.handle.action_id, "handle_id": selected.handle.handle_id},
        )
    if dispatcher is None:
        return _result(packet, DecisionStatus.BLOCKED, decision_id, selected=selected.option_id, reason="host dispatcher is not installed")
    try:
        raw_receipt = dispatcher(selected.handle)
        receipt = DispatchReceipt.from_host(decision_id, selected.handle, raw_receipt)
    except ConsumerError as exc:
        reason = str(exc)
        status = DecisionStatus.STALE if "stale" in reason.casefold() else DecisionStatus.REJECTED
        if "receipt_id" in reason:
            status = DecisionStatus.RECEIPT_MISSING
        return _result(packet, status, decision_id, selected=selected.option_id, reason=reason)
    except Exception as exc:  # Host failures are typed, not leaked.
        return _result(packet, DecisionStatus.DISPATCH_FAILED, decision_id, selected=selected.option_id, reason=f"host dispatch failed: {type(exc).__name__}")
    status = DecisionStatus.UNKNOWN_EFFECT if receipt.unknown_effect else DecisionStatus.DISPATCHED
    return _result(packet, status, decision_id, selected=selected.option_id, receipt=receipt, details={"action_id": selected.handle.action_id})


def action_handle(value: ActionHandle | Mapping[str, object], *, default_kind: str = "ACTION") -> ActionHandle:
    if isinstance(value, ActionHandle):
        return value
    if not isinstance(value, Mapping):
        raise ConsumerError("action handle must be an object")
    return ActionHandle(
        handle_id=_text(value.get("handle_id"), "handle_id", limit=256),
        action_id=_text(value.get("action_id"), "action_id", limit=256),
        kind=str(value.get("kind", default_kind)),
        language=str(value.get("language", "text")),
        revision=_text(value.get("revision"), "handle revision", limit=256),
        required_capabilities=tuple(value.get("required_capabilities", ())) if isinstance(value.get("required_capabilities", ()), Sequence) and not isinstance(value.get("required_capabilities"), (str, bytes, bytearray)) else (),
        required_roles=tuple(value.get("required_roles", ())) if isinstance(value.get("required_roles", ()), Sequence) and not isinstance(value.get("required_roles"), (str, bytes, bytearray)) else (),
        metadata=value.get("metadata", {}) if isinstance(value.get("metadata", {}), Mapping) else {},
        mutates_files=bool(value.get("mutates_files", False)),
        expires_at=value.get("expires_at"),
    )


def capability(value: Capability | Mapping[str, object] | str, *, default_name: str = "route") -> Capability:
    if isinstance(value, Capability):
        return value
    if isinstance(value, str):
        return Capability(default_name, value)
    if not isinstance(value, Mapping):
        raise ConsumerError("capability must be an object")
    return Capability(str(value.get("name", default_name)), str(value.get("state", CapabilityState.UNKNOWN.value)), str(value.get("detail", "")))


__all__ = [
    "ActionHandle",
    "AdapterResult",
    "Capability",
    "CapabilityState",
    "CandidateCoverage",
    "ConsumerError",
    "ConsumerMode",
    "ConsumerPacket",
    "DecisionStatus",
    "DispatchReceipt",
    "MAX_MENU_OPTIONS",
    "MenuOption",
    "PredicateState",
    "Provider",
    "UnresolvedPredicate",
    "action_handle",
    "capability",
    "public_digest",
    "reject_evaluator_material",
    "run_menu",
]
