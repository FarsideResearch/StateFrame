"""The narrow System One facade.

This module is an orchestration boundary, not a replacement contract owner.  It
combines the existing typed prompt, transport, registry, calibration, policy,
and journal lanes into one host-triggered attempt.  A proposal remains advisory
until an existing host owner rechecks its gates and consumes it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence

import kernel_py39_compat  # noqa: F401  # install v46's Python 3.9 shims first

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, Union, cast

try:  # Flat v46 module loading.
    from sys1.adapter import (
        ScoreTransportAdapter,
        TransportOutcome,
        build_chat_request,
        build_native_request,
    )
    from sys1.calibration import CalibrationArtifact, CalibratedDistribution
    from sys1.contracts import DecisionRecord, DecisionSite, QuestionType
    from sys1.policy import (
        GateSnapshot,
        LossSpec,
        PolicyResult,
        Proposal,
        SelectionContext,
        recheck_consumption,
        select_action,
    )
    from sys1.prompt import (
        BooleanSpec,
        ChoiceSpec,
        PromptPreparation,
        ScoreSpec,
        TemplateBoundary,
        prepare_boolean,
        prepare_choice,
        prepare_score,
    )
    from sys1.protocol import (
        QualifiedTemplate,
        ScoreDomain,
        ScoreRequest,
        TokenBinding,
        TransportKind,
        UsageStatus,
    )
    from sys1.records import (
        ExecutionLedger,
        FacadeReceipt,
        LedgerReservation,
        ReceiptAdapter,
        receipt_id,
    )
    from sys1.registry import DecisionSiteRegistry, emit_decision_recorded
except ImportError:  # pragma: no cover - package import path.
    from .sys1.adapter import ScoreTransportAdapter, TransportOutcome, build_chat_request, build_native_request
    from .sys1.calibration import CalibrationArtifact, CalibratedDistribution
    from .sys1.contracts import DecisionRecord, DecisionSite, QuestionType
    from .sys1.policy import (
        GateSnapshot,
        LossSpec,
        PolicyResult,
        Proposal,
        SelectionContext,
        recheck_consumption,
        select_action,
    )
    from .sys1.prompt import BooleanSpec, ChoiceSpec, PromptPreparation, ScoreSpec, TemplateBoundary, prepare_boolean, prepare_choice, prepare_score
    from .sys1.protocol import QualifiedTemplate, ScoreDomain, ScoreRequest, TokenBinding, TransportKind, UsageStatus
    from .sys1.records import ExecutionLedger, FacadeReceipt, LedgerReservation, ReceiptAdapter, receipt_id
    from .sys1.registry import DecisionSiteRegistry, emit_decision_recorded


TypedQuestion: TypeAlias = Union[ChoiceSpec, BooleanSpec, ScoreSpec]
TransportCallable: TypeAlias = Callable[[Mapping[str, object]], object]
WireWrapper: TypeAlias = Callable[[Mapping[str, object]], Mapping[str, object]]
OwnerCallable: TypeAlias = Callable[[Proposal], object]


class FacadeError(ValueError):
    """Raised by final-wire inspection or malformed facade configuration."""


class FacadeStatus(str, Enum):
    SKIPPED = "SKIPPED"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    DETERMINISTIC = "DETERMINISTIC"
    DEDUPED = "DEDUPED"
    PROPOSED = "PROPOSED"
    ABSTAINED = "ABSTAINED"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True, slots=True)
class SystemOneRequest:
    """One host-owned, typed request to the scored System One profile."""

    question: TypedQuestion
    public_snapshot: Mapping[str, object]
    policy_id: str
    budget: object | None = None
    site_id: str = ""
    site_version: str | None = None
    transport_kind: TransportKind = TransportKind.CHAT
    host_triggered: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.question, (ChoiceSpec, BooleanSpec, ScoreSpec)):
            raise FacadeError("question must be ChoiceSpec, BooleanSpec, or ScoreSpec")
        if not isinstance(self.public_snapshot, Mapping):
            raise FacadeError("public_snapshot must be a mapping")
        if not isinstance(self.policy_id, str) or not self.policy_id.strip():
            raise FacadeError("policy_id is required")
        if not isinstance(self.site_id, str):
            raise FacadeError("site_id must be text")
        if not isinstance(self.transport_kind, TransportKind):
            raise FacadeError("transport_kind must be TransportKind")
        if type(self.host_triggered) is not bool:
            raise FacadeError("host_triggered must be boolean")

    @property
    def decision_id(self) -> str:
        value = self.question.decision_id or self.public_snapshot.get("decision_id", "")
        return value if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class SystemOneResult:
    """Typed outcome of one attempt; ``proposal`` is never host authority."""

    status: FacadeStatus
    decision_id: str
    question_type: str | None
    proposal: Proposal | None
    policy_result: PolicyResult | None
    transport: TransportOutcome | None
    preparation: PromptPreparation | None
    final_request: Mapping[str, object] | None
    receipt: FacadeReceipt
    ledger_key: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "decision_id": self.decision_id,
            "question_type": self.question_type,
            "proposal": None if self.proposal is None else self.proposal.to_dict(),
            "policy_result": None if self.policy_result is None else self.policy_result.to_dict(),
            "transport": None if self.transport is None else self.transport.to_dict(),
            "preparation": None if self.preparation is None else self.preparation.to_dict(),
            "final_request": None if self.final_request is None else dict(self.final_request),
            "receipt": self.receipt.to_dict(),
            "ledger_key": self.ledger_key,
        }


@dataclass(frozen=True, slots=True)
class OwnerConsumeResult:
    """Result of the separate existing-owner consumption boundary."""

    accepted: bool
    owner_result: object | None
    policy_result: PolicyResult | None
    receipt: FacadeReceipt

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "owner_result": self.owner_result,
            "policy_result": None if self.policy_result is None else self.policy_result.to_dict(),
            "receipt": self.receipt.to_dict(),
        }


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _status_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return raw if isinstance(raw, str) else str(raw)


def _inspect_nested(value: object, *, path: str = "body") -> None:
    forbidden_markers = (
        "evaluator",
        "grader",
        "benchmark",
        "gold_answer",
        "answer_key",
        "future_outcome",
        "correct_answer",
        "confidence",
        "rationale",
    )
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FacadeError(f"{path} contains a non-text key")
            folded = key.casefold()
            if any(marker in folded for marker in forbidden_markers):
                raise FacadeError(f"final request contains forbidden field {path}.{key}")
            _inspect_nested(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _inspect_nested(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and any(marker in value.casefold() for marker in forbidden_markers):
        raise FacadeError(f"final request contains forbidden material at {path}")


def inspect_scored_wire(body: Mapping[str, object]) -> Mapping[str, object]:
    """Certify the final post-wrapper scored profile before dispatch."""

    if not isinstance(body, Mapping):
        raise FacadeError("final request must be a mapping")
    _inspect_nested(body)
    forbidden_fields = {"tools", "tool_choice", "response_format", "json_schema", "functions", "function_call"}
    present = forbidden_fields.intersection(body)
    if present:
        raise FacadeError(f"scored profile contains forbidden wrapper fields: {sorted(present)}")
    if body.get("max_tokens") != 1 or body.get("n_predict") != 1:
        raise FacadeError("scored profile must reserve exactly one completion token")
    if body.get("profile_kind") != "scored-decision" or body.get("completion_reserve") != 1:
        raise FacadeError("final request is not the purpose-specific scored-decision profile")
    if body.get("stream") is not False:
        raise FacadeError("scored profile must be non-streaming")
    if body.get("logprobs") is not True:
        raise FacadeError("scored profile must request logprobs")
    top_logprobs = body.get("top_logprobs")
    if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, int) or top_logprobs < 1:
        raise FacadeError("scored profile must declare positive top_logprobs")
    domain = body.get("score_domain")
    if domain not in {ScoreDomain.PRE_SAMPLER_LOGPROB.value, ScoreDomain.PRE_SAMPLER_LOGIT.value}:
        raise FacadeError("scored profile must declare a pre-sampler score domain")
    if body.get("post_sampling_probs") is not False:
        raise FacadeError("scored profile cannot request post-sampling probabilities")
    template = body.get("chat_template_kwargs")
    if not isinstance(template, Mapping) or template.get("enable_thinking") is not False:
        raise FacadeError("scored profile must disable thinking in the final template")
    if body.get("reasoning_effort") != "none" or body.get("reasoning_mode") != "off":
        raise FacadeError("scored profile must disable reasoning")
    if body.get("reasoning_budget_tokens") != 0:
        raise FacadeError("scored profile cannot reserve reasoning tokens")
    if "messages" not in body and "prompt" not in body:
        raise FacadeError("scored profile requires a rendered prompt")
    return MappingProxyType(dict(body))


def _question_type(question: TypedQuestion) -> QuestionType:
    if isinstance(question, ChoiceSpec):
        return QuestionType.CHOICE
    if isinstance(question, BooleanSpec):
        return QuestionType.BOOLEAN
    return QuestionType.SCORE


def _as_transport_kind(value: object) -> TransportKind:
    if isinstance(value, TransportKind):
        return value
    try:
        return TransportKind(str(value))
    except ValueError as error:
        raise FacadeError("unsupported transport kind") from error


class SystemOne:
    """Host-triggered typed scored-decision orchestration."""

    def __init__(
        self,
        *,
        registry: DecisionSiteRegistry,
        policies: Mapping[str, LossSpec] = MappingProxyType({}),
        calibrations: Mapping[str, CalibrationArtifact] = MappingProxyType({}),
        tokenizer: object,
        transport: object,
        journal: object | None = None,
        records: ReceiptAdapter | None = None,
        ledger: ExecutionLedger | None = None,
        wrappers: Sequence[WireWrapper] = (),
        boundary_renderer: Callable[[tuple[Mapping[str, str], ...]], TemplateBoundary | str | Mapping[str, object]] | None = None,
        model: str = "decision-tier",
        transport_kind: TransportKind = TransportKind.CHAT,
        retain_diagnostics: bool = False,
    ) -> None:
        if not isinstance(registry, DecisionSiteRegistry):
            raise FacadeError("registry must be DecisionSiteRegistry")
        if not isinstance(policies, Mapping) or not isinstance(calibrations, Mapping):
            raise FacadeError("policies and calibrations must be mappings")
        if not isinstance(model, str) or not model.strip():
            raise FacadeError("model is required")
        self.registry = registry
        self.policies = MappingProxyType(dict(policies))
        self.calibrations = MappingProxyType(dict(calibrations))
        self.tokenizer = tokenizer
        self.transport = transport
        self.records = records or ReceiptAdapter(journal)
        self.ledger = ledger or ExecutionLedger()
        self.wrappers = tuple(wrappers)
        self.boundary_renderer = boundary_renderer
        self.model = model
        self.transport_kind = _as_transport_kind(transport_kind)
        self.adapter = ScoreTransportAdapter(retain_diagnostics=retain_diagnostics)
        self._attempts: dict[str, int] = {}

    def _decision_id(self, request: SystemOneRequest) -> str:
        return request.decision_id or "unidentified"

    def _next_attempt(self, decision_id: str) -> int:
        attempt = self._attempts.get(decision_id, 0) + 1
        self._attempts[decision_id] = attempt
        return attempt

    def _key(self, request: SystemOneRequest, site: DecisionSite | None) -> str:
        question = request.question
        return _digest(
            {
                "decision_id": request.decision_id,
                "site": site.version_identity if site is not None else request.site_id,
                "policy_id": request.policy_id,
                "question_type": _question_type(question).value,
                "semantic_version": question.semantic_version,
            }
        )

    def _receipt(
        self,
        request: SystemOneRequest,
        status: str,
        *,
        dispatch_status: str,
        attempt: int,
        reason: str | None = None,
        request_sha256: str | None = None,
        wire_sha256: str | None = None,
        reserved_tokens: int = 0,
        observed_tokens: int | None = None,
        usage_status: str = UsageStatus.UNKNOWN.value,
        transport_status: str | None = None,
        proposal_status: str | None = None,
        owner_status: str | None = None,
    ) -> FacadeReceipt:
        return FacadeReceipt(
            receipt_id(receipt_decision := self._decision_id(request), attempt, status),
            receipt_decision,
            status,
            dispatch_status,
            reason,
            request_sha256,
            wire_sha256,
            reserved_tokens,
            observed_tokens,
            usage_status,
            transport_status,
            proposal_status,
            owner_status,
        )

    def _result(
        self,
        request: SystemOneRequest,
        status: FacadeStatus,
        receipt: FacadeReceipt,
        *,
        proposal: Proposal | None = None,
        policy_result: PolicyResult | None = None,
        transport: TransportOutcome | None = None,
        preparation: PromptPreparation | None = None,
        final_request: Mapping[str, object] | None = None,
        ledger_key: str | None = None,
    ) -> SystemOneResult:
        self.records.record(receipt)
        return SystemOneResult(
            status,
            self._decision_id(request),
            _question_type(request.question).value,
            proposal,
            policy_result,
            transport,
            preparation,
            final_request,
            receipt,
            ledger_key,
        )

    def _preflight_result(
        self,
        request: SystemOneRequest,
        status: str,
        reason: str,
        attempt: int,
        *,
        ledger_key: str | None = None,
    ) -> SystemOneResult:
        receipt = self._receipt(request, status, dispatch_status="NOT_DISPATCHED", attempt=attempt, reason=reason)
        return self._result(request, FacadeStatus.PREFLIGHT_FAILED, receipt, ledger_key=ledger_key)

    def _find_calibration(self, policy: LossSpec) -> CalibrationArtifact | None:
        direct = self.calibrations.get(policy.policy_id)
        if isinstance(direct, CalibrationArtifact):
            return direct
        direct = self.calibrations.get(policy.calibration_artifact_id)
        if isinstance(direct, CalibrationArtifact):
            return direct
        for artifact in self.calibrations.values():
            if isinstance(artifact, CalibrationArtifact) and artifact.artifact_id == policy.calibration_artifact_id:
                return artifact
        return None

    def _resolve_site(self, request: SystemOneRequest) -> DecisionSite:
        site_id = request.site_id or cast(str, request.public_snapshot.get("site_id", ""))
        if not site_id:
            raise FacadeError("site_id is required")
        site = self.registry.require(site_id, request.site_version)
        if site.type is not _question_type(request.question):
            raise FacadeError("typed question does not match registered site")
        return site

    def _resolve_policy(self, request: SystemOneRequest, site: DecisionSite) -> LossSpec:
        policy = self.policies.get(request.policy_id, site.loss_spec)
        if not isinstance(policy, LossSpec):
            raise FacadeError("policy owner did not provide LossSpec")
        if policy.policy_id != request.policy_id or policy.question_type != _question_type(request.question).value:
            raise FacadeError("policy identity or question type does not match request")
        return policy

    def _resolve_calibration(self, policy: LossSpec) -> CalibrationArtifact:
        artifact = self._find_calibration(policy)
        if artifact is None:
            raise FacadeError("calibration artifact is unavailable")
        if artifact.key_digest != policy.calibration_key_digest or artifact.artifact_id != policy.calibration_artifact_id:
            raise FacadeError("calibration identity does not match policy")
        return artifact

    def _prepare(self, question: TypedQuestion) -> PromptPreparation:
        kwargs: dict[str, object] = {
            "decision_nonce": question.decision_id or "sys1",
            "boundary_renderer": self.boundary_renderer,
        }
        if isinstance(question, ChoiceSpec):
            return prepare_choice(question, self.tokenizer, **kwargs)
        if isinstance(question, BooleanSpec):
            return prepare_boolean(question, self.tokenizer, **kwargs)
        return prepare_score(question, self.tokenizer, **kwargs)

    def _score_request(self, request: SystemOneRequest, preparation: PromptPreparation) -> ScoreRequest:
        labels = tuple(
            TokenBinding(
                option.semantic_id,
                option.label,
                option.token_id,
                tuple(option.token_bytes),
            )
            for option in preparation.options
        )
        return ScoreRequest(
            prompt=preparation.answer_boundary,
            labels=labels,
            template=QualifiedTemplate(
                "qualified:" + preparation.qualification.template_fingerprint,
                preparation.qualification.template_fingerprint,
                preparation.answer_boundary,
            ),
            model=self.model,
            seed=preparation.sampling.seed,
        )

    def _wire(self, preparation: PromptPreparation, score_request: ScoreRequest, kind: TransportKind) -> Mapping[str, object]:
        prepared_body = preparation.request
        if kind is TransportKind.CHAT:
            body = build_chat_request(score_request)
            # Keep the certified typed prompt, while reusing the provider-neutral
            # chat transport builder for all score/logprob fields.
            body["messages"] = prepared_body["messages"]
            body["n_predict"] = prepared_body["n_predict"]
            body["reasoning_effort"] = prepared_body["reasoning_effort"]
            body["reasoning_mode"] = prepared_body["reasoning_mode"]
            body["reasoning_budget_tokens"] = prepared_body["reasoning_budget_tokens"]
        else:
            body = build_native_request(score_request)
            body["chat_template_kwargs"] = {"enable_thinking": False}
            body["reasoning_effort"] = "none"
            body["reasoning_mode"] = "off"
            body["reasoning_budget_tokens"] = 0
        body["score_domain"] = ScoreDomain.PRE_SAMPLER_LOGPROB.value
        body["logprobs"] = True
        body["top_logprobs"] = score_request.top_logprobs
        body["post_sampling_probs"] = False
        body["completion_reserve"] = 1
        body["profile_kind"] = "scored-decision"
        for wrapper in self.wrappers:
            if not callable(wrapper):
                raise FacadeError("wire wrappers must be callable")
            wrapped = wrapper(MappingProxyType(dict(body)))
            if not isinstance(wrapped, Mapping):
                raise FacadeError("wire wrapper must return a mapping")
            body = dict(wrapped)
        return inspect_scored_wire(body)

    def _budget_reserve(self, budget: object | None, decision_id: str) -> bool:
        if budget is None:
            return True
        reserve = getattr(budget, "reserve", None)
        if callable(reserve):
            try:
                value = reserve(1, decision_id=decision_id, purpose="system_one")
            except TypeError:
                value = reserve(1)
            return value is not False
        if isinstance(budget, Mapping):
            available = budget.get("available", budget.get("shared_budget_available", True))
            return available is True
        if isinstance(budget, bool):
            return budget
        if isinstance(budget, int):
            return budget >= 1
        raise FacadeError("budget owner must expose reserve() or an availability mapping")

    def _budget_settle(self, budget: object | None, *, observed_tokens: int | None, reservation: LedgerReservation) -> None:
        if budget is None:
            return
        settle = getattr(budget, "settle", None)
        if not callable(settle):
            return
        try:
            settle(reservation.reservation_id, observed_tokens)
        except TypeError:
            settle(observed_tokens)

    def _invoke_transport(self, body: Mapping[str, object]) -> object:
        request = self.transport
        if callable(request):
            return request(body)
        for name in ("request", "send", "complete"):
            method = getattr(request, name, None)
            if callable(method):
                return method(body)
        raise FacadeError("transport owner must be callable or expose request/send/complete")

    def _gate_snapshot(self, snapshot: Mapping[str, object]) -> GateSnapshot:
        raw = snapshot.get("gates", snapshot)
        if not isinstance(raw, Mapping):
            raise FacadeError("public snapshot gates must be a mapping")
        required = ("freshness_token", "permission_token", "scope_token", "state_token", "budget_token")
        if any(name not in raw for name in required):
            raise FacadeError("public snapshot is missing host gate identity")
        values = {name: raw[name] for name in required}
        for name in ("receipt_fresh", "permission_granted", "scope_allowed", "state_current", "shared_budget_available"):
            if name in raw:
                values[name] = raw[name]
        return GateSnapshot(**values)

    def _selection_context(self, snapshot: Mapping[str, object], gates: GateSnapshot, policy: LossSpec) -> SelectionContext:
        actions = snapshot.get("admissible_actions", policy.actions)
        if isinstance(actions, (str, bytes, bytearray)) or not isinstance(actions, Sequence):
            raise FacadeError("admissible_actions must be a sequence")
        flags = {name: snapshot.get(name, True) for name in ("evidence_sufficient", "input_in_support")}
        if any(type(value) is not bool for value in flags.values()):
            raise FacadeError("selection flags must be boolean")
        return SelectionContext(
            tuple(str(action) for action in actions),
            gates,
            evidence_sufficient=cast(bool, flags["evidence_sufficient"]),
            input_in_support=cast(bool, flags["input_in_support"]),
            transport_available=True,
        )

    def _record_identity(self, request: SystemOneRequest, site: DecisionSite, preparation: PromptPreparation) -> None:
        record = DecisionRecord(
            decision_id=self._decision_id(request),
            site_id=site.site_id,
            site_version_identity=site.version_identity,
            question_id=site.question_id,
            question_version=site.question_version,
            question_type=site.type,
            schema_id=site.schema_id,
            schema_version=site.schema_version,
            evidence_version=site.evidence_version,
            procedure_version=site.procedure_version,
            payload={
                "prompt_fingerprint": preparation.prompt_fingerprint,
                "schema_fingerprint": preparation.schema_fingerprint,
                "typed_question": preparation.kind,
            },
        )
        journal = self.records.journal
        if journal is not None:
            emit_decision_recorded(journal, record)
        else:
            self.records.append("decision_recorded", record=record.to_dict())

    def decide(
        self,
        request: SystemOneRequest | TypedQuestion,
        public_snapshot: Mapping[str, object] | None = None,
        policy_id: str | None = None,
        budget: object | None = None,
        *,
        site_id: str = "",
        site_version: str | None = None,
        host_triggered: bool = True,
        transport_kind: TransportKind | None = None,
    ) -> SystemOneResult:
        """Run one host-triggered packet; no internal retry or effect occurs."""

        if isinstance(request, (ChoiceSpec, BooleanSpec, ScoreSpec)):
            if public_snapshot is None or policy_id is None:
                raise FacadeError("typed decide calls require public_snapshot and policy_id")
            request = SystemOneRequest(
                request,
                public_snapshot,
                policy_id,
                budget,
                site_id,
                site_version,
                transport_kind or self.transport_kind,
                host_triggered,
            )
        if not isinstance(request, SystemOneRequest):
            raise FacadeError("request must be SystemOneRequest or a typed question")
        if not request.host_triggered:
            attempt = self._next_attempt(self._decision_id(request))
            receipt = self._receipt(request, FacadeStatus.SKIPPED.value, dispatch_status="NOT_DISPATCHED", attempt=attempt, reason="HOST_TRIGGER_REQUIRED")
            return self._result(request, FacadeStatus.SKIPPED, receipt)

        attempt = self._next_attempt(self._decision_id(request))
        site: DecisionSite | None = None
        key = self._key(request, site)
        reservation = self.ledger.begin(key)
        if reservation is None:
            completed = self.ledger.completed_receipt(key)
            receipt = self._receipt(request, FacadeStatus.DEDUPED.value, dispatch_status="NOT_DISPATCHED", attempt=attempt, reason="COMPLETED" if completed else "IN_FLIGHT")
            if completed is not None:
                receipt = FacadeReceipt(completed, self._decision_id(request), FacadeStatus.DEDUPED.value, "NOT_DISPATCHED", "COMPLETED")
            return self._result(request, FacadeStatus.DEDUPED, receipt, ledger_key=key)

        try:
            site = self._resolve_site(request)
            key = self._key(request, site)
            if key != reservation.key:
                self.ledger.release(reservation)
                reservation = self.ledger.begin(key)
                if reservation is None:
                    receipt = self._receipt(request, FacadeStatus.DEDUPED.value, dispatch_status="NOT_DISPATCHED", attempt=attempt, reason="COMPLETED")
                    return self._result(request, FacadeStatus.DEDUPED, receipt, ledger_key=key)
            deterministic = self.registry.resolve(site.site_id, site.site_version)
            if deterministic.status.value == "DETERMINISTIC":
                self.ledger.release(reservation)
                receipt = self._receipt(request, FacadeStatus.DETERMINISTIC.value, dispatch_status="NOT_DISPATCHED", attempt=attempt, reason=deterministic.reason)
                return self._result(request, FacadeStatus.DETERMINISTIC, receipt, ledger_key=key)
            policy = self._resolve_policy(request, site)
            artifact = self._resolve_calibration(policy)
            if request.question.answerability != "SUFFICIENT" or request.question.menu_status != "CLOSED":
                raise FacadeError("question is not calibration-eligible")
            preparation = self._prepare(request.question)
            score_request = self._score_request(request, preparation)
            final_request = self._wire(preparation, score_request, request.transport_kind)
            request_sha = _digest(preparation.to_dict())
            wire_sha = _digest(dict(final_request))
            self._record_identity(request, site, preparation)
            if not self._budget_reserve(request.budget, self._decision_id(request)):
                self.ledger.release(reservation)
                receipt = self._receipt(request, FacadeStatus.UNAVAILABLE.value, dispatch_status="NOT_DISPATCHED", attempt=attempt, reason="SHARED_BUDGET_REFUSED", request_sha256=request_sha, wire_sha256=wire_sha)
                return self._result(request, FacadeStatus.UNAVAILABLE, receipt, preparation=preparation, final_request=final_request, ledger_key=key)
        except Exception as error:
            self.ledger.release(reservation)
            return self._preflight_result(request, FacadeStatus.PREFLIGHT_FAILED.value, type(error).__name__, attempt, ledger_key=key)

        dispatched = True
        try:
            response = self._invoke_transport(final_request)
            transport = self.adapter.adapt(score_request, response, kind=request.transport_kind)
        except Exception as error:
            transport = self.adapter.adapt(score_request, {}, kind=request.transport_kind, error=error)
        observed = transport.usage.completion_tokens if transport.usage.status is UsageStatus.KNOWN else None
        usage_status = transport.usage.status.value
        try:
            self._budget_settle(request.budget, observed_tokens=observed, reservation=reservation)
        except Exception as error:
            self.ledger.release(reservation)
            receipt = self._receipt(
                request,
                FacadeStatus.UNAVAILABLE.value,
                dispatch_status="DISPATCHED",
                attempt=attempt,
                reason=type(error).__name__,
                request_sha256=request_sha,
                wire_sha256=wire_sha,
                reserved_tokens=reservation.reserved_tokens,
                observed_tokens=observed,
                usage_status=usage_status,
                transport_status=transport.status.value,
            )
            return self._result(request, FacadeStatus.UNAVAILABLE, receipt, transport=transport, preparation=preparation, final_request=final_request, ledger_key=key)
        if not transport.admitted:
            self.ledger.release(reservation)
            status = FacadeStatus.UNAVAILABLE
            reason = ",".join(transport.diagnostics.codes) or transport.status.value
            receipt = self._receipt(
                request,
                status.value,
                dispatch_status="DISPATCHED" if dispatched else "NOT_DISPATCHED",
                attempt=attempt,
                reason=reason,
                request_sha256=request_sha,
                wire_sha256=wire_sha,
                reserved_tokens=reservation.reserved_tokens,
                observed_tokens=observed,
                usage_status=usage_status,
                transport_status=transport.status.value,
            )
            return self._result(request, status, receipt, transport=transport, preparation=preparation, final_request=final_request, ledger_key=key)

        try:
            raw_scores = {
                label.semantic_id: score
                for label, score in zip(score_request.labels, transport.score_vector.raw_scores if transport.score_vector else ())
            }
            distribution: CalibratedDistribution = artifact.apply(raw_scores, key=artifact.key, labels=tuple(raw_scores))
            gates = self._gate_snapshot(request.public_snapshot)
            context = self._selection_context(request.public_snapshot, gates, policy)
            policy_result = select_action(distribution, policy, context, legal_mass=transport.normalized.legal_mass if transport.normalized else None)
            policy_status = _status_value(policy_result.status)
            if isinstance(policy_result, Proposal):
                status = FacadeStatus.PROPOSED
                proposal = policy_result
            elif policy_status == "STALE":
                status = FacadeStatus.STALE
                proposal = None
            elif policy_status == "ABSTAIN":
                status = FacadeStatus.ABSTAINED
                proposal = None
            else:
                status = FacadeStatus.UNAVAILABLE
                proposal = None
            final_receipt = self._receipt(
                request,
                status.value,
                dispatch_status="DISPATCHED",
                attempt=attempt,
                request_sha256=request_sha,
                wire_sha256=wire_sha,
                reserved_tokens=reservation.reserved_tokens,
                observed_tokens=observed,
                usage_status=usage_status,
                transport_status=transport.status.value,
                proposal_status=policy_status,
            )
            self.ledger.complete(reservation, final_receipt.receipt_id)
            return self._result(request, status, final_receipt, proposal=proposal, policy_result=policy_result, transport=transport, preparation=preparation, final_request=final_request, ledger_key=key)
        except Exception as error:
            self.ledger.release(reservation)
            receipt = self._receipt(request, FacadeStatus.UNAVAILABLE.value, dispatch_status="DISPATCHED", attempt=attempt, reason=type(error).__name__, request_sha256=request_sha, wire_sha256=wire_sha, reserved_tokens=reservation.reserved_tokens, observed_tokens=observed, usage_status=usage_status, transport_status=transport.status.value)
            return self._result(request, FacadeStatus.UNAVAILABLE, receipt, transport=transport, preparation=preparation, final_request=final_request, ledger_key=key)

    def consume(
        self,
        result: SystemOneResult,
        current_snapshot: Mapping[str, object],
        owner: OwnerCallable | None,
    ) -> OwnerConsumeResult:
        """Recheck host gates, then hand an advisory proposal to its owner."""

        if result.proposal is None:
            receipt = FacadeReceipt(
                receipt_id(result.decision_id, 0, "OWNERLESS_SKIP"),
                result.decision_id,
                "OWNERLESS_SKIP",
                "NOT_DISPATCHED",
                "NO_PROPOSAL",
                owner_status="NOT_CONSUMED",
            )
            self.records.record(receipt)
            return OwnerConsumeResult(False, None, result.policy_result, receipt)
        try:
            gates = self._gate_snapshot(current_snapshot)
            check = recheck_consumption(result.proposal, gates)
        except Exception as error:
            check = None
            reason = type(error).__name__
        else:
            reason = None
        if check is None or not check.accepted:
            receipt = FacadeReceipt(
                receipt_id(result.decision_id, 0, "OWNER_RECHECK_FAILED"),
                result.decision_id,
                "OWNER_RECHECK_FAILED",
                "NOT_DISPATCHED",
                reason or "HOST_GATE_RECHECK_FAILED",
                proposal_status=_status_value(check.result.status) if check is not None else None,
                owner_status="NOT_CONSUMED",
            )
            self.records.record(receipt)
            return OwnerConsumeResult(False, None, check.result if check is not None else result.policy_result, receipt)
        if owner is None:
            receipt = FacadeReceipt(
                receipt_id(result.decision_id, 0, "OWNERLESS_SKIP"),
                result.decision_id,
                "OWNERLESS_SKIP",
                "NOT_DISPATCHED",
                "OWNER_REQUIRED",
                proposal_status=_status_value(check.result.status),
                owner_status="NOT_CONSUMED",
            )
            self.records.record(receipt)
            return OwnerConsumeResult(False, None, check.result, receipt)
        try:
            owner_result = owner(result.proposal)
        except Exception as error:
            receipt = FacadeReceipt(
                receipt_id(result.decision_id, 0, "OWNER_FAILED"),
                result.decision_id,
                "OWNER_FAILED",
                "DISPATCHED",
                type(error).__name__,
                proposal_status=_status_value(check.result.status),
                owner_status="FAILED",
            )
            self.records.record(receipt)
            return OwnerConsumeResult(False, None, check.result, receipt)
        receipt = FacadeReceipt(
            receipt_id(result.decision_id, 0, "CONSUMED"),
            result.decision_id,
            "CONSUMED",
            "DISPATCHED",
            owner_status="CONSUMED",
        )
        self.records.record(receipt)
        return OwnerConsumeResult(True, owner_result, check.result, receipt)


__all__ = [
    "BooleanSpec",
    "ChoiceSpec",
    "FacadeError",
    "FacadeStatus",
    "OwnerConsumeResult",
    "ScoreSpec",
    "SystemOne",
    "SystemOneRequest",
    "SystemOneResult",
    "TypedQuestion",
    "inspect_scored_wire",
]
