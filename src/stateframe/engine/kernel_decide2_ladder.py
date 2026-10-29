"""Adaptive, host-owned DECIDE2 escalation ladder.

The ladder is cumulative and stops at the first typed answer that clears the
host gate.  It exposes L1-L8 only; the diagnostic L9/L10 tails from the
FORCELADDER experiment are intentionally not production API.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence

from kernel_decide2 import (
    ALLOWED_LADDER_RUNGS,
    ABSTAIN,
    DecisionFrame,
    DecisionRequest,
    ParsedAnswer,
    Provider,
    build_frame,
    build_micro_request,
    build_work_request,
    combine_subanswers,
    normalize_tool_surface,
    parse_answer,
    raw_output_text,
    safe_error,
)
from kernel_launch_identity_cost import (
    ProviderBoundaryContractError,
    ReasoningProfile,
    enforce_provider_body,
    reasoning_profile as resolve_reasoning_profile,
)


RUNG_BUDGETS: dict[str, int] = {
    "L1": 384,
    "L2": 256,
    "L3": 256,
    "L4": 256,
    "L5": 512,
    "L6": 256,
    "L7": 256,
    "L8": 256,
}
MAX_CALLS = 12
MAX_GENERATED_TOKENS = 3500


@dataclass(frozen=True, slots=True)
class RungReceipt:
    rung: str
    help_kind: str
    status: str
    reasoning_mode: str
    input_hash: str
    provider_calls: int
    generated_tokens: int
    reasoning_tokens: int
    answer: str | None = None
    confidence: float | None = None
    error: str | None = None
    skip_reason: str | None = None
    raw_outputs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-decide2-rung/1",
            "rung": self.rung,
            "help_kind": self.help_kind,
            "status": self.status,
            "reasoning_mode": self.reasoning_mode,
            "input_hash": self.input_hash,
            "provider_calls": self.provider_calls,
            "generated_tokens": self.generated_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "answer": self.answer,
            "confidence": self.confidence,
            "error": self.error,
            "skip_reason": self.skip_reason,
            "raw_outputs": list(self.raw_outputs),
        }


@dataclass(frozen=True, slots=True)
class LadderOutcome:
    answer: ParsedAnswer
    frame: DecisionFrame
    arm: str
    rung: str | None
    triggers: tuple[str, ...]
    rung_receipts: tuple[RungReceipt, ...]
    provider_calls: int
    generated_tokens: int
    reasoning_tokens: int


def _digest_frame(frame: DecisionFrame, extra: str) -> str:
    from kernel_decide2 import _digest  # local private helper keeps one canonical hash

    return _digest(
        {
            "decision_id": frame.request.decision_id,
            "payload": frame.to_payload(),
            "extra": extra,
        }
    )


def _extract_usage(
    raw: object, default: int, reasoning_default: int = 0
) -> tuple[int, int]:
    """Read optional public usage telemetry; fall back to the bounded request cap."""
    usage: object = raw.get("usage") if isinstance(raw, Mapping) else None
    if not isinstance(usage, Mapping):
        return default, reasoning_default
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    reasoning = usage.get(
        "reasoning_tokens", usage.get("output_reasoning_tokens", reasoning_default)
    )
    completion_value = (
        completion if isinstance(completion, int) and completion >= 0 else default
    )
    reasoning_value = (
        reasoning
        if isinstance(reasoning, int) and reasoning >= 0
        else reasoning_default
    )
    return min(completion_value + reasoning_value, default), min(
        reasoning_value, default
    )


def _option_order(options: Sequence[str], seed: int, vote: int) -> tuple[str, ...]:
    result = list(options)
    if vote:
        random.Random(seed + vote * 7919).shuffle(result)
    return tuple(result)


def _winner(
    options: Sequence[str], answers: Sequence[str | None], *, required: int
) -> str | None:
    counts = {option: 0 for option in options}
    for answer in answers:
        if answer in counts and answer != ABSTAIN:
            counts[answer] += 1
    candidates = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    if not candidates or candidates[0][1] < required:
        return None
    if len(candidates) > 1 and candidates[0][1] == candidates[1][1]:
        return None
    return candidates[0][0]


class AdaptiveForceLadder:
    """Run the measured L1-L8 path with explicit host budgets and receipts."""

    def __init__(
        self,
        request: DecisionRequest,
        provider: Provider,
        *,
        seed: int = 1,
        max_calls: int = MAX_CALLS,
        max_tokens: int = MAX_GENERATED_TOKENS,
        tool_surface: Sequence[Mapping[str, object]] | None = None,
        reasoning_profile: ReasoningProfile | None = None,
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        if max_calls <= 0 or max_tokens <= 0:
            raise ValueError("ladder budgets must be positive")
        self.request = request
        self.provider = provider
        self.seed = int(seed)
        self.max_calls = min(int(max_calls), MAX_CALLS)
        self.max_tokens = min(int(max_tokens), MAX_GENERATED_TOKENS)
        self.tool_surface = normalize_tool_surface(tool_surface)
        self.reasoning_profile = reasoning_profile or resolve_reasoning_profile()
        self.provider_calls = 0
        self.generated_tokens = 0
        self.reasoning_tokens = 0
        self.receipts: list[RungReceipt] = []

    def run(self, *, quality_fallback: bool = False) -> LadderOutcome:
        frame = build_frame(self.request, arm="D1")
        initial = self._call_one(
            "L1",
            "decision_procedure",
            frame,
            extra="L1 procedure: apply definitions, criteria, tie-break, and abstain conditions.",
        )
        current = initial
        triggers = self._triggers(current)
        if current.shippable and not triggers:
            return self._outcome(current, frame, "D1", "L1", triggers)
        if current.status == "host_contract_error":
            return self._outcome(current, frame, "D1", "L1", triggers)
        if current.status == "transport_error":
            retry, retry_triggers = self._transport_recovery(frame)
            return self._outcome(
                retry,
                frame,
                "D1",
                "L1",
                (*triggers, "transport_retry", *retry_triggers),
            )
        if quality_fallback:
            vote_answer, vote_receipt = self._vote(
                "E4", frame, votes=3, strict_majority=2
            )
            self.receipts.append(vote_receipt)
            if vote_answer.status == "host_contract_error":
                return self._outcome(
                    vote_answer, replace(frame, arm="E4"), "E4", "E4", triggers
                )
            if vote_answer.status == "transport_error":
                retry, retry_triggers = self._transport_recovery(frame)
                return self._outcome(
                    retry,
                    frame,
                    "D1",
                    "L1",
                    (*triggers, "transport_retry", *retry_triggers),
                )
            if vote_answer.shippable:
                return self._outcome(
                    vote_answer,
                    replace(frame, arm="E4"),
                    "E4",
                    "E4",
                    (*triggers, "quality_fallback"),
                )
            return self._outcome(
                vote_answer,
                replace(frame, arm="E4"),
                "E4",
                "E4",
                (*triggers, "quality_fallback_failed"),
            )

        for rung in ("L2", "L3", "L4", "L5", "L6", "L7", "L8"):
            applicable, reason = self._precondition(rung)
            if not applicable:
                self.receipts.append(self._skipped(rung, reason))
                continue
            if rung == "L8":
                current, receipt = self._vote(rung, frame, votes=5, strict_majority=3)
                self.receipts.append(receipt)
            else:
                if rung == "L5":
                    current = self._bounded_reasoning(
                        frame,
                        "L5 BOUNDED_REASONING: public frame is complete; reason within a 512-token ceiling.",
                    )
                else:
                    help_kind, extra = self._help(rung)
                    call_frame = frame
                    if rung == "L3":
                        call_frame = replace(
                            frame, evidence=self.request.pinned_evidence
                        )
                    current = self._call_one(rung, help_kind, call_frame, extra=extra)
            triggers = tuple(dict.fromkeys((*triggers, *self._triggers(current))))
            if current.status == "transport_error":
                retry, retry_triggers = self._transport_recovery(frame)
                return self._outcome(
                    retry,
                    frame,
                    "D1",
                    "L1",
                    (*triggers, "transport_retry", *retry_triggers),
                )
            if current.status == "host_contract_error":
                return self._outcome(current, frame, "D1", rung, triggers)
            if current.shippable:
                return self._outcome(current, frame, "D1", rung, triggers)
        return self._outcome(
            current,
            frame,
            "D1",
            self.receipts[-1].rung if self.receipts else None,
            triggers,
        )

    def _transport_recovery(
        self, frame: DecisionFrame
    ) -> tuple[ParsedAnswer, tuple[str, ...]]:
        """Spend at most one post-failure call, then fail closed to ABSTAIN."""
        retry = self._call_one(
            "L1",
            "transport_retry",
            frame,
            extra="L1 transport retry: retry once, then return ABSTAIN/escalation without ladder fan-out.",
        )
        if not retry.shippable:
            retry = replace(retry, answer=ABSTAIN, status="abstain")
        return retry, self._triggers(retry)

    def _outcome(
        self,
        answer: ParsedAnswer,
        frame: DecisionFrame,
        arm: str,
        rung: str | None,
        triggers: Sequence[str],
    ) -> LadderOutcome:
        return LadderOutcome(
            answer=answer,
            frame=frame,
            arm=arm,
            rung=rung,
            triggers=tuple(dict.fromkeys(str(item) for item in triggers if item)),
            rung_receipts=tuple(self.receipts),
            provider_calls=self.provider_calls,
            generated_tokens=self.generated_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )

    def _triggers(self, answer: ParsedAnswer) -> tuple[str, ...]:
        triggers: list[str] = []
        if answer.status == "abstain" or answer.answer == ABSTAIN:
            triggers.append("abstain")
        elif answer.status == "transport_error":
            triggers.append("transport_invalid")
        elif answer.status == "host_contract_error":
            triggers.append("host_contract")
        elif answer.status in {"invalid", "subquestions"}:
            triggers.append("malformed_typed_answer")
        if answer.confidence is None or (
            answer.confidence is not None and answer.confidence < 0.75
        ):
            triggers.append("low_confidence")
        if self.request.known_hard:
            triggers.append("known_hard_decision_kind")
        if self.request.distractor_risk:
            triggers.append("distractor_risk")
        return tuple(dict.fromkeys(triggers))

    def _precondition(self, rung: str) -> tuple[bool, str]:
        if rung == "L2":
            return bool(self.request.pinned_evidence), "no host decisive fact card"
        if rung == "L3":
            eligible = (
                self.request.distractor_risk
                and self.request.public_relevance_rule
                and len(self.request.evidence) > len(self.request.pinned_evidence)
            )
            return (
                eligible,
                "requires public relevance rule plus distractor-risk marker",
            )
        if rung == "L4":
            return 2 <= len(
                self.request.subquestions
            ) <= 4, "requires 2-4 host-authored subquestions"
        if rung == "L5":
            return (
                self.request.evidence_complete and not self.request.contradictory,
                "public frame is incomplete or contradictory",
            )
        if rung == "L6":
            return (
                self.request.contrastive_pair is not None,
                "no two live alternatives remain",
            )
        if rung == "L7":
            case = self.request.solved_case
            return bool(
                case and case.case_id != self.request.current_case_id
            ), "no nearest same-kind solved case"
        if rung == "L8":
            return True, ""
        raise ValueError(f"unsupported production rung: {rung}")

    def _help(self, rung: str) -> tuple[str, str]:
        if rung == "L2":
            facts = "; ".join(
                f"{item.item_id}: {item.text}" for item in self.request.pinned_evidence
            )
            return "fact_card", f"L2 DECISIVE_FACT_CARD: {facts}"
        if rung == "L3":
            return (
                "distractor_removal",
                "L3 DISTRACTOR_REMOVAL: exclude non-pinned items under the public relevance rule.",
            )
        if rung == "L4":
            return "decomposition", "L4 SUBQUESTIONS: " + str(
                [item.to_prompt() for item in self.request.subquestions]
            )
        if rung == "L6":
            return (
                "contrastive_pair",
                f"L6 CONTRASTIVE_PAIR: {self.request.contrastive_pair}",
            )
        if rung == "L7":
            return (
                "solved_case",
                f"L7 SOLVED_CASE: {self.request.solved_case.to_prompt() if self.request.solved_case else {}}",
            )
        raise ValueError(f"no help text for {rung}")

    def _skipped(self, rung: str, reason: str) -> RungReceipt:
        return RungReceipt(
            rung=rung,
            help_kind="not_applicable",
            status="skipped",
            reasoning_mode="off",
            input_hash="",
            provider_calls=0,
            generated_tokens=0,
            reasoning_tokens=0,
            skip_reason=reason,
        )

    def _reserve(self, calls: int, tokens: int) -> bool:
        return (
            self.provider_calls + calls <= self.max_calls
            and self.generated_tokens + tokens <= self.max_tokens
        )

    def _call_one(
        self, rung: str, help_kind: str, frame: DecisionFrame, *, extra: str
    ) -> ParsedAnswer:
        if rung not in ALLOWED_LADDER_RUNGS:
            raise ValueError(f"unsupported production rung: {rung}")
        budget = RUNG_BUDGETS[rung]
        if not self._reserve(1, budget):
            answer = ParsedAnswer(None, None, (), "invalid", "ladder budget exhausted")
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind=help_kind,
                    status="budget_exhausted",
                    reasoning_mode="off",
                    input_hash=_digest_frame(frame, extra),
                    provider_calls=0,
                    generated_tokens=0,
                    reasoning_tokens=0,
                    error="ladder budget exhausted",
                )
            )
            return answer
        updated = replace(frame, extra_help=extra)
        body = build_micro_request(
            updated,
            rung=rung,
            max_tokens=budget,
            tool_surface=self.tool_surface,
        )
        self.provider_calls += 1
        self.generated_tokens += budget
        raw_outputs: list[str] = []
        try:
            raw = self.provider(enforce_provider_body(body, profile=self.reasoning_profile))
            raw_text = raw_output_text(raw)
            if raw_text is not None:
                raw_outputs.append(raw_text)
            used, reasoning = _extract_usage(raw, budget)
            self.reasoning_tokens += reasoning
            answer = parse_answer(
                raw,
                self.request.options,
                visible_ids=tuple(item.item_id for item in updated.evidence),
            )
            if rung == "L4":
                answer = self._reduce_subanswers(answer)
            status = "accepted" if answer.shippable else answer.status
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind=help_kind,
                    status=status,
                    reasoning_mode="off",
                    input_hash=_digest_frame(updated, extra),
                    provider_calls=1,
                    generated_tokens=used,
                    reasoning_tokens=reasoning,
                    answer=answer.answer,
                    confidence=answer.confidence,
                    error=answer.error,
                    raw_outputs=tuple(raw_outputs),
                )
            )
            return answer
        except ProviderBoundaryContractError as error:
            answer = ParsedAnswer(None, None, (), "host_contract_error", str(error))
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind=help_kind,
                    status="host_contract_error",
                    reasoning_mode=self.reasoning_profile.mode,
                    input_hash=_digest_frame(updated, extra),
                    provider_calls=1,
                    generated_tokens=budget,
                    reasoning_tokens=0,
                    error=answer.error,
                )
            )
            return answer
        except Exception as error:  # provider failures are typed ladder failures
            answer = ParsedAnswer(None, None, (), "transport_error", safe_error(error))
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind=help_kind,
                    status="transport_error",
                    reasoning_mode="off",
                    input_hash=_digest_frame(updated, extra),
                    provider_calls=1,
                    generated_tokens=budget,
                    reasoning_tokens=0,
                    error=answer.error,
                    raw_outputs=tuple(raw_outputs),
                )
            )
            return answer

    def _reduce_subanswers(self, answer: ParsedAnswer) -> ParsedAnswer:
        """Reduce only host-declared child answers into an offered parent option."""
        if answer.status != "subquestions":
            return answer
        combined = combine_subanswers(self.request, dict(answer.subanswers))
        if combined is None:
            return replace(
                answer,
                status="invalid",
                error="host could not combine typed subanswers",
            )
        if combined == ABSTAIN:
            return replace(answer, answer=ABSTAIN, status="abstain")
        return replace(answer, answer=combined, status="answered")

    def _bounded_reasoning(self, frame: DecisionFrame, extra: str) -> ParsedAnswer:
        rung = "L5"
        budget = RUNG_BUDGETS[rung]
        if not self._reserve(1, budget):
            answer = ParsedAnswer(None, None, (), "invalid", "ladder budget exhausted")
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind="bounded_reasoning",
                    status="budget_exhausted",
                    reasoning_mode=self.reasoning_profile.mode,
                    input_hash=_digest_frame(frame, extra),
                    provider_calls=0,
                    generated_tokens=0,
                    reasoning_tokens=0,
                    error=answer.error,
                )
            )
            return answer
        body = build_work_request(self.request, reason=extra, max_tokens=budget)
        body["rung"] = rung
        body["reasoning_mode"] = "bounded_ladder"
        body["messages"] = [
            {
                "role": "user",
                "content": str(body["messages"][0]["content"]) + "\n" + frame.procedure,
            }
        ]
        self.provider_calls += 1
        self.generated_tokens += budget
        raw_outputs: list[str] = []
        try:
            raw = self.provider(enforce_provider_body(body, profile=self.reasoning_profile))
            raw_text = raw_output_text(raw)
            if raw_text is not None:
                raw_outputs.append(raw_text)
            used, reasoning = _extract_usage(raw, budget, budget)
            self.reasoning_tokens += reasoning
            answer = parse_answer(
                raw,
                self.request.options,
                visible_ids=self.request.visible_evidence_ids,
            )
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind="bounded_reasoning",
                    status="accepted" if answer.shippable else answer.status,
                    reasoning_mode=self.reasoning_profile.mode,
                    input_hash=_digest_frame(frame, extra),
                    provider_calls=1,
                    generated_tokens=used,
                    reasoning_tokens=reasoning,
                    answer=answer.answer,
                    confidence=answer.confidence,
                    error=answer.error,
                    raw_outputs=tuple(raw_outputs),
                )
            )
            return answer
        except ProviderBoundaryContractError as error:
            answer = ParsedAnswer(None, None, (), "host_contract_error", str(error))
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind="bounded_reasoning",
                    status="host_contract_error",
                    reasoning_mode=self.reasoning_profile.mode,
                    input_hash=_digest_frame(frame, extra),
                    provider_calls=1,
                    generated_tokens=budget,
                    reasoning_tokens=0,
                    error=answer.error,
                )
            )
            return answer
        except Exception as error:
            answer = ParsedAnswer(None, None, (), "transport_error", safe_error(error))
            self.receipts.append(
                RungReceipt(
                    rung=rung,
                    help_kind="bounded_reasoning",
                    status="transport_error",
                    reasoning_mode=self.reasoning_profile.mode,
                    input_hash=_digest_frame(frame, extra),
                    provider_calls=1,
                    generated_tokens=budget,
                    reasoning_tokens=budget,
                    error=answer.error,
                    raw_outputs=tuple(raw_outputs),
                )
            )
            return answer

    def _vote(
        self, rung: str, frame: DecisionFrame, *, votes: int, strict_majority: int
    ) -> tuple[ParsedAnswer, RungReceipt]:
        budget = RUNG_BUDGETS[rung] if rung in RUNG_BUDGETS else 256
        if not self._reserve(votes, votes * budget):
            answer = ParsedAnswer(None, None, (), "invalid", "ladder budget exhausted")
            return answer, RungReceipt(
                rung=rung,
                help_kind="shuffle_votes" if rung == "E4" else "five_vote_majority",
                status="budget_exhausted",
                reasoning_mode="off",
                input_hash=_digest_frame(frame, rung),
                provider_calls=0,
                generated_tokens=0,
                reasoning_tokens=0,
                error=answer.error,
            )
        answers: list[ParsedAnswer] = []
        raw_outputs: list[str] = []
        transport_error: ParsedAnswer | None = None
        for vote in range(votes):
            order = _option_order(self.request.options, self.seed, vote)
            updated = replace(
                frame,
                arm="E4" if rung == "E4" else frame.arm,
                extra_help=f"{rung} independent vote {vote + 1}/{votes}",
            )
            body = build_micro_request(
                updated,
                rung=rung,
                max_tokens=budget,
                option_order=order,
                tool_surface=self.tool_surface,
            )
            body["vote_index"] = vote
            body["vote_seed"] = self.seed + vote * 7919
            self.provider_calls += 1
            self.generated_tokens += budget
            try:
                raw = self.provider(
                    enforce_provider_body(body, profile=self.reasoning_profile)
                )
                raw_text = raw_output_text(raw)
                if raw_text is not None:
                    raw_outputs.append(raw_text)
                used, reasoning = _extract_usage(raw, budget)
                self.reasoning_tokens += reasoning
                answers.append(
                    parse_answer(
                        raw,
                        self.request.options,
                        visible_ids=tuple(item.item_id for item in updated.evidence),
                    )
                )
            except ProviderBoundaryContractError as error:
                transport_error = ParsedAnswer(
                    None, None, (), "host_contract_error", str(error)
                )
                answers.append(transport_error)
                break
            except Exception as error:
                transport_error = ParsedAnswer(
                    None, None, (), "transport_error", safe_error(error)
                )
                answers.append(transport_error)
                break
        if transport_error is not None:
            return transport_error, RungReceipt(
                rung=rung,
                help_kind="shuffle_votes" if rung == "E4" else "five_vote_majority",
                status=("host_contract_error"
                        if transport_error.status == "host_contract_error"
                        else "transport_error"),
                reasoning_mode="off",
                input_hash=_digest_frame(frame, rung),
                provider_calls=len(answers),
                generated_tokens=len(answers) * budget,
                reasoning_tokens=0,
                error=transport_error.error,
                raw_outputs=tuple(raw_outputs),
            )
        winner = _winner(
            self.request.options,
            [item.answer for item in answers],
            required=strict_majority,
        )
        confidences = [
            item.confidence
            for item in answers
            if item.answer == winner and item.confidence is not None
        ]
        confidence = min(confidences) if winner is not None and confidences else None
        if winner is None:
            result = ParsedAnswer(
                ABSTAIN, confidence, (), "abstain", "no strict host majority"
            )
            status = "abstain"
        else:
            result = ParsedAnswer(winner, confidence, (), "answered")
            status = "accepted" if result.shippable else "low_confidence"
        return result, RungReceipt(
            rung=rung,
            help_kind="shuffle_votes" if rung == "E4" else "five_vote_majority",
            status=status,
            reasoning_mode="off",
            input_hash=_digest_frame(frame, rung),
            provider_calls=votes,
            generated_tokens=votes * budget,
            reasoning_tokens=0,
            answer=result.answer,
            confidence=result.confidence,
            error=result.error,
            raw_outputs=tuple(raw_outputs),
        )


__all__ = [
    "AdaptiveForceLadder",
    "LadderOutcome",
    "MAX_CALLS",
    "MAX_GENERATED_TOKENS",
    "RUNG_BUDGETS",
    "RungReceipt",
]
