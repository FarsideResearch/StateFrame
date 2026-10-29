"""Host-triggered, bounded decision tier for the V46 kernel.

DECIDE2 is deliberately a proposal service, not an action authority.  A host
owner creates an episode when a typed decision is due; the model receives a
small pinned frame and can only return one offered option plus confidence.
Low-confidence, malformed, contradictory, terminal, and destructive cases
remain host-routed to the work frame.  The legacy advisory ``decide`` tool is
not parsed or invoked by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


SCHEMA = "stateframe-decide2/1"
PROCEDURE_VERSION = "decide2-procedure/v1"
MODE_ENV = "STATEFRAME_DECIDE2"
MICRO_CONFIDENCE_FLOOR = 0.75
MAX_REASON_CHARS = 200
MAX_EVIDENCE = 8
MAX_EVIDENCE_CHARS = 520
MAX_OPTIONS = 8
MICRO_MAX_TOKENS = 384
RAW_OUTPUT_MAX_CHARS = 32_000
DECIDE2_ANSWER_TOOL_NAME = "decide2_answer"

DecisionKind = Literal[
    "next_localization_action",
    "next_evidence_acquisition",
    "recall_item_selection",
    "safe_relevance",
    "next_stage",
    "file_or_cause",
    "relevance",
    "reproduced",
    "done",
    "trustworthy",
    "scope",
    "submit",
    "certify",
    "terminal",
    "delete",
    "deploy",
    "publish",
    "drop",
    "overwrite",
    "force_push",
    "kill",
]
RiskClass = Literal["R0", "R1", "R2", "R3"]
Route = Literal["HOST", "MICRO", "WORK_FRAME"]

MICRO_DECISION_KINDS = frozenset(
    {
        "next_localization_action",
        "next_evidence_acquisition",
        "recall_item_selection",
        "safe_relevance",
    }
)
TERMINAL_DECISION_KINDS = frozenset(
    {
        "done",
        "trustworthy",
        "submit",
        "certify",
        "terminal",
        "delete",
        "deploy",
    }
)
DESTRUCTIVE_EFFECT_KINDS = frozenset(
    {
        "delete",
        "deploy",
        "publish",
        "drop",
        "overwrite",
        "force_push",
        "kill",
    }
)
VALID_DECISION_KINDS = (
    MICRO_DECISION_KINDS
    | frozenset({"next_stage", "file_or_cause", "relevance", "reproduced", "scope"})
    | TERMINAL_DECISION_KINDS
    | DESTRUCTIVE_EFFECT_KINDS
)
ALLOWED_LADDER_RUNGS = tuple(f"L{index}" for index in range(1, 9))
ABSTAIN = "ABSTAIN"


class Decide2Error(ValueError):
    """Raised when a DECIDE2 host/model contract is not safe to use."""


def _compact(value: object, field: str, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str):
        if not required and value is None:
            return ""
        raise Decide2Error(f"{field} must be a string")
    result = " ".join(value.split())
    if required and not result:
        raise Decide2Error(f"{field} must not be empty")
    if len(result) > limit:
        raise Decide2Error(f"{field} exceeds {limit} characters")
    return result


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def raw_output_text(raw: object) -> str | None:
    """Persist a bounded provider result for format diagnosis without its request."""
    if raw is None:
        return None
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    elif isinstance(raw, str):
        text = raw
    else:
        text = json.dumps(
            raw, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")
        )
    if len(text) <= RAW_OUTPUT_MAX_CHARS:
        return text
    return text[:RAW_OUTPUT_MAX_CHARS] + "...[raw output truncated]"


def _float(value: object, field: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Decide2Error(f"{field} must be a number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise Decide2Error(f"{field} must be within [0, 1]")
    return result


def _tuple_strings(
    value: object, field: str, *, limit: int, max_items: int
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise Decide2Error(f"{field} must be a list or tuple")
    result = tuple(
        _compact(item, f"{field}[{index}]", limit) for index, item in enumerate(value)
    )
    if len(result) > max_items:
        raise Decide2Error(f"{field} may contain at most {max_items} items")
    if len(set(result)) != len(result):
        raise Decide2Error(f"{field} must contain unique items")
    return result


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return the explicit lane switch; the standalone contract remains usable OFF."""
    source = os.environ if environ is None else environ
    return str(source.get(MODE_ENV, "off")).strip().lower() in {
        "on",
        "1",
        "true",
        "yes",
    }


def run_system_one_mode(
    facade: object,
    question: object,
    public_snapshot: Mapping[str, object],
    policy_id: str,
    budget: object | None = None,
    *,
    mode: str = "system_one",
    site_id: str = "",
    site_version: str | None = None,
) -> object:
    """Enter the scored profile only through its explicit scoped mode.

    Legacy ``build_micro_request``/``parse_answer`` callers never use this
    adapter, so the fixed DECIDE2 answer-tool contract remains unchanged.
    """

    if mode != "system_one":
        raise Decide2Error("System One scored mode requires mode='system_one'")
    decide = getattr(facade, "decide", None)
    if not callable(decide):
        raise Decide2Error("System One facade must provide decide")
    return decide(
        question,
        public_snapshot,
        policy_id,
        budget,
        site_id=site_id,
        site_version=site_version,
        host_triggered=True,
    )


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """Host-owned public evidence that may be pinned into a micro frame."""

    item_id: str
    text: str
    kind: str = "receipt"
    status: str = "current"

    def __post_init__(self) -> None:
        _compact(self.item_id, "evidence.item_id", 180)
        _compact(self.text, "evidence.text", MAX_EVIDENCE_CHARS)
        _compact(self.kind, "evidence.kind", 80)
        _compact(self.status, "evidence.status", 80)

    def to_prompt(self) -> dict[str, str]:
        return {
            "id": self.item_id,
            "kind": self.kind,
            "status": self.status,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class SubQuestion:
    """A host-authored L4 sub-question and its parent-option reduction."""

    question_id: str
    question: str
    options: tuple[str, ...]
    parent_by_answer: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _compact(self.question_id, "subquestion.question_id", 80)
        _compact(self.question, "subquestion.question", 240)
        options = _tuple_strings(
            self.options, "subquestion.options", limit=120, max_items=8
        )
        if len(options) < 2:
            raise Decide2Error("subquestion.options needs at least two options")
        if len(self.parent_by_answer) > len(options):
            raise Decide2Error("subquestion.parent_by_answer is too large")
        for child, parent in self.parent_by_answer:
            if child not in options or not parent:
                raise Decide2Error("subquestion mapping must use offered child options")

    def to_prompt(self) -> dict[str, object]:
        return {
            "id": self.question_id,
            "question": self.question,
            "options": list(self.options),
        }


@dataclass(frozen=True, slots=True)
class SolvedCase:
    """One public same-kind solved example; evaluator material is never accepted."""

    case_id: str
    question_kind: str
    summary: str
    answer: str

    def __post_init__(self) -> None:
        _compact(self.case_id, "solved_case.case_id", 120)
        _compact(self.question_kind, "solved_case.question_kind", 100)
        _compact(self.summary, "solved_case.summary", 420)
        _compact(self.answer, "solved_case.answer", 120)

    def to_prompt(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "question_kind": self.question_kind,
            "summary": self.summary,
            "answer": self.answer,
        }


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """A host-created decision episode; model-triggered episodes are rejected."""

    decision_id: str
    question_kind: DecisionKind | str
    question: str
    options: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...] = ()
    pinned_ids: tuple[str, ...] = ()
    model_selected_ids: tuple[str, ...] = ()
    trigger_reason: str = ""
    host_triggered: bool = True
    risk_class: RiskClass | str = "R1"
    terminal: bool = False
    destructive: bool = False
    evidence_complete: bool = True
    contradictory: bool = False
    distractor_risk: bool = False
    public_relevance_rule: bool = False
    known_hard: bool = False
    chain_confidence: float | None = None
    criteria_count: int = 0
    subquestions: tuple[SubQuestion, ...] = ()
    contrastive_pair: tuple[str, str] | None = None
    solved_case: SolvedCase | None = None
    current_case_id: str = ""

    def __post_init__(self) -> None:
        _compact(self.decision_id, "decision_id", 120)
        kind = _compact(self.question_kind, "question_kind", 100).lower()
        if kind not in VALID_DECISION_KINDS:
            raise Decide2Error(f"unknown decision kind: {kind}")
        _compact(self.question, "question", 600)
        options = _tuple_strings(
            self.options, "options", limit=160, max_items=MAX_OPTIONS
        )
        if len(options) < 2:
            raise Decide2Error("options needs at least two entries")
        if len(set(options)) != len(options):
            raise Decide2Error("options must be unique")
        if ABSTAIN not in options:
            raise Decide2Error("options must include ABSTAIN")
        if self.risk_class not in {"R0", "R1", "R2", "R3"}:
            raise Decide2Error("risk_class must be R0, R1, R2 or R3")
        evidence_ids = tuple(item.item_id for item in self.evidence)
        if len(evidence_ids) > MAX_EVIDENCE:
            raise Decide2Error(f"evidence may contain at most {MAX_EVIDENCE} items")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise Decide2Error("evidence ids must be unique")
        pins = _tuple_strings(
            self.pinned_ids, "pinned_ids", limit=180, max_items=MAX_EVIDENCE
        )
        if not set(pins).issubset(evidence_ids):
            raise Decide2Error("pinned_ids must identify supplied evidence")
        model_ids = _tuple_strings(
            self.model_selected_ids,
            "model_selected_ids",
            limit=180,
            max_items=MAX_EVIDENCE,
        )
        if not set(model_ids).issubset(evidence_ids):
            raise Decide2Error("model_selected_ids must identify supplied evidence")
        _compact(self.trigger_reason, "trigger_reason", 240, required=False)
        if self.chain_confidence is not None:
            _float(self.chain_confidence, "chain_confidence")
        if (
            not isinstance(self.criteria_count, int)
            or self.criteria_count < 0
            or self.criteria_count > 8
        ):
            raise Decide2Error("criteria_count must be an integer from 0 to 8")
        if len(self.subquestions) > 4:
            raise Decide2Error("at most four subquestions are allowed")
        if self.contrastive_pair is not None:
            pair = _tuple_strings(
                self.contrastive_pair, "contrastive_pair", limit=180, max_items=2
            )
            if len(pair) != 2 or pair[0] == pair[1] or not set(pair).issubset(options):
                raise Decide2Error(
                    "contrastive_pair must contain two distinct offered options"
                )
        if self.solved_case is not None and self.solved_case.question_kind != kind:
            raise Decide2Error("solved_case must match question_kind")
        _compact(self.current_case_id, "current_case_id", 120, required=False)

    @property
    def normalized_kind(self) -> str:
        return str(self.question_kind).strip().lower()

    @property
    def pinned_evidence(self) -> tuple[EvidenceItem, ...]:
        wanted = set(self.pinned_ids)
        return tuple(item for item in self.evidence if item.item_id in wanted)

    @property
    def decision_evidence(self) -> tuple[tuple[EvidenceItem, str], ...]:
        """Return host pins plus validated model picks, without host selectors."""
        pinned = set(self.pinned_ids)
        model = set(self.model_selected_ids)
        return tuple(
            (item, "pin" if item.item_id in pinned else "model")
            for item in self.evidence
            if item.item_id in pinned or item.item_id in model
        )

    @property
    def visible_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item, _ in self.decision_evidence)

    @property
    def terminal_or_destructive(self) -> bool:
        return (
            self.terminal
            or self.destructive
            or self.risk_class == "R3"
            or self.normalized_kind in TERMINAL_DECISION_KINDS
            or self.normalized_kind in DESTRUCTIVE_EFFECT_KINDS
        )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    arm: str
    reason: str
    triggers: tuple[str, ...]
    reasoning_off: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "arm": self.arm,
            "reason": self.reason,
            "triggers": list(self.triggers),
            "reasoning_off": self.reasoning_off,
        }


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    answer: str | None
    confidence: float | None
    evidence_ids: tuple[str, ...]
    status: Literal[
        "answered", "abstain", "invalid", "transport_error",
        "host_contract_error", "subquestions",
    ]
    error: str | None = None
    subanswers: tuple[tuple[str, str], ...] = ()

    @property
    def shippable(self) -> bool:
        return (
            self.status == "answered"
            and self.answer is not None
            and self.answer != ABSTAIN
            and self.confidence is not None
            and self.confidence >= MICRO_CONFIDENCE_FLOOR
        )


@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    decision_id: str
    status: Literal[
        "accepted", "abstain", "invalid", "transport_error",
        "host_contract_error", "escalated", "skipped"
    ]
    route: Route
    arm: str
    answer: str | None
    confidence: float | None
    evidence_ids: tuple[str, ...]
    trigger_reasons: tuple[str, ...]
    provider_calls: int
    generated_tokens: int
    reasoning_tokens: int
    rung: str | None = None
    escalation: "EscalationReceipt | None" = None
    rung_receipts: tuple[Mapping[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "decision_id": self.decision_id,
            "status": self.status,
            "route": self.route,
            "arm": self.arm,
            "answer": self.answer,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "trigger_reasons": list(self.trigger_reasons),
            "provider_calls": self.provider_calls,
            "generated_tokens": self.generated_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "rung": self.rung,
            "escalation": self.escalation.to_dict() if self.escalation else None,
            "rung_receipts": [dict(row) for row in self.rung_receipts],
        }


@dataclass(frozen=True, slots=True)
class EscalationReceipt:
    decision_id: str
    from_route: str
    to_route: str
    triggers: tuple[str, ...]
    help_rungs: tuple[str, ...]
    input_hash: str
    provider_calls: int
    generated_tokens: int
    reasoning_tokens: int
    status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-decide2-escalation/1",
            "decision_id": self.decision_id,
            "from_route": self.from_route,
            "to_route": self.to_route,
            "triggers": list(self.triggers),
            "help_rungs": list(self.help_rungs),
            "input_hash": self.input_hash,
            "provider_calls": self.provider_calls,
            "generated_tokens": self.generated_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class DecisionFrame:
    request: DecisionRequest
    arm: str
    evidence: tuple[EvidenceItem, ...]
    procedure: str
    extra_help: str = ""

    def to_payload(
        self, *, option_order: Sequence[str] | None = None
    ) -> dict[str, object]:
        options = tuple(option_order or self.request.options)
        reasons = {
            item.item_id: reason for item, reason in self.request.decision_evidence
        }
        frame_ids = {item.item_id for item in self.evidence}
        pinned_ids = set(self.request.pinned_ids)
        evidence_policy = (
            "host pins only; automatic extra evidence is disabled"
            if frame_ids.issubset(pinned_ids)
            else "host pins plus validated model picks; automatic extra evidence is disabled"
        )
        return {
            "schema": SCHEMA,
            "task": "Answer one host-triggered typed decision; do not call decide or take an action.",
            "decision_id": self.request.decision_id,
            "question_kind": self.request.normalized_kind,
            "question": self.request.question,
            "options": list(options),
            "evidence": [
                {
                    **item.to_prompt(),
                    "selection_reason": reasons.get(item.item_id, "pin"),
                }
                for item in self.evidence
            ],
            "evidence_policy": evidence_policy,
            "procedure": self.procedure,
            "help": self.extra_help,
            "output": {
                "format": "one JSON object only; no markdown, explanation, or thinking text",
                "schema": {
                    "type": "object",
                    "required": ["answer", "confidence"],
                    "properties": {
                        "answer": {"type": "string", "enum": list(options)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_EVIDENCE,
                        },
                    },
                    "additionalProperties": False,
                },
            },
        }


@dataclass(frozen=True, slots=True)
class DecisionResult:
    receipt: DecisionReceipt
    frame: DecisionFrame | None
    work_frame: Mapping[str, object] | None


def route_request(request: DecisionRequest) -> RouteDecision:
    """Apply host-owned routing before any model call is possible."""
    triggers: list[str] = []
    if not request.host_triggered:
        triggers.append("host_trigger_required")
    if not request.trigger_reason:
        triggers.append("missing_host_trigger_reason")
    if request.terminal_or_destructive:
        triggers.append("terminal_or_destructive")
    if request.normalized_kind not in MICRO_DECISION_KINDS:
        triggers.append("decision_kind_not_micro_qualified")
    if not request.evidence_complete:
        triggers.append("missing_decisive_evidence")
    if request.contradictory:
        triggers.append("contradictory_evidence")
    if (
        request.chain_confidence is not None
        and request.chain_confidence < MICRO_CONFIDENCE_FLOOR
    ):
        triggers.append("low_propagated_confidence")
    if triggers:
        return RouteDecision(
            "WORK_FRAME", "WORK", "; ".join(triggers), tuple(triggers), False
        )
    return RouteDecision(
        "MICRO", "D1", "host-triggered reversible micro decision", (), True
    )


def host_trigger(request: DecisionRequest) -> RouteDecision:
    """Explicit host entry point; there is intentionally no model trigger API."""
    return route_request(request)


def procedure_text(request: DecisionRequest) -> str:
    """Render a compact, host-authored D1 procedure without hidden labels."""
    definitions = ", ".join(f"{option}=offered option" for option in request.options)
    return (
        f"{PROCEDURE_VERSION} kind={request.normalized_kind}; "
        f"OPTIONS {definitions}; "
        "RULE match only visible pin/model evidence to definitions; "
        "TIE acquire or ABSTAIN; "
        "ABSTAIN_IF missing, stale, contradictory, or insufficient decisive evidence"
    )


def build_frame(
    request: DecisionRequest, *, arm: str = "D1", extra_help: str = ""
) -> DecisionFrame:
    """Build a pin/model-picked frame; host selectors never enter the prompt."""
    if arm not in {"D1", "E4"}:
        raise Decide2Error(f"unknown DECIDE2 frame arm: {arm}")
    return DecisionFrame(
        request=request,
        arm=arm,
        evidence=tuple(item for item, _ in request.decision_evidence),
        procedure=procedure_text(request),
        extra_help=_compact(extra_help, "extra_help", 1200, required=False),
    )


def apply_thinking_off(body: Mapping[str, object]) -> dict[str, object]:
    """Apply the exact measured no-thinking switch to a request copy."""
    if not isinstance(body, Mapping):
        raise TypeError("provider request must be a mapping")
    result = copy.deepcopy(dict(body))
    raw_kwargs = result.get("chat_template_kwargs")
    if raw_kwargs is None:
        kwargs: dict[str, object] = {}
    elif isinstance(raw_kwargs, Mapping):
        kwargs = copy.deepcopy(dict(raw_kwargs))
    else:
        raise Decide2Error("chat_template_kwargs must be a mapping")
    kwargs["enable_thinking"] = False
    result["chat_template_kwargs"] = kwargs
    result["reasoning_effort"] = "none"
    result["reasoning_budget_tokens"] = 0
    result["reasoning_mode"] = "off"
    return result


def normalize_tool_surface(
    value: Sequence[Mapping[str, object]] | None,
) -> tuple[Mapping[str, object], ...] | None:
    """Validate and freeze one host-admitted function schema surface."""
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise Decide2Error("tool surface must be a schema list or None")
    if not value:
        raise Decide2Error("tool surface must not be empty")
    result: list[Mapping[str, object]] = []
    names: set[str] = set()
    for index, tool in enumerate(value):
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            raise Decide2Error(f"tool surface item {index} is not a function schema")
        function = tool.get("function")
        if not isinstance(function, Mapping):
            raise Decide2Error(f"tool surface item {index} has no function schema")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise Decide2Error(f"tool surface item {index} has no function name")
        if name in names:
            raise Decide2Error(f"tool surface contains duplicate function {name!r}")
        names.add(name)
        result.append(copy.deepcopy(dict(tool)))
    return tuple(result)


def build_micro_request(
    frame: DecisionFrame,
    *,
    rung: str,
    max_tokens: int = MICRO_MAX_TOKENS,
    option_order: Sequence[str] | None = None,
    tool_surface: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build a provider body for D1/E4/L2-L4/L6-L8 OFF calls."""
    if rung not in ALLOWED_LADDER_RUNGS and rung != "E4":
        raise Decide2Error(f"unsupported production rung: {rung}")
    if max_tokens <= 0:
        raise Decide2Error("max_tokens must be positive")
    options = tuple(option_order or frame.request.options)
    visible_ids = tuple(item.item_id for item in frame.evidence)
    answer_properties: dict[str, object] = {
        "answer": {
            "type": "string",
            "enum": list(options),
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "evidence_ids": {
            "type": "array",
            "items": {
                "type": "string",
                **({"enum": list(visible_ids)} if visible_ids else {}),
            },
            "maxItems": MAX_EVIDENCE,
        },
    }
    required = ["confidence"]
    if rung != "L4":
        required.insert(0, "answer")
    else:
        answer_properties["subanswers"] = {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }
    answer_tool = {
        "type": "function",
        "function": {
            "name": DECIDE2_ANSWER_TOOL_NAME,
            "description": "Return one bounded DECIDE2 answer; do not call an action tool.",
            "parameters": {
                "type": "object",
                "properties": answer_properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }
    body: dict[str, object] = {
        "model": "decision-tier",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    frame.to_payload(option_order=option_order),
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
        ],
        # No response_format here: llama.cpp cannot combine a json_object grammar with the forced
        # tool-call grammar and rejects the request with HTTP 400 "failed to parse grammar"
        # (L107 10-05: 22/27 live requests). The forced answer tool already constrains the output.
        "temperature": 0,
        "max_tokens": max_tokens,
        "tools": [answer_tool],
        "tool_choice": {
            "type": "function",
            "function": {"name": DECIDE2_ANSWER_TOOL_NAME},
        },
        "rung": rung,
        "decision_id": frame.request.decision_id,
    }
    # Validate the frozen owner surface at the host boundary, but never expose
    # action tools to a proposal-only DECIDE2 call.  The dedicated answer tool
    # prevents an A3B response from being mistaken for an executable action.
    normalize_tool_surface(tool_surface)
    return apply_thinking_off(body)


def build_work_request(
    request: DecisionRequest,
    *,
    reason: str,
    max_tokens: int = 512,
) -> dict[str, object]:
    """Build a reasoning-on work frame after micro routing is exhausted."""
    body: dict[str, object] = {
        "model": "work-frame",
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "schema": SCHEMA,
                        "task": "Resolve the host-escalated decision using bounded public evidence.",
                        "decision_id": request.decision_id,
                        "question_kind": request.normalized_kind,
                        "question": request.question,
                        "options": list(request.options),
                        "decision_evidence": [
                            {**item.to_prompt(), "selection_reason": reason}
                            for item, reason in request.decision_evidence
                        ],
                        "escalation_reason": reason,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_budget_tokens": max_tokens,
        "reasoning_mode": "work",
        "chat_template_kwargs": {"enable_thinking": True},
    }
    from kernel_launch_identity_cost import enforce_provider_body, reasoning_profile
    return enforce_provider_body(body, profile=reasoning_profile())


_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _embedded_json(text: str) -> Mapping[str, object] | None:
    """Find one JSON object amid markdown, thinking markers, or trailing prose."""
    candidates = [match.group(1) for match in _FENCED_JSON.finditer(text)]
    candidates.append(_THINK_BLOCK.sub("", text))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        source = candidate.strip()
        if not source:
            continue
        try:
            decoded = json.loads(source)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, Mapping):
            return decoded
        for index, character in enumerate(source):
            if character not in "{[":
                continue
            try:
                decoded, _end = decoder.raw_decode(source[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, Mapping):
                return decoded
    return None


def _answer_object(raw: object) -> Mapping[str, object] | None:
    """Unwrap common OpenAI-compatible response shapes before parsing fields."""
    if isinstance(raw, Mapping):
        if any(key in raw for key in ("answer", "option", "decision", "selected_option", "subanswers")):
            return raw
        for key in ("response", "output", "message"):
            nested = raw.get(key)
            if nested is not None:
                result = _answer_object(nested)
                if result is not None:
                    return result
        choices = raw.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                result = _answer_object(choice)
                if result is not None:
                    return result
        tool_calls = raw.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, Mapping):
                    continue
                function = call.get("function")
                if isinstance(function, Mapping):
                    result = _answer_object(function.get("arguments"))
                    if result is not None:
                        return result
        for key in ("content", "reasoning_content", "text"):
            nested = raw.get(key)
            if nested is not None:
                result = _answer_object(nested)
                if result is not None:
                    return result
        return None
    if isinstance(raw, list):
        text = "".join(
            str(item.get("text", ""))
            for item in raw
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        )
        return _embedded_json(text)
    if isinstance(raw, bytes):
        return _embedded_json(raw.decode("utf-8", errors="replace"))
    if isinstance(raw, str):
        return _embedded_json(raw)
    return None


def _parse_evidence_ids(value: object, visible_ids: set[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    ids = _tuple_strings(
        value, "answer.evidence_ids", limit=180, max_items=MAX_EVIDENCE
    )
    if not set(ids).issubset(visible_ids):
        raise Decide2Error("answer cited evidence outside visible decision ids")
    return ids


def parse_answer(
    raw: object, options: Sequence[str], *, visible_ids: Sequence[str] = ()
) -> ParsedAnswer:
    """Parse only the bounded JSON answer; prose and unknown options are invalid."""
    obj = _answer_object(raw)
    if obj is None:
        return ParsedAnswer(None, None, (), "invalid", "answer was not a JSON object")
    answer = obj.get(
        "answer", obj.get("option", obj.get("decision", obj.get("selected_option")))
    )
    confidence = obj.get("confidence")
    try:
        evidence_ids = _parse_evidence_ids(obj.get("evidence_ids"), set(visible_ids))
    except Decide2Error as error:
        return ParsedAnswer(None, None, (), "invalid", str(error))
    subanswers: list[tuple[str, str]] = []
    raw_subanswers = obj.get("subanswers")
    if isinstance(raw_subanswers, Mapping):
        for key, value in raw_subanswers.items():
            if isinstance(key, str) and isinstance(value, str):
                subanswers.append((key, value))
    elif isinstance(raw_subanswers, list):
        for row in raw_subanswers:
            if (
                isinstance(row, Mapping)
                and isinstance(row.get("id"), str)
                and isinstance(row.get("answer"), str)
            ):
                subanswers.append((row["id"], row["answer"]))
    if answer is None and subanswers:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            return ParsedAnswer(
                None,
                None,
                evidence_ids,
                "invalid",
                "confidence was missing or non-numeric",
                tuple(subanswers),
            )
        try:
            confidence_value = _float(confidence, "confidence")
        except Decide2Error as error:
            return ParsedAnswer(
                None, None, evidence_ids, "invalid", str(error), tuple(subanswers)
            )
        return ParsedAnswer(
            None,
            confidence_value,
            evidence_ids,
            "subquestions",
            subanswers=tuple(subanswers),
        )
    if not isinstance(answer, str) or answer not in options:
        return ParsedAnswer(
            None, None, evidence_ids, "invalid", "answer was not an offered option"
        )
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return ParsedAnswer(
            answer,
            None,
            evidence_ids,
            "invalid",
            "confidence was missing or non-numeric",
        )
    try:
        confidence_value = _float(confidence, "confidence")
    except Decide2Error as error:
        return ParsedAnswer(answer, None, evidence_ids, "invalid", str(error))
    status: Literal["answered", "abstain"] = (
        "abstain" if answer == ABSTAIN else "answered"
    )
    return ParsedAnswer(
        answer, confidence_value, evidence_ids, status, subanswers=tuple(subanswers)
    )


def combine_subanswers(
    request: DecisionRequest, answers: Mapping[str, str]
) -> str | None:
    """Host-reduce L4 child answers; the model cannot invent a parent option."""
    if not request.subquestions or set(answers) != {
        item.question_id for item in request.subquestions
    }:
        return None
    parent_answers: list[str] = []
    for question in request.subquestions:
        value = answers.get(question.question_id)
        if value not in question.options:
            return None
        if value == ABSTAIN:
            return ABSTAIN
        mapping = dict(question.parent_by_answer)
        parent = mapping.get(value)
        if parent not in request.options:
            return None
        parent_answers.append(parent)
    if not parent_answers or len(set(parent_answers)) != 1:
        return ABSTAIN
    return parent_answers[0]


def safe_error(error: Exception) -> str:
    first = str(error).splitlines()[0].strip() if str(error).strip() else "<no message>"
    return f"{type(error).__name__}: {first[:180]}"


def _request_hash(request: DecisionRequest) -> str:
    return _digest(
        {
            "decision_id": request.decision_id,
            "question_kind": request.normalized_kind,
            "question": request.question,
            "options": list(request.options),
            "evidence": [item.to_prompt() for item in request.evidence],
            "pinned_ids": list(request.pinned_ids),
            "model_selected_ids": list(request.model_selected_ids),
        }
    )


def preflight_skip_result(request: DecisionRequest, reason: str) -> DecisionResult:
    """Return a typed zero-call receipt when a live request cannot be built."""
    bounded_reason = _compact(reason, "preflight_skip_reason", 240)
    rung_receipt = {
        "schema": "stateframe-decide2-rung/1",
        "rung": "PREFLIGHT",
        "help_kind": "host_preflight",
        "status": "skipped",
        "reasoning_mode": "off",
        "input_hash": _request_hash(request),
        "provider_calls": 0,
        "generated_tokens": 0,
        "reasoning_tokens": 0,
        "answer": ABSTAIN,
        "confidence": None,
        "error": None,
        "skip_reason": bounded_reason,
    }
    receipt = DecisionReceipt(
        decision_id=request.decision_id,
        status="skipped",
        route="HOST",
        arm="PREFLIGHT",
        answer=ABSTAIN,
        confidence=None,
        evidence_ids=request.visible_evidence_ids,
        trigger_reasons=("invalid_request_contract",),
        provider_calls=0,
        generated_tokens=0,
        reasoning_tokens=0,
        rung="PREFLIGHT",
        rung_receipts=(rung_receipt,),
    )
    return DecisionResult(receipt, None, None)


Provider = Callable[[Mapping[str, object]], object]


class DecisionTier:
    """Host-owned DECIDE2 runner with D1 default, E4 option, and work fallback."""

    def __init__(
        self,
        provider: Provider,
        *,
        seed: int = 1,
        max_calls: int = 12,
        max_tokens: int = 3500,
        tool_surface: Sequence[Mapping[str, object]] | None = None,
    ):
        if not callable(provider):
            raise TypeError("provider must be callable")
        if max_calls <= 0 or max_tokens <= 0:
            raise ValueError("DECIDE2 budgets must be positive")
        self.provider = provider
        self.seed = int(seed)
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.tool_surface = normalize_tool_surface(tool_surface)
        from kernel_launch_identity_cost import reasoning_profile
        self.reasoning_profile = reasoning_profile()

    def run(
        self, request: DecisionRequest, *, quality_fallback: bool = False
    ) -> DecisionResult:
        """Run a host-triggered episode; no model output can create a new episode."""
        route = host_trigger(request)
        if route.route != "MICRO":
            return self._work_result(
                request, route, provider_calls=0, generated_tokens=0, rung_receipts=()
            )
        from kernel_decide2_ladder import AdaptiveForceLadder

        ladder = AdaptiveForceLadder(
            request,
            self.provider,
            seed=self.seed,
            max_calls=self.max_calls,
            max_tokens=self.max_tokens,
            tool_surface=self.tool_surface,
            reasoning_profile=self.reasoning_profile,
        )
        outcome = ladder.run(quality_fallback=quality_fallback)
        receipts = tuple(item.to_dict() for item in outcome.rung_receipts)
        if outcome.answer.shippable:
            receipt = DecisionReceipt(
                decision_id=request.decision_id,
                status="accepted",
                route="MICRO",
                arm=outcome.arm,
                answer=outcome.answer.answer,
                confidence=outcome.answer.confidence,
                evidence_ids=outcome.answer.evidence_ids,
                trigger_reasons=outcome.triggers,
                provider_calls=outcome.provider_calls,
                generated_tokens=outcome.generated_tokens,
                reasoning_tokens=outcome.reasoning_tokens,
                rung=outcome.rung,
                rung_receipts=receipts,
            )
            return DecisionResult(receipt, outcome.frame, None)
        if outcome.answer.status == "host_contract_error":
            receipt = DecisionReceipt(
                decision_id=request.decision_id,
                status="host_contract_error",
                route="MICRO",
                arm=outcome.arm,
                answer=None,
                confidence=None,
                evidence_ids=request.visible_evidence_ids,
                trigger_reasons=(*outcome.triggers, "host_contract"),
                provider_calls=outcome.provider_calls,
                generated_tokens=outcome.generated_tokens,
                reasoning_tokens=outcome.reasoning_tokens,
                rung=outcome.rung,
                rung_receipts=receipts,
            )
            return DecisionResult(receipt, outcome.frame, None)
        escalation = EscalationReceipt(
            decision_id=request.decision_id,
            from_route="MICRO",
            to_route="WORK_FRAME",
            triggers=outcome.triggers,
            help_rungs=tuple(
                item.rung for item in outcome.rung_receipts if item.status != "skipped"
            ),
            input_hash=_request_hash(request),
            provider_calls=outcome.provider_calls,
            generated_tokens=outcome.generated_tokens,
            reasoning_tokens=outcome.reasoning_tokens,
            status="escalated",
        )
        receipt = DecisionReceipt(
            decision_id=request.decision_id,
            status="escalated",
            route="WORK_FRAME",
            arm=outcome.arm,
            answer=outcome.answer.answer,
            confidence=outcome.answer.confidence,
            evidence_ids=outcome.answer.evidence_ids,
            trigger_reasons=outcome.triggers,
            provider_calls=outcome.provider_calls,
            generated_tokens=outcome.generated_tokens,
            reasoning_tokens=outcome.reasoning_tokens,
            rung=outcome.rung,
            escalation=escalation,
            rung_receipts=receipts,
        )
        return DecisionResult(
            receipt,
            outcome.frame,
            build_work_request(request, reason="; ".join(outcome.triggers)),
        )

    @staticmethod
    def _work_result(
        request: DecisionRequest,
        route: RouteDecision,
        *,
        provider_calls: int,
        generated_tokens: int,
        rung_receipts: Sequence[Mapping[str, object]],
    ) -> DecisionResult:
        escalation = EscalationReceipt(
            decision_id=request.decision_id,
            from_route="HOST",
            to_route="WORK_FRAME",
            triggers=route.triggers,
            help_rungs=(),
            input_hash=_request_hash(request),
            provider_calls=provider_calls,
            generated_tokens=generated_tokens,
            reasoning_tokens=0,
            status="host_escalated",
        )
        receipt = DecisionReceipt(
            decision_id=request.decision_id,
            status="escalated",
            route="WORK_FRAME",
            arm="WORK",
            answer=None,
            confidence=None,
            evidence_ids=request.visible_evidence_ids,
            trigger_reasons=route.triggers,
            provider_calls=provider_calls,
            generated_tokens=generated_tokens,
            reasoning_tokens=0,
            escalation=escalation,
            rung_receipts=tuple(rung_receipts),
        )
        return DecisionResult(
            receipt, None, build_work_request(request, reason=route.reason)
        )


__all__ = [
    "ABSTAIN",
    "ALLOWED_LADDER_RUNGS",
    "DecisionFrame",
    "DecisionKind",
    "DecisionReceipt",
    "DecisionRequest",
    "DecisionResult",
    "DecisionTier",
    "DECIDE2_ANSWER_TOOL_NAME",
    "Decide2Error",
    "EvidenceItem",
    "EscalationReceipt",
    "MICRO_CONFIDENCE_FLOOR",
    "MICRO_DECISION_KINDS",
    "MICRO_MAX_TOKENS",
    "ParsedAnswer",
    "Provider",
    "RouteDecision",
    "SolvedCase",
    "SubQuestion",
    "apply_thinking_off",
    "build_frame",
    "build_micro_request",
    "build_work_request",
    "combine_subanswers",
    "enabled",
    "host_trigger",
    "normalize_tool_surface",
    "parse_answer",
    "preflight_skip_result",
    "procedure_text",
    "route_request",
    "run_system_one_mode",
    "raw_output_text",
    "safe_error",
]
