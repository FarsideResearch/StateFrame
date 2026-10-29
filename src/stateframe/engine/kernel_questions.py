"""Typed, host-first questions for the StateFrame build.

This module supplies the typed question seam used by the live StateFrame:

* :class:`QuestionSpec` describes a pure trigger, a typed answer shape, and
  the dependencies that invalidate the question.
* :class:`QuestionScheduler` evaluates a frozen snapshot, answers host
  questions when their host facts are available, and exposes at most one
  model question per step.
* model answers are commitments.  They are validated against host-known
  evidence, but they never become evidence merely by being asserted.

The host answerers use the existing typed organs.  In particular, usage
lookups go through the GET-SURE executor and its ``UsageIndex``; clause
coverage goes through ``kernel_findings``; regressions can come from the
obligation graph; action repetition can consult the pre-dispatch retry gate.
No frame or existing kernel file is mutated here.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias, TypedDict

from detreason.get_sure import (
    ACTION_KINDS,
    ASK_HUMAN,
    CATALOG,
    ActionRequest,
    Channel,
    GetSureContext,
    KNOWLEDGE_CHAIN,
    SEAM_GRAPH_WALK,
    USAGE_INDEX_LOOKUP,
    check_closure,
    exec_knowledge_chain,
    exec_seam_graph_walk,
    exec_usage_index_lookup,
)
from detreason.knowledge import build_usage_index
from detreason.uledger import ClosureRecord, ULedger, URow
from kernel_edit_guard import host_observed_findings
from kernel_findings import lexical_token_count
from kernel_obligations import complete_enough, coverage_receipt
from kernel_pivot import fingerprint as stable_fingerprint
from kernel_retry_gate import RetryGate


__all__ = [
    "AnswerKind",
    "HOST",
    "MODEL_TYPED",
    "JUDGEMENT",
    "QuestionStatus",
    "PENDING",
    "ANSWERED",
    "STALE",
    "SKIPPED",
    "QuestionSpec",
    "Question",
    "QuestionAnsweredEvent",
    "TypedSnapshot",
    "HostAnswer",
    "ScheduleResult",
    "QuestionScheduler",
    "QuestionEngine",
    "Scheduler",
    "QuestionAnswerRejected",
    "DoneCommitment",
    "DoneCommitmentSchema",
    "DONE_COMMITMENT_SCHEMA",
    "COMMITMENT_CHOICES",
    "DEFAULT_QUESTION_SPECS",
    "default_question_specs",
    "schedule_questions",
    "answer_question",
    "validate_model_answer",
    "question_answered_event",
    "make_question_answered_event",
    "question_answered_record",
    "journal_question_answered",
    "render_changed",
    "unresolved_question_ids",
    "render_unresolved_questions",
    "commitment_constraints",
    "action_identity",
    "host_clause_coverage",
    "host_where_used",
    "host_tried_exact_action",
    "host_regressions",
    "host_structural_findings",
    "run_get_sure_executor",
    "GET_SURE_EXECUTORS",
    "QUESTION_FIELD_KEYS",
    "question_mode",
    "enabled",
    # Verified imported seams, kept public for the later wiring pass.
    "ACTION_KINDS",
    "CATALOG",
    "Channel",
    "ClosureRecord",
    "ULedger",
    "URow",
    "check_closure",
    "complete_enough",
    "coverage_receipt",
]


# Project #223: these are the only state keys consumed by the question engine.
# ``StateFrameRuntime._question_snapshot`` is the producer of the same literal
# keys. Keep this mapping public so contract tests can pin both sides.
QUESTION_FIELD_KEYS = MappingProxyType({
    "basis_revision": "basis_revision",
    "task_brief": "task_brief",
    "source": "source",
    "edited_symbols": "edited_symbols",
    "edits": "edits",
    "repo_root": "repo_root",
    "proposed_action": "proposed_action",
    "action_history": "action_history",
    "test_history": "test_history",
    "regressions": "regressions",
    "clause_coverage": "clause_coverage",
    "done_proposed": "done_proposed",
    "evidence_refs": "evidence_refs",
    "where_used": "where_used",
    "symbol": "symbol",
    "retry_gate": "retry_gate",
    "usage_index": "usage_index",
    "seam_graph": "seam_graph",
    "knowledge_ledger": "knowledge_ledger",
    "environment_id": "environment_id",
    "run_id": "run_id",
    "decision_id": "decision_id",
})


def question_mode() -> str:
    """Return the strict question-engine switch; the compatibility default is off."""
    value = os.environ.get("STATEFRAME_QUESTIONS", "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError("STATEFRAME_QUESTIONS must be off or on")
    return value


def enabled() -> bool:
    return question_mode() == "on"


class AnswerKind(StrEnum):
    """Who can supply an answer and what role it has."""

    HOST = "HOST"
    MODEL_TYPED = "MODEL_TYPED"
    JUDGEMENT = "JUDGEMENT"


HOST = AnswerKind.HOST
MODEL_TYPED = AnswerKind.MODEL_TYPED
JUDGEMENT = AnswerKind.JUDGEMENT


class QuestionStatus(StrEnum):
    PENDING = "pending"
    ANSWERED = "answered"
    STALE = "stale"
    SKIPPED = "skipped"


PENDING = QuestionStatus.PENDING
ANSWERED = QuestionStatus.ANSWERED
STALE = QuestionStatus.STALE
SKIPPED = QuestionStatus.SKIPPED


COMMITMENT_CHOICES = ("add_test", "argue_covered", "out_of_scope")


class QuestionAnswerRejected(ValueError):
    """A typed answer did not satisfy its question's schema or evidence gate."""


class QuestionAnsweredEvent(TypedDict, total=False):
    schema: str
    kind: str
    event: str
    event_type: str
    fired: bool
    question_id: str
    qid: str
    spec_qid: str
    basis_revision: object
    status: str
    answer_kind: str
    answered_by: str
    answer: object
    evidence_refs: list[str]
    proof: dict[str, object]
    commitment: dict[str, object] | None


JSONValue: TypeAlias = object
Trigger: TypeAlias = Callable[["TypedSnapshot"], bool]
SubjectResolver: TypeAlias = Callable[["TypedSnapshot"], Iterable[str]]
HostAnswerer: TypeAlias = Callable[..., object]


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _coerce_answer_kind(value: AnswerKind | str) -> AnswerKind:
    if isinstance(value, AnswerKind):
        return value
    text = str(value).strip().upper().replace("-", "_")
    try:
        return AnswerKind[text]
    except KeyError as exc:
        raise ValueError(f"unknown answer kind: {value!r}") from exc


def _coerce_status(value: QuestionStatus | str) -> QuestionStatus:
    if isinstance(value, QuestionStatus):
        return value
    text = str(value).strip().lower()
    try:
        return QuestionStatus(text)
    except ValueError as exc:
        raise ValueError(f"unknown question status: {value!r}") from exc


def _text(value: object, *, limit: int = 400) -> str:
    return " ".join(str(value or "").split())[:limit]


def _jsonable(value: object) -> JSONValue:
    """Convert typed local values to a stable JSON-compatible projection."""

    if isinstance(value, (StrEnum, QuestionStatus, AnswerKind)):
        return value.value
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_jsonable(item) for item in value]
        return sorted(values, key=lambda item: _stable_json(item))
    if dataclasses.is_dataclass(value):
        return _jsonable(
            {
                field_info.name: getattr(value, field_info.name)
                for field_info in dataclasses.fields(value)
                if field_info.name != "_rendered_digest"
            }
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:  # pragma: no cover - defensive for foreign organs
            pass
    return _text(value, limit=800)


def _stable_json(value: object) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value: object) -> str:
    """Use the existing pivot fingerprint for stable action/question identity."""

    return stable_fingerprint(_jsonable(value))


def _freeze(value: object) -> object:
    """Freeze mappings/sequences without copying opaque typed organ objects."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class TypedSnapshot(Mapping[str, object]):
    """An immutable mapping view passed to pure triggers and host answerers."""

    values: Mapping[str, object] = field(default_factory=dict)
    revision: object = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze(dict(self.values)))
        if not self.revision:
            revision = self.values.get("basis_revision", "")
            object.__setattr__(self, "revision", revision)

    def __getitem__(self, key: str) -> object:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def data(self) -> Mapping[str, object]:
        return self.values


def _snapshot(state: TypedSnapshot | Mapping[str, object], revision: object = "") -> TypedSnapshot:
    if isinstance(state, TypedSnapshot):
        if revision and revision != state.revision:
            return TypedSnapshot(state.values, revision)
        return state
    if not isinstance(state, Mapping):
        raise TypeError("question state must be a TypedSnapshot or mapping")
    return TypedSnapshot(state, revision)


def _normal_refs(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (set, frozenset)):
        values = tuple(sorted(value, key=lambda item: str(item)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = tuple(value)
    else:
        values = (value,)
    return tuple(dict.fromkeys(_text(item, limit=240) for item in values if _text(item)))


def _subject_values(
    spec: "QuestionSpec", snapshot: TypedSnapshot
) -> tuple[tuple[str, ...], bool]:
    """Return dependency groups and whether the resolver is per-subject."""

    resolver = spec.subject_resolver
    raw: object
    dynamic = resolver is not None or callable(spec.subject_refs)
    if resolver is not None:
        raw = resolver(snapshot)
    elif callable(spec.subject_refs):
        raw = spec.subject_refs(snapshot)  # type: ignore[misc]
    else:
        raw = spec.subject_refs

    if dynamic:
        refs = tuple(sorted(_normal_refs(raw)))
        return tuple((ref,) for ref in refs) or (("*",),), True
    refs = _normal_refs(raw)
    return (refs or ("*",),), False


@dataclass(frozen=True, slots=True)
class QuestionSpec:
    """The static description of one typed question family."""

    qid: str
    trigger: Trigger
    answer_kind: AnswerKind | str
    answer_schema: object = None
    subject_refs: object = ()
    host_answerer: HostAnswerer | None = None
    priority: int = 100
    subject_resolver: SubjectResolver | None = None
    # ``answerer`` is a compatibility spelling for callers that use the audit's
    # shorter record notation.  It is normalized to ``host_answerer``.
    answerer: HostAnswerer | None = None

    def __post_init__(self) -> None:
        qid = _text(self.qid, limit=160)
        if not qid:
            raise ValueError("question qid is required")
        if not callable(self.trigger):
            raise TypeError("question trigger must be callable")
        answer_kind = _coerce_answer_kind(self.answer_kind)
        answerer = self.host_answerer or self.answerer
        if answerer is not None and not callable(answerer):
            raise TypeError("question host_answerer must be callable")
        object.__setattr__(self, "qid", qid)
        object.__setattr__(self, "answer_kind", answer_kind)
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(self, "host_answerer", answerer)
        object.__setattr__(self, "answerer", answerer)


@dataclass(slots=True)
class Question:
    """One fired question, retained across steps until answered or invalidated."""

    qid: str
    basis_revision: object
    status: QuestionStatus | str = QuestionStatus.PENDING
    answer: object = None
    evidence_refs: tuple[str, ...] = ()
    answered_by: str | None = None
    spec_qid: str = ""
    answer_kind: AnswerKind | str = AnswerKind.HOST
    answer_schema: object = None
    subject_refs: tuple[str, ...] = ()
    changed: bool = True
    emitted: bool = False
    constraints: Mapping[str, object] = field(default_factory=dict)
    _allowed_evidence_refs: tuple[str, ...] = field(default=(), repr=False)
    _rendered_digest: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.qid = _text(self.qid, limit=240)
        if not self.qid:
            raise ValueError("question qid is required")
        self.status = _coerce_status(self.status)
        self.answer_kind = _coerce_answer_kind(self.answer_kind)
        self.evidence_refs = _normal_refs(self.evidence_refs)
        self.subject_refs = _normal_refs(self.subject_refs)
        self._allowed_evidence_refs = _normal_refs(self._allowed_evidence_refs)
        if self.answered_by is not None:
            self.answered_by = _text(self.answered_by, limit=32).lower()
            if self.answered_by not in {"host", "model"}:
                raise ValueError("answered_by must be host or model")
        if self.answered_by == "model" and self.evidence_refs:
            raise QuestionAnswerRejected(
                "a model answer cannot add evidence references"
            )

    @property
    def subject(self) -> str:
        return self.subject_refs[0] if self.subject_refs else "*"

    @property
    def answer_digest(self) -> str:
        return _digest(
            {
                "qid": self.qid,
                "status": self.status.value,
                "answer": self.answer,
                "answered_by": self.answered_by,
                "evidence_refs": self.evidence_refs,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "qid": self.qid,
            "spec_qid": self.spec_qid,
            "basis_revision": _jsonable(self.basis_revision),
            "status": self.status.value,
            "answer_kind": self.answer_kind.value,
            "answer_schema": _jsonable(self.answer_schema),
            "answer": _jsonable(self.answer),
            "evidence_refs": list(self.evidence_refs),
            "answered_by": self.answered_by,
            "subject_refs": list(self.subject_refs),
            "changed": bool(self.changed),
            "emitted": bool(self.emitted),
            "constraints": _jsonable(self.constraints),
        }


@dataclass(frozen=True, slots=True)
class HostAnswer:
    """A host answer plus references to observations the host actually made."""

    answer: object
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_refs", _normal_refs(self.evidence_refs))


@dataclass(frozen=True, slots=True)
class DoneCommitment:
    """The only model answer in this pass: a next-step commitment."""

    choice: str
    evidence_ref: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        choice = _text(self.choice, limit=32).lower()
        if choice not in COMMITMENT_CHOICES:
            raise QuestionAnswerRejected(f"unknown done commitment: {self.choice!r}")
        evidence_ref = _text(self.evidence_ref, limit=240)
        reason = _text(self.reason, limit=400)
        if choice == "argue_covered" and not evidence_ref:
            raise QuestionAnswerRejected("argue_covered requires evidence_ref")
        if choice == "out_of_scope" and not reason:
            raise QuestionAnswerRejected("out_of_scope requires reason")
        if choice != "argue_covered" and evidence_ref:
            raise QuestionAnswerRejected(
                f"{choice} cannot carry an evidence_ref"
            )
        if choice != "out_of_scope" and reason:
            raise QuestionAnswerRejected(f"{choice} cannot carry a reason")
        object.__setattr__(self, "choice", choice)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "reason", reason)

    @classmethod
    def from_value(cls, value: object) -> "DoneCommitment":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise QuestionAnswerRejected("done commitment must be a mapping")
        return cls(
            choice=str(value.get("choice", "")),
            evidence_ref=str(value.get("evidence_ref", "")),
            reason=str(value.get("reason", "")),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "choice": self.choice,
            "evidence_ref": self.evidence_ref,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DoneCommitmentSchema:
    """Schema/validator for a clause's typed done-time commitment."""

    choices: tuple[str, ...] = COMMITMENT_CHOICES

    def validate(
        self, value: object, *, evidence_refs: Iterable[str] = ()
    ) -> DoneCommitment:
        commitment = DoneCommitment.from_value(value)
        if commitment.choice not in self.choices:
            raise QuestionAnswerRejected(
                f"choice {commitment.choice!r} is not in {self.choices!r}"
            )
        allowed = set(_normal_refs(evidence_refs))
        if commitment.choice == "argue_covered" and commitment.evidence_ref not in allowed:
            raise QuestionAnswerRejected(
                f"evidence_ref {commitment.evidence_ref!r} is not a host evidence ref"
            )
        return commitment


DONE_COMMITMENT_SCHEMA = DoneCommitmentSchema()


def _schema_validate(
    schema: object, value: object, *, evidence_refs: Iterable[str] = ()
) -> object:
    if schema is None or schema is object:
        return value
    validator = getattr(schema, "validate", None)
    if callable(validator):
        try:
            return validator(value, evidence_refs=evidence_refs)
        except TypeError:
            return validator(value)
    if isinstance(schema, type):
        if not isinstance(value, schema):
            raise QuestionAnswerRejected(
                f"answer must be {schema.__name__}, got {type(value).__name__}"
            )
        return value
    if isinstance(schema, Mapping):
        choices = schema.get("choices")
        if choices is not None and value not in choices:
            raise QuestionAnswerRejected(f"answer is not one of {tuple(choices)!r}")
        expected_type = schema.get("type")
        if isinstance(expected_type, type) and not isinstance(value, expected_type):
            raise QuestionAnswerRejected(
                f"answer must be {expected_type.__name__}, got {type(value).__name__}"
            )
        return value
    if isinstance(schema, (tuple, list, set, frozenset)):
        values = tuple(schema)
        if values and all(isinstance(item, type) for item in values):
            if not isinstance(value, values):
                raise QuestionAnswerRejected("answer has the wrong type")
        elif value not in values:
            raise QuestionAnswerRejected(f"answer is not one of {values!r}")
        return value
    if callable(schema):
        result = schema(value)
        if result is False:
            raise QuestionAnswerRejected("answer rejected by schema")
        return value if result is True or result is None else result
    return value


def _state_text(state: TypedSnapshot, key: str) -> str:
    value = state.get(key)
    return value if isinstance(value, str) and value.strip() else ""


def _as_records(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _evidence_refs_from_state(state: TypedSnapshot) -> tuple[str, ...]:
    """Collect only host-published evidence identities, never model text."""
    return _normal_refs(state.get("evidence_refs"))


def _coverage_rows(state: TypedSnapshot) -> tuple[dict[str, object], ...]:
    supplied = state.get("clause_coverage")

    rows: list[dict[str, object]] = []
    if isinstance(supplied, Mapping):
        supplied = tuple(
            {"id": key, **(value if isinstance(value, Mapping) else {"coverage": value})}
            for key, value in supplied.items()
        )
    for index, raw in enumerate(_as_records(supplied), 1):
        clause_id = _text(raw.get("id", f"C{index}"), limit=80)
        clause = _text(raw.get("clause", ""), limit=700)
        coverage = _text(raw.get("coverage", "unknown"), limit=40).lower()
        refs = _normal_refs(raw.get("evidence_refs"))
        mapped = coverage in {"yes", "covered", "green", "true", "1"} or bool(refs)
        rows.append(
            {
                "id": clause_id,
                "clause": clause,
                "coverage": "yes" if mapped else coverage or "unknown",
                "mapped": mapped,
                "evidence_refs": refs,
            }
        )
    return tuple(sorted(rows, key=lambda row: str(row["id"])))


def host_clause_coverage(
    state: TypedSnapshot | Mapping[str, object], subject: str | None = None
) -> HostAnswer | None:
    """Answer Q2/Q24 with the existing clause-coverage analyzer."""

    snapshot = _snapshot(state)
    if not (_state_text(snapshot, "task_brief") or
            snapshot.get("clause_coverage") is not None):
        return None
    rows = _coverage_rows(snapshot)
    answer = {
        "clauses": [
            {
                "id": row["id"],
                "clause": row["clause"],
                "coverage": row["coverage"],
                "mapped": row["mapped"],
                "evidence_refs": list(row["evidence_refs"]),
            }
            for row in rows
        ],
        "uncovered": [str(row["id"]) for row in rows if not row["mapped"]],
    }
    refs = tuple(ref for row in rows for ref in row["evidence_refs"])
    refs += _evidence_refs_from_state(snapshot)
    return HostAnswer(answer, tuple(dict.fromkeys(refs)))


def _edited_symbols(state: TypedSnapshot) -> tuple[str, ...]:
    return tuple(sorted(set(_normal_refs(state.get("edited_symbols")))))


def _usage_index_for(state: TypedSnapshot):
    index = state.get("usage_index")
    if index is not None:
        return index
    root = state.get("repo_root")
    if root is None:
        return None
    try:
        return build_usage_index(root)
    except (OSError, TypeError, ValueError):
        return None


def _get_sure_request(
    action_kind: str, target: Mapping[str, object], state: TypedSnapshot
) -> tuple[ActionRequest, GetSureContext] | None:
    index = _usage_index_for(state)
    if action_kind in {USAGE_INDEX_LOOKUP, KNOWLEDGE_CHAIN} and index is None:
        return None
    context = GetSureContext(
        base_facts=dict(state.values),
        usage_index=index,
        seam_graph=dict(state.get("seam_graph", {}) or {}),
        knowledge_ledger=state.get("knowledge_ledger"),
        environment_id=str(state.get("environment_id", "question-engine")),
    )
    target_dict = dict(target)
    identity = _digest({"action": action_kind, "target": target_dict, "revision": state.revision})
    request = ActionRequest(
        action_id=f"question:{identity}",
        run_id=str(state.get("run_id", "question-engine")),
        source_decision_id=str(state.get("decision_id", "question-engine")),
        target_doubt_lineage_ids=(),
        action_kind=action_kind,
        target=target_dict,
        context_digest=context.digest(),
        catalog_digest="",
        cost_estimate_milli=0,
        idempotency_key=identity,
    )
    return request, context


GET_SURE_EXECUTORS: Mapping[str, Callable[..., object]] = {
    USAGE_INDEX_LOOKUP: exec_usage_index_lookup,
    KNOWLEDGE_CHAIN: exec_knowledge_chain,
    SEAM_GRAPH_WALK: exec_seam_graph_walk,
}


def run_get_sure_executor(
    action_kind: str,
    target: Mapping[str, object],
    state: TypedSnapshot | Mapping[str, object],
) -> object:
    """Run one existing deterministic GET-SURE executor on a frozen snapshot."""

    if action_kind not in GET_SURE_EXECUTORS:
        raise ValueError(f"executor is not exposed by the question engine: {action_kind!r}")
    snapshot = _snapshot(state)
    request_context = _get_sure_request(action_kind, target, snapshot)
    if request_context is None:
        return None
    request, context = request_context
    return GET_SURE_EXECUTORS[action_kind](request, context)


def host_where_used(
    state: TypedSnapshot | Mapping[str, object], subject: str | None = None,
    *, planned_ledger=None,
) -> HostAnswer | None:
    """Answer Q6/Q17 with ``build_usage_index`` and GET-SURE's usage executor."""

    snapshot = _snapshot(state)
    symbol = _text(subject or snapshot.get("symbol", ""), limit=160)
    if not symbol:
        symbols = _edited_symbols(snapshot)
        symbol = symbols[0] if symbols else ""
    if not symbol:
        return None

    if planned_ledger is not None:
        readiness_inputs = []
        for key in ("path", "target_path", "reference", "target", "source_path"):
            value = snapshot.get(key)
            if isinstance(value, str) and value.strip():
                readiness_inputs.append(value)
        if readiness_inputs:
            view = planned_ledger.readiness(
                "where_used", readiness_inputs, kind="where_used")
            if view.status != "READY":
                # A deferred where-used check has no usage verdict.  The
                # normal host answerer resumes once its planned inputs exist.
                return HostAnswer({
                    "symbol": symbol, "status": "DEFERRED",
                    "readiness": view.to_dict(),
                })

    supplied = snapshot.get("where_used")
    if isinstance(supplied, Mapping) and symbol in supplied:
        rows = supplied[symbol]
        normalized = sorted(
            [_jsonable(row) for row in rows] if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else [],
            key=lambda row: _stable_json(row),
        )
        return HostAnswer(
            {"symbol": symbol, "uses": normalized, "total": len(normalized), "source": "host_index"},
            tuple(_text(item.get("ref", "")) for item in normalized if isinstance(item, Mapping)),
        )

    index = _usage_index_for(snapshot)
    if index is None:
        return None
    result = run_get_sure_executor(USAGE_INDEX_LOOKUP, {"symbol": symbol, "limit": 256}, snapshot)
    observations = list(getattr(result, "observations", ()))
    observations = sorted(
        (dict(item) for item in observations if isinstance(item, Mapping)),
        key=lambda item: (str(item.get("path", "")), int(item.get("line", 0)), str(item.get("statement", ""))),
    )
    total = int(getattr(index, "count", lambda _symbol: len(observations))(symbol))
    refs = tuple(
        f"{row.get('path')}:{int(row.get('line', 0))}"
        for row in observations
        if row.get("path") and row.get("line")
    )
    return HostAnswer(
        {"symbol": symbol, "uses": observations, "total": total, "source": "get_sure.usage_index_lookup"},
        refs,
    )


def action_identity(action: Mapping[str, object] | object, basis_revision: object = "") -> str:
    """Stable identity for the exact tool/action at one basis revision."""

    if isinstance(action, Mapping):
        payload = {
            str(key): _jsonable(value)
            for key, value in action.items()
            if str(key)
            not in {
                "turn",
                "timestamp",
                "result",
                "status",
                "outcome",
                "basis_revision",
                "revision",
                "evidence_ref",
                "receipt_ref",
                "event_id",
                "step",
                "reason",
                "why",
            }
        }
    else:
        payload = {"action": _jsonable(action)}
    return _digest({"basis_revision": _jsonable(basis_revision), "action": payload})


def _action_records(state: TypedSnapshot) -> tuple[Mapping[str, object], ...]:
    return _as_records(state.get("action_history"))


def _record_action(record: Mapping[str, object]) -> Mapping[str, object]:
    nested = record.get("action")
    return nested if isinstance(nested, Mapping) else {}


def _record_revision(record: Mapping[str, object], default: object) -> object:
    return record.get("basis_revision", default)


def host_tried_exact_action(
    state: TypedSnapshot | Mapping[str, object], subject: str | None = None
) -> HostAnswer | None:
    """Answer Q15 from journal identity, with a read-only retry-gate forecast."""

    snapshot = _snapshot(state)
    raw_action = snapshot.get("proposed_action")
    if not isinstance(raw_action, Mapping):
        return None
    identity = action_identity(raw_action, snapshot.revision)
    prior: list[dict[str, object]] = []
    refs: list[str] = []
    for index, record in enumerate(_action_records(snapshot)):
        candidate = record.get("identity_hash")
        candidate_identity = str(candidate) if candidate else action_identity(
            _record_action(record), _record_revision(record, snapshot.revision)
        )
        if candidate_identity != identity:
            continue
        if _record_revision(record, snapshot.revision) != snapshot.revision:
            continue
        prior.append(
            {
                "index": index,
                "turn": record.get("turn", index),
                "result": record.get("result", ""),
                "evidence_ref": record.get("evidence_ref", ""),
            }
        )
        refs.extend(_normal_refs(record.get("evidence_ref")))

    retry_forecast: object = None
    gate = snapshot.get("retry_gate")
    if gate is not None and callable(getattr(gate, "peek_predispatch", None)):
        try:
            retry_input = dict(raw_action)
            retry_input.setdefault("kind", "edit")
            retry_forecast = gate.peek_predispatch(retry_input)
            to_dict = getattr(retry_forecast, "to_dict", None)
            retry_forecast = to_dict() if callable(to_dict) else _jsonable(retry_forecast)
        except (TypeError, ValueError):
            retry_forecast = {"status": "unavailable"}

    answer = {
        "action_identity": identity,
        "tried_before": bool(prior),
        "prior_actions": sorted(prior, key=lambda row: (str(row["turn"]), int(row["index"]))),
        "retry_gate": retry_forecast,
    }
    return HostAnswer(answer, tuple(dict.fromkeys(refs)))


def _result_kind(record: Mapping[str, object]) -> str:
    value = record.get("result", "")
    value = _text(value, limit=64).upper()
    if value in {"PASS", "PASSED", "GREEN", "OK", "SUCCESS", "REUSED_VERIFICATION"}:
        return "green"
    if value in {"FAIL", "FAILED", "RED", "ERROR", "TESTS_FAILED"}:
        return "red"
    return ""


def _test_records(state: TypedSnapshot) -> tuple[Mapping[str, object], ...]:
    return _as_records(state.get("test_history"))


def _edits_between(
    state: TypedSnapshot, green_turn: int, red_turn: int, green_revision: object, red_revision: object
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, edit in enumerate(_as_records(state.get("edits"))):
        raw_turn = edit.get("turn", index)
        try:
            turn = int(raw_turn)
        except (TypeError, ValueError):
            turn = index
        if green_turn < turn <= red_turn:
            rows.append(dict(_jsonable(edit)))
            continue
        revision = edit.get("basis_revision")
        if revision is not None and revision not in {green_revision, red_revision}:
            rows.append(dict(_jsonable(edit)))
    return sorted(rows, key=lambda row: (str(row.get("turn", "")), _stable_json(row)))


def _graph_regressions(state: TypedSnapshot) -> list[Mapping[str, object]]:
    supplied = state.get("regressions")
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        return [item for item in supplied if isinstance(item, Mapping)]
    return []


def host_regressions(
    state: TypedSnapshot | Mapping[str, object], subject: str | None = None
) -> HostAnswer | None:
    """Answer Q22 from obligation transitions or green→red test history."""

    snapshot = _snapshot(state)
    rows: list[dict[str, object]] = []
    for raw in _graph_regressions(snapshot):
        rows.append(
            {
                "test": str(raw.get("test", "")),
                "green_revision": raw.get("green_revision", ""),
                "red_revision": raw.get("red_revision", ""),
                "green_turn": raw.get("green_turn"),
                "red_turn": raw.get("red_turn"),
                "edits_between": list(raw.get("edits_between", ())),
                "source": "kernel_obligations",
            }
        )

    grouped: dict[str, list[tuple[int, Mapping[str, object]]]] = {}
    for index, record in enumerate(_test_records(snapshot)):
        selector = _text(record.get("test", ""), limit=240)
        kind = _result_kind(record)
        if not selector or not kind:
            continue
        try:
            turn = int(record.get("turn", index))
        except (TypeError, ValueError):
            turn = index
        grouped.setdefault(selector, []).append((turn, record))

    for selector, records in sorted(grouped.items()):
        records.sort(key=lambda item: (item[0], _stable_json(item[1])))
        last_green: tuple[int, Mapping[str, object]] | None = None
        for red_turn, red in records:
            kind = _result_kind(red)
            if kind == "green":
                last_green = (red_turn, red)
                continue
            if last_green is None:
                continue
            green_turn, green = last_green
            green_revision = _record_revision(green, "")
            red_revision = _record_revision(red, "")
            if green_revision == red_revision:
                continue
            rows.append(
                {
                    "test": selector,
                    "green_revision": green_revision,
                    "red_revision": red_revision,
                    "green_turn": green_turn,
                    "red_turn": red_turn,
                    "edits_between": _edits_between(
                        snapshot, green_turn, red_turn, green_revision, red_revision
                    ),
                    "source": "test_history",
                }
            )

    rows = sorted(rows, key=lambda row: (str(row["test"]), str(row.get("red_revision", "")), _stable_json(row)))
    if subject:
        rows = [row for row in rows if str(row["test"]) == subject]
    if not rows:
        return None
    refs = tuple(
        ref
        for row in rows
        for ref in _normal_refs(row.get("evidence_ref"))
    )
    return HostAnswer({"regressions": rows}, tuple(dict.fromkeys(refs)))


def host_structural_findings(
    state: TypedSnapshot | Mapping[str, object], subject: str | None = None
) -> HostAnswer | None:
    """Use the existing edit-guard analyzer for duplicate/stub findings."""

    snapshot = _snapshot(state)
    source = _state_text(snapshot, "source")
    if not source:
        return None
    findings = host_observed_findings(source, force=True)
    return HostAnswer({"findings": list(findings), "clean": not findings}, ())


def _coverage_trigger(state: TypedSnapshot) -> bool:
    return bool(
        _state_text(state, "task_brief")
        or state.get("clause_coverage") is not None
    )


def _where_used_trigger(state: TypedSnapshot) -> bool:
    return bool(_edited_symbols(state))


def _action_trigger(state: TypedSnapshot) -> bool:
    return isinstance(state.get("proposed_action"), Mapping)


def _regression_trigger(state: TypedSnapshot) -> bool:
    return host_regressions(state) is not None


def _structural_trigger(state: TypedSnapshot) -> bool:
    return bool(_state_text(state, "source"))


def _done_proposed(state: TypedSnapshot) -> bool:
    return bool(state.get("done_proposed"))


def _uncovered_clause_ids(state: TypedSnapshot) -> tuple[str, ...]:
    return tuple(str(row["id"]) for row in _coverage_rows(state) if not row["mapped"])


def _done_trigger(state: TypedSnapshot) -> bool:
    return _done_proposed(state) and bool(_uncovered_clause_ids(state))


def _done_subjects(state: TypedSnapshot) -> tuple[str, ...]:
    return _uncovered_clause_ids(state)


def _done_answer(state: TypedSnapshot, subject: str) -> HostAnswer | None:
    rows = {str(row["id"]): row for row in _coverage_rows(state)}
    row = rows.get(subject)
    if row is None:
        return None
    return HostAnswer(
        {
            "clause_id": subject,
            "clause": row["clause"],
            "choices": list(COMMITMENT_CHOICES),
            "answer_role": "commitment_not_evidence",
        },
        tuple(_evidence_refs_from_state(state)),
    )


def default_question_specs() -> tuple[QuestionSpec, ...]:
    """Return the deterministic Q2/Q6/Q15/Q18/Q22/Q25 build-order specs."""

    return (
        QuestionSpec(
            qid="Q02.clause_coverage",
            trigger=_coverage_trigger,
            answer_kind=HOST,
            answer_schema=Mapping,
            subject_refs=("task_brief", "clause_coverage"),
            host_answerer=host_clause_coverage,
            priority=20,
        ),
        QuestionSpec(
            qid="Q06.where_used",
            trigger=_where_used_trigger,
            answer_kind=HOST,
            answer_schema=Mapping,
            subject_resolver=_edited_symbols,
            host_answerer=host_where_used,
            priority=30,
        ),
        QuestionSpec(
            qid="Q15.tried_exact_action",
            trigger=_action_trigger,
            answer_kind=HOST,
            answer_schema=Mapping,
            subject_refs=("proposed_action",),
            host_answerer=host_tried_exact_action,
            priority=10,
        ),
        QuestionSpec(
            qid="Q18.structural_findings",
            trigger=_structural_trigger,
            answer_kind=HOST,
            answer_schema=Mapping,
            subject_refs=("source",),
            host_answerer=host_structural_findings,
            priority=40,
        ),
        QuestionSpec(
            qid="Q22.regression",
            trigger=_regression_trigger,
            answer_kind=HOST,
            answer_schema=Mapping,
            subject_refs=("test_history",),
            host_answerer=host_regressions,
            priority=15,
        ),
        QuestionSpec(
            qid="Q25.done_clause",
            trigger=_done_trigger,
            answer_kind=MODEL_TYPED,
            answer_schema=DONE_COMMITMENT_SCHEMA,
            subject_resolver=_done_subjects,
            host_answerer=_done_answer,
            priority=50,
        ),
    )


DEFAULT_QUESTION_SPECS = default_question_specs()


def _qid_for(spec: QuestionSpec, group: tuple[str, ...], dynamic: bool, count: int) -> str:
    if not dynamic:
        return spec.qid
    return f"{spec.qid}::{group[0]}"


def _subjects_overlap(left: Iterable[str], right: Iterable[str]) -> bool:
    left_values = _normal_refs(tuple(left))
    right_values = _normal_refs(tuple(right))
    if "*" in left_values or "*" in right_values:
        return True
    for a in left_values:
        for b in right_values:
            if a == b or a.startswith(b + "::") or b.startswith(a + "::"):
                return True
            if a.endswith("/") and b.startswith(a):
                return True
            if b.endswith("/") and a.startswith(b):
                return True
    return False


def _edit_subjects(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return _normal_refs(value.get("subject"))
    return _normal_refs(value)


@dataclass(frozen=True, slots=True)
class ScheduleResult(Sequence[Question]):
    """The model-visible delta from one scheduler step."""

    questions: tuple[Question, ...] = ()
    host_answered: tuple[Question, ...] = ()
    model_questions: tuple[Question, ...] = ()
    stale: tuple[Question, ...] = ()
    skipped: tuple[Question, ...] = ()
    events: tuple[QuestionAnsweredEvent, ...] = ()
    commitments: tuple[Mapping[str, object], ...] = ()
    all_questions: tuple[Question, ...] = ()

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, index):
        return self.questions[index]

    def __iter__(self):
        return iter(self.questions)


def _call_host_answerer(
    answerer: HostAnswerer, snapshot: TypedSnapshot, question: Question
) -> HostAnswer | None:
    """Call either the two-argument or one-argument answerer convention."""

    try:
        signature = inspect.signature(answerer)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        accepts_subject = any(
            parameter.kind is parameter.VAR_POSITIONAL for parameter in signature.parameters.values()
        ) or len(positional) >= 2
    except (TypeError, ValueError):  # pragma: no cover - opaque callable
        accepts_subject = True
    value = answerer(snapshot, question.subject) if accepts_subject else answerer(snapshot)
    if value is None:
        return None
    if isinstance(value, HostAnswer):
        return value
    if isinstance(value, Mapping) and "answer" in value:
        return HostAnswer(value["answer"], _normal_refs(value.get("evidence_refs")))
    return HostAnswer(value)


def _commitment_from_answer(value: object) -> DoneCommitment | None:
    if isinstance(value, DoneCommitment):
        return value
    if isinstance(value, Mapping) and "choice" in value:
        return DoneCommitment.from_value(value)
    return None


def commitment_constraints(value: object) -> dict[str, object]:
    """Return host-enforceable menu constraints for a model commitment."""

    commitment = _commitment_from_answer(value)
    if commitment is None:
        return {}
    if commitment.choice == "add_test":
        return {
            "choice": "add_test",
            "required_menu_actions": ("edit", "run_tests"),
            "answer_is_evidence": False,
        }
    if commitment.choice == "argue_covered":
        return {
            "choice": "argue_covered",
            "required_evidence_ref": commitment.evidence_ref,
            "required_menu_actions": ("run_tests",),
            "answer_is_evidence": False,
        }
    return {
        "choice": "out_of_scope",
        "out_of_scope_reason": commitment.reason,
        "required_menu_actions": (),
        "answer_is_evidence": False,
    }


def _allowed_evidence(question: Question, extra: Iterable[str] = ()) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _normal_refs(question._allowed_evidence_refs)
            + _normal_refs(question.evidence_refs)
            + _normal_refs(extra)
        )
    )


def validate_model_answer(
    question: Question,
    answer: object,
    *,
    evidence_refs: Iterable[str] = (),
) -> object:
    """Validate a model slot; supplied model refs are never silently admitted."""

    if question.answer_kind not in {AnswerKind.MODEL_TYPED, AnswerKind.JUDGEMENT}:
        raise QuestionAnswerRejected("a HOST question cannot be answered by the model")
    if _normal_refs(evidence_refs):
        raise QuestionAnswerRejected("model answers cannot create evidence_refs")
    return _schema_validate(
        question.answer_schema,
        answer,
        evidence_refs=_allowed_evidence(question),
    )


def _set_answer(
    question: Question,
    answer: object,
    *,
    answered_by: str,
    evidence_refs: Iterable[str] = (),
) -> Question:
    actor = _text(answered_by, limit=16).lower()
    if actor not in {"host", "model"}:
        raise ValueError("answered_by must be host or model")
    if actor == "model":
        validated = validate_model_answer(question, answer, evidence_refs=evidence_refs)
        refs: tuple[str, ...] = question.evidence_refs
    else:
        validated = _schema_validate(
            question.answer_schema,
            answer,
            evidence_refs=_allowed_evidence(question, evidence_refs),
        )
        refs = _normal_refs(evidence_refs)
    old_digest = question.answer_digest if question.status is QuestionStatus.ANSWERED else None
    question.answer = validated
    question.status = QuestionStatus.ANSWERED
    question.answered_by = actor
    question.evidence_refs = refs
    commitment = _commitment_from_answer(validated)
    question.constraints = commitment_constraints(commitment) if commitment else {}
    question.changed = old_digest != question.answer_digest
    if old_digest is None:
        question.changed = True
    return question


def answer_question(
    question: Question,
    answer: object,
    *,
    answered_by: str = "model",
    evidence_refs: Iterable[str] = (),
) -> Question:
    """Answer a record in place after validation and return it."""

    return _set_answer(
        question,
        answer,
        answered_by=answered_by,
        evidence_refs=evidence_refs,
    )


class QuestionScheduler:
    """Deterministic once-per-episode question scheduler."""

    def __init__(
        self,
        specs: Iterable[QuestionSpec] = DEFAULT_QUESTION_SPECS,
        *,
        model_budget: int = 1,
    ) -> None:
        self.specs = tuple(sorted(specs, key=lambda spec: (spec.priority, spec.qid)))
        self.model_budget = self._budget(model_budget)
        self._active: dict[str, Question] = {}
        self._history: list[Question] = []
        self._last_snapshot: TypedSnapshot | None = None
        self._last_constraints: tuple[Mapping[str, object], ...] = ()

    @staticmethod
    def _budget(value: int) -> int:
        if isinstance(value, bool) or value not in (0, 1):
            raise ValueError("model_budget must be 0 or 1")
        return int(value)

    @property
    def questions(self) -> tuple[Question, ...]:
        return tuple(sorted(self._active.values(), key=lambda q: q.qid))

    @property
    def all_questions(self) -> tuple[Question, ...]:
        return tuple(self._history) + self.questions

    @property
    def history(self) -> tuple[Question, ...]:
        return tuple(self._history)

    @property
    def last_constraints(self) -> tuple[Mapping[str, object], ...]:
        return self._last_constraints

    def _archive(self, question: Question) -> None:
        if question not in self._history:
            self._history.append(question)

    def invalidate(
        self, edited_subjects: Iterable[object], *, basis_revision: object = ""
    ) -> tuple[Question, ...]:
        edits: list[str] = []
        for value in edited_subjects:
            edits.extend(_edit_subjects(value))
        if not edits:
            return ()
        stale: list[Question] = []
        for qid, question in sorted(tuple(self._active.items())):
            if question.status not in {QuestionStatus.PENDING, QuestionStatus.ANSWERED}:
                continue
            if not _subjects_overlap(question.subject_refs, edits):
                continue
            question.status = QuestionStatus.STALE
            question.changed = False
            stale.append(question)
            self._archive(question)
            self._active.pop(qid, None)
        return tuple(stale)

    def _new_question(
        self,
        spec: QuestionSpec,
        group: tuple[str, ...],
        dynamic: bool,
        count: int,
        snapshot: TypedSnapshot,
    ) -> Question:
        qid = _qid_for(spec, group, dynamic, count)
        return Question(
            qid=qid,
            spec_qid=spec.qid,
            basis_revision=snapshot.revision,
            status=PENDING,
            answer_kind=spec.answer_kind,
            answer_schema=spec.answer_schema,
            subject_refs=group,
            _allowed_evidence_refs=_evidence_refs_from_state(snapshot),
        )

    def _active_for_spec(self, spec: QuestionSpec) -> tuple[Question, ...]:
        return tuple(
            sorted(
                (q for q in self._active.values() if q.spec_qid == spec.qid),
                key=lambda q: q.qid,
            )
        )

    def answer(
        self,
        qid: str,
        answer: object,
        *,
        answered_by: str = "model",
        evidence_refs: Iterable[str] = (),
    ) -> Question:
        question = self._active.get(qid)
        if question is None:
            raise KeyError(f"no active question {qid!r}")
        if question.status is not QuestionStatus.PENDING:
            raise QuestionAnswerRejected(
                f"question {qid!r} is {question.status.value}, not pending"
            )
        return _set_answer(
            question,
            answer,
            answered_by=answered_by,
            evidence_refs=evidence_refs,
        )

    answer_question = answer

    def step(
        self,
        state: TypedSnapshot | Mapping[str, object],
        *,
        basis_revision: object = "",
        edited_subjects: Iterable[object] = (),
        model_budget: int | None = None,
        model_answers: Mapping[str, object] | None = None,
    ) -> ScheduleResult:
        """Evaluate one typed snapshot and return only this step's delta."""

        budget = self.model_budget if model_budget is None else self._budget(model_budget)
        snapshot = _snapshot(state, basis_revision)
        previous = self._last_snapshot
        revision_changed = previous is not None and previous.revision != snapshot.revision
        # A producer often carries the same cumulative edited-symbol list on
        # every frame.  Treating that list as a fresh edit invalidated Q06 on
        # every step.  Only a new workspace revision is an edit boundary.
        effective_edits = edited_subjects if previous is None or revision_changed else ()
        stale = list(self.invalidate(effective_edits, basis_revision=snapshot.revision))
        if revision_changed:
            # Clause coverage is a revision-bound host observation even when
            # the caller has no path-level edit subject to report.  Recompute
            # Q02 rather than retaining the answer from the old source.
            for qid, question in sorted(tuple(self._active.items())):
                if (question.status not in {QuestionStatus.PENDING, QuestionStatus.ANSWERED}
                        or question.spec_qid != "Q02.clause_coverage"):
                    continue
                question.status = QuestionStatus.STALE
                question.changed = False
                stale.append(question)
                self._archive(question)
                self._active.pop(qid, None)
        scheduled: list[Question] = []
        host_answered: list[Question] = []
        candidates: list[tuple[QuestionSpec, Question]] = []
        skipped: list[Question] = []

        for spec in self.specs:
            try:
                triggered = bool(spec.trigger(snapshot))
            except (KeyError, TypeError, ValueError):
                triggered = False
            active = list(self._active_for_spec(spec))
            if not triggered:
                for question in active:
                    if question.status is QuestionStatus.PENDING:
                        question.status = QuestionStatus.SKIPPED
                        question.changed = False
                        skipped.append(question)
                        self._archive(question)
                        self._active.pop(question.qid, None)
                continue

            groups, dynamic = _subject_values(spec, snapshot)
            count = len(groups)
            desired_ids: set[str] = set()
            for group in groups:
                qid = _qid_for(spec, group, dynamic, count)
                desired_ids.add(qid)
                question = self._active.get(qid)
                if question is not None and question.subject_refs != group:
                    question.status = QuestionStatus.STALE
                    question.changed = False
                    stale.append(question)
                    self._archive(question)
                    self._active.pop(qid, None)
                    question = None
                if question is None:
                    question = self._new_question(spec, group, dynamic, count, snapshot)
                    self._active[qid] = question
                    if spec.answer_kind is not AnswerKind.MODEL_TYPED:
                        scheduled.append(question)

                if question.status is QuestionStatus.PENDING:
                    if spec.answer_kind is AnswerKind.HOST and spec.host_answerer is not None:
                        result = _call_host_answerer(spec.host_answerer, snapshot, question)
                        if result is not None:
                            _set_answer(
                                question,
                                result.answer,
                                answered_by="host",
                                evidence_refs=result.evidence_refs,
                            )
                            if question not in host_answered:
                                host_answered.append(question)
                            if question not in scheduled:
                                scheduled.append(question)
                    elif spec.answer_kind is AnswerKind.MODEL_TYPED and not question.emitted:
                        candidates.append((spec, question))
                    elif spec.answer_kind is AnswerKind.JUDGEMENT:
                        question.status = QuestionStatus.SKIPPED
                        question.changed = False
                        skipped.append(question)
                        self._archive(question)
                        self._active.pop(question.qid, None)

            # A dynamic dependency set can remove one clause while retaining another.
            for question in active:
                if question.qid not in desired_ids and question.status is QuestionStatus.PENDING:
                    question.status = QuestionStatus.SKIPPED
                    question.changed = False
                    skipped.append(question)
                    self._archive(question)
                    self._active.pop(question.qid, None)

        candidates.sort(key=lambda pair: (pair[0].priority, pair[0].qid, pair[1].qid))
        model_questions: list[Question] = []
        commitments: list[Mapping[str, object]] = []
        events: list[QuestionAnsweredEvent] = []
        if budget:
            for spec, question in candidates[:1]:
                question.emitted = True
                model_questions.append(question)
                if question not in scheduled:
                    scheduled.append(question)
                if model_answers:
                    raw_answer = model_answers.get(question.qid)
                    if raw_answer is None and question.spec_qid in model_answers:
                        raw_answer = model_answers[question.spec_qid]
                    if raw_answer is not None:
                        _set_answer(question, raw_answer, answered_by="model")

        for question in host_answered:
            events.append(question_answered_event(question))
        for question in model_questions:
            if question.status is QuestionStatus.ANSWERED:
                commitments.append(dict(question.constraints))
                events.append(question_answered_event(question))

        self._last_snapshot = snapshot
        self._last_constraints = tuple(commitments)
        scheduled_by_id: dict[str, Question] = {}
        for question in scheduled:
            scheduled_by_id[question.qid] = question
        scheduled = sorted(
            scheduled_by_id.values(),
            key=lambda question: (question.qid, question.subject_refs),
        )
        return ScheduleResult(
            questions=tuple(scheduled),
            host_answered=tuple(host_answered),
            model_questions=tuple(model_questions),
            stale=tuple(stale),
            skipped=tuple(skipped),
            events=tuple(events),
            commitments=tuple(commitments),
            all_questions=self.questions,
        )


def schedule_questions(
    state: TypedSnapshot | Mapping[str, object],
    *,
    specs: Iterable[QuestionSpec] = DEFAULT_QUESTION_SPECS,
    scheduler: QuestionScheduler | None = None,
    model_budget: int | None = None,
    basis_revision: object = "",
    edited_subjects: Iterable[object] = (),
    model_answers: Mapping[str, object] | None = None,
) -> ScheduleResult:
    """Functional convenience wrapper for one or more independent callers."""

    engine = scheduler or QuestionScheduler(
        specs, model_budget=1 if model_budget is None else model_budget
    )
    return engine.step(
        state,
        basis_revision=basis_revision,
        edited_subjects=edited_subjects,
        model_budget=model_budget,
        model_answers=model_answers,
    )


def question_answered_event(
    question: Question, *, step: int | None = None
) -> QuestionAnsweredEvent:
    """Return the journal record proving a triggered question was answered."""

    if question.status is not QuestionStatus.ANSWERED or question.answered_by not in {
        "host",
        "model",
    }:
        raise QuestionAnswerRejected("only an answered question can emit a journal event")
    event: QuestionAnsweredEvent = {
        "schema": "kernel-questions/1",
        "kind": "question_answered",
        "event": "question_answered",
        "event_type": "question_answered",
        "fired": True,
        "question_id": question.qid,
        "qid": question.qid,
        "spec_qid": question.spec_qid,
        "basis_revision": _jsonable(question.basis_revision),
        "status": question.status.value,
        "answer_kind": question.answer_kind.value,
        "answered_by": str(question.answered_by),
        "answer": _jsonable(question.answer),
        "evidence_refs": list(question.evidence_refs),
        "proof": {
            "triggered": True,
            "answer_recorded": True,
            "answer_is_evidence": question.answered_by == "host"
            and bool(question.evidence_refs),
            "model_answer_is_evidence": False,
            "step": step,
        },
        "commitment": _jsonable(question.constraints)
        if question.answered_by == "model"
        else None,
    }
    return event


make_question_answered_event = question_answered_event
question_answered_record = question_answered_event
journal_question_answered = question_answered_event


# Friendly names for the stateful scheduler used by the wiring pass.
QuestionEngine = QuestionScheduler
Scheduler = QuestionScheduler
QuestionScheduler.schedule = QuestionScheduler.step
QuestionScheduler.evaluate = QuestionScheduler.step


def _render_items(questions: Iterable[Question]) -> list[Question]:
    by_id: dict[str, Question] = {}
    for question in questions:
        if isinstance(question, Question):
            by_id[question.qid] = question
    return [by_id[qid] for qid in sorted(by_id)]


def _render_line(question: Question) -> str:
    payload = {
        "qid": question.qid,
        "answered_by": question.answered_by,
        "answer": _jsonable(question.answer),
    }
    if question.evidence_refs:
        payload["evidence_refs"] = list(question.evidence_refs)
    return _stable_json(payload)


def render_changed(
    questions: Iterable[Question] | ScheduleResult, *, budget_tokens: int = 400
) -> str:
    """Render only newly changed answers, in stable order and within a token cap."""

    if isinstance(questions, ScheduleResult):
        values = questions.questions
    else:
        values = questions
    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int) or budget_tokens < 0:
        raise ValueError("budget_tokens must be a non-negative int")
    if budget_tokens == 0:
        return ""

    selected: list[str] = []
    used = 0
    for question in _render_items(values):
        if question.status is not QuestionStatus.ANSWERED or question.answered_by is None:
            continue
        if not question.changed and question._rendered_digest == question.answer_digest:
            continue
        line = _render_line(question)
        cost = lexical_token_count(line)
        if cost > budget_tokens - used:
            continue
        selected.append(line)
        used += cost
        question.changed = False
        question._rendered_digest = question.answer_digest
    return "\n".join(selected)


def unresolved_question_ids(
    questions: Iterable[Question] | QuestionScheduler | ScheduleResult,
) -> tuple[str, ...]:
    """Return active model questions which still have no typed commitment.

    These ids are a host fact about the done attempt.  They are deliberately
    separate from ``render_changed``: the question block remains the normal
    informational delivery, while this projection lets the completion reducer
    journal an unanswered commitment without turning it into a veto.
    """

    if isinstance(questions, QuestionScheduler):
        values = questions.questions
    elif isinstance(questions, ScheduleResult):
        values = questions.questions
    else:
        values = questions
    return tuple(sorted({
        question.qid
        for question in values
        if isinstance(question, Question)
        and question.status is QuestionStatus.PENDING
        and question.answer_kind is AnswerKind.MODEL_TYPED
    }))


def render_unresolved_questions(
    questions: Iterable[str] | QuestionScheduler | ScheduleResult,
) -> str:
    """Render the stable, machine-readable non-blocking done-time fact."""

    if isinstance(questions, (QuestionScheduler, ScheduleResult)):
        values = unresolved_question_ids(questions)
    elif isinstance(questions, (str, bytes)):
        values = (str(questions),)
    else:
        values = tuple(sorted({str(value) for value in questions if str(value).strip()}))
    return _stable_json({"unresolved_questions": list(values)})
