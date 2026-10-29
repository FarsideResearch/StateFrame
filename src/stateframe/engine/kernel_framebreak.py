"""Deterministic frame-breaking questions for exhausted composed replans.

Frame breaking asks whether the question itself is wrong.  This module is a
shadow organ: it extracts only receipt/task material, emits bounded advisory
questions, and never changes a plan or treats model prose as evidence.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState


SCHEMA = "poc.frame_challenge/1"
MIN_REPLANS = 2
MAX_ATOMS = 8
MAX_OPTIONS = 24
MAX_QUESTION_CHARS = 200
MAX_CHALLENGE_CHARS = 900
MAX_DOSSIER_TAIL_CHARS = 420

Record: TypeAlias = Mapping[str, object]

_FAILURE_RESULTS = frozenset({"RED", "PRODUCT_RED", "FAILED", "FAILURE", "ERROR", "NO_HITS", "BLOCKED", "REJECTED"})
_SUCCESS_RESULTS = frozenset({"GREEN", "PRODUCT_GREEN", "PASS", "SUCCESS", "OK", "HITS", "CERTIFIED"})
_TARGET_PATTERN = re.compile(
    r"(?P<file>[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+)"
    r"(?::(?P<line>\d+))?(?::(?P<function>[A-Za-z_][A-Za-z0-9_]*))?"
)


class FrameBreakError(ValueError):
    """Raised when a frame cannot satisfy its bounded contract."""


class FrameOperator(StrEnum):
    """The six bounded ways to ask whether the current frame is wrong."""

    ASSUMPTION_INVERSION = "assumption_inversion"
    OPTION_ESCAPE = "option_escape"
    ROOT_CAUSE_LIFT = "root_cause_lift"
    GOAL_CORRECTION = "goal_correction"
    CONSTRAINT_CHALLENGE = "constraint_challenge"
    SCOPE_SHIFT = "scope_shift"


def _text(value: object, limit: int = 240) -> str:
    rendered = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(rendered) <= limit:
        return rendered
    return rendered[: max(1, limit - 1)].rstrip() + "…"


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _text(value)


def canonical_json(value: object) -> str:
    """Serialize frame sidecars without ordering or timestamp noise."""

    return json.dumps(_canonical(value), ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _records(value: object) -> tuple[Record, ...]:
    if isinstance(value, Mapping):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _values(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        for key in ("atoms", "values", "items", "claims"):
            if key in value:
                return _values(value[key])
        return (value,)
    if isinstance(value, str):
        return tuple(part.strip() for part in re.split(r"[;\n]+", value) if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(value)
    return () if value is None else (value,)


def _first(mapping: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", (), [], {}):
            return value
    return None


def _ordered_receipts(receipts: Iterable[object]) -> tuple[Record, ...]:
    indexed = [(index, record) for index, record in enumerate(_records(tuple(receipts)))]
    return tuple(
        record
        for _, record in sorted(
            indexed,
            key=lambda pair: (_int(pair[1].get("turn")), pair[0]),
        )
    )


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _parse_target(value: object) -> tuple[str, str]:
    rendered = _text(value, 260)
    match = _TARGET_PATTERN.search(rendered)
    if match is None:
        return "", ""
    return _text(match.group("file"), 220), _text(match.group("function"), 120)


def _target_parts(task_row: Record, receipts: Sequence[Record]) -> tuple[str, str]:
    target = _first(task_row, "target", "target_ref", "assumed_target")
    target_mapping = target if isinstance(target, Mapping) else {}
    file_value = _first(task_row, "target_file", "file", "path") or _first(target_mapping, "file", "path", "target_file")
    function_value = _first(task_row, "target_function", "target_fn", "function", "fn", "symbol") or _first(
        target_mapping, "function", "fn", "symbol", "target_function"
    )
    target_file = _text(file_value, 220)
    target_function = _text(function_value, 120)
    if target_file and target_function:
        return target_file, target_function
    parsed = _parse_target(target)
    if parsed == ("", ""):
        for receipt in receipts:
            parsed = _parse_target(_first(receipt, "anchor", "intent", "target", "path"))
            if parsed != ("", ""):
                break
    return target_file or parsed[0], target_function or parsed[1]


def _question_atoms(task_row: Record, receipts: Sequence[Record]) -> tuple[str, ...]:
    raw = _first(task_row, "oracle_atoms", "acceptance_atoms", "current_question_atoms", "question_atoms", "current_question", "question", "oracle")
    values = _values(raw)
    atoms = [_text(item, 180) for item in values if _text(item, 180)]
    if not atoms:
        for receipt in receipts:
            if _text(receipt.get("result_class") or receipt.get("result")).upper() in _FAILURE_RESULTS:
                atoms.extend(
                    _text(receipt.get(key), 180)
                    for key in ("test_selector", "intent", "expected")
                    if _text(receipt.get(key), 180)
                )
                if atoms:
                    break
    return tuple(dict.fromkeys(atoms))[:MAX_ATOMS] or ("the demanded behavior",)


def _top_hypothesis(task_row: Record, receipts: Sequence[Record]) -> tuple[str, str]:
    direct = _first(task_row, "top_hypothesis", "assumed_cause", "top_hypothesis_claim", "cause")
    candidate: object = direct
    if candidate is None:
        hypotheses = _values(_first(task_row, "hypotheses", "claims"))
        mappings = [item for item in hypotheses if isinstance(item, Mapping)]
        if mappings:
            candidate = sorted(mappings, key=lambda item: (_int(item.get("rank", item.get("priority", 0))), str(item.get("id", item.get("claim_id", "")))))[0]
    if isinstance(candidate, Mapping):
        claim = _text(_first(candidate, "claim", "text", "assumed_cause", "description"), 220)
        claim_id = _text(_first(candidate, "claim_id", "hypothesis_id", "id", "key"), 120)
    else:
        claim = _text(candidate, 220)
        claim_id = ""
    if not claim:
        for receipt in receipts:
            if _text(receipt.get("result_class") or receipt.get("result")).upper() in _FAILURE_RESULTS:
                claim = _text(receipt.get("why") or receipt.get("detail") or receipt.get("intent"), 220)
                if claim:
                    break
    claim = claim or "the target boundary owns the failure"
    return claim_id or "hypothesis:" + _digest({"claim": claim}).removeprefix("sha256:")[:16], claim


def _fingerprint_family(record: Record) -> str:
    explicit = _first(record, "failing_fingerprint_family", "fingerprint_family", "failure_family", "family")
    if explicit is not None:
        return _text(explicit, 160)
    return _text(record.get("fingerprint"), 160)


@dataclass(frozen=True, slots=True)
class Frame:
    """The deterministic question frame used by all six operators."""

    oracle_atoms: tuple[str, ...]
    assumed_cause_id: str
    assumed_cause: str
    target_file: str
    target_function: str
    assumed_constraints: tuple[str, ...] = ()
    considered_options: tuple[str, ...] = ()
    failing_fingerprint_family: str = ""

    def __post_init__(self) -> None:
        atoms = tuple(dict.fromkeys(_text(item, 180) for item in self.oracle_atoms if _text(item, 180)))[:MAX_ATOMS]
        options = tuple(dict.fromkeys(_text(item, 160) for item in self.considered_options if _text(item, 160)))[:MAX_OPTIONS]
        constraints = tuple(dict.fromkeys(_text(item, 180) for item in self.assumed_constraints if _text(item, 180)))[:6]
        cause = _text(self.assumed_cause, 220)
        cause_id = _text(self.assumed_cause_id, 120)
        if not atoms or not cause_id or not cause:
            raise FrameBreakError("frame requires oracle atoms and a top hypothesis claim")
        object.__setattr__(self, "oracle_atoms", atoms)
        object.__setattr__(self, "assumed_cause_id", cause_id)
        object.__setattr__(self, "assumed_cause", cause)
        object.__setattr__(self, "target_file", _text(self.target_file, 220) or "unspecified target file")
        object.__setattr__(self, "target_function", _text(self.target_function, 120) or "unspecified target function")
        object.__setattr__(self, "assumed_constraints", constraints)
        object.__setattr__(self, "considered_options", options)
        object.__setattr__(self, "failing_fingerprint_family", _text(self.failing_fingerprint_family, 160) or "no-failing-fingerprint")

    @property
    def current_question(self) -> tuple[str, ...]:
        return self.oracle_atoms

    @property
    def assumed_cause_claim(self) -> str:
        return self.assumed_cause

    @property
    def options(self) -> tuple[str, ...]:
        return self.considered_options

    @property
    def frame_id(self) -> str:
        return "frame:" + _digest({"claim_id": self.assumed_cause_id, "fingerprint_family": self.failing_fingerprint_family}).removeprefix("sha256:")[:24]

    @property
    def target_scope(self) -> str:
        return f"{self.target_file}:{self.target_function}"

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "oracle_atoms": list(self.oracle_atoms),
            "current_question": list(self.current_question),
            "assumed_cause_id": self.assumed_cause_id,
            "assumed_cause": self.assumed_cause,
            "target_file": self.target_file,
            "target_function": self.target_function,
            "assumed_constraints": list(self.assumed_constraints),
            "considered_options": list(self.considered_options),
            "failing_fingerprint_family": self.failing_fingerprint_family,
        }


def extract_frame(task_row: Mapping[str, object], receipts: Iterable[object] = ()) -> Frame:
    """Extract a stable frame from task material and ordered receipt evidence."""

    if not isinstance(task_row, Mapping):
        raise FrameBreakError("task_row must be a mapping")
    ordered = _ordered_receipts(receipts)
    target_file, target_function = _target_parts(task_row, ordered)
    cause_id, cause = _top_hypothesis(task_row, ordered)
    raw_constraints = _first(task_row, "assumed_constraints", "constraints")
    constraints = tuple(_text(item, 180) for item in _values(raw_constraints) if _text(item, 180))
    constraints = tuple(dict.fromkeys((*constraints, *(item for item in (target_file, target_function) if item))))
    raw_options = _first(task_row, "considered_options", "attempt_fingerprints", "option_fingerprints", "options")
    options = [_text(item, 160) for item in _values(raw_options) if _text(item, 160)]
    if not options:
        options = [
            _text(receipt.get("fingerprint"), 160)
            for receipt in ordered
            if _text(receipt.get("fingerprint"), 160)
            and _text(receipt.get("kind")).lower() in {"edit", "test", "grep"}
        ]
    if not options:
        options = [_text(receipt.get("fingerprint"), 160) for receipt in ordered if _text(receipt.get("fingerprint"), 160)]
    failure_family = _text(_first(task_row, "failing_fingerprint_family", "fingerprint_family", "failure_family"), 160)
    if not failure_family:
        for receipt in ordered:
            result = _text(receipt.get("result_class") or receipt.get("result")).upper()
            if result in _FAILURE_RESULTS and _text(receipt.get("fingerprint"), 160):
                failure_family = _fingerprint_family(receipt)
                break
    return Frame(
        oracle_atoms=_question_atoms(task_row, ordered),
        assumed_cause_id=cause_id,
        assumed_cause=cause,
        target_file=target_file,
        target_function=target_function,
        assumed_constraints=constraints,
        considered_options=options,
        failing_fingerprint_family=failure_family,
    )


@dataclass(frozen=True, slots=True)
class OperatorQuestion:
    operator: FrameOperator
    question: str

    def __post_init__(self) -> None:
        try:
            operator = self.operator if isinstance(self.operator, FrameOperator) else FrameOperator(str(self.operator))
        except ValueError as exc:
            raise FrameBreakError(f"unknown frame operator: {self.operator!r}") from exc
        question = _text(self.question, MAX_QUESTION_CHARS).rstrip("?") + "?"
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "question", question[: MAX_QUESTION_CHARS - 1].rstrip("?") + "?")

    def to_dict(self) -> dict[str, str]:
        return {"operator": self.operator.value, "question": self.question}


def _atom(frame: Frame) -> str:
    return _text(frame.oracle_atoms[0], 72)


def _options(frame: Frame) -> str:
    return ",".join(frame.considered_options[:3]) or "no recorded option"


def render_operator_questions(frame: Frame) -> tuple[OperatorQuestion, ...]:
    """Render exactly six grounded, bounded question blocks."""

    atom = _atom(frame)
    cause = _text(frame.assumed_cause, 72)
    scope = _text(frame.target_scope, 86)
    constraints = _text(", ".join(frame.assumed_constraints) or scope, 80)
    options = _text(_options(frame), 64)
    raw = (
        (FrameOperator.ASSUMPTION_INVERSION, f"If '{cause}' is false, what different boundary could explain '{atom}'?") ,
        (FrameOperator.OPTION_ESCAPE, f"What untried option beyond [{options}] tests '{atom}' without editing {scope}?") ,
        (FrameOperator.ROOT_CAUSE_LIFT, f"Could '{cause}' be wrong at caller/contract scope {scope}: what boundary sets '{atom}'?") ,
        (FrameOperator.GOAL_CORRECTION, f"Is the goal wrong: should '{atom}' mean a different observable contract than the one being tested?") ,
        (FrameOperator.CONSTRAINT_CHALLENGE, f"Does the constraint '{constraints}' hold, or does it hide the boundary causing '{atom}'?") ,
        (FrameOperator.SCOPE_SHIFT, f"If {scope} is locally correct, which caller/contract scope next owns '{atom}'?") ,
    )
    return tuple(OperatorQuestion(operator, question) for operator, question in raw)


@dataclass(frozen=True, slots=True)
class FrameBreakDecision:
    fired: bool
    reason: str
    frame_id: str
    replan_count: int = 0
    challenge_once: bool = False

    @property
    def should_fire(self) -> bool:
        return self.fired

    def to_dict(self) -> dict[str, object]:
        return {
            "fired": self.fired,
            "should_fire": self.should_fire,
            "reason": self.reason,
            "frame_id": self.frame_id,
            "replan_count": self.replan_count,
            "challenge_once": self.challenge_once,
        }


def _replan_fired(record: Record) -> bool:
    event = _text(record.get("event") or record.get("event_name")).upper()
    if event == "SNITCH_REPLANNER_TRIGGERED" or event == "FRAME_REPLAN_TRIGGERED":
        emitted = record.get("emitted", True)
        return not (emitted is False or _text(emitted).casefold() in {"false", "no", "0"})
    return bool(record.get("replan_fired") or record.get("replan_triggered") or record.get("should_fire"))


def _frame_fields(record: Record) -> tuple[str, str, str]:
    detail = record.get("detail") if isinstance(record.get("detail"), Mapping) else {}
    frame_id = _text(_first(record, "frame_id", "frame") or _first(detail, "frame_id", "frame"), 160)
    claim_id = _text(_first(record, "assumed_cause_id", "top_hypothesis_claim_id", "hypothesis_id", "claim_id") or _first(detail, "assumed_cause_id", "top_hypothesis_claim_id", "hypothesis_id", "claim_id"), 120)
    family = _text(_first(record, "failing_fingerprint_family", "fingerprint_family", "failure_family", "fingerprint") or _first(detail, "failing_fingerprint_family", "fingerprint_family", "failure_family", "fingerprint"), 160)
    return frame_id, claim_id, family


def _matches_frame(record: Record, frame: Frame) -> bool:
    frame_id, claim_id, family = _frame_fields(record)
    if frame_id:
        return frame_id == frame.frame_id
    return bool(claim_id and family) and claim_id == frame.assumed_cause_id and family == frame.failing_fingerprint_family


def evaluate_frame_break_trigger(
    frame: Frame,
    composed_replans: Iterable[object],
    *,
    challenged_frames: Iterable[object] = (),
) -> FrameBreakDecision:
    """Fire only after two same-frame composed replans, capped once per frame."""

    challenged = {
        _text(item.get("frame_id"), 160) if isinstance(item, Mapping) else _text(item, 160)
        for item in challenged_frames
    }
    if frame.frame_id in challenged:
        return FrameBreakDecision(False, "frame_challenge_already_emitted", frame.frame_id)
    matching = sum(1 for item in _records(tuple(composed_replans)) if _replan_fired(item) and _matches_frame(item, frame))
    if matching < MIN_REPLANS:
        return FrameBreakDecision(False, "same_frame_replan_threshold_not_met", frame.frame_id, matching)
    return FrameBreakDecision(True, "same_frame_replan_convergence_exhausted", frame.frame_id, matching, challenge_once=True)


trigger_frame_break = evaluate_frame_break_trigger
frame_break_trigger = evaluate_frame_break_trigger


def _frame_summary(frame: Frame) -> str:
    return _text(
        f"FRAME {frame.frame_id} | question={'; '.join(frame.oracle_atoms)} | cause={frame.assumed_cause_id}:{frame.assumed_cause} | target={frame.target_scope} | family={frame.failing_fingerprint_family} | options={_options(frame)}",
        320,
    )


@dataclass(frozen=True, slots=True)
class FrameChallenge:
    """Bounded human block plus its machine-readable escalation sidecar."""

    frame: Frame
    decision: FrameBreakDecision
    questions: tuple[OperatorQuestion, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        questions = self.questions or render_operator_questions(self.frame)
        if len(questions) != 6:
            raise FrameBreakError("frame challenge requires six operator questions")
        object.__setattr__(self, "questions", tuple(questions))

    @property
    def summary(self) -> str:
        return _frame_summary(self.frame)

    def render(self, *, max_chars: int = MAX_CHALLENGE_CHARS) -> str:
        limit = max(1, min(int(max_chars), MAX_CHALLENGE_CHARS))
        header = "FRAME CHALLENGE (advisory; signals only)"
        summary = _text(self.summary, 250)
        budget = max(72, (limit - len(header) - len(summary) - 16) // 6)
        lines = [header, summary]
        lines.extend(f"{item.operator.value}: {_text(item.question, budget)}" for item in self.questions)
        rendered = "\n".join(lines)
        if len(rendered) <= limit:
            return rendered
        return "\n".join([header, _text(summary, 180), *[f"{item.operator.value}: {_text(item.question, 72)}" for item in self.questions]])[:limit]

    @property
    def block(self) -> str:
        return self.render()

    @property
    def dossier_tail(self) -> str:
        return _text("; ".join((self.summary, *(item.question for item in self.questions))), MAX_DOSSIER_TAIL_CHARS)

    def to_sidecar(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "frame_id": self.frame.frame_id,
            "frame": self.frame.to_dict(),
            "decision": self.decision.to_dict(),
            "questions": [item.to_dict() for item in self.questions],
            "dossier_tail": self.dossier_tail,
        }

    @property
    def machine_sidecar(self) -> dict[str, object]:
        return self.to_sidecar()

    @property
    def sidecar(self) -> dict[str, object]:
        return self.to_sidecar()

    def sidecar_json(self) -> str:
        return canonical_json(self.to_sidecar())


def build_frame_challenge(
    task_row: Mapping[str, object],
    receipts: Iterable[object],
    composed_replans: Iterable[object],
    *,
    challenged_frames: Iterable[object] = (),
) -> FrameChallenge | None:
    """Extract and challenge a frame only when the composed trigger licenses it."""

    frame = extract_frame(task_row, receipts)
    decision = evaluate_frame_break_trigger(frame, composed_replans, challenged_frames=challenged_frames)
    return FrameChallenge(frame, decision) if decision.fired else None


class FrameEvidenceScore:
    """Unique receipt-fingerprint counts for a candidate frame."""

    def __init__(self, supporting_fingerprints: Iterable[str] = (), contradicting_fingerprints: Iterable[str] = ()) -> None:
        self.supporting_fingerprints = tuple(dict.fromkeys(_text(item, 160) for item in supporting_fingerprints if _text(item, 160)))
        self.contradicting_fingerprints = tuple(dict.fromkeys(_text(item, 160) for item in contradicting_fingerprints if _text(item, 160)))

    @property
    def supporting(self) -> int:
        return len(self.supporting_fingerprints)

    @property
    def supporting_count(self) -> int:
        return self.supporting

    @property
    def contradicting(self) -> int:
        return len(self.contradicting_fingerprints)

    @property
    def contradicting_count(self) -> int:
        return self.contradicting

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, object]:
        return {
            "supporting": self.supporting,
            "contradicting": self.contradicting,
            "supporting_count": self.supporting,
            "contradicting_count": self.contradicting,
            "supporting_fingerprints": list(self.supporting_fingerprints),
            "contradicting_fingerprints": list(self.contradicting_fingerprints),
        }


def _receipt_text(receipt: Record) -> str:
    return canonical_json(receipt).casefold()


def _frame_terms(frame: Frame) -> tuple[str, ...]:
    values = (*frame.oracle_atoms, frame.assumed_cause, frame.target_file, frame.target_function, frame.failing_fingerprint_family)
    terms: list[str] = []
    for value in values:
        rendered = _text(value).casefold()
        if len(rendered) >= 5:
            terms.append(rendered)
        terms.extend(token for token in re.findall(r"[a-z0-9_/-]{5,}", rendered) if token not in {"which", "could", "different", "target"})
    return tuple(dict.fromkeys(terms))


def score_candidate_frame(frame: Frame, receipts: Iterable[object]) -> FrameEvidenceScore:
    """Count unique receipt fingerprints that support or contradict a frame.

    Matching is case-insensitive substring/atom matching; explicit receipt
    labels win, and otherwise failure/success result classes determine polarity.
    """

    supporting: list[str] = []
    contradicting: list[str] = []
    terms = _frame_terms(frame)
    for receipt in _ordered_receipts(receipts):
        fingerprint = _text(receipt.get("fingerprint"), 160)
        if not fingerprint or not any(term in _receipt_text(receipt) for term in terms):
            continue
        explicit_support = receipt.get("supports_frame") is True or receipt.get("frame_support") is True
        explicit_contradiction = receipt.get("contradicts_frame") is True or receipt.get("frame_contradiction") is True
        result = _text(receipt.get("result_class") or receipt.get("result")).upper()
        if explicit_support or (not explicit_contradiction and result in _FAILURE_RESULTS):
            supporting.append(fingerprint)
        if explicit_contradiction or (not explicit_support and result in _SUCCESS_RESULTS):
            contradicting.append(fingerprint)
    return FrameEvidenceScore(supporting, contradicting)


def _framebreak_trigger(evidence: Evidence) -> TriState:
    event = _text(evidence.event.get("event") or evidence.event.get("event_name")).upper()
    if event == "FRAME_BREAK_TRIGGERED":
        return TriState.YES
    if event in {"FRAME_BREAK_SUPPRESSED", "FRAME_BREAK_DEDUPED", "FRAME_BREAK_CHALLENGE_ONCE"}:
        return TriState.NO
    return TriState.UNKNOWN if not evidence.complete else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="bolt.framebreak",
    trigger=_framebreak_trigger,
    fire_events=frozenset({"FRAME_BREAK_TRIGGERED", "FRAME_CHALLENGE_DELIVERED"}),
    suppression_events=frozenset({"FRAME_BREAK_SUPPRESSED", "FRAME_BREAK_DEDUPED"}),
    dedup_events=frozenset({"FRAME_BREAK_DEDUPED"}),
    challenge_once_events=frozenset({"FRAME_BREAK_CHALLENGE_ONCE"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


__all__ = [
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "Frame",
    "FrameBreakDecision",
    "FrameBreakError",
    "FrameChallenge",
    "FrameEvidenceScore",
    "FrameOperator",
    "MIN_REPLANS",
    "OperatorQuestion",
    "build_frame_challenge",
    "canonical_json",
    "evaluate_frame_break_trigger",
    "extract_frame",
    "frame_break_trigger",
    "render_operator_questions",
    "score_candidate_frame",
    "trigger_frame_break",
]
