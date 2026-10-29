"""Execution-selected candidate patches for ticketed, reproduced bug fixes.

BESTOFN is deliberately a host-owned reducer.  Candidate text is produced by a
provider, but candidates are never trusted in the real workspace: every sample
is checked in its own disposable copy, receipts are journaled, and only the
ranked result is published.  The module has no benchmark, grader, hidden, or
gold-data inputs.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from kernel_repro_first import requires_repro


SCHEMA = "stateframe-bestofn/1"
MODE_ENV = "STATEFRAME_BESTOFN"
DEFAULT_MAX_CANDIDATES = 3
MAX_CANDIDATES = 3
DEFAULT_ALLOWANCE = 0.20
_FORBIDDEN_SCOPE_PARTS = frozenset({"backups", "grader", "graders", "hidden", "gold", "private"})
_PASS_VALUES = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "OK", "CERTIFIED"})
_FAIL_VALUES = frozenset({"FAIL", "FAILED", "RED", "ERROR", "NO_TESTS", "TIMEOUT"})


class BestOfNError(ValueError):
    """Raised for an unsafe or malformed candidate contract."""


@dataclass(frozen=True)
class CandidateRequest:
    """One independent sample request sharing one immutable prompt context."""

    candidate_index: int
    seed: int
    prompt: str
    prefix_cache_key: str
    ticket_id: str
    context: object


@dataclass(frozen=True)
class CandidatePatch:
    candidate_id: str
    files: Mapping[str, str]
    seed: int | None = None
    diff_size: int | None = None


@dataclass(frozen=True)
class CheckResult:
    status: str
    output: str = ""
    receipt: Mapping[str, object] | None = None

    @property
    def passed(self) -> bool:
        return self.status.upper() in _PASS_VALUES


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    candidate_index: int
    seed: int | None
    repro: CheckResult
    public: tuple[CheckResult, ...]
    repro_pass: bool
    public_green: bool
    diff_size: int
    status: str
    receipt_id: str

    @property
    def rank(self) -> tuple[int, int, int, int]:
        # Lower tuples win after the two boolean priorities are inverted.
        return (
            0 if self.repro_pass else 1,
            0 if self.public_green else 1,
            self.diff_size,
            self.candidate_index,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "seed": self.seed,
            "repro": asdict(self.repro),
            "public": [asdict(item) for item in self.public],
            "repro_pass": self.repro_pass,
            "public_green": self.public_green,
            "diff_size": self.diff_size,
            "status": self.status,
            "receipt_id": self.receipt_id,
            "rank": list(self.rank),
        }


@dataclass(frozen=True)
class SelectionResult:
    status: str
    winner: CandidatePatch | None
    winner_evaluation: CandidateEvaluation | None
    evaluations: tuple[CandidateEvaluation, ...]
    candidates_requested: int
    candidates_evaluated: int
    disclosure: str
    applied: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "winner": asdict(self.winner) if self.winner is not None else None,
            "winner_evaluation": (
                self.winner_evaluation.to_dict()
                if self.winner_evaluation is not None else None
            ),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "candidates_requested": self.candidates_requested,
            "candidates_evaluated": self.candidates_evaluated,
            "disclosure": self.disclosure,
            "applied": self.applied,
            "fallback_reason": self.fallback_reason,
        }


def mode() -> str:
    value = os.environ.get(MODE_ENV, "auto").strip().lower()
    if value not in {"auto", "off", "on"}:
        raise ValueError(f"{MODE_ENV} must be auto, off, or on")
    return value


def ticketed_repro(ticket: object, repro: object) -> bool:
    """Return true only for a typed bug ticket with a REPROFIRST receipt."""
    if not requires_repro(ticket) or repro is None:
        return False
    if isinstance(repro, Mapping):
        if repro.get("receipt") is not None:
            return True
        return str(repro.get("status", "")).upper() in {"RED", "GREEN", "REPRODUCED"}
    return getattr(repro, "receipt", None) is not None or bool(
        str(getattr(repro, "status", "")).upper() in {"RED", "GREEN", "REPRODUCED"}
    )


def enabled(ticket: object, repro: object, *, selected_mode: str | None = None) -> bool:
    selected = mode() if selected_mode is None else selected_mode
    if selected not in {"auto", "off", "on"}:
        raise ValueError("selected_mode must be auto, off, or on")
    return selected != "off" and ticketed_repro(ticket, repro)


def _ticket_id(ticket: object) -> str:
    if isinstance(ticket, Mapping):
        for key in ("task_id", "ticket_id", "id"):
            if ticket.get(key) not in (None, ""):
                return str(ticket[key])
    for key in ("task_id", "ticket_id", "id"):
        value = getattr(ticket, key, None)
        if value not in (None, ""):
            return str(value)
    return "ticket"


def _prompt_context(context: object) -> tuple[str, object]:
    if isinstance(context, Mapping):
        prompt = context.get("prompt", context.get("messages", ""))
    else:
        prompt = context
    if isinstance(prompt, (list, tuple)):
        prompt_text = json.dumps(prompt, sort_keys=True, ensure_ascii=False, default=str)
    else:
        prompt_text = str(prompt or "")
    return prompt_text, context


def _prefix_key(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _safe_relative_path(value: object) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    path = Path(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BestOfNError(f"candidate path is not workspace-relative: {raw!r}")
    if any(part.casefold() in _FORBIDDEN_SCOPE_PARTS for part in path.parts):
        raise BestOfNError("candidate path is outside the public ticket workspace")
    return path.as_posix()


def _assert_workspace_root(root: Path) -> Path:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise BestOfNError("BESTOFN workspace must be an existing directory")
    return resolved


def _coerce_candidate(value: object, *, index: int, seed: int) -> CandidatePatch:
    if isinstance(value, CandidatePatch):
        candidate = value
    elif isinstance(value, Mapping):
        candidate_id = str(value.get("candidate_id", value.get("id", f"candidate-{index}")))
        files_value = value.get("files")
        if isinstance(files_value, Mapping):
            files = {str(path): str(source) for path, source in files_value.items()}
        elif value.get("path") is not None and value.get("source") is not None:
            files = {str(value["path"]): str(value["source"])}
        elif value.get("path") is not None and value.get("content") is not None:
            files = {str(value["path"]): str(value["content"])}
        else:
            raise BestOfNError("candidate must provide files or path+source")
        raw_size = value.get("diff_size")
        candidate = CandidatePatch(
            candidate_id=candidate_id,
            files=files,
            seed=(int(value["seed"]) if value.get("seed") is not None else seed),
            diff_size=(int(raw_size) if raw_size is not None else None),
        )
    else:
        raise BestOfNError("candidate provider returned an unsupported patch")
    if not candidate.files:
        raise BestOfNError("candidate patch must contain at least one file")
    safe_files = {_safe_relative_path(path): str(source)
                  for path, source in candidate.files.items()}
    return CandidatePatch(
        candidate_id=str(candidate.candidate_id or f"candidate-{index}"),
        files=safe_files,
        seed=candidate.seed if candidate.seed is not None else seed,
        diff_size=candidate.diff_size,
    )


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "__pycache__", ".pytest_cache", ".test-tmp", "htmlcov",
               ".ruff_cache", ".mypy_cache", ".coverage"}
    result = {name for name in names if name in ignored or name.startswith(".coverage.")}
    result.update(name for name in names if (Path(directory) / name).is_symlink())
    return result


def _apply_to_copy(root: Path, candidate: CandidatePatch) -> None:
    root = _assert_workspace_root(root)
    for relative, source in candidate.files.items():
        path = root / _safe_relative_path(relative)
        cursor = root
        for part in path.relative_to(root).parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise BestOfNError("candidate cannot write through a symlinked directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise BestOfNError("candidate cannot write through a symlink")
        path.write_text(source, encoding="utf-8")


def apply_candidate(root: Path, candidate: CandidatePatch) -> None:
    """Apply one validated winner to the real workspace, without overwrite tricks."""
    _apply_to_copy(_assert_workspace_root(Path(root)), candidate)


def _diff_size(root: Path, candidate: CandidatePatch) -> int:
    if candidate.diff_size is not None:
        if type(candidate.diff_size) is not int or candidate.diff_size < 0:
            raise BestOfNError("diff_size must be a non-negative integer")
        return candidate.diff_size
    total = 0
    for relative, source in candidate.files.items():
        path = root / relative
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        matcher = difflib.SequenceMatcher(a=before, b=source, autojunk=False)
        total += sum(max(end_a - start_a, end_b - start_b)
                     for tag, start_a, end_a, start_b, end_b in matcher.get_opcodes()
                     if tag != "equal")
    return total


def _check_result(value: object) -> CheckResult:
    if isinstance(value, CheckResult):
        return value
    if isinstance(value, bool):
        return CheckResult("PASS" if value else "FAIL")
    if isinstance(value, Mapping):
        raw_status = value.get("status", value.get("result", value.get("outcome", "UNKNOWN")))
        output = value.get("output", value.get("observation", value.get("detail", "")))
        receipt = value.get("receipt")
        return CheckResult(
            status=str(raw_status or "UNKNOWN").upper(),
            output=str(output or ""),
            receipt=receipt if isinstance(receipt, Mapping) else None,
        )
    return CheckResult(status="PASS" if str(value).strip().upper() in _PASS_VALUES else "UNKNOWN",
                       output=str(value or ""))


def _check_name(check: object, index: int) -> str:
    value = check.get("name", check.get("selector", f"public-{index}")) if isinstance(check, Mapping) else check
    name = str(value or f"public-{index}")
    if any(part.casefold() in _FORBIDDEN_SCOPE_PARTS for part in Path(name).parts):
        raise BestOfNError("a non-public check was supplied to BESTOFN")
    return name


def _emit(journal: object | None, name: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(name, producer="kernel_bestofn", event_schema=SCHEMA,
             dispatch=False, **fields)


def _budget_limit(budget: object | None) -> int | None:
    for owner in (budget,):
        value = getattr(owner, "limit", None)
        if type(value) is int and value >= 0:
            return value
        snapshot = getattr(owner, "continuation_snapshot", None)
        if callable(snapshot):
            data = snapshot()
            if isinstance(data, Mapping):
                value = data.get("model_decision_limit")
                if type(value) is int and value >= 0:
                    return value
    return None


def _candidate_cap(budget: object | None, requested: int, allowance: float) -> int:
    limit = _budget_limit(budget)
    if limit is None:
        return requested
    allowance_cap = max(1, 1 + math.floor(limit * allowance))
    return min(requested, allowance_cap)


def _admit_extra_candidate(budget: object | None, check_count: int) -> bool:
    if budget is None:
        return True
    qualify = getattr(budget, "qualify_admission", None)
    if callable(qualify):
        try:
            if not bool(qualify(model_decisions=1, automatic_checks=1 + check_count)):
                return False
        except (TypeError, ValueError, AttributeError):
            return False
    charge = getattr(budget, "charge_response", None)
    if callable(charge):
        try:
            if not bool(charge()):
                return False
        except (TypeError, ValueError, AttributeError):
            return False
    return True


def _admit_check(budget: object | None, selector: str) -> bool:
    if budget is None:
        return True
    charge = getattr(budget, "charge_automatic_check", None)
    if not callable(charge):
        return True
    try:
        return bool(charge(selector=selector))
    except (TypeError, ValueError, AttributeError):
        return False


def _candidate_budget_cap(
        candidate_budget: object | None,
        candidate_cost: Mapping[str, object] | None,
        requested: int,
        check_count: int,
) -> tuple[int, str | None]:
    """Admit samples only when token, time, check, and reserve costs are known."""
    if candidate_budget is None:
        return requested, None
    if not isinstance(candidate_cost, Mapping):
        return 0, "candidate cost telemetry is required for BESTOFN budget admission"
    tokens_each = candidate_cost.get("tokens_each")
    seconds_each = candidate_cost.get("seconds_each")
    checks_each = candidate_cost.get("checks_each", 1 + check_count)
    if (
        type(tokens_each) is not int or tokens_each < 0
        or type(checks_each) is not int or checks_each < 0
        or not isinstance(seconds_each, (int, float)) or seconds_each < 0
    ):
        return 0, "candidate token, wall-time, and check costs must be measured"
    limit = getattr(candidate_budget, "limit", None)
    if not callable(limit):
        return 0, "candidate budget has no host limit predicate"
    try:
        admitted = limit(
            requested=requested,
            tokens_each=tokens_each,
            seconds_each=float(seconds_each),
            checks_each=checks_each,
            hard_max=MAX_CANDIDATES,
        )
    except (TypeError, ValueError, AttributeError):
        return 0, "candidate budget could not evaluate measured costs"
    if type(admitted) is not int or admitted < 0:
        return 0, "candidate budget returned an invalid limit"
    if admitted == 0:
        return 0, "candidate budget cannot fund a sample without consuming verification reserve"
    return min(requested, admitted), None


def _evaluate_candidate(
        root: Path,
        candidate: CandidatePatch,
        *,
        index: int,
        repro: object,
        public_checks: Sequence[object],
        check_runner: Callable[[Path, object], object],
        budget: object | None,
        journal: object | None,
) -> CandidateEvaluation:
    """Run one candidate in isolation and mint its host receipt."""
    diff_size = _diff_size(root, candidate)
    with tempfile.TemporaryDirectory(prefix="stateframe-bestofn-") as directory:
        disposable = Path(directory) / "workspace"
        shutil.copytree(root, disposable, ignore=_copy_ignore, symlinks=True)
        _apply_to_copy(disposable, candidate)
        if not _admit_check(budget, "repro"):
            repro_result = CheckResult("UNKNOWN", "reproduction check not funded")
        else:
            repro_result = _check_result(check_runner(disposable, repro))
        public_results: list[CheckResult] = []
        for check_index, check in enumerate(public_checks):
            selector = _check_name(check, check_index)
            if not _admit_check(budget, selector):
                public_results.append(CheckResult("UNKNOWN", "public check not funded"))
                continue
            public_results.append(_check_result(check_runner(disposable, check)))
    repro_pass = repro_result.passed
    public_green = bool(public_results) and all(item.passed for item in public_results)
    if not public_results:
        public_green = True
    status = "PASS" if repro_pass and public_green else "PARTIAL" if public_green or repro_pass else "FAIL"
    identity = {
        "candidate_id": candidate.candidate_id,
        "candidate_index": index,
        "seed": candidate.seed,
        "repro": asdict(repro_result),
        "public": [asdict(item) for item in public_results],
    }
    receipt_id = "bestofn-receipt:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:24]
    evaluation = CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        candidate_index=index,
        seed=candidate.seed,
        repro=repro_result,
        public=tuple(public_results),
        repro_pass=repro_pass,
        public_green=public_green,
        diff_size=diff_size,
        status=status,
        receipt_id=receipt_id,
    )
    _emit(journal, "bestofn_candidate_receipt", candidate=evaluation.to_dict(),
          candidate_id=candidate.candidate_id, candidate_index=index)
    return evaluation


def select_best_candidate(
        root: Path,
        *,
        ticket: object,
        repro: object,
        public_checks: Sequence[object],
        candidate_provider: Callable[[CandidateRequest], object],
        check_runner: Callable[[Path, object], object],
        initial_candidate: object | None = None,
        budget: object | None = None,
        journal: object | None = None,
        context: object = "",
        seed: int = 0,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        allowance: float = DEFAULT_ALLOWANCE,
        apply_winner: Callable[[Path, CandidatePatch], object] | None = None,
        selected_mode: str | None = None,
        candidate_budget: object | None = None,
        candidate_cost: Mapping[str, object] | None = None,
) -> SelectionResult:
    """Evaluate independent samples and publish only the ranked candidate.

    ``initial_candidate`` is the already-proposed model edit and does not spend
    another model decision.  Additional provider calls spend one decision and
    one bounded check unit for the reproduction plus each public selector.
    """
    if not enabled(ticket, repro, selected_mode=selected_mode):
        return SelectionResult("DISABLED", None, None, (), 0, 0,
                               "BESTOFN is inactive because no ticketed REPROFIRST receipt exists.",
                               False)
    if not isinstance(root, Path):
        root = Path(root)
    root = _assert_workspace_root(root)
    if type(max_candidates) is not int or not 1 <= max_candidates <= MAX_CANDIDATES:
        raise BestOfNError(f"max_candidates must be between 1 and {MAX_CANDIDATES}")
    if not isinstance(allowance, (int, float)) or not 0 <= float(allowance) <= 1:
        raise BestOfNError("allowance must be between 0 and 1")
    if not isinstance(public_checks, Sequence) or isinstance(public_checks, (str, bytes, bytearray)):
        raise BestOfNError("public_checks must be a sequence")
    prompt, stable_context = _prompt_context(context)
    request_cap = _candidate_cap(budget, max_candidates, float(allowance))
    request_cap, budget_reason = _candidate_budget_cap(
        candidate_budget, candidate_cost, request_cap, len(public_checks)
    )
    prefix_key = _prefix_key(prompt)
    ticket_id = _ticket_id(ticket)
    candidates: list[CandidatePatch] = []
    fallback_reason: str | None = budget_reason

    if request_cap == 0:
        disclosure = (budget_reason or "candidate budget did not admit a sample") + ". Product fix was not applied."
        return SelectionResult("BUDGET_UNKNOWN", None, None, (), 0, 0,
                               disclosure, False, budget_reason)

    if initial_candidate is not None:
        try:
            candidates.append(_coerce_candidate(initial_candidate, index=1, seed=seed))
        except BestOfNError as error:
            fallback_reason = f"initial candidate rejected: {error}"

    if budget is not None and request_cap < max_candidates:
        fallback_reason = (
            "continuation allowance limited BESTOFN to one sample"
            if request_cap == 1 else
            f"continuation allowance capped BESTOFN at {request_cap} samples"
        )

    evaluations: list[CandidateEvaluation] = []

    def evaluate(candidate: CandidatePatch, index: int) -> CandidateEvaluation:
        evaluation = _evaluate_candidate(
            root, candidate, index=index, repro=repro, public_checks=public_checks,
            check_runner=check_runner, budget=budget, journal=journal,
        )
        evaluations.append(evaluation)
        return evaluation

    def request_candidate(index: int) -> CandidatePatch | None:
        nonlocal fallback_reason
        if not _admit_extra_candidate(budget, len(public_checks)):
            fallback_reason = "continuation budget could not fund another independent candidate"
            return None
        request = CandidateRequest(
            candidate_index=index,
            seed=seed + index - 1,
            prompt=prompt,
            prefix_cache_key=prefix_key,
            ticket_id=ticket_id,
            context=stable_context,
        )
        _emit(journal, "bestofn_candidate_requested", candidate_index=index,
              seed=request.seed, prefix_cache_key=prefix_key, ticket_id=ticket_id)
        try:
            value = candidate_provider(request)
            candidate = _coerce_candidate(value, index=index, seed=request.seed)
            candidates.append(candidate)
            return candidate
        except BestOfNError as error:
            fallback_reason = f"candidate {index} rejected: {error}"
            _emit(journal, "bestofn_candidate_rejected", candidate_index=index,
                  reason=str(error))
            return None

    if candidate_budget is not None:
        # Measured budgeted sampling is sequential: the first fully eligible
        # candidate stops the provider before an unnecessary extra decision.
        next_index = 1
        if candidates:
            first = evaluate(candidates[0], 1)
            next_index = request_cap + 1 if first.repro_pass and first.public_green else 2
        while next_index <= request_cap:
            candidate = request_candidate(next_index)
            if candidate is not None:
                evaluation = evaluate(candidate, next_index)
                if evaluation.repro_pass and evaluation.public_green:
                    break
            next_index += 1
    else:
        next_index = 1 if not candidates else 2
        while next_index <= request_cap:
            candidate = request_candidate(next_index)
            next_index += 1
        for index, candidate in enumerate(candidates, start=1):
            evaluate(candidate, index)

    if not evaluations:
        disclosure = (fallback_reason or "no candidate could be evaluated") + ". Product fix was not applied."
        return SelectionResult("NO_CANDIDATE", None, None, (), request_cap, 0,
                               disclosure, False, fallback_reason)

    winner_evaluation = min(evaluations, key=lambda item: item.rank)
    winner = next(candidate for candidate in candidates
                  if candidate.candidate_id == winner_evaluation.candidate_id)
    all_repro_failed = not any(item.repro_pass for item in evaluations)
    failing_checks: list[str] = []
    if not winner_evaluation.repro_pass:
        failing_checks.append("reproduction")
    failing_checks.extend(
        _check_name(check, index)
        for index, check in enumerate(public_checks)
        if index >= len(winner_evaluation.public) or not winner_evaluation.public[index].passed
    )
    if all_repro_failed or not winner_evaluation.public_green:
        disclosure = (
            f"BESTOFN found no eligible candidate; no product candidate was applied. "
            f"Still failing: {', '.join(failing_checks) or 'reproduction'}."
        )
        status = "NO_ELIGIBLE_CANDIDATE"
        fallback_reason = fallback_reason or "no candidate satisfied reproduction and public checks"
        _emit(journal, "bestofn_no_eligible_candidate",
              candidates_evaluated=len(evaluations), disclosure=disclosure)
        return SelectionResult(status, None, None, tuple(evaluations), request_cap,
                               len(evaluations), disclosure, False, fallback_reason)
    elif fallback_reason is not None and len(evaluations) == 1:
        disclosure = f"BESTOFN used a single-sample fallback: {fallback_reason}."
        status = "SINGLE_SAMPLE_FALLBACK"
    else:
        disclosure = f"BESTOFN selected {winner.candidate_id} by reproduction, public-green, then diff size."
        status = "WINNER_SELECTED"
    try:
        applied_value = (apply_winner(root, winner)
                         if apply_winner is not None else (apply_candidate(root, winner), True)[1])
        applied = applied_value is not False
    except (BestOfNError, OSError, TypeError, ValueError) as error:
        applied = False
        disclosure += f" Winner was not applied: {type(error).__name__}."
    _emit(journal, "bestofn_winner_applied", candidate_id=winner.candidate_id,
          receipt_id=winner_evaluation.receipt_id, status=status, applied=applied,
          disclosure=disclosure)
    return SelectionResult(status, winner, winner_evaluation, tuple(evaluations),
                           request_cap, len(evaluations), disclosure, applied,
                           fallback_reason)


__all__ = [
    "BestOfNError", "CandidateEvaluation", "CandidatePatch", "CandidateRequest",
    "CheckResult", "DEFAULT_ALLOWANCE", "DEFAULT_MAX_CANDIDATES", "MODE_ENV",
    "SCHEMA", "SelectionResult", "apply_candidate", "enabled", "mode",
    "select_best_candidate", "ticketed_repro",
]
