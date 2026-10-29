"""Experimental Fork-A checkpoint workers and an equal-budget comparison tool.

This module is intentionally separate from ``kernel_stateframe_portfolio``.  The
portfolio predates durable checkpoint lineage and its branches are fresh
requests.  Fork-A keeps the smallest useful experiment: one immutable parent
checkpoint, exactly two writable child workspaces, one journal per child, a
shared host budget, and one parent-owned admission operation.

The runtime only constructs :class:`ForkAController` when
``STATEFRAME_FORK_A=on``.  The comparison harness is offline tooling and does
not alter the runtime or choose a default arm.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import os
import shutil
import stat
import time
import tracemalloc
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from kernel_stateframe_portfolio import (
    PortfolioJournal,
    _readonly_tree,
    apply_model_tool_call,
    apply_scripted_edits,
    digest,
    productive_refusal,
)


MODE_ENV = "STATEFRAME_FORK_A"
SCHEMA = "stateframe-fork-a/1"
COMPARISON_SCHEMA = "stateframe-fork-a-comparison/1"
VALID_MODES = frozenset({"off", "on"})
ARM_IDS = ("single_primary", "fresh_subagents", "checkpoint_branches")
WORKER_COUNT = 2
DEFAULT_BUDGET = 2


class ForkAError(RuntimeError):
    """A Fork-A host invariant rejected a transition."""


def mode() -> str:
    value = str(os.environ.get(MODE_ENV, "off")).strip().lower()
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _json_value(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _call(callback: Callable, *available: object) -> object:
    """Call the small callback shapes used by offline and live probes."""
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        parameters = {}
    positional = [parameter for parameter in parameters.values()
                  if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL
           for parameter in parameters.values()):
        return callback(*available)
    return callback(*available[:len(positional)])


@dataclass(slots=True)
class SharedBudget:
    """A host-owned unit budget shared by both checkpoint workers."""

    total: int
    used: int = 0
    events: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.total) is not int or self.total < 1:
            raise ValueError("Fork-A budget must be a positive integer")

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.used)

    def charge(self, *, reason: str = "") -> bool:
        allowed = self.used < self.total
        event = {"reason": str(reason), "used_before": self.used, "allowed": allowed}
        if allowed:
            self.used += 1
            event["used_after"] = self.used
        else:
            event["outcome"] = "needs_review"
        self.events.append(event)
        return allowed

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-fork-a-budget/1",
            "total": self.total,
            "used": self.used,
            "remaining": self.remaining,
            "events": copy.deepcopy(self.events),
        }


@dataclass(frozen=True, slots=True)
class ForkACheckpoint:
    checkpoint_id: str
    workspace: Path
    parent_revision: str
    parent_journal_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-fork-a-checkpoint/1",
            "checkpoint_id": self.checkpoint_id,
            "workspace": str(self.workspace),
            "parent_revision": self.parent_revision,
            "parent_journal_id": self.parent_journal_id,
        }


@dataclass(slots=True)
class ForkAWorker:
    worker_id: str
    worker_index: int
    workspace: Path
    journal_id: str
    journal_path: Path
    request_id: str
    parent_revision: str
    checkpoint_id: str
    request: dict[str, object] = field(default_factory=dict)
    outcome: dict[str, object] = field(default_factory=dict)
    status: str = "created"
    workspace_hash: str | None = None
    candidate_id: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-fork-a-worker/1",
            "worker_id": self.worker_id,
            "worker_index": self.worker_index,
            "journal_id": self.journal_id,
            "request_id": self.request_id,
            "parent_revision": self.parent_revision,
            "checkpoint_id": self.checkpoint_id,
            "status": self.status,
            "workspace_hash": self.workspace_hash,
            "candidate_id": self.candidate_id,
            "error_type": self.error_type,
        }


@dataclass(frozen=True, slots=True)
class ForkAAdmission:
    accepted: bool
    authority: str
    worker_id: str | None = None
    candidate_id: str | None = None
    workspace: Path | None = None
    reason: str = ""
    parent_revision: str | None = None
    revalidation: object = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-fork-a-admission/1",
            "accepted": self.accepted,
            "authority": self.authority,
            "worker_id": self.worker_id,
            "candidate_id": self.candidate_id,
            "workspace": str(self.workspace) if self.workspace is not None else None,
            "reason": self.reason,
            "parent_revision": self.parent_revision,
            "revalidation": _json_value(self.revalidation),
        }


def _journal_sequence(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except FileNotFoundError:
        return 0


def _open_journal(path: Path, journal_id: str) -> PortfolioJournal:
    journal = PortfolioJournal(path, journal_id=journal_id)
    journal.sequence = _journal_sequence(path)
    return journal


_IGNORED_PARTS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".test-tmp", "portfolio",
    "continuation-workspaces", "measurement-runs",
})


def _nested_exclusions(source: Path, exclusions: Sequence[Path]) -> tuple[Path, ...]:
    source = source.resolve()
    return tuple(path for path in exclusions
                 if path != source and path.is_relative_to(source))


def _workspace_revision(root: Path, exclusions: Sequence[Path] = ()) -> str:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ForkAError(f"workspace must be a directory: {root}")
    nested = _nested_exclusions(root, exclusions)
    files: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        resolved = path.resolve() if path.is_symlink() else path
        if any(resolved == excluded or resolved.is_relative_to(excluded)
               for excluded in nested):
            continue
        if path.is_symlink():
            raise ForkAError(f"workspace contains a symlink: {relative.as_posix()}")
        if path.is_file():
            data = path.read_bytes()
            files[relative.as_posix()] = {
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }
    return digest({"files": files})


def _copy_workspace(source: Path, destination: Path,
                    exclusions: Sequence[Path] = ()) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination.exists() or destination.is_symlink():
        raise ForkAError(f"workspace destination already exists: {destination}")
    nested = _nested_exclusions(source, exclusions)
    destination.mkdir(parents=True, exist_ok=False)

    def copy_dir(current_source: Path, current_destination: Path) -> None:
        for entry in sorted(current_source.iterdir(), key=lambda item: item.name):
            if entry.name in _IGNORED_PARTS:
                continue
            resolved = entry.resolve() if entry.is_symlink() else entry
            if any(resolved == excluded or resolved.is_relative_to(excluded)
                   for excluded in nested):
                continue
            target = current_destination / entry.name
            if entry.is_symlink():
                raise ForkAError(f"workspace contains a symlink: {entry}")
            if entry.is_dir():
                target.mkdir()
                copy_dir(entry, target)
            elif entry.is_file():
                shutil.copy2(entry, target)

    copy_dir(source, destination)


def _writable_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        elif path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


class ForkAController:
    """Run two checkpoint-derived workers and admit through the parent only."""

    def __init__(self, parent_workspace: str | Path, *, store_root: str | Path,
                 budget: int = DEFAULT_BUDGET, ticket_id: str = "",
                 parent_ledger: object | None = None,
                 excluded_roots: Sequence[str | Path] = ()):
        self.parent_workspace = Path(parent_workspace).resolve()
        if not self.parent_workspace.is_dir():
            raise ForkAError(f"parent workspace must be a directory: {self.parent_workspace}")
        self.store_root = Path(store_root).resolve()
        self.store_root.mkdir(parents=True, exist_ok=True)
        self.excluded_roots = tuple({
            self.store_root,
            *(Path(path).resolve() for path in excluded_roots),
        })
        self.ticket_id = str(ticket_id or "")
        self.parent_ledger = parent_ledger
        self.budget = SharedBudget(budget)
        self.parent_journal = PortfolioJournal(
            self.store_root / "parent-events.jsonl",
            journal_id="fork-a-parent:" + uuid.uuid4().hex,
        )
        parent_revision = _workspace_revision(self.parent_workspace, self.excluded_roots)
        checkpoint_workspace = self.store_root / "checkpoint" / "workspace"
        _copy_workspace(self.parent_workspace, checkpoint_workspace, self.excluded_roots)
        if _workspace_revision(checkpoint_workspace) != parent_revision:
            raise ForkAError("parent checkpoint did not preserve the source revision")
        _readonly_tree(checkpoint_workspace)
        checkpoint_id = "checkpoint:" + digest({
            "parent_revision": parent_revision,
            "ticket_id": self.ticket_id,
        })[len("sha256:"):len("sha256:") + 24]
        self.checkpoint = ForkACheckpoint(
            checkpoint_id=checkpoint_id,
            workspace=checkpoint_workspace,
            parent_revision=parent_revision,
            parent_journal_id=self.parent_journal.journal_id,
        )
        self.workers: list[ForkAWorker] = []
        self._workers_by_id: dict[str, ForkAWorker] = {}
        self.admission: ForkAAdmission | None = None
        self.status = "ready"
        self.parent_journal.emit(
            "fork_a_started",
            ticket_id=self.ticket_id,
            checkpoint=self.checkpoint.to_dict(),
            worker_count=WORKER_COUNT,
            budget=self.budget.to_dict(),
            parent_only_admission=True,
        )

    def _charge(self, reason: str) -> bool:
        # Do not spend parent allocation when this controller has no local
        # capacity. A parent meter may return False (rather than raise) for a
        # dry-run or test meter, so an explicit false is also a denial.
        if self.budget.remaining <= 0:
            allowed = self.budget.charge(reason=reason)
            if not allowed:
                self.parent_journal.emit(
                    "fork_a_budget_denied", reason=reason, outcome="needs_review",
                )
            return allowed
        if self.parent_ledger is not None and hasattr(self.parent_ledger, "charge"):
            try:
                parent_allowed = self.parent_ledger.charge(
                    "model_calls", reason=f"fork_a:{reason}")
            except Exception as error:  # parent budget is authoritative
                self.parent_journal.emit(
                    "fork_a_budget_denied", reason=reason,
                    error_type=type(error).__name__, outcome="needs_review",
                )
                return False
            if parent_allowed is False:
                self.parent_journal.emit(
                    "fork_a_budget_denied", reason=reason, outcome="needs_review",
                )
                return False
        allowed = self.budget.charge(reason=reason)
        if not allowed:
            self.parent_journal.emit(
                "fork_a_budget_denied", reason=reason, outcome="needs_review",
            )
        return allowed

    def _new_worker(self, index: int) -> ForkAWorker:
        worker_id = f"worker-{index:02d}-{uuid.uuid4().hex[:10]}"
        worker_dir = self.store_root / "workers" / worker_id
        workspace = worker_dir / "workspace"
        _copy_workspace(self.checkpoint.workspace, workspace)
        _writable_tree(workspace)
        journal_id = "fork-a-worker:" + uuid.uuid4().hex
        journal_path = worker_dir / "events.jsonl"
        request_id = "request:" + uuid.uuid4().hex
        worker = ForkAWorker(
            worker_id=worker_id,
            worker_index=index,
            workspace=workspace,
            journal_id=journal_id,
            journal_path=journal_path,
            request_id=request_id,
            parent_revision=self.checkpoint.parent_revision,
            checkpoint_id=self.checkpoint.checkpoint_id,
        )
        journal = PortfolioJournal(journal_path, journal_id=journal_id)
        journal.emit(
            "worker_started",
            worker_id=worker_id,
            worker_index=index,
            checkpoint_id=self.checkpoint.checkpoint_id,
            parent_revision=self.checkpoint.parent_revision,
            request_id=request_id,
        )
        self.parent_journal.emit(
            "worker_spawned",
            worker_id=worker_id,
            worker_index=index,
            checkpoint_id=self.checkpoint.checkpoint_id,
            parent_revision=self.checkpoint.parent_revision,
            journal_id=journal_id,
        )
        self.workers.append(worker)
        self._workers_by_id[worker_id] = worker
        return worker

    @staticmethod
    def _request_for(worker: ForkAWorker, request: Mapping[str, object] | None) -> dict[str, object]:
        base = copy.deepcopy(dict(request or {}))
        return {
            **base,
            "fresh": False,
            "checkpoint_id": worker.checkpoint_id,
            "parent_revision": worker.parent_revision,
            "worker_id": worker.worker_id,
            "journal_id": worker.journal_id,
        }

    def _execute(self, worker: ForkAWorker, *, request: Mapping[str, object] | None,
                 model_request: Callable | None, resumed: bool = False) -> ForkAWorker:
        worker.request = self._request_for(worker, request if request is not None else worker.request)
        journal = _open_journal(worker.journal_path, worker.journal_id)
        if resumed:
            journal.emit("worker_resumed", worker_id=worker.worker_id,
                         request_id=worker.request_id, checkpoint_id=worker.checkpoint_id)
        try:
            if model_request is None:
                outcome: object = {"status": "no_model_request", "needs_review": True}
            else:
                outcome = _call(model_request, worker, worker.request)
            if not isinstance(outcome, Mapping):
                outcome = {"response": outcome}
            outcome = dict(_json_value(outcome))
            edits = outcome.get("edits")
            if isinstance(edits, Sequence) and not isinstance(edits, (str, bytes, bytearray)):
                outcome["applied_edits"] = list(apply_scripted_edits(
                    worker.workspace, edits, journal=journal,
                ))
            calls = outcome.get("tool_calls")
            if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):
                for call in calls[:1]:
                    if isinstance(call, Mapping):
                        outcome["applied_tool_edit"] = list(
                            apply_model_tool_call(worker.workspace, call, journal=journal))
            worker.outcome = outcome
            worker.workspace_hash = _workspace_revision(worker.workspace)
            worker.candidate_id = "fork-a-candidate:" + digest({
                "worker_id": worker.worker_id,
                "workspace_hash": worker.workspace_hash,
                "parent_revision": worker.parent_revision,
            })[len("sha256:"):len("sha256:") + 24]
            worker.status = "completed"
            worker.error_type = None
            journal.emit(
                "worker_response",
                worker_id=worker.worker_id,
                request_id=worker.request_id,
                response=outcome,
                workspace_hash=worker.workspace_hash,
                candidate_id=worker.candidate_id,
                resumed=resumed,
            )
            journal.emit(
                "candidate_proposed",
                worker_id=worker.worker_id,
                candidate_id=worker.candidate_id,
                workspace_hash=worker.workspace_hash,
                parent_revision=worker.parent_revision,
            )
            journal.emit("worker_finished", worker_id=worker.worker_id, status=worker.status)
            self.parent_journal.emit(
                "worker_finished", worker_id=worker.worker_id,
                status=worker.status, candidate_id=worker.candidate_id,
                workspace_hash=worker.workspace_hash,
            )
        except Exception as error:  # a crashed child is durable and resumable
            worker.status = "crashed"
            worker.error_type = type(error).__name__
            worker.outcome = {"status": "crashed", "error": str(error)[:240]}
            journal.emit(
                "worker_crashed", worker_id=worker.worker_id,
                error_type=worker.error_type, detail=str(error)[:240],
            )
            self.parent_journal.emit(
                "worker_crashed", worker_id=worker.worker_id,
                error_type=worker.error_type,
            )
        finally:
            journal.path.touch(exist_ok=True)
        return worker

    def run(self, *, refusal: object | None = None,
            request: Mapping[str, object] | None = None,
            model_request: Callable | None = None) -> tuple[ForkAWorker, ...]:
        """Spawn exactly two workers from the immutable checkpoint once."""
        if self.workers:
            self.parent_journal.emit("fork_a_run_reused", workers=len(self.workers))
            return tuple(self.workers)
        if refusal is not None and not productive_refusal(
                verified_progress=bool(_field(refusal, "verified_progress", False)),
                host_payload=_field(refusal, "host_payload", refusal),
                next_repair_generated=bool(_field(refusal, "next_repair_generated", False)),
                reproduced_localized_violation=_field(
                    refusal, "reproduced_localized_violation", None)):
            self.status = "needs_review"
            self.parent_journal.emit("fork_a_not_triggered", reason="not_productive_refusal")
            return ()
        self.status = "running"
        for index in range(WORKER_COUNT):
            if not self._charge(f"worker:{index}"):
                self.status = "needs_review"
                break
            worker = self._new_worker(index)
            self._execute(worker, request=request, model_request=model_request)
        if len(self.workers) == WORKER_COUNT and all(
                worker.status == "completed" for worker in self.workers):
            self.status = "ready_for_admission"
        self.parent_journal.emit(
            "fork_a_finished", status=self.status, workers=len(self.workers),
            budget=self.budget.to_dict(),
        )
        return tuple(self.workers)

    def resume_worker(self, worker_id: str, *, model_request: Callable,
                      request: Mapping[str, object] | None = None) -> ForkAWorker:
        """Replay one crashed child in its own durable workspace and journal."""
        worker = self._workers_by_id.get(str(worker_id))
        if worker is None:
            raise ForkAError(f"unknown Fork-A worker: {worker_id}")
        if worker.status != "crashed":
            raise ForkAError("only a crashed worker can be resumed")
        if not self._charge(f"resume:{worker.worker_id}"):
            raise ForkAError("shared Fork-A budget exhausted")
        resumed = self._execute(
            worker, request=request, model_request=model_request, resumed=True)
        if len(self.workers) == WORKER_COUNT and all(
                item.status == "completed" for item in self.workers):
            self.status = "ready_for_admission"
            self.parent_journal.emit(
                "fork_a_resumed", status=self.status,
                workers=len(self.workers), budget=self.budget.to_dict(),
            )
        return resumed

    @staticmethod
    def _revalidate(revalidate: Callable, worker: ForkAWorker, workspace: Path) -> object:
        return _call(revalidate, worker, workspace)

    def admit(self, worker_id: str, *, revalidate: Callable | None) -> ForkAAdmission:
        """Parent-only admission of one current, revalidated child snapshot."""
        if self.admission is not None:
            return ForkAAdmission(False, "parent", reason="admission_already_committed")
        worker = self._workers_by_id.get(str(worker_id))
        if worker is None:
            return ForkAAdmission(False, "parent", worker_id=str(worker_id),
                                  reason="unknown_worker")
        if worker.status != "completed" or not worker.workspace_hash:
            return ForkAAdmission(False, "parent", worker_id=worker.worker_id,
                                  reason="worker_not_ready")
        current_parent_revision = _workspace_revision(self.parent_workspace, self.excluded_roots)
        if current_parent_revision != self.checkpoint.parent_revision:
            self.parent_journal.emit(
                "candidate_rejected", worker_id=worker.worker_id,
                reason="stale_parent_revision", current_parent_revision=current_parent_revision,
            )
            return ForkAAdmission(False, "parent", worker_id=worker.worker_id,
                                  reason="stale_parent_revision",
                                  parent_revision=current_parent_revision)
        if revalidate is None:
            return ForkAAdmission(False, "parent", worker_id=worker.worker_id,
                                  reason="revalidation_required",
                                  parent_revision=current_parent_revision)
        recheck_workspace = (
            self.store_root / "rechecks" / worker.worker_id
            / f"attempt-{uuid.uuid4().hex}" / "workspace"
        )
        _copy_workspace(worker.workspace, recheck_workspace)
        if _workspace_revision(recheck_workspace) != worker.workspace_hash:
            return ForkAAdmission(False, "parent", worker_id=worker.worker_id,
                                  reason="worker_workspace_changed",
                                  parent_revision=current_parent_revision)
        try:
            revalidation = self._revalidate(revalidate, worker, recheck_workspace)
        except Exception as error:
            self.parent_journal.emit(
                "candidate_rejected", worker_id=worker.worker_id,
                reason="revalidation_crashed", error_type=type(error).__name__,
            )
            return ForkAAdmission(False, "parent", worker_id=worker.worker_id,
                                  reason="revalidation_crashed",
                                  parent_revision=current_parent_revision)
        revalidated_hash = _workspace_revision(recheck_workspace)
        if revalidated_hash != worker.workspace_hash:
            self.parent_journal.emit(
                "candidate_rejected", worker_id=worker.worker_id,
                reason="revalidation_workspace_changed",
                expected_workspace_hash=worker.workspace_hash,
                actual_workspace_hash=revalidated_hash,
            )
            return ForkAAdmission(False, "parent", worker_id=worker.worker_id,
                                  reason="revalidation_workspace_changed",
                                  parent_revision=current_parent_revision,
                                  revalidation=revalidation)
        passed = bool(revalidation if isinstance(revalidation, bool)
                      else _field(revalidation, "passed",
                                  _field(revalidation, "accepted", False)))
        claimed_parent_revision = _field(revalidation, "parent_revision", None)
        if claimed_parent_revision is not None and claimed_parent_revision != current_parent_revision:
            passed = False
            reason = "revalidation_parent_revision_mismatch"
        else:
            reason = "revalidation_failed"
        if (_workspace_revision(self.parent_workspace, self.excluded_roots)
                != self.checkpoint.parent_revision):
            reason = "stale_parent_revision"
            passed = False
        if not passed:
            self.parent_journal.emit(
                "candidate_rejected", worker_id=worker.worker_id, reason=reason,
                revalidation=_json_value(revalidation),
            )
            return ForkAAdmission(False, "parent", worker_id=worker.worker_id,
                                  reason=reason, parent_revision=current_parent_revision,
                                  revalidation=revalidation)
        candidate_id = worker.candidate_id or (
            "fork-a-candidate:" + digest({"worker_id": worker.worker_id,
                                           "workspace_hash": worker.workspace_hash})
            [len("sha256:"):len("sha256:") + 24])
        admitted_workspace = self.store_root / "admitted" / candidate_id / "workspace"
        _copy_workspace(recheck_workspace, admitted_workspace)
        if _workspace_revision(admitted_workspace) != worker.workspace_hash:
            raise ForkAError("admitted candidate failed integrity check")
        _readonly_tree(admitted_workspace)
        result = ForkAAdmission(
            True, "parent", worker_id=worker.worker_id, candidate_id=candidate_id,
            workspace=admitted_workspace, reason="revalidated_current_candidate",
            parent_revision=current_parent_revision, revalidation=revalidation,
        )
        self.admission = result
        self.status = "admitted"
        self.parent_journal.emit(
            "candidate_admitted", worker_id=worker.worker_id,
            candidate_id=candidate_id, workspace_hash=worker.workspace_hash,
            parent_revision=current_parent_revision,
            revalidation=_json_value(revalidation), authority="parent",
        )
        return result

    def frame_projection(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "mode": "on",
            "status": self.status,
            "worker_count": WORKER_COUNT,
            "parent_only_admission": True,
            "merge_policy": "one_revalidated_candidate_no_arbitrary_merge",
            "checkpoint": {
                "checkpoint_id": self.checkpoint.checkpoint_id,
                "parent_revision": self.checkpoint.parent_revision,
            },
            "workers": [worker.to_dict() for worker in self.workers],
            "admission": self.admission.to_dict() if self.admission is not None else None,
            "budget": self.budget.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ArmMeasurement:
    arm_id: str
    budget: int
    cost: float
    correct: bool
    wall_ms: float
    memory_bytes: int
    crashed: bool = False
    recovered: bool = False
    recovery_status: str = "not_needed"
    budget_compliant: bool = True
    error_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def cost_correct(self) -> dict[str, object]:
        return {"cost": self.cost, "correct": self.correct}

    @property
    def peak_memory_bytes(self) -> int:
        return self.memory_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-fork-a-arm/1",
            "arm_id": self.arm_id,
            "budget": self.budget,
            "cost": self.cost,
            "correct": self.correct,
            "cost_correct": self.cost_correct,
            "wall_ms": self.wall_ms,
            "memory_bytes": self.memory_bytes,
            "peak_memory_bytes": self.memory_bytes,
            "crashed": self.crashed,
            "recovered": self.recovered,
            "recovery_status": self.recovery_status,
            "budget_compliant": self.budget_compliant,
            "error_type": self.error_type,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ForkAComparisonReport:
    budget: int
    arms: dict[str, ArmMeasurement]
    equal_budget: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": COMPARISON_SCHEMA,
            "budget": self.budget,
            "equal_budget": self.equal_budget,
            "arms": {key: value.to_dict() for key, value in self.arms.items()},
        }


def _measurement_values(value: object) -> tuple[float, bool, bool, dict[str, object]]:
    if isinstance(value, Mapping):
        raw_cost = value.get("cost", value.get("tokens", value.get("model_calls", 0)))
        try:
            cost = float(raw_cost)
        except (TypeError, ValueError):
            cost = 0.0
        correct = bool(value.get("correct", value.get("passed", value.get("success", False))))
        recovered = bool(value.get("recovered", False))
        return cost, correct, recovered, dict(value)
    return 0.0, bool(value), False, {"result": _json_value(value)}


class ForkAComparisonHarness:
    """Measure the three Fork-A acceptance arms under one explicit budget."""

    def __init__(self, *, budget: int):
        if type(budget) is not int or budget < 1:
            raise ValueError("comparison budget must be a positive integer")
        self.budget = budget

    def run(self, arms: Mapping[str, Callable], *,
            recovery: Mapping[str, Callable] | None = None) -> ForkAComparisonReport:
        if set(arms) != set(ARM_IDS):
            missing = [arm for arm in ARM_IDS if arm not in arms]
            extra = [arm for arm in arms if arm not in ARM_IDS]
            raise ValueError(f"comparison arms must be {ARM_IDS}; missing={missing}, extra={extra}")
        recovery = recovery or {}
        measurements: dict[str, ArmMeasurement] = {}
        for arm_id in ARM_IDS:
            callback = arms[arm_id]
            started = time.perf_counter_ns()
            crashed = False
            recovered = False
            recovery_status = "not_needed"
            error_type = None
            result: object = None
            metadata: dict[str, object] = {}
            attempt_cost = 0.0
            attempt_metadata: dict[str, object] = {}
            peak = 0
            was_tracing = tracemalloc.is_tracing()
            if not was_tracing:
                tracemalloc.start()
            tracemalloc.reset_peak()
            try:
                result = _call(callback, self.budget, arm_id)
                attempt_cost, _attempt_correct, _attempt_recovered, attempt_metadata = (
                    _measurement_values(result))
                if isinstance(result, Mapping) and result.get("crashed") is True:
                    raise RuntimeError(str(result.get("error", "arm reported a crash")))
            except Exception as error:  # comparison records failures rather than hiding them
                crashed = True
                error_type = type(error).__name__
                recovery_callback = recovery.get(arm_id)
                if recovery_callback is None:
                    recovery_status = "unavailable"
                    result = {"cost": 0, "correct": False}
                else:
                    recovery_status = "attempted"
                    try:
                        result = _call(recovery_callback, self.budget, error)
                        recovered = bool(_field(result, "recovered", True))
                        recovery_status = "recovered" if recovered else "failed"
                    except Exception as recovery_error:
                        error_type = type(recovery_error).__name__
                        result = {"cost": 0, "correct": False}
                        recovery_status = "failed"
            finally:
                _, peak = tracemalloc.get_traced_memory()
                if not was_tracing:
                    tracemalloc.stop()
            wall_ms = (time.perf_counter_ns() - started) / 1_000_000
            cost, correct, result_recovered, metadata = _measurement_values(result)
            if crashed:
                cost += attempt_cost
                metadata = {"attempt": attempt_metadata, "recovery": metadata}
                recovered = recovered or result_recovered
            else:
                recovered = True
            measurements[arm_id] = ArmMeasurement(
                arm_id=arm_id, budget=self.budget, cost=cost, correct=correct,
                wall_ms=wall_ms, memory_bytes=max(0, int(peak)), crashed=crashed,
                recovered=recovered, recovery_status=recovery_status,
                budget_compliant=0 <= cost <= self.budget,
                error_type=error_type, metadata=metadata,
            )
        equal_budget = all(measurement.budget == self.budget
                           and measurement.budget_compliant
                           for measurement in measurements.values())
        return ForkAComparisonReport(self.budget, measurements, equal_budget)


def compare_fork_a(arms: Mapping[str, Callable], *, budget: int,
                   recovery: Mapping[str, Callable] | None = None) -> dict[str, object]:
    """Convenience wrapper for scripts that want JSON-ready tooling output."""
    return ForkAComparisonHarness(budget=budget).run(arms, recovery=recovery).to_dict()


__all__ = [
    "ARM_IDS", "ArmMeasurement", "COMPARISON_SCHEMA", "DEFAULT_BUDGET",
    "ForkAAdmission", "ForkAComparisonHarness", "ForkAComparisonReport",
    "ForkACheckpoint", "ForkAController", "ForkAError", "ForkAWorker",
    "MODE_ENV", "SCHEMA", "SharedBudget", "compare_fork_a", "enabled", "mode",
]
