"""Opt-in candidate portfolio candidate portfolio and fork controller.

The portfolio is deliberately a small host-owned layer. A model may produce a
candidate or a repair request, but it cannot mark a candidate eligible, select
one, or spend the closure reserve. Those decisions are made from immutable
workspace copies and typed host evidence.

The module is useful without a live provider: ``simulate_portfolio`` supplies a
deterministic scripted model used by the offline proof. Runtime integrations
should construct a ``PortfolioManager`` only when ``STATEFRAME_PORTFOLIO=on``;
the switch is strict and defaults to the historical no-op path.
"""
from __future__ import annotations

import copy
import dataclasses
import difflib
import hashlib
import inspect
import json
import math
import os
import shutil
import stat
import time
import uuid

from kernel_measurement import externalize_journal_fields
import kernel_workspace_truth as _workspace_truth
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


MODE_ENV = "STATEFRAME_PORTFOLIO"
SCHEMA = "stateframe-portfolio/1"
VALID_MODES = frozenset({"off", "on"})
DEFAULT_ROOTS = 2
DEFAULT_FORKS = 2
DEFAULT_BUDGET = 8
DEFAULT_CLOSURE_RESERVE = 2

_IGNORED_PARTS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".test-tmp", "portfolio",
    "continuation-workspaces", "measurement-runs",
})


class PortfolioError(RuntimeError):
    """A portfolio invariant rejected a candidate or transition."""


def mode() -> str:
    value = str(os.environ.get(MODE_ENV, "off")).lower()
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _json_value(value: object) -> object:
    """Convert the small set of evidence objects used by host closure to JSON."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _relative_files(root: Path) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise PortfolioError(f"workspace must be a directory: {root}")
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in _IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise PortfolioError(f"workspace contains a symlink: {relative.as_posix()}")
        if path.is_file():
            result.append(path)
    return result


def _inventory(root: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in _relative_files(root):
        data = path.read_bytes()
        rows[path.relative_to(Path(root).resolve()).as_posix()] = {
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
    return rows


def workspace_tree_hash(root: str | Path) -> str:
    """Hash the visible workspace tree, excluding cache and portfolio noise."""
    return digest({"files": _inventory(Path(root))})


def _copy_workspace(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    if destination.exists() or destination.is_symlink():
        raise PortfolioError(f"workspace destination already exists: {destination}")
    # Walk first so a symlink cannot be silently followed by copytree.
    _relative_files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=False,
                    ignore=shutil.ignore_patterns(*_IGNORED_PARTS))


def _readonly_tree(root: Path) -> None:
    """Make a retained snapshot read-only after its hash has been checked."""
    paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    paths.append(root)
    for path in paths:
        try:
            if path.is_dir() and not path.is_symlink():
                path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
            elif path.is_file() and not path.is_symlink():
                path.chmod(stat.S_IRUSR | stat.S_IRGRP)
        except OSError as error:
            raise PortfolioError(f"cannot seal candidate snapshot {path}") from error


def _diff_size(before: Path | None, after: Path) -> int:
    """Return a deterministic changed-line measure used only for tie-breaking."""
    if before is None or not Path(before).is_dir():
        return sum(len(path.read_bytes()) for path in _relative_files(after))
    old = _inventory(Path(before))
    new = _inventory(after)
    total = 0
    for relative in sorted(set(old) | set(new)):
        if relative not in old:
            total += max(1, new[relative]["size"] // 80)
            continue
        if relative not in new:
            total += max(1, old[relative]["size"] // 80)
            continue
        if old[relative]["sha256"] == new[relative]["sha256"]:
            continue
        old_text = (Path(before) / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        new_text = (Path(after) / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        changed = 0
        for tag, left_a, left_b, right_a, right_b in difflib.SequenceMatcher(
                a=old_text, b=new_text, autojunk=False).get_opcodes():
            if tag != "equal":
                changed += max(left_b - left_a, right_b - right_a, 1)
        total += changed or 1
    return total


def _field(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _status(value: object) -> str:
    raw = _field(value, "display_status", "result_class", "status", "result",
                 "verdict", default="")
    return str(getattr(raw, "value", raw) or "").upper()


def _assessment_receipts(assessment: object) -> tuple[object, ...]:
    values = _field(assessment, "receipts", default=())
    if isinstance(values, Mapping):
        return tuple(values.values())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return tuple(values)
    return ()


def _unique_receipts(values: Iterable[object]) -> tuple[object, ...]:
    unique: list[object] = []
    seen: set[str] = set()
    for receipt in values:
        reference = _field(receipt, "receipt_id", "evidence_ref", default=None)
        key = (f"ref:{reference}" if reference is not None
               else digest(_json_value(receipt)))
        if key not in seen:
            seen.add(key)
            unique.append(receipt)
    return tuple(unique)


@dataclass(frozen=True, slots=True)
class Eligibility:
    eligible: bool
    reasons: tuple[str, ...] = ()
    host_checks_current: bool = False
    host_checks_passing: bool = False
    public_suite_green: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "host_checks_current": self.host_checks_current,
            "host_checks_passing": self.host_checks_passing,
            "public_suite_green": self.public_suite_green,
        }


def candidate_eligibility(*, host_assessments: Sequence[object] = (),
                          host_receipts: Sequence[object] = (),
                          current_snapshot: object = None,
                          public_suite: Sequence[object] = (),
                          public_green: bool | None = None) -> Eligibility:
    """Reduce host evidence to the hard candidate-admission predicate.

    Host checks are conjunctive.  A stale receipt, counterexample, UNKNOWN
    assessment, or non-green public result rejects the candidate; no voting or
    best-of obligations are performed here.
    """
    reasons: list[str] = []
    assessments = tuple(host_assessments or ())
    receipts = list(host_receipts or ())
    for assessment in assessments:
        verdict = _status(assessment)
        if verdict != "SATISFIED":
            reasons.append(f"host obligation not satisfied: {verdict or 'UNKNOWN'}")
        receipts.extend(_assessment_receipts(assessment))
    # Duplicate receipt references are harmless, but keeping one copy makes the
    # candidate identity deterministic when the assessment includes its receipts.
    receipts = list(_unique_receipts(receipts))
    current = True
    passing = True
    for receipt in receipts:
        status = _status(receipt)
        if status != "PASS":
            passing = False
            reasons.append(f"host check is not PASS: {status or 'UNKNOWN'}")
        snapshot = _field(receipt, "snapshot_digest", "snapshot", default=None)
        if current_snapshot is not None and snapshot is not None and snapshot != current_snapshot:
            current = False
            reasons.append("host receipt is stale for the current workspace")
    if not receipts and assessments:
        # An assessment may be a reducer projection without receipt objects.  A
        # SATISFIED projection is still current evidence; an open/unknown one
        # was already rejected above.
        passing = not reasons
    public_rows = tuple(public_suite or ())
    if public_green is None:
        public_ok = bool(public_rows) and all(
            (_status(row) in {"PASS", "GREEN", "0"}
             or _field(row, "exit_code", default=None) == 0)
            and _status(row) not in {"RED", "FAIL", "FAILED", "COUNTEREXAMPLE"}
            for row in public_rows)
    else:
        public_ok = bool(public_green)
    if not public_ok:
        reasons.append("public suite is not green")
    eligible = not reasons and current and passing and public_ok
    return Eligibility(eligible, tuple(dict.fromkeys(reasons)), current, passing, public_ok)


@dataclass(frozen=True, slots=True)
class PortfolioPolicy:
    roots: int = DEFAULT_ROOTS
    forks: int = DEFAULT_FORKS
    budget: int = DEFAULT_BUDGET
    closure_reserve: int = DEFAULT_CLOSURE_RESERVE
    temperatures: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for name in ("roots", "forks", "budget", "closure_reserve"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.roots < 1 or self.forks < 1 or self.budget < 1:
            raise ValueError("roots, forks, and budget must be positive")
        if self.closure_reserve > self.budget:
            raise ValueError("closure_reserve cannot exceed budget")
        values = tuple(float(value) for value in self.temperatures)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("temperatures must be finite nonnegative numbers")
        object.__setattr__(self, "temperatures", values)

    @classmethod
    def from_env(cls) -> "PortfolioPolicy":
        def integer(name: str, default: int) -> int:
            raw = os.environ.get(name, str(default))
            try:
                return int(raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a nonnegative integer") from error

        reserve_raw = os.environ.get("STATEFRAME_PORTFOLIO_CLOSURE_RESERVE",
                                     str(DEFAULT_CLOSURE_RESERVE))
        if str(reserve_raw).lower() == "on":
            reserve = DEFAULT_CLOSURE_RESERVE
        elif str(reserve_raw).lower() == "off":
            reserve = 0
        else:
            reserve = integer("STATEFRAME_PORTFOLIO_CLOSURE_RESERVE", DEFAULT_CLOSURE_RESERVE)
        raw_temperatures = os.environ.get("STATEFRAME_PORTFOLIO_TEMPERATURES", "")
        temperatures = tuple(float(item.strip()) for item in raw_temperatures.split(",")
                             if item.strip())
        return cls(
            roots=integer("STATEFRAME_PORTFOLIO_ROOTS", DEFAULT_ROOTS),
            forks=integer("STATEFRAME_PORTFOLIO_FORKS", DEFAULT_FORKS),
            budget=integer("STATEFRAME_PORTFOLIO_BUDGET", DEFAULT_BUDGET),
            closure_reserve=reserve,
            temperatures=temperatures,
        )

    def temperature_for(self, index: int, default: float | None = None) -> float | None:
        if self.temperatures:
            return self.temperatures[index % len(self.temperatures)]
        return default

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-portfolio-policy/1",
            "roots": self.roots,
            "forks": self.forks,
            "budget": self.budget,
            "closure_reserve": self.closure_reserve,
            "temperatures": list(self.temperatures),
        }


class PortfolioJournal:
    """Small append-only journal used for parent and branch-local records."""

    def __init__(self, path: Path, *, journal_id: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_id = journal_id or uuid.uuid4().hex
        self.sequence = 0
        self._blob_cache: dict[str, int] = {}

    def emit(self, kind: str, **fields: object) -> dict[str, object]:
        self.sequence += 1
        normalized_fields = _json_value(fields)
        record = {
            **normalized_fields, "schema": "stateframe-portfolio-journal/1",
            "kind": str(kind), "journal_id": self.journal_id,
            "sequence": self.sequence, "timestamp_ns": time.time_ns(),
        }
        stored_record = {
            **externalize_journal_fields(normalized_fields, self.path, self._blob_cache),
            "schema": record["schema"], "kind": record["kind"],
            "journal_id": record["journal_id"], "sequence": record["sequence"],
            "timestamp_ns": record["timestamp_ns"],
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(stored_record, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return record


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    workspace: Path
    workspace_hash: str
    receipts: tuple[dict[str, object], ...]
    public_suite: tuple[dict[str, object], ...]
    host_assessments: tuple[dict[str, object], ...]
    diff_size: int
    eligibility: dict[str, object]
    ticket_id: str | None = None
    reason: str = ""

    @property
    def eligible(self) -> bool:
        return bool(self.eligibility.get("eligible"))

    def to_dict(self, *, workspace: str | None = None) -> dict[str, object]:
        return {
            "schema": "stateframe-candidate/1",
            "candidate_id": self.candidate_id,
            "workspace": workspace if workspace is not None else str(self.workspace),
            "workspace_hash": self.workspace_hash,
            "receipts": _json_value(self.receipts),
            "public_suite": _json_value(self.public_suite),
            "host_assessments": _json_value(self.host_assessments),
            "diff_size": self.diff_size,
            "eligibility": _json_value(self.eligibility),
            "ticket_id": self.ticket_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object], *, base: Path | None = None) -> "CandidateRecord":
        workspace = Path(str(value.get("workspace", "")))
        if base is not None and not workspace.is_absolute():
            workspace = base / workspace
        return cls(
            candidate_id=str(value.get("candidate_id", "")), workspace=workspace,
            workspace_hash=str(value.get("workspace_hash", "")),
            receipts=tuple(dict(item) for item in value.get("receipts", ())
                           if isinstance(item, Mapping)),
            public_suite=tuple(dict(item) for item in value.get("public_suite", ())
                               if isinstance(item, Mapping)),
            host_assessments=tuple(dict(item) for item in value.get("host_assessments", ())
                                   if isinstance(item, Mapping)),
            diff_size=int(value.get("diff_size", 0)),
            eligibility=dict(value.get("eligibility", {}))
            if isinstance(value.get("eligibility"), Mapping) else {},
            ticket_id=(str(value["ticket_id"]) if value.get("ticket_id") is not None else None),
            reason=str(value.get("reason", "")),
        )


class CandidateStore:
    """Append-only candidate snapshots and a durable pool projection."""

    def __init__(self, root: str | Path, *, baseline: str | Path | None = None,
                 journal: PortfolioJournal | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.baseline = Path(baseline).resolve() if baseline is not None else None
        self.journal = journal
        self.pool_path = self.root / "pool.json"

    def _record_path(self, candidate_id: str) -> Path:
        return self.root / candidate_id.replace(":", "-") / "record.json"

    def _write_exclusive(self, path: Path, value: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(_json_value(value), sort_keys=True, indent=2,
                             ensure_ascii=False) + "\n"
        try:
            with path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_text(encoding="utf-8") != encoded:
                raise PortfolioError(f"immutable portfolio record changed: {path}")

    def preserve(self, workspace: str | Path, *, host_assessments: Sequence[object] = (),
                 host_receipts: Sequence[object] = (), current_snapshot: object = None,
                 public_suite: Sequence[object] = (), public_green: bool | None = None,
                 ticket_id: str | None = None, reason: str = "") -> CandidateRecord | None:
        workspace = Path(workspace).resolve()
        eligibility = candidate_eligibility(
            host_assessments=host_assessments, host_receipts=host_receipts,
            current_snapshot=current_snapshot, public_suite=public_suite,
            public_green=public_green)
        if not eligibility.eligible:
            if self.journal is not None:
                self.journal.emit("candidate_rejected", eligibility=eligibility.to_dict(),
                                  ticket_id=ticket_id, reason=reason)
            return None
        tree_hash = workspace_tree_hash(workspace)
        receipt_values = list(host_receipts or ())
        for assessment in host_assessments or ():
            receipt_values.extend(_assessment_receipts(assessment))
        receipts = []
        for item in _unique_receipts(receipt_values):
            encoded = _json_value(item)
            if isinstance(encoded, Mapping):
                receipts.append(dict(encoded))
        receipts = tuple(receipts)
        assessments = tuple(_json_value(item) for item in host_assessments
                            if isinstance(_json_value(item), Mapping))
        public = tuple(_json_value(item) for item in public_suite
                       if isinstance(_json_value(item), Mapping))
        candidate_key = {
            "workspace_hash": tree_hash, "receipts": receipts,
            "public_suite": public, "host_assessments": assessments,
            "ticket_id": ticket_id,
        }
        candidate_id = "candidate:" + hashlib.sha256(_canonical(candidate_key)).hexdigest()[:24]
        record_path = self._record_path(candidate_id)
        if record_path.exists():
            existing = CandidateRecord.from_dict(json.loads(record_path.read_text()),
                                                 base=record_path.parent)
            if (existing.workspace_hash != tree_hash
                    or existing.receipts != tuple(dict(item) for item in receipts)
                    or existing.public_suite != tuple(dict(item) for item in public)
                    or existing.host_assessments != tuple(dict(item) for item in assessments)
                    or existing.ticket_id != ticket_id):
                raise PortfolioError("candidate id collision or changed immutable record")
            return existing
        candidate_dir = record_path.parent
        snapshot = candidate_dir / "workspace"
        _copy_workspace(workspace, snapshot)
        copied_hash = workspace_tree_hash(snapshot)
        if copied_hash != tree_hash:
            raise PortfolioError("candidate snapshot changed while copying")
        diff_size = _diff_size(self.baseline, snapshot)
        record = CandidateRecord(
            candidate_id=candidate_id, workspace=snapshot, workspace_hash=tree_hash,
            receipts=tuple(dict(item) for item in receipts),
            public_suite=tuple(dict(item) for item in public),
            host_assessments=tuple(dict(item) for item in assessments),
            diff_size=diff_size, eligibility=eligibility.to_dict(),
            ticket_id=ticket_id, reason=reason)
        # Metadata is written before sealing the copied tree.  A failed metadata
        # write leaves a non-eligible temporary tree rather than a misleading
        # candidate record.
        self._write_exclusive(record_path, record.to_dict(workspace="workspace"))
        _readonly_tree(snapshot)
        record_path.chmod(stat.S_IRUSR | stat.S_IRGRP)
        self.commit_pool()
        if self.journal is not None:
            self.journal.emit("candidate_preserved", candidate=record.to_dict(),
                              workspace_hash=tree_hash, receipts=list(receipts),
                              public_suite=list(public), reason=reason)
        return record

    def records(self) -> tuple[CandidateRecord, ...]:
        records: list[CandidateRecord] = []
        for path in sorted(self.root.glob("candidate-*/record.json")):
            try:
                record = CandidateRecord.from_dict(json.loads(path.read_text()),
                                                   base=path.parent)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise PortfolioError(f"invalid candidate record {path}") from error
            if not record.workspace.is_dir() or workspace_tree_hash(record.workspace) != record.workspace_hash:
                raise PortfolioError(f"candidate snapshot failed integrity check: {record.candidate_id}")
            records.append(record)
        return tuple(records)

    def commit_pool(self, candidates: Sequence[CandidateRecord] | None = None) -> Path:
        rows = tuple(candidates) if candidates is not None else self.records()
        payload = {
            "schema": "stateframe-candidate-pool/1",
            "candidates": [row.to_dict(workspace=str(row.workspace)) for row in rows],
            "committed_at": time.time_ns(),
        }
        temporary = self.pool_path.with_name(f".{self.pool_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(_json_value(payload), sort_keys=True, indent=2) + "\n",
                              encoding="utf-8")
        os.replace(temporary, self.pool_path)
        return self.pool_path


def _refusal_field(refusal: object, name: str, default: object = None) -> object:
    return _field(refusal, name, default=default)


def productive_refusal(*, verified_progress: bool, host_payload: object,
                       next_repair_generated: bool = False,
                       reproduced_localized_violation: bool | None = None) -> bool:
    """Return the narrow candidate portfolio fork predicate."""
    if not verified_progress or next_repair_generated:
        return False
    payload = host_payload if isinstance(host_payload, Mapping) else {}
    if str(payload.get("mode", "")).upper() != "REPAIR":
        return False
    if not str(payload.get("focus_obligation", "")).strip():
        return False
    if reproduced_localized_violation is False:
        return False
    localized = (reproduced_localized_violation is True or
                 bool(payload.get("counterexample_ref")) or
                 (payload.get("observed") is not None and payload.get("expected") is not None))
    return localized


@dataclass(slots=True)
class PortfolioLedger:
    policy: PortfolioPolicy
    parent: object | None = None
    used: int = 0
    closure_used: int = 0
    events: list[dict[str, object]] = field(default_factory=list)

    @property
    def available_for_exploration(self) -> int:
        return max(0, self.policy.budget - self.policy.closure_reserve - self.used)

    @property
    def remaining(self) -> int:
        return max(0, self.policy.budget - self.used)

    def charge(self, kind: str, *, reason: str = "") -> bool:
        kind = str(kind)
        exploration = kind not in {"closure", "final_recheck"}
        allowed = (self.used < self.policy.budget - self.policy.closure_reserve
                   if exploration else self.used < self.policy.budget)
        event = {"kind": kind, "reason": reason, "used_before": self.used,
                 "allowed": allowed, "closure_reserve": self.policy.closure_reserve}
        if not allowed:
            event["outcome"] = "needs_review"
            self.events.append(event)
            return False
        if self.parent is not None and hasattr(self.parent, "charge"):
            parent_kind = "challenges" if kind == "final_recheck" else "model_calls"
            try:
                self.parent.charge(parent_kind, reason=f"portfolio:{kind}:{reason}")
            except Exception as error:  # ParentBudgetExhausted is intentionally typed below.
                event.update(outcome="needs_review", parent_error=type(error).__name__)
                self.events.append(event)
                return False
        self.used += 1
        if kind == "closure":
            self.closure_used += 1
        event.update(outcome="charged", used_after=self.used)
        self.events.append(event)
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-portfolio-ledger/1",
            "policy": self.policy.to_dict(), "used": self.used,
            "closure_used": self.closure_used,
            "remaining": self.remaining,
            "available_for_exploration": self.available_for_exploration,
            "events": copy.deepcopy(self.events),
        }


@dataclass(slots=True)
class BranchContext:
    branch_id: str
    workspace: Path
    journal_id: str
    journal_path: Path
    request_id: str
    temperature: float | None
    root_index: int | None = None
    fork_index: int | None = None
    request: dict[str, object] = field(default_factory=dict)
    outcome: dict[str, object] = field(default_factory=dict)


def _call_with_context(callback: Callable, context: BranchContext):
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        parameters = {}
    positional = [item for item in parameters.values()
                  if item.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2:
        return callback(context, context.request)
    return callback(context)


def apply_scripted_edits(workspace: str | Path, edits: Iterable[Mapping[str, object]], *,
                         journal: object = None) -> tuple[str, ...]:
    """Apply exact scripted edits in a branch for deterministic offline runs."""
    root = Path(workspace).resolve()
    from kernel_destructive_edit_guard import guard_write
    applied: list[str] = []
    for edit in edits or ():
        if not isinstance(edit, Mapping):
            raise PortfolioError("scripted edit must be an object")
        raw_path = str(edit.get("path", "")).replace("\\", "/").lstrip("/")
        if not raw_path or raw_path.startswith("../") or "/../" in raw_path:
            raise PortfolioError("scripted edit path escapes workspace")
        path = (root / raw_path).resolve()
        if not path.is_relative_to(root):
            raise PortfolioError("scripted edit path escapes workspace")
        old = edit.get("old")
        new = edit.get("new")
        source = edit.get("source")
        if source is not None:
            if path.exists():
                raise PortfolioError(f"scripted source edit would overwrite {raw_path}")
            refusal = guard_write(
                path, str(source), root=root,
                public_brief=edit.get("public_brief") or edit.get("brief"),
            )
            if refusal is not None:
                raise PortfolioError(refusal)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(source), encoding="utf-8")
        else:
            if not path.is_file() or path.is_symlink():
                if _workspace_truth.enabled():
                    fact = _workspace_truth.exists(root, raw_path, journal=journal)
                    detail = (
                        _workspace_truth.render_fact(fact)
                        if fact.status != "exists"
                        else f"Path {raw_path} exists in the live workspace but is not a file."
                    )
                    raise PortfolioError(detail)
                raise PortfolioError(f"scripted replacement target is missing: {raw_path}")
            current = path.read_text(encoding="utf-8")
            if not isinstance(old, str) or current.count(old) != 1:
                raise PortfolioError(f"scripted replacement is not unique: {raw_path}")
            candidate = current.replace(old, str(new if new is not None else ""), 1)
            refusal = guard_write(
                path, candidate, root=root,
                public_brief=edit.get("public_brief") or edit.get("brief"),
            )
            if refusal is not None:
                raise PortfolioError(refusal)
            path.write_text(candidate, encoding="utf-8")
        applied.append(raw_path)
    return tuple(applied)


def apply_model_tool_call(workspace: str | Path, call: Mapping[str, object], *,
                          journal: object = None) -> tuple[str, ...]:
    """Apply the safe subset of one structured child edit response.

    A live provider may return a normal tool-call envelope rather than the
    deterministic ``edits`` list used by the offline proof.  The branch still
    accepts only exact ``str_replace``/new-file/module replacements; a
    top-level function edit remains pending for the child executor to handle.
    """
    function = call.get("function") if isinstance(call, Mapping) else None
    if not isinstance(function, Mapping):
        return ()
    name = str(function.get("name", ""))
    raw_args = function.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except (TypeError, ValueError):
            return ()
    if not isinstance(raw_args, Mapping):
        return ()
    args = dict(raw_args)
    if name == "str_replace":
        return apply_scripted_edits(workspace, [{
            "path": args.get("path"), "old": args.get("old_str"),
            "new": args.get("new_str", ""),
            "public_brief": args.get("public_brief") or args.get("brief"),
        }], journal=journal)
    if name == "file_create":
        return apply_scripted_edits(workspace, [{
            "path": args.get("path"), "source": args.get("content", ""),
            "public_brief": args.get("public_brief") or args.get("brief"),
        }], journal=journal)
    if name == "replace_module" and isinstance(args.get("source"), str):
        root = Path(workspace).resolve()
        raw_path = str(args.get("path", "")).replace("\\", "/").lstrip("/")
        path = (root / raw_path).resolve()
        if not raw_path or not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            return ()
        source = str(args["source"])
        from kernel_destructive_edit_guard import guard_write
        refusal = guard_write(
            path, source, root=root,
            public_brief=args.get("public_brief") or args.get("brief"),
        )
        if refusal is not None:
            raise PortfolioError(refusal)
        path.write_text(source, encoding="utf-8")
        return (raw_path,)
    return ()


@dataclass(frozen=True, slots=True)
class SelectionResult:
    status: str
    selected_candidate_id: str | None
    pool_commit_path: Path
    selection_path: Path
    hard_eligible_ids: tuple[str, ...] = ()
    discriminating_checks: tuple[dict[str, object], ...] = ()
    tie_break: dict[str, object] = field(default_factory=dict)
    final_recheck: dict[str, object] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-selection/1", "status": self.status,
            "selected_candidate_id": self.selected_candidate_id,
            "pool_commit_path": str(self.pool_commit_path),
            "selection_path": str(self.selection_path),
            "hard_eligible_ids": list(self.hard_eligible_ids),
            "discriminating_checks": _json_value(self.discriminating_checks),
            "tie_break": _json_value(self.tie_break),
            "final_recheck": _json_value(self.final_recheck),
            "reason": self.reason,
        }


class PortfolioManager:
    """Coordinate immutable candidates, productive forks, and selection."""

    def __init__(self, baseline_workspace: str | Path, *, store_root: str | Path,
                 policy: PortfolioPolicy | None = None, ticket_id: str = "",
                 parent_ledger: object | None = None, enabled_mode: bool = True):
        self.baseline_workspace = Path(baseline_workspace).resolve()
        self.store_root = Path(store_root).resolve()
        self.store_root.mkdir(parents=True, exist_ok=True)
        self.policy = policy or PortfolioPolicy.from_env()
        self.ticket_id = str(ticket_id or "")
        self.journal = PortfolioJournal(self.store_root / "portfolio-events.jsonl")
        self.ledger = PortfolioLedger(self.policy, parent=parent_ledger)
        self.store = CandidateStore(self.store_root / "candidates",
                                    baseline=self.baseline_workspace, journal=self.journal)
        self.pool_path = self.store.pool_path
        # A budget refusal is still an auditable portfolio result.  Publish an
        # empty pool projection at construction so needs_review never loses its
        # record merely because no child became eligible.
        self.store.commit_pool(())
        self.status = "ready"
        self.needs_review = False
        self._forked_repair_keys: set[str] = set()
        self._branches: list[BranchContext] = []
        self._selection: SelectionResult | None = None
        self.journal.emit("portfolio_started", policy=self.policy.to_dict(),
                          ticket_id=self.ticket_id, enabled=bool(enabled_mode))

    def _budget_failure(self, reason: str) -> None:
        self.status = "needs_review"
        self.needs_review = True
        self.journal.emit("portfolio_budget_exhausted", reason=reason,
                          ledger=self.ledger.to_dict(), outcome="needs_review")

    def pool(self) -> tuple[CandidateRecord, ...]:
        return self.store.records()

    def preserve(self, workspace: str | Path, **evidence: object) -> CandidateRecord | None:
        return self.store.preserve(workspace, ticket_id=self.ticket_id, **evidence)

    @staticmethod
    def _refusal_key(refusal: object) -> str:
        payload = _refusal_field(refusal, "host_payload", {})
        if not isinstance(payload, Mapping):
            payload = {}
        explicit = _refusal_field(refusal, "next_repair_key", None)
        return str(explicit or digest({
            "focus": payload.get("focus_obligation"),
            "counterexample": payload.get("counterexample_ref"),
            "observed": payload.get("observed"),
            "expected": payload.get("expected"),
        }))

    def _new_branch(self, parent: Path, *, root_index: int | None,
                    fork_index: int | None, temperature: float | None) -> BranchContext:
        prefix = "root" if fork_index is None else "fork"
        branch_id = f"{prefix}-{(root_index if root_index is not None else 0):02d}-"
        branch_id += f"{(fork_index if fork_index is not None else 0):02d}-{uuid.uuid4().hex[:10]}"
        branch_dir = self.store_root / "branches" / branch_id
        workspace = branch_dir / "workspace"
        _copy_workspace(parent, workspace)
        journal = PortfolioJournal(branch_dir / "events.jsonl", journal_id="journal:" + uuid.uuid4().hex)
        request_id = "request:" + uuid.uuid4().hex
        context = BranchContext(
            branch_id=branch_id, workspace=workspace, journal_id=journal.journal_id,
            journal_path=journal.path, request_id=request_id, temperature=temperature,
            root_index=root_index, fork_index=fork_index)
        journal.emit("branch_started", branch_id=branch_id, workspace=str(workspace),
                     request_id=request_id, temperature=temperature,
                     root_index=root_index, fork_index=fork_index)
        self._branches.append(context)
        return context

    def _run_branch(self, context: BranchContext, *, request: Mapping[str, object] | None,
                    model_request: Callable | None) -> BranchContext:
        base_request = copy.deepcopy(dict(request or {}))
        context.request = {
            **base_request, "fresh": True, "request_id": context.request_id,
            "branch_id": context.branch_id, "journal_id": context.journal_id,
            "temperature": context.temperature,
        }
        journal = PortfolioJournal(context.journal_path, journal_id=context.journal_id)
        journal.sequence = 1  # branch_started is the first persisted record.
        journal.emit("model_request", request=context.request, fresh=True)
        if model_request is None:
            outcome: object = {"status": "no_model_request", "needs_review": True}
        else:
            outcome = _call_with_context(model_request, context)
        if not isinstance(outcome, Mapping):
            outcome = {"response": outcome}
        outcome = dict(_json_value(outcome))
        edits = outcome.get("edits")
        if isinstance(edits, Sequence) and not isinstance(edits, (str, bytes, bytearray)):
            outcome["applied_edits"] = list(apply_scripted_edits(
                context.workspace, edits, journal=journal,
            ))
        calls = outcome.get("tool_calls")
        if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):
            for call in calls[:1]:
                if isinstance(call, Mapping):
                        outcome["applied_tool_edit"] = list(
                            apply_model_tool_call(context.workspace, call, journal=journal))
        context.outcome = outcome
        journal.emit("model_response", request_id=context.request_id,
                     response=outcome, fresh=True)
        if outcome.get("needs_review"):
            self._budget_failure("branch model returned needs_review")
        if outcome.get("public_green") is True or outcome.get("public_suite"):
            record = self.preserve(
                context.workspace,
                host_assessments=tuple(outcome.get("host_assessments", ())),
                host_receipts=tuple(outcome.get("host_receipts", ())),
                current_snapshot=outcome.get("current_snapshot"),
                public_suite=tuple(outcome.get("public_suite", ())),
                public_green=outcome.get("public_green"),
                reason=f"branch:{context.branch_id}")
            if record is not None:
                outcome["candidate_id"] = record.candidate_id
                journal.emit("candidate_preserved", candidate_id=record.candidate_id,
                             workspace_hash=record.workspace_hash)
        journal.emit("branch_finished", branch_id=context.branch_id,
                     candidate_id=outcome.get("candidate_id"), status=outcome.get("status", "done"))
        return context

    def fork(self, parent_workspace: str | Path, *, refusal: object,
             request: Mapping[str, object] | None = None, model_request: Callable | None = None,
             temperatures: Sequence[float] | None = None) -> tuple[BranchContext, ...]:
        progress = bool(_refusal_field(refusal, "verified_progress", False))
        payload = _refusal_field(refusal, "host_payload", {})
        generated = bool(_refusal_field(refusal, "next_repair_generated", False))
        if not productive_refusal(verified_progress=progress, host_payload=payload,
                                  next_repair_generated=generated,
                                  reproduced_localized_violation=_refusal_field(
                                      refusal, "reproduced_localized_violation", None)):
            self.journal.emit("fork_not_triggered", reason="not_productive_refusal")
            return ()
        key = self._refusal_key(refusal)
        if key in self._forked_repair_keys:
            self.journal.emit("fork_not_triggered", reason="repair_already_generated",
                              repair_key=key)
            return ()
        self._forked_repair_keys.add(key)
        parent = Path(parent_workspace).resolve()
        children: list[BranchContext] = []
        temperatures = tuple(temperatures) if temperatures is not None else ()
        for index in range(self.policy.forks):
            if not self.ledger.charge("fork", reason=f"repair:{key}:child:{index}"):
                self._budget_failure("fork budget exhausted before K children")
                break
            temperature = (temperatures[index] if index < len(temperatures)
                           else self.policy.temperature_for(index))
            try:
                context = self._new_branch(parent, root_index=None, fork_index=index,
                                           temperature=temperature)
                children.append(self._run_branch(context, request=request,
                                                 model_request=model_request))
            except Exception as error:
                self._budget_failure(f"fork child {index} failed: {type(error).__name__}")
                self.journal.emit("branch_failed", fork_index=index,
                                  error_type=type(error).__name__, detail=str(error)[:240])
        self.journal.emit("fork_finished", repair_key=key, requested=self.policy.forks,
                          created=len(children), branches=[item.branch_id for item in children],
                          ledger=self.ledger.to_dict())
        return tuple(children)

    def independent_roots(self, parent_workspace: str | Path, *, request: Mapping[str, object] | None = None,
                           model_request: Callable | None = None) -> tuple[BranchContext, ...]:
        roots: list[BranchContext] = []
        parent = Path(parent_workspace).resolve()
        for index in range(self.policy.roots):
            if not self.ledger.charge("root", reason=f"independent-root:{index}"):
                self._budget_failure("independent root budget exhausted")
                break
            try:
                context = self._new_branch(
                    parent, root_index=index, fork_index=None,
                    temperature=self.policy.temperature_for(index))
                roots.append(self._run_branch(context, request=request,
                                              model_request=model_request))
            except Exception as error:
                self._budget_failure(f"independent root {index} failed: {type(error).__name__}")
                self.journal.emit("branch_failed", root_index=index,
                                  error_type=type(error).__name__, detail=str(error)[:240])
        self.journal.emit("independent_roots_finished", requested=self.policy.roots,
                          created=len(roots), roots=[item.branch_id for item in roots],
                          ledger=self.ledger.to_dict())
        return tuple(roots)

    @staticmethod
    def _selection_check(check: object, candidate: CandidateRecord) -> dict[str, object]:
        if callable(check):
            result = _call_selection_callback(check, candidate)
        elif isinstance(check, Mapping):
            result = dict(check)
        else:
            result = {"check_id": str(check), "passed": False,
                      "grounded": False, "detail": "invalid selection check"}
        if isinstance(result, bool):
            return {"check_id": "anonymous", "passed": result, "grounded": False,
                    "detail": "boolean checks lack ticket provenance"}
        if not isinstance(result, Mapping):
            return {"check_id": "anonymous", "passed": False, "grounded": False,
                    "detail": "selection check returned a non-object"}
        row = dict(_json_value(result))
        row.setdefault("check_id", "selection-check")
        row["passed"] = bool(row.get("passed"))
        row["grounded"] = bool(row.get("grounded"))
        return row

    def select(self, *, selection_checks: Sequence[object] = (),
               final_recheck: Callable | None = None) -> SelectionResult:
        candidates = self.pool()
        hard = tuple(candidate for candidate in candidates if candidate.eligible and
                     (not self.ticket_id or candidate.ticket_id in {None, self.ticket_id}))
        hard_ids = tuple(candidate.candidate_id for candidate in hard)
        check_rows: list[dict[str, object]] = []
        passing: list[CandidateRecord] = []
        for candidate in hard:
            candidate_rows = []
            for check in selection_checks:
                row = self._selection_check(check, candidate)
                row["candidate_id"] = candidate.candidate_id
                # Selection checks must carry ticket provenance and cannot
                # silently reuse host receipt statuses as a vote.
                row["separate_check_set"] = True
                row["ticket_match"] = row.get("ticket_id") == self.ticket_id
                row["passed"] = bool(row["passed"] and row["grounded"] and row["ticket_match"])
                candidate_rows.append(row)
                check_rows.append(row)
            if all(bool(row.get("passed")) for row in candidate_rows):
                passing.append(candidate)
            elif not selection_checks:
                passing.append(candidate)
        if not selection_checks:
            passing = list(hard)
        selected = min(passing, key=lambda item: (item.diff_size, item.candidate_id)) if passing else None
        tie = {
            "criterion": "smaller_diff_then_candidate_id",
            "ordered_candidates": [item.candidate_id for item in sorted(
                passing, key=lambda item: (item.diff_size, item.candidate_id))],
        }
        final = {"status": "NOT_RUN", "workspace": None}
        if selected is not None:
            clean_dir = self.store_root / "clean-rechecks" / (
                selected.candidate_id.replace(":", "-") + "-" + uuid.uuid4().hex[:8])
            clean_workspace = clean_dir / "workspace"
            _copy_workspace(selected.workspace, clean_workspace)
            integrity = workspace_tree_hash(clean_workspace) == selected.workspace_hash
            final = {"status": "PASS" if integrity else "FAIL",
                     "integrity": integrity, "workspace": str(clean_workspace)}
            if integrity and final_recheck is not None:
                if not self.ledger.charge("final_recheck", reason=selected.candidate_id):
                    final.update(status="NEEDS_REVIEW", reason="closure reserve unavailable")
                else:
                    result = _call_final_recheck(final_recheck, clean_workspace, selected)
                    passed = bool(result if isinstance(result, bool)
                                  else result.get("passed") if isinstance(result, Mapping) else False)
                    final.update(status="PASS" if passed else "FAIL", result=_json_value(result))
            if final.get("status") != "PASS":
                selected = None
        self.store.commit_pool(candidates)
        status = "selected" if selected is not None else "needs_review"
        reason = "" if selected is not None else (
            "no hard-eligible candidate, discriminating check, or final recheck passed")
        result = SelectionResult(
            status=status, selected_candidate_id=selected.candidate_id if selected else None,
            pool_commit_path=self.store.pool_path,
            selection_path=self.store_root / "selection.json",
            hard_eligible_ids=hard_ids, discriminating_checks=tuple(check_rows),
            tie_break=tie, final_recheck=final, reason=reason)
        selection_payload = {
            **result.to_dict(), "schema": "stateframe-selection/1",
            "ticket_id": self.ticket_id, "external_grading_started": False,
            "pool_committed_before_external_grading": True,
            "pool": [item.to_dict(workspace=str(item.workspace)) for item in candidates],
            "ledger": self.ledger.to_dict(),
        }
        self._write_selection(selection_payload)
        self._selection = result
        self.journal.emit("selection_committed", selection=selection_payload,
                          external_grading_started=False)
        if selected is None:
            self.status = "needs_review"
            self.needs_review = True
        return result

    def _write_selection(self, payload: Mapping[str, object]) -> None:
        path = self.store_root / "selection.json"
        encoded = json.dumps(_json_value(payload), sort_keys=True, indent=2) + "\n"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            # A selection is committed once.  A later write may only add the
            # external-grading lifecycle field, never change the selected id.
            if existing.get("selected_candidate_id") != payload.get("selected_candidate_id"):
                raise PortfolioError("selection record cannot be replaced with another candidate")
        path.write_text(encoded, encoding="utf-8")

    def grade_selected(self, grader: Callable) -> object:
        if self._selection is None or self._selection.selected_candidate_id is None:
            raise PortfolioError("cannot grade without a committed selection")
        if self.needs_review:
            raise PortfolioError("portfolio is needs_review; external grading is withheld")
        selected = next((item for item in self.pool()
                         if item.candidate_id == self._selection.selected_candidate_id), None)
        if selected is None:
            raise PortfolioError("selected candidate record is missing")
        clean_dir = self.store_root / "grading" / (selected.candidate_id.replace(":", "-") + "-" + uuid.uuid4().hex[:8])
        clean_workspace = clean_dir / "workspace"
        _copy_workspace(selected.workspace, clean_workspace)
        path = self.store_root / "selection.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["external_grading_started"] = True
        payload["external_grading_candidate_id"] = selected.candidate_id
        self._write_selection(payload)
        self.journal.emit("external_grading_started", candidate_id=selected.candidate_id,
                          pool_path=str(self.pool_path), selection_path=str(path))
        result = _call_grader(grader, clean_workspace, selected)
        self.journal.emit("external_grading_finished", candidate_id=selected.candidate_id,
                          result=_json_value(result))
        return result


def _call_selection_callback(callback: Callable, candidate: CandidateRecord) -> object:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        parameters = {}
    positional = [item for item in parameters.values()
                  if item.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2:
        return callback(candidate, candidate.workspace)
    return callback(candidate)


def _call_final_recheck(callback: Callable, workspace: Path, candidate: CandidateRecord) -> object:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        parameters = {}
    positional = [item for item in parameters.values()
                  if item.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2:
        return callback(workspace, candidate)
    return callback(workspace)


def _call_grader(callback: Callable, workspace: Path, candidate: CandidateRecord) -> object:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        parameters = {}
    positional = [item for item in parameters.values()
                  if item.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                   inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    if len(positional) >= 2:
        return callback(workspace, candidate)
    return callback(workspace)


def simulate_portfolio(base: str | Path, *, output: str | Path | None = None) -> dict[str, object]:
    """Run a deterministic fork/preserve/select/budget-exhaustion proof."""
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    parent = base / "parent"
    if not parent.exists():
        parent.mkdir()
        (parent / "app.py").write_text("VALUE = 'parent'\n", encoding="utf-8")
    policy = PortfolioPolicy(roots=2, forks=2, budget=5, closure_reserve=1,
                             temperatures=(0.3, 1.0))
    manager = PortfolioManager(parent, store_root=Path(output or (base / "records")),
                               policy=policy, ticket_id="simulation-ticket")
    manager.preserve(
        parent,
        host_assessments=[{"obligation_id": "o1", "verdict": "SATISFIED"}],
        host_receipts=[{"check_id": "h1", "status": "PASS"}],
        public_suite=[{"result_class": "PASS"}], reason="parent-green")

    refusal = {
        "verified_progress": True,
        "host_payload": {
            "mode": "REPAIR", "focus_obligation": "o1", "observed": "bad",
            "expected": "good", "counterexample_ref": "simulation-refusal",
        },
    }

    def model(context: BranchContext) -> dict[str, object]:
        extra = "EXTRA = 1\n" if context.fork_index == 1 else ""
        return {
            "edits": [{"path": "app.py", "old": "VALUE = 'parent'\n",
                       "new": "VALUE = 'GOOD'\n" + extra}],
            "host_assessments": [{"obligation_id": "o1", "verdict": "SATISFIED"}],
            "host_receipts": [{"check_id": "h1", "status": "PASS"}],
            "public_suite": [{"result_class": "PASS", "selector": "ticket::public"}],
            "public_green": True,
        }

    branches = manager.fork(parent, refusal=refusal,
                            request={"messages": [{"role": "user", "content": "repair"}]},
                            model_request=model)
    selection = manager.select(
        selection_checks=[lambda candidate, workspace: {
            "check_id": "ticket-output", "ticket_id": "simulation-ticket",
            "grounded": True, "passed": "GOOD" in (
                workspace / "app.py").read_text(encoding="utf-8"),
        }],
        final_recheck=lambda workspace, candidate: "GOOD" in (
            workspace / "app.py").read_text(encoding="utf-8"),
    )
    # The reserve keeps one closure/final-recheck unit unavailable to exploration.
    # Once selection is committed, the independent-root fallback consumes the
    # remaining exploration slot and then truthfully becomes needs_review.
    manager.independent_roots(parent, model_request=lambda context: {})
    pool = manager.pool()
    pool_record = json.loads(manager.pool_path.read_text(encoding="utf-8"))
    selection_record = json.loads((manager.store_root / "selection.json").read_text(encoding="utf-8"))
    return {
        "schema": SCHEMA, "status": manager.status,
        "fork_count": len(branches), "preserved_count": len(pool),
        "selected_candidate_id": selection.selected_candidate_id,
        "pool_record": pool_record, "selection_record": selection_record,
        "ledger": manager.ledger.to_dict(),
    }


__all__ = [
    "CandidateRecord", "CandidateStore", "Eligibility", "PortfolioError",
    "PortfolioJournal", "PortfolioLedger", "PortfolioManager", "PortfolioPolicy",
    "BranchContext", "SelectionResult", "apply_model_tool_call", "apply_scripted_edits",
    "candidate_eligibility",
    "digest", "enabled", "mode", "productive_refusal", "simulate_portfolio",
    "workspace_tree_hash",
]
