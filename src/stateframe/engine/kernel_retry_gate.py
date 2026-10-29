"""Changed-condition retry gate for the single-task repair bench.

The gate is a shadow organ.  It consumes the same mapping-shaped receipts that
already pass through the repair kernel, but it does not execute edits, tests, or
reads.  An applied edit becomes a failed attempt only after its next test is
red; rejected and no-op edits are already failed at the editor boundary.

The implementation intentionally keeps the edit-dispatch seam out of this
module.  A future caller can ask :class:`RetryGate` about an edit before
dispatch, then persist the returned decision receipt through the existing
``organ_out``/receipt path.
"""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState, run_census
from kernel_pivot import fingerprint as stable_fingerprint


SCHEMA = "poc.kernel_retry_gate/1"
ORGAN_ID = "bolt.retry_gate"
MAX_RECEIPTS = 512
MAX_SOURCE_RECEIPTS = 512
MAX_FAILED_ATTEMPTS = 256
MAX_PENDING_ATTEMPTS = 64
MAX_MESSAGE_CHARS = 280

Record: TypeAlias = Mapping[str, object]
ReceiptSink: TypeAlias = Callable[[Mapping[str, object]], object]


class Decision(StrEnum):
    """Decision returned for an observed action."""

    ALLOW = "allow"
    WARN = "warn"
    DENY = "deny"


class RetryGateError(ValueError):
    """Base error for malformed retry-gate inputs."""


def _text(value: object, *, limit: int = 320) -> str:
    return " ".join(str(value or "").split())[:limit]


def _normal_kind(value: object) -> str:
    return _text(value, limit=64).lower().replace("-", "_").replace(" ", "_")


def _int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(value)
    return ()


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


_TARGET_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.[A-Za-z_][A-Za-z0-9_]*"
    r"(?:(?:::|#)[A-Za-z_][A-Za-z0-9_]*)?"
)
_LINE_SUFFIX_RE = re.compile(r"@\d+(?:-\d+)?$")
_ANCHOR_LINE_RE = re.compile(r":\d+(?::|$)")


def _normal_target(value: object) -> str:
    """Normalize a path/function target while removing line-number noise."""

    result = _text(value, limit=240).replace("\\", "/")
    result = _LINE_SUFFIX_RE.sub("", result)
    result = _ANCHOR_LINE_RE.sub("::", result)
    result = result.strip().lstrip("./")
    if "::" in result:
        path, function = result.split("::", 1)
        path = path.rstrip(":").strip().lstrip("./")
        function = _text(function, limit=120)
        result = f"{path}::{function}" if path and function else path
    return result[:240]


def _target_from_text(value: object) -> str:
    raw = _text(value, limit=500)
    if not raw:
        return ""
    match = _TARGET_RE.search(raw)
    if match is not None:
        return _normal_target(match.group(0))
    return ""


def _target_from_record(record: Record) -> str:
    """Resolve an explicit path/function before falling back to legacy intent."""

    for key in (
        "target_path_or_fn",
        "target_path",
        "target_fn",
        "target_function",
        "function",
        "fn",
        "path",
        "file_path",
        "file",
    ):
        value = record.get(key)
        if isinstance(value, Mapping):
            nested = _target_from_record(value)
            if nested:
                return nested
        normalized = _normal_target(value)
        if normalized:
            return normalized

    target = record.get("target")
    if isinstance(target, Mapping):
        normalized = _target_from_record(target)
        if normalized:
            return normalized
    normalized_target = _normal_target(target)
    if normalized_target:
        return normalized_target

    for key in ("intent", "anchor", "traceback", "detail", "line"):
        normalized = _target_from_text(record.get(key))
        if normalized:
            return normalized
    return ""


def _normalize_patch_value(value: object) -> object:
    if isinstance(value, str):
        lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        return "\n".join(line.rstrip() for line in lines).strip()
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_patch_value(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_patch_value(item) for item in value]
    return value


def normalize_edit_digest(value: object) -> str:
    """Return a stable digest/token for an edit description.

    Existing bench receipts expose a short opaque ``fingerprint`` rather than
    the patch body; that token is retained as-is.  Raw patch fields are
    canonicalized and content-addressed so whitespace and mapping order do not
    create a false change.
    """

    if isinstance(value, Mapping):
        for key in (
            "normalized_edit_digest",
            "edit_digest",
            "patch_digest",
            "fingerprint",
        ):
            candidate = _text(value.get(key), limit=240)
            if candidate:
                return normalize_edit_digest(candidate)
        patch_keys = (
            "old",
            "old_text",
            "new",
            "new_text",
            "patch",
            "replacement",
            "operations",
            "changes",
            "content",
        )
        payload = {
            key: _normalize_patch_value(value[key])
            for key in patch_keys
            if key in value
        }
        if payload:
            return stable_fingerprint(payload)
        return ""

    normalized = _text(value, limit=240)
    if not normalized:
        return ""
    if normalized.lower().startswith("sha256:"):
        prefix, digest = normalized.split(":", 1)
        return f"{prefix.lower()}:{digest.lower()}"
    if re.fullmatch(r"[0-9a-fA-F]{8,128}", normalized):
        return normalized.lower()
    return normalized


def _read_set_revision_from_value(value: object) -> int | None:
    if isinstance(value, Mapping):
        for key in (
            "workspace_revision",
            "workspace_revision_of_read_set",
            "read_set_revision",
            "read_revision",
            "revision",
        ):
            if value.get(key) is not None:
                return max(0, _int(value.get(key)))
        return None
    revisions = [
        revision
        for item in _sequence(value)
        if (revision := _read_set_revision_from_value(item)) is not None
    ]
    return max(revisions) if revisions else None


def _read_set_revision(record: Record, default: int | None = None) -> int:
    for key in (
        "read_set_revision",
        "workspace_revision_of_read_set",
        "read_revision",
    ):
        value = record.get(key)
        if value is not None:
            return max(0, _int(value))
    from_read_set = _read_set_revision_from_value(record.get("read_set"))
    if from_read_set is not None:
        return from_read_set
    if default is not None:
        return max(0, default)
    return max(0, _int(record.get("workspace_revision", record.get("revision"))))


def _hypothesis_id(record: Record) -> str:
    for key in (
        "leading_hypothesis_id",
        "hypothesis_id",
        "leading_hypothesis",
        "hypothesis",
    ):
        value = record.get(key)
        if isinstance(value, Mapping):
            value = value.get("id") or value.get("hypothesis_id") or value.get("key")
        normalized = _text(value, limit=180)
        if normalized:
            return normalized
    return ""


@dataclass(frozen=True, slots=True)
class AttemptFingerprint:
    """The five stable components that define one edit attempt."""

    action_kind: str
    target: str
    edit_digest: str
    leading_hypothesis_id: str = ""
    read_set_revision: int = 0

    def __post_init__(self) -> None:
        action_kind = _normal_kind(self.action_kind)
        target = _normal_target(self.target)
        edit_digest = normalize_edit_digest(self.edit_digest)
        hypothesis = _text(self.leading_hypothesis_id, limit=180)
        revision = max(0, _int(self.read_set_revision))
        if not action_kind:
            raise RetryGateError("attempt action kind is required")
        if not target:
            raise RetryGateError("attempt target path/function is required")
        if not edit_digest:
            raise RetryGateError("attempt normalized edit digest is required")
        object.__setattr__(self, "action_kind", action_kind)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "edit_digest", edit_digest)
        object.__setattr__(self, "leading_hypothesis_id", hypothesis)
        object.__setattr__(self, "read_set_revision", revision)

    @classmethod
    def from_mapping(
        cls,
        value: Record,
        *,
        read_set_revision: int | None = None,
    ) -> "AttemptFingerprint":
        action_kind = _normal_kind(
            value.get("action_kind", value.get("kind", value.get("type", "")))
        )
        target = _target_from_record(value)
        digest = normalize_edit_digest(value)
        if not digest:
            digest = normalize_edit_digest(value.get("intent"))
        return cls(
            action_kind=action_kind,
            target=target,
            edit_digest=digest,
            leading_hypothesis_id=_hypothesis_id(value),
            read_set_revision=_read_set_revision(value, read_set_revision),
        )

    @property
    def components(self) -> tuple[str, str, str, str, int]:
        return (
            self.action_kind,
            self.target,
            self.edit_digest,
            self.leading_hypothesis_id,
            self.read_set_revision,
        )

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "action_kind": self.action_kind,
                "target": self.target,
                "edit_digest": self.edit_digest,
                "leading_hypothesis_id": self.leading_hypothesis_id,
                "read_set_revision": self.read_set_revision,
            }
        )

    @property
    def target_path_or_fn(self) -> str:
        return self.target

    @property
    def normalized_edit_digest(self) -> str:
        return self.edit_digest

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "action_kind": self.action_kind,
            "target": self.target,
            "target_path_or_fn": self.target,
            "edit_digest": self.edit_digest,
            "normalized_edit_digest": self.edit_digest,
            "hypothesis_id": self.leading_hypothesis_id,
            "leading_hypothesis_id": self.leading_hypothesis_id,
            "read_set_revision": self.read_set_revision,
            "workspace_revision_of_read_set": self.read_set_revision,
            "fingerprint": self.fingerprint,
        }


def attempt_fingerprint(
    value: Record,
    *,
    read_set_revision: int | None = None,
) -> AttemptFingerprint:
    """Build the typed five-component fingerprint for one action mapping."""

    return AttemptFingerprint.from_mapping(value, read_set_revision=read_set_revision)


@dataclass(frozen=True, slots=True)
class RetryDecision:
    """Bounded result of observing one action through the gate."""

    decision: Decision
    turn: int
    attempt: AttemptFingerprint | None = None
    repeat_number: int = 0
    failed_turn: int | None = None
    reason: str = ""
    message: str = ""
    event: str = ""
    source_fingerprint: str = ""
    deferred: bool = False

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW

    @property
    def warned(self) -> bool:
        return self.decision is Decision.WARN

    @property
    def denied(self) -> bool:
        return self.decision is Decision.DENY

    @property
    def fingerprint(self) -> str:
        return self.attempt.fingerprint if self.attempt else ""

    @property
    def kind(self) -> Decision:
        return self.decision

    @property
    def gates_action(self) -> bool:
        """Whether this result is the decision for the observed action itself."""

        return self.attempt is not None and not self.deferred

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "organ_id": ORGAN_ID,
            "kind": "retry_decision",
            "event": self.event,
            "decision": self.decision.value,
            "turn": self.turn,
            "attempt": self.attempt.to_dict() if self.attempt else None,
            "attempt_fingerprint": self.fingerprint,
            "repeat_number": self.repeat_number,
            "failed_turn": self.failed_turn,
            "reason": self.reason,
            "message": self.message,
            "source_fingerprint": self.source_fingerprint,
            "deferred": self.deferred,
        }

    def render(self) -> str:
        return (self.message or self.reason)[:MAX_MESSAGE_CHARS]


@dataclass(frozen=True, slots=True)
class RetryCensus:
    """Compact counters plus the receipt snapshot used to derive them."""

    allowed: int
    warned: int
    denied: int
    bypassed: int
    failed_attempts: int
    material_change_allows: int
    new_receipt_fingerprints: int
    digest: str
    deferred_denials: int = 0
    events: tuple[Mapping[str, object], ...] = ()

    @property
    def allow(self) -> int:
        return self.allowed

    @property
    def warn(self) -> int:
        return self.warned

    @property
    def deny(self) -> int:
        return self.denied

    @property
    def N(self) -> int:
        return self.allowed + self.warned + self.denied + self.bypassed

    @property
    def M(self) -> int:
        return self.warned + self.denied

    @property
    def S(self) -> int:
        return self.allowed + self.bypassed

    @property
    def D(self) -> int:
        return self.denied

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "organ_id": ORGAN_ID,
            "allowed": self.allowed,
            "warned": self.warned,
            "denied": self.denied,
            "bypassed": self.bypassed,
            "failed_attempts": self.failed_attempts,
            "material_change_allows": self.material_change_allows,
            "new_receipt_fingerprints": self.new_receipt_fingerprints,
            "deferred_denials": self.deferred_denials,
            "N": self.N,
            "M": self.M,
            "S": self.S,
            "D": self.D,
            "digest": self.digest,
        }

    def render(self) -> str:
        return (
            f"RETRY GATE: allow={self.allowed} warn={self.warned} "
            f"deny={self.denied} bypass={self.bypassed} failed={self.failed_attempts}"
        )[:MAX_MESSAGE_CHARS]


@dataclass(frozen=True, slots=True)
class _PendingAttempt:
    attempt: AttemptFingerprint
    turn: int
    repeat_number: int = 0
    failed_turn: int | None = None


@dataclass(frozen=True, slots=True)
class _FailedAttempt:
    attempt: AttemptFingerprint
    failed_turn: int
    confirmed_turn: int
    condition_epoch: int
    reason: str


def _record_fingerprint(record: Record) -> str:
    for key in ("receipt_fingerprint", "evidence_fingerprint"):
        supplied = _text(record.get(key), limit=180)
        if supplied:
            return supplied
    supplied = _text(record.get("fingerprint"), limit=180)
    if supplied:
        # A stable selector/action fingerprint with a changed outcome is a
        # new observation, while repeated identical outcomes remain repeats.
        return stable_fingerprint(
            {
                "fingerprint": supplied,
                "kind": _record_kind(record),
                "result_class": _result_class(record),
            }
        )
    for key in ("receipt_id", "id"):
        supplied = _text(record.get(key), limit=180)
        if supplied:
            return supplied
    material = {
        key: record.get(key)
        for key in (
            "kind",
            "action_kind",
            "intent",
            "path",
            "target_path",
            "file",
            "result_class",
            "result",
            "detail",
            "anchor",
            "hypothesis_id",
            "event",
        )
        if record.get(key) is not None
    }
    return stable_fingerprint(material)


def _record_kind(record: Record) -> str:
    return _normal_kind(record.get("kind", record.get("action_kind", record.get("type", ""))))


def _result_class(record: Record) -> str:
    return _text(record.get("result_class", record.get("result", "")), limit=64).upper()


def _is_test(record: Record) -> bool:
    return _record_kind(record) in {"test", "pytest", "test_run"}


def _is_red_test(record: Record) -> bool:
    return _is_test(record) and _result_class(record) in {"RED", "FAILED", "FAILURE"}


def _is_green_test(record: Record) -> bool:
    return _is_test(record) and _result_class(record) in {"GREEN", "PASSED", "PASS", "OK"}


def _is_read(record: Record) -> bool:
    return _record_kind(record) in {"read", "grep", "search", "observation"}


def _is_successful_read(record: Record) -> bool:
    return _is_read(record) and _result_class(record) in {"OK", "HITS", "PASS", "PASSED", "GREEN"}


def _is_immediate_edit_failure(record: Record) -> bool:
    return _record_kind(record) == "edit" and _result_class(record) in {
        "NO_OP",
        "REJECTED",
        "ERROR",
        "FAILED",
        "FAILURE",
        "BLOCKED",
        "TIMEOUT",
    }


def _is_reobservation(record: Record) -> bool:
    event = _normal_kind(record.get("event", ""))
    return (
        event
        in {
            "invalidation_reobserved",
            "dirty_claim_reobserved",
            "claim_reobserved",
            "reobserve",
            "re_observed",
        }
        or _truthy(record.get("dirty_claim_reobserved"))
        or _truthy(record.get("claim_reobserved"))
        or _truthy(record.get("reobserved"))
    )


def _is_hypothesis_change(record: Record) -> bool:
    event = _normal_kind(record.get("event", ""))
    if event in {
        "hypothesis_changed",
        "hypothesis_change",
        "new_hypothesis",
        "reframe_hypothesis",
        "claim_changed",
    }:
        return True
    return any(
        _truthy(record.get(key))
        for key in ("hypothesis_changed", "changed_hypothesis", "new_hypothesis")
    )


def _retry_trigger(evidence: Evidence) -> TriState:
    event = _text(evidence.event.get("event")).upper()
    if event in {"RETRY_WARN", "RETRY_DENY"}:
        return TriState.YES
    if event in {
        "RETRY_ALLOW",
        "RETRY_ALLOW_CHANGED_CONDITION",
        "RETRY_BYPASS",
        "RETRY_INPUT_BYPASS",
        "RETRY_ATTEMPT_FAILED",
    }:
        return TriState.NO
    return TriState.UNKNOWN if not evidence.complete else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id=ORGAN_ID,
    trigger=_retry_trigger,
    fire_events=frozenset({"RETRY_WARN", "RETRY_DENY"}),
    suppression_events=frozenset(
        {
            "RETRY_ALLOW",
            "RETRY_ALLOW_CHANGED_CONDITION",
            "RETRY_BYPASS",
            "RETRY_INPUT_BYPASS",
            "RETRY_ATTEMPT_FAILED",
        }
    ),
    dedup_events=frozenset({"RETRY_DECISION_DEDUPED"}),
    challenge_once_events=frozenset({"RETRY_WARN"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


class RetryGate:
    """Bounded, deterministic retry state machine over action receipts."""

    organ_id = ORGAN_ID

    def __init__(
        self,
        *,
        max_receipts: int = MAX_RECEIPTS,
        max_failed_attempts: int = MAX_FAILED_ATTEMPTS,
        max_pending_attempts: int = MAX_PENDING_ATTEMPTS,
        receipt_sink: ReceiptSink | None = None,
    ) -> None:
        if max_receipts < 1:
            raise ValueError("max_receipts must be positive")
        if max_failed_attempts < 1:
            raise ValueError("max_failed_attempts must be positive")
        if max_pending_attempts < 1:
            raise ValueError("max_pending_attempts must be positive")
        self.max_failed_attempts = max_failed_attempts
        self.max_pending_attempts = max_pending_attempts
        self.receipt_sink = receipt_sink
        self._receipts: deque[dict[str, object]] = deque(maxlen=max_receipts)
        self._source_receipts: deque[dict[str, object]] = deque(maxlen=MAX_SOURCE_RECEIPTS)
        self._seen_receipt_fingerprints: set[str] = set()
        self._seen_receipt_order: deque[str] = deque(maxlen=MAX_SOURCE_RECEIPTS)
        self._read_set_revisions: dict[str, int] = {}
        self._failed: dict[str, deque[_FailedAttempt]] = {}
        self._failed_order: deque[tuple[str, _FailedAttempt]] = deque()
        self._pending: deque[_PendingAttempt] = deque(maxlen=max_pending_attempts)
        self._repeat_counts: dict[tuple[str, int], int] = {}
        self._condition_epoch = 0
        self._last_hypothesis_id = ""
        self._counts: Counter[str] = Counter()
        self._decisions: deque[RetryDecision] = deque(maxlen=max_receipts)
        self._last_turn = 0

    @property
    def receipts(self) -> tuple[Record, ...]:
        return tuple(dict(receipt) for receipt in self._receipts)

    @property
    def events(self) -> tuple[Record, ...]:
        return self.receipts

    @property
    def source_receipts(self) -> tuple[Record, ...]:
        return tuple(dict(receipt) for receipt in self._source_receipts)

    @property
    def decisions(self) -> tuple[RetryDecision, ...]:
        return tuple(self._decisions)

    @property
    def condition_epoch(self) -> int:
        return self._condition_epoch

    @property
    def warnings(self) -> int:
        return self._counts[Decision.WARN.value]

    @property
    def denials(self) -> int:
        return self._counts[Decision.DENY.value]

    def _emit(
        self,
        event: str,
        *,
        turn: int = 0,
        attempt: AttemptFingerprint | None = None,
        decision: Decision | None = None,
        source_fingerprint: str = "",
        detail: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "organ_id": ORGAN_ID,
            "kind": "retry_gate",
            "event": event,
            "turn": max(0, turn),
            "decision": decision.value if decision else "",
            "attempt_fingerprint": attempt.fingerprint if attempt else "",
            "source_fingerprint": source_fingerprint,
            "detail": dict(detail or {}),
        }
        payload["receipt_id"] = "retry:" + stable_fingerprint(payload).removeprefix("sha256:")[:24]
        self._receipts.append(payload)
        if self.receipt_sink is not None:
            try:
                self.receipt_sink(payload)
            except Exception as exc:  # noqa: BLE001 - sink failure cannot hide the decision
                self._receipts.append(
                    {
                        "schema": SCHEMA,
                        "organ_id": ORGAN_ID,
                        "kind": "retry_gate",
                        "event": "RETRY_RECEIPT_SINK_ERROR",
                        "turn": max(0, turn),
                        "error": type(exc).__name__,
                    }
                )
        return payload

    def _remember_decision(self, result: RetryDecision) -> RetryDecision:
        self._decisions.append(result)
        self._counts[result.decision.value] += 1
        self._emit(
            result.event,
            turn=result.turn,
            attempt=result.attempt,
            decision=result.decision,
            source_fingerprint=result.source_fingerprint,
            detail={
                "repeat_number": result.repeat_number,
                "failed_turn": result.failed_turn,
                "reason": result.reason,
                "message": result.message,
            },
        )
        if result.reason == "material_condition_changed":
            self._counts["material_change_allows"] += 1
        return result

    def _note_source(self, record: Record) -> tuple[str, bool]:
        source_fingerprint = _record_fingerprint(record)
        is_new = source_fingerprint not in self._seen_receipt_fingerprints
        if is_new:
            if len(self._seen_receipt_order) >= MAX_SOURCE_RECEIPTS:
                expired = self._seen_receipt_order.popleft()
                self._seen_receipt_fingerprints.discard(expired)
            self._seen_receipt_order.append(source_fingerprint)
            self._seen_receipt_fingerprints.add(source_fingerprint)
            self._condition_epoch += 1
            self._counts["new_receipt_fingerprints"] += 1
        self._source_receipts.append(dict(record))
        return source_fingerprint, is_new

    def _note_condition_events(self, record: Record) -> None:
        reobserved = _is_reobservation(record)
        hypothesis_changed = _is_hypothesis_change(record)
        hypothesis = _hypothesis_id(record)
        if hypothesis and self._last_hypothesis_id and hypothesis != self._last_hypothesis_id:
            hypothesis_changed = True
        if reobserved:
            self._condition_epoch += 1
            self._counts["dirty_claim_reobservations"] += 1
        if hypothesis_changed:
            self._condition_epoch += 1
            self._counts["hypothesis_changes"] += 1
        if hypothesis:
            self._last_hypothesis_id = hypothesis
        if _truthy(record.get("material_change")):
            self._condition_epoch += 1
            self._counts["explicit_material_changes"] += 1

    def _update_read_set(self, record: Record, *, is_new_receipt: bool) -> None:
        if not _is_successful_read(record) or not is_new_receipt:
            return
        target = _target_from_record(record)
        if not target:
            return
        revision = _read_set_revision(record)
        target_key = target.casefold()
        self._read_set_revisions[target_key] = max(
            revision, self._read_set_revisions.get(target_key, 0)
        )

    def _read_revision_for(self, target: str) -> int | None:
        target_key = target.casefold()
        if target_key in self._read_set_revisions:
            return self._read_set_revisions[target_key]
        path = target.split("::", 1)[0].casefold()
        for known_target, revision in self._read_set_revisions.items():
            known_path = known_target.split("::", 1)[0]
            if known_path == path or known_path.endswith("/" + path) or path.endswith("/" + known_path):
                return revision
        return None

    def _attempt(self, record: Record) -> AttemptFingerprint:
        target = _target_from_record(record)
        return attempt_fingerprint(
            record,
            read_set_revision=_read_set_revision(record, self._read_revision_for(target)),
        )

    def _matching_failure(self, attempt: AttemptFingerprint) -> _FailedAttempt | None:
        candidates = self._failed.get(attempt.fingerprint, ())
        for failure in reversed(candidates):
            if failure.condition_epoch == self._condition_epoch and failure.attempt == attempt:
                return failure
        return None

    def _prior_failure(self, attempt: AttemptFingerprint) -> _FailedAttempt | None:
        candidates = self._failed.get(attempt.fingerprint, ())
        return candidates[-1] if candidates else None

    def _record_failure(
        self,
        attempt: AttemptFingerprint,
        *,
        failed_turn: int,
        confirmed_turn: int,
        reason: str,
    ) -> None:
        failure = _FailedAttempt(
            attempt=attempt,
            failed_turn=max(0, failed_turn),
            confirmed_turn=max(0, confirmed_turn),
            condition_epoch=self._condition_epoch,
            reason=_text(reason, limit=120),
        )
        if len(self._failed_order) >= self.max_failed_attempts:
            old_key, old_failure = self._failed_order.popleft()
            old_failures = self._failed.get(old_key)
            if old_failures is not None:
                try:
                    old_failures.remove(old_failure)
                except ValueError:
                    pass
                if not old_failures:
                    self._failed.pop(old_key, None)
        self._failed.setdefault(attempt.fingerprint, deque()).append(failure)
        self._failed_order.append((attempt.fingerprint, failure))
        self._counts["failed_attempts"] += 1
        self._emit(
            "RETRY_ATTEMPT_FAILED",
            turn=max(0, confirmed_turn),
            attempt=attempt,
            detail={
                "failed_turn": max(0, failed_turn),
                "confirmed_turn": max(0, confirmed_turn),
                "condition_epoch": self._condition_epoch,
                "reason": reason,
            },
        )

    def _confirm_pending(self, record: Record) -> None:
        if not self._pending:
            return
        pending = tuple(self._pending)
        self._pending.clear()
        if not _is_red_test(record):
            return
        turn = max(0, _int(record.get("turn")))
        for item in pending:
            self._record_failure(
                item.attempt,
                failed_turn=item.turn,
                confirmed_turn=turn,
                reason="next_test_red",
            )
            if item.repeat_number == 1:
                message = (
                    "retry denied for the next identical edit: one condition must "
                    "change first: new evidence, a new hypothesis, or a different target."
                )[:MAX_MESSAGE_CHARS]
                refusal = RetryDecision(
                    decision=Decision.DENY,
                    turn=turn,
                    attempt=item.attempt,
                    repeat_number=2,
                    failed_turn=item.failed_turn,
                    reason="warned_repeat_failed_again",
                    message=message,
                    event="RETRY_DENY",
                    source_fingerprint=_record_fingerprint(record),
                    deferred=True,
                )
                self._decisions.append(refusal)
                self._counts[Decision.DENY.value] += 1
                self._counts["deferred_denials"] += 1
                self._emit(
                    "RETRY_DENY",
                    turn=turn,
                    attempt=item.attempt,
                    decision=Decision.DENY,
                    source_fingerprint=_record_fingerprint(record),
                    detail={
                        "deferred": True,
                        "applies_to": "next_identical_edit",
                        "repeat_number": 2,
                        "failed_turn": item.failed_turn,
                        "reason": "warned_repeat_failed_again",
                        "message": message,
                    },
                )

    def _allow(
        self,
        *,
        turn: int,
        attempt: AttemptFingerprint,
        source_fingerprint: str,
        reason: str,
        failed_turn: int | None = None,
    ) -> RetryDecision:
        event = (
            "RETRY_ALLOW_CHANGED_CONDITION"
            if reason == "material_condition_changed"
            else "RETRY_ALLOW"
        )
        result = RetryDecision(
            decision=Decision.ALLOW,
            turn=turn,
            attempt=attempt,
            failed_turn=failed_turn,
            reason=reason,
            message=(
                f"edit allowed: {reason.replace('_', ' ')}"
                if reason
                else "edit allowed"
            )[:MAX_MESSAGE_CHARS],
            event=event,
            source_fingerprint=source_fingerprint,
        )
        return self._remember_decision(result)

    def peek_predispatch(self, record: Record) -> RetryDecision:
        """Read-only forecast of the decision an edit WOULD receive, for a caller that
        must decide before dispatch, before the outcome (result_class) is known.

        Pure: never mutates ``_repeat_counts``, ``_pending``, ``_failed``, receipts, or
        the decision history (:meth:`observe`/:meth:`check` own that bookkeeping, using
        the same record once the real ``result_class`` is available). Calling this
        method any number of times has zero effect on later :meth:`observe` calls for
        the same or a different attempt -- it only reads existing failure history.

        WAVE 3 / D3: this is the wire a pre-dispatch caller uses to ask "would this
        action be denied?" before the editing tool runs. ``observe``/``check`` remain
        the authoritative post-receipt call that records bookkeeping.
        """
        if not isinstance(record, Mapping):
            raise TypeError("retry-gate receipt must be a mapping")
        rec = dict(record)
        turn = max(0, _int(rec.get("turn")))
        if _record_kind(rec) != "edit":
            return RetryDecision(decision=Decision.ALLOW, turn=turn,
                                  reason="non_edit_action_is_not_gated", event="RETRY_PEEK_BYPASS")
        try:
            attempt = self._attempt(rec)
        except (RetryGateError, TypeError, ValueError) as exc:
            return RetryDecision(
                decision=Decision.ALLOW, turn=turn,
                reason=f"malformed_edit_is_not_blocked:{type(exc).__name__}",
                event="RETRY_PEEK_INPUT_BYPASS")
        matching = self._matching_failure(attempt)
        if matching is None:
            return RetryDecision(decision=Decision.ALLOW, turn=turn, attempt=attempt,
                                  reason="no_matching_failed_attempt", event="RETRY_PEEK_ALLOW")
        repeat_key = (attempt.fingerprint, self._condition_epoch)
        # Forecast the repeat number WITHOUT persisting the increment -- observe()/check()
        # perform the real increment once the outcome is known.
        forecast_repeat_number = self._repeat_counts.get(repeat_key, 0) + 1
        if forecast_repeat_number == 1:
            return RetryDecision(
                decision=Decision.WARN, turn=turn, attempt=attempt,
                repeat_number=forecast_repeat_number, failed_turn=matching.failed_turn,
                reason="unchanged_condition_first_repeat_forecast",
                message="this is the same edit that failed at turn "
                        f"{matching.failed_turn} -- what changed?"[:MAX_MESSAGE_CHARS],
                event="RETRY_PEEK_WARN")
        return RetryDecision(
            decision=Decision.DENY, turn=turn, attempt=attempt,
            repeat_number=forecast_repeat_number, failed_turn=matching.failed_turn,
            reason="unchanged_condition_second_repeat_forecast",
            message="retry denied: one condition must change before this edit is "
                    "repeated: new evidence, a new hypothesis, or a different "
                    "target."[:MAX_MESSAGE_CHARS],
            event="RETRY_PEEK_DENY")

    def _evaluate_edit(
        self,
        record: Record,
        *,
        attempt: AttemptFingerprint,
        source_fingerprint: str,
    ) -> RetryDecision:
        turn = max(0, _int(record.get("turn")))
        matching = self._matching_failure(attempt)
        if matching is None:
            prior = self._prior_failure(attempt)
            if prior is not None:
                return self._allow(
                    turn=turn,
                    attempt=attempt,
                    source_fingerprint=source_fingerprint,
                    reason="material_condition_changed",
                    failed_turn=prior.failed_turn,
                )
            return self._allow(
                turn=turn,
                attempt=attempt,
                source_fingerprint=source_fingerprint,
                reason="no_matching_failed_attempt",
            )

        repeat_key = (attempt.fingerprint, self._condition_epoch)
        repeat_number = self._repeat_counts.get(repeat_key, 0) + 1
        self._repeat_counts[repeat_key] = repeat_number
        if repeat_number == 1:
            message = (
                f"this is the same edit that failed at turn {matching.failed_turn} "
                "— what changed?"
            )[:MAX_MESSAGE_CHARS]
            result = RetryDecision(
                decision=Decision.WARN,
                turn=turn,
                attempt=attempt,
                repeat_number=repeat_number,
                failed_turn=matching.failed_turn,
                reason="unchanged_condition_first_repeat",
                message=message,
                event="RETRY_WARN",
                source_fingerprint=source_fingerprint,
            )
            return self._remember_decision(result)

        message = (
            "retry denied: one condition must change before this edit is repeated: "
            "new evidence, a new hypothesis, or a different target."
        )[:MAX_MESSAGE_CHARS]
        result = RetryDecision(
            decision=Decision.DENY,
            turn=turn,
            attempt=attempt,
            repeat_number=repeat_number,
            failed_turn=matching.failed_turn,
            reason="unchanged_condition_second_repeat",
            message=message,
            event="RETRY_DENY",
            source_fingerprint=source_fingerprint,
        )
        return self._remember_decision(result)

    def _bypass(
        self,
        record: Record,
        *,
        source_fingerprint: str,
        reason: str,
        event: str = "RETRY_BYPASS",
    ) -> RetryDecision:
        turn = max(0, _int(record.get("turn")))
        result = RetryDecision(
            decision=Decision.ALLOW,
            turn=turn,
            reason=reason,
            message=reason.replace("_", " ")[:MAX_MESSAGE_CHARS],
            event=event,
            source_fingerprint=source_fingerprint,
        )
        self._counts["bypassed"] += 1
        self._emit(
            event,
            turn=turn,
            decision=Decision.ALLOW,
            source_fingerprint=source_fingerprint,
            detail={"reason": reason},
        )
        self._decisions.append(result)
        return result

    def observe(self, value: Record) -> RetryDecision:
        """Observe one receipt and return its retry decision.

        ``test`` and ``read`` records always return ``allow``.  Their receipts
        still update failure confirmation, read-set revisions, and material
        condition state, so bypassing their gate does not discard evidence.
        """

        if not isinstance(value, Mapping):
            raise TypeError("retry-gate receipt must be a mapping")
        record = dict(value)
        turn = max(0, _int(record.get("turn")))
        self._last_turn = max(self._last_turn, turn)
        source_fingerprint, is_new = self._note_source(record)
        self._note_condition_events(record)
        self._update_read_set(record, is_new_receipt=is_new)
        kind = _record_kind(record)

        if _is_test(record):
            self._confirm_pending(record)
            return self._bypass(
                record,
                source_fingerprint=source_fingerprint,
                reason="tests_are_never_gated",
            )

        if _is_read(record):
            return self._bypass(
                record,
                source_fingerprint=source_fingerprint,
                reason="reads_are_never_gated",
            )

        if kind != "edit":
            return self._bypass(
                record,
                source_fingerprint=source_fingerprint,
                reason="non_edit_action_is_not_gated",
            )

        try:
            attempt = self._attempt(record)
        except (RetryGateError, TypeError, ValueError) as exc:
            self._counts["input_bypasses"] += 1
            return self._bypass(
                record,
                source_fingerprint=source_fingerprint,
                reason=f"malformed_edit_is_not_blocked:{type(exc).__name__}",
                event="RETRY_INPUT_BYPASS",
            )

        decision = self._evaluate_edit(
            record,
            attempt=attempt,
            source_fingerprint=source_fingerprint,
        )
        if decision.decision is not Decision.DENY:
            if _is_immediate_edit_failure(record):
                self._record_failure(
                    attempt,
                    failed_turn=turn,
                    confirmed_turn=turn,
                    reason=f"edit_result:{_result_class(record)}",
                )
            elif _result_class(record) in {"APPLIED", "OK", "SUCCESS"}:
                self._pending.append(
                    _PendingAttempt(
                        attempt=attempt,
                        turn=turn,
                        repeat_number=decision.repeat_number,
                        failed_turn=decision.failed_turn,
                    )
                )
        return decision

    def check(self, value: Record) -> RetryDecision:
        """Compatibility alias for pre-dispatch callers."""

        return self.observe(value)

    inspect = observe
    record_receipt = observe
    on_receipt = observe

    def census(self, *, run_id: str = "retry-gate") -> RetryCensus:
        """Return bounded counters and persist a census receipt."""

        payload = {
            "organ_id": ORGAN_ID,
            "run_id": _text(run_id, limit=80),
            "allowed": self._counts[Decision.ALLOW.value],
            "warned": self._counts[Decision.WARN.value],
            "denied": self._counts[Decision.DENY.value],
            "bypassed": self._counts["bypassed"],
            "failed_attempts": self._counts["failed_attempts"],
            "condition_epoch": self._condition_epoch,
        }
        digest = stable_fingerprint(payload)
        self._emit(
            "RETRY_CENSUS_EMITTED",
            turn=self._last_turn,
            detail={**payload, "digest": digest},
        )
        return RetryCensus(
            allowed=self._counts[Decision.ALLOW.value],
            warned=self._counts[Decision.WARN.value],
            denied=self._counts[Decision.DENY.value],
            bypassed=self._counts["bypassed"],
            failed_attempts=self._counts["failed_attempts"],
            material_change_allows=self._counts["material_change_allows"],
            new_receipt_fingerprints=self._counts["new_receipt_fingerprints"],
            digest=digest,
            deferred_denials=self._counts["deferred_denials"],
            events=self.receipts,
        )

    def liveness_census(self, *, run_id: str = "retry-gate") -> object:
        """Run the repository-wide E397 census against this organ's receipts."""

        self.census(run_id=run_id)
        report = run_census(
            Evidence.from_records(receipts=self.receipts, run_id=run_id),
            specs=CENSUS_SPECS,
            run_id=run_id,
        )
        receipt = getattr(report, "receipt", None)
        if isinstance(receipt, Mapping):
            self._receipts.append(dict(receipt))
        return report


def replay_receipts(
    source: Iterable[Record] | str | Path,
    *,
    gate: RetryGate | None = None,
) -> RetryGate:
    """Replay JSONL or mapping-shaped receipts through a fresh gate."""

    target = gate or RetryGate()
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if isinstance(record, Mapping):
                    target.observe(record)
            return target
    else:
        records = source
    for record in records:
        if isinstance(record, Mapping):
            target.observe(record)
    return target


load_receipts = replay_receipts
RetryDecisionKind = Decision
Attempt = AttemptFingerprint


def render_retry_decision(decision: RetryDecision | None) -> str:
    """Render one bounded model-facing decision line."""

    if decision is None:
        return ""
    return decision.render()


__all__ = [
    "Attempt",
    "AttemptFingerprint",
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "Decision",
    "ORGAN_ID",
    "RetryCensus",
    "RetryDecision",
    "RetryDecisionKind",
    "RetryGate",
    "RetryGateError",
    "SCHEMA",
    "attempt_fingerprint",
    "load_receipts",
    "normalize_edit_digest",
    "render_retry_decision",
    "replay_receipts",
]
