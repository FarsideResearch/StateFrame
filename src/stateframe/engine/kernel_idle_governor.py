"""Enforced liveness governor for collection-only repair actions.

The governor is intentionally small and receipt-backed.  It counts read-like
actions since the last productive receipt, warns once, then blocks repeated
reads and greps until an edit, test, or new contract fact lands.  It is an
action policy organ, not an execution layer: the dispatcher owns applying the
returned decision and placing the refusal in the next observation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from kernel_liveness import Evidence, OrganCensusSpec, TriState
from kernel_pivot import (
    _FAILURE_RESULTS,
    _GREEN_RESULTS,
    _NON_PRODUCT_RESULTS,
)


SCHEMA = "poc.kernel_idle_governor/1"
DEFAULT_N1 = 8
DEFAULT_N2 = 14
WARNING_TEMPLATE = (
    "{count} reads with zero novelty — next turn must edit, test, or state what fact "
    "you still need and why a read answers it"
)
REFUSAL = "read budget exhausted: edit, test, or abstain"

Record: TypeAlias = Mapping[str, object]
Action: TypeAlias = str | Record

COLLECTION_ACTIONS = frozenset({"read", "grep", "list", "collect", "collect_only"})
DENIABLE_ACTIONS = frozenset({"read", "grep"})
PRODUCT_ACTIONS = frozenset(
    {"edit", "test", "contract_fact", "new_contract_fact", "new_contract", "fact"}
)
PRODUCT_RESULT_CLASSES = _GREEN_RESULTS | _FAILURE_RESULTS


class IdleGovernorError(ValueError):
    """Raised when governor thresholds or receipts are malformed."""


class GovernorMode(StrEnum):
    """Current collection policy posture."""

    IDLE = "idle"
    GOVERNED = "governed"


class GovernorDecision(StrEnum):
    """Dispatcher decision for one action."""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"


def post_execution_observation(
        observation: str, decision: GovernorDecision, message: str, *,
        source_bytes_delivered: bool,
) -> str:
    """Never replace an already executed successful source result with a refusal."""
    if decision != GovernorDecision.DENY or source_bytes_delivered:
        return observation
    return "[kernel-governor] REFUSED: " + (
        message or "read budget exhausted — edit, test, or state the missing fact and abstain."
    )


Decision = GovernorDecision


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _normal_kind(value: object) -> str:
    return _text(value).lower().replace("-", "_").replace(" ", "_")


def _action_kind(action: Action) -> str:
    if isinstance(action, Mapping):
        return _normal_kind(action.get("kind", action.get("action_kind", action.get("type", ""))))
    return _normal_kind(action)


def _normal_path(value: object) -> str:
    path = _text(value).replace("\\", "/").lstrip("./")
    return re.sub(r"@\d+(?:-\d+)?$", "", path)


def _action_path(action: Action, explicit_path: object = None) -> str:
    if explicit_path is not None:
        path = _normal_path(explicit_path)
        if path:
            return path
    if not isinstance(action, Mapping):
        return ""
    for key in ("path", "file", "target_path", "file_path"):
        path = _normal_path(action.get(key))
        if path:
            return path
    intent = _text(action.get("intent"))
    match = re.search(r"[\w./-]+\.[A-Za-z0-9_]+(?:@\d+(?:-\d+)?)?", intent)
    return _normal_path(match.group(0)) if match else ""


def _result_class(action: Action, explicit_result: object = None) -> str:
    if explicit_result is not None:
        return _text(explicit_result).upper()
    if isinstance(action, Mapping):
        return _text(action.get("result_class", action.get("result", ""))).upper()
    return ""


def _fingerprint(action: Action) -> str:
    if isinstance(action, Mapping):
        supplied = _text(action.get("fingerprint", action.get("receipt_id", "")))
        if supplied:
            return supplied
        payload = {
            "kind": _action_kind(action),
            "path": _action_path(action),
            "result_class": _result_class(action),
            "intent": _text(action.get("intent")),
            "turn": action.get("turn", action.get("step", "")),
        }
    else:
        payload = {"kind": _action_kind(action)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_product_receipt(action: Action) -> bool:
    kind = _action_kind(action)
    if kind not in PRODUCT_ACTIONS:
        return False
    result = _result_class(action)
    if result in _NON_PRODUCT_RESULTS:
        return False
    if result in PRODUCT_RESULT_CLASSES:
        return True
    # Unknown classes on an explicit product action remain product attempts;
    # only the pivot organ's known non-product vocabulary blocks a reset.
    return True


def _is_non_product_action(action: Action) -> bool:
    """Recognize collection actions and product-shaped attempts that did not land."""

    kind = _action_kind(action)
    return kind in COLLECTION_ACTIONS or (
        kind in PRODUCT_ACTIONS and not _is_product_receipt(action)
    )


@dataclass(frozen=True, slots=True)
class IdleDecisionReceipt:
    """Append-only trace row for one policy decision or reset event."""

    event: str
    turn: int
    action_kind: str
    decision: GovernorDecision
    mode: GovernorMode
    consecutive_non_product: int
    path: str = ""
    result_class: str = ""
    fingerprint: str = ""
    novelty_escape: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "kind": "idle_governor",
            "organ": "idle_governor",
            "event": self.event,
            "receipt_id": "idle:" + hashlib.sha256(
                json.dumps(
                    {
                        "event": self.event,
                        "turn": self.turn,
                        "action_kind": self.action_kind,
                        "decision": self.decision.value,
                        "mode": self.mode.value,
                        "consecutive_non_product": self.consecutive_non_product,
                        "path": self.path,
                        "result_class": self.result_class,
                        "fingerprint": self.fingerprint,
                        "novelty_escape": self.novelty_escape,
                        "message": self.message,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24],
            "turn": self.turn,
            "action_kind": self.action_kind,
            "decision": self.decision.value,
            "mode": self.mode.value,
            "consecutive_non_product": self.consecutive_non_product,
            "path": self.path,
            "result_class": self.result_class,
            "fingerprint": self.fingerprint,
            "novelty_escape": self.novelty_escape,
            "message": self.message,
        }
        return payload


@dataclass(frozen=True, slots=True)
class IdleGovernorCensus:
    """Stable E397 counter shape for warnings, denials, and resets."""

    warnings: int
    denials: int
    resets: int

    def to_dict(self) -> dict[str, int]:
        return {
            "warnings": self.warnings,
            "denials": self.denials,
            "resets": self.resets,
        }


def _int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise IdleGovernorError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise IdleGovernorError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise IdleGovernorError(f"{name} must be a positive integer")
    return result


class IdleGovernor:
    """Receipt-backed action gate that enforces a bounded collection budget."""

    def __init__(self, *, n1: int = DEFAULT_N1, n2: int = DEFAULT_N2) -> None:
        self.n1 = _int(n1, "n1")
        self.n2 = _int(n2, "n2")
        if self.n2 <= self.n1:
            raise IdleGovernorError("n2 must exceed n1")
        self._mode = GovernorMode.IDLE
        self._consecutive_non_product = 0
        self._warning_emitted = False
        self._warnings = 0
        self._denials = 0
        self._resets = 0
        self._read_paths: set[str] = set()
        self._trace: list[dict[str, object]] = []
        self._last_receipt: IdleDecisionReceipt | None = None
        self._preflight: dict[str, IdleDecisionReceipt] = {}

    @property
    def mode(self) -> GovernorMode:
        return self._mode

    @property
    def governed(self) -> bool:
        return self._mode is GovernorMode.GOVERNED

    @property
    def consecutive_non_product(self) -> int:
        return self._consecutive_non_product

    @property
    def warnings(self) -> int:
        return self._warnings

    @property
    def denials(self) -> int:
        return self._denials

    @property
    def resets(self) -> int:
        return self._resets

    @property
    def last_warning(self) -> str | None:
        return next(
            (
                str(row["message"])
                for row in reversed(self._trace)
                if row["event"] == "IDLE_WARNING"
            ),
            None,
        )

    @property
    def last_refusal(self) -> str | None:
        return next(
            (
                str(row["message"])
                for row in reversed(self._trace)
                if row["event"] == "IDLE_DENIED"
            ),
            None,
        )

    @property
    def trace(self) -> tuple[Mapping[str, object], ...]:
        return tuple(dict(row) for row in self._trace)

    @property
    def receipts(self) -> tuple[Mapping[str, object], ...]:
        return self.trace

    @property
    def last_receipt(self) -> IdleDecisionReceipt | None:
        return self._last_receipt

    def should_allow(
        self,
        action_kind: Action,
        *,
        path: object = None,
        turn: int = 0,
        result_class: object = None,
    ) -> GovernorDecision:
        """Decide and trace one action; landed-product resets use observe_receipt."""

        action: Action = action_kind
        kind = _action_kind(action)
        action_path = _action_path(action, path)
        result = _result_class(action, result_class)
        decision_receipt = self._preflight.pop(_fingerprint(action), None)
        if decision_receipt is not None and isinstance(action, Mapping):
            self._last_receipt = decision_receipt
            return decision_receipt.decision

        decision, event, message, novelty_escape = self._decide(
            action, kind=kind, action_path=action_path, result=result
        )
        row = IdleDecisionReceipt(
            event=event,
            turn=self._turn(action, turn),
            action_kind=kind,
            decision=decision,
            mode=self._mode,
            consecutive_non_product=self._consecutive_non_product,
            path=action_path,
            result_class=result,
            fingerprint=_fingerprint(action),
            novelty_escape=novelty_escape,
            message=message,
        )
        self._last_receipt = row
        self._trace.append(row.to_dict())
        if isinstance(action, Mapping):
            self._preflight[row.fingerprint] = row
        return decision

    def _decide(
        self, action: Action, *, kind: str, action_path: str, result: str
    ) -> tuple[GovernorDecision, str, str, bool]:
        product_action = kind in PRODUCT_ACTIONS and (
            not isinstance(action, Mapping) or result not in _NON_PRODUCT_RESULTS
        )
        if product_action:
            return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", False
        if not _is_non_product_action(action):
            return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", False

        # A read is "novel" when its key (path, plus the delivered-content digest when
        # the caller supplies one) has never been DELIVERED. Only re-reads of already
        # delivered, unchanged bytes are budgeted: a first read of a path, a read of
        # changed bytes, or a different range is never refused. Exempt reads (paths the
        # request itself says are required but missing) are never refused either.
        # Keys are recorded only when the read is allowed, so a refusal cannot make
        # a never-seen path look already read (the r2X benchmark starvation).
        novelty_escape = False
        read_key = ""
        if kind == "read" and action_path:
            digest = _text(action.get("content_digest", "")) if isinstance(action, Mapping) else ""
            read_key = action_path + ("#" + digest if digest else "")
            novelty_escape = read_key not in self._read_paths
        exempt = bool(action.get("exempt")) if isinstance(action, Mapping) else False

        self._consecutive_non_product += 1
        if self._mode is GovernorMode.GOVERNED:
            if exempt and kind == "read":
                self._remember_read(read_key)
                return GovernorDecision.ALLOW, "IDLE_EXEMPT", "", novelty_escape
            if novelty_escape:
                self._remember_read(read_key)
                return GovernorDecision.ALLOW, "IDLE_NOVELTY_ESCAPE", "", True
            if kind in DENIABLE_ACTIONS:
                self._denials += 1
                return GovernorDecision.DENY, "IDLE_DENIED", REFUSAL, False
            return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", False

        if self._consecutive_non_product >= self.n2:
            self._mode = GovernorMode.GOVERNED
            if kind in DENIABLE_ACTIONS and not novelty_escape and not (exempt and kind == "read"):
                self._denials += 1
                return GovernorDecision.DENY, "IDLE_DENIED", REFUSAL, False
        self._remember_read(read_key)
        if self._consecutive_non_product >= self.n1 and not self._warning_emitted:
            self._warning_emitted = True
            self._warnings += 1
            warning = WARNING_TEMPLATE.format(count=self._consecutive_non_product)
            return GovernorDecision.WARN, "IDLE_WARNING", warning, False
        return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", novelty_escape

    def _remember_read(self, read_key: str) -> None:
        if read_key:
            self._read_paths.add(read_key)

    @staticmethod
    def _turn(action: Action, explicit_turn: int) -> int:
        raw = action.get("turn", action.get("step", explicit_turn)) if isinstance(action, Mapping) else explicit_turn
        try:
            turn = int(raw)
        except (TypeError, ValueError) as exc:
            raise IdleGovernorError("turn must be an integer") from exc
        if turn < 0:
            raise IdleGovernorError("turn must be non-negative")
        return turn

    def observe_receipt(self, receipt: Record) -> IdleDecisionReceipt:
        """Ingest one action receipt and reset only when product work landed."""

        self.should_allow(receipt)
        row = self._last_receipt
        if row is None:  # pragma: no cover - should_allow always emits a row
            raise IdleGovernorError("decision receipt was not emitted")
        if _is_product_receipt(receipt):
            self._reset(receipt, row.turn)
        return row

    observe = observe_receipt
    record_receipt = observe_receipt
    ingest_receipt = observe_receipt

    def _reset(self, receipt: Record, turn: int) -> None:
        self._consecutive_non_product = 0
        self._mode = GovernorMode.IDLE
        self._warning_emitted = False
        self._resets += 1
        row = IdleDecisionReceipt(
            event="IDLE_RESET",
            turn=turn,
            action_kind=_action_kind(receipt),
            decision=GovernorDecision.ALLOW,
            mode=self._mode,
            consecutive_non_product=0,
            path=_action_path(receipt),
            result_class=_result_class(receipt),
            fingerprint=_fingerprint(receipt),
            message="product receipt landed",
        )
        self._last_receipt = row
        self._trace.append(row.to_dict())

    def census(self) -> dict[str, int]:
        return IdleGovernorCensus(self._warnings, self._denials, self._resets).to_dict()

    def summary(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "mode": self._mode.value,
            "consecutive_non_product": self._consecutive_non_product,
            "n1": self.n1,
            "n2": self.n2,
            **self.census(),
        }


def _idle_governor_trigger(evidence: Evidence) -> TriState:
    event = _text(evidence.event.get("event")).upper()
    if event in CENSUS_SPEC.fire_events:
        return TriState.YES
    if event in CENSUS_SPEC.suppression_events:
        return TriState.NO
    return TriState.UNKNOWN if not evidence.complete else TriState.NO


CENSUS_SPEC = OrganCensusSpec(
    organ_id="idle_governor",
    trigger=_idle_governor_trigger,
    fire_events=frozenset({"IDLE_WARNING", "IDLE_DENIED", "IDLE_GOVERNED"}),
    suppression_events=frozenset({"IDLE_ALLOWED", "IDLE_NOVELTY_ESCAPE", "IDLE_RESET"}),
    dedup_events=frozenset(),
    challenge_once_events=frozenset({"IDLE_WARNING"}),
)
CENSUS_SPECS = (CENSUS_SPEC,)


def load_receipts(path: str | Path) -> tuple[Record, ...]:
    """Load valid JSON object rows from a receipt JSONL tape."""

    records: list[Record] = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            records.append(value)
    return tuple(records)


def replay_receipts(
    receipts: tuple[Record, ...] | list[Record], *, n1: int = DEFAULT_N1, n2: int = DEFAULT_N2
) -> IdleGovernor:
    """Replay a receipt tape through the same pre-dispatch policy path."""

    governor = IdleGovernor(n1=n1, n2=n2)
    for receipt in receipts:
        governor.observe_receipt(receipt)
    return governor


__all__ = [
    "CENSUS_SPEC",
    "CENSUS_SPECS",
    "COLLECTION_ACTIONS",
    "DEFAULT_N1",
    "DEFAULT_N2",
    "DENIABLE_ACTIONS",
    "Decision",
    "GovernorDecision",
    "GovernorMode",
    "IdleDecisionReceipt",
    "IdleGovernor",
    "IdleGovernorCensus",
    "IdleGovernorError",
    "PRODUCT_ACTIONS",
    "PRODUCT_RESULT_CLASSES",
    "REFUSAL",
    "SCHEMA",
    "WARNING_TEMPLATE",
    "load_receipts",
    "replay_receipts",
]
