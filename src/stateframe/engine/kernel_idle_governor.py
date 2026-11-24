"""Enforced liveness governor for collection-only repair actions.

The governor is intentionally small and receipt-backed. It counts repeated
read and search evidence since the last productive receipt, warns once, then
blocks repeated reads and greps until an edit, test, or new contract fact lands.
It is an action policy organ, not an execution layer: the dispatcher owns
applying the returned decision and placing the refusal in the next observation.
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


def _span_suffix(value: object) -> str:
    """The ``@start-end`` span of a read path, kept distinct from the bare path."""
    match = re.search(r"@\d+(?:-\d+)?$", _text(value).replace("\\", "/"))
    return match.group(0) if match else ""


def _result_class(action: Action, explicit_result: object = None) -> str:
    if explicit_result is not None:
        return _text(explicit_result).upper()
    if isinstance(action, Mapping):
        return _text(action.get("result_class", action.get("result", ""))).upper()
    return ""


_FAILED_RESULT_CLASSES = frozenset({
    "BLOCKED", "CANCELLED", "CANCELED", "DENIED", "ERROR", "EXCEPTION",
    "FAILED", "FAIL", "FAILURE", "MISSING", "REJECTED", "REFUSED", "STALE",
    "STALE_REVISION", "STALE_COORDINATE", "TIMEOUT", "TIMED_OUT", "UNAVAILABLE",
    "WITHHELD", "HIDDEN_PATH", "INVALID_ARGUMENT", "OUT_OF_RANGE",
    "SYMBOL_NOT_FOUND",
})


def _undelivered_action(action: Action, kind: str) -> bool:
    if not isinstance(action, Mapping):
        return False
    if action.get("undelivered") is True or action.get("refused") is True:
        return True
    if action.get("delivered") is False:
        return True
    if _result_class(action) in _FAILED_RESULT_CLASSES:
        return True
    if kind == "read":
        delivery = action.get("delivery")
        if not isinstance(delivery, Mapping):
            delivery = action
        if is_undelivered_read(delivery):
            return True
        outcome = _text(delivery.get("host_outcome")).upper()
        status = _text(delivery.get("delivery_status")).upper()
        return outcome in _FAILED_RESULT_CLASSES or status in _FAILED_RESULT_CLASSES
    return False


def _identity_key(action: Action, kind: str, action_path: str) -> str:
    """Return a stable identity for evidence already delivered to the model."""
    if not isinstance(action, Mapping):
        return ""
    explicit = action.get("evidence_identity")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if isinstance(explicit, Mapping) and explicit:
        value: object = dict(explicit)
    elif (kind == "read" and _text(action.get("result_set_digest", ""))
          and _text(action.get("page_digest", ""))):
        value = {
            "kind": "directory", "path": action_path,
            "result_set_digest": _text(action.get("result_set_digest")),
            "page_digest": _text(action.get("page_digest")),
        }
    elif kind == "read":
        delivery = action.get("delivery")
        if not isinstance(delivery, Mapping):
            delivery = action
        revision = delivery.get("revision", action.get("revision"))
        effective = delivery.get("effective_range", action.get("effective_range"))
        byte_start = delivery.get("byte_start", action.get("byte_start"))
        byte_end = delivery.get("byte_end", action.get("byte_end"))
        if isinstance(effective, Mapping):
            range_identity = {
                key: effective[key]
                for key in ("start_line", "end_line", "start_column", "end_column",
                            "byte_start", "byte_end")
                if effective.get(key) is not None
            }
        else:
            range_identity = {}
        if byte_start is not None or byte_end is not None:
            range_identity["byte_start"] = byte_start
            range_identity["byte_end"] = byte_end
        content_digest = _text(
            action.get("content_digest", delivery.get("requested_bytes_sha256", "")))
        if revision is not None or range_identity:
            value = {
                "kind": "read",
                "path": action_path,
                "revision": revision,
                "range": range_identity or {"span": _span_suffix(action.get("path"))},
            }
            if revision is None and content_digest:
                value["content_digest"] = content_digest
        else:
            digest = content_digest
            if not action_path and not _span_suffix(action.get("path")) and not digest:
                return ""
            value = {
                "kind": "read", "path": action_path,
                "span": _span_suffix(action.get("path")), "content_digest": digest,
            }
    elif kind in {"grep", "search"}:
        raw_query = action.get("query", action.get("intent", ""))
        query = raw_query if isinstance(raw_query, str) else str(raw_query or "")
        result_digest = _text(action.get("result_set_digest", action.get("result_digest", "")))
        page_digest = _text(action.get("page_digest", ""))
        page = action.get("grep_page")
        if isinstance(page, Mapping):
            page_digest = page_digest or _text(page.get("sha256", ""))
        if not query.strip() or not result_digest:
            return ""
        value = {
            "kind": "grep", "query": query, "result_set_digest": result_digest,
            "page_digest": page_digest,
        }
    else:
        return ""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_undelivered_read(delivery: object) -> bool:
    """True when a read's requested bytes never arrived (budget miss, withheld, or empty page).

    Proof-backed reuse stubs (REUSED_PROVEN) and non-delivery failures are not undelivered.
    Shared by the read budget and the no-progress watch so both agree on one definition.
    """
    receipt: Mapping[str, object] = delivery if isinstance(delivery, Mapping) else {}
    status = receipt.get("delivery_status")
    count = receipt.get("requested_bytes_delivered", 0)
    if status == "WITHHELD":
        return True
    return status == "DELIVERED" and not (type(count) is int and count > 0)


def read_governor_action(
        path: str, *, turn: int, read_ok: bool, delivery: Mapping[str, object] | None,
        deliver_requested: bool, observation: str,
) -> dict[str, object]:
    """Governor action for one read, built from the executor's delivery receipt.

    Only repeated evidence is budgeted. A read whose requested bytes never arrived
    (budget miss, size limit, withheld or empty page) is ``undelivered``: it spends
    no budget and is never refused, because the model cannot act on bytes it never
    received. A proof-backed reuse stub keeps its path-less shape so repeats count.
    """
    action: dict[str, object] = {"kind": "read", "path": path, "turn": turn}
    receipt: Mapping[str, object] = delivery if isinstance(delivery, Mapping) else {}
    if _undelivered_action({"kind": "read", "delivery": receipt}, "read"):
        action["undelivered"] = True
        return action
    if receipt.get("schema") == "navigation-page/1":
        result_digest = _text(receipt.get("result_set_digest", ""))
        page_digest = _text(receipt.get("page_evidence_digest", ""))
        if result_digest and page_digest:
            action["result_set_digest"] = result_digest
            action["page_digest"] = page_digest
            listing_path = _action_path(action) or _normal_path(receipt.get("path")) or "."
            _attach_evidence_identity(action, "read", listing_path)
            return action
    if not read_ok:
        action["undelivered"] = True
        return action
    if not deliver_requested:
        action["content_digest"] = hashlib.sha256(
            observation.encode("utf-8", "replace")).hexdigest()[:32]
        if is_undelivered_read(receipt):
            action["undelivered"] = True
        else:
            _copy_read_coordinates(action, receipt)
            _attach_evidence_identity(action, "read", _action_path(action))
        return action
    status = receipt.get("delivery_status")
    count = receipt.get("requested_bytes_delivered", 0)
    has_bytes = type(count) is int and count > 0
    if status == "DELIVERED" and has_bytes:
        _copy_read_coordinates(action, receipt)
        action["content_digest"] = str(receipt.get("requested_bytes_sha256") or "")
        _attach_evidence_identity(action, "read", _action_path(action))
    elif is_undelivered_read(receipt):
        action["undelivered"] = True
    else:
        action["path"] = ""
    return action


def _copy_read_coordinates(action: dict[str, object], delivery: Mapping[str, object]) -> None:
    """Carry the source revision and exact returned range into the policy receipt."""
    for field in ("revision", "byte_start", "byte_end", "effective_range"):
        value = delivery.get(field)
        if value is not None:
            action[field] = (
                dict(value) if field == "effective_range" and isinstance(value, Mapping)
                else value
            )


def _attach_evidence_identity(action: dict[str, object], kind: str, action_path: str) -> None:
    identity = _identity_key(action, kind, action_path)
    if identity:
        action["evidence_identity"] = identity


def grep_governor_action(
        query: str, *, turn: int, page: object, delivered: bool,
) -> dict[str, object]:
    """Build a grep action from the exact result page returned to the model."""
    action: dict[str, object] = {"kind": "grep", "query": query, "turn": turn}
    if not delivered or page is None:
        action["undelivered"] = True
        return action
    text = getattr(page, "text", None)
    metadata = getattr(page, "metadata", None)
    if not isinstance(text, str) or not isinstance(metadata, Mapping):
        action["undelivered"] = True
        return action
    result_digest = _text(metadata.get("result_set_digest"))
    page_digest = _text(getattr(page, "evidence_digest", ""))
    if not page_digest:
        page_digest = _text(metadata.get("page_digest"))
    if not page_digest:
        page_digest = "sha256:" + hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    if not result_digest:
        result_digest = page_digest
    action["result_set_digest"] = result_digest
    action["page_digest"] = page_digest
    action["evidence_identity"] = _identity_key(action, "grep", "")
    return action


def _fingerprint(action: Action) -> str:
    if isinstance(action, Mapping):
        supplied = _text(action.get("fingerprint", ""))
        if supplied:
            return supplied
        kind = _action_kind(action)
        payload = {
            "kind": kind,
            "path": _action_path(action),
            "result_class": _result_class(action),
            "intent": _text(action.get("intent", action.get("query", ""))),
            "evidence_identity": _identity_key(action, kind, _action_path(action)),
            "turn": action.get("turn", action.get("step", "")),
        }
        # A tool receipt ID correlates this action with a result, but it is not a
        # sufficient cache key on its own: providers may reuse IDs across calls.
        receipt_id = _text(action.get("receipt_id", ""))
        if receipt_id:
            payload["receipt_id"] = receipt_id
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
    evidence_identity: str = ""
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
                        "evidence_identity": self.evidence_identity,
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
            "evidence_identity": self.evidence_identity,
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
        self._recovery_reads = 0
        self._undelivered_reads = 0
        self._evidence_identities: set[str] = set()
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
    def recovery_reads(self) -> int:
        """Reads of content not currently shown; exempt from the budget."""
        return self._recovery_reads

    @property
    def undelivered_reads(self) -> int:
        """Reads whose requested bytes never arrived; spend no budget, never refused."""
        return self._undelivered_reads

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

        decision, event, message, novelty_escape, evidence_identity = self._decide(
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
            evidence_identity=evidence_identity,
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
    ) -> tuple[GovernorDecision, str, str, bool, str]:
        product_action = kind in PRODUCT_ACTIONS and (
            not isinstance(action, Mapping) or result not in _NON_PRODUCT_RESULTS
        )
        if product_action:
            return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", False, ""
        if not _is_non_product_action(action):
            return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", False, ""

        # The model received none of the requested bytes (budget miss, size limit, empty
        # page): nothing was re-read, so it neither counts toward the budget nor marks
        # the path delivered. The receipt stays visible under its own event.
        if kind in DENIABLE_ACTIONS and _undelivered_action(action, kind):
            if kind == "read":
                self._undelivered_reads += 1
            return GovernorDecision.ALLOW, "IDLE_UNDELIVERED", "", False, ""

        evidence_identity = _identity_key(action, kind, action_path)
        # A read of a path/span that is NOT currently in the request is recovery, not
        # waste: the model cannot act on bytes it cannot see. It is never refused under
        # any budget state, never spends the budget, and is counted separately.
        # ``visible`` is None when the caller could not determine visibility (legacy
        # behavior applies); only an explicit False selects recovery.
        if (kind == "read" and isinstance(action, Mapping)
                and action.get("visible") is False):
            self._recovery_reads += 1
            self._remember_evidence(evidence_identity)
            return GovernorDecision.ALLOW, "IDLE_RECOVERY", "", False, evidence_identity

        # Novelty is based on delivered evidence identity, not action volume. A read
        # identity includes its source revision and exact returned range; grep includes
        # its query, result set, and page. First delivery of evidence is always free.
        # Only repeats of the same evidence advance the idle budget.
        if evidence_identity and evidence_identity not in self._evidence_identities:
            self._remember_evidence(evidence_identity)
            return (GovernorDecision.ALLOW, "IDLE_NOVELTY_ESCAPE", "", True,
                    evidence_identity)
        exempt = bool(action.get("exempt")) if isinstance(action, Mapping) else False

        self._consecutive_non_product += 1
        if self._mode is GovernorMode.GOVERNED:
            if exempt and kind == "read":
                self._remember_evidence(evidence_identity)
                return GovernorDecision.ALLOW, "IDLE_EXEMPT", "", False, evidence_identity
            if kind in DENIABLE_ACTIONS:
                self._denials += 1
                return GovernorDecision.DENY, "IDLE_DENIED", REFUSAL, False, evidence_identity
            return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", False, evidence_identity

        if self._consecutive_non_product >= self.n2:
            self._mode = GovernorMode.GOVERNED
            if kind in DENIABLE_ACTIONS and not (exempt and kind == "read"):
                self._denials += 1
                return GovernorDecision.DENY, "IDLE_DENIED", REFUSAL, False, evidence_identity
        if self._consecutive_non_product >= self.n1 and not self._warning_emitted:
            self._warning_emitted = True
            self._warnings += 1
            warning = WARNING_TEMPLATE.format(count=self._consecutive_non_product)
            return GovernorDecision.WARN, "IDLE_WARNING", warning, False, evidence_identity
        return GovernorDecision.ALLOW, "IDLE_ALLOWED", "", False, evidence_identity

    def _remember_evidence(self, evidence_identity: str) -> None:
        if evidence_identity:
            self._evidence_identities.add(evidence_identity)

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
            **({"recovery_reads": self._recovery_reads} if self._recovery_reads else {}),
            **({"undelivered_reads": self._undelivered_reads} if self._undelivered_reads else {}),
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
    suppression_events=frozenset({"IDLE_ALLOWED", "IDLE_NOVELTY_ESCAPE", "IDLE_RESET",
                                "IDLE_RECOVERY", "IDLE_UNDELIVERED"}),
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
        # Live preflight decisions are cached until their matching action receipt
        # arrives. A persisted receipt tape already contains completed actions, so
        # each row must make a fresh decision even when a batch repeats one wire
        # fingerprint within the same turn.
        governor._preflight.clear()
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
    "is_undelivered_read",
    "load_receipts",
    "grep_governor_action",
    "read_governor_action",
    "replay_receipts",
]
