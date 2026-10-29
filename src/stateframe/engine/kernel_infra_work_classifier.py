"""Host-owned terminal classification for infrastructure versus product work.

The classifier consumes runner-measured fields only.  It never searches model
text or test output for a magic phrase, so a model cannot turn a product
failure into an infrastructure exemption.  The feature is deliberately
opt-in at the integration boundary; callers may pass ``enabled_mode=True``
for a host-controlled evaluation of the pure mechanism.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias


SCHEMA = "kernel.infra_work_classifier/1"
MODE_ENV = "KERNEL_INFRA_WORK_CLASSIFIER"
INFRA_FAULT = "INFRA_FAULT"
PRODUCT_OUTCOME = "PRODUCT_OUTCOME"
FIRED_EVENT = "infra_work_classifier_fired"
PAIR_DROPPED_EVENT = "infra_work_pair_excluded"

JsonObject: TypeAlias = dict[str, object]
EventJournal: TypeAlias = Any


def mode() -> str:
    """Return the strict integration mode, defaulting to the old path."""

    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be 'off' or 'on'")
    return value


def enabled() -> bool:
    return mode() == "on"


@dataclass(frozen=True, slots=True)
class ProductOutcome:
    """A normal task/product result that remains scoreable."""

    value: str
    evidence: Mapping[str, object] = field(default_factory=dict)
    terminal_class: str = PRODUCT_OUTCOME
    charges_attempt: bool = True

    @property
    def is_infra_fault(self) -> bool:
        return False

    def as_dict(self) -> JsonObject:
        return {
            "terminal_class": self.terminal_class,
            "value": self.value,
            "evidence": _json_copy(self.evidence),
            "charges_attempt": self.charges_attempt,
        }


@dataclass(frozen=True, slots=True)
class InfraFault:
    """A host-observed environment failure excluded from product scoring."""

    kind: str
    evidence: Mapping[str, object]
    recovery: str
    charges_attempt: bool = False
    terminal_class: str = INFRA_FAULT

    @property
    def is_infra_fault(self) -> bool:
        return True

    @property
    def retry_allowed(self) -> bool:
        """A caller must pause/rebuild before another provider invocation."""

        return False

    def as_dict(self) -> JsonObject:
        return {
            "terminal_class": self.terminal_class,
            "kind": self.kind,
            "evidence": _json_copy(self.evidence),
            "recovery": self.recovery,
            "charges_attempt": self.charges_attempt,
            "retry_allowed": self.retry_allowed,
        }


TerminalOutcome: TypeAlias = ProductOutcome | InfraFault


@dataclass(frozen=True, slots=True)
class BatchReport:
    """The scoreable projection and separately reported infrastructure rows."""

    scorable_rows: tuple[Mapping[str, object], ...]
    excluded_rows: tuple[Mapping[str, object], ...]
    excluded_pairs: tuple[str, ...]
    infra_faults: tuple[InfraFault, ...]
    arm_rows: Mapping[str, tuple[Mapping[str, object], ...]]

    def as_dict(self) -> JsonObject:
        return {
            "schema": SCHEMA,
            "scorable_rows": [_json_copy(row) for row in self.scorable_rows],
            "excluded_rows": [_json_copy(row) for row in self.excluded_rows],
            "excluded_pairs": list(self.excluded_pairs),
            "infra_faults": [fault.as_dict() for fault in self.infra_faults],
            "arms": {
                arm: [_json_copy(row) for row in rows]
                for arm, rows in self.arm_rows.items()
            },
        }


def classify_terminal(
    signals: Mapping[str, object] | None,
    *,
    product_outcome: object = "FAILED",
    journal: EventJournal | None = None,
    enabled_mode: bool | None = None,
) -> TerminalOutcome:
    """Classify one terminal observation from host-measured signals.

    ``signals`` is intentionally a narrow, runner-owned record.  Values such
    as ``error`` or ``output_tail`` are never inspected.  When the mechanism
    is off, this returns the original product class and emits no event.
    """

    product = ProductOutcome(_product_value(product_outcome))
    active = enabled() if enabled_mode is None else bool(enabled_mode)
    if not active:
        return product

    observed = signals if isinstance(signals, Mapping) else {}
    fault = _fault_from_signals(observed)
    outcome: TerminalOutcome = fault if fault is not None else product
    _emit_fired(journal, outcome)
    return outcome


def prepare_batch(
    rows: Iterable[Mapping[str, object]],
    *,
    pair_key: str = "pair_id",
    arm_key: str = "arm",
    signal_key: str = "signals",
    outcome_key: str = "outcome",
    journal: EventJournal | None = None,
    enabled_mode: bool | None = None,
) -> BatchReport:
    """Classify rows and remove both arms of any pair with an infra fault.

    Rows without an explicit nested ``signals`` mapping are treated as the
    host record itself.  This keeps the kernel useful for small runners while
    the MC pack can provide a normalized nested record.
    """

    active = enabled() if enabled_mode is None else bool(enabled_mode)
    classified: list[tuple[JsonObject, TerminalOutcome, str]] = []
    pair_order: list[str] = []
    pair_faults: dict[str, list[InfraFault]] = {}

    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise TypeError("batch rows must be mappings")
        row = copy.deepcopy(dict(raw))
        pair = _pair_id(row, pair_key, index)
        if pair not in pair_order:
            pair_order.append(pair)
        nested = row.get(signal_key)
        signals = dict(nested) if isinstance(nested, Mapping) else row
        product_value = row.get(outcome_key, row.get("verdict", "FAILED"))
        outcome = classify_terminal(
            signals,
            product_outcome=product_value,
            journal=journal,
            enabled_mode=active,
        )
        classified.append((row, outcome, pair))
        if isinstance(outcome, InfraFault):
            pair_faults.setdefault(pair, []).append(outcome)

    excluded_pairs = tuple(pair for pair in pair_order if pair in pair_faults)
    excluded_pair_set = set(excluded_pairs)
    scorable = tuple(row for row, _outcome, pair in classified if pair not in excluded_pair_set)
    excluded = tuple(row for row, _outcome, pair in classified if pair in excluded_pair_set)
    faults = tuple(
        fault
        for pair in excluded_pairs
        for fault in pair_faults[pair]
    )
    arms: dict[str, list[Mapping[str, object]]] = {}
    for row in scorable:
        arm = str(row.get(arm_key, ""))
        arms.setdefault(arm, []).append(row)

    if active and journal is not None:
        for pair in excluded_pairs:
            pair_rows = [row for row, _outcome, row_pair in classified if row_pair == pair]
            _emit(
                journal,
                PAIR_DROPPED_EVENT,
                classifier_schema=SCHEMA,
                pair_id=pair,
                dropped_arms=sorted({str(row.get(arm_key, "")) for row in pair_rows}),
                fault_kinds=sorted({fault.kind for fault in pair_faults[pair]}),
            )

    return BatchReport(
        scorable_rows=scorable,
        excluded_rows=excluded,
        excluded_pairs=excluded_pairs,
        infra_faults=faults,
        arm_rows={arm: tuple(rows_for_arm) for arm, rows_for_arm in arms.items()},
    )


def _fault_from_signals(signals: Mapping[str, object]) -> InfraFault | None:
    """Apply the host signal precedence table."""

    pod_state = _pod_state(signals)
    pod_reason = _state_reason(pod_state)
    exit_code = _int_value(_first(signals, "exit_code", "returncode"))
    http_status = _http_status(signals)
    stage = _normalized(_first(signals, "stage", "phase", "operation"))
    runner_status = _normalized(_first(signals, "runner_status", "status"))

    watchdog = _truth(_first(signals, "pre_model_watchdog", "watchdog_triggered"))
    watchdog_phase = _normalized(_first(signals, "watchdog_phase", "watchdog_stage"))
    model_started = _first(signals, "model_started", "model_called")
    if watchdog and (
        watchdog_phase in {"", "PRE_MODEL", "BEFORE_MODEL", "DISPATCH"}
        or model_started is False
    ):
        evidence = _evidence(signals, "pre_model_watchdog", "watchdog_triggered", "watchdog_phase")
        return InfraFault("WATCHDOG_PRE_MODEL", evidence, "ENV_REBUILD")

    oom = _truth(_first(signals, "oom_killed", "oom")) or pod_reason in {
        "OOMKILLED",
        "OOM_KILLED",
        "OUTOFMEMORY",
        "OUT_OF_MEMORY",
    }
    if oom or exit_code == 137:
        evidence = _evidence(signals, "oom_killed", "oom", "pod_last_state", "pod_lastState", "exit_code", "returncode")
        return InfraFault("OOM_KILLED", evidence, "ENV_REBUILD")

    if _truth(_first(signals, "orphaned", "orphan")) or _normalized(
        _first(signals, "queue_state", "lifecycle_state")
    ) in {"ORPHANED", "LOST"}:
        evidence = _evidence(signals, "orphaned", "orphan", "queue_state", "lifecycle_state")
        return InfraFault("ORPHANED", evidence, "ENV_REBUILD")

    queue_timeout = _truth(_first(signals, "queue_timeout")) or (
        _truth(_first(signals, "timed_out", "timeout"))
        and stage in {"QUEUE", "DISPATCH", "PROVIDER_QUEUE"}
    )
    if queue_timeout:
        evidence = _evidence(signals, "queue_timeout", "timed_out", "timeout", "stage", "phase")
        return InfraFault("QUEUE_TIMEOUT", evidence, "ENV_REBUILD")

    if http_status == 402:
        return InfraFault("PROVIDER_PAYMENT", _http_evidence(signals), "PROVIDER_PAUSE")
    if http_status == 429:
        return InfraFault("PROVIDER_RATE_LIMIT", _http_evidence(signals), "PROVIDER_PAUSE")
    if http_status is not None and 500 <= http_status <= 599:
        operation = " ".join(
            part for part in (
                stage,
                _normalized(_first(signals, "runner", "service", "component")),
            )
            if part
        )
        kind = "CLONE_5XX" if "CLONE" in operation or "GIT" in operation else "PROVIDER_5XX"
        evidence = _http_evidence(signals)
        evidence.update(_evidence(signals, "stage", "operation"))
        return InfraFault(kind, evidence, "PROVIDER_PAUSE")

    # A red test is a product observation.  Exit 1 is therefore deliberately
    # not included in this host-fault table.
    if runner_status in {"INFRA_ERROR", "COLLECTION_ERROR"}:
        return InfraFault("RUNNER_INFRA", _evidence(signals, "runner_status", "status"), "ENV_REBUILD")
    if _truth(_first(signals, "timed_out", "timeout")) or runner_status == "TIMEOUT":
        return InfraFault("RUNNER_TIMEOUT", _evidence(signals, "timed_out", "timeout", "runner_status", "status"), "ENV_REBUILD")
    if exit_code in {126, 127}:
        return InfraFault("RUNNER_EXECUTION", _evidence(signals, "exit_code", "returncode"), "ENV_REBUILD")
    return None


def _emit_fired(journal: EventJournal | None, outcome: TerminalOutcome) -> None:
    if journal is None:
        return
    fields: JsonObject = {
        "classifier_schema": SCHEMA,
        "terminal_class": outcome.terminal_class,
        "charges_attempt": outcome.charges_attempt,
        "evidence": _json_copy(outcome.evidence),
    }
    if isinstance(outcome, InfraFault):
        fields.update({
            "fault_kind": outcome.kind,
            "recovery": outcome.recovery,
            "retry_allowed": outcome.retry_allowed,
        })
    _emit(journal, FIRED_EVENT, **fields)


def _emit(journal: EventJournal, kind: str, **fields: object) -> object:
    emitter = getattr(journal, "emit", None)
    if callable(emitter):
        return emitter(kind, **fields)
    if callable(journal):
        return journal(kind, **fields)
    raise TypeError("journal must expose emit(kind, **fields)")


def _product_value(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("verdict", "result", "behaviour", "status", "outcome"):
            if key in value:
                value = value[key]
                break
    rendered = " ".join(str(value or "").split()).upper()
    return rendered or "FAILED"


def _pair_id(row: Mapping[str, object], pair_key: str, index: int) -> str:
    for key in (pair_key, "pair_id", "pair", "run_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row:{index}"


def _first(signals: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in signals:
            return signals[key]
    return None


def _http_status(signals: Mapping[str, object]) -> int | None:
    value = _first(signals, "http_status", "status_code", "response_status")
    result = _int_value(value)
    if result is not None:
        return result
    for key in ("http", "response", "provider_response"):
        nested = signals.get(key)
        if isinstance(nested, Mapping):
            result = _int_value(_first(nested, "status", "status_code", "http_status"))
            if result is not None:
                return result
    return None


def _http_evidence(signals: Mapping[str, object]) -> JsonObject:
    evidence = _evidence(signals, "http_status", "status_code", "response_status")
    if evidence:
        return evidence
    for key in ("http", "response", "provider_response"):
        nested = signals.get(key)
        if isinstance(nested, Mapping):
            nested_evidence = _evidence(nested, "status", "status_code", "http_status")
            if nested_evidence:
                return {key: nested_evidence}
    return evidence


def _pod_state(signals: Mapping[str, object]) -> Mapping[str, object] | None:
    for key in ("pod_last_state", "pod_lastState", "last_state", "pod_state"):
        value = signals.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _state_reason(state: Mapping[str, object] | None) -> str:
    if state is None:
        return ""
    return _normalized(state.get("reason", state.get("Reason", state.get("state", ""))))


def _evidence(signals: Mapping[str, object], *keys: str) -> JsonObject:
    """Copy only selected runner fields into the auditable evidence capsule."""

    result: JsonObject = {}
    for key in keys:
        if key in signals and signals[key] is not None:
            result[key] = _json_copy(signals[key])
    return result


def _int_value(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truth(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return value is True


def _normalized(value: object) -> str:
    return " ".join(str(value or "").replace("-", "_").split()).upper()


def _json_copy(value: object) -> object:
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "BatchReport",
    "EventJournal",
    "FIRED_EVENT",
    "INFRA_FAULT",
    "InfraFault",
    "MODE_ENV",
    "PAIR_DROPPED_EVENT",
    "PRODUCT_OUTCOME",
    "ProductOutcome",
    "SCHEMA",
    "TerminalOutcome",
    "classify_terminal",
    "enabled",
    "mode",
    "prepare_batch",
]
