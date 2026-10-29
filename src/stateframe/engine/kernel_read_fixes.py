"""Opt-in repairs for repeated current-source reads and native-batch budgets.

The defaults intentionally describe the pre-fix behavior.  Callers must opt in
to each mechanism independently so an OFF run can retain its historical wire
bytes.
"""
from __future__ import annotations

from kernel_step_limits import UNBOUNDED_GUIDANCE, limit_text, reached, remaining

import os
import time
from dataclasses import dataclass
from collections.abc import Iterable, Mapping


READ_RECEIPT_ENV = "STATEFRAME_READ_RECEIPT"
SOURCE_LABELS_ENV = "STATEFRAME_SOURCE_LABELS"
BUDGET_UNIT_ENV = "STATEFRAME_BUDGET_UNIT"

# Label contract (lane_common.md): every model-visible source-block label, plain
# and legacy.  Producers render through ``source_header``; every consumer that
# recognises a source block must accept ALL of these (see
# probes/test_source_label_contract.py), or it fails silently in one label mode.
PLAIN_SOURCE_LABEL = "Current source"
LEGACY_SOURCE_LABELS = ("Working source", "Retained source", "Observed source", "Repair source")
SOURCE_BLOCK_LABELS = (PLAIN_SOURCE_LABEL, *LEGACY_SOURCE_LABELS)
SOURCE_BLOCK_PREFIXES = (*tuple(f"[{label}:" for label in SOURCE_BLOCK_LABELS),
                         "[Paged source:")


def read_receipt_mode() -> str:
    value = os.environ.get(READ_RECEIPT_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{READ_RECEIPT_ENV} must be off or on")
    return value


def read_receipt_enabled() -> bool:
    return read_receipt_mode() == "on"


def source_labels_mode() -> str:
    value = os.environ.get(SOURCE_LABELS_ENV, "off")
    # ``legacy`` is accepted as a readable alias for OFF because it is useful
    # in replay fixtures and says what the byte-preserving arm means.
    if value == "legacy":
        value = "off"
    if value not in {"off", "plain"}:
        raise ValueError(f"{SOURCE_LABELS_ENV} must be off, legacy, or plain")
    return value


def plain_source_labels_enabled() -> bool:
    return source_labels_mode() == "plain"


def budget_unit_mode() -> str:
    value = os.environ.get(BUDGET_UNIT_ENV, "decision")
    if value == "off":
        value = "decision"
    if value not in {"decision", "response"}:
        raise ValueError(f"{BUDGET_UNIT_ENV} must be off, decision, or response")
    return value


def normalize_path(path: object) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def source_header(path: object, *, role: str = "working") -> str:
    """Return the exact model-visible header for a source block."""
    relative = normalize_path(path)
    if plain_source_labels_enabled():
        return f"[Current source: {relative}]"
    labels = {
        "working": "Working source",
        "current": "Working source",
        "retained": "Retained source",
        "observed": "Observed source",
        "repair": "Repair source",
    }
    return f"[{labels.get(role, role)}: {relative}]"


def source_header_variants(path: object) -> tuple[str, ...]:
    """Every header spelling a consumer must accept for ``path`` (plain first)."""
    relative = normalize_path(path)
    return tuple(f"[{label}: {relative}]" for label in SOURCE_BLOCK_LABELS)


def short_revision(revision: object) -> str:
    value = str(revision or "UNKNOWN")
    if len(value) <= 24:
        return value
    if value.startswith("sha256:"):
        return value[:19] + "..."
    return value[:21] + "..."


def read_unchanged_receipt(*, path: object, revision: object,
                           header: str) -> dict[str, object]:
    """Build a compact, typed receipt without repeating source bytes."""
    relative = normalize_path(path)
    full_header = str(header)
    message = (
        f"READ_UNCHANGED path={relative} revision={short_revision(revision)} "
        f"already_in_frame_as=\"{full_header}\"\n"
        f"Full current source is in the frame under {full_header}."
    )
    return {
        "schema": "stateframe-read-receipt/1",
        "status": "READ_UNCHANGED",
        "result_class": "ALREADY_PRESENT",
        "path": relative,
        "revision": revision,
        "already_in_frame_as": full_header,
        "message": message,
    }


def working_set_receipt(*, group_id: object, revision: object,
                        status: str = "WORKING_SET_DELIVERED") -> dict[str, object]:
    """Build the shared compact receipt for one working-set delivery."""
    ident = str(group_id or "")
    relative = normalize_path(f"workset:{ident}")
    header = f"[Working set: {ident}]"
    return {
        "schema": "stateframe-read-receipt/1",
        "status": str(status),
        "result_class": "WORKSET_CONTEXT",
        "path": relative,
        "revision": revision,
        "already_in_frame_as": header,
        "message": (
            f"Working-set context for {ident} was resolved from the current "
            "artifact snapshot."
        ),
    }


def complete_current_source_evidence(row: object) -> bool:
    """Recognize a complete current full-file row in either read schema."""
    if not isinstance(row, Mapping):
        try:
            row = dict(row)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    if not isinstance(row.get("path"), str) or not row.get("path"):
        return False
    if row.get("revision") is None and row.get("file_revision") is None:
        return False
    if not isinstance(row.get("content"), str):
        return False
    if row.get("freshness") not in (None, "current"):
        return False
    if row.get("start_line", 1) != 1 or row.get("start_column", 0) != 0:
        return False
    for key in ("omitted_chars", "omitted_chars_in_selection"):
        if key in row and row.get(key) not in (None, 0):
            return False
    for key in ("more_content_available", "more_lines_available"):
        if row.get(key) is True:
            return False
    total = row.get("total_lines")
    end = row.get("selected_end_line", row.get("end_line"))
    if total is not None and end is not None and total != end:
        return False
    return True


@dataclass
class ResponseBudget:
    """Shared accounting for history and frame arms.

    In response mode one provider response consumes one budget unit.  Tool
    calls remain separately counted. A finite response allocation caps them at
    four times that allocation; None removes the derived lifetime cap as well.
    """

    limit: int | None
    mode: str = "decision"
    arm: str = "shared"
    journal: object | None = None
    tool_multiplier: int = 4
    responses_used: int = 0
    tool_calls_used: int = 0
    _cap_reported: bool = False
    _response_reported: bool = False
    automatic_check_limit: int | None = None
    terminal_reserve_limit: int = 2
    output_reservation_tokens: int = 4000
    sandbox_deadline_monotonic: float | None = None
    automatic_checks_used: int = 0
    terminal_reserves_used: int = 0
    _deadline_reported: bool = False

    def __post_init__(self) -> None:
        # Generic zero-fuel ledgers remain exhausted; the runner normalizes
        # REACT_STEPS=0/unbounded to None before constructing this ledger.
        if self.limit is not None and (type(self.limit) is not int or self.limit < 0):
            raise ValueError("response budget limit must be a non-negative integer")
        if self.mode not in {"decision", "response"}:
            raise ValueError("budget mode must be decision or response")
        if type(self.tool_multiplier) is not int or self.tool_multiplier < 1:
            raise ValueError("tool multiplier must be a positive integer")
        if self.arm not in {"shared", "history", "frame"}:
            raise ValueError("budget arm must be shared, history, or frame")
        if self.automatic_check_limit is None:
            self.automatic_check_limit = self.limit
        if self.automatic_check_limit is not None and (
                type(self.automatic_check_limit) is not int or self.automatic_check_limit < 0):
            raise ValueError("automatic check limit must be a non-negative integer")
        if type(self.terminal_reserve_limit) is not int or self.terminal_reserve_limit < 0:
            raise ValueError("terminal reserve limit must be a non-negative integer")
        if type(self.output_reservation_tokens) is not int or self.output_reservation_tokens < 0:
            raise ValueError("output allowance must be a non-negative integer")
        if self.sandbox_deadline_monotonic is not None:
            try:
                self.sandbox_deadline_monotonic = float(self.sandbox_deadline_monotonic)
            except (TypeError, ValueError) as error:
                raise ValueError("sandbox deadline must be numeric") from error

    @property
    def tool_call_cap(self) -> int | None:
        return (self.limit * self.tool_multiplier
                if self.mode == "response" and self.limit is not None else None)

    @property
    def steps_used(self) -> int:
        return self.responses_used if self.mode == "response" else self.tool_calls_used

    @property
    def exhausted(self) -> bool:
        return self.mode == "response" and reached(self.limit, self.responses_used)

    def _emit(self, kind: str, **fields: object) -> None:
        emitter = getattr(self.journal, "emit", None)
        if callable(emitter):
            emitter(kind, budget_unit=self.mode, budget_arm=self.arm,
                    response_budget=self.limit, tool_call_cap=self.tool_call_cap,
                    responses_used=self.responses_used,
                    tool_calls_used=self.tool_calls_used, **fields)

    def can_start_response(self) -> bool:
        if self._sandbox_seconds_remaining() is not None and self._sandbox_seconds_remaining() <= 0:
            if not self._deadline_reported:
                self._deadline_reported = True
                self._emit("continuation_deadline_exhausted", reason="sandbox_deadline")
            return False
        if self.mode != "response":
            return True
        if not reached(self.limit, self.responses_used):
            return True
        if not self._response_reported:
            self._response_reported = True
            self._emit("response_budget_exhausted", reason="response_budget_limit")
        return False

    def charge_response(self, *, step: int | None = None) -> bool:
        if not self.can_start_response():
            return False
        if self.mode == "response":
            self.responses_used += 1
            self._emit("budget_unit_charged", unit="response", step=step)
        return True

    def can_start_tool_call(self) -> bool:
        if self._sandbox_seconds_remaining() is not None and self._sandbox_seconds_remaining() <= 0:
            if not self._deadline_reported:
                self._deadline_reported = True
                self._emit("continuation_deadline_exhausted", reason="sandbox_deadline")
            return False
        cap = self.tool_call_cap
        if cap is None or self.tool_calls_used < cap:
            return True
        if not self._cap_reported:
            self._cap_reported = True
            self._emit("tool_call_cap_hit", reason="response_budget_tool_cap")
        return False

    def register_tool_call(self, *, step: int | None = None,
                           tool: str | None = None) -> bool:
        if not self.can_start_tool_call():
            return False
        self.tool_calls_used += 1
        if self.mode == "response":
            self._emit("tool_call_counted", step=step, tool=tool)
        return True

    def record_response(self, *, tool_calls: int = 0,
                        step: int | None = None) -> bool:
        """Test/replay helper: charge one response and its selected calls."""
        if type(tool_calls) is not int or tool_calls < 0:
            raise ValueError("tool_calls must be a nonnegative integer")
        if not self.charge_response(step=step):
            return False
        for index in range(tool_calls):
            if not self.register_tool_call(step=step, tool=f"batch[{index}]"):
                return False
        return True

    @property
    def model_decisions_used(self) -> int:
        return self.responses_used if self.mode == "response" else self.tool_calls_used

    @property
    def model_decisions_remaining(self) -> int | None:
        return remaining(self.limit, self.model_decisions_used)

    @property
    def tool_attempt_limit(self) -> int | None:
        return self.tool_call_cap if self.tool_call_cap is not None else self.limit

    @property
    def tool_attempts_remaining(self) -> int | None:
        return remaining(self.tool_attempt_limit, self.tool_calls_used)

    @property
    def automatic_checks_remaining(self) -> int | None:
        return remaining(self.automatic_check_limit, self.automatic_checks_used)

    @property
    def terminal_reserves_remaining(self) -> int:
        return max(0, self.terminal_reserve_limit - self.terminal_reserves_used)

    def _sandbox_seconds_remaining(self, now: float | None = None) -> float | None:
        if self.sandbox_deadline_monotonic is None:
            return None
        current = time.monotonic() if now is None else float(now)
        return max(0.0, self.sandbox_deadline_monotonic - current)

    def charge_automatic_check(self, *, selector: str | None = None) -> bool:
        if self._sandbox_seconds_remaining() is not None and self._sandbox_seconds_remaining() <= 0:
            if not self._deadline_reported:
                self._deadline_reported = True
                self._emit("continuation_deadline_exhausted", reason="sandbox_deadline")
            return False
        if self.automatic_checks_remaining == 0:
            self._emit("automatic_check_cap_hit", reason="continuation_automatic_check_limit",
                       selector=selector)
            return False
        self.automatic_checks_used += 1
        self._emit("continuation_unit_charged", unit="automatic_check", selector=selector)
        return True

    def reserve_terminal(self, *, reason: str = "completion_owner") -> bool:
        if self._sandbox_seconds_remaining() is not None and self._sandbox_seconds_remaining() <= 0:
            if not self._deadline_reported:
                self._deadline_reported = True
                self._emit("continuation_deadline_exhausted", reason="sandbox_deadline")
            return False
        if self.terminal_reserves_remaining <= 0:
            self._emit("terminal_reserve_exhausted", reason=reason)
            return False
        self.terminal_reserves_used += 1
        self._emit("continuation_unit_charged", unit="terminal_reserve", reason=reason)
        return True

    def qualify_admission(self, *, model_decisions: int = 1,
                          tool_attempts: int = 0,
                          automatic_checks: int = 0,
                          terminal_reserves: int = 0,
                          output_tokens: int | None = None) -> bool:
        """Check every requested continuation unit against this single object."""
        if any(type(value) is not int or value < 0 for value in (
                model_decisions, tool_attempts, automatic_checks, terminal_reserves)):
            raise ValueError("continuation admission counts must be non-negative integers")
        if self.model_decisions_remaining is not None and self.model_decisions_remaining < model_decisions:
            return False
        if self.tool_attempts_remaining is not None and self.tool_attempts_remaining < tool_attempts:
            return False
        if self.automatic_checks_remaining is not None and self.automatic_checks_remaining < automatic_checks:
            return False
        if self.terminal_reserves_remaining < terminal_reserves:
            return False
        if output_tokens is not None and (
                type(output_tokens) is not int or output_tokens < 0):
            raise ValueError("output token admission must be a non-negative integer")
        if output_tokens is not None and output_tokens > self.output_reservation_tokens:
            return False
        deadline = self._sandbox_seconds_remaining()
        return deadline is None or deadline > 0

    def continuation_snapshot(self) -> dict[str, object]:
        return {
            "schema": "stateframe-continuation-limits/1",
            "model_decision_limit": self.limit,
            "model_decisions_used": self.model_decisions_used,
            "model_decisions_remaining": self.model_decisions_remaining,
            "tool_attempt_limit": self.tool_attempt_limit,
            "tool_attempts_used": self.tool_calls_used,
            "tool_attempts_remaining": self.tool_attempts_remaining,
            "automatic_check_limit": self.automatic_check_limit,
            "automatic_checks_used": self.automatic_checks_used,
            "automatic_checks_remaining": self.automatic_checks_remaining,
            "terminal_reserve_limit": self.terminal_reserve_limit,
            "terminal_reserves_used": self.terminal_reserves_used,
            "terminal_reserves_remaining": self.terminal_reserves_remaining,
            "output_reservation_tokens": self.output_reservation_tokens,
            "sandbox_deadline_monotonic": self.sandbox_deadline_monotonic,
            "sandbox_seconds_remaining": self._sandbox_seconds_remaining(),
        }

    def render_continuation_limits(self) -> str:
        snapshot = self.continuation_snapshot()
        seconds = snapshot["sandbox_seconds_remaining"]
        deadline = _time_left_text(self, seconds)
        admission = self.qualify_admission(model_decisions=1, tool_attempts=1,
                                           output_tokens=self.output_reservation_tokens)
        return (
            "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
            f"model_decisions_remaining={limit_text(snapshot['model_decisions_remaining'])}; "
            f"tool_attempts_remaining={limit_text(snapshot['tool_attempts_remaining'])}; "
            f"automatic_checks_remaining={limit_text(snapshot['automatic_checks_remaining'])}; "
            f"terminal_reserves_remaining={snapshot['terminal_reserves_remaining']}; "
            f"output_reservation_tokens={snapshot['output_reservation_tokens']}; "
            f"sandbox_seconds_remaining={deadline}; "
            f"next_action_admission={'allowed' if admission else 'denied'}; "
            "cap_stop_reasons=model_decisions|tool_attempts|automatic_checks|"
            "terminal_reserve|output_reservation|sandbox_deadline"
            + ("; " + UNBOUNDED_GUIDANCE if self.limit is None else "")
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "payload_schema": "stateframe-budget/1",
            "unit": self.mode,
            "arm": self.arm,
            "response_budget": self.limit,
            "responses_used": self.responses_used,
            "tool_calls_used": self.tool_calls_used,
            "tool_call_cap": self.tool_call_cap,
            "steps_used": self.steps_used,
        }



def _time_left_text(limits: object, seconds: object) -> str:
    """Seconds left plus plain minutes and turns at the measured pace.

    The pace clock starts at the first render; pace = elapsed / decisions used so far.
    """
    if seconds is None:
        return "unbounded"
    left = max(0.0, float(seconds))
    now = time.monotonic()
    start = getattr(limits, "_time_left_clock_start", None)
    if not isinstance(start, float):
        start = now
        try:
            setattr(limits, "_time_left_clock_start", start)
            setattr(limits, "_time_left_decisions_at_start", int(getattr(limits, "model_decisions_used", 0) or 0))
        except (AttributeError, TypeError):
            pass
    used = int(getattr(limits, "model_decisions_used", 0) or 0) - int(
        getattr(limits, "_time_left_decisions_at_start", 0) or 0)
    text = f"{left:.1f} (about {int(round(left / 60))} minutes left"
    if used > 0 and now > start:
        pace = (now - start) / used
        if pace > 0:
            text += f", ~{int(left // pace)} turns at your pace"
    return text + ")"


@dataclass
class ContinuationLimits(ResponseBudget):
    """One funded continuation ledger for decisions, tools, checks, and terminal work."""

    continuation_limits_schema = "stateframe-continuation-limits/1"
    continuation_model_decisions_used: int = 0
    terminal_reserves_consumed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        for name, value in (
                ("continuation_model_decisions_used", self.continuation_model_decisions_used),
                ("terminal_reserves_consumed", self.terminal_reserves_consumed)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def model_decisions_used(self) -> int:
        return self.continuation_model_decisions_used

    @property
    def model_decisions_remaining(self) -> int | None:
        return remaining(self.limit, self.model_decisions_used)

    def restore_usage(self, records: Iterable[Mapping[str, object]], *, tool_calls_used: int) -> None:
        """Fold durable charges without replenishing decisions, checks, or reserves."""
        if type(tool_calls_used) is not int or tool_calls_used < 0:
            raise ValueError("restored tool count must be a non-negative integer")
        responses = decisions = checks = reserved = terminal = 0
        for record in records:
            if not isinstance(record, Mapping):
                continue
            kind, unit = record.get("kind"), record.get("unit")
            if kind == "budget_unit_charged" and record.get("budget_unit") == "response":
                responses += 1
                decisions += 1
            elif kind == "continuation_unit_charged":
                decisions += int(unit in {"model_decision", "terminal_response"})
                checks += int(unit == "automatic_check")
                reserved += int(unit == "terminal_reserve")
                terminal += int(unit == "terminal_response")
        self.responses_used = responses
        self.continuation_model_decisions_used = decisions
        self.tool_calls_used = tool_calls_used
        self.automatic_checks_used = checks
        self.terminal_reserves_used = reserved
        self.terminal_reserves_consumed = terminal

    def can_start_response(self) -> bool:
        if self._sandbox_seconds_remaining() is not None and self._sandbox_seconds_remaining() <= 0:
            if not self._deadline_reported:
                self._deadline_reported = True
                self._emit("continuation_deadline_exhausted", reason="sandbox_deadline")
            return False
        reserved = self.terminal_reserves_used > self.terminal_reserves_consumed
        if self.model_decisions_remaining is None or self.model_decisions_remaining > 0 or reserved:
            return True
        self._emit("model_decision_cap_hit", reason="continuation_model_decision_limit")
        return False

    def charge_response(self, *, step: int | None = None) -> bool:
        if not self.can_start_response():
            return False
        using_terminal_reserve = self.model_decisions_remaining == 0
        self.continuation_model_decisions_used += 1
        if using_terminal_reserve:
            self.terminal_reserves_consumed += 1
            self._emit("continuation_unit_charged", unit="terminal_response", step=step)
        elif self.mode == "response":
            self.responses_used += 1
            self._emit("budget_unit_charged", unit="response", step=step)
        else:
            self._emit("continuation_unit_charged", unit="model_decision", step=step)
        return True

    def can_start_tool_call(self) -> bool:
        if self._sandbox_seconds_remaining() is not None and self._sandbox_seconds_remaining() <= 0:
            if not self._deadline_reported:
                self._deadline_reported = True
                self._emit("continuation_deadline_exhausted", reason="sandbox_deadline")
            return False
        if not reached(self.tool_attempt_limit, self.tool_calls_used):
            return True
        self._emit("tool_attempt_cap_hit", reason="continuation_tool_attempt_limit")
        return False

    def register_tool_call(self, *, step: int | None = None,
                           tool: str | None = None) -> bool:
        if not self.can_start_tool_call():
            return False
        self.tool_calls_used += 1
        self._emit("continuation_unit_charged", unit="tool_attempt", step=step, tool=tool)
        return True

    def continuation_snapshot(self) -> dict[str, object]:
        snapshot = super().continuation_snapshot()
        snapshot["terminal_reserves_consumed"] = self.terminal_reserves_consumed
        snapshot["funded_terminal_responses_remaining"] = max(
            0, self.terminal_reserves_used - self.terminal_reserves_consumed)
        return snapshot

    def render_continuation_limits(self) -> str:
        snapshot = self.continuation_snapshot()
        seconds = snapshot["sandbox_seconds_remaining"]
        deadline = _time_left_text(self, seconds)
        ordinary_admission = self.qualify_admission(
            model_decisions=1, tool_attempts=1,
            output_tokens=self.output_reservation_tokens)
        terminal_admission = self.qualify_admission(
            model_decisions=0, tool_attempts=1, terminal_reserves=1,
            output_tokens=self.output_reservation_tokens)
        return (
            "[CONTINUATION LIMITS | stateframe-continuation-limits/1] "
            f"model_decisions_remaining={limit_text(snapshot['model_decisions_remaining'])}; "
            f"tool_attempts_remaining={limit_text(snapshot['tool_attempts_remaining'])}; "
            f"automatic_checks_remaining={limit_text(snapshot['automatic_checks_remaining'])}; "
            f"terminal_reserves_remaining={snapshot['terminal_reserves_remaining']}; "
            f"funded_terminal_responses_remaining={snapshot['funded_terminal_responses_remaining']}; "
            f"output_reservation_tokens={snapshot['output_reservation_tokens']}; "
            f"sandbox_seconds_remaining={deadline}; "
            f'next_action_admission={"allowed" if ordinary_admission or terminal_admission else "denied"}; '
            "cap_stop_reasons=model_decisions|tool_attempts|automatic_checks|"
            "terminal_reserve|output_reservation|sandbox_deadline"
            + ("; " + UNBOUNDED_GUIDANCE if self.limit is None else "")
        )

__all__ = [
    "BUDGET_UNIT_ENV",
    "LEGACY_SOURCE_LABELS",
    "PLAIN_SOURCE_LABEL",
    "SOURCE_BLOCK_LABELS",
    "SOURCE_BLOCK_PREFIXES",
    "READ_RECEIPT_ENV",
    "SOURCE_LABELS_ENV",
    "ContinuationLimits",
    "ResponseBudget",
    "budget_unit_mode",
    "complete_current_source_evidence",
    "normalize_path",
    "plain_source_labels_enabled",
    "read_receipt_enabled",
    "read_receipt_mode",
    "read_unchanged_receipt",
    "short_revision",
    "source_header",
    "source_header_variants",
    "source_labels_mode",
    "working_set_receipt",
]
