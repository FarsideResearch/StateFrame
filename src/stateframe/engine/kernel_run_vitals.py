"""Bounded, read-only counters for one StateFrame run's end-of-run vital event."""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from statistics import median


FLAG_ENV = "STATEFRAME_RUN_VITALS"
SCHEMA = "stateframe-run-vitals/1"
_MAX_CODE_BUCKETS = 8
_MAX_WIRE_REFERENCES = 1024
_MAX_GREENFACT_REQUESTS = 1024
_MAX_GREENFACT_REQUEST_ID_CHARS = 128
_MAX_HOST_SECONDS_SAMPLES = 512
_FRAME_PREFIX = "[StateFrame]\n"
_UNAVAILABLE_NOTICE = "StateFrame unavailable"
_SAFE_CODES = frozenset({
    "delivery_blocked", "CATALOG_UNKNOWN", "not_submittable", "completion_refused",
    "eligibility_revoked",
    "UNRESOLVED_GATING_ITEMS", "readiness_disposition", "DISABLED",
    "wire_truth_bounded_recovery",
})


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Enable observation only for the literal opt-in value."""
    source = os.environ if environ is None else environ
    return source.get("STATEFRAME_RUN_VITALS", "off") == "on"


def _nonnegative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _finite_nonnegative(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _safe_code(value: object) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    code = value.strip()
    if code not in _SAFE_CODES:
        return "OTHER"
    return code


def _add_code(counts: dict[str, int], value: object) -> None:
    code = _safe_code(value)
    if code not in counts and len(counts) >= _MAX_CODE_BUCKETS - 1:
        code = "OTHER"
    counts[code] = counts.get(code, 0) + 1


def _add_codes(counts: dict[str, int], value: object) -> None:
    if isinstance(value, str):
        _add_code(counts, value)
        return
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return
    for index, code in enumerate(value):
        if index >= _MAX_CODE_BUCKETS:
            _add_code(counts, "OTHER")
            break
        _add_code(counts, code)


def _decode_body(payload: object) -> Mapping[str, object] | None:
    candidate = payload
    if isinstance(candidate, Mapping):
        return candidate if isinstance(candidate.get("messages"), Sequence) else None
    for _ in range(2):
        if isinstance(candidate, (bytes, bytearray)):
            try:
                candidate = candidate.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(candidate, str):
            return None
        try:
            candidate = json.loads(candidate)
        except (TypeError, ValueError):
            return None
        if isinstance(candidate, Mapping):
            return candidate if isinstance(candidate.get("messages"), Sequence) else None
    return None


def _wire_summary(payload: object) -> tuple[bool, bool, int] | None:
    body = _decode_body(payload)
    if body is None:
        return None
    messages = body.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return None
    frame_present = False
    notice_messages = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if content.startswith(_FRAME_PREFIX):
            frame_present = True
        if _UNAVAILABLE_NOTICE in content:
            notice_messages += 1
    return frame_present and notice_messages == 0, notice_messages > 0, notice_messages


def _greenfact_suffix_present(payload: object) -> bool:
    body = _decode_body(payload)
    messages = body.get("messages") if body is not None else None
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        return False
    from kernel_greenfact import GREENFACT_TEXT
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            return isinstance(content, str) and content.endswith(GREENFACT_TEXT)
    return False


class RunVitalsAccumulator:
    """Aggregate safe journal facts without retaining prompts or response bodies."""

    def __init__(self) -> None:
        self.requests_submitted = 0
        self.responses_received = 0
        self.retried_submissions = 0
        self.wire_bodies_observed = 0
        self.wire_bodies_missing = 0
        self.frame_sent_count = 0
        self.unavailable_notice_bodies = 0
        self.unavailable_notice_messages = 0
        self.f4_fit_checks = 0
        self.fits_false = 0
        self.accepted_submits = 0
        self.done_calls = 0
        self.rejected_events = 0
        self.greenfact_green_now_requests = 0
        self.greenfact_renders = 0
        self._greenfact_pending_request_ids: set[str] = set()
        self._greenfact_reference_overflow = False
        self.wire_truth_blocks = 0
        self.wire_truth_degrades = 0
        self.done_refusals_by_code: dict[str, int] = {}
        self.submit_refusals_by_code: dict[str, int] = {}
        self.done_challenges_by_code: dict[str, int] = {}
        self.done_facts_rejections_by_code: dict[str, int] = {}
        self.wire_truth_reprepared_by_reason: dict[str, int] = {}
        self._wire_summaries_by_event: dict[str, tuple[bool, bool, int]] = {}
        self._wire_reference_overflow = False
        self._host_seconds_samples: list[float] = []
        self._host_seconds_sample_count = 0
        self._host_seconds_sample_overflow = False
        self._host_model_ratio_total = 0.0
        self._host_model_ratio_count = 0
        self._cache_prompt_tokens = 0
        self._cache_uncached_tokens = 0
        self.cache_layout_fail_open = 0
        self._cache_hit_tokens = 0

    def _observe_greenfact_decision(self, record: Mapping[str, object]) -> None:
        request_id = record.get("request_id")
        if (not isinstance(request_id, str) or not request_id
                or len(request_id) > _MAX_GREENFACT_REQUEST_ID_CHARS):
            self._greenfact_reference_overflow = True
            return
        if request_id in self._greenfact_pending_request_ids:
            return
        if len(self._greenfact_pending_request_ids) >= _MAX_GREENFACT_REQUESTS:
            self._greenfact_reference_overflow = True
            return
        self._greenfact_pending_request_ids.add(request_id)

    def _observe_greenfact_submission(self, record: Mapping[str, object]) -> None:
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or request_id not in self._greenfact_pending_request_ids:
            return
        self._greenfact_pending_request_ids.remove(request_id)
        self.greenfact_green_now_requests += 1
        payload = record.get("payload")
        if payload is None:
            payload = record.get("payload_utf8")
        if _greenfact_suffix_present(payload):
            self.greenfact_renders += 1

    def observe(self, record: Mapping[str, object]) -> None:
        """Read one finalized journal record before it is serialized or externalized."""
        kind = record.get("kind")
        if not isinstance(kind, str) or kind == "run_vitals":
            return
        if kind == "greenfact_green_now":
            self._observe_greenfact_decision(record)
            return
        if kind == "done_fact_rendered":
            return

        if kind == "request_submitted":
            self.requests_submitted += 1
            self._observe_greenfact_submission(record)
            retry_index = _nonnegative_int(record.get("retry_index"))
            if retry_index is not None and retry_index > 0:
                self.retried_submissions += 1
            # `payload` is the final body passed to JSON serialization at this
            # boundary; prefer it to avoid parsing a second copy of the wire.
            summary = _wire_summary(record.get("payload"))
            if summary is None:
                summary = _wire_summary(record.get("payload_utf8"))
            if summary is None:
                reference = record.get("payload_ref")
                delta = record.get("payload_delta")
                if (isinstance(reference, str) and isinstance(delta, Mapping)
                        and delta.get("identical_wire") is True):
                    summary = self._wire_summaries_by_event.get(reference)
            if summary is None:
                self.wire_bodies_missing += 1
            else:
                self.wire_bodies_observed += 1
                frame_sent, has_notice, notice_messages = summary
                self.frame_sent_count += int(frame_sent)
                self.unavailable_notice_bodies += int(has_notice)
                self.unavailable_notice_messages += notice_messages
                event_id = record.get("event_id")
                if isinstance(event_id, str) and event_id:
                    if len(self._wire_summaries_by_event) < _MAX_WIRE_REFERENCES:
                        self._wire_summaries_by_event[event_id] = summary
                    else:
                        self._wire_reference_overflow = True

        elif kind == "response_received":
            self.responses_received += 1
            accounting = record.get("accounting")
            if isinstance(accounting, Mapping):
                from kernel_prompt_accounting import accounting_tokens
                counts = accounting_tokens(accounting, record.get("response"))
                prompt_n = _nonnegative_int(counts["total"])
                cache_n = _nonnegative_int(counts["cached"])
                uncached = _nonnegative_int(counts["uncached"])
                if prompt_n is not None and cache_n is not None and uncached is not None:
                    self._cache_prompt_tokens += prompt_n
                    self._cache_uncached_tokens += uncached
                    self._cache_hit_tokens += cache_n

        elif kind == "stateframe_page_log_failed_open":
            self.cache_layout_fail_open += 1

        elif kind == "f4_budget_fit":
            self.f4_fit_checks += 1
            if record.get("fits") is False:
                self.fits_false += 1

        elif kind == "completion_reducer_decision":
            decision = record.get("decision")
            if isinstance(decision, Mapping) and decision.get("accepted") is True:
                self.accepted_submits += 1

        elif kind == "timeline_event":
            timeline_event = record.get("timeline_event")
            if (isinstance(timeline_event, Mapping)
                    and timeline_event.get("actor") == "agent"
                    and timeline_event.get("content") == "done"):
                self.done_calls += 1

        elif kind == "unified_stop_decision":
            if record.get("action") == "delivery_blocked":
                _add_code(self.done_refusals_by_code, "delivery_blocked")

        elif kind == "green_lock_refused":
            if record.get("action_kind") == "done":
                _add_code(
                    self.done_refusals_by_code,
                    record.get("reason_code", record.get("reason", record.get("code"))),
                )

        elif kind == "catalog_unknowns_completion_blocked":
            _add_code(self.done_refusals_by_code, record.get("reason_code"))

        elif kind == "readiness_submit_refused":
            reasons = record.get("reason_codes")
            if (isinstance(reasons, Sequence) and reasons
                    and not isinstance(reasons, (str, bytes, bytearray))):
                _add_codes(self.submit_refusals_by_code, reasons)
            else:
                _add_code(self.submit_refusals_by_code, record.get("reason"))

        elif kind == "done_facts_computed":
            _add_codes(self.done_facts_rejections_by_code, record.get("rejection_reasons"))

        elif kind == "adversary_challenge_schema_evaluated":
            result = record.get("result")
            if isinstance(result, Mapping):
                challenge_code = result.get("code") or result.get("status")
            else:
                challenge_code = result
            _add_code(self.done_challenges_by_code, challenge_code)

        elif kind == "WIRE_TRUTH_BLOCKED":
            self.wire_truth_blocks += 1

        elif kind == "WIRE_TRUTH_DEGRADED":
            self.wire_truth_degrades += 1

        elif kind == "request_reprepared":
            reason = record.get("reason")
            if isinstance(reason, str) and "wire_truth" in reason:
                _add_code(self.wire_truth_reprepared_by_reason, reason)

        elif kind == "request_prepare_timing":
            host_seconds = _finite_nonnegative(record.get("host_seconds"))
            if host_seconds is not None:
                self._host_seconds_sample_count += 1
                if len(self._host_seconds_samples) < _MAX_HOST_SECONDS_SAMPLES:
                    self._host_seconds_samples.append(host_seconds)
                else:
                    self._host_seconds_sample_overflow = True

        elif kind == "turn_timing":
            wall_gap_ns = _nonnegative_int(record.get("wall_gap_ns"))
            provider_ns = _nonnegative_int(record.get("provider_ns"))
            if wall_gap_ns is not None and provider_ns is not None and provider_ns > 0:
                self._host_model_ratio_total += wall_gap_ns / provider_ns
                self._host_model_ratio_count += 1

        elif kind == "journal_rejected_event":
            self.rejected_events += 1

    def snapshot(self, *, observer_failed: bool = False) -> dict[str, object]:
        host_seconds_per_turn = (
            round(median(self._host_seconds_samples), 3)
            if self._host_seconds_samples else None
        )
        host_model_ratio = (
            round(self._host_model_ratio_total / self._host_model_ratio_count, 3)
            if self._host_model_ratio_count else None
        )
        cache_hit_share = (
            round(self._cache_hit_tokens / self._cache_prompt_tokens, 6)
            if self._cache_prompt_tokens else None
        )
        frame_sent_share = (
            round(self.frame_sent_count / self.requests_submitted, 6)
            if self.requests_submitted else None
        )
        complete = not (
            observer_failed or self.wire_bodies_missing or self._wire_reference_overflow
                or self._host_seconds_sample_overflow or self._greenfact_reference_overflow
        )
        return {
            "vitals_schema": SCHEMA,
            "observation_complete": complete,
            "requests_submitted": self.requests_submitted,
            "responses_received": self.responses_received,
            "retried_submissions": self.retried_submissions,
            "wire_bodies_observed": self.wire_bodies_observed,
            "wire_bodies_missing": self.wire_bodies_missing,
            "frame_sent_count": self.frame_sent_count,
            "frame_sent_share": frame_sent_share,
            "stateframe_unavailable_notice_bodies": self.unavailable_notice_bodies,
            "stateframe_unavailable_notice_messages": self.unavailable_notice_messages,
            "f4_fit_checks": self.f4_fit_checks,
            "fits_false": self.fits_false,
            "accepted_submits": self.accepted_submits,
            "done_calls": self.done_calls,
            "greenfact_fired_when_green": {
                "green_now_requests": self.greenfact_green_now_requests,
                "renders": self.greenfact_renders,
                "alarm": (
                    self.greenfact_green_now_requests > 0
                    and self.greenfact_renders == 0
                ),
            },
            "done_refusals_by_code": dict(sorted(self.done_refusals_by_code.items())),
            "submit_refusals_by_code": dict(sorted(self.submit_refusals_by_code.items())),
            "done_challenges_by_code": dict(sorted(self.done_challenges_by_code.items())),
            "done_facts_rejections_by_code": dict(sorted(
                self.done_facts_rejections_by_code.items())),
            "wire_truth_blocks": self.wire_truth_blocks,
            "wire_truth_degrades": self.wire_truth_degrades,
            "wire_truth_reprepared_by_reason": dict(sorted(
                self.wire_truth_reprepared_by_reason.items())),
            "host_seconds_per_turn": host_seconds_per_turn,
            "host_seconds_per_turn_samples": self._host_seconds_sample_count,
            "host_seconds_sample_truncated": self._host_seconds_sample_overflow,
            "host_model_ratio": host_model_ratio,
            "cache_hit_share": cache_hit_share,
            "cache_prompt_tokens": self._cache_prompt_tokens,
            "cache_uncached_tokens": self._cache_uncached_tokens,
            "cache_layout_fail_open": self.cache_layout_fail_open,
            "cache_hit_tokens": self._cache_hit_tokens,
            "rejected_events": self.rejected_events,
        }


def emit_run_vitals(journal: object) -> dict[str, object] | None:
    """Emit one terminal summary when enabled; never let telemetry affect the run."""
    if not enabled() or getattr(journal, "_run_vitals_emitted", False):
        return None
    accumulator = getattr(journal, "_run_vitals_accumulator", None)
    emitter = getattr(journal, "emit", None)
    if not isinstance(accumulator, RunVitalsAccumulator) or not callable(emitter):
        return None
    try:
        setattr(journal, "_run_vitals_emitted", True)
        observer_failed = bool(getattr(journal, "_run_vitals_observer_failed", False))
        return emitter(
            "run_vitals",
            producer="kernel_run_vitals",
            **accumulator.snapshot(observer_failed=observer_failed),
        )
    except Exception:
        return None


__all__ = ["FLAG_ENV", "SCHEMA", "RunVitalsAccumulator", "emit_run_vitals", "enabled"]
