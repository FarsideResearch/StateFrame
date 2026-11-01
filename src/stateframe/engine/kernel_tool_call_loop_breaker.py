"""Audit tool-call cap endings and stop bounded, non-productive action loops.

The breaker is opt-in for finite runs and required for unbounded runs. Its state is local to one attempt and its
notice is rendered at the next real StateFrame request boundary, so it cannot
silently discard the action that exposed the loop.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


SWITCH = "STATEFRAME_TOOL_CALL_LOOP_BREAKER"
STREAK_ENV = "STATEFRAME_TOOL_CALL_LOOP_BREAKER_STREAK"
DEFAULT_STREAK = 5
MIN_STREAK = 2
MAX_STREAK = 32
LOOP_BREAKER_NOTICE = "[kernel] TOOL_CALL_LOOP_BREAKER"

_EDIT_TOOLS = frozenset({"edit", "str_replace", "replace_module", "file_create", "write", "replace"})
_TEST_TOOLS = frozenset({"run_tests", "test", "pytest"})
_LOOP_TOOLS = frozenset({
    "read", "grep", "file_list", "run_tests", "test", "pytest", "edit",
    "str_replace", "replace_module", "file_create", "write", "replace",
    "probe", "predict", "dependency_fact", "dispute_check", "read_ticket",
    "dispute", "work",
})
_IGNORED_ARGUMENTS = frozenset({"why", "reason"})
_REJECTION_MARKERS = (
    "rejected", "refused", "denied", "no-op", "no op", "error", "failed",
    "not applied", "not executed", "not changed", "unchanged", "no change",
    "no changes", "without change", "not green",
)
_EDIT_PROGRESS_MARKERS = ("applied", "created", "modified", "changed")


def mode() -> str:
    """Return the strict switch mode; an invalid value is configuration error."""
    value = os.environ.get(SWITCH, "off").strip().lower()
    if value in {"off", "0", "false", "no", ""}:
        return "off"
    if value in {"on", "1", "true", "yes"}:
        return "on"
    raise ValueError(f"{SWITCH} must be on or off")


def enabled() -> bool:
    return mode() == "on"


def threshold() -> int:
    raw = os.environ.get(STREAK_ENV, str(DEFAULT_STREAK)).strip()
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{STREAK_ENV} must be an integer from {MIN_STREAK} to {MAX_STREAK}") from error
    if not MIN_STREAK <= value <= MAX_STREAK:
        raise ValueError(f"{STREAK_ENV} must be an integer from {MIN_STREAK} to {MAX_STREAK}")
    return value


def _stable_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _stable_value(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(child) for child in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_stable_value(child) for child in value), key=repr)
    return f"<{type(value).__name__}>"


def action_fingerprint(tool: str, arguments: Mapping[str, object] | None = None) -> str:
    """Hash the action identity without placing model arguments in telemetry."""
    clean_arguments = {
        str(key): _stable_value(value)
        for key, value in (arguments or {}).items()
        if str(key) not in _IGNORED_ARGUMENTS
    }
    payload = json.dumps(
        {"tool": str(tool).strip().lower(), "arguments": clean_arguments},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _tool_name(tool: object) -> str:
    return str(tool or "unknown").strip().lower()


def _contains_marker(text: str, marker: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(marker)}(?![\w])", text) is not None


def _edit_is_productive(tool: str, observation: str) -> bool:
    if tool not in _EDIT_TOOLS:
        return False
    text = str(observation or "").strip().lower()
    if any(_contains_marker(text, marker) for marker in _REJECTION_MARKERS):
        return False
    return any(_contains_marker(text, marker) for marker in _EDIT_PROGRESS_MARKERS)


def _test_is_green(observation: str) -> bool:
    text = str(observation or "").strip().lower()
    if any(_contains_marker(text, marker) for marker in ("not green", "failed", "error")):
        return False
    return (_contains_marker(text, "green")
            or _contains_marker(text, "pytest exit 0")
            or re.match(r"^ok(?:$|[\s.,:;!])", text) is not None)


def terminal_preemption_allowed(runtime: object) -> bool:
    """Let the green-then-exhaust owner win its shared terminal boundary."""
    return (bool(getattr(runtime, "_tool_call_loop_breaker_ready", False))
            and not bool(getattr(runtime, "_green_then_exhaust_boundary_active", False)))


@dataclass(frozen=True)
class LoopDecision:
    tool: str
    fingerprint: str
    streak: int
    productive: bool
    tripped: bool
    notice: str | None = None


class ToolCallLoopBreaker:
    """Track exact action repeats and trip once without productive progress."""

    def __init__(self, *, threshold: int = DEFAULT_STREAK):
        if type(threshold) is not int or not MIN_STREAK <= threshold <= MAX_STREAK:
            raise ValueError(f"threshold must be an integer from {MIN_STREAK} to {MAX_STREAK}")
        self.threshold = threshold
        self.actions_seen = 0
        self.productive_actions = 0
        self.repeated_actions = 0
        self.max_streak = 0
        self.tripped = False
        self._previous_fingerprint: str | None = None
        self._streak = 0
        self._edit_progress_pending = False
        self._pending_notice: str | None = None
        self.first_break_action: str | None = None

    @classmethod
    def from_environment(cls) -> "ToolCallLoopBreaker":
        return cls(threshold=threshold())

    def observe(
        self,
        tool: str,
        arguments: Mapping[str, object] | None,
        observation: str,
    ) -> LoopDecision:
        name = _tool_name(tool)
        fingerprint = action_fingerprint(name, arguments)
        self.actions_seen += 1
        edit_progress = _edit_is_productive(name, observation)
        productive = edit_progress or (
            name in _TEST_TOOLS and self._edit_progress_pending and _test_is_green(observation)
        )
        if productive:
            self.productive_actions += 1
            self._previous_fingerprint = None
            self._streak = 0
            self._edit_progress_pending = edit_progress
        elif name not in _LOOP_TOOLS:
            self._previous_fingerprint = None
            self._streak = 0
            self._edit_progress_pending = False
        else:
            self._edit_progress_pending = False
            if fingerprint == self._previous_fingerprint:
                self._streak += 1
                self.repeated_actions += 1
            else:
                self._streak = 1
            self._previous_fingerprint = fingerprint
            self.max_streak = max(self.max_streak, self._streak)

        notice = None
        if not self.tripped and not productive and self._streak >= self.threshold:
            self.tripped = True
            self.first_break_action = name
            notice = (
                f"{LOOP_BREAKER_NOTICE}: `{name}` repeated {self._streak} times without "
                "a productive workspace change. Choose a different action or stop with "
                "a concrete blocker."
            )
            self._pending_notice = notice
        return LoopDecision(
            tool=name,
            fingerprint=fingerprint,
            streak=self._streak,
            productive=productive,
            tripped=self.tripped and notice is not None,
            notice=notice,
        )

    def take_notice(self) -> str | None:
        notice = self._pending_notice
        self._pending_notice = None
        return notice


class UnboundedProgressWatch:
    """Cover cycles and non-actions that the exact consecutive-call breaker cannot.

    Only host revision/evidence novelty resets the streak. Model text, repeated
    refusals, counters and timestamps never establish progress. Memory stays
    bounded independently of run length. Membership is conservative once the
    digest filter saturates; there is no fixed run-length cap.
    """

    def __init__(self, *, threshold: int = 32) -> None:
        if type(threshold) is not int or threshold < 2:
            raise ValueError("progress threshold must be at least two")
        self.threshold = threshold
        self.streak = 0
        self._seen: dict[str, None] = {}
        # Monotone membership prevents an old identity becoming new when the
        # exact recent cache evicts it. False positives only undercount novelty.
        self._seen_filter = bytearray(128 * 1024)
        self._receipt_cursor = 0
        self._last_boundary: int | None = None
        self._batch_tail_pending: str | None = None

    def snapshot(self) -> dict[str, object]:
        """Persist recent hashes and monotone membership in the worker checkpoint."""
        state = {
            "schema": "unbounded-progress-watch/3", "threshold": self.threshold,
            "streak": self.streak, "seen": list(self._seen),
            "receipt_cursor": self._receipt_cursor,
            "last_boundary": self._last_boundary,
            "seen_filter": base64.b64encode(self._seen_filter).decode("ascii"),
        }
        from kernel_code_bookmarks import enabled as bookmarks_enabled
        if bookmarks_enabled():
            state["batch_tail_pending"] = self._batch_tail_pending
        return state

    @classmethod
    def from_snapshot(cls, state: object) -> UnboundedProgressWatch:
        old_fields = {"schema", "threshold", "streak", "seen", "receipt_cursor", "last_boundary"}
        if (not isinstance(state, Mapping)
                or (state.get("schema"), frozenset(state)) not in {
                    ("unbounded-progress-watch/1", frozenset(old_fields)),
                    ("unbounded-progress-watch/2", frozenset(old_fields | {"batch_tail_pending"})),
                    ("unbounded-progress-watch/3", frozenset(old_fields | {"seen_filter"})),
                    ("unbounded-progress-watch/3", frozenset(old_fields | {"batch_tail_pending", "seen_filter"})),
                }):
            raise ValueError("invalid progress watch snapshot")
        tail = state.get("batch_tail_pending")
        if tail is not None and (not isinstance(tail, str) or not 0 < len(tail) <= 160):
            raise ValueError("invalid progress watch batch tail")
        watch = cls(threshold=state["threshold"])
        for key in ("streak", "receipt_cursor"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError("invalid progress watch counter")
        boundary = state["last_boundary"]
        if boundary is not None and (type(boundary) is not int or boundary < 0):
            raise ValueError("invalid progress watch boundary")
        seen = state["seen"]
        if (not isinstance(seen, list) or len(seen) > 4096
                or any(not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None
                       for key in seen) or len(set(seen)) != len(seen)):
            raise ValueError("invalid progress watch identities")
        watch.streak = state["streak"]
        watch._receipt_cursor = state["receipt_cursor"]
        watch._last_boundary = boundary
        watch._seen = dict.fromkeys(seen)
        if state["schema"] == "unbounded-progress-watch/3":
            encoded = state["seen_filter"]
            try:
                decoded = base64.b64decode(encoded, validate=True) if isinstance(encoded, str) else b""
            except (ValueError, binascii.Error) as error:
                raise ValueError("invalid progress watch identities") from error
            if len(decoded) != len(watch._seen_filter):
                raise ValueError("invalid progress watch identities")
            watch._seen_filter[:] = decoded
            if any(not watch._remember_filter(key) for key in seen):
                raise ValueError("invalid progress watch identities")
        else:
            for key in seen:
                watch._remember_filter(key)
        watch._batch_tail_pending = tail
        return watch

    def _new(self, value: object) -> bool:
        key = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest()
        if self._remember_filter(key):
            return False
        if key in self._seen:
            return False
        self._seen[key] = None
        if len(self._seen) > 4096:
            del self._seen[next(iter(self._seen))]
        return True

    def _remember_filter(self, key: str) -> bool:
        """Return whether a digest was probably seen, without ever forgetting it."""
        digest = bytes.fromhex(key)
        size = len(self._seen_filter) * 8
        indexes = [int.from_bytes(digest[offset:offset + 4], "big") % size
                   for offset in range(0, 20, 4)]
        seen = all(self._seen_filter[index // 8] & (1 << (index % 8))
                   for index in indexes)
        for index in indexes:
            self._seen_filter[index // 8] |= 1 << (index % 8)
        return seen

    @staticmethod
    def _unchanged_read_batch(receipts: list[dict]) -> bool:
        """Discount only after every action in the completed batch is known."""
        if not receipts:
            return False
        for receipt in receipts:
            delivery = receipt.get("delivery")
            if (receipt.get("kind") != "read"
                    or receipt.get("result_class") != "OK"
                    or not isinstance(delivery, Mapping)
                    or delivery.get("unchanged_since_request") is not True
                    or delivery.get("requested_bytes_delivered") != 0
                    or not isinstance(delivery.get("effective_range"), Mapping)):
                return False
        return True

    def observe(self, binding: Mapping[str, object], receipts: list[dict], *,
                boundary: int | None = None, charge_idle: bool = True,
                batch_origin: str | None = None,
                batch_complete: bool = False) -> bool:
        # Worker checkpoints follow this observation but precede dispatch. A
        # restart at that same boundary must not charge the idle turn twice.
        if boundary is not None and boundary == self._last_boundary:
            return self.streak >= self.threshold
        from kernel_code_bookmarks import enabled as bookmarks_enabled
        bookmarks_on = bookmarks_enabled()
        revision = binding.get("product_files") or binding.get("product_revision") or binding.get("artifact")
        progress = self._new(("revision", revision)) if revision is not None else False
        cursor = min(self._receipt_cursor, len(receipts))
        for receipt in receipts[cursor:]:
            if receipt.get("semantic_eligible") is False:
                continue
            kind = str(receipt.get("kind", "")).lower()
            result = str(receipt.get("result_class", "")).upper()
            # A refused attempt or a changing explanation is not fresh evidence.
            if kind not in {"read", "grep", "search", "test", "probe"} or result not in {
                    "OK", "READ", "FOUND", "PASS", "FAIL", "GREEN", "RED",
                    "PASSED", "FAILED", "READ_COMPLETE", "READ_PARTIAL",
                    "HITS", "NO_HITS", "PARTIAL", "FAIL_WITH_WITNESS"}:
                continue
            if kind in {"read", "grep", "search"} and result in {"FAIL", "FAILED", "RED"}:
                continue
            identity = self._identity(binding, receipt, kind, result, revision)
            if identity is not None and self._new(identity):
                progress = True
        charge = 1
        if bookmarks_on and isinstance(batch_origin, str):
            if not batch_complete:
                charge = 0
            elif cursor == len(receipts) and self._batch_tail_pending == batch_origin:
                charge = 0
            else:
                first = next((index for index in range(len(receipts) - 1, -1, -1)
                              if receipts[index].get("request_id") == batch_origin), None)
                rows = receipts[first:] if first is not None else receipts[cursor:]
                charge = 1 if self._unchanged_read_batch(rows) else max(1, len(rows))
                self._batch_tail_pending = batch_origin
        self._receipt_cursor = len(receipts)
        self._last_boundary = boundary
        if progress:
            self.streak = 0
        elif charge_idle:
            self.streak += charge
        return self.streak >= self.threshold

    @staticmethod
    def _identity(binding: Mapping[str, object], receipt: Mapping[str, object],
                  kind: str, result: str, revision: object) -> object:
        if kind == "read":
            delivery = receipt.get("delivery")
            delivery = delivery if isinstance(delivery, Mapping) else {}
            # emit_read_receipt canonicalizes target_path through receipt_target_path;
            # delivery framing may retain the caller's spelling of that path.
            path = receipt.get("target_path") or delivery.get("path")
            source_revision = delivery.get("revision") or receipt.get("source_revision")
            if path:
                for field in ("product_files", "harness_files"):
                    files = binding.get(field)
                    if source_revision is None and isinstance(files, Mapping):
                        source_revision = files.get(path)
                if source_revision is not None:
                    # Pages, query spelling, timestamps and unrelated edits do
                    # not turn a previously read path/revision into new evidence.
                    return ("read", path, source_revision)
                return ("read", path, revision)
            # Compatibility for legacy receipts without typed path fields.
            return ("legacy-read", receipt.get("fingerprint"), revision)
        if kind in {"grep", "search"}:
            hit_list = receipt.get("result_set_digest")
            return ("search", hit_list) if isinstance(hit_list, str) and hit_list else None
        if kind == "test":
            verdict = ("PASS" if result in {"PASS", "PASSED", "GREEN"} else
                       "FAIL" if result in {"FAIL", "FAILED", "RED", "FAIL_WITH_WITNESS"} else result)
            return (kind, revision, receipt.get("fingerprint"), verdict, receipt.get("failure_identity"))
        return (revision, {key: receipt.get(key) for key in (
            "kind", "fingerprint", "result_class", "target_path", "source_revision",
            "start_line", "end_line", "start_column", "end_column", "effective_range",
        )})


def failing_test_identity(output: str, *, check_outputs: object = None,
                          journal: object = None) -> str | None:
    """Hash explicit failing node IDs, never timing, explanation or receipt IDs.

    Uses the runner's FAILED-selector convention; keep parameter IDs because
    different parameter cases can expose different failures.
    """
    sources = [output]
    if isinstance(check_outputs, Mapping):
        # The runner returns compact_observation, whose escaped preview can
        # omit every failing node. Resolve only its exact host output handles,
        # never the latest receipt (which may belong to another selector).
        handles = re.findall(
            r"(?m)^CHECK_FAILURE_SUMMARY [^\n]* full_output_handle=(check-receipt:[0-9a-f]{24})$",
            output,
        )
        for handle in set(handles):
            check = check_outputs.get(handle)
            if not isinstance(check, Mapping) or check.get("outcome") != "BEHAVIOR_FAIL":
                continue
            sources.extend(check[key] for key in ("stdout", "stderr") if isinstance(check.get(key), str))
    from kernel_wire_identity import public_wire_text
    selectors = sorted({selector for source in sources for selector in
                        re.findall(r"(?m)^\s*FAILED\s+([^\s]+\.py::[^\s]+)", source)})
    selectors = sorted({public_wire_text(selector, journal) for selector in selectors})
    if not selectors:
        return None
    return hashlib.sha256(json.dumps(selectors, separators=(",", ":")).encode()).hexdigest()


def stop_unbounded_no_progress(runtime: object, watch: UnboundedProgressWatch, *,
                               boundary: int | None = None) -> bool:
    """Run at every loop boundary, including failed/empty provider decisions."""
    from kernel_done_acceptance import done_accepted
    from kernel_terminal_policy import terminal_outcome_permitted
    if done_accepted(runtime) or terminal_outcome_permitted(
            getattr(runtime, "system_outcome", None), runtime):
        return False
    binding = runtime.binding()
    receipts = getattr(runtime.owner, "_receipts", [])
    lock = getattr(runtime, "_read_lockout_state", None)
    from kernel_read_lockout import enabled as lockout_enabled
    locked = (lockout_enabled() and isinstance(lock, Mapping) and lock.get("remaining", 0) > 0
              and getattr(runtime, "_read_lockout_notice", None))
    # Observe the boundary and receipts while paused, but do not charge an
    # idle turn caused by the host's own two-request restriction. Real new
    # evidence still resets the watch under its ordinary rules.
    batch = getattr(runtime, "native_batch", None)
    batch_origin = (getattr(batch, "origin", None)
                    if getattr(batch, "index", 0) >= 1 else None)
    batch_complete = (isinstance(batch_origin, str)
                      and getattr(batch, "index", 0) == getattr(batch, "size", -1)
                      and not getattr(batch, "pending", ())
                      and getattr(batch, "inflight", None) is None)
    if not watch.observe(binding, receipts, boundary=boundary,
                         charge_idle=not locked, batch_origin=batch_origin,
                         batch_complete=batch_complete):
        return False
    if locked:
        return False
    runtime.system_outcome = "stalled"
    runtime.journal.emit(
        "unbounded_no_progress_terminal", producer="kernel_tool_call_loop_breaker",
        terminal="STALLED", reason="no_new_host_revision_or_evidence",
        streak=watch.streak, threshold=watch.threshold, dispatch=False,
    )
    return True


@dataclass(frozen=True)
class CapAuditSummary:
    actions: int
    tool_calls: int
    tool_call_cap_hits: int
    response_budget_endings: int
    productive_actions: int
    repeated_actions: int
    max_identical_streak: int
    first_break_action: str | None

    @property
    def loops_dominate(self) -> bool:
        return self.repeated_actions > self.productive_actions

    def to_dict(self) -> dict[str, object]:
        return {
            "actions": self.actions,
            "tool_calls": self.tool_calls,
            "tool_call_cap_hits": self.tool_call_cap_hits,
            "response_budget_endings": self.response_budget_endings,
            "productive_actions": self.productive_actions,
            "repeated_actions": self.repeated_actions,
            "max_identical_streak": self.max_identical_streak,
            "first_break_action": self.first_break_action,
            "loops_dominate": self.loops_dominate,
        }


def _record_mapping(event: Mapping[str, object]) -> Mapping[str, object]:
    nested = event.get("record")
    return nested if isinstance(nested, Mapping) else event


def audit_event_stream(
    events: Iterable[Mapping[str, object]],
    *,
    loop_threshold: int = DEFAULT_STREAK,
) -> CapAuditSummary:
    """Replay measurement events without mutating the source journal."""
    breaker = ToolCallLoopBreaker(threshold=loop_threshold)
    actions = 0
    counted_tool_calls = 0
    tool_call_cap_hits = 0
    response_budget_endings = 0
    explicit_response_end = False
    productive_actions = 0
    repeated_actions = 0
    max_identical_streak = 0
    first_break_action: str | None = None
    current_run: object = None

    for event in events:
        if not isinstance(event, Mapping):
            continue
        run_id = event.get("run_id")
        if run_id is not None and run_id != current_run:
            breaker = ToolCallLoopBreaker(threshold=loop_threshold)
            current_run = run_id
            explicit_response_end = False
        kind = str(event.get("kind") or "")
        if kind == "tool_call_counted":
            counted_tool_calls += 1
        if kind in {"tool_call_cap_hit", "tool_call_cap"} \
                or str(event.get("system_outcome") or "") == "tool_call_cap":
            tool_call_cap_hits += 1
        if kind == "response_budget_exhausted":
            response_budget_endings += 1
            explicit_response_end = True
        record = _record_mapping(event)
        if kind == "action_outcome_closed" and record.get("tool"):
            actions += 1
            arguments = {
                "arguments_hash": record.get("arguments_hash"),
                "action_parameters": record.get("action_parameters", {}),
            }
            decision = breaker.observe(
                str(record.get("tool")),
                arguments,
                str(record.get("observation") or record.get("result") or ""),
            )
            productive_actions += int(decision.productive)
            repeated_actions += int(decision.streak >= 2)
            max_identical_streak = max(max_identical_streak, decision.streak)
            if first_break_action is None and decision.tripped:
                first_break_action = decision.tool
        outcome = str(
            event.get("system_outcome") or event.get("outcome") or event.get("result") or ""
        ).lower()
        if kind == "attempt_terminated" and outcome == "response_budget_exhausted" \
                and not explicit_response_end:
            response_budget_endings += 1
            explicit_response_end = True

    return CapAuditSummary(
        actions=actions,
        tool_calls=counted_tool_calls or actions,
        tool_call_cap_hits=tool_call_cap_hits,
        response_budget_endings=response_budget_endings,
        productive_actions=productive_actions,
        repeated_actions=repeated_actions,
        max_identical_streak=max_identical_streak,
        first_break_action=first_break_action,
    )
