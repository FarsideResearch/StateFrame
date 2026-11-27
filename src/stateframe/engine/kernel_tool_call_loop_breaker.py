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

from kernel_edit_tool_names import PROTOCOL_EDIT_TOOL_NAMES


SWITCH = "STATEFRAME_TOOL_CALL_LOOP_BREAKER"
STREAK_ENV = "STATEFRAME_TOOL_CALL_LOOP_BREAKER_STREAK"
DEFAULT_STREAK = 5
MIN_STREAK = 2
MAX_STREAK = 32
LOOP_BREAKER_NOTICE = "[kernel] TOOL_CALL_LOOP_BREAKER"
_SEEN_FILTER_BYTES = 8 * 1024 * 1024
_SEEN_FILTER_HASHES = 24
_LEGACY_SEEN_FILTER_BYTES = 128 * 1024

REQUESTED_EVIDENCE_PROBE = (
    "[Requested repository evidence] What missing repository fact could change your next edit? "
    "If there is one, call read on a single workspace file with an explicit start_line and "
    "end_line (at most 12 lines) that would settle it. Otherwise proceed with your next action."
)
MAX_REQUESTED_EVIDENCE_LINES = 12
REQUESTED_EVIDENCE_IDLE_OBSERVATIONS = 4


def requested_evidence_probe_enabled() -> bool:
    return os.environ.get("STATEFRAME_REQUESTED_EVIDENCE_PROBE", "off").strip().lower() == "on"

_EDIT_TOOLS = frozenset({"edit", "str_replace", "replace_module", "file_create", "write", "replace",
                         *PROTOCOL_EDIT_TOOL_NAMES})
_TEST_TOOLS = frozenset({"run_tests", "test", "pytest"})
_LOOP_TOOLS = frozenset({
    "read", "grep", "file_list", "run_tests", "test", "pytest", "edit",
    "str_replace", "replace_module", "file_create", "write", "replace",
    "probe", "predict", "dependency_fact", "dispute_check", "read_ticket",
    "dispute", "work", *PROTOCOL_EDIT_TOOL_NAMES,
})
_IGNORED_ARGUMENTS = frozenset({"why", "reason"})
# Kinds whose receipts count as fresh evidence. Output-bearing kinds (shell, web) are
# judged by their output digest, so new output is progress and identical output charges.
_OUTPUT_KINDS = frozenset({"shell", "web", "web_fetch", "web_search", "fetch"})
_PROGRESS_KINDS = frozenset({"read", "grep", "search", "test", "probe"}) | _OUTPUT_KINDS
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
    """Always False: the loop breaker is warn-only and never ends a run.

    It only renders a notice at the next request boundary. Terminal ownership
    stays with the green-then-exhaust owner, so this must not preempt it. The
    runtime argument is kept so existing callers keep the same signature.
    """
    del runtime
    return False


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
        # Repeats need the same action AND the same observation. Novel results
        # from an identical call are progress-like and must not trip the breaker.
        repeat_key = f"{fingerprint}:{hashlib.sha256(str(observation or '').encode('utf-8')).hexdigest()[:16]}"
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
            if repeat_key == self._previous_fingerprint:
                self._streak += 1
                self.repeated_actions += 1
            else:
                self._streak = 1
            self._previous_fingerprint = repeat_key
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

    def trip_external(self, tool: str, reason: str) -> str | None:
        """Trip once from an external repeat detector (e.g. identical edit refusals).

        Returns the notice when this call tripped the breaker, else None (already tripped).
        """
        if self.tripped:
            return None
        name = _tool_name(tool)
        self.tripped = True
        self.first_break_action = name
        notice = (f"{LOOP_BREAKER_NOTICE}: `{name}` {reason} Choose a different action or "
                  "stop with a concrete blocker.")
        self._pending_notice = notice
        return notice


# Mask recognizable runtime noise, preserving ordinary numbers, source
# coordinates, assertion values, versions, exit codes and content hashes.
_VOLATILE_OUTPUT = (
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"), "<timestamp>"),
    (re.compile(r"(?im)^Date: [^\r\n]+"), "Date: <timestamp>"),
    (re.compile(r"\b(pid|ppid)\s*[:=]\s*\d+\b", re.I), r"\1=<process>"),
    (re.compile(r"\b(elapsed|duration)\s*[:=]\s*\d+(?:\.\d+)?\s*(?:ms|s|seconds?)\b", re.I), r"\1=<time>"),
    (re.compile(r"(?<!\w)/(?:tmp|var/tmp)/tmp[\w.-]+"), "/tmp/<temporary>"),
    (re.compile(r"\b((?:request|trace|span|correlation)[_-]?id)\s*[:=]\s*(?:0x)?[0-9a-f]+\b", re.I), r"\1=<id>"),
    (re.compile(r"\bport\s*[:=]\s*\d{1,5}\b", re.I), "port=<port>"),
    (re.compile(r"(https?://(?:127\.0\.0\.1|localhost|\[::1\])):\d{1,5}\b", re.I), r"\1:<port>"),
)


def _digest(value: object) -> str:
    text = str(value)
    for pattern, replacement in _VOLATILE_OUTPUT:
        text = pattern.sub(replacement, text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UnboundedProgressWatch:
    """Cover cycles and non-actions that the exact consecutive-call breaker cannot.

    Only host revision/evidence novelty resets the streak. Model text, repeated
    refusals, counters and timestamps never establish progress. Memory stays
    bounded independently of run length. With bookmarks enabled, membership
    is conservative once the digest filter saturates.
    """

    def __init__(self, *, threshold: int = 32) -> None:
        if type(threshold) is not int or threshold < 2:
            raise ValueError("progress threshold must be at least two")
        self.threshold = threshold
        self.streak = 0
        self._seen: dict[str, None] = {}
        # Bookmarks-off retains the released 4,096-entry eviction behavior.
        # Allocate the monotone filter only when bookmarks need it.
        self._seen_filter: bytearray | None = None
        self._receipt_cursor = 0
        self._last_boundary: int | None = None
        self._batch_tail_pending: str | None = None

    def snapshot(self) -> dict[str, object]:
        """Persist the release snapshot unless bookmarks enable monotone membership."""
        from kernel_code_bookmarks import enabled as bookmarks_enabled
        bookmarks_on = bookmarks_enabled()
        state = {
            "schema": "unbounded-progress-watch/3" if bookmarks_on else "unbounded-progress-watch/1",
            "threshold": self.threshold,
            "streak": self.streak, "seen": list(self._seen),
            "receipt_cursor": self._receipt_cursor,
            "last_boundary": self._last_boundary,
        }
        if bookmarks_on:
            if self._seen_filter is None:
                for key in self._seen:
                    self._remember_filter(key)
                if self._seen_filter is None:
                    self._seen_filter = bytearray(_SEEN_FILTER_BYTES)
            state["seen_filter"] = base64.b64encode(self._seen_filter).decode("ascii")
            state["batch_tail_pending"] = self._batch_tail_pending
        if getattr(self, "_requested_evidence_revision", None) is not None:
            state["requested_evidence_revision"] = self._requested_evidence_revision
        if hasattr(self, "_requested_evidence_idle"):
            state["requested_evidence_idle"] = self._requested_evidence_idle
        if getattr(self, "requested_evidence_read", None) is not None:
            state["requested_evidence_read"] = list(self.requested_evidence_read)
        return state

    @classmethod
    def from_snapshot(cls, state: object) -> UnboundedProgressWatch:
        if not isinstance(state, Mapping):
            raise ValueError("invalid progress watch snapshot")
        state = dict(state)
        probe_revision = state.pop("requested_evidence_revision", "")
        probe_idle = state.pop("requested_evidence_idle", None)
        probe_read = state.pop("requested_evidence_read", None)
        old_fields = {"schema", "threshold", "streak", "seen", "receipt_cursor", "last_boundary"}
        if ((state.get("schema"), frozenset(state)) not in {
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
            if len(decoded) not in {_LEGACY_SEEN_FILTER_BYTES, _SEEN_FILTER_BYTES}:
                raise ValueError("invalid progress watch identities")
            if len(decoded) == _LEGACY_SEEN_FILTER_BYTES:
                # The old bitset cannot be enlarged without losing its hashes.
                # Retain exact recent identities; start a larger filter from them.
                size = len(decoded) * 8
                for key in seen:
                    digest = bytes.fromhex(key)
                    indexes = (int.from_bytes(digest[offset:offset + 4], "big") % size
                               for offset in range(0, 20, 4))
                    if not all(decoded[index // 8] & (1 << (index % 8)) for index in indexes):
                        raise ValueError("invalid progress watch identities")
                    watch._remember_filter(key)
            else:
                watch._seen_filter = bytearray(decoded)
                if any(not watch._remember_filter(key) for key in seen):
                    raise ValueError("invalid progress watch identities")
        elif state["schema"] == "unbounded-progress-watch/2":
            for key in seen:
                watch._remember_filter(key)
        else:
            from kernel_code_bookmarks import enabled as bookmarks_enabled
            if bookmarks_enabled():
                for key in seen:
                    watch._remember_filter(key)
        watch._batch_tail_pending = tail
        if probe_idle is not None:
            if type(probe_idle) is not int or not 0 <= probe_idle <= watch.threshold:
                raise ValueError("invalid requested evidence progress")
            watch._requested_evidence_idle = probe_idle
        if probe_revision != "":
            if (probe_idle is None or not isinstance(probe_revision, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", probe_revision)):
                raise ValueError("invalid requested evidence probe revision")
            watch._requested_evidence_revision = probe_revision
        if probe_read is not None:
            if (probe_idle is None or not isinstance(probe_read, list) or len(probe_read) != 2
                    or any(not isinstance(value, str) or not 0 < len(value) <= 160 for value in probe_read)):
                raise ValueError("invalid requested evidence read")
            watch.requested_evidence_read = tuple(probe_read)
        return watch

    def enable_evidence_probe(self) -> None:
        if not hasattr(self, "_requested_evidence_idle"):
            self._requested_evidence_idle = 0

    def evidence_probe_due(self, boundary: int) -> bool:
        idle = getattr(self, "_requested_evidence_idle", 0)
        return (boundary == self._last_boundary and self.streak < self.threshold
                and idle >= min(REQUESTED_EVIDENCE_IDLE_OBSERVATIONS, self.threshold // 2))

    def claim_evidence_probe(self, artifact: object, *, boundary: int) -> bool:
        """Offer one probe partway through an observed, still-live stall."""
        if (not self.evidence_probe_due(boundary)
                or not isinstance(artifact, str) or not artifact):
            return False
        revision = hashlib.sha256(artifact.encode()).hexdigest()
        if getattr(self, "_requested_evidence_revision", None) == revision:
            return False
        self._requested_evidence_revision = revision
        return True

    def _new(self, value: object) -> bool:
        key = hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8")).hexdigest()
        from kernel_code_bookmarks import enabled as bookmarks_enabled
        if bookmarks_enabled() and self._remember_filter(key):
            return False
        if key in self._seen:
            return False
        self._seen[key] = None
        if len(self._seen) > 4096:
            del self._seen[next(iter(self._seen))]
        return True

    def _remember_filter(self, key: str) -> bool:
        """Return probable membership; use 24 positions in an 8 MiB bitset."""
        if self._seen_filter is None:
            self._seen_filter = bytearray(_SEEN_FILTER_BYTES)
        digest = bytes.fromhex(key)
        size = len(self._seen_filter) * 8
        start = int.from_bytes(digest[:8], "big")
        step = int.from_bytes(digest[8:16], "big") | 1
        indexes = [(start + index * step) % size for index in range(_SEEN_FILTER_HASHES)]
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
        from kernel_read_receipts import enabled as allocator_enabled
        for receipt in receipts:
            delivery = receipt.get("delivery")
            delivery = delivery if isinstance(delivery, Mapping) else {}
            legacy_repeat = (delivery.get("unchanged_since_request") is True
                             and delivery.get("requested_bytes_delivered") == 0)
            captured_repeat = (allocator_enabled()
                and delivery.get("working_set_repeat_capture") is True
                and type(delivery.get("requested_bytes_delivered")) is int
                and delivery["requested_bytes_delivered"] > 0)
            if (receipt.get("kind") != "read"
                    or receipt.get("result_class") != "OK"
                    or not (legacy_repeat or captured_repeat)
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
            if kind not in _PROGRESS_KINDS or result not in {
                    "OK", "READ", "FOUND", "PASS", "FAIL", "GREEN", "RED",
                    "PASSED", "FAILED", "READ_COMPLETE", "READ_PARTIAL",
                    "HITS", "NO_HITS", "PARTIAL", "FAIL_WITH_WITNESS",
                    "NO_OP", "HOST_OBSERVED"}:
                continue
            if kind in {"read", "grep", "search"} and result in {"FAIL", "FAILED", "RED"}:
                continue
            # A read that delivered none of its requested bytes is no-progress evidence:
            # its path/range identity must not reset the stall streak.
            if kind == "read":
                from kernel_idle_governor import is_undelivered_read
                if is_undelivered_read(receipt.get("delivery")):
                    continue
            identity = self._identity(binding, receipt, kind, result, revision)
            if identity is not None and self._new(identity):
                progress = True
                # Remember the first delivered range so its identical re-read still charges.
                if kind == "read":
                    self._range_progress(binding, receipt, revision)
            elif kind == "read" and self._range_progress(binding, receipt, revision):
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
        if hasattr(self, "_requested_evidence_idle"):
            # Native steps close in prepare(), after the ordinary stop check. Track
            # novelty here and consider the advisory only when a provider send is due.
            if progress:
                self._requested_evidence_idle = 0
                self.__dict__.pop("_requested_evidence_revision", None)
            elif charge_idle:
                self._requested_evidence_idle = min(self._requested_evidence_idle + 1, self.threshold)
        if progress:
            self.streak = 0
        elif charge_idle:
            self.streak += charge
        return self.streak >= self.threshold

    def _range_progress(self, binding: Mapping[str, object], receipt: Mapping[str, object],
                        revision: object) -> bool:
        """Fresh line range of an already-seen read path is progress; identical re-reads still charge."""
        key = self._range_identity(binding, receipt, revision)
        return key is not None and self._new(key)

    @staticmethod
    def _range_identity(binding: Mapping[str, object], receipt: Mapping[str, object],
                        revision: object) -> object:
        """Identity of a real (non-repeat) delivery's effective range, or None when not applicable."""
        delivery = receipt.get("delivery")
        delivery = delivery if isinstance(delivery, Mapping) else {}
        if (delivery.get("unchanged_since_request") is True
                or delivery.get("working_set_repeat_capture") is True):
            return None
        effective_range = delivery.get("effective_range")
        if not isinstance(effective_range, Mapping):
            return None
        path = receipt.get("target_path") or delivery.get("path")
        if not path:
            return None
        source_revision = delivery.get("revision") or receipt.get("source_revision")
        if source_revision is None:
            files = binding.get("product_files") or binding.get("harness_files")
            if isinstance(files, Mapping):
                source_revision = files.get(path)
        if source_revision is None:
            source_revision = revision
        rng = tuple(sorted((str(k), str(v)) for k, v in effective_range.items()))
        return ("read-range", path, source_revision, rng)

    @staticmethod
    def _identity(binding: Mapping[str, object], receipt: Mapping[str, object],
                  kind: str, result: str, revision: object) -> object:
        if kind in _OUTPUT_KINDS:
            # Raw output also feeds the legacy fingerprint; including it would
            # bypass normalization. New producers retain the full output digest
            # before clipping detail; old receipts fall back to their preview.
            output_digest = receipt.get("progress_output_digest")
            if (not isinstance(output_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", output_digest) is None):
                output_digest = _digest(receipt.get("detail", ""))
            return (kind, revision, receipt.get("target"), result, receipt.get("exit_code"),
                    output_digest)
        if kind == "read" and result == "HOST_OBSERVED":
            # Host-observed reads carry no target_path; identity is the observed target and its output.
            return ("read-observed", receipt.get("target"), _digest(receipt.get("detail", "")))
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
    batch = getattr(runtime, "native_batch", None)
    batch_origin = (getattr(batch, "origin", None)
                    if getattr(batch, "index", 0) >= 1 else None)
    batch_complete = (isinstance(batch_origin, str)
                      and getattr(batch, "index", 0) == getattr(batch, "size", -1)
                      and not getattr(batch, "pending", ())
                      and getattr(batch, "inflight", None) is None)
    if not watch.observe(binding, receipts, boundary=boundary,
                         charge_idle=True, batch_origin=batch_origin,
                         batch_complete=batch_complete):
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
