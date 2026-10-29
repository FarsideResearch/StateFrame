"""Opt-in R-126 bounded stagnation scoring and stop-control.

The cap is deliberately narrower than a general progress governor.  It counts
only host-closed read/probe actions, binds the window to the current workspace
revision, and leaves edits, verification, and ``done`` available.  The
default is inactive so callers that do not opt in do not acquire a new frame
field, journal event, or refusal path.
"""
from __future__ import annotations

import copy
import os
from collections import deque
from collections.abc import Mapping, Sequence

from kernel_measurement import digest


MODE_ENV = "KERNEL_STAGNATION_CAP"
SCHEMA = "stagnation-cap/1"
DEFAULT_WINDOW = 3
MAX_WINDOW = 8
MAX_OBJECTIVE_REVISION = 180
MAX_EVENTS = 8

# Keep this vocabulary aligned with the existing first-green collection/probe
# policy.  Evidence retrieval, dispute, edits, tests, and done remain outside
# the cap because they can change what the host knows or must preserve.
CAP_TOOLS = frozenset({
    "read", "grep", "find", "search", "probe", "predict",
})
_BROKEN_MENU_POLICIES = frozenset({
    "broken", "known_broken", "pre_r44", "known-broken", "known broken",
})


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the validated cap switch, which is ``off`` by default."""
    source = os.environ if environ is None else environ
    raw = source.get(MODE_ENV, "off")
    value = str(raw or "off").strip().lower()
    if value in {"0", "false", "off"}:
        return "off"
    if value in {"1", "true", "on"}:
        return "on"
    raise ValueError(f"{MODE_ENV} must be off or on")


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def _text(value: object, limit: int = 240) -> str:
    if value is None:
        return ""
    return (value if isinstance(value, str) else str(value))[:limit]


def _tool(event: Mapping[str, object], record: Mapping[str, object] | None = None) -> str:
    value = event.get("tool")
    if value in (None, "") and isinstance(record, Mapping):
        value = record.get("tool")
    return _text(value, 64).strip().lower()


def _binding_digest(value: object) -> str:
    if isinstance(value, str) and value:
        return value[:240]
    if isinstance(value, Mapping):
        return digest(value) if value else ""
    return ""


def _event_binding(event: Mapping[str, object], binding: object) -> str:
    explicit = event.get("binding_digest")
    if isinstance(explicit, str) and explicit:
        return explicit[:240]
    explicit = event.get("binding")
    if explicit not in (None, ""):
        return _binding_digest(explicit)
    return _binding_digest(binding)


def _validate_window(window: object) -> int:
    if isinstance(window, bool) or not isinstance(window, int):
        raise ValueError("stagnation cap window must be an integer")
    if not 1 <= window <= MAX_WINDOW:
        raise ValueError(f"stagnation cap window must be between 1 and {MAX_WINDOW}")
    return window


def _eligible(tool: object) -> bool:
    return _text(tool, 64).strip().lower() in CAP_TOOLS


class StagnationCap:
    """A bounded, revision-bound no-progress window.

    The object consumes closed host records only.  ``before_dispatch`` is a
    presentation/control seam: it refuses another read/probe once the window
    is unresolved, but never refuses an edit, test, or done request.
    """

    def __init__(self, objective_revision: object, *, window: int = DEFAULT_WINDOW,
                 enabled: bool | None = None):
        self.objective_revision = _text(objective_revision, MAX_OBJECTIVE_REVISION)
        if not self.objective_revision:
            raise ValueError("stagnation cap requires an objective revision")
        self.window = _validate_window(window)
        self.enabled = mode() == "on" if enabled is None else bool(enabled)
        self._binding_digest: str | None = None
        self._score = 0
        self._active = False
        self._events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self._windows_opened = 0
        self._blocked_actions = 0
        self._subsequent_recoveries = 0
        self._binding_changed_resets = 0
        self._last_reset_reason = "initial"

    def _reset(self, reason: str) -> None:
        self._score = 0
        self._active = False
        self._events.clear()
        self._last_reset_reason = _text(reason, 80)

    def observe(self, event: Mapping[str, object], *, record: Mapping[str, object] | None = None,
                binding: object = None) -> dict:
        """Fold one already-closed host event and return the bounded decision."""
        if not isinstance(event, Mapping):
            raise TypeError("stagnation cap events must be mappings")
        if not self.enabled:
            return self.public()

        current_binding = _event_binding(event, binding)
        if not current_binding:
            raise ValueError("stagnation cap events require a binding revision")
        binding_changed = (
            self._binding_digest is not None and current_binding != self._binding_digest
        )
        was_active = self._active
        if binding_changed:
            self._binding_changed_resets += 1
            self._reset("binding_change")
        self._binding_digest = current_binding

        meaningful = event.get("meaningful_progress") is True
        tool = _tool(event, record)
        if meaningful:
            if was_active and event.get("legitimate_recovery") is True:
                self._subsequent_recoveries += 1
            self._reset("binding_change" if binding_changed else "meaningful_progress")
        elif _eligible(tool):
            self._score = min(self.window, self._score + 1)
            self._events.append({
                "tool": tool,
                "binding_digest": self._binding_digest,
                "sequence": event.get("sequence", event.get("step")),
            })
            if self._score >= self.window:
                if not self._active:
                    self._windows_opened += 1
                self._active = True
        return self.public()

    def before_dispatch(self, tool: object, *, binding: object = None,
                        binding_digest: object = None) -> str | None:
        """Refuse only another capped read/probe on the current revision."""
        if not self.enabled or not self._active or not _eligible(tool):
            return None
        current = _binding_digest(binding_digest if binding_digest not in (None, "") else binding)
        if current and self._binding_digest and current != self._binding_digest:
            self._binding_changed_resets += 1
            self._reset("binding_change")
            self._binding_digest = current
            return None
        self._blocked_actions += 1
        return (
            "[stagnation-cap] UNRESOLVED: no revision-bound progress was observed "
            f"for {self.window} read/probe turns on the current workspace. "
            "This bounded window is stopped; do not infer success. Change the "
            "workspace or choose a different action; the host done gate remains "
            "authoritative."
        )

    def public(self) -> dict:
        """Return only bounded host facts suitable for a model-facing frame."""
        if not self.enabled:
            return {
                "schema": SCHEMA, "enabled": False, "status": "INACTIVE",
                "window": self.window, "score": 0,
            }
        return {
            "schema": SCHEMA,
            "enabled": True,
            "status": "UNRESOLVED" if self._active else "OPEN",
            "explicit_unresolved_stop": self._active,
            "objective_revision": self.objective_revision,
            "binding_digest": self._binding_digest or "",
            "window": self.window,
            "score": min(self._score, self.window),
            "windows_opened": self._windows_opened,
            "blocked_actions": self._blocked_actions,
            "subsequent_legitimate_recoveries": self._subsequent_recoveries,
            "recoveries_lost": 0,
            "binding_changed_resets": self._binding_changed_resets,
            "reset_on_binding_change": True,
            "last_reset_reason": self._last_reset_reason,
        }

    def snapshot(self) -> dict:
        """Return restart-safe state; no raw action arguments are persisted."""
        public = self.public()
        return copy.deepcopy(public)

    def restore(self, snapshot: Mapping[str, object]) -> None:
        if not isinstance(snapshot, Mapping) or snapshot.get("schema") != SCHEMA:
            raise ValueError("invalid stagnation cap snapshot")
        if snapshot.get("enabled") is not True or not self.enabled:
            raise ValueError("stagnation cap snapshot is not enabled")
        if _text(snapshot.get("objective_revision"), MAX_OBJECTIVE_REVISION) != self.objective_revision:
            raise ValueError("stagnation cap objective revision differs")
        window = _validate_window(snapshot.get("window"))
        if window != self.window:
            raise ValueError("stagnation cap window differs")
        binding = _text(snapshot.get("binding_digest"), 240)
        score = snapshot.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= window:
            raise ValueError("invalid stagnation cap score")
        status = _text(snapshot.get("status"), 32)
        if status not in {"OPEN", "UNRESOLVED"}:
            raise ValueError("invalid stagnation cap status")
        if not binding and (status != "OPEN" or score != 0):
            raise ValueError("stagnation cap snapshot has no binding")
        if status == "OPEN" and score >= window:
            raise ValueError("open stagnation cap snapshot is already at its window")
        if status == "UNRESOLVED" and score != window:
            raise ValueError("unresolved stagnation cap snapshot is below its window")
        unresolved = snapshot.get("explicit_unresolved_stop")
        if unresolved is not None:
            if type(unresolved) is not bool or unresolved != (status == "UNRESOLVED"):
                raise ValueError("invalid stagnation cap unresolved state")
        self._binding_digest = binding or None
        self._score = score
        self._active = status == "UNRESOLVED"
        counters = (
            "windows_opened", "blocked_actions", "subsequent_legitimate_recoveries",
            "binding_changed_resets",
        )
        values = {}
        for key in counters:
            value = snapshot.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid stagnation cap counter: {key}")
            values[key] = value
        self._windows_opened = values["windows_opened"]
        self._blocked_actions = values["blocked_actions"]
        self._subsequent_recoveries = values["subsequent_legitimate_recoveries"]
        self._binding_changed_resets = values["binding_changed_resets"]
        self._last_reset_reason = _text(snapshot.get("last_reset_reason"), 80)
        self._events.clear()


def _trace_metadata(trace: Mapping[str, object]) -> tuple[str, str]:
    if trace.get("post_r44") is not True:
        raise ValueError("stagnation calibration requires post-r44 traces")
    policy = _text(trace.get("menu_policy"), 80).strip().lower()
    if policy in _BROKEN_MENU_POLICIES:
        raise ValueError("stagnation calibration excludes the known-broken menu")
    if policy != "repaired":
        raise ValueError("stagnation calibration requires the repaired menu policy")
    match_key = _text(trace.get("match_key"), 180)
    if not match_key:
        raise ValueError("matched stagnation traces require match_key")
    return match_key, _text(trace.get("trace_id") or match_key, 180)


def score_trace(trace: Mapping[str, object], *, window: int = DEFAULT_WINDOW) -> dict:
    """Score one eligible post-r44 trace without treating model prose as evidence."""
    if not isinstance(trace, Mapping):
        raise TypeError("stagnation traces must be mappings")
    window = _validate_window(window)
    match_key, trace_id = _trace_metadata(trace)
    raw_events = trace.get("events")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise ValueError("stagnation trace events must be a sequence")
    cap = StagnationCap(
        _text(trace.get("objective_revision") or match_key, MAX_OBJECTIVE_REVISION),
        window=window, enabled=True)
    futile_turns = 0
    lost_recoveries = 0
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            raise ValueError("stagnation trace events must be mappings")
        tool = _tool(raw)
        meaningful = raw.get("meaningful_progress") is True
        event_binding = raw.get("binding_digest") or raw.get("binding")
        blocked = cap.before_dispatch(tool, binding=event_binding) is not None
        if blocked:
            # A refused dispatch has no host-closed event.  Do not fold it into
            # the cap or pretend that a delayed read actually recovered.
            if meaningful and raw.get("legitimate_recovery") is True:
                lost_recoveries += 1
            elif not meaningful:
                futile_turns += 1
            continue
        cap.observe(raw, record=raw, binding=raw.get("binding_digest") or raw.get("binding"))
    public = cap.public()
    if "capped_correct" not in trace or "capped_accepted_done" not in trace:
        raise ValueError(
            "stagnation trace requires observed capped_correct and capped_accepted_done"
        )
    baseline_correct = trace.get("correct")
    capped_correct = trace.get("capped_correct")
    if type(baseline_correct) is not bool or type(capped_correct) is not bool:
        raise ValueError("stagnation trace correctness fields must be booleans")
    if lost_recoveries:
        capped_correct = False
    correctness_non_inferior = capped_correct >= baseline_correct
    baseline_done = trace.get("accepted_done")
    capped_done = trace.get("capped_accepted_done")
    if (isinstance(baseline_done, bool) or not isinstance(baseline_done, int)
            or baseline_done < 0 or isinstance(capped_done, bool)
            or not isinstance(capped_done, int) or capped_done < 0):
        raise ValueError("stagnation trace accepted_done fields must be non-negative integers")
    return {
        "schema": SCHEMA,
        "trace_id": trace_id,
        "match_key": match_key,
        "window": window,
        "futile_turns_avoided": futile_turns,
        "subsequent_legitimate_recoveries": public["subsequent_legitimate_recoveries"],
        "recoveries_lost": lost_recoveries,
        "correctness_non_inferior": correctness_non_inferior,
        "baseline_correct": baseline_correct,
        "capped_correct": capped_correct,
        "baseline_accepted_done": baseline_done,
        "capped_accepted_done": capped_done,
        "accepted_done_not_inflated": capped_done <= baseline_done,
        "final_status": public["status"],
    }


def screen_cap(traces: Sequence[Mapping[str, object]], *,
               window: int = DEFAULT_WINDOW) -> dict:
    """Screen one cap on matched repaired-menu traces.

    The result is an audit summary, not an authority to mark a task done.  A
    cap is selected only when it loses no marked legitimate recovery and does
    not weaken correctness evidence.
    """
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)) or not traces:
        raise ValueError("stagnation cap screening requires matched traces")
    window = _validate_window(window)
    results = [score_trace(trace, window=window) for trace in traces]
    match_keys = {result["match_key"] for result in results}
    lost = sum(result["recoveries_lost"] for result in results)
    avoided = sum(result["futile_turns_avoided"] for result in results)
    recoveries = sum(result["subsequent_legitimate_recoveries"] for result in results)
    correctness = all(result["correctness_non_inferior"] for result in results)
    done_safe = all(result["accepted_done_not_inflated"] for result in results)
    selected = bool(correctness and done_safe and lost == 0)
    return {
        "schema": SCHEMA,
        "screened": True,
        "selected": selected,
        "window": window,
        "matched_trace_count": len(results),
        "matched_group_count": len(match_keys),
        "futile_turns_avoided": avoided,
        "subsequent_legitimate_recoveries": recoveries,
        "recoveries_lost": lost,
        "correctness_non_inferior": correctness,
        "accepted_done_not_inflated": done_safe,
        "traces": results,
    }
