"""Per-step StateFrame thinking routing and offline checkpoint replay.

The router is deliberately narrower than a general reasoning controller.  It
only looks at host-owned, typed StateFrame fields and can only disable thinking
for an explicitly allow-listed mechanical class.  A decision step always keeps
the provider's normal thinking policy.

The MiniCPM/llama.cpp wire mechanism used here is the same request shape used
by the existing fixer path: ``chat_template_kwargs.enable_thinking=False``.
For OpenAI-compatible providers the router also sends
``reasoning_effort="none"``.  ``reasoning_budget_tokens`` is intentionally not
used as the primary switch: a budget is a capacity setting, while the
chat-template flag is the explicit MiniCPM no-thinking control already present
in this repository.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence


MODE_ENV = "STATEFRAME_THINKING_ROUTER"
CLASSES_ENV = "STATEFRAME_THINKING_OFF_CLASSES"
VALID_MODES = frozenset({"off", "shadow", "on"})
REPLAY_SCHEMA = "stateframe-thinking-replay/1"
COMPARISON_SCHEMA = "stateframe-thinking-comparison/1"
CLASSIFICATION_EVENT = "thinking_router_classified"
_MECHANICAL_CLASSES = frozenset({
    "MECHANICAL_VERIFY",
    "MECHANICAL_READ",
    "TERMINAL_SUBMIT",
})


class StepClass(str, Enum):
    """The four routing classes emitted for a StateFrame step."""

    MECHANICAL_VERIFY = "MECHANICAL_VERIFY"
    MECHANICAL_READ = "MECHANICAL_READ"
    TERMINAL_SUBMIT = "TERMINAL_SUBMIT"
    DECISION = "DECISION"


@dataclass(frozen=True)
class StepClassification:
    """A typed classification and the host-owned signals that led to it."""

    step_class: StepClass
    reason: str
    signals: tuple[str, ...] = ()

    @property
    def class_name(self) -> str:
        return self.step_class.value

    def as_dict(self) -> dict[str, object]:
        return {
            "step_class": self.step_class.value,
            "reason": self.reason,
            "signals": list(self.signals),
        }


def thinking_router_mode() -> str:
    """Read and validate the independent router switch."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in VALID_MODES:
        raise ValueError(f"{MODE_ENV} must be off, shadow or on")
    return value


def _class_token(value: object) -> str:
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _coerce_step_class(value: object) -> StepClass:
    if isinstance(value, StepClass):
        return value
    token = _class_token(value)
    try:
        return StepClass[token]
    except KeyError as error:
        raise ValueError(f"unknown StateFrame thinking off class: {value!r}") from error


def thinking_off_classes() -> frozenset[StepClass]:
    """Return the explicit mechanical classes allowed to turn thinking off.

    ``DECISION`` is rejected even if named in the environment.  This makes the
    safety rule structural rather than dependent on every caller remembering
    to filter it.
    """
    raw = os.environ.get(CLASSES_ENV, "")
    result: set[StepClass] = set()
    for item in raw.split(","):
        if not item.strip():
            continue
        step_class = _coerce_step_class(item)
        if step_class.value not in _MECHANICAL_CLASSES:
            raise ValueError(
                f"{CLASSES_ENV} cannot contain {step_class.value}; "
                "it may contain only MECHANICAL_VERIFY, MECHANICAL_READ "
                "or TERMINAL_SUBMIT"
            )
        result.add(step_class)
    return frozenset(result)


def _normal_action(value: object) -> str:
    """Normalize a host enum-like action without interpreting prose."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


_VERIFY_ACTIONS = frozenset({
    "run_tests", "verify", "verify_current_source", "verify_changed_source",
    "test_changed_source", "test_the_changed_source", "mechanical_verify",
    "verify_or_repair_current_change",
})
_READ_ACTIONS = frozenset({
    "read", "read_file", "read_current_source", "read_requested_file",
    "refresh_read", "recall_evidence", "recall", "mechanical_read",
})
_TERMINAL_ACTIONS = frozenset({
    "submit", "terminal_submit", "finish", "done", "call_done",
    "review_public_requirements_then_done",
})


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _typed_action_values(frame: Mapping[str, object]) -> list[tuple[str, str]]:
    """Collect action enums from host-owned fields only.

    In particular, this function never traverses ``messages``, source text,
    model claims, hypotheses, observations, or free-form intent fields.
    """
    values: list[tuple[str, str]] = []

    def add(container_name: str, container: Mapping[str, object], keys: Sequence[str]) -> None:
        for key in keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                values.append((f"{container_name}.{key}", _normal_action(value)))

    add("frame", frame, ("next_action", "next_step", "phase"))
    for name, keys in (
        ("thinking_router", ("next_action", "next_step", "action", "tool", "kind")),
        ("pending", ("next_action", "next_step", "action", "tool", "kind")),
        ("resume", ("recommended_next_action", "next_action", "next_step")),
    ):
        container = _mapping(frame.get(name))
        if container is not None:
            add(name, container, keys)
    return values


def _green_lock_engaged(frame: Mapping[str, object]) -> bool:
    """Recognize only host-owned green-lock facts."""
    if isinstance(frame.get("green_lock_message"), str) and frame["green_lock_message"].strip():
        return True
    for name in ("green_lock", "terminal", "terminal_submission"):
        value = _mapping(frame.get(name))
        if value is None:
            continue
        if value.get("engaged") is True or value.get("terminal_engaged") is True:
            return True
        status = value.get("status")
        if isinstance(status, str) and _normal_action(status) in {
            "engaged", "green_lock_engaged", "ready_to_submit",
        }:
            return True
    return False


def _blocker_status_classification(frame: Mapping[str, object]) -> StepClassification | None:
    """Use the reducer-owned completion block before any terminal signal."""
    from kernel_completion_status import completion_status_from_frame, enabled
    if not enabled():
        return None
    status = completion_status_from_frame(frame)
    if status.state == "ACCEPTABLE":
        return None
    signals = ("completion_status", status.state)
    action = status.next_feasible_action
    action_name = _normal_action(
        action.get("tool", action.get("action", action.get("next_action", "")))
        if isinstance(action, Mapping) else "")
    if action_name in _VERIFY_ACTIONS:
        return StepClassification(
            StepClass.MECHANICAL_VERIFY,
            "completion_status requires a feasible verification action",
            (*signals, action_name),
        )
    if action_name in _READ_ACTIONS:
        return StepClassification(
            StepClass.MECHANICAL_READ,
            "completion_status requires a feasible read action",
            (*signals, action_name),
        )
    return StepClassification(
        StepClass.DECISION,
        "completion_status blocks terminal submission",
        signals,
    )


def _row_is_applied_edit(row: object) -> bool:
    if not isinstance(row, Mapping):
        return False
    result = row.get("result", row.get("result_class", row.get("observed_result")))
    if not (isinstance(result, str) and result.strip().upper() == "APPLIED"):
        return False
    tool = row.get("tool", row.get("kind", ""))
    if tool in (None, ""):
        return True
    return _normal_action(tool) in {"edit", "str_replace", "file_create", "replace_module"}


def _has_applied_edit(frame: Mapping[str, object]) -> bool:
    for key in ("recent_changes", "recent_edits"):
        rows = frame.get(key)
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            if any(_row_is_applied_edit(row) for row in rows):
                return True
    facts = _mapping(frame.get("facts"))
    if facts is not None:
        rows = facts.get("recent_edits")
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            if any(_row_is_applied_edit(row) for row in rows):
                return True
    for key in ("last_action", "last_receipt"):
        if _row_is_applied_edit(frame.get(key)):
            return True
    return False


def _has_failed_test(frame: Mapping[str, object]) -> bool:
    values: list[object] = []
    facts = _mapping(frame.get("facts"))
    if facts is not None:
        values.append(facts.get("last_test_exit"))
    world = _mapping(frame.get("world"))
    if world is not None:
        last_test = _mapping(world.get("last_test"))
        if last_test is not None:
            values.extend((last_test.get("exit"), last_test.get("exit_code")))
    last_test = _mapping(frame.get("last_test"))
    if last_test is not None:
        values.extend((last_test.get("exit"), last_test.get("exit_code")))
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value != 0:
            return True
        if isinstance(value, str) and _normal_action(value) in {
                "fail", "failed", "failure", "red", "error", "nonzero"}:
            return True
    return False


def classify_frame_state(frame: Mapping[str, object]) -> StepClassification:
    """Classify a step from the typed StateFrame, never from model text.

    Rules, in precedence order:

    * an engaged host green lock is ``TERMINAL_SUBMIT``;
    * a host ``run_tests``/verify next-action enum is ``MECHANICAL_VERIFY``;
    * a host read/recall next-action enum is ``MECHANICAL_READ``;
    * a current applied edit with no failed test after it is the mechanical
      verify hand-off;
    * all remaining states are ``DECISION``.
    """
    if not isinstance(frame, Mapping):
        raise TypeError("StateFrame classifier requires a mapping")

    blocker_classification = _blocker_status_classification(frame)
    if blocker_classification is not None:
        return blocker_classification

    if _green_lock_engaged(frame):
        return StepClassification(
            StepClass.TERMINAL_SUBMIT,
            "host green lock is engaged",
            ("green_lock_engaged",),
        )

    actions = _typed_action_values(frame)
    for source, action in actions:
        if action in _TERMINAL_ACTIONS:
            return StepClassification(
                StepClass.TERMINAL_SUBMIT,
                "host next-action enum requests terminal submission",
                (source, action),
            )
    for source, action in actions:
        if action in _VERIFY_ACTIONS:
            return StepClassification(
                StepClass.MECHANICAL_VERIFY,
                "host next-action enum requests verification",
                (source, action),
            )
    for source, action in actions:
        if action in _READ_ACTIONS:
            return StepClassification(
                StepClass.MECHANICAL_READ,
                "host next-action enum requests a read",
                (source, action),
            )

    if _has_applied_edit(frame) and not _has_failed_test(frame):
        return StepClassification(
            StepClass.MECHANICAL_VERIFY,
            "an applied edit lacks a failed-test decision signal",
            ("applied_edit",),
        )
    return StepClassification(StepClass.DECISION, "no mechanical host hand-off is present")


def apply_thinking_off(body: Mapping[str, object]) -> dict[str, object]:
    """Return a request copy with the explicit no-thinking provider knobs."""
    if not isinstance(body, Mapping):
        raise TypeError("request body must be a mapping")
    result = copy.deepcopy(dict(body))
    raw_kwargs = result.get("chat_template_kwargs")
    if raw_kwargs is None:
        kwargs: dict[str, object] = {}
    elif isinstance(raw_kwargs, Mapping):
        kwargs = copy.deepcopy(dict(raw_kwargs))
    else:
        raise ValueError("chat_template_kwargs must be a mapping when present")
    kwargs["enable_thinking"] = False
    result["chat_template_kwargs"] = kwargs
    # This is understood by OpenAI-compatible reasoning providers; unknown
    # extra JSON fields are already tolerated by the existing request path.
    result["reasoning_effort"] = "none"
    return result


class ThinkingRouter:
    """Run-scoped classifier, journal writer, and request-body policy."""

    def __init__(self, mode: str | None = None, *, off_classes=None, journal=None):
        self.mode = thinking_router_mode() if mode is None else str(mode).strip().lower()
        if self.mode not in VALID_MODES:
            raise ValueError("invalid StateFrame thinking router mode")
        if off_classes is None:
            off_classes = thinking_off_classes()
        parsed = frozenset(_coerce_step_class(item) for item in off_classes)
        invalid = next((item for item in parsed
                        if item.value not in _MECHANICAL_CLASSES), None)
        if invalid is not None:
            raise ValueError(
                f"thinking off classes cannot contain {invalid.value}; must be mechanical")
        self.off_classes = parsed
        self.journal = journal

    def classify(self, frame: Mapping[str, object], *, step: object = None,
                 request_id: str | None = None, snapshot_id: str | None = None
                 ) -> StepClassification:
        classification = classify_frame_state(frame)
        if self.mode != "off" and self.journal is not None:
            self.journal.emit(
                CLASSIFICATION_EVENT,
                step=step,
                step_class=classification.step_class.value,
                reason=classification.reason,
                signals=list(classification.signals),
                mode=self.mode,
                allowed=(self.mode == "on" and
                         classification.step_class in self.off_classes),
                request_id=request_id,
                snapshot_id=snapshot_id,
            )
        return classification

    def should_disable(self, classification: StepClassification | StepClass | str) -> bool:
        if isinstance(classification, StepClassification):
            step_class = classification.step_class
        else:
            step_class = _coerce_step_class(classification)
        return self.mode == "on" and step_class in self.off_classes

    def prepare_body(self, body: Mapping[str, object],
                     classification: StepClassification | StepClass | str) -> dict[str, object] | Mapping[str, object]:
        if not self.should_disable(classification):
            return body
        return apply_thinking_off(body)

    def bind(self, capture: object, classification: StepClassification) -> None:
        """Attach this policy to a RequestCapture without changing its class."""
        setattr(capture, "thinking_router", self)
        setattr(capture, "thinking_router_classification", classification)


def _load_events(source: object) -> tuple[list[dict[str, object]], str | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        from kernel_measurement import resolve_journal_blob_refs

        events: list[dict[str, object]] = []
        blob_cache: dict[str, object] = {}
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as error:
            raise ValueError(f"cannot read event journal: {path}") from error
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from error
                if not isinstance(value, Mapping):
                    raise ValueError(f"event at {path}:{line_number} is not an object")
                try:
                    value = resolve_journal_blob_refs(value, path, blob_cache)
                except (OSError, ValueError) as error:
                    raise ValueError(f"invalid blob reference at {path}:{line_number}") from error
                if not isinstance(value, Mapping):
                    raise ValueError(f"event at {path}:{line_number} did not resolve to an object")
                events.append(dict(value))
        return events, str(path)
    if isinstance(source, Mapping):
        nested = source.get("events")
        if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
            return [dict(item) for item in nested if isinstance(item, Mapping)], None
        raise ValueError("event source mapping must contain an events sequence")
    if isinstance(source, Iterable):
        events = []
        for item in source:
            if not isinstance(item, Mapping):
                raise ValueError("all replay events must be objects")
            events.append(dict(item))
        return events, None
    raise TypeError("replay events must be a JSONL path or an iterable of mappings")


def build_replay_manifest(source: object) -> dict[str, object]:
    """Extract classified exact request bodies from an audit events JSONL."""
    events, source_name = _load_events(source)
    by_request: dict[str, str] = {}
    by_step: dict[object, set[str]] = {}
    run_id = None
    for event in events:
        run_id = run_id or event.get("run_id")
        if event.get("kind") != CLASSIFICATION_EVENT:
            continue
        step_class = event.get("step_class", event.get("class"))
        try:
            parsed = _coerce_step_class(step_class)
        except (TypeError, ValueError):
            raise ValueError("classification event has an invalid step class") from None
        request_id = event.get("request_id")
        step = event.get("step")
        if isinstance(request_id, str) and request_id:
            by_request[request_id] = parsed.value
        if step is not None:
            by_step.setdefault(step, set()).add(parsed.value)

    rows: dict[tuple[str, object], dict[str, object]] = {}
    request_events = 0
    for event in events:
        if event.get("kind") != "request_submitted":
            continue
        request_events += 1
        request_id = event.get("request_id")
        step = event.get("step")
        step_class = by_request.get(request_id) if isinstance(request_id, str) else None
        if step_class is None:
            candidates = by_step.get(step, set())
            if len(candidates) == 1:
                step_class = next(iter(candidates))
        # Child requests (for example a public review) have no router
        # classification and are intentionally outside the per-step manifest.
        if step_class is None:
            continue
        payload = event.get("payload", event.get("request_body"))
        if not isinstance(payload, Mapping):
            raise ValueError("classified request is missing its exact payload")
        key = (str(request_id or ""), step)
        rows[key] = {
            "step": step,
            "class": step_class,
            "request_id": request_id,
            "request": copy.deepcopy(dict(payload)),
        }

    if request_events and not rows:
        raise ValueError("request journal contains no classification for its requests")
    steps = list(rows.values())
    steps.sort(key=lambda row: (
        row.get("step") is None,
        row.get("step") if isinstance(row.get("step"), (int, float, str)) else "",
    ))
    manifest: dict[str, object] = {
        "schema": REPLAY_SCHEMA,
        "run_id": run_id,
        "steps": steps,
    }
    if source_name is not None:
        manifest["source"] = source_name
    return manifest


extract_replay_manifest = build_replay_manifest
checkpoint_replay_manifest = build_replay_manifest


def _action_from_response(response: object) -> dict[str, object] | None:
    if not isinstance(response, Mapping):
        return None
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return None
    calls = message.get("tool_calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)) and calls:
        normalized = []
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            arguments: object = function.get("arguments", "")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            normalized.append({"name": function.get("name"), "arguments": arguments})
        action = {"kind": "tool_calls", "calls": normalized}
        if len(normalized) == 1:
            action["name"] = normalized[0]["name"]
            action["arguments"] = normalized[0]["arguments"]
        return action
    content = message.get("content")
    return {"kind": "content", "content": content if isinstance(content, str) else content}


def _response_rows(source: object) -> dict[object, dict[str, object]]:
    events, _ = _load_events(source)
    classes_by_request: dict[str, str] = {}
    classes_by_step: dict[object, str] = {}
    for event in events:
        if event.get("kind") != CLASSIFICATION_EVENT:
            continue
        value = event.get("step_class", event.get("class"))
        try:
            value = _coerce_step_class(value).value
        except (TypeError, ValueError):
            continue
        request_id = event.get("request_id")
        step = event.get("step")
        if isinstance(request_id, str):
            classes_by_request[request_id] = value
        if step is not None:
            classes_by_step[step] = value
    rows: dict[object, dict[str, object]] = {}
    for event in events:
        if event.get("kind") == "response_received":
            request_id = event.get("request_id")
            step = event.get("step")
            if isinstance(request_id, str) and request_id in classes_by_request:
                step_class = classes_by_request[request_id]
            elif step in classes_by_step:
                step_class = classes_by_step[step]
            else:
                # A comparison can also be run against a GPU replay tape that
                # lacks router events; then the step itself remains useful.
                step_class = None
            action = _action_from_response(event.get("response"))
            if action is None:
                continue
            rows[step] = {"step": step, "class": step_class, "action": action,
                          "request_id": request_id}
    return rows


def _manifest_action_rows(source: object) -> dict[object, dict[str, object]] | None:
    if isinstance(source, Mapping) and isinstance(source.get("steps"), Sequence):
        rows = {}
        for entry in source["steps"]:
            if not isinstance(entry, Mapping) or "step" not in entry:
                continue
            action = entry.get("chosen_action", entry.get("action"))
            if action is not None:
                rows[entry["step"]] = {
                    "step": entry["step"], "class": entry.get("class"), "action": action,
                    "request_id": entry.get("request_id"),
                }
        return rows
    return None


def compare_chosen_actions(left: object, right: object) -> dict[str, object]:
    """Compare chosen tool/action outputs from two event tapes or replay results."""
    left_rows = _manifest_action_rows(left) or _response_rows(left)
    right_rows = _manifest_action_rows(right) or _response_rows(right)
    steps = sorted(set(left_rows) | set(right_rows), key=lambda value: (
        value is None, value if isinstance(value, (int, float, str)) else ""))
    comparisons = []
    differences = []
    for step in steps:
        lrow, rrow = left_rows.get(step), right_rows.get(step)
        left_action = lrow.get("action") if lrow else None
        right_action = rrow.get("action") if rrow else None
        row = {
            "step": step,
            "left_class": lrow.get("class") if lrow else None,
            "right_class": rrow.get("class") if rrow else None,
            "left_action": left_action,
            "right_action": right_action,
            "same": left_action == right_action,
        }
        comparisons.append(row)
        if not row["same"]:
            differences.append(row)
    return {
        "schema": COMPARISON_SCHEMA,
        "same": not differences,
        "steps": comparisons,
        "differences": differences,
    }


compare_replay_actions = compare_chosen_actions


def _write_json(value: object, output: str | None) -> None:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or compare StateFrame thinking replay tapes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("replay", help="extract exact classified requests from events.jsonl")
    replay.add_argument("events", type=Path)
    replay.add_argument("-o", "--output")
    compare = subparsers.add_parser("compare", help="compare chosen actions from two event/replay files")
    compare.add_argument("left", type=Path)
    compare.add_argument("right", type=Path)
    compare.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    if args.command == "replay":
        _write_json(build_replay_manifest(args.events), args.output)
        return 0
    left = _load_json_or_events(args.left)
    right = _load_json_or_events(args.right)
    _write_json(compare_chosen_actions(left, right), args.output)
    return 0


def _load_json_or_events(path: Path) -> object:
    if path.suffix == ".jsonl":
        return path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path


if __name__ == "__main__":  # pragma: no cover - CLI exercised by operator
    raise SystemExit(main())


__all__ = [
    "CLASSIFICATION_EVENT", "CLASSES_ENV", "COMPARISON_SCHEMA", "MODE_ENV",
    "REPLAY_SCHEMA", "StepClass", "StepClassification", "ThinkingRouter",
    "apply_thinking_off", "build_replay_manifest", "checkpoint_replay_manifest",
    "classify_frame_state", "compare_chosen_actions", "compare_replay_actions",
    "extract_replay_manifest", "main", "thinking_off_classes", "thinking_router_mode",
]
