"""Bounded model-to-host decision continuity for StateFrame requests.

The continuity block is deliberately independent from the settled decision
cache.  It carries only a small, model-owned working note and recent action
outcomes; the host validates evidence references and never turns the note into
an authoritative fact.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence

from kernel_context import token_estimate


MODE_ENV = "STATEFRAME_DECISION_CONTINUITY"
SCHEMA = "stateframe-decision-continuity/1"
BLOCK_HEADER = "[Decision continuity]"
DELTA_FIELD = "decision_delta"
MAX_EXCHANGES = 2
MAX_RENDER_TOKENS = 384
MAX_HYPOTHESIS_CHARS = 240
MAX_ACTION_CHARS = 180
MAX_REJECTED_CHARS = 180
MAX_EVIDENCE_REFS = 8
MAX_REJECTED = 4
TYPE_WORKING_HYPOTHESIS = "WORKING_HYPOTHESIS"
TYPE_REJECTED_HYPOTHESIS = "REJECTED_HYPOTHESIS"
TYPE_DISCRIMINATING_ACTION = "DISCRIMINATING_ACTION"
TYPE_ACTION_CALL = "HOST_ACTION_CALL"
TYPE_ACTION_RESULT = "HOST_ACTION_RESULT"
RESUME_PREDICATE_KEYS = tuple(f"R{index}" for index in range(1, 9))
RESUME_PREDICATE_VALUES = frozenset({"TRUE", "FALSE", "UNKNOWN"})


class DecisionContinuityError(ValueError):
    """A malformed model delta that the host must not install."""


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value in {"0", "off"}:
        configured = "off"
    elif value in {"1", "on"}:
        configured = "on"
    elif value == "slots":
        configured = "slots"
    else:
        raise ValueError(f"{MODE_ENV} must be 0/1, off/on, or slots")
    # F4's handoff reads the five typed slots.  Select that continuity organ
    # automatically so STATEFRAME_F4=on is self-contained; the legacy mode is
    # unchanged when F4 is off.
    if os.environ.get("STATEFRAME_F4", "off").strip().lower() == "on":
        if configured == "on":
            raise ValueError(
                "STATEFRAME_F4 requires STATEFRAME_DECISION_CONTINUITY=off or slots")
        return "slots"
    return configured


def enabled() -> bool:
    return mode() in {"on", "slots"}


def delta_schema() -> dict[str, object]:
    """Return the optional, closed action-side delta contract."""
    if mode() == "slots":
        from kernel_decision_continuity_slots import delta_schema as slots_schema
        return slots_schema()
    def text(maximum: int) -> dict[str, object]:
        return {"type": "string", "minLength": 1, "maxLength": maximum}
    return {
        "type": "object",
        "properties": {
            "hypothesis": text(MAX_HYPOTHESIS_CHARS),
            "evidence_refs": {
                "type": "array", "maxItems": MAX_EVIDENCE_REFS,
                "items": text(180),
            },
            "rejected_hypotheses": {
                "type": "array", "maxItems": MAX_REJECTED,
                "items": text(MAX_REJECTED_CHARS),
            },
            "next_action": text(MAX_ACTION_CHARS),
        },
        "additionalProperties": False,
        "minProperties": 1,
    }


action_delta_schema = delta_schema


def _string(value: object, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise DecisionContinuityError(
            f"{field} must be a non-empty string of at most {maximum} characters")
    return value.strip()


def _string_list(value: object, field: str, maximum_items: int,
                 maximum_chars: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise DecisionContinuityError(
            f"{field} must be a list of at most {maximum_items} strings")
    result: list[str] = []
    for item in value:
        item = _string(item, f"{field} item", maximum_chars)
        if item in result:
            raise DecisionContinuityError(f"{field} contains duplicate values")
        result.append(item)
    return result


def normalize_delta(value: object) -> dict[str, object]:
    """Validate and copy one model-authored patch without trusting its refs."""
    if mode() == "slots":
        from kernel_decision_continuity_slots import normalize_delta as slots_normalize
        try:
            return slots_normalize(value)
        except ValueError as error:
            raise DecisionContinuityError(str(error)) from error
    if not isinstance(value, Mapping):
        raise DecisionContinuityError("decision_delta must be an object")
    allowed = {"hypothesis", "evidence_refs", "rejected_hypotheses", "next_action"}
    unknown = set(value) - allowed
    if unknown:
        raise DecisionContinuityError(
            "decision_delta contains unknown fields: " + ", ".join(sorted(map(str, unknown))))
    if not value:
        raise DecisionContinuityError("decision_delta must contain at least one field")
    result: dict[str, object] = {}
    if "hypothesis" in value:
        result["hypothesis"] = _string(value["hypothesis"], "hypothesis",
                                        MAX_HYPOTHESIS_CHARS)
    if "evidence_refs" in value:
        result["evidence_refs"] = _string_list(
            value["evidence_refs"], "evidence_refs", MAX_EVIDENCE_REFS, 180)
    if "rejected_hypotheses" in value:
        result["rejected_hypotheses"] = _string_list(
            value["rejected_hypotheses"], "rejected_hypotheses", MAX_REJECTED,
            MAX_REJECTED_CHARS)
    if "next_action" in value:
        result["next_action"] = _string(value["next_action"], "next_action",
                                          MAX_ACTION_CHARS)
    return result


def split_action_arguments(arguments: object, *, call: Mapping[str, object] | None = None,
                           message: Mapping[str, object] | None = None
                           ) -> tuple[object, object | None]:
    """Remove the optional delta from ordinary tool args only when enabled."""
    if mode() == "slots":
        from kernel_decision_continuity_slots import (
            split_action_arguments as slots_split_action_arguments,
        )
        return slots_split_action_arguments(arguments, call=call, message=message)
    if not enabled() or not isinstance(arguments, Mapping):
        return arguments, None
    ordinary = copy.deepcopy(dict(arguments))
    delta = ordinary.pop(DELTA_FIELD, None)
    for container in (call, message):
        if isinstance(container, Mapping) and DELTA_FIELD in container:
            delta = container[DELTA_FIELD]
    return ordinary, copy.deepcopy(delta) if delta is not None else None


def _function_parameters(tool: Mapping[str, object]) -> dict[str, object] | None:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return None
    parameters = function.get("parameters")
    return copy.deepcopy(dict(parameters)) if isinstance(parameters, Mapping) else None


def tool_surface(tools: object) -> object:
    """Add an optional delta field to tool schemas without mutating the menu."""
    if mode() == "slots":
        from kernel_decision_continuity_slots import tool_surface as slots_tool_surface
        return slots_tool_surface(tools)
    if not enabled() or not isinstance(tools, list):
        return tools
    surfaced = copy.deepcopy(tools)
    schema = delta_schema()
    for tool in surfaced:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        parameters = _function_parameters(tool) or {
            "type": "object", "properties": {}, "additionalProperties": False,
        }
        properties = parameters.get("properties")
        parameters["properties"] = (copy.deepcopy(dict(properties))
                                     if isinstance(properties, Mapping) else {})
        parameters["properties"][DELTA_FIELD] = schema
        function["parameters"] = parameters
    from kernel_tool_schema_diet import filter_tools as filter_single_carrier_tools
    return filter_single_carrier_tools(surfaced)


def _clip(value: object, maximum: int) -> str:
    text = str(value or "")
    if len(text) <= maximum:
        return text
    return text[:max(0, maximum - 1)] + "…"


def _compact_value(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(value, 96)
    if depth >= 2:
        return _clip(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                default=str), 96)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in list(value)[:6]:
            result[_clip(key, 48)] = _compact_value(value[key], depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_compact_value(item, depth=depth + 1) for item in list(value)[:6]]
    return _clip(value, 96)


def _record_ref(record: Mapping[str, object]) -> str:
    for key in ("step_record_id", "record_id", "receipt_id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    refs = record.get("receipt_ids")
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        for ref in refs:
            if isinstance(ref, str) and ref:
                return ref
    step = record.get("step")
    return f"step:{step}" if step is not None else "exchange"


def _exchange(record: Mapping[str, object], pending: Mapping[str, object] | None) -> dict[str, object]:
    pending = pending if isinstance(pending, Mapping) else {}
    arguments = pending.get("arguments")
    if not isinstance(arguments, Mapping):
        arguments = record.get("arguments", record.get("action_parameters", {}))
    if not isinstance(arguments, Mapping):
        arguments = {}
    refs = record.get("receipt_ids", ())
    receipt_refs = ([str(ref) for ref in list(refs)[:4] if str(ref).strip()]
                    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)) else [])
    singular_ref = record.get("receipt_id")
    if isinstance(singular_ref, str) and singular_ref and singular_ref not in receipt_refs:
        receipt_refs.insert(0, singular_ref)
    receipt_refs = receipt_refs[:4]
    tool = record.get("tool", pending.get("tool", ""))
    result_class = record.get("result", record.get("result_class", "UNKNOWN"))
    return {
        "record_ref": _clip(_record_ref(record), 96),
        "step": record.get("step"),
        "call": {"type": TYPE_ACTION_CALL, "tool": _clip(tool, 64),
                 "args": _compact_value(arguments)},
        "result": {
            "type": TYPE_ACTION_RESULT,
            "class": _clip(result_class, 64),
            "receipt_refs": receipt_refs,
            "detail": _clip(record.get("observation", record.get("detail", "")), 120),
        },
    }


def _empty_cache() -> dict[str, object]:
    return {
        "hypothesis": None,
        "evidence_refs": [],
        "rejected_hypotheses": [],
        "next_discriminating_action": None,
    }


def _typed_hypothesis(text: object, source: str = "model_delta") -> dict[str, str] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "type": TYPE_WORKING_HYPOTHESIS,
        "status": "HYPOTHESIS",
        "text": _clip(text.strip(), MAX_HYPOTHESIS_CHARS),
        "source": _clip(source, 48),
    }


def _typed_rejected(text: object, source: str = "model_delta",
                    evidence_refs: Sequence[object] = ()) -> dict[str, object] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    return {
        "type": TYPE_REJECTED_HYPOTHESIS,
        "status": "REJECTED",
        "text": _clip(text.strip(), MAX_REJECTED_CHARS),
        "source": _clip(source, 48),
        "evidence_refs": [str(ref) for ref in evidence_refs
                          if isinstance(ref, str) and ref.strip()][:2],
    }


def _typed_resume_predicates(value: object) -> dict[str, str] | None:
    """Keep only the shadow's host-computed R1-R8 tri-state values."""
    if not isinstance(value, Mapping):
        return None
    result = {
        key: str(value[key]) for key in RESUME_PREDICATE_KEYS
        if isinstance(value.get(key), str) and value[key] in RESUME_PREDICATE_VALUES
    }
    return result or None


def _typed_action(text: object, source: str = "model_delta", *,
                 resume_predicates: object = None) -> dict[str, object] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    result: dict[str, object] = {
        "type": TYPE_DISCRIMINATING_ACTION,
        "status": "ADVISORY",
        "text": _clip(text.strip(), MAX_ACTION_CHARS),
        "source": _clip(source, 48),
    }
    predicates = _typed_resume_predicates(resume_predicates)
    if predicates is not None:
        result["resume_predicates"] = predicates
    return result


def _object_dict(value: object) -> dict[str, object] | None:
    """Turn one existing organ's mapping/dataclass view into a safe mapping."""
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except Exception:  # pragma: no cover - advisory adapter boundary
            return None
        return dict(result) if isinstance(result, Mapping) else None
    return None


def _organ_text(value: object, *, keys: Sequence[str] = ()) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    mapping = _object_dict(value)
    if mapping is None:
        return ""
    for key in keys:
        candidate = mapping.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return " ".join(candidate.split())
    return ""


def _organ_refs(value: object) -> list[str]:
    """Extract only explicit receipt-like references; validation happens in the store."""
    result: list[str] = []
    if isinstance(value, Mapping):
        for key in ("receipt_ids", "evidence_refs", "provenance"):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                nested = nested.get("receipt_ids", nested.get("evidence_refs", ()))
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                result.extend(str(item) for item in nested if str(item).strip())
        for key in ("receipt_id", "step_record_id", "event_id"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                result.append(item)
    return list(dict.fromkeys(result))


def _negative_entry(value: object, *, source: str) -> tuple[str, list[str]] | None:
    mapping = _object_dict(value)
    if mapping is None:
        text = _organ_text(value)
        return (text, []) if text else None
    text = _organ_text(mapping, keys=("text", "detail", "reason", "action_signature", "kind"))
    if not text:
        return None
    return text, _organ_refs(mapping)


def project_organ_state(*, decision_state: object = None, disposition: object = None,
                        rejected_candidate: object = None, hypotheses: object = None,
                        composed: object = None, frame: Mapping[str, object] | None = None,
                        assistant_statement: object = None) -> dict[str, object]:
    """Project the already-landed cognition organs into typed cache candidates.

    This is an adapter only.  It never invents a receipt, turns a model statement
    into a fact, or invokes a selector.  The selector result captured by the
    composed organ is read as an advisory action; the store validates its refs.
    Callers should run this after an action/receipt boundary, not as a prose
    regeneration step for every render.
    """
    projection: dict[str, object] = {"evidence_refs": []}

    # KERNEL_DECISION_STATE is model-owned state.  Its explicit note/objective is
    # still a hypothesis at this seam; it is never copied into verified facts.
    state = decision_state
    if state is not None:
        notes = getattr(state, "notes", {})
        items = getattr(state, "items", {})
        values = list(notes.values()) if isinstance(notes, Mapping) else []
        values += [item for item in (items.values() if isinstance(items, Mapping) else ())]
        values.sort(key=lambda item: (0 if item.get("status") == "active" else 1,
                                      -int(item.get("updated_step", 0) or 0))
                   if isinstance(item, Mapping) else (2, 0))
        for item in values:
            text = _organ_text(item, keys=("text", "objective", "next_action"))
            if text:
                projection["hypothesis"] = _typed_hypothesis(text, "KERNEL_DECISION_STATE")
                projection["evidence_refs"] = _organ_refs(item)
                break

    # STATEFRAME_DISPOSITION v2 contributes a candidate blocker as an explicitly
    # uncertain working theory, never as a completion fact.
    if "hypothesis" not in projection and disposition is not None:
        blocker = getattr(disposition, "refusal_blocker", None)
        if isinstance(blocker, str) and blocker.strip():
            projection["hypothesis"] = _typed_hypothesis(
                "Candidate may be blocked by: " + blocker, "STATEFRAME_DISPOSITION")

    # KERNEL_ASSISTANT_STATEMENT is historical model text.  It is a fallback only
    # when no explicit decision state/disposition supplied the working hypothesis.
    if "hypothesis" not in projection:
        statement = _organ_text(assistant_statement, keys=("text",))
        if statement:
            projection["hypothesis"] = _typed_hypothesis(
                statement, "KERNEL_ASSISTANT_STATEMENT")

    rejected: list[dict[str, object]] = []
    tracker = hypotheses
    if tracker is not None:
        for method_name, source in (("kill_list", "hypotheses.kill_list"),
                                    ("negative_work", "hypotheses.negative_work")):
            method = getattr(tracker, method_name, None)
            if not callable(method):
                continue
            try:
                values = method()
            except Exception:  # pragma: no cover - advisory adapter boundary
                values = ()
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                for value in values:
                    entry = _negative_entry(value, source=source)
                    if entry:
                        item = _typed_rejected(entry[0], source, entry[1])
                        if item:
                            rejected.append(item)

    # The offline shadow's append-only negative_work is consumed when a recorded
    # projection is available; no live shadow import or execution is introduced.
    shadow_predicates: dict[str, str] | None = None
    if isinstance(frame, Mapping):
        for value in frame.get("negative_work", ()) or ():
            entry = _negative_entry(value, source="stateframe_shadow.negative_work")
            if entry:
                item = _typed_rejected(entry[0], "stateframe_shadow.negative_work", entry[1])
                if item:
                    rejected.append(item)
        resume = frame.get("resume")
        if isinstance(resume, Mapping):
            shadow_predicates = _typed_resume_predicates(
                resume.get("predicates", resume))
            recommended = resume.get("recommended_next_action")
            if isinstance(recommended, str) and recommended.strip():
                projection["next_discriminating_action"] = _typed_action(
                    recommended, "stateframe_shadow.resume",
                    resume_predicates=shadow_predicates)
    if not rejected and composed is not None:
        for value in getattr(composed, "_negative", ()) or ():
            entry = _negative_entry(value, source="composed_cognition.negative_work")
            if entry:
                item = _typed_rejected(entry[0], "composed_cognition.negative_work", entry[1])
                if item:
                    rejected.append(item)

    # KERNEL_REJECTED_CANDIDATE is a rejected, unverified proposal.  Keep it in
    # the same bounded negative channel so it cannot be mistaken for source truth.
    candidate = getattr(rejected_candidate, "candidate", None) if rejected_candidate is not None else None
    if isinstance(candidate, Mapping):
        candidate_text = _organ_text(candidate, keys=("feedback", "instruction", "tool", "path"))
        if candidate_text:
            item = _typed_rejected(candidate_text, "KERNEL_REJECTED_CANDIDATE",
                                   _organ_refs(candidate))
            if item:
                rejected.append(item)

    if rejected:
        unique: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for item in rejected:
            key = (str(item.get("source", "")), str(item.get("text", "")))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        projection["rejected_hypotheses"] = unique[:MAX_REJECTED]

    # composed_cognition already called select_discriminating_action at its receipt
    # boundary.  Read that typed result instead of re-running the selector on each
    # request; frontier() remains the fallback context for an abstaining selector.
    selection = getattr(composed, "_last_selection", None) if composed is not None else None
    selection_map = _object_dict(selection)
    if selection_map is not None:
        selected = selection_map.get("selected_action_id") or selection_map.get("baseline_action_id")
        if isinstance(selected, str) and selected.strip():
            projection["next_discriminating_action"] = _typed_action(
                selected, "composed_cognition.select_discriminating_action",
                resume_predicates=shadow_predicates)
        projection["evidence_refs"] = list(dict.fromkeys(
            [*projection.get("evidence_refs", ()), *_organ_refs(selection_map)]))
    if "next_discriminating_action" not in projection and tracker is not None:
        frontier = getattr(tracker, "frontier", None)
        if callable(frontier):
            try:
                values = frontier()
            except Exception:  # pragma: no cover - advisory adapter boundary
                values = ()
            if isinstance(values, Sequence) and values:
                projection["next_discriminating_action"] = _typed_action(
                    values[0], "hypotheses.frontier",
                    resume_predicates=shadow_predicates)
    return projection


class DecisionContinuityStore:
    """Two-exchange host ledger plus a patchable model-owned decision note."""

    def __new__(cls, *args: object, **kwargs: object) -> "DecisionContinuityStore":
        if cls is DecisionContinuityStore and mode() == "slots":
            from kernel_decision_continuity_slots import SlotDecisionContinuityStore
            return SlotDecisionContinuityStore(*args, **kwargs)  # type: ignore[return-value]
        return super().__new__(cls)

    def __init__(self, *, exchanges: Sequence[Mapping[str, object]] = (),
                 cache: Mapping[str, object] | None = None):
        self._exchanges = [copy.deepcopy(dict(item)) for item in exchanges
                           if isinstance(item, Mapping)][-MAX_EXCHANGES:]
        self._cache = _empty_cache()
        if isinstance(cache, Mapping):
            self._restore_cache(cache)

    def _restore_cache(self, cache: Mapping[str, object]) -> None:
        hypothesis = cache.get("hypothesis")
        if isinstance(hypothesis, Mapping) and isinstance(hypothesis.get("text"), str):
            self._cache["hypothesis"] = _typed_hypothesis(
                hypothesis["text"], str(hypothesis.get("source", "model_delta")))
        elif isinstance(hypothesis, str) and hypothesis.strip():
            self._cache["hypothesis"] = _typed_hypothesis(hypothesis)
        refs = cache.get("evidence_refs")
        if isinstance(refs, list):
            self._cache["evidence_refs"] = [str(ref) for ref in refs[:MAX_EVIDENCE_REFS]
                                             if str(ref).strip()]
        rejected = cache.get("rejected_hypotheses")
        if isinstance(rejected, list):
            restored: list[dict[str, object]] = []
            for item in rejected[:MAX_REJECTED]:
                if isinstance(item, Mapping):
                    value = _typed_rejected(
                        item.get("text"), str(item.get("source", "model_delta")),
                        item.get("evidence_refs", ()))
                else:
                    value = _typed_rejected(item)
                if value is not None:
                    restored.append(value)
            self._cache["rejected_hypotheses"] = restored
        next_action = cache.get("next_discriminating_action", cache.get("next_action"))
        if isinstance(next_action, Mapping):
            self._cache["next_discriminating_action"] = _typed_action(
                next_action.get("text"), str(next_action.get("source", "model_delta")),
                resume_predicates=next_action.get("resume_predicates"))
        elif isinstance(next_action, str) and next_action.strip():
            self._cache["next_discriminating_action"] = _typed_action(next_action)

    @classmethod
    def from_state(cls, state: Mapping[str, object], *,
                   host_envelope: Mapping[str, object] | None = None,
                   existing_refs: set[str] | Sequence[str] = ()) -> "DecisionContinuityStore":
        if mode() == "slots":
            from kernel_decision_continuity_slots import SlotDecisionContinuityStore
            return SlotDecisionContinuityStore.from_state(
                state, host_envelope=host_envelope, existing_refs=existing_refs
            )  # type: ignore[return-value]
        exchanges = state.get("recent_exchanges", ())
        cache = state.get("cache")
        return cls(exchanges=exchanges if isinstance(exchanges, list) else (),
                   cache=cache if isinstance(cache, Mapping) else None)

    @property
    def exchanges(self) -> tuple[dict[str, object], ...]:
        return tuple(copy.deepcopy(self._exchanges))

    @property
    def cache(self) -> dict[str, object]:
        return copy.deepcopy(self._cache)

    @property
    def has_content(self) -> bool:
        return bool(self._exchanges or any(
            value not in (None, [], "") for value in self._cache.values()))

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "recent_exchanges": copy.deepcopy(self._exchanges),
            "cache": copy.deepcopy(self._cache),
        }

    def record_exchange(self, record: Mapping[str, object], *,
                        pending: Mapping[str, object] | None = None) -> bool:
        if not isinstance(record, Mapping):
            return False
        item = _exchange(record, pending)
        identity = item["record_ref"]
        self._exchanges = [row for row in self._exchanges
                           if row.get("record_ref") != identity]
        self._exchanges.append(item)
        self._exchanges = self._exchanges[-MAX_EXCHANGES:]
        return True

    def apply_delta(self, value: object, *, existing_refs: set[str] | Sequence[str] = (),
                    step: int | None = None) -> dict[str, object]:
        try:
            delta = normalize_delta(value)
        except DecisionContinuityError as error:
            return {"accepted": False, "error": str(error), "step": step}
        known = {str(ref) for ref in (existing_refs or ())}
        dropped_refs: list[str] = []
        if "hypothesis" in delta:
            self._cache["hypothesis"] = _typed_hypothesis(delta["hypothesis"])
        if "evidence_refs" in delta:
            kept: list[str] = []
            for ref in delta["evidence_refs"]:
                if ref in known:
                    kept.append(ref)
                else:
                    dropped_refs.append(ref)
            self._cache["evidence_refs"] = kept
        if "rejected_hypotheses" in delta:
            self._cache["rejected_hypotheses"] = [
                item for text in delta["rejected_hypotheses"]
                if (item := _typed_rejected(text)) is not None]
        if "next_action" in delta:
            self._cache["next_discriminating_action"] = _typed_action(delta["next_action"])
        return {
            "accepted": True,
            "step": step,
            "changed": sorted(delta),
            "dropped_refs": dropped_refs,
        }

    ingest = apply_delta

    def sync_organ_projection(self, projection: Mapping[str, object], *,
                              existing_refs: set[str] | Sequence[str] = ()) -> dict[str, object]:
        """Install bounded values emitted by existing host cognition organs.

        A model DELTA remains the owner of a field it explicitly patched.  Empty
        fields can be filled by a host organ, and the negative channel merges
        organ rows so a later host receipt cannot erase a model's rejected work.
        All provenance refs are filtered against the host id set here.
        """
        if not isinstance(projection, Mapping):
            return {"accepted": False, "error": "organ projection must be an object"}
        known = {str(ref) for ref in (existing_refs or ())}
        dropped: list[str] = []
        changed: list[str] = []
        hypothesis = projection.get("hypothesis")
        current_hypothesis = self._cache.get("hypothesis")
        if current_hypothesis is None and isinstance(hypothesis, Mapping):
            candidate = _typed_hypothesis(hypothesis.get("text"),
                                           str(hypothesis.get("source", "host_organ")))
            if candidate is not None:
                self._cache["hypothesis"] = candidate
                changed.append("hypothesis")

        refs = projection.get("evidence_refs")
        if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
            kept = []
            for ref in refs:
                ref = str(ref)
                if ref in known:
                    if ref not in kept:
                        kept.append(ref)
                else:
                    dropped.append(ref)
            if not self._cache["evidence_refs"] and kept:
                self._cache["evidence_refs"] = kept[:MAX_EVIDENCE_REFS]
                changed.append("evidence_refs")

        rejected = projection.get("rejected_hypotheses")
        if isinstance(rejected, Sequence) and not isinstance(rejected, (str, bytes)):
            existing = list(self._cache.get("rejected_hypotheses", ()))
            keys = {(str(item.get("source", "")), str(item.get("text", "")))
                    for item in existing if isinstance(item, Mapping)}
            for value in rejected:
                mapping = _object_dict(value)
                if mapping is None:
                    continue
                valid_refs: list[str] = []
                for ref in mapping.get("evidence_refs", ()) or ():
                    ref = str(ref)
                    if ref in known:
                        valid_refs.append(ref)
                    else:
                        dropped.append(ref)
                item = _typed_rejected(mapping.get("text"),
                                       str(mapping.get("source", "host_organ")), valid_refs)
                if item is None:
                    continue
                key = (str(item["source"]), str(item["text"]))
                if key not in keys:
                    keys.add(key)
                    existing.append(item)
            bounded = existing[-MAX_REJECTED:]
            if bounded != self._cache.get("rejected_hypotheses"):
                self._cache["rejected_hypotheses"] = bounded
                changed.append("rejected_hypotheses")

        next_action = projection.get("next_discriminating_action")
        host_predicates = (_typed_resume_predicates(next_action.get("resume_predicates"))
                           if isinstance(next_action, Mapping) else None)
        current_action = self._cache.get("next_discriminating_action")
        if (isinstance(current_action, Mapping) and host_predicates is not None
                and current_action.get("resume_predicates") != host_predicates):
            # Host resume diagnostics may enrich a model-owned action, but they
            # never replace its text or source.
            self._cache["next_discriminating_action"] = {
                **dict(current_action), "resume_predicates": host_predicates}
            changed.append("next_discriminating_action")
        elif current_action is None and isinstance(next_action, Mapping):
            candidate = _typed_action(next_action.get("text"),
                                       str(next_action.get("source", "host_organ")),
                                       resume_predicates=host_predicates)
            if candidate is not None:
                self._cache["next_discriminating_action"] = candidate
                changed.append("next_discriminating_action")
        return {"accepted": True, "changed": changed,
                "dropped_refs": list(dict.fromkeys(dropped))}

    def drop_dangling_refs(self, existing_refs: set[str] | Sequence[str] = ()) -> list[str]:
        """Revalidate persisted evidence before every rendered request."""
        known = {str(ref) for ref in (existing_refs or ())}
        current = list(self._cache["evidence_refs"])
        kept = [ref for ref in current if ref in known]
        dropped = [ref for ref in current if ref not in known]
        self._cache["evidence_refs"] = kept
        for item in self._cache.get("rejected_hypotheses", ()):
            if not isinstance(item, Mapping):
                continue
            refs = item.get("evidence_refs", ())
            if not isinstance(refs, list):
                continue
            item_kept = [ref for ref in refs if ref in known]
            dropped.extend(ref for ref in refs if ref not in known)
            item["evidence_refs"] = item_kept
        return list(dict.fromkeys(dropped))

    def _payload(self, exchanges: Sequence[Mapping[str, object]] | None = None,
                 cache: Mapping[str, object] | None = None) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "recent_exchanges": [copy.deepcopy(dict(item)) for item in (
                exchanges if exchanges is not None else self._exchanges)],
            "decision_cache": {
                "authority": "MODEL_HYPOTHESIS_ONLY",
                **copy.deepcopy(dict(cache if cache is not None else self._cache)),
            },
        }

    @staticmethod
    def _message(payload: Mapping[str, object]) -> dict[str, str]:
        content = BLOCK_HEADER + "\n" + json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {"role": "user", "content": content}

    def render_message(self, *, max_tokens: int = MAX_RENDER_TOKENS) -> dict[str, str]:
        """Render a deterministic block, shrinking detail before dropping history."""
        exchanges = copy.deepcopy(self._exchanges)
        cache = copy.deepcopy(self._cache)
        message = self._message(self._payload(exchanges, cache))
        # Preserve the newest real exchange; older history is the first thing
        # sacrificed when the cache and history compete for the shared budget.
        while token_estimate(message) > max_tokens and len(exchanges) > 1:
            exchanges.pop(0)
            message = self._message(self._payload(exchanges, cache))
        if token_estimate(message) > max_tokens:
            # Cache fields are already bounded, but a restored state may contain
            # oversized exchange values.  Keep the newest outcome's identity and
            # result class while dropping free-form detail and arguments.
            compacted = []
            for item in exchanges[-1:]:
                result = item.get("result", {}) if isinstance(item, Mapping) else {}
                call = item.get("call", {}) if isinstance(item, Mapping) else {}
                compacted.append({
                    "record_ref": _clip(item.get("record_ref", ""), 48),
                    "step": item.get("step"),
                    "call": {"tool": _clip(call.get("tool", ""), 40), "args": {}},
                    "result": {"class": _clip(result.get("class", "UNKNOWN"), 40),
                               "receipt_refs": list(result.get("receipt_refs", ()))[:2]},
                })
            message = self._message(self._payload(compacted, cache))
        if token_estimate(message) > max_tokens:
            # The cache itself is model text, so preserve its labels while making
            # the last fallback mechanically bounded for arbitrary restored data.
            # Keep one typed action/result exchange whenever one exists: a cache
            # that fits only by deleting the refusal it explains is not continuity.
            last_exchange = []
            if exchanges:
                item = exchanges[-1]
                result = item.get("result", {}) if isinstance(item, Mapping) else {}
                call = item.get("call", {}) if isinstance(item, Mapping) else {}
                last_exchange = [{
                    "record_ref": _clip(item.get("record_ref", ""), 48),
                    "step": item.get("step"),
                    "call": {"type": TYPE_ACTION_CALL,
                             "tool": _clip(call.get("tool", ""), 40), "args": {}},
                    "result": {"type": TYPE_ACTION_RESULT,
                               "class": _clip(result.get("class", "UNKNOWN"), 40),
                               "receipt_refs": list(result.get("receipt_refs", ()))[:2]},
                }]
            hypothesis = cache.get("hypothesis")
            if isinstance(hypothesis, Mapping):
                hypothesis = {**dict(hypothesis),
                              "text": _clip(hypothesis.get("text", ""), 96)}
            next_action = cache.get("next_discriminating_action")
            if isinstance(next_action, Mapping):
                next_action = {**dict(next_action),
                               "text": _clip(next_action.get("text", ""), 96)}
            fallback = {
                "schema": SCHEMA,
                "recent_exchanges": last_exchange,
                "decision_cache": {
                    "authority": "MODEL_HYPOTHESIS_ONLY",
                    "hypothesis": hypothesis,
                    "evidence_refs": list(cache.get("evidence_refs", ()))[:2],
                    "rejected_hypotheses": [],
                    "next_discriminating_action": next_action,
                },
            }
            message = self._message(fallback)
        # This assertion guards the contract in development and protects every
        # caller from accidentally adding an unbudgeted continuity block.
        if token_estimate(message) > max_tokens:
            raise AssertionError("decision continuity block exceeds its token budget")
        return message

    render = render_message


def compose_message(store: DecisionContinuityStore, *,
                    max_tokens: int = MAX_RENDER_TOKENS) -> dict[str, str]:
    return store.render_message(max_tokens=max_tokens)


def measure_request(messages: Sequence[Mapping[str, object]],
                    continuity_message: Mapping[str, object], *,
                    tools: object = None) -> dict[str, int]:
    """Return additive serialized-token accounting for one recorded request."""
    before = sum(token_estimate(message) for message in messages)
    if tools:
        before += token_estimate(tools)
    added = token_estimate(continuity_message)
    return {"before_tokens": before, "after_tokens": before + added,
            "added_tokens": added, "budget_tokens": MAX_RENDER_TOKENS}


def token_cost(message: Mapping[str, object]) -> int:
    return token_estimate(message)
