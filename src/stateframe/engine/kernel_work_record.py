"""Host-owned typed work records for durable goals, steps, and decisions.

The work record is deliberately separate from hypotheses and from completion
evidence.  The model may propose typed operations, but this module validates and
installs them atomically.  The host owns identifiers, status transitions, and
the bounded rendering surface.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
from collections.abc import Mapping, Sequence

from kernel_obligation_seed import (
    PROVENANCE as HOST_BRIEF_PROVENANCE,
    PublicBrief,
    enabled as obligation_seed_enabled,
    evidence_ref_satisfies,
    extract_obligations,
    host_seed_operations,
    open_obligations as _open_seed_obligations,
    public_manifest_for_owner,
)

MODE_ENV = "STATEFRAME_WORK_RECORD"
SCHEMA = "stateframe-work-record/1"

GOAL_STATUSES = ("open", "done", "blocked", "dropped")
STEP_STATUSES = ("todo", "doing", "done", "blocked", "dropped")
ALL_STATUSES = tuple(dict.fromkeys((*GOAL_STATUSES, *STEP_STATUSES)))

MAX_GOALS = 256
MAX_STEPS = 512
MAX_DECISIONS = 256
MAX_OPS = 32
MAX_TITLE = 240
MAX_STEP_TEXT = 400
MAX_CHOICE = 320
MAX_DROP_REASON = 240
MAX_EVIDENCE_REF = 180
MAX_FIELD_KEY = 64
MAX_FIELDS = 16
MAX_BASIS_REFS = 8
MAX_REF = 180
MAX_FIELD_VALUE_BYTES = 1200
MAX_FIELD_DEPTH = 3
MAX_RENDER_CHARS = 8_192
MAX_RENDER_OPEN = 32
MAX_RENDER_SUMMARIES = 64
MAX_RENDER_DECISIONS = 8

_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_FIELD_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


def _host_brief_step(row: Mapping[str, object]) -> bool:
    fields = row.get("fields", {})
    return (isinstance(fields, Mapping)
            and fields.get("provenance") == HOST_BRIEF_PROVENANCE
            and isinstance(fields.get("obligation_id"), str))


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _clip_seed(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 3)] + "..."


def _string(value: object, name: str, limit: int, *, required: bool = True) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        suffix = " and nonempty" if required else ""
        raise ValueError(f"{name} must be a string of at most {limit} characters{suffix}")
    return value


def _optional_string(value: object, name: str, limit: int) -> str | None:
    if value is None:
        return None
    return _string(value, name, limit)


def _refs(value: object, name: str, *, limit: int = MAX_BASIS_REFS) -> list[str]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{name} must be a list of at most {limit} references")
    result = []
    for item in value:
        ref = _string(item, name + " item", MAX_REF)
        if ref in result:
            raise ValueError(f"{name} contains duplicate references")
        result.append(ref)
    return result


def _field_value(value: object, *, depth: int = 0) -> object:
    """Validate a bounded JSON value without allowing an unbounded field sink."""
    if depth > MAX_FIELD_DEPTH:
        raise ValueError(f"field value nesting exceeds {MAX_FIELD_DEPTH} levels")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_STEP_TEXT:
            raise ValueError(f"field string values must be at most {MAX_STEP_TEXT} characters")
        return copy.deepcopy(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("field numeric values must be finite")
        return value
    if isinstance(value, list):
        if len(value) > 16:
            raise ValueError("field arrays may contain at most 16 values")
        return [_field_value(item, depth=depth + 1) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 16:
            raise ValueError("field objects may contain at most 16 keys")
        result = {}
        for key, item in value.items():
            field_key = _string(key, "field object key", MAX_FIELD_KEY)
            result[field_key] = _field_value(item, depth=depth + 1)
        return result
    raise ValueError("field value must be JSON-compatible")


class WorkRecordError(ValueError):
    """A typed rejection carrying the unchanged host state."""

    def __init__(self, reason: str, current_state: Mapping[str, object] | None = None):
        self.reason = str(reason)
        self.current_state = copy.deepcopy(dict(current_state or {}))
        super().__init__(self.reason)


def _state_error(reason: str, state: Mapping[str, object]) -> WorkRecordError:
    return WorkRecordError(reason, state)


def _record_id(value: object, name: str = "id") -> str:
    ident = _string(value, name, 64)
    if not _ID_RE.fullmatch(ident):
        raise ValueError(f"{name} must match {_ID_RE.pattern}")
    return ident


def _fields(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("fields must be an object")
    if len(value) > MAX_FIELDS:
        raise ValueError(f"fields may contain at most {MAX_FIELDS} keys")
    result = {}
    for key, item in value.items():
        field_key = _string(key, "field key", MAX_FIELD_KEY)
        if not _FIELD_KEY_RE.fullmatch(field_key):
            raise ValueError(f"field key must match {_FIELD_KEY_RE.pattern}")
        result[field_key] = _field_value(item)
    if len(_canonical(result).encode("utf-8")) > MAX_FIELD_VALUE_BYTES * MAX_FIELDS:
        raise ValueError("fields exceed the bounded record field budget")
    return result


def _validate_record_shape(state: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(state, Mapping):
        raise ValueError("work record state must be an object")
    allowed_state = {"schema", "goals", "steps", "decisions"}
    unknown = set(state) - allowed_state
    if unknown:
        raise ValueError("unknown work-record top-level key: " + sorted(map(str, unknown))[0])
    if state.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("unsupported work-record schema")
    for collection, limit in (("goals", MAX_GOALS), ("steps", MAX_STEPS),
                              ("decisions", MAX_DECISIONS)):
        rows = state.get(collection)
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError(f"{collection} must be a list of at most {limit} records")

    goals = []
    steps = []
    decisions = []
    seen: set[str] = set()

    for raw in state["goals"]:
        if not isinstance(raw, Mapping):
            raise ValueError("goal records must be objects")
        unknown = set(raw) - {"id", "title", "status", "drop_reason", "fields"}
        if unknown:
            raise ValueError("unknown goal record key: " + sorted(map(str, unknown))[0])
        ident = _record_id(raw.get("id"), "goal id")
        if ident in seen:
            raise ValueError("record ids must be unique")
        seen.add(ident)
        status = raw.get("status")
        if status not in GOAL_STATUSES:
            raise ValueError("unknown goal status")
        row = {"id": ident, "title": _string(raw.get("title"), "goal title", MAX_TITLE),
               "status": status, "fields": _fields(raw.get("fields", {}))}
        if "drop_reason" in raw:
            reason = _string(raw["drop_reason"], "drop_reason", MAX_DROP_REASON)
            if status != "dropped":
                raise ValueError("drop_reason is only valid for dropped records")
            row["drop_reason"] = reason
        if status == "dropped" and "drop_reason" not in row:
            raise ValueError("dropped goals require drop_reason")
        goals.append(row)

    goal_ids = {row["id"] for row in goals}
    for raw in state["steps"]:
        if not isinstance(raw, Mapping):
            raise ValueError("step records must be objects")
        unknown = set(raw) - {"id", "goal_id", "text", "status", "drop_reason",
                              "evidence_ref", "fields"}
        if unknown:
            raise ValueError("unknown step record key: " + sorted(map(str, unknown))[0])
        ident = _record_id(raw.get("id"), "step id")
        if ident in seen:
            raise ValueError("record ids must be unique")
        seen.add(ident)
        goal_id = _record_id(raw.get("goal_id"), "goal_id")
        if goal_id not in goal_ids:
            raise ValueError("step references an unknown goal")
        status = raw.get("status")
        if status not in STEP_STATUSES:
            raise ValueError("unknown step status")
        row = {"id": ident, "goal_id": goal_id,
               "text": _string(raw.get("text"), "step text", MAX_STEP_TEXT),
               "status": status, "fields": _fields(raw.get("fields", {}))}
        if "drop_reason" in raw:
            reason = _string(raw["drop_reason"], "drop_reason", MAX_DROP_REASON)
            if status != "dropped":
                raise ValueError("drop_reason is only valid for dropped records")
            row["drop_reason"] = reason
        if status == "dropped" and "drop_reason" not in row:
            raise ValueError("dropped steps require drop_reason")
        if "evidence_ref" in raw:
            row["evidence_ref"] = _string(raw["evidence_ref"], "evidence_ref", MAX_EVIDENCE_REF)
        steps.append(row)

    decision_ids = set()
    for raw in state["decisions"]:
        if not isinstance(raw, Mapping):
            raise ValueError("decision records must be objects")
        unknown = set(raw) - {"id", "choice", "basis_refs", "supersedes", "fields"}
        if unknown:
            raise ValueError("unknown decision record key: " + sorted(map(str, unknown))[0])
        ident = _record_id(raw.get("id"), "decision id")
        if ident in seen:
            raise ValueError("record ids must be unique")
        seen.add(ident)
        decision_ids.add(ident)
        row = {"id": ident, "choice": _string(raw.get("choice"), "choice", MAX_CHOICE),
               "basis_refs": _refs(raw.get("basis_refs"), "basis_refs"),
               "fields": _fields(raw.get("fields", {}))}
        if "supersedes" in raw and raw["supersedes"] is not None:
            row["supersedes"] = _record_id(raw["supersedes"], "supersedes")
        decisions.append(row)

    for row in decisions:
        if "supersedes" in row and row["supersedes"] not in decision_ids:
            raise ValueError("decision supersedes an unknown decision")
    return {"schema": SCHEMA, "goals": goals, "steps": steps, "decisions": decisions}


def _summary(row: Mapping[str, object], *, kind: str) -> dict[str, object]:
    if row.get("status") == "dropped":
        return {"id": row["id"], "status": "dropped", "reason": row["drop_reason"]}
    title = row.get("title") if kind == "goal" else row.get("text")
    return {"id": row["id"], "status": row["status"], "title": title}


def _take_bounded(rows: list[dict], limit: int) -> tuple[list[dict], int]:
    if len(rows) <= limit:
        return rows, 0
    # Preserve the oldest frontier and the newest additions; both ends are
    # deterministic and the omitted count makes loss explicit to the model.
    left = limit // 2
    right = limit - left
    return rows[:left] + rows[-right:], len(rows) - limit


def render_work_record(state: Mapping[str, object] | "WorkRecordStore",
                       *, max_chars: int = MAX_RENDER_CHARS,
                       compact: bool = False) -> str:
    """Render a deterministic, bounded model-facing work-record block."""
    if isinstance(state, WorkRecordStore):
        state = state.snapshot()
    validated = _validate_record_shape(state)
    if type(max_chars) is not int or max_chars < 256:
        raise ValueError("max_chars must be an integer of at least 256")
    if compact:
        open_goals = [
            {"id": row["id"], "status": row["status"], "title": row["title"]}
            for row in validated["goals"] if row["status"] in {"open", "blocked"}
        ]
        open_steps = []
        for row in validated["steps"]:
            if row["status"] not in {"todo", "doing", "blocked"}:
                continue
            item = {"id": row["id"], "status": row["status"], "text": row["text"]}
            fields = row.get("fields", {})
            if isinstance(fields, Mapping) and fields.get("obligation_id"):
                item["obligation_id"] = fields["obligation_id"]
            open_steps.append(item)
        decisions = [
            {"id": row["id"], "choice": row["choice"],
             "basis_refs": list(row["basis_refs"])}
            for row in validated["decisions"][-3:]
        ]
        payload = {
            "schema": SCHEMA,
            "frontier": {"goals": open_goals[-MAX_RENDER_OPEN:],
                          "steps": open_steps[-MAX_RENDER_OPEN:]},
            "decisions": decisions,
            "omitted": {
                "goals_closed": sum(row["status"] not in {"open", "blocked"}
                                     for row in validated["goals"]),
                "steps_closed": sum(row["status"] not in {"todo", "doing", "blocked"}
                                     for row in validated["steps"]),
                "decisions_older": max(0, len(validated["decisions"]) - len(decisions)),
            },
        }
        encoded = _canonical(payload)
        return "[WORK RECORD]\n" + encoded + "\n[/WORK RECORD]"
    goals_open = [copy.deepcopy(row) for row in validated["goals"]
                  if row["status"] in {"open", "blocked"}]
    goals_done = [_summary(row, kind="goal") for row in validated["goals"]
                  if row["status"] == "done"]
    goals_dropped = [_summary(row, kind="goal") for row in validated["goals"]
                     if row["status"] == "dropped"]
    steps_open = [copy.deepcopy(row) for row in validated["steps"]
                  if row["status"] in {"todo", "doing", "blocked"}]
    steps_done = [_summary(row, kind="step") for row in validated["steps"]
                  if row["status"] == "done"]
    steps_dropped = [_summary(row, kind="step") for row in validated["steps"]
                     if row["status"] == "dropped"]
    decisions = validated["decisions"]
    recent_decisions = [copy.deepcopy(row) for row in decisions[-MAX_RENDER_DECISIONS:]]
    older_decisions = [{"id": row["id"]} for row in decisions[:-MAX_RENDER_DECISIONS]]
    goals_open, goals_open_omitted = _take_bounded(goals_open, MAX_RENDER_OPEN)
    steps_open, steps_open_omitted = _take_bounded(steps_open, MAX_RENDER_OPEN)
    goals_done, goals_done_omitted = _take_bounded(goals_done, MAX_RENDER_SUMMARIES)
    goals_dropped, goals_dropped_omitted = _take_bounded(goals_dropped, MAX_RENDER_SUMMARIES)
    steps_done, steps_done_omitted = _take_bounded(steps_done, MAX_RENDER_SUMMARIES)
    steps_dropped, steps_dropped_omitted = _take_bounded(steps_dropped, MAX_RENDER_SUMMARIES)
    payload = {
        "schema": SCHEMA,
        "goals": {"open": goals_open, "done": goals_done, "dropped": goals_dropped},
        "steps": {"open": steps_open, "done": steps_done, "dropped": steps_dropped},
        "decisions": recent_decisions + older_decisions,
        "omitted": {
            "goals_open": goals_open_omitted, "goals_done": goals_done_omitted,
            "goals_dropped": goals_dropped_omitted, "steps_open": steps_open_omitted,
            "steps_done": steps_done_omitted, "steps_dropped": steps_dropped_omitted,
        },
    }
    prefix_len = len("[WORK RECORD]\n") + len("\n[/WORK RECORD]")
    payload_limit = max(1, max_chars - prefix_len)
    encoded = _canonical(payload)
    if len(encoded) > payload_limit:
        # The record fields are already bounded.  If a caller requests a
        # smaller-than-default display, reduce only low-priority summaries and
        # older decision ids; visible open records remain structurally complete.
        for key in ("goals", "steps"):
            for bucket in ("done", "dropped"):
                payload[key][bucket] = payload[key][bucket][: max(1, len(payload[key][bucket]) // 2)]
        payload["decisions"] = payload["decisions"][-MAX_RENDER_DECISIONS:]
        payload["omitted"]["render_compacted"] = True
        encoded = _canonical(payload)
    if len(encoded) > payload_limit:
        # This final case is only reachable with an unusually small custom cap.
        # Keep a typed, parseable summary rather than silently clipping JSON.
        compact = {"schema": SCHEMA, "goals": {"open": [], "done": [], "dropped": []},
                   "steps": {"open": [], "done": [], "dropped": []}, "decisions": [],
                   "omitted": {"render_compacted": True}}
        encoded = _canonical(compact)
    return "[WORK RECORD]\n" + encoded + "\n[/WORK RECORD]"


def open_work(state: Mapping[str, object] | "WorkRecordStore") -> list[str]:
    """Return host-record ids that remain open; pure and read-only.

    Ordinary goal ids precede step ids, preserving host insertion order.  A
    seeded root goal is structural and is omitted; its open steps are exposed
    by obligation id so the reducer and green lock name the actual ticket
    clauses.  Blocked work remains open because it is unresolved; done and
    dropped records do not.
    """
    if isinstance(state, WorkRecordStore):
        state = state.snapshot()
    if isinstance(state, Mapping) and isinstance(state.get("work_record"), Mapping):
        state = state["work_record"]
    validated = _validate_record_shape(state)
    host_seeded = any(_host_brief_step(row) for row in validated["steps"])
    result: list[str] = []
    for row in validated["goals"]:
        if row["status"] not in {"open", "blocked"}:
            continue
        fields = row.get("fields", {})
        if (host_seeded and isinstance(fields, Mapping)
                and fields.get("provenance") == HOST_BRIEF_PROVENANCE
                and fields.get("seeded_obligations") is True):
            continue
        result.append(row["id"])
    for row in validated["steps"]:
        if row["status"] not in {"todo", "doing", "blocked"}:
            continue
        if _host_brief_step(row):
            ident = row["fields"].get("obligation_id")
            if isinstance(ident, str):
                result.append(ident)
                continue
        result.append(row["id"])
    return list(dict.fromkeys(result))


def open_obligations(state: Mapping[str, object] | "WorkRecordStore") -> list[str]:
    """Expose unresolved HOST_BRIEF obligation ids as a pure host hook."""
    return _open_seed_obligations(state)


def _op_schema(op: str, properties: Mapping[str, object], required: Sequence[str]) -> dict:
    return {"type": "object", "properties": {"op": {"const": op}, **dict(properties)},
            "required": list(required), "additionalProperties": False}


_TEXT = lambda limit: {"type": "string", "maxLength": limit}
_ID = {"type": "string", "maxLength": 64}
_VALUE = {"type": ["string", "number", "boolean", "array", "object", "null"]}

OP_SCHEMAS = (
    _op_schema("add_goal", {"title": _TEXT(MAX_TITLE)}, ["op", "title"]),
    _op_schema("add_step", {"goal_id": _ID, "text": _TEXT(MAX_STEP_TEXT),
                             "after": {"anyOf": [_ID, {"type": "null"}]}},
                ["op", "goal_id", "text"]),
    _op_schema("split_step", {"id": _ID, "into": {"type": "array", "minItems": 2,
                                                    "maxItems": 8, "items": _TEXT(MAX_STEP_TEXT)}},
                ["op", "id", "into"]),
    _op_schema("amend_step", {"id": _ID, "text": _TEXT(MAX_STEP_TEXT)}, ["op", "id", "text"]),
    _op_schema("set_status", {"id": _ID, "status": {"type": "string", "enum": list(ALL_STATUSES)},
                               "evidence_ref": {"type": ["string", "null"],
                                                 "maxLength": MAX_EVIDENCE_REF}},
                ["op", "id", "status"]),
    _op_schema("drop", {"id": _ID, "reason": _TEXT(MAX_DROP_REASON)}, ["op", "id", "reason"]),
    _op_schema("add_decision", {"choice": _TEXT(MAX_CHOICE),
                                 "basis_refs": {"type": "array", "maxItems": MAX_BASIS_REFS,
                                                 "items": _TEXT(MAX_REF)},
                                 "supersedes": {"anyOf": [_ID, {"type": "null"}]}},
                ["op", "choice", "basis_refs"]),
    _op_schema("add_field", {"record_id": _ID, "key": _TEXT(MAX_FIELD_KEY), "value": _VALUE},
                ["op", "record_id", "key", "value"]),
)

INTENT_OP_SCHEMAS = (
    _op_schema(
        "claim_implemented",
        {"id": _ID, "evidence_refs": {"type": "array", "maxItems": 8,
                                         "items": _TEXT(MAX_REF)}},
        ["op", "id", "evidence_refs"],
    ),
)


WORK_TOOL = {"type": "function", "function": {
    "name": "work",
    "description": "Apply a bounded list of typed goal, step, decision, or field operations. "
                   "The host assigns ids and rejects invalid or stale mutations with the current "
                   "record. There is no replace or overwrite operation; use amend_step or add_field "
                   "only where its contract permits. Every drop requires a reason.",
    "parameters": {"type": "object", "properties": {
        "ops": {"type": "array", "minItems": 1, "maxItems": MAX_OPS,
                "items": {"oneOf": list(OP_SCHEMAS)}},
    }, "required": ["ops"], "additionalProperties": False},
}}

WORK_INTENT_TOOL = {"type": "function", "function": {
    "name": "work",
    "description": "Submit an intent-only implementation claim for a host obligation. "
                   "The host does not accept model-controlled status changes; it re-runs "
                   "the bound check and keeps the typed record compact.",
    "parameters": {"type": "object", "properties": {
        "ops": {"type": "array", "minItems": 1, "maxItems": 8,
                "items": {"oneOf": list(INTENT_OP_SCHEMAS)}},
    }, "required": ["ops"], "additionalProperties": False},
}}


def tool_surface(base: object, *, intent_only: bool = False) -> object:
    """Append the work tool only when enabled; OFF returns the exact input object."""
    if not enabled() or base is None:
        return base
    result = list(base)
    desired = WORK_INTENT_TOOL if intent_only else WORK_TOOL
    existing = {tool.get("function", {}).get("name"): tool for tool in result
                if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)}
    if "work" in existing:
        if existing["work"] == desired:
            return result
        # STATEFRAME_DECISION_CACHE adds one optional, host-stripped memory
        # field to native tool schemas.  It must not make the existing work
        # tool look like a conflicting definition to this older surface.
        from kernel_decision_cache import enabled as decision_cache_enabled
        if decision_cache_enabled():
            comparable = copy.deepcopy(existing["work"])
            parameters = comparable.get("function", {}).get("parameters", {})
            if isinstance(parameters, Mapping):
                properties = parameters.get("properties")
                if isinstance(properties, Mapping) and "memory" in properties:
                    properties = dict(properties)
                    properties.pop("memory", None)
                    parameters = dict(parameters)
                    parameters["properties"] = properties
                    comparable["function"] = {**comparable["function"],
                                                "parameters": parameters}
                    if comparable == desired:
                        return result
        if intent_only and existing["work"] == WORK_TOOL:
            index = next(index for index, tool in enumerate(result)
                         if tool.get("function", {}).get("name") == "work")
            result[index] = copy.deepcopy(desired)
            return result
        if existing["work"] != desired:
            raise ValueError("conflicting work tool definition")
    result.append(copy.deepcopy(desired))
    return result


def instructions(base: str, *, intent_only: bool = False) -> str:
    if not enabled():
        return base
    if intent_only:
        return base + (
            "\nThe `work` tool is intent-only in host-closure mode: submit only a "
            "claim_implemented operation with host evidence references. The host owns "
            "status and re-runs the bound checks; do not attempt to set or drop records."
        )
    text = base + (
        "\nThe `work` tool is the durable typed work record. Use it for goals, ordered steps, "
        "and decisions that must survive the next frame. It is host-owned: ids are assigned "
        "by the host, invalid operations are rejected with the current state, drops require "
        "a reason, and there is no replace operation. Keep mutations small; use evidence_ref "
        "when reopening a completed item."
    )
    if obligation_seed_enabled():
        text += " HOST_BRIEF steps require an evidence_ref when closing."
    return text


class WorkRecordStore:
    """An atomic host-owned store with no model-controlled ids or top-level keys."""

    def __init__(self, brief: str = "", *, state: Mapping[str, object] | None = None,
                 manifest: Mapping[str, object] | None = None,
                 obligations: Sequence[object] | None = None):
        if state is not None:
            clean = _validate_record_shape(state)
            self.goals = clean["goals"]
            self.steps = clean["steps"]
            self.decisions = clean["decisions"]
        else:
            title = _clip_seed(brief, MAX_TITLE) or "Complete the task"
            text = _clip_seed(brief, MAX_STEP_TEXT) or title
            self.goals = [{"id": "goal-0", "title": title, "status": "open", "fields": {}}]
            self.steps = []
            self.decisions = []
            self._counters = {"goal": 0, "step": 0, "decision": 0}
            if obligation_seed_enabled():
                seed_manifest = manifest
                if seed_manifest is None:
                    candidate = getattr(brief, "manifest", None)
                    seed_manifest = candidate if isinstance(candidate, Mapping) else None
                seed = (tuple(obligations) if obligations is not None else
                        extract_obligations(str(brief), seed_manifest))
            else:
                seed = ()
            if seed:
                self.goals[0]["fields"] = {
                    "provenance": HOST_BRIEF_PROVENANCE,
                    "seeded_obligations": True,
                }
                for operation in host_seed_operations(seed):
                    self._apply_one(operation, host_seed=True)
            else:
                self.steps = [{"id": "step-0", "goal_id": "goal-0", "text": text,
                               "status": "todo", "fields": {}}]
        self._counters = {"goal": self._next_counter("goal", self.goals),
                          "step": self._next_counter("step", self.steps),
                          "decision": self._next_counter("decision", self.decisions)}
        self.sequence = 0

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> "WorkRecordStore":
        return cls(state=state)

    @staticmethod
    def _next_counter(prefix: str, rows: Sequence[Mapping[str, object]]) -> int:
        values = []
        for row in rows:
            ident = row.get("id")
            if isinstance(ident, str) and ident.startswith(prefix + "-"):
                suffix = ident[len(prefix) + 1:]
                if suffix.isdigit():
                    values.append(int(suffix))
        return max(values, default=-1) + 1

    def _new_id(self, prefix: str) -> str:
        ident = f"{prefix}-{self._counters[prefix]}"
        self._counters[prefix] += 1
        return ident

    def snapshot(self) -> dict[str, object]:
        return {"schema": SCHEMA, "goals": copy.deepcopy(self.goals),
                "steps": copy.deepcopy(self.steps), "decisions": copy.deepcopy(self.decisions)}

    def render(self, *, max_chars: int = MAX_RENDER_CHARS) -> str:
        return render_work_record(self.snapshot(), max_chars=max_chars)

    def _records(self) -> dict[str, dict]:
        return {row["id"]: row for row in (*self.goals, *self.steps, *self.decisions)}

    def _goal(self, ident: str) -> dict:
        row = next((row for row in self.goals if row["id"] == ident), None)
        if row is None:
            raise ValueError("unknown goal id: " + ident)
        return row

    def _step(self, ident: str) -> dict:
        row = next((row for row in self.steps if row["id"] == ident), None)
        if row is None:
            raise ValueError("unknown step id: " + ident)
        return row

    def _record(self, ident: object) -> dict:
        return self._records().get(ident) or (_ for _ in ()).throw(
            ValueError("unknown record id: " + str(ident)))

    @staticmethod
    def _exact_op(op: object, name: str, required: set[str], optional: set[str] = set()) -> None:
        if not isinstance(op, Mapping):
            raise ValueError("each work operation must be an object")
        if op.get("op") != name:
            raise ValueError("expected operation " + name)
        unknown = set(op) - ({"op"} | required | optional)
        if unknown:
            raise ValueError(f"unknown fields for {name}: " + sorted(map(str, unknown))[0])
        missing = required - set(op)
        if missing:
            raise ValueError(f"{name} requires " + sorted(missing)[0])

    def _apply_one(self, op: Mapping[str, object], *, host_seed: bool = False
                   ) -> str | list[str] | None:
        if not isinstance(op, Mapping):
            raise ValueError("each work operation must be an object")
        name = op.get("op")
        if name == "add_goal":
            self._exact_op(op, name, {"title"})
            if len(self.goals) >= MAX_GOALS:
                raise ValueError(f"goal capacity reached ({MAX_GOALS})")
            self.goals.append({"id": self._new_id("goal"),
                               "title": _string(op["title"], "title", MAX_TITLE),
                               "status": "open", "fields": {}})
            return self.goals[-1]["id"]
        if name == "add_step":
            optional = {"after", "fields"} if host_seed else {"after"}
            self._exact_op(op, name, {"goal_id", "text"}, optional)
            if len(self.steps) >= MAX_STEPS:
                raise ValueError(f"step capacity reached ({MAX_STEPS})")
            goal_id = _record_id(op["goal_id"], "goal_id")
            goal = self._goal(goal_id)
            if goal["status"] in {"done", "dropped"}:
                raise ValueError("cannot add a step to a locked goal; reopen it with evidence_ref first")
            row = {"id": self._new_id("step"), "goal_id": goal_id,
                   "text": _string(op["text"], "text", MAX_STEP_TEXT),
                   "status": "todo", "fields": (_fields(op.get("fields", {}))
                                                    if host_seed else {})}
            after = op.get("after")
            if after is not None:
                after_id = _record_id(after, "after")
                after_row = self._step(after_id)
                if after_row["goal_id"] != goal_id:
                    raise ValueError("after must reference a step in the same goal")
                index = self.steps.index(after_row) + 1
                self.steps.insert(index, row)
            else:
                self.steps.append(row)
            return row["id"]
        if name == "split_step":
            self._exact_op(op, name, {"id", "into"})
            ident = _record_id(op["id"])
            original = self._step(ident)
            self._assert_not_locked(original, action="split")
            values = op["into"]
            if not isinstance(values, list) or not 2 <= len(values) <= 8:
                raise ValueError("into must contain 2 to 8 step texts")
            texts = [_string(value, "into item", MAX_STEP_TEXT) for value in values]
            original["status"] = "dropped"
            original["drop_reason"] = "split into " + ", ".join(self._clip_reason(item) for item in texts)
            index = self.steps.index(original) + 1
            created = []
            for part_index, text in enumerate(texts, start=1):
                fields = copy.deepcopy(original["fields"])
                if _host_brief_step(original):
                    parent = str(fields.get("obligation_id"))
                    fields["parent_obligation_id"] = parent
                    fields["obligation_id"] = f"{parent}-part-{part_index}"
                row = {"id": self._new_id("step"), "goal_id": original["goal_id"],
                       "text": text, "status": "todo", "fields": fields}
                self.steps.insert(index, row)
                index += 1
                created.append(row["id"])
            return created
        if name == "amend_step":
            self._exact_op(op, name, {"id", "text"})
            row = self._step(_record_id(op["id"]))
            self._assert_not_locked(row, action="amend")
            text = _string(op["text"], "text", MAX_STEP_TEXT)
            if row["text"] == text:
                raise ValueError("amend_step is a semantic no-op")
            row["text"] = text
            return row["id"]
        if name == "set_status":
            self._exact_op(op, name, {"id", "status"}, {"evidence_ref"})
            ident = _record_id(op["id"])
            row = self._record(ident)
            if "choice" in row:
                raise ValueError("decisions do not have a status")
            status = op["status"]
            statuses = GOAL_STATUSES if "title" in row else STEP_STATUSES
            if status not in statuses:
                raise ValueError(f"status {status!r} is invalid for this record")
            if status == "dropped":
                raise ValueError("use drop with a required reason for dropped records")
            previous = row["status"]
            evidence = _optional_string(op.get("evidence_ref"), "evidence_ref", MAX_EVIDENCE_REF)
            if previous == "dropped":
                raise ValueError("dropped records are terminal")
            if status == "done" and _host_brief_step(row):
                if not evidence:
                    raise ValueError("HOST_BRIEF steps require evidence_ref to close")
                if not evidence_ref_satisfies(row, evidence):
                    raise ValueError("HOST_BRIEF evidence_ref must name an edit or test receipt")
            if previous == "done" and status != "done" and not evidence:
                raise ValueError("done records are locked; reopening requires evidence_ref")
            if previous == status and (evidence is None or row.get("evidence_ref") == evidence):
                raise ValueError("set_status is a semantic no-op")
            row["status"] = status
            if evidence is not None and "title" not in row:
                row["evidence_ref"] = evidence
            return ident
        if name == "drop":
            self._exact_op(op, name, {"id", "reason"})
            ident = _record_id(op["id"])
            row = self._record(ident)
            if "choice" in row:
                raise ValueError("decisions cannot be dropped; record a superseding decision")
            self._assert_not_locked(row, action="drop")
            reason = _string(op["reason"], "reason", MAX_DROP_REASON)
            row["status"] = "dropped"
            row["drop_reason"] = reason
            row.pop("evidence_ref", None)
            return ident
        if name == "add_decision":
            self._exact_op(op, name, {"choice", "basis_refs"}, {"supersedes"})
            if len(self.decisions) >= MAX_DECISIONS:
                raise ValueError(f"decision capacity reached ({MAX_DECISIONS})")
            choice = _string(op["choice"], "choice", MAX_CHOICE)
            basis = _refs(op["basis_refs"], "basis_refs")
            supersedes = op.get("supersedes")
            if supersedes is not None:
                supersedes = _record_id(supersedes, "supersedes")
                if supersedes not in {row["id"] for row in self.decisions}:
                    raise ValueError("supersedes must reference an existing decision")
            row = {"id": self._new_id("decision"), "choice": choice,
                   "basis_refs": basis, "fields": {}}
            if supersedes is not None:
                row["supersedes"] = supersedes
            self.decisions.append(row)
            return row["id"]
        if name == "add_field":
            self._exact_op(op, name, {"record_id", "key", "value"})
            row = self._record(_record_id(op["record_id"], "record_id"))
            if "status" in row:
                self._assert_not_locked(row, action="add a field to")
            key = _string(op["key"], "key", MAX_FIELD_KEY)
            if not _FIELD_KEY_RE.fullmatch(key):
                raise ValueError(f"key must match {_FIELD_KEY_RE.pattern}")
            if key in row["fields"]:
                raise ValueError("field already exists; no replace or overwrite operation exists")
            if len(row["fields"]) >= MAX_FIELDS:
                raise ValueError(f"field capacity reached ({MAX_FIELDS})")
            row["fields"][key] = _field_value(op["value"])
            return row["id"] + ".fields." + key
        raise ValueError("unknown work operation; replace and overwrite are not supported")

    @staticmethod
    def _clip_reason(value: str) -> str:
        return value if len(value) <= 48 else value[:45] + "..."

    @staticmethod
    def _assert_not_locked(row: Mapping[str, object], *, action: str) -> None:
        if row.get("status") == "done":
            raise ValueError(f"done records are locked; reopen with evidence_ref before {action}")
        if row.get("status") == "dropped":
            raise ValueError(f"dropped records are terminal; cannot {action}")

    def apply(self, arguments: Mapping[str, object] | Sequence[Mapping[str, object]]) -> dict[str, object]:
        """Validate and commit one work call atomically."""
        if isinstance(arguments, Mapping):
            if set(arguments) != {"ops"}:
                raise WorkRecordError("work arguments must contain only ops", self.snapshot())
            ops = arguments.get("ops")
        else:
            ops = arguments
        try:
            if not isinstance(ops, list) or not 1 <= len(ops) <= MAX_OPS:
                raise ValueError(f"ops must be a list of 1 to {MAX_OPS} operations")
            open_steps = sum(row["status"] in {"todo", "doing", "blocked"}
                             for row in self.steps)
            rewrite_ops = sum(isinstance(op, Mapping) and op.get("op") in {"drop", "split_step"}
                              for op in ops)
            bulk_limit = max(2, math.ceil(open_steps * 0.20))
            if rewrite_ops > bulk_limit:
                raise ValueError(f"bulk rewrite guard: {rewrite_ops} drop/split operations exceed "
                                 f"the limit of {bulk_limit} for {open_steps} open steps")
            staged = WorkRecordStore.from_state(self.snapshot())
            created = []
            for op in ops:
                result = staged._apply_one(op)
                if result is not None and isinstance(op, Mapping) and op.get("op") in {
                    "add_goal", "add_step", "split_step", "add_decision"
                }:
                    created.extend(result if isinstance(result, list) else [result])
            staged.sequence = self.sequence + 1
        except WorkRecordError:
            raise
        except (TypeError, ValueError, KeyError) as error:
            raise WorkRecordError(str(error), self.snapshot()) from error
        self.goals, self.steps, self.decisions = staged.goals, staged.steps, staged.decisions
        self._counters = staged._counters
        self.sequence = staged.sequence
        return {"schema": SCHEMA, "sequence": self.sequence, "created": created,
                "state": self.snapshot()}


def brief_for_owner(owner: object) -> str:
    """Extract only the public task brief used for the initial host seed."""
    policy = getattr(owner, "information_policy", None)
    brief = getattr(policy, "ordinary_brief", None)
    if not (isinstance(brief, str) and brief.strip()):
        row = getattr(owner, "task_row", {})
        if isinstance(row, Mapping):
            for key in ("_bridge_ordinary_brief", "msg", "brief", "symptom"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    brief = value
                    break
    brief = brief if isinstance(brief, str) else ""
    if obligation_seed_enabled():
        return PublicBrief(brief, public_manifest_for_owner(owner))
    return brief


__all__ = [
    "ALL_STATUSES", "GOAL_STATUSES", "MAX_RENDER_CHARS", "MODE_ENV", "OP_SCHEMAS",
    "SCHEMA", "STEP_STATUSES", "WORK_TOOL", "WorkRecordError", "WorkRecordStore",
    "brief_for_owner", "enabled", "instructions", "mode", "open_obligations", "open_work",
    "render_work_record", "tool_surface",
]
