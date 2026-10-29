"""Host-validated five-slot decision continuity.

This module is an opt-in sub-mode of the existing continuity channel.  The
model can patch the five semantic slots, while the host owns the envelope,
reference checks, probe transitions, and invalidation notices.  A snapshot is
always the complete slot state; a delta never replaces an omitted slot.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence

from kernel_context import token_estimate


SCHEMA = "stateframe-decision-continuity/slots/1"
BLOCK_HEADER = "[Decision continuity]"
MAX_RENDER_TOKENS = 384
TARGET_RENDER_TOKENS = 320
MAX_TEXT = 180
MAX_SENTENCE = 220
MAX_SCOPE = 120
MAX_RECONSIDER = 180
MAX_REF = 180
MAX_SUPPORT_REFS = 3
MAX_REJECTED = 2
PROBE_LIFECYCLES = frozenset({"proposed", "issued", "finished", "failed", "blocked"})
TERMINAL_PROBE_LIFECYCLES = frozenset({"finished", "failed", "blocked"})
DISPOSITIONS = frozenset({"contradicted", "deprioritized", "attempt_failed"})
INTERPRETATION_PENDING = "result available; interpretation pending"

_SLOT_NAMES = ("hypothesis", "basis", "rejected", "probe", "step")
_ENVELOPE_FIELDS = frozenset({
    "run_id", "step", "revision", "workspace_revision", "source_revision",
    "last_action", "last_result", "last_receipt_id", "receipt_refs",
    "last_exchange", "changed_paths", "invalidated_paths", "updated_at",
    "dependencies",
})


class SlotDecisionContinuityError(ValueError):
    """A model patch or restored slot that cannot be trusted."""


def _work_carry_enabled() -> bool:
    try:
        from kernel_work_carry import enabled
        return enabled()
    except (ImportError, ValueError):
        return False


def _clip(value: object, maximum: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= maximum else text[: max(0, maximum - 1)] + "…"


def _text(value: object, field: str, maximum: int, *, required: bool = True) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise SlotDecisionContinuityError(
            f"{field} must be a non-empty string of at most {maximum} characters")
    return value.strip()


def _ref(value: object, field: str) -> str:
    return _text(value, field, MAX_REF)


def _refs(value: object, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise SlotDecisionContinuityError(
            f"{field} must be a list of at most {maximum} references")
    result: list[str] = []
    for item in value:
        item = _ref(item, f"{field} item")
        if item in result:
            raise SlotDecisionContinuityError(f"{field} contains duplicate references")
        result.append(item)
    return result


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SlotDecisionContinuityError(f"{field} must be an object")
    return value


def _allowed(value: Mapping[str, object], allowed: set[str], field: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise SlotDecisionContinuityError(
            f"{field} contains unknown fields: " + ", ".join(sorted(map(str, unknown))))


def _nullable_object(properties: Mapping[str, object], required: Sequence[str]) -> dict[str, object]:
    return {
        "type": ["object", "null"],
        "properties": copy.deepcopy(dict(properties)),
        "required": list(required),
        "additionalProperties": False,
    }


def delta_schema() -> dict[str, object]:
    """Return the closed model-write contract for the slots sub-mode."""
    def text(maximum: int) -> dict[str, object]:
        return {"type": "string", "minLength": 1, "maxLength": maximum}
    hypothesis = _nullable_object(
        {"text": text(MAX_TEXT), "falsifiable": {"type": "boolean"},
         "scope": text(MAX_SCOPE)},
        ("text", "falsifiable"),
    )
    basis = _nullable_object(
        {
            "support_refs": {
                "type": "array", "maxItems": MAX_SUPPORT_REFS,
                "items": text(MAX_REF),
            },
            "adverse_ref": {"type": ["string", "null"], "maxLength": MAX_REF},
            "sentence": text(MAX_SENTENCE),
        },
        ("support_refs", "adverse_ref", "sentence"),
    )
    rejected = {
        "type": ["array", "null"], "maxItems": MAX_REJECTED,
        "items": {
            "type": "object",
            "properties": {
                "text": text(MAX_TEXT), "disposition": {
                    "type": "string", "enum": sorted(DISPOSITIONS),
                }, "scope": text(MAX_SCOPE),
                "reconsider_condition": text(MAX_RECONSIDER),
            },
            "required": ["text", "disposition", "scope", "reconsider_condition"],
            "additionalProperties": False,
        },
    }
    probe = _nullable_object(
        {"question": text(MAX_TEXT), "action": text(MAX_TEXT),
         "predicted_result": text(MAX_TEXT), "contrary_meaning": text(MAX_TEXT)},
        ("question", "action", "predicted_result", "contrary_meaning"),
    )
    step = _nullable_object({"text": text(MAX_TEXT), "done_when": text(MAX_TEXT)},
                            ("text", "done_when"))
    return {
        "type": "object",
        "properties": {
            "hypothesis": hypothesis, "basis": basis, "rejected": rejected,
            "probe": probe, "step": step,
        },
        "additionalProperties": False,
        "minProperties": 1,
    }


def _hypothesis(value: object, *, restored: bool = False) -> dict[str, object] | None:
    if value is None:
        return None
    raw = _mapping(value, "hypothesis")
    if not restored:
        _allowed(raw, {"text", "falsifiable", "scope"}, "hypothesis")
    text = raw.get("text")
    if not isinstance(text, str) and restored:
        text = raw.get("statement", raw.get("claim"))
    text = _text(text, "hypothesis.text", MAX_TEXT)
    falsifiable = raw.get("falsifiable")
    if not isinstance(falsifiable, bool):
        raise SlotDecisionContinuityError("hypothesis.falsifiable must be boolean")
    result: dict[str, object] = {
        "text": text, "falsifiable": falsifiable,
        "status": "unverified",
    }
    scope = raw.get("scope")
    if isinstance(scope, str) and scope.strip():
        result["scope"] = _clip(scope, MAX_SCOPE)
    if restored and str(raw.get("status", "")).lower() == "stale":
        result["status"] = "stale"
        invalidated = raw.get("invalidated_by")
        if isinstance(invalidated, list):
            result["invalidated_by"] = [_clip(item, MAX_SCOPE) for item in invalidated[:2]
                                         if isinstance(item, str) and item.strip()]
    return result


def _basis(value: object, *, restored: bool = False) -> dict[str, object] | None:
    if value is None:
        return None
    raw = _mapping(value, "basis")
    if not restored:
        _allowed(raw, {"support_refs", "adverse_ref", "sentence"}, "basis")
    support = _refs(raw.get("support_refs", []), "basis.support_refs", MAX_SUPPORT_REFS)
    adverse = raw.get("adverse_ref")
    if adverse is not None:
        adverse = _ref(adverse, "basis.adverse_ref")
    sentence = _text(raw.get("sentence"), "basis.sentence", MAX_SENTENCE)
    # Insertion order is intentional: adverse evidence survives the first
    # compaction pass and is visible before ordinary support refs.
    result: dict[str, object] = {
        "adverse_ref": adverse, "support_refs": support, "sentence": sentence,
        "status": "unverified",
    }
    if restored and str(raw.get("status", "")).lower() == "stale":
        result["status"] = "stale"
        stale = raw.get("stale_refs")
        if isinstance(stale, list):
            result["stale_refs"] = [_clip(item, MAX_REF) for item in stale[:MAX_SUPPORT_REFS + 1]
                                     if isinstance(item, str) and item.strip()]
        paths = raw.get("invalidated_by")
        if isinstance(paths, list):
            result["invalidated_by"] = [_clip(item, MAX_SCOPE) for item in paths[:2]
                                         if isinstance(item, str) and item.strip()]
    return result


def _rejected(value: object, *, restored: bool = False) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_REJECTED:
        raise SlotDecisionContinuityError(
            f"rejected must contain at most {MAX_REJECTED} entries")
    result: list[dict[str, str]] = []
    allowed = {"text", "disposition", "scope", "reconsider_condition"}
    for raw_value in value:
        raw = _mapping(raw_value, "rejected item")
        if not restored:
            _allowed(raw, allowed, "rejected item")
        disposition = str(raw.get("disposition", "")).strip().lower()
        if disposition not in DISPOSITIONS:
            raise SlotDecisionContinuityError(
                "rejected.disposition must be contradicted, deprioritized, or attempt_failed")
        result.append({
            "text": _text(raw.get("text"), "rejected.text", MAX_TEXT),
            "disposition": disposition,
            "scope": _text(raw.get("scope"), "rejected.scope", MAX_SCOPE),
            "reconsider_condition": _text(
                raw.get("reconsider_condition"), "rejected.reconsider_condition", MAX_RECONSIDER),
        })
    return result


def _probe(value: object, *, restored: bool = False) -> dict[str, object] | None:
    if value is None:
        return None
    raw = _mapping(value, "probe")
    if not restored:
        _allowed(raw, {"question", "action", "predicted_result", "contrary_meaning"}, "probe")
    result: dict[str, object] = {
        "question": _text(raw.get("question"), "probe.question", MAX_TEXT),
        "action": _text(raw.get("action"), "probe.action", MAX_TEXT),
        "predicted_result": _text(
            raw.get("predicted_result"), "probe.predicted_result", MAX_TEXT),
        "contrary_meaning": _text(
            raw.get("contrary_meaning"), "probe.contrary_meaning", MAX_TEXT),
    }
    lifecycle = str(raw.get("lifecycle", raw.get("status", "proposed"))).strip().lower()
    if lifecycle == "pending":
        lifecycle = "finished" if raw.get("result") is not None else "proposed"
    if lifecycle not in PROBE_LIFECYCLES:
        lifecycle = "finished" if raw.get("result") is not None else "proposed"
    if raw.get("result") is not None:
        # A result is terminal even if a stale or poisoned snapshot called it
        # pending/issued.  Never put a completed probe back in the queue.
        if lifecycle not in TERMINAL_PROBE_LIFECYCLES:
            lifecycle = "finished"
        result["result"] = _compact_result(raw.get("result"))
        result["interpretation"] = INTERPRETATION_PENDING
    result["lifecycle"] = lifecycle
    result["status"] = lifecycle
    if lifecycle in TERMINAL_PROBE_LIFECYCLES and "result" in result:
        result["interpretation"] = INTERPRETATION_PENDING
    return result


def _step(value: object, *, restored: bool = False) -> dict[str, str] | None:
    if value is None:
        return None
    raw = _mapping(value, "step")
    if not restored:
        _allowed(raw, {"text", "done_when"}, "step")
    return {
        "text": _text(raw.get("text"), "step.text", MAX_TEXT),
        "done_when": _text(raw.get("done_when"), "step.done_when", MAX_TEXT),
    }


def _compact_result(value: object, maximum: int = MAX_TEXT) -> object:
    if isinstance(value, str):
        return _clip(value, maximum)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in ("result", "status", "receipt_ref", "detail"):
            if key in value:
                item = value[key]
                result[key] = _clip(item, maximum) if isinstance(item, str) else item
        return result or _clip(value, maximum)
    return _clip(value, maximum)


def normalize_delta(value: object) -> dict[str, object]:
    """Validate one model patch without accepting host-owned metadata."""
    if not isinstance(value, Mapping):
        raise SlotDecisionContinuityError("decision_delta must be an object")
    _allowed(value, set(_SLOT_NAMES), "decision_delta")
    if not value:
        raise SlotDecisionContinuityError("decision_delta must contain at least one slot")
    result: dict[str, object] = {}
    if "hypothesis" in value:
        result["hypothesis"] = _hypothesis(value["hypothesis"])
    if "basis" in value:
        result["basis"] = _basis(value["basis"])
    if "rejected" in value:
        result["rejected"] = _rejected(value["rejected"])
    if "probe" in value:
        result["probe"] = _probe(value["probe"])
        if result["probe"] is not None:
            result["probe"]["lifecycle"] = "proposed"
            result["probe"]["status"] = "proposed"
            result["probe"].pop("result", None)
            result["probe"].pop("interpretation", None)
    if "step" in value:
        result["step"] = _step(value["step"])
    return result


def split_action_arguments(arguments: object, *, call: Mapping[str, object] | None = None,
                           message: Mapping[str, object] | None = None
                           ) -> tuple[object, object | None]:
    """Remove the optional slot delta from ordinary native-tool arguments."""
    if not isinstance(arguments, Mapping):
        return arguments, None
    ordinary = copy.deepcopy(dict(arguments))
    delta = ordinary.pop("decision_delta", None)
    for container in (call, message):
        if isinstance(container, Mapping) and "decision_delta" in container:
            delta = container["decision_delta"]
    return ordinary, copy.deepcopy(delta) if delta is not None else None


def tool_surface(tools: object) -> object:
    """Add the closed optional slot delta without mutating the menu."""
    if not isinstance(tools, list):
        return tools
    surfaced = copy.deepcopy(tools)
    schema = delta_schema()
    for tool in surfaced:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        parameters = function.get("parameters")
        parameters = (copy.deepcopy(dict(parameters))
                      if isinstance(parameters, Mapping) else {
                          "type": "object", "properties": {}, "additionalProperties": False,
                      })
        properties = parameters.get("properties")
        parameters["properties"] = (copy.deepcopy(dict(properties))
                                     if isinstance(properties, Mapping) else {})
        parameters["properties"]["decision_delta"] = schema
        function["parameters"] = parameters
    from kernel_tool_schema_diet import filter_tools as filter_single_carrier_tools
    return filter_single_carrier_tools(surfaced)


def _empty_slots() -> dict[str, object]:
    return {"hypothesis": None, "basis": None, "rejected": [], "probe": None, "step": None}


def _path(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./").rstrip("/")


def _path_related(left: str, right: str) -> bool:
    left, right = _path(left), _path(right)
    return bool(left and right and (left == right or left.endswith("/" + right)
                                    or right.endswith("/" + left)))


def _bounded_scalar(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if -(10**15) < value < 10**15 else "<large-integer>"
    if isinstance(value, float):
        text = str(value)
        return value if len(text) <= 16 else _clip(text, 16)
    return _clip(value, MAX_TEXT) if isinstance(value, str) else value


def _safe_envelope(value: object, current: Mapping[str, object] | None = None) -> dict[str, object]:
    result: dict[str, object] = {"authority": "host"}
    if isinstance(current, Mapping):
        for key, item in current.items():
            if key in _ENVELOPE_FIELDS:
                result[key] = copy.deepcopy(item)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key not in _ENVELOPE_FIELDS:
                continue
            if key in {"changed_paths", "invalidated_paths"}:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    result[key] = [_clip(_path(path), MAX_SCOPE) for path in item[:4]
                                   if _path(path)]
            elif key == "receipt_refs":
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    result[key] = [_clip(ref, MAX_REF) for ref in item[:4]
                                   if isinstance(ref, str) and ref.strip()]
            elif key == "last_exchange":
                result[key] = _compact_result(item)
            elif key == "dependencies" and isinstance(item, Mapping):
                result[key] = {
                    _clip(ref, MAX_REF): (
                        [_clip(path, MAX_SCOPE) for path in paths[:4]
                         if isinstance(path, str) and _path(path)]
                        if isinstance(paths, Sequence) and not isinstance(paths, (str, bytes))
                        else _clip(paths, MAX_SCOPE)
                    )
                    for ref, paths in list(item.items())[:4]
                    if str(ref).strip()
                }
            elif isinstance(item, (str, int, float, bool)):
                result[key] = _bounded_scalar(item)
    result["authority"] = "host"
    return result


def _record_ref(record: Mapping[str, object]) -> str:
    for key in ("step_record_id", "record_id", "receipt_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    refs = record.get("receipt_ids")
    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)):
        for ref in refs:
            if isinstance(ref, str) and ref.strip():
                return ref.strip()
    return "exchange"


def _record_path(record: Mapping[str, object], pending: Mapping[str, object] | None) -> str:
    """Extract a host-observed target path for dependency binding only."""
    for source in (record, pending if isinstance(pending, Mapping) else {}):
        for key in ("target_path", "path", "file"):
            value = source.get(key)
            if isinstance(value, str) and _path(value):
                return _path(value)
        arguments = source.get("arguments", source.get("action_parameters"))
        if isinstance(arguments, Mapping):
            value = arguments.get("target_path", arguments.get("path", arguments.get("file")))
            if isinstance(value, str) and _path(value):
                return _path(value)
    return ""


def _exchange(record: Mapping[str, object], pending: Mapping[str, object] | None) -> dict[str, object]:
    pending = pending if isinstance(pending, Mapping) else {}
    refs = record.get("receipt_ids", ())
    receipt_refs = ([_clip(ref, MAX_REF) for ref in refs if str(ref).strip()][:3]
                    if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)) else [])
    return {
        "record_ref": _clip(_record_ref(record), MAX_REF),
        "tool": _clip(record.get("tool", pending.get("tool", "")), 64),
        "result": _clip(record.get("result", record.get("result_class", "UNKNOWN")), 64),
        "receipt_refs": receipt_refs,
    }


def _restore_exchange(value: Mapping[str, object]) -> dict[str, object]:
    refs = value.get("receipt_refs", value.get("receipt_ids", ()))
    refs = ([_clip(ref, MAX_REF) for ref in refs if str(ref).strip()][:3]
            if isinstance(refs, Sequence) and not isinstance(refs, (str, bytes)) else [])
    return {
        "record_ref": _clip(value.get("record_ref", value.get("step_record_id", "exchange")),
                             MAX_REF),
        "tool": _clip(value.get("tool", value.get("kind", "")), 64),
        "result": _clip(value.get("result", value.get("result_class", "UNKNOWN")), 64),
        "receipt_refs": refs,
    }


class SlotDecisionContinuityStore:
    """A complete five-slot snapshot with host-owned lifecycle metadata."""

    def __init__(self, *, exchanges: Sequence[Mapping[str, object]] = (),
                 cache: Mapping[str, object] | None = None,
                 slots: Mapping[str, object] | None = None,
                 host_envelope: Mapping[str, object] | None = None,
                 ref_dependencies: Mapping[str, object] | None = None,
                 work_carry: Mapping[str, object] | None = None):
        self._slots = _empty_slots()
        self._host_envelope = _safe_envelope(host_envelope)
        self._ref_dependencies: dict[str, tuple[str, ...]] = {}
        self._exchanges: list[dict[str, object]] = []
        self._work_carry = None
        if _work_carry_enabled():
            from kernel_work_carry import WorkCarryStore
            self._work_carry = WorkCarryStore.from_state(work_carry)
        for item in exchanges if isinstance(exchanges, Sequence) else ():
            if isinstance(item, Mapping):
                self._exchanges.append(_restore_exchange(item))
        self._exchanges = self._exchanges[-2:]
        if isinstance(ref_dependencies, Mapping):
            self.set_ref_dependencies(ref_dependencies)
        raw_slots = slots if isinstance(slots, Mapping) else cache
        if isinstance(raw_slots, Mapping):
            self._restore_slots(raw_slots)

    @classmethod
    def from_state(cls, state: Mapping[str, object], *,
                   host_envelope: Mapping[str, object] | None = None,
                   existing_refs: set[str] | Sequence[str] = ()) -> "SlotDecisionContinuityStore":
        if not isinstance(state, Mapping):
            return cls(host_envelope=host_envelope)
        slots = state.get("slots")
        if not isinstance(slots, Mapping):
            slots = state.get("cache")
        if not isinstance(slots, Mapping):
            slots = {key: state.get(key) for key in _SLOT_NAMES if key in state}
        restored_envelope = host_envelope
        if restored_envelope is None and isinstance(state.get("host_envelope"), Mapping):
            restored_envelope = state["host_envelope"]
        store = cls(
            exchanges=state.get("recent_exchanges", ())
            if isinstance(state.get("recent_exchanges"), list) else (),
            slots=slots,
            host_envelope=restored_envelope if isinstance(restored_envelope, Mapping) else None,
            ref_dependencies=state.get("ref_dependencies")
            if isinstance(state.get("ref_dependencies"), Mapping) else None,
            work_carry=state.get("work_carry")
            if _work_carry_enabled() and isinstance(state.get("work_carry"), Mapping)
            else None,
        )
        store.drop_dangling_refs(existing_refs)
        return store

    def _restore_slots(self, raw: Mapping[str, object]) -> None:
        for name, normalizer in (
            ("hypothesis", _hypothesis), ("basis", _basis),
            ("rejected", _rejected), ("probe", _probe), ("step", _step),
        ):
            if name not in raw:
                continue
            try:
                self._slots[name] = normalizer(raw[name], restored=True)
            except SlotDecisionContinuityError:
                # Poisoned or stale memory is advisory.  Drop only the bad slot.
                self._slots[name] = [] if name == "rejected" else None

    @property
    def slots(self) -> dict[str, object]:
        return copy.deepcopy(self._slots)

    @property
    def cache(self) -> dict[str, object]:
        return self.slots

    @property
    def exchanges(self) -> tuple[dict[str, object], ...]:
        return tuple(copy.deepcopy(self._exchanges))

    @property
    def has_content(self) -> bool:
        return bool(any(value not in (None, [], "") for value in self._slots.values())
                    or any(key != "authority" for key in self._host_envelope)
                    or (self._work_carry is not None and self._work_carry.has_content))

    def snapshot(self) -> dict[str, object]:
        slots = copy.deepcopy(self._slots)
        result = {
            "schema": SCHEMA,
            "mode": "slots",
            "host_envelope": copy.deepcopy(self._host_envelope),
            "ref_dependencies": {
                ref: list(paths) for ref, paths in self._ref_dependencies.items()
            },
            "slots": slots,
            # The direct names make the full snapshot convenient to inspect;
            # ``slots`` remains the canonical persisted grouping.
            **copy.deepcopy(slots),
            "recent_exchanges": copy.deepcopy(self._exchanges),
        }
        if self._work_carry is not None:
            result["work_carry"] = self._work_carry.snapshot()
        return result

    @property
    def work_carry(self) -> dict[str, object] | None:
        return self._work_carry.snapshot() if self._work_carry is not None else None

    def set_work_carry(self, value: Mapping[str, object] | None) -> dict[str, object]:
        """Bind the existing decision slots to one host-selected work item."""
        if self._work_carry is None:
            return {"accepted": False, "ignored": "work carry is off"}
        result = self._work_carry.set_work_carry(value)
        if result.get("accepted") and "work_changed" in result.get("changed", ()):
            self._slots = _empty_slots()
            self._exchanges = []
            self._ref_dependencies = {}
            self._host_envelope = _safe_envelope({})
            result["cleared_decision_state"] = True
        return result

    def record_work_action(self, record: Mapping[str, object], *,
                           pending: Mapping[str, object] | None = None) -> bool:
        """Record a host receipt without adding another F4 transcript exchange."""
        return bool(self._work_carry is not None
                    and self._work_carry.record_action(record, pending=pending))

    def set_host_envelope(self, envelope: Mapping[str, object] | None) -> dict[str, object]:
        before = copy.deepcopy(self._host_envelope)
        self._host_envelope = _safe_envelope(envelope, self._host_envelope)
        return {"accepted": True,
                "changed": [key for key in self._host_envelope
                            if before.get(key) != self._host_envelope.get(key)]}

    update_host_envelope = set_host_envelope

    def set_ref_dependencies(self, dependencies: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(dependencies, Mapping):
            return {"accepted": False, "error": "ref dependencies must be an object"}
        for ref, paths in dependencies.items():
            if not isinstance(ref, str) or not ref.strip():
                continue
            if isinstance(paths, str):
                paths = [paths]
            if isinstance(paths, Sequence) and not isinstance(paths, (bytes, str)):
                normalized = tuple(dict.fromkeys(
                    _clip(_path(path), MAX_SCOPE) for path in paths if _path(path)))
                if normalized:
                    self._ref_dependencies[_clip(ref.strip(), MAX_REF)] = normalized
        return {"accepted": True, "changed": sorted(self._ref_dependencies)}

    def record_exchange(self, record: Mapping[str, object], *,
                        pending: Mapping[str, object] | None = None) -> bool:
        if not isinstance(record, Mapping):
            return False
        item = _exchange(record, pending)
        self._exchanges = [row for row in self._exchanges
                           if row.get("record_ref") != item["record_ref"]]
        self._exchanges.append(item)
        self._exchanges = self._exchanges[-2:]
        self.set_host_envelope({
            "step": record.get("step"), "last_action": item["tool"],
            "last_result": item["result"], "last_receipt_id": (
                item["receipt_refs"][0] if item["receipt_refs"] else None),
            "receipt_refs": item["receipt_refs"], "last_exchange": item,
        })
        path = _record_path(record, pending)
        if path and item["receipt_refs"]:
            self.set_ref_dependencies({ref: [path] for ref in item["receipt_refs"]})
        kind = str(record.get("tool", record.get("kind", ""))).strip().lower()
        result = str(record.get("result", record.get("result_class", ""))).strip().upper()
        if (path and result == "APPLIED"
                and kind in {"edit", "str_replace", "file_create", "replace_module",
                             "write", "replace"}):
            self.invalidate_after_edit(path, step=record.get("step"))
        if self._work_carry is not None:
            self._work_carry.record_action(record, pending=pending)
        probe = self._slots.get("probe")
        if isinstance(probe, Mapping) and probe.get("lifecycle") == "proposed":
            action = str(probe.get("action", "")).strip().lower()
            if action and action in str(item.get("tool", "")).lower():
                self.issue_probe()
                result_class = str(item.get("result", "")).lower()
                terminal = ("finished" if result_class in {"applied", "ok", "pass", "passed", "green", "success"}
                            else "failed" if result_class in {"failed", "error", "red"}
                            else "blocked" if "refus" in result_class or "block" in result_class
                            else None)
                if terminal:
                    self.record_probe_result(terminal, {"result": item["result"]})
        return True

    def observe_host_receipt(self, receipt: Mapping[str, object]) -> dict[str, object]:
        if not isinstance(receipt, Mapping):
            return {"accepted": False, "error": "receipt must be an object"}
        self.set_host_envelope({
            "run_id": receipt.get("run_id"),
            "step": receipt.get("turn", receipt.get("step")),
            "revision": receipt.get("revision", receipt.get("workspace_revision")),
            "workspace_revision": receipt.get("workspace_revision"),
            "source_revision": receipt.get("source_revision"),
            "last_action": receipt.get("kind", receipt.get("tool")),
            "last_result": receipt.get("result_class", receipt.get("result")),
            "last_receipt_id": receipt.get("receipt_id"),
            "receipt_refs": receipt.get("receipt_ids", receipt.get("evidence_refs", [])),
            "dependencies": receipt.get("dependencies", receipt.get("ref_dependencies")),
        })
        dependencies = receipt.get("dependencies", receipt.get("ref_dependencies"))
        if isinstance(dependencies, Mapping):
            self.set_ref_dependencies(dependencies)
        kind = str(receipt.get("kind", receipt.get("tool", ""))).lower()
        result = str(receipt.get("result_class", receipt.get("result", ""))).upper()
        path = receipt.get("path", receipt.get("target_path", receipt.get("file")))
        if kind in {"edit", "str_replace", "write", "replace", "file_create"} and result == "APPLIED" and path:
            return self.invalidate_after_edit(str(path), step=receipt.get("turn"))
        return {"accepted": True, "invalidated_refs": []}

    def apply_delta(self, value: object, *, existing_refs: set[str] | Sequence[str] = (),
                    step: int | None = None) -> dict[str, object]:
        try:
            delta = normalize_delta(value)
        except SlotDecisionContinuityError as error:
            return {"accepted": False, "error": str(error), "step": step}
        current_probe = self._slots.get("probe")
        incoming_probe = delta.get("probe", "__absent__")
        if incoming_probe != "__absent__" and isinstance(current_probe, Mapping):
            lifecycle = current_probe.get("lifecycle")
            if lifecycle == "issued" or lifecycle in TERMINAL_PROBE_LIFECYCLES:
                return {"accepted": False, "error": "probe is one-shot", "step": step}
        known = {str(ref) for ref in (existing_refs or ()) if str(ref).strip()}
        changed: list[str] = []
        dropped: list[str] = []
        next_slots = copy.deepcopy(self._slots)
        if "hypothesis" in delta:
            next_slots["hypothesis"] = copy.deepcopy(delta["hypothesis"])
            changed.append("hypothesis")
        if "basis" in delta:
            basis = copy.deepcopy(delta["basis"])
            if isinstance(basis, Mapping):
                kept = []
                for ref in basis.get("support_refs", []):
                    if ref in known:
                        kept.append(ref)
                    else:
                        dropped.append(ref)
                adverse = basis.get("adverse_ref")
                if adverse is not None and adverse not in known:
                    dropped.append(adverse)
                    adverse = None
                basis["adverse_ref"] = adverse
                basis["support_refs"] = kept[:MAX_SUPPORT_REFS]
            next_slots["basis"] = basis
            changed.append("basis")
        if "rejected" in delta:
            next_slots["rejected"] = copy.deepcopy(delta["rejected"])
            changed.append("rejected")
        if "probe" in delta:
            next_slots["probe"] = copy.deepcopy(delta["probe"])
            changed.append("probe")
        if "step" in delta:
            next_slots["step"] = copy.deepcopy(delta["step"])
            changed.append("step")
        self._slots = next_slots
        self.set_host_envelope({"step": step} if step is not None else None)
        return {"accepted": True, "step": step, "changed": changed,
                "dropped_refs": list(dict.fromkeys(dropped))}

    ingest = apply_delta

    def drop_dangling_refs(self, existing_refs: set[str] | Sequence[str] = ()) -> list[str]:
        known = {str(ref) for ref in (existing_refs or ()) if str(ref).strip()}
        basis = self._slots.get("basis")
        dropped: list[str] = []
        if isinstance(basis, Mapping):
            support = []
            for ref in basis.get("support_refs", []):
                if ref in known:
                    support.append(ref)
                else:
                    dropped.append(ref)
            adverse = basis.get("adverse_ref")
            if adverse is not None and adverse not in known:
                dropped.append(adverse)
                adverse = None
            stale_refs = []
            for ref in basis.get("stale_refs", []):
                if ref in known:
                    stale_refs.append(ref)
                else:
                    dropped.append(ref)
            basis["support_refs"] = support[:MAX_SUPPORT_REFS]
            basis["adverse_ref"] = adverse
            if "stale_refs" in basis:
                basis["stale_refs"] = stale_refs[:MAX_SUPPORT_REFS + 1]
        if self._work_carry is not None:
            dropped.extend(self._work_carry.drop_dangling_refs(existing_refs))
        return list(dict.fromkeys(dropped))

    def invalidate_after_edit(self, path: str, *, step: int | None = None) -> dict[str, object]:
        edit_path = _clip(_path(path), MAX_SCOPE)
        if not edit_path:
            return {"accepted": False, "error": "edit path is required", "invalidated_refs": []}
        basis = self._slots.get("basis")
        refs = []
        if isinstance(basis, Mapping):
            refs = [ref for ref in [*basis.get("support_refs", []), basis.get("adverse_ref")]
                    if isinstance(ref, str) and ref]
        affected = [ref for ref in refs if any(
            _path_related(edit_path, dependency)
            for dependency in self._ref_dependencies.get(ref, ())
        )]
        self.set_host_envelope({
            "step": step, "changed_paths": [edit_path],
            "invalidated_paths": [edit_path] if affected else [],
        })
        if self._work_carry is not None:
            self._work_carry.invalidate_after_edit(edit_path)
        if not affected or not isinstance(basis, Mapping):
            return {"accepted": True, "invalidated_refs": []}
        stale = list(dict.fromkeys([*basis.get("stale_refs", []), *affected]))
        basis["status"] = "stale"
        basis["stale_refs"] = stale[:MAX_SUPPORT_REFS + 1]
        basis["invalidated_by"] = list(dict.fromkeys(
            [*basis.get("invalidated_by", []), edit_path]))[:2]
        hypothesis = self._slots.get("hypothesis")
        if isinstance(hypothesis, Mapping):
            hypothesis["status"] = "stale"
            hypothesis["invalidated_by"] = list(dict.fromkeys(
                [*hypothesis.get("invalidated_by", []), edit_path]))[:2]
        return {"accepted": True, "invalidated_refs": affected,
                "invalidated_slots": ["basis", "hypothesis"]}

    invalidate_dependencies = invalidate_after_edit
    observe_edit = invalidate_after_edit

    def issue_probe(self) -> dict[str, object]:
        probe = self._slots.get("probe")
        if not isinstance(probe, Mapping) or probe.get("lifecycle") != "proposed":
            return {"accepted": False, "error": "probe is not proposed"}
        probe["lifecycle"] = "issued"
        probe["status"] = "issued"
        return {"accepted": True, "lifecycle": "issued"}

    def record_probe_result(self, lifecycle: str, result: object = None) -> dict[str, object]:
        probe = self._slots.get("probe")
        lifecycle = str(lifecycle or "").strip().lower()
        if lifecycle not in TERMINAL_PROBE_LIFECYCLES:
            return {"accepted": False, "error": "probe result must be terminal"}
        if not isinstance(probe, Mapping) or probe.get("lifecycle") != "issued":
            return {"accepted": False, "error": "probe is not issued"}
        probe["lifecycle"] = lifecycle
        probe["status"] = lifecycle
        probe["result"] = _compact_result(result)
        probe["interpretation"] = INTERPRETATION_PENDING
        return {"accepted": True, "lifecycle": lifecycle,
                "interpretation": INTERPRETATION_PENDING}

    finish_probe = record_probe_result

    @staticmethod
    def _ordered_basis(value: Mapping[str, object], level: int) -> dict[str, object]:
        ref_limit = 120 if level == 0 else 80 if level == 1 else 48 if level == 2 else 30
        sentence_limit = 160 if level == 0 else 110 if level == 1 else 80 if level == 2 else 50
        adverse = (_clip(value.get("adverse_ref"), ref_limit)
                   if value.get("adverse_ref") is not None else None)
        support = [_clip(ref, ref_limit) for ref in value.get("support_refs", ())]
        if level >= 2:
            support = support[:1]
        elif level == 1:
            support = support[:2]
        result: dict[str, object] = {"adverse_ref": adverse, "support_refs": support,
                                     "sentence": _clip(value.get("sentence"),
                                                        sentence_limit)}
        if value.get("status") == "stale":
            result["status"] = "stale"
            result["stale_refs"] = [
                _clip(ref, ref_limit) for ref in list(value.get("stale_refs", ()))[:2]
            ]
            result["invalidated_by"] = [
                _clip(path, ref_limit) for path in list(value.get("invalidated_by", ()))[:1]
            ]
        return result

    @classmethod
    def _compact_slots(cls, slots: Mapping[str, object], level: int) -> dict[str, object]:
        result: dict[str, object] = {"hypothesis": None, "basis": None,
                                     "rejected": [], "probe": None, "step": None}
        hypothesis = slots.get("hypothesis")
        if isinstance(hypothesis, Mapping):
            hypothesis_limit = 130 if level == 0 else 90 if level == 1 else 64 if level == 2 else 44
            result["hypothesis"] = {
                "text": _clip(hypothesis.get("text"), hypothesis_limit),
                "falsifiable": bool(hypothesis.get("falsifiable")),
                "status": hypothesis.get("status", "unverified"),
            }
            if hypothesis.get("scope") and level == 0:
                result["hypothesis"]["scope"] = _clip(hypothesis.get("scope"), MAX_SCOPE)
            if hypothesis.get("invalidated_by"):
                result["hypothesis"]["invalidated_by"] = list(
                    _clip(path, MAX_SCOPE if level < 3 else 32)
                    for path in hypothesis.get("invalidated_by", ())
                )[:1]
        basis = slots.get("basis")
        if isinstance(basis, Mapping):
            result["basis"] = cls._ordered_basis(basis, level)
        rejected = slots.get("rejected")
        if isinstance(rejected, list):
            for item in rejected[:MAX_REJECTED if level < 2 else 1]:
                if not isinstance(item, Mapping):
                    continue
                field_limit = 100 if level == 0 else 64 if level == 1 else 48 if level == 2 else 36
                result["rejected"].append({
                    "text": _clip(item.get("text"), field_limit),
                    "disposition": item.get("disposition"),
                    "scope": _clip(item.get("scope"), 64 if level < 3 else 28),
                    "reconsider_condition": _clip(
                        item.get("reconsider_condition"),
                        100 if level == 0 else 60 if level == 1 else 48 if level == 2 else 36),
                })
        probe = slots.get("probe")
        if isinstance(probe, Mapping):
            probe_limit = 100 if level == 0 else 64 if level == 1 else 48 if level == 2 else 40
            result["probe"] = {
                "question": _clip(probe.get("question"), probe_limit),
                "action": _clip(probe.get("action"), probe_limit),
                "predicted_result": _clip(
                    probe.get("predicted_result"), 90 if level == 0 else 54 if level == 1 else 48 if level == 2 else 40),
                "contrary_meaning": _clip(
                    probe.get("contrary_meaning"), 90 if level == 0 else 54 if level == 1 else 48 if level == 2 else 40),
                "lifecycle": probe.get("lifecycle", "proposed"),
                "status": probe.get("status", probe.get("lifecycle", "proposed")),
            }
            if probe.get("result") is not None:
                result["probe"]["result"] = _compact_result(
                    probe.get("result"), 180 if level < 2 else 64 if level == 2 else 44)
                result["probe"]["interpretation"] = INTERPRETATION_PENDING
        step = slots.get("step")
        if isinstance(step, Mapping):
            step_limit = 100 if level == 0 else 64 if level == 1 else 48 if level == 2 else 40
            result["step"] = {
                "text": _clip(step.get("text"), step_limit),
                "done_when": _clip(step.get("done_when"), step_limit),
            }
        return result

    @staticmethod
    def _compact_envelope(envelope: Mapping[str, object], level: int) -> dict[str, object]:
        result: dict[str, object] = {"authority": "host"}
        ordered = ("step", "revision", "workspace_revision", "last_action",
                   "last_result", "last_receipt_id", "receipt_refs",
                   "changed_paths", "invalidated_paths", "dependencies",
                   "last_exchange")
        for key in ordered:
            if key not in envelope or envelope[key] in (None, [], ""):
                continue
            if level >= 3 and key not in {
                "step", "revision", "workspace_revision",
            }:
                continue
            value = envelope[key]
            if key == "last_exchange" and level >= 1:
                continue
            if isinstance(value, str):
                result[key] = _clip(value, 32 if level >= 3 else 64 if level else MAX_TEXT)
            elif isinstance(value, list):
                result[key] = [
                    _clip(item, 32 if level >= 3 else 64 if level else MAX_REF)
                    if isinstance(item, str) else item
                    for item in value[:1 if level >= 3 else 2 if level >= 1 else 4]
                ]
            elif isinstance(value, Mapping):
                result[key] = {
                    _clip(ref, 64 if level else MAX_REF): (
                        [_clip(path, 64 if level else MAX_SCOPE) for path in paths[:2 if level >= 1 else 4]]
                        if isinstance(paths, list) else _clip(paths, 64 if level else MAX_SCOPE)
                    )
                    for ref, paths in list(value.items())[:2 if level >= 1 else 4]
                }
            else:
                result[key] = _bounded_scalar(value)
        return result

    @classmethod
    def _payload(cls, slots: Mapping[str, object], envelope: Mapping[str, object],
                 level: int) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "host_envelope": cls._compact_envelope(envelope, level),
            "slots": cls._compact_slots(slots, level),
        }

    @staticmethod
    def _message(payload: Mapping[str, object]) -> dict[str, str]:
        return {"role": "user", "content": BLOCK_HEADER + "\n" + json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False)}

    def render_message(self, *, max_tokens: int = MAX_RENDER_TOKENS) -> dict[str, str]:
        """Render a compact full snapshot, preserving adverse evidence first."""
        if max_tokens <= 0:
            raise ValueError("continuity token budget must be positive")
        for level in range(4):
            message = self._message(self._payload(self._slots, self._host_envelope, level))
            if token_estimate(message) <= min(max_tokens, TARGET_RENDER_TOKENS):
                return message
        # Level three is intentionally terse but still contains all five slot
        # labels and the adverse reference.  If a caller asks for a smaller
        # custom budget, fail closed instead of silently omitting a slot.
        message = self._message(self._payload(self._slots, self._host_envelope, 3))
        if token_estimate(message) > max_tokens:
            raise AssertionError("decision continuity slots exceed token budget")
        return message

    render = render_message

    def sync_organ_projection(self, projection: Mapping[str, object], *,
                              existing_refs: set[str] | Sequence[str] = ()) -> dict[str, object]:
        """Accept the legacy advisory seam without promoting incomplete slots.

        The old organ projector does not provide the required falsifiability,
        adverse evidence, or probe fields.  In the slots sub-mode those partial
        candidates are deliberately ignored; only a complete explicit slot
        projection is eligible for host-side installation.
        """
        if not isinstance(projection, Mapping):
            return {"accepted": False, "error": "organ projection must be an object"}
        raw_slots = projection.get("slots")
        if not isinstance(raw_slots, Mapping):
            return {"accepted": True, "changed": [], "dropped_refs": []}
        return self.apply_delta(raw_slots, existing_refs=existing_refs)


__all__ = [
    "BLOCK_HEADER", "DISPOSITIONS", "INTERPRETATION_PENDING", "SCHEMA",
    "DecisionContinuityStore", "SlotDecisionContinuityError", "SlotDecisionContinuityStore",
    "delta_schema",
    "normalize_delta", "split_action_arguments", "tool_surface",
]

DecisionContinuityStore = SlotDecisionContinuityStore
