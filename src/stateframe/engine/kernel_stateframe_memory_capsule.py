"""Opt-in, typed cross-run memory for the StateFrame model boundary.

The host owns durable records.  The model receives only a deterministic
presentation of at most three complete records.  Memory is advisory text: it
never participates in tool authorization, completion, or evidence authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TextIO


MODE_ENV = "STATEFRAME_MEMORY_CAPSULE"
CAP_ENV = "STATEFRAME_MEMORY_CAP"
SCHEMA = "stateframe-memory-capsule/2"
RECORD_SCHEMA = "stateframe-memory-record/2"
CAPSULE_CAPS = (384, 768, 1536)
DEFAULT_CAP = 768
MAX_RECORDS = 256
MAX_CAPSULE_RECORDS = 3
MAX_INPUT_ROWS = 4_096
MAX_INPUT_BYTES = 4_000_000
MAX_RECORD_BYTES = 16_000
MAX_STRING = 1_200
MAX_FIELDS = 32
MAX_ITEMS = 32
MAX_JSON_DEPTH = 5
MAX_PREDICATES = 24
MAX_PREDICATE_DEPTH = 5
MAX_RECEIPTS = 24
MAX_DEPENDENCIES = 32
MODEL_CAP_ESTIMATOR = "serialized-json-chars/4-proxy; not provider-tokenizer measured"
CAPSULE_FRAME_FIELD = "memory_capsule_v2"
CAPSULE_HEADER_PREFIX = "[Memory capsule v2;"

LIVE_LIFECYCLES = frozenset({"ACTIVE", "ADOPTED", "PROVEN"})
KNOWN_LIFECYCLES = frozenset({
    "ACTIVE", "ADOPTED", "PROVEN", "PROPOSED", "RETIRED", "INVALIDATED",
    "SUPERSEDED", "DISPUTED", "REJECTED",
})
_PERMISSION_FIELDS = frozenset({
    "allow", "allow_tools", "allowed_tools", "authorization", "authorizations",
    "capabilities", "capability", "credential", "credentials", "grant", "grants",
    "permission", "permissions", "tool_grant", "tool_grants", "tool_permissions",
})
_FIELD_NAME_KEYS = frozenset({"field", "attribute", "predicate_field"})
_VALID_OPERATORS = frozenset({
    "eq", "equals", "ne", "not_equals", "in", "not_in", "contains",
    "not_contains", "exists", "present", "all", "any",
})


class MemoryCapsuleError(ValueError):
    """Base error for malformed or unsafe memory data."""


class MemorySecurityError(MemoryCapsuleError):
    """Memory attempted to carry an authority-shaped field."""


class MemoryRecordError(MemoryCapsuleError):
    """A record failed the typed, bounded schema."""


class FrozenDict(Mapping[str, object]):
    """Recursively used immutable mapping with stable key order."""

    __slots__ = ("_items", "_values")

    def __init__(self, values: Mapping[str, object] | Iterable[tuple[str, object]]):
        rows = (tuple(values.items()) if isinstance(values, Mapping) else tuple(values))
        if any(not isinstance(key, str) for key, _ in rows):
            raise MemoryRecordError("mapping keys must be strings")
        if len({key for key, _ in rows}) != len(rows):
            raise MemoryRecordError("mapping contains duplicate keys")
        object.__setattr__(self, "_items", tuple(sorted(rows, key=lambda item: item[0])))
        object.__setattr__(self, "_values", MappingProxyType(dict(self._items)))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("FrozenDict is immutable")

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._items)!r})"

    def __deepcopy__(self, memo: dict[int, object]) -> "FrozenDict":
        del memo
        return self


def mode(value: object = None) -> str:
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    if not isinstance(raw, str):
        raise ValueError(f"{MODE_ENV} must be off or on")
    normalized = raw.strip().lower()
    if normalized in {"0", "off"}:
        return "off"
    if normalized in {"1", "on"}:
        return "on"
    raise ValueError(f"{MODE_ENV} must be off or on")


def enabled(value: object = None) -> bool:
    return mode(value) == "on"


def configured_cap(value: object = None) -> int:
    raw = os.environ.get(CAP_ENV, str(DEFAULT_CAP)) if value is None else value
    try:
        cap = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{CAP_ENV} must be 384, 768, or 1536") from error
    if cap not in CAPSULE_CAPS:
        raise ValueError(f"{CAP_ENV} must be 384, 768, or 1536")
    return cap


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return FrozenDict((key, _freeze_json(child)) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _canonical(value: object) -> str:
    return json.dumps(_thaw_json(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _bounded_json(value: object, *, depth: int = 0, label: str = "value") -> object:
    if depth > MAX_JSON_DEPTH:
        raise MemoryRecordError(f"{label} exceeds the maximum nesting depth")
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_STRING:
            raise MemoryRecordError(f"{label} exceeds {MAX_STRING} characters")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MemoryRecordError(f"{label} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_FIELDS:
            raise MemoryRecordError(f"{label} has too many fields")
        result: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise MemoryRecordError(f"{label} keys must be strings")
            result[key] = _bounded_json(child, depth=depth + 1, label=f"{label}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_ITEMS:
            raise MemoryRecordError(f"{label} has too many items")
        return [_bounded_json(child, depth=depth + 1, label=f"{label}[{index}]")
                for index, child in enumerate(value)]
    raise MemoryRecordError(f"{label} must be JSON-compatible")


def _normal_key(value: object) -> str:
    return "_".join(str(value).strip().lower().replace("-", "_").split())


def _permission_paths(value: object, prefix: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            normalized = _normal_key(key_text)
            if normalized in _PERMISSION_FIELDS:
                found.append(path)
            if normalized in _FIELD_NAME_KEYS and _normal_key(child) in _PERMISSION_FIELDS:
                found.append(path + "=" + str(child))
            found.extend(_permission_paths(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(_permission_paths(child, f"{prefix}[{index}]"))
    return tuple(found)


def _reject_permission_fields(value: object, *, label: str) -> None:
    paths = _permission_paths(value)
    if paths:
        raise MemorySecurityError(
            f"memory data cannot carry authority fields in {label}: " + ", ".join(paths[:4])
        )


def _text(value: object, label: str, *, limit: int = MAX_STRING,
          required: bool = True) -> str:
    if not isinstance(value, str):
        raise MemoryRecordError(f"{label} must be a string")
    rendered = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if required and not rendered:
        raise MemoryRecordError(f"{label} must be nonempty")
    if len(rendered) > limit:
        raise MemoryRecordError(f"{label} exceeds {limit} characters")
    return rendered


def _rows(value: object, *, label: str) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_PREDICATES:
            raise MemoryRecordError(f"{label} has too many rows")
        rows = tuple(value)
        if any(not isinstance(item, Mapping) for item in rows):
            raise MemoryRecordError(f"{label} rows must be objects")
        return rows  # type: ignore[return-value]
    raise MemoryRecordError(f"{label} must be an object or list of objects")


def _normalise_predicate(row: Mapping[str, object], *, depth: int = 0) -> FrozenDict:
    if depth > MAX_PREDICATE_DEPTH:
        raise MemoryRecordError("applicability exceeds the maximum nesting depth")
    _reject_permission_fields(row, label="applicability")
    raw_operator = row.get("op", row.get("operator", "eq"))
    if not isinstance(raw_operator, str):
        raise MemoryRecordError("applicability operator must be a string")
    operator = raw_operator.strip().lower()
    if operator not in _VALID_OPERATORS:
        raise MemoryRecordError(f"unsupported applicability operator: {operator}")
    if operator in {"all", "any"}:
        children = row.get("predicates", row.get("value"))
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes, bytearray)):
            raise MemoryRecordError(f"applicability {operator} needs predicate children")
        if not children:
            raise MemoryRecordError(f"applicability {operator} needs at least one child")
        if len(children) > MAX_PREDICATES:
            raise MemoryRecordError("applicability has too many nested predicates")
        normalized = []
        for child in children:
            if not isinstance(child, Mapping):
                raise MemoryRecordError("nested applicability predicates must be objects")
            normalized.append(_normalise_predicate(child, depth=depth + 1))
        return FrozenDict({"op": operator, "predicates": tuple(normalized)})
    field_name = row.get("field", row.get("attribute"))
    if not isinstance(field_name, str) or not field_name.strip():
        raise MemoryRecordError("applicability predicates need a field")
    if _normal_key(field_name) in _PERMISSION_FIELDS:
        raise MemorySecurityError("applicability cannot select a permission field")
    item: dict[str, object] = {
        "field": _text(field_name, "applicability.field", limit=128),
        "op": operator,
    }
    if operator not in {"exists", "present"}:
        item["value"] = _freeze_json(
            _bounded_json(row.get("value"), label="applicability.value"))
    return FrozenDict(item)


def _normalise_predicates(value: object) -> tuple[FrozenDict, ...]:
    if isinstance(value, Mapping) and not any(
            key in value for key in ("field", "attribute", "op", "operator")):
        if any(key in value for key in ("value", "predicates")):
            raise MemoryRecordError("applicability mapping needs a field or operator")
        value = tuple({"field": key, "op": "eq", "value": child}
                      for key, child in value.items())
    rows = _rows(value, label="applicability")
    if len(rows) > MAX_PREDICATES:
        raise MemoryRecordError("applicability has too many predicates")
    return tuple(_normalise_predicate(row) for row in rows)


def _normalise_receipts(value: object) -> tuple[FrozenDict, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        raise MemoryRecordError("evidence_receipts must be an object or list")
    if len(values) > MAX_RECEIPTS:
        raise MemoryRecordError("evidence_receipts has too many receipts")
    result: list[FrozenDict] = []
    for receipt in values:
        if isinstance(receipt, str):
            receipt = {"ref": receipt}
        if not isinstance(receipt, Mapping):
            raise MemoryRecordError("evidence receipts must be objects")
        _reject_permission_fields(receipt, label="evidence_receipts")
        row = _bounded_json(receipt, label="evidence_receipt")
        assert isinstance(row, dict)
        reference = row.get("ref", row.get("receipt_id", row.get("id")))
        if reference is None:
            raise MemoryRecordError("each evidence receipt needs ref")
        row["ref"] = _text(reference, "evidence_receipt.ref", limit=240)
        result.append(FrozenDict(_freeze_json(row)))
    return tuple(result)


@dataclass(frozen=True)
class Utility:
    benefit: float = 0.0
    cost: float = 0.0
    uses: int = 0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        for name in ("benefit", "cost", "confidence"):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise MemoryRecordError(f"utility.{name} must be finite")
        if self.cost < 0 or self.benefit < 0 or not 0 <= self.confidence <= 1:
            raise MemoryRecordError("utility values are outside their bounds")
        if type(self.uses) is not int or self.uses < 0:
            raise MemoryRecordError("utility.uses must be a nonnegative integer")

    def to_dict(self) -> dict[str, object]:
        return {"benefit": self.benefit, "cost": self.cost,
                "uses": self.uses, "confidence": self.confidence}

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


def _normalise_utility(value: object) -> Utility:
    if value is None:
        return Utility()
    if isinstance(value, Utility):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Utility(benefit=float(value))
    if not isinstance(value, Mapping):
        raise MemoryRecordError("utility must be an object or number")
    _reject_permission_fields(value, label="utility")
    try:
        return Utility(
            benefit=float(value.get("benefit", value.get("value", 0.0))),
            cost=float(value.get("cost", 0.0)),
            uses=int(value.get("uses", value.get("use_count", 0))),
            confidence=float(value.get("confidence", 0.0)),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise MemoryRecordError("utility contains an invalid number") from error


def _normalise_dependencies(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values = value
    else:
        raise MemoryRecordError("dependencies must be strings, objects, or a list")
    if len(values) > MAX_DEPENDENCIES:
        raise MemoryRecordError("too many dependencies")
    result: list[object] = []
    for item in values:
        _reject_permission_fields(item, label="dependencies")
        if isinstance(item, Mapping):
            result.append(_freeze_json(_bounded_json(item, label="dependency")))
        else:
            result.append(_text(item, "dependency", limit=240))
    return tuple(result)


def _normalise_lifecycle(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get("state", value.get("status"))
    if not isinstance(value, str):
        raise MemoryRecordError("lifecycle must be a string")
    lifecycle = value.strip().upper().replace("-", "_") or "ACTIVE"
    if lifecycle not in KNOWN_LIFECYCLES:
        raise MemoryRecordError(f"unknown memory lifecycle: {lifecycle}")
    return lifecycle


def _string_sequence(value: object, *, label: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise MemoryRecordError(f"{label} must be a list")
    return tuple(value)


@dataclass(frozen=True)
class MemoryRecord:
    kind: str
    scope: str
    claim: str
    limit: str
    applicability: tuple[FrozenDict, ...] = ()
    evidence_receipts: tuple[FrozenDict, ...] = ()
    lifecycle: str = "ACTIVE"
    utility: Utility = field(default_factory=Utility)
    dependencies: tuple[object, ...] = ()
    record_id: str = ""
    invalidated_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _reject_permission_fields({
            "kind": self.kind, "scope": self.scope, "claim": self.claim, "limit": self.limit,
            "applicability": self.applicability, "evidence_receipts": self.evidence_receipts,
            "utility": self.utility.to_dict() if isinstance(self.utility, Utility) else self.utility,
            "dependencies": self.dependencies,
        }, label="record")
        kind = _text(self.kind, "kind", limit=120)
        scope = _text(self.scope, "scope", limit=240)
        claim = _text(self.claim, "claim")
        limit = _text(self.limit, "limit")
        applicability = _normalise_predicates(self.applicability)
        evidence = _normalise_receipts(self.evidence_receipts)
        lifecycle = _normalise_lifecycle(self.lifecycle)
        utility = _normalise_utility(self.utility)
        dependencies = _normalise_dependencies(self.dependencies)
        invalidated_raw = _string_sequence(self.invalidated_by, label="invalidated_by")
        if len(invalidated_raw) > MAX_DEPENDENCIES:
            raise MemoryRecordError("too many invalidation references")
        invalidated_by = tuple(_text(item, "invalidated_by item", limit=240)
                               for item in invalidated_raw)
        record_id = self.record_id
        if record_id:
            record_id = _text(record_id, "record_id", limit=96)
        payload = {
            "schema": RECORD_SCHEMA, "kind": kind, "scope": scope, "claim": claim,
            "limit": limit, "applicability": list(applicability),
            "evidence_receipts": list(evidence), "lifecycle": lifecycle,
            "utility": utility.to_dict(), "dependencies": list(dependencies),
        }
        if not record_id:
            record_id = "memory-" + _digest(payload)[len("sha256:"):len("sha256:") + 16]
        serialized_size = len(_canonical({
            **payload, "record_id": record_id, "invalidated_by": list(invalidated_by),
        }).encode("utf-8"))
        if serialized_size > MAX_RECORD_BYTES:
            raise MemoryRecordError(f"memory record exceeds {MAX_RECORD_BYTES} bytes")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "claim", claim)
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "applicability", applicability)
        object.__setattr__(self, "evidence_receipts", evidence)
        object.__setattr__(self, "lifecycle", lifecycle)
        object.__setattr__(self, "utility", utility)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "invalidated_by", invalidated_by)

    @classmethod
    def from_dict(cls, value: object) -> "MemoryRecord":
        if not isinstance(value, Mapping):
            raise MemoryRecordError("memory record must be an object")
        if value.get("schema") != RECORD_SCHEMA:
            raise MemoryRecordError(f"memory record schema must be {RECORD_SCHEMA!r}")
        _reject_permission_fields(value, label="record")
        return cls(
            kind=value.get("kind", "observation"),
            scope=value.get("scope", "unspecified"),
            claim=value.get("claim", value.get("content", value.get("statement", ""))),
            limit=value.get("limit", value.get("boundary", "Evidence is limited to the recorded scope.")),
            applicability=value.get("applicability", value.get("applicability_predicates",
                                                                  value.get("predicates", ()))),
            evidence_receipts=value.get("evidence_receipts", value.get("evidence",
                                                                        value.get("receipts", ()))),
            lifecycle=value.get("lifecycle", "ACTIVE"),
            utility=value.get("utility", {}),
            dependencies=value.get("dependencies", ()),
            record_id=value.get("record_id", value.get("id", "")),
            invalidated_by=value.get("invalidated_by", ()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": RECORD_SCHEMA, "record_id": self.record_id,
            "kind": self.kind, "scope": self.scope, "claim": self.claim,
            "limit": self.limit,
            "applicability": [_thaw_json(row) for row in self.applicability],
            "evidence_receipts": [_thaw_json(row) for row in self.evidence_receipts],
            "lifecycle": self.lifecycle, "utility": self.utility.to_dict(),
            "dependencies": [_thaw_json(item) for item in self.dependencies],
            "invalidated_by": list(self.invalidated_by),
        }


def _as_record(value: object) -> MemoryRecord:
    return value if isinstance(value, MemoryRecord) else MemoryRecord.from_dict(value)


def _lookup(context: Mapping[str, object], field_name: str) -> tuple[bool, object]:
    def get_from(source: Mapping[str, object]) -> tuple[bool, object]:
        current: object = source
        for part in field_name.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return False, None
            current = current[part]
        return True, current

    found, value = get_from(context)
    if found:
        return found, value
    nested = context.get("frame")
    if isinstance(nested, Mapping):
        return get_from(nested)
    return False, None


def _same_json(left: object, right: object) -> bool:
    try:
        return _canonical(left) == _canonical(right)
    except (TypeError, ValueError, OverflowError):
        return left == right


def _predicate_matches(predicate: Mapping[str, object], context: Mapping[str, object]) -> bool:
    operator = str(predicate.get("op", "eq")).lower()
    if operator in {"all", "any"}:
        children = predicate.get("predicates", predicate.get("value", ()))
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes, bytearray)):
            return False
        results = [isinstance(child, Mapping) and _predicate_matches(child, context)
                   for child in children]
        return all(results) if operator == "all" else any(results)
    field_name = predicate.get("field")
    if not isinstance(field_name, str):
        return False
    found, actual = _lookup(context, field_name)
    if operator in {"exists", "present"}:
        return found
    expected = predicate.get("value")
    if operator in {"eq", "equals"}:
        return found and _same_json(actual, expected)
    if operator in {"ne", "not_equals"}:
        return not found or not _same_json(actual, expected)
    expected_plain = _thaw_json(expected)
    if operator == "in":
        return (found and isinstance(expected_plain, Sequence)
                and not isinstance(expected_plain, (str, bytes, bytearray))
                and any(_same_json(actual, item) for item in expected_plain))
    if operator == "not_in":
        return (not found or not (isinstance(expected_plain, Sequence)
                and not isinstance(expected_plain, (str, bytes, bytearray))
                and any(_same_json(actual, item) for item in expected_plain)))
    if operator in {"contains", "not_contains"}:
        try:
            result = expected_plain in _thaw_json(actual)  # type: ignore[operator]
        except (TypeError, ValueError):
            result = False
        return result if operator == "contains" else not result
    return False


def record_applies(record: MemoryRecord | Mapping[str, object],
                   context: Mapping[str, object] | None) -> bool:
    item = _as_record(record)
    return all(_predicate_matches(predicate, context if isinstance(context, Mapping) else {})
               for predicate in item.applicability)


def _authoritative_revisions(context: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(context, Mapping):
        return {}
    revisions: dict[str, object] = {}
    for key in ("artifact_revisions", "current_artifact_revisions", "dependency_revisions"):
        value = context.get(key)
        if isinstance(value, Mapping):
            revisions.update({str(name): revision for name, revision in value.items()})
    if context.get("environment_revision") is not None:
        revisions["environment_revision"] = context["environment_revision"]
    nested = context.get("frame")
    if isinstance(nested, Mapping):
        for key, value in _authoritative_revisions(nested).items():
            revisions.setdefault(key, value)
    return revisions


def _dependency_key_and_expected(dependency: object) -> tuple[str, object | None]:
    if isinstance(dependency, Mapping):
        key = next((dependency.get(name) for name in
                    ("key", "id", "name", "facet", "target", "path")
                    if dependency.get(name) is not None), None)
        expected = next((dependency.get(name) for name in
                         ("revision", "digest", "value")
                         if dependency.get(name) is not None), None)
        if key is None:
            return _canonical(dependency), expected
        return str(key), expected
    text = str(dependency)
    for separator in ("=", "@"):
        if separator in text:
            key, expected = text.split(separator, 1)
            if key and expected:
                return key, expected
    return text, None


def dependency_is_current(record: MemoryRecord | Mapping[str, object],
                          context: Mapping[str, object] | None) -> bool:
    """Check versioned dependencies against authoritative current revisions.

    A versioned dependency with no authoritative current value is stale by
    construction.  Unversioned labels are descriptive and remain usable until
    an explicit invalidation observation names them.
    """
    item = _as_record(record)
    revisions = _authoritative_revisions(context)
    for dependency in item.dependencies:
        key, expected = _dependency_key_and_expected(dependency)
        if expected is None:
            continue
        if key not in revisions or not _same_json(revisions[key], expected):
            return False
    return True


def _dependency_identity(value: object) -> str:
    if isinstance(value, Mapping):
        for key in ("id", "key", "name", "facet", "target", "path"):
            if value.get(key) is not None:
                return str(value[key])
        return _canonical(value)
    text = str(value)
    for separator in ("=", "@"):
        if separator in text:
            return text.split(separator, 1)[0]
    return text


def _changed_items(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        return tuple({"key": str(key), "value": child}
                     for key, child in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value, key=_canonical))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_DEPENDENCIES:
            raise MemoryCapsuleError("dependency change budget exceeded")
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _dependency_changed(dependency: object, changes: Sequence[object]) -> bool:
    dependency_id, dependency_expected = _dependency_key_and_expected(dependency)
    for changed in changes:
        changed_id, changed_value = _dependency_key_and_expected(changed)
        if dependency_id != changed_id:
            continue
        if isinstance(dependency, Mapping) or isinstance(changed, Mapping):
            if dependency_expected is None or changed_value is None:
                return True
            if not _same_json(dependency_expected, changed_value):
                return True
            continue
        if dependency_expected is None or changed_value is None:
            return True
        if not _same_json(dependency_expected, changed_value):
            return True
    return False


def invalidate_by_dependencies(
    records: Iterable[MemoryRecord | Mapping[str, object]],
    changed_dependencies: object,
) -> tuple[MemoryRecord, ...]:
    changes = _changed_items(changed_dependencies)
    result: list[MemoryRecord] = []
    for row_number, raw in enumerate(records, start=1):
        if row_number > MAX_INPUT_ROWS:
            raise MemoryCapsuleError("memory invalidation input row budget exceeded")
        record = _as_record(raw)
        if (record.lifecycle not in LIVE_LIFECYCLES
                or not any(_dependency_changed(dep, changes) for dep in record.dependencies)):
            result.append(record)
            continue
        changed = tuple(dict.fromkeys(_dependency_identity(item) for item in changes))
        result.append(replace(record, lifecycle="INVALIDATED", invalidated_by=changed))
    return tuple(result)


invalidate_dependencies = invalidate_by_dependencies


def _utility_rank(record: MemoryRecord) -> tuple[float, float, int, str, str, str]:
    utility = record.utility
    score = utility.benefit * max(utility.confidence, 0.1) - utility.cost
    return (-score, -utility.confidence, -utility.uses,
            record.kind, record.scope, record.record_id)


def _record_text(record: MemoryRecord) -> str:
    evidence = ", ".join(str(row.get("ref", "")) for row in record.evidence_receipts) or "none"
    return (f"Kind: {record.kind}\nScope: {record.scope}\nClaim: {record.claim}\n"
            f"Limit: {record.limit}\nEvidence: {evidence}")


def _model_message_json(text: str) -> str:
    return json.dumps({"role": "user", "content": text}, sort_keys=True,
                      ensure_ascii=False, allow_nan=False)


def serialized_message_chars(text: str) -> int:
    return len(_model_message_json(text))


def serialized_message_token_proxy(text: str) -> int:
    return (serialized_message_chars(text) + 3) // 4


def is_capsule_text(value: object) -> bool:
    return isinstance(value, str) and value.startswith(CAPSULE_HEADER_PREFIX)


@dataclass(frozen=True)
class MemoryCapsule:
    cap: int
    records: tuple[MemoryRecord, ...]
    text: str
    no_memory: bool
    token_count: int
    digest: str
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema, "cap": self.cap,
            "records": [record.to_dict() for record in self.records],
            "text": self.text, "no_memory": self.no_memory,
            "token_count": self.token_count, "digest": self.digest,
            "estimator": MODEL_CAP_ESTIMATOR,
        }

    def to_model_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema, "cap": self.cap, "text": self.text,
            "no_memory": self.no_memory, "token_count": self.token_count,
            "serialized_chars": serialized_message_chars(self.text),
            "digest": self.digest, "estimator": MODEL_CAP_ESTIMATOR,
        }

    def model_message(self) -> dict[str, str]:
        return {"role": "user", "content": self.text}


def _render_selected(records: Sequence[MemoryRecord], cap: int) -> str:
    header = f"{CAPSULE_HEADER_PREFIX} cap={cap}; records={len(records)}]"
    if not records:
        return header + "\nNo memory available."
    return "\n".join((header, *(
        f"Memory {index}:\n{_record_text(record)}"
        for index, record in enumerate(records, start=1))))


def _fit_selected(records: Sequence[MemoryRecord], cap: int) -> tuple[tuple[MemoryRecord, ...], str]:
    selected: list[MemoryRecord] = []
    for record in records:
        candidate = tuple((*selected, record))
        if serialized_message_token_proxy(_render_selected(candidate, cap)) <= cap:
            selected.append(record)
        if len(selected) >= MAX_CAPSULE_RECORDS:
            break
    chosen = tuple(selected)
    return chosen, _render_selected(chosen, cap)


def compile_capsule(
    records: Iterable[MemoryRecord | Mapping[str, object]] | "MemoryStore",
    *, context: Mapping[str, object] | None = None, cap: int = DEFAULT_CAP,
) -> MemoryCapsule:
    if cap not in CAPSULE_CAPS:
        raise ValueError("memory capsule cap must be 384, 768, or 1536")
    if isinstance(records, MemoryStore):
        records = records.active_records(context)
    selected: dict[str, MemoryRecord] = {}
    input_bytes = 0
    for row_number, raw in enumerate(records, start=1):
        if row_number > MAX_INPUT_ROWS:
            raise MemoryCapsuleError("memory compiler input row budget exceeded")
        record = _as_record(raw)
        input_bytes += len(_canonical(record.to_dict()).encode("utf-8"))
        if input_bytes > MAX_INPUT_BYTES:
            raise MemoryCapsuleError("memory compiler input byte budget exceeded")
        if (record.lifecycle not in LIVE_LIFECYCLES
                or not record_applies(record, context)
                or not dependency_is_current(record, context)):
            continue
        previous = selected.get(record.record_id)
        if previous is None or _canonical(record.to_dict()) < _canonical(previous.to_dict()):
            selected[record.record_id] = record
    ordered = tuple(sorted(selected.values(), key=_utility_rank)[:MAX_CAPSULE_RECORDS])
    exposed, text = _fit_selected(ordered, cap)
    token_count = serialized_message_token_proxy(text)
    payload = {"schema": SCHEMA, "cap": cap,
               "records": [record.to_dict() for record in exposed], "text": text,
               "no_memory": not exposed, "token_count": token_count,
               "estimator": MODEL_CAP_ESTIMATOR}
    return MemoryCapsule(cap=cap, records=exposed, text=text, no_memory=not exposed,
                         token_count=token_count, digest=_digest(payload))


render_capsule = compile_capsule
compile_memory_capsule = compile_capsule


def _record_preference(record: MemoryRecord) -> str:
    return _canonical(record.to_dict())


def _source_records(source: object) -> Iterator[MemoryRecord | Mapping[str, object]]:
    if isinstance(source, MemoryStore):
        yield from source.records
        return
    if isinstance(source, MemoryCapsule):
        yield from source.records
        return
    if isinstance(source, Mapping) and source.get("schema") == SCHEMA:
        restored = MemoryStore.from_state(source, frozen=True)
        yield from restored.records
        return
    if isinstance(source, Mapping) and "records" in source:
        source = source["records"]
    if isinstance(source, Mapping):
        yield source
        return
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        for item in source:
            if not isinstance(item, (Mapping, MemoryRecord)):
                raise MemoryRecordError("memory source rows must be objects")
            yield item
        return
    raise MemoryRecordError("memory source must be a record or finite record sequence")


class MemoryStore:
    """Bounded record store; a frozen instance has no public mutation path."""

    __slots__ = ("_records", "_frozen")

    def __init__(self, records: Iterable[MemoryRecord | Mapping[str, object]] = (), *,
                 frozen: bool = False):
        self._records: tuple[MemoryRecord, ...] = ()
        self._frozen = bool(frozen)
        input_bytes = 0
        for row_number, raw in enumerate(records, start=1):
            if row_number > MAX_INPUT_ROWS:
                raise MemoryCapsuleError("memory store input row budget exceeded")
            record = _as_record(raw)
            input_bytes += len(_canonical(record.to_dict()).encode("utf-8"))
            if input_bytes > MAX_INPUT_BYTES:
                raise MemoryCapsuleError("memory store input byte budget exceeded")
            self._put(record)

    @property
    def frozen(self) -> bool:
        return self._frozen

    def _put(self, record: MemoryRecord) -> None:
        rows = list(self._records)
        for index, existing in enumerate(rows):
            if existing.record_id != record.record_id:
                continue
            if _record_preference(record) < _record_preference(existing):
                rows[index] = record
            self._records = tuple(rows)
            return
        if len(rows) >= MAX_RECORDS:
            raise MemoryCapsuleError("memory store capacity reached")
        rows.append(record)
        self._records = tuple(rows)

    @property
    def records(self) -> tuple[MemoryRecord, ...]:
        return self._records

    def all_records(self) -> tuple[MemoryRecord, ...]:
        return self.records

    def active_records(self, context: Mapping[str, object] | None = None) -> tuple[MemoryRecord, ...]:
        return tuple(record for record in self._records
                     if record.lifecycle in LIVE_LIFECYCLES
                     and record_applies(record, context)
                     and dependency_is_current(record, context))

    def add(self, record: MemoryRecord | Mapping[str, object]) -> MemoryRecord:
        if self.frozen:
            raise RuntimeError("memory store is frozen")
        item = _as_record(record)
        self._put(item)
        return item

    append = add

    def invalidate(self, changed_dependencies: object) -> tuple[MemoryRecord, ...]:
        if self.frozen:
            raise RuntimeError("memory store is frozen")
        updated = invalidate_by_dependencies(self._records, changed_dependencies)
        self._records = tuple(updated)
        return updated

    invalidate_dependencies = invalidate
    refresh = invalidate

    def update_utility(self, record_id: str, *, benefit_delta: float = 0.0,
                       cost_delta: float = 0.0, use_delta: int = 0) -> MemoryRecord:
        if self.frozen:
            raise RuntimeError("memory store is frozen")
        for index, record in enumerate(self._records):
            if record.record_id != record_id:
                continue
            utility = record.utility
            updated = replace(record, utility=Utility(
                benefit=utility.benefit + float(benefit_delta),
                cost=max(0.0, utility.cost + float(cost_delta)),
                uses=max(0, utility.uses + int(use_delta)),
                confidence=utility.confidence,
            ))
            rows = list(self._records)
            rows[index] = updated
            self._records = tuple(rows)
            return updated
        raise KeyError(record_id)

    def compile(self, context: Mapping[str, object] | None = None, *,
                cap: int = DEFAULT_CAP) -> MemoryCapsule:
        return compile_capsule(self, context=context, cap=cap)

    def snapshot(self) -> dict[str, object]:
        return {"schema": SCHEMA, "frozen": self.frozen,
                "records": [record.to_dict() for record in self._records],
                "content_digest": self.content_digest}

    @property
    def content_digest(self) -> str:
        return _digest({"schema": SCHEMA,
                        "records": [record.to_dict() for record in self._records]})

    @classmethod
    def from_state(cls, state: object, *, frozen: bool | None = None) -> "MemoryStore":
        if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
            raise MemoryCapsuleError("unsupported memory store schema")
        if frozen is True and state.get("frozen") is not True:
            raise MemoryCapsuleError("memory store state is not frozen")
        expected_digest = state.get("content_digest")
        if not isinstance(expected_digest, str) or not expected_digest.startswith("sha256:"):
            raise MemoryCapsuleError("memory store content digest is required")
        rows = state.get("records", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise MemoryCapsuleError("memory store records must be a list")
        if len(rows) > MAX_INPUT_ROWS:
            raise MemoryCapsuleError("memory store state row budget exceeded")
        store = cls(rows, frozen=bool(state.get("frozen", False) if frozen is None else frozen))
        if expected_digest != store.content_digest:
            raise MemoryCapsuleError("memory store content digest mismatch")
        return store

    @classmethod
    def from_owner(cls, owner: object, *, frozen: bool = True) -> "MemoryStore":
        """Read explicit owner/task inputs only; never crawl run directories."""
        sources: list[object] = []
        for name in ("memory_records", "memory_bank", "memory_capsule_records"):
            value = getattr(owner, name, None)
            if value is not None:
                sources.append(value)
        row = getattr(owner, "task_row", {})
        if isinstance(row, Mapping):
            for name in ("memory_records", "memory_bank", "memory_capsule_records"):
                if row.get(name) is not None:
                    sources.append(row[name])
        path = os.environ.get("STATEFRAME_MEMORY_CAPSULE_STORE", "").strip()
        if path:
            sources.append(load_json_records(Path(path)))

        def records() -> Iterator[MemoryRecord | Mapping[str, object]]:
            for source in sources:
                yield from _source_records(source)

        return cls(records(), frozen=frozen)

    from_runtime = from_owner


MemoryCapsuleStore = MemoryStore


def permissions_from_memory(records: object) -> tuple[object, ...]:
    if isinstance(records, MemoryCapsule):
        records = records.records
    if isinstance(records, Mapping):
        records = (records,)
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        for record in records:
            _reject_permission_fields(_as_record(record).to_dict(), label="record")
    return ()


def memory_can_grant_permissions(*_args: object, **_kwargs: object) -> bool:
    return False


def apply_memory_to_tools(tools: object, *_args: object, **_kwargs: object) -> object:
    return tools


def append_capsule_to_frame(
    frame: Mapping[str, object], store: MemoryStore | None, *,
    context: Mapping[str, object] | None = None, cap: int = DEFAULT_CAP,
) -> dict[str, object]:
    projected = dict(frame)
    if store is None:
        return projected
    frame_context = dict(context or {})
    frame_context.setdefault("frame", frame)
    capsule = store.compile(frame_context, cap=cap)
    projected[CAPSULE_FRAME_FIELD] = capsule.to_model_dict()
    return projected


frame_projection = append_capsule_to_frame


def prepared_capsule_message(
    store: MemoryStore | None, *, context: Mapping[str, object] | None = None,
    cap: int = DEFAULT_CAP,
) -> tuple[MemoryCapsule, dict[str, str]] | None:
    if store is None:
        return None
    capsule = store.compile(context or {}, cap=cap)
    message = capsule.model_message()
    if serialized_message_token_proxy(message["content"]) > cap:
        raise MemoryCapsuleError("prepared memory capsule exceeded its serialized cap")
    return capsule, message


def _guard_path(path: Path) -> None:
    protected = {"private", "hidden_tests", "near_miss", "gold", "oracle"}
    resolved = path.expanduser().resolve(strict=False)
    if any(part.lower() in protected for part in resolved.parts):
        raise ValueError("refusing to read protected memory data path")


def _bounded_lines(handle: TextIO, *, max_bytes: int) -> Iterator[bytes]:
    line = bytearray()
    while True:
        chunk = handle.read(4096)
        if not chunk:
            if line:
                yield bytes(line)
            return
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        if not isinstance(chunk, (bytes, bytearray)):
            raise MemoryCapsuleError("memory source stream returned non-text data")
        parts = bytes(chunk).split(b"\n")
        for index, part in enumerate(parts):
            line.extend(part)
            physical_bytes = len(line) + (1 if index < len(parts) - 1 else 0)
            if physical_bytes > max_bytes:
                raise MemoryCapsuleError("memory input line exceeds the bounded input size")
            if index < len(parts) - 1:
                yield bytes(line)
                line.clear()


def load_json_records(source: str | Path | TextIO) -> tuple[dict[str, object], ...]:
    """Parse bounded JSONL in place; protected run trees are never read."""
    close = False
    if isinstance(source, (str, Path)):
        path = Path(source)
        _guard_path(path)
        handle = path.open("rb")
        close = True
    else:
        if getattr(source, "stateframe_memory_trusted", False) is not True:
            raise ValueError("trusted memory stream provenance is required")
        handle = source
    rows: list[dict[str, object]] = []
    total_bytes = 0
    try:
        for line_number, line in enumerate(_bounded_lines(handle, max_bytes=MAX_RECORD_BYTES), start=1):
            total_bytes += len(line) + 1
            if line_number > MAX_INPUT_ROWS or total_bytes > MAX_INPUT_BYTES:
                raise MemoryCapsuleError("memory input budget exceeded")
            if not line.strip():
                continue
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise MemoryCapsuleError(f"invalid memory JSON at row {line_number}") from error
            if isinstance(value, Mapping) and isinstance(value.get("records"), list):
                if len(rows) + len(value["records"]) > MAX_INPUT_ROWS:
                    raise MemoryCapsuleError("memory record budget exceeded")
                if any(not isinstance(item, Mapping) for item in value["records"]):
                    raise MemoryCapsuleError(f"memory row {line_number} contains malformed records")
                rows.extend(dict(item) for item in value["records"])
            elif isinstance(value, Mapping):
                rows.append(dict(value))
            else:
                raise MemoryCapsuleError(f"memory row {line_number} must be an object")
    finally:
        if close:
            handle.close()
    return tuple(rows)


__all__ = [
    "CAPSULE_CAPS", "CAPSULE_FRAME_FIELD", "CAPSULE_HEADER_PREFIX", "DEFAULT_CAP",
    "MODE_ENV", "MODEL_CAP_ESTIMATOR", "SCHEMA", "RECORD_SCHEMA",
    "MemoryCapsule", "MemoryCapsuleError", "MemoryRecord", "MemoryRecordError",
    "MemorySecurityError", "MemoryStore", "MemoryCapsuleStore", "Utility", "FrozenDict",
    "apply_memory_to_tools", "append_capsule_to_frame", "compile_capsule",
    "compile_memory_capsule", "configured_cap", "enabled", "frame_projection",
    "invalidate_by_dependencies", "invalidate_dependencies", "load_json_records",
    "memory_can_grant_permissions", "mode", "permissions_from_memory",
    "prepared_capsule_message", "record_applies", "dependency_is_current",
    "render_capsule", "serialized_message_chars", "serialized_message_token_proxy",
    "is_capsule_text",
]
