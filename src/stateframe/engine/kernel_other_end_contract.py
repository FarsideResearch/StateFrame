"""Host-owned binding of both ends of a changed boundary identity.

The binder is deliberately a consumer of the where-used organ.  It does not
walk a repository or resolve references a second time.  The closure report
supplies the changed identities and evidence sites; the coding pack adds typed
endpoint rows (serializer/schema, route, message, or column) when it can bind
their roles.  Only endpoints with the same typed binding are compared.

This is a TRUTH mechanism.  It is model agnostic, and its strict switch is
default-off so an unconfigured runtime has the exact baseline projection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from kernel_obligation_assess import (
    ADMITTED,
    APPLIES,
    APPLICABILITY_UNKNOWN,
    CheckBundle,
    CheckDefinition,
    ContractCell,
    ObligationSpec,
    PROVISIONAL,
    UNBOUND,
    VALIDATED,
)


MODE_ENV = "STATEFRAME_OTHER_END_CONTRACT"
SCHEMA = "stateframe-other-end-contract/1"
EVENT_KIND = "other_end_contract_fired"
ORPHAN_EVENT_KIND = "contract_orphan"
CONTRACT_ORPHAN = "CONTRACT_ORPHAN"
CONTRACT_DISPUTED = "CONTRACT_DISPUTED"

PRODUCER = "producer"
CONSUMER = "consumer"
_ROLES = frozenset({PRODUCER, CONSUMER})
_BOUNDARY_FIELDS = (
    "kind", "transport", "service", "method", "route", "version", "schema",
    "field_path", "broker", "topic", "table", "column", "scope", "contract_id",
    "binding_id",
)
_TYPED_FIELDS = (
    "kind", "service", "method", "route", "version", "schema", "field_path",
    "broker", "topic", "table", "column", "scope", "contract_id", "binding_id",
    "wire_name",
)
_ROLE_ALIASES = {
    "writer": PRODUCER,
    "write": PRODUCER,
    "source": PRODUCER,
    "out": PRODUCER,
    "output": PRODUCER,
    "reader": CONSUMER,
    "read": CONSUMER,
    "sink": CONSUMER,
    "in": CONSUMER,
    "input": CONSUMER,
}
_KIND_ALIASES = {
    "serializer": "schema",
    "serialization": "schema",
    "json": "schema",
    "dict": "schema",
    "mapping": "schema",
    "route_table": "route",
    "http": "route",
    "endpoint": "route",
    "orm": "column",
    "database": "column",
    "db": "column",
    "queue": "message",
    "broker": "message",
    "event": "message",
}
_IDENTITY_KEYS = (
    "identity", "wire_name", "field_path", "field", "key", "route", "topic", "column",
    "name",
)
_ENDPOINT_COLLECTION_KEYS = (
    "boundary_endpoints", "contract_endpoints", "typed_endpoints", "endpoints",
    "boundary_bindings", "contract_bindings",
)
_OUTPUT_KEYS = ("outputs", "produces", "emits", "writes", "output_values")
_INPUT_KEYS = ("accepts", "consumes", "reads", "inputs", "accepted_values")


def mode() -> str:
    """Return the strict default-off switch value."""
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


TIER_ENV = "STATEFRAME_OTHER_END_CONTRACT_TIER"


def tier_mode() -> str:
    """§62 gating tier: `admitted` (default; only admitted violations block) or `strict` (lane-original)."""
    value = os.environ.get(TIER_ENV, "admitted").strip().lower()
    if value not in {"admitted", "strict"}:
        raise ValueError(f"{TIER_ENV} must be admitted or strict")
    return value


def _copy_json(value: object) -> object:
    """Make endpoint evidence safe to put in a frozen report and journal."""
    try:
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str)
    except (TypeError, ValueError):
        return str(value)
    return copy.deepcopy(value)


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _values(value: object) -> tuple[str, ...] | None:
    """Normalize an optional finite value set without inventing one."""
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        items = value.keys()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        return (str(value),)
    result: list[str] = []
    for item in items:
        rendered = str(item)
        if rendered not in result:
            result.append(rendered)
    return tuple(result)


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        return dict(converted) if isinstance(converted, Mapping) else {}
    return {}


def _role(value: object) -> str:
    raw = _text(value).lower().replace("-", "_").replace(" ", "_")
    return _ROLE_ALIASES.get(raw, raw if raw in _ROLES else "")


def _kind(value: object) -> str:
    raw = _text(value).lower().replace("-", "_").replace(" ", "_")
    return _KIND_ALIASES.get(raw, raw)


def _identity(row: Mapping[str, object]) -> str:
    for key in _IDENTITY_KEYS:
        value = _text(row.get(key))
        if value:
            return value
    return ""


def _line_text(row: Mapping[str, object]) -> str:
    value = row.get("text", row.get("source_text", row.get("evidence", "")))
    return value if isinstance(value, str) else ""


def _infer_boundary(row: Mapping[str, object], identity: str) -> dict[str, object]:
    raw = row.get("boundary", row.get("typed_endpoint", row.get("endpoint", {})))
    boundary: dict[str, object] = dict(raw) if isinstance(raw, Mapping) else {}
    if isinstance(raw, str) and raw.strip():
        boundary.setdefault("kind", raw)
    occurrence_kinds = {"literal", "reference", "definition", "import", "producer", "consumer"}
    for key in _BOUNDARY_FIELDS:
        if (key in row and key not in boundary and row.get(key) is not None
                and not (key == "kind" and _text(row.get(key)).lower() in occurrence_kinds)):
            boundary[key] = row[key]
    text = _line_text(row)
    path = _text(row.get("path")).lower()
    lowered = f"{text} {path}".lower()
    if not boundary.get("kind"):
        if identity.startswith("/") or "route" in lowered or "endpoint" in lowered:
            boundary["kind"] = "route"
        elif re.search(r"\b(mapped_column|column|select\(|insert\(|update\()", lowered):
            boundary["kind"] = "column"
        elif re.search(r"\b(publish|consume|queue|topic|broker|message|emit)\b", lowered):
            boundary["kind"] = "message"
        elif (re.search(r"\b(json|serialize|deserialize|payload|request|response|dict|mapping)\b", lowered)
              or re.search(r"\{\s*['\"]", text)):
            boundary["kind"] = "schema"
    if boundary.get("kind"):
        boundary["kind"] = _kind(boundary["kind"])
    if identity.startswith("/"):
        boundary.setdefault("route", identity)
    elif boundary.get("kind") == "route":
        boundary.setdefault("route", identity)
    elif boundary.get("kind") == "column":
        boundary.setdefault("column", identity)
    elif boundary.get("kind") == "message":
        boundary.setdefault("topic", identity)
    else:
        boundary.setdefault("field_path", identity)
    return boundary


def _infer_role(row: Mapping[str, object], identity: str) -> str:
    raw = _role(row.get("role", row.get("endpoint_role", row.get("side"))))
    if raw:
        return raw
    kind = _text(row.get("kind")).lower()
    if kind in _ROLES or kind in _ROLE_ALIASES:
        return _role(kind)
    text = _line_text(row)
    path = _text(row.get("path")).lower()
    if re.search(r"(?:\[\s*['\"]|\.get\(\s*['\"]|json\.loads|deserialize|parse|request\.)", text, re.I):
        return CONSUMER
    if re.search(r"(?:\[\s*['\"]\w+['\"]\s*\]\s*=|json\.dumps|serialize|to_dict|publish|send|emit)", text, re.I):
        return PRODUCER
    if re.search(r"(?:consumer|reader|deserialize|parse|request|receive|fetch|frontend)", path, re.I):
        return CONSUMER
    if re.search(r"(?:producer|writer|serialize|response|publish|route)", path, re.I):
        return PRODUCER
    return ""


def _boundary_rows(value: object, *, defaults: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    """Flatten pair-shaped or list-shaped coding-pack endpoint material."""
    defaults = defaults or {}
    rows: list[dict[str, object]] = []
    if isinstance(value, Mapping):
        if "producer" in value or "consumer" in value:
            for role in (PRODUCER, CONSUMER):
                sides = value.get(role)
                if sides is None:
                    continue
                if isinstance(sides, Mapping):
                    sides = (sides,)
                elif isinstance(sides, str):
                    sides = ({"surface": sides},)
                if isinstance(sides, Sequence) and not isinstance(sides, (bytes, bytearray, str)):
                    for side in sides:
                        if isinstance(side, Mapping):
                            row = {**defaults, **dict(value), **dict(side), "role": role}
                            rows.append(row)
            if rows:
                return rows
        if "path" in value or any(key in value for key in _IDENTITY_KEYS) or "role" in value:
            rows.append({**defaults, **dict(value)})
            return rows
        for key, item in value.items():
            if isinstance(item, Mapping):
                rows.extend(_boundary_rows(item, defaults={**defaults, "identity": key}))
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                rows.extend(_boundary_rows(item, defaults={**defaults, "identity": key}))
        return rows
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            rows.extend(_boundary_rows(item, defaults=defaults))
    return rows


def _endpoint_rows(closure: Mapping[str, object], explicit: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if explicit is not None:
        rows.extend(_boundary_rows(explicit))
    for key in _ENDPOINT_COLLECTION_KEYS:
        if key in closure:
            rows.extend(_boundary_rows(closure[key]))
    # A where-used implementation may attach roles directly to its typed site
    # records.  Sites without role/boundary evidence stay ordinary references.
    # A closure can carry both the typed coding-pack rows and the raw
    # where-used sites that motivated them.  Do not reinterpret the latter a
    # second time: its literal ``kind`` is an occurrence kind, not a typed
    # endpoint kind, and doing so creates duplicate bindings.  Site rows remain
    # a useful fallback when no coding-pack endpoint collection was supplied.
    sites = closure.get("sites", ())
    if not rows and isinstance(sites, Sequence) and not isinstance(sites, (str, bytes, bytearray)):
        for site in sites:
            if not isinstance(site, Mapping):
                continue
            if (_role(site.get("role", site.get("endpoint_role", site.get("side"))))
                    or isinstance(site.get("boundary"), Mapping)
                    or isinstance(site.get("typed_endpoint"), Mapping)
                    or _infer_role(site, _identity(site))):
                rows.append(dict(site))
    return rows


@dataclass(frozen=True, slots=True)
class Endpoint:
    identity: str
    wire_name: str
    role: str
    boundary: dict[str, object]
    path: str
    surface: str
    site_id: str = ""
    source: str = "where_used"
    outputs: tuple[str, ...] | None = None
    accepts: tuple[str, ...] | None = None
    aliases: tuple[str, ...] = ()

    @property
    def endpoint_id(self) -> str:
        return "endpoint-" + hashlib.sha256(
            json.dumps(self.to_dict(include_id=False), sort_keys=True,
                       ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:20]

    @property
    def typed_binding(self) -> tuple[tuple[str, str], ...]:
        boundary = self.boundary
        contract_id = _text(boundary.get("contract_id") or boundary.get("binding_id"))
        if contract_id:
            return (("contract_id", contract_id),)
        result: list[tuple[str, str]] = []
        for key in _TYPED_FIELDS:
            if key == "wire_name":
                value = self.wire_name
            else:
                value = boundary.get(key)
            rendered = _text(value)
            if not rendered:
                continue
            if key == "kind":
                rendered = _kind(rendered)
            result.append((key, rendered))
        return tuple(result)

    def to_dict(self, *, include_id: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "identity": self.identity,
            "wire_name": self.wire_name,
            "role": self.role,
            "boundary": _copy_json(self.boundary),
            "path": self.path,
            "surface": self.surface,
            "site_id": self.site_id,
            "source": self.source,
            "outputs": list(self.outputs) if self.outputs is not None else None,
            "accepts": list(self.accepts) if self.accepts is not None else None,
            "aliases": list(self.aliases),
            "typed_binding": [list(item) for item in self.typed_binding],
        }
        if include_id:
            result["endpoint_id"] = self.endpoint_id
        return result


def _endpoint_from_row(row: Mapping[str, object]) -> Endpoint | None:
    identity = _identity(row)
    boundary = _infer_boundary(row, identity)
    wire_name = _text(row.get("wire_name"))
    if not wire_name:
        for key in ("field_path", "key", "route", "topic", "column"):
            wire_name = _text(boundary.get(key))
            if wire_name:
                break
    wire_name = wire_name or identity
    role = _infer_role(row, identity)
    if not identity or role not in _ROLES:
        return None
    path = _text(row.get("path", row.get("file", "")))
    surface = _text(row.get("surface", row.get("name", ""))) or wire_name
    outputs = next((row.get(key) for key in _OUTPUT_KEYS if key in row), None)
    accepts = next((row.get(key) for key in _INPUT_KEYS if key in row), None)
    aliases = _values(row.get("aliases")) or ()
    return Endpoint(
        identity=identity,
        wire_name=wire_name,
        role=role,
        boundary=boundary,
        path=path,
        surface=surface,
        site_id=_text(row.get("site_id")),
        source=_text(row.get("source", "where_used")) or "where_used",
        outputs=_values(outputs),
        accepts=_values(accepts),
        aliases=aliases,
    )


@dataclass(frozen=True, slots=True)
class ContractObligation:
    obligation_id: str
    identity: str
    status: str
    reason_code: str
    tier: str
    missing_role: str = ""
    required_action: str = ""
    producer_endpoint_ids: tuple[str, ...] = ()
    consumer_endpoint_ids: tuple[str, ...] = ()
    contract_cell: ContractCell | None = None
    evidence_refs: tuple[str, ...] = ()
    dispute: str | None = None

    @property
    def blocking(self) -> bool:
        if tier_mode() == "strict":
            return self.status in {"OPEN", "ADMITTED_VIOLATION", "DISPUTED"}
        return self.tier == "ADMITTED_VIOLATION"

    def to_dict(self) -> dict[str, object]:
        return {
            "obligation_id": self.obligation_id,
            "identity": self.identity,
            "status": self.status,
            "reason_code": self.reason_code,
            "tier": self.tier,
            "missing_role": self.missing_role,
            "required_action": self.required_action,
            "producer_endpoint_ids": list(self.producer_endpoint_ids),
            "consumer_endpoint_ids": list(self.consumer_endpoint_ids),
            "contract_cell": self.contract_cell.to_dict() if self.contract_cell else None,
            "evidence_refs": list(self.evidence_refs),
            "dispute": self.dispute,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class ContractBinding:
    binding_id: str
    identity: str
    typed_binding: tuple[tuple[str, str], ...]
    producers: tuple[Endpoint, ...]
    consumers: tuple[Endpoint, ...]
    compatibility: str
    obligation: ContractObligation
    contract_cell: ContractCell
    check: CheckDefinition

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "identity": self.identity,
            "typed_binding": [list(item) for item in self.typed_binding],
            "producers": [item.to_dict() for item in self.producers],
            "consumers": [item.to_dict() for item in self.consumers],
            "compatibility": self.compatibility,
            "obligation": self.obligation.to_dict(),
            "contract_cell": self.contract_cell.to_dict(),
            "check": self.check.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ContractReport:
    fired: bool
    closure_event_id: str | None = None
    changed_identities: tuple[str, ...] = ()
    endpoints: tuple[Endpoint, ...] = ()
    bindings: tuple[ContractBinding, ...] = ()
    orphans: tuple[ContractBinding, ...] = ()
    unresolved: tuple[ContractBinding, ...] = ()
    event_id: str | None = None
    journaled: bool = False
    coverage: dict[str, object] | None = None

    @property
    def blocking(self) -> bool:
        return any(item.obligation.blocking for item in self.bindings)

    @property
    def obligations(self) -> tuple[ContractObligation, ...]:
        return tuple(item.obligation for item in self.bindings)

    @property
    def contract_cells(self) -> tuple[ContractCell, ...]:
        return tuple(item.contract_cell for item in self.bindings)

    @property
    def checks(self) -> tuple[CheckDefinition, ...]:
        return tuple(item.check for item in self.bindings)

    def to_dict(self) -> dict[str, object]:
        if not self.fired:
            return {"schema": SCHEMA, "fired": False}
        return {
            "schema": SCHEMA,
            "fired": True,
            "closure_event_id": self.closure_event_id,
            "changed_identities": list(self.changed_identities),
            "endpoints": [item.to_dict() for item in self.endpoints],
            "bindings": [item.to_dict() for item in self.bindings],
            "orphan_obligation_ids": [item.obligation.obligation_id for item in self.orphans],
            "unresolved_obligation_ids": [item.obligation.obligation_id for item in self.unresolved],
            "event_id": self.event_id,
            "journaled": self.journaled,
            "coverage": _copy_json(self.coverage or {}),
            "blocking": self.blocking,
        }


def _changed_identities(closure: Mapping[str, object]) -> tuple[str, ...]:
    raw = closure.get("changed_identities", ())
    values: list[str] = []
    if isinstance(raw, Mapping):
        raw = raw.keys()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        values.extend(_text(item) for item in raw)
    elif isinstance(raw, str):
        values.append(raw.strip())
    if not values:
        changes = closure.get("identity_changes", ())
        if isinstance(changes, Mapping):
            changes = changes.values()
        if isinstance(changes, Sequence) and not isinstance(changes, (str, bytes, bytearray)):
            for item in changes:
                if isinstance(item, Mapping):
                    values.append(_text(item.get("identity")))
    return tuple(dict.fromkeys(item for item in values if item))


def _relevant(endpoint: Endpoint, changed: set[str]) -> bool:
    if not changed:
        return True
    names = {endpoint.identity, endpoint.wire_name, *endpoint.aliases}
    return bool(names & changed)


def _group_key(endpoint: Endpoint) -> tuple[object, ...]:
    typed = endpoint.typed_binding
    return typed if typed else (("unbound_endpoint", endpoint.endpoint_id),)


def _compatibility(producers: Sequence[Endpoint], consumers: Sequence[Endpoint]) -> tuple[str, str]:
    if not producers:
        return "ORPHAN", "missing producer endpoint"
    if not consumers:
        return "ORPHAN", "missing consumer endpoint"
    produced: set[str] = set()
    accepted: set[str] | None = None
    for endpoint in producers:
        if endpoint.outputs is None:
            return "UNKNOWN", "producer output domain requires a contract check"
        produced.update(endpoint.outputs)
    for endpoint in consumers:
        if endpoint.accepts is None:
            return "UNKNOWN", "consumer input domain requires a contract check"
        values = set(endpoint.accepts)
        accepted = values if accepted is None else accepted & values
    if accepted is None:
        return "UNKNOWN", "contract check has no admitted input domain"
    if produced <= accepted:
        return "COMPATIBLE", "producer outputs are accepted by every consumer"
    return "INCOMPATIBLE", "producer outputs are not a subset of consumer accepts"


def _template(producers: Sequence[Endpoint], consumers: Sequence[Endpoint]) -> str:
    all_endpoints = (*producers, *consumers)
    kinds = {_text(item.boundary.get("kind")) for item in all_endpoints}
    return "ERROR_TRANSLATION" if "error" in kinds else "VALUE_TRANSPORT"


def _make_binding(
    identity: str,
    key: tuple[object, ...],
    producers: tuple[Endpoint, ...],
    consumers: tuple[Endpoint, ...],
    *,
    closure_event_id: str | None,
) -> ContractBinding:
    compatibility, detail = _compatibility(producers, consumers)
    binding_id = "contract-" + hashlib.sha256(
        json.dumps({"identity": identity, "typed_binding": list(key)}, sort_keys=True,
                   default=str).encode("utf-8")
    ).hexdigest()[:20]
    obligation_id = f"contract:{binding_id}"
    missing_role = ""
    if compatibility == "ORPHAN":
        missing_role = PRODUCER if not producers else CONSUMER
        status = "OPEN"
        reason_code = CONTRACT_ORPHAN
        # A one-sided group is unproven (planned sibling, external consumer, missed binding): check candidate
        # unless the strict tier asks for the lane-original admitted treatment.
        tier = "ADMITTED_VIOLATION" if tier_mode() == "strict" else "CHECK_CANDIDATE"
        required_action = f"Bind the missing {missing_role} endpoint for this boundary contract"
        admission = UNBOUND
    elif compatibility == "INCOMPATIBLE":
        status = "ADMITTED_VIOLATION"
        reason_code = "CONTRACT_INCOMPATIBLE"
        tier = "ADMITTED_VIOLATION"
        required_action = "Satisfy the directional compatibility obligation: producer outputs must be accepted"
        admission = ADMITTED
    elif compatibility == "UNKNOWN":
        status = "OPEN"
        reason_code = "CONTRACT_CHECK_REQUIRED"
        tier = "CHECK_CANDIDATE"
        required_action = "Run the generated producer-to-consumer contract check"
        admission = PROVISIONAL
    else:
        status = "CLEARED"
        reason_code = "CONTRACT_COMPATIBLE"
        tier = "CLEARED"
        required_action = "Retain the host contract witness; no file change is required"
        admission = ADMITTED

    surface_parts = [item.surface for item in (*producers, *consumers) if item.surface]
    surface = " -> ".join(dict.fromkeys(surface_parts)) or identity
    fixture = {
        "producer": [item.to_dict() for item in producers],
        "consumer": [item.to_dict() for item in consumers],
        "compatibility": compatibility,
    }
    cell = ContractCell(
        obligation=obligation_id,
        surface=surface,
        case_partition={
            "kind": "other_end_contract",
            "identity": identity,
            "typed_binding": [list(item) for item in key],
            "direction": "producer_to_consumer",
            "compatibility": compatibility,
        },
        fixture=fixture,
        observable_selector={"producer": "output", "consumer": "input"},
        predicate={
            "kind": "directional_compatibility",
            "rule": "producer_outputs_subset_of_consumer_accepts",
            "detail": detail,
        },
        oracle_sources=tuple(item for item in (
            f"where_used:{closure_event_id}" if closure_event_id else "where_used:closure",
            f"boundary_template:{_template(producers, consumers)}",
        ) if item),
        applicability=APPLIES if key and compatibility != "UNKNOWN" else APPLICABILITY_UNKNOWN,
        admission=admission,
    )
    check = CheckDefinition(
        check_id=f"check:{binding_id}",
        obligation_id=obligation_id,
        template="COMPATIBILITY_WITNESS",
        entry_point=None,
        fixture=fixture,
        expected={"compatibility": "COMPATIBLE"},
        provenance={
            "producer": "kernel_other_end_contract",
            "authority": "HOST_OBSERVED",
            "direction": "producer_to_consumer",
        },
        oracle_basis="PRESERVED_BEHAVIOR",
        contract_cell=cell,
    )
    obligation = ContractObligation(
        obligation_id=obligation_id,
        identity=identity,
        status=status,
        reason_code=reason_code,
        tier=tier,
        missing_role=missing_role,
        required_action=required_action,
        producer_endpoint_ids=tuple(item.endpoint_id for item in producers),
        consumer_endpoint_ids=tuple(item.endpoint_id for item in consumers),
        contract_cell=cell,
        evidence_refs=tuple(item.site_id for item in (*producers, *consumers) if item.site_id),
    )
    return ContractBinding(
        binding_id=binding_id,
        identity=identity,
        typed_binding=tuple((str(name), str(value)) for name, value in key),
        producers=producers,
        consumers=consumers,
        compatibility=compatibility,
        obligation=obligation,
        contract_cell=cell,
        check=check,
    )


def _emit(journal: object | None, kind: str, **fields: object) -> str | None:
    if journal is None:
        return None
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("contract binder journal must expose emit()")
    event = emitter(kind, **fields)
    if isinstance(event, Mapping):
        value = event.get("event_id")
        return str(value) if value else None
    return None


def bind_contracts(
    closure: object,
    *,
    journal: object | None = None,
    endpoints: object = None,
    edit_receipt_id: str | None = None,
) -> ContractReport:
    """Bind typed producer/consumer pairs from one where-used closure report.

    The OFF branch returns before reading ``closure`` or ``endpoints``.  When
    enabled, absent closure data is recorded as insufficient coverage rather
    than being silently treated as a complete contract.
    """
    if mode() != "on":
        return ContractReport(False)

    closure_map = _mapping(closure)
    closure_event_id = _text(closure_map.get("event_id")) or None
    changed = _changed_identities(closure_map)
    raw_rows = _endpoint_rows(closure_map, endpoints)
    normalized: list[Endpoint] = []
    seen: set[str] = set()
    changed_set = set(changed)
    for raw in raw_rows:
        endpoint = _endpoint_from_row(raw)
        if endpoint is None or not _relevant(endpoint, changed_set):
            continue
        if endpoint.endpoint_id in seen:
            continue
        seen.add(endpoint.endpoint_id)
        normalized.append(endpoint)

    grouped: dict[tuple[object, ...], list[Endpoint]] = defaultdict(list)
    for endpoint in normalized:
        grouped[_group_key(endpoint)].append(endpoint)
    bindings: list[ContractBinding] = []
    for key, rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        producers = tuple(item for item in rows if item.role == PRODUCER)
        consumers = tuple(item for item in rows if item.role == CONSUMER)
        identity = (producers[0].wire_name if producers else consumers[0].wire_name
                    if consumers else "")
        bindings.append(_make_binding(identity, key, producers, consumers,
                                      closure_event_id=closure_event_id))
    bindings_tuple = tuple(bindings)
    orphans = tuple(item for item in bindings_tuple
                    if item.obligation.reason_code == CONTRACT_ORPHAN)
    unresolved = tuple(item for item in bindings_tuple
                       if item.obligation.status == "OPEN" and item not in orphans)
    coverage = {
        "closure_available": bool(closure_map),
        "closure_fired": bool(closure_map.get("fired", False)),
        "binding_source": "where_used_closure",
        "typed_endpoint_count": len(normalized),
        "changed_identity_count": len(changed),
        "unbound_endpoint_count": sum(not item.typed_binding for item in normalized),
        "external_consumers": "UNKNOWN",
    }
    event_id = _emit(
        journal,
        EVENT_KIND,
        producer="kernel_other_end_contract",
        authority="HOST_OBSERVED",
        fired=True,
        closure_event_id=closure_event_id,
        edit_receipt_id=edit_receipt_id,
        changed_identities=list(changed),
        typed_endpoint_count=len(normalized),
        binding_count=len(bindings_tuple),
        orphan_count=len(orphans),
        unresolved_count=len(unresolved),
        contract_check_count=len(bindings_tuple),
        coverage=coverage,
        dispatch=False,
    )
    for binding in orphans:
        _emit(
            journal,
            ORPHAN_EVENT_KIND,
            producer="kernel_other_end_contract",
            authority="HOST_OBSERVED",
            fired=True,
            obligation_id=binding.obligation.obligation_id,
            identity=binding.identity,
            missing_role=binding.obligation.missing_role,
            reason_code=CONTRACT_ORPHAN,
            typed_binding=[list(item) for item in binding.typed_binding],
            dispatch=False,
        )
    return ContractReport(
        fired=True,
        closure_event_id=closure_event_id,
        changed_identities=changed,
        endpoints=tuple(normalized),
        bindings=bindings_tuple,
        orphans=orphans,
        unresolved=unresolved,
        event_id=event_id,
        journaled=event_id is not None,
        coverage=coverage,
    )


def build_contract_bundle(report: ContractReport, *, task_id: str = "other-end-contract") -> CheckBundle | None:
    """Adapt generated checks to the existing host closure vocabulary."""
    if not report.fired:
        return None
    specs = tuple(
        ObligationSpec(
            obligation_id=item.obligation.obligation_id,
            clause_span={"text": f"Both ends of {item.identity} remain compatible"},
            interpretation=VALIDATED,
            targets=tuple(dict.fromkeys(
                endpoint.path for endpoint in (*item.producers, *item.consumers)
                if endpoint.path)),
            acceptance_policy=(item.check.check_id,),
            clause=f"Both ends of {item.identity} remain compatible",
            provenance={"producer": "kernel_other_end_contract"},
            contract_cells=(item.contract_cell,),
        )
        for item in report.bindings
    )
    return CheckBundle(
        task_id=task_id,
        specs=specs,
        checks=report.checks,
        path="kernel_other_end_contract",
        provenance={"schema": SCHEMA, "closure_event_id": report.closure_event_id},
        contract_cells=report.contract_cells,
    )


def sync_completion_authority(authority: object, report: ContractReport) -> object:
    """Project unresolved contract obligations into the existing host authority.

    The authority stores only its generic evidence-gap vocabulary; the report
    retains the more specific contract reason.  A later host-observed
    compatible binding supplies a current PASS for the same obligation id.
    """
    if mode() != "on" or authority is None or not report.fired:
        return authority
    required = getattr(authority, "required_evidence", frozenset())
    candidate_digest = getattr(authority, "candidate_digest", None)
    revision = getattr(authority, "current_revision", None)
    require_evidence = getattr(authority, "require_evidence", None)
    record_receipt = getattr(authority, "record_receipt", None)
    for obligation in report.obligations:
        obligation_id = obligation.obligation_id
        if obligation.status == "CLEARED":
            if obligation_id in required and callable(record_receipt):
                record_receipt({
                    "receipt_id": f"other-end-contract-pass:{obligation_id}:"
                    f"{revision or candidate_digest or 'current'}",
                    "obligation_id": obligation_id,
                    "status": "PASS",
                    "candidate_digest": candidate_digest,
                    "revision": revision,
                })
            continue
        if tier_mode() != "strict" and obligation.tier != "ADMITTED_VIOLATION":
            continue  # §62: check candidates / disputes are surfaced, never completion blockers
        if obligation_id not in required and callable(require_evidence):
            require_evidence(obligation_id, reason_code="REQUIRED_EVIDENCE_GAP")
    return authority


def dispute_contract(
    report: ContractReport,
    obligation_id: str,
    *,
    reason: str,
    host_accepted: bool = False,
    journal: object | None = None,
) -> ContractReport:
    """Record a host-accepted contract dispute without editing source files."""
    if not host_accepted:
        raise ValueError("only the host can accept a contract dispute")
    reason = str(reason).strip()
    if not reason:
        raise ValueError("contract dispute requires a reason")
    updated: list[ContractBinding] = []
    found = False
    for binding in report.bindings:
        if binding.obligation.obligation_id != str(obligation_id):
            updated.append(binding)
            continue
        found = True
        obligation = replace(
            binding.obligation,
            status="DISPUTED",
            reason_code=CONTRACT_DISPUTED,
            tier="DISPUTED",
            required_action="Resolve the host-accepted contract dispute before relying on this boundary",
            dispute=reason,
        )
        updated.append(replace(binding, obligation=obligation))
    if not found:
        raise KeyError(obligation_id)
    bindings = tuple(updated)
    event_id = _emit(
        journal,
        CONTRACT_DISPUTED,
        producer="kernel_other_end_contract",
        authority="HOST_ACCEPTED_DISPUTE",
        fired=True,
        obligation_id=str(obligation_id),
        reason=reason,
        dispatch=False,
    )
    return replace(
        report,
        bindings=bindings,
        orphans=tuple(item for item in bindings
                      if item.obligation.reason_code == CONTRACT_ORPHAN),
        unresolved=tuple(item for item in bindings
                         if item.obligation.status in {"OPEN", "DISPUTED"}
                         and item.obligation.reason_code != CONTRACT_ORPHAN),
        journaled=report.journaled or event_id is not None,
    )


# Friendly aliases for host integrations that name the organ after its brief.
analyze_contracts = bind_contracts
bind_other_end_contract = bind_contracts


__all__ = [
    "CONTRACT_DISPUTED", "CONTRACT_ORPHAN", "CONSUMER", "ContractBinding", "ContractObligation",
    "ContractReport", "Endpoint", "EVENT_KIND", "MODE_ENV", "ORPHAN_EVENT_KIND",
    "PRODUCER", "SCHEMA", "analyze_contracts", "bind_contracts",
    "bind_other_end_contract", "build_contract_bundle", "dispute_contract",
    "enabled", "mode", "sync_completion_authority",
]
