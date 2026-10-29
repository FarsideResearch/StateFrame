"""Host-owned S1 requirement spine and the S1-to-S7 identity bridge.

The requirements ledger remains the only durable semantic owner.  This module
validates the typed extension, creates host-scoped row references, and checks
stage artifacts against that extension.  It does not keep process state, mint
model evidence, execute checks, or provide a second completion owner.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence


SCHEMA = "requirement-spine/1"
TABLE_SCHEMA = "requirement-spine-table/1"
STAGE_RECORD_SCHEMA = "requirement-spine-stage-record/1"
TEMPLATE_SCHEMA = "repro-template/1"
STAGES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")
DOWNSTREAM_STAGES = STAGES[1:]
ROLES = frozenset({"REPAIR", "PRESERVE", "NEW_BEHAVIOR"})   # NEW_BEHAVIOR: Q3 / E9
OBLIGATIONS = frozenset({"REQUIRED", "ADVISORY"})
ADMISSION_STATES = frozenset({"PROPOSED", "NEEDS_INFORMATION", "ADMITTED", "REJECTED"})
OBSERVABLE_FAMILIES = frozenset({
    "RETURN", "EXCEPTION", "PERSISTED_STATE", "CALLS", "EVENT_ORDER",
    "CONFIG_EFFECT", "TIMING_BOUND", "ARTIFACT",
})
PREDICATE_RELATIONS = frozenset({
    "EQ", "NE", "LT", "LE", "GT", "GE", "CONTAINS", "ABSENT", "BEFORE",
    "UNCHANGED", "RAISES", "DOES_NOT_RAISE", "RELATION",
})
QUANTIFIERS = frozenset({"EACH_CASE", "EXISTS_CASE", "ORDERED_TRACE"})
ARTIFACT_HASH = re.compile(r"^[0-9a-f]{64}$")


class SpineError(ValueError):
    """Raised when the host cannot prove a spine or stage reference."""


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SpineError(f"{field} must be a string")
    result = value.strip()
    if not allow_empty and not result:
        raise SpineError(f"{field} must be non-empty")
    if "\x00" in result:
        raise SpineError(f"{field} contains NUL")
    return result


def _closed(value: object, fields: set[str], *, field: str) -> Mapping[str, object]:
    record = _mapping(value)
    if record is None or set(record) != fields:
        raise SpineError(f"{field} must be a closed record with fields {sorted(fields)}")
    return record


def _hash_value(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _artifact(value: object, *, field: str) -> dict[str, object]:
    record = _closed(value, {"id", "sha256"}, field=field)
    ident = _text(record["id"], field=f"{field}.id")
    digest = _text(record["sha256"], field=f"{field}.sha256")
    if not ARTIFACT_HASH.fullmatch(digest):
        raise SpineError(f"{field}.sha256 must be a SHA-256 hex digest")
    return {"id": ident, "sha256": digest}


def _req_ref(value: object, *, ledger_id: str, field: str) -> dict[str, object]:
    record = _closed(value, {"ledger_id", "id", "version"}, field=field)
    ref_ledger = _text(record["ledger_id"], field=f"{field}.ledger_id")
    ident = _text(record["id"], field=f"{field}.id")
    version = record["version"]
    if ref_ledger != ledger_id or type(version) is not int or version < 1:
        raise SpineError(f"{field} is not a reference to this ledger")
    return {"ledger_id": ref_ledger, "id": ident, "version": version}


def _value(value: object, *, field: str, depth: int = 0) -> object:
    if depth > 8:
        raise SpineError(f"{field} exceeds value nesting limit")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SpineError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, list):
        if len(value) > 64:
            raise SpineError(f"{field} exceeds array limit")
        return [_value(item, field=f"{field}[]", depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 64 or any(not isinstance(key, str) or not key for key in value):
            raise SpineError(f"{field} contains invalid object keys")
        return {
            key: _value(item, field=f"{field}.{key}", depth=depth + 1)
            for key, item in value.items()
        }
    raise SpineError(f"{field} is not finite JSON data")


def _support(value: object, *, span_ids: set[str], field: str) -> dict[str, object]:
    record = _closed(value, {"span_refs", "rule"}, field=field)
    refs = [_text(item, field=f"{field}.span_refs[]") for item in _items(record["span_refs"])]
    if not refs or len(refs) != len(set(refs)) or not set(refs).issubset(span_ids):
        raise SpineError(f"{field}.span_refs must name unique ledger spans")
    rule = record["rule"]
    if rule is not None:
        rule = _text(rule, field=f"{field}.rule")
    return {"span_refs": refs, "rule": rule}


def _term(value: object, *, span_ids: set[str], slot_ids: set[str], field: str) -> dict[str, object]:
    record = _mapping(value)
    if record is None or set(record) - {"kind", "value", "support", "slot_id", "question"}:
        raise SpineError(f"{field} is not a valid term")
    kind = _text(record.get("kind"), field=f"{field}.kind")
    if kind == "LITERAL":
        if set(record) != {"kind", "value", "support"}:
            raise SpineError(f"{field} literal shape is invalid")
        return {
            "kind": kind,
            "value": _value(record["value"], field=f"{field}.value"),
            "support": _support(record["support"], span_ids=span_ids, field=f"{field}.support"),
        }
    if kind == "SLOT":
        if set(record) != {"kind", "slot_id"}:
            raise SpineError(f"{field} slot shape is invalid")
        slot_id = _text(record["slot_id"], field=f"{field}.slot_id")
        if slot_id not in slot_ids:
            raise SpineError(f"{field}.slot_id is not declared by its trigger")
        return {"kind": kind, "slot_id": slot_id}
    if kind == "UNRESOLVED":
        if set(record) != {"kind", "question", "support"}:
            raise SpineError(f"{field} unresolved shape is invalid")
        return {
            "kind": kind,
            "question": _text(record["question"], field=f"{field}.question"),
            "support": _support(record["support"], span_ids=span_ids, field=f"{field}.support"),
        }
    raise SpineError(f"{field}.kind is not supported")


def _predicate(value: object, *, span_ids: set[str], slot_ids: set[str], field: str) -> dict[str, object]:
    record = _closed(value, {"id", "subject", "relation", "rhs", "support", "relation_text"}, field=field)
    ident = _text(record["id"], field=f"{field}.id")
    relation = _text(record["relation"], field=f"{field}.relation")
    if relation not in PREDICATE_RELATIONS:
        raise SpineError(f"{field}.relation is not supported")
    relation_text = record["relation_text"]
    if relation == "RELATION" and not isinstance(relation_text, str):
        raise SpineError(f"{field}.relation_text is required for RELATION")
    if relation != "RELATION" and relation_text is not None:
        raise SpineError(f"{field}.relation_text is only valid for RELATION")
    return {
        "id": ident,
        "subject": _text(record["subject"], field=f"{field}.subject"),
        "relation": relation,
        "rhs": _term(record["rhs"], span_ids=span_ids, slot_ids=slot_ids, field=f"{field}.rhs"),
        "support": _support(record["support"], span_ids=span_ids, field=f"{field}.support"),
        "relation_text": relation_text if relation_text is None else _text(relation_text, field=f"{field}.relation_text"),
    }


def _predicate_list(value: object, *, span_ids: set[str], slot_ids: set[str], field: str) -> list[dict[str, object]]:
    result = [
        _predicate(item, span_ids=span_ids, slot_ids=slot_ids, field=f"{field}[{index}]")
        for index, item in enumerate(_items(value))
    ]
    ids = [str(item["id"]) for item in result]
    if len(ids) != len(set(ids)):
        raise SpineError(f"{field} contains duplicate predicate IDs")
    return result


def _trigger(value: object, *, span_ids: set[str], field: str) -> dict[str, object]:
    record = _closed(value, {"action", "support", "conditions", "slots", "quantifier", "case_partition_refs", "exception_refs"}, field=field)
    slots: list[dict[str, object]] = []
    slot_ids: set[str] = set()
    for index, item in enumerate(_items(record["slots"])):
        slot = _closed(item, {"id", "role", "value_type", "domain", "support"}, field=f"{field}.slots[{index}]")
        ident = _text(slot["id"], field=f"{field}.slots[{index}].id")
        if ident in slot_ids:
            raise SpineError(f"{field}.slots contains duplicate IDs")
        slot_ids.add(ident)
        role = _text(slot["role"], field=f"{field}.slots[{index}].role")
        if role not in {"INPUT", "INITIAL_STATE", "CONFIG", "EVENT", "EXPECTED"}:
            raise SpineError(f"{field}.slots[{index}].role is not supported")
        value_type = _text(slot["value_type"], field=f"{field}.slots[{index}].value_type")
        if value_type not in {"BOOL", "NUMBER", "STRING", "ARRAY", "OBJECT", "NULL", "UNRESOLVED"}:
            raise SpineError(f"{field}.slots[{index}].value_type is not supported")
        slots.append({
            "id": ident,
            "role": role,
            "value_type": value_type,
            "domain": _predicate_list(slot["domain"], span_ids=span_ids, slot_ids=slot_ids, field=f"{field}.slots[{index}].domain"),
            "support": _support(slot["support"], span_ids=span_ids, field=f"{field}.slots[{index}].support"),
        })
    return {
        "action": _text(record["action"], field=f"{field}.action"),
        "support": _support(record["support"], span_ids=span_ids, field=f"{field}.support"),
        "conditions": _predicate_list(record["conditions"], span_ids=span_ids, slot_ids=slot_ids, field=f"{field}.conditions"),
        "slots": slots,
        "quantifier": _text(record["quantifier"], field=f"{field}.quantifier"),
        "case_partition_refs": [_text(item, field=f"{field}.case_partition_refs[]") for item in _items(record["case_partition_refs"])],
        "exception_refs": [_text(item, field=f"{field}.exception_refs[]") for item in _items(record["exception_refs"])],
    }


def _observable(value: object, *, span_ids: set[str], slot_ids: set[str], field: str) -> dict[str, object]:
    record = _closed(value, {"id", "family", "surface_hint", "surface_support", "predicates", "binding_status", "binding_receipt"}, field=field)
    family = _text(record["family"], field=f"{field}.family")
    if family not in OBSERVABLE_FAMILIES:
        raise SpineError(f"{field}.family is not supported")
    status = _text(record["binding_status"], field=f"{field}.binding_status")
    if status not in {"UNBOUND", "PUBLIC_BOUND"}:
        raise SpineError(f"{field}.binding_status is not supported")
    receipt = None if record["binding_receipt"] is None else _artifact(record["binding_receipt"], field=f"{field}.binding_receipt")
    if status == "PUBLIC_BOUND" and receipt is None:
        raise SpineError(f"{field}.binding_receipt is required for PUBLIC_BOUND")
    if status == "UNBOUND" and receipt is not None:
        raise SpineError(f"{field}.binding_receipt cannot bind an UNBOUND observable")
    predicates = _predicate_list(record["predicates"], span_ids=span_ids, slot_ids=slot_ids, field=f"{field}.predicates")
    if not predicates:
        raise SpineError(f"{field}.predicates must be non-empty")
    return {
        "id": _text(record["id"], field=f"{field}.id"),
        "family": family,
        "surface_hint": _text(record["surface_hint"], field=f"{field}.surface_hint"),
        "surface_support": _support(record["surface_support"], span_ids=span_ids, field=f"{field}.surface_support"),
        "predicates": predicates,
        "binding_status": status,
        "binding_receipt": receipt,
    }


def _wrong_behavior(value: object, *, span_ids: set[str], slot_ids: set[str], observation_ids: set[str], field: str) -> dict[str, object]:
    record = _closed(value, {"id", "description", "under_trigger", "extra_conditions", "violates_observation_ids", "rejection_predicate", "origin", "support"}, field=field)
    under_trigger = _text(record["under_trigger"], field=f"{field}.under_trigger")
    if under_trigger not in {"SAME", "REFINED"}:
        raise SpineError(f"{field}.under_trigger is not supported")
    origin = _text(record["origin"], field=f"{field}.origin")
    if origin not in {"REPORTED_SYMPTOM", "PUBLIC_COUNTEREXAMPLE", "HYPOTHETICAL_CONTROL"}:
        raise SpineError(f"{field}.origin is not supported")
    violations = [_text(item, field=f"{field}.violates_observation_ids[]") for item in _items(record["violates_observation_ids"])]
    if not violations or not set(violations).issubset(observation_ids):
        raise SpineError(f"{field}.violates_observation_ids must name observables")
    return {
        "id": _text(record["id"], field=f"{field}.id"),
        "description": _text(record["description"], field=f"{field}.description"),
        "under_trigger": under_trigger,
        "extra_conditions": _predicate_list(record["extra_conditions"], span_ids=span_ids, slot_ids=slot_ids, field=f"{field}.extra_conditions"),
        "violates_observation_ids": violations,
        "rejection_predicate": _predicate(record["rejection_predicate"], span_ids=span_ids, slot_ids=slot_ids, field=f"{field}.rejection_predicate"),
        "origin": origin,
        "support": _support(record["support"], span_ids=span_ids, field=f"{field}.support"),
    }


def _spine_row(value: object, *, span_ids: set[str], ledger_id: str, known_refs: set[tuple[str, int]], field: str) -> dict[str, object]:
    record = _closed(value, {"schema", "role", "obligation", "derivation", "reported_symptom", "trigger", "observables", "wrong_behaviors", "ambiguities", "joint_scenario_group", "depends_on", "lineage", "semantic_admission", "admission_receipt"}, field=field)
    schema = _text(record["schema"], field=f"{field}.schema")
    if schema != SCHEMA:
        raise SpineError(f"{field}.schema is unsupported")
    role = _text(record["role"], field=f"{field}.role")
    obligation = _text(record["obligation"], field=f"{field}.obligation")
    if role not in ROLES or obligation not in OBLIGATIONS:
        raise SpineError(f"{field} role/obligation is unsupported")
    derivation = _closed(record["derivation"], {"kind", "premise_rows", "premise_spans", "rule", "admission_receipt"}, field=f"{field}.derivation")
    derivation_kind = _text(derivation["kind"], field=f"{field}.derivation.kind")
    if derivation_kind not in {"EXPLICIT", "IMPLIED"}:
        raise SpineError(f"{field}.derivation.kind is unsupported")
    premise_spans = [_text(item, field=f"{field}.derivation.premise_spans[]") for item in _items(derivation["premise_spans"])]
    if not premise_spans or not set(premise_spans).issubset(span_ids):
        raise SpineError(f"{field}.derivation.premise_spans must reference source spans")
    premise_rows = [_req_ref(item, ledger_id=ledger_id, field=f"{field}.derivation.premise_rows[]") for item in _items(derivation["premise_rows"])]
    if not all((str(item["id"]), int(item["version"])) in known_refs for item in premise_rows):
        raise SpineError(f"{field}.derivation.premise_rows contains an unknown row")
    derivation_receipt = None if derivation["admission_receipt"] is None else _artifact(derivation["admission_receipt"], field=f"{field}.derivation.admission_receipt")
    reported = record["reported_symptom"]
    if reported is not None:
        reported_record = _closed(reported, {"text", "support"}, field=f"{field}.reported_symptom")
        reported = {"text": _text(reported_record["text"], field=f"{field}.reported_symptom.text"), "support": _support(reported_record["support"], span_ids=span_ids, field=f"{field}.reported_symptom.support")}
    trigger = _trigger(record["trigger"], span_ids=span_ids, field=f"{field}.trigger")
    slot_ids = {str(item["id"]) for item in _items(trigger["slots"]) if isinstance(item, Mapping)}
    observables = [_observable(item, span_ids=span_ids, slot_ids=slot_ids, field=f"{field}.observables[{index}]") for index, item in enumerate(_items(record["observables"]))]
    if not observables:
        raise SpineError(f"{field}.observables must be non-empty")
    observable_ids = [str(item["id"]) for item in observables]
    if len(observable_ids) != len(set(observable_ids)):
        raise SpineError(f"{field}.observables contains duplicate IDs")
    wrong = [_wrong_behavior(item, span_ids=span_ids, slot_ids=slot_ids, observation_ids=set(observable_ids), field=f"{field}.wrong_behaviors[{index}]") for index, item in enumerate(_items(record["wrong_behaviors"]))]
    if not wrong:
        raise SpineError(f"{field}.wrong_behaviors must be non-empty")
    ambiguities: list[dict[str, object]] = []
    for index, item in enumerate(_items(record["ambiguities"])):
        ambiguity = _closed(item, {"id", "question", "alternatives", "materiality", "status", "selected_alternative_id", "resolution_receipt"}, field=f"{field}.ambiguities[{index}]")
        status = _text(ambiguity["status"], field=f"{field}.ambiguities[{index}].status")
        if status not in {"OPEN", "RESOLVED"}:
            raise SpineError(f"{field}.ambiguities[{index}].status is unsupported")
        materiality = _text(ambiguity["materiality"], field=f"{field}.ambiguities[{index}].materiality")
        if materiality not in {"EXPECTATION", "SCOPE", "BINDING"}:
            raise SpineError(f"{field}.ambiguities[{index}].materiality is unsupported")
        alternatives: list[dict[str, object]] = []
        for alt_index, raw_alt in enumerate(_items(ambiguity["alternatives"])):
            alt = _closed(raw_alt, {"id", "interpretation", "support"}, field=f"{field}.ambiguities[{index}].alternatives[{alt_index}]")
            alternatives.append({"id": _text(alt["id"], field="alternative.id"), "interpretation": _text(alt["interpretation"], field="alternative.interpretation"), "support": _support(alt["support"], span_ids=span_ids, field="alternative.support")})
        selected = ambiguity["selected_alternative_id"]
        if selected is not None and _text(selected, field=f"{field}.ambiguities[{index}].selected_alternative_id") not in {str(item["id"]) for item in alternatives}:
            raise SpineError(f"{field}.ambiguities[{index}] selects an unknown alternative")
        receipt = None if ambiguity["resolution_receipt"] is None else _artifact(ambiguity["resolution_receipt"], field=f"{field}.ambiguities[{index}].resolution_receipt")
        if status == "RESOLVED" and (selected is None or receipt is None):
            raise SpineError(f"{field}.ambiguities[{index}] resolved without host evidence")
        ambiguities.append({"id": _text(ambiguity["id"], field=f"{field}.ambiguities[{index}].id"), "question": _text(ambiguity["question"], field=f"{field}.ambiguities[{index}].question"), "alternatives": alternatives, "materiality": materiality, "status": status, "selected_alternative_id": selected, "resolution_receipt": receipt})
    semantic = _text(record["semantic_admission"], field=f"{field}.semantic_admission")
    if semantic not in ADMISSION_STATES:
        raise SpineError(f"{field}.semantic_admission is unsupported")
    admission_receipt = None if record["admission_receipt"] is None else _artifact(record["admission_receipt"], field=f"{field}.admission_receipt")
    if semantic == "ADMITTED" and admission_receipt is None:
        raise SpineError(f"{field}.admission_receipt is required for ADMITTED")
    if semantic != "ADMITTED" and admission_receipt is not None:
        raise SpineError(f"{field}.admission_receipt requires ADMITTED")
    quantifier = str(trigger["quantifier"])
    if quantifier not in QUANTIFIERS:
        raise SpineError(f"{field}.trigger.quantifier is unsupported")
    depends_on = [_req_ref(item, ledger_id=ledger_id, field=f"{field}.depends_on[]") for item in _items(record["depends_on"])]
    if not all((str(item["id"]), int(item["version"])) in known_refs for item in depends_on):
        raise SpineError(f"{field}.depends_on contains an unknown row")
    lineage: list[dict[str, object]] = []
    for index, item in enumerate(_items(record["lineage"])):
        link = _closed(item, {"relation", "predecessor"}, field=f"{field}.lineage[{index}]")
        relation = _text(link["relation"], field=f"{field}.lineage[{index}].relation")
        if relation not in {"REFINES", "SPLIT_FROM", "MERGED_FROM"}:
            raise SpineError(f"{field}.lineage relation is unsupported")
        predecessor = _req_ref(link["predecessor"], ledger_id=ledger_id, field=f"{field}.lineage[{index}].predecessor")
        if (str(predecessor["id"]), int(predecessor["version"])) not in known_refs:
            raise SpineError(f"{field}.lineage predecessor is unknown")
        lineage.append({"relation": relation, "predecessor": predecessor})
    return {
        "schema": SCHEMA,
        "role": role,
        "obligation": obligation,
        "derivation": {"kind": derivation_kind, "premise_rows": premise_rows, "premise_spans": premise_spans, "rule": None if derivation["rule"] is None else _text(derivation["rule"], field=f"{field}.derivation.rule"), "admission_receipt": derivation_receipt},
        "reported_symptom": reported,
        "trigger": trigger,
        "observables": observables,
        "wrong_behaviors": wrong,
        "ambiguities": ambiguities,
        "joint_scenario_group": None if record["joint_scenario_group"] is None else _text(record["joint_scenario_group"], field=f"{field}.joint_scenario_group"),
        "depends_on": depends_on,
        "lineage": lineage,
        "semantic_admission": semantic,
        "admission_receipt": admission_receipt,
    }


def _row_payload(requirement: Mapping[str, object]) -> dict[str, object]:
    spine = requirement.get("spine")
    if not isinstance(spine, Mapping):
        raise SpineError(f"requirement {requirement.get('id')} is missing spine")
    return {
        "id": str(requirement.get("id")),
        "version": requirement.get("version"),
        "source_span_ids": list(requirement.get("source_span_ids", ())),
        "exact_clause": requirement.get("exact_clause"),
        "spine": copy.deepcopy(dict(spine)),
    }


def _semantic_rows(ledger: Mapping[str, object]) -> list[dict[str, object]]:
    return [_row_payload(row) for row in ledger.get("requirements", ()) if isinstance(row, Mapping) and isinstance(row.get("spine"), Mapping)]


def table_hash(ledger: Mapping[str, object]) -> str:
    """Hash semantic rows only; mutable assessments and stage receipts are excluded."""
    return _hash_value(_semantic_rows(ledger))


def requirement_ref(ledger: Mapping[str, object], requirement: Mapping[str, object]) -> dict[str, object]:
    ledger_id = _text(ledger.get("ticket_source_id"), field="ledger.ticket_source_id")
    ident = _text(requirement.get("id"), field="requirement.id")
    version = requirement.get("version")
    if type(version) is not int or version < 1:
        raise SpineError("requirement.version must be a positive integer")
    return {"ledger_id": ledger_id, "id": ident, "version": version}


def _header(ledger: Mapping[str, object], *, revision: int, active: Sequence[Mapping[str, object]], policy_ref: dict[str, object]) -> dict[str, object]:
    source_id = _text(ledger.get("ticket_source_id"), field="ledger.ticket_source_id")
    source = next((row for row in ledger.get("sources", ()) if isinstance(row, Mapping) and row.get("id") == source_id), None)
    if source is None:
        raise SpineError("ticket source is missing")
    refs = [requirement_ref(ledger, row) for row in active]
    return {
        "ledger_id": source_id,
        "table_revision": revision,
        "table_hash": table_hash(ledger),
        "ticket_source_id": source_id,
        "ticket_sha256": _text(source.get("sha256"), field="ticket_source.sha256"),
        "active_rows": refs,
        "policy_ref": policy_ref,
    }


def validate_spine_ledger(ledger: Mapping[str, object]) -> bool:
    """Validate the extension after the base ledger has been validated."""
    spine = _closed(ledger.get("spine"), {"schema", "header", "aliases", "admission_receipts", "stage_record_ids", "admission_requests"}, field="ledger.spine")
    if spine["schema"] != SCHEMA:
        raise SpineError("ledger.spine.schema is unsupported")
    ledger_id = _text(ledger.get("ticket_source_id"), field="ledger.ticket_source_id")
    requirements = [row for row in ledger.get("requirements", ()) if isinstance(row, Mapping)]
    by_ref = {(str(row.get("id")), row.get("version")): row for row in requirements}
    active = [row for row in requirements if isinstance(row.get("spine"), Mapping)]
    if not active:
        raise SpineError("spine must contain at least one active requirement")
    span_ids = {str(row.get("id")) for row in ledger.get("spans", ()) if isinstance(row, Mapping)}
    known_refs = {(str(row.get("id")), int(row.get("version"))) for row in requirements if type(row.get("version")) is int and row.get("version") >= 1}
    for index, row in enumerate(active):
        source_refs = {str(item) for item in _items(row.get("source_span_ids"))}
        if not source_refs or not source_refs.issubset(span_ids):
            raise SpineError(f"requirements[{index}] has invalid source spans")
        _spine_row(row["spine"], span_ids=span_ids, ledger_id=ledger_id, known_refs=known_refs, field=f"requirements[{index}].spine")
    header = _closed(spine["header"], {"ledger_id", "table_revision", "table_hash", "ticket_source_id", "ticket_sha256", "active_rows", "policy_ref"}, field="ledger.spine.header")
    if header["ledger_id"] != ledger_id or header["ticket_source_id"] != ledger_id:
        raise SpineError("spine header ledger identity mismatch")
    if type(header["table_revision"]) is not int or header["table_revision"] < 1:
        raise SpineError("spine table_revision must be positive")
    if header["table_hash"] != table_hash(ledger):
        raise SpineError("spine table hash is stale")
    source = next(row for row in ledger.get("sources", ()) if isinstance(row, Mapping) and row.get("id") == ledger_id)
    if header["ticket_sha256"] != source.get("sha256"):
        raise SpineError("spine ticket hash is stale")
    refs = [_req_ref(item, ledger_id=ledger_id, field="spine.header.active_rows[]") for item in _items(header["active_rows"])]
    if {(str(item["id"]), int(item["version"])) for item in refs} != set(by_ref) & {(str(row.get("id")), int(row.get("version"))) for row in active}:
        raise SpineError("spine active rows do not equal row projections")
    _artifact(header["policy_ref"], field="spine.header.policy_ref")
    for index, alias in enumerate(_items(spine["aliases"])):
        item = _closed(alias, {"alias", "scope", "ref"}, field=f"spine.aliases[{index}]")
        _text(item["alias"], field=f"spine.aliases[{index}].alias")
        if item["scope"] != header["table_hash"]:
            raise SpineError("spine alias scope is stale")
        ref = _req_ref(item["ref"], ledger_id=ledger_id, field=f"spine.aliases[{index}].ref")
        if (str(ref["id"]), int(ref["version"])) not in {(str(row.get("id")), int(row.get("version"))) for row in active}:
            raise SpineError("spine alias references an inactive row")
    for index, receipt in enumerate(_items(spine["admission_receipts"])):
        _artifact(receipt, field=f"spine.admission_receipts[{index}]")
    for index, record_id in enumerate(_items(spine["stage_record_ids"])):
        _text(record_id, field=f"spine.stage_record_ids[{index}]")
    for index, request in enumerate(_items(spine["admission_requests"])):
        _closed(request, {"request_id", "receipt"}, field=f"spine.admission_requests[{index}]")
        _text(request["request_id"], field=f"spine.admission_requests[{index}].request_id")
        _artifact(request["receipt"], field=f"spine.admission_requests[{index}].receipt")
    return True


def _proposal_spine(proposal: Mapping[str, object]) -> tuple[list[str], Mapping[str, object]]:
    source_value = proposal.get("source_span_ids", proposal.get("ticket_span_ids"))
    source_refs = [_text(item, field="proposal.source_span_ids[]") for item in _items(source_value)]
    if not source_refs:
        raise SpineError("proposal must provide source_span_ids")
    raw_spine = proposal.get("spine", proposal)
    if not isinstance(raw_spine, Mapping):
        raise SpineError("proposal.spine must be an object")
    return source_refs, raw_spine


def admit_spine_table(
        ledger: Mapping[str, object],
        proposals: Sequence[Mapping[str, object]] | Mapping[str, object],
        *,
        expected_table_revision: int | None = None,
        request_id: str = "host-s1-admission",
        semantic_review: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically project typed S1 proposals onto existing ledger rows.

    The model may propose temporary decomposition data, but canonical row IDs,
    version, aliases, table hash, and admission receipts are host-generated.
    """
    from kernel_requirements_ledger import validate_ledger

    validate_ledger(ledger)
    result = copy.deepcopy(dict(ledger))
    proposal_items = proposals.get("rows", ()) if isinstance(proposals, Mapping) else proposals
    normalized_proposals = [item for item in _items(proposal_items) if isinstance(item, Mapping)]
    if not normalized_proposals or len(normalized_proposals) != len(_items(proposal_items)):
        raise SpineError("proposals must be a non-empty sequence of objects")
    existing_spine = result.get("spine")
    if isinstance(existing_spine, Mapping):
        header = _mapping(existing_spine.get("header"))
        current_revision = header.get("table_revision") if header is not None else None
        if expected_table_revision is not None and current_revision != expected_table_revision:
            raise SpineError("stale S1 table revision")
        prior_requests = [item for item in _items(existing_spine.get("admission_requests")) if isinstance(item, Mapping)]
        if any(item.get("request_id") == request_id for item in prior_requests):
            validate_spine_ledger(result)
            return result
        if current_revision is not None:
            raise SpineError("semantic table changes require a new admission request and revision")
    elif expected_table_revision not in (None, 0):
        raise SpineError("expected S1 table revision does not match an unadmitted ledger")
    requirements = [row for row in result.get("requirements", ()) if isinstance(row, Mapping)]
    if not requirements:
        raise SpineError("S1 cannot admit a table with no ledger requirements")
    span_ids = {str(row.get("id")) for row in result.get("spans", ()) if isinstance(row, Mapping)}
    assigned: set[str] = set()
    active_rows: list[dict[str, object]] = []
    for index, proposal in enumerate(normalized_proposals):
        source_refs, raw_spine = _proposal_spine(proposal)
        if not set(source_refs).issubset(span_ids):
            raise SpineError(f"proposal[{index}] references an unknown source span")
        candidate = next((row for row in requirements if str(row.get("id")) not in assigned and set(str(item) for item in _items(row.get("source_span_ids"))) == set(source_refs)), None)
        if candidate is None:
            raise SpineError(f"proposal[{index}] does not map to an unclaimed ledger row")
        assigned.add(str(candidate["id"]))
        active_rows.append(candidate)
    if len(active_rows) != len(requirements):
        raise SpineError("S1 admission must account for every ledger requirement row")
    review = semantic_review if isinstance(semantic_review, Mapping) else {}
    review_decision = str(review.get("decision", "")).upper()
    review_id = str(review.get("receipt_id", "")).strip()
    if review_decision == "SUPPORTED" and not review_id:
        raise SpineError("SUPPORTED semantic review requires a host receipt_id")
    for index, (proposal, row) in enumerate(zip(normalized_proposals, active_rows)):
        source_refs, raw_spine = _proposal_spine(proposal)
        typed = dict(copy.deepcopy(dict(raw_spine)))
        typed["schema"] = SCHEMA
        semantic = "ADMITTED" if review_decision == "SUPPORTED" else "NEEDS_INFORMATION" if any(
            isinstance(item, Mapping) and str(item.get("status", "")).upper() == "OPEN"
            for item in _items(typed.get("ambiguities"))
        ) else "PROPOSED"
        typed["semantic_admission"] = semantic
        typed["admission_receipt"] = None
        if semantic == "ADMITTED":
            receipt_hash = _hash_value({"request_id": request_id, "row": row.get("id"), "source_refs": source_refs, "review": review_id})
            typed["admission_receipt"] = {"id": f"artifact:s1-admission:{receipt_hash[:20]}", "sha256": receipt_hash}
        row["spine"] = typed
    request_digest = _hash_value({"request_id": request_id, "rows": [str(row.get("id")) for row in active_rows], "review": review_id})
    next_revision = 1
    policy_ref = {"id": "policy:requirement-spine/1", "sha256": _hash_value({"schema": SCHEMA, "owner": "requirements-ledger"})}
    result["spine"] = {
        "schema": SCHEMA,
        "header": _header(result, revision=next_revision, active=active_rows, policy_ref=policy_ref),
        "aliases": [
            {"alias": f"req-{index:03d}", "scope": table_hash(result), "ref": requirement_ref(result, row)}
            for index, row in enumerate(active_rows)
        ],
        "admission_receipts": [
            {"id": f"artifact:s1-request:{request_digest[:20]}", "sha256": request_digest}
        ],
        "stage_record_ids": [],
        "admission_requests": [{"request_id": _text(request_id, field="request_id"), "receipt": {"id": f"artifact:s1-request:{request_digest[:20]}", "sha256": request_digest}}],
    }
    # The row extension is intentionally not written to the base schema's
    # assessment/receipt fields; those remain owned by existing host adapters.
    from kernel_requirements_ledger import validate_ledger
    validate_ledger(result)
    return result


def stage_record(
        ledger: Mapping[str, object],
        *,
        stage: str,
        row_refs: Sequence[Mapping[str, object]],
        payload: Mapping[str, object],
        stage_epoch: int,
        base_revision: str,
        candidate_revision: str | None = None,
        producer: str = "MODEL_PROPOSAL",
        admission: str = "PENDING",
        record_id: str | None = None,
) -> dict[str, object]:
    """Create one immutable stage envelope; payload remains an artifact body."""
    validate_spine_ledger(ledger)
    if stage not in DOWNSTREAM_STAGES:
        raise SpineError("stage records are only for S2-S7")
    if type(stage_epoch) is not int or stage_epoch < 1:
        raise SpineError("stage_epoch must be positive")
    if producer not in {"MODEL_PROPOSAL", "HOST"} or admission not in {"PENDING", "ADMITTED", "REJECTED"}:
        raise SpineError("invalid stage producer/admission")
    refs = [_req_ref(item, ledger_id=str(ledger["ticket_source_id"]), field="row_refs[]") for item in row_refs]
    if not refs or len({(str(item["id"]), int(item["version"])) for item in refs}) != len(refs):
        raise SpineError("stage row_refs must be non-empty and unique")
    active = {(str(item["id"]), int(item["version"])) for item in _items(ledger["spine"]["header"]["active_rows"]) if isinstance(item, Mapping)}
    if not {(str(item["id"]), int(item["version"])) for item in refs}.issubset(active):
        raise SpineError("stage row_refs contain an inactive requirement")
    payload_copy = _value(dict(payload), field="payload")
    payload_hash = _hash_value(payload_copy)
    table = ledger["spine"]["header"]
    ident = record_id or f"stage:{stage.lower()}:{payload_hash[:20]}"
    return {
        "schema": STAGE_RECORD_SCHEMA,
        "id": _text(ident, field="stage_record.id"),
        "stage": stage,
        "ledger_id": str(table["ledger_id"]),
        "table_revision": table["table_revision"],
        "table_hash": table["table_hash"],
        "row_refs": refs,
        "stage_epoch": stage_epoch,
        "base_revision": _text(base_revision, field="base_revision"),
        "candidate_revision": None if candidate_revision is None else _text(candidate_revision, field="candidate_revision"),
        "input_artifacts": [],
        "payload_ref": {"id": f"artifact:stage-payload:{payload_hash[:20]}", "sha256": payload_hash},
        "producer": producer,
        "admission": admission,
        "admission_receipt": None,
    }


def validate_stage_record(ledger: Mapping[str, object], record: Mapping[str, object]) -> bool:
    validate_spine_ledger(ledger)
    expected = {"schema", "id", "stage", "ledger_id", "table_revision", "table_hash", "row_refs", "stage_epoch", "base_revision", "candidate_revision", "input_artifacts", "payload_ref", "producer", "admission", "admission_receipt"}
    if set(record) != expected or record.get("schema") != STAGE_RECORD_SCHEMA:
        raise SpineError("stage record shape is not closed")
    stage = _text(record["stage"], field="stage")
    if stage not in DOWNSTREAM_STAGES:
        raise SpineError("stage record stage is unsupported")
    header = ledger["spine"]["header"]
    if record["ledger_id"] != header["ledger_id"] or record["table_revision"] != header["table_revision"] or record["table_hash"] != header["table_hash"]:
        raise SpineError("stage record points to a stale table")
    refs = [_req_ref(item, ledger_id=str(header["ledger_id"]), field="stage.row_refs[]") for item in _items(record["row_refs"])]
    if not refs or len(refs) != len({(str(item["id"]), int(item["version"])) for item in refs}):
        raise SpineError("stage row_refs must be unique and non-empty")
    active = {(str(item["id"]), int(item["version"])) for item in _items(header["active_rows"]) if isinstance(item, Mapping)}
    if not {(str(item["id"]), int(item["version"])) for item in refs}.issubset(active):
        raise SpineError("stage row_refs contain an inactive row")
    if type(record["stage_epoch"]) is not int or record["stage_epoch"] < 1:
        raise SpineError("stage_epoch must be positive")
    _text(record["id"], field="stage_record.id")
    _text(record["base_revision"], field="base_revision")
    if record["candidate_revision"] is not None:
        _text(record["candidate_revision"], field="candidate_revision")
    for index, artifact in enumerate(_items(record["input_artifacts"])):
        _artifact(artifact, field=f"stage.input_artifacts[{index}]")
    _artifact(record["payload_ref"], field="stage.payload_ref")
    producer = _text(record["producer"], field="producer")
    admission = _text(record["admission"], field="admission")
    if producer not in {"MODEL_PROPOSAL", "HOST"} or admission not in {"PENDING", "ADMITTED", "REJECTED"}:
        raise SpineError("stage producer/admission is unsupported")
    if record["admission_receipt"] is not None:
        _artifact(record["admission_receipt"], field="stage.admission_receipt")
    return True


def record_stage_consumption(ledger: Mapping[str, object], record: Mapping[str, object]) -> dict[str, object]:
    """Append a bounded stage index/evidence projection to the same ledger."""
    from kernel_requirements_ledger import validate_ledger

    validate_stage_record(ledger, record)
    result = copy.deepcopy(dict(ledger))
    spine = result["spine"]
    stage_ids = list(spine["stage_record_ids"])
    if record["id"] not in stage_ids:
        stage_ids.append(str(record["id"]))
    spine["stage_record_ids"] = stage_ids
    evidence = list(result.get("evidence", ()))
    if not any(isinstance(item, Mapping) and item.get("id") == record["id"] for item in evidence):
        evidence.append({
            "id": str(record["id"]),
            "kind": "inspection",
            "producer": "host",
            "revision": str(record["table_hash"]),
            "locator": None,
            "span_ids": [str(item["id"]) for item in record["row_refs"]],
            "receipt_id": (
                record["admission_receipt"].get("id")
                if isinstance(record.get("admission_receipt"), Mapping)
                else None
            ),
            "claim": f"{record['stage']} consumed canonical requirement row references",
            "scope": str(record["stage"]),
            "artifact_ref": str(record["payload_ref"]["id"]),
        })
    result["evidence"] = evidence
    validate_ledger(result)
    return result


def stage_consumption(ledger: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Prove every S2-S7 stage accounted for every active row exactly once."""
    validate_spine_ledger(ledger)
    active = {(str(item["id"]), int(item["version"])) for item in _items(ledger["spine"]["header"]["active_rows"]) if isinstance(item, Mapping)}
    by_stage: dict[str, set[tuple[str, int]]] = {stage: set() for stage in DOWNSTREAM_STAGES}
    record_ids: set[str] = set()
    for record in records:
        validate_stage_record(ledger, record)
        ident = str(record["id"])
        if ident in record_ids:
            raise SpineError("duplicate stage record identity")
        record_ids.add(ident)
        stage = str(record["stage"])
        refs = {(str(item["id"]), int(item["version"])) for item in _items(record["row_refs"]) if isinstance(item, Mapping)}
        if by_stage[stage] & refs:
            raise SpineError(f"stage {stage} consumed a row more than once")
        by_stage[stage].update(refs)
    stages = {
        stage: {
            "row_count": len(rows),
            "row_refs": [{"ledger_id": str(ledger["ticket_source_id"]), "id": ident, "version": version} for ident, version in sorted(rows)],
            "complete": rows == active,
            "missing": [{"id": ident, "version": version} for ident, version in sorted(active - rows)],
        }
        for stage, rows in by_stage.items()
    }
    return {
        "schema": "requirement-spine-consumption/1",
        "ledger_id": ledger["ticket_source_id"],
        "table_revision": ledger["spine"]["header"]["table_revision"],
        "table_hash": ledger["spine"]["header"]["table_hash"],
        "stages": stages,
        "complete": all(bool(item["complete"]) for item in stages.values()),
    }


def _bound_slot(value: object, *, field: str) -> str:
    record = _mapping(value)
    if record is None:
        raise SpineError(f"{field} must be a host binding record")
    status = str(record.get("status", "")).upper()
    if status in {"GAP", "UNBOUND"}:
        code = _text(record.get("code", "UNRESOLVED_BINDING"), field=f"{field}.code")
        raise SpineError(f"{code}:{field}")
    if status not in {"KNOWN", "BOUND", "PUBLIC_BOUND"}:
        raise SpineError(f"{field}.status is unsupported")
    value_text = _text(record.get("value"), field=f"{field}.value")
    evidence = [_text(item, field=f"{field}.evidence[]") for item in _items(record.get("evidence"))]
    if not evidence:
        raise SpineError(f"{field} has no public evidence")
    return value_text


def compile_repro_template(ledger: Mapping[str, object], row_ref: Mapping[str, object], binding: Mapping[str, object]) -> dict[str, object]:
    """Lower one admitted row to a recipe IR, never to model-authored code."""
    validate_spine_ledger(ledger)
    ref = _req_ref(row_ref, ledger_id=str(ledger["ticket_source_id"]), field="row_ref")
    target = next((row for row in ledger["requirements"] if isinstance(row, Mapping) and row.get("id") == ref["id"] and row.get("version") == ref["version"]), None)
    if target is None or not isinstance(target.get("spine"), Mapping):
        raise SpineError("row_ref is not an active spine row")
    spine = target["spine"]
    if spine.get("semantic_admission") != "ADMITTED":
        return {"schema": TEMPLATE_SCHEMA, "requirement_ref": ref, "status": "GAP", "gaps": [{"code": "SEMANTIC_NOT_ADMITTED", "next_action": "obtain host semantic admission"}]}
    required = ("entrypoint", "invocation", "recipe", "inputs", "observations", "scope_inventory", "control_recipe")
    missing = [field for field in required if field not in binding]
    if missing:
        return {"schema": TEMPLATE_SCHEMA, "requirement_ref": ref, "status": "GAP", "gaps": [{"code": "MISSING_BINDING", "fields": missing}]}
    try:
        entrypoint = _bound_slot(binding["entrypoint"], field="binding.entrypoint")
        invocation = _bound_slot(binding["invocation"], field="binding.invocation")
        recipe = _bound_slot(binding["recipe"], field="binding.recipe")
        control = _bound_slot(binding["control_recipe"], field="binding.control_recipe")
        scope = _bound_slot(binding["scope_inventory"], field="binding.scope_inventory")
    except SpineError as error:
        code, _, field = str(error).partition(":")
        return {"schema": TEMPLATE_SCHEMA, "requirement_ref": ref, "status": "GAP", "gaps": [{"code": code, "field": field}]}
    observations = [item for item in _items(binding["observations"]) if isinstance(item, Mapping)]
    expected_ids = {str(item["id"]) for item in _items(spine["observables"]) if isinstance(item, Mapping)}
    observed_ids = {str(item.get("observation_id", "")) for item in observations}
    if observed_ids != expected_ids or len(observations) != len(expected_ids):
        return {"schema": TEMPLATE_SCHEMA, "requirement_ref": ref, "status": "GAP", "gaps": [{"code": "OBSERVATION_BINDING_INCOMPLETE", "missing": sorted(expected_ids - observed_ids), "unexpected": sorted(observed_ids - expected_ids)}]}
    for index, observation in enumerate(observations):
        _text(observation.get("observation_id"), field=f"binding.observations[{index}].observation_id")
        _text(observation.get("path"), field=f"binding.observations[{index}].path")
        _text(observation.get("evidence"), field=f"binding.observations[{index}].evidence")
    inputs = _value(binding["inputs"], field="binding.inputs")
    digest = _hash_value({"ref": ref, "binding": binding})
    return {
        "schema": TEMPLATE_SCHEMA,
        "template_id": f"template:{digest[:20]}",
        "requirement_ref": ref,
        "status": "COMPILE_READY",
        "family": str(_items(spine["observables"])[0]["family"]),
        "trigger": copy.deepcopy(spine["trigger"]),
        "steps": [{"recipe_id": recipe, "entrypoint": entrypoint, "invocation": invocation, "inputs": inputs}],
        "observations": copy.deepcopy(observations),
        "predicates": copy.deepcopy([item for observable in _items(spine["observables"]) if isinstance(observable, Mapping) for item in _items(observable.get("predicates"))]),
        "control_recipe": control,
        "scope_inventory": scope,
        "digest": digest,
    }


def compile_repro_templates(ledger: Mapping[str, object], bindings: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    validate_spine_ledger(ledger)
    results: list[dict[str, object]] = []
    for row in ledger["requirements"]:
        if not isinstance(row, Mapping) or not isinstance(row.get("spine"), Mapping):
            continue
        ref = requirement_ref(ledger, row)
        result = compile_repro_template(ledger, ref, bindings.get(str(ref["id"]), {}))
        results.append(result)
    return {
        "schema": "repro-template-batch/1",
        "ledger_id": ledger["ticket_source_id"],
        "table_hash": ledger["spine"]["header"]["table_hash"],
        "templates": results,
        "ready_count": sum(item.get("status") == "COMPILE_READY" for item in results),
        "gap_count": sum(item.get("status") == "GAP" for item in results),
        "complete": bool(results) and all(item.get("status") == "COMPILE_READY" for item in results),
    }


def frame_lab_compile_check(tasks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Run the offline 11-task row-to-template check with no provider calls."""
    if len(tasks) != 11:
        raise SpineError("S1 request qualification compile check requires exactly 11 tasks")
    results: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        task_id = _text(task.get("task_id"), field=f"tasks[{index}].task_id")
        if task_id in seen:
            raise SpineError("request qualification task IDs must be unique")
        seen.add(task_id)
        ledger = task.get("ledger")
        bindings = task.get("bindings")
        if not isinstance(ledger, Mapping) or not isinstance(bindings, Mapping):
            raise SpineError(f"tasks[{index}] must provide ledger and bindings")
        batch = compile_repro_templates(ledger, bindings)
        results.append({"task_id": task_id, "ready_count": batch["ready_count"], "gap_count": batch["gap_count"], "complete": batch["complete"]})
    return {
        "schema": "frame-lab-s1-compile/1",
        "task_count": len(results),
        "tasks": results,
        "all_rows_compile": all(bool(item["complete"]) for item in results),
    }


def bind_idealpath_owner(owner: object, ledger: Mapping[str, object]) -> dict[str, object]:
    """Bind the existing IDEALPATH reducer to the ledger's read-only row set."""
    validate_spine_ledger(ledger)
    method = getattr(owner, "bind_requirement_table", None)
    if not callable(method):
        raise SpineError("IDEALPATH owner does not expose the spine projection seam")
    projection = {
        "schema": TABLE_SCHEMA,
        "ledger_id": ledger["spine"]["header"]["ledger_id"],
        "table_revision": ledger["spine"]["header"]["table_revision"],
        "table_hash": ledger["spine"]["header"]["table_hash"],
        "active_rows": copy.deepcopy(ledger["spine"]["header"]["active_rows"]),
    }
    method(projection)
    return projection


def consume_with_idealpath(owner: object, ledger: Mapping[str, object], record: Mapping[str, object]) -> dict[str, object]:
    validate_stage_record(ledger, record)
    method = getattr(owner, "consume_requirement_stage", None)
    if not callable(method):
        raise SpineError("IDEALPATH owner does not expose the stage consumption seam")
    return dict(method(record))


__all__ = [
    "ADMISSION_STATES", "DOWNSTREAM_STAGES", "SCHEMA", "SpineError", "STAGE_RECORD_SCHEMA",
    "TABLE_SCHEMA", "bind_idealpath_owner", "compile_repro_template", "compile_repro_templates",
    "frame_lab_compile_check", "admit_spine_table", "requirement_ref", "record_stage_consumption",
    "stage_consumption", "stage_record", "table_hash", "validate_spine_ledger", "validate_stage_record",
    "consume_with_idealpath",
]
