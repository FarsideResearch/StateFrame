"""Revision-bound host facts for the host facts StateFrame control block.

This module is intentionally a projection layer. It consumes records already
owned by the runtime and can only restate their typed result. It never
evaluates source, invents a check, or upgrades an observation into an
implementation claim.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_HOST_FACTS"
SCHEMA = "stateframe-host-facts/1"
STATE_SCHEMA = "stateframe-current-state/1"
EXPOSURE_SCHEMA = "stateframe-exposure-manifest/1"

CONTRACT = "CONTRACT"
OBSERVATION = "OBSERVATION"
ACTION_RESULT = "ACTION_RESULT"
DERIVED = "DERIVED"

CURRENT = "CURRENT"
STALE = "STALE"
CONFLICT = "CONFLICT"
_KINDS = frozenset({CONTRACT, OBSERVATION, ACTION_RESULT, DERIVED})
_STATUSES = frozenset({CURRENT, STALE, CONFLICT})

_EDIT_KINDS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_TEST_KINDS = frozenset({
    "test", "run_tests", "pytest", "verification", "test_receipt",
    "test_subprocess_completed", "test_subprocess_complete", "auto_verify_completed",
})


def host_facts_mode() -> str:
    """Return the opt-in mode, rejecting silent typo variants."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def mode() -> str:
    return host_facts_mode()


def enabled() -> bool:
    return host_facts_mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value: object) -> object:
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError):
        return str(value)


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (TypeError, ValueError, AttributeError):
            converted = None
        if isinstance(converted, Mapping):
            return copy.deepcopy(dict(converted))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            converted = dataclasses.asdict(value)
        except (TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return copy.deepcopy(dict(converted))
    return {}


def _rows(value: object, *, singleton_keys: Sequence[str] = ()) -> list[dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        keys = set(value)
        if keys.intersection(singleton_keys) or not value:
            return [copy.deepcopy(dict(value))] if value else []
        values = list(value.values())
        if values and all(isinstance(item, Mapping) or hasattr(item, "to_dict")
                          for item in values):
            return [row for item in values if (row := _mapping(item))]
        return [copy.deepcopy(dict(value))]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            row = _mapping(item)
            if row:
                result.append(row)
        return result
    row = _mapping(value)
    return [row] if row else []


def _first(value: object, *keys: str) -> object:
    if isinstance(value, Mapping):
        for key in keys:
            if value.get(key) is not None:
                return value[key]
        return None
    for key in keys:
        found = getattr(value, key, None)
        if found is not None:
            return found
    return None


def _text(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def current_revision(value: object, *, fallback: object = None) -> str | None:
    """Find the host's current product revision without using model prose."""
    if value is not None and not isinstance(value, Mapping):
        return str(value)
    if isinstance(value, Mapping):
        for key in ("current_revision", "revision", "product_revision", "binding_revision", "product"):
            candidate = value.get(key)
            if candidate is not None and not isinstance(candidate, Mapping):
                return str(candidate)
        for key in ("binding", "current_binding", "after_binding", "revision_binding_state", "world"):
            candidate = current_revision(value.get(key))
            if candidate is not None:
                return candidate
        revisions = value.get("artifact_revisions")
        if isinstance(revisions, Mapping) and len(revisions) == 1:
            only = next(iter(revisions.values()))
            if only is not None:
                return str(only)
    if fallback is None:
        return None
    return str(fallback)


def _row_revision(row: Mapping[str, object]) -> str | None:
    for key in ("current_revision", "revision", "product_revision", "artifact_revision",
                "source_revision", "after_revision", "candidate_revision"):
        value = row.get(key)
        if value is not None and not isinstance(value, Mapping):
            return str(value)
    for key in ("snapshot_digest", "snapshot"):
        value = row.get(key)
        if value is not None and not isinstance(value, Mapping):
            return str(value)
    for key in ("binding", "after_binding", "current_binding", "revision_binding_state"):
        binding = row.get(key)
        if isinstance(binding, Mapping):
            found = current_revision(binding)
            if found is not None:
                return found
    receipts = row.get("receipts")
    nested = _rows(receipts, singleton_keys=("check_id", "status", "result", "snapshot_digest"))
    for receipt in reversed(nested):
        found = _row_revision(receipt)
        if found is not None:
            return found
    return None


def _bound_status(row: Mapping[str, object], revision: str | None, *, implicit: bool = False) -> str:
    observed = _row_revision(row)
    if observed is None and implicit and revision is not None:
        observed = revision
    if observed is None or revision is None:
        return STALE
    return CURRENT if str(observed) == str(revision) else STALE


def _kind_name(row: Mapping[str, object]) -> str:
    return str(row.get("kind", row.get("tool", row.get("action", ""))) or "").lower()


def _evidence_ids(row: Mapping[str, object], fallback: object) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("receipt_id", "evidence_ref", "event_id", "record_id", "instance_key"):
        value = row.get(key)
        if value:
            values.append(str(value))
    if not values:
        values.append("record:" + digest(fallback))
    return tuple(dict.fromkeys(values))


def _subject(row: Mapping[str, object], *keys: str, default: str = "workspace") -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value = ",".join(str(item) for item in value)
        if value is not None and str(value).strip():
            return _text(value, 180)
    changed = row.get("changed_files")
    if isinstance(changed, Sequence) and not isinstance(changed, (str, bytes, bytearray)):
        paths = [str(item.get("path")) for item in changed if isinstance(item, Mapping) and item.get("path")]
        if paths:
            return ",".join(dict.fromkeys(paths))
    return default


def _safe_value(value: object) -> object:
    """Keep facts JSON-shaped while avoiding unbounded narrative receipts."""
    try:
        encoded = _canonical(value)
    except (TypeError, ValueError):
        return _text(value)
    if len(encoded) <= 8_000:
        return _copy(value)
    return {"digest": digest(value), "truncated": True}


@dataclass(frozen=True, slots=True)
class HostFact:
    kind: str
    predicate: str
    subject: str
    value: object
    scope: object = None
    evidence_ids: tuple[str, ...] = ()
    dependency_fingerprint: str = ""
    status: str = CURRENT
    revision: str | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).upper()
        status = str(self.status).upper()
        if kind not in _KINDS:
            raise ValueError(f"unknown host fact kind: {kind}")
        if status not in _STATUSES:
            raise ValueError(f"unknown host fact status: {status}")
        if not str(self.predicate).strip() or not str(self.subject).strip():
            raise ValueError("host facts require predicate and subject")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "predicate", str(self.predicate))
        object.__setattr__(self, "subject", str(self.subject))
        object.__setattr__(self, "value", _safe_value(self.value))
        object.__setattr__(self, "scope", _safe_value(self.scope))
        object.__setattr__(self, "evidence_ids", tuple(dict.fromkeys(
            str(item) for item in self.evidence_ids if str(item).strip())))
        object.__setattr__(self, "dependency_fingerprint",
                           str(self.dependency_fingerprint or digest({
                               "kind": kind, "predicate": self.predicate,
                               "subject": self.subject, "value": self.value,
                               "scope": self.scope, "revision": self.revision,
                           })))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "revision", None if self.revision is None else str(self.revision))

    @property
    def fact_id(self) -> str:
        return digest({"kind": self.kind, "predicate": self.predicate,
                       "subject": self.subject, "value": self.value,
                       "scope": self.scope, "dependency_fingerprint": self.dependency_fingerprint})

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "HostFact":
        return cls(
            str(value.get("kind", DERIVED)), str(value.get("predicate", "fact")),
            str(value.get("subject", "workspace")), value.get("value"),
            value.get("scope"), tuple(value.get("evidence_ids", ())),
            str(value.get("dependency_fingerprint", "")),
            str(value.get("status", CURRENT)), value.get("revision"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "fact_id": self.fact_id,
            "kind": self.kind,
            "predicate": self.predicate,
            "subject": self.subject,
            "value": _copy(self.value),
            "scope": _copy(self.scope),
            "evidence_ids": list(self.evidence_ids),
            "dependency_fingerprint": self.dependency_fingerprint,
            "status": self.status,
            "revision": self.revision,
        }


def _make_fact(kind: str, predicate: str, subject: str, value: object, row: Mapping[str, object],
               revision: str | None, *, implicit_revision: bool = False,
               scope: object = None, status: str | None = None) -> HostFact:
    row_revision = _row_revision(row)
    bound = status or _bound_status(row, revision, implicit=implicit_revision)
    dependency = row.get("dependency_fingerprint")
    if not dependency:
        dependency = digest({
            "revision": revision, "observed_revision": row_revision,
            "kind": kind, "predicate": predicate, "subject": subject,
            "value": value, "scope": scope,
        })
    return HostFact(
        kind=kind, predicate=predicate, subject=subject, value=value,
        scope=scope if scope is not None else {"revision": row_revision or revision},
        evidence_ids=_evidence_ids(row, {"kind": kind, "subject": subject, "value": value}),
        dependency_fingerprint=str(dependency), status=bound,
        revision=row_revision or revision,
    )


def _collect(frame: Mapping[str, object], explicit: object, keys: Sequence[str],
             *, singleton_keys: Sequence[str] = ()) -> list[dict[str, object]]:
    if explicit is not None:
        return _rows(explicit, singleton_keys=singleton_keys)
    result: list[dict[str, object]] = []
    for key in keys:
        result.extend(_rows(frame.get(key), singleton_keys=singleton_keys))
    return result


def _looks_like_contract(row: Mapping[str, object]) -> bool:
    return bool({"obligation_id", "clause", "acceptance_policy", "targets", "contract_cells"}
                .intersection(row)) and not ({"verdict", "check_ids", "receipts"}.intersection(row))


def _looks_like_assessment(row: Mapping[str, object]) -> bool:
    return bool({"verdict", "check_ids", "assessment", "obligation_id"}.intersection(row)) and (
        "verdict" in row or "check_ids" in row or "assessment" in row)


def _edit_rows(frame: Mapping[str, object], explicit: object) -> list[dict[str, object]]:
    result = _collect(frame, explicit, ("edit_receipts", "action_receipts", "recent_changes",
                                        "latest_receipts"),
                      singleton_keys=("kind", "tool", "result_class", "target_path"))
    if explicit is None:
        result.extend(row for row in _rows(frame.get("receipts"), singleton_keys=("kind", "tool"))
                     if _kind_name(row) in _EDIT_KINDS)
        last = _mapping(frame.get("last_action"))
        if last and _kind_name(last) in _EDIT_KINDS:
            result.append(last)
    return [row for row in result if _kind_name(row) in _EDIT_KINDS
            or "result_class" in row and row.get("target_path")]


def _test_rows(frame: Mapping[str, object], explicit: object) -> list[dict[str, object]]:
    result = _collect(frame, explicit, ("test_receipts", "test_results", "recent_test_feedback",
                                        "probe_receipts", "latest_receipts"),
                      singleton_keys=("kind", "tool", "result", "result_class", "selectors"))
    if explicit is None:
        result.extend(row for row in _rows(frame.get("receipts"), singleton_keys=("kind", "tool"))
                     if _kind_name(row) in _TEST_KINDS)
        last = _mapping(frame.get("last_action"))
        if last and (_kind_name(last) in _TEST_KINDS or
                     any(key in last for key in ("selectors", "exit_code", "passed"))):
            result.append(last)
    return [row for row in result if (_kind_name(row) in _TEST_KINDS
                                      or any(key in row for key in ("selectors", "exit_code", "passed")))]


def _other_action_rows(frame: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in ("last_action", "action_result"):
        row = _mapping(frame.get(key))
        if not row:
            continue
        kind = _kind_name(row)
        if kind in _EDIT_KINDS or kind in _TEST_KINDS:
            continue
        if kind or any(name in row for name in ("result", "status", "observation")):
            rows.append(row)
    return rows


def derive_host_facts(
    frame: Mapping[str, object] | None = None,
    *,
    records: Mapping[str, object] | None = None,
    current_revision: object = None,
    revision: object = None,
    edit_receipts: object = None,
    test_receipts: object = None,
    host_check_receipts: object = None,
    obligation_assessments: object = None,
    dependency_facts: object = None,
    contract_specs: object = None,
    signature_bindings: object = None,
) -> tuple[HostFact, ...]:
    """Project only existing host-owned records into revision-bound facts."""
    source = dict(frame or {})
    if records:
        source.update(dict(records))
    revision = (str(current_revision if current_revision is not None else revision)
                if (current_revision is not None or revision is not None) else
                globals()["current_revision"](source))
    facts: list[HostFact] = []

    for row in _edit_rows(source, edit_receipts):
        subject = _subject(row, "target_path", "path", "target", default="workspace")
        value = {
            "result": str(row.get("result_class", row.get("result", row.get("status", "UNKNOWN")))).upper(),
            "tool": _kind_name(row),
            "target": subject,
            "turn": row.get("turn"),
        }
        facts.append(_make_fact(ACTION_RESULT, "edit_result", subject, value, row, revision))

    for row in _other_action_rows(source):
        subject = _subject(row, "target_path", "path", "target", "tool", default="workspace")
        value = {
            "tool": _kind_name(row),
            "result": str(row.get("result_class", row.get("result", row.get("status", "UNKNOWN")))).upper(),
            "target": subject,
            "turn": row.get("turn"),
        }
        facts.append(_make_fact(ACTION_RESULT, "action_result", subject, value, row, revision))

    for row in _test_rows(source, test_receipts):
        subject = _subject(row, "target", "suite", "selector", "test", "selectors")
        selectors = row.get("selectors", row.get("selector"))
        if isinstance(selectors, str):
            selectors = [selectors]
        elif not isinstance(selectors, Sequence):
            selectors = []
        value = {
            "result": str(row.get("result", row.get("result_class", row.get("status", "UNKNOWN")))).upper(),
            "selectors": [str(item) for item in selectors],
            "exit_code": row.get("exit_code", row.get("exit")),
            "target": subject,
            "turn": row.get("turn"),
        }
        facts.append(_make_fact(OBSERVATION, "test_result", subject, value, row, revision))

    dependency_rows = _collect(
        source, dependency_facts,
        ("dependency_facts", "dependency_observations", "dependency_fact_receipts"),
        singleton_keys=("package", "version", "fact_key", "observation", "target"),
    )
    for row in dependency_rows:
        fact_key = row.get("fact_key")
        package = row.get("package")
        version = row.get("version")
        if (isinstance(fact_key, Sequence) and not isinstance(fact_key, (str, bytes, bytearray))
                and len(fact_key) >= 2):
            package = package or fact_key[0]
            version = version or fact_key[1]
        package = _text(package or "UNKNOWN", 120)
        version = _text(version or "UNKNOWN", 120)
        target = _subject(row, "target", default="dependency")
        subject = f"{package}@{version}:{target}"
        value = {
            "record_type": "observation",
            "fact_key": [package, version],
            "module": _text(row.get("module", ""), 180),
            "target": target,
            "operation": _text(row.get("operation", "attribute"), 40),
            "status": str(row.get("status", row.get("result_class", "UNKNOWN"))).upper(),
            "observation": _safe_value(row.get("observation", row.get("shape"))),
            "authority": str(row.get("authority", "HOST_OBSERVED")),
            "observer_complete": bool(row.get("observer_complete", True)),
            "normative": False,
        }
        facts.append(_make_fact(
            OBSERVATION, "dependency_behavior", subject, value, row, revision,
            implicit_revision=True,
            scope={"fact_key": [package, version], "target": target},
        ))

    check_rows = _collect(source, host_check_receipts,
                          ("host_check_receipts", "host_repair_check_receipts", "host_checks"),
                          singleton_keys=("check_id", "status", "result", "receipt_id"))
    host_closure = source.get("host_closure")
    if host_check_receipts is None and isinstance(host_closure, Mapping):
        check_rows.extend(_rows(host_closure.get("receipts"),
                                singleton_keys=("check_id", "status", "result")))
    for row in check_rows:
        subject = _subject(row, "execution_id", "check_id", "check", "id")
        value = {
            "status": str(row.get("status", row.get("result", row.get("result_class", "UNKNOWN")))).upper(),
            "execution": str(row.get("execution", "UNKNOWN")).upper(),
            "observed": _safe_value(row.get("observed", row.get("observed_values"))),
            "expected": _safe_value(row.get("expected")),
            "counterexample": _safe_value(row.get("counterexample")),
        }
        canonical_scope = row.get("selector_scope")
        canonical_check = row.get("execution_id") not in (None, "")
        scope = None
        if canonical_check:
            value.update({
                "execution_id": str(row.get("execution_id")),
                "receipt_id": str(row.get("receipt_id", "")),
                "selector": str(row.get("selector", "")),
                "selector_scope": _safe_value(canonical_scope or ()),
                "outcome": str(row.get("outcome", "UNKNOWN")).upper(),
                "failure_phase": str(row.get("failure_phase", "UNKNOWN")).upper(),
                "reached_observation": row.get("reached_observation"),
                "execution_complete": row.get("execution_complete") is True,
                "termination_reason": str(row.get("termination_reason", "UNKNOWN")).upper(),
                "counts": _safe_value(row.get("counts", {})),
                "wrapper_parse_error": _safe_value(row.get("wrapper_parse_error", "")),
            })
            scope = {"selector_scope": _safe_value(canonical_scope or ())}
        facts.append(_make_fact(
            OBSERVATION, "host_check_result", subject, value, row, revision,
            scope=scope,
        ))

    assessment_rows = _collect(source, obligation_assessments,
                               ("obligation_assessments", "assessments"),
                               singleton_keys=("obligation_id", "verdict", "reason"))
    if obligation_assessments is None and isinstance(host_closure, Mapping):
        assessment_rows.extend(_rows(host_closure.get("assessments"),
                                      singleton_keys=("obligation_id", "verdict", "status")))
    for row in assessment_rows:
        if not _looks_like_assessment(row):
            continue
        subject = _subject(row, "obligation_id", "id", default="obligation")
        value = {
            "verdict": str(row.get("verdict", row.get("status", "UNKNOWN"))).upper(),
            "reason": str(row.get("reason", row.get("reason_code", "UNKNOWN"))),
            "check_ids": [str(item) for item in row.get("check_ids", ())]
                           if isinstance(row.get("check_ids", ()), Sequence)
                           and not isinstance(row.get("check_ids", ()), (str, bytes)) else [],
        }
        facts.append(_make_fact(DERIVED, "obligation_assessment", subject, value, row, revision,
                                implicit_revision=True))

    contract_rows = _collect(source, contract_specs,
                             ("contract_specs", "obligation_specs", "contracts"),
                             singleton_keys=("obligation_id", "id", "clause", "acceptance_policy"))
    if contract_specs is None:
        contract_rows.extend(row for row in _rows(source.get("obligations"),
                                                  singleton_keys=("obligation_id", "clause"))
                             if _looks_like_contract(row))
        completion_contract = _mapping(source.get("completion_contract"))
        if completion_contract:
            contract_rows.append({
                "obligation_id": "completion_contract",
                "clause": completion_contract.get("clause", ""),
                "acceptance_policy": completion_contract.get(
                    "required_public_checks", completion_contract.get("checks", ())),
                "targets": completion_contract.get("targets", ()),
                "interpretation": "HOST_CONTRACT",
                "revision": revision,
            })
    for row in contract_rows:
        if not _looks_like_contract(row):
            continue
        subject = _subject(row, "obligation_id", "id", default="contract")
        value = {
            "clause": _text(row.get("clause", row.get("clause_span", "")), 1_000),
            "targets": [str(item) for item in row.get("targets", row.get("mentioned_files", ())) ]
                       if isinstance(row.get("targets", row.get("mentioned_files", ())), Sequence)
                       and not isinstance(row.get("targets", row.get("mentioned_files", ())), (str, bytes)) else [],
            "acceptance_policy": [str(item) for item in row.get(
                "acceptance_policy", row.get("checks", ())) ]
                if isinstance(row.get("acceptance_policy", row.get("checks", ())), Sequence)
                and not isinstance(row.get("acceptance_policy", row.get("checks", ())), (str, bytes)) else [],
            "interpretation": str(row.get("interpretation", "HOST_CONTRACT")),
        }
        contract_row = {**row, "revision": row.get("revision", revision)}
        facts.append(_make_fact(CONTRACT, "obligation_contract", subject, value,
                                contract_row, revision, implicit_revision=True))

    binding_rows = _collect(source, signature_bindings,
                            ("signature_bindings", "signature_binding", "bindings", "binding"),
                            singleton_keys=("product_revision", "artifact", "signature", "binding"))
    for row in binding_rows:
        if "signature" not in row and "product_revision" not in row and "artifact" not in row:
            continue
        subject = _subject(row, "artifact", "path", "subject", default="workspace")
        value = {key: _safe_value(row[key]) for key in (
            "signature", "product_revision", "artifact", "artifact_revision", "product_files")
            if key in row}
        facts.append(_make_fact(OBSERVATION, "signature_binding", subject, value, row, revision,
                                implicit_revision=True))

    # Merge byte-identical duplicate records while retaining every evidence id.
    merged: dict[str, HostFact] = {}
    for fact in facts:
        key = digest({"kind": fact.kind, "predicate": fact.predicate,
                      "subject": fact.subject, "value": fact.value,
                      "scope": fact.scope, "revision": fact.revision})
        prior = merged.get(key)
        if prior is None:
            merged[key] = fact
        else:
            merged[key] = dataclasses.replace(
                prior, evidence_ids=tuple(dict.fromkeys((*prior.evidence_ids, *fact.evidence_ids))))
    facts = list(merged.values())

    # A same-revision disagreement is a typed conflict; latest-wins is not a
    # valid projection policy.  Scope/revision are deliberately not part of
    # the grouping key so the detector sees the entire current subject.
    groups: dict[tuple[str, str, str], list[HostFact]] = {}
    for fact in facts:
        groups.setdefault((fact.kind, fact.predicate, fact.subject), []).append(fact)
    conflicted: set[str] = set()
    for group in groups.values():
        current_values = {json.dumps(fact.value, sort_keys=True, default=str)
                          for fact in group if fact.status == CURRENT}
        if len(current_values) > 1:
            conflicted.update(fact.fact_id for fact in group if fact.status == CURRENT)
    return tuple(sorted(
        (dataclasses.replace(fact, status=CONFLICT) if fact.fact_id in conflicted else fact
         for fact in facts),
        key=lambda fact: (fact.kind, fact.predicate, fact.subject, fact.fact_id)))


def _coerce_facts(value: object) -> tuple[HostFact, ...]:
    if isinstance(value, Mapping):
        value = value.values()
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        return ()
    result = []
    for item in value:
        if isinstance(item, HostFact):
            result.append(item)
        elif isinstance(item, Mapping):
            try:
                result.append(HostFact.from_mapping(item))
            except (TypeError, ValueError):
                continue
    return tuple(result)


def _decision_mapping(decision: object) -> dict[str, object]:
    if decision is None:
        return {}
    row = _mapping(decision)
    return row


def _submission(decision: object, open_work: Sequence[HostFact]) -> dict[str, object]:
    """Return the reducer's typed submission without reproducing advice prose."""
    row = _decision_mapping(decision)
    if not row:
        return {"state": "PENDING", "outcome": "PENDING",
                "authority": "completion_reducer", "blocking": []}
    # The blocker lane is the authority when it is present in a merged runtime.
    try:
        from kernel_completion_status import completion_status_from_decision
    except (ImportError, ModuleNotFoundError):
        status_value = None
    else:
        try:
            status_value = completion_status_from_decision(decision)
        except (TypeError, ValueError, AttributeError):
            status_value = None
    if status_value is not None:
        status = _mapping(status_value)
        if status:
            return {**status, "authority": "completion_reducer"}

    outcome = str(row.get("outcome", row.get("status", "PENDING"))).upper()
    if row.get("accepted") is True or outcome in {"ACCEPTED", "SEALED"}:
        state = "ACCEPTABLE"
    elif outcome in {"NEEDS_WORK", "REFUSED"}:
        state = "NEEDS_WORK"
    elif outcome in {"BLOCKED", "BASIS_CHANGED", "BOUNDED"}:
        state = "BLOCKED"
    else:
        state = "PENDING"
    blockers: list[object] = []
    payload = row.get("needs_work")
    if isinstance(payload, Mapping):
        raw = payload.get("blocking", payload.get("obligations", ()))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            blockers.extend(_copy(raw))
    if not blockers:
        blockers.extend({"obligation_id": fact.subject, "status": fact.status}
                        for fact in open_work if fact.predicate == "obligation_assessment")
    if not blockers:
        ids = row.get("needs_work_ids", ())
        if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
            blockers.extend({"obligation_id": str(ident)} for ident in ids)
    result = {
        "state": state, "outcome": outcome, "authority": "completion_reducer",
        "blocking": blockers, "reason_codes": list(row.get("reason_codes", ()))
        if isinstance(row.get("reason_codes", ()), Sequence)
        and not isinstance(row.get("reason_codes", ()), (str, bytes)) else [],
    }
    if isinstance(row.get("repair_action"), Mapping):
        result["next_feasible_action"] = _copy(row["repair_action"])
    return result


def permission_projection(tools: object = (), edit_authority: object = None,
                          *, edit_contract: object = None,
                          capability_data: object = None) -> dict[str, object]:
    """Project the exact menu and EditAuthority consumed by the executor."""
    if edit_authority is None and isinstance(edit_contract, Mapping):
        edit_authority = edit_contract.get("edit_authority", edit_contract)
    menu: list[object]
    if isinstance(tools, Mapping):
        menu = list(tools.values())
    elif isinstance(tools, Iterable) and not isinstance(tools, (str, bytes, bytearray)):
        menu = list(tools)
    else:
        menu = []
    names: list[str] = []
    for item in menu:
        row = _mapping(item)
        function = row.get("function") if isinstance(row.get("function"), Mapping) else row
        name = function.get("name") if isinstance(function, Mapping) else None
        if name and str(name) not in names:
            names.append(str(name))
    authority = _copy(edit_authority) if edit_authority is not None else None
    payload = {"tool_menu": menu, "edit_authority": authority,
               "capability_data": _copy(capability_data)}
    return {
        "authority": "executor_tool_menu+EditAuthority",
        "tool_names": names,
        "edit_authority": authority,
        "digest": digest(payload),
    }


@dataclass(frozen=True, slots=True)
class CanonicalControlState(Mapping[str, object]):
    revision: str
    last_action: tuple[dict[str, object], ...] = ()
    current_results: tuple[dict[str, object], ...] = ()
    contract: tuple[dict[str, object], ...] = ()
    open_work: tuple[dict[str, object], ...] = ()
    submission: dict[str, object] = dataclasses.field(default_factory=dict)
    permissions: dict[str, object] = dataclasses.field(default_factory=dict)
    fact_ids: tuple[str, ...] = ()
    terminal_status: dict[str, object] | None = None

    @property
    def schema(self) -> str:
        return STATE_SCHEMA

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema": STATE_SCHEMA,
            "revision": self.revision,
            "last_action": [_copy(item) for item in self.last_action],
            "current_results": [_copy(item) for item in self.current_results],
            "contract": [_copy(item) for item in self.contract],
            "open_work": [_copy(item) for item in self.open_work],
            "submission": _copy(self.submission),
            "permissions": _copy(self.permissions),
            "fact_ids": list(self.fact_ids),
        }
        if self.terminal_status is not None:
            result["terminal_status"] = _copy(self.terminal_status)
        return result

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


ControlState = CanonicalControlState


def build_current_state(
    frame: Mapping[str, object] | None = None,
    *,
    facts: object = None,
    revision: object = None,
    current_revision: object = None,
    tools: object = (),
    edit_authority: object = None,
    edit_contract: object = None,
    permission_data: Mapping[str, object] | None = None,
    completion_decision: object = None,
) -> CanonicalControlState:
    source = dict(frame or {})
    current = str(revision if revision is not None else current_revision if current_revision is not None else
                  globals()["current_revision"](source, fallback="UNKNOWN") or "UNKNOWN")
    host_facts = _coerce_facts(facts if facts is not None else source.get("host_facts"))
    if not host_facts:
        host_facts = derive_host_facts(source, current_revision=current)
    ordered = sorted(host_facts, key=lambda fact: (fact.kind, fact.predicate,
                                                   fact.subject, fact.fact_id))
    action_facts = [fact for fact in ordered
                    if fact.kind == ACTION_RESULT and fact.status in {CURRENT, CONFLICT}]
    if not action_facts:
        # A test/check receipt is an observation, but it is still the latest
        # host action when no edit action was recorded.
        action_facts = [fact for fact in ordered
                        if fact.predicate in {"test_result", "host_check_result"}
                        and fact.status in {CURRENT, CONFLICT}]
    if not action_facts:
        action_facts = [fact for fact in ordered
                        if fact.kind == ACTION_RESULT
                        or fact.predicate in {"test_result", "host_check_result"}]
    def action_turn(fact: HostFact) -> int:
        value = fact.value if isinstance(fact.value, Mapping) else {}
        raw = value.get("turn") if isinstance(value, Mapping) else None
        return raw if type(raw) is int else -1
    action_facts.sort(key=lambda fact: (action_turn(fact), fact.fact_id))
    last_action = tuple(fact.to_dict() for fact in action_facts[-1:])
    results = tuple(fact.to_dict() for fact in ordered
                    if fact.kind != CONTRACT and fact.status in {CURRENT, CONFLICT})
    contracts = tuple(fact.to_dict() for fact in ordered if fact.kind == CONTRACT)
    open_facts: list[dict[str, object]] = []
    for fact in ordered:
        if fact.kind == CONTRACT:
            continue
        value = fact.value if isinstance(fact.value, Mapping) else {}
        verdict = str(value.get("verdict", "")) if isinstance(value, Mapping) else ""
        if fact.status in {STALE, CONFLICT} or (
                fact.predicate == "obligation_assessment" and verdict.upper() != "SATISFIED"):
            open_facts.append(fact.to_dict())
    if permission_data is None:
        permissions = permission_projection(tools, edit_authority, edit_contract=edit_contract)
    else:
        permissions = copy.deepcopy(dict(permission_data))
    open_host_facts = [HostFact.from_mapping(row) for row in open_facts]
    decision = completion_decision
    if decision is None:
        for key in ("completion_reducer_decision", "completion_decision", "atomic_completion",
                    "completion_status"):
            if source.get(key) is not None:
                decision = source[key]
                break
    decision_row = _decision_mapping(decision)
    needs_work = decision_row.get("needs_work")
    if isinstance(needs_work, Mapping):
        blocking = needs_work.get("blocking", needs_work.get("obligations", ()))
        if isinstance(blocking, Sequence) and not isinstance(blocking, (str, bytes)):
            for item in blocking:
                open_facts.append({
                    "schema": "stateframe-open-work/1",
                    "source": "completion_reducer",
                    "status": "NEEDS_WORK",
                    "item": _copy(item),
                })
    ids = decision_row.get("needs_work_ids", ())
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        known = {str(row.get("obligation_id", row.get("item", "")))
                 for row in open_facts if isinstance(row, Mapping)}
        for ident in ids:
            if str(ident) not in known:
                open_facts.append({"schema": "stateframe-open-work/1",
                                   "source": "completion_reducer", "status": "NEEDS_WORK",
                                   "obligation_id": str(ident)})
    terminal = source.get("terminal_status")
    if isinstance(terminal, Mapping):
        # The coherence projection has already reduced host assessments.  Do
        # not re-open a non-blocking UNKNOWN row while building this control
        # state from the raw fact stream.
        from kernel_obligation_assess import assessment_is_blocking
        open_ids = {str(item) for item in terminal.get("open_ids", ())}
        filtered: list[dict[str, object]] = []
        for row in open_facts:
            ident = row.get("obligation_id", row.get("id", row.get("subject")))
            if ident is not None and str(ident) not in open_ids:
                continue
            verdict = str(row.get("verdict", row.get("status", ""))).upper()
            if verdict in {"UNKNOWN", "AMBIGUOUS", "UNRESOLVED"} and not assessment_is_blocking(row):
                continue
            filtered.append(row)
        open_facts = filtered
        submission = _copy(terminal.get("submission", {}))
    else:
        submission = _submission(decision, open_host_facts)
    return CanonicalControlState(
        revision=current, last_action=last_action, current_results=results,
        contract=contracts, open_work=tuple(open_facts),
        submission=submission,
        permissions=permissions,
        fact_ids=tuple(fact.fact_id for fact in ordered),
        terminal_status=(copy.deepcopy(dict(terminal)) if isinstance(terminal, Mapping) else None),
    )


def _json_line(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def render_current_state(state: CanonicalControlState | Mapping[str, object]) -> str:
    from kernel_terminal_surface_unify import (
        ADVISORY_OBLIGATIONS_FIELD, ADVISORY_OBLIGATIONS_LABEL,
    )
    row = state.to_dict() if isinstance(state, CanonicalControlState) else dict(state)
    revision = _text(row.get("revision", "UNKNOWN"), 240).replace('"', "&quot;")
    sections = [
        ("LAST ACTION", row.get("last_action", [])),
        ("CURRENT RESULTS", row.get("current_results", [])),
        ("CONTRACT", row.get("contract", [])),
        ("OPEN WORK", row.get("open_work", [])),
        ("SUBMISSION", row.get("submission", {})),
        ("PERMISSIONS", row.get("permissions", {})),
    ]
    terminal = row.get("terminal_status")
    coherent_terminal = (isinstance(terminal, Mapping)
                         and terminal.get("schema") == "stateframe-terminal-status/1")
    if coherent_terminal and row.get(ADVISORY_OBLIGATIONS_FIELD):
        sections.insert(4, (ADVISORY_OBLIGATIONS_LABEL,
                            row[ADVISORY_OBLIGATIONS_FIELD]))
    lines = [f'<current_state revision="{revision}">']
    if isinstance(terminal, Mapping):
        lines.extend([
            "public_check_status: " + str(terminal.get("public_check_status", "BLOCKED")),
            "terminal_status: " + str(terminal.get("state", "BLOCKED")),
        ])
    for label, value in sections:
        lines.extend((label, _json_line(value)))
    lines.append("</current_state>")
    return "\n".join(lines)


current_state_block = render_current_state
render_control_state = render_current_state


_REDUNDANT_FIELDS = frozenset({
    "current_state", "host_facts", "completion_verdict", "atomic_completion",
    "completion_status", "completion_contract", "verification", "world",
    "last_action", "latest_receipts", "recent_test_feedback", "test_receipts",
    "probe_receipts", "fault_receipts", "integrity_receipts", "recent_changes",
    "pending", "resume", "green_lock_message", "jobqueue_finish_rendered",
    "host_observed_findings", "obligations", "objective", "ticket", "task_brief",
    "action_prerequisites", "work_record", "final_review", "review", "host_review",
    "completion_advice", "completion_guidance", "done_advice", "submission_advice",
    "repair_advice", "obligation_prose", "obligation_text", "ticket_text",
    "task_description", "brief", "symptom", "working_view",
})


def suppress_redundant_frame_fields(frame: Mapping[str, object], *, mode: str | None = None) -> dict:
    """Suppress duplicate control prose only in the opt-in model projection."""
    selected = host_facts_mode() if mode is None else mode
    if selected not in {"off", "on"}:
        raise ValueError("host facts mode must be off or on")
    projected = copy.deepcopy(dict(frame))
    if selected == "off":
        return projected
    for key in _REDUNDANT_FIELDS:
        projected.pop(key, None)
    return projected


suppress_redundant_fields = suppress_redundant_frame_fields
project_frame = suppress_redundant_frame_fields


def append_current_state(messages: Sequence[dict], state: CanonicalControlState | Mapping[str, object],
                         *, provider_request: bool = True, mode: str | None = None) -> list[dict]:
    selected = host_facts_mode() if mode is None else mode
    result = copy.deepcopy(list(messages))
    if selected == "on" and provider_request:
        result.append({"role": "user", "content": render_current_state(state)})
    return result


@dataclass(frozen=True, slots=True)
class ExposureManifest(Mapping[str, object]):
    request_id: str
    revision: str
    fact_ids: tuple[str, ...]
    permission_digest: str
    completion_digest: str
    state_digest: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": EXPOSURE_SCHEMA,
            "request_id": self.request_id,
            "revision": self.revision,
            "fact_ids": list(self.fact_ids),
            "permission_digest": self.permission_digest,
            "completion_digest": self.completion_digest,
            "state_digest": self.state_digest,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def build_exposure_manifest(
    state_or_frame: CanonicalControlState | Mapping[str, object],
    *, request_id: object = "",
    state: CanonicalControlState | Mapping[str, object] | None = None,
    facts: object = None,
    tools: object = (),
    edit_authority: object = None,
    edit_contract: object = None,
    completion_decision: object = None,
) -> ExposureManifest:
    if isinstance(state_or_frame, CanonicalControlState):
        current_state = state_or_frame
    elif state is not None:
        current_state = state if isinstance(state, CanonicalControlState) else build_current_state(state)
    else:
        current_state = build_current_state(state_or_frame, facts=facts, tools=tools,
                                            edit_authority=edit_authority,
                                            edit_contract=edit_contract,
                                            completion_decision=completion_decision)
    row = current_state.to_dict() if isinstance(current_state, CanonicalControlState) else dict(current_state)
    permissions = row.get("permissions", {})
    submission = row.get("submission", {})
    fact_ids = tuple(str(item) for item in row.get("fact_ids", ()) if str(item))
    return ExposureManifest(
        request_id=str(request_id), revision=str(row.get("revision", "UNKNOWN")),
        fact_ids=fact_ids,
        permission_digest=str(permissions.get("digest", digest(permissions)))
        if isinstance(permissions, Mapping) else digest(permissions),
        completion_digest=digest(submission), state_digest=digest(row),
    )


def journal_exposure_manifest(journal: object, manifest: ExposureManifest | Mapping[str, object],
                             *, dispatch: bool = False) -> object:
    payload = manifest.to_dict() if isinstance(manifest, ExposureManifest) else dict(manifest)
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        return None
    return emitter("exposure_manifest", producer="kernel_host_facts", manifest=payload,
                   payload_schema=EXPOSURE_SCHEMA, fact_ids=payload.get("fact_ids", ()),
                   permission_digest=payload.get("permission_digest"),
                   completion_digest=payload.get("completion_digest"), dispatch=dispatch)


def render_host_facts(facts: Sequence[HostFact | Mapping[str, object]]) -> str:
    rows = [fact.to_dict() if isinstance(fact, HostFact) else dict(fact) for fact in facts]
    return "[HOST_FACTS]\n" + _json_line(rows)


HostFacts = HostFact
derive_facts = derive_host_facts
build_control_state = build_current_state
build_canonical_control_state = build_current_state
render_current_state_block = render_current_state
