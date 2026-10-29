"""Revision-bound counterexample facts shared by coherence and repair seams.

The host may demote a check for completion authority without making its
executed observation disappear.  This module keeps that distinction in one
place: extraction is conservative about execution/revision, while rendering
is deliberately small and typed so every consumer uses the same labels.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence


COUNTEREXAMPLE_TYPE = "ADVISORY/COUNTEREXAMPLE"
COUNTEREXAMPLE_BLOCK_HEADER = "[ADVISORY/COUNTEREXAMPLE]"
COUNTEREXAMPLE_SCHEMA = "stateframe-advisory-counterexample/1"
COUNTEREXAMPLE_PUBLIC_FIELD = "advisory_counterexamples"
COUNTEREXAMPLES_FIELD = "counterexamples"
COUNTEREXAMPLE_LOCK_LABEL = "counterexamples"
COUNTEREXAMPLE_FIELDS = (
    "check_id", "clause", "expected", "observed", "target_symbol", "target_file",
)
READY_WITH_COUNTEREXAMPLES = "READY_WITH_COUNTEREXAMPLES"
REPAIR_ACTION = "repair"

_STATUS_KEYS = ("underlying_status", "underlying_result", "status", "result", "result_class")
_REVISION_KEYS = (
    "snapshot_digest", "snapshot", "repository_revision", "product_revision",
    "artifact_revision", "revision", "current_revision",
)
_RECEIPT_MARKERS = frozenset({
    "check_id", "check", "status", "underlying_status", "result", "result_class",
    "execution", "snapshot_digest", "receipt_id", "observed", "expected",
})
_INVALID_REASONS = frozenset({
    "not_entailed", "wrong_binding", "invalid_fixture", "wrong_observable",
    "conflicting_source", "check_invalid", "check_rebound", "self_contradictory",
    "unbound_entrypoint", "unbound_invocation", "contradicts_public_test",
    "unsatisfiable_contract_cell", "advisory_only",
})


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            result = converter()
        except (AttributeError, TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            result = dataclasses.asdict(value)
        except (TypeError, ValueError):
            result = None
        if isinstance(result, Mapping):
            return copy.deepcopy(dict(result))
    return {}


def _value(value: object, *names: str, default: object = None) -> object:
    row = _mapping(value)
    for name in names:
        if name in row:
            return row[name]
        if hasattr(value, name):
            return copy.deepcopy(getattr(value, name))
    return default


def _rows(value: object) -> list[dict[str, object]]:
    if isinstance(value, Mapping):
        if _RECEIPT_MARKERS.intersection(value):
            return [copy.deepcopy(dict(value))]
        result: list[dict[str, object]] = []
        for key, child in value.items():
            row = _mapping(child)
            if row:
                row.setdefault("id", str(key))
                result.append(row)
        return result
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for child in value:
            row = _mapping(child)
            if row:
                result.append(row)
        return result
    return []


def _nested_rows(value: object, *keys: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in _rows(value):
        for key in keys:
            result.extend(_rows(row.get(key)))
    return result


def _text(value: object) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _status(value: object) -> str:
    for key in _STATUS_KEYS:
        candidate = _value(value, key, default=None)
        if candidate not in (None, ""):
            result = str(candidate).upper()
            if result != "REUSED":
                return result
    return ""


def _execution(value: object, *, presentation: bool = False) -> str:
    candidate = _value(value, "execution", default=None)
    if candidate not in (None, ""):
        return str(candidate).upper()
    # Bounded host-closure presentation omits execution after the executor has
    # already recorded it.  Raw receipts still take the strict path.
    return "EXECUTED" if presentation else ""


def _revision_values(value: object) -> list[object]:
    row = _mapping(value)
    result: list[object] = []
    for key in _REVISION_KEYS:
        candidate = row.get(key)
        if candidate not in (None, "", "UNKNOWN", "MISSING") and not isinstance(candidate, Mapping):
            result.append(candidate)
    for key in ("binding", "after_binding", "current_binding", "revision_binding"):
        nested = row.get(key)
        if isinstance(nested, Mapping):
            result.extend(_revision_values(nested))
    return result


def _unique_values(values: Iterable[object]) -> tuple[object, ...]:
    result: list[object] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, Mapping) or value in (None, "", "UNKNOWN", "MISSING"):
            continue
        marker = str(value)
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return tuple(result)


def _runtime_revisions(runtime: object | None, frame: Mapping[str, object] | None,
                       explicit: object = None) -> tuple[object, ...]:
    values: list[object] = []
    if explicit is not None:
        values.extend(_revision_values({"revision": explicit}))
    if runtime is not None:
        for name in (
                "host_closure_snapshot", "host_facts_revision", "product_revision",
                "current_revision", "revision"):
            values.extend(_revision_values({"revision": getattr(runtime, name, None)}))
        binding = getattr(runtime, "binding", None)
        if callable(binding):
            try:
                values.extend(_revision_values(binding()))
            except (AttributeError, OSError, TypeError, ValueError):
                pass
    if isinstance(frame, Mapping):
        values.extend(_revision_values(frame))
        for key in ("host_closure", "current_state", "host_facts", "requirements_ledger"):
            child = frame.get(key)
            if isinstance(child, Mapping):
                values.extend(_revision_values(child))
    return _unique_values(values)


def _receipt_is_current(row: Mapping[str, object], revisions: Sequence[object], *,
                        presentation: bool = False) -> bool:
    if bool(row.get("stale", False)) or str(row.get("revision_status", "")).upper() == "STALE":
        return False
    receipt_revisions = _revision_values(row)
    if not receipt_revisions:
        marker = row.get("current", row.get("current_revision"))
        return bool(marker is True or str(marker).upper() == "CURRENT" or not revisions)
    if not revisions:
        return True
    return any(str(receipt) == str(current)
               for receipt in receipt_revisions for current in revisions)


def _looks_like_receipt(row: Mapping[str, object]) -> bool:
    return bool(_RECEIPT_MARKERS.intersection(row)) and bool(
        _text(row.get("check_id", row.get("check", row.get("id", "")))))


def _bundle_values(runtime: object | None, frame: Mapping[str, object] | None,
                   key: str) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    if runtime is not None:
        for name in ("_host_check_bundle_cache", "host_check_bundle_cache",
                     "host_check_bundle", "check_bundle"):
            bundle = getattr(runtime, name, None)
            if callable(bundle):
                try:
                    bundle = bundle()
                except (AttributeError, OSError, TypeError, ValueError):
                    bundle = None
            if bundle is not None:
                values.extend(_rows(_value(bundle, key, default=())))
        loader = getattr(runtime, "_host_check_bundle", None)
        if callable(loader):
            try:
                bundle = loader()
            except (AttributeError, OSError, TypeError, ValueError):
                bundle = None
            if bundle is not None:
                values.extend(_rows(_value(bundle, key, default=())))
        for name in ("host_check_definitions", "host_check_specs", "checks", "obligations"):
            values.extend(_rows(getattr(runtime, name, ())))
    if isinstance(frame, Mapping):
        for name in ("checks", "host_checks", "host_check_definitions", "obligations", "obligation_specs"):
            values.extend(_rows(frame.get(name)))
        ledger = frame.get("requirements_ledger")
        if isinstance(ledger, Mapping):
            values.extend(_rows(ledger.get("checks" if key == "checks" else "requirements")))
    return values


def _definition_map(runtime: object | None, frame: Mapping[str, object] | None) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in _bundle_values(runtime, frame, "checks"):
        ident = _text(row.get("check_id", row.get("id", "")))
        if ident:
            result[ident] = row
    return result


def _spec_map(runtime: object | None, frame: Mapping[str, object] | None) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in _bundle_values(runtime, frame, "specs"):
        ident = _text(row.get("obligation_id", row.get("id", "")))
        if ident:
            result[ident] = row
    return result


def _ledger(runtime: object | None, frame: Mapping[str, object] | None) -> Mapping[str, object] | None:
    if runtime is not None and isinstance(getattr(runtime, "requirements_ledger", None), Mapping):
        return getattr(runtime, "requirements_ledger")
    if isinstance(frame, Mapping) and isinstance(frame.get("requirements_ledger"), Mapping):
        return frame["requirements_ledger"]
    return None


def _ledger_check(ledger: Mapping[str, object] | None, check_id: str) -> dict[str, object]:
    if not isinstance(ledger, Mapping):
        return {}
    return next((copy.deepcopy(dict(row)) for row in _rows(ledger.get("checks"))
                 if _text(row.get("id", row.get("check_id", ""))) == check_id), {})


def _assessment_rows(runtime: object | None, frame: Mapping[str, object] | None) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    if runtime is not None:
        result.extend(_rows(getattr(runtime, "host_closure_assessments", ())))
    if isinstance(frame, Mapping):
        result.extend(_rows(frame.get("obligation_assessments")))
        result.extend(_rows(frame.get("assessments")))
        closure = frame.get("host_closure")
        if isinstance(closure, Mapping):
            result.extend(_rows(closure.get("assessments")))
    return result


def _assessment_by_check(runtime: object | None, frame: Mapping[str, object] | None
                         ) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for row in _assessment_rows(runtime, frame):
        ids = row.get("check_ids", row.get("checks", ()))
        if isinstance(ids, str):
            ids = (ids,)
        if isinstance(ids, Sequence):
            for ident in ids:
                if _text(ident):
                    result.setdefault(str(ident), row)
    return result


def _receipt_sources(runtime: object | None, frame: Mapping[str, object] | None
                     ) -> list[tuple[dict[str, object], bool, dict[str, object]]]:
    """Return (receipt, presentation, parent-assessment) rows."""
    result: list[tuple[dict[str, object], bool, dict[str, object]]] = []

    def add(value: object, *, presentation: bool = False,
            parent: Mapping[str, object] | None = None) -> None:
        for row in _rows(value):
            if _looks_like_receipt(row):
                result.append((row, presentation, dict(parent or {})))
            for key in ("receipts", "host_check_receipts", "host_check_results"):
                nested = row.get(key)
                if nested is not None:
                    add(nested, presentation=presentation, parent=row)

    if runtime is not None:
        for name in ("host_check_receipts", "host_repair_check_receipts", "host_check_results"):
            add(getattr(runtime, name, ()))
        for assessment in _rows(getattr(runtime, "host_closure_assessments", ())):
            add(assessment.get("receipts"), parent=assessment)
        cache = getattr(runtime, "host_check_cache", None)
        if isinstance(cache, Mapping):
            add(cache.values())
    if isinstance(frame, Mapping):
        for name in (
                "host_check_receipts", "host_repair_check_receipts", "host_check_results",
                "host_checks", "host_obligation_checks", "check_receipts"):
            add(frame.get(name))
        for name in ("receipts", "test_receipts"):
            add(frame.get(name))
        closure = frame.get("host_closure")
        if isinstance(closure, Mapping):
            add(closure.get("receipts"), presentation=True)
            for assessment in _rows(closure.get("assessments")):
                add(assessment.get("receipts"), presentation=True, parent=assessment)
        for fact in _rows(frame.get("host_facts")):
            if fact.get("predicate") != "host_check_result":
                continue
            value = fact.get("value")
            row = _mapping(value)
            if row:
                row.setdefault("check_id", fact.get("subject", ""))
                row.setdefault("snapshot_digest", fact.get("revision"))
                add(row, parent=fact)
        ledger = frame.get("requirements_ledger")
        if isinstance(ledger, Mapping):
            add(ledger.get("receipts"))
            add(ledger.get("current_receipts"))
    return result


def _first_text(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clause(check: Mapping[str, object], assessment: Mapping[str, object],
            spec: Mapping[str, object], ledger_check: Mapping[str, object]) -> str:
    for source in (check, ledger_check, assessment, spec):
        clause = source.get("clause")
        if isinstance(clause, Mapping):
            clause = clause.get("text")
        if isinstance(clause, str) and clause.strip():
            return clause.strip()
        provenance = source.get("provenance")
        if isinstance(provenance, Mapping):
            span = provenance.get("clause_span")
            if isinstance(span, Mapping) and _text(span.get("text")):
                return _text(span.get("text"))
        span = source.get("clause_span")
        if isinstance(span, Mapping) and _text(span.get("text")):
            return _text(span.get("text"))
    return ""


def _symbol(check: Mapping[str, object], receipt: Mapping[str, object],
            admission: Mapping[str, object]) -> str:
    for source in (receipt, check, admission):
        candidate = source.get("target_symbol", source.get("symbol", source.get("entry_point")))
        if isinstance(candidate, Mapping):
            candidate = candidate.get("symbol", candidate.get("name", ""))
        if _text(candidate):
            return _text(candidate)
    return ""


def _entry_file(symbol: str) -> str:
    module = symbol.split(":", 1)[0].strip()
    if not module or "/" in module and module.endswith(".py"):
        return module
    if module.endswith(".py"):
        return module.replace(".", "/") if "/" not in module else module
    return module.replace(".", "/") + ".py"


def _target_file(check: Mapping[str, object], receipt: Mapping[str, object],
                 admission: Mapping[str, object], spec: Mapping[str, object],
                 symbol: str) -> str:
    for source in (receipt, check, admission):
        candidate = source.get("target_file", source.get("file", source.get("path")))
        if isinstance(candidate, Mapping):
            candidate = candidate.get("path", candidate.get("file", ""))
        if _text(candidate):
            return _text(candidate).replace("\\", "/")
    targets = spec.get("targets", spec.get("mentioned_files", ()))
    if isinstance(targets, str):
        targets = (targets,)
    if isinstance(targets, Sequence):
        for target in targets:
            if _text(target):
                return _text(target).replace("\\", "/")
    return _entry_file(symbol)


def _optional_bool(value: object, *names: str) -> bool | None:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
        elif hasattr(value, name):
            candidate = getattr(value, name)
        else:
            continue
        if isinstance(candidate, bool):
            return candidate
        if str(candidate).lower() in {"true", "yes", "1", "valid", "supported"}:
            return True
        if str(candidate).lower() in {"false", "no", "0", "invalid", "unsupported"}:
            return False
    return None


def _contains_true(value: object, names: set[str], depth: int = 0) -> bool:
    if depth > 8:
        return False
    row = _mapping(value)
    for key, child in row.items():
        if str(key).lower() in names and child is True:
            return True
        if isinstance(child, (Mapping, Sequence)) and not isinstance(
                child, (str, bytes, bytearray)) and _contains_true(child, names, depth + 1):
            return True
    return False


def _policy_classification(check: Mapping[str, object], receipt: Mapping[str, object],
                           ledger: Mapping[str, object] | None,
                           current_revision: object) -> Mapping[str, object] | None:
    """Use the repair policy when present, without requiring that lane offline."""
    try:
        policy = importlib.import_module("kernel_green_lock_repair")
    except (ImportError, ModuleNotFoundError):
        return None
    policy_enabled = getattr(policy, "enabled", None)
    if callable(policy_enabled):
        try:
            if not policy_enabled():
                return None
        except (AttributeError, TypeError, ValueError):
            return None
    classifier = getattr(policy, "classify_check", None)
    if callable(classifier):
        try:
            decision = classifier(check)
        except (AttributeError, TypeError, ValueError):
            decision = None
        if decision is not None:
            gating = _optional_bool(decision, "gating", "accepted")
            if gating is not None:
                reason = _text(_value(decision, "reason", default=""))
                return {
                    "counterexample_authority": gating,
                    "valid": gating,
                    "classification": reason or ("GATING" if gating else "ADVISORY"),
                }
    for name in ("classify_counterexample", "counterexample_classification",
                 "classify_check_receipt"):
        function = getattr(policy, name, None)
        if not callable(function):
            continue
        for kwargs in (
                {"check": check, "receipt": receipt, "ledger": ledger,
                 "current_revision": current_revision},
                {"check": check, "receipt": receipt, "current_revision": current_revision},
                {"check": check, "receipt": receipt},
        ):
            try:
                result = function(**kwargs)
            except (AttributeError, TypeError, ValueError):
                continue
            if isinstance(result, Mapping):
                return result
            if isinstance(result, bool):
                return {"counterexample_authority": result, "valid": result}
    for name in ("counterexample_is_valid", "is_valid_counterexample"):
        function = getattr(policy, name, None)
        if not callable(function):
            continue
        try:
            result = function(check, receipt, current_revision=current_revision)
        except (AttributeError, TypeError, ValueError):
            continue
        if isinstance(result, bool):
            return {"counterexample_authority": result, "valid": result}
    return None


def classify_counterexample(check: Mapping[str, object] | None,
                            receipt: Mapping[str, object], *,
                            ledger: Mapping[str, object] | None = None,
                            current_revision: object = None) -> dict[str, object]:
    """Classify visibility separately from repair/readiness authority.

    A valid ticket-derived check can be ``SHADOW``/not admitted while still
    being a real current observation.  Unbound, contradictory, or explicitly
    invalid checks remain visible but do not remove READY authority.
    """
    check = dict(check or {})
    nested = receipt.get("check")
    if isinstance(nested, Mapping):
        merged = dict(nested)
        merged.update(check)
        check = merged
    explicit = _optional_bool(receipt, "counterexample_authority", "valid", "authoritative")
    if explicit is None:
        explicit = _optional_bool(check, "counterexample_authority", "valid", "authoritative")
    counterexample = receipt.get("counterexample")
    counterexample_row = counterexample if isinstance(counterexample, Mapping) else {}
    invalid_marker = any(key in counterexample_row
                         for key in ("missing_symbol", "binding_error", "wrong_binding"))
    interpretation = str(check.get("interpretation", "")).upper()
    admission = check.get("admission")
    admission_row = admission if isinstance(admission, Mapping) else {}
    reason = _text(admission_row.get("reason", check.get("classification_reason", ""))).lower()
    admission_tier = _text(admission_row.get("tier", "")).upper()
    provenance = check.get("provenance")
    policy = provenance.get("green_lock_repair") if isinstance(provenance, Mapping) else None
    policy_admission = _text(policy.get("admission", "")).upper() if isinstance(policy, Mapping) else ""
    invalid_flags = _contains_true(check, {
        "contradicts_public_expectation", "contradicts_public_test",
        "public_test_contradiction", "contradiction", "unsatisfiable",
        "unsatisfiable_contract", "impossible", "self_contradictory",
    })
    if (not _text(check.get("entry_point"))
            or interpretation in {"AMBIGUOUS", "PROPOSED"}
            or bool(check.get("self_contradictory", check.get("contradictory", False)))
            or invalid_marker or invalid_flags or reason in _INVALID_REASONS
            or admission_tier in {"ADVISORY", "REJECTED"}
            or policy_admission in {"ADVISORY", "REJECTED"}):
        authority = False
        classification = reason.upper() or "ADVISORY"
    elif explicit is False:
        authority = False
        classification = "ADVISORY"
    else:
        policy = _policy_classification(check, receipt, ledger, current_revision)
        policy_value = _optional_bool(policy, "counterexample_authority", "valid", "authoritative")
        if policy_value is not None:
            authority = policy_value
            classification = _text(_value(policy, "classification", "reason", default="")) or (
                "VALID" if authority else "ADVISORY")
        elif explicit is True:
            authority = explicit
            classification = "VALID" if authority else "ADVISORY"
        else:
            # Existing admission is the shared fallback.  SHADOW is a valid
            # observation lacking completion admission; ADVISORY/REJECTED is
            # not a counterexample authority.
            authority = None
            try:
                from kernel_check_admission import admit_check
                decision = admit_check(check, ledger, current_revision=current_revision)
            except (ImportError, TypeError, ValueError):
                decision = None
            tier = str(_value(decision, "tier", default="")).upper()
            if tier in {"GATING", "SHADOW"}:
                authority = True
                classification = tier
            elif tier in {"ADVISORY", "REJECTED"}:
                authority = False
                classification = _text(_value(decision, "reason", default="")) or tier
            else:
                basis = str(check.get("oracle_basis", receipt.get("oracle_basis", ""))).upper()
                authority = bool(
                    basis in {"TICKET", "PUBLIC_TEST", "PRESERVED_BEHAVIOR"}
                    and _text(check.get("entry_point"))
                    and (_text(check.get("clause")) or _text(check.get("obligation_id")))
                )
                classification = "VALID" if authority else "ADVISORY"
    admitted = _optional_bool(receipt, "admitted", "admission_authority")
    if admitted is None:
        admitted = _optional_bool(check, "admitted", "admission_authority")
    if admitted is None:
        try:
            from kernel_check_admission import is_admitted_counterexample
            admitted = bool(is_admitted_counterexample(
                check, receipt, current_revision=current_revision))
        except (ImportError, TypeError, ValueError):
            admitted = False
    tier = _text(admission_row.get("tier", ""))
    return {
        "counterexample_authority": bool(authority),
        "admitted": bool(admitted),
        "classification": str(classification).upper(),
        "admission_tier": tier.upper() if tier else None,
    }


def _record_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        _text(row.get("check_id")),
        _text(row.get("receipt_id", row.get("evidence_ref", ""))),
        json.dumps((row.get("observed"), row.get("expected")), sort_keys=True, default=str),
    )


def _first_present(sources: Sequence[Mapping[str, object]], *keys: str) -> object:
    for raw_source in sources:
        source = raw_source if isinstance(raw_source, Mapping) else {}
        for key in keys:
            value = source.get(key)
            if value not in (None, ""):
                return copy.deepcopy(value)
    return None


def _binding_status(sources: Sequence[Mapping[str, object]], *, bound_hint: bool) -> str:
    value = _first_present(
        sources, "binding_status", "entrypoint_binding", "binding", "binding_verdict")
    if isinstance(value, Mapping):
        value = _first_present([value], "status", "verdict", "binding", "value")
    if value not in (None, ""):
        return str(value).upper()
    return "BOUND" if bound_hint else "UNBOUND"


def _mutation_sites(sources: Sequence[Mapping[str, object]]) -> tuple[object, ...]:
    values: list[object] = []
    for raw_source in sources:
        source = raw_source if isinstance(raw_source, Mapping) else {}
        for key in ("mutation_site_refs", "mutation_sites", "mutation_site",
                    "input_mutation_sites", "input_mutation_trace", "mutations"):
            value = source.get(key)
            if value in (None, ""):
                continue
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                values.extend(copy.deepcopy(list(value)))
            else:
                values.append(copy.deepcopy(value))
    unique: list[object] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
        if marker not in seen:
            seen.add(marker)
            unique.append(value)
    return tuple(unique)


def collect_counterexamples(runtime: object | None = None,
                            frame: Mapping[str, object] | None = None, *,
                            current_revision: object = None) -> tuple[dict[str, object], ...]:
    """Collect all executed current-revision counterexamples, including shadow ones."""
    revisions = _runtime_revisions(runtime, frame, current_revision)
    definitions = _definition_map(runtime, frame)
    specs = _spec_map(runtime, frame)
    ledger = _ledger(runtime, frame)
    assessments = _assessment_by_check(runtime, frame)
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, object]] = []
    sources = list(_receipt_sources(runtime, frame))
    # r38: a LATER current-revision PASS of the same check closes an earlier
    # counterexample (same rule as the open-obligation policy), so every consumer
    # of this collector agrees on which observations are still live.
    closed_after: dict[str, int] = {}
    for position, (receipt, presentation, _parent) in enumerate(sources):
        if not _receipt_is_current(receipt, revisions, presentation=presentation):
            continue
        ident = _text(receipt.get("check_id", receipt.get("check", receipt.get("id", ""))))
        if ident and _status(receipt) in {"PASS", "PASSED", "SATISFIED"}:
            closed_after[ident] = position
    for position, (receipt, presentation, parent) in enumerate(sources):
        if _status(receipt) != "COUNTEREXAMPLE" or _execution(receipt, presentation=presentation) != "EXECUTED":
            continue
        ident = _text(receipt.get("check_id", receipt.get("check", receipt.get("id", ""))))
        if ident and closed_after.get(ident, -1) > position:
            continue
        if not _receipt_is_current(receipt, revisions, presentation=presentation):
            continue
        check_id = _text(receipt.get("check_id", receipt.get("check", receipt.get("id", ""))))
        if not check_id:
            continue
        key = _record_key(receipt)
        if key in seen:
            continue
        seen.add(key)
        check = definitions.get(check_id, {})
        ledger_check = _ledger_check(ledger, check_id)
        if ledger_check:
            merged_check = dict(ledger_check)
            merged_check.update(check)
            check = merged_check
        assessment = assessments.get(check_id, parent)
        obligation_id = _text(
            receipt.get("obligation_id", check.get("obligation_id",
                         assessment.get("obligation_id", ""))))
        spec = specs.get(obligation_id, {})
        classification = classify_counterexample(
            check, receipt, ledger=ledger, current_revision=(revisions[0] if revisions else None))
        admission = check.get("admission") if isinstance(check.get("admission"), Mapping) else {}
        symbol = _symbol(check, receipt, admission)
        observed = copy.deepcopy(receipt.get("observed", receipt.get("observed_values")))
        expected = copy.deepcopy(receipt.get("expected", check.get("expected")))
        counterexample = receipt.get("counterexample")
        counterexample_row = counterexample if isinstance(counterexample, Mapping) else {}
        sources_for_repair = (receipt, check, counterexample_row, admission, spec, assessment)
        mutation_sites = _mutation_sites(sources_for_repair)
        explicit_reproducible = _optional_bool(
            receipt, "reproducible", "reproducibility")
        if explicit_reproducible is None:
            explicit_reproducible = _optional_bool(check, "reproducible", "reproducibility")
        bound_hint = bool(_text(check.get("entry_point")) and symbol)
        oracle_basis = _first_present(
            sources_for_repair, "oracle_basis", "ticket_basis", "basis")
        demotion_reason = _first_present(
            sources_for_repair, "demotion_reason", "classification_reason", "reason")
        repair_fields: dict[str, object] = {}
        if (getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
                or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on"):
            if not bool(classification.get("counterexample_authority")):
                if not _text(check.get("entry_point")):
                    demotion_reason = "UNBOUND_ENTRYPOINT"
                elif bool(check.get("self_contradictory", check.get("contradictory", False))):
                    demotion_reason = "SELF_CONTRADICTORY"
                elif (any(key in counterexample_row for key in
                          ("missing_symbol", "binding_error", "wrong_binding"))
                      or _contains_true(check, {
                          "contradicts_public_expectation", "contradicts_public_test",
                          "public_test_contradiction", "contradiction", "unsatisfiable",
                          "unsatisfiable_contract", "impossible", "self_contradictory",
                      })):
                    demotion_reason = "INVALID_BINDING"
                else:
                    demotion_reason = classification.get("classification") or demotion_reason
            repair_fields = {
                "oracle_basis": oracle_basis,
                "binding": _binding_status(sources_for_repair, bound_hint=bound_hint),
                "execution": "EXECUTED",
                "reproducible": (bool(explicit_reproducible)
                                  if explicit_reproducible is not None else True),
                "mutation_site_refs": [item for item in mutation_sites
                                        if isinstance(item, str)],
                "mutation_sites": [item for item in mutation_sites
                                    if not isinstance(item, str)],
                "demotion_reason": (
                    str(demotion_reason) if demotion_reason not in (None, "")
                    else str(classification.get("classification") or "ADVISORY")),
                # every identity of the current revision this receipt was proven against
                "revision_aliases": [str(item) for item in revisions],
            }
        row = {
            "schema": COUNTEREXAMPLE_SCHEMA,
            "type": COUNTEREXAMPLE_TYPE,
            "check_id": check_id,
            "obligation_id": obligation_id,
            "clause": _clause(check, assessment, spec, ledger_check),
            "expected": expected,
            "observed": observed,
            "target_symbol": symbol,
            "target_file": _target_file(check, receipt, admission, spec, symbol),
            "revision": copy.deepcopy(next(iter(_revision_values(receipt)),
                                             revisions[0] if revisions else None)),
            **classification,
            **repair_fields,
        }
        if os.environ.get("STATEFRAME_TOKENLAB_COUNTEREXAMPLE_REFUSAL", "off").strip().lower() == "on":
            try:
                from kernel_tokenlab_counterexample_refusal import enrich_row
                row = enrich_row(row, check=check, receipt=receipt)
            except (ImportError, TypeError, ValueError):
                pass
        if os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on":
            try:
                from kernel_requirement_alignment import enrich_counterexample
                row = enrich_counterexample(row, ledger=ledger, check=check)
            except (ImportError, TypeError, ValueError):
                pass
        if os.environ.get("STATEFRAME_NAMED_LAYER_REPAIR", "off").strip().lower() == "on":
            # autopsy-a3b-r44c-ldev01 A: a resolved entry point that rejects a ticket-literal call is a
            # repairable SIGNATURE_GAP, not INVALID_BINDING.  Unresolved/unaligned rows are unchanged.
            from kernel_named_layer_repair import apply_signature_gap
            row = apply_signature_gap(row, receipt)
        result.append(row)
    return tuple(result)


def counterexample_ids(rows: object) -> tuple[str, ...]:
    result: list[str] = []
    for row in _rows(rows):
        ident = _text(row.get("check_id", row.get("obligation_id", row.get("id", ""))))
        if ident and ident not in result:
            result.append(ident)
        obligation = _text(row.get("obligation_id"))
        if obligation and obligation not in result:
            result.append(obligation)
    return tuple(result)


def _target_path(value: object, runtime: object | None = None) -> str:
    text = _text(value).replace("\\", "/")
    if not text:
        return ""
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is not None:
        try:
            from pathlib import Path
            candidate = Path(text)
            if not candidate.is_absolute():
                candidate = Path(root) / candidate
            return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
        except (OSError, TypeError, ValueError):
            pass
    return text.lstrip("./")


def _target_tokens(value: object, runtime: object | None = None) -> dict[str, set[str]]:
    paths: set[str] = set()
    symbols: set[str] = set()
    ids: set[str] = set()
    path_keys = {"file", "path", "target", "target_file", "target_path", "location"}
    symbol_keys = {"entry", "entry_point", "function", "symbol", "target_symbol"}
    id_keys = {"check", "check_id", "obligation", "obligation_id", "selector", "id"}

    def visit(item: object, key: str = "", depth: int = 0) -> None:
        if depth > 8:
            return
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                visit(child, str(raw_key).lower(), depth + 1)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child, key, depth + 1)
            return
        text = _text(item)
        if not text:
            return
        if key in path_keys:
            path = _target_path(text, runtime)
            if path:
                paths.add(path)
        if key in symbol_keys:
            symbols.add(text)
            module, separator, function = text.partition(":")
            if separator:
                symbols.update({module, function, module.replace(".", "/")})
                paths.add(_target_path(module.replace(".", "/") + ".py", runtime))
        if key in id_keys:
            ids.add(text)

    visit(value)
    return {"paths": {item for item in paths if item}, "symbols": symbols, "ids": ids}


def repair_counterexample_targets(runtime: object | None = None,
                                  frame: Mapping[str, object] | None = None
                                  ) -> tuple[dict[str, object], ...]:
    """Return current, valid observations that a repair controller may target."""
    if (getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on"
            or getattr(runtime, "stateframe_repair_blocks_done_mode", "off") == "on"):
        try:
            from kernel_readiness_authority import runtime_snapshot
            decision = runtime_snapshot(runtime, frame)
        except (AttributeError, ImportError, TypeError, ValueError):
            decision = None
        if decision is not None:
            result: list[dict[str, object]] = []
            authority_rows = list(getattr(decision, "blockers", ()) or ())
            authority_rows.extend(getattr(decision, "repairable_advisories", ()) or ())
            for raw in authority_rows:
                row = copy.deepcopy(dict(raw))
                row["targets"] = _target_tokens({
                    "check_id": row.get("check_id"),
                    "obligation_id": row.get("obligation_id"),
                    "target_symbol": row.get("target_symbol"),
                    "target_file": row.get("target_file"),
                    "mutation_site_refs": row.get("mutation_site_refs", ()),
                }, runtime)
                result.append(row)
            return tuple(result)
    result: list[dict[str, object]] = []
    for row in collect_counterexamples(runtime, frame):
        if not bool(row.get("counterexample_authority")):
            continue
        target = _target_tokens({
            "check_id": row.get("check_id"),
            "obligation_id": row.get("obligation_id"),
            "target_symbol": row.get("target_symbol"),
            "target_file": row.get("target_file"),
        }, runtime)
        enriched = copy.deepcopy(row)
        enriched["targets"] = target
        result.append(enriched)
    return tuple(result)


def action_targets_counterexample(action: object, arguments: object = None,
                                  runtime: object | None = None,
                                  frame: Mapping[str, object] | None = None) -> bool:
    """Match an edit/read/test argument to a current repair target.

    The repair lane owns refusal and verification policy.  This small shared
    predicate lets it accept a valid shadow observation without treating an
    invalid or contradictory observation as completion authority.
    """
    candidates = _target_tokens(arguments, runtime)
    if not any(candidates.values()):
        return False
    for row in repair_counterexample_targets(runtime, frame):
        expected = _target_tokens({
            "check_id": row.get("check_id"),
            "obligation_id": row.get("obligation_id"),
            "target_symbol": row.get("target_symbol"),
            "target_file": row.get("target_file"),
        }, runtime)
        if candidates["ids"] & expected["ids"]:
            return True
        if candidates["symbols"] & expected["symbols"]:
            return True
        if candidates["paths"] & expected["paths"]:
            return True
    return False


def render_counterexample_blocks(rows: object) -> str:
    """Render the single shared typed block label consumed by every surface."""
    lines: list[str] = []
    for row in _rows(rows):
        fields = (
            "schema", "type", *COUNTEREXAMPLE_FIELDS,
            "obligation_id", "revision", "counterexample_authority", "admitted",
            "classification", "admission_tier",
        )
        if (os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on"
                and (row.get("alignment_evidence") is not None
                     or row.get("structural_diff") is not None)):
            fields += (
                "alignment_evidence", "alignment_requirement_id", "literal_grounded",
                "repairable", "structural_diff",
            )
        payload = {key: copy.deepcopy(row.get(key)) for key in fields if key in row}
        payload.setdefault("schema", COUNTEREXAMPLE_SCHEMA)
        payload.setdefault("type", COUNTEREXAMPLE_TYPE)
        lines.append(COUNTEREXAMPLE_BLOCK_HEADER)
        lines.append(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), default=str))
    return "\n".join(lines)


def parse_counterexample_blocks(value: object) -> list[dict[str, object]]:
    """Parse only the producer's label, used by contract tests/repair consumers."""
    lines = str(value or "").splitlines()
    result: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if line.strip() != COUNTEREXAMPLE_BLOCK_HEADER or index + 1 >= len(lines):
            continue
        try:
            row = json.loads(lines[index + 1])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(row, Mapping) and row.get("type") == COUNTEREXAMPLE_TYPE:
            result.append(dict(row))
    return result


__all__ = [
    "COUNTEREXAMPLE_BLOCK_HEADER", "COUNTEREXAMPLE_FIELDS", "COUNTEREXAMPLE_SCHEMA",
    "COUNTEREXAMPLES_FIELD", "COUNTEREXAMPLE_LOCK_LABEL", "COUNTEREXAMPLE_PUBLIC_FIELD",
    "COUNTEREXAMPLE_TYPE",
    "READY_WITH_COUNTEREXAMPLES", "REPAIR_ACTION",
    "action_targets_counterexample", "classify_counterexample", "collect_counterexamples",
    "counterexample_ids", "parse_counterexample_blocks", "render_counterexample_blocks",
    "repair_counterexample_targets",
]
