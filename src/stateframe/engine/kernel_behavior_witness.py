"""Typed public behavior witnesses and their bounded host adjudication."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


MODE_ENV = "STATEFRAME_BEHAVIOR_WITNESS"
ALL_DONE_GATE_ENV = "STATEFRAME_WITNESS_GATE_ALL_DONE"
REFUSAL_SETTLE_ENV = "STATEFRAME_PROBE_REFUSAL_SETTLE"
SCHEMA = "stateframe-behavior-witness/1"
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
_SOURCE_UNIT_ENTRYPOINT_RE = re.compile(r"^[A-Za-z0-9_.<>/-]+$")
_PASS = frozenset({"COMPLETED", "PASS", "PASSED", "GREEN", "SUCCESS", "OK"})
_OPEN_REASONS = frozenset({
    "MISSING_CHECKS", "UNBOUND_OBSERVABLE", "NO_VERIFIER", "REQUIRED_EVIDENCE_GAP",
    "CLAUSE_COVERAGE", "COVERAGE_OPEN", "UNKNOWN", "INCONCLUSIVE", "NOT_EXECUTED",
})
_OBSERVED_MARKER = "__STATEFRAME_WITNESS_OBSERVED__"
_UNSET = object()
MAX_REVISED_PROPOSALS = 1
MAX_DONE_GATE_REQUESTS = 3
UNRUNNABLE_RESULT = "UNRUNNABLE"
_REVISABLE_ADMISSION_REASONS = frozenset({
    "entrypoint_unknown_module",
    "entrypoint_unknown_attribute",
    "entrypoint_private",
    "entrypoint_public_source_not_parseable",
    "entrypoint_not_in_public_interface",
})
_PUBLIC_INIT_FILES = frozenset({"__init__.py", "__main__.py"})
_PRIVATE_PATH_PARTS = frozenset({"private", "hidden", "gold", "oracle", "backup", "backups"})
_PUBLIC_SUFFIXES = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".php", ".vue",
})


@dataclass(frozen=True)
class _ClassInterface:
    methods: Mapping[str, tuple[str, bool]]
    constructor_requires_input: bool


@dataclass(frozen=True)
class _ModuleInterface:
    functions: frozenset[str]
    classes: Mapping[str, _ClassInterface]
    cli: bool
    language: str = "python"


@dataclass(frozen=True)
class PublicInterfaceIndex:
    """AST-only view of the current public product modules.

    The index contains names and call-shape metadata only; admission never imports
    or executes a product module. ``text`` is retained solely for literal provenance.
    """

    modules: Mapping[str, _ModuleInterface]
    cli_modules: frozenset[str]
    source_paths: tuple[str, ...]
    text: str
    parse_errors: Mapping[str, str]
    # Source visibility is separate from public API naming.  A caller still
    # needs an explicit opaque handle before a private/nested unit is admitted.
    source_units: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class _EntrypointMatch:
    kind: str
    module: str
    owner: str = ""
    attribute: str = ""
    constructor_requires_input: bool = False
    method_requires_input: bool = False


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def all_done_gate_mode() -> str:
    value = os.environ.get(ALL_DONE_GATE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{ALL_DONE_GATE_ENV} must be off or on")
    return value


def refusal_settle_mode() -> str:
    value = os.environ.get(REFUSAL_SETTLE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{REFUSAL_SETTLE_ENV} must be off or on")
    return value


def refusal_settle_enabled() -> bool:
    return refusal_settle_mode() == "on"


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: object) -> str:
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value if isinstance(value, str) else str(value or "")


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _material_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_material_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            keys.update(_material_keys(item))
    return keys


def _literal_rows(value: object, path: str) -> tuple[tuple[str, object], ...]:
    """Return JSON literals with paths; this does not interpret their meaning."""
    rows: list[tuple[str, object]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            rows.append((f"{path}.{key} (key)", str(key)))
            rows.extend(_literal_rows(item, f"{path}.{key}"))
        return tuple(rows)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            rows.extend(_literal_rows(item, f"{path}[{index}]"))
        return tuple(rows)
    return ((path, value),)


def _literal_pattern(value: object) -> str | None:
    if value is None:
        return r"(?<![A-Za-z0-9_])(?:null|None)(?![A-Za-z0-9_])"
    if isinstance(value, bool):
        text = "true|True" if value else "false|False"
        return rf"(?<![A-Za-z0-9_])(?:{text})(?![A-Za-z0-9_])"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = re.escape(str(value))
        return rf"(?<![A-Za-z0-9_.]){text}(?![A-Za-z0-9_.])"
    if isinstance(value, str) and value:
        if re.fullmatch(r"[A-Za-z0-9_]+", value):
            return rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])"
        return re.escape(value)
    return None


def _literal_in(text: str, value: object) -> bool:
    pattern = _literal_pattern(value)
    return pattern is not None and re.search(pattern, text) is not None


def _rationale_shows_arithmetic(rationale: str, value: object) -> bool:
    """Recognize a model-displayed numeric expression without evaluating it."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    pattern = _literal_pattern(value)
    if pattern is None:
        return False
    for match in re.finditer(pattern, rationale):
        fragment = rationale[max(0, match.start() - 48):match.end() + 48]
        if re.search(r"(?:[+*/%]|->|=>|=)", fragment) is not None:
            return True
    return False


def _assessment_dict(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    result: dict[str, object] = {}
    for name in ("obligation_id", "id", "verdict", "status", "reason", "reason_code",
                 "check_ids", "checks", "clause", "exact_clause", "changed_surface",
                 "surface_changed", "changed", "changed_surfaces", "changed_surface_ids",
                 "coverage_required", "coverage_disposition", "expected", "observed",
                 "receipts", "kind", "entrypoint", "public_entrypoint"):
        item = getattr(value, name, None)
        if item is not None:
            result[name] = item
    return result


def _assessments(runtime: object, extra: Sequence[object] = ()) -> tuple[dict[str, object], ...]:
    values: list[object] = []
    for name in ("host_closure_assessments", "behavior_witness_assessments",
                 "obligation_assessments"):
        candidate = getattr(runtime, name, None)
        if candidate:
            values.extend(_sequence(candidate))
    values.extend(extra)
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in values:
        mapped = _assessment_dict(item)
        if not mapped:
            continue
        ident = _text(mapped.get("obligation_id", mapped.get("id", "")))
        if ident and not _text(mapped.get("kind")):
            ledger = getattr(runtime, "requirements_ledger", None)
            requirements = _field(ledger, "requirements", ())
            for requirement in _sequence(requirements):
                if (_text(_field(requirement, "id", _field(requirement, "obligation_id", "")))
                        == ident):
                    kind = _text(_field(requirement, "kind", "")).strip().lower()
                    if kind:
                        mapped["kind"] = kind
                    break
        key = ident or _json_digest(mapped)
        if key in seen:
            continue
        seen.add(key)
        result.append(mapped)
    return tuple(result)


def _current_revision(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            return copy.deepcopy(binding())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    return copy.deepcopy(getattr(runtime, "product_revision", ""))


def _changed_surface(value: Mapping[str, object]) -> bool:
    for name in ("changed_surface", "surface_changed", "changed", "changed_surfaces",
                 "changed_surface_ids", "surface_ids"):
        candidate = value.get(name)
        if isinstance(candidate, bool):
            if candidate:
                return True
        elif isinstance(candidate, str):
            if candidate.strip():
                return True
        elif isinstance(candidate, Mapping):
            if candidate:
                return True
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            if candidate:
                return True
    return False


def _is_behavior_clause(assessment: Mapping[str, object]) -> bool:
    clause = _text(assessment.get("clause", assessment.get("exact_clause", ""))).lower()
    reason = _text(assessment.get("reason", assessment.get("reason_code", ""))).upper()
    kind = _text(assessment.get("kind", "")).strip().lower()
    checks = assessment.get("check_ids", assessment.get("checks", ()))
    if (not clause or reason not in _OPEN_REASONS
            or (kind and kind not in {"behavior", "error_behavior"})):
        return False
    # Required public-surface coverage is a typed obligation property. It is
    # behavior evidence even when the edit did not carry a changed-surface
    # marker, such as an unexercised alternate entrypoint.
    if checks and not _changed_surface(assessment) and not bool(
            assessment.get("coverage_required", False)):
        return False
    return _text(assessment.get("status", assessment.get("verdict", ""))).upper() not in {
        "PASS", "PASSED", "SATISFIED", "COMPLETED", "RESOLVED",
    }


def _same_clause(left: object, right: object) -> bool:
    return bool(left) and bool(right) and _text(left) == _text(right)


def _latest_record(records: object, clause_id: str, revision: object) -> Mapping[str, object] | None:
    latest: Mapping[str, object] | None = None
    for record in _sequence(records):
        if not isinstance(record, Mapping):
            continue
        if (_text(record.get("clause_id")) == clause_id
                and _text(record.get("revision_digest")) == _json_digest(revision)):
            latest = record
    return latest


def _passed_record(records: object, clause_id: str, revision: object) -> bool:
    latest = _latest_record(records, clause_id, revision)
    return latest is not None and _text(latest.get("result")).upper() == "PASS"


def _typed_receipt_passes(assessment: Mapping[str, object]) -> bool:
    """Treat a reducer-selected current PASS as already exercised evidence."""
    for receipt in _sequence(assessment.get("receipts", ())):
        status = _text(_field(receipt, "status", _field(receipt, "result", ""))).upper()
        if status in _PASS:
            return True
    return False


def _unrunnable_record(records: object, clause_id: str, revision: object) -> bool:
    revision_digest = _json_digest(revision)
    latest: Mapping[str, object] | None = None
    for record in _sequence(records):
        if (isinstance(record, Mapping)
                and _text(record.get("clause_id")) == clause_id
                and _text(record.get("revision_digest")) == revision_digest):
            latest = record
    return (latest is not None
            and _text(latest.get("result")).upper() == UNRUNNABLE_RESULT)


def _typed_probe_available(runtime: object) -> bool:
    """A witness gate may block only when its typed producer is on the wire."""
    try:
        from kernel_completion_owner import _typed_probe_tool
        current_menu = getattr(runtime, "completion_menu", None)
        if isinstance(current_menu, Sequence) and not isinstance(current_menu, (str, bytes, bytearray)):
            if not current_menu or "probe" not in {str(item) for item in current_menu}:
                return False
        saw_surface = False
        for name in ("_completion_owner_full_tools", "_completion_owner_dispatch_tools"):
            if hasattr(runtime, name):
                saw_surface = True
            for tool in _sequence(getattr(runtime, name, ())):
                if _typed_probe_tool(tool):
                    return True
        # Legacy gate-only test doubles do not model the request surface.  Keep
        # their bounded gate semantics; a live surface with no typed probe must
        # remain unavailable rather than inventing an executable capability.
        if not saw_surface and not hasattr(runtime, "completion_menu"):
            return True
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    return False


def _probe_remaining(runtime: object) -> int | None:
    try:
        from kernel_route_budget import _budget
        remaining = _budget(runtime).get("remaining")
        return int(remaining) if type(remaining) is int else None
    except (ImportError, AttributeError, TypeError, ValueError):
        value = getattr(runtime, "probe_remaining", None)
        return int(value) if type(value) is int else None


def _valid_witness_entrypoint(value: object) -> bool:
    entrypoint = _text(value).strip()
    return bool(_ENTRYPOINT_RE.fullmatch(entrypoint)) and entrypoint != "public.module.function"


def _witness_submittable(runtime: object, wanted: Sequence[str],
                         assessments: Sequence[Mapping[str, object]], records: object,
                         revision: object) -> bool:
    """Require the complete host-backed surface before classifying work as pending."""
    if not _typed_probe_available(runtime):
        return False
    remaining = _probe_remaining(runtime)
    if remaining is None or remaining <= 0:
        return False
    entrypoints: dict[str, str] = {}
    for assessment in assessments:
        clause_id = _text(assessment.get("obligation_id", assessment.get("id")))
        if clause_id:
            entrypoints[clause_id] = _text(
                assessment.get("entrypoint", assessment.get("public_entrypoint", "")))
    revision_digest = _json_digest(revision)
    for record in _sequence(records):
        if (isinstance(record, Mapping)
                and _text(record.get("revision_digest")) == revision_digest):
            clause_id = _text(record.get("clause_id"))
            if clause_id and clause_id not in entrypoints:
                entrypoints[clause_id] = _text(record.get("entrypoint"))
    requested = tuple(dict.fromkeys(_text(item) for item in wanted if _text(item)))
    if requested:
        return all(_valid_witness_entrypoint(entrypoints.get(clause_id))
                   for clause_id in requested)
    return any(_valid_witness_entrypoint(value) for value in entrypoints.values())


def pending_public_evidence_work(runtime: object, open_ids: Sequence[str] = (),
                                 assessments: Sequence[object] = ()) -> bool:
    """Return whether public behavior evidence still blocks an uncertified submit."""
    if not enabled():
        return False
    revision = _current_revision(runtime)
    records = getattr(runtime, "behavior_witness_records", ())
    typed_assessments = _assessments(runtime, assessments)
    wanted = {str(item) for item in open_ids if str(item)}
    candidates = [item for item in typed_assessments
                  if not wanted or _text(item.get("obligation_id", item.get("id"))) in wanted]
    if not candidates and wanted:
        # Settle explicit host refusals before treating an absent assessment as
        # pending.  An UNRUNNABLE probe is a truthful terminal witness result.
        if all(_unrunnable_record(records, clause_id, revision)
               or _passed_record(records, clause_id, revision)
               for clause_id in wanted):
            return False
        # An open id without a host-readable assessment remains unresolved only
        # while a concrete current witness can actually be submitted.
        return _witness_submittable(runtime, wanted, candidates, records, revision)
    for assessment in candidates:
        if not _is_behavior_clause(assessment):
            continue
        clause_id = _text(assessment.get("obligation_id", assessment.get("id")))
        if _typed_receipt_passes(assessment) or _unrunnable_record(records, clause_id, revision):
            continue
        if (not _passed_record(records, clause_id, revision)
                and _witness_submittable(runtime, (clause_id,), (assessment,), records, revision)):
            return True
    latest: dict[str, Mapping[str, object]] = {}
    for record in _sequence(records):
        if (isinstance(record, Mapping) and record.get("admitted") is True
                and _text(record.get("revision_digest")) == _json_digest(revision)):
            latest[_text(record.get("clause_id"))] = record
    for record in latest.values():
        if (_text(record.get("result")).upper() != "PASS"
                and _witness_submittable(runtime, (str(record.get("clause_id",)),),
                                         (record,), records, revision)):
            return True
    return False


def pending_clauses(runtime: object, open_ids: Sequence[str] = (),
                    assessments: Sequence[object] = ()) -> tuple[dict[str, object], ...]:
    wanted = {str(item) for item in open_ids if str(item)}
    records = getattr(runtime, "behavior_witness_records", ())
    revision = _current_revision(runtime)
    return tuple(item for item in _assessments(runtime, assessments)
                  if (not wanted or _text(item.get("obligation_id", item.get("id"))) in wanted)
                  and _is_behavior_clause(item)
                  and not _passed_record(
                      records, _text(item.get("obligation_id", item.get("id"))), revision)
                  and not _typed_receipt_passes(item)
                  and not _unrunnable_record(
                      records, _text(item.get("obligation_id", item.get("id"))), revision))


def request_block(runtime: object, open_ids: Sequence[str] = (),
                  assessments: Sequence[object] = ()) -> str:
    rows = pending_clauses(runtime, open_ids, assessments)
    if not rows:
        return ""
    clauses = []
    for row in rows:
        entrypoint = _text(row.get("entrypoint", row.get("public_entrypoint", "")))
        if not _ENTRYPOINT_RE.fullmatch(entrypoint) or entrypoint == "public.module.function":
            continue
        clauses.append({"clause_id": _text(row.get("obligation_id", row.get("id"))),
                        "entrypoint": entrypoint,
                        "clause": _text(row.get("clause"))[:600]})
    if not clauses:
        fallback_id = _text(rows[0].get("obligation_id", rows[0].get("id", ""))) if rows else ""
        fallback_path = _text(rows[0].get("path", rows[0].get("file", ""))) if rows else ""
        if not fallback_path:
            files = getattr(getattr(runtime, "owner", None), "task_row", {})
            files = files.get("files") if isinstance(files, Mapping) else None
            if isinstance(files, Mapping) and files:
                fallback_path = sorted(str(path) for path in files)[0]
        recovery = ({"tool": "read", "arguments": {
            "path": fallback_path, "scope": "public", "cursor": 0,
        }} if fallback_path else {"tool": "host_blocked", "arguments": {
            "reason": "no public path or callable is currently admitted",
        }})
        return ("[PUBLIC BEHAVIOR WITNESS]\n"
                "No executable public callable is currently admitted. Use this bounded "
                "recovery route; the clause remains UNKNOWN until a current public contract "
                "is located.\n" + json.dumps({"clause_id": fallback_id,
                "status": "UNRUNNABLE", "recovery": recovery},
                sort_keys=True, ensure_ascii=False))
    content = ("[PUBLIC BEHAVIOR WITNESS]\n"
            "Propose one typed JSON object per listed clause in the next probe turn, "
            "using the exact clause_id and entrypoint pair shown below; include input, "
            "expected, and rationale fields. "
            "Input and expected values must be derivable from this public brief and interface; "
            "host records provenance and runs the admitted case.\n"
            + json.dumps(clauses, sort_keys=True, ensure_ascii=False))
    revision = _current_revision(runtime)
    wanted = {str(item) for item in open_ids if str(item)}
    latest: dict[str, Mapping[str, object]] = {}
    for record in _sequence(getattr(runtime, "behavior_witness_records", ())):
        if (isinstance(record, Mapping) and record.get("admitted") is True
                and _text(record.get("revision_digest")) == _json_digest(revision)
                and _text(record.get("result")).upper() != "PASS"
                and (not wanted or _text(record.get("clause_id")) in wanted)):
            latest[_text(record.get("clause_id"))] = record
    for record in latest.values():
        content += ("\n\n[PUBLIC CASE JUDGMENT]\n"
                    "The admitted case did not match. The model must judge code versus expectation; "
                    "a revised proposal is allowed once and must pass the same admission checks. "
                    "A mismatch is case evidence, not clause certification.\n"
                    + json.dumps({"clause": record.get("clause"), "input": record.get("input"),
                                  "expected": record.get("expected"), "observed": record.get("observed")},
                                 sort_keys=True, ensure_ascii=False, default=str))
    return content


def record_unrunnable_proposal(runtime: object, proposal: Mapping[str, object], *,
                               reason: str, clause_id: str, clause: str = "",
                               revision: object = None) -> dict[str, object]:
    """Record a typed proposal that the public runner could not execute."""
    current_revision = _current_revision(runtime) if revision is None else copy.deepcopy(revision)
    record = {
        "schema": SCHEMA, "witness_id": "", "clause_id": str(clause_id),
        "admitted": False, "result": UNRUNNABLE_RESULT, "reason": str(reason)[:240],
        "clause": str(clause), "revision": copy.deepcopy(current_revision),
        "revision_digest": _json_digest(current_revision),
        "proposal": copy.deepcopy(dict(proposal)), "evidence_scope": "typed_admission",
        "certifies_clause": False,
    }
    records = getattr(runtime, "behavior_witness_records", None)
    if isinstance(records, list):
        records.append(copy.deepcopy(record))
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("behavior_witness_unrunnable", producer="kernel_behavior_witness",
             event_schema=SCHEMA, clause_id=str(clause_id), reason=record["reason"],
             revision_digest=record["revision_digest"], dispatch=False)
    return record


def _binding_snapshot(runtime: object) -> object:
    binding = getattr(runtime, "binding", None)
    try:
        return copy.deepcopy(binding() if callable(binding) else binding)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def record_refused_probe(runtime: object, proposal: Mapping[str, object], *,
                         reason: str, clause_id: str, clause: str = "",
                         bolt: object = None, revision: object = None) -> dict[str, object]:
    """Settle an admission refusal as a typed, non-executed probe action.

    The refusal receipt is emitted only behind the v42 switch.  The normal
    witness record remains ``UNRUNNABLE`` and never certifies the clause; the
    action receipt exists solely to close the continuation effect and preserve
    its lineage for the next request.
    """
    record = record_unrunnable_proposal(
        runtime, proposal, reason=reason, clause_id=clause_id, clause=clause,
        revision=revision)
    if not refusal_settle_enabled():
        return record
    before = _binding_snapshot(runtime)
    lineage: dict[str, object] = {}
    continuation = getattr(runtime, "continuation", None)
    lineage_fn = getattr(continuation, "lineage", None)
    if callable(lineage_fn):
        try:
            candidate = lineage_fn()
            if isinstance(candidate, Mapping):
                lineage = dict(candidate)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            lineage = {}
    request_id = getattr(runtime, "current_request_id", None)
    if request_id is None and bolt is not None:
        request_id = getattr(bolt, "current_request_id", None)
    turn = getattr(getattr(runtime, "owner", None), "current_step", None)
    effect_id = lineage.get("continuation_effect_id")
    dispatch_id = lineage.get("continuation_dispatch_ref")
    receipt: dict[str, object] | None = None
    receipt_fn = getattr(bolt, "receipt", None)
    if callable(receipt_fn):
        material = json.dumps(dict(proposal), sort_keys=True, ensure_ascii=False, default=str)
        detail = json.dumps({
            "disposition": "REFUSED", "executed": False, "reason": str(reason)[:240],
            "request_id": request_id, "turn": turn,
            "effect_id": effect_id, "dispatch_id": dispatch_id,
            **lineage, "before_binding": before, "after_binding": before,
        }, sort_keys=True, ensure_ascii=False, default=str)
        receipt_fn("probe", str(clause_id)[:120], material, "REFUSED", detail=detail)
        rows = getattr(bolt, "_receipts", ())
        if isinstance(rows, Sequence) and rows and isinstance(rows[-1], dict):
            receipt = rows[-1]
    if receipt is None:
        record["settled"] = False
        record["settlement_reason"] = "action_receipt_unavailable"
        return record
    after = _binding_snapshot(runtime)
    receipt.update({
        "schema": "stateframe-probe-refusal-receipt/1",
        "disposition": "REFUSED", "executed": False, "reason": str(reason)[:240],
        "request_id": request_id, "turn": turn,
        "effect_id": effect_id, "dispatch_id": dispatch_id,
        "before_binding": copy.deepcopy(before), "after_binding": copy.deepcopy(after),
        "binding_unchanged": before == after, **lineage,
    })
    amend = getattr(runtime, "amend_receipt", None)
    if callable(amend):
        amend(receipt)
    record.update({
        "settled": bool(before == after), "executed": False,
        "disposition": "REFUSED", "reason": str(reason)[:240],
        "request_id": request_id, "turn": turn,
        "effect_id": effect_id, "dispatch_id": dispatch_id,
        "continuation_effect_id": effect_id,
        "continuation_dispatch_ref": dispatch_id,
        "receipt_id": receipt.get("receipt_id", ""),
        "receipt": copy.deepcopy(receipt),
        "before_binding": copy.deepcopy(before), "after_binding": copy.deepcopy(after),
    })
    records = getattr(runtime, "behavior_witness_records", None)
    if isinstance(records, list) and records:
        records[-1] = copy.deepcopy(record)
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("behavior_witness_refusal_settled", producer="kernel_behavior_witness",
             event_schema=SCHEMA, receipt_id=record["receipt_id"], clause_id=str(clause_id),
             reason=record["reason"], request_id=request_id, turn=turn,
             continuation_effect_id=effect_id, continuation_dispatch_ref=dispatch_id,
             binding_unchanged=before == after, executed=False, dispatch=False)
    return record


def _stage_uncertified(runtime: object, clause_ids: Sequence[str], *, reason: str) -> bool:
    try:
        from kernel_uncertified_submit import _green_current, mode as submit_mode, stage
        if submit_mode() != "on" or not _green_current(runtime):
            return False
        return stage(runtime, clause_ids, reason_codes=(reason,))
    except (AttributeError, ImportError, TypeError, ValueError):
        return False


def done_gate(runtime: object, *, admissible: bool, detail: str = "",
              reason_codes: Sequence[str] = (), open_ids: Sequence[str] = (),
              assessments: Sequence[object] = (),
              repair_action: Mapping[str, object] | None = None) -> dict[str, object]:
    """Apply bounded behavior evidence before any done acceptance boundary."""
    result = {
        "admissible": bool(admissible), "detail": str(detail or ""),
        "reason_codes": tuple(str(code) for code in reason_codes if str(code)),
        "repair_action": copy.deepcopy(repair_action), "changed": False,
        "requested": False, "witness_ids": (), "request_count": int(
            getattr(runtime, "_stateframe_witness_gate_requests", 0) or 0),
    }
    if all_done_gate_mode() != "on" or not enabled():
        return result
    rows = pending_clauses(runtime, open_ids, assessments)
    revision = _current_revision(runtime)
    records = getattr(runtime, "behavior_witness_records", ())
    typed_assessments = _assessments(runtime, assessments)
    typed_ids = tuple(_text(row.get("obligation_id", row.get("id"))) for row in rows)
    unrunnable_ids = tuple(
        _text(item.get("obligation_id", item.get("id")))
        for item in typed_assessments
        if _is_behavior_clause(item)
        and _text(item.get("obligation_id", item.get("id")))
        and _unrunnable_record(
            records, _text(item.get("obligation_id", item.get("id"))), revision))
    ids = tuple(dict.fromkeys((*typed_ids, *unrunnable_ids)))
    result["witness_ids"] = ids
    if not ids:
        return result
    # A refused non-green completion is already owned by another gate. Only
    # an admissible or green uncertified candidate reaches this evidence gate.
    candidate = bool(admissible)
    if not candidate:
        try:
            from kernel_uncertified_submit import _green_current
            candidate = _green_current(runtime)
        except (AttributeError, ImportError, TypeError, ValueError):
            candidate = False
    if not candidate:
        return result
    # A missing or exhausted producer is not a model-owned pending gate.  The
    # completion owner will preserve the full surface and record the refusal.
    if not _typed_probe_available(runtime):
        return result
    remaining = _probe_remaining(runtime)
    if remaining is not None and remaining <= 0:
        return result
    if (not all(
            _ENTRYPOINT_RE.fullmatch(_text(row.get("entrypoint", row.get("public_entrypoint", ""))))
            and _text(row.get("entrypoint", row.get("public_entrypoint", "")))
            != "public.module.function"
            for row in rows)
            and any(hasattr(runtime, name) for name in (
                "_completion_owner_full_tools", "_completion_owner_dispatch_tools", "completion_menu"))):
        return result
    if not rows and unrunnable_ids:
        staged = _stage_uncertified(runtime, unrunnable_ids, reason="BEHAVIOR_WITNESS_UNRUNNABLE")
        if staged:
            result["changed"] = True
            result["uncertified_staged"] = True
        else:
            result.update(
                admissible=False,
                detail=(str(detail or "") + ("; " if detail else "")
                        + "uncertified witness fallback unavailable"),
                reason_codes=tuple(dict.fromkeys((*result["reason_codes"],
                                                  "BEHAVIOR_WITNESS_FALLBACK_UNAVAILABLE"))),
                changed=True,
            )
        return result
    request_count = int(getattr(runtime, "_stateframe_witness_gate_requests", 0) or 0)
    if request_count < MAX_DONE_GATE_REQUESTS:
        request_count += 1
        runtime._stateframe_witness_gate_requests = request_count
        block = request_block(runtime, ids, assessments)
        result.update(
            admissible=False,
            detail=(str(detail or "") + ("; " if detail else "") + block),
            reason_codes=tuple(dict.fromkeys((*result["reason_codes"],
                                              "BEHAVIOR_WITNESS_REQUIRED"))),
            repair_action={"tool": "probe", "arguments": {},
                           "reason": "provide one typed public witness proposal"},
            changed=True, requested=True, request_count=request_count,
        )
        emit = getattr(getattr(runtime, "journal", None), "emit", None)
        if callable(emit):
            emit("behavior_witness_done_gate", producer="kernel_behavior_witness",
                 event_schema=SCHEMA, accepted=False, request_number=request_count,
                 clause_ids=list(ids), dispatch=False)
        return result
    staged = _stage_uncertified(runtime, ids, reason="BEHAVIOR_WITNESS_REQUEST_CAP")
    if staged:
        result.update(changed=True, uncertified_staged=True, request_count=request_count)
    else:
        result.update(
            admissible=False,
            detail=(str(detail or "") + ("; " if detail else "")
                    + "uncertified witness fallback unavailable"),
            reason_codes=tuple(dict.fromkeys((*result["reason_codes"],
                                              "BEHAVIOR_WITNESS_FALLBACK_UNAVAILABLE"))),
            repair_action={"tool": "probe", "arguments": {},
                           "reason": "uncertified witness fallback is unavailable"},
            changed=True, request_count=request_count,
        )
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("behavior_witness_done_gate_exhausted", producer="kernel_behavior_witness",
             event_schema=SCHEMA, accepted=bool(staged), request_cap=MAX_DONE_GATE_REQUESTS,
             clause_ids=list(ids), dispatch=False)
    return result


def _interface_text(interface: object) -> str:
    return interface.text if isinstance(interface, PublicInterfaceIndex) else _text(interface)


def selected_host_obligation(runtime: object, clause_id: str, clause: str = "") -> dict[str, object] | None:
    """Select the exact open host behavior row for live proposal admission."""
    candidates = [
        *tuple(getattr(runtime, "host_closure_assessments", ()) or ()),
        *tuple(getattr(runtime, "behavior_witness_assessments", ()) or ()),
        *tuple(getattr(runtime, "obligation_assessments", ()) or ()),
    ]
    for candidate in candidates:
        state = _obligation_state(candidate, changed_surface=None)
        if (state["obligation_id"] == str(clause_id)
                and state["clause"] == str(clause)
                and _is_behavior_clause(state)):
            return state
    return None


def _runtime_product_paths(runtime: object) -> tuple[str, ...]:
    binding = getattr(runtime, "binding", None)
    try:
        current = binding() if callable(binding) else binding
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        current = {}
    if not isinstance(current, Mapping):
        current = {}
    files = current.get("product_files", {})
    if not isinstance(files, Mapping) or not files:
        files = getattr(runtime, "product_files", {})
    paths = tuple(sorted({_text(path).replace("\\", "/") for path in files if _text(path)})) \
        if isinstance(files, Mapping) else ()
    if paths:
        return paths
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    target = getattr(runtime, "target", None)
    if root_value is None or target is None:
        return ()
    try:
        relative = Path(target).resolve().relative_to(Path(root_value).resolve()).as_posix()
    except (OSError, TypeError, ValueError):
        return ()
    return (relative,) if relative.endswith(".py") else ()


def _public_product_path(value: str) -> bool:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or path.suffix.lower() not in _PUBLIC_SUFFIXES or not path.parts):
        return False
    for part in path.parts[:-1]:
        lower = part.lower()
        if (part.startswith(".") or part.startswith("_")
                or lower in _PRIVATE_PATH_PARTS or lower in {"tests", "test"}
                or lower.startswith("test_") or lower.endswith("_test")):
            return False
    filename = path.name
    if filename not in _PUBLIC_INIT_FILES and filename.startswith("_"):
        return False
    lower_name = filename.lower()
    return not (lower_name in {"test.py", "conftest.py"}
                or lower_name.startswith("test_") or lower_name.endswith("_test"))


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).parts)
    if not parts:
        return ""
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        suffix = PurePosixPath(parts[-1]).suffix
        parts[-1] = parts[-1][:-len(suffix)] if suffix else parts[-1]
    return ".".join(parts)


def _generic_interface(source: str, suffix: str) -> tuple[frozenset[str], dict[str, _ClassInterface]]:
    """Recognize public source units without claiming they are executable here."""
    function_patterns = {
        ".js": r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)",
        ".jsx": r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)",
        ".ts": r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)",
        ".tsx": r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)",
        ".go": r"\bfunc\s+([A-Z_]\w*)",
        ".rs": r"\bpub\s+(?:async\s+)?fn\s+([A-Za-z_]\w*)",
        ".java": r"\bpublic\s+(?:static\s+)?[\w<>\[\]]+\s+([A-Za-z_]\w*)\s*\(",
        ".php": r"\bfunction\s+([A-Za-z_]\w*)\s*\(",
        ".vue": r"\b(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)",
    }
    class_pattern = r"\b(?:export\s+)?(?:class|struct)\s+([A-Za-z_]\w*)"
    functions = frozenset(re.findall(function_patterns.get(suffix.lower(), "(?!)"), source))
    classes = {name: _ClassInterface({}, False)
               for name in re.findall(class_pattern, source)}
    return functions, classes


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _required_positional_arguments(function: ast.FunctionDef | ast.AsyncFunctionDef,
                                    *, drop_first: bool) -> bool:
    arguments = [*function.args.posonlyargs, *function.args.args]
    if drop_first and arguments:
        arguments = arguments[1:]
    defaults = [None] * (len(arguments) - len(function.args.defaults))
    defaults.extend(function.args.defaults)
    return any(default is None for default in defaults)


def _method_kind(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    decorators = {_dotted_name(item) for item in function.decorator_list}
    if "classmethod" in decorators:
        return "classmethod"
    if "staticmethod" in decorators:
        return "staticmethod"
    return "instance_method"


def _class_interface(node: ast.ClassDef) -> _ClassInterface:
    methods: dict[str, tuple[str, bool]] = {}
    constructor_requires_input = False
    for child in node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if child.name == "__init__":
            constructor_requires_input = _required_positional_arguments(child, drop_first=True)
        elif child.name != "__del__":
            methods[child.name] = (
                _method_kind(child), _required_positional_arguments(child, drop_first=True))
    return _ClassInterface(methods, constructor_requires_input)


def _is_cli_module(tree: ast.Module, path: str) -> bool:
    if PurePosixPath(path).name == "__main__.py":
        return True
    has_argparse = any(
        (isinstance(node, ast.Import) and any(alias.name == "argparse" for alias in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "argparse")
        for node in ast.walk(tree)
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        if len(node.test.ops) != 1 or not isinstance(node.test.ops[0], ast.Eq):
            continue
        values = [node.test.left, *node.test.comparators]
        if any(isinstance(item, ast.Name) and item.id == "__name__" for item in values) and any(
                isinstance(item, ast.Constant) and item.value == "__main__" for item in values):
            return True
    return has_argparse


def build_public_interface_index(runtime: object) -> PublicInterfaceIndex | None:
    """Build an AST-only index from the run's current bound product files."""
    paths = _runtime_product_paths(runtime)
    if not any(PurePosixPath(path).suffix.lower() in _PUBLIC_SUFFIXES for path in paths):
        return None
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None) or getattr(runtime, "workspace_root", None)
    try:
        root = Path(root_value) if root_value else None
    except (TypeError, ValueError):
        root = None
    modules: dict[str, _ModuleInterface] = {}
    cli_modules: set[str] = set()
    source_paths: list[str] = []
    parse_errors: dict[str, str] = {}
    source_units: dict[str, Mapping[str, object]] = {}
    text_parts: list[str] = []
    for relative in paths:
        if not _public_product_path(relative):
            continue
        source = None
        if root is not None:
            candidate = root / PurePosixPath(relative)
            try:
                if not candidate.is_symlink() and candidate.is_file():
                    source = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError, TypeError, ValueError):
                source = None
        if source is None:
            continue
        source_paths.append(relative)
        text_parts.append(f"# public source: {relative}\n{source}")
        try:
            from kernel_aha10_honest_loop import source_units as build_source_units
            module_alias = _module_name(relative)
            for unit in build_source_units(relative, source, revision=_json_digest(source)):
                alias = (f"{module_alias}.{unit.name}" if module_alias and unit.name != "<module>"
                         else f"{module_alias}.<module>" if module_alias else unit.name)
                unit_record = unit.as_dict()
                if root is not None:
                    unit_record["load_path"] = str(root / PurePosixPath(relative))
                source_units[alias] = copy.deepcopy(unit_record)
                source_units[unit.handle] = copy.deepcopy(unit_record)
        except (ImportError, OSError, SyntaxError, TypeError, ValueError):
            # Public API indexing remains available when a source adapter is
            # unavailable; that unit is then an explicit unsupported outcome.
            pass
        suffix = PurePosixPath(relative).suffix.lower()
        tree: ast.Module | None = None
        if suffix == ".py":
            try:
                tree = ast.parse(source, filename=relative)
            except SyntaxError as error:
                parse_errors[relative] = str(error)
                continue
        if suffix == ".py":
            functions = frozenset(
                node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
            classes = {
                node.name: _class_interface(node)
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name != "__all__"
            }
            language = "python"
        else:
            functions, classes = _generic_interface(source, suffix)
            language = suffix.removeprefix(".")
        module = _module_name(relative)
        if not module:
            continue
        modules[module] = _ModuleInterface(
            functions, classes, _is_cli_module(tree, relative) if tree is not None else False,
            language=language)
        if modules[module].cli:
            cli_modules.add(module)
    return PublicInterfaceIndex(
        modules=modules, cli_modules=frozenset(cli_modules),
        source_paths=tuple(source_paths), text="\n".join(text_parts),
        parse_errors=parse_errors, source_units=source_units,
    )


def context_for_runtime(runtime: object, row: object, clause_id: str) -> tuple[str, object, str]:
    """Return host-retained brief/interface text and the selected clause."""
    source = row if isinstance(row, Mapping) else {}
    owner = getattr(runtime, "owner", None)
    task_row = getattr(owner, "task_row", {})
    task_row = task_row if isinstance(task_row, Mapping) else {}
    brief = next((_text(getattr(runtime, name, "")) for name in
                  ("behavior_witness_brief", "public_brief")
                  if _text(getattr(runtime, name, ""))), "")
    if not brief:
        brief = next((_text(task_row.get(name)) for name in
                      ("public_brief", "brief", "msg", "_bridge_ordinary_brief")
                      if _text(task_row.get(name))), "")
    interface: object = build_public_interface_index(runtime)
    if interface is None:
        interface = next((_text(getattr(runtime, name, "")) for name in
                          ("behavior_witness_interface", "public_interface")
                          if _text(getattr(runtime, name, ""))), "")
        if not interface:
            values = [task_row.get(name) for name in
                      ("public_interface", "interface", "public_source", "source", "public_tests", "files")]
            interface = "\n".join(
                json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
                if isinstance(item, (Mapping, list, tuple)) else _text(item)
                for item in values if item)
    clause = _text(source.get("clause", ""))
    if not clause:
        clause = next((_text(item.get("clause")) for item in _assessments(runtime)
                       if _text(item.get("obligation_id", item.get("id"))) == str(clause_id)), "")
    return brief, interface, clause


def _text_entrypoint_is_public(entrypoint: str, interface: str) -> bool:
    if re.search(rf"(?<![A-Za-z0-9_.]){re.escape(entrypoint)}(?![A-Za-z0-9_.])", interface):
        return True
    module, function = entrypoint.rsplit(".", 1)
    path = module.replace(".", "/") + ".py"
    if (re.search(rf"(?<![A-Za-z0-9_]){re.escape(path)}(?![A-Za-z0-9_])", interface) is not None
            and re.search(rf"\bdef\s+{re.escape(function)}\b", interface) is not None):
        return True
    # `module.Class.method`: the module path is everything before the class segment.
    # Live v41 batch (meter) rejected `app.models.MeterReading.from_payload` because only
    # `module.function` was recognised.
    if "." in module:
        owner_module, owner = module.rsplit(".", 1)
        owner_path = owner_module.replace(".", "/") + ".py"
        return (re.search(rf"(?<![A-Za-z0-9_]){re.escape(owner_path)}(?![A-Za-z0-9_])", interface)
                is not None
                and re.search(rf"\bclass\s+{re.escape(owner)}\b", interface) is not None
                and re.search(rf"\bdef\s+{re.escape(function)}\b", interface) is not None)
    return False


def _private_entrypoint_segment(segment: str) -> bool:
    lower = segment.lower()
    return (segment.startswith("_") and segment != "__main__"
            or lower in _PRIVATE_PATH_PARTS or lower in {"tests", "test"}
            or lower.startswith("test_") or lower.endswith("_test"))


def _entrypoint_match(entrypoint: str, index: PublicInterfaceIndex) -> tuple[_EntrypointMatch | None, str, str]:
    parts = entrypoint.split(".")
    # Source provenance and API spelling are separate dimensions.  A public
    # module may expose an underscore-named implementation callable; private
    # modules, test paths, and private owners remain refused.  Only the final
    # callable segment may be underscore-prefixed.
    if any(_private_entrypoint_segment(part) for part in parts[:-1]):
        return None, "entrypoint_private", "private or test path/name is not public"
    if entrypoint in index.cli_modules:
        return _EntrypointMatch("cli", entrypoint), "", ""
    if len(parts) < 2:
        return None, "entrypoint_unknown_module", "entrypoint has no public module"
    top_module = ".".join(parts[:-1])
    attribute = parts[-1]
    top = index.modules.get(top_module)
    if top is not None:
        if attribute in top.functions:
            kind = "function" if top.language == "python" else "unsupported"
            return _EntrypointMatch(kind, top_module, attribute=attribute), "", ""
        if attribute in top.classes:
            class_info = top.classes[attribute]
            kind = "constructor" if top.language == "python" else "unsupported"
            return _EntrypointMatch(
                kind, top_module, owner=attribute,
                constructor_requires_input=class_info.constructor_requires_input,
            ), "", ""
        return None, "entrypoint_unknown_attribute", f"unknown attribute {attribute!r} on module {top_module!r}"
    if len(parts) >= 3:
        owner_module = ".".join(parts[:-2])
        owner = parts[-2]
        method = parts[-1]
        owner_module_info = index.modules.get(owner_module)
        owner_class = owner_module_info.classes.get(owner) if owner_module_info else None
        if owner_class is not None:
            method_info = owner_class.methods.get(method)
            if method_info is not None:
                method_kind, method_requires_input = method_info
                kind = method_kind if owner_module_info.language == "python" else "unsupported"
                return _EntrypointMatch(
                    kind, owner_module, owner=owner, attribute=method,
                    constructor_requires_input=owner_class.constructor_requires_input,
                    method_requires_input=method_requires_input,
                ), "", ""
            return None, "entrypoint_unknown_attribute", f"unknown method {method!r} on class {owner!r}"
        if owner_module_info is not None:
            return None, "entrypoint_unknown_module", f"unknown module {top_module!r}"
    return None, "entrypoint_unknown_module", f"unknown module {top_module!r}"


def source_unit_match(entrypoint: str, index: PublicInterfaceIndex,
                      handle: object = None) -> tuple[_EntrypointMatch | None, str, str]:
    """Resolve source-visible units only with an explicit opaque handle."""
    if not handle:
        return None, "source_unit_handle_required", "private/nested source units require an opaque handle"
    row = index.source_units.get(str(handle))
    if not isinstance(row, Mapping):
        return None, "source_unit_unavailable", "no stable source-unit handle is bound"
    if handle and str(row.get("handle")) != str(handle):
        return None, "source_unit_handle_mismatch", "source-unit handle does not match the index"
    if str(row.get("status", "READY")) != "READY":
        return None, "source_unit_unrunnable", str(row.get("status"))
    # The handle is the authority.  The label may be a module, private member,
    # nested selector, or a non-Python worker identity such as
    # ``book-worker::process_message``; dotted package spelling is not required.
    terminal = re.split(r"[.#:/]+", entrypoint)[-1]
    unit_name = str(row.get("name", ""))
    unit_terminal = re.split(r"[.#:/]+", unit_name)[-1]
    identities = {
        str(row.get("path", "")), str(row.get("qualified_name", "")),
        unit_name, unit_terminal,
    }
    if not ({entrypoint, terminal} & identities):
        return None, "source_unit_identity_mismatch", "entrypoint does not name the bound source unit"
    # The opaque row owns loading.  ``entrypoint`` is only a selector label;
    # dotted package spelling must never become source authority.
    module = str(row.get("path", ""))
    return _EntrypointMatch("source_unit", module, attribute=unit_name), "", ""


def _entrypoint_admission(entrypoint: str, interface: object) -> tuple[_EntrypointMatch | None, str, str]:
    if isinstance(interface, PublicInterfaceIndex):
        return _entrypoint_match(entrypoint, interface)
    if _text_entrypoint_is_public(entrypoint, _interface_text(interface)):
        module, attribute = entrypoint.rsplit(".", 1)
        return _EntrypointMatch("function", module, attribute=attribute), "", ""
    return None, "entrypoint_not_in_public_interface", "text interface rule did not match entrypoint"


def _entrypoint_is_public(entrypoint: str, interface: object) -> bool:
    match, reason, _detail = _entrypoint_admission(entrypoint, interface)
    return match is not None and not reason


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return _assessment_dict(value)


def _obligation_state(value: object, *, changed_surface: object = None) -> dict[str, object]:
    state = _as_mapping(value)
    if changed_surface is not None:
        state["changed_surface"] = changed_surface
    checks = state.get("check_ids", state.get("checks", ()))
    if isinstance(checks, str):
        checks = (checks,) if checks else ()
    elif not isinstance(checks, Sequence) or isinstance(checks, (bytes, bytearray)):
        checks = ()
    reason = _text(state.get("reason", state.get("reason_code", ""))).upper()
    status = _text(state.get("status", state.get("verdict", ""))).upper()
    return {
        "obligation_id": _text(state.get("obligation_id", state.get("id", ""))),
        "clause": _text(state.get("clause", state.get("exact_clause", ""))),
        "reason": reason,
        "status": status,
        "check_ids": tuple(str(item) for item in checks if str(item)),
        "changed_surface": _changed_surface(state),
    }


def _select_obligation(*, clause_id: str, clause: str, obligation: object,
                       open_obligation: object, open_obligations: Sequence[object],
                       changed_surface: object) -> dict[str, object] | None:
    candidates: list[object] = []
    for item in (obligation, open_obligation):
        if item is not None:
            candidates.append(item)
    candidates.extend(open_obligations)
    for candidate in candidates:
        state = _obligation_state(candidate, changed_surface=changed_surface)
        if (state["obligation_id"] == str(clause_id)
                and state["clause"] == str(clause)
                and _is_behavior_clause(state)):
            return state
    return None


def _json_value(value: object) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _json_source(value: object) -> str:
    """Embed JSON data as data, never as Python source syntax."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return "json.loads(" + repr(encoded) + ")"


_FIXTURE_HELPER = '''
def _stateframe_record(value):
    if isinstance(value, dict):
        record = value.get("$record")
        if isinstance(record, dict) and len(value) == 1:
            return types.SimpleNamespace(**{
                str(key): _stateframe_record(item) for key, item in record.items()
            })
        return {key: _stateframe_record(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stateframe_record(item) for item in value]
    return value
'''


def _fixture_source(value: object) -> str:
    """Create only typed data fixtures; never interpolate model source."""
    return "_stateframe_record(" + _json_source(value) + ")"


def _json_args(value: object) -> tuple[object, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("args must be a JSON array")
    return tuple(copy.deepcopy(value))


def _json_kwargs(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("kwargs must be a JSON object")
    return copy.deepcopy(dict(value))


def _literal_provenance(value: object, *, label: str, brief: str, interface: object,
                        rationale: str,
                        public_partition_literals: Sequence[object] = ()) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    model_derived: list[dict[str, object]] = []
    for path, literal in _literal_rows(value, label):
        source = ""
        if _literal_in(brief, literal):
            source = "public_brief"
        elif _literal_in(_interface_text(interface), literal):
            source = "public_interface"
        elif literal in public_partition_literals:
            source = "public_partition"
        elif _rationale_shows_arithmetic(rationale, literal):
            source = "model-derived"
        if not source:
            raise ValueError(f"literal at {path} is not traceable to public material or explicit model arithmetic")
        row = {"path": path, "literal": copy.deepcopy(literal), "source": source}
        rows.append(row)
        if source == "model-derived":
            model_derived.append(row)
    return rows, model_derived


def admit_proposal(proposal: Mapping[str, object], *, clause_id: str, clause: str,
                   brief: str, interface: object, revision: object,
                   obligation: object = None, open_obligation: object = None,
                   open_obligations: Sequence[object] = (), changed_surface: object = None,
                   journal: object = None,
                   public_partition_literals: Sequence[object] = ()) -> dict[str, object]:
    """Admit a model case from typed public provenance, never a host oracle."""
    required = ("clause_id", "entrypoint", "expected", "rationale")
    missing = [name for name in required if name not in proposal]
    if "input" not in proposal and "args" not in proposal and "kwargs" not in proposal:
        missing.append("input_or_args_kwargs")
    base = {"schema": SCHEMA, "admitted": False, "clause_id": str(clause_id),
            "revision_count": 0}
    if missing:
        return {**base, "reason": "typed_fields_missing", "missing": missing}
    if _text(proposal.get("clause_id")) != str(clause_id):
        return {**base, "reason": "clause_id_mismatch"}
    entrypoint = _text(proposal.get("entrypoint"))
    source_handle = _text(proposal.get("source_unit_handle"))
    if not ((_ENTRYPOINT_RE.fullmatch(entrypoint)) or
            (source_handle and 0 < len(entrypoint) <= 240
             and not any(ord(char) < 32 for char in entrypoint))):
        return {**base, "reason": "entrypoint_not_public_shape"}
    if not clause.strip() or clause not in brief:
        return {**base, "reason": "clause_not_in_public_brief"}
    if source_handle:
        if not isinstance(interface, PublicInterfaceIndex):
            return {**base, "reason": "source_unit_index_required",
                    "detail": "opaque source-unit handles require the host source index"}
        entrypoint_match, entrypoint_reason, entrypoint_detail = source_unit_match(
            entrypoint, interface, source_handle)
    else:
        entrypoint_match, entrypoint_reason, entrypoint_detail = _entrypoint_admission(entrypoint, interface)
    if (entrypoint_match is None and entrypoint_reason == "entrypoint_private"
            and isinstance(interface, PublicInterfaceIndex)):
        entrypoint_match, entrypoint_reason, entrypoint_detail = source_unit_match(
            entrypoint, interface, proposal.get("source_unit_handle"),
        )
    if entrypoint_match is None:
        return {**base, "reason": entrypoint_reason, "detail": entrypoint_detail,
                "legacy_reason": "entrypoint_not_in_public_interface"}
    explicit_args = proposal.get("args") if "args" in proposal else None
    explicit_kwargs = proposal.get("kwargs") if "kwargs" in proposal else None
    try:
        call_args = _json_args(explicit_args)
        call_kwargs = _json_kwargs(explicit_kwargs)
    except ValueError as error:
        return {**base, "reason": "args_kwargs_must_be_json", "detail": str(error)}
    proposal_input = proposal.get("input")
    if proposal_input is None and (call_args is not None or call_kwargs is not None):
        proposal_input = {
            "args": list(call_args or ()),
            "kwargs": dict(call_kwargs or {}),
        }
    expected = proposal.get("expected")
    rationale = _text(proposal.get("rationale"))
    if not rationale.strip():
        return {**base, "reason": "rationale_required"}
    probe_code = _text(proposal.get("code"))[:12000]
    if re.search(r"\bassert\s+True\b|\breturn\s+True\b", probe_code):
        return {**base, "reason": "vacuous_probe"}
    if re.search(r"(?:inspect\.getsource|open\s*\(|read_text\s*\(|ast\.parse)", probe_code):
        return {**base, "reason": "source_clone_probe"}
    if proposal.get("expected_mutable") is True or (
            isinstance(expected, Mapping) and expected.get("$mutable") is True):
        return {**base, "reason": "mutable_expectation"}
    if proposal_input in ({}, [], None) and expected in ({}, [], None, True):
        return {**base, "reason": "vacuous_input"}
    if (not _json_value(proposal_input) or not _json_value(expected)
            or (call_args is not None and not _json_value(call_args))
            or (call_kwargs is not None and not _json_value(call_kwargs))):
        return {**base, "reason": "input_expected_must_be_json"}
    selected = _select_obligation(
        clause_id=str(clause_id), clause=clause, obligation=obligation,
        open_obligation=open_obligation, open_obligations=open_obligations,
        changed_surface=changed_surface)
    if selected is None:
        return {**base, "reason": "obligation_not_open_or_changed"}
    try:
        if call_args is not None or call_kwargs is not None:
            call_literals, call_model_derived = [], []
            for label, values in (("args", list(call_args or ())),
                                  ("kwargs", list((call_kwargs or {}).values()))):
                literals, derived = _literal_provenance(
                    values, label=label, brief=brief, interface=interface, rationale=rationale,
                    public_partition_literals=public_partition_literals)
                call_literals.extend(literals)
                call_model_derived.extend(derived)
            input_literals, input_model_derived = [], []
        else:
            call_literals, call_model_derived = [], []
            input_literals, input_model_derived = _literal_provenance(
                proposal_input, label="input", brief=brief, interface=interface, rationale=rationale,
                public_partition_literals=public_partition_literals)
        expected_literals, expected_model_derived = _literal_provenance(
            expected, label="expected", brief=brief, interface=interface, rationale=rationale)
    except (TypeError, ValueError) as error:
        return {**base, "reason": "literal_provenance_missing", "detail": str(error)[:240]}
    model_derived = [*input_model_derived, *expected_model_derived]
    provenance = {
        "brief_digest": _json_digest(brief), "interface_digest": _json_digest(_interface_text(interface)),
        "clause_digest": _json_digest(clause), "clause_source": "public_brief",
        "interface_source": "public_interface", "entrypoint": entrypoint,
        "input_literals": input_literals, "call_literals": call_literals,
        "expected_literals": expected_literals,
        "model_derived_literals": [*model_derived, *call_model_derived],
        "rationale_digest": _json_digest(rationale), "obligation": selected,
    }
    source_identity = source_handle or (
        str(entrypoint_match.module) + ":" + str(entrypoint_match.attribute or "")
        + ":" + _json_digest(_interface_text(interface))[:16]
    )
    loader_capability = (
        "source-unit" if entrypoint_match.kind == "source_unit"
        else "public-import" if entrypoint_match.kind == "function"
        else "unsupported"
    )
    result = {**base, "admitted": True, "entrypoint": entrypoint,
              "entrypoint_kind": entrypoint_match.kind,
              "entrypoint_module": entrypoint_match.module,
              "entrypoint_owner": entrypoint_match.owner,
              "entrypoint_attribute": entrypoint_match.attribute,
              "constructor_requires_input": entrypoint_match.constructor_requires_input,
              "method_requires_input": entrypoint_match.method_requires_input,
              "source_unit_handle": (
                  _text(interface.source_units.get(str(proposal.get("source_unit_handle")), {}).get("handle"))
                  if entrypoint_match.kind == "source_unit" and isinstance(interface, PublicInterfaceIndex)
                  else ""
              ),
              "source_unit": (
                  copy.deepcopy(dict(interface.source_units.get(str(proposal.get("source_unit_handle")), {})))
                  if entrypoint_match.kind == "source_unit" and isinstance(interface, PublicInterfaceIndex)
                  else {}
              ),
              "source_identity": source_identity,
              "loader_capability": loader_capability,
               "input": copy.deepcopy(proposal_input), "expected": copy.deepcopy(expected),
               "args": list(call_args) if call_args is not None else None,
               "kwargs": dict(call_kwargs) if call_kwargs is not None else None,
               "uses_typed_call": call_args is not None or call_kwargs is not None,
               "observation_projection": copy.deepcopy(proposal.get("projection")),
              "rationale": rationale[:1000],
              "clause": clause, "revision": copy.deepcopy(revision),
              "revision_digest": _json_digest(revision), "provenance": provenance,
              "partition": "model-proposed-public-case",
              "proposal_digest": _json_digest(proposal)}
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("behavior_witness_admitted", producer="kernel_behavior_witness",
             event_schema=SCHEMA, clause_id=str(clause_id), provenance=provenance,
             partition=result["partition"], dispatch=False)
    return result


def admit_revised_proposal(proposal: Mapping[str, object], *, original: Mapping[str, object],
                           clause_id: str, clause: str, brief: str, interface: object,
                           revision: object, obligation: object = None,
                           open_obligation: object = None,
                           open_obligations: Sequence[object] = (),
                           changed_surface: object = None, journal: object = None) -> dict[str, object]:
    """Allow one model revision after a mismatch, preserving original evidence."""
    original_reason = _text(original.get("reason"))
    is_failed_case = _text(original.get("result")).upper() == "FAIL"
    is_rejected_entrypoint = (original.get("admitted") is False
                              and original_reason in _REVISABLE_ADMISSION_REASONS)
    if not is_failed_case and not is_rejected_entrypoint:
        return {"schema": SCHEMA, "admitted": False, "clause_id": str(clause_id),
                "reason": "revision_requires_failed_case"}
    revision_count = int(original.get("revision_count", 0) or 0)
    if revision_count >= MAX_REVISED_PROPOSALS:
        return {"schema": SCHEMA, "admitted": False, "clause_id": str(clause_id),
                "reason": "revision_limit_reached"}
    result = admit_proposal(
        proposal, clause_id=clause_id, clause=clause, brief=brief, interface=interface,
        revision=revision, obligation=obligation, open_obligation=open_obligation,
        open_obligations=open_obligations, changed_surface=changed_surface, journal=journal)
    result["revision_count"] = revision_count + 1
    result["revision_of"] = _text(original.get("witness_id", original.get("proposal_digest")))
    return result


def _legacy_probe_source(admission: Mapping[str, object]) -> str:
    """Build a probe that calls the public entrypoint and reports its observation."""
    entrypoint = _text(admission.get("entrypoint"))
    module, function = entrypoint.rsplit(".", 1)
    expected = _json_source(admission.get("expected", {}))
    fixture = _fixture_source(admission.get("input", {}))
    if admission.get("uses_typed_call"):
        call = "_public_entry(*_args, **_kwargs)"
        call_setup = [
            f"_args = {_json_source(admission.get('args') or [])}",
            f"_kwargs = {_json_source(admission.get('kwargs') or {})}",
        ]
    else:
        call = "_public_entry(_fixture)"
        call_setup = []
    return (f"from {module} import {function} as _public_entry\n"
            "import asyncio\n"
            "import inspect\n"
            "import json\n"
            "import types\n"
            + _FIXTURE_HELPER
            + f"_fixture = {fixture}\n"
            + "\n".join(call_setup) + ("\n" if call_setup else "")
            + f"_observed = {call}\n"
            "if inspect.isawaitable(_observed):\n"
            "    _observed = asyncio.run(_observed)\n"
            f"_expected = {expected}\n"
            "_value = _observed\n"
            "_projection = " + _json_source(admission.get("observation_projection")) + "\n"
            "if _projection == 'body' and isinstance(_observed, dict):\n"
            "    _value = _observed.get('body')\n"
            f"print({_OBSERVED_MARKER!r} + json.dumps(_value, sort_keys=True, ensure_ascii=False, default=str))\n"
             "assert _value == _expected, "
             "repr({'expected': _expected, 'observed': _value})\n")


def _source_unit_probe_source(admission: Mapping[str, object]) -> str:
    """Load a bound Python unit by opaque path, never by package spelling."""
    unit = admission.get("source_unit")
    unit = unit if isinstance(unit, Mapping) else {}
    language = _text(unit.get("language")).casefold()
    if language != "python":
        return "raise RuntimeError('UNRUNNABLE: language adapter unavailable')\n"
    load_path = _text(unit.get("load_path", unit.get("path")))
    name = _text(unit.get("name"))
    kind = _text(unit.get("kind"))
    fixture = _fixture_source(admission.get("input", {}))
    expected = _json_source(admission.get("expected", {}))
    if admission.get("uses_typed_call"):
        call_setup = (
            f"_args = {_json_source(admission.get('args') or [])}\n"
            f"_kwargs = {_json_source(admission.get('kwargs') or {})}\n"
        )
        invocation = "_target(*_args, **_kwargs)"
    else:
        call_setup = ""
        invocation = "_target(_fixture)"
    return (
        "import asyncio\nimport importlib.util\nimport inspect\nimport json\n"
        f"_spec = importlib.util.spec_from_file_location('_stateframe_bound_unit', {load_path!r})\n"
        "if _spec is None or _spec.loader is None:\n"
        "    raise RuntimeError('UNRUNNABLE: source loader unavailable')\n"
        "_module = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_module)\n"
        f"_fixture = {fixture}\n"
        f"_expected = {expected}\n"
        f"_unit_kind = {kind!r}\n"
        f"_unit_name = {name!r}\n"
        "if _unit_kind in {'file', 'module_initialization'}:\n"
        "    _value = {'loaded': True, 'unit': _unit_name}\n"
        "else:\n"
        "    _target = _module\n"
        "    for _part in _unit_name.split('.'):\n"
        "        _target = getattr(_target, _part)\n"
        f"{call_setup}"
        f"    _value = {invocation}\n"
        "    if inspect.isawaitable(_value):\n"
        "        _value = asyncio.run(_value)\n"
        f"print({_OBSERVED_MARKER!r} + json.dumps(_value, sort_keys=True, ensure_ascii=False, default=str))\n"
        "assert _value == _expected, repr({'expected': _expected, 'observed': _value})\n"
    )


def probe_source(admission: Mapping[str, object]) -> str:
    """Build an execution probe for a function, class, method, or public CLI."""
    entrypoint = _text(admission.get("entrypoint"))
    kind = _text(admission.get("entrypoint_kind"))
    expected = _json_source(admission.get("expected", {}))
    fixture = _fixture_source(admission.get("input", {}))
    if kind == "source_unit":
        return _source_unit_probe_source(admission)
    lines = ["import asyncio", "import inspect", "import json", "import types",
             _FIXTURE_HELPER, f"_fixture = {fixture}"]
    if kind == "cli":
        lines.extend([
            "import runpy",
            "import sys",
            "_argv = _fixture.get('argv', []) if isinstance(_fixture, dict) else _fixture",
            f"sys.argv = [{entrypoint!r}] + [str(item) for item in (_argv or [])]",
            f"_observed = runpy.run_module({entrypoint!r}, run_name='__main__')",
        ])
    elif kind == "constructor":
        module = _text(admission.get("entrypoint_module")) or entrypoint.rsplit(".", 1)[0]
        owner = _text(admission.get("entrypoint_owner")) or entrypoint.rsplit(".", 1)[1]
        if admission.get("uses_typed_call"):
            lines.extend([
                f"_args = {_json_source(admission.get('args') or [])}",
                f"_kwargs = {_json_source(admission.get('kwargs') or {})}",
            ])
            constructor_call = "_public_entry(*_args, **_kwargs)"
        else:
            constructor_call = "_public_entry(_fixture)" if admission.get("constructor_requires_input", True) else "_public_entry()"
        lines.extend([f"from {module} import {owner} as _public_entry", f"_observed = {constructor_call}"])
    elif kind in {"classmethod", "staticmethod"}:
        module = _text(admission.get("entrypoint_module"))
        owner = _text(admission.get("entrypoint_owner"))
        method = _text(admission.get("entrypoint_attribute"))
        call = ("_public_owner." + method + "(*_args, **_kwargs)"
                if admission.get("uses_typed_call") else
                "_public_owner." + method + "(_fixture)")
        if admission.get("uses_typed_call"):
            lines.extend([
                f"_args = {_json_source(admission.get('args') or [])}",
                f"_kwargs = {_json_source(admission.get('kwargs') or {})}",
            ])
        lines.extend([
            f"from {module} import {owner} as _public_owner",
            f"_observed = {call}",
        ])
    elif kind == "instance_method":
        module = _text(admission.get("entrypoint_module"))
        owner = _text(admission.get("entrypoint_owner"))
        method = _text(admission.get("entrypoint_attribute"))
        constructor_requires_input = bool(admission.get("constructor_requires_input"))
        method_requires_input = bool(admission.get("method_requires_input"))
        if admission.get("uses_typed_call"):
            lines.extend([
                f"_args = {_json_source(admission.get('args') or [])}",
                f"_kwargs = {_json_source(admission.get('kwargs') or {})}",
            ])
            constructor = "_public_owner()"
            call = f"_public_instance.{method}(*_args, **_kwargs)"
        else:
            constructor = "_public_owner(_fixture)" if constructor_requires_input else "_public_owner()"
            call = f"_public_instance.{method}(_fixture)" if method_requires_input else f"_public_instance.{method}()"
        lines.extend([
            f"from {module} import {owner} as _public_owner",
            f"_public_instance = {constructor}",
            f"_observed = {call}",
        ])
    else:
        return _legacy_probe_source(admission)
    lines.extend([
        "if inspect.isawaitable(_observed):",
        "    _observed = asyncio.run(_observed)",
        f"_expected = {expected}",
        "_value = _observed",
        "_projection = " + _json_source(admission.get("observation_projection")),
        "if _projection == 'body' and isinstance(_observed, dict):",
        "    _value = _observed.get('body')",
        "if not isinstance(_value, (dict, list, tuple, str, int, float, bool, type(None))) and hasattr(_value, '__dict__'):",
        "    _value = vars(_value)",
        f"print({_OBSERVED_MARKER!r} + json.dumps(_value, sort_keys=True, ensure_ascii=False, default=str))",
        "assert _value == _expected, repr({'expected': _expected, 'observed': _value})",
    ])
    return "\n".join(lines) + "\n"


def _observed_from_receipt(receipt: object) -> object:
    direct = _field(receipt, "observed", None)
    if direct is not None:
        return copy.deepcopy(direct)
    output = _field(receipt, "shown_output", _field(receipt, "output", ""))
    for line in _text(output).splitlines():
        if line.startswith(_OBSERVED_MARKER):
            try:
                return json.loads(line[len(_OBSERVED_MARKER):])
            except (TypeError, ValueError):
                return line[len(_OBSERVED_MARKER):]
    return _text(output)


def _observation_bridge(receipt: object, admission: Mapping[str, object]) -> dict[str, object] | None:
    """Accept only a host trace tied to public source loading.

    A serialized return value is useful diagnostics, but it is not by itself a
    behavior witness.  Adapters must provide source identity, loader/fixture
    capability, and a non-empty event or side-effect trace.
    """
    raw = _field(receipt, "observation_trace", _field(receipt, "evidence_bridge", None))
    if not isinstance(raw, Mapping):
        return None
    source_identity = raw.get("source_identity", raw.get("source_handle"))
    loader = raw.get("loader_capability", raw.get("fixture_capability"))
    trace = raw.get("events", raw.get("side_effects", raw.get("trace")))
    if source_identity in (None, "") or loader in (None, ""):
        return None
    if not isinstance(trace, Mapping) and not (
            isinstance(trace, Sequence) and not isinstance(trace, (str, bytes, bytearray))
            and len(trace) > 0):
        return None
    expected_source = admission.get("source_identity")
    if expected_source not in (None, "") and str(source_identity) != str(expected_source):
        return None
    return copy.deepcopy(dict(raw))


def bind_probe_result(runtime: object, admission: Mapping[str, object], receipt: object,
                      *, observed: object = _UNSET) -> dict[str, object]:
    status = _text(_field(receipt, "status")).upper()
    exit_code = _field(receipt, "exit_code")
    receipt_body = _field(receipt, "body", _field(receipt, "response_body", None))
    full_observed = observed if observed is not _UNSET else _observed_from_receipt(receipt)
    bridge = _observation_bridge(receipt, admission)
    bridge_observed = (bridge.get("observed", bridge.get("value", full_observed))
                       if bridge is not None else full_observed)
    observation_matches = bridge is not None and bridge_observed == admission.get("expected")
    receipt_revision = _text(_field(receipt, "revision"))
    admission_revision = admission.get("revision")
    revision_matches = not receipt_revision or not isinstance(admission_revision, str) \
        or receipt_revision == admission_revision
    passed = (status in _PASS and (exit_code in (None, 0)) and observation_matches
              and revision_matches)
    unsupported = not passed and status in _PASS and bridge is None
    result = "PASS" if passed else "UNRUNNABLE" if unsupported else "FAIL"
    record = {"schema": SCHEMA, "witness_id": _text(_field(receipt, "receipt_id")),
              "clause_id": _text(admission.get("clause_id")), "admitted": True,
              "clause": _text(admission.get("clause")),
              "entrypoint": _text(admission.get("entrypoint")),
              "result": result, "revision_digest": admission.get("revision_digest"),
               "input": copy.deepcopy(admission.get("input")),
               "expected": copy.deepcopy(admission.get("expected")),
               "observed": (observed if observed is not _UNSET
                             else _field(receipt, "shown_output", "")),
               "observed_full": copy.deepcopy(full_observed),
               "receipt_status": status,
               "exit_code": exit_code,
               "receipt_body": copy.deepcopy(receipt_body),
               "observation_bridge": bridge,
               "observation_projection": copy.deepcopy(admission.get("observation_projection")),
               "call_args": copy.deepcopy(admission.get("args")),
               "call_kwargs": copy.deepcopy(admission.get("kwargs")),
               "partition": admission.get("partition"), "provenance": copy.deepcopy(admission.get("provenance", {})),
               "evidence_scope": "single_admitted_case", "certifies_clause": False,
               "revision_count": int(admission.get("revision_count", 0) or 0),
               "runner_status": status, "observation_matches": observation_matches,
               "revision_matches": revision_matches,
               "model_judgment": "required" if result == "FAIL" else None,
               "receipt_id": _text(_field(receipt, "receipt_id"))}
    records = getattr(runtime, "behavior_witness_records", None)
    if isinstance(records, list):
        records.append(copy.deepcopy(record))
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("behavior_witness_bound", producer="kernel_behavior_witness", event_schema=SCHEMA,
             clause_id=record["clause_id"], result=record["result"], admitted=True,
              input=record["input"], expected=record["expected"], observed=record["observed"],
              observed_full=record["observed_full"], receipt_status=record["receipt_status"],
              exit_code=record["exit_code"], receipt_body=record["receipt_body"],
              partition=record["partition"], receipt_id=record["receipt_id"], dispatch=False)
    return record


def execute_admitted(runtime: object, admission: Mapping[str, object], runner: object,
                     *, question: str | None = None) -> dict[str, object]:
    """Execute one admitted case through the supplied public probe runner."""
    if admission.get("admitted") is not True:
        return {"schema": SCHEMA, "admitted": False, "result": "NOT_RUN",
                "reason": "admission_required"}
    if _text(admission.get("entrypoint_kind")) == "source_unit":
        run_unit = getattr(runner, "run_source_unit", None)
        if callable(run_unit):
            receipt = run_unit(admission, question or
                               "public source-unit behavior witness " + _text(admission.get("clause_id")))
        else:
            run = getattr(runner, "run", None)
            source_unit = admission.get("source_unit")
            language = _text(source_unit.get("language")).casefold() if isinstance(source_unit, Mapping) else ""
            if language != "python" or not callable(run):
                return {"schema": SCHEMA, "admitted": True, "result": UNRUNNABLE_RESULT,
                        "reason": "authorized_source_unit_loader_unavailable",
                        "source_unit_handle": _text(admission.get("source_unit_handle"))}
            receipt = run(probe_source(admission), question or
                          "public source-unit behavior witness " + _text(admission.get("clause_id")))
        record_observation = getattr(runtime, "record_observation_probe", None)
        if callable(record_observation):
            record_observation(receipt)
        return bind_probe_result(runtime, admission, receipt,
                                 observed=_observed_from_receipt(receipt))
    if _text(admission.get("entrypoint_kind")) == "unsupported":
        return {"schema": SCHEMA, "admitted": True, "result": UNRUNNABLE_RESULT,
                "reason": "language_adapter_unavailable",
                "language": _text(admission.get("entrypoint_module"))}
    run = getattr(runner, "run", None)
    if not callable(run):
        receipt: object = {"receipt_id": "", "status": "RUNNER_ERROR", "exit_code": None,
                           "shown_output": "runner.run is unavailable"}
    else:
        receipt = run(probe_source(admission), question or
                      "public behavior witness " + _text(admission.get("clause_id")))
    record_observation = getattr(runtime, "record_observation_probe", None)
    if callable(record_observation):
        record_observation(receipt)
    return bind_probe_result(runtime, admission, receipt,
                             observed=_observed_from_receipt(receipt))


def rerun_failed_witnesses(runtime: object, runner: object) -> int:
    """Replay the latest failed admitted case for each clause after an edit."""
    if not enabled() or not callable(getattr(runner, "run", None)):
        return 0
    latest: dict[str, Mapping[str, object]] = {}
    for record in _sequence(getattr(runtime, "behavior_witness_records", ())):
        if (isinstance(record, Mapping) and record.get("admitted") is True
                and _text(record.get("result")).upper() == "FAIL"):
            latest[_text(record.get("clause_id"))] = record
    if not latest:
        return 0
    revision = _current_revision(runtime)
    count = 0
    for record in latest.values():
        admission = {
            "admitted": True, "clause_id": _text(record.get("clause_id")),
            "clause": _text(record.get("clause")),
            "entrypoint": _text(record.get("entrypoint")),
            "input": copy.deepcopy(record.get("input")),
            "expected": copy.deepcopy(record.get("expected")),
            "args": copy.deepcopy(record.get("call_args")),
            "kwargs": copy.deepcopy(record.get("call_kwargs")),
            "uses_typed_call": record.get("call_args") is not None or record.get("call_kwargs") is not None,
            "observation_projection": copy.deepcopy(record.get("observation_projection")),
            "partition": record.get("partition"),
            "provenance": copy.deepcopy(record.get("provenance", {})),
            "revision": revision, "revision_digest": _json_digest(revision),
        }
        receipt = runner.run(probe_source(admission),
                             "rerun public behavior witness " + admission["clause_id"])
        record_observation = getattr(runtime, "record_observation_probe", None)
        if callable(record_observation):
            record_observation(receipt)
        bind_probe_result(runtime, admission, receipt, observed=_observed_from_receipt(receipt))
        count += 1
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        emit("behavior_witness_rerun_after_edit", producer="kernel_behavior_witness",
             event_schema=SCHEMA, count=count, clauses=sorted(latest), dispatch=False)
    return count


def bind_solver_authored_check_receipt(
        receipt: Mapping[str, object], *, source_unit_handle: str, revision: str,
        input_value: object, parent_identity: str, candidate_identity: str,
        clause_id: str, execution_store: object = None) -> dict[str, object]:
    """Validate an executed solver receipt without creating its check/oracle."""
    from kernel_aha10_honest_loop import validate_solver_check_receipt
    return validate_solver_check_receipt(
        receipt, source_unit_handle=source_unit_handle, revision=revision,
        input_value=input_value, parent_identity=parent_identity,
        candidate_identity=candidate_identity, clause_id=clause_id,
        execution_store=execution_store,
    )


def admit_authored_check(
        proposal: Mapping[str, object], *, public_text: str,
        parent_revision: object, candidate_revision: object,
        authorized_paths: Sequence[str] = (), protected_paths: Sequence[str] = (),
        existing_check_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Expose the separate host-admitted authored-check route."""
    from kernel_authored_check import admit_authored_check as _admit
    return _admit(
        proposal, public_text=public_text, parent_revision=parent_revision,
        candidate_revision=candidate_revision, authorized_paths=authorized_paths,
        protected_paths=protected_paths, existing_check_ids=existing_check_ids,
    )


def execute_authored_check(
        runtime: object, admission: Mapping[str, object], *,
        parent_snapshot: Mapping[str, object], candidate_snapshot: Mapping[str, object],
        adapter: object = None, observation_only: bool = False,
) -> dict[str, object]:
    """Execute one frozen authored artifact without upgrading generic probes."""
    from kernel_authored_check import execute_authored_check as _execute
    return _execute(
        admission, parent_snapshot=parent_snapshot, candidate_snapshot=candidate_snapshot,
        adapter=adapter, observation_only=observation_only, runtime=runtime,
    )


def authored_check_schema() -> dict[str, object]:
    from kernel_authored_check import authored_check_schema as _schema
    return _schema()


__all__ = [
    "ALL_DONE_GATE_ENV", "MAX_DONE_GATE_REQUESTS", "REFUSAL_SETTLE_ENV", "SCHEMA", "UNRUNNABLE_RESULT",
    "PublicInterfaceIndex", "build_public_interface_index",
    "admit_authored_check", "admit_proposal", "all_done_gate_mode", "authored_check_schema",
    "bind_probe_result", "bind_solver_authored_check_receipt",
    "done_gate", "enabled", "mode",
    "context_for_runtime", "pending_clauses", "pending_public_evidence_work",
    "execute_admitted", "execute_authored_check", "admit_revised_proposal", "probe_source", "request_block",
    "record_refused_probe", "record_unrunnable_proposal", "refusal_settle_enabled", "source_unit_match",
    "refusal_settle_mode", "rerun_failed_witnesses", "selected_host_obligation",
]
