"""Fail-closed audits built on :mod:`kernel_control_plane_receipts`.

These functions are intentionally pure.  They audit host-provided menus,
exports, discovery manifests, and experiment rows; they do not execute model
calls or grant completion/certification authority.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from kernel_control_plane_receipts import (
    ATTEMPT_CLASSES,
    ControlPlaneError,
    HostReceipt,
    IdentityError,
    RunIdentity,
    Settlement,
    UNKNOWN,
    UNAVAILABLE,
    bytes_digest,
    digest,
    require_same_identity,
)


AuditStatus = str
PASS: Final = "PASS"
FAIL: Final = "FAIL"
UNKNOWN_STATUS: Final = UNKNOWN
UNAVAILABLE_STATUS: Final = UNAVAILABLE


@dataclass(frozen=True)
class AuditResult:
    status: AuditStatus
    findings: tuple[str, ...] = ()
    evidence: Mapping[str, object] | None = None

    @property
    def ok(self) -> bool:
        return self.status == PASS


def _action_name(row: Mapping[str, object]) -> str:
    return str(row.get("name", row.get("action", ""))).strip()


def _cycles(graph: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    found: list[str] = []

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node in visiting:
            start = path.index(node) if node in path else 0
            found.append("->".join((*path[start:], node)))
            return
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph.get(node, ()):
            visit(str(dependency), (*path, node))
        visiting.remove(node)
        visited.add(node)

    for name in graph:
        visit(str(name), ())
    return tuple(dict.fromkeys(found))


def audit_menu_dispatch(
    wire_menu: Sequence[Mapping[str, object]],
    dispatch: Mapping[str, Mapping[str, object]],
    *,
    terminal_outcome: str | None = None,
    prerequisites: Mapping[str, Sequence[str]] | None = None,
) -> AuditResult:
    """Ensure the final wire menu and dispatcher expose the same authority."""
    findings: list[str] = []
    names = tuple(_action_name(row) for row in wire_menu)
    if any(not name for name in names):
        findings.append("WIRE_ACTION_WITHOUT_NAME")
    missing = tuple(name for name in names if name not in dispatch)
    if missing:
        findings.append("MISSING_DISPATCH:" + ",".join(missing))
    mismatches = []
    for name in names:
        row = dispatch.get(name)
        if row is None:
            continue
        wire_admitted = bool(next(
            (candidate.get("admitted") for candidate in wire_menu if _action_name(candidate) == name),
            True,
        ))
        dispatch_admitted = bool(row.get("admitted", False))
        if wire_admitted != dispatch_admitted:
            mismatches.append(name)
    if mismatches:
        findings.append("WIRE_DISPATCH_MISMATCH:" + ",".join(mismatches))
    cycles = _cycles(prerequisites or {})
    if cycles:
        findings.append("CIRCULAR_PREREQUISITE:" + ";".join(cycles))
    admitted = tuple(name for name, row in dispatch.items() if bool(row.get("admitted", False)))
    if not admitted and not terminal_outcome:
        findings.append("REFUSAL_ONLY_WITHOUT_TERMINAL")
    return AuditResult(
        status=FAIL if findings else PASS,
        findings=tuple(findings),
        evidence={"wire_actions": names, "admitted_actions": admitted,
                  "terminal_outcome": terminal_outcome or UNKNOWN},
    )


def audit_state_separation(states: Mapping[str, object]) -> AuditResult:
    """Reject a single green/ready bit being used as all completion authorities."""
    missing = tuple(name for name in (
        "public_green", "obligation_complete", "ready_to_submit", "accepted",
        "certified", "externally_correct",
    ) if name not in states)
    if missing:
        return AuditResult(UNKNOWN_STATUS, ("MISSING_STATE:" + ",".join(missing),),
                           {"states": dict(states)})
    return AuditResult(PASS, evidence={"states": dict(states)})


def audit_task_contract(contract: Mapping[str, object]) -> AuditResult:
    """Check ticket meaning, witness availability, and evidence roles explicitly."""
    required = (
        "ticket_id", "ticket_version", "required_behavior", "preserved_behavior",
        "completion_evidence", "absent_symbols", "parent_discriminates",
    )
    missing = tuple(name for name in required if name not in contract)
    if missing:
        return AuditResult(UNKNOWN_STATUS, ("TASK_CONTRACT_MISSING:" + ",".join(missing),),
                           {"contract": dict(contract)})
    for name in ("ticket_id", "ticket_version"):
        if not str(contract.get(name, "")).strip():
            return AuditResult(UNKNOWN_STATUS, ("TASK_CONTRACT_EMPTY:" + name,),
                               {"contract": dict(contract)})
    for name in ("required_behavior", "preserved_behavior", "completion_evidence", "absent_symbols"):
        value = contract.get(name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return AuditResult(UNKNOWN_STATUS, ("TASK_CONTRACT_INVALID:" + name,),
                               {"contract": dict(contract)})
    if not contract["required_behavior"] or not contract["preserved_behavior"]:
        return AuditResult(UNKNOWN_STATUS, ("TASK_CONTRACT_EMPTY_BEHAVIOR",),
                           {"contract": dict(contract)})
    if not isinstance(contract.get("parent_discriminates"), bool):
        return AuditResult(UNKNOWN_STATUS, ("PARENT_DISCRIMINATION_UNKNOWN",),
                           {"contract": dict(contract)})
    witness = contract.get("witness")
    if not isinstance(witness, Mapping):
        return AuditResult(UNKNOWN_STATUS, ("WITNESS_UNAVAILABLE",), {"contract": dict(contract)})
    witness_status = str(witness.get("status", UNKNOWN))
    if witness_status in {"UNAVAILABLE", "MISSING", "UNEXECUTED"}:
        return AuditResult(UNAVAILABLE_STATUS, ("WITNESS_UNAVAILABLE",), {"witness": dict(witness)})
    if contract.get("parent_discriminates") is False:
        return AuditResult(FAIL, ("PARENT_PUBLIC_CHECK_NON_DISCRIMINATING",), {"contract": dict(contract)})
    roles = contract.get("evidence_roles")
    if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)) or not roles:
        return AuditResult(UNKNOWN_STATUS, ("EVIDENCE_ROLES_MISSING",), {"contract": dict(contract)})
    return AuditResult(PASS, evidence={"ticket_id": contract["ticket_id"],
                                       "ticket_version": contract["ticket_version"],
                                       "witness": dict(witness), "evidence_roles": tuple(roles)})


def audit_edit_surface(*, required_sites: Sequence[str], observed_sites: Sequence[str]) -> AuditResult:
    required = tuple(dict.fromkeys(str(site) for site in required_sites))
    observed = set(str(site) for site in observed_sites)
    if not required:
        return AuditResult(UNKNOWN_STATUS, ("EDIT_SURFACE_UNAVAILABLE",))
    missing = tuple(site for site in required if site not in observed)
    if missing:
        return AuditResult(UNKNOWN_STATUS, ("EDIT_SURFACE_MISSING:" + ",".join(missing),),
                           {"required": required, "observed": tuple(sorted(observed))})
    return AuditResult(PASS, evidence={"required": required, "observed": tuple(sorted(observed))})


def audit_refusal_progress(refusals: Sequence[Mapping[str, object]]) -> AuditResult:
    """Detect repeated refusals that consume budget without changing relevant state."""
    if not refusals:
        return AuditResult(UNAVAILABLE_STATUS, ("NO_REFUSAL_ROWS",))
    stalled = tuple(
        str(row.get("receipt_id", UNKNOWN)) for row in refusals
        if row.get("before_state") == row.get("after_state")
        and not bool(row.get("productive_action", False))
    )
    if stalled:
        return AuditResult(FAIL, ("ZERO_PROGRESS_REFUSAL:" + ",".join(stalled),),
                           {"stalled_receipts": stalled})
    return AuditResult(PASS, evidence={"refusal_count": len(refusals)})


def audit_closure_reachability(*, terminal_reserve_reachable: object,
                               uncertified_submit_reachable: object,
                               certification_required: object) -> AuditResult:
    """Keep escape reachability separate from certification weakening."""
    if any(value is None for value in (
        terminal_reserve_reachable, uncertified_submit_reachable, certification_required,
    )):
        return AuditResult(UNKNOWN_STATUS, ("CLOSURE_REACHABILITY_UNKNOWN",))
    if not bool(terminal_reserve_reachable) or not bool(uncertified_submit_reachable):
        return AuditResult(FAIL, ("CLOSURE_ESCAPE_UNREACHABLE",))
    if not bool(certification_required):
        return AuditResult(FAIL, ("CERTIFICATION_WEAKENED",))
    return AuditResult(PASS, evidence={
        "terminal_reserve_reachable": True,
        "uncertified_submit_reachable": True,
        "certification_required": True,
    })


def audit_receipts(receipts: Sequence[HostReceipt], *, expected: RunIdentity) -> AuditResult:
    """Require actual host effects and full identity for every claimed evidence row."""
    if not receipts:
        return AuditResult(UNKNOWN_STATUS, ("NO_RECEIPTS",))
    findings = []
    try:
        require_same_identity(receipts, expected)
    except (ControlPlaneError, IdentityError) as exc:
        findings.append(type(exc).__name__)
    for receipt in receipts:
        if not receipt.actual_effect:
            findings.append("RESERVATION_OR_NO_EFFECT:" + receipt.receipt_id)
        if not receipt.payload_digest:
            findings.append("MISSING_PAYLOAD_DIGEST:" + receipt.receipt_id)
    return AuditResult(FAIL if findings else PASS, tuple(dict.fromkeys(findings)),
                       {"receipt_ids": tuple(receipt.receipt_id for receipt in receipts)})


def collection_fingerprint(rows: Sequence[object], *, generation: object) -> str:
    """Fingerprint every row so interior mutation cannot hide behind endpoints."""
    normalized = []
    for row in rows:
        normalized.append(row.as_dict() if isinstance(row, HostReceipt) else row)
    return digest({"generation": generation, "rows": normalized})


def audit_collection_mutation(before: Sequence[object], after: Sequence[object], *,
                              generation_before: object, generation_after: object) -> AuditResult:
    before_key = collection_fingerprint(before, generation=generation_before)
    after_key = collection_fingerprint(after, generation=generation_after)
    changed = before_key != after_key
    return AuditResult(PASS if changed else UNKNOWN_STATUS,
                       () if changed else ("INTERIOR_MUTATION_NOT_DETECTED",),
                       {"before": before_key, "after": after_key})


def audit_telemetry(*, emit_attempted: object, emit_failed: object,
                    diagnostic_visible: object, error_code: object = None) -> AuditResult:
    if not bool(emit_attempted):
        return AuditResult(UNAVAILABLE_STATUS, ("TELEMETRY_NOT_ATTEMPTED",))
    if bool(emit_failed) and not bool(diagnostic_visible):
        return AuditResult(FAIL, ("TELEMETRY_FAILURE_SWALLOWED",), {"error_code": error_code or UNKNOWN})
    return AuditResult(PASS, evidence={"emit_failed": bool(emit_failed), "error_code": error_code})


def audit_settlements(settlements: Sequence[Settlement]) -> AuditResult:
    seen: set[str] = set()
    findings: list[str] = []
    for settlement in settlements:
        if settlement.effect_id in seen:
            findings.append("DUPLICATE_SETTLEMENT:" + settlement.effect_id)
        seen.add(settlement.effect_id)
        for name in ("origin", "signal", "initiator", "evidence_id", "identity_digest"):
            if not str(getattr(settlement, name, "")).strip():
                findings.append("SETTLEMENT_MISSING:" + name)
    return AuditResult(FAIL if findings else PASS, tuple(dict.fromkeys(findings)),
                       {"settled_effects": tuple(sorted(seen))})


def cache_key(inputs: Mapping[str, object], *, fields: Sequence[str]) -> str:
    """Build a cache key from an explicit dependency list, including interior rows."""
    selected = {field: inputs.get(field, UNKNOWN) for field in fields}
    return digest({"fields": tuple(fields), "values": selected})


def audit_cache_inputs(inputs: Mapping[str, object], *, key_fields: Sequence[str],
                       required_fields: Sequence[str]) -> AuditResult:
    missing = tuple(field for field in required_fields if field not in key_fields)
    if missing:
        return AuditResult(FAIL, ("CACHE_KEY_MISSING:" + ",".join(missing),),
                           {"key_fields": tuple(key_fields)})
    return AuditResult(PASS, evidence={"cache_key": cache_key(inputs, fields=key_fields),
                                       "key_fields": tuple(key_fields)})


@dataclass(frozen=True)
class DiscoveryManifest:
    """Completeness facts, including things an ordinary recursive walk omitted."""

    included: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    generated: tuple[str, ...] = ()
    symlinks: tuple[str, ...] = ()
    unsupported_suffixes: tuple[str, ...] = ()
    dependency_boundaries: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "included": self.included, "excluded": self.excluded, "generated": self.generated,
            "symlinks": self.symlinks, "unsupported_suffixes": self.unsupported_suffixes,
            "dependency_boundaries": self.dependency_boundaries, "unreadable": self.unreadable,
        }


def audit_discovery(manifest: DiscoveryManifest | Mapping[str, object]) -> AuditResult:
    required_metadata = (
        "excluded", "generated", "symlinks", "unsupported_suffixes", "dependency_boundaries",
    )
    if isinstance(manifest, DiscoveryManifest):
        value = manifest
        missing_metadata: tuple[str, ...] = ()
    else:
        def values_for(name: str) -> tuple[str, ...]:
            raw = manifest.get(name, ())
            if raw is None:
                return ()
            if isinstance(raw, (str, bytes)):
                return (str(raw),)
            return tuple(str(item) for item in raw)

        missing_metadata = tuple(name for name in required_metadata if name not in manifest)
        value = DiscoveryManifest(
            **{name: values_for(name)
               for name in DiscoveryManifest.__dataclass_fields__}
        )
    if missing_metadata or value.unreadable:
        return AuditResult(UNKNOWN_STATUS, ("DISCOVERY_INCOMPLETE",), value.as_dict())
    return AuditResult(PASS, evidence=value.as_dict())


@dataclass(frozen=True)
class RecoveryPlan:
    status: AuditStatus
    refusal_facts: tuple[str, ...]
    source_handles: tuple[str, ...]
    actions: tuple[str, ...]
    protected_paths: tuple[str, ...]


def rejected_edit_recovery(
    refusal: Mapping[str, object],
    *,
    source_handles: Sequence[str],
    protected_paths: Sequence[str] = (),
) -> RecoveryPlan:
    """Return narrow recovery choices; never turn a rejected edit into broad rewrite."""
    facts = tuple(str(refusal.get(name, UNKNOWN)) for name in (
        "tool", "authority", "reason", "failed_precondition", "current_revision", "candidate_digest",
    ))
    protected = tuple(dict.fromkeys(str(path) for path in protected_paths))
    handles = tuple(dict.fromkeys(str(handle) for handle in source_handles if str(handle).strip()))
    actions = ("reread_current_span", "inspect_callers", "submit_exact_revision_edit") if handles else ()
    status = PASS if handles and refusal.get("reason") else UNKNOWN_STATUS
    return RecoveryPlan(status, facts, handles, actions, protected)


@dataclass(frozen=True)
class GradeReceipt:
    identity: RunIdentity
    control_kind: str
    sealed_digest: str
    executed: bool
    public_contract_satisfied: bool | None
    status: AuditStatus
    private_behavior_required: bool
    runner_evidence: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-grade-receipt/1",
            "identity": self.identity.as_dict(),
            "control_kind": self.control_kind,
            "sealed_digest": self.sealed_digest,
            "executed": self.executed,
            "public_contract_satisfied": self.public_contract_satisfied,
            "status": self.status,
            "private_behavior_required": self.private_behavior_required,
            "runner_evidence": dict(self.runner_evidence),
        }


def _grade_case(case: Mapping[str, object], runner: Callable[[bytes], Mapping[str, object]]) -> GradeReceipt:
    identity = RunIdentity.from_mapping(case.get("identity", {}))
    sealed = case.get("sealed_bytes")
    if not isinstance(sealed, bytes):
        raise ControlPlaneError("grade case requires exact sealed_bytes")
    actual_digest = bytes_digest(sealed)
    expected_digest = str(case.get("export_digest", identity.export_digest))
    if actual_digest != expected_digest:
        raise IdentityError("grade input bytes do not match export digest")
    control_kind = str(case.get("control_kind", "CANDIDATE"))
    try:
        raw = runner(sealed)
    except Exception as exc:  # host execution failures are evidence, not grader crashes
        return GradeReceipt(identity, control_kind, actual_digest, False, None, UNKNOWN_STATUS,
                            False, {"runner_error": type(exc).__name__})
    if not isinstance(raw, Mapping):
        return GradeReceipt(identity, control_kind, actual_digest, False, None, UNKNOWN_STATUS,
                            False, {"reason": "runner returned no mapping"})
    executed = bool(raw.get("executed", False))
    contract = raw.get("public_contract_satisfied")
    contract_value = contract if isinstance(contract, bool) else None
    private_required = bool(raw.get("private_behavior_required", False))
    if not executed:
        status = UNKNOWN_STATUS
    elif private_required:
        status = UNAVAILABLE_STATUS
    elif contract_value is True:
        status = PASS
    elif contract_value is False:
        status = FAIL
    else:
        status = UNKNOWN_STATUS
    return GradeReceipt(identity, control_kind, actual_digest, executed, contract_value,
                        status, private_required, dict(raw))


def grade_export(
    case: Mapping[str, object],
    runner: Callable[[bytes], Mapping[str, object]],
    *,
    parent_case: Mapping[str, object] | None = None,
    alternative_case: Mapping[str, object] | None = None,
) -> tuple[GradeReceipt, ...]:
    """Grade exact sealed bytes and optionally require parent/alternative controls."""
    cases = [case]
    if parent_case is not None:
        cases.append(parent_case)
    if alternative_case is not None:
        cases.append(alternative_case)
    identities = [RunIdentity.from_mapping(item.get("identity", {})) for item in cases]
    base = identities[0]
    # Controls intentionally have different candidate/export/revision values,
    # but they must belong to the same run, task version, release, image,
    # archive, environment, and tool policy.
    execution_fields = (
        "run_uuid", "task_id", "task_version", "release_digest", "image_digest",
        "archive_id", "environment_digest", "tool_policy_digest",
    )
    if any(tuple(getattr(item, field) for field in execution_fields)
           != tuple(getattr(base, field) for field in execution_fields)
           for item in identities[1:]):
        raise IdentityError("grade controls do not share execution identity")
    receipts = tuple(_grade_case(item, runner) for item in cases)
    if parent_case is not None and receipts[1].control_kind != "UNCHANGED_PARENT":
        raise ControlPlaneError("parent control must be UNCHANGED_PARENT")
    if alternative_case is not None and receipts[-1].control_kind != "ALTERNATIVE":
        raise ControlPlaneError("alternative control must be ALTERNATIVE")
    return receipts


def validate_grade_controls(receipts: Sequence[GradeReceipt]) -> AuditResult:
    kinds = {receipt.control_kind for receipt in receipts}
    findings: list[str] = []
    if "UNCHANGED_PARENT" not in kinds:
        findings.append("MISSING_UNCHANGED_PARENT")
    if "ALTERNATIVE" not in kinds:
        findings.append("MISSING_ALTERNATIVE_CONTROL")
    if any(receipt.private_behavior_required for receipt in receipts):
        findings.append("PRIVATE_BEHAVIOR_REQUIRED")
    if any(not receipt.executed for receipt in receipts):
        findings.append("UNEXECUTED_GRADE_CASE")
    if any(receipt.status == UNKNOWN_STATUS for receipt in receipts):
        findings.append("UNKNOWN_GRADE")
    return AuditResult(FAIL if findings else PASS, tuple(findings),
                       {"controls": tuple(sorted(kinds))})


def account_attempts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Keep live, oracle, filler, refusal, interruption, zero-request, and salvage separate."""
    counts = {name: 0 for name in ATTEMPT_CLASSES}
    malformed = 0
    eligible: list[Mapping[str, object]] = []
    for row in rows:
        kind = str(row.get("attempt_class", "UNKNOWN"))
        if kind not in ATTEMPT_CLASSES:
            malformed += 1
            continue
        counts[kind] += 1
        try:
            requests = int(row.get("requests", 0))
        except (TypeError, ValueError):
            malformed += 1
            continue
        if kind == "LIVE_MODEL" and requests > 0 and bool(row.get("executed", False)):
            eligible.append(row)
    return {
        "counts": counts,
        "malformed": malformed,
        "live_model_attempts": len(eligible),
        "denominator": len(eligible),
        "rows": tuple(dict(row) for row in rows),
    }


def audit_attempt_metadata(rows: Sequence[Mapping[str, object]]) -> AuditResult:
    """Require served-model and transport facts before an experiment row is counted."""
    required = ("served_model", "provider", "transport", "refused", "truncated", "reasoning_only")
    findings: list[str] = []
    for index, row in enumerate(rows):
        missing = tuple(name for name in required if name not in row)
        if missing:
            findings.append(f"ROW_{index}_MISSING:" + ",".join(missing))
        try:
            requests = int(row.get("requests", 0))
        except (TypeError, ValueError):
            findings.append(f"ROW_{index}_REQUESTS_INVALID")
            continue
        if str(row.get("attempt_class", "")) == "LIVE_MODEL" and requests > 0:
            if not str(row.get("served_model", "")).strip():
                findings.append(f"ROW_{index}_SERVED_MODEL_UNKNOWN")
    if findings:
        return AuditResult(UNKNOWN_STATUS, tuple(findings))
    return AuditResult(PASS, evidence={"rows": len(rows)})


def treatment_estimate(rows: Sequence[Mapping[str, object]], *, treatment_field: str) -> AuditResult:
    """Compute a public-only, independent estimate; duplicates are not samples."""
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    findings: list[str] = []
    for row in rows:
        if str(row.get("attempt_class")) != "LIVE_MODEL" or not bool(row.get("executed", False)):
            continue
        if bool(row.get("evaluator_only", False)) or bool(row.get("private_reference_used", False)):
            findings.append("NON_PUBLIC_SELECTION")
            continue
        group = str(row.get(treatment_field, UNKNOWN))
        sample = str(row.get("sample_id", UNKNOWN))
        key = (group, sample)
        if key in seen:
            findings.append("DUPLICATE_SAMPLE:" + sample)
            continue
        seen.add(key)
        groups[group].append(row)
    if findings:
        return AuditResult(UNKNOWN_STATUS, tuple(dict.fromkeys(findings)), {"groups": groups})
    if not groups:
        return AuditResult(UNAVAILABLE_STATUS, ("NO_PUBLIC_LIVE_SAMPLES",), {"groups": {}})
    values = {
        group: sum(1 for row in group_rows if bool(row.get("correct", False))) / len(group_rows)
        for group, group_rows in groups.items()
    }
    return AuditResult(PASS, evidence={"groups": {key: len(value) for key, value in groups.items()},
                                        "estimates": values})


def retain_history(rows: Sequence[Mapping[str, object]], *, cap: int) -> tuple[dict[str, object], ...]:
    """Cap presentation without evicting failed hypotheses/open obligations/current proof."""
    if cap < 1:
        raise ValueError("history cap must be positive")
    priority = {
        "FAILED_HYPOTHESIS": 0, "OPEN_OBLIGATION": 1, "LAST_VERIFIED_CANDIDATE": 2,
        "CURRENT": 3, "STALE": 8,
    }
    ranked = sorted(enumerate(rows), key=lambda item: (
        priority.get(str(item[1].get("retention_class", "CURRENT")), 5),
        -int(item[1].get("sequence", item[0])),
    ))
    selected = [row for _, row in ranked[:cap]]
    return tuple(sorted((dict(row) for row in selected), key=lambda row: int(row.get("sequence", 0))))


def audit_run_join(records: Sequence[Mapping[str, object]], *, expected: RunIdentity | None = None) -> AuditResult:
    try:
        identity = require_same_identity(records, expected)
    except (IdentityError, ControlPlaneError) as exc:
        return AuditResult(FAIL, (type(exc).__name__,), {"reason": str(exc)})
    return AuditResult(PASS, evidence={"identity": identity.as_dict()})


__all__ = [
    "AuditResult", "DiscoveryManifest", "FAIL", "GradeReceipt", "PASS", "RecoveryPlan",
    "UNKNOWN_STATUS", "UNAVAILABLE_STATUS", "account_attempts", "audit_attempt_metadata",
    "audit_cache_inputs", "audit_closure_reachability", "audit_collection_mutation",
    "audit_discovery", "audit_edit_surface", "audit_menu_dispatch", "audit_receipts",
    "audit_refusal_progress", "audit_run_join", "audit_settlements", "audit_state_separation",
    "audit_task_contract", "audit_telemetry", "cache_key", "collection_fingerprint",
    "grade_export", "rejected_edit_recovery", "retain_history", "treatment_estimate",
    "validate_grade_controls",
]
