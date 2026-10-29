"""Host-owned S47 contracts for causal repair, verification, and settlement.

The provider may propose records, edits, checks, and a finish summary.  This
module only admits host-issued identities and computes the decisions from
immutable inputs.  It is deliberately small and dependency-free so the same
predicates can be used by the IDEALPATH reducer, the post-green boundary, and
plain-unittest probes.
"""
from __future__ import annotations

import copy
import difflib
import hashlib
import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath


_AUTH_WORD = "per" + "mit"
SCHEMA = "stateframe-s47-causal-fix/1"
CAUSE_SCHEMA = "stateframe-s47-cause-proposal/1"
ADMISSION_SCHEMA = "stateframe-s47-cause-admission/1"
PERMIT_SCHEMA = "stateframe-s47-edit-" + _AUTH_WORD + "/1"
PATCH_SCHEMA = "stateframe-s47-patch-receipt/1"
IMPACT_SCHEMA = "stateframe-s47-impact-receipt/1"
REGRESSION_SCHEMA = "stateframe-s47-regression/1"
MANIFEST_SCHEMA = "stateframe-s47-verification-manifest/1"
DECISION_SCHEMA = "stateframe-s47-verification-decision/1"
TERMINAL_SCHEMA = "stateframe-s47-terminal-disposition/1"

SUPPORTED = "SUPPORTED"
PROVISIONAL = "PROVISIONAL"
REFUTED = "REFUTED"
STALE = "STALE"
BLOCKED = "BLOCKED"

CAUSE_KINDS = frozenset({
    "READ_FIELD", "BRANCH", "ORDER", "STATE_WRITE", "CALL_BINDING", "CONFIG",
})
EDGE_RELATIONS = frozenset({"CALL", "DATA", "CONTROL", "BEFORE", "STATE", "EXCEPTION"})
PREDICATE_OPS = frozenset({
    "EQ", "NE", "BEFORE", "CONTAINS", "RAISES", "COUNT_EQ", "STATE_EQ",
})
OUTCOMES = frozenset({
    "PASS", "BEHAVIOR_FAIL", "INVALID_CHECK", "ENV_ERROR", "ENVIRONMENT",
    "RESOURCE_LIMIT", "NOT_RUN", "UNKNOWN",
})
FORBIDDEN_PUBLIC = (
    "hidden", "gold", "backup", "private", "reference fix", "reference patch",
    "solution file", "grader output", "evaluator output",
)


class S47ContractError(ValueError):
    """Raised only for malformed host-side arguments, never provider prose."""


@dataclass(frozen=True)
class CandidateBudget:
    """A finite host reserve for optional independent candidate sampling."""

    total_tokens: int
    reserve_tokens: int
    total_seconds: float
    reserve_seconds: float
    total_checks: int
    used_tokens: int = 0
    used_seconds: float = 0.0
    used_checks: int = 0
    candidate_count: int = 0

    def admit(self, *, candidates: int, tokens_each: int, seconds_each: float,
              checks_each: int) -> bool:
        """Return whether all requested samples fit without spending reserve."""
        if any(type(value) is not int or value < 0 for value in (
                candidates, tokens_each, checks_each, self.total_tokens,
                self.reserve_tokens, self.total_checks, self.used_tokens,
                self.used_checks, self.candidate_count)):
            return False
        if any(value < 0 for value in (seconds_each, self.total_seconds,
                                       self.reserve_seconds, self.used_seconds)):
            return False
        return (
            self.used_tokens + candidates * tokens_each <= self.total_tokens - self.reserve_tokens
            and self.used_seconds + candidates * seconds_each
            <= self.total_seconds - self.reserve_seconds
            and self.used_checks + candidates * checks_each <= self.total_checks
        )

    def limit(self, *, requested: int, tokens_each: int, seconds_each: float,
              checks_each: int, hard_max: int = 3) -> int:
        """Choose the largest affordable N; N=1 is the safe default."""
        if type(requested) is not int or requested < 1:
            return 0
        upper = min(requested, max(1, hard_max))
        for count in range(upper, 0, -1):
            if self.admit(candidates=count, tokens_each=tokens_each,
                          seconds_each=seconds_each, checks_each=checks_each):
                return count
        return 0


class HostSettlement:
    """Bounded host progress state; menu previews never consume dispatches."""

    def __init__(self, *, attempt_id: str, blocker_key: str, max_dispatches: int = 2) -> None:
        if not _safe_id(attempt_id) or not _safe_id(blocker_key):
            raise S47ContractError("settlement identity is required")
        if type(max_dispatches) is not int or max_dispatches < 1:
            raise S47ContractError("max_dispatches must be positive")
        self.attempt_id = attempt_id
        self.blocker_key = blocker_key
        self.max_dispatches = max_dispatches
        self.dispatches = 0
        self._last_action_key = ""
        self.terminal: dict[str, object] | None = None

    def menu(self, actions: Sequence[Mapping[str, object]]) -> dict[str, object]:
        """Project a menu without incrementing counters or sealing state."""
        return {
            "schema": "stateframe-s47-finish-state/1",
            "attempt_id": self.attempt_id,
            "blocker_key": self.blocker_key,
            "remaining_recovery_dispatches": max(0, self.max_dispatches - self.dispatches),
            "legal_actions": copy.deepcopy(list(actions)),
        }

    def dispatch(self, action_key: str, *, meaningful_progress: bool = False) -> dict[str, object]:
        """Consume one actual action, sharing the bound across aliases."""
        if self.terminal is not None:
            return {"allowed": False, "code": "TERMINAL", "next_action": "receipt"}
        if not _safe_id(action_key):
            return {"allowed": False, "code": "INVALID_ACTION", "next_action": "host_stop"}
        if meaningful_progress:
            self.dispatches = 0
            self._last_action_key = action_key
            return {"allowed": True, "code": "NOVEL_EVIDENCE", "remaining": self.max_dispatches}
        self.dispatches += 1
        self._last_action_key = action_key
        if self.dispatches >= self.max_dispatches:
            self.terminal = {
                "schema": TERMINAL_SCHEMA,
                "attempt_id": self.attempt_id,
                "disposition": "UNCERTIFIED",
                "reason_codes": ["RECOVERY_EXHAUSTED"],
                "accepted": False,
                "certified": False,
            }
            return {"allowed": False, "code": "SETTLED", "next_action": "host_stop"}
        return {"allowed": True, "code": "RECOVERY_FUNDED", "remaining": self.max_dispatches - self.dispatches}

    def settle(self, *, disposition: str, reason_codes: Sequence[str]) -> dict[str, object]:
        """Seal one host result and return it idempotently."""
        if self.terminal is not None:
            return copy.deepcopy(self.terminal)
        value = str(disposition).upper()
        if value not in {"CERTIFIED", "UNCERTIFIED", "INFRA"}:
            raise S47ContractError("invalid terminal disposition")
        certified = value == "CERTIFIED"
        self.terminal = {
            "schema": TERMINAL_SCHEMA,
            "attempt_id": self.attempt_id,
            "disposition": value,
            "reason_codes": list(dict.fromkeys(str(item) for item in reason_codes if str(item))),
            "accepted": certified,
            "certified": certified,
        }
        return copy.deepcopy(self.terminal)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, *, limit: int = 2000) -> str:
    if not isinstance(value, str):
        return ""
    result = value.replace("\x00", " ").strip()
    if not result or len(result) > limit:
        return ""
    lowered = result.casefold()
    if any(term in lowered for term in FORBIDDEN_PUBLIC):
        return ""
    return result


def _safe_id(value: object, *, limit: int = 240) -> bool:
    return bool(_text(value, limit=limit))


def _items(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _ids(value: object, *, max_items: int = 64) -> tuple[str, ...]:
    values = _items(value)
    if len(values) > max_items:
        return ()
    result = tuple(_text(item, limit=240) for item in values)
    return result if result and all(result) else ()


def _registry_set(registry: Mapping[str, object], key: str) -> set[str]:
    values = registry.get(key, ())
    if isinstance(values, Mapping):
        values = values.keys()
    return {value for value in _ids(values, max_items=4096)}


def _known(registry: Mapping[str, object], ref: object, *groups: str) -> bool:
    value = _text(ref, limit=400)
    return bool(value) and any(value in _registry_set(registry, group) for group in groups)


def _known_or_literal(registry: Mapping[str, object], ref: object, *groups: str) -> bool:
    value = _text(ref, limit=400)
    return bool(value) and (any(value in _registry_set(registry, group) for group in groups)
                            or value.startswith(("literal:", "value:")))


def _require_map(value: object, name: str) -> tuple[Mapping[str, object] | None, list[str]]:
    result = _mapping(value)
    return result, [] if result is not None else [name]


def _predicate(value: object, registry: Mapping[str, object], prefix: str) -> list[str]:
    item = _mapping(value)
    if item is None:
        return [prefix]
    missing: list[str] = []
    observable = item.get("observable_ref")
    if not _known(registry, observable, "observables", "source_spans", "nodes"):
        missing.append(prefix + ".observable_ref")
    if str(item.get("op", "")) not in PREDICATE_OPS:
        missing.append(prefix + ".op")
    if not _known_or_literal(registry, item.get("value_ref"), "values", "events", "states"):
        missing.append(prefix + ".value_ref")
    return missing


def validate_cause_proposal(proposal: Mapping[str, object],
                            registry: Mapping[str, object]) -> dict[str, object]:
    """Validate a revision-bound causal proposal without minting host status."""
    if not isinstance(proposal, Mapping) or not isinstance(registry, Mapping):
        return _invalid(CAUSE_SCHEMA, "CAUSE_SHAPE", ("proposal_or_registry",))
    missing: list[str] = []
    if proposal.get("schema") != CAUSE_SCHEMA:
        missing.append("schema")
    epoch = proposal.get("state_epoch")
    if type(epoch) is not int or epoch != registry.get("state_epoch"):
        missing.append("state_epoch")
    for field in ("gate_id", "base_ref", "baseline_ref"):
        if not _safe_id(proposal.get(field)):
            missing.append(field)
    if _text(proposal.get("base_ref")) != _text(registry.get("base_ref")):
        missing.append("base_ref_current")
    requirements = _ids(proposal.get("requirement_ids"), max_items=8)
    known_requirements = _registry_set(registry, "requirement_ids")
    if not requirements or not set(requirements).issubset(known_requirements):
        missing.append("requirement_ids")
    if not _known(registry, proposal.get("baseline_ref"), "observation_refs", "s3_skip_refs"):
        missing.append("baseline_ref_known")

    trigger, trigger_missing = _require_map(proposal.get("trigger"), "trigger")
    decision, decision_missing = _require_map(proposal.get("decision"), "decision")
    symptom, symptom_missing = _require_map(proposal.get("symptom"), "symptom")
    prediction, prediction_missing = _require_map(proposal.get("prediction"), "prediction")
    alternative, alternative_missing = _require_map(proposal.get("alternative"), "alternative")
    missing.extend(trigger_missing + decision_missing + symptom_missing
                   + prediction_missing + alternative_missing)
    spans = ("source_spans", "nodes")
    if trigger is not None:
        if not _known(registry, trigger.get("entry_span_id"), *spans):
            missing.append("trigger.entry_span_id")
        if not _safe_id(trigger.get("scenario_ref")):
            missing.append("trigger.scenario_ref")
        if not _ids(trigger.get("precondition_refs"), max_items=16):
            missing.append("trigger.precondition_refs")
    if decision is not None:
        if not _known(registry, decision.get("span_id"), *spans):
            missing.append("decision.span_id")
        if str(decision.get("kind", "")) not in CAUSE_KINDS:
            missing.append("decision.kind")
        if not _known_or_literal(registry, decision.get("operation_ref"), "operations", "source_spans"):
            missing.append("decision.operation_ref")
        if not _ids(decision.get("input_binding_refs"), max_items=16):
            missing.append("decision.input_binding_refs")
        if not _known_or_literal(registry, decision.get("contract_ref"),
                                 "contracts", "requirements", "requirement_ids"):
            missing.append("decision.contract_ref")
    if symptom is not None:
        if _text(symptom.get("requirement_id")) not in requirements:
            missing.append("symptom.requirement_id")
        if not _known(registry, symptom.get("observable_ref"), *spans, "observables"):
            missing.append("symptom.observable_ref")
        if not _known(registry, symptom.get("outcome_span_id"), *spans):
            missing.append("symptom.outcome_span_id")

    chain = _items(proposal.get("chain"))
    if not chain or len(chain) > 8:
        missing.append("chain")
    else:
        known_nodes = _registry_set(registry, "nodes") | _registry_set(registry, "source_spans")
        known_evidence = (_registry_set(registry, "observation_refs")
                          | _registry_set(registry, "evidence_refs"))
        for index, edge_value in enumerate(chain):
            edge = _mapping(edge_value)
            if edge is None:
                missing.append(f"chain[{index}]")
                continue
            for field in ("from_node_id", "to_node_id"):
                if _text(edge.get(field)) not in known_nodes:
                    missing.append(f"chain[{index}].{field}")
            if str(edge.get("relation", "")) not in EDGE_RELATIONS:
                missing.append(f"chain[{index}].relation")
            evidence = _ids(edge.get("evidence_ids"), max_items=8)
            if not evidence or not set(evidence).issubset(known_evidence):
                missing.append(f"chain[{index}].evidence_ids")

    if prediction is not None:
        if not _known(registry, prediction.get("probe_plan_ref"), "probe_plans"):
            missing.append("prediction.probe_plan_ref")
        if not _known(registry, prediction.get("input_fixture_ref"), "fixtures"):
            missing.append("prediction.input_fixture_ref")
        for key in ("observable_refs", "baseline_expectations", "candidate_expectations"):
            values = _items(prediction.get(key))
            if not values or len(values) > 8:
                missing.append("prediction." + key)
        for index, value in enumerate(_items(prediction.get("baseline_expectations"))):
            missing.extend(_predicate(value, registry, f"prediction.baseline_expectations[{index}]"))
        for index, value in enumerate(_items(prediction.get("candidate_expectations"))):
            missing.extend(_predicate(value, registry, f"prediction.candidate_expectations[{index}]"))
        intervention = _mapping(prediction.get("intervention"))
        if intervention is None or _text(intervention.get("decision_ref")) != _text(
                decision.get("span_id") if decision else None):
            missing.append("prediction.intervention.decision_ref")
        baseline_digest = digest(prediction.get("baseline_expectations"))
        candidate_digest = digest(prediction.get("candidate_expectations"))
        if baseline_digest == candidate_digest:
            missing.append("prediction_not_falsifiable")

    if alternative is not None:
        if _text(alternative.get("scenario_ref")) not in {
                _text(trigger.get("scenario_ref")) if trigger else ""}:
            missing.append("alternative.scenario_ref")
        if not _known(registry, alternative.get("decision_span_id"), *spans):
            missing.append("alternative.decision_span_id")
        if not _safe_id(alternative.get("mechanism_claim")):
            missing.append("alternative.mechanism_claim")
        support = _ids(alternative.get("support_refs"), max_items=8)
        if not support or not set(support).issubset(_registry_set(registry, "evidence_refs")
                                                    | _registry_set(registry, "observation_refs")):
            missing.append("alternative.support_refs")
        if not _known(registry, alternative.get("probe_plan_ref"), "probe_plans"):
            missing.append("alternative.probe_plan_ref")
        expectations = _items(alternative.get("alternative_expectations"))
        if not expectations:
            missing.append("alternative.alternative_expectations")
        for index, value in enumerate(expectations):
            missing.extend(_predicate(value, registry, f"alternative.alternative_expectations[{index}]"))
        if digest(expectations) == digest(prediction.get("candidate_expectations") if prediction else None):
            missing.append("alternative_not_discriminating")
        if not _safe_id(alternative.get("repair_difference_claim")):
            missing.append("alternative.repair_difference_claim")

    scope = _ids(proposal.get("proposed_scope_span_ids"), max_items=32)
    if not scope or not set(scope).issubset(_registry_set(registry, "source_spans")):
        missing.append("proposed_scope_span_ids")
    unresolved = proposal.get("unresolved_binding_ids", ())
    if unresolved is not None and (
        isinstance(unresolved, (str, bytes, bytearray))
        or not isinstance(unresolved, Sequence)
        or len(unresolved) > 32
        or any(not _safe_id(item) for item in unresolved)
    ):
        missing.append("unresolved_binding_ids")
    return _invalid(CAUSE_SCHEMA, "CAUSE_INVALID", tuple(dict.fromkeys(missing))) if missing else {
        "schema": CAUSE_SCHEMA,
        "valid": True,
        "reason_code": "CAUSE_SHAPE_AND_BINDINGS_VALID",
        "proposal_digest": digest(proposal),
        "state_epoch": epoch,
        "base_ref": _text(proposal.get("base_ref")),
    }


def _invalid(schema: str, reason: str, missing: Sequence[str]) -> dict[str, object]:
    return {
        "schema": schema,
        "valid": False,
        "reason_code": reason,
        "missing": list(dict.fromkeys(str(item) for item in missing if str(item))),
    }


def admit_cause(proposal: Mapping[str, object], registry: Mapping[str, object],
                *, review: Mapping[str, object] | None = None) -> dict[str, object]:
    """Mint a host cause admission; skipped reproduction remains provisional."""
    checked = validate_cause_proposal(proposal, registry)
    if checked.get("valid") is not True:
        return {
            "schema": ADMISSION_SCHEMA, "status": BLOCKED,
            "proposal_digest": digest(proposal), "missing_predicate_ids": checked.get("missing", ()),
            "next_action_ids": ["READ_CAUSE_BINDING"], "permit_ref": None,
        }
    baseline_ref = _text(proposal.get("baseline_ref"))
    skips = _registry_set(registry, "s3_skip_refs")
    observation = registry.get("observations", {}).get(baseline_ref) if isinstance(
        registry.get("observations"), Mapping) else None
    is_skip = baseline_ref in skips or (isinstance(observation, Mapping)
                                        and observation.get("kind") == "S3_SKIP")
    review_findings = _items((review or {}).get("findings"))
    review_ok = bool(review_findings) and all(
        isinstance(item, Mapping)
        and str(item.get("result", "")) in {"SUPPORTED", "CONTRADICTED"}
        and _ids(item.get("evidence_ids"), max_items=8)
        for item in review_findings
    ) and any(str(item.get("result", "")) == "SUPPORTED" for item in review_findings
              if isinstance(item, Mapping)) and any(
                  str(item.get("result", "")) == "CONTRADICTED"
                  for item in review_findings if isinstance(item, Mapping)
              )
    baseline_ok = isinstance(observation, Mapping) and observation.get("host_owned") is True \
        and str(observation.get("outcome", "")).upper() == "BEHAVIOR_FAIL" \
        and observation.get("reached") is True
    status = PROVISIONAL if is_skip or not baseline_ok or not review_ok else SUPPORTED
    missing = []
    if is_skip:
        missing.append("S3_SKIPPED")
    if not baseline_ok:
        missing.append("baseline_discriminating_observation")
    if not review_ok:
        missing.append("independent_cause_review")
    return {
        "schema": ADMISSION_SCHEMA,
        "cause_id": "cause:" + digest(proposal)[:24],
        "proposal_digest": checked["proposal_digest"],
        "base_ref": checked["base_ref"],
        "requirement_ids": list(_ids(proposal.get("requirement_ids"), max_items=8)),
        "observation_refs": [baseline_ref],
        "status": status,
        "reachability": "DYNAMIC_REACHED" if baseline_ok else "UNRESOLVED",
        "alternative_status": "RULED_OUT_FOR_SCENARIO" if review_ok else "UNTESTED",
        "missing_predicate_ids": missing,
        "next_action_ids": [] if status == SUPPORTED else ["RUN_CAUSE_DISCRIMINATOR"],
        "permit_ref": None,
    }


def _safe_product_path(value: object) -> str:
    raw = _text(value, limit=500).replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or raw.startswith("tests/"):
        return ""
    if any(part.casefold() in {"backups", "grader", "graders", "hidden", "gold", "private"}
           for part in path.parts):
        return ""
    return path.as_posix()


def issue_edit_permit(cause_admission: Mapping[str, object], registry: Mapping[str, object],
                      *, write_targets: Sequence[Mapping[str, object]],
                      behavioral_delta_id: str, invariant_ids: Sequence[str],
                      caller_obligation_ids: Sequence[str],
                      regression_obligation_ids: Sequence[str],
                      graph_snapshot_id: str, reserve_id: str,
                      failure_receipt_ids: Sequence[str], repro_bundle_hash: str,
                      mode: str = "STRICT") -> dict[str, object]:
    """Mint a cause-bound edit authorization; model labels alone cannot authorize it."""
    mode = str(mode).upper()
    status = str(cause_admission.get("status", ""))
    if mode not in {"STRICT", "DIAGNOSTIC_ONLY"}:
        return _invalid(PERMIT_SCHEMA, "INVALID_MODE", ("mode",))
    if status != SUPPORTED and mode != "DIAGNOSTIC_ONLY":
        return _invalid(PERMIT_SCHEMA, "CAUSE_NOT_SUPPORTED", ("cause_admission",))
    if mode == "STRICT" and (
        cause_admission.get("schema") != ADMISSION_SCHEMA
        or not _safe_id(cause_admission.get("cause_id"))
        or not _safe_id(cause_admission.get("proposal_digest"))
        or not _ids(cause_admission.get("requirement_ids"), max_items=8)
    ):
        return _invalid(PERMIT_SCHEMA, "CAUSE_RECEIPT_INCOMPLETE", ("cause_admission",))
    targets: list[dict[str, object]] = []
    for index, target_value in enumerate(_items(write_targets)):
        target = _mapping(target_value)
        if target is None:
            return _invalid(PERMIT_SCHEMA, "INVALID_WRITE_TARGET", (f"write_targets[{index}]",))
        path = _safe_product_path(target.get("path"))
        source_hash = _text(target.get("source_hash"), limit=200)
        symbols = _ids(target.get("symbol_ids"), max_items=32)
        operations = _ids(target.get("operation_kinds"), max_items=16)
        if not path or not source_hash or not symbols or not operations:
            return _invalid(PERMIT_SCHEMA, "INVALID_WRITE_TARGET", (f"write_targets[{index}]",))
        targets.append({"path": path, "source_hash": source_hash,
                        "symbol_ids": list(symbols), "operation_kinds": list(operations)})
    if mode == "STRICT" and (
        not targets
        or not _ids(invariant_ids, max_items=32)
        or not _ids(caller_obligation_ids, max_items=64)
        or not _ids(regression_obligation_ids, max_items=32)
        or not _ids(failure_receipt_ids, max_items=16)
        or not _safe_id(registry.get("base_files_digest"))
    ):
        return _invalid(PERMIT_SCHEMA, "PERMIT_OBLIGATIONS_REQUIRED", ("obligations",))
    required = {
        "behavioral_delta_id": behavioral_delta_id,
        "graph_snapshot_id": graph_snapshot_id,
        "reserve_id": reserve_id,
        "repro_bundle_hash": repro_bundle_hash,
    }
    if any(not _safe_id(value) for value in required.values()):
        return _invalid(PERMIT_SCHEMA, "PERMIT_IDENTITY_REQUIRED", tuple(required))
    known_graphs = _registry_set(registry, "graph_snapshot_ids")
    if known_graphs and graph_snapshot_id not in known_graphs:
        return _invalid(PERMIT_SCHEMA, "STALE_GRAPH", ("graph_snapshot_id",))
    permit_core = {
        "ticket_hash": _text(registry.get("ticket_hash")),
        "cause_receipt_id": _text(cause_admission.get("cause_id")),
        "cause_epoch": registry.get("state_epoch"),
        "base_product_digest": _text(registry.get("base_product_digest") or registry.get("base_ref")),
        "base_files_digest": _text(registry.get("base_files_digest")),
        "evidence_basis_digest": digest({"cause": cause_admission, "registry": registry.get("state_epoch")}),
        "graph_snapshot_id": graph_snapshot_id,
        "failure_receipt_ids": list(_ids(failure_receipt_ids, max_items=16)),
        "repro_bundle_hash": repro_bundle_hash,
        "write_targets": targets,
        "behavioral_delta_id": behavioral_delta_id,
        "invariant_ids": list(_ids(invariant_ids, max_items=32)),
        "caller_obligation_ids": list(_ids(caller_obligation_ids, max_items=64)),
        "regression_obligation_ids": list(_ids(regression_obligation_ids, max_items=32)),
        "reserve_id": reserve_id,
        "mode": mode,
    }
    permit = {"schema": PERMIT_SCHEMA, **permit_core}
    permit["permit_id"] = _AUTH_WORD + ":" + digest(permit_core)[:24]
    return permit


def _file_digest(source: str) -> str:
    return digest({"source": source})


def files_digest(files: Mapping[str, str]) -> str:
    return digest({str(path): str(source) for path, source in sorted(files.items())})


def _permit_target(permit: Mapping[str, object], path: str) -> Mapping[str, object] | None:
    for item in _items(permit.get("write_targets")):
        target = _mapping(item)
        if target is not None and target.get("path") == path:
            return target
    return None


def _permit_identity_valid(permit: Mapping[str, object]) -> bool:
    permit_id = _text(permit.get("permit_id"), limit=200)
    if not permit_id or ":" not in permit_id:
        return False
    core = {str(key): value for key, value in permit.items()
            if key not in {"schema", "permit_id"}}
    return permit_id == _AUTH_WORD + ":" + digest(core)[:24]


def apply_edit_transaction(base_files: Mapping[str, str], proposal: Mapping[str, object],
                           permit: Mapping[str, object], *, graph: Mapping[str, object] | None = None
                           ) -> dict[str, object]:
    """Apply every edit to an isolated copy and return no partial candidate on failure."""
    if not isinstance(base_files, Mapping) or not isinstance(proposal, Mapping) \
            or not isinstance(permit, Mapping):
        return _invalid(PATCH_SCHEMA, "PATCH_SHAPE", ("base_proposal_permit",))
    if (
        permit.get("schema") != PERMIT_SCHEMA
        or not _permit_identity_valid(permit)
        or proposal.get("permit_id") != permit.get("permit_id")
    ):
        return _invalid(PATCH_SCHEMA, "PERMIT_MISMATCH", ("permit_id",))
    if permit.get("base_files_digest") and files_digest(base_files) != permit.get("base_files_digest"):
        return _invalid(PATCH_SCHEMA, "STALE_BASE", ("base_files_digest",))
    edits = _items(proposal.get("edits"))
    if not edits:
        return _invalid(PATCH_SCHEMA, "NO_EDITS", ("edits",))
    candidate = {str(path): str(source) for path, source in base_files.items()}
    touched: set[str] = set()
    hunk_records: list[dict[str, object]] = []
    for index, edit_value in enumerate(edits):
        edit = _mapping(edit_value)
        if edit is None:
            return _invalid(PATCH_SCHEMA, "EDIT_SHAPE", (f"edits[{index}]",))
        path = _safe_product_path(edit.get("path"))
        target = _permit_target(permit, path)
        basis = _ids(edit.get("basis_ids"), max_items=16)
        old_text = edit.get("old_text", edit.get("old"))
        replacement = edit.get("replacement", edit.get("new"))
        expected_hash = _text(edit.get("expected_source_hash"), limit=200)
        if (not path or target is None or path in touched or not basis
                or not all(_basis_allowed(item, permit) for item in basis)
                or not isinstance(old_text, str) or not isinstance(replacement, str)):
            return _invalid(PATCH_SCHEMA, "EDIT_NOT_PERMITTED", (f"edits[{index}]",))
        before = candidate.get(path)
        target_hash = _text(target.get("source_hash"), limit=200)
        if (
            before is None
            or not target_hash
            or _file_digest(before) != target_hash
            or (expected_hash and _file_digest(before) != expected_hash)
        ):
            return _invalid(PATCH_SCHEMA, "STALE_SOURCE", (path,))
        if before.count(old_text) != 1:
            return _invalid(PATCH_SCHEMA, "SOURCE_ANCHOR_NOT_UNIQUE", (path,))
        candidate[path] = before.replace(old_text, replacement, 1)
        touched.add(path)
        hunk_records.append({"edit_id": _text(edit.get("edit_id")) or f"edit:{index}",
                             "path": path, "basis_ids": list(basis)})

    actual_hunks: list[dict[str, object]] = []
    for path in touched:
        before_lines = str(base_files[path]).splitlines(keepends=True)
        after_lines = candidate[path].splitlines(keepends=True)
        opcodes = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False).get_opcodes()
        for hunk_index, (tag, a_start, a_end, b_start, b_end) in enumerate(opcodes):
            if tag == "equal":
                continue
            matching = [item for item in hunk_records if item["path"] == path]
            if not matching:
                return _invalid(PATCH_SCHEMA, "UNMAPPED_HUNK", (path,))
            actual_hunks.append({
                "hunk_id": f"hunk:{path}:{hunk_index}",
                "path": path,
                "edit_ids": [str(item["edit_id"]) for item in matching],
                "basis_ids": sorted({
                    basis for item in matching for basis in _ids(item.get("basis_ids"), max_items=16)
                }),
                "actual": {
                    "before_start": a_start + 1, "before_end": a_end,
                    "after_start": b_start + 1, "after_end": b_end,
                },
            })
    if not actual_hunks:
        return _invalid(PATCH_SCHEMA, "NO_PRODUCT_CHANGE", ("edits",))
    impact = build_impact_receipt(graph or {},
                                  changed_symbol_ids=_symbols_for_paths(permit, touched),
                                  graph_snapshot_id=str(permit.get("graph_snapshot_id", "")))
    patch_core = {
        "permit_id": permit["permit_id"],
        "base_digest": files_digest(base_files),
        "candidate_digest": files_digest(candidate),
        "hunks": actual_hunks,
        "impact_receipt_id": impact.get("impact_receipt_id"),
        "mapping_status": "COMPLETE",
        "integrity_status": "PASS",
        "application_status": "APPLIED",
    }
    patch = {"schema": PATCH_SCHEMA, "patch_id": "patch:" + digest(patch_core)[:24], **patch_core,
             "candidate_files": candidate, "impact_receipt": impact}
    return patch


def _basis_allowed(value: str, permit: Mapping[str, object]) -> bool:
    allowed = {
        _text(permit.get("cause_receipt_id")),
        *_ids(permit.get("invariant_ids"), max_items=64),
        *_ids(permit.get("caller_obligation_ids"), max_items=64),
        *_ids(permit.get("regression_obligation_ids"), max_items=64),
    }
    return value in allowed


def _symbols_for_paths(permit: Mapping[str, object], paths: set[str]) -> tuple[str, ...]:
    result: list[str] = []
    for item in _items(permit.get("write_targets")):
        target = _mapping(item)
        if target is not None and target.get("path") in paths:
            result.extend(_ids(target.get("symbol_ids"), max_items=32))
    return tuple(dict.fromkeys(result))


def build_impact_receipt(graph: Mapping[str, object], *, changed_symbol_ids: Sequence[str],
                          graph_snapshot_id: str, max_nodes: int = 128) -> dict[str, object]:
    """Project reverse call impact and preserve unresolved graph frontiers."""
    if not isinstance(graph, Mapping):
        graph = {}
    changed = set(_ids(changed_symbol_ids, max_items=64))
    edges = [_mapping(item) for item in _items(graph.get("edges"))]
    edges = [item for item in edges if item is not None]
    reverse: dict[str, list[Mapping[str, object]]] = {}
    for edge in edges:
        target = _text(edge.get("target") or edge.get("callee") or edge.get("to"))
        if target:
            reverse.setdefault(target, []).append(edge)
    queue = deque(changed)
    seen = set(changed)
    affected: list[dict[str, object]] = []
    while queue and len(seen) <= max_nodes:
        target = queue.popleft()
        for edge in reverse.get(target, ()):
            consumer = _text(edge.get("source") or edge.get("caller") or edge.get("from"))
            if not consumer:
                continue
            affected.append({
                "edge_id": _text(edge.get("edge_id")) or "edge:" + digest(edge)[:16],
                "kind": str(edge.get("kind", "CALL")).upper(),
                "consumer_id": consumer,
                "contract_id": _text(edge.get("contract_id")),
                "evidence_refs": list(_ids(edge.get("evidence_refs"), max_items=8)),
            })
            if consumer not in seen:
                seen.add(consumer)
                queue.append(consumer)
    unresolved = list(_ids(graph.get("unresolved_boundaries"), max_items=64))
    if len(seen) > max_nodes:
        unresolved.append("CALL_GRAPH_FRONTIER_TRUNCATED")
    obligations = []
    checks_by_symbol = graph.get("checks_by_symbol", {})
    if isinstance(checks_by_symbol, Mapping):
        for symbol in seen:
            obligations.extend(_ids(checks_by_symbol.get(symbol), max_items=32))
    core = {
        "graph_snapshot_id": graph_snapshot_id,
        "changed_symbol_ids": sorted(changed),
        "source_bindings": sorted(seen),
        "affected_edges": affected,
        "unresolved_boundaries": list(dict.fromkeys(unresolved)),
        "required_check_ids": list(dict.fromkeys(obligations)),
        "dirty_receipt_ids": list(_ids(graph.get("dirty_receipt_ids"), max_items=64)),
    }
    return {"schema": IMPACT_SCHEMA, "impact_receipt_id": "impact:" + digest(core)[:24], **core}


def regression_pair(parent: Mapping[str, object], candidate: Mapping[str, object], *,
                    expected_check_hash: str | None = None) -> dict[str, object]:
    """Validate one real parent-red/candidate-green scenario pair."""
    blockers: list[str] = []
    if not isinstance(parent, Mapping) or not isinstance(candidate, Mapping):
        blockers.append("receipt_shape")
    else:
        parent_outcome = str(parent.get("outcome", parent.get("status", ""))).upper()
        candidate_outcome = str(candidate.get("outcome", candidate.get("status", ""))).upper()
        if parent_outcome in {"FAIL", "FAILED", "RED"}:
            parent_outcome = "BEHAVIOR_FAIL"
        if candidate_outcome in {"GREEN", "PASSED", "SUCCESS", "OK"}:
            candidate_outcome = "PASS"
        if parent_outcome != "BEHAVIOR_FAIL":
            blockers.append("parent_not_behavior_fail")
        if candidate_outcome != "PASS":
            blockers.append("candidate_not_pass")
        if parent.get("reached") is not True or candidate.get("reached") is not True:
            blockers.append("scenario_not_reached")
        if parent.get("product_origin") != "production" or candidate.get("product_origin") != "production":
            blockers.append("not_production_subject")
        if parent.get("check_hash") != candidate.get("check_hash"):
            blockers.append("check_identity_mismatch")
        if expected_check_hash is not None and parent.get("check_hash") != expected_check_hash:
            blockers.append("unexpected_check_hash")
        if parent.get("fixture_sanity") is not True or candidate.get("fixture_sanity") is not True:
            blockers.append("fixture_sanity")
    return {
        "schema": REGRESSION_SCHEMA,
        "valid": not blockers,
        "status": "ADMITTED" if not blockers else "REJECTED",
        "blockers": list(dict.fromkeys(blockers)),
        "parent_receipt_id": _text(parent.get("receipt_id")) if isinstance(parent, Mapping) else "",
        "candidate_receipt_id": _text(candidate.get("receipt_id")) if isinstance(candidate, Mapping) else "",
    }


def verify_manifest(manifest: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
    """Evaluate trustworthy green as a host conjunction, never as a score."""
    blockers: list[str] = []
    if not isinstance(manifest, Mapping):
        return {
            "schema": DECISION_SCHEMA,
            "manifest_digest": digest(manifest),
            "candidate_digest": "",
            "status": "INCOMPLETE",
            "blockers": ["MANIFEST_SHAPE"],
            "terminal_disposition_ref": None,
        }
    if not isinstance(context, Mapping):
        context = {}
    if manifest.get("schema") != MANIFEST_SCHEMA:
        blockers.append("manifest_schema")
    receipts = context.get("receipts", {})
    if not isinstance(receipts, Mapping):
        receipts = {}
    rows = _items(manifest.get("rows"))
    if not rows:
        blockers.append("NO_SCENARIO_ROWS")
    for index, row_value in enumerate(rows):
        row = _mapping(row_value)
        if row is None:
            blockers.append(f"row[{index}]")
            continue
        required_row_fields = {
            "row_id", "baseline_receipt", "candidate_receipt", "check_hash",
        }
        if not required_row_fields.issubset(row) or any(
            not _safe_id(row.get(field)) for field in required_row_fields
        ):
            blockers.append(f"row[{index}]:ROW_IDENTITY")
            continue
        baseline = receipts.get(row.get("baseline_receipt"))
        candidate = receipts.get(row.get("candidate_receipt"))
        pair = regression_pair(baseline or {}, candidate or {},
                               expected_check_hash=_text(row.get("check_hash")) or None)
        if not pair["valid"]:
            blockers.extend(f"{row.get('row_id', index)}:{item}" for item in pair["blockers"])
        expected_nodes = _text(row.get("expected_nodes_hash"))
        if expected_nodes and (
                not isinstance(baseline, Mapping)
                or not isinstance(candidate, Mapping)
                or baseline.get("collected_nodes_hash") != expected_nodes
                or baseline.get("executed_nodes_hash") != expected_nodes
                or candidate.get("collected_nodes_hash") != expected_nodes
                or candidate.get("executed_nodes_hash") != expected_nodes):
            blockers.append(f"{row.get('row_id', index)}:EXPECTED_NODE_SET_MISMATCH")
    public = _items(manifest.get("public_receipts"))
    if not public:
        blockers.append("PUBLIC_REGRESSION_MISSING")
    for receipt in public:
        item = receipts.get(receipt) if not isinstance(receipt, Mapping) else receipt
        if not isinstance(item, Mapping) or str(item.get("outcome", item.get("status", ""))).upper() != "PASS":
            blockers.append("PUBLIC_REGRESSION_NOT_GREEN")
        if isinstance(item, Mapping) and item.get("complete") is not True:
            blockers.append("PUBLIC_EXECUTION_INCOMPLETE")
    controls = _items(manifest.get("control_receipts"))
    if not controls:
        blockers.append("WRONG_REPAIR_CONTROL_MISSING")
    for receipt in controls:
        item = receipts.get(receipt) if not isinstance(receipt, Mapping) else receipt
        if not isinstance(item, Mapping) or item.get("valid") is not True:
            blockers.append("CONTROL_INVALID")
        elif item.get("reached") is not True or str(item.get("outcome", "")).upper() != "BEHAVIOR_FAIL":
            blockers.append("CONTROL_SURVIVED_OR_NOT_REACHED")
    integrity = context.get("integrity", manifest.get("integrity_receipt"))
    if not isinstance(integrity, Mapping) or any(integrity.get(key) is not True for key in (
            "oracle_unchanged", "no_test_edits", "execution_identity", "solver_independent")):
        blockers.append("ORACLE_OR_EXECUTION_INTEGRITY")
    graph = context.get("depgraph", manifest.get("depgraph"))
    if not isinstance(graph, Mapping) or graph.get("admitted") is not True or _items(graph.get("dirty_node_ids")):
        blockers.append("EVIDENCE_DIRTY_OR_DEPGRAPH_UNAVAILABLE")
    quality = context.get("quality", manifest.get("quality_receipts"))
    if not isinstance(quality, Mapping) or quality.get("complete") is not True or quality.get("passed") is not True:
        blockers.append("QUALITY_CHECKS_INCOMPLETE")
    candidate_digest = _text(manifest.get("candidate_digest"))
    export = context.get("export", manifest.get("export_receipt"))
    if not isinstance(export, Mapping) or export.get("stored") is not True or export.get("artifact_digest") != candidate_digest:
        blockers.append("EXPORT_IDENTITY_OR_STORAGE")
    origin = str(manifest.get("repro_origin", ""))
    if origin == "S3_SKIPPED":
        blockers.append("S3_SKIPPED_NO_REPRO")
    raw_unresolved = manifest.get("unresolved_ids", ())
    if raw_unresolved is not None and (
        isinstance(raw_unresolved, (str, bytes, bytearray))
        or not isinstance(raw_unresolved, Sequence)
        or len(raw_unresolved) > 128
        or any(not _safe_id(item) for item in raw_unresolved)
    ):
        blockers.append("UNRESOLVED_ID_SHAPE")
    unresolved = _ids(raw_unresolved, max_items=128)
    blockers.extend("UNRESOLVED:" + item for item in unresolved)
    blockers = list(dict.fromkeys(blockers))
    return {
        "schema": DECISION_SCHEMA,
        "manifest_digest": digest(manifest),
        "candidate_digest": candidate_digest,
        "status": "READY_FOR_S7" if not blockers else "INCOMPLETE",
        "blockers": blockers,
        "terminal_disposition_ref": None,
    }


def settle_terminal(finish: Mapping[str, object], decision: Mapping[str, object], *,
                    artifact: Mapping[str, object] | None, state: dict[str, object]) -> dict[str, object]:
    """Settle one attempt exactly once; late model claims cannot reopen it."""
    if not isinstance(finish, Mapping) or not isinstance(decision, Mapping) or not isinstance(state, dict):
        return _terminal("INFRA", ("TERMINAL_SHAPE_INVALID",), "", "", "")
    attempt = _text(finish.get("attempt_id"))
    scope = _text(finish.get("scope_id"))
    snapshot = _text(finish.get("snapshot_id"))
    if not attempt or not scope or not snapshot:
        return _terminal("INFRA", ("TERMINAL_IDENTITY_MISSING",), attempt, scope, snapshot)
    key = digest({"attempt": attempt, "scope": scope, "snapshot": snapshot})
    prior = state.get(key)
    if isinstance(prior, Mapping):
        return copy.deepcopy(dict(prior))
    blockers = list(_items(decision.get("blockers"))) if isinstance(decision, Mapping) else ["NO_DECISION"]
    disposition = "UNCERTIFIED"
    reasons = [str(item) for item in blockers if str(item)]
    if decision.get("status") != "READY_FOR_S7":
        reasons = reasons or ["VERIFICATION_INCOMPLETE"]
    elif not isinstance(artifact, Mapping) or artifact.get("stored") is not True:
        disposition = "INFRA"
        reasons = ["ARTIFACT_NOT_STORED"]
    elif artifact.get("artifact_digest") != decision.get("candidate_digest"):
        disposition = "INFRA"
        reasons = ["ARTIFACT_DIGEST_MISMATCH"]
    else:
        disposition = "CERTIFIED"
        reasons = []
    result = _terminal(disposition, reasons, attempt, scope, snapshot)
    state[key] = copy.deepcopy(result)
    return result


def _terminal(disposition: str, reasons: Sequence[str], attempt: str,
              scope: str, snapshot: str) -> dict[str, object]:
    certified = disposition == "CERTIFIED"
    core = {
        "schema": TERMINAL_SCHEMA,
        "terminal_id": "terminal:" + digest({"attempt": attempt, "scope": scope, "snapshot": snapshot})[:24],
        "attempt_id": attempt,
        "scope_id": scope,
        "snapshot_id": snapshot,
        "disposition": disposition,
        "reason_codes": list(dict.fromkeys(str(item) for item in reasons if str(item))),
        "delivery": "STORED" if certified else "NOT_STORED",
        "accepted": certified,
        "certified": certified,
    }
    core["terminal_digest"] = digest(core)
    return core


__all__ = [
    "ADMISSION_SCHEMA", "BLOCKED", "CAUSE_SCHEMA", "CandidateBudget", "DECISION_SCHEMA",
    "HostSettlement", "MANIFEST_SCHEMA", "PATCH_SCHEMA", "PERMIT_SCHEMA", "PROVISIONAL",
    "REGRESSION_SCHEMA", "SCHEMA", "S47ContractError", "STALE", "SUPPORTED", "TERMINAL_SCHEMA",
    "admit_cause", "apply_edit_transaction", "build_impact_receipt", "digest", "files_digest",
    "issue_edit_permit", "regression_pair", "settle_terminal", "validate_cause_proposal",
    "verify_manifest",
]
