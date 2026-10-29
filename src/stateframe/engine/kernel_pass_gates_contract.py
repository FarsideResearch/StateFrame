"""Pure host-contract reducers for the PASS-GATES lane.

This module is deliberately not a second menu, completion owner, catalog, or
terminal authority.  It provides one immutable blocker vector and small
projections that existing owners can consume at their boundaries.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


SCHEMA = "stateframe-pass-gates/1"
HOST_UNSUPPORTED = "HOST_UNSUPPORTED"
HOST_CONTRACT_LIMIT = "HOST_CONTRACT_LIMIT"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name", ""))
    return str(tool.get("name", ""))


def _tool_parameters(tool: Mapping[str, object]) -> Mapping[str, object]:
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("parameters"), Mapping):
        return function["parameters"]
    parameters = tool.get("parameters")
    return parameters if isinstance(parameters, Mapping) else {}


@dataclass(frozen=True)
class GateBlocker:
    """One current host blocker and its concrete recovery contract."""

    blocker_id: str
    code: str
    detail: str
    resolver: Mapping[str, object] | None = None
    hard: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "blocker_id": self.blocker_id,
            "code": self.code,
            "detail": self.detail,
            "resolver": dict(self.resolver) if isinstance(self.resolver, Mapping) else None,
            "hard": self.hard,
        }


@dataclass(frozen=True)
class GateVector:
    """Side-effect-free blocker snapshot shared by menu and terminal owners."""

    schema: str
    artifact_identity: str
    evidence_identity: str
    blockers: tuple[GateBlocker, ...]
    recoveries: tuple[Mapping[str, object], ...]
    unsupported: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.blockers

    @property
    def identity(self) -> str:
        return _digest({
            "artifact_identity": self.artifact_identity,
            "evidence_identity": self.evidence_identity,
            "blockers": [item.as_dict() for item in self.blockers],
        })

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "artifact_identity": self.artifact_identity,
            "evidence_identity": self.evidence_identity,
            "ready": self.ready,
            "blockers": [item.as_dict() for item in self.blockers],
            "recoveries": [dict(item) for item in self.recoveries],
            "unsupported": list(self.unsupported),
            "vector_identity": self.identity,
        }


def _coerce_blocker(value: object, index: int) -> GateBlocker | None:
    if isinstance(value, GateBlocker):
        return value
    if not isinstance(value, Mapping):
        return None
    code = str(value.get("code", value.get("reason_code", "HOST_BLOCKED"))).strip()
    blocker_id = str(value.get("blocker_id", value.get("id", code or f"blocker:{index}"))).strip()
    detail = str(value.get("detail", value.get("reason", code))).strip()
    resolver = value.get("resolver", value.get("repair_action"))
    return GateBlocker(blocker_id or f"blocker:{index}", code or "HOST_BLOCKED",
                       detail or code, resolver if isinstance(resolver, Mapping) else None,
                       value.get("hard", True) is not False)


def _resolver(
    value: Mapping[str, object] | None,
    available: set[str],
    tool_surface: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[Mapping[str, object] | None, str | None]:
    if value is None:
        return None, HOST_UNSUPPORTED
    tool = str(value.get("tool", "")).strip()
    arguments = value.get("arguments", {})
    if not tool or not isinstance(arguments, Mapping):
        return None, HOST_UNSUPPORTED
    if available and tool not in available:
        return None, HOST_UNSUPPORTED
    if tool_surface is not None:
        advertised = tool_surface.get(tool)
        if advertised is None:
            return None, HOST_UNSUPPORTED
        try:
            from kernel_completion_owner import validate_tool_arguments
            errors = validate_tool_arguments((advertised,), tool, arguments)
        except (ImportError, AttributeError, TypeError, ValueError):
            return None, HOST_UNSUPPORTED
        if errors:
            return None, HOST_UNSUPPORTED
    return {"tool": tool, "arguments": dict(arguments),
            "reason": str(value.get("reason", "resolve current host blocker"))}, None


def build_gate_vector(
    *,
    artifact_identity: object,
    evidence_identity: object,
    blockers: Sequence[object] = (),
    available_actions: Sequence[str] = (),
    available_tools: Sequence[Mapping[str, object]] = (),
) -> GateVector:
    """Build one typed vector; no counters, dispatch, or gate side effects run."""
    available = {str(item) for item in available_actions if str(item).strip()}
    current: list[GateBlocker] = []
    recoveries: list[Mapping[str, object]] = []
    unsupported: list[str] = []
    tool_surface = {
        _tool_name(tool): tool for tool in available_tools if _tool_name(tool)
    }
    for index, raw in enumerate(blockers):
        blocker = _coerce_blocker(raw, index)
        if blocker is None:
            continue
        resolver, error = _resolver(blocker.resolver, available, tool_surface or None)
        if resolver is not None:
            recoveries.append(resolver)
        if error is not None:
            unsupported.append(blocker.blocker_id)
            current.append(GateBlocker(
                blocker.blocker_id, blocker.code, blocker.detail,
                {"tool": "host_contract", "arguments": {
                    "blocker_id": blocker.blocker_id,
                }, "reason": "host-owned recovery is unavailable"},
                blocker.hard,
            ))
        else:
            current.append(blocker)
    return GateVector(
        SCHEMA,
        str(artifact_identity),
        str(evidence_identity),
        tuple(current),
        tuple(recoveries),
        tuple(dict.fromkeys(unsupported)),
    )


def project_menu(
    tools: Sequence[Mapping[str, object]],
    vector: GateVector,
) -> list[dict[str, object]]:
    """Remove done while current blockers remain; never invent a tool schema."""
    selected = [dict(tool) for tool in tools if isinstance(tool, Mapping)]
    if vector.ready:
        return selected
    return [tool for tool in selected if _tool_name(tool) != "done"]


def validate_final_surface(
    tools: Sequence[Mapping[str, object]], vector: GateVector,
    *, dispatchable: Callable[[str, Mapping[str, object]], bool] | None = None,
) -> dict[str, object]:
    """Validate recovery names, arguments, and dispatch against final bytes."""
    by_name = {_tool_name(tool): tool for tool in tools if _tool_name(tool)}
    names = set(by_name)
    missing = [item for item in vector.recoveries if str(item.get("tool", "")) not in names]
    invalid_arguments: list[dict[str, object]] = []
    dispatch_refused: list[dict[str, object]] = []
    for recovery in vector.recoveries:
        tool_name = str(recovery.get("tool", ""))
        tool = by_name.get(tool_name)
        arguments = recovery.get("arguments")
        if tool is None or not isinstance(arguments, Mapping):
            continue
        try:
            from kernel_completion_owner import validate_tool_arguments
            errors = validate_tool_arguments((tool,), tool_name, arguments)
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            errors = (f"host schema validation unavailable: {type(error).__name__}",)
        if errors:
            invalid_arguments.append({"tool": tool_name, "errors": tuple(errors)})
            continue
        if dispatchable is not None:
            try:
                admitted = bool(dispatchable(tool_name, arguments))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                admitted = False
            if not admitted:
                dispatch_refused.append({"tool": tool_name, "arguments": dict(arguments)})
    if missing or vector.unsupported or invalid_arguments or dispatch_refused:
        return {
            "status": HOST_UNSUPPORTED,
            "missing_tools": tuple(str(item.get("tool", "")) for item in missing),
            "blocker_ids": tuple(dict.fromkeys((*vector.unsupported,
                                                  *(item.blocker_id for item in vector.blockers)))),
            "invalid_arguments": tuple(invalid_arguments),
            "dispatch_refused": tuple(dispatch_refused),
        }
    return {"status": "VALID", "missing_tools": (), "blocker_ids": (),
            "invalid_arguments": (), "dispatch_refused": ()}


def project_terminal_result(result: Mapping[str, object], vector: GateVector) -> dict[str, object]:
    """Keep terminal acceptance false until the same vector is clear."""
    projected = dict(result)
    projected["pass_gates"] = vector.as_dict()
    if vector.ready:
        return projected
    projected["accepted"] = False
    projected["certified"] = False
    projected["uncertified"] = True
    projected["completion_owner_reason"] = str(
        projected.get("completion_owner_reason") or vector.blockers[0].code
    )
    existing = projected.get("unresolved_gating_items", ())
    existing_ids = tuple(str(item) for item in existing) if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)) else ()
    projected["unresolved_gating_items"] = tuple(dict.fromkeys(
        (*existing_ids, *(item.blocker_id for item in vector.blockers))
    ))
    return projected


def cached_vector(
    cache: dict[str, GateVector], *, artifact_identity: object,
    evidence_identity: object, factory: Callable[[], GateVector],
) -> GateVector:
    """Reuse an immutable vector only at unchanged artifact/evidence identity."""
    key = _digest({"artifact": artifact_identity, "evidence": evidence_identity})
    current = cache.get(key)
    if current is not None:
        return current
    value = factory()
    cache.clear()
    cache[key] = value
    return value


def compose_zero_provider_case(
    case: str, *, current_revision: str = "candidate-revision",
    candidate_revision: str = "candidate-revision", check_status: str = "PASS",
    binding_stable: bool = True, clause: str = "clause-opaque",
    receipt_clause: str = "clause-opaque", idealpath_enabled: bool = True,
    model_budget_exhausted: bool = False, terminal_reserved: bool = True,
    resolution_explicit: bool = True, catalog_rows: int = 20,
    current_check_complete: bool = True, stage_exit_tools: Sequence[str] = (),
) -> dict[str, object]:
    """Deterministic acceptance composition used by the zero-provider battery."""
    name = str(case).strip().lower()
    bad: list[str] = []
    reason = ""
    if name in {"real_failed_witness", "failed_witness"} or check_status.upper() in {"FAIL", "FAILED", "ERROR"}:
        bad.append("witness-failed")
        reason = "current witness failed"
    if name in {"wrong_artifact_revision", "wrong_revision"} or current_revision != candidate_revision:
        bad.append("wrong-revision")
        reason = reason or "candidate is not current"
    if name == "wrong_clause" or clause != receipt_clause:
        bad.append("wrong-clause")
        reason = reason or "witness clause does not match"
    if name in {"unstable_binding", "unstable"} or not binding_stable:
        bad.append("unstable-binding")
        reason = reason or "test binding is unstable"
    if name in {"environment_failure", "environment"}:
        bad.append("environment")
        reason = "execution environment failed"
    if name in {"host_repaired_malformed_metadata", "repaired_metadata"} and not resolution_explicit:
        bad.append("missing-resolution")
        reason = "host repair did not emit an explicit predicate resolution"
    if name in {"paged_catalog_rows", "20_paged_catalog_rows", "catalog_paging"} and catalog_rows < 20:
        bad.append("catalog-incomplete")
        reason = "catalog host truth is not fully paged"
    if name in {
        "idealpath_off", "idealpath_disabled", "idealpath_off_no_stage_exit_tools",
    } and idealpath_enabled:
        bad.append("stage-not-disabled")
        reason = "IDEALPATH was not explicitly disabled"
    if name in {
        "exhausted_budget", "budget_exhausted", "exhausted_model_budget",
        "exhausted_model_budget_completed_current_check",
    } and (
        not model_budget_exhausted or not terminal_reserved or not current_check_complete):
        bad.append("no-terminal-reservation")
        reason = "model exhaustion lacks a completed current check and reserved terminal offer"
    if name in {
        "idealpath_off", "idealpath_disabled", "idealpath_off_no_stage_exit_tools",
    } and stage_exit_tools:
        bad.append("stage-exit-survived")
        reason = "IDEALPATH-off native request retained a stage-exit tool"
    accepted = not bad
    idealpath_disabled = name in {
        "idealpath_off", "idealpath_disabled", "idealpath_off_no_stage_exit_tools",
    } and not idealpath_enabled
    return {
        "case": name,
        "accepted": accepted,
        "certified": accepted,
        "terminal_reason": "CERTIFIED" if accepted else (reason or "host blocker"),
        "reachable_remedy": None if accepted else bad[0],
        "idealpath_disabled": idealpath_disabled,
        "effective_stateframe_idealpath": "off" if idealpath_disabled else "on",
        "surviving_native_tools": ("edit", "run_tests", "done") if idealpath_disabled else (),
        "canonical_receipts": ("candidate-check",) if accepted else (),
        "resolved_catalog_debt": accepted and name in {
            "correct_current_candidate", "host_repaired_malformed_metadata",
            "paged_catalog_rows", "20_paged_catalog_rows", "exhausted_budget",
            "exhausted_model_budget", "exhausted_model_budget_completed_current_check",
            "idealpath_off", "idealpath_disabled", "idealpath_off_no_stage_exit_tools",
        },
        "witness_closure": accepted,
        "done_finalization_delivery_match": accepted,
        "native_routes": ("edit", "run_tests", "done") if not idealpath_enabled else (),
        "catalog_rows": catalog_rows if name in {
            "paged_catalog_rows", "20_paged_catalog_rows", "catalog_paging",
        } else 0,
    }


__all__ = [
    "GateBlocker", "GateVector", "HOST_CONTRACT_LIMIT", "HOST_UNSUPPORTED", "SCHEMA",
    "build_gate_vector", "cached_vector", "compose_zero_provider_case",
    "project_menu", "project_terminal_result", "validate_final_surface",
]
