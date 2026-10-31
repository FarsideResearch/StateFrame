"""Host-owned, revision-bound obligation checks.

The work record names requirements, but it is not evidence that a requirement
is implemented.  This module is the small host authority between those two
things.  It deliberately has no model or work-record mutation path: a receipt
is useful only when it is bound to the complete current workspace snapshot,
the frozen check definition, its fixture, and the execution environment.

The public interface is intentionally useful outside the StateFrame runtime as
well.  Tests can construct the typed records and exercise the pure reducer
without starting a worker; the runtime uses the same records after running the
registered checks in an isolated copy of the workspace.
"""
from __future__ import annotations

import copy
import ast
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from kernel_test_python import test_python


MODE_ENV = "STATEFRAME_HOST_CLOSURE"
STATIC_BUNDLES_ENV = "STATEFRAME_STATIC_BUNDLES"
SCHEMA = "stateframe-host-closure/1"
CHECK_SCHEMA = "stateframe-host-checks/1"

SATISFIED = "SATISFIED"
UNSATISFIED = "UNSATISFIED"
UNKNOWN = "UNKNOWN"

PASS = "PASS"
COUNTEREXAMPLE = "COUNTEREXAMPLE"
NOT_EXECUTED = "NOT_EXECUTED"
HARNESS_ERROR = "HARNESS_ERROR"
REUSED = "REUSED"

EXECUTED = "EXECUTED"
SETUP_ERROR = "SETUP_ERROR"
SKIPPED = "SKIPPED"
EXECUTION_STATUSES = frozenset({EXECUTED, SETUP_ERROR, SKIPPED})

APPLIES = "APPLIES"
DOES_NOT_APPLY = "DOES_NOT_APPLY"
APPLICABILITY_UNKNOWN = "UNKNOWN"
ADMITTED = "ADMITTED"
PROVISIONAL = "PROVISIONAL"
CONFLICTED = "CONFLICTED"
UNBOUND = "UNBOUND"

# Reason names are data carried by the typed assessment.  Keeping these as
# constants makes the public reducer API pleasant to use without making the
# executor treat an unknown interpretation as a pass.
NO_VERIFIER = "NO_VERIFIER"
AMBIGUOUS_CONTRACT = "AMBIGUOUS_CONTRACT"
UNBOUND_ENTRYPOINT = "UNBOUND_ENTRYPOINT"
UNBOUND_INVOCATION = "UNBOUND_INVOCATION"

VALIDATED = "VALIDATED"
PROPOSED = "PROPOSED"
AMBIGUOUS = "AMBIGUOUS"

REPAIR = "REPAIR"
VERIFY = "VERIFY"
DIAGNOSE = "DIAGNOSE"
REVIEW = "REVIEW"

REPAIR_PACKET_SCHEMA = "stateframe-repair-packet/1"
REPAIR_PACKET_AUDIT_SCHEMA = "stateframe-repair-packet-audit/1"
REPAIR_RECOVERY_POLICY = (
    "rebind",
    "syntax",
    "same_failure_x2_expand",
    "coupled_regress",
    "no_new_observation",
    "restart_from_baseline",
)
MAX_REPAIR_EDIT_PROPOSALS = 4
MAX_REPAIR_CALLS = 8

TEMPLATES = frozenset({
    "OPTIONAL_INPUT",
    "OUTPUT_EQUALS",
    "OUTPUT_DELTA",
    "REJECT_WITH_STATUS",
    "PRESERVE_CALL_SHAPE",
    "PRESERVE_INPUT",
    "REPLAY_EQUIVALENT",
    "PRESERVE_EXISTING_VALIDATION",
    "COMPATIBILITY_WITNESS",
    "INVOCATION_WITNESS",
    "VALUE_TRANSPORT",
    "ERROR_TRANSLATION",
    "ROUNDING_DISCRIMINATION",
    "SUM_CONSISTENCY",
})
ADVISORY = "ADVISORY"
ORACLE_BASES = frozenset({"TICKET", "PUBLIC_TEST", "PRESERVED_BEHAVIOR", ADVISORY})
RECEIPT_RESULTS = frozenset({PASS, COUNTEREXAMPLE, NOT_EXECUTED, HARNESS_ERROR})
UNKNOWN_REASONS = frozenset({
    "NO_VERIFIER",
    "AMBIGUOUS_CONTRACT",
    "UNBOUND_ENTRYPOINT",
    "MISSING_CHECKS",
    "STALE",
    "NOT_EXECUTED",
    "HARNESS_ERROR",
    SETUP_ERROR,
    SKIPPED,
    "CONFLICTING_RECEIPTS",
    "UNVALIDATED_INTERPRETATION",
    UNBOUND_INVOCATION,
    "ADVISORY_ONLY",
    "UNBOUND_OBSERVABLE",
})

CHECK_DIR = Path(__file__).resolve().parent / "checks"
MAX_OUTPUT_CHARS = 8_000
MAX_FIXTURE_BYTES = 32_000
MAX_TIMEOUT_SECONDS = 30.0
HOST_CLOSURE_RENDER_CAP = 6_000


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def precision_features_enabled() -> bool:
    """Whether opt-in boundary/host precision enrichments may alter receipts."""
    return (os.environ.get("STATEFRAME_HOST_CLOSURE", "off").lower() == "on"
            or os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on")


def static_bundles_mode() -> str:
    value = os.environ.get(STATIC_BUNDLES_ENV, "off").lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{STATIC_BUNDLES_ENV} must be on or off")
    return value


def static_bundles_enabled() -> bool:
    return static_bundles_mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _identity(value: object) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        return value
    return digest(value)


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))


def _copy_mapping(value: object) -> dict[str, object]:
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _receipt_status(value: object) -> str:
    """Return the underlying result, never the display-only ``REUSED`` label."""
    if isinstance(value, Mapping):
        candidates = (
            value.get("underlying_status"),
            value.get("underlying_result"),
            value.get("result"),
            value.get("status"),
            value.get("result_class"),
        )
    else:
        candidates = (
            getattr(value, "status", None),
            getattr(value, "underlying_status", None),
            getattr(value, "result", None),
            getattr(value, "result_class", None),
        )
    for candidate in candidates:
        status = str(candidate or "").upper()
        if status and status != REUSED:
            return status
    return "REUSED" if any(str(candidate or "").upper() == REUSED
                            for candidate in candidates) else ""


def _receipt_execution(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("execution", "")).upper()
    return str(getattr(value, "execution", "")).upper()


def receipt_is_executed_counterexample(value: object) -> bool:
    """Return whether a receipt proves a current, executed counterexample."""
    return (_receipt_execution(value) == EXECUTED
            and _receipt_status(value) == COUNTEREXAMPLE)


def _definition_evidence_complete(value: object) -> bool:
    """Keep boundary-compiled checks advisory until their certificate is complete."""
    if ((isinstance(value, CheckDefinition) and value.oracle_basis == ADVISORY)
            or (isinstance(value, Mapping)
                and str(value.get("oracle_basis", "")).upper() == ADVISORY)):
        return False
    provenance = (value.provenance if isinstance(value, CheckDefinition)
                  else value.get("provenance", {}) if isinstance(value, Mapping) else {})
    if not isinstance(provenance, Mapping) or provenance.get("compiler") != \
            "stateframe-boundary-compiler/1":
        return True
    certificate = provenance.get("admission_certificate")
    return (isinstance(certificate, Mapping)
            and certificate.get("admission") == "GATING"
            and certificate.get("interpretation") == "CONFIRMED"
            and certificate.get("evidence_complete") is True)


def assessment_is_blocking(value: object) -> bool:
    """Reduce receipts, with an explicit normative coverage gap as a blocker.

    Typed assessments carry their receipts.  A small number of renderer/test
    adapters expose only a legacy ``blocking`` flag; retain that adapter path
    when no receipt collection exists, while serialized host assessments are
    always reduced from their explicit receipt evidence.
    """
    if os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on":
        try:
            from kernel_green_lock_repair import is_advisory_assessment
            if is_advisory_assessment(value):
                return False
        except (ImportError, TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        if bool(value.get("coverage_required")):
            verdict = str(value.get("verdict", value.get("status", ""))).upper()
            if verdict != SATISFIED or not (value.get("receipts") or
                                             value.get("evidence_refs") or
                                             value.get("evidence")):
                return True
        if "receipts" not in value:
            return False
        receipts = value.get("receipts")
    elif hasattr(value, "receipts"):
        if bool(getattr(value, "coverage_required", False)):
            verdict = str(getattr(value, "verdict", getattr(value, "status", ""))).upper()
            if verdict != SATISFIED or not (getattr(value, "receipts", ()) or
                                             getattr(value, "evidence_refs", ())):
                return True
        receipts = getattr(value, "receipts")
    else:
        return bool(getattr(value, "blocking", False))
    if isinstance(receipts, Mapping):
        if {"status", "underlying_status", "result_class", "execution"}.intersection(receipts):
            receipts = (receipts,)
        else:
            receipts = receipts.values()
    if isinstance(receipts, Iterable) and not isinstance(receipts, (str, bytes)):
        return any(receipt_is_executed_counterexample(receipt) for receipt in receipts)
    return False


def _bounded_json(value: object, limit: int = MAX_FIXTURE_BYTES) -> object:
    encoded = _canonical(value)
    if len(encoded) > limit:
        raise ValueError(f"host-check fixture exceeds {limit} bytes")
    return copy.deepcopy(value)


@dataclass(frozen=True, init=False)
class ContractCell:
    """One host-bound, public-material contract observation.

    Cells are deliberately data-only.  ``predicate`` is a description of the
    assertion the existing check executor performs; it is never evaluated by
    the compiler and it never carries a model-generated expectation.
    """

    obligation: str
    surface: str
    case_partition: object
    fixture: object
    observable_selector: object
    predicate: object
    oracle_sources: tuple[str, ...]
    applicability: str
    admission: str
    cell_id: str

    def __init__(self, obligation: str | None = None, surface: str = "",
                 case_partition: object = "", fixture: object = None,
                 observable_selector: object = None, predicate: object = None,
                 oracle_sources: object = (), applicability: str = APPLIES,
                 admission: str = ADMITTED, cell_id: str = "",
                 *, obligation_id: str | None = None) -> None:
        ident = obligation if obligation is not None else obligation_id
        if not isinstance(ident, str) or not ident.strip():
            raise ValueError("contract cell requires an obligation")
        if not isinstance(surface, str) or not surface.strip():
            raise ValueError("contract cell requires a surface")
        applicability = str(applicability or APPLICABILITY_UNKNOWN).upper()
        if applicability not in {APPLIES, DOES_NOT_APPLY, APPLICABILITY_UNKNOWN}:
            raise ValueError("unknown contract-cell applicability")
        admission = str(admission or UNBOUND).upper()
        if admission not in {ADMITTED, PROVISIONAL, CONFLICTED, UNBOUND}:
            raise ValueError("unknown contract-cell admission")
        payload = {
            "obligation": ident,
            "surface": surface,
            "case_partition": case_partition,
            "fixture": fixture,
            "observable_selector": observable_selector,
            "predicate": predicate,
            "oracle_sources": list(_strings(oracle_sources)),
            "applicability": applicability,
            "admission": admission,
        }
        payload = _bounded_json(payload)
        object.__setattr__(self, "obligation", ident)
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "case_partition", copy.deepcopy(payload["case_partition"]))
        object.__setattr__(self, "fixture", copy.deepcopy(payload["fixture"]))
        object.__setattr__(self, "observable_selector", copy.deepcopy(payload["observable_selector"]))
        object.__setattr__(self, "predicate", copy.deepcopy(payload["predicate"]))
        object.__setattr__(self, "oracle_sources", tuple(payload["oracle_sources"]))
        object.__setattr__(self, "applicability", applicability)
        object.__setattr__(self, "admission", admission)
        object.__setattr__(self, "cell_id", str(cell_id or digest(payload)))

    @property
    def obligation_id(self) -> str:
        return self.obligation

    @property
    def contract_cell_hash(self) -> str:
        return digest(self.to_dict(include_hash=False))

    @property
    def hash(self) -> str:
        return self.contract_cell_hash

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ContractCell":
        if not isinstance(value, Mapping):
            raise TypeError("contract cell must be an object")
        return cls(
            value.get("obligation", value.get("obligation_id")),
            str(value.get("surface", "")), value.get("case_partition", ""),
            value.get("fixture"), value.get("observable_selector"),
            value.get("predicate"), value.get("oracle_sources", ()),
            str(value.get("applicability", APPLIES)),
            str(value.get("admission", ADMITTED)), str(value.get("cell_id", "")),
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "schema": "stateframe-contract-cell/1",
            "cell_id": self.cell_id,
            "obligation": self.obligation,
            "obligation_id": self.obligation,
            "surface": self.surface,
            "case_partition": copy.deepcopy(self.case_partition),
            "fixture": copy.deepcopy(self.fixture),
            "observable_selector": copy.deepcopy(self.observable_selector),
            "predicate": copy.deepcopy(self.predicate),
            "oracle_sources": list(self.oracle_sources),
            "applicability": self.applicability,
            "admission": self.admission,
        }
        if include_hash:
            result["contract_cell_hash"] = self.contract_cell_hash
        return result


@dataclass(frozen=True, init=False)
class ObligationSpec:
    """A frozen host interpretation of one public clause."""

    obligation_id: str
    clause_span: object
    interpretation: str
    targets: tuple[str, ...]
    acceptance_policy: tuple[str, ...]
    clause: str
    provenance: dict[str, object]
    coupled_obligations: tuple[str, ...]
    contract_cells: tuple[ContractCell, ...]
    coverage_required: bool = field(default=False, repr=False, compare=False)
    coverage_disposition: str = field(default="REQUIRED", repr=False, compare=False)

    def __init__(self, obligation_id: str | None = None, clause_span: object = "",
                 interpretation: str = VALIDATED, targets: object = (),
                 acceptance_policy: object = (), clause: str = "",
                 provenance: object = None, coupled_obligations: object = (),
                 *, id: str | None = None, checks: object = None,
                 contract_cells: object = None, cells: object = None,
                 coverage_required: bool = False,
                 coverage_disposition: str = "REQUIRED") -> None:
        ident = obligation_id if obligation_id is not None else id
        if not isinstance(ident, str) or not ident.strip():
            raise ValueError("obligation spec requires an id")
        interpretation = str(interpretation or AMBIGUOUS).upper()
        if interpretation not in {VALIDATED, PROPOSED, AMBIGUOUS}:
            raise ValueError("unknown obligation interpretation")
        object.__setattr__(self, "obligation_id", ident)
        object.__setattr__(self, "clause_span", copy.deepcopy(clause_span))
        object.__setattr__(self, "interpretation", interpretation)
        object.__setattr__(self, "targets", _strings(targets))
        if checks is not None and not _strings(acceptance_policy):
            acceptance_policy = checks
        object.__setattr__(self, "acceptance_policy", _strings(acceptance_policy))
        text = clause or (clause_span.get("text", "")
                          if isinstance(clause_span, Mapping) else clause_span)
        object.__setattr__(self, "clause", str(text or ""))
        object.__setattr__(self, "provenance", _copy_mapping(provenance))
        object.__setattr__(self, "coupled_obligations", _strings(coupled_obligations))
        object.__setattr__(self, "coverage_required", bool(coverage_required))
        object.__setattr__(self, "coverage_disposition",
                           str(coverage_disposition or "REQUIRED").upper())
        raw_cells = contract_cells if contract_cells is not None else cells
        if isinstance(raw_cells, Mapping):
            raw_cells = raw_cells.values()
        parsed_cells: list[ContractCell] = []
        if isinstance(raw_cells, Sequence) and not isinstance(raw_cells, (str, bytes, bytearray)):
            for item in raw_cells:
                if isinstance(item, ContractCell):
                    parsed_cells.append(item)
                elif isinstance(item, Mapping):
                    parsed_cells.append(ContractCell.from_mapping(item))
        object.__setattr__(self, "contract_cells", tuple(parsed_cells))

    @property
    def id(self) -> str:
        return self.obligation_id

    @property
    def checks(self) -> tuple[str, ...]:
        """Alias used by early callers of the frozen obligation schema."""
        return self.acceptance_policy

    @property
    def acceptance(self) -> tuple[str, ...]:
        return self.acceptance_policy

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ObligationSpec":
        if not isinstance(value, Mapping):
            raise TypeError("obligation spec must be an object")
        return cls(
            value.get("obligation_id", value.get("id")),
            value.get("clause_span", value.get("clause", "")),
            str(value.get("interpretation", VALIDATED)),
            value.get("targets", value.get("mentioned_files", ())),
            value.get("acceptance_policy", value.get("checks", ())),
            str(value.get("clause", "")),
            value.get("provenance", {}),
            value.get("coupled_obligations", value.get("coupled", ())),
            contract_cells=value.get("contract_cells", value.get("cells", ())),
            coverage_required=bool(value.get("coverage_required", False)),
            coverage_disposition=str(value.get("coverage_disposition", "REQUIRED")),
        )

    def to_dict(self, *, identity: object = None) -> dict[str, object]:
        result = {
            "obligation_id": self.obligation_id,
            "id": self.obligation_id,
            "clause_span": copy.deepcopy(self.clause_span),
            "interpretation": self.interpretation,
            "targets": list(self.targets),
            "acceptance_policy": list(self.acceptance_policy),
            "clause": self.clause,
            "provenance": copy.deepcopy(self.provenance),
            "coupled_obligations": list(self.coupled_obligations),
        }
        if self.contract_cells:
            result["contract_cells"] = [cell.to_dict() for cell in self.contract_cells]
        if self.coverage_required or self.coverage_disposition != "REQUIRED":
            result["coverage_required"] = self.coverage_required
            result["coverage_disposition"] = self.coverage_disposition
        return result


@dataclass(frozen=True, init=False)
class CheckDefinition:
    """One frozen template instance from a public check data file."""

    check_id: str
    obligation_id: str
    template: str
    entry_point: str | None
    fixture: object
    expected: object
    provenance: dict[str, object]
    oracle_basis: str
    fixture_id: str
    selection: tuple[str, ...]
    coupled_obligations: tuple[str, ...]
    cheap: bool
    interpretation: str
    contract_cell: ContractCell | None

    def __init__(self, check_id: str, obligation_id: str, template: str,
                 entry_point: str | None = None, fixture: object = None,
                 expected: object = None, provenance: object = None,
                 oracle_basis: str = "TICKET", fixture_id: str = "",
                 selection: object = (), coupled_obligations: object = (),
                 cheap: bool = True, interpretation: str = VALIDATED,
                 contract_cell: object = None, cell: object = None) -> None:
        if not isinstance(check_id, str) or not check_id.strip():
            raise ValueError("check definition requires check_id")
        template = str(template or "").upper()
        if template not in TEMPLATES:
            raise ValueError(f"unknown host check template: {template}")
        basis = str(oracle_basis or "TICKET").upper()
        if basis not in ORACLE_BASES:
            raise ValueError("unknown host check oracle basis")
        interpretation = str(interpretation or AMBIGUOUS).upper()
        if interpretation not in {VALIDATED, PROPOSED, AMBIGUOUS}:
            raise ValueError("unknown check interpretation")
        fixture = _bounded_json({} if fixture is None else fixture)
        expected = _bounded_json(expected)
        object.__setattr__(self, "check_id", check_id)
        object.__setattr__(self, "obligation_id", str(obligation_id))
        object.__setattr__(self, "template", template)
        object.__setattr__(self, "entry_point", entry_point if isinstance(entry_point, str)
                           and entry_point.strip() else None)
        object.__setattr__(self, "fixture", fixture)
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "provenance", _copy_mapping(provenance))
        object.__setattr__(self, "oracle_basis", basis)
        object.__setattr__(self, "fixture_id", str(fixture_id or check_id))
        object.__setattr__(self, "selection", _strings(selection))
        object.__setattr__(self, "coupled_obligations", _strings(coupled_obligations))
        object.__setattr__(self, "cheap", bool(cheap))
        object.__setattr__(self, "interpretation", interpretation)
        if contract_cell is None:
            contract_cell = cell
        if isinstance(contract_cell, Mapping):
            contract_cell = ContractCell.from_mapping(contract_cell)
        if contract_cell is not None and not isinstance(contract_cell, ContractCell):
            raise TypeError("contract_cell must be a ContractCell")
        object.__setattr__(self, "contract_cell", contract_cell)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CheckDefinition":
        if not isinstance(value, Mapping):
            raise TypeError("check definition must be an object")
        return cls(
            str(value.get("check_id", value.get("id", ""))),
            str(value.get("obligation_id", value.get("obligation", ""))),
            str(value.get("template", "")),
            value.get("entry_point"), value.get("fixture"), value.get("expected"),
            value.get("provenance", {}), str(value.get("oracle_basis", "TICKET")),
            str(value.get("fixture_id", "")), value.get("selection", value.get("selectors", ())),
            value.get("coupled_obligations", value.get("coupled", ())),
            bool(value.get("cheap", True)), str(value.get("interpretation", VALIDATED)),
            value.get("contract_cell", value.get("cell")),
        )

    @property
    def digest(self) -> str:
        return digest(self.to_dict(include_digest=False))

    @property
    def fixture_digest(self) -> str:
        return digest({"fixture_id": self.fixture_id, "fixture": self.fixture})

    @property
    def contract_cell_hash(self) -> str:
        return (self.contract_cell.contract_cell_hash
                if self.contract_cell is not None else "")

    def instance_key(self, snapshot: object, environment: object) -> str:
        return digest({
            "snapshot": _identity(snapshot),
            "check": self.digest,
            "selection": list(self.selection),
            "fixture": self.fixture_digest,
            "environment": _identity(environment),
        })

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        result = {
            "check_id": self.check_id,
            "obligation_id": self.obligation_id,
            "template": self.template,
            "entry_point": self.entry_point,
            "fixture": copy.deepcopy(self.fixture),
            "expected": copy.deepcopy(self.expected),
            "provenance": copy.deepcopy(self.provenance),
            "oracle_basis": self.oracle_basis,
            "fixture_id": self.fixture_id,
            "selection": list(self.selection),
            "coupled_obligations": list(self.coupled_obligations),
            "cheap": self.cheap,
            "interpretation": self.interpretation,
        }
        if self.contract_cell is not None:
            result["contract_cell"] = self.contract_cell.to_dict()
        if include_digest:
            result["check_digest"] = self.digest
        return result


CheckSpec = CheckDefinition


@dataclass(frozen=True, init=False)
class CheckReceipt:
    """Executor-owned result bound to one exact check instance."""

    check_id: str
    status: str
    snapshot_digest: str
    check_digest: str
    fixture_digest: str
    environment_digest: str
    oracle_basis: str
    observed: object
    expected: object
    counterexample: object
    reason: str
    evidence_ref: str
    receipt_id: str
    instance_key: str
    reused: bool
    duplicate_count: int
    selection: tuple[str, ...]
    contract_cell_hash: str
    execution: str
    tested_domain: object
    observed_values: object
    trace_refs: tuple[str, ...]
    identity: Mapping[str, object] | None = field(default=None, compare=False, repr=False)
    literal_grounded: bool | None = field(default=None, compare=False, repr=False)
    alignment_evidence: object = field(default=None, compare=False, repr=False)
    alignment_requirement_id: str = field(default="", compare=False, repr=False)

    def __init__(self, check_id: str, status: str = NOT_EXECUTED,
                 snapshot_digest: object = None, check_digest: object = None,
                 fixture_digest: object = None, environment_digest: object = None,
                 oracle_basis: str = "TICKET", observed: object = None,
                 expected: object = None, counterexample: object = None,
                 reason: str = "", evidence_ref: str = "", receipt_id: str = "",
                 instance_key: str = "", reused: bool = False,
                 duplicate_count: int = 0, selection: object = (),
                 *, snapshot: object = None, check: object = None,
                 fixture: object = None, environment: object = None,
                 result: str | None = None, result_class: str | None = None,
                 underlying_status: str | None = None,
                 contract_cell_hash: object = None,
                 execution: str | None = None,
                 tested_domain: object = None,
                 observed_values: object = None,
                 trace_refs: object = (),
                 identity: Mapping[str, object] | None = None,
                 literal_grounded: bool | None = None,
                 alignment_evidence: object = None,
                 alignment_requirement_id: object = "") -> None:
        if snapshot_digest is None:
            snapshot_digest = snapshot
        if check_digest is None:
            check_digest = check
        if fixture_digest is None:
            fixture_digest = fixture
        if environment_digest is None:
            environment_digest = environment
        status = str(underlying_status or result or result_class or status or NOT_EXECUTED).upper()
        if status == REUSED:
            candidate = str(underlying_status or "").upper()
            status = candidate if candidate in RECEIPT_RESULTS else NOT_EXECUTED
            reused = True
        if status not in RECEIPT_RESULTS:
            raise ValueError(f"unknown check receipt result: {status}")
        basis = str(oracle_basis or "TICKET").upper()
        if basis not in ORACLE_BASES:
            raise ValueError("unknown receipt oracle basis")
        if execution is None:
            execution = (SKIPPED if status == NOT_EXECUTED else
                         SETUP_ERROR if status == HARNESS_ERROR else EXECUTED)
        execution = str(execution).upper()
        if execution not in EXECUTION_STATUSES:
            raise ValueError(f"unknown check execution status: {execution}")
        duplicate_count = max(0, int(duplicate_count))
        snap = _identity(snapshot_digest)
        check_id_digest = _identity(check_digest if check_digest is not None else check_id)
        fixture_id_digest = _identity(fixture_digest)
        env_digest = _identity(environment_digest)
        instance = str(instance_key or digest({
            "snapshot": snap, "check": check_id_digest,
            "fixture": fixture_id_digest, "environment": env_digest,
            "selection": list(_strings(selection)),
        }))
        core = {
            "check_id": check_id, "status": status, "snapshot_digest": snap,
            "check_digest": check_id_digest, "fixture_digest": fixture_id_digest,
            "environment_digest": env_digest, "oracle_basis": basis,
            "observed": observed, "expected": expected,
            "counterexample": counterexample, "reason": str(reason or ""),
            "instance_key": instance, "selection": list(_strings(selection)),
            "contract_cell_hash": str(contract_cell_hash or ""),
            "execution": execution, "tested_domain": tested_domain,
            "observed_values": observed if observed_values is None else observed_values,
            "trace_refs": list(_strings(trace_refs)),
        }
        if literal_grounded is not None:
            core["literal_grounded"] = bool(literal_grounded)
        if alignment_evidence is not None:
            core["alignment_evidence"] = copy.deepcopy(alignment_evidence)
        if alignment_requirement_id not in (None, ""):
            core["alignment_requirement_id"] = str(alignment_requirement_id)
        receipt = receipt_id or digest(core)
        evidence = evidence_ref or ("host-check:" + receipt)
        object.__setattr__(self, "check_id", str(check_id))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "snapshot_digest", snap)
        object.__setattr__(self, "check_digest", check_id_digest)
        object.__setattr__(self, "fixture_digest", fixture_id_digest)
        object.__setattr__(self, "environment_digest", env_digest)
        object.__setattr__(self, "oracle_basis", basis)
        object.__setattr__(self, "observed", copy.deepcopy(observed))
        object.__setattr__(self, "expected", copy.deepcopy(expected))
        object.__setattr__(self, "counterexample", copy.deepcopy(counterexample))
        object.__setattr__(self, "reason", str(reason or ""))
        object.__setattr__(self, "evidence_ref", evidence)
        object.__setattr__(self, "receipt_id", receipt)
        object.__setattr__(self, "instance_key", instance)
        object.__setattr__(self, "reused", bool(reused))
        object.__setattr__(self, "duplicate_count", duplicate_count)
        object.__setattr__(self, "selection", _strings(selection))
        object.__setattr__(self, "contract_cell_hash", str(contract_cell_hash or ""))
        object.__setattr__(self, "execution", execution)
        object.__setattr__(self, "tested_domain", copy.deepcopy(tested_domain))
        object.__setattr__(self, "observed_values", copy.deepcopy(
            observed if observed_values is None else observed_values))
        object.__setattr__(self, "trace_refs", _strings(trace_refs))
        object.__setattr__(self, "literal_grounded",
                           None if literal_grounded is None else bool(literal_grounded))
        object.__setattr__(self, "alignment_evidence", copy.deepcopy(alignment_evidence))
        object.__setattr__(self, "alignment_requirement_id", str(alignment_requirement_id or ""))
        if identity is not None:
            if not isinstance(identity, Mapping):
                raise ValueError("check receipt identity must be a mapping")
            object.__setattr__(self, "identity", copy.deepcopy(dict(identity)))
        else:
            object.__setattr__(self, "identity", None)

    @property
    def display_status(self) -> str:
        return REUSED if self.reused else self.status

    @property
    def result(self) -> str:
        return self.status

    @property
    def snapshot(self) -> str:
        return self.snapshot_digest

    @property
    def check(self) -> str:
        return self.check_digest

    @property
    def fixture(self) -> str:
        return self.fixture_digest

    @property
    def environment(self) -> str:
        return self.environment_digest

    @property
    def result_class(self) -> str:
        return self.display_status

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CheckReceipt":
        raw_status = str(value.get("status", NOT_EXECUTED)).upper()
        underlying = str(value.get("underlying_status", "")).upper()
        if underlying not in RECEIPT_RESULTS:
            underlying = str(value.get("underlying_result", "")).upper()
        if underlying not in RECEIPT_RESULTS:
            underlying = str(value.get("result", "")).upper()
        status = underlying if underlying in RECEIPT_RESULTS else raw_status
        return cls(
            str(value.get("check_id", value.get("check", ""))), status,
            value.get("snapshot_digest", value.get("snapshot")),
            value.get("check_digest", value.get("check")),
            value.get("fixture_digest", value.get("fixture")),
            value.get("environment_digest", value.get("environment")),
            str(value.get("oracle_basis", "TICKET")), value.get("observed"),
            value.get("expected"), value.get("counterexample"),
            str(value.get("reason", "")), str(value.get("evidence_ref", "")),
            str(value.get("receipt_id", "")), str(value.get("instance_key", "")),
            bool(value.get("reused", False) or raw_status == REUSED
                 or str(value.get("result_class", "")).upper() == REUSED),
            int(value.get("duplicate_count", 0)), value.get("selection", ()),
            underlying_status=status if status in RECEIPT_RESULTS else None,
            contract_cell_hash=value.get("contract_cell_hash", value.get("cell_hash")),
            execution=value.get("execution"), tested_domain=value.get("tested_domain"),
            observed_values=value.get("observed_values"), trace_refs=value.get("trace_refs", ()),
            identity=(value.get("identity") if isinstance(value.get("identity"), Mapping)
                      else {key: value[key] for key in (
                          "run_id", "code_hash", "release_id", "harness_version")
                      if key in value} or None),
            literal_grounded=(value.get("literal_grounded")
                              if "literal_grounded" in value else None),
            alignment_evidence=(value.get("alignment_evidence")
                                if "alignment_evidence" in value else None),
            alignment_requirement_id=value.get("alignment_requirement_id", ""),
        )

    def to_dict(self, *, identity: object = None) -> dict[str, object]:
        result = {
            "schema": "stateframe-host-check-receipt/1",
            "check_id": self.check_id,
            "status": self.display_status,
            "underlying_status": self.status,
            "result_class": self.display_status,
            "snapshot_digest": self.snapshot_digest,
            "check_digest": self.check_digest,
            "fixture_digest": self.fixture_digest,
            "environment_digest": self.environment_digest,
            "oracle_basis": self.oracle_basis,
            "observed": copy.deepcopy(self.observed),
            "expected": copy.deepcopy(self.expected),
            "counterexample": copy.deepcopy(self.counterexample),
            "reason": self.reason,
            "evidence_ref": self.evidence_ref,
            "receipt_id": self.receipt_id,
            "instance_key": self.instance_key,
            "reused": self.reused,
            "duplicate_count": self.duplicate_count,
            "selection": list(self.selection),
            "contract_cell_hash": self.contract_cell_hash,
            "execution": self.execution,
            "tested_domain": copy.deepcopy(self.tested_domain),
            "observed_values": copy.deepcopy(self.observed_values),
            "trace_refs": list(self.trace_refs),
        }
        if self.literal_grounded is not None:
            result["literal_grounded"] = self.literal_grounded
        if self.alignment_evidence is not None:
            result["alignment_evidence"] = copy.deepcopy(self.alignment_evidence)
        if self.alignment_requirement_id:
            result["alignment_requirement_id"] = self.alignment_requirement_id
        attached = identity if identity is not None else self.identity
        if attached is not None:
            from kernel_verdict_identity import stamp_record
            stamped = stamp_record(result, attached)
            if isinstance(stamped, Mapping):
                return dict(stamped)
        return result


@dataclass(frozen=True)
class ObligationAssessment:
    """Pure-reducer output for one obligation."""

    obligation_id: str
    verdict: str
    reason: str
    check_ids: tuple[str, ...] = ()
    receipts: tuple[CheckReceipt, ...] = ()
    observed: object = None
    expected: object = None
    evidence_refs: tuple[str, ...] = ()
    clause: str = ""
    location_hint: str | None = None
    coupled_obligations: tuple[str, ...] = ()
    localization: dict[str, object] | None = None
    coverage_required: bool = field(default=False, repr=False, compare=False)
    coverage_disposition: str = field(default="REQUIRED", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.verdict not in {SATISFIED, UNSATISFIED, UNKNOWN}:
            raise ValueError("unknown obligation assessment verdict")
        object.__setattr__(self, "check_ids", _strings(self.check_ids))
        object.__setattr__(self, "receipts", tuple(self.receipts))
        object.__setattr__(self, "evidence_refs", _strings(self.evidence_refs))
        object.__setattr__(self, "coupled_obligations", _strings(self.coupled_obligations))
        object.__setattr__(self, "localization", copy.deepcopy(self.localization)
                           if isinstance(self.localization, Mapping) else None)
        object.__setattr__(self, "coverage_required", bool(self.coverage_required))
        object.__setattr__(self, "coverage_disposition",
                           str(self.coverage_disposition or "REQUIRED").upper())

    @property
    def id(self) -> str:
        return self.obligation_id

    @property
    def status(self) -> str:
        return self.verdict

    @property
    def reason_code(self) -> str:
        return self.reason

    @property
    def evidence(self) -> tuple[str, ...]:
        return self.evidence_refs

    @property
    def blocking(self) -> bool:
        return assessment_is_blocking(self)

    def to_dict(self, *, identity: object = None) -> dict[str, object]:
        result = {
            "schema": "stateframe-obligation-assessment/1",
            "obligation_id": self.obligation_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "check_ids": list(self.check_ids),
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "observed": copy.deepcopy(self.observed),
            "expected": copy.deepcopy(self.expected),
            "evidence_refs": list(self.evidence_refs),
            "clause": self.clause,
            "location_hint": self.location_hint,
            "coupled_obligations": list(self.coupled_obligations),
        }
        if self.localization is not None:
            result["localization"] = copy.deepcopy(self.localization)
        if self.coverage_required or self.coverage_disposition != "REQUIRED":
            result["coverage_required"] = self.coverage_required
            result["coverage_disposition"] = self.coverage_disposition
        attached = identity
        if attached is not None:
            from kernel_verdict_identity import stamp_record
            stamped = stamp_record(result, attached)
            if isinstance(stamped, Mapping):
                return dict(stamped)
        return result


def _projection_field(value: object, *names: str, default: object = None) -> object:
    """Read a presentation field from either a typed record or a journal row."""
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _projection_values(value: object) -> tuple[object, ...]:
    if isinstance(value, Mapping):
        receipt_keys = {"check_id", "status", "result_class", "underlying_status"}
        return (value,) if receipt_keys.intersection(value) else tuple(value.values())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _projection_revision(value: object) -> object:
    return _projection_field(value, "snapshot_digest", "snapshot", "revision",
                             default=None)


def _projection_status(value: object, *, display: bool = False) -> str:
    names = (("display_status", "result_class", "status", "result") if display else
             ("underlying_status", "status", "result", "result_class"))
    raw = _projection_field(value, *names, default="")
    return str(raw or "").upper()


def _short_revision(value: object) -> str:
    if value is None:
        return ""
    identity = _identity(value)
    return identity.split(":", 1)[-1][:12]


def _projection_check_obligations(definitions: object,
                                  assessments: Sequence[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for definition in _projection_values(definitions):
        check_id = _projection_field(definition, "check_id", "id", default="")
        obligation_id = _projection_field(definition, "obligation_id", "obligation", default="")
        if str(check_id) and str(obligation_id):
            result[str(check_id)] = str(obligation_id)
    for assessment in assessments:
        obligation_id = _projection_field(assessment, "obligation_id", "id", default="")
        check_ids = _projection_field(assessment, "check_ids", "checks", default=())
        for check_id in _strings(check_ids):
            result.setdefault(check_id, str(obligation_id))
    return result


def _projection_active_obligation(active_gap: object) -> str:
    if isinstance(active_gap, Mapping):
        active_gap = active_gap.get("focus_obligation", active_gap.get("obligation_id", ""))
    return str(active_gap or "")


def _projection_counterexample(assessment: object, latest: Mapping[str, object],
                               check_ids: Sequence[str]) -> tuple[object, object, object]:
    observed = _projection_field(assessment, "observed", default=None)
    expected = _projection_field(assessment, "expected", default=None)
    counterexample = _projection_field(assessment, "counterexample", default=None)
    receipts = _projection_values(_projection_field(assessment, "receipts", default=()))
    selected = next((receipt for receipt in receipts
                     if _projection_status(receipt) == COUNTEREXAMPLE), None)
    if selected is None:
        selected = next((latest[check_id] for check_id in check_ids
                         if check_id in latest
                         and _projection_status(latest[check_id]) == COUNTEREXAMPLE), None)
    if selected is not None:
        selected_observed = _projection_field(selected, "observed", default=None)
        selected_expected = _projection_field(selected, "expected", default=None)
        selected_counterexample = _projection_field(selected, "counterexample", default=None)
        if observed is None:
            observed = selected_observed
        if expected is None:
            expected = selected_expected
        if counterexample is None:
            counterexample = selected_counterexample
    return (copy.deepcopy(observed), copy.deepcopy(expected),
            copy.deepcopy(counterexample))


def _projection_assessment_row(assessment: object, latest: Mapping[str, object],
                               check_obligations: Mapping[str, str],
                               active_obligation: str) -> dict[str, object]:
    obligation_id = str(_projection_field(assessment, "obligation_id", "id", default=""))
    status = str(_projection_field(assessment, "verdict", "status", default=UNKNOWN)).upper()
    if status not in {SATISFIED, UNSATISFIED, UNKNOWN}:
        status = UNKNOWN
    check_ids = _strings(_projection_field(assessment, "check_ids", "checks", default=()))
    if not check_ids:
        check_ids = tuple(check_id for check_id, obligation_id_value in check_obligations.items()
                          if obligation_id_value == obligation_id)
    failing: list[str] = []
    for check_id in check_ids:
        receipt = latest.get(check_id)
        if receipt is not None and _projection_status(receipt) in {
                COUNTEREXAMPLE, HARNESS_ERROR, NOT_EXECUTED}:
            failing.append(check_id)
    if status != SATISFIED and not failing:
        # Missing or stale receipts are themselves the failing evidence for a
        # compact row; preserve the reducer's full reason in host state.
        failing = list(check_ids)
    row: dict[str, object] = {
        "obligation_id": obligation_id,
        "status": status,
        "failing_check_ids": failing,
    }
    if obligation_id == active_obligation:
        observed, expected, counterexample = _projection_counterexample(
            assessment, latest, check_ids)
        row.update(observed=observed, expected=expected, counterexample=counterexample)
    return row


def _projection_json_size(value: Mapping[str, object]) -> int:
    return len(json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), default=str))


def bounded_host_closure_projection(*, assessments: Sequence[object] = (),
                                    receipts: object = (), current_revision: object = None,
                                    definitions: object = (), active_gap: object = None,
                                    check_file: object = None,
                                    cap: int = HOST_CLOSURE_RENDER_CAP,
                                    degrade: bool = False,
                                    journal: object = None) -> dict[str, object]:
    """Render the host closure as a bounded, typed presentation view.

    This function only copies reducer output.  The typed assessments and full
    receipt journal remain owned by the caller, so completion and green-lock
    decisions never consume this projection.
    """
    if type(cap) is not int or cap <= 0:
        raise ValueError("host closure render cap must be a positive integer")
    if degrade:
        from kernel_host_closure_render_degrade import validate_render_cap
        validate_render_cap(cap)
    assessment_values = tuple(assessments or ())
    receipt_values = _projection_values(receipts)
    if current_revision is None:
        revisions = [_projection_revision(receipt) for receipt in receipt_values]
        revisions = [revision for revision in revisions if revision is not None]
        current_revision = revisions[-1] if revisions else None
    current_identity = (_identity(current_revision) if current_revision is not None else None)
    oracle_literal_canary = None
    if os.environ.get("STATEFRAME_ORACLE_TICKET_LITERALS", "off").strip().lower() == "on":
        from kernel_oracle_literal_canary import enabled as oracle_literals_enabled
        if oracle_literals_enabled():
            from kernel_oracle_literal_canary import STALE, is_stale, receipt_revision
            oracle_literal_canary = (STALE, is_stale, receipt_revision)
    check_obligations = _projection_check_obligations(definitions, assessment_values)
    assessment_check_obligations = _projection_check_obligations((), assessment_values)
    check_obligations = {**assessment_check_obligations, **check_obligations}

    latest: dict[str, object] = {}
    stale_receipts: list[object] = []
    for receipt in receipt_values:
        check_id = str(_projection_field(receipt, "check_id", "check", "id", default=""))
        if not check_id:
            continue
        revision = _projection_revision(receipt)
        if (oracle_literal_canary
                and oracle_literal_canary[1](receipt, current_revision)):
            stale_receipts.append(receipt)
            continue
        if current_identity is not None and _identity(revision) != current_identity:
            continue
        latest[check_id] = receipt

    compact_receipts = []
    for check_id, receipt in latest.items():
        explicit_obligation = _projection_field(
            receipt, "obligation_id", "obligation", default="")
        obligation_id = str(explicit_obligation or check_obligations.get(check_id, ""))
        compact_receipts.append({
            "check_id": check_id,
            "obligation_id": obligation_id,
            "status": _projection_status(receipt, display=True),
            "revision": _short_revision(_projection_revision(receipt)),
        })
    if oracle_literal_canary:
        _stale_label, _is_stale, _receipt_revision = oracle_literal_canary
        for receipt in stale_receipts:
            check_id = str(_projection_field(receipt, "check_id", "check", "id", default=""))
            explicit_obligation = _projection_field(
                receipt, "obligation_id", "obligation", default="")
            obligation_id = str(explicit_obligation or check_obligations.get(check_id, ""))
            compact_receipts.append({
                "check_id": check_id,
                "obligation_id": obligation_id,
                "status": _stale_label,
                "revision": str(_receipt_revision(receipt)),
            })

    active_obligation = _projection_active_obligation(active_gap)
    compact_assessments = [
        _projection_assessment_row(
            assessment, latest, check_obligations, active_obligation)
        for assessment in assessment_values
    ]
    view: dict[str, object] = {
        "schema": "stateframe-host-closure/1",
        "mode": "on",
        "check_file": copy.deepcopy(check_file),
        "snapshot": copy.deepcopy(current_revision),
        "assessments": compact_assessments,
        "receipts": compact_receipts,
    }

    omitted_satisfied = 0
    removable = [index for index, row in enumerate(compact_assessments)
                 if row["status"] == SATISFIED
                 and row["obligation_id"] != active_obligation]
    while _projection_json_size(view) > cap and removable:
        index = removable.pop()
        compact_assessments.pop(index)
        omitted_satisfied += 1
        view["assessments"] = compact_assessments
        view["omitted_satisfied"] = omitted_satisfied
    if _projection_json_size(view) > cap:
        if degrade:
            # r43: shed optional detail, then summarise mandatory rows with exact
            # counts and accounted ids; never crash, never exceed the cap.
            from kernel_host_closure_render_degrade import degrade_projection
            return degrade_projection(view, cap=cap, active_id=active_obligation,
                                      journal=journal)
        raise ValueError("host closure mandatory projection exceeds render cap")
    return view


@dataclass(frozen=True)
class CheckBundle:
    task_id: str
    specs: tuple[ObligationSpec, ...]
    checks: tuple[CheckDefinition, ...]
    path: str
    provenance: dict[str, object]
    contract_cells: tuple[ContractCell, ...] = ()

    def checks_for(self, obligation_id: str) -> tuple[CheckDefinition, ...]:
        return tuple(check for check in self.checks if check.obligation_id == obligation_id)

    @property
    def cells(self) -> tuple[ContractCell, ...]:
        return self.contract_cells


@dataclass(frozen=True)
class MutationSite:
    """A source-level write reachable from a public entry point.

    The receiver is recorded as a caller-argument origin rather than inferred
    from the entry function's name.  This catches the common indirect shape
    ``publish(payload) -> Model.from_payload(payload) -> payload.pop(...)``.
    """

    path: str
    function: str
    operation: str
    receiver: str
    receiver_origin: str
    line: int
    column: int
    expression: str
    call_chain: tuple[str, ...] = ()

    @property
    def op(self) -> str:
        """Short schema name used by the history journal."""
        return self.operation

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "function": self.function,
            "op": self.operation,
            "operation": self.operation,
            "receiver": self.receiver,
            "receiver_origin": self.receiver_origin,
            "line": self.line,
            "column": self.column,
            "expression": self.expression,
            "call_chain": list(self.call_chain),
        }


_MUTATION_METHODS = frozenset({
    "pop", "popitem", "append", "extend", "insert", "update", "setdefault",
    "clear", "remove",
})
_FORBIDDEN_SCAN_PARTS = frozenset({"private", "hidden_tests", "near_miss", "gold", "oracle"})


def _visible_python_paths(root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    paths: list[Path] = []
    if not root.is_dir():
        return ()
    for path in sorted(root.rglob("*.py")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        # The history scanner is deliberately unable to inspect private or
        # grading-only material, including hidden directories and backups.
        if any(part.startswith(".") or part in _FORBIDDEN_SCAN_PARTS
               or part in {"tests", "test"} or part.startswith("test_")
               or "backup" in part.lower() for part in relative.parts):
            continue
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    return tuple(paths)


@dataclass(frozen=True)
class _SourceDefinition:
    path: Path
    module: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    qualname: str


def _source_definitions(root: Path) -> tuple[_SourceDefinition, ...]:
    definitions: list[_SourceDefinition] = []
    for path in _visible_python_paths(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
            if module.endswith(".__init__"):
                module = module.removesuffix(".__init__")
        except (OSError, UnicodeError, SyntaxError):
            continue

        def visit(node: ast.AST, prefix: tuple[str, ...]) -> None:
            next_prefix = prefix
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                next_prefix = (*prefix, node.name)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    definitions.append(_SourceDefinition(path, module, node, ".".join(next_prefix)))
            for child in ast.iter_child_nodes(node):
                visit(child, next_prefix)

        visit(tree, ())
    return tuple(definitions)


def _entry_definition(definitions: Sequence[_SourceDefinition], root: Path,
                      entry_point: str) -> _SourceDefinition | None:
    if not isinstance(entry_point, str) or ":" not in entry_point:
        return None
    module, attribute = entry_point.split(":", 1)
    candidates = [item for item in definitions
                  if item.module == module and (item.qualname == attribute
                                                or item.qualname.endswith("." + attribute)
                                                or item.node.name == attribute)]
    if candidates:
        return sorted(candidates, key=lambda item: (item.qualname.count("."), item.qualname))[0]
    # Keep entry-point resolution useful for a module path with an __init__.
    module_path = root.joinpath(*module.split("."))
    if module_path.is_dir():
        candidates = [item for item in definitions
                      if item.path == module_path / "__init__.py"
                      and (item.qualname == attribute or item.node.name == attribute)]
        if candidates:
            return candidates[0]
    return None


def _expression_key(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_key(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return f"{_expression_key(node.value)}[...]"
    return ""


def _mutation_origin(node: ast.AST, aliases: Mapping[str, str]) -> str | None:
    key = _expression_key(node)
    if key and key in aliases:
        return aliases[key]
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _mutation_origin(node.value, aliases)
    if isinstance(node, ast.Call):
        # A getter returns an alias to a nested value; a copy deliberately
        # breaks the caller alias.  Mutation of the call receiver itself is
        # still recorded by the Call visitor below.
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in {"copy", "deepcopy"}:
                return None
            return _mutation_origin(node.func.value, aliases)
        if isinstance(node.func, ast.Name) and node.func.id in {"dict", "list", "set", "tuple"}:
            return None
    return None


def _definition_calls(definition: _SourceDefinition,
                      definitions: Sequence[_SourceDefinition]) -> tuple[_SourceDefinition, ...]:
    same_module = [item for item in definitions if item.module == definition.module]
    result: list[_SourceDefinition] = []
    seen: set[tuple[str, str]] = set()
    for node in ast.walk(definition.node):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.id if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if not name:
            continue
        matches = [item for item in same_module if item.node.name == name]
        if not matches:
            matches = [item for item in definitions if item.node.name == name]
        for match in sorted(matches, key=lambda item: (item.module != definition.module,
                                                        item.qualname)):
            key = (str(match.path), match.qualname)
            if key not in seen and match.node is not definition.node:
                seen.add(key)
                result.append(match)
                break
    return tuple(result)


def _scan_definition(root: Path, definition: _SourceDefinition,
                     call_chain: tuple[str, ...]) -> tuple[MutationSite, ...]:
    try:
        source = definition.path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ()
    aliases: dict[str, str] = {}
    arguments = definition.node.args
    for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
        aliases[argument.arg] = f"param:{argument.arg}"
    if arguments.vararg is not None:
        aliases[arguments.vararg.arg] = f"param:{arguments.vararg.arg}"
    if arguments.kwarg is not None:
        aliases[arguments.kwarg.arg] = f"param:{arguments.kwarg.arg}"
    sites: list[MutationSite] = []
    nodes = sorted(ast.walk(definition.node),
                   key=lambda item: (getattr(item, "lineno", 0),
                                     getattr(item, "col_offset", 0)))

    def record(operation: str, receiver_node: ast.AST, node: ast.AST) -> None:
        origin = _mutation_origin(receiver_node, aliases)
        if origin is None or not origin.startswith("param:"):
            return
        expression = ast.get_source_segment(source, node) or ""
        sites.append(MutationSite(
            path=definition.path.relative_to(root).as_posix(),
            function=definition.qualname,
            operation=operation,
            receiver=ast.get_source_segment(source, receiver_node) or _expression_key(receiver_node),
            receiver_origin="caller_arg",
            line=int(getattr(node, "lineno", 0)),
            column=int(getattr(node, "col_offset", 0)),
            expression=expression,
            call_chain=call_chain,
        ))

    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            origin = _mutation_origin(value, aliases)
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                key = _expression_key(target)
                if key and origin is not None:
                    aliases[key] = origin
                if isinstance(target, ast.Subscript):
                    record("setitem", target.value, node)
                elif isinstance(target, ast.Attribute):
                    record("setitem", target.value, node)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Subscript):
                record("setitem", node.target.value, node)
            elif isinstance(node.target, ast.Attribute):
                record("setitem", node.target.value, node)
            elif isinstance(node.target, ast.Name):
                record("setitem", node.target, node)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript):
                    record("del", target.value, node)
                elif isinstance(target, ast.Attribute):
                    record("del", target.value, node)
                else:
                    record("del", target, node)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _MUTATION_METHODS:
                record(node.func.attr, node.func.value, node)
    # Preserve source order while de-duplicating a nested AST traversal.
    unique: dict[tuple[object, ...], MutationSite] = {}
    for site in sorted(sites, key=lambda item: (item.line, item.column, item.operation)):
        unique[(site.path, site.function, site.operation, site.line, site.column,
               site.expression)] = site
    return tuple(unique.values())


def mutation_sites(root: str | Path, entry_point: str) -> tuple[MutationSite, ...]:
    """Find caller-argument writes over the entry point's reachable call chain."""
    root = Path(root).resolve()
    definitions = _source_definitions(root)
    entry = _entry_definition(definitions, root, entry_point)
    if entry is None:
        return ()
    sites: list[MutationSite] = []
    visited: set[tuple[str, str]] = set()
    pending: list[tuple[_SourceDefinition, tuple[str, ...]]] = [(entry, (entry.qualname,))]
    while pending:
        definition, chain = pending.pop(0)
        key = (str(definition.path), definition.qualname)
        if key in visited:
            continue
        visited.add(key)
        sites.extend(_scan_definition(root, definition, chain))
        for child in _definition_calls(definition, definitions):
            pending.append((child, (*chain, child.qualname)))
    return tuple(sites)


find_mutation_sites = mutation_sites


_PRESERVATION_RE = re.compile(
    r"\b(?:preserv\w*|replay\w*|equivalent\w*|unchanged|not\s+mutat\w*|"
    r"do\s+not\s+mutat\w*|same\s+result\w*|idempotent)\b", re.IGNORECASE)


def has_preservation_clause(clause: object) -> bool:
    return bool(_PRESERVATION_RE.search(str(clause or "")))


def fixture_from_public_test(source: object, entry_point: str | None = None) -> dict[str, object] | None:
    """Extract the first literal entry-point call from a public test snippet."""
    if isinstance(source, Mapping):
        candidate = source.get("fixture", source)
        if isinstance(candidate, Mapping) and "args" in candidate:
            return copy.deepcopy(dict(candidate))
        if isinstance(candidate, Mapping):
            return {"args": [copy.deepcopy(dict(candidate))], "input_index": 0}
        return None
    if isinstance(source, Path):
        try:
            source = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None
    if not isinstance(source, str):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    values: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    leaf = str(entry_point or "").split(":", 1)[-1].split(".")[-1]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = (node.func.id if isinstance(node.func, ast.Name)
                  else node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if leaf and called != leaf:
            continue
        if not node.args:
            continue
        try:
            argument = ast.literal_eval(node.args[0])
        except (ValueError, TypeError, SyntaxError):
            argument = values.get(node.args[0].id) if isinstance(node.args[0], ast.Name) else None
        if argument is None:
            continue
        return {"args": [copy.deepcopy(argument)], "input_index": 0}
    return None


def bind_input_preservation(clause: str = "", entry_point: str | None = None, *,
                            fixture: object = None, root: str | Path | None = None,
                            public_test_source: object = None,
                            mutation_site_records: Sequence[MutationSite] | None = None,
                            obligation_id: str = "input-preservation",
                            check_id_prefix: str | None = None,
                            selection: object = (), targets: object = (),
                            coupled_obligations: object = (), clause_span: object = None,
                            interpretation: str = VALIDATED) -> tuple[CheckDefinition, ...]:
    """Create the paired host checks required by an input-preservation clause."""
    if not has_preservation_clause(clause) or not isinstance(entry_point, str):
        return ()
    sites = tuple(mutation_site_records or ())
    if not sites and root is not None:
        sites = mutation_sites(root, entry_point)
    if fixture is None:
        fixture = fixture_from_public_test(public_test_source, entry_point)
    if fixture is None:
        return ()
    if isinstance(fixture, Mapping) and "args" not in fixture:
        fixture = {"args": [copy.deepcopy(dict(fixture))], "input_index": 0}
    if not isinstance(fixture, Mapping) or not isinstance(fixture.get("args"), list):
        return ()
    site_payload = [site.to_dict() if isinstance(site, MutationSite) else copy.deepcopy(site)
                    for site in sites]
    provenance = {
        "kind": "input_preservation_binding",
        "entry_point": entry_point,
        "clause": str(clause),
        "clause_span": copy.deepcopy(clause_span),
        "mutation_sites": site_payload,
    }
    prefix = str(check_id_prefix or obligation_id)
    common = {
        "entry_point": entry_point,
        "fixture": copy.deepcopy(dict(fixture)),
        "expected": {"preserved": True},
        "provenance": provenance,
        "oracle_basis": "TICKET",
        "fixture_id": f"{prefix}:input-preservation",
        "selection": selection,
        "coupled_obligations": coupled_obligations,
        "interpretation": interpretation,
    }
    return (
        CheckDefinition(f"{prefix}.PRESERVE_INPUT", obligation_id,
                        "PRESERVE_INPUT", **common),
        CheckDefinition(f"{prefix}.REPLAY_EQUIVALENT", obligation_id,
                        "REPLAY_EQUIVALENT", **{**common,
                        "fixture_id": f"{prefix}:replay-equivalent"}),
    )


def input_preservation_bundle(*, task_id: str, clause: str, entry_point: str,
                             fixture: object = None, root: str | Path | None = None,
                             public_test_source: object = None,
                             mutation_site_records: Sequence[MutationSite] | None = None,
                             selection: object = (), targets: object = (),
                             coupled_obligations: object = (), clause_span: object = None,
                             obligation_id: str = "input-preservation") -> CheckBundle | None:
    checks = bind_input_preservation(
        clause, entry_point, fixture=fixture, root=root,
        public_test_source=public_test_source,
        mutation_site_records=mutation_site_records,
        obligation_id=obligation_id, selection=selection, targets=targets,
        coupled_obligations=coupled_obligations, clause_span=clause_span,
    )
    if not checks:
        return None
    check_ids = tuple(check.check_id for check in checks)
    spec = ObligationSpec(
        obligation_id, clause_span or {"text": clause}, VALIDATED, targets,
        check_ids, clause, checks[0].provenance, coupled_obligations,
    )
    return CheckBundle(task_id, (spec,), checks, "<dynamic-input-preservation>",
                       {"kind": "dynamic_input_preservation",
                        "mutation_sites": copy.deepcopy(checks[0].provenance.get("mutation_sites", ()))})


def augment_input_preservation_bundle(bundle: CheckBundle | None, root: str | Path, *,
                                      public_test_source: object = None) -> CheckBundle | None:
    """Add typed preservation/replay checks to a frozen public check bundle.

    Existing public check files remain authoritative for their ordinary checks.
    This augmentation only fills the old template gap: a preservation clause
    with a reachable caller-alias mutation gets the paired host checks, and an
    ambiguous replay clause becomes validated once those checks are bound.
    """
    if bundle is None:
        return None
    root = Path(root).resolve()
    original_checks = tuple(bundle.checks)
    checks = list(original_checks)
    existing_ids = {check.check_id for check in checks}
    changed_specs: list[ObligationSpec] = []
    for spec in bundle.specs:
        if not has_preservation_clause(spec.clause):
            changed_specs.append(spec)
            continue
        same_spec = [check for check in original_checks if check.obligation_id == spec.obligation_id]
        candidates = list(same_spec)
        if not candidates:
            target_set = set(spec.targets)
            candidates = [check for check in original_checks
                          if check.entry_point and (
                              check.entry_point.split(":", 1)[0].replace(".", "/") + ".py" in target_set
                              or any(str(target).endswith(
                                  check.entry_point.split(":", 1)[0].replace(".", "/") + ".py")
                                     for target in target_set))]
        bound: list[CheckDefinition] = []
        for candidate in candidates:
            if not candidate.entry_point:
                continue
            sites = mutation_sites(root, candidate.entry_point)
            if candidate.template != "PRESERVE_INPUT" and not sites:
                continue
            fixture = candidate.fixture if isinstance(candidate.fixture, Mapping) else None
            prefix = f"{spec.obligation_id}.{candidate.entry_point.rsplit(':', 1)[-1].replace('.', '_')}"
            generated = bind_input_preservation(
                spec.clause, candidate.entry_point, fixture=fixture, root=root,
                public_test_source=public_test_source, mutation_site_records=sites,
                obligation_id=spec.obligation_id, check_id_prefix=prefix,
                selection=candidate.selection, targets=spec.targets,
                coupled_obligations=spec.coupled_obligations,
                clause_span=spec.clause_span,
            )
            for item in generated:
                if item.check_id not in existing_ids:
                    bound.append(item)
                    existing_ids.add(item.check_id)
            if bound:
                # One entry path is sufficient for the paired contract; its
                # mutation-site provenance names every reachable write.
                break
        if bound:
            checks.extend(bound)
            acceptance = tuple(dict.fromkeys((*spec.acceptance_policy,
                                               *(check.check_id for check in bound))))
            provenance = copy.deepcopy(spec.provenance)
            provenance["dynamic_input_preservation"] = True
            provenance["mutation_sites"] = copy.deepcopy(
                bound[0].provenance.get("mutation_sites", ()))
            changed_specs.append(ObligationSpec(
                spec.obligation_id, spec.clause_span,
                VALIDATED if spec.interpretation == AMBIGUOUS else spec.interpretation,
                spec.targets, acceptance, spec.clause, provenance,
                spec.coupled_obligations,
                coverage_required=spec.coverage_required,
                coverage_disposition=spec.coverage_disposition,
                contract_cells=spec.contract_cells,
            ))
        else:
            changed_specs.append(spec)
    if len(checks) == len(original_checks) and changed_specs == list(bundle.specs):
        return bundle
    return CheckBundle(
        bundle.task_id, tuple(changed_specs), tuple(checks), bundle.path,
        {**copy.deepcopy(bundle.provenance), "input_preservation_augmented": True},
        bundle.contract_cells,
    )


def workspace_snapshot(root: str | Path, *, binding: object = None) -> str:
    """Hash every visible workspace file, with cache/bytecode noise excluded."""
    root = Path(root).resolve()
    rows: list[dict[str, object]] = []
    ignored = {".git", "__pycache__", ".pytest_cache", ".test-tmp"}
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if any(part in ignored for part in relative.parts):
                continue
            try:
                content = path.read_bytes()
            except OSError:
                continue
            rows.append({
                "path": relative.as_posix(),
                "sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
                "size": len(content),
            })
    # Binding is included only as an explicit environment/workspace context;
    # it cannot replace the file inventory above.
    return digest({"files": rows, "binding": binding})


def _check_path(task_id: str) -> Path:
    if not isinstance(task_id, str) or not task_id or "/" in task_id or "\\" in task_id:
        raise ValueError("task id is not a safe check-file name")
    return CHECK_DIR / (task_id + ".json")


def check_file_path(task_id: str) -> Path | None:
    path = _check_path(task_id)
    return path if path.is_file() else None


def check_bundle_text(task_id: str) -> tuple[str, str] | None:
    """Return (source_label, json_text) for a frozen bundle, or None.

    Frozen releases are Python-only, so checks/*.json never exists inside one; the generated
    kernel_check_bundles module is the in-release copy (drift-tested against checks/*.json).
    A present checks/ file wins so source-tree edits are visible before regeneration.
    """
    if not static_bundles_enabled():
        return None
    path = check_file_path(task_id)
    if path is not None:
        return ("checks/" + path.name, path.read_text(encoding="utf-8"))
    try:
        from kernel_check_bundles import BUNDLES
    except ImportError:
        return None
    text = BUNDLES.get(task_id)
    return ("kernel_check_bundles:" + task_id, text) if isinstance(text, str) else None


def _specs_from_checks(checks: Sequence[CheckDefinition]) -> tuple[ObligationSpec, ...]:
    grouped: dict[str, list[str]] = {}
    for check in checks:
        grouped.setdefault(check.obligation_id, []).append(check.check_id)
    return tuple(ObligationSpec(ident, "", VALIDATED, (), ids)
                 for ident, ids in grouped.items())


def load_check_bundle(task_id: str, *, public_material: Mapping[str, object] | None = None
                      ) -> CheckBundle | None:
    """Load only a frozen public binding file; absent means no verifier."""
    found = check_bundle_text(task_id)
    if found is None:
        return None
    label, text = found
    try:
        raw = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid host check bundle {label}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema") != CHECK_SCHEMA:
        raise ValueError("host check file has the wrong schema")
    checks_raw = raw.get("checks", ())
    checks = tuple(CheckDefinition.from_mapping(item) for item in checks_raw
                   if isinstance(item, Mapping))
    specs_raw = raw.get("obligations", ())
    specs = tuple(ObligationSpec.from_mapping(item) for item in specs_raw
                  if isinstance(item, Mapping))
    if not specs:
        specs = _specs_from_checks(checks)
    by_id = {spec.obligation_id: spec for spec in specs}
    for check in checks:
        if check.obligation_id not in by_id:
            by_id[check.obligation_id] = ObligationSpec(
                check.obligation_id, check.provenance.get("clause_span", ""),
                check.interpretation, (), (check.check_id,),
                provenance=check.provenance,
                coupled_obligations=check.coupled_obligations)
    specs = tuple(by_id.values())
    raw_cells = raw.get("contract_cells", raw.get("cells", ()))
    cells: list[ContractCell] = []
    if isinstance(raw_cells, Mapping):
        raw_cells = raw_cells.values()
    if isinstance(raw_cells, Sequence) and not isinstance(raw_cells, (str, bytes, bytearray)):
        cells.extend(ContractCell.from_mapping(item) for item in raw_cells
                     if isinstance(item, Mapping))
    for check in checks:
        if check.contract_cell is not None and check.contract_cell not in cells:
            cells.append(check.contract_cell)
    for spec in specs:
        for cell in spec.contract_cells:
            if cell not in cells:
                cells.append(cell)
    generator_events: list[dict[str, object]] = []
    generator_rejections: list[dict[str, object]] = []
    if public_material is not None:
        from kernel_check_generator_fixes import (
            enabled as generator_fixes_enabled, enforce_bundle,
        )
        if generator_fixes_enabled():
            material = dict(public_material)
            reconciled = enforce_bundle(
                CheckBundle(str(raw.get("task_id", task_id)), specs, checks, label,
                            _copy_mapping(raw.get("provenance", {})), tuple(cells)),
                brief=material.get("brief", ""),
                sources=material.get("sources", {}),
                tests=material.get("tests", {}),
                requirements_ledger=material.get(
                    "requirements_ledger", material.get("coverage_dispositions")),
            )
            checks = tuple(reconciled.checks)
            specs = tuple(reconciled.specs)
            cells = list(reconciled.contract_cells)
            reconciled_provenance = reconciled.provenance
            generator_events = [dict(event) for event in
                                reconciled_provenance.get("generator_events", ())
                                if isinstance(event, Mapping)]
            generator_rejections = [dict(item) for item in
                                    reconciled_provenance.get("generator_policy", {}).get(
                                        "rejected_checks", ())
                                    if isinstance(item, Mapping)]
            raw = {**dict(raw), "provenance": reconciled_provenance}
    provenance = _copy_mapping(raw.get("provenance", {}))
    if generator_events or generator_rejections:
        policy = _copy_mapping(provenance.get("generator_policy", {}))
        policy.update({
            "schema": "stateframe-check-generator-policy/1",
            "enabled": True,
            "rejected_checks": generator_rejections,
        })
        provenance.update({
            "generator_events": generator_events,
            "generator_rejections": generator_rejections,
            "generator_policy": policy,
        })
    return CheckBundle(
        str(raw.get("task_id", task_id)), specs, checks, label,
        provenance, tuple(cells))


def load_checks(task_id: str, *, public_material: Mapping[str, object] | None = None
                ) -> tuple[CheckDefinition, ...]:
    bundle = load_check_bundle(task_id, public_material=public_material)
    return bundle.checks if bundle is not None else ()


def load_obligation_specs(task_id: str) -> tuple[ObligationSpec, ...]:
    bundle = load_check_bundle(task_id)
    return bundle.specs if bundle is not None else ()


def _definition_for(check_id: str, definitions: object) -> CheckDefinition | None:
    if isinstance(definitions, Mapping):
        candidate = definitions.get(check_id)
        if isinstance(candidate, CheckDefinition):
            return candidate
        if isinstance(candidate, Mapping):
            return CheckDefinition.from_mapping(candidate)
    if isinstance(definitions, Iterable) and not isinstance(definitions, (str, bytes)):
        for candidate in definitions:
            if isinstance(candidate, CheckDefinition) and candidate.check_id == check_id:
                return candidate
            if isinstance(candidate, Mapping) and str(candidate.get("check_id", candidate.get("id", ""))) == check_id:
                return CheckDefinition.from_mapping(candidate)
    return None


def _receipt_for(value: object) -> CheckReceipt | None:
    if isinstance(value, CheckReceipt):
        return value
    if isinstance(value, Mapping):
        try:
            return CheckReceipt.from_mapping(value)
        except (TypeError, ValueError):
            return None
    return None


def _receipts_for(check_id: str, receipts: object) -> list[CheckReceipt]:
    if isinstance(receipts, Mapping):
        values = receipts.values() if not {"check_id", "status", "result_class"}.intersection(receipts) else (receipts,)
    elif isinstance(receipts, Iterable) and not isinstance(receipts, (str, bytes)):
        values = receipts
    else:
        values = ()
    result = []
    for value in values:
        receipt = _receipt_for(value)
        if receipt is not None and receipt.check_id == check_id:
            result.append(receipt)
    return result


def _localization_for(spec: ObligationSpec, receipts: Sequence[CheckReceipt],
                      checks: object) -> dict[str, object] | None:
    """Summarize an observed edge without treating a guess as a location."""
    executed = [receipt for receipt in receipts if receipt.execution == EXECUTED]
    if not executed:
        return None
    good = [receipt for receipt in executed if receipt.status == PASS]
    bad = [receipt for receipt in executed if receipt.status == COUNTEREXAMPLE]
    if not good and not bad:
        return None
    targets = list(spec.targets)
    for receipt in receipts:
        definition = _definition_for(receipt.check_id, checks)
        if definition is None or not definition.entry_point:
            continue
        module = definition.entry_point.split(":", 1)[0].replace(".", "/") + ".py"
        if module not in targets:
            targets.append(module)
    evidence = [receipt.evidence_ref for receipt in (*good, *bad) if receipt.evidence_ref]
    confidence = "UNLOCALIZED"
    if good and bad and evidence:
        confidence = "CONFIRMED_EDGE"
    elif bad:
        confidence = "SUSPECT_REGION"
    return {
        "last_good_obs": copy.deepcopy(good[-1].observed if good else None),
        "first_bad_obs": copy.deepcopy(bad[0].observed if bad else None),
        "candidate_targets": list(dict.fromkeys(targets)),
        "confidence": confidence,
        "evidence_refs": list(dict.fromkeys(evidence)),
    }


def _identity_failure(receipts: object, identity: object) -> str | None:
    """Return a typed reducer reason when check evidence is not this run's."""
    if identity is None:
        return None
    from kernel_verdict_identity import (
        IdentityError, IdentityMismatch, MixedCodeHashError, MixedReleaseError,
        enabled as identity_enabled, validate_batch,
    )
    if not identity_enabled():
        return None
    if isinstance(receipts, Mapping):
        values = (receipts.values() if not {"check_id", "status", "result_class", "identity"}
                  .intersection(receipts) else (receipts,))
    elif isinstance(receipts, Iterable) and not isinstance(receipts, (str, bytes)):
        values = receipts
    else:
        values = (receipts,)
    try:
        validate_batch(list(values), current=identity)
    except MixedReleaseError:
        return "MIXED_RELEASE_ID"
    except MixedCodeHashError:
        return "STALE_CODE_HASH"
    except IdentityMismatch:
        return "STALE_CODE_HASH"
    except IdentityError:
        return "MISSING_RUN_IDENTITY"
    return None


def _oracle_assessment_context(spec: ObligationSpec, definition: object, *,
                               ticket: object = None, ticket_text: object = None,
                               ledger: object = None, public_api: object = None,
                               public_contract: object = None, contract: object = None,
                               sources: object = None,
                               public_context: object = None) -> dict[str, object]:
    """Resolve one authoritative public context, with clause as last fallback."""
    spec_provenance = spec.provenance if isinstance(spec.provenance, Mapping) else {}
    definition_provenance = (
        definition.provenance if isinstance(definition, CheckDefinition)
        else definition.get("provenance", {}) if isinstance(definition, Mapping) else {}
    )
    if not isinstance(definition_provenance, Mapping):
        definition_provenance = {}
    context = public_context if isinstance(public_context, Mapping) else {}

    def choose(current: object, *names: str) -> object:
        if current is not None:
            return current
        for container in (context, spec_provenance, definition_provenance):
            for name in names:
                value = container.get(name) if isinstance(container, Mapping) else None
                if value is not None:
                    return value
        return None

    authoritative_ticket = choose(
        ticket if ticket is not None else ticket_text,
        "ticket", "ticket_text", "public_ticket", "public_brief", "brief", "brief_text",
    )
    if authoritative_ticket is None:
        authoritative_ticket = spec.clause
    return {
        "ticket": authoritative_ticket,
        "ledger": choose(ledger, "ledger", "requirements_ledger"),
        "public_api": choose(public_api, "public_api", "repo_public_api", "api"),
        "public_contract": choose(
            public_contract if public_contract is not None else contract,
            "public_contract", "repo_contract", "contract",
        ),
        "sources": choose(sources, "sources", "public_sources", "public_tests",
                           "public_test_source", "public_test_src", "test_source", "test_src"),
    }


def certifying_authored_receipt(receipts: object, obligation_id: str,
                                revision: object, *, clause: object = None,
                                clause_span: object = None
                                ) -> Mapping[str, object] | None:
    """Return a solver-authored check receipt that honestly certifies ``obligation_id``.

    Honesty gates (kernel_authored_check.execute_authored_check): the host ran the
    frozen check on BOTH sides, it FAILED on the unfixed parent and PASSED on the
    candidate, both sides reached the observation with integrity, and the host
    marked ``certifies_clause``.  Vacuous (green on parent), never-run, observation
    -only, simulated or stale (other candidate revision) receipts never certify.

    Meaning gate (W3/W3B): ``clause`` and ``clause_span`` are HOST obligation
    data.  The receipt must carry the dispatcher's binding to this exact clause
    identity, allowing only a verified frame/bundle id alias.  An arbitrary or
    foreign id, a contained-but-wrong-clause quote, drifted text, or missing host
    span never certifies (fail closed).
    """
    from kernel_authored_check import binding_certifies
    current = str(revision or "").strip()
    if (not current or not str(clause or "").strip()
            or isinstance(receipts, (str, bytes)) or not isinstance(receipts, Iterable)):
        return None
    for row in receipts:
        if not isinstance(row, Mapping):
            continue
        provenance = row.get("provenance")
        if not (isinstance(provenance, Mapping) and provenance.get("kind") == "AUTHORED_CHECK"
                and provenance.get("source") == "host_frozen_pair"
                and provenance.get("simulation") is not True):
            continue
        if not binding_certifies(row, obligation_id, clause, clause_span=clause_span):
            continue
        if not (str(row.get("receipt_id") or "").strip()
                and str(row.get("execution_identity") or "").strip()
                and row.get("simulation") is not True):
            continue
        if not (row.get("certifies_clause") is True and row.get("result") == "BEHAVIOR_FIXED"
                and row.get("status") == PASS and row.get("observation_only") is not True
                and row.get("parent_outcome") == "BEHAVIOR_FAIL" and row.get("candidate_outcome") == PASS
                and row.get("reached_observation") is True and row.get("integrity") is True
                and str(row.get("parent_revision") or "") != current
                and str(row.get("candidate_revision") or "") == current):
            continue
        return row
    return None


def assess_obligation(spec: ObligationSpec | Mapping[str, object], receipts: object = (), *,
                      snapshot: object = None, environment: object = None,
                      checks: object = (), identity: object = None,
                      ticket: object = None, ticket_text: object = None,
                      ledger: object = None, public_api: object = None,
                      public_contract: object = None, contract: object = None,
                      sources: object = None, public_context: object = None,
                      authored_revision: object = None) -> ObligationAssessment:
    """Purely reduce one spec and already-produced receipts.

    ``authored_revision`` is the live candidate product revision; a certifying
    solver-authored check receipt closes a ``MISSING_CHECKS`` obligation only
    when its candidate revision equals it (absent -> fail closed).
    """
    if not isinstance(spec, ObligationSpec):
        spec = ObligationSpec.from_mapping(spec)
    check_ids = spec.acceptance_policy
    assessment_coverage = {
        "coverage_required": spec.coverage_required,
        "coverage_disposition": spec.coverage_disposition,
    }
    if spec.interpretation != VALIDATED:
        reason = "AMBIGUOUS_CONTRACT" if spec.interpretation == AMBIGUOUS else "UNVALIDATED_INTERPRETATION"
        return ObligationAssessment(spec.obligation_id, UNKNOWN, reason, check_ids=check_ids,
                                    clause=spec.clause,
                                    coupled_obligations=spec.coupled_obligations,
                                    **assessment_coverage)
    # W3: run identity is validated BEFORE any shortcut; an empty acceptance
    # policy must never let stale/foreign-run evidence certify.
    identity_reason = _identity_failure(receipts, identity)
    if identity_reason is not None:
        return ObligationAssessment(
            spec.obligation_id, UNKNOWN, identity_reason, check_ids=check_ids,
            clause=spec.clause, coupled_obligations=spec.coupled_obligations,
            **assessment_coverage)
    if not check_ids:
        authored = certifying_authored_receipt(receipts, spec.obligation_id, authored_revision,
                                               clause=spec.clause, clause_span=spec.clause_span)
        if authored is not None:
            return ObligationAssessment(
                spec.obligation_id, SATISFIED, "PASS",
                check_ids=(str(authored.get("check_id") or ""),),
                evidence_refs=(str(authored.get("receipt_id") or ""),),
                clause=spec.clause, coupled_obligations=spec.coupled_obligations,
                **assessment_coverage)
        return ObligationAssessment(spec.obligation_id, UNKNOWN, "MISSING_CHECKS",
                                    clause=spec.clause,
                                    coupled_obligations=spec.coupled_obligations,
                                    **assessment_coverage)
    current_snapshot = _identity(snapshot)
    current_environment = _identity(environment)
    selected: list[CheckReceipt] = []
    evidence: list[str] = []
    observed = None
    expected = None
    unknown_reasons: list[str] = []
    counterexamples: list[CheckReceipt] = []
    for check_id in check_ids:
        candidates = _receipts_for(check_id, receipts)
        definition = _definition_for(check_id, checks)
        if (definition is not None
                and os.environ.get("STATEFRAME_ORACLE_GROUNDING", "off").strip().lower() == "on"):
            try:
                from kernel_oracle_grounding import ground_expectation
                grounding_context = _oracle_assessment_context(
                    spec, definition, ticket=ticket, ticket_text=ticket_text,
                    ledger=ledger, public_api=public_api,
                    public_contract=public_contract, contract=contract,
                    sources=sources, public_context=public_context)
                grounding = ground_expectation(
                    definition, **grounding_context,
                )
                if grounding.unknown:
                    unknown_reasons.append("UNBOUND_OBSERVABLE")
                    continue
            except (ImportError, TypeError, ValueError):
                pass
        if (definition is not None
                and str(definition.oracle_basis).upper() == ADVISORY):
            unknown_reasons.append("ADVISORY_ONLY")
            continue
        if not _definition_evidence_complete(definition):
            unknown_reasons.append("AMBIGUOUS_CONTRACT")
            continue
        expected_check_digest = definition.digest if definition is not None else None
        expected_fixture_digest = definition.fixture_digest if definition is not None else _identity(None)
        current: list[CheckReceipt] = []
        stale = False
        for receipt in candidates:
            if receipt.snapshot_digest != current_snapshot or receipt.environment_digest != current_environment:
                stale = True
                continue
            if expected_check_digest is not None and receipt.check_digest != expected_check_digest:
                stale = True
                continue
            if definition is not None and receipt.fixture_digest != expected_fixture_digest:
                stale = True
                continue
            current.append(receipt)
        if not current:
            unknown_reasons.append("STALE" if stale else "MISSING_CHECKS")
            continue
        selected.extend(current)
        evidence.extend(receipt.evidence_ref for receipt in current if receipt.evidence_ref)
        executable = [receipt for receipt in current if receipt.execution == EXECUTED]
        if any(receipt_is_executed_counterexample(receipt) for receipt in executable):
            counterexamples.extend(receipt for receipt in executable
                                    if receipt_is_executed_counterexample(receipt))
        elif any(receipt.status == PASS for receipt in executable):
            passes = [receipt for receipt in executable if receipt.status == PASS]
            distinct_passes = {_canonical((receipt.observed, receipt.expected))
                               for receipt in passes}
            if len(distinct_passes) > 1:
                unknown_reasons.append("CONFLICTING_RECEIPTS")
        for receipt in current:
            if receipt.execution == EXECUTED:
                if receipt.status == HARNESS_ERROR:
                    unknown_reasons.append("HARNESS_ERROR")
                elif receipt.status == NOT_EXECUTED:
                    unknown_reasons.append(
                        receipt.reason if receipt.reason in {
                            UNBOUND_ENTRYPOINT, AMBIGUOUS_CONTRACT, UNBOUND_INVOCATION,
                        } else "NOT_EXECUTED")
                continue
            if receipt.execution == SETUP_ERROR and receipt.status == HARNESS_ERROR:
                unknown_reasons.append("HARNESS_ERROR")
            elif receipt.execution == SETUP_ERROR:
                unknown_reasons.append(SETUP_ERROR)
            elif receipt.execution == SKIPPED:
                if receipt.reason in {UNBOUND_ENTRYPOINT, AMBIGUOUS_CONTRACT, UNBOUND_INVOCATION}:
                    unknown_reasons.append(receipt.reason)
                elif receipt.status == NOT_EXECUTED:
                    unknown_reasons.append("NOT_EXECUTED")
                else:
                    unknown_reasons.append(SKIPPED)
            else:
                # The constructor validates execution, but retain a typed
                # uncertainty if a future receipt class reaches this reducer.
                unknown_reasons.append("NOT_EXECUTED")
        chosen = next((receipt for receipt in executable
                       if receipt_is_executed_counterexample(receipt)),
                      next((receipt for receipt in current if receipt.status == PASS), current[-1]))
        if observed is None:
            observed, expected = chosen.observed, chosen.expected
    if counterexamples:
        chosen = counterexamples[0]
        localization = _localization_for(spec, selected, checks)
        return ObligationAssessment(
            spec.obligation_id, UNSATISFIED, "COUNTEREXAMPLE", check_ids,
            tuple(selected), chosen.observed, chosen.expected,
            tuple(dict.fromkeys(evidence)), spec.clause,
            _location_hint(spec, checks), spec.coupled_obligations, localization,
            **assessment_coverage)
    if unknown_reasons:
        reason = next((item for item in (
            "UNBOUND_OBSERVABLE", "STALE", "MISSING_CHECKS", "HARNESS_ERROR", SETUP_ERROR, SKIPPED,
            "NOT_EXECUTED",
            UNBOUND_ENTRYPOINT, UNBOUND_INVOCATION, AMBIGUOUS_CONTRACT,
            "CONFLICTING_RECEIPTS", "ADVISORY_ONLY"
        ) if item in unknown_reasons), unknown_reasons[0])
        return ObligationAssessment(
            spec.obligation_id, UNKNOWN, reason, check_ids, tuple(selected), observed, expected,
            tuple(dict.fromkeys(evidence)), spec.clause,
            _location_hint(spec, checks), spec.coupled_obligations,
            _localization_for(spec, selected, checks), **assessment_coverage)
    localization = _localization_for(spec, selected, checks)
    return ObligationAssessment(
        spec.obligation_id, SATISFIED, "PASS", check_ids, tuple(selected), observed, expected,
        tuple(dict.fromkeys(evidence)), spec.clause,
        _location_hint(spec, checks), spec.coupled_obligations, localization,
        **assessment_coverage)


def reduce_obligation(spec: ObligationSpec | Mapping[str, object], receipts: object = (), **kwargs) -> ObligationAssessment:
    """Compatibility name for the pure obligation reducer."""
    return assess_obligation(spec, receipts, **kwargs)


def assess_obligations(specs: object, receipts: object = (), **kwargs) -> tuple[ObligationAssessment, ...]:
    if isinstance(specs, Mapping):
        values = specs.values()
    elif isinstance(specs, Iterable) and not isinstance(specs, (str, bytes)):
        values = specs
    else:
        values = ()
    return tuple(assess_obligation(spec, receipts, **kwargs) for spec in values)


def _location_hint(spec: ObligationSpec, checks: object) -> str | None:
    if spec.targets:
        return spec.targets[0]
    for check_id in spec.acceptance_policy:
        definition = _definition_for(check_id, checks)
        if definition is None or not definition.entry_point:
            continue
        entry = definition.entry_point.split(":", 1)[0].replace(".", "/")
        return entry + (".py" if not entry.endswith(".py") else "")
    return None


def _menu_names(menu: object) -> tuple[str, ...]:
    if isinstance(menu, Mapping):
        menu = menu.values()
    if isinstance(menu, str):
        menu = (menu,)
    if not isinstance(menu, Iterable):
        return ()
    names: list[str] = []
    for item in menu:
        if isinstance(item, str):
            name = item
        elif isinstance(item, Mapping):
            fn = item.get("function")
            name = fn.get("name") if isinstance(fn, Mapping) else item.get("name")
        else:
            name = getattr(item, "name", None)
        if isinstance(name, str) and name.strip() and name not in names:
            names.append(name)
    return tuple(names)


def _definition_for_assessment(assessment: ObligationAssessment, definitions: object) -> CheckDefinition | None:
    for check_id in assessment.check_ids:
        candidate = _definition_for(check_id, definitions)
        if candidate is not None:
            return candidate
    return None


BLOCKER_CLASSES = frozenset({
    "PRODUCT_FAILURE", "EVIDENCE_MISSING", "EVIDENCE_STALE", "HOST_UNAVAILABLE",
})

_HOST_BLOCKER_REASONS = frozenset({
    "HARNESS_ERROR", "SETUP_ERROR", "SKIPPED", "UNRUNNABLE", "UNSUPPORTED",
    "ENVIRONMENT", "HOST_UNAVAILABLE",
})
_STALE_BLOCKER_REASONS = frozenset({
    "STALE", "STALE_EVIDENCE", "CHECK_NOT_CURRENT", "REVISION_MISMATCH",
})
_MISSING_BLOCKER_REASONS = frozenset({
    "MISSING_CHECKS", "NO_BOUND_CHECK", "NO_VERIFIER", "NOT_EXECUTED",
    "REQUIRED_EVIDENCE_GAP",
})


def classify_blocker(assessment: ObligationAssessment, *, host_available: bool = True,
                     current_binding: object = None) -> dict[str, object]:
    """Classify one open obligation without changing its terminal truth."""
    reason = str(assessment.reason or "").strip().upper()
    verdict = str(assessment.verdict or "").strip().upper()
    if not host_available or reason in _HOST_BLOCKER_REASONS:
        blocker = "HOST_UNAVAILABLE"
    elif reason in _STALE_BLOCKER_REASONS:
        blocker = "EVIDENCE_STALE"
    elif reason in _MISSING_BLOCKER_REASONS or verdict == UNKNOWN:
        blocker = "EVIDENCE_MISSING"
    elif verdict == UNSATISFIED or reason in {"COUNTEREXAMPLE", "VIOLATED", "PRODUCT_FAILURE"}:
        blocker = "PRODUCT_FAILURE"
    else:
        blocker = "EVIDENCE_MISSING"
    binding = copy.deepcopy(current_binding) if isinstance(current_binding, Mapping) else current_binding
    return {
        "schema": "stateframe-blocker-classification/1",
        "class": blocker,
        "obligation_id": assessment.obligation_id,
        "reason": reason or "UNKNOWN",
        "verdict": verdict or UNKNOWN,
        "current_binding": binding,
        "validated": blocker in BLOCKER_CLASSES and binding is not None,
    }


def resolver_actions(
        assessment: ObligationAssessment, definition: CheckDefinition | None, *,
        menu: object, authored_proposal: object = None, host_available: bool = True,
        current_binding: object = None) -> tuple[dict[str, object], ...]:
    """Return bounded, dispatchable routes for the classified blocker."""
    names = set(_menu_names(menu))
    blocker = classify_blocker(
        assessment, host_available=host_available, current_binding=current_binding)
    selection = list(definition.selection) if definition is not None else []
    path = assessment.location_hint or (
        definition.entry_point.split(":", 1)[0].replace(".", "/") + ".py"
        if definition is not None and definition.entry_point else None)
    routes: list[dict[str, object]] = []

    def add(tool: str, args: Mapping[str, object], clears: str) -> None:
        if tool not in names or not isinstance(args, Mapping) or not clears.strip():
            return
        routes.append({
            "tool": tool,
            "args": copy.deepcopy(dict(args)),
            "cost": 1,
            "clears": clears,
            "blocker": blocker["class"],
            "dispatchable": True,
        })

    if blocker["class"] == "PRODUCT_FAILURE":
        if path:
            add("edit", {"path": path}, "repair the current product counterexample")
            add("str_replace", {"path": path}, "repair the current product counterexample")
        if selection:
            add("run_tests", {"target": " ".join(selection), "selectors": selection},
                "recheck the repaired current candidate")
    elif blocker["class"] == "EVIDENCE_MISSING":
        proposal = authored_proposal
        if isinstance(proposal, Mapping):
            proposal_args = proposal.get("arguments", proposal)
            if isinstance(proposal_args, Mapping) and proposal_args:
                add("authored_check", proposal_args, "create the missing public-bound check")
        if path and "probe" in names:
            add("probe", {
                "clause_id": assessment.obligation_id,
                "entrypoint": definition.entry_point if definition is not None else path,
                "input": {},
                "expected": copy.deepcopy(assessment.expected),
                "rationale": "exercise the missing public clause witness",
            }, "produce the missing behavioral witness")
        if selection:
            add("run_tests", {"target": " ".join(selection), "selectors": selection},
                "execute the exact current public check")
    elif blocker["class"] == "EVIDENCE_STALE":
        if selection:
            add("run_tests", {"target": " ".join(selection), "selectors": selection},
                "refresh stale evidence on the current binding")
    elif blocker["class"] == "HOST_UNAVAILABLE":
        add("host_blocked", {"obligation_id": assessment.obligation_id},
            "record typed host unavailability without certifying")
    return tuple(routes)


def _action_for_gap(assessment: ObligationAssessment, definition: CheckDefinition | None,
                    *, menu: object, source_delivered: object = None,
                    duplicate_count: int = 0, authored_proposal: object = None,
                    host_available: bool = True,
                    current_binding: object = None) -> dict[str, object] | None:
    names = _menu_names(menu)
    typed_routes = resolver_actions(
        assessment, definition, menu=menu, authored_proposal=authored_proposal,
        host_available=host_available, current_binding=current_binding)
    path = assessment.location_hint or (definition.entry_point.split(":", 1)[0].replace(".", "/") + ".py"
                                        if definition is not None and definition.entry_point else None)
    selection = list(definition.selection) if definition is not None else []
    if duplicate_count >= 1:
        if path and "read" in names:
            return {"tool": "read", "arguments": {"path": path},
                    "reason": "foreground the current source for the focused obligation"}
        if path and "edit" in names:
            return {"tool": "edit", "arguments": {"path": path},
                    "reason": "repair the focused obligation after the reused receipt"}
        if path and "str_replace" in names:
            return {"tool": "str_replace", "arguments": {"path": path},
                    "reason": "repair the focused obligation after the reused receipt"}
        # A cached instance must never be turned into an identical test retry
        # merely because the menu lacks a source/edit affordance.
        return None
    if typed_routes:
        route = typed_routes[0]
        return {
            "tool": route["tool"],
            "arguments": copy.deepcopy(route["args"]),
            "reason": route["clears"],
            "cost": route["cost"],
            "clears": route["clears"],
            "blocker": route["blocker"],
        }
    if assessment.reason in {"MISSING_CHECKS", "STALE", "NOT_EXECUTED"} and "run_tests" in names and selection:
        return {"tool": "run_tests", "arguments": {"target": " ".join(selection),
                                                        "selectors": selection},
                "reason": "refresh the exact current public observation"}
    if assessment.verdict == UNSATISFIED:
        if path and "read" in names and not source_delivered:
            return {"tool": "read", "arguments": {"path": path},
                    "reason": "read the current source at the counterexample location"}
        if path and "edit" in names:
            return {"tool": "edit", "arguments": {"path": path},
                    "reason": "repair the focused counterexample"}
        if path and "str_replace" in names:
            return {"tool": "str_replace", "arguments": {"path": path},
                    "reason": "repair the focused counterexample"}
    if assessment.reason in {"HARNESS_ERROR", "CONFLICTING_RECEIPTS"} and "read" in names and path:
        return {"tool": "read", "arguments": {"path": path},
                "reason": "diagnose the host observation"}
    if "run_tests" in names and selection:
        return {"tool": "run_tests", "arguments": {"target": " ".join(selection),
                                                        "selectors": selection},
                "reason": "run the registered public check"}
    return None


def rank_obligations(assessments: Sequence[ObligationAssessment], *, ages: Mapping[str, int] | None = None,
                     duplicate_focus: str | None = None) -> list[ObligationAssessment]:
    """Rank actionable gaps: observation first, then counterexamples, with aging."""
    ages = ages or {}
    gaps = [assessment for assessment in assessments if assessment.blocking]

    def key(item: ObligationAssessment) -> tuple[int, int, int, int, str]:
        observation = item.reason in {"MISSING_CHECKS", "STALE", "NOT_EXECUTED"}
        counterexample = item.reason == "COUNTEREXAMPLE"
        severity = 3 if item.reason in {"HARNESS_ERROR", "CONFLICTING_RECEIPTS"} else 0
        age = min(99, int(ages.get(item.obligation_id, 0)))
        duplicate_count = max((receipt.duplicate_count for receipt in item.receipts), default=0)
        # The first duplicate keeps the same focus so its cached receipt can
        # foreground source/edit.  Only the second no-information observation
        # widens to another gap before applying the ordinary ranking.
        duplicate = 1 if (duplicate_focus and item.obligation_id == duplicate_focus
                          and duplicate_count >= 2) else 0
        return (duplicate, 0 if observation else 1 if counterexample else 2,
                -severity, -age, item.obligation_id)
    return sorted(gaps, key=key)


def needs_work_payload(assessments: Sequence[ObligationAssessment], *, definitions: object = (),
                       menu: object = (), ages: Mapping[str, int] | None = None,
                       source_delivered_at_revision: object = None,
                       duplicate_focus: str | None = None,
                       authored_proposal: object = None,
                       host_available: bool = True) -> dict[str, object] | None:
    ranked = rank_obligations(assessments, ages=ages, duplicate_focus=duplicate_focus)
    if not ranked:
        return None
    focus = ranked[0]
    definition = _definition_for_assessment(focus, definitions)
    duplicate_count = max((receipt.duplicate_count for receipt in focus.receipts), default=0)
    action = _action_for_gap(focus, definition, menu=menu,
                             source_delivered=source_delivered_at_revision,
                             duplicate_count=duplicate_count,
                             authored_proposal=authored_proposal,
                             host_available=host_available,
                             current_binding=source_delivered_at_revision)
    if duplicate_count >= 2 and duplicate_focus == focus.obligation_id:
        # rank_obligations already moved to another gap when one exists.  If
        # this is still the same gap, there is no productive information route
        # left for the current instance; the reducer will emit typed BLOCKED.
        action = None
    mode_value = {
        UNSATISFIED: REPAIR,
    }.get(focus.verdict, VERIFY if focus.reason in {"MISSING_CHECKS", "STALE", "NOT_EXECUTED"}
           else DIAGNOSE if focus.reason in {"HARNESS_ERROR", "CONFLICTING_RECEIPTS"}
           else REVIEW)
    remaining = sum(1 for item in assessments if item.blocking)
    counterexample_ref = next((receipt.evidence_ref for receipt in focus.receipts
                               if receipt_is_executed_counterexample(receipt)), None)
    payload = {
        "schema": "stateframe-needs-work/1",
        "mode": mode_value,
        "focus_obligation": focus.obligation_id,
        "observed": copy.deepcopy(focus.observed),
        "expected": copy.deepcopy(focus.expected),
        "location_hint": focus.location_hint,
        "coupled_preserve_constraints": list(focus.coupled_obligations),
        "remaining_count": remaining,
        "next_action": copy.deepcopy(action),
        "counterexample_ref": counterexample_ref,
        "receipt_status": (focus.receipts[0].display_status if focus.receipts else None),
        "reason": focus.reason,
        "clause": focus.clause,
        "duplicate_count": duplicate_count,
        "source_delivered_at_revision": source_delivered_at_revision,
    }
    blocker = classify_blocker(
        focus, host_available=host_available,
        current_binding=source_delivered_at_revision)
    payload["blocker"] = blocker
    payload["resolvers"] = list(resolver_actions(
        focus, definition, menu=menu, authored_proposal=authored_proposal,
        host_available=host_available, current_binding=source_delivered_at_revision))
    if precision_features_enabled() and definition is not None:
        payload["fixture"] = copy.deepcopy(definition.fixture)
    from kernel_model_profile import gap_list_relaxed
    if gap_list_relaxed():
        compact_gaps = []
        for priority, item in enumerate(ranked, 1):
            item_definition = _definition_for_assessment(item, definitions)
            item_duplicate_count = max(
                (receipt.duplicate_count for receipt in item.receipts), default=0)
            compact_gaps.append({
                "priority": priority,
                "obligation_id": item.obligation_id,
                "reason": item.reason,
                "clause": item.clause,
                "observed": copy.deepcopy(item.observed),
                "expected": copy.deepcopy(item.expected),
                "location_hint": item.location_hint,
                "next_action": copy.deepcopy(_action_for_gap(
                    item, item_definition, menu=menu,
                    source_delivered=source_delivered_at_revision,
                    duplicate_count=item_duplicate_count)),
                "counterexample_ref": next(
                    (receipt.evidence_ref for receipt in item.receipts
                     if receipt_is_executed_counterexample(receipt)), None),
                "duplicate_count": item_duplicate_count,
            })
        payload["gaps"] = compact_gaps
    return payload


def _assessment_for_packet(payload: Mapping[str, object],
                           assessments: Sequence[object]) -> ObligationAssessment | None:
    focus = str(payload.get("focus_obligation", ""))
    for item in assessments:
        if isinstance(item, ObligationAssessment) and item.obligation_id == focus:
            return item
        if isinstance(item, Mapping) and str(item.get("obligation_id", item.get("id", ""))) == focus:
            try:
                return ObligationAssessment(
                    focus, str(item.get("verdict", UNKNOWN)), str(item.get("reason", "")),
                    item.get("check_ids", ()), (), item.get("observed"), item.get("expected"),
                    item.get("evidence_refs", ()), str(item.get("clause", "")),
                    item.get("location_hint"), item.get("coupled_obligations", ()),
                    item.get("localization"))
            except (TypeError, ValueError):
                return None
    return None


def _packet_witness(assessment: ObligationAssessment | None) -> str:
    if assessment is None:
        return "NO_BOUNDED_HOST_WITNESS"
    for receipt in assessment.receipts:
        counterexample = receipt.counterexample
        if isinstance(counterexample, Mapping):
            for key in ("detail", "error", "message", "reason"):
                value = counterexample.get(key)
                if value:
                    return str(value)
            if counterexample:
                return json.dumps(counterexample, sort_keys=True, ensure_ascii=False)
        if receipt.reason and receipt.reason not in {PASS, COUNTEREXAMPLE}:
            return str(receipt.reason)
    return "NO_BOUNDED_HOST_WITNESS"


def repair_packet(payload: Mapping[str, object] | None, *,
                  assessments: Sequence[object] = (), definitions: object = (),
                  current_revision: object = None,
                  source_bytes: Mapping[str, str] | None = None,
                  source_revisions: Mapping[str, object] | None = None,
                  episode: Mapping[str, object] | None = None,
                  public_suite: Sequence[object] = (),
                  repair_checks: Sequence[object] = (),
                  edit_scope: Mapping[str, object] | None = None) -> dict[str, object] | None:
    """Build the ordered one-gap repair packet used after a host refusal."""
    if not isinstance(payload, Mapping) or not payload.get("focus_obligation"):
        return None
    assessment = _assessment_for_packet(payload, assessments)
    coupled = _strings(payload.get("coupled_preserve_constraints", ()))
    if assessment is not None:
        coupled = tuple(dict.fromkeys((*coupled, *assessment.coupled_obligations)))
    selected_receipt = None
    if assessment is not None:
        selected_receipt = next(
            (receipt for receipt in assessment.receipts
             if receipt_is_executed_counterexample(receipt)),
            assessment.receipts[-1] if assessment.receipts else None,
        )
    observed = (assessment.observed if assessment is not None else payload.get("observed"))
    expected = (assessment.expected if assessment is not None else payload.get("expected"))
    if selected_receipt is not None:
        observed = selected_receipt.observed if selected_receipt.observed is not None else observed
        expected = selected_receipt.expected if selected_receipt.expected is not None else expected
    selected_definition = (_definition_for(selected_receipt.check_id, definitions)
                           if selected_receipt is not None else None)
    if selected_definition is None and assessment is not None:
        selected_definition = _definition_for_assessment(assessment, definitions)
    files: list[dict[str, object]] = []
    for path, content in sorted((source_bytes or {}).items(), key=lambda pair: str(pair[0])):
        if not isinstance(content, str):
            continue
        normalized = str(path).replace("\\", "/").lstrip("./")
        files.append({
            "path": normalized,
            "revision": (source_revisions or {}).get(normalized, current_revision),
            "source_bytes": content,
            "source_digest": digest(content),
        })
    failing = {
        "host_run": {
            "check_ids": list(assessment.check_ids) if assessment is not None
            else list(payload.get("check_ids", ())),
            "receipt_id": selected_receipt.receipt_id if selected_receipt is not None else
            payload.get("counterexample_ref"),
            "result_class": selected_receipt.display_status if selected_receipt is not None else
            payload.get("receipt_status"),
            "snapshot": (selected_receipt.snapshot_digest
                          if selected_receipt is not None else None),
            "check_digest": (selected_receipt.check_digest
                              if selected_receipt is not None else
                              selected_definition.digest if selected_definition is not None else None),
            "fixture_digest": (selected_receipt.fixture_digest
                                if selected_receipt is not None else
                                selected_definition.fixture_digest
                                if selected_definition is not None else None),
            "environment_digest": (selected_receipt.environment_digest
                                    if selected_receipt is not None else None),
            "entry_point": (selected_definition.entry_point
                             if selected_definition is not None else None),
            "template": (selected_definition.template
                          if selected_definition is not None else None),
            "fixture_id": (selected_definition.fixture_id
                            if selected_definition is not None else None),
            "selection": (list(selected_definition.selection)
                           if selected_definition is not None else []),
        },
        "observed": copy.deepcopy(observed),
        "required": copy.deepcopy(expected),
        "witness": _packet_witness(assessment),
        "counterexample_ref": payload.get("counterexample_ref"),
    }
    localization = (assessment.localization if assessment is not None
                    else payload.get("localization"))
    if isinstance(localization, Mapping):
        # A confirmed edge is meaningful only when the receipt evidence is
        # present in the packet.  Otherwise keep the weaker host label.
        localization = copy.deepcopy(dict(localization))
        if (localization.get("confidence") == "CONFIRMED_EDGE"
                and not localization.get("evidence_refs")):
            localization["confidence"] = "SUSPECT_REGION"
        failing["localization"] = localization
    if precision_features_enabled():
        failing["fixture"] = copy.deepcopy(
            selected_definition.fixture if selected_definition is not None
            else (selected_receipt.counterexample.get("fixture")
                  if selected_receipt is not None
                  and isinstance(selected_receipt.counterexample, Mapping) else None))
    scope = copy.deepcopy(dict(edit_scope)) if isinstance(edit_scope, Mapping) else None
    if isinstance(scope, Mapping) and isinstance(scope.get("edit_authority"), Mapping):
        # The wire packet gets the same typed authority object, never the
        # legacy instruction/tool-matrix copy that used to repeat scope text.
        scope = copy.deepcopy(dict(scope["edit_authority"]))
        shortcuts = scope.get("shortcuts")
        if isinstance(shortcuts, list):
            for shortcut in shortcuts:
                if isinstance(shortcut, Mapping):
                    shortcut.pop("label", None)
    if scope is not None:
        from kernel_bound_edit import bind_target, enabled as bound_edit_enabled
        editable_paths = set(_strings(scope.get("editable_paths", ())))
        for row in files:
            row["editable"] = str(row.get("path", "")) in editable_paths
        if bound_edit_enabled():
            bound_targets: dict[str, object] = {}
            for row in files:
                path = str(row.get("path", ""))
                if not path or not isinstance(row.get("source_bytes"), str):
                    continue
                bound_targets[path] = bind_target(
                    path=path, symbol="", revision=row.get("revision", current_revision),
                    source=str(row["source_bytes"]),
                )
            scope["bound_edit"] = "on"
            scope["bound_targets"] = bound_targets
    episode_mapping = episode if isinstance(episode, Mapping) else {}
    packet_next = (None if episode_mapping.get("budget_exhausted")
                   else copy.deepcopy(payload.get("next_action")))
    packet = {
        "schema": REPAIR_PACKET_SCHEMA,
        "required": {
            "focus_obligation": str(payload.get("focus_obligation")),
            "clause": str(payload.get("clause", assessment.clause if assessment else "")),
            "clause_words": str(payload.get("clause", assessment.clause if assessment else "")),
            "coupled_preserves": list(coupled),
            "reason": str(payload.get("reason", assessment.reason if assessment else "")),
        },
        "failing_execution": failing,
        "current_source": {"revision": current_revision, "files": files},
        "keep_working": {
            "recovery_policy": list(REPAIR_RECOVERY_POLICY),
            "episode": copy.deepcopy(dict(episode or {})),
            "budget": {
                "edit_proposals_max": MAX_REPAIR_EDIT_PROPOSALS,
                "calls_max": MAX_REPAIR_CALLS,
                "edit_proposals_used": int((episode or {}).get("edit_proposals", 0) or 0),
                "calls_used": int((episode or {}).get("calls", 0) or 0),
            },
            "public_suite": copy.deepcopy(list(public_suite)),
            "repair_check": copy.deepcopy(list(repair_checks)),
        },
        "next": packet_next,
        "editability": scope,
    }
    if isinstance(payload.get("gaps"), Sequence) and not isinstance(
            payload.get("gaps"), (str, bytes)):
        packet["gap_list"] = copy.deepcopy(list(payload["gaps"]))
    witness = str(failing.get("witness", ""))
    source_rows = packet["current_source"]["files"]
    source_present = bool(source_rows) and all(
        isinstance(row, Mapping) and isinstance(row.get("source_bytes"), str)
        and bool(row.get("source_bytes")) for row in source_rows)
    audit = {
        "schema": REPAIR_PACKET_AUDIT_SCHEMA,
        "valid": bool(selected_receipt is not None
                       and witness and witness != "NO_BOUNDED_HOST_WITNESS"
                       and source_present),
        "witness_present": bool(witness and witness != "NO_BOUNDED_HOST_WITNESS"),
        "source_present": source_present,
        "host_receipt_present": selected_receipt is not None,
        "source_file_count": len(source_rows),
        "reason": ("VALID" if selected_receipt is not None
                   and witness and witness != "NO_BOUNDED_HOST_WITNESS"
                   and source_present else "MISSING_WITNESS_OR_SOURCE"),
    }
    if precision_features_enabled():
        packet["validity_audit"] = audit
    return packet


def render_repair_packet(packet: Mapping[str, object] | None) -> str:
    """Render packet sections in the structured repair order without losing source bytes."""
    if not isinstance(packet, Mapping) or packet.get("schema") != REPAIR_PACKET_SCHEMA:
        raise ValueError("invalid repair packet")
    audit = packet.get("validity_audit")
    if (precision_features_enabled() and isinstance(audit, Mapping)
            and audit.get("valid") is False):
        return ""
    required = packet.get("required", {})
    failing = packet.get("failing_execution", {})
    current = packet.get("current_source", {})
    keep = packet.get("keep_working", {})
    lines = ["[REPAIR PACKET]", "REQUIRED", _canonical(required).decode("utf-8"),
             "FAILING EXECUTION", _canonical(failing).decode("utf-8"),
             "CURRENT SOURCE", "revision=" + str(current.get("revision", "UNKNOWN"))]
    files = current.get("files", ()) if isinstance(current, Mapping) else ()
    for row in files:
        if not isinstance(row, Mapping):
            continue
        marker = (" editable=true" if row.get("editable") is True else "")
        lines.extend(("--- " + str(row.get("path", "")) + " @ "
                      + str(row.get("revision", current.get("revision", "UNKNOWN"))) + marker,
                      str(row.get("source_bytes", ""))))
    lines.extend(("KEEP WORKING", _canonical(keep).decode("utf-8"), "NEXT",
                  _canonical(packet.get("next")).decode("utf-8"), "EDITABILITY",
                  _canonical(packet.get("editability")).decode("utf-8"),
                  "[/REPAIR PACKET]"))
    return "\n".join(lines)


def repair_episode(payload: Mapping[str, object] | None, *, last_action: object = None,
                   last_outcome: object = None,
                   previous: Mapping[str, object] | None = None) -> dict[str, object] | None:
    if not isinstance(payload, Mapping):
        return None
    previous = previous if isinstance(previous, Mapping) else {}
    same_focus = previous.get("focus") == payload.get("focus_obligation")
    previous_calls = int(previous.get("calls", 0) or 0) if same_focus else 0
    previous_edits = int(previous.get("edit_proposals", 0) or 0) if same_focus else 0
    action_tool = (last_action.get("tool", last_action.get("kind"))
                   if isinstance(last_action, Mapping) else None)
    edit_proposal = str(action_tool or "") in {
        "edit", "str_replace", "file_create", "replace_module", "write", "replace"
    }
    current_failure = payload.get("counterexample_ref") or {
        "observed": payload.get("observed"), "expected": payload.get("expected"),
        "reason": payload.get("reason"),
    }
    previous_failure = previous.get("failure_key") if same_focus else None
    same_failure = same_focus and digest(current_failure) == previous_failure
    same_failure_count = (int(previous.get("same_failure_count", 0) or 0) + 1
                          if same_failure else 0)
    same_revision = (same_focus
                     and payload.get("source_delivered_at_revision")
                     == previous.get("source_delivered_at_revision"))
    no_new_observation = bool(
        same_revision
        and str(action_tool or "") in {"read", "run_tests", "done", "work"}
        and str(last_outcome or "").upper() not in {"APPLIED", "EDIT_APPLIED"}
    )
    recovery = "rebind"
    outcome = str(last_outcome or "").upper()
    if outcome in {"SYNTAX_ERROR", "SYNTAX"}:
        recovery = "syntax"
    elif outcome in {"COUPLED_REGRESSION", "COUPLED_REGRESS"}:
        recovery = "coupled_regress"
    elif (previous.get("recovery") == "same_failure_x2_expand"
          and same_failure_count >= 3):
        recovery = "restart_from_baseline"
    elif same_failure_count >= 2:
        recovery = "same_failure_x2_expand"
    elif no_new_observation:
        recovery = "no_new_observation"
    calls = min(MAX_REPAIR_CALLS, previous_calls + 1)
    edits = min(MAX_REPAIR_EDIT_PROPOSALS, previous_edits + (1 if edit_proposal else 0))
    budget_exhausted = calls >= MAX_REPAIR_CALLS or edits >= MAX_REPAIR_EDIT_PROPOSALS
    next_action = None if budget_exhausted else copy.deepcopy(payload.get("next_action"))
    result = {
        "schema": "stateframe-repair-episode/1",
        "focus": payload.get("focus_obligation"),
        "counterexample_ref": payload.get("counterexample_ref"),
        "source_delivered_at_revision": payload.get("source_delivered_at_revision"),
        "last_action": copy.deepcopy(last_action),
        "last_outcome": copy.deepcopy(last_outcome),
        "duplicate_count": int(payload.get("duplicate_count", 0) or 0),
        "next_action": next_action,
        "calls": calls,
        "edit_proposals": edits,
        "budgets": {"edit_proposals": MAX_REPAIR_EDIT_PROPOSALS,
                    "calls": MAX_REPAIR_CALLS},
        "within_budget": not budget_exhausted,
        "budget_exhausted": budget_exhausted,
        "failure_key": digest(current_failure),
        "same_failure_count": same_failure_count,
        "no_new_observation": no_new_observation,
        "recovery": recovery,
        "recovery_policy": list(REPAIR_RECOVERY_POLICY),
    }
    if precision_features_enabled():
        result.update({
            "obligation_status": (UNKNOWN if budget_exhausted else UNSATISFIED),
            "unknown_reason": ("REPAIR_BUDGET_EXHAUSTED" if budget_exhausted else None),
        })
    return result


def dispute_check(check_id: object, basis: object, claim: object, *,
                  definitions: object = (), receipts: object = ()) -> dict[str, object]:
    """Re-evaluate one host check from typed public-test/ticket references."""
    refs = (list(basis) if isinstance(basis, Sequence)
            and not isinstance(basis, (str, bytes, bytearray)) else [])
    normalized = [str(item) for item in refs if isinstance(item, str) and item.strip()]
    result = {
        "schema": "stateframe-dispute-check/1",
        "check_id": str(check_id or ""),
        "basis": normalized,
        "claim": str(claim or ""),
    }
    if not normalized or any(not (item in {"public_test_ref", "ticket_clause_ref"}
                                  or item.startswith("public_test_ref:")
                                  or item.startswith("ticket_clause_ref:"))
                             for item in normalized):
        return {**result, "status": "REFUSED", "reason": "INVALID_BASIS_REFERENCE"}
    definition = _definition_for(str(check_id or ""), definitions)
    if definition is None:
        return {**result, "status": "REFUSED", "reason": "UNKNOWN_CHECK"}
    ticket_basis = definition.oracle_basis == "TICKET" and any(
        item == "ticket_clause_ref" or item.startswith("ticket_clause_ref:")
        for item in normalized)
    if not ticket_basis:
        return {**result, "status": "RETIRED", "reason": "NO_TICKET_BASIS",
                "retired": True}
    public_definitions = []
    if isinstance(definitions, Mapping):
        definition_values = definitions.values()
    elif isinstance(definitions, Iterable) and not isinstance(definitions, (str, bytes)):
        definition_values = definitions
    else:
        definition_values = ()
    for raw in definition_values:
        candidate = (raw if isinstance(raw, CheckDefinition)
                     else CheckDefinition.from_mapping(raw) if isinstance(raw, Mapping) else None)
        if (candidate is not None and candidate.oracle_basis == "PUBLIC_TEST"
                and candidate.obligation_id == definition.obligation_id):
            public_definitions.append(candidate)
    public_ids = {item.check_id for item in public_definitions}
    public_passes = []
    if isinstance(receipts, Mapping):
        receipt_values = receipts.values() if "check_id" not in receipts else (receipts,)
    elif isinstance(receipts, Iterable) and not isinstance(receipts, (str, bytes)):
        receipt_values = receipts
    else:
        receipt_values = ()
    for raw in receipt_values:
        receipt = _receipt_for(raw)
        if (receipt is not None and receipt.check_id in public_ids
                and receipt.execution == EXECUTED and receipt.status == PASS):
            public_passes.append(receipt)
    contradiction = any(
        _canonical(candidate.expected) != _canonical(definition.expected)
        or _canonical(receipt.observed) != _canonical(definition.expected)
        for candidate in public_definitions for receipt in public_passes
        if receipt.check_id == candidate.check_id
    )
    if contradiction:
        return {**result, "status": "RETIRED", "reason": "PUBLIC_TEST_CONTRADICTION",
                "retired": True,
                "public_receipt_refs": [receipt.receipt_id for receipt in public_passes]}
    return {**result, "status": "REFUSED", "reason": "CLAIM_NOT_ESTABLISHED",
            "retired": False, "basis_required": ["ticket_clause_ref", "public_test_ref"]}


def _entry_point_state(root: Path, entry_point: str | None) -> str:
    """Classify static entry-point availability without importing product code."""
    if not isinstance(entry_point, str) or ":" not in entry_point:
        return "MISSING"
    module, attribute = entry_point.split(":", 1)
    if not module or not attribute:
        return "MISSING"
    module_path = root.joinpath(*module.split("."))
    source_path = (module_path.with_suffix(".py")
                   if module_path.with_suffix(".py").is_file()
                   else module_path / "__init__.py")
    if not source_path.is_file():
        return "MISSING"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, UnicodeError, SyntaxError):
        return "UNRUNNABLE"
    names = {node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Name, ast.Attribute, ast.Lambda)):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif (isinstance(node, ast.AnnAssign)
              and isinstance(node.value, (ast.Name, ast.Attribute, ast.Lambda))
              and isinstance(node.target, ast.Name)):
            names.add(node.target.id)
    # ast.Import carries aliases in .names; only the alias has .asname (crashed 09-25 on any import).
    names.update((item.asname or item.name.split(".")[0])
                 for node in tree.body if isinstance(node, ast.Import) for item in node.names)
    names.update(item.asname or item.name for node in tree.body if isinstance(node, ast.ImportFrom)
                 for item in node.names if item.name != "*")
    first, *rest = attribute.split(".")
    if first not in names:
        return "MISSING"
    # A nested attribute is conservatively checked only for a class method;
    # imports and properties are resolved in the isolated execution process.
    if not rest:
        return "PRESENT"
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == first:
            return ("PRESENT" if any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == rest[-1] for item in node.body) else "MISSING")
    return "MISSING"


def _entry_point_exists(root: Path, entry_point: str | None) -> bool:
    """Compatibility wrapper for callers that only need the static boolean."""
    return _entry_point_state(root, entry_point) == "PRESENT"


_UNBOUND_BIND_LINE = '        return "UNBOUND", str(error)\n    except (ValueError, RuntimeError) as error:'
_SIGNATURE_BIND_LINE = ('        return "SIGNATURE_MISMATCH", str(error)\n'
                        '    except (ValueError, RuntimeError) as error:')


def _script_for_check(check: CheckDefinition) -> str:
    script = _script_for_check_base(check)
    # STATEFRAME_NAMED_LAYER_REPAIR (autopsy-a3b-r44c-ldev01 A): bind_target runs only after resolve()
    # succeeded, so a bind TypeError there is a SIGNATURE_MISMATCH on a real entry point, not an unbound
    # check.  Off: the script text is byte-identical to r45.
    from kernel_named_layer_repair import enabled as named_layer_repair_enabled
    if named_layer_repair_enabled() and _UNBOUND_BIND_LINE in script:
        script = script.replace(_UNBOUND_BIND_LINE, _SIGNATURE_BIND_LINE, 1)
    return script


def _script_for_check_base(check: CheckDefinition) -> str:
    definition = {
        "template": check.template,
        "entry_point": check.entry_point,
        "fixture": check.fixture,
        "expected": check.expected,
    }
    # The script is data-only and runs in a fresh interpreter.  Product source
    # is imported from the read-only workspace; no candidate code is executed in
    # the StateFrame host process.
    return """
import copy, importlib, inspect, json, os, sys, traceback
sys.path.insert(0, os.getcwd())
definition = json.loads(%s)

class MutationProbeDict(dict):
    def __init__(self, value=(), tracker=None):
        super().__init__(value)
        self._tracker = tracker if tracker is not None else []
    def _mark(self, operation):
        self._tracker.append(operation)
    def __setitem__(self, key, value):
        self._mark(("set", key))
        return super().__setitem__(key, value)
    def __delitem__(self, key):
        self._mark(("delete", key))
        return super().__delitem__(key)
    def pop(self, key, *default):
        self._mark(("pop", key))
        return super().pop(key, *default)
    def popitem(self):
        self._mark(("popitem",))
        return super().popitem()
    def clear(self):
        self._mark(("clear",))
        return super().clear()
    def update(self, *args, **kwargs):
        self._mark(("update",))
        return super().update(*args, **kwargs)
    def setdefault(self, key, *default):
        self._mark(("setdefault", key))
        return super().setdefault(key, *default)
    def copy(self):
        return MutationProbeDict(self, self._tracker)
    def __deepcopy__(self, memo):
        # A deep copy is a new ownership boundary.  Writes to that copy are
        # not writes to the caller's input; shallow aliases remain tracked.
        result = MutationProbeDict(tracker=[])
        memo[id(self)] = result
        for key, value in self.items():
            dict.__setitem__(result, copy.deepcopy(key, memo), copy.deepcopy(value, memo))
        return result

def track(value, tracker):
    if isinstance(value, dict):
        return MutationProbeDict(((key, track(item, tracker)) for key, item in value.items()), tracker)
    if isinstance(value, list):
        return [track(item, tracker) for item in value]
    return value

def resolve(path):
    module, attribute = path.split(":", 1)
    value = importlib.import_module(module)
    for part in attribute.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError("entry point is not callable")
    return value

def bind_target(target, args, kwargs):
    try:
        inspect.signature(target).bind(*args, **kwargs)
    except TypeError as error:
        return "UNBOUND", str(error)
    except (ValueError, RuntimeError) as error:
        return "UNKNOWN", str(error)
    return "BOUND", ""

def call(case):
    case = case if isinstance(case, dict) else {}
    args = case.get("args", [])
    kwargs = case.get("kwargs", {})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise TypeError("fixture call requires args list and kwargs object")
    args = copy.deepcopy(args)
    kwargs = copy.deepcopy(kwargs)
    target = resolve(definition["entry_point"])
    binding, binding_error = bind_target(target, args, kwargs)
    if binding != "BOUND":
        return {"ok": False, "binding": binding, "binding_error": binding_error,
                "exception": {"type": "BindingError", "message": binding_error},
                "input_before": None, "input_after": None, "input_mutations": []}
    input_index = case.get("input_index")
    before = None
    tracker = []
    if isinstance(input_index, int) and 0 <= input_index < len(args):
        before = copy.deepcopy(args[input_index])
        args[input_index] = track(args[input_index], tracker)
    try:
        output = target(*args, **kwargs)
        return {"ok": True, "output": output, "input_before": before,
                "input_after": (args[input_index] if isinstance(input_index, int)
                                 and 0 <= input_index < len(args) else None),
                "input_mutations": tracker, "binding": binding}
    except BaseException as error:
        return {"ok": False, "exception": {"type": type(error).__name__,
                "message": str(error)}, "input_before": before,
                "input_after": (args[input_index] if isinstance(input_index, int)
                                 and 0 <= input_index < len(args) else None),
                "input_mutations": tracker, "binding": binding}

def path_get(value, path):
    for part in str(path).split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(path)
    return value

def _contains_text(value, needle):
    if isinstance(value, dict):
        return any(_contains_text(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains_text(item, needle) for item in value)
    return needle in str(value)

def same(actual, expected):
    if not isinstance(actual, (dict, list, tuple, str, int, float, bool, type(None))) \
            and hasattr(actual, "to_dict"):
        actual = actual.to_dict()
    if isinstance(expected, dict) and "contains" in expected:
        wanted = expected["contains"]
        return isinstance(actual, dict) and all(actual.get(k) == v for k, v in wanted.items())
    if isinstance(expected, dict) and "error_contains" in expected:
        return _contains_text(actual, expected["error_contains"])
    if (os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on"
            and isinstance(actual, dict) and isinstance(expected, dict)
            and expected.get("status") == "4xx"):
        status = actual.get("status")
        return isinstance(status, int) and 400 <= status < 500 and all(
            key == "status" or same(actual.get(key), value)
            for key, value in expected.items())
    if isinstance(expected, dict) and isinstance(actual, dict) \
            and any(isinstance(value, dict) and "error_contains" in value
                    for value in expected.values()):
        return all(same(actual.get(key), value) if key in actual else False
                   for key, value in expected.items())
    return actual == expected

def run():
    template = definition["template"]
    fixture = definition.get("fixture") or {}
    expected = definition.get("expected")
    if template == "OPTIONAL_INPUT":
        cases = fixture.get("cases", [])
        outputs = [call(case) for case in cases]
        actual = [item.get("output") if item.get("ok") else item for item in outputs]
        uncertain = any(item.get("binding") != "BOUND" for item in outputs)
        ok = not uncertain and all(item.get("ok") for item in outputs) and same(actual, expected)
        return ok, actual, expected, {"cases": outputs, "binding_uncertainty": uncertain}
    if template in {"OUTPUT_EQUALS", "VALUE_TRANSPORT", "ERROR_TRANSLATION"}:
        result = call(fixture)
        actual = result.get("output") if result.get("ok") else result
        uncertain = result.get("binding") != "BOUND"
        return (not uncertain and bool(result.get("ok")) and same(actual, expected), actual,
                expected, {**result, "binding_uncertainty": uncertain})
    if template == "ROUNDING_DISCRIMINATION":
        result = call(fixture)
        actual = result.get("output") if result.get("ok") else result
        uncertain = result.get("binding") != "BOUND"
        try:
            observed = path_get(actual, expected["path"])
            ok = (not uncertain and bool(result.get("ok"))
                  and observed == expected["value"])
        except (KeyError, TypeError, IndexError):
            observed = None
            ok = False
        return ok, {"output": actual, "observed": observed}, expected, {
            **result, "binding_uncertainty": uncertain}
    if template == "SUM_CONSISTENCY":
        result = call(fixture)
        actual = result.get("output") if result.get("ok") else result
        uncertain = result.get("binding") != "BOUND"
        try:
            total = path_get(actual, expected["total_path"])
            components = [path_get(actual, path) for path in expected["component_paths"]]
            ok = not uncertain and bool(result.get("ok")) \
                and total == sum(components)
            observed = {"total": total, "components": components,
                        "sum": sum(components)}
        except (KeyError, TypeError, IndexError):
            ok = False
            observed = {"output": actual}
        return ok, observed, expected, {**result, "binding_uncertainty": uncertain}
    if template == "OUTPUT_DELTA":
        base = call(fixture.get("base", {}))
        variant = call(fixture.get("variant", {}))
        if base.get("binding") != "BOUND" or variant.get("binding") != "BOUND":
            return False, {"base": base, "variant": variant}, expected, {
                "base": base, "variant": variant, "binding_uncertainty": True}
        if not base.get("ok") or not variant.get("ok"):
            return False, {"base": base, "variant": variant}, expected, {"base": base, "variant": variant}
        anchor = expected.get("anchor") if isinstance(expected, dict) else None
        if not isinstance(anchor, dict) or not isinstance(anchor.get("path"), str) or "value" not in anchor:
            raise ValueError("OUTPUT_DELTA requires an absolute anchor path and value")
        base_value = path_get(base["output"], anchor["path"])
        variant_value = path_get(variant["output"], expected["path"])
        base_delta_value = path_get(base["output"], expected["path"])
        variant_delta_value = variant_value
        ok = (base_value == anchor["value"] and
              variant_delta_value - base_delta_value == expected["delta"])
        return ok, {"base": base["output"], "variant": variant["output"],
                    "anchor": base_value, "delta": variant_delta_value - base_delta_value}, expected, {}
    if template == "REJECT_WITH_STATUS":
        result = call(fixture)
        actual = result.get("output") if result.get("ok") else result
        status = actual.get("status") if isinstance(actual, dict) else None
        expected_status = expected.get("status") if isinstance(expected, dict) else expected
        uncertain = result.get("binding") != "BOUND"
        status_ok = (status == expected_status or (
            os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on"
            and expected_status == "4xx" and isinstance(status, int)
            and 400 <= status < 500))
        return (not uncertain and result.get("ok") and status_ok, actual,
                expected, {**result, "binding_uncertainty": uncertain})
    if template == "PRESERVE_CALL_SHAPE":
        calls = fixture.get("calls", [])
        results = [call(case) for case in calls]
        actual = [item.get("output") if item.get("ok") else item for item in results]
        expected_outputs = expected.get("outputs") if isinstance(expected, dict) else None
        uncertain = any(item.get("binding") != "BOUND" for item in results)
        ok = not uncertain and all(item.get("ok") for item in results)
        if expected_outputs is not None:
            ok = ok and same(actual, expected_outputs)
        return ok, actual, expected, {"calls": results, "binding_uncertainty": uncertain}
    if template == "PRESERVE_INPUT":
        result = call(fixture)
        before, after = result.get("input_before"), result.get("input_after")
        uncertain = result.get("binding") != "BOUND"
        ok = (bool(result.get("ok")) and before == after
              and not result.get("input_mutations") and not uncertain)
        return ok, {"output": result.get("output"), "input_before": before,
                    "input_after": after}, expected, {**result,
                    "binding_uncertainty": uncertain}
    if template == "REPLAY_EQUIVALENT":
        case = fixture if isinstance(fixture, dict) else {}
        args = case.get("args", [])
        kwargs = case.get("kwargs", {})
        if not isinstance(args, list) or not isinstance(kwargs, dict):
            raise TypeError("replay fixture call requires args list and kwargs object")
        args = copy.deepcopy(args)
        kwargs = copy.deepcopy(kwargs)
        target = resolve(definition["entry_point"])
        binding, binding_error = bind_target(target, args, kwargs)
        if binding != "BOUND":
            return False, {"binding": binding, "binding_error": binding_error}, expected, {
                "binding_uncertainty": True}
        input_index = case.get("input_index")
        tracker = []
        before = None
        if isinstance(input_index, int) and 0 <= input_index < len(args):
            before = copy.deepcopy(args[input_index])
            args[input_index] = track(args[input_index], tracker)
        try:
            first = target(*args, **kwargs)
            second = target(*args, **kwargs)
            observed = {"first": first, "second": second,
                        "input_before": before,
                        "input_after": (args[input_index] if isinstance(input_index, int)
                                         and 0 <= input_index < len(args) else None),
                        "input_mutations": tracker}
            return first == second, observed, expected, {"calls": 2, "binding_uncertainty": False}
        except BaseException as error:
            return False, {"exception": {"type": type(error).__name__,
                                          "message": str(error)},
                           "input_before": before,
                           "input_after": (args[input_index] if isinstance(input_index, int)
                                            and 0 <= input_index < len(args) else None),
                           "input_mutations": tracker}, expected, {"calls": 2,
                                                                    "binding_uncertainty": False}
    if template == "INVOCATION_WITNESS":
        cases = fixture.get("cases", []) if isinstance(fixture, dict) else []
        if not cases:
            cases = [fixture if isinstance(fixture, dict) else {}]
        results = [call(case) for case in cases]
        uncertain = any(item.get("binding") != "BOUND" for item in results)
        ok = not uncertain and all(item.get("ok") for item in results)
        actual = [item.get("output") if item.get("ok") else item for item in results]
        return ok, actual, expected, {"calls": results, "binding_uncertainty": uncertain}
    if template == "COMPATIBILITY_WITNESS":
        calls = fixture.get("calls", []) if isinstance(fixture, dict) else []
        results = [call(case) for case in calls]
        uncertain = any(item.get("binding") != "BOUND" for item in results)
        ok = not uncertain and all(item.get("ok") for item in results)
        actual = [item.get("output") if item.get("ok") else item for item in results]
        return ok, actual, expected, {"calls": results, "binding_uncertainty": uncertain}
    if template == "PRESERVE_EXISTING_VALIDATION":
        result = call(fixture)
        actual = result.get("output") if result.get("ok") else result
        uncertain = result.get("binding") != "BOUND"
        return (not uncertain and bool(result.get("ok")) and same(actual, expected), actual,
                expected, {**result, "binding_uncertainty": uncertain})
    raise ValueError("unknown template")

try:
    ok, observed, expected, detail = run()
    uncertainty = isinstance(detail, dict) and detail.get("binding_uncertainty") is True
    print(json.dumps({"ok": bool(ok), "observed": observed, "expected": expected,
                      "detail": detail,
                      "execution": "SKIPPED" if uncertainty else "EXECUTED",
                      "reason": "UNBOUND_INVOCATION" if uncertainty else ("PASS" if ok else "COUNTEREXAMPLE")},
             ensure_ascii=False, default=str))
except BaseException as error:
    print(json.dumps({"error": {"type": type(error).__name__, "message": str(error)},
                      "traceback": traceback.format_exc()[-3000:]}, ensure_ascii=False))
    raise
""" % json.dumps(json.dumps(definition, ensure_ascii=False, sort_keys=True, default=str))


def _promised_binding_cell(check: CheckDefinition) -> bool:
    """Whether a binding failure violates a public contract cell promise."""
    if os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() != "on":
        return False
    cell = check.contract_cell
    if cell is None or not isinstance(cell.case_partition, Mapping):
        return False
    kind = str(cell.case_partition.get("kind", ""))
    if kind in {"compatibility", "compatibility_alias", "compatibility_input",
                "omitted", "rename_coverage", "bind_and_execute"}:
        return True
    return kind == "explicit" and str(cell.case_partition.get("surface", "")) == "cli"


def _binding_counterexample(check: CheckDefinition, *, missing: bool = False,
                            detail: object = None) -> dict[str, object]:
    surface = check.contract_cell.surface if check.contract_cell is not None else "unknown"
    if missing:
        return {"missing_symbol": check.entry_point, "surface": surface}
    return {"arity_error": detail or "promised call does not bind",
            "surface": surface}


def _receipt_alignment_active(ledger: Mapping[str, object] | None) -> bool:
    return (ledger is not None
            and os.environ.get("STATEFRAME_NAMED_LAYER_REPAIR", "off").strip().lower() == "on"
            and os.environ.get("STATEFRAME_REQUIREMENT_ALIGNMENT", "off").strip().lower() == "on")


def _attach_receipt_alignment(receipt: CheckReceipt, *, check: CheckDefinition,
                              ledger: Mapping[str, object] | None) -> CheckReceipt:
    """Attach alignment facts before journaling, keeping the receipt producer authoritative."""
    if not _receipt_alignment_active(ledger) or receipt.status != COUNTEREXAMPLE:
        return receipt
    try:
        from kernel_requirement_alignment import receipt_alignment_fields
        fields = receipt_alignment_fields(receipt.to_dict(), check=check, ledger=ledger)
    except (ImportError, KeyError, TypeError, ValueError):
        return receipt
    if not fields:
        return receipt
    payload = receipt.to_dict()
    payload.pop("receipt_id", None)
    payload.pop("evidence_ref", None)
    payload.update(fields)
    return CheckReceipt.from_mapping(payload)


def run_check(root: str | Path, check: CheckDefinition | Mapping[str, object], *,
              snapshot: object = None, environment: object = None,
              cache: dict[str, CheckReceipt] | None = None,
              timeout: float = 8.0, identity: object = None,
              requirements_ledger: Mapping[str, object] | None = None) -> CheckReceipt:
    """Execute one frozen check in a fresh, disposable workspace process."""
    if not isinstance(check, CheckDefinition):
        check = CheckDefinition.from_mapping(check)
    root = Path(root).resolve()
    if os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on":
        try:
            from kernel_green_lock_repair import classify_check
            policy = classify_check(check, starter_root=root)
            check = policy.check
        except (ImportError, TypeError, ValueError):
            pass
    snap = snapshot if snapshot is not None else workspace_snapshot(root)
    environment = environment if environment is not None else {
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    instance = check.instance_key(snap, environment)
    if _receipt_alignment_active(requirements_ledger):
        instance = digest({"base_instance": instance, "requirements_ledger": requirements_ledger})
    if cache is not None and instance in cache:
        prior = cache[instance]
        if not isinstance(prior, CheckReceipt):
            prior = CheckReceipt.from_mapping(prior)
        if identity is not None:
            from kernel_verdict_identity import IdentityError, assert_current, enabled as identity_enabled
            if identity_enabled():
                try:
                    assert_current(prior.to_dict(), identity)
                except IdentityError:
                    # A cache hit from another code tree is not reusable
                    # evidence.  Re-run the check instead of relabeling it.
                    cache.pop(instance, None)
                else:
                    reused = dataclasses.replace(prior, reused=True,
                                                  duplicate_count=prior.duplicate_count + 1)
                    cache[instance] = reused
                    return reused
            else:
                reused = dataclasses.replace(prior, reused=True,
                                              duplicate_count=prior.duplicate_count + 1)
                cache[instance] = reused
                return reused
        else:
            reused = dataclasses.replace(prior, reused=True,
                                          duplicate_count=prior.duplicate_count + 1)
            cache[instance] = reused
            return reused

    def finish(value: CheckReceipt) -> CheckReceipt:
        value = _attach_receipt_alignment(value, check=check, ledger=requirements_ledger)
        if identity is None:
            return value
        from kernel_verdict_identity import stamp_record
        stamped = stamp_record(value.to_dict(), identity)
        return CheckReceipt.from_mapping(stamped) if isinstance(stamped, Mapping) else value
    common = {
        "check_id": check.check_id,
        "snapshot_digest": _identity(snap),
        "check_digest": check.digest,
        "fixture_digest": check.fixture_digest,
        "environment_digest": _identity(environment),
        "oracle_basis": check.oracle_basis,
        "expected": check.expected,
        "instance_key": instance,
        "selection": check.selection,
        "contract_cell_hash": (check.contract_cell.contract_cell_hash
                                if check.contract_cell is not None else ""),
        "tested_domain": (check.contract_cell.case_partition
                          if check.contract_cell is not None else None),
        "trace_refs": check.selection,
    }
    if check.interpretation != VALIDATED:
        receipt = CheckReceipt(**common, status=NOT_EXECUTED,
                               reason=("AMBIGUOUS_CONTRACT" if check.interpretation == AMBIGUOUS
                                       else "UNVALIDATED_INTERPRETATION"))
        receipt = finish(receipt)
        if cache is not None:
            cache[instance] = receipt
        return receipt
    entry_state = _entry_point_state(root, check.entry_point)
    if entry_state != "PRESENT":
        if entry_state == "MISSING" and _promised_binding_cell(check):
            receipt = CheckReceipt(
                **common, status=COUNTEREXAMPLE, execution=EXECUTED,
                reason=COUNTEREXAMPLE,
                counterexample=_binding_counterexample(check, missing=True),
                observed={"binding": "UNBOUND", "binding_error": "missing entry point"},
            )
        else:
            receipt = CheckReceipt(**common, status=NOT_EXECUTED, reason="UNBOUND_ENTRYPOINT")
        receipt = finish(receipt)
        if cache is not None:
            cache[instance] = receipt
        return receipt
    try:
        timeout = min(MAX_TIMEOUT_SECONDS, max(0.1, float(timeout)))
        from kernel_safe_probe import run_probe
        # Read-only execution makes a copied workspace unnecessary and avoids
        # following repository symlinks while copying untrusted input.
        process = run_probe(
            [test_python(), "-I", "-B", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); " + _script_for_check(check), str(root)],
            Path(root), timeout=timeout)
        if process.timed_out:
            raise subprocess.TimeoutExpired("obligation check", timeout)
        stdout = process.output.decode("utf-8", "replace")[-MAX_OUTPUT_CHARS:]
        if process.exit_code != 0:
            raise RuntimeError(stdout or "sandboxed check failed")
        line = next((item for item in reversed(stdout.splitlines()) if item.strip()), "")
        payload = json.loads(line)
        if "error" in payload:
            raise RuntimeError(payload["error"].get("message", "check error"))
        passed = payload.get("ok") is True
        observed = payload.get("observed")
        detail = payload.get("detail")
        execution = str(payload.get("execution", EXECUTED)).upper()
        reason = str(payload.get("reason", "PASS" if passed else "COUNTEREXAMPLE"))
        unbound = reason == UNBOUND_INVOCATION or execution == SKIPPED
        binding_uncertainty = unbound and (
            check.contract_cell is not None
            or check.template in {
                "INVOCATION_WITNESS", "COMPATIBILITY_WITNESS", "PRESERVE_CALL_SHAPE",
            }
        )
        # Union of boundary_precision (any grounded contract cell) and check_coverage (promised
        # surface/alias/arity) triggers: a promised binding that fails is a witnessed violation.
        binding_counterexample = bool(
            binding_uncertainty and (
                (precision_features_enabled() and check.contract_cell is not None)
                or _promised_binding_cell(check)))
        binding_error = None
        if isinstance(detail, Mapping):
            binding_error = detail.get("binding_error")
            if not binding_error:
                def find_binding_error(value: object) -> object:
                    if isinstance(value, Mapping):
                        if value.get("binding_error"):
                            return value.get("binding_error")
                        for child in value.values():
                            found = find_binding_error(child)
                            if found:
                                return found
                    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                        for child in value:
                            found = find_binding_error(child)
                            if found:
                                return found
                    return None
                binding_error = find_binding_error(detail)
        counterexample = None
        if (precision_features_enabled() and not passed and not binding_uncertainty) or binding_counterexample:
            counterexample = {
                "fixture": copy.deepcopy(check.fixture),
                "observed": copy.deepcopy(observed),
                "expected": copy.deepcopy(payload.get("expected", check.expected)),
                "detail": copy.deepcopy(detail),
            }
            if binding_error:
                counterexample["binding_error"] = str(binding_error)
            if binding_counterexample and _promised_binding_cell(check):
                # check_coverage shape: surface + arity_error/missing_symbol for repair hints.
                counterexample.update(_binding_counterexample(check, detail=detail))
        elif not passed and not binding_uncertainty:
            counterexample = {
                "observed": copy.deepcopy(observed),
                "expected": copy.deepcopy(payload.get("expected", check.expected)),
                "detail": copy.deepcopy(detail),
            }
        if (counterexample is not None
                and os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on"):
            try:
                from kernel_green_lock_repair import annotate_preservation_counterexample
                counterexample = annotate_preservation_counterexample(
                    root, check, counterexample, detail)
            except (ImportError, OSError, TypeError, ValueError):
                # Diagnostic enrichment must never change the underlying
                # check verdict when a source tree cannot be scanned.
                pass
        receipt = CheckReceipt(
            **({**common, "oracle_basis": "ADVISORY"}
               if (os.environ.get("STATEFRAME_GREEN_LOCK_REPAIR", "off").lower() == "on"
                   and binding_uncertainty) else common),
            status=(COUNTEREXAMPLE if binding_counterexample else
                    NOT_EXECUTED if binding_uncertainty else
                    PASS if passed else COUNTEREXAMPLE),
            observed=observed,
            counterexample=counterexample,
            reason=(COUNTEREXAMPLE if binding_counterexample else
                    UNBOUND_INVOCATION if binding_uncertainty else
                    "PASS" if passed else "COUNTEREXAMPLE"),
            execution=EXECUTED if binding_counterexample else (SKIPPED if unbound else execution),
            observed_values=observed,
        )
    except subprocess.TimeoutExpired:
        receipt = CheckReceipt(**common, status=HARNESS_ERROR,
                               reason=f"TIMEOUT:{timeout:g}s",
                               counterexample={"error": "TIMEOUT"})
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        receipt = CheckReceipt(**common, status=HARNESS_ERROR,
                               reason=str(error)[:MAX_OUTPUT_CHARS],
                               counterexample={"error": str(error)[:MAX_OUTPUT_CHARS]})
    receipt = finish(receipt)
    if cache is not None:
        cache[instance] = receipt
    return receipt


def run_registered_checks(root: str | Path, checks: Sequence[CheckDefinition], *,
                          snapshot: object = None, environment: object = None,
                          cache: dict[str, CheckReceipt] | None = None,
                          check_ids: Sequence[str] | None = None,
                          timeout: float = 8.0, identity: object = None,
                          requirements_ledger: Mapping[str, object] | None = None) -> tuple[CheckReceipt, ...]:
    selected = set(check_ids or ())
    return tuple(run_check(root, check, snapshot=snapshot, environment=environment,
                           cache=cache, timeout=timeout, identity=identity,
                           requirements_ledger=requirements_ledger)
                 for check in checks if check.cheap and (not selected or check.check_id in selected))


def unknown_no_verifier(obligation_id: str, *, clause: str = "",
                        coupled_obligations: Sequence[str] = ()) -> ObligationAssessment:
    return ObligationAssessment(obligation_id, UNKNOWN, NO_VERIFIER, clause=clause,
                                coupled_obligations=tuple(coupled_obligations))


__all__ = [
    "ADMITTED", "ADVISORY", "AMBIGUOUS", "AMBIGUOUS_CONTRACT", "APPLIES", "APPLICABILITY_UNKNOWN",
    "CONFLICTED", "COUNTEREXAMPLE", "ContractCell", "CheckBundle", "CheckDefinition",
    "CheckReceipt", "CheckSpec", "DIAGNOSE", "HARNESS_ERROR", "NOT_EXECUTED", "ObligationAssessment",
    "NO_VERIFIER", "UNBOUND_ENTRYPOINT", "UNBOUND_INVOCATION", "MutationSite", "ObligationSpec", "ORACLE_BASES", "PASS", "PROPOSED", "REPAIR", "REUSED", "REVIEW", "SATISFIED",
    "SETUP_ERROR", "SKIPPED", "EXECUTED", "EXECUTION_STATUSES", "DOES_NOT_APPLY", "PROVISIONAL", "UNBOUND",
    "TEMPLATES", "UNKNOWN", "UNSATISFIED", "VALIDATED", "VERIFY", "REPAIR_PACKET_SCHEMA",
    "HOST_CLOSURE_RENDER_CAP", "REPAIR_PACKET_AUDIT_SCHEMA", "STATIC_BUNDLES_ENV",
    "REPAIR_RECOVERY_POLICY", "MAX_REPAIR_CALLS", "MAX_REPAIR_EDIT_PROPOSALS",
    "BLOCKER_CLASSES", "classify_blocker", "resolver_actions",
    "assessment_is_blocking", "receipt_is_executed_counterexample", "precision_features_enabled",
    "static_bundles_mode", "static_bundles_enabled",
    "assess_obligation",
    "assess_obligations", "augment_input_preservation_bundle", "bind_input_preservation",
    "bounded_host_closure_projection",
    "check_file_path", "digest", "enabled", "fixture_from_public_test",
    "find_mutation_sites", "has_preservation_clause", "input_preservation_bundle",
    "load_check_bundle", "load_checks", "mutation_sites",
    "load_obligation_specs", "mode", "needs_work_payload", "rank_obligations", "reduce_obligation",
    "repair_episode", "repair_packet", "render_repair_packet", "dispute_check", "run_check",
    "run_registered_checks", "unknown_no_verifier", "workspace_snapshot",
]
