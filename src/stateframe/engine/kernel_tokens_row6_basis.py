"""Pure semantic identity records for the v29 TOKENS repetition observer.

This module deliberately knows nothing about runtimes, models, files, journals, or
authority decisions.  It accepts an explicit public mapping and keeps semantic
identity separate from receipt provenance and presentation metadata.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias

SCHEMA = "tokens-row6-basis/1"
_MISSING = object()


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, (str, int, float, bool)) else ""


def _first(mapping: Mapping[str, object], names: Sequence[str]) -> object:
    for name in names:
        value = mapping.get(name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return None


def _section(source: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = source.get(name)
    return value if isinstance(value, Mapping) else {}


def _strings(value: object, *, sort: bool = False, limit: int = 64) -> tuple[str, ...]:
    if isinstance(value, str):
        values = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values = tuple(_text(item) for item in value if _text(item))
    else:
        values = ()
    unique = tuple(dict.fromkeys(values))[:limit]
    return tuple(sorted(unique)) if sort else unique


def _canonical_scalar(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite semantic value")
        return value
    raise ValueError(f"unsupported semantic value: {type(value).__name__}")


def _canonical_map(value: Mapping[str, object], allowed: frozenset[str]) -> dict[str, object]:
    """Canonicalize one explicitly allowlisted flat map.

    Nested maps are not recursively stripped by key name.  Callers must select
    and normalize each nested semantic field separately.
    """
    result: dict[str, object] = {}
    for key in sorted(allowed):
        if key in value:
            result[key] = _canonical_scalar(value[key])
    return result


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Binding:
    identity: str
    revision: str
    digest: str = ""
    source_digest: str = ""
    harness_digest: str = ""
    environment_digest: str = ""

    def semantic(self) -> dict[str, str]:
        return {
            "identity": self.identity,
            "revision": self.revision,
            "digest": self.digest,
            "source_digest": self.source_digest,
            "harness_digest": self.harness_digest,
            "environment_digest": self.environment_digest,
        }


@dataclass(frozen=True)
class Dependency:
    name: str
    version: str
    value: str = ""
    external: bool = False

    def semantic(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version, "value": self.value,
                "external": self.external}


@dataclass(frozen=True)
class Clause:
    clause_id: str
    status: str
    reason: str = ""
    applicability: str = ""
    freshness: str = ""
    evidence_refs: tuple[str, ...] = ()

    def semantic(self) -> dict[str, object]:
        return {"id": self.clause_id, "status": self.status, "reason": self.reason,
                "applicability": self.applicability, "freshness": self.freshness}

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic(), "evidence_refs": list(self.evidence_refs)}


@dataclass(frozen=True)
class Evidence:
    check_id: str
    check_version: str
    inputs: tuple[str, ...]
    fixture: str
    source_binding: str
    harness_binding: str
    environment_binding: str
    verdict: str
    witness_digest: str
    admission: str
    freshness: str
    provenance_refs: tuple[str, ...] = ()

    def semantic(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "check_version": self.check_version,
            "inputs": list(self.inputs),
            "fixture": self.fixture,
            "source_binding": self.source_binding,
            "harness_binding": self.harness_binding,
            "environment_binding": self.environment_binding,
            "verdict": self.verdict,
            "witness_digest": self.witness_digest,
            "admission": self.admission,
            "freshness": self.freshness,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.semantic(), "provenance_refs": list(self.provenance_refs)}


@dataclass(frozen=True)
class Effect:
    effect_id: str
    status: str
    tool: str
    before_digest: str = ""
    after_digest: str = ""
    artifact_refs: tuple[str, ...] = ()
    unique_observation: bool = False

    def semantic(self) -> dict[str, object]:
        return {"id": self.effect_id, "status": self.status, "tool": self.tool,
                "before_digest": self.before_digest, "after_digest": self.after_digest,
                "artifact_refs": list(self.artifact_refs),
                "unique_observation": self.unique_observation}


@dataclass(frozen=True)
class ObservationCoverage:
    spans: tuple[str, ...] = ()
    cursors: tuple[str, ...] = ()
    revisions: tuple[str, ...] = ()
    complete: bool = False
    final_delivery_proven: bool = False

    def semantic(self) -> dict[str, object]:
        return {"spans": list(self.spans), "cursors": list(self.cursors),
                "revisions": list(self.revisions), "complete": self.complete,
                "final_delivery_proven": self.final_delivery_proven}


@dataclass(frozen=True)
class RepairWitness:
    reachable: bool
    action_ids: tuple[str, ...] = ()
    clause_ids: tuple[str, ...] = ()
    witness_refs: tuple[str, ...] = ()

    def semantic(self) -> dict[str, object]:
        return {"reachable": self.reachable, "action_ids": list(self.action_ids),
                "clause_ids": list(self.clause_ids), "witness_refs": list(self.witness_refs)}


@dataclass(frozen=True)
class KnownBasis:
    request_identity: str
    contract_revision: str
    product: Binding
    harness: Binding
    environment: tuple[Dependency, ...]
    tool_authority: Binding
    selectors: tuple[str, ...]
    clauses: tuple[Clause, ...]
    evidence: tuple[Evidence, ...]
    effects: tuple[Effect, ...]
    unexecuted_batch_suffix: tuple[str, ...]
    coverage: ObservationCoverage
    repair_witness: RepairWitness
    provenance_refs: tuple[str, ...] = field(default_factory=tuple)

    def semantic_payload(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "request_identity": self.request_identity,
            "contract_revision": self.contract_revision,
            "product": self.product.semantic(),
            "harness": self.harness.semantic(),
            "environment": [item.semantic() for item in self.environment],
            "tool_authority": self.tool_authority.semantic(),
            "selectors": list(self.selectors),
            "clauses": [item.semantic() for item in self.clauses],
            "evidence": [item.semantic() for item in self.evidence],
            "effects": [item.semantic() for item in self.effects],
            "unexecuted_batch_suffix": list(self.unexecuted_batch_suffix),
            "coverage": self.coverage.semantic(),
            "repair_witness": self.repair_witness.semantic(),
        }

    @property
    def semantic_key(self) -> str:
        return _digest(self.semantic_payload())

    def to_dict(self) -> dict[str, object]:
        return {"schema": SCHEMA, "kind": "known", "semantic_key": self.semantic_key,
                "semantic": self.semantic_payload(),
                "provenance_refs": list(self.provenance_refs),
                "clauses": [item.to_dict() for item in self.clauses],
                "evidence": [item.to_dict() for item in self.evidence]}


@dataclass(frozen=True)
class UnknownBasis:
    reason: str
    missing_fields: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()

    @property
    def semantic_key(self) -> None:
        return None

    def to_dict(self) -> dict[str, object]:
        return {"schema": SCHEMA, "kind": "unknown", "reason": self.reason,
                "missing_fields": list(self.missing_fields),
                "provenance_refs": list(self.provenance_refs)}


Basis: TypeAlias = KnownBasis | UnknownBasis


def _binding(value: object, *, label: str) -> Binding | None:
    row = value if isinstance(value, Mapping) else {}
    identity = _text(_first(row, ("identity", "id", "name")))
    revision = _text(_first(row, ("revision", "version", "binding_revision")))
    if not identity or not revision:
        return None
    return Binding(
        identity=identity,
        revision=revision,
        digest=_text(row.get("digest")),
        source_digest=_text(_first(row, ("source_digest", "source"))),
        harness_digest=_text(_first(row, ("harness_digest", "harness"))),
        environment_digest=_text(_first(row, ("environment_digest", "environment"))),
    )


def _dependencies(value: object) -> tuple[Dependency, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[Dependency] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        name = _text(item.get("name"))
        version = _text(_first(item, ("version", "revision", "digest")))
        sensitive = bool(item.get("external", False)) or any(
            word in name.casefold() for word in ("time", "random", "network", "external"))
        if not name or (sensitive and not version):
            return None
        result.append(Dependency(name, version, _text(item.get("value")), sensitive))
    return tuple(sorted(result, key=lambda item: (item.name, item.version, item.value)))


def _clauses(value: object) -> tuple[Clause, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[Clause] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        clause_id = _text(_first(item, ("id", "clause_id", "obligation_id")))
        status = _text(_first(item, ("status", "state", "verdict"))).upper()
        if not clause_id or not status:
            return None
        result.append(Clause(
            clause_id=clause_id,
            status=status,
            reason=_text(item.get("reason")),
            applicability=_text(item.get("applicability")),
            freshness=_text(item.get("freshness")),
            evidence_refs=_strings(_first(item, ("evidence_refs", "receipts")), sort=True),
        ))
    return tuple(sorted(result, key=lambda item: item.clause_id))


def _evidence(value: object) -> tuple[Evidence, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[Evidence] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        check_id = _text(_first(item, ("check_id", "id", "check")))
        check_version = _text(_first(item, ("check_version", "version", "revision")))
        verdict = _text(_first(item, ("verdict", "status", "result"))).upper()
        admission = _text(item.get("admission")).upper()
        freshness = _text(item.get("freshness")).upper()
        witness = _text(_first(item, ("witness_digest", "output_digest", "witness")))
        if not all((check_id, check_version, verdict, admission, freshness, witness)):
            return None
        result.append(Evidence(
            check_id=check_id,
            check_version=check_version,
            inputs=_strings(item.get("inputs"), sort=False),
            fixture=_text(item.get("fixture")),
            source_binding=_text(item.get("source_binding")),
            harness_binding=_text(item.get("harness_binding")),
            environment_binding=_text(item.get("environment_binding")),
            verdict=verdict,
            witness_digest=witness,
            admission=admission,
            freshness=freshness,
            provenance_refs=_strings(_first(item, ("provenance_refs", "receipt_refs", "receipts")),
                                     sort=False),
        ))
    return tuple(sorted(result, key=lambda item: (item.check_id, item.check_version,
                                                   item.witness_digest)))


def _effects(value: object) -> tuple[Effect, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    result: list[Effect] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        effect_id = _text(_first(item, ("effect_id", "id")))
        status = _text(item.get("status")).upper()
        tool = _text(item.get("tool")).lower()
        before_digest = _text(item.get("before_digest"))
        after_digest = _text(item.get("after_digest"))
        artifact_refs = _strings(item.get("artifact_refs"), sort=True)
        if (not effect_id or not status or not tool
                or status in {"UNKNOWN", "UNRESOLVED"}
                or (status in {"PENDING", "OPEN"}
                    and not (before_digest or after_digest or artifact_refs))):
            return None
        result.append(Effect(
            effect_id=effect_id,
            status=status,
            tool=tool,
            before_digest=before_digest,
            after_digest=after_digest,
            artifact_refs=artifact_refs,
            unique_observation=bool(item.get("unique_observation", False)),
        ))
    return tuple(sorted(result, key=lambda item: item.effect_id))


def _coverage(value: object) -> ObservationCoverage:
    row = value if isinstance(value, Mapping) else {}
    return ObservationCoverage(
        spans=_strings(row.get("spans"), sort=True),
        cursors=_strings(row.get("cursors"), sort=True),
        revisions=_strings(row.get("revisions"), sort=True),
        complete=bool(row.get("complete", False)),
        final_delivery_proven=bool(row.get("final_delivery_proven", False)),
    )


def _repair(value: object) -> RepairWitness:
    row = value if isinstance(value, Mapping) else {}
    return RepairWitness(
        reachable=bool(row.get("reachable", False)),
        action_ids=_strings(row.get("action_ids"), sort=True),
        clause_ids=_strings(row.get("clause_ids"), sort=True),
        witness_refs=_strings(row.get("witness_refs"), sort=True),
    )


def build_basis(source: Mapping[str, object] | None = None) -> Basis:
    """Build one semantic basis or explicit UNKNOWN without inventing identity.

    The accepted fields are intentionally small and public.  Transport IDs,
    timestamps, sequence numbers, duplicate receipt counts, and presentation
    snapshots are never read as semantic fields.
    """
    root = source if isinstance(source, Mapping) else {}
    request = _section(root, "request")
    contract = _section(root, "contract")
    product = _binding(_section(root, "product") or root.get("product_binding"), label="product")
    harness = _binding(_section(root, "harness") or root.get("harness_binding"), label="harness")
    authority = _binding(_section(root, "tool_authority") or root.get("authority"), label="tool_authority")
    environment = _dependencies(root.get("environment"))
    clauses = _clauses(root.get("clauses"))
    evidence = _evidence(root.get("evidence"))
    effects = _effects(root.get("effects"))
    coverage_source = root.get("observation_coverage")
    repair_source = root.get("repair_witness")
    coverage = _coverage(coverage_source)
    repair = _repair(repair_source)
    selectors_present = "selectors" in root
    coverage_present = (isinstance(coverage_source, Mapping)
                        and bool(coverage.spans or coverage.cursors or coverage.revisions))
    repair_present = isinstance(repair_source, Mapping) and "reachable" in repair_source
    request_identity = _text(_first(request, ("identity", "id", "request_id")))
    contract_revision = _text(_first(contract, ("revision", "version", "identity")))
    missing = tuple(name for name, value in (
        ("request.identity", request_identity),
        ("contract.revision", contract_revision),
        ("product.binding", product),
        ("harness.binding", harness),
        ("environment.dependencies", environment),
        ("tool_authority.binding", authority),
        ("clauses", clauses),
        ("evidence", evidence),
        ("effects", effects),
        ("selectors", selectors_present),
        ("observation_coverage", coverage_present),
        ("repair_witness", repair_present),
    ) if not value)
    if missing:
        return UnknownBasis("missing semantic evidence", missing,
                            _strings(root.get("provenance_refs"), sort=False))
    suffix = _strings(root.get("unexecuted_batch_suffix"), sort=False)
    selectors = _strings(root.get("selectors"), sort=True)
    provenance = _strings(root.get("provenance_refs"), sort=False)
    return KnownBasis(
        request_identity=request_identity,
        contract_revision=contract_revision,
        product=product,
        harness=harness,
        environment=environment,
        tool_authority=authority,
        selectors=selectors,
        clauses=clauses,
        evidence=evidence,
        effects=effects,
        unexecuted_batch_suffix=suffix,
        coverage=coverage,
        repair_witness=repair,
        provenance_refs=provenance,
    )


__all__ = [
    "Basis", "Binding", "Clause", "Dependency", "Effect", "Evidence", "KnownBasis",
    "ObservationCoverage", "RepairWitness", "SCHEMA", "UnknownBasis", "build_basis",
]
