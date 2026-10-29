"""Bounded, disposable edit-site sensitivity for frozen authored checks."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from kernel_authored_check_lifecycle import (
    ExecutionResult,
    FrozenCheck,
    LifecycleError,
    PASS,
    UNKNOWN,
    UNRUNNABLE,
    execute_check,
)


SCHEMA = "stateframe-authored-check-mutation/1"
EDIT_SENSITIVE = "EDIT_SENSITIVE"
SURVIVED = "SURVIVED"
INVALID_MUTANT = "INVALID_MUTANT"
CLONE = "CLONE"
OMITTED = "OMITTED"


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise LifecycleError("mutation sources must be text or bytes")


def _files(value: object) -> dict[str, bytes]:
    raw = value.get("files", value) if isinstance(value, Mapping) else {}
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, bytes] = {}
    for raw_path, source in raw.items():
        path = str(raw_path).replace("\\", "/")
        parts = PurePosixPath(path).parts
        if not path or path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
            continue
        result[path] = _bytes(source)
    return result


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _region_offsets(region: Mapping[str, object], source: bytes) -> tuple[int, int] | None:
    start = region.get("start_offset", region.get("start"))
    end = region.get("end_offset", region.get("end"))
    if type(start) is int and type(end) is int:
        return start, end
    start_line, end_line = region.get("start_line"), region.get("end_line")
    if type(start_line) is not int or type(end_line) is not int or start_line < 1 or end_line < start_line:
        return None
    lines = source.splitlines(keepends=True)
    if end_line > len(lines):
        return None
    begin = sum(len(item) for item in lines[:start_line - 1])
    finish = sum(len(item) for item in lines[:end_line])
    return begin, finish


@dataclass(frozen=True)
class MutationVariant:
    variant_id: str
    path: str
    start: int
    end: int
    group_id: str
    files: Mapping[str, bytes]
    status: str = "PLANNED"
    reason: str = ""

    def snapshot(self, revision: str) -> dict[str, object]:
        return {"revision": revision, "files": dict(self.files), "binding_stable": True}

    def as_dict(self) -> dict[str, object]:
        return {"variant_id": self.variant_id, "path": self.path, "start": self.start,
                "end": self.end, "group_id": self.group_id, "status": self.status,
                "reason": self.reason, "files_digest": _digest({
                    path: _sha(source) for path, source in sorted(self.files.items())})}


@dataclass(frozen=True)
class MutationPlan:
    check_identity: str
    candidate_revision: str
    parent_revision: str
    variants: tuple[MutationVariant, ...]
    chosen_regions: tuple[Mapping[str, object], ...]
    omitted_regions: tuple[Mapping[str, object], ...]
    budget_status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA, "check_identity": self.check_identity,
            "candidate_revision": self.candidate_revision, "parent_revision": self.parent_revision,
            "variants": [item.as_dict() for item in self.variants],
            "chosen_regions": [dict(item) for item in self.chosen_regions],
            "omitted_regions": [dict(item) for item in self.omitted_regions],
            "budget_status": self.budget_status,
        }


@dataclass(frozen=True)
class MutationReceipt:
    """Diagnostic sensitivity evidence linked to an existing authored receipt."""

    check_identity: str
    variant_id: str
    result: str
    reason: str
    authored_receipt_id: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MutationReceipt":
        variant = value.get("variant")
        variant_id = variant.get("variant_id", "") if isinstance(variant, Mapping) else ""
        return cls(
            check_identity=str(value.get("check_identity", "")),
            variant_id=str(variant_id),
            result=str(value.get("result", "")),
            reason=str(value.get("reason", "")),
            authored_receipt_id=str(value.get("authored_receipt_id", "")),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA, "check_identity": self.check_identity,
            "variant_id": self.variant_id, "result": self.result,
            "reason": self.reason, "authored_receipt_id": self.authored_receipt_id,
        }


def _region_list(edit_receipt: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    value = edit_receipt.get("regions", edit_receipt.get("hunks", ()))
    return tuple(item for item in value if isinstance(item, Mapping)) \
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def plan_edit_mutation(parent_ref: Mapping[str, object], candidate_ref: Mapping[str, object],
                       edit_receipt: Mapping[str, object], check_ref: FrozenCheck,
                       budget: Mapping[str, object] | None = None) -> MutationPlan:
    """Build scratch variants from immutable bytes; never writes the workspace."""
    if not isinstance(check_ref, FrozenCheck):
        raise LifecycleError("mutation planning requires FrozenCheck")
    parent_files, candidate_files = _files(parent_ref), _files(candidate_ref)
    parent_revision = str(parent_ref.get("revision", parent_ref.get("product_revision", "")))
    candidate_revision = str(candidate_ref.get("revision", candidate_ref.get("product_revision", "")))
    budget = budget if isinstance(budget, Mapping) else {}
    max_variants = budget.get("max_variants", 2)
    max_variants = max_variants if type(max_variants) is int and max_variants >= 0 else 2
    regions = _region_list(edit_receipt)
    if parent_files == candidate_files:
        clone = MutationVariant(
            variant_id=_digest({"check": check_ref.identity, "files": _digest(candidate_files)}),
            path="", start=0, end=0, group_id="", files=dict(candidate_files), status=CLONE,
            reason="candidate_bytes_equal_parent_bytes",
        )
        return MutationPlan(check_ref.identity, candidate_revision, parent_revision, (clone,), (), (), "CLONE")
    variants: list[MutationVariant] = []
    chosen: list[Mapping[str, object]] = []
    omitted: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for region in regions:
        path = str(region.get("path", "")).replace("\\", "/")
        parent_source, candidate_source = parent_files.get(path), candidate_files.get(path)
        if parent_source is None or candidate_source is None:
            omitted.append({"path": path, "reason": "region_file_missing"})
            continue
        if region.get("parent_sha256") and str(region["parent_sha256"]) != _sha(parent_source):
            omitted.append({"path": path, "reason": "parent_hash_mismatch"})
            continue
        if region.get("candidate_sha256") and str(region["candidate_sha256"]) != _sha(candidate_source):
            omitted.append({"path": path, "reason": "candidate_hash_mismatch"})
            continue
        offsets = _region_offsets(region, candidate_source)
        parent_offsets = _region_offsets(region, parent_source)
        if offsets is None or parent_offsets is None:
            omitted.append({"path": path, "reason": "region_offsets_invalid"})
            continue
        start, end = offsets
        parent_start, parent_end = parent_offsets
        if not (0 <= start <= end <= len(candidate_source)
                and 0 <= parent_start <= parent_end <= len(parent_source)):
            omitted.append({"path": path, "reason": "region_bounds_invalid"})
            continue
        files = dict(candidate_files)
        files[path] = candidate_source[:start] + parent_source[parent_start:parent_end] + candidate_source[end:]
        files_digest = _digest({path: _sha(source) for path, source in sorted(files.items())})
        if files_digest in seen:
            omitted.append({"path": path, "reason": "equivalent_mutation"})
            continue
        seen.add(files_digest)
        if len(variants) >= max_variants:
            omitted.append({"path": path, "reason": "mutation_budget_exhausted"})
            continue
        group_id = str(region.get("group_id", ""))
        variant_id = _digest({"check": check_ref.identity, "files": files_digest})
        variants.append(MutationVariant(variant_id, path, start, end, group_id, files))
        chosen.append({"path": path, "start": start, "end": end,
                       "group_id": group_id, "variant_id": variant_id})
    return MutationPlan(check_ref.identity, candidate_revision, parent_revision,
                        tuple(variants), tuple(chosen), tuple(omitted),
                        "OK" if variants else "NO_APPLICABLE_VARIANT")


def execute_mutation(check: FrozenCheck, candidate_snapshot: Mapping[str, object],
                     variant: MutationVariant, capability: object = None,
                     candidate_execution: ExecutionResult | None = None) -> dict[str, object]:
    """Run the unchanged check on candidate and one disposable mutant."""
    if variant.status == CLONE:
        return {"schema": SCHEMA, "status": CLONE, "result": CLONE,
                "check_identity": check.identity, "variant": variant.as_dict()}
    candidate = candidate_execution or execute_check(check, candidate_snapshot, capability)
    mutant = execute_check(check, variant.snapshot(str(candidate_snapshot.get("revision", "mutant"))), capability)
    candidate_status = candidate.status
    if (candidate.check_identity != check.identity or candidate_status != PASS
            or not candidate.reached_target or not candidate.integrity
            or not candidate.binding_stable):
        return {"schema": SCHEMA, "status": UNKNOWN, "result": UNKNOWN,
                "check_identity": check.identity, "variant": variant.as_dict(),
                "reason": "candidate_did_not_prove_pass"}
    if mutant.status in {UNRUNNABLE, "ENVIRONMENT", "UNSUPPORTED", "UNKNOWN"} or not mutant.integrity:
        result = INVALID_MUTANT
        reason = mutant.reason or "mutant_setup_or_observation_invalid"
    elif (mutant.status == "BEHAVIOR_FAIL" and mutant.reached_target
          and mutant.integrity and mutant.binding_stable):
        result, reason = EDIT_SENSITIVE, "candidate_pass_mutant_behavior_fail"
    else:
        result, reason = SURVIVED, "mutant_survived_same_observation"
    return {
        "schema": SCHEMA, "status": result, "result": result,
        "check_identity": check.identity, "check_hash": check.check_hash,
        "variant": variant.as_dict(), "candidate": candidate.as_dict(),
        "mutant": mutant.as_dict(), "reason": reason,
    }


def attach_mutation_receipt(runtime: object, receipt: Mapping[str, object]) -> bool:
    """Retain mutation evidence beside, never inside, the completion receipt."""
    if not isinstance(receipt, Mapping):
        return False
    check_identity = str(receipt.get("check_identity", "")).strip()
    variant = receipt.get("variant")
    variant_id = str(variant.get("variant_id", "")).strip() if isinstance(variant, Mapping) else ""
    if not check_identity or not variant_id or not str(receipt.get("result", "")).strip():
        return False
    authored = getattr(runtime, "_authored_check_execution_receipts", None)
    if not isinstance(authored, Mapping):
        return False
    parent = authored.get(check_identity)
    if not isinstance(parent, Mapping):
        parent = next((item for item in authored.values()
                       if isinstance(item, Mapping) and item.get("check_identity") == check_identity), None)
    if not isinstance(parent, Mapping) or not str(parent.get("receipt_id", "")).strip():
        return False
    data = dict(receipt)
    data["authored_receipt_id"] = str(parent["receipt_id"])
    store = getattr(runtime, "_authored_check_mutation_receipts", None)
    if not isinstance(store, dict):
        store = {}
        setattr(runtime, "_authored_check_mutation_receipts", store)
    store.setdefault((check_identity, variant_id), data)
    return True


__all__ = [
    "CLONE", "EDIT_SENSITIVE", "INVALID_MUTANT", "MutationPlan", "MutationReceipt",
    "MutationVariant", "OMITTED", "SCHEMA", "SURVIVED", "attach_mutation_receipt",
    "execute_mutation", "plan_edit_mutation",
]
