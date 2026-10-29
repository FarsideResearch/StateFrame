"""Admission checks for information-balanced StateFrame comparisons.

The comparator receives only a small projection of the run manifest.  It never
needs the symptom text or its hashes: repair mode, brief provenance, and the
declared information policy are sufficient to keep unlike cohorts out of a
paired claim.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any


SWITCH = "STATEFRAME_COMPARISON_ADMISSION"
SCHEMA = "comparison-admission/1"
PROVENANCE_FIELDS = (
    "repair_mode",
    "brief_provenance",
    "information_policy_version",
    "initial_oracle_symptom",
    "live_grader_feedback",
    "target_disclosure",
)
_REPAIR_MODES = frozenset({"ORIGINAL_TASK", "ASSISTED_REPAIR"})
_EXPECTED_BRIEF_PROVENANCE = {
    "ORIGINAL_TASK": "task_and_existing_baseline_brief_policy",
    "ASSISTED_REPAIR": "held_out_oracle_failure",
}
_POLICY_VALUES = {
    "brief_provenance": frozenset({
        "task_and_existing_baseline_brief_policy",
        "held_out_oracle_failure",
    }),
    "information_policy_version": frozenset({"information-policy/1"}),
    "initial_oracle_symptom": frozenset({"ON", "OFF"}),
    "live_grader_feedback": frozenset({"FINAL_ONLY", "STATUS_ONLY", "DIAGNOSTIC"}),
    "target_disclosure": frozenset({"FILE_AND_SYMBOL"}),
}


def enabled() -> bool:
    """Return whether comparison admission is explicitly enabled."""

    return os.environ.get(SWITCH, "off").strip().lower() in {
        "1", "true", "yes", "on",
    }


def provenance_from_manifest(manifest: Mapping[str, Any] | object) -> dict[str, Any]:
    """Project only the fields needed for admission from a runtime manifest."""

    if not isinstance(manifest, Mapping):
        return {}
    return {field: manifest[field] for field in PROVENANCE_FIELDS if field in manifest}


def _record_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    projected = record.get("comparison_provenance")
    if isinstance(projected, Mapping):
        return dict(projected)
    manifest = record.get("manifest")
    if isinstance(manifest, Mapping):
        return provenance_from_manifest(manifest)
    return provenance_from_manifest(record)


def _identity(record: Mapping[str, Any]) -> dict[str, str]:
    return {
        field: str(record.get(field, ""))
        for field in ("run", "task", "arm")
    }


def _validate(provenance: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    missing = [field for field in PROVENANCE_FIELDS if field not in provenance]
    invalid: list[str] = []
    for field in PROVENANCE_FIELDS:
        if field in missing:
            continue
        value = provenance[field]
        if not isinstance(value, str) or not value.strip():
            invalid.append(field)
            continue
        if field == "repair_mode" and value not in _REPAIR_MODES:
            invalid.append(field)
            continue
        allowed = _POLICY_VALUES.get(field)
        if allowed is not None and value not in allowed:
            invalid.append(field)
    repair_mode = provenance.get("repair_mode")
    brief_provenance = provenance.get("brief_provenance")
    if (repair_mode in _EXPECTED_BRIEF_PROVENANCE
            and brief_provenance in _POLICY_VALUES["brief_provenance"]
            and brief_provenance != _EXPECTED_BRIEF_PROVENANCE[repair_mode]):
        invalid.extend(("repair_mode", "brief_provenance"))
    return missing, list(dict.fromkeys(invalid))


def _report() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "enabled": True,
        "admitted": False,
        "status": "REJECTED",
        "reason": "unknown",
        "cohort": "UNKNOWN",
        "repair_modes": [],
        "missing_provenance": [],
        "invalid_provenance": [],
        "excluded_runs": [],
        "admitted_runs": [],
        "policy_signatures": [],
    }


def admit_comparison(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Return a safe admission receipt, or ``None`` while the switch is off.

    A comparison is admitted only when every attempt has complete, supported
    provenance and every projected field is identical across the cohort.  A
    missing or malformed historical manifest therefore yields an explicit
    rejection rather than silently shrinking the comparison.
    """

    if not enabled():
        return None

    report = _report()
    if not records:
        report["reason"] = "no_attempts"
        return report

    valid: list[tuple[dict[str, str], dict[str, Any]]] = []
    for record in records:
        identity = _identity(record)
        provenance = _record_provenance(record)
        missing, invalid = _validate(provenance)
        if missing:
            report["missing_provenance"].append({
                **identity,
                "fields": missing,
            })
        if invalid:
            report["invalid_provenance"].append({
                **identity,
                "fields": invalid,
            })
        if missing or invalid:
            report["excluded_runs"].append(identity)
            continue
        valid.append((identity, {
            field: provenance[field] for field in PROVENANCE_FIELDS
        }))

    if report["missing_provenance"] or report["invalid_provenance"]:
        report["reason"] = "missing_or_invalid_provenance"
        return report

    modes = sorted({provenance["repair_mode"] for _, provenance in valid})
    report["repair_modes"] = modes
    report["cohort"] = modes[0] if len(modes) == 1 else "MIXED"
    signatures: dict[tuple[str, ...], dict[str, Any]] = {}
    for _, provenance in valid:
        signature = tuple(provenance[field] for field in PROVENANCE_FIELDS)
        signatures.setdefault(signature, provenance)
    report["policy_signatures"] = list(signatures.values())
    report["admitted_runs"] = [identity for identity, _ in valid]

    if len(modes) > 1:
        report["reason"] = "mixed_repair_modes"
        return report
    if len(signatures) > 1:
        report["reason"] = "mixed_information_policy"
        return report

    report["admitted"] = True
    report["status"] = "ADMITTED"
    report["reason"] = "uniform_provenance"
    return report
