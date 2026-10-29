"""Host-side admission evidence for newly enabled release mechanisms.

This module is deliberately separate from request preparation.  It validates
the evidence attached to a frozen release; it does not add a runtime token,
journal field, or model call.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

SWITCH_ENV = "STATEFRAME_MECHANISM_ADMISSION"
REQUIRED_ENV = "STATEFRAME_MECHANISM_ADMISSION_REQUIRED"
ADMISSION_FILE = "mechanism-admission.json"
MANIFEST_FIELD = "mechanism_admission"
SCHEMA = "stateframe-mechanism-admission/1"

_REQUIRED_FIELDS = (
    "newly_enabled",
    "reproduced_failure",
    "hypothesis",
    "live_path_proof",
    "matched_on_off",
    "ablation_plan",
    "rollback",
)
_UNSET = object()


@dataclass(frozen=True)
class AdmissionPolicy:
    """The immutable admission decision used by freeze and runner paths."""

    enabled: bool
    required_switches: tuple[str, ...]
    declaration_present: bool
    chain_mode: str
    run_mode: str


def mode(value: object | None = None) -> str:
    """Return the strict gate mode; an unknown value fails closed."""
    raw = os.environ.get(SWITCH_ENV, "off") if value is None else value
    if raw not in {"off", "on"}:
        raise ValueError(f"{SWITCH_ENV} must be exactly 'off' or 'on'")
    return raw


def admission_enabled() -> bool:
    return mode() == "on"


def required_switches(value: object = _UNSET) -> list[str]:
    """Parse a promotion-switch declaration without inventing one when absent."""
    if value is _UNSET:
        raw = os.environ[REQUIRED_ENV] if REQUIRED_ENV in os.environ else ""
    else:
        raw = value
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = list(raw)
    else:
        raise ValueError(f"{REQUIRED_ENV} must be comma-separated text or a sequence")
    result = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{REQUIRED_ENV} contains an empty switch")
        result.append(item.strip())
    return sorted(set(result))


def declared_required_switches(environment: Mapping | None = None) -> tuple[bool, list[str]]:
    """Return whether the promotion set was explicitly supplied and its values."""
    values = os.environ if environment is None else environment
    if not isinstance(values, Mapping):
        raise ValueError("admission environment must be a mapping")
    if REQUIRED_ENV not in values:
        return False, []
    return True, required_switches(values[REQUIRED_ENV])


def resolve_policy(
    run_environment: Mapping | None = None,
    chain_environment: Mapping | None = None,
) -> AdmissionPolicy:
    """Combine chain and run policy without allowing a run to weaken the chain."""
    chain = os.environ if chain_environment is None else chain_environment
    run = {} if run_environment is None else run_environment
    if not isinstance(chain, Mapping) or not isinstance(run, Mapping):
        raise ValueError("admission environments must be mappings")
    chain_mode = mode(chain.get(SWITCH_ENV, "off"))
    run_mode = mode(run.get(SWITCH_ENV, "off"))
    chain_declared, chain_required = declared_required_switches(chain)
    run_declared, run_required = declared_required_switches(run)
    required = tuple(sorted(set(chain_required) | set(run_required)))
    enabled = chain_mode == "on" or run_mode == "on"
    return AdmissionPolicy(
        enabled=enabled,
        required_switches=required,
        declaration_present=chain_declared or run_declared,
        chain_mode=chain_mode,
        run_mode=run_mode,
    )


def require_declaration(policy: AdmissionPolicy) -> None:
    """Reject an enabled gate that has no independent promotion declaration."""
    if policy.enabled and not policy.declaration_present:
        raise ValueError(
            f"{REQUIRED_ENV} promotion switch declaration is required when "
            f"{SWITCH_ENV}=on"
        )


def _switches(value: object) -> list[str]:
    return required_switches(value)


def _mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    return value


def _text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")


def _nonempty(value: object, label: str) -> None:
    if isinstance(value, str):
        _text(value, label)
    elif isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray))
    ):
        if not value:
            raise ValueError(f"{label} must not be empty")
    else:
        raise ValueError(f"{label} must be non-empty text, object, or sequence")


def _marked(value: Mapping, flag: str, status: str, label: str) -> None:
    if value.get(flag) is not True and str(value.get("status", "")).upper() != status:
        raise ValueError(f"{label} must be marked {status}")


def _validate_entry(entry: object) -> dict:
    value = dict(_mapping(entry, "admission entry"))
    for field in _REQUIRED_FIELDS:
        if field not in value:
            raise ValueError(f"admission entry missing {field}")

    _text(value.get("mechanism"), "mechanism")
    _text(value.get("switch"), "switch")
    if value.get("newly_enabled") is not True:
        raise ValueError("newly_enabled must be true")

    failure = _mapping(value["reproduced_failure"], "reproduced_failure")
    _text(failure.get("id"), "reproduced_failure.id")
    _text(failure.get("evidence"), "reproduced_failure.evidence")
    _marked(failure, "reproduced", "REPRODUCED", "reproduced_failure")

    _text(value["hypothesis"], "hypothesis")

    proof = _mapping(value["live_path_proof"], "live_path_proof")
    _text(proof.get("consumer"), "live_path_proof.consumer")
    _text(proof.get("evidence"), "live_path_proof.evidence")
    trace = proof.get("trace")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes, bytearray)) or not trace:
        raise ValueError("live_path_proof.trace must be a non-empty sequence")
    for index, step in enumerate(trace):
        _text(step, f"live_path_proof.trace[{index}]")
    _marked(proof, "proved", "PROVED", "live_path_proof")

    comparison = _mapping(value["matched_on_off"], "matched_on_off")
    if comparison.get("matched") is not True and str(comparison.get("status", "")).upper() != "MATCHED":
        raise ValueError("matched_on_off must be marked MATCHED")
    _nonempty(comparison.get("on"), "matched_on_off.on")
    _nonempty(comparison.get("off"), "matched_on_off.off")
    _nonempty(comparison.get("comparison"), "matched_on_off.comparison")

    _nonempty(value["ablation_plan"], "ablation_plan")

    rollback = _mapping(value["rollback"], "rollback")
    _text(rollback.get("switch"), "rollback.switch")
    if rollback["switch"] != value["switch"]:
        raise ValueError("rollback.switch must match switch")
    _nonempty(rollback.get("criterion"), "rollback.criterion")
    return value


def _entries(value: object) -> list[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("admission entries must be a sequence")
    return list(value)


def validate_manifest(value: object, required_switches: object | None = None) -> dict:
    """Validate a manifest block and return a detached, normalized copy."""
    block = _mapping(value, MANIFEST_FIELD)
    if block.get("schema") != SCHEMA:
        raise ValueError(f"{MANIFEST_FIELD} has an unsupported schema")
    entries = [_validate_entry(item) for item in _entries(block.get("entries"))]
    switches = [entry["switch"] for entry in entries]
    if len(set(switches)) != len(switches):
        raise ValueError("admission entries must have unique switches")
    mechanisms = [entry["mechanism"] for entry in entries]
    if len(set(mechanisms)) != len(mechanisms):
        raise ValueError("admission entries must have unique mechanisms")

    declared = _switches(block.get("required_switches", []))
    requested = _switches(required_switches) if required_switches is not None else []
    missing_declared = sorted(set(declared) - set(switches))
    if missing_declared:
        raise ValueError(
            "manifest-declared switch has no admission entry: "
            + ", ".join(missing_declared)
        )
    missing = sorted(set(requested) - set(switches))
    if missing:
        raise ValueError(f"required switch has no admission entry: {', '.join(missing)}")
    return {
        "schema": SCHEMA,
        "required_switches": sorted(set(declared) | set(requested)),
        "entries": entries,
    }


def build_manifest(value: object, required_switches: object | None = None) -> dict:
    """Build the canonical block embedded in ``release-manifest.json``."""
    if isinstance(value, Mapping) and "entries" in value:
        raw = dict(value)
    else:
        raw = {"schema": SCHEMA, "entries": value}
    raw.setdefault("schema", SCHEMA)
    return validate_manifest(raw, required_switches=required_switches)


def load_manifest(source: Path, required_switches: object | None = None) -> dict:
    path = Path(source) / ADMISSION_FILE
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{ADMISSION_FILE} is required when {SWITCH_ENV}=on")
    try:
        raw = json.loads(path.resolve(strict=True).read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid {ADMISSION_FILE}") from exc
    return build_manifest(raw, required_switches=required_switches)
