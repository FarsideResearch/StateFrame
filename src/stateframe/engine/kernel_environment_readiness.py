"""Adapter from environment evidence to the existing readiness authority."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

from kernel_environment_state import EnvironmentSnapshot


ENVIRONMENT_CHECK_PREFIX = "environment:"


def _check_row(check: Mapping[str, object]) -> dict[str, object] | None:
    if not bool(check.get("required")):
        return None
    status = str(check.get("status") or "unknown").lower()
    if status == "passed":
        return None
    check_id = str(check.get("check_id") or "unknown")
    row = {
        "check_id": ENVIRONMENT_CHECK_PREFIX + check_id,
        "counterexample_authority": True,
        "repairable": False,
        "reason": "environment check is " + status,
        "clause": str(check.get("predicate") or "required environment predicate"),
        "environment_observation": True,
        "evidence_ref": check.get("evidence_ref"),
        "entity_id": check.get("entity_id"),
        "status": status,
    }
    return row


def required_counterexamples(snapshot: EnvironmentSnapshot) -> tuple[dict[str, object], ...]:
    return tuple(
        row for check in snapshot.required_checks
        if isinstance(check, Mapping)
        for row in [_check_row(check)]
        if row is not None
    )


def augment_status(status: object, snapshot: EnvironmentSnapshot | None) -> object:
    """Return a status view consumed by ``kernel_readiness_authority.decide``.

    Existing readiness remains the reducer.  This adapter only contributes
    explicit required environment predicates, so telemetry cannot become a
    blocker merely because it exists.
    """
    if snapshot is None:
        return status
    rows = required_counterexamples(snapshot)
    if not rows:
        return status
    existing = tuple(getattr(status, "counterexamples", ()) or ())
    existing_ids = {str(row.get("check_id")) for row in existing if isinstance(row, Mapping)}
    merged = existing + tuple(row for row in rows if str(row["check_id"]) not in existing_ids)
    if isinstance(status, SimpleNamespace):
        payload = vars(status).copy()
        payload["counterexamples"] = merged
        return SimpleNamespace(**payload)
    try:
        payload = {
            name: getattr(status, name)
            for name in (
                "counterexamples", "current_revision", "green", "blocking",
                "public_check_status", "selectors", "repair_action",
                "stateframe_repair_blocks_done_mode",
            )
            if hasattr(status, name)
        }
        payload["counterexamples"] = merged
        return SimpleNamespace(**payload)
    except TypeError:
        return status


@dataclass(frozen=True)
class EnvironmentReadiness:
    snapshot_seq: int
    blocker_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_snapshot(cls, snapshot: EnvironmentSnapshot) -> "EnvironmentReadiness":
        rows = required_counterexamples(snapshot)
        return cls(
            snapshot_seq=snapshot.snapshot_seq,
            blocker_ids=tuple(str(row["check_id"]) for row in rows),
            evidence_refs=tuple(
                str(row["evidence_ref"])
                for row in rows
                if row.get("evidence_ref")
            ),
        )


__all__ = [
    "ENVIRONMENT_CHECK_PREFIX", "EnvironmentReadiness", "augment_status",
    "required_counterexamples",
]
