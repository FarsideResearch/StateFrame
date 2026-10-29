"""Clause-to-surface coverage and fail-closed truthful completion."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from kernel_gap_switches import emit, truth_enabled


SWITCH = "STATEFRAME_GAPS_SURFACE_COVERAGE"
PRODUCER = "kernel_gap_surface_coverage"
STATUSES = frozenset({"MAPPED", "UNVERIFIED", "CONTRADICTED", "VERIFIED", "UNRESOLVED"})


@dataclass(frozen=True)
class SurfaceBinding:
    requirement_id: str
    surface_id: str
    clause: str
    priority: int = 0
    product_revision: str = ""


@dataclass(frozen=True)
class CoverageRow:
    binding: SurfaceBinding
    status: str
    evidence_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.binding.requirement_id,
            "surface_id": self.binding.surface_id,
            "clause": self.binding.clause,
            "priority": self.binding.priority,
            "status": self.status,
            "evidence_id": self.evidence_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CoverageReport:
    enabled: bool
    current_revision: str
    rows: tuple[CoverageRow, ...]
    decision: str
    next_surface: str

    @property
    def done_allowed(self) -> bool:
        return self.enabled and self.decision == "READY"

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "current_revision": self.current_revision,
            "rows": [row.to_dict() for row in self.rows],
            "decision": self.decision,
            "next_surface": self.next_surface,
            "done_allowed": self.done_allowed,
        }


def _binding(value: SurfaceBinding | Mapping[str, object]) -> SurfaceBinding:
    if isinstance(value, SurfaceBinding):
        return value
    return SurfaceBinding(
        str(value.get("requirement_id", "")), str(value.get("surface_id", "")),
        str(value.get("clause", "")), int(value.get("priority", 0) or 0),
        str(value.get("product_revision", "")),
    )


def reduce_coverage(bindings: Sequence[SurfaceBinding | Mapping[str, object]],
                    receipts: Sequence[Mapping[str, object]], *, current_revision: str,
                    journal: object | None = None) -> CoverageReport:
    if not truth_enabled(SWITCH):
        return CoverageReport(False, str(current_revision), (), "DISABLED", "")
    rows: list[CoverageRow] = []
    for raw_binding in bindings:
        binding = _binding(raw_binding)
        if (binding.product_revision
                and binding.product_revision != str(current_revision)):
            rows.append(CoverageRow(binding, "UNVERIFIED",
                                    reason="binding_revision_stale"))
            continue
        matches = [receipt for receipt in receipts
                   if str(receipt.get("requirement_id", receipt.get("check_id", "")))
                   == binding.requirement_id
                   and str(receipt.get("surface_id", "")) == binding.surface_id]
        current = [receipt for receipt in matches
                   if str(receipt.get("product_revision", receipt.get("revision", "")))
                   == str(current_revision)]
        passing = next((receipt for receipt in reversed(current)
                        if str(receipt.get("status", "")).upper() in {"PASS", "VERIFIED", "SATISFIED"}), None)
        contradictory = next((receipt for receipt in reversed(current)
                              if str(receipt.get("status", "")).upper()
                              in {"FAIL", "COUNTEREXAMPLE", "VIOLATED"}), None)
        unresolved = next((receipt for receipt in reversed(current)
                           if str(receipt.get("status", "")).upper() in {"UNRESOLVED", "UNKNOWN"}), None)
        requirement_matches = [receipt for receipt in receipts
                               if str(receipt.get("requirement_id", receipt.get("check_id", "")))
                               == binding.requirement_id]
        if passing is not None and str(passing.get("evidence_id", passing.get("receipt_id", ""))):
            row = CoverageRow(binding, "VERIFIED",
                              str(passing.get("evidence_id", passing.get("receipt_id", ""))),
                              "same_surface_same_revision")
        elif contradictory is not None:
            row = CoverageRow(binding, "CONTRADICTED",
                              str(contradictory.get("evidence_id", contradictory.get("receipt_id", ""))),
                              "current_counterexample")
        elif unresolved is not None:
            row = CoverageRow(binding, "UNRESOLVED",
                              str(unresolved.get("evidence_id", unresolved.get("receipt_id", ""))),
                              "interpretation_or_observation_unresolved")
        elif matches or requirement_matches:
            row = CoverageRow(binding, "UNVERIFIED", reason="wrong_surface_or_revision")
        else:
            row = CoverageRow(binding, "MAPPED", reason="no_evidence")
        rows.append(row)
    blockers = [row for row in rows if row.status == "CONTRADICTED"]
    if blockers:
        decision = "BLOCKED"
    elif rows and all(row.status == "VERIFIED" for row in rows):
        decision = "READY"
    else:
        decision = "UNRESOLVED"
    pending = sorted((row for row in rows if row.status != "VERIFIED"),
                     key=lambda row: (-row.binding.priority, row.binding.surface_id))
    next_surface = pending[0].binding.surface_id if pending else ""
    report = CoverageReport(True, str(current_revision), tuple(rows), decision, next_surface)
    emit(journal, "gaps_surface_coverage", producer=PRODUCER, enabled_value=True,
         decision=report.decision, row_count=len(report.rows),
         verified_count=sum(row.status == "VERIFIED" for row in report.rows),
         next_surface=report.next_surface)
    return report


def render_coverage(report: CoverageReport) -> str:
    if not report.enabled:
        return ""
    return "[Clause-to-surface coverage]\n" + json.dumps({
        "decision": report.decision,
        "next_surface": report.next_surface,
        "rows": [row.to_dict() for row in report.rows[:8]],
    }, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CoverageReport", "CoverageRow", "STATUSES", "SWITCH", "SurfaceBinding",
    "reduce_coverage", "render_coverage",
]
