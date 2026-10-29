"""Host-derived coverage for tool observations (rulebook candidate N2).

Inert until ``STATEFRAME_PARTIAL_OBSERVATION=on``.  Coverage is computed by the
HOST from its own truncation code paths (for grep: the number of matching lines
the subprocess returned versus the number the runner kept) and bound to the
host receipt id.  Nothing here parses the model-visible text, so a tool output
that merely *says* "coverage: complete" cannot promote itself.

Two effects, both precision-first (observation, never a completion veto):

* the observation the model receives carries a host-authored coverage note when
  the host cut the result; and
* an absence/completeness claim (the where-used ``NOT_AFFECTED`` disposition,
  the one such decision the host adjudicates today) is rejected with
  ``OBSERVATION_PARTIAL(ref)`` when its cited receipt is not COMPLETE.  Only an
  actual rejection emits ``partial_obs_rejected``.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final


MODE_ENV: Final = "STATEFRAME_PARTIAL_OBSERVATION"
SCHEMA: Final = "rulebook-partial-observation/2"

COMPLETE: Final = "COMPLETE"
TRUNCATED: Final = "TRUNCATED"
UNKNOWN: Final = "UNKNOWN"


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the validated switch value; absent means the legacy path."""
    values = os.environ if environ is None else environ
    value = str(values.get(MODE_ENV, "off")).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


@dataclass(frozen=True)
class Coverage:
    tool: str
    reference: str
    status: str
    shown: int
    total: int

    @property
    def complete(self) -> bool:
        return self.status == COMPLETE

    def rendered(self) -> str:
        if self.status == TRUNCATED:
            return f"TRUNCATED({self.shown},{self.total})"
        return self.status

    def as_dict(self) -> dict[str, object]:
        return {"tool": self.tool, "reference": self.reference, "status": self.status,
                "shown": self.shown, "total": self.total}


def host_coverage(tool: str, reference: str, *, shown: int, total: int) -> Coverage:
    """Coverage from host counts: shown < total is a host truncation."""
    shown, total = max(0, int(shown)), max(0, int(total))
    status = COMPLETE if shown >= total else TRUNCATED
    return Coverage(str(tool), str(reference)[:120], status, shown, total)


def coverage_note(coverage: Coverage) -> str:
    """Host-authored line appended to a CUT observation ("" when complete)."""
    if coverage.complete:
        return ""
    return (f"[coverage: {coverage.rendered()} — the host showed {coverage.shown} of "
            f"{coverage.total} matching lines; this is NOT the complete result. "
            "Narrow the pattern or scope before concluding that something is absent.]")


class ObservationLedger:
    """Receipt-id → host coverage; adjudicates absence/completeness claims."""

    def __init__(self) -> None:
        self._by_receipt: dict[str, Coverage] = {}

    def record(self, receipt_id: str, coverage: Coverage) -> None:
        if receipt_id:
            self._by_receipt[str(receipt_id)] = coverage

    def get(self, receipt_id: str) -> Coverage | None:
        return self._by_receipt.get(str(receipt_id))

    def partial_refs(self, evidence_refs: Sequence[str]) -> tuple[str, ...]:
        """Refs whose HOST-RECORDED coverage is not COMPLETE.

        Precision-first: only a receipt the host itself recorded as cut is
        rejected.  Receipts from tools without an instrumented truncation path
        (not in the ledger) are not adjudicated here, so this gate cannot
        starve an otherwise valid disposition of evidence.
        """
        return tuple(str(ref) for ref in evidence_refs
                     if (coverage := self._by_receipt.get(str(ref))) is not None
                     and not coverage.complete)


def rejection_text(refs: Sequence[str]) -> str:
    return ("OBSERVATION_PARTIAL(" + ",".join(refs) + "): the cited evidence is not a "
            "complete observation, so it cannot support an absence/completeness claim. "
            "Re-run the query uncut (narrower pattern or scope) and cite that receipt.")


__all__ = [
    "COMPLETE", "Coverage", "MODE_ENV", "ObservationLedger", "SCHEMA", "TRUNCATED",
    "UNKNOWN", "coverage_note", "host_coverage", "mode", "rejection_text",
]
