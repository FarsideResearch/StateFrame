"""Wiring census and OUTLINEGENERIC seam for ORPHAN_C6.

The census is an integration handoff, not a live-path registrar.  It records
which proposals have a new adapter, which current owner still needs a firing
qualification, and which historical owner is absent from ``src_v46``.  The
BUILDLEAD session is responsible for integrating an adapter into a protected
runtime owner.

Source-owner checks consume the existing ``mc-outline/1`` API.  They inspect
the shared records (imports, module statements, types, and functions) and do
not contain language-specific parsers or Python AST logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum


SCHEMA = "stateframe-orphan-c6-wiring/1"


class WiringStatus(str, Enum):
    ADAPTER_READY = "ADAPTER_READY"
    QUALIFICATION_REQUIRED = "QUALIFICATION_REQUIRED"
    DEFERRED_MISSING_OWNER = "DEFERRED_MISSING_OWNER"
    DEFERRED_DEFAULT_OFF = "DEFERRED_DEFAULT_OFF"
    EXISTING_CONCURRENT_OWNER = "EXISTING_CONCURRENT_OWNER"


class OutlineClaimStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True)
class WiringGap:
    gap_id: str
    report_member: str
    proposal: str
    status: WiringStatus
    owner_path: str | None
    owner_symbol: str | None
    consumer: str
    integration_note: str

    def __post_init__(self) -> None:
        for field in ("gap_id", "report_member", "proposal", "consumer", "integration_note"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty")
        if self.owner_path is not None and not self.owner_path.strip():
            raise ValueError("owner_path cannot be blank")
        if self.owner_symbol is not None and not self.owner_symbol.strip():
            raise ValueError("owner_symbol cannot be blank")

    def to_dict(self) -> dict[str, object]:
        return {
            "gap_id": self.gap_id,
            "report_member": self.report_member,
            "proposal": self.proposal,
            "status": self.status.value,
            "owner_path": self.owner_path,
            "owner_symbol": self.owner_symbol,
            "consumer": self.consumer,
            "integration_note": self.integration_note,
        }


@dataclass(frozen=True)
class WiringCensus:
    gaps: tuple[WiringGap, ...]

    def __post_init__(self) -> None:
        ids = [gap.gap_id for gap in self.gaps]
        if len(ids) != len(set(ids)):
            raise ValueError("wiring gap IDs must be unique")

    def by_id(self, gap_id: str) -> WiringGap:
        for gap in self.gaps:
            if gap.gap_id == gap_id:
                return gap
        raise KeyError(gap_id)

    def to_dict(self) -> dict[str, object]:
        return {"schema": SCHEMA, "gaps": [gap.to_dict() for gap in self.gaps]}


@dataclass(frozen=True)
class OutlineClaim:
    name: str
    kind: str = "symbol"
    line: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("outline claim name must be non-empty")
        if self.kind not in {"symbol", "module_statement"}:
            raise ValueError("outline claim kind is unsupported")
        if self.line is not None and (isinstance(self.line, bool) or not isinstance(self.line, int) or self.line < 1):
            raise ValueError("outline claim line must be positive")


@dataclass(frozen=True)
class OutlineClaimResult:
    claim: OutlineClaim
    status: OutlineClaimStatus
    path: str
    language: str
    parser: str
    degraded: bool
    line: int | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.claim.name,
            "kind": self.claim.kind,
            "requested_line": self.claim.line,
            "status": self.status.value,
            "path": self.path,
            "language": self.language,
            "parser": self.parser,
            "degraded": self.degraded,
            "line": self.line,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class OutlineClaimReport:
    schema: str
    results: tuple[OutlineClaimResult, ...]
    degraded: bool

    @property
    def confirmed(self) -> bool:
        return bool(self.results) and all(result.status is OutlineClaimStatus.FOUND for result in self.results)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "confirmed": self.confirmed,
            "degraded": self.degraded,
            "results": [result.to_dict() for result in self.results],
        }


def default_wiring_census() -> WiringCensus:
    """Return the current, deliberately conservative BUILDLEAD handoff."""
    return WiringCensus(
        gaps=(
            WiringGap(
                "C6-SWITCH-POLICY",
                "bd2_switchpolicy",
                "host-owned course cache, basis invalidation, and strict comparison hysteresis",
                WiringStatus.ADAPTER_READY,
                "kernel_switch_policy",
                "SwitchPolicy.evaluate",
                "kernel_decision_cache.DecisionCacheStore + kernel_tokenlab_stall_breaker.StallWireOwner",
                "BUILDLEAD must consume the decision at the shared owner boundary; this module performs no dispatch.",
            ),
            WiringGap(
                "C6-WIRE-PROFILE",
                "our_fault",
                "pairing seed, output reservation, wire cap, finish metadata, and diagnostic omission contract",
                WiringStatus.ADAPTER_READY,
                "kernel_switch_wire_contract",
                "prepare_wire_request",
                "campaigns.mr4.run_campaign._build",
                "Use the returned payload/receipt at the actual transport boundary; do not add a provider seed-label field.",
            ),
            WiringGap(
                "C6-RD3",
                "prior_art_map",
                "receipt-recovery reducer for repeat/refusal clearing",
                WiringStatus.DEFERRED_MISSING_OWNER,
                None,
                None,
                "existing settled-receipt/invalidation owners",
                "The historical src_rd3 implementation is not in src_v46; do not silently claim a port.",
            ),
            WiringGap(
                "C6-RD4",
                "prior_art_map",
                "bounded impact slice for observed cross-surface consumers",
                WiringStatus.DEFERRED_MISSING_OWNER,
                None,
                None,
                "current source-window, catalog, and check-receipt owners",
                "The historical src_rd4 implementation is not in src_v46; a future adapter needs current producers and a partial frontier.",
            ),
            WiringGap(
                "C6-Q8",
                "prior_art_map",
                "request-boundary and factual-carry activation",
                WiringStatus.DEFERRED_MISSING_OWNER,
                None,
                None,
                "current request/evidence contract owners",
                "kernel_navigation_receipt is a navigation receipt, not the absent Q8 request-boundary owner.",
            ),
            WiringGap(
                "C6-COMMON-SENSE",
                "prior_art_map",
                "common-sense rule catalogue qualification and fired/released proof",
                WiringStatus.QUALIFICATION_REQUIRED,
                "kernel_common_sense_governor",
                "decide",
                "kernel_completion_owner._common_sense_decision",
                "The owner is wired/on; BUILDLEAD must prove a real opportunity fires and reaches the owner output.",
            ),
            WiringGap(
                "C6-HISTORY-HATCH",
                "escape_hatch",
                "bounded history continuity hatch",
                WiringStatus.DEFERRED_DEFAULT_OFF,
                None,
                None,
                "deadlock screen + working notebook + durable journal owners",
                "Keep this proposal OFF until host-conflict negatives and notebook rendering are qualified.",
            ),
            WiringGap(
                "C6-FRAMEBUILD",
                "framesuccess",
                "single frame/final-wire authority and builder integration",
                WiringStatus.EXISTING_CONCURRENT_OWNER,
                "framebuild.builder",
                "FrameBuilder.build + prepare_request",
                "FRAMEBUILD WP0-WP5 owners",
                "Do not create a second builder or edit builder/projection/actions.py; BUILDLEAD integrates adapters there.",
            ),
        )
    )


def _records(outline: Mapping[str, object], field: str) -> tuple[Mapping[str, object], ...]:
    value = outline.get(field)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _line(record: Mapping[str, object]) -> int | None:
    value = record.get("line")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _candidate_records(outline: Mapping[str, object], claim: OutlineClaim) -> tuple[Mapping[str, object], ...]:
    if claim.kind == "module_statement":
        return _records(outline, "module_statements")
    return _records(outline, "functions") + _records(outline, "types")


def _claim_matches(record: Mapping[str, object], claim: OutlineClaim) -> bool:
    if claim.kind == "symbol":
        return record.get("name") == claim.name
    source = record.get("source")
    if isinstance(source, str) and claim.name in source:
        return True
    calls = record.get("calls")
    if isinstance(calls, Sequence) and not isinstance(calls, (str, bytes, bytearray)):
        return any(isinstance(item, Mapping) and claim.name in str(item.get("expression", "")) for item in calls)
    return False


def verify_outline_claims(snapshot: object, claims: Sequence[OutlineClaim]) -> OutlineClaimReport:
    """Verify owner claims through OUTLINEGENERIC's shared schema."""
    try:
        from kernel_evidence_contracts import outline_for_snapshot
    except ImportError:  # pragma: no cover - package loader path.
        from .kernel_evidence_contracts import outline_for_snapshot

    outline = outline_for_snapshot(snapshot)
    schema = str(outline.get("schema", ""))
    if schema != "mc-outline/1":
        raise ValueError("OUTLINEGENERIC returned an unsupported schema")
    path = str(outline.get("file", ""))
    language = str(outline.get("language", "unknown"))
    parser = str(outline.get("parser", "unknown"))
    degraded = bool(outline.get("degraded", False))
    results: list[OutlineClaimResult] = []
    for claim in claims:
        matches = [record for record in _candidate_records(outline, claim) if _claim_matches(record, claim)]
        if not matches:
            status = OutlineClaimStatus.UNKNOWN if degraded else OutlineClaimStatus.NOT_FOUND
            detail = "degraded outline cannot certify absence" if degraded else "shared outline has no matching record"
            results.append(OutlineClaimResult(claim, status, path, language, parser, degraded, detail=detail))
            continue
        lines = [line for line in (_line(record) for record in matches) if line is not None]
        line = min(lines) if lines else None
        if claim.line is not None and line != claim.line:
            results.append(OutlineClaimResult(
                claim, OutlineClaimStatus.MISMATCH, path, language, parser, degraded, line=line,
                detail=f"matching record is at line {line}, requested {claim.line}",
            ))
            continue
        results.append(OutlineClaimResult(
            claim, OutlineClaimStatus.FOUND, path, language, parser, degraded, line=line,
            detail="shared outline record matched",
        ))
    return OutlineClaimReport(schema, tuple(results), degraded)


__all__ = [
    "OutlineClaim", "OutlineClaimReport", "OutlineClaimResult", "OutlineClaimStatus",
    "SCHEMA", "WiringCensus", "WiringGap", "WiringStatus", "default_wiring_census",
    "verify_outline_claims",
]
