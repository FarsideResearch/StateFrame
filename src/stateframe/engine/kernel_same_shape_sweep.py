"""Opt-in same-shape sibling sweep: model hypothesis, host query, obligations.

The signature is a guidance scaffold because a model proposes the violated
contract.  Once accepted, compiling and running the query are host-owned
truth operations.  This module never edits source or treats a scan as proof
that no other behavioral instance exists.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from code_same_shape_sweep import (
    CompiledPythonQuery,
    PythonPatternValidation,
    PythonSite,
    compile_python_query,
    scan_python_source,
    scan_python_workspace,
    validate_python_query,
)


MODE_ENV = "STATEFRAME_SAME_SHAPE_SWEEP"
SCHEMA = "same-shape-sweep/1"
DISPOSITIONS = ("applicable", "safe", "covered", "out_of_scope", "unknown")
LEVELS = ("suggestion", "review", "repair")
MAX_CANDIDATES = 256
MAX_PACKET_SITES = 5
MAX_CONTRACT = 400


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _bounded(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _strings(value: object, *, limit: int = 12, item_limit: int = 120) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    rows: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in rows:
            rows.append(text[:item_limit])
    return tuple(rows[:limit])


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    raise TypeError("same-shape signature must be a mapping")


def _field(raw: Mapping[str, object], *names: str, default: object = None) -> object:
    for name in names:
        if raw.get(name) is not None:
            return raw[name]
    return default


@dataclass(frozen=True, slots=True)
class DefectSignature:
    """Typed model proposal naming the violated contract, not just syntax."""

    contract: str
    operation: str
    attribute: str | None = None
    callee: str | None = None
    receiver: str | None = None
    query: str | None = None
    roles: tuple[str, ...] = ()
    safe_lookalikes: tuple[str, ...] = ()
    level: str = "review"
    anchor_path: str | None = None
    anchor_line: int | None = None
    anchor_column: int | None = None
    anchor_qualname: str | None = None
    anchor_text: str | None = None
    generalizations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.contract, str) or not self.contract.strip():
            raise ValueError("same-shape signature requires a non-empty contract")
        if len(self.contract) > MAX_CONTRACT:
            raise ValueError("same-shape signature contract is too long")
        if self.level not in LEVELS:
            raise ValueError(f"same-shape level must be one of {LEVELS}")
        if len(self.generalizations) > 2:
            raise ValueError("same-shape sweep permits at most two generalizations")
        for name in ("anchor_line", "anchor_column"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | DefectSignature) -> DefectSignature:
        if isinstance(value, cls):
            return value
        raw = _mapping(value)
        contract = _field(raw, "contract", "violated_contract")
        operation = _field(raw, "operation", "defect_kind", "shape", "node_type")
        if not isinstance(contract, str) or not contract.strip():
            raise ValueError("same-shape signature requires a non-empty contract")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("same-shape signature requires an operation")
        anchor = raw.get("anchor")
        anchor = anchor if isinstance(anchor, Mapping) else {}
        line = _field(anchor, "line", default=_field(raw, "anchor_line"))
        column = _field(anchor, "column", default=_field(raw, "anchor_column"))
        generalizations = _strings(raw.get("generalizations", ()), limit=3)
        return cls(
            contract=_bounded(contract, MAX_CONTRACT), operation=operation.strip(),
            attribute=_field(raw, "attribute", "field", "member"),
            callee=_field(raw, "callee", "function", "call"),
            receiver=_field(raw, "receiver", "receiver_name", "object"),
            query=_field(raw, "query", "literal", "grep"),
            roles=_strings(raw.get("roles", ())),
            safe_lookalikes=_strings(_field(raw, "safe_lookalikes", "safe_receivers",
                                             default=())),
            level=str(raw.get("level", "review")).strip().lower(),
            anchor_path=_field(anchor, "path", default=_field(raw, "anchor_path")),
            anchor_line=line,
            anchor_column=column,
            anchor_qualname=_field(anchor, "qualname", default=_field(raw, "anchor_qualname")),
            anchor_text=_field(anchor, "text", default=_field(raw, "anchor_text")),
            generalizations=generalizations,
        )

    @property
    def anchor(self) -> dict[str, object] | None:
        values = {
            "path": self.anchor_path, "line": self.anchor_line,
            "column": self.anchor_column, "qualname": self.anchor_qualname,
            "text": self.anchor_text,
        }
        values = {key: value for key, value in values.items() if value is not None}
        return values or None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA + "/signature",
            "contract": self.contract,
            "operation": self.operation,
            "attribute": self.attribute,
            "callee": self.callee,
            "receiver": self.receiver,
            "query": self.query,
            "roles": list(self.roles),
            "safe_lookalikes": list(self.safe_lookalikes),
            "level": self.level,
            "anchor": self.anchor,
            "generalizations": list(self.generalizations),
        }


@dataclass(frozen=True, slots=True)
class SimilarSiteObligation:
    id: str
    kind: str
    level: str
    site: PythonSite
    contract: str
    dispositions: tuple[str, ...]
    required_evidence: tuple[str, ...]
    status: str = "open"

    def __post_init__(self) -> None:
        if self.kind != "SIMILAR_SITE":
            raise ValueError("same-shape obligations must be SIMILAR_SITE")
        if self.level not in LEVELS:
            raise ValueError("unknown same-shape obligation level")
        if self.status != "open":
            raise ValueError("new same-shape obligations must be open")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "level": self.level,
            "site": self.site.to_dict(),
            "contract": self.contract,
            "dispositions": list(self.dispositions),
            "required_evidence": list(self.required_evidence),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class SweepResult:
    status: str
    fired: bool
    level: str
    signature: DefectSignature | None
    query: CompiledPythonQuery | None
    validation: PythonPatternValidation | None
    survivors: tuple[PythonSite, ...]
    top5: tuple[PythonSite, ...]
    obligations: tuple[SimilarSiteObligation, ...]
    review_packet: str
    scope_complete: bool = False
    error: str | None = None

    @classmethod
    def off(cls) -> SweepResult:
        return cls("OFF", False, "review", None, None, None, (), (), (), "", False)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "fired": self.fired,
            "level": self.level,
            "signature": self.signature.to_dict() if self.signature else None,
            "query": self.query.to_dict() if self.query else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "survivors": [site.to_dict() for site in self.survivors],
            "top5": [site.to_dict() for site in self.top5],
            "obligations": [item.to_dict() for item in self.obligations],
            "review_packet": self.review_packet,
            "scope_complete": self.scope_complete,
            "error": self.error,
        }


def _emit(journal: object, event: str, **fields: object) -> None:
    emit = getattr(journal, "emit", None)
    if callable(emit):
        # Keep journal-owned fields out of every event.  ``payload_schema`` is
        # intentionally used instead of the reserved ``schema`` key.
        emit(event, payload_schema=SCHEMA, **fields)


def _anchor(signature: DefectSignature, path: str) -> dict[str, object] | None:
    value = signature.anchor
    if value is None:
        return None
    result = dict(value)
    result.setdefault("path", path)
    return result


def _obligation_id(query: CompiledPythonQuery, site: PythonSite) -> str:
    raw = f"{query.query_digest}:{site.behavior_id}"
    return "similar-" + raw.split(":", 1)[-1][:24]


def _required_evidence(level: str) -> tuple[str, ...]:
    if level == "suggestion":
        return ()
    if level == "repair":
        return ("candidate_counterexample", "current_candidate_scan")
    return ("disposition", "current_candidate_scan")


def _review_packet(signature: DefectSignature, validation: PythonPatternValidation,
                   top5: Sequence[PythonSite], level: str) -> str:
    payload = {
        "authority": "HOST_QUERY_WITH_MODEL_SIGNATURE",
        "contract": signature.contract,
        "level": level,
        "pattern_validation": validation.to_dict(),
        "top5": [site.to_dict() for site in top5],
        "dispositions": list(DISPOSITIONS),
        "instruction": (
            "Review each behavioral site. A completed scan means only that this "
            "scope was searched; it does not mean no other instance exists."
        ),
    }
    return "[SIMILAR_SITE REVIEW PACKET]\n" + json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compile_signature(signature: Mapping[str, object] | DefectSignature) -> CompiledPythonQuery:
    return compile_python_query(DefectSignature.from_mapping(signature))


def validate_pattern(
    query: CompiledPythonQuery,
    base_source: str,
    candidate_source: str,
    *,
    path: str,
    signature: DefectSignature | None = None,
    safe_lookalike_sources: Iterable[str] = (),
) -> PythonPatternValidation:
    return validate_python_query(
        query, base_source, candidate_source, path=path,
        anchor=signature.anchor if signature is not None else None,
        safe_lookalike_sources=safe_lookalike_sources,
    )


def signature_request(*, path: str, base_source: str, candidate_source: str,
                      first_green_receipt: Mapping[str, object] | None = None) -> dict[str, object]:
    """Build the bounded model input for the scaffold half of the mechanism."""
    import difflib

    diff = "".join(difflib.unified_diff(
        base_source.splitlines(keepends=True), candidate_source.splitlines(keepends=True),
        fromfile=path + "@base", tofile=path + "@candidate", n=2))
    return {
        "schema": SCHEMA + "/signature-request",
        "path": path,
        "contract_prompt": (
            "Name the violated behavioral contract repaired by this diff. "
            "Return a typed operation and roles; do not describe only syntax."
        ),
        "diff": diff[:4000],
        "base_source": base_source[:12000],
        "candidate_source": candidate_source[:12000],
        "first_green_receipt": dict(first_green_receipt or {}),
    }


def run_sweep(
    *,
    signature: Mapping[str, object] | DefectSignature,
    base_source: str,
    candidate_source: str,
    path: str,
    root: str | Path | None = None,
    journal: object = None,
    safe_lookalike_sources: Iterable[str] = (),
    maximum: int = MAX_CANDIDATES,
    top_k: int = MAX_PACKET_SITES,
) -> SweepResult:
    """Validate, scan the current candidate, and seed typed sibling obligations."""
    if mode() != "on":
        return SweepResult.off()
    try:
        typed = DefectSignature.from_mapping(signature)
        query = compile_signature(typed)
    except (TypeError, ValueError) as error:
        _emit(journal, "same_shape_sweep_fired", producer="kernel_same_shape_sweep",
              sweep_status="SIGNATURE_REJECTED", dispatch=False)
        _emit(journal, "same_shape_sweep_signature_rejected",
              producer="kernel_same_shape_sweep", reason=str(error)[:300], dispatch=False)
        return SweepResult("SIGNATURE_REJECTED", True, "review", None, None, None,
                           (), (), (), "", False, str(error))
    _emit(journal, "same_shape_sweep_fired", producer="kernel_same_shape_sweep",
          sweep_status="STARTED", operation=query.operation,
          level=typed.level, query_digest=query.query_digest, dispatch=False)
    _emit(journal, "same_shape_sweep_query_compiled", producer="code_same_shape_sweep",
          operation=query.operation, query_digest=query.query_digest,
          query_text=query.query_text, contract=query.contract, dispatch=False)
    validation = validate_pattern(
        query, base_source, candidate_source, path=path, signature=typed,
        safe_lookalike_sources=safe_lookalike_sources,
    )
    _emit(journal, "same_shape_sweep_pattern_validated", producer="code_same_shape_sweep",
          query_digest=query.query_digest, accepted=validation.accepted,
          base_matches=validation.base_matches,
          candidate_matches=validation.candidate_matches,
          repaired_matches=validation.repaired_matches,
          safe_lookalike_matches=validation.safe_lookalike_matches,
          reason=validation.reason, dispatch=False)
    if not validation.accepted:
        return SweepResult("PATTERN_REJECTED", True, typed.level, typed, query,
                           validation, (), (), (), "", False, validation.reason)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("same-shape maximum must be a positive integer")
    if type(top_k) is not int or not 1 <= top_k <= MAX_PACKET_SITES:
        raise ValueError("same-shape top_k must be between 1 and 5")
    overrides = {path: candidate_source}
    if root is None:
        survivors = scan_python_source(candidate_source, path=path, query=query)
    else:
        survivors = scan_python_workspace(
            root, query, overrides=overrides, include_paths=(path,), maximum=maximum)
    # The repaired file is part of the scan, but the validation above proves
    # its anchor is gone.  Every remaining hit is a behavioral sibling site.
    survivors = tuple(sorted(survivors, key=lambda site: (-site.score, site.path, site.line,
                                                            site.column)))[:maximum]
    top5 = survivors[:top_k]
    required = _required_evidence(typed.level)
    obligations: list[SimilarSiteObligation] = []
    for site in survivors:
        obligation = SimilarSiteObligation(
            id=_obligation_id(query, site), kind="SIMILAR_SITE", level=typed.level,
            site=site, contract=typed.contract, dispositions=DISPOSITIONS,
            required_evidence=required,
        )
        obligations.append(obligation)
        _emit(journal, "same_shape_sweep_obligation_seeded",
              producer="kernel_same_shape_sweep", obligation_id=obligation.id,
              site_id=site.behavior_id, site_path=site.path, site_line=site.line,
              level=typed.level, required_evidence=list(required), dispatch=False)
    packet = _review_packet(typed, validation, top5, typed.level)
    _emit(journal, "same_shape_sweep_scan_completed", producer="kernel_same_shape_sweep",
          query_digest=query.query_digest, survivor_count=len(survivors),
          top_count=len(top5), obligation_count=len(obligations),
          scope_complete=False, dispatch=False)
    return SweepResult("SCANNED", True, typed.level, typed, query, validation,
                       survivors, top5, tuple(obligations), packet, False)


def review_precision(
    obligations: Sequence[SimilarSiteObligation],
    dispositions: Mapping[str, object],
) -> float | None:
    """Measure reviewed precision; unknown/unreviewed sites stay out of the denominator."""
    reviewed = 0
    applicable = 0
    for obligation in obligations:
        raw = dispositions.get(obligation.id)
        choice = raw.get("disposition") if isinstance(raw, Mapping) else raw
        choice = str(choice or "").strip().lower()
        if choice not in DISPOSITIONS or choice == "unknown":
            continue
        reviewed += 1
        applicable += choice == "applicable"
    return (applicable / reviewed) if reviewed else None


__all__ = [
    "DISPOSITIONS", "LEVELS", "MODE_ENV", "SCHEMA", "DefectSignature",
    "SimilarSiteObligation", "SweepResult", "compile_signature", "enabled",
    "mode", "review_precision", "run_sweep", "signature_request",
    "validate_pattern",
]
