"""Typed, bodyless intermediate representation for deterministic fact compilation.

The compiler consumes an :class:`AllowedTaskView` and emits only bounded semantic
facts.  Source text is an input to analysis, never a field in the resulting IR.
The small IR is intentionally independent from the legacy family-specific
``SemanticEvidence`` payloads in ``kernel_semantics.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias


SCHEMA = "poc.contract_facts/1"

_FORBIDDEN_KEYS = frozenset(
    {
        "answer",
        "answer_key",
        "diff",
        "fixed",
        "fixed_source",
        "fixed_src",
        "gold",
        "gold_src",
        "patch",
        "solution",
    }
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
ParentSource: TypeAlias = object


class SemanticState(str, Enum):
    """Semantic truth of a fact, independent of whether it was observed live."""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


# ``FactState`` is the name used by callers that mirror the legacy bridge.
FactState = SemanticState


class Availability(str, Enum):
    """How the compiler obtained (or failed to obtain) a fact."""

    STATIC = "STATIC"
    OBSERVED = "OBSERVED"
    NOT_OBSERVED = "NOT_OBSERVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class Relation(str, Enum):
    DECISION = "DECISION"
    EFFECT = "EFFECT"


class ContractStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"


def _canonical_value(value: object) -> object:
    """Convert supported values to deterministic JSON-safe data."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_canonical_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True, default=str),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def canonical_json(value: object) -> str:
    """Serialize an IR value without object-address-dependent representations."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _clip_text(value: object, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _forbidden_key(key: object) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in _FORBIDDEN_KEYS or any(
        token in normalized
        for token in ("fixed_src", "gold_src", "answer_key", "patch_text")
    )


def _validate_metadata(value: object, location: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _forbidden_key(key):
                raise ValueError(f"{location} contains prohibited answer-key field")
            _validate_metadata(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, child in enumerate(value):
            _validate_metadata(child, f"{location}[{index}]")


def _mapping_copy(value: Mapping[str, object]) -> Mapping[str, object]:
    _validate_metadata(value)
    canonical = _canonical_value(value)
    if not isinstance(canonical, dict):
        return MappingProxyType({})
    return MappingProxyType(canonical)


@dataclass(frozen=True, order=True)
class SourceSpan:
    """A source location that explains one emitted fact."""

    path: str
    start_line: int
    end_line: int | None = None
    symbol: str = ""
    role: str = ""

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("source span path is required")
        if self.start_line < 1:
            raise ValueError("source span start_line must be positive")
        end_line = self.start_line if self.end_line is None else self.end_line
        if end_line < self.start_line:
            raise ValueError("source span end_line must not precede start_line")
        object.__setattr__(self, "end_line", end_line)

    @property
    def line(self) -> int:
        return self.start_line

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "role": self.role,
        }


@dataclass(frozen=True)
class LawRef:
    """Citation for the kernel law that makes a fact consequential."""

    law_id: str = "SNITCH_ARCHITECTURE"
    source_path: str = "kernel_status.md"
    start_line: int = 61
    end_line: int = 64
    statement: str = (
        "Consequential declarations carry a deterministic later-check; provable "
        "contradictions are actionable."
    )

    def __post_init__(self) -> None:
        if not self.law_id.strip():
            raise ValueError("law_id is required")
        if not self.source_path.strip():
            raise ValueError("law source path is required")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("invalid law source span")

    @property
    def law(self) -> str:
        return self.law_id

    @property
    def path(self) -> str:
        return self.source_path

    @property
    def span(self) -> SourceSpan:
        return SourceSpan(
            path=self.source_path,
            start_line=self.start_line,
            end_line=self.end_line,
            symbol=self.law_id,
            role="kernel_law",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "law_id": self.law_id,
            "path": self.source_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "statement": self.statement,
        }


@dataclass(frozen=True)
class WitnessVector:
    """A bounded, source-derived vector that makes a fact concrete."""

    dimensions: tuple[tuple[str, object], ...] = ()
    label: str = "source-derived"
    basis: str = "STATIC_SOURCE"
    source_spans: tuple[SourceSpan, ...] = ()

    def __post_init__(self) -> None:
        pairs = tuple((str(key), value) for key, value in self.dimensions)
        object.__setattr__(self, "dimensions", pairs)
        object.__setattr__(self, "source_spans", tuple(self.source_spans))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object],
        *,
        label: str = "source-derived",
        basis: str = "STATIC_SOURCE",
        source_spans: Sequence[SourceSpan] = (),
    ) -> "WitnessVector":
        return cls(
            dimensions=tuple(
                (str(key), values[key])
                for key in sorted(values, key=lambda item: str(item))
            ),
            label=label,
            basis=basis,
            source_spans=tuple(source_spans),
        )

    @property
    def values(self) -> Mapping[str, object]:
        return MappingProxyType(dict(self.dimensions))

    def to_dict(self) -> dict[str, object]:
        return {
            "dimensions": {key: _canonical_value(value) for key, value in self.dimensions},
            "label": self.label,
            "basis": self.basis,
            "source_spans": tuple(span.to_dict() for span in self.source_spans),
        }


@dataclass(frozen=True)
class FactCell:
    """One typed observation or contract comparison."""

    fact_id: str
    relation: Relation | str
    subject: str
    dimensions: tuple[str, ...]
    observed: object | None
    expected: object | None
    state: SemanticState | str
    availability: Availability | str
    basis: str
    witness_vector: WitnessVector | None
    source_spans: tuple[SourceSpan, ...]
    receipt_refs: tuple[str, ...] = ()
    law_refs: tuple[LawRef, ...] = ()
    unknown_reason: str | None = None
    unresolved_spans: tuple[SourceSpan, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        relation = (
            self.relation
            if isinstance(self.relation, Relation)
            else Relation(str(self.relation).upper())
        )
        state = (
            self.state
            if isinstance(self.state, SemanticState)
            else SemanticState(str(self.state).upper())
        )
        availability = (
            self.availability
            if isinstance(self.availability, Availability)
            else Availability(str(self.availability).upper())
        )
        if not self.fact_id.strip():
            raise ValueError("fact_id is required")
        if not self.subject.strip():
            raise ValueError("fact subject is required")
        source_spans = tuple(self.source_spans)
        law_refs = tuple(self.law_refs)
        unresolved_spans = tuple(self.unresolved_spans)
        if not source_spans:
            raise ValueError(f"fact {self.fact_id} has no parent source span")
        if not law_refs:
            raise ValueError(f"fact {self.fact_id} has no law reference")
        if state is SemanticState.UNKNOWN:
            if not _clip_text(self.unknown_reason):
                raise ValueError(f"UNKNOWN fact {self.fact_id} needs a reason")
            if not unresolved_spans:
                raise ValueError(f"UNKNOWN fact {self.fact_id} needs an unresolved span")
        elif self.unknown_reason or unresolved_spans:
            raise ValueError(f"non-UNKNOWN fact {self.fact_id} has UNKNOWN metadata")
        if availability is Availability.NOT_OBSERVED and state is not SemanticState.UNKNOWN:
            raise ValueError("NOT_OBSERVED facts must have semantic state UNKNOWN")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "availability", availability)
        object.__setattr__(self, "dimensions", tuple(str(item) for item in self.dimensions))
        object.__setattr__(self, "source_spans", source_spans)
        object.__setattr__(self, "law_refs", law_refs)
        object.__setattr__(self, "receipt_refs", tuple(str(item) for item in self.receipt_refs))
        object.__setattr__(self, "unresolved_spans", unresolved_spans)
        object.__setattr__(self, "notes", tuple(_clip_text(item) for item in self.notes if item))
        if self.unknown_reason:
            object.__setattr__(self, "unknown_reason", _clip_text(self.unknown_reason))

    @property
    def is_contradiction(self) -> bool:
        return self.state is SemanticState.FALSE

    @property
    def provenance(self) -> tuple[SourceSpan, ...]:
        return self.source_spans

    @property
    def span(self) -> SourceSpan:
        return self.source_spans[0]

    @property
    def unresolved_span(self) -> SourceSpan | None:
        return self.unresolved_spans[0] if self.unresolved_spans else None

    @property
    def law_ref(self) -> LawRef:
        return self.law_refs[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "relation": self.relation.value,
            "subject": self.subject,
            "dimensions": self.dimensions,
            "observed": _canonical_value(self.observed),
            "expected": _canonical_value(self.expected),
            "state": self.state.value,
            "availability": self.availability.value,
            "basis": self.basis,
            "witness_vector": (
                self.witness_vector.to_dict() if self.witness_vector is not None else None
            ),
            "source_spans": tuple(span.to_dict() for span in self.source_spans),
            "receipt_refs": self.receipt_refs,
            "law_refs": tuple(ref.to_dict() for ref in self.law_refs),
            "unknown_reason": self.unknown_reason,
            "unresolved_spans": tuple(span.to_dict() for span in self.unresolved_spans),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ContractFacts:
    """Content-addressed collection of generic, typed contract facts."""

    cells: tuple[FactCell, ...]
    witness_vectors: tuple[WitnessVector, ...]
    coverage: Mapping[str, object]
    limitations: tuple[str, ...]
    analyzer_version: str
    input_digest: str
    task_id: str = ""
    subject: str = ""
    source_digest: str = ""
    evidence_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "witness_vectors", tuple(self.witness_vectors))
        object.__setattr__(self, "coverage", _mapping_copy(self.coverage))
        object.__setattr__(
            self,
            "limitations",
            tuple(_clip_text(item) for item in self.limitations if _clip_text(item)),
        )
        if not self.analyzer_version.strip():
            raise ValueError("analyzer_version is required")
        if not self.input_digest.strip():
            raise ValueError("input_digest is required")
        if not self.evidence_id:
            core = {
                "schema": SCHEMA,
                "cells": tuple(cell.to_dict() for cell in self.cells),
                "witness_vectors": tuple(
                    vector.to_dict() for vector in self.witness_vectors
                ),
                "coverage": self.coverage,
                "limitations": self.limitations,
                "analyzer_version": self.analyzer_version,
                "input_digest": self.input_digest,
                "task_id": self.task_id,
                "subject": self.subject,
                "source_digest": self.source_digest,
            }
            object.__setattr__(self, "evidence_id", "cf:" + _digest(core)[:24])

    @property
    def status(self) -> ContractStatus:
        if any(cell.state is SemanticState.FALSE for cell in self.cells):
            return ContractStatus.CONTRADICTED
        if any(cell.state is SemanticState.UNKNOWN for cell in self.cells):
            return ContractStatus.UNKNOWN
        return ContractStatus.SUPPORTED

    @property
    def contradictions(self) -> tuple[FactCell, ...]:
        return tuple(cell for cell in self.cells if cell.is_contradiction)

    @property
    def unknowns(self) -> tuple[FactCell, ...]:
        return tuple(cell for cell in self.cells if cell.state is SemanticState.UNKNOWN)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "evidence_id": self.evidence_id,
            "task_id": self.task_id,
            "subject": self.subject,
            "source_digest": self.source_digest,
            "cells": tuple(cell.to_dict() for cell in self.cells),
            "witness_vectors": tuple(
                vector.to_dict() for vector in self.witness_vectors
            ),
            "coverage": _canonical_value(self.coverage),
            "limitations": self.limitations,
            "analyzer_version": self.analyzer_version,
            "input_digest": self.input_digest,
            "status": self.status.value,
        }

    def serialize(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class AllowedTaskView:
    """The only task input accepted by the generic compiler.

    ``parent_src`` may be one source string, a path-to-source mapping, or the
    metadata-shaped ``{"files": {path: source}}`` value understood by the
    existing source context.  ``fixed_src``/diff/gold fields are deliberately
    absent from this type and are ignored by :meth:`from_result_row`.
    """

    task_id: str
    target_file: str
    target_symbol: str
    parent_src: ParentSource
    symptom: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)
    traceback_anchor: SourceSpan | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id is required")
        if not self.target_file.strip():
            raise ValueError("target_file is required")
        if not self.target_symbol.strip():
            raise ValueError("target_symbol is required")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _mapping_copy(self.metadata))
        object.__setattr__(self, "symptom", _clip_text(self.symptom, 1000))

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "AllowedTaskView":
        """Build an allowlisted view without reading answer-key values."""

        parent_src = row.get("parent_src")
        if parent_src is None:
            raise ValueError("parent_src is required")
        task_id = _first_text(row, "task_id", "id")
        path = _first_text(row, "target_file", "path", "file", "rel")
        symbol = _first_text(row, "target_symbol", "fn", "function", "name")
        repo = _first_text(row, "repo")
        if not task_id:
            task_id = ":".join(part for part in (repo, path, symbol) if part) or "task"
        metadata: dict[str, object] = {}
        supplied_metadata = row.get("metadata")
        if isinstance(supplied_metadata, Mapping):
            metadata.update(supplied_metadata)
        for key in (
            "family",
            "shape",
            "tier",
            "repo",
            "path",
            "rel",
            "msg_source",
            "symptom_source",
            "strict_tier",
            "expressible_now",
        ):
            if key in row:
                metadata[key] = row[key]
        symptom = _first_text(row, "symptom", "msg")
        anchor = row.get("traceback_anchor")
        return cls(
            task_id=task_id,
            target_file=path or "source.py",
            target_symbol=symbol or "unknown",
            parent_src=parent_src,
            symptom=symptom,
            metadata=metadata,
            traceback_anchor=anchor if isinstance(anchor, SourceSpan) else None,
        )

    from_result_row = from_mapping

    @property
    def parent_digest(self) -> str:
        return _digest(_source_identity(self.parent_src, self.target_file))

    def to_dict(self) -> dict[str, object]:
        """Return safe metadata and identity only; never the parent body."""

        return {
            "task_id": self.task_id,
            "target_file": self.target_file,
            "target_symbol": self.target_symbol,
            "parent_digest": self.parent_digest,
            "symptom": self.symptom,
            "metadata": _canonical_value(self.metadata),
            "traceback_anchor": (
                self.traceback_anchor.to_dict() if self.traceback_anchor else None
            ),
        }


def _first_text(row: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_identity(source: object, target_file: str) -> object:
    if isinstance(source, str):
        return {target_file or "source.py": source}
    if isinstance(source, Mapping):
        files = source.get("files")
        if isinstance(files, Mapping):
            return {
                "files": {
                    str(path): content
                    for path, content in files.items()
                    if isinstance(content, str)
                },
                "generation": source.get("generation", 0),
                "workspace_revision": source.get("workspace_revision", 0),
                "truncated": source.get("truncated", False),
            }
        if all(isinstance(key, str) and isinstance(value, str) for key, value in source.items()):
            return dict(source)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return tuple(_canonical_value(item) for item in source)
    return {"type": type(source).__name__}


__all__ = [
    "AllowedTaskView",
    "Availability",
    "ContractFacts",
    "ContractStatus",
    "FactCell",
    "FactState",
    "JsonValue",
    "LawRef",
    "Relation",
    "SCHEMA",
    "SemanticState",
    "SourceSpan",
    "WitnessVector",
    "canonical_json",
]
