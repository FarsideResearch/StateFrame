"""Data workspace evidence packets: the default-off switch and the typed packet contract.

Stage 1 defines the contract only. Nothing calls it yet, so with the switch off the engine behaves
exactly as before.

A packet is one query result with its qualifiers. Qualifiers (what was asked, how much was covered,
how numbers were produced, what was selected, where it came from, what the reader sees) are never
dropped when a packet is fitted to a byte budget; only tail rows are.

``ok`` means the query ran. ``empty`` means it ran and the source had no data, so it carries no rows.
An ``ok`` packet with zero rows is valid: the query ran and matched nothing.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import kernel_scope_receipt as scope_receipt

SWITCH_ENV = "STATEFRAME_DATA_WORKSPACE"
SWITCH_VALUES = ("off", "on")
RECEIPT_SCHEMA = "stateframe-data-packet/1"
ROWS_TRUNCATED_KEY = "rows_truncated"
ROWS_SHOWN_KEY = "rows_shown"

Certainty: TypeAlias = Literal["exact", "lower_bound", "estimate", "unavailable"]
Status: TypeAlias = Literal["ok", "empty", "failed"]
QualifierValue: TypeAlias = str | int | bool | None
CellValue: TypeAlias = str | int | None

_CERTAINTIES: frozenset[str] = frozenset({"exact", "lower_bound", "estimate", "unavailable"})
_STATUSES: frozenset[str] = frozenset({"ok", "empty", "failed"})
_QUALIFIER_FIELDS = ("input", "coverage", "numeric", "selection", "storage", "display")


class PacketInvariantError(ValueError):
    """A packet, count or receipt broke the contract. ``code`` is stable; callers branch on it."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def data_workspace_enabled(environ: Mapping[str, str]) -> bool:
    """``STATEFRAME_DATA_WORKSPACE`` off (default) | on; anything else fails closed."""
    mode = str(environ.get(SWITCH_ENV, "off")).strip().lower()
    if mode not in SWITCH_VALUES:
        raise ValueError(f"{SWITCH_ENV} must be one of: {', '.join(SWITCH_VALUES)}")
    return mode == "on"


def _is_int(value: object) -> bool:
    return type(value) is int


def _scalar_ok(value: object) -> bool:
    return value is None or type(value) in (str, int, bool)


def _copy_mapping(name: str, value: Mapping[str, object], allowed: tuple[type, ...] | None) -> dict[str, object]:
    """Plain-dict copy with string keys; ``allowed`` restricts value types (None = any scalar)."""
    copied: dict[str, object] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise PacketInvariantError("mapping_key_invalid", f"{name} keys must be non-empty strings")
        if allowed is not None and type(item) not in allowed:
            raise PacketInvariantError("mapping_value_invalid", f"{name}.{key} has an unsupported type")
        if allowed is None and not _scalar_ok(item):
            raise PacketInvariantError("mapping_value_invalid", f"{name}.{key} has an unsupported type")
        copied[key] = item
    return copied


@dataclass(frozen=True)
class TypedCount:
    """A count with an explicit certainty. ``value`` is None exactly when certainty is ``unavailable``."""

    value: int | None
    certainty: Certainty
    basis: str

    def __post_init__(self) -> None:
        if self.certainty not in _CERTAINTIES:
            raise PacketInvariantError("typed_count_invalid", f"unknown certainty {self.certainty!r}")
        if type(self.basis) is not str or not self.basis:
            raise PacketInvariantError("typed_count_invalid", "count basis must be a non-empty string")
        if self.certainty == "unavailable":
            if self.value is not None:
                raise PacketInvariantError("typed_count_invalid", "unavailable count must have no value")
            return
        if not _is_int(self.value) or self.value < 0:  # type: ignore[operator]
            raise PacketInvariantError("typed_count_invalid", "count value must be a non-negative int")

    def to_dict(self) -> dict[str, object]:
        return {"value": self.value, "certainty": self.certainty, "basis": self.basis}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TypedCount:
        if set(data) != {"value", "certainty", "basis"}:
            raise PacketInvariantError("typed_count_invalid", "count needs exactly value, certainty, basis")
        value = data["value"]
        if value is not None and not _is_int(value):
            raise PacketInvariantError("typed_count_invalid", "count value must be an int or None")
        certainty = data["certainty"]
        if type(certainty) is not str or certainty not in _CERTAINTIES:
            raise PacketInvariantError("typed_count_invalid", "count certainty is not recognised")
        basis = data["basis"]
        if type(basis) is not str:
            raise PacketInvariantError("typed_count_invalid", "count basis must be a string")
        return cls(value=value, certainty=certainty, basis=basis)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Qualifiers:
    """What the packet is about. Each group is a flat mapping of str keys to scalar values."""

    input: Mapping[str, QualifierValue] = field(default_factory=dict)
    coverage: Mapping[str, QualifierValue] = field(default_factory=dict)
    numeric: Mapping[str, QualifierValue] = field(default_factory=dict)
    selection: Mapping[str, QualifierValue] = field(default_factory=dict)
    storage: Mapping[str, QualifierValue] = field(default_factory=dict)
    display: Mapping[str, QualifierValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for group in _QUALIFIER_FIELDS:
            copied = _copy_mapping(group, getattr(self, group), None)
            object.__setattr__(self, group, copied)
        complete = self.coverage.get("complete")
        if "complete" in self.coverage and type(complete) is not bool:
            raise PacketInvariantError("qualifier_invalid", "coverage.complete must be a bool")

    def to_dict(self) -> dict[str, object]:
        return {group: dict(getattr(self, group)) for group in _QUALIFIER_FIELDS}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Qualifiers:
        if set(data) != set(_QUALIFIER_FIELDS):
            raise PacketInvariantError("qualifier_invalid", "qualifiers need exactly the six groups")
        groups: dict[str, Mapping[str, QualifierValue]] = {}
        for group in _QUALIFIER_FIELDS:
            part = data[group]
            if not isinstance(part, Mapping):
                raise PacketInvariantError("qualifier_invalid", f"qualifiers.{group} must be a mapping")
            groups[group] = _copy_mapping(group, part, None)  # type: ignore[assignment]
        return cls(**groups)  # type: ignore[arg-type]


@dataclass(frozen=True)
class EvidencePacket:
    """One query result. Call ``validate()`` before trusting, serialising or budget-fitting it."""

    status: Status
    revision: str
    qualifiers: Qualifiers
    counts: Mapping[str, TypedCount]
    rows: tuple[Mapping[str, CellValue], ...]
    aggregates: Mapping[str, TypedCount]
    error: str | None

    def validate(self) -> None:
        """Raise ``PacketInvariantError`` with a stable code when the packet breaks the contract."""
        if self.status not in _STATUSES:
            raise PacketInvariantError("status_invalid", f"unknown status {self.status!r}")
        if type(self.revision) is not str or not self.revision:
            raise PacketInvariantError("revision_missing", "revision must be a non-empty string")
        for name, count in (*self.counts.items(), *self.aggregates.items()):
            if type(name) is not str or not name:
                raise PacketInvariantError("typed_count_invalid", "count names must be non-empty strings")
            if not isinstance(count, TypedCount):
                raise PacketInvariantError("typed_count_invalid", f"{name} is not a TypedCount")
            TypedCount(value=count.value, certainty=count.certainty, basis=count.basis)  # re-check (e)
        for row in self.rows:
            _copy_mapping("row", row, (str, int, type(None)))
        if self.status == "failed":
            if not self.error:
                raise PacketInvariantError("failed_without_error", "failed packets must carry an error")
            if self.rows or self.aggregates:
                raise PacketInvariantError("failed_has_result", "failed packets carry no rows or aggregates")
            return
        if self.error is not None:
            raise PacketInvariantError("success_with_error", "ok and empty packets must not carry an error")
        if self.status == "empty" and self.rows:
            raise PacketInvariantError("empty_with_rows", "empty packets carry no rows")
        complete = self.qualifiers.coverage.get("complete") is True
        if not complete and any(count.certainty == "exact" for count in self.aggregates.values()):
            raise PacketInvariantError(
                "exact_over_incomplete_coverage",
                "aggregates over incomplete coverage cannot be exact",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "revision": self.revision,
            "qualifiers": self.qualifiers.to_dict(),
            "counts": {name: count.to_dict() for name, count in self.counts.items()},
            "rows": [dict(row) for row in self.rows],
            "aggregates": {name: count.to_dict() for name, count in self.aggregates.items()},
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EvidencePacket:
        expected = {"status", "revision", "qualifiers", "counts", "rows", "aggregates", "error"}
        if set(data) != expected:
            raise PacketInvariantError("packet_fields_invalid", "packet fields do not match the contract")
        qualifiers_raw = data["qualifiers"]
        counts_raw = data["counts"]
        aggregates_raw = data["aggregates"]
        rows_raw = data["rows"]
        if not (isinstance(qualifiers_raw, Mapping) and isinstance(counts_raw, Mapping)
                and isinstance(aggregates_raw, Mapping) and isinstance(rows_raw, list)):
            raise PacketInvariantError("packet_fields_invalid", "packet sections have the wrong shape")
        status = data["status"]
        revision = data["revision"]
        error = data["error"]
        if type(status) is not str or status not in _STATUSES:
            raise PacketInvariantError("status_invalid", "packet status is not recognised")
        if type(revision) is not str:
            raise PacketInvariantError("revision_missing", "revision must be a string")
        if error is not None and type(error) is not str:
            raise PacketInvariantError("packet_fields_invalid", "error must be a string or null")
        rows: list[Mapping[str, CellValue]] = []
        for row in rows_raw:
            if not isinstance(row, Mapping):
                raise PacketInvariantError("packet_fields_invalid", "each row must be a mapping")
            rows.append(_copy_mapping("row", row, (str, int, type(None))))  # type: ignore[arg-type]
        packet = cls(
            status=status,  # type: ignore[arg-type]
            revision=revision,
            qualifiers=Qualifiers.from_dict(qualifiers_raw),
            counts={str(name): TypedCount.from_dict(item) for name, item in counts_raw.items()},  # type: ignore[union-attr]
            rows=tuple(rows),
            aggregates={str(name): TypedCount.from_dict(item) for name, item in aggregates_raw.items()},  # type: ignore[union-attr]
            error=error,
        )
        packet.validate()
        return packet


def satisfies_binding(packet: EvidencePacket, bound_revision: str) -> bool:
    """True only when the packet's revision string is exactly the bound one (r7 never satisfies r8)."""
    return packet.revision == bound_revision


def packet_bytes(packet: EvidencePacket) -> bytes:
    """Canonical JSON for a validated packet; the same packet always yields the same bytes."""
    packet.validate()
    return scope_receipt.canonical_json_bytes(packet.to_dict())


def receipt(packet: EvidencePacket) -> dict[str, object]:
    """A self-verifying envelope: the packet plus the SHA-256 of its canonical bytes."""
    return {
        "schema": RECEIPT_SCHEMA,
        "sha256": hashlib.sha256(packet_bytes(packet)).hexdigest(),
        "packet": packet.to_dict(),
    }


def from_receipt(data: Mapping[str, object]) -> EvidencePacket:
    """Rebuild and verify a receipt. Schema, packet validity and digest must all hold."""
    if data.get("schema") != RECEIPT_SCHEMA:
        raise PacketInvariantError("receipt_schema_invalid", "receipt schema is not stateframe-data-packet/1")
    packet_raw = data.get("packet")
    if not isinstance(packet_raw, Mapping):
        raise PacketInvariantError("receipt_schema_invalid", "receipt packet must be a mapping")
    packet = EvidencePacket.from_dict(packet_raw)
    digest = hashlib.sha256(packet_bytes(packet)).hexdigest()
    if data.get("sha256") != digest:
        raise PacketInvariantError("receipt_digest_mismatch", "receipt sha256 does not match its packet")
    return packet


def _with_truncation(packet: EvidencePacket, rows_shown: int) -> EvidencePacket:
    display = {**packet.qualifiers.display, ROWS_TRUNCATED_KEY: True, ROWS_SHOWN_KEY: rows_shown}
    qualifiers = Qualifiers(
        input=packet.qualifiers.input,
        coverage=packet.qualifiers.coverage,
        numeric=packet.qualifiers.numeric,
        selection=packet.qualifiers.selection,
        storage=packet.qualifiers.storage,
        display=display,
    )
    return EvidencePacket(
        status=packet.status,
        revision=packet.revision,
        qualifiers=qualifiers,
        counts=packet.counts,
        rows=packet.rows[:rows_shown],
        aggregates=packet.aggregates,
        error=packet.error,
    )


def fit_rows(packet: EvidencePacket, max_bytes: int) -> EvidencePacket:
    """Drop tail rows until the canonical packet fits ``max_bytes``.

    Qualifiers, counts, aggregates and error are never dropped. When rows are dropped, the display
    group records ``rows_truncated`` and ``rows_shown``; those keys are present while sizing, so the
    returned packet fits its own header. Size grows monotonically with the row count, so the largest
    fitting prefix is found by binary search.
    """
    packet.validate()
    if type(max_bytes) is not int or max_bytes < 0:
        raise PacketInvariantError("budget_invalid", "max_bytes must be a non-negative int")
    if len(packet_bytes(packet)) <= max_bytes:
        return packet
    low, high = 0, len(packet.rows) - 1  # the full row set is already known not to fit
    best: int | None = None
    while low <= high:
        mid = (low + high) // 2
        if len(packet_bytes(_with_truncation(packet, mid))) <= max_bytes:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best is None:
        raise PacketInvariantError("qualifiers_exceed_budget", "qualifiers alone exceed the byte budget")
    return _with_truncation(packet, best)
