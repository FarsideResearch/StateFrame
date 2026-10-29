"""Pure Catalog A receipt projection (C1).

The projection keeps historical observations and computes current
applicability separately.  It accepts the mapping-shaped receipts already
produced by the host.  A receipt object may also expose ``as_dict`` or
``to_dict``; this is the small compatibility boundary used by existing
receipt producers.

CATC0 is imported when available.  Its closed assertion envelope is an
authority/admission boundary, while C1 keeps a small receipt-facing view so
the reducer can remain incremental and storage-free.  If CATC0 is absent, the
local ``ReceiptLike`` protocol remains the fallback contract.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from enum import Enum
from typing import Protocol

try:
    from typing import TypeAlias
except ImportError:  # Python 3.9 compatibility for the v46 runner.
    TypeAlias = object  # type: ignore[misc,assignment]


if sys.version_info < (3, 10) and not getattr(dataclasses.dataclass, "_stateframe_py39", False):
    _dataclass = dataclasses.dataclass

    def _py39_dataclass(cls=None, /, **kwargs):
        kwargs.pop("slots", None)
        kwargs.pop("weakref_slot", None)
        if cls is None:
            return lambda target: _dataclass(target, **kwargs)
        return _dataclass(cls, **kwargs)

    _py39_dataclass._stateframe_py39 = True
    dataclasses.dataclass = _py39_dataclass

from dataclasses import dataclass

try:
    from enum import StrEnum
except ImportError:  # Python 3.9 compatibility for the v46 runner.
    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value

try:  # Flat v46 module loading.
    from kernel_catalog_contracts import ContentHandle as C0ContentHandle
    from kernel_catalog_contracts import CoverageRef as C0CoverageRef
    from kernel_catalog_contracts import EntityRef as C0EntityRef
    from kernel_catalog_contracts import Namespace as C0Namespace
except ImportError:  # pragma: no cover - package import fallback or pre-CATC0 tree.
    try:
        from .kernel_catalog_contracts import ContentHandle as C0ContentHandle
        from .kernel_catalog_contracts import CoverageRef as C0CoverageRef
        from .kernel_catalog_contracts import EntityRef as C0EntityRef
        from .kernel_catalog_contracts import Namespace as C0Namespace
    except ImportError:  # CATC0 is optional for isolated C1 development.
        C0ContentHandle = None  # type: ignore[assignment,misc]
        C0CoverageRef = None  # type: ignore[assignment,misc]
        C0EntityRef = None  # type: ignore[assignment,misc]
        C0Namespace = None  # type: ignore[assignment,misc]


SCHEMA = "catalog-a-projection/1"
DEFAULT_TOKEN_BUDGET = 512
UNKNOWN_REVISION = "UNKNOWN"
CATC0_AVAILABLE = all(item is not None for item in (C0ContentHandle, C0CoverageRef, C0EntityRef, C0Namespace))
ContentHandle = C0ContentHandle
CoverageHandle = C0CoverageRef
ContentRef = ContentHandle

Receipt: TypeAlias = Mapping[str, object]


class ReceiptLike(Protocol):
    """Minimal local protocol for existing receipt producers."""

    def as_dict(self) -> Mapping[str, object]:
        """Return the host-admitted receipt as a mapping."""


class CatalogError(ValueError):
    """Base error for malformed or unsupported catalog input."""


class CatalogIntegrityError(CatalogError):
    """A receipt identity was reused for different content."""


class UnsupportedReceiptError(CatalogError):
    """A receipt is outside the C1 read/search/edit/check boundary."""


class AssertionState(StrEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


class Availability(StrEnum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNAVAILABLE = "UNAVAILABLE"
    DENIED = "DENIED"
    UNSUPPORTED = "UNSUPPORTED"


class Resolution(StrEnum):
    READ = "READ"
    SEARCH = "SEARCH"
    CHECK = "CHECK"
    REFRESH = "REFRESH"
    USER_INPUT = "USER_INPUT"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class EntityRef:
    """Stable scoped identity; a locator alone is not treated as identity."""

    entity_id: str
    kind: str
    locator: str = ""
    namespace: tuple[tuple[str, str], ...] = ()
    generation: str | None = None

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise CatalogError("entity_id must not be empty")
        if not self.kind.strip():
            raise CatalogError("entity kind must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "locator": self.locator,
            "namespace": dict(self.namespace),
            "generation": self.generation,
        }

    def to_contract(self) -> object:
        """Adapt this receipt-facing identity to the imported CATC0 entity."""
        if not CATC0_AVAILABLE:
            raise CatalogError("kernel_catalog_contracts is not available")
        values = dict(self.namespace)
        namespace = C0Namespace(
            workspace_id=values.get("workspace_id", "default"),
            project_id=values.get("project_id"),
            resource_root_id=values.get("resource_root_id"),
        )
        return C0EntityRef(
            entity_id=self.entity_id,
            kind=self.kind,
            namespace=namespace,
            locator=self.locator,
            generation=self.generation,
        )


@dataclass(frozen=True, slots=True)
class DependencyRef:
    entity_id: str
    revision: str | None
    coverage_ref: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "revision": self.revision,
            "coverage_ref": self.coverage_ref,
        }


@dataclass(frozen=True, slots=True)
class CatalogAssertion:
    """One versioned observation and its computed applicability."""

    assertion_id: str
    subject: EntityRef
    predicate: str
    value: object
    revision: str | None
    dependencies: tuple[DependencyRef, ...]
    receipt_refs: tuple[str, ...]
    source_span_refs: tuple[str, ...]
    observed_at: object
    admitted_seq: int
    state: AssertionState
    availability: Availability
    stale_reason: str | None = None
    invalidated_by: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "assertion_id": self.assertion_id,
            "subject": self.subject.as_dict(),
            "predicate": self.predicate,
            "value": _copy(self.value),
            "revision": self.revision,
            "dependencies": [item.as_dict() for item in self.dependencies],
            "receipt_refs": list(self.receipt_refs),
            "source_span_refs": list(self.source_span_refs),
            "observed_at": _copy(self.observed_at),
            "admitted_seq": self.admitted_seq,
            "state": self.state.value,
            "availability": self.availability.value,
            "stale_reason": self.stale_reason,
            "invalidated_by": list(self.invalidated_by),
        }


@dataclass(frozen=True, slots=True)
class UnknownRecord:
    """A scoped missing or incomplete observation, never a guessed value."""

    unknown_id: str
    subject: EntityRef | None
    predicate: str
    reason: str
    resolution: Resolution
    required_by: tuple[str, ...]
    receipt_refs: tuple[str, ...]
    admitted_seq: int
    blocking_scope: str
    check_identity: str | None = None
    revision: str | None = None
    dependency_fingerprint: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "unknown_id": self.unknown_id,
            "subject": self.subject.as_dict() if self.subject else None,
            "predicate": self.predicate,
            "reason": self.reason,
            "resolution": self.resolution.value,
            "required_by": list(self.required_by),
            "receipt_refs": list(self.receipt_refs),
            "admitted_seq": self.admitted_seq,
            "blocking_scope": self.blocking_scope,
            "check_identity": self.check_identity,
            "revision": self.revision,
            "dependency_fingerprint": self.dependency_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class RevisionBinding:
    entity_id: str
    revision: str | None
    admitted_seq: int
    receipt_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "revision": self.revision,
            "admitted_seq": self.admitted_seq,
            "receipt_id": self.receipt_id,
        }


@dataclass(frozen=True, slots=True)
class CatalogChange:
    """A semantic projection delta for the exclusive sequence cursor."""

    sequence: int
    receipt_id: str
    kind: str
    entity_ids: tuple[str, ...]
    added_assertion_ids: tuple[str, ...]
    invalidated_assertion_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    reason: str
    resolved_unknown_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "receipt_id": self.receipt_id,
            "kind": self.kind,
            "entity_ids": list(self.entity_ids),
            "added_assertion_ids": list(self.added_assertion_ids),
            "invalidated_assertion_ids": list(self.invalidated_assertion_ids),
            "unknown_ids": list(self.unknown_ids),
            "reason": self.reason,
            "resolved_unknown_ids": list(self.resolved_unknown_ids),
        }


@dataclass(frozen=True, slots=True)
class QueryOmission:
    item_id: str
    reason: str
    estimated_tokens: int

    def as_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "reason": self.reason,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Bounded query response; entries never exceed ``token_budget``."""

    snapshot_seq: int
    basis_digest: str
    entries: tuple[object, ...] = ()
    unknowns: tuple[UnknownRecord, ...] = ()
    omissions: tuple[QueryOmission, ...] = ()
    estimated_tokens: int = 0
    token_budget: int = DEFAULT_TOKEN_BUDGET
    next_cursor: int | None = None

    @property
    def assertions(self) -> tuple[CatalogAssertion, ...]:
        return tuple(item for item in self.entries if isinstance(item, CatalogAssertion))

    @property
    def changes(self) -> tuple[CatalogChange, ...]:
        return tuple(item for item in self.entries if isinstance(item, CatalogChange))

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "snapshot_seq": self.snapshot_seq,
            "basis_digest": self.basis_digest,
            "entries": [_as_public(item) for item in self.entries],
            "unknowns": [item.as_dict() for item in self.unknowns],
            "omissions": [item.as_dict() for item in self.omissions],
            "estimated_tokens": self.estimated_tokens,
            "token_budget": self.token_budget,
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True, slots=True)
class _AssertionDraft:
    subject: EntityRef
    predicate: str
    value: object
    revision: str | None
    dependencies: tuple[DependencyRef, ...]
    source_span_refs: tuple[str, ...]
    observed_at: object
    availability: Availability = Availability.PRESENT
    requires_revision: bool = True


@dataclass(frozen=True, slots=True)
class _UnknownDraft:
    subject: EntityRef | None
    predicate: str
    reason: str
    resolution: Resolution
    required_by: tuple[str, ...]
    blocking_scope: str
    check_identity: str | None = None
    revision: str | None = None
    dependency_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogState:
    """Immutable projection state; applying a receipt returns a new state."""

    watermark: int = 0
    assertion_records: tuple[CatalogAssertion, ...] = ()
    unknown_records: tuple[UnknownRecord, ...] = ()
    change_records: tuple[CatalogChange, ...] = ()
    revision_bindings: tuple[RevisionBinding, ...] = ()
    receipt_digests: tuple[tuple[str, str], ...] = ()

    @property
    def assertions(self) -> tuple[CatalogAssertion, ...]:
        return self.assertion_records

    @property
    def unknown_records_view(self) -> tuple[UnknownRecord, ...]:
        return self.unknown_records

    @property
    def changes_view(self) -> tuple[CatalogChange, ...]:
        return self.change_records

    def apply(self, receipt: object) -> "CatalogState":
        return reduce_receipt(self, receipt)

    def reduce(self, receipt: object) -> "CatalogState":
        """Reducer-shaped alias for integration code that owns a state object."""
        return self.apply(receipt)

    def by_entity(
        self,
        entity_id: str,
        *,
        predicates: Sequence[str] = (),
        mode: str = "current",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cursor: int = 0,
    ) -> QueryResult:
        _validate_mode(mode)
        wanted = {str(item) for item in predicates if str(item)}
        records = [
            item for item in self.assertion_records
            if _assertion_matches_entity(item, entity_id)
            and (not wanted or item.predicate in wanted)
            and (mode == "historical" or item.state in {AssertionState.CURRENT, AssertionState.CONFLICT})
        ]
        records.sort(key=lambda item: (item.admitted_seq, item.assertion_id))
        return self._budget(records, token_budget=token_budget, cursor=cursor)

    def lookup(
        self,
        entity_id: str,
        predicates: Sequence[str] = (),
        *,
        mode: str = "current",
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cursor: int = 0,
    ) -> QueryResult:
        """Report-compatible alias for the by-entity query."""
        return self.by_entity(
            entity_id,
            predicates=predicates,
            mode=mode,
            token_budget=token_budget,
            cursor=cursor,
        )

    def changed_since(
        self,
        sequence: int,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cursor: int = 0,
    ) -> QueryResult:
        if type(sequence) is not int or sequence < 0:
            raise CatalogError("sequence must be a non-negative integer")
        records = [item for item in self.change_records if item.sequence > sequence]
        records.sort(key=lambda item: (item.sequence, item.receipt_id))
        return self._budget(records, token_budget=token_budget, cursor=cursor)

    def changes(
        self,
        sequence: int,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cursor: int = 0,
    ) -> QueryResult:
        """Report-compatible alias with an exclusive sequence cursor."""
        return self.changed_since(sequence, token_budget=token_budget, cursor=cursor)

    def stale(
        self,
        *,
        entity_id: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cursor: int = 0,
    ) -> QueryResult:
        records = [
            item for item in self.assertion_records
            if item.state == AssertionState.STALE
            and (entity_id is None or _assertion_matches_entity(item, entity_id))
        ]
        records.sort(key=lambda item: (item.admitted_seq, item.assertion_id))
        return self._budget(records, token_budget=token_budget, cursor=cursor)

    def unknowns(
        self,
        *,
        entity_id: str | None = None,
        required_by: str | None = None,
        predicate: str | None = None,
        scope: str | None = None,
        blocking_scope: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cursor: int = 0,
    ) -> QueryResult:
        if scope is not None and blocking_scope is not None and scope != blocking_scope:
            raise CatalogError("scope and blocking_scope must match when both are supplied")
        requested_scope = blocking_scope if blocking_scope is not None else scope
        records = [
            item for item in self.unknown_records
            if (entity_id is None or (item.subject is not None and item.subject.entity_id == entity_id))
            and (required_by is None or required_by in item.required_by)
            and (predicate is None or item.predicate == predicate)
            and (requested_scope is None or item.blocking_scope == requested_scope)
        ]
        records.sort(key=lambda item: (item.admitted_seq, item.unknown_id))
        return self._budget_unknowns(records, token_budget=token_budget, cursor=cursor)

    def blocking_unknowns(
        self, *, required_by: str | None = None, blocking_scope: str | None = None,
    ) -> tuple[UnknownRecord, ...]:
        """Return host truth without applying a rendering/token budget."""
        records = tuple(
            item for item in self.unknown_records
            if (blocking_scope is not None or item.blocking_scope != "diagnostic")
            if (required_by is None or required_by in item.required_by)
            and (blocking_scope is None or item.blocking_scope == blocking_scope)
        )
        return tuple(sorted(records, key=lambda item: (item.admitted_seq, item.unknown_id)))

    def unknown(
        self,
        *,
        entity_id: str | None = None,
        required_by: str | None = None,
        predicate: str | None = None,
        scope: str | None = None,
        blocking_scope: str | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        cursor: int = 0,
    ) -> QueryResult:
        """Singular alias for callers using the report's query wording."""
        return self.unknowns(
            entity_id=entity_id,
            required_by=required_by,
            predicate=predicate,
            scope=scope,
            blocking_scope=blocking_scope,
            token_budget=token_budget,
            cursor=cursor,
        )

    def _budget(
        self,
        records: Sequence[object],
        *,
        token_budget: int,
        cursor: int,
    ) -> QueryResult:
        budget = _validate_budget(token_budget)
        start = _validate_cursor(cursor, len(records))
        selected: list[object] = []
        omissions: list[QueryOmission] = []
        used = 0
        next_cursor: int | None = None
        for index in range(start, len(records)):
            item = records[index]
            cost = _estimate_tokens(item)
            if used + cost > budget:
                omissions.append(QueryOmission(_item_id(item), "TOKEN_BUDGET", cost))
                next_cursor = index + 1 if index + 1 < len(records) else None
                break
            selected.append(item)
            used += cost
        return QueryResult(
            snapshot_seq=self.watermark,
            basis_digest=self.basis_digest(),
            entries=tuple(selected),
            omissions=tuple(omissions),
            estimated_tokens=used,
            token_budget=budget,
            next_cursor=next_cursor,
        )

    def _budget_unknowns(
        self,
        records: Sequence[UnknownRecord],
        *,
        token_budget: int,
        cursor: int,
    ) -> QueryResult:
        budget = _validate_budget(token_budget)
        start = _validate_cursor(cursor, len(records))
        selected: list[UnknownRecord] = []
        omissions: list[QueryOmission] = []
        used = 0
        next_cursor: int | None = None
        for index in range(start, len(records)):
            item = records[index]
            cost = _estimate_tokens(item)
            if used + cost > budget:
                omissions.append(QueryOmission(item.unknown_id, "TOKEN_BUDGET", cost))
                next_cursor = index + 1 if index + 1 < len(records) else None
                break
            selected.append(item)
            used += cost
        return QueryResult(
            snapshot_seq=self.watermark,
            basis_digest=self.basis_digest(),
            unknowns=tuple(selected),
            omissions=tuple(omissions),
            estimated_tokens=used,
            token_budget=budget,
            next_cursor=next_cursor,
        )

    def basis_digest(self) -> str:
        payload = {
            "watermark": self.watermark,
            "assertions": [item.as_dict() for item in self.assertion_records],
            "unknowns": [item.as_dict() for item in self.unknown_records],
            "changes": [item.as_dict() for item in self.change_records],
            "revisions": [item.as_dict() for item in self.revision_bindings],
        }
        return _digest(payload)


def empty_catalog() -> CatalogState:
    """Return an empty projection."""
    return CatalogState()


def reduce_receipt(state: CatalogState, receipt: object) -> CatalogState:
    """Apply one admitted read/search/edit/check receipt without side effects."""

    if not isinstance(state, CatalogState):
        raise TypeError("state must be a CatalogState")
    record = normalize_receipt(_receipt_mapping(receipt))
    receipt_digest = _digest(record)
    receipt_id = _receipt_id(record, receipt_digest)
    seen = dict(state.receipt_digests)
    prior_digest = seen.get(receipt_id)
    if prior_digest is not None:
        if prior_digest != receipt_digest:
            raise CatalogIntegrityError(f"receipt identity reused with different content: {receipt_id}")
        return state

    kind = _receipt_kind(record)
    sequence = _sequence(record, state.watermark + 1)
    namespace = _namespace(record)
    entities = _entities(record, kind, namespace)
    revision_updates = _revision_updates(record, kind, entities, namespace)
    revisions = {item.entity_id: item for item in state.revision_bindings}
    assertions = list(state.assertion_records)
    unknowns = list(state.unknown_records)
    invalidated_ids: list[str] = []
    change_reasons: list[str] = []

    for entity_id, revision in revision_updates:
        prior = revisions.get(entity_id)
        if prior is not None and prior.revision != revision and _binding_is_newer(prior, sequence, receipt_id):
            assertions, affected = _invalidate(
                assertions,
                entity_id=entity_id,
                receipt_id=receipt_id,
                reason="revision_changed" if revision is not None else "revision_changed_without_binding",
            )
            invalidated_ids.extend(affected)
            change_reasons.append("revision_changed")
        if prior is None or _binding_is_newer(prior, sequence, receipt_id):
            revisions[entity_id] = RevisionBinding(entity_id, revision, sequence, receipt_id)

    drafts = _assertion_drafts(record, kind, entities, revisions, namespace)
    added_ids: list[str] = []
    for draft in drafts:
        assertion, assertions = _upsert_assertion(
            assertions,
            draft=draft,
            receipt_id=receipt_id,
            sequence=sequence,
            record=record,
            revisions=revisions,
        )
        if assertion.assertion_id not in added_ids:
            added_ids.append(assertion.assertion_id)

    resolved_unknown_ids = _resolved_unknown_ids_for_state(
        state.unknown_records,
        record,
        kind=kind,
        entities=entities,
    )
    if resolved_unknown_ids:
        retained_unknowns = [
            item for item in unknowns if item.unknown_id not in resolved_unknown_ids
        ]
        if len(retained_unknowns) != len(unknowns):
            change_reasons.append("unknown_resolved")
        unknowns = retained_unknowns
    unknown_drafts = _unknown_drafts(record, kind, entities, drafts, revisions)
    unknown_ids: list[str] = []
    for draft in unknown_drafts:
        unknown, unknowns = _upsert_unknown(
            unknowns,
            draft=draft,
            receipt_id=receipt_id,
            sequence=sequence,
        )
        if unknown.unknown_id not in unknown_ids:
            unknown_ids.append(unknown.unknown_id)

    if not drafts and not unknown_drafts:
        change_reasons.append("receipt_observed")
    reason = ";".join(dict.fromkeys(change_reasons)) or kind
    change = CatalogChange(
        sequence=sequence,
        receipt_id=receipt_id,
        kind=kind,
        entity_ids=tuple(sorted({item.entity_id for item in entities})),
        added_assertion_ids=tuple(added_ids),
        invalidated_assertion_ids=tuple(dict.fromkeys(invalidated_ids)),
        unknown_ids=tuple(unknown_ids),
        reason=reason,
        resolved_unknown_ids=tuple(sorted(resolved_unknown_ids)),
    )
    seen[receipt_id] = receipt_digest
    return CatalogState(
        watermark=max(state.watermark, sequence),
        assertion_records=tuple(assertions),
        unknown_records=tuple(unknowns),
        change_records=state.change_records + (change,),
        revision_bindings=tuple(sorted(revisions.values(), key=lambda item: item.entity_id)),
        receipt_digests=tuple(sorted(seen.items())),
    )


def replay(receipts: Iterable[object], state: CatalogState | None = None) -> CatalogState:
    """Replay an iterable in journal order; duplicate delivery is a no-op."""
    current = state or empty_catalog()
    for receipt in receipts:
        current = reduce_receipt(current, receipt)
    return current


# Naming aliases keep the pure API easy to discover without introducing a
# second mutable reducer class.
CatalogProjection = CatalogState
CatalogReducer = CatalogState
apply_receipt = reduce_receipt
project_receipts = replay


def _receipt_mapping(receipt: object) -> dict[str, object]:
    if isinstance(receipt, Mapping):
        return copy.deepcopy(dict(receipt))
    for name in ("as_dict", "to_dict"):
        method = getattr(receipt, name, None)
        if callable(method):
            value = method()
            if isinstance(value, Mapping):
                return copy.deepcopy(dict(value))
    raise CatalogError("receipt must be a mapping or expose as_dict()/to_dict()")


def normalize_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    """Adapt one admitted host receipt into the language-neutral C1 shape.

    Native runners commonly keep product/test/environment bindings nested under
    ``binding`` and ``after_binding``.  The adapter copies those facts into the
    canonical envelope; it never reads the current workspace to fill a missing
    historical binding.
    """
    record = copy.deepcopy(dict(receipt))
    kind = _receipt_kind(record)
    if kind != "check":
        return record

    execution_id = _host_execution_id(record)
    if execution_id is not None:
        record.setdefault("host_execution_id", execution_id)

    before = _binding_mapping(record, "binding", "before_binding", "before")
    after = _binding_mapping(record, "after_binding", "after", "candidate_binding")
    if before is not None:
        record.setdefault("before_binding", copy.deepcopy(before))
    if after is not None:
        record.setdefault("after_binding", copy.deepcopy(after))

    # Copy opaque product/test/environment identities from the host receipt.
    # Missing historical values remain missing; the adapter never consults
    # current workspace bytes to fill them.
    for prefix, binding in (("before", before), ("after", after)):
        if binding is None:
            continue
        canonical = _canonical_binding(binding)
        record.setdefault(f"canonical_{prefix}_binding", canonical)
        for name, value in canonical.items():
            record.setdefault(f"{prefix}_{name}", value)

    check_identity = _check_identity(record)
    if check_identity is not None:
        record.setdefault("check_identity", check_identity)
    selectors = _selectors(record)
    if selectors:
        record.setdefault("selectors", list(selectors))

    product_revision = (
        _binding_value(after, "product_revision", "artifact", "revision", "content_digest")
    )
    if product_revision is not None and record.get("product_revision") in (None, ""):
        record["product_revision"] = product_revision
    environment_revision = (
        _binding_value(after, "environment_revision", "environment", "test_environment_revision")
    )
    if environment_revision is not None:
        record.setdefault("environment_revision", environment_revision)

    stability = record.get("binding_stable")
    if type(stability) is not bool:
        detail = record.get("binding_stability")
        if isinstance(detail, Mapping):
            status = str(detail.get("status", "")).upper()
            if status in {"STABLE", "UNSTABLE"}:
                stability = status == "STABLE"
        elif before is not None and after is not None:
            # Do not infer stability from equality: the runner's stability
            # policy includes generated/test-artifact filtering and write sets.
            stability = None
    if type(stability) is bool:
        record["binding_stable"] = stability

    outcome = _canonical_check_outcome(record)
    if outcome is not None:
        record.setdefault("outcome", outcome)
        record.setdefault("status", outcome)
    record.setdefault("canonical_check_identity", record.get("check_identity"))
    record.setdefault("canonical_selectors", list(selectors))
    # Absence is meaningful.  Do not turn a missing execution binding into a
    # present-but-unstable binding that host-fact adapters must certify.
    if before is not None:
        record.setdefault("canonical_before_binding", copy.deepcopy(before))
    if after is not None:
        record.setdefault("canonical_after_binding", copy.deepcopy(after))
    record.setdefault("canonical_outcome", outcome or "UNKNOWN")
    return record


def _host_execution_id(record: Mapping[str, object]) -> str | None:
    """Return the opaque host execution identity shared by rich/legacy views."""
    for key in ("host_execution_id", "execution_id", "run_id", "command_id", "invocation_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value).strip()
    execution = record.get("execution")
    if isinstance(execution, Mapping):
        for key in ("host_execution_id", "execution_id", "run_id", "id"):
            value = execution.get(key)
            if value not in (None, ""):
                return str(value).strip()
    return None


def coalesce_execution_receipts(receipts: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Join legacy/rich projections that describe one host execution."""
    result: list[dict[str, object]] = []
    by_execution: dict[str, int] = {}
    projection_rank: dict[str, int] = {}

    def rank(record: Mapping[str, object]) -> int:
        projection = str(record.get("projection", record.get("receipt_projection", ""))).lower()
        if projection in {"rich", "canonical", "adapted"}:
            return 2
        if projection in {"legacy", "raw"}:
            return 0
        return int(any(key in record for key in (
            "canonical_before_binding", "canonical_after_binding", "observation",
            "observed", "after_binding",
        )))

    for raw in receipts:
        if not isinstance(raw, Mapping):
            continue
        record = copy.deepcopy(dict(raw))
        execution_id = _host_execution_id(record)
        if not execution_id or execution_id not in by_execution:
            if execution_id:
                by_execution[execution_id] = len(result)
                projection_rank[execution_id] = rank(record)
                record.setdefault("host_execution_id", execution_id)
                source_ids = [str(record["receipt_id"])] if record.get("receipt_id") else []
                record.setdefault("source_receipt_ids", source_ids)
            result.append(record)
            continue
        index = by_execution[execution_id]
        merged = result[index]
        source_ids = list(merged.get("source_receipt_ids", ()))
        if record.get("receipt_id"):
            source_ids.append(str(record["receipt_id"]))
        # Rich execution fields win; legacy fields only fill missing values.
        if rank(record) > projection_rank.get(execution_id, 0):
            preferred, fallback = record, merged
            projection_rank[execution_id] = rank(record)
        else:
            preferred, fallback = merged, record
        merged = copy.deepcopy(dict(preferred))
        for key, value in fallback.items():
            if key in {"receipt_id", "source_receipt_ids"}:
                continue
            if key not in merged or merged[key] in (None, "", (), [], {}):
                merged[key] = value
        merged["source_receipt_ids"] = list(dict.fromkeys(source_ids))
        merged["host_execution_id"] = execution_id
        result[index] = merged
    return tuple(result)


def _binding_mapping(record: Mapping[str, object], *keys: str) -> Mapping[str, object] | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _binding_value(binding: Mapping[str, object] | None, *keys: str) -> str | None:
    if binding is None:
        return None
    for key in keys:
        value = binding.get(key)
        if value not in (None, ""):
            return _revision_text(value)
    return None


def _canonical_binding(binding: Mapping[str, object]) -> dict[str, object]:
    """Normalize opaque product, test, and environment binding dimensions."""
    aliases = {
        "product_revision": ("product_revision", "artifact", "revision", "content_digest"),
        "test_revision": ("test_revision", "test_binding", "check_revision", "test_digest"),
        "environment_revision": (
            "environment_revision", "environment", "test_environment_revision",
            "environment_digest",
        ),
    }
    result: dict[str, object] = {}
    for name, keys in aliases.items():
        value = _binding_value(binding, *keys)
        if value is not None:
            result[name] = value
    for key in ("binding_id", "test_id", "environment_id", "stable"):
        if binding.get(key) not in (None, ""):
            result[key] = copy.deepcopy(binding[key])
    return result


def _selectors(record: Mapping[str, object]) -> tuple[str, ...]:
    raw = record.get("selectors", record.get("selector"))
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raw = ()
    values: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            value = item.get("id") or item.get("name") or item.get("selector")
        else:
            value = item
        if value not in (None, ""):
            values.append(str(value).strip())
    if not values:
        for key in ("check_id", "suite", "command"):
            value = record.get(key)
            if value not in (None, ""):
                values.append(str(value).strip())
                break
    return tuple(dict.fromkeys(item for item in values if item))


def _check_identity(record: Mapping[str, object]) -> str | None:
    explicit = record.get("check_identity")
    if explicit not in (None, ""):
        return str(explicit).strip()
    for key in ("check_id", "selector", "suite", "command"):
        value = record.get(key)
        if value not in (None, ""):
            return f"check:{str(value).strip()}"
    selectors = _selectors(record)
    return "check:" + _digest({"selectors": selectors})[:24] if selectors else None


def _canonical_check_outcome(record: Mapping[str, object]) -> str | None:
    from kernel_step_contract import environment_check_reason, product_check_failure
    if product_check_failure(record):
        return "FAIL"
    if environment_check_reason(record) is not None:
        return "ENVIRONMENT"
    known = {
        "PASS": "PASS", "PASSED": "PASS", "GREEN": "PASS", "OK": "PASS",
        "SUCCESS": "PASS", "SUCCEEDED": "PASS", "COMPLETE": "PASS",
        "COMPLETED": "PASS",
        "FAIL": "FAIL", "FAILED": "FAIL", "RED": "FAIL",
        "NONZERO": "FAIL", "NONZERO_EXIT": "FAIL",
        "UNRUNNABLE": "UNRUNNABLE", "UNSUPPORTED": "UNRUNNABLE",
        "NOT_RUN": "UNRUNNABLE", "TIMEOUT": "ENVIRONMENT",
        "ENVIRONMENT": "ENVIRONMENT",
        "RESOURCE_LIMIT": "ENVIRONMENT",
    }
    for key in ("outcome", "result_class", "result", "status", "verdict"):
        value = record.get(key)
        if value not in (None, ""):
            text = str(getattr(value, "value", value)).strip().upper()
            normalized = known.get(text, text or "UNKNOWN")
            if normalized == "PASS" and type(record.get("exit_code")) is int \
                    and record["exit_code"] != 0:
                return "FAIL"
            return normalized
    exit_code = record.get("exit_code")
    if type(exit_code) is int:
        return "PASS" if exit_code == 0 else "FAIL"
    return None


def _receipt_id(record: Receipt, digest: str) -> str:
    value = record.get("receipt_id") or record.get("event_id") or record.get("id")
    return str(value).strip() if value is not None and str(value).strip() else f"receipt:{digest[:24]}"


def _receipt_kind(record: Receipt) -> str:
    raw = str(record.get("kind") or record.get("type") or record.get("action") or record.get("tool") or "")
    lowered = raw.strip().lower().replace("-", "_")
    if lowered in {"read", "file_read", "cat", "open"} or "read" in lowered:
        return "read"
    if lowered in {"search", "grep", "find", "ripgrep", "rg"} or any(item in lowered for item in ("search", "grep")):
        return "search"
    if lowered in {"edit", "write", "apply_patch", "patch", "file_edit"} or any(item in lowered for item in ("edit", "patch", "write")):
        return "edit"
    if lowered in {"check", "test", "run_tests", "verification", "pytest", "unittest"} or any(item in lowered for item in ("check", "test", "verify")):
        return "check"
    raise UnsupportedReceiptError(f"unsupported receipt kind: {raw or '<missing>'}")


def _sequence(record: Receipt, fallback: int) -> int:
    value = record.get("admitted_seq", record.get("sequence", record.get("seq", fallback)))
    if type(value) is not int or value < 0:
        raise CatalogError("admitted_seq/sequence/seq must be a non-negative integer")
    return value


def _namespace(record: Receipt) -> tuple[tuple[str, str], ...]:
    source = record.get("namespace")
    values: dict[str, object] = dict(source) if isinstance(source, Mapping) else {}
    for key in ("workspace_id", "project_id", "resource_root_id"):
        if key not in values and record.get(key) not in (None, ""):
            values[key] = record[key]
    return tuple(sorted((str(key), str(value)) for key, value in values.items() if str(value).strip()))


def _entities(record: Receipt, kind: str, namespace: tuple[tuple[str, str], ...]) -> tuple[EntityRef, ...]:
    found: list[EntityRef] = []

    def add(value: object, default_kind: str = "artifact", *, preserve_id: bool = False) -> None:
        if isinstance(value, Mapping):
            entity_id = str(value.get("entity_id") or value.get("id") or "").strip()
            preserve_mapping_id = bool(entity_id)
            entity_kind = str(value.get("kind") or default_kind).strip().lower()
            locator = _locator(value.get("locator") or value.get("path") or value.get("target") or entity_id)
            local_namespace = _namespace(value) or namespace
            generation = _optional_text(value.get("generation"))
        else:
            entity_id = str(value or "").strip()
            preserve_mapping_id = False
            entity_kind = default_kind
            locator = _locator(entity_id)
            local_namespace = namespace
            generation = None
        if not entity_id:
            return
        if not preserve_id and not preserve_mapping_id and not (":" in entity_id or entity_id.startswith("entity:")) and locator:
            entity_id = _scoped_entity_id(entity_kind, locator, local_namespace)
        candidate = EntityRef(entity_id, entity_kind or default_kind, locator, local_namespace, generation)
        if candidate.entity_id not in {item.entity_id for item in found}:
            found.append(candidate)

    explicit = record.get("entities")
    if isinstance(explicit, Mapping):
        explicit = tuple(explicit.values())
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes, bytearray)):
        for item in explicit:
            add(item, preserve_id=True)
    for key in ("entity_id", "entity_ids"):
        value = record.get(key)
        if key == "entity_ids" and isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for item in value:
                add(item, preserve_id=True)
        elif key == "entity_id":
            add(value, "check" if kind == "check" else "artifact", preserve_id=True)

    if kind == "check":
        check_id = record.get("check_id") or record.get("selector") or record.get("suite")
        if check_id:
            add(f"check:{check_id}", "check")
    hit_values = record.get("hits", record.get("matches", ()))
    if kind == "search" and isinstance(hit_values, Sequence) and not isinstance(hit_values, (str, bytes, bytearray)):
        for hit in hit_values:
            if isinstance(hit, Mapping):
                hit_entity_id = hit.get("entity_id")
                add(
                    hit_entity_id or hit.get("path") or hit.get("file") or hit.get("target"),
                    "file",
                    preserve_id=hit_entity_id is not None,
                )
            else:
                add(hit, "file")

    path_keys = ("target_path", "path", "file_path", "file", "target", "locator")
    for key in path_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            add(value, "file")
            break

    if kind == "search" and not found:
        scope = record.get("scope") or record.get("root") or record.get("query") or "search"
        add(f"search:{_digest(scope)[:16]}", "search")
    return tuple(found)


def _scoped_entity_id(kind: str, locator: str, namespace: tuple[tuple[str, str], ...]) -> str:
    if not namespace:
        return f"{kind}:{locator}"
    # Keep generated IDs compatible with CATC0's identifier grammar.
    scope = ".".join(f"{key}.{value}" for key, value in namespace)
    return f"{scope}:{kind}:{locator}"


def _locator(value: object) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("./")


def _revision_value(record: Receipt, *, edit_after: bool = False) -> str | None:
    is_check = _receipt_kind(record) == "check"
    binding_keys = (
        ("after_binding", "canonical_after_binding")
        if is_check else
        ("after_binding", "canonical_after_binding", "binding", "canonical_before_binding")
        if edit_after else
        ("binding", "canonical_before_binding", "after_binding", "canonical_after_binding")
    )
    for key in binding_keys:
        nested = _binding_mapping(record, key)
        nested_value = _binding_value(
            nested, "product_revision", "artifact", "revision", "content_digest",
        )
        if nested_value is not None:
            return nested_value
    keys = (
        ("after_revision", "new_revision", "after_digest", "after_content_digest")
        if edit_after else
        ("revision", "source_revision", "product_revision", "artifact_revision", "workspace_revision", "content_digest", "digest")
    )
    for key in keys:
        if record.get(key) not in (None, ""):
            return _revision_text(record[key])
    return None


def _revision_text(value: object) -> str:
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return "sha256:" + _digest(value)
    return str(value).strip()


def _revision_updates(
    record: Receipt,
    kind: str,
    entities: Sequence[EntityRef],
    namespace: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str | None], ...]:
    updates: dict[str, str | None] = {}
    explicit = record.get("entity_revisions")
    if isinstance(explicit, Mapping):
        for entity_id, revision in explicit.items():
            updates[_dependency_entity_id(entity_id, namespace)] = (
                _revision_text(revision) if revision not in (None, "") else None
            )
    revision = _revision_value(record, edit_after=kind == "edit")
    applied = _edit_applied(record) if kind == "edit" else True
    if kind != "edit" or applied:
        for entity in entities:
            if revision is not None or kind == "edit":
                updates.setdefault(entity.entity_id, revision)
    dependencies = record.get("dependencies", record.get("depends_on", ()))
    if isinstance(dependencies, Mapping):
        dependencies = tuple(dependencies.values())
    if isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes, bytearray)):
        for item in dependencies:
            if not isinstance(item, Mapping):
                continue
            entity_id = _dependency_entity_id(
                item.get("entity_id") or item.get("id") or item.get("path") or "",
                namespace,
            )
            if entity_id and item.get("revision") not in (None, ""):
                updates[entity_id] = _revision_text(item["revision"])
    return tuple(sorted(updates.items()))


def _assertion_drafts(
    record: Receipt,
    kind: str,
    entities: Sequence[EntityRef],
    revisions: Mapping[str, RevisionBinding],
    namespace: tuple[tuple[str, str], ...],
) -> tuple[_AssertionDraft, ...]:
    source_spans = _strings(record.get("source_span_refs", record.get("spans", ())))
    observed_at = record.get("observed_at", record.get("timestamp_ns", record.get("timestamp")))
    global_revision = _revision_value(record, edit_after=kind == "edit")
    drafts: list[_AssertionDraft] = []

    if kind == "edit":
        if not _edit_applied(record):
            return ()
        for entity in entities:
            revision = global_revision or _binding_revision(revisions, entity.entity_id)
            dependencies = (DependencyRef(entity.entity_id, revision),)
            drafts.append(_AssertionDraft(
                entity,
                "edit.applied",
                _compact_record(record, ("operation", "target_path", "before_revision", "after_revision", "changed_spans")),
                revision,
                dependencies,
                source_spans,
                observed_at,
                requires_revision=revision is not None,
            ))
        return tuple(drafts)

    if kind == "read":
        complete = _complete(record)
        for entity in entities:
            revision = global_revision or _binding_revision(revisions, entity.entity_id)
            dependencies = (DependencyRef(entity.entity_id, revision),)
            value = _compact_record(record, (
                "path", "target_path", "file_path", "content_digest", "digest", "byte_length",
                "encoding", "coverage", "complete", "status", "result", "parse_status",
            ))
            drafts.append(_AssertionDraft(
                entity,
                "file.observed",
                value,
                revision,
                dependencies,
                source_spans,
                observed_at,
                availability=Availability.PRESENT if complete else Availability.UNAVAILABLE,
                requires_revision=True,
            ))
        return tuple(drafts)

    if kind == "search":
        hits = record.get("hits", record.get("matches", ()))
        hit_by_id: dict[str, object] = {}
        if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes, bytearray)):
            for hit in hits:
                if isinstance(hit, Mapping):
                    path = hit.get("entity_id") or hit.get("path") or hit.get("file") or hit.get("target")
                    if path:
                        hit_by_id[str(path)] = _copy(hit)
                elif hit:
                    hit_by_id[str(hit)] = str(hit)
        for entity in entities:
            revision = global_revision or _binding_revision(revisions, entity.entity_id)
            value = hit_by_id.get(entity.entity_id, _compact_record(record, (
                "query", "pattern", "path", "roots", "glob", "coverage", "complete", "truncated", "hit_count",
            )))
            drafts.append(_AssertionDraft(
                entity,
                "search.observed",
                value,
                revision,
                (DependencyRef(entity.entity_id, revision),),
                source_spans,
                observed_at,
                availability=Availability.PRESENT,
                requires_revision=revision is not None,
            ))
        return tuple(drafts)

    # check
    if not _check_settled(record):
        # No fake check value (W4 / P0.1): an unsettled check is recorded only
        # as an unknown debt.  An UNAVAILABLE "UNKNOWN" assertion used to
        # conflict with the later PASS and leave check.result unavailable.
        return tuple(drafts)
    for entity in entities:
        revision = global_revision or _binding_revision(revisions, entity.entity_id)
        dependencies = _dependencies_for(record, entities, revision, namespace)
        drafts.append(_AssertionDraft(
            entity,
            "check.result",
            _compact_record(record, (
                "check_id", "selector", "suite", "status", "result", "result_class", "outcome",
                "exit_code", "passed", "failed", "skipped", "total", "counts", "product_revision",
                "output_reference", "applicability",
            )),
            revision,
            dependencies,
            source_spans,
            observed_at,
            availability=Availability.PRESENT if _check_settled(record) else Availability.UNAVAILABLE,
            requires_revision=True,
        ))
    return tuple(drafts)


def _dependencies_for(
    record: Receipt,
    entities: Sequence[EntityRef],
    revision: str | None,
    namespace: tuple[tuple[str, str], ...],
) -> tuple[DependencyRef, ...]:
    found: list[DependencyRef] = []
    raw = record.get("dependencies", record.get("depends_on", ()))
    if isinstance(raw, Mapping):
        raw = tuple(raw.values())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if isinstance(item, Mapping):
                entity_id = _dependency_entity_id(
                    item.get("entity_id") or item.get("id") or item.get("path") or "",
                    namespace,
                )
                item_revision = item.get("revision", revision)
                coverage = _optional_text(item.get("coverage_ref"))
            else:
                entity_id = _dependency_entity_id(item, namespace)
                item_revision = revision
                coverage = None
            if entity_id:
                found.append(DependencyRef(entity_id, _revision_text(item_revision) if item_revision not in (None, "") else None, coverage))
    if not found:
        found = [DependencyRef(entity.entity_id, revision) for entity in entities]
    return _unique_dependencies(found)


def _unknown_drafts(
    record: Receipt,
    kind: str,
    entities: Sequence[EntityRef],
    drafts: Sequence[_AssertionDraft],
    revisions: Mapping[str, RevisionBinding],
) -> tuple[_UnknownDraft, ...]:
    result: list[_UnknownDraft] = []
    required_by = _strings(record.get("required_by", record.get("required_for", ())))
    scope = _optional_text(record.get("scope"))
    if not scope:
        # Incidental exploration and failed/no-op edits remain history facts;
        # an admitted contract may opt them into a blocking scope explicitly.
        scope = "diagnostic" if kind in {"read", "search", "edit"} and not required_by else "receipt"

    explicit = record.get("unknowns", record.get("unknown"))
    if isinstance(explicit, Mapping):
        explicit = (explicit,)
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes, bytearray)):
        for item in explicit:
            if not isinstance(item, Mapping):
                continue
            subject = _unknown_subject(item, entities)
            result.append(_UnknownDraft(
                subject,
                str(item.get("predicate") or "catalog.unknown"),
                str(item.get("reason") or "unknown"),
                _resolution(item.get("resolution"), kind),
                _strings(item.get("required_by", required_by)),
                str(item.get("blocking_scope") or scope),
            ))

    if kind == "read" and (not _complete(record) or not _revision_value(record)):
        result.extend(_UnknownDraft(
            entity,
            "file.content",
            "incomplete_read" if not _complete(record) else "missing_revision",
            Resolution.READ if _complete(record) else Resolution.REFRESH,
            required_by,
            scope,
        ) for entity in entities)
    elif kind == "search":
        if _search_incomplete(record):
            result.append(_UnknownDraft(
                entities[0] if entities else None,
                "search.coverage",
                "search_coverage_incomplete",
                Resolution.SEARCH,
                required_by,
                scope,
            ))
        elif not _search_has_hits(record):
            result.append(_UnknownDraft(
                entities[0] if entities else None,
                "search.no_match",
                "no_match_only_within_declared_scope",
                Resolution.SEARCH,
                required_by,
                scope,
            ))
    elif kind == "edit" and not _edit_applied(record):
        result.append(_UnknownDraft(
            entities[0] if entities else None,
            "edit.applied",
            "edit_not_applied",
            Resolution.REFRESH,
            required_by,
            scope,
        ))
    elif kind == "check":
        # One opening -> one debt (W4 / P0.1): a receipt that declares its own
        # unknowns is an epistemic observation, not a test execution.  It must
        # not also mint a synthetic check.result twin under a derived subject,
        # which no explicit resolution could ever name.
        declared_result = bool(result)
        if not _check_settled(record) and not declared_result:
            result.append(_UnknownDraft(
                drafts[0].subject if drafts else (entities[0] if entities else None),
                "check.result",
                "check_not_settled",
                Resolution.CHECK,
                required_by,
                scope,
            ))
        if not _revision_value(record):
            result.append(_UnknownDraft(
                drafts[0].subject if drafts else (entities[0] if entities else None),
                "check.revision",
                "missing_revision_binding",
                Resolution.REFRESH,
                required_by,
                scope,
            ))
        if (_is_executed_check(record)
                and (("binding" in record or "after_binding" in record
                      or "canonical_before_binding" in record or "canonical_after_binding" in record)
                     and record.get("binding_stable") is not True
                     or record.get("binding_stable") is False)):
            result.append(_UnknownDraft(
                drafts[0].subject if drafts else (entities[0] if entities else None),
                "check.binding",
                "unstable_binding",
                Resolution.REFRESH,
                required_by,
                scope,
            ))
        missing_dependencies = tuple(
            dependency.entity_id
            for draft in drafts
            for dependency in draft.dependencies
            if revisions.get(dependency.entity_id) is None
        )
        if missing_dependencies:
            result.append(_UnknownDraft(
                drafts[0].subject if drafts else (entities[0] if entities else None),
                "check.dependency",
                "missing_dependency_revision",
                Resolution.REFRESH,
                required_by,
                scope,
            ))
    if kind == "check":
        check_identity = str(record.get("check_identity") or _check_identity(record) or "") or None
        revision = _revision_value(record)
        dependency_fingerprint = _dependency_fingerprint(record)
        result = [replace(
            item,
            check_identity=check_identity,
            revision=revision,
            dependency_fingerprint=dependency_fingerprint,
        ) for item in result]
    return _unique_unknown_drafts(result)


def _is_executed_check(record: Receipt) -> bool:
    """Distinguish an executed runner receipt from a host fact projection."""
    raw_kind = str(record.get("kind") or record.get("type") or "").casefold()
    return (
        raw_kind in {"test", "check", "verification", "test_subprocess_completed",
                     "test_execution", "run_tests"}
        or _host_execution_id(record) is not None
        or ("exit_code" in record and "selectors" in record)
    )


def _upsert_assertion(
    assertions: list[CatalogAssertion],
    *,
    draft: _AssertionDraft,
    receipt_id: str,
    sequence: int,
    record: Receipt,
    revisions: Mapping[str, RevisionBinding],
) -> tuple[CatalogAssertion, list[CatalogAssertion]]:
    state, stale_reason = _draft_state(draft, revisions)
    assertion_id = "assertion:" + _digest({
        "subject": draft.subject.entity_id,
        "predicate": draft.predicate,
        "value": draft.value,
        "revision": draft.revision,
        "dependencies": [item.as_dict() for item in draft.dependencies],
    })[:24]
    candidate = CatalogAssertion(
        assertion_id=assertion_id,
        subject=draft.subject,
        predicate=draft.predicate,
        value=copy.deepcopy(draft.value),
        revision=draft.revision,
        dependencies=draft.dependencies,
        receipt_refs=(receipt_id,),
        source_span_refs=draft.source_span_refs,
        observed_at=copy.deepcopy(draft.observed_at),
        admitted_seq=sequence,
        state=state,
        availability=draft.availability,
        stale_reason=stale_reason,
    )
    for index, existing in enumerate(assertions):
        if existing.assertion_id == assertion_id:
            merged = replace(
                existing,
                receipt_refs=_append_unique(existing.receipt_refs, receipt_id),
                source_span_refs=_append_unique(existing.source_span_refs, *draft.source_span_refs),
                observed_at=candidate.observed_at if sequence >= existing.admitted_seq else existing.observed_at,
                admitted_seq=max(existing.admitted_seq, sequence),
                state=candidate.state if sequence >= existing.admitted_seq else existing.state,
                availability=candidate.availability if sequence >= existing.admitted_seq else existing.availability,
                stale_reason=candidate.stale_reason if sequence >= existing.admitted_seq else existing.stale_reason,
            )
            assertions[index] = merged
            return merged, assertions

    # Conflicting observations on the same bound subject/predicate are kept,
    # but neither is advertised as current authority.
    conflicts: list[int] = []
    for index, existing in enumerate(assertions):
        if (
            existing.subject.entity_id == candidate.subject.entity_id
            and existing.predicate == candidate.predicate
            and existing.revision == candidate.revision
            and existing.state == AssertionState.CURRENT
            and _canonical_value(existing.value) != _canonical_value(candidate.value)
        ):
            conflicts.append(index)
    for index in conflicts:
        assertions[index] = replace(
            assertions[index],
            state=AssertionState.CONFLICT,
            stale_reason="conflicting_observation",
            invalidated_by=_append_unique(assertions[index].invalidated_by, receipt_id),
        )
    if conflicts:
        candidate = replace(candidate, state=AssertionState.CONFLICT, stale_reason="conflicting_observation")
    assertions.append(candidate)
    return candidate, assertions


def _upsert_unknown(
    unknowns: list[UnknownRecord],
    *,
    draft: _UnknownDraft,
    receipt_id: str,
    sequence: int,
) -> tuple[UnknownRecord, list[UnknownRecord]]:
    unknown_id = "unknown:" + _digest({
        "subject": draft.subject.entity_id if draft.subject else None,
        "predicate": draft.predicate,
        "reason": draft.reason,
        "resolution": draft.resolution.value,
        "blocking_scope": draft.blocking_scope,
        "check_identity": draft.check_identity,
        "revision": draft.revision,
        "dependency_fingerprint": draft.dependency_fingerprint,
    })[:24]
    candidate = UnknownRecord(
        unknown_id,
        draft.subject,
        draft.predicate,
        draft.reason,
        draft.resolution,
        draft.required_by,
        (receipt_id,),
        sequence,
        draft.blocking_scope,
        draft.check_identity,
        draft.revision,
        draft.dependency_fingerprint,
    )
    for index, existing in enumerate(unknowns):
        if existing.unknown_id == unknown_id:
            merged = replace(
                existing,
                receipt_refs=_append_unique(existing.receipt_refs, receipt_id),
                required_by=_append_unique(existing.required_by, *draft.required_by),
                admitted_seq=max(existing.admitted_seq, sequence),
            )
            unknowns[index] = merged
            return merged, unknowns
    unknowns.append(candidate)
    return candidate, unknowns


def _invalidate(
    assertions: list[CatalogAssertion],
    *,
    entity_id: str,
    receipt_id: str,
    reason: str,
) -> tuple[list[CatalogAssertion], tuple[str, ...]]:
    affected: list[str] = []
    for index, assertion in enumerate(assertions):
        if assertion.state not in {AssertionState.CURRENT, AssertionState.CONFLICT}:
            continue
        if not _assertion_matches_entity(assertion, entity_id):
            continue
        assertions[index] = replace(
            assertion,
            state=AssertionState.STALE,
            stale_reason=reason,
            invalidated_by=_append_unique(assertion.invalidated_by, receipt_id),
        )
        affected.append(assertion.assertion_id)
    return assertions, tuple(affected)


def _draft_state(
    draft: _AssertionDraft,
    revisions: Mapping[str, RevisionBinding],
) -> tuple[AssertionState, str | None]:
    if not draft.requires_revision:
        return AssertionState.CURRENT, None
    if draft.revision is None:
        return AssertionState.UNKNOWN, "missing_revision_binding"
    for dependency in draft.dependencies:
        binding = revisions.get(dependency.entity_id)
        if binding is None or binding.revision is None:
            return AssertionState.UNKNOWN, "missing_dependency_revision"
        if dependency.revision != binding.revision:
            return AssertionState.STALE, "revision_not_current"
    return AssertionState.CURRENT, None


def _assertion_matches_entity(assertion: CatalogAssertion, entity_id: str) -> bool:
    return assertion.subject.entity_id == entity_id or any(
        item.entity_id == entity_id for item in assertion.dependencies
    )


def _complete(record: Receipt) -> bool:
    if record.get("complete") is False or record.get("truncated") is True:
        return False
    status = _status(record)
    return status not in {"ERROR", "FAILED", "DENIED", "UNAVAILABLE", "TIMEOUT", "NOT_RUN"}


def _search_incomplete(record: Receipt) -> bool:
    coverage = record.get("coverage")
    if isinstance(coverage, Mapping):
        if coverage.get("complete") is False or coverage.get("truncated") is True:
            return True
    return record.get("complete") is False or record.get("truncated") is True


def _search_has_hits(record: Receipt) -> bool:
    for key in ("hits", "matches", "results"):
        value = record.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return bool(value)
    # Legacy runners sometimes emit only ``status=FOUND``.  It is accepted as
    # a scoped result, never as a global fact: _entities() still binds it to
    # the declared query/root namespace.
    status = _status(record)
    if status == "FOUND" and _complete(record):
        return True
    return False


def _edit_applied(record: Receipt) -> bool:
    explicit = record.get("applied")
    if type(explicit) is bool:
        return explicit
    status = _status(record)
    return status in {"APPLIED", "APPLY", "SUCCESS", "SUCCEEDED", "COMMITTED"}


def _check_settled(record: Receipt) -> bool:
    if type(record.get("settled")) is bool:
        return bool(record["settled"])
    return _status(record) in {
        "PASS", "PASSED", "GREEN", "FAIL", "FAILED", "BEHAVIOR_FAIL", "ERROR",
        "INVALID_CHECK", "ENVIRONMENT", "RESOURCE_LIMIT", "NOT_RUN",
        "UNSUPPORTED", "COMPLETE", "COMPLETED",
    }


def _status(record: Receipt) -> str:
    canonical = record.get("canonical_outcome")
    if canonical not in (None, ""):
        return str(canonical).upper()
    for key in ("result_class", "outcome", "result", "status", "verdict"):
        if record.get(key) not in (None, ""):
            return str(record[key]).upper()
    return "UNKNOWN"


def _resolution(value: object, kind: str) -> Resolution:
    text = str(value or "").upper()
    if text in Resolution.__members__:
        return Resolution[text]
    return {
        "read": Resolution.READ,
        "search": Resolution.SEARCH,
        "check": Resolution.CHECK,
    }.get(kind, Resolution.REFRESH)


def _unknown_subject(item: Mapping[str, object], entities: Sequence[EntityRef]) -> EntityRef | None:
    raw = item.get("subject") or item.get("subject_ref") or item.get("entity_id")
    if isinstance(raw, Mapping):
        entity_id = str(raw.get("entity_id") or raw.get("id") or "").strip()
        if entity_id:
            return EntityRef(entity_id, str(raw.get("kind") or "artifact"), _locator(raw.get("locator")))
    if raw:
        for entity in entities:
            if entity.entity_id == str(raw):
                return entity
        return EntityRef(str(raw), "artifact", _locator(raw))
    return entities[0] if entities else None


def _resolved_unknown_ids(record: Receipt) -> frozenset[str]:
    """Return host-admitted unknown identities cleared by this receipt.

    Resolution is deliberately an explicit receipt field.  A model message,
    confidence claim, or repeated description has no path into this helper.
    """
    raw = record.get("resolves_unknown_ids", record.get("resolved_unknown_ids", ()))
    if not raw and isinstance(record.get("resolution"), Mapping):
        raw = record["resolution"].get("unknown_ids", record["resolution"].get("resolves", ()))
    if not raw and isinstance(record.get("supersession"), Mapping):
        raw = record["supersession"].get("unknown_ids", record["supersession"].get("resolves", ()))
    if isinstance(raw, str):
        raw = (raw,)
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        return frozenset()
    return frozenset(
        str(item).strip() for item in raw
        if str(item).strip()
    )


def _dependency_fingerprint(record: Receipt) -> str | None:
    raw = record.get("dependencies", record.get("depends_on", ()))
    if isinstance(raw, Mapping):
        raw = tuple(raw.values())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return None
    values: list[dict[str, object]] = []
    for item in raw:
        if isinstance(item, Mapping):
            values.append({
                "entity_id": item.get("entity_id") or item.get("id") or item.get("path"),
                "revision": item.get("revision"),
            })
        else:
            values.append({"entity_id": item})
    return _digest(values)[:24] if values else None


def _host_admitted(record: Receipt) -> bool:
    actor = str(record.get("actor", record.get("author", ""))).strip().lower()
    provenance = record.get("provenance")
    if actor in {"model", "assistant", "solver"}:
        return False
    if isinstance(provenance, Mapping) and str(provenance.get("kind", "")).upper() in {
        "MODEL", "MODEL_CLAIM", "ASSISTANT_MESSAGE",
    }:
        return False
    return record.get("host_admitted") is True or record.get("admitted") is True or record.get("kind") in {
        "check", "test", "test_subprocess_completed", "verification",
    }


def _resolved_unknown_ids_for_state(
    unknowns: Sequence[UnknownRecord],
    record: Receipt,
    *,
    kind: str,
    entities: Sequence[EntityRef],
) -> frozenset[str]:
    explicit: set[str] = set()
    if _host_admitted(record):
        requested = _resolved_unknown_ids(record)
        by_id = {item.unknown_id: item for item in unknowns}
        scopes = record.get("resolution_scopes")
        for unknown_id in requested:
            item = by_id.get(unknown_id)
            scoped_record: Receipt = record
            if isinstance(scopes, Mapping) and isinstance(scopes.get(unknown_id), Mapping):
                scoped_record = dict(record)
                scoped_record["resolution_scope"] = scopes[unknown_id]
            if item is not None and _resolution_scope_matches(item, scoped_record):
                explicit.add(unknown_id)
        from kernel_step_contract import _evictfix_enabled
        if (kind == "check" and _evictfix_enabled()
                and _canonical_check_outcome(record) == "ENVIRONMENT"):
            check_identity = _check_identity(record)
            if check_identity:
                explicit.update(
                    item.unknown_id for item in unknowns
                    if item.predicate == "check.result"
                    and item.check_identity == check_identity
                )
    # A later PASS is evidence, not a resolution command. Without an explicit
    # predicate-scoped relation it must leave every prior unknown open.
    return frozenset(explicit)


def _resolution_scope_matches(item: UnknownRecord, record: Receipt) -> bool:
    """Require any supplied resolution scope to match the open predicate facts."""
    raw_scope = record.get("resolution_scope", record.get("resolution"))
    scope = raw_scope if isinstance(raw_scope, Mapping) else record
    predicate = scope.get("predicate")
    if predicate not in (None, "") and str(predicate) != item.predicate:
        return False
    check_identity = scope.get("check_identity", scope.get("check_id"))
    if scope.get("check_identity") in (None, "") and scope.get("check_id") not in (None, ""):
        check_identity = f"check:{scope['check_id']}"
    if check_identity not in (None, "") and item.check_identity not in (None, str(check_identity)):
        return False
    revision = scope.get("revision", scope.get("product_revision"))
    if revision not in (None, "") and item.revision not in (None, _revision_text(revision)):
        # A host-admitted current check may explicitly supersede a stale
        # binding obligation.  The relation is still predicate/check scoped;
        # an ordinary PASS without this flag remains non-resolving.
        if scope.get("supersedes_obsolete") is not True:
            return False
        # W4 / P0.1: cross-revision supersession is legal for any predicate,
        # but only when the resolution names that exact predicate (a host
        # measurement of the same stable key at a newer revision).
        if str(predicate or "") != str(item.predicate):
            return False
    subject = scope.get("subject", scope.get("subject_id", scope.get("entity_id", scope.get("path"))))
    if subject not in (None, "") and item.subject is not None:
        if isinstance(subject, Mapping):
            subject = subject.get("entity_id", subject.get("id", subject.get("path")))
        if str(subject) not in {item.subject.entity_id, item.subject.locator}:
            return False
    dependency = scope.get("dependency_fingerprint")
    if dependency not in (None, "") and item.dependency_fingerprint not in (None, str(dependency)):
        return False
    return True


def _unique_unknown_drafts(values: Sequence[_UnknownDraft]) -> tuple[_UnknownDraft, ...]:
    result: list[_UnknownDraft] = []
    seen: set[tuple[object, ...]] = set()
    for value in values:
        key = (
            value.subject.entity_id if value.subject else None,
            value.predicate,
            value.reason,
            value.resolution,
            value.blocking_scope,
            value.check_identity,
            value.revision,
            value.dependency_fingerprint,
        )
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _compact_record(record: Receipt, keys: Sequence[str]) -> dict[str, object]:
    return {key: _copy(record[key]) for key in keys if key in record}


def _binding_revision(revisions: Mapping[str, RevisionBinding], entity_id: str) -> str | None:
    binding = revisions.get(entity_id)
    return binding.revision if binding else None


def _binding_is_newer(binding: RevisionBinding, sequence: int, receipt_id: str) -> bool:
    """Use receipt ID as a deterministic tie-break for duplicate sequences."""
    return (sequence, receipt_id) > (binding.admitted_seq, binding.receipt_id)


def _dependency_entity_id(value: object, namespace: tuple[tuple[str, str], ...]) -> str:
    """Normalize dependency paths to the same IDs used by read/edit receipts."""
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" in text or text.startswith("entity:"):
        return text
    locator = _locator(text)
    if "/" in locator or "." in locator:
        return _scoped_entity_id("file", locator, namespace)
    return text


def _unique_dependencies(values: Sequence[DependencyRef]) -> tuple[DependencyRef, ...]:
    result: list[DependencyRef] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)


def _strings(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Mapping):
        value = tuple(value.values())
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return (str(value),) if str(value).strip() else ()
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _append_unique(values: Sequence[str], *new_values: str) -> tuple[str, ...]:
    result = list(values)
    for value in new_values:
        if value and value not in result:
            result.append(value)
    return tuple(result)


def _validate_mode(mode: str) -> None:
    if mode not in {"current", "historical"}:
        raise CatalogError("mode must be 'current' or 'historical'")


def _validate_budget(value: int) -> int:
    if type(value) is not int or value < 0:
        raise CatalogError("token_budget must be a non-negative integer")
    return value


def _validate_cursor(value: int, length: int) -> int:
    if type(value) is not int or value < 0:
        raise CatalogError("cursor must be a non-negative integer")
    return min(value, length)


def _item_id(value: object) -> str:
    if isinstance(value, CatalogAssertion):
        return value.assertion_id
    if isinstance(value, CatalogChange):
        return value.receipt_id
    if isinstance(value, UnknownRecord):
        return value.unknown_id
    return _digest(value)[:24]


def _estimate_tokens(value: object) -> int:
    encoded = json.dumps(
        _as_public(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return max(1, math.ceil(len(encoded) / 4))


def _as_public(value: object) -> object:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _as_public(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_public(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return copy.deepcopy(value)


def _canonical_value(value: object) -> str:
    return json.dumps(_as_public(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_value(value).encode("utf-8")).hexdigest()


def _copy(value: object) -> object:
    return copy.deepcopy(value)


__all__ = [
    "AssertionState",
    "Availability",
    "CatalogAssertion",
    "CatalogChange",
    "CatalogError",
    "CatalogIntegrityError",
    "CatalogProjection",
    "CatalogReducer",
    "CatalogState",
    "CATC0_AVAILABLE",
    "ContentHandle",
    "ContentRef",
    "CoverageHandle",
    "DependencyRef",
    "EntityRef",
    "QueryOmission",
    "QueryResult",
    "ReceiptLike",
    "Resolution",
    "RevisionBinding",
    "SCHEMA",
    "UNKNOWN_REVISION",
    "UnknownRecord",
    "UnsupportedReceiptError",
    "apply_receipt",
    "empty_catalog",
    "reduce_receipt",
    "replay",
    "project_receipts",
    "normalize_receipt",
]
