"""C3 bridge from the pure catalog projection to existing authority owners.

The bridge carries identity and generation vectors only.  It does not parse
source, promote claims, or decide whether a behavioral result is true.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:  # Flat v46 module loading.
    from kernel_catalog import CatalogState, empty_catalog, reduce_receipt
    from kernel_invalidation import InvalidationResult, LedgerIndex
except ImportError:  # pragma: no cover - package import fallback.
    from .kernel_catalog import CatalogState, empty_catalog, reduce_receipt
    from .kernel_invalidation import InvalidationResult, LedgerIndex


@dataclass(frozen=True, slots=True)
class CatalogBridgeResult:
    """Host evidence emitted by one catalog-to-owner synchronization."""

    state: CatalogState
    changed_dependency_ids: tuple[str, ...] = ()
    stale_assertion_ids: tuple[str, ...] = ()
    invalidation_results: tuple[InvalidationResult, ...] = ()
    crosssession_assessments: tuple[Mapping[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "watermark": self.state.watermark,
            "changed_dependency_ids": list(self.changed_dependency_ids),
            "stale_assertion_ids": list(self.stale_assertion_ids),
            "invalidation_results": [item.to_dict() for item in self.invalidation_results],
            "crosssession_assessments": [dict(item) for item in self.crosssession_assessments],
        }


class CatalogClaimBridge:
    """Synchronize C1 applicability with DEPGRAPH and adjacent owners."""

    def __init__(
        self,
        ledger: LedgerIndex,
        *,
        decision_graph: object | None = None,
        crosssession: object | None = None,
        state: CatalogState | None = None,
    ) -> None:
        if not isinstance(ledger, LedgerIndex):
            raise TypeError("ledger must be a LedgerIndex")
        self.ledger = ledger
        self.decision_graph = decision_graph
        self.crosssession = crosssession
        self.state = state or empty_catalog()

    def apply(self, receipt: object, *, turn: int = 0) -> CatalogBridgeResult:
        """Reduce one host receipt and publish its applicability effects."""

        return self.sync(reduce_receipt(self.state, receipt), turn=turn)

    def sync(self, state: CatalogState, *, turn: int = 0) -> CatalogBridgeResult:
        """Reconcile catalog generations and current assertion applicability."""

        if not isinstance(state, CatalogState):
            raise TypeError("state must be a CatalogState")
        invalidations: list[InvalidationResult] = []
        changed_dependency_ids: list[str] = []
        current_generations = {
            binding.entity_id: (binding.revision or "UNKNOWN")
            for binding in state.revision_bindings
        }
        for dependency_id, generation in sorted(current_generations.items()):
            previous = self.ledger.dependency_generation(dependency_id)
            result = self.ledger.observe_dependency_generation(
                dependency_id,
                generation,
                turn=turn,
            )
            invalidations.append(result)
            if previous is not None and previous.generation != generation:
                changed_dependency_ids.append(dependency_id)

        stale_assertion_ids: list[str] = []
        for assertion in state.assertions:
            applicability = assertion.state.value
            if applicability != "CURRENT":
                stale_assertion_ids.append(assertion.assertion_id)
            dependencies = tuple(assertion.dependencies)
            paths = tuple(
                dependency.entity_id
                for dependency in dependencies
                if getattr(dependency, "entity_id", "")
            )
            self.ledger.add_catalog_claim(
                assertion.assertion_id,
                what=f"{assertion.predicate}: {assertion.subject.entity_id}",
                evidence_refs=assertion.receipt_refs,
                paths=paths,
                workspace_revision=assertion.admitted_seq,
                dependency_generations=dependencies,
                applicability=applicability,
            )
            self._publish_decision_applicability(assertion.assertion_id, applicability, assertion.stale_reason)

        assessments = self._refresh_crosssession(current_generations, state)
        self.state = state
        return CatalogBridgeResult(
            state=state,
            changed_dependency_ids=tuple(dict.fromkeys(changed_dependency_ids)),
            stale_assertion_ids=tuple(dict.fromkeys(stale_assertion_ids)),
            invalidation_results=tuple(invalidations),
            crosssession_assessments=assessments,
        )

    def _publish_decision_applicability(
        self,
        assertion_id: str,
        applicability: str,
        reason: str | None,
    ) -> None:
        mark_item = getattr(self.decision_graph, "mark_item", None)
        if not callable(mark_item):
            return
        current = applicability == "CURRENT"
        try:
            mark_item(
                assertion_id,
                state="CURRENT" if current else "STALE",
                reason="" if current else (reason or "catalog assertion is not current"),
            )
        except (TypeError, ValueError):
            # A projection cannot turn an owner adapter failure into authority.
            return

    @staticmethod
    def _versions_for_crosssession(
        state: CatalogState,
        generations: Mapping[str, str | None],
    ) -> dict[str, str]:
        versions = {
            dependency_id: generation
            for dependency_id, generation in generations.items()
            if generation is not None
        }
        for assertion in state.assertions:
            generation = generations.get(assertion.subject.entity_id)
            locator = getattr(assertion.subject, "locator", "")
            if generation is not None and locator:
                versions[str(locator).replace("\\", "/")] = generation
        for dependency_id, generation in tuple(versions.items()):
            if dependency_id.startswith("file:"):
                versions.setdefault(dependency_id[5:], generation)
        return versions

    def _refresh_crosssession(
        self,
        generations: Mapping[str, str | None],
        state: CatalogState,
    ) -> tuple[Mapping[str, object], ...]:
        memory = self.crosssession
        latest = getattr(memory, "latest", None)
        revalidate = getattr(memory, "revalidate", None)
        if not callable(latest) or not callable(revalidate):
            return ()
        versions = self._versions_for_crosssession(state, generations)
        assessments: list[Mapping[str, object]] = []
        for claim in latest():
            memory_id = getattr(claim, "memory_id", "")
            if not memory_id:
                continue
            assessment = revalidate(memory_id, versions=versions)
            as_dict = getattr(assessment, "as_dict", None)
            if callable(as_dict):
                value = as_dict()
                if isinstance(value, Mapping):
                    assessments.append(dict(value))
        return tuple(assessments)


def reduce_sciledger_dependency_change(
    state: object,
    *,
    dependency_kind: str,
    generation: str,
    event_id: str,
    domain_id: str = "catalog",
    claim_ids: Sequence[str] = (),
    previous_generation: str | None = None,
    reason: str = "catalog dependency changed",
    sequence: int | None = None,
    previous_event_hash: str | None = None,
    expected_basis_digest: str | None = None,
) -> object:
    """Feed a catalog generation change through sciledger's pure reducer."""

    from tools.sciledger.reducer import GENESIS_HASH, reduce_event
    from tools.sciledger.schema import Event, EventType, OwnerRole

    state_sequence = getattr(state, "sequence", 0)
    state_previous_hash = getattr(state, "previous_event_hash", GENESIS_HASH)
    payload: dict[str, object] = {
        "kind": dependency_kind,
        "digest": generation,
        "claim_ids": list(claim_ids),
        "reason": reason,
    }
    if previous_generation is not None:
        payload["previous_digest"] = previous_generation
    event = Event.create(
        EventType.DEPENDENCY_CHANGED,
        payload,
        event_id=event_id,
        domain_id=domain_id,
        sequence=sequence if sequence is not None else int(state_sequence) + 1,
        previous_event_hash=previous_event_hash or state_previous_hash,
        actor=OwnerRole.HOST,
        expected_basis_digest=expected_basis_digest,
    )
    return reduce_event(state, event)


def reduce_sciledger_evidence_loss(
    state: object,
    *,
    evidence_ids: Sequence[str],
    reason: str,
    event_id: str,
    domain_id: str = "catalog",
    claim_ids: Sequence[str] = (),
    sequence: int | None = None,
    previous_event_hash: str | None = None,
    expected_basis_digest: str | None = None,
) -> object:
    """Feed evidence loss through sciledger's existing retest/decision effects."""

    from tools.sciledger.reducer import GENESIS_HASH, reduce_event
    from tools.sciledger.schema import Event, EventType, OwnerRole

    state_sequence = getattr(state, "sequence", 0)
    state_previous_hash = getattr(state, "previous_event_hash", GENESIS_HASH)
    payload = {
        "evidence_ids": list(evidence_ids),
        "claim_ids": list(claim_ids),
        "reason": reason,
    }
    event = Event.create(
        EventType.EVIDENCE_INVALIDATED,
        payload,
        event_id=event_id,
        domain_id=domain_id,
        sequence=sequence if sequence is not None else int(state_sequence) + 1,
        previous_event_hash=previous_event_hash or state_previous_hash,
        actor=OwnerRole.HOST,
        expected_basis_digest=expected_basis_digest,
    )
    return reduce_event(state, event)


__all__ = [
    "CatalogBridgeResult",
    "CatalogClaimBridge",
    "reduce_sciledger_dependency_change",
    "reduce_sciledger_evidence_loss",
]
