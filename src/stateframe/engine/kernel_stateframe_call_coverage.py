"""Journal-owned call coverage; only a bounded frontier crosses the wire.

Every distinct folded ID is appended once. Completed projections carry a proof
of their index prefix and row chain; intermediate folds are never journaled.
The journal validates storage lineage; this index validates coverage lineage.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
import threading
import uuid
import weakref

CONTRACT = "journal-call-coverage/1"
CONTEXT_KEY = "_call_coverage_index"
EVENT = "framefit5_call_coverage"
FRONTIER_COUNT = 64
FRONTIER_ID_BYTES = 256
FRONTIER_BYTES = 4096


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _error(message: str) -> None:
    from kernel_stateframe_framefit5 import FrameFit5Error

    raise FrameFit5Error(message)


def _frontier(previous: list[str], added: list[str]) -> list[str]:
    # Oversized identities stay exact in the host index, never as ambiguous
    # prefixes on the wire. A single hostile ID cannot grow the frontier.
    result = (previous + [item for item in added
                          if len(item.encode("utf-8")) <= FRONTIER_ID_BYTES])[-FRONTIER_COUNT:]
    while len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode()) > FRONTIER_BYTES:
        result.pop(0)
    return result


@dataclass(frozen=True, slots=True)
class _CoverageSnapshot:
    """One complete, immutable view of the journal-owned coverage prefix."""

    sequence: int
    ids: tuple[str, ...]
    members: frozenset[str]
    chains: tuple[str, ...]
    frontier: tuple[str, ...]
    proofs: frozenset[str]


def _journal_guard(journal: object):
    """Serialize binding with journal replay when the journal exposes a lock."""
    ownership = getattr(journal, "ownership", None)
    if callable(ownership):
        return ownership()
    lock = getattr(journal, "_lock", None)
    return lock if hasattr(lock, "__enter__") else nullcontext()


class CoverageIndex:
    """One index per journal owner, shared across detached action contexts.

    Multi-field consistency requires reading ``snapshot``. Legacy per-field
    reads (sequence, ids, members, chains, frontier, proofs) are individually
    consistent, but are not jointly consistent.
    """

    def __init__(self, journal: object, *, _register: bool = True):
        self._lock = threading.RLock()
        with self._lock:
            self.journal = journal
            run_id = getattr(journal, "run_id", None)
            journal_path = getattr(journal, "path", None)
            self.context_handle = (
                _sha(["coverage-index", run_id, str(journal_path)])
                if isinstance(run_id, str) and run_id and journal_path is not None
                else uuid.uuid4().hex
            )
            self.sequence = 0
            self.ids: list[str] = []
            self.members: set[str] = set()
            self.chains: list[str] = [_sha([])]
            self.frontier: list[str] = []
            self.proofs: set[str] = set()
        if _register:
            self._publish_context_index()

    @property
    def snapshot(self) -> _CoverageSnapshot:
        """Build a consistent immutable view on demand."""
        with self._lock:
            return _CoverageSnapshot(
                sequence=self.sequence,
                ids=tuple(self.ids),
                members=frozenset(self.members),
                chains=tuple(self.chains),
                frontier=tuple(self.frontier),
                proofs=frozenset(self.proofs),
            )

    @property
    def _snapshot(self) -> _CoverageSnapshot | None:
        """Expose snapshots only for the journal's published index."""
        if getattr(self.journal, "_framefit_call_coverage_index", None) is not self:
            return None
        return self.snapshot

    def _publish_context_index(self) -> None:
        """Register only after the index is fully initialized and journal-bound."""
        with _CONTEXT_INDEX_LOCK:
            _CONTEXT_INDEXES[self.context_handle] = self

    def _replace_state_from(self, staged: CoverageIndex) -> None:
        """Atomically adopt private replay state without copying its containers."""
        if not isinstance(staged, CoverageIndex) or staged.journal is not self.journal:
            _error("call coverage snapshot belongs to a different journal")
        with self._lock:
            self.sequence = staged.sequence
            self.ids = staged.ids
            self.members = staged.members
            self.chains = staged.chains
            self.frontier = staged.frontier
            self.proofs = staged.proofs

    def __deepcopy__(self, memo: dict) -> CoverageIndex:
        # Host authority is not model state, and the journal owns locks/leases.
        return self

    def sync(self) -> None:
        while True:
            with self._lock:
                previous_sequence = self.sequence
            cursor = self.journal.receipt_cursor()
            if cursor.sequence < previous_sequence:
                _error("call coverage journal cursor moved backwards")
            borrow_records = getattr(self.journal, "_borrow_records_since", None)
            records = (borrow_records(previous_sequence) if callable(borrow_records)
                       else self.journal.cached_records_since(previous_sequence))
            with self._lock:
                if self.sequence != previous_sequence:
                    continue
                for record in records:
                    sequence = record["sequence"]
                    if sequence != self.sequence + 1:
                        _error("call coverage journal sequence gap")
                    if record.get("kind") == EVENT:
                        self._accept(record.get("coverage_proof"), sequence=sequence)
                    else:
                        self._advance_sequence(sequence)
                return

    def _validate_proof(self, proof: object) -> tuple[list[str], list[str], list[str], dict[str, object]]:
        if not isinstance(proof, dict) or set(proof) != {
                "previous_count", "previous_chain", "added", "digest"}:
            _error("call coverage journal proof is malformed")
        digest, added = proof["digest"], proof["added"]
        from kernel_wire_identity import wire_run_id
        if (not isinstance(digest, dict) or digest.get("coverage_contract") != CONTRACT
                or digest.get("coverage_run_id") not in (wire_run_id(self.journal), self.journal.run_id)
                or not isinstance(added, list)
                or any(not isinstance(item, str) or not item for item in added)
                or len(set(added)) != len(added) or self.members.intersection(added)
                or type(proof["previous_count"]) is not int
                or proof["previous_count"] != len(self.ids)
                or proof["previous_chain"] != self.chains[-1]):
            _error("call coverage journal ancestry is malformed")
        chain, chains = self.chains[-1], []
        for call_id in added:
            chain = _sha([chain, call_id])
            chains.append(chain)
        frontier = _frontier(self.frontier, added)
        if (type(digest.get("call_count")) is not int
                or digest["call_count"] != len(self.ids) + len(added)
                or digest.get("call_ids_sha256") != chain
                # The minimum wire projection may omit the display frontier.
                # Both forms still bind the entire index count/chain and must
                # match a durable proof of this exact digest before validation.
                or digest.get("call_ids") not in (sorted(frontier), [])):
            _error("call coverage journal count or chain mismatch")
        from kernel_stateframe_capacity import validate_read_ranges

        validate_read_ranges({"action_ledger": {"reads": [], "history_digest": digest}}, [])
        return added, chains, frontier, digest

    def _accept(self, proof: object, *, sequence: int | None = None) -> None:
        with self._lock:
            added, chains, frontier, digest = self._validate_proof(proof)
            if sequence is not None and (
                    type(sequence) is not int or sequence != self.sequence + 1):
                _error("call coverage journal sequence gap")
            self.ids.extend(added)
            self.members.update(added)
            self.chains.extend(chains)
            self.frontier = frontier
            self.proofs.add(_sha(digest))
            if sequence is not None:
                self.sequence = sequence

    def _advance_sequence(self, sequence: int) -> None:
        with self._lock:
            if type(sequence) is not int or sequence != self.sequence + 1:
                _error("call coverage journal sequence gap")
            self.sequence = sequence

    def covered_ids(self, digest: Mapping[str, object]) -> set[str]:
        self.sync()
        with self._lock:
            if _sha(digest) not in self.proofs:
                _error("call coverage digest has no matching host proof")
            count = digest["call_count"]
            result = set(self.ids[:count])
            if (len(result) != count or self.chains[count] != digest["call_ids_sha256"]
                    or not result.issubset(self.members)):
                _error("call coverage host inventory count mismatch")
            return result

    def commit(self, proof: dict[str, object]) -> None:
        self.sync()
        with self._lock:
            if _sha(proof["digest"]) in self.proofs:
                return
            if (proof["previous_count"] != len(self.ids)
                    or proof["previous_chain"] != self.chains[-1]):
                _error("call coverage index changed during detached fitting")
            self._validate_proof(proof)
        # Durability precedes frame publication. A failed append leaves the
        # detached candidate unadopted and never authorizes its wire proof.
        self.journal.emit(EVENT, coverage_contract=CONTRACT, coverage_proof=proof)
        self.sync()


_CONTEXT_INDEXES: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
_CONTEXT_INDEX_LOCK = threading.RLock()


def bind_index(context: dict[str, object], journal: object) -> dict[str, object]:
    """Attach a JSON-safe locator; the actual host index stays off the wire."""
    if all(callable(getattr(journal, name, None))
           for name in ("cached_records_since", "receipt_cursor", "emit")):
        index = getattr(journal, "_framefit_call_coverage_index", None)
        already_bound = (isinstance(index, CoverageIndex) and index.journal is journal
                         and _CONTEXT_INDEXES.get(index.context_handle) is index)
        if not already_bound:
            with _journal_guard(journal):
                index = getattr(journal, "_framefit_call_coverage_index", None)
                if index is None:
                    index = CoverageIndex(journal, _register=False)
                if not isinstance(index, CoverageIndex) or index.journal is not journal:
                    _error("call coverage index belongs to a different journal")
                with index._lock:
                    if getattr(journal, "_framefit_call_coverage_index", None) is None:
                        setattr(journal, "_framefit_call_coverage_index", index)
                    index._publish_context_index()
        context[CONTEXT_KEY] = index.context_handle
        history = context.get("journal_receipt_history")
        if (isinstance(history, dict)
                and history.get("schema") == "framefit-receipt-history/1"):
            history.pop("projection_sha256", None)
            history["projection_sha256"] = _sha({
                "journal_receipt_count": context.get("journal_receipt_count"),
                "journal_receipt_records": context.get("journal_receipt_records"),
                "journal_receipt_history": history,
                CONTEXT_KEY: context[CONTEXT_KEY],
            })
    return context


def context_index(context: object) -> CoverageIndex | None:
    handle = context.get(CONTEXT_KEY) if isinstance(context, Mapping) else None
    if isinstance(handle, CoverageIndex):
        return handle
    # Writers publish fully initialized objects under the GIL-protected registry
    # lock; WeakValueDictionary.get() is safe for these lock-free reads.
    index = _CONTEXT_INDEXES.get(handle) if isinstance(handle, str) else None
    return index if isinstance(index, CoverageIndex) else None


class CoverageSession:
    """Detached candidate fold transaction; no durable writes until commit."""

    def __init__(self, index: CoverageIndex, digest: Mapping[str, object] | None):
        self.index = index
        self.legacy_ids: set[str] = set()
        self.previous_ids: set[str] = set()
        if digest is not None:
            if "coverage_contract" in digest:
                if digest["coverage_contract"] != CONTRACT:
                    _error("unknown call coverage contract")
                self.previous_ids = index.covered_ids(digest)
            else:
                self.legacy_ids = set(digest["call_ids"])
        index.sync()
        # Coverage is cumulative across committed projections, even when a
        # later request rebuilds rows or restores detail for a previously folded
        # ID. Row hashes still describe that candidate's exact C2 fold sequence.
        with index._lock:
            self.ids = set(index.members)
            self.chain = index.chains[-1]
            self.frontier = list(index.frontier)
            self.previous_count = len(index.ids)
            self.previous_chain = self.chain
        self.added: list[str] = []
        self.digest: dict[str, object] | None = None

    @property
    def covered_ids(self) -> set[str]:
        """Only the prefix represented by this candidate, not newer host work."""
        return self.ids if self.digest is not None else self.previous_ids

    def fold(self, digest: dict[str, object], ids: set[str]) -> None:
        added = sorted((ids | self.legacy_ids) - self.ids)
        self.legacy_ids.clear()
        for call_id in added:
            self.chain = _sha([self.chain, call_id])
        self.ids.update(added)
        self.added.extend(added)
        self.frontier = _frontier(self.frontier, added)
        self._project_digest(digest, include_frontier=True)

    def _project_digest(self, digest: dict[str, object], *, include_frontier: bool) -> None:
        from kernel_wire_identity import wire_run_id
        digest.update(coverage_contract=CONTRACT,
                       call_ids=sorted(self.frontier) if include_frontier else [],
                       coverage_run_id=wire_run_id(self.index.journal),
                      call_count=len(self.ids), call_ids_sha256=self.chain)
        self.digest = copy.deepcopy(digest)

    def minimize_frontier(self, digest: dict[str, object]) -> bool:
        """One bounded last-resort reduction; complete coverage stays on host."""
        if digest.get("coverage_contract") != CONTRACT or not digest.get("call_ids"):
            return False
        self._project_digest(digest, include_frontier=False)
        return True

    def commit(self) -> None:
        if self.digest is not None:
            self.index.commit({"previous_count": self.previous_count,
                               "previous_chain": self.previous_chain,
                               "added": list(self.added), "digest": self.digest})
