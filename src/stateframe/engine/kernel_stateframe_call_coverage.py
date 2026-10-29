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


class CoverageIndex:
    """One index per journal owner, shared across detached action contexts."""

    def __init__(self, journal: object):
        self.journal = journal
        run_id = getattr(journal, "run_id", None)
        journal_path = getattr(journal, "path", None)
        self.context_handle = (
            _sha(["coverage-index", run_id, str(journal_path)])
            if isinstance(run_id, str) and run_id and journal_path is not None
            else uuid.uuid4().hex
        )
        _CONTEXT_INDEXES[self.context_handle] = self
        self.sequence = 0
        self.ids: list[str] = []
        self.members: set[str] = set()
        self.chains: list[str] = [_sha([])]
        self.frontier: list[str] = []
        self.proofs: set[str] = set()

    def __deepcopy__(self, memo: dict) -> CoverageIndex:
        # Host authority is not model state, and the journal owns locks/leases.
        return self

    def sync(self) -> None:
        cursor = self.journal.receipt_cursor()
        if cursor.sequence < self.sequence:
            _error("call coverage journal cursor moved backwards")
        for record in self.journal.cached_records_since(self.sequence):
            if record["sequence"] != self.sequence + 1:
                _error("call coverage journal sequence gap")
            if record.get("kind") == EVENT:
                self._accept(record.get("coverage_proof"))
            self.sequence = record["sequence"]

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

    def _accept(self, proof: object) -> None:
        added, chains, frontier, digest = self._validate_proof(proof)
        self.ids.extend(added)
        self.members.update(added)
        self.chains.extend(chains)
        self.frontier = frontier
        self.proofs.add(_sha(digest))

    def covered_ids(self, digest: Mapping[str, object]) -> set[str]:
        self.sync()
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


def bind_index(context: dict[str, object], journal: object) -> dict[str, object]:
    """Attach a JSON-safe locator; the actual host index stays off the wire."""
    if all(callable(getattr(journal, name, None))
           for name in ("cached_records_since", "receipt_cursor", "emit")):
        index = getattr(journal, "_framefit_call_coverage_index", None)
        if index is None:
            index = CoverageIndex(journal)
            setattr(journal, "_framefit_call_coverage_index", index)
        if not isinstance(index, CoverageIndex) or index.journal is not journal:
            _error("call coverage index belongs to a different journal")
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
