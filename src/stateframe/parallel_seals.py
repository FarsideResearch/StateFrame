"""Seal operation identity, fresh owner proofs, and artifact-only helper entry points."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import secrets
import time
from typing import TYPE_CHECKING

from .parallel_journal import Ticket
if TYPE_CHECKING:
    from .engine.kernel_parallel_records import HostRequest
    from .parallel_host import _Worker


def operation_id(request: HostRequest) -> str:
    # Attempt/sequence identify delivery, not the operation. Resume can redeliver it.
    identity = {"call": request.call_id, "op": request.op, "scope": request.scope,
                "payload": request.payload}
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class QuiescenceProof:
    incarnation: int
    generation: int
    nonce: str
    helper_epoch: int
    scan_started_ns: int
    scan_finished_ns: int


@dataclass
class SealOperation:
    worker: _Worker
    request: HostRequest
    job: Path
    result: Path
    generation: int
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    fence_ns: int = field(default_factory=time.monotonic_ns)
    phase: str = "PENDING"
    ticket: Ticket | None = None
    helper: int = 0
    helper_epoch: int = 0
    helper_losses: int = 0  # retained across fresh tickets, nonces and helper epochs
    retry_at: float = 0.0
    staged: dict[str, object] | None = None
    proof: QuiescenceProof | None = None
    superseded: list[str] = field(default_factory=list)  # nonces whose stage files await removal

    def accepts(self, proof: QuiescenceProof, epoch: int) -> bool:
        return (proof.incarnation == self.worker.incarnation
                and proof.generation == self.generation == self.worker.generation
                and proof.nonce == self.nonce and proof.helper_epoch == epoch
                and proof.scan_started_ns >= self.fence_ns
                and proof.scan_finished_ns >= proof.scan_started_ns)

    def stage_for(self, nonce: str) -> Path:
        return self.job.with_name(self.job.stem + ".stage-" + nonce + ".json")

    def invalidate(self) -> None:
        self.superseded.append(self.nonce)  # its stage file is dead once no helper holds it
        self.nonce = secrets.token_hex(16)
        self.fence_ns = time.monotonic_ns()
        self.phase, self.staged, self.proof = "PENDING", None, None

    def discard_superseded_stages(self) -> None:
        """Remove stage files of invalidated nonces. Call only while no helper ticket is
        outstanding for this operation: a helper may still be writing an older stage."""
        while self.superseded:
            nonce = self.superseded[-1]
            try:
                self.stage_for(nonce).unlink(missing_ok=True)
            except OSError:
                return  # retry on a later pass; never raise into the lifecycle lane
            self.superseded.pop()


def artifact_operation(kind: str, payload: dict[str, object]) -> dict[str, object]:
    """No process handles or inspection authority enter the helper."""
    from .engine.kernel_parallel_records import HostRequest, Proposal
    from .parallel_host import (ParallelHostError, ParallelSupervisor, _WorkerIndex,
                                _inside, _read_json, _regular_bytes, _write_json)

    request = HostRequest.from_json(payload["request"])
    call_root = Path(str(payload["call_root"]))
    token = payload["token"]
    if kind == "seal_build":
        supervisor = ParallelSupervisor.__new__(ParallelSupervisor)
        supervisor._journal = None
        supervisor._workers = _WorkerIndex()
        worker = supervisor._load_worker(request.call_id, str(request.payload["unit_id"]),
                                         request.payload["incarnation"], call_root)
        if worker is None or worker.state != "EXITED" or worker.terminal != "SUBMITTED":
            raise ParallelHostError("seal helper lacks a durable exited worker")
        stage = _inside(Path(str(payload["stage"])), call_root / "host" / "seals", "seal stage")
        result = supervisor._build_seal_artifacts(request, call_root, worker, stage=stage)
        return {"token": token, "staged": result}
    if kind != "seal_publish":
        raise ParallelHostError("unknown seal helper operation")
    staged = payload["staged"]
    stage = _inside(Path(staged["stage_path"]), call_root / "host" / "seals", "seal stage")
    proposal = Proposal.from_json(_read_json(stage, "staged seal proposal"))
    if (proposal.digest() != staged["proposal_digest"] or proposal.call_id != request.call_id
            or proposal.unit_id != request.payload["unit_id"]
            or proposal.incarnation != request.payload["incarnation"]):
        raise ParallelHostError("staged seal identity or digest mismatch")
    target = _inside(Path(staged["proposal_path"]), call_root / "proposals", "seal publication")
    expected = call_root / "proposals" / proposal.unit_id / "proposal.json"
    if target != expected:
        raise ParallelHostError("staged seal publication path mismatch")
    # Reauthenticate staged references before publishing. Build retained every scope/receipt check.
    for entry in proposal.entries:
        blob = _regular_bytes(target.parent / "blobs" / str(entry.after_sha256),
                              maximum=16 << 20, label="staged seal blob")
        if hashlib.sha256(blob).hexdigest() != entry.after_sha256:
            raise ParallelHostError("staged seal blob digest mismatch")
    _write_json(target, proposal.to_json())
    _write_json(target.parent / "digest.json", {"proposal_digest": proposal.digest()})
    return {"token": token, "payload": {"proposal_path": str(target), "proposal_digest": proposal.digest()}}
