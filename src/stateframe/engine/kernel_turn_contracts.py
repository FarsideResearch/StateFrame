from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

class Unavailable(RuntimeError):
    """Evidence, capability or durable storage unavailable. Never grants authority."""

class Stale(Unavailable):
    """Lease, input, cursor or expected head changed. Reacquire and re-evaluate."""

class Corrupt(Unavailable):
    """Identity, chain or codec failed verification. Quarantine and recover."""

class Context(NamedTuple):
    principal: str
    run: str
    attempt: str
    private_dir: Path
    profile_digest: str

class Head(NamedTuple):
    seq: int
    digest: str

class Basis(NamedTuple):
    parent: str
    live: str
    epoch: int
    journal: Head
    step: int
    authority: str

class Key(NamedTuple):
    family: str
    selector: bytes

class Dep(NamedTuple):
    key: Key
    revision: str

class Ref(NamedTuple):
    codec: str
    sha256: str
    size: int

class Artifact(NamedTuple):
    value: Ref
    deps: tuple[Dep, ...]

class Event(NamedTuple):
    before: Head
    after: Head
    kind: str
    changed: tuple[Key, ...]
    payload: Ref

class Page(NamedTuple):
    data: Artifact
    next_cursor: bytes | None

class Snapshot(NamedTuple):
    root: str
    admission: Ref
    epoch: int

class Workspace(Protocol):
    def admit(self, source: Path, policy: Ref) -> Snapshot:
        """Freeze and validate parent bytes, filters, limits and isolation before send."""
    def pin(self, scope: Ref) -> AbstractContextManager[Snapshot]:
        """Acquire a reconciled live reader fence; release on scope exit."""
    def retain(self, root: str) -> AbstractContextManager[Snapshot]:
        """Lease immutable bytes of an admitted root for a reader; no live-freshness claim."""
    def writing(self, paths: tuple[bytes, ...] | None) -> AbstractContextManager[None]:
        """Invalidate before granting write; exception/opaque effects require reconcile."""
    def changes(self, old: Snapshot, new: Snapshot) -> Iterator[bytes]:
        """Stream changed relative raw paths; both snapshots retained, deterministic order."""
    def read(self, at: Snapshot, path: bytes) -> bytes:
        """Read one admitted regular file from the immutable view with containment checks."""
    def materialize(self, parent: Snapshot, destination: Path) -> None:
        """Create private writable copy at an unused host location or fail before first write."""

class Facts(Protocol):
    def lookup(self, at: Basis, key: Key, cursor: bytes | None, limit: int) -> Page:
        """Return ordered bounded verified rows or certified absence; incomplete raises."""
    def build(self, parent: Snapshot, stop: Callable[[], bool]) -> None:
        """Ingest immutable parent once; publish complete keyed views and honor cancellation."""
    def current(self, at: Basis, deps: tuple[Dep, ...]) -> bool:
        """Validate scoped membership, absence and versions for all reads at this basis."""

class Panels(Protocol):
    def fold(self, previous: Artifact | None, at: Basis, events: Iterator[Event]) -> Artifact:
        """Stage ordered slot manifest through at.journal; caller publishes cursor and refs."""
    def render(self, manifest: Artifact, layout: Ref) -> bytes:
        """Join ordered, encoding-layer-specific fragments into the selected frame payload."""

class Proofs(Protocol):
    def guard(self, at: Basis, use: Literal["menu", "send", "act"], inputs: Ref) -> Artifact:
        """Return current decision and blockers for this use; unavailable cannot authorize."""
    def delivery(self, at: Basis, candidate: Ref, guard: Artifact) -> Artifact:
        """Return complete legacy verdict/reasons on exact immutable candidate and evidence."""

class Launch(Protocol):
    def bootstrap(self, source: Path, policy: Ref, task: Ref) -> tuple[Snapshot, Artifact]:
        """Return admitted parent and deterministic minimal frame after all mandatory gates."""
    def start_build(self, parent: Snapshot) -> AbstractContextManager[None]:
        """Queue optional ingestion on entry; cancel and release on exit, including errors."""

class MeterId(NamedTuple):
    route: str
    model: str
    artifacts: str
    behavior: str

class Count(NamedTuple):
    meter: MeterId
    input_sha256: str
    tokens: int

class Meter(Protocol):
    def count(self, meter: MeterId, content: str, deadline: float | None) -> Count:
        """Count exactly the supplied text locally; return raw token count before overhead."""

class Candidate(NamedTuple):
    wire: bytes
    inputs: Ref
    guard: Artifact
    meter: MeterId

class Qualified(NamedTuple):
    basis: Basis
    wire: bytes
    certificate: Artifact
    count: Count

class Qualification(Protocol):
    def qualify(self, at: Basis, candidate: Candidate, deadline: float | None) -> Qualified:
        """Bind final provider bytes, action/source/budget proof and exact measurement."""
    def handoff(self, q: Qualified, submit: Callable[[bytes], None]) -> None:
        """Recheck under dispatch fence and invoke submit once with sealed bytes."""

class Commit(NamedTuple):
    basis: Basis
    previous: Ref | None
    boundary_id: str
    roots: tuple[tuple[str, Ref], ...]

class Publication(Protocol):
    def transaction(self) -> AbstractContextManager[None]:
        """Hold staging/reader pins across operations; exit drops only transient references."""
    def stage(self, codec: str, data: bytes) -> Ref:
        """Write verified immutable data in this namespace, with a staging pin."""
    def load(self, ref: Ref) -> bytes:
        """Resolve under current namespace; verify codec, size and digest or raise."""
    def head(self) -> Ref | None:
        """Return last complete run manifest, or None only for proven first launch."""
    def commit(self, proposed: Commit) -> Ref:
        """Durably CAS previous head and referenced graph; same boundary/content is idempotent."""
    def suffix(self, after: Head, through: Head) -> Iterator[Event]:
        """Iterate verified contiguous committed records in (after, through], including amendments."""

class Overlap(Protocol):
    def offer(self, at: Basis, key: Key, work: Callable[[], Artifact]) -> None:
        """Schedule optional pure pinned work if capacity permits; never wait for space."""
    def take(self, at: Basis, key: Key) -> Artifact | None:
        """Try completed output, revalidate all dependencies, return None on miss or stale."""
    def cancel(self) -> None:
        """Invalidate outstanding tickets; late completions have no publication authority."""

class Live(Protocol):
    def observing(self, scope: Ref) -> AbstractContextManager[Basis]:
        """Fence workspace and host authority together up to the submission linearization point."""

class Ports(NamedTuple):
    workspace: Workspace
    panels: Panels
    proofs: Proofs
    launch: Launch
    facts: Facts
    qualification: Qualification
    publication: Publication
    overlap: Overlap
    meter: Meter
