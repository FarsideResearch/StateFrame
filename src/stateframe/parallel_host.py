"""File-channel subframe supervisor for one StateFrame parent run."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import collections
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import weakref

from .engine.kernel_parallel_records import (CheckReceipt, HostAnswer, HostRequest, ParallelRecordError, Proposal,
                                              ProposalEntry, UnitSpec, canonical_json)
from .engine.kernel_parallel_txn import TxnStoreError
from .engine.kernel_exec_tracker import (ContainmentUnverified, ProcessTable, _boot_id,
                                         live_group_rows, load_tolerant_seconds)
from . import outcomes, workcopy
from .parallel_journal import Journal, JournalBusy, JournalFault, JournalProtocolError, Ticket
from .parallel_channel import ChannelWrites, read_synced_record
from .parallel_seals import QuiescenceProof, SealOperation, operation_id
from .parallel_runtime import ProtocolErrorLog, channel_loop, lifecycle_loop
from .parallel_resources import reserve_preparation
from .parallel_worker import (SnapshotManifest, TickProcessTable, WorkerHandle, WorkerLaunchError,
                               WorkerPreparationError, kernel_start_identity, mac_process_argv,
                               cleanup_seconds, launch_worker, prepare_relaunch,
                               prepare_worker, read_snapshot_manifest)


_UNIT_ID = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
_CALL_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HOST_STATE_SCHEMA = "stateframe-parallel-host-worker/1"
_CHANNEL_META_SCHEMA = "stateframe-parallel-channel-meta/1"
_CHANNEL_RECORD_SCHEMA = "stateframe-parallel-channel-record/1"
_HOST_ACK_SCHEMA = "stateframe-parallel-ack/1"
_MAX_PROPOSAL_FILE_BYTES = 16 << 20
_MAX_PROPOSAL_BYTES = 256 << 20
_MAX_CHANNEL_REQUEST_BYTES = 1 << 20
_MAX_CHANNEL_REQUESTS_PER_PASS = 64
# At most two sync intents per recovering sequence; leave publication credits free.
_MAX_CHANNEL_REVALIDATIONS = 8
_MAX_PENDING_ACK_EVIDENCE = 256
_TICK_WORK_SECONDS = 0.05
_METRICS_REFRESH_SECONDS = 5.0  # host-metrics.json refresh cadence when latency is unchanged
_LATENCY_SAMPLES = 8192  # bounded rolling window behind the p99 / p99.9 figures
# Only passive recovered-exit observations share this view. Signal paths retain fresh
# scans and every saved member still undergoes its own birth-identity revalidation.
_RECORDED_OBSERVATION: ContextVar[TickProcessTable | None] = ContextVar(
    "stateframe-recorded-observation", default=None)


@lru_cache(maxsize=8)
def _host_runtime_id(pid: int) -> str:
    try:
        identity = ProcessTable().identity(pid)
        boot_id = _boot_id()
    except (OSError, RuntimeError, ValueError) as error:
        raise ParallelHostError("subframe host runtime identity is unavailable") from error
    if identity is None or identity.pid != pid:
        raise ParallelHostError("subframe host runtime identity is unavailable")
    return hashlib.sha256(canonical_json({"pid": pid, "start": list(identity.start),
                                          "boot_id": boot_id})).hexdigest()


def _ack_evidence_loop(supervisor_ref: weakref.ReferenceType, work_queue: queue.Queue,
                       stop_event: threading.Event) -> None:
    current = threading.current_thread()
    try:
        while not stop_event.is_set():
            supervisor = supervisor_ref()
            if supervisor is None or supervisor._runtime_stop:
                return
            item = supervisor._take_ack_evidence(work_queue)
            if item is None:
                del supervisor
                stop_event.wait(0.05)
                continue
            path, digest = item
            try:
                published = supervisor._publish_ack_evidence(path, digest)
            except Exception as error:
                supervisor._protocol_errors.append(
                    f"{path.name}: ACK evidence publication deferred: {type(error).__name__}: {str(error)[:200]}")
                published = False
            finally:
                supervisor._finish_ack_evidence(path, digest, published)
                work_queue.task_done()
                supervisor._publish_pending_ack_evidence()
            if not published:
                stop_event.wait(0.05)
            del supervisor
    finally:
        supervisor = supervisor_ref()
        if supervisor is not None:
            with supervisor._ack_evidence_worker_lock:
                if supervisor._ack_evidence_thread is current:
                    supervisor._ack_evidence_thread = None


class _LatencyStat:
    """Thread-safe max plus bounded-window percentiles for one host latency series."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: collections.deque[float] = collections.deque(maxlen=_LATENCY_SAMPLES)
        self.max = 0.0
        self.count = 0

    def record(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        with self._lock:
            self._samples.append(seconds)
            self.count += 1
            if seconds > self.max:
                self.max = seconds

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            samples = tuple(self._samples)
            maximum, count = self.max, self.count
        ordered = sorted(samples)

        def percentile(fraction: float) -> float:
            if not ordered:
                return 0.0
            return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]
        return {"max": maximum, "p99": percentile(0.99), "p99_9": percentile(0.999), "count": count}
_KERNEL_IDENTITY = re.compile(r"^(?:linux:[0-9a-fA-F-]{36}:\d+|darwin:\d+:\d+)$")


class ParallelHostError(RuntimeError):
    """A subframe channel or supervisor artifact failed validation."""


class _LaunchCancelled(ParallelHostError):
    """Preparation lost admission before a process was spawned."""


class ProcessInspectionDeferred(ParallelHostError):
    """Process liveness is unknown; retry without granting seal authority."""


def _inspection_deferred(error: BaseException) -> bool:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, (ProcessInspectionDeferred, subprocess.TimeoutExpired)):
            return True
        cause = cause.__cause__
    return False


@dataclass(frozen=True, slots=True)
class _MaterializedFile:
    data: bytes
    digest: str
    stat_identity: tuple[int, int, int, int, int]


@dataclass
class _Worker:
    call_id: str
    unit_id: str
    execution_id: str
    incarnation: int
    call_root: Path
    spec_path: Path
    spec_digest: str
    run_dir: Path
    snapshot_path: Path
    start_manifest_digest: str
    launch_payload: dict[str, object]
    snapshot_manifest_digest: str | None = None
    state: str = "STARTING"
    terminal: str | None = None
    exit_code: int | None = None
    error: str | None = None
    pid: int | None = None
    pgid: int | None = None
    created_epoch: float = field(default_factory=time.time)
    started_epoch: float | None = None
    finished_epoch: float | None = None
    exited_seq: int | None = None
    queued: bool = False
    preparing: bool = False
    finalizing: bool = False
    finalize_retry_at: float = 0.0
    group_verified: bool = False  # the exit's process group was checked clear outside the host lock
    cancel_requested: str | None = None
    handle: WorkerHandle | None = None
    thread: threading.Thread | None = None
    commit_lock: threading.RLock = field(default_factory=threading.RLock)
    generation: int = 0
    launch_committed: bool = False
    launch_commit_event: threading.Event = field(default_factory=threading.Event)
    commit_intent: dict[str, object] | None = None
    commit_ticket: Ticket | None = None
    commit_version: int = 0
    commit_failures: int = 0  # consecutive non-lost rejections of the current intent
    commit_retry_at: float = 0.0  # monotonic clock before which the intent is not resubmitted
    slot_marker: tuple[int, float] | None = None
    status_view: dict[str, object] | None = None
    # Recorded-group fence: None, PENDING, SIGNALLING, TERM, KILL, FAILED.
    fence_stage: str | None = None
    fence_deadline: float = 0.0  # monotonic clock at which the current fence window elapses
    fence_write: dict[str, object] | None = None
    fence_ticket: Ticket | None = None
    fence_acked_digest: str | None = None
    fence_write_failures: int = 0
    fence_write_retry_at: float = 0.0

    @property
    def state_path(self) -> Path:
        return self.call_root / "host" / "units" / self.unit_id / f"{self.incarnation}.json"

    @property
    def launch_path(self) -> Path:
        return self.call_root / "host" / "units" / self.unit_id / f"{self.incarnation}.launch.json"


class _WorkerIndex(dict[tuple[str, str, int], _Worker]):
    """Keep latest-incarnation lookup independent of filesystem and worker count."""

    def __init__(self) -> None:
        super().__init__()
        self.latest: dict[tuple[str, str], _Worker] = {}

    def __setitem__(self, key: tuple[str, str, int], worker: _Worker) -> None:
        super().__setitem__(key, worker)
        prior = self.latest.get(key[:2])
        if prior is None or worker.incarnation >= prior.incarnation:
            self.latest[key[:2]] = worker


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ParallelHostError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        if path.is_symlink():
            raise ParallelHostError(f"{label} is a symlink")
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ParallelHostError) as error:
        raise ParallelHostError(f"{label} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ParallelHostError(f"{label} must be a JSON object")
    return value


def _read_cursor(path: Path) -> object:
    """The last served position, or None. A cursor only orders service fairly and is written
    without fsync, so one a crash left torn or empty means "start from the beginning"."""
    if not path.exists():
        return None
    try:
        return _read_json(path, "channel cursor").get("last")
    except ParallelHostError:
        return None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: object, *, replace_existing: bool = False,
                durable: bool = True) -> None:
    """Atomically publish JSON. `durable=False` skips both fsyncs (the rename is still atomic):
    for records a crash may lose without consequence, such as scan cursors and metrics, whose
    fsyncs otherwise stall the supervisor tick for as long as the disk's journal commit takes."""
    encoded = canonical_json(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ParallelHostError(f"refusing to write through symlink: {path}")
    if path.exists() and not replace_existing:
        try:
            prior = path.read_bytes()
        except OSError as error:
            raise ParallelHostError(f"cannot inspect existing channel record {path}: {error}") from error
        if prior == encoded:
            # A previous attempt may have linked the record, then lost its directory-fsync ACK.
            if durable:
                _fsync_directory(path.parent)
            return
        raise ParallelHostError(f"immutable channel record already differs: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            if durable:
                os.fsync(output.fileno())
        if replace_existing:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != encoded:
                    raise ParallelHostError(f"immutable channel record raced with different content: {path}")
        if durable:
            _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _inside(path: Path, root: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ParallelHostError(f"{label} must be absolute")
    resolved_root = root.resolve()
    candidate = Path(os.path.abspath(path))
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ParallelHostError(f"{label} escapes the parallel artifact root") from error
    current = resolved_root
    for part in candidate.relative_to(resolved_root).parts:
        current /= part
        if current.is_symlink():
            raise ParallelHostError(f"{label} path contains a symlink: {current}")
    return candidate


def _regular_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ParallelHostError(f"cannot open {label}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ParallelHostError(f"{label} is not a regular file")
        if metadata.st_size > maximum:
            raise ParallelHostError(f"{label} exceeds {maximum} bytes")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ParallelHostError(f"{label} exceeds {maximum} bytes")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _regular_bytes_with_identity(
        path: Path, *, maximum: int, label: str, hash_bytes: bool = True
        ) -> tuple[bytes, str | None, tuple[int, int, int, int, int]]:
    """Read once with stable identity; skip hashing when an authenticated digest exists."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ParallelHostError(f"cannot open {label}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ParallelHostError(f"{label} is not a regular file")
        if before.st_size > maximum:
            raise ParallelHostError(f"{label} exceeds {maximum} bytes")
        chunks: list[bytes] = []
        digest = hashlib.sha256() if hash_bytes else None
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            if digest is not None:
                digest.update(chunk)
            total += len(chunk)
            if total > maximum:
                raise ParallelHostError(f"{label} exceeds {maximum} bytes")
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after) or after.st_size != total:
            raise ParallelHostError(f"{label} changed while reading")
        try:
            path_after = path.lstat()
        except OSError as error:
            raise ParallelHostError(f"{label} disappeared while reading") from error
        if (not stat.S_ISREG(path_after.st_mode)
                or _stat_identity(after) != _stat_identity(path_after)):
            raise ParallelHostError(f"{label} changed while reading")
        return (b"".join(chunks), digest.hexdigest() if digest is not None else None,
                _stat_identity(after))
    finally:
        os.close(descriptor)


def _regular_path_identity(path: Path) -> tuple[int, int, int, int, int]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ParallelHostError(f"cannot inspect materialized file: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ParallelHostError("materialized file is no longer a regular file")
    return _stat_identity(metadata)


def _entry_identity(root: Path, relative: str) -> tuple[str, str]:
    """Hash a regular file or symlink without following a path out of the workspace."""
    path = root / relative
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ParallelHostError(f"cannot inspect workspace entry {relative}: {error}") from error
    if stat.S_ISLNK(mode):
        try:
            target = os.readlink(path).encode("utf-8")
        except (OSError, UnicodeError) as error:
            raise ParallelHostError(f"cannot inspect workspace symlink {relative}: {error}") from error
        return "symlink", hashlib.sha256(target).hexdigest()
    if not stat.S_ISREG(mode):
        raise ParallelHostError(f"workspace entry is not a regular file or symlink: {relative}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ParallelHostError(f"cannot open workspace entry {relative}: {error}") from error
    try:
        opened_mode = os.fstat(descriptor).st_mode
        if not stat.S_ISREG(opened_mode):
            raise ParallelHostError(f"workspace entry changed type while hashing: {relative}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return "file", digest.hexdigest()
    finally:
        os.close(descriptor)


def _workspace_paths(root: Path) -> list[str]:
    if root.is_symlink() or not root.is_dir():
        raise ParallelHostError("subframe workspace is missing or unsafe")
    try:
        return sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                      if path.is_symlink() or not path.is_dir())
    except OSError as error:
        raise ParallelHostError(f"cannot list subframe workspace: {error}") from error


def _write_blob(path: Path, data: bytes, digest: str) -> None:
    if hashlib.sha256(data).hexdigest() != digest:
        raise ParallelHostError(f"proposal blob digest mismatch: {digest}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ParallelHostError(f"proposal blob path is a symlink: {path}")
    if path.exists():
        existing = _regular_bytes(path, maximum=_MAX_PROPOSAL_FILE_BYTES, label="proposal blob")
        if hashlib.sha256(existing).hexdigest() != digest:
            raise ParallelHostError(f"stored proposal blob is corrupt: {digest}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".blob-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _regular_bytes(path, maximum=_MAX_PROPOSAL_FILE_BYTES, label="proposal blob")
            if hashlib.sha256(existing).hexdigest() != digest:
                raise ParallelHostError(f"proposal blob raced with corrupt content: {digest}")
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _valid_unit_id(value: object) -> str:
    if not isinstance(value, str) or not _UNIT_ID.fullmatch(value):
        raise ParallelHostError("subframe id must match [a-z][a-z0-9_]{0,23}")
    return value


def _terminal(run_dir: Path) -> str | None:
    result = outcomes.terminal_receipt(run_dir).get("result")
    return result if isinstance(result, str) else None


# A durable commit the journal helper rejects (not loses) is retried a bounded number of times
# with backoff; after that the worker is failed and status reports the rejection to the client.
_COMMIT_REJECTION_LIMIT = 3
_COMMIT_BACKOFF_SECONDS = 0.05
# Lost ACKs (the helper spent its restart budget or made no progress) are not rejections, but
# resubmitting them without end would spin forever against a dead journal. After this many
# consecutive losses the host records a journal fault, fails the retained intents so the client
# sees them, and stops resubmitting; any later ACK clears the fault.
_JOURNAL_LOSS_LIMIT = 3
# Seal losses span tickets: each retry needs a new proof, but never a new loss budget.
_SEAL_HELPER_LOSS_LIMIT = 6
_SEAL_HELPER_BACKOFF_SECONDS = 0.05


class ParallelSupervisor:
    """Serve immutable parent requests and own every subframe process group for one run."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.parallel_root = self.run_dir / "parallel"
        self._lock = threading.RLock()
        self._workers = _WorkerIndex()
        self._closed = False
        self._process_table = TickProcessTable()
        self._protocol_errors = ProtocolErrorLog()
        self.max_tick_latency_seconds = 0.0
        # What the scale gate bounds: a channel pass, a lifecycle pass and request-to-answer latency.
        self._channel_pass_latency = _LatencyStat()
        self._lifecycle_pass_latency = _LatencyStat()
        self._answer_latency = _LatencyStat()
        # This lock protects only metric snapshots, never file or process I/O.
        self._metrics_lock = threading.Lock()
        self._phase_seconds = dict.fromkeys(("commits", "workers", "seals", "seal_discovery", "channels"), 0.0)
        self._max_phase_seconds = dict(self._phase_seconds)
        self._metrics_max_written: tuple[float, ...] = ()
        self._metrics_written_at = -math.inf
        self._metrics_retry_at = -math.inf
        self.tick_count = 0
        self._seal_ops: dict[Path, SealOperation] = {}
        self._seal_helpers: list[Journal] = []
        self._seal_cursor = 0
        self._sealed_jobs: set[Path] = set()
        self._journal_lost_acks = 0  # consecutive lost commit ACKs, reset by any delivered ACK
        self._journal_fault: str | None = None  # set at _JOURNAL_LOSS_LIMIT or on Journal.fault
        self._capacity_value: int | None = None
        self._tick_deadline: float | None = None
        self._ack_evidence_service_active = False
        self._channel_cursor: str = ""
        self._channel_completed: set[tuple[Path, int]] = set()
        self._channel_revalidating: set[tuple[Path, int]] = set()
        self._channel_revalidation_cursor: dict[Path, int] = {}
        self._journal: Journal | None = None
        self._channel_writes: ChannelWrites | None = None
        self._ack_evidence_lock = threading.Lock()
        # A reservation is created before an answer or seal result is submitted. Its
        # boolean marks whether the source durability ACK has arrived and is ready to enqueue.
        self._ack_evidence_pending: dict[Path, tuple[str, bool]] = {}
        # Queued, reserved, and in-flight items are disjoint, so their aggregate has one
        # hard cap without a separate lifetime index for publications.
        self._ack_evidence_scheduled: dict[Path, str] = {}
        self._ack_evidence_queue: queue.Queue[tuple[Path, str]] = queue.Queue(
            maxsize=_MAX_PENDING_ACK_EVIDENCE)
        self._ack_evidence_worker_lock = threading.Lock()
        self._ack_evidence_thread: threading.Thread | None = None
        self._ack_evidence_stop = threading.Event()
        self._channel_thread: threading.Thread | None = None
        self._uncommitted_seal_jobs: set[Path] = set()
        self._runtime_stop = False
        self._runtime_start_lock = threading.Lock()
        self._runtime_threads: list[threading.Thread] = []
        spec_path = self.run_dir / "spec.json"
        if spec_path.is_file():
            spec = _read_json(spec_path, "parent run spec")
            mode = spec.get("parallel_mode", "auto")
            limits = spec.get("parallel_limits")
            cap = limits.get("max_concurrency", 4) if isinstance(limits, dict) else 4
            if type(cap) is not int or cap < 1:
                raise ParallelHostError("parent subframe max_concurrency is invalid")
            if mode == "auto":
                threading.Thread(target=self._probe_capacity, args=(spec, cap), daemon=True,
                                 name="stateframe-parallel-slot-probe").start()
            elif isinstance(mode, str) and mode.isdecimal() and int(mode) > 0:
                self._capacity_value = min(int(mode), cap)
            else:
                self._capacity_value = cap
        else:
            self._capacity_value = 4
        self._hydrate_status()
        weakref.finalize(self, self._ack_evidence_stop.set)

    def _hydrate_status(self) -> None:
        """Recover durable views once, before any memory-only status request is served."""
        if self.parallel_root.is_symlink() or not self.parallel_root.is_dir():
            return
        # Live recovered groups share one observation, rather than a private scan per record.
        with self._process_table.tick():
            for path in sorted(self.parallel_root.glob("*/host/units/*/[0-9]*.json")):
                if not path.stem.isdecimal():
                    continue
                call_root = path.parents[3]
                try:
                    _inside(path, self.parallel_root, "worker status")
                    worker = self._load_worker(call_root.name, path.parent.name, int(path.stem), call_root)
                except (OSError, ParallelHostError) as error:
                    self._protocol_errors.append(f"status recovery refused: {error}")
                    continue
                if worker is None or worker.state == "EXITED":
                    continue
                if worker.pid is not None and worker.pgid is not None:
                    self._finalize_recorded_worker(worker)
                elif worker.state == "STARTING":
                    self._recover_unstarted_worker(worker)

    def _recover_unstarted_worker(self, worker: _Worker) -> None:
        """A STARTING record without a supervisor pid: the prior host died inside its launch.
        Nothing else ever starts or fails such a worker (the lifecycle lane supervises handles,
        the queue selects `queued` workers), so decide here. A process record for the incarnation
        means the launch may have happened: adopt its identity and supervise it as a recorded
        group. Otherwise re-queue the same incarnation (its proc dir is absent, so no birth
        identity can be confused with the relaunch)."""
        process_file = worker.run_dir / "proc" / str(worker.incarnation) / "process.json"
        worker.queued = False  # ambiguous launch artifacts must never enter preparation
        try:
            _inside(process_file.absolute(), worker.run_dir, "old subframe process record")
            if (worker.pid is None) != (worker.pgid is None):
                raise ParallelHostError("interrupted subframe launch has a partial process identity")
            if process_file.is_symlink() or process_file.exists():
                record = _read_json(process_file, "old subframe process record")
                pid, pgid = record.get("pid"), record.get("pgid")
                if (type(pid) is not int or type(pgid) is not int or pid < 1 or pid != pgid
                        or record.get("incarnation") not in (None, worker.incarnation)):
                    raise ParallelHostError("old subframe process record has an invalid identity")
                _recorded_kernel_identity(record)
                worker.pid, worker.pgid = pid, pgid
                self._persist_worker(worker)
                self._finalize_recorded_worker(worker)
                return
            if process_file.parent.exists():
                raise ParallelHostError("interrupted subframe launch lacks a complete process identity; "
                                        "inspect the process directory before retrying resume")
            if worker.cancel_requested is not None:
                self._fence_worker_record(worker, worker.cancel_requested)  # no process: CANCELLED
            else:
                worker.queued = True
                worker.preparing = False
                self._persist_worker(worker)
        except (OSError, ParallelHostError) as error:
            worker.error = f"interrupted subframe launch could not be recovered: {str(error)[:300]}"
            self._protocol_errors.append(f"{worker.unit_id}: {worker.error}")
            self._persist_worker(worker)

    def _finalize_recorded_worker(self, worker: _Worker) -> None:
        """A resumed worker has no handle, so its recorded process identities must prove its exit.
        The probe fails closed: any doubt leaves the worker as recorded, where the lifecycle lane
        (`_supervise_recorded_worker`) or a cancel moves it later."""
        process_file = worker.run_dir / "proc" / str(worker.incarnation) / "process.json"
        try:
            if worker.cancel_requested is not None:
                self._fence_worker_record(worker, worker.cancel_requested)
                return
            record = _read_json(process_file, "old subframe process record")
            if record.get("pid") != worker.pid or record.get("pgid") != worker.pgid:
                raise ParallelHostError("old subframe process record differs from supervisor identity")
            birth = _recorded_kernel_identity(record)
            if self._recorded_group_rows_live(worker.pgid):
                return
            if self._recorded_group_rows_live(worker.pgid, fresh=True):
                return
            if _recorded_group_live(worker.run_dir, worker.pid, worker.pgid, birth, process_file):
                return
            terminal = _terminal(worker.run_dir) or "ENGINE_ERROR"
            with self._lock:
                if worker.state in {"EXITED", "EXITING"} or worker.cancel_requested is not None:
                    return
                worker.finished_epoch = time.time()
                worker.terminal = terminal
                if worker.terminal == "ENGINE_ERROR":
                    worker.error = "subframe exited while its supervisor was down, without a terminal receipt"
                else:
                    worker.error = None
                worker.state = "EXITED"
                worker.group_verified = True
                try:
                    self._persist_worker(worker)
                finally:
                    worker.group_verified = False
        except (OSError, ParallelHostError) as error:
            self._protocol_errors.append(
                f"{worker.unit_id}: resumed subframe exit could not be verified: {str(error)[:300]}")

    def _recorded_group_rows_live(self, pgid: int, *, fresh: bool = False) -> bool:
        """Cheap pre-filter through the shared tick table: is any non-zombie row in ``pgid``?
        A failed scan is UNKNOWN (deferred), never an empty table; the identity-checked probe
        (`_recorded_group_live`) still decides every transition."""
        try:
            if (fresh and sys.platform == "linux" and pgid > 0
                    and _RECORDED_OBSERVATION.get() is self._process_table):
                # A complete tick scan already found no group. Linux can now prove atomic
                # absence without a second host enumeration; signal 0 sends no signal.
                # This is only a passive pre-filter: saved births are still checked below
                # by _recorded_group_live. Signal and seal-proof paths never use this scope.
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    return False
                except (OSError, OverflowError):
                    pass  # Inconclusive: retain the strict fresh-snapshot path.
            rows = self._process_table.snapshot() if fresh else self._process_table.tick_snapshot()
            if not rows or os.getpid() not in rows:
                raise ProcessInspectionDeferred("authoritative host process scan omits its observer; retry pending")
            if sys.platform == "darwin":
                return bool(live_group_rows(rows, pgid))
        except (OSError, RuntimeError, ValueError) as error:
            raise ProcessInspectionDeferred("authoritative host process scan failed; retry pending") from error
        return any(row.pgid == pgid and not row.zombie for row in rows.values())

    def _supervise_recorded_worker(self, worker: _Worker) -> None:
        """Lifecycle-lane supervision of a worker known only by its record (no handle). A requested
        fence advances one non-blocking step per pass; otherwise the worker is finalized once its
        recorded group is gone, so a group that exits by itself after resume does not stay RUNNING."""
        if (worker.handle is not None or worker.preparing or worker.pid is None or worker.pgid is None
                or worker.state not in {"STARTING", "RUNNING"} or worker.fence_stage == "FAILED"):
            return
        if worker.cancel_requested is not None or worker.fence_stage is not None:
            try:
                self._fence_worker_record(worker, worker.cancel_requested or "subframe cancel requested", advance=True)
            except JournalBusy:
                return  # retained process-record intent awaits admission or its durability ACK
            except (OSError, ParallelHostError) as error:
                if _inspection_deferred(error):
                    return  # a failed observation is not evidence; retry next pass
                message = f"subframe fence failed: {str(error)[:500]}"
                with self._lock:
                    worker.fence_stage = "FAILED"
                    if worker.error != message:
                        self._protocol_errors.append(f"{worker.unit_id}: {message}")
                    worker.error = message
            return
        now = time.monotonic()
        if now < worker.finalize_retry_at:
            return
        try:
            if self._recorded_group_rows_live(worker.pgid):
                return
        except ProcessInspectionDeferred:
            return
        token = _RECORDED_OBSERVATION.set(self._process_table)
        try:
            self._finalize_recorded_worker(worker)
        finally:
            _RECORDED_OBSERVATION.reset(token)
        if worker.state not in {"EXITED", "EXITING"}:
            worker.finalize_retry_at = now + 1.0  # a record fault is logged once per second, not per pass

    def _probe_capacity(self, spec: dict[str, object], cap: int) -> None:
        from .runner import RunError, auto_parallel_concurrency
        try:
            value = auto_parallel_concurrency(spec, cap)
        except (OSError, ValueError, RunError) as error:
            self._protocol_errors.append(f"subframe slot probe failed: {error}")
            value = 0
        with self._lock:
            self._capacity_value = value
            if value == 0:
                for worker in self._workers.values():
                    if worker.queued and worker.state == "STARTING":
                        self._refuse_no_capacity(worker)
        self._start_waiting_workers()

    def _refuse_no_capacity(self, worker: _Worker) -> None:
        worker.queued = False
        worker.terminal = "RESOURCES_UNAVAILABLE"
        worker.state = "EXITED"
        worker.error = "no idle subframe model slots are available"
        worker.finished_epoch = time.time()
        self._persist_worker(worker)

    def _has_capacity(self, worker: _Worker) -> bool:
        if self._capacity_value is None or self._capacity_value < 1:
            return False
        active = sum(1 for other in self._workers.values()
                     if other is not worker and (other.preparing or
                         (other.handle is not None and other.state != "EXITED")))
        return active < self._capacity_value

    def tick(self) -> None:
        """Bounded memory-only heartbeat; lifecycle and I/O lanes progress independently."""
        started = time.monotonic()
        try:
            self._tick()
        finally:
            self.tick_count += 1
            self.max_tick_latency_seconds = max(self.max_tick_latency_seconds,
                                                time.monotonic() - started)

    def _tick(self) -> None:
        if self._closed or self._runtime_stop:
            return
        if self._journal is None:
            self._start_runtime()

    def _start_runtime(self) -> None:
        if not self._runtime_start_lock.acquire(blocking=False):
            return
        try:
            if self._journal is not None or self._runtime_stop:
                return
            self._ensure_ack_evidence_worker()
            self._channel_writes = ChannelWrites()
            weakref.finalize(self, self._channel_writes.close)
            self._journal = Journal()
            weakref.finalize(self, self._journal.close)
            for name, target in (("channels", channel_loop), ("lifecycle", lifecycle_loop)):
                thread = threading.Thread(target=target, args=(weakref.ref(self),), daemon=True,
                                          name=f"stateframe-parallel-{name}")
                self._runtime_threads.append(thread)
                if name == "channels":
                    self._channel_thread = thread
                thread.start()
        finally:
            self._runtime_start_lock.release()

    def _stop_runtime(self) -> None:
        self._runtime_stop = True
        self._ack_evidence_stop.set()
        if self._journal is not None:
            self._journal.close()
        if self._channel_writes is not None:
            self._channel_writes.close()
        for helper in self._seal_helpers:
            helper.close()

    def _record_phase(self, phase: str, seconds: float) -> None:
        with self._metrics_lock:
            self._phase_seconds[phase] = max(0.0, seconds)
            self._max_phase_seconds[phase] = max(self._max_phase_seconds[phase], seconds)

    def _service_lifecycle(self) -> None:
        """Time each lifecycle phase, including failed passes, without changing its authority."""
        started = time.monotonic()
        try:
            for phase, action in (("commits", self._pump_commits), ("workers", self._advance_workers),
                                  ("seals", self._advance_seals)):
                phase_started = time.monotonic()
                try:
                    action()
                finally:
                    self._record_phase(phase, time.monotonic() - phase_started)
        finally:
            self._lifecycle_pass_latency.record(time.monotonic() - started)

    def _write_json(self, path: Path, value: object, *, replace_existing: bool = False,
                    durable: bool = True) -> None:
        # Artifact helpers use a partial supervisor with no runtime. Cursors and metrics
        # deliberately remain non-durable; live channel durability has its own bounded lane.
        numbered_answer = durable and path.parent.name == "an" and path.suffix == ".json" \
            and path.stem.isdigit()
        seal_result = durable and path.name.endswith(".result.json")
        answer_timer = self._answer_timer(path.parent, int(path.stem)) if numbered_answer else None
        ack_digest = hashlib.sha256(canonical_json(value) + b"\n").hexdigest() \
            if numbered_answer or seal_result else None
        ack_needs_publication = ack_digest is not None and not self._ack_evidence_is_current(path, ack_digest)
        ack_reserved_here = (self._reserve_ack_evidence(path, ack_digest)
                             if ack_needs_publication else False)

        def on_ack() -> None:
            if answer_timer is not None:
                answer_timer()
            if ack_needs_publication:
                self._queue_ack_evidence(path, ack_digest)

        writer = getattr(self, "_channel_writes", None)
        if writer is None or not durable:
            try:
                _write_json(path, value, replace_existing=replace_existing, durable=durable)
            except Exception:
                if ack_reserved_here:
                    self._release_ack_evidence(path, ack_digest)
                raise
            if ack_needs_publication:
                on_ack()
                self._publish_pending_ack_evidence()
            return
        asynchronous = threading.current_thread() is self._channel_thread
        try:
            writer.write(path, value, replace_existing=replace_existing, wait=not asynchronous,
                         on_ack=on_ack if ack_digest is not None else answer_timer)
        except JournalFault as error:
            if asynchronous:
                raise JournalBusy(f"channel durability unavailable: {error}; retry pending") from error
            if ack_reserved_here:
                self._release_ack_evidence(path, ack_digest)
            raise ParallelHostError(f"durable write unavailable: {error}") from error
        except ParallelHostError:
            if ack_reserved_here:
                self._release_ack_evidence(path, ack_digest)
            raise
        if numbered_answer or seal_result:
            # A successful return proves either this intent's ACK was consumed or the same
            # bytes already carry this runtime's consumed ACK.
            if ack_needs_publication:
                self._queue_ack_evidence(path, ack_digest)
            self._publish_pending_ack_evidence()

    def _ack_evidence_queued_digest_locked(self, path: Path) -> str | None:
        """Return a queued digest while holding _ack_evidence_lock."""
        work_queue = getattr(self, "_ack_evidence_queue", None)
        if work_queue is None:
            return None
        with work_queue.mutex:
            return next((queued_digest for queued_path, queued_digest in work_queue.queue
                         if queued_path == path), None)

    def _ack_evidence_total_locked(self) -> int:
        pending = getattr(self, "_ack_evidence_pending", {})
        scheduled = getattr(self, "_ack_evidence_scheduled", {})
        work_queue = getattr(self, "_ack_evidence_queue", None)
        queued = work_queue.qsize() if work_queue is not None else 0
        return len(pending) + queued + len(scheduled)

    def _reserve_ack_evidence(self, path: Path, digest: str) -> bool:
        """Reserve bounded memory before making an answer or seal result visible."""
        lock = getattr(self, "_ack_evidence_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._ack_evidence_lock = lock
        pending = getattr(self, "_ack_evidence_pending", None)
        if pending is None:
            pending = {}
            self._ack_evidence_pending = pending
        with lock:
            scheduled = getattr(self, "_ack_evidence_scheduled", None)
            if scheduled is None:
                scheduled = {}
                self._ack_evidence_scheduled = scheduled
            prior = pending.get(path)
            if prior is not None:
                if prior[0] != digest:
                    raise JournalBusy("ACK evidence changed while publication is pending; retry pending")
                return False
            prior = scheduled.get(path)
            if prior is None:
                prior = self._ack_evidence_queued_digest_locked(path)
            if prior is not None:
                if prior != digest:
                    raise JournalBusy("ACK evidence changed while publication is pending; retry pending")
                return False
            if self._ack_evidence_total_locked() >= _MAX_PENDING_ACK_EVIDENCE:
                raise JournalBusy("ACK evidence reservations are full; retry pending")
            pending[path] = (digest, False)
            return True

    def _release_ack_evidence(self, path: Path, digest: str) -> None:
        lock = getattr(self, "_ack_evidence_lock", None)
        pending = getattr(self, "_ack_evidence_pending", None)
        if lock is None or pending is None:
            return
        with lock:
            reservation = pending.get(path)
            if reservation == (digest, False):
                pending.pop(path, None)

    def _queue_ack_evidence(self, path: Path, digest: str) -> None:
        lock = getattr(self, "_ack_evidence_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._ack_evidence_lock = lock
        pending = getattr(self, "_ack_evidence_pending", None)
        if pending is None:
            pending = {}
            self._ack_evidence_pending = pending
        with lock:
            reservation = pending.get(path)
            if reservation is not None:
                if reservation[0] != digest:
                    self._protocol_errors.append(f"{path.name}: ACK evidence ACK did not match its reservation")
                    return
                pending[path] = (digest, True)
                return
            scheduled = getattr(self, "_ack_evidence_scheduled", {})
            prior = scheduled.get(path)
            if prior is None:
                prior = self._ack_evidence_queued_digest_locked(path)
            if prior == digest:
                return
            if prior is not None:
                self._protocol_errors.append(f"{path.name}: ACK evidence ACK conflicts with pending publication")
                return
            # Callers reserve before submission; refusing a missing reservation keeps the
            # callback nonblocking and leaves the result untrusted until resume revalidates it.
            self._protocol_errors.append(f"{path.name}: ACK evidence ACK arrived without a reservation")

    def _ack_evidence_identity(self, path: Path) -> tuple[int, str, str] | None:
        if path.parent.name == "an" and path.suffix == ".json" and path.stem.isdigit():
            attempt_dir = path.parent.parent
            match = re.fullmatch(r"attempt-(\d+)", attempt_dir.name)
            if match is None:
                return None
            call_root = attempt_dir.parent
            attempt = int(match.group(1))
        elif path.name.endswith(".result.json"):
            job = path.with_name(path.name.removesuffix(".result.json") + ".json")
            if job.parent.name != "seals" or job.parent.parent.name != "host":
                return None
            call_root = job.parent.parent.parent
            try:
                sealed = self._seal_job(job)
                if sealed is None:
                    return None
                request = HostRequest.from_json(sealed["request"])
            except (OSError, JournalBusy, ParallelHostError, ParallelRecordError, TypeError):
                return None
            attempt = request.attempt
            if request.call_id != call_root.name:
                return None
        else:
            return None
        if call_root.parent != self.parallel_root:
            return None
        try:
            runtime = _host_runtime_id(os.getpid())
        except ParallelHostError:
            return None
        return attempt, call_root.name, runtime

    def _publish_pending_ack_evidence(self) -> None:
        if getattr(self, "_ack_evidence_service_active", False):
            return
        lock = getattr(self, "_ack_evidence_lock", None)
        pending = getattr(self, "_ack_evidence_pending", None)
        if lock is None or pending is None:
            return
        work_queue = getattr(self, "_ack_evidence_queue", None)
        direct = (work_queue is None or
                  (threading.current_thread() is getattr(self, "_channel_thread", None)
                   and getattr(self, "_ack_evidence_thread", None) is None))
        if direct:
            # Test and direct-use channel lanes may bypass _start_runtime. Keep thread
            # startup out of their measured pass; this tiny non-durable hint follows ACK.
            items: list[tuple[Path, str]] = []
            with lock:
                scheduled = getattr(self, "_ack_evidence_scheduled", {})
                for path, (digest, ready) in tuple(pending.items()):
                    if not ready:
                        continue
                    pending.pop(path, None)
                    scheduled[path] = digest
                    items.append((path, digest))
            for path, digest in items:
                try:
                    published = self._publish_ack_evidence(path, digest)
                except Exception as error:
                    self._protocol_errors.append(
                        f"{path.name}: ACK evidence publication deferred: {type(error).__name__}: {str(error)[:200]}")
                    published = False
                self._finish_ack_evidence(path, digest, published)
            return
        if not self._ensure_ack_evidence_worker():
            return
        with lock:
            for path, (digest, ready) in tuple(pending.items()):
                if not ready:
                    continue
                pending.pop(path, None)
                try:
                    work_queue.put_nowait((path, digest))
                except queue.Full:
                    # The slot was already reserved; restoring it cannot exceed the cap.
                    pending[path] = (digest, True)

    def _take_ack_evidence(self, work_queue: queue.Queue) -> tuple[Path, str] | None:
        lock = getattr(self, "_ack_evidence_lock", None)
        if lock is None:
            return None
        with lock:
            try:
                path, digest = work_queue.get_nowait()
            except queue.Empty:
                return None
            self._ack_evidence_scheduled[path] = digest
            return path, digest

    def _finish_ack_evidence(self, path: Path, digest: str, published: bool) -> None:
        lock = getattr(self, "_ack_evidence_lock", None)
        if lock is None:
            return
        with lock:
            scheduled = getattr(self, "_ack_evidence_scheduled", {})
            if scheduled.get(path) != digest:
                return
            scheduled.pop(path, None)
            if not published:
                self._ack_evidence_pending[path] = (digest, True)

    def _ensure_ack_evidence_worker(self) -> bool:
        lock = getattr(self, "_ack_evidence_worker_lock", None)
        stop_event = getattr(self, "_ack_evidence_stop", None)
        work_queue = getattr(self, "_ack_evidence_queue", None)
        if lock is None or stop_event is None or work_queue is None:
            return False
        with lock:
            if getattr(self, "_ack_evidence_thread", None) is not None:
                return True
            if stop_event.is_set() or getattr(self, "_runtime_stop", False):
                return False
            worker = threading.Thread(
                target=_ack_evidence_loop,
                args=(weakref.ref(self), work_queue, stop_event),
                daemon=True, name="stateframe-parallel-ack-evidence")
            self._ack_evidence_thread = worker
            self._runtime_threads.append(worker)
            try:
                worker.start()
            except RuntimeError as error:
                self._ack_evidence_thread = None
                self._protocol_errors.append(
                    f"ACK evidence worker did not start: {type(error).__name__}: {str(error)[:200]}")
                return False
        return True

    def _publish_ack_evidence(self, path: Path, digest: str) -> bool:
        identity = self._ack_evidence_identity(path)
        if identity is None:
            return False
        attempt, call_id, runtime = identity
        evidence = {"schema": _HOST_ACK_SCHEMA, "attempt": attempt, "call_id": call_id,
                    "runtime": runtime, "sha256": digest}
        ack_path = path.with_name(f"{path.stem}.ack.json")
        try:
            # This atomic hint follows the source durability ACK. If it is lost in a crash,
            # the durable source is re-ACKed before the hint is recreated.
            _write_json(ack_path, evidence, replace_existing=True, durable=False)
        except (OSError, ParallelHostError) as error:
            self._protocol_errors.append(
                f"{path.name}: ACK evidence publication deferred: {type(error).__name__}: {str(error)[:200]}")
            return False
        return True

    def _ack_evidence_is_scheduled(self, path: Path) -> bool:
        lock = getattr(self, "_ack_evidence_lock", None)
        scheduled = getattr(self, "_ack_evidence_scheduled", None)
        pending = getattr(self, "_ack_evidence_pending", None)
        if lock is None or scheduled is None or pending is None:
            return False
        with lock:
            reservation = pending.get(path)
            return ((reservation is not None and reservation[1]) or path in scheduled
                    or self._ack_evidence_queued_digest_locked(path) is not None)

    def _ack_evidence_is_current(self, path: Path, digest: str) -> bool:
        ack_path = path.with_name(f"{path.stem}.ack.json")
        if not os.path.lexists(ack_path):
            return False
        try:
            raw = _regular_bytes(ack_path, maximum=4096, label="ACK evidence")
            evidence = json.loads(raw, object_pairs_hook=_strict_object)
        except (OSError, ParallelHostError, UnicodeError, json.JSONDecodeError):
            return False
        identity = self._ack_evidence_identity(path)
        if not isinstance(evidence, dict) or identity is None:
            return False
        attempt, call_id, runtime = identity
        return (set(evidence) == {"schema", "attempt", "call_id", "runtime", "sha256"}
                and evidence.get("schema") == _HOST_ACK_SCHEMA
                and type(evidence.get("attempt")) is int and evidence.get("attempt") == attempt
                and evidence.get("call_id") == call_id and evidence.get("runtime") == runtime
                and evidence.get("sha256") == digest)

    def _channel_barrier(self, path: Path) -> None:
        writer = getattr(self, "_channel_writes", None)
        if writer is not None:
            try:
                writer.barrier(path)
            except JournalFault as error:
                self._protocol_errors.append(f"channel durability unavailable: {error}")
                raise JournalBusy(f"channel durability unavailable: {error}; retry pending") from error

    def _read_channel_json(self, path: Path, label: str) -> dict[str, object] | None:
        writer = getattr(self, "_channel_writes", None)
        if writer is None:
            return read_synced_record(path, label)
        try:
            return writer.read(path, label, wait=threading.current_thread() is not self._channel_thread)
        except JournalFault as error:
            self._protocol_errors.append(f"channel durability unavailable: {error}")
            raise JournalBusy(f"channel durability unavailable: {error}; retry pending") from error

    def _channel_answer_complete(self, an_dir: Path, seq: int) -> bool:
        try:
            answer_path = self._answer_path(an_dir, seq)
            if self._ack_evidence_is_scheduled(answer_path):
                return False
            complete = (self._read_channel_json(self._record_path(an_dir, seq), "channel identity record") is not None
                        and self._read_channel_json(answer_path, "channel answer") is not None)
            if not complete:
                return False
            answer_bytes = _regular_bytes(answer_path, maximum=_MAX_CHANNEL_REQUEST_BYTES,
                                          label="channel answer")
            digest = hashlib.sha256(answer_bytes).hexdigest()
            if self._ack_evidence_is_current(answer_path, digest):
                return True
            # Both reads above returned only after this runtime consumed their durability
            # ACKs. Schedule the additive hint once; do not replay the answer while it writes.
            self._reserve_ack_evidence(answer_path, digest)
            self._queue_ack_evidence(answer_path, digest)
            self._publish_pending_ack_evidence()
            return self._ack_evidence_is_current(answer_path, digest)
        except (JournalBusy, OSError, ParallelHostError):
            return False

    def _pending_channel_sequences(self, an_dir: Path, sequences: list[int]) -> list[int]:
        if self._tick_deadline is None:
            return sequences  # direct callers still validate replays and conflicts
        if self._budget_expired():
            return []
        pending: list[int] = []
        recovered: list[int] = []
        for seq in sequences:
            if (an_dir, seq) in self._channel_completed:
                continue
            if self._answer_path(an_dir, seq).exists():
                recovered.append(seq)
            else:
                pending.append(seq)
        prior = self._channel_revalidation_cursor.get(an_dir, -1)
        start = next((index for index, seq in enumerate(recovered) if seq > prior), 0)
        checked = 0
        for offset in range(len(recovered)):
            if self._budget_expired() or checked >= _MAX_CHANNEL_REVALIDATIONS:
                break
            seq = recovered[(start + offset) % len(recovered)]
            key = (an_dir, seq)
            if key not in self._channel_revalidating:
                if len(self._channel_revalidating) >= _MAX_CHANNEL_REVALIDATIONS:
                    continue
                self._channel_revalidating.add(key)
            self._channel_revalidation_cursor[an_dir] = seq
            checked += 1
            if self._channel_answer_complete(an_dir, seq):
                # Immutable sequence completion outlives the writer's small digest cache.
                # Retain no record bytes or callbacks, and rebuild this evidence on resume.
                self._channel_completed.add(key)
                self._channel_revalidating.discard(key)
            elif self._ack_evidence_is_scheduled(self._answer_path(an_dir, seq)):
                # The source reads already consumed this runtime's durability ACKs. The
                # sidecar worker now owns eventual publication, so avoid re-ACKing history.
                self._channel_completed.add(key)
                self._channel_revalidating.discard(key)
                continue
            else:
                pending.append(seq)
        return sorted(pending)

    def _service_channels(self) -> None:
        started = time.monotonic()
        self._ack_evidence_service_active = True
        try:
            if self._channel_writes is not None:
                self._uncommitted_seal_jobs.difference_update(self._channel_writes.drain())
            self._tick_deadline = started + _TICK_WORK_SECONDS
            self._service_channel_pass()
        finally:
            self._tick_deadline = None
            self._channel_pass_latency.record(time.monotonic() - started)
            self._ack_evidence_service_active = False
        # Advisory I/O must not ride the journal or mask an exception from the pass.
        # A helper keeps normal early returns on this path, including idle passes.
        self._publish_pending_ack_evidence()
        self._flush_metrics()

    def _service_channel_pass(self) -> None:
        if self._budget_expired() or self._closed:
            return
        if self.parallel_root.is_symlink() or not self.parallel_root.is_dir():
            return
        started = time.monotonic()
        try:
            self._publish_seal_results()
            self._discover_seals()
        except Exception as error:  # seal bookkeeping faults never stop channel service
            self._protocol_errors.append(f"seal discovery deferred: {type(error).__name__}: {str(error)[:300]}")
        finally:
            self._record_phase("seal_discovery", time.monotonic() - started)
        if self._budget_expired():
            return
        started = time.monotonic()
        try:
            attempts = sorted((directory for call in self.parallel_root.iterdir()
                               if call.is_dir() and not call.is_symlink()
                               for directory in call.glob("attempt-*")
                               if directory.is_dir() and not directory.is_symlink()
                               and (directory / "rq").is_dir()), key=str)
            after = next((i for i, item in enumerate(attempts)
                          if str(item) > self._channel_cursor), 0)
            for offset in range(min(len(attempts), 8)):
                if self._budget_expired():
                    break
                attempt = attempts[(after + offset) % len(attempts)]
                self._channel_cursor = str(attempt)
                self.serve(attempt / "rq", attempt / "an")
        finally:
            self._record_phase("channels", time.monotonic() - started)

    def _answer_timer(self, an_dir: Path, seq: int) -> Callable[[], None] | None:
        """Retain arrival time in the bounded write intent until its ACK is consumed.

        Replays of visible answers are not new publications, even after the writer's
        ACK cache rolls over. Once accepted, the timer survives visibility and retries;
        only successful settlement can record it, with no filesystem I/O in the callback.
        """
        if self._answer_path(an_dir, seq).exists():
            return None
        arrived: float | None = None
        try:
            for path in (an_dir.parent / "rq").iterdir():
                if (path.suffix != ".json" or not path.stem.isdigit()
                        or int(path.stem) != seq or path.is_symlink()):
                    continue
                try:
                    timestamp = path.stat().st_mtime
                except OSError:
                    continue
                arrived = timestamp if arrived is None else min(arrived, timestamp)
        except OSError:
            pass
        if arrived is None:
            return None
        age = time.time() - arrived
        started = time.monotonic()
        latency = self._answer_latency
        return lambda: latency.record(age + time.monotonic() - started)

    def _flush_metrics(self) -> None:
        """Publish lifetime maxima and rolling-window percentiles outside every host lock."""
        now = time.monotonic()
        if now < self._metrics_retry_at:
            return
        answers = self._answer_latency.snapshot()
        channel = self._channel_pass_latency.snapshot()
        lifecycle = self._lifecycle_pass_latency.snapshot()
        with self._metrics_lock:
            phases = dict(self._phase_seconds)
            max_phases = dict(self._max_phase_seconds)
        # Compare every maximum: a large answer latency must not hide a new pass maximum.
        maxima = (self.max_tick_latency_seconds, float(answers["max"]), float(channel["max"]),
                  float(lifecycle["max"]), *max_phases.values())
        stale = now - self._metrics_written_at >= _METRICS_REFRESH_SECONDS
        if not (maxima != self._metrics_max_written or stale):
            return
        if self.parallel_root.is_symlink() or not self.parallel_root.is_dir():
            return
        self._metrics_max_written = maxima
        self._metrics_written_at = now
        written = self._write_metrics(self.parallel_root / "host-metrics.json",
                                   {"schema": "stateframe-parallel-host-metrics/1",
                                    "max_tick_latency_seconds": self.max_tick_latency_seconds,
                                    "ticks": self.tick_count,
                                    "answer_latency_seconds": answers,
                                    "channel_pass_seconds": channel,
                                    "lifecycle_pass_seconds": lifecycle,
                                    "percentile_window_samples": _LATENCY_SAMPLES,
                                    "phase_seconds": phases,
                                    "max_phase_seconds": max_phases})
        if written is False:
            # A disk fault must not cause retries on every new latency maximum.
            self._metrics_retry_at = now + _METRICS_REFRESH_SECONDS

    def _write_metrics(self, path: Path, value: dict[str, object]) -> bool:
        """Best-effort non-durable write; no journal or supervisor lock is involved."""
        try:
            _write_json(path, value, replace_existing=True, durable=False)
        except (OSError, ParallelHostError) as error:
            self._protocol_errors.append(f"host metrics not written: {type(error).__name__}: {str(error)[:300]}")
            return False
        return True

    def _budget_expired(self) -> bool:
        return self._tick_deadline is not None and time.monotonic() >= self._tick_deadline

    def serve(self, rq_dir: Path, an_dir: Path) -> tuple[HostAnswer, ...]:
        """Serve pending `rq/*.json` files into `an/*.json`; repeated requests are idempotent."""
        rq_dir, an_dir = Path(rq_dir), Path(an_dir)
        if rq_dir.name != "rq" or an_dir.name != "an" or an_dir.parent != rq_dir.parent:
            raise ParallelHostError("channel directories must be sibling rq/ and an/ directories")
        attempt_dir = rq_dir.parent
        match = re.fullmatch(r"attempt-(\d+)", attempt_dir.name)
        if match is None or attempt_dir.parent.parent != self.parallel_root:
            raise ParallelHostError("channel path is outside this run's parallel call namespace")
        call_root = attempt_dir.parent
        attempt = int(match.group(1))
        if call_root.name in (".", "..") or "/" in call_root.name or "\\" in call_root.name:
            raise ParallelHostError("channel call id is unsafe")
        if (self.parallel_root.is_symlink() or call_root.is_symlink() or attempt_dir.is_symlink()
                or rq_dir.is_symlink() or (an_dir.exists() and an_dir.is_symlink())):
            raise ParallelHostError("channel directories must not be symlinks")
        an_dir.mkdir(parents=True, exist_ok=True)

        groups: dict[int, list[Path]] = {}
        for path in rq_dir.iterdir():
            if path.suffix != ".json" or not path.stem.isdigit() or path.is_symlink():
                continue
            groups.setdefault(int(path.stem), []).append(path)
        answers: list[HostAnswer] = []
        cursor_path = attempt_dir / "channel-cursor.json"
        prior = _read_cursor(cursor_path)
        # Recovery cannot fill the publication lane or repeatedly revalidate all history.
        sequences = self._pending_channel_sequences(an_dir, sorted(groups))
        start = next((index for index, value in enumerate(sequences)
                      if value > prior), 0) if type(prior) is int else 0
        for offset in range(min(len(sequences), _MAX_CHANNEL_REQUESTS_PER_PASS)):
            if self._budget_expired():
                break
            seq = sequences[(start + offset) % len(sequences)]
            self._write_json(cursor_path, {"last": seq}, replace_existing=True, durable=False)
            paths = sorted(groups[seq], key=lambda item: item.name)
            requests = []
            invalid = []
            for path in paths:
                try:
                    requests.append(self._read_request(path, call_root, attempt, seq))
                except ParallelHostError as error:
                    invalid.append((path, error))
            if invalid:
                path, error = invalid[0]
                answer = self._protocol_error(attempt, call_root.name, seq, self._request_op(path), str(error))
                try:
                    answers.append(self._record_channel_error(an_dir, answer, path))
                except JournalBusy:
                    continue
                continue
            if any(self._request_identity(item) != self._request_identity(requests[0]) for item in requests[1:]):
                answer = self._answer(requests[0], "ERROR", None,
                                      {"error": "duplicate sequence has different request identity"})
                try:
                    self._persist_conflict(an_dir, answer, requests[0])
                except JournalBusy:
                    continue
                answers.append(answer)
                continue
            try:
                answer = self._serve_request(requests[0], call_root, an_dir, seq)
            except JournalBusy:
                continue  # unanswered: this sequence is re-served on a later pass, the rest go on
            answers.append(answer)
        return tuple(answers)

    def _read_request(self, path: Path, call_root: Path, attempt: int, seq: int) -> HostRequest:
        try:
            data = _regular_bytes(path, maximum=_MAX_CHANNEL_REQUEST_BYTES, label=f"host request {path.name}")
            raw = json.loads(data, object_pairs_hook=_strict_object)
            request = HostRequest.from_json(raw)
        except (OSError, UnicodeError, json.JSONDecodeError, ParallelRecordError, ParallelHostError) as error:
            raise ParallelHostError(f"host request {path.name} is invalid: {error}") from error
        if request.attempt != attempt or request.call_id != call_root.name or request.seq != seq:
            raise ParallelHostError("host request identity does not match its channel path")
        actual = hashlib.sha256(canonical_json(request.payload)).hexdigest()
        if request.payload_sha256 != actual:
            raise ParallelHostError("host request payload digest does not match its payload")
        return request

    @staticmethod
    def _request_op(path: Path) -> str:
        try:
            data = _regular_bytes(path, maximum=_MAX_CHANNEL_REQUEST_BYTES, label=f"host request {path.name}")
            value = json.loads(data, object_pairs_hook=_strict_object)
        except (OSError, UnicodeError, json.JSONDecodeError, ParallelHostError):
            return "protocol"
        op = value.get("op") if isinstance(value, dict) else None
        return op if isinstance(op, str) and op else "protocol"

    def _record_channel_error(self, an_dir: Path, answer: HostAnswer, path: Path) -> HostAnswer:
        raw_digest = "unreadable"
        try:
            data = _regular_bytes(path, maximum=_MAX_CHANNEL_REQUEST_BYTES, label=f"host request {path.name}")
            raw_digest = hashlib.sha256(data).hexdigest()
        except ParallelHostError:
            pass
        record = self._channel_record(an_dir, answer.seq)
        if record is not None:
            if record["invalid_request"] is True and record["request_sha256"] == raw_digest:
                return self._publish_recorded_answer(an_dir, answer.seq, record)
            suffix = hashlib.sha256(raw_digest.encode("ascii")).hexdigest()[:12]
            self._write_json(an_dir / f"{answer.seq:03d}.conflict-{suffix}.json", answer.to_json())
            return answer
        self._store_answer(an_dir, answer.seq, raw_digest, answer, invalid_request=True)
        return answer

    @staticmethod
    def _request_identity(request: HostRequest) -> str:
        return hashlib.sha256(canonical_json(request.to_json())).hexdigest()

    @staticmethod
    def _answer_path(an_dir: Path, seq: int) -> Path:
        return an_dir / f"{seq:03d}.json"

    @staticmethod
    def _record_path(an_dir: Path, seq: int) -> Path:
        return an_dir / f"{seq:03d}.record.json"

    def _channel_record(self, an_dir: Path, seq: int) -> dict[str, object] | None:
        path = self._record_path(an_dir, seq)
        value = self._read_channel_json(path, "channel identity and answer record")
        if value is None:
            if self._answer_path(an_dir, seq).exists():
                raise ParallelHostError(f"channel answer lacks its atomic identity record: {path}")
            return None
        if (set(value) != {"schema", "request_sha256", "invalid_request", "answer"}
                or value.get("schema") != _CHANNEL_RECORD_SCHEMA
                or not isinstance(value.get("request_sha256"), str)
                or type(value.get("invalid_request")) is not bool):
            raise ParallelHostError(f"channel identity and answer record is invalid: {path}")
        try:
            answer = HostAnswer.from_json(value["answer"])
        except (ParallelRecordError, TypeError) as error:
            raise ParallelHostError(f"channel answer record is invalid: {error}") from error
        if answer.seq != seq:
            raise ParallelHostError("channel answer record sequence differs from its path")
        return value

    def _publish_recorded_answer(self, an_dir: Path, seq: int,
                                 record: dict[str, object]) -> HostAnswer:
        answer = HostAnswer.from_json(record["answer"])
        answer_path = self._answer_path(an_dir, seq)
        self._write_json(answer_path, answer.to_json())
        # Callers have already ACKed the immutable identity record.
        self._channel_completed.add((an_dir, seq))
        self._channel_revalidating.discard((an_dir, seq))
        return answer

    def _store_answer(self, an_dir: Path, seq: int, request_digest: str,
                      answer: HostAnswer, *, invalid_request: bool = False) -> None:
        record = {"schema": _CHANNEL_RECORD_SCHEMA, "request_sha256": request_digest,
                  "invalid_request": invalid_request, "answer": answer.to_json()}
        self._write_json(self._record_path(an_dir, seq), record)
        self._publish_recorded_answer(an_dir, seq, record)

    def _serve_request(self, request: HostRequest, call_root: Path, an_dir: Path, seq: int) -> HostAnswer:
        answer_path = self._answer_path(an_dir, seq)
        request_digest = self._request_identity(request)
        record = self._channel_record(an_dir, seq)
        if record is not None:
            answer = self._load_stored_answer(answer_path, record, request)
            if answer is not None:
                return answer
            error_answer = self._answer(request, "ERROR", None,
                                        {"error": "duplicate sequence has a different request identity"})
            self._persist_conflict(an_dir, error_answer, request)
            return error_answer
        try:
            status, result_id, payload = self._dispatch(request, call_root)
            answer = self._answer(request, status, result_id, payload)
        except (ParallelHostError, ParallelRecordError, TxnStoreError, WorkerPreparationError,
                WorkerLaunchError, OSError, ValueError, TypeError) as error:
            if _inspection_deferred(error):
                # A deferral (seal admission full, process table unavailable) is transient: the
                # record is immutable, so it must stay unanswered and be re-served, never ERROR.
                raise JournalBusy(f"{type(error).__name__}: {str(error)[:300]}; retry pending") from error
            answer = self._answer(request, "ERROR", None,
                                  {"error": f"{type(error).__name__}: {str(error)[:500]}"})
        if request.op == "launch" and answer.status != "ERROR":
            worker = self._workers.latest.get((request.call_id, str(request.payload.get("unit_id"))))
            if worker is None:
                raise JournalBusy("launch answer awaits its accepted worker intent")
            # Never sleep on the channel thread: poll the durable launch ACK and otherwise leave
            # the sequence unanswered, so it is re-served within one pass of the ACK landing.
            if not worker.launch_commit_event.is_set():
                if worker.commit_intent is None and not worker.launch_committed:
                    # The journal rejected the launch intent durably and the lifecycle lane
                    # dropped it; the launch can never become durable.
                    answer = self._answer(request, "ERROR", None, {"error": str(
                        worker.error or "durable worker commit rejected: launch intent dropped")[:500]})
                else:
                    raise JournalBusy("launch answer awaits its durable intent ACK; retry pending")
        self._store_answer(an_dir, seq, request_digest, answer)
        return answer

    def _load_stored_answer(self, answer_path: Path, record: dict[str, object],
                            request: HostRequest) -> HostAnswer | None:
        answer = self._publish_recorded_answer(answer_path.parent, request.seq, record)
        if (answer.attempt, answer.call_id, answer.seq) != (request.attempt, request.call_id, request.seq):
            raise ParallelHostError(f"stored channel answer integrity failed: {answer_path}")
        if record["invalid_request"] is True:
            return None
        if record["request_sha256"] != self._request_identity(request):
            return None
        return answer

    def _persist_conflict(self, an_dir: Path, answer: HostAnswer, request: HostRequest) -> None:
        if self._channel_record(an_dir, answer.seq) is None:
            self._store_answer(an_dir, answer.seq, self._request_identity(request), answer)
            return
        suffix = hashlib.sha256(canonical_json(request.to_json())).hexdigest()[:12]
        conflict_path = an_dir / f"{answer.seq:03d}.conflict-{suffix}.json"
        self._write_json(conflict_path, answer.to_json())

    @staticmethod
    def _answer(request: HostRequest, status: str, result_id: str | None,
                payload: dict[str, object]) -> HostAnswer:
        return HostAnswer(request.attempt, request.call_id, request.seq, request.op, status,
                          result_id, payload)

    @staticmethod
    def _protocol_error(attempt: int, call_id: str, seq: int, op: str, message: str) -> HostAnswer:
        return HostAnswer(attempt, call_id, seq, op, "ERROR", None, {"error": message[:500]})

    def _dispatch(self, request: HostRequest, call_root: Path) -> tuple[str, str | None, dict[str, object]]:
        if not isinstance(request.scope, str) or not request.scope:
            raise ParallelHostError("host request scope must be a nonempty string")
        if request.op == "launch":
            return self._launch(request, call_root)
        if request.op == "status":
            return "OK", None, self._status(request, call_root)
        if request.op == "seal":
            return self._enqueue_seal(request, call_root)
        if request.op == "cancel":
            return "OK", None, self._cancel(request, call_root)
        raise ParallelHostError(f"unsupported host operation {request.op!r}")

    def _enqueue_seal(self, request: HostRequest, call_root: Path) -> tuple[str, str | None, dict[str, object]]:
        if set(request.payload) != {"unit_id", "incarnation"}:
            raise ParallelHostError("seal payload keys do not match the host interface")
        unit_id = _valid_unit_id(request.payload["unit_id"])
        incarnation = request.payload["incarnation"]
        if type(incarnation) is not int or incarnation < 1:
            raise ParallelHostError("seal incarnation must be a positive integer")
        # Record, job and result reads happen before the host lock; identity is re-validated inside.
        recorded = self._recorded_workers(request.call_id, unit_id, call_root)
        seal_root = call_root / "host" / "seals"
        job = seal_root / f"{unit_id}-{incarnation}.json"
        result_path = seal_root / f"{unit_id}-{incarnation}.result.json"
        accepted = request
        job_value = {"schema": "stateframe-parallel-seal-job/2", "operation_id": operation_id(request),
                     "request": request.to_json()}
        recorded_job = self._seal_job(job)
        recorded_result = self._seal_result(result_path)
        if recorded_job is not None:
            job_value = recorded_job
            accepted = HostRequest.from_json(job_value["request"])
            if operation_id(accepted) != operation_id(request):
                raise ParallelHostError("seal operation digest conflicts with its accepted incarnation")
        elif recorded_result is not None:
            raise ParallelHostError("seal result lacks its accepted operation identity")
        with self._lock:
            worker = self._latest_worker(request.call_id, unit_id, call_root, recorded)
            if worker is None or worker.incarnation != incarnation or worker.state != "EXITED" or worker.exited_seq is None:
                raise ParallelHostError("cannot seal an unfinished or stale subframe")
            if recorded_result is not None:
                return str(recorded_result["status"]), None, recorded_result["payload"]
            operation = self._seal_ops.get(job)
            if operation is None:
                if len(self._seal_ops) >= 32:
                    raise ProcessInspectionDeferred("seal admission is full; retry pending")
                operation = SealOperation(worker, accepted, job, result_path, worker.generation)
                operation.helper = len(self._seal_ops) % 2
                self._uncommitted_seal_jobs.add(job)
                self._seal_ops[job] = operation  # reservation precedes any materialization
        if operation.phase == "DONE":
            result = self._seal_result(result_path)
            if result is None:
                raise JournalBusy("completed seal awaits its durable result")
            return str(result["status"]), None, result["payload"]
        self._write_json(job, job_value)
        self._uncommitted_seal_jobs.discard(job)
        result = self._seal_result(result_path)
        if result is not None:
            return str(result["status"]), None, result["payload"]
        if self._journal is None:
            self._start_runtime()
        return "PENDING", str(result_path), {"state": "SEALING", "operation_id": operation_id(accepted)}

    def _seal_result(self, path: Path) -> dict[str, object] | None:
        value = self._read_channel_json(path, "subframe seal result")
        if value is not None and (set(value) != {"status", "payload"}
                                  or value["status"] not in {"OK", "ERROR"}):
            raise ParallelHostError("subframe seal result is invalid")
        return value

    def _seal_job(self, path: Path) -> dict[str, object] | None:
        value = self._read_channel_json(path, "subframe seal job")
        if value is None:
            return None
        schema = value.get("schema")
        if schema == "stateframe-parallel-seal-job/1" and set(value) == {"schema", "request"}:
            return value
        if schema == "stateframe-parallel-seal-job/2" and set(value) == {"schema", "request", "operation_id"}:
            request = HostRequest.from_json(value["request"])
            if value["operation_id"] == operation_id(request):
                return value
        raise ParallelHostError("subframe seal job identity is invalid")

    def _discover_seals(self) -> None:
        """Filesystem discovery belongs to channel I/O, never to the lifecycle watchdog."""
        if not self.parallel_root.is_dir() or self.parallel_root.is_symlink():
            return
        for job in sorted(self.parallel_root.glob("*/host/seals/*.json")):
            if (job.name.endswith((".result.json", ".ack.json")) or ".stage-" in job.name or ".direct-stage" in job.name
                    or job in self._sealed_jobs or job in self._seal_ops):
                continue
            try:
                value = self._seal_job(job)
                if value is None:
                    continue
                result_path = job.with_name(job.stem + ".result.json")
                result = self._seal_result(result_path)
                if result is not None:
                    result_bytes = _regular_bytes(result_path, maximum=_MAX_CHANNEL_REQUEST_BYTES,
                                                  label="subframe seal result")
                    digest = hashlib.sha256(result_bytes).hexdigest()
                    if not self._ack_evidence_is_current(result_path, digest):
                        self._reserve_ack_evidence(result_path, digest)
                        self._queue_ack_evidence(result_path, digest)
                        self._publish_pending_ack_evidence()
                    self._sealed_jobs.add(job)
                    continue
                if len(self._seal_ops) >= 32:
                    return
                request = HostRequest.from_json(value["request"])
                self._enqueue_seal(request, job.parents[2])
            except (ProcessInspectionDeferred, JournalBusy, OSError):
                continue  # transient: this job is retried on a later pass
            except (ParallelHostError, ParallelRecordError) as error:
                # Corrupt or superseded job (e.g. an incarnation that was relaunched): it can
                # never seal. Publish its ERROR result so the channel keeps serving and the
                # client's seal request answers deterministically instead of dying with the lane.
                self._fail_seal_job(job, f"seal job is unsatisfiable: {str(error)[:400]}")

    def _fail_seal_job(self, job: Path, message: str) -> None:
        result_path = job.with_name(job.stem + ".result.json")
        value = {"status": "ERROR", "payload": {"error": message[:500]}}
        digest = hashlib.sha256(canonical_json(value) + b"\n").hexdigest()
        ack_needs_publication = not self._ack_evidence_is_current(result_path, digest)
        if ack_needs_publication:
            self._reserve_ack_evidence(result_path, digest)
        try:
            if self._channel_writes is None and self._journal is not None:
                # Before the channel lane exists, seal failures still need the configured
                # journal's ACK. Foreground channel writes remain independent of that lane.
                self._journal.call("write", {"path": str(result_path), "value": value,
                                             "replace": False, "durable": True})
                if ack_needs_publication:
                    self._queue_ack_evidence(result_path, digest)
                    self._publish_pending_ack_evidence()
            else:
                self._write_json(result_path, value)
        except ParallelHostError as error:
            if isinstance(error.__cause__, JournalFault) or not result_path.exists():
                # Nothing durable was published (journal unavailable): retry on a later pass.
                self._protocol_errors.append(f"seal result publication deferred: {str(error)[:300]}")
                return
            try:
                if self._seal_result(result_path) is None:
                    return
            except (JournalBusy, JournalProtocolError, ParallelHostError, OSError) as deferred:
                self._protocol_errors.append(f"seal result verification deferred: {str(deferred)[:300]}")
                return
        except Exception as error:  # journal fault: keep the job discoverable for a retry
            self._protocol_errors.append(f"seal result publication deferred: {type(error).__name__}: {str(error)[:300]}")
            return
        self._sealed_jobs.add(job)
        self._protocol_errors.append(f"{job.name}: {message[:300]}")

    def _fresh_seal_proof(self, operation: SealOperation, epoch: int) -> QuiescenceProof:
        worker = operation.worker
        if (worker.generation != operation.generation or self._closed
                or self._workers.latest.get((worker.call_id, worker.unit_id)) is not worker
                or worker.state != "EXITED" or worker.exited_seq is None):
            raise ProcessInspectionDeferred("seal incarnation or generation is no longer current")
        fence = time.monotonic_ns()
        operation.fence_ns = fence
        generation = self._process_table._generation
        if worker.pgid is not None and self._group_live(worker):
            raise ProcessInspectionDeferred("owned descendants still live; seal remains pending")
        started = (getattr(self._process_table._tick_state, "fresh_started_ns", 0)
                   if worker.pgid is not None else fence)
        finished = time.monotonic_ns()
        if started < fence or generation != self._process_table._generation:
            raise ProcessInspectionDeferred("quiescence scan predates fence or overlaps invalidation")
        proof = QuiescenceProof(worker.incarnation, worker.generation, operation.nonce,
                                epoch, started, finished)
        if not operation.accepts(proof, epoch):
            raise ProcessInspectionDeferred("quiescence proof was invalidated")
        return proof

    @staticmethod
    def _seal_token(operation: SealOperation) -> list[object]:
        return [operation.worker.incarnation, operation.generation, operation.nonce]

    def _advance_seals(self) -> None:
        if not self._seal_ops or self._closed:
            return
        if not self._seal_helpers:
            self._seal_helpers = [Journal(role="seal") for _ in range(2)]
            for helper in self._seal_helpers:
                weakref.finalize(self, helper.close)
        # The main journal's epoch is deliberately not bound here: the EXITED durability a seal
        # depends on was proven at admission (exited_seq) and does not change when that helper
        # restarts, while each operation's proof already binds its own seal helper's epoch.
        operations = tuple(self._seal_ops.values())
        deadline = time.monotonic() + _TICK_WORK_SECONDS
        for offset in range(min(2, len(operations))):
            if time.monotonic() >= deadline:
                break
            operation = operations[self._seal_cursor % len(operations)]
            self._seal_cursor += 1
            if operation.job in self._uncommitted_seal_jobs or operation.phase in {"DONE", "COMMITTING"}:
                continue
            helper = self._seal_helpers[operation.helper]
            if operation.ticket is None:
                # No helper holds this operation: dead stage files go, then stale reservations.
                operation.discard_superseded_stages()
                if self._seal_evicted(operation):
                    continue
                if time.monotonic() < operation.retry_at:
                    continue
            if operation.ticket is not None:
                ticket_token = operation.ticket.payload.get("token")
                result = helper.take(operation.ticket)
                if result is None:
                    continue
                operation.ticket = None
                if result.get("spawn_failed") is True:
                    # The helper exhausted its restart budget without ever starting: nothing was
                    # staged or published. A deterministic ERROR releases the reservation and the
                    # launch gate instead of resubmitting forever.
                    self._complete_seal(operation, {"status": "ERROR", "payload": {
                        "error": f"seal helper unavailable: {str(result.get('error', ''))[:400]}"}})
                    continue
                if result.get("lost"):
                    operation.helper_losses += 1
                    operation.invalidate()  # a retry must obtain a new nonce and owner proof
                    if operation.helper_losses >= _SEAL_HELPER_LOSS_LIMIT:
                        self._complete_seal(operation, {"status": "ERROR", "payload": {
                            "error": "seal helper unavailable: helper loss budget exhausted"}})
                    else:
                        operation.retry_at = time.monotonic() + min(
                            _SEAL_HELPER_BACKOFF_SECONDS * 2 ** (operation.helper_losses - 1), 2.0)
                    continue
                if (ticket_token != self._seal_token(operation) or helper.epoch != operation.helper_epoch
                        or operation.worker.generation != operation.generation):
                    operation.invalidate()
                    continue
                if result.get("ok") is not True:
                    self._complete_seal(operation, {"status": "ERROR", "payload": {"error": result.get("error", "seal failed")}})
                    continue
                if result.get("token") != self._seal_token(operation):
                    operation.invalidate()
                    continue
                if operation.phase == "BUILDING":
                    operation.staged = result["staged"]
                    operation.phase = "STAGED"
                elif operation.phase == "PUBLISHING":
                    self._complete_seal(operation, {"status": "OK", "payload": result["payload"]})
                    continue
            try:
                # Reserve helper epoch before proof. Helpers are lazy-spawned; epoch zero's
                # initial launch is the only allowed change without invalidating a proof.
                epoch = helper.epoch or 1
                proof = self._fresh_seal_proof(operation, epoch)
                operation.proof = proof
                payload = {"request": operation.request.to_json(), "call_root": str(operation.worker.call_root),
                           "token": self._seal_token(operation)}
                if operation.phase == "STAGED":
                    payload["staged"] = operation.staged
                    kind, phase = "seal_publish", "PUBLISHING"
                else:
                    payload["stage"] = str(operation.job.with_name(operation.job.stem + ".stage-" + operation.nonce + ".json"))
                    kind, phase = "seal_build", "BUILDING"
                if not self._lock.acquire(blocking=False):
                    continue
                try:
                    if not operation.accepts(proof, epoch) or self._closed:
                        continue
                    operation.ticket = helper.submit(kind, payload)
                    operation.helper_epoch = epoch
                    operation.phase = phase
                finally:
                    self._lock.release()
            except (ProcessInspectionDeferred, JournalBusy, OSError):
                continue  # UNKNOWN is pending, never a publication permit.

    def _complete_seal(self, operation: SealOperation, result: dict[str, object]) -> None:
        # Journal publication can stall; it runs in the channel lane, not the watchdog.
        operation.staged = {"result": result}
        operation.phase = "COMMITTING"

    def _seal_evicted(self, operation: SealOperation) -> bool:
        """Release a reservation whose worker can no longer be sealed (ERROR result), so the
        admission cap never fills with dead operations. Call with no helper ticket outstanding.
        A latest EXITED worker whose generation moved under the reservation (a cancel on an
        already-exited worker changes nothing) re-adopts it: the seal retries from a fresh
        nonce and fence instead of deferring forever."""
        worker = operation.worker
        if (self._workers.latest.get((worker.call_id, worker.unit_id)) is not worker
                or worker.state != "EXITED" or worker.exited_seq is None):
            self._complete_seal(operation, {"status": "ERROR",
                                            "payload": {"error": "seal incarnation is no longer current"}})
            return True
        if worker.generation != operation.generation:
            operation.generation = worker.generation
            operation.invalidate()
        return False

    def _publish_seal_results(self) -> None:
        for job, operation in tuple(self._seal_ops.items()):
            if operation.phase != "COMMITTING":
                continue
            # An OK result is a publication permit and must still match its generation; an
            # ERROR result publishes nothing and may always release the reservation.
            permit = operation.staged["result"].get("status") == "OK"
            if permit and operation.worker.generation != operation.generation:
                operation.invalidate()
                continue
            with self._lock:
                if permit and operation.worker.generation != operation.generation:
                    operation.invalidate()
                    continue
                staged = operation.staged["result"]
            # Channel I/O (journal round trip) never runs under the host lock.
            try:
                self._write_json(operation.result, staged)
            except ParallelHostError as error:
                if isinstance(error.__cause__, JournalFault) or not operation.result.exists():
                    # Nothing durable was published (journal unavailable): retry on a later pass.
                    self._protocol_errors.append(f"{job.name}: seal publication deferred: {str(error)[:300]}")
                    continue
                # A differing result already exists; it is immutable and authoritative.
                try:
                    if self._seal_result(operation.result) is None:
                        continue
                except (JournalBusy, JournalProtocolError, ParallelHostError, OSError) as deferred:
                    self._protocol_errors.append(f"{job.name}: seal result verification deferred: {str(deferred)[:300]}")
                    continue
                self._protocol_errors.append(f"{job.name}: seal result conflict: {str(error)[:300]}")
            except Exception as error:  # journal fault: retry publication on a later pass
                self._protocol_errors.append(f"{job.name}: seal publication deferred: {str(error)[:300]}")
                continue
            with self._lock:
                if permit and operation.worker.generation != operation.generation:
                    operation.invalidate()
                    continue
                operation.phase = "DONE"
                self._sealed_jobs.add(job)
                self._seal_ops.pop(job, None)
            self._discard_seal_stages(operation)

    @staticmethod
    def _discard_seal_stages(operation: SealOperation) -> None:
        """A published result ends the operation: every stage file minted for it is dead."""
        try:
            for stage in operation.job.parent.glob(operation.job.stem + ".stage-*.json"):
                stage.unlink(missing_ok=True)
        except OSError:
            pass  # best-effort hygiene; the result is already durable
        operation.superseded.clear()

    def _launch(self, request: HostRequest, call_root: Path) -> tuple[str, str, dict[str, object]]:
        requested_epoch = time.time()
        required = {"unit_id", "execution_id", "spec_path", "spec_digest", "run_dir", "incarnation",
                    "start_snapshot_path", "start_manifest_digest", "snapshot_manifest_digest"}
        payload = request.payload
        if set(payload) != required:
            raise ParallelHostError("launch payload keys do not match the host interface")
        unit_id = _valid_unit_id(payload.get("unit_id"))
        execution_id = payload.get("execution_id")
        incarnation = payload.get("incarnation")
        spec_digest = payload.get("spec_digest")
        manifest_digest = payload.get("start_manifest_digest")
        snapshot_manifest_digest = payload.get("snapshot_manifest_digest")
        if not isinstance(execution_id, str) or not execution_id:
            raise ParallelHostError("launch execution_id must be a nonempty string")
        if type(incarnation) is not int or incarnation < 1:
            raise ParallelHostError("launch incarnation must be a positive integer")
        if not isinstance(spec_digest, str) or not _SHA256.fullmatch(spec_digest):
            raise ParallelHostError("launch spec_digest must be a lowercase sha256")
        if not isinstance(manifest_digest, str) or not _SHA256.fullmatch(manifest_digest):
            raise ParallelHostError("launch start_manifest_digest must be a lowercase sha256")
        if not isinstance(snapshot_manifest_digest, str) or not _SHA256.fullmatch(snapshot_manifest_digest):
            raise ParallelHostError("launch snapshot_manifest_digest must be a lowercase sha256")
        spec_path = _inside(Path(str(payload["spec_path"])), call_root, "spec_path")
        run_dir = _inside(Path(str(payload["run_dir"])), call_root, "run_dir")
        snapshot_path = _inside(Path(str(payload["start_snapshot_path"])), call_root, "start_snapshot_path")
        expected_run = call_root / "units" / unit_id
        if run_dir != expected_run:
            raise ParallelHostError("launch run_dir does not match the call's subframe run directory")
        if spec_path != expected_run / "unit-spec.json":
            raise ParallelHostError("launch spec_path does not match the subframe run directory")
        if not spec_path.is_file() or not snapshot_path.is_dir():
            raise ParallelHostError("launch spec or start snapshot is missing")
        if snapshot_path.parent != call_root / "snapshots":
            raise ParallelHostError("launch snapshot is outside the call's snapshot namespace")
        try:
            snapshot = read_snapshot_manifest(snapshot_path, snapshot_manifest_digest)
        except (OSError, WorkerPreparationError) as error:
            raise ParallelHostError(f"launch snapshot manifest is invalid: {error}") from error
        if snapshot.source_digest != manifest_digest:
            raise ParallelHostError("launch start snapshot source digest mismatch")

        launch_payload = dict(payload)
        key = (request.call_id, unit_id, incarnation)
        # Durable records and the prior group's /proc probe are read outside the host lock
        # (a strict scan is tens of ms at load); the lock only re-validates identity.
        recorded = self._recorded_workers(request.call_id, unit_id, call_root)
        with self._lock:
            probed = self._latest_worker(request.call_id, unit_id, call_root, recorded)
        probed_live: bool | None = None
        if (probed is not None and probed.pid is not None and probed.pgid is not None
                and (probed.handle is None or probed.handle.poll() is not None)):
            process_file = probed.run_dir / "proc" / str(probed.incarnation) / "process.json"
            record = _read_json(process_file, "prior subframe process record")
            birth = _recorded_kernel_identity(record)
            probed_live = _recorded_group_live(probed.run_dir, probed.pid, probed.pgid, birth, process_file)
        with self._lock:
            if self._closed:
                raise ParallelHostError("supervisor is shutting down")
            existing = self._worker_for_incarnation(request.call_id, unit_id, incarnation, call_root)
            if existing is not None:
                if existing.launch_payload != launch_payload:
                    raise ParallelHostError("launch incarnation conflicts with its durable launch intent")
                self._workers[key] = existing
                result_id = f"{request.call_id}:{unit_id}:{incarnation}"
                return ("PENDING" if existing.state == "STARTING" else "OK", result_id,
                        {"state": existing.state})
            latest = self._latest_worker(request.call_id, unit_id, call_root, recorded)
            if latest is not None:
                if any(op.worker is latest and op.phase != "DONE" for op in self._seal_ops.values()):
                    raise ParallelHostError("prior incarnation is reserved for sealing")
                if latest.state == "EXITING":
                    raise ParallelHostError("prior incarnation awaits its durable exit commit")
                if latest.execution_id != execution_id:
                    raise ParallelHostError("subframe execution identity changed within one call")
                if (latest.spec_path != spec_path or latest.spec_digest != spec_digest
                        or latest.run_dir != run_dir or latest.snapshot_path != snapshot_path
                        or latest.start_manifest_digest != manifest_digest
                        or latest.snapshot_manifest_digest != snapshot_manifest_digest):
                    raise ParallelHostError("subframe restart changed its frozen execution artifacts")
                if latest.incarnation > incarnation:
                    raise ParallelHostError("launch incarnation is stale")
                if latest.handle is not None and latest.handle.poll() is None:
                    raise ParallelHostError("prior incarnation is still running")
                if latest.pid is not None and latest.pgid is not None:
                    if latest is not probed or probed_live is None:
                        raise ProcessInspectionDeferred("prior incarnation changed during launch; retry pending")
                    if probed_live:
                        raise ParallelHostError("prior incarnation's process group is still running")
            worker = _Worker(request.call_id, unit_id, execution_id, incarnation, call_root, spec_path,
                             spec_digest, run_dir, snapshot_path, manifest_digest, launch_payload,
                             snapshot_manifest_digest=snapshot_manifest_digest, created_epoch=requested_epoch)
            self._persist_worker(worker, launch=True)
            self._workers[key] = worker
            self._start_preparation(worker)
        return "PENDING", f"{request.call_id}:{unit_id}:{incarnation}", {"state": "STARTING"}

    def _worker_state_value(self, worker: _Worker) -> dict[str, object]:
        return {
            "schema": _HOST_STATE_SCHEMA,
            "call_id": worker.call_id,
            "unit_id": worker.unit_id,
            "execution_id": worker.execution_id,
            "incarnation": worker.incarnation,
            "spec_path": str(worker.spec_path),
            "spec_digest": worker.spec_digest,
            "run_dir": str(worker.run_dir),
            "snapshot_path": str(worker.snapshot_path),
            "start_manifest_digest": worker.start_manifest_digest,
            "snapshot_manifest_digest": worker.snapshot_manifest_digest,
            "launch_payload": worker.launch_payload,
            "state": worker.state,
            "terminal": worker.terminal,
            "exit_code": worker.exit_code,
            "error": worker.error,
            "pid": worker.pid,
            "pgid": worker.pgid,
            "created_epoch": worker.created_epoch,
            "started_epoch": worker.started_epoch,
            "finished_epoch": worker.finished_epoch,
            "exited_seq": worker.exited_seq,
            "queued": worker.queued,
            "cancel_requested": worker.cancel_requested,
        }

    def _group_live(self, worker: _Worker) -> bool:
        """Is this worker's process group still live? A worker this supervisor launched is judged
        through its handle (one coalesced scan, and a recycled group number is not its group);
        a worker known only from its record falls back to the pgid probe."""
        check = getattr(worker.handle, "group_live", None)
        if check is None:
            # Shared strict table: the scan stamps this thread, so a seal proof can fence on it.
            if _group_has_live_processes(worker.pgid, getattr(self, "_process_table", None)):
                return True
            process_file = worker.run_dir / "proc" / str(worker.incarnation) / "process.json"
            if worker.pid is not None:
                try:
                    record = _read_json(process_file, "old subframe process record")
                except ParallelHostError as error:
                    raise ProcessInspectionDeferred("recorded ownership unavailable; retry pending") from error
                return _recorded_group_live(worker.run_dir, worker.pid, worker.pgid,
                                            _recorded_kernel_identity(record), process_file)
            return False
        try:
            return bool(check())
        except ContainmentUnverified as error:
            raise ProcessInspectionDeferred(
                f"cannot inspect the subframe process group: {error}; retry pending") from error

    def _require_group_clear(self, worker: _Worker) -> None:
        if worker.pgid is not None and self._group_live(worker):
            raise ParallelHostError("cannot number a subframe exit while its process group is live")

    def _persist_worker(self, worker: _Worker, *, launch: bool = False) -> None:
        if self._journal is not None:
            self._queue_worker_commit(worker, launch=launch)
            return
        # The liveness probe scans the process table: a caller that can verify the group before
        # taking the host lock sets `group_verified`, so status, cancel and tick are not queued
        # behind the scan.
        if worker.state == "EXITED" and worker.exited_seq is None and not worker.group_verified:
            self._require_group_clear(worker)
        with self._lock:
            if worker.state == "EXITED" and worker.exited_seq is None:
                counter_path = worker.call_root / "host" / "exited-sequence.json"
                counter = _read_json(counter_path, "subframe exit sequence") if counter_path.exists() else {"last": 0}
                if set(counter) != {"last"} or type(counter["last"]) is not int or counter["last"] < 0:
                    raise ParallelHostError("subframe exit sequence counter is invalid")
                prior_sequences = []
                for path in (worker.call_root / "host" / "units").glob("*/[0-9]*.json"):
                    if path.stem.isdigit():
                        row = _read_json(path, "prior subframe exit record")
                        sequence = row.get("exited_seq")
                        if type(sequence) is int:
                            prior_sequences.append(sequence)
                if prior_sequences and max(prior_sequences) > counter["last"]:
                    raise ParallelHostError("subframe exit sequence counter trails durable exit records")
                worker.exited_seq = counter["last"] + 1
                self._write_json(counter_path, {"last": worker.exited_seq}, replace_existing=True)
            value = self._worker_state_value(worker)
            if launch:
                self._write_json(worker.launch_path, {"schema": _HOST_STATE_SCHEMA,
                                                 "launch_payload": worker.launch_payload,
                                                 "created_epoch": worker.created_epoch})
            self._write_json(worker.state_path, value, replace_existing=worker.state_path.exists())
            worker.status_view = self._status_value(worker)
            self._workers[(worker.call_id, worker.unit_id, worker.incarnation)] = worker
            if launch:
                worker.launch_committed = True
                worker.launch_commit_event.set()

    def _queue_worker_commit(self, worker: _Worker, *, launch: bool = False) -> None:
        with worker.commit_lock:
            if worker.state == "EXITING":
                return
            if worker.state == "EXITED" and worker.exited_seq is None and not worker.group_verified:
                self._require_group_clear(worker)
            value = self._worker_state_value(worker)
            prior_launch = worker.commit_intent.get("launch") if worker.commit_intent else None
            worker.commit_version += 1
            worker.commit_failures = 0
            worker.commit_retry_at = 0.0
            worker.commit_intent = {"path": str(worker.state_path), "value": value,
                                    "version": worker.commit_version,
                                    "launch": ({"schema": _HOST_STATE_SCHEMA,
                                                "launch_payload": worker.launch_payload,
                                                "created_epoch": worker.created_epoch}
                                               if launch else prior_launch)}
            self._workers[(worker.call_id, worker.unit_id, worker.incarnation)] = worker
            if worker.state == "EXITED" and worker.exited_seq is None:
                worker.state = "EXITING"
                worker.terminal = None
                worker.exit_code = None
                worker.finished_epoch = None
            elif worker.status_view is None:
                worker.status_view = self._status_value(worker)
            self._submit_commit(worker)

    def _submit_commit(self, worker: _Worker) -> None:
        with worker.commit_lock:
            if worker.commit_ticket is not None or worker.commit_intent is None or self._journal is None:
                return
            if time.monotonic() < worker.commit_retry_at:
                return  # bounded backoff after a helper rejection; the lifecycle lane retries
            try:
                worker.commit_ticket = self._journal.submit("commit", worker.commit_intent)
            except JournalBusy:
                pass  # One retained intent per incarnation; saturation never blocks watchdogs.

    def _pump_commits(self) -> None:
        journal = self._journal
        if journal is None:
            return
        with self._lock:  # _WorkerIndex mutates under the host lock; an unlocked view can tear
            workers = tuple(self._workers.values())
        for worker in workers:
            with worker.commit_lock:
                ticket = worker.commit_ticket
                if ticket is not None:
                    result = journal.take(ticket)
                    if result is not None:
                        worker.commit_ticket = None
                        if result.get("lost") is True:
                            self._journal_lost_acks += 1
                        else:
                            self._journal_lost_acks = 0  # any delivered ACK proves the lane is alive
                            self._journal_fault = None
                        if result.get("ok") is True:
                            value = result.get("value")
                            expected = ticket.payload["value"]
                            if not isinstance(value, dict):
                                result = {"ok": False, "error": "commit returned an invalid record"}
                            else:
                                sequence = value.get("exited_seq")
                                if expected["state"] == "EXITED":
                                    valid = (type(sequence) is int and sequence > 0
                                             and value == dict(expected, exited_seq=sequence))
                                else:
                                    valid = value == expected
                                if not valid:
                                    result = {"ok": False, "error": "commit record identity mismatch"}
                        if result.get("ok") is True:
                            if ticket.payload.get("launch") is not None:
                                worker.launch_committed = True
                                worker.launch_commit_event.set()
                        if result.get("ok") is True and ticket.payload.get("version") == worker.commit_version:
                            if value["state"] == "EXITED":
                                worker.exited_seq = value["exited_seq"]
                                worker.terminal = value["terminal"]
                                worker.exit_code = value["exit_code"]
                                worker.error = value["error"]
                                worker.finished_epoch = value["finished_epoch"]
                                worker.state = "EXITED"
                            worker.status_view = self._status_value(worker)
                            worker.commit_intent = None
                        elif result.get("ok") is not True:
                            self._note_commit_rejection(worker, ticket, result)
                if worker.commit_intent is not None and worker.commit_ticket is None:
                    # Journal.fault (restart budget spent) or the host's own loss count: the lane
                    # cannot ACK, so the retained intent is failed instead of resubmitted.
                    fault = getattr(journal, "fault", None)
                    if fault is None and self._journal_lost_acks >= _JOURNAL_LOSS_LIMIT:
                        fault = f"{self._journal_lost_acks} consecutive durable commit acknowledgments lost"
                    if fault is not None:
                        if self._journal_fault is None:
                            self._protocol_errors.append(f"journal fault: {str(fault)[:300]}")
                        self._journal_fault = str(fault)[:300]
                        self._fail_worker_commit(worker, "durable worker commit unavailable: " + self._journal_fault)
                        continue
                self._submit_commit(worker)

    def _note_commit_rejection(self, worker: _Worker, ticket: Ticket, result: dict[str, object]) -> None:
        message = str(result.get("error", "unknown"))[:300]
        if result.get("lost") is True or ticket.payload.get("version") != worker.commit_version:
            # A lost helper, or the ACK of a superseded intent, says nothing about the current one.
            worker.error = "durable worker commit pending: " + message
            if result.get("lost") is True:
                # Never resubmit a lost intent hot: the reactor already spent a restart budget on
                # it. Back off by the run of consecutive losses so a dead lane is probed, not spun.
                step = min(self._journal_lost_acks, _COMMIT_REJECTION_LIMIT)
                worker.commit_retry_at = time.monotonic() + _COMMIT_BACKOFF_SECONDS * 2 ** step
            return
        worker.commit_failures += 1
        if worker.commit_failures < _COMMIT_REJECTION_LIMIT:
            worker.error = "durable worker commit pending: " + message
            if worker.commit_failures > 1:  # one immediate retry absorbs a transient fault
                worker.commit_retry_at = time.monotonic() + _COMMIT_BACKOFF_SECONDS * 2 ** worker.commit_failures
            return
        # The helper rejects this record durably (identity conflict, invalid counter, trailing
        # records): resubmitting it forever would hide the fault. Fail the worker so status
        # reports the rejection; its record cannot become EXITED without a durable sequence.
        self._fail_worker_commit(worker, "durable worker commit rejected: " + message)

    def _fail_worker_commit(self, worker: _Worker, message: str) -> None:
        """Terminally fail the retained intent. A rejected exit commit ends the incarnation: the
        worker becomes EXITED without a durable sequence (status answers the fault, cancel is a
        no-op, and a launch of the next incarnation is accepted once its group is proven gone)
        instead of staying EXITING forever and blocking the unit. A rejected non-terminal record
        only cancels the live leader; its exit is then committed as a fresh intent."""
        worker.commit_intent = None
        worker.error = message
        handle = worker.handle
        if worker.state == "EXITING":
            worker.state = "EXITED"
            worker.exited_seq = None
            worker.terminal = "ENGINE_ERROR"
            if worker.finished_epoch is None:
                worker.finished_epoch = time.time()
        elif handle is not None and worker.state != "EXITED":
            try:
                if handle.poll() is None:
                    handle.cancel("durable worker commit rejected")
            except (OSError, WorkerLaunchError, ContainmentUnverified, ProcessInspectionDeferred) as error:
                self._protocol_errors.append(f"{worker.unit_id}: cancel after commit rejection failed: {error}")
        worker.status_view = dict(self._status_value(worker), state="EXITED", exited_seq=None,
                                  terminal="ENGINE_ERROR", error=worker.error)
        self._protocol_errors.append(f"{worker.unit_id}: {worker.error}")

    def _load_worker(self, call_id: str, unit_id: str, incarnation: int,
                     call_root: Path) -> _Worker | None:
        path = call_root / "host" / "units" / unit_id / f"{incarnation}.json"
        if not path.is_file():
            return None
        value = _read_json(path, "subframe supervisor record")
        expected = {
            "schema", "call_id", "unit_id", "execution_id", "incarnation", "spec_path", "spec_digest",
            "run_dir", "snapshot_path", "start_manifest_digest", "launch_payload", "state", "terminal",
            "exit_code", "error", "pid", "pgid", "created_epoch", "started_epoch", "finished_epoch", "queued",
            "cancel_requested", "snapshot_manifest_digest", "exited_seq",
        }
        if set(value) != expected or value.get("schema") != _HOST_STATE_SCHEMA:
            raise ParallelHostError(f"subframe supervisor record has invalid fields: {path}")
        if (value.get("call_id"), value.get("unit_id"), value.get("incarnation")) != (
                call_id, unit_id, incarnation):
            raise ParallelHostError(f"subframe supervisor record identity mismatch: {path}")
        if (not isinstance(value["snapshot_manifest_digest"], str)
                or not _SHA256.fullmatch(value["snapshot_manifest_digest"])
                or value["state"] not in {"STARTING", "RUNNING", "EXITED"}
                or (value["state"] == "EXITED") != (type(value["exited_seq"]) is int)
                or (type(value["exited_seq"]) is int and value["exited_seq"] < 1)
                or type(value["created_epoch"]) not in (int, float)
                or not math.isfinite(value["created_epoch"])
                or (value["started_epoch"] is not None and
                    (type(value["started_epoch"]) not in (int, float)
                     or not math.isfinite(value["started_epoch"])))):
            raise ParallelHostError(f"subframe supervisor record has invalid status values: {path}")
        worker = _Worker(
            call_id=call_id,
            unit_id=unit_id,
            execution_id=str(value["execution_id"]),
            incarnation=incarnation,
            call_root=call_root,
            spec_path=Path(str(value["spec_path"])),
            spec_digest=str(value["spec_digest"]),
            run_dir=Path(str(value["run_dir"])),
            snapshot_path=Path(str(value["snapshot_path"])),
            start_manifest_digest=str(value["start_manifest_digest"]),
            launch_payload=dict(value["launch_payload"]),
            snapshot_manifest_digest=str(value["snapshot_manifest_digest"]),
            state=str(value["state"]),
            terminal=value["terminal"] if isinstance(value["terminal"], str) else None,
            exit_code=value["exit_code"] if type(value["exit_code"]) is int else None,
            error=value["error"] if isinstance(value["error"], str) else None,
            pid=value["pid"] if type(value["pid"]) is int else None,
            pgid=value["pgid"] if type(value["pgid"]) is int else None,
            created_epoch=float(value["created_epoch"]),
            started_epoch=float(value["started_epoch"]) if type(value["started_epoch"]) in (int, float) else None,
            finished_epoch=float(value["finished_epoch"]) if type(value["finished_epoch"]) in (int, float) else None,
            exited_seq=value["exited_seq"] if type(value["exited_seq"]) is int else None,
            queued=value["queued"] is True,
            cancel_requested=(value["cancel_requested"]
                              if isinstance(value["cancel_requested"], str) else None),
        )
        if worker.started_epoch is not None:
            # The lease marker lives in the subframe's own proc dir and may be gone after a
            # restart; a missing marker is not a reason to forget a durable launch record.
            marker = self._slot_marker(worker)
            if marker is not None and marker[1] != worker.started_epoch:
                raise ParallelHostError("running subframe has no matching slot lease marker")
            worker.slot_marker = marker
        worker.status_view = self._status_value(worker)
        worker.launch_committed = True
        worker.launch_commit_event.set()
        self._workers[(call_id, unit_id, incarnation)] = worker
        return worker

    def _worker_for_incarnation(self, call_id: str, unit_id: str, incarnation: int,
                                call_root: Path) -> _Worker | None:
        key = (call_id, unit_id, incarnation)
        return self._workers.get(key) or self._load_worker(call_id, unit_id, incarnation, call_root)

    def _latest_worker(self, call_id: str, unit_id: str, call_root: Path,
                       recorded: list[_Worker] | None = None) -> _Worker | None:
        """Latest incarnation across memory and disk; pass `recorded` (read outside the host
        lock via _recorded_workers) so the locked section does no file I/O."""
        memory = [worker for (candidate_call, candidate_unit, _), worker in self._workers.items()
                  if candidate_call == call_id and candidate_unit == unit_id]
        if recorded is None:
            recorded = self._recorded_workers(call_id, unit_id, call_root)
        memory.extend(self._workers.get((call_id, unit_id, worker.incarnation)) or worker
                      for worker in recorded)
        return max(memory, key=lambda worker: worker.incarnation) if memory else None

    def _recorded_workers(self, call_id: str, unit_id: str, call_root: Path) -> list[_Worker]:
        """Durable incarnation records not yet in memory (directory glob plus record reads)."""
        loaded: list[_Worker] = []
        directory = call_root / "host" / "units" / unit_id
        if directory.is_dir():
            for path in directory.glob("[0-9]*.json"):
                stem = path.name.removesuffix(".json")
                if stem.isdigit() and not stem.endswith(".launch"):
                    number = int(stem)
                    worker = self._workers.get((call_id, unit_id, number)) or self._load_worker(call_id, unit_id, number, call_root)
                    if worker is not None:
                        loaded.append(worker)
        return loaded

    def _start_preparation(self, worker: _Worker) -> None:
        if worker.preparing or self._closed or worker.cancel_requested is not None or worker.state != "STARTING":
            return
        if self._journal is not None and not worker.launch_committed:
            worker.queued = True
            self._persist_worker(worker)
            return
        # None means the probe is pending; zero is a completed refusal, not a queue.
        if self._capacity_value == 0:
            self._refuse_no_capacity(worker)
            return
        if not self._has_capacity(worker):
            worker.queued = True
            self._persist_worker(worker)
            return
        worker.preparing = True
        worker.queued = False
        worker.error = None
        self._persist_worker(worker)
        thread = threading.Thread(target=self._prepare_and_launch, args=(worker,), daemon=True,
                                  name=f"stateframe-parallel-launch-{worker.unit_id}-{worker.incarnation}")
        worker.thread = thread
        thread.start()

    @contextmanager
    def _launch_admission(self, worker: _Worker) -> Iterator[Callable[[WorkerHandle], None]]:
        # Durable launch preparation precedes this fence; process-record durability follows
        # it. Cancellation sees either the preparation reservation or the installed handle.
        with self._lock:
            if self._closed or worker.cancel_requested is not None:
                raise _LaunchCancelled()

            def install(handle: WorkerHandle) -> None:
                attach = getattr(handle, "attach_tick_table", None)
                if attach is not None:
                    attach(self._process_table)
                worker.pid = handle.pid
                worker.pgid = handle.pgid
                worker.handle = handle

            yield install

    def _prepare_and_launch(self, worker: _Worker) -> None:
        try:
            spec_value = _read_json(worker.spec_path, "subframe spec")
            unit_spec = UnitSpec.from_json(spec_value)
            if unit_spec.digest() != worker.spec_digest:
                raise ParallelHostError("subframe spec digest does not match launch intent")
            if (unit_spec.call_id != worker.call_id or unit_spec.unit_id != worker.unit_id
                    or unit_spec.execution_id != worker.execution_id
                    or unit_spec.start_manifest_digest != worker.start_manifest_digest):
                raise ParallelHostError("subframe spec identity does not match launch intent")
            if worker.snapshot_manifest_digest is None:
                raise ParallelHostError("subframe launch lacks a snapshot manifest digest")
            snapshot = read_snapshot_manifest(worker.snapshot_path, worker.snapshot_manifest_digest)
            if snapshot.source_digest != worker.start_manifest_digest:
                raise ParallelHostError("start snapshot digest does not match launch intent")
            repo_bytes = sum((worker.snapshot_path / relative).stat().st_size for relative in snapshot.files)
            with reserve_preparation(repo_bytes, worker.run_dir) as (ready, reason):
                if not ready:
                    with self._lock:
                        worker.preparing = False
                        if worker.cancel_requested is not None or self._closed:
                            worker.terminal = "CANCELLED"
                            worker.state = "EXITED"
                            worker.error = worker.cancel_requested or "host supervisor closed during subframe preparation"
                            worker.finished_epoch = time.time()
                        elif self._other_live_workers(worker):
                            worker.queued = True
                            worker.state = "STARTING"
                            worker.error = reason
                        else:
                            worker.terminal = "RESOURCES_UNAVAILABLE"
                            worker.state = "EXITED"
                            worker.error = reason
                            worker.finished_epoch = time.time()
                        self._persist_worker(worker)
                    return
                with self._lock:
                    if self._closed or worker.cancel_requested is not None:
                        worker.preparing = False
                        worker.terminal = "CANCELLED"
                        worker.state = "EXITED"
                        worker.error = worker.cancel_requested or "host supervisor closed before subframe launch"
                        worker.finished_epoch = time.time()
                        self._persist_worker(worker)
                        return
                parent_spec = _read_json(self.run_dir / "spec.json", "parent run spec")
                parent_spec["snapshot_manifest_digest"] = snapshot.digest
                prepare_worker(unit_spec, worker.spec_path, worker.run_dir, worker.snapshot_path,
                               parent_spec=parent_spec)
            # A relaunch rebuilds the workspace from the immutable snapshot (a whole-repo copy).
            # Do it before the launch fence so tick, status, cancel and every other worker's
            # deadline enforcement never wait on the copy; the fence below still decides.
            prepare_relaunch(worker.run_dir, worker.incarnation)
            with self._lock:
                if self._closed or worker.cancel_requested is not None:
                    worker.preparing = False
                    worker.terminal = "CANCELLED"
                    worker.state = "EXITED"
                    worker.error = worker.cancel_requested or "host supervisor closed after subframe preparation"
                    worker.finished_epoch = time.time()
                    self._persist_worker(worker)
                    return
            try:
                launch_worker(worker.run_dir, worker.incarnation, True,
                              admission=lambda: self._launch_admission(worker))
            except _LaunchCancelled:
                with self._lock:
                    worker.preparing = False
                    worker.terminal = "CANCELLED"
                    worker.state = "EXITED"
                    worker.error = (worker.cancel_requested
                                    or "host supervisor closed during subframe launch preparation")
                    worker.finished_epoch = time.time()
                    self._persist_worker(worker)
                return
            with self._lock:
                worker.preparing = False
                self._persist_worker(worker)
        except (ParallelHostError, ParallelRecordError, TxnStoreError, WorkerPreparationError,
                WorkerLaunchError, OSError, ValueError, TypeError) as error:
            self._launch_failed(worker, error)
        except Exception as error:
            self._launch_failed(worker, error)

    def _launch_failed(self, worker: _Worker, error: Exception) -> None:
        handle = worker.handle
        reaped_code: int | None = None
        if handle is not None:
            try:
                if handle.poll() is None:
                    handle.cancel("subframe launch state could not be recorded")
                else:
                    reaped_code = handle.reap(timeout=cleanup_seconds())
                    if reaped_code is None:
                        raise ContainmentUnverified(
                            "subframe leader could not be reaped after launch failure")
            except (OSError, WorkerLaunchError, ContainmentUnverified) as cleanup_error:
                error = ParallelHostError(f"{error}; cleanup failed: {cleanup_error}")
        with self._lock:
            worker.preparing = False
            worker.queued = False
            worker.terminal = "ENGINE_ERROR"
            worker.error = f"{type(error).__name__}: {str(error)[:500]}"
            if handle is not None and reaped_code is not None:
                worker.state = "EXITED"
                worker.exit_code = reaped_code
                worker.finished_epoch = time.time()
                try:
                    self._persist_worker(worker)
                except (OSError, ParallelHostError) as persist_error:
                    error = ParallelHostError(
                        f"{error}; cleaned launch state could not be recorded: {persist_error}")
                    worker.state = "RUNNING"
                    worker.exited_seq = None
                    worker.finalize_retry_at = time.monotonic() + 1.0
                    self._protocol_errors.append(str(error))
                else:
                    return
            if handle is not None:
                worker.state = "RUNNING"
                worker.error = f"{type(error).__name__}: {str(error)[:500]}"
                try:
                    self._persist_worker(worker)
                except (OSError, ParallelHostError) as persist_error:
                    self._protocol_errors.append(
                        f"subframe launch failure state could not be persisted: {persist_error}")
                return
            worker.state = "EXITED"
            worker.finished_epoch = time.time()
            self._persist_worker(worker)

    def _other_live_workers(self, worker: _Worker) -> bool:
        # One scan serves every sibling's poll (it used to be one scan per sibling).
        with self._process_table.tick():
            for other in self._workers.values():
                if other is worker or other.call_id != worker.call_id:
                    continue
                if other.handle is not None and other.handle.poll() is None:
                    return True
                if other.preparing:
                    return True
        return False

    def _slot_marker(self, worker: _Worker) -> tuple[int, float] | None:
        marker = worker.run_dir / "proc" / str(worker.incarnation) / "slot-lease.json"
        if not marker.exists() and not marker.is_symlink():
            return None
        row = _read_json(marker, "subframe slot lease marker")
        epoch = row.get("acquired_epoch")
        if (set(row) != {"schema", "slot", "pid", "acquired_epoch"}
                or row["schema"] != "stateframe-subframe-slot-lease/1"
                or type(row["slot"]) is not int or row["slot"] < 0
                or row["pid"] != worker.pid
                or type(epoch) not in (int, float) or not math.isfinite(epoch)
                or epoch < worker.created_epoch or epoch > time.time() + 1):
            raise ParallelHostError("subframe slot lease marker has invalid identity or time")
        return row["slot"], float(epoch)

    def _notice_running(self, worker: _Worker) -> None:
        handle = worker.handle
        if (handle is None or worker.started_epoch is not None or worker.finalizing
                or worker.state in {"EXITED", "EXITING"}):
            return
        marker = self._slot_marker(worker)
        if marker is None:
            return
        if not self._lock.acquire(blocking=False):
            return
        try:
            # The marker read may span finalizer admission or its terminal commit. Revalidate
            # under that same fence before changing state or superseding any retained intent.
            if (worker.handle is not handle or worker.started_epoch is not None or worker.finalizing
                    or worker.state in {"EXITED", "EXITING"}):
                return
            worker.slot_marker = marker
            worker.started_epoch = marker[1]
            worker.state = "RUNNING"
            self._persist_worker(worker)
            start_clock = getattr(handle, "note_running", None)
            if start_clock is not None:
                start_clock(marker[1])  # the run wall starts at the slot lease, not at launch
        finally:
            self._lock.release()

    def _advance_workers(self) -> None:
        # One process-table scan serves every worker's supervision in this pass.
        with self._process_table.tick():
            self._advance_workers_in_tick()

    def _advance_workers_in_tick(self) -> None:
        workers = tuple(self._workers.values())
        for worker in workers:
            self._supervise_recorded_worker(worker)  # handle-less (resumed) workers only
            self._advance_worker(worker)
        self._start_waiting_workers()

    def _advance_worker(self, worker: _Worker) -> None:
        handle = worker.handle
        if handle is None or worker.finalizing or worker.state in {"EXITED", "EXITING"}:
            return
        try:
            try:
                self._notice_running(worker)
            except ParallelHostError as error:
                # A torn or foreign slot-lease marker is not evidence about the process: the worker
                # stays STARTING until a valid marker appears, and the tick must still supervise it.
                if _inspection_deferred(error):
                    raise
                message = f"subframe slot lease marker unreadable: {str(error)[:500]}"
                if worker.error != message:
                    # Logged once per distinct failure, not once per tick.
                    self._protocol_errors.append(f"{worker.unit_id}: {message}")
                worker.error = message
            if worker.cancel_requested is not None:
                handle.cancel(worker.cancel_requested)
            handle.enforce_deadline()
            handle.escalate()
            code = handle.poll()
        except (OSError, ContainmentUnverified, ProcessInspectionDeferred) as error:
            # A failed observation is not evidence that a live worker failed.
            # Retain its slot and unsealable state until inspection succeeds.
            worker.error = f"subframe supervision deferred: {type(error).__name__}: {str(error)[:500]}"
            return
        except (WorkerLaunchError, WorkerPreparationError) as error:
            worker.error = f"{type(error).__name__}: {str(error)[:500]}"
            try:
                handle.cancel("subframe supervision error")
            except (WorkerLaunchError, ContainmentUnverified) as cleanup_error:
                self._protocol_errors.append(str(cleanup_error))
            code = handle.poll()
        if code is None or not self._lock.acquire(blocking=False):
            return
        try:
            # A finalizer may have queued EXITED while this process probe was in flight.
            # Recheck its ownership and EXITING under the admission fence: another finalizer
            # would change the frozen terminal payload while its durable ACK is outstanding.
            if (worker.preparing or worker.finalizing or worker.state in {"EXITED", "EXITING"}
                    or time.monotonic() < worker.finalize_retry_at):
                return
            thread = threading.Thread(target=self._finalize_worker, args=(worker, code), daemon=True,
                                      name=f"stateframe-parallel-finalize-{worker.unit_id}")
            worker.thread = thread
            worker.finalizing = True
        finally:
            self._lock.release()
        try:
            thread.start()
        except RuntimeError:
            worker.thread = None
            worker.finalizing = False
            raise

    def _finalize_worker(self, worker: _Worker, code: int) -> None:
        handle = worker.handle
        if handle is None:
            return
        try:
            reaped_code = handle.reap()
            if reaped_code is None:
                raise ParallelHostError(f"subframe leader {handle.pgid} was not ready to reap")
            code = reaped_code
            marker = self._slot_marker(worker) if worker.started_epoch is None else None
            # Verify the group is clear before taking the host lock (a scan must not queue
            # status, cancel and tick behind it); _persist_worker then trusts this check.
            self._require_group_clear(worker)
            worker.group_verified = True
            with self._lock:
                # A short-lived worker may acquire its slot after the last lifecycle probe.
                # Its terminal owner records that metadata without publishing RUNNING again.
                if worker.started_epoch is None and marker is not None:
                    worker.slot_marker = marker
                    worker.started_epoch = marker[1]
                worker.exit_code = code
                worker.finished_epoch = time.time()
                if handle.timed_out:
                    worker.terminal = "TIMED_OUT"
                elif handle.cancel_reason is not None or worker.cancel_requested is not None:
                    worker.terminal = "CANCELLED"
                else:
                    worker.terminal = _terminal(worker.run_dir) or "ENGINE_ERROR"
                    if worker.terminal != "ENGINE_ERROR":
                        worker.error = None
                if worker.terminal == "ENGINE_ERROR" and not worker.error:
                    worker.error = f"subframe exited with status {code} without a terminal receipt"
                # EXITED is the point at which readers may rely on terminal being final.
                worker.state = "EXITED"
                self._persist_worker(worker)
        except (OSError, ParallelHostError, WorkerLaunchError, ContainmentUnverified) as error:
            with self._lock:
                worker.error = f"subframe finalization failed: {error}"
                worker.terminal = "ENGINE_ERROR"
                # Unknown group state cannot authorize a seal or a replacement incarnation.
                worker.state = "RUNNING"
                worker.finalize_retry_at = time.monotonic() + 1.0
                try:
                    self._persist_worker(worker)
                except (OSError, ParallelHostError) as persist_error:
                    self._protocol_errors.append(
                        f"subframe finalization state could not be persisted: {persist_error}")
        finally:
            worker.group_verified = False
            worker.finalizing = False

    def _start_waiting_workers(self) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            queued = [worker for worker in self._workers.values()
                      if worker.queued and not worker.preparing and worker.state == "STARTING"]
            for worker in queued:
                if self._closed or not self._has_capacity(worker):
                    continue
                self._start_preparation(worker)
        finally:
            self._lock.release()

    @staticmethod
    def _status_value(worker: _Worker) -> dict[str, object]:
        return {"state": worker.state, "terminal": worker.terminal, "exit_code": worker.exit_code,
                "exited_seq": worker.exited_seq, "running_since": worker.started_epoch,
                "finished_epoch": worker.finished_epoch,
                "slot": worker.slot_marker[0] if worker.slot_marker is not None else None,
                "slot_wait_seconds": (max(0.0, worker.started_epoch - worker.created_epoch)
                                      if worker.started_epoch is not None else 0.0)}

    def _status(self, request: HostRequest, call_root: Path) -> dict[str, object]:
        """Read the latest published view. Never acquire locks or refresh observations here."""
        if set(request.payload) != {"unit_id"}:
            raise ParallelHostError("status request must identify one subframe")
        unit_id = _valid_unit_id(request.payload.get("unit_id"))
        worker = self._workers.latest.get((request.call_id, unit_id))
        if worker is None:
            raise ParallelHostError(f"subframe {unit_id!r} has no launch record")
        view = worker.status_view if worker.status_view is not None else self._status_value(worker)
        if view["state"] == "EXITED" and view["exited_seq"] is None:
            failure = view.get("error")
            raise ParallelHostError(failure if isinstance(failure, str) and failure
                                    else "exited subframe has no durable exit sequence")
        return dict(view)

    def _cancel(self, request: HostRequest, call_root: Path) -> dict[str, object]:
        if set(request.payload) != {"unit_id", "reason"}:
            raise ParallelHostError("cancel payload keys do not match the host interface")
        unit_id = _valid_unit_id(request.payload.get("unit_id"))
        reason = request.payload.get("reason")
        if not isinstance(reason, str) or not reason or len(reason) > 300:
            raise ParallelHostError("cancel reason must contain 1 to 300 characters")
        with self._lock:
            worker = self._latest_worker(request.call_id, unit_id, call_root)
            if worker is None:
                raise ParallelHostError(f"subframe {unit_id!r} has no launch record")
            with worker.commit_lock:
                intent = worker.commit_intent
                terminal_pending = worker.state == "EXITING" or (
                    isinstance(intent, dict) and isinstance(intent.get("value"), dict)
                    and intent["value"].get("state") == "EXITED")
            if worker.state == "EXITED" or terminal_pending:
                # The exit is final or awaits its durable commit: a second terminal record would
                # conflict with the first forever, and a finished worker needs no scan fence.
                return {"state": "EXITING" if terminal_pending else "EXITED"}
            worker.generation += 1
            if worker.handle is not None:
                worker.cancel_requested = reason
                worker.error = reason
                self._persist_worker(worker)
                return {"state": "CANCEL_REQUESTED"}
            if worker.preparing:
                worker.cancel_requested = reason
                worker.error = reason
                self._persist_worker(worker)
                return {"state": "CANCEL_REQUESTED"}
            if worker.state == "EXITED" or worker.pid is None or worker.pgid is None:
                if worker.state != "EXITED":
                    self._fence_worker_record(worker, reason)
                return {"state": worker.state}
            if worker.fence_stage in {"PENDING", "SIGNALLING", "TERM", "KILL"}:
                # The lifecycle lane is already fencing the recorded group; the answer is the
                # same as for a first request, and completion is reported through status.
                worker.cancel_requested = reason
                worker.error = reason
                self._persist_worker(worker)
                return {"state": "CANCEL_REQUESTED"}
            worker.cancel_requested = reason
            worker.error = reason
            worker.fence_stage = None  # a failed earlier fence is retried by this request
        # This call only queues a recorded fence. Even its first identity scan and signal belong
        # to the lifecycle lane, so the channel never waits for process inspection or cleanup.
        self._fence_worker_record(worker, reason)
        return {"state": worker.state if worker.state in {"EXITED", "EXITING"} else "CANCEL_REQUESTED"}

    def _seal(self, request: HostRequest, call_root: Path) -> dict[str, object]:
        if set(request.payload) != {"unit_id", "incarnation"}:
            raise ParallelHostError("seal payload keys do not match the host interface")
        unit_id = _valid_unit_id(request.payload.get("unit_id"))
        incarnation = request.payload.get("incarnation")
        if type(incarnation) is not int or incarnation < 1:
            raise ParallelHostError("seal incarnation must be a positive integer")
        worker = self._latest_worker(request.call_id, unit_id, call_root)
        if worker is None:
            raise ParallelHostError(f"subframe {unit_id!r} has no launch record")
        if incarnation != worker.incarnation:
            raise ParallelHostError(f"refusing stale incarnation {incarnation}; current is {worker.incarnation}")
        if worker.state != "EXITED":
            raise ParallelHostError("cannot seal a subframe that has not exited")
        if worker.terminal != "SUBMITTED":
            raise ParallelHostError(f"cannot seal subframe terminal state {worker.terminal!r}")
        if worker.pgid is not None and self._group_live(worker):
            raise ParallelHostError("cannot seal while the subframe process group remains live")
        stage = call_root / "host" / "seals" / f"{unit_id}-{incarnation}.direct-stage.json"
        result = self._build_seal_artifacts(request, call_root, worker, stage=stage)
        if worker.pgid is not None and self._group_live(worker):
            raise ProcessInspectionDeferred("owned descendants appeared before seal publication")
        proposal = Proposal.from_json(_read_json(stage, "staged direct proposal"))
        self._write_json(Path(result["proposal_path"]), proposal.to_json())
        self._write_json(Path(result["proposal_path"]).parent / "digest.json", {"proposal_digest": proposal.digest()})
        return {"proposal_path": result["proposal_path"], "proposal_digest": result["proposal_digest"]}

    def _build_seal_artifacts(self, request: HostRequest, call_root: Path, worker: _Worker,
                              *, stage: Path) -> dict[str, object]:
        """Artifact-only body. The lifecycle owner proves quiescence around this work."""
        unit_id, incarnation = worker.unit_id, worker.incarnation
        try:
            unit_spec = UnitSpec.from_json(_read_json(worker.spec_path, "subframe spec"))
        except ParallelRecordError as error:
            raise ParallelHostError(f"subframe spec is invalid at seal: {error}") from error
        if unit_spec.digest() != worker.spec_digest:
            raise ParallelHostError("subframe spec changed before sealing")
        if (unit_spec.call_id, unit_spec.unit_id, unit_spec.execution_id,
                unit_spec.start_manifest_digest) != (
                worker.call_id, worker.unit_id, worker.execution_id, worker.start_manifest_digest):
            raise ParallelHostError("subframe spec identity changed before sealing")
        if worker.snapshot_manifest_digest is None:
            raise ParallelHostError("subframe launch lacks a snapshot manifest digest")
        try:
            snapshot = read_snapshot_manifest(worker.snapshot_path, worker.snapshot_manifest_digest)
        except (OSError, WorkerPreparationError) as error:
            raise ParallelHostError(f"subframe start snapshot is invalid at seal: {error}") from error
        if snapshot.source_digest != worker.start_manifest_digest:
            raise ParallelHostError("start snapshot changed before sealing")
        proposal_root = call_root / "proposals"
        proposal_dir = proposal_root / unit_id
        if proposal_root.is_symlink() or proposal_dir.is_symlink():
            raise ParallelHostError("proposal artifact directories must not be symlinks")
        proposal_path = proposal_dir / "proposal.json"
        digest_path = proposal_dir / "digest.json"
        blob_root = proposal_dir / "blobs"
        if (proposal_path.is_symlink() or digest_path.is_symlink() or blob_root.is_symlink()):
            raise ParallelHostError("proposal artifacts must not be symlinks")
        if proposal_path.is_file():
            existing = _read_json(proposal_path, "sealed proposal")
            try:
                proposal = Proposal.from_json(existing)
            except ParallelRecordError as error:
                raise ParallelHostError(f"sealed proposal is invalid: {error}") from error
            if (proposal.call_id, proposal.unit_id, proposal.execution_id, proposal.incarnation,
                    proposal.contract_digest) != (worker.call_id, worker.unit_id, worker.execution_id,
                                                   incarnation, unit_spec.contract.digest()):
                raise ParallelHostError("sealed proposal identity does not match the current worker")
            for entry in proposal.entries:
                blob = _regular_bytes(proposal_dir / "blobs" / str(entry.after_sha256),
                                      maximum=_MAX_PROPOSAL_FILE_BYTES, label="sealed proposal blob")
                if hashlib.sha256(blob).hexdigest() != entry.after_sha256:
                    raise ParallelHostError(f"sealed proposal blob is corrupt: {entry.after_sha256}")
            if digest_path.exists():
                digest_record = _read_json(digest_path, "sealed proposal digest")
                if digest_record.get("proposal_digest") != proposal.digest():
                    raise ParallelHostError("sealed proposal digest record does not match the proposal")
            else:
                self._write_json(digest_path, {"proposal_digest": proposal.digest()})
            self._write_json(stage, proposal.to_json())
            return {"stage_path": str(stage), "proposal_path": str(proposal_path), "proposal_digest": proposal.digest()}

        workspace = worker.run_dir / "workspace"
        candidate, materialized = self._materialize_submission(
            worker, unit_spec, workspace, snapshot=snapshot)
        before_paths = set(snapshot.files)
        after_paths = set(_workspace_paths(workspace))
        before_identity = {
            path: ("file", snapshot.file_digests[path]) for path in before_paths
        }
        after_identity: dict[str, tuple[str, str]] = {}
        for path in after_paths:
            known = materialized.get(path)
            if known is None:
                after_identity[path] = _entry_identity(workspace, path)
                continue
            path_identity = _regular_path_identity(workspace / path)
            if path_identity == known.stat_identity:
                after_identity[path] = ("file", known.digest)
                continue
            current_identity = _entry_identity(workspace, path)
            if current_identity != ("file", known.digest):
                raise ParallelHostError(
                    f"workspace file changed after sealed candidate validation: {path}")
            after_identity[path] = current_identity
        allowed_existing = set(unit_spec.contract.owns)
        allowed_creates = set(unit_spec.contract.creates)
        required_creates = allowed_creates
        if not required_creates <= after_paths:
            missing = sorted(required_creates - after_paths)
            raise ParallelHostError(f"subframe did not create declared paths: {missing}")
        deleted = sorted(before_paths - after_paths)
        if deleted:
            raise ParallelHostError(f"subframe deleted files, which subframe proposals do not support: {deleted}")
        changed = []
        for relative in sorted(after_paths):
            if relative not in before_paths:
                changed.append(relative)
                continue
            try:
                current_mode = stat.S_IMODE((workspace / relative).lstat().st_mode)
                start_mode = snapshot.files[relative]
            except OSError as error:
                raise ParallelHostError(f"workspace entry disappeared while sealing: {relative}") from error
            if current_mode & 0o111 != start_mode & 0o111:
                raise ParallelHostError(f"subframe changed file mode on {relative}")
            if before_identity[relative] != after_identity[relative]:
                changed.append(relative)
        if any(path not in allowed_existing | allowed_creates for path in changed):
            refused = sorted(path for path in changed if path not in allowed_existing | allowed_creates)
            raise ParallelHostError(f"subframe changed unowned paths: {refused}")

        entries: list[ProposalEntry] = []
        blobs: list[tuple[str, bytes]] = []
        total_bytes = 0
        for relative in changed:
            path = _inside((workspace / relative).absolute(), workspace, f"changed file {relative}")
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise ParallelHostError(f"changed subframe file disappeared: {relative}") from error
            if not stat.S_ISREG(mode):
                raise ParallelHostError(f"changed subframe path is not a regular file: {relative}")
            mode_bits = stat.S_IMODE(mode)
            before_kind, before_digest = before_identity.get(relative, ("file", None))
            if before_kind != "file":
                raise ParallelHostError(f"subframe changed a symlink, which subframe proposals do not support: {relative}")
            if before_digest is not None:
                before_mode = snapshot.files[relative]
                if mode_bits & 0o111 != before_mode & 0o111:
                    raise ParallelHostError(f"subframe changed executable bits on {relative}")
                mode_bits = before_mode
            elif mode_bits & 0o111:
                raise ParallelHostError(f"subframe created an executable file: {relative}")
            cached = materialized.get(relative)
            if cached is not None and _regular_path_identity(path) != cached.stat_identity:
                if _entry_identity(workspace, relative) != ("file", cached.digest):
                    raise ParallelHostError(
                        f"workspace file changed after sealed candidate validation: {relative}")
            data = (cached.data if cached is not None else
                    _regular_bytes(path, maximum=_MAX_PROPOSAL_FILE_BYTES,
                                   label=f"changed file {relative}"))
            total_bytes += len(data)
            if total_bytes > _MAX_PROPOSAL_BYTES:
                raise ParallelHostError(f"subframe proposal exceeds {_MAX_PROPOSAL_BYTES} bytes")
            digest = cached.digest if cached is not None else hashlib.sha256(data).hexdigest()
            entries.append(ProposalEntry(relative, "modify" if before_digest else "create",
                                         before_digest, digest, mode_bits))
            blobs.append((digest, data))

        contract_digest = unit_spec.contract.digest()
        receipt = self._check_receipt(worker, unit_spec, workspace, candidate)
        proposal = Proposal(request.call_id, unit_id, worker.execution_id, incarnation,
                            tuple(entries), contract_digest, receipt)
        for digest, data in blobs:
            _write_blob(proposal_dir / "blobs" / digest, data, digest)
        self._write_json(stage, proposal.to_json())
        return {"stage_path": str(stage), "proposal_path": str(proposal_path), "proposal_digest": proposal.digest()}

    @staticmethod
    def _materialize_submission(
            worker: _Worker, spec: UnitSpec, workspace: Path, *,
            snapshot: SnapshotManifest | None = None,
    ) -> tuple[dict[str, bytes], dict[str, _MaterializedFile]]:
        terminal = outcomes.terminal_receipt(worker.run_dir)
        if terminal.get("result") != "SUBMITTED" or outcomes.sealed_artifact_digest(terminal) is None:
            raise ParallelHostError("subframe has no verified submitted candidate seal")
        from .runner import RunError, final_candidate_with_file_digests

        try:
            candidate, sealed_digests = final_candidate_with_file_digests(worker.run_dir)
        except (RunError, OSError, ValueError) as error:
            raise ParallelHostError(f"subframe submitted candidate is invalid: {error}") from error
        if not candidate:
            raise ParallelHostError("subframe submitted candidate has no files")
        allowed = set(spec.contract.owns) | set(spec.contract.creates)
        if set(candidate) - allowed:
            raise ParallelHostError(f"subframe submitted unowned paths: {sorted(set(candidate) - allowed)}")
        if any(data is None for data in candidate.values()):
            raise ParallelHostError("subframe submitted a deletion; subframe proposals cannot delete files")
        candidate_bytes = {
            path: data for path, data in candidate.items() if isinstance(data, bytes)
        }
        if len(candidate_bytes) != len(candidate):
            raise ParallelHostError("subframe candidate contains non-byte file content")
        # Per-file content hashes keyed by path (the WorkCopy baseline), not a tree digest.
        if snapshot is None:
            try:
                snapshot = read_snapshot_manifest(
                    worker.snapshot_path, str(worker.snapshot_manifest_digest))
            except (OSError, WorkerPreparationError) as error:
                raise ParallelHostError(f"subframe snapshot is invalid while materializing: {error}") from error
        baseline: dict[str, str] = {}
        for path in spec.contract.owns:
            digest = snapshot.file_digests.get(path)
            if digest is None:
                raise ParallelHostError(f"subframe owned path is missing from its snapshot: {path}")
            baseline[path] = digest
        work = workcopy.WorkCopy(worker.snapshot_path, workspace, baseline, snapshot=worker.snapshot_path)
        try:
            workcopy.apply_files(work, candidate, declared=allowed)
        except (workcopy.WorkCopyError, OSError) as error:
            raise ParallelHostError(f"cannot materialize subframe submission: {error}") from error
        if sealed_digests is None:
            for path, data in candidate_bytes.items():
                if (workspace / path).read_bytes() != data:
                    raise ParallelHostError(f"subframe submitted path was not materialized: {path}")
            return candidate_bytes, {}
        materialized: dict[str, _MaterializedFile] = {}
        for path, data in candidate_bytes.items():
            safe_path = _inside((workspace / path).absolute(), workspace, f"materialized file {path}")
            actual, _digest, identity = _regular_bytes_with_identity(
                safe_path, maximum=_MAX_PROPOSAL_FILE_BYTES,
                label=f"materialized file {path}", hash_bytes=False)
            if actual != data or path not in sealed_digests:
                raise ParallelHostError(f"subframe submitted file digest differs after materialization: {path}")
            materialized[path] = _MaterializedFile(data, sealed_digests[path], identity)
        return candidate_bytes, materialized

    @staticmethod
    def _check_receipt(worker: _Worker, spec: UnitSpec, workspace: Path,
                       candidate: dict[str, bytes] | None = None) -> CheckReceipt | None:
        raw = outcomes.terminal_receipt(worker.run_dir).get("check_receipt")
        if raw is None:
            if spec.contract.check.command_kind != "none" and spec.contract.check.selectors:
                raise ParallelHostError("subframe terminal receipt omits its required worker check receipt")
            return None
        try:
            receipt = CheckReceipt.from_json(raw)
        except ParallelRecordError as error:
            raise ParallelHostError(f"subframe check receipt is invalid: {error}") from error
        if receipt.recipe_digest != spec.contract.check.digest():
            raise ParallelHostError("subframe check receipt recipe differs from the frozen subframe recipe")
        if receipt.policy_digest != spec.policy_digest:
            raise ParallelHostError("subframe check receipt policy differs from the subframe spec")
        if receipt.status == "NOT_RUN":
            if spec.contract.check.command_kind != "none" and spec.contract.check.selectors:
                raise ParallelHostError("subframe check receipt claims NOT_RUN for a required recipe")
            return receipt
        if receipt.status != "PASS":
            return None
        if spec.contract.check.command_kind == "none" or not spec.contract.check.selectors:
            raise ParallelHostError("subframe check receipt claims PASS without a check recipe")
        from .engine.kernel_effects import scan
        from .engine.kernel_unit_policy import _check_ignored, _manifest_digest

        manifest = scan(workspace, exclude=_check_ignored)
        if receipt.product_manifest_digest != _manifest_digest(manifest):
            if candidate is None or not ParallelSupervisor._matches_checked_candidate_modes(
                    receipt, manifest, candidate):
                return None
        evidence = _inside(Path(receipt.raw_evidence_ref), worker.run_dir, "subframe check evidence")
        raw = _regular_bytes(evidence, maximum=_MAX_PROPOSAL_FILE_BYTES, label="subframe check evidence")
        if evidence.stem != hashlib.sha256(raw).hexdigest():
            raise ParallelHostError("subframe check evidence digest differs from its path")
        return receipt

    @staticmethod
    def _matches_checked_candidate_modes(
            receipt: CheckReceipt, manifest: Mapping[str, object],
            candidate: dict[str, bytes]) -> bool:
        """Normalize write-bit differences against modes captured by the checked manifest."""
        checked_modes = receipt.product_manifest_modes
        if (not isinstance(checked_modes, dict)
                or set(checked_modes) != set(manifest)):
            return False
        adjusted = dict(manifest)
        from .engine.kernel_effects import Entry

        for relative, existing in manifest.items():
            checked_mode = checked_modes.get(relative)
            if (not isinstance(existing, Entry) or type(checked_mode) is not int
                    or checked_mode < 0 or checked_mode > 0o7777
                    or existing.mode & ~0o222 != checked_mode & ~0o222):
                return False
            adjusted[relative] = replace(existing, mode=checked_mode)

        for relative, expected_bytes in candidate.items():
            existing = manifest.get(relative)
            if (not isinstance(expected_bytes, bytes) or not isinstance(existing, Entry)
                    or existing.kind != "file"
                    or existing.digest != hashlib.sha256(expected_bytes).hexdigest()
                    or existing.size != len(expected_bytes)):
                return False

        from .engine.kernel_unit_policy import _manifest_digest

        return receipt.product_manifest_digest == _manifest_digest(adjusted)

    def _persist_conflict_error(self, an_dir: Path, answer: HostAnswer, request: HostRequest) -> None:
        self._persist_conflict(an_dir, answer, request)

    def _advance_one(self, worker: _Worker) -> None:
        handle = worker.handle
        if handle is None:
            return
        handle.enforce_deadline()
        handle.escalate()
        code = handle.poll()
        if code is None:
            return
        handle.reap()
        worker.exit_code = code
        worker.finished_epoch = time.time()
        worker.terminal = "TIMED_OUT" if handle.timed_out else "CANCELLED" if handle.cancel_reason else \
            (_terminal(worker.run_dir) or "ENGINE_ERROR")
        worker.state = "EXITED"
        self._persist_worker(worker)

    def cancel_all(self, reason: str) -> None:
        """Fence active subframes with SIGTERM; `reap()` applies the bounded escalation."""
        self._closed = True
        errors = []
        with self._lock:
            workers = list(self._workers.values())
            for worker in workers:
                worker.generation += 1
            # One scan decides which workers are live; the signals below each take their own.
            with self._process_table.tick():
                live = {id(worker) for worker in workers
                        if worker.handle is not None and worker.handle.poll() is None}
            for worker in workers:
                if worker.preparing:
                    worker.cancel_requested = str(reason)[:300]
                    worker.error = str(reason)[:300]
                    self._persist_worker(worker)
                if id(worker) in live:
                    try:
                        worker.handle.cancel(reason)
                        worker.error = str(reason)[:300]
                        self._persist_worker(worker)
                    except (WorkerLaunchError, ContainmentUnverified) as error:
                        errors.append(str(error))
        if errors:
            self._protocol_errors.extend(errors)

    def reap(self) -> None:
        """Wait for cancellation and containment with load-aware cleanup allowances."""
        with self._lock:
            workers = list(self._workers.values())
        grace = cleanup_seconds()
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            live = False
            with self._process_table.tick():  # one scan per pass for every worker's poll
                for worker in workers:
                    handle = worker.handle
                    if handle is not None and handle.poll() is None:
                        live = True
                        handle.escalate(grace=grace)
            if not live:
                break
            time.sleep(0.05)
        with self._process_table.tick():
            gone = {id(worker) for worker in workers
                    if worker.handle is not None and worker.handle.poll() is not None}
        for worker in workers:
            handle = worker.handle
            if handle is not None and id(worker) not in gone:
                handle.escalate(grace=0.0)
                try:
                    if handle.reap(timeout=cleanup_seconds()) is None:
                        raise WorkerLaunchError(f"subframe process {handle.pid} did not exit after SIGKILL")
                except (OSError, WorkerLaunchError, ContainmentUnverified) as error:
                    self._protocol_errors.append(f"subframe reap failed: {error}")
                if handle.poll() is not None:
                    gone.add(id(worker))
            if handle is not None and id(worker) in gone:
                try:
                    handle.reap(timeout=cleanup_seconds())
                except (OSError, WorkerLaunchError, ContainmentUnverified) as error:
                    self._protocol_errors.append(f"subframe containment unverified: {error}")
                handle.reader.join(timeout=cleanup_seconds())
            if worker.thread is not None and worker.thread.is_alive():
                worker.thread.join(timeout=max(130.0, load_tolerant_seconds(
                    "STATEFRAME_PARALLEL_PREPARE_SECONDS") + cleanup_seconds()))
                if worker.thread.is_alive():
                    self._protocol_errors.append(f"subframe preparation thread did not stop: {worker.unit_id}")
        self._advance_workers()
        # Every retained intent is a RUNNING/STARTING -> EXITED transition the legacy sync path
        # had already written; dropping one strands a handle-less worker on resume. Pump until
        # the journal is drained. Only a dead journal or the ACK-progress watchdog (a helper that
        # cannot start or makes no progress; never elapsed time) ends the wait early: the pump
        # then fails the retained intents through the journal fault so the client sees them.
        while self._journal is not None and any(w.commit_intent is not None for w in workers):
            self._pump_commits()
            reactor = getattr(self._journal, "_thread", None)
            if getattr(self._journal, "_closed", False) or (reactor is not None and not reactor.is_alive()):
                self._protocol_errors.append("shutdown retains unacknowledged durable worker commits")
                break
            if self._journal_fault is not None and not any(w.commit_ticket is not None for w in workers):
                break  # faulted journal: nothing is outstanding, nothing may be resubmitted
            time.sleep(0.01)
        self._stop_runtime()

    def _write_fence_record(self, worker: _Worker, path: Path, value: dict[str, object]) -> None:
        """Retain one process-record write; watchdogs only submit and poll its ACK.

        Hydration/artifact-only supervisors have no runtime journal and keep the foreground
        writer. A running lifecycle lane never waits on that writer or the channel lane.
        """
        journal = self._journal
        if journal is None:
            self._write_json(path, value, replace_existing=True)
            return
        payload = {"path": str(path), "value": json.loads(canonical_json(value)),
                   "replace": True, "durable": True}
        digest = hashlib.sha256(canonical_json(payload)).hexdigest()
        with worker.commit_lock:
            if worker.fence_ticket is not None:
                result = journal.take(worker.fence_ticket)
                if result is None:
                    raise JournalBusy("process record awaits its durable ACK; retry pending")
                worker.fence_ticket = None
                if result.get("ok") is True:
                    worker.fence_acked_digest = hashlib.sha256(canonical_json(worker.fence_write)).hexdigest()
                    worker.fence_write = None
                    worker.fence_write_failures = 0
                else:
                    worker.fence_write_failures += 1
                    worker.fence_write_retry_at = time.monotonic() + min(1.0, 0.05 * worker.fence_write_failures)
                    if (worker.fence_write_failures >= 3 or result.get("closed") is True
                            or result.get("error_class") == "ParallelHostError"):
                        raise ParallelHostError(f"process record durability failed: {result.get('error')}")
            if worker.fence_acked_digest == digest and worker.fence_write is None:
                return
            fault = getattr(journal, "fault", None)
            if fault is not None:
                raise ParallelHostError(f"process record durability unavailable: {fault}")
            if worker.fence_write is None:
                worker.fence_write = payload
            if time.monotonic() >= worker.fence_write_retry_at:
                worker.fence_ticket = journal.submit("write", worker.fence_write)
            raise JournalBusy("process record awaits its durable ACK; retry pending")

    def _fence_worker_record(self, worker: _Worker, reason: str, *, advance: bool = False) -> None:
        if worker.state in {"EXITED", "EXITING"}:
            raise ParallelHostError("cannot fence a subframe whose exit is final or being committed")
        if worker.pid is None or worker.pgid is None:
            process_dir = worker.run_dir / "proc" / str(worker.incarnation)
            _inside((process_dir / "process.json").absolute(), worker.run_dir, "old subframe process record")
            if worker.pid is not None or worker.pgid is not None or process_dir.exists():
                raise ParallelHostError("cannot fence an interrupted launch without its complete process identity")
            worker.queued = False
            worker.terminal = "CANCELLED"
            worker.state = "EXITED"
            worker.error = reason[:300]
            worker.finished_epoch = time.time()
            self._persist_worker(worker)
            return
        with self._lock:
            if worker.fence_stage in (None, "FAILED"):
                worker.cancel_requested = reason
                worker.error = reason[:300]
                worker.fence_stage = "PENDING"
                self._persist_worker(worker)
                return
            if not advance:
                return  # channel callers can never advance a fence, including a concurrent one
        record = _read_json(worker.run_dir / "proc" / str(worker.incarnation) / "process.json",
                            "old subframe process record")
        if record.get("pid") != worker.pid or record.get("pgid") != worker.pgid:
            raise ParallelHostError("old subframe process record differs from supervisor identity")
        birth = _recorded_kernel_identity(record)
        process_file = worker.run_dir / "proc" / str(worker.incarnation) / "process.json"
        # One non-blocking lifecycle step; never sleeps. PENDING sends SIGTERM and arms the
        # cleanup window; each later pass pre-filters liveness through the shared table and runs the
        # identity-checked probe only at a transition: escalate to SIGKILL after the first window,
        # give up (manual recovery) after the second, and finalize once the group is proven gone.
        now = time.monotonic()
        with self._lock:
            stage = worker.fence_stage
            if stage == "SIGNALLING":
                return  # another thread is sending the signal for this stage
            if stage == "PENDING":
                worker.fence_stage = "SIGNALLING"
        if stage == "PENDING":
            try:
                _signal_recorded_group(worker.run_dir, worker.pid, worker.pgid, birth, signal.SIGTERM,
                                       process_file, lambda path, value: self._write_fence_record(worker, path, value))
            except BaseException as error:
                with self._lock:
                    worker.fence_stage = "PENDING" if isinstance(error, JournalBusy) or _inspection_deferred(error) else "FAILED"
                raise
            finally:
                self._process_table.invalidate()
            with self._lock:
                worker.fence_stage = "TERM"
                worker.fence_deadline = time.monotonic() + cleanup_seconds()
                worker.error = worker.cancel_requested or reason[:300]
                self._persist_worker(worker)  # a resumed host finds the cancel and resumes the fence
            return
        group_live = self._recorded_group_rows_live(worker.pgid)
        if group_live or _recorded_group_live(worker.run_dir, worker.pid, worker.pgid, birth, process_file):
            if now < worker.fence_deadline:
                return
            if _recorded_group_live(worker.run_dir, worker.pid, worker.pgid, birth, process_file):
                if stage == "KILL":
                    raise _manual_orphan_error(worker.pgid, _live_group_members(worker.pgid))
                with self._lock:
                    worker.fence_stage = "SIGNALLING"
                try:
                    _signal_recorded_group(worker.run_dir, worker.pid, worker.pgid, birth, signal.SIGKILL,
                                           process_file, lambda path, value: self._write_fence_record(worker, path, value))
                except BaseException as error:
                    with self._lock:
                        worker.fence_stage = stage if isinstance(error, JournalBusy) or _inspection_deferred(error) else "FAILED"
                    raise
                finally:
                    self._process_table.invalidate()
                with self._lock:
                    worker.fence_stage = "KILL"
                    worker.fence_deadline = time.monotonic() + cleanup_seconds()
                return
        # A cached empty-group observation is only a hint. Completion requires a scan begun
        # after this fence asked, even when another observer cached the group as gone.
        if self._recorded_group_rows_live(worker.pgid, fresh=True):
            return
        record = _read_json(process_file, "old subframe process record")
        if (record.get("pid") != worker.pid or record.get("pgid") != worker.pgid
                or _recorded_kernel_identity(record) != birth):
            raise ParallelHostError("old subframe process record changed during fencing")
        record["fenced"] = True
        self._write_fence_record(worker, process_file, record)
        with self._lock:
            if worker.state in {"EXITED", "EXITING"}:
                return
            worker.terminal = "CANCELLED"
            worker.state = "EXITED"
            worker.error = worker.cancel_requested or reason[:300]
            worker.finished_epoch = time.time()
            worker.fence_stage = None
            worker.group_verified = True
            try:
                self._persist_worker(worker)
            finally:
                worker.group_verified = False


def _process_table() -> list[tuple[int, int, str, list[str]]]:
    if sys.platform == "linux":
        try:
            # Liveness needs no command line. Kernel identities avoid spawning ps
            # behind the same overloaded scheduler whose children we supervise.
            shared = _RECORDED_OBSERVATION.get()
            table = shared if shared is not None else ProcessTable()
            table.require_complete = True
            rows = shared.tick_snapshot() if shared is not None else table.snapshot()
            return [(row.pid, row.pgid, "Z" if row.zombie else "R", [])
                    for row in rows.values()]
        except (OSError, RuntimeError, ValueError) as error:
            raise ProcessInspectionDeferred("authoritative host process scan failed; retry pending") from error
    for attempt in range(2):
        try:
            result = subprocess.run(["ps", "-ww", "-axo", "pid=,pgid=,stat=,command="], check=False,
                                    capture_output=True, text=True,
                                    timeout=load_tolerant_seconds("STATEFRAME_PARALLEL_INSPECT_SECONDS"))
            break
        except (OSError, subprocess.TimeoutExpired) as error:
            if attempt == 1:
                raise ProcessInspectionDeferred(
                    f"cannot inspect old subframe processes: {type(error).__name__}; retry pending") from error
    if result.returncode != 0:
        raise ParallelHostError(f"process table probe exited {result.returncode}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            raise ParallelHostError("process table probe returned an incomplete row")
        try:
            pid, pgid = int(parts[0]), int(parts[1])
        except ValueError as error:
            raise ParallelHostError("process table probe returned an invalid process identity") from error
        try:
            argv = shlex.split(parts[3]) if len(parts) == 4 else []
        except ValueError:
            # Keep the row for liveness and pidfd group membership even if argv cannot be parsed.
            argv = []
        rows.append((pid, pgid, parts[2], argv))
    if not rows or not any(pid == os.getpid() for pid, _, _, _ in rows):
        raise ProcessInspectionDeferred("process table probe omits its observer; retry pending")
    return rows


def _group_has_live_processes(pgid: int, table: TickProcessTable | None = None) -> bool:
    """Is any non-zombie process still in group ``pgid``?

    With ``table`` the probe is one scan of the supervisor's shared strict table, so the
    calling thread's fresh-scan stamp is set (a seal proof fences on it); a failed scan is
    UNKNOWN (deferred), never an empty table. Without it a private strict scan is used."""
    if table is None:
        return any(group == pgid and not state.startswith("Z") for _, group, state, _ in _process_table())
    try:
        rows = table.snapshot()
        if sys.platform == "darwin":
            return bool(live_group_rows(rows, pgid))
    except (OSError, RuntimeError, ValueError) as error:
        raise ProcessInspectionDeferred("authoritative host process scan failed; retry pending") from error
    return any(row.pgid == pgid and not row.zombie for row in rows.values())


def _recorded_kernel_identity(record: dict[str, object]) -> str:
    value = record.get("kernel_identity")
    if not isinstance(value, str) or not _KERNEL_IDENTITY.fullmatch(value):
        raise ParallelHostError("old subframe process record lacks a precise kernel birth identity; "
                                "inspect and stop any surviving members manually before retrying resume")
    return value


def _process_identity(pid: int) -> tuple[int, str, str, list[str]] | None:
    try:
        table = ProcessTable()
        table.require_complete = True
        row = table.identity(pid)
        if row is None:
            return None
        birth = kernel_start_identity(pid, row=row)
        if birth is None:
            raise RuntimeError("process birth identity is unavailable")
        if row.zombie:
            return row.pgid, "Z", birth, []
        if sys.platform == "linux":
            argv = [os.fsdecode(part) for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                    if part]
        elif sys.platform == "darwin":
            argv = mac_process_argv(pid)
            if argv is None:
                raise RuntimeError("process arguments are unavailable")
        else:
            raise ParallelHostError(f"subframe process fencing is unsupported on {sys.platform}")
        return row.pgid, "Z" if row.zombie else "R", birth, argv
    except (OSError, RuntimeError, ValueError) as error:
        raise ProcessInspectionDeferred(f"cannot inspect subframe process {pid}; retry pending") from error


def _verified_leader(run_dir: Path, pid: int, pgid: int, birth: str, child: Path) -> bool:
    row = _process_identity(pid)
    return bool(row is not None and row[0] == pgid and row[1] != "Z"
                and row[2] == birth and str(child) in row[3] and str(run_dir) in row[3])


def _manual_orphan_error(pgid: int, pids: list[int]) -> ParallelHostError:
    return ParallelHostError(
        f"old subframe group {pgid} has orphaned members {sorted(pids)}; inspect them and, if they belong "
        f"to this run, stop them manually with: kill -TERM {' '.join(str(pid) for pid in sorted(pids))}; "
        "then retry resume")


def _live_group_members(pgid: int) -> list[int]:
    return [pid for pid, group, state, _ in _process_table() if group == pgid and not state.startswith("Z")]


def _verified_live_members(pgid: int, expected: dict[str, str]) -> list[tuple[int, str]]:
    live = set(_live_group_members(pgid))
    verified = []
    unknown = []
    # Birth-bound descendants retain ownership across setsid/setpgid and reparenting.
    # The original group still rejects unowned members; a reused detached PID is unrelated.
    for member in live | {int(raw_pid) for raw_pid in expected}:
        identity = _process_identity(member)
        birth = expected.get(str(member))
        if identity is None and member in live:
            raise ProcessInspectionDeferred("process disappeared during family inspection; retry pending")
        if identity is not None and identity[1] == "Z":
            continue
        if (identity is not None and birth is not None and identity[2] == birth
                and (member != pgid or identity[0] == pgid)):
            verified.append((member, birth))
        elif member in live:
            unknown.append(member)
    if unknown:
        raise _manual_orphan_error(pgid, unknown)
    return verified


def _signal_one(pid: int, pgid: int, birth: str, sig: int) -> bool:
    if sys.platform == "linux":
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise ParallelHostError("Linux pidfd signalling is unavailable")
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            return False
        try:
            current = _process_identity(pid)
            if current is None or current[0] != pgid or current[1] == "Z" or current[2] != birth:
                return False
            try:
                signal.pidfd_send_signal(descriptor, sig)
            except ProcessLookupError:
                return False
            return True
        finally:
            os.close(descriptor)
    current = _process_identity(pid)
    if current is None or current[0] != pgid or current[1] == "Z" or current[2] != birth:
        return False
    # macOS cannot pin a recovered non-child PID between verification and signal.
    raise ParallelHostError(
        f"subframe containment unverified: cannot safely signal recovered process {pid} without pidfd")


def _signal_recorded_group(run_dir: Path, pid: int, pgid: int, birth: str, sig: int,
                           process_file: Path,
                           persist: Callable[[Path, dict[str, object]], None] | None = None) -> None:
    if (type(pid) is not int or type(pgid) is not int or pid != pgid or pid < 1
            or pid == os.getpid() or pgid == os.getpgrp()
            or not isinstance(birth, str) or not _KERNEL_IDENTITY.fullmatch(birth)):
        raise ParallelHostError("recorded subframe pid/pgid kernel identity is invalid")
    record = _read_json(process_file, "old subframe process record")
    if record.get("kernel_identity") != birth or record.get("pid") != pid or record.get("pgid") != pgid:
        raise ParallelHostError("old subframe process record differs from its kernel identity")
    known = record.get("verified_members")
    if not isinstance(known, dict) or str(pid) in known or any(not isinstance(key, str) or not key.isdecimal()
                                            or len(key) > 10 or not key.isascii() or key.startswith("0")
                                            or not isinstance(value, str)
                                            or not _KERNEL_IDENTITY.fullmatch(value)
                                            for key, value in known.items()):
        raise ParallelHostError("old subframe member identities are invalid")
    spec_path = run_dir / "spec.json"
    if spec_path.is_symlink() or not spec_path.is_file():
        raise ParallelHostError(f"old subframe spec is missing or unsafe: {spec_path}")
    spec = _read_json(spec_path, "old subframe spec")
    child = Path(str(spec.get("engine_dir", ""))).parent / "_child.py"
    leader = _verified_leader(run_dir, pid, pgid, birth, child)
    if leader:
        for member in _live_group_members(pgid):
            if member == pid:
                continue
            identity = _process_identity(member)
            if identity is None or identity[0] != pgid or identity[1] == "Z":
                raise ProcessInspectionDeferred("process changed during family discovery; retry pending")
            if str(member) in known and known[str(member)] != identity[2]:
                raise _manual_orphan_error(pgid, [member])
            known[str(member)] = identity[2]
        if not _verified_leader(run_dir, pid, pgid, birth, child):
            raise ParallelHostError(f"refusing to signal subframe group {pgid}; leader exited during inspection")
        # Persist before the first signal so a crash retains escalation targets.
        record["verified_members"] = known
        if persist is None:
            _write_json(process_file, record, replace_existing=True)
    if persist is not None:
        # Recovery may see a linked identity record before its helper ACK. Even after the
        # leader exits, consume/revalidate that intent before using any recorded target.
        persist(process_file, record)
    active = _verified_live_members(pgid, {**known, str(pid): birth})
    if not active:
        return  # No birth-verified live targets remain, including detached descendants.
    try:
        for member, member_birth in active:
            if member != pid:
                current = _process_identity(member)
                if current is not None and current[2] == member_birth:
                    _signal_one(member, current[0], member_birth, sig)
        if any(member == pid for member, _ in active):
            _signal_one(pid, pgid, birth, sig)
    except (PermissionError, OSError) as error:
        raise ParallelHostError(f"cannot signal old subframe process group {pgid}: {error}") from error


def _recorded_group_live(run_dir: Path, pid: int, pgid: int, birth: str,
                         process_file: Path) -> bool:
    record = _read_json(process_file, "old subframe process record")
    if record.get("pid") != pid or record.get("pgid") != pgid or record.get("kernel_identity") != birth:
        raise ParallelHostError("old subframe process record differs from its kernel identity")
    known = record.get("verified_members")
    if not isinstance(known, dict) or str(pid) in known:
        raise ParallelHostError("old subframe member identities are invalid")
    for raw_pid, member_birth in known.items():
        if (not isinstance(raw_pid, str) or not raw_pid.isdecimal()
                or len(raw_pid) > 10 or not raw_pid.isascii() or raw_pid.startswith("0")
                or not isinstance(member_birth, str) or not _KERNEL_IDENTITY.fullmatch(member_birth)):
            raise ParallelHostError("old subframe member identities are invalid")
    if _verified_live_members(pgid, {**known, str(pid): birth}):
        return True
    # The former handle may have retained detached/reparented descendants absent from
    # this record. An empty saved set or PGID cannot recover that lost family authority.
    # Only the handle's completed containment sweep can certify the whole family gone.
    try:
        proof = _read_json(process_file.with_name("family-exited.json"), "subframe family exit proof")
    except (OSError, ParallelHostError) as error:
        raise ProcessInspectionDeferred("recorded family authority unavailable; retry pending") from error
    if proof != {"pid": pid, "pgid": pgid, "kernel_identity": birth}:
        raise ProcessInspectionDeferred("recorded family exit proof differs; retry pending")
    return False


def fence_and_reap(run_dir: Path, descriptor: object) -> None:
    """Terminate every old incarnation named by a validated recovery descriptor."""
    run_dir = Path(run_dir).resolve()
    executions = getattr(descriptor, "executions", None)
    if not isinstance(executions, tuple):
        raise ParallelHostError("parallel recovery descriptor has no executions tuple")
    pending: dict[int, tuple[Path, int, int, str, Path]] = {}
    parallel_root = run_dir / "parallel"
    if parallel_root.is_symlink():
        raise ParallelHostError("parallel recovery root is a symlink")
    call_id = getattr(descriptor, "call_id", None)
    if not isinstance(call_id, str) or not _CALL_ID.fullmatch(call_id):
        raise ParallelHostError("parallel recovery descriptor has no call_id")
    expected_runs: dict[str, Path] = {}
    for execution in executions:
        raw_run = Path(str(getattr(execution, "run_dir", "")))
        execution_run = _inside(raw_run, parallel_root, "recovery execution run_dir")
        unit_id = getattr(execution, "unit_id", None)
        if not isinstance(unit_id, str) or not _UNIT_ID.fullmatch(unit_id):
            raise ParallelHostError("subframe recovery identity is invalid")
        expected_run = parallel_root / call_id / "units" / unit_id
        if execution_run != expected_run:
            raise ParallelHostError("recovery execution run_dir does not match call and subframe identity")
        if not execution_run.is_dir() or execution_run.is_symlink():
            raise ParallelHostError("recovery execution run_dir is missing or unsafe")
        expected_runs[unit_id] = execution_run
        last_incarnation = getattr(execution, "last_incarnation", None)
        if type(last_incarnation) is not int or last_incarnation < 0:
            raise ParallelHostError("recovery execution incarnation is invalid")
    units_root = parallel_root / call_id / "units"
    if units_root.is_symlink():
        raise ParallelHostError("recovery subframe directory is a symlink")
    unrecorded: list[str] = []
    for execution_run in sorted(units_root.iterdir()) if units_root.is_dir() else ():
        if execution_run.is_symlink() or not execution_run.is_dir():
            raise ParallelHostError(f"recovery subframe artifact is unsafe: {execution_run}")
        _valid_unit_id(execution_run.name)
        proc_root = execution_run / "proc"
        if proc_root.is_symlink():
            raise ParallelHostError(f"recovery subframe process directory is unsafe: {proc_root}")
        if proc_root.exists() and not proc_root.is_dir():
            raise ParallelHostError(f"recovery subframe process directory is invalid: {proc_root}")
        incarnations = sorted(proc_root.iterdir()) if proc_root.is_dir() else ()
        if execution_run.name not in expected_runs and incarnations:
            unrecorded.append(execution_run.name)
        for process_dir in incarnations:
            if (process_dir.is_symlink() or not process_dir.is_dir()
                    or not process_dir.name.isdecimal() or int(process_dir.name) < 1):
                raise ParallelHostError(f"old subframe incarnation path is invalid: {process_dir}")
            process_file = process_dir / "process.json"
            if not process_file.is_file() or process_file.is_symlink():
                raise ParallelHostError(
                    f"old subframe incarnation lacks a complete process identity; inspect: {process_dir}")
            launch_intent = process_dir / "launch-intent.json"
            if launch_intent.exists() or launch_intent.is_symlink():
                intent = _read_json(launch_intent, "old subframe launch intent")
                if (set(intent) != {"schema", "incarnation", "run_dir", "command"}
                        or intent.get("schema") != "stateframe-subframe-launch-intent/1"
                        or intent.get("incarnation") != int(process_dir.name)
                        or intent.get("run_dir") != str(execution_run)
                        or not isinstance(intent.get("command"), list)
                        or any(not isinstance(arg, str) for arg in intent["command"])):
                    raise ParallelHostError(f"old subframe launch intent is invalid: {launch_intent}")
                spec = _read_json(execution_run / "spec.json", "old subframe spec")
                child = str(Path(str(spec.get("engine_dir", ""))).parent / "_child.py")
                if intent["command"] != [spec.get("parallel_engine_python"), "-B", child,
                                          str(execution_run)]:
                    raise ParallelHostError(f"old subframe launch intent differs from its spec: {launch_intent}")
            process_file = _inside(process_file.absolute(), execution_run, "old subframe process record")
            if process_file.is_symlink() or not process_file.is_file():
                raise ParallelHostError(f"old subframe process record is unsafe: {process_file}")
            try:
                incarnation = int(process_file.parent.name)
            except ValueError as error:
                raise ParallelHostError(f"old subframe incarnation path is invalid: {process_file}") from error
            record = _read_json(process_file, "old subframe process record")
            pid, pgid = record.get("pid"), record.get("pgid")
            if (set(record) not in ({"incarnation", "pid", "pgid", "started_at", "started_epoch"},
                                    {"incarnation", "pid", "pgid", "started_at", "started_epoch",
                                     "kernel_identity", "verified_members"},
                                    {"incarnation", "pid", "pgid", "started_at", "started_epoch",
                                     "kernel_identity", "verified_members", "fenced"})
                    or record.get("incarnation") != incarnation
                    or type(pid) is not int or type(pgid) is not int or pid != pgid
                    or ("fenced" in record and type(record["fenced"]) is not bool)):
                raise ParallelHostError(f"old subframe process identity is invalid: {process_file}")
            birth = _recorded_kernel_identity(record)
            if pgid in pending:
                raise ParallelHostError(f"duplicate old subframe process group {pgid}")
            pending[pgid] = (execution_run, pid, pgid, birth, process_file)
    groups = list(pending.values())
    for execution_run, pid, pgid, birth, process_file in groups:
        _signal_recorded_group(execution_run, pid, pgid, birth, signal.SIGTERM, process_file)
    deadline = time.monotonic() + cleanup_seconds()
    while time.monotonic() < deadline and any(_recorded_group_live(*group) for group in groups):
        time.sleep(0.05)
    for execution_run, pid, pgid, birth, process_file in groups:
        if _recorded_group_live(execution_run, pid, pgid, birth, process_file):
            _signal_recorded_group(execution_run, pid, pgid, birth, signal.SIGKILL, process_file)
    deadline = time.monotonic() + cleanup_seconds()
    while time.monotonic() < deadline and any(_recorded_group_live(*group) for group in groups):
        time.sleep(0.05)
    live = [pgid for execution_run, pid, pgid, birth, process_file in groups
            if _recorded_group_live(execution_run, pid, pgid, birth, process_file)]
    if live:
        pgid = live[0]
        raise _manual_orphan_error(pgid, _live_group_members(pgid))
    for _, _, _, _, process_file in groups:
        record = _read_json(process_file, "old subframe process record")
        if record.get("fenced") is not True:
            record["fenced"] = True
            _write_json(process_file, record, replace_existing=True)
    if unrecorded:
        raise ParallelHostError(f"recovery descriptor omitted recorded subframes: {sorted(unrecorded)}")
