"""Cooperative endpoint leases, host resource checks, and durable subframe send budgets."""
from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
from typing import Iterator, Sequence

from .endpoint import Endpoint, EndpointError
from .engine.kernel_check_permits import acquire as check_permit
from .engine.kernel_safe_paths import safe_home
from .engine.kernel_exec_tracker import load_tolerant_seconds

__all__ = ("ResourceError", "SendLedger", "SendLedgerError", "SlotHandoffDurabilityError", "SlotLease",
           "SlotPoolError", "check_permit", "idle_slots", "lease_slot", "machine_ok", "parse_slot_pool",
           "reserve_preparation", "slot_candidates")


# Tests can override this root; otherwise resolve HOME when a lease is acquired.
LEASE_ROOT: Path | None = None
SEND_LEDGER_SCHEMA = "stateframe-send-ledger/1"
_MEMORY_PRESSURE = re.compile(r"System-wide memory free percentage:\s*([0-9]+(?:\.[0-9]+)?)%")
_SLOT_HANDOFF_SCHEMA = "stateframe-slot-lease-handoff/1"
_SLOT_HANDOFF_MAX_ATTEMPTS = 8
_SLOT_CLAIM_ENV = "SF_SUBFRAME_SLOT_CLAIM"
_SLOT_CLAIM_VERSION = "durable-v1"
_KERNEL_IDENTITY = re.compile(r"^(?:linux:[0-9a-fA-F-]{36}:\d+|darwin:\d+:\d+)$")
_MAX_SLOT_HANDOFF_BYTES = 64 << 10
_PREPARATION_LOCK = threading.Lock()
_PREPARING_REPO_BYTES: dict[int, int] = {}


class ResourceError(RuntimeError):
    """A host resource probe or lease could not be trusted."""


class SlotHandoffDurabilityError(OSError):
    """A slot handoff could not be durably published within its retry bound."""

    def __init__(self, slot: int, attempts: int, last_error: BaseException):
        self.slot = slot
        self.attempts = attempts
        self.last_error = last_error
        self.cause_type = type(last_error).__name__
        super().__init__(errno.EIO,
                         f"slot {slot} handoff was not durable after {attempts} attempts "
                         f"({self.cause_type})")


class SlotPoolError(ValueError):
    """A configured subframe slot pool is malformed."""


class SendLedgerError(ValueError):
    """A subframe send ledger is invalid or conflicts with a prior settlement."""


class SlotLease:
    """A slot flock released on close or handed to the host's family-exit proof."""

    def __init__(self, slot: int, path: Path, descriptor: int):
        self.slot = slot
        self.path = path
        self._descriptor = descriptor
        self._handoff_complete = False

    def close(self) -> None:
        if self._descriptor < 0:
            return
        descriptor, self._descriptor = self._descriptor, -1
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def handoff_to_host(self) -> None:
        """Keep this slot claimed until the worker host publishes verified family quiescence.

        Worker launches publish the durable claim during slot acquisition, before the
        engine can start work. This call revalidates it after unverified cleanup and
        keeps the flock descriptor open while proof lookup or marker sync failures retry.
        The independent claim survives this process exiting until the exact host proof
        is present.
        """
        if self._descriptor < 0:
            if self._handoff_complete:
                return
            raise ResourceError("slot lease is already closed")
        retry_delay = 0.05
        retry_reported = False
        attempts = 0
        while True:
            attempts += 1
            try:
                proof_path, identity, incarnation = _current_worker_family_proof()
                _publish_slot_handoff(self.path, self.slot, proof_path, identity, incarnation)
                break
            except BaseException as error:  # noqa: BLE001 - failure must not release the flock
                if attempts >= _SLOT_HANDOFF_MAX_ATTEMPTS:
                    raise SlotHandoffDurabilityError(self.slot, attempts, error) from error
                if not retry_reported:
                    try:
                        logging.getLogger(__name__).error(
                            "slot lease handoff is retrying (slot=%d, error=%s)",
                            self.slot, type(error).__name__)
                    except BaseException:  # noqa: BLE001 - logging cannot release the flock
                        pass
                    retry_reported = True
                retry_at = time.monotonic() + retry_delay
                while True:
                    remaining = retry_at - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        time.sleep(remaining)
                    except BaseException:  # noqa: BLE001 - preserve backoff despite signals
                        continue
                retry_delay = min(retry_delay * 2, 5.0)
        # Do not issue LOCK_UN or close the descriptor. It remains held until this
        # process exits; the durable host handoff keeps the slot reserved afterwards.
        self._handoff_complete = True
        self._descriptor = -1

    def __enter__(self) -> "SlotLease":
        if self._descriptor < 0:
            raise ResourceError("slot lease is already closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _slot_id(entry: object) -> tuple[int, bool] | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("id")
    if type(value) is not int:
        value = entry.get("id_slot")
    processing = entry.get("is_processing")
    if type(value) is not int or value < 0 or type(processing) is not bool:
        return None
    return value, processing


def idle_slots(endpoint: str) -> list[int] | None:
    """Return unique idle server slots, or None when GET /slots is unsupported."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a nonempty URL")
    try:
        slots = Endpoint(endpoint, "stateframe-parallel-slots", timeout_s=2.0).get_slots()
    except (EndpointError, ValueError):
        return None
    if slots is None:
        return None
    by_id: dict[int, bool] = {}
    duplicates: set[int] = set()
    for entry in slots:
        parsed = _slot_id(entry)
        if parsed is None:
            continue
        slot, processing = parsed
        if slot in by_id:
            duplicates.add(slot)
        else:
            by_id[slot] = processing
    return sorted(slot for slot, processing in by_id.items()
                  if not processing and slot not in duplicates)


def parse_slot_pool(value: object, *, source: str = "--parallel-slots") -> tuple[int, ...] | None:
    """Parse an ordered pool from CLI/profile text or a serialized run-spec array."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
        if not value.strip():
            raise SlotPoolError(f"{source} must be a comma-separated list of unique non-negative slot ids")
        slots: list[object] = []
        for part in parts:
            item = part.strip()
            if not item or not item.isascii() or not item.isdecimal():
                raise SlotPoolError(f"{source} must be a comma-separated list of unique non-negative slot ids")
            try:
                slots.append(int(item, 10))
            except ValueError as error:
                raise SlotPoolError(
                    f"{source} must be a comma-separated list of unique non-negative slot ids") from error
    elif isinstance(value, (list, tuple)):
        slots = list(value)
    else:
        raise SlotPoolError(f"{source} must be a comma-separated list of unique non-negative slot ids")

    if not slots:
        raise SlotPoolError(f"{source} must include at least one slot id")
    if any(type(slot) is not int or slot < 0 for slot in slots):
        raise SlotPoolError(f"{source} must contain only non-negative integer slot ids")
    if len(set(slots)) != len(slots):
        raise SlotPoolError(f"{source} must not contain duplicate slot ids")
    return tuple(slots)


def slot_candidates(pool: Sequence[int] | None, idle: Sequence[int] | None, parent_slot: object,
                    *, fallback_capacity: int) -> list[int]:
    """Build lease candidates, preserving pool order and the permitted parent-slot handoff."""
    if type(fallback_capacity) is not int or fallback_capacity < 0:
        raise ValueError("fallback_capacity must be a non-negative integer")
    parsed_pool = parse_slot_pool(pool, source="--parallel-slots") if pool is not None else None
    if idle is None:
        idle_ids = None
    else:
        idle_ids = tuple(idle)
        if any(type(slot) is not int or slot < 0 for slot in idle_ids):
            raise ValueError("idle slots must be non-negative integer ids")
        if len(set(idle_ids)) != len(idle_ids):
            raise ValueError("idle slots must not contain duplicate ids")

    if parsed_pool is None:
        candidates = list(range(fallback_capacity)) if idle_ids is None else list(idle_ids)
    elif idle_ids is None:
        candidates = list(parsed_pool)
    else:
        idle_set = set(idle_ids)
        candidates = [slot for slot in parsed_pool if slot in idle_set]

    if (type(parent_slot) is int and parent_slot >= 0
            and (parsed_pool is None or parent_slot in parsed_pool)):
        candidates = [parent_slot, *(slot for slot in candidates if slot != parent_slot)]
    return candidates


def _secure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError as error:
            raise ResourceError(f"cannot inspect slot lease directory {current}: {error}") from error
        if not stat.S_ISDIR(mode):
            raise ResourceError(f"slot lease path is not a real directory: {current}")


def _lease_root() -> Path:
    if LEASE_ROOT is not None:
        return Path(LEASE_ROOT)
    home = safe_home()
    if home is None:
        raise ResourceError("home directory is unavailable for slot leases")
    return home / ".stateframe" / "slot-leases"


def _record_slot_marker(marker: Path, slot: int) -> None:
    """Publish the slot-lease marker atomically: a supervisor reading it concurrently sees the whole
    record or no file at all, never a partial write. The marker is written once per incarnation."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if marker.exists() or marker.is_symlink():
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(marker))
    encoded = json.dumps({"schema": "stateframe-subframe-slot-lease/1", "slot": slot, "pid": os.getpid(),
                          "acquired_epoch": time.time()},
                         sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
    try:
        with os.fdopen(temporary_fd, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, marker)
        directory_fd = os.open(marker.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            os.unlink(temporary)


def _read_small_json(path: Path) -> object | None:
    """Read a bounded regular JSON file without following a final-component symlink."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | nofollow)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_SLOT_HANDOFF_BYTES:
            return None
        chunks: list[bytes] = []
        size = 0
        while size <= _MAX_SLOT_HANDOFF_BYTES:
            chunk = os.read(descriptor, min(1024, _MAX_SLOT_HANDOFF_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > _MAX_SLOT_HANDOFF_BYTES:
            return None
        return json.loads(b"".join(chunks))
    except (OSError, ValueError, RecursionError):
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _worker_family_proof() -> tuple[Path, dict[str, object], int]:
    raw_incarnation = os.environ.get("SF_SUBFRAME_INCARNATION")
    if raw_incarnation is None or not raw_incarnation.isdecimal() or int(raw_incarnation) < 1:
        raise ValueError("worker incarnation is unavailable")
    incarnation = int(raw_incarnation)
    process_dir = Path.cwd() / "proc" / raw_incarnation
    if process_dir.is_symlink() or not process_dir.is_dir():
        raise ValueError("worker process directory is unsafe")
    record = _read_small_json(process_dir / "process.json")
    if not isinstance(record, dict):
        raise ValueError("worker process record is unavailable or invalid")
    pid, pgid = record.get("pid"), record.get("pgid")
    kernel_identity = record.get("kernel_identity")
    if (record.get("incarnation") != incarnation or type(pid) is not int or pid != os.getpid()
            or type(pgid) is not int or pgid != pid or not isinstance(kernel_identity, str)
            or not _KERNEL_IDENTITY.fullmatch(kernel_identity)):
        raise ValueError("worker process record does not match this lease owner")
    identity: dict[str, object] = {"pid": pid, "pgid": pgid, "kernel_identity": kernel_identity}
    return process_dir / "family-exited.json", identity, incarnation


def _current_worker_family_proof() -> tuple[Path, dict[str, object], int]:
    return _worker_family_proof()


def _worker_family_proof_for_slot_claim() -> tuple[Path, dict[str, object], int]:
    """Wait briefly for the launching host's durable identity record to appear."""
    raw_incarnation = os.environ.get("SF_SUBFRAME_INCARNATION")
    if raw_incarnation is None or not raw_incarnation.isdecimal() or int(raw_incarnation) < 1:
        raise ValueError("worker incarnation is unavailable")
    process_record = Path.cwd() / "proc" / raw_incarnation / "process.json"
    deadline = time.monotonic() + 10.0
    retry_delay = 0.005
    while True:
        try:
            process_record.lstat()
        except FileNotFoundError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ValueError("worker process identity was not published")
            time.sleep(min(retry_delay, remaining))
            retry_delay = min(retry_delay * 2, 0.1)
        except OSError as error:
            raise ValueError("worker process identity cannot be inspected") from error
        else:
            return _worker_family_proof()


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_slot_claim(lock_path: Path, slot: int, proof_path: Path,
                        identity: dict[str, object], incarnation: int) -> None:
    """Durably claim a worker's slot before returning it to slot-using code.

    Retrying an exact visible record repeats its directory durability barrier. A
    conflicting record is never replaced because it may represent a live family.
    """
    marker = lock_path.with_name(f"{lock_path.stem}.handoff.json")
    _secure_directory(marker.parent)
    value = {"schema": _SLOT_HANDOFF_SCHEMA, "slot": slot, "incarnation": incarnation,
             "proof_path": str(proof_path), "identity": identity}
    if marker.exists() or marker.is_symlink():
        if (marker.is_symlink() or not stat.S_ISREG(marker.lstat().st_mode)
                or _read_small_json(marker) != value):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(marker))
        _fsync_directory(marker.parent)
        return

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    temporary_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o600)
    try:
        with os.fdopen(temporary_fd, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, marker)
        _fsync_directory(marker.parent)
    finally:
        if temporary.exists():
            os.unlink(temporary)


def _publish_slot_handoff(lock_path: Path, slot: int, proof_path: Path,
                          identity: dict[str, object], incarnation: int) -> None:
    _publish_slot_claim(lock_path, slot, proof_path, identity, incarnation)


def _safe_proof_path(path: Path, incarnation: int) -> bool:
    if (not path.is_absolute() or path.name != "family-exited.json"
            or path.parent.name != str(incarnation) or path.parent.parent.name != "proc"):
        return False
    current = Path(path.anchor)
    for part in path.parts[1:-1]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except OSError:
            return False
        if not stat.S_ISDIR(mode):
            return False
    return True


def _slot_handoff_blocks(lock_path: Path, slot: int) -> bool:
    """Keep a slot unavailable unless its exact host family-exit proof is durable."""
    marker = lock_path.with_name(f"{lock_path.stem}.handoff.json")
    try:
        mode = marker.lstat().st_mode
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if not stat.S_ISREG(mode):
        return True
    record = _read_small_json(marker)
    expected_keys = {"schema", "slot", "incarnation", "proof_path", "identity"}
    if not isinstance(record, dict) or set(record) != expected_keys:
        return True
    incarnation, proof_value, identity = (record.get("incarnation"), record.get("proof_path"),
                                          record.get("identity"))
    if (record.get("schema") != _SLOT_HANDOFF_SCHEMA or type(record.get("slot")) is not int
            or record.get("slot") != slot
            or type(incarnation) is not int or incarnation < 1 or not isinstance(proof_value, str)
            or not isinstance(identity, dict) or set(identity) != {"pid", "pgid", "kernel_identity"}):
        return True
    pid, pgid, kernel_identity = (identity.get("pid"), identity.get("pgid"),
                                  identity.get("kernel_identity"))
    if (type(pid) is not int or pid < 1 or type(pgid) is not int or pgid != pid
            or not isinstance(kernel_identity, str) or not _KERNEL_IDENTITY.fullmatch(kernel_identity)):
        return True
    proof_path = Path(proof_value)
    if not _safe_proof_path(proof_path, incarnation) or _read_small_json(proof_path) != identity:
        return True
    try:
        os.unlink(marker)
        directory_fd = os.open(marker.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        return True
    return False


def lease_slot(endpoint: str, candidates: Sequence[int]) -> SlotLease | None:
    """Acquire the first available cooperative slot lock in candidate order."""
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("endpoint must be a nonempty string")
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of slot ids")
    slots = list(candidates)
    if any(type(slot) is not int or slot < 0 for slot in slots):
        raise ValueError("candidates must contain non-negative integer slot ids")
    if len(set(slots)) != len(slots):
        raise ValueError("candidates must not contain duplicate slot ids")
    endpoint_hash = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:16]
    directory = _lease_root() / endpoint_hash
    _secure_directory(directory)
    claim_mode = os.environ.get(_SLOT_CLAIM_ENV)
    if claim_mode not in (None, _SLOT_CLAIM_VERSION):
        raise ResourceError("subframe slot claim protocol is unsupported")
    if claim_mode == _SLOT_CLAIM_VERSION and "SF_SUBFRAME_INCARNATION" not in os.environ:
        raise ResourceError("durable subframe slot claims require a worker incarnation")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    for slot in slots:
        path = directory / f"{slot}.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR | nofollow, 0o600)
        except OSError as error:
            raise ResourceError(f"cannot open slot lease {path}: {error}") from error
        try:
            mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(mode):
                raise ResourceError(f"slot lease is not a regular file: {path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                continue
            lease = SlotLease(slot, path, descriptor)
            if _slot_handoff_blocks(path, slot):
                logging.getLogger(__name__).warning(
                    "slot %d remains unavailable: unresolved or invalid host family-exit claim", slot)
                lease.close()
                descriptor = -1
                continue
            incarnation = os.environ.get("SF_SUBFRAME_INCARNATION")
            if incarnation is not None:
                if not incarnation.isdecimal() or int(incarnation) < 1:
                    lease.close()
                    descriptor = -1
                    raise ResourceError("subframe lease incarnation is invalid")
                process_dir = Path.cwd() / "proc" / incarnation
                if process_dir.is_symlink() or not process_dir.is_dir():
                    lease.close()
                    descriptor = -1
                    raise ResourceError("subframe lease process directory is unsafe")
                if claim_mode == _SLOT_CLAIM_VERSION:
                    try:
                        proof_path, identity, claim_incarnation = _worker_family_proof_for_slot_claim()
                        _publish_slot_claim(path, slot, proof_path, identity, claim_incarnation)
                    except (OSError, ValueError, ResourceError) as error:
                        lease.close()
                        descriptor = -1
                        raise ResourceError(
                            f"cannot record durable subframe slot claim: {type(error).__name__}") from error
                marker = process_dir / "slot-lease.json"
                try:
                    _record_slot_marker(marker, slot)
                except OSError as error:
                    lease.close()
                    descriptor = -1
                    raise ResourceError(f"cannot record subframe slot lease: {error}") from error
                if os.environ.get("STATEFRAME_ROUTE_SCOPE") == "unit":
                    # Route body authoring must use the slot this process actually owns.
                    os.environ["STATEFRAME_ROUTE_SLOT"] = str(slot)
            return lease
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            raise
    return None


def _memory_available() -> tuple[float | None, str]:
    if sys.platform == "darwin":
        for attempt in range(2):
            try:
                result = subprocess.run(["memory_pressure"], check=False, capture_output=True,
                                        text=True, timeout=load_tolerant_seconds(
                                            "STATEFRAME_PARALLEL_INSPECT_SECONDS"))
                break
            except (OSError, subprocess.TimeoutExpired) as error:
                if attempt == 1:
                    return None, f"memory_pressure probe failed: {type(error).__name__}"
        if result.returncode != 0:
            return None, f"memory_pressure exited {result.returncode}"
        match = _MEMORY_PRESSURE.search(result.stdout)
        if match is None:
            return None, "memory_pressure did not report free percentage"
        value = float(match.group(1))
        return (value, "memory_pressure") if 0 <= value <= 100 else (None, "invalid memory free percentage")
    if sys.platform.startswith("linux"):
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                name, separator, remainder = line.partition(":")
                if separator and name in {"MemTotal", "MemAvailable"}:
                    fields = remainder.split()
                    if fields and fields[0].isdigit():
                        values[name] = int(fields[0])
        except (OSError, UnicodeError) as error:
            return None, f"/proc/meminfo probe failed: {type(error).__name__}"
        total, available = values.get("MemTotal"), values.get("MemAvailable")
        if not total or available is None or available > total:
            return None, "/proc/meminfo lacks valid MemTotal and MemAvailable values"
        return 100.0 * available / total, "/proc/meminfo MemAvailable"
    return None, f"memory probe is unsupported on {sys.platform}"


def machine_ok(repo_bytes: int, artifact_root: Path | None = None) -> tuple[bool, str]:
    """Require room for two repository-sized copies plus 1 GiB and 20% free memory."""
    if type(repo_bytes) is not int or repo_bytes < 0:
        raise ValueError("repo_bytes must be a nonnegative integer")
    required_disk = 2 * repo_bytes + (1 << 30)
    disk_root = Path(artifact_root) if artifact_root is not None else safe_home()
    if disk_root is None:
        return False, "home directory is unavailable for disk free-space probe"
    try:
        free_disk = shutil.disk_usage(disk_root).free
    except OSError as error:
        return False, f"disk free-space probe failed: {type(error).__name__}"
    if free_disk < required_disk:
        return False, f"disk free {free_disk} bytes is below required {required_disk} bytes"
    free_percent, source = _memory_available()
    if free_percent is None:
        return False, source
    if free_percent < 20.0:
        return False, f"{source} reports {free_percent:.1f}% free; 20.0% is required"
    return True, f"disk free {free_disk} bytes; {source} reports {free_percent:.1f}% free"


@contextmanager
def reserve_preparation(repo_bytes: int, artifact_root: Path) -> Iterator[tuple[bool, str]]:
    """Reserve both copies across this process's preparation threads on one filesystem."""
    if type(repo_bytes) is not int or repo_bytes < 0:
        raise ValueError("repo_bytes must be a nonnegative integer")
    try:
        device = artifact_root.stat().st_dev
    except OSError as error:
        yield False, f"disk free-space probe failed: {type(error).__name__}"
        return
    with _PREPARATION_LOCK:
        reserved = _PREPARING_REPO_BYTES.get(device, 0)
        ready, reason = machine_ok(repo_bytes + reserved, artifact_root)
        if ready and repo_bytes:
            _PREPARING_REPO_BYTES[device] = reserved + repo_bytes
    try:
        yield ready, reason
    finally:
        if ready and repo_bytes:
            # Once copying stops, disk_usage accounts for the actual allocation, including
            # reflink savings or partial copies left by failure. Never credit an estimate.
            with _PREPARATION_LOCK:
                remaining = _PREPARING_REPO_BYTES[device] - repo_bytes
                if remaining:
                    _PREPARING_REPO_BYTES[device] = remaining
                else:
                    del _PREPARING_REPO_BYTES[device]


def _pairs_no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SendLedgerError(f"send ledger contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SendLedgerError(f"send ledger value is not canonical JSON: {error}") from error


class SendLedger:
    """Append-only, fsynced per-send authorization and settlement records."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _parent_fd(self) -> int:
        path = self.path
        if (not path.is_absolute() or ".." in path.parts
                or path.name != "sends-ledger.jsonl" or len(path.parents) < 4
                or path.parents[1].name != "units" or path.parents[3].name != "parallel"
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", path.parent.name)
                or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", path.parents[2].name)):
            raise SendLedgerError("send ledger path is outside a run-scoped subframe artifact")
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.anchor, flags)
        try:
            for part in path.parts[1:-1]:
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except OSError as error:
            os.close(descriptor)
            raise SendLedgerError(f"send ledger ancestor is unsafe: {error}") from error

    @contextmanager
    def _locked(self) -> Iterator[int]:
        parent = self._parent_fd()
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        try:
            try:
                descriptor = os.open(self.path.name, os.O_CREAT | os.O_EXCL | os.O_RDWR | nofollow,
                                     0o600, dir_fd=parent)
                created = True
            except FileExistsError:
                descriptor = os.open(self.path.name, os.O_RDWR | nofollow, dir_fd=parent)
                created = False
        except OSError as error:
            os.close(parent)
            raise SendLedgerError(f"cannot open send ledger {self.path}: {error}") from error
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise SendLedgerError(f"send ledger is not a regular file: {self.path}")
            if metadata.st_nlink != 1:
                raise SendLedgerError(f"send ledger hardlink count is {metadata.st_nlink}: {self.path}")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if os.fstat(descriptor).st_nlink != 1:
                raise SendLedgerError(f"send ledger became hardlinked: {self.path}")
            if created:
                os.fsync(parent)
            yield descriptor
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
                os.close(parent)

    @staticmethod
    def _read(descriptor: int) -> dict[str, tuple[str, dict[str, object] | None]]:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        if data and not data.endswith(b"\n"):
            raise SendLedgerError("send ledger has an incomplete final record")
        result: dict[str, tuple[str, dict[str, object] | None]] = {}
        for number, line in enumerate(data.splitlines(), start=1):
            try:
                record = json.loads(line, object_pairs_hook=_pairs_no_duplicates)
            except (json.JSONDecodeError, UnicodeError, SendLedgerError) as error:
                raise SendLedgerError(f"send ledger record {number} is invalid: {error}") from error
            if not isinstance(record, dict):
                raise SendLedgerError(f"send ledger record {number} must be an object")
            state = record.get("state")
            expected = {"schema", "send_id", "state"} if state == "AUTHORIZED" else {
                "schema", "send_id", "state", "usage"} if state == "SETTLED" else set()
            send_id = record.get("send_id")
            if (set(record) != expected or record.get("schema") != SEND_LEDGER_SCHEMA
                    or not isinstance(send_id, str) or not send_id or len(send_id) > 200):
                raise SendLedgerError(f"send ledger record {number} has invalid fields")
            prior = result.get(send_id)
            if state == "AUTHORIZED":
                if prior is not None:
                    raise SendLedgerError(f"send {send_id!r} has duplicate authorization")
                result[send_id] = ("AUTHORIZED", None)
            else:
                usage = record.get("usage")
                if not isinstance(usage, dict) or prior is None or prior[0] != "AUTHORIZED":
                    raise SendLedgerError(f"send {send_id!r} has settlement without one authorization")
                result[send_id] = ("SETTLED", usage)
        return result

    @staticmethod
    def _append(descriptor: int, record: dict[str, object]) -> None:
        if os.fstat(descriptor).st_nlink != 1:
            raise SendLedgerError("send ledger became hardlinked before append")
        line = _canonical(record) + b"\n"
        os.lseek(descriptor, 0, os.SEEK_END)
        written = 0
        while written < len(line):
            count = os.write(descriptor, line[written:])
            if count <= 0:
                raise SendLedgerError("short write to send ledger")
            written += count
        os.fsync(descriptor)

    @staticmethod
    def _send_id(send_id: str) -> None:
        if not isinstance(send_id, str) or not send_id or len(send_id) > 200:
            raise ValueError("send_id must be a nonempty string of at most 200 characters")

    def authorize(self, send_id: str) -> None:
        self._send_id(send_id)
        with self._locked() as descriptor:
            records = self._read(descriptor)
            if send_id in records:
                return
            self._append(descriptor, {"schema": SEND_LEDGER_SCHEMA, "send_id": send_id,
                                      "state": "AUTHORIZED"})

    def authorize_with_cap(self, send_id: str, cap: int) -> bool:
        """Reserve one physical attempt under the same lock used by other incarnations."""
        self._send_id(send_id)
        if type(cap) is not int or cap < 0:
            raise ValueError("cap must be a nonnegative integer")
        with self._locked() as descriptor:
            records = self._read(descriptor)
            if send_id in records or len(records) >= cap:
                return False
            self._append(descriptor, {"schema": SEND_LEDGER_SCHEMA, "send_id": send_id,
                                      "state": "AUTHORIZED"})
            return True

    def settle(self, send_id: str, usage: dict[str, object]) -> None:
        self._send_id(send_id)
        if not isinstance(usage, dict) or any(not isinstance(key, str) for key in usage):
            raise ValueError("usage must be an object with string keys")
        canonical_usage = json.loads(_canonical(usage).decode("utf-8"))
        with self._locked() as descriptor:
            records = self._read(descriptor)
            prior = records.get(send_id)
            if prior is None:
                raise SendLedgerError(f"send {send_id!r} was not authorized")
            if prior[0] == "SETTLED":
                if _canonical(prior[1]) != _canonical(canonical_usage):
                    raise SendLedgerError(f"send {send_id!r} has a conflicting settlement")
                return
            self._append(descriptor, {"schema": SEND_LEDGER_SCHEMA, "send_id": send_id,
                                      "state": "SETTLED", "usage": canonical_usage})

    def remaining(self, cap: int) -> int:
        if type(cap) is not int or cap < 0:
            raise ValueError("cap must be a nonnegative integer")
        with self._locked() as descriptor:
            authorized = len(self._read(descriptor))
        return max(0, cap - authorized)
