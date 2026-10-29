"""Opt-in host-capacity and worker-isolation primitives for the capability runner.

The normal release path does no host-capacity I/O.  ``CAPABILITY_HOST_CAPACITY=on``
is an explicit study arm: it reads the effective Linux CPU/memory limits, derives a
conservative worker cap, keeps audit slots in a separate pool, and gives the runner
the process-group identity it must retain before a worker leader can exit.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import errno
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import IO


MODE_ENV = "CAPABILITY_HOST_CAPACITY"
LOCK_DIR_ENV = "CAPABILITY_HOST_CAPACITY_LOCK_DIR"
SCHEMA = "capability-host-capacity/1"
DEFAULT_FALLBACK_WORKER_CAP = 1
DEFAULT_AUDIT_ALLOWANCE = 1
DEFAULT_LOCK_DIR = "/tmp/capability-host-capacity"

_UNSET = object()
_NO_LIMIT = frozenset({"max", "none", "unlimited"})


class CapacityRefusal(RuntimeError):
    """The host cannot admit another worker or audit operation."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict environment switch without touching host resources."""
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


@dataclass(frozen=True, slots=True)
class HostLimits:
    """Host limits observed at one runner boundary."""

    platform: str
    affinity_cpus: int | None
    cgroup_cpus: float | None
    effective_cpus: float | None
    memory_limit_bytes: int | None
    cpu_source: str
    memory_source: str
    fallback: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "affinity_cpus": self.affinity_cpus,
            "cgroup_cpus": self.cgroup_cpus,
            "effective_cpus": self.effective_cpus,
            "memory_limit_bytes": self.memory_limit_bytes,
            "cpu_source": self.cpu_source,
            "memory_source": self.memory_source,
            "fallback": self.fallback,
        }


@dataclass(frozen=True, slots=True)
class CapacityPlan:
    enabled: bool
    limits: HostLimits
    worker_cap: int
    audit_allowance: int
    worker_memory_bytes: int | None
    lock_dir: Path
    fallback: bool
    rendered_notice: str = ""

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "switch": MODE_ENV,
            "enabled": self.enabled,
            "limits": self.limits.as_dict(),
            "worker_cap": self.worker_cap,
            "audit_allowance": self.audit_allowance,
            "worker_memory_bytes": self.worker_memory_bytes,
            "fallback": self.fallback,
        }

    def acquire_worker(self, *, timeout_s: float | None = None) -> "SlotLease":
        if not self.enabled:
            return SlotLease(None)
        return SlotPool(self.lock_dir / "workers", self.worker_cap).acquire(
            timeout_s=timeout_s
        )

    def acquire_audit(self, *, timeout_s: float | None = None) -> "SlotLease":
        if not self.enabled:
            return SlotLease(None)
        return SlotPool(self.lock_dir / "audits", self.audit_allowance).acquire(
            timeout_s=timeout_s
        )


class SlotLease:
    """An exclusive file-lock lease; release is idempotent."""

    def __init__(self, stream: IO[str] | None):
        self._stream = stream

    def release(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "SlotLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class SlotPool:
    """A small cross-process counting semaphore made from one file per slot."""

    def __init__(self, directory: Path, capacity: int, *, poll_s: float = 0.05):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("slot capacity must be a positive integer")
        if not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError("slot polling interval must be finite and positive")
        self.directory = Path(directory)
        self.capacity = capacity
        self.poll_s = poll_s

    def acquire(self, *, timeout_s: float | None = None) -> SlotLease:
        if timeout_s is not None and (
            isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s < 0
        ):
            raise ValueError("slot timeout must be finite and non-negative")
        self.directory.mkdir(parents=True, exist_ok=True)
        deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
        while True:
            for index in range(self.capacity):
                path = self.directory / f"slot-{index}.lock"
                stream = path.open("a+")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                except OSError as exc:
                    stream.close()
                    if exc.errno in {errno.EACCES, errno.EAGAIN}:
                        continue
                    raise
                return SlotLease(stream)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CapacityRefusal(
                        f"capacity slots exhausted: {self.directory} ({self.capacity})"
                    )
                time.sleep(min(self.poll_s, remaining))
            else:
                time.sleep(self.poll_s)


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _safe_relative_parts(relative: str | None) -> tuple[str, ...]:
    clean = (relative or "").strip().strip("/")
    if not clean or ".." in Path(clean).parts:
        return ()
    return tuple(part for part in Path(clean).parts if part not in {"", "."})


def _cgroup_relative_paths() -> dict[str, str]:
    raw = _text(Path("/proc/self/cgroup")) or ""
    paths: dict[str, str] = {}
    for line in raw.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        relative = "/".join(_safe_relative_parts(fields[2]))
        if fields[0] == "0":
            paths.setdefault("unified", relative)
            continue
        for controller in fields[1].split(","):
            controller = controller.strip()
            if controller:
                paths.setdefault(controller, relative)
    return paths


def _relative_cgroup_path() -> str:
    """Return the unified-v2 path, retaining the old helper's public shape."""
    return _cgroup_relative_paths().get("unified", "")


def _cgroup_candidates(
    root: Path,
    relative: str | None,
    *,
    controller: str | None = None,
) -> tuple[Path, ...]:
    """Return deepest-to-root candidates for v2 and common v1 mount layouts."""
    parts = _safe_relative_parts(relative)
    controllers = (controller,) if controller is not None else (None,)
    if controller == "cpu":
        # Linux commonly mounts cpu and cpuacct together, while distributions
        # also expose either controller separately.
        controllers = ("cpu", "cpu,cpuacct", "cpuacct", None)
    candidates: list[Path] = []
    for mounted_controller in controllers:
        base = root if mounted_controller is None else root / mounted_controller
        for depth in range(len(parts), 0, -1):
            candidate = base.joinpath(*parts[:depth])
            if candidate not in candidates:
                candidates.append(candidate)
        if base not in candidates:
            candidates.append(base)
        if mounted_controller is not None and parts:
            # Some test and legacy layouts place the controller directory
            # beneath the hierarchy rather than mounting it above it.
            for depth in range(len(parts), 0, -1):
                candidate = root.joinpath(*parts[:depth], mounted_controller)
                if candidate not in candidates:
                    candidates.append(candidate)
    return tuple(candidates)


def _parse_cpu_max(raw: str | None) -> float | None:
    if not raw:
        return None
    fields = raw.split()
    if not fields or fields[0].lower() in _NO_LIMIT:
        return None
    if len(fields) < 2:
        return None
    try:
        quota, period = int(fields[0]), int(fields[1])
    except (TypeError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    value = quota / period
    return value if math.isfinite(value) and value > 0 else None


def _parse_v1_cpu(quota_raw: str | None, period_raw: str | None) -> float | None:
    if not quota_raw or quota_raw.lower() in _NO_LIMIT:
        return None
    try:
        quota, period = int(quota_raw), int(period_raw or "")
    except (TypeError, ValueError):
        return None
    if quota <= 0 or period <= 0:
        return None
    value = quota / period
    return value if math.isfinite(value) and value > 0 else None


def _parse_memory(raw: str | None) -> int | None:
    if not raw or raw.lower() in _NO_LIMIT:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    # v1 uses an all-bits sentinel for an unlimited hierarchy.
    if value <= 0 or value >= 1 << 60:
        return None
    return value


def _affinity_count(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value > 0 else None
    try:
        count = len(value)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None
    return count if count > 0 else None


def detect_limits(
    *,
    platform_name: str | None = None,
    cgroup_root: Path | str = "/sys/fs/cgroup",
    cgroup_relative_path: str | None = None,
    cgroup_cpu_relative_path: str | None = None,
    cgroup_memory_relative_path: str | None = None,
    affinity_cpus: object = _UNSET,
) -> HostLimits:
    """Read effective limits without treating the host's total RAM as a quota.

    ``cgroup_root`` and ``affinity_cpus`` are injectable so the release probe can
    exercise nested v1/v2 hierarchies without touching the host cgroup.  Non-Linux
    systems intentionally return unknown limits; the caller must provide the
    configured fallback cap instead of guessing from the workstation.
    """
    platform = str(platform_name or sys.platform).lower()
    if not platform.startswith("linux"):
        return HostLimits(
            platform=platform, affinity_cpus=None, cgroup_cpus=None,
            effective_cpus=None, memory_limit_bytes=None,
            cpu_source="configured_fallback", memory_source="configured_fallback",
            fallback=True,
        )

    if affinity_cpus is _UNSET:
        try:
            affinity_cpus = os.sched_getaffinity(0)
        except (AttributeError, OSError):
            affinity_cpus = None
    affinity = _affinity_count(affinity_cpus)
    root = Path(cgroup_root)
    proc_paths = _cgroup_relative_paths() if (
        cgroup_relative_path is None
        and cgroup_cpu_relative_path is None
        and cgroup_memory_relative_path is None
    ) else {}
    unified_relative = (
        cgroup_relative_path
        if cgroup_relative_path is not None
        else proc_paths.get("unified")
    )
    cpu_relative = (
        cgroup_cpu_relative_path
        if cgroup_cpu_relative_path is not None
        else unified_relative
        if cgroup_relative_path is not None
        else proc_paths.get("cpu") or proc_paths.get("cpuacct") or proc_paths.get("unified")
    )
    memory_relative = (
        cgroup_memory_relative_path
        if cgroup_memory_relative_path is not None
        else unified_relative
        if cgroup_relative_path is not None
        else proc_paths.get("memory") or proc_paths.get("unified")
    )

    quota_values: list[tuple[float, str]] = []
    cpu_source = "affinity" if affinity is not None else "configured_fallback"
    for candidate in _cgroup_candidates(root, unified_relative):
        raw = _text(candidate / "cpu.max")
        if raw is not None:
            quota = _parse_cpu_max(raw)
            if quota is not None:
                quota_values.append((quota, "cgroup_v2"))

    for candidate in _cgroup_candidates(root, cpu_relative, controller="cpu"):
        raw_quota = _text(candidate / "cpu.cfs_quota_us")
        raw_period = _text(candidate / "cpu.cfs_period_us")
        if raw_quota is None and raw_period is None:
            raw_quota = _text(candidate / "cpu" / "cpu.cfs_quota_us")
            raw_period = _text(candidate / "cpu" / "cpu.cfs_period_us")
        quota = _parse_v1_cpu(raw_quota, raw_period)
        if quota is not None:
            quota_values.append((quota, "cgroup_v1"))

    quota = min(quota_values, default=(None, ""), key=lambda item: item[0])[0]
    sources = {source for _value, source in quota_values}
    if sources:
        source_name = "cgroup_v2" if "cgroup_v2" in sources else "cgroup_v1"
        cpu_source = f"{source_name}+affinity" if affinity is not None else source_name

    memory_values: list[tuple[int, str]] = []
    memory_source = "configured_fallback"
    for candidate in _cgroup_candidates(root, unified_relative):
        raw = _text(candidate / "memory.max")
        if raw is not None:
            memory = _parse_memory(raw)
            if memory is not None:
                memory_values.append((memory, "cgroup_v2"))

    for candidate in _cgroup_candidates(root, memory_relative, controller="memory"):
        raw = _text(candidate / "memory.limit_in_bytes")
        if raw is not None:
            memory = _parse_memory(raw)
            if memory is not None:
                memory_values.append((memory, "cgroup_v1"))
        raw = _text(candidate / "memory" / "memory.limit_in_bytes")
        if raw is not None:
            memory = _parse_memory(raw)
            if memory is not None:
                memory_values.append((memory, "cgroup_v1"))

    memory = min(memory_values, default=(None, ""), key=lambda item: item[0])[0]
    if memory_values:
        memory_source = min(memory_values, key=lambda item: item[0])[1]

    values = [float(value) for value in (affinity, quota) if value is not None]
    effective = min(values) if values else None
    if effective is None:
        cpu_source = "configured_fallback"
    return HostLimits(
        platform=platform,
        affinity_cpus=affinity,
        cgroup_cpus=quota,
        effective_cpus=effective,
        memory_limit_bytes=memory,
        cpu_source=cpu_source,
        memory_source=memory_source,
        fallback=effective is None and memory is None,
    )


def _positive_int(config: Mapping[str, object], key: str, default: int | None) -> int | None:
    value = config.get(key, default)
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _switch(spec: Mapping[str, object]) -> str:
    environment = spec.get("environment", {})
    environment = environment if isinstance(environment, Mapping) else {}
    raw = environment.get(MODE_ENV, os.environ.get(MODE_ENV, "off"))
    value = str(raw).strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def _notice(plan: CapacityPlan) -> str:
    cpu = "unknown" if plan.limits.effective_cpus is None else f"{plan.limits.effective_cpus:g}"
    memory = "unbounded" if plan.limits.memory_limit_bytes is None else str(
        plan.limits.memory_limit_bytes
    )
    return (
        f"[HOST CAPACITY {SCHEMA}]\n"
        f"effective_cpu={cpu}; memory_limit_bytes={memory}; "
        f"worker_admission_cap={plan.worker_cap}; audit_allowance={plan.audit_allowance}\n"
        "[/HOST CAPACITY]"
    )


def build_plan(
    spec: Mapping[str, object] | None = None,
    *,
    limits: HostLimits | None = None,
    platform_name: str | None = None,
    cgroup_root: Path | str = "/sys/fs/cgroup",
    cgroup_relative_path: str | None = None,
    affinity_cpus: object = _UNSET,
) -> CapacityPlan:
    """Build an OFF-safe plan from host facts and host-owned run configuration."""
    spec = spec if isinstance(spec, Mapping) else {}
    if _switch(spec) == "off":
        off_limits = limits or HostLimits(
            platform=str(platform_name or sys.platform), affinity_cpus=None,
            cgroup_cpus=None, effective_cpus=None, memory_limit_bytes=None,
            cpu_source="off", memory_source="off", fallback=False,
        )
        return CapacityPlan(False, off_limits, 0, 0, None, Path(DEFAULT_LOCK_DIR), False)

    config = spec.get("host_capacity", {})
    if not isinstance(config, Mapping):
        raise ValueError("host_capacity must be an object")
    limits = limits or detect_limits(
        platform_name=platform_name,
        cgroup_root=cgroup_root,
        cgroup_relative_path=cgroup_relative_path,
        affinity_cpus=affinity_cpus,
    )
    fallback_memory = _positive_int(config, "fallback_memory_bytes", None)
    if limits.memory_limit_bytes is None and fallback_memory is not None:
        limits = replace(limits, memory_limit_bytes=fallback_memory,
                         memory_source="configured_fallback", fallback=True)

    fallback_cap = _positive_int(config, "fallback_worker_cap", DEFAULT_FALLBACK_WORKER_CAP)
    configured_cap = _positive_int(config, "worker_cap", None)
    audit_allowance = _positive_int(config, "audit_allowance", DEFAULT_AUDIT_ALLOWANCE)
    worker_memory = _positive_int(config, "worker_memory_bytes", None)
    if fallback_cap is None or audit_allowance is None:
        raise AssertionError("positive defaults must not be None")

    cpu_cap = (
        max(1, math.floor(limits.effective_cpus))
        if limits.effective_cpus is not None else fallback_cap
    )
    worker_cap = min(cpu_cap, configured_cap) if configured_cap is not None else cpu_cap
    if worker_memory is not None and limits.memory_limit_bytes is not None:
        memory_cap = max(1, limits.memory_limit_bytes // worker_memory)
        worker_cap = min(worker_cap, memory_cap)
    lock_raw = config.get("lock_dir")
    lock_raw = lock_raw if lock_raw is not None else os.environ.get(LOCK_DIR_ENV, DEFAULT_LOCK_DIR)
    lock_dir = Path(str(lock_raw))
    if not lock_dir.is_absolute():
        raise ValueError("host capacity lock_dir must be absolute")
    plan = CapacityPlan(
        True, limits, max(1, worker_cap), audit_allowance, worker_memory,
        lock_dir, limits.fallback,
    )
    return replace(plan, rendered_notice=_notice(plan))


def install_audit_admission(runtime: object, plan: CapacityPlan) -> None:
    """Place nested test/audit work in its own bounded pool when the arm is ON."""
    if not plan.enabled:
        return
    original = getattr(runtime, "run_tests", None)
    if not callable(original):
        raise TypeError("audit runtime must expose callable run_tests")
    if getattr(original, "_host_capacity_audit", False):
        raise ValueError("host capacity audit admission already installed")

    def admitted(*args: object, **kwargs: object) -> object:
        with plan.acquire_audit():
            return original(*args, **kwargs)

    setattr(admitted, "_host_capacity_audit", True)
    setattr(runtime, "run_tests", admitted)


def captured_process_group(process: object) -> int | None:
    """Capture the worker's process group immediately after spawn."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        return os.getpgid(pid)
    except (AttributeError, OSError, ProcessLookupError):
        # ``start_new_session=True`` makes the PID the group leader.  Keeping
        # this fallback avoids a second lookup after the leader exits.
        return pid


def _wait(process: object, timeout: float) -> None:
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return
    try:
        wait(timeout=timeout)
    except (OSError, ProcessLookupError, TimeoutError, subprocess.TimeoutExpired):
        pass


def terminate_process_group(
    process: object,
    *,
    pgid: int | None,
    grace_s: float = 10.0,
) -> None:
    """Bounded TERM/KILL cleanup of the captured group, including descendants."""
    if not isinstance(grace_s, (int, float)) or isinstance(grace_s, bool):
        raise ValueError("cleanup grace must be numeric")
    if not math.isfinite(grace_s) or grace_s <= 0:
        raise ValueError("cleanup grace must be finite and positive")
    if isinstance(pgid, int) and pgid > 0:
        try:
            if pgid == os.getpgrp():
                raise ValueError("refusing to signal the supervisor process group")
        except AttributeError:
            pass
    group_signalled = False
    if isinstance(pgid, int) and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
            group_signalled = True
        except (AttributeError, OSError, ProcessLookupError):
            pass
    if not group_signalled:
        try:
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
        except (OSError, ProcessLookupError):
            pass
    _wait(process, float(grace_s))

    if isinstance(pgid, int) and pgid > 0:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (AttributeError, OSError, ProcessLookupError):
            pass
    try:
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
    except (OSError, ProcessLookupError):
        pass
    _wait(process, float(grace_s))


__all__ = [
    "MODE_ENV", "SCHEMA", "CapacityPlan", "CapacityRefusal", "HostLimits", "mode",
    "SlotLease", "SlotPool", "build_plan", "captured_process_group", "detect_limits",
    "install_audit_admission", "terminate_process_group",
]
