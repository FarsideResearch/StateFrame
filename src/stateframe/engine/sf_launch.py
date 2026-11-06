"""Exec-only resource launcher. No Python callbacks run between fork and exec."""
from __future__ import annotations

import ctypes
import errno
import json
import math
import os
import platform
import resource
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

_NPROC_HEADROOM = 128
_CGROUP_LEAF_PATTERN_LOCK = threading.RLock()
_CGROUP_LEAF_PATTERN_INITIALIZED = False
_CGROUP_LEAF_PATTERN_ORIGIN: Path | None = None
_CGROUP_LEAF_PATTERN_PARENT: Path | None = None
_CGROUP_LEAF_PATTERN_APPLIED = False
_CGROUP_LEAF_PATTERN_STATUS = "not_delegated"
_CGROUP_LEAF_PATTERN_CREATED_PARENT: Path | None = None
_CGROUP_LEAF_PATTERN_CREATOR_PID: int | None = None


def _current_cgroup() -> Path | None:
    """Return our delegated v2 cgroup, if its mount is visible."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open("/proc/self/cgroup", encoding="ascii") as stream:
            relative = next(line.split(":", 2)[2].strip() for line in stream
                            if line.startswith("0::"))
        root = Path("/sys/fs/cgroup").resolve()
        path = (root / relative.lstrip("/")).resolve()
        path.relative_to(root)
        return path if (path / "cgroup.controllers").is_file() else None
    except (OSError, RuntimeError, StopIteration, ValueError):
        return None


def _cgroup_mount_root() -> Path:
    return Path("/sys/fs/cgroup").resolve()


def _reset_cgroup_leaf_pattern_state() -> None:
    """Clear process-local setup state and ownership proof."""
    global _CGROUP_LEAF_PATTERN_LOCK, _CGROUP_LEAF_PATTERN_INITIALIZED
    global _CGROUP_LEAF_PATTERN_ORIGIN, _CGROUP_LEAF_PATTERN_PARENT
    global _CGROUP_LEAF_PATTERN_APPLIED, _CGROUP_LEAF_PATTERN_STATUS
    global _CGROUP_LEAF_PATTERN_CREATED_PARENT, _CGROUP_LEAF_PATTERN_CREATOR_PID
    _CGROUP_LEAF_PATTERN_LOCK = threading.RLock()
    _CGROUP_LEAF_PATTERN_INITIALIZED = False
    _CGROUP_LEAF_PATTERN_ORIGIN = None
    _CGROUP_LEAF_PATTERN_PARENT = None
    _CGROUP_LEAF_PATTERN_APPLIED = False
    _CGROUP_LEAF_PATTERN_STATUS = "not_delegated"
    _CGROUP_LEAF_PATTERN_CREATED_PARENT = None
    _CGROUP_LEAF_PATTERN_CREATOR_PID = None


def _after_fork_cgroup_leaf_pattern() -> None:
    """Keep proof only when this child inherited a successfully created leaf."""
    global _CGROUP_LEAF_PATTERN_LOCK
    _CGROUP_LEAF_PATTERN_LOCK = threading.RLock()
    if (_CGROUP_LEAF_PATTERN_INITIALIZED and _CGROUP_LEAF_PATTERN_APPLIED
            and _CGROUP_LEAF_PATTERN_CREATED_PARENT is not None
            and _CGROUP_LEAF_PATTERN_CREATOR_PID is not None):
        return
    _reset_cgroup_leaf_pattern_state()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_cgroup_leaf_pattern)


def _cgroup_leaf_pattern_status() -> str:
    """Return the host-audit status without changing execution receipts."""
    with _CGROUP_LEAF_PATTERN_LOCK:
        return _CGROUP_LEAF_PATTERN_STATUS


def _cgroup_leaf_pattern_writable(cgroup: Path) -> bool:
    """Check whether the visible delegated cgroup has its required write points."""
    try:
        mount_root = _cgroup_mount_root()
        if cgroup.resolve() == mount_root:
            return False
        return (cgroup.is_dir()
                and (cgroup / "cgroup.procs").is_file()
                and (cgroup / "cgroup.controllers").is_file()
                and (cgroup / "cgroup.subtree_control").is_file()
                and os.access(cgroup, os.W_OK | os.X_OK)
                and os.access(cgroup / "cgroup.procs", os.W_OK)
                and os.access(cgroup / "cgroup.subtree_control", os.W_OK))
    except OSError:
        return False


def _read_cgroup_pids(cgroup: Path) -> list[int]:
    try:
        values = (cgroup / "cgroup.procs").read_text(encoding="ascii").split()
    except FileNotFoundError:
        return []
    pids = []
    for value in values:
        pid = int(value)
        if pid <= 0:
            raise ValueError("cgroup.procs contains a non-positive pid")
        pids.append(pid)
    return list(dict.fromkeys(pids))


def _process_details(pid: int) -> tuple[int, int]:
    """Return (parent pid, real uid) from procfs for one visible process."""
    proc = Path("/proc") / str(pid)
    stat = (proc / "stat").read_text(encoding="ascii")
    end = stat.rfind(")")
    if end < 0:
        raise ValueError("malformed proc stat")
    fields = stat[end + 1:].split()
    if len(fields) < 2:
        raise ValueError("short proc stat")
    return int(fields[1]), proc.stat().st_uid


def _owned_cgroup_pids(pids: list[int], *, root_pid: int | None = None
                       ) -> tuple[list[int], list[int]]:
    """Select the current process and descendants; ancestors remain foreign."""
    current_pid = os.getpid() if root_pid is None else root_pid
    candidates = set(pids)
    details: dict[int, tuple[int, int]] = {}
    for pid in candidates | {current_pid}:
        try:
            details[pid] = _process_details(pid)
        except (OSError, ValueError, IndexError):
            continue
    real_uid = os.getuid()
    current = details.get(current_pid)
    if current is None or current[1] != real_uid:
        return [], sorted(candidates)

    descendants: list[tuple[int, int]] = []
    for pid in candidates:
        if pid == current_pid or pid not in details:
            continue
        cursor = pid
        depth = 0
        visited = set()
        while cursor in details and cursor not in visited:
            visited.add(cursor)
            parent_pid = details[cursor][0]
            if parent_pid == current_pid:
                if details[pid][1] == real_uid:
                    descendants.append((depth + 1, pid))
                break
            if parent_pid not in details:
                break
            cursor = parent_pid
            depth += 1

    owned = [pid for _depth, pid in sorted(descendants, reverse=True)]
    if current_pid in candidates:
        owned.append(current_pid)
    owned_set = set(owned)
    return owned, sorted(candidates - owned_set)


def _controller_names(cgroup: Path, filename: str) -> set[str]:
    return set((cgroup / filename).read_text(encoding="ascii").split())


def _supervisor_parent(cgroup: Path) -> Path | None:
    """Recognize only a leaf this process family created during this lifetime."""
    if cgroup.name != "supervisor":
        return None
    parent = cgroup.parent
    created_parent = _CGROUP_LEAF_PATTERN_CREATED_PARENT
    creator_pid = _CGROUP_LEAF_PATTERN_CREATOR_PID
    if (created_parent is None or creator_pid is None
            or parent != created_parent or cgroup != created_parent / "supervisor"):
        return None
    try:
        if (parent.resolve() == _cgroup_mount_root()
                or not _cgroup_leaf_pattern_writable(parent)):
            return None
        if any(child.is_dir() for child in cgroup.iterdir()):
            return None
        available = _controller_names(parent, "cgroup.controllers") & {"memory", "pids"}
        enabled = _controller_names(parent, "cgroup.subtree_control")
        if not available or not available.issubset(enabled):
            return None
        members = _read_cgroup_pids(cgroup)
        owned, foreign = _owned_cgroup_pids(members, root_pid=creator_pid)
        if os.getpid() in owned and not foreign:
            return parent
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def _set_cgroup_controllers(cgroup: Path, controllers: set[str], enabled: bool) -> None:
    operation = "+" if enabled else "-"
    (cgroup / "cgroup.subtree_control").write_text(
        " ".join(operation + name for name in sorted(controllers)), encoding="ascii")


@contextmanager
def _cgroup_leaf_pattern_process_lock(cgroup: Path):
    """Serialize leaf setup across launcher processes sharing this cgroup."""
    if not sys.platform.startswith("linux"):
        yield
        return
    import fcntl

    lock_parent = cgroup.parent if cgroup.name == "supervisor" else cgroup
    descriptor = os.open(lock_parent / "cgroup.procs",
                         os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    locked = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


def _rollback_leaf_pattern(parent: Path, supervisor: Path,
                           enabled_by_us: set[str]) -> tuple[bool, str]:
    """Undo our unused controllers, return our processes, and remove the leaf."""
    try:
        children = {child for child in parent.iterdir() if child.is_dir()}
    except OSError:
        return False, "read_children"
    if children - {supervisor}:
        return False, "other_child_cgroups"

    try:
        enabled = _controller_names(parent, "cgroup.subtree_control")
    except (OSError, RuntimeError, ValueError):
        return False, "read_subtree_control"
    added = enabled & enabled_by_us
    try:
        if added:
            _set_cgroup_controllers(parent, added, False)
        if added & _controller_names(parent, "cgroup.subtree_control"):
            return False, "disable_controllers"
    except (OSError, RuntimeError, ValueError) as error:
        return False, f"disable_controllers:{type(error).__name__}"

    try:
        members = _read_cgroup_pids(supervisor)
        owned, foreign = _owned_cgroup_pids(members)
        if foreign:
            return False, "foreign_supervisor_process"
        for pid in owned:
            (parent / "cgroup.procs").write_text(str(pid), encoding="ascii")
        if _read_cgroup_pids(supervisor):
            return False, "return_processes"
        try:
            supervisor.rmdir()
        except OSError as error:
            if error.errno != errno.ENOENT:
                raise
    except (OSError, RuntimeError, ValueError) as error:
        return False, f"return_processes:{type(error).__name__}"
    return True, ""


def _try_cgroup_leaf_pattern(cgroup: Path) -> tuple[Path | None, bool, str]:
    """Move the process family under supervisor before delegating sibling controls."""
    global _CGROUP_LEAF_PATTERN_CREATED_PARENT, _CGROUP_LEAF_PATTERN_CREATOR_PID
    existing_parent = _supervisor_parent(cgroup)
    if existing_parent is not None:
        return existing_parent, True, "applied"
    if cgroup.name == "supervisor":
        return None, False, "failed:supervisor_foreign"
    try:
        if cgroup.resolve() == _cgroup_mount_root():
            return cgroup, False, "not_delegated"
        available = _controller_names(cgroup, "cgroup.controllers") & {"memory", "pids"}
    except OSError:
        return cgroup, False, "not_delegated"
    if not available or not _cgroup_leaf_pattern_writable(cgroup):
        return cgroup, False, "not_delegated"

    try:
        members = _read_cgroup_pids(cgroup)
    except (OSError, ValueError):
        return cgroup, False, "not_delegated"
    if os.getpid() not in members:
        return cgroup, False, "failed:self_pid_missing"
    owned, foreign = _owned_cgroup_pids(members)
    if foreign:
        return cgroup, False, "failed:foreign_process"
    supervisor = cgroup / "supervisor"
    if supervisor.exists():
        return cgroup, False, "failed:supervisor_exists"

    stage = "create_supervisor"
    supervisor_created = False
    enabled_by_us: set[str] = set()
    try:
        supervisor.mkdir()
        supervisor_created = True
        stage = "move_processes"
        for pid in owned:
            (supervisor / "cgroup.procs").write_text(str(pid), encoding="ascii")
        for _attempt in range(4):
            remaining = _read_cgroup_pids(cgroup)
            if not remaining:
                break
            more_owned, remaining_foreign = _owned_cgroup_pids(
                remaining, root_pid=os.getpid())
            if remaining_foreign:
                raise RuntimeError("foreign_process")
            for pid in more_owned:
                (supervisor / "cgroup.procs").write_text(str(pid), encoding="ascii")
        if _read_cgroup_pids(cgroup):
            raise RuntimeError("processes_remain")
        stage = "enable_controllers"
        enabled_before_write = _controller_names(cgroup, "cgroup.subtree_control")
        to_enable = available - enabled_before_write
        if to_enable:
            try:
                _set_cgroup_controllers(cgroup, to_enable, True)
            finally:
                try:
                    enabled_by_us = (
                        (_controller_names(cgroup, "cgroup.subtree_control")
                         - enabled_before_write) & to_enable)
                except (OSError, RuntimeError, ValueError):
                    # If ownership cannot be established, rollback leaves the controls on.
                    enabled_by_us = set()
        enabled = _controller_names(cgroup, "cgroup.subtree_control")
        if not available.issubset(enabled):
            raise RuntimeError("controllers_not_enabled")
        _CGROUP_LEAF_PATTERN_CREATED_PARENT = cgroup
        _CGROUP_LEAF_PATTERN_CREATOR_PID = os.getpid()
        return cgroup, True, "applied"
    except (OSError, RuntimeError, ValueError) as error:
        reason = (str(error) if isinstance(error, RuntimeError)
                  and str(error) in {
                      "foreign_process", "processes_remain", "controllers_not_enabled"}
                  else f"{stage}:{type(error).__name__}")
        rollback_ok = False
        rollback_reason = ""
        if supervisor_created:
            rollback_ok, rollback_reason = _rollback_leaf_pattern(
                cgroup, supervisor, enabled_by_us)
        else:
            rollback_ok = True
        not_delegated = (stage == "create_supervisor"
                         and isinstance(error, OSError)
                         and error.errno in {errno.EACCES, errno.EPERM, errno.EROFS})
        if not rollback_ok:
            status = f"failed:{reason}:rollback_{rollback_reason or 'incomplete'}"
            return None, False, status
        status = "not_delegated" if not_delegated else f"failed:{reason}"
        return cgroup, False, status


def _command_cgroup_parent() -> tuple[Path | None, bool]:
    """Initialize the leaf once per process and return its sibling-group parent."""
    global _CGROUP_LEAF_PATTERN_INITIALIZED, _CGROUP_LEAF_PATTERN_ORIGIN
    global _CGROUP_LEAF_PATTERN_PARENT, _CGROUP_LEAF_PATTERN_APPLIED
    global _CGROUP_LEAF_PATTERN_STATUS
    global _CGROUP_LEAF_PATTERN_CREATED_PARENT, _CGROUP_LEAF_PATTERN_CREATOR_PID
    current = _current_cgroup()
    with _CGROUP_LEAF_PATTERN_LOCK:
        if _CGROUP_LEAF_PATTERN_INITIALIZED and current == _CGROUP_LEAF_PATTERN_ORIGIN:
            return _CGROUP_LEAF_PATTERN_PARENT, _CGROUP_LEAF_PATTERN_APPLIED
        if (_CGROUP_LEAF_PATTERN_INITIALIZED and _CGROUP_LEAF_PATTERN_APPLIED
                and _CGROUP_LEAF_PATTERN_ORIGIN is not None
                and current == _CGROUP_LEAF_PATTERN_ORIGIN / "supervisor"):
            if _supervisor_parent(current) == _CGROUP_LEAF_PATTERN_ORIGIN:
                return _CGROUP_LEAF_PATTERN_PARENT, True
            _CGROUP_LEAF_PATTERN_PARENT = None
            _CGROUP_LEAF_PATTERN_APPLIED = False
            _CGROUP_LEAF_PATTERN_STATUS = "failed:supervisor_foreign"
            return None, False
        _CGROUP_LEAF_PATTERN_INITIALIZED = True
        if current is None:
            _CGROUP_LEAF_PATTERN_ORIGIN = None
            _CGROUP_LEAF_PATTERN_PARENT = None
            _CGROUP_LEAF_PATTERN_APPLIED = False
            _CGROUP_LEAF_PATTERN_STATUS = "not_delegated"
            _CGROUP_LEAF_PATTERN_CREATED_PARENT = None
            _CGROUP_LEAF_PATTERN_CREATOR_PID = None
            return None, False
        try:
            with _cgroup_leaf_pattern_process_lock(current):
                current = _current_cgroup() or current
                if (_CGROUP_LEAF_PATTERN_INITIALIZED
                        and current == _CGROUP_LEAF_PATTERN_ORIGIN):
                    return _CGROUP_LEAF_PATTERN_PARENT, _CGROUP_LEAF_PATTERN_APPLIED
                if (_CGROUP_LEAF_PATTERN_INITIALIZED and _CGROUP_LEAF_PATTERN_APPLIED
                        and _CGROUP_LEAF_PATTERN_ORIGIN is not None
                        and current == _CGROUP_LEAF_PATTERN_ORIGIN / "supervisor"):
                    if _supervisor_parent(current) == _CGROUP_LEAF_PATTERN_ORIGIN:
                        return _CGROUP_LEAF_PATTERN_PARENT, True
                    _CGROUP_LEAF_PATTERN_PARENT = None
                    _CGROUP_LEAF_PATTERN_APPLIED = False
                    _CGROUP_LEAF_PATTERN_STATUS = "failed:supervisor_foreign"
                    return None, False
                _CGROUP_LEAF_PATTERN_ORIGIN = current
                _CGROUP_LEAF_PATTERN_PARENT = current
                _CGROUP_LEAF_PATTERN_APPLIED = False
                _CGROUP_LEAF_PATTERN_CREATED_PARENT = None
                _CGROUP_LEAF_PATTERN_CREATOR_PID = None
                parent, applied, status = _try_cgroup_leaf_pattern(current)
        except OSError as error:
            _CGROUP_LEAF_PATTERN_ORIGIN = current
            _CGROUP_LEAF_PATTERN_PARENT = (
                None if current.name == "supervisor" else current)
            _CGROUP_LEAF_PATTERN_APPLIED = False
            _CGROUP_LEAF_PATTERN_CREATED_PARENT = None
            _CGROUP_LEAF_PATTERN_CREATOR_PID = None
            _CGROUP_LEAF_PATTERN_STATUS = f"failed:leaf_lock:{type(error).__name__}"
            return _CGROUP_LEAF_PATTERN_PARENT, False
        _CGROUP_LEAF_PATTERN_PARENT = parent
        _CGROUP_LEAF_PATTERN_APPLIED = applied
        _CGROUP_LEAF_PATTERN_STATUS = status
        return parent, applied


def _command_cgroup(ident: str, nproc: int,
                    memory_bytes: int | None = None) -> Path | None:
    """Create a delegated per-command group, keeping pids and memory optional."""
    if memory_bytes is not None and (type(memory_bytes) is not int or memory_bytes <= 0):
        raise ValueError("memory_bytes must be a positive integer")
    parent, leaf_pattern_applied = _command_cgroup_parent()
    if parent is None:
        return None
    group = parent / f"stateframe-{os.getpid()}-{ident}"
    return _create_command_cgroup(
        group, nproc, memory_bytes if leaf_pattern_applied else None,
        allow_memory_only=leaf_pattern_applied)


def _create_command_cgroup(group: Path, nproc: int, memory_bytes: int | None, *,
                           allow_memory_only: bool) -> Path | None:
    """Create a command group while preserving the original pids-only fallback."""
    try:
        group.mkdir()
        pids_configured = False
        try:
            pids_max = group / "pids.max"
            if pids_max.is_file():
                # Allow a few bubblewrap setup tasks. The in-namespace limit
                # independently bounds the check's tasks when delegation is absent.
                pids_max.write_text(str(nproc + 8), encoding="ascii")
                pids_configured = True
        except OSError:
            group.rmdir()
            return None
    except OSError:
        return None
    if memory_bytes is not None:
        memory_max = group / "memory.max"
        try:
            if memory_max.is_file():
                memory_max.write_text(str(memory_bytes), encoding="ascii")
                for name, value in (("memory.swap.max", "0"),
                                    ("memory.oom.group", "1")):
                    control = group / name
                    if control.is_file():
                        try:
                            control.write_text(value, encoding="ascii")
                        except OSError:
                            # memory.max is still a hard limit; optional kernel
                            # controls can be absent or independently delegated.
                            pass
        except OSError:
            # A pids-only group remains useful. Callers detect that memory.max
            # was not installed and apply RLIMIT_DATA to the child instead.
            pass
    memory_configured = (memory_bytes is not None
                         and _cgroup_memory_limit_configured(group, memory_bytes))
    if not pids_configured and not (allow_memory_only and memory_configured):
        try:
            group.rmdir()
        except OSError:
            pass
        return None
    return group


def _proc_swaps_has_no_devices(path: Path = Path("/proc/swaps")) -> bool:
    """Return true only when procfs confirms that no swap device is active."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return False
    if not lines or lines[0].split() != ["Filename", "Type", "Size", "Used", "Priority"]:
        return False
    return not any(line.strip() for line in lines[1:])


def _cgroup_memory_limit_configured(
        group: Path | None, memory_bytes: int, *,
        proc_swaps_path: Path = Path("/proc/swaps"),
) -> bool:
    """Confirm exact memory and swap limits before selecting cgroup enforcement."""
    if group is None or type(memory_bytes) is not int or memory_bytes <= 0:
        return False
    try:
        if (group / "memory.max").read_text(encoding="ascii").strip() != str(memory_bytes):
            return False
        try:
            swap_limit = (group / "memory.swap.max").read_text(encoding="ascii").strip()
        except FileNotFoundError:
            if not _proc_swaps_has_no_devices(proc_swaps_path):
                return False
        else:
            if swap_limit != "0":
                return False
        return True
    except OSError:
        return False


def _lower_only(which: int, value: int) -> None:
    """Set a POSIX limit without raising an inherited hard or soft cap."""
    soft, hard = resource.getrlimit(which)
    infinity = resource.RLIM_INFINITY
    bound = value if hard == infinity else min(value, hard)
    new_soft = bound if soft == infinity else min(bound, soft)
    resource.setrlimit(which, (new_soft, bound))


def _enter_command_cgroup(group: Path, memory_bytes: int) -> str:
    """Join the command group or install the mandatory Linux memory fallback."""
    memory_configured = _cgroup_memory_limit_configured(group, memory_bytes)
    joined = False
    try:
        (group / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
        joined = True
    except OSError:
        pass
    if memory_configured and joined:
        return "cgroup_memory_max"
    if not hasattr(resource, "RLIMIT_DATA"):
        raise RuntimeError("Linux RLIMIT_DATA memory fallback is unavailable")
    _lower_only(resource.RLIMIT_DATA, memory_bytes)
    return "rlimit_data"


def _user_task_count() -> int:
    """Count readable threads charged to our real UID, not procfs owners."""
    uid = os.getuid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return 0
    count = 0
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/status", "rb") as status:
                fields = {line.split(b":", 1)[0]: line.split(b":", 1)[1].split()
                          for line in status if line.startswith((b"Uid:", b"Threads:"))}
            if int(fields[b"Uid"][0]) != uid:
                continue
            threads = int(fields[b"Threads"][0])
            if threads > 0:
                count += threads
        except (OSError, KeyError, IndexError, ValueError):
            continue  # Hidden, vanished, or incomplete procfs entry.
    return count


def _set_relative_nproc_limit(ceiling: int) -> None:
    """Apply the single uid-wide NPROC setter without raising an inherited hard cap."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    infinity = resource.RLIM_INFINITY
    bound = ceiling if hard == infinity else min(ceiling, hard)
    new_soft = bound if soft == infinity else min(bound, soft)
    resource.setrlimit(resource.RLIMIT_NPROC, (new_soft, bound))


def install_syscall_filter(*, network_none: bool) -> None:
    """Linux bwrap raises lo even with --unshare-net; deny socket creation too.

    Every target is denied cross-process memory/fd access to the trusted
    supervisor. Filters survive exec/fork. Reject alternate syscall ABIs and,
    for network-none, io_uring so it cannot bypass the socket syscall policy.
    """
    machine = platform.machine()
    if machine == "x86_64":
        arch = 0xC000003E
        calls = (101, 310, 311, 438)  # ptrace, process_vm_*, pidfd_getfd
        network_calls = (41, 42, 43, 49, 50, 53, 288, 425, 426, 427)
    elif machine in {"aarch64", "arm64"}:
        arch = 0xC00000B7
        calls = (117, 270, 271, 438)
        network_calls = (198, 199, 200, 201, 202, 203, 242, 425, 426, 427)
    else:
        raise RuntimeError(f"execution seccomp is unsupported on {machine}")
    if network_none:
        calls += network_calls

    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                    ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]

    deny = 0x00050000 | errno.EPERM
    instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, arch), (0x06, 0, 0, 0x80000000),
                    (0x20, 0, 0, 0), (0x35, 0, 1, 0x40000000), (0x06, 0, 0, deny)]
    for number in calls:
        instructions += [(0x15, 0, 1, number), (0x06, 0, 0, deny)]
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    filters = (Filter * len(instructions))(*(Filter(*row) for row in instructions))
    program = Program(len(filters), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot set no_new_privs")
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot install execution seccomp filter")


def launch(spec: dict) -> None:
    argv = spec["argv"]
    if not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
        raise ValueError("launcher argv must be nonempty strings without NUL")
    limits = spec["limits"]
    # Close inherited authority before reducing NOFILE; do not leave high fds open.
    maximum = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if maximum == resource.RLIM_INFINITY:
        maximum = os.sysconf("SC_OPEN_MAX")
    os.closerange(3, int(maximum))
    if spec.get("new_session", True):
        os.setsid()
    os.umask(0o077)
    if sys.platform == "darwin":
        # macOS reserves an address space larger than the physical-memory budget
        # before Python starts. The independent supervisor enforces aggregate RSS.
        if spec.get("memory_supervised") is not True:
            raise ValueError("macOS execution requires supervised resident-memory accounting")
    else:
        memory_mechanism = spec.get("memory_limit_mechanism", "rlimit_data")
        if memory_mechanism == "rlimit_data":
            if not hasattr(resource, "RLIMIT_DATA"):
                raise RuntimeError("Linux RLIMIT_DATA memory fallback is unavailable")
            if spec.get("memory_supervised") is not True:
                raise ValueError(
                    "Linux RLIMIT_DATA fallback requires supervised resident-memory accounting")
            _lower_only(resource.RLIMIT_DATA, limits["memory_bytes"])
        elif memory_mechanism != "cgroup_memory_max":
            raise ValueError("unsupported Linux memory limit mechanism")
    process_limits = [(resource.RLIMIT_CPU, math.ceil(limits["cpu_seconds"])),
                      (resource.RLIMIT_FSIZE, limits["file_bytes"]),
                      (resource.RLIMIT_NOFILE, limits["nofile"])]
    for kind, value in process_limits:
        # Never silently keep a weaker limit if the OS refuses this request.
        hard = resource.getrlimit(kind)[1]
        value = min(value, hard) if hard != resource.RLIM_INFINITY else value
        resource.setrlimit(kind, (value, value))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.chdir(spec["cwd"])
    if sys.platform.startswith("linux"):
        # Count immediately before exec; the optional cgroup and supervisor provide
        # independent bounds when namespace visibility or concurrent tasks race us.
        _set_relative_nproc_limit(_user_task_count() + limits["nproc"] + _NPROC_HEADROOM)
    os.execvpe(argv[0], argv, spec["env"])


def _exec_wrapped(argv: list[str], environment) -> None:
    """Exec the command this wrapper stands in front of, reporting a missing program exactly
    as bubblewrap does when it execs the command itself. Callers (outer checks) parse that
    line to say "check command not found: NAME" instead of reporting a failing check."""
    try:
        os.execvpe(argv[0], argv, environment)
    except FileNotFoundError:
        print(f"bwrap: execvp {argv[0]}: No such file or directory", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    try:
        if len(sys.argv) >= 4 and sys.argv[1] == "--nproc-budget":
            budget = int(sys.argv[2])
            if not 1 <= budget <= 256:
                raise ValueError("NPROC budget is outside the execution policy bound")
            _set_relative_nproc_limit(_user_task_count() + budget + _NPROC_HEADROOM)
            _exec_wrapped(sys.argv[3:], os.environ)
        if len(sys.argv) >= 4 and sys.argv[1] == "--cgroup-enter":
            group = Path(sys.argv[2])
            cursor = 3
            memory_bytes = None
            report_fd = None
            while cursor < len(sys.argv) and sys.argv[cursor].startswith("--"):
                option = sys.argv[cursor]
                if option == "--memory-budget" and cursor + 1 < len(sys.argv):
                    memory_bytes = int(sys.argv[cursor + 1])
                    cursor += 2
                elif option == "--memory-result-fd" and cursor + 1 < len(sys.argv):
                    report_fd = int(sys.argv[cursor + 1])
                    cursor += 2
                elif option == "--":
                    cursor += 1
                    break
                else:
                    break
            if memory_bytes is None:
                try:
                    (group / "cgroup.procs").write_text(str(os.getpid()), encoding="ascii")
                except OSError:
                    pass  # Legacy pids-only wrapper; RLIMIT_NPROC remains the fallback.
                memory_mechanism = None
            else:
                memory_mechanism = _enter_command_cgroup(group, memory_bytes)
                if report_fd is not None:
                    marker = b"C" if memory_mechanism == "cgroup_memory_max" else b"D"
                    os.write(report_fd, marker)
                    os.close(report_fd)
            command = sys.argv[cursor:]
            if not command:
                raise ValueError("cgroup-enter requires a wrapped command")
            _exec_wrapped(command, os.environ)
        if len(sys.argv) >= 4 and sys.argv[1] in {"--network-none", "--process-filter"}:
            environment = json.loads(sys.argv[2])
            install_syscall_filter(network_none=sys.argv[1] == "--network-none")
            _exec_wrapped(sys.argv[3:], environment)
        if len(sys.argv) == 3 and sys.argv[1] == "--fd":
            with os.fdopen(int(sys.argv[2]), "rb") as stream:
                data = stream.read((1 << 20) + 1)
            if len(data) > 1 << 20:
                raise ValueError("launcher spec exceeds 1 MiB")
            spec = json.loads(data)
        elif len(sys.argv) == 2:
            spec = json.loads(sys.argv[1])
        else:
            raise ValueError("expected JSON spec or --fd FD")
        launch(spec)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"execution launcher failed: {error}", file=sys.stderr)
        return 125
    return 125


if __name__ == "__main__":
    raise SystemExit(main())
