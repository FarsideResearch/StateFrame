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
from pathlib import Path

_NPROC_HEADROOM = 128


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


def _command_cgroup(ident: str, nproc: int) -> Path | None:
    """Use delegated pids.max when available; RLIMIT_NPROC remains the fallback."""
    parent = _current_cgroup()
    if parent is None:
        return None
    group = parent / f"stateframe-{os.getpid()}-{ident}"
    try:
        group.mkdir()
        try:
            if not (group / "pids.max").is_file():
                group.rmdir()
                return None
            # Allow a few bubblewrap setup tasks. The in-namespace limit
            # independently bounds the check's tasks when delegation is absent.
            (group / "pids.max").write_text(str(nproc + 8), encoding="ascii")
        except OSError:
            group.rmdir()
            return None
    except OSError:
        return None
    return group


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
        resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
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


def main() -> int:
    try:
        if len(sys.argv) >= 4 and sys.argv[1] == "--nproc-budget":
            budget = int(sys.argv[2])
            if not 1 <= budget <= 256:
                raise ValueError("NPROC budget is outside the execution policy bound")
            _set_relative_nproc_limit(_user_task_count() + budget + _NPROC_HEADROOM)
            os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
        if len(sys.argv) >= 4 and sys.argv[1] == "--cgroup-enter":
            try:
                with open(os.path.join(sys.argv[2], "cgroup.procs"), "w", encoding="ascii") as stream:
                    stream.write(str(os.getpid()))
            except OSError:
                pass  # RLIMIT_NPROC remains the mandatory fallback.
            os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
        if len(sys.argv) >= 4 and sys.argv[1] in {"--network-none", "--process-filter"}:
            environment = json.loads(sys.argv[2])
            install_syscall_filter(network_none=sys.argv[1] == "--network-none")
            os.execvpe(sys.argv[3], sys.argv[3:], environment)
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
