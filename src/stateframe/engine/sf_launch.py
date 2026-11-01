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
    # The supervisor bounds tracked descendants in the command's PID namespace.
    # RLIMIT_NPROC is uid-wide on Linux and can refuse a safe command when unrelated
    # same-uid threads exist, including tasks hidden in another PID namespace.
    for kind, value in process_limits:
        # Never silently keep a weaker limit if the OS refuses this request.
        hard = resource.getrlimit(kind)[1]
        value = min(value, hard) if hard != resource.RLIM_INFINITY else value
        resource.setrlimit(kind, (value, value))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.chdir(spec["cwd"])
    os.execvpe(argv[0], argv, spec["env"])


def main() -> int:
    try:
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
