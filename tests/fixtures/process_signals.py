"""Signals in tests are limited to held processes or verified launch identities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import ctypes
import errno
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    started_epoch: float
    command: str
    kernel_identity: str = ""


class _BSDInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "flags", "status", "xstatus", "pid", "ppid", "uid", "gid", "ruid", "rgid",
        "svuid", "svgid", "rfu")]
    _fields_ += [("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32)]
    _fields_ += [(name, ctypes.c_uint32) for name in (
        "nfiles", "pgid", "pjobc", "tdev", "tpgid", "nice")]
    _fields_ += [("start_sec", ctypes.c_uint64), ("start_usec", ctypes.c_uint64)]


def kernel_start_identity(pid: int) -> str:
    """Independent kernel birth reader for test signal authority."""
    if sys.platform == "linux":
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing = stat_text.rfind(") ")
        if closing < 0:
            raise RuntimeError(f"cannot parse process {pid} start ticks")
        ticks = int(stat_text[closing + 2:].split()[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-fA-F-]{36}", boot_id):
            raise RuntimeError("Linux boot identity is invalid")
        return f"linux:{boot_id}:{ticks}"
    if sys.platform == "darwin":
        lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int,
                                     ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        info = _BSDInfo()
        count = lib.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if count == 0:
            error = ctypes.get_errno()
            if error in (errno.ESRCH, errno.ENOENT):
                raise ProcessLookupError(f"process {pid} is no longer running")
            raise OSError(error, "cannot inspect test process identity", pid)
        if count != ctypes.sizeof(info):
            raise RuntimeError("incomplete macOS process identity")
        return f"darwin:{info.start_sec}:{info.start_usec}"
    raise RuntimeError(f"test process identity is unsupported on {sys.platform}")


def _process_identity(pid: int) -> ProcessIdentity:
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="],
        capture_output=True, text=True, check=False, timeout=5,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ProcessLookupError(f"process {pid} is no longer running")
    line = result.stdout.strip()
    match = re.match(r"^([A-Za-z]{3}\s+[A-Za-z]{3}\s+\d{1,2}\s+"
                     r"\d{2}:\d{2}:\d{2}\s+\d{4})\s+(.+)$", line)
    if match is None:
        raise RuntimeError(f"cannot verify process {pid} start time and command")
    try:
        started = datetime.strptime(match.group(1), "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError as error:
        raise RuntimeError(f"cannot verify process {pid} start time") from error
    command = match.group(2).strip()
    if not command:
        raise RuntimeError(f"cannot verify process {pid} command")
    return ProcessIdentity(pid, started, command, kernel_start_identity(pid))


def record_identity(pid: int, *, launched_after: float, command_fragment: str,
                    launched_before: float | None = None) -> ProcessIdentity:
    """Capture a launch-bound identity before using a PID obtained outside a Popen."""
    if type(pid) is not int or pid <= 0 or not command_fragment:
        raise ValueError("a positive pid and expected command are required")
    identity = _process_identity(pid)
    if (identity.started_epoch < launched_after - 2 or identity.started_epoch > time.time() + 2
            or (launched_before is not None and identity.started_epoch > launched_before + 2)
            or command_fragment not in identity.command):
        raise RuntimeError(f"process {pid} does not match its launch identity")
    return identity


def _safe_target(pid: int) -> int:
    if type(pid) is not int or pid <= 0 or pid == os.getpid():
        raise ValueError("refusing an invalid or self process id")
    pgid = os.getpgid(pid)
    if pgid == os.getpgrp():
        raise ValueError("refusing the test runner's process group")
    return pgid


def signal_owned(popen: subprocess.Popen[object], sig: signal.Signals | int) -> None:
    """Signal a Popen held by this test, including its private process group."""
    if not isinstance(popen, subprocess.Popen) or popen.poll() is not None:
        raise ProcessLookupError("owned process is no longer running")
    pgid = _safe_target(popen.pid)
    if pgid == popen.pid:
        os.killpg(pgid, sig)
    else:
        popen.send_signal(sig)


def signal_verified(pid: int, recorded_identity: ProcessIdentity,
                    sig: signal.Signals | int) -> None:
    """Refuse a recycled PID or changed command before signalling its process/group."""
    if not isinstance(recorded_identity, ProcessIdentity) or recorded_identity.pid != pid:
        raise ValueError("recorded process identity does not match pid")
    pgid = _safe_target(pid)
    current = _process_identity(pid)
    if (current.kernel_identity != recorded_identity.kernel_identity
            or current.command != recorded_identity.command):
        raise RuntimeError(f"process {pid} no longer matches its recorded identity")
    if pgid == pid:
        os.killpg(pid, sig)
    else:
        os.kill(pid, sig)
