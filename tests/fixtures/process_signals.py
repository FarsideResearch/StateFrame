"""Signals in tests are limited to held processes or verified launch identities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import ctypes
import errno
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from engine_run import EngineProcess


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


def signal_owned(popen: subprocess.Popen[object] | EngineProcess,
                 sig: signal.Signals | int) -> None:
    """Signal a held child or monitored child group without trusting a free PID."""
    if isinstance(popen, EngineProcess):
        if popen.returncode is not None:
            raise ProcessLookupError("owned process is no longer running")
        if popen.process_group_id != popen.pid:
            raise RuntimeError("monitored process group does not match its launch identity")
        pgid = _safe_target(popen.pid)
        if pgid != popen.process_group_id:
            raise RuntimeError("owned process is no longer in its monitored process group")
        os.killpg(pgid, sig)
        return
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


def monitored_group_exists(process_group_id: int) -> bool:
    """Probe a monitored child's process group without signalling it."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_checkpoint_saved(run_dir: Path, *, deadline: float, minimum: int = 1,
                              what: str) -> None:
    """Wait for complete checkpoint-save journal records before interrupting a run."""
    if minimum < 1:
        raise ValueError("minimum checkpoint saves must be positive")
    events = run_dir / "checkpoints" / "events.jsonl"
    while True:
        saved = 0
        last_event = None
        try:
            lines = events.read_bytes().splitlines(keepends=True)
        except FileNotFoundError:
            lines = ()
        for line in lines:
            if not line.endswith(b"\n"):
                continue
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                last_event = "<invalid JSON>"
                continue
            if isinstance(row, dict):
                last_event = row.get("event")
                saved += last_event == "checkpoint_saved"
        if saved >= minimum:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(
                f"{what}: checkpoint save not seen in {run_dir}; "
                f"saw {saved}/{minimum} saves; last event: {last_event!r}")
        time.sleep(min(0.1, remaining))


def signal_run_verified(run_dir: Path, pid: int, identity: ProcessIdentity,
                        sig: signal.Signals | int, *, deadline: float, what: str,
                        require_checkpoint_saved: bool = True) -> None:
    """Interrupt a verified run process only after its checkpoint writer finishes."""
    if require_checkpoint_saved:
        wait_for_checkpoint_saved(run_dir, deadline=deadline, what=what)
    signal_verified(pid, identity, sig)


def signal_run_owned(run_dir: Path, popen: subprocess.Popen[object] | EngineProcess,
                     sig: signal.Signals | int, *, deadline: float, what: str,
                     require_checkpoint_saved: bool = True, group: bool = False) -> None:
    """Interrupt a held run process, optionally including its private group."""
    if require_checkpoint_saved:
        wait_for_checkpoint_saved(run_dir, deadline=deadline, what=what)
    if group:
        signal_owned(popen, sig)
    else:
        if isinstance(popen, EngineProcess):
            if popen.returncode is not None or popen.process_group_id != popen.pid:
                raise ProcessLookupError("owned run process is no longer running")
            if _safe_target(popen.pid) != popen.process_group_id:
                raise RuntimeError("owned process is no longer in its monitored process group")
            os.kill(popen.pid, sig)
            return
        if not isinstance(popen, subprocess.Popen) or popen.poll() is not None:
            raise ProcessLookupError("owned run process is no longer running")
        popen.send_signal(sig)
