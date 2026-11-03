"""Sandbox-local hook process driver with independent bounded streams."""

from __future__ import annotations

import base64
import json
import os
import selectors
import signal
import subprocess
import sys
import time

SCHEMA = "stateframe-hook-envelope/1"


def _stop_group(process: subprocess.Popen) -> bool:
    if not _child_unreaped(process):
        return False
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The group disappeared before the sweep; no members remain to kill.
        return True
    except PermissionError:
        # Darwin reports EPERM for a group containing only its exited leader.
        # Hooks cannot change UID inside this sandbox; a live member remains
        # signalable, so EPERM after leader exit means the sweep is complete.
        if sys.platform == "darwin" and _child_exited(process):
            return True
        return False
    return True


def _child_unreaped(process: subprocess.Popen) -> bool:
    """Only this driver's direct child may reserve the session's group ID."""
    if process.returncode is not None:
        return False
    if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        # poll() reaps on exit, so an exited leader loses signal authority.
        return process.poll() is None
    try:
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except OSError:
        return False
    return True


def _child_exited(process: subprocess.Popen) -> bool:
    if process.returncode is not None:
        return True
    if not hasattr(os, "waitid") or not hasattr(os, "WNOWAIT"):
        return process.poll() is not None
    try:
        return os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


def _envelope(**changes: object) -> dict:
    value = {
        "schema": SCHEMA,
        "exit_code": None,
        "signal": None,
        "timed_out": False,
        "launch_error": None,
        "stdout_b64": "",
        "stderr_b64": "",
        "stdout_total": 0,
        "stderr_total": 0,
        "cleanup_ok": True,
    }
    value.update(changes)
    return value


def run(request: dict) -> dict:
    try:
        if type(request) is not dict or set(request) != {
            "argv",
            "payload_b64",
            "timeout",
            "stream_cap",
        }:
            raise ValueError("invalid driver request fields")
        argv, timeout, cap = request["argv"], request["timeout"], request["stream_cap"]
        if (
            type(argv) is not list
            or not argv
            or any(type(item) is not str or not item or "\0" in item for item in argv)
        ):
            raise ValueError("argv must be nonempty strings")
        if type(timeout) is not int or not 1 <= timeout <= 590 or cap != 32768:
            raise ValueError("invalid timeout or stream cap")
        payload = base64.b64decode(request["payload_b64"], validate=True)
        if len(payload) > 256 << 10:
            raise ValueError("hook input exceeds 256 KiB")
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            close_fds=True,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        return _envelope(launch_error=f"{type(error).__name__}: {error}")

    selector = selectors.DefaultSelector()
    assert (
        process.stdin is not None
        and process.stdout is not None
        and process.stderr is not None
    )
    for stream, kind in ((process.stdout, "out"), (process.stderr, "err")):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, kind)
    os.set_blocking(process.stdin.fileno(), False)
    if payload:
        selector.register(process.stdin, selectors.EVENT_WRITE, "in")
    else:
        process.stdin.close()
    retained = {"out": bytearray(), "err": bytearray()}
    totals = {"out": 0, "err": 0}
    position = 0
    deadline = time.monotonic() + timeout
    killed_at = None
    timed_out = False
    cleanup_ok = True
    try:
        while selector.get_map() or not _child_exited(process):
            now = time.monotonic()
            if now >= deadline and killed_at is None:
                timed_out = True
                killed_at = now
                if not _stop_group(process):
                    cleanup_ok = False
                if not process.stdin.closed:
                    try:
                        selector.unregister(process.stdin)
                    except KeyError:
                        pass
                    process.stdin.close()
            if killed_at is not None and now - killed_at > 2 and selector.get_map():
                cleanup_ok = False
                break
            for key, _ in selector.select(0.05):
                stream = key.fileobj
                if key.data == "in":
                    try:
                        written = os.write(
                            stream.fileno(), payload[position : position + 65536]
                        )
                    except BrokenPipeError:
                        written = len(payload) - position
                    position += written
                    if position >= len(payload):
                        selector.unregister(stream)
                        stream.close()
                    continue
                try:
                    chunk = os.read(stream.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                kind = key.data
                totals[kind] += len(chunk)
                need = cap - len(retained[kind])
                retained[kind].extend(chunk[: max(0, need)])
        # The leader may have exited while a descendant held a stream open.
        # Signal before wait() reaps it and releases the group identity.
        if not _stop_group(process):
            cleanup_ok = False
        try:
            code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            cleanup_ok = False
            process.kill()
            code = process.wait(timeout=2)
    except (OSError, subprocess.SubprocessError) as error:
        cleanup_ok = False
        if not _stop_group(process):
            cleanup_ok = False
        try:
            code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            code = -signal.SIGKILL
        return _envelope(
            exit_code=code if code >= 0 else None,
            signal=-code if code < 0 else None,
            launch_error=f"driver stream failure: {error}",
            stdout_b64=base64.b64encode(retained["out"]).decode("ascii"),
            stderr_b64=base64.b64encode(retained["err"]).decode("ascii"),
            stdout_total=totals["out"],
            stderr_total=totals["err"],
            cleanup_ok=False,
        )
    finally:
        selector.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    return _envelope(
        exit_code=code if code >= 0 else None,
        signal=-code if code < 0 else None,
        timed_out=timed_out,
        stdout_b64=base64.b64encode(retained["out"]).decode("ascii"),
        stderr_b64=base64.b64encode(retained["err"]).decode("ascii"),
        stdout_total=totals["out"],
        stderr_total=totals["err"],
        cleanup_ok=cleanup_ok,
    )


def main() -> int:
    try:
        request = json.loads(sys.stdin.buffer.read(1 << 20))
        result = run(request)
    except (ValueError, UnicodeError) as error:
        result = _envelope(launch_error=f"invalid driver input: {error}")
    sys.stdout.write(
        json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
