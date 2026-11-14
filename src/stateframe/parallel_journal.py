"""Bounded JSON-lines journal process; no lifecycle handles cross this boundary.

Only the I/O reactor waits for the process. Producers use reserved bounded credits;
completion credits are returned when the lifecycle owner consumes the acknowledgment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import builtins
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time

_MAX_MESSAGE = 16 << 10
_MAX_PENDING = 64
# Helper restarts back off exponentially and are capped per request, so a helper that cannot
# start never becomes a fork storm and no caller waits on it forever.
_RESTART_BACKOFF_SECONDS = 0.05
_RESTART_BACKOFF_MAX_SECONDS = 2.0
_MAX_RESTARTS_PER_REQUEST = 6
# Progress watchdog: request writes and response reads wait in bounded slices. A helper that
# keeps accepting or producing bytes is never cut off however long it takes; one that transfers
# nothing for the no-progress window is wedged (hung fsync, SIGSTOP, swap) and is killed, since
# EOF on its stdin cannot unstick it.
_PROGRESS_SLICE_SECONDS = 0.25
_NO_PROGRESS_SLICES = 40
_HELPER_EXIT_WAIT_SECONDS = 1.0


class JournalBusy(RuntimeError):
    """The durable work lane is full; retain the intent and retry."""


class JournalFault(RuntimeError):
    """Durability is unavailable through helper loss, closure, or a filesystem failure;
    nothing was ACKed, so the caller may retry within its budget or defer."""


class _HelperWedged(OSError):
    """The helper transferred no bytes for the whole no-progress window."""


class JournalProtocolError(RuntimeError):
    """The helper answered and refused the operation for a reason other than an immutable
    record conflict (unknown operation, invalid payload). Retrying the same intent cannot
    succeed; the caller reports it instead of deferring."""


def checked_response(result: dict[str, object] | None) -> dict[str, object]:
    """Classify synchronous and polled ACKs identically, including retryable I/O failures."""
    if result is None:
        raise JournalFault("journal acknowledgment missing")
    if result.get("ok") is True:
        return result
    message = str(result.get("error", "journal operation was not acknowledged"))
    error_class = result.get("error_class")
    error_type = getattr(builtins, str(error_class), None)
    if (result.get("lost") is True or result.get("closed") is True
            or error_class is None
            or (isinstance(error_type, type) and issubclass(error_type, OSError))):
        raise JournalFault(message)
    if error_class == "ParallelHostError":
        from stateframe.parallel_host import ParallelHostError
        raise ParallelHostError(message)
    raise JournalProtocolError(message)


@dataclass
class Ticket:
    id: str
    kind: str
    payload: dict[str, object]
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, object] | None = None


class Journal:
    """One writer process with bounded outstanding requests, including unconsumed ACKs."""

    def __init__(self, *, role: str = "journal") -> None:
        self.role = role
        self.epoch = 0
        self._credits = threading.BoundedSemaphore(_MAX_PENDING)
        self._queue: queue.Queue[Ticket | None] = queue.Queue(_MAX_PENDING + 1)
        self._closed = False
        self.restart_failures = 0  # consecutive helper losses, reset by any ACK
        self.fault: str | None = None  # last restart-budget exhaustion, cleared by any ACK
        self._state_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="stateframe-journal-reactor")
        self._thread.start()

    def submit(self, kind: str, payload: dict[str, object]) -> Ticket:
        with self._state_lock:
            return self._submit(kind, payload)

    def _submit(self, kind: str, payload: dict[str, object]) -> Ticket:
        if self._closed:
            raise JournalBusy("journal is closing")
        raw = json.dumps({"kind": kind, "payload": payload}, sort_keys=True, separators=(",", ":"))
        if len(raw.encode("utf-8")) > _MAX_MESSAGE - 128:
            raise JournalBusy("journal control message exceeds 16 KiB")
        if not self._credits.acquire(blocking=False):
            raise JournalBusy("journal queue is full")
        ticket = Ticket(hashlib.sha256(raw.encode()).hexdigest(), kind, payload)
        self._queue.put_nowait(ticket)
        return ticket

    def take(self, ticket: Ticket) -> dict[str, object] | None:
        if not ticket.done.is_set():
            return None
        result = ticket.result
        if result is None:
            return None
        ticket.result = None
        self._credits.release()
        return result

    def call(self, kind: str, payload: dict[str, object]) -> dict[str, object]:
        ticket = self.submit(kind, payload)
        ticket.done.wait()  # channel I/O only, never the tick or watchdog
        return checked_response(self.take(ticket))

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put_nowait(None)  # reserved shutdown lane, outside ordinary credits

    def _spawn(self) -> subprocess.Popen:
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--" + self.role]
        self.epoch += 1
        return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)

    def _backoff(self, seconds: float) -> None:
        # Sliced so close() is never delayed by a long restart backoff.
        deadline = time.monotonic() + seconds
        while not self._closed:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, _RESTART_BACKOFF_SECONDS))

    @staticmethod
    def _release(process: subprocess.Popen, *, wedged: bool) -> None:
        """Let go of a lost helper. EOF on stdin asks a live child to exit; a wedged child cannot
        read that EOF, so it is killed through its own handle (never a numeric PID) and reaped
        with a bounded wait so it cannot linger as a zombie."""
        if wedged:
            try:
                process.kill()
            except OSError:
                pass
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if wedged:
            try:
                process.wait(timeout=_HELPER_EXIT_WAIT_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _write_request(self, process: subprocess.Popen, raw: bytes) -> None:
        """Send partial nonblocking writes under the same progress window as the ACK read."""
        assert process.stdin is not None
        descriptor = process.stdin.fileno()
        os.set_blocking(descriptor, False)
        remaining = memoryview(raw)
        window = _PROGRESS_SLICE_SECONDS * _NO_PROGRESS_SLICES
        deadline = time.monotonic() + window
        while remaining:
            # As with ACK reads, accept immediately available progress after descheduling.
            wait = max(0.0, deadline - time.monotonic())
            _, ready, _ = select.select([], [descriptor], [], min(wait, _PROGRESS_SLICE_SECONDS))
            if not ready:
                if time.monotonic() >= deadline:
                    # Do not kill on a timeout observed before reactor descheduling.
                    _, ready, _ = select.select([], [descriptor], [], 0.0)
                    if not ready:
                        raise _HelperWedged("journal helper made no progress writing request")
                else:
                    continue
            try:
                count = os.write(descriptor, remaining)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise _HelperWedged("journal helper made no progress writing request") from None
                continue  # readiness can change; the progress deadline still applies
            if count <= 0:
                raise OSError("journal helper lost while writing request")
            remaining = remaining[count:]
            deadline = time.monotonic() + window

    def _read_response(self, process: subprocess.Popen, buffer: bytearray) -> bytes:
        """Read one response line while counting no-progress slices (ACK-progress watchdog)."""
        assert process.stdout is not None
        descriptor = process.stdout.fileno()
        idle_slices = 0
        while True:
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[:newline + 1])
                del buffer[:newline + 1]
                if buffer:
                    raise OSError("journal helper sent unsolicited output")
                return line
            if len(buffer) > _MAX_MESSAGE:
                raise OSError("journal helper response exceeds its size limit")
            ready, _, _ = select.select([descriptor], [], [], _PROGRESS_SLICE_SECONDS)
            if not ready:
                idle_slices += 1
                if idle_slices >= _NO_PROGRESS_SLICES:
                    # The reactor may have been descheduled after select timed out.
                    # Consume bytes already available before acting on that stale timeout;
                    # this poll adds no waiting time to the no-progress window.
                    ready, _, _ = select.select([descriptor], [], [], 0.0)
                    if not ready:
                        raise _HelperWedged(f"journal helper made no progress for {idle_slices} slices")
                else:
                    continue
            chunk = os.read(descriptor, 1 << 16)
            if not chunk:
                raise OSError("journal helper lost or invalid response")
            buffer += chunk
            idle_slices = 0  # bytes arrived: the helper is making progress

    def _run(self) -> None:
        process = None
        buffer = bytearray()
        try:
            while True:
                ticket = self._queue.get()
                if ticket is None:
                    break
                restarts = 0
                while not self._closed:
                    try:
                        if process is None:
                            process = self._spawn()
                            self._process = process
                            buffer = bytearray()
                        assert process.stdin is not None and process.stdout is not None
                        message = {"id": ticket.id, "kind": ticket.kind, "payload": ticket.payload}
                        self._write_request(process, json.dumps(message, separators=(",", ":")).encode() + b"\n")
                        raw = self._read_response(process, buffer)
                        if len(raw) > _MAX_MESSAGE:
                            raise OSError("journal helper lost or invalid response")
                        result = json.loads(raw)
                        if not isinstance(result, dict) or result.get("id") != ticket.id:
                            raise OSError("journal helper response identity mismatch")
                        ticket.result = result
                        ticket.done.set()
                        self.restart_failures = 0
                        self.fault = None
                        break
                    except (OSError, ValueError) as error:
                        # Loss is never ACK. Same digest-bound intent is retried after restart.
                        spawn_failed = process is None  # no helper ever held this request
                        self.epoch += 1
                        if process is not None:
                            self._release(process, wedged=isinstance(error, _HelperWedged))
                            process = None
                        self._process = None
                        self.restart_failures += 1
                        if self.role == "seal" and not spawn_failed:
                            ticket.result = {"ok": False, "lost": True, "error": "seal helper lost"}
                            ticket.done.set()
                            break  # publication permits MUST NOT replay without a fresh owner proof
                        # A seal helper that never started consumed no proof: the same request
                        # waits through the shared backoff and cap instead of being resubmitted
                        # by the lifecycle lane at visit rate (fork storm, permanent reservation).
                        restarts += 1
                        if restarts >= _MAX_RESTARTS_PER_REQUEST:
                            self.fault = (f"journal helper unavailable after {restarts} restarts: "
                                          f"{type(error).__name__}: {str(error)[:200]}")
                            ticket.result = {"ok": False, "lost": True, "spawn_failed": spawn_failed,
                                             "error": self.fault}
                            ticket.done.set()
                            break  # the caller decides whether to retry; no silent spin
                        # Consecutive losses across requests escalate straight to the long backoff.
                        step = min(self.restart_failures, _MAX_RESTARTS_PER_REQUEST) - 1
                        self._backoff(min(_RESTART_BACKOFF_SECONDS * 2 ** step,
                                          _RESTART_BACKOFF_MAX_SECONDS))
                if self._closed and not ticket.done.is_set():
                    ticket.result = {"ok": False, "closed": True,
                                     "error": "journal closed with operation pending"}
                    ticket.done.set()
        finally:
            if process is not None and process.stdin is not None:
                process.stdin.close()
            self._process = None


def commit_worker(payload: dict[str, object], counters: dict[str, int]) -> dict[str, object]:
    from stateframe.parallel_host import ParallelHostError, _read_json, _write_json

    path = Path(str(payload["path"]))
    value = dict(payload["value"])
    launch = payload.get("launch")
    if launch is not None:
        _write_json(path.with_name(path.stem + ".launch.json"), launch)
    if value["state"] == "EXITED":
        prior = _read_json(path, "durable worker") if path.exists() else None
        if prior is not None and prior.get("state") == "EXITED":
            if (prior.get("execution_id"), prior.get("incarnation")) != (value["execution_id"], value["incarnation"]):
                raise ParallelHostError("terminal commit identity conflict")
            sequence = prior["exited_seq"]
            if dict(value, exited_seq=sequence) != prior:
                raise ParallelHostError("terminal commit replay differs from durable state")
        else:
            root = path.parents[2]
            key = str(root)
            counter_path = root / "exited-sequence.json"
            if key not in counters:
                counter = _read_json(counter_path, "exit sequence") if counter_path.exists() else {"last": 0}
                if set(counter) != {"last"} or type(counter["last"]) is not int or counter["last"] < 0:
                    raise ParallelHostError("exit sequence counter is invalid")
                last = counter["last"]
                for record in (root / "units").glob("*/[0-9]*.json"):
                    if record.stem.isdecimal():
                        row = _read_json(record, "exit recovery record")
                        seq = row.get("exited_seq")
                        if type(seq) is int and seq > last:
                            raise ParallelHostError("exit counter trails durable exit records")
                counters[key] = last
            sequence = counters[key] + 1
            _write_json(counter_path, {"last": sequence}, replace_existing=True)
            counters[key] = sequence
        value["exited_seq"] = sequence
    _write_json(path, value, replace_existing=True)
    return {"value": value}


def serve() -> None:
    from stateframe.parallel_host import _write_json

    counters: dict[str, int] = {}
    for raw in iter(lambda: sys.stdin.buffer.readline(_MAX_MESSAGE + 1), b""):
        if len(raw) > _MAX_MESSAGE or not raw.endswith(b"\n"):
            return
        message = json.loads(raw)
        identifier = message["id"]
        try:
            payload = message["payload"]
            if message["kind"] == "write":
                _write_json(Path(payload["path"]), payload["value"],
                            replace_existing=payload["replace"], durable=payload["durable"])
                result = {}
            elif message["kind"] == "sync":
                from stateframe.parallel_channel import sync_record
                sync_record(payload)
                result = {}
            elif message["kind"] in {"seal_build", "seal_publish"}:
                from stateframe.parallel_seals import artifact_operation
                result = artifact_operation(message["kind"], payload)
            elif message["kind"] == "commit":
                result = commit_worker(payload, counters)
            else:
                raise ValueError("unknown journal operation")
            answer = {"id": identifier, "ok": True, **result}
        except Exception as error:
            # The class name travels separately so Journal.call can raise the matching typed error
            # (an immutable record conflict is authoritative; anything else is a protocol fault).
            answer = {"id": identifier, "ok": False, "error_class": type(error).__name__,
                      "error": f"{type(error).__name__}: {str(error)[:500]}"}
        sys.stdout.write(json.dumps(answer, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    # Explicit source path, independent of installed packages and the fresh test HOME.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    serve()
