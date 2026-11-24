"""Read-only historical export and local trace delivery."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import select
import signal
import socket
import sqlite3
import stat
import fcntl
import subprocess
import sys
import threading
import time
from typing import Mapping

from ..engine.kernel_safe_paths import safe_home
from .authority import admit, application_credentials, config_path_allowed, has_application_alias, transport_paths_allowed
from .config import Settings, resolve
from .identity import from_seed
from .operations import CHECKS, RUNTIMES, terminal_fields
from .projection import Projector
from .schema import coverage_merge
from .sources import Reader, policy_view
from .store import DB_FILENAME, Store
from .transport import destination_ids
from .types import Coverage, Cursor, Family, Namespace, Reason, ResultKind, SourceOccurrence


_MAX_IPC_FRAME = 256 * 1024 + 1024
_STATE_DIR = ".stateframe-otel"
_STATE_FILES = frozenset(DB_FILENAME + suffix for suffix in ("", "-journal", "-wal", "-shm"))
_MAX_SENDER_FRAME = 1024
_EXPORT_REFUSALS = {
    "source_unavailable": "stateframe-otel: export source unavailable",
    "source_busy": "stateframe-otel: export source busy",
    "admission_incomplete": "stateframe-otel: export admission incomplete",
    "summary_missing": "stateframe-otel: export summary missing",
    "summary_refused": "stateframe-otel: export summary refused",
    "config_refused": "stateframe-otel: export config refused",
    "transport_refused": "stateframe-otel: export transport path refused",
    "source_changed": "stateframe-otel: export source changed",
}


def worker_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    result: dict[str, str] = {}
    if isinstance(source.get("PATH"), str):
        result["PATH"] = source["PATH"]
    result["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONSAFEPATH"] = "1"
    return result


@dataclass(slots=True)
class SenderTask:
    child: subprocess.Popen
    channel: socket.socket
    lease_fd: int
    deadline_ns: int
    buffer: bytearray
    result: object | None = None


def protect_process(*, sender: bool = False) -> None:
    """Disable core dumps before any destination or TLS input is read."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sender:
        _cap_resource(resource.RLIMIT_NOFILE, 32)
        _cap_resource(resource.RLIMIT_CPU, 8)
        if hasattr(resource, "RLIMIT_FSIZE"):
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    if sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        prctl = getattr(libc, "prctl", None)
        if prctl is None or prctl(4, 0, 0, 0, 0) != 0:
            raise OSError("isolation_unverified")


def _cap_resource(kind: int, cap: int) -> None:
    soft, hard = resource.getrlimit(kind)
    hard_cap = cap if hard == resource.RLIM_INFINITY else min(cap, hard)
    soft_cap = min(hard_cap, soft) if soft != resource.RLIM_INFINITY else hard_cap
    resource.setrlimit(kind, (soft_cap, hard_cap))


def _notice(message: str, code: int) -> int:
    print(f"stateframe-otel: export {message}", file=sys.stderr)
    return code


def _refuse(reason: str) -> int:
    print(_EXPORT_REFUSALS[reason], file=sys.stderr)
    return 2


def _source_busy(root: Path) -> bool:
    return _run_lock_held(root) or _run_lock_held(root, ".engine.lock")


def _sampled(config, trace_id: bytes) -> bool:
    if config.sampler == "always_off":
        return False
    if config.sampler == "traceidratio":
        return int.from_bytes(trace_id, "big") < int(config.sample_ratio * (1 << 128))
    return True


def export(root: Path, *, settings=(), stopped: bool = False, trust_source: bool = False) -> int:
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return _notice("disabled", 2)
    root = Path(os.path.abspath(root))
    reader = store = lock = None
    try:
        protect_process()
        view = policy_view(root)
        if view.seed is None:
            return _refuse("source_unavailable")
        if _source_busy(root):
            return _refuse("source_busy")
        if not admit(root, view).complete:
            return _refuse("admission_incomplete")
        from .authority import _read_record
        summary_path = root / "summary.json"
        summary = _read_record(summary_path)
        if not stopped and (summary is None or type(summary.get("runtime_outcome")) is not str):
            return _refuse("summary_refused" if os.path.lexists(summary_path) else "summary_missing")
        cli_settings = Settings(**dict(settings))
        if not config_path_allowed(cli_settings.config, root, view.grants):
            return _refuse("config_refused")
        credentials = application_credentials(os.environ, _read_record(root / "spec.json"))
        resolved = resolve(cli_settings, os.environ, credentials)
        destination = resolved.traces
        if destination is None or resolved.diagnostics:
            return _refuse("config_refused")
        if not transport_paths_allowed(destination, root, view.grants):
            return _refuse("transport_refused")
        home = safe_home()
        if home is None:
            return _notice("not delivered (state unavailable)", 1)
        state_path = home / _STATE_DIR
        if state_path.is_relative_to(root):
            return _refuse("config_refused")
        state_root = _state_root(create=True)
        if state_root is None:
            return _notice("not delivered (state unavailable)", 1)
        lock = _lock_installation(state_root)
        if lock is None:
            return _notice("not delivered (store busy)", 1)
        key = _identity_key(state_root, create=True)
        if key is None:
            return _notice("not delivered (state unavailable)", 1)
        run_ref, trace_id = from_seed(key, view.seed)
        store = Store(state_root, key)
        if not trust_source and not store.known(trace_id):
            return _notice("input refused (add --trust-source if you trust its records)", 2)
        dest, legacy = destination_ids(key, destination, store.destinations(trace_id))
        reader = Reader(root, key)
        occurrences, coverage = reader.poll()
        if _source_busy(root):
            return _refuse("source_busy")
        if policy_view(root) != view:
            return _refuse("source_changed")
        if any(reason in (Reason.SOURCE_BUSY, Reason.SOURCE_UNTRUSTED) for reason, _ in coverage.reasons):
            return _refuse("admission_incomplete")
        projector = Projector(key, run_ref, trace_id, levels=resolved.levels,
                              detail_attempt=view.attempt, price_rates=resolved.price_rates)
        spans = []
        spaces = set()
        source_credentials = set(credentials)
        for occurrence in occurrences:
            spaces.add(occurrence.space)
            if occurrence.family is Family.SPEC:
                source_credentials.update(application_credentials(os.environ, occurrence.row))
            spans.extend(span for item in projector.consume(occurrence) for span in item.spans)
        if source_credentials.difference(credentials) and has_application_alias(os.environ, destination, tuple(source_credentials)):
            return _refuse("config_refused")
        if stopped and summary is None:
            space = Namespace(run_ref, view.attempt, None, None)
            boundary = SourceOccurrence(space, Family.STOPPED, 1, None, {},
                                        Cursor(bytes(16), bytes(32), 0, 1, None))
            spans.extend(span for item in projector.consume(boundary) for span in item.spans)
        for space in sorted(spaces, key=lambda item: (item.attempt, item.unit or b"", item.incarnation or 0)):
            spans.extend(span for item in projector.seal(space, Coverage(True, ())) for span in item.spans)
        coverage = coverage_merge(coverage, projector.coverage)
        if stopped and summary is None:
            coverage = coverage_merge(coverage, Coverage(False, ((Reason.SOURCE_MISSING, None),)))
        runtime, check, *_ = terminal_fields(summary or {})
        if _source_busy(root) or policy_view(root) != view:
            return _refuse("source_changed")
        sampled = _sampled(resolved, trace_id)
        store.record(dest, trace_id, tuple(spans), runtime=runtime, check=check,
                     complete=coverage.complete, reasons=coverage.reasons, sampled=sampled,
                     levels=resolved.levels, legacy=legacy)
        while sampled and (batch := store.next_batch(dest, trace_id)) is not None:
            result = _send(batch, dest, destination)
            if result.kind in {ResultKind.ACCEPTED, ResultKind.WARNING} and result.accepted == len(batch.spans):
                store.settle(dest, trace_id, batch, partial=False)
                continue
            if result.kind is ResultKind.PARTIAL:
                store.settle(dest, trace_id, batch, partial=True)
                continue
            reason = (result.reason or Reason.PEER_FAILURE).value
            return _notice(f"not delivered (receiver failure: {reason})", 1)
        if sampled and store.partial(dest, trace_id):
            return _notice("not delivered (receiver accepted only part of a batch)", 1)
        if not coverage.complete:
            reasons = ", ".join(reason.value for reason, _ in coverage.reasons)
            return _notice(f"incomplete coverage ({reasons})", 1)
        return 0
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return _notice("not delivered (local export failure)", 1)
    finally:
        if reader is not None:
            reader.close()
        if store is not None:
            store.close()
        if lock is not None:
            os.close(lock)


def status(run: Path) -> tuple[int, dict[str, object]]:
    root = Path(os.path.abspath(run))
    view = policy_view(root)
    report = {"schema": "stateframe-otel-status/1", "signals": {}, "history": "absent", "complete": False}
    if view.seed is None or not admit(root, view).complete:
        return 2, {**report, "reason": "source_untrusted"}
    try:
        state_root = _state_root(create=False)
        key = _identity_key(state_root, create=False) if state_root is not None else None
        if key is None:
            return 1, report
        _, trace_id = from_seed(key, view.seed)
        report["trace_id"] = trace_id.hex()
        store = Store(state_root, key, read_only=True)
        try:
            stored = store.status(trace_id)
        finally:
            store.close()
        if stored is None:
            return 1, report
        if stored["runtime"] not in RUNTIMES or stored["check"] not in CHECKS:
            raise ValueError("invalid_field")
        stored["reasons"] = [[Reason(reason).value, count] for reason, count in stored["reasons"]
                             if count is None or type(count) is int and 0 <= count <= (1 << 53) - 1]
        return 0, {**report, **stored, "history": "known"}
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return 1, {**report, "reason": "state_unavailable"}


def _lock_installation(root: Path) -> int | None:
    fd = _open_private_lock(root / "install.lock")
    if fd is None:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def purge() -> int:
    lock = None
    try:
        root = _state_root(create=False)
        if root is None:
            return 0
        lock = _lock_installation(root)
        if lock is None:
            return 2
        files = []
        if any(path.name not in _STATE_FILES | {"identity.key", "install.lock"} for path in root.iterdir()):
            return 2
        for name in _STATE_FILES:
            path = root / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or info.st_mode & 0o077):
                return 2
            files.append(path)
        for path in files:
            path.unlink()
        return 2 if any(path.name not in {"identity.key", "install.lock"} for path in root.iterdir()) else 0
    except OSError:
        return 2
    finally:
        if lock is not None:
            os.close(lock)


def main(argv: list[str] | None = None) -> int:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = sys.argv[1:] if argv is None else argv
    try:
        fields = _parse_internal_args(args)
        protect_process(sender=True)
        return _sender_main(**fields)
    except (OSError, ValueError):
        return 1


def _parse_internal_args(args: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    fields = {"--sender-fd": "sender_fd", "--lease-fd": "lease_fd"}
    index = 0
    while index < len(args):
        name = args[index]
        target = fields.get(name)
        if target is None or target in result or index + 1 >= len(args):
            raise ValueError("invalid_field")
        raw = args[index + 1]
        if not raw.isascii() or not raw.isdigit() or len(raw) > 10:
            raise ValueError("invalid_field")
        result[target] = int(raw)
        index += 2
    if (set(result) != {"sender_fd", "lease_fd"}
            or len(set(result.values())) != len(result)):
        raise ValueError("invalid_field")
    return result


def _spawn_sender(batch, dest: bytes, settings, deadline_ns: int) -> SenderTask | None:
    child = parent = child_sock = None
    lease_read = lease_write = None
    try:
        from . import schema

        parent, child_sock = socket.socketpair()
        parent.settimeout(0.25)
        lease_read, lease_write = os.pipe()
        for fd in (child_sock.fileno(), lease_read):
            os.set_inheritable(fd, False)
        python_executable = sys.executable
        if not python_executable:
            raise OSError("python executable unavailable")
        child = subprocess.Popen(
            [python_executable, "-c", "from stateframe.otel.companion import main; raise SystemExit(main())",
             "--sender-fd", str(child_sock.fileno()), "--lease-fd", str(lease_read)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, pass_fds=(child_sock.fileno(), lease_read), start_new_session=True,
            env=worker_environment(),
        )
        child_sock.close()
        child_sock = None
        frames = (
            _ipc_frame(1, {"settings": _settings_wire(settings)}),
            _ipc_frame(2, {"dest": dest, "signal": "traces", "protocol": settings.protocol,
                           "timeout_ms": settings.timeout_ms}),
            _ipc_frame(3, {"dest": dest, "deadline_ns": deadline_ns,
                           "batch": schema.pack_local(batch)}),
        )
        for frame in frames:
            parent.sendall(frame)
        parent.setblocking(False)
        os.close(lease_read)
        lease_read = None
        return SenderTask(child, parent, lease_write, deadline_ns, bytearray())
    except BaseException:
        for channel in (parent, child_sock):
            if channel is not None:
                try:
                    channel.close()
                except OSError:
                    pass
        for fd in (lease_read, lease_write):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if child is not None:
            _reap_owned_child(child)
        return None


def _send(batch, dest: bytes, settings):
    """Send one batch through a disposable worker with a bounded transport deadline."""
    deadline_ns = time.monotonic_ns() + settings.timeout_ms * 1_000_000
    sender = _spawn_sender(batch, dest, settings, deadline_ns)
    if sender is None:
        return _unknown_result()
    try:
        while time.monotonic_ns() < deadline_ns:
            done, result = _poll_sender(sender)
            if done:
                return result
            try:
                select.select([sender.channel], [], [], min(
                    0.05, max(0, deadline_ns - time.monotonic_ns()) / 1_000_000_000,
                ))
            except (OSError, ValueError):
                break
        return _unknown_result()
    except BaseException:
        return _unknown_result()
    finally:
        _close_sender(sender)
        _reap_owned_child(sender.child)


def _poll_sender(sender: SenderTask):
    _read_sender_bytes(sender.channel, sender.buffer)
    result = _take_sender_result(sender.buffer)
    if result is not None:
        sender.result = result
    code = sender.child.poll()
    if sender.result is not None and code is not None:
        return True, sender.result
    if code is not None:
        return True, _unknown_result()
    if time.monotonic_ns() >= sender.deadline_ns:
        return True, _unknown_result()
    return False, None


def _read_sender_bytes(channel: socket.socket, buffer: bytearray) -> None:
    try:
        while True:
            part = channel.recv(4096)
            if not part:
                return
            buffer.extend(part)
            if len(buffer) > _MAX_SENDER_FRAME + 4:
                del buffer[:-(_MAX_SENDER_FRAME + 4)]
    except BlockingIOError:
        return
    except OSError:
        return


def _take_sender_result(buffer: bytearray):
    if len(buffer) < 4:
        return None
    size = int.from_bytes(buffer[:4], "big")
    if not 1 <= size <= _MAX_SENDER_FRAME:
        buffer.clear()
        return _unknown_result()
    if len(buffer) < 4 + size:
        return None
    frame = bytes(buffer[:4 + size])
    del buffer[:4 + size]
    return _result_from_wire(frame)


def _reap_owned_child(child: subprocess.Popen) -> None:
    try:
        child.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        from ..engine.kernel_hook_driver import _stop_group

        _stop_group(child)
        child.wait()
    except OSError:
        return


def _close_sender(sender: SenderTask) -> None:
    try:
        sender.channel.close()
    except OSError:
        pass
    try:
        os.close(sender.lease_fd)
    except OSError:
        pass


def _sender_main(sender_fd: int, lease_fd: int) -> int:
    for fd in (sender_fd, lease_fd):
        os.set_inheritable(fd, False)
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_lease, args=(lease_fd, stop), daemon=True)
    watcher.start()
    sock = socket.socket(fileno=sender_fd)
    try:
        first = _recv_frame(sock, 32 * 1024)
        second = _recv_frame(sock, 1024)
        third = _recv_frame(sock, _MAX_IPC_FRAME)
        if (first is None or first[0] != 1 or second is None or second[0] != 2
                or third is None or third[0] != 3):
            return 1
        settings_raw = first[1].get("settings")
        destination = second[1].get("dest")
        signal_name = second[1].get("signal")
        protocol = second[1].get("protocol")
        timeout_ms = second[1].get("timeout_ms")
        request = third[1]
        if (not isinstance(destination, bytes) or len(destination) != 16
                or type(timeout_ms) is not int or not 100 <= timeout_ms <= 30_000
                or signal_name != "traces"
                or request["dest"] != destination or request["deadline_ns"] <= 0):
            return 1
        from . import schema
        from .transport import TransportSettings, send
        settings = _settings_from_wire(TransportSettings, settings_raw)
        if settings.protocol != protocol or settings.timeout_ms != timeout_ms:
            return 1
        batch = schema.unpack_local(request["batch"])
        if getattr(batch.signal, "value", batch.signal) != signal_name:
            return 1
        result = send(batch, settings, request["deadline_ns"])
        sock.sendall(_ipc_frame(4, _result_wire(result)))
        return 0
    except BaseException:
        try:
            sock.sendall(_ipc_frame(4, _result_wire(_unknown_result())))
        except BaseException:
            pass
        return 1
    finally:
        stop.set()
        sock.close()
        try:
            os.close(lease_fd)
        except OSError:
            pass


def _watch_lease(fd: int, stop: threading.Event | None = None) -> None:
    while stop is None or not stop.is_set():
        try:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if ready and not os.read(fd, 1):
                os._exit(70)
        except OSError:
            os._exit(70)


def _ipc_frame(tag: int, payload) -> bytes:
    if tag == 1:
        if type(payload) is not dict or set(payload) != {"settings"}:
            raise ValueError("invalid_field")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        limit = 32 * 1024
    elif tag == 2:
        if (type(payload) is not dict or set(payload) != {"dest", "signal", "protocol", "timeout_ms"}
                or type(payload["dest"]) is not bytes or len(payload["dest"]) != 16):
            raise ValueError("invalid_field")
        body = json.dumps({"dest": payload["dest"].hex(), "signal": payload["signal"],
                           "protocol": payload["protocol"], "timeout_ms": payload["timeout_ms"]}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True).encode("ascii")
        limit = 1024
    elif tag == 3:
        if (type(payload) is not dict or set(payload) != {"dest", "deadline_ns", "batch"}
                or type(payload["dest"]) is not bytes or len(payload["dest"]) != 16
                or type(payload["deadline_ns"]) is not int or payload["deadline_ns"] <= 0
                or type(payload["batch"]) is not bytes):
            raise ValueError("invalid_field")
        if payload["deadline_ns"] > 0xFFFFFFFFFFFFFFFF or len(payload["batch"]) > 256 * 1024:
            raise ValueError("payload_rejected")
        body = (payload["dest"] + payload["deadline_ns"].to_bytes(8, "big") + payload["batch"])
        limit = _MAX_IPC_FRAME
    elif tag == 4:
        if type(payload) is not dict or set(payload) != {
            "kind", "accepted", "rejected", "reason", "retry_delay_s", "may_have_delivered",
        }:
            raise ValueError("invalid_field")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        limit = _MAX_SENDER_FRAME
    else:
        raise ValueError("invalid_field")
    frame = bytes((tag,)) + body
    if len(body) > limit or len(frame) > _MAX_IPC_FRAME:
        raise ValueError("payload_rejected")
    return len(frame).to_bytes(4, "big") + frame


def _recv_frame(sock: socket.socket, limit: int):
    prefix = _recv_exact(sock, 4)
    if len(prefix) != 4:
        return None
    length = int.from_bytes(prefix, "big")
    if not 1 <= length <= limit:
        return None
    raw = _recv_exact(sock, length)
    if len(raw) != length:
        return None
    tag = raw[0]
    if tag == 3:
        payload = raw[1:]
        if len(payload) < 24 or len(payload) > 24 + 256 * 1024:
            return None
        return tag, {"dest": payload[:16], "deadline_ns": int.from_bytes(payload[16:24], "big"),
                     "batch": payload[24:]}
    value = json.loads(raw[1:].decode("ascii"), object_pairs_hook=_unique_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid_field")))
    if type(value) is not dict:
        return None
    if tag == 1:
        return (tag, value) if set(value) == {"settings"} and type(value["settings"]) is dict else None
    if tag == 2:
        if set(value) != {"dest", "signal", "protocol", "timeout_ms"}:
            return None
        dest = value.get("dest")
        if type(dest) is not str or len(dest) != 32:
            return None
        return tag, {"dest": bytes.fromhex(dest), "signal": value.get("signal"),
                     "protocol": value.get("protocol"),
                     "timeout_ms": value.get("timeout_ms")}
    if tag == 4:
        return (tag, value) if set(value) == {
            "kind", "accepted", "rejected", "reason", "retry_delay_s", "may_have_delivered",
        } else None
    return None


def _result_wire(result) -> dict[str, object]:
    from .types import Reason, ResultKind

    try:
        kind = ResultKind(getattr(result.kind, "value", result.kind)).value
        reason_raw = getattr(result, "reason", None)
        reason = Reason(getattr(reason_raw, "value", reason_raw)).value if reason_raw is not None else None
        accepted = _wire_count(result.accepted)
        rejected = _wire_count(result.rejected)
        retry = result.retry_delay_s
        if retry is not None and (type(retry) not in {float, int} or not 0 <= retry <= 30):
            raise ValueError("invalid_field")
        if type(result.may_have_delivered) is not bool:
            raise ValueError("invalid_field")
        return {"kind": kind, "accepted": accepted, "rejected": rejected,
                "reason": reason, "retry_delay_s": retry,
                "may_have_delivered": result.may_have_delivered}
    except (AttributeError, TypeError, ValueError):
        fallback = _unknown_result()
        return {"kind": fallback.kind.value, "accepted": None, "rejected": None,
                "reason": fallback.reason.value, "retry_delay_s": None,
                "may_have_delivered": True}


def _wire_count(value):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= (1 << 53) - 1:
        raise ValueError("invalid_field")
    return value


def _result_from_wire(raw: bytes):
    from .types import ExportResult, Reason, ResultKind
    if not raw:
        return _unknown_result()
    decoded = _recv_bytes_frame(raw)
    if decoded is None or decoded[0] != 4:
        return _unknown_result()
    value = decoded[1]
    try:
        if set(value) != {"kind", "accepted", "rejected", "reason", "retry_delay_s", "may_have_delivered"}:
            raise ValueError("invalid_field")
        reason = Reason(value["reason"]) if value["reason"] is not None else None
        accepted = _wire_count(value["accepted"])
        rejected = _wire_count(value["rejected"])
        retry = value["retry_delay_s"]
        if retry is not None and (type(retry) not in {float, int} or not 0 <= retry <= 30):
            raise ValueError("invalid_field")
        if type(value["may_have_delivered"]) is not bool:
            raise ValueError("invalid_field")
        return ExportResult(ResultKind(value["kind"]), accepted, rejected, reason,
                            retry, value["may_have_delivered"])
    except (KeyError, ValueError, TypeError):
        return _unknown_result()


def _recv_bytes_frame(raw: bytes):
    if len(raw) < 5 or int.from_bytes(raw[:4], "big") != len(raw) - 4:
        return None
    tag = raw[4]
    try:
        value = json.loads(raw[5:].decode("ascii"), object_pairs_hook=_unique_pairs)
    except (ValueError, UnicodeError, json.JSONDecodeError):
        return None
    return (tag, value) if tag == 4 and type(value) is dict else None


def _settings_wire(settings) -> dict[str, object]:
    return {
        "endpoint": settings.endpoint, "protocol": settings.protocol,
        "headers": [[name, value] for name, value in settings.headers],
        "compression": settings.compression, "timeout_ms": settings.timeout_ms,
        "ca": str(settings.ca) if settings.ca is not None else None,
        "cert": str(settings.cert) if settings.cert is not None else None,
        "key_file": str(settings.key_file) if settings.key_file is not None else None,
    }


def _settings_from_wire(cls, raw):
    if (type(raw) is not dict or set(raw) != {
        "endpoint", "protocol", "headers", "compression", "timeout_ms", "ca", "cert", "key_file",
    }):
        raise ValueError("invalid_field")
    endpoint, protocol, headers, compression, timeout = (
        raw["endpoint"], raw["protocol"], raw["headers"], raw["compression"], raw["timeout_ms"]
    )
    if (type(endpoint) is not str or len(endpoint) > 2048
            or protocol not in {"http/protobuf", "http/json"}
            or compression not in {"none", "gzip"}
            or type(timeout) is not int or not 100 <= timeout <= 30_000
            or type(headers) is not list or len(headers) > 32):
        raise ValueError("invalid_field")
    parsed_headers = []
    for pair in headers:
        if (type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str
                or type(pair[1]) is not str or len(pair[0]) > 256 or len(pair[1]) > 8192):
            raise ValueError("invalid_field")
        parsed_headers.append((pair[0], pair[1]))
    tls_paths = []
    for name in ("ca", "cert", "key_file"):
        value = raw[name]
        if value is not None and (type(value) is not str or not Path(value).is_absolute() or len(value) > 4096):
            raise ValueError("invalid_field")
        tls_paths.append(Path(value) if value is not None else None)
    return cls(endpoint, protocol, tuple(parsed_headers), compression, timeout, *tls_paths)


def _unknown_result():
    from .types import ExportResult, Reason, ResultKind
    return ExportResult(ResultKind.UNKNOWN, None, None, Reason.PEER_FAILURE, None, True)


def _state_root(*, create: bool) -> Path | None:
    home = safe_home()
    if home is None:
        return None
    root = home / _STATE_DIR
    try:
        root.lstat()
    except FileNotFoundError:
        if not create:
            return None
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
    if not _private_directory(root):
        raise PermissionError("state_unavailable")
    return root


def _open_private_lock(path: Path) -> int | None:
    fd = None
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size != 0):
            os.close(fd)
            return None
        return fd
    except OSError:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        return None


def _private_directory(path: Path) -> bool:
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700
            or resolved != path):
        return False
    from .authority import unsafe_parent_mode

    current = resolved
    while True:
        try:
            ancestor = current.lstat()
        except OSError:
            return False
        if stat.S_ISLNK(ancestor.st_mode) or not stat.S_ISDIR(ancestor.st_mode):
            return False
        if unsafe_parent_mode(current, ancestor):
            return False
        if current.parent == current:
            return True
        current = current.parent


def _run_lock_held(root: Path, name: str = ".run.lock") -> bool:
    try:
        fd = os.open(root / name, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return False
    except (ImportError, OSError):
        return True
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
    finally:
        os.close(fd)


def _identity_key(root: Path, *, create: bool) -> bytes | None:
    path = root / "identity.key"
    try:
        flags = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
        writable = create and not any(os.path.lexists(root / name) for name in _STATE_FILES)
        fd = os.open(path, flags | (os.O_RDWR | os.O_CREAT if writable else os.O_RDONLY), 0o600)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
                return None
            value = os.read(fd, 33)
            if len(value) == 32:
                return value
            if not writable or len(value) > 32:
                return None
            value = os.urandom(32)
            os.lseek(fd, 0, os.SEEK_SET)
            offset = 0
            while offset < len(value):
                written = os.write(fd, value[offset:])
                if written <= 0:
                    raise OSError("state_unavailable")
                offset += written
            os.fsync(fd)
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return value
        finally:
            os.close(fd)
    except OSError:
        return None


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        data = sock.recv(size - len(chunks))
        if not data:
            break
        chunks.extend(data)
    return bytes(chunks)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid_field")
        result[key] = value
    return result
