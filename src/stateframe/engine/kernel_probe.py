"""The optional observation probe.

This module owns the host side of ``probe``. The child is deliberately a
plain, fresh Python process: stdout is data, never a receipt, verdict, or
authority. The host creates the typed receipt after the process exits and
keeps the receipt identity outside the child's output stream.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

from kernel_test_isolation import isolated_command
from kernel_non_product_paths import NON_PRODUCT_PARTS, is_non_product


MODE_ENV = "STATEFRAME_PROBE"
MAX_CODE_BYTES = 8 * 1024
MAX_PROBE_CODE_BYTES = MAX_CODE_BYTES
MAX_CPU_SECONDS = 2.0
MAX_WALL_SECONDS = 5.0
MAX_MEMORY_BYTES = 512 * 1024 * 1024
MAX_CAPTURE_BYTES = 32 * 1024
MAX_SHOWN_TOKENS = 512
MAX_PROBES_PER_TASK = 6

_UNKNOWN_REVISION = "UNKNOWN"
_META_FD_ENV = "STATEFRAME_PROBE_META_FD"
_SNAPSHOT_ROOT_ENV = "STATEFRAME_PROBE_ROOT"
_EXCLUDED_PARTS = frozenset(NON_PRODUCT_PARTS)
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_VACUOUS_PROBE = re.compile(r"(?:\bassert\s+True\b|\bpass\s*$|\breturn\s+True\b)", re.MULTILINE)
_CLONE_PROBE = re.compile(r"(?:inspect\.getsource|open\s*\(|read_text\s*\(|ast\.parse)")


class ProbeStatus(str, Enum):
    COMPLETED = "COMPLETED"
    RAISED = "RAISED"
    TIMED_OUT = "TIMED_OUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    POLICY_DENIED = "POLICY_DENIED"
    RUNNER_ERROR = "RUNNER_ERROR"
    NOT_RUN = "NOT_RUN"
    STALE = "STALE"
    UNRUNNABLE = "UNRUNNABLE"


def observation_probe_mode() -> str:
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return observation_probe_mode() == "on"


def _digest_manifest(manifest: Mapping[str, str]) -> str:
    payload = json.dumps(sorted(manifest.items()), separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _included(relative: pathlib.PurePosixPath) -> bool:
    return bool(relative.parts) and not is_non_product(relative)


def _read_regular_file(source: pathlib.Path) -> tuple[bytes, int]:
    """Read one regular file without following a concurrent symlink swap."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(f"not a regular file: {source}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            return handle.read(), metadata.st_mode & 0o777
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _manifest(root: pathlib.Path) -> dict[str, str]:
    """Hash only positive-inclusion files in ``root``.

    Symlinks are excluded rather than resolved.  This prevents a link in the
    visible tree from smuggling a private or host path into the snapshot.
    """
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))
    result: dict[str, str] = {}
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = pathlib.Path(directory)
        relative_base = base.relative_to(root).as_posix()
        dirnames[:] = [name for name in dirnames
                       if not (base / name).is_symlink()
                       and not is_non_product(pathlib.PurePosixPath(
                           name if relative_base == "." else f"{relative_base}/{name}"))]
        for name in filenames:
            source = base / name
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(root).as_posix()
            pure = pathlib.PurePosixPath(relative)
            if not _included(pure):
                continue
            try:
                data, _ = _read_regular_file(source)
            except OSError:
                # A concurrent edit is handled by the before/after revision
                # comparison.  Do not turn it into an unsafe partial copy.
                raise
            result[relative] = hashlib.sha256(data).hexdigest()
    return result


def workspace_revision(root: pathlib.Path) -> str:
    return _digest_manifest(_manifest(pathlib.Path(root)))


def _copy_snapshot(root: pathlib.Path, destination: pathlib.Path) -> tuple[str, bool]:
    """Copy visible files and report whether the source stayed stable."""
    root = root.resolve()
    before = _manifest(root)
    for relative in sorted(before):
        source = root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Read and write bytes explicitly.  copy2 would preserve a symlink if a
        # race replaced a source after the manifest walk.
        data, mode = _read_regular_file(source)
        target.write_bytes(data)
        try:
            target.chmod(mode)
        except OSError:
            pass
    after = _manifest(root)
    return _digest_manifest(before), before == after


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value)


def _bound_shown(value: str) -> tuple[str, int, bool]:
    from kernel_tokenlab_v13 import bound_shown_bytes, probe_render_enabled
    if probe_render_enabled():
        # : byte-sliced + run-collapsed (the token re-join turned caret lines into '^ ^ ^' runs
        # the model echoed to the completion cap).  Off: the legacy token join below, byte-identical.
        shown, shown_bytes, truncated, runs = bound_shown_bytes(value)
        _PROBE_RENDER_PENDING[0] = (len(value.encode("utf-8", errors="replace")), shown_bytes, runs, truncated)
        return shown, shown_bytes, truncated
    tokens = _tokens(value)
    if len(tokens) <= MAX_SHOWN_TOKENS:
        return value, len(tokens), False
    shown = " ".join(tokens[:MAX_SHOWN_TOKENS]) + " ...[shown output truncated]"
    return shown, MAX_SHOWN_TOKENS, True


_PROBE_RENDER_PENDING: list = [None]  # (before, after, collapsed_runs, truncated) of the last render


def drain_probe_render(journal: object) -> bool:
    """own event of STATEFRAME_TOKENLAB_PROBE_RENDER for the last rendered probe output."""
    pending, _PROBE_RENDER_PENDING[0] = _PROBE_RENDER_PENDING[0], None
    if pending is None or not hasattr(journal, "emit"):
        return False
    journal.emit("probe_render_applied", producer="kernel_tokenlab_v13", dispatch=False,
                 bytes_before=pending[0], bytes_after=pending[1], collapsed_runs=pending[2],
                 shown_truncated=pending[3], transformed=bool(pending[2] or pending[3]))
    return True


@dataclass(frozen=True)
class ProbeDefinition:
    """Host-authored identity for one proposed observation."""

    definition_id: str
    code: str
    question: str | None
    revision: str
    task_id: str = ""
    code_digest: str = ""

    @classmethod
    def create(cls, code: str, question: str | None, revision: str,
               task_id: str = "", *, position: int = 0) -> "ProbeDefinition":
        encoded = code.encode("utf-8", errors="surrogatepass")
        code_digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        question_digest = hashlib.sha256(
            (question if isinstance(question, str) else "").encode(
                "utf-8", errors="surrogatepass")).hexdigest()
        from kernel_wire_identity import stable_wire_id
        return cls(
            definition_id=stable_wire_id(task_id, "probe-definition", position,
                                         code_digest, question_digest, revision),
            code=code,
            question=question if isinstance(question, str) and question else None,
            revision=revision,
            task_id=str(task_id or ""),
            code_digest=code_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-probe-definition/1",
            "definition_id": self.definition_id,
            "revision": self.revision,
            "task_id": self.task_id,
            "question": self.question,
            "code_digest": self.code_digest,
            "code_bytes": len(self.code.encode("utf-8", errors="surrogatepass")),
        }


@dataclass(frozen=True)
class ProbeReceipt:
    """Host observation, deliberately separate from the child's stdout."""

    definition_id: str
    receipt_id: str
    status: ProbeStatus
    revision: str
    captured_output: str = ""
    shown_output: str = ""
    shown_tokens: int = 0
    capture_truncated: bool = False
    shown_truncated: bool = False
    exit_code: int | None = None
    reason: str = ""
    authority: str = "HOST_OBSERVED"
    snapshot_id: str = ""
    wall_seconds: float = 0.0
    policy_events: tuple[str, ...] = field(default_factory=tuple)
    child_group_killed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProbeStatus):
            object.__setattr__(self, "status", ProbeStatus(str(self.status)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-probe-receipt/1",
            "definition_id": self.definition_id,
            "receipt_id": self.receipt_id,
            "status": self.status.value,
            "revision": self.revision,
            "output": self.shown_output,
            "capture_truncated": self.capture_truncated,
            "shown_truncated": self.shown_truncated,
            "shown_tokens": self.shown_tokens,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "authority": self.authority,
            "snapshot_id": self.snapshot_id,
            "wall_seconds": round(self.wall_seconds, 4),
            "policy_events": list(self.policy_events),
            "child_group_killed": self.child_group_killed,
        }

    def to_frame(self) -> dict[str, object]:
        """Return the bounded, model-facing projection.

        Code and question are intentionally absent.  A receipt ID is made by
        this host object; nothing printed by the child can create one.
        """
        return {
            "schema": "stateframe-probe-receipt/1",
            "receipt_id": self.receipt_id,
            "definition_id": self.definition_id,
            "revision": self.revision,
            "status": self.status.value,
            "output": self.shown_output,
            "capture_truncated": self.capture_truncated,
            "shown_truncated": self.shown_truncated,
            "shown_tokens": self.shown_tokens,
            "exit_code": self.exit_code,
            "reason": self.reason,
            "authority": self.authority,
        }


def render_probe_receipt(receipt: ProbeReceipt) -> str:
    """Render only the bounded host result for a tool response."""
    frame = receipt.to_frame()
    output = frame.pop("output", "")
    header = (f"probe {frame['status']} receipt={frame['receipt_id']} "
              f"revision={frame['revision']}")
    flags = []
    if frame.get("capture_truncated"):
        flags.append("capture_truncated=true")
    if frame.get("shown_truncated"):
        flags.append("shown_truncated=true")
    if frame.get("reason"):
        flags.append("reason=" + str(frame["reason"]))
    suffix = (" " + " ".join(flags)) if flags else ""
    return header + suffix + "\nHOST_OBSERVED: " + (output or "<no output>")


@dataclass
class ProbeBudget:
    count: int = 0
    limit: int = MAX_PROBES_PER_TASK
    attempts: int = 0

    def note_attempt(self) -> None:
        self.attempts += 1

    def available(self) -> bool:
        return self.count < self.limit

    def reserve(self) -> bool:
        if not self.available():
            return False
        self.count += 1
        return True


@dataclass(frozen=True)
class _ChildResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    capture_truncated: bool
    timed_out: bool
    child_group_killed: bool
    policy_events: tuple[str, ...]


def _wrapper_source(code: str) -> str:
    # The child may print anything, including a forged receipt.  The only data
    # sent on the side channel is a host-policy audit marker; no receipt fields
    # are accepted from this source.
    return """import json
import importlib.util
import os
import pathlib
import socket
import sys
import traceback

_events = []
_root = pathlib.Path(os.environ.get(%r, ".")).resolve()
_meta_fd = int(os.environ.get(%r, "-1"))
_excluded = %r

def _visible_pycache_read(relative):
    parts = tuple(relative.parts)
    if (parts.count("__pycache__") != 1 or len(parts) < 2
            or parts[-2] != "__pycache__" or not parts[-1].endswith(".pyc")):
        return False
    source_directory = _root.joinpath(*parts[:-2])
    candidate = _root.joinpath(*parts)
    try:
        sources = tuple(source_directory.iterdir())
    except OSError:
        return False
    for source in sources:
        if (not source.name.endswith(".py") or source.is_symlink()
                or not source.is_file()):
            continue
        source_relative = pathlib.PurePosixPath(*(parts[:-2] + (source.name,)))
        if any(part.startswith(".") or part in _excluded
               for part in source_relative.parts):
            continue
        try:
            expected = pathlib.Path(importlib.util.cache_from_source(str(source)))
        except (OSError, TypeError, ValueError):
            continue
        if expected == candidate:
            return True
    return False

def _path_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else None

def _root_path(value):
    value = _path_value(value)
    if not value:
        return None
    candidate = pathlib.Path(value)
    if not candidate.is_absolute():
        candidate = _root / candidate
    try:
        return candidate.resolve(strict=False).relative_to(_root)
    except (OSError, ValueError):
        return None

def _audit(event, args):
    if event == "open":
        value = args[0] if args else None
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else "r"
        flags = args[1] if len(args) > 1 and isinstance(args[1], int) else 0
        relative = _root_path(value)
        if (any(flag in mode for flag in ("w", "a", "x", "+")) or
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            _events.append("WRITE")
        elif relative is not None and _visible_pycache_read(relative):
            _events.append("PYCACHE_READ_ALLOWED")
        elif relative is not None and any(
                part.startswith(".") or part in _excluded for part in relative.parts):
            _events.append("HIDDEN_PATH")
    elif event == "os.open":
        value = args[0] if args else None
        flags = args[1] if len(args) > 1 and isinstance(args[1], int) else 0
        relative = _root_path(value)
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            _events.append("WRITE")
        elif relative is not None and _visible_pycache_read(relative):
            _events.append("PYCACHE_READ_ALLOWED")
        elif relative is not None and any(
                part.startswith(".") or part in _excluded for part in relative.parts):
            _events.append("HIDDEN_PATH")
    elif event in {"socket.connect", "socket.bind", "socket.listen", "socket.send",
                   "socket.sendto", "socket.sendmsg"}:
        _events.append("NETWORK")

sys.addaudithook(_audit)
sys.dont_write_bytecode = True
_exit = 0
try:
    exec(compile(%r, "<stateframe-probe>", "exec"),
         {"__name__": "__main__", "__file__": "<stateframe-probe>"}, {})
except SystemExit as error:
    _exit = error.code if isinstance(error.code, int) else 1
except BaseException:
    traceback.print_exc()
    _exit = 1
finally:
    if _meta_fd >= 0:
        try:
            os.write(_meta_fd, json.dumps({"policy_events": list(dict.fromkeys(_events))}).encode())
        except OSError:
            pass
sys.exit(_exit)
""" % (_SNAPSHOT_ROOT_ENV, _META_FD_ENV, tuple(_EXCLUDED_PARTS), code)


def _kill_group(process: subprocess.Popen) -> bool:
    killed = False
    try:
        os.killpg(process.pid, signal.SIGKILL)
        killed = True
    except (OSError, ProcessLookupError):
        pass
    try:
        process.kill()
        killed = True
    except (OSError, ProcessLookupError):
        pass
    return killed


def _collect_process(process: subprocess.Popen, wall_seconds: float,
                     capture_limit: int) -> tuple[bytes, bytes, bool, bool, bool]:
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    truncated = False
    streams = {}
    for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
        if stream is None:
            continue
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
        streams[stream.fileno()] = name
    deadline = time.monotonic() + max(0.001, wall_seconds)
    timed_out = False
    group_killed = False
    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = process.poll() is None
            if timed_out:
                group_killed = _kill_group(process) or group_killed
            break
        events = selector.select(min(remaining, 0.05))
        if not events and process.poll() is not None:
            # The next select drains EOF from each pipe.
            continue
        for key, _ in events:
            stream = key.fileobj
            name = key.data
            try:
                data = os.read(stream.fileno(), 8192)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if not data:
                try:
                    selector.unregister(stream)
                except Exception:
                    pass
                continue
            available = max(0, capture_limit - total)
            if available:
                buffers[name].extend(data[:available])
            if len(data) > available:
                truncated = True
            total += len(data)
    if timed_out:
        # Drain a killed group without allowing a detached descriptor to hold
        # the host forever.  Bytes beyond the cap are discarded.
        drain_deadline = time.monotonic() + 0.5
        while selector.get_map() and time.monotonic() < drain_deadline:
            for key, _ in selector.select(0.05):
                stream = key.fileobj
                name = key.data
                try:
                    data = os.read(stream.fileno(), 8192)
                except (BlockingIOError, OSError):
                    data = b""
                if not data:
                    try:
                        selector.unregister(stream)
                    except Exception:
                        pass
                    continue
                available = max(0, capture_limit - total)
                if available:
                    buffers[name].extend(data[:available])
                if len(data) > available:
                    truncated = True
                total += len(data)
    for key in list(selector.get_map().values()):
        try:
            selector.unregister(key.fileobj)
        except Exception:
            pass
    selector.close()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        group_killed = _kill_group(process) or group_killed
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), truncated, timed_out, group_killed


def _read_policy_metadata(read_fd: int) -> tuple[str, ...]:
    os.set_blocking(read_fd, False)
    data = bytearray()
    deadline = time.monotonic() + 0.25
    while time.monotonic() < deadline:
        try:
            chunk = os.read(read_fd, 8192)
        except BlockingIOError:
            time.sleep(0.005)
            continue
        except OSError:
            break
        if not chunk:
            break
        data.extend(chunk)
        if len(data) >= 8192:
            break
    try:
        parsed = json.loads(bytes(data).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError):
        return ()
    events = parsed.get("policy_events") if isinstance(parsed, dict) else ()
    if not isinstance(events, (list, tuple)):
        return ()
    return tuple(dict.fromkeys(str(item) for item in events if str(item)))


def _run_child(snapshot: pathlib.Path, code: str, *, executable: str,
               wall_seconds: float, cpu_seconds: float, memory_bytes: int) -> _ChildResult:
    meta_read, meta_write = os.pipe()
    os.set_inheritable(meta_write, True)
    env = dict(os.environ)
    env[_META_FD_ENV] = str(meta_write)
    env[_SNAPSHOT_ROOT_ENV] = str(snapshot)
    try:
        from kernel_execution_policy import sandboxed_popen
        process, _capabilities = sandboxed_popen(
            [executable, "-B", "-c", _wrapper_source(code)],
            snapshot,
            env=env,
            internal_env={_META_FD_ENV: meta_write, _SNAPSHOT_ROOT_ENV: snapshot},
            readonly_workspace=True,
            cpu_seconds=cpu_seconds,
            memory_bytes=memory_bytes,
            file_bytes=512 * 1024,
            pass_fds=(meta_write,),
            isolation_builder=isolated_command,
        )
    except BaseException:
        os.close(meta_read)
        raise
    finally:
        os.close(meta_write)
    try:
        stdout, stderr, capture_truncated, timed_out, group_killed = _collect_process(
            process, wall_seconds, MAX_CAPTURE_BYTES)
        policy_events = _read_policy_metadata(meta_read)
    finally:
        os.close(meta_read)
    return _ChildResult(process.returncode, stdout, stderr, capture_truncated,
                        timed_out, group_killed, policy_events)


class ProbeRunner:
    """Per-task bounded runner for observation-only snippets."""

    def __init__(self, root: pathlib.Path, *, task_id: str = "", budget: ProbeBudget | None = None,
                 task_deadline: float | None = None, finish_reserve_seconds: float = 0.0,
                 wall_seconds: float = MAX_WALL_SECONDS, cpu_seconds: float = MAX_CPU_SECONDS,
                 memory_bytes: int = MAX_MEMORY_BYTES,
                 executable: str | None = None):
        self.root = pathlib.Path(root).resolve()
        self.task_id = str(task_id or "")
        self._definition_ordinal = 0
        self.budget = budget if budget is not None else ProbeBudget()
        if task_deadline is None:
            try:
                task_deadline = float(os.environ["CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"])
            except (KeyError, TypeError, ValueError):
                task_deadline = None
        else:
            try:
                task_deadline = float(task_deadline)
            except (TypeError, ValueError):
                task_deadline = None
        self.task_deadline = task_deadline
        self.finish_reserve_seconds = max(0.0, float(finish_reserve_seconds))
        self.wall_seconds = min(MAX_WALL_SECONDS, max(0.001, float(wall_seconds)))
        self.cpu_seconds = min(MAX_CPU_SECONDS, max(0.001, float(cpu_seconds)))
        self.memory_bytes = min(MAX_MEMORY_BYTES, max(1, int(memory_bytes)))
        from kernel_test_python import selected_test_python
        self.executable = executable if isinstance(executable, str) and executable else selected_test_python()

    def _definition(self, code: str, question: str | None,
                    revision: str) -> ProbeDefinition:
        self._definition_ordinal += 1
        return ProbeDefinition.create(code, question, revision, self.task_id,
                                      position=self._definition_ordinal)

    def _receipt(self, definition: ProbeDefinition, status: ProbeStatus, *,
                 revision: str | None = None, reason: str = "", **kwargs) -> ProbeReceipt:
        from kernel_wire_identity import stable_wire_id
        return ProbeReceipt(
            definition_id=definition.definition_id,
            receipt_id=stable_wire_id(self.task_id, "probe-receipt",
                                      definition.definition_id, status.value),
            status=status,
            revision=revision or definition.revision,
            reason=reason,
            **kwargs,
        )

    def _deadline(self, now: float, deadline: float | None,
                  reserve: float | None) -> tuple[float | None, str | None]:
        task_deadline = self.task_deadline if deadline is None else deadline
        reserve_seconds = (self.finish_reserve_seconds if reserve is None
                           else max(0.0, float(reserve)))
        if task_deadline is None:
            return now + self.wall_seconds, None
        cutoff = float(task_deadline) - reserve_seconds
        if cutoff <= now:
            return None, "deadline_unavailable"
        return min(cutoff, now + self.wall_seconds), None

    def run(self, code: str, question: str | None = None, *,
            revision: str | None = None, deadline: float | None = None,
            finish_reserve_seconds: float | None = None,
            feature_enabled: bool | None = None) -> ProbeReceipt:
        """Execute one proposal and return a host-created typed receipt."""
        if (observation_probe_mode() == "off") if feature_enabled is None else not feature_enabled:
            definition = self._definition(str(code or ""), question,
                                          _UNKNOWN_REVISION)
            return self._receipt(definition, ProbeStatus.NOT_RUN, reason="feature_off")
        code = code if isinstance(code, str) else ""
        self.budget.note_attempt()
        try:
            code_bytes = len(code.encode("utf-8", errors="surrogatepass"))
        except (UnicodeError, AttributeError):
            code_bytes = MAX_CODE_BYTES + 1
        if not code.strip():
            definition = self._definition("", question, _UNKNOWN_REVISION)
            return self._receipt(definition, ProbeStatus.UNRUNNABLE,
                                 reason="code_required")
        if code_bytes > MAX_CODE_BYTES:
            definition = self._definition(code, question, _UNKNOWN_REVISION)
            return self._receipt(definition, ProbeStatus.UNRUNNABLE,
                                 reason="code_limit")
        if _VACUOUS_PROBE.search(code) or _CLONE_PROBE.search(code):
            definition = self._definition(code, question, _UNKNOWN_REVISION)
            return self._receipt(definition, ProbeStatus.UNRUNNABLE,
                                 reason="vacuous_or_source_clone")
        from kernel_tokenlab_v13 import (
            EVENT_PROBE_PYTEST_CHECK,
            probe_invokes_pytest,
            probe_pytest_guard_enabled,
            probe_render_enabled,
        )
        if probe_render_enabled() and probe_pytest_guard_enabled():
            _runner_invocation = probe_invokes_pytest(code)
            if hasattr(getattr(self, "journal", None), "emit"):
                self.journal.emit(EVENT_PROBE_PYTEST_CHECK, producer="kernel_tokenlab_v13", dispatch=False,
                                  invocation_detected=_runner_invocation,
                                  policy="use_run_tests")
            if _runner_invocation:
                # : pytest inside a probe is refused -- run_tests is the check tool.
                definition = self._definition(code, question, _UNKNOWN_REVISION)
                if hasattr(getattr(self, "journal", None), "emit"):
                    self.journal.emit("probe_pytest_refused", producer="kernel_tokenlab_v13", dispatch=False,
                                      policy="use_run_tests")
                return self._receipt(definition, ProbeStatus.POLICY_DENIED, reason="use_run_tests")
        now = time.monotonic()
        end, unavailable = self._deadline(now, deadline, finish_reserve_seconds)
        if unavailable is not None:
            definition = self._definition(code, question, _UNKNOWN_REVISION)
            return self._receipt(definition, ProbeStatus.NOT_RUN, reason=unavailable)
        try:
            before = workspace_revision(self.root)
            if revision is not None and str(revision) != before:
                definition = self._definition(code, question, before)
                return self._receipt(definition, ProbeStatus.STALE,
                                     reason="revision_changed_before_snapshot")
        except Exception as error:  # noqa: BLE001 - typed runner failure
            definition = self._definition(code, question, _UNKNOWN_REVISION)
            return self._receipt(definition, ProbeStatus.RUNNER_ERROR,
                                 reason=f"workspace_snapshot:{type(error).__name__}")
        definition = self._definition(code, question, before)
        snapshot_id = before
        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="stateframe-probe-") as temporary:
                snapshot = pathlib.Path(temporary)
                snapshot_revision, stable = _copy_snapshot(self.root, snapshot)
                if not stable or snapshot_revision != before:
                    return self._receipt(definition, ProbeStatus.STALE,
                                         reason="revision_changed_during_snapshot",
                                         snapshot_id=snapshot_id,
                                         wall_seconds=time.monotonic() - started)
                remaining = min(self.wall_seconds, float(end) - time.monotonic())
                if remaining <= 0:
                    return self._receipt(definition, ProbeStatus.NOT_RUN,
                                         reason="deadline_unavailable", snapshot_id=snapshot_id,
                                         wall_seconds=time.monotonic() - started)
                remaining = max(0.001, remaining)
                # Charge execution only after validation, revision capture, and
                # deadline admission. Invalid/host-blocked attempts consume
                # only ``attempts`` and cannot exhaust execution credits.
                if not self.budget.reserve():
                    return self._receipt(definition, ProbeStatus.NOT_RUN,
                                         reason="task_probe_limit", snapshot_id=snapshot_id,
                                         wall_seconds=time.monotonic() - started)
                child = _run_child(
                    snapshot, code, executable=self.executable,
                    wall_seconds=remaining, cpu_seconds=self.cpu_seconds,
                    memory_bytes=self.memory_bytes)
                raw_bytes = child.stdout + child.stderr
                raw = raw_bytes.decode("utf-8", errors="replace")
                shown, shown_tokens, shown_truncated = _bound_shown(raw)
                try:
                    after = workspace_revision(self.root)
                except Exception:
                    after = _UNKNOWN_REVISION
                if after != before:
                    status = ProbeStatus.STALE
                    reason = "revision_changed_after_probe"
                elif child.timed_out:
                    status = ProbeStatus.TIMED_OUT
                    reason = "wall_limit"
                elif (b"sandbox_apply" in raw_bytes or b"bwrap" in raw_bytes
                      and b"Operation not permitted" in raw_bytes):
                    status = ProbeStatus.RUNNER_ERROR
                    reason = "sandbox_backend_unavailable"
                else:
                    denied_policy_events = tuple(
                        event for event in child.policy_events
                        if event != "PYCACHE_READ_ALLOWED")
                    if denied_policy_events:
                        status = ProbeStatus.POLICY_DENIED
                        reason = ",".join(denied_policy_events)
                    elif child.returncode == 0:
                        status = ProbeStatus.COMPLETED
                        reason = ""
                    elif child.returncode is not None and (
                            child.returncode < 0 and -child.returncode in {
                                signal.SIGXCPU, signal.SIGKILL, signal.SIGSEGV,
                            } or child.returncode in {137, 152}):
                        status = ProbeStatus.RESOURCE_LIMIT
                        reason = "resource_limit"
                    elif child.returncode is None:
                        status = ProbeStatus.RUNNER_ERROR
                        reason = "missing_return_code"
                    else:
                        status = ProbeStatus.RAISED
                        reason = "nonzero_exit"
                return self._receipt(
                    definition, status, snapshot_id=snapshot_id,
                    captured_output=raw,
                    shown_output=shown,
                    shown_tokens=shown_tokens,
                    capture_truncated=child.capture_truncated,
                    shown_truncated=shown_truncated,
                    exit_code=child.returncode,
                    reason=reason,
                    wall_seconds=time.monotonic() - started,
                    policy_events=child.policy_events,
                    child_group_killed=child.child_group_killed,
                )
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
            return self._receipt(
                definition, ProbeStatus.RUNNER_ERROR,
                reason=f"runner:{type(error).__name__}", snapshot_id=snapshot_id,
                wall_seconds=time.monotonic() - started)


__all__ = [
    "MAX_CAPTURE_BYTES", "MAX_CODE_BYTES", "MAX_CPU_SECONDS", "MAX_MEMORY_BYTES",
    "MAX_PROBES_PER_TASK", "MAX_SHOWN_TOKENS", "MAX_WALL_SECONDS", "ProbeBudget",
    "ProbeDefinition", "ProbeReceipt", "ProbeRunner", "ProbeStatus", "enabled",
    "observation_probe_mode", "render_probe_receipt", "workspace_revision",
]
