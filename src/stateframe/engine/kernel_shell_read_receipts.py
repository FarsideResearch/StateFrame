"""Audit opaque shell replies; capture only provably exact, single-file reads."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shlex
import stat

from kernel_read_receipts import (begin_nested_read_attempt, capture_failed, capture_read,
                                  capture_skipped, enabled, end_nested_read_attempt, relative_path)

MAX_SNAPSHOT_BYTES = 16 << 20


@dataclass(frozen=True)
class ShellReadSnapshot:
    path: Path
    raw: bytes
    payload: bytes
    first_line: int
    last_line: int
    byte_start: int
    byte_end: int


def _bounded_bytes(path: Path) -> bytes | None:
    # Avoid blocking on special files and bound allocation even if a file grows.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_SNAPSHOT_BYTES:
            return None
        raw = stream.read(MAX_SNAPSHOT_BYTES + 1)
    return raw if len(raw) <= MAX_SNAPSHOT_BYTES else None


def _snapshot(tool: object, command: object) -> ShellReadSnapshot | None:
    if not isinstance(command, str):
        return None
    tokens = shlex.split(command)
    # Expansion, redirection, pipelines and compound commands get opaque audits.
    if not tokens or any(any(c in token for c in "$`*?[]~;|&<>()\n") for token in tokens):
        return None
    name, args = tokens[0], tokens[1:]
    if name not in {"cat", "sed", "head", "tail", "/bin/cat", "/usr/bin/cat",
                    "/bin/sed", "/usr/bin/sed", "/bin/head", "/usr/bin/head",
                    "/bin/tail", "/usr/bin/tail"}:
        return None
    name = Path(name).name
    first, last, count = 1, None, 10
    from_start = False
    if name == "sed":
        if len(args) != 3 or args[0] != "-n":
            return None
        match = re.fullmatch(r"([1-9][0-9]*)(?:,([1-9][0-9]*|\$))?p", args[1])
        if match is None:
            return None
        first = int(match[1])
        last = None if match[2] == "$" else int(match[2] or match[1])
        if last is not None and last < first:
            return None
        args = args[2:]
    elif name in {"head", "tail"}:
        if args and args[0] == "-n" and len(args) >= 2:
            number, args = args[1], args[2:]
        elif args and args[0].startswith("-n"):
            number, args = args[0][2:], args[1:]
        elif args and re.fullmatch(r"-[0-9]+", args[0]):
            number, args = args[0][1:], args[1:]
        elif args and args[0].startswith("--lines="):
            number, args = args[0].split("=", 1)[1], args[1:]
        else:
            number = "10"
        if not re.fullmatch(r"\+?[0-9]+", number) or (number.startswith("+") and name != "tail"):
            return None
        count, from_start = int(number), number.startswith("+")
    if args and args[0] == "--":
        args = args[1:]
    if len(args) != 1 or args[0] == "-" or args[0].startswith("-"):
        return None
    cwd = tool.shell_state.directory(tool.root) if tool.shell_state is not None else tool.root
    path = (cwd / args[0]).resolve()
    if relative_path(tool.root, path) is None:
        return None
    try:
        raw = _bounded_bytes(path)
    except OSError:  # Missing or unreadable: an ordinary opaque shell read, not a capture failure.
        return None
    if raw is None:
        return None
    line_count = raw.count(b"\n") + int(bool(raw) and not raw.endswith(b"\n"))
    if name == "head":
        last = count
    elif name == "tail":
        first = max(1, count) if from_start else max(1, line_count - count + 1)
    last = min(last if last is not None else line_count, line_count)
    if first > last:  # No source bytes were delivered.
        return None
    start, end = 0, len(raw)
    # Keep memory bounded for files containing millions of very short lines.
    for line, match in enumerate(re.finditer(b"\n", raw), 1):
        if line == first - 1:
            start = match.end()
        if line == last:
            end = match.end()
            break
    return ShellReadSnapshot(path, raw, raw[start:end], first, last, start, end)


def execute_with_read_audit(tool: object, args: dict, call_id: str):
    """Called only by the enabled shell boundary; never change its reply."""
    from kernel_pilot import ACTIVE_RUNTIME
    runtime = ACTIVE_RUNTIME.get() if enabled() else None
    if runtime is None:
        return tool._execute(args, call_id)
    # Nested: never close or erase a read attempt that encloses this shell call.
    scope, prior = begin_nested_read_attempt(runtime, args, call_id=call_id)
    try:
        try:
            snapshot = _snapshot(tool, args.get("command"))
        except Exception as error:
            capture_failed(runtime, error)
            snapshot = None
        reply = tool._execute(args, call_id)
        if snapshot is not None and reply.exit_code == 0 and not reply.fatal:
            try:
                from kernel_execution_reply_budget import _parts
                _, output, _, omitted = _parts(reply.observation, "shell", tool.run_dir, call_id)
                if _bounded_bytes(snapshot.path) != snapshot.raw:
                    capture_skipped(runtime, reason="stale")
                elif output != snapshot.payload or omitted:
                    capture_skipped(runtime, reason="transformed_or_truncated")
                else:
                    payload = output.decode("utf-8")
                    capture_read(runtime, path=relative_path(tool.root, snapshot.path),
                        metadata={"result_class": "OK",
                                  "revision": "sha256:" + hashlib.sha256(snapshot.raw).hexdigest(),
                                  "effective_range": {"start_line": snapshot.first_line,
                                                      "end_line": snapshot.last_line},
                                  "byte_start": snapshot.byte_start, "byte_end": snapshot.byte_end},
                        payload=payload, original_page=reply.observation, source="disk", arguments=args)
            except Exception as error:
                capture_failed(runtime, error)
        status = "OK" if scope.capture is not None else "SHELL_READ"
        if reply.observation.startswith("REFUSED "):
            status = "DEADLINE" if reply.observation.startswith("REFUSED TIMEOUT:") else "REFUSED"
        scope.finish(status)
        return reply
    except BaseException as error:
        scope.finish(error=error)
        raise
    finally:
        end_nested_read_attempt(runtime, scope, prior)
