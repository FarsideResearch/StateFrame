"""Trusted protocol client for one sandboxed, bounded stdio conversation."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
import platform
import select
import selectors
import subprocess
import sys
import time

if __package__:
    from .kernel_exec_session import CleanupReport, ExecResult, ExecSessionError
    from .kernel_exec_tracker import ProcessTable
    from .kernel_test_isolation import capability_command
else:
    from kernel_exec_session import CleanupReport, ExecResult, ExecSessionError
    from kernel_exec_tracker import ProcessTable
    from kernel_test_isolation import capability_command

MAX_MESSAGE = 4 << 20
MAX_STDOUT = 8 << 20
VERSIONS = {"2025-06-18", "2025-03-26", "2024-11-05"}


class ProtocolError(ValueError):
    pass


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value}")


def _json_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key}")
        value[key] = item
    return value


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def converse(request: dict, cap, cwd: Path) -> object:
    argv = request.get("argv")
    timeout = request.get("timeout_seconds")
    if (not isinstance(argv, list) or not argv or any(not isinstance(v, str) or not v for v in argv)
            or type(timeout) not in (int, float) or not 0 < timeout <= 600
            or not isinstance(request.get("client_version"), str) or not request["client_version"]
            or request.get("operation") not in {"list", "call"}):
        raise ProtocolError("invalid driver request")
    deadline = time.monotonic() + timeout
    if platform.system() not in {"Darwin", "Linux"}:
        raise ProtocolError("MCP containment is unsupported on this platform")
    if platform.system() == "Linux" and not cap.sandboxed:
        raise ProtocolError("MCP containment requires a sandboxed Linux PID namespace")
    launcher = Path(__file__).with_name("sf_launch.py").resolve()
    spec = {"argv": argv, "cwd": str(cwd), "env": dict(cap.env),
            "limits": asdict(cap.limits), "memory_supervised": platform.system() == "Darwin",
            "new_session": False}
    command = [sys.executable, "-I", str(launcher), json.dumps(spec, separators=(",", ":"))]
    command, environment = capability_command(command, cap,
        single_process=platform.system() == "Darwin",
        private_pid_unit=platform.system() == "Linux")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, env=environment, cwd=cwd,
                               close_fds=True, start_new_session=True)
    assert process.stdin is not None and process.stdout is not None
    selector = None
    buffer = bytearray()
    total = 0
    counter = 0
    table = None
    server_identity = None

    def send(value: dict) -> None:
        data = _json(value)
        if len(data) > MAX_MESSAGE:
            raise ProtocolError("client message exceeds 4 MiB")
        try:
            fd = process.stdin.fileno()
            os.set_blocking(fd, False)
            offset = 0
            while offset < len(data):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                    raise TimeoutError("MCP server deadline exceeded")
                try:
                    offset += os.write(fd, data[offset:])
                except BlockingIOError:
                    continue
        except TimeoutError:
            raise
        except (BrokenPipeError, OSError) as error:
            raise ProtocolError("server exited before accepting a request") from error

    def receive(expected: int) -> dict:
        nonlocal total
        while True:
            if b"\n" in buffer:
                line, _, rest = buffer.partition(b"\n")
                buffer[:] = rest
                if len(line) > MAX_MESSAGE:
                    raise ProtocolError("server message exceeds 4 MiB")
                try:
                    value = json.loads(line, parse_constant=_reject_constant,
                                       object_pairs_hook=_json_object)
                except (UnicodeError, ValueError) as error:
                    raise ProtocolError("server sent invalid JSON") from error
                if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
                    raise ProtocolError("server sent invalid JSON-RPC message")
                if "method" in value:
                    if not isinstance(value["method"], str):
                        raise ProtocolError("server request method must be text")
                    if "id" in value:
                        response = {"jsonrpc": "2.0", "id": value["id"]}
                        if value["method"] == "roots/list":
                            response["result"] = {"roots": []}
                        else:
                            response["error"] = {"code": -32601, "message": "Method not found"}
                        send(response)
                    continue
                if value.get("id") != expected or type(value.get("id")) is not int:
                    raise ProtocolError("server response id does not match request")
                if ("result" in value) == ("error" in value):
                    raise ProtocolError("server response must contain exactly one result or error")
                if "error" in value:
                    error = value["error"]
                    if (not isinstance(error, dict) or type(error.get("code")) is not int
                            or not isinstance(error.get("message"), str)):
                        raise ProtocolError("server sent malformed JSON-RPC error")
                    raise ProtocolError(f"server JSON-RPC error: {error.get('message', 'unknown') if isinstance(error, dict) else 'unknown'}")
                return value["result"]
            if len(buffer) > MAX_MESSAGE:
                raise ProtocolError("server message exceeds 4 MiB")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("MCP server deadline exceeded")
            if table is not None and server_identity is not None:
                try:
                    memory = table.resident_bytes(server_identity)
                except (OSError, RuntimeError) as error:
                    raise ExecSessionError("cannot supervise MCP server resident memory") from error
                if memory > cap.limits.memory_bytes:
                    raise ProtocolError("server resident memory exceeds capability limit")
            if not selector.select(min(remaining, .25)):
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                raise ProtocolError("server exited before responding")
            total += len(chunk)
            if total > MAX_STDOUT:
                raise ProtocolError("server stdout exceeds 8 MiB")
            buffer.extend(chunk)

    def call(method: str, params: dict) -> object:
        nonlocal counter
        counter += 1
        send({"jsonrpc": "2.0", "id": counter, "method": method, "params": params})
        return receive(counter)

    try:
        if platform.system() == "Darwin":
            try:
                table = ProcessTable()
                server_identity = table.identity(process.pid)
            except (OSError, RuntimeError) as error:
                raise ExecSessionError("cannot establish MCP server process identity") from error
            if server_identity is None and process.poll() is None:
                raise ExecSessionError("cannot establish MCP server process identity")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        initialized = call("initialize", {"protocolVersion": "2025-06-18",
            "capabilities": {"roots": {"listChanged": False}},
            "clientInfo": {"name": "StateFrame", "version": request["client_version"]}})
        if (not isinstance(initialized, dict) or initialized.get("protocolVersion") not in VERSIONS
                or not isinstance(initialized.get("capabilities"), dict)):
            raise ProtocolError("server negotiated an unsupported protocol version")
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        if request["operation"] == "call":
            name = request.get("tool")
            arguments = request.get("arguments")
            if not isinstance(name, str) or not name or not isinstance(arguments, dict):
                raise ProtocolError("invalid tools/call request")
            result = call("tools/call", {"name": name, "arguments": arguments})
            if (not isinstance(result, dict) or not isinstance(result.get("content"), list)
                    or type(result.get("isError", False)) is not bool):
                raise ProtocolError("invalid tools/call response")
            for item in result["content"]:
                if not isinstance(item, dict) or not isinstance(item.get("type"), str):
                    raise ProtocolError("invalid tools/call content item")
                if item["type"] == "text" and not isinstance(item.get("text"), str):
                    raise ProtocolError("invalid text content item")
                if item["type"] == "resource":
                    resource = item.get("resource")
                    uri = resource.get("uri") if isinstance(resource, dict) else item.get("uri")
                    if not isinstance(uri, str):
                        raise ProtocolError("invalid resource content item")
            return result
        listed = []
        cursor = None
        cursors = set()
        for _ in range(32):
            params = {"cursor": cursor} if cursor is not None else {}
            page = call("tools/list", params)
            if not isinstance(page, dict) or not isinstance(page.get("tools"), list):
                raise ProtocolError("invalid tools/list response")
            listed.extend(page["tools"])
            if len(listed) > 512:
                raise ProtocolError("tools/list exceeds 512 tools")
            cursor = page.get("nextCursor")
            if cursor is None:
                return listed
            if not isinstance(cursor, str) or not cursor or cursor in cursors:
                raise ProtocolError("invalid or repeated tools/list cursor")
            cursors.add(cursor)
        raise ProtocolError("tools/list exceeds 32 pages")
    finally:
        if selector is not None:
            selector.close()
        try:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            except PermissionError as error:
                raise ExecSessionError("MCP server containment unit cannot be killed") from error
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired as error:
                raise ExecSessionError("MCP server cleanup deadline exceeded") from error
            if process.poll() is None:
                raise ExecSessionError("MCP server survived cleanup")
            if table is not None and server_identity is not None:
                try:
                    survivor = table.identity(server_identity.pid)
                except (OSError, RuntimeError) as error:
                    raise ExecSessionError("cannot verify MCP server termination") from error
                if survivor is not None and survivor.start == server_identity.start:
                    raise ExecSessionError("MCP server survived cleanup")
        finally:
            process.stdin.close()
            process.stdout.close()


def exchange(request: dict, cap, cwd: Path) -> ExecResult:
    """Return a bounded envelope after the server containment unit is dead."""
    started = time.monotonic()
    try:
        value = converse(request, cap, cwd)
        envelope = {"ok": True, "result": value}
        encoded = _json(envelope)
        if len(encoded) > MAX_MESSAGE:
            raise ProtocolError("driver result exceeds 4 MiB")
    except ExecSessionError:
        raise
    except (ProtocolError, TimeoutError, OSError, ValueError, TypeError, RuntimeError) as error:
        encoded = _json({"ok": False, "code": "TIMEOUT" if isinstance(error, TimeoutError)
                         else "MCP_PROTOCOL_ERROR", "error": f"{type(error).__name__}: {error}"[:500]})
    elapsed = time.monotonic() - started
    return ExecResult(0, False, encoded, False, len(encoded), elapsed,
                      CleanupReport(True, 0, (), ""))
