"""One bounded MCP stdio conversation. Invoked only inside an ExecCapability."""
from __future__ import annotations

import json
import os
import select
import selectors
import signal
import subprocess
import sys
import time

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


def converse(request: dict) -> object:
    argv = request.get("argv")
    timeout = request.get("timeout_seconds")
    if (not isinstance(argv, list) or not argv or any(not isinstance(v, str) or not v for v in argv)
            or type(timeout) not in (int, float) or not 0 < timeout <= 600
            or not isinstance(request.get("client_version"), str) or not request["client_version"]
            or request.get("operation") not in {"list", "call"}):
        raise ProtocolError("invalid driver request")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    assert process.stdin is not None and process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    buffer = bytearray()
    total = 0
    counter = 0
    descendants: dict[int, str] = {}
    next_scan = 0.0

    def process_table() -> dict[int, tuple[int, str]]:
        try:
            listing = subprocess.run(["ps", "-eo", "pid=,ppid=,lstart="], capture_output=True,
                                     text=True, timeout=1, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return {}
        table = {}
        for line in listing.splitlines():
            fields = line.split()
            if len(fields) < 7 or not all(field.isdecimal() for field in fields[:2]):
                continue
            table[int(fields[0])] = (int(fields[1]), " ".join(fields[2:]))
        return table

    root_start = process_table().get(process.pid, (0, ""))[1]

    def record_descendants() -> None:
        # A peer can create a new session before it fails or times out. Retain
        # observed PIDs and start times while the parent is alive.
        table = process_table()
        children: dict[int, list[int]] = {}
        for pid, (parent, _) in table.items():
            children.setdefault(parent, []).append(pid)
        pending = ([process.pid] if root_start and table.get(process.pid, (0, None))[1] == root_start
                   else [])
        pending.extend(pid for pid, started in descendants.items()
                       if table.get(pid, (0, None))[1] == started)
        while pending:
            for pid in children.get(pending.pop(), ()):
                if pid not in descendants:
                    descendants[pid] = table[pid][1]
                    pending.append(pid)

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
        nonlocal total, next_scan
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
            # Scanning spawns ps; bound it to a few times per second, not once per chunk.
            if time.monotonic() >= next_scan:
                record_descendants()
                next_scan = time.monotonic() + .25
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
        selector.close()
        record_descendants()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if process.poll() is None:
                process.kill()
        current = process_table()
        for pid, started in descendants.items():
            if pid not in current or current[pid][1] != started:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)


def main() -> int:
    try:
        raw = sys.stdin.buffer.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise ProtocolError("driver request exceeds 1 MiB")
        request = json.loads(raw, parse_constant=_reject_constant,
                             object_pairs_hook=_json_object)
        if not isinstance(request, dict):
            raise ProtocolError("driver request must be an object")
        value = converse(request)
        envelope = {"ok": True, "result": value}
        encoded = _json(envelope)
        if len(encoded) > MAX_MESSAGE:
            raise ProtocolError("driver result exceeds 4 MiB")
    except (ProtocolError, TimeoutError, OSError, ValueError, TypeError, subprocess.TimeoutExpired) as error:
        encoded = _json({"ok": False, "code": "TIMEOUT" if isinstance(error, TimeoutError)
                         else "MCP_PROTOCOL_ERROR", "error": f"{type(error).__name__}: {error}"[:500]})
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
