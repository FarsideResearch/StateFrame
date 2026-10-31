"""Small real stdio peer for MCP client tests."""
import json
import os
import socket
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
if mode == "hang" and len(sys.argv) > 2:
    with open(sys.argv[2], "w", encoding="utf-8") as marker:
        marker.write(str(os.getpid()))


def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def response(ident, result):
    send({"jsonrpc": "2.0", "id": ident, "result": result})


while line := sys.stdin.readline():
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        if mode == "early_exit":
            sys.exit(0)
        if mode == "hang":
            time.sleep(3600)
        response(request["id"], {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                 "serverInfo": {"name": "fixture", "version": "1"}})
    elif method == "notifications/initialized":
        if mode == "server_request":
            send({"jsonrpc": "2.0", "id": 70, "method": "roots/list"})
    elif method == "tools/list":
        if mode == "malformed":
            sys.stdout.write("{oops\n")
            sys.stdout.flush()
            continue
        if mode == "wrong_id":
            response(999, {"tools": []})
            continue
        base = {"name": "echo", "description": "Echo a value", "inputSchema": {"type": "object",
                "properties": {"value": {"type": "string"}}, "required": ["value"]}}
        if mode == "switch_list":
            variant = sys.argv[2] if len(sys.argv) > 2 else "variant.txt"
            if os.path.isfile(variant):
                with open(variant, encoding="utf-8") as source:
                    if source.read().strip() == "changed":
                        base["description"] = "Changed after initial discovery"
        if mode == "pagination" and not request.get("params", {}).get("cursor"):
            response(request["id"], {"tools": [base], "nextCursor": "next"})
        else:
            tools = ([{"name": "add", "inputSchema": {"type": "object", "properties": {
                "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}]
                if mode == "pagination" else [base])
            if mode == "many":
                tools = [{"name": f"tool_{i}", "description": "description " * 30,
                    "inputSchema": {"type": "object", "properties": {"value": {
                        "type": "string", "description": "value " * 60}}, "required": ["value"]}}
                    for i in range(40)]
            response(request["id"], {"tools": tools})
    elif method == "tools/call":
        if mode == "huge":
            response(request["id"], {"content": [{"type": "text", "text": "x" * (5 << 20)}]})
        elif mode == "is_error":
            response(request["id"], {"content": [{"type": "text", "text": "tool rejected"}], "isError": True})
        elif mode == "probe_write":
            try:
                with open("mcp-forbidden.txt", "w", encoding="utf-8") as target:
                    target.write("unexpected")
                text = "wrote"
            except OSError as error:
                text = type(error).__name__
            response(request["id"], {"content": [{"type": "text", "text": text}]})
        elif mode == "probe_network":
            try:
                with socket.create_connection(("127.0.0.1", int(sys.argv[2])), timeout=.5):
                    text = "connected"
            except OSError as error:
                text = type(error).__name__
            response(request["id"], {"content": [{"type": "text", "text": text}]})
        elif mode == "probe_outside":
            try:
                with open(sys.argv[2], "w", encoding="utf-8") as target:
                    target.write("unexpected")
                text = "wrote"
            except OSError as error:
                text = type(error).__name__
            response(request["id"], {"content": [{"type": "text", "text": text}]})
        else:
            params = request["params"]
            value = params["arguments"].get("value", "")
            response(request["id"], {"content": [{"type": "text", "text": value}]})
