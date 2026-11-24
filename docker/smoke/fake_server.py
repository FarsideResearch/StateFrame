#!/usr/bin/env python3
"""Stdlib-only compatible endpoint for the container smoke checks."""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODEL = "sf-eks-scripted"
STEPS = (
    ("read", {"path": "calc.py"}),
    ("str_replace", {"path": "calc.py", "old_str": "return 41", "new_str": "return 42"}),
    ("run_tests", {"target": "tests/test_calc.py"}),
    ("done", {"summary": "Changed value() to return the expected answer and ran its test."}),
)


class State:
    def __init__(self, directory: Path, pause_at: int | None) -> None:
        self.directory = directory
        self.pause_at = pause_at
        self.lock = threading.Lock()
        self.position = 0
        self.requests = 0
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, value: str) -> None:
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)

    def next_step(self, offered: set[str], forced: str | None) -> tuple[str, dict[str, str]]:
        with self.lock:
            if self.position >= len(STEPS):
                step = ("done", {"summary": "The scripted edit is complete."})
            else:
                step = STEPS[self.position]
            if step[0] not in offered or (forced is not None and forced != step[0]):
                raise ValueError(f"script expects {step[0]} but offered tools were {sorted(offered)}")
            return step

    def delivered(self, tool: str) -> None:
        with self.lock:
            if self.position < len(STEPS) and STEPS[self.position][0] == tool:
                self.position += 1
            self.write("script-position", str(self.position))

    def count_request(self) -> int:
        with self.lock:
            self.requests += 1
            count = self.requests
            self.write("chat-count", str(count))
            return count


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def handler_for(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format: str, *_args: object) -> None:
            return

        def send_json(self, status: int, value: object) -> None:
            raw = _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
            elif path == "/props":
                self.send_json(200, {"default_generation_settings": {"n_ctx": 65536},
                                     "model_path": "stdlib-scripted-endpoint"})
            elif path == "/slots":
                self.send_json(200, [{"id": 0, "n_ctx": 65536, "is_processing": False}])
            elif path == "/health":
                self.send_json(200, {"ok": True})
            else:
                self.send_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            if self.headers.get("Content-Encoding", "").lower() == "gzip":
                raw = gzip.decompress(raw)
            try:
                body = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                self.send_json(400, {"error": {"message": "invalid JSON"}})
                return
            if path == "/tokenize":
                text = body.get("content", "") if isinstance(body, dict) else ""
                self.send_json(200, {"tokens": list(range(max(1, len(str(text)) // 4)))})
                return
            if path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "not found"}})
                return
            request_number = state.count_request()
            if state.pause_at == request_number and not (state.directory / "paused").exists():
                state.write("paused", str(request_number))
                while not ((state.directory / "release").exists()
                           or (state.directory / "abandon").exists()):
                    time.sleep(0.05)
                if (state.directory / "abandon").exists():
                    return
            tools = body.get("tools") or []
            offered = {
                item.get("function", {}).get("name")
                for item in tools if isinstance(item, dict)
                and isinstance(item.get("function"), dict)
            }
            choice = body.get("tool_choice")
            forced = None
            if isinstance(choice, dict) and isinstance(choice.get("function"), dict):
                forced = choice["function"].get("name")
            try:
                tool, arguments = state.next_step(offered, forced)
            except ValueError as error:
                self.send_json(409, {"error": {"message": str(error)}})
                return
            call_id = f"call_{request_number}"
            call = {"id": call_id, "type": "function", "function": {
                "name": tool, "arguments": json.dumps(arguments, separators=(",", ":"))}}
            response = {
                "id": f"chatcmpl-{request_number}", "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": "", "tool_calls": [call]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            try:
                if body.get("stream") is True:
                    self.close_connection = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    chunks = (
                        {"id": response["id"], "object": "chat.completion.chunk", "model": MODEL,
                         "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                             {"index": 0, "id": call_id, "type": "function",
                              "function": {"name": tool, "arguments": ""}}]}, "finish_reason": None}]},
                        {"id": response["id"], "object": "chat.completion.chunk", "model": MODEL,
                         "choices": [{"index": 0, "delta": {"tool_calls": [
                             {"index": 0, "function": {"arguments": call["function"]["arguments"]}}]},
                                      "finish_reason": None}]},
                        {"id": response["id"], "object": "chat.completion.chunk", "model": MODEL,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
                    )
                    for chunk in chunks:
                        self.wfile.write(b"data: " + _json_bytes(chunk) + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self.send_json(200, response)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            state.delivered(tool)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--state-dir", type=Path, default=Path("/tmp/sf-eks-server-state"))
    parser.add_argument("--pause-at", type=int)
    args = parser.parse_args()
    state = State(args.state_dir, args.pause_at)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(state))
    print(f"http://{args.host}:{server.server_address[1]}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
