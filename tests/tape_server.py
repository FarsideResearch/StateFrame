"""Loopback-only OpenAI-compatible server for recorded allocation responses."""
from __future__ import annotations

import gzip
import json
import re
import threading
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_PIECE = re.compile(r" ?[A-Za-z]{1,7}|\d|[^\sA-Za-z\d]{1,2}|\s+")
_MAX_BODY_BYTES = 32 * 1024 * 1024


class TapeServer:
    """Serve a fixed sequence of assistant messages and calibrated token counts."""

    def __init__(self, script: Mapping[str, object] | str | Path) -> None:
        if isinstance(script, (str, Path)):
            try:
                with gzip.open(script, "rt", encoding="utf-8") as source:
                    loaded_script = json.load(source)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("replay script must be readable gzip JSON") from error
            if not isinstance(loaded_script, Mapping):
                raise ValueError("replay script must be a JSON object")
            script = loaded_script
        if not isinstance(script, Mapping):
            raise ValueError("replay script must be a JSON object")
        turns = script.get("turns")
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)) or not turns:
            raise ValueError("replay script must contain at least one turn")
        self._turns = tuple(turns)
        self._model = str(script.get("model") or "allocation-replay-model")
        self._divisor = float(script.get("tokenize_divisor", 1.245))
        if self._divisor <= 0:
            raise ValueError("tokenize_divisor must be positive")
        slots = script.get("slots", [65536] * 8)
        self._slots = tuple(slots) if isinstance(slots, Sequence) and not isinstance(slots, (str, bytes)) else (65536,)
        self._lock = threading.Lock()
        self._turn_index = 0
        self._request_count = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    def _token_count(self, text: str) -> int:
        return max(1, round(len(_PIECE.findall(text)) / self._divisor))

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        tape = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: object) -> None:
                return

            def _send_json(self, status: int, payload: object) -> None:
                data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                path = self.path.rstrip("/")
                if path.endswith("/models"):
                    self._send_json(200, {"object": "list", "data": [{"id": tape._model, "object": "model"}]})
                elif path.endswith("/slots"):
                    self._send_json(200, [
                        {"id": index, "n_ctx": size, "is_processing": False}
                        for index, size in enumerate(tape._slots)
                    ])
                else:
                    self._send_json(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    self._send_json(400, {"error": {"message": "invalid content length"}})
                    return
                if length < 0 or length > _MAX_BODY_BYTES:
                    self._send_json(413, {"error": {"message": "request body too large"}})
                    return
                raw = self.rfile.read(length)
                try:
                    if self.headers.get("Content-Encoding") == "gzip":
                        raw = gzip.decompress(raw)
                    body = json.loads(raw)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    self._send_json(400, {"error": {"message": "invalid JSON request"}})
                    return

                path = self.path.rstrip("/")
                if path.endswith("/tokenize"):
                    content = body.get("content") if isinstance(body, Mapping) else ""
                    self._send_json(
                        200, {"tokens": list(range(tape._token_count(str(content or ""))))},
                    )
                    return
                if not path.endswith("/chat/completions"):
                    self._send_json(404, {"error": {"message": "not found"}})
                    return

                messages = body.get("messages") if isinstance(body, Mapping) else []
                messages = messages if isinstance(messages, list) else []
                with tape._lock:
                    index = tape._turn_index
                    tape._turn_index += 1
                    tape._request_count += 1
                turn = tape._turns[min(index, len(tape._turns) - 1)]
                message = turn.get("message") if isinstance(turn, Mapping) else None
                if not isinstance(message, Mapping):
                    self._send_json(500, {"error": {"message": "invalid replay turn"}})
                    return
                prompt = tape._token_count(json.dumps(messages, ensure_ascii=False))
                self._send_json(200, {
                    "id": f"chatcmpl-{index + 1}",
                    "object": "chat.completion",
                    "model": tape._model,
                    "choices": [{"index": 0, "finish_reason": "tool_calls", "message": dict(message)}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": 200, "total_tokens": prompt + 200},
                })

        return Handler

    def start(self) -> "TapeServer":
        if self._thread is not None:
            raise RuntimeError("tape server is already running")
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("tape server did not stop")
        self._thread = None

    def __enter__(self) -> "TapeServer":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.close()
