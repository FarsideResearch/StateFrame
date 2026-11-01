"""A scripted OpenAI-compatible chat endpoint, for tests and offline demos.

It answers ``POST /v1/chat/completions`` with tool calls taken, in order, from a script
file, and ``GET /v1/models`` with one model. Every other route answers 404, like a plain
OpenAI-compatible server (no llama.cpp extras). It is not a model: it proves the CLI,
the engine wiring and the request recording without a GPU.

Script file (JSON):
    {"model": "scripted", "steps": [
        {"tool": "read", "arguments": {"path": "calc/ops.py"}},
        {"tool": "str_replace", "arguments": {"path": "...", "old_str": "...", "new_str": "..."}},
        {"tool": "run_tests", "arguments": {}},
        {"tool": "done", "arguments": {"summary": "..."}}
    ]}
Steps are required by default. A step whose tool is not offered is skipped so later compatible
An explicitly optional step whose tool is not offered is skipped so later steps can run. A
required step is refused immediately, and ``assert_consumed()`` names any required step that was
skipped or left pending. Mark an intentionally optional step with ``"required": false``. When the
steps run out it keeps answering ``done``; qualification callers must assert consumption after
the run.

Run it:  python -m stateframe.scripted_endpoint --script s.json --port 0  (prints the URL)
"""
from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ScriptStepUnavailable(RuntimeError):
    """A required scripted action is missing from the offered tool menu."""


class Script:
    def __init__(self, document: dict) -> None:
        self.model = str(document.get("model") or "scripted")
        steps = document.get("steps") or []
        if not isinstance(steps, list):
            raise ValueError("script steps must be a list")
        self.steps = list(steps)
        for index, step in enumerate(self.steps, start=1):
            if not isinstance(step, dict):
                raise ValueError(f"script step {index} must be an object")
            if not isinstance(step.get("tool"), str) or not step["tool"]:
                raise ValueError(f"script step {index} must name a tool")
            if "required" in step and not isinstance(step["required"], bool):
                raise ValueError(f"script step {index} required must be a boolean")
            if "repeat" in step and not isinstance(step["repeat"], bool):
                raise ValueError(f"script step {index} repeat must be a boolean")
            delay = step.get("delay_s", 0)
            if type(delay) not in (int, float) or not 0 <= delay <= 30:
                raise ValueError(f"script step {index} delay_s must be between 0 and 30")
        self.position = 0
        self._states = ["pending"] * len(self.steps)
        self.lock = threading.Lock()
        self.requests: list[dict] = []
        self.request_log = Path(document["request_log"]) if isinstance(document.get("request_log"), str) else None
        # Optional llama.cpp-style POST .../tokenize, for tests of the token meter. Absent = 404
        # (the meter then uses its conservative estimate). Modes:
        #   {"mode": "ok"}                          answer every call
        #   {"mode": "fail", "status": 503}         error every call
        #   {"mode": "slow", "delay": 3.0}          answer every call after a delay
        #   {"mode": "slow_once", "delay": 0.5}     delay only the first call, then answer
        self.tokenize = dict(document.get("tokenize") or {})
        # Optional llama.cpp-style GET /slots: a list of per-slot n_ctx, e.g. [32768] (absent = 404).
        self.slots = [int(n) for n in document.get("slots") or []]
        self.tokenize_calls = 0
        self.slots_calls = 0
        # Optional path: one JSON line per answered /tokenize call, so a test can see which server counted.
        self.tokenize_log = (Path(document["tokenize_log"]) if isinstance(document.get("tokenize_log"), str)
                             else None)

    def next_call(self, offered: list[str]) -> dict:
        with self.lock:
            while self.position < len(self.steps):
                index = self.position
                step = self.steps[index]
                if step["tool"] in offered:
                    if not step.get("repeat", False):
                        self.position += 1
                    self._states[index] = "consumed"
                    return step
                self.position += 1
                self._states[index] = "skipped"
                if step.get("required", True):
                    raise ScriptStepUnavailable(
                        f"required script step {index + 1} ({step['tool']!r}) "
                        "is not offered by the endpoint tool menu")
            return {"tool": "done", "arguments": {"summary": "scripted run finished"}}

    def assert_consumed(self) -> None:
        """Fail with each required step that was skipped or never reached."""
        with self.lock:
            missing = [
                f"step {index + 1} ({step['tool']!r}, {self._states[index]})"
                for index, step in enumerate(self.steps)
                if step.get("required", True) and self._states[index] != "consumed"
            ]
        if missing:
            raise AssertionError("required scripted steps were not consumed: " + "; ".join(missing))


def _usage(body: dict) -> dict:
    size = len(json.dumps(body.get("messages") or []))
    prompt = max(1, size // 4)
    return {"prompt_tokens": prompt, "completion_tokens": 16, "total_tokens": prompt + 16}


def make_handler(script: Script):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:  # quiet
            return

        def _send(self, status: int, payload: object) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _tokenize(self, raw: bytes) -> None:
            config = script.tokenize
            with script.lock:
                script.tokenize_calls += 1
                first = script.tokenize_calls == 1
            mode = config.get("mode", "ok")
            if mode == "fail":
                self._send(int(config.get("status", 503)), {"error": {"message": "tokenizer unavailable"}})
                return
            if mode == "slow" or (mode == "slow_once" and first):
                time.sleep(float(config.get("delay", 3.0)))
            try:
                content = str(json.loads(raw).get("content") or "")
            except ValueError:
                self._send(400, {"error": {"message": "invalid JSON"}})
                return
            # Roughly 4 characters per token, as integer ids; only the count matters to the meter.
            count = max(1, len(content) // 4)
            if script.tokenize_log is not None:
                with script.lock, script.tokenize_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"path": self.path, "tokens": count}) + "\n")
            self._send(200, {"tokens": list(range(count))})

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": script.model, "object": "model"}]})
            elif self.path.rstrip("/") == "/slots" and script.slots:
                with script.lock:
                    script.slots_calls += 1
                self._send(200, [{"id": i, "n_ctx": n, "is_processing": False} for i, n in enumerate(script.slots)])
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if self.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            if self.path.rstrip("/").endswith("/tokenize") and script.tokenize:
                self._tokenize(raw)
                return
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                body = json.loads(raw)
            except ValueError:
                self._send(400, {"error": {"message": "invalid JSON"}})
                return
            script.requests.append(body)
            if script.request_log is not None:
                with script.lock, script.request_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(body, ensure_ascii=False) + "\n")
            offered = [t.get("function", {}).get("name") for t in body.get("tools") or []
                       if isinstance(t, dict)]
            try:
                step = script.next_call(offered)
            except ScriptStepUnavailable as error:
                self._send(409, {"error": {"message": str(error)}})
                return
            if step.get("delay_s"):
                time.sleep(float(step["delay_s"]))
            call = {"id": f"call_{len(script.requests)}", "type": "function",
                    "function": {"name": step["tool"],
                                 "arguments": json.dumps(step.get("arguments") or {})}}
            self._send(200, {
                "id": f"chatcmpl-{len(script.requests)}", "object": "chat.completion",
                "model": script.model,
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": "",
                                         "tool_calls": [call]}}],
                "usage": _usage(body),
            })

    return Handler


def serve(script: Script, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(script))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Scripted OpenAI-compatible endpoint (tests, demos)")
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    script = Script(json.loads(args.script.read_text()))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(script))
    print(f"http://{args.host}:{server.server_address[1]}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
