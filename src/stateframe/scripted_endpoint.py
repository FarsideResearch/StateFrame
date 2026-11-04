"""A scripted OpenAI-compatible chat endpoint, for tests and offline demos.

It answers ``POST /v1/chat/completions`` with tool calls taken, in order, from a script
file, ``GET /v1/models`` with one model, and ``GET /props`` with llama.cpp's public
identification shape. ``GET /slots`` remains optional. It is not a model: it proves the
CLI, engine wiring and request recording without a GPU.

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
import hmac
import json
import threading
import time
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_STRICT_TOP_LEVEL_KEYS = frozenset({
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls",
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "n",
    "stop", "stream", "stream_options", "seed", "logprobs", "top_logprobs",
    "response_format", "frequency_penalty", "presence_penalty", "logit_bias",
    "user", "reasoning_effort", "service_tier", "store", "metadata",
    "modalities", "audio", "moderation", "prediction", "functions", "function_call",
    "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "safety_identifier",
    "verbosity", "web_search_options",
})
_STRICT_MESSAGE_KEYS = frozenset({"role", "content", "name", "tool_call_id", "tool_calls",
                                  "function_call", "refusal", "audio"})
_STRICT_CONTENT_KEYS = frozenset({"type", "text", "image_url", "input_audio", "file", "refusal",
                                  "prompt_cache_breakpoint"})
_STRICT_TOOL_KEYS = frozenset({"type", "function"})
_STRICT_FUNCTION_KEYS = frozenset({"name", "description", "parameters", "strict"})
_STRICT_LEGACY_FUNCTION_KEYS = frozenset({"name", "description", "parameters"})
_STRICT_TOOL_CALL_KEYS = frozenset({"id", "type", "function"})
_STRICT_TOOL_CALL_FUNCTION_KEYS = frozenset({"name", "arguments"})
_STRICT_MESSAGE_FUNCTION_CALL_KEYS = frozenset({"name", "arguments"})
_STRICT_AUDIO_MESSAGE_KEYS = frozenset({"id"})
_STRICT_SCHEMA_KEYS = frozenset({
    "$schema", "$id", "$ref", "$anchor", "$dynamicRef", "$dynamicAnchor", "$defs",
    "$comment", "$vocabulary", "definitions", "type", "properties", "patternProperties",
    "additionalProperties", "unevaluatedProperties", "propertyNames", "required",
    "dependentRequired", "dependentSchemas", "items", "prefixItems", "additionalItems",
    "contains", "minContains", "maxContains", "unevaluatedItems", "enum", "const",
    "anyOf", "oneOf", "allOf", "not", "if", "then", "else", "title", "description",
    "default", "deprecated", "readOnly", "writeOnly", "examples", "format",
    "contentEncoding", "contentMediaType", "contentSchema", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength",
    "pattern", "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
})
_STRICT_RESPONSE_FORMAT_KEYS = frozenset({"type", "json_schema"})
_STRICT_JSON_SCHEMA_FORMAT_KEYS = frozenset({"name", "description", "strict", "schema"})
_STRICT_CACHE_BREAKPOINT_KEYS = frozenset({"mode", "ttl"})
_STRICT_PROMPT_CACHE_OPTIONS_KEYS = frozenset({"mode", "ttl"})
_STRICT_MODERATION_KEYS = frozenset({"model", "policy"})
_STRICT_MODERATION_POLICY_KEYS = frozenset({"input", "output"})
_STRICT_MODERATION_MODE_KEYS = frozenset({"mode"})


def _strict_schema_error(value: object, path: str) -> str | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, Mapping):
        return f"{path} must be an object"
    unknown = set(value) - _STRICT_SCHEMA_KEYS
    if unknown:
        return f"unknown keys at {path}: {', '.join(sorted(map(str, unknown)))}"
    for field in ("properties", "$defs", "definitions", "patternProperties", "dependentSchemas"):
        nested = value.get(field)
        if isinstance(nested, Mapping):
            for name, schema in nested.items():
                error = _strict_schema_error(schema, f"{path}.{field}.{name}")
                if error:
                    return error
    for field in ("items", "additionalItems", "additionalProperties", "not", "if", "then", "else",
                  "contains", "propertyNames", "unevaluatedItems", "unevaluatedProperties", "contentSchema"):
        nested = value.get(field)
        if isinstance(nested, Mapping):
            error = _strict_schema_error(nested, f"{path}.{field}")
            if error:
                return error
    for field in ("items", "prefixItems"):
        nested = value.get(field)
        if isinstance(nested, list):
            for index, schema in enumerate(nested):
                error = _strict_schema_error(schema, f"{path}.{field}[{index}]")
                if error:
                    return error
    for field in ("anyOf", "oneOf", "allOf"):
        nested = value.get(field)
        if isinstance(nested, list):
            for index, schema in enumerate(nested):
                error = _strict_schema_error(schema, f"{path}.{field}[{index}]")
                if error:
                    return error
    return None


def _strict_openai_error(body: object) -> str | None:
    if not isinstance(body, Mapping):
        return "request body must be an object"

    def check(value: Mapping[str, object], allowed: frozenset[str], path: str) -> str | None:
        unknown = set(value) - allowed
        return f"unknown keys at {path}: {', '.join(sorted(map(str, unknown)))}" if unknown else None

    error = check(body, _STRICT_TOP_LEVEL_KEYS, "request")
    if error:
        return error
    messages = body.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                return f"messages[{index}] must be an object"
            path = f"messages[{index}]"
            error = check(message, _STRICT_MESSAGE_KEYS, path)
            if error:
                return error
            content = message.get("content")
            if content is not None and not isinstance(content, (str, list)):
                return f"{path}.content must be a string or content-part array"
            if isinstance(content, list):
                for part_index, part in enumerate(content):
                    if not isinstance(part, Mapping):
                        return f"{path}.content[{part_index}] must be an object"
                    error = check(part, _STRICT_CONTENT_KEYS, f"{path}.content[{part_index}]")
                    if error:
                        return error
                    for field, allowed in (("image_url", frozenset({"url", "detail"})),
                                           ("input_audio", frozenset({"data", "format"})),
                                           ("file", frozenset({"file_data", "file_id", "filename"})),
                                           ("prompt_cache_breakpoint", _STRICT_CACHE_BREAKPOINT_KEYS)):
                        nested = part.get(field)
                        if isinstance(nested, Mapping):
                            error = check(nested, allowed, f"{path}.content[{part_index}].{field}")
                            if error:
                                return error
            audio = message.get("audio")
            if isinstance(audio, Mapping):
                error = check(audio, _STRICT_AUDIO_MESSAGE_KEYS, f"{path}.audio")
                if error:
                    return error
            function_call = message.get("function_call")
            if isinstance(function_call, Mapping):
                error = check(function_call, _STRICT_MESSAGE_FUNCTION_CALL_KEYS,
                              f"{path}.function_call")
                if error:
                    return error
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call_index, call in enumerate(calls):
                    if not isinstance(call, Mapping):
                        return f"{path}.tool_calls[{call_index}] must be an object"
                    call_path = f"{path}.tool_calls[{call_index}]"
                    error = check(call, _STRICT_TOOL_CALL_KEYS, call_path)
                    if error:
                        return error
                    function = call.get("function")
                    if not isinstance(function, Mapping):
                        return f"{call_path}.function must be an object"
                    error = check(function, _STRICT_TOOL_CALL_FUNCTION_KEYS, f"{call_path}.function")
                    if error:
                        return error
    tools = body.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            if not isinstance(tool, Mapping):
                return f"tools[{index}] must be an object"
            path = f"tools[{index}]"
            error = check(tool, _STRICT_TOOL_KEYS, path)
            if error:
                return error
            function = tool.get("function")
            if not isinstance(function, Mapping):
                return f"{path}.function must be an object"
            error = check(function, _STRICT_FUNCTION_KEYS, f"{path}.function")
            if error:
                return error
            parameters = function.get("parameters")
            if parameters is not None:
                error = _strict_schema_error(parameters, f"{path}.function.parameters")
                if error:
                    return error
    functions = body.get("functions")
    if isinstance(functions, list):
        for index, function in enumerate(functions):
            if not isinstance(function, Mapping):
                return f"functions[{index}] must be an object"
            error = check(function, _STRICT_LEGACY_FUNCTION_KEYS, f"functions[{index}]")
            if error:
                return error
            schema = function.get("parameters")
            if schema is not None:
                error = _strict_schema_error(schema, f"functions[{index}].parameters")
                if error:
                    return error
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, Mapping):
        error = check(tool_choice, frozenset({"type", "function"}), "tool_choice")
        if error:
            return error
        function = tool_choice.get("function")
        if isinstance(function, Mapping):
            error = check(function, frozenset({"name"}), "tool_choice.function")
            if error:
                return error
    function_call = body.get("function_call")
    if isinstance(function_call, Mapping):
        error = check(function_call, frozenset({"name"}), "function_call")
        if error:
            return error
    response_format = body.get("response_format")
    if isinstance(response_format, Mapping):
        error = check(response_format, _STRICT_RESPONSE_FORMAT_KEYS, "response_format")
        if error:
            return error
        schema_format = response_format.get("json_schema")
        if isinstance(schema_format, Mapping):
            error = check(schema_format, _STRICT_JSON_SCHEMA_FORMAT_KEYS, "response_format.json_schema")
            if error:
                return error
            schema = schema_format.get("schema")
            if schema is not None:
                error = _strict_schema_error(schema, "response_format.json_schema.schema")
                if error:
                    return error
    for field, allowed in (("stream_options", frozenset({"include_usage"})),
                           ("prompt_cache_options", _STRICT_PROMPT_CACHE_OPTIONS_KEYS)):
        value = body.get(field)
        if isinstance(value, Mapping):
            error = check(value, allowed, field)
            if error:
                return error
    audio = body.get("audio")
    if isinstance(audio, Mapping):
        error = check(audio, frozenset({"voice", "format"}), "audio")
        if error:
            return error
    prediction = body.get("prediction")
    if isinstance(prediction, Mapping):
        error = check(prediction, frozenset({"type", "content"}), "prediction")
        if error:
            return error
    search = body.get("web_search_options")
    if isinstance(search, Mapping):
        error = check(search, frozenset({"search_context_size", "user_location"}), "web_search_options")
        if error:
            return error
        location = search.get("user_location")
        if isinstance(location, Mapping):
            error = check(location, frozenset({"type", "approximate"}), "web_search_options.user_location")
            if error:
                return error
            approximate = location.get("approximate")
            if isinstance(approximate, Mapping):
                error = check(approximate, frozenset({"city", "country", "region", "timezone"}),
                              "web_search_options.user_location.approximate")
                if error:
                    return error
    moderation = body.get("moderation")
    if isinstance(moderation, Mapping):
        error = check(moderation, _STRICT_MODERATION_KEYS, "moderation")
        if error:
            return error
        policy = moderation.get("policy")
        if isinstance(policy, Mapping):
            error = check(policy, _STRICT_MODERATION_POLICY_KEYS, "moderation.policy")
            if error:
                return error
            for channel in ("input", "output"):
                value = policy.get(channel)
                if isinstance(value, Mapping):
                    error = check(value, _STRICT_MODERATION_MODE_KEYS, f"moderation.policy.{channel}")
                    if error:
                        return error
    return None


class ScriptStepUnavailable(RuntimeError):
    """A required scripted action is missing from the offered tool menu."""


class Script:
    def __init__(self, document: dict) -> None:
        self.model = str(document.get("model") or "scripted")
        strict_openai = document.get("strict_openai", False)
        if not isinstance(strict_openai, bool):
            raise ValueError("strict_openai must be a boolean")
        self.strict_openai = strict_openai
        self.strict_openai_errors: list[str] = []
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
        chat = document.get("chat")
        if chat is None:
            chat = {}
        if not isinstance(chat, dict):
            raise ValueError("chat settings must be an object")
        self.chat_require_key = chat.get("require_key")
        if self.chat_require_key is not None and (
                not isinstance(self.chat_require_key, str) or not self.chat_require_key):
            raise ValueError("chat require_key must be a non-empty string")
        self.chat_authorization_present: list[bool] = []
        self.chat_authorized: list[bool] = []
        # Optional llama.cpp-style POST .../tokenize, for tests of the token meter. Absent = 404
        # (the meter then uses its conservative estimate). Modes:
        #   {"mode": "ok"}                          answer every call
        #   {"mode": "fail", "status": 503}         error every call
        #   {"mode": "slow", "delay": 3.0}          answer every call after a delay
        #   {"mode": "slow_once", "delay": 0.5}     delay only the first call, then answer
        #   {"mode": "bad_json"}                     return invalid JSON
        #   {"mode": "no_tokens"}                    return JSON with no tokens
        self.tokenize = dict(document.get("tokenize") or {})
        self.record_tokenize_request_bytes = self.tokenize.get("record_request_bytes") is True
        required_tokenize_key = self.tokenize.get("require_key")
        if required_tokenize_key is not None and (
                not isinstance(required_tokenize_key, str) or not required_tokenize_key):
            raise ValueError("tokenize require_key must be a non-empty string")
        self.tokenize_authorization_present: list[bool] = []
        self.tokenize_authorized: list[bool] = []
        self.tokenize_request_bytes: list[bytes] = []
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
            if mode == "bad_json":
                raw_response = b"not-json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw_response)))
                self.end_headers()
                self.wfile.write(raw_response)
                return
            if mode == "no_tokens":
                self._send(200, {})
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
            if self.path.rstrip("/") == "/props":
                if script.strict_openai:
                    self._send(404, {"error": {"message": "not found"}})
                else:
                    self._send(200, {
                        "default_generation_settings": {"n_ctx": 65536},
                        "model_path": "scripted-endpoint",
                    })
            elif self.path.rstrip("/") == "/v1/models":
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
                authorization = self.headers.get("Authorization") or ""
                required_key = script.tokenize.get("require_key")
                authorized = (
                    required_key is None
                    or hmac.compare_digest(authorization, f"Bearer {required_key}")
                )
                with script.lock:
                    script.tokenize_authorization_present.append(bool(authorization))
                    script.tokenize_authorized.append(authorized)
                    if script.record_tokenize_request_bytes:
                        script.tokenize_request_bytes.append(raw)
                if not authorized:
                    self._send(401, {"error": {"message": "unauthorized"}})
                    return
                self._tokenize(raw)
                return
            if self.path.rstrip("/") != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            authorization = self.headers.get("Authorization") or ""
            authorized = (
                script.chat_require_key is None
                or hmac.compare_digest(authorization, f"Bearer {script.chat_require_key}")
            )
            with script.lock:
                script.chat_authorization_present.append(bool(authorization))
                script.chat_authorized.append(authorized)
            if not authorized:
                self._send(401, {"error": {"message": "unauthorized"}})
                return
            try:
                body = json.loads(raw)
            except ValueError:
                self._send(400, {"error": {"message": "invalid JSON"}})
                return
            script.requests.append(body)
            if script.strict_openai:
                schema_error = _strict_openai_error(body)
                if schema_error:
                    script.strict_openai_errors.append(schema_error)
                    self._send(400, {"error": {"message": schema_error}})
                    return
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
