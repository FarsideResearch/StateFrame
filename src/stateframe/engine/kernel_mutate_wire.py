"""Strict, bounded mutation wire documents and JSON schema subset."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from urllib.parse import urlsplit


class WireError(ValueError):
    """A malformed or untrusted mutation protocol value."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WireError("DUPLICATE_KEY", key)
        result[key] = value
    return result


def _constant(raw):
    raise WireError("NONFINITE", raw)


def strict_loads(data, *, max_depth=16, max_int_digits=40):
    if not isinstance(data, (bytes, bytearray)):
        raise WireError("TYPE")
    try:
        source = bytes(data).decode("utf-8", "strict")
        value = json.loads(source, object_pairs_hook=_pairs,
                           parse_int=lambda raw: _bounded_int(raw, max_int_digits),
                           parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise WireError("JSON", str(error)) from error
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if depth > max_depth:
            raise WireError("DEPTH")
        if isinstance(item, str):
            try:
                item.encode("utf-8", "strict")
            except UnicodeError as error:
                raise WireError("SURROGATE") from error
        elif isinstance(item, float) and not math.isfinite(item):
            raise WireError("NONFINITE")
        elif isinstance(item, dict):
            for key, child in item.items():
                stack.extend(((key, depth + 1), (child, depth + 1)))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return value


def _bounded_int(raw, digits):
    if len(raw.lstrip("-")) > digits:
        raise WireError("INTEGER_LIMIT")
    return int(raw)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeError) as error:
        raise WireError("JSON", str(error)) from error


def request_key(url: str, accept: str | None = None) -> str:
    _url(url)
    if accept is not None:
        _header_text(accept)
    return "sha256:" + hashlib.sha256(_canonical({"method": "GET", "url": url,
                                                    "accept": accept or ""})).hexdigest()


def _exact(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise WireError("ENVELOPE")


def _int(value, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise WireError("INTEGER")


def _text(value, cap):
    if not isinstance(value, str):
        raise WireError("TEXT_LIMIT")
    try:
        size = len(value.encode("utf-8", "strict"))
    except UnicodeError as error:
        raise WireError("SURROGATE") from error
    if size > cap:
        raise WireError("TEXT_LIMIT")


def _header_text(value):
    _text(value, 8192)
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise WireError("HEADER_VALUE")


def _url(value):
    _text(value, 2048)
    if any(ord(char) <= 32 or ord(char) == 127 for char in value):
        raise WireError("URL")
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise WireError("URL") from error
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise WireError("URL")


def _headers(headers):
    if not isinstance(headers, dict) or len(_canonical(headers)) > 8192:
        raise WireError("HEADERS")
    for key, value in headers.items():
        _header_text(key)
        _header_text(value)


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise WireError("DIGEST")


def _fetch(value):
    _exact(value, ("method", "url", "accept"))
    if value["method"] != "GET":
        raise WireError("METHOD")
    _url(value["url"])
    _header_text(value["accept"])


@dataclass(frozen=True)
class Frame:
    kind: str
    result: object = None
    edits: tuple = ()
    fetch: tuple = ()
    error: object = None
    stderr: str = ""


def parse_stdout(data: bytes, *, call: str, max_body=256 << 10, recorded_keys=(), output_schema=None,
                 phase=None) -> Frame:
    if not isinstance(data, bytes) or len(data) > max_body + 24:
        raise WireError("SIZE")
    match = re.match(rb"SFMW1 ([1-9][0-9]*)\n", data)
    if match is None:
        raise WireError("HEADER")
    if len(match.group(1)) > len(str(max_body)):
        raise WireError("LENGTH")
    size = int(match.group(1))
    body = data[match.end():]
    if size > max_body or size != len(body):
        raise WireError("LENGTH")
    value = strict_loads(body)
    _exact(value, ("wire", "call", "kind"), ("result", "edits", "fetch", "error", "stderr"))
    if type(value["wire"]) is not int or value["wire"] != 1 or value["call"] != call:
        raise WireError("CALL")
    kind = value["kind"]
    if kind == "result":
        _exact(value, ("wire", "call", "kind", "result"), ("edits",))
        if not isinstance(value["result"], dict):
            raise WireError("RESULT")
        if output_schema is not None:
            validate_schema(value["result"], output_schema)
        edits = value.get("edits", [])
        if not isinstance(edits, list) or len(edits) > 16:
            raise WireError("EDITS")
        seen = set()
        for edit in edits:
            _exact(edit, ("op", "path", "base_sha256", "text"))
            if edit["op"] not in ("replace", "create", "delete") or not isinstance(edit["path"], str):
                raise WireError("EDIT")
            path = edit["path"]
            if path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/")) or path in seen:
                raise WireError("EDIT_PATH")
            seen.add(path)
            if edit["base_sha256"] is not None and not re.fullmatch("[0-9a-f]{64}", edit["base_sha256"]):
                raise WireError("EDIT_DIGEST")
            if edit["text"] is not None:
                _text(edit["text"], 256 << 10)
        return Frame(kind, result=value["result"], edits=tuple(edits))
    if kind == "need":
        if phase is not None and (type(phase) is not int or phase >= 3 or phase < 0):
            raise WireError("PHASE")
        _exact(value, ("wire", "call", "kind", "fetch"))
        fetch = value["fetch"]
        if not isinstance(fetch, list) or not 1 <= len(fetch) <= 2:
            raise WireError("NEED_COUNT")
        keys = set(recorded_keys)
        if len(keys) + len(fetch) > 4:
            raise WireError("FETCH_LIMIT")
        for item in fetch:
            _fetch(item)
            key = request_key(item["url"], item["accept"])
            if key in keys:
                raise WireError("NEED_RECORDED")
            keys.add(key)
        return Frame(kind, fetch=tuple(fetch))
    if kind == "error":
        _exact(value, ("wire", "call", "kind", "error", "stderr"))
        _exact(value["error"], ("type", "message", "trace"))
        if value["error"]["type"] not in ("HandlerError", "BootError", "ContractError", "SandboxError"):
            raise WireError("ERROR_TYPE")
        _text(value["error"]["message"], 2048)
        _text(value["error"]["trace"], 8192)
        _text(value["stderr"], 32768)
        return Frame(kind, error=value["error"], stderr=value["stderr"])
    raise WireError("KIND")


def build_stdin_doc(**fields) -> bytes:
    _exact(fields, ("wire", "call", "launch", "phase", "role", "input", "inputs", "responses", "grants", "max_fetch"))
    if type(fields["wire"]) is not int or fields["wire"] != 1:
        raise WireError("WIRE")
    for name in ("call", "launch"):
        _text(fields[name], 128)
    _int(fields["phase"], 0, 3)
    if fields["role"] not in ("payload", "validator", "smoke"):
        raise WireError("ROLE")
    if not isinstance(fields["input"], dict) or not isinstance(fields["inputs"], list) or not isinstance(fields["responses"], list):
        raise WireError("INPUT")
    if len(fields["inputs"]) > 16 or len(fields["responses"]) > 4:
        raise WireError("INPUT_LIMIT")
    _exact(fields["grants"], ("network", "writes"))
    if any(type(fields["grants"][name]) is not bool for name in ("network", "writes")):
        raise WireError("GRANTS")
    _int(fields["max_fetch"], 0, 4)
    input_ids = set()
    input_bytes = 0
    for item in fields["inputs"]:
        _exact(item, ("id", "path", "bytes", "name"))
        _int(item["bytes"], 0, 4 << 20)
        for name in ("id", "path", "name"):
            _text(item[name], 4096)
        ident = item["id"]
        if (re.fullmatch(r"[a-z][a-z0-9_]{0,39}", ident) is None or ident in input_ids
                or item["path"] != "/in/" + ident or not item["name"]
                or "/" in item["name"] or "\\" in item["name"]):
            raise WireError("INPUT_PATH")
        input_ids.add(ident)
        input_bytes += item["bytes"]
    if input_bytes > 16 << 20:
        raise WireError("INPUT_LIMIT")
    keys = set()
    for item in fields["responses"]:
        _exact(item, ("key", "method", "url", "accept", "status", "headers", "final_url", "body", "bytes", "error"))
        _fetch({name: item[name] for name in ("method", "url", "accept")})
        _digest(item["key"])
        if item["key"] != request_key(item["url"], item["accept"]) or item["key"] in keys:
            raise WireError("RESPONSE_KEY")
        keys.add(item["key"])
        _int(item["status"], 0, 599)
        _int(item["bytes"])
        _headers(item["headers"])
        _url(item["final_url"])
        _text(item["body"], 4096)
        if re.fullmatch(r"/resp/[0-9a-f]{64}", item["body"]) is None:
            raise WireError("RESPONSE_PATH")
        if item["error"] is not None:
            _text(item["error"], 2048)
    data = _canonical(fields)
    if len(data) > 1 << 20:
        raise WireError("SIZE")
    strict_loads(data)
    return data


def validate_schema(value, schema, *, schema_limit=8192, max_depth=8):
    """Validate one JSON value against the admitted, bounded schema subset."""
    if len(_canonical(schema)) > schema_limit:
        raise WireError("SCHEMA_SIZE")
    allowed = {"type", "properties", "required", "items", "enum", "maxLength", "maxItems", "additionalProperties"}
    definitions = [(schema, 0)]
    while definitions:
        node, depth = definitions.pop()
        if depth > max_depth or not isinstance(node, dict) or node.keys() - allowed or node.get("type") not in (
                "object", "array", "string", "integer", "number", "boolean", "null"):
            raise WireError("SCHEMA")
        if "properties" in node:
            if node["type"] != "object" or not isinstance(node["properties"], dict):
                raise WireError("SCHEMA")
            definitions.extend((child, depth + 1) for child in node["properties"].values())
        if "items" in node:
            if node["type"] != "array":
                raise WireError("SCHEMA")
            definitions.append((node["items"], depth + 1))
        if "additionalProperties" in node and (node["type"] != "object" or node["additionalProperties"] is not False):
            raise WireError("SCHEMA")
        if "required" in node and (node["type"] != "object" or not isinstance(node["required"], list)
                                   or any(not isinstance(key, str) for key in node["required"])):
            raise WireError("SCHEMA")
        if "enum" in node and not isinstance(node["enum"], list):
            raise WireError("SCHEMA")
        for name in ("maxLength", "maxItems"):
            if name in node and (type(node[name]) is not int or node[name] < 0):
                raise WireError("SCHEMA")
        if ("maxLength" in node and node["type"] != "string") or ("maxItems" in node and node["type"] != "array"):
            raise WireError("SCHEMA")
    stack = [(value, schema, 0)]
    types = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
             "string": lambda x: isinstance(x, str), "integer": lambda x: type(x) is int,
             "number": lambda x: type(x) in (int, float) and math.isfinite(x),
             "boolean": lambda x: type(x) is bool, "null": lambda x: x is None}
    while stack:
        item, node, depth = stack.pop()
        if depth > max_depth or not isinstance(node, dict) or node.keys() - allowed:
            raise WireError("SCHEMA")
        typ = node.get("type")
        if typ not in types or not types[typ](item):
            raise WireError("SCHEMA_TYPE")
        if "enum" in node and (not isinstance(node["enum"], list) or not any(
                candidate == item and (type(candidate) is type(item) or
                                       {type(candidate), type(item)} == {int, float})
                for candidate in node["enum"])):
            raise WireError("SCHEMA_ENUM")
        if typ == "object":
            props = node.get("properties", {})
            required = node.get("required", [])
            if not isinstance(props, dict) or not isinstance(required, list) or any(not isinstance(x, str) for x in required):
                raise WireError("SCHEMA")
            if any(key not in item for key in required):
                raise WireError("SCHEMA_REQUIRED")
            if node.get("additionalProperties") is False and item.keys() - props.keys():
                raise WireError("SCHEMA_EXTRA")
            stack.extend((child, props[key], depth + 1) for key, child in item.items() if key in props)
        if typ == "array":
            if "maxItems" in node and len(item) > node["maxItems"]:
                raise WireError("SCHEMA_MAX_ITEMS")
            if "items" in node:
                stack.extend((child, node["items"], depth + 1) for child in item)
        if typ == "string" and "maxLength" in node and len(item) > node["maxLength"]:
            raise WireError("SCHEMA_MAX_LENGTH")
    return True
