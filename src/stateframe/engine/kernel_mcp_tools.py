"""Explicit MCP stdio servers: validation, bounded schemas and discovery."""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

if __package__:
    from .kernel_custom_tools import (CustomToolError, NETWORK_ORDER, _canonical, _digest,
        _file_digest, _json_constant, _json_object, resolve_executable)
    from .kernel_exec_capability import executable_read_roots, shell_capability
    from .kernel_exec_session import ExecSession, ExecSessionError
else:
    from kernel_custom_tools import (CustomToolError, NETWORK_ORDER, _canonical, _digest,
        _file_digest, _json_constant, _json_object, resolve_executable)
    from kernel_exec_capability import executable_read_roots, shell_capability
    from kernel_exec_session import ExecSession, ExecSessionError

if __package__ and str(Path(__file__).parent) not in sys.path:
    # Host runner imports this namespace package; engine modules use flat imports.
    sys.path.insert(0, str(Path(__file__).parent))

SERVER_NAME = re.compile(r"^[a-z][a-z0-9_]{1,20}$")
TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
MAX_MESSAGE = 4 << 20
MAX_PAGES = 32
SCHEMA_BUDGET = 1500
DRIVER = Path(__file__).with_name("kernel_mcp_driver.py")


class McpError(CustomToolError):
    """An MCP definition, discovery response or driver result is invalid."""


def _fail(where: str, field: str, reason: str) -> None:
    raise McpError(f"{where}:{field}: {reason}")


def _description(value: object, limit: int = 300) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _omitted_name(server: str, remote: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_./:-]", "_", remote)
    if len(safe) > 80:
        safe = safe[:65] + "#" + _digest(remote.encode(errors="replace"))[:12]
    return f"{server}:{safe}"


def _schema(value: object, depth: int = 0) -> dict:
    """Keep a small, executable JSON Schema subset; never pass arbitrary server data."""
    if depth == 0 and (not isinstance(value, dict) or value.get("type") != "object"):
        raise McpError("inputSchema must have type object")
    if depth > 4 or not isinstance(value, dict):
        return {"type": "object", "properties": {}, "additionalProperties": False} if depth == 0 else {"type": "string"}
    kind = value.get("type")
    if depth == 0:
        kind = "object"
    if kind not in {"object", "array", "string", "integer", "number", "boolean"}:
        return {"type": "string"}
    result: dict = {"type": kind}
    if isinstance(value.get("description"), str):
        result["description"] = value["description"][:200]
    if kind == "object":
        raw = value.get("properties")
        props = raw if isinstance(raw, dict) else {}
        result["properties"] = {name: _schema(spec, depth + 1) for name, spec in list(props.items())[:16]
                                if isinstance(name, str) and len(name) <= 64}
        required = value.get("required")
        if required is not None and (not isinstance(required, list) or
                any(not isinstance(name, str) or name not in result["properties"] for name in required)
                or len(required) != len(set(required))):
            raise McpError("inputSchema has unsupported or duplicate required properties")
        result["required"] = required or []
        result["additionalProperties"] = False
    elif kind == "array":
        result["items"] = _schema(value.get("items"), depth + 1)
        result["maxItems"] = min(32, value.get("maxItems", 32)) if type(value.get("maxItems")) is int and value["maxItems"] >= 0 else 32
    else:
        enum = value.get("enum")
        if isinstance(enum, list):
            admissible = [item for item in enum[:20] if _type_ok(item, kind)]
            if admissible:
                result["enum"] = admissible
        if kind == "string":
            result["maxLength"] = min(1000, value.get("maxLength", 1000)) if type(value.get("maxLength")) is int and value["maxLength"] >= 0 else 1000
    if depth == 0 and len(_canonical(result)) > 8192:
        raise McpError("inputSchema exceeds 8 KiB after sanitizing")
    return result


def _type_ok(value: object, kind: str) -> bool:
    return (type(value) is str if kind == "string" else
            type(value) is int if kind == "integer" else
            type(value) in (int, float) and math.isfinite(value) if kind == "number" else
            type(value) is bool if kind == "boolean" else
            isinstance(value, dict) if kind == "object" else isinstance(value, list))


def validate_arguments(schema: dict, arguments: Mapping[str, object]) -> None:
    def check(spec: dict, value: object, path: str) -> None:
        kind = spec["type"]
        if not _type_ok(value, kind):
            raise McpError(f"{path}: wrong type")
        if kind == "object":
            props = spec["properties"]
            if set(value) - set(props) or set(spec["required"]) - set(value):
                raise McpError(f"{path}: missing or unknown properties")
            for key, item in value.items():
                check(props[key], item, f"{path}.{key}")
        elif kind == "array":
            if len(value) > spec["maxItems"]:
                raise McpError(f"{path}: too many items")
            for index, item in enumerate(value):
                check(spec["items"], item, f"{path}[{index}]")
        elif kind == "string" and (len(value) > spec["maxLength"] or "\0" in value):
            raise McpError(f"{path}: string constraint violated")
        if "enum" in spec and value not in spec["enum"]:
            raise McpError(f"{path}: outside enum")
    if not isinstance(arguments, Mapping):
        raise McpError("arguments: expected object")
    check(schema, dict(arguments), "arguments")


@dataclass(frozen=True)
class McpServer:
    name: str
    trust: str
    argv: tuple[str, ...]
    executable: Path
    executable_digest: str
    definition_digest: str
    network: str
    writes: str
    replay: str
    timeout_seconds: float
    allow: tuple[str, ...] | None

    def verify_executable(self) -> None:
        try:
            current = _file_digest(self.executable)
        except OSError as error:
            raise McpError(f"{self.name}: executable cannot be read: {error}") from error
        if current != self.executable_digest:
            raise McpError(f"{self.name}: executable changed since load")


@dataclass(frozen=True)
class McpTool:
    name: str
    server: McpServer
    remote_name: str
    description: str
    parameters: dict
    definition_digest: str

    @property
    def executable(self) -> Path:
        return self.server.executable

    @property
    def network(self) -> str:
        return self.server.network

    @property
    def writes(self) -> str:
        return self.server.writes

    @property
    def replay(self) -> str:
        return self.server.replay

    @property
    def timeout_seconds(self) -> float:
        return self.server.timeout_seconds

    def verify_executable(self) -> None:
        self.server.verify_executable()

    def build_argv(self, arguments: Mapping[str, object]) -> list[str]:
        validate_arguments(self.parameters, arguments)
        self.stdin_bytes(arguments)
        return [sys.executable, "-I", str(DRIVER)]

    def stdin_bytes(self, arguments: Mapping[str, object]) -> bytes:
        data = _canonical({"argv": [str(self.server.executable), *self.server.argv[1:]],
                           "operation": "call", "tool": self.remote_name,
                           "arguments": dict(arguments), "timeout_seconds": self.timeout_seconds})
        if len(data) > 65536:
            raise McpError("MCP arguments exceed 64 KiB")
        return data

    def request(self, arguments: Mapping[str, object]) -> bytes:
        return self.stdin_bytes(arguments)

    def read_roots(self) -> tuple[Path, ...]:
        return (*executable_read_roots(self.server.executable),
                *executable_read_roots(Path(sys.executable)), DRIVER)


@dataclass(frozen=True)
class McpSet:
    servers: tuple[McpServer, ...]
    tools: tuple[McpTool, ...] = ()
    digest: str = ""
    omitted: tuple[str, ...] = ()

    def __iter__(self):
        return iter(self.servers)

    def __bool__(self) -> bool:
        return bool(self.servers)

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": tool.name,
                 "description": tool.description, "parameters": tool.parameters}}
                for tool in self.tools]


def load_mcp_files(paths: Sequence[Path], *, shell_network: str) -> McpSet:
    if not isinstance(shell_network, str) or shell_network not in NETWORK_ORDER:
        raise McpError("shell_network: invalid network mode")
    servers: list[McpServer] = []
    names: set[str] = set()
    for path in paths:
        path = Path(path).expanduser().resolve()
        try:
            if path.stat().st_size > 1 << 20:
                _fail(str(path), "servers", "file exceeds 1 MiB")
            raw = path.read_bytes()
            if len(raw) > 1 << 20:
                _fail(str(path), "servers", "file exceeds 1 MiB")
            document = json.loads(raw, parse_constant=_json_constant,
                                  object_pairs_hook=_json_object)
        except (OSError, UnicodeError, ValueError) as error:
            raise McpError(f"{path}: cannot read MCP JSON: {error}") from error
        if not isinstance(document, dict) or set(document) != {"schema", "servers"} or document["schema"] != "stateframe-mcp/1":
            _fail(str(path), "schema", "expected stateframe-mcp/1 with servers")
        if not isinstance(document["servers"], list):
            _fail(str(path), "servers", "expected list")
        for index, definition in enumerate(document["servers"]):
            where = f"{path}:{definition.get('name', index) if isinstance(definition, dict) else index}"
            if not isinstance(definition, dict):
                _fail(where, "server", "expected object")
            unknown = set(definition) - {"name", "trust", "argv", "network", "writes", "replay", "timeout_seconds", "tools"}
            if unknown:
                _fail(where, sorted(unknown)[0], "unsupported field")
            name = definition.get("name")
            if not isinstance(name, str) or not SERVER_NAME.fullmatch(name) or name in names:
                _fail(where, "name", "invalid or duplicate server name")
            names.add(name)
            trust = definition.get("trust")
            if not isinstance(trust, str) or not trust.strip() or len(trust) > 300:
                _fail(where, "trust", "expected nonempty statement of at most 300 characters")
            argv = definition.get("argv")
            if not isinstance(argv, list) or not argv or len(argv) > 32 or any(
                    not isinstance(item, str) or not item or "\0" in item or "{" in item or "}" in item for item in argv) or sum(map(len, argv)) > 16384:
                _fail(where, "argv", "expected 1 to 32 fixed string items without placeholders")
            executable = resolve_executable(argv, where)
            network = definition.get("network", "none")
            if not isinstance(network, str) or network not in NETWORK_ORDER:
                _fail(where, "network", "expected none, loopback, or full")
            if NETWORK_ORDER[network] > NETWORK_ORDER[shell_network]:
                _fail(where, "network", "exceeds --shell-network")
            writes = definition.get("writes", "none")
            if writes not in ("none", "workspace"):
                _fail(where, "writes", "expected none or workspace")
            replay = definition.get("replay", "external")
            if replay not in ("external", "local", "observation"):
                _fail(where, "replay", "expected external, local, or observation")
            if replay == "observation" and (network != "none" or writes != "none"):
                _fail(where, "replay", "observation requires network none and writes none")
            if network == "full" and replay != "external":
                _fail(where, "replay", "full network requires external replay")
            timeout = definition.get("timeout_seconds")
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
                _fail(where, "timeout_seconds", "expected finite number greater than 0 and at most 600")
            allow = definition.get("tools")
            if allow is not None and (not isinstance(allow, list) or len(allow) > 256 or
                    any(not isinstance(item, str) or not item for item in allow) or len(set(allow)) != len(allow)):
                _fail(where, "tools", "expected unique nonempty tool names")
            try:
                executable_digest = _file_digest(executable)
            except OSError as error:
                _fail(where, "argv", f"executable cannot be hashed: {error}")
            servers.append(McpServer(name, trust, tuple(argv), executable, executable_digest,
                                     _digest(_canonical(definition)), network, writes, replay,
                                     float(timeout), tuple(allow) if allow is not None else None))
    return McpSet(tuple(servers))


def _driver(server: McpServer, session, base_cap, workspace: Path, request: bytes) -> dict:
    server.verify_executable()
    discovery = json.loads(request).get("operation") == "list"
    argument_files = tuple(Path(arg).resolve() for arg in server.argv[1:]
                           if Path(arg).is_absolute() and Path(arg).is_file())
    cap = replace(base_cap, purpose="custom_tool", network=server.network,
                  workspace_writable=server.writes == "workspace" and not discovery,
                  read_roots=(*base_cap.read_roots, *executable_read_roots(server.executable),
                              *executable_read_roots(Path(sys.executable)), DRIVER, *argument_files),
                  limits=replace(base_cap.limits, wall_seconds=server.timeout_seconds,
                                 output_bytes=4 << 20))
    result = session.run([sys.executable, "-I", str(DRIVER)], cap, stdin=request, cwd=workspace)
    if not result.cleanup.ok:
        raise McpError(f"{server.name}: cleanup failed: {result.cleanup.detail}")
    if result.timed_out:
        raise McpError(f"{server.name}: discovery timed out")
    if result.output_truncated or result.exit_code != 0:
        detail = result.output[:300].decode("utf-8", errors="replace").strip()
        raise McpError(f"{server.name}: driver failed or output exceeded limit: {detail}")
    try:
        envelope = json.loads(result.output)
    except (ValueError, UnicodeError) as error:
        raise McpError(f"{server.name}: malformed driver envelope") from error
    if not isinstance(envelope, dict) or envelope.get("ok") is not True or "result" not in envelope:
        raise McpError(f"{server.name}: {envelope.get('error', 'invalid driver envelope') if isinstance(envelope, dict) else 'invalid driver envelope'}")
    return envelope["result"]


def discover_mcp(config: McpSet, session, base_cap, workspace: Path, *, custom_schemas=(),
                 endpoint: str | None = None) -> McpSet:
    # Host and child must admit the same tools, so both use the engine meter.
    # Its own flat imports resolve through the engine path added at import time.
    if __package__:
        from .kernel_request_diet import RequestTokenMeter
    else:
        from kernel_request_diet import RequestTokenMeter
    meter = RequestTokenMeter(endpoint=endpoint, template_overhead_tokens=0)
    budget_custom = json.loads(json.dumps(custom_schemas))
    for schema in budget_custom:
        schema["function"]["parameters"]["properties"].setdefault("why", {"type": "string"})
    admitted: list[McpTool] = []
    omitted: list[str] = []
    seen: set[str] = {item.get("function", {}).get("name") for item in custom_schemas
                      if isinstance(item, dict) and isinstance(item.get("function"), dict)
                      and isinstance(item["function"].get("name"), str)}
    pinned = []
    for server in config.servers:
        request = _canonical({"argv": [str(server.executable), *server.argv[1:]],
                              "operation": "list", "timeout_seconds": server.timeout_seconds})
        value = _driver(server, session, base_cap, workspace, request)
        if not isinstance(value, list) or len(value) > 512:
            raise McpError(f"{server.name}: invalid tools/list result")
        pinned.append((server.definition_digest, server.executable_digest, value))
        for raw in value:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                omitted.append(f"{server.name}:invalid tool")
                continue
            remote = raw["name"]
            if server.allow is not None and remote not in server.allow:
                continue
            if not remote or len(remote) > 128:
                omitted.append(_omitted_name(server.name, remote))
                continue
            name = f"mcp_{server.name}__{re.sub('[^a-z0-9_]', '_', remote.lower())}"
            if len(name) > 64 or not TOOL_NAME.fullmatch(name) or name in seen:
                omitted.append(_omitted_name(server.name, remote))
                continue
            seen.add(name)
            try:
                parameters = _schema(raw.get("inputSchema"))
            except McpError:
                omitted.append(_omitted_name(server.name, remote))
                continue
            tool = McpTool(name, server, remote, _description(raw.get("description")),
                           parameters, _digest(_canonical((server.definition_digest,
                           server.executable_digest, value))))
            candidate = [*budget_custom, *(McpSet((), tuple(admitted + [tool])).schemas())]
            count, _ = meter.measure({"messages": [], "tools": candidate})
            if count > SCHEMA_BUDGET:
                omitted.append(name)
                continue
            admitted.append(tool)
    return McpSet(config.servers, tuple(admitted), _digest(_canonical(pinned)), tuple(omitted))


def discover(servers: Sequence[McpServer], workspace: Path, run_dir: Path, *,
             shell_network: str, host_authorization: bool, custom_schemas=(),
             endpoint: str | None = None) -> McpSet:
    """Use the same sandbox boundary on the host and in the engine child."""
    workspace = Path(workspace).resolve(strict=True)
    run_dir = Path(run_dir).resolve(strict=True)
    scratch = run_dir / "mcp-discovery-home"
    scratch.mkdir(mode=0o700, exist_ok=True)
    cache = run_dir / "mcp-discovery-cache"
    cache.mkdir(mode=0o700, exist_ok=True)
    cap = shell_capability(workspace=workspace, run_env_dir=cache, scratch_home=scratch,
                           network=shell_network, writable=True)
    cap = replace(cap, sandboxed=not host_authorization)
    try:
        with ExecSession(run_dir) as session:
            session.configure(cap)
            return discover_mcp(McpSet(tuple(servers)), session, cap, workspace,
                                custom_schemas=custom_schemas, endpoint=endpoint)
    except ExecSessionError as error:
        raise McpError(f"MCP discovery boundary failed: {error}") from error


def run_driver(server: McpServer, action: str, workspace: Path, run_dir: Path, *,
               shell_network: str, host_authorization: bool,
               tool: str = "", arguments: dict | None = None) -> dict:
    """Run one isolated protocol exchange through the same discovery boundary."""
    workspace = Path(workspace).resolve(strict=True)
    run_dir = Path(run_dir).resolve(strict=True)
    scratch = run_dir / "mcp-call-home"
    scratch.mkdir(mode=0o700, exist_ok=True)
    cache = run_dir / "mcp-call-cache"
    cache.mkdir(mode=0o700, exist_ok=True)
    cap = shell_capability(workspace=workspace, run_env_dir=cache, scratch_home=scratch,
                           network=shell_network, writable=True)
    cap = replace(cap, sandboxed=not host_authorization)
    request = _canonical({"argv": [str(server.executable), *server.argv[1:]],
                          "operation": action, "tool": tool, "arguments": arguments or {},
                          "timeout_seconds": server.timeout_seconds})
    with ExecSession(run_dir) as session:
        session.configure(cap)
        value = _driver(server, session, cap, workspace, request)
    return {"ok": True, "tools" if action == "list" else "result": value}


def result_text(payload: dict) -> str:
    parts = []
    for item in payload["content"][:128]:
        if not isinstance(item, dict):
            raise ValueError("invalid content item")
        kind = item.get("type")
        if kind == "text" and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif kind == "image":
            parts.append("[image omitted]")
        elif kind == "resource":
            resource = item.get("resource")
            uri = resource.get("uri") if isinstance(resource, dict) else item.get("uri")
            parts.append(f"[resource: {str(uri)[:300]}]")
        else:
            parts.append(f"[{str(kind)[:40]} omitted]")
    return "\n".join(parts)
