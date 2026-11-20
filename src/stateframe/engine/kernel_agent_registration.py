"""Reusable static registration and launch-local initial source capture."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
import os
import sys

from stateframe import __version__


def _enabled_store():
    try:
        store = import_module("stateframe.send1_probe_cache")
    except ModuleNotFoundError as error:
        if error.name != "stateframe.send1_probe_cache":
            raise
        return None
    return store if store.enabled() else None


def _ordered_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _registration_key(store, version, *inputs):
    try:
        host = store.host_fingerprint()
        if not host:
            return None
        identity = _ordered_json((version, __version__, host, sys.implementation.cache_tag,
                                  os.environ.get("PATH", os.defpath), *inputs))
    except (OSError, TypeError, ValueError, RecursionError):
        return None
    if len(identity) > 256 * 1024:
        return None
    return hashlib.sha256(identity.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class RegistrationData:
    builtin_schema_snapshot: tuple[dict, ...]
    custom_schemas: list[dict]


def _custom_schemas(custom):
    schemas = custom.schemas(detached=True)
    for schema in schemas:
        schema["function"]["parameters"]["properties"].setdefault("why", {"type": "string"})
    return schemas


def _build_registration(builtins, custom):
    return RegistrationData(deepcopy(builtins), _custom_schemas(custom))


def _read_registration(facts, builtin_wire, custom):
    if type(facts) is not dict or set(facts) != {"builtin", "custom"} or facts["builtin"] != builtin_wire:
        return None
    wire = facts["custom"]
    if not isinstance(wire, str) or len(wire) > 256 * 1024:
        return None
    schemas = json.loads(wire)
    if type(schemas) is not list or len(schemas) != len(custom.tools):
        return None
    for schema, tool in zip(schemas, custom.tools):
        expected = {**tool.parameters, "additionalProperties": False}
        expected["properties"] = {**expected["properties"]}
        expected["properties"].setdefault("why", {"type": "string"})
        if (type(schema) is not dict or list(schema) != ["type", "function"]
                or schema["type"] != "function" or type(schema["function"]) is not dict):
            return None
        function = schema["function"]
        if (list(function) != ["name", "description", "parameters"]
                or function["name"] != tool.name or function["description"] != tool.description
                or _ordered_json(function["parameters"]) != _ordered_json(expected)):
            return None
    return RegistrationData(tuple(json.loads(builtin_wire)), schemas)


def registration_data(shell, custom, mcp_servers=(), *, resume_active=False):
    """Return detached inputs for AgentTools. The caller binds and admits live tools."""
    if __package__:
        from . import kernel_agent_schemas, kernel_git_commit, kernel_shell_tool
    else:
        import kernel_agent_schemas
        import kernel_git_commit
        import kernel_shell_tool

    collision_names = ()
    if custom.tools:
        from kernel_completion_owner import TOOL_SPECS
        from kernel_custom_tools import CustomToolError
        from kernel_lean import _KNOWN_TOOL_NAMES

        collisions = sorted(tool.name for tool in custom.tools
                            if tool.name in TOOL_SPECS or tool.name in _KNOWN_TOOL_NAMES)
        if collisions:
            raise CustomToolError("custom tools duplicate built-in capabilities: " + ", ".join(collisions))
        collision_names = sorted(set(TOOL_SPECS) | _KNOWN_TOOL_NAMES)

    builtins = (kernel_shell_tool.SHELL_SCHEMA, *kernel_agent_schemas.BUILTIN_SCHEMAS)
    if resume_active or mcp_servers or len(custom.tools) > 32:
        return _build_registration(builtins, custom)
    store = _enabled_store()
    if store is None:
        return _build_registration(builtins, custom)

    from stateframe.startup_cache_writes import defer

    def registration_key(builtin_wire):
        # Bind the schema projection to the validated executable and interpreter identities.
        from kernel_custom_tools import _tool_input_identity

        custom_inputs = [
            (tool.name, tool.description, tool.parameters, tool.definition_digest,
             _tool_input_identity(tool.executable, tool.executable_digest))
            for tool in custom.tools
        ]
        commit_schemas = kernel_git_commit.SCHEMAS if kernel_git_commit.enabled() else ()
        return _registration_key(
            store, "agent-registration/1", hashlib.sha256(builtin_wire.encode("ascii")).hexdigest(),
            (shell.spec.get("engine_source") or {}).get("shipped_hash"), commit_schemas, collision_names,
            shell.settings.as_dict(), shell.spec.get("send1_wave4"), "mutations" in shell.spec,
            shell.spec.get("shell_path_dirs"), shell.spec.get("shell_read_roots"),
            custom_inputs,
        )

    builtin_wire = key = None
    if getattr(store, "reads_available", lambda: True)():
        try:
            builtin_wire = _ordered_json(builtins)
            key = registration_key(builtin_wire)
            if key is None:
                return _build_registration(builtins, custom)
            result = _read_registration(store.lookup("agent-registration", key), builtin_wire, custom)
            if result is not None:
                return result
        except (OSError, TypeError, ValueError, RecursionError):
            pass
    result = _build_registration(builtins, custom)

    def publish_registration():
        wire = builtin_wire if builtin_wire is not None else _ordered_json(builtins)
        cache_key = key if key is not None else registration_key(wire)
        # Build from the validated definitions so the live menu can own its copy.
        custom_wire = _ordered_json(_custom_schemas(custom))
        if cache_key is not None and len(wire) + len(custom_wire) <= 256 * 1024:
            store.publish("agent-registration", cache_key, {"builtin": wire, "custom": custom_wire})

    try:
        defer("agent-registration", publish_registration)
    except (OSError, TypeError, ValueError, RecursionError):
        pass
    return result


def initial_product_sources(root, fileset, row):
    """Return (ordered product paths, initial source snapshot) for a fresh multi-file run."""
    from probes.react_repair import _initial_source_snapshot, _is_product_path, _product_source_paths

    purpose = "initial-product-sources"
    observation = None
    if _enabled_store() is not None:
        try:
            from stateframe.workspace_start_snapshot import startup_observation
        except ModuleNotFoundError as error:
            if error.name != "stateframe.workspace_start_snapshot":
                raise
        else:
            observation = startup_observation(root, purpose)
    validate = getattr(observation, "validate_for_use", None)
    if callable(validate) and observation.root == root.absolute() and observation.purpose == purpose:
        members = set(fileset.paths).union(observation.files)
        paths = tuple(sorted(path for path in members if _is_product_path(path, row)))
        # The existing reader can reuse the compact byte spool and preserves
        # the original text decoding, missing targets, and read failures.
        sources = _initial_source_snapshot(root, paths) if paths else None
        if validate(purpose):
            return paths, sources
    paths = _product_source_paths(root, fileset, row)
    return paths, _initial_source_snapshot(root, paths) if paths else None
