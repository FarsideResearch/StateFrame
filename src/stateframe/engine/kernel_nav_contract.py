# Vendored navigation contract: the release ships only the kernel, so the
# kernel must not import tools.* at runtime. probes/test_nav_contract_drift.py keeps the two byte-identical below this header.
"""Canonical typed navigation contract shared by FRAMEBUILD and StateFrame.

The provider sees one ``inspect`` action.  Older ``operation`` payloads are
decoded only at this boundary so replay can remain readable without widening
the provider schema or creating a second navigation authority.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping


INSPECT_ACTION_NAME = "inspect"
NAV_CONTRACT_VERSION = "framebuild-navtools-v2"
_MAX_SCOPE = 256
_MAX_QUERY = 240
_MAX_QUESTION = 512
_MAX_DEPTH = 2
_HANDLE_RE = r"^(?:expand|frame-expand|navcont|navcursor):[A-Za-z0-9._:-]+"


class InspectContractError(ValueError):
    """Raised when a typed navigation request cannot be admitted."""


def _text(maximum: int, *, pattern: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"type": "string", "maxLength": maximum}
    if pattern is not None:
        result["pattern"] = pattern
    return result


def _object(properties: Mapping[str, object], required: tuple[str, ...]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": copy.deepcopy(dict(properties)),
        "required": list(required),
        "additionalProperties": False,
    }


def _target_schema() -> dict[str, object]:
    # Exactly one target identity is required.  A locator is deliberately
    # line-bounded; a path alone is not a source read contract.
    return {
        "oneOf": [
            _object({"entity_id": _text(256)}, ("entity_id",)),
            _object({"handle": _text(512, pattern=r"^expand:[A-Za-z0-9._:-]+$")}, ("handle",)),
            _object(
                {
                    "path": _text(1024),
                    "start_line": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
                    "end_line": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
                },
                ("path", "start_line", "end_line"),
            ),
        ]
    }


def _verbose_inspect_schema() -> dict[str, object]:
    """Return the closed six-variant provider schema."""

    scope = _text(_MAX_SCOPE)
    query = _text(_MAX_QUERY)
    filter_schema = _object(
        {
            "kind": {"type": "string", "enum": ["code", "document", "section", "dataset", "schema", "partition", "corpus"]},
            "path": _text(1024),
            "language": _text(64),
            "version": _text(128),
        },
        (),
    )
    variants = [
        _object(
            {
                "operation": {"const": "read"},
                "path": _text(1024),
                "mode": {"type": "string", "enum": ["signatures", "map", "span", "diff-since-last-seen", "full"]},
                "symbol_id": _text(512),
                "start_line": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
                "end_line": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
                "baseline_handle": _text(16_384, pattern=_HANDLE_RE),
                "cursor": _text(16_384, pattern=r"^frame-cursor:[0-9]+:[A-Za-z0-9]+$"),
            },
            ("operation", "path", "mode"),
        ),
        _object(
            {
                "operation": {"const": "expand"},
                "handle": _text(16_384, pattern=_HANDLE_RE),
                "mode": {"type": "string", "enum": ["span", "full"]},
                "cursor": _text(16_384, pattern=r"^frame-cursor:[0-9]+:[A-Za-z0-9]+$"),
            },
            ("operation", "handle", "mode"),
        ),
        _object(
            {
                "op": {"const": "orient"},
                "phase": {"type": "string", "enum": ["localize", "edit", "verify"]},
                "scope": scope,
                "focus": _target_schema(),
                "question": _text(_MAX_QUESTION),
            },
            ("op", "phase", "scope"),
        ),
        _object(
            {
                "op": {"const": "lookup"},
                "scope": scope,
                "query": query,
                "kind": {"type": "string", "enum": ["symbol", "document", "section", "dataset", "schema", "partition", "corpus"]},
                "match": {"type": "string", "enum": ["id", "exact_name"]},
            },
            ("op", "scope", "query"),
        ),
        _object(
            {
                "op": {"const": "search"},
                "scope": scope,
                "query": query,
                "mode": {"type": "string", "enum": ["literal", "pattern", "semantic"]},
                "filter": filter_schema,
                "cursor": _text(16_384, pattern=r"^navcursor:[A-Za-z0-9._:-]+$"),
                "page_size": {"type": "integer", "minimum": 1, "maximum": 24},
            },
            ("op", "scope", "query", "mode"),
        ),
        _object(
            {
                "op": {"const": "related"},
                "entity": _target_schema(),
                "relation": {"type": "string", "enum": ["callers", "callees", "references", "impact", "lineage"]},
                "depth": {"type": "integer", "minimum": 1, "maximum": _MAX_DEPTH},
            },
            ("op", "entity", "relation"),
        ),
        _object(
            {
                "op": {"const": "map"},
                "scope": scope,
                "depth": {"type": "integer", "minimum": 1, "maximum": _MAX_DEPTH},
                "cursor": _text(16_384, pattern=r"^navcursor:[A-Za-z0-9._:-]+$"),
                "page_size": {"type": "integer", "minimum": 1, "maximum": 24},
            },
            ("op", "scope"),
        ),
        {
            "oneOf": [
                _object(
                    {
                        "op": {"const": "read"},
                        "target": _target_schema(),
                        "view": {"type": "string", "enum": ["span", "outline", "signatures", "diff"]},
                        "since": _text(256),
                    },
                    ("op", "target", "view"),
                ),
                _object(
                    {
                        "op": {"const": "read"},
                        "continuation": _text(16_384, pattern=_HANDLE_RE),
                        "view": {"type": "string", "enum": ["span", "outline", "signatures", "diff"]},
                    },
                    ("op", "continuation", "view"),
                ),
            ]
        },
    ]
    # The aggregate property map keeps older contract census code able to see
    # every admitted field; each branch remains closed and owns exactness.
    aggregate: dict[str, object] = {}
    for variant in variants:
        branches = variant.get("oneOf", ())
        candidates = branches if branches else (variant,)
        for branch in candidates:
            if isinstance(branch, Mapping) and isinstance(branch.get("properties"), Mapping):
                for key in branch["properties"]:
                    # Root properties only provide a closed-name census.  The
                    # selected branch owns enums, consts, patterns, and
                    # required fields; merging those constraints here would
                    # make unrelated variants reject each other first.
                    aggregate.setdefault(str(key), {})
    return {
        "type": "object",
        "properties": aggregate,
        "additionalProperties": False,
        "oneOf": variants,
    }


def inspect_schema() -> dict[str, object]:
    """Return the branch-owned closed schema used on the provider wire."""
    return _compact_inspect_schema()


def _compact_inspect_schema() -> dict[str, object]:
    """Build one bounded provider schema without cross-branch constraints."""

    direct_read = _object(
        {
            "operation": {"const": "read"},
            "path": _text(1024),
            "mode": {"type": "string", "enum": ["signatures", "map", "span", "diff-since-last-seen", "full"]},
            "symbol_id": _text(512),
            "start_line": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
            "end_line": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
            "baseline_handle": _text(2_048, pattern=_HANDLE_RE),
            "cursor": _text(2_048, pattern=r"^frame-cursor:[0-9]+:[A-Za-z0-9]+$"),
        },
        ("operation", "path", "mode"),
    )
    direct_expand = _object(
        {
            "operation": {"const": "expand"},
            "handle": _text(2_048, pattern=_HANDLE_RE),
            "mode": {"type": "string", "enum": ["span", "full"]},
            "cursor": _text(2_048, pattern=r"^frame-cursor:[0-9]+:[A-Za-z0-9]+$"),
        },
        ("operation", "handle", "mode"),
    )
    legacy_read = {
        "type": "object",
        "properties": {"op": {"const": "read"}},
        "required": ["op", "view"],
        "oneOf": [
            {"required": ["target"], "not": {"required": ["continuation"]}},
            {"required": ["continuation"], "not": {"required": ["target"]}},
        ],
    }
    variants: list[dict[str, object]] = [
        direct_read,
        direct_expand,
        {"type": "object", "properties": {"op": {"const": "orient"}}, "required": ["op", "phase", "scope"]},
        {"type": "object", "properties": {"op": {"const": "lookup"}}, "required": ["op", "scope", "query"]},
        {"type": "object", "properties": {"op": {"const": "search"}}, "required": ["op", "scope", "query", "mode"]},
        {"type": "object", "properties": {"op": {"const": "related"}}, "required": ["op", "entity", "relation"]},
        {"type": "object", "properties": {"op": {"const": "map"}}, "required": ["op", "scope"]},
        legacy_read,
    ]
    names = {
        "operation", "path", "mode", "symbol_id", "start_line", "end_line", "baseline_handle", "handle", "cursor",
        "op", "scope", "phase", "focus", "question", "query", "kind", "match", "filter", "entity", "relation",
        "depth", "target", "view", "since", "continuation",
    }
    return {
        "type": "object",
        "properties": {name: {} for name in sorted(names)},
        "additionalProperties": False,
        "oneOf": variants,
    }


def _navtools_n2_schema() -> dict[str, object]:
    """Compact six-variant provider schema; legacy aliases stay host-only."""
    def text(maximum: int) -> dict[str, object]:
        return {"type": "string", "maxLength": maximum}

    target = {
        "type": "object",
        "properties": {
            "entity_id": text(256), "handle": text(512), "path": text(1024),
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "additionalProperties": False,
        "oneOf": [
            {"required": ["entity_id"]}, {"required": ["handle"]},
            {"required": ["path", "start_line", "end_line"]},
        ],
    }
    properties: dict[str, object] = {
        "op": text(16), "scope": text(_MAX_SCOPE), "phase": text(16),
        "focus": text(512), "question": text(_MAX_QUESTION), "query": text(_MAX_QUERY),
        "kind": text(32), "match": text(16), "mode": text(16),
        "filter": _object({"kind": text(32), "path": text(1024)}, ()),
        "entity": text(512), "relation": text(16),
        "depth": {"type": "integer", "minimum": 1, "maximum": _MAX_DEPTH},
        "target": target, "view": text(16), "since": text(256),
        "continuation": text(16_384), "cursor": text(16_384),
    }
    def branch(tag: str, required: tuple[str, ...]) -> dict[str, object]:
        return {"type": "object", "properties": {"op": {"const": tag}}, "required": list(required)}
    variants: list[dict[str, object]] = [
        branch("orient", ("op", "phase", "scope")),
        branch("lookup", ("op", "scope", "query")),
        branch("search", ("op", "scope", "query", "mode")),
        branch("related", ("op", "entity", "relation")),
        branch("map", ("op", "scope")),
        {
            "type": "object", "properties": {"op": {"const": "read"}},
            "required": ["op", "view"],
            "oneOf": [
                {"required": ["target"], "not": {"required": ["continuation"]}},
                {"required": ["continuation"], "not": {"required": ["target"]}},
            ],
        },
    ]
    return {"type": "object", "properties": properties, "additionalProperties": False, "oneOf": variants}


_LEGACY_OPERATIONS = {
    "orient": "orient",
    "lookup": "lookup",
    "search": "search",
    "grep": "search",
    "related": "related",
    "who_calls": "related",
    "references": "related",
    "map": "map",
    "list": "map",
    "ls": "map",
    "list_files": "map",
    "read": "read",
}


def _legacy_target(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    text = str(value or "").strip()
    if not text:
        text = "legacy:target"
    if "/" in text or text.endswith(('.py', '.go', '.rs', '.ts', '.js', '.java', '.php', '.vue')):
        return {"path": text, "start_line": 1, "end_line": 1}
    return {"entity_id": text}


def normalize_inspect_arguments(arguments: Mapping[str, object]) -> dict[str, object]:
    """Normalize a legacy inspect payload without relaxing the typed schema."""

    if not isinstance(arguments, Mapping):
        raise InspectContractError("inspect arguments must be an object")
    if "operation" not in arguments:
        return copy.deepcopy(dict(arguments))
    if "op" in arguments:
        raise InspectContractError("inspect cannot contain both op and legacy operation")
    operation = str(arguments.get("operation", "")).strip().casefold()
    if operation in {"read", "expand"}:
        direct_fields = {"path", "mode", "handle", "symbol_id", "start_line", "end_line", "baseline_handle", "cursor"}
        if any(field in arguments for field in direct_fields):
            return copy.deepcopy(dict(arguments))
    if operation == "read":
        result = {
            str(key): copy.deepcopy(value)
            for key, value in arguments.items()
            if key not in {
                "operation", "mode", "baseline_handle", "cursor", "path", "target",
                "symbol_id", "start_line", "end_line",
            }
        }
        result["op"] = "read"
        raw_target = arguments.get("target", arguments.get("path", arguments.get("symbol_id", "legacy:source")))
        result["target"] = _legacy_target(raw_target)
        if (
            isinstance(result["target"], Mapping)
            and isinstance(result["target"].get("path"), str)
            and type(arguments.get("start_line")) is int
            and type(arguments.get("end_line")) is int
        ):
            result["target"] = {
                "path": result["target"]["path"],
                "start_line": arguments["start_line"],
                "end_line": arguments["end_line"],
            }
        mode = str(arguments.get("mode", "span"))
        result["view"] = {
            "full": "span", "signatures": "signatures", "map": "outline",
            "span": "span", "diff-since-last-seen": "diff",
        }.get(mode, "span")
        if mode == "diff-since-last-seen" and isinstance(arguments.get("baseline_handle"), str):
            result["since"] = arguments["baseline_handle"]
        return result
    if operation == "expand":
        handle = arguments.get("handle")
        if not isinstance(handle, str) or not handle.strip():
            raise InspectContractError("legacy expand requires a handle")
        return {
            "op": "read",
            "continuation": handle,
            "view": "span" if arguments.get("mode") == "span" else "outline",
        }
    op = _LEGACY_OPERATIONS.get(operation)
    if op is None:
        raise InspectContractError(f"legacy inspect operation is unsupported: {operation or '<empty>'}")
    result = {str(key): copy.deepcopy(value) for key, value in arguments.items() if key != "operation"}
    result.pop("page", None)
    result.pop("page_size", None)
    result["op"] = op
    if op == "orient":
        result.setdefault("scope", "legacy:public")
        result.setdefault("phase", "localize")
    elif op == "lookup":
        result.setdefault("scope", "legacy:public")
        result.setdefault("query", result.get("symbol", "legacy"))
        result.pop("symbol", None)
        result.setdefault("kind", "symbol")
        result.setdefault("match", "exact_name")
    elif op == "search":
        result.setdefault("scope", "legacy:public")
        result.setdefault("query", result.get("pattern", "legacy"))
        result.pop("pattern", None)
        result.setdefault("mode", "literal")
    elif op == "related":
        target = _legacy_target(result.pop("entity", result.pop("symbol", result.get("query", "legacy:entity"))))
        result["entity"] = str(target.get("entity_id") or target.get("path") or target.get("handle") or "legacy:entity")
        result.setdefault("relation", "references" if operation == "references" else "callers")
        result.pop("query", None)
    elif op == "read":
        result["target"] = _legacy_target(result.pop("target", result.pop("path", "legacy:source")))
        result.setdefault("view", "span")
    elif op == "map":
        result.setdefault("scope", "legacy:public")
        result.pop("path", None)
        result.setdefault("depth", 1)
    return result


def validate_inspect_semantics(arguments: Mapping[str, object]) -> None:
    """Apply cross-field rules not expressible in the provider subset."""

    normalized = normalize_inspect_arguments(arguments)
    operation = normalized.get("operation")
    if operation == "read":
        mode = normalized.get("mode")
        if mode == "diff-since-last-seen" and not isinstance(normalized.get("baseline_handle"), str):
            raise InspectContractError("read diff-since-last-seen requires baseline_handle")
        if mode != "diff-since-last-seen" and "baseline_handle" in normalized:
            raise InspectContractError("baseline_handle is valid only for diff-since-last-seen")
        if normalized.get("start_line") is not None and normalized.get("end_line") is not None \
                and normalized["end_line"] < normalized["start_line"]:
            raise InspectContractError("read line range is reversed")
        return
    if operation == "expand":
        return
    op = normalized.get("op")
    allowed = {
        "orient": {"op", "phase", "scope", "focus", "question"},
        "lookup": {"op", "scope", "query", "kind", "match"},
        "search": {"op", "scope", "query", "mode", "filter", "cursor"},
        "related": {"op", "entity", "relation", "depth"},
        "map": {"op", "scope", "depth", "cursor"},
        "read": {"op", "target", "view", "since", "continuation"},
    }
    if op not in allowed:
        raise InspectContractError("inspect.op must select exactly one supported variant")
    unknown = sorted(set(normalized) - allowed[op])
    if unknown:
        raise InspectContractError(f"inspect.{op} has unknown field(s): {', '.join(unknown)}")
    required_text = {
        "orient": ("phase", "scope"),
        "lookup": ("scope", "query"),
        "search": ("scope", "query", "mode"),
        "related": ("entity", "relation"),
        "map": ("scope",),
        "read": ("view",),
    }
    for field in required_text[op]:
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise InspectContractError(f"inspect.{op}.{field} must be a non-empty string")
    allowed_values = {
        "phase": {"localize", "edit", "verify"},
        "kind": {"symbol", "document", "section", "dataset", "schema", "partition", "corpus"},
        "match": {"id", "exact_name"},
        "mode": {"literal", "pattern", "semantic"},
        "relation": {"callers", "callees", "references", "impact", "lineage"},
        "view": {"span", "outline", "signatures", "diff"},
    }
    for field, values in allowed_values.items():
        if field in normalized and normalized[field] not in values:
            raise InspectContractError(f"inspect.{op}.{field} is outside the supported enum")
    if op == "read" and "since" in normalized and normalized.get("view") != "diff":
        raise InspectContractError("read.since is valid only with the diff view")
    if op == "read" and "continuation" in normalized and "since" in normalized:
        raise InspectContractError("read continuation cannot carry a diff baseline")
    if op == "read" and "continuation" in normalized and (
        not isinstance(normalized["continuation"], str)
        or re.fullmatch(_HANDLE_RE, normalized["continuation"]) is None
    ):
        raise InspectContractError("read.continuation is not a signed navigation handle")
    if op in {"search", "map"} and "cursor" in normalized:
        cursor = normalized["cursor"]
        if not isinstance(cursor, str) or re.fullmatch(r"^navcursor:[A-Za-z0-9._:-]+$", cursor) is None:
            raise InspectContractError(f"inspect.{op}.cursor is not a signed navigation cursor")
    if op == "related" and normalized.get("depth", 1) > _MAX_DEPTH:
        raise InspectContractError("related depth exceeds the hard navigation bound")
    if op == "read" and "target" in normalized:
        target = normalized["target"]
        if not isinstance(target, Mapping):
            raise InspectContractError("read.target must be a typed target object")
        allowed_target_fields = {"entity_id", "handle", "path", "start_line", "end_line"}
        unknown_target = sorted(set(target) - allowed_target_fields)
        if unknown_target:
            raise InspectContractError("read.target has unknown field(s): " + ", ".join(unknown_target))
        target_variants = (
            set(target) == {"entity_id"},
            set(target) == {"handle"},
            set(target) == {"path", "start_line", "end_line"},
        )
        if sum(target_variants) != 1:
            raise InspectContractError("read.target must select exactly one entity, handle, or bounded locator")
        for field in ("entity_id", "handle", "path"):
            if field in target and (not isinstance(target[field], str) or not target[field].strip()):
                raise InspectContractError(f"read.target.{field} must be a non-empty string")
        if "end_line" in target:
            if not all(type(target.get(key)) is int for key in ("start_line", "end_line")):
                raise InspectContractError("read target line bounds must be integers")
            if any(target[key] > 1_000_000 for key in ("start_line", "end_line")):
                raise InspectContractError("read target line bounds exceed the hard limit")
            if target["end_line"] < target["start_line"]:
                raise InspectContractError("read target line range is reversed")


def inspect_variant(arguments: Mapping[str, object]) -> str:
    """Return the canonical variant after legacy normalization."""

    normalized = normalize_inspect_arguments(arguments)
    if normalized.get("operation") in {"read", "expand"}:
        return str(normalized["operation"])
    value = normalized.get("op")
    if not isinstance(value, str) or value not in {"orient", "lookup", "search", "related", "map", "read"}:
        raise InspectContractError("inspect.op must select exactly one supported variant")
    return value


__all__ = [
    "INSPECT_ACTION_NAME",
    "InspectContractError",
    "NAV_CONTRACT_VERSION",
    "inspect_schema",
    "inspect_variant",
    "normalize_inspect_arguments",
    "validate_inspect_semantics",
]
