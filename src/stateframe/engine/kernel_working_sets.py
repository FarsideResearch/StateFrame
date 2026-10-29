"""Opt-in named working sets as a version-aware selection view.

Working sets are deliberately not another memory store.  ``WorksetStore`` keeps
only host-owned group records and the currently retained expansion; resolution
is performed against the caller's current artifact/check/receipt indexes every
time a group is expanded.  The model can supply labels and typed references,
but it cannot write summaries, status, freshness, or authority claims.

The runtime mounts this module behind ``STATEFRAME_WORKING_SETS=on``.  The
public classes are also useful to offline probes and to the frame-gears lane,
which can project ``working_set_manifest`` without importing the runtime.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence


MODE_ENV = "STATEFRAME_WORKING_SETS"
SCHEMA = "stateframe.workset/1"
VIEW_SCHEMA = "stateframe.workset-view/1"
EXPANSION_SCHEMA = "stateframe.expansion/1"
DIRECTORY_SCHEMA = "stateframe.workset-directory/1"
TOOL_NAME = "working_set"
OPS = ("create", "extend", "remove", "merge", "retire", "expand")
REF_KINDS = frozenset({
    "file", "symbol", "requirement", "check", "check_run", "test", "receipt",
})
IMMUTABLE_KINDS = frozenset({"check_run", "test", "receipt"})
MAX_GROUPS = 64
MAX_MEMBERS_SOFT = 16
MAX_MEMBERS = 64
MAX_OPERATION_RESULTS = 128
MAX_VISIBLE = 6
MAX_LABEL = 120
MAX_PAGE_TOKENS = 1800
MAX_PAGE_ITEMS = 64
MAX_ID = 96
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@-]{0,95}$")
_PATH_FORBIDDEN = frozenset({"private", "hidden_tests", "near_miss", "gold", "oracle"})
_MODEL_TEXT_FIELDS = frozenset({
    "summary", "status", "description", "details", "note", "notes", "reason",
    "message", "comment", "text", "freshness", "authority", "applicability",
})
_ARGUMENTS = frozenset({
    "op", "label", "group_id", "target_id", "source_id", "refs", "members",
    "member_refs", "idempotency_key", "cursor", "snapshot_id",
    "budget_tokens", "max_items", "keep_resident",
})


class WorkingSetError(ValueError):
    """A typed, non-mutating rejection of a working-set operation."""

    def __init__(self, reason: str, *, code: str = "INVALID_ARGUMENTS"):
        self.reason = str(reason)
        self.code = str(code)
        super().__init__(self.reason)


def mode(value: object = None) -> str:
    raw = os.environ.get(MODE_ENV, "off") if value is None else value
    if not isinstance(raw, str) or raw.strip().lower() not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return raw.strip().lower()


def enabled(value: object = None) -> bool:
    return mode(value) == "on"


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if _sequence(value):
        return list(value)
    return [value]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def token_count(value: object) -> int:
    """Use the repository's deterministic serialized-bytes/4 estimate."""
    if value in (None, "", [], {}):
        return 0
    raw = value.encode("utf-8") if isinstance(value, str) else _canonical(value)
    return int(math.ceil(len(raw) / 4))


def _string(value: object, name: str, limit: int = MAX_ID, *, required: bool = True) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        suffix = " and nonempty" if required else ""
        raise WorkingSetError(f"{name} must be a string of at most {limit} characters{suffix}")
    return value


def _identity(value: object, name: str) -> str:
    result = _string(value, name)
    if not _ID_RE.fullmatch(result):
        raise WorkingSetError(f"{name} has an invalid identity")
    return result


def _path(value: object) -> str:
    raw = _string(value, "path", MAX_ID).replace("\\", "/")
    parts = tuple(part for part in raw.split("/") if part)
    if raw.startswith("/") or ".." in parts or any(part.casefold() in _PATH_FORBIDDEN for part in parts):
        raise WorkingSetError("reference path is outside the public workspace", code="PATH_DENIED")
    return "/".join(parts)


def _reject_model_text(value: Mapping[str, object], *, where: str) -> None:
    found = _MODEL_TEXT_FIELDS.intersection(value)
    if found:
        raise WorkingSetError(
            f"{where} contains model text field {sorted(found)[0]}; label is the only model text",
            code="MODEL_TEXT_NOT_ALLOWED")


def _reject_model_text_tree(value: object, *, where: str) -> None:
    """Reject prose-bearing keys even when host metadata is nested."""
    if isinstance(value, Mapping):
        _reject_model_text(value, where=where)
        for key, item in value.items():
            _reject_model_text_tree(item, where=f"{where}.{key}")
    elif _sequence(value):
        for item in value:
            _reject_model_text_tree(item, where=where)


def normalize_ref(value: object) -> dict[str, object]:
    """Normalize one typed identity while rejecting nested groups and prose."""
    if isinstance(value, str):
        raw = value.strip()
        prefix, separator, ident = raw.partition(":")
        if not separator or prefix not in REF_KINDS:
            raise WorkingSetError("members must be typed references", code="UNTYPED_REFERENCE")
        if prefix == "file":
            return {"kind": prefix, "path": _path(ident)}
        if prefix == "symbol":
            symbol_separator = "::" if "::" in ident else ":"
            path, symbol_marker, symbol = ident.rpartition(symbol_separator)
            if not symbol_marker:
                raise WorkingSetError("symbol references need path and symbol")
            return {"kind": prefix, "path": _path(path),
                    "symbol": _identity(symbol, "symbol")}
        return {"kind": prefix, "id": _identity(ident, "reference id")}
    if not isinstance(value, Mapping):
        raise WorkingSetError("member reference must be an object")
    _reject_model_text(value, where="member reference")
    allowed = {"kind", "type", "id", "path", "symbol", "requirement_id", "check_id",
               "test_id", "receipt_id", "revision", "line", "start_line", "end_line"}
    unknown = set(value) - allowed
    if unknown:
        if any(str(item).casefold() in _MODEL_TEXT_FIELDS for item in unknown):
            raise WorkingSetError("member reference contains model text field",
                                  code="MODEL_TEXT_NOT_ALLOWED")
        raise WorkingSetError("unknown member reference field: " + sorted(map(str, unknown))[0])
    kind = value.get("kind", value.get("type"))
    kind = _string(kind, "reference kind", 32).strip().lower()
    if kind not in REF_KINDS:
        raise WorkingSetError("workset references may not point to another workset",
                              code="NESTED_WORKSET")
    result: dict[str, object] = {"kind": kind}
    if kind in {"file", "symbol"}:
        result["path"] = _path(value.get("path", value.get("id")))
        if kind == "symbol":
            result["symbol"] = _identity(value.get("symbol"), "symbol")
    elif kind == "requirement":
        result["id"] = _identity(value.get("id", value.get("requirement_id")), "requirement id")
    elif kind == "check":
        result["id"] = _identity(value.get("id", value.get("check_id")), "check id")
    elif kind == "check_run":
        result["id"] = _identity(value.get("id"), "check-run id")
    elif kind == "test":
        result["id"] = _identity(value.get("id", value.get("test_id")), "test id")
    elif kind == "receipt":
        result["id"] = _identity(value.get("id", value.get("receipt_id")), "receipt id")
    if value.get("revision") is not None:
        result["revision"] = _string(value["revision"], "reference revision", 200)
    for key in ("line", "start_line", "end_line"):
        if key in value:
            number = value[key]
            if type(number) is not int or number < 1:
                raise WorkingSetError(f"{key} must be a positive integer")
            result[key] = number
    return result


def normalize_member(value: object) -> dict[str, object]:
    if isinstance(value, Mapping) and "ref" in value:
        _reject_model_text(value, where="member")
        unknown = set(value) - {"ref", "binding"}
        if unknown:
            raise WorkingSetError("unknown member field: " + sorted(map(str, unknown))[0])
        ref = normalize_ref(value["ref"])
        binding = value.get("binding")
    else:
        ref = normalize_ref(value)
        binding = None
    if binding is None:
        binding = "immutable" if ref["kind"] in IMMUTABLE_KINDS else "live"
    if binding not in {"live", "immutable"}:
        raise WorkingSetError("member binding must be live or immutable")
    return {"ref": ref, "binding": binding}


def _members(value: object, *, allow_empty: bool = True) -> list[dict[str, object]]:
    if value is None:
        if allow_empty:
            return []
        raise WorkingSetError("members are required")
    if not isinstance(value, list):
        raise WorkingSetError("members must be a list")
    if len(value) > MAX_MEMBERS:
        raise WorkingSetError(f"a workset may contain at most {MAX_MEMBERS} members",
                              code="MEMBER_LIMIT")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in value:
        member = normalize_member(item)
        key = _canonical(member["ref"]).decode("utf-8")
        if key in seen:
            continue
        seen.add(key)
        result.append(member)
    return result


def validate_record(record: Mapping[str, object]) -> dict[str, object]:
    """Validate and copy the sole canonical group-record shape."""
    if not isinstance(record, Mapping):
        raise WorkingSetError("workset record must be an object")
    required = {"schema", "id", "label", "scope", "created_by", "revision", "lifecycle", "members"}
    if set(record) != required:
        unknown = set(record) - required
        missing = required - set(record)
        if unknown:
            raise WorkingSetError("unknown workset record field: " + sorted(map(str, unknown))[0])
        raise WorkingSetError("workset record is missing " + sorted(missing)[0])
    if record["schema"] != SCHEMA:
        raise WorkingSetError("unsupported workset schema")
    ident = _identity(record["id"], "workset id")
    label = _string(record["label"], "workset label", MAX_LABEL)
    if any(ord(char) < 32 for char in label):
        raise WorkingSetError("workset label contains control text")
    scope = record["scope"]
    if not isinstance(scope, Mapping):
        raise WorkingSetError("workset scope must be host-owned metadata")
    _reject_model_text_tree(scope, where="workset scope")
    scope = copy.deepcopy(dict(scope))
    if any(not isinstance(key, str) or key in _MODEL_TEXT_FIELDS for key in scope):
        raise WorkingSetError("workset scope contains model text")
    created_by = _string(record["created_by"], "created_by", 64)
    revision = record["revision"]
    if type(revision) is not int or revision < 1:
        raise WorkingSetError("workset revision must be a positive integer")
    lifecycle = record["lifecycle"]
    if lifecycle not in {"active", "retired"}:
        raise WorkingSetError("workset lifecycle must be active or retired")
    members = _members(record["members"])
    return {"schema": SCHEMA, "id": ident, "label": label, "scope": scope,
            "created_by": created_by, "revision": revision,
            "lifecycle": lifecycle, "members": members}


def _scope(context: Mapping[str, object], extra: object = None) -> dict[str, object]:
    result: dict[str, object] = {}
    task_id = context.get("task_id")
    if isinstance(task_id, str) and task_id:
        result["task_id"] = task_id
    if isinstance(extra, Mapping):
        for key, value in extra.items():
            if key in _MODEL_TEXT_FIELDS or not isinstance(key, str):
                continue
            if isinstance(value, (str, int, bool)) or value is None:
                result[key] = copy.deepcopy(value)
    return result


def _current_revisions(context: Mapping[str, object]) -> dict[str, str]:
    value = context.get("current_revisions", context.get("artifact_revisions", {}))
    if not isinstance(value, Mapping):
        return {}
    return {str(key).replace("\\", "/").lstrip("./"): str(item)
            for key, item in value.items() if str(key)}


def _sources(context: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    value = context.get("sources", context.get("source_by_path", {}))
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, str):
                result[str(key).replace("\\", "/").lstrip("./")] = item
    return result


def _mapping_for(context: Mapping[str, object], old_path: str, old_symbol: str = "") -> tuple[str, str, bool]:
    mapping = context.get("rename_map", context.get("mappings", {}))
    if not isinstance(mapping, Mapping):
        return old_path, old_symbol, False
    candidates = []
    if old_symbol:
        candidates.extend([f"{old_path}:{old_symbol}", f"{old_path}::{old_symbol}"])
    candidates.append(old_path)
    for key in candidates:
        if key not in mapping:
            continue
        value = mapping[key]
        if isinstance(value, str):
            return value.replace("\\", "/").lstrip("./"), old_symbol, True
        if isinstance(value, Mapping):
            path = value.get("path", value.get("new_path", old_path))
            symbol = value.get("symbol", value.get("new_symbol", old_symbol))
            if isinstance(path, str) and isinstance(symbol, str):
                return path.replace("\\", "/").lstrip("./"), symbol, True
    return old_path, old_symbol, False


def _archive_source(context: Mapping[str, object], path: str, revision: str | None,
                    *, immutable: bool = False) -> tuple[str | None, str | None]:
    archive = context.get("archive")
    if archive is None:
        return None, None
    revisions = getattr(archive, "revisions", {})
    blobs = getattr(archive, "blobs", {})
    if not isinstance(revisions, Mapping) or not isinstance(blobs, Mapping):
        return None, None
    selected = None
    if immutable and revision:
        selected = next((item for item in revisions.values()
                         if isinstance(item, Mapping) and item.get("path") == path
                         and item.get("revision") == revision), None)
    if selected is None:
        latest = getattr(archive, "latest", {})
        reference = latest.get(path) if isinstance(latest, Mapping) else None
        selected = revisions.get(reference) if reference else None
    if not isinstance(selected, Mapping):
        return None, None
    raw = blobs.get(selected.get("revision"))
    if not isinstance(raw, bytes):
        return None, None
    return raw.decode("utf-8", errors="replace"), str(selected.get("revision") or "")


def _symbol_text(source: str, symbol: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    lines = source.splitlines(keepends=True)
    target = symbol.split(".")[-1]
    matches = [node for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and node.name == target]
    if len(matches) != 1:
        return None
    node = matches[0]
    start_line = min([node.lineno] + [item.lineno for item in node.decorator_list])
    end_line = getattr(node, "end_lineno", node.lineno)
    return "".join(lines[start_line - 1:end_line])


def _rows(value: object, *keys: str) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if _sequence(candidate):
                return [item for item in candidate if isinstance(item, Mapping)]
        return [value]
    if _sequence(value):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _id(row: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value):
            return str(value)
    return ""


def _binding_is_current(row: Mapping[str, object], context: Mapping[str, object]) -> bool:
    current = _current_revisions(context)
    revision = row.get("revision", row.get("file_revision"))
    if isinstance(revision, str) and revision:
        path = row.get("path", row.get("target_path"))
        if isinstance(path, str) and path in current:
            return current[path] == revision
        if current:
            return revision in set(current.values())
    before = row.get("after_binding", row.get("binding"))
    if isinstance(before, Mapping):
        observed = before.get("product_files", before)
        if isinstance(observed, Mapping):
            return all(current.get(str(path)) == str(value) for path, value in observed.items())
    return True


def resolve_reference(member: Mapping[str, object], context: Mapping[str, object] | None = None) -> dict[str, object]:
    """Resolve one live/immutable member through the host artifact indexes."""
    context = context if isinstance(context, Mapping) else {}
    normalized = normalize_ref(member.get("ref", member))
    binding = member.get("binding") or ("immutable" if normalized["kind"] in IMMUTABLE_KINDS else "live")
    custom = context.get("resolver")
    if callable(custom):
        result = custom(copy.deepcopy(normalized), copy.deepcopy(dict(context)))
        if isinstance(result, Mapping):
            item = copy.deepcopy(dict(result))
            item.setdefault("ref", normalized)
            item.setdefault("binding", binding)
            item.setdefault("resolution", "resolved")
            item.setdefault("applicability", "current")
            return item

    kind = str(normalized["kind"])
    if kind in {"file", "symbol"}:
        old_path = str(normalized["path"])
        old_symbol = str(normalized.get("symbol", ""))
        path, symbol, mapped = _mapping_for(context, old_path, old_symbol)
        revisions = _current_revisions(context)
        if path not in revisions:
            return {"ref": normalized, "binding": binding, "resolution": "tombstone",
                    "applicability": "unknown", "tombstone": True,
                    "path": path, **({"symbol": symbol} if symbol else {})}
        source = _sources(context).get(path)
        source_revision = revisions.get(path)
        if binding == "immutable" and normalized.get("revision"):
            historical, historical_revision = _archive_source(
                context, path, str(normalized.get("revision")), immutable=True)
            if historical is not None:
                source, source_revision = historical, historical_revision
        if source is None:
            source, archive_revision = _archive_source(
                context, path, source_revision, immutable=binding == "immutable")
            source_revision = source_revision or archive_revision
        if source is None:
            return {"ref": normalized, "binding": binding, "resolution": "unresolved",
                    "applicability": "unknown", "path": path,
                    "revision": source_revision, **({"symbol": symbol} if symbol else {})}
        content = _symbol_text(source, symbol) if kind == "symbol" else source
        if kind == "symbol" and content is None:
            return {"ref": normalized, "binding": binding, "resolution": "tombstone",
                    "applicability": "unknown", "tombstone": True, "path": path,
                    "revision": source_revision, "symbol": symbol}
        historical = bool(binding == "immutable" and normalized.get("revision")
                          and str(normalized.get("revision")) != str(revisions.get(path)))
        return {"ref": normalized, "binding": binding, "resolution": "mapped" if mapped else "resolved",
                "applicability": "historical" if historical else "current",
                "path": path, "revision": source_revision,
                **({"symbol": symbol} if symbol else {}),
                "content": content, "content_hash": _digest(content)}

    ledger = context.get("requirements_ledger", context.get("ledger", {}))
    if kind == "requirement":
        row = next((item for item in _rows(ledger, "requirements")
                    if _id(item, "id", "requirement_id") == normalized["id"]), None)
        if row is None:
            return {"ref": normalized, "binding": binding, "resolution": "tombstone",
                    "applicability": "unknown", "tombstone": True}
        return {"ref": normalized, "binding": binding, "resolution": "resolved",
                "applicability": "current", "requirement": copy.deepcopy(dict(row))}
    if kind == "check":
        rows = [*_rows(ledger, "checks"), *_rows(context.get("checks"), "checks")]
        row = next((item for item in rows if _id(item, "id", "check_id") == normalized["id"]), None)
        if row is None:
            return {"ref": normalized, "binding": binding, "resolution": "tombstone",
                    "applicability": "unknown", "tombstone": True}
        current = _binding_is_current(row, context)
        return {"ref": normalized, "binding": binding, "resolution": "resolved",
                "applicability": "current" if current else "stale", "check": copy.deepcopy(dict(row))}
    if kind == "check_run":
        rows = [*_rows(context.get("check_runs"), "check_runs"),
                *_rows(context.get("test_receipts"), "test_receipts")]
        row = next((item for item in rows if _id(item, "id", "receipt_id", "check_run_id") == normalized["id"]), None)
        if row is None:
            return {"ref": normalized, "binding": "immutable", "resolution": "tombstone",
                    "applicability": "unknown", "tombstone": True}
        revision = row.get("revision", row.get("product_revision", row.get("artifact_revision")))
        current = _binding_is_current(row, context)
        return {"ref": normalized, "binding": "immutable", "resolution": "resolved",
                "applicability": "current" if current else "stale",
                "observed_revision": revision, "check_run": copy.deepcopy(dict(row))}
    if kind == "test":
        rows = [*_rows(context.get("tests"), "tests"),
                *_rows(context.get("test_receipts"), "test_receipts")]
        row = next((item for item in rows if _id(item, "id", "test_id", "receipt_id") == normalized["id"]), None)
        if row is None:
            return {"ref": normalized, "binding": "immutable", "resolution": "tombstone",
                    "applicability": "unknown", "tombstone": True}
        return {"ref": normalized, "binding": "immutable", "resolution": "resolved",
                "applicability": "current" if _binding_is_current(row, context) else "stale",
                "test": copy.deepcopy(dict(row))}
    if kind == "receipt":
        rows = _rows(context.get("receipts"), "receipts")
        row = next((item for item in rows if _id(item, "receipt_id", "id") == normalized["id"]), None)
        if row is None:
            return {"ref": normalized, "binding": "immutable", "resolution": "tombstone",
                    "applicability": "unknown", "tombstone": True}
        return {"ref": normalized, "binding": "immutable", "resolution": "resolved",
                "applicability": "current" if _binding_is_current(row, context) else "stale",
                "receipt": copy.deepcopy(dict(row))}
    raise WorkingSetError("unsupported reference kind")


def snapshot_id(context: Mapping[str, object] | None = None) -> str:
    """Return a revision snapshot that excludes mutable source prose."""
    context = context if isinstance(context, Mapping) else {}
    revisions = _current_revisions(context)
    ledger = context.get("requirements_ledger", context.get("ledger", {}))
    ledger_revision = ledger.get("current_revision") if isinstance(ledger, Mapping) else None
    mapping = context.get("rename_map", context.get("mappings", {}))
    runs = []
    for row in _rows(context.get("check_runs"), "check_runs") + _rows(context.get("test_receipts"), "test_receipts"):
        runs.append(copy.deepcopy(dict(row)))
    ledger_facts = {}
    if isinstance(ledger, Mapping):
        for key in ("requirements", "checks"):
            value = ledger.get(key)
            if value not in (None, [], {}):
                ledger_facts[key] = value
    live_facts = {}
    for name, keys in (
            ("checks", ("checks",)), ("tests", ("tests",)),
            ("receipts", ("receipts",))):
        facts = []
        for row in _rows(context.get(name), *keys):
            facts.append(copy.deepcopy(dict(row)))
        if facts:
            live_facts[name] = facts
    return _fact_digest({"revisions": revisions, "ledger_revision": ledger_revision,
                    "ledger_facts": ledger_facts,
                    "live_facts": live_facts,
                    "rename_map": mapping if isinstance(mapping, Mapping) else {},
                    "immutable_runs": runs})


def _fact_digest(value: object) -> str:
    """Digest arbitrary host facts; non-JSON leaves are canonicalized by repr."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     default=repr).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def group_snapshot_id(group: Mapping[str, object], context: Mapping[str, object] | None = None) -> str:
    """Fingerprint exactly the observation an expansion of ``group`` returns.

    Scoped to the group's referenced facts: unrelated host state (delivery
    receipts, other files) cannot evict a resident expansion, while any change
    to a referenced record -- including binding/non-whitelisted fields -- does,
    because the complete resolved observation is hashed.
    """
    context = context if isinstance(context, Mapping) else {}
    members = group.get("members", []) if isinstance(group, Mapping) else []
    observations = []
    for member in members if isinstance(members, list) else []:
        try:
            observations.append(resolve_reference(member, context))
        except WorkingSetError as error:
            observations.append({"ref": member.get("ref"), "error": error.code})
    return _fact_digest({"group_id": group.get("id"), "observations": observations})


def _coverage(items: Sequence[Mapping[str, object]]) -> dict[str, int]:
    result = {key: 0 for key in ("total", "resolved", "unresolved", "tombstones",
                                 "current", "stale", "historical", "unknown")}
    result["total"] = len(items)
    for item in items:
        resolution = str(item.get("resolution", "unresolved"))
        applicability = str(item.get("applicability", "unknown"))
        if resolution in {"resolved", "mapped"}:
            result["resolved"] += 1
        if resolution == "tombstone":
            result["tombstones"] += 1
        elif resolution not in {"resolved", "mapped"}:
            result["unresolved"] += 1
        result[applicability if applicability in result else "unknown"] += 1
    return result


def _compact_item(item: Mapping[str, object]) -> dict[str, object]:
    """Keep expansion host facts bounded without copying arbitrary context."""
    allowed = {"ref", "binding", "resolution", "applicability", "tombstone", "path",
               "symbol", "revision", "observed_revision", "content", "content_hash",
               "requirement", "check", "check_run", "test", "receipt"}
    result = {key: copy.deepcopy(value) for key, value in item.items() if key in allowed}
    return result


class WorkingSetStore:
    """Durable group records plus one retained expansion."""

    def __init__(self, *, emit: Callable[..., object] | None = None,
                 task_id: str = "") -> None:
        self.emit = emit
        self.task_id = str(task_id or "")
        self.groups: dict[str, dict[str, object]] = {}
        self.redirects: dict[str, str] = {}
        self.active_id: str | None = None
        self.active_expansion: dict[str, object] | None = None
        self.last_eviction: dict[str, object] | None = None
        self._next_group = 1
        self._operation_results: dict[str, dict[str, object]] = {}

    def _event(self, kind: str, **fields: object) -> None:
        if callable(self.emit):
            self.emit(kind, **copy.deepcopy(fields))

    def _new_id(self) -> str:
        while True:
            candidate = f"g{self._next_group}"
            self._next_group += 1
            if candidate not in self.groups:
                return candidate

    def resolve_id(self, group_id: object) -> str:
        ident = _identity(group_id, "group id")
        seen = set()
        while ident in self.redirects:
            if ident in seen:
                raise WorkingSetError("workset redirect cycle")
            seen.add(ident)
            ident = self.redirects[ident]
        return ident

    def resolve_group(self, group_id: object) -> dict[str, object]:
        ident = self.resolve_id(group_id)
        group = self.groups.get(ident)
        if group is None:
            raise WorkingSetError("unknown workset group", code="NOT_FOUND")
        return copy.deepcopy(group)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "stateframe.worksets/1",
            "groups": [copy.deepcopy(row) for row in self.groups.values()],
            "redirects": copy.deepcopy(self.redirects),
            "active_id": self.active_id,
            "active_expansion": copy.deepcopy(self.active_expansion),
            "last_eviction": copy.deepcopy(self.last_eviction),
            # The bounded replay ledger makes explicit idempotency keys remain
            # idempotent after a continuation journal is restored.  It is
            # host state, not a second model-authored record store.
            "operation_results": copy.deepcopy(self._operation_results),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping) or state.get("schema") != "stateframe.worksets/1":
            raise WorkingSetError("invalid working-set store snapshot")
        groups = state.get("groups")
        if not isinstance(groups, list) or len(groups) > MAX_GROUPS:
            raise WorkingSetError("invalid working-set group collection")
        self.groups = {}
        for raw in groups:
            row = validate_record(raw)
            if row["id"] in self.groups:
                raise WorkingSetError("duplicate workset id")
            self.groups[str(row["id"])] = row
        redirects = state.get("redirects", {})
        if not isinstance(redirects, Mapping):
            raise WorkingSetError("invalid workset redirects")
        self.redirects = {str(key): str(value) for key, value in redirects.items()}
        self.active_id = str(state["active_id"]) if state.get("active_id") is not None else None
        if self.active_id is not None and self.active_id not in self.groups:
            raise WorkingSetError("active workset is unknown")
        self.active_expansion = copy.deepcopy(state.get("active_expansion"))
        self.last_eviction = copy.deepcopy(state.get("last_eviction"))
        operation_results = state.get("operation_results", {})
        if not isinstance(operation_results, Mapping) or len(operation_results) > MAX_OPERATION_RESULTS:
            raise WorkingSetError("invalid bounded workset operation ledger")
        self._operation_results = {
            str(key): copy.deepcopy(dict(value))
            for key, value in operation_results.items()
            if isinstance(key, str) and isinstance(value, Mapping)
        }
        if len(self._operation_results) != len(operation_results):
            raise WorkingSetError("invalid workset operation result")
        numeric = [int(row[1:]) for row in self.groups if row.startswith("g") and row[1:].isdigit()]
        self._next_group = max(numeric, default=0) + 1

    def replay_event(self, event: Mapping[str, object]) -> None:
        if not isinstance(event, Mapping):
            return
        state = event.get("state")
        if isinstance(state, Mapping) and state.get("schema") == "stateframe.worksets/1":
            self.restore(state)
        if event.get("kind") == "working_set_evicted":
            eviction = event.get("eviction")
            if isinstance(eviction, Mapping):
                self.last_eviction = copy.deepcopy(dict(eviction))
                if self.active_expansion is not None:
                    self.active_expansion = {
                        **self.active_expansion, "resident": False,
                        "eviction": copy.deepcopy(dict(eviction)),
                    }

    def _cache_key(self, op: str, args: Mapping[str, object]) -> str:
        supplied = args.get("idempotency_key")
        if isinstance(supplied, str) and supplied.strip():
            value = {"idempotency_key": supplied.strip()}
            if op == "expand":
                value["context_snapshot"] = args.get(
                    "_context_snapshot", args.get("snapshot_id"))
                value["group_revision"] = args.get("_group_revision")
        else:
            value = {key: args.get(key) for key in sorted(args)}
        return _digest({"op": op, "args": value})

    def _save(self, *, op: str, args: Mapping[str, object], result: dict[str, object]) -> dict[str, object]:
        key = self._cache_key(op, args)
        if key in self._operation_results:
            self._operation_results.pop(key, None)
        self._operation_results[key] = copy.deepcopy(result)
        while len(self._operation_results) > MAX_OPERATION_RESULTS:
            self._operation_results.pop(next(iter(self._operation_results)))
        self._event("working_set_updated", operation=op, state=self.snapshot(),
                    idempotency_key=key)
        return result

    def _cached(self, op: str, args: Mapping[str, object]) -> dict[str, object] | None:
        # Replay only honours an explicit idempotency_key; without one every
        # call applies against current state (extend/remove/extend must work).
        # Read-only expand keeps request-keyed replay (its key already carries
        # the group-scoped snapshot and group revision).
        supplied = args.get("idempotency_key")
        if op != "expand" and not (isinstance(supplied, str) and supplied.strip()):
            return None
        value = self._operation_results.get(self._cache_key(op, args))
        if value is None:
            return None
        result = copy.deepcopy(value)
        result["idempotent"] = True
        return result

    def _mutable_group(self, group_id: object) -> tuple[str, dict[str, object]]:
        ident = self.resolve_id(group_id)
        group = self.groups.get(ident)
        if group is None:
            raise WorkingSetError("unknown workset group", code="NOT_FOUND")
        if group.get("created_by") == "host":
            raise WorkingSetError(
                "host-generated working-set groups are read-only",
                code="HOST_GROUP_READ_ONLY",
            )
        return ident, copy.deepcopy(group)

    def create(self, args: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
        # Replay is checked before capacity and member validation so a retry
        # remains idempotent even after the bounded group limit is reached.
        cached = self._cached("create", args)
        if cached is not None:
            return cached
        refs = args.get("refs", args.get("members", []))
        members = _members(refs)
        if len(self.groups) >= MAX_GROUPS:
            raise WorkingSetError("workset capacity is 64 groups", code="GROUP_LIMIT")
        label = _string(args.get("label"), "workset label", MAX_LABEL)
        ident = self._new_id()
        group = validate_record({
            "schema": SCHEMA, "id": ident, "label": label,
            "scope": _scope(context),
            "created_by": "model", "revision": 1, "lifecycle": "active", "members": members,
        })
        self.groups[ident] = group
        result = {"schema": "stateframe.workset-result/1", "status": "APPLIED",
                  "operation": "create", "group": copy.deepcopy(group), "idempotent": False}
        return self._save(op="create", args=args, result=result)

    def _member_args(self, args: Mapping[str, object]) -> list[dict[str, object]]:
        raw = args.get("refs", args.get("members", args.get("member_refs", [])))
        return _members(raw)

    def extend(self, args: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
        del context
        cached = self._cached("extend", args)
        if cached is not None:
            return cached
        ident, group = self._mutable_group(args.get("group_id"))
        additions = self._member_args(args)
        existing = {_canonical(item["ref"]).decode("utf-8") for item in group["members"]}
        before = len(group["members"])
        for member in additions:
            key = _canonical(member["ref"]).decode("utf-8")
            if key not in existing:
                group["members"].append(member)
                existing.add(key)
        if len(group["members"]) > MAX_MEMBERS:
            raise WorkingSetError("workset hard member limit is 64", code="MEMBER_LIMIT")
        changed = len(group["members"]) != before
        group["revision"] += int(changed)
        self.groups[ident] = validate_record(group)
        evicted = self._evict("group_revision_changed", group_id=ident) if changed and self.active_id == ident else None
        result = {"schema": "stateframe.workset-result/1", "status": "APPLIED",
                  "operation": "extend", "group": copy.deepcopy(self.groups[ident]),
                  "soft_member_limit": MAX_MEMBERS_SOFT, "idempotent": not changed}
        if evicted is not None:
            result["evicted"] = evicted
        return self._save(op="extend", args=args, result=result)

    def remove(self, args: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
        del context
        cached = self._cached("remove", args)
        if cached is not None:
            return cached
        ident, group = self._mutable_group(args.get("group_id"))
        remove_members = self._member_args(args)
        remove_keys = {_canonical(item["ref"]).decode("utf-8") for item in remove_members}
        before = len(group["members"])
        group["members"] = [item for item in group["members"]
                             if _canonical(item["ref"]).decode("utf-8") not in remove_keys]
        changed = len(group["members"]) != before
        group["revision"] += int(changed)
        self.groups[ident] = validate_record(group)
        evicted = self._evict("group_revision_changed", group_id=ident) if changed and self.active_id == ident else None
        result = {"schema": "stateframe.workset-result/1", "status": "APPLIED",
                  "operation": "remove", "group": copy.deepcopy(self.groups[ident]),
                  "idempotent": not changed}
        if evicted is not None:
            result["evicted"] = evicted
        return self._save(op="remove", args=args, result=result)

    def merge(self, args: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
        del context
        cached = self._cached("merge", args)
        if cached is not None:
            return cached
        source_id = self.resolve_id(args.get("group_id", args.get("source_id")))
        target_id = self.resolve_id(args.get("target_id"))
        if self.groups.get(source_id, {}).get("created_by") == "host" \
                or self.groups.get(target_id, {}).get("created_by") == "host":
            raise WorkingSetError(
                "host-generated working-set groups are read-only",
                code="HOST_GROUP_READ_ONLY",
            )
        if source_id == target_id:
            return {"schema": "stateframe.workset-result/1", "status": "APPLIED",
                    "operation": "merge", "group": copy.deepcopy(self.groups[target_id]),
                    "redirect": target_id, "idempotent": True}
        source = copy.deepcopy(self.groups[source_id])
        target = copy.deepcopy(self.groups[target_id])
        existing = {_canonical(item["ref"]).decode("utf-8") for item in target["members"]}
        for member in source["members"]:
            key = _canonical(member["ref"]).decode("utf-8")
            if key not in existing:
                target["members"].append(member)
                existing.add(key)
        if len(target["members"]) > MAX_MEMBERS:
            raise WorkingSetError("merged workset exceeds the hard member limit", code="MEMBER_LIMIT")
        target["revision"] += 1
        source["revision"] += 1
        source["lifecycle"] = "retired"
        self.groups[target_id] = validate_record(target)
        self.groups[source_id] = validate_record(source)
        self.redirects[source_id] = target_id
        evicted = None
        if self.active_id == source_id:
            evicted = self._evict("merge_redirect", group_id=source_id)
            self.active_id = target_id
            self.active_expansion = None
        elif self.active_id == target_id:
            evicted = self._evict("group_revision_changed", group_id=target_id)
        result = {"schema": "stateframe.workset-result/1", "status": "APPLIED",
                  "operation": "merge", "group": copy.deepcopy(target),
                  "redirect": target_id, "idempotent": False}
        if evicted is not None:
            result["evicted"] = evicted
        return self._save(op="merge", args=args, result=result)

    def retire(self, args: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
        del context
        cached = self._cached("retire", args)
        if cached is not None:
            return cached
        ident, group = self._mutable_group(args.get("group_id"))
        changed = group["lifecycle"] != "retired"
        group["lifecycle"] = "retired"
        group["revision"] += int(changed)
        self.groups[ident] = validate_record(group)
        if self.active_id == ident:
            self._evict("group_retired")
            self.active_id = None
        result = {"schema": "stateframe.workset-result/1", "status": "APPLIED",
                  "operation": "retire", "group": copy.deepcopy(group), "idempotent": not changed}
        return self._save(op="retire", args=args, result=result)

    def _evict(self, reason: str, *, group_id: str | None = None) -> dict[str, object] | None:
        ident = group_id or self.active_id
        if ident is None and self.active_expansion is None:
            return None
        row = {"group_id": ident, "reason": reason, "explicit": True}
        self.last_eviction = row
        if self.active_expansion is not None:
            self.active_expansion = {**self.active_expansion, "resident": False,
                                     "eviction": copy.deepcopy(row)}
        self._event("working_set_evicted", eviction=row)
        return row

    def expand(self, args: Mapping[str, object], context: Mapping[str, object]) -> dict[str, object]:
        # Expansions are cached by the exact request, including the snapshot and cursor.
        ident = self.resolve_id(args.get("group_id"))
        group = self.groups[ident]
        current_snapshot = group_snapshot_id(group, context)
        cache_args = {**args, "_context_snapshot": current_snapshot,
                      "_group_revision": group["revision"]}
        cached = self._cached("expand", cache_args)
        if cached is not None:
            if cached.get("items"):
                # A replayed delivery must be resident again, or the reply
                # promises a next-request page the frame will not carry.
                if self.active_id is not None and self.active_id != ident:
                    self._evict("active_selection_changed", group_id=self.active_id)
                self.active_id = ident
                self.active_expansion = {**copy.deepcopy(cached), "resident": True,
                                         "eviction": None}
            return cached
        if group["lifecycle"] == "retired":
            raise WorkingSetError("retired worksets cannot be expanded", code="RETIRED")
        supplied_snapshot = args.get("snapshot_id")
        cursor = args.get("cursor")
        if cursor is None:
            cursor_number = 0
        elif type(cursor) is int and 0 <= cursor <= MAX_MEMBERS:
            cursor_number = cursor
        else:
            raise WorkingSetError("cursor must be a nonnegative integer")
        if supplied_snapshot is not None and supplied_snapshot != current_snapshot:
            return {"schema": EXPANSION_SCHEMA, "status": "SNAPSHOT_CHANGED",
                    "group_id": ident, "snapshot_id": current_snapshot,
                    "restart_cursor": None, "items": [],
                    "coverage": {key: 0 for key in ("total", "resolved", "unresolved", "tombstones",
                                                     "current", "stale", "historical", "unknown")}}
        budget = args.get("budget_tokens", MAX_PAGE_TOKENS)
        if type(budget) is not int or budget < 0:
            raise WorkingSetError("budget_tokens must be a nonnegative integer")
        max_items = args.get("max_items", MAX_PAGE_ITEMS)
        if type(max_items) is not int or not 1 <= max_items <= MAX_PAGE_ITEMS:
            raise WorkingSetError(f"max_items must be from 1 through {MAX_PAGE_ITEMS}")
        resolved = [_compact_item(resolve_reference(member, context)) for member in group["members"]]
        coverage = _coverage(resolved)
        base = {"schema": EXPANSION_SCHEMA, "status": "OK", "group_id": ident,
                "snapshot_id": current_snapshot, "cursor": cursor_number,
                "coverage": coverage}
        used = token_count(base) + 12
        selected: list[dict[str, object]] = []
        index = cursor_number
        while index < len(resolved) and len(selected) < max_items:
            cost = token_count(resolved[index])
            if used + cost > budget:
                break
            selected.append(copy.deepcopy(resolved[index]))
            used += cost
            index += 1
        if not selected and cursor_number < len(resolved):
            result = {**base, "status": "NEEDS_MORE_BUDGET", "items": [],
                      "next_cursor": cursor_number, "required_tokens": used + token_count(resolved[cursor_number]),
                      "delivery_reservation": {"next_request_tokens": 0, "resident": False},
                      "idempotent": False}
            return self._save(op="expand", args=cache_args, result=result)
        needs_more = index < len(resolved)
        reservation = max(0, used - token_count(base))
        result = {**base, "status": "NEEDS_MORE_BUDGET" if needs_more else "OK",
                  "items": selected, "next_cursor": index if needs_more else None,
                  "delivery_reservation": {"next_request_tokens": reservation, "resident": True},
                  "used_tokens": used, "idempotent": False}
        if self.active_id is not None and self.active_id != ident:
            evicted = self._evict("active_selection_changed", group_id=self.active_id)
        else:
            evicted = None
        self.active_id = ident
        self.last_eviction = copy.deepcopy(evicted)
        self.active_expansion = {**copy.deepcopy(result), "resident": True,
                                 "eviction": copy.deepcopy(evicted)}
        if evicted is not None:
            result["evicted"] = evicted
        return self._save(op="expand", args=cache_args, result=result)

    def apply(self, args: Mapping[str, object], context: Mapping[str, object] | None = None) -> dict[str, object]:
        if not isinstance(args, Mapping):
            raise WorkingSetError("working_set arguments must be an object")
        _reject_model_text(args, where="working_set arguments")
        unknown = set(args) - _ARGUMENTS
        if unknown:
            raise WorkingSetError("unknown working_set argument: " + sorted(map(str, unknown))[0])
        op = args.get("op")
        if op not in OPS:
            raise WorkingSetError(f"op must be one of {', '.join(OPS)}")
        context = context if isinstance(context, Mapping) else {}
        if op == "create":
            return self.create(args, context)
        if op == "extend":
            return self.extend(args, context)
        if op == "remove":
            return self.remove(args, context)
        if op == "merge":
            return self.merge(args, context)
        if op == "retire":
            return self.retire(args, context)
        return self.expand(args, context)


def _virtual_requirement_members(requirement: Mapping[str, object]) -> list[dict[str, object]]:
    refs: list[dict[str, object]] = [{"ref": {"kind": "requirement",
                                                 "id": _id(requirement, "id", "requirement_id")},
                                     "binding": "live"}]
    for raw in requirement.get("check_refs", ()):
        check_id = raw.get("id") if isinstance(raw, Mapping) else raw
        if check_id:
            refs.append({"ref": {"kind": "check", "id": str(check_id)}, "binding": "live"})
    return _members(refs)


def _virtual_group_id(requirement_id: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", requirement_id).strip("-") or "requirement"
    return "req-" + clean[:80]


class WorkingSetContext:
    """Runtime adapter: durable store, host resolution, and delivery receipts."""

    def __init__(self, *, journal: object | None = None, task_id: str = "",
                 emit: Callable[..., object] | None = None) -> None:
        self.journal = journal
        self.task_id = str(task_id or "")
        if emit is None and journal is not None:
            emit = getattr(journal, "emit", None)
        self.store = WorkingSetStore(emit=emit, task_id=self.task_id)
        self._restore(journal)

    def _restore(self, journal: object | None) -> None:
        records = getattr(journal, "records", None)
        if not callable(records):
            return
        try:
            for event in records():
                if isinstance(event, Mapping) and event.get("kind") in {
                        "working_set_updated", "working_set_evicted"}:
                    self.store.replay_event(event)
        except (RuntimeError, ValueError, TypeError):
            # A fresh non-resumable AuditJournal has no records API. Resume
            # validation belongs to the journal; a bad unrelated replay event
            # must not make ordinary reads unavailable.
            return

    @property
    def groups(self) -> dict[str, dict[str, object]]:
        return self.store.groups

    def resolve_group(self, group_id: object) -> dict[str, object]:
        return self.store.resolve_group(group_id)

    def _runtime_context(self, runtime: object | None, *, frame: Mapping[str, object] | None = None,
                         source: str | None = None, path: str | None = None) -> dict[str, object]:
        context = dict(frame) if isinstance(frame, Mapping) else {}
        if runtime is None:
            context.setdefault("task_id", self.task_id)
            return context
        owner = getattr(runtime, "owner", None)
        context["task_id"] = getattr(owner, "task_id", self.task_id)
        try:
            binding = runtime.binding()
        except (AttributeError, TypeError, ValueError, OSError):
            binding = {}
        if isinstance(binding, Mapping):
            revisions = {**binding.get("product_files", {}), **binding.get("harness_files", {})}
            if revisions:
                context["current_revisions"] = revisions
        if path and isinstance(source, str):
            context.setdefault("sources", {})
            context["sources"] = {**context["sources"], path: source}
            current = context.setdefault("current_revisions", {})
            if isinstance(current, Mapping) and path not in current:
                current = dict(current)
                current[path] = "sha256:" + hashlib.sha256(source.encode()).hexdigest()
                context["current_revisions"] = current
        retained = getattr(runtime, "retained", None)
        archive = getattr(retained, "archive", None)
        if archive is not None:
            context["archive"] = archive
        ledger = getattr(runtime, "requirements_ledger", None)
        if isinstance(ledger, Mapping):
            context["requirements_ledger"] = ledger
        context["check_runs"] = list(getattr(runtime, "test_receipts", ()) or ())
        context["tests"] = list(getattr(runtime, "test_receipts", ()) or ())
        context["checks"] = list(getattr(runtime, "host_closure_assessments", ()) or ())
        if owner is not None:
            context["receipts"] = list(getattr(owner, "_receipts", ()) or ())
            row = getattr(owner, "task_row", {})
            if isinstance(row, Mapping):
                for key in ("rename_map", "mappings"):
                    if key in row:
                        context["rename_map"] = row[key]
                        break
        return context

    def sync_virtual_groups(self, ledger: object) -> list[dict[str, object]]:
        if not isinstance(ledger, Mapping):
            return []
        requirements = _rows(ledger, "requirements")
        created: list[dict[str, object]] = []
        for requirement in requirements:
            req_id = _id(requirement, "id", "requirement_id")
            if not req_id:
                continue
            ident = _virtual_group_id(req_id)
            if not ident or ident == "req-":
                ident = "req-requirement"
            existing = self.store.groups.get(ident)
            if existing is not None and existing.get("created_by") != "host":
                continue
            if existing is None and len(self.store.groups) >= MAX_GROUPS:
                break
            try:
                next_revision = int(existing["revision"]) if existing is not None else 1
                group = validate_record({
                    "schema": SCHEMA, "id": ident, "label": f"requirement {req_id}",
                    "scope": {"task_id": self.task_id, "virtual": "requirement", "requirement_id": req_id},
                    "created_by": "host", "revision": next_revision,
                    "lifecycle": "active",
                    "members": _virtual_requirement_members(requirement),
                })
            except WorkingSetError:
                continue
            if existing is not None and all(
                    existing.get(key) == group.get(key)
                    for key in ("schema", "id", "label", "scope", "created_by", "lifecycle", "members")):
                continue
            if existing is not None:
                group["revision"] = int(existing["revision"]) + 1
            self.store.groups[ident] = group
            created.append(copy.deepcopy(group))
            self.store._event(
                "working_set_virtual_updated" if existing is not None
                else "working_set_virtual_created", group=group)
        return created

    def operation(self, args: Mapping[str, object], *, runtime: object | None = None,
                  context: Mapping[str, object] | None = None, frame: Mapping[str, object] | None = None,
                  source: str | None = None, path: str | None = None) -> dict[str, object]:
        built = self._runtime_context(runtime, frame=frame, source=source, path=path)
        if isinstance(context, Mapping):
            built.update(copy.deepcopy(dict(context)))
        self.sync_virtual_groups(built.get("requirements_ledger"))
        try:
            result = self.store.apply(args, built)
        except WorkingSetError:
            raise
        if str(args.get("op")) == "expand":
            result.setdefault("delivery", "next_request_reserved")
            result.setdefault("read_path", "artifact_resolution")
        return result

    def frame(self, *, runtime: object | None = None, context: Mapping[str, object] | None = None,
              frame: Mapping[str, object] | None = None, source: str | None = None,
              path: str | None = None, frame_budget_tokens: int = 4000,
              gear: str | None = None) -> dict[str, object]:
        built = self._runtime_context(runtime, frame=frame, source=source, path=path)
        if isinstance(context, Mapping):
            built.update(copy.deepcopy(dict(context)))
        self.sync_virtual_groups(built.get("requirements_ledger"))
        active_expansion = self.store.active_expansion
        active_group = self.store.groups.get(self.store.active_id or "")
        current_snapshot = (group_snapshot_id(active_group, built)
                            if isinstance(active_group, Mapping) else None)
        if (isinstance(active_expansion, Mapping)
                and active_expansion.get("resident")
                and active_expansion.get("snapshot_id") not in {None, current_snapshot}):
            self.store._evict("snapshot_changed")
        view = project_working_sets(self.store, context=built, gear=gear,
                                    frame_budget_tokens=frame_budget_tokens)
        return view

    def directory_tokens(self, directory: object) -> int:
        return token_count(directory)

    def checkpoint(self) -> dict[str, object]:
        return self.store.snapshot()

    def receipt(self, result: Mapping[str, object]) -> dict[str, object]:
        group = result.get("group")
        group = group if isinstance(group, Mapping) else {}
        group_id = result.get("group_id", group.get("id", ""))
        revision = result.get("snapshot_id", group.get("revision", ""))
        status = {
            "OK": "WORKING_SET_DELIVERED",
            "NEEDS_MORE_BUDGET": "WORKING_SET_NEEDS_MORE_BUDGET",
            "SNAPSHOT_CHANGED": "WORKING_SET_SNAPSHOT_CHANGED",
            "REJECTED": "WORKING_SET_REJECTED",
        }.get(str(result.get("status", "OK")), "WORKING_SET_DELIVERED")
        from kernel_read_fixes import working_set_receipt
        return working_set_receipt(group_id=group_id, revision=revision, status=status)


def _directory_rows(store: WorkingSetStore) -> list[dict[str, object]]:
    rows = []
    for group in store.groups.values():
        active = group["id"] == store.active_id and bool(
            store.active_expansion and store.active_expansion.get("resident"))
        rows.append({"id": group["id"], "label": group["label"],
                     "lifecycle": group["lifecycle"], "revision": group["revision"],
                     "member_count": len(group["members"]),
                     "estimated_tokens": max(1, token_count(group["members"])),
                     "residency": "active" if active else "cold"})
    rows.sort(key=lambda row: (row["residency"] != "active", str(row["id"])))
    return rows


def _bounded_directory(rows: list[dict[str, object]], max_tokens: int) -> list[dict[str, object]]:
    if max_tokens <= 0:
        return []
    selected: list[dict[str, object]] = []
    for row in rows[:MAX_VISIBLE]:
        candidate = [*selected, row]
        if token_count(candidate) > max_tokens:
            break
        selected.append(row)
    return selected


def directory_block(store: WorkingSetStore, *, frame_budget_tokens: int = 4000,
                    gear: str | None = None) -> list[dict[str, object]] | None:
    """Build the <=6-handle directory; never put expansion bodies in it."""
    rows = _directory_rows(store)
    if gear == "g1" and not any(row["residency"] != "active" for row in rows):
        return None
    budget = max(0, int(frame_budget_tokens * 0.05))
    return _bounded_directory(rows, budget)


def project_working_sets(store: WorkingSetStore, *, context: Mapping[str, object] | None = None,
                         gear: str | None = None, frame_budget_tokens: int = 4000) -> dict[str, object]:
    context = context if isinstance(context, Mapping) else {}
    directory = directory_block(store, frame_budget_tokens=frame_budget_tokens, gear=gear)
    active = None
    if store.active_id is not None:
        expansion = store.active_expansion or {}
        resident = bool(expansion.get("resident"))
        # An evicted expansion keeps only its handle + eviction reason; stale
        # bodies and reservations must never survive into later requests.
        active = {"group_id": store.active_id,
                  "snapshot_id": expansion.get("snapshot_id"),
                  "resident": resident,
                  "reserved_tokens": int(expansion.get("delivery_reservation", {}).get(
                      "next_request_tokens", 0)) if resident else 0,
                  "eviction": copy.deepcopy(store.last_eviction),
                  "items": copy.deepcopy(expansion.get("items", [])) if resident else []}
    return {"schema": VIEW_SCHEMA, "directory_schema": DIRECTORY_SCHEMA,
            "directory": directory or [], "visible_count": len(directory or []),
            "active": active, "group_count": len(store.groups),
            "snapshot_id": snapshot_id(context) if context else None}


def project_g1_manifest(manifest: object) -> dict[str, object] | None:
    """Keep a retained active expansion in the thin g1 presentation.

    The compact directory is useful in g1 only when there is a cold/inactive
    handle to select.  The active expansion remains visible even when the
    directory is empty, which is the residency part of the working-set
    contract.  Returning ``None`` for an empty manifest keeps the legacy g1
    projection unchanged for the opt-in arm's initial request.
    """
    if not enabled() or not isinstance(manifest, Mapping):
        return None
    directory = manifest.get("directory")
    active = manifest.get("active")
    rows: list[dict[str, object]] = []
    if isinstance(directory, Sequence) and not isinstance(directory, (str, bytes, bytearray)):
        rows = [copy.deepcopy(row) for row in directory if isinstance(row, Mapping)]
    if not rows and not isinstance(active, Mapping):
        return None
    result = copy.deepcopy(dict(manifest))
    inactive = [row for row in rows if row.get("residency") != "active"]
    if not inactive:
        result["directory"] = []
        result["visible_count"] = 0
    else:
        result["directory"] = rows[:MAX_VISIBLE]
        result["visible_count"] = len(result["directory"])
    return result


def tool_surface(base: list[dict]) -> list[dict]:
    """Add exactly one structured context tool in the ON arm."""
    if not enabled() or any(
            isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
            and tool["function"].get("name") == TOOL_NAME for tool in base):
        return base
    return [*base, {"type": "function", "function": {
        "name": TOOL_NAME,
        "description": "Create or expand one flat named working set of existing typed references. "
                       "The host resolves freshness, applicability, and coverage; label is the only text field.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": list(OPS)},
            "label": {"type": "string", "maxLength": MAX_LABEL},
            "group_id": {"type": "string"}, "source_id": {"type": "string"},
            "target_id": {"type": "string"},
            "refs": {"type": "array", "items": {"type": "object"}},
            "members": {"type": "array", "items": {"type": "object"}},
            "member_refs": {"type": "array", "items": {"type": "object"}},
            "idempotency_key": {"type": "string", "maxLength": MAX_ID},
            "cursor": {"type": "integer", "minimum": 0},
            "snapshot_id": {"type": "string"},
            "budget_tokens": {"type": "integer", "minimum": 0, "maximum": MAX_PAGE_TOKENS},
            "max_items": {"type": "integer", "minimum": 1, "maximum": MAX_PAGE_ITEMS},
            "keep_resident": {"type": "boolean"},
        }, "required": ["op"], "additionalProperties": False},
    }}]


def apply_operation(manager: WorkingSetContext, arguments: Mapping[str, object], **kwargs: object) -> str:
    """JSON tool-result helper used by the runtime dispatch seam."""
    try:
        result = manager.operation(arguments, **kwargs)
    except WorkingSetError as error:
        result = {"schema": "stateframe.workset-result/1", "status": "REJECTED",
                  "code": error.code, "reason": error.reason}
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


__all__ = [
    "DIRECTORY_SCHEMA", "EXPANSION_SCHEMA", "MAX_MEMBERS", "MAX_MEMBERS_SOFT",
    "MAX_OPERATION_RESULTS", "MAX_VISIBLE",
    "OPS", "REF_KINDS", "SCHEMA", "TOOL_NAME", "VIEW_SCHEMA", "WorkingSetContext",
    "WorkingSetError", "WorkingSetStore", "apply_operation", "directory_block", "enabled",
    "mode", "normalize_member", "normalize_ref", "project_g1_manifest",
    "project_working_sets", "resolve_reference",
    "snapshot_id", "token_count", "tool_surface", "validate_record",
]
