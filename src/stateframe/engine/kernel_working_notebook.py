"""Model-owned working notes carried beside useful actions.

The notebook is deliberately separate from host facts: note text is copied
verbatim, while the runtime binds file/receipt references and dependency
revisions when the action closes.  The feature is off unless the v44 switch (or
the legacy ``KERNEL_WORKING_NOTES`` switch) explicitly enables it.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping


ENV = "STATEFRAME_WORKING_NOTES"
LEGACY_ENV = "KERNEL_WORKING_NOTES"
NOTE_CAPABLE_TOOLS = frozenset({
    "read", "grep", "edit", "str_replace", "file_create", "create_file",
    "run_tests", "write", "replace", "replace_module",
})
NOTE_KINDS = ("plan", "decision", "hypothesis", "rejected", "todo")
MODEL_STATUSES = ("active", "retired")
HOST_STATUSES = ("active", "retired", "stale-check", "superseded")
MAX_ID = 40
MAX_TEXT = 600
MAX_SCOPE_ID = 120
MAX_REVISION = 160
MAX_REFS = 8
DEFAULT_BUDGET_TOKENS = 900
MIN_BUDGET_TOKENS = 600
MAX_BUDGET_TOKENS = 1200
NOTE_SCHEMA = "working-note/1"
RETRIEVAL_SCHEMA = "working-note-retrieval/1"


def _switch_value(name: str, default: str = "off") -> str:
    value = str(os.environ.get(name, default) or default).strip().lower()
    if value in {"", "0", "off", "false", "no"}:
        return "off"
    if value in {"1", "on", "true", "yes"}:
        return "on"
    raise ValueError(f"{name} must be on or off")


def mode() -> str:
    """Resolve the new gate, retaining the old contract for earlier lanes."""
    if ENV in os.environ:
        return _switch_value(ENV)
    return _switch_value(LEGACY_ENV)


def enabled() -> bool:
    return mode() == "on"


def budget_tokens() -> int:
    raw = os.environ.get("STATEFRAME_WORKING_NOTES_BUDGET", str(DEFAULT_BUDGET_TOKENS))
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("STATEFRAME_WORKING_NOTES_BUDGET must be an integer") from error
    if not MIN_BUDGET_TOKENS <= value <= MAX_BUDGET_TOKENS:
        raise ValueError(
            f"STATEFRAME_WORKING_NOTES_BUDGET must be between {MIN_BUDGET_TOKENS} and "
            f"{MAX_BUDGET_TOKENS}"
        )
    return value


def _text_schema(limit: int) -> dict[str, object]:
    return {"type": "string", "maxLength": limit}


def note_schema() -> dict[str, object]:
    """Return the compact sidecar schema advertised on useful action tools."""
    return {
        "type": "object",
        "description": (
            "Optional model-owned note written with this useful action. Text is verbatim and "
            "unverified; it never changes host facts, authorization, tests or completion. "
            "Use status=retired or empty text to retire a prior note."
        ),
        "properties": {
            "id": _text_schema(MAX_ID),
            "expected_version": {"type": "integer", "minimum": 0},
            "scope": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["request", "workstream"]},
                    "id": _text_schema(MAX_SCOPE_ID),
                },
                "required": ["kind", "id"],
                "additionalProperties": False,
            },
            "kind": {"type": "string", "enum": list(NOTE_KINDS)},
            "text": _text_schema(MAX_TEXT),
            "refs": {
                "type": "object",
                "properties": {
                    "receipts": {"type": "array", "maxItems": MAX_REFS, "items": _text_schema(160)},
                    "files": {"type": "array", "maxItems": MAX_REFS, "items": _text_schema(240)},
                },
                "required": ["receipts", "files"],
                "additionalProperties": False,
            },
            "dependency_revision": _text_schema(MAX_REVISION),
            "status": {"type": "string", "enum": list(MODEL_STATUSES)},
            "supersedes": {"type": ["string", "null"], "maxLength": MAX_ID},
        },
        "required": [
            "id", "expected_version", "scope", "kind", "text", "refs",
            "dependency_revision", "status", "supersedes",
        ],
        "additionalProperties": False,
    }


def _tool(name: str, description: str, properties: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def retrieval_tool() -> dict[str, object]:
    return _tool(
        "note_retrieve",
        "Retrieve your archived or stale-check notes by id or question. Results are unverified model notes.",
        {"id": _text_schema(MAX_ID), "question": _text_schema(300)},
    )


def tool_surface(base: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add the note sidecar without mutating the OFF-arm input list."""
    # The legacy gate owns its old note_update-only surface.  This preserves
    # earlier lane contracts while STATEFRAME_WORKING_NOTES opts into v44.
    if not enabled() or ENV not in os.environ:
        return base
    result = copy.deepcopy(base)
    for tool in result:
        function = tool.get("function")
        if not isinstance(function, dict) or function.get("name") not in NOTE_CAPABLE_TOOLS:
            continue
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            continue
        properties = parameters.setdefault("properties", {})
        if isinstance(properties, dict):
            properties.setdefault("note", note_schema())
        required = parameters.get("required")
        if isinstance(required, list):
            parameters["required"] = [item for item in required if item != "note"]
    names = {
        tool["function"].get("name")
        for tool in result
        if isinstance(tool.get("function"), Mapping)
    }
    if "note_retrieve" not in names:
        result.append(retrieval_tool())
    return result


def split_action_arguments(arguments: object, *, call: Mapping[str, object] | None = None) -> tuple[dict, dict | None]:
    """Remove the note from dispatch arguments while retaining its wire payload."""
    raw = copy.deepcopy(arguments) if isinstance(arguments, Mapping) else {}
    note = raw.pop("note", None)
    if note is None and isinstance(call, Mapping):
        function = call.get("function")
        encoded = function.get("arguments") if isinstance(function, Mapping) else None
        if isinstance(encoded, str):
            import json
            try:
                decoded = json.loads(encoded)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded = None
            if isinstance(decoded, Mapping):
                note = copy.deepcopy(decoded.get("note"))
    return dict(raw), copy.deepcopy(note) if note is not None else None


def approximate_tokens(value: object) -> int:
    text = value if isinstance(value, str) else str(value)
    return max(1, (len(text) + 3) // 4)


def _string_list(value: object, *, limit: int, name: str) -> list[str]:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{name} must be a list of at most {limit} strings")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > 240:
            raise ValueError(f"{name} contains an invalid reference")
        if item not in result:
            result.append(item)
    return result


def validate_note(value: object, *, allow_legacy: bool = False) -> dict[str, object]:
    """Validate a model sidecar; host bindings are added by ``DecisionState``."""
    if not isinstance(value, Mapping):
        raise ValueError("note must be an object")
    if allow_legacy and set(value) <= {"id", "expected_version", "text", "evidence_steps"}:
        evidence = value.get("evidence_steps", [])
        if not isinstance(evidence, list):
            raise ValueError("evidence_steps must be a list")
        return {
            "id": value.get("id"), "expected_version": value.get("expected_version"),
            "scope": {"kind": "request", "id": "legacy"}, "kind": "hypothesis",
            "text": value.get("text"), "refs": {"receipts": [], "files": []},
            "dependency_revision": "", "status": "active" if value.get("text") else "retired",
            "supersedes": None, "evidence_steps": copy.deepcopy(evidence),
        }
    expected = set(note_schema()["properties"])
    if set(value) != expected:
        raise ValueError("note fields do not match the working-note contract")
    identifier = value["id"]
    if (not isinstance(identifier, str) or not identifier.strip() or len(identifier) > MAX_ID
            or not identifier[0].isalnum()
            or any(not (char.isalnum() or char in "_-") for char in identifier)):
        raise ValueError("note id is invalid")
    if type(value["expected_version"]) is not int or value["expected_version"] < 0:
        raise ValueError("note expected_version must be a nonnegative integer")
    scope = value["scope"]
    if (not isinstance(scope, Mapping) or set(scope) != {"kind", "id"}
            or scope["kind"] not in {"request", "workstream"}
            or not isinstance(scope["id"], str) or not scope["id"].strip()
            or len(scope["id"]) > MAX_SCOPE_ID):
        raise ValueError("note scope must identify a request or workstream")
    if value["kind"] not in NOTE_KINDS:
        raise ValueError("note kind is invalid")
    if not isinstance(value["text"], str) or len(value["text"]) > MAX_TEXT:
        raise ValueError("note text is too long")
    refs = value["refs"]
    if not isinstance(refs, Mapping) or set(refs) != {"receipts", "files"}:
        raise ValueError("note refs must contain receipts and files")
    refs = {"receipts": _string_list(refs["receipts"], limit=MAX_REFS, name="receipts"),
            "files": _string_list(refs["files"], limit=MAX_REFS, name="files")}
    revision = value["dependency_revision"]
    if not isinstance(revision, str) or len(revision) > MAX_REVISION:
        raise ValueError("note dependency_revision is invalid")
    if value["status"] not in MODEL_STATUSES:
        raise ValueError("note status is invalid")
    supersedes = value["supersedes"]
    if supersedes is not None and (not isinstance(supersedes, str) or len(supersedes) > MAX_ID):
        raise ValueError("note supersedes is invalid")
    if value["status"] == "active" and not value["text"].strip():
        raise ValueError("active notes require nonempty text")
    if value["status"] == "retired" and value["text"]:
        raise ValueError("retired notes must use empty text")
    normalized = copy.deepcopy(dict(value))
    normalized["refs"] = refs
    return normalized


def retrieval_matches(note: Mapping[str, object], *, note_id: str | None, question: str | None) -> bool:
    if note_id and note.get("id") == note_id:
        return True
    if not question:
        return bool(note_id is None)
    haystack = " ".join(
        str(note.get(key, "")) for key in ("id", "text", "kind", "status", "scope", "refs")
    ).casefold()
    return all(term in haystack for term in question.casefold().split())


def render_header() -> str:
    return "[YOUR NOTES | model-owned, unverified | host facts and receipts win]"
