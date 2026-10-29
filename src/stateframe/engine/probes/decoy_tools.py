"""Deterministic Mission-Control-shaped decoy tools for ReAct regime tests.

Decoys are deliberately useful-looking schemas with no implementation in this
repair sandbox.  Their response says that no change was made, so a model can
recover without mistaking the event for progress.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence


def _schema_tool(
    name: str,
    description: str,
    properties: Mapping[str, Mapping[str, object]],
    required: Sequence[str],
) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: dict(value) for key, value in properties.items()},
                "required": list(required),
            },
        },
    }


_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_BOOLEAN = {"type": "boolean"}

DECOY_KIND_CROSS_DOMAIN = "cross_domain"
DECOY_KIND_NEAR_DUPLICATE = "near_duplicate"
DECOY_KINDS = (DECOY_KIND_CROSS_DOMAIN, DECOY_KIND_NEAR_DUPLICATE)


# Ordered catalog: changing this order changes an experiment's surface and must
# therefore be an intentional measurement change.
DECOY_CATALOG: tuple[dict[str, object], ...] = (
    _schema_tool("seo_list_keywords", "List tracked SEO keywords for a workspace.",
                 {"seed_term": _STRING, "market": _STRING, "limit": _INTEGER},
                 ("seed_term", "market")),
    _schema_tool("social_create_post", "Create a social post draft.",
                 {"channel": _STRING, "copy": _STRING, "publish_at": _STRING},
                 ("channel", "copy")),
    _schema_tool("helm_create_initiative", "Create an initiative under the company helm.",
                 {"name": _STRING, "objective": _STRING, "owner": _STRING},
                 ("name", "objective")),
    _schema_tool("content_generate_draft", "Generate a content draft from a brief.",
                 {"brief": _STRING, "format": _STRING, "audience": _STRING},
                 ("brief", "format")),
    _schema_tool("kanban_move_task", "Move a task to a workflow column.",
                 {"task_id": _STRING, "column": _STRING}, ("task_id", "column")),
    _schema_tool("research_start_run", "Start a bounded research run.",
                 {"question": _STRING, "source_limit": _INTEGER}, ("question",)),
    _schema_tool("press_send_newsletter", "Queue a newsletter for a subscriber segment.",
                 {"campaign_id": _STRING, "segment": _STRING}, ("campaign_id", "segment")),
    _schema_tool("leads_update", "Update a lead record.",
                 {"lead_id": _STRING, "fields": {"type": "object"}}, ("lead_id", "fields")),
    _schema_tool("sniper_generate_battlecard", "Generate a competitor battlecard.",
                 {"competitor": _STRING, "product": _STRING}, ("competitor", "product")),
    _schema_tool("brand_check_compliance", "Check an asset against brand rules.",
                 {"asset_id": _STRING, "channel": _STRING}, ("asset_id", "channel")),
    _schema_tool("audience_create_segment", "Create an audience segment.",
                 {"name": _STRING, "criteria": _STRING}, ("name", "criteria")),
    _schema_tool("seo_refresh_audit", "Refresh an SEO audit for a page.",
                 {"url": _STRING, "scope": _STRING}, ("url",)),
    _schema_tool("social_schedule_campaign", "Schedule a social campaign.",
                 {"campaign_id": _STRING, "start_at": _STRING}, ("campaign_id", "start_at")),
    _schema_tool("helm_create_operation", "Create an operational workstream.",
                 {"name": _STRING, "purpose": _STRING}, ("name", "purpose")),
    _schema_tool("content_publish_article", "Publish an approved article.",
                 {"draft_id": _STRING, "channel": _STRING}, ("draft_id", "channel")),
    _schema_tool("kanban_assign_task", "Assign a task to a team member.",
                 {"task_id": _STRING, "assignee": _STRING}, ("task_id", "assignee")),
    _schema_tool("research_get_sources", "List sources collected by a research run.",
                 {"run_id": _STRING, "limit": _INTEGER}, ("run_id",)),
    _schema_tool("press_preview_campaign", "Preview a press campaign.",
                 {"campaign_id": _STRING}, ("campaign_id",)),
    _schema_tool("leads_create", "Create a lead record.",
                 {"email": _STRING, "source": _STRING}, ("email", "source")),
    _schema_tool("sniper_list_competitors", "List tracked competitors.",
                 {"market": _STRING}, ("market",)),
    _schema_tool("brand_create_guideline", "Create a brand guideline.",
                 {"name": _STRING, "rule": _STRING}, ("name", "rule")),
    _schema_tool("seo_publish_metadata", "Publish page metadata.",
                 {"page_id": _STRING, "title": _STRING, "description": _STRING},
                 ("page_id", "title")),
    _schema_tool("social_get_calendar", "Read a social publishing calendar.",
                 {"channel": _STRING, "month": _STRING}, ("channel", "month")),
    _schema_tool("helm_list_directives", "List active helm directives.",
                 {"status": _STRING}, ("status",)),
    _schema_tool("content_list_drafts", "List content drafts.",
                 {"status": _STRING, "limit": _INTEGER}, ("status",)),
    _schema_tool("kanban_list_tasks", "List tasks in a workflow column.",
                 {"column": _STRING, "limit": _INTEGER}, ("column",)),
    _schema_tool("research_export_brief", "Export a research brief.",
                 {"run_id": _STRING, "format": _STRING}, ("run_id", "format")),
    _schema_tool("press_list_subscribers", "List newsletter subscribers.",
                 {"segment": _STRING, "limit": _INTEGER}, ("segment",)),
    _schema_tool("leads_list_pipeline", "List leads in a pipeline.",
                 {"stage": _STRING, "limit": _INTEGER}, ("stage",)),
    _schema_tool("sniper_compare_positioning", "Compare competitor positioning.",
                 {"competitor": _STRING, "market": _STRING}, ("competitor", "market")),
    _schema_tool("brand_list_assets", "List assets covered by brand controls.",
                 {"channel": _STRING, "limit": _INTEGER}, ("channel",)),
    _schema_tool("audience_refresh_profile", "Refresh an audience profile.",
                 {"profile_id": _STRING}, ("profile_id",)),
    _schema_tool("operations_get_status", "Read an operations status board.",
                 {"operation_id": _STRING}, ("operation_id",)),
)


# Each sibling has a tempting name but a different contract from its real tool.  Keep
# families together here; surface construction expands selected siblings beside their real
# counterpart so the model sees a plausible local choice rather than unrelated clutter.
NEAR_DUPLICATE_FAMILIES: tuple[tuple[str, tuple[dict[str, object], ...]], ...] = (
    ("grep", (
        _schema_tool("search_regex", "Search the repository with explicit regex flags.",
                     {"pattern": _STRING, "flags": _STRING}, ("pattern", "flags")),
        _schema_tool("find_symbol", "Find a named symbol in the repository.",
                     {"symbol": _STRING, "path": _STRING}, ("symbol",)),
        _schema_tool("grep_files", "Search only selected repository files.",
                     {"pattern": _STRING, "files": {"type": "array"}},
                     ("pattern", "files")),
    )),
    ("read", (
        _schema_tool("read_lines", "Read an inclusive line range from a file.",
                     {"path": _STRING, "start": _INTEGER, "end": _INTEGER},
                     ("path", "start", "end")),
        _schema_tool("read_symbol", "Read one named symbol from a file.",
                     {"path": _STRING, "symbol": _STRING}, ("path", "symbol")),
        _schema_tool("read_head", "Read the first lines of a file.",
                     {"path": _STRING, "line_count": _INTEGER}, ("path", "line_count")),
    )),
    ("edit", (
        _schema_tool("edit_range", "Replace a line range in a file.",
                     {"path": _STRING, "start_line": _INTEGER, "end_line": _INTEGER,
                      "replacement": _STRING},
                     ("path", "start_line", "end_line", "replacement")),
        _schema_tool("apply_patch", "Apply a unified diff to a file.",
                     {"path": _STRING, "patch": _STRING}, ("path", "patch")),
        _schema_tool("edit_symbol", "Replace a named symbol in a file.",
                     {"path": _STRING, "symbol": _STRING, "source": _STRING},
                     ("path", "symbol", "source")),
    )),
    ("str_replace", (
        _schema_tool("replace_all", "Replace every exact occurrence in a file.",
                     {"path": _STRING, "old_str": _STRING, "new_str": _STRING,
                      "max_replacements": _INTEGER},
                     ("path", "old_str", "new_str", "max_replacements")),
        _schema_tool("search_replace_regex", "Replace regex matches in a file.",
                     {"path": _STRING, "pattern": _STRING, "replacement": _STRING},
                     ("path", "pattern", "replacement")),
        _schema_tool("replace_range", "Replace a character range in a file.",
                     {"path": _STRING, "start": _INTEGER, "end": _INTEGER,
                      "replacement": _STRING},
                     ("path", "start", "end", "replacement")),
    )),
    ("file_create", (
        _schema_tool("create_module", "Create a Python module and package parents.",
                     {"module_path": _STRING, "source": _STRING, "package": _STRING},
                     ("module_path", "source", "package")),
        _schema_tool("create_file", "Create a file, optionally replacing it.",
                     {"path": _STRING, "content": _STRING, "overwrite": _BOOLEAN},
                     ("path", "content", "overwrite")),
        _schema_tool("scaffold_file", "Create a file from a named template.",
                     {"path": _STRING, "template": _STRING, "variables": {"type": "object"}},
                     ("path", "template", "variables")),
    )),
    ("run_tests", (
        _schema_tool("run_test_file", "Run every test in one test file.",
                     {"test_path": _STRING, "timeout": _INTEGER},
                     ("test_path", "timeout")),
        _schema_tool("run_pytest_selector", "Run one pytest selector.",
                     {"selector": _STRING, "cwd": _STRING}, ("selector", "cwd")),
        _schema_tool("run_tests_verbose", "Run tests with verbose output.",
                     {"target": _STRING, "verbosity": _INTEGER}, ("target", "verbosity")),
    )),
    ("done", (
        _schema_tool("finish", "Finish after verifying the repair.",
                     {"summary": _STRING, "evidence": _STRING}, ("summary", "evidence")),
        _schema_tool("submit", "Submit the repair for review.",
                     {"summary": _STRING, "status": _STRING}, ("summary", "status")),
        _schema_tool("done_with_summary", "Declare completion with explicit checks.",
                     {"summary": _STRING, "checks": {"type": "array"}},
                     ("summary", "checks")),
    )),
)

NEAR_DUPLICATE_CATALOG: tuple[dict[str, object], ...] = tuple(
    sibling
    for _, siblings in NEAR_DUPLICATE_FAMILIES
    for sibling in siblings
)
NEAR_DUPLICATE_SIBLING_OF = {
    str(sibling["function"]["name"]): real_name
    for real_name, siblings in NEAR_DUPLICATE_FAMILIES
    for sibling in siblings
}

_DECOY_NAMES = frozenset(
    str(tool["function"]["name"])
    for tool in DECOY_CATALOG
)

_DECOY_NAMES_BY_KIND = {
    DECOY_KIND_CROSS_DOMAIN: _DECOY_NAMES,
    DECOY_KIND_NEAR_DUPLICATE: frozenset(NEAR_DUPLICATE_SIBLING_OF),
}


def decoy_kind(kind: str | None = None) -> str:
    """Resolve and validate the selected catalog, defaulting to today's cross-domain rig."""
    selected = kind if kind is not None else os.environ.get("REACT_DECOY_KIND")
    selected = selected or DECOY_KIND_CROSS_DOMAIN
    if selected not in DECOY_KINDS:
        raise ValueError(f"REACT_DECOY_KIND must be one of {DECOY_KINDS}")
    return selected


def _catalog(kind: str | None = None) -> tuple[dict[str, object], ...]:
    return (DECOY_CATALOG if decoy_kind(kind) == DECOY_KIND_CROSS_DOMAIN
            else NEAR_DUPLICATE_CATALOG)


def decoy_names(count: int, kind: str | None = None) -> tuple[str, ...]:
    """Return the stable names selected for a requested decoy count."""
    catalog = _catalog(kind)
    if count < 0 or count > len(catalog):
        raise ValueError(f"decoy count must be between 0 and {len(catalog)}")
    return tuple(str(tool["function"]["name"]) for tool in catalog[:count])


def decoy_tools(count: int, kind: str | None = None) -> tuple[dict[str, object], ...]:
    """Return independent schema objects for the first ``count`` catalog entries."""
    catalog = _catalog(kind)
    return tuple(json.loads(json.dumps(tool)) for tool in catalog[:len(decoy_names(count, kind))])


def near_duplicate_sibling(name: str) -> str | None:
    """Return the real tool family for a near-duplicate name, if it has one."""
    return NEAR_DUPLICATE_SIBLING_OF.get(name)


def interleave_tools(
    real_tools: Sequence[Mapping[str, object]],
    decoys: Sequence[Mapping[str, object]],
    *,
    seed: int = 17,
    kind: str | None = None,
) -> list[dict[str, object]]:
    """Insert decoys at deterministic positions while preserving real-tool order."""
    if decoy_kind(kind) == DECOY_KIND_NEAR_DUPLICATE:
        by_real: dict[str, list[Mapping[str, object]]] = {}
        unmatched: list[Mapping[str, object]] = []
        for tool in decoys:
            name = str(tool["function"]["name"])
            real_name = near_duplicate_sibling(name)
            if real_name is None:
                unmatched.append(tool)
            else:
                by_real.setdefault(real_name, []).append(tool)

        surface: list[dict[str, object]] = []
        real_names = {str(tool["function"]["name"]) for tool in real_tools}
        for real in real_tools:
            real_name = str(real["function"]["name"])
            surface.append(dict(real))
            siblings = by_real.get(real_name, [])
            siblings = sorted(
                siblings,
                key=lambda tool: hashlib.sha256(
                    f"{seed}|{real_name}|{tool['function']['name']}".encode()
                ).hexdigest(),
            )
            surface.extend(dict(tool) for tool in siblings)
        for tool in unmatched:
            if str(tool["function"]["name"]) not in real_names:
                surface.append(dict(tool))
        return surface

    surface = [dict(tool) for tool in real_tools]
    for index, tool in enumerate(decoys):
        position = (seed * 7 + index * 31) % (len(surface) + 1)
        surface.insert(position, dict(tool))
    return surface


def build_tool_surface(
    real_tools: Sequence[Mapping[str, object]],
    count: int,
    *,
    seed: int = 17,
    kind: str | None = None,
) -> Sequence[Mapping[str, object]]:
    """Build the model surface; count=0 returns the exact real object unchanged."""
    if count == 0:
        return real_tools
    selected_kind = decoy_kind(kind)
    return interleave_tools(
        real_tools, decoy_tools(count, selected_kind), seed=seed, kind=selected_kind
    )


def is_decoy_name(name: str, kind: str | None = None) -> bool:
    return name in _DECOY_NAMES_BY_KIND[decoy_kind(kind)]


def is_near_duplicate_name(name: str) -> bool:
    return name in _DECOY_NAMES_BY_KIND[DECOY_KIND_NEAR_DUPLICATE]


def _matches_type(value: object, expected: object) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def validate_tool_arguments(
    tools: Sequence[Mapping[str, object]],
    name: str,
    arguments: object,
) -> tuple[str, ...]:
    """Validate the small JSON-Schema subset used by this harness."""
    schema_tool = next(
        (tool for tool in tools
         if isinstance(tool.get("function"), Mapping)
         and tool["function"].get("name") == name),
        None,
    )
    if schema_tool is None:
        return ()
    if not isinstance(arguments, Mapping):
        return ("arguments must be an object",)
    function = schema_tool.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    if not isinstance(parameters, Mapping):
        return ()
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    errors = [
        f"missing required parameter: {required}"
        for required in parameters.get("required", ())
        if required not in arguments
    ]
    for key, value in arguments.items():
        prop = properties.get(key)
        if isinstance(prop, Mapping) and not _matches_type(value, prop.get("type")):
            errors.append(f"parameter {key} must be {prop.get('type')}")
        elif (isinstance(prop, Mapping) and isinstance(prop.get("enum"), list)
              and value not in prop["enum"]):
            # r45 (r44c waiver W2): the menu's enum is the owner's restriction (e.g. repair-lock read targets)
            errors.append(f"parameter {key} must be one of {json.dumps(prop['enum'])}")
    return tuple(errors)


def decoy_response(
    name: str,
    arguments: Mapping[str, object],
    call_id: str,
    kind: str | None = None,
) -> dict[str, object]:
    """Return a recoverable, explicitly non-success response for a decoy call."""
    payload = json.dumps(dict(arguments), sort_keys=True, default=str)
    digest = hashlib.sha256(f"{name}|{call_id}|{payload}".encode()).hexdigest()[:12]
    result = {
        "ok": False,
        "status": "unavailable",
        "tool": name,
        "request_id": f"sim-{digest}",
        "message": "Mission Control data plane is unavailable in this repair sandbox; no change was made.",
        "decoy": True,
    }
    if decoy_kind(kind) == DECOY_KIND_NEAR_DUPLICATE:
        result["decoy_kind"] = DECOY_KIND_NEAR_DUPLICATE
        result["sibling_of"] = near_duplicate_sibling(name)
    return result
