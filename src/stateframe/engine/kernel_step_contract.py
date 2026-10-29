"""Opt-in schema-constrained proposals; syntax never substitutes for authority."""
from __future__ import annotations

from kernel_step_limits import remaining, step_guidance

import copy
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from kernel_evictfix_mode import enabled as evictfix_enabled


ACTION_SCHEMA_ENV = "STATEFRAME_ACTION_SCHEMA"
ACTION_SCHEMA_NAME = "stateframe_action_envelope"
ACTION_CLAIMS = ("complete", "partial", "blocked")
SYSTEM_PROMPT_ENV = "STATEFRAME_SYSTEM_PROMPT"
ENTRY_FILE_ONLY_ENV = "STATEFRAME_ENTRY_FILE_ONLY"

CLAIM_RULE_TEXT = (
    "Before answering, check every span: if part of it is something the reporter directly "
    "saw or felt and part is their explanation for it, it is two claims, not one. Each claim "
    "gets exactly one kind."
)


def render_claim_rule_first_turn(
    messages: list[dict[str, str]],
    ticket_text: object,
) -> bool:
    """Append the confirmed report-reading rule to the first ticket-bearing user turn."""
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        return False
    if (len(messages) < 2 or messages[0].get("role") != "system"
            or messages[1].get("role") != "user"):
        raise ValueError("claim rule requires a system message followed by the first user message")
    user_text = messages[1].get("content")
    if not isinstance(user_text, str):
        raise ValueError("claim rule requires a textual first user message")
    messages[1]["content"] = user_text + "\n\n" + CLAIM_RULE_TEXT
    return True


def emit_claim_rule_rendered(journal: object) -> None:
    """Record rule placement after runtime initialization without ticket contents."""
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("claim rule render requires a host journal")
    emitter("claim_rule_rendered", placement="user_turn", version="v3")


def entry_file_only_enabled() -> bool:
    """Return whether explicitly ticket-authorized file-only hints may be rendered."""
    value = os.environ.get(ENTRY_FILE_ONLY_ENV, "on")
    if value not in {"off", "on"}:
        raise ValueError(f"{ENTRY_FILE_ONLY_ENV} must be off or on")
    return value == "on"


def system_prompt_variant() -> str:
    """Return the selected system-prompt text without changing the v1 default."""
    value = os.environ.get(SYSTEM_PROMPT_ENV, "v1")
    if value not in {"v1", "v2", "v3"}:
        raise ValueError(f"{SYSTEM_PROMPT_ENV} must be v1, v2 or v3")
    return value


def entry_hint_line(
    path: str | None,
    symbol: str,
    *,
    ticket_entry_file: bool = False,
) -> str:
    """Render a grounded entry hint, dropping blank-symbol hints by default."""
    variant = system_prompt_variant()
    file_only_enabled = entry_file_only_enabled()
    path_text = str(path or "").strip()
    symbol_text = str(symbol or "").strip()
    if variant == "v1":
        if path_text and symbol_text:
            return f"The target is {symbol_text} in {path_text}; this is the actual workspace path."
        if path_text and ticket_entry_file and file_only_enabled:
            return f"The target file is {path_text}; this is the actual workspace path."
        return ""
    if path_text and symbol_text:
        return (f"The entry point is {symbol_text} in {path_text}. It is where the call chain starts, "
                "not the only file you may need to change.")
    if path_text and ticket_entry_file and file_only_enabled:
        return (f"The entry file is {path_text}. It is where the call chain starts, "
                "not the only file you may need to change.")
    return ""


def render_blank_entry_template(
    system_template: str,
    path: str | None,
    *,
    ticket_entry_file: bool = False,
) -> str:
    """Replace React's blank-symbol entry bullet with a grounded line or drop it."""
    variant = system_prompt_variant()
    original_line = "  * The file you almost certainly need is `{REL}`, and the function is `{FN}`."
    if variant == "v1":
        template_lines = (original_line,)
    else:
        template_lines = (
            original_line,
            "  * The entry point is `{FN}` in `{REL}`. It is where the call chain starts, "
            "not the only file you may need to change.",
        )
    line = entry_hint_line(path, "", ticket_entry_file=ticket_entry_file)
    replacement = f"  * {line}" if line else ""
    for template_line in template_lines:
        system_template = system_template.replace(template_line, replacement)
    return system_template


V2_DONE_SECTION = """When you are done
- You are done when BOTH are true: every checklist line points to a file and line that implements it, and
  the relevant tests pass on the current code. Then call done immediately. Do not re-verify further, and do
  not keep deliberating once both are true.
"""

V2_GENERIC_GUIDANCE = """How to do a ticket like this

1. Turn the ticket into a checklist before editing. Every sentence can hide several requirements:
   a default, an allowed set, an error status and message, which layers it must pass through,
   what must stay compatible, what must not change. Write each as one short line in your reasoning.
   The public test usually checks only one of them. Passing it does not mean the ticket is done.

2. Follow the value through every layer the ticket names. If the ticket says "thread it through the
   model, service, API, and CLI", each of those files needs to handle it. Read the call chain from
   the entry point: CLI -> API handler -> service -> model. A value that is dropped at any layer is a bug.

3. Follow errors all the way to the caller. Raising the right exception is half the job. Check what the
   caller actually receives: if the ticket says "return HTTP 400 with the error text X", find the code that
   turns the exception into a response and make sure the status and text are exactly right.

4. Keep old calls working. If an existing call like `run(a, b)` must stay valid, add the new argument as an
   optional parameter with the stated default. Keep old names as aliases when the ticket says so.

5. Do not change what the caller passed in. Copy dicts and lists before modifying them.

6. Before calling done, check the checklist against the code, line by line. For each requirement, name
   the file and line that satisfies it. If one has no line, you are not done.

Worked example (different task, same shape):
Ticket: "Fitness class booking gains optional `level`, default "open"; "open" and "advanced" are valid;
anything else -> HTTP 422 'unknown level'; pass it through model, API and CLI; old book(class_id, member) works."
Checklist: default open | allowed {open, advanced} | unknown -> 422 + exact text | model stores it |
API passes it and maps the error to 422 | CLI book() gets an optional level param | old 2-arg call works.
A service-only change passes a "default is open" test and still fails four of these seven lines.
(Illustration only: never copy its names, values or status code into your task.)

What you may edit
- You may change ANY declared product file (model, service, API, CLI, validation). You are never limited to
  the entry point. Use str_replace for any exact change in any of those files. The edit tool is only a
  shortcut for replacing a whole function; not having a shortcut for a file does NOT mean you can't edit it.
- You may not edit the provided tests. That is the only restriction; do not spend time re-checking it.

When you are done
- You are done when BOTH are true: every checklist line points to a file and line that implements it, and
  the relevant tests pass on the current code. Then call done immediately. Do not re-verify further, and do
  not keep deliberating once both are true.

Working rules
- Make a real change as soon as you know enough; run the relevant test after each change.
- Seek and attack the root cause, not the symptom. When a check fails, find where the wrong value is first
  produced (which file, which line) and fix it there, not where the failure shows up. Read the error message
  before editing again; do not patch around it.
- After an edit, look at the whole changed region: a method must still be inside its class, decorators and
  indentation must be right.
- Do not rerun an unchanged passing check. Do not re-read a file whose current content you already have."""


# v3 = v2's reasoning guidance, rewritten for the StateFrame architecture. v2's "call done immediately"
# rule is a stopping policy; in frame the host's completion authority owns stopping, and v2's rule made
# ldev_01 quit early (batch 255). v3 hands stopping to the host and teaches the model how to read the frame.
V3_FRAME_GUIDANCE = V2_GENERIC_GUIDANCE.replace(V2_DONE_SECTION, """Reading the frame
- Each step you get a fresh frame, not a transcript. The host builds it from what actually happened:
  the current source, what you edited, which checks ran on which revision, and what is still open.
  Treat host-recorded facts as true; do not spend a step re-checking something the frame already states.
- Your own earlier notes and hypotheses in the frame are claims, not facts. Confirm them against source or a check.
- Open work and blocking items tell you what the host still sees as unfinished. Work on those first.
- One tool call per step: pick the move that changes the state the most, usually an edit or the check
  that decides between two explanations.

When you are done
- Call done when every checklist line points to a file and line that implements it and the relevant tests
  pass on the current code. The host verifies completion; if it lists something still open, address that item
  instead of re-arguing the finish.
""")


def prompt_guidance(variant: str) -> str:
    """Generic ticket guidance for the non-v1 system prompts."""
    from kernel_frame_budget import enabled as frame_budget_enabled, prompt_guidance as frame_budget_guidance
    if frame_budget_enabled():
        return frame_budget_guidance(variant)
    from kernel_frame_diet import enabled as frame_diet_enabled, guidance as frame_diet_guidance
    if frame_diet_enabled():
        return frame_diet_guidance(variant)
    return V3_FRAME_GUIDANCE if variant == "v3" else V2_GENERIC_GUIDANCE

def action_schema_enabled() -> bool:
    """Return whether the visible JSON action envelope is the active wire format."""
    value = os.environ.get(ACTION_SCHEMA_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{ACTION_SCHEMA_ENV} must be off or on")
    return value == "on"


# Keep the selectors deliberately narrow.  Arbitrary source/code strings remain opaque
# values and continue to receive host validation after decode; only values whose catalog
# is owned by the host are candidates for a per-request enum.
_PATH_FIELDS = frozenset({"path", "paths", "target_path", "file_path"})
_TEST_FIELDS = frozenset({"target", "selector", "selectors", "test_selector", "test_path"})
_REFERENCE_FIELDS = frozenset({
    "reference", "references", "receipt_ref", "receipt_id", "receipt_ids",
    "evidence_ref", "evidence_refs", "basis_refs",
})
_WORK_REFERENCE_FIELDS = frozenset({
    "work_id", "work_ids", "depends_on", "superseded_by", "goal_id", "record_id",
    "supersedes", "after", "id",
})


def _evictfix_enabled() -> bool:
    """Keep the parallel trial reversible without changing the legacy arm."""
    return evictfix_enabled()


_MISSING_MODULE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s*No module named\s+['\"]?([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
)


def environment_check_reason(row: Mapping[str, object]) -> str | None:
    """Name a host-proven test environment failure without masking source errors."""
    return _s5_environment_check_reason(row)


_PRODUCT_TRACE_FILE = re.compile(r'(?m)^\s*File ["\']([^"\']+\.py)["\'], line \d+')
_COLLECTED_FILE = re.compile(r'(?m)^ERROR collecting ([^\s]+\.py)')
_CODE_IMPORT_ERROR = re.compile(r'(?m)^(?:SyntaxError|IndentationError|TabError):')
_HOST_ENV_OUTCOMES = frozenset({
    "ENVIRONMENT", "ENVIRONMENT_BLOCKED", "HOST_ERROR", "HARNESS_ERROR",
})
_DECLARED_ENV_REASONS = frozenset({
    "test harness failed", "test import failed", "test collection or harness failed",
})


def _environment_output(row: Mapping[str, object]) -> str:
    parts = (row.get(key) for key in (
        "output", "observation", "detail", "stdout", "stderr", "test_output"))
    return "\n".join(part[:6000] for part in parts if isinstance(part, str))[:12000]


def _environment_roots(row: Mapping[str, object]) -> tuple[Path, ...]:
    """Use only host-provided source roots inside the checkout, without importing code."""
    raw_product = row.get("product_root")
    if not isinstance(raw_product, (str, os.PathLike)) or not str(raw_product):
        return ()
    product = Path(raw_product).resolve()
    roots = [product]
    supplied = row.get("source_roots")
    if supplied is None:
        selectors = row.get("selectors")
        if isinstance(selectors, Sequence) and not isinstance(selectors, (str, bytes)):
            try:
                from kernel_ship_tree_smoke import source_roots
                supplied = source_roots(product, [str(item) for item in selectors])
            except (OSError, TypeError, ValueError):
                supplied = ()
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes)):
        for raw in supplied:
            if not isinstance(raw, (str, os.PathLike)) or not str(raw):
                continue
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = product / candidate
            candidate = candidate.resolve()
            if candidate.is_relative_to(product) and candidate not in roots:
                roots.append(candidate)
    return tuple(roots)


def _path_in_product(raw: str, roots: tuple[Path, ...]) -> bool:
    path = Path(raw)
    candidates = ((path,) if path.is_absolute()
                  else tuple(root / path for root in roots))
    return any(candidate.resolve().is_relative_to(root)
               for candidate in candidates for root in roots)


def _module_in_product(module: str, roots: tuple[Path, ...]) -> bool:
    """A product package prefix makes a missing child a product import defect."""
    parts = module.split(".")
    for root in roots:
        for count in range(1, len(parts) + 1):
            stem = root.joinpath(*parts[:count])
            if stem.is_dir() or stem.with_suffix(".py").is_file():
                return True
    return False


def product_check_failure(row: Mapping[str, object]) -> bool:
    """Identify import/collection errors caused by source under the checkout."""
    if row.get("environment_classification") == "PRODUCT_FAILURE":
        return True
    output = _environment_output(row)
    declared = row.get("environment_reason")
    declared_module = (re.fullmatch(
        r"missing dependency: ([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", declared.strip())
        if isinstance(declared, str) else None)
    if not any(marker in output for marker in (
            "ModuleNotFoundError", "ImportError", "ERROR collecting ",
            "SyntaxError", "IndentationError", "TabError")) and not declared_module:
        return False
    roots = _environment_roots(row)
    if not roots:
        return False
    if declared_module and _module_in_product(declared_module.group(1), roots):
        return True
    missing = _MISSING_MODULE.search(output)
    if missing and _module_in_product(missing.group(1), roots):
        return True
    frames = _PRODUCT_TRACE_FILE.findall(output)
    product_frame = any(_path_in_product(path, roots) for path in frames)
    if _CODE_IMPORT_ERROR.search(output) and product_frame:
        return True
    # An external missing package is an environment defect even when the
    # importing test appears in the traceback or pytest's collection banner.
    if missing:
        return False
    if product_frame and ("ImportError" in output or "ERROR collecting " in output):
        return True
    return any(_path_in_product(path, roots)
               for path in _COLLECTED_FILE.findall(output))


def _s5_environment_check_reason(row: Mapping[str, object]) -> str | None:
    """Classify only host-typed failures after excluding local source defects."""
    if not _evictfix_enabled() or not isinstance(row, Mapping):
        return None
    exit_code = row.get("exit_code")
    if type(exit_code) is not int or exit_code not in {2, 3, 4, 5}:
        return None
    if product_check_failure(row):
        return None
    status = str(row.get("outcome") or row.get("canonical_outcome") or "").upper()
    if status == "BEHAVIOR_FAIL":
        return None
    declared = row.get("environment_reason")
    if isinstance(declared, str):
        declared = declared.strip()
        if declared not in _DECLARED_ENV_REASONS and not re.fullmatch(
                r"missing dependency: [A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", declared):
            declared = None
    else:
        declared = None
    typed_host = status in _HOST_ENV_OUTCOMES or declared is not None
    if exit_code in {2, 4} and not typed_host:
        return None
    output = _environment_output(row)
    if exit_code == 3 and not typed_host and "INTERNALERROR>" not in output:
        return None
    missing = _MISSING_MODULE.search(output)
    if missing:
        return f"missing dependency: {missing.group(1)[:120]}"
    # Generic import and collection prose is not proof of a host failure.
    if ("ImportError while importing test module" in output
            or re.search(r"(?:^|\n)(?:ImportError|ModuleNotFoundError):", output)
            or "ERROR collecting " in output):
        return None
    if "INTERNALERROR>" in output or "pytest: error:" in output:
        return "test collection or harness failed"
    if declared is not None:
        return declared[:160]
    return "test harness failed" if typed_host else None


def successful_lookup_status(tool: str, reason: str, observation: str) -> str:
    """Classify successful read-only navigation by its host result, not a reason tag."""
    if not _evictfix_enabled():
        return "REJECTED"
    name = str(tool).strip().lower()
    code = str(reason).strip().upper().split(":", 1)[0]
    # The host's typed lookup result wins over arbitrary matching source text.
    # A grep hit can itself begin with "ERROR" or "REJECTED".
    if name == "grep" and code in {"HITS", "NO_HITS", "RESULT_HITS", "RESULT_NO_HITS"}:
        return "APPLIED"
    if name == "where_defined" and code in {"HOST_OBSERVED", "RESULT_HOST_OBSERVED"}:
        return "APPLIED"
    return "REJECTED"


def exact_source_message_locations(
        messages: Sequence[Mapping[str, object]], expected_source: object,
) -> list[int]:
    """A missing candidate never proves a missing-content request message."""
    if not isinstance(expected_source, str):
        return []
    return [index for index, message in enumerate(messages)
            if isinstance(message, Mapping) and message.get("role") == "user"
            and message.get("content") == expected_source]


def reconcile_final_source_access(
        messages: list[dict[str, object]], delivery: Mapping[str, object],
) -> bool:
    """Bind the source-access claim to byte-proven pages in the fitted request."""
    if not _evictfix_enabled():
        return False
    paths = delivery.get("paths") if isinstance(delivery, Mapping) else None
    attached: set[str] = set()
    if delivery.get("proof_status") == "VERIFIED" and isinstance(paths, Mapping):
        attached = {
            str(path).replace("\\", "/") for path, row in paths.items()
            if isinstance(row, Mapping) and row.get("ranges")
            and row.get("final_message_locations")
        }
    from kernel_stateframe_envelope import STATE_HEADER
    prefix = STATE_HEADER + "\n"
    decoder = json.JSONDecoder()
    changed = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(prefix):
            continue
        try:
            payload, end = decoder.raw_decode(content[len(prefix):])
        except (TypeError, ValueError, RecursionError):
            continue
        if not isinstance(payload, dict):
            continue
        schema = payload.get("schema")
        if schema == "stateframe-f4/1" and isinstance(payload.get("state"), dict):
            nested = True
            state = payload["state"]
        elif schema == "stateframe-operational/1":
            nested = False
            state = payload
        else:
            continue
        access = state.get("source_access")
        if not isinstance(access, Mapping):
            continue
        path = str(access.get("path") or "").replace("\\", "/")
        included = bool(path and path in attached)
        if (access.get("included") is included
                and access.get("no_read_call_required_for_this_included_source") is included):
            continue
        state["source_access"] = {
            **access,
            "included": included,
            "no_read_call_required_for_this_included_source": included,
        }
        if nested:
            payload["state"] = state
        message["content"] = (prefix + json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + content[len(prefix) + end:])
        changed = True
    return changed


def commit_environment_notice_delivery(runtime: object,
                                       messages: object) -> bool:
    """Commit the pending notice only after a response to these exact wire messages."""
    if not _evictfix_enabled():
        return False
    pending = getattr(runtime, "_evictfix_environment_notice_pending", None)
    if not isinstance(pending, str) or not pending or not isinstance(messages, Sequence):
        return False
    expected = (f"Tests cannot run here: {pending}. "
                "Continue with source inspection and other available verification.")
    from kernel_stateframe_envelope import STATE_HEADER
    prefix = STATE_HEADER + "\n"
    decoder = json.JSONDecoder()
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(prefix):
            continue
        try:
            payload, _ = decoder.raw_decode(content[len(prefix):])
        except (TypeError, ValueError, RecursionError):
            continue
        if not isinstance(payload, Mapping):
            continue
        schema = payload.get("schema")
        if schema == "stateframe-f4/1" and isinstance(payload.get("state"), Mapping):
            state = payload["state"]
        elif schema == "stateframe-operational/1":
            state = payload
        else:
            continue
        if state.get("environment_check_notice") == expected:
            runtime._evictfix_environment_notice_pending = None
            runtime._evictfix_environment_notice_announced = True
            return True
    return False


def _known_values(known: Mapping[str, object] | None, *names: str) -> list[object]:
    """Return ordered, JSON-compatible host values without inventing selectors."""
    if not isinstance(known, Mapping):
        return []
    candidates: list[object] = []
    field_enums = known.get("field_enums")
    if isinstance(field_enums, Mapping):
        for name in names:
            if name in field_enums:
                candidates.append(field_enums[name])
    for name in names:
        if name in known:
            candidates.append(known[name])
    result: list[object] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            values = [candidate]
        elif isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes, bytearray)):
            values = list(candidate)
        else:
            continue
        for value in values:
            if isinstance(value, (str, int, float, bool)) or value is None:
                if value not in result:
                    result.append(value)
    return result


def _schema_allows_null(schema: Mapping[str, object]) -> bool:
    if isinstance(schema.get("type"), list) and "null" in schema["type"]:
        return True
    for branch_name in ("anyOf", "oneOf"):
        branches = schema.get(branch_name)
        if isinstance(branches, list) and any(
                isinstance(branch, Mapping)
                and ((branch.get("type") == "null")
                     or _schema_allows_null(branch))
                for branch in branches):
            return True
    return False


def _mutable_schema_copy(value: object) -> object:
    """Detach schema containers even when owner-frozen deepcopy retains identity."""
    if isinstance(value, Mapping):
        return {str(key): _mutable_schema_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_schema_copy(item) for item in value]
    return copy.deepcopy(value)


def _enum_values(schema: Mapping[str, object], values: Sequence[object]) -> dict:
    """Copy a scalar schema and install a host-owned enum, preserving its metadata."""
    result = _mutable_schema_copy(schema)
    existing = result.get("enum")
    if isinstance(existing, list):
        values = [value for value in values if value in existing]
    values = list(values)
    if (_schema_allows_null(result) and None not in values
            and (not isinstance(existing, list) or None in existing)):
        values.append(None)
    result["enum"] = values
    return result


def _enum_field(schema: Mapping[str, object], values: Sequence[object]) -> dict:
    """Apply a host enum to a scalar field or to the items of an array field."""
    result = _mutable_schema_copy(schema)
    if result.get("type") == "array" and isinstance(result.get("items"), Mapping):
        result["items"] = _enum_values(result["items"], values)
        return result
    return _enum_values(result, values)


def _host_reference_schema(tool_name: str, field_name: str, value: object,
                           known: Mapping[str, object] | None) -> object:
    """Install enums only for selectors whose values are already host-known."""
    if not isinstance(value, Mapping):
        return value
    result = _mutable_schema_copy(value)
    if field_name in _PATH_FIELDS and tool_name not in {
            "file_create", "create_file", "create_module", "scaffold_file"}:
        paths = _known_values(known, "paths", "observed_paths")
        if paths:
            return _enum_field(result, paths)
    if tool_name == "run_tests" and field_name in _TEST_FIELDS:
        selectors = _known_values(known, "test_selectors", "selectors", "tests")
        if selectors:
            return _enum_field(result, selectors)
    if field_name in _REFERENCE_FIELDS:
        references = _known_values(known, "references", "receipt_ids", "evidence_refs")
        if references:
            return _enum_field(result, references)
    if ((tool_name == "work" and field_name in _WORK_REFERENCE_FIELDS)
            or field_name in {"work_id", "depends_on", "superseded_by"}):
        work_ids = _known_values(known, "work_ids", "work_record_ids")
        if work_ids:
            return _enum_field(result, work_ids)
    return result


def _strict_schema(value: object, *, tool_name: str,
                   known: Mapping[str, object] | None, field_name: str = "") -> object:
    """Deep-copy a tool schema and close every object in its schema tree."""
    if not isinstance(value, Mapping):
        return _mutable_schema_copy(value)
    result = _mutable_schema_copy(value)
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        result["properties"] = {
            str(name): _strict_schema(prop, tool_name=tool_name, known=known,
                                      field_name=str(name))
            for name, prop in properties.items()
        }
        result["additionalProperties"] = False
    elif (result.get("type") == "object"
          or (isinstance(result.get("type"), list)
              and "object" in result["type"])):
        result["additionalProperties"] = False
    additional = result.get("additionalProperties")
    if isinstance(additional, Mapping):
        result["additionalProperties"] = _strict_schema(
            additional, tool_name=tool_name, known=known)
    for branch_name in ("oneOf", "anyOf", "allOf"):
        branches = result.get(branch_name)
        if isinstance(branches, list):
            result[branch_name] = [
                _strict_schema(branch, tool_name=tool_name, known=known)
                for branch in branches
            ]
    items = result.get("items")
    if isinstance(items, Mapping):
        result["items"] = _strict_schema(items, tool_name=tool_name, known=known)
    result = _host_reference_schema(tool_name, field_name, result, known)
    if field_name in {"evidence_refs", "receipt_refs", "receipt_ids", "basis_refs"} and isinstance(result.get("items"), Mapping):
        refs = _known_values(known, "receipt_ids", "evidence_refs", "references")
        if refs:
            result["items"] = _enum_values(result["items"], refs)
    if field_name in {"depends_on", "work_ids"} and isinstance(result.get("items"), Mapping):
        work_ids = _known_values(known, "work_ids", "work_record_ids")
        if work_ids:
            result["items"] = _enum_values(result["items"], work_ids)
    if field_name == "proof_steps" and isinstance(result.get("items"), Mapping):
        step_numbers = _known_values(known, "step_numbers", "receipt_steps")
        if step_numbers:
            result["items"] = _enum_values(result["items"], step_numbers)
    return result


def _action_parameters(tool: Mapping[str, object], *, known: Mapping[str, object] | None) -> dict:
    function = tool.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
        raise ValueError("action menu contains a tool without a function name")
    name = function["name"]
    parameters = function.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = {"type": "object", "properties": {}, "required": []}
    result = _strict_schema(parameters, tool_name=name, known=known)
    if not isinstance(result, dict) or result.get("type") != "object":
        raise ValueError(f"tool {name} must have an object argument schema")
    result.setdefault("properties", {})
    result.setdefault("required", [])
    result["additionalProperties"] = False
    if name == "done":
        properties = result["properties"]
        properties["claim"] = {"type": "string", "enum": list(ACTION_CLAIMS)}
        evidence_refs = _known_values(known, "receipt_ids", "evidence_refs", "references")
        evidence_items: dict[str, object] = {"type": "string", "enum": evidence_refs}
        properties["evidence_refs"] = {
            "type": "array", "maxItems": 16, "items": evidence_items,
        }
        result["required"] = list(dict.fromkeys(
            [*result.get("required", []), "claim", "evidence_refs"]))
    return result


def action_schema(tools: Sequence[Mapping[str, object]], *,
                  known: Mapping[str, object] | None = None) -> dict:
    """Build the strict visible-output union from exactly this request's live menu."""
    single_carrier_enabled = False
    carrier_tool_name = None
    schema_tools: Sequence[Mapping[str, object]] = tools
    try:
        from kernel_tool_schema_diet import (
            CARRIER_TOOL_NAME,
            enabled as schema_single_carrier_enabled,
            filter_tools as filter_single_carrier_tools,
        )
        single_carrier_enabled = schema_single_carrier_enabled()
        carrier_tool_name = CARRIER_TOOL_NAME
        if single_carrier_enabled:
            schema_tools = filter_single_carrier_tools(
                tools if isinstance(tools, list) else list(tools))
    except (ImportError, ModuleNotFoundError):
        pass
    memory_property: dict[str, object] = {}
    try:
        from kernel_decision_cache import enabled as decision_cache_enabled, memory_schema
        if decision_cache_enabled():
            memory_property = {"memory": memory_schema()}
    except (ImportError, ModuleNotFoundError):
        pass
    continuity_property: dict[str, object] = {}
    try:
        from kernel_decision_continuity import enabled as continuity_enabled, delta_schema
        if continuity_enabled():
            continuity_property = {"decision_delta": delta_schema()}
    except (ImportError, ModuleNotFoundError):
        pass
    variants = []
    names: list[str] = []
    for tool in schema_tools:
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else None
        if not isinstance(name, str) or not name:
            raise ValueError("action menu contains a tool without a function name")
        if name in names:
            raise ValueError(f"action menu contains duplicate tool: {name}")
        names.append(name)
        action_parameters = _action_parameters(tool, known=known)
        if single_carrier_enabled:
            if name == carrier_tool_name and isinstance(action_parameters.get("properties"), dict):
                action_parameters["properties"].update(memory_property)
                action_parameters["properties"].update(continuity_property)
            variant_optional_properties = {}
        else:
            variant_optional_properties = {
                **memory_property,
                **continuity_property,
            }
        variants.append({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": [name]},
                "args": action_parameters,
                **variant_optional_properties,
            },
            "required": ["action", "args"],
            "additionalProperties": False,
        })
    if not variants:
        raise ValueError("action menu is empty")
    # The root is closed as well.  Its action enum is useful to providers and
    # inspection tools, while the per-action oneOf branches preserve the
    # action-to-argument correlation instead of accepting a mismatched pair.
    root_optional_properties = {} if single_carrier_enabled else {
        **memory_property,
        **continuity_property,
    }
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": names},
            "args": {"oneOf": [variant["properties"]["args"] for variant in variants]},
            **root_optional_properties,
        },
        "required": ["action", "args"],
        "additionalProperties": False,
        "oneOf": variants,
    }


def configure_action_envelope(body: dict, tools: Sequence[Mapping[str, object]], *,
                              known: Mapping[str, object] | None = None) -> None:
    """Replace native tool dispatch with a strict, visible-content action envelope."""
    schema = action_schema(tools, known=known)
    # A forced visible schema and native tool definitions are competing provider
    # interfaces.  The action arm is deliberately content-only, even though the
    # caller still uses the live tool menu to construct the union.
    body.pop("tools", None)
    body.pop("grammar", None)
    body.update(
        tool_choice="none",
        parallel_tool_calls=False,
        response_format={"type": "json_schema", "json_schema": {
            "name": ACTION_SCHEMA_NAME, "strict": True, "schema": schema,
        }},
    )
    if "grammar" in body:
        raise AssertionError("action schema requests must not send grammar")


def _strict_json_loads(content: str) -> object:
    if not isinstance(content, str):
        raise TypeError("action content must be a string")

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(content, parse_constant=reject_constant,
                      object_pairs_hook=reject_duplicate)


def _validate_json_schema(value: object, schema: Mapping[str, object], path: str = "$") -> list[str]:
    from kernel_completion_owner import validate_json_schema
    return list(validate_json_schema(schema, value, path))


def decode_action_envelope(content: str, tools: Sequence[Mapping[str, object]], *,
                           known: Mapping[str, object] | None = None,
                           call_id: str = "action_0") -> list[dict]:
    """Decode only visible JSON and adapt it to the existing native-call shape."""
    value = _strict_json_loads(content)
    schema = action_schema(tools, known=known)
    errors = _validate_json_schema(value, schema)
    if errors:
        raise ValueError("invalid action envelope: " + "; ".join(errors[:8]))
    action = value["action"]
    arguments = value["args"]
    single_carrier_enabled = False
    try:
        from kernel_tool_schema_diet import enabled as schema_single_carrier_enabled
        single_carrier_enabled = schema_single_carrier_enabled()
    except (ImportError, ModuleNotFoundError):
        pass
    arguments_for_call = copy.deepcopy(arguments) if single_carrier_enabled else arguments
    _missing_sidecar = object()
    carrier_memory: object = _missing_sidecar
    carrier_delta: object = _missing_sidecar
    if single_carrier_enabled and isinstance(arguments_for_call, dict):
        if "memory" in arguments_for_call:
            carrier_memory = arguments_for_call.pop("memory")
        if "decision_delta" in arguments_for_call:
            carrier_delta = arguments_for_call.pop("decision_delta")
    call = {"id": call_id, "type": "function", "function": {
        "name": action, "arguments": json.dumps(arguments_for_call),
    }}
    try:
        from kernel_decision_cache import enabled as decision_cache_enabled
        if decision_cache_enabled():
            if single_carrier_enabled and carrier_memory is not _missing_sidecar:
                call["memory"] = copy.deepcopy(carrier_memory)
            elif not single_carrier_enabled and "memory" in value:
                call["memory"] = copy.deepcopy(value["memory"])
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        from kernel_decision_continuity import enabled as continuity_enabled
        if continuity_enabled():
            if single_carrier_enabled and carrier_delta is not _missing_sidecar:
                call["decision_delta"] = copy.deepcopy(carrier_delta)
            elif not single_carrier_enabled and "decision_delta" in value:
                call["decision_delta"] = copy.deepcopy(value["decision_delta"])
    except (ImportError, ModuleNotFoundError):
        pass
    return [call]


# Short aliases make the wire-format helpers easy to use from small probe tests while
# retaining the descriptive names used by the request path.
action_envelope_enabled = action_schema_enabled
action_envelope_schema = action_schema
build_action_schema = action_schema
configure_action_schema = configure_action_envelope
decode_action = decode_action_envelope


def record_action_envelope_metrics(journal: object, *, step: int,
                                   menu: Sequence[Mapping[str, object]],
                                   content: str, reasoning_content: object,
                                   action: str | None = None,
                                   arguments: object = None,
                                   schema_rejected: bool = False,
                                   outcome: str = "complete") -> None:
    """Append off-prompt degeneracy metrics to the run journal."""
    if journal is None or not hasattr(journal, "emit"):
        return
    state = getattr(journal, "_action_envelope_metrics", None)
    if not isinstance(state, dict):
        state = {"responses": 0, "valid_actions": 0, "schema_rejections": 0,
                 "first_enum_values": 0, "string_values": 0, "empty_strings": 0,
                 "empty_outputs": 0,
                 "reasoning_content_length_per_step": []}
        setattr(journal, "_action_envelope_metrics", state)
    names = [tool.get("function", {}).get("name") for tool in menu
             if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)]
    state["responses"] += 1
    state["schema_rejections"] += int(schema_rejected)
    state["valid_actions"] += int(action is not None)
    state["empty_outputs"] += int(not content)
    first = bool(action is not None and names and action == names[0])
    state["first_enum_values"] += int(first)

    def count_strings(value: object) -> None:
        if isinstance(value, str):
            state["string_values"] += 1
            state["empty_strings"] += int(value == "")
        elif isinstance(value, Mapping):
            for item in value.values():
                count_strings(item)
        elif isinstance(value, list):
            for item in value:
                count_strings(item)

    count_strings(arguments)
    reasoning = reasoning_content if isinstance(reasoning_content, str) else ""
    state["reasoning_content_length_per_step"].append({
        "step": step, "chars": len(reasoning), "bytes": len(reasoning.encode("utf-8")),
    })
    responses = state["responses"]
    valid = state["valid_actions"]
    strings = state["string_values"]
    state_view = {
        "responses": responses,
        "valid_actions": valid,
        "first_enum_value_rate": (state["first_enum_values"] / valid if valid else 0.0),
        "empty_string_rate": state["empty_outputs"] / responses,
        "argument_empty_string_rate": (state["empty_strings"] / strings if strings else 0.0),
        "schema_reject_rate": state["schema_rejections"] / responses,
        "reasoning_content_length_per_step": copy.deepcopy(
            state["reasoning_content_length_per_step"]),
    }
    journal.emit("action_envelope_metrics", step=step, action=action,
                 outcome=outcome, schema_rejected=schema_rejected,
                 visible_content_empty=not bool(content), metrics=state_view,
                 dispatch=False)


def enabled() -> bool:
    value = os.environ.get("REACT_STEP_CONTRACT", "0")
    if value not in {"0", "1"}:
        raise ValueError("REACT_STEP_CONTRACT must be 0 or 1")
    return value == "1"


def schema(tools: list[dict], state_id: str) -> dict:
    variants = []
    for tool in tools:
        function = tool["function"]
        parameters = _mutable_schema_copy(function["parameters"])
        parameters["additionalProperties"] = False
        variants.append({"type": "object", "properties": {
            "state_id": {"const": state_id}, "tool": {"const": function["name"]},
            "arguments": parameters}, "required": ["state_id", "tool", "arguments"],
            "additionalProperties": False})
    return {"anyOf": variants}


def configure(body: dict, tools: list[dict], state_id: str) -> None:
    body.update(tool_choice="none", parallel_tool_calls=False,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "kernel_step", "strict": True, "schema": schema(tools, state_id)}})


def decode(content: str, tools: list[dict], state_id: str) -> list[dict]:
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"state_id", "tool", "arguments"}:
        raise ValueError("step must have exactly state_id, tool and arguments")
    if value["state_id"] != state_id:
        raise ValueError("stale or unbound state_id")
    name, arguments = value["tool"], value["arguments"]
    functions = {tool["function"]["name"]: tool["function"] for tool in tools}
    if not isinstance(name, str) or name not in functions or not isinstance(arguments, dict):
        raise ValueError("unknown tool or non-object arguments")
    properties = functions[name]["parameters"].get("properties", {})
    if set(arguments) - set(properties):
        raise ValueError("unknown argument")
    from kernel_completion_owner import validate_tool_arguments
    errors = list(validate_tool_arguments(tools, name, arguments))
    if errors:
        raise ValueError("invalid arguments: " + "; ".join(errors))
    return [{"id": "step_" + state_id.removeprefix("sha256:")[:24], "type": "function",
             "function": {"name": name, "arguments": json.dumps(arguments)}}]


INSTRUCTIONS = """
Step contract: choose exactly one provided tool and return only the schema-constrained
JSON object with state_id, tool, and arguments. The server supplies the required state_id.
Do not output shell commands, XML, Markdown fences, or a second action. Tools and paths
are validated before execution. A rejected action does not change the workspace.
The optional why argument records your current unverified hypothesis or next objective.
Use str_replace for changes inside classes or functions; edit replaces a complete top-level function or class.
"""

FRAME_INSTRUCTIONS = """
StateFrame contract: each request is a fresh decision with no conversation history.
The original requirements, current StateFrame, and Working source are your working state.
Working source is the current file and counts as having read that file. Do not reread it
merely to orient yourself. Earlier reads and edit intentions are historical and may be stale;
the included Working source is current. A passing check is valid only for the source and tests
it actually checked; edits invalidate that evidence.
Use the recorded observations and open work to select one useful action. If sufficient
source and requirements are present, implement the next change and test it. After a failed
test, use its evidence to revise the code or hypothesis. Declare done only when the public
requirements are met and current relevant checks passed. Private grading remains unknown.
"""

STATEFRAME_V2_CONTRACT = (
    "StateFrame contract: you work from the current state shown below instead of a conversation transcript.\n"
    "It already contains the requirements, the current source and what has been done and checked. Continue from it:\n"
    "do not restate the ticket or re-trace the code you can already see; decide the next useful action."
)


def frame_instructions() -> str:
    """Return the arm-specific contract, preserving the v1 text by default."""
    if system_prompt_variant() == "v1":
        return FRAME_INSTRUCTIONS
    return FRAME_INSTRUCTIONS.replace(
        "StateFrame contract: each request is a fresh decision with no conversation history.",
        STATEFRAME_V2_CONTRACT,
    )


PROCEDURE_GUIDANCE = (
    "Before editing, identify the behavior that must change and what must remain true. "
    "Choose a small example that distinguishes the intended behavior from a plausible wrong implementation. "
    "After editing, check that example against the actual code, using available tools when useful. "
    "Passing supplied tests establishes only what those tests cover. "
    "Adapt this method to the task; no separate planning or note-writing turn is required."
)


def working_view_mode() -> str:
    """Return the strictly validated live working-view mode."""
    value = os.environ.get("KERNEL_WORKING_VIEW", "off")
    if value not in {"off", "facts", "resolved", "actions", "decision", "findings"}:
        raise ValueError("KERNEL_WORKING_VIEW must be off, facts, resolved, actions, decision or findings")
    return value


def operating_instructions(
    path: str | None,
    symbol: str,
    steps: int | None,
    *,
    ticket_entry_file: bool = False,
) -> str:
    variant = system_prompt_variant()
    entry_line = entry_hint_line(path, symbol, ticket_entry_file=ticket_entry_file)
    guidance = os.environ.get("STATEFRAME_BUDGET_GUIDANCE", "0")
    if guidance not in {"0", "1"}:
        raise ValueError("STATEFRAME_BUDGET_GUIDANCE must be 0 or 1")
    if variant == "v1":
        instructions = f'''Complete the supplied Python task in the isolated workspace.
The repository root is your base directory. Use relative paths without a leading slash.
{entry_line}
Use the provided tools to inspect files, make changes, run pytest checks, and finish with done.
Choose one tool per decision. {step_guidance(steps, decisions=True)}
Current Working source, when supplied, counts as reading that file before editing.
Use str_replace for a change inside a class or function; its old_str must match exactly once.
The edit tool replaces a complete top-level function or class. file_create creates new files.
run_tests accepts a pytest path or selector, not shell commands or package installations.
The public task requirements define the objective. Existing acceptance tests cannot be edited.
Make a testable change once the requirements and relevant source are sufficient.
Use failing checks to identify what to change. Test the changed revision with relevant public checks.
Do not rerun an unchanged passing check without a specific reason.
Review requirements as well as tests. Call done when the task is satisfied and current checks pass.
Passing public checks does not reveal the private grade. Intentions and hypotheses are not proof.
If blocked, preserve the concrete failure and change the approach rather than repeat it.'''
    else:
        instructions = f'''Complete the supplied Python task in the isolated workspace.
The repository root is your base directory. Use relative paths without a leading slash.
{entry_line}
Use the provided tools to inspect files, make changes, run pytest checks, and finish with done.
Choose one tool per decision. {step_guidance(steps, decisions=True)}
Current Working source, when supplied, counts as reading that file before editing.
Use str_replace for a change inside a class or function; its old_str must match exactly once.
The edit tool replaces a complete top-level function or class. file_create creates new files.
run_tests accepts a pytest path or selector, not shell commands or package installations.
{prompt_guidance(variant)}'''
    if guidance == "1":
        instructions += (
            "\nThere is no step limit. Continue while making useful progress. "
            if steps is None else
            "\nYour budget is a ceiling, not a target. You do not need to use it up. "
        ) + "When the requirements are satisfied and current checks pass, call done."
    procedure = os.environ.get("STATEFRAME_PROCEDURE_GUIDANCE", "0")
    if procedure not in {"0", "1"}:
        raise ValueError("STATEFRAME_PROCEDURE_GUIDANCE must be 0 or 1")
    if procedure == "1":
        instructions += "\n" + PROCEDURE_GUIDANCE
    working_view_mode()
    return instructions


REASONING_EFFORTS = frozenset({'none', 'low', 'medium', 'high', 'xhigh'})


# 10-08 CONFIRMED_BUILD (FRAMELAB, pooled 100 bands): a 4k thinking budget matches unbounded on correctness (benchmark 57 vs
# 57%, benchmark 79 vs 80%, benchmark 94 vs 95%) and caps the runaway tail (max 17.7k tok / 363 s -> 4.7k / 161 s); thinking OFF
# loses 18 points on benchmark. ONE value: REACT_REASONING_BUDGET_TOKENS overrides it (OPERATOR2's THINK_CAP_LIVE can move it).
DEFAULT_REASONING_BUDGET_TOKENS = 4096


def configure_reasoning_budget(body: dict) -> None:
    """Apply explicit reasoning settings before the final wire guard.

    REACT_REASONING_EFFORT is the OpenAI-compatible provider knob (DeepInfra DeepSeek V4
    ignores llama.cpp's reasoning_budget_tokens and runs with thinking OFF unless it
    receives reasoning_effort; measured 09-25).  Never 'max' (MC rule #45).

    An explicit global OFF profile wins over optional capacity overrides.  The
    final provider boundary repeats this transformation after all routers and
    budgets, so this helper is construction convenience rather than proof.
    """
    import os
    from kernel_launch_identity_cost import enforce_provider_body, reasoning_profile

    profile = reasoning_profile()
    if profile.off:
        transformed = enforce_provider_body(body, profile=profile)
        body.clear()
        body.update(transformed)
        return
    effort = os.environ.get('REACT_REASONING_EFFORT')
    if effort is not None:
        if effort not in REASONING_EFFORTS:
            raise ValueError('REACT_REASONING_EFFORT must be one of none|low|medium|high|xhigh')
        body['reasoning_effort'] = effort
    raw = os.environ.get('REACT_REASONING_BUDGET_TOKENS')
    budget = DEFAULT_REASONING_BUDGET_TOKENS if raw is None else int(raw)
    if budget < -1:
        raise ValueError('REACT_REASONING_BUDGET_TOKENS must be -1 or nonnegative')
    body['reasoning_budget_tokens'] = budget


def live_contract(source: str, symbol: str, path: str, frame: dict,
                  budget: dict, step_limit: int | None, step: int) -> dict:
    """Operational facts for the next decision, without offline replay prerequisites."""
    from kernel_tool_applicability import target_kind as classify_target
    from kernel_sweep_fixes import (
        completion_projection, enabled as sweep_fixes_enabled,
        phase_for_completion,
    )
    target_kind = classify_target(source, symbol)
    sweep_on = sweep_fixes_enabled()
    atomic = completion_projection(frame) if sweep_on else None
    verified = (atomic.get("accepted") is True if atomic is not None
                else frame.get('valid_product_verification') is True)
    edits = frame.get('recent_changes', [])
    completion_contract = {
        'required_public_checks': frame.get('required_public_checks', []),
        'current_public_checks': frame.get('verification'),
        'requirements_review': 'Check the public objective; tests may cover only part of it.',
        'stop_action': 'done', 'automatic_success_stop': False,
    }
    if atomic is not None:
        completion_contract['atomic_completion'] = atomic
        completion_contract['completion_authority'] = 'completion_reducer'
    phase = (phase_for_completion(atomic) if atomic is not None else
             'review_requirements_then_done' if verified else
             'verify_or_repair_current_change' if edits else 'make_a_testable_change')
    verified_paths = frame.get("verified_attached_source_paths")
    included = (True if not _evictfix_enabled() else
                isinstance(verified_paths, (list, tuple, set, frozenset))
                and path.replace("\\", "/") in verified_paths)
    result = {'source_access': {'path': path, 'included': included, 'target_kind': target_kind,
                                'no_read_call_required_for_this_included_source': included},
              'action_prerequisites': {
                  'str_replace': 'Use a unique exact old_str from the current source.',
                  'edit': ('Available for this top-level function.' if target_kind == 'top_level_function'
                           else 'Available for this qualified function.'
                           if target_kind == 'qualified_function'
                           else 'Available for this top-level class.'
                           if target_kind == 'class'
                           else 'Not applicable to this target. Use str_replace; edit accepts top-level functions or classes only.'),
                  'run_tests': 'Use the explicit public check paths listed in completion_contract.'},
              'completion_contract': completion_contract,
              'phase': phase,
              'budget': {'decisions_remaining': remaining(step_limit, step),
                         'enforced': budget['enforced'],
                         'remaining': {('additional_attempts' if key == 'attempts' else key):
                                       remaining(value, budget['used'].get(key, 0))
                                       for key, value in budget['limits'].items()}},
              'pending': {'concurrent_tool_execution': False,
                          'last_action_result': (frame.get('last_action') or {}).get('result', 'none')},
              'recent_change_verification': ('completion_reducer_accepted' if atomic is not None and verified
                                             else 'completion_reducer_not_accepted' if atomic is not None
                                             else 'current_required_public_checks_passed' if verified
                                             else 'not_currently_verified')}
    from kernel_completion_status import (
        completion_status_from_frame, enabled as blocker_render_enabled,
        suppress_conflicting_completion,
    )
    has_reducer_projection = any(
        frame.get(key) is not None
        for key in ("completion_reducer_decision", "completion_decision", "atomic_completion",
                    "completion_status")
    )
    if blocker_render_enabled() and has_reducer_projection:
        result = suppress_conflicting_completion(
            result, completion_status_from_frame(frame))
    return result


def operational_view(frame: dict) -> dict:
    """Only decision-relevant information; full provenance stays in the audit frame."""
    guidance = os.environ.get("STATEFRAME_RESUME_GUIDANCE", "0")
    if guidance not in {"0", "1"}:
        raise ValueError("STATEFRAME_RESUME_GUIDANCE must be 0 or 1")
    source = frame.get('source_access', {})
    result = {key: frame.get(key) for key in (
        'phase', 'source_access', 'action_prerequisites', 'last_action', 'recent_changes',
        'recent_changes_omitted', 'recent_change_verification', 'completion_contract', 'budget', 'pending',
        'recent_test_feedback', 'recent_test_feedback_omitted', 'completion_verdict',
        'recent_negative_actions', 'recent_negative_actions_omitted')}
    if 'environment_check_notice' in frame:
        result['environment_check_notice'] = frame['environment_check_notice']
    if 'atomic_completion' in frame:
        result['atomic_completion'] = copy.deepcopy(frame['atomic_completion'])
    result['schema'] = 'stateframe-operational/1'
    if guidance == "1" and isinstance(frame.get("resume"), dict):
        result["resume"] = copy.deepcopy(frame["resume"])
        result["resume"]["phase"] = frame.get("phase")
    if 'decision_state' in frame:
        result['decision_state'] = frame['decision_state']
    if 'retained_context' in frame:
        result['retained_context'] = frame['retained_context']
    if 'frame_series' in frame:
        result['frame_series'] = frame['frame_series']
    for key in ('pattern_cards', 'portfolio'):
        if key in frame:
            result[key] = copy.deepcopy(frame[key])
    if 'green_lock_message' in frame:
        result['green_lock_message'] = frame['green_lock_message']
    # Host closure is opt-in.  Do not add even an empty field to the OFF
    # operational view, preserving the existing provider payload byte shape.
    if 'host_closure' in frame:
        result['host_closure'] = copy.deepcopy(frame['host_closure'])
    if 'repair_episode' in frame:
        result['repair_episode'] = copy.deepcopy(frame['repair_episode'])
    if 'boot_packet' in frame:
        result['boot_packet'] = copy.deepcopy(frame['boot_packet'])
    if frame.get('temporary_tool_restriction'):
        result['temporary_tool_restriction'] = frame['temporary_tool_restriction']
    result['objective'] = frame.get('objective', {}).get('public_requirements', frame.get('objective'))
    result['earlier_other_file_reads'] = [record for record in frame.get('recent_reads', [])
                                         if record['path'] != source.get('path')][-2:]
    from kernel_model_profile import hypothesis_filter_relaxed
    hypotheses = frame.get('hypotheses', [])
    if hypothesis_filter_relaxed():
        result['hypotheses_not_facts'] = [
            {'claim': item.get('claim'), 'status': item.get('status', 'unverified')}
            for item in hypotheses if isinstance(item, Mapping)]
    else:
        result['hypotheses_not_facts'] = [{'claim': item.get('claim'), 'status': item.get('status', 'unverified')}
                                         for item in hypotheses[-3:]]
    from kernel_completion_status import (
        completion_status_from_frame, enabled as blocker_render_enabled,
        suppress_conflicting_completion,
    )
    if blocker_render_enabled():
        result = suppress_conflicting_completion(
            result, completion_status_from_frame(frame))
    if os.environ.get("STATEFRAME_FRAME_CLARITY", "off") == "on":
        from kernel_frame_clarity import project_model_frame
        return project_model_frame(result)
    return result


EDIT_NTURN_ENV = "EDIT_NTURN"
EDIT_NTURN_SCOPE_TURNS_ENV = "STATEFRAME_EDIT_NTURN_SCOPE_TURNS"
EDIT_NTURN_VERSION = "nturn_bare_v1"
EDIT_NTURN_SCOPE_TURNS_VERSION = "scope_turns_aha10_three_site_v1"
EDIT_NTURN_INSTRUCTION = "Continue the edit."
# An explicit request for no further edits must not invalidate a correct result.
# prose with no done. The host never asks for a magic string; it says what to call. The old reply is still
# recognised by is_edit_nturn_no_change_reply so an episode ends cleanly either way.
EDIT_NTURN_NO_CHANGE_LINE = (
    "If no further edits are needed, call done."
)
EDIT_NTURN_NO_CHANGE_RESPONSE = "No further edits needed."
EDIT_NTURN_FIXED_CONTINUATIONS = 2
EDIT_NTURN_SITE_SOURCES = frozenset({
    "model_plan", "localize_result", "scope_receipt", "str_replace_targets",
    "model_action_targets", "fixed_horizon",
})
EDIT_NTURN_RUN_BUDGET_SECONDS = 1800.0
EDIT_NTURN_EDIT_TURN_SECONDS = 12.0
EDIT_NTURN_DEFAULT_VERIFICATION_RESERVE_SECONDS = 300.0

_EDIT_NTURN_TEXT_PATH = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"(?P<path>(?:\.{0,2}/)?(?:[A-Za-z0-9_.@+-]+/)*"
    r"[A-Za-z0-9_.@+-]+\.[A-Za-z][A-Za-z0-9_+-]*)"
    r"(?::[0-9]+(?::[0-9]+)?)?"
    r"(?![A-Za-z0-9_./-])"
)
_EDIT_NTURN_PLAN_SIGNAL = re.compile(
    r"\b(?:plan|site|file|change|edit|fix|modify|update|replace|create|add|remove|touch|target)\w*\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EditNTurnPlanSite:
    """One public plan site, retaining its display owner and repository path."""

    owner: str
    path: str
    symbol: str = ""


@dataclass(frozen=True)
class EditNTurnSiteResolution:
    """Safe, model-visible site handoff metadata in the source plan's order."""

    sites: tuple[str, ...]
    source_kind: str
    record_id: str
    parsed_count: int
    accepted_count: int
    rejection_reason: str
    plan_sites: tuple[EditNTurnPlanSite, ...] = ()


_REASONING_OWNER = re.compile(r"(?i)^(?:i|i'm|i'll|we|let me|let's|need to)\b|\b(?:of|the|a|an|in|to|for|with|on|at|and|or)$")


def _normalize_edit_site_owner(value: object) -> str:
    if not isinstance(value, str):
        return ""
    owner = " ".join(value.split()).strip("`\"'<>[]{}(),;:.-")
    return owner[:160]


def _merge_edit_site_records(
    *groups: Sequence[EditNTurnPlanSite],
) -> tuple[EditNTurnPlanSite, ...]:
    """Deduplicate plan sites by public owner/path while retaining first-seen order."""
    sites: list[EditNTurnPlanSite] = []
    indexes: dict[tuple[str, str], int] = {}
    for group in groups:
        for site in group:
            key = (site.path, site.owner.casefold())
            index = indexes.get(key)
            if index is None:
                indexes[key] = len(sites)
                sites.append(site)
            elif not sites[index].symbol and site.symbol:
                sites[index] = EditNTurnPlanSite(site.owner, site.path, site.symbol)
    return tuple(sites)


def _structured_edit_site_records(
    value: object,
    *,
    keys: tuple[str, ...],
) -> tuple[EditNTurnPlanSite, ...]:
    """Read ordered owner/path rows from an explicit public plan or receipt."""
    found: list[EditNTurnPlanSite] = []

    def add(path_value: object, owner_value: object = None, symbol_value: object = None) -> None:
        path = _normalize_edit_site(path_value)
        if path is None:
            return
        symbol = _normalize_edit_site_owner(symbol_value)
        owner = _normalize_edit_site_owner(owner_value) or symbol
        found.append(EditNTurnPlanSite(owner=owner, path=path, symbol=symbol))

    def visit(item: object) -> None:
        if isinstance(item, EditNTurnPlanSite):
            add(item.path, item.owner, item.symbol)
            return
        if isinstance(item, str):
            add(item)
            return
        if isinstance(item, Mapping):
            path_value = next((item.get(key) for key in (
                "path", "file", "target_path", "relative_path", "rel")
                if isinstance(item.get(key), str)), None)
            if path_value is not None:
                owner_value = next((item.get(key) for key in (
                    "label", "owner", "symbol", "name", "title", "site_name")
                    if isinstance(item.get(key), str) and item.get(key).strip()), None)
                add(path_value, owner_value, item.get("symbol"))
            for nested_key in keys:
                if nested_key in item:
                    visit(item.get(nested_key))
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                visit(child)

    visit(value)
    return _merge_edit_site_records(found)


def _resolve_public_plan_paths(
    sites: Sequence[EditNTurnPlanSite],
    repository_root: object,
) -> tuple[EditNTurnPlanSite, ...]:
    """Resolve public plan paths within the active workspace; drop ambiguous basenames."""
    if repository_root is None:
        return tuple(sites)
    from pathlib import Path
    from kernel_tool_paths import AmbiguousToolPath, resolve_tool_path

    try:
        root = Path(repository_root).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return ()
    if not root.is_dir():
        return ()

    resolved: list[EditNTurnPlanSite] = []
    for site in sites:
        try:
            target = resolve_tool_path(root, site.path)
            if not target.exists() and "/" not in site.path:
                continue
            relative_path = target.resolve().relative_to(root).as_posix()
        except (AmbiguousToolPath, OSError, RuntimeError, ValueError):
            continue
        resolved.append(EditNTurnPlanSite(
            owner=site.owner, path=relative_path, symbol=site.symbol,
        ))
    return _merge_edit_site_records(resolved)


def _model_plan_site_records(*texts: object) -> tuple[EditNTurnPlanSite, ...]:
    """Extract public owner labels and paths from model-authored plan lines."""
    candidates: list[tuple[EditNTurnPlanSite, bool]] = []
    for text in texts:
        if not isinstance(text, str):
            continue

        lines = text.splitlines()
        plan_heading = next((index for index, line in enumerate(lines)
                             if line.strip().strip("#*").strip().rstrip(":").strip().casefold()
                             in {"plan", "implementation plan", "edit plan", "change plan"}), None)
        if plan_heading is not None:
            end = next((index for index in range(plan_heading + 1, len(lines))
                        if re.match(r"^\s*(?:#{1,6}\s+|\*\*[^*].+\*\*\s*:?)\s*$",
                                    lines[index])), len(lines))
            lines = lines[plan_heading + 1:end]

        for line in lines:
            explicit_row = bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line))
            if not explicit_row and not _EDIT_NTURN_PLAN_SIGNAL.search(line):
                continue
            for match in _EDIT_NTURN_TEXT_PATH.finditer(line):
                path = _normalize_edit_site(match.group("path"))
                if path is None:
                    continue
                label = line[:match.start()]
                label = re.sub(r"^\s*(?:[-*+]\s*|\d+[.)]\s*)", "", label)
                label = label.replace("`", "").strip()
                label = re.sub(
                    r"^\s*(?:(?:I\s+will|please)\s+)?"
                    r"(?:handle|edit|update|modify|fix|replace|change|create|add|remove|touch)\s+",
                    "", label, flags=re.IGNORECASE,
                )
                label = re.sub(r"\s+(?:in|at|from|within|on)\s*$", "", label,
                               flags=re.IGNORECASE)
                # A code symbol is the clearest owner on explicit plan rows. Numbered
                # plan steps often put their target before or after the path.
                if explicit_row:
                    path_basename = path.rsplit("/", 1)[-1]
                    code_labels = []
                    for value in re.findall(r"`([^`]+)`", line):
                        code_path = _normalize_edit_site(value)
                        if (code_path is not None
                                and code_path.rsplit("/", 1)[-1] == path_basename):
                            continue
                        code_labels.append(value.strip())
                    if code_labels:
                        label = re.split(r"\s*(?:=|\s+as\s+)\s*", code_labels[0], maxsplit=1)[0]
                if not label or label.casefold() in {"in", "at", "from", "within", "on"}:
                    # Suffix labels are accepted only when the public plan marks them
                    # with punctuation such as an em dash or colon.
                    suffix = line[match.end():].strip(" `\t")
                    suffix_match = re.match(r"^(?:[-—–:|]\s*)(.+)$", suffix)
                    label = suffix_match.group(1) if suffix_match else ""
                owner = _normalize_edit_site_owner(label)
                candidates.append((EditNTurnPlanSite(owner=owner, path=path), explicit_row))

    # Repeated prose mentions are not additional sites. Explicit numbered/bulleted
    # plan rows can name distinct owners in one file, so retain those in source order.
    explicit_paths = {site.path for site, explicit in candidates if explicit}
    ordered: list[EditNTurnPlanSite] = []
    seen_path_owners: set[tuple[str, str]] = set()
    seen_prose_paths: set[str] = set()
    for site, explicit in candidates:
        if explicit:
            key = (site.path, site.owner.casefold())
            if key not in seen_path_owners:
                seen_path_owners.add(key)
                ordered.append(site)
        elif site.path not in explicit_paths and site.path not in seen_prose_paths:
            seen_prose_paths.add(site.path)
            ordered.append(site)
    return tuple(ordered)


def _normalize_edit_site(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().strip("`\"'<>[]{}(),;")
    candidate = re.sub(r":\d+(?::\d+)?$", "", candidate)
    candidate = candidate.replace("\\", "/")
    if not candidate or candidate.startswith("/") or re.match(r"^[A-Za-z]:", candidate):
        return None
    parts = [part for part in candidate.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return None
    normalized = "/".join(parts)
    if normalized.casefold() in {".", "..", "diagnostic", "workspace", "full"}:
        return None
    return normalized


def parse_model_plan_sites(*texts: object) -> tuple[str, ...]:
    """Extract generic file paths only from model-authored planning/action lines."""
    return tuple(dict.fromkeys(
        site.path for site in _model_plan_site_records(*texts)))


def parse_model_plan_site_records(*texts: object) -> tuple[EditNTurnPlanSite, ...]:
    """Extract the ordered public owner/path rows used by scope-turn rendering."""
    return _model_plan_site_records(*texts)


def _structured_edit_sites(value: object, *, keys: tuple[str, ...]) -> tuple[str, ...]:
    """Read only typed path/site fields from model or worker-produced records."""
    found: list[str] = []
    seen: set[str] = set()

    def add(candidate: object) -> None:
        if isinstance(candidate, str):
            site = _normalize_edit_site(candidate)
            if site is not None and site not in seen:
                seen.add(site)
                found.append(site)
        elif isinstance(candidate, Mapping):
            for path_key in ("path", "file", "target_path", "relative_path", "rel"):
                if path_key in candidate:
                    add(candidate.get(path_key))
            for nested_key in keys:
                if nested_key in candidate:
                    add(candidate.get(nested_key))
        elif isinstance(candidate, Sequence) and not isinstance(
                candidate, (str, bytes, bytearray)):
            for item in candidate:
                add(item)

    add(value)
    return tuple(found)


def _has_oracle_prefix(value: object, _seen: set[int] | None = None) -> bool:
    if _seen is None:
        _seen = set()
    identity = id(value)
    if identity in _seen:
        return False
    _seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            key == "oracle_prefix" or _has_oracle_prefix(item, _seen)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_oracle_prefix(item, _seen) for item in value)
    return False


def _receipt_record_id(value: object, default: str) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in reversed(value):
            candidate = _receipt_record_id(item, "")
            if candidate:
                return candidate
        return default
    if isinstance(value, Mapping):
        for key in ("record_id", "receipt_id", "event_id", "request_id"):
            candidate = value.get(key)
            if isinstance(candidate, (str, int)) and str(candidate).strip():
                return str(candidate)
        receipt = value.get("receipt")
        if isinstance(receipt, Mapping):
            for key in ("record_id", "receipt_id", "event_id", "request_id"):
                candidate = receipt.get(key)
                if isinstance(candidate, (str, int)) and str(candidate).strip():
                    return str(candidate)
    return default


def resolve_edit_nturn_sites(
    *,
    repository_root: object = None,
    model_plan_text: object = None,
    model_plan_sites: object = None,
    model_plan_record_id: object = None,
    localization_result: object = None,
    scope_receipt: object = None,
    model_action_targets: object = None,
    str_replace_targets: object = None,
    live_config: object = None,
    record_id: object = None,
) -> EditNTurnSiteResolution:
    """Resolve only public model/worker site evidence, preserving plan order and labels."""
    default_record_id = str(record_id or "")
    text_plan_rows = _resolve_public_plan_paths(
        _model_plan_site_records(model_plan_text), repository_root,
    )
    structured_plan_rows = _resolve_public_plan_paths(
        _structured_edit_site_records(
            model_plan_sites,
            keys=("sites", "planned_sites", "files", "targets", "units"),
        ),
        repository_root,
    )
    structured_paths = {site.path for site in structured_plan_rows}
    plan_rows = _merge_edit_site_records(
        structured_plan_rows,
        tuple(site for site in text_plan_rows if site.path not in structured_paths),
    )
    groups = (
        ("model_plan", plan_rows, str(model_plan_record_id or default_record_id)),
        ("localize_result", _structured_edit_site_records(
            localization_result if isinstance(localization_result, Mapping)
            and localization_result.get("status") == "SEARCHED" else None,
            keys=("localized_sites", "planned_sites", "sites", "suspects", "units",
                  "files", "paths", "targets"),
        ), _receipt_record_id(localization_result, default_record_id)),
        ("scope_receipt", _structured_edit_site_records(
            scope_receipt,
            keys=("scope", "sites", "paths", "targets", "files", "changed_paths"),
        ), _receipt_record_id(scope_receipt, default_record_id)),
        ("str_replace_targets", _structured_edit_site_records(
            str_replace_targets,
            keys=("edits", "targets", "files", "sites", "paths"),
        ), _receipt_record_id(str_replace_targets, default_record_id)),
        ("model_action_targets", _structured_edit_site_records(
            model_action_targets,
            keys=("edits", "targets", "files", "sites", "paths"),
        ), _receipt_record_id(model_action_targets, default_record_id)),
    )
    oracle_rejected = (_has_oracle_prefix(live_config)
                       or _has_oracle_prefix(model_plan_sites)
                       or _has_oracle_prefix(localization_result)
                       or _has_oracle_prefix(scope_receipt)
                       or _has_oracle_prefix(model_action_targets)
                       or _has_oracle_prefix(str_replace_targets))
    populated = [
        (kind, rows, source_id)
        for kind, rows, source_id in groups
        if rows and not _has_oracle_prefix(
            model_plan_sites if kind == "model_plan" else
            localization_result if kind == "localize_result" else
            scope_receipt if kind == "scope_receipt" else
            str_replace_targets if kind == "str_replace_targets" else model_action_targets)
    ]
    preferred = next((item for item in populated if len(item[1]) >= 2), None)
    if preferred is None and populated:
        source_kind, _source_rows, source_id = populated[0]
        rows = _merge_edit_site_records(*(item[1] for item in populated))
    elif preferred is not None:
        source_kind, rows, source_id = preferred
    elif oracle_rejected:
        source_kind, rows, source_id = "oracle_prefix", (), default_record_id
    else:
        source_kind, rows, source_id = "unknown", (), default_record_id

    rejection_reason = (
        "oracle_source_rejected" if oracle_rejected and not rows else
        "no_model_sites" if not rows else
        "insufficient_sites" if len(rows) < 2 else ""
    )
    parsed_count = sum(len(group_rows) for _kind, group_rows, _source_id in populated)
    return EditNTurnSiteResolution(
        sites=tuple(site.path for site in rows),
        source_kind=source_kind,
        record_id=source_id,
        parsed_count=parsed_count,
        accepted_count=len(rows),
        rejection_reason=rejection_reason,
        plan_sites=tuple(rows),
    )


def emit_edit_nturn_sites_resolved(
    journal: object, resolution: EditNTurnSiteResolution,
) -> None:
    """Record one bounded site-source receipt without persisting site names."""
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("edit n-turn site resolution requires a host journal")
    emitter(
        "edit_nturn_sites_resolved",
        source_kind=resolution.source_kind,
        record_id=resolution.record_id,
        parsed_count=resolution.parsed_count,
        accepted_count=resolution.accepted_count,
        rejection_reason=resolution.rejection_reason,
    )


def edit_nturn_enabled() -> bool:
    """Read the opt-in cumulative EDIT switch; the default preserves current runs."""
    # Enable incremental step contracts by default after behavioral verification.
    # 10/12 incl. a clean band; expected no-op at N=1. EDIT_NTURN=off remains the single kill switch.
    value = os.environ.get(EDIT_NTURN_ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{EDIT_NTURN_ENV} must be on or off")
    return value == "on"


def edit_nturn_scope_turns_enabled() -> bool:
    """Enable measured public-owner labels by default; off keeps the same bare horizon."""
    value = os.environ.get(EDIT_NTURN_SCOPE_TURNS_ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{EDIT_NTURN_SCOPE_TURNS_ENV} must be on or off")
    return value == "on"


def planned_edit_sites(*sources: object) -> tuple[object, ...]:
    """Compatibility helper for non-live callers; never reads oracle-derived sites."""
    sites: list[str] = []
    for source in sources:
        if isinstance(source, Mapping) and _has_oracle_prefix(source):
            continue
        sites.extend(_structured_edit_sites(
            source,
            keys=("localized_sites", "planned_sites", "sites", "suspects", "units"),
        ))
    return tuple(dict.fromkeys(sites))


def edit_nturn_site_count(*sources: object) -> int:
    """Count only sites named by an explicit worker-visible plan/localization."""
    return len(planned_edit_sites(*sources))


def edit_nturn_verification_reserve_seconds(task: object = None) -> float:
    """Read a run's explicit wall-clock verification reserve or use the safe default."""
    value = None
    if isinstance(task, Mapping):
        value = task.get("verification_reserve_seconds")
        if value is None:
            value = task.get("finish_reserve_seconds")
    if value is None:
        return EDIT_NTURN_DEFAULT_VERIFICATION_RESERVE_SECONDS
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("verification_reserve_seconds must be a nonnegative number") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("verification_reserve_seconds must be a nonnegative number")
    return seconds


def edit_nturn_required_seconds(
    verification_reserve_seconds: float,
    *,
    edit_turn_seconds: float = EDIT_NTURN_EDIT_TURN_SECONDS,
) -> float:
    """Budget two measured edit-turn durations plus the run's verification reserve."""
    reserve = float(verification_reserve_seconds)
    turn = float(edit_turn_seconds)
    if (not math.isfinite(reserve) or reserve < 0
            or not math.isfinite(turn) or turn < 0):
        raise ValueError("edit n-turn budget inputs must be finite and nonnegative")
    return (turn * 2.0) + reserve


def edit_nturn_budget_allows(
    remaining_seconds: float | None,
    verification_reserve_seconds: float,
    *,
    edit_turn_seconds: float = EDIT_NTURN_EDIT_TURN_SECONDS,
) -> bool:
    """Fail closed unless the next turn and verification reserve fit before the cap."""
    if remaining_seconds is None:
        return False
    try:
        remaining = float(remaining_seconds)
    except (TypeError, ValueError):
        return False
    return (math.isfinite(remaining) and remaining >= edit_nturn_required_seconds(
        verification_reserve_seconds, edit_turn_seconds=edit_turn_seconds))


def is_edit_action_line(line: object) -> bool:
    """Recognize the loop's supported product-edit action spellings."""
    if not isinstance(line, str) or not line.strip():
        return False
    return line.strip().split(maxsplit=1)[0] in {
        "edit", "str_replace", "insert_before", "file_create", "replace_module",
    }


def edit_action_applied(line: object, observation: object) -> bool:
    """Recognize a successful edit observation from the normal dispatch path."""
    if not is_edit_action_line(line) or not isinstance(observation, str):
        return False
    first_line = observation.splitlines()[0].strip().lower() if observation else ""
    return "applied." in first_line or first_line.startswith("created ")


def start_edit_nturn(enabled: bool, n_sites: int) -> "EditNTurnSession | None":
    """Start only for an enabled, explicitly multi-site plan."""
    if type(n_sites) is not int or n_sites < 1:
        raise ValueError("n_sites must be a positive integer")
    if not enabled or n_sites < 2:
        return None
    return EditNTurnSession(n_sites, site_source="model_plan")


def start_edit_nturn_episode(
    enabled: bool,
    journal: object,
    *,
    repository_root: object = None,
    model_plan_text: object = None,
    model_plan_sites: object = None,
    model_plan_record_id: object = None,
    localization_result: object = None,
    scope_receipt: object = None,
    model_action_targets: object = None,
    str_replace_targets: object = None,
    live_config: object = None,
    record_id: object = None,
) -> tuple["EditNTurnSession | None", EditNTurnSiteResolution | None]:
    """Resolve and journal one edit episode, using a fixed two-turn fallback as needed."""
    if not enabled:
        return None, None
    resolution = resolve_edit_nturn_sites(
        repository_root=repository_root,
        model_plan_text=model_plan_text,
        model_plan_sites=model_plan_sites,
        model_plan_record_id=model_plan_record_id,
        localization_result=localization_result,
        scope_receipt=scope_receipt,
        model_action_targets=model_action_targets,
        str_replace_targets=str_replace_targets,
        live_config=live_config,
        record_id=record_id,
    )
    emit_edit_nturn_sites_resolved(journal, resolution)
    if resolution.accepted_count >= 2:
        # R2 defines N as the number of cumulative continuation opportunities, one
        # per accepted public site. The initial edit happened before this episode.
        # The off switch preserves this horizon while restoring bare continuation text.
        scope_turns_enabled = edit_nturn_scope_turns_enabled()
        session = EditNTurnSession(
            resolution.accepted_count,
            site_source=resolution.source_kind,
            max_turns=1 + resolution.accepted_count,
            scope_sites=resolution.plan_sites,
            scope_turns_enabled=scope_turns_enabled,
        )
    elif os.environ.get("STATEFRAME_EDIT_NTURN_FIXED_HORIZON", "off").strip().lower() == "on":
        session = EditNTurnSession(
            resolution.accepted_count,
            site_source="fixed_horizon",
            max_turns=1 + EDIT_NTURN_FIXED_CONTINUATIONS,
        )
    else:
        # Keep the fixed-horizon fallback disabled until it passes behavioral checks.
        # benchmark (repeat "Continue the edit." after a correct edit can over-edit). Single-turn unless the model's own
        # plan names >= 2 sites; the ladder flips this switch after its no-change/coverage-stop variant.
        session = None
    return session, resolution


class EditNTurnSession:
    """Track the first applied EDIT and its bounded cumulative continuations."""

    def __init__(
        self,
        n_sites: int,
        *,
        site_source: str = "model_plan",
        max_turns: int | None = None,
        scope_sites: Sequence[EditNTurnPlanSite] = (),
        scope_turns_enabled: bool = False,
    ) -> None:
        if type(n_sites) is not int or n_sites < 0:
            raise ValueError("n_sites must be a nonnegative integer")
        if site_source == "fixed_horizon":
            expected_turns = 1 + EDIT_NTURN_FIXED_CONTINUATIONS
            if max_turns != expected_turns:
                raise ValueError("fixed-horizon edit n-turn requires exactly two continuations")
        else:
            if n_sites < 2:
                raise ValueError("a planned edit n-turn requires at least two sites")
            expected_turns = n_sites if max_turns is None else max_turns
            if expected_turns not in {n_sites, n_sites + 1}:
                raise ValueError("planned edit n-turn must allow one continuation per site")
        if scope_turns_enabled and len(scope_sites) != n_sites:
            raise ValueError("scope-turn labels must match the accepted public site count")
        self.n_sites = n_sites
        self.site_source = site_source
        self.max_turns = expected_turns
        self.scope_sites = tuple(scope_sites)
        self.scope_turns_enabled = bool(scope_turns_enabled)
        self.changed_owner_revisions: dict[str, int] = {}
        self.guarded_turns: list[int] = []
        self.scope_event_emitted = False
        self.turns_used = 1
        import time as _time
        self.started_at = _time.monotonic()  # the first EDIT turn's applied result; edit_wall_seconds counts from here
        self.pending = False
        self.finished = False
        self.event_emitted = False

    @property
    def turns_remaining(self) -> int:
        return max(0, self.max_turns - self.turns_used)

    def queue_extra_turn(
        self,
        remaining_seconds: float | None,
        verification_reserve_seconds: float,
    ) -> str | None:
        """Queue one exact continuation if a site turn and its safety budget remain."""
        if self.finished or self.pending or self.turns_remaining <= 0:
            return None
        if not edit_nturn_budget_allows(
                remaining_seconds, verification_reserve_seconds):
            return None
        self.pending = True
        return EDIT_NTURN_INSTRUCTION

    def request_started(self) -> bool:
        """Count a queued turn only after the provider returned a response."""
        if self.finished or not self.pending:
            return False
        self.pending = False
        self.turns_used += 1
        return True

    def record_edit_result(self, *, applied: bool) -> None:
        """Stop the sequence as soon as a continuation does not apply an edit."""
        if not applied:
            self.finished = True

    def record_changed_owner(self, owner: str, revision: int) -> bool:
        """Remember only an exact site label/symbol backed by the current host edit."""
        if type(revision) is not int or revision < 0:
            return False
        for site in self.scope_sites:
            if owner in {site.owner, site.symbol} and owner:
                self.changed_owner_revisions[owner] = revision
                return True
        return False

    def continuation_instruction(
        self,
        *,
        current_revision: int,
        public_checks_green: bool,
    ) -> str:
        """Render the next public site label, falling back to bare text when guarded."""
        if not self.scope_turns_enabled or not self.scope_sites:
            return EDIT_NTURN_INSTRUCTION
        index = self.turns_used - 1
        if index < 0 or index >= len(self.scope_sites):
            return EDIT_NTURN_INSTRUCTION
        site = self.scope_sites[index]
        if not site.owner:
            return EDIT_NTURN_INSTRUCTION
        site_owners = {site.owner, site.symbol} - {""}
        if (public_checks_green and any(
                self.changed_owner_revisions.get(owner) == current_revision
                for owner in site_owners)):
            turn = index + 1
            if turn not in self.guarded_turns:
                self.guarded_turns.append(turn)
            return EDIT_NTURN_INSTRUCTION
        return render_edit_nturn_scope_turn(site, turn=index + 1, total=len(self.scope_sites))

    def finish(self) -> None:
        self.pending = False
        self.finished = True


def render_edit_nturn_scope_turn(
    site: EditNTurnPlanSite,
    *,
    turn: int,
    total: int,
) -> str:
    """Render the exact R2 template from one resolver-approved public plan site."""
    if not (1 <= turn <= total) or total < 1:
        raise ValueError("scope-turn index must be within the accepted site count")
    owner = _normalize_edit_site_owner(site.owner)
    path = _normalize_edit_site(site.path)
    if path is None:
        raise ValueError("scope-turn labels require a repository path")
    # b1424: a model-plan prose fragment ("I need to read more of the") was spliced in as the owner, producing
    # "handle I need to read more of the in <path>". Only a code identifier may be named; otherwise name the
    # path alone.
    if owner and not _REASONING_OWNER.search(owner):
        return f"Edit turn {turn}/{total}: handle {owner} in {path}. Continue the edit."
    return f"Edit turn {turn}/{total}: continue in {path}. Continue the edit."


def append_edit_nturn_instruction(
    messages: list[dict[str, object]],
    instruction: str = EDIT_NTURN_INSTRUCTION,
) -> None:
    """Append one exact user continuation and the shared early-stop line."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("edit n-turn instruction must be nonempty text")
    messages.append({
        "role": "user",
        "content": instruction + "\n" + EDIT_NTURN_NO_CHANGE_LINE,
    })


def emit_edit_nturn_scope_turns_rendered(
    journal: object,
    session: EditNTurnSession,
) -> None:
    """Journal the model-visible public schedule without consulting grader data.

    Only benchmark's three-owner schedule was measured; other site counts remain untested.
    """
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("scope-turn render requires a host journal")
    if not session.scope_turns_enabled or len(session.scope_sites) < 2:
        return
    emitter(
        "edit_nturn_scope_turns_rendered",
        n_sites=session.n_sites,
        labels=[
            {"i": index, "owner": site.owner, "path": site.path}
            for index, site in enumerate(session.scope_sites, start=1)
        ],
        guarded_turns=sorted(set(session.guarded_turns)),
    )
    session.scope_event_emitted = True


def is_edit_nturn_no_change_reply(value: object) -> bool:
    """Recognize the exact shared no-change reply, which ends an episode cleanly."""
    return isinstance(value, str) and value.strip().casefold() == \
        EDIT_NTURN_NO_CHANGE_RESPONSE.casefold()


def emit_edit_nturn_rendered(journal: object, *, n_sites: int, turns_used: int,
                             edit_wall_seconds: float,
                             site_source: str = "model_plan") -> None:
    """Journal the completed sequence without copying site names into telemetry.

    ONE spelling with the reader (rule #223): tools/ceiling/edit_specs/live_edit_nturn_report.py reads n_sites,
    turns_used (extra turns fired = turns_used - 1) and edit_wall_seconds. site_source tells whether the horizon
    came from a model-owned plan or the fixed fallback. Emitted after every applied first EDIT."""
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("edit n-turn render requires a host journal")
    if site_source not in EDIT_NTURN_SITE_SOURCES:
        raise ValueError("site_source is not a supported edit n-turn source")
    emitter("edit_nturn_rendered", n_sites=int(n_sites), turns_used=int(turns_used),
            edit_wall_seconds=max(0.0, float(edit_wall_seconds)),
            site_source=site_source, version=EDIT_NTURN_VERSION)


def emit_edit_nturn_skipped_budget(
    journal: object,
    *,
    n_sites: int,
    turns_used: int,
    remaining_seconds: float,
    required_seconds: float,
    reason: str = "wall_clock",
) -> None:
    """Record why the next planned edit turn was withheld."""
    emitter = getattr(journal, "emit", None)
    if not callable(emitter):
        raise TypeError("edit n-turn budget skip requires a host journal")
    emitter("edit_nturn_skipped_budget", n_sites=n_sites, turns_used=turns_used,
            remaining_seconds=max(0.0, float(remaining_seconds)),
            required_seconds=float(required_seconds), reason=reason,
            version=EDIT_NTURN_VERSION)
