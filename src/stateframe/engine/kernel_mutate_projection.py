"""Bounded mutation frame text, authoring menu, and durable send reservations."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence


AUTHORING_TOOLS = frozenset({"mutate", "read", "list_files", "grep"})
AUTHORING_EFFECT_TOOLS = frozenset({
    "done", "edit", "str_replace", "insert_before", "file_create", "file_delete",
    "file_move", "shell", "git", "work", "decide", "predict",
})


class MutationSendRefused(RuntimeError):
    """A send reservation failed; the parent can continue with a failed episode."""

    def __init__(self, receipt: str):
        self.receipt = receipt
        super().__init__(receipt)


def observe_committed_tool(runtime: object, measurement: object, line: str,
                           turn_index: int, *, continue_turn: bool,
                           hook_call_id: str | None = None) -> None:
    """Attach observing work to a settled tool result before the next request."""
    capture = getattr(measurement, "_active_capture", None)
    parent = getattr(capture, "request_id", None)
    if not isinstance(parent, str) or not parent:
        return
    tool = line.split(" ", 1)[0]
    if tool != "mutate":
        runtime.invoke_hook(event="post_tool", facts={
            "tool": tool, "status": "observed", "call_id": hook_call_id or parent})
    if continue_turn:
        for observation in runtime.tick(parent, turn_index):
            runtime.last = observation[:512]


def observe_tool_reply(runtime: object, measurement: object, messages: Sequence[object],
                       call_id: str, turn_index: int, *, seen: set[str],
                       continue_turn: bool) -> None:
    """Bridge every native tool branch once after its result has been assembled."""
    if call_id in seen:
        return
    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, (list, tuple)):
            continue
        for call in calls:
            if not isinstance(call, Mapping) or call.get("id") != call_id:
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            if isinstance(name, str) and name:
                seen.add(call_id)
                observe_committed_tool(runtime, measurement, name, turn_index,
                                       continue_turn=continue_turn, hook_call_id=call_id)
            return


def publish_mutation_edit_receipts(runtime: object, call_id: str, bolt: object,
                                   measurement: object) -> tuple[str, ...]:
    """Use the native edit ledger and workspace invalidation for committed proposals."""
    paths = runtime.committed_edit_paths(call_id)
    if not paths:
        return ()
    measurement._note_applied_mutation(paths, "mutation_edit")
    for path in paths:
        bolt.receipt("edit", f"mutation edit {path}", path, "APPLIED",
                     "mutation edit applied.", target_path=path, changed_paths=paths)
    return paths


def _gate_resolver_names(runtime: object | None, tools: Sequence[object]) -> frozenset[str]:
    """Read the same live recovery vector the final seal validates."""
    factory = getattr(runtime, "pass_gates_vector", None)
    if not callable(factory):
        return frozenset()
    vector = factory(tools)
    recoveries = getattr(vector, "recoveries", ())
    if not isinstance(recoveries, (tuple, list)):
        raise RuntimeError("MUTATION_FRAME pass-gate recoveries are unavailable")
    names: set[str] = set()
    for recovery in recoveries:
        if not isinstance(recovery, Mapping) or not isinstance(recovery.get("tool"), str):
            raise RuntimeError("MUTATION_FRAME pass-gate recovery is invalid")
        names.add(recovery["tool"])
    return frozenset(names)


def authoring_dispatch_allowed(name: str, runtime: object | None = None,
                               tools: Sequence[object] = ()) -> bool:
    if name in AUTHORING_TOOLS:
        return True
    if name in AUTHORING_EFFECT_TOOLS:
        return False
    offered = {tool["function"].get("name") for tool in tools
               if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)}
    return name in offered and name in _gate_resolver_names(runtime, tools)


def require_authoring_surface(capture: object) -> None:
    contract = getattr(capture, "request_action_contract", None)
    if not isinstance(contract, Mapping):
        raise RuntimeError("MUTATION_FRAME callable contract is missing")
    names = contract.get("callable_actions", contract.get("advertised_actions"))
    tools = contract.get("admitted_tool_schemas", ())
    runtime = getattr(capture, "completion_owner_runtime", None)
    if (not isinstance(names, (tuple, list)) or "mutate" not in names
            or any(not isinstance(name, str)
                   or not authoring_dispatch_allowed(name, runtime, tools) for name in names)):
        raise RuntimeError("MUTATION_FRAME callable contract is broader than the authoring frame")


def authoring_tools(tools: Sequence[object], runtime: object | None = None) -> list[object]:
    allowed = AUTHORING_TOOLS | (_gate_resolver_names(runtime, tools) - AUTHORING_EFFECT_TOOLS)
    return [tool for tool in tools if isinstance(tool, Mapping)
            and isinstance(tool.get("function"), Mapping)
            and tool["function"].get("name") in allowed]


def authoring_view(runtime: object) -> str:
    episode = getattr(runtime, "episode", None)
    if not isinstance(episode, Mapping):
        return ""
    goal = str(episode.get("goal", ""))[:100]
    draft = episode.get("draft")
    pages = []
    if isinstance(draft, Mapping):
        for name, source in list(draft.items())[:2]:
            content = str(source)
            digest = hashlib.sha256(content.encode()).hexdigest()[:12]
            pages.append(f"{str(name)[:40]} sha256:{digest}: {content[:80]}")
    return (f"MUTATE episode {str(episode.get('id', ''))[:40]} goal: {goal}; "
            f"sends left: {episode.get('sends_left', 0)}; "
            f"diagnostic: {str(episode.get('diagnostic', 'none'))[:80]}\n" + "\n".join(pages))[:400]


def _proposal_for_task(runtime: object, task: object) -> Mapping[str, object] | None:
    propose = getattr(runtime, "propose_prebuilt", None)
    if not isinstance(task, str) or not task or not callable(propose):
        return None
    proposal = propose(task)
    return proposal if isinstance(proposal, Mapping) else None


def _proposal_lines(proposal: Mapping[str, object]) -> tuple[str, str, str] | None:
    unit, summary = proposal.get("unit"), proposal.get("summary")
    args = proposal.get("args")
    trigger_file = proposal.get("trigger_file")
    input_id = proposal.get("input_id")
    input_schema = proposal.get("input_schema")
    file_format = proposal.get("format")
    if (not isinstance(unit, str) or not unit or any(not char.isprintable() for char in unit)
            or not isinstance(summary, str)
            or not isinstance(args, Mapping) or set(args) != {"input", "inputs"}
            or not isinstance(trigger_file, str) or not trigger_file
            or any(not char.isprintable() for char in trigger_file)
            or input_id != "in0" or not isinstance(file_format, str)
            or not file_format or len(file_format) > 64
            or any(not char.isprintable() for char in file_format)
            or not isinstance(input_schema, Mapping)):
        return None
    input_value = args.get("input")
    properties = input_schema.get("properties")
    required = input_schema.get("required")
    file_id_keys = [key for key in required if isinstance(key, str) and key.endswith("_id")] \
        if isinstance(required, list) else []
    if (not isinstance(input_value, Mapping) or len(file_id_keys) != 1
            or input_value.get(file_id_keys[0]) != input_id
            or args.get("inputs") != [trigger_file]
            or not isinstance(properties, Mapping)):
        return None
    if "format" in input_value and ("format" not in properties or input_value.get("format") != file_format):
        return None
    try:
        if __package__:
            from .kernel_mutate_wire import validate_schema
        else:
            from kernel_mutate_wire import validate_schema
        validate_schema(input_value, input_schema)
    except (ImportError, TypeError, ValueError):
        return None
    description = next((line.strip() for line in summary.splitlines() if line.strip()), "")[:160]
    if not description or any(not char.isprintable() for char in description):
        return None
    if not description.endswith("."):
        description += "."
    try:
        encoded_args = json.dumps(args, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, UnicodeError):
        return None
    ready = f"ready: {unit} — {description}"
    ready_inputs = f"ready inputs: {input_id}={trigger_file}"
    host = (f"Host: {unit} fits this task. Call mutate use {unit} {encoded_args} "
            "or decline (say why).")
    if len("\n".join((ready, ready_inputs, host)).encode("utf-8")) > 1024:
        return None
    return ready, ready_inputs, host


def panel(runtime: object, *, task: str = "") -> str:
    ready = getattr(runtime, "ready", {})
    names = list(ready) if isinstance(ready, Mapping) else []
    shown = ", ".join(str(name)[:32] for name in names[:4]) or "none"
    more = f" +{len(names) - 4}" if len(names) > 4 else ""
    hooks = getattr(runtime, "hook_lines", ())
    hook_text = str(hooks[-1]).encode("utf-8")[:512].decode("utf-8", "ignore") if hooks else ""
    hook_line = f" | observer: {hook_text}" if hook_text else ""
    proposal = _proposal_for_task(runtime, task)
    proposal_lines = _proposal_lines(proposal) if proposal is not None else None
    if proposal_lines is not None:
        return "\n".join(proposal_lines)
    text = (f"MUTATE mutations: {getattr(runtime, 'level', 'off')} | "
            f"ready: {shown}{more} | last: {str(getattr(runtime, 'last', 'none'))[:80]}{hook_line}")
    return text[:700]


def authoring_plan(owner: object, tools: object):
    """Collapse the parent projection to the task and an allocator-measured episode view."""
    from kernel_context import (FRAME_CEILING_TOKENS, FRAME_TARGET_TOKENS, RenderOverflow,
                                SelectableItem, SelectionPriority, ViewPlan, token_estimate)
    from kernel_measurement import canonical_bytes

    row = getattr(getattr(owner, "owner", None), "task_row", {})
    task = str(row.get("msg", "")) if isinstance(row, Mapping) else ""
    task_preview = task[:160]
    runtime = getattr(owner, "mutation_runtime", None)
    if runtime is None:
        from kernel_mutate_runtime import service
        runtime = service()
    mutation_text = authoring_view(runtime)
    proposal = _proposal_for_task(runtime, task)
    proposal_lines = _proposal_lines(proposal) if proposal is not None else None
    if proposal_lines is not None:
        mutation_text += "\n" + "\n".join(proposal_lines)
    messages = (
        {"role": "system", "content": "Task: " + task_preview},
        {"role": "system", "content": mutation_text},
    )
    candidates = {f"mutate:{index}": dict(message) for index, message in enumerate(messages)}
    items = tuple(SelectableItem(
        item_id=f"mutate:{index}", priority=SelectionPriority.REQUIRED_CONTROL_CONTEXT,
        group=f"mutate:{index}", text=canonical_bytes(message).decode(),
        token_estimate=token_estimate(message), action_required=True,
        revision="mutation-episode") for index, message in enumerate(messages))
    used = sum(item.token_estimate for item in items)
    ceiling = int(getattr(owner, "frame_ceiling", FRAME_CEILING_TOKENS))
    available = ceiling - token_estimate(tools) if tools else ceiling
    if used > available:
        return RenderOverflow("mutate", available, used, "", "mutation_frame_ceiling"), candidates
    return ViewPlan(items, (), used, min(available, FRAME_TARGET_TOKENS),
                    "stateframe-mutation-frame/1", applied=True), candidates


def reserve_send(runtime: object, request_id: object, attempt: int, ordinal: int,
                 *, replay: bool = False) -> bool:
    episode = getattr(runtime, "episode", None)
    if not isinstance(episode, dict):
        return True
    key = f"{episode.get('id')}:{request_id}:{attempt}:{ordinal}"
    used = episode.setdefault("send_keys", set())
    journal = getattr(runtime, "mutation_journal", None)
    if journal is None:
        journal = getattr(runtime, "journal", None)
    rows = list(journal.events()) if callable(getattr(journal, "events", None)) else []
    if replay:
        if not callable(getattr(journal, "events", None)):
            raise RuntimeError("MUTATION_EVIDENCE_TAMPERED send reservation journal is missing")
        matches = [str(row.get("key")) for row in rows
                   if row.get("kind") == "send_reserved"
                   and row.get("episode") == episode.get("id")
                   and row.get("request_id") == str(request_id)
                   and row.get("ordinal") == ordinal]
        if len(matches) != 1:
            raise RuntimeError("MUTATION_EVIDENCE_TAMPERED send reservation is missing or ambiguous")
        key = matches[0]
    if key in used:
        return True
    existing = [] if replay else [row for row in rows
                                  if row.get("kind") == "send_reserved" and row.get("key") == key]
    if len(existing) > 1:
        raise RuntimeError("MUTATION_EVIDENCE_TAMPERED duplicate send reservation")
    if existing:
        used.add(key)
        episode["sends_left"] = max(0, int(episode.get("sends_left", 0)) - 1)
        return True
    if int(episode.get("sends_left", 0)) <= 0:
        return False
    if replay:
        used.add(key)
        episode["sends_left"] = int(episode["sends_left"]) - 1
        return True
    count = sum(1 for row in rows if row.get("kind") == "send_reserved")
    if count >= 10:
        return False
    journal.append("send_reserved", episode=episode["id"], key=key,
                   request_id=str(request_id), attempt=attempt, ordinal=ordinal)
    used.add(key)
    episode["sends_left"] = int(episode["sends_left"]) - 1
    return True
