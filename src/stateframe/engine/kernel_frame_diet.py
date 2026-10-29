"""Frame diet: shrink the per-request StateFrame prompt without removing capability.

Inert unless ``STATEFRAME_FRAME_DIET=on`` (OFF returns the exact inputs, so the
legacy request bytes are unchanged).  Three independent mechanisms:

1. Job tools: the four ``job_*`` schemas collapse into ONE ``job`` tool with an
   action enum, offered only after a host-observed long-running signal (a check
   or probe that exceeded its synchronous timeout, or a job already on record).
   Never derived from ticket text.
2. System prompt: v2/v3 guidance is replaced by the invariants plus the minimum
   playbook; the worked example is a per-model scaffold (default off).
3. Ticket text: the bootstrap FULL_TICKET field references the Task message when
   that message already carries the identical text in the same request.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_FRAME_DIET"
WORKED_EXAMPLE_ENV = "STATEFRAME_WORKED_EXAMPLE"
JOB_TOOL_NAME = "job"
JOB_ACTIONS = {"launch": "launch_job", "status": "job_status",
               "output": "job_output", "kill": "job_kill"}
LEGACY_JOB_TOOLS = frozenset(JOB_ACTIONS.values())
LONG_RUN_SIGNALS = frozenset({"check_timeout", "probe_timeout", "job_on_record"})


def _switch(env: str) -> bool:
    value = os.environ.get(env, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{env} must be off or on")
    return value == "on"


def enabled() -> bool:
    return _switch(MODE_ENV)


def worked_example_enabled() -> bool:
    return _switch(WORKED_EXAMPLE_ENV)


# ---------------------------------------------------------------- 1. job tool
def job_tool() -> dict[str, object]:
    """One schema for every job operation; arguments are validated by JobWatch."""
    return {
        "type": "function",
        "function": {
            "name": JOB_TOOL_NAME,
            "description": ("Background job for a long command. launch needs cmd, expected_s, "
                            "done_signal; status/output/kill need job_id. The host never kills "
                            "a job automatically."),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": list(JOB_ACTIONS)},
                    "cmd": {"type": "string"},
                    "expected_s": {"type": "number", "exclusiveMinimum": 0},
                    "done_signal": {"type": "string"},
                    "job_id": {"type": "string"},
                    "tail_lines": {"type": "integer", "minimum": 1},
                    "why": {"type": "string"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    }


def job_menu(tools: object, *, long_run_signal: bool) -> object:
    """Replace legacy job schemas by the single gated ``job`` tool (diet ON only)."""
    if not enabled() or not isinstance(tools, list):
        return tools
    had_jobs = any(isinstance(tool, Mapping)
                   and tool.get("function", {}).get("name") in LEGACY_JOB_TOOLS | {JOB_TOOL_NAME}
                   for tool in tools)
    kept = [tool for tool in tools
            if not (isinstance(tool, Mapping)
                    and tool.get("function", {}).get("name") in LEGACY_JOB_TOOLS | {JOB_TOOL_NAME})]
    if had_jobs and long_run_signal:
        kept.append(job_tool())
    return kept


def job_call(name: str, arguments: Mapping[str, object] | None) -> tuple[str, dict]:
    """Map a ``job`` call to the organ's legacy operation; other names pass through."""
    if name != JOB_TOOL_NAME:
        return name, dict(arguments or {})
    args = dict(arguments or {})
    action = args.pop("action", None)
    operation = JOB_ACTIONS.get(action) if isinstance(action, str) else None
    # An unknown action becomes an unknown operation name: JobWatch rejects it typed.
    return (operation or f"job:{action}"), args


# ---------------------------------------------------------- 2. system prompt
_PLAYBOOK = """How to work
1. List every requirement in the ticket (defaults, allowed values, exact errors, each named layer,
   compatibility); the public test checks few of them. Follow each value and error to what the caller receives.
2. Keep existing calls working; copy inputs before changing them. Any declared product file is editable
   with str_replace; provided tests are not.
"""

_EXAMPLE = """Worked example (different task, same shape; illustration only, never copy its names or values):
"Booking gains optional `level`, default "open"; "open"/"advanced" valid; else HTTP 422 'unknown level';
through model, API and CLI; old 2-arg call works." -> seven checklist lines; a service-only change passes
a default test and still fails four of them.
"""

_FRAME = """Host-recorded facts are true; your notes are claims. Work open and blocking items first.
Do not re-read a file whose current content you have. Do not rerun an unchanged passing check.
"""

_DONE_V3 = """Call done when every requirement maps to a file and line and relevant tests pass on current code;
the host verifies completion and lists anything still open.
"""

_DONE_V2 = """Call done when every requirement maps to a file and line and relevant tests pass on current code.
"""

_TRUTH = """The public requirements define the objective. Intentions and hypotheses are not proof.
A passing check covers only the source it ran on and does not reveal the private grade.
If blocked, preserve the concrete failure and change the approach rather than repeat it.
Behavioral invariants:
- Evidence outranks intuition; an assumption is not evidence.
- When evidence contradicts your hypothesis, revise the hypothesis — do not defend it.
- Root causes outrank symptoms.
- Check the small details that could invalidate your conclusion; skip the ones that cannot.
- Prefer the smallest complete change that addresses the cause.
- Construct conclusions that would survive an attempt to disprove them.
- Be decisive when evidence is sufficient; be skeptical when it is not."""


def guidance(variant: str) -> str:
    """Diet replacement for ``kernel_step_contract.prompt_guidance`` (v2/v3)."""
    parts = [_PLAYBOOK]
    if worked_example_enabled():
        parts.append(_EXAMPLE)
    if variant == "v3":
        parts.extend([_FRAME, _DONE_V3])
    else:
        parts.append(_DONE_V2)
    parts.append(_TRUTH)
    return "\n".join(parts)


# ------------------------------------------------------------ 3. ticket once
TICKET_REFERENCE = "(identical to the Task message in this request; not repeated)"


def ticket_already_sent(ticket: str, messages: Sequence[Mapping[str, object]]) -> bool:
    """True when another user message in this request already carries the full ticket."""
    if not enabled() or not ticket.strip():
        return False
    return any(isinstance(message, Mapping) and message.get("role") == "user"
               and ticket in str(message.get("content") or "")
               for message in messages)


__all__ = [
    "JOB_ACTIONS", "JOB_TOOL_NAME", "LEGACY_JOB_TOOLS", "LONG_RUN_SIGNALS", "MODE_ENV",
    "TICKET_REFERENCE", "WORKED_EXAMPLE_ENV", "enabled", "guidance", "job_call", "job_menu",
    "job_tool", "ticket_already_sent", "worked_example_enabled",
]
