"""Optional prompt and task-state optimizations with independent controls."""
from __future__ import annotations

import ast
import os
import re
from collections.abc import Mapping

PROBE_RENDER_ENV = "STATEFRAME_TOKENLAB_PROBE_RENDER"
PROBE_PYTEST_GUARD_ENV = "STATEFRAME_TOKENLAB_PROBE_PYTEST_GUARD"
SYSTEM_TRIM_ENV = "STATEFRAME_TOKENLAB_SYSTEM_TRIM"
GET_SURE_HINT_ENV = "STATEFRAME_TOKENLAB_GET_SURE_HINT_DROP"
EVENT_PROBE_RENDER = "probe_render_applied"
EVENT_PROBE_PYTEST = "probe_pytest_refused"
EVENT_PROBE_PYTEST_CHECK = "probe_pytest_admission_checked"
EVENT_SYSTEM_TRIM = "system_trim_applied"
EVENT_GET_SURE = "get_sure_hint_dropped"
PRODUCER = "kernel_tokenlab_v13"

PROBE_SHOWN_BYTES = 2048
_RUN_RE = re.compile(r"([^\w\s])\1{5,}", re.UNICODE)
PYTEST_REFUSAL = ("probe refused: it invokes pytest/unittest. Use run_tests for checks; probe is for one bounded "
                  "observation-only snippet.")
_TEST_RUNNERS = frozenset({"pytest", "py.test", "unittest"})

# system prompt: phantom paragraphs (exact producer text) and the boss's operating lines
WORK_TOOL_PARAGRAPHS = (
    "\nThe `work` tool is intent-only in host-closure mode: submit only a claim_implemented operation with host "
    "evidence references. The host owns status and re-runs the bound checks; do not attempt to set or drop records.",
    "\nThe `work` tool is the durable typed work record. Use it for goals, ordered steps, and decisions that must "
    "survive the next frame. It is host-owned: ids are assigned by the host, invalid operations are rejected with "
    "the current state, drops require a reason, and there is no replace operation. Keep mutations small; use "
    "evidence_ref when reopening a completed item.",
)
MEMORY_PARAGRAPH = ("\nOptional decision memory rides on this action. In the visible action envelope, use the optional "
                    "top-level memory object only for ADOPT, CHALLENGE, or SUPERSEDE typed drafts. It is never "
                    "required. Cite only receipt references present in the current StateFrame; malformed memory is "
                    "ignored by the host.")
AUTHORITY_LINE = "The single EditAuthority in the current frame describes editable paths and shortcuts."
BOSS_EDIT_LINE = "The edit tool replaces a complete top-level function or class."
BOOT_LINE = ("\nBoot asks for assumptions, a plan, and needed files. Read only the host-requested range; edits and "
             "checks advance the host phase.")
BOSS_LINES = (
    "Use failing checks to identify what to change. Test the changed revision with relevant public checks.",
    "Do not rerun an unchanged passing check without a specific reason.",
    "Review requirements as well as tests. Call done when the task is satisfied and current checks pass.",
)
BOSS_LINES_ANCHOR = "run_tests accepts a pytest path or selector, not shell commands or package installations."


def _mode(env: str, environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(env, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{env} must be off or on")
    return value


def probe_render_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _mode(PROBE_RENDER_ENV, environ) == "on"


def probe_pytest_guard_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _mode(PROBE_PYTEST_GUARD_ENV, environ) == "on"


def system_trim_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _mode(SYSTEM_TRIM_ENV, environ) == "on"


def get_sure_hint_drop_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return _mode(GET_SURE_HINT_ENV, environ) == "on"


def _emit(journal: object, kind: str, **payload: object) -> bool:
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return False
    try:
        emit(kind, producer=PRODUCER, dispatch=False, **payload)
    except Exception:  # noqa: BLE001
        return False
    return True


# --------------------------------------------------------------------------------------------- 1. probe render
def collapse_runs(text: str) -> tuple[str, int]:
    """Collapse every run of >= 6 identical characters to 3 of them + '…(xN)'.  Returns (text, runs)."""
    count = 0

    def repl(match: re.Match) -> str:
        nonlocal count
        count += 1
        run = match.group(0)
        return run[:3] + f"…(x{len(run)})"

    return _RUN_RE.sub(repl, text), count


def bound_shown_bytes(value: str, *, limit: int = PROBE_SHOWN_BYTES) -> tuple[str, int, bool, int]:
    """Byte-sliced, run-collapsed shown output: (shown, shown_bytes, truncated, collapsed_runs)."""
    collapsed, runs = collapse_runs(value)
    raw = collapsed.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return collapsed, len(raw), False, runs
    shown = raw[:limit].decode("utf-8", errors="ignore") + " ...[shown output truncated]"
    return shown, limit, True, runs


def probe_invokes_pytest(code: str) -> bool:
    """Recognise an executable test-runner call, not a word in evidence text."""
    try:
        tree = ast.parse(code or "")
    except (SyntaxError, TypeError, ValueError):
        return False

    def literal_command(node: ast.AST) -> list[str] | None:
        try:
            value = ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError):
            return None
        if isinstance(value, str):
            return value.strip().split()
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return list(value)
        return None

    def command_sequence(node: ast.AST) -> list[str | None] | None:
        """Keep literal argv slots while tolerating ``sys.executable``."""
        if isinstance(node, (ast.List, ast.Tuple)):
            result: list[str | None] = []
            for item in node.elts:
                try:
                    value = ast.literal_eval(item)
                except (ValueError, TypeError, SyntaxError):
                    value = None
                result.append(value if isinstance(value, str) else None)
            return result
        return literal_command(node)

    def is_runner_command(argv: list[str | None] | None) -> bool:
        if not argv:
            return False
        first = (argv[0] or "").rsplit("/", 1)[-1].lower()
        if first in _TEST_RUNNERS:
            return True
        for index, item in enumerate(argv[:-1]):
            if item == "-m" and (argv[index + 1] or "").lower() in _TEST_RUNNERS:
                return True
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner, method = func.value.id.lower(), func.attr.lower()
            if method == "main" and owner in _TEST_RUNNERS:
                return True
            if owner == "subprocess" and method in {"run", "call", "check_call", "check_output", "popen"}:
                if node.args and is_runner_command(command_sequence(node.args[0])):
                    return True
            if owner == "os" and method == "system" and node.args:
                if is_runner_command(command_sequence(node.args[0])):
                    return True
    return False


# --------------------------------------------------------------------------------------------- 3. system trim
def trim_system(text: str, *, capabilities: Mapping[str, bool] | None = None) -> tuple[str, dict[str, object]]:
    """Trim only instructions for capabilities absent from the final policy."""
    before = text
    policy = {"work": False, "memory_sidecar": False, "boot_read": False, "edit_authority": True}
    if isinstance(capabilities, Mapping):
        policy.update({key: bool(value) for key, value in capabilities.items() if key in policy})
    removed: list[str] = []
    added: list[str] = []
    if not policy["work"]:
        for paragraph in WORK_TOOL_PARAGRAPHS:
            if paragraph in text:
                text = text.replace(paragraph, "")
                removed.append("work_tool_paragraph")
    if not policy["memory_sidecar"] and MEMORY_PARAGRAPH in text:
        text = text.replace(MEMORY_PARAGRAPH, "")
        removed.append("memory_sidecar_paragraph")
    if not policy["edit_authority"] and AUTHORITY_LINE in text:
        text = text.replace(AUTHORITY_LINE, "")
        removed.append("edit_authority_line")
    if not policy["boot_read"] and BOOT_LINE in text:
        text = text.replace(BOOT_LINE, "")
        removed.append("boot_phase_line")
    missing = [line for line in BOSS_LINES if line not in text]
    if missing:
        block = "\n".join(missing)
        if BOSS_LINES_ANCHOR in text:
            text = text.replace(BOSS_LINES_ANCHOR, BOSS_LINES_ANCHOR + "\n" + block, 1)
        else:
            text = text.rstrip("\n") + "\n" + block
        added.extend(f"boss_line_{index}" for index, line in enumerate(BOSS_LINES) if line in missing)
    detail = {"removed": removed, "added": added, "bytes_before": len(before.encode("utf-8")),
              "bytes_after": len(text.encode("utf-8")), "capabilities": policy}
    return text, detail


__all__ = ["AUTHORITY_LINE", "BOOT_LINE", "BOSS_EDIT_LINE", "BOSS_LINES", "EVENT_GET_SURE", "EVENT_PROBE_PYTEST",
           "EVENT_PROBE_PYTEST_CHECK", "EVENT_PROBE_RENDER", "EVENT_SYSTEM_TRIM", "GET_SURE_HINT_ENV",
           "MEMORY_PARAGRAPH", "PROBE_PYTEST_GUARD_ENV", "PROBE_RENDER_ENV",
           "PROBE_SHOWN_BYTES", "PRODUCER", "PYTEST_REFUSAL", "SYSTEM_TRIM_ENV", "WORK_TOOL_PARAGRAPHS",
           "bound_shown_bytes", "collapse_runs", "get_sure_hint_drop_enabled", "probe_invokes_pytest",
           "probe_pytest_guard_enabled", "probe_render_enabled", "system_trim_enabled", "trim_system"]
