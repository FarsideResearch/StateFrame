"""AST scanner for the reason-log guard: every kill / refusal / fail-open site must say why.

Shape follows `_class_guard_scanners.scan_unavailable_events`: parse each file once, give every
function its own scope (nested defs excluded), classify sites, then require a logging call in the
SAME scope.  A site is satisfied by an `emit_*` helper call or by a known logging wrapper.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from _class_guard_scanners import _function_scope, _parse_file, _qualified_name, _scope_nodes

EMITTERS = {"emit_reason", "emit_kill", "emit_refusal", "emit_timeout"}
# Wrappers that already log for their caller (see kernel_exec_tracker / kernel_job_watch).
WRAPPERS = {"kill_launch_unit", "_kill_group", "_seal_violation"}
WRAPPER_PREFIXES = ("_log_",)
# Raw signal senders. kill_launch_unit is a logging wrapper, not a raw sender.
RAW_KILLERS = {"killpg", "pidfd_send_signal", "signal_launch_group"}
PROCESS_RECEIVER = re.compile(r"(^|\.)(process|proc|child|popen|p|self\._?proc(ess)?)$")
SIGNAL_NAMES = re.compile(r"SIGKILL|SIGTERM")
REFUSAL_RAISES = {"RequestSealError", "RequestDietDeferred", "ContextCapacityError", "ContainmentUnverified"}
# Defining these IS the drop/fail-open point, so the definition must log; callers rely on it.
DROP_DEFS = {"note_budget_omission", "fail_open_pending"}


@dataclass(frozen=True)
class ReasonScan:
    scanned: int
    with_reason: int
    allowlisted: int
    violations: tuple[str, ...]
    stale_allowlist: tuple[str, ...] = ()

    @property
    def receipt(self) -> str:
        return (f"reason-log scan: scanned={self.scanned} with_reason={self.with_reason} "
                f"allowlisted={self.allowlisted} violations={len(self.violations)}")


def _is_zero(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == 0


def _mentions_signal(call: ast.Call) -> bool:
    for arg in (*call.args, *(item.value for item in call.keywords)):
        for node in ast.walk(arg):
            if isinstance(node, ast.Attribute) and SIGNAL_NAMES.search(node.attr):
                return True
            if isinstance(node, ast.Name) and SIGNAL_NAMES.search(node.id):
                return True
    return False


def _kill_site(call: ast.Call) -> bool:
    name = _qualified_name(call.func)
    last = name.rsplit(".", 1)[-1]
    if name == "signal.signal" or last in WRAPPERS:
        return False
    if last in RAW_KILLERS:
        # killpg(pgid, 0) / os.kill(pid, 0) are liveness probes, not kills.
        return not (len(call.args) >= 2 and _is_zero(call.args[1]))
    if name in {"os.kill"} or last == "kill" and name.startswith("os."):
        return not (len(call.args) >= 2 and _is_zero(call.args[1]))
    if last in {"kill", "terminate", "send_signal"}:
        receiver = name.rsplit(".", 1)[0] if "." in name else ""
        return bool(PROCESS_RECEIVER.search(receiver)) or _mentions_signal(call)
    return False


def _refusal_site(node: ast.AST) -> bool:
    if isinstance(node, ast.Raise) and node.exc is not None:
        target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        return _qualified_name(target).rsplit(".", 1)[-1] in REFUSAL_RAISES
    return False


def _logs(call: ast.Call) -> bool:
    last = _qualified_name(call.func).rsplit(".", 1)[-1]
    return last in EMITTERS or last in WRAPPERS or last.startswith(WRAPPER_PREFIXES)


def _label(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def scan_reason_sites(paths: Iterable[Path], *, root: Path,
                      allowlist: dict[str, str] | None = None) -> ReasonScan:
    allowlist = allowlist or {}
    scanned = with_reason = allowlisted = 0
    violations: list[str] = []
    used: set[str] = set()
    for path in sorted(Path(item) for item in paths):
        resolved = path.resolve()
        if resolved.name == "kernel_reason_log.py":
            continue
        stat = resolved.stat()
        tree = _parse_file(str(resolved), stat.st_mtime_ns, stat.st_size)
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        label = _label(path, root)
        scopes: dict[ast.AST, tuple[list[ast.AST], list[ast.AST]]] = {}
        for node in ast.walk(tree):
            site = None
            if isinstance(node, ast.Call) and _kill_site(node):
                site = "kill"
            elif _refusal_site(node):
                site = "refusal"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in DROP_DEFS:
                site = "drop"
            if site is None:
                continue
            scope = node if site == "drop" else _function_scope(node, parents)
            if scope is None:
                continue
            if scope not in scopes:
                scopes[scope] = (_scope_nodes(scope, tree), [])
            scanned += 1
            function = getattr(scope, "name", "<lambda>")
            key = f"{label}:{function}"
            if any(isinstance(item, ast.Call) and _logs(item) for item in scopes[scope][0]):
                with_reason += 1
            elif key in allowlist:
                allowlisted += 1
                used.add(key)
            else:
                violations.append(f"{label}:{function}:{node.lineno} ({site})")
    stale = tuple(sorted(set(allowlist) - used))
    return ReasonScan(scanned, with_reason, allowlisted, tuple(violations), stale)
