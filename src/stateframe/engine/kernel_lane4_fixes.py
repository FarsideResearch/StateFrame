"""Bounded task-state optimizations with independent enablement switches."""
from __future__ import annotations

import os
import shlex
from typing import Mapping, Sequence

GREP_QUOTING_ENV = "STATEFRAME_GREP_QUOTING"
BOOT_READ_ENV = "STATEFRAME_BOOT_READ"
DECIDE_NOTICE_GATED_ENV = "STATEFRAME_DECIDE_NOTICE_GATED"

GREP_EVENT = "grep_native_args_quoted"
BOOT_READ_EVENT = "boot_read_offered"
DECIDE_NOTICE_EVENT = "decide_notice_suppressed"


def _on(env: str) -> bool:
    return os.environ.get(env, "off").strip().lower() == "on"


def grep_quoting_enabled() -> bool:
    return _on(GREP_QUOTING_ENV)


def boot_read_enabled() -> bool:
    return _on(BOOT_READ_ENV)


def decide_notice_gated_enabled() -> bool:
    return _on(DECIDE_NOTICE_GATED_ENV)


# --- grep ------------------------------------------------------------------------------

def quoted_grep_line(args: Mapping[str, object] | None) -> tuple[str, dict]:
    """Return (dispatch line, detail).  ``detail["quoted"]`` is True only when quoting changed
    a token, i.e. the old join would have been re-split by the dispatcher's ``shlex.split``."""
    args = args if isinstance(args, Mapping) else {}
    pattern = str(args.get("pattern") or "")
    path = str(args["path"]) if args.get("path") else ""
    q_pattern = shlex.quote(pattern) if pattern else ""
    q_path = shlex.quote(path) if path else ""
    line = ("grep " + q_pattern + ((" " + q_path) if path else "")).strip()
    pattern_changed = bool(pattern) and q_pattern != pattern
    path_changed = bool(path) and q_path != path
    return line, {"quoted": pattern_changed or path_changed,
                  "pattern_quoted": pattern_changed, "path_quoted": path_changed,
                  "has_path": bool(path)}


# --- boot menu -------------------------------------------------------------------------

def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


def tool_names(tools: object) -> list[str]:
    if not isinstance(tools, list):
        return []
    return [name for name in (_tool_name(tool) for tool in tools) if name]


def boot_read_candidate(tools: object, surface: Sequence[dict] | None) -> dict | None:
    """The ``read`` schema to offer on the boot request, or None when read is already on the
    menu or the raw surface has no read tool (never invents a schema)."""
    if not isinstance(tools, list) or "read" in tool_names(tools):
        return None
    for tool in surface or ():
        if _tool_name(tool) == "read":
            return tool
    return None


# --- decide notice ---------------------------------------------------------------------

def strip_notice(prepared: Sequence[dict], notice: str) -> tuple[list[dict], int]:
    """Remove ``notice`` from string message contents; drop messages left empty.
    Returns (messages, removed_count).  Non-string contents are never touched."""
    out: list[dict] = []
    removed = 0
    for message in prepared:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not notice or not isinstance(content, str) or notice not in content:
            out.append(message)
            continue
        removed += content.count(notice)
        remaining = content.replace(notice, "").strip()
        if remaining:
            updated = dict(message)
            updated["content"] = remaining
            out.append(updated)
    return out, removed


# --- pre-green same-revision run_tests (lane 4 M5) -----------------------------------------

PREGREEN_TEST_REUSE_ENV = "STATEFRAME_PREGREEN_TEST_REUSE"
PREGREEN_TEST_REUSE_EVENT = "pregreen_test_reused"
_CACHE_ATTR = "_lane4_pregreen_test_cache"
_GREEN_ATTR = "_lane4_pregreen_seen_green"


def pregreen_test_reuse_enabled() -> bool:
    return _on(PREGREEN_TEST_REUSE_ENV)


def _revision_key(runtime: object) -> tuple[str, str] | None:
    """(artifact, environment) of the live workspace, or None when unknowable (then never reuse)."""
    binding_fn = getattr(runtime, "binding", None)
    if not callable(binding_fn):
        return None
    try:
        binding = binding_fn()
    except Exception:  # noqa: BLE001 - unknowable revision => execute the real check
        return None
    if not isinstance(binding, Mapping):
        return None
    artifact, environment = binding.get("artifact"), binding.get("environment")
    if not artifact:
        return None
    return str(artifact), str(environment or "")


def _selector_key(selectors: Sequence[object]) -> tuple[str, ...]:
    return tuple(sorted({str(item) for item in selectors or () if str(item).strip()}))


def record_test_result(runtime: object, selectors: Sequence[object], observation: str,
                       exit_code: object, *, source: str) -> None:
    """Remember a RED public result for (revision, selectors).  Any GREEN ends pre-green for
    the run and clears the cache: reuse is strictly a pre-green, same-revision, RED replay."""
    if runtime is None:
        return
    try:
        code = int(exit_code)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if code == 0:
        setattr(runtime, _GREEN_ATTR, True)
        setattr(runtime, _CACHE_ATTR, {})
        return
    if getattr(runtime, _GREEN_ATTR, False):
        return
    revision = _revision_key(runtime)
    key = _selector_key(selectors)
    if revision is None or not key:
        return
    cache = getattr(runtime, _CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
    cache[(revision, key)] = {"observation": str(observation or ""), "exit_code": code,
                              "source": source}
    setattr(runtime, _CACHE_ATTR, cache)


def pregreen_reuse(runtime: object, selectors: Sequence[object]) -> tuple[str, dict] | None:
    """The cached RED reply for a same-revision pre-green re-run, or None (execute for real)."""
    if runtime is None or getattr(runtime, _GREEN_ATTR, False):
        return None
    cache = getattr(runtime, _CACHE_ATTR, None)
    if not isinstance(cache, dict) or not cache:
        return None
    revision = _revision_key(runtime)
    key = _selector_key(selectors)
    if revision is None or not key:
        return None
    hit = cache.get((revision, key))
    if hit is None:
        return None
    rev12 = revision[0].split(":")[-1][:12]
    pin = (f"tests@{rev12}: RED (exit {hit['exit_code']}; {hit['source']} at this revision). "
           "No file changed since, so re-running at this revision returns the same result; "
           "edit the source before running the tests again.")
    return (pin + "\n" + hit["observation"],
            {"rev12": rev12, "exit_code": hit["exit_code"], "source": hit["source"],
             "selectors": list(key)})


# --- empty read resolve (lane 4 M1 pair / verdict L1c batch variant) -----------------------

EMPTY_READ_RESOLVE_ENV = "STATEFRAME_EMPTY_READ_RESOLVE"
EMPTY_READ_RESOLVE_EVENT = "empty_read_dispatched"


def empty_read_resolve_enabled() -> bool:
    return _on(EMPTY_READ_RESOLVE_ENV)


def resolve_empty_read(runtime: object, tool: str, arguments: Mapping[str, object] | None
                       ) -> tuple[str, list[str], str] | None:
    """For an empty-argument read with unread product files: ("dispatch", unread paths, "") and the
    caller reads the first one.  Otherwise None: all-in-frame / no-index replies stay with
    kernel_rejected_loop_breaker, whose readable-path index (binding product/harness files split
    unread vs in-frame) is reused here rather than re-implemented."""
    from kernel_rejected_loop_breaker import _binding_files, _in_frame_paths, is_empty_read
    if not is_empty_read(tool, arguments):
        return None
    product, _harness = _binding_files(runtime)
    in_frame = _in_frame_paths(runtime) if runtime is not None else set()
    # Paths this resolver already dispatched count as read even when the stub-sweep cut (the
    # in-frame source) is off, so repeated `read {}` walks the files instead of re-reading one.
    dispatched = getattr(runtime, "_lane4_empty_read_dispatched", None)
    if not isinstance(dispatched, set):
        dispatched = set()
    unread = [path for path in product if path not in in_frame and path not in dispatched]
    if not unread:
        return None
    if runtime is not None:
        try:
            setattr(runtime, "_lane4_empty_read_dispatched", dispatched | {unread[0]})
        except AttributeError:
            pass
    return "dispatch", unread, ""
