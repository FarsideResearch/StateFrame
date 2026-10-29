"""Opt-in post-green product-edit lock for the StateFrame repair loop."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from kernel_obligation_assess import assessment_is_blocking
from kernel_verdict import SATISFIED, evaluate_required_tests

MODE_ENV = "STATEFRAME_GREEN_LOCK"
GREEN_LOCK_MESSAGE = (
    "GREEN LOCK: public checks green for selectors S at revision R; "
    "open obligations [ids]"
)
REFUSAL_SUFFIX = " (refused once; repeat the same call to override)"
TERMINAL_GREEN_LOCK_PREFIX = (
    "GREEN LOCK: public checks green for selectors "
)
TERMINAL_GREEN_LOCK_SUFFIX = "; open obligations "
_GREEN_STATUSES = frozenset({
    "GREEN", "CERTIFIED", "PASSED", "PASS", "SUCCESS", "REUSED_VERIFICATION", "OK",
})
_EDIT_ACTIONS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_TERMINAL_ACTIONS = frozenset({"done", "run_tests"})
_PROGRESS_ACTIONS = _EDIT_ACTIONS | frozenset({"read", "work"})


def _repair_blocks_done_enabled(runtime: object) -> bool:
    marker = getattr(runtime, "stateframe_repair_blocks_done_mode", None)
    if marker is not None:
        return marker == "on"
    return os.environ.get("STATEFRAME_REPAIR_BLOCKS_DONE", "off").strip().lower() == "on"


def _get_sure_open(runtime: object) -> bool:
    """Let an open get-sure clause keep the normal repair rail reachable."""
    if runtime is None:
        return False
    if (getattr(runtime, "stateframe_get_sure_mode", "off") != "on"
            and os.environ.get("STATEFRAME_GET_SURE", "off").strip().lower() != "on"):
        return False
    if getattr(runtime, "stateframe_readiness_authority_mode", "off") == "on":
        return False  # r39: the readiness authority owns the gate (repair is outright there, wire3)
    try:
        from kernel_stateframe_get_sure import has_violations, runtime_table
        table = (runtime.get_sure_table() if callable(getattr(runtime, "get_sure_table", None))
                 else runtime_table(runtime))
        return has_violations(table)
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return False


def green_lock_mode() -> str:
    """Return the strictly validated green-lock mode."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on", "terminal"}:
        raise ValueError(f"{MODE_ENV} must be off, on or terminal")
    return value


def enabled() -> bool:
    """Return whether the post-green product edit lock is active."""
    return green_lock_mode() in {"on", "terminal"}


def _normalized_path(path: object) -> str:
    return str(path).replace("\\", "/").strip().lstrip("./")


def is_product_path(path: object) -> bool:
    """Use the repair loop's lexical test-file boundary for mutation decisions."""
    normalized = _normalized_path(path)
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    name = parts[-1]
    return not (
        parts[0] in {"test", "tests"}
        or "/tests/" in "/" + normalized
        or normalized.startswith("test_")
        or "/test_" in "/" + normalized
        or name == "conftest.py"
    )


def _status_is_green(value: object) -> bool:
    if isinstance(value, Mapping):
        status = value.get("result_class", value.get("result", value.get("status")))
        if isinstance(status, str):
            return status.upper() in _GREEN_STATUSES
        exit_code = value.get("exit_code", value.get("exit"))
        return exit_code == 0
    return isinstance(value, str) and value.upper() in _GREEN_STATUSES


def _selectors(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _declared_selectors(runtime: object, frame: Mapping[str, object] | None = None) -> list[str]:
    owner = getattr(runtime, "owner", None)
    for candidate in (
        getattr(owner, "f2p_ids", ()),
        getattr(owner, "f2p_inworld", ()),
        frame.get("required_public_checks") if isinstance(frame, Mapping) else (),
    ):
        selectors = _selectors(candidate)
        if selectors:
            return list(dict.fromkeys(selectors))
    if isinstance(frame, Mapping):
        contract = frame.get("completion_contract")
        if isinstance(contract, Mapping):
            return list(dict.fromkeys(_selectors(contract.get("required_public_checks"))))
    return []


def _normalize_obligation_ids(value: object) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("open_obligations", "open_ids", "ids", "obligations"):
            if key in value:
                return _normalize_obligation_ids(value[key])
        result = []
        for ident, item in value.items():
            status = item.get("status", "open") if isinstance(item, Mapping) else item
            if str(status).lower() in {"done", "dropped", "closed"}:
                continue
            if isinstance(ident, str) and ident.strip() and ident not in result:
                result.append(ident)
        return result
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            ident = item.get("id", item.get("obligation_id"))
            status = str(item.get("status", "open")).lower()
            if status in {"done", "dropped", "closed"}:
                continue
        else:
            ident = item
        if isinstance(ident, str) and ident.strip() and ident not in result:
            result.append(ident)
    return result


def _open_requirement_ids(runtime: object, frame: Mapping[str, object] | None = None) -> list[str]:
    """Open host-owned requirements: the lock's obligation view plus the completion owner's
    authoritative ``open_ids`` (the owner reported R1/R2 open at the very moment the lock engaged
    in batch 846).  Read-only; never a completion decision."""
    result = list(_open_obligation_ids(runtime, frame))
    try:
        from kernel_completion_owner import open_ids
        for ident in open_ids(runtime):
            if str(ident).strip() and str(ident) not in result:
                result.append(str(ident))
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    return result


def _open_obligation_ids(runtime: object, frame: Mapping[str, object] | None = None) -> list[str]:
    """Read host-owned open obligations without making the lock a completion gate."""
    try:
        from kernel_obligation_seed import open_obligations
    except ImportError:
        open_obligations = None
    states = []
    work_record = getattr(runtime, "work_record", None)
    if work_record is not None:
        states.append(work_record)
    if isinstance(frame, Mapping):
        if isinstance(frame.get("work_record"), Mapping):
            states.append(frame["work_record"])
        if "open_obligations" in frame:
            states.append({"open_obligations": frame["open_obligations"]})
    result: list[str] = []
    if open_obligations is not None:
        for state in states:
            try:
                values = open_obligations(state)
            except (TypeError, ValueError):
                values = ()
            for ident in _normalize_obligation_ids(values):
                if ident not in result:
                    result.append(ident)
    candidate = getattr(runtime, "open_obligations", None)
    if callable(candidate):
        try:
            values = candidate()
        except (TypeError, ValueError):
            values = ()
        for ident in _normalize_obligation_ids(values):
            if ident not in result:
                result.append(ident)
    return result


def _unverified_window_open(runtime: object) -> bool:
    try:
        from kernel_live_counterexample_challenge import unverified_window_open
        return bool(unverified_window_open(runtime))
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        return False


def widen_terminal_menu(current_tools: list[dict], open_obligations: object,
                        *, all_tools: Sequence[dict] | None = None) -> list[dict]:
    """Add progress tools to an engaged terminal menu only for open obligations.

    ``current_tools`` is the menu produced by the existing terminal filter.
    ``all_tools`` is the pre-filter surface, so the caller can restore only
    actions that were actually available before the terminal narrowing.  With
    no open obligations this returns the exact object unchanged.
    """
    ids = _normalize_obligation_ids(open_obligations)
    if not ids:
        return current_tools
    source = list(all_tools) if all_tools is not None else list(current_tools)
    result = list(current_tools)
    names = {tool.get("function", {}).get("name") for tool in result
             if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)}
    for tool in source:
        if not isinstance(tool, Mapping) or not isinstance(tool.get("function"), Mapping):
            continue
        name = tool["function"].get("name")
        if name in _PROGRESS_ACTIONS and name not in names:
            result.append(tool)
            names.add(name)
    return result


def terminal_menu_with_obligations(current_tools: list[dict], open_obligations: object,
                                   *, all_tools: Sequence[dict] | None = None) -> list[dict]:
    """Compatibility alias for runtimes that name the terminal-menu seam."""
    return widen_terminal_menu(current_tools, open_obligations, all_tools=all_tools)


def green_lock_fact(runtime: object, frame: Mapping[str, object] | None = None) -> str:
    """Return the runtime-bound typed public-green fact for host frame wiring."""
    lock = getattr(runtime, "green_lock", None)
    if isinstance(lock, GreenLock):
        if lock._coherence_enabled():
            from kernel_green_coherence import render_enabled, terminal_status
            if render_enabled(runtime):
                return terminal_status(runtime, frame).lock_text()
        return lock._green_fact(frame)
    selectors = _declared_selectors(runtime, frame)
    ids = _open_obligation_ids(runtime, frame)
    selector_text = json.dumps(selectors, ensure_ascii=False, separators=(",", ":"))
    revision = _product_revision(_current_binding(runtime))
    revision_text = json.dumps(revision, ensure_ascii=False, separators=(",", ":"), default=str)
    obligations_text = json.dumps(ids, ensure_ascii=False, separators=(",", ":")) if ids else "none"
    result = (TERMINAL_GREEN_LOCK_PREFIX + selector_text + " at revision "
              + revision_text + TERMINAL_GREEN_LOCK_SUFFIX + obligations_text)
    if os.environ.get("STATEFRAME_FRAME_CLARITY", "off") == "on":
        from kernel_frame_clarity import sanitize_text
        return sanitize_text(result)
    return result


def _product_revision(binding: object) -> object:
    if not isinstance(binding, Mapping):
        return None
    for key in ("product_revision", "artifact", "revision"):
        value = binding.get(key)
        if value not in (None, "", "MISSING", "UNKNOWN"):
            return value
    return None


def _current_binding(runtime: object) -> Mapping[str, object] | None:
    try:
        value = runtime.binding()
    except (AttributeError, OSError, UnicodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _binding_product_files(binding: Mapping[str, object] | None) -> dict[str, object]:
    files = binding.get("product_files") if isinstance(binding, Mapping) else None
    if not isinstance(files, Mapping):
        return {}
    return {
        _normalized_path(path): value
        for path, value in files.items()
        if _normalized_path(path) and is_product_path(path)
    }


def _product_files_value(binding: object) -> dict[str, object] | None:
    if not isinstance(binding, Mapping) or not isinstance(binding.get("product_files"), Mapping):
        return None
    return _binding_product_files(binding)


def _stable_product_identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                         separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _editable_product_paths(runtime: object) -> tuple[str, ...]:
    """Resolve the host-declared editable product paths for this attempt."""
    owner = getattr(runtime, "owner", None)
    for holder in (getattr(owner, "task_row", None),
                   getattr(runtime, "edit_contract", None),
                   getattr(owner, "edit_contract", None)):
        if not isinstance(holder, Mapping) or "editable_paths" not in holder:
            continue
        raw_paths = holder.get("editable_paths")
        if isinstance(raw_paths, str):
            candidates = (raw_paths,)
        elif isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (bytes, bytearray)):
            candidates = raw_paths
        else:
            continue
        paths = []
        for candidate in candidates:
            raw = str(candidate).strip().replace("\\", "/")
            while raw.startswith("./"):
                raw = raw[2:]
            parts = raw.split("/")
            if (not raw or raw.startswith("/") or ".." in parts
                    or not is_product_path(raw)):
                continue
            paths.append(raw)
        if paths:
            return tuple(sorted(set(paths)))

    target = _target_relative(runtime)
    return (target,) if target and is_product_path(target) else ()


def _workspace_product_file_digests(runtime: object,
                                    paths: Sequence[str]) -> dict[str, object] | None:
    """Hash editable files from the attempt workspace, without reading oracle state."""
    root_value = getattr(getattr(runtime, "owner", None), "root", None)
    if root_value is None or not paths:
        return None
    try:
        root = Path(root_value).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return None

    files: dict[str, object] = {}
    for relative in paths:
        raw = str(relative).replace("\\", "/").strip()
        while raw.startswith("./"):
            raw = raw[2:]
        parts = raw.split("/")
        if (not raw or raw.startswith("/") or ".." in parts
                or not is_product_path(raw)):
            return None
        candidate = root
        missing = False
        for part in parts:
            candidate = candidate / part
            try:
                if candidate.is_symlink():
                    return None
                candidate.lstat()
            except FileNotFoundError:
                missing = True
                break
            except (OSError, RuntimeError, ValueError):
                return None
        if missing:
            # Record known absence so a later file creation counts as product work.
            files[raw] = None
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
            if not candidate.is_file():
                return None
            files[raw] = "sha256:" + hashlib.sha256(candidate.read_bytes()).hexdigest()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
    return files or None


def _capture_parent_product_identity(runtime: object) -> Mapping[str, object] | None:
    """Capture a single immutable parent snapshot at GreenLock attempt startup."""
    previous = getattr(runtime, "_parent_product_identity_capture", None)
    if (isinstance(previous, Mapping)
            and isinstance(previous.get("files"), Mapping)
            and isinstance(previous.get("paths"), Sequence)):
        return previous

    paths = _editable_product_paths(runtime)
    files = _workspace_product_file_digests(runtime, paths)
    if files is None:
        return None
    method = "editable_workspace_file_sha256"
    digest = _stable_product_identity(files)
    capture: dict[str, object] = {
        "method": method,
        "paths": tuple(files),
        "files": files,
        "file_count": sum(value is not None for value in files.values()),
        "digest": digest,
    }
    setattr(runtime, "_parent_product_identity_capture", capture)
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit("parent_product_identity_captured", producer="kernel_green_lock",
                 dispatch=False, method=method, file_count=capture["file_count"],
                 digest=digest)
        except Exception:  # noqa: BLE001 - telemetry must not change the gate
            pass
    return capture


def _usable_product_revision(value: object) -> object | None:
    if value in (None, "") or str(value).strip().upper() in {"MISSING", "UNKNOWN", "NONE"}:
        return None
    return value


def product_revision_change_from_parent(runtime: object) -> tuple[bool, str, str]:
    """Compare current product identity with its host-captured parent identity.

    Product file digests are authoritative when both snapshots expose them.
    Otherwise compare explicit product revision identities. Missing parent
    identity is not evidence that product work occurred.
    """
    current = _current_binding(runtime)
    lock = getattr(runtime, "green_lock", None)
    owner = getattr(runtime, "owner", None)
    parent = None
    for holder in (lock, runtime, owner):
        if holder is None:
            continue
        for name in ("_starting_binding", "starting_binding", "parent_binding", "base_binding"):
            candidate = getattr(holder, name, None)
            if isinstance(candidate, Mapping):
                parent = candidate
                break
        if parent is not None:
            break

    current_files = _product_files_value(current)
    parent_files = _product_files_value(parent)
    captured_workspace_parent = False
    starting_editable_files = (getattr(lock, "_starting_editable_product_files", None)
                               if lock is not None else None)
    starting_editable_paths = (getattr(lock, "_starting_editable_product_paths", ())
                               if lock is not None else ())
    if isinstance(starting_editable_files, Mapping) and starting_editable_files:
        # Compare the same task-authorized workspace paths at both boundaries.
        # Product bindings can omit these paths or describe a broader artifact.
        parent_files = dict(starting_editable_files)
        current_files = _workspace_product_file_digests(runtime, starting_editable_paths)
        captured_workspace_parent = True
    if parent_files is None and lock is not None:
        starting_files = getattr(lock, "_starting_product_files", None)
        if isinstance(starting_files, Mapping) and isinstance(
                getattr(lock, "_starting_binding", None), Mapping):
            parent_files = dict(starting_files)
    current_revision = _usable_product_revision(_product_revision(current))
    if current_revision is None:
        for name in ("product_revision", "current_revision", "revision"):
            current_revision = _usable_product_revision(getattr(runtime, name, None))
            if current_revision is not None:
                break
    parent_revision = _usable_product_revision(_product_revision(parent))
    if parent_revision is None and lock is not None:
        parent_revision = _usable_product_revision(getattr(lock, "_starting_revision", None))
    if parent_revision is None:
        for holder in (runtime, owner):
            for name in ("parent_revision", "base_revision", "untouched_parent_revision",
                         "parent_digest", "base_digest"):
                candidate = (_usable_product_revision(getattr(holder, name, None))
                             if holder is not None else None)
                if candidate is not None:
                    parent_revision = candidate
                    break
            if parent_revision is not None:
                break
    row = getattr(owner, "task_row", None)
    if parent_revision is None and isinstance(row, Mapping):
        for name in ("untouched_parent_revision", "parent_revision", "base_revision",
                     "parent_digest", "base_digest"):
            candidate = _usable_product_revision(row.get(name))
            if candidate is not None:
                parent_revision = candidate
                break

    revision = str(current_revision or (
        _stable_product_identity(current_files) if current_files is not None else "UNKNOWN"))[:180]
    if captured_workspace_parent and current_files is None:
        return False, revision, "parent_product_revision_unavailable"
    if current_files is not None and parent_files is not None:
        changed = current_files != parent_files
        return (changed, revision,
                "product_file_digests_differ_from_parent" if changed
                else "product_file_digests_match_parent")
    if current_revision is not None and parent_revision is not None:
        changed = str(current_revision) != str(parent_revision)
        return (changed, revision,
                "product_revision_differs_from_parent" if changed
                else "product_revision_matches_parent")
    return False, revision, "parent_product_revision_unavailable"


# Only a POSITIVE match proves the product is the untouched parent. "parent_product_revision_unavailable"
# is unknown, NOT unchanged (fixtures and runs without parent identity must keep their prior post-green behaviour).
PROVEN_PARENT_UNCHANGED = frozenset({"product_file_digests_match_parent", "product_revision_matches_parent"})


def product_parent_proven_unchanged(runtime: object) -> bool:
    changed, _revision, comparison = product_revision_change_from_parent(runtime)
    return (not changed) and comparison in PROVEN_PARENT_UNCHANGED


def note_unchanged_parent_green_suppressed(runtime: object, *, reason: str) -> None:
    """Journal the first suppressed post-green start for each unchanged revision."""
    _changed, revision, comparison = product_revision_change_from_parent(runtime)
    seen = getattr(runtime, "_postgreen_unchanged_parent_suppression_revisions", None)
    if not isinstance(seen, set):
        seen = set()
        setattr(runtime, "_postgreen_unchanged_parent_suppression_revisions", seen)
    if revision in seen:
        return
    seen.add(revision)
    emit = getattr(getattr(runtime, "journal", None), "emit", None)
    if callable(emit):
        try:
            emit("postgreen_suppressed_unchanged_parent",
                 producer="kernel_green_lock", dispatch=False,
                 revision=revision, reason=f"{reason}: {comparison}")
        except Exception:  # noqa: BLE001 - telemetry never changes the gate
            return


def current_public_verdict_is_green(runtime: object) -> bool:
    """Use the shared public-check authority; unknown evidence is not green."""
    try:
        from kernel_greenlock_terminal_fixes import public_verdict
        return public_verdict(runtime) == "GREEN"
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _target_relative(runtime: object) -> str:
    owner = getattr(runtime, "owner", None)
    root = getattr(owner, "root", None)
    target = getattr(runtime, "target", None)
    if root is None or target is None:
        return ""
    try:
        return Path(target).resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, TypeError, ValueError):
        return ""


def _product_sources(runtime: object, binding: Mapping[str, object] | None,
                     fallback_paths: Sequence[str] = ()) -> dict[str, str]:
    root = getattr(getattr(runtime, "owner", None), "root", None)
    if root is None:
        return {}
    paths = set(_binding_product_files(binding))
    if not paths:
        paths.update(path for path in fallback_paths if is_product_path(path))
        target = _target_relative(runtime)
        if target and is_product_path(target):
            paths.add(target)
    sources: dict[str, str] = {}
    for relative in paths:
        try:
            path = Path(root) / relative
            if path.is_symlink() or not path.is_file():
                continue
            sources[relative] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, TypeError, ValueError):
            continue
    return sources


def _bound_current(receipt: Mapping[str, object], current: Mapping[str, object]) -> bool:
    after_binding = receipt.get("after_binding", receipt.get("binding"))
    return (
        receipt.get("binding_stable") is True
        and receipt.get("binding") == current
        and after_binding == current
    )


def _receipts_are_green(runtime: object, required: Sequence[str],
                        current: Mapping[str, object] | None) -> bool:
    if current is None or not required:
        return False
    return evaluate_required_tests(current, required,
                                   getattr(runtime, "test_receipts", ())).state == SATISFIED


def _frame_is_green(frame: Mapping[str, object], current_revision: object) -> bool:
    if current_revision is None:
        return False
    contract = frame.get("completion_contract")
    required = (contract.get("required_public_checks", ())
                if isinstance(contract, Mapping) else frame.get("required_public_checks", ()))
    events = []
    for key in ("test_receipts", "recent_test_feedback", "probe_receipts", "receipts"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            value = list(value.values())
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            events.extend(row for row in value if isinstance(row, Mapping))
    return evaluate_required_tests(frame, required, events).state == SATISFIED


class GreenLock:
    """Stateful lock attached to one runtime and its existing workspace bindings."""

    def __init__(self, runtime: object, *, mode: str | None = None):
        self.runtime = runtime
        self.mode = green_lock_mode() if mode is None else mode
        if self.mode not in {"off", "on", "terminal"}:
            raise ValueError(f"{MODE_ENV} must be off, on or terminal")
        self._starting_binding: Mapping[str, object] | None = None
        self._starting_revision: object = None
        self._starting_product_files: dict[str, object] = {}
        self._starting_product_sources: dict[str, str] = {}
        self._starting_editable_product_files: dict[str, object] = {}
        self._starting_editable_product_paths: tuple[str, ...] = ()
        if self.mode in {"on", "terminal"}:
            self._starting_binding = _current_binding(runtime)
            self._starting_revision = _product_revision(self._starting_binding)
            parent_identity = _capture_parent_product_identity(runtime)
            if isinstance(parent_identity, Mapping):
                parent_files = parent_identity.get("files")
                parent_paths = parent_identity.get("paths")
                if isinstance(parent_files, Mapping) and isinstance(parent_paths, Sequence):
                    self._starting_editable_product_files = dict(parent_files)
                    self._starting_editable_product_paths = tuple(
                        str(path) for path in parent_paths)
            if self.mode == "terminal":
                self._starting_product_files = _binding_product_files(self._starting_binding)
                self._starting_product_sources = _product_sources(
                    runtime, self._starting_binding, self._starting_product_files)
        self._green = False
        self._green_revision: object = None
        self._product_edit_applied = False
        self._deferred_revision = None  # green seen with open requirements: lock deferred, journaled once
        self._refused_targets: set[str] = set()
        self._overridden_targets: set[str] = set()
        self._first_action_refused: str | None = None
        self._first_action_overridden = False
        # The coherence arm counts host refusals by action kind, rather than
        # requiring byte-identical arguments.  The legacy fields above remain
        # intact for the OFF/ON compatibility path.
        self._refusal_count = 0
        self._consecutive_refusals = 0
        self._refused_action_kind: str | None = None
        self._override_action_kind: str | None = None
        self._host_refusal_count = 0
        self._post_green_refusal_count = 0
        self._post_green_refusal_state_version = None
        self._repair_blocks_done_non_progress = 0
        self._terminal_view_fingerprint: bytes | None = None
        self._terminal_view_repeats = 0
        self._terminal_fallback_revision: object = None
        if getattr(runtime, "green_lock", None) is None:
            runtime.green_lock = self

    @property
    def green(self) -> bool:
        return self._green

    @property
    def terminal_engaged(self) -> bool:
        return self.mode == "terminal" and self._green

    def _emit(self, kind: str, **fields: object) -> None:
        journal = getattr(self.runtime, "journal", None)
        if journal is not None and hasattr(journal, "emit"):
            journal.emit("green_lock_" + kind, producer="kernel_green_lock", **fields)

    def _coherence_enabled(self) -> bool:
        try:
            from kernel_green_coherence import runtime_enabled
            return runtime_enabled(self.runtime)
        except (ImportError, ValueError):
            return False

    def _reset_refusal_state(self) -> None:
        self._refused_targets.clear()
        self._overridden_targets.clear()
        self._first_action_refused = None
        self._first_action_overridden = False
        self._refusal_count = 0
        self._consecutive_refusals = 0
        self._refused_action_kind = None
        self._override_action_kind = None
        self._host_refusal_count = 0
        self._post_green_refusal_count = 0
        self._post_green_refusal_state_version = None
        try:
            from kernel_repair_blocks_done import reset_progress
            reset_progress(self.runtime)
        except (ImportError, TypeError, ValueError):
            pass

    def _readiness_snapshot(self, frame: Mapping[str, object] | None = None):
        if (getattr(self.runtime, "stateframe_readiness_authority_mode", "off") != "on"
                and not _repair_blocks_done_enabled(self.runtime)):
            return None
        try:
            from kernel_readiness_authority import runtime_snapshot
            return runtime_snapshot(self.runtime, frame)
        except (AttributeError, ImportError, TypeError, ValueError):
            return None

    def _readiness_refusal(self) -> str | None:
        decision = self._readiness_snapshot()
        refusal = getattr(decision, "refusal_text", None)
        return refusal() if callable(refusal) else None

    def _repair_blocks_done_refusal(self) -> str | None:
        if not _repair_blocks_done_enabled(self.runtime):
            return None
        try:
            from kernel_repair_blocks_done import terminal_stop_active, stopped_refusal
            if not terminal_stop_active(self.runtime):
                return None
            decision = self._readiness_snapshot()
            rows = tuple(getattr(decision, "blockers", ()) or ()) if decision is not None else ()
            return stopped_refusal(
                rows, count=getattr(self.runtime, "_repair_blocks_done_non_progress", None))
        except (ImportError, TypeError, ValueError):
            return None

    def _record_post_green_refusal(self, action_kind: str) -> bool:
        """Count every refused post-green action in the readiness path."""
        decision = self._readiness_snapshot()
        if decision is None:
            return False
        if (_repair_blocks_done_enabled(self.runtime)
                and getattr(decision, "repair_blocks_done_active", False)
                and getattr(decision, "phase", "") == "blocked_counterexample"):
            try:
                from kernel_repair_blocks_done import record_non_progress
                return record_non_progress(self.runtime, decision=decision,
                                           action_kind=action_kind)
            except (ImportError, TypeError, ValueError):
                return False
        ready_to_submit = bool(getattr(decision, "ready_to_submit", False))
        repair_optional = bool(getattr(decision, "repairable_advisories", ()))
        if not ready_to_submit and not repair_optional:
            return False
        try:
            from kernel_terminal_surface_unify import enabled as stall_enabled
            from kernel_terminal_surface_unify import end_post_green_stall
        except (ImportError, ValueError):
            return False
        if not stall_enabled(self.runtime):
            return False
        # r39: no reset on decision-version drift (a volatile advisory digest would silently
        # re-arm the count forever); _reset_refusal_state after an accepted edit/restore is the
        # only reset, so every post-green refusal counts toward the host-submit backstop.
        self._post_green_refusal_state_version = getattr(decision, "state_version", None)
        self._post_green_refusal_count += 1
        self._refused_action_kind = action_kind
        self._emit("post_green_refusal", action_kind=action_kind,
                   product_revision=self._green_revision,
                   refusal_count=self._post_green_refusal_count)
        threshold = 2 if ready_to_submit else 3
        if self._post_green_refusal_count < threshold:
            return False
        end_post_green_stall(
            self.runtime, refusal_count=self._post_green_refusal_count,
            selector=(getattr(decision, "selectors", ()) or ())[0]
            if getattr(decision, "selectors", ()) else None)
        return True

    def note_host_refusal(self, selector: str | None = None) -> bool:
        """Record a host-owned unchanged-green re-check.

        Host closure owns the refusal itself.  The lock only records the public
        action and narrows the next menu; this avoids two independent refusal
        gates consuming the same ``run_tests`` call.
        """
        # Sync through the same current-green seam used by the menu before
        # recording the refusal.  Calling _green_now() directly leaves the
        # lock's cached green state stale, so the next menu render can erase
        # this host-owned count while refreshing the lock.
        if not self._coherence_enabled() or not self.sync():
            return False
        if getattr(self.runtime, "stateframe_host_closure_mode", "off") != "on":
            return False
        self._host_refusal_count += 1
        self._refused_action_kind = "run_tests"
        self._emit("host_refusal", action_kind="run_tests", selector=selector,
                   product_revision=self._green_revision,
                   refusal_count=self._host_refusal_count,
                   owner="stateframe_host_closure")
        if self._host_refusal_count < 2:
            return False
        from kernel_terminal_surface_unify import end_post_green_stall
        end_post_green_stall(
            self.runtime, refusal_count=self._host_refusal_count, selector=selector)
        return True

    def _register_coherent_refusal(self, action_kind: str) -> bool:
        """Return whether this refusal crosses the coherent override threshold."""
        self._refusal_count += 1
        self._consecutive_refusals += 1
        if self._refused_action_kind is None:
            self._refused_action_kind = action_kind
        from kernel_green_coherence import K_CONSECUTIVE_REFUSALS
        crossed = (self._refusal_count >= 2
                   or self._consecutive_refusals >= K_CONSECUTIVE_REFUSALS)
        if crossed:
            self._override_action_kind = self._refused_action_kind or action_kind
            self._first_action_overridden = True
            self._emit("override", action_kind=self._override_action_kind,
                       product_revision=self._green_revision,
                       refusal_count=self._refusal_count,
                       consecutive_refusals=self._consecutive_refusals,
                       reason="repeated_non_done_refusals")
        return crossed

    def _current_revision(self) -> object:
        return _product_revision(_current_binding(self.runtime))

    def _capture_starting_binding_if_available(
            self, frame: Mapping[str, object] | None = None) -> bool:
        """Capture a late-bound baseline only before any product edit is known."""
        if (self.mode not in {"on", "terminal"}
                or self._starting_revision is not None
                or self._has_applied_product_edit(frame)):
            return False
        current = _current_binding(self.runtime)
        revision = _product_revision(current)
        if not isinstance(current, Mapping) or revision is None:
            return False
        self._starting_binding = dict(current)
        self._starting_revision = revision
        if self.mode == "terminal":
            self._starting_product_files = _binding_product_files(current)
            self._starting_product_sources = _product_sources(
                self.runtime, current, tuple(self._starting_product_files))
        return True

    @staticmethod
    def _row_is_applied_product_edit(row: Mapping[str, object]) -> bool:
        result = row.get("result_class", row.get("observed_result", row.get("result")))
        if str(result or "").upper() != "APPLIED":
            return False
        kind = str(row.get("kind", row.get("tool", ""))).lower()
        if kind and kind not in _EDIT_ACTIONS and kind != "edit":
            return False
        path = row.get("target_path", row.get("path", ""))
        if not path:
            attempt = row.get("attempt")
            path = attempt.get("path", "") if isinstance(attempt, Mapping) else ""
        if not path:
            changed_files = row.get("changed_files")
            if (isinstance(changed_files, Sequence)
                    and not isinstance(changed_files, (str, bytes))):
                return any(isinstance(change, Mapping)
                           and is_product_path(change.get("path", ""))
                           for change in changed_files)
        return bool(path) and is_product_path(path)

    def _has_applied_product_edit(self, frame: Mapping[str, object] | None = None) -> bool:
        if self._product_edit_applied:
            return True
        owner = getattr(self.runtime, "owner", None)
        for row in getattr(owner, "_receipts", ()):
            if isinstance(row, Mapping) and self._row_is_applied_product_edit(row):
                return True
        changes = frame.get("recent_changes") if isinstance(frame, Mapping) else None
        if isinstance(changes, Sequence) and not isinstance(changes, (str, bytes)):
            return any(isinstance(row, Mapping) and self._row_is_applied_product_edit(row)
                       for row in changes)
        return False

    def _product_changed_after_edit(self, revision: object,
                                    frame: Mapping[str, object] | None = None) -> bool:
        # Runtime setup can precede task-workspace binding. Learn the original
        # revision at the first host boundary before an accepted edit appears.
        self._capture_starting_binding_if_available(frame)
        return (self._has_applied_product_edit(frame) and self._starting_revision is not None
                and revision is not None and revision != self._starting_revision)

    def _terminal_change_diff(self) -> str:
        current = _current_binding(self.runtime)
        current_files = _binding_product_files(current)
        fallback_paths = set(self._starting_product_files)
        fallback_paths.update(self._starting_product_sources)
        current_sources = _product_sources(self.runtime, current, fallback_paths)
        paths = sorted(set(self._starting_product_files)
                       | set(current_files)
                       | set(self._starting_product_sources)
                       | set(current_sources))
        changed: list[tuple[str, str | None, str | None]] = []
        for path in paths:
            before = self._starting_product_sources.get(path)
            after = current_sources.get(path)
            if (self._starting_product_files.get(path) != current_files.get(path)
                    or before != after):
                changed.append((path, before, after))
        if not changed:
            return ("--- a/<product>\n+++ b/<product>\n"
                    "@@ product revision changed; no textual diff was captured @@")

        lines: list[str] = []
        for path, before, after in changed:
            before_known = path not in self._starting_product_files or path in self._starting_product_sources
            after_known = path not in current_files or path in current_sources
            if not before_known or not after_known:
                lines.extend((f"--- a/{path}", f"+++ b/{path}",
                              "@@ product source unavailable for this file @@"))
                continue
            diff = list(difflib.unified_diff(
                [] if before is None else before.splitlines(),
                [] if after is None else after.splitlines(),
                fromfile="a/" + path, tofile="b/" + path, n=1, lineterm=""))
            lines.extend(diff or [f"--- a/{path}", f"+++ b/{path}",
                                  "@@ no textual change @@"])
        limit = 40
        if len(lines) > limit:
            omitted = len(lines) - (limit - 1)
            lines = lines[:limit - 1] + [f"[... {omitted} diff lines omitted ...]"]
        return "\n".join(lines)

    @staticmethod
    def _receipt_revision(receipt: Mapping[str, object]) -> object:
        return _product_revision(receipt.get("after_binding", receipt.get("binding")))

    def _terminal_verification(self, frame: Mapping[str, object]) -> str:
        required = _declared_selectors(self.runtime, frame)
        receipts = [row for row in getattr(self.runtime, "test_receipts", ())
                    if isinstance(row, Mapping)]
        current = _current_binding(self.runtime)
        authority = evaluate_required_tests(current, required, receipts)
        lines = []
        if authority.state != SATISFIED:
            reason = ",".join(authority.reason_codes) or "no current evidence"
            return "; ".join(f"{selector} {authority.state} ({reason})" for selector in required)
        for selector in required:
            green_index = None
            for index in range(len(receipts) - 1, -1, -1):
                receipt = receipts[index]
                if (selector in _selectors(receipt.get("selectors", receipt.get("selector")))
                        and current is not None and _bound_current(receipt, current)
                        and _status_is_green(receipt)):
                    green_index = index
                    break
            if green_index is None:
                revision = self._green_revision
                lines.append(f"{selector} NOT RUN -> GREEN at revision {revision}")
                continue
            green_receipt = receipts[green_index]
            prior_red = next((receipt for receipt in reversed(receipts[:green_index])
                              if selector in _selectors(receipt.get(
                                  "selectors", receipt.get("selector")))
                              and not _status_is_green(receipt)), None)
            previous = "RED" if prior_red is not None else "NOT RUN"
            revision = self._receipt_revision(green_receipt) or self._green_revision
            lines.append(f"{selector} {previous} -> GREEN at revision {revision}")
        return "; ".join(lines) or "required public checks NOT RUN -> GREEN at revision " + str(
            self._green_revision)

    def terminal_view_allowed(self, view: object) -> bool:
        """Allow two identical terminal views, then expose the normal frame."""
        if not self.terminal_engaged:
            return False
        try:
            fingerprint = bytes(view) if isinstance(view, (bytes, bytearray)) else json.dumps(
                view, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                default=str).encode()
        except (TypeError, ValueError):
            fingerprint = repr(view).encode()
        if fingerprint == self._terminal_view_fingerprint:
            self._terminal_view_repeats += 1
        else:
            self._terminal_view_fingerprint = fingerprint
            self._terminal_view_repeats = 1
        if self._terminal_view_repeats < 3:
            return True
        revision = self._green_revision
        if getattr(self.runtime, "stateframe_post_green_work_menu_mode", "off") == "on":
            from kernel_post_green_menu import owns_lock
            if owns_lock(self):
                # r44 review F1: the repeat fallback changes PRESENTATION only.  Clearing the
                # green here re-opened edits as ``not_engaged`` and bypassed the cycle cap and
                # the pending-verification guard; the lock/policy state stays as it is.
                self._emit("terminal_fallback", product_revision=revision,
                           reason="third_identical_terminal_render_presentation_only",
                           consecutive_renders=self._terminal_view_repeats)
                return False

        self._terminal_fallback_revision = revision
        self._green = False
        self._green_revision = None
        self._emit("terminal_fallback", product_revision=revision,
                   reason="third_identical_terminal_render",
                   consecutive_renders=self._terminal_view_repeats)
        return False

    def _green_fact(self, frame: Mapping[str, object] | None = None) -> str:
        """Return a typed public-green fact, never a completion claim."""
        if self._coherence_enabled():
            from kernel_green_coherence import render_enabled, terminal_status
            if render_enabled(self.runtime):
                return terminal_status(self.runtime, frame).lock_text()
        selectors = _declared_selectors(self.runtime, frame)
        ids = _open_obligation_ids(self.runtime, frame)
        selector_text = json.dumps(selectors, ensure_ascii=False, separators=(",", ":"))
        revision = self._green_revision
        if revision is None:
            revision = self._current_revision()
        revision_text = json.dumps(revision, ensure_ascii=False,
                                   separators=(",", ":"), default=str)
        obligations_text = (json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
                            if ids else "none")
        result = (TERMINAL_GREEN_LOCK_PREFIX + selector_text + " at revision "
                  + revision_text + TERMINAL_GREEN_LOCK_SUFFIX + obligations_text)
        if os.environ.get("STATEFRAME_FRAME_CLARITY", "off") == "on":
            from kernel_frame_clarity import sanitize_text
            return sanitize_text(result)
        return result

    def terminal_status_line(self) -> str:
        """Return the compact typed terminal handoff fact for the current proof."""
        return self._green_fact()

    @staticmethod
    def _one_line(value: object, limit: int = 180) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[:max(1, limit - 3)] + "..."

    def terminal_frame_facts(self, frame: Mapping[str, object], *,
                             last_edit: object = None, last_step: object = None) -> str:
        """Render the bounded StateFrame handoff used only by terminal mode."""
        if not self.terminal_engaged:
            raise ValueError("terminal frame facts require an engaged terminal lock")
        selectors = _declared_selectors(self.runtime, frame)
        status = None
        if self._coherence_enabled():
            from kernel_green_coherence import render_enabled, render_terminal_facts, terminal_status
            if render_enabled(self.runtime):
                status = terminal_status(self.runtime, frame)
        lines = ["[StateFrame]"]
        if status is not None:
            lines.extend(render_terminal_facts(status, include_footer=False))
        else:
            lines.append(self.terminal_status_line())
        if selectors:
            lines.append("checks=" + ", ".join(self._one_line(item) for item in selectors))
        lines.append("change:")
        lines.extend(self._terminal_change_diff().splitlines())
        lines.append("verification: " + self._terminal_verification(frame))
        if last_step is not None and self._one_line(last_step, 320):
            lines.append("your_last_step: model-written, unverified: "
                         + self._one_line(last_step, 320))
        if last_edit is not None:
            if isinstance(last_edit, Mapping):
                path = last_edit.get("path", last_edit.get("target_path", ""))
                summary = next((last_edit.get(key) for key in (
                    "summary", "detail", "observation", "result")
                    if last_edit.get(key)), "")
                edit = " ".join(part for part in (
                    self._one_line(path), self._one_line(summary)) if part)
            else:
                edit = self._one_line(last_edit)
            if edit:
                lines.append("last_edit=" + edit)
        lines.append("revision=" + self._one_line(self._green_revision))
        if status is not None:
            lines.extend(status.footer_lines())
        rendered = "\n".join(lines) + "\n"
        if os.environ.get("STATEFRAME_FRAME_CLARITY", "off") == "on":
            from kernel_frame_clarity import sanitize_text
            rendered = sanitize_text(rendered)
            current_sources = frame.get("current_sources")
            if isinstance(current_sources, Sequence) and not isinstance(current_sources, (str, bytes)):
                labels = [str(row.get("label")) for row in current_sources
                          if isinstance(row, Mapping) and row.get("label")]
                if labels:
                    rendered += "\n".join(labels) + "\n"
        return rendered

    def record_terminal_render(self, *, omitted_block_kinds: Sequence[str],
                               char_sizes_before: Mapping[str, int],
                               char_sizes_after: Mapping[str, int]) -> None:
        """Journal the terminal render's bounded block reduction."""
        before = dict(char_sizes_before)
        after = dict(char_sizes_after)
        self._emit(
            "terminal_render",
            product_revision=self._green_revision,
            omitted_block_kinds=list(omitted_block_kinds),
            char_sizes={"before": before, "after": after},
            char_sizes_before=before,
            char_sizes_after=after,
            before_chars=sum(before.values()),
            after_chars=sum(after.values()),
        )

    def _terminal_refusal(self, action_kind: str | None = None) -> str:
        readiness_refusal = self._readiness_refusal()
        if readiness_refusal is not None:
            return readiness_refusal
        if action_kind == "run_tests" and self._coherence_enabled():
            from kernel_readiness_authority import BLOCKED_PENDING_DECISION, blocked_by_counterexample
            if blocked_by_counterexample(self.runtime):
                return BLOCKED_PENDING_DECISION + "\n" + self.terminal_status_line()
            from kernel_terminal_surface_unify import REFUSAL_LEAD
            return REFUSAL_LEAD + "\n" + self.terminal_status_line()
        return self.terminal_status_line() + REFUSAL_SUFFIX

    @staticmethod
    def _action_identity(kind: object, arguments: object) -> str:
        try:
            encoded = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str,
                                  separators=(",", ":"))
        except (TypeError, ValueError):
            encoded = repr(arguments)
        return str(kind) + ":" + encoded

    def _workspace_path(self, path: object) -> str:
        root = getattr(getattr(self.runtime, "owner", None), "root", None)
        if root is not None:
            try:
                candidate = Path(path)
                if not candidate.is_absolute():
                    candidate = Path(root) / candidate
                return candidate.resolve().relative_to(Path(root).resolve()).as_posix()
            except (OSError, TypeError, ValueError):
                pass
        return _normalized_path(path)

    def _green_now(self, frame: Mapping[str, object] | None = None) -> bool:
        if self._coherence_enabled():
            from kernel_green_coherence import green_now
            return green_now(self, frame)
        if (not self._repair_enabled()
                and getattr(self.runtime, "stateframe_host_closure_mode", "off") == "on"
                and any(assessment_is_blocking(item)
                        for item in getattr(self.runtime, "host_closure_assessments", ()) )):
            # A public GREEN is not terminal while a bound host obligation is
            # unsatisfied.  The runtime keeps the repair menu live so the
            # mutation site and its failing check can be repaired.
            return False
        current = _current_binding(self.runtime)
        revision = _product_revision(current)
        if revision is None or not self._product_changed_after_edit(revision, frame):
            return False
        required = _declared_selectors(self.runtime, frame)
        if not required:
            return False
        receipts = getattr(self.runtime, "test_receipts", ())
        if receipts:
            return evaluate_required_tests(current, required, receipts).state == SATISFIED
        if isinstance(frame, Mapping) and _frame_is_green(frame, revision):
            return True
        return False

    def _repair_enabled(self) -> bool:
        if _repair_blocks_done_enabled(self.runtime):
            return True
        try:
            from kernel_green_lock_repair import enabled as repair_enabled
            return repair_enabled()
        except (ImportError, ValueError):
            return False

    def _repair_evidence(self, frame: Mapping[str, object] | None = None):
        if not self._repair_enabled():
            return ()
        try:
            from kernel_green_lock_repair import readiness_repair_counterexamples
            return readiness_repair_counterexamples(self.runtime, frame)
        except (ImportError, TypeError, ValueError):
            return ()

    def sync(self, frame: Mapping[str, object] | None = None) -> bool:
        """Reconcile lock state with the current public proof and product binding."""
        if self.mode not in {"on", "terminal"}:
            return False
        revision = self._current_revision()
        if (self.mode == "terminal" and revision == self._terminal_fallback_revision
                and revision is not None):
            self._green = False
            self._green_revision = None
            return False
        if revision != self._terminal_fallback_revision:
            self._terminal_fallback_revision = None
        green = self._green_now(frame)
        if green:
            # 10-06 batch 846 (sample task): the parent was already public-GREEN; after the FIRST of
            # three hunks the auto-verify was green again and the lock engaged its terminal menu
            # ("done" only), so the remaining hunks could never be applied and the export graded
            # INCORRECT.  A green public check while host-owned requirements remain open is not a
            # completion signal: defer the lock, keep the repair surface, record why.
            open_requirements = _open_requirement_ids(self.runtime, frame)
            if open_requirements:
                if self._deferred_revision != revision:
                    self._deferred_revision = revision
                    self._emit("deferred", product_revision=revision,
                               reason="open_requirements", open_ids=list(open_requirements))
                self._green = False
                self._green_revision = None
                return False
            if not self._green or self._green_revision != revision:
                self._green = True
                self._green_revision = revision
                self._reset_refusal_state()
                self._terminal_view_fingerprint = None
                self._terminal_view_repeats = 0
                self._emit("engaged", product_revision=revision,
                           selectors=_declared_selectors(self.runtime, frame))
                if getattr(self.runtime, "stateframe_post_green_work_menu_mode", "off") == "on":
                    from kernel_post_green_menu import note_engaged
                    note_engaged(self, revision)
            return True
        self._green = False
        self._green_revision = None
        self._reset_refusal_state()
        return False

    def decorate_test_result(self, observation: str,
                             frame: Mapping[str, object] | None = None) -> str:
        """Offer the same lock line in a history-arm test result when it engages."""
        text = str(observation)
        if not self.sync(frame):
            return text
        message = self._green_fact(frame)
        if message in text:
            return text
        return text + ("\n" if text else "") + message

    def before_action(self, kind: object, arguments: Mapping[str, object] | None = None
                      ) -> str | None:
        """Refuse terminal mode's first repeated-safe non-done action."""
        from kernel_done_is_submit_authority import enabled as done_submit_enabled
        if (str(kind or "").strip().lower() == "done"
                and done_submit_enabled(self.runtime)):
            return None
        self._capture_starting_binding_if_available()
        from kernel_greenlock_terminal_fixes import gate_action
        return gate_action(self, kind, arguments,
                           lambda: self._before_action_legacy(kind, arguments))

    def _before_action_legacy(self, kind: object, arguments: Mapping[str, object] | None = None
                              ) -> str | None:
        """Legacy action gate retained as the OFF-switch implementation."""
        if self.mode == "terminal":
            # r43: one allow authority shared with the request menu (opt-in).
            from kernel_greenlock_menu_authority import applies, dispatch_action
            if applies(self):
                return dispatch_action(self, kind, arguments)
        action_kind = str(kind or "").strip().lower()
        arguments = arguments if isinstance(arguments, Mapping) else {}
        if self.mode == "terminal":
            from kernel_postgreen_no_veto import decide as no_veto_decide
            from kernel_postgreen_no_veto import enabled as no_veto_enabled
            if no_veto_enabled(self.runtime):
                self.sync()
                verdict = no_veto_decide(self, action_kind, arguments)
                if verdict is not None:
                    return None if verdict.allowed else verdict.message
        stopped = self._repair_blocks_done_refusal()
        if stopped is not None:
            return stopped
        if action_kind in {"read", "probe"} and _unverified_window_open(self.runtime):
            # r41b option 1: the terminal menu offers read/probe beside done while disclosed
            # unverified checks are open and this revision's challenge is unspent; the gate
            # must allow exactly what the menu offers (r39 lesson: menu == gate).
            try:
                from kernel_live_counterexample_challenge import note_window_action
                used = note_window_action(self.runtime)
            except (ImportError, AttributeError, TypeError, ValueError):
                used = None
            self._emit("action_allowed", action_kind=action_kind,
                       product_revision=self._green_revision, reason="unverified_checks_window",
                       window_actions=used)
            return None
        if _get_sure_open(self.runtime) and action_kind in (
                _EDIT_ACTIONS | frozenset({"run_tests", "test", "pytest", "predict"})):
            return None
        open_counterexamples = ()
        if self._coherence_enabled():
            from kernel_green_menu_open_obligation import (
                done_refusal, open_counterexamples as current_counterexamples,
                reset_override_state_if_evidence_changed,
            )
            reset_override_state_if_evidence_changed(self, runtime=self.runtime)
            open_counterexamples = current_counterexamples(self.runtime)
            if self.mode == "terminal" and action_kind == "done":
                refusal = done_refusal(self.runtime)
                if refusal is not None:
                    if self._record_post_green_refusal(action_kind):
                        refusal = self._repair_blocks_done_refusal() or refusal
                    self._emit("refused", action_kind=action_kind,
                               product_revision=self._green_revision,
                               reason="open_counterexample")
                    return refusal
        if self.mode != "terminal":
            return None
        repair_enabled = self._repair_enabled()
        if action_kind == "done" and not repair_enabled:
            return None
        if not self.sync():
            return None
        if repair_enabled:
            evidence = self._repair_evidence()
            if action_kind == "done":
                decision = self._readiness_snapshot()
                if decision is not None and not bool(getattr(decision, "blockers", ())):
                    return None
                if evidence:
                    from kernel_green_lock_repair import repair_refusal
                    refusal = repair_refusal(action_kind, evidence)
                    if self._record_post_green_refusal(action_kind):
                        refusal = self._repair_blocks_done_refusal() or refusal
                    self._emit("repair_refused", action_kind=action_kind,
                               product_revision=self._green_revision,
                               reason="VALID_COUNTEREXAMPLE", evidence_count=len(evidence))
                    return refusal
                return None
            from kernel_green_lock_repair import repair_refusal
            try:
                from kernel_green_lock_repair import action_linked_to_counterexample
                if action_linked_to_counterexample(
                        action_kind, arguments, self.runtime):
                    self._emit("repair_allowed", action_kind=action_kind,
                               product_revision=self._green_revision,
                               reason="OBLIGATION_LINKED", evidence_count=len(evidence))
                    if action_kind in _EDIT_ACTIONS:
                        return self.before_mutation(
                            arguments.get("path", ""), action_kind=action_kind)
                    try:
                        from kernel_repair_blocks_done import reset_progress
                        reset_progress(self.runtime)
                    except (ImportError, TypeError, ValueError):
                        pass
                    return None
            except (ImportError, TypeError, ValueError):
                pass
            refusal = (repair_refusal(action_kind, evidence)
                       if evidence else self._terminal_refusal())
            if self._record_post_green_refusal(action_kind):
                refusal = (self._repair_blocks_done_refusal()
                           or self._readiness_refusal() or refusal)
            self._emit("repair_refused", action_kind=action_kind,
                       product_revision=self._green_revision,
                       reason="UNRELATED_ACTION" if evidence else "POST_GREEN_ACTION")
            return refusal
        from kernel_frame_edit_integrity import enabled as frame_integrity_enabled
        if frame_integrity_enabled():
            self._emit("action_allowed", action_kind=action_kind,
                       product_revision=self._green_revision,
                       reason="frame_edit_integrity_recovery")
            return None
        from kernel_model_profile import green_menu_relaxed
        if green_menu_relaxed():
            return None
        if open_counterexamples and action_kind == "run_tests":
            self._emit("action_allowed", action_kind=action_kind,
                       product_revision=self._green_revision,
                       reason="open_counterexample_reverification")
            return None
        if action_kind in _EDIT_ACTIONS:
            return self.before_mutation(arguments.get("path", ""), action_kind=action_kind)
        if self._coherence_enabled():
            # Host closure performs the unchanged-green check and owns its
            # typed refusal.  Let that call reach redundant_verification so it
            # becomes a public StepLedger action exactly once.
            if (action_kind == "run_tests"
                    and getattr(self.runtime, "stateframe_host_closure_mode", "off") == "on"):
                return None
            # Count refusals by action kind, not exact argument identity.  Once
            # the threshold is reached, the selected action is the sole escape
            # hatch beside done, so varied ``why`` fields cannot reopen the
            # broad terminal menu.
            if self._override_action_kind is not None:
                if (action_kind == "run_tests"
                        and self._refused_action_kind == "run_tests"):
                    self._emit("refused", action_kind=action_kind,
                               product_revision=self._green_revision,
                               reason="recheck_menu_done_only")
                    return self._terminal_refusal(action_kind)
                if action_kind == self._override_action_kind:
                    self._emit("override", action_kind=action_kind,
                               product_revision=self._green_revision,
                               reason="narrowed_terminal_menu")
                    return None
                self._emit("refused", action_kind=action_kind,
                           product_revision=self._green_revision,
                           reason="narrowed_terminal_menu")
                return self._terminal_refusal()
            crossed = self._register_coherent_refusal(action_kind)
            self._first_action_refused = self._action_identity(action_kind, arguments)
            if crossed:
                return None
            self._emit("refused", action_kind=action_kind,
                       product_revision=self._green_revision)
            return self._terminal_refusal(action_kind)
        identity = self._action_identity(action_kind, arguments)
        if self._first_action_refused is None:
            self._first_action_refused = identity
            self._emit("refused", action_kind=action_kind,
                       product_revision=self._green_revision)
            return self._terminal_refusal(action_kind)
        if self._first_action_refused == identity and not self._first_action_overridden:
            self._first_action_overridden = True
            self._emit("override", action_kind=action_kind,
                       product_revision=self._green_revision)
            return None
        if not self._first_action_overridden:
            self._emit("refused", action_kind=action_kind,
                       product_revision=self._green_revision)
            return self._terminal_refusal()
        return None

    def before_mutation(self, path: object, *, action_kind: str = "edit") -> str | None:
        """Return refusal text before a product write, or ``None`` to dispatch it."""
        self._capture_starting_binding_if_available()
        from kernel_greenlock_terminal_fixes import gate_mutation
        return gate_mutation(self, path, action_kind,
                             lambda: self._before_mutation_legacy(path, action_kind=action_kind))

    def _before_mutation_legacy(self, path: object, *, action_kind: str = "edit") -> str | None:
        """Legacy mutation gate retained as the OFF-switch implementation."""
        if self.mode == "terminal":
            # r43: one allow authority shared with the request menu (opt-in).
            from kernel_greenlock_menu_authority import applies, dispatch_mutation
            if applies(self):
                return dispatch_mutation(self, path, action_kind=action_kind)
        from kernel_postgreen_no_veto import decide as no_veto_decide
        from kernel_postgreen_no_veto import enabled as no_veto_enabled
        if no_veto_enabled(self.runtime):
            self.sync()
            verdict = no_veto_decide(self, action_kind, {"path": path})
            if verdict is not None:
                return None if verdict.allowed else verdict.message
        stopped = self._repair_blocks_done_refusal()
        if stopped is not None:
            return stopped
        if _get_sure_open(self.runtime):
            try:
                from kernel_green_lock_repair import controller_for
                controller = controller_for(self.runtime)
                if controller is not None:
                    controller.capture_before(path)
            except (ImportError, TypeError, ValueError):
                pass
            return None
        target = self._workspace_path(path)
        if self.mode not in {"on", "terminal"} or not is_product_path(target) or not self.sync():
            return None
        if self.mode == "terminal":
            from kernel_model_profile import green_menu_relaxed
            if green_menu_relaxed():
                return None
        if self._repair_enabled():
            try:
                from kernel_green_lock_repair import action_linked_to_counterexample
                if action_linked_to_counterexample(
                        action_kind, {"path": target}, self.runtime):
                    self._emit("repair_allowed", path=target, action_kind=action_kind,
                               product_revision=self._green_revision,
                               reason="OBLIGATION_LINKED")
                    from kernel_green_lock_repair import controller_for
                    controller = controller_for(self.runtime)
                    if controller is not None:
                        controller.capture_before(path)
                    try:
                        from kernel_repair_blocks_done import reset_progress
                        reset_progress(self.runtime)
                    except (ImportError, TypeError, ValueError):
                        pass
                    return None
            except (ImportError, TypeError, ValueError):
                pass
            from kernel_green_lock_repair import repair_refusal
            evidence = self._repair_evidence()
            refusal = (repair_refusal(action_kind, evidence)
                       if evidence else (self._readiness_refusal()
                                         or self._green_fact() + REFUSAL_SUFFIX))
            if self._record_post_green_refusal(action_kind):
                refusal = (self._repair_blocks_done_refusal()
                           or self._readiness_refusal() or refusal)
            self._emit("repair_refused", path=target, action_kind=action_kind,
                       product_revision=self._green_revision,
                       reason="UNRELATED_EDIT" if evidence else "POST_GREEN_EDIT")
            return refusal
        from kernel_frame_edit_integrity import enabled as frame_integrity_enabled
        if frame_integrity_enabled():
            self._emit("action_allowed", path=target, action_kind=action_kind,
                       product_revision=self._green_revision,
                       reason="frame_edit_integrity_recovery")
            return None
        open_counterexamples = ()
        if self._coherence_enabled():
            from kernel_green_menu_open_obligation import (
                open_counterexamples as current_counterexamples,
                reset_override_state_if_evidence_changed,
            )
            reset_override_state_if_evidence_changed(self, runtime=self.runtime)
            open_counterexamples = current_counterexamples(self.runtime)
        if open_counterexamples:
            self._emit("action_allowed", path=target, action_kind=action_kind,
                       product_revision=self._green_revision,
                       reason="open_counterexample_repair")
            return None
        if (self._coherence_enabled() and self._override_action_kind is not None
                and not open_counterexamples):
            if action_kind != self._override_action_kind:
                self._emit("refused", path=target, action_kind=action_kind,
                           product_revision=self._green_revision,
                           reason="narrowed_terminal_menu")
                return self._green_fact() + REFUSAL_SUFFIX
        if target in self._refused_targets:
            if self._coherence_enabled():
                self._register_coherent_refusal(action_kind)
            self._overridden_targets.add(target)
            self._emit("override", path=target, product_revision=self._green_revision,
                       **({"action_kind": action_kind} if self.mode == "terminal" else {}))
            return None
        if self._coherence_enabled():
            crossed = self._register_coherent_refusal(action_kind)
            self._refused_targets.add(target)
            self._first_action_refused = self._action_identity(action_kind, {"path": target})
            if crossed:
                return None
            self._emit("refused", path=target, product_revision=self._green_revision,
                       **({"action_kind": action_kind} if self.mode == "terminal" else {}))
            return self._green_fact() + REFUSAL_SUFFIX
        self._refused_targets.add(target)
        if self.mode == "terminal" and self._first_action_refused is None:
            self._first_action_refused = self._action_identity(action_kind, {"path": target})
        self._emit("refused", path=target, product_revision=self._green_revision,
                   **({"action_kind": action_kind} if self.mode == "terminal" else {}))
        return self._green_fact() + REFUSAL_SUFFIX

    @staticmethod
    def _remember_product_edit_request(runtime: object) -> None:
        request_id = getattr(runtime, "current_request_id", None)
        if not isinstance(request_id, str) or not request_id.strip():
            return
        try:
            setattr(runtime, "_last_product_edit_request_id", request_id)
        except (AttributeError, TypeError):
            pass

    def applied_product_edit(self, path: object) -> None:
        """Clear the lock after an applied product edit changes the revision."""
        if self.mode in {"on", "terminal"} and is_product_path(self._workspace_path(path)):
            self._remember_product_edit_request(self.runtime)
            if getattr(self.runtime, "stateframe_post_green_work_menu_mode", "off") == "on":
                # Count the post-green work cycle while the lock still shows the green it left.
                from kernel_post_green_menu import note_edit
                note_edit(self, self._workspace_path(path))
            fallback_same_revision = False
            if self.mode == "terminal":
                current_revision = self._current_revision()
                fallback_same_revision = (
                    current_revision is not None
                    and current_revision == self._terminal_fallback_revision)
            self._product_edit_applied = True
            self._green = False
            self._green_revision = None
            if not fallback_same_revision:
                self._terminal_fallback_revision = None
                self._terminal_view_fingerprint = None
                self._terminal_view_repeats = 0
            self._reset_refusal_state()
            if getattr(self.runtime, "stateframe_readiness_authority_mode", "off") == "on":
                setattr(self.runtime, "_readiness_request_decision", None)
                setattr(self.runtime, "_readiness_decision", None)

    def repair_restored(self, snapshot: object) -> None:
        """Restore the pre-edit green state after a failed public re-check."""
        if self.mode not in {"on", "terminal"}:
            return
        self._green = bool(getattr(snapshot, "green", False))
        self._green_revision = getattr(snapshot, "green_revision", None)
        self._product_edit_applied = bool(getattr(snapshot, "product_edit_applied", False))
        self._terminal_fallback_revision = getattr(snapshot, "fallback_revision", None)
        self._terminal_view_fingerprint = None
        self._terminal_view_repeats = 0
        self._reset_refusal_state()
        if getattr(self.runtime, "stateframe_readiness_authority_mode", "off") == "on":
            setattr(self.runtime, "_readiness_request_decision", None)


def guard_product_mutation(path: object) -> str | None:
    """Find the active runtime's lock without changing callers' OFF behavior."""
    # The repro-first gate is an independent kernel switch.  Keep its lookup
    # ahead of the green-lock switch so a bug ticket cannot bypass reproduction
    # merely because post-green locking is disabled.
    if os.environ.get("STATEFRAME_REPRO_FIRST_GATE", "off") != "off":
        from kernel_repro_first import repro_first_gate_mode
        if repro_first_gate_mode() == "on":
            try:
                from kernel_pilot import ACTIVE_RUNTIME
                runtime = ACTIVE_RUNTIME.get()
            except (ImportError, LookupError):
                runtime = None
            gate = getattr(runtime, "repro_first_gate", None)
            if gate is not None:
                current_hash = None
                owner = getattr(runtime, "owner", None)
                root = getattr(runtime, "root", None) or getattr(owner, "root", None)
                if root is not None:
                    from kernel_repro_first import code_hash
                    current_hash = code_hash(root)
                from kernel_repro_first import product_edit_allowed
                if not product_edit_allowed(
                        gate, code_hash=current_hash, is_test=not is_product_path(path)):
                    # Keep the gate's canonical refusal text/event while using the
                    # shared predicate that menu projection consumes.
                    return gate.refuse_edit(
                        path, code_hash=current_hash,
                        is_test=not is_product_path(path))
                refusal = gate.refuse_edit(
                    path, code_hash=current_hash,
                    is_test=not is_product_path(path))
                if refusal is not None:
                    return refusal
    if not enabled():
        # Candidate preservation is an independent observation seam.  It must
        # still see a product mutation when terminal green-lock admission is
        # disabled, but it never changes the legacy OFF return value.
        try:
            from kernel_pilot import ACTIVE_RUNTIME
            runtime = ACTIVE_RUNTIME.get()
            from kernel_green_lock_repair import controller_for
            controller = controller_for(runtime)
            if controller is not None:
                controller.capture_before(path)
        except (ImportError, LookupError, TypeError, ValueError):
            pass
        return None
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    lock = getattr(runtime, "green_lock", None)
    from kernel_greenlock_menu_authority import applies as _authority_applies
    if _get_sure_open(runtime) and not _authority_applies(lock):
        try:
            from kernel_green_lock_repair import controller_for
            controller = controller_for(runtime)
            if controller is not None:
                controller.capture_before(path)
        except (ImportError, TypeError, ValueError):
            pass
        return None
    refusal = lock.before_mutation(path) if lock is not None else None
    if refusal is None:
        try:
            from kernel_green_lock_repair import controller_for
            controller = controller_for(runtime)
            if controller is not None:
                controller.capture_before(path)
        except (ImportError, TypeError, ValueError):
            pass
    return refusal


def guard_action(kind: object, arguments: Mapping[str, object] | None = None) -> str | None:
    """Refuse terminal mode's first non-done action at the dispatch boundary."""
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    try:
        from kernel_stateframe_hstate_plus import green_lock_guard, history_arm
        if history_arm(runtime):
            # H+state+ owns the history arm's intentionally narrow green-lock
            # rule; all other actions keep their full capability surface.
            return green_lock_guard(runtime, kind, arguments)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass
    if green_lock_mode() != "terminal":
        return None
    if str(kind or "").strip().lower() in _EDIT_ACTIONS:
        # Existing mutation guards own the per-target edit semantics and run
        # immediately before the actual write.
        return None
    lock = getattr(runtime, "green_lock", None)
    from kernel_greenlock_menu_authority import applies as _authority_applies
    if _get_sure_open(runtime) and str(kind or "").strip().lower() in {
            "run_tests", "test", "pytest", "predict", *_EDIT_ACTIONS} and not _authority_applies(lock):
        return None
    return lock.before_action(kind, arguments) if lock is not None else None


def note_applied_product_edit(path: object) -> None:
    """Notify the active runtime after a low-level editor actually writes."""
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    if runtime is not None:
        try:
            from kernel_post_edit_refresh import stage_post_edit_windows
            stage_post_edit_windows(runtime, path)
        except (ImportError, OSError, TypeError, ValueError):
            pass
        try:
            from kernel_read_reuse_visibility import clear_candidates
            clear_candidates(runtime, path=path)
        except (ImportError, OSError, TypeError, ValueError):
            pass
    lock = getattr(runtime, "green_lock", None)
    if enabled() and lock is not None:
        lock.applied_product_edit(path)
    try:
        from kernel_green_lock_repair import controller_for
        controller = controller_for(runtime)
        if controller is not None:
            controller.after_edit(path)
    except (ImportError, TypeError, ValueError):
        pass


__all__ = [
    "GREEN_LOCK_MESSAGE", "MODE_ENV", "REFUSAL_SUFFIX", "TERMINAL_GREEN_LOCK_PREFIX",
    "TERMINAL_GREEN_LOCK_SUFFIX", "GreenLock", "enabled", "green_lock_mode",
    "guard_action", "guard_product_mutation", "is_product_path",
    "green_lock_fact", "terminal_menu_with_obligations", "widen_terminal_menu",
    "note_applied_product_edit",
]
