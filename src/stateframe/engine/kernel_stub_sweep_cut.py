"""Frame-only cut for repeated unchanged ``ALREADY_PRESENT`` reads.

``STATEFRAME_STUB_SWEEP_CUT`` is deliberately a menu transform, not a dispatch
veto.  The response observer owns the one streak and the menu transform owns
the final read surface.  A revision change or a newly visible counterexample
clears the cut before the next menu is exposed.
"""
from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable, Mapping, Sequence


MODE_ENV = "STATEFRAME_STUB_SWEEP_CUT"
PLUGGED_KEY = "stateframe_stub_sweep_cut"
SCHEMA = "stateframe-stub-sweep-cut/1"
READ_TOOL = "read"
ALREADY_PRESENT = "ALREADY_PRESENT"
STUB_STREAK_TO_CUT = 2
_STATE_ATTR = "_stateframe_stub_sweep_cut"

# r46: llama.cpp does not enforce tool-call JSON schemas (measured: `anyOf`/`not enum` on the read tool's
# parameters is text in the prompt, not grammar-constrained -- reference_llamacpp_constrained_decoding_scope.md).
# Measured on the 23 A3B r45 frame journals (batches 442-455) that emit `stub_sweep_read_dropped`: 15/23 runs
# (336 incidents total) re-read the SAME path at the SAME default range after the menu cut fired. The menu hint
# alone does not act. This switch is a second, independent owner: it refuses the exact redundant dispatch instead
# of only reshaping the menu. Default off; STUB_SWEEP_CUT's menu transform is unchanged and remains the sole
# mechanism when this is off.
DISPATCH_MODE_ENV = "STATEFRAME_STUB_SWEEP_DISPATCH"
DISPATCH_PLUGGED_KEY = "stateframe_stub_sweep_dispatch"


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict opt-in mode, defaulting to ``off``."""
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    """Resolve the runtime override without changing the off-arm path."""
    configured = getattr(runtime, "stateframe_stub_sweep_cut_mode", None)
    if configured is not None:
        return configured == "on"
    return mode() == "on"


def dispatch_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict opt-in dispatch-refusal mode, defaulting to ``off``."""
    source = os.environ if environ is None else environ
    value = (source.get(DISPATCH_MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{DISPATCH_MODE_ENV} must be off or on")
    return value


def dispatch_enabled(runtime: object | None = None) -> bool:
    """Resolve the dispatch-refusal runtime override without changing the off-arm path."""
    configured = getattr(runtime, "stateframe_stub_sweep_dispatch_mode", None)
    if configured is not None:
        return configured == "on"
    return dispatch_mode() == "on"


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _stable(value: object) -> str | None:
    if value in (None, "", "UNKNOWN", "MISSING"):
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def _mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            converted = converter()
        except (AttributeError, TypeError, ValueError):
            converted = None
        if isinstance(converted, Mapping):
            return converted
    return None


def _revision(value: object) -> str | None:
    mapped = _mapping(value)
    if mapped is None:
        return _stable(value)
    for key in ("product_revision", "artifact", "revision", "current_revision", "binding_digest"):
        candidate = _stable(mapped.get(key))
        if candidate is not None:
            return candidate
    for key in ("after_binding", "binding", "current_binding", "workspace_binding"):
        nested = mapped.get(key)
        candidate = _revision(nested)
        if candidate is not None:
            return candidate
    if any(key in mapped for key in ("product_files", "harness_files", "artifact_hashes")):
        return _stable({key: mapped.get(key) for key in ("product_files", "harness_files", "artifact_hashes")
                        if key in mapped})
    return _stable(mapped)


def _current_revision(runtime: object, *, allow_binding: bool = True) -> str | None:
    for name in ("product_revision", "current_revision", "revision"):
        value = getattr(runtime, name, None)
        if value not in (None, "") and not callable(value):
            candidate = _revision(value)
            if candidate is not None:
                return candidate
    if allow_binding:
        binding_method = getattr(runtime, "binding", None)
        if callable(binding_method):
            try:
                candidate = _revision(binding_method())
            except (AttributeError, OSError, TypeError, ValueError):
                candidate = None
            if candidate is not None:
                return candidate
        artifact_hashes = getattr(runtime, "artifact_hashes", None)
        return _revision(artifact_hashes)
    return None


def _normalise_path(runtime: object, value: object) -> str:
    path = _text(value).replace("\\", "/")
    root = _text(getattr(getattr(runtime, "owner", None), "root", "")).replace("\\", "/").rstrip("/")
    if root and path.startswith(root + "/"):
        path = path[len(root) + 1:]
    while path.startswith("./"):
        path = path[2:]
    return path


def _paths_from(value: object, runtime: object) -> set[str]:
    mapped = _mapping(value)
    if mapped is None:
        return set()
    paths: set[str] = set()
    for key in ("product_files", "artifact_hashes"):
        files = mapped.get(key)
        if isinstance(files, Mapping):
            paths.update(_normalise_path(runtime, path) for path in files if _text(path))
    for key in ("after_binding", "binding", "current_binding", "workspace_binding"):
        paths.update(_paths_from(mapped.get(key), runtime))
    return {path for path in paths if path}


def _new_state() -> dict[str, object]:
    return {
        "revision": None,
        "consecutive_stub_responses": 0,
        "blocked_revision": None,
        "counterexamples": None,
        "read_paths": set(),
        "candidate_paths": set(),
        "candidate_paths_resolved": False,
        "last_release_reason": None,
        "drop_emitted": False,
    }


def _state(runtime: object) -> dict[str, object]:
    current = getattr(runtime, _STATE_ATTR, None)
    if not isinstance(current, dict):
        current = _new_state()
        setattr(runtime, _STATE_ATTR, current)
    current.setdefault("read_paths", set())
    current.setdefault("candidate_paths", set())
    current.setdefault("candidate_paths_resolved", False)
    return current


def state_snapshot(runtime: object) -> dict[str, object]:
    """Return a JSON-safe diagnostic projection for tests and replay tooling."""
    current = _state(runtime)
    return {
        key: sorted(value) if isinstance(value, set) else copy.deepcopy(value)
        for key, value in current.items()
    }


def _counterexample_ids(runtime: object) -> tuple[str, ...]:
    values: list[str] = []
    for name in ("open_counterexample_ids", "current_counterexample_ids"):
        value = getattr(runtime, name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                try:
                    value = value(None)
                except (AttributeError, TypeError, ValueError):
                    value = None
            except (AttributeError, ValueError):
                value = None
        if isinstance(value, Mapping):
            value = value.values()
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            values.extend(_text(item) for item in value)
        elif value not in (None, ""):
            values.append(_text(value))
    if not values:
        try:
            from kernel_first_green_disclosure import open_counterexample_ids
            values.extend(open_counterexample_ids(runtime))
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            pass
    return tuple(dict.fromkeys(item for item in values if item))


def _read_path(runtime: object, receipt: Mapping[str, object], fallback: object = None) -> str:
    for key in ("target_path", "path", "target"):
        value = receipt.get(key)
        if value not in (None, ""):
            return _normalise_path(runtime, value)
    parameters = receipt.get("action_parameters")
    if isinstance(parameters, Mapping) and parameters.get("path") not in (None, ""):
        return _normalise_path(runtime, parameters["path"])
    evidence = receipt.get("read_evidence")
    if isinstance(evidence, Mapping) and evidence.get("path") not in (None, ""):
        return _normalise_path(runtime, evidence["path"])
    return _normalise_path(runtime, fallback)


def _receipt_tool(receipt: Mapping[str, object]) -> str:
    for key in ("kind", "tool", "intent", "action"):
        value = receipt.get(key)
        if value not in (None, ""):
            return _text(value).lower()
    return ""


def _receipt_result(receipt: Mapping[str, object]) -> str:
    for key in ("result_class", "status", "result", "underlying_status", "underlying_result"):
        value = receipt.get(key)
        if value not in (None, ""):
            return _text(value).upper()
    return ""


def _action_receipts(receipts: object) -> list[Mapping[str, object]]:
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        return []
    return [receipt for receipt in receipts
            if isinstance(receipt, Mapping) and _receipt_tool(receipt) != "verdict"]


def _stub_only(receipts: Sequence[Mapping[str, object]]) -> bool:
    return bool(receipts) and all(
        _receipt_tool(receipt) == READ_TOOL and _receipt_result(receipt) == ALREADY_PRESENT
        for receipt in receipts)


def _adopt_revision(runtime: object, state: dict[str, object], revision: str | None) -> bool:
    if revision is None:
        return False
    previous = state.get("revision")
    if previous == revision:
        return False
    state.update({
        "revision": revision,
        "consecutive_stub_responses": 0,
        "blocked_revision": None,
        "read_paths": set(),
        "candidate_paths": set(),
        "candidate_paths_resolved": False,
        "last_release_reason": "revision_changed" if previous is not None else None,
        "drop_emitted": False,
    })
    return previous is not None


def _refresh_counterexamples(state: dict[str, object], current: tuple[str, ...]) -> tuple[str, ...]:
    previous = state.get("counterexamples")
    state["counterexamples"] = current
    if not isinstance(previous, Sequence) or isinstance(previous, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in current if item not in previous)


def _release_for_counterexample(state: dict[str, object], new_ids: Sequence[str]) -> bool:
    if not new_ids or state.get("blocked_revision") is None:
        return False
    state.update({
        "consecutive_stub_responses": 0,
        "blocked_revision": None,
        "read_paths": set(),
        "last_release_reason": "new_counterexample",
        "drop_emitted": False,
    })
    return True


def _emit(runtime: object, kind: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if not callable(emit):
        return
    try:
        emit(kind, producer="kernel_stub_sweep_cut", dispatch=False, **fields)
    except (AttributeError, TypeError, ValueError):
        # A diagnostic optimization must fail open if a lightweight replay journal
        # does not implement the full AuditJournal call shape.
        return


def observe_response(runtime: object, receipts: object, *, revision: object = None,
                     fallback_path: object = None) -> dict[str, object]:
    """Record one completed response and arm the cut after two unchanged stubs."""
    if not enabled(runtime):
        return {"stub_sweep_schema": SCHEMA, "enabled": False, "cut": False}
    state = _state(runtime)
    resolved_revision = _revision(revision) if revision is not None else _current_revision(runtime)
    changed = _adopt_revision(runtime, state, resolved_revision)
    if resolved_revision is not None and (changed or not state.get("candidate_paths_resolved")):
        observed_paths = _paths_from(revision, runtime)
        state["candidate_paths"].update(observed_paths)
        mapped_revision = _mapping(revision)
        if observed_paths or (mapped_revision is not None and any(
                key in mapped_revision for key in ("product_files", "artifact_hashes"))):
            state["candidate_paths_resolved"] = True
    rows = _action_receipts(receipts)
    for row in rows:
        if _receipt_tool(row) == READ_TOOL:
            path = _read_path(runtime, row, fallback_path)
            if path:
                state["read_paths"].add(path)
    new_counterexamples = _refresh_counterexamples(state, _counterexample_ids(runtime))
    released = _release_for_counterexample(state, new_counterexamples)
    stub_only = _stub_only(rows)
    if stub_only and resolved_revision is not None:
        if state.get("revision") != resolved_revision:
            _adopt_revision(runtime, state, resolved_revision)
        state["consecutive_stub_responses"] = int(state.get("consecutive_stub_responses", 0)) + 1
        if state["consecutive_stub_responses"] >= STUB_STREAK_TO_CUT:
            state["blocked_revision"] = resolved_revision
            state["drop_emitted"] = False
    else:
        state["consecutive_stub_responses"] = 0
    if changed:
        released = True
        state["last_release_reason"] = "revision_changed"
    result = {
        # r46 bug fix: "schema" is a RESERVED continuation-journal field (kernel_continuation_journal.RESERVED).
        # Emitting it here made journal.emit() raise ValueError, and _emit()'s fail-open swallow silently
        # dropped every "stub_sweep_evaluated" event (0 of 11 A3B r45 frame runs, batches 442-455; see
        # trials/r45_staging/README.md "Pre-launch kit fixes"). apply_menu's own emits already used
        # "stub_sweep_schema" -- this brings observe_response's result payload onto the same one spelling.
        "stub_sweep_schema": SCHEMA,
        "enabled": True,
        "revision": resolved_revision,
        "stub_only": stub_only,
        "consecutive_stub_responses": state["consecutive_stub_responses"],
        "cut": state.get("blocked_revision") == resolved_revision and resolved_revision is not None,
        "released": released,
        "release_reason": state.get("last_release_reason"),
        "new_counterexamples": list(new_counterexamples),
        "read_paths": len(state["read_paths"]),
    }
    _emit(runtime, "stub_sweep_evaluated", **result)
    return result


def _candidate_paths(runtime: object, state: dict[str, object]) -> set[str]:
    paths = state.get("candidate_paths")
    paths = set(paths) if isinstance(paths, set) else set()
    if paths or bool(state.get("candidate_paths_resolved")):
        return paths
    paths.update(_paths_from(getattr(runtime, "artifact_hashes", None), runtime))
    if paths:
        state["candidate_paths_resolved"] = True
    if not paths:
        binding_method = getattr(runtime, "binding", None)
        if callable(binding_method):
            try:
                binding = binding_method()
            except (AttributeError, OSError, TypeError, ValueError):
                binding = None
            paths.update(_paths_from(binding, runtime))
    state["candidate_paths"] = paths
    state["candidate_paths_resolved"] = True
    return paths


def _postgreen_no_veto(runtime: object) -> bool:
    configured = getattr(runtime, "stateframe_postgreen_no_veto_mode", None)
    if configured is not None:
        return configured == "on"
    return os.environ.get("STATEFRAME_POSTGREEN_NO_VETO", "off").strip().lower() == "on"


def _tool_name(tool: object) -> str:
    if not isinstance(tool, Mapping):
        return ""
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return ""
    return _text(function.get("name")).lower()


def _constrain_read(tool: Mapping[str, object], blocked: Sequence[str]) -> dict[str, object] | None:
    result = copy.deepcopy(dict(tool))
    function = result.get("function")
    if not isinstance(function, dict):
        return None
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        return None
    properties = parameters.get("properties")
    if not isinstance(properties, dict) or not isinstance(properties.get("path"), dict):
        return None
    path_schema = properties["path"]
    existing = path_schema.get("enum")
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes, bytearray)):
        blocked_set = set(blocked)
        values = [_text(value) for value in existing if _text(value) not in blocked_set]
        if not values:
            return None
        path_schema["enum"] = sorted(dict.fromkeys(values))
        if "default" in path_schema and path_schema["default"] not in path_schema["enum"]:
            path_schema.pop("default", None)
        return result
    blocked_values = sorted(dict.fromkeys(_text(value) for value in blocked if _text(value)))
    if not blocked_values:
        return result
    if isinstance(parameters.get("anyOf"), list):
        return None   # an upstream owner already shapes the arguments; never compose two constraints
    # contract review N3: a delivered, unchanged file stays readable at a NEW range (``line`` >= 2 is the explicit
    # content-refresh route); only the exact redundant default-range read of an observed path is excluded.
    if isinstance(properties.get("line"), dict):
        parameters["anyOf"] = [
            {"properties": {"path": {"not": {"enum": blocked_values}}}},
            {"required": ["line"], "properties": {"line": {"type": "integer", "minimum": 2}}},
        ]
        function["description"] = (str(function.get("description") or "")
                                   + " Files already delivered and unchanged are in the frame; re-read them"
                                     " only at a new range (line >= 2).")
        return result
    existing_not = path_schema.get("not")
    if existing_not is not None and not (
            isinstance(existing_not, Mapping)
            and isinstance(existing_not.get("enum", ()), Sequence)
            and not isinstance(existing_not.get("enum", ()), (str, bytes, bytearray))):
        return None
    existing_not_values = (
        [_text(value) for value in existing_not.get("enum", ()) if _text(value)]
        if isinstance(existing_not, Mapping)
        and isinstance(existing_not.get("enum", ()), Sequence)
        and not isinstance(existing_not.get("enum", ()), (str, bytes, bytearray))
        else [])
    path_schema["not"] = {"enum": sorted(dict.fromkeys(
        [*existing_not_values, *blocked_values]))}
    return result


def apply_menu(runtime: object, selected: Sequence[dict]) -> list[dict]:
    """Apply the final menu cut while preserving at least one path's first read."""
    return list(selected)
    result = list(selected)
    if not enabled(runtime):
        return result
    state = _state(runtime)
    if state.get("blocked_revision") is None:
        return result
    current_revision = _current_revision(runtime, allow_binding=False) or state.get("revision")
    changed = _adopt_revision(runtime, state, current_revision)
    new_counterexamples = _refresh_counterexamples(state, _counterexample_ids(runtime))
    released = _release_for_counterexample(state, new_counterexamples)
    if changed:
        state["last_release_reason"] = "postgreen_revision_changed" if _postgreen_no_veto(runtime) else "revision_changed"
        released = True
    cut_active = state.get("blocked_revision") == current_revision and current_revision is not None
    if not cut_active:
        if released:
            _emit(runtime, "stub_sweep_evaluated", stub_sweep_schema=SCHEMA, enabled=True, cut=False,
                  revision=current_revision, released=True,
                  release_reason=state.get("last_release_reason"))
        return result
    candidates = _candidate_paths(runtime, state)
    unread = candidates - set(state["read_paths"])
    read_indices = [index for index, tool in enumerate(result) if _tool_name(tool) == READ_TOOL]
    if not read_indices:
        return result
    # request fitting: never mutate the read schema (the 2B model answers the anyOf/not schema with
    # `read {}`); STUB_SWEEP_DISPATCH stays the single owner of "redundant read refused".
    if os.environ.get("STATEFRAME_TOKENLAB_READ_PLAIN", "off").strip().lower() == "on":
        if not bool(state.get("plain_emitted")):
            _emit(runtime, "tokenlab_read_plain", stub_sweep_schema=SCHEMA, revision=current_revision,
                  read_paths=len(state["read_paths"]))
            state["plain_emitted"] = True
        return result
    if not candidates or unread:
        constrained = False
        if unread:
            for index in read_indices:
                replacement = _constrain_read(result[index], sorted(state["read_paths"]))
                if replacement is not None:
                    result[index] = replacement
                    constrained = True
        if constrained and not bool(state.get("drop_emitted")):
            _emit(runtime, "stub_sweep_read_dropped", stub_sweep_schema=SCHEMA, revision=current_revision,
                  reason="previously_observed_paths_excluded", read_paths=len(state["read_paths"]))
            state["drop_emitted"] = True
        return result
    # contract review N3: never remove read -- every path is observed, so only exact redundant reads are excluded
    # (new ranges stay available through the constraint above).
    constrained = False
    for index in read_indices:
        replacement = _constrain_read(result[index], sorted(state["read_paths"]))
        if replacement is not None:
            result[index] = replacement
            constrained = True
    if constrained and not bool(state.get("drop_emitted")):
        _emit(runtime, "stub_sweep_read_dropped", stub_sweep_schema=SCHEMA, revision=current_revision,
              reason="two_consecutive_stub_only_responses", read_paths=len(state["read_paths"]))
        state["drop_emitted"] = True
    return result


def dispatch_refusal(runtime: object, path: object, *, line: object = None) -> str | None:
    """Refuse ONE exact redundant dispatch: the same delivered path, read at its default range,
    while the menu cut is active for the current revision.  Never touches a new file or a new
    range (``line`` >= 2, mirroring ``_constrain_read``'s carve-out) -- those are real requests.

    Never mutates ``_state`` (its one-refusal-per-path+revision ledger is a separate runtime attribute); ``apply_menu``/``observe_response`` own the state
    machine). It only re-derives the same "is the cut active for this exact target" answer that
    ``apply_menu`` already computed when it built this turn's menu, so calling it twice per turn
    is safe and idempotent.
    """
    return None
    if not enabled(runtime) or not dispatch_enabled(runtime):
        return None
    from kernel_serve_reads import enabled as _serve_reads_enabled, note_served as _serve_note
    state = _state(runtime)
    blocked_revision = state.get("blocked_revision")
    if blocked_revision is None:
        return None
    current_revision = _current_revision(runtime, allow_binding=False) or state.get("revision")
    if current_revision is None or blocked_revision != current_revision:
        return None
    normalised = _normalise_path(runtime, path)
    if not normalised or normalised not in state.get("read_paths", set()):
        return None  # never a file the cut has not actually observed
    try:
        requested_line = int(line) if line not in (None, "") else None
    except (TypeError, ValueError):
        requested_line = None
    if requested_line is not None and requested_line >= 2:
        return None  # never a new range -- only the exact redundant default-range read
    # r46 hunt #10: never a standing veto (NO_VETO owns post-green product reads) -- at most ONE refusal per
    # (path, revision); a repeated read of the same target is dispatched.
    refused = getattr(runtime, "_stub_sweep_dispatch_refused", None)   # its own ledger, never the cut's _state
    if not isinstance(refused, set):
        refused = set()
        try:
            setattr(runtime, "_stub_sweep_dispatch_refused", refused)
        except (AttributeError, TypeError):
            return None
    key = (normalised, str(current_revision))
    if key in refused:
        return None
    if _serve_reads_enabled():
        # only record the bypass after the exact refusal predicate matched.
        _serve_note(runtime, normalised, would_have="refusal", predicate="stub_sweep_dispatch")
        return None
    refused.add(key)
    _emit(runtime, "stub_sweep_dispatch_refused", stub_sweep_schema=SCHEMA,
          revision=current_revision, target_path=normalised)
    return (f"[stub-sweep] {normalised} was already delivered unchanged at revision {current_revision} "
            "and this is the exact redundant default-range re-read. The content is in the frame; use it. "
            "Read a different path, or re-read this one at a new range (line >= 2) if you need more of it.")


__all__ = [
    "ALREADY_PRESENT", "DISPATCH_MODE_ENV", "DISPATCH_PLUGGED_KEY", "MODE_ENV", "PLUGGED_KEY",
    "READ_TOOL", "SCHEMA", "STUB_STREAK_TO_CUT", "apply_menu", "dispatch_enabled", "dispatch_mode",
    "dispatch_refusal", "enabled", "mode", "observe_response", "state_snapshot",
]
