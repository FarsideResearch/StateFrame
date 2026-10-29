"""First-green disclosure and bounded progress bookkeeping for the r40 frame.

The feature is intentionally separate from the older green-lock text and the
older done arbiter.  With ``STATEFRAME_FIRST_GREEN_DISCLOSURE=off`` every
public helper returns its input unchanged and the existing policies retain
their old behavior.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence


MODE_ENV = "STATEFRAME_FIRST_GREEN_DISCLOSURE"
DISCLOSURE_PREFIX = "GREEN LOCK / open counterexamples disclosure:"
IDENTICAL_RERUN_PREFIX = "[first-green-disclosure] REFUSED_IDENTICAL_GREEN_RERUN:"
ACTION_HINT = (
    "make a real source edit, read a file not read since the challenge, "
    "or run a new probe before done"
)
_EDIT_KINDS = frozenset({
    "edit", "str_replace", "file_create", "replace_module", "write", "replace",
})
_READ_KINDS = frozenset({"read", "grep", "find", "search"})
_PROBE_KINDS = frozenset({"probe", "predict"})
_REFUSED_RESULTS = frozenset({"REFUSED", "REFUSED_PREDISPATCH", "ERROR", "FAILED", "FAIL", "STAGED"})  # M3
_READ_SUCCESS_RESULTS = frozenset({
    "OK", "READ", "READ_OK", "OBSERVED", "HOST_OBSERVED_READ",
    "RECALLED", "RECALLED_EVIDENCE",
})
_PROBE_SUCCESS_RESULTS = frozenset({
    "HOST_OBSERVED", "PROBED", "COMPLETED", "SUCCESS", "OK",
})
_GREEN_RESULTS = frozenset({"PASS", "PASSED", "GREEN", "SUCCESS", "OK", "CERTIFIED"})


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = (source.get(MODE_ENV, "off") or "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(runtime: object | None = None) -> bool:
    configured = getattr(runtime, "stateframe_first_green_disclosure_mode", None)
    return (configured == "on") if configured is not None else mode() == "on"


def _product_changed_from_parent(runtime: object, *, reason: str,
                                 record_suppression: bool = False) -> bool:
    from kernel_green_lock import (
        note_unchanged_parent_green_suppressed,
        product_revision_change_from_parent,
    )
    changed, _revision, _comparison = product_revision_change_from_parent(runtime)
    from kernel_green_lock import PROVEN_PARENT_UNCHANGED as _PROVEN_UNCHANGED
    changed = changed or _comparison not in _PROVEN_UNCHANGED  # unknown parent identity is not "unchanged"
    if not changed and record_suppression:
        note_unchanged_parent_green_suppressed(runtime, reason=reason)
    return changed


def begin_challenge(runtime: object, revision: object) -> None:
    """Start a new done-challenge episode and forget pre-challenge actions."""
    if not enabled(runtime):
        return
    if not _product_changed_from_parent(
            runtime, reason="first_green_disclosure_challenge", record_suppression=True):
        return
    from kernel_green_lock import current_public_verdict_is_green
    if not current_public_verdict_is_green(runtime):
        return
    runtime._fgd_challenge_revision = str(revision)
    runtime._fgd_progress = False
    runtime._fgd_read_paths = set()
    runtime._fgd_probe_fingerprints = set()
    runtime._fgd_identical_reruns = {}


def progress_since_challenge(runtime: object) -> bool:
    return (bool(getattr(runtime, "_fgd_progress", False))
            if enabled(runtime) and _product_changed_from_parent(
                runtime, reason="first_green_disclosure_progress") else False)


def note_action(runtime: object, receipt: object, material: object = "") -> bool:
    """Record only non-redundant actions completed after the challenge.

    A passing test is deliberately not progress.  Reads and probes are keyed
    so that a repeated read/probe cannot satisfy the override accidentally.
    """
    if not enabled(runtime) or getattr(runtime, "_fgd_challenge_revision", None) is None:
        return False
    if not _product_changed_from_parent(runtime, reason="first_green_disclosure_action"):
        return False
    row = _mapping(receipt)
    kind = str(row.get("kind", row.get("tool", ""))).strip().lower()
    result = str(row.get("result_class", row.get("status", ""))).strip().upper()
    if result in _REFUSED_RESULTS:
        return False
    changed = row.get("changed", True)
    if changed is False:
        return False

    if kind in _EDIT_KINDS and result == "APPLIED":
        path = _path_from(row, material)
        if path and _is_product_path(path):
            runtime._fgd_progress = True
            getattr(runtime, "_fgd_identical_reruns", {}).clear()
            _emit(runtime, "first_green_disclosure_progress", action="edit", path=path)
            return True
        return False

    if kind in _READ_KINDS and result in _READ_SUCCESS_RESULTS:
        # A model-facing command line is not read evidence.  Only the
        # executor's resolved target_path or a host-produced read_evidence
        # path can establish that a file was actually read.
        path = _read_evidence_path(row)
        if not path:
            return False
        paths = getattr(runtime, "_fgd_read_paths", None)
        if paths is None:
            paths = set()
            runtime._fgd_read_paths = paths
        if path in paths:
            return False
        paths.add(path)
        runtime._fgd_progress = True
        getattr(runtime, "_fgd_identical_reruns", {}).clear()
        _emit(runtime, "first_green_disclosure_progress", action="read", path=path)
        return True

    if kind in _PROBE_KINDS and result in _PROBE_SUCCESS_RESULTS:
        fingerprint = _probe_fingerprint(row, material)
        fingerprints = getattr(runtime, "_fgd_probe_fingerprints", None)
        if fingerprints is None:
            fingerprints = set()
            runtime._fgd_probe_fingerprints = fingerprints
        if fingerprint in fingerprints:
            return False
        fingerprints.add(fingerprint)
        runtime._fgd_progress = True
        getattr(runtime, "_fgd_identical_reruns", {}).clear()
        _emit(runtime, "first_green_disclosure_progress", action="probe",
              fingerprint=fingerprint)
        return True
    return False


def decorate_test_result(runtime: object, observation: object,
                         frame: Mapping[str, object] | None = None) -> str:
    """Attach the disclosure to the first green result for a product revision."""
    text = str(observation)
    if not enabled(runtime) or not _is_green(observation):
        return text
    from kernel_green_lock import product_revision_change_from_parent
    changed, _current_revision, comparison = product_revision_change_from_parent(runtime)
    from kernel_green_lock import PROVEN_PARENT_UNCHANGED as _PROVEN_UNCHANGED
    changed = changed or comparison not in _PROVEN_UNCHANGED  # unknown parent identity is not "unchanged"
    if not changed:
        _product_changed_from_parent(
            runtime, reason="first_green_disclosure", record_suppression=True)
        if comparison == "product_file_digests_match_parent":
            fact = ("public checks already pass on the parent revision; "
                    "they do not prove the ticket.")
        elif comparison == "product_revision_matches_parent":
            fact = ("public checks already pass on the parent revision; "
                    "they do not prove the ticket.")
        else:
            fact = ("public checks are GREEN, but a changed product revision is not established; "
                    "they do not prove the ticket.")
        return text + ("\n" if text else "") + fact
    revision = _product_revision(runtime, frame)
    key = str(revision)
    seen = getattr(runtime, "_fgd_disclosed_revisions", None)
    if seen is None:
        seen = set()
        runtime._fgd_disclosed_revisions = seen
    if DISCLOSURE_PREFIX in text:
        seen.add(key)
        return text
    if key in seen:
        return text
    seen.add(key)
    identifiers = open_counterexample_ids(runtime, frame)
    live_ce = getattr(runtime, "stateframe_live_ce_challenge_mode", "off") == "on"
    rows, error = [], None
    if live_ce:
        # r41b: the done challenge must act on exactly what was disclosed for this revision.
        # The open set is frame-scoped (readiness snapshot), so capture it HERE, where the
        # frame exists; at done time there is no frame and the recomputed set was empty
        # (smokefull-330-ldev_01: 5 IDs disclosed, done naturally accepted).
        try:
            rows = _disclosed_rows(runtime, frame, identifiers)
        except Exception as exc:  # noqa: BLE001 - witness capture must never break a test result
            error = type(exc).__name__
    body = _disclosure_body(identifiers, rows) if live_ce else None
    if body is None:
        rendered = json.dumps(list(identifiers), ensure_ascii=False, separators=(",", ":"))
        body = f"open counterexamples {rendered}"
    post_green_notice = getattr(runtime, "stateframe_post_green_work_menu_mode", "off") == "on"
    if post_green_notice:
        # STATEFRAME_POST_GREEN_WORK_MENU rule 3: never a bare PASSED.
        from kernel_post_green_menu import green_notice_head
        line = (f"{DISCLOSURE_PREFIX} {green_notice_head(runtime)} at revision {json.dumps(key)}; "
                f"{body}. If one remains, {ACTION_HINT}.")
    else:
        line = (f"{DISCLOSURE_PREFIX} first green for revision {json.dumps(key)}; "
                f"{body}. If one remains, {ACTION_HINT}.")
    _emit(runtime, "first_green_disclosure", revision=key,
          counterexample_ids=list(identifiers))
    # r41 (F4 x first-green, smokefull-319-ldev_09): the first green is usually the host's
    # automatic post-edit verification, whose synthetic call/result pair F4 never renders, so
    # the decorated text alone never reached the model.  Keep the disclosure as revision-bound
    # state for presentations that must show it (current_disclosure()).
    runtime._fgd_last_disclosure = {"revision": key, "text": line}
    if post_green_notice:
        # re-rendered live by current_disclosure(): the open-clause count must stay current
        runtime._fgd_last_disclosure["tail"] = f"; {body}. If one remains, {ACTION_HINT}."
    if live_ce:
        runtime._fgd_disclosed_open = {"revision": key, "ids": list(identifiers),
                                       "rows": rows, "capture_error": error}
        try:
            from kernel_live_counterexample_challenge import window_notice
            notice = window_notice(runtime, frame)
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            notice = None
        if notice:
            # proactive UNVERIFIED notice in the tool reply the window opens on
            line = line + "\n" + notice
    return text + ("\n" if text else "") + line


def _disclosure_body(identifiers: Sequence[str], rows: Sequence[Mapping[str, object]]) -> str | None:
    """r41b relabel: host-demoted check problems are "unverified checks", not counterexamples."""
    try:
        from kernel_live_counterexample_challenge import label_ids
        return label_ids(list(identifiers), rows)
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _disclosed_rows(runtime: object, frame: Mapping[str, object] | None,
                    identifiers: Sequence[str]) -> list[dict[str, object]]:
    """Witness rows (observed/expected) for the disclosed IDs, best effort."""
    rows: list[dict[str, object]] = []
    try:
        from kernel_green_menu_open_obligation import open_counterexamples
        for item in open_counterexamples(runtime, frame):
            to_dict = getattr(item, "to_dict", None)
            rows.append(dict(to_dict() if callable(to_dict) else item))
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        pass
    try:
        from kernel_counterexample_visibility import collect_counterexamples
        rows.extend(dict(row) for row in collect_counterexamples(runtime, frame))
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        pass
    wanted = {str(item) for item in identifiers}
    return [row for row in rows if str(row.get("check_id") or row.get("obligation_id") or "")
            in wanted]


def disclosed_open(runtime: object, frame: Mapping[str, object] | None = None
                   ) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """(ids, witness rows) disclosed for the CURRENT product revision, else ((), [])."""
    record = getattr(runtime, "_fgd_disclosed_open", None)
    if (not enabled(runtime) or not isinstance(record, Mapping)
            or not _product_changed_from_parent(runtime, reason="first_green_disclosure_open_ids")):
        return (), []
    if str(record.get("revision")) != str(_product_revision(runtime, frame)):
        return (), []
    ids = tuple(str(item) for item in record.get("ids") or () if str(item).strip())
    rows = [dict(row) for row in record.get("rows") or () if isinstance(row, Mapping)]
    return ids, rows


def current_disclosure(runtime: object, frame: Mapping[str, object] | None = None) -> str | None:
    """The first-green disclosure line while its product revision is still current, else None."""
    if (not enabled(runtime)
            or not _product_changed_from_parent(runtime, reason="first_green_disclosure_render")):
        return None
    record = getattr(runtime, "_fgd_last_disclosure", None)
    if not isinstance(record, Mapping):
        return None
    if str(record.get("revision")) != str(_product_revision(runtime, frame)):
        return None
    text = record.get("text")
    tail = record.get("tail")
    if (isinstance(tail, str) and tail
            and getattr(runtime, "stateframe_post_green_work_menu_mode", "off") == "on"):
        from kernel_post_green_menu import green_notice_head
        return (f"{DISCLOSURE_PREFIX} {green_notice_head(runtime)} at revision "
                f"{json.dumps(str(record.get('revision')))}{tail}")
    return text if isinstance(text, str) and text else None


def identical_rerun_refused_next(runtime: object) -> bool:
    """Would the NEXT unchanged-green rerun at the current revision be refused (note_identical_green_rerun)?"""
    if (not enabled(runtime)
            or not _product_changed_from_parent(runtime, reason="first_green_disclosure_rerun")):
        return False
    counts = getattr(runtime, "_fgd_identical_reruns", None)
    if not counts or not open_counterexample_ids(runtime):
        return False
    revision = str(_product_revision(runtime, None))
    return any(key[1] == revision and int(value) >= 1 for key, value in counts.items())


def note_identical_green_rerun(runtime: object, selector: object, reused: object) -> str | None:
    """Refuse the second unchanged-green rerun when a live witness is open."""
    if (not enabled(runtime) or not reused
            or not _product_changed_from_parent(runtime, reason="first_green_disclosure_rerun")):
        return None
    identifiers = open_counterexample_ids(runtime)
    key = (str(selector), str(_product_revision(runtime, None)))
    counts = getattr(runtime, "_fgd_identical_reruns", None)
    if counts is None:
        counts = {}
        runtime._fgd_identical_reruns = counts
    count = int(counts.get(key, 0)) + 1
    counts[key] = count
    if count < 2 or not identifiers:
        return None
    body = None
    if getattr(runtime, "stateframe_live_ce_challenge_mode", "off") == "on":
        _ids_now, witnesses = disclosed_open(runtime)
        body = _disclosure_body(identifiers, witnesses)
    if body is None:
        rendered = json.dumps(list(identifiers), ensure_ascii=False, separators=(",", ":"))
        body = f"open counterexamples {rendered}"
    text = (f"{IDENTICAL_RERUN_PREFIX} {body}; "
            f"the identical green check is not new evidence — {ACTION_HINT}.")
    _emit(runtime, "first_green_disclosure_rerun_refused", selector=str(selector),
          revision=key[1], counterexample_ids=list(identifiers), repeat_number=count)
    return text


def open_counterexample_ids(runtime: object, frame: Mapping[str, object] | None = None
                            ) -> tuple[str, ...]:
    """Read the live IDs without making the disclosure depend on one arm."""
    values = []
    for name in ("open_counterexample_ids", "current_counterexample_ids"):
        value = getattr(runtime, name, None)
        if callable(value):
            try:
                value = value(frame)
            except (AttributeError, TypeError, ValueError):
                value = None
        values.extend(_ids(value))
    if values:
        return tuple(dict.fromkeys(values))
    try:
        from kernel_green_menu_open_obligation import counterexample_ids
        values.extend(counterexample_ids(runtime, frame))
    except (ImportError, AttributeError, TypeError, ValueError, OSError):
        pass
    if not values:
        try:
            from kernel_counterexample_visibility import (
                collect_counterexamples, counterexample_ids,
            )
            values.extend(counterexample_ids(collect_counterexamples(runtime, frame)))
        except (ImportError, AttributeError, TypeError, ValueError, OSError):
            pass
    return tuple(dict.fromkeys(str(item) for item in values if str(item).strip()))


def _is_green(value: object) -> bool:
    if isinstance(value, Mapping):
        result = str(value.get("result_class", value.get("status", ""))).upper()
        return result in _GREEN_RESULTS or value.get("exit_code", value.get("exit")) == 0
    text = str(value)
    return bool(re.search(r"pytest exit\s+0\b", text, flags=re.IGNORECASE)
                or "ALL TESTS PASSED" in text.upper())


def _product_revision(runtime: object, frame: Mapping[str, object] | None) -> object:
    for name in ("product_revision", "current_revision", "revision"):
        value = getattr(runtime, name, None)
        if value not in (None, ""):
            return value
    lock = getattr(runtime, "green_lock", None)
    current = getattr(lock, "_current_revision", None)
    if callable(current):
        try:
            value = current()
            if value not in (None, ""):
                return value
        except (AttributeError, TypeError, ValueError):
            pass
    if isinstance(frame, Mapping):
        for name in ("product_revision", "current_revision", "revision"):
            if frame.get(name) not in (None, ""):
                return frame[name]
        binding = frame.get("binding")
        if isinstance(binding, Mapping):
            for name in ("product_revision", "artifact", "revision"):
                if binding.get(name) not in (None, ""):
                    return binding[name]
    binding = getattr(runtime, "binding", None)
    if callable(binding):
        try:
            value = binding()
        except (AttributeError, OSError, TypeError, ValueError):
            value = None
        if isinstance(value, Mapping):
            for name in ("product_revision", "artifact", "revision", "binding_digest"):
                if value.get(name) not in (None, ""):
                    return value[name]
            try:
                return json.dumps(value, sort_keys=True, default=str,
                                  separators=(",", ":"))
            except (TypeError, ValueError):
                pass
        if value not in (None, ""):
            return value
    return "unknown"


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        try:
            value = converter()
        except (AttributeError, TypeError, ValueError):
            value = None
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _path_from(row: Mapping[str, object], material: object) -> str:
    for name in ("target_path", "resolved_path", "resolved_target_path",
                 "path", "file", "filename", "target_file"):
        value = row.get(name)
        if value not in (None, ""):
            return _normalize_path(value)
    return ""


def _read_evidence_path(row: Mapping[str, object]) -> str:
    """Return only a host-resolved source path from a successful read receipt."""
    evidence = row.get("read_evidence")
    if isinstance(evidence, Mapping):
        for name in ("path", "resolved_path", "target_path"):
            value = evidence.get(name)
            if value not in (None, ""):
                return _normalize_path(value)
    for name in ("target_path", "resolved_path", "resolved_target_path"):
        value = row.get(name)
        if value not in (None, ""):
            return _normalize_path(value)
    return ""


def _normalize_path(value: object) -> str:
    return str(value).replace("\\", "/").strip().lstrip("./")


def _is_product_path(path: object) -> bool:
    try:
        from kernel_green_lock import is_product_path
        return bool(is_product_path(path))
    except (ImportError, AttributeError, TypeError, ValueError):
        normalized = _normalize_path(path)
        parts = [item for item in normalized.split("/") if item]
        return bool(parts) and parts[0] not in {"test", "tests"} and not parts[-1].startswith("test_")


def _probe_fingerprint(row: Mapping[str, object], material: object) -> str:
    explicit = row.get("fingerprint", row.get("probe_id"))
    if explicit not in (None, ""):
        return str(explicit)
    try:
        return json.dumps({"kind": row.get("kind", row.get("tool", "probe")),
                           "material": str(material),
                           "code": row.get("code", row.get("detail", ""))},
                          sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr((row.get("kind"), material))


def _ids(value: object) -> list[str]:
    if isinstance(value, Mapping):
        value = value.values()
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, Sequence):
        return []
    result = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("check_id", item.get("obligation_id", item.get("id", "")))
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _emit(runtime: object, event: str, **fields: object) -> None:
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit(event, producer="kernel_first_green_disclosure", dispatch=False, **fields)


__all__ = [
    "ACTION_HINT", "DISCLOSURE_PREFIX", "IDENTICAL_RERUN_PREFIX", "MODE_ENV",
    "begin_challenge", "decorate_test_result", "enabled", "mode", "note_action",
    "identical_rerun_refused_next", "note_identical_green_rerun", "open_counterexample_ids", "progress_since_challenge",
]
