"""Opt-in model-facing clarity projection for the state section clarity frame.

The host keeps its ordinary revisions, receipts, journals, and work records. This
module only creates presentation copies. It is intentionally dependency-light so
the legacy renderer can import it lazily and remain byte-identical when the switch
is off.
"""
from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Mapping, Sequence


ENV = "STATEFRAME_FRAME_CLARITY"
_DIGEST = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
_DIGEST_WITH_PREFIX = re.compile(r"(?i)sha256:[0-9a-f]{64}")
_REVISION_KEYS = frozenset({
    "revision", "current_revision", "product_revision", "binding_revision",
    "after_binding", "before_binding", "file_revision", "source_revision",
    "candidate_revision", "checked_revision", "pass_revision", "green_revision",
})
_FILE_REVISION_KEYS = frozenset({"file_revision", "source_revision"})
_WORKSPACE_REVISION_KEYS = frozenset({
    "current_revision", "product_revision", "binding_revision", "after_binding",
    "before_binding", "candidate_revision", "checked_revision", "pass_revision",
    "green_revision",
})
_PATH_KEYS = frozenset({"path", "file", "target_path", "file_path", "source_path"})
_CHECK_KEYS = frozenset({"name", "id", "check_name", "test_name", "test",
                         "selector", "selectors", "covered_selectors", "test_selector",
                         "test_path", "check", "checks", "target"})


def mode() -> str:
    value = os.environ.get(ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _path(value: object) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _same_revision(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    return str(left).removeprefix("sha256:") == str(right).removeprefix("sha256:")


def _raw_revision(frame: Mapping[str, object], artifacts: Mapping[str, object]) -> object:
    for key in ("workspace_revision", "current_revision", "product_revision", "revision"):
        if frame.get(key) is not None:
            return frame[key]
    binding = frame.get("binding")
    if isinstance(binding, Mapping):
        for key in ("product_revision", "current_revision", "revision"):
            if binding.get(key) is not None:
                return binding[key]
    world = frame.get("world")
    if isinstance(world, Mapping):
        binding = world.get("revision_binding")
        if isinstance(binding, Mapping):
            for key in ("product_revision", "current_revision", "revision"):
                if binding.get(key) is not None:
                    return binding[key]
    if len(artifacts) == 1:
        return next(iter(artifacts.values()))
    return None


def _artifact_revisions(frame: Mapping[str, object], observed: Sequence[Mapping[str, object]],
                        retained: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    raw = frame.get("artifact_revisions")
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            name = _path(key)
            if name:
                result[name] = value
    for collection in (observed, retained):
        for row in collection:
            if not isinstance(row, Mapping):
                continue
            name = _path(row.get("path"))
            if not name:
                continue
            value = row.get("file_revision", row.get("revision", row.get("source_revision")))
            if value is not None and name not in result:
                result[name] = value
    current_sources = frame.get("current_sources")
    if _sequence(current_sources):
        for row in current_sources:
            if isinstance(row, Mapping):
                name = _path(row.get("path"))
                if name and name not in result:
                    result[name] = row.get("revision", row.get("file_revision", row.get("version")))
    packet = frame.get("repair_packet")
    current = packet.get("current_source") if isinstance(packet, Mapping) else None
    files = current.get("files") if isinstance(current, Mapping) else None
    if _sequence(files):
        for row in files:
            if not isinstance(row, Mapping):
                continue
            name = _path(row.get("path"))
            if name and name not in result:
                result[name] = row.get("revision", row.get("file_revision"))
    return result


def _selector_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if _sequence(value):
        return [str(item) for item in value if str(item)]
    return []


def _path_values(value: object) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("path", "file", "target_path", "file_path", "source_path"):
            if isinstance(value.get(key), str):
                return [_path(value[key])]
        return []
    if _sequence(value):
        result: list[str] = []
        for item in value:
            result.extend(_path_values(item))
        return result
    return [_path(value)] if isinstance(value, str) and _path(value) else []


def _check_names(frame: Mapping[str, object]) -> list[str]:
    values: set[str] = set()
    for key in ("required_public_checks", "f2p_ids", "f2p_inworld"):
        values.update(_selector_values(frame.get(key)))
    contract = frame.get("completion_contract")
    if isinstance(contract, Mapping):
        values.update(_selector_values(contract.get("required_public_checks")))
        current = contract.get("current_public_checks")
        if isinstance(current, Mapping):
            for key in _CHECK_KEYS:
                values.update(_selector_values(current.get(key)))
    for key in ("verification", "current_public_checks"):
        current = frame.get(key)
        if isinstance(current, Mapping):
            for name in _CHECK_KEYS:
                values.update(_selector_values(current.get(name)))
    for key in ("recent_test_feedback", "test_receipts", "latest_test_results"):
        rows = frame.get(key)
        if _sequence(rows):
            for row in rows:
                if isinstance(row, Mapping):
                    for name in _CHECK_KEYS:
                        values.update(_selector_values(row.get(name)))
    return sorted(values)


def _identity(frame: Mapping[str, object], observed: Sequence[Mapping[str, object]],
              retained: Sequence[Mapping[str, object]]) -> dict[str, object]:
    artifacts = _artifact_revisions(frame, observed, retained)
    workspace_raw = _raw_revision(frame, artifacts)
    source_aliases = {name: f"F{index}"
                      for index, name in enumerate(sorted(artifacts), 1)}
    checks = _check_names(frame)
    check_aliases = {name: f"{name}@T{index}" for index, name in enumerate(checks, 1)}
    return {
        "workspace_raw": workspace_raw,
        "workspace": "W1" if workspace_raw is not None else "W0",
        "artifacts": artifacts,
        "sources": source_aliases,
        "checks": check_aliases,
    }


def _selector_alias(value: object, aliases: Mapping[str, str]) -> object:
    if isinstance(value, str):
        return aliases.get(value, value if "@T" in value else value)
    if _sequence(value):
        return [_selector_alias(item, aliases) for item in value]
    return value


def _check_is_current(row: Mapping[str, object], identity: Mapping[str, object]) -> bool:
    existing = row.get("check_valid_for_workspace")
    if type(existing) is bool:
        return existing
    workspace_raw = identity.get("workspace_raw")
    if any(row.get(key) == identity.get("workspace") for key in _REVISION_KEYS):
        return True
    for key in ("revision", "current_revision", "product_revision", "binding_revision",
                "after_binding", "checked_revision", "pass_revision"):
        if row.get(key) is not None:
            return _same_revision(row[key], workspace_raw)
    binding = row.get("binding")
    if isinstance(binding, Mapping):
        for key in ("product_revision", "current_revision", "revision"):
            if binding.get(key) is not None:
                return _same_revision(binding[key], workspace_raw)
    return False


def _replace_digest_text(text: str, workspace: str) -> str:
    text = _DIGEST_WITH_PREFIX.sub(workspace, text)
    return _DIGEST.sub(workspace, text)


def sanitize_text(value: object, *, workspace: str = "W1") -> str:
    text = _replace_digest_text(str(value or ""), workspace)
    text = re.sub(r"\bNO_COMPLETION_ATTEMPT\b", "NOT_SUBMITTED", text)
    text = re.sub(r"\bPENDING\b", "NOT_SUBMITTED", text)
    return text


def _project_value(value: object, *, key: str, path_hint: str, identity: Mapping[str, object]) -> object:
    sources = identity["sources"]
    workspace = str(identity["workspace"])
    if isinstance(value, Mapping):
        result: dict[object, object] = {}
        local_path = path_hint
        for raw_key, raw_value in value.items():
            name = str(raw_key)
            if name in _PATH_KEYS and isinstance(raw_value, str):
                local_path = _path(raw_value)
            output_key = (_replace_digest_text(raw_key, workspace)
                          if isinstance(raw_key, str) else raw_key)
            result[output_key] = _project_value(raw_value, key=name, path_hint=local_path,
                                                identity=identity)
        return result
    if _sequence(value):
        return [_project_value(item, key=key, path_hint=path_hint, identity=identity)
                for item in value]
    if isinstance(value, str):
        if key in {"status", "result", "outcome", "reason", "reason_code", "reason_codes"}:
            if value == "PENDING":
                return "NOT_SUBMITTED"
            if value == "NO_COMPLETION_ATTEMPT":
                return "NOT_SUBMITTED"
        if key in _FILE_REVISION_KEYS and path_hint in sources:
            return sources[path_hint]
        if key in _WORKSPACE_REVISION_KEYS:
            return workspace
        return _replace_digest_text(value, workspace)
    return value


def _prune_negative_actions(frame: dict[str, object]) -> None:
    rows = frame.get("negative_actions", frame.get("recent_negative_actions"))
    if not _sequence(rows):
        return
    changed: set[str] = set()
    for change in frame.get("recent_changes", ()) if _sequence(frame.get("recent_changes")) else ():
        if not isinstance(change, Mapping):
            continue
        result = str(change.get("observed_result", change.get("result_class", change.get("result", "")))).upper()
        if result != "APPLIED":
            continue
        paths = change.get("changed_files", change.get("paths", change.get("path")))
        changed.update(item for item in _path_values(paths) if item)
    green_pass = False
    for key in ("recent_test_feedback", "test_receipts", "last_action"):
        values = frame.get(key)
        if isinstance(values, Mapping):
            values = [values]
        if not _sequence(values):
            continue
        green_pass = green_pass or any(
            isinstance(item, Mapping)
            and str(item.get("result", item.get("status", item.get("result_class", "")))).upper()
            in {"GREEN", "PASS", "PASSED", "CERTIFIED", "REUSED_VERIFICATION", "OK"}
            for item in values
        )
    kept = []
    for row in rows:
        if not isinstance(row, Mapping):
            kept.append(row)
            continue
        arguments = row.get("arguments")
        row_path = _path(row.get("path", row.get("target_path", row.get("file_path"))))
        if not row_path and isinstance(arguments, Mapping):
            row_path = _path(arguments.get("path", arguments.get("target_path")))
        kind = str(row.get("kind", row.get("reason", row.get("action", row.get("result", ""))))).lower()
        resolved = row_path and any(
            row_path == item or row_path.endswith("/" + item) or item.endswith("/" + row_path)
            for item in changed)
        green_resolved = green_pass and ("green" in kind or "read" in kind)
        if (resolved and ("refus" in kind or "reject" in kind or row.get("resolved") is True)) or green_resolved:
            continue
        kept.append(row)
    if "negative_actions" in frame:
        frame["negative_actions"] = kept
    if "recent_negative_actions" in frame:
        frame["recent_negative_actions"] = kept


def _prune_empty(frame: dict[str, object]) -> None:
    # Retained source bytes are delivered once as source blocks below.  The
    # graph's candidate/reference catalogs are navigation telemetry, not facts
    # needed for the next decision.
    frame.pop("retained_context", None)
    cards = frame.get("pattern_cards")
    if (isinstance(cards, Mapping) and _sequence(cards.get("cards"))
            and not cards.get("cards")):
        frame.pop("pattern_cards", None)
    closure = frame.get("host_closure")
    if isinstance(closure, Mapping):
        assessments = closure.get("assessments")
        receipts = closure.get("receipts")
        if (not _sequence(assessments) or not assessments) and (not _sequence(receipts) or not receipts):
            if not any(closure.get(key) for key in ("active_gap", "open", "required")):
                frame.pop("host_closure", None)
    for key in (
        "frame_series", "portfolio", "generated_artifact_catalog", "generated_artifacts_catalog",
        "retention_candidates", "retention_candidate_catalog", "retention_references",
        "retention_reference_catalog", "retained_catalog", "retained_context_catalog",
        "settled_state",
    ):
        value = frame.get(key)
        if key in {"frame_series", "portfolio"}:
            frame.pop(key, None)
            continue
        if (value is None or (isinstance(value, Mapping) and not value)
                or (_sequence(value) and not value)):
            frame.pop(key, None)
    for key in ("generated_artifacts", "retained_context"):
        value = frame.get(key)
        if isinstance(value, Mapping):
            useful = [item for name, item in value.items()
                      if name not in {"catalog", "candidates", "references", "candidate_refs"}
                      and item not in (None, [], {})]
            if not useful:
                frame.pop(key, None)
        elif (value is None or (isinstance(value, Mapping) and not value)
              or (_sequence(value) and not value)):
            frame.pop(key, None)
    _prune_negative_actions(frame)


def _changed_paths(frame: Mapping[str, object]) -> set[str]:
    result: set[str] = set()
    rows = frame.get("recent_changes")
    if not _sequence(rows):
        return result
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        status = str(row.get("observed_result", row.get("result_class", row.get("result", "")))).upper()
        if status != "APPLIED":
            continue
        values = row.get("changed_files", row.get("paths", row.get("path")))
        result.update(item for item in _path_values(values) if item)
    return result


def _claim_mentions_path(item: object, paths: set[str]) -> bool:
    if not paths:
        return False
    text = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (Mapping, list)) else str(item)
    normalized = text.replace("\\", "/")
    return any(path in normalized or normalized.find(path.rsplit("/", 1)[-1]) >= 0 for path in paths)


def drop_stale_hypotheses(frame: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(frame))
    paths = _changed_paths(result)
    if not paths:
        return result
    for key in ("hypotheses_not_facts", "hypotheses"):
        values = result.get(key)
        if _sequence(values):
            result[key] = [item for item in values if not _claim_mentions_path(item, paths)]
    return result


def _source_headers(identity: Mapping[str, object], *, omit: set[str] | None = None) -> list[dict[str, str]]:
    from kernel_read_fixes import plain_source_labels_enabled, source_header
    sources = identity["sources"]
    omitted = omit or set()
    result = []
    for path, alias in sorted(sources.items()):
        if path in omitted:
            continue
        label = (source_header(path, role="current")
                 if plain_source_labels_enabled()
                 else f"CURRENT_SOURCE {path} version={alias}")
        result.append({"path": path, "version": alias,
                       "source_ref": f"{path}@{alias}", "label": label})
    return result


def project_model_frame(frame: Mapping[str, object], *, current_path: str | None = None,
                        observed_sources: Sequence[Mapping[str, object]] = (),
                        retained_evidence: Sequence[Mapping[str, object]] = ()) -> dict[str, object]:
    """Return a compact, typed, non-authoritative model projection."""
    public = drop_stale_hypotheses(frame)
    _prune_empty(public)
    raw_public = copy.deepcopy(public)
    identity = _identity(public, observed_sources, retained_evidence)
    observed_paths = {_path(row.get("path")) for row in observed_sources
                      if isinstance(row, Mapping) and _path(row.get("path"))}
    public = _project_value(public, key="", path_hint="", identity=identity)
    artifacts = public.get("artifact_revisions")
    if isinstance(artifacts, Mapping):
        public["artifact_revisions"] = {
            _path(path): identity["sources"].get(_path(path), "F0")
            for path in sorted(artifacts)
        }
    public["workspace_snapshot"] = identity["workspace"]
    public["identity_aliases"] = {
        "workspace": identity["workspace"],
        "sources": {path: f"{path}@{alias}" for path, alias in sorted(identity["sources"].items())},
        "checks": dict(sorted(identity["checks"].items())),
    }
    public["current_sources"] = _source_headers(identity, omit=observed_paths)

    def project_check(value: object, raw_value: object = None) -> object:
        if not isinstance(value, Mapping):
            return value
        row = copy.deepcopy(dict(value))
        for key in _CHECK_KEYS:
            if key in row:
                row[key] = _selector_alias(row[key], identity["checks"])
        if "check" not in row:
            selectors = next((row.get(key) for key in
                              ("selector", "selectors", "covered_selectors", "test_selector", "target")
                              if row.get(key) is not None), None)
            aliases = _selector_alias(selectors, identity["checks"])
            if aliases not in (None, [], ""):
                row["check"] = aliases[0] if _sequence(aliases) and len(aliases) == 1 else aliases
        source_row = raw_value if isinstance(raw_value, Mapping) else value
        row["check_valid_for_workspace"] = _check_is_current(source_row, identity)
        for key in _REVISION_KEYS:
            if key in row:
                row[key] = identity["workspace"]
        return row

    contract = public.get("completion_contract")
    if isinstance(contract, Mapping):
        contract = dict(contract)
        contract["required_public_checks"] = _selector_alias(
            contract.get("required_public_checks", []), identity["checks"])
        if isinstance(contract.get("current_public_checks"), Mapping):
            raw_contract = raw_public.get("completion_contract")
            raw_checks = (raw_contract.get("current_public_checks")
                          if isinstance(raw_contract, Mapping) else None)
            contract["current_public_checks"] = project_check(
                contract["current_public_checks"], raw_checks)
        public["completion_contract"] = contract
    for key in ("required_public_checks", "f2p_ids", "f2p_inworld"):
        if key in public:
            public[key] = _selector_alias(public[key], identity["checks"])
    for key in ("verification", "current_public_checks"):
        if isinstance(public.get(key), Mapping):
            public[key] = project_check(public[key], raw_public.get(key))
    for key in ("recent_test_feedback", "test_receipts", "latest_test_results"):
        rows = public.get(key)
        if _sequence(rows):
            raw_rows = raw_public.get(key)
            public[key] = [
                project_check(row, raw_rows[index] if _sequence(raw_rows) and index < len(raw_rows) else None)
                if isinstance(row, Mapping) else row
                for index, row in enumerate(rows)
            ]
    for key in ("atomic_completion", "completion_decision", "completion_verdict"):
        value = public.get(key)
        if isinstance(value, Mapping):
            public[key] = {
                "submission_status": _completion_value_status(value),
                "authority": value.get("authority", "host_completion_reducer"),
            }
    if isinstance(public.get("completion_contract"), Mapping):
        contract = dict(public["completion_contract"])
        atomic = contract.get("atomic_completion")
        if isinstance(atomic, Mapping):
            contract["atomic_completion"] = {
                "submission_status": _completion_value_status(atomic),
                "authority": atomic.get("authority", "host_completion_reducer"),
            }
        public["completion_contract"] = contract
    # Revision fields which are not part of an explicit check are still typed.
    public = _project_value(public, key="", path_hint="", identity=identity)
    public["workspace_snapshot"] = identity["workspace"]
    public["current_sources"] = _source_headers(identity, omit=observed_paths)
    return public


def project_serialized_stateframe(text: str) -> str:
    """Project a lean ``[StateFrame]`` block without changing its envelope."""
    if not enabled() or not isinstance(text, str):
        return text
    if not text.startswith("[StateFrame]\n"):
        return sanitize_text(text)
    try:
        value = json.loads(text.split("\n", 1)[1])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return sanitize_text(text)
    if not isinstance(value, Mapping):
        return sanitize_text(text)
    return "[StateFrame]\n" + json.dumps(project_model_frame(value), ensure_ascii=False,
                                           sort_keys=True, separators=(",", ":"))


def current_source_block(path: str, content: str, version: str) -> str:
    from kernel_read_fixes import plain_source_labels_enabled, source_header
    if plain_source_labels_enabled():
        return f"{source_header(path, role='current')}\n{content}"
    return f"CURRENT_SOURCE {path} version={version}\n{content}"


def current_source_alias(frame: Mapping[str, object], path: str,
                         revision: object = None) -> str:
    existing = frame.get("identity_aliases")
    if isinstance(existing, Mapping) and isinstance(existing.get("sources"), Mapping):
        value = existing["sources"].get(_path(path))
        if (isinstance(value, str) and "@" in value
                and not (value.endswith("@F0") and revision is not None)):
            return value
    name = _path(path)
    source_frame = frame
    if revision is not None:
        artifacts = frame.get("artifact_revisions")
        if not isinstance(artifacts, Mapping) or name not in artifacts:
            source_frame = dict(frame)
            source_frame["artifact_revisions"] = dict(artifacts) if isinstance(artifacts, Mapping) else {}
            source_frame["artifact_revisions"][name] = revision
    identity = _identity(source_frame, (), ())
    return f"{name}@{identity['sources'].get(name, 'F0')}"


def _status(value: object) -> str:
    text = str(value or "").upper()
    if text in {"UNVERIFIED", "BLOCKED", "UNKNOWN"}:
        return "BLOCKED"                          # not yet checked is not a failure (r45 triage)
    return "PASS" if text in {"GREEN", "PASS", "PASSED", "CERTIFIED", "SATISFIED", "ACCEPTED"} else "FAIL"


def _current_check(frame: Mapping[str, object]) -> Mapping[str, object] | object:
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping):
        return {"status": terminal.get("public_check_status", "BLOCKED"),
                "revision": terminal.get("current_revision"),
                "selectors": terminal.get("selectors", ())}
    contract = frame.get("completion_contract")
    if isinstance(contract, Mapping) and contract.get("current_public_checks") is not None:
        return contract["current_public_checks"]
    return frame.get("verification")


def _completion_value_status(value: object) -> str:
    if isinstance(value, Mapping):
        if (value.get("accepted") is True
                or str(value.get("outcome", value.get("state", ""))).upper()
                in {"ACCEPTED", "COMPLETE", "SATISFIED"}):
            return "ACCEPTED"
        if str(value.get("outcome", value.get("state", ""))).upper() in {
                "REJECTED", "BLOCKED", "VIOLATED", "NEEDS_WORK"}:
            return "REJECTED"
    return "NOT_SUBMITTED"


def _completion_status(frame: Mapping[str, object]) -> str:
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping):
        submission = terminal.get("submission")
        if isinstance(submission, Mapping):
            return "ACCEPTED" if str(submission.get("state", "")).upper() == "READY" else "REJECTED"
    value = frame.get("atomic_completion",
                      frame.get("completion_decision",
                               frame.get("completion_verdict",
                                        frame.get("completion_submission"))))
    return _completion_value_status(value)


def _requirements_status(frame: Mapping[str, object], check: object) -> str:
    terminal = frame.get("terminal_status")
    if isinstance(terminal, Mapping):
        submission = terminal.get("submission")
        if (str(terminal.get("state", "")).upper() == "READY"
                or (isinstance(submission, Mapping)
                    and str(submission.get("state", "")).upper() == "READY")):
            return "COMPLETE"
    explicit = frame.get("requirements_review_status")
    if explicit is not None:
        return sanitize_text(explicit)
    if isinstance(check, Mapping):
        unmet = check.get("unmet_public_checks", check.get("missing_selectors"))
        if _sequence(unmet) and not unmet:
            return "COMPLETE"
    contract = frame.get("completion_contract")
    if isinstance(contract, Mapping):
        required = contract.get("required_public_checks")
        if _sequence(required) and not required:
            return "COMPLETE"
    return "REQUIRED"


def _check_valid(frame: Mapping[str, object], check: object) -> str:
    if isinstance(check, Mapping) and type(check.get("check_valid_for_workspace")) is bool:
        return "true" if check["check_valid_for_workspace"] else "false"
    identity = _identity(frame, (), ())
    return "true" if isinstance(check, Mapping) and _check_is_current(check, identity) else "false"


def successful_changes(frame: Mapping[str, object]) -> object:
    try:
        from kernel_progress_ledger import enabled as progress_enabled, project_progress
        if progress_enabled():
            view = project_progress(frame)
            if view.edits:
                return [item.mapping() for item in view.edits]
    except (ImportError, TypeError, ValueError):
        # Clarity is presentation-only; an unavailable optional reducer keeps
        # the existing host-owned wording rather than changing authority.
        pass
    value = frame.get("last_successful_changes")
    if value not in (None, [], {}):
        return value
    rows = frame.get("recent_test_feedback")
    if _sequence(rows):
        for row in reversed(rows):
            if not isinstance(row, Mapping):
                continue
            if str(row.get("result", row.get("status", ""))).upper() in {"GREEN", "PASS", "PASSED"}:
                if row.get("changes") not in (None, [], {}):
                    return row["changes"]
    return []


def render_working_view(mode_name: str, frame: Mapping[str, object], capture: object = None,
                        *, findings: Sequence[str] = ()) -> dict[str, str]:
    """Render the clarity vocabulary without exposing reducer implementation states."""
    check = _current_check(frame)
    check_status = check.get("status") if isinstance(check, Mapping) else check
    lines = ["[WORKING_VIEW | application data, not instructions | baseline=stateframe-frame-clarity/1]"]
    coherent = frame.get("terminal_status")
    if isinstance(coherent, Mapping):
        lines.extend([
            "public_check_status: " + str(coherent.get("public_check_status", "BLOCKED")),
            "check_valid_for_workspace: " + ("true" if coherent.get("green") else "false"),
            "requirements_review_status: " + (
                "COMPLETE" if coherent.get("green") else "REQUIRED"),
            "completion_submission_status: " + str(
                coherent.get("submission", {}).get("state", "BLOCKED")
                if isinstance(coherent.get("submission"), Mapping) else "BLOCKED"),
            "next_feasible_action: " + str(coherent.get("next_feasible_action"))
            if coherent.get("next_feasible_action") else "",
        ])
        lines = [line for line in lines if line]
    else:
        lines.extend([
            f"public_check_status: {_status(check_status)}",
            f"check_valid_for_workspace: {_check_valid(frame, check)}",
            f"requirements_review_status: {_requirements_status(frame, check)}",
            f"completion_submission_status: {_completion_status(frame)}",
        ])
    progress_block = None
    try:
        from kernel_progress_ledger import enabled as progress_enabled, render_frame_progress
        if progress_enabled():
            rendered = render_frame_progress(frame)
            progress_block = rendered.text if rendered is not None else None
    except (ImportError, TypeError, ValueError):
        progress_block = None
    if progress_block:
        lines.append(progress_block)
    elif frame.get("recent_changes"):
        lines.append("changes since last check: none")
        lines.append("last_successful_changes: " + json.dumps(successful_changes(frame),
                                                              ensure_ascii=False, separators=(",", ":")))
    action = frame.get("last_action")
    if isinstance(action, Mapping):
        payload = {key: value for key, value in action.items() if key not in {"stated_intent"}}
        lines.append("last_action: " + sanitize_text(json.dumps(payload, ensure_ascii=False,
                                                                 sort_keys=True, separators=(",", ":"))))
    if mode_name == "findings":
        lines.extend(findings)
    if mode_name in {"resolved", "actions", "decision", "findings"}:
        passage = ""
        if capture is not None and mode_name == "resolved":
            passage = sanitize_text(getattr(capture, "response_reasoning_content", ""))
            response = sanitize_text(getattr(capture, "response_content", ""))
            if response:
                passage = (passage + "\n" if passage else "") + response
            if passage:
                lines.extend([
                    "MODEL_STATEMENT (unverified prior model text):", "<<<", passage, ">>>",
                ])
        if mode_name in {"actions", "decision", "findings"}:
            lines.append("recorded_action_is_host_observed: true")
    return {"role": "user", "content": "\n".join(lines)}


def bootstrap_message(task_row: Mapping[str, object], *, path: str, source: str,
                      frame: Mapping[str, object],
                      sent_messages: Sequence[Mapping[str, object]] = ()) -> str:
    from kernel_read_fixes import plain_source_labels_enabled, source_header
    ticket = next((task_row.get(key) for key in
                   ("msg", "brief", "_bridge_ordinary_brief", "symptom")
                   if isinstance(task_row.get(key), str) and task_row.get(key).strip()), "")
    public_source = next((task_row.get(key) for key in
                          ("public_test_source", "public_test_src", "test_source", "test_src")
                          if isinstance(task_row.get(key), (str, Mapping))), None)
    version = "F1"
    current_sources = frame.get("current_sources")
    if _sequence(current_sources):
        row = next((item for item in current_sources
                    if isinstance(item, Mapping) and _path(item.get("path")) == _path(path)), None)
        if isinstance(row, Mapping) and row.get("version"):
            version = str(row["version"])
    delivered = (source_header(path, role="working")
                 if plain_source_labels_enabled() else f"{path}@{version}")
    from kernel_frame_diet import TICKET_REFERENCE, ticket_already_sent
    ticket_text = sanitize_text(ticket)
    if ticket_already_sent(ticket_text, sent_messages):
        ticket_text = TICKET_REFERENCE
    lines = ["[FRAME_CLARITY_BOOTSTRAP]", "FULL_TICKET:", ticket_text,
             "WORKING_SOURCE_DELIVERED:", delivered]
    if public_source is not None:
        lines.extend(["PUBLIC_TEST_SOURCE:", sanitize_text(
            json.dumps(public_source, ensure_ascii=False) if isinstance(public_source, Mapping)
            else public_source)])
    lines.append("Use the current contract and host facts; do not infer completion from an unsubmitted state.")
    return "\n".join(lines)


CONTINUATION_CONTRACT = (
    "FRAME_CLARITY_CONTINUATION: use the existing ticket and current host state; "
    "take the next justified action, then rely on the host result. Do not restate the ticket."
)


def continuation_state(frame: Mapping[str, object]) -> str:
    check = _current_check(frame)
    return CONTINUATION_CONTRACT + "\n" + json.dumps({
        "public_check_status": _status(check.get("status") if isinstance(check, Mapping) else check),
        "requirements_review_status": _requirements_status(frame, check),
        "completion_submission_status": _completion_status(frame),
        "workspace_snapshot": "W1",
    }, ensure_ascii=False, sort_keys=True)


def continuation_text(content: object) -> str:
    """Keep continuation state while replacing its procedural instruction tail."""
    text = str(content or "")
    if text.startswith("[Continuation]\n"):
        try:
            value = json.loads(text.split("\n", 1)[1])
            if isinstance(value, Mapping):
                value = dict(value)
                value["instructions"] = CONTINUATION_CONTRACT
                return "[Continuation]\n" + json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (IndexError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return CONTINUATION_CONTRACT + "\n" + sanitize_text(text)


def continuation_system_text(content: object) -> str:
    """Remove the long first-request checklist/example from later system text."""
    text = str(content or "")
    marker = "How to do a ticket like this"
    end = "What you may edit"
    if marker in text and end in text:
        before, remainder = text.split(marker, 1)
        _, after = remainder.split(end, 1)
        text = before + CONTINUATION_CONTRACT + "\n\n" + end + after
    text = re.sub(r"(?im)^\s*\d+\.\s+Turn the ticket into a checklist[^\n]*\n?", "", text)
    text = re.sub(r"(?im)^\s*\d+\.\s+Before calling done, check the checklist[^\n]*\n?", "", text)
    return text


def applied_edit_seen(frame: Mapping[str, object]) -> bool:
    return bool(_changed_paths(frame))


def has_current_source(evidence: Mapping[str, object], frame: Mapping[str, object]) -> bool:
    path = _path(evidence.get("path"))
    artifacts = frame.get("artifact_revisions")
    if not path or not isinstance(artifacts, Mapping) or path not in artifacts:
        return False
    revision = evidence.get("file_revision", evidence.get("revision", evidence.get("source_revision")))
    return revision is not None and _same_revision(revision, artifacts[path])


def empty_generated_context(text: object) -> bool:
    value = str(text or "")
    if not value.strip():
        return True
    lowered = value.lower()
    return all(token not in lowered for token in ("generated", "artifact", "requested"))
