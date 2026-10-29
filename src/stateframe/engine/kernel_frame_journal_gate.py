"""Journal-backed corrections for the StateFrame facts on a provider request."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass


MODE_ENV = "STATEFRAME_FRAME_JOURNAL_GATE"
_OFF_VALUES = frozenset({"", "0", "false", "no", "off"})
_ON_VALUES = frozenset({"1", "true", "yes", "on"})
MAX_EDIT_ROWS = 128
MAX_PANEL_CHARS = 512_000
_HEX_REVISION = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_SAFE_SELECTOR = re.compile(r"^[A-Za-z0-9_./:@+-]{1,256}$")
_JOURNAL_EVENT_KINDS = frozenset({
    "action_receipt", "check_receipt", "completion_authority_state",
})


@dataclass(frozen=True)
class CheckFact:
    status: str
    revision: str | None
    selector: str | None
    sequence: int


@dataclass(frozen=True)
class JournalFrameFacts:
    applied_edit_count: int
    applied_edits: tuple[dict[str, object], ...]
    last_check: CheckFact | None
    current_revision: str | None


class JournalFactIndex:
    """Compact projection of persisted journal records; no runtime state is read."""

    def __init__(self) -> None:
        self._applied_edit_count = 0
        self._applied_edits: deque[dict[str, object]] = deque(maxlen=MAX_EDIT_ROWS)
        self._last_edit_sequence = -1
        self._last_check: CheckFact | None = None
        self._current_revision: str | None = None
        self._current_revision_sequence = -1

    def observe(self, event: Mapping[str, object]) -> None:
        sequence = _event_sequence(event)
        kind = event.get("kind")

        if kind == "action_receipt":
            receipt = event.get("receipt")
            if _is_applied_edit(receipt):
                self._applied_edit_count += 1
                self._last_edit_sequence = max(self._last_edit_sequence, sequence)
                if sequence >= self._current_revision_sequence:
                    self._current_revision = None
                    self._current_revision_sequence = sequence
                self._applied_edits.append(_edit_row(receipt, sequence))

        explicit_revision = _explicit_current_revision(event)
        if explicit_revision is not None and sequence >= self._current_revision_sequence:
            self._current_revision = explicit_revision
            self._current_revision_sequence = sequence

        if kind == "check_receipt":
            check = _check_fact(event, sequence)
            if check is not None and sequence >= (self._last_check.sequence if self._last_check else -1):
                self._last_check = check
                if check.revision is not None and sequence >= self._last_edit_sequence:
                    self._current_revision = check.revision
                    self._current_revision_sequence = sequence

    def snapshot(self) -> JournalFrameFacts:
        return JournalFrameFacts(
            applied_edit_count=self._applied_edit_count,
            applied_edits=tuple(dict(row) for row in self._applied_edits),
            last_check=self._last_check,
            current_revision=self._current_revision,
        )


def enabled(environment: Mapping[str, str] | None = None) -> bool:
    """Default-off switch parser; unknown values fail open to the legacy path."""

    values = os.environ if environment is None else environment
    raw = str(values.get(MODE_ENV, "off")).strip().lower()
    if raw in _ON_VALUES:
        return True
    if raw in _OFF_VALUES:
        return False
    return False


def observe_journal_event(journal: object, event: Mapping[str, object]) -> None:
    """Index only receipt records after their journal append has completed."""

    if event.get("kind") not in _JOURNAL_EVENT_KINDS:
        return
    index = getattr(journal, "_frame_journal_index", None)
    if not isinstance(index, JournalFactIndex):
        index = JournalFactIndex()
        setattr(journal, "_frame_journal_index", index)
    index.observe(event)


def snapshot_for_journal(journal: object) -> JournalFrameFacts:
    index = getattr(journal, "_frame_journal_index", None)
    if isinstance(index, JournalFactIndex):
        return index.snapshot()
    return JournalFactIndex().snapshot()


def facts_from_events(events: Iterable[Mapping[str, object]]) -> JournalFrameFacts:
    """Reduce persisted event rows, used by the pinned journal regression guard."""

    index = JournalFactIndex()
    for event in events:
        if isinstance(event, Mapping):
            index.observe(event)
    return index.snapshot()


def reconcile_request_body(
    journal: object,
    body: object,
    *,
    request_id: object = None,
    step: object = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Correct only journal-backed StateFrame panels; failures never escape to send."""

    if not enabled(environment) or not isinstance(body, MutableMapping):
        return
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list):
        return
    if not any(_panel_kind(message) == "stateframe" for message in raw_messages):
        return

    try:
        facts = _facts_for(journal)
        if facts is None:
            return
    except Exception:
        return

    emit = getattr(journal, "emit", None)
    mismatch_count = 0
    corrected_count = 0

    def mismatch(field: str, frame_value: object, journal_value: object) -> None:
        nonlocal mismatch_count
        mismatch_count += 1
        if callable(emit):
            try:
                emit(
                    "frame_journal_mismatch",
                    producer="kernel_frame_journal_gate",
                    request_id=_safe_short_text(request_id),
                    step=step if type(step) is int else None,
                    field=field,
                    frame=_safe_event_value(frame_value),
                    journal=_safe_event_value(journal_value),
                    dispatch=False,
                )
            except Exception:
                pass

    kept: list[object] = []
    panel_count = 0
    for message in raw_messages:
        kind = _panel_kind(message)
        if kind is None:
            kept.append(message)
            continue
        panel_count += 1
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(message, MutableMapping) or not isinstance(content, str):
            _emit_panel_dropped(emit, kind, request_id, step)
            continue
        if len(content) > MAX_PANEL_CHARS:
            _emit_panel_dropped(emit, kind, request_id, step)
            continue

        try:
            if kind == "stateframe":
                replacement, changed = _reconcile_stateframe(content, facts, mismatch)
            elif kind == "delivery_manifest":
                replacement, changed = _reconcile_delivery_manifest(content, facts, mismatch)
            else:
                replacement, changed = _reconcile_certificate(content, facts, mismatch)
        except Exception:
            _emit_panel_dropped(emit, kind, request_id, step)
            continue

        if replacement is None:
            _emit_panel_dropped(emit, kind, request_id, step)
            continue
        if changed:
            message["content"] = replacement
            corrected_count += 1
        kept.append(message)

    raw_messages[:] = kept
    if callable(emit):
        try:
            emit(
                "frame_journal_gate_checked",
                producer="kernel_frame_journal_gate",
                request_id=_safe_short_text(request_id),
                step=step if type(step) is int else None,
                panel_count=panel_count,
                mismatch_count=mismatch_count,
                corrected_panel_count=corrected_count,
                dispatch=False,
            )
        except Exception:
            pass


def _facts_for(journal: object) -> JournalFrameFacts | None:
    getter = getattr(journal, "frame_journal_snapshot", None)
    snapshot: object
    if callable(getter):
        snapshot = getter()
    else:
        index = getattr(journal, "_frame_journal_index", None)
        snapshot = index.snapshot() if isinstance(index, JournalFactIndex) else None
    if isinstance(snapshot, JournalFrameFacts):
        return snapshot
    if not isinstance(snapshot, Mapping):
        return None

    count = snapshot.get("applied_edit_count")
    edits = snapshot.get("applied_edits")
    check = snapshot.get("last_check")
    current_revision = _safe_revision(snapshot.get("current_revision"))
    if type(count) is not int or count < 0 or not isinstance(edits, (tuple, list)):
        return None
    safe_edits = tuple(
        dict(row) for row in edits[:MAX_EDIT_ROWS] if isinstance(row, Mapping)
    )
    safe_check = None
    if isinstance(check, CheckFact):
        safe_check = check
    elif isinstance(check, Mapping):
        sequence = check.get("sequence")
        status = _safe_status(check.get("status"))
        revision = _safe_revision(check.get("revision"))
        selector = _safe_selector(check.get("selector"))
        if type(sequence) is int and status is not None:
            safe_check = CheckFact(status, revision, selector, sequence)
    return JournalFrameFacts(count, safe_edits, safe_check, current_revision)


def _is_applied_edit(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    return value.get("kind") == "edit" and str(value.get("result_class", "")).upper() == "APPLIED"


def _edit_row(receipt: Mapping[str, object], sequence: int) -> dict[str, object]:
    row: dict[str, object] = {
        "status": "APPLIED",
        "receipt_id": _safe_short_text(receipt.get("receipt_id")) or f"sequence:{sequence}",
    }
    path = _safe_product_path(receipt.get("target_path"))
    if path is not None:
        row["path"] = path
    revision = receipt.get("workspace_revision")
    if type(revision) is int and revision >= 0:
        row["workspace_revision"] = revision
    return row


def _check_fact(event: Mapping[str, object], sequence: int) -> CheckFact | None:
    status = _safe_status(event.get("status"))
    if status is None:
        return None
    revision = _safe_revision(event.get("revision"))
    selector = _safe_selector(event.get("selector"))
    return CheckFact(status, revision, selector, sequence)


def _explicit_current_revision(event: Mapping[str, object]) -> str | None:
    candidates: list[object] = [event.get("current_revision")]
    state = event.get("state")
    if isinstance(state, Mapping):
        candidates.append(state.get("current_revision"))
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        candidates.append(payload.get("current_revision"))
        nested_state = payload.get("state")
        if isinstance(nested_state, Mapping):
            candidates.append(nested_state.get("current_revision"))
    for value in candidates:
        revision = _safe_revision(value)
        if revision is not None:
            return revision
    return None


def _event_sequence(event: Mapping[str, object]) -> int:
    value = event.get("sequence")
    return value if type(value) is int and value >= 0 else 0


def _safe_status(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    status = value.strip().upper()
    if status in {"PASS", "FAIL", "ERROR", "UNKNOWN", "NOT_RUN", "BLOCKED", "TIMEOUT", "SKIPPED"}:
        return status
    return None


def _safe_revision(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    revision = value.strip()
    if not revision or len(revision) > 72 or not _HEX_REVISION.fullmatch(revision):
        return None
    if not revision.startswith("sha256:"):
        revision = "sha256:" + revision.lower()
    else:
        revision = "sha256:" + revision[7:].lower()
    return revision


def _short_revision(value: str) -> str:
    return value.removeprefix("sha256:")[:12]


def _visible_revision(value: str | None, compact: bool) -> str | None:
    if value is None:
        return None
    return _short_revision(value) if compact else value


def _revision_matches(rendered: object, expected: str | None) -> bool:
    if expected is None:
        return rendered in (None, "")
    if not isinstance(rendered, str):
        return False
    return rendered.strip() in {
        expected,
        _short_revision(expected),
        f"sha256:{_short_revision(expected)}",
    }


def _safe_rendered_revision(value: object) -> str | None:
    revision = _safe_revision(value)
    if revision is not None:
        return revision
    if not isinstance(value, str):
        return None
    rendered = value.strip()
    if re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{12}", rendered):
        return rendered.lower()
    return None


def _panel_uses_short_revisions(content: str) -> bool:
    return re.search(r"sha256:[0-9a-fA-F]{64}", content) is None


def _safe_selector(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    selector = value.strip()
    if _SAFE_SELECTOR.fullmatch(selector) and ".." not in selector and not selector.startswith("/"):
        return selector
    return None


def _safe_product_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    path = value.strip()
    if not path or len(path) > 256 or path.startswith("/") or "\\" in path:
        return None
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.@+-]+(?:/[A-Za-z0-9_.@+-]+)*", path):
        return None
    return path


def _safe_short_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 128 or any(ord(char) < 32 for char in text):
        return None
    return text


def _safe_event_value(value: object) -> object:
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, str):
        return value[:160] if _safe_short_text(value) is not None else "<omitted>"
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in (
            "count", "receipt_count", "receipt_rows_match", "status",
            "tested_revision", "current_revision", "revision",
        ):
            if key in value:
                safe = _safe_event_value(value[key])
                result[key] = safe
        return result
    return "<omitted>"


def _panel_kind(message: object) -> str | None:
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if not isinstance(content, str):
        return None
    if content.startswith("[StateFrame]\n"):
        return "stateframe"
    if content.startswith("[DELIVERY_MANIFEST "):
        return "delivery_manifest"
    if content.startswith("[VERIFICATION CERTIFICATE]"):
        return "verification_certificate"
    return None


def _decode_json_panel(content: str, *, marker: str) -> tuple[str, dict[str, object], str] | None:
    if not content.startswith(marker):
        return None
    start = content.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(content[start:])
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    return content[:start], value, content[start + end:]


def _reconcile_stateframe(
    content: str,
    facts: JournalFrameFacts,
    mismatch: object,
) -> tuple[str | None, bool]:
    decoded = _decode_json_panel(content, marker="[StateFrame]\n")
    if decoded is None:
        return None, False
    prefix, frame, suffix = decoded
    changed = _reconcile_stateframe_revision(frame, facts, mismatch)
    changed = _reconcile_public_checks(frame, facts, mismatch) or changed
    if not changed:
        return content, False
    return _encode_panel(prefix, frame, suffix), True


def _reconcile_stateframe_revision(
    frame: MutableMapping[str, object],
    facts: JournalFrameFacts,
    mismatch: object,
) -> bool:
    current = facts.current_revision
    if current is None:
        return False
    changed = False
    for key, expected in (
        ("revision", current),
        ("rev", _short_revision(current)),
        ("current_revision", current),
    ):
        if key in frame and frame.get(key) != expected:
            mismatch("current_revision", frame.get(key), current)
            frame[key] = expected
            changed = True
    return changed


def _reconcile_public_checks(
    frame: MutableMapping[str, object],
    facts: JournalFrameFacts,
    mismatch: object,
) -> bool:
    state = frame.get("state")
    if state is None:
        if facts.last_check is None:
            return False
        state = {}
        frame["state"] = state
    elif not isinstance(state, MutableMapping):
        raise ValueError("StateFrame state is not an object")

    public_checks = state.get("public_checks")
    if public_checks is None:
        public_checks = {}
        state["public_checks"] = public_checks
    elif not isinstance(public_checks, MutableMapping):
        raise ValueError("StateFrame public_checks is not an object")

    expected_status = facts.last_check.status if facts.last_check is not None else "NOT_RUN"
    expected_tested = facts.last_check.revision if facts.last_check is not None else None
    compact = "rev" in frame and "revision" not in frame
    visible_tested = _visible_revision(expected_tested, compact)
    visible_current = _visible_revision(facts.current_revision, compact)
    if _public_checks_match(public_checks, expected_status, visible_tested, visible_current):
        return False

    mismatch(
        "last_check",
        _public_checks_value(public_checks),
        _check_journal_value(facts.last_check, facts.current_revision),
    )
    public_checks.update({
        "state": expected_status,
        "selectors": (
            [facts.last_check.selector]
            if facts.last_check is not None and facts.last_check.selector else []
        ),
        "tested_revision": visible_tested,
        "evidence": (
            [{"source": "run journal", "state": expected_status, "revision": visible_tested}]
            if visible_tested is not None else []
        ),
    })
    if visible_current is None:
        public_checks.pop("current_revision", None)
        public_checks.pop("stale", None)
    else:
        public_checks["current_revision"] = visible_current
        public_checks["stale"] = (
            expected_tested != facts.current_revision
            if expected_tested is not None and facts.current_revision is not None else None
        )
    return True


def _public_checks_match(
    panel: Mapping[str, object],
    status: str,
    tested_revision: str | None,
    current_revision: str | None,
) -> bool:
    if panel.get("state") != status or panel.get("tested_revision") != tested_revision:
        return False
    if current_revision is not None and panel.get("current_revision") != current_revision:
        return False
    if current_revision is None and "current_revision" in panel:
        return False
    stale = (tested_revision != current_revision) if tested_revision is not None and current_revision is not None else None
    return panel.get("stale") == stale if "stale" in panel else True


def _public_checks_value(panel: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": _safe_status(panel.get("state")) or "<unknown>",
        "tested_revision": _safe_rendered_revision(panel.get("tested_revision")),
        "current_revision": _safe_rendered_revision(panel.get("current_revision")),
    }


def _check_journal_value(check: CheckFact | None, current: str | None) -> dict[str, object]:
    return {
        "status": check.status if check is not None else "NOT_RUN",
        "tested_revision": check.revision if check is not None else None,
        "current_revision": current,
    }


def _reconcile_delivery_manifest(
    content: str,
    facts: JournalFrameFacts,
    mismatch: object,
) -> tuple[str | None, bool]:
    decoded = _decode_json_panel(content, marker="[DELIVERY_MANIFEST")
    if decoded is None:
        return None, False
    prefix, manifest, suffix = decoded
    compact = _panel_uses_short_revisions(content)
    changed = _reconcile_manifest_edit_count(manifest, facts, mismatch)
    changed = _reconcile_manifest_check(manifest, facts, compact, mismatch) or changed
    changed = _reconcile_manifest_current_revision(
        manifest, facts, compact, mismatch,
    ) or changed
    if not changed:
        return content, False
    return _encode_panel(prefix, manifest, suffix), True


def _reconcile_manifest_edit_count(
    manifest: MutableMapping[str, object],
    facts: JournalFrameFacts,
    mismatch: object,
) -> bool:
    if facts.applied_edit_count > len(facts.applied_edits):
        raise ValueError("journal edit rows exceed the bounded render window")

    expected_edits = [dict(row) for row in facts.applied_edits]
    rendered_edits = manifest.get("applied_edits")
    rendered_count = manifest.get("applied_edit_count")
    edits_match = isinstance(rendered_edits, list) and rendered_edits == expected_edits
    count_matches = (
        type(rendered_count) is int
        and rendered_count == facts.applied_edit_count
    )
    if edits_match and count_matches:
        return False

    frame_value = {
        "count": rendered_count,
        "receipt_count": (
            len(rendered_edits) if isinstance(rendered_edits, list) else None
        ),
        "receipt_rows_match": edits_match,
    }
    mismatch(
        "applied_edits",
        frame_value,
        {
            "count": facts.applied_edit_count,
            "receipt_rows_match": True,
        },
    )
    manifest["applied_edits"] = expected_edits
    manifest["applied_edit_count"] = facts.applied_edit_count
    return True


def _reconcile_manifest_check(
    manifest: MutableMapping[str, object],
    facts: JournalFrameFacts,
    compact: bool,
    mismatch: object,
) -> bool:
    rendered = manifest.get("latest_check")
    status = None
    revision: object = manifest.get("latest_check_revision")
    if isinstance(rendered, Mapping):
        status = _safe_status(rendered.get("status"))
        nested_revision = rendered.get("tested_revision", rendered.get("revision"))
        if nested_revision not in (None, ""):
            revision = nested_revision
    elif isinstance(rendered, str):
        status = _safe_status(rendered)

    expected_status = facts.last_check.status if facts.last_check is not None else None
    expected_revision = facts.last_check.revision if facts.last_check is not None else None
    if status == expected_status and _revision_matches(revision, expected_revision):
        return False

    mismatch(
        "last_check",
        {"status": status, "tested_revision": revision},
        _check_journal_value(facts.last_check, facts.current_revision),
    )
    visible_revision = _visible_revision(expected_revision, compact)
    manifest["latest_check"] = (
        {
            "status": facts.last_check.status,
            "revision": visible_revision,
            "selector": facts.last_check.selector,
        } if facts.last_check is not None else None
    )
    manifest["latest_check_revision"] = visible_revision
    return True


def _reconcile_manifest_current_revision(
    manifest: MutableMapping[str, object],
    facts: JournalFrameFacts,
    compact: bool,
    mismatch: object,
) -> bool:
    if "current_revision" not in manifest:
        return False
    rendered = manifest.get("current_revision")
    if _revision_matches(rendered, facts.current_revision):
        return False
    mismatch("current_revision", rendered, facts.current_revision)
    if facts.current_revision is None:
        manifest.pop("current_revision", None)
    else:
        manifest["current_revision"] = _visible_revision(facts.current_revision, compact)
    return True


def _reconcile_certificate(
    content: str,
    facts: JournalFrameFacts,
    mismatch: object,
) -> tuple[str | None, bool]:
    if _certificate_matches(content, facts):
        return content, False

    mismatch(
        "last_check",
        _certificate_frame_value(content),
        _check_journal_value(facts.last_check, facts.current_revision),
    )
    suffix_index = content.find(" Observation only;")
    suffix = content[suffix_index:] if suffix_index >= 0 else ""
    if facts.last_check is None:
        status_line = "LAST CHECK: no check receipt is present in the run journal."
    else:
        check = facts.last_check
        selector = f" on {check.selector}" if check.selector else ""
        compact_revisions = _panel_uses_short_revisions(content)
        tested = _visible_revision(check.revision, compact_revisions) or "revision unavailable"
        if facts.current_revision is None:
            freshness = "current revision unavailable in journal; freshness unknown"
        else:
            current = _visible_revision(facts.current_revision, compact_revisions)
            freshness = (
                f"current revision {current}; current"
                if check.revision == facts.current_revision
                else f"current revision {current}; stale"
            )
        status_line = (
            f"LAST CHECK: {check.status}{selector}; tested revision {tested}; {freshness}."
        )
    return "[VERIFICATION CERTIFICATE] " + status_line + suffix, True


def _certificate_matches(content: str, facts: JournalFrameFacts) -> bool:
    check = facts.last_check
    if check is None:
        return (
            "no check receipt is present in the run journal" in content.lower()
            or "GATE_NEVER_RAN" in content
        )
    status_match = re.search(r"\bLAST CHECK:\s*([A-Z_]+)", content)
    if status_match is None or status_match.group(1) != check.status:
        return False
    if check.revision is not None and not _text_contains_revision(content, check.revision):
        return False
    if facts.current_revision is not None and not _text_contains_revision(
        content, facts.current_revision,
    ):
        return False
    return "GATE_NEVER_RAN" not in content


def _text_contains_revision(content: str, revision: str) -> bool:
    return any(
        value in content
        for value in (
            revision,
            _short_revision(revision),
            f"sha256:{_short_revision(revision)}",
        )
    )


def _certificate_frame_value(content: str) -> dict[str, object]:
    upper = content.upper()
    status = next(
        (item for item in (
            "PASS", "FAIL", "ERROR", "UNKNOWN", "GATE_NEVER_RAN",
        ) if item in upper),
        None,
    )
    revision_match = re.search(
        r"sha256:[0-9a-fA-F]{64}|(?<![0-9a-fA-F])[0-9a-fA-F]{12}(?![0-9a-fA-F])",
        content,
    )
    return {
        "status": status or "<unknown>",
        "tested_revision": revision_match.group(0) if revision_match else None,
    }


def _encode_panel(prefix: str, value: Mapping[str, object], suffix: str) -> str:
    return prefix + json.dumps(value, ensure_ascii=False, separators=(",", ":")) + suffix


def _emit_panel_dropped(
    emit: object,
    panel: str,
    request_id: object,
    step: object,
) -> None:
    if not callable(emit):
        return
    try:
        emit(
            "frame_journal_panel_dropped",
            producer="kernel_frame_journal_gate",
            request_id=_safe_short_text(request_id),
            step=step if type(step) is int else None,
            panel=panel,
            reason="journal_reconciliation_failed",
            dispatch=False,
        )
    except Exception:
        pass


__all__ = [
    "CheckFact",
    "JournalFactIndex",
    "JournalFrameFacts",
    "MODE_ENV",
    "enabled",
    "facts_from_events",
    "observe_journal_event",
    "reconcile_request_body",
    "snapshot_for_journal",
]
