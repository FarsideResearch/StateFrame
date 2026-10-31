"""Read-result truth and per-request delivery facts, without retaining history."""
from __future__ import annotations

import copy
import hashlib
import os
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence

ENV = "STATEFRAME_REREAD_RESULT_TRUTH"
EVICTION_NOTICE = "[not shown] {path} @rev {rev}: re-read to see it"
_SEEN_LIMIT = 256
_REPLAY_SCHEMA = "stateframe-code-bookmark-read-replay/1"


def enabled() -> bool:
    value = os.environ.get(ENV, "on").strip().lower()
    if value not in {"on", "off"}:
        raise ValueError(f"{ENV} must be on or off")
    return value == "on"


def eviction_notice(path: str, revision: object) -> str:
    return EVICTION_NOTICE.format(path=path, rev=revision or "unavailable")


def read_is_owed(fact: Mapping[str, object] | object,
                 revision: object = None) -> bool:
    """A source read remains a delivery obligation only until its file is edited."""
    if not isinstance(fact, Mapping):
        return False
    path = fact.get("path")
    if not isinstance(path, str) or not path:
        return False
    read_revision = fact.get("read_revision")
    current_revision = fact.get("current_revision")
    if isinstance(revision, str) and isinstance(read_revision, str) and revision != read_revision:
        return False
    if not isinstance(read_revision, str) or not isinstance(current_revision, str):
        return True  # Unknown revision cannot prove that an edit discharged the debt.
    return read_revision == current_revision


def result_note(measurement: object, identity: str, generation: str | None,
                revision: object, action: object) -> str | None:
    """Remember only the latest read identity per file, never source text."""
    owner = getattr(measurement, "owner", None)
    if owner is not None:
        generation = str(getattr(owner, "workspace_revision", 0))
    if not enabled() or generation is None or not isinstance(revision, str):
        return None
    if type(action) is not int or action < 0:
        return None
    seen = getattr(measurement, "_reread_result_truth", None)
    if not isinstance(seen, OrderedDict):
        seen = OrderedDict()
        try:
            setattr(measurement, "_reread_result_truth", seen)
        except (AttributeError, TypeError):
            return None  # Optional identity tracking must never deny a read.
    previous = seen.get(identity)
    key = (generation, revision)
    failure = getattr(measurement, "_latest_check_receipt", None)
    active = getattr(measurement, "_code_bookmarks_active", False) is True
    assertion = (str(failure.get("assertion_context_preview") or "").strip()
                 if active and isinstance(failure, Mapping)
                 and str(failure.get("status") or "").upper() in {"FAIL", "FAILED", "BEHAVIOR_FAIL"}
                 else "")
    assertion = " ".join(assertion.split())[:200]
    failure_key = (
        str(failure.get("receipt_id") or "")[:160],
        str(failure.get("revision") or "")[:128],
        str(failure.get("status") or "")[:64],
        str(failure.get("outcome") or "")[:64],
        str(failure.get("failure_phase") or "")[:64],
        str(failure.get("runner_tail_digest") or hashlib.sha256(
            str(failure.get("runner_tail") or "").encode("utf-8")).hexdigest()),
        assertion,
    ) if isinstance(failure, Mapping) else None
    if not assertion:
        failure_key = None
    request_id = getattr(measurement, "current_request_id", None)
    request_id = request_id if isinstance(request_id, str) and 0 < len(request_id) <= 120 else None
    if previous is not None and previous[:2] == key:
        seen.move_to_end(identity)
        if failure_key is not None and len(previous) > 3 and previous[3] == failure_key:
            prior = (f"request {previous[4]}" if len(previous) > 4 and previous[4]
                     else f"action {previous[2]}")
            return (f"[read] requested bytes are unchanged since {prior}; "
                    f"current failing assertion: {assertion}. "
                    "Current requested content is in this result; "
                    "the next useful action is an edit or a test run.")
        seen[identity] = (*key, action, failure_key, request_id)
        return (f"[read] unchanged since action {previous[2]} @rev {generation}; "
                "current requested content is in this result.")
    seen[identity] = (*key, action, failure_key, request_id)
    seen.move_to_end(identity)
    while len(seen) > _SEEN_LIMIT:
        seen.popitem(last=False)
    return None


def checkpoint_replay(measurement: object) -> dict[str, object]:
    """Persist only bounded identities needed for the next unchanged read."""
    from kernel_code_bookmarks import MAX_BOOKMARKS

    seen = getattr(measurement, "_reread_result_truth", None)
    rows = [[identity, [entry[0], entry[1], entry[2],
                        list(entry[3]) if isinstance(entry[3], tuple) else None,
                        entry[4]]]
            for identity, entry in seen.items()
            if isinstance(identity, str) and len(identity) <= 2048
            and isinstance(entry, tuple) and len(entry) == 5
            ] if isinstance(seen, OrderedDict) else []
    latest = getattr(measurement, "_latest_check_receipt", None)
    failure = {}
    if isinstance(latest, Mapping):
        for key in ("receipt_id", "revision", "status", "outcome", "failure_phase",
                    "assertion_context_preview"):
            failure[key] = str(latest.get(key) or "")[:512]
        failure["runner_tail_digest"] = str(latest.get("runner_tail_digest") or hashlib.sha256(
            str(latest.get("runner_tail") or "").encode("utf-8")).hexdigest())
    carry = getattr(measurement, "_code_bookmark_carry", ())
    bookmarks = [
        {"path": row["path"], "symbol": row["symbol"],
         "read_only": row.get("read_only") is True}
        for row in carry[:MAX_BOOKMARKS]
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
        and isinstance(row.get("symbol"), str)
    ] if isinstance(carry, list) else []
    return {"schema": _REPLAY_SCHEMA, "active": True,
            "seen": rows[-_SEEN_LIMIT:], "latest_check": failure,
            "bookmarks": bookmarks}


def restore_replay(measurement: object, snapshot: object) -> None:
    """Restore the same read-note decision after a validated checkpoint."""
    from kernel_code_bookmarks import MAX_BOOKMARKS
    from kernel_public_localization import public_path

    if (not isinstance(snapshot, Mapping) or snapshot.get("schema") != _REPLAY_SCHEMA
            or snapshot.get("active") is not True):
        raise ValueError("invalid bookmark read replay")
    rows, latest = snapshot.get("seen"), snapshot.get("latest_check")
    bookmarks = snapshot.get("bookmarks", [])
    if (not isinstance(rows, list) or len(rows) > _SEEN_LIMIT
            or not isinstance(latest, Mapping)
            or not isinstance(bookmarks, list) or len(bookmarks) > MAX_BOOKMARKS):
        raise ValueError("invalid bookmark read replay")
    restored_bookmarks = []
    for bookmark in bookmarks:
        if (not isinstance(bookmark, Mapping)
                or set(bookmark) != {"path", "symbol", "read_only"}
                or not isinstance(bookmark["path"], str)
                or public_path(bookmark["path"]) != bookmark["path"]
                or not bookmark["path"].endswith(".py")
                or not isinstance(bookmark["symbol"], str)
                or not 0 < len(bookmark["symbol"]) <= 256
                or type(bookmark["read_only"]) is not bool):
            raise ValueError("invalid bookmark read replay")
        restored_bookmarks.append(dict(bookmark))
    restored: OrderedDict[str, tuple[object, ...]] = OrderedDict()
    for row in rows:
        if (not isinstance(row, list) or len(row) != 2
                or not isinstance(row[0], str) or len(row[0]) > 2048
                or not isinstance(row[1], list) or len(row[1]) != 5):
            raise ValueError("invalid bookmark read replay")
        if row[0] in restored:
            raise ValueError("invalid bookmark read replay")
        generation, revision, action, failure_key, request_id = row[1]
        if (not isinstance(generation, str) or len(generation) > 512
                or not isinstance(revision, str) or len(revision) > 128
                or type(action) is not int or action < 0
                or (failure_key is not None and (not isinstance(failure_key, list)
                    or len(failure_key) != 7 or any(not isinstance(value, str)
                                                    or len(value) > 200 for value in failure_key)))
                or (request_id is not None and (not isinstance(request_id, str)
                                                or len(request_id) > 120))):
            raise ValueError("invalid bookmark read replay")
        restored[row[0]] = (generation, revision, action,
                            tuple(failure_key) if failure_key is not None else None,
                            request_id)
    allowed = {"receipt_id", "revision", "status", "outcome", "failure_phase",
               "assertion_context_preview", "runner_tail_digest"}
    if (set(latest) - allowed or any(not isinstance(value, str) or len(value) > 512
                                    for value in latest.values())):
        raise ValueError("invalid bookmark read replay")
    digest = latest.get("runner_tail_digest")
    if digest is not None and re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid bookmark read replay")
    measurement._reread_result_truth = restored
    measurement._latest_check_receipt = dict(latest)
    measurement._code_bookmarks_active = True
    measurement._code_bookmark_carry = restored_bookmarks


def _complete_body(path: str, revision: object,
                   segments: Sequence[Mapping[str, object]]) -> bool:
    """Require contiguous, revision-bound bytes hashing to the whole file."""
    if not isinstance(revision, str) or not revision.startswith("sha256:"):
        return False
    covered: dict[int, str] = {}
    for segment in segments:
        if segment.get("path") != path or segment.get("revision") != revision:
            continue
        start, end, text = (segment.get(key) for key in ("start_line", "end_line", "text"))
        if (type(start) is not int or type(end) is not int or start < 1
                or end < start or not isinstance(text, str)):
            continue
        lines = text.splitlines(keepends=True)
        if len(lines) != end - start + 1:
            continue
        for number, line in enumerate(lines, start):
            if number in covered and covered[number] != line:
                return False
            covered[number] = line
    if not covered or min(covered) != 1 or max(covered) != len(covered):
        return False
    raw = "".join(covered[index] for index in range(1, len(covered) + 1)).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest() == revision


def reconcile_frame(frame: dict[str, object],
                    messages: Sequence[Mapping[str, object]]) -> None:
    """Replace delivery claims using only source carriers in this candidate."""
    if not enabled():
        return
    from kernel_stateframe_capacity import _read_source_segments
    from kernel_request_source_window import _read_page_source_range

    facts = frame.get("read_facts", [])
    ledger = frame.get("action_ledger", {})
    reads = ledger.get("reads", []) if isinstance(ledger, Mapping) else []
    # Use the same selected-source authority as receipt validation and repair.
    # Open-site navigation cannot resurrect an evicted read body.
    segments = _read_source_segments(messages, frame, reads)
    revisions = {fact["path"]: fact.get("current_revision") for fact in facts
                 if isinstance(fact, Mapping) and isinstance(fact.get("path"), str)}
    for message in messages:
        parsed = _read_page_source_range(message.get("content"), {}, revisions)
        if parsed is not None and isinstance(parsed[2], str):
            path, (start, end), text = parsed
            segments.append({"path": path, "revision": revisions[path],
                             "start_line": start, "end_line": end, "text": text})
    notices: dict[str, str | None] = {}
    bookmarks = frame.get("code_bookmarks")
    bookmarks = bookmarks if isinstance(bookmarks, list) else []
    from kernel_stateframe_capacity import _delivered
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        path = fact.get("path")
        if not isinstance(path, str):
            continue
        revision = fact.get("current_revision")
        if not read_is_owed(fact):
            fact["body_delivery"] = "not_owed"
            fact.pop("delivery_notice", None)
            fact.pop("visible_symbols", None)
            notices[path] = None
            continue
        present = _complete_body(path, revision, segments)
        visible = []
        if not present:
            for bookmark in bookmarks:
                if (not isinstance(bookmark, Mapping) or bookmark.get("path") != path
                        or bookmark.get("revision") != revision):
                    continue
                for prefix in ("shown", "compact"):
                    start, end = (bookmark.get(prefix + suffix)
                                  for suffix in ("_start_line", "_end_line"))
                    if type(start) is not int or type(end) is not int:
                        continue
                    row = {"path": path, "revision": revision,
                           "lines": {"start": start, "end": end},
                           "content_sha256": bookmark.get(prefix + "_sha256")}
                    if _delivered(row, segments):
                        visible.append(str(bookmark.get("symbol")))
                        break
        fact["body_delivery"] = ("present" if present else
                                 "bookmarked" if visible else "not_shown")
        if present:
            fact.pop("delivery_notice", None)
            fact.pop("visible_symbols", None)
            notices[path] = "fresh"
        elif visible:
            fact["visible_symbols"] = sorted(set(visible))
            notice = "[bookmark shown] " + path + "::" + ", ".join(fact["visible_symbols"])
            fact["delivery_notice"] = notice
            notices[path] = notice
        else:
            fact.pop("visible_symbols", None)
            notice = eviction_notice(path, revision)
            fact["delivery_notice"] = notice
            notices[path] = notice

    for row in frame.get("prior_user_context", []):
        content = row.get("content")
        if (not isinstance(content, str) or not content.startswith("[kernel] (")
                or "earlier exchanges trimmed to fit context. CAPSULE" not in content.splitlines()[0]
                or "\nWORLD KNOWLEDGE " not in content):
            continue
        lines = content.splitlines(keepends=True)
        in_knowledge = False
        for index, line in enumerate(lines):
            if line.startswith("WORLD KNOWLEDGE "):
                in_knowledge = True
                continue
            if in_knowledge and line.startswith("+"):
                in_knowledge = False
            if not in_knowledge or not line.startswith("- "):
                continue
            for path, notice in sorted(notices.items(), key=lambda item: -len(item[0])):
                if not line.startswith((f"- {path} ", f"- {path}::")):
                    continue
                label = r" \| (?:fresh|\[not shown\].*?: re-read to see it|\[bookmark shown\][^|]*) \| (read@\d+)"
                if notice is None:
                    if re.search(label, line):
                        lines[index] = ""
                else:
                    lines[index] = re.sub(
                        label, lambda match: f" | {notice} | {match.group(1)}", line,
                    )
                break
        updated = "".join(lines)
        if updated != content:
            row["content"] = updated
            row["content_sha256"] = hashlib.sha256(updated.encode("utf-8")).hexdigest()


def reconcile_source_truth(messages: list[dict[str, object]],
                           frame: dict[str, object], *,
                           required_deferred_paths: Sequence[object] = (),
                           include_host_fact: bool = True) -> None:
    """Run read-delivery and open-source notice reconciliation together."""
    reconcile_frame(frame, messages)
    if include_host_fact:
        from kernel_stateframe_framefit5 import _reconcile_framefit5_source_notices

        _reconcile_framefit5_source_notices(
            messages, frame, required_deferred_paths=required_deferred_paths,
        )


def refresh_messages(
        messages: list[dict[str, object]], *, required_deferred_paths: Sequence[object] = (),
) -> None:
    if not enabled():
        return
    from kernel_stateframe_framefit5 import _serialize_state_message, _state_payload

    required_paths = (
        tuple(path for path in required_deferred_paths if isinstance(path, str) and path)
        if not isinstance(required_deferred_paths, (str, bytes, bytearray)) else ()
    )
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith("[StateFrame]\n"):
            continue
        payload, state = _state_payload(message)
        frame = state.get("framefit5")
        if isinstance(frame, dict):
            before = copy.deepcopy(frame)
            include_host_fact = any(
                isinstance(row, Mapping)
                and str(row.get("content", "")).startswith("[Host fact]")
                for row in messages
            )
            reconcile_source_truth(
                messages, frame, required_deferred_paths=required_paths,
                include_host_fact=include_host_fact or bool(required_paths),
            )
            if frame != before:
                message["content"] = _serialize_state_message(payload)


def validate_delivery(frame: Mapping[str, object],
                      messages: Sequence[Mapping[str, object]]) -> None:
    """Reject a stale delivery claim if a later transform removed its carrier."""
    if not enabled() or not any("body_delivery" in row for row in frame.get("read_facts", [])):
        return
    expected = copy.deepcopy(dict(frame))
    reconcile_frame(expected, messages)
    if (expected.get("read_facts") != frame.get("read_facts")
            or expected.get("prior_user_context") != frame.get("prior_user_context")):
        from kernel_stateframe_framefit5 import FrameFit5Error
        raise FrameFit5Error("read delivery facts do not match this request's source carriers")
