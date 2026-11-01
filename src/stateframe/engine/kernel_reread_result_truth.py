"""Read-result truth and per-request delivery facts, without retaining history."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence

ENV = "STATEFRAME_REREAD_RESULT_TRUTH"
EVICTION_NOTICE = "[not shown] {path} @rev {rev}: re-read to see it"
_SEEN_LIMIT = 256
_REPLAY_SCHEMA = "stateframe-code-bookmark-read-replay/2"


def _current_product_revision(measurement: object) -> str | None:
    binding = getattr(measurement, "binding", None)
    if callable(binding):
        try:
            value = binding()
        except Exception:
            # A failed optional freshness lookup must not deny the exact read.
            return None
        if isinstance(value, Mapping) and isinstance(value.get("product_revision"), str):
            return value["product_revision"]
    value = getattr(measurement, "current_product_revision", None)
    return value if isinstance(value, str) else None


def _canonical_failure(receipt: object) -> dict[str, str] | None:
    if not isinstance(receipt, Mapping) or receipt.get("status") != "FAIL":
        return None
    receipt_id, revision = receipt.get("receipt_id"), receipt.get("revision")
    if (not isinstance(receipt_id, str) or not 0 < len(receipt_id) <= 160
            or not isinstance(revision, str) or not 0 < len(revision) <= 128):
        return None
    fields = {key: str(receipt.get(key) or "") for key in ("outcome", "failure_phase")}
    if any(len(value) > 64 for value in fields.values()):
        return None
    if (fields["outcome"].upper() in {"INVALID_CHECK", "ENVIRONMENT", "NOT_RUN", "UNKNOWN"}
            or fields["failure_phase"].upper() in {
                "COLLECTION", "IMPORT", "SETUP", "HOST", "EXECUTION", "TIMEOUT"}):
        return None
    assertion = " ".join(str(receipt.get("assertion_context_preview") or "").split())[:200].rstrip()
    if not assertion:
        return None
    # The runner tail is private on the live path and absent from the compact
    # journal projection. Bind both paths to the same public assertion instead.
    digest = hashlib.sha256(assertion.encode("utf-8")).hexdigest()
    return {"receipt_id": receipt_id, "revision": revision, "status": "FAIL",
            **fields, "runner_tail_digest": digest, "assertion": assertion}


def _current_failure(measurement: object) -> dict[str, str] | None:
    revision = _current_product_revision(measurement)
    context = getattr(measurement, "_framefit_action_context", None)
    frame = context.get("framefit5") if isinstance(context, Mapping) else None
    verdict = frame.get("last_verdict") if isinstance(frame, Mapping) else None
    if revision is None and isinstance(verdict, Mapping):
        candidate = verdict.get("current_revision")
        revision = candidate if isinstance(candidate, str) else None
    if revision is None:
        return None
    latest = getattr(measurement, "_latest_check_receipt", None)
    failure = _canonical_failure(latest)
    if failure is not None:
        receipts = getattr(measurement, "host_check_receipts", ())
        authenticated = isinstance(receipts, list) and any(
            _canonical_failure(row) == failure
            for row in receipts)
        if authenticated and failure["revision"] == revision:
            return failure
    if (isinstance(latest, Mapping) and latest.get("revision") == revision
            and latest.get("status") != "FAIL"):
        return None
    if (isinstance(verdict, Mapping) and verdict.get("status") == "FAIL"
            and verdict.get("tested_revision") == revision
            and verdict.get("current_revision") == revision):
        # The compact journal projection is durable; the runtime's public
        # receipt list can be absent after resume. A later edit invalidates
        # this frame until the next capture rebuilds its verdict.
        edits = frame.get("applied_edits", ())
        frame_edits = [row for row in edits if isinstance(row, Mapping)] if isinstance(
            edits, (list, tuple)) else []
        latest_edit_turn = max((row["turn"] for row in frame_edits
                                if type(row.get("turn")) is int), default=-1)
        latest_edit_ids = {row.get("edit_receipt") for row in frame_edits
                           if row.get("turn") == latest_edit_turn}
        owner_receipts = getattr(getattr(measurement, "owner", None), "_receipts", ())
        if not any(isinstance(row, Mapping) and row.get("kind") == "edit"
                   and row.get("result_class") == "APPLIED"
                   and (type(row.get("turn")) is not int
                        or row["turn"] > latest_edit_turn
                        or (row["turn"] == latest_edit_turn
                            and row.get("receipt_id") not in latest_edit_ids))
                   for row in owner_receipts):
            records = context.get("journal_receipt_records", ())
            for row in reversed(records):
                if (isinstance(row, Mapping)
                        and row.get("receipt_id") == verdict.get("check_receipt_id")
                        and row.get("revision") == revision):
                    failure = _canonical_failure(row)
                    if failure is not None:
                        return failure
    replayed = getattr(measurement, "_bookmark_failure_data", None)
    if isinstance(replayed, Mapping) and replayed.get("revision") == revision:
        return _canonical_failure({**replayed, "assertion_context_preview": replayed.get("assertion")})
    return None


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
                revision: object, action: object, *,
                requested_range: object = None, effective_range: object = None) -> str | None:
    """Remember bounded page identities, never source text."""
    owner = getattr(measurement, "owner", None)
    if owner is not None:
        generation = str(getattr(owner, "workspace_revision", 0))
    if not enabled() or generation is None or not isinstance(revision, str):
        return None
    if type(action) is not int or action < 0:
        return None
    from kernel_code_bookmarks import enabled as bookmarks_enabled
    active = bookmarks_enabled()
    if active and (not isinstance(generation, str) or not isinstance(identity, str)
                   or len(identity) > 2048 or len(generation) > 512
                   or len(revision) > 128):
        return None
    if active:
        # A read of a new interval is new evidence even when the file revision
        # matches a previous request. Hash coordinates to bound checkpoint size.
        coordinates = json.dumps([requested_range, effective_range], sort_keys=True,
                                 separators=(",", ":"), default=str)
        identity = hashlib.sha256((identity + "\0" + coordinates).encode()).hexdigest()
    seen = getattr(measurement, "_reread_result_truth", None)
    if not isinstance(seen, OrderedDict):
        seen = OrderedDict()
        try:
            setattr(measurement, "_reread_result_truth", seen)
        except (AttributeError, TypeError):
            return None  # Optional identity tracking must never deny a read.
    previous = seen.get(identity)
    key = (generation, revision)
    failure = _current_failure(measurement) if active else None
    failure_key = tuple(failure.values()) if failure is not None else None
    request_number = getattr(measurement, "_frame_budget_request_count", None)
    has_provider_number = type(request_number) is int and request_number > 0
    request_id = (str(request_number) if has_provider_number
                  else getattr(measurement, "current_request_id", None))
    request_id = request_id if isinstance(request_id, str) and 0 < len(request_id) <= 120 else None
    if previous is not None and previous[:2] == key:
        seen.move_to_end(identity)
        prior_request = previous[4] if len(previous) > 4 else None
        if active and (failure_key is not None or len(previous) < 4
                       or previous[3] is None):
            seen[identity] = (*key, previous[2], failure_key, request_id)
        if active and prior_request and (prior_request != request_id
                                         or not has_provider_number):
            assertion = (f" current failing assertion: {failure['assertion']}."
                         if failure_key is not None and len(previous) > 3
                         and previous[3] == failure_key else "")
            return (f"[read] requested bytes are unchanged since request {prior_request};"
                    f"{assertion} Current requested content is in this result.")
        return (f"[read] unchanged since action {previous[2]} @rev {generation}; "
                "current requested content is in this result.")
    seen[identity] = (*key, action, failure_key, request_id) if active else (*key, action)
    seen.move_to_end(identity)
    while len(seen) > _SEEN_LIMIT:
        seen.popitem(last=False)
    return None


def checkpoint_replay(measurement: object) -> dict[str, object]:
    """Persist only bounded identities needed for the next unchanged read."""
    from kernel_code_bookmarks import MAX_BOOKMARKS
    from kernel_public_localization import public_path

    seen = getattr(measurement, "_reread_result_truth", None)
    rows = [[identity, [entry[0], entry[1], entry[2],
                        list(entry[3]) if len(entry) == 5 and isinstance(entry[3], tuple) else None,
                        entry[4] if len(entry) == 5 else None]]
            for identity, entry in seen.items()
            if isinstance(identity, str) and len(identity) <= 2048
            and isinstance(entry, tuple) and len(entry) in {3, 5}
            and isinstance(entry[0], str) and len(entry[0]) <= 512
            and isinstance(entry[1], str) and len(entry[1]) <= 128
            and type(entry[2]) is int and entry[2] >= 0
            ] if isinstance(seen, OrderedDict) else []
    failure = _current_failure(measurement)
    carry = getattr(measurement, "_code_bookmark_carry", ())
    bookmarks = [
        {"path": row["path"], "symbol": row["symbol"],
         "read_only": row.get("read_only") is True}
        for row in carry[:MAX_BOOKMARKS]
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
        and isinstance(row.get("symbol"), str) and len(row["symbol"]) <= 256
        and public_path(row["path"]) == row["path"] and row["path"].endswith(".py")
    ] if isinstance(carry, list) else []
    snapshot = {"schema": _REPLAY_SCHEMA,
                "active": getattr(measurement, "_code_bookmarks_active", False) is True,
                "seen": rows[-_SEEN_LIMIT:], "failure": failure,
                "bookmarks": bookmarks}
    from kernel_read_lockout import checkpoint as lockout_checkpoint
    lockout = lockout_checkpoint(measurement)
    if lockout is not None:
        snapshot["read_lockout"] = lockout
    restore_replay(_ReplayValidationTarget(), snapshot)
    return snapshot


class _ReplayValidationTarget:
    pass


def restore_replay(measurement: object, snapshot: object) -> None:
    """Restore the same read-note decision after a validated checkpoint."""
    from kernel_code_bookmarks import MAX_BOOKMARKS
    from kernel_public_localization import public_path

    if (not isinstance(snapshot, Mapping) or snapshot.get("schema") != _REPLAY_SCHEMA
            or type(snapshot.get("active")) is not bool):
        raise ValueError("invalid bookmark read replay")
    rows, failure = snapshot.get("seen"), snapshot.get("failure")
    bookmarks = snapshot.get("bookmarks", [])
    if (not isinstance(rows, list) or len(rows) > _SEEN_LIMIT
            or (failure is not None and not isinstance(failure, Mapping))
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
                            (tuple(failure_key[:5]) + (
                                hashlib.sha256(failure_key[6].encode("utf-8")).hexdigest(),
                                failure_key[6]) if failure_key is not None else None),
                            request_id)
    if failure is not None:
        if (set(failure) != {"receipt_id", "revision", "status", "outcome",
                            "failure_phase", "runner_tail_digest", "assertion"}
                or not isinstance(failure.get("runner_tail_digest"), str)
                or re.fullmatch(r"[0-9a-f]{64}", failure["runner_tail_digest"]) is None):
            raise ValueError("invalid bookmark read replay")
        failure = _canonical_failure({**failure,
                                      "assertion_context_preview": failure.get("assertion")})
        if failure is None:
            raise ValueError("invalid bookmark read replay")
    measurement._reread_result_truth = restored
    measurement._bookmark_failure_data = dict(failure) if failure is not None else None
    measurement._code_bookmarks_active = snapshot["active"]
    measurement._code_bookmark_carry = restored_bookmarks
    if "read_lockout" in snapshot:
        from kernel_read_lockout import restore as restore_lockout
        restore_lockout(measurement, snapshot["read_lockout"])


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
