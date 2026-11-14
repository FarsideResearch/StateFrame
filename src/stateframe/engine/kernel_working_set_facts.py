"""Assignment-derived delivery truth for the copied semantic provider body.

Only known host-owned frame fields and volatile-tail source assertions are
rewritten. Task text, opaque user suffixes and native tool payloads are opaque.
"""
from __future__ import annotations

import hashlib
import json
import re


def _whole_lines(slot) -> bool:
    span = slot.identity.lossy_span
    if span is not None:
        return span.presentation_version == "model-source-window/1"
    header = slot.page.split("\n", 1)[0]
    position = re.search(r"effective=lines:\d+-\d+@(\d+):\d+", header)
    if position is not None and int(position[1]) != 0:
        return False
    return not re.search(r"clip=(?:max_chars|max_bytes)(?: |$)", header)


def _covered(slots, path, revision, first, last) -> bool:
    if not isinstance(path, str) or not isinstance(revision, str):
        return False
    if type(first) is not int or type(last) is not int or first < 1 or last < first:
        return False
    end = first - 1
    intervals = sorted((s.identity.first_line, s.identity.last_line) for s in slots
                       if s.identity.path == path and s.identity.revision == revision
                       and _whole_lines(s))
    for start, stop in intervals:
        if start > end + 1:
            break
        end = max(end, stop)
        if end >= last:
            return True
    return False


def _path_complete(full, path, revision) -> bool:
    totals = set()
    for slot in full:
        if slot.identity.path != path or slot.identity.revision != revision:
            continue
        header = slot.page.split("\n", 1)[0]
        match = re.search(r"\blines=\d+-\d+/(\d+)(?: |$)", header)
        if match is not None:
            totals.add(int(match[1]))
        elif "[Model source window]\n" in slot.page:
            encoded = slot.page.split("[Model source window]\n", 1)[1].split("\n", 1)[0]
            total = json.loads(encoded).get("total_lines")
            if type(total) is int:
                totals.add(total)
    return len(totals) == 1 and _covered(full, path, revision, 1, next(iter(totals)))


def _update_frame(frame, snapshot, full):
    revisions = dict(snapshot.current_revisions)
    all_sources = snapshot.sources
    notices = {}
    for fact in frame.get("read_facts", ()):
        if not isinstance(fact, dict):
            continue
        path, revision = fact.get("path"), fact.get("read_revision")
        represented = [s for s in all_sources if s.identity.path == path
                       and s.identity.revision == revision]
        lines = fact.get("lines")
        lines = lines if isinstance(lines, dict) else {}
        # A per-path fact represents all known captures at that revision, not
        # whichever same-path excerpt happens to remain. Historical aggregates
        # with unknown receipt coordinates fail closed rather than inheriting
        # authority from the current range.
        history = fact.get("historical_receipts")
        unknown_history = isinstance(history, dict) and bool(history.get("count", 0))
        receipts = fact.get("receipts")
        declared_count = fact.get("receipt_count", len(receipts) if isinstance(receipts, list) else 1)
        captured_count = sum(max(1, len(s.capture_ids)) for s in represented)
        unrepresented_receipts = (type(declared_count) is not int or declared_count < 1
                                  or declared_count > captured_count)
        present = (bool(represented) and not unknown_history
                   and not unrepresented_receipts
                   and all(s in full for s in represented)
                   and revision == revisions.get(path)
                   and _covered(full, path, revision, lines.get("start"), lines.get("end")))
        fact["body_delivery"] = "present" if present else "not_shown"
        fact.pop("visible_symbols", None)
        current = revisions.get(path, fact.get("current_revision", revision))
        if present:
            fact.pop("delivery_notice", None)
            notices[path] = "fresh"
        else:
            notice = f"[not shown] {path} @rev {current or 'unavailable'}: re-read to see it"
            fact["delivery_notice"] = notice
            notices[path] = notice

    sites = list(frame.get("open_sites", ()))
    if isinstance(frame.get("next_site"), dict):
        sites.append(frame["next_site"])
    for site in sites:
        if not isinstance(site, dict):
            continue
        lines = site.get("lines")
        lines = lines if isinstance(lines, dict) else {}
        path, revision = site.get("path"), site.get("revision")
        present = (revision == revisions.get(path)
                   and _covered(full, path, revision, lines.get("start"), lines.get("end")))
        site["read_this_request"] = present
        # current_bytes is an independently embedded source carrier in the
        # legacy frame. It must not survive a deferred slot or stale revision.
        # Even for FULL it must be exact bytes of a selected captured interval.
        value = site.get("current_bytes")
        if (not present or not isinstance(value, str)
                or not any(value and value in s.payload and s.identity.path == path
                           and s.identity.revision == revision for s in full)):
            site["current_bytes"] = None

    # Capsule labels are delivery claims, while their journal history remains
    # immutable. Re-hash the copied capsule whenever its assignment label moves.
    for row in frame.get("prior_user_context", ()):
        if not isinstance(row, dict) or not isinstance(row.get("content"), str):
            continue
        content = row["content"]
        if "earlier exchanges trimmed to fit context. CAPSULE" not in content.split("\n", 1)[0]:
            continue
        lines = content.splitlines(keepends=True)
        knowledge = False
        for index, line in enumerate(lines):
            if line.startswith("WORLD KNOWLEDGE "):
                knowledge = True
            elif line.startswith("+"):
                knowledge = False
            elif knowledge and line.startswith("- "):
                for path, notice in notices.items():
                    if line.startswith((f"- {path} ", f"- {path}::")):
                        lines[index] = re.sub(
                            r" \| (?:fresh|\[not shown\].*?: re-read to see it|\[bookmark shown\][^|]*) \| (read@\d+)",
                            lambda match: f" | {notice} | {match[1]}", line)
                        break
        updated = "".join(lines)
        if updated != content:
            row["content"] = updated
            row["content_sha256"] = hashlib.sha256(updated.encode()).hexdigest()


def rebuild_delivery_truth(body, snapshot, full):
    """Update typed host fields in an already detached semantic body."""
    for message in body["messages"]:
        content = message.get("content")
        if not isinstance(content, str) or message.get("role") != "user":
            continue
        if content.startswith("[StateFrame]\n"):
            payload = json.loads(content.split("\n", 1)[1])
            state = payload.get("state", payload)
            frame = state.get("framefit5")
            if isinstance(frame, dict):
                _update_frame(frame, snapshot, full)
            access = state.get("source_access")
            if isinstance(access, dict):
                path = access.get("path")
                represented = [s for s in snapshot.sources if s.identity.path == path]
                revision = dict(snapshot.current_revisions).get(path)
                # This legacy field has no receipt coordinates. Require all
                # represented intervals at the positively observed revision;
                # a single selected excerpt cannot forgive another interval.
                included = (bool(represented) and all(
                    s in full and s.identity.revision == revision for s in represented)
                    and _path_complete(full, path, revision))
                access["included"] = included
                access["no_read_call_required_for_this_included_source"] = included
            message["content"] = "[StateFrame]\n" + json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        elif content.startswith("[WIRE VOLATILE TAIL]"):
            # This exact assembler-owned line must not carry a previous
            # selection into a new assignment. The renderer emits its scoped
            # host facts from certified FULL carriers instead.
            message["content"] = re.sub(
                r"(?m)^\[host\] The current source of [^\n]+ lines \d+-\d+ is in this request "
                r"\(see the read pages above\)\.(?: You can edit it now\.)?\n?", "", content)
