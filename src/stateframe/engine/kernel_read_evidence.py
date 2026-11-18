"""MR-8 read-evidence producer (STATEFRAME_ROUTING_SCHEMA=2 only).

A read receipt is "new evidence" when its (path, content revision, line range) identity has not
been returned earlier in this run. The router projection resets ``reads_without_discriminating_action``
on a read row carrying ``new_evidence=True`` under v2. The flag is not ``meaningful_action``, so the
read never becomes a committed or completed action receipt. A repeated read of unchanged content
is not new, so the counter keeps climbing.

The seen set lives on the run's measurement object, is bounded (oldest identities drop first),
and is never created under schema 1.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Mapping

from kernel_routing_schema import routing_v2

SEEN_CAP = 4096
SEEN_ATTR = "_read_evidence_seen"

_REVISION = re.compile(r"sha256:[0-9a-f]{64}")


def _identity(delivery: Mapping[str, object]) -> str:
    """Content identity from the read's existing dedup receipt; empty when none is available."""
    dedup = delivery.get("read_dedup")
    source = dedup if isinstance(dedup, Mapping) else delivery
    revision = source.get("revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        return ""
    interval = source.get("effective_range")
    if not isinstance(interval, Mapping):
        return revision
    start, end = interval.get("start_line"), interval.get("end_line")
    if type(start) is not int or type(end) is not int:
        return revision
    return f"{revision}:{start}-{end}"


def note_read_evidence(runtime: object | None, path: str, delivery: Mapping[str, object]) -> bool:
    """Record one successful read and return True when its content is new to this run.

    Always False under schema 1, and no state is touched there.
    """
    if runtime is None or not routing_v2():
        return False
    seen = getattr(runtime, SEEN_ATTR, None)
    if seen is None:
        seen = OrderedDict()
        setattr(runtime, SEEN_ATTR, seen)
    key = (path, _identity(delivery))
    if key in seen:
        seen.move_to_end(key)
        return False
    seen[key] = True
    while len(seen) > SEEN_CAP:
        seen.popitem(last=False)
    return True
