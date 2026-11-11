"""Strict host parser for one disposable mutation fetch broker reply."""
from __future__ import annotations

import re

if __package__:
    from .kernel_mutate_wire import strict_loads
else:
    from kernel_mutate_wire import strict_loads


_HEADER = re.compile(rb"SFMB1 ([1-9][0-9]*) (0|[1-9][0-9]*)\n")
_FIELDS = {"status", "headers", "final_url", "redirects", "error"}


def parse_broker(raw: bytes, *, max_bytes: int) -> tuple[dict, bytes]:
    if type(max_bytes) is not int or not 0 <= max_bytes <= 8 << 20:
        raise ValueError("invalid broker byte limit")
    if not isinstance(raw, bytes) or len(raw) > max_bytes + (8 << 10) + 64:
        raise ValueError("broker frame exceeds limit")
    match = _HEADER.match(raw)
    if match is None or len(match.group(1)) > 5 or len(match.group(2)) > 8:
        raise ValueError("invalid broker frame header")
    metadata_size, body_size = int(match.group(1)), int(match.group(2))
    if metadata_size > 8 << 10 or body_size > max_bytes or len(raw) != match.end() + metadata_size + body_size:
        raise ValueError("broker frame length differs")
    value = strict_loads(raw[match.end():match.end() + metadata_size])
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("invalid broker metadata")
    if (value["status"] is not None and (type(value["status"]) is not int or
                                          not 0 <= value["status"] <= 599)):
        raise ValueError("invalid broker status")
    if not isinstance(value["headers"], dict) or not isinstance(value["redirects"], list):
        raise ValueError("invalid broker headers or redirects")
    if (value["final_url"] is not None and not isinstance(value["final_url"], str)) or (
            value["error"] is not None and not isinstance(value["error"], str)):
        raise ValueError("invalid broker final URL or error")
    body = raw[match.end() + metadata_size:]
    if value["error"] is not None and body:
        raise ValueError("broker error includes a body")
    return value, body
