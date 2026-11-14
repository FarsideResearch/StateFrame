"""Pure proof of an executable recovery call for one omitted captured interval.

A path or range printed in a notice never authorizes recovery. The frozen host
receipt supplies coordinates, while the actual final schema and owner supply
callability. This module neither reads source files nor invokes a tool.
"""
from __future__ import annotations

import json
import re

if __package__:
    from .kernel_completion_owner import validate_json_schema
    from .kernel_source_delivery import (MAX_READ_CHARS, MAX_READ_LINES,
                                         MIN_READ_CHARS, normalize_read_request)
else:
    from kernel_completion_owner import validate_json_schema
    from kernel_source_delivery import (MAX_READ_CHARS, MAX_READ_LINES,
                                        MIN_READ_CHARS, normalize_read_request)


class RecoveryRouteUnavailable(ValueError):
    """The frozen final read surface cannot recover this exact interval."""


_COORDINATES = frozenset({"path", "line", "start_line", "end_line", "max_lines", "max_chars",
                          "column", "cursor", "symbol", "expected_revision", "page_size"})
_PAGE_POSITION = re.compile(r"effective=lines:(\d+)-(\d+)@(\d+):(\d+)(?: |$)")


def _cursor_position(value, revision):
    if not isinstance(value, str):
        return None
    parts = value.split(":")
    digest = revision.removeprefix("sha256:")
    if (len(parts) not in {4, 5} or parts[0] != "read1"
            or parts[1] not in {digest, digest[:24]}):
        return None
    try:
        line, column = int(parts[2]), int(parts[3])
        scope = int(parts[4]) if len(parts) == 5 else None
    except ValueError:
        return None
    if line < 1 or column < 0 or (scope is not None and scope < line):
        return None
    return line, column, scope


def _start_column(slot, original):
    identity = slot.identity
    if identity.lossy_span is not None:
        return identity.lossy_span.first_column
    match = _PAGE_POSITION.search(slot.page.split("\n", 1)[0])
    if match is not None:
        if int(match[1]) != identity.first_line or int(match[2]) != identity.last_line:
            raise RecoveryRouteUnavailable("captured page coordinates differ from receipt")
        return int(match[3])
    position = _cursor_position(original.get("cursor"), identity.revision)
    if position is not None and position[0] == identity.first_line:
        return position[1]
    column = original.get("column", 0)
    if type(column) is not int or column < 0:
        raise RecoveryRouteUnavailable("captured source column is unavailable")
    return column


def _schema(snapshot):
    if snapshot.recovery_schema:
        value = json.loads(snapshot.recovery_schema)
        if not isinstance(value, dict):
            raise RecoveryRouteUnavailable("frozen read schema is malformed")
        return value
    # Direct, host-constructed snapshots use the established source pager's
    # argument contract. Production freeze_preparation records the actual final
    # tool schema and marks missing tools unavailable before reaching this path.
    return {"type": "object", "required": ["path"], "properties": {
        "path": {"type": "string"}, "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
        "max_lines": {"type": "integer", "minimum": 1, "maximum": MAX_READ_LINES},
        "max_chars": {"type": "integer", "minimum": MIN_READ_CHARS, "maximum": MAX_READ_CHARS},
        "column": {"type": "integer", "minimum": 0}, "cursor": {"type": "string"},
        "expected_revision": {"type": "string"}}}


def validate_recovery(snapshot, slot) -> dict[str, object]:
    """Return proven concrete read arguments, or refuse the omission.

    Captured initial arguments can describe a larger range or symbol than the
    admitted page. Rebuild from the receipt rather than replaying that claim.
    Cursor-only surfaces retain an existing cursor after exact coordinate and
    revision checks; no new cursor format or positional guess is introduced.
    """
    if not snapshot.recovery_allowed or not snapshot.recovery_owner_allowed:
        raise RecoveryRouteUnavailable("frozen tool or owner refuses source recovery")
    original = json.loads(slot.recovery_arguments)
    identity = slot.identity
    if not isinstance(original, dict) or original.get("path") != identity.path:
        raise RecoveryRouteUnavailable("recovery path differs from receipt")
    current = dict(snapshot.current_revisions).get(identity.path)
    if current is not None and current != identity.revision:
        raise RecoveryRouteUnavailable("captured interval is not at the frozen current revision")
    line_count = identity.last_line - identity.first_line + 1
    if not 1 <= line_count <= MAX_READ_LINES or len(slot.payload) > MAX_READ_CHARS:
        raise RecoveryRouteUnavailable("captured interval exceeds a single read page")
    column = _start_column(slot, original)
    max_chars = max(MIN_READ_CHARS, len(slot.payload))
    extras = {key: value for key, value in original.items() if key not in _COORDINATES}
    common = {**extras, "path": identity.path, "max_chars": max_chars}
    if column:
        common["column"] = column
    variants = [
        {**common, "start_line": identity.first_line, "end_line": identity.last_line},
        {**common, "start_line": identity.first_line, "max_lines": line_count},
    ]
    position = _cursor_position(original.get("cursor"), identity.revision)
    if (position is not None and position[:2] == (identity.first_line, column)
            and (position[2] is None or position[2] >= identity.last_line)):
        cursor_common = {key: value for key, value in common.items() if key != "column"}
        variants.append({**cursor_common, "cursor": original["cursor"], "max_lines": line_count})
    schema = _schema(snapshot)
    properties = schema.get("properties", {})
    # Revision guards are used whenever the final schema admits them. Replaying
    # an expected revision from a different capture is never permitted.
    revision_allowed = (isinstance(properties, dict) and "expected_revision" in properties
                        or schema.get("additionalProperties", True) is not False)
    for arguments in variants:
        guarded = {**arguments, "expected_revision": identity.revision} if revision_allowed else arguments
        if validate_json_schema(schema, guarded):
            continue
        normalized = normalize_read_request(guarded)
        if normalized.mode not in {"lines", "cursor"} or normalized.max_lines < line_count:
            continue
        if normalized.max_chars < len(slot.payload):
            continue
        return guarded
    raise RecoveryRouteUnavailable("final read schema cannot recover the captured interval")
