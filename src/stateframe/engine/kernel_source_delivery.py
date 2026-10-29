"""Exact, bounded source pages shared by read receipts and final delivery."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

from kernel_paged_source import page_window_args

DEFAULT_MAX_LINES = 120
MAX_READ_LINES = 1000
DEFAULT_MAX_CHARS = 12000
MAX_READ_CHARS = 64000
MIN_READ_CHARS = 512
READ_CONTRACT_FIELDS = (
    "requested_range",
    "effective_range",
    "clip_reason",
    "next_cursor",
)


class ReadArgumentError(ValueError):
    """A malformed or contradictory read contract the caller can repair."""

    def __init__(self, code: str, field: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.field = field
        self.reason = reason

    def as_dict(self, *, path: str) -> dict[str, object]:
        return {
            "schema": "stateframe-read-argument-error/1",
            "status": "REJECTED",
            "code": self.code,
            "tool": "read",
            "field": self.field,
            "path": path,
            "reason": self.reason,
            "repair": "Use one inclusive start_line/end_line range, or a revision-bound cursor with max_lines/max_chars.",
        }


@dataclass(frozen=True, slots=True)
class NormalizedReadRequest:
    """One validated read request and the limits used by the source pager."""

    arguments: Mapping[str, object]
    mode: str
    requested_start_line: int | None
    requested_end_line: int | None
    end_line_explicit: bool
    max_lines: int
    max_chars: int
    max_bytes: int | None
    identity_arguments: Mapping[str, object]


def _read_integer(value: object, *, field: str, minimum: int,
                  maximum: int) -> int:
    if isinstance(value, bool):
        raise ReadArgumentError("INVALID_READ_ARGUMENT", field,
                                f"{field} must be an integer from {minimum} to {maximum}.")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdecimal():
        number = int(value.strip())
    else:
        raise ReadArgumentError("INVALID_READ_ARGUMENT", field,
                                f"{field} must be an integer from {minimum} to {maximum}.")
    if number < minimum or number > maximum:
        raise ReadArgumentError("READ_LIMIT_OUT_OF_RANGE", field,
                                f"{field} must be from {minimum} to {maximum}.")
    return number


def normalize_read_request(args: object, *, default_start_line: int = 1,
                           page_line_limit: int | None = None,
                           page_byte_limit: int | None = None,
                           ) -> NormalizedReadRequest:
    """Validate read coordinates and apply an optional internal page-line budget."""
    options = dict(args) if isinstance(args, Mapping) else {}
    if page_line_limit is not None:
        page_line_limit = _read_integer(
            page_line_limit, field="page_line_limit", minimum=1,
            maximum=MAX_READ_LINES,
        )
    if page_byte_limit is not None:
        page_byte_limit = _read_integer(
            page_byte_limit, field="page_byte_limit", minimum=1,
            maximum=MAX_READ_CHARS * 4,
        )
    expected_revision = options.get("expected_revision")
    if expected_revision not in (None, ""):
        if (not isinstance(expected_revision, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_revision)):
            raise ReadArgumentError(
                "INVALID_READ_ARGUMENT", "expected_revision",
                "expected_revision must be a full sha256 source revision.",
            )
    line = options.get("line")
    start = options.get("start_line")
    line_present = line not in (None, "")
    start_present = start not in (None, "")
    end = options.get("end_line")
    end_present = end not in (None, "")
    cursor = options.get("cursor")
    cursor_present = cursor not in (None, "")
    symbol = options.get("symbol")
    symbol_present = symbol not in (None, "")

    if line_present and start_present:
        line_number = _read_integer(line, field="line", minimum=1, maximum=10**9)
        start_number = _read_integer(start, field="start_line", minimum=1, maximum=10**9)
        if line_number != start_number:
            raise ReadArgumentError(
                "CONTRADICTORY_READ_COORDINATES", "line,start_line",
                "line and start_line identify different starting lines.",
            )
    elif line_present or start_present:
        value = start if start_present else line
        start_number = _read_integer(
            value, field="start_line" if start_present else "line",
            minimum=1, maximum=10**9)
    else:
        start_number = max(1, int(default_start_line))

    explicit_max_lines = options.get("max_lines") not in (None, "")
    if cursor_present and (line_present or start_present or end_present or symbol_present):
        raise ReadArgumentError(
            "CONTRADICTORY_READ_COORDINATES", "cursor",
            "cursor cannot be combined with line, start_line, end_line, or symbol.",
        )
    if symbol_present and (line_present or start_present or end_present):
        raise ReadArgumentError(
            "CONTRADICTORY_READ_COORDINATES", "symbol",
            "symbol cannot be combined with explicit line coordinates.",
        )
    if end_present and explicit_max_lines:
        raise ReadArgumentError(
            "CONTRADICTORY_READ_COORDINATES", "end_line,max_lines",
            "end_line defines the inclusive range; omit max_lines or use a cursor page.",
        )

    if cursor_present and not isinstance(cursor, str):
        raise ReadArgumentError("INVALID_READ_CURSOR", "cursor",
                                "cursor must be a revision-bound string.")
    if symbol_present and not isinstance(symbol, str):
        raise ReadArgumentError("INVALID_READ_ARGUMENT", "symbol",
                                "symbol must be a non-empty string.")
    if symbol_present and not symbol.strip():
        raise ReadArgumentError("INVALID_READ_ARGUMENT", "symbol",
                                "symbol must be a non-empty string.")

    max_chars = _read_integer(
        options.get("max_chars") if options.get("max_chars") not in (None, "") else DEFAULT_MAX_CHARS,
        field="max_chars", minimum=MIN_READ_CHARS, maximum=MAX_READ_CHARS)
    if end_present:
        end_number = _read_integer(end, field="end_line", minimum=1, maximum=10**9)
        if end_number < start_number:
            raise ReadArgumentError(
                "CONTRADICTORY_READ_COORDINATES", "start_line,end_line",
                "end_line must be greater than or equal to start_line.",
            )
        max_lines = min(MAX_READ_LINES, end_number - start_number + 1)
        if page_line_limit is not None:
            max_lines = min(max_lines, page_line_limit)
        mode = "lines"
    else:
        end_number = None
        default_max_lines = (
            page_line_limit if page_line_limit is not None else DEFAULT_MAX_LINES
        )
        max_lines = _read_integer(
            options.get("max_lines") if explicit_max_lines else default_max_lines,
            field="max_lines", minimum=1, maximum=MAX_READ_LINES)
        if page_line_limit is not None:
            max_lines = min(max_lines, page_line_limit)
        mode = "cursor" if cursor_present else "symbol" if symbol_present else "lines"

    column_value = options.get("column")
    column = (None if column_value in (None, "") else
              _read_integer(column_value, field="column", minimum=0, maximum=10**9))
    if cursor_present and column is not None:
        raise ReadArgumentError(
            "CONTRADICTORY_READ_COORDINATES", "cursor,column",
            "cursor already contains the line and column position; omit column.",
        )

    canonical = dict(options)
    canonical.pop("line", None)
    if line_present or start_present:
        canonical["start_line"] = start_number
    if end_present:
        canonical["end_line"] = end_number
    if column is not None:
        canonical["column"] = column
    identity: dict[str, object] = {
        "mode": mode,
        "max_lines": max_lines,
        "max_chars": max_chars,
    }
    if page_byte_limit is not None:
        identity["max_bytes"] = page_byte_limit
    if mode == "cursor":
        identity["cursor"] = cursor
    elif mode == "symbol":
        identity["symbol"] = symbol.strip()
    else:
        identity["start_line"] = start_number
        identity["end_line"] = end_number
        identity["end_line_explicit"] = end_present
        if column is not None:
            identity["column"] = column
    if expected_revision not in (None, ""):
        identity["expected_revision"] = expected_revision

    return NormalizedReadRequest(
        arguments=canonical,
        mode=mode,
        requested_start_line=None if mode == "cursor" else start_number,
        requested_end_line=end_number,
        end_line_explicit=end_present,
        max_lines=max_lines,
        max_chars=max_chars,
        max_bytes=page_byte_limit,
        identity_arguments=identity,
    )


@dataclass(frozen=True)
class SourceDelivery:
    result_class: str
    text: str
    content: str
    metadata: Mapping[str, object]


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if value in (None, "", 0):
        return default
    try:
        return max(minimum, min(maximum, int(value)))
    except (TypeError, ValueError):
        return default


def _revision(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _cursor(raw: bytes, line: int, column: int, scope_end: int | None = None) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    suffix = f":{scope_end}" if scope_end is not None else ""
    return f"read1:{digest}:{line}:{column}{suffix}"


def _position_from_cursor(raw: bytes, value: object,
                          default_line: int, default_column: int
                          ) -> tuple[int, int, str | None, int | None]:
    if value in (None, ""):
        return default_line, default_column, None, None
    parts = str(value).split(":")
    expected = hashlib.sha256(raw).hexdigest()
    if len(parts) not in {4, 5} or parts[0] != "read1" or parts[1] != expected:
        # Accept cursors minted before the full revision digest was introduced;
        # newly returned cursors always carry the complete SHA-256 revision.
        if (len(parts) not in {4, 5} or parts[0] != "read1"
                or parts[1] != expected[:24]):
            return default_line, default_column, "READ_CURSOR_STALE", None
    try:
        line, column = int(parts[2]), int(parts[3])
        scope_end = int(parts[4]) if len(parts) == 5 else None
    except ValueError:
        return default_line, default_column, "READ_CURSOR_INVALID", None
    if line < 1 or column < 0:
        return default_line, default_column, "READ_CURSOR_INVALID", None
    if scope_end is not None and scope_end < line:
        return default_line, default_column, "READ_CURSOR_INVALID", None
    return line, column, None, scope_end


def _header(path: str, revision: str, *, start: int, end: int, total: int,
            chars: int, byte_count: int, truncated: bool,
            decoded_with_replacement: bool, requested: str,
            effective: str, clip_reason: str | None) -> str:
    return (
        f"[read page] path={path} revision={revision} lines={start}-{end}/{total} "
        f"requested={requested} effective={effective} "
        f"clip={clip_reason or 'none'} chars={chars} bytes={byte_count} "
        f"truncated={str(truncated).lower()} "
        f"decoded_with_replacement={str(decoded_with_replacement).lower()}"
    )


def _range_label(value: Mapping[str, object] | None) -> str:
    if value is None:
        return "none"
    if value.get("kind") == "lines":
        return f"lines:{value.get('start_line')}-{value.get('end_line')}"
    if value.get("kind") == "cursor":
        return f"cursor:{str(value.get('cursor') or '')[:28]}"
    return f"symbol:{str(value.get('symbol') or '')[:80]}"


def _prefix_within_limits(value: str, *, max_chars: int,
                          max_bytes: int | None) -> str:
    candidate = value[:max_chars]
    if max_bytes is None or len(candidate.encode("utf-8")) <= max_bytes:
        return candidate
    selected: list[str] = []
    byte_count = 0
    for character in candidate:
        character_bytes = len(character.encode("utf-8"))
        if byte_count + character_bytes > max_bytes:
            break
        selected.append(character)
        byte_count += character_bytes
    return "".join(selected)


def _requested_range(request: NormalizedReadRequest) -> dict[str, object]:
    if request.mode == "lines":
        return {
            "kind": "lines",
            "start_line": request.requested_start_line,
            "end_line": request.requested_end_line or (
                (request.requested_start_line or 1) + request.max_lines - 1),
            "end_line_explicit": request.end_line_explicit,
        }
    options = request.arguments
    identity: dict[str, object] = {
        "kind": request.mode,
        "max_lines": request.max_lines,
        "max_chars": request.max_chars,
    }
    if request.mode == "cursor":
        identity["cursor"] = options.get("cursor")
    elif request.mode == "symbol":
        identity["symbol"] = options.get("symbol")
    return identity


def deliver_source(raw: bytes, display: str, start: int, *, args: object,
                   metadata_out: MutableMapping[str, object] | None = None) -> SourceDelivery:
    """Shape one source page before either receipt creation or message serialization."""
    request = (args if isinstance(args, NormalizedReadRequest) else
               normalize_read_request(args, default_start_line=start + 1))
    options = request.arguments
    revision = _revision(raw)

    def finish(result_class: str, text: str, content: str,
               metadata: dict[str, object]) -> SourceDelivery:
        content_bytes = content.encode("utf-8", errors="replace")
        metadata.setdefault("requested_range", _requested_range(request))
        metadata.setdefault("effective_range", None)
        metadata.setdefault("clip_reason", None)
        metadata.setdefault("next_cursor", metadata.get("continuation"))
        metadata.setdefault("byte_length", len(content_bytes))
        metadata.setdefault("page_bytes_sha256", "sha256:" + hashlib.sha256(
            content_bytes).hexdigest())
        if metadata_out is not None:
            metadata_out.update(metadata)
        return SourceDelivery(result_class, text, content, metadata)

    try:
        decoded = raw.decode("utf-8")
        replaced = False
    except UnicodeDecodeError:
        decoded = raw.decode("utf-8", errors="replace")
        replaced = True
    lines = decoded.splitlines(keepends=True)
    total = len(lines)
    requested_start, range_end, range_error = page_window_args(options, decoded)
    if request.mode == "lines":
        requested_start = request.requested_start_line
        range_end = request.requested_end_line
    if range_error is not None:
        requested_range = {
            "kind": "lines", "start_line": request.requested_start_line,
            "end_line": request.requested_end_line or
            ((request.requested_start_line or 1) + request.max_lines - 1),
            "end_line_explicit": request.end_line_explicit,
        } if request.mode == "lines" else {
            "kind": request.mode,
            **({"cursor": options.get("cursor")} if request.mode == "cursor" else
               {"symbol": options.get("symbol")} if request.mode == "symbol" else {}),
            "max_lines": request.max_lines,
            "max_chars": request.max_chars,
        }
        metadata = {
            "schema": "read-page/1", "path": display, "revision": revision,
            "result_class": range_error, "requested_symbol": options.get("symbol"),
            "total_lines": total, "truncated": False, "continuation": None,
            "requested_range": requested_range, "effective_range": None,
            "clip_reason": "symbol_not_found",
        }
        text = (f"{range_error}: the requested symbol is not present in the current "
                "source revision; read the symbol outline or choose a line range.")
        return finish(range_error, text, "", metadata)
    requested_column = _bounded_int(options.get("column"), 0, 0, 10**9)
    requested_line, requested_column, cursor_error, cursor_scope_end = _position_from_cursor(
        raw, options.get("cursor"),
        requested_start if requested_start is not None else start + 1,
        requested_column,
    )
    if cursor_scope_end is not None and range_end is None:
        range_end = cursor_scope_end
    paged_scope = (
        request.mode in {"cursor", "symbol"}
        or request.end_line_explicit
        or any(key in options for key in ("line", "start_line"))
        or start > 0
    )
    max_lines = request.max_lines
    max_chars = request.max_chars
    if range_end is not None:
        max_lines = min(max_lines, max(1, range_end - requested_line + 1))
    page_end = min(total, range_end) if range_end is not None else total

    if cursor_error is not None:
        requested_range = {
            "kind": "lines", "start_line": request.requested_start_line,
            "end_line": request.requested_end_line or
            ((request.requested_start_line or 1) + request.max_lines - 1),
            "end_line_explicit": request.end_line_explicit,
        } if request.mode == "lines" else {
            "kind": request.mode,
            **({"cursor": options.get("cursor")} if request.mode == "cursor" else
               {"symbol": options.get("symbol")} if request.mode == "symbol" else {}),
            "max_lines": request.max_lines,
            "max_chars": request.max_chars,
        }
        metadata = {
            "schema": "read-page/1", "path": display, "revision": revision,
            "result_class": cursor_error, "requested_start_line": requested_line,
            "requested_start_column": requested_column, "total_lines": total,
            "truncated": False, "continuation": None,
            "requested_range": requested_range, "effective_range": None,
            "clip_reason": "stale_cursor" if cursor_error == "READ_CURSOR_STALE"
            else "invalid_cursor",
        }
        text = (f"{cursor_error}: the source changed or the continuation is invalid; "
                "repeat the read from its path and line.")
        return finish(cursor_error, text, "", metadata)

    if requested_line > total or (total == 0 and requested_line >= 1):
        requested_range = {
            "kind": "lines", "start_line": request.requested_start_line,
            "end_line": request.requested_end_line or
            ((request.requested_start_line or 1) + request.max_lines - 1),
            "end_line_explicit": request.end_line_explicit,
        } if request.mode == "lines" else {
            "kind": request.mode,
            **({"cursor": options.get("cursor")} if request.mode == "cursor" else
               {"symbol": options.get("symbol")} if request.mode == "symbol" else {}),
            "max_lines": request.max_lines,
            "max_chars": request.max_chars,
        }
        metadata = {
            "schema": "read-page/1", "path": display, "revision": revision,
            "result_class": "OUT_OF_RANGE", "requested_start_line": requested_line,
            "requested_start_column": requested_column, "start_line": requested_line,
            "selected_end_line": requested_line - 1, "shown_end_line": requested_line - 1,
            "start_column": requested_column, "shown_end_column": requested_column,
            "total_lines": total, "shown_chars": 0, "truncated": False,
            "more_content_available": False, "more_lines_available": False,
            "continuation": None, "decoded_with_replacement": replaced,
            "requested_range": requested_range, "effective_range": None,
            "clip_reason": "end_of_file",
        }
        if paged_scope:
            metadata["scope_end_line"] = range_end
        text = (f"{display.rsplit('/', 1)[-1]} has {total} lines; you asked to start at "
                f"{requested_line}, which is past the end. Valid range is 1-{total}.")
        return finish("OUT_OF_RANGE", text, "", metadata)

    line_index = requested_line - 1
    first_line = lines[line_index]
    if requested_column > len(first_line):
        metadata = {
            "schema": "read-page/1", "path": display, "revision": revision,
            "result_class": "OUT_OF_RANGE", "requested_start_line": requested_line,
            "requested_start_column": requested_column, "total_lines": total,
            "truncated": False, "continuation": None, "decoded_with_replacement": replaced,
            "clip_reason": "column_out_of_range",
        }
        text = (f"{display.rsplit('/', 1)[-1]} line {requested_line} has "
                f"{len(first_line)} characters; you asked for column {requested_column}.")
        return finish("OUT_OF_RANGE", text, "", metadata)

    # Reserve room for the receipt header and continuation line.  The source body
    # is the only part allowed to consume the remaining character budget.
    overhead = min(max_chars - 1, max(512, len(display) + len(revision) + 320))
    body_budget = max(1, max_chars - overhead)
    max_bytes = request.max_bytes
    remaining_bytes: int | None = None
    if max_bytes is not None:
        byte_overhead = min(
            max_bytes - 1,
            max(512, len(display.encode("utf-8")) + len(revision.encode("utf-8")) + 320),
        )
        remaining_bytes = max(1, max_bytes - byte_overhead)
    body_parts: list[str] = []
    remaining = body_budget
    current_line = line_index
    current_column = requested_column
    end_line = current_line
    end_column = current_column
    next_line: int | None = None
    next_column: int | None = None
    consumed_lines = 0
    line_limit_hit = False
    character_limit_hit = False
    byte_limit_hit = False
    while current_line < page_end and consumed_lines < max_lines and remaining > 0:
        portion = lines[current_line][current_column:]
        portion_bytes = len(portion.encode("utf-8"))
        if (len(portion) <= remaining
                and (remaining_bytes is None or portion_bytes <= remaining_bytes)):
            body_parts.append(portion)
            remaining -= len(portion)
            if remaining_bytes is not None:
                remaining_bytes -= portion_bytes
            end_line = current_line
            end_column = current_column + len(portion)
            consumed_lines += 1
            current_line += 1
            current_column = 0
            if current_line < page_end and consumed_lines < max_lines:
                continue
            if current_line < page_end:
                next_line, next_column = current_line + 1, 0
                line_limit_hit = consumed_lines >= max_lines
            break
        take_text = _prefix_within_limits(
            portion, max_chars=remaining,
            max_bytes=remaining_bytes,
        )
        if not take_text:
            next_line, next_column = current_line + 1, current_column
            byte_limit_hit = remaining_bytes is not None
            character_limit_hit = remaining <= 0
            break
        take = len(take_text)
        take_bytes = len(take_text.encode("utf-8"))
        body_parts.append(take_text)
        end_line = current_line
        end_column = current_column + take
        next_line, next_column = current_line + 1, end_column
        character_limit_hit = take == remaining and take < len(portion)
        byte_limit_hit = (
            remaining_bytes is not None
            and take < remaining
            and len(portion[:remaining].encode("utf-8")) > remaining_bytes
        )
        remaining -= take
        if remaining_bytes is not None:
            remaining_bytes -= take_bytes
        break

    body = "".join(body_parts)
    if next_line is None and current_line < page_end:
        next_line, next_column = current_line + 1, 0
        line_limit_hit = True
    truncated = next_line is not None
    if truncated and remaining <= 0:
        character_limit_hit = True
    if truncated and remaining_bytes is not None and remaining_bytes <= 0:
        byte_limit_hit = True
    clip_reason = ("max_bytes" if byte_limit_hit else
                   "max_chars" if character_limit_hit else
                   "max_lines" if line_limit_hit else None)
    continuation = (_cursor(
        raw, next_line, next_column or 0,
        range_end if paged_scope else None,
    ) if truncated else None)
    more_lines = bool(next_line is not None and next_line <= total)
    if next_line is not None and next_column:
        more_text = f"more content in line {next_line}; page with cursor={continuation}"
    elif truncated:
        more_text = f"{total - next_line + 1} more lines; page with cursor={continuation}"
    else:
        more_text = "0 more; complete=true"
    body_bytes = body.encode("utf-8", errors="replace")
    requested_range = {
        "kind": "lines", "start_line": request.requested_start_line,
        "end_line": request.requested_end_line or
        ((request.requested_start_line or 1) + request.max_lines - 1),
        "end_line_explicit": request.end_line_explicit,
    } if request.mode == "lines" else {
        "kind": request.mode,
        **({"cursor": options.get("cursor")} if request.mode == "cursor" else
           {"symbol": options.get("symbol")} if request.mode == "symbol" else {}),
        "max_lines": request.max_lines,
        "max_chars": request.max_chars,
    }
    effective_range = {
        "start_line": requested_line, "end_line": end_line + 1,
        "start_column": requested_column, "end_column": end_column,
        "byte_start": sum(len(item.encode("utf-8")) for item in lines[:line_index]) +
        len(lines[line_index][:requested_column].encode("utf-8")),
        "byte_end": (sum(len(item.encode("utf-8")) for item in lines[:line_index]) +
                     len(lines[line_index][:requested_column].encode("utf-8")) + len(body_bytes)),
    }
    if (request.mode == "lines" and request.requested_end_line is not None
            and request.requested_end_line > total and clip_reason is None):
        clip_reason = "end_of_file"
    requested_label = _range_label(requested_range)
    effective_label = (f"lines:{effective_range['start_line']}-{effective_range['end_line']}"
                       f"@{effective_range['start_column']}:{effective_range['end_column']}")
    header = _header(display, revision, start=requested_line, end=end_line + 1,
                     total=total, chars=len(body), byte_count=len(body_bytes),
                     truncated=truncated, decoded_with_replacement=replaced,
                     requested=requested_label, effective=effective_label,
                     clip_reason=clip_reason)
    text = header + "\n" + more_text + "\n" + body
    metadata = {
        "schema": "read-page/1", "path": display, "revision": revision,
        "result_class": "OK", "start_line": requested_line,
        "start_column": requested_column, "selected_end_line": end_line + 1,
        "shown_end_line": end_line + 1, "shown_end_column": end_column,
        "total_lines": total, "shown_chars": len(body), "max_lines": max_lines,
        "max_chars": max_chars, "truncated": truncated,
        "max_bytes": max_bytes,
        "more_content_available": truncated, "more_lines_available": more_lines,
        "continuation": continuation, "decoded_with_replacement": replaced,
        "next_cursor": continuation, "requested_range": requested_range,
        "effective_range": effective_range, "clip_reason": clip_reason,
        "token_count": max(1, (len(body) + 3) // 4),
        "decoding": "utf-8", "presentation_version": "read-page/1",
        "byte_length": len(body.encode("utf-8", errors="replace")),
        "page_bytes_sha256": "sha256:" + hashlib.sha256(body_bytes).hexdigest(),
    }
    if not replaced:
        metadata["byte_start"] = sum(len(item.encode("utf-8")) for item in lines[:line_index]) + len(
            lines[line_index][:requested_column].encode("utf-8"))
        metadata["byte_end"] = metadata["byte_start"] + len(body.encode("utf-8"))
    if paged_scope:
        metadata.update({"requested_symbol": options.get("symbol"), "scope_end_line": range_end})
    return finish("OK", text, body, metadata)


__all__ = [
    "DEFAULT_MAX_CHARS", "DEFAULT_MAX_LINES", "MAX_READ_CHARS", "MAX_READ_LINES",
    "MIN_READ_CHARS", "NormalizedReadRequest", "ReadArgumentError", "SourceDelivery",
    "deliver_source", "normalize_read_request",
]
