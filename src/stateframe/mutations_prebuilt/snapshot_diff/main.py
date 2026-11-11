"""Compare two immutable input snapshots without touching the workspace."""
from collections import Counter
import hashlib
import io


MAX_INPUT = 4 << 20
MAX_LINES = 10000
MAX_DIFF_LINES = 200
MAX_LINE_CHARS = 1000


def _read(ctx, input_id):
    if not isinstance(input_id, str) or input_id not in ctx.inputs:
        raise ValueError("snapshot input id is required")
    path = ctx.inputs[input_id]
    raw = ctx.input_bytes(input_id, limit=MAX_INPUT + 1)
    truncated = len(raw) > MAX_INPUT
    raw = raw[:MAX_INPUT]
    lines = []
    for line in io.StringIO(raw.decode("utf-8", errors="replace")):
        if len(lines) >= MAX_LINES:
            truncated = True
            break
        line = line.rstrip("\r\n")
        if len(line) > MAX_LINE_CHARS:
            truncated = True
        lines.append(line[:MAX_LINE_CHARS])
    return raw, lines, path.stat().st_size, truncated


def _bounded_lines(values):
    return [line[:MAX_LINE_CHARS] for line in values[:MAX_DIFF_LINES]]


def handle(input, ctx):
    left, left_lines, left_size, left_truncated = _read(ctx, input.get("left_id"))
    right, right_lines, right_size, right_truncated = _read(ctx, input.get("right_id"))
    left_counts = Counter(left_lines)
    right_counts = Counter(right_lines)
    added = []
    removed = []
    for line, count in right_counts.items():
        added.extend([line] * max(0, count - left_counts[line]))
    for line, count in left_counts.items():
        removed.extend([line] * max(0, count - right_counts[line]))
    added.sort()
    removed.sort()
    return {
        "changed": left != right or left_size != right_size,
        "left_bytes": left_size,
        "right_bytes": right_size,
        "left_sha256": hashlib.sha256(left).hexdigest(),
        "right_sha256": hashlib.sha256(right).hexdigest(),
        "added_lines": _bounded_lines(added),
        "removed_lines": _bounded_lines(removed),
        "truncated": (left_truncated or right_truncated or
                      len(added) > MAX_DIFF_LINES or len(removed) > MAX_DIFF_LINES),
    }
