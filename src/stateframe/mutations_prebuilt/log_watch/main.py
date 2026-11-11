"""Count bounded log-tail patterns and classify snapshot rotation or truncation."""


MAX_TAIL_BYTES = 64 << 10
MAX_PATTERNS = 20
MAX_PATTERN_CHARS = 256


def _change(previous, current):
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return "stable"
    rotated = previous.get("identity") != current.get("identity")
    old_size = previous.get("size")
    new_size = current.get("size")
    old_start = previous.get("start")
    new_start = current.get("start")
    truncated = (isinstance(old_size, int) and isinstance(new_size, int) and
                 new_size < old_size)
    if not rotated and isinstance(old_start, int) and isinstance(new_start, int):
        truncated = truncated or new_start < old_start
    if rotated and truncated:
        return "rotated_truncated"
    if rotated:
        return "rotated"
    if truncated:
        return "truncated"
    return "stable"


def handle(input, ctx):
    input_id = input.get("input_id")
    patterns = input.get("patterns", [])
    if not isinstance(input_id, str) or input_id not in ctx.inputs:
        raise ValueError("input_id is required")
    if (not isinstance(patterns, list) or len(patterns) > MAX_PATTERNS or
            any(not isinstance(item, str) or not item or len(item) > MAX_PATTERN_CHARS
                for item in patterns)):
        raise ValueError("patterns must be a bounded list of strings")
    raw = ctx.input_bytes(input_id, limit=MAX_TAIL_BYTES)
    text = raw.decode("utf-8", errors="replace")
    counts = [{"pattern": pattern, "count": text.count(pattern)} for pattern in patterns]
    return {
        "tail_bytes": len(raw),
        "line_count": len(text.splitlines()),
        "snapshot_change": _change(input.get("previous"), input.get("current")),
        "counts": counts,
    }
