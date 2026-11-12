"""Bounded file size, line, encoding, newline, BOM, and binary facts."""


MAX_READ = 4 << 20


def _encoding(data):
    boms = ((b"\x00\x00\xfe\xff", "utf-32-be"),
            (b"\xff\xfe\x00\x00", "utf-32-le"),
            (b"\xef\xbb\xbf", "utf-8"),
            (b"\xfe\xff", "utf-16-be"),
            (b"\xff\xfe", "utf-16-le"))
    for marker, name in boms:
        if data.startswith(marker):
            return name, marker, data[len(marker):]
    try:
        data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "unknown", b"", data
    return "utf-8", b"", data


def _newline(text):
    crlf = text.count("\r\n")
    remaining = text.replace("\r\n", "")
    lf = remaining.count("\n")
    cr = remaining.count("\r")
    found = [name for name, count in (("crlf", crlf), ("lf", lf), ("cr", cr)) if count]
    if len(found) > 1:
        return "mixed"
    return found[0] if found else "none"


def handle(input, ctx):
    input_id = input.get("input_id")
    if not isinstance(input_id, str) or input_id not in ctx.inputs:
        raise ValueError("input_id is required")
    path = ctx.inputs[input_id]
    data = ctx.input_bytes(input_id, limit=MAX_READ)
    size_bytes = path.stat().st_size
    encoding, marker, content = _encoding(data)
    binary = (encoding == "unknown" or
              (encoding == "utf-8" and b"\x00" in content))
    if not binary and content:
        controls = sum(byte < 32 and byte not in (9, 10, 12, 13) for byte in content)
        binary = controls / len(content) > 0.10
    text = ""
    if not binary:
        codec = encoding if encoding != "utf-8" else "utf-8"
        try:
            text = content.decode(codec, errors="strict")
        except (UnicodeDecodeError, LookupError):
            binary = True
            encoding = "unknown"
    line_count = len(text.splitlines()) if text else 0
    bom = encoding if marker else "none"
    return {
        "size_bytes": size_bytes,
        "line_count": line_count,
        "encoding": encoding,
        "newline": _newline(text),
        "bom": bom,
        "binary": binary,
    }
