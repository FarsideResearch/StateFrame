"""Extract complete DSML tool invocations from reply text."""

import json

_MARKER = "｜DSML｜"


def _attributes(content: str, start: int, end: int) -> list[tuple[str, str, bool]]:
    items: list[tuple[str, str, bool]] = []
    pos = start
    while pos < end:
        while pos < end and content[pos].isspace():
            pos += 1
        if pos >= end or content[pos] == "/": break
        key_start = pos
        while pos < end and not content[pos].isspace() and content[pos] not in "=/":
            pos += 1
        if pos == key_start: pos += 1; continue
        key = content[key_start:pos]
        while pos < end and content[pos].isspace():
            pos += 1
        value, double_quoted = "", False
        if pos < end and content[pos] == "=":
            pos += 1
            while pos < end and content[pos].isspace():
                pos += 1
            if pos < end and content[pos] in "\"'":
                quote = content[pos]
                double_quoted = quote == '"'
                pos += 1
                value_start = pos
                while pos < end and content[pos] != quote:
                    pos += 1
                value = content[value_start:pos]
                if pos < end:
                    pos += 1
            else:
                value_start = pos
                while pos < end and not content[pos].isspace():
                    pos += 1
                value = content[value_start:pos]
        items.append((key, value, double_quoted))
    return items

def _next_tag(content: str, start: int):
    size, pos = len(content), start
    while pos < size:
        marker_at = content.find(_MARKER, pos)
        if marker_at < 0:
            return None
        if marker_at > 0 and content[marker_at - 1] == "<":
            tag_start, closing = marker_at - 1, False
        elif marker_at > 1 and content[marker_at - 2:marker_at] == "</":
            tag_start, closing = marker_at - 2, True
        else:
            pos = marker_at + len(_MARKER)
            continue
        name_start = marker_at + len(_MARKER)
        name_end = name_start
        while name_end < size and not content[name_end].isspace() and content[name_end] not in "/>":
            name_end += 1
        element = content[name_start:name_end].casefold()
        quote, cursor = "", name_end
        while cursor < size:
            char = content[cursor]
            if char == "<" and content.startswith(_MARKER, cursor + 1):
                # A later DSML opener recovers an unfinished header, including an open quote.
                pos = cursor
                break
            if quote:
                if char == quote:
                    quote = ""
            elif char in "\"'":
                quote = char
            elif char == ">":
                attrs = _attributes(content, name_end, cursor) if not closing else []
                exact_close = closing and name_end == cursor
                return (tag_start, cursor + 1, element, closing, exact_close, attrs)
            cursor += 1
        else:
            return None
        if cursor >= size:
            return None
    return None

def _first_name(attrs: list[tuple[str, str, bool]]) -> str | None:
    # Requiring name first is the simplest rule consistent with the corpus.
    if attrs and attrs[0][0] == "name" and attrs[0][2] and attrs[0][1]: return attrs[0][1]
    return None

def _parameter_value(raw: str, attrs: list[tuple[str, str, bool]]) -> object:
    mode = "true"
    for key, value, quoted in attrs[1:]:
        if key == "string" and quoted:
            mode = value
            break
    if mode == "false":
        try:
            return json.loads(raw)
        except (ValueError, RecursionError):
            pass
    return raw

def parse_dsml(content: str) -> list[dict]:
    """Return complete invocations in encounter order; omit partial calls."""
    if not isinstance(content, str) or not content: return []
    results: list[dict] = []
    active, current_name = False, None
    arguments: dict[str, object] = {}
    parameter: tuple[str | None, list[tuple[str, str, bool]], int] | None = None
    pos = 0
    while True:
        tag = _next_tag(content, pos)
        if tag is None: break
        tag_start, tag_end, element, closing, exact_close, attrs = tag
        pos = tag_end
        if not active:
            if not closing and element == "invoke":
                name = _first_name(attrs)
                if name is not None:
                    active, current_name = True, name
                    arguments, parameter = {}, None
            continue
        if closing and exact_close and element == "invoke":  # First close ends this flat invocation.
            if current_name:
                results.append({"id": f"dsml_{len(results)}", "function": {
                    "name": current_name, "arguments": json.dumps(arguments)}})
            active, current_name = False, None
            continue
        if parameter is not None:
            param_name, param_attrs, value_start = parameter
            if closing and exact_close and element == "parameter":
                if param_name is not None:
                    arguments[param_name] = _parameter_value(
                        content[value_start:tag_start], param_attrs
                    )
                parameter = None
            continue
        if not closing and element == "parameter": parameter = (_first_name(attrs), attrs, tag_end)
    return results
