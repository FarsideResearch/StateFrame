"""Compare two explicitly discovered Python callable contracts from current source."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import tokenize
from collections import Counter
from collections.abc import Mapping
from pathlib import PurePosixPath

from kernel_request_diet import _current_source_segments, _read_page_source_segment
from kernel_request_source_window import _range_is_covered
from kernel_stateframe_action_ledger import ActionLedgerError, _read_open_site_file_bytes

HEADER = "[Repository API attribute rows]"
COLUMNS = ("API", "Signature inputs", "Quoted contract", "Declaration")
MAX_BLOCK_BYTES = 2048
_BLOCK = re.compile(r"\n\n" + re.escape(HEADER) + r'\n\{"columns": [^\n]*\}')
_QUALIFIED = re.compile(r"(?<![\w.])(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*(?!\w|\.\w)")
_DISCOVERY_TOOLS = {"read", "grep", "where_defined", "references", "who_calls"}


def _text(value):
    return value if isinstance(value, str) else ""


def _discovered_names(exchange):
    if not exchange:
        return set()
    assistant, *replies = exchange
    functions = [call.get("function") for call in assistant.get("tool_calls", ())]
    if not functions or any(not isinstance(function, Mapping)
                            or function.get("name") not in _DISCOVERY_TOOLS
                            for function in functions):
        return set()
    fields = [_text(assistant.get(key)) for key in ("content", "reasoning_content")]
    fields.extend(_text(function.get("arguments")) for function in functions)
    fields.extend(_text(reply.get("content")) for reply in replies)
    return {name for field in fields for name in _QUALIFIED.findall(field)}


def _completed_edit_text(actions, applied):
    """Recover literal edits that the typed transcript no longer describes."""
    edits = []
    for action in actions:
        if action.get("action_kind") != "edit" or action.get("result_kind") != "APPLIED":
            continue
        arguments = action.get("canonical_arguments")
        if not isinstance(arguments, Mapping):
            return None
        path = arguments.get("path", arguments.get("target_path"))
        if action.get("observation") == f"created {path}":
            edits.append({"path": path, "operation": "create", "new_text": arguments.get("content")})
        else:
            edits.append({"path": path, "old_text": arguments.get("old_str"),
                          "new_text": arguments.get("new_str")})
    return edits if Counter(row["path"] for row in edits) == applied else None


def _mentions_identifier(text, identifiers):
    """Inspect edit fragments, including a final expression with open delimiters."""
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            tokens.append(token)
    except tokenize.TokenError as error:
        # str_replace records fragments, not necessarily complete Python
        # statements. The tokenizer has yielded every lexical token before an
        # unmatched opening delimiter reaches EOF, so those names are usable.
        message = error.args[0] if error.args else ""
        if "EOF" not in message or "multi-line statement" not in message:
            return None
    except IndentationError:
        return None
    return any(token.type == tokenize.NAME and token.string in identifiers
               for token in tokens)


def _choice_is_open(frame, names, actions):
    """Earlier edits leave the choice open when their text rules out a reference."""
    ledger = frame.get("action_ledger")
    if not isinstance(ledger, Mapping):
        return False
    history = ledger.get("history_digest") or {}
    edits = ledger.get("edits", ())
    applied = Counter(row.get("file") for row in frame.get("applied_edits", ()))
    if ((history.get("kind_counts") or {}).get("edit")
            or applied - Counter(row.get("path") for row in edits)):
        edits = _completed_edit_text(actions, applied) if applied else None
        if edits is None:
            return False
    identifiers = {name.rsplit(".", 1)[-1] for name in names}
    for edit in edits:
        old = "" if edit.get("operation") == "create" else edit.get("old_text")
        new = edit.get("new_text")
        if not isinstance(old, str) or not isinstance(new, str):
            return False
        for text in (old, new):
            mentions = _mentions_identifier(text, identifiers)
            if mentions is None or mentions:
                return False
    return True


_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _declarations(body, prefix, rebound):
    """Find direct declarations and flag competing bindings in their namespaces."""
    from kernel_parallel_admission import _scope_bindings

    definitions = [node for node in body if isinstance(node, _DEFINITIONS)]
    other = ast.Module(body=[node for node in body if not isinstance(node, _DEFINITIONS)],
                       type_ignores=[])
    bound, declared = _scope_bindings(other)
    counts = Counter(node.name for node in definitions)
    for node in definitions:
        name = f"{prefix}.{node.name}"
        stable = (counts[node.name] == 1 and node.name not in bound | declared
                  and "*" not in bound and name not in rebound)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield name, node, stable
        elif stable and not node.decorator_list:
            yield from _declarations(node.body, name, rebound)


def _rebound_exports(tree, module):
    """Catch explicit writes to qualified attributes and potentially writable globals."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            yield from (f"{module}.{name}" for name in node.names)
        elif isinstance(node, ast.NamedExpr):
            yield f"{module}.{node.target.id}"
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            target = ast.unparse(node)
            if _QUALIFIED.fullmatch(target):
                yield f"{module}.{target}"


def _inputs(lines, node):
    # Token positions are character columns. Parsing the declaration starting at
    # its physical line avoids mixing AST byte columns with token positions.
    declaration_lines = lines[node.lineno - 1:node.body[0].lineno]
    prefix = "".join(declaration_lines)
    depth = 0
    start = None
    for token in tokenize.generate_tokens(io.StringIO(prefix).readline):
        if token.type != tokenize.OP:
            continue
        if token.string == "(":
            if not depth:
                start = token.end
            depth += 1
        elif token.string == ")" and depth:
            depth -= 1
            if not depth:
                line, column = start
                end_line, end_column = token.start
                if line == end_line:
                    return declaration_lines[line - 1][column:end_column]
                chunks = declaration_lines[line - 1:end_line]
                chunks[0], chunks[-1] = chunks[0][column:], chunks[-1][:end_column]
                return "".join(chunks)
    return None


def _modules(path):
    parts = PurePosixPath(path).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if parts and all(part.isidentifier() for part in parts):
        yield ".".join(parts)
        if parts[0] == "src" and len(parts) > 1:
            yield ".".join(parts[1:])


def _current_pages(messages, exchange, runtime=None):
    """Find visible source pages from the latest read and retained source windows."""
    from kernel_host_authorship import host_spans, ledger_of, occurrences, row_kinds

    assistant, *replies = exchange
    read_ids = {call["id"] for call in assistant["tool_calls"]
                if call["function"].get("name") == "read"}
    for message in replies:
        if message.get("tool_call_id") in read_ids:
            page = _read_page_source_segment(message)
            if page is not None:
                yield page
    capture = getattr(runtime, "_active_capture", None)
    ledger = ledger_of(capture) or ledger_of(runtime)
    kinds = row_kinds(ledger, messages, capture=capture)
    repeated = occurrences(ledger, messages, capture=capture)
    source_kinds = {"plain_source", "model_window", "raw_window"}
    for index, message in enumerate(messages):
        # A retained source page can outlive the step-scoped authorship ledger
        # that first recorded it. Unknown rows still need the parsed source
        # window, delivered-range proof, and live-byte revision check below.
        if (not isinstance(message, Mapping) or message.get("role") != "user"
                or kinds[index] not in source_kinds | {None}):
            continue
        content = _text(message.get("content"))
        spans = host_spans(ledger, message, occurrence=repeated[index])
        for segment in _current_source_segments([message]):
            header = f"[Current source: {segment['path']}]\n"
            start = content.find(header)
            if start >= 0 and (kinds[index] is None or any(
                    left <= start and start + len(header) <= right for left, right in spans)):
                yield {**segment, "bytes": segment["text"].encode("utf-8")}


def comparison(messages, exchange, frame, root, delivered, *, actions=(), runtime=None):
    """Return the supported two-row comparison and host proof, if eligible."""
    names = _discovered_names(exchange)
    if len(names) < 2:
        return None
    pages = {}
    for page in _current_pages(messages, exchange, runtime):
        path = page["path"]
        if path.endswith(".py"):
            pages.setdefault(path, []).append(page)

    ranges = delivered(messages)
    matches = {}
    for path, carriers in pages.items():
        modules = tuple(_modules(path))
        if not any(name.startswith(module + ".") for name in names for module in modules):
            continue
        try:
            raw = _read_open_site_file_bytes(root, path)
            source = raw.decode("utf-8")
            tree = ast.parse(source)
            declarations = [
                declaration for module in modules
                for declaration in _declarations(tree.body, module, set(_rebound_exports(tree, module)))
            ]
        except (ActionLedgerError, SyntaxError, ValueError, RecursionError):
            return None
        revision = "sha256:" + hashlib.sha256(raw).hexdigest()
        lines = raw.splitlines(keepends=True)
        source_lines = source.splitlines(keepends=True)
        # Python and the source pager need to agree on declaration coordinates.
        if len(lines) != len(source_lines):
            return None
        spans = ranges.get((path, revision), ())
        verified = [(page["start_line"], page["end_line"]) for page in carriers
                    if page["revision"] == revision and page["bytes"] == b"".join(
                        lines[page["start_line"] - 1:page["end_line"]])]
        for name, node, stable in declarations:
            if name not in names:
                continue
            if (not stable or name in matches or node.decorator_list
                    or getattr(node, "type_params", ()) or not ast.get_docstring(node)):
                return None
            clause = node.body[0]
            first, last = node.lineno, clause.end_lineno
            if not (_range_is_covered(spans, first, last)
                    and _range_is_covered(verified, first, last)):
                return None
            try:
                inputs = _inputs(source_lines, node)
            except (tokenize.TokenError, IndentationError):
                return None
            quoted = ast.get_source_segment(source, clause)
            if inputs is None or not quoted:
                return None
            location = {"path": path, "revision": revision, "start_line": first,
                        "end_line": last}
            matches[name] = (inputs, quoted, location)
    if len(matches) != 2 or len({(row[2]["path"], row[2]["start_line"])
                                 for row in matches.values()}) != 2:
        return None
    if not _choice_is_open(frame, matches, actions):
        return None

    rows = [[name, *matches[name]] for name in sorted(matches)]
    block = HEADER + "\n" + json.dumps({"columns": COLUMNS, "rows": rows}, ensure_ascii=True)
    if len(block.encode()) > MAX_BLOCK_BYTES:
        return None
    receipt = {
        "layout": "api_attribute_rows",
        "declarations": [{"api": row[0], **row[3]} for row in rows],
        "block_sha256": "sha256:" + hashlib.sha256(block.encode()).hexdigest(),
    }
    return block, receipt


def strip(payload):
    """Remove previous assembler rows before evaluating the current request."""
    if isinstance(payload, str):
        # In the payload the first block starts directly after the tail marker.
        return _BLOCK.sub("", "\n\n" + payload).removeprefix("\n\n") or None
    if isinstance(payload, list):
        return [part for part in payload if not _BLOCK.fullmatch(_part_text(part))]
    return payload


def blocks(content):
    parts = content if isinstance(content, list) else [content]
    return [match.group().removeprefix("\n\n")
            for part in parts
            for match in _BLOCK.finditer(_part_text(part))]


def _part_text(part):
    return _text(part.get("text") if isinstance(part, Mapping) else part)


def fits(messages, tools, runtime, allocation, sources):
    """Require spare budget before pricing an optional tail alongside source."""
    from kernel_context import token_estimate
    from kernel_request_budget import FREE_STAGE, _lean_request_limits

    capture = getattr(runtime, "_active_capture", None)
    capacity_retry = getattr(capture, "_capacity_retry_limits", None)
    surface = getattr(capture, "final_tool_surface", None)
    if isinstance(surface, Mapping):
        tools = surface.get("tools", tools)
    body = {"messages": messages, "tools": tools}
    byte_limit, token_limit = _lean_request_limits(body, FREE_STAGE)
    if capacity_retry is not None:
        byte_limit = min(byte_limit, capacity_retry.max_request_bytes)
        token_limit = min(token_limit, capacity_retry.max_prompt_tokens)
    if allocation is not None:
        byte_limit = min(byte_limit, allocation.capacity["byte_ceiling"])
        token_limit = min(token_limit, allocation.capacity["token_ceiling"])
    ceiling = getattr(runtime, "frame_ceiling", token_limit)
    # Account conservatively for source the fitter may attach after assembly.
    # Current snapshots may overlap the visible pages; reserve that room too.
    reserve = MAX_BLOCK_BYTES
    if sources:
        reserve += sum(len(json.dumps(path)) + len(json.dumps(source))
                       for path, source in sources.items())
    working_set = getattr(runtime, "working_set_state", None)
    if working_set is not None:
        for obligation in working_set.active_obligations():
            identity = obligation.identity
            if identity.byte_start is None or identity.byte_end is None:
                return False
            reserve += 6 * (identity.byte_end - identity.byte_start)
    size = len(json.dumps(body).encode()) + reserve
    return (size <= byte_limit
            and token_estimate(body) + reserve <= min(token_limit, ceiling))
