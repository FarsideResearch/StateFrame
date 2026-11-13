"""Relevance selection for the ``relevant`` rung of the view ladder (B6).

Picks which lines of an already delivered source range survive when the packer
degrades the read to a FOCUS view.  Pure: the module computes plain
``(start, end, kind, edge)`` tuples over the held text and never touches the
diet, the finalizer or the dispatch loop; the only renderer dependency is the
range-marker format, so the bytes a selection promises are the bytes rendered.  Runtime state is read
only through the narrow accessors at the bottom of this module; every accessor
degrades to "no seeds of this kind" when its source is absent, so a renamed or
missing engine seam never breaks selection.

Seed priority (lower number wins; within a priority, ascending line):

    0  model      ranges the model itself asked to keep (``keep``/``read(view=)``)
    1  requested  closure-plan units for the path and the range the read asked for
    2  frame/test failing trace frames, pytest ``file.py:NN:`` locations, failing test defs
    3  changed    definitions touched by edits applied in this run
    4  symbol/lit ticket symbols and literals matched inside the held range
    5  call       one-hop neighbours (callers/callees found syntactically), <= 8 per seed

The selection is filled against a byte budget in that order, merged when two
ranges are within ``MERGE_GAP`` omitted lines of each other, clipped to the held
range, and then frozen per artifact so the rendered view never shimmers across
requests (cache-prefix stability).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass

from kernel_view_ladder.render import range_marker

RELEVANCE_VERSION = 1

# Lines of context kept around a frame/test line (mirrors source_window's
# ``_DEFAULT_CONTEXT_LINES``, symmetric: the line, 12 above, 12 below).
CONTEXT_LINES = 12
# Context kept around a call site.
CALL_CONTEXT_LINES = 2
# Two ranges merge when at most this many lines would be omitted between them.
MERGE_GAP = 2
# One-hop neighbours admitted per seed.
MAX_NEIGHBOURS = 8
# Upper bound on seeds collected from any one source, so a noisy receipt cannot
# make selection quadratic.
MAX_SEEDS_PER_SOURCE = 64

PRIORITY_MODEL = 0
PRIORITY_REQUESTED = 1
PRIORITY_FAILURE = 2
PRIORITY_CHANGED = 3
PRIORITY_IDENTIFIER = 4
PRIORITY_NEIGHBOUR = 5

EDGE_OBSERVED = "observed"      # from a trace, receipt or runner output
EDGE_RESOLVED = "resolved"      # a name matched a definition inside the held range
EDGE_UNRESOLVED = "unresolved"  # a call site whose target is not in the held range
EDGES = (EDGE_OBSERVED, EDGE_RESOLVED, EDGE_UNRESOLVED)

KINDS = ("model", "requested", "unit", "frame", "test", "changed", "symbol", "lit", "call")

# ``(?<![\w.])<name>\s*\(`` from spec 1.5; ``name`` is escaped per call.
_CALL_SITE = r"(?<![\w.]){name}\s*\("
_PY_DEF = re.compile(r"^(?P<indent>[ \t]*)(?:async\s+)?(?:def|class)\s+(?P<name>[A-Za-z_]\w*)")
_GENERIC_DEF = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+)?(?:default\s+)?(?:pub(?:\(crate\))?\s+)?(?:async\s+)?"
    r"(?:static\s+)?(?:def|class|function|fn|func|struct|impl|interface|type|enum|trait|module)"
    r"\b\s*(?P<name>[A-Za-z_$][\w$]*)")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

Selection = tuple[tuple[int, int, str, str], ...]


@dataclass(frozen=True)
class Seed:
    """One candidate range inside the held range, before budget fill."""

    kind: str
    start: int
    end: int
    priority: int
    edge: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown seed kind {self.kind!r}")
        if self.edge not in EDGES:
            raise ValueError(f"unknown edge label {self.edge!r}")
        if self.start > self.end:
            raise ValueError(f"seed range {self.start}-{self.end} is inverted")


@dataclass(frozen=True)
class Definition:
    """A definition span inside the held range (absolute line numbers)."""

    name: str
    start: int
    end: int


# --------------------------------------------------------------------------- outline


def outline(held_text: str, held_start: int, *, language: str = "python") -> tuple[Definition, ...]:
    """Definition spans of the held text, in absolute line numbers.

    Python uses ``ast`` (with a ``textwrap.dedent`` retry when the slice does
    not parse on its own); anything else, or Python that still fails to parse,
    uses the generic line scanner whose block ends at the next definition with
    an indent no deeper than its own.
    """
    lines = held_text.splitlines()
    if language == "python":
        rows = _python_outline(held_text) or _python_outline(textwrap.dedent(held_text))
        if rows is not None:
            return tuple(Definition(name, start + held_start - 1, end + held_start - 1)
                         for name, start, end in rows)
    return tuple(_scan_outline(lines, held_start))


def _python_outline(text: str) -> list[tuple[str, int, int]] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    rows: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            decorators = getattr(node, "decorator_list", ())
            for decorator in decorators:
                start = min(start, int(getattr(decorator, "lineno", start) or start))
            if start > 0:
                rows.append((node.name, start, end))
    rows.sort(key=lambda row: (row[1], row[2], row[0]))
    return rows


def _scan_outline(lines: Sequence[str], held_start: int) -> list[Definition]:
    heads: list[tuple[int, int, str]] = []  # (index, indent, name)
    for index, line in enumerate(lines):
        match = _PY_DEF.match(line) or _GENERIC_DEF.match(line)
        if match and match.group("name"):
            indent = len(match.group("indent").expandtabs(4))
            heads.append((index, indent, match.group("name")))
    rows: list[Definition] = []
    for index, indent, name in heads:
        rows.append(Definition(name, index + held_start, _block_end(lines, index, indent) + held_start))
    return rows


def _block_end(lines: Sequence[str], index: int, indent: int) -> int:
    """Index of the last line of the block opened at ``index`` (spec 1.6 scanner rule).

    With braces on the head line the block ends where the brace balance returns
    to zero; otherwise at the line before the next non-blank, non-comment line
    whose indent is no deeper than the head's.  Trailing blank lines are trimmed.
    """
    head = lines[index]
    if "{" in head:
        depth = 0
        for later in range(index, len(lines)):
            depth += lines[later].count("{") - lines[later].count("}")
            if depth <= 0 and later > index or (depth <= 0 and "}" in lines[later]):
                return later
    end = len(lines) - 1
    for later in range(index + 1, len(lines)):
        text = lines[later]
        stripped = text.strip()
        if not stripped or stripped.startswith(("#", "//", "*", "/*")):
            continue
        if len(text[:len(text) - len(text.lstrip())].expandtabs(4)) <= indent:
            end = later - 1
            break
    while end > index and not lines[end].strip():
        end -= 1
    return end


# --------------------------------------------------------------------------- seeds


def seeds_from_runtime(
        runtime: object, *, path: str, held_text: str, held_start: int, held_end: int,
        language: str = "python",
        override_ranges: Iterable[tuple[int, int]] = (),
        requested_ranges: Iterable[tuple[int, int]] = (),
        definitions: Sequence[Definition] | None = None,
) -> tuple[Seed, ...]:
    """Collect every seed for ``path`` over the held range, in priority order.

    ``override_ranges`` are ranges the model chose itself (top priority, kind
    ``model``); ``requested_ranges`` are what the read asked for
    (``start_line``/``end_line``).  Everything else is read from the runtime
    through the accessors; a missing source contributes nothing.
    """
    defs = tuple(definitions) if definitions is not None else outline(
        held_text, held_start, language=language)
    lines = held_text.splitlines()
    seeds: list[Seed] = []
    seeds.extend(_explicit_seeds(override_ranges, "model", PRIORITY_MODEL, held_start, held_end))
    seeds.extend(_explicit_seeds(requested_ranges, "requested", PRIORITY_REQUESTED,
                                 held_start, held_end))
    seeds.extend(_unit_seeds(closure_units(runtime, path), held_start, held_end))
    failure_text = failure_text_for(runtime)
    seeds.extend(_frame_seeds(evidence_lines(runtime, path, failure_text), held_start, held_end))
    seeds.extend(_test_seeds(failing_tests(runtime, path, failure_text), defs, held_start, held_end))
    seeds.extend(_changed_seeds(applied_edit_lines(runtime, path), defs, held_start, held_end))
    ticket = ticket_text(runtime)
    seeds.extend(_symbol_seeds(target_names(runtime, ticket), defs, lines, held_start, held_end))
    seeds.extend(_literal_seeds(ticket_literals(ticket), lines, held_start, held_end))
    return finish_seeds(seeds, defs=defs, lines=lines, held_start=held_start, held_end=held_end)


def finish_seeds(seeds: Iterable[Seed], *, defs: Sequence[Definition], lines: Sequence[str],
                 held_start: int, held_end: int) -> tuple[Seed, ...]:
    """Clip, add one-hop neighbours (priority 5) and sort deterministically."""
    base = _dedupe(_clip(seed, held_start, held_end) for seed in seeds)
    neighbours = _neighbour_seeds(base, defs, lines, held_start, held_end)
    return tuple(sorted(_dedupe(base + neighbours), key=_seed_order))


def _seed_order(seed: Seed) -> tuple[int, int, int, str, str]:
    return seed.priority, seed.start, seed.end, seed.kind, seed.label


def _explicit_seeds(ranges: Iterable[tuple[int, int]], kind: str, priority: int,
                    held_start: int, held_end: int) -> list[Seed]:
    out: list[Seed] = []
    for pair in ranges:
        try:
            start, end = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start > end or end < held_start or start > held_end:
            continue
        out.append(Seed(kind, start, end, priority, EDGE_RESOLVED, f"{kind}:{start}-{end}"))
    return out[:MAX_SEEDS_PER_SOURCE]


def _unit_seeds(units: Sequence[tuple[int, int, str]], held_start: int, held_end: int) -> list[Seed]:
    out: list[Seed] = []
    for start, end, role in units:
        if start > end or end < held_start or start > held_end:
            continue
        out.append(Seed("unit", start, end, PRIORITY_REQUESTED, EDGE_RESOLVED, f"unit:{role}"))
    return out[:MAX_SEEDS_PER_SOURCE]


def _frame_seeds(frame_lines: Sequence[tuple[int, str]], held_start: int, held_end: int) -> list[Seed]:
    out: list[Seed] = []
    for line, function in frame_lines:
        if not held_start <= line <= held_end:
            continue
        out.append(Seed("frame", line - CONTEXT_LINES, line + CONTEXT_LINES, PRIORITY_FAILURE,
                        EDGE_OBSERVED, f"frame:{line}" + (f":{function}" if function else "")))
    return out[:MAX_SEEDS_PER_SOURCE]


def _test_seeds(node_ids: Sequence[str], defs: Sequence[Definition],
                held_start: int, held_end: int) -> list[Seed]:
    by_name = _definitions_by_name(defs)
    out: list[Seed] = []
    for node_id in node_ids:
        # ``Class::test_x[param]`` -> ``test_x``
        name = node_id.rsplit("::", 1)[-1].split("[", 1)[0]
        for definition in by_name.get(name, ()):
            out.append(Seed("test", definition.start, definition.end, PRIORITY_FAILURE,
                            EDGE_OBSERVED, f"test:{node_id}"))
    return out[:MAX_SEEDS_PER_SOURCE]


def _changed_seeds(edit_lines: Sequence[int], defs: Sequence[Definition],
                   held_start: int, held_end: int) -> list[Seed]:
    out: list[Seed] = []
    for line in edit_lines:
        if not held_start <= line <= held_end:
            continue
        enclosing = _innermost(defs, line)
        if enclosing is None:
            out.append(Seed("changed", line, line, PRIORITY_CHANGED, EDGE_OBSERVED, f"changed:{line}"))
        else:
            out.append(Seed("changed", enclosing.start, enclosing.end, PRIORITY_CHANGED,
                            EDGE_RESOLVED, f"changed:{enclosing.name}"))
    return out[:MAX_SEEDS_PER_SOURCE]


def _symbol_seeds(names: Sequence[str], defs: Sequence[Definition], lines: Sequence[str],
                  held_start: int, held_end: int) -> list[Seed]:
    by_name = _definitions_by_name(defs)
    out: list[Seed] = []
    for name in names:
        short = name.rsplit(".", 1)[-1]
        matched = by_name.get(name) or by_name.get(short)
        if matched:
            for definition in matched:
                out.append(Seed("symbol", definition.start, definition.end, PRIORITY_IDENTIFIER,
                                EDGE_RESOLVED, f"symbol:{name}"))
            continue
        # Not defined in range: its call sites are unresolved neighbours.
        for line in call_sites(short, lines, held_start):
            out.append(Seed("call", line - CALL_CONTEXT_LINES, line + CALL_CONTEXT_LINES,
                            PRIORITY_NEIGHBOUR, EDGE_UNRESOLVED, f"call:{short}"))
    return out[:MAX_SEEDS_PER_SOURCE]


def _literal_seeds(tokens: Sequence[tuple[str, str]], lines: Sequence[str],
                   held_start: int, held_end: int) -> list[Seed]:
    """Line-only seeds for ticket literals (query spelling or original spelling)."""
    out: list[Seed] = []
    for token, original in tokens:
        if not token:
            continue
        alternatives = sorted({re.escape(token), re.escape(original or token)})
        pattern = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9_])",
                             re.IGNORECASE)
        for index, line in enumerate(lines):
            if pattern.search(line):
                absolute = held_start + index
                out.append(Seed("lit", absolute, absolute, PRIORITY_IDENTIFIER, EDGE_RESOLVED,
                                f"lit:{token}"))
    return out[:MAX_SEEDS_PER_SOURCE]


def _neighbour_seeds(seeds: Sequence[Seed], defs: Sequence[Definition], lines: Sequence[str],
                     held_start: int, held_end: int) -> list[Seed]:
    """Callers and callees of priority 1-4 definition seeds, <= 8 per seed."""
    by_name = _definitions_by_name(defs)
    out: list[Seed] = []
    for seed in seeds:
        if seed.priority >= PRIORITY_NEIGHBOUR or seed.kind in ("lit", "frame", "call"):
            continue
        owners = [d for d in defs if d.start == seed.start and d.end == seed.end]
        if not owners:
            continue
        found: list[Seed] = []
        for owner in owners:
            # callees: names called inside the owner span that are defined in range
            body = lines[owner.start - held_start: owner.end - held_start + 1]
            callee_names = sorted({name for line in body for name in _called_names(line)
                                   if name in by_name and name != owner.name})
            for name in callee_names:
                for definition in by_name[name]:
                    found.append(Seed("call", definition.start, definition.end, PRIORITY_NEIGHBOUR,
                                      EDGE_RESOLVED, f"call:{name}"))
            # callers: call sites of the owner's name outside its own span
            for line in call_sites(owner.name, lines, held_start):
                if owner.start <= line <= owner.end:
                    continue
                found.append(Seed("call", line - CALL_CONTEXT_LINES, line + CALL_CONTEXT_LINES,
                                  PRIORITY_NEIGHBOUR, EDGE_RESOLVED, f"call:{owner.name}"))
        found.sort(key=_seed_order)
        out.extend(_clip(item, held_start, held_end) for item in found[:MAX_NEIGHBOURS])
    return out


def call_sites(name: str, lines: Sequence[str], held_start: int) -> list[int]:
    """Absolute line numbers where ``name(`` is called (spec 1.5 call-site regex)."""
    if not name or not _IDENT.fullmatch(name):
        return []
    pattern = re.compile(_CALL_SITE.format(name=re.escape(name)))
    return [held_start + index for index, line in enumerate(lines) if pattern.search(line)]


def _called_names(line: str) -> list[str]:
    return re.findall(r"(?<![\w.])([A-Za-z_]\w*)\s*\(", line)


def _definitions_by_name(defs: Sequence[Definition]) -> dict[str, list[Definition]]:
    table: dict[str, list[Definition]] = {}
    for definition in defs:
        table.setdefault(definition.name, []).append(definition)
    return table


def _innermost(defs: Sequence[Definition], line: int) -> Definition | None:
    enclosing = [d for d in defs if d.start <= line <= d.end]
    if not enclosing:
        return None
    return min(enclosing, key=lambda d: (d.end - d.start, d.start))


def _clip(seed: Seed, held_start: int, held_end: int) -> Seed:
    start, end = max(held_start, seed.start), min(held_end, seed.end)
    if start == seed.start and end == seed.end:
        return seed
    return Seed(seed.kind, start, end, seed.priority, seed.edge, seed.label)


def _dedupe(seeds: Iterable[Seed]) -> list[Seed]:
    seen: set[Seed] = set()
    out: list[Seed] = []
    for seed in seeds:
        if seed.start > seed.end or seed in seen:
            continue
        seen.add(seed)
        out.append(seed)
    return out


# --------------------------------------------------------------------------- selection


def select(seeds: Iterable[Seed], *, held_text: str, held_start: int, held_end: int,
           budget_bytes: int) -> Selection | None:
    """Fill ``budget_bytes`` with seeds in priority order; ``None`` when nothing fits.

    The budget counts the UTF-8 bytes of every kept line plus one marker line
    per merged range, so the caller's deficit target is never exceeded.  A seed
    that does not fit is skipped and later, smaller seeds are still tried.
    Ranges within ``MERGE_GAP`` omitted lines merge (the gap lines are kept and
    counted).  Output is sorted, disjoint, clipped to the held range and
    independent of the input order of equal-priority seeds.
    """
    lines = held_text.splitlines(keepends=True)
    if len(lines) != held_end - held_start + 1:
        raise ValueError(f"held text has {len(lines)} lines but range is {held_start}-{held_end}")
    sizes = [len(line.encode("utf-8", "surrogatepass")) for line in lines]
    ordered = sorted(_dedupe(_clip(seed, held_start, held_end) for seed in seeds), key=_seed_order)
    chosen: list[Seed] = []
    merged: list[tuple[int, int, str, str]] = []
    for seed in ordered:
        trial = _merge(chosen + [seed])
        if _selection_bytes(trial, sizes, held_start, lines) <= budget_bytes:
            chosen.append(seed)
            merged = trial
    return tuple(merged) if merged else None


def _merge(seeds: Sequence[Seed]) -> list[tuple[int, int, str, str]]:
    """Merge seeds within MERGE_GAP; the highest-priority seed names the range."""
    by_line = sorted(seeds, key=lambda s: (s.start, s.end, s.priority, s.kind, s.label))
    groups: list[tuple[int, int, Seed]] = []  # (start, end, best seed)
    for seed in by_line:
        if groups and seed.start <= groups[-1][1] + MERGE_GAP + 1:
            start, end, best = groups[-1]
            if _seed_order(seed) < _seed_order(best):
                best = seed
            groups[-1] = (start, max(end, seed.end), best)
            continue
        groups.append((seed.start, seed.end, seed))
    return [(start, end, best.kind, best.edge) for start, end, best in groups]


def _selection_bytes(selection: Sequence[tuple[int, int, str, str]], sizes: Sequence[int],
                     held_start: int, lines: Sequence[str] = ()) -> int:
    """Wire bytes of the rendered rows: one marker row (+LF) per range plus the kept lines.

    The renderer terminates a range whose last kept line has no LF (a file's final line, or
    a lone-CR line) with one LF before the next marker, so that byte is counted here too.
    """
    total = 0
    for start, end, kind, edge in selection:
        total += len(range_marker(start, end, f"{kind}, {edge}").encode("utf-8")) + 1
        total += sum(sizes[start - held_start: end - held_start + 1])
        if lines and not lines[end - held_start].endswith("\n"):
            total += 1
    return total


def selection_bytes(selection: Selection, held_text: str, held_start: int) -> int:
    """Bytes a selection costs on the wire (kept lines plus marker lines)."""
    lines = held_text.splitlines(keepends=True)
    sizes = [len(line.encode("utf-8", "surrogatepass")) for line in lines]
    return _selection_bytes(selection, sizes, held_start, lines)


def omitted_ranges(selection: Selection, held_start: int, held_end: int) -> tuple[tuple[int, int], ...]:
    """Exact complement of the selection inside the held range."""
    out: list[tuple[int, int]] = []
    cursor = held_start
    for start, end, _kind, _edge in selection:
        if start > cursor:
            out.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor <= held_end:
        out.append((cursor, held_end))
    return tuple(out)


def selection_signature(selection: Selection | None) -> str:
    """Stable digest of a selection (part of ``vid``)."""
    payload = json.dumps({"v": RELEVANCE_VERSION, "ranges": list(selection or ())},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- frozen selection


def frozen_selection(state: MutableMapping[object, dict], artifact_key: object,
                     compute: Callable[[], Selection | None]) -> Selection | None:
    """Return the artifact's frozen selection, computing and freezing it once.

    ``state`` is ``runtime._view_ladder_state`` (or the pending copy); the
    entry's ``frozen_selection`` is written the first time a non-empty
    selection exists and never rewritten afterwards, so new anchors do not
    change an already rendered view.  A re-decision (fresh read, new keep) is a
    new artifact key, never a mutation of this entry.  ``None`` (rung not
    applicable) is not frozen: a later request with evidence may still select.
    """
    entry = state.setdefault(artifact_key, {})
    frozen = entry.get("frozen_selection")
    if frozen is not None:
        return _normalise(frozen)
    selection = compute()
    if selection:
        entry["frozen_selection"] = tuple(selection)
    return selection


def _normalise(value: object) -> Selection:
    rows: list[tuple[int, int, str, str]] = []
    for row in value if isinstance(value, (list, tuple)) else ():
        start, end, kind, edge = row
        rows.append((int(start), int(end), str(kind), str(edge)))
    return tuple(rows)


def relevant_selection(
        runtime: object, state: MutableMapping[object, dict], artifact_key: object, *,
        path: str, held_text: str, held_start: int, held_end: int, budget_bytes: int,
        language: str = "python",
        override_ranges: Iterable[tuple[int, int]] = (),
        requested_ranges: Iterable[tuple[int, int]] = (),
) -> Selection | None:
    """One-call entry point: collect seeds, fill the budget, freeze per artifact."""
    def compute() -> Selection | None:
        seeds = seeds_from_runtime(
            runtime, path=path, held_text=held_text, held_start=held_start, held_end=held_end,
            language=language, override_ranges=override_ranges, requested_ranges=requested_ranges)
        return select(seeds, held_text=held_text, held_start=held_start, held_end=held_end,
                      budget_bytes=budget_bytes)
    return frozen_selection(state, artifact_key, compute)


def fallback_selection(held_text: str, held_start: int, held_end: int, budget_bytes: int, *,
                       language: str = "python", head_lines: int = 12) -> Selection | None:
    """Sane selection when no seed exists: head of the range plus definition lines."""
    defs = outline(held_text, held_start, language=language)
    seeds = [Seed("requested", held_start, min(held_end, held_start + head_lines - 1),
                  PRIORITY_REQUESTED, EDGE_RESOLVED, "head")]
    seeds.extend(Seed("symbol", d.start, d.start, PRIORITY_IDENTIFIER, EDGE_RESOLVED, f"def:{d.name}")
                 for d in defs)
    return select(seeds, held_text=held_text, held_start=held_start, held_end=held_end,
                  budget_bytes=budget_bytes)


# --------------------------------------------------------------------------- runtime accessors
# Each accessor is the only place that knows an engine seam's name.  A missing
# or renamed seam yields an empty result (the seed is simply absent), never an
# exception out of selection.


def ticket_text(runtime: object) -> str:
    try:
        from kernel_request_source_window import _ticket_text
        text = _ticket_text(runtime)
    except Exception:  # noqa: BLE001 -- degraded seed, never a selection failure
        return ""
    return text if isinstance(text, str) else ""


def target_names(runtime: object, ticket: str) -> tuple[str, ...]:
    try:
        from kernel_request_source_window import _target_names
        names = _target_names(runtime, ticket)
    except Exception:  # noqa: BLE001
        return ()
    return tuple(str(n) for n in names if isinstance(n, str) and n)[:MAX_SEEDS_PER_SOURCE]


def ticket_literals(ticket: str) -> tuple[tuple[str, str], ...]:
    """``(query, original)`` literal pairs; ``original`` is the ticket spelling
    when the query was normalised (``HTTP/2`` -> ``http2``), else the query."""
    if not ticket:
        return ()
    try:
        from kernel_request_source_window import _token_origins, literal_tokens
        tokens = literal_tokens(ticket)
        origins = _token_origins(ticket)
    except Exception:  # noqa: BLE001
        return ()
    out: list[tuple[str, str]] = []
    for token in tokens:
        if isinstance(token, str) and token:
            original = origins.get(token) if isinstance(origins, Mapping) else None
            out.append((token, original if isinstance(original, str) and original else token))
    return tuple(out)[:MAX_SEEDS_PER_SOURCE]


def closure_units(runtime: object, path: str) -> tuple[tuple[int, int, str], ...]:
    try:
        from kernel_request_evidence_closure import closure_plan_for_runtime, units_for_path
        units = units_for_path(closure_plan_for_runtime(runtime), path)
    except Exception:  # noqa: BLE001
        return ()
    out: list[tuple[int, int, str]] = []
    for unit in units:
        start, end = getattr(unit, "start_line", None), getattr(unit, "end_line", None)
        if type(start) is int and type(end) is int and start <= end:
            out.append((start, end, str(getattr(unit, "role", "") or "")))
    return tuple(out)


def failure_text_for(runtime: object) -> str:
    """Concatenated runner output visible to the engine for this run."""
    pieces: list[str] = []
    latest = getattr(runtime, "_latest_check_receipt", None)
    receipts = getattr(runtime, "host_check_receipts", None)
    rows: list[object] = [latest] if latest is not None else []
    if isinstance(receipts, list):
        rows.extend(receipts[-4:])
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for key in ("runner_tail", "assertion_context", "assertion_context_preview", "stdout", "stderr"):
            value = row.get(key)
            if isinstance(value, str) and value:
                pieces.append(value)
    try:
        from kernel_reread_result_truth import _current_failure
        failure = _current_failure(runtime)
    except Exception:  # noqa: BLE001
        failure = None
    if isinstance(failure, Mapping) and isinstance(failure.get("assertion"), str):
        pieces.append(failure["assertion"])
    return "\n".join(dict.fromkeys(pieces))


def evidence_lines(runtime: object, path: str, text: str) -> tuple[tuple[int, str], ...]:
    """``(line, function)`` pairs for ``path`` from tracebacks and pytest locations."""
    if not text:
        return ()
    try:
        from kernel_code_bookmarks import _evidence_locations
        rows = _evidence_locations(text, (path,))
    except Exception:  # noqa: BLE001
        rows = _traceback_only(text, path)
    out: list[tuple[int, str]] = []
    for row in rows:
        try:
            row_path, line, function = row
        except (TypeError, ValueError):
            continue
        if row_path == path and type(line) is int and line > 0:
            out.append((line, str(function or "")))
    return tuple(dict.fromkeys(out))[:MAX_SEEDS_PER_SOURCE]


def _traceback_only(text: str, path: str) -> list[tuple[str, int, str]]:
    try:
        from tracebacks import repo_index_from_paths, traceback_signal
        signal = traceback_signal(text, repo_index_from_paths((path,)))
    except Exception:  # noqa: BLE001
        return []
    rows: list[tuple[str, int, str]] = []
    for anchor in signal.get("anchors", ()) if isinstance(signal, Mapping) else ():
        if isinstance(anchor, Mapping) and anchor.get("repo_path") == path:
            line = anchor.get("line")
            if type(line) is int:
                rows.append((path, line, str(anchor.get("function") or "")))
    return rows


def failing_tests(runtime: object, path: str, text: str) -> tuple[str, ...]:
    if not text:
        return ()
    try:
        from stateframe.test_runners import failing_test_nodes
        nodes = failing_test_nodes((path,), text).get(path, [])
    except Exception:  # noqa: BLE001
        return ()
    return tuple(str(n) for n in nodes)[:MAX_SEEDS_PER_SOURCE]


def applied_edit_lines(runtime: object, path: str) -> tuple[int, ...]:
    """Start lines of edits applied to ``path`` in this run (framefit ledger)."""
    context = getattr(runtime, "_framefit_action_context", None)
    frame = context.get("framefit5") if isinstance(context, Mapping) else None
    edits = frame.get("applied_edits") if isinstance(frame, Mapping) else None
    out: list[int] = []
    for edit in edits if isinstance(edits, (list, tuple)) else ():
        if not isinstance(edit, Mapping):
            continue
        if edit.get("path", edit.get("file")) != path:
            continue
        lines = edit.get("lines")
        start = lines.get("start") if isinstance(lines, Mapping) else edit.get("start_line")
        if type(start) is int and start > 0:
            out.append(start)
    return tuple(sorted(set(out)))[:MAX_SEEDS_PER_SOURCE]
