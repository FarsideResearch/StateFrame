"""Deterministic rendering for cross-run repair memory.

The renderer is deliberately independent from the repair runner.  It accepts the
serialised ``AttemptMemoryV1`` shape (or the dataclass itself), ranks only
evidence that is safe to show to the next agent, and emits a small prompt block.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence


MAX_ATTEMPTS = 8
MAX_THEORIES_PER_ATTEMPT = 6
MAX_FAILED_LEGS_PER_ATTEMPT = 2
MAX_VISIBLE_ATOMS = 5
MAX_FRONTIER_ROWS = 3
DEFAULT_MAX_TOKENS = 800


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        candidate = to_dict()
        if isinstance(candidate, Mapping):
            return candidate
    raise TypeError("attempt memory must be a mapping or expose to_dict()")


def _text(value: object, limit: int = 180) -> str:
    rendered = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    rendered = " ".join(rendered.split())
    if "UNKNOWN" in rendered.upper():
        rendered = rendered.upper().replace("UNKNOWN", "unavailable")
    return rendered[:limit]


def _sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _attempt_no(record: Mapping[str, object]) -> int:
    key = record.get("key")
    if isinstance(key, Mapping):
        value = key.get("attempt_no")
    else:
        value = record.get("attempt_no")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _tag(record: Mapping[str, object]) -> str:
    key = record.get("key")
    if isinstance(key, Mapping):
        return _text(key.get("tag"), 100)
    return _text(record.get("tag"), 100)


def _record_sort_key(record: Mapping[str, object]) -> tuple[int, str, str]:
    key = record.get("key")
    run_id = ""
    if isinstance(key, Mapping):
        run_id = _text(key.get("run_id"), 100)
    return (_attempt_no(record), _tag(record), run_id)


def _agent_world_only(record: Mapping[str, object]) -> bool:
    world = record.get("world")
    if not isinstance(world, Mapping):
        return False
    if _text(world.get("provenance"), 60).upper() != "AGENT_WORLD_ONLY":
        return False
    audit = record.get("audit")
    if isinstance(audit, Mapping):
        for key in ("fixed_source_seen", "grader_imported"):
            value = audit.get(key)
            if value is True or _text(value, 20).lower() in {"true", "yes", "1"}:
                return False
    return True


def _theory_sort_key(theory: Mapping[str, object]) -> tuple[str, str]:
    return (_text(theory.get("claim"), 160).lower(), _text(theory.get("fp"), 80))


def _failed_sort_key(leg: Mapping[str, object]) -> tuple[int, str, str]:
    try:
        turn = int(leg.get("turn", 0))
    except (TypeError, ValueError):
        turn = 0
    return (turn, _text(leg.get("test_selector"), 180), _text(leg.get("fp"), 80))


def _atom_rank(atom: Mapping[str, object]) -> tuple[int, int, str, str]:
    state = _text(atom.get("state"), 40).upper()
    basis = _text(atom.get("basis"), 60).upper()
    state_rank = {"CONTRADICTED": 0, "SUPPORTED": 1}.get(state, 9)
    basis_rank = 0 if basis in {"TWO_SIDED", "TEST_AST", "TARGET_AST"} else 1
    return (state_rank, basis_rank, _text(atom.get("kind"), 80), _text(atom.get("fp"), 80))


def _frontier_sort_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (_text(row.get("key"), 160), _text(row.get("fp"), 80))


def _full_ast_fingerprints(record: Mapping[str, object]) -> tuple[str, ...]:
    values: list[str] = []
    for edit in _sequence(record.get("edits")):
        capture = _text(edit.get("capture"), 40).upper()
        fp = _text(edit.get("ast_delta_fp"), 100)
        if fp and capture != "LEGACY_CLIPPED" and fp not in values:
            values.append(fp)
    return tuple(values)


def _render_theories(record: Mapping[str, object]) -> str:
    theories = sorted(_sequence(record.get("theories")), key=_theory_sort_key)
    claims = [_text(item.get("claim"), 100) for item in theories if _text(item.get("claim"), 100)]
    if claims:
        return "/".join(claims[:MAX_THEORIES_PER_ATTEMPT])

    edits = _sequence(record.get("edits"))
    paths = sorted({_text(item.get("path"), 100) for item in edits if _text(item.get("path"), 100)})
    if paths:
        return "edits in " + ", ".join(paths[:3])
    terminal = record.get("terminal")
    if isinstance(terminal, Mapping) and (terminal.get("reads") or terminal.get("greps")):
        return "read/grep-only investigation"
    return "no recorded product edit"


def _render_failed_legs(record: Mapping[str, object]) -> str:
    legs = sorted(_sequence(record.get("failed_legs")), key=_failed_sort_key)
    rendered: list[str] = []
    for leg in legs[:MAX_FAILED_LEGS_PER_ATTEMPT]:
        selector = _text(leg.get("test_selector"), 120) or "test"
        anchor = _text(leg.get("anchor"), 160)
        rendered.append(f"{selector} RED" + (f" [{anchor}]" if anchor else ""))
    return "; ".join(rendered)


def _terminal_line(record: Mapping[str, object]) -> str:
    terminal = record.get("terminal")
    if not isinstance(terminal, Mapping):
        return ""
    shape = _text(terminal.get("shape"), 60).upper()
    if shape == "PARTIAL":
        return "terminal: PARTIAL receipt-only; no terminal invented"
    if shape in {"BUDGET_EXHAUSTED", "FAIL_FAST_NO_ACTION"}:
        if (terminal.get("reads") or terminal.get("greps")) and not (
            terminal.get("edits") or terminal.get("tests")
        ):
            return f"terminal: {shape.lower().replace('_', ' ')}; stop read/grep repetition"
        return f"terminal: {shape.lower().replace('_', ' ')}"
    if shape:
        return f"terminal: {shape.lower().replace('_', ' ')}"
    return ""


def _visible_atoms(records: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    candidates: list[Mapping[str, object]] = []
    for record in records:
        for atom in _sequence(record.get("atoms")):
            state = _text(atom.get("state"), 40).upper()
            expected = _text(atom.get("expected"), 180)
            observed = _text(atom.get("observed"), 180)
            if state not in {"SUPPORTED", "CONTRADICTED"}:
                continue
            if "unavailable" in (expected + observed).lower():
                continue
            candidates.append(atom)
    deduped: dict[str, Mapping[str, object]] = {}
    for atom in sorted(candidates, key=_atom_rank):
        fp = _text(atom.get("fp"), 100) or _text(atom.get("kind"), 80)
        deduped.setdefault(fp, atom)
    return tuple(deduped.values())[:MAX_VISIBLE_ATOMS]


def _visible_frontier(records: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    candidates: list[Mapping[str, object]] = []
    for record in records:
        for row in _sequence(record.get("frontier")):
            if _text(row.get("status"), 60).upper() == "PROVEN_UNEXPLORED":
                candidates.append(row)
    deduped: dict[str, Mapping[str, object]] = {}
    for row in sorted(candidates, key=_frontier_sort_key):
        key = _text(row.get("key"), 160) or _text(row.get("fp"), 80)
        deduped.setdefault(key, row)
    return tuple(deduped.values())[:MAX_FRONTIER_ROWS]


def _count_tokens(value: str) -> int:
    return len(value.split())


def _clip_line(value: str, tokens: int) -> str:
    if tokens <= 0:
        return ""
    words = value.split()
    if len(words) <= tokens:
        return value
    if tokens == 1:
        return words[0]
    return " ".join(words[: tokens - 1]) + " …"


def _bound_lines(header: str, body: Sequence[str], footer: str, max_tokens: int) -> str:
    budget = max(1, int(max_tokens))
    fixed = _count_tokens(header) + _count_tokens(footer)
    selected: list[str] = []
    remaining = budget - fixed
    for line in body:
        if remaining <= 0:
            break
        line_tokens = _count_tokens(line)
        if line_tokens <= remaining:
            selected.append(line)
            remaining -= line_tokens
            continue
        clipped = _clip_line(line, remaining)
        if clipped:
            selected.append(clipped)
        remaining = 0
        break
    result_lines = [header, *selected, footer]
    result = "\n".join(line for line in result_lines if line)
    if _count_tokens(result) <= budget:
        return result
    return _clip_line(result, budget)


def render_capsule(
    attempts: Iterable[object],
    *,
    current_attempt: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_chars: int | None = None,
) -> str:
    """Render the bounded ``PRIOR ATTEMPTS`` block in stable order.

    ``UNKNOWN`` evidence, receipt-clipped edit identities, and unproven frontier
    rows are intentionally absent from the rendered prompt.
    """

    records = [_mapping(item) for item in attempts]
    records = [record for record in records if _agent_world_only(record)]
    records = [record for record in records if _attempt_no(record) > 0]
    records.sort(key=_record_sort_key)
    if current_attempt is not None:
        records = [record for record in records if _attempt_no(record) < current_attempt]
    records = records[:MAX_ATTEMPTS]

    header = "PRIOR ATTEMPTS: agent-world only"
    body: list[str] = []
    for record in records:
        attempt_no = _attempt_no(record)
        tried = _render_theories(record)
        line = f"attempt {attempt_no} tried {tried}"
        failed = _render_failed_legs(record)
        if failed:
            line += f" (failed leg: {failed})"
        terminal = _terminal_line(record)
        if terminal:
            line += f" ({terminal})"
        body.append(line + ".")

        fingerprints = _full_ast_fingerprints(record)
        if fingerprints:
            body.append("do not re-try AST delta " + ", ".join(fingerprints[:3]) + " unchanged.")
        elif _sequence(record.get("edits")):
            body.append("receipt-only edit identities are clipped; no exact repeat blocker.")

    atoms = _visible_atoms(records)
    if atoms:
        body.append("known atoms:")
        for atom in atoms:
            kind = _text(atom.get("kind"), 70) or "contract"
            expected = _text(atom.get("expected"), 140)
            observed = _text(atom.get("observed"), 140)
            state = _text(atom.get("state"), 40).lower()
            body.append(f"- {kind}: expected {expected}; observed {observed} [{state}]")

    frontier = _visible_frontier(records)
    if frontier:
        body.append("unexplored:")
        for row in frontier:
            rendered = _text(row.get("render"), 220) or _text(row.get("key"), 180)
            body.append(f"- {rendered}")
        footer = ""
    else:
        footer = "unexplored: none_proven"

    if not records and not body:
        body.append("none")
    rendered = _bound_lines(header, body, footer, max_tokens)
    if max_chars is not None and len(rendered) > max_chars:
        rendered = rendered[: max(1, int(max_chars))].rstrip() + " …"
    return rendered


def render_prior_attempts(
    attempts: Iterable[object],
    *,
    current_attempt: int | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_chars: int | None = None,
) -> str:
    """Compatibility name for the prompt-weld renderer."""

    return render_capsule(
        attempts,
        current_attempt=current_attempt,
        max_tokens=max_tokens,
        max_chars=max_chars,
    )


def capsule_digest(capsule: str) -> str:
    """Return a content digest suitable for a prompt/trace receipt."""

    return hashlib.sha256(capsule.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MAX_VISIBLE_ATOMS",
    "capsule_digest",
    "render_capsule",
    "render_prior_attempts",
]
