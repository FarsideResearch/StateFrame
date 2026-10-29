"""Render live, executable ticket-literal sites at the request boundary."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kernel_request_evidence_closure import ticket_identity_for_runtime
import kernel_request_source_window as source_window

MODE_ENV = "STATEFRAME_REMAINING_SITES"
GREEN_NAG_MODE_ENV = "STATEFRAME_GREEN_NAG_SITES"
_PANEL_PREFIX = "[host] Ticket sites still containing "
_ASSIGNMENT_VALUE = (
    r"(?:true|false|True|False|null|None|undefined|[-+]?\d+(?:\.\d+)?|"
    r"0[xX][0-9A-Fa-f]+|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[^`]*`)"
)
_NUMERIC_VERSION = re.compile(r"\d+(?:\.\d+)+")
_CHANGE_VERBS = re.compile(
    r"\b(?P<verb>remove|delete|drop|eliminate|strip|disable|change|replace|modify|update|set)\b",
    re.IGNORECASE,
)
_NEGATED_CHANGE = re.compile(
    r"\b(?:do\s+not|don't|never|must\s+not|should\s+not)\s+"
    r"(?:remove|delete|drop|eliminate|strip|disable|change|replace|modify|update|set)\b",
    re.IGNORECASE,
)
_RETAIN_BEFORE = re.compile(
    r"\b(?:keep|retain|preserve|leave)\b(?:\s+[\w./:=`'\"-]+){0,8}\s*$",
    re.IGNORECASE,
)
_RETAIN_AFTER = re.compile(
    r"^\s*(?:(?:must|should|shall|will)\s+)?(?:remain|stay|be\s+(?:kept|retained|preserved|left)|"
    r"unchanged|as[- ]is)\b",
    re.IGNORECASE,
)
_CHANGE_AFTER = re.compile(
    r"^\s*(?:(?:must|should|shall|needs?\s+to|has\s+to|is\s+to)\s+)?"
    r"(?:be\s+)?(?:removed|deleted|dropped|eliminated|changed|replaced|modified|updated|disabled|"
    r"turned\s+off|set\s+to\s+(?:false|off)|not\s+(?:present|used|enabled))\b",
    re.IGNORECASE,
)
_NEGATED_CHANGE_AFTER = re.compile(
    r"^\s*(?:(?:must|should|shall|will)\s+)?not\s+(?:be\s+)?"
    r"(?:removed|deleted|dropped|eliminated|changed|replaced|modified|updated|disabled|touched|altered)\b",
    re.IGNORECASE,
)
_MUST_NOT_AFTER = re.compile(r"^\s*(?:must|should|shall|may|can)\s+not\b", re.IGNORECASE)
_NOT_USE_BEFORE = re.compile(
    r"\b(?:(?:must|should|shall|do)\s+not\s+use|avoid)\b"
    r"(?:\s+[\w./:=`'\"-]+){0,8}\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True, slots=True)
class TicketSite:
    path: str
    line: int


@dataclass(frozen=True, slots=True)
class GreenNagDecision:
    message: str | None
    fired: bool
    remaining: int | None
    inline: bool
    record: bool


def enabled(environ: Mapping[str, object] | None = None) -> bool:
    """Enable the optional panel only by explicit opt-in."""
    values = os.environ if environ is None else environ
    return str(values.get(MODE_ENV, "off")).strip().lower() == "on"


def green_nag_enabled(environ: Mapping[str, object] | None = None) -> bool:
    """Enable site-aware post-green edit feedback only by explicit opt-in."""
    values = os.environ if environ is None else environ
    return str(values.get(GREEN_NAG_MODE_ENV, "off")).strip().lower() == "on"


def _parent_sources(runtime: object, paths: Sequence[str]) -> dict[str, str]:
    owner = getattr(runtime, "owner", None)
    row = getattr(owner, "task_row", {})
    files = row.get("files") if isinstance(row, Mapping) else None
    if not isinstance(files, Mapping):
        return {}
    result: dict[str, str] = {}
    for path in paths:
        payload = files.get(path)
        if isinstance(payload, Mapping) and isinstance(payload.get("parent_src"), str):
            result[path] = payload["parent_src"]
    return result


def _source_matches(path: str, source: str, pattern: re.Pattern[str]) -> list[str]:
    lines = source.splitlines()
    docstring_lines = (
        source_window._python_docstring_lines(source)
        if path.casefold().endswith(".py") else frozenset()
    )
    matches: list[str] = []
    for number, line in enumerate(lines, 1):
        if (pattern.search(line)
                and source_window._site_role(path, source, number, docstring_lines) == "executable"):
            matches.append(line)
    return matches


def _parent_matches(
        parents: Mapping[str, str], paths: Sequence[str], pattern: re.Pattern[str],
) -> dict[str, list[str]]:
    return {
        path: _source_matches(path, parents[path], pattern)
        for path in paths if path in parents
    }


def _assignment_literals(line: str, token: str) -> set[str]:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_$])(?P<literal>{re.escape(token)}\s*(?::=|=|:)\s*"
        rf"{_ASSIGNMENT_VALUE})(?![A-Za-z0-9_$])",
        re.IGNORECASE,
    )
    return {match.group("literal") for match in pattern.finditer(line)}


def _ticket_requires_change(ticket: str, literal: str) -> bool:
    """Require local, unambiguous ticket language that changes this literal."""
    if not literal:
        return False
    literal_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(literal)}(?![A-Za-z0-9_])", re.IGNORECASE,
    )
    clauses = re.split(r"[\r\n;]+|(?<=[.!?])\s+", ticket)
    changed = False
    retained = False
    for clause in clauses:
        for match in literal_pattern.finditer(clause):
            before = clause[max(0, match.start() - 120):match.start()]
            after = clause[match.end():match.end() + 120]
            if (_RETAIN_BEFORE.search(before)
                    or _RETAIN_AFTER.search(after)
                    or _NEGATED_CHANGE_AFTER.search(after)):
                retained = True
                continue

            if _NEGATED_CHANGE.search(before):
                retained = True
                continue

            prefix_directive = False
            directives = list(_CHANGE_VERBS.finditer(before))
            for directive in reversed(directives):
                between = before[directive.end():]
                if len(between.split()) > 10:
                    continue
                verb = directive.group("verb").casefold()
                if verb in {"change", "replace", "modify", "update", "set"} and re.search(
                        r"\b(?:with|to|as)\b", between, re.IGNORECASE):
                    continue
                prefix_directive = True
                break

            after_words = re.findall(r"[a-z]+", after.casefold())
            disabled_index = next(
                (index for index, word in enumerate(after_words[:6]) if word in {"disabled", "off", "false"}),
                None,
            )
            disabled_directive = (
                disabled_index is not None
                and not set(after_words[:disabled_index]) & {"not", "never", "cannot", "can't", "shouldn't"}
                and "enabled" not in after_words[:disabled_index]
            )
            not_use_directive = bool(_NOT_USE_BEFORE.search(before))
            must_not_directive = bool(_MUST_NOT_AFTER.search(after))
            suffix_directive = bool(_CHANGE_AFTER.search(after)) or disabled_directive or not_use_directive
            if must_not_directive and not _NEGATED_CHANGE_AFTER.search(after):
                suffix_directive = True
            changed = changed or prefix_directive or suffix_directive

    return changed and not retained


def _pattern_for_token(
        token: str,
        original: str,
        parents: Mapping[str, str],
        paths: Sequence[str],
) -> str | None:
    """Choose an exact ticket/source literal; never emit a normalized query."""
    if not original or not all(character.isprintable() for character in original):
        return None
    exact_pattern = source_window._literal_pattern(original)
    exact_parent_sites = _parent_matches(parents, paths, exact_pattern)
    if any(exact_parent_sites.values()):
        return original

    # Dotted numeric versions must stay exact; a normalized token cannot
    # authorize a search for a different source string.
    if original != token and _NUMERIC_VERSION.search(original):
        return None

    lookup = source_window._literal_pattern(token, original)
    parent_sites = _parent_matches(parents, paths, lookup)
    expressions = {
        expression
        for lines in parent_sites.values()
        for line in lines
        for expression in _assignment_literals(line, token)
    }
    if len(expressions) == 1:
        return next(iter(expressions))

    # A non-normalized identifier is itself an exact ticket literal.
    if token == original and any(parent_sites.values()):
        return token
    return None


def _ticket_patterns(
        runtime: object,
        parents: Mapping[str, str],
        paths: Sequence[str],
) -> tuple[str, ...]:
    identity = ticket_identity_for_runtime(runtime)
    if identity is None:
        return ()
    entry = source_window.entry_symbol(runtime)
    tokens = source_window.literal_tokens(identity.text, exclude=(entry,) if entry else ())
    origins = source_window._token_origins(identity.text)
    quoted = {
        match.group(2).strip().casefold()
        for match in source_window._QUOTED.finditer(identity.text)
        if match.group(2).strip()
    }
    result: list[str] = []
    for token in tokens:
        original = origins.get(token, token)
        if (original.casefold() not in quoted
                and not re.search(r"[0-9./:=_-]", original)):
            continue
        if not _ticket_requires_change(identity.text, original):
            continue
        pattern = _pattern_for_token(token, original, parents, paths)
        if pattern and pattern not in result:
            result.append(pattern)
    # Prefer complete assignment literals to their nested identifiers.
    return tuple(
        pattern for pattern in result
        if not (
            re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", pattern)
            and any(
                len(other) > len(pattern)
                and re.search(rf"(?<![A-Za-z0-9_$]){re.escape(pattern)}(?![A-Za-z0-9_$])", other)
                for other in result
            )
        )
    )


def remaining_site_inventory(runtime: object) -> list[dict[str, object]]:
    """Return deterministic ticket-literal sites from declared candidate files only."""
    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", None)
    if root_value is None:
        raise ValueError("remaining-site workspace root is unavailable")
    root = Path(root_value).resolve(strict=True)
    paths = tuple(
        path for path in source_window.editable_product_paths(runtime)
        if all(character.isprintable() for character in path)
    )
    if not paths:
        return []
    parents = _parent_sources(runtime, paths)
    patterns = _ticket_patterns(runtime, parents, paths)
    if not patterns:
        return []

    path_order = {path: index for index, path in enumerate(paths)}
    literal_order = {literal: index for index, literal in enumerate(patterns)}
    rows: list[tuple[int, int, int, dict[str, object]]] = []
    seen: set[tuple[str, int, str]] = set()
    for literal in patterns:
        sites = source_window.literal_site_inventory(root, paths, (literal,))
        # A broad literal is not allowed to consume the shared inventory budget.
        if len(sites) >= source_window._MAX_SITES:
            continue
        for site in sites:
            if site.role != "executable" or literal not in site.text:
                continue
            identity = (site.path, site.line, literal)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append((
                path_order.get(site.path, len(path_order)),
                site.line,
                literal_order[literal],
                {
                    "path": site.path,
                    "line": site.line,
                    "literal": literal,
                    "text": site.text,
                    "revision": "sha256:" + site.revision,
                },
            ))
    rows.sort(key=lambda row: row[:3])
    return [row[3] for row in rows]


def _read_source(root: Path, relative: str) -> str | None:
    try:
        path = root / relative
        if path.stat().st_size > 2_000_000:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None


def current_workspace_revision(runtime: object) -> str | None:
    """Return a short content revision for the current editable source snapshot."""
    try:
        owner = getattr(runtime, "owner", None)
        root_value = getattr(owner, "root", None)
        if root_value is None:
            return None
        root = Path(root_value).resolve()
        paths = tuple(
            path for path in source_window.editable_product_paths(runtime)
            if isinstance(path, str) and path and all(character.isprintable() for character in path)
        )
        if not paths:
            return None
        digest = hashlib.sha256()
        for relative in sorted(set(paths)):
            source = _read_source(root, relative)
            if source is None:
                return None
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(source.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:8]
    except Exception:
        return None


def inventory_ticket_sites(runtime: object) -> tuple[TicketSite, ...] | None:
    """Inventory exact executable ticket-literal sites at the current revision.

    ``None`` means the inventory is incomplete or unavailable; an empty tuple
    means every recognized ticket literal was checked and no live site remains.
    """
    try:
        owner = getattr(runtime, "owner", None)
        root_value = getattr(owner, "root", None)
        if root_value is None:
            return None
        root = Path(root_value).resolve()
        paths = tuple(
            path for path in source_window.editable_product_paths(runtime)
            if isinstance(path, str) and path and all(character.isprintable() for character in path)
        )
        if not paths:
            return None
        parents = _parent_sources(runtime, paths)
        if any(path not in parents for path in paths):
            return None
        patterns = _ticket_patterns(runtime, parents, paths)
        if not patterns:
            return None
        if any(_read_source(root, path) is None for path in paths):
            return None

        resolved_patterns = 0
        sites: set[TicketSite] = set()
        for literal in patterns:
            literal_sites = source_window.literal_site_inventory(root, paths, (literal,))
            if len(literal_sites) >= source_window._MAX_SITES:
                continue
            resolved_patterns += 1
            for site in literal_sites:
                if site.role == "executable" and literal in str(site.text):
                    sites.add(TicketSite(path=site.path, line=site.line))
        if resolved_patterns == 0:
            return None
        if len(sites) > source_window._MAX_SITES:
            return None
        return tuple(sorted(sites))
    except Exception:
        return None


def decide_green_nag(
        nudge: str | None,
        runtime: object,
        tested_revision: str | None,
        *,
        environ: Mapping[str, object] | None = None,
) -> GreenNagDecision:
    """Keep the legacy notice when off; otherwise gate it on a complete inventory."""
    if not green_nag_enabled(environ):
        return GreenNagDecision(nudge, False, None, False, False)
    if not isinstance(nudge, str) or not nudge:
        return GreenNagDecision(None, False, None, True, False)

    sites = inventory_ticket_sites(runtime)
    if sites is None:
        return GreenNagDecision(None, False, None, True, True)
    remaining = len(sites)
    if remaining == 0:
        return GreenNagDecision(nudge, True, 0, True, True)

    revision = str(tested_revision or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{8}", revision) is None:
        return GreenNagDecision(None, False, remaining, True, True)
    locations = ", ".join(f"{site.path}:{site.line}" for site in sites)
    noun, verb = ("site", "remains") if remaining == 1 else ("sites", "remain")
    message = (
        f"[host] public tests PASS at rev {revision}; {remaining} ticket {noun} {verb}: {locations}"
    )
    return GreenNagDecision(message, False, remaining, True, True)


def record_green_nag_decision(bolt: object, decision: GreenNagDecision) -> None:
    """Record the bounded fired/count decision on the existing KernelBolt trace."""
    if not decision.record:
        return
    recorder = getattr(bolt, "_rec", None)
    if not callable(recorder):
        return
    detail = json.dumps(
        {"fired": decision.fired, "remaining": decision.remaining},
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        recorder("gate", "green_nag_decision", detail)
    except Exception:
        pass


def _changed_descriptions(
        parents: Mapping[str, str],
        paths: Sequence[str],
        parent_counts: Mapping[str, int],
        current_counts: Mapping[str, int],
        current_sources: Mapping[str, str | None],
) -> str:
    order = {path: index for index, path in enumerate(paths)}
    changed: list[str] = []
    for path in paths:
        parent = parents.get(path)
        current = current_sources.get(path)
        if parent is None or current is None or parent == current:
            continue

        parent_sites = parent_counts.get(path, 0)
        current_sites = current_counts.get(path, 0)
        # A newly added shared-policy file can implement the fix without
        # containing any of the ticket literals that remain at other sites.
        if parent_sites > current_sites or (parent == "" and current != ""):
            changed.append(path)
    changed.sort(key=lambda path: (current_counts.get(path, 0) > 0, order[path]))
    descriptions: list[str] = []
    for path in changed:
        if (current_counts.get(path, 0) > 0
                and parent_counts.get(path, 0) > current_counts.get(path, 0)):
            descriptions.append(f"the other {Path(path).name} site(s)")
        else:
            descriptions.append(path)
    if not descriptions:
        return "none"
    if len(descriptions) == 1:
        return descriptions[0]
    return ", ".join(descriptions[:-1]) + " and " + descriptions[-1]


def render_remaining_sites(runtime: object) -> str:
    """Render current executable literal sites; return empty on uncertainty/error."""
    try:
        owner = getattr(runtime, "owner", None)
        root_value = getattr(owner, "root", None)
        if root_value is None:
            return ""
        root = Path(root_value).resolve()
        paths = tuple(
            path for path in source_window.editable_product_paths(runtime)
            if all(character.isprintable() for character in path)
        )
        if not paths:
            return ""
        parents = _parent_sources(runtime, paths)
        patterns = _ticket_patterns(runtime, parents, paths)
        if not patterns:
            return ""

        # Inventory each literal separately and keep only sites whose line holds the literal
        # verbatim. On b1433 c10 the fuzzy "HTTP/1.1" pattern matched 39 unrelated lines,
        # hit the shared site cap and blanked the panel, http2=True included. A noisy literal now drops
        # only itself.
        grouped: dict[str, list[source_window.Site]] = {}
        for literal in patterns:
            literal_sites = source_window.literal_site_inventory(root, paths, (literal,))
            if len(literal_sites) >= source_window._MAX_SITES:
                continue
            grouped[literal] = [
                site for site in literal_sites
                if site.role == "executable" and literal in str(site.text)
            ]
        patterns = tuple(literal for literal in patterns if literal in grouped)

        current_sources = {path: _read_source(root, path) for path in parents}
        candidate_paths = {site.path for sites in grouped.values() for site in sites}
        changed_lines_by_path: dict[str, frozenset[int]] = {}
        for path in candidate_paths:
            parent = parents.get(path)
            current = current_sources.get(path)
            if parent is not None and current is not None:
                changed_lines_by_path[path] = source_window.changed_lines(parent, current)
        grouped = {
            literal: [
                site for site in sites
                if site.path in changed_lines_by_path
                and site.line not in changed_lines_by_path[site.path]
            ]
            for literal, sites in grouped.items()
        }
        panels: list[str] = []
        for literal in patterns:
            sites = grouped[literal]
            if not sites:
                continue

            parent_pattern = source_window._literal_pattern(literal)
            parent_site_lines = _parent_matches(parents, paths, parent_pattern)
            parent_counts = {path: len(lines) for path, lines in parent_site_lines.items()}
            current_counts: dict[str, int] = {}
            for site in sites:
                current_counts[site.path] = current_counts.get(site.path, 0) + 1
            locations = ", ".join(f"{site.path}:{site.line}" for site in sites)
            changed = _changed_descriptions(
                parents, paths, parent_counts, current_counts, current_sources)
            panels.append(
                f"\n[host] Ticket sites still containing {literal}: {locations}. "
                f"Sites already changed: {changed}."
            )
        return "".join(panels)
    except Exception:
        return ""


def _strip_panel_suffix(content: str) -> str:
    marker = "\n" + _PANEL_PREFIX
    start = content.rfind(marker)
    if start < 0:
        return content
    suffix_lines = content[start + 1:].splitlines()
    if suffix_lines and all(
            line.startswith(_PANEL_PREFIX)
            and " Sites already changed: " in line
            and line.endswith(".")
            for line in suffix_lines):
        return content[:start]
    return content


def maybe_append_remaining_sites(
        messages: object,
        runtime: object,
        *,
        environ: Mapping[str, object] | None = None,
) -> bool:
    """Replace the prior panel on the last tool result, failing open on all errors."""
    if not enabled(environ) or not isinstance(messages, list):
        return False
    try:
        matching = [
            index for index, message in enumerate(messages)
            if isinstance(message, Mapping) and message.get("role") == "tool"
        ]
        if not matching:
            return False
        message = messages[matching[-1]]
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            return False
        content = message["content"]
        without_old_panel = _strip_panel_suffix(content)
        panel = render_remaining_sites(runtime)
        updated = without_old_panel + panel
        if updated == content:
            return False
        message["content"] = updated
        return True
    except Exception:
        return False


__all__ = [
    "GREEN_NAG_MODE_ENV",
    "GreenNagDecision",
    "MODE_ENV",
    "TicketSite",
    "current_workspace_revision",
    "decide_green_nag",
    "enabled",
    "green_nag_enabled",
    "inventory_ticket_sites",
    "maybe_append_remaining_sites",
    "record_green_nag_decision",
    "remaining_site_inventory",
    "render_remaining_sites",
]
