"""Read an explicit file-scope rule out of a task's text.

A task may say "Change no other file" or "Only edit src/app.py". When it does, the run may
change only the files the task names. This module returns those named paths so the row
builder can scope edits with the existing edit-scope machinery. Pure and stdlib only.

No rule returns None. A recognized rule with an unsupported or ambiguous target is rejected so it
cannot silently become an unrestricted run.
"""
from __future__ import annotations

import re

_NO_OTHER_RULE = re.compile(
    r"\b(?:(?:change|edit|modify|touch|alter)\s+no\s+other\s+files?"
    r"|(?:do\s+not|don't|never)\s+(?:change|edit|modify|touch|alter)"
    r"\s+(?:any\s+)?other\s+files?)\b", re.IGNORECASE)
_ONLY_RULE = re.compile(
    r"\b(?:only\s+(?:edit|change|modify|touch|write)"
    r"|(?:edit|change|modify)\s+only)\b", re.IGNORECASE)
_WRITE_VERB = re.compile(
    r"(?<![\w./\\-])(?:write|create|edit|change|modify|update|save|put|set|fix|add|touch|alter)"
    r"(?![\w./\\-])", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|never|not|without)\s+"
    r"(?:(?:only|ever|also|\w+ly)\s+)*$", re.IGNORECASE)
_COORDINATED_VERB = re.compile(r"\b(?:and|or)\s+(?:also\s+)?$", re.IGNORECASE)
_CONDITIONAL = re.compile(
    r"(?<![\w./\\-])(?:if|unless|when|provided)(?![\w./\\-])", re.IGNORECASE)
_URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s<>\"'`]+")
_QUOTED = re.compile(
    r"```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`"
    r'|"(?:\\.|[^"\\])*"'
    r"|(?<!\w)'(?:\\.|[^'\\])*'(?!\w)")
# Consume whole tokens, including unsupported absolute paths: never recover a basename.
_TOKEN = re.compile(r'''`[^`\n]*`|"[^"\n]*"|'[^'\n]*'|[^\s,;!?`"'()<>{}\[\]]+''')
_RELATIVE_PATH = re.compile(r"[\w.@+-]+(?:/[\w.@+-]+)*")
_DOTTED_NAME = re.compile(r"[\w@+-]+(?:\.[\w@+-]+)*\.[A-Za-z][A-Za-z0-9_-]*")
_VERSION = re.compile(r"v\d+(?:\.[\w-]+)+", re.IGNORECASE)
_WELL_KNOWN = {"Dockerfile", "Makefile", "LICENSE", "README"}
_ABBREVIATIONS = {"i.e", "e.g"}
# These words may introduce a target, but cannot connect two target paths.
_TARGET_PREFIX = {
    "a", "an", "the", "your", "final", "answer", "output", "result", "results",
    "implementation", "file", "files", "document", "code", "changes", "new", "existing",
    "to", "into", "in", "at", "named", "called", "only",
}
_LIST_SEPARATOR = re.compile(r"\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)", re.IGNORECASE)
# Abbreviation dots are not sentence boundaries; path dots have no following whitespace.
_SENTENCE_END = re.compile(r"(?<!\bi\.e)(?<!\be\.g)[.;!?]+(?=\s|\Z)", re.IGNORECASE)


class TaskFileScopeError(ValueError):
    """A recognized file-scope rule has no safe, repository-relative target."""


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] in "`\"'" and token[-1] == token[0]:
        token = token[1:-1]
    return token.rstrip(".:").replace("\\", "/")


def _unsupported(token: str) -> bool:
    path = _unquote(token)
    return (path.startswith(("/", "~")) or bool(re.match(r"^[A-Za-z]:", path))
            or ".." in path.split("/"))


def _path(token: str) -> str | None:
    path = _unquote(token)
    if _unsupported(token):
        return None
    while path.startswith("./"):
        path = path[2:]
    if not _RELATIVE_PATH.fullmatch(path):
        return None
    if path.lower() in _ABBREVIATIONS or _VERSION.fullmatch(path):
        return None
    if "/" in path or path in _WELL_KNOWN or _DOTTED_NAME.fullmatch(path):
        return path
    return None


def _mask_context(task: str) -> str:
    """Keep quoted paths, but hide documentary prose and complete URL spans."""
    # Preserve terminal sentence punctuation even when the URL regex consumed it.
    task = _URL.sub(lambda m: " " * len(m[0].rstrip(".;!?"))
                    + m[0][len(m[0].rstrip(".;!?")):], task)
    def mask_quote(match: re.Match[str]) -> str:
        span = match[0]
        if (not span.startswith(("```", "~~~"))
                and not _ONLY_RULE.search(span) and not _NO_OTHER_RULE.search(span)
                and (_path(span) or _unsupported(span))):
            return span
        return " " * len(span)

    return _QUOTED.sub(mask_quote, task)


def _targets(text: str) -> tuple[str, ...] | None:
    """Read a verb's target list; None means an unsupported target, () no target."""
    paths: list[str] = []
    pos = 0
    while pos < len(text):
        if paths:
            separator = _LIST_SEPARATOR.match(text, pos)
            if not separator:
                break
            pos = separator.end()
        else:
            while pos < len(text) and (text[pos].isspace() or text[pos] == ","):
                pos += 1
        token = _TOKEN.match(text, pos)
        if not token:
            break
        if _unsupported(token[0]):
            return None
        path = _path(token[0])
        if path:
            paths.append(path)
        elif paths or _unquote(token[0]).lower() not in _TARGET_PREFIX | _ABBREVIATIONS:
            break
        pos = token.end()
    return tuple(dict.fromkeys(paths))


def _previous_paths(sentence: str) -> tuple[str, ...] | None:
    paths: list[str] = []
    for token in _TOKEN.finditer(sentence):
        if _unsupported(token[0]):
            return None
        path = _path(token[0])
        if path:
            paths.append(path)
    unique = tuple(dict.fromkeys(paths))
    return unique if len(unique) == 1 else None


def _affirmative_verbs(sentence: str) -> list[re.Match[str]]:
    """Negation also governs coordinated verbs: 'do not edit X or update Y'."""
    affirmative: list[re.Match[str]] = []
    end = 0
    negated = False
    for verb in _WRITE_VERB.finditer(sentence):
        prefix = sentence[end:verb.start()]
        negated = bool(_NEGATION.search(prefix)) or (
            negated and bool(_COORDINATED_VERB.search(prefix)))
        end = verb.end()
        if not negated:
            affirmative.append(verb)
    return affirmative


def parse_task_file_scope(task: str, *, strict: bool = False) -> tuple[str, ...] | None:
    """Return the repository paths a task's file-scope rule allows, or None.

    Only unconditional rules count. Targets must be governed by an affirmative write
    verb; examples, qualifiers and negated clauses cannot grant write permission. A recognized
    rule with an unsupported or ambiguous target raises ``TaskFileScopeError``. Row builders
    pass ``strict=True`` so a bare "change no other file" rule cannot leave the run unscoped.
    """
    if not isinstance(task, str) or not task.strip():
        return None
    sentences = [s.strip() for s in _SENTENCE_END.split(_mask_context(task)) if s.strip()]
    allowed: list[str] = []
    no_other = False
    for index, sentence in enumerate(sentences):
        if _CONDITIONAL.search(sentence):
            continue
        no_other = no_other or bool(_NO_OTHER_RULE.search(sentence))
        affirmative_starts = {verb.start() for verb in _affirmative_verbs(sentence)}
        for rule in _ONLY_RULE.finditer(sentence):
            verb = _WRITE_VERB.search(rule[0])
            if verb is None or rule.start() + verb.start() not in affirmative_starts:
                continue
            tail = sentence[rule.end():].strip()
            if re.fullmatch(r"the\s+file\s+above", tail, re.IGNORECASE):
                targets = _previous_paths(sentences[index - 1]) if index else None
            else:
                targets = _targets(tail)
            if targets is None:
                raise TaskFileScopeError(
                    "task file-scope rule has an unsupported or ambiguous target; "
                    "use repository-relative paths")
            if not targets:
                raise TaskFileScopeError(
                    "task file-scope rule does not name a usable path; "
                    "name repository-relative file paths")
            allowed.extend(targets)
    if no_other:
        for sentence in sentences:
            for verb in _affirmative_verbs(sentence):
                targets = _targets(sentence[verb.end():])
                if targets is None:
                    raise TaskFileScopeError(
                        "task file-scope rule has an unsupported or ambiguous target; "
                        "use repository-relative paths")
                allowed.extend(targets)
    result = tuple(dict.fromkeys(allowed))
    if strict and no_other and not result:
        raise TaskFileScopeError(
            "task file-scope rule does not name an allowed file; "
            "name at least one repository-relative file path")
    return result or None
