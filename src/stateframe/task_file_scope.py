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
_GERUND_VERB = r"(?:editing|changing|modifying|writing|updating|touching)"
_GERUND_ONLY_RULE = re.compile(
    rf"(?:(?P<t1_context>^|(?<![\w./\\-])(?:by|while)\s+)(?P<t1_verb>{_GERUND_VERB})\s+only\b)|"
    rf"(?P<t2_context>^|(?<![\w./\\-])by\s+)only\s+(?P<t2_verb>{_GERUND_VERB})\b",
    re.IGNORECASE)
_GERUND_BLOCKED_CONTEXT = re.compile(
    r"(?<![\w./\\-])(?:(?:don|doesn|didn|isn|aren|wasn|weren|can|couldn|mayn|mightn|"
    r"mustn|shalln|shan|shouldn|willn|won|wouldn|haven|hasn|hadn|needn|daren|ain)[’']t|"
    r"cannot|"
    r"do|does|did|am|is|are|was|were|be|been|being|have|has|had|can|could|"
    r"may|might|must|shall|should|will|would|never|not|without)(?![\w./\\-])", re.IGNORECASE)
_WRITE_VERB = re.compile(
    r"(?<![\w./\\-])(?:write|creat(?:e|ing)|edit|change|modify|update|save|put|set|fix|add|touch|alter)"
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
_SENTENCE_END = re.compile(r"(?<!\bi\.e)(?<!\be\.g)([.;!?]+)(?=\s|\Z)", re.IGNORECASE)


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


def _targets(text: str, *, end_offset: list[int] | None = None) -> tuple[str, ...] | None:
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
            if end_offset is not None: end_offset[:] = [token.end()]
            return None
        path = _path(token[0])
        if path:
            paths.append(path)
        elif paths or _unquote(token[0]).lower() not in _TARGET_PREFIX | _ABBREVIATIONS:
            break
        pos = token.end()
        if end_offset is not None: end_offset[:] = [pos]
    return tuple(dict.fromkeys(paths))


def _expand_gerund_rule(sentence: str, *, question: bool) -> str:
    """Present only the two closed gerund forms to the existing ONLY-rule parser."""
    if question: return sentence
    def normalize(match: re.Match[str]) -> str:
        t2 = match.group("t2_verb") is not None
        if _GERUND_BLOCKED_CONTEXT.search(sentence[:match.start()]): return match[0]
        tail = sentence[match.end():]
        connector = match.group("t1_context") or match.group("t2_context") or ""
        if not re.fullmatch(r"the\s+file\s+above", tail.strip(), re.IGNORECASE):
            end: list[int] = []
            _targets(tail, end_offset=end)
            remainder = tail[end[0] if end else 0:]
            if t2 and (suffix := re.match(r"\s+is\s+(?:allowed|permitted)\b", remainder, re.IGNORECASE)):
                remainder = remainder[suffix.end():]
            if remainder.strip() and not re.match(r"\s*[,;]", remainder):
                continuation = re.match(r"\s+(?:to|so|and)\s+", remainder, re.IGNORECASE)
                if not continuation or not _WRITE_VERB.match(remainder[continuation.end():]): return match[0]
        return connector + "Only edit"
    return _GERUND_ONLY_RULE.sub(normalize, sentence)


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
    pass ``strict=True`` to detect missing targets and degrade with a visible warning.
    """
    if not isinstance(task, str) or not task.strip():
        return None
    split = _SENTENCE_END.split(_mask_context(task))
    sentence_parts = [(split[i].strip(), "?" in split[i + 1] if i + 1 < len(split) else False)
                      for i in range(0, len(split), 2) if split[i].strip()]
    allowed: list[str] = []
    no_other = False
    for index, (sentence, question) in enumerate(sentence_parts):
        if _CONDITIONAL.search(sentence):
            if strict and (_ONLY_RULE.search(sentence) or _NO_OTHER_RULE.search(sentence) or _GERUND_ONLY_RULE.search(sentence)):
                raise TaskFileScopeError(
                    "task file-scope rule is conditional and cannot be safely scoped; "
                    "use an unconditional repository-relative path")
            continue
        no_other = no_other or bool(_NO_OTHER_RULE.search(sentence))
        sentence = _expand_gerund_rule(sentence, question=question)
        affirmative_starts = {verb.start() for verb in _affirmative_verbs(sentence)}
        for rule in _ONLY_RULE.finditer(sentence):
            verb = _WRITE_VERB.search(rule[0])
            if verb is None or rule.start() + verb.start() not in affirmative_starts:
                continue
            tail = sentence[rule.end():].strip()
            if re.fullmatch(r"the\s+file\s+above", tail, re.IGNORECASE):
                targets = _previous_paths(sentence_parts[index - 1][0]) if index else None
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
        for sentence, _ in sentence_parts:
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
