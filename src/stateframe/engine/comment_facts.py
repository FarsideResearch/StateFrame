"""Comments-as-typed-facts with PRECOMPUTED AGE (FINDINGS.md E29 tail).

Comments enter the 1B's context as a TYPED fact channel, never as inline prose noise.
The staleness sensor is git-blame(comment) vs git-blame(the code beneath it), and the
age arithmetic is done HERE — date math is banned model work. Emitted facts read:

    NOTE (updated with this code, current): <comment text>
    NOTE (14 months older than the code it describes - possibly stale): <comment text>
    DOCSTRING (updated with this code, current): <first 200 chars>

Every failure mode (bad path, unparseable file, no git, detached blame) returns [].
This module never raises.
"""

from __future__ import annotations

import ast
import io
import os
import re
import subprocess
import tokenize
from typing import Dict, Iterable, List, Optional, Tuple

# Average days per month; the label is a human-scale approximation, not an SLA.
_DAYS_PER_MONTH = 30.44
_SECONDS_PER_MONTH = _DAYS_PER_MONTH * 24 * 3600

_CURRENT_LABEL = "updated with this code, current"
_BLAME_TIMEOUT_SECONDS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Brace scrubbing — brace literals in prose hang the 1B (standing rule)
# ─────────────────────────────────────────────────────────────────────────────

_INNERMOST_BRACES = re.compile(r"\{[^{}]*\}")


def strip_braces(text: str) -> str:
    """Replace brace literals with parens so no '{' or '}' ever reaches the model."""
    prev = None
    out = text
    # Collapse innermost-first so nested literals fully resolve.
    while prev != out:
        prev = out
        out = _INNERMOST_BRACES.sub(lambda m: "(" + m.group(0)[1:-1] + ")", out)
    return out.replace("{", "(").replace("}", ")")


# ─────────────────────────────────────────────────────────────────────────────
# git blame
# ─────────────────────────────────────────────────────────────────────────────

def _blame_committer_times(
    git_root: str, rel_path: str, lines: Iterable[int]
) -> Dict[int, int]:
    """Map 1-based line number -> committer epoch seconds. Empty dict on any failure.

    One `git blame` call carrying an `-L n,n` range per requested line: same data as
    N separate calls, a fraction of the process spawns.
    """
    wanted = sorted({n for n in lines if isinstance(n, int) and n > 0})
    if not wanted:
        return {}

    cmd = ["git", "-C", git_root, "blame", "--porcelain"]
    for n in wanted:
        cmd += ["-L", f"{n},{n}"]
    cmd += ["--", rel_path]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_BLAME_TIMEOUT_SECONDS,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}

    try:
        text = proc.stdout.decode("utf-8", errors="strict")
    except Exception:
        return {}

    # Porcelain emits `committer-time` only on a sha's FIRST appearance; later groups
    # for the same sha omit it, so carry a sha -> time table across the whole stream.
    times: Dict[int, int] = {}
    sha_time: Dict[str, int] = {}
    current_sha: Optional[str] = None
    current_line: Optional[int] = None
    header = re.compile(r"^([0-9a-f]{7,40}) \d+ (\d+)(?: \d+)?$")

    for raw in text.splitlines():
        m = header.match(raw)
        if m:
            current_sha = m.group(1)
            current_line = int(m.group(2))
            known = sha_time.get(current_sha)
            if known is not None:
                times[current_line] = known
            continue
        if raw.startswith("committer-time ") and current_sha is not None:
            try:
                ts = int(raw.split(" ", 1)[1].strip())
            except Exception:
                continue
            sha_time[current_sha] = ts
            if current_line is not None:
                times[current_line] = ts

    return times


# ─────────────────────────────────────────────────────────────────────────────
# Age labelling (precomputed — the model never does date math)
# ─────────────────────────────────────────────────────────────────────────────

def _age_label(comment_ts: Optional[int], code_ts: Optional[int], stale_months: int) -> Optional[str]:
    """Label describing how much older the comment is than the code below it."""
    if comment_ts is None or code_ts is None:
        return None
    delta = code_ts - comment_ts  # positive => comment predates the code
    if delta <= 0:
        return _CURRENT_LABEL
    months = int(delta // _SECONDS_PER_MONTH)
    if months < max(1, int(stale_months)):
        return _CURRENT_LABEL
    unit = "month" if months == 1 else "months"
    return f"{months} {unit} older than the code it describes - possibly stale"


# ─────────────────────────────────────────────────────────────────────────────
# Source scanning
# ─────────────────────────────────────────────────────────────────────────────

def _find_function(tree: ast.AST, fn_name: str):
    """First function/async-function node named fn_name, at any nesting depth."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
            return node
    return None


def _comment_lines(source: str) -> Dict[int, str]:
    """1-based line -> comment text, for every `#` comment token in the module."""
    out: Dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string.lstrip("#").strip()
    except Exception:
        # A tokenize error late in the file still leaves the comments found so far.
        pass
    return out


def _is_code_line(raw: str) -> bool:
    stripped = raw.strip()
    return bool(stripped) and not stripped.startswith("#")


def _first_code_line_after(lines: List[str], start: int, limit: int) -> Optional[int]:
    """First 1-based code line strictly after `start`, up to and including `limit`."""
    for n in range(start + 1, min(limit, len(lines)) + 1):
        if _is_code_line(lines[n - 1]):
            return n
    return None


def _function_span(node, lines: List[str], comments: Dict[int, str]) -> Tuple[int, int]:
    """(scan_start, scan_end) covering the function plus its immediately-above comments."""
    starts = [node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]
    top = min(starts)
    # Walk upward through a contiguous block of comment-only lines directly above.
    scan_start = top
    n = top - 1
    while n >= 1 and n in comments and not _is_code_line(_uncommented(lines[n - 1])):
        scan_start = n
        n -= 1
    end = getattr(node, "end_lineno", None) or top
    return scan_start, end


def _uncommented(raw: str) -> str:
    """The code portion of a physical line, for 'is this comment-only?' checks."""
    stripped = raw.strip()
    return "" if stripped.startswith("#") else stripped


def _docstring_line(node, lines: List[str]) -> Optional[int]:
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if not isinstance(first, ast.Expr):
        return None
    value = first.value
    if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
        return None
    return first.lineno


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_comment_facts(
    repo_root: str,
    rel_path: str,
    fn_name: str,
    stale_months: int = 6,
    git_root: Optional[str] = None,
) -> List[str]:
    """Typed comment facts for `fn_name` in `rel_path`, with age precomputed.

    `rel_path` is relative to `git_root` (which defaults to `repo_root`) for BOTH the
    filesystem read and the blame, so a nested checkout can pass the real git root.
    Returns [] on any parse, IO, or git failure — never raises.
    """
    try:
        root = git_root or repo_root
        abs_path = os.path.join(root, rel_path)
        with open(abs_path, "r", encoding="utf-8") as fh:
            source = fh.read()

        tree = ast.parse(source)
        node = _find_function(tree, fn_name)
        if node is None:
            return []

        lines = source.splitlines()
        comments = _comment_lines(source)
        scan_start, scan_end = _function_span(node, lines, comments)

        # Comment line -> the first code line it sits above (its subject).
        pairs: List[Tuple[int, int, str]] = []
        for cline in sorted(comments):
            if not (scan_start <= cline <= scan_end):
                continue
            code_line = _first_code_line_after(lines, cline, scan_end)
            if code_line is None:
                continue
            pairs.append((cline, code_line, comments[cline]))

        doc_line = _docstring_line(node, lines)
        doc_text = ast.get_docstring(node) if doc_line else None
        doc_code_line = None
        if doc_line is not None and doc_text:
            doc_end = getattr(node.body[0], "end_lineno", doc_line)
            doc_code_line = _first_code_line_after(lines, doc_end, scan_end)

        needed = {c for c, _, _ in pairs} | {k for _, k, _ in pairs}
        if doc_line is not None and doc_code_line is not None:
            needed |= {doc_line, doc_code_line}
        if not needed:
            return []

        times = _blame_committer_times(root, rel_path, needed)
        if not times:
            return []

        facts: List[str] = []

        if doc_line is not None and doc_code_line is not None and doc_text:
            label = _age_label(times.get(doc_line), times.get(doc_code_line), stale_months)
            if label:
                body = " ".join(doc_text.split())[:200]
                if body:
                    facts.append(f"DOCSTRING ({label}): {strip_braces(body)}")

        for cline, code_line, text in pairs:
            if not text:
                continue
            label = _age_label(times.get(cline), times.get(code_line), stale_months)
            if not label:
                continue
            facts.append(f"NOTE ({label}): {strip_braces(text)}")

        return facts
    except Exception:
        # Fail soft: a comment channel must never take the harness down.
        return []
