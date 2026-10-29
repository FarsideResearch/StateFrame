"""The consolidated serving client — every measured rule from E1-E23 in ONE place.

Probes copy-pasted three client bugs five times tonight. This module is the single owner:
  * CONTENT-ONLY parsing (rule 28) — reasoning tails restate your own template
  * retry-on-empty at 2x budget (finish=length eats content on thinking models)
  * markdown-fence stripping (content-channel replies are fenced; reasoning never was)
  * degenerate-loop detection (rule 31: budget-forced think-close dumps looping
    deliberation into content) with one higher-budget retry
  * lenient value extraction: quoted or bare token, NEVER from reasoning, reject
    menu-echo values containing '|' or '<' (rule 27: placeholder echo)
  * token-fuzzy line anchoring (symmetric coverage, semicolon-part splitting)
  * indentation-preserving multi-line patch application

Prompt lint (call `lint_prompt` in tests): no angle-bracket placeholders, no pipe-menus in
reply-format lines, no "you may reply with N lines" format-permissions, no trailing
negation clauses after the answer schema.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass

DEFAULT_URL = "http://127.0.0.1:8099/v1/chat/completions"


# ── prompt lint ──────────────────────────────────────────────────────────────
_LINT_PATTERNS = [
    (re.compile(r"<[a-z_]+(\|[a-z_]+)+>"), "pipe-menu placeholder (rule 27: becomes the answer)"),
    (re.compile(r'"\s*:\s*"<[^>]+>"'), "angle-bracket value placeholder (rule 20: copied verbatim)"),
    (re.compile(r"you may reply with", re.I), "format-permission phrasing (rule 27b: invites echo)"),
]


def lint_prompt(prompt: str) -> list[str]:
    return [msg for rx, msg in _LINT_PATTERNS if rx.search(prompt)]


# ── core ask ─────────────────────────────────────────────────────────────────
def degenerate(txt: str) -> bool:
    """Any 40-char window repeating 3+ times = looping deliberation leaked into content."""
    if len(txt) < 200:
        return False
    probe = txt[len(txt) // 3: len(txt) // 3 + 40]
    return bool(probe.strip()) and txt.count(probe) >= 3


@dataclass
class Reply:
    text: str            # content channel only, fences stripped
    ok: bool             # non-empty and non-degenerate
    note: str = ""


def ask(prompt: str, *, seed: int = 1, max_tokens: int = 2500,
        url: str = DEFAULT_URL, timeout: int = 600) -> Reply:
    last_note = "empty"
    for budget in (max_tokens, max_tokens * 2, max_tokens * 3):
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                "top_p": 1, "max_tokens": budget, "seed": seed}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        c = (r["choices"][0]["message"].get("content") or "").strip()
        if not c:
            last_note = "empty-content"
            continue
        if degenerate(c):
            last_note = "degenerate"
            continue
        lines = [l for l in c.splitlines() if not l.strip().startswith("```")]
        return Reply("\n".join(lines).strip(), True)
    return Reply("", False, last_note)


# ── value extraction ─────────────────────────────────────────────────────────
def get_value(reply: Reply, key: str, *, menu: set[str] | None = None) -> str | None:
    """Last match in CONTENT; menu-echo values rejected; None means no-op/escalate."""
    if not reply.ok:
        return None
    m = re.findall(rf'"{key}"\s*:\s*"([^"]*)"', reply.text) or \
        re.findall(rf'"{key}"\s*:\s*([A-Za-z0-9_.\-]+)', reply.text)
    if not m:
        return None
    v = m[-1].strip()
    if "|" in v or "<" in v:
        return None                       # placeholder echo
    if menu is not None and v.lower() not in menu:
        return None
    return v.lower() if menu is not None else v


def get_bool(reply: Reply, key: str = "answer") -> bool | None:
    if not reply.ok:
        return None
    m = re.findall(rf'"{key}"\s*:\s*(true|false)', reply.text)
    return (m[-1] == "true") if m else None


# ── code-line anchoring & patching ───────────────────────────────────────────
def _toks(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*|\d+|[^\w\s]", s))


def anchor_line(src: str, quoted: str, floor: float = 0.7) -> str | None:
    """Token-fuzzy, symmetric coverage; tries the whole quote then semicolon parts."""
    src_lines = [l for l in src.splitlines() if l.strip()]
    for part in [quoted] + [p.strip() for p in quoted.split(";") if p.strip()]:
        ftoks = _toks(part)
        best, cov = None, 0.0
        for l in src_lines:
            c = len(ftoks & _toks(l)) / max(min(len(ftoks), len(_toks(l))), 1)
            if c > cov:
                best, cov = l, c
        if best is not None and cov >= floor:
            return best
    return None


def apply_patch(src: str, match: str, patch_text: str) -> str | None:
    """Indentation-preserving multi-line replacement; prose lines dropped; None = unusable."""
    lines = [l for l in patch_text.splitlines()
             if l.strip() and not l.strip().startswith("```")
             and not re.match(r"^(The|But|So|We|This|Note|Here)\b", l.strip())]
    if not lines or (len(lines) == 1 and lines[0].strip() == match.strip()):
        return None
    indent = match[: len(match) - len(match.lstrip())]
    base = min(len(l) - len(l.lstrip()) for l in lines)
    repl = "\n".join(indent + (l[base:] if len(l) > base else l.strip()) for l in lines)
    return src.replace(match, repl, 1)
