"""Signature diff-check: did this edit change a function's signature and leave a caller that no longer fits?

Pure and read-only. For every def/class in the changed files whose signature differs between the
before and after index, the call sites resolving to it in the after index are judged against the
new signature (positional count within [required, max], every keyword name exists unless **kw, no
missing required parameter). Only calls that were consistent before the edit are reported, so a
caller that already did not fit stays quiet, and a caller the edit kept consistent is never
reported (the auto-close that kernel_where_used lacks: it lists callers but cannot judge them).
Calls with *args/**kwargs, decorated defs (a decorator may change the call shape) and instance
calls whose receiver is not resolved are skipped: unknown stays quiet.
Prior art: kernel_crossfile_existence_check (same index, same shape of API) and
kernel_crossfile_index.call_sites. Nothing in the engine calls this yet;
STATEFRAME_CROSSFILE_SIGNATURE_CHECK defaults to off.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from kernel_crossfile_index import CallSite, CrossFileIndex, Signature, _norm

CROSSFILE_SIGNATURE_CHECK_ENV = "STATEFRAME_CROSSFILE_SIGNATURE_CHECK"
# Decorators that leave the call shape of the underlying function as written.
_TRANSPARENT = frozenset({"staticmethod", "classmethod"})


def enabled() -> bool:
    """Read the switch; junk values raise so a typo cannot silently run in the wrong mode."""
    raw = os.environ.get(CROSSFILE_SIGNATURE_CHECK_ENV, "off").strip().lower()
    if raw == "on":
        return True
    if raw == "off":
        return False
    raise ValueError(f"{CROSSFILE_SIGNATURE_CHECK_ENV} must be 'on' or 'off', got {raw!r}")


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    callee: str
    reason: str


def _misfit(sig: Signature, site: CallSite, skip_first: bool) -> str | None:
    """Why `site` cannot call `sig`, or None when it fits (or cannot be judged)."""
    params = list(sig.params)
    if skip_first:  # bound call: the receiver fills the first non-keyword-only parameter
        for i, p in enumerate(params):
            if p.kind != "kwonly":
                del params[i]
                break
    pos = [p for p in params if p.kind != "kwonly"]
    named = {p.name: p for p in params if p.kind != "posonly"}
    for k in site.kwnames:
        if k not in named and sig.varkw is None:
            return f"unexpected keyword '{k}'"
    if site.has_star:
        return None
    if site.nargs > len(pos) and sig.varargs is None:
        return f"{site.nargs} positional arguments for at most {len(pos)}"
    for p in pos[:site.nargs]:
        if p.name in site.kwnames and p.kind != "posonly":
            return f"'{p.name}' given twice"
    for i, p in enumerate(params):
        if p.has_default:
            continue
        filled = (p.kind != "kwonly" and pos.index(p) < site.nargs) or (p.kind != "posonly" and p.name in site.kwnames)
        if not filled:
            return f"missing required '{p.name}'"
    return None


def _is_method(index: CrossFileIndex, path: str, qualname: str) -> bool:
    scope = index._scopes(path).get(qualname)
    return scope is not None and scope.cls is not None


def _judgeable(index: CrossFileIndex, qual: str) -> tuple[bool, bool] | None:
    """(is_method, is_class) when the def is a plain def/class whose call shape is its signature."""
    d = index._def(qual)
    if d is None or d.kind not in ("def", "class") or d.signature is None:
        return None
    facts = index._facts(d.path)
    decs = facts.decorators.get(d.qualname, ()) if facts else ()
    if any(dec.split("(")[0] not in _TRANSPARENT for dec in decs):
        return None
    if d.kind == "class":
        return False, True
    return _is_method(index, d.path, d.qualname), False


def _bound_sites(index: CrossFileIndex, qual: str, sites: list[CallSite]) -> list[tuple[CallSite, bool]]:
    """Pair each call with whether the receiver fills the first parameter. A call through a class
    name (Cls.method(obj, ...)) is not bound; a call whose receiver is unresolved is dropped."""
    refs = {(r.path, r.lineno): r for r in index.users(qual) if r.kind == "call"}
    out: list[tuple[CallSite, bool]] = []
    for s in sites:
        r = refs.get((s.path, s.lineno))
        if r is None or r.receiver is None:
            continue
        out.append((s, True))
    return out


def _shapes(index: CrossFileIndex, qual: str) -> list[tuple[CallSite, bool]] | None:
    """Call sites of `qual` with their bound flag, or None when the def cannot be judged."""
    kinds = _judgeable(index, qual)
    if kinds is None:
        return None
    sites = index.call_sites(qual)
    if kinds[0]:
        return _bound_sites(index, qual, sites)
    return [(s, False) for s in sites]


def check_signature_edit(index_before: CrossFileIndex | None, index_after: CrossFileIndex,
                         changed_paths: Iterable[str | Path]) -> list[Finding]:
    """Call sites the edit left inconsistent with a changed signature, sorted, one per site.
    `changed_paths` are relative to the index root. Without a before index nothing is a change."""
    if index_before is None:
        return []
    changed = sorted({_norm(p) for p in changed_paths if str(p).endswith(".py")})
    found: dict[tuple[str, int, str], Finding] = {}
    for path in changed:
        for d in index_after.defs(path):
            qual = d.qualified
            old = index_before.signature(qual)
            new = index_after.signature(qual)
            if d.kind not in ("def", "class") or old is None or new is None or old == new:
                continue
            old_def = index_before._def(qual)
            if old_def is None or old_def.kind != d.kind:
                continue
            after_sites = _shapes(index_after, qual)
            if after_sites is None:
                continue
            before_sites = _shapes(index_before, qual) or []
            # A shape that already failed against the old signature is not the edit's doing.
            broken_before = {(s.path, s.nargs, s.kwnames, s.has_star)
                             for s, b in before_sites if _misfit(old, s, b) is not None}
            for s, bound in after_sites:
                reason = _misfit(new, s, bound)
                if reason is None or (s.path, s.nargs, s.kwnames, s.has_star) in broken_before:
                    continue
                found.setdefault((s.path, s.lineno, qual), Finding(s.path, s.lineno, qual, reason))
    return sorted(found.values(), key=lambda f: (f.path, f.line, f.callee, f.reason))


__all__ = ["CROSSFILE_SIGNATURE_CHECK_ENV", "Finding", "check_signature_edit", "enabled"]
