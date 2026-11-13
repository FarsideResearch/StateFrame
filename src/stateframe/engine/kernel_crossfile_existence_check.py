"""Existence diff-check: did this edit leave an import or use of a repo symbol pointing at nothing?

Pure and read-only. Given the index of the tree before the edit (or None) and the paths the
edit touched, it re-reads those files from `root`, updates the index incrementally and reports
only breaks the edit introduced. A rename or deletion breaks the *importers*, not the edited
file, so importers of every changed module are checked too. Breaks that already existed in
the before index are not reported. Prior art: kernel_crossfile_index.dangling_refs (the
resolver this wraps) and kernel_workspace_truth (unknown stays quiet, never a false absent).
Nothing in the engine calls this yet; STATEFRAME_CROSSFILE_EXISTENCE_CHECK defaults to off.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from kernel_crossfile_index import CrossFileIndex, _norm

CROSSFILE_EXISTENCE_CHECK_ENV = "STATEFRAME_CROSSFILE_EXISTENCE_CHECK"


def enabled() -> bool:
    """Read the switch; junk values raise so a typo cannot silently run in the wrong mode."""
    raw = os.environ.get(CROSSFILE_EXISTENCE_CHECK_ENV, "off").strip().lower()
    if raw == "on":
        return True
    if raw == "off":
        return False
    raise ValueError(f"{CROSSFILE_EXISTENCE_CHECK_ENV} must be 'on' or 'off', got {raw!r}")


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    line: int
    name: str
    reason: str


def _stems(path: str) -> set[str]:
    """Module names other files could use to import `path` (a package __init__ is its directory)."""
    p = PurePosixPath(path)
    return {p.parent.name if p.stem == "__init__" else p.stem} - {""}


def _dependents(index: CrossFileIndex, changed: set[str]) -> set[str]:
    """Files with an import that may point at a changed module (over-inclusive on purpose)."""
    stems = set().union(*(_stems(p) for p in changed)) if changed else set()
    out: set[str] = set()
    for p in index.files:
        facts = index._facts(p)
        if facts is None or p in changed:
            continue
        for imp in facts.imports:
            if stems & (set(imp.module.split(".")) | {imp.name}):
                out.add(p)
                break
    return out


def check_edit(index_before: CrossFileIndex | None, root: str | Path, changed_paths: Iterable[str | Path]
               ) -> list[Finding]:
    """Findings the edit introduced, sorted. `changed_paths` are relative to `root`; a path that no
    longer exists counts as deleted. With no before index the whole tree is built fresh and every
    break in the changed files and their importers is reported (nothing to diff against)."""
    base = Path(root).resolve()
    changed = sorted({_norm(p) for p in changed_paths if str(p).endswith(".py")})
    if not changed:
        return []
    if index_before is None:
        after = CrossFileIndex.build(base)
    else:
        after = index_before
        for rel in changed:
            full = base / rel
            try:
                data = full.read_bytes() if full.is_file() else None
            except OSError:
                data = None
            after = after.update(rel, data)
    targets = sorted(set(changed) | _dependents(after, set(changed)))
    flags = after.dangling_refs(targets)
    # Compare without the line number: an edit above a pre-existing break only shifts it.
    known = {f[:1] + f[2:] for f in index_before.dangling_refs(targets)} if index_before is not None else set()
    return [Finding(*f) for f in flags if f[:1] + f[2:] not in known]


__all__ = ["CROSSFILE_EXISTENCE_CHECK_ENV", "Finding", "check_edit", "enabled"]
