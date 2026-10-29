"""graphs/corpus.py — deterministic repository walking.

Prior art, deliberately reused rather than reinvented: ``edit_sets.build_provider_index``
walks ``sorted(files)``, counts ``files_scanned``, and REPORTS its cap through
``ProviderIndex.truncated`` instead of silently applying it. Same discipline here.

The two real corpora this platform benches against are the ones
``probes/mr_harvest2.SOURCES`` already names, so the registry below matches them
by construction rather than hardcoding a second, drifting list.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

#: Repo key -> (git repo root, subtree). Matches probes/mr_harvest2.SOURCES.
#: Overridable via the SAME ``REPO_ROOTS`` env var ``probes/unified_gym._repo_root``
#: uses, so a stranger's checkout works without editing this file, and an unknown
#: key RAISES rather than silently benching the wrong corpus (E98).
_LEARNING_AI = pathlib.Path("/nonexistent/lab-repo")

DEFAULT_REPOS: Dict[str, Tuple[pathlib.Path, str]] = {
    "aha": (_LEARNING_AI, "backend/app"),
    "mc": (_LEARNING_AI / "mission-control", "backend/app"),
}

#: Never walk our own harness (mirrors mr_harvest2.EXCLUDE_PREFIXES) or vendored,
#: cached, or generated trees. Matched as path segments, not substrings.
EXCLUDE_SEGMENTS: Tuple[str, ...] = (
    "__pycache__",
    ".git",
    ".pytest_cache",
    "node_modules",
    "site-packages",
    ".venv",
    "venv",
)


def repo_root(key: str) -> pathlib.Path:
    """Resolve a repo key to its git root. Unknown key raises (E98 discipline)."""
    roots = {k: v[0] for k, v in DEFAULT_REPOS.items()}
    env = os.environ.get("REPO_ROOTS")
    if env:
        roots.update({k: pathlib.Path(v) for k, v in json.loads(env).items()})
    if key not in roots:
        raise KeyError(f"unknown repo key {key!r}; add it to REPO_ROOTS")
    root = roots[key]
    if not root.is_dir():
        raise FileNotFoundError(f"repo root for {key!r} missing: {root}")
    return root


def subtree_of(key: str) -> str:
    if key not in DEFAULT_REPOS:
        return "."
    return DEFAULT_REPOS[key][1]


@dataclass
class Corpus:
    """A walked repository subtree: ``{rel_path: source}`` plus its own receipt fields.

    ``rel_path`` is relative to the SUBTREE (e.g. ``api/support.py``), matching how
    ``unified_gym.all_tasks`` derives ``rel`` — so a graph entity ID and a bench
    task refer to the same file by the same name.
    """

    repo_key: str
    root: pathlib.Path
    subtree: str
    files: Dict[str, str] = field(default_factory=dict)
    files_seen: int = 0
    read_failed: int = 0
    truncated: bool = False

    def items(self) -> Iterable[Tuple[str, str]]:
        for path in sorted(self.files):
            yield path, self.files[path]

    @property
    def abs_subtree(self) -> pathlib.Path:
        return self.root / self.subtree


def _excluded(parts: Tuple[str, ...]) -> bool:
    return any(seg in EXCLUDE_SEGMENTS for seg in parts)


def load_corpus(
    repo_key: str,
    max_files: Optional[int] = None,
    subtree: Optional[str] = None,
) -> Corpus:
    """Walk one repo subtree into a deterministic ``{rel_path: source}`` map.

    Sorted traversal, no clock, no randomness. A cap is RECORDED
    (``Corpus.truncated``) rather than silently applied.
    """
    root = repo_root(repo_key)
    sub = subtree if subtree is not None else subtree_of(repo_key)
    base = root / sub
    if not base.is_dir():
        raise FileNotFoundError(f"subtree missing for {repo_key!r}: {base}")

    candidates = []
    for path in base.rglob("*.py"):
        rel_parts = path.relative_to(base).parts
        if _excluded(rel_parts):
            continue
        candidates.append(path.relative_to(base).as_posix())
    candidates.sort()

    corpus = Corpus(repo_key=repo_key, root=root, subtree=sub)
    corpus.files_seen = len(candidates)
    if max_files is not None and len(candidates) > max_files:
        corpus.truncated = True
        candidates = candidates[:max_files]

    for rel in candidates:
        try:
            corpus.files[rel] = (base / rel).read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            # An unreadable file is never a silent exclusion (#218).
            corpus.read_failed += 1
    return corpus


__all__ = ["Corpus", "DEFAULT_REPOS", "EXCLUDE_SEGMENTS", "load_corpus", "repo_root", "subtree_of"]
