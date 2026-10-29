"""Incremental, language-neutral source facts for host where-used previews.

The index is an optimization of host parsing only.  It never admits a
candidate, closes an obligation, or changes the authority of ``kernel_where_used``.
Unsupported execution is represented explicitly as coverage, not PASS.
"""

from __future__ import annotations

import hashlib
import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping


_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv", "node_modules",
    "dist", "build",
})
_EXTRACTORS = {
    ".py": "python-ast/1", ".go": "go-identifiers/1", ".rs": "rust-identifiers/1",
    ".ts": "typescript-identifiers/1", ".tsx": "typescript-identifiers/1",
    ".js": "javascript-identifiers/1", ".jsx": "javascript-identifiers/1",
    ".java": "java-identifiers/1", ".php": "php-identifiers/1",
    ".vue": "vue-identifiers/1", ".json": "data-keys/1", ".yaml": "data-keys/1",
    ".yml": "data-keys/1", ".md": "document-terms/1", ".txt": "document-terms/1",
}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class FileReferenceFact:
    path: str
    content_digest: str
    extractor_identity: str
    coverage: str
    references: tuple[str, ...]


class ReferenceIndex:
    """Cache immutable per-file facts by content digest and extractor identity."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else None
        self._facts: dict[str, FileReferenceFact] = {}
        self._raw_by_digest: dict[tuple[str, str], FileReferenceFact] = {}
        self.generation = 0
        self._last_revision = ""
        self._last_root = ""
        self._pack_cache: dict[tuple[str, str, str], object] = {}

    @staticmethod
    def _digest(raw: bytes) -> str:
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _value_digest(value: object) -> str:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
        return ReferenceIndex._digest(raw)

    @staticmethod
    def _files(root: Path) -> tuple[Path, ...]:
        stack = [root]
        result: list[Path] = []
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name, reverse=True)
            except (OSError, NotADirectoryError):
                continue
            for entry in entries:
                if entry.name in _SKIP_DIRS:
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        result.append(Path(entry.path))
                except OSError:
                    continue
        return tuple(sorted(result, key=lambda path: path.relative_to(root).as_posix()))

    def _fact(self, relative: str, raw: bytes) -> FileReferenceFact:
        suffix = Path(relative).suffix.lower()
        extractor = _EXTRACTORS.get(suffix)
        digest = self._digest(raw)
        if extractor is None:
            return FileReferenceFact(relative, digest, "unsupported/1", "UNSUPPORTED", ())
        key = (digest, extractor)
        cached = self._raw_by_digest.get(key)
        if cached is not None:
            return FileReferenceFact(relative, cached.content_digest, cached.extractor_identity,
                                     cached.coverage, cached.references)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            fact = FileReferenceFact(relative, digest, extractor, "PARTIAL", ())
        else:
            refs = tuple(sorted(set(_IDENTIFIER.findall(text))))
            if suffix == ".py":
                try:
                    ast.parse(text, filename=relative)
                except (SyntaxError, ValueError, TypeError):
                    coverage = "PARTIAL"
                else:
                    coverage = "COMPLETE"
            elif suffix in {".json", ".yaml", ".yml", ".md", ".txt"}:
                coverage = "COMPLETE"
            else:
                # The language-neutral extractors provide lexical facts only;
                # they must not claim complete semantic coverage for polyglot
                # code they cannot parse authoritatively.
                coverage = "PARTIAL"
            fact = FileReferenceFact(relative, digest, extractor, coverage, refs)
        self._raw_by_digest[key] = fact
        return fact

    def refresh(self, root: str | Path | None = None, *, overlays: Mapping[str, str] | None = None,
                root_revision: object = None) -> tuple[FileReferenceFact, ...]:
        root_path = Path(root or self.root or ".").resolve()
        self.root = root_path
        overlay_map: dict[str, str] = {}
        for path, value in (overlays or {}).items():
            relative = PurePosixPath(str(path).replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("overlay path escapes the workspace root")
            key = relative.as_posix()
            if not key or key == ".":
                raise ValueError("overlay path is empty")
            overlay_map[key] = str(value)
        current: dict[str, FileReferenceFact] = {}
        for path in self._files(root_path):
            relative = path.relative_to(root_path).as_posix()
            if relative in overlay_map:
                raw = overlay_map[relative].encode("utf-8")
            else:
                try:
                    raw = path.read_bytes()
                except OSError:
                    continue
            current[relative] = self._fact(relative, raw)
        for relative, text in overlay_map.items():
            if relative not in current:
                current[relative] = self._fact(relative, text.encode("utf-8"))
        root_key = str(root_path)
        if current != self._facts or root_key != self._last_root:
            self.generation += 1
            self._facts = current
            self._last_root = root_key
        # Caller revision identifies the surrounding workspace observation, but
        # must not invalidate immutable content facts by itself.
        self._last_revision = str(root_revision or "")
        return tuple(current.values())

    def coverage(self) -> dict[str, object]:
        counts: dict[str, int] = {}
        unsupported: list[str] = []
        for fact in self._facts.values():
            counts[fact.coverage] = counts.get(fact.coverage, 0) + 1
            if fact.coverage != "COMPLETE":
                unsupported.append(fact.path)
        return {"status": "COMPLETE" if not unsupported else "PARTIAL",
                "counts": counts, "unsupported_paths": sorted(unsupported),
                "generation": self.generation}

    def code_pack(self, root: str | Path | None = None, *, overlays: Mapping[str, str] | None = None,
                  root_revision: object = None) -> object:
        """Return the canonical where-used pack after refreshing file facts."""
        root_path = Path(root or self.root or ".").resolve()
        self.refresh(root_path, overlays=overlays, root_revision=root_revision)
        overlay_digest = self._value_digest({str(key): str(value) for key, value in (overlays or {}).items()})
        facts_digest = self._value_digest(tuple(
            (fact.path, fact.content_digest, fact.extractor_identity,
             fact.coverage, fact.references)
            for fact in sorted(self._facts.values(), key=lambda item: item.path)))
        cache_key = (str(root_path), facts_digest, overlay_digest)
        cached = self._pack_cache.get(cache_key)
        if cached is not None:
            return cached
        from kernel_where_used import build_code_pack
        pack = build_code_pack(root_path, include_literals=False, overlays=overlays)
        if len(self._pack_cache) >= 8:
            self._pack_cache.clear()
        self._pack_cache[cache_key] = pack
        return pack


__all__ = ["FileReferenceFact", "ReferenceIndex"]
