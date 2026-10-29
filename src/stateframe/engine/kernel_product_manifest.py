"""Host-owned product/build-input identity for language-general work.

The manifest answers a narrower question than a source parser: which bytes and
file modes are authorized inputs to this workspace?  It deliberately has no
extension allow-list.  Parser coverage and revision membership are separate
capabilities, so a Dockerfile, Makefile, Go module, SQL file, extensionless
script, or binary asset can invalidate old evidence without needing a parser.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from kernel_test_artifact_policy import discover_test_artifact_policy


SCHEMA = "stateframe-product-manifest/1"
REVISION_PREFIX = "sha256:"

# These are scope boundaries, not language filters.  A repository may still
# authorize an unusual extension or an extensionless input inside its product
# roots.  Generated/dependency trees are excluded before any file is read.
DEFAULT_EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".hypothesis", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".venv", "venv",
    "node_modules", "vendor", "dist", "build", "out", "target", "coverage",
    "backups", "backup", "private", "hidden", "hidden_tests", "near_miss",
    "gold", "oracle",
})


class ProductManifestError(ValueError):
    """Raised when an identity scope is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    path: str
    kind: str
    mode: int
    size: int
    digest: str
    readable: bool = True
    link_target: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "mode": self.mode,
            "size": self.size,
            "digest": self.digest,
            "readable": self.readable,
            "link_target": self.link_target,
        }


@dataclass(frozen=True, slots=True)
class ProductManifest:
    root: str
    scope: tuple[str, ...]
    entries: tuple[ManifestEntry, ...]
    skipped: tuple[str, ...] = ()

    @property
    def revision(self) -> str:
        payload = {
            "schema": SCHEMA,
            "scope": list(self.scope),
            "entries": [entry.to_dict() for entry in self.entries],
            "skipped": list(self.skipped),
        }
        return REVISION_PREFIX + hashlib.sha256(_canonical(payload)).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "root": self.root,
            "scope": list(self.scope),
            "entries": [entry.to_dict() for entry in self.entries],
            "skipped": list(self.skipped),
            "revision": self.revision,
        }


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _normal_scope(root: Path, values: Sequence[str | Path] | None) -> tuple[str, ...]:
    if not values:
        return (".",)
    result: list[str] = []
    for raw in values:
        candidate = Path(str(raw).replace("\\", "/"))
        target = (candidate if candidate.is_absolute() else root / candidate).resolve()
        try:
            relative = target.relative_to(root).as_posix() or "."
        except ValueError as error:
            raise ProductManifestError(f"manifest scope escapes root: {raw}") from error
        if relative not in result:
            result.append(relative)
    return tuple(sorted(result))


def _excluded(relative: Path, excluded_dirs: frozenset[str]) -> bool:
    return any(part.casefold() in excluded_dirs for part in relative.parts)


def _entry_for(path: Path, relative: str) -> ManifestEntry:
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    if stat.S_ISLNK(info.st_mode):
        target = os.readlink(path)
        digest = REVISION_PREFIX + hashlib.sha256(
            _canonical({"kind": "symlink", "target": target})
        ).hexdigest()
        return ManifestEntry(relative, "symlink", mode, 0, digest,
                             link_target=target)
    if not stat.S_ISREG(info.st_mode):
        digest = REVISION_PREFIX + hashlib.sha256(
            _canonical({"kind": "special", "mode": mode})
        ).hexdigest()
        return ManifestEntry(relative, "special", mode, int(info.st_size), digest)
    try:
        raw = path.read_bytes()
    except (OSError, UnicodeError) as error:
        # Keep an explicit incomplete-input fact in the identity.  An
        # unreadable authorized file must never silently disappear from scope.
        error_digest = REVISION_PREFIX + hashlib.sha256(
            _canonical({"kind": "unreadable", "error": type(error).__name__,
                        "size": int(info.st_size), "mode": mode})
        ).hexdigest()
        return ManifestEntry(relative, "unreadable", mode, int(info.st_size),
                             error_digest, readable=False)
    return ManifestEntry(
        relative, "file", mode, len(raw),
        REVISION_PREFIX + hashlib.sha256(raw).hexdigest(),
    )


def _walk_scope(root: Path, scope: str, excluded_dirs: frozenset[str]) -> tuple[Path, ...]:
    start = root if scope == "." else (root / scope)
    if not start.exists() and not start.is_symlink():
        return ()
    if start.is_file() or start.is_symlink():
        relative = start.relative_to(root)
        return () if _excluded(relative, excluded_dirs) else (start,)
    rows: list[Path] = []
    for current, directories, files in os.walk(start, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(root)
        symlink_dirs = [current_path / name for name in directories
                        if (current_path / name).is_symlink()]
        rows.extend(
            path for path in symlink_dirs
            if not _excluded(path.relative_to(root), excluded_dirs)
        )
        directories[:] = sorted(
            name for name in directories
            if not (current_path / name).is_symlink()
            and not _excluded(relative_current / name, excluded_dirs)
        )
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root)
            if not _excluded(relative, excluded_dirs):
                rows.append(path)
    return tuple(rows)


def build_product_manifest(
    root: str | Path,
    *,
    scope: Sequence[str | Path] | None = None,
    excluded_dirs: Iterable[str] = (),
) -> ProductManifest:
    """Build a complete, deterministic identity for authorized inputs."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ProductManifestError(f"manifest root is not a directory: {root}")
    excluded = frozenset({*DEFAULT_EXCLUDED_DIRS,
                          *(str(item).strip().casefold() for item in excluded_dirs
                            if str(item).strip())})
    normalized_scope = _normal_scope(root_path, scope)
    test_artifacts = discover_test_artifact_policy(root_path)
    paths: set[Path] = set()
    for item in normalized_scope:
        paths.update(_walk_scope(root_path, item, excluded))
    entries: list[ManifestEntry] = []
    skipped: set[str] = set()
    for path in sorted(paths, key=lambda item: item.relative_to(root_path).as_posix()):
        relative = path.relative_to(root_path).as_posix()
        if test_artifacts.matches(relative):
            continue
        try:
            entry = _entry_for(path, relative)
        except (OSError, UnicodeError) as error:
            skipped.add(f"{relative} ({type(error).__name__})")
            continue
        entries.append(entry)
        if not entry.readable:
            skipped.add(relative)
    return ProductManifest(str(root_path), normalized_scope, tuple(entries),
                           tuple(sorted(skipped)))


def product_revision(root: str | Path, *, scope: Sequence[str | Path] | None = None,
                     excluded_dirs: Iterable[str] = ()) -> str:
    """Return the shared product/build-input revision used by all v24 owners."""
    return build_product_manifest(root, scope=scope, excluded_dirs=excluded_dirs).revision


__all__ = [
    "DEFAULT_EXCLUDED_DIRS", "ManifestEntry", "ProductManifest",
    "ProductManifestError", "SCHEMA", "build_product_manifest", "product_revision",
]
