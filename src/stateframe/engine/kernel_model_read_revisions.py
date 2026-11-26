"""Track source revisions the model received and reject stale path edits."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping


_REVISION = re.compile(r"sha256:[0-9a-f]{64}\Z")


def revision_of(source: bytes) -> str:
    return "sha256:" + hashlib.sha256(source).hexdigest()


def workspace_path(root: str | Path, path: str | Path) -> str | None:
    """Return a canonical workspace-relative path for an already-resolved target."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None


def refusal_for_revision(*, relative_path: str, expected_revision: str,
                         current_source: bytes | None, operation: str,
                         current_missing: bool = False) -> str:
    """Render the ordinary tool refusal for a known stale or unverifiable source."""
    current_revision = ("missing" if current_missing else
                        revision_of(current_source) if current_source is not None else
                        "unavailable")
    if current_revision == expected_revision:
        return ""
    if current_revision == "unavailable":
        detail = "the current revision could not be verified"
    elif current_revision == "missing":
        detail = "the file is missing"
    else:
        detail = f"the current revision is {current_revision}"
    return (f"{operation} REJECTED: {relative_path} changed since your last read; {detail}. "
            f"Re-read it with `read {relative_path}` and retry. Nothing was changed.")


class ModelReadRevisionLedger:
    """Per-run path revisions returned by successful, model-visible reads."""

    def __init__(self, values: Mapping[str, str] | None = None) -> None:
        self._revisions: dict[str, str] = {}
        for path, revision in (values or {}).items():
            self.record(str(path), str(revision))

    def record(self, relative_path: str, revision: str) -> None:
        normalized = Path(relative_path)
        if (not relative_path or normalized.is_absolute() or ".." in normalized.parts
                or normalized.as_posix() != relative_path or not _REVISION.fullmatch(revision)):
            return
        self._revisions[relative_path] = revision

    def expected_for(self, root: str | Path, path: str | Path) -> str | None:
        relative = workspace_path(root, path)
        return self._revisions.get(relative) if relative is not None else None

    def observe_read(self, root: str | Path, path: str | Path, revision: object) -> None:
        relative = workspace_path(root, path)
        if relative is not None and isinstance(revision, str):
            self.record(relative, revision)

    def observe_applied(self, root: str | Path, path: str | Path,
                        current_source: bytes) -> None:
        """Advance the model's known revision after its own edit was reported applied."""
        relative = workspace_path(root, path)
        if relative is not None:
            self.record(relative, revision_of(current_source))

    def snapshot(self) -> dict[str, str]:
        """Return the JSON-safe task state used by resumable runs."""
        return dict(self._revisions)

    def refusal(self, root: str | Path, path: str | Path, *, operation: str,
                current_source: bytes | None = None) -> str | None:
        relative = workspace_path(root, path)
        expected = self._revisions.get(relative) if relative is not None else None
        if expected is None or relative is None:
            return None
        if current_source is None:
            try:
                from kernel_scope_widen_guard import read as guarded_read
                current_source = guarded_read(Path(path))
            except FileNotFoundError:
                return refusal_for_revision(
                    relative_path=relative, expected_revision=expected,
                    current_source=None, current_missing=True, operation=operation,
                )
            except (OSError, ValueError):
                return refusal_for_revision(
                    relative_path=relative, expected_revision=expected,
                    current_source=None, operation=operation,
                )
        message = refusal_for_revision(
            relative_path=relative, expected_revision=expected,
            current_source=current_source, operation=operation,
        )
        return message or None


__all__ = ["ModelReadRevisionLedger", "refusal_for_revision", "revision_of", "workspace_path"]
