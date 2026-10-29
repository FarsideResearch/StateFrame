"""Immutable, bounded product overlays for candidate recovery.

This module is deliberately filesystem-oriented but has no knowledge of checks,
grades, completion, or model instructions.  A checkpoint is a complete
content-addressed manifest.  Recovery stages and validates a new generation
before changing the requested workspace, and compares the workspace again
under an exclusive lock before activation.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA = "stateframe-candidate-checkpoint/1"
DEFAULT_MAX_CHECKPOINTS = 32
MAX_PATHS = 4096


class CheckpointError(RuntimeError):
    """Base error for a refused or failed checkpoint operation."""


class CheckpointCapacityError(CheckpointError):
    """The bounded store cannot accept another immutable checkpoint."""


class CheckpointPathError(CheckpointError):
    """A manifest path is outside its root or crosses a symlink."""


class CheckpointConflictError(CheckpointError):
    """Workspace changed or another writer owns recovery activation."""


class CheckpointWriteError(CheckpointError):
    """A staged generation or activation could not be verified."""


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _manifest_digest(rows: Iterable[tuple[str, str | None, int, int, bool]]) -> str:
    payload = [list(row) for row in sorted(rows)]
    return _sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())


def _as_bytes(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise TypeError("manifest content must be bytes, str, bytearray, or None")


def _root_path(root: object) -> Path:
    path = Path(root).expanduser()
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise CheckpointPathError(f"checkpoint root is unavailable: {type(error).__name__}") from error


def _safe_relative(root: Path, value: object) -> str:
    raw = str(value or "").replace("\\", "/")
    if not raw or raw.startswith("/") or raw == "." or "\x00" in raw:
        raise CheckpointPathError("manifest path must be a non-empty relative path")
    candidate = Path(raw)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise CheckpointPathError("manifest path contains an unsafe component")
    current = root
    for part in candidate.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise CheckpointPathError(f"symlink path is not recoverable: {raw}")
        except OSError as error:
            raise CheckpointPathError(f"cannot inspect manifest path: {type(error).__name__}") from error
    try:
        resolved = (root / candidate).resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise CheckpointPathError(f"manifest path escapes root: {raw}") from error
    return candidate.as_posix()


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    digest: str | None
    size: int
    mode: int
    present: bool
    content: bytes | None = None

    def __post_init__(self) -> None:
        if not self.path or Path(self.path).is_absolute():
            raise ValueError("manifest entry path must be relative")
        if self.present:
            if self.content is None or self.digest != _sha256(self.content):
                raise ValueError("manifest entry content and digest do not match")
            if self.size != len(self.content):
                raise ValueError("manifest entry size does not match content")
        elif self.digest is not None or self.size != 0 or self.content is not None:
            raise ValueError("absent manifest entry must have no content")


@dataclass(frozen=True)
class ProductManifest:
    """A complete immutable product path set, including explicit absences."""

    entries: tuple[ManifestEntry, ...]
    digest: str
    complete: bool = True

    def __post_init__(self) -> None:
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("product manifest contains duplicate paths")
        expected = _manifest_digest(
            (entry.path, entry.digest, entry.size, entry.mode, entry.present)
            for entry in self.entries
        )
        if self.digest != expected:
            raise ValueError("product manifest digest is not content-addressed")
        if not self.complete:
            raise ValueError("candidate recovery requires a complete product manifest")

    @classmethod
    def from_mapping(
            cls,
            values: Mapping[str, object],
            *,
            modes: Mapping[str, int] | None = None,
            complete: bool = True,
    ) -> "ProductManifest":
        if not isinstance(values, Mapping):
            raise TypeError("manifest source must be a mapping")
        if len(values) > MAX_PATHS:
            raise ValueError("manifest has too many paths")
        mode_map = modes or {}
        entries: list[ManifestEntry] = []
        for raw_path, raw_content in sorted(values.items(), key=lambda item: str(item[0])):
            path = str(raw_path).replace("\\", "/")
            path_parts = Path(path).parts
            if (not path or path.startswith("/") or "\x00" in path
                    or any(part in {"", ".", ".."} for part in path_parts)):
                raise CheckpointPathError("manifest path is not a safe relative path")
            content = _as_bytes(raw_content)
            entries.append(ManifestEntry(
                path=path,
                digest=_sha256(content) if content is not None else None,
                size=len(content) if content is not None else 0,
                mode=int(mode_map.get(path, 0o644)),
                present=content is not None,
                content=content,
            ))
        digest = _manifest_digest(
            (entry.path, entry.digest, entry.size, entry.mode, entry.present)
            for entry in entries
        )
        return cls(tuple(entries), digest, complete=complete)

    @classmethod
    def from_root(
            cls,
            root: object,
            paths: Iterable[object],
            *,
            complete: bool = True,
    ) -> "ProductManifest":
        base = _root_path(root)
        normalized = sorted({_safe_relative(base, path) for path in paths})
        if len(normalized) > MAX_PATHS:
            raise ValueError("manifest has too many paths")
        values: dict[str, bytes | None] = {}
        modes: dict[str, int] = {}
        for relative in normalized:
            target = base / relative
            try:
                stat = target.lstat()
            except FileNotFoundError:
                values[relative] = None
                continue
            except OSError as error:
                raise CheckpointPathError(f"cannot read manifest path: {type(error).__name__}") from error
            if target.is_symlink() or not target.is_file():
                raise CheckpointPathError(f"manifest path is not a regular file: {relative}")
            try:
                values[relative] = target.read_bytes()
            except OSError as error:
                raise CheckpointPathError(f"cannot read manifest bytes: {type(error).__name__}") from error
            modes[relative] = stat.st_mode & 0o777
        return cls.from_mapping(values, modes=modes, complete=complete)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.entries)

    def entry(self, path: object) -> ManifestEntry:
        key = str(path).replace("\\", "/")
        for entry in self.entries:
            if entry.path == key:
                return entry
        raise KeyError(key)

    def as_metadata(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "digest": self.digest,
            "complete": self.complete,
            "entries": [
                {
                    "path": entry.path,
                    "digest": entry.digest,
                    "size": entry.size,
                    "mode": entry.mode,
                    "present": entry.present,
                }
                for entry in self.entries
            ],
        }


@dataclass(frozen=True)
class ActivationReceipt:
    schema: str
    status: str
    checkpoint_id: str
    generation_id: str
    old_digest: str
    new_digest: str
    reason: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "status": self.status,
            "checkpoint_id": self.checkpoint_id,
            "generation_id": self.generation_id,
            "old_digest": self.old_digest,
            "new_digest": self.new_digest,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StoredCheckpoint:
    checkpoint_id: str
    manifest: ProductManifest
    run_id: str
    task_id: str
    parent_revision: str
    effect_id: str
    branch_id: str
    created_index: int
    model_owned: bool = True
    settled: bool = True
    baseline: bool = False


class CheckpointStore:
    """Bounded immutable checkpoint registry with optional durable bytes."""

    def __init__(self, root: object | None = None, *, max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS):
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        self.root = Path(root).expanduser() if root is not None else None
        self.max_checkpoints = int(max_checkpoints)
        self._items: dict[str, StoredCheckpoint] = {}
        self._sequence = 0
        self._load_existing()

    def _load_existing(self) -> None:
        if self.root is None or not self.root.is_dir():
            return
        for target in sorted(self.root.iterdir(), key=lambda path: path.name):
            if not target.is_dir() or target.is_symlink() or target.name.startswith("."):
                continue
            metadata_path = target / "checkpoint.json"
            manifest_path = target / "manifest.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                manifest_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
                values: dict[str, bytes | None] = {}
                modes: dict[str, int] = {}
                for row in manifest_metadata.get("entries", ()):
                    relative = str(row["path"])
                    file_path = target / "files" / relative
                    values[relative] = file_path.read_bytes() if bool(row.get("present")) else None
                    modes[relative] = int(row.get("mode", 0o644))
                manifest = ProductManifest.from_mapping(values, modes=modes,
                                                        complete=bool(manifest_metadata.get("complete", False)))
                if manifest.digest != str(manifest_metadata.get("digest", "")):
                    continue
                created = int(metadata.get("created_index", self._sequence))
                self._items[target.name] = StoredCheckpoint(
                    checkpoint_id=target.name, manifest=manifest,
                    run_id=str(metadata.get("run_id", "")), task_id=str(metadata.get("task_id", "")),
                    parent_revision=str(metadata.get("parent_revision", "")),
                    effect_id=str(metadata.get("effect_id", "")),
                    branch_id=str(metadata.get("branch_id", "")), created_index=created,
                    model_owned=bool(metadata.get("model_owned", True)),
                    settled=bool(metadata.get("settled", True)),
                    baseline=bool(metadata.get("baseline", False)),
                )
                self._sequence = max(self._sequence, created + 1)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                # A partial/crashed capture is not an eligible checkpoint.
                continue

    def _storage_root(self) -> Path | None:
        if self.root is None:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def capture(
            self,
            manifest: ProductManifest,
            *,
            checkpoint_id: str,
            run_id: str,
            task_id: str,
            parent_revision: str,
            effect_id: str,
            branch_id: str,
            model_owned: bool = True,
            settled: bool = True,
            baseline: bool = False,
    ) -> StoredCheckpoint:
        if not manifest.complete:
            raise CheckpointError("cannot store an incomplete product manifest")
        checkpoint_key = str(checkpoint_id)
        if not checkpoint_key or checkpoint_key in {".", ".."} or Path(checkpoint_key).name != checkpoint_key:
            raise CheckpointPathError("checkpoint id must be a single safe path component")
        if checkpoint_key in self._items:
            existing = self._items[checkpoint_key]
            if existing.manifest.digest == manifest.digest:
                return existing
            raise CheckpointError("checkpoint id is already bound to another manifest")
        if len(self._items) >= self.max_checkpoints:
            raise CheckpointCapacityError("bounded checkpoint store is full")
        checkpoint = StoredCheckpoint(
            checkpoint_id=checkpoint_key, manifest=manifest,
            run_id=str(run_id), task_id=str(task_id),
            parent_revision=str(parent_revision), effect_id=str(effect_id),
            branch_id=str(branch_id), created_index=self._sequence,
            model_owned=bool(model_owned), settled=bool(settled), baseline=bool(baseline),
        )
        storage = self._storage_root()
        if storage is not None:
            self._persist_checkpoint(storage, checkpoint)
        self._items[checkpoint.checkpoint_id] = checkpoint
        self._sequence += 1
        return checkpoint

    def _persist_checkpoint(self, storage: Path, checkpoint: StoredCheckpoint) -> None:
        target = storage / checkpoint.checkpoint_id
        if target.exists() or target.is_symlink():
            raise CheckpointError("checkpoint storage path already exists")
        temporary = Path(tempfile.mkdtemp(prefix=".candidate-", dir=str(storage)))
        try:
            manifest_payload = checkpoint.manifest.as_metadata()
            (temporary / "manifest.json").write_text(
                json.dumps(manifest_payload, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            (temporary / "checkpoint.json").write_text(json.dumps({
                "schema": SCHEMA,
                "checkpoint_id": checkpoint.checkpoint_id,
                "run_id": checkpoint.run_id,
                "task_id": checkpoint.task_id,
                "parent_revision": checkpoint.parent_revision,
                "effect_id": checkpoint.effect_id,
                "branch_id": checkpoint.branch_id,
                "created_index": checkpoint.created_index,
                "model_owned": checkpoint.model_owned,
                "settled": checkpoint.settled,
                "baseline": checkpoint.baseline,
            }, sort_keys=True), encoding="utf-8")
            for entry in checkpoint.manifest.entries:
                if not entry.present or entry.content is None:
                    continue
                target_file = temporary / "files" / entry.path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_bytes(entry.content)
                os.chmod(target_file, entry.mode or 0o644)
            os.replace(temporary, target)
        except (OSError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            raise CheckpointWriteError(f"checkpoint capture failed: {type(error).__name__}") from error

    def get(self, checkpoint_id: object) -> StoredCheckpoint | None:
        return self._items.get(str(checkpoint_id))

    def all(self) -> tuple[StoredCheckpoint, ...]:
        return tuple(sorted(self._items.values(), key=lambda item: item.created_index))

    def bytes_for(self, checkpoint: StoredCheckpoint) -> dict[str, bytes | None]:
        storage = self._storage_root()
        if storage is None:
            return {entry.path: entry.content for entry in checkpoint.manifest.entries}
        result: dict[str, bytes | None] = {}
        for entry in checkpoint.manifest.entries:
            if not entry.present:
                result[entry.path] = None
                continue
            path = storage / checkpoint.checkpoint_id / "files" / entry.path
            try:
                data = path.read_bytes()
            except OSError as error:
                raise CheckpointWriteError(f"checkpoint bytes unavailable: {type(error).__name__}") from error
            if _sha256(data) != entry.digest:
                raise CheckpointWriteError("checkpoint bytes failed digest verification")
            result[entry.path] = data
        return result

    def current_manifest(self, workspace: object, checkpoint: StoredCheckpoint) -> ProductManifest:
        return ProductManifest.from_root(workspace, checkpoint.manifest.paths)

    def stage_generation(self, checkpoint: StoredCheckpoint, *, generation_id: str | None = None) -> Path:
        storage = self._storage_root()
        if storage is None:
            storage = Path(tempfile.mkdtemp(prefix="candidate-generations-"))
        generations = storage / "generations"
        generations.mkdir(parents=True, exist_ok=True)
        generation_key = generation_id or f"generation-{checkpoint.created_index:08d}"
        if Path(generation_key).name != generation_key or generation_key in {".", ".."}:
            raise CheckpointPathError("generation id must be a single safe path component")
        generation = generations / generation_key
        if generation.exists() or generation.is_symlink():
            raise CheckpointWriteError("generation id already exists")
        temporary = Path(tempfile.mkdtemp(prefix=".generation-", dir=str(generations)))
        try:
            data = self.bytes_for(checkpoint)
            for entry in checkpoint.manifest.entries:
                if not entry.present:
                    continue
                target = temporary / entry.path
                target.parent.mkdir(parents=True, exist_ok=True)
                content = data[entry.path]
                if content is None or _sha256(content) != entry.digest:
                    raise CheckpointWriteError("staged generation content failed validation")
                target.write_bytes(content)
                os.chmod(target, entry.mode or 0o644)
            (temporary / "manifest.json").write_text(
                json.dumps(checkpoint.manifest.as_metadata(), sort_keys=True), encoding="utf-8")
            os.replace(temporary, generation)
            return generation
        except (OSError, ValueError) as error:
            shutil.rmtree(temporary, ignore_errors=True)
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointWriteError(f"generation staging failed: {type(error).__name__}") from error

    def activate(
            self,
            checkpoint: StoredCheckpoint,
            workspace: object,
            *,
            expected_current_digest: str,
            generation_id: str | None = None,
    ) -> ActivationReceipt:
        base = _root_path(workspace)
        current = self.current_manifest(base, checkpoint)
        if current.digest != str(expected_current_digest):
            raise CheckpointConflictError("current revision compare-and-swap failed")
        generation = self.stage_generation(checkpoint, generation_id=generation_id)
        generation_label = generation.name
        lock_path = (self._storage_root() or base) / ".recovery.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = lock_path.open("a+")
        try:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (ImportError, OSError) as error:
                raise CheckpointConflictError("external worker or recovery lock is active") from error
            # Re-check after acquiring the lock so a writer racing the initial
            # compare cannot be mistaken for the expected workspace.
            locked_current = self.current_manifest(base, checkpoint)
            if locked_current.digest != str(expected_current_digest):
                raise CheckpointConflictError("current revision changed before activation")
            before = locked_current
            data = self.bytes_for(checkpoint)
            original: dict[str, bytes | None] = {
                entry.path: (base / entry.path).read_bytes() if entry.present else None
                for entry in before.entries
                if (base / entry.path).is_file()
            }
            try:
                for entry in checkpoint.manifest.entries:
                    target = base / entry.path
                    if target.is_symlink():
                        raise CheckpointPathError(f"activation target became a symlink: {entry.path}")
                    if entry.present:
                        temporary = target.with_name(f".{target.name}.candidate-{generation_label}")
                        temporary.parent.mkdir(parents=True, exist_ok=True)
                        temporary.write_bytes(data[entry.path] or b"")
                        os.chmod(temporary, entry.mode or 0o644)
                        os.replace(temporary, target)
                    elif target.exists():
                        target.unlink()
                verified = self.current_manifest(base, checkpoint)
                if verified.digest != checkpoint.manifest.digest:
                    raise CheckpointWriteError("activated bytes failed manifest verification")
            except (OSError, CheckpointError) as error:
                for entry in before.entries:
                    target = base / entry.path
                    old = original.get(entry.path)
                    try:
                        if old is None:
                            if target.exists() and not target.is_symlink():
                                target.unlink()
                        else:
                            target.write_bytes(old)
                    except OSError:
                        # The receipt remains a failed activation; never report recovery.
                        pass
                if isinstance(error, CheckpointError):
                    raise
                raise CheckpointWriteError(f"activation failed: {type(error).__name__}") from error
            return ActivationReceipt(
                schema=SCHEMA, status="ACTIVATED", checkpoint_id=checkpoint.checkpoint_id,
                generation_id=generation_label, old_digest=before.digest,
                new_digest=checkpoint.manifest.digest,
            )
        finally:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            lock_file.close()


__all__ = [
    "ActivationReceipt", "CheckpointCapacityError", "CheckpointConflictError",
    "CheckpointError", "CheckpointPathError", "CheckpointStore", "CheckpointWriteError",
    "ManifestEntry", "ProductManifest", "SCHEMA", "StoredCheckpoint",
]
