"""Immutable final-product identity and terminal selection for C07.

The existing graders remain responsible for executing a candidate.  This module
only describes the bytes that an execution and its grade are allowed to join.
It deliberately refuses path/mtime inference, stale favorable terminal rows,
and deletion-blind overlays.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Final


FINAL_PRODUCT_SCHEMA: Final[str] = "sciledger/final-product-manifest/1"
FINAL_PRODUCT_MANIFEST_NAME: Final[str] = "final-product-manifest.json"
_TERMINAL_CLASSES = frozenset(
    {"DONE", "COMPLETE", "ACCEPTED", "REJECTED", "FAILED", "INTERRUPTED", "UNKNOWN"}
)


class FinalProductError(ValueError):
    """Raised when a final-product join is ambiguous or not byte-bound."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_value(value: object) -> str:
    return digest_bytes(canonical_bytes(value))


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FinalProductError(f"{label} must be non-empty text")
    return value.strip()


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise FinalProductError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _optional_digest(value: object, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _safe_path(value: object, label: str) -> str:
    path = _text(value, label).replace("\\", "/")
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise FinalProductError(f"{label} must be a safe relative POSIX path")
    return parsed.as_posix()


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise FinalProductError(f"{label} must be boolean")
    return value


def _step(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FinalProductError(f"{label} must be a non-negative integer")
    return value


def _hash_files(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise FinalProductError(f"product root is not a regular directory: {root}")
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise FinalProductError(f"product root contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = _safe_path(path.relative_to(root).as_posix(), "product path")
        result[relative] = digest_bytes(path.read_bytes())
    return result


def tree_digest(files: Mapping[str, str]) -> str:
    normalized = {
        _safe_path(path, "tree path"): _digest(file_digest, f"tree hash {path}")
        for path, file_digest in files.items()
    }
    return digest_value(normalized)


def overlay_digest(additions: Mapping[str, str], deletions: tuple[str, ...]) -> str:
    normalized_additions = {
        _safe_path(path, "addition path"): _digest(file_digest, f"addition hash {path}")
        for path, file_digest in additions.items()
    }
    normalized_deletions = tuple(sorted(_safe_path(path, "deletion path") for path in deletions))
    overlap = set(normalized_additions) & set(normalized_deletions)
    if overlap:
        raise FinalProductError(f"a path cannot be both added and deleted: {sorted(overlap)}")
    return digest_value({"additions": normalized_additions, "deletions": normalized_deletions})


@dataclass(frozen=True)
class FinalProductManifest:
    """Complete immutable join key for one terminal product."""

    run_id: str
    solver_run_id: str
    task_id: str
    task_bundle_digest: str
    image_digest: str
    release_digest: str
    parent_tree_digest: str
    final_event_id: str
    final_step: int
    additions: Mapping[str, str]
    deletions: tuple[str, ...]
    overlay_digest: str
    effective_product_digest: str
    terminal_class: str
    certified: bool
    salvage: bool = False
    candidate_boundary: str | None = None
    certification_digest: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("run_id", self.run_id),
            ("solver_run_id", self.solver_run_id),
            ("task_id", self.task_id),
            ("final_event_id", self.final_event_id),
            ("terminal_class", self.terminal_class),
        ):
            _text(value, f"final_product.{label}")
        for label, value in (
            ("task_bundle_digest", self.task_bundle_digest),
            ("image_digest", self.image_digest),
            ("release_digest", self.release_digest),
            ("parent_tree_digest", self.parent_tree_digest),
            ("overlay_digest", self.overlay_digest),
            ("effective_product_digest", self.effective_product_digest),
        ):
            _digest(value, f"final_product.{label}")
        _step(self.final_step, "final_product.final_step")
        if self.terminal_class not in _TERMINAL_CLASSES:
            raise FinalProductError(f"unknown terminal class: {self.terminal_class}")
        _boolean(self.certified, "final_product.certified")
        _boolean(self.salvage, "final_product.salvage")
        normalized = {
            _safe_path(path, "final_product.addition path"): _digest(
                file_digest, f"final_product.addition {path}"
            )
            for path, file_digest in self.additions.items()
        }
        deleted = tuple(sorted(_safe_path(path, "final_product.deletion path") for path in self.deletions))
        if tuple(normalized) != tuple(self.additions):
            object.__setattr__(self, "additions", dict(sorted(normalized.items())))
        if deleted != self.deletions:
            object.__setattr__(self, "deletions", deleted)
        expected_overlay = overlay_digest(self.additions, self.deletions)
        if expected_overlay != self.overlay_digest:
            raise FinalProductError("final_product.overlay_digest does not match additions/deletions")
        if self.certified and self.salvage:
            raise FinalProductError("salvage products cannot be certified")
        if self.certified and self.certification_digest is None:
            raise FinalProductError("certified products require certification_digest")
        if self.certification_digest is not None:
            _digest(self.certification_digest, "final_product.certification_digest")
        if self.candidate_boundary is not None:
            _safe_path(self.candidate_boundary, "final_product.candidate_boundary")

    @classmethod
    def from_trees(
        cls,
        *,
        parent_root: Path,
        final_root: Path,
        run_id: str,
        solver_run_id: str,
        task_id: str,
        task_bundle_digest: str,
        image_digest: str,
        release_digest: str,
        final_event_id: str,
        final_step: int,
        terminal_class: str,
        certified: bool,
        salvage: bool = False,
        candidate_boundary: str | None = None,
        certification_digest: str | None = None,
    ) -> "FinalProductManifest":
        parent = _hash_files(parent_root)
        final = _hash_files(final_root)
        additions = {path: digest for path, digest in final.items() if parent.get(path) != digest}
        deletions = tuple(sorted(path for path in parent if path not in final))
        return cls(
            run_id=run_id,
            solver_run_id=solver_run_id,
            task_id=task_id,
            task_bundle_digest=task_bundle_digest,
            image_digest=image_digest,
            release_digest=release_digest,
            parent_tree_digest=tree_digest(parent),
            final_event_id=final_event_id,
            final_step=final_step,
            additions=additions,
            deletions=deletions,
            overlay_digest=overlay_digest(additions, deletions),
            effective_product_digest=tree_digest(final),
            terminal_class=terminal_class,
            certified=certified,
            salvage=salvage,
            candidate_boundary=candidate_boundary,
            certification_digest=certification_digest,
        )

    @property
    def manifest_digest(self) -> str:
        return digest_value(self.to_dict(include_manifest_digest=False))

    def to_dict(self, *, include_manifest_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": FINAL_PRODUCT_SCHEMA,
            "run_id": self.run_id,
            "solver_run_id": self.solver_run_id,
            "task_id": self.task_id,
            "task_bundle_digest": self.task_bundle_digest,
            "image_digest": self.image_digest,
            "release_digest": self.release_digest,
            "parent_tree_digest": self.parent_tree_digest,
            "final_event_id": self.final_event_id,
            "final_step": self.final_step,
            "additions": dict(sorted(self.additions.items())),
            "deletions": list(self.deletions),
            "overlay_digest": self.overlay_digest,
            "effective_product_digest": self.effective_product_digest,
            "terminal_class": self.terminal_class,
            "certified": self.certified,
            "salvage": self.salvage,
            "candidate_boundary": self.candidate_boundary,
            "certification_digest": self.certification_digest,
        }
        if include_manifest_digest:
            value["manifest_digest"] = digest_value(value)
        return value

    @classmethod
    def from_dict(cls, value: object) -> "FinalProductManifest":
        if not isinstance(value, Mapping) or value.get("schema") != FINAL_PRODUCT_SCHEMA:
            raise FinalProductError("unknown final-product manifest schema")
        additions = value.get("additions")
        deletions = value.get("deletions")
        if not isinstance(additions, Mapping) or not isinstance(deletions, list):
            raise FinalProductError("final-product additions/deletions are malformed")
        manifest = cls(
            run_id=_text(value.get("run_id"), "final_product.run_id"),
            solver_run_id=_text(value.get("solver_run_id"), "final_product.solver_run_id"),
            task_id=_text(value.get("task_id"), "final_product.task_id"),
            task_bundle_digest=_digest(value.get("task_bundle_digest"), "final_product.task_bundle_digest"),
            image_digest=_digest(value.get("image_digest"), "final_product.image_digest"),
            release_digest=_digest(value.get("release_digest"), "final_product.release_digest"),
            parent_tree_digest=_digest(value.get("parent_tree_digest"), "final_product.parent_tree_digest"),
            final_event_id=_text(value.get("final_event_id"), "final_product.final_event_id"),
            final_step=_step(value.get("final_step"), "final_product.final_step"),
            additions={str(path): str(file_digest) for path, file_digest in additions.items()},
            deletions=tuple(_text(path, "final_product.deletion") for path in deletions),
            overlay_digest=_digest(value.get("overlay_digest"), "final_product.overlay_digest"),
            effective_product_digest=_digest(value.get("effective_product_digest"), "final_product.effective_product_digest"),
            terminal_class=_text(value.get("terminal_class"), "final_product.terminal_class"),
            certified=_boolean(value.get("certified"), "final_product.certified"),
            salvage=_boolean(value.get("salvage"), "final_product.salvage"),
            candidate_boundary=value.get("candidate_boundary") if value.get("candidate_boundary") is None else _text(value.get("candidate_boundary"), "final_product.candidate_boundary"),
            certification_digest=_optional_digest(value.get("certification_digest"), "final_product.certification_digest"),
        )
        stored_digest = value.get("manifest_digest")
        if stored_digest is not None and stored_digest != manifest.manifest_digest:
            raise FinalProductError("final-product manifest digest mismatch")
        return manifest


def read_manifest(path: str | Path) -> FinalProductManifest:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinalProductError(f"cannot read final-product manifest: {target}") from exc
    return FinalProductManifest.from_dict(value)


def write_manifest(path: str | Path, manifest: FinalProductManifest) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite final-product manifest: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def select_authoritative_terminal(records: list[Mapping[str, object]] | tuple[Mapping[str, object], ...]) -> Mapping[str, object]:
    """Select the latest typed terminal record, never the first favorable row."""

    if not records:
        raise FinalProductError("no terminal records were supplied")
    by_event: dict[str, Mapping[str, object]] = {}
    for record in records:
        event_id = _text(record.get("event_id"), "terminal.event_id")
        step = _step(record.get("step"), "terminal.step")
        terminal_class = _text(record.get("terminal_class"), "terminal.terminal_class")
        if terminal_class not in _TERMINAL_CLASSES:
            raise FinalProductError(f"terminal record has unknown class: {terminal_class}")
        prior = by_event.get(event_id)
        if prior is not None and dict(prior) != dict(record):
            raise FinalProductError(f"terminal event was reused with different contents: {event_id}")
        by_event[event_id] = dict(record) | {"step": step, "terminal_class": terminal_class}
    ordered = sorted(
        by_event.values(),
        key=lambda item: (
            _step(item.get("sequence", 0), "terminal.sequence"),
            _step(item.get("step"), "terminal.step"),
            _text(item.get("event_id"), "terminal.event_id"),
        ),
    )
    selected = dict(ordered[-1])
    selected["authoritative"] = True
    selected["terminal_digest"] = digest_value({key: value for key, value in selected.items() if key != "terminal_digest" and key != "authoritative"})
    return selected


def join_grade(manifest: FinalProductManifest, grade: Mapping[str, object]) -> dict[str, object]:
    """Join a grade only when it names the exact current product and solver run."""

    solver_run_id = grade.get("solver_run_id")
    if solver_run_id != manifest.solver_run_id:
        raise FinalProductError("grade solver_run_id does not match final product")
    if grade.get("run_id") != manifest.run_id:
        raise FinalProductError("grade run_id does not match final product")
    for field, expected in (
        ("task_id", manifest.task_id),
        ("task_bundle_digest", manifest.task_bundle_digest),
        ("image_digest", manifest.image_digest),
        ("release_digest", manifest.release_digest),
        ("parent_tree_digest", manifest.parent_tree_digest),
    ):
        if grade.get(field) != expected:
            raise FinalProductError(f"grade {field} does not match final product")
    if grade.get("effective_product_digest") != manifest.effective_product_digest:
        raise FinalProductError("grade effective product digest does not match final product")
    if grade.get("overlay_digest") != manifest.overlay_digest:
        raise FinalProductError("grade overlay digest does not match final product")
    if grade.get("terminal_event_id") != manifest.final_event_id:
        raise FinalProductError("grade terminal event does not match final product")
    if grade.get("certified") is True and not manifest.certified:
        raise FinalProductError("uncertified final product cannot receive certified grade")
    return {
        "schema": "sciledger/final-product-grade-join/1",
        "run_id": manifest.run_id,
        "solver_run_id": manifest.solver_run_id,
        "task_id": manifest.task_id,
        "terminal_event_id": manifest.final_event_id,
        "overlay_digest": manifest.overlay_digest,
        "effective_product_digest": manifest.effective_product_digest,
        "certified": bool(grade.get("certified") is True and manifest.certified),
        "salvage": manifest.salvage,
        "grade_claim": grade.get("claim", "UNKNOWN"),
    }


__all__ = [
    "FINAL_PRODUCT_MANIFEST_NAME",
    "FINAL_PRODUCT_SCHEMA",
    "FinalProductError",
    "FinalProductManifest",
    "canonical_bytes",
    "digest_bytes",
    "digest_value",
    "join_grade",
    "overlay_digest",
    "read_manifest",
    "select_authoritative_terminal",
    "tree_digest",
    "write_manifest",
]
