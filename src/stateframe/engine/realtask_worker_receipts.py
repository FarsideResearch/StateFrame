"""Typed, public-boundary receipts for real-task worker validation.

The benchmark controller knows the fixed revision and the saved parent world,
but the worker must receive only the public task row.  A receipt binds the
normalized runner result to a digest of that public row and to the logical
worker root (``parent`` or ``gold``).  It deliberately omits paths to
controller workspaces, subprocess output, source text, and fixed revisions.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from task_manifest import ManifestValidationError, validate_public_manifest


RECEIPT_SCHEMA = "realtask-worker-receipt/1"
ReceiptRoot: TypeAlias = Literal["parent", "gold"]

_ALLOWED_RUNNERS = frozenset({"pytest", "vitest", "eslint", "static"})
_ALLOWED_STATUSES = frozenset(
    {
        "PASS",
        "ASSERTION_FAIL",
        "NO_TESTS",
        "COLLECTION_ERROR",
        "HARNESS_ERROR",
        "INFRA_ERROR",
        "TIMEOUT",
        "UNKNOWN",
    }
)
_PRIVATE_KEYS = frozenset(
    {
        "fixed_src",
        "fixed_source",
        "gold_src",
        "reference_src",
        "reference_files",
        "reference_patch",
        "hidden_tests",
        "reference_selectors",
        "expected_answer",
        "required_plan_atoms",
        "oracle_digest",
        "private",
        "private_oracle",
        "gold_revision",
        "fix_revision",
        "parent_revision",
        "commit_subject",
        "commit_message",
        "patch",
        "diff",
    }
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ReceiptError(ValueError):
    """Raised when worker evidence cannot be safely attested."""


def _private_paths(value: object, prefix: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if name in _PRIVATE_KEYS:
                found.append(path)
            found.extend(_private_paths(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_private_paths(item, f"{prefix}[{index}]"))
    return tuple(found)


def _public_task_digest(row: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReceiptError("public task is not JSON serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"receipt {field} must be a non-empty string")
    return value


def _int_or_none(value: object, field: str) -> int | None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise ReceiptError(f"receipt {field} must be an integer or null")
    return value


def _selected(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ReceiptError("receipt selected must be a sequence of strings")
    selected = tuple(value)
    if any(not isinstance(item, str) or not item for item in selected):
        raise ReceiptError("receipt selected must be a sequence of strings")
    return selected


def _canonical_payload(
    *,
    task_id: str,
    root: ReceiptRoot,
    public_task_digest: str,
    runner: str,
    status: str,
    exit_code: int | None,
    collected: int,
    selected: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema": RECEIPT_SCHEMA,
        "task_id": task_id,
        "root": root,
        "public_task_digest": public_task_digest,
        "runner": runner,
        "status": status,
        "exit_code": exit_code,
        "collected": collected,
        "selected": list(selected),
    }


def _receipt_id(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_status_exit(status: str, exit_code: int | None) -> None:
    if status == "PASS" and exit_code != 0:
        raise ReceiptError("PASS worker result has contradictory exit_code")
    if status == "ASSERTION_FAIL" and exit_code == 0:
        raise ReceiptError("ASSERTION_FAIL worker result has contradictory exit_code")


@dataclass(frozen=True, slots=True)
class WorkerReceipt:
    """A deterministic controller-side attestation of one worker invocation."""

    schema: str
    task_id: str
    root: ReceiptRoot
    public_task_digest: str
    runner: str
    status: str
    exit_code: int | None
    collected: int
    selected: tuple[str, ...]
    receipt_id: str

    def __post_init__(self) -> None:
        if self.schema != RECEIPT_SCHEMA:
            raise ReceiptError("unsupported worker receipt schema")
        if not self.task_id:
            raise ReceiptError("receipt task_id must be a non-empty string")
        if self.root not in {"parent", "gold"}:
            raise ReceiptError("receipt root must be parent or gold")
        if not _DIGEST.fullmatch(self.public_task_digest):
            raise ReceiptError("receipt public_task_digest must be a SHA-256 digest")
        if self.runner not in _ALLOWED_RUNNERS:
            raise ReceiptError("receipt runner is not allowlisted")
        if self.status not in _ALLOWED_STATUSES:
            raise ReceiptError("receipt status is not supported")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise ReceiptError("receipt exit_code must be an integer or null")
        if not isinstance(self.collected, int) or isinstance(self.collected, bool) or self.collected < 0:
            raise ReceiptError("receipt collected must be a non-negative integer")
        if not isinstance(self.selected, tuple) or any(
            not isinstance(item, str) or not item for item in self.selected
        ):
            raise ReceiptError("receipt selected must be a tuple of strings")
        if not _DIGEST.fullmatch(self.receipt_id):
            raise ReceiptError("receipt_id must be a SHA-256 digest")
        _validate_status_exit(self.status, self.exit_code)
        payload = _canonical_payload(
            task_id=self.task_id,
            root=self.root,
            public_task_digest=self.public_task_digest,
            runner=self.runner,
            status=self.status,
            exit_code=self.exit_code,
            collected=self.collected,
            selected=self.selected,
        )
        if _receipt_id(payload) != self.receipt_id:
            raise ReceiptError("receipt_id does not match receipt contents")

    @property
    def parent_red(self) -> bool:
        return (
            self.root == "parent"
            and self.status == "ASSERTION_FAIL"
            and self.exit_code not in (None, 0)
            and self.collected > 0
        )

    @property
    def gold_green(self) -> bool:
        return (
            self.root == "gold"
            and self.status == "PASS"
            and self.exit_code == 0
            and self.collected > 0
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "root": self.root,
            "public_task_digest": self.public_task_digest,
            "runner": self.runner,
            "status": self.status,
            "exit_code": self.exit_code,
            "collected": self.collected,
            "selected": list(self.selected),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "WorkerReceipt":
        """Load a receipt and verify its content-addressed identity."""

        if not isinstance(value, Mapping):
            raise ReceiptError("worker receipt must be an object")
        leaked = _private_paths(value)
        if leaked:
            raise ReceiptError(f"worker receipt contains private field: {leaked[0]}")
        schema = _text(value.get("schema"), "schema")
        if schema != RECEIPT_SCHEMA:
            raise ReceiptError("unsupported worker receipt schema")
        task_id = _text(value.get("task_id"), "task_id")
        root = value.get("root")
        if root not in {"parent", "gold"}:
            raise ReceiptError("receipt root must be parent or gold")
        public_task_digest = _text(value.get("public_task_digest"), "public_task_digest")
        if not _DIGEST.fullmatch(public_task_digest):
            raise ReceiptError("receipt public_task_digest must be a SHA-256 digest")
        runner = _text(value.get("runner"), "runner")
        if runner not in _ALLOWED_RUNNERS:
            raise ReceiptError("receipt runner is not allowlisted")
        status = _text(value.get("status"), "status")
        if status not in _ALLOWED_STATUSES:
            raise ReceiptError("receipt status is not supported")
        exit_code = _int_or_none(value.get("exit_code"), "exit_code")
        collected = value.get("collected")
        if not isinstance(collected, int) or isinstance(collected, bool) or collected < 0:
            raise ReceiptError("receipt collected must be a non-negative integer")
        selected = _selected(value.get("selected"))
        receipt_id = _text(value.get("receipt_id"), "receipt_id")
        if not _DIGEST.fullmatch(receipt_id):
            raise ReceiptError("receipt_id must be a SHA-256 digest")
        _validate_status_exit(status, exit_code)
        payload = _canonical_payload(
            task_id=task_id,
            root=root,
            public_task_digest=public_task_digest,
            runner=runner,
            status=status,
            exit_code=exit_code,
            collected=collected,
            selected=selected,
        )
        if _receipt_id(payload) != receipt_id:
            raise ReceiptError("receipt_id does not match receipt contents")
        return cls(
            schema=schema,
            task_id=task_id,
            root=root,
            public_task_digest=public_task_digest,
            runner=runner,
            status=status,
            exit_code=exit_code,
            collected=collected,
            selected=selected,
            receipt_id=receipt_id,
        )

    def rebind_public_task(self, row: Mapping[str, object]) -> "WorkerReceipt":
        """Bind this same worker evidence to a final public task row.

        Extraction validates a candidate before assigning its stable ``rt-dev``
        id.  Rebinding changes only the task id and public-row digest; runner
        status, exit code, collection count, and selectors remain immutable.
        """

        if not isinstance(row, Mapping):
            raise ReceiptError("public task must be an object")
        leaked = _private_paths(row)
        if leaked:
            raise ReceiptError(f"public task contains private field: {leaked[0]}")
        try:
            manifest = validate_public_manifest(row)
        except ManifestValidationError as exc:
            raise ReceiptError(f"public task is invalid: {exc}") from exc
        if manifest.runner != self.runner or tuple(manifest.selectors) != self.selected:
            raise ReceiptError("public task changed runner or selectors during rebind")
        task_id = _text(row.get("task_id"), "task_id")
        payload = _canonical_payload(
            task_id=task_id,
            root=self.root,
            public_task_digest=_public_task_digest(row),
            runner=self.runner,
            status=self.status,
            exit_code=self.exit_code,
            collected=self.collected,
            selected=self.selected,
        )
        return WorkerReceipt(
            schema=RECEIPT_SCHEMA,
            task_id=task_id,
            root=self.root,
            public_task_digest=str(payload["public_task_digest"]),
            runner=self.runner,
            status=self.status,
            exit_code=self.exit_code,
            collected=self.collected,
            selected=self.selected,
            receipt_id=_receipt_id(payload),
        )


def make_worker_receipt(
    row: Mapping[str, object], root: str, result: object
) -> WorkerReceipt:
    """Normalize one allowlisted worker result against the public task row."""

    if not isinstance(row, Mapping):
        raise ReceiptError("public task must be an object")
    leaked = _private_paths(row)
    if leaked:
        raise ReceiptError(f"public task contains private field: {leaked[0]}")
    try:
        manifest = validate_public_manifest(row)
    except ManifestValidationError as exc:
        raise ReceiptError(f"public task is invalid: {exc}") from exc
    if root not in {"parent", "gold"}:
        raise ReceiptError("receipt root must be parent or gold")
    task_id = _text(row.get("task_id"), "task_id")
    runner = _text(getattr(result, "runner", None), "runner")
    if runner not in _ALLOWED_RUNNERS or runner != manifest.runner:
        raise ReceiptError("worker result runner does not match public task runner")
    status = _text(getattr(result, "status", None), "status")
    if status not in _ALLOWED_STATUSES:
        raise ReceiptError("worker result status is not supported")
    exit_code = _int_or_none(getattr(result, "exit_code", None), "exit_code")
    collected = getattr(result, "collected", None)
    if not isinstance(collected, int) or isinstance(collected, bool) or collected < 0:
        raise ReceiptError("worker result collected must be a non-negative integer")
    selected = _selected(getattr(result, "selected", None))
    expected = tuple(manifest.selectors)
    if selected != expected:
        raise ReceiptError("worker result selected tests do not match public selectors")
    _validate_status_exit(status, exit_code)
    payload = _canonical_payload(
        task_id=task_id,
        root=root,  # type: ignore[arg-type]
        public_task_digest=_public_task_digest(row),
        runner=runner,
        status=status,
        exit_code=exit_code,
        collected=collected,
        selected=selected,
    )
    return WorkerReceipt(
        schema=RECEIPT_SCHEMA,
        task_id=task_id,
        root=root,  # type: ignore[arg-type]
        public_task_digest=str(payload["public_task_digest"]),
        runner=runner,
        status=status,
        exit_code=exit_code,
        collected=collected,
        selected=selected,
        receipt_id=_receipt_id(payload),
    )


__all__ = ["RECEIPT_SCHEMA", "ReceiptError", "WorkerReceipt", "make_worker_receipt"]
