"""Truth-boundary contracts for the plain-loop engine (STATEFRAME_AGENT_LOOP=plain).

Stdlib only. Owns the invariants the plain engine keeps at its boundary:
edit authority (declared product files only; new files are scratch), exact-once
replacement, revision-bound read/edit/check receipts, completion truth, tool
call/result pairing, and an append-only journal. Nothing here talks to the
model or changes what it sees; receipts are host-side records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time

ENGINE_ID = "stateframe-plain-loop/1"
ROOT_ALIAS = "/workspace"


class PlainContractError(ValueError):
    """A boundary invariant failed; never converted into a success."""


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_revision(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except FileNotFoundError:
        return "absent"


class EditAuthority:
    """Declared product files are editable and exported; new files are scratch.

    Existing files outside the declared set are refused: a plain ``write_file``
    is not permission to overwrite repository content the task did not declare.
    """

    def __init__(self, root: Path, declared: Sequence[str]):
        self.root = Path(root).resolve()
        clean = []
        for raw in declared:
            relative = self.relative(raw)
            if relative == "." or relative in clean:
                raise PlainContractError(f"invalid or duplicate declared path: {raw!r}")
            clean.append(relative)
        if not clean:
            raise PlainContractError("plain engine requires at least one declared product file")
        self.declared = tuple(sorted(clean))
        self.scratch_created: set[str] = set()

    def resolve(self, raw: object) -> Path:
        """Baseline path semantics: strip the /workspace alias, stay inside root."""
        text = str(raw or ".").removeprefix(ROOT_ALIAS).lstrip("/") or "."
        path = (self.root / text).resolve()
        if path != self.root and self.root not in path.parents:
            raise ValueError("path escapes /workspace")
        return path

    def relative(self, raw: object) -> str:
        path = self.resolve(raw)
        return "." if path == self.root else path.relative_to(self.root).as_posix()

    def classify(self, raw: object) -> tuple[str, Path, str]:
        """Return (scope, path, relative): product | scratch | refused."""
        path = self.resolve(raw)
        relative = "." if path == self.root else path.relative_to(self.root).as_posix()
        if relative in self.declared:
            return "product", path, relative
        if relative in self.scratch_created or not path.exists():
            return "scratch", path, relative
        return "refused", path, relative

    def refusal(self, relative: str) -> str:
        return (f"error: {relative} is not editable; only these existing files may be changed: "
                + ", ".join(self.declared))

    def product_revision(self) -> str:
        digest = hashlib.sha256()
        for relative in self.declared:
            data = (self.root / relative).read_bytes() if (self.root / relative).is_file() else b"\0absent"
            digest.update(relative.encode() + b"\0" + hashlib.sha256(data).digest())
        return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class ReadReceipt:
    path: str
    revision: str
    requested: tuple[int | None, int | None]
    returned: tuple[int, int]
    total_lines: int
    widened: bool
    chars: int
    truncated_by_cap: bool


@dataclass(frozen=True)
class EditReceipt:
    tool: str
    path: str
    scope: str
    applied: bool
    old_revision: str
    new_revision: str
    product_revision: str
    occurrences: int | None
    reason: str = ""


@dataclass(frozen=True)
class CheckReceipt:
    argv: tuple[str, ...]
    selectors: tuple[str, ...]
    product_revision: str
    exit_code: int | None
    timed_out: bool
    synthetic: bool
    output_sha256: str
    output_chars: int
    error: str = ""

    @property
    def public_green(self) -> bool:
        return (self.exit_code == 0 and not self.timed_out and not self.synthetic
                and not self.error and bool(self.selectors))


@dataclass
class ReceiptBook:
    """Revision-bound facts. A green check for revision N never certifies N+1."""

    reads: int = 0
    product_edits: int = 0
    scratch_edits: int = 0
    refused_edits: int = 0
    checks: list[CheckReceipt] = field(default_factory=list)

    def latest_green(self) -> CheckReceipt | None:
        return next((check for check in reversed(self.checks) if check.public_green), None)

    def completion(self, current_revision: str) -> dict[str, object]:
        green = self.latest_green()
        if green is not None and green.product_revision == current_revision:
            return {"status": "VERIFIED_PUBLIC_GREEN", "revision": current_revision,
                    "selectors": list(green.selectors),
                    "limitation": "public checks only; task requirements beyond them are unverified"}
        if green is not None:
            reason = "latest public green check is for an older product revision"
        elif any(check.synthetic for check in self.checks):
            reason = "only synthetic (replay) check receipts exist"
        else:
            reason = "no public check passed"
        return {"status": "UNVERIFIED", "revision": current_revision, "reason": reason}


def validate_pairing(messages: Sequence[Mapping[str, object]]) -> None:
    """Every assistant tool call is answered, in order, before the next turn."""
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            if message.get("role") == "tool":
                raise PlainContractError(f"orphan tool result at message {index - 1}")
            continue
        for call in calls:
            if index >= len(messages) or messages[index].get("role") != "tool":
                raise PlainContractError(f"unanswered tool call at message {index}")
            if messages[index].get("tool_call_id", "") != (call.get("id") or ""):
                raise PlainContractError(f"tool result id mismatch at message {index}")
            index += 1


def validate_schema_agreement(tools: Sequence[Mapping[str, object]], supported: set[str]) -> None:
    names = [tool["function"]["name"] for tool in tools]
    if len(set(names)) != len(names):
        raise PlainContractError("duplicate tool schema names")
    missing = set(names) - supported - {"done"}
    if missing:
        raise PlainContractError(f"tool schemas without a dispatcher: {sorted(missing)}")


class PlainJournal:
    """Append-only event log; ``sequence`` is shared by every record kind."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "events.jsonl"
        self.sequence = 0

    def write(self, kind: str, **fields: object) -> int:
        self.sequence += 1
        record = {"kind": kind, "sequence": self.sequence, "timestamp_ns": time.time_ns(), **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return self.sequence

    def receipt(self, kind: str, receipt: object) -> int:
        return self.write(kind, receipt=asdict(receipt))

    def save(self, name: str, value: object) -> Path:
        path = self.directory / name
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, default=_jsonable)
            stream.write("\n")
        return path


def _jsonable(value: object) -> object:
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


__all__ = [
    "ENGINE_ID", "ROOT_ALIAS", "CheckReceipt", "EditAuthority", "EditReceipt", "PlainContractError",
    "PlainJournal", "ReadReceipt", "ReceiptBook", "file_revision", "sha256_bytes", "validate_pairing",
    "validate_schema_agreement",
]
