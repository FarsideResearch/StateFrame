"""Opt-in real-task benchmark extraction and worker validation.

This module is deliberately separate from the StateFrame kernel.  It turns a
single-parent git change into a public, symptom-only task while keeping the
fixed revision in a controller-only :class:`TaskPrivate` object.  The worker
validator materializes the parent and fixed revisions independently, overlays
the public tests, and only hands the public row to the test runner.

The feature is off unless ``STATEFRAME_REAL_TASK_BENCHMARK=on``.  Extraction
and validation can be used offline by calling the functions or the ``build``
CLI; the runtime hook only presents a task's public symptom ticket.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from task_manifest import ManifestValidationError, validate_public_manifest
from realtask_worker_receipts import WorkerReceipt, make_worker_receipt


MODE_ENV = "STATEFRAME_REAL_TASK_BENCHMARK"
SCHEMA = "realtask-benchmark/1"
VALIDATION_SCHEMA = "realtask-validation/2"
CONTROLLER_SCHEMA = "realtask-controller/1"
PENDING_SCHEMA = "realtask-pending/1"
TICKET_FIELDS = ("trigger", "observed", "expected")
TICKET_HEADER = "[Real-task symptom v1]"

_DOC_DIRS = frozenset({"doc", "docs", "documentation"})
_DOC_ROOT_NAMES = frozenset(
    {
        "authors",
        "changelog",
        "code_of_conduct",
        "contributing",
        "license",
        "readme",
        "security",
    }
)
_DOC_SUFFIXES = frozenset({".adoc", ".markdown", ".md", ".rst"})
_SOURCE_SUFFIXES = frozenset(
    {".c", ".cc", ".cpp", ".go", ".java", ".js", ".jsx", ".mjs", ".py", ".rb", ".rs", ".ts", ".tsx", ".vue"}
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
_HEX_TOKEN = re.compile(r"(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])", re.IGNORECASE)


class BenchmarkError(ValueError):
    """Raised when a candidate cannot satisfy the public task contract."""


def mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the strict runtime mode, defaulting to the byte-preserving OFF path."""

    environment = os.environ if environ is None else environ
    value = environment.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def is_ticket_block(content: object) -> bool:
    return str(content or "").startswith(TICKET_HEADER)


def render_symptom_ticket(symptom: str) -> str:
    clean = " ".join(str(symptom).split())
    if not clean:
        raise BenchmarkError("symptom ticket must not be empty")
    return f"{TICKET_HEADER}\n{clean}"


def append_symptom_ticket(
    messages: Sequence[dict[str, object]],
    task_row: Mapping[str, object] | None,
    *,
    provider_request: bool = True,
    enabled_value: bool | None = None,
) -> list[dict[str, object]]:
    """Append the public symptom to a provider request when the switch is ON."""

    on = enabled() if enabled_value is None else enabled_value
    if not on or not provider_request or not isinstance(task_row, Mapping):
        return list(messages)
    symptom = task_row.get("symptom")
    if not isinstance(symptom, str) or not symptom.strip():
        return list(messages)
    if any(is_ticket_block(message.get("content")) for message in messages):
        return list(messages)
    prepared = list(messages)
    prepared.append({"role": "user", "content": render_symptom_ticket(symptom)})
    return prepared


def _normal_path(value: str) -> str:
    return value.replace("\\", "/").strip("/")


def _safe_relative_path(value: str) -> str:
    if not value or "\x00" in value:
        raise BenchmarkError("archive member path is empty or contains NUL")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        raise BenchmarkError(f"archive member is absolute: {value!r}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise BenchmarkError(f"archive member escapes the worker root: {value!r}")
    return "/".join(parts)


def is_documentation_path(path: str) -> bool:
    """Identify checked-in docs without treating source comments as docs."""

    normalized = _normal_path(path)
    if not normalized:
        return False
    parts = normalized.split("/")
    lowered = [part.lower() for part in parts]
    if any(part in _DOC_DIRS for part in lowered):
        return True
    basename = lowered[-1]
    stem = Path(basename).stem
    if len(parts) == 1 and stem in _DOC_ROOT_NAMES:
        return True
    return Path(basename).suffix.lower() in _DOC_SUFFIXES


class GitCommandError(BenchmarkError):
    """A git command failed while constructing a reproducible snapshot."""


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
        }
    )
    return environment


def _git_bytes(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        env=_git_environment(),
        check=False,
    )
    if completed.returncode:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise GitCommandError(f"git {' '.join(args)} failed: {stderr[:240]}")
    return completed.stdout


def _git_text(repo: Path, *args: str) -> str:
    return _git_bytes(repo, *args).decode("utf-8")


def _tree_ish(revision: str, subtree: str = "") -> str:
    """``rev`` or ``rev:subtree`` -- a subtree task's world is that directory."""
    return f"{revision}:{_safe_relative_path(subtree)}" if subtree else revision


def _git_file(repo: Path, revision: str, path: str, subtree: str = "") -> str | None:
    full = f"{_safe_relative_path(subtree)}/{path}" if subtree else path
    try:
        value = _git_bytes(repo, "show", f"{revision}:{full}")
    except GitCommandError:
        return None
    if b"\x00" in value:
        return None
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return None


@dataclass(frozen=True, slots=True)
class ArchiveSnapshot:
    """Controller-side facts for one docs-stripped git archive."""

    revision: str
    archive_digest: str
    tree_digest: str
    file_count: int
    stripped_paths: tuple[str, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "schema": "git-archive/v1",
            "tree_digest": self.tree_digest,
            "docs_stripped": True,
            "file_count": self.file_count,
        }


def _safe_destination(root: Path, relative: str) -> Path:
    safe = _safe_relative_path(relative)
    target = root.joinpath(*safe.split("/"))
    resolved_root = root.resolve()
    resolved_target = target.resolve(strict=False)
    if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
        raise BenchmarkError(f"archive member resolves outside worker root: {relative!r}")
    return target


def _safe_link_target(value: str) -> str:
    if not value or value.startswith("/") or "\x00" in value:
        raise BenchmarkError("archive link target is unsafe")
    normalized = value.replace("\\", "/")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise BenchmarkError("archive link target escapes worker root")
    return normalized


def _tree_digest(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    entries = sorted(root.rglob("*"), key=lambda item: item.as_posix())
    for item in entries:
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            files += 1
            digest.update(f"F:{relative}\0".encode())
            digest.update(f"L:{os.readlink(item)}\0".encode())
        elif item.is_dir():
            digest.update(f"D:{relative}\0".encode())
        else:
            files += 1
            digest.update(f"F:{relative}\0".encode())
            digest.update(item.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest(), files


def git_archive_snapshot(
    repo: str | Path,
    revision: str,
    destination: str | Path,
    *,
    strip_docs: bool = True,
) -> ArchiveSnapshot:
    """Materialize a safe ``git archive`` snapshot and hash its resulting tree."""

    repository = Path(repo).resolve()
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise BenchmarkError(f"snapshot destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    archive = _git_bytes(repository, "archive", "--format=tar", revision)
    stripped: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        for member in stream.getmembers():
            name = member.name.rstrip("/")
            if not name:
                continue
            safe_name = _safe_relative_path(name)
            if strip_docs and is_documentation_path(safe_name):
                stripped.append(safe_name)
                continue
            target = _safe_destination(root, safe_name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise BenchmarkError(f"archive member has no contents: {safe_name}")
                target.write_bytes(source.read())
                target.chmod(member.mode & 0o777)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(_safe_link_target(member.linkname))
            else:
                raise BenchmarkError(f"unsupported archive member type: {safe_name}")
    tree_digest, file_count = _tree_digest(root)
    return ArchiveSnapshot(
        revision=revision,
        archive_digest=hashlib.sha256(archive).hexdigest(),
        tree_digest=tree_digest,
        file_count=file_count,
        stripped_paths=tuple(sorted(set(stripped))),
    )


def _write_snapshot_tar(root: Path, destination: Path) -> None:
    """Deterministic tar of a stripped tree (sorted, zero mtime/owner)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as archive:
        for item in sorted(root.rglob("*"), key=lambda path: path.as_posix()):
            info = archive.gettarinfo(str(item), arcname=item.relative_to(root).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            if info.isreg():
                with item.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    partial.replace(destination)


def persist_snapshot(root: Path, snapshot_dir: str | Path, tree_digest: str) -> Path:
    """Store a stripped parent world under its tree digest (idempotent)."""
    target = Path(snapshot_dir) / f"{tree_digest}.tar"
    if not target.exists():
        _write_snapshot_tar(root, target)
    return target


def materialize_task_world(row: Mapping[str, object], snapshot_dir: str | Path,
                           destination: str | Path) -> Path:
    """Rebuild a task's full parent world from the saved snapshot, verify its
    digest against the public ``base_digest``, then overlay the public tests."""
    digest = row.get("base_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise BenchmarkError("task row has no base_digest")
    source = Path(snapshot_dir) / f"{digest}.tar"
    if not source.is_file() or source.is_symlink():
        raise BenchmarkError(f"saved snapshot missing for {digest}")
    root = Path(destination)
    if root.exists() and any(root.iterdir()):
        raise BenchmarkError(f"task world destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:") as stream:
        for member in stream.getmembers():
            target = _safe_destination(root, _safe_relative_path(member.name))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                handle = stream.extractfile(member)
                target.write_bytes(handle.read() if handle else b"")
                target.chmod(member.mode & 0o777)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(_safe_link_target(member.linkname))
            else:
                raise BenchmarkError(f"unsupported snapshot member: {member.name}")
    rebuilt, _count = _tree_digest(root)
    if rebuilt != digest:
        raise BenchmarkError("saved snapshot does not reproduce base_digest")
    for item in row.get("public_tests", ()):
        if not isinstance(item, Mapping):
            raise BenchmarkError("public test is not an object")
        _safe_overlay(root, str(item["path"]), str(item["source"]))
    return root


@dataclass(frozen=True, slots=True)
class TaskPrivate:
    """Controller-only provenance.  Never serialize this object to a worker."""

    parent_revision: str
    gold_revision: str
    commit_subject: str
    changed_paths: tuple[str, ...]
    diff_text: str
    subtree: str = ""
    repository: str = ""

    def controller_record(self, task_id: str, tree_digest: str) -> dict[str, object]:
        """Controller-only provenance to revalidate without re-mining a moving HEAD."""
        return {"schema": CONTROLLER_SCHEMA, "task_id": task_id,
                "repository": self.repository, "subtree": self.subtree,
                "parent_revision": self.parent_revision,
                "gold_revision": self.gold_revision, "base_digest": tree_digest}


@dataclass(frozen=True, slots=True)
class TaskBundle:
    """Public task plus controller-only fixed-revision provenance."""

    public: dict[str, object]
    private: TaskPrivate

    def public_payload(self) -> dict[str, object]:
        return json.loads(json.dumps(self.public))


@dataclass(frozen=True, slots=True)
class LeakageCheck:
    ok: bool
    findings: tuple[str, ...]


def _all_strings(value: object, prefix: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.append((prefix, value))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_all_strings(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_all_strings(item, f"{prefix}[{index}]"))
    return found


def _private_key_paths(value: object, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            path = f"{prefix}.{name}" if prefix else name
            if name in _PRIVATE_KEYS:
                found.append(path)
            found.extend(_private_key_paths(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(_private_key_paths(item, f"{prefix}[{index}]"))
    return found


def leakage_check(bundle: TaskBundle) -> LeakageCheck:
    """Reject private fields and provenance tokens in the public prompt surface."""

    findings: list[str] = []
    public = bundle.public
    findings.extend(
        f"private field exposed: {key}" for key in _private_key_paths(public)
    )
    prompt_fields = {
        key: value
        for key, value in _all_strings(public)
        if key.endswith("symptom") or key.endswith("ticket")
    }
    private = bundle.private
    private_tokens = [private.parent_revision, private.gold_revision]
    private_tokens.extend(part for part in _HEX_TOKEN.findall(private.diff_text))
    if private.commit_subject.strip():
        private_tokens.append(private.commit_subject.strip())
    for path in private.changed_paths:
        private_tokens.extend((path, Path(path).name))
    for field, text in prompt_fields.items():
        lowered = text.casefold()
        if private.commit_subject.strip().casefold() in lowered:
            findings.append(f"commit subject leaked in {field}")
        for token in private_tokens:
            if token and token.casefold() in lowered:
                findings.append(f"private provenance token leaked in {field}")
                break
        if _HEX_TOKEN.search(text):
            findings.append(f"revision-like token leaked in {field}")
    return LeakageCheck(ok=not findings, findings=tuple(dict.fromkeys(findings)))


def _test_path(path: str) -> bool:
    normalized = _normal_path(path)
    lower = normalized.lower()
    name = lower.rsplit("/", 1)[-1]
    return (
        "/tests/" in f"/{lower}/"
        or name.startswith("test_")
        or ".test." in name
        or ".spec." in name
    ) and Path(name).suffix in _SOURCE_SUFFIXES


def _product_path(path: str) -> bool:
    normalized = _normal_path(path)
    if is_documentation_path(normalized) or _test_path(normalized):
        return False
    return Path(normalized).suffix.lower() in _SOURCE_SUFFIXES


def _runner_for(paths: Sequence[str]) -> str:
    return "vitest" if any(Path(path).suffix.lower() in {".js", ".jsx", ".mjs", ".ts", ".tsx", ".vue"} for path in paths) else "pytest"


def symptom_from_ticket(ticket: object) -> str:
    """A reviewed behavioral ticket -> the public symptom (fix round, review #3).

    Keyword-generated tickets were removed: they misdescribed tasks and left the
    tests as the effective specification.  Every accepted task needs an
    explicitly reviewed ticket naming the trigger, the observed failure and the
    expected behavior; there is no generic fallback.
    """
    if not isinstance(ticket, Mapping):
        raise BenchmarkError("reviewed ticket must be an object")
    values = {}
    for field in TICKET_FIELDS:
        value = ticket.get(field)
        if not isinstance(value, str) or len(" ".join(value.split())) < 12:
            raise BenchmarkError(f"reviewed ticket needs a specific {field!r}")
        values[field] = " ".join(value.split()).rstrip(".")
    if ticket.get("reviewed") is not True:
        raise BenchmarkError("ticket was not marked reviewed")
    return (f"Trigger: {values['trigger']}. Observed: {values['observed']}. "
            f"Expected: {values['expected']}.")


def _ticket_for(tickets: Mapping[str, object], resolved: str) -> object | None:
    matches = [value for key, value in tickets.items()
               if len(str(key)) >= 7 and resolved.startswith(str(key))]
    if len(matches) > 1:
        raise BenchmarkError("ambiguous reviewed-ticket revision key")
    return matches[0] if matches else None


def _commit_details(
    repo: Path, revision: str, subtree: str = ""
) -> tuple[str, str, tuple[str, ...], str, str]:
    line = _git_text(repo, "rev-list", "--parents", "-n", "1", revision).strip()
    parts = line.split()
    if len(parts) != 2:
        raise BenchmarkError("candidate must have exactly one parent")
    resolved, parent = parts
    everything = tuple(
        item
        for item in _git_bytes(repo, "diff", "--name-only", "-z", parent, resolved).decode("utf-8").split("\x00")
        if item
    )
    if subtree:
        prefix = _safe_relative_path(subtree) + "/"
        if any(not path.startswith(prefix) for path in everything):
            # A change that also edits files outside the task world is not
            # reproducible inside that world; never half-extract it.
            raise BenchmarkError("candidate changes files outside the subtree")
        paths = tuple(path[len(prefix):] for path in everything)
    else:
        paths = everything
    subject = _git_text(repo, "show", "-s", "--format=%s", resolved).strip()
    diff = _git_text(repo, "diff", "--no-ext-diff", "--unified=0", parent, resolved, "--")
    return resolved, parent, subject, paths, diff


def _public_task(
    task_id: str,
    *,
    base_snapshot: ArchiveSnapshot,
    files: Sequence[tuple[str, str]],
    public_tests: Sequence[tuple[str, str]],
    runner: str,
    requires_db: bool,
    symptom: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "schema": SCHEMA,
        "split": "dev",
        "task_id": task_id,
        "track": "javascript" if runner == "vitest" else "python",
        "rung": "dev",
        "world": "git-archive",
        "base_digest": base_snapshot.tree_digest,
        "snapshot": base_snapshot.public_dict(),
        "files": [{"path": path, "source": source} for path, source in files],
        "public_tests": [
            {"path": path, "source": source} for path, source in public_tests
        ],
        "runner": runner,
        "requires_db": requires_db,
        "selectors": [path for path, _source in public_tests],
        "allowed_product_paths": sorted(
            {
                path.rsplit("/", 1)[0]
                for path, _source in files
                if "/" in path
            }
        ),
        "symptom": symptom,
    }
    return row


def validate_public_task(row: Mapping[str, object]) -> None:
    """Validate one public row without accepting any controller-only fields."""

    leaked = _private_key_paths(row)
    if leaked:
        raise BenchmarkError(
            f"public task contains private field(s): {', '.join(leaked)}"
        )
    if row.get("split") != "dev":
        raise BenchmarkError("only dev rows may be emitted by this builder")
    symptom = row.get("symptom")
    if not isinstance(symptom, str) or not symptom.strip():
        raise BenchmarkError("public task needs a non-empty symptom")
    try:
        validate_public_manifest(row)
    except ManifestValidationError as exc:
        raise BenchmarkError(str(exc)) from exc


def extract_dev_tasks(
    repo: str | Path,
    revisions: Sequence[str] | None = None,
    *,
    limit: int = 5,
    requires_db: bool = False,
    subtree: str = "",
    tickets: Mapping[str, object] | None = None,
    snapshot_dir: str | Path | None = None,
) -> tuple[TaskBundle, ...]:
    """Extract up to ``limit`` single-parent source+test changes as dev CANDIDATES.

    A candidate's symptom comes only from a reviewed ticket (``tickets``, keyed
    by revision prefix).  Candidates without one are still extracted (so they
    can be validated and reviewed) with ``symptom == ""``; ``build_dev_tasks``
    never publishes them.  With ``snapshot_dir`` the stripped parent world is
    persisted under its tree digest.
    """

    if limit <= 0:
        raise BenchmarkError("limit must be positive")
    repository = Path(repo).resolve()
    tickets = dict(tickets or {})
    candidates = list(revisions or ())
    if not candidates:
        log_args = ["rev-list", "--first-parent", "--max-count", str(max(limit * 8, limit)), "HEAD"]
        if subtree:
            log_args += ["--", _safe_relative_path(subtree)]
        candidates = _git_text(repository, *log_args).splitlines()
    bundles: list[TaskBundle] = []
    for revision in candidates:
        if len(bundles) >= limit:
            break
        try:
            resolved, parent, subject, changed_paths, diff_text = _commit_details(
                repository, revision, subtree)
        except (BenchmarkError, UnicodeDecodeError):
            continue
        product_paths = tuple(path for path in changed_paths if _product_path(path))
        test_paths = tuple(path for path in changed_paths if _test_path(path))
        if not product_paths or not test_paths:
            continue
        files = [(path, _git_file(repository, parent, path, subtree)) for path in product_paths]
        public_tests = [(path, _git_file(repository, resolved, path, subtree)) for path in test_paths]
        if any(source is None for _path, source in (*files, *public_tests)):
            continue
        ticket = _ticket_for(tickets, resolved)
        symptom = symptom_from_ticket(ticket) if ticket is not None else ""
        task_id = f"rt-cand-{len(bundles) + 1:03d}"
        with tempfile.TemporaryDirectory(prefix="realtask-snapshot-") as temp_name:
            base = Path(temp_name) / "base"
            snapshot = git_archive_snapshot(repository, _tree_ish(parent, subtree), base)
            if snapshot_dir is not None:
                persist_snapshot(base, snapshot_dir, snapshot.tree_digest)
        public = _public_task(
            task_id,
            base_snapshot=snapshot,
            files=files,
            public_tests=public_tests,
            runner=_runner_for((*product_paths, *test_paths)),
            requires_db=requires_db,
            symptom=symptom,
        )
        bundle = TaskBundle(
            public=public,
            private=TaskPrivate(
                parent_revision=parent,
                gold_revision=resolved,
                commit_subject=subject,
                changed_paths=changed_paths,
                diff_text=diff_text,
                subtree=subtree,
                repository=str(repository),
            ),
        )
        if symptom:
            checked = leakage_check(bundle)
            if not checked.ok:
                raise BenchmarkError(
                    f"{task_id} failed leakage check: {'; '.join(checked.findings)}")
            validate_public_task(public)
        bundles.append(bundle)
    return tuple(bundles)


WorkerRunner: TypeAlias = Callable[[Mapping[str, object], Path], Any]
ReceiptPublisher: TypeAlias = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class RealTaskWorkerValidation:
    task_id: str
    valid: bool
    parent_status: str
    parent_collected: int
    gold_status: str
    gold_collected: int
    blocker: str | None = None
    parent_receipt: WorkerReceipt | None = None
    gold_receipt: WorkerReceipt | None = None

    def __post_init__(self) -> None:
        receipts = (self.parent_receipt, self.gold_receipt)
        if any(receipt is not None and receipt.task_id != self.task_id for receipt in receipts):
            raise BenchmarkError("worker receipt task id does not match validation task")
        if self.parent_receipt is not None:
            if (
                self.parent_receipt.status != self.parent_status
                or self.parent_receipt.collected != self.parent_collected
                or self.parent_receipt.root != "parent"
            ):
                raise BenchmarkError("parent receipt does not match validation summary")
        if self.gold_receipt is not None:
            if (
                self.gold_receipt.status != self.gold_status
                or self.gold_receipt.collected != self.gold_collected
                or self.gold_receipt.root != "gold"
            ):
                raise BenchmarkError("gold receipt does not match validation summary")
        if self.parent_receipt is not None and self.gold_receipt is not None:
            if self.parent_receipt.public_task_digest != self.gold_receipt.public_task_digest:
                raise BenchmarkError("parent and gold receipts bind different public tasks")
        if self.valid and not (
            self.parent_receipt is not None
            and self.gold_receipt is not None
            and self.parent_receipt.parent_red
            and self.gold_receipt.gold_green
        ):
            raise BenchmarkError("valid worker validation requires red parent and green gold receipts")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": VALIDATION_SCHEMA,
            "task_id": self.task_id,
            "valid": self.valid,
            "parent": {"status": self.parent_status, "collected": self.parent_collected},
            "gold": {"status": self.gold_status, "collected": self.gold_collected},
            "parent_receipt": (
                self.parent_receipt.as_dict() if self.parent_receipt is not None else None
            ),
            "gold_receipt": (
                self.gold_receipt.as_dict() if self.gold_receipt is not None else None
            ),
            "blocker": self.blocker,
        }


def _safe_overlay(root: Path, path: str, source: str) -> None:
    target = _safe_destination(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")


def _default_worker_runner(
    row: Mapping[str, object], root: Path, *, timeout: float
) -> Any:
    from task_runners import run_task_tests

    # Worker setup: a repository's own pytest addopts (e.g. a coverage floor
    # meant for the full suite) must not turn a passing selected test file into
    # a failure verdict.  Only the pass/fail of the selected public tests counts.
    return run_task_tests(row, root, db_guard=lambda: True, timeout=timeout,
                          env={"PYTEST_ADDOPTS": "-o addopts="})


def _validate_in_root(
    bundle: TaskBundle,
    repository: Path,
    root: Path,
    *,
    worker_runner: WorkerRunner | None,
    timeout: float,
    snapshot_dir: str | Path | None = None,
) -> RealTaskWorkerValidation:
    base_root = root / "parent"
    gold_root = root / "gold"
    private = bundle.private
    if snapshot_dir is not None:
        # The parent world the worker gets is the SAVED snapshot (round trip).
        materialize_task_world(bundle.public, snapshot_dir, base_root)
    else:
        base_snapshot = git_archive_snapshot(
            repository, _tree_ish(private.parent_revision, private.subtree), base_root)
        if base_snapshot.tree_digest != bundle.public.get("base_digest"):
            raise BenchmarkError("parent archive digest does not match public base_digest")
    git_archive_snapshot(repository, _tree_ish(private.gold_revision, private.subtree), gold_root)
    for item in bundle.public.get("public_tests", ()):
        if not isinstance(item, Mapping):
            raise BenchmarkError("public test is not an object")
        path = item.get("path")
        source = item.get("source")
        if not isinstance(path, str) or not isinstance(source, str):
            raise BenchmarkError("public test path/source is invalid")
        if snapshot_dir is None:
            _safe_overlay(base_root, path, source)
        _safe_overlay(gold_root, path, source)
    runner = worker_runner or (lambda row, root: _default_worker_runner(row, root, timeout=timeout))
    public = bundle.public_payload()
    parent_result = runner(public, base_root)
    gold_result = runner(public, gold_root)
    for label, result in (("parent", parent_result), ("gold", gold_result)):
        if not hasattr(result, "status") or not hasattr(result, "collected"):
            raise BenchmarkError(f"worker runner returned no typed {label} result")
    parent_receipt = make_worker_receipt(bundle.public, "parent", parent_result)
    gold_receipt = make_worker_receipt(bundle.public, "gold", gold_result)
    parent_status = parent_receipt.status
    gold_status = gold_receipt.status
    parent_collected = parent_receipt.collected
    gold_collected = gold_receipt.collected
    valid = parent_receipt.parent_red and gold_receipt.gold_green
    blocker = None
    if not valid:
        blocker = (
            "oracle requires parent ASSERTION_FAIL and gold PASS with collected tests"
        )
    return RealTaskWorkerValidation(
        task_id=str(bundle.public["task_id"]),
        valid=valid,
        parent_status=parent_status,
        parent_collected=parent_collected,
        gold_status=gold_status,
        gold_collected=gold_collected,
        blocker=blocker,
        parent_receipt=parent_receipt,
        gold_receipt=gold_receipt,
    )


def validate_realtask_in_worker(
    bundle: TaskBundle,
    repo: str | Path,
    *,
    worker_runner: WorkerRunner | None = None,
    timeout: float = 300,
    work_root: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
) -> RealTaskWorkerValidation:
    """Validate parent-red/gold-green in isolated archive roots.

    The injected runner is intentionally passed only the public row.  The
    private fixed revision is used by this controller function to create the
    gold worker root and is never a task-row field.  Candidates without a
    reviewed ticket are validated too (schema only; never published).
    """

    try:
        validate_public_manifest(bundle.public)
    except ManifestValidationError as exc:
        raise BenchmarkError(str(exc)) from exc
    if _private_key_paths(bundle.public):
        raise BenchmarkError("public task contains private fields")
    if work_root is None:
        with tempfile.TemporaryDirectory(prefix="realtask-worker-") as temp_name:
            return _validate_in_root(bundle, Path(repo).resolve(), Path(temp_name),
                                     worker_runner=worker_runner, timeout=timeout,
                                     snapshot_dir=snapshot_dir)
    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise BenchmarkError(f"worker root is not empty: {root}")
    return _validate_in_root(bundle, Path(repo).resolve(), root,
                             worker_runner=worker_runner, timeout=timeout,
                             snapshot_dir=snapshot_dir)


def write_public_jsonl(path: str | Path, bundles: Sequence[TaskBundle]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for bundle in bundles:
        validate_public_task(bundle.public)
        rows.append(json.dumps(bundle.public_payload(), sort_keys=True, separators=(",", ":")))
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_validation_jsonl(
    path: str | Path, results: Sequence[RealTaskWorkerValidation]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")) for result in results]
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _renumbered(bundle: TaskBundle, task_id: str) -> TaskBundle:
    public = dict(bundle.public, task_id=task_id)
    return TaskBundle(public=public, private=bundle.private)


def build_dev_tasks(
    repo: str | Path,
    *,
    revisions: Sequence[str] | None = None,
    limit: int = 5,
    task_output: str | Path,
    validation_output: str | Path,
    pending_output: str | Path,
    controller_output: str | Path,
    snapshot_dir: str | Path,
    requires_db: bool = False,
    timeout: float = 300,
    subtree: str = "",
    tickets: Mapping[str, object] | None = None,
    worker_runner: WorkerRunner | None = None,
    receipt_publisher: ReceiptPublisher | None = None,
    candidate_limit: int | None = None,
) -> tuple[
    tuple[TaskBundle, ...], tuple[RealTaskWorkerValidation, ...], tuple[dict, ...]
]:
    """Publish ONLY candidates that validated parent-red/gold-green AND carry a
    reviewed ticket (fix round, review #1).  Everything else goes to the
    controller-side pending file with its reason.  Accepted rows are
    renumbered ``rt-dev-NNN`` in order."""

    candidates = extract_dev_tasks(
        repo, revisions, limit=candidate_limit or max(limit * 4, limit),
        requires_db=requires_db, subtree=subtree, tickets=tickets,
        snapshot_dir=snapshot_dir)
    accepted: list[TaskBundle] = []
    results: list[RealTaskWorkerValidation] = []
    pending: list[dict] = []
    for bundle in candidates:
        if len(accepted) >= limit:
            break
        result = validate_realtask_in_worker(
            bundle,
            repo,
            timeout=timeout,
            worker_runner=worker_runner,
            snapshot_dir=snapshot_dir,
        )
        reasons = []
        if not result.valid:
            reasons.append(result.blocker or "worker validation failed")
        if not bundle.public.get("symptom"):
            reasons.append("no reviewed ticket")
        if reasons:
            pending.append({"schema": PENDING_SCHEMA,
                            "candidate": bundle.public["task_id"],
                            "gold_revision": bundle.private.gold_revision,
                            "reasons": reasons, "validation": result.as_dict()})
            continue
        task_id = f"rt-dev-{len(accepted) + 1:03d}"
        published = _renumbered(bundle, task_id)
        accepted.append(published)
        parent_receipt = (
            result.parent_receipt.rebind_public_task(published.public)
            if result.parent_receipt is not None else None
        )
        gold_receipt = (
            result.gold_receipt.rebind_public_task(published.public)
            if result.gold_receipt is not None else None
        )
        results.append(
            RealTaskWorkerValidation(
                task_id=task_id,
                valid=result.valid,
                parent_status=result.parent_status,
                parent_collected=result.parent_collected,
                gold_status=result.gold_status,
                gold_collected=result.gold_collected,
                blocker=result.blocker,
                parent_receipt=parent_receipt,
                gold_receipt=gold_receipt,
            )
        )
    if receipt_publisher is not None:
        for result in results:
            try:
                receipt_publisher(result.as_dict())
            except Exception as exc:  # noqa: BLE001 - release boundary fails closed
                raise BenchmarkError("worker receipt publisher failed") from exc
    write_public_jsonl(task_output, accepted)
    write_validation_jsonl(validation_output, results)
    for path, rows in (
            (pending_output, [json.dumps(row, sort_keys=True) for row in pending]),
            (controller_output, [json.dumps(b.private.controller_record(
                str(b.public["task_id"]), str(b.public["base_digest"])), sort_keys=True)
                for b in accepted])):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return tuple(accepted), tuple(results), tuple(pending)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--task-output", required=True, type=Path)
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument("--pending-output", required=True, type=Path)
    parser.add_argument("--controller-output", required=True, type=Path)
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--tickets", type=Path, help="reviewed tickets JSON {revision: ticket}")
    parser.add_argument("--subtree", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--commit", dest="commits", action="append")
    parser.add_argument("--requires-db", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    tickets = json.loads(args.tickets.read_text(encoding="utf-8")) if args.tickets else {}
    accepted, results, pending = build_dev_tasks(
        args.repo,
        revisions=args.commits,
        limit=args.limit,
        task_output=args.task_output,
        validation_output=args.validation_output,
        pending_output=args.pending_output,
        controller_output=args.controller_output,
        snapshot_dir=args.snapshot_dir,
        requires_db=args.requires_db,
        timeout=args.timeout,
        subtree=args.subtree,
        tickets=tickets,
        candidate_limit=args.candidate_limit,
    )
    print(json.dumps({"requested": args.limit, "accepted": len(accepted),
                      "pending": len(pending)}))
    # The build gate GATES: fewer validated, ticketed tasks than requested fails.
    return 0 if len(accepted) >= args.limit else 2


if __name__ == "__main__":
    raise SystemExit(main())
