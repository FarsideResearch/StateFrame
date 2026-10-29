"""CAPWIRE: bounded, public-only FL-04 checkpoint capture for live V46 runs."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from kernel_done_acceptance import done_accepted


FLAG = "STATEFRAME_FL_CHECKPOINT_CAPTURE"
SCHEMA = "stateframe-fl-checkpoint/1"
TRUTHY = frozenset({"1", "on", "true", "yes"})
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.:-]+")
_MANIFEST_SCHEMA = "frame-lab/checkpoint-manifest/1"
_CAPTURE_DOCUMENT_SCHEMA = "frame-lab/settled-checkpoint/1"
_PRIVATE_SEGMENTS = frozenset({
    ".git", ".pytest_cache", "__pycache__", "backup", "backups", "gold",
    "grader", "hidden", "private", "production", "secret",
})


class CaptureBoundaryError(RuntimeError):
    """A public checkpoint cannot be exported without weakening its contract."""


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(FLAG, "off")).strip().lower() in TRUTHY


def _positive_int(name: str, default: int, *, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise CaptureBoundaryError(f"{name} must be an integer") from exc
    if value <= 0 or value > maximum:
        raise CaptureBoundaryError(f"{name} is outside its bounded range")
    return value


def _limits() -> tuple[int, int, int, int]:
    return (
        _positive_int("STATEFRAME_FL_CAPTURE_EVERY_K", 5, maximum=1000),
        _positive_int("STATEFRAME_FL_CAPTURE_MAX_COUNT", 32, maximum=1000),
        _positive_int("STATEFRAME_FL_CAPTURE_MAX_FILES", 64, maximum=10000),
        _positive_int("STATEFRAME_FL_CAPTURE_MAX_BYTES", 4 * 1024 * 1024, maximum=16 * 1024 * 1024),
    )


def _safe_error(error: BaseException) -> str:
    first = str(error).splitlines()[0].strip() if str(error).strip() else "<no message>"
    return f"{type(error).__name__}: {first[:240]}"


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaptureBoundaryError("checkpoint state is not canonical JSON") from exc


@dataclass(frozen=True)
class _CompatWorkspaceFile:
    path: str
    digest: str
    size: int
    mode: int
    kind: str = "file"

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path, "digest": self.digest, "size": self.size,
            "mode": self.mode, "kind": self.kind,
        }


@dataclass(frozen=True)
class _CompatManifest:
    checkpoint_id: str
    run_id: str
    request_id: str
    event_id: str | None
    journal_prefix_hash: str | None
    journal_cutoff: int
    request_hash: str
    model_identity: Mapping[str, object]
    sampler_identity: Mapping[str, object]
    template_identity: str
    requirement_revision: str | None
    requirement_provenance: object
    workspace_files: tuple[object, ...]
    declared_absences: tuple[str, ...]
    tree_digest: str | None
    check_definitions: tuple[object, ...]
    runtime_identity: Mapping[str, object]
    kernel_artifact_hash: str | None
    effective_flags: Mapping[str, object]
    adapter_schema_version: str
    notes_state: Mapping[str, object]
    receipt_lineage: tuple[str, ...]
    completion_state: Mapping[str, object]
    budgets: Mapping[str, object]
    pending_request: Mapping[str, object] | None
    pending_effect: Mapping[str, object] | None
    capture_boundary: str
    reconstruction_method: str
    admission_grade: object
    exclusions: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    permitted_tools: tuple[object, ...]
    symlink_policy: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": _MANIFEST_SCHEMA, "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id, "request_id": self.request_id,
            "event_id": self.event_id,
            "journal_prefix_hash": self.journal_prefix_hash,
            "journal_cutoff": self.journal_cutoff, "request_hash": self.request_hash,
            "model_identity": dict(self.model_identity),
            "sampler_identity": dict(self.sampler_identity),
            "template_identity": self.template_identity,
            "requirement_revision": self.requirement_revision,
            "requirement_provenance": None,
            "workspace_files": [
                item.to_mapping() if hasattr(item, "to_mapping") else dict(item)
                for item in self.workspace_files
            ],
            "declared_absences": list(self.declared_absences),
            "tree_digest": self.tree_digest,
            "check_definitions": [], "runtime_identity": dict(self.runtime_identity),
            "kernel_artifact_hash": self.kernel_artifact_hash,
            "effective_flags": dict(self.effective_flags),
            "adapter_schema_version": self.adapter_schema_version,
            "notes_state": dict(self.notes_state),
            "receipt_lineage": list(self.receipt_lineage),
            "completion_state": dict(self.completion_state),
            "budgets": dict(self.budgets),
            "pending_request": self.pending_request,
            "pending_effect": self.pending_effect,
            "capture_boundary": self.capture_boundary,
            "reconstruction_method": self.reconstruction_method,
            "admission_grade": getattr(self.admission_grade, "value", self.admission_grade),
            "exclusions": list(self.exclusions), "unknown_fields": list(self.unknown_fields),
            "permitted_tools": [], "symlink_policy": self.symlink_policy,
        }


@dataclass(frozen=True)
class _CompatCheckpoint:
    manifest: _CompatManifest
    request_payload: str | bytes
    public_state: Mapping[str, object]
    kernel_state: Mapping[str, object]
    admission_state: Mapping[str, object]
    settled_effects: tuple[object, ...] = ()
    release: object | None = None
    workspace_root: Path | None = None


@dataclass(frozen=True)
class _CompatExport:
    artifact_hash: str
    manifest_path: Path
    blob_paths: tuple[Path, ...]
    document: Mapping[str, object]


class _CompatExporter:
    """Release-local facade matching the FL-04 export surface."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.blobs = self.root / "blobs" / "sha256"
        self.manifests = self.root / "manifests"

    @staticmethod
    def _immutable(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise CaptureBoundaryError(f"immutable export collision: {path.name}")
            return
        path.write_bytes(data)

    def export(self, checkpoint: _CompatCheckpoint) -> _CompatExport:
        manifest = checkpoint.manifest
        if manifest.pending_request is not None or manifest.pending_effect is not None:
            raise CaptureBoundaryError("settled capture contains pending work")
        blob_paths: list[Path] = []
        refs: dict[str, str] = {}
        for item in manifest.workspace_files:
            source = (checkpoint.workspace_root or Path("/nonexistent")) / item.path
            data = source.read_bytes()
            digest = _digest_bytes(data)
            if digest != item.digest:
                raise CaptureBoundaryError(f"workspace drift: {item.path}")
            blob = self.blobs / digest.removeprefix("sha256:")
            self._immutable(blob, data)
            blob_paths.append(blob)
            refs[item.path] = digest
        payload = checkpoint.request_payload
        payload = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        if not isinstance(payload, str):
            raise CaptureBoundaryError("request payload is not UTF-8 text")
        document: dict[str, object] = {
            "schema": _CAPTURE_DOCUMENT_SCHEMA, "checkpoint": manifest.to_mapping(),
            "request_payload_utf8": payload,
            "public_state": copy.deepcopy(dict(checkpoint.public_state)),
            "admission_state": copy.deepcopy(dict(checkpoint.admission_state)),
            "kernel_state": copy.deepcopy(dict(checkpoint.kernel_state)),
            "settled_effects": [], "workspace_blobs": refs,
        }
        raw = _canonical(document)
        artifact_hash = _digest_bytes(raw)
        name = _SAFE_ID.sub("_", manifest.checkpoint_id).strip("._") + ".json"
        manifest_path = self.manifests / name
        self._immutable(manifest_path, raw)
        return _CompatExport(artifact_hash, manifest_path, tuple(blob_paths), document)


@dataclass(frozen=True)
class _CompatResult:
    enabled: bool
    state: object
    export: _CompatExport | None


class _CompatHook:
    def __init__(self, *, enabled: bool, exporter: _CompatExporter | None = None) -> None:
        if enabled and exporter is None:
            raise CaptureBoundaryError("enabled hook requires an exporter")
        self.enabled = enabled
        self.exporter = exporter

    def observe(self, state: object, checkpoint: _CompatCheckpoint) -> _CompatResult:
        if not self.enabled:
            return _CompatResult(False, state, None)
        assert self.exporter is not None
        return _CompatResult(True, state, self.exporter.export(checkpoint))


@dataclass(frozen=True)
class _Fl04Api:
    AdmissionGrade: object
    CheckpointManifest: object
    SettledCheckpoint: object
    SnapshotExporter: object
    SnapshotHook: object
    WorkspaceFile: object
    bytes_digest: object
    canonical_bytes: object
    tree_digest: object
    classify_manifest: object | None
    external: bool


_API: _Fl04Api | None = None


def _fl04_api() -> _Fl04Api:
    global _API
    if _API is not None:
        return _API
    try:
        configured_micro = os.environ.get("STATEFRAME_CHECKPOINT_MICRO_PATH")
        if not configured_micro:
            raise ImportError("checkpoint extension is not configured")
        candidate = Path(configured_micro)
        if not candidate.is_dir():
            raise ImportError("checkpoint extension directory is unavailable")
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        from checkpoint import AdmissionGrade, CheckpointManifest, WorkspaceFile, classify_manifest
        from checkpoint import bytes_digest, canonical_bytes, tree_digest
        from restore.capture import SettledCheckpoint, SnapshotExporter, SnapshotHook
        _API = _Fl04Api(
            AdmissionGrade, CheckpointManifest, SettledCheckpoint, SnapshotExporter,
            SnapshotHook, WorkspaceFile, bytes_digest, canonical_bytes, tree_digest,
            classify_manifest, True,
        )
        return _API
    except ImportError:
        class _Grade:
            WORKSPACE_EXACT = "WORKSPACE_EXACT"
            KERNEL_EXACT = "KERNEL_EXACT"

        def _tree(entries: Sequence[object], declared_absences: Sequence[str] = ()) -> str:
            rows = [
                item.to_mapping() if hasattr(item, "to_mapping") else dict(item)
                for item in entries
            ]
            return _digest_bytes(_canonical({
                "files": sorted(rows, key=lambda row: row["path"]),
                "declared_absences": sorted(declared_absences),
            }))

        def _classify(manifest: object) -> object:
            if getattr(manifest, "kernel_artifact_hash", None) is None:
                return _Grade.WORKSPACE_EXACT
            return _Grade.KERNEL_EXACT

        _API = _Fl04Api(
            _Grade, _CompatManifest, _CompatCheckpoint, _CompatExporter, _CompatHook,
            _CompatWorkspaceFile, _digest_bytes, _canonical, _tree, _classify, False,
        )
        return _API


def _journal_capture(
    journal_path: Path, max_line_bytes: int,
) -> tuple[Mapping[str, object], str]:
    """Find the settled request and hash its prefix without loading the journal.

    The old implementation allocated the complete journal and rejected a late
    capture when that allocation exceeded ``STATEFRAME_FL_CAPTURE_MAX_BYTES``.
    This keeps one JSONL record in memory at a time, then streams only through
    the immutable request cutoff to build the prefix digest.
    """

    submitted: Mapping[str, object] | None = None
    try:
        with journal_path.open("rb") as stream:
            for raw_line in stream:
                if len(raw_line) > max_line_bytes:
                    raise CaptureBoundaryError("public journal prefix exceeds capture bound")
                if not raw_line.strip():
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise CaptureBoundaryError("public journal contains invalid JSON") from exc
                if not isinstance(value, dict):
                    raise CaptureBoundaryError("public journal record is not an object")
                if value.get("kind") != "request_submitted":
                    continue
                payload = value.get("payload_utf8")
                request_id = value.get("request_id")
                if isinstance(payload, str) and isinstance(request_id, str) and request_id:
                    submitted = value
    except OSError as exc:
        raise CaptureBoundaryError("public journal cannot be read") from exc
    if submitted is None:
        raise CaptureBoundaryError("public journal has no submitted request anchor")
    cutoff = submitted.get("sequence")
    if not isinstance(cutoff, int) or cutoff < 1:
        raise CaptureBoundaryError("request anchor has no valid journal cutoff")

    digest = hashlib.sha256()
    found_cutoff = False
    try:
        with journal_path.open("rb") as stream:
            for raw_line in stream:
                if len(raw_line) > max_line_bytes:
                    raise CaptureBoundaryError("public journal prefix exceeds capture bound")
                if not raw_line.strip():
                    continue
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise CaptureBoundaryError("public journal contains invalid JSON") from exc
                if not isinstance(value, dict):
                    raise CaptureBoundaryError("public journal record is not an object")
                sequence = value.get("sequence")
                if not isinstance(sequence, int):
                    raise CaptureBoundaryError("public journal prefix cannot be aligned")
                if sequence > cutoff:
                    break
                digest.update(raw_line)
                if sequence == cutoff:
                    found_cutoff = True
                    break
    except OSError as exc:
        raise CaptureBoundaryError("public journal cannot be read") from exc
    if not found_cutoff:
        raise CaptureBoundaryError("public journal cutoff is not present")
    return submitted, "sha256:" + digest.hexdigest()


def _safe_flags() -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in sorted(os.environ.items()):
        if not (key.startswith("STATEFRAME_") or key == "KERNEL_INTERVENTIONS"):
            continue
        upper = key.upper()
        if any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "AUTH", "URL")):
            continue
        if isinstance(value, str) and len(value) <= 160:
            result[key] = value
    result[FLAG] = "on"
    return result


def _workspace_files(
    root: Path, api: _Fl04Api, max_files: int, max_bytes: int,
) -> tuple[tuple[object, ...], str, tuple[str, ...]]:
    root = root.resolve()
    if not root.is_dir():
        raise CaptureBoundaryError("solver workspace is not a directory")
    entries: list[object] = []
    exclusions: list[str] = []
    total_bytes = 0
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        kept: list[str] = []
        for name in sorted(directories):
            if name.startswith(".") or name.lower() in _PRIVATE_SEGMENTS:
                exclusions.append((current_path / name).relative_to(root).as_posix())
            else:
                kept.append(name)
        directories[:] = kept
        for name in sorted(filenames):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if name.startswith(".") or any(
                part.startswith(".") or part.lower() in _PRIVATE_SEGMENTS
                for part in Path(relative).parts
            ):
                exclusions.append(relative)
                continue
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise CaptureBoundaryError(f"unsupported solver workspace entry: {relative}")
            size = int(info.st_size)
            if len(entries) >= max_files or total_bytes + size > max_bytes:
                raise CaptureBoundaryError("solver workspace exceeds capture bound")
            data = path.read_bytes()
            digest = api.bytes_digest(data)
            entries.append(api.WorkspaceFile(
                relative, digest, size, stat.S_IMODE(info.st_mode), "file"))
            total_bytes += size
    ordered = tuple(sorted(entries, key=lambda item: item.path))
    return ordered, api.tree_digest(ordered), tuple(sorted(set(exclusions)))


def _public_state(runtime: object, action: str, observation: str) -> dict[str, object]:
    owner = getattr(runtime, "owner", None)
    binding = getattr(runtime, "binding", None)
    binding_value = binding() if callable(binding) else {}
    binding_value = binding_value if isinstance(binding_value, Mapping) else {}
    target = getattr(owner, "target", None)
    target_path = ""
    root = Path(getattr(owner, "root", Path.cwd())).resolve()
    if target is not None:
        try:
            target_path = Path(target).resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            target_path = ""
    last_exit = getattr(runtime, "last_test_exit", None)
    if last_exit is None:
        receipts = getattr(runtime, "test_receipts", ())
        if receipts and isinstance(receipts[-1], Mapping):
            last_exit = receipts[-1].get("exit_code")
    check_state = "PASSED" if last_exit == 0 else "FAILED" if last_exit is not None else "UNKNOWN"
    open_requirement_ids: tuple[str, ...] = ()
    open_requirements: list[dict[str, object]] = []
    try:
        from kernel_clause_discriminator import open_clauses
        open_rows = open_clauses(runtime)
        open_requirement_ids = tuple(str(row["clause_ref"]) for row in open_rows)
        open_requirements = [
            {"id": str(row["clause_ref"]), "clause": str(row.get("clause", "")),
             "status": "open", "reason": str(row.get("reason", "coverage required"))}
            for row in open_rows
        ]
        setattr(runtime, "open_requirement_ids", open_requirement_ids)
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # Checkpoint capture remains fail-open, but it must never invent a green
        # clause state when the public-spec adapter is unavailable.
        open_requirement_ids = tuple(str(item) for item in
                                     getattr(runtime, "open_requirement_ids", ()) or ())
    selectors_fn = getattr(runtime, "required_public_selectors", None)
    selectors = selectors_fn() if callable(selectors_fn) else ()
    return {
        "schema": "stateframe-fl-public-state/1",
        "revision": str(binding_value.get("product_revision") or "live-public"),
        "revision_state": "unknown",
        "public_checks": {
            "state": check_state,
            "selectors": [str(item) for item in selectors or ()],
        },
        "open_work_completeness": "UNKNOWN",
        "open_requirement_ids": list(open_requirement_ids),
        "open_requirements": open_requirements,
        "named_files": [target_path] if target_path else [],
        "changed_files": [], "unchanged_named_files": [],
        "refusal_codes": ["PUBLIC_ACTION_REFUSAL"] if _is_refusal(action, observation) else [],
        "tool_menu": [],
    }


def _is_refusal(action: str, observation: str) -> bool:
    text = f"{action} {observation}".lower()
    return any(token in text for token in (
        "refus", "reject", "challeng", "blocked", "unsupported", "no-op", "no op",
    ))


def _is_stall(runtime: object, action: str, observation: str) -> bool:
    text = f"{action} {observation}".lower()
    if "stall" in text:
        return True
    cycles = getattr(runtime, "_v46_stall_cycles", {})
    return isinstance(cycles, Mapping) and any(
        isinstance(value, int) and value >= 2 for value in cycles.values())


def _kernel_state(runtime: object, *, settled: int, action: str, observation: str) -> dict[str, object]:
    owner = getattr(runtime, "owner", None)
    receipts = getattr(owner, "_receipts", ())
    last_receipt: dict[str, object] = {}
    lineage: list[str] = []
    recent = receipts[-16:] if isinstance(receipts, Sequence) else ()
    for item in recent:
        if not isinstance(item, Mapping):
            continue
        receipt_id = item.get("receipt_id")
        if isinstance(receipt_id, str):
            lineage.append(receipt_id)
        if not last_receipt:
            last_receipt = {
                key: item.get(key)
                for key in ("receipt_id", "kind", "tool", "result_class", "target")
                if item.get(key) is not None
            }
    budget = getattr(runtime, "decision_budget", None)
    budget_value = (
        budget.snapshot()
        if budget is not None and callable(getattr(budget, "snapshot", None))
        else {}
    )
    if not isinstance(budget_value, Mapping):
        budget_value = {}
    return {
        "schema": SCHEMA, "settled_decisions": settled,
        "current_step": getattr(owner, "current_step", None),
        "last_action": {
            "kind": str(action)[:120],
            "observation_digest": _digest_bytes(observation.encode("utf-8")),
        },
        "last_receipt": last_receipt, "receipt_lineage": lineage,
        "completion": {
            "system_outcome": getattr(runtime, "system_outcome", None),
            "accepted": bool(done_accepted(runtime)),
        },
        "budgets": dict(budget_value), "pending_request": None, "pending_effect": None,
    }


def _trigger(
    controller: "_Controller", runtime: object, action: str, observation: str, every_k: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if action.strip().lower() == "done" and not controller.done_seen:
        controller.done_seen = True
        reasons.append("first_done_attempt")
    if _is_refusal(action, observation):
        reasons.append("challenge_or_refusal")
    if _is_stall(runtime, action, observation):
        reasons.append("stall_detection")
    if controller.settled % every_k == 0:
        reasons.append(f"every_{every_k}_decisions")
    return tuple(reasons)


@dataclass
class _Controller:
    exporter: object
    hook: object
    settled: int = 0
    captures: int = 0
    done_seen: bool = False
    disabled: bool = False


def _controller(runtime: object) -> _Controller:
    current = getattr(runtime, "_fl_checkpoint_controller", None)
    if isinstance(current, _Controller):
        return current
    api = _fl04_api()
    journal = getattr(runtime, "journal", None)
    journal_path = getattr(journal, "path", None)
    if not isinstance(journal_path, Path):
        raise CaptureBoundaryError("measurement journal path is unavailable")
    exporter = api.SnapshotExporter(journal_path.parent / "fl-checkpoints")
    hook = api.SnapshotHook(enabled=True, exporter=exporter)
    current = _Controller(exporter=exporter, hook=hook)
    setattr(runtime, "_fl_checkpoint_controller", current)
    return current


def _capture(runtime: object, action: str, observation: str, reasons: tuple[str, ...]) -> object:
    api = _fl04_api()
    journal = runtime.journal
    journal_path = Path(journal.path).resolve()
    measurement_dir = journal_path.parent.resolve()
    if not journal_path.is_relative_to(measurement_dir) or journal_path.name != "events.jsonl":
        raise CaptureBoundaryError("journal path is outside the measurement directory")
    _, max_count, max_files, max_bytes = _limits()
    controller = _controller(runtime)
    if controller.captures >= max_count:
        raise CaptureBoundaryError("per-run capture count bound reached")
    submitted, journal_prefix_hash = _journal_capture(journal_path, max_bytes)
    cutoff = submitted["sequence"]
    payload = submitted["payload_utf8"]
    assert isinstance(payload, str)
    owner = runtime.owner
    workspace_root = Path(owner.root).resolve()
    files, tree_hash, exclusions = _workspace_files(workspace_root, api, max_files, max_bytes)
    public_state = _public_state(runtime, action, observation)
    public_state["journal_prefix_ref"] = {
        "path": "events.jsonl",
        "cutoff": cutoff,
        "sha256": journal_prefix_hash,
    }
    kernel_state = _kernel_state(
        runtime, settled=controller.settled, action=action, observation=observation)
    kernel_hash = api.bytes_digest(api.canonical_bytes(kernel_state))
    run_id = str(journal.run_id)
    checkpoint_id = _SAFE_ID.sub("_", f"{run_id}-{controller.settled:06d}")
    manifest = api.CheckpointManifest(
        checkpoint_id=checkpoint_id, run_id=run_id,
        request_id=str(submitted["request_id"]),
        event_id=str(submitted.get("event_id")) if submitted.get("event_id") else None,
        journal_prefix_hash=journal_prefix_hash,
        journal_cutoff=cutoff,
        request_hash=api.bytes_digest(payload.encode("utf-8")),
        model_identity={
            "requested_model": str(journal.manifest.get("model", "unknown"))[:160],
            "served_model": str(
                getattr(runtime, "served_model", os.environ.get("REACT_SERVED_MODEL", "unknown"))
            )[:160],
        },
        sampler_identity={},
        template_identity=str(journal.manifest.get("chat_template", "unknown"))[:160],
        requirement_revision=str(public_state["revision"]),
        requirement_provenance=None, workspace_files=files,
        declared_absences=(), tree_digest=tree_hash, check_definitions=(),
        runtime_identity={"runtime": "stateframe-v46", "workspace_root": "solver-workspace"},
        kernel_artifact_hash=kernel_hash, effective_flags=_safe_flags(),
        adapter_schema_version=SCHEMA, notes_state={},
        receipt_lineage=tuple(kernel_state["receipt_lineage"]),
        completion_state=kernel_state["completion"], budgets=kernel_state["budgets"],
        pending_request=None, pending_effect=None,
        capture_boundary="settled-after-action-before-next-request",
        reconstruction_method="live-public-export",
        admission_grade=api.AdmissionGrade.KERNEL_EXACT,
        exclusions=tuple(sorted({"hidden/gold/grader/private paths excluded", *exclusions})),
        unknown_fields=(), permitted_tools=(), symlink_policy="reject",
    )
    checkpoint = api.SettledCheckpoint(
        manifest=manifest, request_payload=payload, public_state=public_state,
        kernel_state=kernel_state,
        admission_state={"check_state": public_state["public_checks"]["state"]},
        workspace_root=workspace_root,
    )
    result = controller.hook.observe(kernel_state, checkpoint)
    export = getattr(result, "export", None)
    if export is None:
        raise CaptureBoundaryError("FL-04 hook returned no export")
    controller.captures += 1
    workspace_grade = api.AdmissionGrade.WORKSPACE_EXACT
    kernel_grade = manifest.admission_grade
    if api.classify_manifest is not None:
        workspace_grade = api.classify_manifest(dataclasses.replace(manifest, kernel_artifact_hash=None))
        kernel_grade = api.classify_manifest(manifest)
    journal.emit(
        "fl_checkpoint_captured", producer="kernel_fl_checkpoint_capture",
        event_schema=SCHEMA, checkpoint_id=checkpoint_id,
        artifact_hash=export.artifact_hash,
        manifest_path="fl-checkpoints/manifests/" + export.manifest_path.name,
        workspace_admission_grade=getattr(workspace_grade, "value", str(workspace_grade)),
        kernel_admission_grade=getattr(kernel_grade, "value", str(kernel_grade)),
        reasons=list(reasons), settled_decisions=controller.settled,
        capture_count=controller.captures, workspace_file_count=len(files),
        workspace_byte_count=sum(int(item.size) for item in files),
        api="fl04" if api.external else "fl04-release-facade", dispatch=False,
    )
    return export


def observe_settled_boundary(runtime: object, action: str, observation: str) -> None:
    """Offer one settled boundary to CAPWIRE; a bound failure disables capture once."""
    if not enabled():
        return
    controller: _Controller | None = None
    try:
        controller = _controller(runtime)
        if controller.disabled:
            return
        controller.settled += 1
        every_k, _, _, _ = _limits()
        reasons = _trigger(controller, runtime, str(action), str(observation), every_k)
        if reasons:
            _capture(runtime, str(action), str(observation), reasons)
    except CaptureBoundaryError as error:
        if controller is None:
            journal = getattr(runtime, "journal", None)
            emit = getattr(journal, "emit", None)
            if callable(emit):
                emit(
                    "fl_checkpoint_capture_error", producer="kernel_fl_checkpoint_capture",
                    event_schema=SCHEMA, error=_safe_error(error), dispatch=False,
                )
            return
        controller.disabled = True
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit(
                "fl_checkpoint_capture_disabled", producer="kernel_fl_checkpoint_capture",
                event_schema=SCHEMA, reason="capture_boundary_error", error=_safe_error(error),
                dispatch=False,
            )
    except Exception as error:  # noqa: BLE001 - capture must never break a run
        journal = getattr(runtime, "journal", None)
        emit = getattr(journal, "emit", None)
        if callable(emit):
            emit(
                "fl_checkpoint_capture_error", producer="kernel_fl_checkpoint_capture",
                event_schema=SCHEMA, error=_safe_error(error), dispatch=False,
            )


__all__ = ["FLAG", "SCHEMA", "enabled", "observe_settled_boundary"]
