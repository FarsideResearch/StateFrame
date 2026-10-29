"""Bind clean Stage 6 completion to its immutable product and candidate evidence."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Protocol

from tools.sciledger.final_product import (
    FINAL_PRODUCT_MANIFEST_NAME,
    FinalProductError,
    FinalProductManifest,
    write_manifest,
)
from tools.sciledger.run_real_binding import select_candidate_boundary


LIVE_SCHEMA = "aha-bug-replay-stage6-live/v1"
CANDIDATE_SCHEMA = "aha-bug-replay-stage6-candidate-diff/v3"


class EventSink(Protocol):
    def __call__(self, kind: str, **fields: object) -> object: ...


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FinalProductError(f"{label} must be non-empty text")
    return value.strip()


def stage6_identity(live_manifest_path: Path, *, task_id: str,
                    solver_run_id: str) -> dict[str, str]:
    """Read task and artifact identity from the host-staged live manifest."""
    try:
        value = json.loads(live_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalProductError("Stage 6 live identity manifest is unreadable") from exc
    if not isinstance(value, Mapping) or value.get("schema") != LIVE_SCHEMA:
        raise FinalProductError("Stage 6 live identity manifest has an unknown schema")
    tasks = value.get("tasks")
    if not isinstance(tasks, list):
        raise FinalProductError("Stage 6 live identity manifest has no task list")
    matches = [row for row in tasks
               if isinstance(row, Mapping) and row.get("task_id") == task_id]
    if len(matches) != 1:
        raise FinalProductError(
            "Stage 6 live identity must contain exactly one matching task")
    certificates = value.get("artifact_certificates")
    if not isinstance(certificates, Mapping):
        raise FinalProductError("Stage 6 live identity has no artifact certificates")
    task = matches[0]
    return {
        "run_id": _text(task.get("outer_run_id"), "live task outer_run_id"),
        "solver_run_id": _text(solver_run_id, "solver_run_id"),
        "task_id": _text(task.get("task_id"), "task_id"),
        "task_bundle_digest": _text(task.get("task_bundle_id"), "task_bundle_id"),
        "image_digest": _text(
            certificates.get("image_manifest_certificate_sha256"),
            "image_manifest_certificate_sha256"),
        "release_digest": _text(certificates.get("release_id"), "release_id"),
    }


def clean_terminal_event(journal_path: Path, *,
                         solver_run_id: str) -> Mapping[str, object]:
    """Return the latest durable PASS terminal for this solver attempt."""
    terminal: Mapping[str, object] | None = None
    try:
        with journal_path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (isinstance(row, Mapping)
                        and row.get("kind") == "attempt_terminated"
                        and row.get("run_id") == solver_run_id):
                    terminal = row
    except (OSError, UnicodeError) as exc:
        raise FinalProductError("solver terminal journal is unreadable") from exc

    if terminal is None:
        raise FinalProductError("solver journal has no terminal event for this run")
    result = terminal.get("result")
    system_outcome = terminal.get("system_outcome")
    external_grade_pending = (
        system_outcome == "solver_stopped"
        and terminal.get("result_authority") == "diagnostic_only_external_grade_pending")
    if (not isinstance(result, Mapping) or result.get("result") != "PASS"
            or (system_outcome is not None and not external_grade_pending)):
        raise FinalProductError("solver terminal event is not a clean accepted finish")
    steps = result.get("steps")
    if type(steps) is not int or steps < 0:
        raise FinalProductError("terminal result steps must be a non-negative integer")
    return {
        "event_id": _text(terminal.get("event_id"), "terminal event_id"),
        "run_id": solver_run_id,
        "result": "PASS",
        "system_outcome": system_outcome,
        "step": steps,
    }


def _candidate_document(path: Path) -> Mapping[str, object] | None:
    if path.is_symlink() or not path.is_file() or path.suffix != ".json":
        return None
    if not path.stem.isascii() or not path.stem.isdecimal():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (not isinstance(document, Mapping)
            or document.get("schema") != CANDIDATE_SCHEMA
            or type(document.get("proposals")) is not int
            or document["proposals"] < 0
            or int(path.stem) != document["proposals"]):
        return None
    return document


def final_candidate_boundary(run_dir: Path, candidate_dir: Path) -> str:
    """Return a selector-compatible candidate path, canonicalizing padded names."""
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise FinalProductError("Stage 6 run directory is missing or invalid")
    if candidate_dir.is_symlink() or candidate_dir.name != "candidates" or not candidate_dir.is_dir():
        raise FinalProductError("candidate boundary directory is missing or invalid")
    run_root = run_dir.resolve()
    candidate_root = candidate_dir.resolve()
    if not candidate_root.is_relative_to(run_root):
        raise FinalProductError("candidate boundary directory escapes the run")

    candidates: list[tuple[int, Path]] = []
    for path in candidate_root.iterdir():
        document = _candidate_document(path)
        if document is not None:
            candidates.append((int(document["proposals"]), path))
    if not candidates:
        raise FinalProductError("clean finish has no valid Stage 6 candidate boundary")

    proposal = max(index for index, _ in candidates)
    latest = [path for index, path in candidates if index == proposal]
    source = next((path for path in latest if path.stem == str(proposal)), latest[0])
    source_bytes = source.read_bytes()
    if any(path.read_bytes() != source_bytes for path in latest):
        raise FinalProductError("final candidate proposal index has conflicting boundaries")

    canonical = candidate_root / f"{proposal}.json"
    if canonical != source:
        if canonical.is_symlink():
            raise FinalProductError("canonical candidate boundary is a symlink")
        if canonical.exists():
            if not canonical.is_file() or canonical.read_bytes() != source_bytes:
                raise FinalProductError("canonical candidate boundary already has different contents")
        else:
            with canonical.open("xb") as handle:
                handle.write(source_bytes)
                handle.flush()
    if _candidate_document(canonical) is None:
        raise FinalProductError("canonical candidate boundary is invalid")
    if not canonical.resolve().is_relative_to(run_root):
        raise FinalProductError("canonical candidate boundary escapes the run")
    return canonical.relative_to(run_root).as_posix()


def emit_stage6_clean_finish_manifest(
    *,
    clean_finish: bool,
    run_dir: Path,
    parent_root: Path,
    final_root: Path,
    live_manifest_path: Path,
    journal_path: Path,
    solver_run_id: str,
    task_id: str,
    candidate_dir: Path,
    event_sink: EventSink | None = None,
) -> Path | None:
    """Serialize one clean Stage 6 product using the canonical Sciledger model."""
    if not clean_finish:
        return None
    if run_dir.name != f"{task_id}-a3b":
        raise FinalProductError("Stage 6 manifest run directory does not match task identity")

    identity = stage6_identity(
        live_manifest_path, task_id=task_id, solver_run_id=solver_run_id)
    terminal = clean_terminal_event(journal_path, solver_run_id=solver_run_id)
    if terminal["run_id"] != identity["solver_run_id"]:
        raise FinalProductError("terminal event does not match solver identity")
    boundary = final_candidate_boundary(run_dir, candidate_dir)
    manifest = FinalProductManifest.from_trees(
        parent_root=parent_root,
        final_root=final_root,
        run_id=identity["run_id"],
        solver_run_id=identity["solver_run_id"],
        task_id=identity["task_id"],
        task_bundle_digest=identity["task_bundle_digest"],
        image_digest=identity["image_digest"],
        release_digest=identity["release_digest"],
        final_event_id=str(terminal["event_id"]),
        final_step=int(terminal["step"]),
        terminal_class="DONE",
        certified=False,
        salvage=False,
        candidate_boundary=boundary,
    )

    manifest_path = run_dir / FINAL_PRODUCT_MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite final-product manifest: {manifest_path}")
    write_manifest(manifest_path, manifest)
    selected = select_candidate_boundary(run_dir)
    expected = (run_dir / boundary).resolve()
    if selected != expected:
        raise FinalProductError("RUNREAL selected a different candidate boundary")
    if event_sink is not None:
        event_sink("final_product_manifest_written", producer="kernel_final_product_manifest",
                   fact_owner="HOST", dispatch=False, task_id=task_id,
                   solver_run_id=solver_run_id, terminal_event_ref=terminal["event_id"],
                   candidate_boundary=boundary,
                   manifest_payload_hash="sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   proof_boundary="postterminal_host_artifact", on_wire=False)
    return manifest_path


__all__ = [
    "CANDIDATE_SCHEMA",
    "LIVE_SCHEMA",
    "clean_terminal_event",
    "emit_stage6_clean_finish_manifest",
    "final_candidate_boundary",
    "stage6_identity",
]
