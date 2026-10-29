"""Stage 6 terminal settlement for the plain-loop engine, in RUNREAL's existing vocabulary.

RUNREAL (bin/run_real.sh) selects the graded candidate ONLY through a final-product
manifest, and otherwise routes on journaled host facts. A COMPLETE run with neither
is labelled HOST_ERROR (batches 1046-1049). This module emits exactly what the full
kernel emits, through the same canonical writer, so RUNREAL needs no plain-specific
branch:

- model called ``done`` with product files changed -> ``attempt_terminated`` (PASS =
  clean finish, never a grade) + ``kernel_final_product_manifest`` manifest;
  if writing it fails -> ``final_product_manifest_failed`` (RUNREAL salvage-grades);
- anything else -> ``final_product_no_candidate`` with reason ``no_product_edit`` or
  ``not_completed`` and ``salvage_gradable`` when product files changed.

Fail-closed: no path fabricates a manifest; every failure is a journaled host fact.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import uuid

LIVE_MANIFEST_PATH = Path("/stage/live-manifest.json")  # same constant the full kernel uses


def settle_stage6(*, outcome: Mapping[str, object], row: Mapping[str, object], final_root: Path,
                  parent_root: Path | None, product_files_changed: int, journal,
                  live_manifest_path: Path | None = None) -> dict[str, object]:
    capture = row.get("_bridge_capture")
    if not capture:
        return {"status": "NO_CAPTURE"}
    changed = max(0, int(product_files_changed))
    end = str(outcome.get("end") or "")
    if end != "done" or changed == 0:
        reason = "no_product_edit" if changed == 0 else "not_completed"
        journal.write("final_product_no_candidate", reason=reason, result=end[:200],
                      product_files_changed=changed, salvage_gradable=changed > 0,
                      fact_owner="HOST", producer="kernel_plain_settlement", dispatch=False)
        return {"status": "NO_CANDIDATE", "reason": reason}

    solver_run_id = "plain-" + uuid.uuid4().hex
    event_id = uuid.uuid4().hex
    journal.write("attempt_terminated", run_id=solver_run_id, event_id=event_id,
                  result={"result": "PASS", "steps": int(outcome.get("steps") or 0)}, system_outcome=None,
                  result_meaning="plain engine: model called done after changing product files; not a grade")
    try:
        if parent_root is None:
            raise RuntimeError("no parent snapshot is known for this workspace")
        from kernel_final_product_manifest import emit_stage6_clean_finish_manifest
        capture_dir = Path(str(capture)).resolve()
        manifest = emit_stage6_clean_finish_manifest(
            clean_finish=True, run_dir=capture_dir.parent.parent, parent_root=Path(parent_root),
            final_root=final_root, live_manifest_path=live_manifest_path or LIVE_MANIFEST_PATH,
            journal_path=Path(journal.path), solver_run_id=solver_run_id,
            task_id=str(row.get("task_id") or ""), candidate_dir=capture_dir,
            event_sink=lambda kind, **fields: journal.write(kind, **fields))
    except Exception as error:  # host bookkeeping never flips the outcome; RUNREAL salvage-grades
        journal.write("final_product_manifest_failed", producer="kernel_plain_settlement", fact_owner="HOST",
                      dispatch=False, error_type=type(error).__name__, message=str(error)[:400],
                      fallback="timeout_salvage_export")
        return {"status": "MANIFEST_FAILED", "error": f"{type(error).__name__}: {error}"[:400]}
    return {"status": "MANIFEST_WRITTEN", "manifest": str(manifest), "solver_run_id": solver_run_id}


__all__ = ["LIVE_MANIFEST_PATH", "settle_stage6"]
