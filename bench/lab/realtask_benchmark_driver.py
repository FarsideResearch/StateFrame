"""Maintainer-only batch/corpus driver; absent from release artifacts."""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_ENGINE = _Path(__file__).resolve().parents[2] / "src/stateframe/engine"
if str(_ENGINE) not in _sys.path:
    _sys.path.insert(0, str(_ENGINE))

import argparse  # noqa: E402 - engine path is set above
from realtask_benchmark import (  # noqa: E402 - engine path is set above
    BenchmarkError,
    Mapping,
    PENDING_SCHEMA,
    Path,
    RealTaskWorkerValidation,
    ReceiptPublisher,
    Sequence,
    TaskBundle,
    WorkerRunner,
    extract_dev_tasks,
    json,
    validate_public_task,
    validate_realtask_in_worker,
)

import realtask_benchmark as _shipped  # noqa: E402 - engine path is set above
__doc__ = _shipped.__doc__


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
