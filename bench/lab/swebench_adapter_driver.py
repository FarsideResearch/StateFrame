"""Maintainer-only batch/corpus driver; absent from release artifacts."""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_ENGINE = _Path(__file__).resolve().parents[2] / "src/stateframe/engine"
if str(_ENGINE) not in _sys.path:
    _sys.path.insert(0, str(_ENGINE))

import argparse  # noqa: E402 - engine path is set above
from collections import Counter  # noqa: E402 - engine path is set above
from build_tasks.swebench_adapter import (  # noqa: E402 - engine path is set above
    AdapterError,
    DATASET_ID,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT,
    DEFAULT_SKIPPED_OUTPUT,
    Mapping,
    MaterializationRequired,
    PatchAnalysis,
    Path,
    SWEInstance,
    Sequence,
    SkipInstance,
    WorkspaceMaterializer,
    _FUNCTION_RE,
    _patch_paths,
    analyze_gold_patch,
    convert_instance,
    dataclass,
    json,
    load_instances,
    sys,
)


@dataclass(frozen=True)
class BuildReport:
    """Deterministic output summary emitted by build_corpus."""

    emitted: tuple[dict[str, object], ...]
    skipped: tuple[dict[str, object], ...]
    reason_counts: Mapping[str, int]


@dataclass(frozen=True)
class ConversionSummary:
    """Patch-shape census produced without cloning or reading a repository."""

    considered: int
    would_emit_multi_file: int
    would_emit_single_file: int
    skipped: int
    reason_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "considered": self.considered,
            "would_emit_multi_file": self.would_emit_multi_file,
            "would_emit_single_file": self.would_emit_single_file,
            "skipped": self.skipped,
            "reason_counts": dict(self.reason_counts),
        }


def _conversion_function_names(analysis: PatchAnalysis) -> dict[str, list[str]]:
    """Use unified-diff function context for a no-checkout shape census."""
    changed: dict[str, list[str]] = {}
    for path, hunks in analysis.files.items():
        names = sorted(
            {
                match.group("name")
                for hunk in hunks
                for match in _FUNCTION_RE.finditer(hunk.section)
            }
        )
        if not names:
            raise SkipInstance(
                "ambiguous_source_mapping",
                f"conversion-only census cannot map a hunk in {path} without checkout source",
            )
        changed[path] = names
    return changed


def summarize_conversion(instances: Sequence[SWEInstance]) -> ConversionSummary:
    """Count row shapes and named skips without materializing repositories."""
    multi_file = 0
    single_file = 0
    skipped = 0
    counts: Counter[str] = Counter()
    for instance in instances:
        try:
            analysis = analyze_gold_patch(instance.patch)
            if analysis.added_paths:
                raise SkipInstance(
                    "target_missing",
                    "new target files have no complete parent source: "
                    + ", ".join(analysis.added_paths),
                )
            code_paths = set(analysis.paths)
            test_paths = set(_patch_paths(instance.test_patch or ""))
            overlap = sorted(code_paths & test_paths)
            if overlap:
                raise SkipInstance(
                    "test_patch_overlaps_target",
                    "hidden test patch also changes code paths: " + ", ".join(overlap),
                )
            changed_functions = _conversion_function_names(analysis)
            function_count = sum(
                len(functions) for functions in changed_functions.values()
            )
            if len(analysis.paths) > 1 or function_count > 1:
                multi_file += 1
            else:
                single_file += 1
        except SkipInstance as exc:
            skipped += 1
            counts[exc.reason] += 1
    return ConversionSummary(
        considered=len(instances),
        would_emit_multi_file=multi_file,
        would_emit_single_file=single_file,
        skipped=skipped,
        reason_counts=dict(sorted(counts.items())),
    )


def build_corpus(
    instances: Sequence[SWEInstance],
    *,
    output: Path | str,
    skipped_output: Path | str,
    cache_dir: Path | str,
    limit: int,
    materialize: bool,
) -> BuildReport:
    """Emit the first N convertible rows and a complete named skip report."""
    if limit < 1:
        raise AdapterError("limit must be at least 1")
    if not materialize:
        has_convertible_shape = False
        for instance in instances:
            try:
                analyze_gold_patch(instance.patch)
            except SkipInstance:
                continue
            has_convertible_shape = True
            break
        if has_convertible_shape:
            raise MaterializationRequired(
                "SWE-bench source conversion requires checkout access; pass --materialize"
            )

    materializer = WorkspaceMaterializer(cache_dir)
    emitted: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for instance in instances:
        try:
            analyze_gold_patch(instance.patch)
        except SkipInstance as exc:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": exc.reason,
                "detail": exc.detail,
            }
            skipped.append(record)
            counts[exc.reason] += 1
            continue
        if len(emitted) >= limit:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": "limit_reached",
                "detail": f"first {limit} convertible instances already emitted",
            }
            skipped.append(record)
            counts["limit_reached"] += 1
            continue
        try:
            row = convert_instance(instance, materializer=materializer)
        except SkipInstance as exc:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": exc.reason,
                "detail": exc.detail,
            }
            skipped.append(record)
            counts[exc.reason] += 1
            continue
        except AdapterError as exc:
            record = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "status": "SKIPPED",
                "reason": "materialization_failed",
                "detail": str(exc)[:1000],
            }
            skipped.append(record)
            counts["materialization_failed"] += 1
            continue
        emitted.append(row)

    _write_jsonl(Path(output), emitted)
    skipped_doc = {
        "schema": "swebench-lite-skipped/1",
        "status": "SKIPPED",
        "dataset_id": DATASET_ID,
        "considered": len(instances),
        "emitted": len(emitted),
        "skipped": len(skipped),
        "reason_counts": dict(sorted(counts.items())),
        "items": skipped,
    }
    skipped_path = Path(skipped_output)
    skipped_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path.write_text(
        json.dumps(skipped_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return BuildReport(tuple(emitted), tuple(skipped), dict(sorted(counts.items())))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a local SWE-bench-Lite JSONL into repair rows."
    )
    parser.add_argument(
        "--source", required=True, type=Path, help="local JSONL; never fetched here"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skipped-output", type=Path, default=DEFAULT_SKIPPED_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--limit", type=int, default=10, help="number of convertible rows to emit"
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="explicitly allow git clone/worktree creation in --cache-dir",
    )
    parser.add_argument(
        "--conversion-only",
        action="store_true",
        help="census patch shapes without cloning repositories or writing rows",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        instances = load_instances(args.source)
        if args.conversion_only:
            summary = summarize_conversion(instances)
            print(json.dumps(summary.as_dict(), sort_keys=True))
            return 0
        report = build_corpus(
            instances,
            output=args.output,
            skipped_output=args.skipped_output,
            cache_dir=args.cache_dir,
            limit=args.limit,
            materialize=args.materialize,
        )
    except (AdapterError, OSError) as exc:
        print(f"swebench-adapter: {exc}", file=sys.stderr)
        return 2
    print(
        f"emitted={len(report.emitted)} skipped={len(report.skipped)} "
        f"reasons={dict(report.reason_counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
