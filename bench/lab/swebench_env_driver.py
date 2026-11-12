"""Maintainer-only batch/corpus driver; absent from release artifacts."""
from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys

_ENGINE = _Path(__file__).resolve().parents[2] / "src/stateframe/engine"
if str(_ENGINE) not in _sys.path:
    _sys.path.insert(0, str(_ENGINE))

import argparse  # noqa: E402 - engine path is set above
from build_tasks.swebench_adapter import make_runner_test, verify_instance  # noqa: E402 - engine path is set above
from build_tasks.swebench_env import (  # noqa: E402 - engine path is set above
    DEFAULT_TIMEOUT_SECONDS,
    EnvironmentBatch,
    EnvironmentManager,
    EnvironmentProvider,
    EnvironmentResult,
    Mapping,
    Path,
    Sequence,
    WorkspaceLookupError,
    _safe_instance_id,
    _tail,
    dataclass,
    environment_dir,
    find_workspace,
    json,
    os,
    sys,
    tempfile,
)


@dataclass(frozen=True)
class ProofRecord:
    """One row's environment/setup status and optional two-sided oracle evidence."""

    instance_id: str
    python: Path | None
    install_ok: bool
    base_red: bool | None
    gold_green: bool | None
    blocker: str | None
    base_tail: str
    gold_tail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "python": str(self.python) if self.python else None,
            "install_ok": self.install_ok,
            "base_red": self.base_red,
            "gold_green": self.gold_green,
            "blocker": self.blocker,
            "base_tail": self.base_tail,
            "gold_tail": self.gold_tail,
        }


@dataclass(frozen=True)
class ProofReport:
    """All enriched rows and proof records produced by one bounded run."""

    batch: EnvironmentBatch
    proofs: tuple[ProofRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": len(self.batch.rows),
            "environments": [
                environment.as_dict() for environment in self.batch.environments
            ],
            "proofs": [proof.as_dict() for proof in self.proofs],
        }


def _row_instance_id(row: Mapping[str, object]) -> str:
    value = row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("row.instance_id must be a non-empty string")
    return _safe_instance_id(value)


def _refresh_runner(row: dict[str, object]) -> None:
    """Regenerate the grader-only bridge so persisted rows use current runner rules."""
    repo = row.get("repo")
    selectors_value = row.get("fail_to_pass")
    if not isinstance(repo, str) or not isinstance(selectors_value, (list, tuple)):
        return
    selectors = tuple(
        selector for selector in selectors_value if isinstance(selector, str)
    )
    if len(selectors) != len(selectors_value) or not selectors:
        return
    row["test_src"] = make_runner_test(selectors, repo=repo)


def load_rows(path: Path | str) -> list[dict[str, object]]:
    """Load emitted JSONL rows without importing the dataset layer."""
    source = Path(path)
    rows: list[dict[str, object]] = []
    for line_no, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_no}: row must be a JSON object")
        rows.append(dict(value))
    return rows


def write_rows(path: Path | str, rows: Sequence[Mapping[str, object]]) -> None:
    """Write enriched rows through a same-directory temporary file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.tmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    cache_dir: Path | str,
    workspace_cache_dir: Path | str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    manager: EnvironmentProvider | None = None,
    output: Path | str | None = None,
) -> EnvironmentBatch:
    """Install each row's environment and add its absolute ``python`` path."""
    provider = manager or EnvironmentManager(cache_dir, timeout=timeout)
    workspace_root = Path(workspace_cache_dir or cache_dir)
    enriched: list[dict[str, object]] = []
    environments: list[EnvironmentResult] = []
    for source_row in rows:
        row = dict(source_row)
        instance_id = _row_instance_id(row)
        _refresh_runner(row)
        try:
            workspace = find_workspace(workspace_root, row)
            result = provider.prepare(
                instance_id,
                workspace / "base",
                repo=row.get("repo") if isinstance(row.get("repo"), str) else None,
            )
        except WorkspaceLookupError as exc:
            result = EnvironmentResult(
                instance_id,
                environment_dir(cache_dir, instance_id),
                None,
                False,
                blocker=str(exc),
            )
        row["python"] = str(result.python) if result.python else None
        row["env_ready"] = result.ready
        row["env_receipts"] = [receipt.as_dict() for receipt in result.receipts]
        if result.repair_contract is not None:
            row["env_repair_contract"] = result.repair_contract.as_dict()
        else:
            row.pop("env_repair_contract", None)
        if result.blocker:
            row["env_blocker"] = result.blocker
        else:
            row.pop("env_blocker", None)
        enriched.append(row)
        environments.append(result)
    batch = EnvironmentBatch(tuple(enriched), tuple(environments))
    if output is not None:
        write_rows(output, batch.rows)
    return batch


def _verification_blocker(base_red: bool, gold_green: bool) -> str | None:
    if base_red and gold_green:
        return None
    if not base_red and not gold_green:
        return "ENV_VERIFY_ORACLE_NOT_PROVEN: base was not a strict red and gold was not green"
    if not base_red:
        return "ENV_VERIFY_PARENT_NOT_RED: FAIL_TO_PASS did not fail strictly on base"
    return "ENV_VERIFY_GOLD_NOT_GREEN: FAIL_TO_PASS did not pass strictly on gold"


def run_proof(
    rows: Sequence[Mapping[str, object]],
    *,
    cache_dir: Path | str,
    workspace_cache_dir: Path | str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    django_limit: int = 3,
    manager: EnvironmentProvider | None = None,
    output: Path | str | None = None,
) -> ProofReport:
    """Prepare all rows and verify the first ``django_limit`` Django rows."""
    if django_limit < 1:
        raise ValueError("django_limit must be at least 1")
    batch = prepare_rows(
        rows,
        cache_dir=cache_dir,
        workspace_cache_dir=workspace_cache_dir,
        timeout=timeout,
        manager=manager,
        output=output,
    )
    proofs: list[ProofRecord] = []
    django_seen = 0
    workspace_root = Path(workspace_cache_dir or cache_dir)
    for row, environment in zip(batch.rows, batch.environments, strict=True):
        instance_id = _row_instance_id(row)
        repo = row.get("repo")
        is_django = repo == "django/django"
        if not environment.ready:
            proofs.append(
                ProofRecord(
                    instance_id,
                    environment.python,
                    environment.install_ok,
                    None,
                    None,
                    environment.blocker or "ENV_SETUP_FAILED: no usable environment",
                    "NOT RUN: environment blocker",
                    "NOT RUN: environment blocker",
                )
            )
            if is_django:
                django_seen += 1
            continue
        if not is_django:
            proofs.append(
                ProofRecord(
                    instance_id,
                    environment.python,
                    environment.install_ok,
                    None,
                    None,
                    "ENV_VERIFY_NOT_REQUESTED: proof slice is scoped to Django",
                    "NOT RUN: proof slice is scoped to Django",
                    "NOT RUN: proof slice is scoped to Django",
                )
            )
            continue
        django_seen += 1
        if django_seen > django_limit:
            proofs.append(
                ProofRecord(
                    instance_id,
                    environment.python,
                    environment.install_ok,
                    None,
                    None,
                    f"ENV_VERIFY_NOT_REQUESTED: proof slice limited to first {django_limit} Django rows",
                    "NOT RUN: proof slice limit",
                    "NOT RUN: proof slice limit",
                )
            )
            continue
        try:
            workspace = find_workspace(workspace_root, row)
            verification = verify_instance(
                row,
                workspace,
                python_executable=str(environment.python),
                timeout=timeout,
            )
            base_red = (
                not verification.parent.timed_out
                and verification.parent.returncode == 1
                and verification.parent.inner_returncode == 1
            )
            gold_green = (
                not verification.gold.timed_out
                and verification.gold.returncode == 0
                and verification.gold.inner_returncode is None
            )
            proofs.append(
                ProofRecord(
                    instance_id,
                    environment.python,
                    environment.install_ok,
                    base_red,
                    gold_green,
                    _verification_blocker(base_red, gold_green),
                    _tail(verification.parent.output),
                    _tail(verification.gold.output),
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            blocker = f"ENV_VERIFY_EXCEPTION: {type(exc).__name__}: {exc}"
            proofs.append(
                ProofRecord(
                    instance_id,
                    environment.python,
                    environment.install_ok,
                    None,
                    None,
                    blocker,
                    "NOT RUN: verifier raised an environment/setup exception",
                    "NOT RUN: verifier raised an environment/setup exception",
                )
            )
    return ProofReport(batch, tuple(proofs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision per-instance SWE-bench environments and run a Django proof slice."
    )
    parser.add_argument(
        "--rows", required=True, type=Path, help="emitted SWE-bench JSONL"
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="environment cache; venvs are created under <cache-dir>/envs/",
    )
    parser.add_argument(
        "--workspace-cache-dir",
        type=Path,
        help="optional existing adapter cache containing instances/; defaults to --cache-dir",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="enriched JSONL destination; defaults to <rows stem>_env.jsonl",
    )
    parser.add_argument(
        "--proof-output", type=Path, help="optional JSON proof/evidence destination"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="total per-instance venv plus install/proof bound in seconds",
    )
    parser.add_argument(
        "--django-limit",
        type=int,
        default=3,
        help="number of Django rows to verify with the two-sided oracle",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = load_rows(args.rows)
        output = args.output or args.rows.with_name(
            f"{args.rows.stem}_env{args.rows.suffix}"
        )
        report = run_proof(
            rows,
            cache_dir=args.cache_dir,
            workspace_cache_dir=args.workspace_cache_dir,
            timeout=args.timeout,
            django_limit=args.django_limit,
            output=output,
        )
        if args.proof_output:
            args.proof_output.parent.mkdir(parents=True, exist_ok=True)
            args.proof_output.write_text(
                json.dumps(
                    report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
    except (OSError, ValueError, WorkspaceLookupError) as exc:
        print(f"swebench-env: {exc}", file=sys.stderr)
        return 2
    ready = sum(environment.ready for environment in report.batch.environments)
    blocked = len(report.batch.environments) - ready
    proven = sum(
        proof.base_red is True and proof.gold_green is True for proof in report.proofs
    )
    print("instance\tready\tpython\tblocker")
    for environment in report.batch.environments:
        print(
            "\t".join(
                (
                    environment.instance_id,
                    str(environment.ready).lower(),
                    str(environment.python) if environment.python else "null",
                    environment.blocker or "-",
                )
            )
        )
    print(
        f"rows={len(report.batch.rows)} ready={ready} blocked={blocked} proven={proven}"
    )
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
