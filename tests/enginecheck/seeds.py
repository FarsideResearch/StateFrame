"""Race-regression seed registry for enginecheck.

Seeds are regression tests under tests/race_seeds/ that must prove they catch a race. Each one is marked with
`@pytest.mark.race_seed(seed_id, known_bad=<sha>, expect=...)` or `@pytest.mark.race_seed(seed_id, mutant=<name>, expect=...)`.

Seed-author contract:
- Exactly one of known_bad (hex commit sha that still has the race) or mutant (a name for an injected race).
- expect is the intended failure. An exception class name (AssertionError) must equal the failure type exactly. Any
  other text is matched as a substring of the failure message. Prefer an exception name for known_bad seeds: a symbol
  missing from the bad tree raises AttributeError or TypeError, and matching on that text is fragile.
- Mutant seeds read os.environ["ENGINECHECK_RACE_MUTANT"] and activate their own mutant only when it equals their name.
  The GREEN run and known_bad runs always get that variable set to "" so a parent env cannot leak a mutant in.
- Known-bad seeds run on `git archive <sha>` of the bad tree. The bad tree keeps its own tests/conftest.py on purpose:
  that is the code the seed targets. The seed harness (tests/race_seeds/, tests/enginecheck/ and the seed file) is
  overlaid from committed HEAD, never from the working tree, so uncommitted edits cannot change the verdict.
- Seeds must live under tests/race_seeds/. A race_seed marker anywhere else under tests/ fails collection.

prove_seed: GREEN means the seed passes on the current tree. RED means it fails through `expect` on the known-bad tree
or with its mutant active. A seed is ok only when both hold.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# A direct run starts with tests/enginecheck/ on sys.path, not tests/. Put tests/ (for enginecheck) and src/ next to it.
_TESTS_DIR = Path(__file__).resolve().parents[1]
for _path in (_TESTS_DIR, _TESTS_DIR.parent / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from enginecheck.proof import red_proof, run_nodes  # noqa: E402

_SEED_DIR = "tests/race_seeds"
_HARNESS_PATHS = ("tests/enginecheck", "tests/race_seeds")
_MUTANT_ENV = "ENGINECHECK_RACE_MUTANT"
_DUMP_ENV = "ENGINECHECK_SEED_DUMP"
_COLLECT_TIMEOUT_SECONDS = 600
_GIT_TIMEOUT_SECONDS = 120
_TAIL_CHARS = 1000
_DETAIL_CHARS = 300


class SeedRegistryError(RuntimeError):
    """Collection or registry failure: a broken marker, a duplicate seed id, a stray marker or a git/export error."""


@dataclass(frozen=True)
class Seed:
    seed_id: str
    node_id: str
    known_bad: str | None
    mutant: str | None
    expect: str


@dataclass(frozen=True)
class SeedProof:
    seed_id: str
    green: bool
    red_intended: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.green and self.red_intended


def collect_seeds(repo: Path) -> list[Seed]:
    """Collect every race_seed under tests/race_seeds with strict markers. The conftest validates and dumps them."""
    with tempfile.TemporaryDirectory(prefix="enginecheck-seeds-") as tmp:
        dump = Path(tmp) / "seeds.json"
        command = [
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--rootdir={repo.resolve()}",
            "--strict-markers", "--collect-only", "-q", _SEED_DIR,
        ]
        env = {key: value for key, value in os.environ.items() if key != "PYTEST_CURRENT_TEST"}
        env[_DUMP_ENV] = str(dump)
        try:
            proc = subprocess.run(
                command, cwd=repo, env=env, capture_output=True, text=True,
                timeout=_COLLECT_TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SeedRegistryError(f"seed collection exceeded {_COLLECT_TIMEOUT_SECONDS}s") from exc
        if proc.returncode != 0:
            tail = (proc.stdout + proc.stderr)[-_TAIL_CHARS:].strip()
            raise SeedRegistryError(f"seed collection failed (exit {proc.returncode}): {tail}")
        strays = _stray_markers(repo)
        if strays:
            raise SeedRegistryError(f"race_seed marker outside tests/race_seeds in: {', '.join(strays)}")
        try:
            raw = json.loads(dump.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SeedRegistryError("seed collection wrote no valid dump") from exc
    return _build_registry(raw)


def _stray_markers(repo: Path) -> list[str]:
    """Files under tests/ outside tests/race_seeds that use a `.race_seed` attribute. AST-based, so strings do not match."""
    seed_root = (repo / _SEED_DIR).resolve()
    strays: list[str] = []
    for path in sorted((repo / "tests").rglob("*.py")):
        if path.resolve().is_relative_to(seed_root):
            continue
        rel = path.relative_to(repo).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            raise SeedRegistryError(f"cannot scan {rel} for race_seed markers: {exc}") from exc
        if any(isinstance(node, ast.Attribute) and node.attr == "race_seed" for node in ast.walk(tree)):
            strays.append(rel)
    return strays


def _build_registry(raw: object) -> list[Seed]:
    if not isinstance(raw, list):
        raise SeedRegistryError("seed dump is not a list")
    seeds: list[Seed] = []
    by_id: dict[str, str] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise SeedRegistryError("seed dump entry is not an object")
        seed = Seed(
            seed_id=_text(entry, "seed_id"),
            node_id=_text(entry, "node_id"),
            known_bad=_optional_text(entry, "known_bad"),
            mutant=_optional_text(entry, "mutant"),
            expect=_text(entry, "expect"),
        )
        if seed.seed_id in by_id:
            raise SeedRegistryError(
                f"duplicate seed id {seed.seed_id!r}: {by_id[seed.seed_id]} and {seed.node_id}"
            )
        by_id[seed.seed_id] = seed.node_id
        seeds.append(seed)
    return seeds


def _text(entry: Mapping[str, object], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise SeedRegistryError(f"seed dump field {key!r} missing or empty")
    return value


def _optional_text(entry: Mapping[str, object], key: str) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SeedRegistryError(f"seed dump field {key!r} must be a non-empty string or null")
    return value


def prove_seed(seed: Seed, repo: Path, tmp: Path) -> SeedProof:
    """Prove one seed: GREEN on the current tree, RED through `expect` on the known-bad tree or with its mutant.

    Raises SeedRegistryError when the known-bad tree or the committed harness cannot be exported. Export dirs are
    created under `tmp` and removed on every exit path.
    """
    green_report = run_nodes([seed.node_id], cwd=repo, env={_MUTANT_ENV: ""})
    green = green_report.ok
    green_detail = "green" if green else f"green-fail: {green_report.summary()}"

    if seed.mutant is not None:
        red = red_proof(seed.node_id, cwd=repo, expect=seed.expect, env={_MUTANT_ENV: seed.mutant})
    else:
        tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"enginecheck-known-bad-{seed.seed_id}-", dir=tmp) as export_root:
            export_dir = _export_known_bad(seed, repo, Path(export_root))
            red = red_proof(seed.node_id, cwd=export_dir, expect=seed.expect, env={_MUTANT_ENV: ""})

    detail = (
        f"{green_detail}; red status={red.status} intended={red.intended} "
        f"message={red.message[:_DETAIL_CHARS]!r}"
    )
    return SeedProof(seed.seed_id, green, red.intended, detail)


def _export_known_bad(seed: Seed, repo: Path, export_root: Path) -> Path:
    """Build the known-bad tree: `git archive <known_bad>`, then the committed seed harness from HEAD over it."""
    if seed.known_bad is None:
        raise SeedRegistryError("seed has no known_bad sha")
    dest = export_root / "tree"
    dest.mkdir()
    _extract(_git_archive(repo, seed.known_bad, ()), dest, required=())

    seed_file = seed.node_id.split("::", 1)[0]
    harness = (*_HARNESS_PATHS, seed_file)
    for rel in harness:
        stale = dest / rel
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()
    _extract(_git_archive(repo, "HEAD", harness), dest, required=harness)
    return dest


def _git_archive(repo: Path, rev: str, paths: tuple[str, ...]) -> bytes:
    command = ["git", "-C", str(repo), "archive", "--format=tar", rev, *paths]
    try:
        proc = subprocess.run(command, capture_output=True, check=True, timeout=_GIT_TIMEOUT_SECONDS)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "replace").strip()
        raise SeedRegistryError(f"git archive {rev} failed: {stderr or exc}") from exc
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise SeedRegistryError(f"git archive {rev} failed: {exc}") from exc
    return proc.stdout


def _extract(archive: bytes, dest: Path, required: tuple[str, ...]) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        names = {name.rstrip("/") for name in tar.getnames()}
        for rel in required:
            if not any(name == rel or name.startswith(rel + "/") for name in names):
                raise SeedRegistryError(f"harness path missing in committed HEAD: {rel}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def _seed_row(seed: Seed) -> dict[str, str | None]:
    return {
        "seed_id": seed.seed_id,
        "node_id": seed.node_id,
        "known_bad": seed.known_bad,
        "mutant": seed.mutant,
        "expect": seed.expect,
    }


def _select(seeds: list[Seed], requested: list[str] | None) -> list[Seed]:
    """Return the requested seeds sorted by id, or every seed. Unknown ids raise SeedRegistryError."""
    if requested is None:
        return seeds
    by_id = {seed.seed_id: seed for seed in seeds}
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise SeedRegistryError(f"unknown seed id(s) {unknown}; registered: {sorted(by_id)}")
    return [by_id[seed_id] for seed_id in sorted(set(requested))]


def _cmd_list(seeds: list[Seed]) -> int:
    print(json.dumps([_seed_row(seed) for seed in seeds], indent=2))
    return 0


def _cmd_prove(seeds: list[Seed], repo: Path, requested: list[str] | None) -> int:
    selected = _select(seeds, requested)
    if not selected:
        raise SeedRegistryError("no race seeds registered under tests/race_seeds; nothing to prove")
    failed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="enginecheck-seeds-prove-") as tmp:
        for seed in selected:
            proof = prove_seed(seed, repo, Path(tmp) / seed.seed_id)
            print(f"{proof.seed_id} {'OK' if proof.ok else 'NOT_OK'} {proof.detail}")
            if not proof.ok:
                failed.append(proof.seed_id)
    if failed:
        print(f"SEEDS NOT_OK failed={failed!r}")
        return 1
    print(f"SEEDS OK n={len(selected)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seeds.py", description="List or prove enginecheck race_seed regression seeds."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    list_cmd = commands.add_parser("list", help="print registered seeds as a JSON array sorted by seed_id")
    list_cmd.add_argument("--repo", required=True, type=Path, help="repo root holding tests/race_seeds")
    prove_cmd = commands.add_parser("prove", help="prove seeds (GREEN on tree, RED through expect)")
    prove_cmd.add_argument("--repo", required=True, type=Path, help="repo root holding tests/race_seeds")
    prove_cmd.add_argument(
        "--seed", nargs="+", action="extend", metavar="NN", help="seed ids to prove (default: all)"
    )
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    try:
        seeds = collect_seeds(repo)
        if args.command == "list":
            return _cmd_list(seeds)
        return _cmd_prove(seeds, repo, args.seed)
    except SeedRegistryError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
