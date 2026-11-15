"""Race-regression seed registry for enginecheck.

Seeds are regression tests under tests/race_seeds/ that must prove they catch a race. Each one is marked with
`@pytest.mark.race_seed(seed_id, known_bad=<sha>, expect=...)` or `@pytest.mark.race_seed(seed_id, mutant=<name>, expect=...)`.

Seed-author contract:
- Exactly one of known_bad (hex commit sha that still has the race) or mutant (a name for an injected race).
- expect is the intended failure: an exception class name (AssertionError) or a substring of the failure message.
- Mutant seeds read os.environ["ENGINECHECK_RACE_MUTANT"] and activate their own mutant only when it equals their name.
- Known-bad seeds run against `git archive <sha>` with the seed file, tests/race_seeds/ and tests/enginecheck/ copied in.

prove_seed: GREEN means the seed passes on the current tree. RED means it fails through `expect` on the known-bad tree
or with its mutant active. A seed is ok only when both hold.
"""
from __future__ import annotations

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
from typing import Mapping

from enginecheck.proof import red_proof, run_nodes

_SEED_DIR = "tests/race_seeds"
_SEED_DIRS = ("tests/enginecheck",)
_SEED_FILES = ("tests/race_seeds/__init__.py", "tests/race_seeds/conftest.py")
_MUTANT_ENV = "ENGINECHECK_RACE_MUTANT"
_DUMP_ENV = "ENGINECHECK_SEED_DUMP"
_COLLECT_TIMEOUT_SECONDS = 600
_GIT_TIMEOUT_SECONDS = 120
_TAIL_CHARS = 1000
_DETAIL_CHARS = 300


class SeedRegistryError(RuntimeError):
    """Collection or registry failure: a broken marker, a duplicate seed id or a collection error."""


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
    """Collect every race_seed under tests/race_seeds. The conftest validates markers and dumps them as JSON."""
    with tempfile.TemporaryDirectory(prefix="enginecheck-seeds-") as tmp:
        dump = Path(tmp) / "seeds.json"
        command = [
            sys.executable, "-m", "pytest", "-p", "no:cacheprovider", f"--rootdir={repo.resolve()}",
            "--collect-only", "-q", _SEED_DIR,
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
        try:
            raw = json.loads(dump.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SeedRegistryError("seed collection wrote no valid dump") from exc
    return _build_registry(raw)


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
    """Prove one seed: GREEN on the current tree, RED through `expect` on the known-bad tree or with its mutant."""
    green_report = run_nodes([seed.node_id], cwd=repo)
    green = green_report.ok
    green_detail = "green" if green else f"green-fail: {green_report.summary()}"

    if seed.mutant is not None:
        red = red_proof(seed.node_id, cwd=repo, expect=seed.expect, env={_MUTANT_ENV: seed.mutant})
    else:
        try:
            export_dir = _export_known_bad(seed, repo, tmp)
        except SeedRegistryError as exc:
            return SeedProof(seed.seed_id, green, False, f"{green_detail}; known_bad export failed: {exc}")
        red = red_proof(seed.node_id, cwd=export_dir, expect=seed.expect)

    detail = (
        f"{green_detail}; red status={red.status} intended={red.intended} "
        f"message={red.message[:_DETAIL_CHARS]!r}"
    )
    return SeedProof(seed.seed_id, green, red.intended, detail)


def _export_known_bad(seed: Seed, repo: Path, tmp: Path) -> Path:
    """Extract `git archive <known_bad>` into a fresh dir under tmp, then overlay the seed harness from the current tree."""
    if seed.known_bad is None:
        raise SeedRegistryError("seed has no known_bad sha")
    try:
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", seed.known_bad],
            capture_output=True, check=True, timeout=_GIT_TIMEOUT_SECONDS,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise SeedRegistryError(f"git archive {seed.known_bad} failed: {exc}") from exc

    dest = tmp / f"known-bad-{seed.seed_id}"
    try:
        dest.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SeedRegistryError(f"export dir {dest} already exists") from exc
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)

    seed_file = seed.node_id.split("::", 1)[0]
    for rel in (seed_file, *_SEED_FILES):
        source = repo / rel
        if not source.is_file():
            raise SeedRegistryError(f"harness file missing in tree: {rel}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for rel in _SEED_DIRS:
        if not (repo / rel).is_dir():
            raise SeedRegistryError(f"harness dir missing in tree: {rel}")
        shutil.copytree(
            repo / rel, dest / rel, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    return dest
