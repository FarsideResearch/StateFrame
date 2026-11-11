"""Turn a run's commit log (STATEFRAME_GIT_COMMIT) into reviewable `git format-patch` series.

The engine records each `git_commit` as a snapshot in the run's content store and a line in
`run/commits.jsonl`. After the run, this module replays the log into a private repository inside the run
folder, whose first commit is the run's starting snapshot, and writes one patch series per branch to
`run/commits/<branch>/`. The user applies a series with `git am`. The user's own repository is never read
or written, and git runs with hooks, system and global configuration disabled.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

AUTHOR = ("StateFrame", "stateframe@localhost")
# Fixed, increasing timestamps keep commit ids identical when a run is exported again.
EPOCH = 1_700_000_000
_STARTUP_HOME_IDENTITY_UNSET = object()


class CommitExportError(RuntimeError):
    pass


def _git_env(home: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"})
    return env


def _git(repo: Path, env: dict[str, str], *args: str, stdin: bytes | None = None) -> bytes:
    command = ["git", "-c", "core.hooksPath=" + os.devnull, "-c", "commit.gpgsign=false",
               "-c", "core.autocrlf=false", "-c", "core.fsmonitor=false", "-C", str(repo), *args]
    done = subprocess.run(command, env=env, input=stdin, capture_output=True, timeout=120)
    if done.returncode != 0:
        raise CommitExportError(f"git {args[0]} failed: {done.stderr.decode(errors='replace')[-400:]}")
    return done.stdout


def _write_tree(worktree: Path, files: dict[str, list], blob) -> None:
    for child in worktree.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for relative, (kind, mode, digest) in sorted(files.items()):
        target = worktree / relative
        if ".git" in Path(relative).parts:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = blob(digest)
        if kind == "symlink":
            os.symlink(data.decode("utf-8", errors="surrogateescape"), target)
        else:
            target.write_bytes(data)
            os.chmod(target, 0o755 if mode & 0o111 else 0o644)


def _commit(repo: Path, env: dict[str, str], message: str, index: int) -> str:
    _git(repo, env, "add", "-A", "--", ".")
    stamp = f"{EPOCH + index * 60} +0000"
    commit_env = {**env, "GIT_AUTHOR_NAME": AUTHOR[0], "GIT_AUTHOR_EMAIL": AUTHOR[1], "GIT_AUTHOR_DATE": stamp,
                  "GIT_COMMITTER_NAME": AUTHOR[0], "GIT_COMMITTER_EMAIL": AUTHOR[1], "GIT_COMMITTER_DATE": stamp}
    _git(repo, commit_env, "commit", "-q", "--allow-empty", "--no-verify", "-F", "-", stdin=message.encode())
    return _git(repo, env, "rev-parse", "HEAD").decode().strip()


def export_commits(
        run_dir: Path, starting_tree: dict[str, list] | None = None, *,
        startup_home_identity: object = _STARTUP_HOME_IDENTITY_UNSET,
) -> dict | None:
    """Write `run/commits/<branch>/NNNN-*.patch`. Returns a summary, or None when nothing was committed."""
    run_dir = Path(run_dir)
    log_path = run_dir / "commits.jsonl"
    if not log_path.is_file():
        return None
    entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    commits = [entry for entry in entries if entry.get("kind") == "commit"]
    if not commits:
        return None
    if shutil.which("git") is None:
        raise CommitExportError("git is not installed; commit snapshots remain in commits.jsonl")
    from .engine import kernel_checkpoint

    def verify_home_identity() -> None:
        if startup_home_identity is _STARTUP_HOME_IDENTITY_UNSET:
            kernel_checkpoint.verify_stateframe_home_identity(run_dir)
        else:
            kernel_checkpoint.verify_stateframe_home_identity(
                run_dir, expected_identity=startup_home_identity)

    try:
        verify_home_identity()
    except kernel_checkpoint.CheckpointError as error:
        raise CommitExportError(kernel_checkpoint._failure_reason(error)) from error
    captured_secret_generation = kernel_checkpoint._SECRET_GENERATION
    import sys
    engine = Path(__file__).resolve().parent / "engine"
    sys.path.insert(0, str(engine))
    try:
        from kernel_effects import ContentStore
    finally:
        sys.path.remove(str(engine))
    store = ContentStore(run_dir)

    def checked_bytes(data: bytes) -> bytes:
        # Git can encode binary payloads, paths and messages; inspect their original bytes.
        try:
            kernel_checkpoint._check_registered_secrets(data)
        except kernel_checkpoint.CheckpointError as error:
            raise CommitExportError("checkpoint refused: registered secret in state") from error
        return data

    def checked_blob(digest: str) -> bytes:
        return checked_bytes(store.get(digest))

    def tree(digest: str) -> dict[str, list]:
        return json.loads(checked_blob(digest))

    def checked_tree(files: dict[str, list]) -> dict[str, list]:
        for relative in files:
            checked_bytes(os.fsencode(relative))
        return files

    if starting_tree is None:
        starting_tree = _starting_tree(run_dir, store)
    output = run_dir / "commits"
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise CommitExportError("commit export destination is not a regular directory")
    stage = Path(tempfile.mkdtemp(prefix=".commits-export-", dir=run_dir))
    backup = run_dir / f".commits-previous-{uuid.uuid4().hex}"
    try:
        with tempfile.TemporaryDirectory(prefix="commit-export-", dir=run_dir) as scratch:
            scratch_path = Path(scratch)
            home = scratch_path / "home"
            home.mkdir()
            env = _git_env(home)
            repo = scratch_path / "repo"
            repo.mkdir()
            _git(repo, env, "init", "-q", "-b", "main")
            _write_tree(repo, checked_tree(starting_tree), checked_blob)
            base = _commit(repo, env, "Starting state", 0)
            heads = {"main": base}
            starts = {"main": base}
            current = "main"
            written: dict[str, int] = {}
            for index, entry in enumerate(entries, start=1):
                if entry.get("kind") == "branch":
                    checked_bytes(os.fsencode(entry["name"]))
                    heads[entry["name"]] = starts[entry["name"]] = heads[entry["from"]]
                    current = entry["name"]
                    continue
                _git(repo, env, "checkout", "-q", "-B", current, heads[current])
                _write_tree(repo, checked_tree(tree(entry["tree"])), checked_blob)
                checked_bytes(entry["message"].encode())
                heads[current] = _commit(repo, env, entry["message"], index)
                written[current] = written.get(current, 0) + 1
            summary = {"count": len(commits), "dir": str(output), "branches": {}}
            for name, count in written.items():
                start = starts[name]
                target = stage / name
                target.mkdir()
                _git(repo, env, "format-patch", "-q", "--no-signature", "-o", str(target),
                     f"{start}..{heads[name]}")
                summary["branches"][name] = {
                    "commits": count, "patches": sorted(p.name for p in target.iterdir())}
        try:
            for directory, _, filenames in os.walk(stage, followlinks=False):
                for filename in filenames:
                    artifact = Path(directory) / filename
                    if artifact.is_symlink() or not artifact.is_file():
                        raise CommitExportError("commit export contains a non-regular artifact")
                    kernel_checkpoint.check_registered_secret_file(artifact)
        except kernel_checkpoint.CheckpointError as error:
            raise CommitExportError(kernel_checkpoint._failure_reason(error)) from error
        verify_home_identity()
        if kernel_checkpoint._SECRET_GENERATION != captured_secret_generation:
            raise kernel_checkpoint.CheckpointError("STATEFRAME_HOME_IDENTITY_CHANGED")
        if output.exists():
            os.replace(output, backup)
        try:
            os.replace(stage, output)
        except OSError:
            if backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return summary
    except kernel_checkpoint.CheckpointError as error:
        raise CommitExportError(kernel_checkpoint._failure_reason(error)) from error
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _starting_tree(run_dir: Path, store) -> dict[str, list]:
    """The run's starting snapshot (run/parent) as a tree record, captured into the store."""
    import sys
    engine = Path(__file__).resolve().parent / "engine"
    sys.path.insert(0, str(engine))
    try:
        from kernel_effects import scan
        from kernel_git_commit import tree_record
        from kernel_git_view import excluded_path
    finally:
        sys.path.remove(str(engine))
    parent = run_dir / "parent"
    manifest = scan(parent, exclude=lambda _: False)
    store.capture(parent, manifest)
    return tree_record(manifest, excluded_path)
