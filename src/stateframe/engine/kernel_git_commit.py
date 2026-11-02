"""Opt-in commit and branch tools (STATEFRAME_GIT_COMMIT): reviewable commits of the run's work.

Off by default. When on, the model gets `git_commit(message)` and `git_branch(name)`. Neither touches the
user's repository or the workspace: a commit snapshots the current workspace into the run's content store
and appends one line to `run/commits.jsonl`; a branch line starts a new series from the current branch's
head. After the run, the host turns the log into one `git format-patch` series per branch under
`run/commits/` (see `stateframe.commit_export`), ready for `git am`.

Both tools are observations of the workspace: on replay the recorded reply is served and the log line
written by the original call must be present, so a resumed run neither commits twice nor loses a commit.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

MODE_ENV = "STATEFRAME_GIT_COMMIT"
_MODES = frozenset({"off", "on"})
LOG_NAME = "commits.jsonl"
DEFAULT_BRANCH = "main"
MAX_MESSAGE = 2000
MAX_COMMITS = 200
_BRANCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")

TOOL_NAMES = ("git_commit", "git_branch")


def _schema(name, description, properties, required):
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": {**properties, "why": {"type": "string"}},
                       "required": required, "additionalProperties": False}}}


SCHEMAS = (
    _schema("git_commit", "Record the current workspace as a commit on the current branch. The user receives "
            "each branch as a patch series after the run. Does not change files.", {
        "message": {"type": "string", "maxLength": MAX_MESSAGE}}, ["message"]),
    _schema("git_branch", "Start a new branch from the current branch's latest commit; later commits go to it.", {
        "name": {"type": "string", "maxLength": 63}}, ["name"]),
)


def mode(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    value = str(source.get(MODE_ENV, "off")).strip().lower()
    if value not in _MODES:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    return mode(environ) == "on"


def registered_names() -> frozenset[str]:
    """Names for the engine's static tool registries: empty when off, so no registry, journal record or
    request changes by a single byte unless the switch is on. The engine reads its environment once."""
    return frozenset(TOOL_NAMES) if enabled() else frozenset()


def registered_schemas() -> tuple:
    return SCHEMAS if enabled() else ()


class CommitLog:
    """Append-only, fsynced log of commit and branch records for one run."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / LOG_NAME

    def entries(self) -> list[dict]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    def append(self, entry: Mapping[str, object]) -> None:
        line = json.dumps(dict(entry), sort_keys=True, ensure_ascii=False) + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def by_call(self, call_id: str) -> dict | None:
        return next((entry for entry in self.entries() if entry.get("call_id") == call_id), None)


def current_branch(entries: list[dict]) -> str:
    branches = [entry["name"] for entry in entries if entry.get("kind") == "branch"]
    return branches[-1] if branches else DEFAULT_BRANCH


def branch_head_tree(entries: list[dict], branch: str) -> str | None:
    """The tree of the branch's latest commit, following a branch back to where it started."""
    for entry in reversed(entries):
        if entry.get("kind") == "commit" and entry.get("branch") == branch:
            return entry["tree"]
        if entry.get("kind") == "branch" and entry.get("name") == branch:
            return entry.get("start_tree")
    return None


def tree_record(manifest: Mapping[str, object], excluded) -> dict[str, list]:
    """Regular files and symlinks of a workspace manifest, minus secret paths: path -> [kind, mode, digest]."""
    result = {}
    for path, entry in manifest.items():
        if entry.kind in {"file", "symlink"} and not excluded(path):
            result[path] = [entry.kind, entry.mode, entry.digest]
    return dict(sorted(result.items()))


def commit(log: CommitLog, store, manifest, workspace: Path, call_id: str, message: object, excluded) -> str:
    if not isinstance(message, str) or not message.strip():
        return "REFUSED BAD_ARGUMENT: message must be nonempty text"
    if len(message) > MAX_MESSAGE:
        return f"REFUSED BAD_ARGUMENT: message exceeds {MAX_MESSAGE} characters"
    entries = log.entries()
    if sum(1 for entry in entries if entry.get("kind") == "commit") >= MAX_COMMITS:
        return f"REFUSED COMMIT_LIMIT: at most {MAX_COMMITS} commits per run"
    store.capture(workspace, manifest)
    files = tree_record(manifest, excluded)
    tree = store.put(json.dumps(files, sort_keys=True).encode())
    branch = current_branch(entries)
    if tree == branch_head_tree(entries, branch):
        return f"REFUSED NOTHING_TO_COMMIT: the workspace matches the latest commit on {branch}"
    sequence = 1 + sum(1 for entry in entries if entry.get("kind") == "commit")
    log.append({"kind": "commit", "call_id": call_id, "seq": sequence, "branch": branch,
                "message": message.strip(), "tree": tree})
    return f"commit {sequence} on {branch}: {message.strip().splitlines()[0][:120]}"


def branch(log: CommitLog, call_id: str, name: object) -> str:
    if not isinstance(name, str) or not _BRANCH.match(name) or name.endswith(".lock"):
        return "REFUSED BAD_ARGUMENT: branch name must be 1-63 letters, digits, '.', '_' or '-'"
    entries = log.entries()
    names = {DEFAULT_BRANCH, *(entry["name"] for entry in entries if entry.get("kind") == "branch")}
    if name in names:
        return f"REFUSED BRANCH_EXISTS: {name}"
    parent = current_branch(entries)
    log.append({"kind": "branch", "call_id": call_id, "name": name, "from": parent,
                "start_tree": branch_head_tree(entries, parent)})
    return f"branch {name} started from {parent}; later commits go to {name}"
