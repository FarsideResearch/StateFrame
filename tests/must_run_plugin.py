"""Enforce the reviewed release-test matrix and record fresh run evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import pwd
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest
from _timeouts import scaled


_PROVISIONS = frozenset({
    "bwrap",
    "filesystem_modes",
    "host_sandbox",
    "parity_baseline",
    "pid_namespace",
    "sandbox_exec",
    "serial_lane",
    "unknown_uid_bwrap",
    "uv_sandbox",
})
_JOB_PATTERN = re.compile(r"[a-z][a-z0-9_-]*\Z")
_PROVISION_PATTERN = re.compile(r"[a-z][a-z0-9_]*\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_OBJECT_ID_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_NONCE_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_MAX_TICKET_AGE_NS = 6 * 60 * 60 * 1_000_000_000
_SANDBOX_DIRS = ("/usr/bin", "/bin", "/usr/local/bin", "/usr/sbin")


class GateInputError(ValueError):
    """A manifest, ticket, or repository identity is invalid."""


@dataclass(frozen=True, slots=True)
class GateJob:
    job_id: str
    platform: str
    required_provisions: tuple[str, ...]
    conditional_provisions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateCell:
    scenario_id: str
    nodeid: str
    item_nodeids: tuple[str, ...]
    jobs: dict[str, dict[str, str]]
    prerequisites: tuple[str, ...]
    optional_prerequisites: tuple[str, ...]
    baseline: str | None
    witness_schema: str
    witness_assertion: str | None


@dataclass(frozen=True, slots=True)
class OptionalDependency:
    module: str
    extra: str
    tool: str
    tests: tuple[str, ...]
    absence_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateManifest:
    jobs: dict[str, GateJob]
    baselines: dict[str, dict[str, str]]
    cells: tuple[GateCell, ...]
    optional_dependencies: tuple[OptionalDependency, ...]


@dataclass(slots=True)
class GateState:
    manifest: GateManifest
    manifest_digest: str
    must_run: bool
    shard_id: str = "full"
    is_worker: bool = False
    job_id: str | None = None
    job: GateJob | None = None
    ticket: dict[str, object] | None = None
    evidence_path: Path | None = None
    candidate_digest: str | None = None
    provisions: frozenset[str] = frozenset()
    provision_problems: dict[str, str | None] = field(default_factory=dict)
    changed_paths: frozenset[str] = frozenset()
    cells_by_nodeid: dict[str, list[GateCell]] = field(default_factory=dict)
    cells_by_item_nodeid: dict[str, list[GateCell]] = field(default_factory=dict)
    collected: dict[str, list[pytest.Item]] = field(default_factory=dict)
    selected_nodeids: set[str] = field(default_factory=set)
    reports: dict[str, dict[str, list[pytest.TestReport]]] = field(default_factory=dict)
    assertion_passes: dict[str, int] = field(default_factory=dict)
    assertion_sources: dict[str, set[str]] = field(default_factory=dict)
    assertion_witnesses: dict[str, bool] = field(default_factory=dict)
    parity_witnesses: dict[str, dict[str, object]] = field(default_factory=dict)
    active_call_ids: set[str] = field(default_factory=set)
    violations: dict[str, list[str]] = field(default_factory=dict)
    optional_skips: dict[str, dict[str, object]] = field(default_factory=dict)
    scanned_cell_ids: set[str] = field(default_factory=set)
    deferred_cell_ids: set[str] = field(default_factory=set)
    deferred_evidence: list[dict[str, object]] = field(default_factory=list)
    cell_evidence: list[dict[str, object]] = field(default_factory=list)
    summary: str = ""
    started_at_ns: int = 0


_STATE_KEY = pytest.StashKey[GateState]()
_ACTIVE_STATE: GateState | None = None


def _git(root: Path, *arguments: str,
         timeout: float = scaled(15.0)) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GateInputError(f"git command failed: {arguments[0]}: {error}") from error


def _git_text(root: Path, *arguments: str) -> str:
    result = _git(root, *arguments)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        suffix = f": {detail[0][:180]}" if detail else ""
        raise GateInputError(f"git {arguments[0]} failed{suffix}")
    return result.stdout.decode("utf-8", errors="strict").strip()


def _candidate_digest(root: Path) -> str:
    """Hash tracked and non-ignored worktree files, including modes and symlink targets."""
    result = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        suffix = f": {detail[0][:180]}" if detail else ""
        raise GateInputError(f"candidate file inventory failed{suffix}")
    digest = hashlib.sha256()
    names = sorted(set(result.stdout.split(b"\0")[:-1]))
    if not names:
        raise GateInputError(f"candidate repository has no inventoried files: {root}")
    for raw_name in names:
        relative = os.fsdecode(raw_name)
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise GateInputError(f"git returned an unsafe candidate path: {relative!r}")
        path = root.joinpath(*relative_path.parts)
        digest.update(raw_name + b"\0")
        try:
            info = path.lstat()
        except FileNotFoundError:
            digest.update(b"missing\0")
            continue
        digest.update(f"{stat.S_IFMT(info.st_mode):o}:{stat.S_IMODE(info.st_mode):o}\0".encode())
        if stat.S_ISLNK(info.st_mode):
            digest.update(os.fsencode(os.readlink(path)) + b"\0")
        elif stat.S_ISREG(info.st_mode):
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
        elif stat.S_ISDIR(info.st_mode):
            digest.update(b"directory\0")
            nested = _git(path, "rev-parse", "--verify", "HEAD^{commit}")
            if nested.returncode == 0:
                digest.update(nested.stdout.strip() + b"\0")
        else:
            raise GateInputError(f"unsupported candidate file type: {relative}")
    return digest.hexdigest()


def _manifest_path(root: Path, explicit_path: Path | None = None) -> Path:
    reviewed_path = root / "tests" / "must_run.json"
    if reviewed_path.is_symlink():
        raise GateInputError("reviewed must-run manifest cannot be a symlink")
    if explicit_path is not None:
        path = explicit_path if explicit_path.is_absolute() else root / explicit_path
    else:
        configured = os.environ.get("SF_GATE_MANIFEST", "").strip()
        path = Path(configured).expanduser() if configured else reviewed_path
        if not path.is_absolute():
            path = root / path
    try:
        resolved_path = path.resolve()
        resolved_reviewed_path = reviewed_path.resolve()
    except (OSError, RuntimeError) as error:
        raise GateInputError(f"must-run manifest path cannot be resolved: {error}") from error
    if resolved_path != resolved_reviewed_path:
        raise GateInputError("must-run manifest must be the in-repository tests/must_run.json")
    return reviewed_path


def _manifest_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise GateInputError(f"must-run manifest cannot be read: {path}: {error}") from error


def _string_tuple(value: object, *, field_name: str, pattern: re.Pattern[str] | None = None) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise GateInputError(f"manifest {field_name} must be a list of non-empty strings")
    values = tuple(value)
    if len(values) != len(set(values)):
        raise GateInputError(f"manifest {field_name} contains duplicates")
    if pattern is not None and any(pattern.fullmatch(item) is None for item in values):
        raise GateInputError(f"manifest {field_name} contains an invalid value")
    return values


def _assertion_key(source: str) -> str:
    try:
        parsed = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise GateInputError(f"invalid assertion witness: {error}") from error
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
        raise GateInputError("assertion witness must be one assert statement")
    return ast.dump(parsed.body[0].test, include_attributes=False)


def _pytest_assertion_key(source: str) -> str:
    """Normalize pytest's original assertion expression or statement."""
    try:
        expression = ast.parse(source, mode="eval").body
    except SyntaxError:
        return _assertion_key(source)
    return ast.dump(expression, include_attributes=False)


def _read_manifest(root: Path, explicit_path: Path | None = None) -> tuple[GateManifest, str]:
    path = _manifest_path(root, explicit_path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise GateInputError(f"must-run manifest cannot be read: {path}: {error}") from error
    required_keys = {"schema", "jobs", "baselines", "cells", "optional_dependencies"}
    if not isinstance(payload, dict) or set(payload) != required_keys or payload.get("schema") != 3:
        raise GateInputError(f"must-run manifest has an unsupported schema or keys: {path}")

    raw_jobs = payload["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise GateInputError("must-run manifest must define required matrix jobs")
    jobs: dict[str, GateJob] = {}
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict) or set(raw_job) != {
            "id", "platform", "required", "required_provisions", "conditional_provisions"
        }:
            raise GateInputError("invalid job entry in must-run manifest")
        job_id = raw_job["id"]
        platform_name = raw_job["platform"]
        if (not isinstance(job_id, str) or _JOB_PATTERN.fullmatch(job_id) is None
                or not isinstance(platform_name, str) or platform_name not in {"linux", "darwin"}
                or raw_job["required"] is not True):
            raise GateInputError(f"invalid required job in must-run manifest: {job_id!r}")
        required = _string_tuple(raw_job["required_provisions"], field_name="required_provisions",
                                 pattern=_PROVISION_PATTERN)
        conditional = _string_tuple(raw_job["conditional_provisions"], field_name="conditional_provisions",
                                    pattern=_PROVISION_PATTERN)
        if set(required) & set(conditional) or set(required + conditional) - _PROVISIONS:
            raise GateInputError(f"invalid provisions for required job {job_id}")
        if job_id in jobs:
            raise GateInputError(f"duplicate required job: {job_id}")
        jobs[job_id] = GateJob(job_id, platform_name, required, conditional)

    raw_baselines = payload["baselines"]
    if not isinstance(raw_baselines, dict) or not raw_baselines:
        raise GateInputError("must-run manifest must define its comparison baselines")
    baselines: dict[str, dict[str, str]] = {}
    for name, value in raw_baselines.items():
        if not isinstance(name, str) or not name or not isinstance(value, dict):
            raise GateInputError(f"invalid baseline entry: {name!r}")
        if name == "immediate":
            if value != {"source": "SF_GATE_BASE"}:
                raise GateInputError("immediate baseline must use the SF_GATE_BASE source")
            baselines[name] = {"source": "SF_GATE_BASE"}
            continue
        if set(value) != {"commit", "tree_digest"}:
            raise GateInputError(f"invalid pinned baseline: {name!r}")
        commit = value["commit"]
        tree_digest = value["tree_digest"]
        if (not isinstance(commit, str) or _OBJECT_ID_PATTERN.fullmatch(commit) is None
                or not isinstance(tree_digest, str) or _OBJECT_ID_PATTERN.fullmatch(tree_digest) is None):
            raise GateInputError(f"invalid commit or tree digest for baseline {name}")
        baselines[name] = {"commit": commit, "tree_digest": tree_digest}

    raw_cells = payload["cells"]
    if not isinstance(raw_cells, list) or not raw_cells:
        raise GateInputError("must-run manifest must contain required scenario cells")
    cells: list[GateCell] = []
    seen_scenarios: set[str] = set()
    seen_job_nodes: set[tuple[str, str]] = set()
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, dict) or set(raw_cell) != {
            "scenario_id", "nodeid", "jobs", "prerequisites", "optional_prerequisites",
            "baseline", "witness_schema", "item_nodeids", "witness_assertion",
        }:
            raise GateInputError("invalid scenario cell in must-run manifest")
        scenario_id = raw_cell["scenario_id"]
        nodeid = raw_cell["nodeid"]
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in seen_scenarios:
            raise GateInputError(f"invalid or duplicate scenario id: {scenario_id!r}")
        if not isinstance(nodeid, str) or not nodeid.strip() or nodeid.startswith("/"):
            raise GateInputError(f"invalid exact node id in scenario {scenario_id}")
        node_path, separator, test_path = nodeid.partition("::")
        relative = PurePosixPath(node_path)
        if (not separator or not test_path or relative.is_absolute() or ".." in relative.parts
                or not (root.joinpath(*relative.parts)).is_file()):
            raise GateInputError(f"scenario {scenario_id} has an invalid or missing test path: {nodeid}")
        item_nodeids = _string_tuple(raw_cell["item_nodeids"], field_name="item_nodeids")
        if not item_nodeids or any(
            item != nodeid and not (item.startswith(nodeid + "[") and item.endswith("]"))
            for item in item_nodeids
        ):
            raise GateInputError(f"scenario {scenario_id} has item IDs outside its test node")
        prerequisites = _string_tuple(raw_cell["prerequisites"], field_name="prerequisites",
                                      pattern=_PROVISION_PATTERN)
        optional_prerequisites = _string_tuple(
            raw_cell["optional_prerequisites"], field_name="optional_prerequisites",
            pattern=_PROVISION_PATTERN,
        )
        if set(prerequisites) - _PROVISIONS or set(optional_prerequisites) - set(prerequisites):
            raise GateInputError(f"invalid prerequisites in scenario {scenario_id}")
        baseline = raw_cell["baseline"]
        if baseline is not None and (not isinstance(baseline, str) or baseline not in baselines):
            raise GateInputError(f"unknown baseline in scenario {scenario_id}: {baseline!r}")
        witness_schema = raw_cell["witness_schema"]
        if not isinstance(witness_schema, str) or witness_schema not in {"assertion_match", "parity_trace"}:
            raise GateInputError(f"unknown witness schema in scenario {scenario_id}")
        if (witness_schema == "parity_trace") != (baseline is not None):
            raise GateInputError(f"parity scenarios must name exactly one baseline: {scenario_id}")
        witness_assertion = raw_cell["witness_assertion"]
        if witness_schema == "assertion_match":
            if not isinstance(witness_assertion, str) or not witness_assertion.strip():
                raise GateInputError(f"scenario {scenario_id} needs an assertion witness")
            _assertion_key(witness_assertion)
        elif witness_assertion is not None:
            raise GateInputError(f"parity scenario {scenario_id} cannot define an assertion witness")
        raw_dispositions = raw_cell["jobs"]
        if not isinstance(raw_dispositions, dict) or set(raw_dispositions) != set(jobs):
            raise GateInputError(f"scenario {scenario_id} must specify every required job")
        dispositions: dict[str, dict[str, str]] = {}
        for job_id, raw_disposition in raw_dispositions.items():
            if raw_disposition == "RUN":
                disposition = {"disposition": "RUN"}
            elif isinstance(raw_disposition, dict) and set(raw_disposition) == {"disposition", "platform", "reason"}:
                target_platform = raw_disposition["platform"]
                reason = raw_disposition["reason"]
                if (raw_disposition["disposition"] != "NA_PLATFORM"
                        or not isinstance(target_platform, str)
                        or target_platform not in {"linux", "darwin"}
                        or jobs[job_id].platform == target_platform
                        or not isinstance(reason, str) or not reason.strip()):
                    raise GateInputError(f"invalid platform disposition for {scenario_id}/{job_id}")
                disposition = {"disposition": "NA_PLATFORM", "platform": target_platform, "reason": reason}
            else:
                raise GateInputError(f"invalid disposition for scenario {scenario_id}/{job_id}")
            for prerequisite in prerequisites:
                job = jobs[job_id]
                if prerequisite not in job.required_provisions + job.conditional_provisions:
                    if disposition["disposition"] == "RUN":
                        raise GateInputError(
                            f"job {job_id} does not declare prerequisite {prerequisite} for {scenario_id}"
                        )
            if (disposition["disposition"] == "RUN"
                    and (job_id, nodeid) in seen_job_nodes):
                raise GateInputError(f"duplicate exact node id in job {job_id}: {nodeid}")
            seen_job_nodes.add((job_id, nodeid))
            dispositions[job_id] = disposition
        seen_scenarios.add(scenario_id)
        cells.append(GateCell(scenario_id, nodeid, item_nodeids, dispositions, prerequisites,
                              optional_prerequisites, baseline, witness_schema, witness_assertion))

    raw_optional = payload["optional_dependencies"]
    if not isinstance(raw_optional, list):
        raise GateInputError("optional_dependencies must be a list")
    optional_dependencies: list[OptionalDependency] = []
    seen_optional: set[tuple[str, str, tuple[str, ...]]] = set()
    for value in raw_optional:
        if not isinstance(value, dict) or set(value) != {
            "module", "extra", "tool", "tests", "absence_reasons",
        }:
            raise GateInputError("invalid optional dependency map entry")
        module = value["module"]
        extra = value["extra"]
        tool = value["tool"]
        if (not isinstance(module, str) or not module.startswith("src/stateframe/")
                or ".." in PurePosixPath(module).parts
                or not isinstance(extra, str) or not extra.strip()
                or not isinstance(tool, str) or not tool.strip()):
            raise GateInputError("invalid module, extra, or tool in optional dependency map")
        tests = _string_tuple(value["tests"], field_name="optional dependency tests")
        absence_reasons = _string_tuple(value["absence_reasons"], field_name="absence_reasons")
        if not tests or not absence_reasons:
            raise GateInputError("optional dependency entries need tests and an absence reason")
        key = (module, extra, tests)
        if key in seen_optional:
            raise GateInputError(f"duplicate optional dependency map entry: {key}")
        seen_optional.add(key)
        optional_dependencies.append(OptionalDependency(module, extra, tool, tests, absence_reasons))

    manifest = GateManifest(jobs, baselines, tuple(cells), tuple(optional_dependencies))
    manifest_digest = hashlib.sha256(raw).hexdigest()
    return manifest, manifest_digest


def _platform_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return sys.platform


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise GateInputError(f"refusing to replace existing evidence or ticket: {path}") from error
    except OSError as error:
        raise GateInputError(f"cannot create evidence or ticket {path}: {error}") from error
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise GateInputError(f"short write while creating evidence or ticket: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _prepare(root: Path, manifest_path: Path, ticket_dir: Path, evidence_dir: Path) -> None:
    manifest, manifest_digest = _read_manifest(root, manifest_path)
    candidate_digest = _candidate_digest(root)
    if ticket_dir.resolve() == evidence_dir.resolve():
        raise GateInputError("ticket and evidence directories must be different")
    for directory in (ticket_dir, evidence_dir):
        if directory.is_symlink():
            raise GateInputError(f"ticket and evidence directories cannot be symlinks: {directory}")
        if directory.exists():
            if not directory.is_dir() or any(directory.iterdir()):
                raise GateInputError(f"ticket and evidence directories must be new and empty: {directory}")
        else:
            directory.mkdir(parents=True)
    issued_at_ns = time.time_ns()
    for job in manifest.jobs.values():
        ticket = {
            "schema": 1,
            "job_id": job.job_id,
            "run_nonce": secrets.token_hex(16),
            "candidate_digest": candidate_digest,
            "manifest_digest": manifest_digest,
            "issued_at_ns": issued_at_ns,
        }
        _write_exclusive(ticket_dir / f"{job.job_id}.json", _json_bytes(ticket))
    print(f"Prepared {len(manifest.jobs)} fresh gate tickets; candidate={candidate_digest} manifest={manifest_digest}")
    for job in manifest.jobs.values():
        print(
            f"{job.job_id}: SF_GATE_JOB_ID={job.job_id} "
            f"SF_GATE_TICKET={ticket_dir / (job.job_id + '.json')} "
            f"SF_GATE_EVIDENCE={evidence_dir / (job.job_id + '.json')}"
        )


def _bwrap_problem() -> str | None:
    if not sys.platform.startswith("linux"):
        return "bwrap is only available on Linux"
    executable = _sandbox_executable("bwrap")
    if executable is None:
        return "bwrap is unavailable in PATH and standard install locations"
    return _command_problem([
        executable, "--unshare-user", "--disable-userns", "--assert-userns-disabled",
        "--ro-bind", "/", "/", "--", "/bin/sh", "-c", "exit 0",
    ])


def _sandbox_exec_problem() -> str | None:
    if sys.platform != "darwin":
        return "sandbox-exec is only available on Darwin"
    executable = _sandbox_executable("sandbox-exec")
    if executable is None:
        return "sandbox-exec is unavailable in PATH and standard install locations"
    return _command_problem([executable, "-p", "(version 1) (allow default)", "/bin/sh", "-c", "exit 0"])


def _sandbox_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable
    for directory in _SANDBOX_DIRS:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _command_problem(command: list[str], *, timeout: float = scaled(8.0)) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace", timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"could not run {Path(command[0]).name}: {error}"
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout).strip().splitlines()
    suffix = f": {detail[0][:180]}" if detail else ""
    return f"{Path(command[0]).name} exited {result.returncode}{suffix}"


def _host_sandbox_problem() -> str | None:
    if sys.platform.startswith("linux"):
        return _bwrap_problem()
    if sys.platform == "darwin":
        return _sandbox_exec_problem()
    return f"host sandbox is unsupported on {sys.platform}"


def _pid_namespace_problem() -> str | None:
    if not sys.platform.startswith("linux"):
        return "PID namespaces are only available on Linux"

    executable = _sandbox_executable("bwrap")
    if executable is not None:
        tool = "bwrap"
        command_prefix = [
            executable, "--die-with-parent", "--unshare-user", "--uid", "0", "--gid", "0",
            "--unshare-pid",
        ]
    else:
        executable = _sandbox_executable("unshare")
        if executable is None:
            return "neither bwrap nor unshare is available for the PID namespace probe"
        tool = "unshare"
        command_prefix = [
            executable, "--user", "--map-root-user", "--pid", "--fork", "--mount-proc", "--",
        ]

    try:
        with tempfile.TemporaryDirectory(prefix="sf-must-run-pidns-") as directory:
            if tool == "bwrap":
                command_prefix.extend([
                    "--ro-bind", "/", "/", "--bind", directory, directory,
                    "--proc", "/proc", "--dev", "/dev", "--chdir", directory, "--",
                ])
            command = [*command_prefix, sys.executable, "-c", "import os; print(os.getpid())"]
            probe_timeout = scaled(3.0)
            result = subprocess.run(
                command, cwd=directory, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, errors="replace", timeout=probe_timeout, check=False,
            )
    except subprocess.TimeoutExpired:
        return f"{tool} PID namespace probe timed out after {probe_timeout:g} seconds"
    except OSError as error:
        return f"could not run {tool} PID namespace probe: {error}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[0][:180]}" if detail else ""
        return f"{tool} exited {result.returncode}{suffix}"
    child_pid = result.stdout.strip()
    if child_pid != "1":
        return f"{tool} did not create a PID namespace (child pid was {child_pid or 'empty'})"
    return None


def _filesystem_modes_problem() -> str | None:
    try:
        with tempfile.TemporaryDirectory(prefix="sf-must-run-mode-") as directory:
            protected = Path(directory) / "protected"
            protected.mkdir()
            protected.chmod(0o500)
            try:
                descriptor = os.open(protected / "probe", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except PermissionError:
                return None
            else:
                os.close(descriptor)
                return "the current user can write through restrictive directory mode bits"
            finally:
                protected.chmod(0o700)
    except OSError as error:
        return f"could not verify directory mode enforcement: {error}"


def _baseline_problem(root: Path, manifest: GateManifest) -> str | None:
    configured = os.environ.get("SF_TEST_BASELINE_IMMEDIATE", "").strip()
    if not configured:
        return "SF_TEST_BASELINE_IMMEDIATE is not set"
    try:
        baseline_root = Path(configured).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        return f"SF_TEST_BASELINE_IMMEDIATE cannot be resolved: {error}"
    if baseline_root == root.resolve():
        return "parity baseline resolves to the candidate directory"
    expected = manifest.baselines.get("immediate")
    if expected != {"source": "SF_GATE_BASE"}:
        return "manifest does not define the immediate baseline from SF_GATE_BASE"
    base = os.environ.get("SF_GATE_BASE", "").strip()
    if _OBJECT_ID_PATTERN.fullmatch(base) is None:
        return "SF_GATE_BASE must be a full 40-character commit id for parity baseline"
    try:
        resolved_base = _git_text(root, "rev-parse", "--verify", f"{base}^{{commit}}")
        candidate_parent = _git_text(root, "rev-parse", "--verify", "HEAD^1^{commit}")
    except GateInputError as error:
        return f"SF_GATE_BASE cannot be resolved: {error}"
    if resolved_base != base:
        return "SF_GATE_BASE does not resolve to the supplied full commit id"
    if candidate_parent != base:
        return f"SF_GATE_BASE must equal candidate HEAD^: expected {candidate_parent}, found {base}"
    try:
        expected_tree = _git_text(root, "rev-parse", "--verify", f"{base}^{{tree}}")
        commit = _git_text(baseline_root, "rev-parse", "--verify", "HEAD^{commit}")
        tree = _git_text(baseline_root, "rev-parse", "--verify", "HEAD^{tree}")
        status = _git_text(baseline_root, "status", "--porcelain=v1", "--untracked-files=all")
    except GateInputError as error:
        return f"baseline provenance cannot be verified: {error}"
    if commit != base:
        return f"immediate baseline HEAD does not match SF_GATE_BASE: expected {base}, found {commit}"
    if tree != expected_tree:
        return "immediate baseline tree does not match SF_GATE_BASE"
    if status:
        return "parity baseline worktree is dirty"
    if not (baseline_root / "src" / "stateframe" / "runner.py").is_file():
        return "immediate baseline has no src/stateframe/runner.py"

    release = manifest.baselines.get("release_0_2_3")
    if release is not None:
        try:
            found_commit = _git_text(root, "rev-parse", "--verify", f"{release['commit']}^{{commit}}")
            found_tree = _git_text(root, "rev-parse", "--verify", f"{release['commit']}^{{tree}}")
        except GateInputError as error:
            return f"pinned 0.2.3 baseline is unavailable: {error}"
        if found_commit != release["commit"] or found_tree != release["tree_digest"]:
            return "pinned 0.2.3 baseline provenance differs from the reviewed manifest"
    return _host_sandbox_problem()


def _serial_lane_problem(config: pytest.Config | None) -> str | None:
    if config is not None and _xdist_active(config):
        return "serial-lane nodes require a run without xdist"
    return None


def _xdist_active(config: pytest.Config) -> bool:
    if hasattr(config, "workerinput"):
        return True
    count = getattr(getattr(config, "option", None), "numprocesses", None)
    return count not in (None, 0, "0")


def _required_shards(manifest: GateManifest) -> tuple[str, ...]:
    if any("serial_lane" in cell.prerequisites for cell in manifest.cells):
        return ("xdist", "serial_lane")
    return ("full",)


def _unknown_uid_bwrap_problem() -> str | None:
    problem = _bwrap_problem()
    if problem is not None:
        return problem
    if os.getuid() == 0:
        return "the unknown-uid case cannot run as root"
    try:
        pwd.getpwuid(os.getuid())
    except KeyError:
        return None
    except OSError as error:
        return f"could not inspect the current uid's passwd entry: {error}"
    return "the current non-root uid has a passwd entry"


def _uv_sandbox_problem() -> str | None:
    problem = _sandbox_exec_problem()
    if problem is not None:
        return problem
    if shutil.which("uv") is None:
        return "uv is not on PATH"
    try:
        store = Path.home() / ".local" / "share" / "uv" / "python"
        if not store.is_dir():
            return f"uv-managed Python store is unavailable: {store}"
    except (OSError, RuntimeError) as error:
        return f"could not inspect the uv-managed Python store: {error}"
    return None


def _provision_problem(name: str, root: Path, manifest: GateManifest,
                       config: pytest.Config | None) -> str | None:
    if name == "bwrap":
        return _bwrap_problem()
    if name == "sandbox_exec":
        return _sandbox_exec_problem()
    if name == "host_sandbox":
        return _host_sandbox_problem()
    if name == "pid_namespace":
        return _pid_namespace_problem()
    if name == "parity_baseline":
        return _baseline_problem(root, manifest)
    if name == "filesystem_modes":
        return _filesystem_modes_problem()
    if name == "serial_lane":
        return _serial_lane_problem(config)
    if name == "unknown_uid_bwrap":
        return _unknown_uid_bwrap_problem()
    if name == "uv_sandbox":
        return _uv_sandbox_problem()
    return f"unrecognized provision: {name}"


def _read_ticket(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateInputError(f"gate ticket cannot be read: {path}: {error}") from error
    fields = {"schema", "job_id", "run_nonce", "candidate_digest", "manifest_digest", "issued_at_ns"}
    if (not isinstance(value, dict) or set(value) != fields
            or type(value.get("schema")) is not int or value["schema"] != 1):
        raise GateInputError(f"gate ticket has invalid fields: {path}")
    if (not isinstance(value["job_id"], str) or _JOB_PATTERN.fullmatch(value["job_id"]) is None
            or not isinstance(value["run_nonce"], str) or _NONCE_PATTERN.fullmatch(value["run_nonce"]) is None
            or not isinstance(value["candidate_digest"], str)
            or _SHA256_PATTERN.fullmatch(value["candidate_digest"]) is None
            or not isinstance(value["manifest_digest"], str)
            or _SHA256_PATTERN.fullmatch(value["manifest_digest"]) is None
            or type(value["issued_at_ns"]) is not int):
        raise GateInputError(f"gate ticket contains invalid identity values: {path}")
    return value


def _changed_paths(root: Path) -> frozenset[str]:
    base = os.environ.get("SF_GATE_BASE", "").strip()
    if _OBJECT_ID_PATTERN.fullmatch(base) is None:
        raise GateInputError("SF_GATE_BASE must be a full 40-character commit id")
    resolved = _git_text(root, "rev-parse", "--verify", f"{base}^{{commit}}")
    if resolved != base:
        raise GateInputError("SF_GATE_BASE does not resolve to the supplied full commit id")
    candidate_parent = _git_text(root, "rev-parse", "--verify", "HEAD^1^{commit}")
    if candidate_parent != base:
        raise GateInputError(f"SF_GATE_BASE must equal candidate HEAD^: expected {candidate_parent}, found {base}")
    output = _git_text(root, "diff", "--name-only", "--no-renames", f"{base}...HEAD", "--")
    return frozenset(line for line in output.splitlines() if line)


def _read_provisions() -> frozenset[str]:
    parts = [part.strip() for part in os.environ.get("SF_GATE_PROVISIONS", "").split(",") if part.strip()]
    if len(parts) != len(set(parts)):
        raise GateInputError("SF_GATE_PROVISIONS contains duplicate entries")
    unknown = set(parts) - _PROVISIONS
    if unknown:
        raise GateInputError(f"SF_GATE_PROVISIONS contains unknown entries: {sorted(unknown)}")
    return frozenset(parts)


def _record_violation(state: GateState, key: str, reason: str) -> None:
    state.violations.setdefault(key, []).append(reason)


def _setup_state(config: pytest.Config, root: Path, manifest: GateManifest,
                 manifest_digest: str, must_run: bool) -> GateState:
    state = GateState(manifest=manifest, manifest_digest=manifest_digest, must_run=must_run)
    state.is_worker = hasattr(config, "workerinput")
    if not must_run:
        return state
    state.shard_id = os.environ.get("SF_GATE_SHARD", "full").strip() or "full"
    if state.shard_id not in {"full", "xdist", "serial_lane"}:
        _record_violation(state, "gate:shard", f"unrecognized SF_GATE_SHARD: {state.shard_id}")
    if state.shard_id == "xdist" and not _xdist_active(config):
        _record_violation(state, "gate:shard", "SF_GATE_SHARD=xdist requires an xdist run")
    elif state.shard_id == "serial_lane" and _xdist_active(config):
        _record_violation(state, "gate:shard", "SF_GATE_SHARD=serial_lane requires a serial run")
    elif state.shard_id == "full" and _xdist_active(config):
        _record_violation(state, "gate:shard", "xdist must-run runs require SF_GATE_SHARD=xdist")
    state.started_at_ns = time.time_ns()
    try:
        state.job_id = os.environ.get("SF_GATE_JOB_ID", "").strip() or None
        if state.job_id is None:
            _record_violation(state, "job:missing", "SF_GATE_JOB_ID is required")
        elif state.job_id not in manifest.jobs:
            _record_violation(state, f"job:{state.job_id}", "job is not declared in the reviewed manifest")
        else:
            state.job = manifest.jobs[state.job_id]
            if state.job.platform != _platform_name():
                _record_violation(
                    state, f"job:{state.job_id}",
                    f"job requires {state.job.platform}, current host is {_platform_name()}",
                )
        state.provisions = _read_provisions()
        state.candidate_digest = _candidate_digest(root)
        state.changed_paths = _changed_paths(root)
        if config.getini("enable_assertion_pass_hook") is not True:
            _record_violation(
                state, "gate:witness", "enable_assertion_pass_hook must be true for substantive witnesses",
            )
        ticket_value = os.environ.get("SF_GATE_TICKET", "").strip()
        evidence_value = os.environ.get("SF_GATE_EVIDENCE", "").strip()
        if not ticket_value:
            _record_violation(state, "gate:ticket", "SF_GATE_TICKET is required")
        if not evidence_value:
            _record_violation(state, "gate:evidence", "SF_GATE_EVIDENCE is required")
        if ticket_value and evidence_value and state.job is not None:
            ticket_path = Path(ticket_value).expanduser().resolve()
            evidence_path = Path(evidence_value).expanduser()
            suffix = ".json" if state.shard_id == "full" else f".{state.shard_id}.json"
            expected_name = f"{state.job.job_id}{suffix}"
            if evidence_path.name != expected_name:
                _record_violation(
                    state, "gate:evidence",
                    f"evidence filename must be {expected_name} for this job and shard",
                )
            try:
                ticket = _read_ticket(ticket_path)
            except GateInputError as error:
                _record_violation(state, "gate:ticket", str(error))
            else:
                state.ticket = ticket
                if ticket["job_id"] != state.job.job_id:
                    _record_violation(state, "gate:ticket", "ticket job id does not match SF_GATE_JOB_ID")
                if ticket["manifest_digest"] != manifest_digest:
                    _record_violation(state, "gate:ticket", "ticket manifest digest is stale")
                if ticket["candidate_digest"] != state.candidate_digest:
                    _record_violation(state, "gate:ticket", "ticket candidate digest is stale")
                now = time.time_ns()
                issued = ticket["issued_at_ns"]
                if issued > now or now - issued > _MAX_TICKET_AGE_NS:
                    _record_violation(state, "gate:ticket", "ticket nonce is stale")
                if evidence_path.exists() or evidence_path.is_symlink():
                    _record_violation(state, "gate:evidence", "refusing pre-existing evidence; it may be stale")
                elif not evidence_path.parent.is_dir():
                    _record_violation(state, "gate:evidence", "evidence parent directory does not exist")
                else:
                    state.evidence_path = evidence_path

        state.provision_problems = {
            name: _provision_problem(name, root, manifest, config)
            for name in _PROVISIONS
        }
        if state.job is not None:
            for name in state.job.required_provisions:
                if name not in state.provisions:
                    _record_violation(state, f"provision:{name}", "required provision was not claimed")
                problem = state.provision_problems[name]
                if problem is not None:
                    _record_violation(state, f"provision:{name}", problem)
            for name in state.job.conditional_provisions:
                problem = state.provision_problems[name]
                claimed = name in state.provisions
                if problem is None and not claimed:
                    _record_violation(state, f"provision:{name}", "available provision was omitted")
                elif problem is not None and claimed:
                    _record_violation(state, f"provision:{name}", f"claimed provision is unavailable: {problem}")
            for name in state.provisions:
                if (name not in state.job.required_provisions
                        and name not in state.job.conditional_provisions):
                    _record_violation(state, f"provision:{name}", "provision is not assigned to this job")
        if os.environ.get("SF_TEST_PARITY_BASELINE"):
            _record_violation(state, "gate:baseline", "SF_TEST_PARITY_BASELINE is retired")
    except GateInputError as error:
        _record_violation(state, "gate:configuration", str(error))
    if state.job is not None:
        for cell in manifest.cells:
            disposition = cell.jobs[state.job.job_id]
            if disposition["disposition"] == "RUN":
                for prerequisite in cell.prerequisites:
                    problem = state.provision_problems.get(prerequisite, "provision was not checked")
                    if problem is not None and prerequisite not in cell.optional_prerequisites:
                        _record_violation(state, f"cell:{cell.scenario_id}",
                                          f"required prerequisite {prerequisite} unavailable: {problem}")
            state.cells_by_nodeid.setdefault(cell.nodeid, []).append(cell)
            for item_nodeid in cell.item_nodeids:
                state.cells_by_item_nodeid.setdefault(item_nodeid, []).append(cell)
    return state


def _must_run_enabled() -> bool:
    value = os.environ.get("SF_GATE_MUST_RUN", "0").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"", "0", "false", "no", "off"}:
        return False
    raise pytest.UsageError("SF_GATE_MUST_RUN must be 0 or 1")


def pytest_configure(config: pytest.Config) -> None:
    global _ACTIVE_STATE
    root = Path(str(config.rootpath)).resolve()
    try:
        must_run = _must_run_enabled()
    except pytest.UsageError:
        raise
    if must_run:
        try:
            manifest, digest = _read_manifest(root)
        except GateInputError as error:
            raise pytest.UsageError(str(error)) from error
    else:
        manifest = GateManifest({}, {}, (), ())
        digest = ""
    state = _setup_state(config, root, manifest, digest, must_run)
    config.stash[_STATE_KEY] = state
    _ACTIVE_STATE = state


def pytest_itemcollected(item: pytest.Item) -> None:
    state = item.config.stash[_STATE_KEY]
    if not state.must_run:
        return
    for cell in state.cells_by_item_nodeid.get(item.nodeid, ()):
        state.collected.setdefault(cell.scenario_id, []).append(item)


def pytest_collection_finish(session: pytest.Session) -> None:
    state = session.config.stash[_STATE_KEY]
    if state.must_run:
        state.selected_nodeids = {item.nodeid for item in session.items}


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.passed:
        return
    state = _ACTIVE_STATE
    if state is None:
        return
    if not state.must_run or state.job is None:
        return
    for cell in state.manifest.cells:
        if cell.jobs[state.job.job_id]["disposition"] != "RUN":
            continue
        if any(item_nodeid.partition("::")[0] == report.nodeid for item_nodeid in cell.item_nodeids):
            _record_violation(state, f"cell:{cell.scenario_id}",
                              f"collection error for {cell.nodeid}: {report.longrepr}")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item: pytest.Item):
    state = item.config.stash[_STATE_KEY]
    if state.must_run:
        state.active_call_ids.add(item.nodeid)
    try:
        yield
    finally:
        if state.must_run:
            state.active_call_ids.discard(item.nodeid)


def pytest_assertion_pass(item: pytest.Item, lineno: int, orig: str, expl: str) -> None:
    del lineno, expl
    state = item.config.stash[_STATE_KEY]
    if state.must_run and item.nodeid in state.active_call_ids:
        state.assertion_passes[item.nodeid] = state.assertion_passes.get(item.nodeid, 0) + 1
        try:
            state.assertion_sources.setdefault(item.nodeid, set()).add(_pytest_assertion_key(orig))
        except GateInputError as error:
            _record_violation(state, f"witness:{item.nodeid}", str(error))


_REPORT_WITNESS_PROPERTY = "stateframe_must_run_witness"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    outcome = yield
    report = outcome.get_result()
    state = item.config.stash[_STATE_KEY]
    if not state.must_run or report.when != "call":
        return
    cells = state.cells_by_item_nodeid.get(report.nodeid, ())
    if not cells:
        return
    if len(cells) != 1:
        _record_violation(state, f"witness:{report.nodeid}", "item maps to multiple reviewed scenarios")
        return
    cell = cells[0]
    witness: dict[str, object] = {
        "schema": 1,
        "assertion_passes": state.assertion_passes.get(report.nodeid, 0),
        "assertion_witness": (
            cell.witness_schema == "assertion_match"
            and _assertion_key(cell.witness_assertion or "") in state.assertion_sources.get(report.nodeid, set())
        ),
    }
    if cell.witness_schema == "parity_trace":
        parity = state.parity_witnesses.get(report.nodeid)
        if parity is not None:
            witness["parity_witness"] = parity
    report.user_properties = [
        (name, value) for name, value in report.user_properties if name != _REPORT_WITNESS_PROPERTY
    ]
    report.user_properties.append((_REPORT_WITNESS_PROPERTY, json.dumps(witness, sort_keys=True)))


def _load_report_witness(report: pytest.TestReport) -> dict[str, object] | None:
    values = [value for name, value in report.user_properties if name == _REPORT_WITNESS_PROPERTY]
    if len(values) != 1 or not isinstance(values[0], str):
        return None
    try:
        value = json.loads(values[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) - {"schema", "assertion_passes", "assertion_witness", "parity_witness"}:
        return None
    if (type(value.get("schema")) is not int or value["schema"] != 1
            or type(value.get("assertion_passes")) is not int or value["assertion_passes"] < 0
            or type(value.get("assertion_witness")) is not bool):
        return None
    return value


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    state = _ACTIVE_STATE
    if state is None or not state.must_run or state.is_worker:
        return
    state.selected_nodeids.add(report.nodeid)
    state.reports.setdefault(report.nodeid, {}).setdefault(report.when, []).append(report)
    cells = state.cells_by_item_nodeid.get(report.nodeid, ())
    if report.when == "call" and cells:
        witness = _load_report_witness(report)
        if witness is None:
            _record_violation(state, f"witness:{report.nodeid}", "call report has no valid worker witness record")
        else:
            state.assertion_passes[report.nodeid] = witness["assertion_passes"]
            state.assertion_witnesses[report.nodeid] = witness["assertion_witness"]
            parity = witness.get("parity_witness")
            if isinstance(parity, dict):
                state.parity_witnesses[report.nodeid] = parity

    has_platform_disposition = (
        state.job is not None
        and any(cell.jobs[state.job.job_id]["disposition"] == "NA_PLATFORM" for cell in cells)
    )
    optional_skip_valid = (
        _record_optional_skip(state, report)
        if report.outcome == "skipped" and not has_platform_disposition else False
    )
    for cell in cells:
        if state.job is None:
            continue
        if state.shard_id == "xdist" and "serial_lane" in cell.prerequisites:
            continue
        disposition = cell.jobs[state.job.job_id]
        if disposition["disposition"] == "NA_PLATFORM":
            if report.when == "call" and report.outcome != "skipped":
                _record_violation(state, f"cell:{cell.scenario_id}",
                                  "platform-not-applicable test call did not skip")
            continue
        effective_status, _ = _cell_job_status(cell, state.job, state)
        if effective_status == "prerequisite_na":
            if report.outcome == "skipped" and not optional_skip_valid:
                _record_violation(state, f"cell:{cell.scenario_id}",
                                  "optional prerequisite skip has no reviewed dependency mapping")
            continue
        if report.outcome == "skipped":
            _record_violation(state, f"cell:{cell.scenario_id}",
                              f"required node skipped during {report.when}: {report.nodeid}")
        if getattr(report, "wasxfail", None) is not None:
            label = "XPASS" if report.outcome == "passed" else "XFAIL"
            _record_violation(state, f"cell:{cell.scenario_id}",
                              f"required node reported {label} during {report.when}: {report.nodeid}")


def _optional_node_matches(pattern: str, nodeid: str) -> bool:
    if pattern.endswith("::*"):
        return nodeid.startswith(pattern[:-1])
    return nodeid == pattern or nodeid.startswith(pattern + "[")


def _tool_problem(tool: str) -> str | None:
    names = tuple(part.strip() for part in tool.split(",") if part.strip())
    if not names:
        return "no dependency command was declared"
    problems = [problem for name in names if (problem := _single_tool_problem(name)) is not None]
    return "; ".join(problems) if problems else None


def _single_tool_problem(tool: str) -> str | None:
    if tool.startswith("python:"):
        module_name = tool.removeprefix("python:")
        if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module_name) is None:
            raise GateInputError(f"invalid Python module probe: {module_name!r}")
        try:
            module_spec = importlib.util.find_spec(module_name)
        except (ImportError, ModuleNotFoundError, OSError, ValueError) as error:
            raise GateInputError(f"could not probe Python module {module_name}: {error}") from error
        return None if module_spec is not None else f"Python module {module_name} is not installed"
    if tool == "bwrap":
        return _bwrap_problem()
    if tool == "sandbox-exec":
        return _sandbox_exec_problem()
    if tool == "uv_sandbox":
        return _uv_sandbox_problem()
    if tool == "filesystem_modes":
        return _filesystem_modes_problem()
    if tool == "unknown_uid_bwrap":
        return _unknown_uid_bwrap_problem()
    if tool == "platform_sandbox":
        return _host_sandbox_problem()
    if os.path.isabs(tool):
        return None if Path(tool).is_file() and os.access(tool, os.X_OK) else f"{tool} is absent"
    return None if shutil.which(tool) else f"{tool} is not on PATH"


def _module_touched(module: str, changed_paths: frozenset[str]) -> bool:
    normalized = module.rstrip("/")
    return any(path == normalized or path.startswith(normalized + "/") for path in changed_paths)


def _looks_like_optional_dependency_skip(reason: str) -> bool:
    lowered = reason.casefold()
    return any(
        hint in lowered
        for hint in (
            "not installed",
            "not available",
            "not found",
            "unavailable",
            "is absent",
            "are absent",
            "absent",
            "missing:",
            "required for",
            "extra (",
            "not on path",
            "no module named",
        )
    )


def _record_optional_skip(state: GateState, report: pytest.TestReport) -> bool:
    reason = str(report.longrepr)
    entries = [
        entry for entry in state.manifest.optional_dependencies
        if any(_optional_node_matches(pattern, report.nodeid) for pattern in entry.tests)
    ]
    if not entries:
        if _looks_like_optional_dependency_skip(reason):
            _record_violation(
                state,
                f"optional:unmapped:{report.nodeid}",
                "optional dependency skip has no reviewed module/extra/test mapping",
            )
        return False
    valid_entries: list[OptionalDependency] = []
    for entry in entries:
        if not any(value in reason for value in entry.absence_reasons):
            continue
        try:
            tool_problem = _tool_problem(entry.tool)
        except GateInputError as error:
            _record_violation(state, f"optional:{entry.extra}:{report.nodeid}",
                              f"dependency availability could not be verified: {error}")
            continue
        if tool_problem is None:
            key = f"{entry.extra}:{report.nodeid}"
            _record_violation(state, f"optional:{key}",
                              "skip reason names a mapped dependency that is available")
            continue
        valid_entries.append(entry)
    if not valid_entries:
        _record_violation(state, f"optional:{report.nodeid}",
                          "skip was not caused by a reviewed missing dependency")
        return False
    for entry in valid_entries:
        key = f"{entry.extra}:{report.nodeid}"
        state.optional_skips.setdefault(key, {
            "nodeid": report.nodeid,
            "module": entry.module,
            "extra": entry.extra,
            "tool": entry.tool,
            "reason": reason,
            "validated_absence": True,
        })
        if _module_touched(entry.module, state.changed_paths):
            _record_violation(state, f"optional:{key}",
                              f"dependency not provisioned for a touched module: {entry.module}")
    return True


class RunWitness:
    """Scenario-specific evidence in addition to a passing test call."""

    def __init__(self, item: pytest.Item, state: GateState) -> None:
        self._item = item
        self._state = state

    def record_parity(
        self,
        *,
        baseline_trace: object,
        candidate_trace: object,
        expected_trace_prefixes: tuple[str, ...],
        comparison_completed: bool,
    ) -> None:
        if not self._state.must_run:
            return
        cells = self._state.cells_by_nodeid.get(self._item.nodeid, ())
        if not any(cell.witness_schema == "parity_trace" for cell in cells):
            raise AssertionError(f"{self._item.nodeid} is not a parity scenario")
        baseline = _trace_modules(baseline_trace, "baseline")
        candidate = _trace_modules(candidate_trace, "candidate")
        if not expected_trace_prefixes or any(not prefix.strip() for prefix in expected_trace_prefixes):
            raise AssertionError("parity witness needs non-empty expected trace coverage")
        for label, modules in (("baseline", baseline), ("candidate", candidate)):
            if not any(module == prefix or module.startswith(prefix + ".")
                       for module in modules for prefix in expected_trace_prefixes):
                raise AssertionError(f"{label} trace has no expected module coverage")
        if comparison_completed is not True:
            raise AssertionError("parity comparison did not complete")
        self._state.parity_witnesses[self._item.nodeid] = {
            "baseline_count": len(baseline),
            "candidate_count": len(candidate),
            "baseline_digest": hashlib.sha256("\n".join(sorted(baseline)).encode()).hexdigest(),
            "candidate_digest": hashlib.sha256("\n".join(sorted(candidate)).encode()).hexdigest(),
            "expected_trace_prefixes": list(expected_trace_prefixes),
            "comparison_completed": True,
        }


def _trace_modules(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        raise AssertionError(f"{label} execution trace is not a module sequence")
    modules = tuple(sorted({item for item in value if isinstance(item, str) and item}))
    if not modules:
        raise AssertionError(f"{label} execution trace is empty")
    return modules


@pytest.fixture
def must_run_witness(request: pytest.FixtureRequest) -> RunWitness:
    return RunWitness(request.node, request.config.stash[_STATE_KEY])


def _cell_job_status(cell: GateCell, job: GateJob, state: GateState) -> tuple[str, str | None]:
    disposition = cell.jobs[job.job_id]
    kind = disposition["disposition"]
    if kind == "NA_PLATFORM":
        target_platform = disposition["platform"]
        if job.platform == target_platform:
            return "violated", "NA_PLATFORM contradicts the job platform"
        return "platform_na", disposition["reason"]
    missing = [name for name in cell.optional_prerequisites
               if state.provision_problems.get(name) is not None]
    if missing:
        reasons = [state.provision_problems[name] or "unavailable" for name in missing]
        return "prerequisite_na", "; ".join(reasons)
    return "RUN", None


def _skip_reports_problem(reports: object) -> str | None:
    if not isinstance(reports, dict) or not reports:
        return "skip evidence has no phase reports"
    if set(reports) - {"setup", "call", "teardown"}:
        return "skip evidence contains an unknown test phase"
    skipped = 0
    for phase, outcomes in reports.items():
        if not isinstance(outcomes, list) or len(outcomes) != 1:
            return f"skip evidence needs one {phase} report"
        if not isinstance(outcomes[0], str) or outcomes[0] not in {"passed", "skipped"}:
            return f"skip evidence has a failing {phase} report"
        skipped += outcomes[0] == "skipped"
    if skipped != 1:
        return "skip evidence needs exactly one skipped phase"
    return None


def _collected_item_nodeids(state: GateState, cell: GateCell) -> list[str]:
    if state.shard_id == "xdist" and not state.is_worker:
        # The controller receives worker reports, not worker Item instances. Missing report phases
        # below distinguish a collected/executed item from one that was ignored or deselected.
        return list(cell.item_nodeids)
    found = state.collected.get(cell.scenario_id, [])
    by_nodeid: dict[str, list[pytest.Item]] = {}
    for item in found:
        by_nodeid.setdefault(item.nodeid, []).append(item)
    expected = set(cell.item_nodeids)
    if set(by_nodeid) != expected:
        _record_violation(
            state,
            f"cell:{cell.scenario_id}",
            f"item inventory mismatch: missing={sorted(expected - set(by_nodeid))} "
            f"unexpected={sorted(set(by_nodeid) - expected)}",
        )
    for nodeid, matching in by_nodeid.items():
        if len(matching) != 1:
            _record_violation(state, f"cell:{cell.scenario_id}", f"duplicate collected item {nodeid}")
    return [nodeid for nodeid in cell.item_nodeids if by_nodeid.get(nodeid)]


def _item_report_outcomes(state: GateState, nodeid: str) -> dict[str, list[str]]:
    return {
        phase: [report.outcome for report in phase_reports]
        for phase, phase_reports in state.reports.get(nodeid, {}).items()
    }


def _validate_run_cell(state: GateState, cell: GateCell, job: GateJob) -> dict[str, object]:
    status, disposition_reason = _cell_job_status(cell, job, state)
    evidence: dict[str, object] = {
        "scenario_id": cell.scenario_id,
        "nodeid": cell.nodeid,
        "item_nodeids": list(cell.item_nodeids),
        "status": status,
        "baseline": cell.baseline,
        "witness_schema": cell.witness_schema,
    }
    item_nodeids = _collected_item_nodeids(state, cell)
    if item_nodeids:
        state.scanned_cell_ids.add(cell.scenario_id)
    for nodeid in cell.item_nodeids:
        if nodeid not in state.selected_nodeids:
            _record_violation(
                state, f"cell:{cell.scenario_id}",
                f"required item was deselected or produced no run report: {nodeid}",
            )
    item_evidence: list[dict[str, object]] = []

    if status in {"platform_na", "prerequisite_na"}:
        evidence["reason"] = disposition_reason
        if status == "prerequisite_na":
            evidence["missing_prerequisites"] = [
                name for name in cell.optional_prerequisites
                if state.provision_problems.get(name) is not None
            ]
        if len(item_nodeids) != len(cell.item_nodeids):
            evidence["status"] = "missing"
            return evidence
        for nodeid in item_nodeids:
            outcomes = _item_report_outcomes(state, nodeid)
            report_problem = _skip_reports_problem(outcomes)
            skipped = [
                report for phase_reports in state.reports.get(nodeid, {}).values()
                for report in phase_reports if report.outcome == "skipped"
            ]
            if report_problem is not None:
                _record_violation(
                    state, f"cell:{cell.scenario_id}",
                    f"{nodeid} lacks clean skip evidence: {report_problem}",
                )
                evidence["status"] = "unqualified"
                continue
            if status == "platform_na" and not any(
                disposition_reason and disposition_reason in str(report.longrepr) for report in skipped
            ):
                _record_violation(
                    state, f"cell:{cell.scenario_id}",
                    f"{nodeid} skip reason differs from the reviewed manifest",
                )
                evidence["status"] = "unqualified"
            if status == "prerequisite_na" and nodeid not in {
                str(skip.get("nodeid")) for skip in state.optional_skips.values()
            }:
                _record_violation(
                    state, f"cell:{cell.scenario_id}",
                    f"{nodeid} has no reviewed optional dependency skip",
                )
                evidence["status"] = "unqualified"
            item_evidence.append({
                "nodeid": nodeid,
                "reports": outcomes,
                "skip_reasons": [str(report.longrepr) for report in skipped],
            })
        evidence["items"] = item_evidence
        return evidence

    if len(item_nodeids) != len(cell.item_nodeids):
        evidence["status"] = "missing"
        return evidence
    all_clean = True
    for nodeid in item_nodeids:
        phase_results: dict[str, str] = {}
        for phase in ("setup", "call", "teardown"):
            reports = state.reports.get(nodeid, {}).get(phase, [])
            if len(reports) != 1:
                _record_violation(
                    state, f"cell:{cell.scenario_id}",
                    f"{nodeid}: expected one {phase} report, found {len(reports)}",
                )
                phase_results[phase] = "missing" if not reports else "duplicate"
                all_clean = False
                continue
            report = reports[0]
            phase_results[phase] = report.outcome
            if report.outcome != "passed" or getattr(report, "wasxfail", None) is not None:
                _record_violation(
                    state, f"cell:{cell.scenario_id}",
                    f"{nodeid}: {phase} report did not pass cleanly: {report.outcome}",
                )
                all_clean = False
        passes = state.assertion_passes.get(nodeid, 0)
        assertion_witness = state.assertion_witnesses.get(nodeid, False)
        if cell.witness_schema == "assertion_match":
            if passes < 1:
                _record_violation(state, f"cell:{cell.scenario_id}", f"{nodeid}: no passing assertion")
                all_clean = False
            if not assertion_witness:
                _record_violation(
                    state, f"cell:{cell.scenario_id}",
                    f"{nodeid}: scenario-specific assertion witness is missing",
                )
                all_clean = False
        item_evidence.append({
            "nodeid": nodeid,
            "reports": phase_results,
            "assertion_passes": passes,
            "assertion_witness": assertion_witness,
        })

    if cell.witness_schema == "parity_trace":
        witness = state.parity_witnesses.get(cell.item_nodeids[0])
        if witness is None:
            _record_violation(state, f"cell:{cell.scenario_id}", "baseline/candidate trace witness is missing")
            all_clean = False
        else:
            evidence["parity_witness"] = witness
    evidence["items"] = item_evidence
    evidence["status"] = "qualified" if all_clean else "unqualified"
    return evidence


def _write_run_evidence(state: GateState) -> None:
    if state.is_worker or state.evidence_path is None or state.job is None or state.ticket is None:
        return
    shard_cells = _cells_for_shard(state.manifest, state.shard_id)
    payload = {
        "schema": 2,
        "job_id": state.job.job_id,
        "shard_id": state.shard_id,
        "run_nonce": state.ticket["run_nonce"],
        "candidate_digest": state.candidate_digest,
        "manifest_digest": state.manifest_digest,
        "started_at_ns": state.started_at_ns,
        "finished_at_ns": time.time_ns(),
        "expected_cell_ids": [cell.scenario_id for cell in shard_cells],
        "scanned_cell_ids": sorted(state.scanned_cell_ids),
        "deferred_cell_ids": sorted(state.deferred_cell_ids),
        "cells": [
            item for item in state.cell_evidence
            if item.get("scenario_id") in {cell.scenario_id for cell in shard_cells}
        ],
        "deferred_cells": state.deferred_evidence,
        "optional_skips": sorted(state.optional_skips.values(), key=lambda item: str(item["nodeid"])),
        "violations": {key: list(reasons) for key, reasons in sorted(state.violations.items())},
    }
    try:
        _write_exclusive(state.evidence_path, _json_bytes(payload))
    except GateInputError as error:
        _record_violation(state, "gate:evidence", str(error))


def _run_summary(state: GateState) -> str:
    expected = len(_cells_for_shard(state.manifest, state.shard_id))
    if state.shard_id == "xdist":
        expected += sum("serial_lane" in cell.prerequisites for cell in state.manifest.cells)
    scanned = len(state.scanned_cell_ids) + len(state.deferred_cell_ids)
    required = sum(
        1 for cell in state.manifest.cells
        if state.job is not None and cell.jobs[state.job.job_id]["disposition"] == "RUN"
    )
    collected = sum(1 for cell in state.manifest.cells if set(cell.item_nodeids) <= state.selected_nodeids)
    qualified = sum(1 for item in state.cell_evidence if item["status"] == "qualified")
    platform_na = sum(1 for item in state.cell_evidence if item["status"] == "platform_na")
    deferred = len(state.deferred_cell_ids)
    # Count reviewed dependency/test absence records separately from skipped test nodes.
    # A touched-module violation remains separately visible in optional_touched_violations.
    optional_na = len(state.optional_skips)
    optional_skipped = len({item["nodeid"] for item in state.optional_skips.values()})
    optional_touched = sum(
        1 for key, reasons in state.violations.items()
        if key.startswith("optional:") and any("touched module" in reason for reason in reasons)
    )
    violated = len(state.violations)
    not_applicable = sum(
        1 for item in state.cell_evidence if item["status"] in {"platform_na", "prerequisite_na"}
    )
    applicable = expected - not_applicable
    return (
        f"GUARD must_run: expected={expected} applicable={applicable} scanned={scanned} "
        f"violated={violated} not_applicable={not_applicable} required={required} "
        f"shard={state.shard_id} deferred={deferred} "
        f"collected={collected} qualified={qualified} platform_na={platform_na} optional_na={optional_na} "
        f"optional_skipped={optional_skipped} optional_touched_violations={optional_touched}"
    )


def _cells_for_shard(manifest: GateManifest, shard_id: str) -> tuple[GateCell, ...]:
    if shard_id == "serial_lane":
        return tuple(cell for cell in manifest.cells if "serial_lane" in cell.prerequisites)
    if shard_id == "xdist":
        return tuple(cell for cell in manifest.cells if "serial_lane" not in cell.prerequisites)
    return manifest.cells


def _validate_deferred_cell(state: GateState, cell: GateCell, job: GateJob) -> dict[str, object]:
    evidence: dict[str, object] = {
        "scenario_id": cell.scenario_id,
        "nodeid": cell.nodeid,
        "item_nodeids": list(cell.item_nodeids),
        "status": "deferred",
    }
    if state.provision_problems.get("serial_lane") is None:
        _record_violation(state, f"cell:{cell.scenario_id}", "serial-lane deferral was claimed as available")
    if cell.jobs[job.job_id]["disposition"] != "RUN":
        _record_violation(state, f"cell:{cell.scenario_id}", "only RUN cells can be deferred to the serial shard")
    item_nodeids = _collected_item_nodeids(state, cell)
    if len(item_nodeids) != len(cell.item_nodeids):
        evidence["status"] = "missing"
        return evidence
    reports_by_node: list[dict[str, object]] = []
    for nodeid in item_nodeids:
        if nodeid not in state.selected_nodeids:
            _record_violation(state, f"cell:{cell.scenario_id}", f"serial-lane item was deselected: {nodeid}")
        outcomes = _item_report_outcomes(state, nodeid)
        report_problem = _skip_reports_problem(outcomes)
        skipped = [
            report for phase_reports in state.reports.get(nodeid, {}).values()
            for report in phase_reports if report.outcome == "skipped"
        ]
        if report_problem is not None or len(skipped) != 1 or skipped[0].when != "setup" \
                or "serial_lane:" not in str(skipped[0].longrepr):
            _record_violation(
                state, f"cell:{cell.scenario_id}",
                f"xdist item lacks the serial-lane deferral marker: {nodeid}",
            )
            evidence["status"] = "unqualified"
        reports_by_node.append({
            "nodeid": nodeid,
            "reports": outcomes,
            "skip_reasons": [str(report.longrepr) for report in skipped],
        })
    evidence["items"] = reports_by_node
    state.deferred_cell_ids.add(cell.scenario_id)
    return evidence


def _session_evidence(session: pytest.Session, state: GateState) -> None:
    job = state.job
    if job is None:
        for cell in state.manifest.cells:
            _record_violation(state, f"cell:{cell.scenario_id}", "no valid job selected for this run")
        return
    if state.shard_id == "xdist":
        deferred_cells = tuple(cell for cell in state.manifest.cells if "serial_lane" in cell.prerequisites)
        for cell in deferred_cells:
            deferred = _validate_deferred_cell(state, cell, job)
            state.deferred_evidence.append(deferred)
            state.cell_evidence.append(deferred)
    for cell in _cells_for_shard(state.manifest, state.shard_id):
        state.cell_evidence.append(_validate_run_cell(state, cell, job))
    expected_ids = {cell.scenario_id for cell in _cells_for_shard(state.manifest, state.shard_id)}
    scanned_ids = set(state.scanned_cell_ids)
    if expected_ids != scanned_ids:
        _record_violation(state, "gate:coverage",
                          f"cell inventory mismatch: missing={sorted(expected_ids - scanned_ids)} "
                          f"unexpected={sorted(scanned_ids - expected_ids)}")
    if state.shard_id == "xdist":
        full_ids = {cell.scenario_id for cell in state.manifest.cells}
        accounted = scanned_ids | state.deferred_cell_ids
        if accounted != full_ids:
            _record_violation(
                state, "gate:coverage",
                f"xdist plus deferred inventory mismatch: missing={sorted(full_ids - accounted)} "
                f"unexpected={sorted(accounted - full_ids)}",
            )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    state = session.config.stash[_STATE_KEY]
    if state.is_worker or not state.must_run:
        return
    _session_evidence(session, state)
    state.summary = _run_summary(state)
    if state.shard_id == "xdist":
        expected_ids = {cell.scenario_id for cell in state.manifest.cells}
        scanned_ids = state.scanned_cell_ids | state.deferred_cell_ids
    else:
        expected_ids = {cell.scenario_id for cell in _cells_for_shard(state.manifest, state.shard_id)}
        scanned_ids = state.scanned_cell_ids
    if expected_ids != scanned_ids or state.violations:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    _write_run_evidence(state)
    state.summary = _run_summary(state)
    if state.violations:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    del exitstatus
    state = config.stash[_STATE_KEY]
    if state.is_worker or not state.must_run:
        return
    terminalreporter.write_line(state.summary)
    for key, reasons in sorted(state.violations.items()):
        terminalreporter.write_line(f"GUARD must_run violation: {key}: {'; '.join(reasons)}")
    for cell in state.cell_evidence:
        if cell["status"] in {"missing", "unqualified", "prerequisite_na", "platform_na"}:
            terminalreporter.write_line(
                f"GUARD must_run cell: {cell['scenario_id']} {cell['status']} "
                f"{cell.get('reason', cell['nodeid'])}"
            )


def _evidence_problem(
    path: Path, ticket: dict[str, object], manifest: GateManifest, expected_shard: str
) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"evidence cannot be read: {error}"
    keys = {
        "schema", "job_id", "shard_id", "run_nonce", "candidate_digest", "manifest_digest",
        "started_at_ns", "finished_at_ns", "expected_cell_ids", "scanned_cell_ids",
        "deferred_cell_ids", "cells", "deferred_cells", "optional_skips", "violations",
    }
    if (not isinstance(payload, dict) or set(payload) != keys
            or type(payload.get("schema")) is not int or payload["schema"] != 2):
        return "evidence has invalid fields or schema"
    if payload.get("shard_id") != expected_shard:
        return "evidence shard id does not match its expected file"
    for field_name in ("job_id", "run_nonce", "candidate_digest", "manifest_digest"):
        if payload.get(field_name) != ticket[field_name]:
            return f"stale or wrong-run evidence: {field_name} does not match its ticket"
    if type(payload.get("started_at_ns")) is not int or type(payload.get("finished_at_ns")) is not int:
        return "evidence timestamps are invalid"
    now = time.time_ns()
    if (payload["started_at_ns"] < ticket["issued_at_ns"] or payload["started_at_ns"] > now
            or payload["finished_at_ns"] < payload["started_at_ns"]
            or payload["finished_at_ns"] > now
            or now - payload["finished_at_ns"] > _MAX_TICKET_AGE_NS):
        return "evidence is stale or predates its ticket"
    issued_at_ns = ticket["issued_at_ns"]
    if issued_at_ns > now or now - issued_at_ns > _MAX_TICKET_AGE_NS:
        return "ticket is stale"
    shard_cells = _cells_for_shard(manifest, expected_shard)
    expected = [cell.scenario_id for cell in shard_cells]
    if payload.get("expected_cell_ids") != expected:
        return f"{expected_shard} evidence inventory differs from the reviewed manifest"
    scanned = payload.get("scanned_cell_ids")
    if (not isinstance(scanned, list) or any(not isinstance(item, str) for item in scanned)
            or len(scanned) != len(set(scanned))):
        return "evidence scanned scenario inventory is invalid"
    if set(scanned) != set(expected):
        return f"evidence coverage mismatch: missing={sorted(set(expected) - set(scanned))}"
    serial_cells = tuple(cell for cell in manifest.cells if "serial_lane" in cell.prerequisites)
    expected_deferred = [cell.scenario_id for cell in serial_cells] if expected_shard == "xdist" else []
    deferred_ids = payload.get("deferred_cell_ids")
    if (not isinstance(deferred_ids, list) or any(not isinstance(item, str) for item in deferred_ids)
            or len(deferred_ids) != len(set(deferred_ids)) or set(deferred_ids) != set(expected_deferred)):
        return "evidence deferred scenario inventory differs from the reviewed manifest"
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != len(expected):
        return "evidence cell reports are missing or duplicated"
    if any(not isinstance(item, dict) or not isinstance(item.get("scenario_id"), str) for item in cells):
        return "evidence cell reports have invalid scenario identifiers"
    by_id = {item.get("scenario_id"): item for item in cells if isinstance(item, dict)}
    if len(by_id) != len(expected) or set(by_id) != set(expected):
        return "evidence cell reports do not match the scenario inventory"
    violations = payload.get("violations")
    if not isinstance(violations, dict) or violations:
        return "job evidence records violations"
    optional_skips = payload.get("optional_skips")
    if not isinstance(optional_skips, list):
        return "optional skip accounting is invalid"
    optional_skip_nodes: set[str] = set()
    for skip in optional_skips:
        fields = {"nodeid", "module", "extra", "tool", "reason", "validated_absence"}
        if (not isinstance(skip, dict) or set(skip) != fields
                or any(not isinstance(skip.get(name), str) or not skip[name].strip()
                       for name in ("nodeid", "module", "extra", "tool", "reason"))
                or type(skip.get("validated_absence")) is not bool):
            return "optional skip evidence has invalid fields"
        matching = [
            entry for entry in manifest.optional_dependencies
            if (entry.module == skip["module"] and entry.extra == skip["extra"]
                and entry.tool == skip["tool"]
                and any(_optional_node_matches(pattern, skip["nodeid"]) for pattern in entry.tests)
                and any(reason in skip["reason"] for reason in entry.absence_reasons))
        ]
        if not matching or skip["validated_absence"] is not True:
            return f"optional skip is not a reviewed dependency absence: {skip['nodeid']}"
        optional_skip_nodes.add(skip["nodeid"])
    job_id = payload.get("job_id")
    if not isinstance(job_id, str) or job_id not in manifest.jobs:
        return "evidence names an unknown matrix job"
    for cell in shard_cells:
        item = by_id[cell.scenario_id]
        status = item.get("status")
        if not isinstance(status, str):
            return f"scenario {cell.scenario_id} has an invalid status"
        if (item.get("nodeid") != cell.nodeid or item.get("baseline") != cell.baseline
                or item.get("witness_schema") != cell.witness_schema
                or item.get("item_nodeids") != list(cell.item_nodeids)):
            return f"scenario {cell.scenario_id} evidence identity differs from the manifest"
        disposition = cell.jobs[job_id]
        expected_disposition = disposition["disposition"]
        item_reports = item.get("items")
        if (not isinstance(item_reports, list)
                or any(not isinstance(value, dict) for value in item_reports)
                or [value.get("nodeid") for value in item_reports] != list(cell.item_nodeids)):
            return f"scenario {cell.scenario_id} item reports differ from the manifest"
        if status == "qualified":
            if expected_disposition != "RUN":
                return f"scenario {cell.scenario_id} was qualified against a non-RUN disposition"
            for item_report in item_reports:
                reports = item_report.get("reports")
                if (not isinstance(reports, dict)
                        or set(reports) != {"setup", "call", "teardown"}
                        or any(reports.get(phase) != "passed" for phase in ("setup", "call", "teardown"))
                        or type(item_report.get("assertion_passes")) is not int
                        or item_report["assertion_passes"] < 0):
                    return f"scenario {cell.scenario_id} has an incomplete passing item witness"
                if cell.witness_schema == "assertion_match":
                    if (item_report["assertion_passes"] < 1
                            or item_report.get("assertion_witness") is not True):
                        return f"scenario {cell.scenario_id} lacks its scenario-specific assertion witness"
            if cell.witness_schema == "parity_trace":
                witness = item.get("parity_witness")
                if (not isinstance(witness, dict) or witness.get("comparison_completed") is not True
                        or type(witness.get("baseline_count")) is not int or witness["baseline_count"] < 1
                        or type(witness.get("candidate_count")) is not int or witness["candidate_count"] < 1
                        or not _SHA256_PATTERN.fullmatch(str(witness.get("baseline_digest", "")))
                        or not _SHA256_PATTERN.fullmatch(str(witness.get("candidate_digest", "")))):
                    return f"scenario {cell.scenario_id} has invalid parity trace evidence"
            continue
        if status == "platform_na":
            if expected_disposition != "NA_PLATFORM" or item.get("reason") != disposition.get("reason"):
                return f"scenario {cell.scenario_id} has an unreviewed platform disposition"
            for item_report in item_reports:
                report_problem = _skip_reports_problem(item_report.get("reports"))
                reasons = item_report.get("skip_reasons")
                if (report_problem is not None or not isinstance(reasons, list)
                        or not reasons or any(not isinstance(reason, str) for reason in reasons)
                        or not any(disposition["reason"] in reason for reason in reasons)):
                    return f"scenario {cell.scenario_id} lacks clean platform skip evidence"
            continue
        if status == "prerequisite_na":
            missing = item.get("missing_prerequisites")
            if (expected_disposition != "RUN" or not isinstance(missing, list) or not missing
                    or any(not isinstance(value, str) for value in missing)
                    or set(missing) - set(cell.optional_prerequisites)):
                return f"scenario {cell.scenario_id} has an unreviewed prerequisite disposition"
            for item_report in item_reports:
                if (_skip_reports_problem(item_report.get("reports")) is not None
                        or item_report.get("nodeid") not in optional_skip_nodes):
                    return f"scenario {cell.scenario_id} has invalid prerequisite skip evidence"
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                return f"scenario {cell.scenario_id} has an unreviewed prerequisite disposition"
            continue
        if status not in {"platform_na", "prerequisite_na"}:
            return f"scenario {cell.scenario_id} is not qualified: {status}"
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            return f"scenario {cell.scenario_id} has no explicit not-applicable reason"

    deferred_cells = payload.get("deferred_cells")
    if not isinstance(deferred_cells, list) or len(deferred_cells) != len(expected_deferred):
        return "xdist deferred cell evidence is missing or duplicated"
    deferred_by_id = {
        item.get("scenario_id"): item for item in deferred_cells if isinstance(item, dict)
    }
    if len(deferred_by_id) != len(expected_deferred) or set(deferred_by_id) != set(expected_deferred):
        return "xdist deferred cell evidence differs from the reviewed manifest"
    for cell in serial_cells if expected_shard == "xdist" else ():
        item = deferred_by_id[cell.scenario_id]
        if (item.get("status") != "deferred" or item.get("nodeid") != cell.nodeid
                or item.get("item_nodeids") != list(cell.item_nodeids)
                or cell.jobs[job_id]["disposition"] != "RUN"):
            return f"scenario {cell.scenario_id} has invalid xdist deferral identity"
        reports = item.get("items")
        if (not isinstance(reports, list)
                or any(not isinstance(value, dict) for value in reports)
                or [value.get("nodeid") for value in reports] != list(cell.item_nodeids)):
            return f"scenario {cell.scenario_id} has invalid xdist deferral reports"
        for item_report in reports:
            item_reports = item_report.get("reports")
            skip_reasons = item_report.get("skip_reasons")
            if (_skip_reports_problem(item_reports) is not None
                    or not isinstance(item_reports, dict)
                    or item_reports.get("setup") != ["skipped"]
                    or item_reports.get("call", []) != []
                    or item_reports.get("teardown", []) not in ([], ["passed"])
                    or not isinstance(skip_reasons, list)
                    or len(skip_reasons) != 1
                    or not isinstance(skip_reasons[0], str)
                    or "serial_lane:" not in skip_reasons[0]):
                return f"scenario {cell.scenario_id} lacks the serial-lane deferral marker"
    return None


def _aggregate(root: Path, manifest_path: Path, ticket_dir: Path, evidence_dir: Path) -> int:
    manifest, manifest_digest = _read_manifest(root, manifest_path)
    candidate_digest = _candidate_digest(root)
    violations: list[str] = []
    scanned_jobs: set[str] = set()
    nonces: set[str] = set()
    for job in manifest.jobs.values():
        ticket_path = ticket_dir / f"{job.job_id}.json"
        try:
            ticket = _read_ticket(ticket_path)
        except GateInputError as error:
            violations.append(f"{job.job_id}: {error}")
            continue
        if ticket["job_id"] != job.job_id:
            violations.append(f"{job.job_id}: ticket has the wrong job id")
            continue
        if ticket["manifest_digest"] != manifest_digest:
            violations.append(f"{job.job_id}: ticket manifest digest is stale")
            continue
        if ticket["candidate_digest"] != candidate_digest:
            violations.append(f"{job.job_id}: ticket candidate digest is stale")
            continue
        if ticket["run_nonce"] in nonces:
            violations.append(f"{job.job_id}: nonce was reused by another job")
            continue
        nonces.add(str(ticket["run_nonce"]))
        job_problems = []
        for shard_id in _required_shards(manifest):
            suffix = ".json" if shard_id == "full" else f".{shard_id}.json"
            evidence_path = evidence_dir / f"{job.job_id}{suffix}"
            problem = _evidence_problem(evidence_path, ticket, manifest, shard_id)
            if problem is not None:
                job_problems.append(f"{shard_id}: {problem}")
        if job_problems:
            violations.append(f"{job.job_id}: {'; '.join(job_problems)}")
            continue
        scanned_jobs.add(job.job_id)
    expected = set(manifest.jobs)
    violated = len(violations)
    print(
        f"GUARD must_run_aggregate: expected={len(expected)} scanned={len(scanned_jobs)} "
        f"violated={violated}"
    )
    for violation in violations:
        print(f"GUARD must_run_aggregate violation: {violation}")
    return 0 if expected == scanned_jobs and violated == 0 else 1


def _print_provisions(root: Path, manifest: GateManifest, job_id: str) -> int:
    job = manifest.jobs.get(job_id)
    if job is None:
        print(f"unknown job: {job_id}", file=sys.stderr)
        return 2
    available: list[str] = []
    problems: list[str] = []
    for name in job.required_provisions:
        problem = _provision_problem(name, root, manifest, None)
        if problem is None:
            available.append(name)
        else:
            problems.append(f"{name}: {problem}")
    for name in job.conditional_provisions:
        problem = _provision_problem(name, root, manifest, None)
        if problem is None:
            available.append(name)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(",".join(available))
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Prepare and verify must-run release gate evidence")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidate-root", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--ticket-dir", type=Path, required=True)
    prepare.add_argument("--evidence-dir", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--candidate-root", type=Path, required=True)
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--ticket-dir", type=Path, required=True)
    aggregate.add_argument("--evidence-dir", type=Path, required=True)
    provisions = subparsers.add_parser("provisions")
    provisions.add_argument("--candidate-root", type=Path, required=True)
    provisions.add_argument("--manifest", type=Path, required=True)
    provisions.add_argument("--job", required=True)
    arguments = parser.parse_args()
    root = arguments.candidate_root.resolve()
    try:
        manifest, _ = _read_manifest(root, arguments.manifest.resolve())
        if arguments.command == "prepare":
            _prepare(root, arguments.manifest.resolve(), arguments.ticket_dir.resolve(),
                     arguments.evidence_dir.resolve())
            return 0
        if arguments.command == "aggregate":
            return _aggregate(root, arguments.manifest.resolve(), arguments.ticket_dir.resolve(),
                              arguments.evidence_dir.resolve())
        return _print_provisions(root, manifest, arguments.job)
    except GateInputError as error:
        print(f"must-run gate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
