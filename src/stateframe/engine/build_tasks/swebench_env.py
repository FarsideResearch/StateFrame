"""Provision isolated Python environments and prove SWE-bench rows locally.

The adapter deliberately keeps checkout materialization and environment setup separate.
This module owns the latter boundary: one venv per instance, bounded installation, a
serialized ``python`` path for the harness, and an honest parent-red/gold-green proof
record.  Expected dependency and interpreter failures are returned as named ``ENV_*``
blockers instead of being mistaken for benchmark failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

# Keep both ``python -m build_tasks.swebench_env`` and the documented direct script
# invocation usable when build_tasks remains a namespace package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from build_tasks.swebench_adapter import make_runner_test, verify_instance


DEFAULT_TIMEOUT_SECONDS = 600
SUPPORTED_PYTHON_VERSIONS = ("3.11", "3.10")
ENVIRONMENT_RECEIPT_SCHEMA = "poc.environment.receipt/1"
REPAIR_CONTRACT_ID = "env.install_dependencies/1"
UV_VERSION_PIN = "0.12.9"
UV_BOOTSTRAP_PIP_ARGS = (
    "python3",
    "-m",
    "pip",
    "install",
    "--break-system-packages",
    f"uv=={UV_VERSION_PIN}",
)
UV_BOOTSTRAP_PYTHON_ARGS = ("uv", "python", "install", "3.11")
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_READY_MARKER = ".swebench-env.json"
_FAILED_MARKER = ".swebench-env-failed.json"
_CREATING_MARKER = ".swebench-env-creating"
_LOCK_SUFFIX = ".lock"
_UV_BOOTSTRAP_MARKER = ".swebench-uv-bootstrap.json"
_UV_BOOTSTRAP_LOCK = ".swebench-uv-bootstrap.lock"
_REPAIR_CONTRACT_MARKER = ".swebench-repair-contract.json"
_ENV_RECEIPT_MARKER = ".swebench-env-receipt.json"
_TAIL_LINES = 40
_DEPENDENCY_MANIFEST_NAMES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "tox.ini",
    ".python-version",
)
_C_BUILD_SIGNATURE_RE = re.compile(
    r"failed building wheel|subprocess-exited-with-error|missing header|"
    r"fatal error: .*\.h: .*no such file",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CommandEvidence:
    """Bounded subprocess evidence retained for reports and tests."""

    args: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def output(self) -> str:
        return (self.stdout + self.stderr).strip()

    @property
    def tail(self) -> str:
        return _tail(self.output)

    def as_dict(self) -> dict[str, object]:
        return {
            "args": list(self.args),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_tail": _tail(self.stdout),
            "stderr_tail": _tail(self.stderr),
        }


@dataclass(frozen=True)
class PythonSelection:
    """The launcher selected before a per-instance venv is created."""

    backend: str
    requested_version: str
    executable: Path


@dataclass(frozen=True)
class EnvironmentReceipt:
    """Typed, non-chargeable evidence for one environment probe or setup step."""

    probe: str
    result_class: str
    exit_code: int
    timed_out: bool
    output_digest: str
    evidence_ref: str
    chargeable: bool = False
    blocker_code: str | None = None
    kind: str = "environment"
    schema: str = ENVIRONMENT_RECEIPT_SCHEMA

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": self.schema,
            "kind": self.kind,
            "probe": self.probe,
            "result_class": self.result_class,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "output_digest": self.output_digest,
            "evidence_ref": self.evidence_ref,
            "chargeable": self.chargeable,
        }
        if self.blocker_code is not None:
            value["blocker_code"] = self.blocker_code
        return value

    @classmethod
    def from_dict(cls, value: object) -> "EnvironmentReceipt | None":
        if not isinstance(value, Mapping):
            return None
        schema = value.get("schema")
        kind = value.get("kind")
        probe = value.get("probe")
        result_class = value.get("result_class")
        exit_code = value.get("exit_code")
        timed_out = value.get("timed_out")
        output_digest = value.get("output_digest")
        evidence_ref = value.get("evidence_ref")
        chargeable = value.get("chargeable")
        if schema != ENVIRONMENT_RECEIPT_SCHEMA or kind != "environment":
            return None
        if not all(
            isinstance(item, str)
            for item in (probe, result_class, output_digest, evidence_ref)
        ):
            return None
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            return None
        if not isinstance(timed_out, bool) or not isinstance(chargeable, bool):
            return None
        blocker_code = value.get("blocker_code")
        if blocker_code is not None and not isinstance(blocker_code, str):
            return None
        return cls(
            probe=probe,
            result_class=result_class,
            exit_code=exit_code,
            timed_out=timed_out,
            output_digest=output_digest,
            evidence_ref=evidence_ref,
            chargeable=chargeable,
            blocker_code=blocker_code,
        )


@dataclass(frozen=True)
class RepairContract:
    """Persisted bounded dependency-install procedure and its progress."""

    instance_id: str
    manifest_digest: str
    goal: str
    precondition: dict[str, object]
    risk: dict[str, object]
    repair: dict[str, object]
    success: tuple[str, ...]
    failure: dict[str, object]
    contract_id: str = REPAIR_CONTRACT_ID
    status: str = "pending"
    completed_steps: tuple[str, ...] = ()
    receipts: tuple[EnvironmentReceipt, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "goal": self.goal,
            "precondition": dict(self.precondition),
            "risk": dict(self.risk),
            "repair": dict(self.repair),
            "success": list(self.success),
            "failure": dict(self.failure),
            "instance_id": self.instance_id,
            "manifest_digest": self.manifest_digest,
            "status": self.status,
            "completed_steps": list(self.completed_steps),
            "receipts": [receipt.as_dict() for receipt in self.receipts],
        }

    @classmethod
    def from_dict(cls, value: object) -> "RepairContract | None":
        if not isinstance(value, Mapping):
            return None
        contract_id = value.get("contract_id")
        instance_id = value.get("instance_id")
        manifest_digest = value.get("manifest_digest")
        goal = value.get("goal")
        status = value.get("status", "pending")
        if not all(
            isinstance(item, str)
            for item in (contract_id, instance_id, manifest_digest, goal, status)
        ):
            return None
        if contract_id != REPAIR_CONTRACT_ID:
            return None
        precondition = _object_dict(value.get("precondition"))
        risk = _object_dict(value.get("risk"))
        repair = _object_dict(value.get("repair"))
        failure = _object_dict(value.get("failure"))
        success_value = value.get("success")
        completed_value = value.get("completed_steps", [])
        receipts_value = value.get("receipts", [])
        if not isinstance(success_value, list) or not isinstance(completed_value, list):
            return None
        if not isinstance(receipts_value, list):
            return None
        success = tuple(item for item in success_value if isinstance(item, str))
        completed_steps = tuple(
            item for item in completed_value if isinstance(item, str)
        )
        receipts = tuple(
            receipt
            for raw_receipt in receipts_value
            if (receipt := EnvironmentReceipt.from_dict(raw_receipt)) is not None
        )
        return cls(
            instance_id=instance_id,
            manifest_digest=manifest_digest,
            goal=goal,
            precondition=precondition,
            risk=risk,
            repair=repair,
            success=success,
            failure=failure,
            contract_id=contract_id,
            status=status,
            completed_steps=completed_steps,
            receipts=receipts,
        )


@dataclass(frozen=True)
class UVBootstrapResult:
    """One cache-root uv bootstrap decision, including its durable receipts."""

    ready: bool
    executable: Path | None
    blocker: str | None
    receipts: tuple[EnvironmentReceipt, ...] = ()

    @property
    def receipt(self) -> EnvironmentReceipt | None:
        return self.receipts[0] if self.receipts else None


@dataclass(frozen=True)
class EnvironmentResult:
    """One instance's environment result, including an explicit blocker if needed."""

    instance_id: str
    environment_dir: Path
    python: Path | None
    install_ok: bool
    blocker: str | None = None
    backend: str | None = None
    requested_version: str | None = None
    venv: CommandEvidence | None = None
    install: CommandEvidence | None = None
    reused: bool = False
    receipts: tuple[EnvironmentReceipt, ...] = ()
    repair_contract: RepairContract | None = None

    @property
    def ready(self) -> bool:
        return self.install_ok and self.python is not None and self.blocker is None

    def as_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "environment_dir": str(self.environment_dir),
            "python": str(self.python) if self.python else None,
            "install_ok": self.install_ok,
            "blocker": self.blocker,
            "backend": self.backend,
            "requested_version": self.requested_version,
            "reused": self.reused,
            "ready": self.ready,
            "venv": self.venv.as_dict() if self.venv else None,
            "install": self.install.as_dict() if self.install else None,
            "receipts": [receipt.as_dict() for receipt in self.receipts],
            "repair_contract": (
                self.repair_contract.as_dict() if self.repair_contract else None
            ),
        }


@dataclass(frozen=True)
class EnvironmentBatch:
    """Enriched rows plus the setup evidence in matching order."""

    rows: tuple[dict[str, object], ...]
    environments: tuple[EnvironmentResult, ...]


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


class EnvironmentProvider(Protocol):
    """Small injection boundary used by synthetic row-enrichment tests."""

    def prepare(
        self,
        instance_id: str,
        repo_dir: Path,
        *,
        repo: str | None = None,
    ) -> EnvironmentResult: ...


def _tail(value: str, *, lines: int = _TAIL_LINES) -> str:
    return "\n".join(value.strip().splitlines()[-lines:])


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _output_digest(output: str) -> str:
    return f"sha256:{hashlib.sha256(output.encode('utf-8')).hexdigest()}"


def _blocker_code(blocker: str | None) -> str | None:
    if not blocker:
        return None
    code, separator, _detail = blocker.partition(":")
    return code if separator and code.startswith("ENV_") else None


def _receipt_from_evidence(
    *,
    probe: str,
    evidence: CommandEvidence,
    evidence_ref: str,
    success: bool,
    blocker_code: str | None = None,
) -> EnvironmentReceipt:
    return EnvironmentReceipt(
        probe=probe,
        result_class="ENV_READY" if success else "ENV_BLOCKED",
        exit_code=evidence.returncode,
        timed_out=evidence.timed_out,
        output_digest=_output_digest(evidence.output),
        evidence_ref=evidence_ref,
        blocker_code=blocker_code,
    )


def _summary_receipt(
    instance_id: str,
    receipts: Sequence[EnvironmentReceipt],
    *,
    ready: bool,
    blocker: str | None,
) -> EnvironmentReceipt:
    if receipts:
        exit_code = receipts[-1].exit_code
        timed_out = any(receipt.timed_out for receipt in receipts)
        digest_input = "\n".join(receipt.output_digest for receipt in receipts)
    else:
        exit_code = 0 if ready else 1
        timed_out = False
        digest_input = ""
    return EnvironmentReceipt(
        probe="environment_setup",
        result_class="ENV_READY" if ready else "ENV_BLOCKED",
        exit_code=exit_code,
        timed_out=timed_out,
        output_digest=_output_digest(digest_input),
        evidence_ref=f"env/{instance_id}/environment_setup/0",
        blocker_code=_blocker_code(blocker),
    )


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.tmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _receipt_paths(env_dir: Path) -> tuple[Path, ...]:
    return (env_dir / _ENV_RECEIPT_MARKER,)


def _read_environment_receipts(env_dir: Path) -> tuple[EnvironmentReceipt, ...]:
    for path in _receipt_paths(env_dir):
        payload = _read_marker(path)
        if payload is None:
            continue
        receipts: list[EnvironmentReceipt] = []
        summary = EnvironmentReceipt.from_dict(payload)
        if summary is not None:
            receipts.append(summary)
        nested = payload.get("receipts")
        if isinstance(nested, list):
            receipts.extend(
                receipt
                for raw_receipt in nested
                if (receipt := EnvironmentReceipt.from_dict(raw_receipt)) is not None
            )
        if receipts:
            return tuple(receipts)
    return ()


def _persist_environment_receipts(
    env_dir: Path,
    *,
    instance_id: str,
    receipts: Sequence[EnvironmentReceipt],
    ready: bool,
    blocker: str | None,
) -> tuple[EnvironmentReceipt, ...]:
    summary = _summary_receipt(
        instance_id,
        receipts,
        ready=ready,
        blocker=blocker,
    )
    all_receipts = (summary, *receipts)
    payload = summary.as_dict()
    payload.update(
        {
            "instance_id": instance_id,
            "receipts": [receipt.as_dict() for receipt in receipts],
        }
    )
    try:
        env_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(env_dir / _ENV_RECEIPT_MARKER, payload)
    except OSError:
        # The setup result remains honest; the marker writer below records the
        # operational failure and the caller can retry in a fresh cache.
        pass
    return all_receipts


def _manifest_paths(repository: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in _DEPENDENCY_MANIFEST_NAMES:
        candidate = repository / name
        if candidate.is_file():
            paths.append(candidate)
    for pattern in ("requirements*.txt", "requirements*.in"):
        paths.extend(
            candidate
            for candidate in sorted(repository.glob(pattern))
            if candidate.is_file()
        )
    return tuple(dict.fromkeys(paths))


def _dependency_manifest_digest(repository: Path) -> tuple[str, tuple[str, ...]]:
    manifest_paths = _manifest_paths(repository)
    digest = hashlib.sha256()
    if not manifest_paths:
        digest.update(b"<no dependency manifest>\n")
    for path in manifest_paths:
        relative = path.relative_to(repository).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}", tuple(
        path.relative_to(repository).as_posix() for path in manifest_paths
    )


def _new_repair_contract(
    instance_id: str,
    *,
    manifest_digest: str,
    manifest_names: Sequence[str],
    timeout: int,
    initial_receipts: Sequence[EnvironmentReceipt] = (),
) -> RepairContract:
    recipe_source = ", ".join(manifest_names) or "repository package metadata"
    evidence_refs = [receipt.evidence_ref for receipt in initial_receipts]
    return RepairContract(
        instance_id=instance_id,
        manifest_digest=manifest_digest,
        goal="make target importable and oracle collectable",
        precondition={
            "signal_class": "MISSING_DEPENDENCY",
            "manifest_digest": manifest_digest,
            "workspace_revision": 0,
            "evidence_refs": evidence_refs,
            "target_unchanged": True,
        },
        risk={
            "network": "allowlisted_only",
            "filesystem": "environment_cache_only",
            "service": "none",
            "secrets": "none",
            "timeout_seconds": timeout,
        },
        repair={
            "kind": "venv_install",
            "recipe_source": recipe_source,
            "recipe_digest": manifest_digest,
            "max_runs": 1,
            "idempotency_key": f"{instance_id}:{manifest_digest}",
        },
        success=("python_version", "target_import", "oracle_collect"),
        failure={
            "blocker_code": "ENV_INSTALL_FAILED",
            "terminal_action": "ENV_BLOCKER",
            "chargeable": False,
        },
        receipts=tuple(initial_receipts),
    )


def _contract_with_observation(
    contract: RepairContract,
    receipt: EnvironmentReceipt,
    *,
    completed_step: str | None = None,
    status: str | None = None,
) -> RepairContract:
    completed_steps = contract.completed_steps
    if completed_step is not None and completed_step not in completed_steps:
        completed_steps = (*completed_steps, completed_step)
    receipts = (*contract.receipts, receipt)
    precondition = dict(contract.precondition)
    precondition["evidence_refs"] = [item.evidence_ref for item in receipts]
    return replace(
        contract,
        precondition=precondition,
        completed_steps=completed_steps,
        receipts=receipts,
        status=status or contract.status,
    )


def _contract_paths(env_dir: Path) -> tuple[Path, ...]:
    return (
        env_dir / _REPAIR_CONTRACT_MARKER,
        env_dir.with_name(env_dir.name + _REPAIR_CONTRACT_MARKER),
    )


def _read_repair_contract(env_dir: Path) -> RepairContract | None:
    for path in _contract_paths(env_dir):
        contract = RepairContract.from_dict(_read_marker(path))
        if contract is not None:
            return contract
    return None


def _write_repair_contract(env_dir: Path, contract: RepairContract) -> None:
    payload = contract.as_dict()
    paths = [env_dir.with_name(env_dir.name + _REPAIR_CONTRACT_MARKER)]
    if env_dir.is_dir():
        paths.append(env_dir / _REPAIR_CONTRACT_MARKER)
    for path in paths:
        _atomic_write_json(path, payload)


def _receipt_blocker(
    *,
    probe: str,
    blocker: str,
    returncode: int = 1,
    timed_out: bool = False,
) -> EnvironmentReceipt:
    return EnvironmentReceipt(
        probe=probe,
        result_class="ENV_BLOCKED",
        exit_code=returncode,
        timed_out=timed_out,
        output_digest=_output_digest(blocker),
        evidence_ref=f"env/{probe}/0",
        blocker_code=_blocker_code(blocker),
    )


def _run_command(
    args: Sequence[str | Path],
    *,
    cwd: Path | None = None,
    timeout: int,
) -> CommandEvidence:
    normalized = tuple(str(arg) for arg in args)
    try:
        process = subprocess.run(
            list(normalized),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandEvidence(
            normalized,
            returncode=124,
            stdout=_decode_output(exc.stdout),
            stderr=_decode_output(exc.stderr),
            timed_out=True,
        )
    except OSError as exc:
        return CommandEvidence(
            normalized, returncode=127, stderr=f"{type(exc).__name__}: {exc}"
        )
    return CommandEvidence(
        normalized,
        returncode=process.returncode,
        stdout=_decode_output(process.stdout),
        stderr=_decode_output(process.stderr),
    )


def _remaining(deadline: float) -> int | None:
    seconds = math.ceil(deadline - time.monotonic())
    return seconds if seconds > 0 else None


def _validate_timeout(timeout: int) -> None:
    if timeout < 1:
        raise ValueError("timeout must be at least 1 second")


def _safe_instance_id(instance_id: str) -> str:
    if (
        not instance_id
        or instance_id in {".", ".."}
        or not _INSTANCE_ID_RE.fullmatch(instance_id)
    ):
        raise ValueError(
            "instance_id must contain only letters, digits, underscore, period, or hyphen"
        )
    return instance_id


def environment_dir(cache_dir: Path | str, instance_id: str) -> Path:
    """Return the exact per-instance venv path under ``<cache>/envs``."""
    safe_id = _safe_instance_id(instance_id)
    return Path(cache_dir).expanduser().resolve() / "envs" / safe_id


def _venv_python(env_dir: Path) -> Path | None:
    candidates = (env_dir / "bin" / "python", env_dir / "Scripts" / "python.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _probe_python_version(executable: Path, *, timeout: int) -> str | None:
    version, _evidence = _probe_python_version_evidence(executable, timeout=timeout)
    return version


def _probe_python_version_evidence(
    executable: Path,
    *,
    timeout: int,
) -> tuple[str | None, CommandEvidence]:
    result = _run_command(
        [
            executable,
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        timeout=timeout,
    )
    if result.returncode != 0 or result.timed_out:
        return None, result
    for line in reversed(result.stdout.splitlines()):
        if re.fullmatch(r"\d+\.\d+", line.strip()):
            return line.strip(), result
    return None, result


def _system_python_candidates(version: str) -> tuple[Path, ...]:
    name = f"python{version}"
    candidates: list[Path] = []
    on_path = shutil.which(name)
    if on_path:
        candidates.append(Path(on_path).resolve())
    major, minor = version.split(".")
    for candidate in (
        Path(f"/opt/homebrew/opt/python@{version}/bin/{name}"),
        Path(f"/usr/local/opt/python@{version}/bin/{name}"),
        Path(f"/opt/homebrew/bin/{name}"),
        Path(f"/usr/local/bin/{name}"),
    ):
        if candidate.is_file():
            candidates.append(candidate.resolve())
    current = sys.version_info
    if (current.major, current.minor) == (int(major), int(minor)):
        candidates.append(Path(sys.executable).resolve())
    return tuple(dict.fromkeys(candidates))


def _pyenv_python_candidates(pyenv: Path, version: str) -> tuple[Path, ...]:
    result = _run_command([pyenv, "which", f"python{version}"], timeout=30)
    if result.returncode != 0 or result.timed_out:
        return ()
    candidates: list[Path] = []
    for line in result.stdout.splitlines():
        candidate = Path(line.strip()).expanduser()
        if candidate.is_file():
            candidates.append(candidate.resolve())
    return tuple(dict.fromkeys(candidates))


def select_python() -> PythonSelection | None:
    """Select uv 3.11, then pyenv/system 3.11 or 3.10, in that order."""
    uv = shutil.which("uv")
    if uv:
        return PythonSelection("uv", "3.11", Path(uv).resolve())

    pyenv = shutil.which("pyenv")
    if pyenv:
        pyenv_path = Path(pyenv).resolve()
        for version in SUPPORTED_PYTHON_VERSIONS:
            for candidate in _pyenv_python_candidates(pyenv_path, version):
                if _probe_python_version(candidate, timeout=30) == version:
                    return PythonSelection("pyenv", version, candidate)

    for version in SUPPORTED_PYTHON_VERSIONS:
        for candidate in _system_python_candidates(version):
            if _probe_python_version(candidate, timeout=30) == version:
                return PythonSelection("system", version, candidate)
    return None


def _no_python_blocker() -> str:
    current = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        "ENV_NO_COMPATIBLE_PYTHON: requires Python 3.11 or 3.10; "
        f"uv={'available' if shutil.which('uv') else 'not found'}; "
        f"pyenv={'available' if shutil.which('pyenv') else 'not found'}; "
        f"current interpreter={current}"
    )


def _is_astropy(repo: str | None, repo_dir: Path) -> bool:
    label = (repo or "").lower()
    return label == "astropy/astropy" or repo_dir.name.lower() == "astropy"


def _install_blocker(
    *,
    repo: str | None,
    repo_dir: Path,
    result: CommandEvidence,
) -> str:
    detail = result.tail.replace("\n", " | ")[:600] or "no installer output"
    if _is_astropy(repo, repo_dir):
        if result.timed_out:
            return (
                "ENV_ASTROPY_C_EXTENSION_TIMEOUT: astropy editable install exceeded "
                f"the bounded timeout; C-extension build cost is environment-blocked; {detail}"
            )
        return (
            "ENV_ASTROPY_C_EXTENSION_FAILED: astropy editable install failed while "
            f"building dependencies/extensions; {detail}"
        )
    if _C_BUILD_SIGNATURE_RE.search(result.output):
        return f"ENV_C_BUILD_FAILED: dependency build emitted a C-extension signature; {detail}"
    if result.timed_out:
        return f"ENV_INSTALL_TIMEOUT: editable install plus pytest exceeded the bound; {detail}"
    return f"ENV_INSTALL_FAILED: editable install plus pytest exited {result.returncode}; {detail}"


def _venv_blocker(label: str, result: CommandEvidence) -> str:
    detail = result.tail.replace("\n", " | ")[:600] or "no command output"
    if result.timed_out:
        return f"ENV_{label}_TIMEOUT: {detail}"
    return f"ENV_{label}_FAILED: exit={result.returncode}; {detail}"


def _uv_bootstrap_marker(cache_dir: Path) -> Path:
    return cache_dir / _UV_BOOTSTRAP_MARKER


def _bootstrap_payload(
    *,
    status: str,
    receipt: EnvironmentReceipt,
    receipts: Sequence[EnvironmentReceipt],
    blocker: str | None = None,
) -> dict[str, object]:
    return {
        "schema": "swebench-uv-bootstrap/1",
        "status": status,
        "uv_version": UV_VERSION_PIN,
        "blocker": blocker,
        "recipe": [
            f"python3 -m pip install --break-system-packages 'uv=={UV_VERSION_PIN}'",
            "uv python install 3.11",
        ],
        "receipt": receipt.as_dict(),
        "receipts": [item.as_dict() for item in receipts],
    }


def _bootstrap_result_from_marker(cache_dir: Path) -> UVBootstrapResult | None:
    marker_path = _uv_bootstrap_marker(cache_dir)
    if not marker_path.exists():
        return None
    marker = _read_marker(marker_path)
    if marker is None:
        blocker = "ENV_UV_BOOTSTRAP_FAILED: bootstrap marker is unreadable"
        return UVBootstrapResult(
            ready=False,
            executable=None,
            blocker=blocker,
            receipts=(_receipt_blocker(probe="uv_bootstrap", blocker=blocker),),
        )
    status = marker.get("status")
    summary = EnvironmentReceipt.from_dict(marker.get("receipt"))
    nested_value = marker.get("receipts")
    nested = tuple(
        receipt
        for raw_receipt in nested_value
        if (receipt := EnvironmentReceipt.from_dict(raw_receipt)) is not None
    ) if isinstance(nested_value, list) else ()
    receipts = ((summary,) if summary is not None else ()) + nested
    if status == "ready" and summary is not None:
        uv = shutil.which("uv")
        return UVBootstrapResult(
            ready=True,
            executable=Path(uv).resolve() if uv else Path("uv"),
            blocker=None,
            receipts=receipts,
        )
    marker_blocker = marker.get("blocker")
    blocker = (
        marker_blocker
        if isinstance(marker_blocker, str) and marker_blocker
        else "ENV_UV_BOOTSTRAP_FAILED: bootstrap marker records a failed recipe"
    )
    if summary is None:
        receipts = (*receipts, _receipt_blocker(probe="uv_bootstrap", blocker=blocker))
    return UVBootstrapResult(
        ready=False,
        executable=None,
        blocker=blocker,
        receipts=receipts,
    )


def _ensure_uv_bootstrap(
    cache_dir: Path,
    *,
    timeout: int,
) -> UVBootstrapResult:
    """Run the approved uv recipe once for a cache root, then reuse its marker."""
    existing = _bootstrap_result_from_marker(cache_dir)
    if existing is not None:
        return existing
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / _UV_BOOTSTRAP_LOCK
    try:
        lock_path.mkdir()
    except FileExistsError:
        blocker = "ENV_UV_BOOTSTRAP_IN_PROGRESS: another setup owns the cache bootstrap lock"
        return UVBootstrapResult(
            ready=False,
            executable=None,
            blocker=blocker,
            receipts=(_receipt_blocker(probe="uv_bootstrap", blocker=blocker),),
        )

    try:
        existing = _bootstrap_result_from_marker(cache_dir)
        if existing is not None:
            return existing
        deadline = time.monotonic() + timeout
        receipts: list[EnvironmentReceipt] = []

        remaining = _remaining(deadline)
        if remaining is None:
            blocker = "ENV_UV_BOOTSTRAP_FAILED: timeout expired before pip bootstrap"
            summary = _receipt_blocker(
                probe="uv_bootstrap",
                blocker=blocker,
                returncode=124,
                timed_out=True,
            )
            _atomic_write_json(
                _uv_bootstrap_marker(cache_dir),
                _bootstrap_payload(
                    status="failed",
                    receipt=summary,
                    receipts=receipts,
                    blocker=blocker,
                ),
            )
            return UVBootstrapResult(False, None, blocker, (summary,))
        pip_result = _run_command(UV_BOOTSTRAP_PIP_ARGS, timeout=remaining)
        pip_ok = pip_result.returncode == 0 and not pip_result.timed_out
        pip_receipt = _receipt_from_evidence(
            probe="uv_bootstrap_pip",
            evidence=pip_result,
            evidence_ref="env/bootstrap/uv_pip/0",
            success=pip_ok,
            blocker_code="ENV_UV_BOOTSTRAP_FAILED" if not pip_ok else None,
        )
        receipts.append(pip_receipt)
        if not pip_ok:
            blocker = f"ENV_UV_BOOTSTRAP_FAILED: pip install exited {pip_result.returncode}"
            summary = _receipt_from_evidence(
                probe="uv_bootstrap",
                evidence=pip_result,
                evidence_ref="env/bootstrap/uv/0",
                success=False,
                blocker_code="ENV_UV_BOOTSTRAP_FAILED",
            )
            _atomic_write_json(
                _uv_bootstrap_marker(cache_dir),
                _bootstrap_payload(
                    status="failed",
                    receipt=summary,
                    receipts=receipts,
                    blocker=blocker,
                ),
            )
            return UVBootstrapResult(False, None, blocker, (summary, *receipts))

        remaining = _remaining(deadline)
        if remaining is None:
            blocker = "ENV_UV_BOOTSTRAP_FAILED: timeout expired before Python 3.11 install"
            summary = _receipt_blocker(
                probe="uv_bootstrap",
                blocker=blocker,
                returncode=124,
                timed_out=True,
            )
            _atomic_write_json(
                _uv_bootstrap_marker(cache_dir),
                _bootstrap_payload(
                    status="failed",
                    receipt=summary,
                    receipts=receipts,
                    blocker=blocker,
                ),
            )
            return UVBootstrapResult(False, None, blocker, (summary, *receipts))
        python_result = _run_command(UV_BOOTSTRAP_PYTHON_ARGS, timeout=remaining)
        python_ok = python_result.returncode == 0 and not python_result.timed_out
        python_receipt = _receipt_from_evidence(
            probe="uv_bootstrap_python",
            evidence=python_result,
            evidence_ref="env/bootstrap/uv_python/0",
            success=python_ok,
            blocker_code="ENV_UV_BOOTSTRAP_FAILED" if not python_ok else None,
        )
        receipts.append(python_receipt)
        if not python_ok:
            blocker = (
                "ENV_UV_BOOTSTRAP_FAILED: uv python install 3.11 exited "
                f"{python_result.returncode}"
            )
            summary = _receipt_from_evidence(
                probe="uv_bootstrap",
                evidence=python_result,
                evidence_ref="env/bootstrap/uv/0",
                success=False,
                blocker_code="ENV_UV_BOOTSTRAP_FAILED",
            )
            _atomic_write_json(
                _uv_bootstrap_marker(cache_dir),
                _bootstrap_payload(
                    status="failed",
                    receipt=summary,
                    receipts=receipts,
                    blocker=blocker,
                ),
            )
            return UVBootstrapResult(False, None, blocker, (summary, *receipts))

        combined = CommandEvidence(
            args=UV_BOOTSTRAP_PYTHON_ARGS,
            returncode=0,
            stdout=f"{pip_result.output}\n{python_result.output}".strip(),
            stderr="",
        )
        summary = _receipt_from_evidence(
            probe="uv_bootstrap",
            evidence=combined,
            evidence_ref="env/bootstrap/uv/0",
            success=True,
        )
        _atomic_write_json(
            _uv_bootstrap_marker(cache_dir),
            _bootstrap_payload(status="ready", receipt=summary, receipts=receipts),
        )
        uv = shutil.which("uv")
        return UVBootstrapResult(
            ready=True,
            executable=Path(uv).resolve() if uv else Path("uv"),
            blocker=None,
            receipts=(summary, *receipts),
        )
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def _read_marker(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_failure_marker(
    env_dir: Path,
    *,
    instance_id: str,
    blocker: str,
    backend: str | None,
    requested_version: str | None,
    venv: CommandEvidence | None,
    install: CommandEvidence | None,
    manifest_digest: str | None = None,
    receipts: Sequence[EnvironmentReceipt] = (),
    repair_contract: RepairContract | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": "swebench-env/1",
        "status": "failed",
        "instance_id": instance_id,
        "blocker": blocker,
        "backend": backend,
        "requested_version": requested_version,
        "venv": venv.as_dict() if venv else None,
        "install": install.as_dict() if install else None,
        "manifest_digest": manifest_digest,
        "receipts": [receipt.as_dict() for receipt in receipts],
        "repair_contract_id": (
            repair_contract.contract_id if repair_contract else None
        ),
    }
    try:
        env_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(env_dir / _FAILED_MARKER, payload)
    except OSError:
        return


def _failed_environment(
    instance_id: str,
    env_dir: Path,
    blocker: str,
    *,
    backend: str,
    requested_version: str,
    python: Path | None,
    venv: CommandEvidence | None,
    install: CommandEvidence | None,
    manifest_digest: str | None = None,
    receipts: Sequence[EnvironmentReceipt] = (),
    repair_contract: RepairContract | None = None,
) -> EnvironmentResult:
    persisted_receipts = _persist_environment_receipts(
        env_dir,
        instance_id=instance_id,
        receipts=receipts,
        ready=False,
        blocker=blocker,
    )
    _write_failure_marker(
        env_dir,
        instance_id=instance_id,
        blocker=blocker,
        backend=backend,
        requested_version=requested_version,
        venv=venv,
        install=install,
        manifest_digest=manifest_digest,
        receipts=persisted_receipts,
        repair_contract=repair_contract,
    )
    return EnvironmentResult(
        instance_id,
        env_dir,
        python,
        False,
        blocker=blocker,
        backend=backend,
        requested_version=requested_version,
        venv=venv,
        install=install,
        receipts=persisted_receipts,
        repair_contract=repair_contract,
    )


class EnvironmentManager:
    """Create and cache one isolated environment at ``<cache>/envs/<instance_id>``."""

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        _validate_timeout(timeout)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.timeout = timeout

    def _cached_result(
        self,
        instance_id: str,
        env_dir: Path,
        *,
        manifest_digest: str | None = None,
    ) -> EnvironmentResult | None:
        marker = _read_marker(env_dir / _READY_MARKER)
        python = _venv_python(env_dir)
        if (
            marker is None
            or marker.get("status") != "ready"
            or marker.get("instance_id") != instance_id
            or python is None
        ):
            return None
        marker_digest = marker.get("manifest_digest")
        if (
            manifest_digest is not None
            and isinstance(marker_digest, str)
            and marker_digest != manifest_digest
        ):
            return EnvironmentResult(
                instance_id=instance_id,
                environment_dir=env_dir,
                python=python,
                install_ok=False,
                blocker="ENV_MANIFEST_CHANGED: ready environment belongs to another manifest",
                receipts=_read_environment_receipts(env_dir),
                repair_contract=_read_repair_contract(env_dir),
            )
        backend_value = marker.get("backend")
        requested_value = marker.get("requested_version")
        return EnvironmentResult(
            instance_id=instance_id,
            environment_dir=env_dir,
            python=python,
            install_ok=True,
            backend=backend_value if isinstance(backend_value, str) else None,
            requested_version=(
                requested_value if isinstance(requested_value, str) else None
            ),
            reused=True,
            receipts=_read_environment_receipts(env_dir),
            repair_contract=_read_repair_contract(env_dir),
        )

    def _failed_result(
        self,
        instance_id: str,
        env_dir: Path,
        *,
        manifest_digest: str | None = None,
    ) -> EnvironmentResult | None:
        marker = _read_marker(env_dir / _FAILED_MARKER)
        if marker is None or marker.get("instance_id") != instance_id:
            return None
        blocker = marker.get("blocker")
        if not isinstance(blocker, str) or not blocker:
            return None
        marker_digest = marker.get("manifest_digest")
        if (
            manifest_digest is not None
            and isinstance(marker_digest, str)
            and marker_digest != manifest_digest
        ):
            blocker = "ENV_MANIFEST_CHANGED: failed environment belongs to another manifest"
        backend = marker.get("backend")
        requested = marker.get("requested_version")
        python = _venv_python(env_dir)
        return EnvironmentResult(
            instance_id=instance_id,
            environment_dir=env_dir,
            python=python,
            install_ok=False,
            blocker=blocker,
            backend=backend if isinstance(backend, str) else None,
            requested_version=requested if isinstance(requested, str) else None,
            receipts=_read_environment_receipts(env_dir),
            repair_contract=_read_repair_contract(env_dir),
        )

    def prepare(
        self,
        instance_id: str,
        repo_dir: Path,
        *,
        repo: str | None = None,
    ) -> EnvironmentResult:
        """Prepare an environment, returning a named blocker instead of hiding failure."""
        safe_id = _safe_instance_id(instance_id)
        repository = Path(repo_dir).expanduser().resolve()
        env_dir = environment_dir(self.cache_dir, safe_id)
        if not repository.is_dir():
            return EnvironmentResult(
                safe_id,
                env_dir,
                None,
                False,
                blocker=f"ENV_REPOSITORY_NOT_FOUND: {repository}",
            )

        try:
            manifest_digest, manifest_names = _dependency_manifest_digest(repository)
        except OSError as exc:
            return EnvironmentResult(
                safe_id,
                env_dir,
                None,
                False,
                blocker=f"ENV_MANIFEST_READ_FAILED: {type(exc).__name__}: {exc}",
            )

        cached = (
            self._cached_result(
                safe_id,
                env_dir,
                manifest_digest=manifest_digest,
            )
            if env_dir.is_dir()
            else None
        )
        if cached is not None:
            return cached
        failed = (
            self._failed_result(
                safe_id,
                env_dir,
                manifest_digest=manifest_digest,
            )
            if env_dir.is_dir()
            else None
        )
        if failed is not None:
            return failed
        if env_dir.exists() and not env_dir.is_dir():
            return EnvironmentResult(
                safe_id,
                env_dir,
                None,
                False,
                blocker=f"ENV_PATH_CONFLICT: environment path is not a directory: {env_dir}",
            )
        if env_dir.is_dir() and (env_dir / _CREATING_MARKER).exists():
            return EnvironmentResult(
                safe_id,
                env_dir,
                None,
                False,
                blocker="ENV_SETUP_IN_PROGRESS: another setup owns the per-instance directory",
            )

        existing_contract = _read_repair_contract(env_dir)
        if existing_contract is not None:
            expected_key = f"{safe_id}:{manifest_digest}"
            stored_key = existing_contract.repair.get("idempotency_key")
            if (
                existing_contract.instance_id != safe_id
                or existing_contract.manifest_digest != manifest_digest
                or stored_key != expected_key
            ):
                return EnvironmentResult(
                    safe_id,
                    env_dir,
                    _venv_python(env_dir),
                    False,
                    blocker="ENV_REPAIR_CONTRACT_MISMATCH: persisted contract is for another manifest",
                    receipts=existing_contract.receipts,
                    repair_contract=existing_contract,
                )
            if existing_contract.status == "blocked":
                blocker_value = existing_contract.failure.get("blocker_code")
                blocker = (
                    blocker_value
                    if isinstance(blocker_value, str)
                    else "ENV_REPAIR_CONTRACT_BLOCKED"
                )
                return EnvironmentResult(
                    safe_id,
                    env_dir,
                    _venv_python(env_dir),
                    False,
                    blocker=blocker,
                    receipts=existing_contract.receipts,
                    repair_contract=existing_contract,
                )

        selection = select_python()
        bootstrap_receipts: tuple[EnvironmentReceipt, ...] = ()
        if selection is None:
            bootstrap = _ensure_uv_bootstrap(
                self.cache_dir,
                timeout=self.timeout,
            )
            bootstrap_receipts = bootstrap.receipts
            if not bootstrap.ready:
                return EnvironmentResult(
                    safe_id,
                    env_dir,
                    None,
                    False,
                    blocker=bootstrap.blocker
                    or "ENV_UV_BOOTSTRAP_FAILED: unknown bootstrap failure",
                    backend="uv",
                    requested_version="3.11",
                    receipts=bootstrap.receipts,
                )
            selection = select_python() or PythonSelection(
                backend="uv",
                requested_version="3.11",
                executable=bootstrap.executable or Path("uv"),
            )

        contract = existing_contract or _new_repair_contract(
            safe_id,
            manifest_digest=manifest_digest,
            manifest_names=manifest_names,
            timeout=self.timeout,
            initial_receipts=bootstrap_receipts,
        )
        receipts = list(contract.receipts)
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        lock_dir = env_dir.with_name(env_dir.name + _LOCK_SUFFIX)
        try:
            lock_dir.mkdir()
        except FileExistsError:
            return EnvironmentResult(
                safe_id,
                env_dir,
                _venv_python(env_dir),
                False,
                blocker="ENV_SETUP_IN_PROGRESS: another setup owns the per-instance lock",
                backend=selection.backend,
                requested_version=selection.requested_version,
                receipts=tuple(receipts),
                repair_contract=contract,
            )

        deadline = time.monotonic() + self.timeout
        venv_result: CommandEvidence | None = None
        install_result: CommandEvidence | None = None

        def fail(
            blocker: str,
            *,
            python: Path | None = None,
            venv: CommandEvidence | None = None,
            install: CommandEvidence | None = None,
        ) -> EnvironmentResult:
            nonlocal contract
            failure = dict(contract.failure)
            failure["blocker_code"] = _blocker_code(blocker) or "ENV_SETUP_FAILED"
            contract = replace(contract, status="blocked", failure=failure)
            try:
                _write_repair_contract(env_dir, contract)
            except OSError:
                pass
            return _failed_environment(
                safe_id,
                env_dir,
                blocker,
                backend=selection.backend,
                requested_version=selection.requested_version,
                python=python,
                venv=venv,
                install=install,
                manifest_digest=manifest_digest,
                receipts=receipts,
                repair_contract=contract,
            )

        try:
            cached = (
                self._cached_result(
                    safe_id,
                    env_dir,
                    manifest_digest=manifest_digest,
                )
                if env_dir.is_dir()
                else None
            )
            if cached is not None:
                return cached
            failed = (
                self._failed_result(
                    safe_id,
                    env_dir,
                    manifest_digest=manifest_digest,
                )
                if env_dir.is_dir()
                else None
            )
            if failed is not None:
                return failed
            if env_dir.exists() and existing_contract is None:
                return EnvironmentResult(
                    safe_id,
                    env_dir,
                    None,
                    False,
                    blocker="ENV_PARTIAL_ENVIRONMENT: existing directory has no ready marker or repair contract",
                    backend=selection.backend,
                    requested_version=selection.requested_version,
                )

            _write_repair_contract(env_dir, contract)
            completed_steps = set(contract.completed_steps)
            if "venv_create" not in completed_steps:
                if env_dir.exists():
                    blocker = (
                        "ENV_PARTIAL_ENVIRONMENT: venv creation was not completed and "
                        "the destination already exists"
                    )
                    return fail(blocker)
                (lock_dir / _CREATING_MARKER).write_text(
                    f"{safe_id}\n",
                    encoding="utf-8",
                )
                remaining = _remaining(deadline)
                if remaining is None:
                    blocker = "ENV_VENV_CREATE_TIMEOUT: setup deadline expired before venv creation"
                    receipts.append(
                        _receipt_blocker(
                            probe="venv_create",
                            blocker=blocker,
                            returncode=124,
                            timed_out=True,
                        )
                    )
                    return fail(blocker)
                if selection.backend == "uv":
                    venv_args: Sequence[str | Path] = (
                        selection.executable,
                        "venv",
                        "--python",
                        selection.requested_version,
                        "--seed",
                        env_dir,
                    )
                else:
                    venv_args = (selection.executable, "-m", "venv", env_dir)
                venv_result = _run_command(venv_args, cwd=repository, timeout=remaining)
                venv_ok = venv_result.returncode == 0 and not venv_result.timed_out
                receipts.append(
                    _receipt_from_evidence(
                        probe="venv_create",
                        evidence=venv_result,
                        evidence_ref=f"env/{safe_id}/venv_create/0",
                        success=venv_ok,
                        blocker_code="ENV_VENV_CREATE_FAILED" if not venv_ok else None,
                    )
                )
                contract = _contract_with_observation(
                    contract,
                    receipts[-1],
                    completed_step="venv_create" if venv_ok else None,
                    status=None if venv_ok else "blocked",
                )
                _write_repair_contract(env_dir, contract)
                if not venv_ok:
                    return fail(
                        _venv_blocker("VENV_CREATE", venv_result),
                        venv=venv_result,
                    )

            python = _venv_python(env_dir)
            if python is None:
                blocker = "ENV_VENV_MISSING_PYTHON: venv completed without a Python executable"
                return fail(blocker, venv=venv_result)

            completed_steps = set(contract.completed_steps)
            if "python_version" not in completed_steps:
                remaining = _remaining(deadline)
                if remaining is None:
                    blocker = "ENV_INSTALL_TIMEOUT: setup deadline expired before version probe"
                    receipts.append(
                        _receipt_blocker(
                            probe="python_version",
                            blocker=blocker,
                            returncode=124,
                            timed_out=True,
                        )
                    )
                    return fail(blocker, python=python, venv=venv_result)
                actual_version, version_evidence = _probe_python_version_evidence(
                    python,
                    timeout=min(30, remaining),
                )
                version_ok = (
                    version_evidence.returncode == 0
                    and not version_evidence.timed_out
                    and actual_version == selection.requested_version
                )
                version_blocker = (
                    "ENV_WRONG_PYTHON"
                    if actual_version != selection.requested_version
                    else None
                )
                receipts.append(
                    _receipt_from_evidence(
                        probe="python_version",
                        evidence=version_evidence,
                        evidence_ref=f"env/{safe_id}/python_version/0",
                        success=version_ok,
                        blocker_code=version_blocker,
                    )
                )
                contract = _contract_with_observation(
                    contract,
                    receipts[-1],
                    completed_step="python_version" if version_ok else None,
                    status=None if version_ok else "blocked",
                )
                _write_repair_contract(env_dir, contract)
                if not version_ok:
                    blocker = (
                        "ENV_WRONG_PYTHON: requested "
                        f"{selection.requested_version}, venv reports {actual_version or 'unknown'}"
                    )
                    return fail(blocker, python=python, venv=venv_result)

            completed_steps = set(contract.completed_steps)
            if "install_dependencies" not in completed_steps:
                remaining = _remaining(deadline)
                if remaining is None:
                    blocker = "ENV_INSTALL_TIMEOUT: setup deadline expired before installation"
                    receipts.append(
                        _receipt_blocker(
                            probe="install_dependencies",
                            blocker=blocker,
                            returncode=124,
                            timed_out=True,
                        )
                    )
                    return fail(blocker, python=python, venv=venv_result)
                install_result = _run_command(
                    [python, "-m", "pip", "install", "-e", ".", "pytest"],
                    cwd=repository,
                    timeout=remaining,
                )
                install_ok = (
                    install_result.returncode == 0 and not install_result.timed_out
                )
                blocker = (
                    None
                    if install_ok
                    else _install_blocker(
                        repo=repo,
                        repo_dir=repository,
                        result=install_result,
                    )
                )
                receipts.append(
                    _receipt_from_evidence(
                        probe="install_dependencies",
                        evidence=install_result,
                        evidence_ref=f"env/{safe_id}/install_dependencies/0",
                        success=install_ok,
                        blocker_code=_blocker_code(blocker),
                    )
                )
                contract = _contract_with_observation(
                    contract,
                    receipts[-1],
                    completed_step="install_dependencies" if install_ok else None,
                    status="completed" if install_ok else "blocked",
                )
                _write_repair_contract(env_dir, contract)
                if not install_ok:
                    return fail(
                        blocker or "ENV_INSTALL_FAILED: dependency installation failed",
                        python=python,
                        venv=venv_result,
                        install=install_result,
                    )
            else:
                contract = replace(contract, status="completed")
                _write_repair_contract(env_dir, contract)

            persisted_receipts = _persist_environment_receipts(
                env_dir,
                instance_id=safe_id,
                receipts=receipts,
                ready=True,
                blocker=None,
            )
            _atomic_write_json(
                env_dir / _READY_MARKER,
                {
                    "schema": "swebench-env/1",
                    "status": "ready",
                    "instance_id": safe_id,
                    "backend": selection.backend,
                    "requested_version": selection.requested_version,
                    "python": str(python),
                    "repository": str(repository),
                    "manifest_digest": manifest_digest,
                    "repair_contract_id": contract.contract_id,
                    "receipts": [receipt.as_dict() for receipt in persisted_receipts],
                },
            )
            (env_dir / _CREATING_MARKER).unlink(missing_ok=True)
            return EnvironmentResult(
                safe_id,
                env_dir,
                python,
                True,
                backend=selection.backend,
                requested_version=selection.requested_version,
                venv=venv_result,
                install=install_result,
                receipts=persisted_receipts,
                repair_contract=contract,
            )
        except OSError as exc:
            blocker = f"ENV_FILESYSTEM_FAILED: {type(exc).__name__}: {exc}"
            if env_dir.is_dir():
                return fail(
                    blocker,
                    python=_venv_python(env_dir),
                    venv=venv_result,
                    install=install_result,
                )
            return EnvironmentResult(
                safe_id,
                env_dir,
                None,
                False,
                blocker=blocker,
                backend=selection.backend,
                requested_version=selection.requested_version,
                receipts=tuple(receipts),
                repair_contract=contract,
            )
        finally:
            shutil.rmtree(lock_dir, ignore_errors=True)


def prepare_environment(
    instance_id: str,
    repo_dir: Path | str,
    *,
    cache_dir: Path | str,
    repo: str | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> EnvironmentResult:
    """Convenience wrapper for one instance's isolated environment."""
    return EnvironmentManager(cache_dir, timeout=timeout).prepare(
        instance_id,
        Path(repo_dir),
        repo=repo,
    )


class WorkspaceLookupError(RuntimeError):
    """The existing materialized cache cannot be mapped to one row safely."""


def _candidate_workspace(value: object) -> Path | None:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if path.name == "base":
            path = path.parent
        if (path / "base").is_dir() and (path / "gold").is_dir():
            return path
        return None
    base = getattr(value, "base_dir", None)
    gold = getattr(value, "gold_dir", None)
    if (
        isinstance(base, Path)
        and isinstance(gold, Path)
        and base.is_dir()
        and gold.is_dir()
    ):
        if base.parent == gold.parent:
            return base.parent
    return None


def _git_head(path: Path) -> str:
    git = shutil.which("git")
    if not git:
        raise WorkspaceLookupError(
            "ENV_GIT_UNAVAILABLE: git is required to scan worktrees"
        )
    result = _run_command([git, "-C", path, "rev-parse", "HEAD"], timeout=30)
    if result.returncode != 0 or result.timed_out:
        raise WorkspaceLookupError(
            f"ENV_WORKSPACE_LOOKUP_FAILED: cannot read HEAD for {path}: {result.tail}"
        )
    return result.stdout.strip()


def find_workspace(cache_dir: Path | str, row: Mapping[str, object]) -> Path:
    """Find one materialized ``base``/``gold`` pair by row SHA, failing on ambiguity."""
    for key in ("workspace", "_workspace"):
        candidate = _candidate_workspace(row.get(key))
        if candidate is not None:
            return candidate

    cache = Path(cache_dir).expanduser().resolve()
    cache_key = row.get("cache_key")
    if isinstance(cache_key, str) and cache_key:
        candidate = _candidate_workspace(cache / "instances" / cache_key)
        if candidate is not None:
            return candidate

    commit_value = row.get("base_commit", row.get("sha"))
    if not isinstance(commit_value, str) or not commit_value:
        raise WorkspaceLookupError(
            "ENV_WORKSPACE_ROW_MISSING_COMMIT: row has no base_commit/sha"
        )
    instances_dir = cache / "instances"
    if not instances_dir.is_dir():
        raise WorkspaceLookupError(f"ENV_WORKSPACE_CACHE_MISSING: {instances_dir}")
    matches: list[Path] = []
    for candidate in sorted(instances_dir.iterdir()):
        if not candidate.is_dir() or not (candidate / "base").is_dir():
            continue
        if not (candidate / "gold").is_dir():
            continue
        try:
            head = _git_head(candidate / "base")
        except WorkspaceLookupError:
            continue
        if head == commit_value or head.startswith(commit_value):
            matches.append(candidate)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise WorkspaceLookupError(
            f"ENV_WORKSPACE_NOT_FOUND: no materialized worktree at {instances_dir} for {commit_value}"
        )
    raise WorkspaceLookupError(
        f"ENV_WORKSPACE_AMBIGUOUS: {len(matches)} worktrees match base commit {commit_value}"
    )


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
