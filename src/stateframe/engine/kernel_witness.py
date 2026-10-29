"""O19 host compiler for public, executable bug witnesses.

This module owns the small bridge between a public typed check and the existing
content-addressed witness runner.  It deliberately accepts no test source,
patch, oracle, or runtime object from a model.  REPROSPEC remains the authority
for typed setup/observation validation and WITNESS RUNNER remains the authority
for isolation, budgets, receipts, and replay.
"""

from __future__ import annotations

import ast
import copy
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kernel_reprospec import (
    _compile_typed_check_spec as _compile_host_check,
    _validate_typed_check_spec as _validate_host_check,
    build_symbol_packet,
    symbol_fixture_menu,
    symbol_target_menu,
)
from witness_runner import (
    BUDGET_HALTED,
    PROBED_CODE_CRASHED,
    RETURNED,
    RUNNER_FAILED,
    RUNNER_REJECTED,
    ContentStore,
    FrozenExecutable,
    InstrumentationPlan,
    LocalWitnessRunner,
    RunBudgets,
    RunResult,
    freeze_executable,
    sha256_json,
)


SCHEMA = "o19-public-witness/1"
MAX_PUBLIC_FILES = 64
MAX_PUBLIC_SOURCE_BYTES = 512_000
MAX_BUG_REPORT_CHARS = 8_000
MAX_PARTITIONS = 16
_PARTITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PRIVATE_PATH_PARTS = frozenset({".git", "__pycache__", "build", "dist", "node_modules"})
_DRIVER = """

if __name__ == "__main__":
    _o19_suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    _o19_result = unittest.TextTestRunner(verbosity=0).run(_o19_suite)
    if _o19_result.testsRun != 1:
        raise RuntimeError("O19 witness did not execute exactly one public check")
    if _o19_result.errors:
        raise RuntimeError("O19 witness setup or execution errored")
    if _o19_result.failures:
        def _o19_public_expectation_failed():
            raise AssertionError("O19 public expectation failed")
        _o19_public_expectation_failed()
"""


class WitnessCompileError(ValueError):
    """Raised for a host-side witness contract or compilation failure."""


@dataclass(frozen=True)
class PublicCheck:
    """A public REPROSPEC proposal; no executable source is accepted."""

    requirement_ref: str
    expected_relation: str
    plausible_wrong_impl: str
    distinguishing_scenario: Mapping[str, object]
    check_spec: Mapping[str, object]

    @classmethod
    def from_mapping(cls, value: object) -> "PublicCheck":
        if not isinstance(value, Mapping):
            raise WitnessCompileError("public check must be an object")
        required = {
            "requirement_ref", "expected_relation", "plausible_wrong_impl",
            "distinguishing_scenario", "check_spec",
        }
        if set(value) != required:
            raise WitnessCompileError("public check fields are host-owned and fixed")
        scenario = value.get("distinguishing_scenario")
        spec = value.get("check_spec")
        if not isinstance(scenario, Mapping) or not isinstance(spec, Mapping):
            raise WitnessCompileError("public check scenario and check_spec must be objects")
        return cls(
            requirement_ref=str(value.get("requirement_ref") or ""),
            expected_relation=str(value.get("expected_relation") or ""),
            plausible_wrong_impl=str(value.get("plausible_wrong_impl") or ""),
            distinguishing_scenario=copy.deepcopy(dict(scenario)),
            check_spec=copy.deepcopy(dict(spec)),
        )

    def as_arguments(self) -> dict[str, object]:
        """Return exactly the existing host compiler's typed-check shape."""
        return {
            "requirement_ref": self.requirement_ref,
            "expected_relation": self.expected_relation,
            "plausible_wrong_impl": self.plausible_wrong_impl,
            "distinguishing_scenario": copy.deepcopy(dict(self.distinguishing_scenario)),
            "check_spec": copy.deepcopy(dict(self.check_spec)),
        }

    def digest(self) -> str:
        try:
            return sha256_json(self.as_arguments())
        except (TypeError, ValueError) as error:
            raise WitnessCompileError("public check contains non-JSON data") from error


@dataclass(frozen=True)
class PublicPartition:
    """One independently named repair or preservation expectation."""

    partition_id: str
    role: str
    independence_key: str
    check: PublicCheck

    @classmethod
    def from_mapping(cls, value: object) -> "PublicPartition":
        if not isinstance(value, Mapping):
            raise WitnessCompileError("partition must be an object")
        required = {"partition_id", "role", "independence_key", "check"}
        if set(value) != required:
            raise WitnessCompileError("partition fields are host-owned and fixed")
        return cls(
            partition_id=str(value.get("partition_id") or ""),
            role=str(value.get("role") or ""),
            independence_key=str(value.get("independence_key") or ""),
            check=PublicCheck.from_mapping(value.get("check")),
        )


@dataclass(frozen=True)
class WitnessRequest:
    """All public inputs needed to compile one frozen witness family."""

    bug_report: str
    target_path: str
    public_files: Mapping[str, str]
    primary: PublicCheck
    partitions: tuple[PublicPartition, ...]
    budgets: RunBudgets = RunBudgets()


@dataclass(frozen=True)
class CompiledPartition:
    partition_id: str
    role: str
    independence_key: str
    check: PublicCheck
    source: str
    executable: FrozenExecutable


@dataclass(frozen=True)
class CompiledWitness:
    schema: str
    bug_report: str
    target_path: str
    packet: Mapping[str, object]
    store: ContentStore
    budgets: RunBudgets
    primary: CompiledPartition
    partitions: tuple[CompiledPartition, ...]


@dataclass(frozen=True)
class ParentPartitionResult:
    partition_id: str
    role: str
    status: str
    result: RunResult
    detail: str


@dataclass(frozen=True)
class ParentValidation:
    primary: ParentPartitionResult
    partitions: tuple[ParentPartitionResult, ...]

    @property
    def valid(self) -> bool:
        if self.primary.status != "PARENT_FAIL":
            return False
        if not self.partitions or not any(item.role == "preserve" for item in self.partitions):
            return False
        return all(
            item.status == ("PARENT_FAIL" if item.role == "repair" else "PRESERVED")
            for item in self.partitions
        )


def _public_path(raw: object) -> str:
    if not isinstance(raw, str):
        raise WitnessCompileError("public file paths must be strings")
    path = raw.replace("\\", "/")
    parsed = PurePosixPath(path)
    parts = parsed.parts
    if (
        not path or parsed.is_absolute() or not parts or ".." in parts
        or any(part in {"", "."} for part in parts)
        or parsed.suffix != ".py"
    ):
        raise WitnessCompileError(f"unsupported public product path: {raw!r}")
    if parts[0] == "backend":
        raise WitnessCompileError("public paths must be relative to the Python import root")
    parents = parts[:-1]
    if any(
        part.startswith(".") or part.startswith("_")
        or part.lower() in _PRIVATE_PATH_PARTS
        or part.lower() == "test" or part.lower().startswith("test_")
        or part.lower().endswith("_test")
        for part in parents
    ):
        raise WitnessCompileError(f"private or test path is not public: {raw!r}")
    filename = parts[-1]
    if filename != "__init__.py" and filename.startswith("_"):
        raise WitnessCompileError(f"private product module is not public: {raw!r}")
    lower_name = filename.lower()
    if lower_name == "conftest.py" or lower_name == "test.py" \
            or lower_name.startswith("test_") or lower_name.endswith("_test.py"):
        raise WitnessCompileError(f"test material is not accepted: {raw!r}")
    return "/".join(parts)


def _normalize_public_files(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise WitnessCompileError("public_files must be a non-empty mapping")
    if len(value) > MAX_PUBLIC_FILES:
        raise WitnessCompileError("public_files exceeds the host bound")
    normalized: dict[str, str] = {}
    for raw_path, source in value.items():
        path = _public_path(raw_path)
        if not isinstance(source, str):
            raise WitnessCompileError(f"source for {path!r} must be text")
        if len(source.encode("utf-8")) > MAX_PUBLIC_SOURCE_BYTES:
            raise WitnessCompileError(f"source for {path!r} exceeds the host bound")
        if path in normalized:
            raise WitnessCompileError(f"duplicate normalized public path: {path}")
        try:
            ast.parse(source, filename=path)
        except SyntaxError as error:
            raise WitnessCompileError(f"public source does not parse: {path}: {error.msg}") from error
        normalized[path] = source
    return normalized


def _validate_request(request: WitnessRequest) -> dict[str, str]:
    if not isinstance(request, WitnessRequest):
        raise WitnessCompileError("compile_public_witness requires WitnessRequest")
    if not request.bug_report or len(request.bug_report) > MAX_BUG_REPORT_CHARS:
        raise WitnessCompileError("bug_report must be non-empty and bounded")
    target_path = _public_path(request.target_path)
    files = _normalize_public_files(request.public_files)
    if target_path not in files:
        raise WitnessCompileError("target_path is not present in public_files")
    if not isinstance(request.primary, PublicCheck):
        raise WitnessCompileError("primary must be a PublicCheck")
    if not isinstance(request.budgets, RunBudgets):
        raise WitnessCompileError("budgets must be a RunBudgets value")
    if request.budgets.wall_clock_ms <= 0 or request.budgets.startup_ms <= 0 \
            or request.budgets.teardown_ms <= 0:
        raise WitnessCompileError("execution budgets must have positive lifecycle limits")
    for name in ("stdout_bytes", "stderr_bytes", "observation_events", "observation_bytes"):
        if getattr(request.budgets, name) <= 0:
            raise WitnessCompileError(f"execution budget {name} must be positive")
    if request.budgets.cpu_ms is not None and request.budgets.cpu_ms <= 0:
        raise WitnessCompileError("cpu_ms must be positive when supplied")
    if request.budgets.memory_bytes is not None and request.budgets.memory_bytes <= 0:
        raise WitnessCompileError("memory_bytes must be positive when supplied")
    if not isinstance(request.partitions, tuple) or not request.partitions:
        raise WitnessCompileError("at least one independent partition is required")
    if len(request.partitions) > MAX_PARTITIONS:
        raise WitnessCompileError("partitions exceed the host bound")
    ids = set()
    keys = set()
    check_digests = {request.primary.digest()}
    requirement_refs = {request.primary.requirement_ref}
    has_preserve = False
    for partition in request.partitions:
        if not isinstance(partition, PublicPartition):
            raise WitnessCompileError("partitions must contain PublicPartition values")
        if not _PARTITION_ID.fullmatch(partition.partition_id):
            raise WitnessCompileError(f"invalid partition id: {partition.partition_id!r}")
        if partition.role not in {"repair", "preserve"}:
            raise WitnessCompileError(f"unsupported partition role: {partition.role!r}")
        if not partition.independence_key or partition.independence_key in keys:
            raise WitnessCompileError("partition independence keys must be non-empty and unique")
        if partition.partition_id in ids or partition.partition_id == "primary":
            raise WitnessCompileError("partition ids must be unique and cannot be primary")
        if not isinstance(partition.check, PublicCheck):
            raise WitnessCompileError("partition check must be a PublicCheck")
        digest = partition.check.digest()
        if digest in check_digests:
            raise WitnessCompileError("duplicate typed checks cannot claim independence")
        if partition.check.requirement_ref in requirement_refs:
            raise WitnessCompileError("requirement references must be unique across partitions")
        ids.add(partition.partition_id)
        keys.add(partition.independence_key)
        check_digests.add(digest)
        requirement_refs.add(partition.check.requirement_ref)
        has_preserve |= partition.role == "preserve"
    if not has_preserve:
        raise WitnessCompileError("at least one preserve partition is required")
    return files


def _packet(request: WitnessRequest, files: Mapping[str, str]) -> dict[str, object]:
    packet = build_symbol_packet(
        Path("."),
        (request.target_path,),
        target_path=request.target_path,
        source_map=files,
        known_paths=tuple(files),
    )
    if packet.get("truncated"):
        raise WitnessCompileError("public symbol packet was truncated")
    if not any(
        isinstance(module, Mapping) and module.get("source_path") == request.target_path
        for module in packet.get("modules", ())
    ):
        raise WitnessCompileError("target source was not admitted to the public symbol packet")
    return packet


def _normalized_check(check: PublicCheck, packet: Mapping[str, object]) -> tuple[dict[str, object], str]:
    arguments = check.as_arguments()
    check_spec = arguments["check_spec"]
    assert isinstance(check_spec, Mapping)
    expected = check_spec.get("expected")
    if not isinstance(expected, Mapping) or expected.get("relation") != check.expected_relation:
        raise WitnessCompileError(
            f"{check.requirement_ref}: expected_relation is not bound to check_spec.expected"
        )
    normalized, error = _validate_host_check(arguments, packet)
    if error is not None or normalized is None:
        detail = error.get("message", "typed check rejected") if error else "typed check rejected"
        raise WitnessCompileError(f"{check.requirement_ref}: {detail}")
    source, compile_error = _compile_host_check(arguments, packet)
    if compile_error is not None or source is None:
        detail = compile_error.get("message", "host compiler rejected check") if compile_error else "host compiler rejected check"
        raise WitnessCompileError(f"{check.requirement_ref}: {detail}")
    complete_source = source + _DRIVER
    if len(complete_source) > MAX_PUBLIC_SOURCE_BYTES:
        raise WitnessCompileError(f"{check.requirement_ref}: compiled witness is oversized")
    try:
        compile(complete_source, "<o19-witness>", "exec")
    except SyntaxError as error:
        raise WitnessCompileError(f"{check.requirement_ref}: compiled witness is invalid: {error.msg}") from error
    return normalized, complete_source


def _required_paths(packet: Mapping[str, object], normalized: Mapping[str, object], files: Mapping[str, str]) -> tuple[str, ...]:
    targets = symbol_target_menu(packet)
    fixtures = symbol_fixture_menu(packet)
    paths: set[str] = set()
    setup = normalized.get("setup")
    if not isinstance(setup, list):
        raise WitnessCompileError("normalized setup is not a list")
    for step in setup:
        if not isinstance(step, Mapping):
            raise WitnessCompileError("normalized setup contains a non-object")
        for field, menu in (("target", targets), ("fixture", fixtures)):
            identifier = step.get(field)
            if identifier is None:
                continue
            entry = menu.get(str(identifier))
            source_path = entry.get("source_path") if isinstance(entry, Mapping) else None
            if not isinstance(source_path, str) or source_path not in files:
                raise WitnessCompileError(f"typed binding {identifier!r} has no public source")
            paths.add(source_path)
    if not paths:
        raise WitnessCompileError("typed check has no public source binding")
    # Package initializers are not symbol targets, but their import-time behavior
    # is part of the faithful module closure and must travel with the witness.
    for path in tuple(paths):
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts)):
            initializer = "/".join((*parts[:index], "__init__.py"))
            if initializer in files:
                paths.add(initializer)
    return tuple(sorted(paths))


def _entrypoint(partition_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", partition_id).strip("_") or "partition"
    return f"__o19_witness_{safe}.py"


def _compile_partition(
        *, request: WitnessRequest, packet: Mapping[str, object], files: Mapping[str, str],
        store: ContentStore, partition_id: str, role: str, independence_key: str,
        check: PublicCheck,
) -> CompiledPartition:
    normalized, source = _normalized_check(check, packet)
    required = _required_paths(packet, normalized, files)
    entrypoint = _entrypoint(partition_id)
    artifacts = {path: files[path] for path in required}
    artifacts[entrypoint] = source
    fixture_digest = sha256_json({
        "schema": SCHEMA,
        "bug_report": request.bug_report,
        "partition_id": partition_id,
        "independence_key": independence_key,
        "check": check.as_arguments(),
        "public_files": {path: files[path] for path in sorted(files)},
    })
    executable = freeze_executable(
        store,
        artifacts,
        entrypoint,
        fixture_digest=fixture_digest,
        deterministic_seed=0,
        network_policy="disabled",
    )
    return CompiledPartition(
        partition_id=partition_id,
        role=role,
        independence_key=independence_key,
        check=check,
        source=source,
        executable=executable,
    )


def compile_public_witness(request: WitnessRequest) -> CompiledWitness:
    """Compile a bounded witness family without touching a live runtime."""
    files = _validate_request(request)
    packet = _packet(request, files)
    store = ContentStore()
    primary = _compile_partition(
        request=request, packet=packet, files=files, store=store,
        partition_id="primary", role="repair", independence_key="primary",
        check=request.primary,
    )
    partitions = tuple(
        _compile_partition(
            request=request, packet=packet, files=files, store=store,
            partition_id=item.partition_id, role=item.role,
            independence_key=item.independence_key, check=item.check,
        )
        for item in request.partitions
    )
    return CompiledWitness(
        schema=SCHEMA,
        bug_report=request.bug_report,
        target_path=request.target_path,
        packet=copy.deepcopy(dict(packet)),
        store=store,
        budgets=request.budgets,
        primary=primary,
        partitions=partitions,
    )


def _parent_result(
        compiled: CompiledPartition, result: RunResult,
) -> ParentPartitionResult:
    outcome = result.outcome
    if outcome.classification == RETURNED:
        if compiled.role == "preserve":
            return ParentPartitionResult(compiled.partition_id, compiled.role, "PRESERVED", result, "public check passed")
        return ParentPartitionResult(compiled.partition_id, compiled.role, "PARENT_PASS", result, "repair check passed on parent")
    if (
        compiled.role == "repair"
        and outcome.classification == PROBED_CODE_CRASHED
        and outcome.exception is not None
        and outcome.exception.qualified_type == "AssertionError"
        and outcome.exception.function_qualname == "_o19_public_expectation_failed"
    ):
        return ParentPartitionResult(compiled.partition_id, compiled.role, "PARENT_FAIL", result, "host assertion driver failed")
    if outcome.classification in {BUDGET_HALTED, RUNNER_FAILED, RUNNER_REJECTED}:
        detail = f"runner outcome is not product behavior: {outcome.classification}"
    elif outcome.exception is not None:
        detail = f"product/setup exception is not a public expectation failure: {outcome.exception.qualified_type}"
    else:
        detail = f"unsupported parent outcome: {outcome.classification}"
    return ParentPartitionResult(compiled.partition_id, compiled.role, "INVALID_PARENT", result, detail)


def validate_parent_failure(
        witness: CompiledWitness, *, runner: LocalWitnessRunner | None = None,
) -> ParentValidation:
    """Run each frozen partition once and require a typed parent failure."""
    if not isinstance(witness, CompiledWitness):
        raise WitnessCompileError("validate_parent_failure requires CompiledWitness")
    canonical_tmp = Path(tempfile.gettempdir()).resolve()
    active_runner = runner or LocalWitnessRunner(witness.store, workdir=str(canonical_tmp))
    plan = InstrumentationPlan.empty()
    primary = _parent_result(
        witness.primary,
        active_runner.run(witness.primary.executable, plan, witness.budgets),
    )
    partitions = tuple(
        _parent_result(
            item,
            active_runner.run(item.executable, plan, witness.budgets),
        )
        for item in witness.partitions
    )
    return ParentValidation(primary=primary, partitions=partitions)


def compile_and_validate_parent(request: WitnessRequest) -> tuple[CompiledWitness, ParentValidation]:
    """Convenience host-only operation for the O19 admission boundary."""
    witness = compile_public_witness(request)
    return witness, validate_parent_failure(witness)


__all__ = [
    "CompiledPartition", "CompiledWitness", "ParentPartitionResult",
    "ParentValidation", "PublicCheck", "PublicPartition", "SCHEMA",
    "RunBudgets", "WitnessCompileError", "WitnessRequest", "compile_and_validate_parent",
    "compile_public_witness", "validate_parent_failure",
]
