"""Bounded, passive setup/check discovery from repository declarations.

Discovery reads manifests and records candidates.  It never executes a script,
invokes a package manager, imports project code, or upgrades a declaration into
host evidence.
"""
from __future__ import annotations

import configparser
import hashlib
import json
import shlex
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kernel_environment_recipes import RecipeCandidate, RecipeRegistry, SetupRecipe
from kernel_environment_types import (
    Provenance,
    RequirementBinding,
    ResourcePredicate,
    stable_digest,
)
from kernel_product_manifest import build_product_manifest
from kernel_runner_receipts import CheckDescriptor


MAX_DISCOVERY_FILES = 256
MAX_DISCOVERY_BYTES = 2 * 1024 * 1024
LOCK_NAMES = ("package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml", "yarn.lock")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _read(root: Path, relative: str, *, max_bytes: int = MAX_DISCOVERY_BYTES) -> tuple[bytes | None, str | None]:
    path = root / relative
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None, "missing-or-over-budget"
        return path.read_bytes(), None
    except (OSError, UnicodeError) as error:
        return None, type(error).__name__


def _provenance(relative: str, raw: bytes) -> Provenance:
    return Provenance(
        kind="repo_declared",
        producer_id="kernel_environment_discovery",
        producer_version="1",
        source_ref=relative,
        source_digest=_digest_bytes(raw),
    )


def _candidate(
    *, recipe_id: str, version: str, source_ref: str, raw: bytes,
    command_ref: str, input_refs: Sequence[str], ensures: Sequence[str],
    requires: Sequence[str] = (), output_facets: Sequence[str] = (),
    idempotency: str = "unknown", coverage: str = "complete",
) -> RecipeCandidate:
    recipe = SetupRecipe(
        recipe_id=recipe_id,
        version=version,
        environment_scope="repository",
        provenance=(_provenance(source_ref, raw),),
        command_descriptor_ref=command_ref,
        input_refs=tuple(input_refs),
        output_resource_facets=tuple(output_facets),
        requires=tuple(requires),
        ensures=tuple(ensures),
        effects={"read": tuple(input_refs), "write": tuple(output_facets)},
        idempotency=idempotency,
        source_digest=_digest_bytes(raw),
        coverage=coverage,
    )
    return RecipeCandidate(recipe, source_ref, _digest_bytes(raw))


def _json_object(raw: bytes) -> Mapping[str, object] | None:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _lockfile(root: Path) -> tuple[str, bytes] | None:
    for name in LOCK_NAMES:
        raw, _error = _read(root, name)
        if raw is not None:
            return name, raw
    return None


def lockfile_digest(root: str | Path) -> str | None:
    found = _lockfile(Path(root).resolve())
    return _digest_bytes(found[1]) if found else None


def migration_set_digest(root: str | Path, *, directory: str = "migrations") -> str | None:
    root_path = Path(root).resolve()
    base = root_path / directory
    if not base.is_dir():
        return None
    rows: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*")):
        if len(rows) >= MAX_DISCOVERY_FILES:
            break
        if not path.is_file() or path.suffix.casefold() not in {".sql", ".py"}:
            continue
        try:
            relative = path.relative_to(root_path).as_posix()
            rows.append((relative, _digest_bytes(path.read_bytes())))
        except (OSError, UnicodeError):
            continue
    return stable_digest(rows) if rows else None


def service_code_digest(root: str | Path, paths: Sequence[str]) -> str:
    root_path = Path(root).resolve()
    rows: list[tuple[str, str]] = []
    for relative in sorted(set(paths)):
        raw, error = _read(root_path, relative)
        if raw is None:
            rows.append((relative, "unavailable:" + str(error)))
        else:
            rows.append((relative, _digest_bytes(raw)))
    return stable_digest(rows)


def dependency_predicate(root: str | Path, *, resource_ref: str = "dependencies") -> ResourcePredicate:
    digest = lockfile_digest(root)
    if digest is None:
        raise ValueError("a lockfile is required for a dependency parity predicate")
    return ResourcePredicate(
        predicate_id="dependencies.match.lockfile",
        resource_ref=resource_ref,
        facet="installed_for_lockfile",
        operator="equals",
        expected=digest,
        verifier_descriptor_ref="verify:dependency-inventory",
    )


def migration_predicate(root: str | Path, *, resource_ref: str = "database") -> ResourcePredicate:
    digest = migration_set_digest(root)
    if digest is None:
        raise ValueError("a migration directory with supported files is required")
    return ResourcePredicate(
        predicate_id="database.migrations.match",
        resource_ref=resource_ref,
        facet="applied_migration_set",
        operator="equals",
        expected=digest,
        verifier_descriptor_ref="verify:migration-set",
    )


def service_loaded_code_predicate(
    root: str | Path, paths: Sequence[str], *, resource_ref: str = "service:dev-server",
) -> ResourcePredicate:
    return ResourcePredicate(
        predicate_id="service.loaded_code.match",
        resource_ref=resource_ref,
        facet="loaded_code_revision",
        operator="equals",
        expected=service_code_digest(root, paths),
        freshness_policy="live_at_acceptance",
        verifier_descriptor_ref="verify:service-loaded-code",
    )


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    root: str
    recipes: tuple[RecipeCandidate, ...]
    checks: tuple[CheckDescriptor, ...]
    predicates: tuple[ResourcePredicate, ...]
    bindings: tuple[RequirementBinding, ...]
    coverage: str
    source_files: tuple[str, ...]
    omissions: tuple[str, ...] = ()

    @property
    def registry(self) -> RecipeRegistry:
        return RecipeRegistry(candidate.recipe for candidate in self.recipes)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "stateframe-environment-discovery/1",
            "root": self.root,
            "recipes": [candidate.recipe.to_dict() for candidate in self.recipes],
            "checks": [
                {"check_id": item.check_id, "kind": item.kind, "runner": item.runner,
                 "argv": list(item.argv), "adapter": item.adapter}
                for item in self.checks
            ],
            "predicates": [item.to_dict() for item in self.predicates],
            "bindings": [item.to_dict() for item in self.bindings],
            "coverage": self.coverage,
            "source_files": list(self.source_files),
            "omissions": list(self.omissions),
        }


def _node_discovery(root: Path, raw: bytes, source: str) -> tuple[list[RecipeCandidate], list[CheckDescriptor], list[ResourcePredicate]]:
    payload = _json_object(raw)
    if payload is None:
        return [], [], []
    scripts = payload.get("scripts")
    script_map = scripts if isinstance(scripts, Mapping) else {}
    lock = _lockfile(root)
    lock_ref = lock[0] if lock else None
    inputs = (source, lock_ref) if lock_ref else (source,)
    candidates: list[RecipeCandidate] = []
    checks: list[CheckDescriptor] = []
    predicates: list[ResourcePredicate] = []
    dependency_requirement: ResourcePredicate | None = None
    if lock_ref:
        lock_recipe = _candidate(
            recipe_id="node.dependencies.install",
            version="1",
            source_ref=source,
            raw=raw,
            command_ref="manifest:node:install",
            input_refs=inputs,
            ensures=("dependencies.node.installed",),
            output_facets=("dependencies.installed_for_lockfile",),
            idempotency="convergent",
        )
        candidates.append(lock_recipe)
        dependency_requirement = dependency_predicate(root, resource_ref="dependencies:node")
        predicates.append(dependency_requirement)
    service_requirement: ResourcePredicate | None = None
    for name in sorted(str(key) for key in script_map):
        command = script_map.get(name)
        if not isinstance(command, str) or not command.strip():
            continue
        recipe_id = f"node.script.{name}"
        candidates.append(_candidate(
            recipe_id=recipe_id,
            version="1",
            source_ref=source,
            raw=raw,
            command_ref=f"manifest:{source}:scripts:{name}",
            input_refs=inputs,
            requires=("dependencies.node.installed",) if lock_ref else (),
            ensures=(f"script.node.{name}.declared",),
            idempotency="unknown",
        ))
        if name in {"test", "check", "verify"}:
            checks.append(CheckDescriptor(
                check_id=f"node:{name}", runner="npm", argv=("npm", "run", name),
                kind="TEST", adapter="tap", timeout=300.0,
            ))
        if name in {"dev", "start", "serve"}:
            try:
                tokens = tuple(shlex.split(command))
            except ValueError:
                tokens = ()
            source_paths = tuple(
                token for token in tokens[1:]
                if token.endswith((".mjs", ".js", ".cjs")) and (root / token).is_file()
            )
            if source_paths:
                service_requirement = service_loaded_code_predicate(
                    root, source_paths, resource_ref="service:dev-server"
                )
                predicates.append(service_requirement)
    if checks:
        requirements = tuple(item for item in (dependency_requirement, service_requirement) if item is not None)
        checks = [
            CheckDescriptor(
                check_id=check.check_id, runner=check.runner, argv=check.argv, cwd=check.cwd,
                kind=check.kind, adapter=check.adapter, timeout=check.timeout, env=check.env,
                environment_predicates=requirements,
            )
            for check in checks
        ]
    return candidates, checks, predicates


def _python_discovery(root: Path, pyproject_raw: bytes | None, tox_raw: bytes | None) -> tuple[list[RecipeCandidate], list[CheckDescriptor], list[ResourcePredicate]]:
    candidates: list[RecipeCandidate] = []
    checks: list[CheckDescriptor] = []
    predicates: list[ResourcePredicate] = []
    if pyproject_raw is not None:
        try:
            payload = tomllib.loads(pyproject_raw.decode("utf-8"))
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            payload = {}
        build_system = payload.get("build-system") if isinstance(payload, Mapping) else None
        if isinstance(build_system, Mapping):
            candidates.append(_candidate(
                recipe_id="python.dependencies.install",
                version="1",
                source_ref="pyproject.toml",
                raw=pyproject_raw,
                command_ref="manifest:python:install",
                input_refs=("pyproject.toml",),
                ensures=("dependencies.python.installed",),
                output_facets=("dependencies.installed_for_lockfile",),
                idempotency="convergent",
            ))
    if tox_raw is None:
        return candidates, checks, predicates
    parser = configparser.ConfigParser(interpolation=None, delimiters=("="))
    try:
        parser.read_string(tox_raw.decode("utf-8"))
    except (configparser.Error, UnicodeDecodeError):
        return candidates, checks, predicates
    envlist = parser.get("tox", "envlist", fallback="py")
    environments = tuple(item.strip() for item in envlist.replace("\n", ",").split(",") if item.strip())
    for env_name in environments:
        section = "testenv:" + env_name
        deps = parser.get(section, "deps", fallback=parser.get("testenv", "deps", fallback=""))
        commands = parser.get(section, "commands", fallback=parser.get("testenv", "commands", fallback="pytest"))
        candidate = _candidate(
            recipe_id=f"python.tox.{env_name}",
            version="1",
            source_ref="tox.ini",
            raw=tox_raw,
            command_ref=f"manifest:tox:{env_name}",
            input_refs=("tox.ini", "pyproject.toml"),
            ensures=(f"tox.{env_name}.ready",),
            output_facets=(f"dependencies:python:{env_name}",),
            idempotency="convergent",
        )
        candidates.append(candidate)
        checks.append(CheckDescriptor(
            check_id=f"tox:{env_name}", runner="tox", argv=("tox", "-e", env_name),
            kind="TEST", adapter="pytest", timeout=300.0,
        ))
        if deps.strip() or commands.strip():
            # Keep the declaration visible to callers without treating it as proof.
            predicates.append(ResourcePredicate(
                predicate_id=f"tox.{env_name}.declared",
                resource_ref=f"tox:{env_name}", facet="declared_dependencies",
                operator="equals", expected=stable_digest({"deps": deps, "commands": commands}),
                verifier_descriptor_ref=f"verify:tox:{env_name}",
            ))
    return candidates, checks, predicates


def discover_repository(root: str | Path, *, max_files: int = MAX_DISCOVERY_FILES,
                        max_bytes: int = MAX_DISCOVERY_BYTES) -> DiscoveryReport:
    """Discover static candidates with explicit partial-coverage reporting."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    source_files: list[str] = []
    omissions: list[str] = []
    recipes: list[RecipeCandidate] = []
    checks: list[CheckDescriptor] = []
    predicates: list[ResourcePredicate] = []
    package_raw, package_error = _read(root_path, "package.json", max_bytes=max_bytes)
    if package_raw is not None:
        source_files.append("package.json")
        node_recipes, node_checks, node_predicates = _node_discovery(root_path, package_raw, "package.json")
        recipes.extend(node_recipes)
        checks.extend(node_checks)
        predicates.extend(node_predicates)
    elif package_error != "missing-or-over-budget":
        omissions.append("package.json:" + str(package_error))
    pyproject_raw, _ = _read(root_path, "pyproject.toml", max_bytes=max_bytes)
    tox_raw, _ = _read(root_path, "tox.ini", max_bytes=max_bytes)
    if pyproject_raw is not None:
        source_files.append("pyproject.toml")
    if tox_raw is not None:
        source_files.append("tox.ini")
    py_recipes, py_checks, py_predicates = _python_discovery(root_path, pyproject_raw, tox_raw)
    recipes.extend(py_recipes)
    checks.extend(py_checks)
    predicates.extend(py_predicates)
    migration_digest = migration_set_digest(root_path)
    if migration_digest is not None:
        source_files.append("migrations/")
        migration_raw = json.dumps({"migration_digest": migration_digest}, sort_keys=True).encode()
        recipes.append(_candidate(
            recipe_id="database.migrations.verify",
            version="1",
            source_ref="migrations/",
            raw=migration_raw,
            command_ref="manifest:migrations:verify",
            input_refs=("migrations/",),
            ensures=("database.migrations.current",),
            output_facets=("database.applied_migration_set",),
            idempotency="pure_probe",
        ))
        predicates.append(migration_predicate(root_path))
    migration_requirement = next(
        (item for item in predicates if item.predicate_id == "database.migrations.match"), None
    )
    if migration_requirement is not None:
        checks = [
            CheckDescriptor(
                check_id=check.check_id, runner=check.runner, argv=check.argv, cwd=check.cwd,
                kind=check.kind, adapter=check.adapter, timeout=check.timeout, env=check.env,
                environment_predicates=(
                    (*check.environment_predicates, migration_requirement)
                    if check.check_id.startswith("node:") else check.environment_predicates
                ),
                binding_mode=check.binding_mode,
            )
            for check in checks
        ]
    try:
        manifest = build_product_manifest(root_path)
    except (OSError, ValueError) as error:
        omissions.append("product-manifest:" + type(error).__name__)
        coverage = "partial"
    else:
        manifest_bytes = sum(entry.size for entry in manifest.entries)
        if len(manifest.entries) > max_files:
            omissions.append("product-manifest:file-limit")
            coverage = "partial"
        elif manifest_bytes > max_bytes:
            omissions.append("product-manifest:byte-limit")
            coverage = "partial"
        else:
            coverage = "complete"
    if len(source_files) + len(recipes) > max_files:
        omissions.append("discovery:file-limit")
        coverage = "partial"
    return DiscoveryReport(
        root=str(root_path), recipes=tuple(recipes), checks=tuple(checks),
        predicates=tuple(predicates),
        bindings=tuple(
            RequirementBinding(
                check_id=check.check_id,
                contract_version="1",
                predicate_refs=tuple(item.predicate_id for item in check.environment_predicates),
                mode=check.binding_mode,
            )
            for check in checks if check.environment_predicates
        ),
        coverage=coverage,
        source_files=tuple(dict.fromkeys(source_files)), omissions=tuple(sorted(set(omissions))),
    )


discover_repo_recipes = discover_repository
discover_setup_recipes = discover_repository


__all__ = [
    "DiscoveryReport", "LOCK_NAMES", "MAX_DISCOVERY_BYTES", "MAX_DISCOVERY_FILES",
    "dependency_predicate", "discover_repository", "discover_repo_recipes",
    "discover_setup_recipes", "lockfile_digest", "migration_predicate",
    "migration_set_digest", "service_code_digest", "service_loaded_code_predicate",
]
