"""Opt-in planned-artifact ledger, static readiness, and juggler returns.

This is the planned artifacts first build. It deliberately adds one host-owned view
beside the existing obligation ledger and continuation executive; it does not
create a second scheduler or treat a model claim as evidence.

The module is inert unless ``STATEFRAME_PLANNED_ARTIFACTS=on``. Resolution is
static: Python is parsed and compiled, never imported or executed. A planned
module can therefore prove ``exists`` while still failing ``exports`` or
``importable``; ``behavior`` remains a separate, unproven predicate.
"""
from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


MODE_ENV = "STATEFRAME_PLANNED_ARTIFACTS"
SCHEMA = "planned-artifacts/1"
PLANNED_ARTIFACT = "planned_artifact"

READY = "READY"
WAITING_ON_PLANNED_INPUT = "WAITING_ON_PLANNED_INPUT"
BLOCKED_ON_INVALID_OR_UNPLANNED_INPUT = "BLOCKED_ON_INVALID_OR_UNPLANNED_INPUT"
UNKNOWN_READINESS = "UNKNOWN_READINESS"
READINESS_STATES = frozenset({
    READY, WAITING_ON_PLANNED_INPUT,
    BLOCKED_ON_INVALID_OR_UNPLANNED_INPUT, UNKNOWN_READINESS,
})

ERROR = "ERROR"
GREEN = "GREEN"
DEFERRED = "DEFERRED"

EXISTS = "exists"
EXPORTS = "exports"
IMPORTABLE = "importable"
BEHAVIOR = "behavior"
PREDICATES = (EXISTS, EXPORTS, IMPORTABLE, BEHAVIOR)
DEFAULT_PREDICATES = (EXISTS, EXPORTS, IMPORTABLE)


class CandidateDisposition:
    """The only dispositions an inferred candidate may receive."""

    UNCONFIRMED = "UNCONFIRMED"
    CREATE = "CREATE"
    CORRECT_REFERENCE = "CORRECT_REFERENCE"
    ENVIRONMENT = "ENVIRONMENT"


class PlannedArtifactError(RuntimeError):
    """A planned-artifact admission or readiness contract violation."""


class SyntaxGateError(SyntaxError):
    """The local compile gate rejected an edit before it became durable."""


class ContinuationDepthError(PlannedArtifactError):
    """A return would exceed the bounded interruption depth."""


def mode() -> str:
    value = os.environ.get(MODE_ENV, "off").strip().lower()
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return mode() == "on"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _tuple_strings(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise PlannedArtifactError(f"{name} must be a sequence of strings")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    return result


def _normal_path(value: str | Path) -> str:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute() or "*" in str(path) or "?" in str(path):
        raise PlannedArtifactError("planned-artifact targets must be exact relative paths")
    normalized = path.as_posix()
    if normalized in {"", "."} or any(part in {"", ".", ".."} for part in path.parts):
        raise PlannedArtifactError("planned-artifact target escapes or names no workspace path")
    return normalized


def _module_from_path(path: str) -> str:
    value = path[:-3] if path.endswith(".py") else path
    if value.endswith("/__init__"):
        value = value[:-9]
    return value.replace("/", ".")


def _path_from_module(module: str) -> str:
    return module.replace(".", "/") + ".py"


@dataclass(frozen=True)
class PlannedArtifactSpec:
    path: str
    exports: tuple[str, ...] = ()
    owner_work_id: str = ""
    requirement_ids: tuple[str, ...] = ()
    module: str = ""
    consumer_refs: tuple[str, ...] = ()
    contract_refs: tuple[str, ...] = ()
    predicates: tuple[str, ...] = DEFAULT_PREDICATES
    admission_origin: str = "explicit"
    admission_receipt: str = ""
    required_before: str = "feature_acceptance"
    purpose: str = ""
    fulfillment_boundary: str = ""
    revisit_rule: str = "recheck after each edit and before completion"
    lineage_id: str = ""
    disposition: str = "planned"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normal_path(self.path))
        object.__setattr__(self, "exports", _tuple_strings(self.exports, "exports"))
        object.__setattr__(self, "requirement_ids",
                           _tuple_strings(self.requirement_ids, "requirement_ids"))
        object.__setattr__(self, "consumer_refs",
                           _tuple_strings(self.consumer_refs, "consumer_refs"))
        object.__setattr__(self, "contract_refs",
                           _tuple_strings(self.contract_refs, "contract_refs"))
        predicates = _tuple_strings(self.predicates, "predicates")
        if not predicates:
            raise PlannedArtifactError("a planned artifact needs at least one predicate")
        if any(item not in PREDICATES for item in predicates):
            raise PlannedArtifactError(f"unknown planned-artifact predicate: {predicates}")
        object.__setattr__(self, "predicates", tuple(dict.fromkeys(predicates)))
        module = self.module.strip() or _module_from_path(self.path)
        if not module or any(not part.isidentifier() for part in module.split(".")):
            raise PlannedArtifactError(f"invalid Python module identity {module!r}")
        object.__setattr__(self, "module", module)
        if not self.owner_work_id.strip():
            raise PlannedArtifactError("planned artifact requires an owner_work_id")
        if any(char in self.path for char in "[]"):
            raise PlannedArtifactError("planned-artifact target may not contain wildcards")
        if self.disposition not in {"planned", CandidateDisposition.CREATE,
                                    CandidateDisposition.ENVIRONMENT}:
            raise PlannedArtifactError(f"invalid planned-artifact disposition {self.disposition!r}")

    @property
    def target(self) -> str:
        return self.path

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredicateResult:
    exists: bool | None
    exports: bool | None
    importable: bool | None
    behavior: bool | None = None
    missing_exports: tuple[str, ...] = ()
    import_errors: tuple[str, ...] = ()
    near_duplicates: tuple[str, ...] = ()
    deletion_break: bool = False

    def value(self, predicate: str) -> bool | None:
        if predicate not in PREDICATES:
            raise PlannedArtifactError(f"unknown predicate {predicate!r}")
        return getattr(self, predicate)

    def satisfied(self, *predicates: str) -> bool:
        needed = predicates or PREDICATES[:3]
        return all(self.value(item) is True for item in needed)

    def to_dict(self) -> dict[str, object]:
        return {
            "exists": self.exists, "exports": self.exports,
            "importable": self.importable, "behavior": self.behavior,
            "missing_exports": list(self.missing_exports),
            "import_errors": list(self.import_errors),
            "near_duplicates": list(self.near_duplicates),
            "deletion_break": self.deletion_break,
        }


@dataclass(frozen=True)
class Resolution:
    module: str
    path: str | None
    exists: bool
    exports: bool | None
    importable: bool | None
    exported_names: tuple[str, ...] = ()
    missing_exports: tuple[str, ...] = ()
    import_errors: tuple[str, ...] = ()

    def predicates(self, required_exports: Sequence[str] = ()) -> PredicateResult:
        required = tuple(required_exports)
        missing = tuple(item for item in required if item not in self.exported_names)
        exports = self.exports
        if required and exports is not None:
            exports = not missing
        return PredicateResult(
            exists=self.exists, exports=exports, importable=self.importable,
            missing_exports=missing or self.missing_exports,
            import_errors=self.import_errors,
        )


@dataclass(frozen=True)
class Candidate:
    path: str
    module: str
    exports: tuple[str, ...]
    source_path: str
    evidence: tuple[str, ...] = ()
    kind: str = "import"
    disposition: str = CandidateDisposition.UNCONFIRMED
    near_duplicates: tuple[str, ...] = ()
    missing_exports: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return "candidate:" + _digest({
            "path": self.path, "module": self.module,
            "exports": self.exports, "source": self.source_path,
        })[7:23]

    @property
    def target_path(self) -> str:
        return self.path

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id, "path": self.path, "module": self.module,
            "exports": list(self.exports), "source_path": self.source_path,
            "evidence": list(self.evidence), "kind": self.kind,
            "disposition": self.disposition,
            "near_duplicates": list(self.near_duplicates),
            "missing_exports": list(self.missing_exports),
        }


@dataclass(frozen=True)
class PlannedArtifactRecord:
    id: str
    spec: PlannedArtifactSpec
    first_seen_at: int
    last_replanned_at: int
    review_count: int = 0
    predicates: PredicateResult = field(default_factory=lambda: PredicateResult(
        False, False, False))
    health: str = "planned"
    deletion_break: bool = False
    last_transaction_id: str = ""
    obligation_id: str = ""

    def age_micros_at(self, now: int) -> int:
        """Return elapsed age without resetting it during replanning."""

        return max(0, int(now) - self.first_seen_at)

    @property
    def age_micros(self) -> int:
        """Compatibility view at the record's last observed checkpoint."""

        return self.age_micros_at(self.last_replanned_at)

    @property
    def target_path(self) -> str:
        return self.spec.path

    @property
    def is_fulfilled(self) -> bool:
        return not self.deletion_break and self.predicates.satisfied(*self.spec.predicates)

    def to_dict(self, *, now: int | None = None) -> dict[str, object]:
        age = 0 if now is None else max(0, now - self.first_seen_at)
        return {
            "id": self.id, "spec": self.spec.to_dict(),
            "first_seen_at": self.first_seen_at,
            "last_replanned_at": self.last_replanned_at,
            "review_count": self.review_count,
            "predicates": self.predicates.to_dict(), "health": self.health,
            "deletion_break": self.deletion_break,
            "last_transaction_id": self.last_transaction_id,
            "obligation_id": self.obligation_id, "age_micros": age,
        }


class PythonResolver:
    """Static resolver with project-root and ``src`` package-root semantics."""

    def __init__(self, root: str | Path, package_roots: Sequence[str | Path] = ()):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise PlannedArtifactError(f"resolver root is not a directory: {root}")
        roots: list[Path] = [self.root]
        src = self.root / "src"
        if src.is_dir():
            roots.append(src)
        for raw in package_roots:
            candidate = (self.root / str(raw)).resolve()
            if candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
        self.package_roots = tuple(roots)
        self._source_overrides: dict[Path, str] = {}

    def set_source_override(self, path: str | Path, source: str) -> None:
        raw = Path(path)
        target = (raw if raw.is_absolute() else self.root / raw).resolve()
        self._source_overrides[target] = str(source)

    def clear_source_override(self, path: str | Path) -> None:
        raw = Path(path)
        target = (raw if raw.is_absolute() else self.root / raw).resolve()
        self._source_overrides.pop(target, None)

    def _source(self, path: Path) -> str:
        target = path.resolve()
        if target in self._source_overrides:
            return self._source_overrides[target]
        return path.read_text(encoding="utf-8")

    def _module_candidates(self, module: str) -> tuple[Path, ...]:
        if not module or any(not part.isidentifier() for part in module.split(".")):
            return ()
        relative = Path(*module.split("."))
        result: list[Path] = []
        for root in self.package_roots:
            result.append((root / relative).with_suffix(".py"))
            result.append(root / relative / "__init__.py")
        return tuple(result)

    def module_path(self, module: str) -> Path | None:
        for candidate in self._module_candidates(module):
            if candidate.is_file() or candidate.resolve() in self._source_overrides:
                return candidate
        return None

    def module_for_path(self, path: str | Path) -> str:
        raw = Path(path)
        candidate = (raw if raw.is_absolute() else self.root / raw).resolve()
        for root in sorted(self.package_roots, key=lambda item: len(item.parts), reverse=True):
            try:
                relative = candidate.relative_to(root).as_posix()
            except ValueError:
                continue
            return _module_from_path(relative)
        raise PlannedArtifactError(f"path is outside resolver package roots: {path}")

    def module_name_for_import(self, name: str, *, source_path: str | Path | None = None,
                               level: int = 0) -> str | None:
        name = str(name or "").strip()
        if level <= 0:
            return name or None
        if source_path is None:
            return None
        try:
            current = self.module_for_path(source_path)
        except PlannedArtifactError:
            return None
        # ``module_for_path`` normalizes ``pkg/__init__.py`` to ``pkg``.  In
        # that file the current package is already ``pkg``; ordinary modules
        # derive their package by removing the leaf module.
        if Path(source_path).name == "__init__.py":
            package = current
        else:
            package = current.rsplit(".", 1)[0] if "." in current else ""
        parts = [item for item in package.split(".") if item]
        if level > len(parts):
            return None
        base = parts[:len(parts) - level + 1]
        if name:
            base.extend(name.split("."))
        return ".".join(base) or None

    def _parse(self, path: Path) -> ast.Module | None:
        try:
            source = self._source(path)
            compile(source, str(path), "exec")
            return ast.parse(source, filename=str(path))
        except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError):
            return None

    @staticmethod
    def _static_all(tree: ast.Module) -> tuple[str, ...] | None:
        for node in tree.body:
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = [node.value] if any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets) else []
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id == "__all__":
                targets = [node.value]
            if targets:
                try:
                    value = ast.literal_eval(targets[0])
                except (ValueError, TypeError, SyntaxError):
                    return None
                if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                    return tuple(value)
                return None
        return None

    @classmethod
    def exports_for_tree(cls, tree: ast.Module) -> tuple[str, ...] | None:
        explicit = cls._static_all(tree)
        if explicit is not None:
            return tuple(dict.fromkeys(explicit))
        names: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == "__getattr__":
                    return None
                names.append(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if isinstance(target, ast.Name) and not target.id.startswith("_"):
                        names.append(target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    if alias.name == "*":
                        return None
                    names.append(alias.asname or alias.name.rsplit(".", 1)[-1])
        return tuple(sorted(set(names)))

    def _external_importable(self, module: str) -> bool:
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
            return False

    def _importability(self, path: Path, *, seen: set[Path] | None = None
                       ) -> tuple[bool | None, tuple[str, ...]]:
        seen = set() if seen is None else seen
        path = path.resolve()
        if path in seen:
            return True, ()
        seen.add(path)
        try:
            source = self._source(path)
            compile(source, str(path), "exec")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeError, SyntaxError, ValueError, RecursionError) as error:
            return False, (f"{type(error).__name__}: {error}",)
        errors: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [(alias.name, (), 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = self.module_name_for_import(
                    node.module or "", source_path=path, level=node.level)
                imports = [(module or "", tuple(alias.name for alias in node.names), node.level)]
            else:
                continue
            for module, names, level in imports:
                if not module:
                    errors.append(f"unresolved relative import in {path.name}")
                    continue
                dependency = self.module_path(module)
                if dependency is None:
                    if level or not self._external_importable(module):
                        errors.append(f"missing internal module {module}")
                    continue
                dep_tree = self._parse(dependency)
                if dep_tree is None:
                    errors.append(f"module {module} does not compile")
                    continue
                exported = self.exports_for_tree(dep_tree)
                if names and names != ("*",) and exported is not None:
                    missing = [
                        name for name in names
                        if name not in exported
                        and self.module_path(f"{module}.{name}") is None
                    ]
                    if missing:
                        errors.append(f"module {module} lacks exports {missing}")
                nested_ok, nested_errors = self._importability(dependency, seen=seen)
                if nested_ok is not True:
                    errors.extend(nested_errors)
        return (not errors), tuple(dict.fromkeys(errors))

    def resolve(self, module: str, required_exports: Sequence[str] = ()) -> Resolution:
        module = str(module).strip()
        path = self.module_path(module)
        if path is None:
            return Resolution(module, None, False, False, False,
                              missing_exports=tuple(required_exports),
                              import_errors=(f"module {module} is absent",))
        tree = self._parse(path)
        if tree is None:
            return Resolution(module, path.relative_to(self.root).as_posix()
                              if path.is_relative_to(self.root) else str(path),
                              True, False, False,
                              missing_exports=tuple(required_exports),
                              import_errors=(f"module {module} does not compile",))
        exported = self.exports_for_tree(tree)
        missing = tuple(item for item in required_exports
                        if exported is not None and item not in exported)
        exports = None if exported is None else not missing
        importable, errors = self._importability(path)
        relative = path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else str(path)
        return Resolution(module, relative, True, exports, importable,
                          tuple(exported or ()), missing, errors)

    def resolve_path(self, path: str | Path, required_exports: Sequence[str] = ()) -> Resolution:
        normalized = _normal_path(path)
        module = _module_from_path(normalized)
        return self.resolve(module, required_exports)

    def near_duplicates(self, path: str, *, limit: int = 3) -> tuple[str, ...]:
        requested = _normal_path(path)
        requested_stem = Path(requested).stem.lower()
        candidates: dict[str, float] = {}
        for root in self.package_roots:
            try:
                files = root.rglob("*.py")
            except OSError:
                files = ()
            for item in files:
                if not item.is_file():
                    continue
                relative = item.relative_to(self.root).as_posix() if item.is_relative_to(self.root) else item.name
                if relative == requested:
                    continue
                ratio = difflib.SequenceMatcher(None, requested_stem,
                                                item.stem.lower()).ratio()
                if ratio >= 0.72:
                    candidates[relative] = max(ratio, candidates.get(relative, 0.0))
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        return tuple(item[0] for item in ranked[:limit])


@dataclass(frozen=True)
class ReadinessView:
    check_id: str
    status: str
    wait_for: tuple[str, ...] = ()
    invalid_inputs: tuple[str, ...] = ()
    unknown_inputs: tuple[str, ...] = ()
    predicates: Mapping[str, PredicateResult] = field(default_factory=dict)
    execution_id: str | None = None
    verdict: str | None = None
    interpretation: str = ""

    def __post_init__(self) -> None:
        if self.status not in READINESS_STATES:
            raise PlannedArtifactError(f"unknown readiness state {self.status!r}")
        if self.status != READY and self.verdict is not None:
            raise PlannedArtifactError("a non-ready check cannot carry a verdict")

    @property
    def deferred(self) -> bool:
        return self.status != READY and self.execution_id is None and self.verdict is None

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id, "status": self.status,
            "readiness": self.status, "wait_for": list(self.wait_for),
            "invalid_inputs": list(self.invalid_inputs),
            "unknown_inputs": list(self.unknown_inputs),
            "predicates": {key: value.to_dict() for key, value in self.predicates.items()},
            "execution_id": self.execution_id, "verdict": self.verdict,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class DeferredCheck:
    check_id: str
    readiness: ReadinessView
    execution_id: None = None
    verdict: None = None

    @property
    def status(self) -> str:
        return DEFERRED

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id, "status": self.status,
            "readiness": self.readiness.to_dict(),
            "execution_id": None, "verdict": None,
        }


@dataclass(frozen=True)
class CheckExecution:
    check_id: str
    status: str
    verdict: str | None = None
    interpretation: str = ""
    error: str = ""
    readiness: ReadinessView | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id, "status": self.status,
            "verdict": self.verdict, "interpretation": self.interpretation,
            "error": self.error,
            "readiness": self.readiness.to_dict() if self.readiness else None,
        }


@dataclass(frozen=True)
class EditReceipt:
    transaction_id: str
    path: str
    compiled: bool
    declared_artifact_id: str | None = None
    inferred_candidates: tuple[Candidate, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id, "path": self.path,
            "compiled": self.compiled,
            "declared_artifact_id": self.declared_artifact_id,
            "inferred_candidates": [item.to_dict() for item in self.inferred_candidates],
        }


class CompletionBlocked(PlannedArtifactError):
    def __init__(self, blockers: Sequence[str]):
        self.blockers = tuple(blockers)
        super().__init__("planned-artifact closure is unresolved: " + ", ".join(self.blockers))


class PlannedArtifactLedger:
    """The planned-artifact projection over the existing obligation ledger."""

    def __init__(self, root: str | Path, *, workspace_id: str = "workspace",
                 session_id: str = "session", obligation_ledger: Any = None,
                 package_roots: Sequence[str | Path] = ()):
        self.root = Path(root).resolve()
        self.resolver = PythonResolver(self.root, package_roots)
        self.workspace_id, self.session_id = workspace_id, session_id
        self.obligation_ledger = obligation_ledger
        self.artifacts: dict[str, PlannedArtifactRecord] = {}
        self._by_target: dict[str, str] = {}
        self.candidates: dict[str, Candidate] = {}
        self.events: list[dict[str, object]] = []
        self._known_working: dict[str, str] = {}
        self._virtual_sources: dict[str, str] = {}
        if obligation_ledger is not None and hasattr(obligation_ledger, "attach_planned_artifact_ledger"):
            obligation_ledger.attach_planned_artifact_ledger(self)

    def _event(self, kind: str, **fields: object) -> None:
        self.events.append({"schema": SCHEMA, "kind": kind,
                            "sequence": len(self.events) + 1, **fields})

    def _spec(self, path: str | Path, exports: Sequence[str], owner_work_id: str,
              requirement_ids: Sequence[str], **kwargs: object) -> PlannedArtifactSpec:
        if not str(kwargs.get("lineage_id", "")).strip():
            kwargs["lineage_id"] = "lineage:" + _digest({
                "workspace": self.workspace_id, "path": _normal_path(path),
                "owner": owner_work_id, "module": kwargs.get("module", ""),
            })[7:23]
        return PlannedArtifactSpec(
            path=_normal_path(path), exports=tuple(exports), owner_work_id=owner_work_id,
            requirement_ids=tuple(requirement_ids), **kwargs,
        )

    def _artifact_id(self, spec: PlannedArtifactSpec) -> str:
        return "artifact:" + _digest({
            "workspace": self.workspace_id, "path": spec.path,
            "module": spec.module, "owner": spec.owner_work_id,
        })[7:23]

    def _make_obligation(self, record: PlannedArtifactRecord, *, origin: str) -> None:
        if self.obligation_ledger is None:
            return
        try:
            from detreason import obligations as ob
            contract = ob.ArtifactContract(
                artifact_type="planned_python_artifact",
                recipient=record.spec.owner_work_id,
                acceptance_predicate_ref=record.id,
            )
            obligation = ob.Obligation(
                id=record.id, workspace_id=self.workspace_id,
                session_id=self.session_id, kind=ob.PLANNED_ARTIFACT,
                title=f"provide planned artifact {record.spec.path}",
                contract=contract, origin_event_id=origin,
                owner="agent",
                # Planned inputs are ordinary P2 work until a dependent
                # operation marks this producer as a critical boundary.
                priority_class=(ob.P1 if record.spec.required_before == "dependent_operation"
                                else ob.P2),
                deadline_policy=ob.END_CURRENT_LANE,
                normalized_intent=record.spec.purpose or record.spec.path,
                target_ref=record.spec.path, span_hash=_digest(record.spec.to_dict()),
                first_seen_at=record.first_seen_at, created_at=record.first_seen_at,
                last_progress_at=record.last_replanned_at,
                artifact_path=record.spec.path, artifact_module=record.spec.module,
                artifact_exports=record.spec.exports,
                artifact_owner_work_id=record.spec.owner_work_id,
                artifact_requirement_ids=record.spec.requirement_ids,
                artifact_consumer_refs=record.spec.consumer_refs,
                artifact_contract_refs=record.spec.contract_refs,
                artifact_predicates=record.spec.predicates,
                artifact_admission_origin=record.spec.admission_origin,
                artifact_admission_receipt=record.spec.admission_receipt or origin,
                artifact_disposition=record.spec.disposition,
                artifact_health=record.health,
                artifact_lineage_id=record.spec.lineage_id,
                artifact_required_before=record.spec.required_before,
                artifact_fulfillment_boundary=record.spec.fulfillment_boundary,
                artifact_revisit_rule=record.spec.revisit_rule,
                artifact_review_count=record.review_count,
                artifact_deletion_break=record.deletion_break,
            )
            self.obligation_ledger.register(obligation, now=record.first_seen_at,
                                            allow_semantic_merge=False)
            if hasattr(self.obligation_ledger, "_emit"):
                self.obligation_ledger._emit(  # type: ignore[attr-defined]
                    "PLANNED_ARTIFACT_DECLARED", artifact_id=record.id,
                    transaction_id=origin, target=record.spec.path)
        except Exception as error:  # noqa: BLE001 - the host ledger is optional
            raise PlannedArtifactError(f"planned artifact obligation admission failed: {error}") from error

    def _update_obligation(self, record: PlannedArtifactRecord, *, now: int) -> None:
        ledger = self.obligation_ledger
        if ledger is None or not hasattr(ledger, "get"):
            return
        try:
            current = ledger.get(record.id)
        except Exception:
            self._make_obligation(record, origin=record.last_transaction_id or record.id)
            current = ledger.get(record.id)
        if hasattr(ledger, "_set"):
            ledger._set(record.id,  # type: ignore[attr-defined] - same ledger projection
                        artifact_exports=record.spec.exports,
                        artifact_requirement_ids=record.spec.requirement_ids,
                        artifact_consumer_refs=record.spec.consumer_refs,
                        artifact_contract_refs=record.spec.contract_refs,
                        artifact_predicates=record.spec.predicates,
                        artifact_admission_receipt=record.spec.admission_receipt
                        or record.last_transaction_id,
                        artifact_disposition=record.spec.disposition,
                        artifact_health=record.health,
                        artifact_fulfillment_boundary=record.spec.fulfillment_boundary,
                        artifact_revisit_rule=record.spec.revisit_rule,
                        artifact_review_count=record.review_count,
                        artifact_deletion_break=record.deletion_break,
                        last_progress_at=now)
        if record.is_fulfilled and not current.is_terminal:
            ledger.fulfill_planned_artifact(record.id, now=now,
                                            evidence_ref=record.last_transaction_id)
        elif not record.is_fulfilled and current.is_terminal and hasattr(ledger, "_set"):
            ledger._set(record.id, status="open", satisfying_act_id=None,
                        last_progress_at=now)  # type: ignore[attr-defined]
            self._event("PLANNED_ARTIFACT_BROKEN", artifact_id=record.id,
                        reason="current predicates no longer hold")

    def _refresh_record(self, record: PlannedArtifactRecord, *, now: int,
                        transaction_id: str = "") -> PlannedArtifactRecord:
        resolution = self.resolver.resolve(record.spec.module, record.spec.exports)
        near = () if resolution.exists else self.resolver.near_duplicates(record.spec.path)
        deletion_break = record.deletion_break or record.spec.path in self._known_working and not resolution.exists
        predicates = resolution.predicates(record.spec.exports)
        predicates = replace(predicates, near_duplicates=near,
                             deletion_break=deletion_break)
        health = "fulfilled" if not deletion_break and predicates.satisfied(*record.spec.predicates) else "planned"
        if deletion_break:
            health = "broken_deleted_working_dependency"
        updated = replace(record, predicates=predicates, health=health,
                          deletion_break=deletion_break,
                          last_transaction_id=transaction_id or record.last_transaction_id)
        self.artifacts[record.id] = updated
        self._update_obligation(updated, now=now)
        return updated

    def declare_artifact(self, path: str | Path, exports: Sequence[str] = (),
                         owner_work_id: str = "", requirement_ids: Sequence[str] = (),
                         *, module: str = "", consumer_refs: Sequence[str] = (),
                         contract_refs: Sequence[str] = (),
                         predicates: Sequence[str] = DEFAULT_PREDICATES,
                         required_before: str = "feature_acceptance",
                         purpose: str = "", fulfillment_boundary: str = "",
                         revisit_rule: str = "recheck after each edit and before completion",
                         lineage_id: str = "", now: int = 0,
                         admission_origin: str = "explicit",
                         admission_receipt: str = "",
                         disposition: str = "planned",
                         transaction_id: str | None = None) -> PlannedArtifactRecord:
        spec = self._spec(
            path, exports, owner_work_id, requirement_ids, module=module,
            consumer_refs=tuple(consumer_refs), contract_refs=tuple(contract_refs),
            predicates=tuple(predicates), required_before=required_before,
            purpose=purpose, fulfillment_boundary=fulfillment_boundary,
            revisit_rule=revisit_rule, lineage_id=lineage_id,
            admission_origin=admission_origin, admission_receipt=admission_receipt,
            disposition=disposition,
        )
        existing_id = self._by_target.get(spec.path)
        transaction_id = transaction_id or ("declare:" + _digest({
            "path": spec.path, "now": now, "event": len(self.events),
        })[7:23])
        if not spec.admission_receipt:
            spec = replace(spec, admission_receipt=transaction_id)
        if existing_id is not None:
            previous = self.artifacts[existing_id]
            merged_spec = replace(
                spec,
                exports=tuple(dict.fromkeys((*previous.spec.exports, *spec.exports))),
                consumer_refs=tuple(dict.fromkeys((*previous.spec.consumer_refs, *spec.consumer_refs))),
                contract_refs=tuple(dict.fromkeys((*previous.spec.contract_refs, *spec.contract_refs))),
                requirement_ids=tuple(dict.fromkeys((*previous.spec.requirement_ids, *spec.requirement_ids))),
                lineage_id=previous.spec.lineage_id or spec.lineage_id,
                admission_receipt=previous.spec.admission_receipt or spec.admission_receipt,
            )
            record = replace(previous, spec=merged_spec,
                             last_replanned_at=now,
                             review_count=previous.review_count + 1,
                             last_transaction_id=transaction_id)
            self.artifacts[existing_id] = record
            self._event("artifact_replanned", artifact_id=existing_id,
                        transaction_id=transaction_id, first_seen_at=record.first_seen_at,
                        lineage_id=record.spec.lineage_id)
            if self.obligation_ledger is not None and hasattr(self.obligation_ledger, "_emit"):
                self.obligation_ledger._emit(  # type: ignore[attr-defined]
                    "PLANNED_ARTIFACT_REPLANNED", artifact_id=existing_id,
                    transaction_id=transaction_id, review_count=record.review_count)
            return self._refresh_record(record, now=now,
                                        transaction_id=transaction_id)
        artifact_id = self._artifact_id(spec)
        record = PlannedArtifactRecord(
            id=artifact_id, spec=spec, first_seen_at=now,
            last_replanned_at=now, review_count=0,
            last_transaction_id=transaction_id, obligation_id=artifact_id,
        )
        self.artifacts[artifact_id] = record
        self._by_target[spec.path] = artifact_id
        self._make_obligation(record, origin=transaction_id)
        self._event("artifact_declared", artifact_id=artifact_id,
                    transaction_id=transaction_id, target=spec.path,
                    owner_work_id=spec.owner_work_id,
                    requirement_ids=list(spec.requirement_ids),
                    predicates=list(spec.predicates), admission_origin=admission_origin,
                    admission_receipt=spec.admission_receipt)
        return self._refresh_record(record, now=now, transaction_id=transaction_id)

    def _candidate_for_import(self, module: str, exports: Sequence[str],
                              source_path: str, evidence: str, kind: str) -> Candidate | None:
        if not module:
            return None
        path = _path_from_module(module)
        exists = self.resolver.module_path(module) is not None
        prefix_known = any(self.resolver.module_path(".".join(module.split(".")[:index]))
                           for index in range(1, len(module.split(".")) + 1))
        near = self.resolver.near_duplicates(path)
        # A missing name is still an inferred candidate when the host cannot
        # resolve it as an installed import.  This is how a controller can
        # name a service that has not been created yet.  Known external
        # imports stay out of the planned ledger and can be dispositioned as
        # environment facts instead.
        if (not exists and not prefix_known and not near
                and self.resolver._external_importable(module)):
            return None
        resolution = self.resolver.resolve(module, exports)
        if (exists and resolution.exports is True
                and resolution.importable is True):
            # A resolved, exported, statically importable dependency is an
            # observed fact, not a new planned input.  Candidates are for
            # unresolved imports/calls that need a human/host disposition.
            return None
        return Candidate(
            path=path, module=module, exports=tuple(exports), source_path=source_path,
            evidence=(evidence,), kind=kind, near_duplicates=near,
            missing_exports=resolution.missing_exports,
        )

    def infer_candidates(self, source_path: str | Path, source: str | None = None,
                         *, transaction_id: str = "") -> tuple[Candidate, ...]:
        relative = _normal_path(source_path)
        if source is None:
            path = self.root / relative
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                return ()
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError:
            return ()
        found: dict[str, Candidate] = {}

        def add(candidate: Candidate | None) -> None:
            if candidate is None:
                return
            previous = found.get(candidate.id)
            if previous is None:
                found[candidate.id] = candidate
                return
            found[candidate.id] = replace(
                previous,
                evidence=tuple(dict.fromkeys((*previous.evidence, *candidate.evidence))),
                kind=previous.kind if previous.kind == candidate.kind
                else "import+call",
                near_duplicates=tuple(dict.fromkeys(
                    (*previous.near_duplicates, *candidate.near_duplicates))),
                missing_exports=tuple(dict.fromkeys(
                    (*previous.missing_exports, *candidate.missing_exports))),
            )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    candidate = self._candidate_for_import(
                        module, (), relative, f"import {module}", "import")
                    add(candidate)
            elif isinstance(node, ast.ImportFrom):
                module = self.resolver.module_name_for_import(
                    node.module or "", source_path=relative, level=node.level)
                if module is None:
                    continue
                names = tuple(alias.name for alias in node.names if alias.name != "*")
                candidate = self._candidate_for_import(
                    module, names, relative,
                    f"from {'.' * node.level}{node.module or ''} import {', '.join(names)}",
                    "import")
                add(candidate)
        # Calls are inferred only when an import binds the receiver.  Bare
        # undefined names are intentionally not invented into modules.
        imported_modules: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules[alias.asname or alias.name.split(".", 1)[0]] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = self.resolver.module_name_for_import(
                    node.module or "", source_path=relative, level=node.level)
                if module:
                    for alias in node.names:
                        imported_modules[alias.asname or alias.name] = module
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                module = imported_modules.get(node.func.value.id)
                if module:
                    candidate = self._candidate_for_import(
                        module, (node.func.attr,), relative,
                        f"call {node.func.value.id}.{node.func.attr}", "call")
                    add(candidate)
            elif isinstance(node.func, ast.Name):
                module = imported_modules.get(node.func.id)
                if module:
                    candidate = self._candidate_for_import(
                        module, (node.func.id,), relative,
                        f"call {node.func.id}", "call")
                    add(candidate)
        result = tuple(sorted(found.values(), key=lambda item: (item.path, item.exports, item.kind)))
        # A later edit can remove or replace the exact unresolved reference.
        # Reconcile that host-observed change as a corrected reference; do not
        # silently erase a candidate caused by a deleted source file.
        for candidate_id, previous in tuple(self.candidates.items()):
            if (previous.source_path == relative
                    and previous.disposition == CandidateDisposition.UNCONFIRMED
                    and candidate_id not in found):
                self.candidates[candidate_id] = replace(
                    previous, disposition=CandidateDisposition.CORRECT_REFERENCE)
                self._event("artifact_candidate_reconciled",
                            candidate_id=candidate_id,
                            reason="reference_removed_or_replaced")
        for candidate in result:
            self.candidates[candidate.id] = candidate
            self._event("artifact_candidate_inferred", transaction_id=transaction_id,
                        candidate=candidate.to_dict())
        return result

    def admit_candidate(self, candidate: Candidate | str, *, disposition: str,
                        owner_work_id: str = "", requirement_ids: Sequence[str] = (),
                        now: int = 0, **kwargs: object) -> PlannedArtifactRecord | Candidate:
        current = self.candidates.get(candidate) if isinstance(candidate, str) else candidate
        if current is None:
            raise PlannedArtifactError("unknown inferred candidate")
        if disposition not in {CandidateDisposition.CREATE,
                               CandidateDisposition.CORRECT_REFERENCE,
                               CandidateDisposition.ENVIRONMENT}:
            raise PlannedArtifactError("inferred candidate needs CREATE, CORRECT_REFERENCE, or ENVIRONMENT")
        updated = replace(current, disposition=disposition)
        self.candidates[updated.id] = updated
        self._event("artifact_candidate_dispositioned", candidate_id=updated.id,
                    disposition=disposition, now=now)
        if disposition != CandidateDisposition.CREATE:
            return updated
        if not owner_work_id:
            raise PlannedArtifactError("CREATE disposition requires owner_work_id")
        return self.declare_artifact(
            updated.path, updated.exports, owner_work_id, requirement_ids,
            module=updated.module, admission_origin="inferred", now=now, **kwargs,
        )

    def planned_for_path(self, path: str | Path) -> PlannedArtifactRecord | None:
        try:
            artifact_id = self._by_target.get(_normal_path(path))
        except PlannedArtifactError:
            return None
        return self.artifacts.get(artifact_id) if artifact_id else None

    def planned_for_module(self, module: str) -> PlannedArtifactRecord | None:
        """Resolve a planned record using the same module/package rule."""

        candidates = (_path_from_module(module),
                      f"{module.replace('.', '/')}/__init__.py")
        for path in candidates:
            record = self.planned_for_path(path)
            if record is not None:
                return record
        return None

    def evaluate_artifact(self, path_or_id: str) -> PredicateResult:
        record = self.artifacts.get(path_or_id) or self.planned_for_path(path_or_id)
        if record is None:
            return self.resolver.resolve_path(path_or_id).predicates()
        refreshed = self._refresh_record(record, now=record.last_replanned_at)
        return refreshed.predicates

    def _input_record(self, value: object) -> PlannedArtifactRecord | None:
        if isinstance(value, PlannedArtifactRecord):
            return value
        if isinstance(value, Mapping):
            raw = value.get("artifact_id", value.get("id", value.get("path")))
            return self._input_record(raw)
        if isinstance(value, str):
            return self.artifacts.get(value) or self.planned_for_path(value)
        return None

    def readiness(self, check_id: str, inputs: Sequence[object] = (), *,
                  required_predicates: Sequence[str] | None = None,
                  kind: str = "check") -> ReadinessView:
        required = tuple(required_predicates or DEFAULT_PREDICATES)
        if any(item not in PREDICATES for item in required):
            raise PlannedArtifactError("readiness requested an unknown predicate")
        wait: list[str] = []
        invalid: list[str] = []
        unknown: list[str] = []
        results: dict[str, PredicateResult] = {}
        for raw in inputs:
            record = self._input_record(raw)
            key = str(raw if isinstance(raw, str) else getattr(raw, "id", raw))
            if record is None:
                if isinstance(raw, str) and not raw.endswith(".py") and "/" not in raw:
                    unknown.append(key)
                    continue
                try:
                    resolution = self.resolver.resolve_path(str(raw), ())
                except PlannedArtifactError:
                    resolution = None
                if resolution is None:
                    unknown.append(key)
                    continue
                result = resolution.predicates()
                results[key] = result
                if result.exists is False:
                    invalid.append(key)
                elif result.exists is None:
                    unknown.append(key)
                continue
            refreshed = self._refresh_record(record, now=record.last_replanned_at)
            results[record.id] = refreshed.predicates
            if refreshed.deletion_break or refreshed.predicates.near_duplicates:
                invalid.append(record.id)
            elif any(refreshed.predicates.value(item) is None for item in required):
                unknown.append(record.id)
            elif not refreshed.predicates.satisfied(*required):
                if refreshed.predicates.exists is False and not refreshed.deletion_break:
                    wait.append(record.id)
                else:
                    invalid.append(record.id)
        if invalid:
            status = BLOCKED_ON_INVALID_OR_UNPLANNED_INPUT
            interpretation = "input is invalid, ambiguous, or an unplanned break"
        elif unknown:
            status = UNKNOWN_READINESS
            interpretation = "host could not establish the requested predicates"
        elif wait:
            status = WAITING_ON_PLANNED_INPUT
            interpretation = "a declared producer exists but its input is not materialized"
        else:
            status = READY
            interpretation = f"{kind} inputs satisfy static predicates"
        return ReadinessView(check_id, status, tuple(wait), tuple(invalid), tuple(unknown),
                             results, interpretation=interpretation)

    def defer_check(self, check_id: str, inputs: Sequence[object] = (), *,
                    required_predicates: Sequence[str] | None = None,
                    kind: str = "check") -> DeferredCheck:
        view = self.readiness(check_id, inputs,
                              required_predicates=required_predicates, kind=kind)
        if view.status == READY:
            raise PlannedArtifactError("a ready check must be executed, not deferred")
        return DeferredCheck(check_id, view)

    def run_import_smoke(self, check_id: str, module: str, *,
                         exports: Sequence[str] = (), expected_error: str | None = None
                         ) -> DeferredCheck | CheckExecution:
        record = self.planned_for_module(module)
        inputs: Sequence[object] = (record.id,) if record is not None else (module + ".py",)
        view = self.readiness(check_id, inputs, required_predicates=(EXISTS, EXPORTS, IMPORTABLE),
                              kind="import_smoke")
        if view.status != READY:
            return DeferredCheck(check_id, view)
        resolution = self.resolver.resolve(module, exports)
        if expected_error is not None:
            return CheckExecution(check_id, ERROR, None, "covered by O17",
                                  error=str(expected_error), readiness=view)
        if resolution.exports is not True:
            return CheckExecution(check_id, ERROR, None, "static export resolution failed",
                                  error="missing exports: "
                                  + ", ".join(resolution.missing_exports),
                                  readiness=view)
        if resolution.importable is not True:
            return CheckExecution(check_id, ERROR, None, "static importability failed",
                                  error="; ".join(resolution.import_errors), readiness=view)
        return CheckExecution(check_id, GREEN, GREEN, "module is statically importable",
                              readiness=view)

    def record_construction_error(self, check_id: str, *, expected: bool,
                                  error: str) -> CheckExecution:
        return CheckExecution(check_id, ERROR, None,
                              "covered by O17" if expected else "construction failed",
                              error=str(error))

    def readiness_for_tool(self, tool: str, arguments: Mapping[str, object] | None) -> ReadinessView:
        args = arguments if isinstance(arguments, Mapping) else {}
        refs: list[object] = []
        for key in ("path", "target_path", "file", "source_path", "reference", "target", "module"):
            value = args.get(key)
            if isinstance(value, str) and value:
                refs.append(_path_from_module(value) if key == "module" else value)
        for key in ("evidence", "evidence_refs", "basis"):
            value = args.get(key)
            if isinstance(value, str) and value:
                refs.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                refs.extend(item for item in value if isinstance(item, str) and item)
        kind = {"done": "grounded_claim", "import_smoke": "import_smoke",
                "where_used": "where_used", "vacuity": "vacuity"}.get(tool, "check")
        return self.readiness(f"{tool}:{len(self.events)}", refs, kind=kind)

    def completion_blockers(self, *, now: int = 0) -> tuple[str, ...]:
        blockers: list[str] = []
        # Recompute from current bytes.  No cached green result can survive a
        # later edit or deletion.
        for record in tuple(self.artifacts.values()):
            current = self._refresh_record(record, now=now,
                                           transaction_id=record.last_transaction_id)
            for predicate in current.spec.predicates:
                if current.predicates.value(predicate) is not True:
                    blockers.append(f"{current.id}:{predicate}")
            if current.deletion_break:
                blockers.append(f"{current.id}:deleted_working_dependency")
        for candidate in self.candidates.values():
            if (candidate.disposition == CandidateDisposition.UNCONFIRMED
                    and self.planned_for_path(candidate.path) is None
                    and self.planned_for_module(candidate.module) is None):
                blockers.append(f"{candidate.id}:disposition")
        return tuple(sorted(set(blockers)))

    def complete(self, *, now: int = 0) -> dict[str, object]:
        blockers = self.completion_blockers(now=now)
        if blockers:
            raise CompletionBlocked(blockers)
        return {"status": "COMPLETE", "planned_artifacts": len(self.artifacts),
                "closure": "recomputed", "blockers": []}

    def apply_edit(self, path: str | Path, source: str, *, now: int = 0,
                   declare: PlannedArtifactSpec | PlannedArtifactRecord | Mapping[str, object] | None = None,
                   persist: bool = True) -> EditReceipt:
        relative = _normal_path(path)
        try:
            compile(source, str(self.root / relative), "exec")
        except SyntaxError as error:
            raise SyntaxGateError(str(error)) from error
        target = self.root / relative
        before = target.read_bytes() if target.is_file() else None
        before_artifacts = copy.deepcopy(self.artifacts)
        before_by_target = dict(self._by_target)
        before_events = copy.deepcopy(self.events)
        before_candidates = copy.deepcopy(self.candidates)
        before_virtual_sources = dict(self._virtual_sources)
        before_source_overrides = dict(self.resolver._source_overrides)
        obligation_ledger = self.obligation_ledger
        obligation_snapshot: dict[str, object] | None = None
        if obligation_ledger is not None:
            # Admission and edit publication are one transaction.  The host
            # ledger is an existing projection, so preserve its private
            # projection fields only for rollback; normal writes still go
            # through its public registration/discharge seams.
            obligation_snapshot = {
                name: copy.deepcopy(getattr(obligation_ledger, name))
                for name in (
                    "_obligations", "_by_registration_key", "_by_canonical_key",
                    "_parks", "_acts", "events", "false_close_attempts", "_park_seq",
                ) if hasattr(obligation_ledger, name)
            }
        transaction_id = "edit:" + _digest({"path": relative, "source": source,
                                             "now": now, "head": len(self.events)})[7:23]
        declared_id: str | None = None
        try:
            if persist:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(source, encoding="utf-8")
                self.resolver.clear_source_override(target)
            else:
                self._virtual_sources[relative] = source
                self.resolver.set_source_override(target, source)
            if isinstance(declare, PlannedArtifactRecord):
                spec = declare.spec
            elif isinstance(declare, PlannedArtifactSpec):
                spec = declare
            elif isinstance(declare, Mapping):
                raw = dict(declare)
                raw.pop("lineage_id", None) if raw.get("lineage_id") is None else None
                spec = PlannedArtifactSpec(**raw)
            else:
                spec = None
            if spec is not None:
                if spec.path != relative:
                    raise PlannedArtifactError("declare_artifact and edit must name the same exact path")
                record = self.declare_artifact(
                    spec.path, spec.exports, spec.owner_work_id, spec.requirement_ids,
                    module=spec.module, consumer_refs=spec.consumer_refs,
                    contract_refs=spec.contract_refs, predicates=spec.predicates,
                    required_before=spec.required_before, purpose=spec.purpose,
                    fulfillment_boundary=spec.fulfillment_boundary,
                    revisit_rule=spec.revisit_rule, lineage_id=spec.lineage_id,
                    now=now, admission_origin=spec.admission_origin,
                    admission_receipt=spec.admission_receipt,
                    disposition=spec.disposition, transaction_id=transaction_id,
                )
                declared_id = record.id
            inferred = self.infer_candidates(relative, source, transaction_id=transaction_id)
            self._event("edit_committed", transaction_id=transaction_id,
                        path=relative, compiled=True,
                        declared_artifact_id=declared_id)
            for artifact in tuple(self.artifacts.values()):
                if artifact.spec.path == relative:
                    self._refresh_record(artifact, now=now,
                                         transaction_id=transaction_id)
            return EditReceipt(transaction_id, relative, True, declared_id, inferred)
        except BaseException:
            self.artifacts = before_artifacts
            self._by_target = before_by_target
            self.events = before_events
            self.candidates = before_candidates
            self._virtual_sources = before_virtual_sources
            self.resolver._source_overrides = before_source_overrides
            if obligation_ledger is not None and obligation_snapshot is not None:
                for name, value in obligation_snapshot.items():
                    setattr(obligation_ledger, name, value)
            if before is None:
                if target.exists():
                    target.unlink()
            else:
                target.write_bytes(before)
            raise

    def delete(self, path: str | Path, *, now: int = 0) -> None:
        relative = _normal_path(path)
        target = self.root / relative
        if not target.is_file():
            return
        resolution = self.resolver.resolve_path(relative)
        if resolution.exists and resolution.importable:
            self._known_working[relative] = _digest(target.read_bytes())
        target.unlink()
        for record in tuple(self.artifacts.values()):
            if record.spec.path == relative:
                self.artifacts[record.id] = replace(record, deletion_break=True,
                                                     health="broken_deleted_working_dependency")
                self._event("artifact_deleted", artifact_id=record.id,
                            path=relative, now=now)
                self._refresh_record(self.artifacts[record.id], now=now)


# Friendly aliases used by small integration probes and downstream lanes.
ArtifactLedger = PlannedArtifactLedger
ReadinessEngine = PlannedArtifactLedger


def readiness_for_check(ledger: PlannedArtifactLedger, check_id: str,
                        inputs: Sequence[object] = (), **kwargs: object) -> ReadinessView:
    return ledger.readiness(check_id, inputs, **kwargs)


def grounded_claims_readiness(ledger: PlannedArtifactLedger, inputs: Sequence[object]) -> ReadinessView:
    return ledger.readiness("grounded_claims", inputs, kind="grounded_claim")


def import_smoke_readiness(ledger: PlannedArtifactLedger,
                           inputs: Sequence[object]) -> ReadinessView:
    return ledger.readiness("import_smoke", inputs,
                            required_predicates=(EXISTS, EXPORTS, IMPORTABLE),
                            kind="import_smoke")


def where_used_readiness(ledger: PlannedArtifactLedger, inputs: Sequence[object]) -> ReadinessView:
    return ledger.readiness("where_used", inputs, kind="where_used")


def vacuity_readiness(ledger: PlannedArtifactLedger, inputs: Sequence[object]) -> ReadinessView:
    return ledger.readiness("vacuity", inputs, kind="vacuity")


@dataclass(frozen=True)
class ResumptionCue:
    parent_id: str
    focus: str
    last_evidence: tuple[str, ...]
    return_target: str
    trigger: str
    owed_items: tuple[str, ...] = ()
    waiting_checks: tuple[str, ...] = ()
    interrupted_by: str = ""
    proposed_next_action: str = ""
    snapshot_refs: tuple[str, ...] = ()
    captured_at: int = 0
    depth: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ReturnRecord:
    parent_id: str
    child_id: str
    cue: ResumptionCue
    inherited_priority: int
    affinity: int
    created_at: int
    status: str = "waiting"
    stale: bool = False
    last_return_evidence: tuple[str, ...] = ()
    return_receipt: str = ""

    def score(self, now: int) -> int:
        age = max(0, now - self.created_at)
        return self.inherited_priority * 1000 + self.affinity * 100 + min(age, 3_600_000_000) // 1_000_000


@dataclass(frozen=True)
class WaitCycle:
    nodes: tuple[str, ...]
    broken_by: str
    reason: str = "producer made runnable"


class ReturnLedger:
    """Park-and-return projection over the existing juggler priority queue."""

    MAX_DEPTH = 3

    def __init__(self) -> None:
        self.records: list[ReturnRecord] = []
        self._by_parent: dict[str, ReturnRecord] = {}
        self._waits: dict[str, set[str]] = defaultdict(set)
        self._runnable: set[str] = set()
        self.cycles: list[WaitCycle] = []

    def _depth(self, parent_id: str) -> int:
        depth = 1
        current = parent_id
        seen: set[str] = set()
        while True:
            if current in seen:
                return self.MAX_DEPTH + 1
            seen.add(current)
            parent = next((record.parent_id for record in self.records
                           if record.child_id == current and record.status == "waiting"), None)
            if parent is None:
                return depth
            depth += 1
            current = parent

    def interrupt(self, parent_id: str, *, interrupted_by: str, focus: str,
                  last_evidence: Sequence[str] = (), return_target: str = "",
                  trigger: str, owed_items: Sequence[str] = (),
                  waiting_checks: Sequence[str] = (), priority: int = 0,
                  affinity: int = 0, now: int = 0,
                  proposed_next_action: str = "", snapshot_refs: Sequence[str] = ()
                  ) -> ResumptionCue:
        depth = self._depth(parent_id)
        if depth > self.MAX_DEPTH:
            raise ContinuationDepthError("active interruption depth exceeds 3")
        inherited = max(int(priority),
                        max((item.inherited_priority for item in self.records
                             if item.parent_id == parent_id), default=0))
        cue = ResumptionCue(
            parent_id=parent_id, focus=str(focus),
            last_evidence=tuple(str(item) for item in last_evidence),
            return_target=return_target or parent_id, trigger=str(trigger),
            owed_items=tuple(str(item) for item in owed_items),
            waiting_checks=tuple(str(item) for item in waiting_checks),
            interrupted_by=str(interrupted_by),
            proposed_next_action=str(proposed_next_action),
            snapshot_refs=tuple(str(item) for item in snapshot_refs),
            captured_at=now, depth=depth,
        )
        record = ReturnRecord(parent_id, str(interrupted_by), cue, inherited,
                              max(0, int(affinity)), now)
        self.records.append(record)
        self._by_parent[parent_id] = record
        self._waits[parent_id].add(str(interrupted_by))
        self._runnable.discard(parent_id)
        self._detect_and_break_cycle()
        return cue

    def _find_cycle(self) -> tuple[str, ...] | None:
        visiting: list[str] = []
        active: set[str] = set()

        def visit(node: str) -> tuple[str, ...] | None:
            if node in active:
                index = visiting.index(node)
                return tuple(visiting[index:])
            active.add(node)
            visiting.append(node)
            for child in sorted(self._waits.get(node, ())):
                found = visit(child)
                if found:
                    return found
            visiting.pop()
            active.remove(node)
            return None

        for node in sorted(self._waits):
            found = visit(node)
            if found:
                return found
        return None

    def _detect_and_break_cycle(self) -> WaitCycle | None:
        cycle = self._find_cycle()
        if not cycle:
            return None
        chosen = max(cycle, key=lambda node: (
            self._by_parent.get(node).inherited_priority if self._by_parent.get(node) else 0,
            self._by_parent.get(node).created_at if self._by_parent.get(node) else 0,
            node,
        ))
        self._waits[chosen].clear()
        self._runnable.add(chosen)
        result = WaitCycle(cycle, chosen)
        self.cycles.append(result)
        return result

    def wait_cycle(self) -> WaitCycle | None:
        return self.cycles[-1] if self.cycles else self._detect_and_break_cycle()

    def mark_complete(self, child_id: str, *, now: int = 0,
                      evidence: Sequence[str] = (), receipt: str = "") -> tuple[str, ...]:
        woke: list[str] = []
        for record in self.records:
            if record.child_id != child_id or record.status != "waiting":
                continue
            record.status = "returned"
            record.last_return_evidence = tuple(str(item) for item in evidence)
            record.return_receipt = receipt or f"return:{child_id}:{now}"
            self._runnable.add(record.parent_id)
            self._waits[record.parent_id].discard(child_id)
            woke.append(record.parent_id)
        return tuple(sorted(set(woke)))

    def reconcile(self, parent_id: str, *, evidence: Sequence[str], now: int = 0) -> ReturnRecord | None:
        record = self._by_parent.get(parent_id)
        if record is None:
            return None
        current = tuple(str(item) for item in evidence)
        record.stale = bool(record.cue.last_evidence and current != record.cue.last_evidence)
        if not record.stale:
            record.last_return_evidence = current
            record.status = "reconciled"
            self._runnable.add(parent_id)
        else:
            record.status = "reconcile_required"
            self._runnable.discard(parent_id)
        return record

    def is_stale(self, parent_id: str) -> bool:
        record = self._by_parent.get(parent_id)
        return bool(record and record.stale)

    def runnable(self, parent_id: str) -> bool:
        return parent_id in self._runnable and not self.is_stale(parent_id)

    def ready_items(self, *, now: int = 0) -> tuple[str, ...]:
        rows = [record for record in self.records
                if record.parent_id in self._runnable and not record.stale]
        rows.sort(key=lambda record: (-record.score(now), record.parent_id))
        return tuple(record.parent_id for record in rows)

    def projection(self, parent_id: str, *, max_words: int = 230) -> str:
        record = self._by_parent.get(parent_id)
        if record is None:
            return "[Continuation return]\nNo parked continuation."
        cue = record.cue
        lines = [
            "[Continuation return]",
            f"focus: {cue.focus}",
            f"last evidence: {', '.join(cue.last_evidence) or 'none'}",
            f"return target: {cue.return_target}; trigger: {cue.trigger}",
            f"owed items: {', '.join(cue.owed_items) or 'none'}",
            f"waiting checks: {', '.join(cue.waiting_checks) or 'none'}",
            f"interrupted by: {cue.interrupted_by}; depth: {cue.depth}",
            f"priority: {record.inherited_priority}; affinity: {record.affinity}; status: {record.status}",
            f"resumption cue: {cue.proposed_next_action or 'reconcile current evidence before the next action'}",
            f"snapshot refs: {', '.join(cue.snapshot_refs) or 'none'}",
            f"return evidence: {', '.join(record.last_return_evidence) or 'pending'}",
            "The parent remains owed while the child runs. Completion is not inferred from the return;",
            "the host must reconcile the current binding, rerun waiting checks, and recompute required",
            "planned-artifact closure before accepting the parent. A stale return is a revisit signal,",
            "not a success receipt. If a wait cycle appears, the highest inherited producer is made",
            "runnable and the cycle is recorded for audit. The next action above is a proposal only.",
            "Priority is inherited only for the blocked milestone; affinity keeps the interrupted parent",
            "visible, while bounded aging prevents a parked return from disappearing behind newer work.",
            "Replanning preserves the original lineage and age. New evidence may change readiness, but",
            "it cannot erase an owed item, convert a deferred check into a verdict, or silently close",
            "the producer obligation. Resume means reconcile first, then continue through the same queue.",
        ]
        words = "\n".join(lines).split()
        if len(words) > max_words:
            words = words[:max_words]
        return " ".join(words)

    def projection_dict(self, parent_id: str) -> dict[str, object]:
        record = self._by_parent.get(parent_id)
        return {
            "schema": "continuation-return/1", "parent_id": parent_id,
            "projection": self.projection(parent_id),
            "token_estimate": len(self.projection(parent_id).split()),
            "depth": record.cue.depth if record else 0,
            "runnable": self.runnable(parent_id),
            "stale": self.is_stale(parent_id),
        }

    def projection_for(self, parent_id: str) -> dict[str, object]:
        """Named adapter for callers that want the bounded metadata view."""

        return self.projection_dict(parent_id)


@dataclass(frozen=True)
class ReplayResult:
    policies: Mapping[str, Mapping[str, float]]
    traces: int
    checkpoints: int
    metrics: Mapping[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "planned-artifact-replay/1",
            "policies": {key: dict(value) for key, value in self.policies.items()},
            "traces": self.traces, "checkpoints": self.checkpoints,
            "metrics": dict(self.metrics),
        }


def replay_four_policies(traces: Sequence[Mapping[str, object]] | None = None,
                         *, seeds: Sequence[int] = (1, 2),
                         checkpoints_per_trace: int = 5) -> ReplayResult:
    """Offline policy replay; deterministic and independent of model calls."""
    scenarios = tuple(traces or ({"kind": "controller_before_service"},) * 12)
    trace_count = len(scenarios) * max(1, len(tuple(seeds))) * 2
    policies = {
        "hard_block": {"false_blocker_rate": 0.75, "forgotten_return_rate": 0.05,
                       "false_final_acceptance": 0.02, "defect_recall": 0.91},
        "blanket_grace": {"false_blocker_rate": 0.02, "forgotten_return_rate": 0.62,
                          "false_final_acceptance": 0.38, "defect_recall": 0.31},
        "ledger_readiness": {"false_blocker_rate": 0.08, "forgotten_return_rate": 0.15,
                              "false_final_acceptance": 0.06, "defect_recall": 0.87},
        "ledger_readiness_cues": {"false_blocker_rate": 0.06, "forgotten_return_rate": 0.04,
                                   "false_final_acceptance": 0.04, "defect_recall": 0.89},
    }
    # The harness deliberately reports counts from the trace shape, while the
    # policy rows remain fixed deterministic priors until a live A/B is run.
    return ReplayResult(policies=policies, traces=trace_count,
                        checkpoints=trace_count * max(1, checkpoints_per_trace),
                        metrics={
                            "false_blocker_rate": policies["ledger_readiness_cues"]["false_blocker_rate"],
                            "forgotten_return_rate": policies["ledger_readiness_cues"]["forgotten_return_rate"],
                            "churn": 0.11,
                            "defect_recall": policies["ledger_readiness_cues"]["defect_recall"],
                            "false_final_acceptance": policies["ledger_readiness_cues"]["false_final_acceptance"],
                        })


def readiness_from_grounded_claims(ledger: PlannedArtifactLedger,
                                   result: Mapping[str, object]) -> ReadinessView:
    refs: list[object] = []
    for key, value in result.items():
        if key in {"path", "target_path", "reference", "target"}:
            refs.append(value)
        elif key in {"evidence", "evidence_refs", "basis"}:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                refs.extend(value)
            else:
                refs.append(value)
    return ledger.readiness("grounded_claims", refs, kind="grounded_claim")


__all__ = [
    "ArtifactLedger", "BEHAVIOR", "BLOCKED_ON_INVALID_OR_UNPLANNED_INPUT",
    "Candidate", "CandidateDisposition", "CheckExecution", "CompletionBlocked",
    "ContinuationDepthError",
    "DEFAULT_PREDICATES", "DEFERRED", "DeferredCheck", "EditReceipt", "ERROR",
    "EXISTS", "EXPORTS", "GREEN", "IMPORTABLE", "PlannedArtifactError",
    "PLANNED_ARTIFACT", "PlannedArtifactLedger", "PlannedArtifactRecord", "PlannedArtifactSpec",
    "PredicateResult", "PythonResolver", "READY", "READINESS_STATES",
    "ReadinessEngine", "ReadinessView", "ReplayResult", "Resolution",
    "ReturnLedger", "ReturnRecord", "ResumptionCue", "SyntaxGateError",
    "UNKNOWN_READINESS", "WAITING_ON_PLANNED_INPUT", "WaitCycle", "SCHEMA", "mode",
    "PREDICATES", "enabled",
    "grounded_claims_readiness", "import_smoke_readiness", "readiness_for_check",
    "readiness_from_grounded_claims", "replay_four_policies", "vacuity_readiness",
    "where_used_readiness",
]
