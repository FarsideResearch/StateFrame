"""Typed public/private task manifests for the ladder evaluator.

The checked-in build corpora predate the manifest contract.  ``load_task_manifest``
normalizes those controller rows without exposing their reference implementation
to the public worker payload.  New callers can pass the public/private shape from
the ladder design directly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, cast


RunnerName: TypeAlias = Literal["pytest", "vitest", "eslint", "static"]

ALLOWED_RUNNERS: frozenset[str] = frozenset({"pytest", "vitest", "eslint", "static"})
PUBLIC_FIELDS: tuple[str, ...] = (
    "task_id",
    "track",
    "rung",
    "world",
    "base_digest",
    "files",
    "public_tests",
    "runner",
    "requires_db",
    "selectors",
    "allowed_product_paths",
)
PRIVATE_FIELDS: frozenset[str] = frozenset(
    {
        "reference_files",
        "reference_patch",
        "hidden_tests",
        "reference_selectors",
        "expected_answer",
        "required_plan_atoms",
        "oracle_digest",
    }
)
_LEGACY_PRIVATE_FIELDS: frozenset[str] = frozenset(
    {"fixed_src", "gold_src", "fixed_source", "reference_src"}
)


class ManifestValidationError(ValueError):
    """Raised when a task row cannot be represented safely by the contract."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        prefix = f"{field}: " if field else ""
        super().__init__(prefix + message)


@dataclass(frozen=True, slots=True)
class TaskFile:
    """A path and source blob visible in one side of a task world."""

    path: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "source": self.source}

    @classmethod
    def from_value(cls, value: object, field: str) -> "TaskFile":
        if not isinstance(value, Mapping):
            raise ManifestValidationError("must be an object", field)
        raw_path = value.get("path")
        if not isinstance(raw_path, str):
            raise ManifestValidationError("path must be a string", f"{field}.path")
        raw_source = value.get("source", value.get("content"))
        if not isinstance(raw_source, str):
            raise ManifestValidationError("source must be a string", f"{field}.source")
        path = _validate_relative_path(raw_path, f"{field}.path")
        return cls(path=path, source=raw_source)


@dataclass(frozen=True, slots=True)
class PublicTaskManifest:
    """The only task material that may cross into the agent worker."""

    task_id: str
    track: str
    rung: str
    world: str
    base_digest: str
    files: tuple[TaskFile, ...]
    public_tests: tuple[TaskFile, ...]
    runner: RunnerName
    requires_db: bool
    selectors: tuple[str, ...]
    allowed_product_paths: tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "PublicTaskManifest":
        _reject_private_public_keys(raw)
        required = {
            "task_id",
            "track",
            "rung",
            "world",
            "base_digest",
            "files",
            "public_tests",
            "runner",
            "requires_db",
            "selectors",
            "allowed_product_paths",
        }
        missing = sorted(key for key in required if key not in raw)
        if missing:
            raise ManifestValidationError(
                f"missing required field(s): {', '.join(missing)}"
            )

        task_id = _required_text(raw["task_id"], "task_id")
        track = _required_text(raw["track"], "track")
        rung = _required_text(raw["rung"], "rung")
        world = _required_text(raw["world"], "world")
        base_digest = _required_text(raw["base_digest"], "base_digest")
        files = _file_tuple(raw["files"], "files")
        public_tests = _file_tuple(raw["public_tests"], "public_tests")
        runner = _runner(raw["runner"])
        requires_db = _required_bool(raw["requires_db"], "requires_db")
        selectors = _string_tuple(raw["selectors"], "selectors")
        allowed = _allowed_paths(raw["allowed_product_paths"])

        if not files and world not in {"answer", "planning"}:
            raise ManifestValidationError(
                "must contain at least one product file outside answer/planning worlds",
                "files",
            )
        if not selectors and public_tests and runner not in {"static", "eslint"}:
            raise ManifestValidationError("must select at least one test", "selectors")
        for item in selectors:
            _validate_selector(item, "selectors")
        _reject_duplicate_paths(files, "files")
        _reject_duplicate_paths(public_tests, "public_tests")
        for item in files:
            if not _path_is_allowed(item.path, allowed):
                raise ManifestValidationError(
                    f"file path is outside allowed_product_paths: {item.path}",
                    "files",
                )

        return cls(
            task_id=task_id,
            track=track,
            rung=rung,
            world=world,
            base_digest=base_digest,
            files=files,
            public_tests=public_tests,
            runner=runner,
            requires_db=requires_db,
            selectors=selectors,
            allowed_product_paths=allowed,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe worker payload with no private oracle fields."""

        return {
            "task_id": self.task_id,
            "track": self.track,
            "rung": self.rung,
            "world": self.world,
            "base_digest": self.base_digest,
            "files": [item.to_dict() for item in self.files],
            "public_tests": [item.to_dict() for item in self.public_tests],
            "runner": self.runner,
            "requires_db": self.requires_db,
            "selectors": list(self.selectors),
            "allowed_product_paths": list(self.allowed_product_paths),
        }

    @property
    def primary_file(self) -> TaskFile | None:
        return self.files[0] if self.files else None


@dataclass(frozen=True, slots=True)
class PrivateOracleManifest:
    """Controller-only reference material and answer oracle metadata."""

    reference_files: tuple[TaskFile, ...] = ()
    reference_patch: str | None = None
    hidden_tests: tuple[TaskFile, ...] = ()
    reference_selectors: tuple[str, ...] = ()
    expected_answer: object | None = None
    required_plan_atoms: tuple[str, ...] = ()
    oracle_digest: str = ""

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, object],
        *,
        fallback_reference_path: str | None = None,
        fallback_reference_source: str | None = None,
        fallback_selectors: Sequence[str] = (),
    ) -> "PrivateOracleManifest":
        references_value = raw.get("reference_files")
        if references_value is None and fallback_reference_source is not None:
            if fallback_reference_path is None:
                raise ManifestValidationError(
                    "a reference path is required when fixed source is supplied",
                    "reference_files",
                )
            references_value = [
                {"path": fallback_reference_path, "source": fallback_reference_source}
            ]
        reference_files = _file_tuple(
            references_value if references_value is not None else [], "reference_files"
        )
        hidden_tests = _file_tuple(raw.get("hidden_tests", []), "hidden_tests")
        reference_selectors = _string_tuple(
            raw.get("reference_selectors", list(fallback_selectors)),
            "reference_selectors",
        )
        for item in reference_selectors:
            _validate_selector(item, "reference_selectors")
        required_plan_atoms = _string_tuple(
            raw.get("required_plan_atoms", []), "required_plan_atoms"
        )
        reference_patch = raw.get("reference_patch")
        if reference_patch is not None and not isinstance(reference_patch, str):
            raise ManifestValidationError("must be a string or null", "reference_patch")
        oracle_digest = raw.get("oracle_digest")
        if oracle_digest is not None and not isinstance(oracle_digest, str):
            raise ManifestValidationError("must be a string", "oracle_digest")

        digest_input = {
            "reference_files": [item.to_dict() for item in reference_files],
            "reference_patch": reference_patch,
            "hidden_tests": [item.to_dict() for item in hidden_tests],
            "reference_selectors": list(reference_selectors),
            "expected_answer": raw.get("expected_answer"),
            "required_plan_atoms": list(required_plan_atoms),
        }
        computed_digest = hashlib.sha256(
            json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        digest = oracle_digest or computed_digest
        if not digest:
            raise ManifestValidationError("must not be empty", "oracle_digest")

        return cls(
            reference_files=reference_files,
            reference_patch=cast(str | None, reference_patch),
            hidden_tests=hidden_tests,
            reference_selectors=reference_selectors,
            expected_answer=raw.get("expected_answer"),
            required_plan_atoms=required_plan_atoms,
            oracle_digest=digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_files": [item.to_dict() for item in self.reference_files],
            "reference_patch": self.reference_patch,
            "hidden_tests": [item.to_dict() for item in self.hidden_tests],
            "reference_selectors": list(self.reference_selectors),
            "expected_answer": self.expected_answer,
            "required_plan_atoms": list(self.required_plan_atoms),
            "oracle_digest": self.oracle_digest,
        }


@dataclass(frozen=True, slots=True)
class TaskManifest:
    """Validated public/private pair; ``public`` is safe to send to an agent."""

    public: PublicTaskManifest
    private: PrivateOracleManifest = field(default_factory=PrivateOracleManifest)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "TaskManifest":
        normalized = _normalize_row(row)
        public_raw = {key: normalized[key] for key in PUBLIC_FIELDS}
        public = PublicTaskManifest.from_mapping(public_raw)

        private_raw = _private_mapping(row)
        path = public.primary_file.path if public.primary_file else None
        fixed_source = _first_legacy_reference(row)
        private = PrivateOracleManifest.from_mapping(
            private_raw,
            fallback_reference_path=path,
            fallback_reference_source=fixed_source,
            fallback_selectors=public.selectors,
        )
        return cls(public=public, private=private)

    @property
    def task_id(self) -> str:
        return self.public.task_id

    @property
    def runner(self) -> RunnerName:
        return self.public.runner

    @property
    def requires_db(self) -> bool:
        return self.public.requires_db

    def public_dict(self) -> dict[str, object]:
        return self.public.to_dict()

    def private_dict(self) -> dict[str, object]:
        return self.private.to_dict()

    def agent_payload(self) -> dict[str, object]:
        """Alias documenting the intentional public/private boundary."""

        return self.public_dict()

    @property
    def public_spec(self) -> PublicTaskManifest:
        return self.public

    @property
    def private_oracle(self) -> PrivateOracleManifest:
        return self.private

    def to_public(self) -> dict[str, object]:
        return self.public_dict()

    def to_private(self) -> dict[str, object]:
        return self.private_dict()


def load_task_manifest(
    value: TaskManifest | Mapping[str, object] | str | Path,
) -> TaskManifest:
    """Load one row from a mapping, JSON object, or JSON file."""

    if isinstance(value, TaskManifest):
        return value
    row = _load_mapping(value)
    return TaskManifest.from_row(row)


def load_task_corpus(path: str | Path) -> tuple[TaskManifest, ...]:
    """Load a JSONL corpus while retaining one validated manifest per row."""

    source = Path(path)
    manifests: list[TaskManifest] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestValidationError(
                f"invalid JSON: {exc.msg}", f"{source}:{line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ManifestValidationError(
                "JSONL row must be an object", f"{source}:{line_number}"
            )
        try:
            manifests.append(TaskManifest.from_row(value))
        except ManifestValidationError as exc:
            raise ManifestValidationError(str(exc), f"{source}:{line_number}") from exc
    return tuple(manifests)


load_task_rows = load_task_corpus
load_manifest = load_task_manifest


def validate_task_row(row: Mapping[str, object]) -> TaskManifest:
    """Validate and normalize one legacy or manifest-shaped controller row."""

    return TaskManifest.from_row(row)


def validate_task_manifest(value: TaskManifest | Mapping[str, object]) -> TaskManifest:
    """Validate a manifest or row and return the normalized manifest."""

    if not isinstance(value, TaskManifest):
        return TaskManifest.from_row(value)
    PublicTaskManifest.from_mapping(value.public.to_dict())
    PrivateOracleManifest.from_mapping(value.private.to_dict())
    return value


def validate_public_manifest(raw: Mapping[str, object]) -> PublicTaskManifest:
    """Validate a worker-facing payload and reject private oracle leakage."""

    return PublicTaskManifest.from_mapping(raw)


validate_manifest = validate_task_manifest
PublicTaskSpec = PublicTaskManifest
PrivateTaskOracle = PrivateOracleManifest


def is_valid_task_row(row: Mapping[str, object]) -> bool:
    """Boolean convenience wrapper for inventory/preflight callers."""

    try:
        validate_task_row(row)
    except (ManifestValidationError, TypeError, ValueError):
        return False
    return True


def _load_mapping(value: Mapping[str, object] | str | Path) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Path):
        raw = value.read_text(encoding="utf-8")
    elif isinstance(value, str):
        try:
            candidate = Path(value)
            raw = (
                candidate.read_text(encoding="utf-8") if candidate.is_file() else value
            )
        except OSError:
            raw = value
    else:
        raise ManifestValidationError("must be a mapping, JSON object, or file path")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise ManifestValidationError("JSON value must be an object")
    return parsed


def _normalize_row(row: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ManifestValidationError("task row must be an object")
    normalized: dict[str, object] = {}

    path = _legacy_product_path(row, row.get("files"))
    fn = row.get("fn")
    fn_text = fn if isinstance(fn, str) else "task"

    files = row.get("files")
    if files is None:
        parent_source = row.get("parent_src")
        if not isinstance(parent_source, str):
            raise ManifestValidationError("source is required", "parent_src")
        files = [{"path": path, "source": parent_source}]
    normalized["files"] = files

    public_tests = row.get("public_tests")
    if public_tests is None:
        test_source = row.get("test_src")
        if not isinstance(test_source, str):
            public_tests = []
        else:
            test_path = _legacy_test_path(row, fn_text)
            public_tests = [{"path": test_path, "source": test_source}]
    normalized["public_tests"] = public_tests

    task_id = row.get("task_id", row.get("id"))
    if "task_id" in row and not isinstance(task_id, str):
        raise ManifestValidationError("must be a string", "task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = row.get("sha")
    if not isinstance(task_id, str) or not task_id.strip():
        task_id = f"{row.get('repo', 'task')}:{fn_text}:{path}"
    normalized["task_id"] = task_id

    track = row.get("track")
    if "track" in row and not isinstance(track, str):
        raise ManifestValidationError("must be a string", "track")
    if not isinstance(track, str) or not track.strip():
        track = "vue" if _infer_runner(row, public_tests) == "vitest" else "python"
    normalized["track"] = track

    rung = row.get("rung")
    if "rung" in row and not isinstance(rung, str):
        raise ManifestValidationError("must be a string", "rung")
    if not isinstance(rung, str) or not rung.strip():
        family = row.get("family")
        rung = {"build_short": "P1", "build_medium": "P2"}.get(
            family if isinstance(family, str) else "", "LEGACY"
        )
    normalized["rung"] = rung

    world = row.get("world")
    if "world" in row and not isinstance(world, str):
        raise ManifestValidationError("must be a string", "world")
    if not isinstance(world, str) or not world.strip():
        world = "synthetic" if row.get("repo") == "build" else "git"
    normalized["world"] = world

    base_digest = row.get("base_digest")
    if "base_digest" in row and not isinstance(base_digest, str):
        raise ManifestValidationError("must be a string", "base_digest")
    if not isinstance(base_digest, str) or not base_digest.strip():
        base_digest = row.get("sha") or row.get("provenance_sha") or task_id
    if not isinstance(base_digest, str) or not base_digest.strip():
        raise ManifestValidationError("must be a non-empty string", "base_digest")
    normalized["base_digest"] = base_digest

    runner = row.get("runner")
    if "runner" in row and not isinstance(runner, str):
        raise ManifestValidationError("must be a string", "runner")
    normalized["runner"] = (
        runner if isinstance(runner, str) else _infer_runner(row, public_tests)
    )

    requires_db = row.get("requires_db")
    if "requires_db" in row and not isinstance(requires_db, bool):
        raise ManifestValidationError("must be a boolean", "requires_db")
    if not isinstance(requires_db, bool):
        requires_db = _default_requires_db(row, world, normalized["runner"])
    normalized["requires_db"] = requires_db

    selectors = row.get("selectors")
    if selectors is None:
        selectors = [item["path"] for item in public_tests if isinstance(item, Mapping)]
    normalized["selectors"] = selectors

    allowed = row.get("allowed_product_paths")
    if allowed is None:
        allowed = _derive_allowed_paths(path, files)
    normalized["allowed_product_paths"] = allowed

    return normalized


def _private_mapping(row: Mapping[str, object]) -> dict[str, object]:
    private: dict[str, object] = {}
    nested = row.get("private", row.get("oracle", row.get("private_oracle")))
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ManifestValidationError("must be an object", "private")
        private.update({str(key): value for key, value in nested.items()})
    for key in PRIVATE_FIELDS:
        if key in row:
            private[key] = row[key]
    return private


def _first_legacy_reference(row: Mapping[str, object]) -> str | None:
    for key in _LEGACY_PRIVATE_FIELDS:
        value = row.get(key)
        if isinstance(value, str):
            return value
    return None


def _legacy_product_path(row: Mapping[str, object], files: object = None) -> str:
    path = row.get("path")
    if "path" in row and not isinstance(path, str):
        raise ManifestValidationError("must be a string", "path")
    validated_path = (
        _validate_relative_path(path, "path")
        if isinstance(path, str) and path.strip()
        else None
    )
    rel = row.get("rel")
    if "rel" in row and not isinstance(rel, str):
        raise ManifestValidationError("must be a string", "rel")
    # The legacy synthetic branches deliberately use app/<rel>; `path` is a
    # redundant corpus field and must not change that historical world layout.
    if row.get("repo") == "build" and isinstance(rel, str) and rel.strip():
        normalized_rel = rel.replace("\\", "/")
        if normalized_rel.startswith("./"):
            normalized_rel = normalized_rel[2:]
        return _validate_relative_path(f"app/{normalized_rel}", "rel")
    if validated_path is not None:
        return validated_path
    if isinstance(rel, str) and rel.strip():
        normalized_rel = rel.replace("\\", "/")
        if normalized_rel.startswith("./"):
            normalized_rel = normalized_rel[2:]
        return _validate_relative_path(f"app/{normalized_rel}", "rel")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        first = next(iter(files), None)
        if isinstance(first, Mapping) and isinstance(first.get("path"), str):
            return cast(str, first["path"])
        if not files:
            return ""
    raise ManifestValidationError("path, rel, or files[0].path is required", "path")


def _legacy_test_path(row: Mapping[str, object], fn: str) -> str:
    del fn
    from build_test_path import resolve_build_test_path

    return resolve_build_test_path(row)


def _infer_runner(row: Mapping[str, object], public_tests: object) -> RunnerName:
    raw_track = row.get("track")
    if raw_track == "vue":
        return "vitest"
    if isinstance(public_tests, Sequence) and not isinstance(
        public_tests, (str, bytes)
    ):
        for value in public_tests:
            if isinstance(value, Mapping):
                path = value.get("path")
                if isinstance(path, str) and path.endswith(
                    (".js", ".jsx", ".ts", ".tsx")
                ):
                    return "vitest"
    return "pytest"


def _default_requires_db(
    row: Mapping[str, object], world: object, runner: object
) -> bool:
    if row.get("repo") == "build" or row.get("no_db") is True:
        return False
    if world in {"synthetic", "answer", "planning", "no-db"}:
        return False
    if runner == "vitest" and row.get("track") == "vue":
        return False
    return True


def _derive_allowed_paths(path: str, files: object) -> list[str]:
    paths: list[str] = [path]
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        paths = [
            item.get("path")
            for item in files
            if isinstance(item, Mapping) and isinstance(item.get("path"), str)
        ] or paths
    prefixes = sorted({item.split("/", 1)[0] + "/" for item in paths if "/" in item})
    return prefixes or ["app/"]


def _file_tuple(value: object, field: str) -> tuple[TaskFile, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestValidationError("must be an array", field)
    return tuple(
        TaskFile.from_value(item, f"{field}[{index}]")
        for index, item in enumerate(value)
    )


def _reject_duplicate_paths(items: Sequence[TaskFile], field: str) -> None:
    seen: set[str] = set()
    for index, item in enumerate(items):
        if item.path in seen:
            raise ManifestValidationError("duplicate path", f"{field}[{index}].path")
        seen.add(item.path)


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestValidationError("must be an array", field)
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ManifestValidationError(
                "items must be non-empty strings", f"{field}[{index}]"
            )
        result.append(item)
    return tuple(result)


def _allowed_paths(value: object) -> tuple[str, ...]:
    values = _string_tuple(value, "allowed_product_paths")
    normalized: list[str] = []
    for index, item in enumerate(values):
        path = _validate_relative_path(
            item.rstrip("/"), f"allowed_product_paths[{index}]"
        )
        normalized.append(path + "/")
    return tuple(dict.fromkeys(normalized))


def _runner(value: object) -> RunnerName:
    if not isinstance(value, str) or value not in ALLOWED_RUNNERS:
        raise ManifestValidationError(
            f"must be one of {', '.join(sorted(ALLOWED_RUNNERS))}", "runner"
        )
    return cast(RunnerName, value)


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError("must be a non-empty string", field)
    return value


def _required_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestValidationError("must be a boolean", field)
    return value


def _validate_relative_path(value: str, field: str) -> str:
    if not value or "\x00" in value:
        raise ManifestValidationError("must be a non-empty safe relative path", field)
    if value.startswith(("/", "\\")) or ":" in value.split("/", 1)[0]:
        raise ManifestValidationError("must be relative", field)
    parts = value.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ManifestValidationError("contains an unsafe path component", field)
    return "/".join(parts)


def _validate_selector(value: str, field: str) -> None:
    if "\x00" in value:
        raise ManifestValidationError("contains NUL", field)
    base = value.split("::", 1)[0].strip("`'\"")
    if not base or base.startswith(("/", "\\")):
        raise ManifestValidationError("must identify a relative test", field)
    if "/" in base or base.endswith((".py", ".js", ".jsx", ".ts", ".tsx")):
        normalized = base.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        _validate_relative_path(normalized, field)


def _path_is_allowed(path: str, allowed: Sequence[str]) -> bool:
    return any(path.startswith(prefix) for prefix in allowed)


def _reject_private_public_keys(raw: Mapping[str, object]) -> None:
    leaked = sorted((PRIVATE_FIELDS | _LEGACY_PRIVATE_FIELDS) & set(raw))
    if leaked:
        raise ManifestValidationError(
            f"private oracle field(s) are not allowed in public payload: {', '.join(leaked)}"
        )
