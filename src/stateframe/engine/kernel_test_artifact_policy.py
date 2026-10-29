"""Host-owned policy for test-run files that must not alter product identity.

The policy is derived from the effective pytest/coverage configuration and the
observed write set of the current host run.  It deliberately does not use a
global filename blocklist: a source or test file with an unusual name remains
part of the binding unless the host has declared it as a test artifact.
"""
from __future__ import annotations

import configparser
import copy
import hashlib
import json
import os
import shlex
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path


SCHEMA = "stateframe-test-artifact-policy/1"
_PYTEST_CONFIG_NAMES = ("pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")
_COVERAGE_CONFIG_NAMES = (".coveragerc", "coverage.ini", "setup.cfg", "tox.ini", "pyproject.toml")
_COVERAGE_SWITCHES = frozenset({
    "--cov", "--cov-branch", "--cov-context", "--cov-fail-under",
    "--cov-report", "--cov-config", "--no-cov-on-fail",
})
_COVERAGE_REPORT_DEFAULTS = {
    "html": "htmlcov",
    "xml": "coverage.xml",
    "json": "coverage.json",
    "lcov": "coverage.lcov",
    "annotate": "coverage-annotate",
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _normal_relative(root: Path, value: object) -> str | None:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw or raw in {"-", ":"}:
        return None
    raw = os.path.expandvars(raw)
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, RuntimeError, ValueError):
            return None
    try:
        relative = (root / candidate).resolve().relative_to(root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None
    return relative or "."


def _tokens(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            return tuple(shlex.split(value))
        except ValueError:
            return tuple(value.split())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return ()


def _config_value(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(str(item) for item in value)
    return str(value or "")


def _normal_path(value: object) -> str:
    path = str(value).replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path


@dataclass(frozen=True, slots=True)
class ArtifactDeclaration:
    """One host declaration for an exact path or a generated path root."""

    path: str
    kind: str
    source: str
    recursive: bool = False
    prefix: bool = False

    def matches(self, relative: str) -> bool:
        relative = _normal_path(relative)
        if self.prefix:
            return relative.startswith(self.path)
        if self.recursive:
            if relative == self.path or relative.startswith(self.path + "/"):
                return True
            # Python and pytest create these caches beside every source/test
            # package, not only at repository root.  They are generated output
            # wherever the matching cache directory appears in the tree.
            if self.path in {"__pycache__", ".pytest_cache"}:
                return self.path in relative.split("/")
            return False
        return relative == self.path

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "source": self.source,
            "recursive": self.recursive,
            "prefix": self.prefix,
        }


@dataclass(frozen=True, slots=True)
class TestArtifactPolicy:
    """Effective test-output declarations used by all binding consumers."""

    declarations: tuple[ArtifactDeclaration, ...] = ()
    pytest_config: str | None = None
    coverage_requested: bool = False
    coverage_disabled: bool = False
    sanitized_addopts: tuple[str, ...] = ()

    def matches(self, path: object) -> bool:
        relative = _normal_path(path)
        return any(item.matches(relative) for item in self.declarations)

    def generated_write_paths(self, write_set: Iterable[object]) -> tuple[str, ...]:
        return tuple(sorted({
            _normal_path(path)
            for path in write_set
            if self.matches(path)
        }))

    def with_coverage_disabled(self) -> "TestArtifactPolicy":
        return replace(self, coverage_disabled=True)

    def to_dict(self, *, observed_write_set: Iterable[object] = ()) -> dict[str, object]:
        observed = tuple(sorted({_normal_path(item)
                                 for item in observed_write_set if str(item).strip()}))
        generated = self.generated_write_paths(observed)
        return {
            "schema": SCHEMA,
            "declarations": [item.to_dict() for item in self.declarations],
            "pytest_config": self.pytest_config,
            "coverage_requested": self.coverage_requested,
            "coverage_disabled": self.coverage_disabled,
            "observed_write_set": list(observed),
            "ignored_generated_paths": list(generated),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "TestArtifactPolicy | None":
        if not isinstance(value, Mapping):
            return None
        declarations = _declarations_from_mapping(value)
        return cls(
            declarations=declarations,
            pytest_config=(str(value["pytest_config"])
                           if value.get("pytest_config") not in (None, "") else None),
            coverage_requested=value.get("coverage_requested") is True,
            coverage_disabled=value.get("coverage_disabled") is True,
        )


def _declarations_from_mapping(value: Mapping[str, object]) -> tuple[ArtifactDeclaration, ...]:
    declarations: list[ArtifactDeclaration] = []
    raw = value.get("declarations", ())
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for item in raw:
            if not isinstance(item, Mapping) or not str(item.get("path") or "").strip():
                continue
            declarations.append(ArtifactDeclaration(
                path=str(item["path"]), kind=str(item.get("kind") or "test_output"),
                source=str(item.get("source") or "host"),
                recursive=item.get("recursive") is True,
                prefix=item.get("prefix") is True,
            ))
    return tuple(declarations)


def _add(declarations: list[ArtifactDeclaration], *, root: Path, value: object,
         kind: str, source: str, recursive: bool = False, prefix: bool = False) -> None:
    path = _normal_relative(root, value)
    if path is None or path == ".":
        return
    declaration = ArtifactDeclaration(path, kind, source, recursive, prefix)
    if declaration not in declarations:
        declarations.append(declaration)


def _pytest_arguments(tokens: Sequence[str]) -> list[str]:
    """Tokens AFTER the pytest program position (`pytest`/`py.test` executable or `-m pytest`); [] when pytest is not invoked."""
    items = [str(token) for token in tokens]
    for index, token in enumerate(items):
        if token == "-m" and index + 1 < len(items) and items[index + 1] == "pytest":
            return items[index + 2:]
        if Path(token).name in {"pytest", "py.test"}:
            return items[index + 1:]
    return []


def _explicit_config(root: Path, tokens: Sequence[str]) -> Path | None:
    """The ``-c``/``--config-file`` pytest will actually use, when given.

    Audit 2026-10-05 #2: the live check passes ``-c backend/pytest.ini`` (the
    nearest subproject config); reading only ``<root>/pytest.ini`` missed its
    ``--cov ... --cov-report=html`` addopts, so coverage was never disabled.
    """
    # 10-22 (census b1542/b1606 'File name too long'): only PYTEST's own arguments count. A profiled command such as
    # `python -c "<wrapper>" pytest -c backend/pytest.ini` has Python's -c first; scanning the whole argv returned the wrapper
    # body as the config path (ENAMETOOLONG in _pytest_config). Same own-program rule as the machine-wide pytest detectors.
    tokens = _pytest_arguments(tokens)
    for index, token in enumerate(tokens):
        value = None
        if token in {"-c", "--config-file"} and index + 1 < len(tokens):
            value = tokens[index + 1]
        elif token.startswith("--config-file="):
            value = token.split("=", 1)[1]
        if value:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else root / candidate
    return None


def _pytest_config(root: Path, explicit: Path | None = None,
                   ) -> tuple[Path | None, dict[str, object]]:
    """Read the effective pytest config without importing project code."""
    if explicit is not None and not explicit.is_file():
        return None, {}  # e.g. ``-c /dev/null``: no config, no addopts
    for name in ((explicit.name,) if explicit is not None else _PYTEST_CONFIG_NAMES):
        path = explicit if explicit is not None else root / name
        if not path.is_file():
            continue
        if path.name == "pyproject.toml":
            try:
                import tomllib
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                values = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
                return path, dict(values) if isinstance(values, Mapping) else {}
            except (OSError, UnicodeError, ValueError, ImportError, AttributeError):
                return path, {}
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(path, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeError):
            return path, {}
        for section in ("pytest", "tool:pytest"):
            if parser.has_section(section):
                return path, {key: value for key, value in parser.items(section)}
    return None, {}


def _coverage_config(root: Path) -> tuple[dict[str, object], tuple[str, ...]]:
    values: dict[str, object] = {}
    paths: list[str] = []
    for name in _COVERAGE_CONFIG_NAMES:
        path = root / name
        if not path.is_file():
            continue
        if path.name == "pyproject.toml":
            try:
                import tomllib
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                tool = data.get("tool", {})
                coverage = tool.get("coverage", {}) if isinstance(tool, Mapping) else {}
                if isinstance(coverage, Mapping):
                    if not coverage:
                        continue
                    paths.append(name)
                    for section, section_values in coverage.items():
                        if isinstance(section_values, Mapping):
                            values.update({f"{section}.{key}": item
                                           for key, item in section_values.items()})
                continue
            except (OSError, UnicodeError, ValueError, ImportError, AttributeError):
                continue
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(path, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeError):
            continue
        # ``setup.cfg``/``tox.ini`` namespace coverage sections as
        # ``coverage:run``; standalone ``.coveragerc``/``coverage.ini`` use
        # the bare coverage section names.  Read both forms so data_file,
        # parallel, and html.directory are host declarations rather than
        # accidental filename heuristics.
        coverage_sections = [section for section in parser.sections()
                             if section.startswith("coverage:")]
        if not coverage_sections and path.name in {".coveragerc", "coverage.ini"}:
            coverage_sections = [section for section in parser.sections()
                                 if section in {
                                     "run", "report", "html", "xml", "json",
                                     "lcov", "annotate", "paths", "setup",
                                 }]
        if not coverage_sections:
            continue
        paths.append(name)
        for section in parser.sections():
            if section in coverage_sections:
                normalized_section = section.removeprefix("coverage:")
                values.update({f"{normalized_section}.{key}": value
                               for key, value in parser.items(section)})
    return values, tuple(paths)


def _coverage_arg(token: str) -> bool:
    return token == "--cov" or token.startswith("--cov=") or token.startswith("--cov-")


def _strip_coverage_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    """Remove only pytest-cov options, preserving test selection and other flags."""
    result: list[str] = []
    skip_next = False
    takes_value = {"--cov", "--cov-report", "--cov-config", "--cov-context", "--cov-fail-under"}
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in takes_value:
            skip_next = True
            continue
        if _coverage_arg(token) or token in {"--no-cov-on-fail"}:
            continue
        result.append(token)
    return tuple(result)


def discover_test_artifact_policy(
        root: str | Path, *, command: object = (),
        environment: Mapping[str, str] | None = None) -> TestArtifactPolicy:
    """Discover generated paths from effective config and test-run arguments."""
    root_path = Path(root).resolve()
    env = dict(os.environ if environment is None else environment)
    command_tokens = _tokens(command)
    explicit_config = _explicit_config(root_path, command_tokens)
    config_path, pytest_values = _pytest_config(root_path, explicit_config)
    config_addopts = _tokens(pytest_values.get("addopts", ()))
    env_addopts = _tokens(env.get("PYTEST_ADDOPTS", ()))
    effective = (*config_addopts, *env_addopts, *command_tokens)
    declarations: list[ArtifactDeclaration] = []

    # These declarations come from pytest/Python execution defaults, not from
    # a filename blocklist.  The cache provider can be disabled by the runner,
    # but retaining this fallback keeps an externally launched pytest honest.
    _add(declarations, root=root_path, value=pytest_values.get("cache_dir", ".pytest_cache"),
         kind="pytest_cache", source="pytest-effective-cache_dir", recursive=True)
    _add(declarations, root=root_path, value="__pycache__",
         kind="python_bytecode", source="python-test-run-default", recursive=True)

    coverage_values, coverage_files = _coverage_config(root_path)
    coverage_requested = any(_coverage_arg(token) for token in effective) or bool(coverage_files)
    coverage_data_file = env.get("COVERAGE_FILE") or coverage_values.get("run.data_file")
    if coverage_requested:
        data_path = coverage_data_file or ".coverage"
        _add(declarations, root=root_path, value=data_path,
             kind="coverage_data", source=("COVERAGE_FILE" if env.get("COVERAGE_FILE")
                                            else "coverage-effective-data_file"))
        parallel = str(coverage_values.get("run.parallel", "false")).casefold() in {
            "true", "1", "yes", "on",
        }
        if parallel:
            _add(declarations, root=root_path, value=data_path + ".",
                 kind="coverage_parallel_data", source="coverage.run.parallel", prefix=True)
        for index, token in enumerate(effective):
            if not token.startswith("--cov-report"):
                continue
            report = token.split("=", 1)[1] if "=" in token else (
                effective[index + 1] if index + 1 < len(effective) else "")
            report_kind, _, report_path = report.partition(":")
            default = _COVERAGE_REPORT_DEFAULTS.get(report_kind)
            _add(declarations, root=root_path, value=report_path or default,
                 kind=f"coverage_{report_kind or 'report'}",
                 source="pytest-cov-effective-cov-report", recursive=report_kind in {"html", "annotate"})
        html_directory = coverage_values.get("html.directory")
        if html_directory is not None:
            _add(declarations, root=root_path, value=html_directory,
                 kind="coverage_html", source="coverage.html.directory", recursive=True)
    for index, token in enumerate(effective):
        if token.startswith("--junitxml="):
            _add(declarations, root=root_path, value=token.split("=", 1)[1],
                 kind="junit_xml", source="pytest-effective-junitxml")
        elif token == "--junitxml" and index + 1 < len(effective):
            _add(declarations, root=root_path, value=effective[index + 1],
                 kind="junit_xml", source="pytest-effective-junitxml")
        if ("junitxml=" in token or
                (token == "-o" and index + 1 < len(effective)
                 and "junitxml=" in effective[index + 1])):
            junit_token = token if "junitxml=" in token else effective[index + 1]
            _add(declarations, root=root_path, value=junit_token.split("junitxml=", 1)[1],
                 kind="junit_xml", source="pytest-effective-junitxml")
    for key in ("junitxml", "junit_xml"):
        if key in pytest_values:
            _add(declarations, root=root_path, value=pytest_values[key],
                 kind="junit_xml", source="pytest-effective-junitxml")
    return TestArtifactPolicy(
        declarations=tuple(declarations),
        pytest_config=(
            (config_path.relative_to(root_path).as_posix()
             if config_path.is_relative_to(root_path) else config_path.as_posix())
            if config_path else None),
        coverage_requested=coverage_requested,
        sanitized_addopts=_strip_coverage_tokens(config_addopts),
    )


@dataclass(frozen=True, slots=True)
class _CompiledArtifactMatcher:
    """Optimized matcher built only by the opted-in normalization path."""

    declarations: tuple[ArtifactDeclaration, ...]
    _exact_paths: frozenset[str] = field(init=False, repr=False, compare=False)
    _prefixes: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _recursive_roots: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _recursive_cache_dirs: frozenset[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        exact: set[str] = set()
        prefixes: list[str] = []
        recursive: list[str] = []
        cache_dirs: set[str] = set()
        for declaration in self.declarations:
            if declaration.prefix:
                prefixes.append(declaration.path)
            elif declaration.recursive:
                if declaration.path in {"__pycache__", ".pytest_cache"}:
                    cache_dirs.add(declaration.path)
                else:
                    recursive.append(declaration.path)
            else:
                exact.add(declaration.path)
        object.__setattr__(self, "_exact_paths", frozenset(exact))
        object.__setattr__(self, "_prefixes", tuple(prefixes))
        object.__setattr__(self, "_recursive_roots", tuple(recursive))
        object.__setattr__(self, "_recursive_cache_dirs", frozenset(cache_dirs))

    def matches(self, path: object) -> bool:
        relative = _normal_path(path)
        if any(relative.startswith(prefix) for prefix in self._prefixes):
            return True
        if any(relative == root or relative.startswith(root + "/")
               for root in self._recursive_roots):
            return True
        if self._recursive_cache_dirs:
            components = relative.split("/")
            if any(name in components for name in self._recursive_cache_dirs):
                return True
        return relative in self._exact_paths


_FILTERED_BINDING_CACHE_MAX_BYTES = 8 * 1024 * 1024
_FILTERED_BINDING_CACHE_MAX_ENTRIES = 256
_FILTERED_BINDING_CACHE_LOCK = threading.RLock()
_FILTERED_BINDING_CACHE: OrderedDict[
    tuple[str, str, tuple[ArtifactDeclaration, ...]],
    tuple[dict[str, object], str, int],
] = OrderedDict()
_FILTERED_BINDING_CACHE_BYTES = 0
_FILTERED_BINDING_CACHE_STATS = {"hits": 0, "misses": 0}


def _legacy_filtered_group(
        files: Mapping[str, object], declarations: Sequence[ArtifactDeclaration],
) -> tuple[dict[str, object], str]:
    filtered = {str(path): value for path, value in files.items()
                if not any(declaration.matches(str(path))
                           for declaration in declarations)}
    return filtered, _digest(filtered)


def _filtered_group_cache_key(
        group: str, files: Mapping[str, object],
        declarations: tuple[ArtifactDeclaration, ...],
) -> tuple[str, str, tuple[ArtifactDeclaration, ...]] | None:
    try:
        # Recompute from this input: file maps are mutable and stored revisions
        # or object identities cannot prove that their content is unchanged.
        group_digest = _digest(files)
    except Exception:
        return None
    return group, group_digest, declarations


def _filtered_group_cache_get(
        key: tuple[str, str, tuple[ArtifactDeclaration, ...]],
) -> tuple[dict[str, object], str] | None:
    with _FILTERED_BINDING_CACHE_LOCK:
        entry = _FILTERED_BINDING_CACHE.get(key)
        if entry is None:
            _FILTERED_BINDING_CACHE_STATS["misses"] += 1
            return None
        _FILTERED_BINDING_CACHE.move_to_end(key)
    filtered, revision, _size = entry
    try:
        detached = copy.deepcopy(filtered)
    except Exception:
        with _FILTERED_BINDING_CACHE_LOCK:
            if _FILTERED_BINDING_CACHE.get(key) is entry:
                _drop_filtered_group_cache_entry(key)
            _FILTERED_BINDING_CACHE_STATS["misses"] += 1
        return None
    with _FILTERED_BINDING_CACHE_LOCK:
        _FILTERED_BINDING_CACHE_STATS["hits"] += 1
    return detached, revision


def _drop_filtered_group_cache_entry(
        key: tuple[str, str, tuple[ArtifactDeclaration, ...]],
) -> None:
    global _FILTERED_BINDING_CACHE_BYTES
    entry = _FILTERED_BINDING_CACHE.pop(key, None)
    if entry is not None:
        _FILTERED_BINDING_CACHE_BYTES -= entry[2]


def _filtered_group_cache_put(
        key: tuple[str, str, tuple[ArtifactDeclaration, ...]],
        filtered: dict[str, object], revision: str,
) -> None:
    global _FILTERED_BINDING_CACHE_BYTES
    try:
        detached = copy.deepcopy(filtered)
        payload_size = len(_canonical({
            "group": key[0], "revision": key[1],
            "declarations": [declaration.to_dict() for declaration in key[2]],
            "files": detached,
        }))
    except Exception:
        return
    if payload_size <= 0 or payload_size > _FILTERED_BINDING_CACHE_MAX_BYTES:
        return
    with _FILTERED_BINDING_CACHE_LOCK:
        _drop_filtered_group_cache_entry(key)
        _FILTERED_BINDING_CACHE[key] = (detached, revision, payload_size)
        _FILTERED_BINDING_CACHE_BYTES += payload_size
        while (_FILTERED_BINDING_CACHE
               and (_FILTERED_BINDING_CACHE_BYTES > _FILTERED_BINDING_CACHE_MAX_BYTES
                    or len(_FILTERED_BINDING_CACHE) > _FILTERED_BINDING_CACHE_MAX_ENTRIES)):
            oldest = next(iter(_FILTERED_BINDING_CACHE))
            _drop_filtered_group_cache_entry(oldest)


def _filtered_group(
        group: str, files: Mapping[str, object], matcher: _CompiledArtifactMatcher,
) -> tuple[dict[str, object], str]:
    try:
        # Hash and filter one detached snapshot so an in-place edit cannot make
        # a cached result disagree with its content-address key.
        source: Mapping[str, object] = copy.deepcopy(dict(files))
        content_verified = True
    except Exception:
        source = files
        content_verified = False
    key = (_filtered_group_cache_key(group, source, matcher.declarations)
           if content_verified else None)
    if key is not None:
        cached = _filtered_group_cache_get(key)
        if cached is not None:
            return cached
    try:
        filtered = {str(path): value for path, value in source.items()
                    if not matcher.matches(path)}
        revision = _digest(filtered)
    except Exception:
        # The optimization is advisory; the legacy matcher owns fallback output.
        filtered, revision = _legacy_filtered_group(source, matcher.declarations)
    if key is not None:
        _filtered_group_cache_put(key, filtered, revision)
    return filtered, revision


def _optimized_filter_binding(
        binding: Mapping[str, object], matcher: _CompiledArtifactMatcher,
) -> dict[str, object]:
    result = dict(binding)
    for group, revision_key in (("product_files", "product_revision"),
                                ("harness_files", "test_harness_revision")):
        files = binding.get(group)
        if not isinstance(files, Mapping):
            continue
        filtered, revision = _filtered_group(group, files, matcher)
        result[group] = filtered
        result[revision_key] = revision
    if "product_revision" in result:
        result["artifact"] = result["product_revision"]
    if "environment_revision" in result:
        result["environment"] = result["environment_revision"]
    return result


def filter_binding(binding: Mapping[str, object], policy: TestArtifactPolicy) -> dict[str, object]:
    """Filter declared outputs with the legacy public helper semantics."""
    return _legacy_filter_binding(binding, policy.declarations)


def _filtered_binding_cache_clear() -> None:
    global _FILTERED_BINDING_CACHE_BYTES
    with _FILTERED_BINDING_CACHE_LOCK:
        _FILTERED_BINDING_CACHE.clear()
        _FILTERED_BINDING_CACHE_BYTES = 0
        _FILTERED_BINDING_CACHE_STATS.update(hits=0, misses=0)


def _filtered_binding_cache_info() -> dict[str, int]:
    with _FILTERED_BINDING_CACHE_LOCK:
        return {
            **_FILTERED_BINDING_CACHE_STATS,
            "entries": len(_FILTERED_BINDING_CACHE),
            "bytes": _FILTERED_BINDING_CACHE_BYTES,
        }


def normalize_public_test_bindings(
        current: Mapping[str, object], receipts: Sequence[object],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Use the opted-in compiled matcher and content-addressed group cache."""
    policies = [
        TestArtifactPolicy.from_mapping(row.get("test_artifact_policy"))
        for row in receipts if isinstance(row, Mapping)
    ]
    declarations = _deduplicate_declarations(
        declaration for policy in policies if policy is not None
        for declaration in policy.declarations)
    matcher = _CompiledArtifactMatcher(declarations=declarations)
    normalized_rows: list[dict[str, object]] = []
    for row in receipts:
        if not isinstance(row, Mapping):
            continue
        normalized = dict(row)
        for field_name in ("binding", "after_binding"):
            binding = row.get(field_name)
            if isinstance(binding, Mapping):
                normalized[field_name] = _optimized_filter_binding(binding, matcher)
        normalized_rows.append(normalized)
    return _optimized_filter_binding(current, matcher), tuple(normalized_rows)


def _deduplicate_declarations(
        declarations: Iterable[ArtifactDeclaration],
) -> tuple[ArtifactDeclaration, ...]:
    result: list[ArtifactDeclaration] = []
    seen: set[bytes] = set()
    for declaration in declarations:
        key = _canonical(declaration.to_dict())
        if key in seen:
            continue
        seen.add(key)
        result.append(declaration)
    return tuple(result)


def _normalize_public_test_bindings_legacy(
        current: Mapping[str, object], receipts: Sequence[object],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Retain the pre-fact normalization path for opt-out and shadow parity."""
    declarations = tuple(
        declaration
        for row in receipts if isinstance(row, Mapping)
        for policy_value in (row.get("test_artifact_policy"),)
        if isinstance(policy_value, Mapping)
        for declaration in _declarations_from_mapping(policy_value)
    )
    normalized_rows: list[dict[str, object]] = []
    for row in receipts:
        if not isinstance(row, Mapping):
            continue
        normalized = dict(row)
        for field_name in ("binding", "after_binding"):
            binding = row.get(field_name)
            if isinstance(binding, Mapping):
                normalized[field_name] = _legacy_filter_binding(binding, declarations)
        normalized_rows.append(normalized)
    return _legacy_filter_binding(current, declarations), tuple(normalized_rows)


def normalize_public_test_bindings_for_mode(
        current: Mapping[str, object], receipts: Sequence[object], mode: str,
) -> tuple[
    dict[str, object], tuple[dict[str, object], ...], tuple[str, ...], str | None,
]:
    """Select pure normalization mode without repeating any effectful checks."""
    if mode not in {"off", "shadow", "on"}:
        mode = "off"
    if mode == "off":
        legacy = _normalize_public_test_bindings_legacy(current, receipts)
        return legacy[0], legacy[1], (), None

    try:
        optimized = normalize_public_test_bindings(current, receipts)
    except Exception:
        legacy = _normalize_public_test_bindings_legacy(current, receipts)
        return legacy[0], legacy[1], (), "optimized_path_error"

    if mode == "on":
        return optimized[0], optimized[1], (), None

    legacy = _normalize_public_test_bindings_legacy(current, receipts)
    try:
        mismatch_fields = _normalization_mismatch_fields(optimized, legacy)
    except Exception:
        mismatch_fields = ("normalization_output",)
    return legacy[0], legacy[1], mismatch_fields, None


def _normalization_mismatch_fields(
        optimized: tuple[dict[str, object], tuple[dict[str, object], ...]],
        legacy: tuple[dict[str, object], tuple[dict[str, object], ...]],
) -> tuple[str, ...]:
    fields: set[str] = set()
    if optimized[0] != legacy[0]:
        fields.add("normalized_current_binding")
    if optimized[1] != legacy[1]:
        fields.add("normalized_receipt_rows")
    if _normalization_digest_signature(optimized) != _normalization_digest_signature(legacy):
        fields.add("digests")
    return tuple(sorted(fields))


def _normalization_digest_signature(
        result: tuple[dict[str, object], tuple[dict[str, object], ...]],
) -> tuple[object, ...]:
    digest_fields = (
        "product_revision", "test_harness_revision", "environment_revision",
        "artifact", "environment",
    )

    def binding_signature(binding: object) -> tuple[tuple[str, object], ...] | None:
        if not isinstance(binding, Mapping):
            return None
        return tuple((field_name, binding[field_name])
                     for field_name in digest_fields if field_name in binding)

    current, rows = result
    row_signatures = tuple(
        (
            binding_signature(row.get("binding")),
            binding_signature(row.get("after_binding")),
        )
        for row in rows
    )
    return binding_signature(current), row_signatures


def _legacy_filter_binding(
        binding: Mapping[str, object], declarations: Sequence[ArtifactDeclaration],
) -> dict[str, object]:
    result = dict(binding)
    for group, revision_key in (("product_files", "product_revision"),
                                ("harness_files", "test_harness_revision")):
        files = binding.get(group)
        if not isinstance(files, Mapping):
            continue
        filtered, revision = _legacy_filtered_group(files, declarations)
        result[group] = filtered
        result[revision_key] = revision
    if "product_revision" in result:
        result["artifact"] = result["product_revision"]
    if "environment_revision" in result:
        result["environment"] = result["environment_revision"]
    return result


def snapshot_workspace(root: str | Path) -> dict[str, str]:
    """Capture all in-root file identities for a host test write-set."""
    root_path = Path(root).resolve()
    snapshot: dict[str, str] = {}
    try:
        paths = sorted(root_path.rglob("*"))
    except (OSError, RuntimeError):
        return snapshot
    for path in paths:
        try:
            relative = path.relative_to(root_path).as_posix()
            if path.is_symlink():
                snapshot[relative] = "symlink:" + str(path.readlink())
            elif path.is_file():
                snapshot[relative] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, RuntimeError, ValueError):
            continue
    return snapshot


def changed_write_set(before: Mapping[str, str], after: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(sorted(path for path in set(before) | set(after)
                        if before.get(path) != after.get(path)))


def binding_stability(
        before: Mapping[str, object], after: Mapping[str, object], *,
        policy: TestArtifactPolicy | None = None,
        write_set: Iterable[object] = ()) -> dict[str, object]:
    """Return typed host diagnostics for a test's before/after binding."""
    active_policy = policy or TestArtifactPolicy()
    filtered_before = filter_binding(before, active_policy)
    filtered_after = filter_binding(after, active_policy)
    changed_paths: list[str] = []
    for group in ("product_files", "harness_files"):
        left = filtered_before.get(group, {})
        right = filtered_after.get(group, {})
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            changed_paths.extend(
                f"{group}:{path}" for path in sorted(set(left) | set(right))
                if left.get(path) != right.get(path))
    changed_fields = sorted({
        str(key) for key in set(filtered_before) | set(filtered_after)
        if key not in {"product_files", "harness_files", "product_revision",
                       "test_harness_revision", "artifact", "environment"}
        and filtered_before.get(key) != filtered_after.get(key)
    })
    changed_paths.extend(f"binding:{key}" for key in changed_fields)
    changed_paths = sorted(set(changed_paths))
    product_changed = any(path.startswith("product_files:") for path in changed_paths)
    harness_changed = any(path.startswith("harness_files:") for path in changed_paths)
    if not changed_paths:
        code = "NO_MEANINGFUL_BINDING_CHANGE"
        status = "STABLE"
    elif product_changed and harness_changed:
        code = "PRODUCT_AND_TEST_HARNESS_CHANGED"
        status = "UNSTABLE"
    elif product_changed:
        code = "PRODUCT_FILES_CHANGED"
        status = "UNSTABLE"
    elif harness_changed:
        code = "TEST_HARNESS_CHANGED"
        status = "UNSTABLE"
    else:
        code = "BINDING_FIELDS_CHANGED"
        status = "UNSTABLE"
    observed = tuple(_normal_path(item) for item in write_set)
    ignored = active_policy.generated_write_paths(observed)
    return {
        "schema": "stateframe-binding-stability/1",
        "status": status,
        "code": code,
        "changed_paths": changed_paths,
        "observed_write_set": sorted(set(observed)),
        "ignored_generated_paths": list(ignored),
        "artifact_policy": active_policy.to_dict(observed_write_set=observed),
    }


def _scratch_coverage_file(root: str | Path) -> str:
    """A per-workspace coverage data path OUTSIDE the solver root."""
    import tempfile
    key = hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / "stateframe-coverage" / key
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / ".coverage")


def prepare_pytest_command(
        command: Sequence[object], root: str | Path, *,
        environment: Mapping[str, str] | None = None,
) -> tuple[tuple[str, ...], dict[str, str], TestArtifactPolicy]:
    """Disable pytest-cov output while preserving non-coverage test options."""
    env = dict(os.environ if environment is None else environment)
    original = tuple(str(item) for item in command)
    policy = discover_test_artifact_policy(root, command=original, environment=env)
    if not policy.coverage_requested:
        return original, env, policy
    sanitized_command = _strip_coverage_tokens(original)
    sanitized_env = dict(env)
    sanitized_env["PYTEST_ADDOPTS"] = " ".join(
        _strip_coverage_tokens(_tokens(env.get("PYTEST_ADDOPTS", ())))
    )
    # Defensive check: anything that still starts coverage
    # (a conftest, a plugin) writes its data file to scratch, never the root.
    sanitized_env["COVERAGE_FILE"] = _scratch_coverage_file(root)
    if not any(token == "no:cov" for token in sanitized_command):
        sanitized_command = (*sanitized_command, "-p", "no:cov")
    # pytest's command-line override clears config addopts; the non-coverage
    # options are then replayed explicitly, preserving selection/strictness.
    if policy.sanitized_addopts:
        sanitized_command = (*sanitized_command, "-o", "addopts=", *policy.sanitized_addopts)
    else:
        sanitized_command = (*sanitized_command, "-o", "addopts=")
    return sanitized_command, sanitized_env, policy.with_coverage_disabled()


def policy_from_binding(binding: Mapping[str, object]) -> TestArtifactPolicy | None:
    return TestArtifactPolicy.from_mapping(binding.get("test_artifact_policy"))


__all__ = [
    "ArtifactDeclaration", "SCHEMA", "TestArtifactPolicy", "binding_stability",
    "changed_write_set", "discover_test_artifact_policy", "filter_binding",
    "policy_from_binding", "prepare_pytest_command", "snapshot_workspace",
]
