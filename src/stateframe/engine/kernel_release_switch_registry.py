"""Derive the release's readable STATEFRAME switches from shipped Python sources."""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import stat
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Mapping

SCHEMA = "stateframe-switch-registry/1"
# Host setup and retention switches are read before engine creation and are never
# handed to the engine child, even when their literals appear in engine modules.
HOST_ONLY_SWITCHES = frozenset({
    "STATEFRAME_ENDPOINT_MAX_JSON_BYTES",
    "STATEFRAME_ENV_SEED",
    "STATEFRAME_REQUIRED_TEST_LEDGER",
})
# Compatibility spellings are normalized at public boundaries, not registered
# engine switches. Keep this exemption exact and guarded by the namespace tests.
DEPRECATED_SWITCH_ALIASES = frozenset({"STATEFRAME_ROUTE_BUDGET"})
# Removed switches: a host that still exports one is told it is ignored, never refused.
# The source scanner skips this table by name so these keys never register as readers.
RETIRED_SWITCHES = {
    "STATEFRAME_READ_LOCKOUT": "removed in this release: the engine never hides read routes",
}
RETIRED_TABLE_NAME = "RETIRED_SWITCHES"


class StateFrameSwitchPreflightError(RuntimeError):
    """A configured STATEFRAME switch has no reader in this declared release."""

    outcome = "HARNESS_CONTRACT"
    failure_origin = "harness"
    harness_contract = True
    transport_retryable = False
    transport_class = "contract"

    def __init__(self, detail: Mapping[str, object], *, journal_path: Path):
        self.detail = dict(detail)
        self.journal_path = journal_path
        super().__init__("unconsumed STATEFRAME settings: " + ", ".join(
            str(name) for name in self.detail.get("unknown_switches", ())))


def _target_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(name for child in node.elts for name in _target_names(child))
    return ()


def _literal_container_switches(node: ast.AST) -> set[str]:
    """Collect switch names from static container definitions such as FLAGS maps."""
    if isinstance(node, ast.Constant):
        return ({node.value} if isinstance(node.value, str)
                and node.value.startswith("STATEFRAME_") else set())
    if isinstance(node, ast.Dict):
        children = (*node.keys, *node.values)
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        children = tuple(node.elts)
    else:
        return set()
    return set().union(*(_literal_container_switches(child) for child in children))


class _SwitchReaderVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.constants: dict[str, str] = {}
        self.container_switches: dict[str, set[str]] = {}
        self.loaded_names: set[str] = set()
        self.direct_reads: set[str] = set()
        self.definition_reads: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded_names.add(node.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value.startswith("STATEFRAME_"):
                for target in node.targets:
                    for name in _target_names(target):
                        self.constants[name] = value.value
        else:
            switches = _literal_container_switches(value)
            if switches:
                for target in node.targets:
                    for name in _target_names(target):
                        # Retired names are listed for reporting only; they are not readers.
                        if name != RETIRED_TABLE_NAME:
                            self.container_switches[name] = switches
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if value.value.startswith("STATEFRAME_"):
                for name in _target_names(node.target):
                    self.constants[name] = value.value
        else:
            switches = _literal_container_switches(value)
            if switches:
                for name in _target_names(node.target):
                    # Retired names are listed for reporting only; they are not readers.
                    if name != RETIRED_TABLE_NAME:
                        self.container_switches[name] = switches
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        function_name = (
            function.attr if isinstance(function, ast.Attribute)
            else function.id if isinstance(function, ast.Name) else ""
        )
        if function_name in {"get", "getenv", "_switch", "SwitchRef"} and node.args:
            key = node.args[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value.startswith("STATEFRAME_"):
                    self.direct_reads.add(key.value)
        if function_name == "LeanStackSlot" and len(node.args) >= 2:
            key = node.args[1]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value.startswith("STATEFRAME_"):
                    self.definition_reads.add(key.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            if key.value.startswith("STATEFRAME_"):
                self.direct_reads.add(key.value)
        self.generic_visit(node)

    def switches(self) -> set[str]:
        return (
            self.direct_reads
            | self.definition_reads
            | {value for name, value in self.constants.items() if name in self.loaded_names}
            | {value for name, values in self.container_switches.items()
               if name in self.loaded_names for value in values}
        )


def _declared_python_files(root: Path) -> tuple[Path, ...]:
    declarations_path = root / "release_declarations.json"
    if not declarations_path.is_file():
        # 10-13: an ASSEMBLED runtime release does not carry the source tree's
        # release_declarations.json (release_assembly consumes it); every live run on a
        # release with this registry died at StateFrameRuntime() with FileNotFoundError
        # while source-tree smokes passed. The release's own manifest lists exactly the
        # assembled files, which already reflect the declarations. Neither file: fail closed.
        manifest_path = root / "release-manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"neither release_declarations.json nor release-manifest.json under {root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(listed, dict) or not listed:
            raise ValueError(f"release-manifest.json under {root} lists no files")
        return tuple(sorted(
            root / name for name in listed
            if isinstance(name, str) and name.endswith(".py") and (root / name).is_file()
        ))
    declarations = json.loads(declarations_path.read_text(encoding="utf-8"))
    patterns = declarations.get("python_modules", ())
    exclusions = tuple(str(item) for item in declarations.get("excluded_paths", ()))
    files: set[Path] = set()
    for raw_pattern in patterns:
        pattern = str(raw_pattern)
        for path in root.glob(pattern):
            if not path.is_file() or path.suffix != ".py":
                continue
            relative = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatchcase(relative, excluded) for excluded in exclusions):
                continue
            files.add(path)
    return tuple(sorted(files))


# The shipped and cached registries both carry a fingerprint of the exact source
# bytes. Checking it is much cheaper than parsing all the declared modules.
REGISTRY_FILE = "switch-registry.json"
_MAX_REGISTRY_BYTES = 1024 * 1024


def _sources_fingerprint(root: Path, files: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii") + b"\n")
    return "sha256:" + digest.hexdigest()


def _switch_digest(switches: frozenset[str]) -> str:
    encoded = json.dumps(sorted(switches), separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _scan(files: tuple[Path, ...]) -> frozenset[str]:
    switches: set[str] = set()
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            continue
        reader = _SwitchReaderVisitor()
        reader.visit(tree)
        switches.update(reader.switches())
    return frozenset(switches - HOST_ONLY_SWITCHES - DEPRECATED_SWITCH_ALIASES)


def _report_registry_fallback(cause: str, why: str) -> None:
    try:
        from kernel_reason_log import emit_reason
    except ImportError:
        from .kernel_reason_log import emit_reason
    emit_reason("fallback", cause, "switch_registry", "could not reuse the switch registry", why,
                level="DEBUG", journal=False)


def _read_registry(path: Path, fingerprint: str) -> tuple[frozenset[str], str] | None:
    """Read one bounded, regular registry file. A missing cache entry is normal."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("registry is not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(_MAX_REGISTRY_BYTES + 1)
        finally:
            os.close(fd)
        if len(data) > _MAX_REGISTRY_BYTES:
            raise ValueError("registry is too large")
        document = json.loads(data.decode("utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as error:
        _report_registry_fallback("switch_registry_read_failed", type(error).__name__)
        return None
    listed = document.get("switches") if isinstance(document, dict) else None
    if (not isinstance(listed, list)
            or not all(isinstance(name, str) and name.startswith("STATEFRAME_") for name in listed)):
        _report_registry_fallback("switch_registry_invalid", "invalid switch rows")
        return None
    switches = frozenset(listed)
    digest = _switch_digest(switches)
    if document.get("schema") != SCHEMA or document.get("digest") != digest:
        _report_registry_fallback("switch_registry_invalid", "schema or digest mismatch")
        return None
    return (switches, digest) if document.get("sources_sha256") == fingerprint else None


def _registry_cache_path(root: Path) -> Path | None:
    # The host and child inherit the same HOME, even with a run-specific state directory.
    home = Path(os.environ.get("HOME", ""))
    if not home.is_absolute():
        return None
    # Keep one entry per checkout and Python parser, replacing it on source edits.
    identity = json.dumps([str(root), sys.implementation.name, sys.version_info[:3]])
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return home / ".cache" / "stateframe" / "engine" / f"switch-registry-v1-{key}.json"


def _write_cached_registry(path: Path, fingerprint: str, switches: frozenset[str], digest: str) -> None:
    temporary = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"schema": SCHEMA, "sources_sha256": fingerprint,
                       "switches": sorted(switches), "digest": digest}, stream)
        temporary.replace(path)
    except OSError as error:
        _report_registry_fallback("switch_registry_write_failed", type(error).__name__)
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                _report_registry_fallback("switch_registry_cleanup_failed", type(cleanup_error).__name__)


def build_registry_document(root: Path) -> dict[str, object]:
    """The registry for ``root`` computed by a full parse: what switch-registry.json must contain."""
    root = Path(root).resolve()
    files = _declared_python_files(root)
    switches = _scan(files)
    return {"schema": SCHEMA, "sources_sha256": _sources_fingerprint(root, files),
            "switches": sorted(switches), "digest": _switch_digest(switches)}


def write_registry(root: Path) -> Path:
    """Regenerate switch-registry.json for a release (run after any engine source change)."""
    path = Path(root).resolve() / REGISTRY_FILE
    path.write_text(json.dumps(build_registry_document(root), indent=1) + "\n", encoding="utf-8")
    return path


@lru_cache(maxsize=4)
def _release_registry(root_text: str) -> tuple[frozenset[str], str]:
    root = Path(root_text).resolve()
    files = _declared_python_files(root)
    try:
        fingerprint = _sources_fingerprint(root, files)
    except OSError as error:
        _report_registry_fallback("switch_registry_fingerprint_failed", type(error).__name__)
        fingerprint = None
    cache = _registry_cache_path(root) if fingerprint is not None else None
    if fingerprint is not None:
        shipped = _read_registry(root / REGISTRY_FILE, fingerprint)
        if shipped is not None:
            return shipped
        if cache is not None:
            cached = _read_registry(cache, fingerprint)
            if cached is not None:
                return cached
    frozen = _scan(files)
    digest = _switch_digest(frozen)
    if cache is not None:
        # Publish only when the declared sources still match the pre-scan fingerprint.
        try:
            unchanged = _sources_fingerprint(root, _declared_python_files(root)) == fingerprint
        except (OSError, ValueError) as error:
            _report_registry_fallback("switch_registry_fingerprint_failed", type(error).__name__)
            unchanged = False
        if unchanged:
            _write_cached_registry(cache, fingerprint, frozen, digest)
        else:
            _report_registry_fallback("switch_registry_publish_skipped", "could not verify scanned sources")
    return frozen, digest


def release_stateframe_switches(root: Path | None = None) -> tuple[frozenset[str], str]:
    """Return switches read by code declared in this release and its registry digest."""
    selected_root = root or Path(__file__).resolve().parent
    return _release_registry(str(selected_root.resolve()))


def unconsumed_stateframe_switches(
        environment: Mapping[str, object] | None = None, *, root: Path | None = None,
) -> dict[str, object]:
    """Report configured STATEFRAME keys absent from the release-derived reader registry."""
    selected_environment = environment if environment is not None else os.environ
    registered, registry_digest = release_stateframe_switches(root)
    configured = [
        str(name) for name in selected_environment
        if str(name).startswith("STATEFRAME_") and str(name) not in registered
    ]
    retired = sorted(name for name in configured if name in RETIRED_SWITCHES)
    unknown = sorted(name for name in configured if name not in RETIRED_SWITCHES)
    return {
        "schema": SCHEMA,
        "registry_digest": registry_digest,
        "registered_switch_count": len(registered),
        "unknown_switches": unknown,
        "retired_switches": retired,
    }


def enforce_stateframe_switch_registry(
        environment: Mapping[str, object] | None = None, *, journal: object,
        root: Path | None = None,
) -> dict[str, object]:
    """Journal a typed run-start refusal before any provider request on registry drift."""
    detail = unconsumed_stateframe_switches(environment, root=root)
    unknown = detail["unknown_switches"]
    retired = detail["retired_switches"]
    emit = getattr(journal, "emit", None)
    if retired and callable(emit):
        # One informational row per run: the host keeps running, the operator sees why.
        emit("retired_switch_ignored",
             producer="kernel_release_switch_registry",
             dispatch=False, registry_schema=detail["schema"],
             registry_digest=detail["registry_digest"],
             retired_switches=retired,
             reasons={name: RETIRED_SWITCHES[name] for name in retired})
    if unknown:
        if callable(emit):
            emit("stateframe_switch_preflight_refused",
                 producer="kernel_release_switch_registry",
                 outcome="HARNESS_CONTRACT", failure_origin="harness",
                 dispatch=False, registry_schema=detail["schema"],
                 registry_digest=detail["registry_digest"],
                 registered_switch_count=detail["registered_switch_count"],
                 unknown_switches=detail["unknown_switches"])
        raise StateFrameSwitchPreflightError(detail, journal_path=journal.path)
    return detail


__all__ = [
    "REGISTRY_FILE", "RETIRED_SWITCHES", "SCHEMA", "StateFrameSwitchPreflightError",
    "build_registry_document",
    "enforce_stateframe_switch_registry", "release_stateframe_switches",
    "unconsumed_stateframe_switches", "write_registry",
]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    if sys.argv[1:] == ["--write"]:
        print(f"wrote {write_registry(root)}")
    elif sys.argv[1:] == ["--check"]:
        expected = json.dumps(build_registry_document(root), indent=1) + "\n"
        if (root / REGISTRY_FILE).read_text(encoding="utf-8") != expected:
            raise SystemExit("switch-registry.json is stale")
        print("switch-registry.json is current")
    else:
        raise SystemExit("usage: python -I kernel_release_switch_registry.py --write|--check")
