"""Host-owned identity for receipts, checks, verdicts, and their consumers.

The identity arm is deliberately a small seam.  It does not inspect model
text and it never guesses a revision: ``code_hash`` is recomputed from the
workspace whenever a host record is stamped.  The legacy arm does not even
copy a mapping, which keeps the disabled wire shape byte-for-byte unchanged.
"""
from __future__ import annotations

import copy
import contextvars
import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SWITCH = "STATEFRAME_VERDICT_IDENTITY"
SCHEMA = "stateframe-verdict-identity/1"
FIRED_EVENT = "verdict_identity_fired"
IDENTITY_FIELDS = ("run_id", "code_hash", "release_id", "harness_version")
DEFAULT_HARNESS_VERSION = "stateframe-harness/1"

_IGNORED_PARTS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".test-tmp", ".mypy_cache",
    ".ruff_cache",
})
_TARGET_KEYS = frozenset({
    "receipt", "receipts", "test_receipt", "test_receipts", "check_receipt",
    "check_receipts", "check_result", "check_results", "test_result",
    "test_results", "result", "results", "verdict", "verdicts",
    "final_verdict", "completion_verdict", "decision", "decisions",
    "assessment", "assessments", "outcome", "outcomes", "authority",
})
_TARGET_KINDS = frozenset({
    "receipt", "test_receipt", "test_result", "test_subprocess_completed",
    "test_subprocess_complete", "host_test_receipt", "host_obligation_check",
    "host_repair_check", "auto_verify_completed", "auto_verify_failed",
    "probe_result", "completion_reducer_decision", "completion_authority_decision",
    "completion_authority_state", "v2_done_decision", "verdict",
    "final_verdict", "attempt_terminated", "phase.passed", "phase.failed",
    "phase.escalated", "batch_verdict", "batch_scored", "board_update",
})


class IdentityError(ValueError):
    """Base error for missing or incompatible host execution identity."""


class MissingIdentityError(IdentityError):
    """A record cannot be used because one of the four required fields is absent."""


class IdentityMismatch(IdentityError):
    """A receipt or verdict is not evidence for the current execution."""


class MixedReleaseError(IdentityError):
    """An aggregate contains records from more than one release."""


class MixedCodeHashError(IdentityError):
    """An aggregate contains records from more than one workspace revision."""


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """The four host-issued fields carried by one emitted record."""

    run_id: str
    code_hash: str
    release_id: str
    harness_version: str

    def __post_init__(self) -> None:
        for name in IDENTITY_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"run identity requires non-empty {name}")
            object.__setattr__(self, name, value.strip())

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in IDENTITY_FIELDS}

    as_dict = to_dict

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RunIdentity":
        if not isinstance(value, Mapping):
            raise TypeError("run identity must be a mapping")
        missing = [name for name in IDENTITY_FIELDS if not str(value.get(name, "")).strip()]
        if missing:
            raise MissingIdentityError("record is missing " + ", ".join(missing))
        return cls(**{name: str(value[name]) for name in IDENTITY_FIELDS})


@dataclass(frozen=True, slots=True)
class IdentityValidation:
    """Result of checking one record against a current host identity."""

    identity: RunIdentity
    historical: bool = False


@dataclass(frozen=True, slots=True)
class RunIdentityProvider:
    """Stable run/release metadata plus a live workspace hash provider."""

    root: Path
    run_id: str
    release_id: str | None = None
    harness_version: str | None = None
    manifest: Mapping[str, object] | None = None
    excluded_paths: tuple[str | Path, ...] = ()

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("identity root must be an existing directory")
        object.__setattr__(self, "root", root)
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("identity provider requires run_id")
        release = self.release_id or release_id_for(root, self.manifest)
        harness = self.harness_version or harness_version_for()
        object.__setattr__(self, "release_id", str(release).strip())
        object.__setattr__(self, "harness_version", str(harness).strip())
        object.__setattr__(
            self, "excluded_paths",
            tuple(Path(path).expanduser().absolute() for path in self.excluded_paths),
        )

    def current(self) -> RunIdentity:
        """Capture the workspace tree at the instant the host emits a record."""
        return RunIdentity(
            run_id=self.run_id,
            code_hash=workspace_tree_hash(self.root, exclude=self.excluded_paths),
            release_id=str(self.release_id),
            harness_version=str(self.harness_version),
        )

    snapshot = current


_ACTIVE_PROVIDER: contextvars.ContextVar[RunIdentityProvider | None] = contextvars.ContextVar(
    "stateframe_verdict_identity_provider", default=None
)


def mode(value: object = None) -> str:
    """Return the strict feature mode; the unset/default arm is OFF."""
    raw = os.environ.get(SWITCH, "off") if value is None else value
    normalized = str(raw).strip().lower()
    if normalized == "off":
        return "off"
    if normalized == "on":
        return "on"
    raise ValueError(f"{SWITCH} must be off or on")


def enabled() -> bool:
    return mode() == "on"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _ignored(relative: Path) -> bool:
    return any(part in _IGNORED_PARTS for part in relative.parts)


def workspace_tree_hash(root: str | Path, *, exclude: Iterable[str | Path] = ()) -> str:
    """Compute a deterministic content tree hash without consulting git.

    File names, file bytes, regular-file mode, and symlink targets are
    included.  Volatile interpreter/cache directories are excluded, matching
    the host workspace snapshot used by the obligation kernel.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError("workspace hash root must be an existing directory")
    excluded = tuple(Path(path).expanduser().absolute() for path in exclude)
    rows: list[dict[str, object]] = []
    paths = sorted(base.rglob("*"), key=lambda path: path.relative_to(base).as_posix())
    for path in paths:
        relative = path.relative_to(base)
        if _ignored(relative):
            continue
        absolute = path.absolute()
        if any(absolute == item or absolute.is_relative_to(item) for item in excluded):
            continue
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                rows.append({
                    "path": relative.as_posix(),
                    "kind": "symlink",
                    "target": os.readlink(path),
                })
            elif stat.S_ISREG(info.st_mode):
                data = path.read_bytes()
                rows.append({
                    "path": relative.as_posix(),
                    "kind": "file",
                    "mode": stat.S_IMODE(info.st_mode),
                    "size": len(data),
                    "sha256": _sha256(data),
                })
        except (OSError, UnicodeError):
            # A file that disappears during capture is not silently treated as
            # absent; its path is retained as a host-observed error row.
            rows.append({"path": relative.as_posix(), "kind": "unreadable"})
    return _sha256(_canonical({"schema": "stateframe-workspace-tree/1", "files": rows}))


def release_id_for(root: str | Path, manifest: Mapping[str, object] | None = None) -> str:
    """Resolve the host release identity used by a run."""
    for name in ("STATEFRAME_RELEASE_ID", "RELEASE_ID", "CAPABILITY_RELEASE_ID"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    if isinstance(manifest, Mapping):
        for key in ("release_id", "runtime_release_id", "release"):
            value = manifest.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    path = Path(root).expanduser().resolve()
    release_manifest = path / "release-manifest.json"
    if release_manifest.is_file() and path.name:
        return path.name
    return "unreleased"


def harness_version_for() -> str:
    for name in ("STATEFRAME_HARNESS_VERSION", "HARNESS_VERSION"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return DEFAULT_HARNESS_VERSION


def provider_for(root: str | Path, run_id: str, *, manifest: Mapping[str, object] | None = None,
                 release_id: str | None = None, harness_version: str | None = None,
                 excluded_paths: Iterable[str | Path] = ()) -> RunIdentityProvider:
    return RunIdentityProvider(
        Path(root), run_id, release_id, harness_version, manifest,
        tuple(excluded_paths),
    )


def _provider(value: object = None) -> RunIdentityProvider | None:
    if isinstance(value, RunIdentityProvider):
        return value
    if value is None:
        return _ACTIVE_PROVIDER.get()
    return None


def _resolve_identity(value: object = None) -> RunIdentity:
    provider = _provider(value)
    if provider is not None:
        return provider.current()
    if isinstance(value, RunIdentity):
        return value
    if isinstance(value, Mapping):
        return RunIdentity.from_mapping(value)
    if value is None:
        raise MissingIdentityError("identity is required while the mechanism is on")
    raise TypeError("identity must be RunIdentity, RunIdentityProvider, or a mapping")


@contextmanager
def identity_context(provider: RunIdentityProvider) -> Iterator[RunIdentityProvider]:
    token = _ACTIVE_PROVIDER.set(provider)
    try:
        yield provider
    finally:
        _ACTIVE_PROVIDER.reset(token)


def _raw_record(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    raise TypeError("host record must be a mapping or expose to_dict()")


def stamp_record(record: Mapping[str, object] | object, identity: object = None,
                 *, historical: bool = False) -> dict[str, object] | object:
    """Add host identity to a record; OFF returns the original object untouched."""
    if not enabled():
        return record
    raw = _raw_record(record)
    stamped = copy.deepcopy(dict(raw))
    stamped.update(_resolve_identity(identity).to_dict())
    if historical:
        stamped["historical"] = True
    return stamped


stamp_receipt = stamp_record
stamp_check = stamp_record
stamp_check_result = stamp_record
stamp_verdict = stamp_record
stamp_result = stamp_record


def _historical(record: Mapping[str, object]) -> bool:
    return bool(record.get("historical") is True or
                str(record.get("identity_status", "")).lower() == "historical")


def _current_identity(value: object) -> RunIdentity:
    if isinstance(value, RunIdentityProvider):
        return value.current()
    return _resolve_identity(value)


def identity_of(record: Mapping[str, object] | object) -> RunIdentity:
    return RunIdentity.from_mapping(_raw_record(record))


def assert_current(record: Mapping[str, object] | object, current: object = None, *,
                   allow_historical: bool = False, historical: bool = False) -> IdentityValidation:
    """Reject stale code/release evidence unless it is explicitly historical."""
    if not enabled():
        raw = _raw_record(record)
        return IdentityValidation(
            identity=RunIdentity.from_mapping(raw) if all(name in raw for name in IDENTITY_FIELDS)
            else RunIdentity("off", "off", "off", "off"),
            historical=bool(historical or _historical(raw)),
        )
    raw = _raw_record(record)
    identity = RunIdentity.from_mapping(raw)
    is_historical = bool(historical or _historical(raw))
    if is_historical:
        if not allow_historical:
            raise IdentityMismatch("historical identity requires explicit allow_historical=True")
        return IdentityValidation(identity, historical=True)
    if current is None:
        return IdentityValidation(identity, historical=False)
    expected = _current_identity(current)
    if identity.code_hash != expected.code_hash:
        raise IdentityMismatch(
            f"code_hash mismatch: record={identity.code_hash} current={expected.code_hash}"
        )
    if identity.release_id != expected.release_id:
        raise IdentityMismatch(
            f"release_id mismatch: record={identity.release_id} current={expected.release_id}"
        )
    if identity.harness_version != expected.harness_version:
        raise IdentityMismatch(
            "harness_version mismatch: "
            f"record={identity.harness_version} current={expected.harness_version}"
        )
    return IdentityValidation(identity, historical=False)


def _record_items(records: object) -> list[Mapping[str, object]]:
    if isinstance(records, Mapping):
        single_markers = {*IDENTITY_FIELDS, "identity", "kind", "producer", "receipt",
                          "status", "result", "result_class", "check_id", "receipt_id"}
        if single_markers.intersection(records):
            return [records]
        values = list(records.values())
        if values and all(isinstance(item, Mapping) or callable(getattr(item, "to_dict", None))
                          for item in values):
            return [_raw_record(item) for item in values]
        return [records]
    if isinstance(records, Iterable) and not isinstance(records, (str, bytes)):
        return [_raw_record(item) for item in records]
    raise TypeError("aggregate records must be a mapping or iterable of mappings")


def validate_batch(records: object, current: object = None, *,
                   allow_historical: bool = False, historical: bool = False) -> tuple[Mapping[str, object], ...]:
    """Validate a batch before scoring, board update, or completion reduction."""
    items = _record_items(records)
    if not enabled():
        return tuple(items)
    identities: list[tuple[Mapping[str, object], RunIdentity, bool]] = []
    for item in items:
        identity = identity_of(item)
        is_historical = bool(historical or _historical(item))
        identities.append((item, identity, is_historical))
    comparable = [item for item in identities if not item[2]]
    releases = {item[1].release_id for item in comparable}
    if len(releases) > 1:
        raise MixedReleaseError("batch contains mixed release_id values: " + ", ".join(sorted(releases)))
    code_hashes = {item[1].code_hash for item in comparable}
    if len(code_hashes) > 1:
        raise MixedCodeHashError("batch contains mixed code_hash values")
    for item, _identity, is_historical in identities:
        assert_current(
            item,
            current,
            allow_historical=allow_historical or is_historical,
            historical=is_historical,
        )
    return tuple(items)


assert_batch_compatible = validate_batch
validate_aggregate = validate_batch


def _stamp_nested(value: object, identity: RunIdentity, key: str = "") -> object:
    if isinstance(value, Mapping):
        result = {str(name): copy.deepcopy(child) for name, child in value.items()}
        if key in _TARGET_KEYS:
            result.update(identity.to_dict())
        for name, child in tuple(result.items()):
            if name in _TARGET_KEYS:
                result[name] = _stamp_nested(child, identity, name)
        return result
    if isinstance(value, list):
        return [_stamp_nested(child, identity, key) for child in value]
    if isinstance(value, tuple):
        return tuple(_stamp_nested(child, identity, key) for child in value)
    return copy.deepcopy(value)


def prepare_journal_event(journal: object, kind: str, fields: Mapping[str, object]) -> tuple[dict[str, object], bool]:
    """Prepare a journal event and report whether a fired marker is needed."""
    provider = getattr(journal, "_verdict_identity_provider", None)
    if not enabled() or not isinstance(provider, RunIdentityProvider):
        return dict(fields), False
    identity = provider.current()
    prepared = _stamp_nested(fields, identity)
    prepared.update({
        "code_hash": identity.code_hash,
        "release_id": identity.release_id,
        "harness_version": identity.harness_version,
    })
    normalized_kind = str(kind).strip().lower()
    # Once attached, the identity organ owns the host emission boundary.  It
    # stamps ordinary events as well as receipts, so emit one proof event for
    # every such action; the proof event itself is the sole recursion break.
    should_fire = normalized_kind != FIRED_EVENT
    return prepared, should_fire


def journal_identity_fired(journal: object, source_event: str, identity: object = None) -> None:
    """Publish the non-model, host-observed proof that the organ acted."""
    if not enabled():
        return
    provider = _provider(identity) or getattr(journal, "_verdict_identity_provider", None)
    current = _resolve_identity(provider)
    journal.emit(
        FIRED_EVENT,
        producer="kernel_verdict_identity",
        source_event=str(source_event),
        code_hash=current.code_hash,
        release_id=current.release_id,
        harness_version=current.harness_version,
        dispatch=False,
    )


def attach_journal(journal: object, provider: RunIdentityProvider) -> RunIdentityProvider | None:
    """Attach live identity stamping to either journal implementation."""
    if not enabled():
        return None
    if not isinstance(provider, RunIdentityProvider):
        raise TypeError("journal identity requires RunIdentityProvider")
    setattr(journal, "_verdict_identity_provider", provider)
    journal_identity_fired(journal, "identity_attached", provider)
    return provider


def provider_from_journal(journal: object) -> RunIdentityProvider | None:
    value = getattr(journal, "_verdict_identity_provider", None)
    return value if isinstance(value, RunIdentityProvider) else None


def stamp_with_journal(record: Mapping[str, object] | object, journal: object) -> dict[str, object] | object:
    """Convenience for callers that emit a structured record outside ``emit``."""
    provider = provider_from_journal(journal)
    return stamp_record(record, provider.current() if provider is not None else None)
