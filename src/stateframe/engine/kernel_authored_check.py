"""Host-admitted, revision-bound authored checks.

This is the narrow executable-check boundary used by the benchmark authored route.
It freezes the solver's separate check artifact before execution, then asks one
language adapter to run those exact bytes against independent parent and
candidate snapshots.  Generic probes deliberately do not use this module:
their receipts remain observation-only.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Protocol


SCHEMA = "stateframe-authored-check/1"
ADMISSION_SCHEMA = "stateframe-authored-check-admission/1"
PHASES = ("collection", "setup", "target", "assertion")
PASS = "PASS"
BEHAVIOR_FAIL = "BEHAVIOR_FAIL"
UNKNOWN = "UNKNOWN"
UNRUNNABLE = "UNRUNNABLE"
UNSUPPORTED = "UNSUPPORTED"
ENVIRONMENT = "ENVIRONMENT"
STALE = "STALE"
OBSERVATION_ONLY = "OBSERVATION_ONLY"
BASELINE_ONLY = "BASELINE_ONLY"
NON_DISCRIMINATING = "NON_DISCRIMINATING"
STILL_FAILING = "STILL_FAILING"
REGRESSION = "REGRESSION"
_LANGUAGES = {
    ".py": "python", ".go": "go", ".rs": "rust", ".ts": "typescript",
    ".tsx": "typescript", ".js": "javascript", ".jsx": "javascript",
    ".java": "java", ".php": "php", ".vue": "vue",
}
_SUPPORTED_LANGUAGES = frozenset(_LANGUAGES.values())
_IMPORT_FAILURE = re.compile(
    r"(?:ImportError|ModuleNotFoundError|No module named|collection error|"
    r"test collection failed)", re.IGNORECASE,
)
_TEST_COUNT = re.compile(r"(?:Ran\s+([1-9]\d*)\s+tests?|([1-9]\d*)\s+tests?\s+passed)", re.IGNORECASE)


class AuthoredCheckError(ValueError):
    """A host-side authored-check admission refusal."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class CheckAdapter(Protocol):
    language: str

    def run(
            self, check: Mapping[str, object], snapshot: Mapping[str, object], *,
            revision: str, side: str,
    ) -> Mapping[str, object]: ...


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else str(value or "").strip()


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise AuthoredCheckError("SOURCE", "authored check source must be text or bytes")


def _path(value: object, *, label: str = "path") -> str:
    raw = _text(value).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if (not raw or candidate.is_absolute()
            or any(part in {"", ".", ".."} for part in candidate.parts)):
        raise AuthoredCheckError("PATH", f"{label} must be a canonical relative path")
    return candidate.as_posix()


def _paths(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        value = value.keys()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    result: list[str] = []
    for item in value:
        try:
            path = _path(item)
        except AuthoredCheckError:
            continue
        if path not in result:
            result.append(path)
    return tuple(result)


def _language(path: str, declared: object) -> str:
    language = _text(declared).casefold()
    inferred = _LANGUAGES.get(PurePosixPath(path).suffix.casefold(), "unknown")
    if language and language != inferred:
        raise AuthoredCheckError("LANGUAGE", f"declared language {language!r} does not match {path}")
    return language or inferred


def _environment(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AuthoredCheckError("ENVIRONMENT", "environment must be a string map")
    result: dict[str, str] = {}
    for key, raw in value.items():
        name = _text(key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name):
            raise AuthoredCheckError("ENVIRONMENT", "environment variable names must be identifiers")
        if not isinstance(raw, str) or len(raw) > 512:
            raise AuthoredCheckError("ENVIRONMENT", "environment values must be bounded strings")
        result[name] = raw
    if len(result) > 64:
        raise AuthoredCheckError("ENVIRONMENT", "environment has too many variables")
    return result


def _selector(selector: object, path: str) -> str:
    value = _text(selector)
    if not value or value.split("::", 1)[0].replace("\\", "/").lstrip("./") != path:
        raise AuthoredCheckError("SELECTOR", "selector must name the admitted check artifact")
    if not re.fullmatch(r"[^\s:]+(?:\.py)?(?:::[A-Za-z_][A-Za-z0-9_]*)*", value):
        raise AuthoredCheckError("SELECTOR", "selector contains unsupported syntax")
    return value


def _authority(proposal: Mapping[str, object], path: str,
               authorized_paths: Sequence[str]) -> dict[str, object]:
    raw = proposal.get("write_authority")
    if not isinstance(raw, Mapping):
        raise AuthoredCheckError("WRITE_AUTHORITY_REQUIRED", "authored checks require explicit scratch write authority")
    authority = dict(raw)
    authority_ref = _text(authority.get("authority_ref"))
    if not authority_ref:
        raise AuthoredCheckError("WRITE_AUTHORITY_REQUIRED", "write_authority.authority_ref is required")
    allowed = _paths(authority.get("allowed_paths", authorized_paths))
    if path not in allowed:
        raise AuthoredCheckError("OUT_OF_SCOPE", f"authored check path is outside write authority: {path}")
    authority["authority_ref"] = authority_ref
    authority["allowed_paths"] = list(allowed)
    return authority


@dataclass(frozen=True)
class AuthoredCheck:
    check_id: str
    requirement_id: str
    clause: str
    requirement_quote: str
    path: str
    selector: str
    language: str
    source: bytes
    environment: Mapping[str, str]
    write_authority: Mapping[str, object]
    observation_marker: str
    parent_revision: str
    candidate_revision: str
    source_sha256: str
    environment_digest: str
    check_hash: str
    invocation: Mapping[str, object] = field(default_factory=dict)

    def artifact(self) -> dict[str, object]:
        artifact = {
            "schema": SCHEMA,
            "check_id": self.check_id,
            "requirement_id": self.requirement_id,
            "clause": self.clause,
            "requirement_quote": self.requirement_quote,
            "path": self.path,
            "selector": self.selector,
            "language": self.language,
            "source": self.source.decode("utf-8", errors="strict"),
            "source_sha256": self.source_sha256,
            "environment": dict(self.environment),
            "environment_digest": self.environment_digest,
            "observation_marker": self.observation_marker,
            "write_authority": dict(self.write_authority),
            "parent_revision": self.parent_revision,
            "candidate_revision": self.candidate_revision,
            "check_hash": self.check_hash,
        }
        if self.invocation:
            artifact["invocation"] = dict(self.invocation)
        return artifact


def _check_from_admission(admission: Mapping[str, object]) -> AuthoredCheck:
    artifact = admission.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AuthoredCheckError("SCHEMA", "authored admission has no artifact")
    source = _bytes(artifact.get("source"))
    path = _path(artifact.get("path"))
    environment = _environment(artifact.get("environment"))
    source_sha = "sha256:" + hashlib.sha256(source).hexdigest()
    if source_sha != artifact.get("source_sha256"):
        raise AuthoredCheckError("STALE_CHECK", "authored check bytes changed after admission")
    environment_digest = _digest(environment)
    if environment_digest != artifact.get("environment_digest"):
        raise AuthoredCheckError("STALE_CHECK", "authored check environment changed after admission")
    language = _language(path, artifact.get("language"))
    selector = _selector(artifact.get("selector"), path)
    artifact_authority = artifact.get("write_authority")
    if not isinstance(artifact_authority, Mapping):
        raise AuthoredCheckError("STALE_CHECK", "authored check write authority is missing")
    try:
        authority = _authority({"write_authority": artifact_authority}, path, ())
        admission_authority = _authority(admission, path, ())
    except AuthoredCheckError as error:
        # A post-admission scope mutation is stale identity, not a fresh
        # capability result.  Never let malformed authority turn into proof.
        raise AuthoredCheckError("STALE_CHECK", "authored check write authority changed after admission") from error
    if _digest(authority) != _digest(admission_authority):
        raise AuthoredCheckError("STALE_CHECK", "authored check write authority changed after admission")
    marker = _text(artifact.get("observation_marker"))
    if (not marker or len(marker) > 160
            or not re.fullmatch(r"[A-Za-z0-9_.:-]+", marker)):
        raise AuthoredCheckError("STALE_CHECK", "authored check observation marker changed")
    parent_revision = _text(artifact.get("parent_revision"))
    candidate_revision = _text(artifact.get("candidate_revision"))
    if (not parent_revision or not candidate_revision
            or parent_revision != _text(admission.get("parent_revision"))
            or candidate_revision != _text(admission.get("candidate_revision"))):
        raise AuthoredCheckError("STALE_CHECK", "authored check revision pair changed after admission")
    expected = _check_hash(
        check_id=_text(artifact.get("check_id")),
        requirement_id=_text(artifact.get("requirement_id")),
        clause=_text(artifact.get("clause")),
        requirement_quote=_text(artifact.get("requirement_quote")),
        path=path, selector=selector, language=language,
        source_sha256=source_sha, environment_digest=environment_digest,
        observation_marker=marker, write_authority=authority,
        parent_revision=parent_revision, candidate_revision=candidate_revision,
    )
    if (expected != artifact.get("check_hash")
            or expected != admission.get("check_hash")
            or _text(admission.get("check_id")) != _text(artifact.get("check_id"))
            or _text(admission.get("clause_id")) != _text(artifact.get("requirement_id"))
            or _text(admission.get("selector")) != selector):
        raise AuthoredCheckError("STALE_CHECK", "authored check identity changed after admission")
    return AuthoredCheck(
        check_id=_text(artifact.get("check_id")),
        requirement_id=_text(artifact.get("requirement_id")),
        clause=_text(artifact.get("clause")),
        requirement_quote=_text(artifact.get("requirement_quote")),
        path=path,
        selector=selector,
        language=language,
        source=source,
        environment=environment,
        write_authority=authority,
        observation_marker=marker,
        parent_revision=parent_revision,
        candidate_revision=candidate_revision,
        source_sha256=source_sha,
        environment_digest=_digest(environment),
        check_hash=expected,
        invocation=(dict(artifact["invocation"])
                    if isinstance(artifact.get("invocation"), Mapping) else {}),
    )


BINDING_SCHEMA = "stateframe.authored_obligation_binding.v1"


def normalize_clause(text: object) -> str:
    """Whitespace-normalized clause text; the only spelling bindings compare."""
    return " ".join(str(text or "").split())


def clause_digest(text: object) -> str:
    normalized = normalize_clause(text)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


_CLAUSE_SPAN_KEYS = (
    "source_id", "source_sha256", "source_digest", "path", "start_byte", "end_byte",
    "start_offset", "end_offset", "start", "end", "start_line", "start_column", "end_line",
    "end_column", "line", "column", "span_id", "id",
)


def _normalized_clause_span(span: object) -> object:
    """Keep only host-owned source-location fields for clause identity."""
    if isinstance(span, Mapping):
        values = {
            key: copy.deepcopy(span[key]) for key in _CLAUSE_SPAN_KEYS
            if key in span and span[key] not in (None, "")
        }
        # A bare host span id is useful only when no richer location was
        # supplied; proposal text is deliberately never accepted as location.
        has_location = any(key in values for key in (
                "start_byte", "end_byte", "start_offset", "end_offset", "start", "end",
                "start_line", "start_column", "end_line", "end_column",
                "line", "column"))
        if has_location:
            values.pop("span_id", None)
            values.pop("id", None)
        else:
            values = {key: values[key] for key in ("source_id", "path", "span_id", "id")
                      if key in values}
        return values or None
    if isinstance(span, Sequence) and not isinstance(span, (str, bytes, bytearray)):
        values = tuple(copy.deepcopy(item) for item in span)
        return values or None
    if isinstance(span, str) and span.strip():
        return {"span_id": span.strip()}
    return None


def clause_identity(text: object, source_span: object) -> str:
    """Digest normalized host clause text together with its host source span."""
    normalized = normalize_clause(text)
    span = _normalized_clause_span(source_span)
    if not normalized or span is None:
        return ""
    payload = json.dumps({"clause": normalized, "source_span": span},
                        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                        default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _effective_clause_span(host_clause: object, source_span: object) -> object:
    """Do not mistake legacy clause-text fallback for a source span."""
    if (isinstance(source_span, str)
            and normalize_clause(source_span) == normalize_clause(host_clause)):
        return None
    return source_span


def binding_receipt_ids_are_host_consistent(receipt: Mapping[str, object]) -> bool:
    """Require every receipt clause reference to be host-approved."""
    binding = receipt.get("host_obligation_binding") if isinstance(receipt, Mapping) else None
    if not isinstance(binding, Mapping):
        return False
    canonical = _text(binding.get("obligation_id"))
    alias = _text(binding.get("host_requirement_id"))
    references = {
        _text(receipt.get("requirement_id")),
        _text(receipt.get("clause_id")),
    }
    references.discard("")
    accepted = {value for value in (canonical, alias) if value}
    if not (canonical and references and references.issubset(accepted)):
        return False
    if alias and alias != canonical:
        return bool(
            _text(binding.get("clause_identity"))
            and _text(receipt.get("host_clause_identity"))
            == _text(binding.get("clause_identity"))
        )
    return True


def obligation_binding(obligation_id: object, host_clause: object, quote: object,
                       *, source: str, clause_span: object = None) -> dict[str, object] | None:
    """Bind an authored check to one HOST-OWNED obligation clause (W3).

    ``host_clause`` must come from host data (obligation spec / requirements
    ledger), never from the proposal.  After whitespace normalization, the
    solver quote must equal the whole host clause; fragments and spans that add
    adjacent clauses do not bind.  Returns ``None`` when unbound.
    """
    ident = _text(obligation_id)
    clause = normalize_clause(host_clause)
    covered = normalize_clause(quote)
    if not ident or not clause or not covered or clause != covered:
        return None
    binding = {"schema": BINDING_SCHEMA, "obligation_id": ident,
               "clause_sha256": clause_digest(clause), "source": str(source)}
    identity = clause_identity(clause, clause_span)
    if clause_span is not None and not identity:
        return None
    if identity:
        binding["clause_identity"] = identity
    return binding


def binding_certifies(receipt: Mapping[str, object], obligation_id: str, host_clause: object,
                      *, clause_span: object = None, allow_alias: bool = False) -> bool:
    """True only when ``receipt`` carries a host binding to exactly this clause."""
    binding = receipt.get("host_obligation_binding") if isinstance(receipt, Mapping) else None
    if not isinstance(binding, Mapping) or binding.get("schema") != BINDING_SCHEMA:
        return False
    if not binding_receipt_ids_are_host_consistent(receipt):
        return False
    expected = clause_digest(host_clause)
    if (not expected or binding.get("clause_sha256") != expected
            or normalize_clause(host_clause) != normalize_clause(receipt.get("requirement_quote"))):
        return False
    expected_identity = clause_identity(
        host_clause, _effective_clause_span(host_clause, clause_span))
    bound_identity = _text(binding.get("clause_identity"))
    if expected_identity:
        return (bound_identity == expected_identity
                and (allow_alias
                     or _text(binding.get("obligation_id")) == _text(obligation_id)))
    if bound_identity:
        # A host identity cannot be checked without the host source span.
        return False
    return _text(binding.get("obligation_id")) == _text(obligation_id)


def _check_hash(*, check_id: str, requirement_id: str, clause: str,
                requirement_quote: str, path: str, selector: str, language: str,
                source_sha256: str, environment_digest: str,
                observation_marker: str, write_authority: Mapping[str, object],
                parent_revision: str, candidate_revision: str) -> str:
    return _digest({
        "check_id": check_id, "requirement_id": requirement_id, "clause": clause,
        "requirement_quote": requirement_quote, "path": path, "selector": selector,
        "language": language, "source_sha256": source_sha256,
        "environment_digest": environment_digest,
        "observation_marker": observation_marker,
        "write_authority": dict(write_authority),
        # Revisions belong to execution identity, not frozen check identity.
    })


def admit_authored_check(
        proposal: Mapping[str, object], *, public_text: str,
        parent_revision: object, candidate_revision: object,
        authorized_paths: Sequence[str] = (), protected_paths: Sequence[str] = (),
        existing_check_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Freeze one separate authored check from public evidence only."""
    try:
        if not isinstance(proposal, Mapping):
            raise AuthoredCheckError("SCHEMA", "authored check proposal must be an object")
        check_id = _text(proposal.get("check_id"))
        requirement_id = _text(proposal.get("requirement_id", proposal.get("clause_id")))
        clause = _text(proposal.get("clause"))
        quote = _text(proposal.get("requirement_quote", clause))
        if not check_id or not requirement_id or not clause or not quote:
            raise AuthoredCheckError("REQUIREMENT", "check, requirement, clause, and quote are required")
        if check_id in {str(item).strip() for item in existing_check_ids}:
            raise AuthoredCheckError("DUPLICATE_CHECK", f"authored check already exists: {check_id}")
        if not isinstance(public_text, str) or quote not in public_text:
            raise AuthoredCheckError("PUBLIC_PROVENANCE", "requirement quote is not present in public text")
        path = _path(proposal.get("path"), label="authored check path")
        protected = set(_paths(protected_paths))
        if path in protected:
            raise AuthoredCheckError("PROTECTED_TEST", "pre-existing public tests are immutable")
        authority = _authority(proposal, path, _paths(authorized_paths))
        parent = _text(parent_revision)
        candidate = _text(candidate_revision)
        if not parent or not candidate:
            raise AuthoredCheckError("REVISION", "parent and candidate revisions are required")
        source = _bytes(proposal.get("source"))
        if not source.strip() or len(source) > 2 * 1024 * 1024:
            raise AuthoredCheckError("SOURCE", "authored check source must be non-empty and bounded")
        language = _language(path, proposal.get("language"))
        selector = _selector(proposal.get("selector", path), path)
        environment = _environment(proposal.get("environment"))
        marker = _text(proposal.get("observation_marker")) or "__STATEFRAME_AUTHORED_OBSERVED__"
        if len(marker) > 160 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", marker):
            raise AuthoredCheckError("OBSERVATION", "observation_marker must be a bounded token")
        source_sha = "sha256:" + hashlib.sha256(source).hexdigest()
        environment_digest = _digest(environment)
        check_hash = _check_hash(
            check_id=check_id, requirement_id=requirement_id, clause=clause,
            requirement_quote=quote, path=path, selector=selector, language=language,
            source_sha256=source_sha, environment_digest=environment_digest,
            observation_marker=marker, write_authority=authority,
            parent_revision=parent, candidate_revision=candidate,
        )
        check = AuthoredCheck(
            check_id, requirement_id, clause, quote, path, selector, language, source,
            environment, authority, marker, parent, candidate,
            source_sha, environment_digest, check_hash,
        )
        capability = "EXECUTABLE" if language == "python" else (
            "NATIVE_ADAPTER_REQUIRED" if language in _SUPPORTED_LANGUAGES else "UNRUNNABLE")
        response = {
            "schema": ADMISSION_SCHEMA,
            "admitted": True,
            "status": "ADMITTED",
            "capability": capability,
            "artifact": check.artifact(),
            "write_authority": dict(authority),
            "parent_revision": parent,
            "candidate_revision": candidate,
            "revision_pair": "BASELINE_ONLY" if parent == candidate else "PAIRED",
            "check_hash": check_hash,
            "selector": selector,
            "clause_id": requirement_id,
            "check_id": check_id,
            "requirement_quote": quote,
            "source_unit_handle": _text(proposal.get("source_unit_handle")),
        }
        for key in ("reached_consumers", "coverage_complete", "required_partitions", "binding_stable"):
            if key in proposal:
                response[key] = proposal[key]
        return response
    except AuthoredCheckError as error:
        return {
            "schema": ADMISSION_SCHEMA, "admitted": False, "status": "REJECTED",
            "capability": "UNKNOWN", "code": error.code, "reason": str(error),
        }


def _snapshot(snapshot: object) -> tuple[str, dict[str, bytes]]:
    if not isinstance(snapshot, Mapping):
        return "", {}
    revision = _text(snapshot.get("revision", snapshot.get("product_revision")))
    files_value = snapshot.get("files")
    if files_value is None:
        files_value = {key: value for key, value in snapshot.items()
                       if key not in {"revision", "product_revision"}}
    if not isinstance(files_value, Mapping):
        return revision, {}
    files: dict[str, bytes] = {}
    for raw_path, source in files_value.items():
        try:
            relative = _path(raw_path)
            files[relative] = _bytes(source)
        except AuthoredCheckError:
            continue
    return revision, files


def _await(value: object) -> object:
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    # The host may already be inside an event loop.  Execute the awaitable in
    # a short-lived thread so the synchronous receipt API never returns a
    # coroutine as if it were an observation.
    import threading
    result: list[object] = []
    error: list[BaseException] = []

    def resolve() -> None:
        try:
            result.append(asyncio.run(value))
        except BaseException as exc:  # pragma: no cover - defensive loop bridge
            error.append(exc)

    thread = threading.Thread(target=resolve, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0] if result else None


def _phase_map(status: str) -> dict[str, str]:
    return {phase: status for phase in PHASES}


def _normalize_execution(raw: object, *, check: AuthoredCheck, revision: str,
                         side: str, snapshot_digest: str) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raw = {"status": UNRUNNABLE, "reason": "adapter returned no execution mapping"}
    status = _text(raw.get("status", raw.get("outcome"))).upper()
    if status not in {PASS, BEHAVIOR_FAIL, UNKNOWN, UNRUNNABLE, UNSUPPORTED, ENVIRONMENT, STALE}:
        status = UNKNOWN
    output = "\n".join(str(raw.get(name, "")) for name in ("stdout", "stderr", "output", "observation") if raw.get(name))
    phases = raw.get("phases", raw.get("phase_outcomes"))
    phase_map = _phase_map(status)
    if isinstance(phases, Mapping):
        for phase in PHASES:
            value = _text(phases.get(phase)).upper()
            if value in {PASS, BEHAVIOR_FAIL, UNKNOWN, UNRUNNABLE, UNSUPPORTED,
                         ENVIRONMENT, STALE}:
                phase_map[phase] = value
    for phase in ("collection", "setup"):
        if phase_map[phase] in {UNRUNNABLE, UNSUPPORTED, ENVIRONMENT}:
            status = phase_map[phase]
    events = raw.get("events", ())
    event_names = {
        str(item.get("kind", item.get("event", ""))).strip().lower()
        for item in events if isinstance(item, Mapping)
    } if isinstance(events, Sequence) and not isinstance(events, (str, bytes, bytearray)) else set()
    reached = raw.get("reached_observation")
    if not isinstance(reached, bool):
        # Printed markers are diagnostic text, never host reach evidence.
        reached = bool({"target_entry", "target_reached", "observation_reached"} & event_names)
    integrity = raw.get("integrity", True)
    integrity = integrity is True
    if status in {UNSUPPORTED, UNRUNNABLE, ENVIRONMENT, STALE}:
        integrity = False
    if status in {PASS, BEHAVIOR_FAIL} and not reached:
        status = UNKNOWN
        phase_map["assertion"] = UNKNOWN
    test_count = raw.get("tests", raw.get("test_count"))
    skipped = raw.get("skipped", raw.get("skip_count", 0))
    if (type(test_count) is int and test_count <= 0) or (
            type(test_count) is int and test_count > 0 and type(skipped) is int
            and skipped >= test_count):
        status = UNKNOWN
        phase_map["assertion"] = UNKNOWN
    target_exception = _text(raw.get("target_exception"))
    if target_exception and reached and raw.get("no_failure_assertion") is True:
        status = BEHAVIOR_FAIL
        phase_map["assertion"] = BEHAVIOR_FAIL
    return {
        "side": side,
        "revision": revision,
        "snapshot_digest": snapshot_digest,
        "simulation": raw.get("simulation") is True,
        "status": status,
        "outcome": status,
        "phases": phase_map,
        "reached_observation": reached,
        "integrity": integrity,
        "exit_code": raw.get("exit_code"),
        "tests": test_count,
        "skipped": skipped,
        "events": tuple(dict(item) for item in events
                         if isinstance(item, Mapping)) if isinstance(events, Sequence)
        and not isinstance(events, (str, bytes, bytearray)) else (),
        "observation": raw.get("observation", output[-4000:]),
        "reason": _text(raw.get("reason")),
        "target_exception": target_exception,
    }


def _snapshot_digest(files: Mapping[str, bytes]) -> str:
    return _digest([(path, "sha256:" + hashlib.sha256(source).hexdigest())
                    for path, source in sorted(files.items())])


class UnsupportedAdapter:
    """Language-neutral stub: unsupported runtimes are never called a pass."""

    def __init__(self, language: str):
        self.language = language

    def run(self, check: Mapping[str, object], snapshot: Mapping[str, object], *,
            revision: str, side: str) -> Mapping[str, object]:
        return {"status": UNSUPPORTED, "reason": f"no execution adapter for {self.language}",
                "reached_observation": False}


class PythonUnittestAdapter:
    """Execute the frozen Python check bytes through isolated unittest."""

    language = "python"

    def __init__(self, runner: object = None) -> None:
        # Production uses kernel_execution_policy.  Tests may inject a result
        # factory so the contract remains verifiable on hosts that prohibit
        # nested sandbox-exec, without creating an unsandboxed fallback.
        self._runner = runner

    @staticmethod
    def _unittest_target(selector: str) -> str:
        path, *members = selector.split("::")
        module = path[:-3] if path.endswith(".py") else path
        module = module.replace("/", ".").replace("\\", ".")
        return ".".join((module, *members))

    def run(self, check: Mapping[str, object], snapshot: Mapping[str, object], *,
            revision: str, side: str) -> Mapping[str, object]:
        revision_value, files = _snapshot(snapshot)
        revision = revision or revision_value
        path = _path(check.get("path"))
        files[path] = _bytes(check.get("source"))
        with tempfile.TemporaryDirectory(prefix="authored-check-") as directory:
            root = Path(directory).resolve()
            try:
                for relative, source in files.items():
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source)
                environment = {
                    str(key): str(value) for key, value in
                    (check.get("environment") if isinstance(check.get("environment"), Mapping) else {}).items()
                }
                environment["PYTHONPATH"] = str(root)
                environment.setdefault("PYTHONNOUSERSITE", "1")
                from kernel_test_python import selected_test_python
                command = [selected_test_python(), "-B", "-m", "unittest",
                           self._unittest_target(_text(check.get("selector")))]
                runner = self._runner
                if runner is None:
                    from kernel_execution_policy import run_sandboxed
                    runner = run_sandboxed
                if not callable(runner):
                    raise TypeError("Python authored-check runner is not callable")
                result = runner(
                    command, root, env=environment, timeout=120,
                    output_limit=256 * 1024,
                )
                completed = result.completed
                output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
                raw_events = getattr(result, "events", ())
                events = tuple(dict(item) for item in raw_events
                               if isinstance(item, Mapping)) \
                    if isinstance(raw_events, Sequence) and not isinstance(
                        raw_events, (str, bytes, bytearray)) else ()
                event_names = {str(item.get("event", item.get("kind", ""))).casefold()
                               for item in events}
                # The authored source owns the marker, so printed marker text
                # is retained only as a diagnostic and can never prove target
                # entry.  A repository runner must provide structured events.
                reached = bool({"target_entry", "target_reached", "observation_reached"}
                               & event_names)
                assertion_observed = bool({"assertion", "assertion_result", "test_passed",
                                           "test_failed"} & event_names)
                tests = _TEST_COUNT.search(output)
                if getattr(result, "timed_out", False):
                    phase = UNRUNNABLE
                elif _IMPORT_FAILURE.search(output):
                    phase = ENVIRONMENT
                elif completed.returncode == 0 and reached and assertion_observed:
                    phase = PASS
                elif completed.returncode != 0 and reached and assertion_observed:
                    phase = BEHAVIOR_FAIL
                else:
                    phase = UNKNOWN
                return {
                    "status": phase,
                    "stdout": completed.stdout or "",
                    "stderr": completed.stderr or "",
                    "exit_code": completed.returncode,
                    "tests": int((tests.group(1) or tests.group(2))) if tests else None,
                    "reached_observation": reached,
                    "assertion_observed": assertion_observed,
                    "events": events,
                    "integrity": not getattr(result, "timed_out", False),
                    "observation": output[-4000:],
                }
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                return {"status": UNRUNNABLE, "reason": f"python adapter unavailable: {type(error).__name__}"}


def _adapter_for(language: str, adapter: object = None,
                 check: Mapping[str, object] | None = None) -> object:
    if adapter is not None:
        return adapter
    if language == "python":
        if isinstance(check, Mapping) and isinstance(check.get("invocation"), Mapping):
            try:
                from authored_check_adapters import adapter_for
                return adapter_for(language)
            except (ImportError, RuntimeError, TypeError, ValueError):
                return UnsupportedAdapter(language)
        return PythonUnittestAdapter()
    try:
        from authored_check_adapters import adapter_for
        return adapter_for(language)
    except (ImportError, RuntimeError, TypeError, ValueError):
        return UnsupportedAdapter(language)


def _binding_stable(runtime: object, parent: Mapping[str, object],
                    candidate: Mapping[str, object]) -> bool:
    """Read stability only from host snapshots/runtime, never proposal text."""
    runtime_value = getattr(runtime, "binding_stable", None) if runtime is not None else None
    if isinstance(runtime_value, bool):
        return runtime_value
    parent_value = parent.get("binding_stable")
    candidate_value = candidate.get("binding_stable")
    if isinstance(parent_value, bool) or isinstance(candidate_value, bool):
        return parent_value is True and candidate_value is True
    return False


def _run_adapter(adapter: object, check: Mapping[str, object], snapshot: Mapping[str, object], *,
                 revision: str, side: str) -> object:
    run = getattr(adapter, "run", None)
    if callable(run):
        return _await(run(check, snapshot, revision=revision, side=side))
    if callable(adapter):
        return _await(adapter(check, snapshot, revision, side))
    return None


def _run_adapter_safely(adapter: object, check: Mapping[str, object],
                        snapshot: Mapping[str, object], *, revision: str,
                        side: str) -> object:
    """Turn adapter faults into a phase-tagged capability result."""
    try:
        return _run_adapter(adapter, check, snapshot, revision=revision, side=side)
    except Exception as error:  # adapter code is an untrusted execution boundary
        detail = f"adapter execution failed: {type(error).__name__}"
        return {
            "status": UNRUNNABLE,
            "reason": detail,
            "phases": {phase: UNRUNNABLE for phase in PHASES},
            "reached_observation": False,
            "integrity": False,
        }


def execute_authored_check(
        admission: Mapping[str, object], *, parent_snapshot: Mapping[str, object],
        candidate_snapshot: Mapping[str, object], adapter: object = None,
        observation_only: bool = False, runtime: object = None,
) -> dict[str, object]:
    """Run the frozen artifact against both independent snapshots."""
    try:
        if admission.get("admitted") is not True:
            return {"schema": SCHEMA, "status": "REJECTED", "result": UNRUNNABLE,
                    "certifies": False, "certifies_clause": False,
                    "reason": "admission_required"}
        check = _check_from_admission(admission)
        parent_revision, parent_files = _snapshot(parent_snapshot)
        candidate_revision, candidate_files = _snapshot(candidate_snapshot)
        admitted_parent = check.parent_revision
        admitted_candidate = check.candidate_revision
        if parent_revision and admitted_parent and parent_revision != admitted_parent:
            raise AuthoredCheckError("STALE_REVISION", "parent snapshot is not the admitted revision")
        if candidate_revision and admitted_candidate and candidate_revision != admitted_candidate:
            raise AuthoredCheckError("STALE_REVISION", "candidate snapshot is not the admitted revision")
        parent_revision = parent_revision or check.parent_revision
        candidate_revision = candidate_revision or check.candidate_revision
        if not parent_revision or not candidate_revision:
            raise AuthoredCheckError("REVISION", "parent and candidate revisions are required")
        parent_view = {"revision": parent_revision, "files": parent_files}
        candidate_view = {"revision": candidate_revision, "files": candidate_files}
        selected = _adapter_for(check.language, adapter, check.artifact())
        parent_raw = _run_adapter_safely(selected, check.artifact(), parent_view,
                                          revision=parent_revision, side="parent")
        parent = _normalize_execution(
            parent_raw, check=check, revision=parent_revision, side="parent",
            snapshot_digest=_snapshot_digest(parent_files),
        )
        baseline_only = parent_revision == candidate_revision
        if baseline_only:
            candidate = {
                "side": "candidate", "revision": candidate_revision,
                "snapshot_digest": _snapshot_digest(candidate_files),
                "status": "NOT_RUN", "outcome": "NOT_RUN",
                "phases": {phase: "NOT_RUN" for phase in PHASES},
                "reached_observation": False, "integrity": False,
                "exit_code": None, "tests": None, "skipped": None,
                "observation": "baseline-only execution; candidate side not run",
                "reason": "equal_revisions",
            }
        else:
            candidate_raw = _run_adapter_safely(selected, check.artifact(), candidate_view,
                                                revision=candidate_revision, side="candidate")
            candidate = _normalize_execution(
                candidate_raw, check=check, revision=candidate_revision, side="candidate",
                snapshot_digest=_snapshot_digest(candidate_files),
            )
        if observation_only:
            result = OBSERVATION_ONLY
            certifies = False
            status = OBSERVATION_ONLY
        elif baseline_only:
            result, status, certifies = BASELINE_ONLY, UNKNOWN, False
        elif UNSUPPORTED in {parent["status"], candidate["status"]}:
            result, status, certifies = UNSUPPORTED, UNSUPPORTED, False
        elif UNRUNNABLE in {parent["status"], candidate["status"]}:
            result, status, certifies = UNRUNNABLE, UNRUNNABLE, False
        elif ENVIRONMENT in {parent["status"], candidate["status"]}:
            result, status, certifies = ENVIRONMENT, ENVIRONMENT, False
        elif (parent["status"] == BEHAVIOR_FAIL and candidate["status"] == PASS
              and parent["reached_observation"] and candidate["reached_observation"]
              and parent["integrity"] and candidate["integrity"]):
            result, status, certifies = "BEHAVIOR_FIXED", PASS, True
        else:
            result, status, certifies = UNKNOWN, UNKNOWN, False
        if runtime is not None:
            from kernel_authored_check_context import read_runtime_authored_check_context
            context = read_runtime_authored_check_context(runtime)
        else:
            context = {}
        binding = {}
        binding_fn = getattr(runtime, "binding", None) if runtime is not None else None
        if callable(binding_fn):
            try:
                value = binding_fn()
                binding = dict(value) if isinstance(value, Mapping) else {}
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                binding = {}
        execution_identity = _digest({
            "check_identity": check.check_hash,
            "parent_revision": parent_revision,
            "candidate_revision": candidate_revision,
            "parent_snapshot": parent["snapshot_digest"],
            "candidate_snapshot": candidate["snapshot_digest"],
            "run_identity": context.get("run_identity"),
        })
        candidate_digest = _text(
            context.get("candidate_digest")
            or binding.get("candidate_digest")
            or binding.get("artifact")
        ) or _snapshot_digest(candidate_files)
        run_identity = _text(context.get("run_identity")) or execution_identity
        task_id = _text(context.get("task_id")) or "unknown-task"
        public_source_span = copy.deepcopy(context.get("public_source_span", {
            "source_unit_handle": _text(context.get("source_unit_handle"))
            or _text(admission.get("source_unit_handle")) or "unknown-source",
            "quote": check.requirement_quote,
        }))
        simulation = parent.get("simulation") is True and candidate.get("simulation") is True
        receipt = {
            "schema": SCHEMA, "receipt_id": "authored-check:" + _digest({
                "check_hash": check.check_hash, "parent": parent, "candidate": candidate,
                "observation_only": observation_only,
            })[7:31],
            "check_id": check.check_id, "language": check.language,
            "clause_id": check.requirement_id,
            "requirement_id": check.requirement_id, "clause": check.clause,
            "requirement_quote": check.requirement_quote, "selector": check.selector,
            "test_path": check.path, "check_hash": check.check_hash,
            "content_sha256": check.source_sha256,
            "environment_digest": check.environment_digest,
            "parent_revision": parent_revision, "candidate_revision": candidate_revision,
            "baseline_only": baseline_only,
            "parent": parent, "candidate": candidate,
            "parent_outcome": parent["status"], "candidate_outcome": candidate["status"],
            # These normalized identity fields are the canonical host-contract
            # shape consumed by row-role/pass-gates adapters; this module does
            # not create a second completion decision.
            "check_identity": check.check_hash, "oracle_hash": check.check_hash,
            "execution_identity": execution_identity,
            "run_identity": run_identity,
            "task_id": task_id,
            "public_source_span": public_source_span,
            "observable": copy.deepcopy(context.get("observable", check.clause)),
            "scope": copy.deepcopy(context.get("scope", check.clause)),
            "environment_identity": check.environment_digest,
            "candidate_identity": candidate_revision,
            "candidate_digest": candidate_digest,
            "simulation": simulation,
            "source_unit_handle": _text(admission.get("source_unit_handle")),
            "revision": candidate_revision,
            "parent_result": {**parent, "outcome": parent["status"], "revision": parent_revision},
            "candidate_result": {**candidate, "outcome": candidate["status"], "revision": candidate_revision},
            "reached_observation": bool(parent["reached_observation"] and candidate["reached_observation"]),
            "integrity": bool(parent["integrity"] and candidate["integrity"]),
            "binding_stable": _binding_stable(runtime, parent_snapshot, candidate_snapshot),
            "reached_consumers": tuple(_text(item) for item in admission.get("reached_consumers", ())
                                        if _text(item)),
            "coverage_complete": admission.get("coverage_complete") is True,
            "required_partitions": tuple(_text(item) for item in admission.get("required_partitions", ())
                                          if _text(item)),
            "result": result, "status": status, "certifies_clause": certifies,
            "observation_only": observation_only,
            "write_authority": dict(check.write_authority),
            "provenance": {
                "kind": "AUTHORED_CHECK", "source": "host_frozen_pair",
                "simulation": simulation,
            },
        }
        # W3: the host binding is set by the host dispatcher on the admission
        # (never copied from a proposal); absent -> the reducer cannot certify.
        binding = admission.get("host_obligation_binding")
        receipt["host_obligation_binding"] = (copy.deepcopy(dict(binding))
                                              if isinstance(binding, Mapping) else None)
        if runtime is not None:
            receipt = stamp_host_identity(runtime, receipt)
            record_authored_receipt(runtime, receipt)
        return receipt
    except AuthoredCheckError as error:
        return {"schema": SCHEMA, "status": error.code,
                "result": STALE if error.code in {"STALE_CHECK", "STALE_REVISION"} else UNKNOWN,
                "certifies_clause": False, "reason": str(error)}
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return {"schema": SCHEMA, "status": UNRUNNABLE, "result": UNRUNNABLE,
                "certifies_clause": False, "reason": f"host adapter error: {type(error).__name__}"}


def stamp_host_identity(runtime: object, receipt: Mapping[str, object]) -> dict[str, object]:
    """Stamp the runtime's verdict identity onto a host receipt (OFF: unchanged).

    W3: identity validation precedes every certification shortcut, so host
    authored receipts must carry the current run identity to be usable.
    """
    data = dict(receipt)
    binding = data.get("host_obligation_binding")
    if isinstance(binding, Mapping) and _text(binding.get("clause_identity")):
        # This is copied from the host binding created before execution; it is
        # never derived from proposal or model text at receipt-stamp time.
        data["host_clause_identity"] = _text(binding.get("clause_identity"))
    provider = getattr(runtime, "verdict_identity_provider", None)
    if provider is None:
        return data
    from kernel_verdict_identity import IdentityError, stamp_record
    try:
        return dict(stamp_record(data, provider))
    except (IdentityError, TypeError, ValueError):
        # Unstamped evidence stays fail-closed under identity ON.
        return data


def record_authored_receipt(runtime: object, receipt: Mapping[str, object]) -> bool:
    """Append to the existing host receipt stream without creating completion state."""
    receipt_id = _text(receipt.get("receipt_id"))
    if not receipt_id:
        return False
    rows = getattr(runtime, "host_check_receipts", None)
    if not isinstance(rows, list):
        rows = []
        setattr(runtime, "host_check_receipts", rows)
    if not any(isinstance(item, Mapping) and item.get("receipt_id") == receipt_id for item in rows):
        rows.append(dict(receipt))
    row_store = getattr(runtime, "_row_role_authored_receipts", None)
    if not isinstance(row_store, dict):
        row_store = {}
        setattr(runtime, "_row_role_authored_receipts", row_store)
    row_store.setdefault(receipt_id, dict(receipt))
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("authored_check_receipt", producer="kernel_authored_check", event_schema=SCHEMA,
             receipt_id=receipt_id, clause_id=receipt.get("clause_id"),
             result=receipt.get("result"), certifies_clause=receipt.get("certifies_clause") is True,
             parent_revision=receipt.get("parent_revision"), candidate_revision=receipt.get("candidate_revision"),
             dispatch=False)
    return True


def authored_check_schema() -> dict[str, object]:
    """Public proposal schema; source bytes remain host-frozen after admission."""
    return {
        "type": "object",
        "required": ["check_id", "requirement_id", "clause", "requirement_quote",
                     "path", "selector", "source", "write_authority"],
        "properties": {
            "check_id": {"type": "string", "minLength": 1},
            "requirement_id": {"type": "string", "minLength": 1},
            "clause": {"type": "string", "minLength": 1},
            "requirement_quote": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "selector": {"type": "string", "minLength": 1},
            "language": {"enum": sorted(_SUPPORTED_LANGUAGES)},
            "source": {"type": "string", "minLength": 1, "maxLength": 2_000_000},
            "environment": {"type": "object"},
            "observation_marker": {"type": "string"},
            "write_authority": {"type": "object"},
            "source_unit_handle": {"type": "string"},
            "authorized_paths": {"type": "array", "items": {"type": "string"}},
            "protected_paths": {"type": "array", "items": {"type": "string"}},
            "public_text": {"type": "string"},
            "parent_revision": {"type": "string"},
            "candidate_revision": {"type": "string"},
        },
        "additionalProperties": False,
    }


__all__ = [
    "ADMISSION_SCHEMA", "AuthoredCheck", "BINDING_SCHEMA", "binding_certifies", "clause_digest",
    "binding_receipt_ids_are_host_consistent", "clause_identity",
    "normalize_clause", "obligation_binding", "stamp_host_identity", "AuthoredCheckError", "BEHAVIOR_FAIL",
    "BASELINE_ONLY", "ENVIRONMENT", "NON_DISCRIMINATING", "OBSERVATION_ONLY", "PASS", "PHASES",
    "REGRESSION", "SCHEMA", "STALE", "STILL_FAILING",
    "UNKNOWN", "UNRUNNABLE", "UNSUPPORTED", "PythonUnittestAdapter", "UnsupportedAdapter",
    "admit_authored_check", "authored_check_schema", "execute_authored_check",
    "record_authored_receipt",
]
