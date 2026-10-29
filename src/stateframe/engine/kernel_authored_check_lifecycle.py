"""Single-owner lifecycle for host-admitted authored checks.

The existing authored-check module owns proposal validation and the public tool
adapter.  This module owns only the durable boundary around it: logical check
versions, immutable executable bytes, one-side execution, pair reduction,
bounded replay scheduling, and attachment of typed receipts to existing
consumers.  It never chooses a requirement, writes the workspace, or decides
completion.
"""
from __future__ import annotations

import hashlib
import asyncio
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from kernel_authored_check import (
    BASELINE_ONLY,
    BEHAVIOR_FAIL,
    ENVIRONMENT,
    PASS,
    REGRESSION,
    UNKNOWN,
    UNRUNNABLE,
    UNSUPPORTED,
    admit_authored_check,
    record_authored_receipt,
)


SCHEMA = "stateframe-authored-check-lifecycle/1"
REPAIR = "REPAIR"
PRESERVATION = "PRESERVATION"
PAIR_OUTCOMES = frozenset({
    "BEHAVIOR_FIXED", "PRESERVED", "NON_DISCRIMINATING", "STILL_FAILING",
    REGRESSION, BASELINE_ONLY, UNKNOWN, UNRUNNABLE, UNSUPPORTED, ENVIRONMENT,
    "OBSERVATION_ONLY",
})
_NON_BEHAVIORAL = frozenset({
    "", "NOT_RUN", "UNKNOWN", "UNRUNNABLE", "UNSUPPORTED", "ENVIRONMENT",
    "STALE", "TIMEOUT", "TIMED_OUT", "SETUP_FAIL", "COMPILE_FAIL", "SKIPPED",
})


class LifecycleError(ValueError):
    """A host-owned lifecycle contract refusal."""


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else str(value or "").strip()


def _await(value: object) -> object:
    """Resolve an adapter coroutine without leaking it into a receipt."""
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(value)
    import threading
    result: list[object] = []
    errors: list[BaseException] = []

    def resolve() -> None:
        try:
            result.append(asyncio.run(value))
        except BaseException as error:  # pragma: no cover - loop bridge
            errors.append(error)

    thread = threading.Thread(target=resolve, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0] if result else None


def _digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8")
    raise LifecycleError("all executable artifacts must contain text or bytes")


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _relative(path: object) -> str:
    value = _text(path).replace("\\", "/")
    parts = value.split("/")
    if not value or value.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise LifecycleError(f"non-canonical artifact path: {value!r}")
    return value


def _manifest(files: Mapping[str, object]) -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    for raw_path, raw_source in sorted(files.items(), key=lambda item: str(item[0])):
        path = _relative(raw_path)
        data = _bytes(raw_source)
        result.append({
            "path": path,
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "bytes": str(len(data)),
        })
    return tuple(result)


def _host_context(context: object) -> Mapping[str, object]:
    if isinstance(context, Mapping):
        return context
    values = getattr(context, "__dict__", None)
    return values if isinstance(values, Mapping) else {}


@dataclass(frozen=True)
class FrozenCheck:
    """Immutable logical check/version with content-addressed executable bytes."""

    check_id: str
    version: str
    identity: str
    role: str
    clause_id: str
    admission: Mapping[str, object]
    check_bytes: bytes
    fixture_bytes: Mapping[str, bytes]
    harness_bytes: Mapping[str, bytes]
    dependency_bytes: Mapping[str, bytes]
    dependency_manifest: tuple[Mapping[str, str], ...]
    ticket_span: Mapping[str, object]
    observation_contract: Mapping[str, object]
    store: object | None = field(default=None, repr=False, compare=False)
    executable: object | None = field(default=None, repr=False, compare=False)

    @property
    def check_hash(self) -> str:
        return _text(self.admission.get("check_hash"))

    @property
    def path(self) -> str:
        return _text(self.admission.get("artifact", {}).get("path")) \
            if isinstance(self.admission.get("artifact"), Mapping) else ""

    @property
    def artifact(self) -> Mapping[str, object]:
        value = self.admission.get("artifact")
        return value if isinstance(value, Mapping) else {}

    def execution_identity(self, revision: object, snapshot_digest: object,
                           *, side: str) -> str:
        return _digest({
            "check_identity": self.identity,
            "revision": _text(revision),
            "snapshot_digest": _text(snapshot_digest),
            "side": _text(side),
        })

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "check_id": self.check_id,
            "version": self.version,
            "identity": self.identity,
            "role": self.role,
            "clause_id": self.clause_id,
            "admission": dict(self.admission),
            "dependency_manifest": [dict(item) for item in self.dependency_manifest],
            "ticket_span": dict(self.ticket_span),
            "observation_contract": dict(self.observation_contract),
        }


@dataclass(frozen=True)
class ExecutionResult:
    check_identity: str
    side: str
    revision: str
    snapshot_digest: str
    status: str
    outcome: str
    reached_target: bool
    integrity: bool
    phases: Mapping[str, str]
    tests: int | None = None
    skipped: int | None = None
    events: tuple[Mapping[str, object], ...] = ()
    reason: str = ""
    target_exception: str = ""
    binding_stable: bool = False
    raw: Mapping[str, object] = field(default_factory=dict, repr=False, compare=False)
    manifest: tuple[Mapping[str, str], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "check_identity": self.check_identity,
            "side": self.side,
            "revision": self.revision,
            "snapshot_digest": self.snapshot_digest,
            "status": self.status,
            "outcome": self.outcome,
            "reached_target": self.reached_target,
            "reached_observation": self.reached_target,
            "integrity": self.integrity,
            "phases": dict(self.phases),
            "tests": self.tests,
            "skipped": self.skipped,
            "events": [dict(item) for item in self.events],
            "reason": self.reason,
            "target_exception": self.target_exception,
            "binding_stable": self.binding_stable,
            "manifest": [dict(item) for item in self.manifest],
        }


def _source_map(value: object) -> dict[str, bytes]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, bytes] = {}
    for raw_path, raw_source in value.items():
        try:
            result[_relative(raw_path)] = _bytes(raw_source)
        except LifecycleError:
            continue
    return result


def _store_for(context: Mapping[str, object]) -> object:
    store = context.get("content_store")
    if store is not None:
        return store
    from witness_runner import ContentStore
    return ContentStore()


def _freeze_executable(store: object, files: Mapping[str, bytes], entrypoint: str,
                       context: Mapping[str, object]) -> object | None:
    try:
        from witness_runner import freeze_executable
        fixture_digest = _digest(_manifest(files))
        return freeze_executable(
            store, files, entrypoint,
            fixture_digest=fixture_digest,
            argv=tuple(str(item) for item in _sequence(context.get("argv", ()))),
            environment=(context.get("environment")
                         if isinstance(context.get("environment"), Mapping) else {}),
            runtime_image_digest=_text(context.get("runtime_image_digest")) or None,
            dependency_lock_digest=_text(context.get("dependency_lock_digest")) or "sha256:none",
            sandbox_policy_digest=_text(context.get("sandbox_policy_digest")) or None,
            deterministic_seed=context.get("deterministic_seed", 0),
            network_policy="disabled",
        )
    except (ImportError, OSError, TypeError, ValueError):
        return None


def freeze_check(context: object, proposal: Mapping[str, object]) -> FrozenCheck:
    """Freeze one public proposal using host-owned identity and scratch authority."""
    host = _host_context(context)
    if not isinstance(proposal, Mapping):
        raise LifecycleError("authored check proposal must be a mapping")
    public_text = host.get("public_text")
    parent_revision = _text(host.get("parent_revision"))
    candidate_revision = _text(host.get("candidate_revision", parent_revision))
    if not isinstance(public_text, str) or not public_text.strip():
        raise LifecycleError("host public text is required")
    if not parent_revision or not candidate_revision:
        raise LifecycleError("host parent and candidate revisions are required")
    authority = host.get("scratch_authority", host.get("write_authority"))
    if not isinstance(authority, Mapping):
        raise LifecycleError("host scratch authority is required")
    allowed_paths = tuple(_text(item) for item in _sequence(authority.get("allowed_paths")) if _text(item))
    protected_paths = tuple(_text(item) for item in _sequence(host.get("protected_paths", ())) if _text(item))
    proposal_path = _relative(proposal.get("path"))
    if proposal_path in set(protected_paths):
        raise LifecycleError("authored check path overlaps a protected path")
    if allowed_paths and proposal_path not in set(allowed_paths):
        raise LifecycleError("authored check path is outside host scratch authority")

    host_proposal = dict(proposal)
    # These fields are deliberately overwritten, even when the model supplied
    # plausible values.  The host is the only authority for provenance/revision.
    host_proposal.pop("public_text", None)
    host_proposal.pop("parent_revision", None)
    host_proposal.pop("candidate_revision", None)
    host_proposal["write_authority"] = dict(authority)
    admission = admit_authored_check(
        host_proposal,
        public_text=public_text,
        parent_revision=parent_revision,
        candidate_revision=candidate_revision,
        authorized_paths=allowed_paths,
        protected_paths=protected_paths,
        existing_check_ids=(),
    )
    if admission.get("admitted") is not True:
        raise LifecycleError(f"authored check rejected: {admission.get('code', 'UNKNOWN')}")
    artifact = admission.get("artifact")
    if not isinstance(artifact, Mapping):
        raise LifecycleError("admission did not return a frozen artifact")
    invocation = host.get("invocation", host.get("check_invocation"))
    if isinstance(invocation, Mapping):
        artifact = dict(artifact)
        artifact["invocation"] = dict(invocation)
        admission = dict(admission)
        admission["artifact"] = artifact
    fixture_bytes = _source_map(host.get("fixture_files"))
    harness_bytes = _source_map(host.get("harness_files"))
    dependency_bytes = _source_map(host.get("dependency_files"))
    check_bytes = _bytes(artifact.get("source"))
    files = {proposal_path: check_bytes}
    for group in (fixture_bytes, harness_bytes, dependency_bytes):
        for path, source in group.items():
            if path in files and files[path] != source:
                raise LifecycleError(f"frozen artifact collision: {path}")
            files[path] = source
    store = _store_for(host)
    executable = _freeze_executable(store, files, proposal_path, host)
    dependency_manifest = tuple(dict(item) for item in _manifest(files))
    role = _text(host.get("role", proposal.get("role", REPAIR))).upper() or REPAIR
    if role not in {REPAIR, PRESERVATION}:
        raise LifecycleError("authored check role must be REPAIR or PRESERVATION")
    ticket_span = host.get("ticket_span", {})
    ticket_span = dict(ticket_span) if isinstance(ticket_span, Mapping) else {}
    observation_contract = host.get("observation_contract", {})
    observation_contract = dict(observation_contract) if isinstance(observation_contract, Mapping) else {}
    version = _digest({
        "check_hash": _text(admission.get("check_hash")),
        "fixture_manifest": dependency_manifest,
        "role": role,
        "ticket_span": ticket_span,
        "observation_contract": observation_contract,
    })
    identity = _digest({
        "check_hash": _text(admission.get("check_hash")),
        "check_id": _text(admission.get("check_id")),
        "version": version,
    })
    admission = dict(admission)
    admission["check_version"] = version
    admission["logical_check_identity"] = identity
    admission["role"] = role
    admission["dependency_manifest"] = [dict(item) for item in dependency_manifest]
    admission["ticket_span"] = ticket_span
    admission["observation_contract"] = observation_contract
    return FrozenCheck(
        check_id=_text(admission.get("check_id")), version=version, identity=identity,
        role=role, clause_id=_text(admission.get("clause_id")), admission=admission,
        check_bytes=check_bytes, fixture_bytes=fixture_bytes, harness_bytes=harness_bytes,
        dependency_bytes=dependency_bytes,
        dependency_manifest=dependency_manifest, ticket_span=ticket_span,
        observation_contract=observation_contract, store=store, executable=executable,
    )


def _snapshot_files(snapshot: Mapping[str, object]) -> dict[str, bytes]:
    value = snapshot.get("files", snapshot)
    if not isinstance(value, Mapping):
        return {}
    return _source_map(value)


def _snapshot_digest(files: Mapping[str, bytes]) -> str:
    return _digest(_manifest(files))


def _execution_files(check: FrozenCheck, snapshot: Mapping[str, object]) -> dict[str, bytes] | None:
    """Join mutable product bytes with immutable check/fixture/harness bytes."""
    files = _snapshot_files(snapshot)
    frozen = {check.path: check.check_bytes,
              **check.fixture_bytes, **check.harness_bytes, **check.dependency_bytes}
    for path, source in frozen.items():
        existing = files.get(path)
        if existing is not None and existing != source:
            return None
        files[path] = source
    return files


def _snapshot_complete(check: FrozenCheck, snapshot: Mapping[str, object]) -> tuple[bool, str]:
    files = _execution_files(check, snapshot)
    if files is None:
        return False, "SNAPSHOT_FROZEN_ARTIFACT_COLLISION"
    for item in check.dependency_manifest:
        path = item.get("path", "")
        expected = item.get("sha256", "")
        source = files.get(path)
        if source is None:
            return False, f"SNAPSHOT_INCOMPLETE:{path}"
        actual = "sha256:" + hashlib.sha256(source).hexdigest()
        if expected and actual != expected:
            return False, f"SNAPSHOT_DRIFT:{path}"
    declared = snapshot.get("manifest")
    if declared is not None:
        if snapshot.get("manifest_complete") is not True:
            return False, "SNAPSHOT_MANIFEST_INCOMPLETE"
        if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes, bytearray)):
            return False, "SNAPSHOT_MANIFEST_INVALID"
        for item in declared:
            if not isinstance(item, Mapping):
                return False, "SNAPSHOT_MANIFEST_INVALID"
            try:
                path = _relative(item.get("path"))
            except LifecycleError:
                return False, "SNAPSHOT_MANIFEST_INVALID_PATH"
            source = files.get(path)
            if source is None:
                return False, f"SNAPSHOT_INCOMPLETE:{path}"
            expected = _text(item.get("sha256"))
            actual = "sha256:" + hashlib.sha256(source).hexdigest()
            if expected and expected != actual:
                return False, f"SNAPSHOT_DRIFT:{path}"
    return True, ""


def _execution_from_raw(check: FrozenCheck, raw: object, *, side: str,
                        revision: str, snapshot_digest: str,
                        manifest: tuple[Mapping[str, str], ...] = ()) -> ExecutionResult:
    data = dict(raw) if isinstance(raw, Mapping) else {}
    status = _text(data.get("status", data.get("outcome", UNKNOWN))).upper()
    allowed = {PASS, BEHAVIOR_FAIL, UNKNOWN, UNRUNNABLE, UNSUPPORTED, ENVIRONMENT, "STALE", "TIMEOUT"}
    if status not in allowed:
        status = UNKNOWN
    events_value = data.get("events", ())
    events = tuple(dict(item) for item in events_value
                   if isinstance(item, Mapping)) if isinstance(events_value, Sequence) \
        and not isinstance(events_value, (str, bytes, bytearray)) else ()
    event_names = {str(item.get("kind", item.get("event", ""))).lower() for item in events}
    reached = data.get("target_entry_reached", data.get("reached_observation"))
    if not isinstance(reached, bool):
        reached = bool({"target_entry", "target_reached", "observation_reached"} & event_names)
    phases = data.get("phases", data.get("phase_outcomes", {}))
    phases = {str(key): _text(value).upper() for key, value in phases.items()} \
        if isinstance(phases, Mapping) else {}
    for phase in ("collection", "setup"):
        if phases.get(phase) in {UNRUNNABLE, UNSUPPORTED, ENVIRONMENT}:
            status = phases[phase]
    tests = data.get("tests", data.get("test_count"))
    tests = tests if type(tests) is int else None
    skipped = data.get("skipped", data.get("skip_count"))
    skipped = skipped if type(skipped) is int else None
    if status in {PASS, BEHAVIOR_FAIL} and not reached:
        status = UNKNOWN
    if tests is not None and (tests <= 0 or (skipped is not None and skipped >= tests)):
        status = UNKNOWN
    target_exception = _text(data.get("target_exception"))
    if target_exception and reached and data.get("no_failure_assertion") is True:
        status = BEHAVIOR_FAIL
    return ExecutionResult(
        check_identity=check.identity, side=side, revision=revision,
        snapshot_digest=snapshot_digest, status=status, outcome=status,
        reached_target=reached, integrity=data.get("integrity") is True,
        phases=phases, tests=tests, skipped=skipped, events=events,
        reason=_text(data.get("reason")), target_exception=target_exception,
        binding_stable=data.get("binding_stable") is True, raw=data,
        manifest=manifest,
    )


def execute_check(check: FrozenCheck, snapshot: Mapping[str, object],
                  capability: object = None) -> ExecutionResult:
    """Execute exactly one frozen check side against one complete snapshot."""
    if not isinstance(check, FrozenCheck):
        raise LifecycleError("execute_check requires FrozenCheck")
    if not isinstance(snapshot, Mapping):
        return ExecutionResult(check.identity, "unknown", "", "", UNRUNNABLE,
                               UNRUNNABLE, False, False, {}, reason="snapshot_mapping_required")
    side = _text(snapshot.get("side", "candidate")).lower() or "candidate"
    revision = _text(snapshot.get("revision", snapshot.get("product_revision")))
    if not revision:
        return ExecutionResult(check.identity, side, "", "", UNRUNNABLE,
                               UNRUNNABLE, False, False, {}, reason="snapshot_revision_required")
    files = _execution_files(check, snapshot)
    if files is None:
        return ExecutionResult(check.identity, side, revision, "", UNRUNNABLE,
                               UNRUNNABLE, False, False, {}, reason="SNAPSHOT_FROZEN_ARTIFACT_COLLISION")
    digest = _snapshot_digest(files)
    complete, reason = _snapshot_complete(check, snapshot)
    if not complete:
        return ExecutionResult(check.identity, side, revision, digest, UNRUNNABLE,
                               UNRUNNABLE, False, False, {}, reason=reason)
    runner = getattr(capability, "run", None) if capability is not None else None
    if not callable(runner) and callable(capability):
        runner = capability
    if runner is None:
        try:
            from authored_check_adapters import adapter_for
            adapter = adapter_for(_text(check.artifact.get("language")))
            runner = getattr(adapter, "run", None)
        except (ImportError, OSError, TypeError, ValueError):
            runner = None
    if not callable(runner):
        return ExecutionResult(check.identity, side, revision, digest, UNSUPPORTED,
                               UNSUPPORTED, False, False, {}, reason="adapter_unavailable")
    try:
        raw = _await(runner(
            check.artifact, {"revision": revision, "files": files,
                             "side": side, "binding_stable": snapshot.get("binding_stable")},
            revision=revision, side=side))
    except TypeError:
        try:
            raw = _await(runner(check.artifact, {"revision": revision, "files": files}, revision, side))
        except Exception as error:  # execution boundary
            raw = {"status": UNRUNNABLE, "reason": f"adapter_error:{type(error).__name__}"}
    except Exception as error:  # execution boundary
        raw = {"status": UNRUNNABLE, "reason": f"adapter_error:{type(error).__name__}"}
    return _execution_from_raw(check, raw, side=side, revision=revision,
                               snapshot_digest=digest, manifest=_manifest(files))


def _as_execution(value: object) -> ExecutionResult:
    if isinstance(value, ExecutionResult):
        return value
    if isinstance(value, Mapping):
        return ExecutionResult(
            check_identity=_text(value.get("check_identity", value.get("check_hash"))),
            side=_text(value.get("side", "unknown")), revision=_text(value.get("revision")),
            snapshot_digest=_text(value.get("snapshot_digest")),
            status=_text(value.get("status", value.get("outcome", UNKNOWN))).upper(),
            outcome=_text(value.get("outcome", value.get("status", UNKNOWN))).upper(),
            reached_target=value.get("reached_target", value.get("reached_observation")) is True,
            integrity=value.get("integrity") is True,
            phases=value.get("phases", {}) if isinstance(value.get("phases", {}), Mapping) else {},
            tests=value.get("tests") if type(value.get("tests")) is int else None,
            skipped=value.get("skipped") if type(value.get("skipped")) is int else None,
            events=tuple(item for item in _sequence(value.get("events", ())) if isinstance(item, Mapping)),
            reason=_text(value.get("reason")), target_exception=_text(value.get("target_exception")),
            binding_stable=value.get("binding_stable") is True, raw=value,
        )
    raise LifecycleError("pair_executions requires execution receipts")


def pair_executions(parent: object, candidate: object, *, role: str = REPAIR,
                    clause_id: str = "", check: FrozenCheck | None = None) -> dict[str, object]:
    """Reduce two executions without turning setup failure into product behavior."""
    before, after = _as_execution(parent), _as_execution(candidate)
    role_value = _text(role).upper() or REPAIR
    identity = before.check_identity
    if identity != after.check_identity or (check is not None and identity != check.identity):
        outcome = UNKNOWN
        reason = "CHECK_IDENTITY_MISMATCH"
    elif before.revision == after.revision:
        outcome = BASELINE_ONLY
        reason = "equal_revisions_baseline_only"
    elif before.status in _NON_BEHAVIORAL or after.status in _NON_BEHAVIORAL:
        outcome = UNRUNNABLE if {before.status, after.status} & {UNRUNNABLE, UNSUPPORTED} else UNKNOWN
        reason = before.reason or after.reason or "execution_not_behaviorally_attributable"
    elif not (before.reached_target and after.reached_target
              and before.integrity and after.integrity):
        outcome, reason = UNKNOWN, "target_reach_or_integrity_incomplete"
    elif role_value == PRESERVATION:
        if before.status == PASS and after.status == PASS:
            outcome, reason = "PRESERVED", "current_preservation_pass"
        elif before.status == PASS and after.status == BEHAVIOR_FAIL:
            outcome, reason = REGRESSION, "current_preservation_regression"
        else:
            outcome, reason = UNKNOWN, "preservation_parent_not_green"
    elif before.status == BEHAVIOR_FAIL and after.status == PASS:
        outcome, reason = "BEHAVIOR_FIXED", "parent_behavior_fail_candidate_pass"
    elif before.status == PASS and after.status == PASS:
        outcome, reason = "NON_DISCRIMINATING", "parent_and_candidate_green"
    elif before.status == BEHAVIOR_FAIL and after.status == BEHAVIOR_FAIL:
        outcome, reason = "STILL_FAILING", "parent_and_candidate_behavior_fail"
    elif before.status == PASS and after.status == BEHAVIOR_FAIL:
        outcome, reason = REGRESSION, "candidate_regressed"
    else:
        outcome, reason = UNKNOWN, "unrecognized_pair"
    certifies = outcome == "BEHAVIOR_FIXED" and role_value == REPAIR
    execution_identity = _digest({
        "check_identity": identity,
        "parent": before.as_dict(),
        "candidate": after.as_dict(),
    })
    source_span = dict(check.ticket_span) if check is not None else {}
    if not source_span:
        source_span = {"path": check.path if check is not None else "",
                       "provenance": "host_frozen_check"}
    environment_identity = (
        _text(check.artifact.get("environment_digest")) if check is not None else ""
    )
    source_unit_handle = (
        _text(check.admission.get("source_unit_handle")) if check is not None else ""
    ) or (check.path if check is not None else "")
    task_id = _text(check.admission.get("task_id")) if check is not None else ""
    task_id = task_id or (check.clause_id if check is not None else "")
    return {
        "schema": SCHEMA,
        "receipt_id": "authored-lifecycle:" + _digest({
            "identity": identity, "parent": before.as_dict(), "candidate": after.as_dict(),
            "role": role_value,
        })[7:31],
        "check_identity": identity,
        "check_hash": identity,
        "oracle_hash": identity,
        "execution_identity": execution_identity,
        "run_identity": execution_identity,
        "task_id": task_id,
        "public_source_span": source_span,
        "environment_identity": environment_identity,
        "source_unit_handle": source_unit_handle,
        "check_id": check.check_id if check is not None else "",
        "check_version": check.version if check is not None else "",
        "language": _text(check.artifact.get("language")) if check is not None else "",
        "test_path": check.path if check is not None else "",
        "selector": _text(check.artifact.get("selector")) if check is not None else "",
        "requirement_quote": _text(check.artifact.get("requirement_quote")) if check is not None else "",
        "clause": _text(check.artifact.get("clause")) if check is not None else "",
        "content_sha256": _text(check.artifact.get("source_sha256")) if check is not None else "",
        "write_authority": (dict(check.artifact.get("write_authority", {}))
                            if check is not None and isinstance(check.artifact.get("write_authority"), Mapping)
                            else {}),
        "clause_id": _text(clause_id) or (check.clause_id if check is not None else ""),
        "requirement_id": _text(clause_id) or (check.clause_id if check is not None else ""),
        "role": role_value,
        "parent": before.as_dict(), "candidate": after.as_dict(),
        "parent_result": before.as_dict(), "candidate_result": after.as_dict(),
        "parent_revision": before.revision, "candidate_revision": after.revision,
        "revision": after.revision,
        "parent_identity": _digest({"revision": before.revision,
                                     "snapshot": before.snapshot_digest}),
        "candidate_identity": _digest({"revision": after.revision,
                                        "snapshot": after.snapshot_digest}),
        "candidate_digest": after.snapshot_digest,
        "parent_manifest": [dict(item) for item in before.manifest],
        "candidate_manifest": [dict(item) for item in after.manifest],
        "manifest_complete": bool(before.manifest and after.manifest),
        "parent_outcome": before.status, "candidate_outcome": after.status,
        "reached_observation": before.reached_target and after.reached_target,
        "integrity": before.integrity and after.integrity,
        "binding_stable": before.binding_stable and after.binding_stable,
        "result": outcome, "status": PASS if outcome in {"BEHAVIOR_FIXED", "PRESERVED"} else outcome,
        "reason": reason, "certifies_clause": certifies,
        "observation_only": False,
        "provenance": {"kind": "AUTHORED_CHECK", "source": "host_frozen_pair"},
    }


def register_frozen_check(runtime: object, check: FrozenCheck) -> FrozenCheck:
    """Persist logical/version identity; identical delivery is idempotent."""
    registry = getattr(runtime, "_authored_check_versions", None)
    if not isinstance(registry, dict):
        registry = {}
        setattr(runtime, "_authored_check_versions", registry)
    key = (check.check_id, check.version)
    existing = registry.get(key)
    if isinstance(existing, FrozenCheck):
        if existing.identity != check.identity:
            raise LifecycleError("frozen check version identity collision")
        return existing
    registry[key] = check
    return check


def attach_receipt(runtime: object, receipt: Mapping[str, object]) -> bool:
    """Fold one normalized receipt into existing host stores and consumers."""
    if (not isinstance(receipt, Mapping)
            or not _text(receipt.get("receipt_id"))
            or not _text(receipt.get("receipt_id")).startswith("authored-lifecycle:")):
        return False
    if receipt.get("schema") != SCHEMA:
        return False
    provenance = receipt.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("source") != "host_frozen_pair":
        return False
    required = (
        "check_identity", "check_hash", "execution_identity", "run_identity",
        "check_id", "check_version",
        "parent_revision", "candidate_revision", "parent_result", "candidate_result",
        "result", "status", "role", "manifest_complete",
    )
    if any(key not in receipt for key in required):
        return False
    if _text(receipt.get("result")).upper() not in PAIR_OUTCOMES:
        return False
    if _text(receipt.get("role")).upper() not in {REPAIR, PRESERVATION}:
        return False
    if (_text(receipt.get("check_hash")) != _text(receipt.get("check_identity"))
            or _text(receipt.get("run_identity")) != _text(receipt.get("execution_identity"))):
        return False
    if (not isinstance(receipt.get("parent_result"), Mapping)
            or not isinstance(receipt.get("candidate_result"), Mapping)):
        return False
    parent_result = receipt["parent_result"]
    candidate_result = receipt["candidate_result"]
    if (_text(parent_result.get("revision")) != _text(receipt.get("parent_revision"))
            or _text(candidate_result.get("revision")) != _text(receipt.get("candidate_revision"))
            or _text(parent_result.get("check_identity")) != _text(receipt.get("check_identity"))
            or _text(candidate_result.get("check_identity")) != _text(receipt.get("check_identity"))):
        return False
    try:
        from kernel_authored_check import (
            BINDING_SCHEMA, binding_receipt_ids_are_host_consistent, stamp_host_identity,
        )
    except (ImportError, AttributeError):
        return False
    data = stamp_host_identity(runtime, dict(receipt))
    binding = data.get("host_obligation_binding")
    if not isinstance(binding, Mapping):
        return False
    if (binding.get("schema") != BINDING_SCHEMA
            or not _text(binding.get("obligation_id"))
            or not _text(binding.get("clause_sha256"))
            or not binding_receipt_ids_are_host_consistent(data)):
        return False
    requirement_id = _text(data.get("requirement_id", data.get("clause_id")))
    # A frame requirement id may alias the bundle's canonical obligation id;
    # the host clause identity is the authority for that cross-namespace join.
    accepted_requirement_ids = {
        _text(binding.get("obligation_id")),
        _text(binding.get("host_requirement_id")),
    }
    if requirement_id and requirement_id not in accepted_requirement_ids:
        return False
    data.setdefault("host_admitted", True)
    data.setdefault("immutable", True)
    data.setdefault("solver_authored", True)
    data.setdefault("source_unit_handle", "")
    data.setdefault("coverage_complete", False)
    record_authored_receipt(runtime, data)
    clause_receipts = getattr(runtime, "_clause_discriminator_receipts", None)
    if not isinstance(clause_receipts, list):
        clause_receipts = []
        setattr(runtime, "_clause_discriminator_receipts", clause_receipts)
    if not any(isinstance(item, Mapping) and item.get("receipt_id") == data.get("receipt_id")
               for item in clause_receipts):
        clause_receipts.append(data)
    store = getattr(runtime, "_authored_check_execution_receipts", None)
    if not isinstance(store, dict):
        store = {}
        setattr(runtime, "_authored_check_execution_receipts", store)
    store.setdefault(_text(data.get("execution_identity", data.get("receipt_id"))), data)
    try:
        from kernel_completion_owner import recompute_authored_completion
        recompute_authored_completion(runtime)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # Receipt storage remains canonical even for small offline runtimes
        # that do not attach the full completion owner.
        pass
    return True


def schedule_bound_checks(edit_receipt: Mapping[str, object],
                          frozen_checks: Sequence[FrozenCheck],
                          remaining_budget: object) -> dict[str, object]:
    """Plan at most one host replay per changed revision/check identity."""
    edit = edit_receipt if isinstance(edit_receipt, Mapping) else {}
    revision = _text(edit.get("candidate_revision", edit.get("current_revision")))
    changed = {str(item).replace("\\", "/") for item in _sequence(edit.get("changed_paths", edit.get("paths", ())))}
    dependencies = changed | {str(item).replace("\\", "/") for item in _sequence(edit.get("dependency_paths", ( )))}
    budget = remaining_budget if isinstance(remaining_budget, Mapping) else {}
    limit = budget.get("max_checks", budget.get("remaining_checks", len(frozen_checks)))
    limit = limit if type(limit) is int and limit >= 0 else len(frozen_checks)
    jobs: list[dict[str, object]] = []
    omitted: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for check in frozen_checks:
        if len(jobs) >= limit:
            omitted.append({"check_id": check.check_id, "reason": "budget_exhausted"})
            continue
        relevant = not dependencies or any(
            path == check.path or path in {item.get("path") for item in check.dependency_manifest}
            for path in dependencies
        )
        if not relevant:
            omitted.append({"check_id": check.check_id, "reason": "unaffected_edit"})
            continue
        key = (check.identity, revision, check.role)
        if key in seen:
            continue
        seen.add(key)
        jobs.append({
            "check": check,
            "check_identity": check.identity,
            "check_version": check.version,
            "candidate_revision": revision,
            "execution_key": _digest({"check": check.identity, "revision": revision}),
            "role": check.role,
            "reason": "relevant_edit",
        })
    return {"schema": SCHEMA, "jobs": tuple(jobs), "omitted": tuple(omitted),
            "scheduled": len(jobs), "revision": revision}


def replay_bound_checks(runtime: object, schedule: Mapping[str, object],
                        snapshots: Mapping[str, object], capability: object = None
                        ) -> tuple[dict[str, object], ...]:
    """Execute each scheduled frozen check once against supplied scratch snapshots."""
    jobs = schedule.get("jobs", ()) if isinstance(schedule, Mapping) else ()
    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes, bytearray)):
        return ()
    results: list[dict[str, object]] = []
    for job in jobs:
        if not isinstance(job, Mapping) or not isinstance(job.get("check"), FrozenCheck):
            continue
        check = job["check"]
        ref = snapshots.get(check.identity) if isinstance(snapshots, Mapping) else None
        if not isinstance(ref, Mapping):
            results.append({"schema": SCHEMA, "result": UNRUNNABLE, "status": UNRUNNABLE,
                            "check_identity": check.identity,
                            "reason": "replay_snapshots_unavailable"})
            continue
        parent_snapshot = ref.get("parent")
        candidate_snapshot = ref.get("candidate")
        if not isinstance(parent_snapshot, Mapping) or not isinstance(candidate_snapshot, Mapping):
            results.append({"schema": SCHEMA, "result": UNRUNNABLE, "status": UNRUNNABLE,
                            "check_identity": check.identity,
                            "reason": "replay_parent_candidate_snapshots_required"})
            continue
        parent = execute_check(check, {**parent_snapshot, "side": "parent"}, capability)
        candidate = execute_check(check, {**candidate_snapshot, "side": "candidate"}, capability)
        receipt = pair_executions(parent, candidate, role=check.role,
                                  clause_id=check.clause_id, check=check)
        try:
            from kernel_clause_discriminator import _host_obligation_binding
            binding = _host_obligation_binding(runtime, check.admission)
        except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            binding = None
        if binding is not None:
            receipt["host_obligation_binding"] = dict(binding)
            receipt["coverage_complete"] = check.admission.get("coverage_complete") is True
            if "source_unit_handle" in check.admission:
                receipt["source_unit_handle"] = check.admission["source_unit_handle"]
            attach_receipt(runtime, receipt)
        else:
            receipt["reason"] = "host_obligation_binding_unavailable"
            receipt["certifies_clause"] = False
        results.append(receipt)
    return tuple(results)


__all__ = [
    "BASELINE_ONLY", "ExecutionResult", "FrozenCheck", "LifecycleError", "PAIR_OUTCOMES",
    "PRESERVATION", "REPAIR", "SCHEMA", "attach_receipt", "execute_check",
    "freeze_check", "pair_executions", "register_frozen_check", "replay_bound_checks",
    "schedule_bound_checks",
]
