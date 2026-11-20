"""Host-side guard and delivery proofs for a single run attempt."""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from kernel_turn_contracts import (
    Artifact,
    Basis,
    Context,
    Corrupt,
    Dep,
    Facts,
    Key,
    Proofs,
    Publication,
    Ref,
    Stale,
    Unavailable,
    Workspace,
)

MODE_ENV = "STATEFRAME_TURN_DELIVERY_PROOFS"
_SCOPE = "turn/scope/1"
_INPUTS = "turn/guard-inputs/1"
_GUARD = "turn/guard/1"
_CANDIDATE = "turn/delivery-candidate/1"
_VERDICT = "turn/verdict/1"
_EVIDENCE = ("selector_coverage", "public_harness", "menu", "obligations")

Use = Literal["menu", "send", "act"]
LegacyVerifier = Callable[[bytes, Mapping[str, object]], Mapping[str, object]]


def enabled() -> bool:
    return os.environ.get(MODE_ENV, "off").strip().lower() == "on"


def _encode(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise Corrupt("invalid proof record") from error


def _load(publication: Publication, ref: Ref, codec: str) -> dict[str, object]:
    if ref.codec != codec:
        raise Corrupt("unexpected proof record codec")
    data = publication.load(ref)
    try:
        value = json.loads(data)
    except (ValueError, RecursionError) as error:
        raise Corrupt("invalid proof record") from error
    if not isinstance(value, dict) or _encode(value) != data:
        raise Corrupt("noncanonical proof record")
    return value


def _ref(value: object) -> Ref:
    if isinstance(value, list) and len(value) == 3:
        codec, digest, size = value
        if (isinstance(codec, str) and isinstance(digest, str)
                and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest)
                and type(size) is int and size >= 0):
            return Ref(codec, digest, size)
    raise Corrupt("invalid proof reference")


def _dependencies(value: object) -> tuple[Dep, ...]:
    if not isinstance(value, list) or not value:
        raise Unavailable("guard dependency coverage is missing")
    deps = []
    for row in value:
        if (not isinstance(row, list) or len(row) != 3
                or not all(isinstance(item, str) for item in row) or not row[0] or not row[2]):
            raise Corrupt("invalid guard dependency")
        family, selector, revision = row
        try:
            selector_bytes = bytes.fromhex(selector)
        except ValueError as error:
            raise Corrupt("invalid guard dependency selector") from error
        if selector_bytes.hex() != selector:
            raise Corrupt("noncanonical guard dependency selector")
        deps.append(Dep(Key(family, selector_bytes), revision))
    if len({dep.key for dep in deps}) != len(deps):
        raise Corrupt("duplicate guard dependency")
    return tuple(deps)


def _strings(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise Corrupt("invalid guard action or blocker list")
    return value


def _basis(at: Basis) -> dict[str, object]:
    return {**at._asdict(), "journal": at.journal._asdict()}


def _namespace(ctx: Context) -> list[str]:
    return [ctx.principal, ctx.run, ctx.attempt, ctx.profile_digest]


def stage_guard_inputs(
        ctx: Context, publication: Publication, at: Basis, use: Use, *,
        scope: Ref, deps: Sequence[Dep], evidence: Mapping[str, Ref],
        allowed_actions: Sequence[str], blockers: Sequence[str],
) -> Ref:
    """Capture a host decision after mandatory pre-use checks and automatic verification.

    The host supplies the scoped dependencies of the whole decision, including
    negative dependencies, and records retaining the evidence used. Take this
    capture under the same live fence as the decision.
    """
    with publication.transaction():
        return publication.stage(_INPUTS, _encode({
            "namespace": _namespace(ctx),
            "basis": _basis(at),
            "use": use,
            "publication_head": publication.head(),
            "scope": scope,
            "deps": [[dep.key.family, dep.key.selector.hex(), dep.revision] for dep in deps],
            "evidence": dict(evidence),
            "allowed_actions": list(allowed_actions),
            "blockers": list(blockers),
        }))


def stage_candidate(publication: Publication, at: Basis, guard: Artifact, sealed: Ref) -> Ref:
    """Bind an act guard to the existing sealed candidate without re-encoding its bytes."""
    with publication.transaction():
        return publication.stage(_CANDIDATE, _encode({
            "basis": _basis(at),
            "guard": guard.value,
            "sealed": sealed,
        }))


class _DeliveryProofs:
    def __init__(self, ctx: Context, workspace: Workspace, facts: Facts,
                 publication: Publication):
        self._namespace = _namespace(ctx)
        self._workspace = workspace
        self._facts = facts
        self._publication = publication
        self._defer = enabled()
        self._verify: LegacyVerifier | None = None
        self._basis: Basis | None = None
        self._verdicts: dict[Ref, Artifact] = {}

    def _inputs(self, at: Basis, use: Use, inputs: Ref) -> dict[str, object]:
        record = _load(self._publication, inputs, _INPUTS)
        if (record.get("namespace") != self._namespace or record.get("basis") != _basis(at)
                or record.get("use") != use):
            raise Stale("guard inputs belong to a different attempt, basis or use")
        return record

    def _fresh(self, at: Basis, record: dict[str, object], deps: tuple[Dep, ...]) -> None:
        if "publication_head" not in record:
            raise Corrupt("guard publication identity is missing")
        value = record["publication_head"]
        expected = None if value is None else _ref(value)
        if not self._facts.current(at, deps) or self._publication.head() != expected:
            raise Stale("guard dependencies changed")

    def _decision(self, at: Basis, use: Use, inputs: Ref,
                  record: dict[str, object]) -> dict[str, object]:
        evidence = record.get("evidence")
        if not isinstance(evidence, dict) or any(name not in evidence for name in _EVIDENCE):
            raise Unavailable("guard evidence is incomplete")
        for value in evidence.values():
            _ref(value)
        return {
            "inputs": inputs,
            "use": use,
            "authority": at.authority,
            "allowed_actions": _strings(record.get("allowed_actions")),
            "evidence": evidence,
            "blockers": _strings(record.get("blockers")),
        }

    def _scope(self, record: dict[str, object]) -> Ref:
        scope = _ref(record.get("scope"))
        if scope.codec != _SCOPE:
            raise Corrupt("unexpected proof scope codec")
        return scope

    def guard(self, at: Basis, use: Use, inputs: Ref) -> Artifact:
        record = self._inputs(at, use, inputs)
        deps = _dependencies(record.get("deps"))
        with self._workspace.pin(self._scope(record)) as live:
            if (live.root, live.epoch) != (at.live, at.epoch):
                raise Stale("guard workspace generation changed")
            self._fresh(at, record, deps)
            decision = self._decision(at, use, inputs, record)
            with self._publication.transaction():
                result = self._publication.stage(_GUARD, _encode(decision))
            return Artifact(result, deps)

    def delivery(self, at: Basis, candidate: Ref, guard: Artifact) -> Artifact:
        if self._verify is None:
            raise Unavailable("legacy delivery verifier is unavailable")
        decision = _load(self._publication, guard.value, _GUARD)
        inputs = _ref(decision.get("inputs"))
        record = self._inputs(at, "act", inputs)
        deps = _dependencies(record.get("deps"))
        if (guard.deps != deps or _encode(decision) !=
                _encode(self._decision(at, "act", inputs, record))):
            raise Corrupt("delivery guard does not match its inputs")
        sealed = _load(self._publication, candidate, _CANDIDATE)
        if sealed.get("basis") != _basis(at) or _ref(sealed.get("guard")) != guard.value:
            raise Stale("candidate was sealed against a different guard")
        sealed_ref = _ref(sealed.get("sealed"))
        with self._workspace.pin(self._scope(record)) as live:
            if (live.root, live.epoch) != (at.live, at.epoch):
                raise Stale("delivery workspace generation changed")
            evidence = decision["evidence"]
            assert isinstance(evidence, dict)  # validated by _decision above
            for value in evidence.values():
                self._publication.load(_ref(value))
            sealed_bytes = self._publication.load(sealed_ref)
            self._fresh(at, record, deps)
            if self._basis != at:
                self._verdicts.clear()
                self._basis = at
            cached = self._verdicts.get(candidate)
            if cached is not None:
                self._publication.load(cached.value)
                return cached
            verdict = self._verify(sealed_bytes, decision)
            if not isinstance(verdict, Mapping) or not verdict:
                raise Unavailable("legacy delivery verdict is unavailable")
            self._fresh(at, record, deps)
            with self._publication.transaction():
                result = self._publication.stage(_VERDICT, _encode(dict(verdict)))
            artifact = Artifact(result, deps)
            if self._defer:
                self._verdicts[candidate] = artifact
            return artifact


def open_port(ctx: Context, *, workspace: Workspace, facts: Facts,
              publication: Publication) -> Proofs:
    return _DeliveryProofs(ctx, workspace, facts, publication)


def bind_legacy_verifier(proofs: Proofs, verify: LegacyVerifier) -> None:
    """Attach the read-only verdict producer owned by the host's finish boundary.

    It must run the existing candidate, smoke, check and blocker gates in
    legacy order while pinned. Emit receipts and publish only after delivery
    returns and the host has validated current authority at handoff.
    """
    if not isinstance(proofs, _DeliveryProofs) or proofs._verify is not None:
        raise ValueError("proof verifier requires a new delivery proof port")
    proofs._verify = verify
