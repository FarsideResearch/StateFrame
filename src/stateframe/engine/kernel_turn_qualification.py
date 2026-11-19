"""Certify exact provider bytes and recheck authority at dispatch."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

from kernel_turn_contracts import (
    Artifact, Basis, Candidate, Context, Corrupt, Count, Live, Meter, MeterId,
    Panels, Proofs, Publication, Qualification, Qualified, Ref, Stale, Unavailable,
)

FIT_CODEC = "turn/fit-inputs/1"
CERTIFICATE_CODEC = "turn/certificate/1"
_RESIDENT_CANDIDATES = 3


def encode_record(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def stage_fit_inputs(
        publication: Publication, *, at: Basis, scope: Ref, guard_inputs: Ref,
        guard: Artifact, meter: MeterId, request_id: str, budget: dict[str, object],
) -> Ref:
    """Stage the final budget receipt after the existing host seal passes."""
    with publication.transaction():
        return publication.stage(FIT_CODEC, encode_record({
            "basis": at, "scope": scope, "guard_inputs": guard_inputs, "guard": guard.value,
            "request_id": request_id, "meter": meter, "request_budget_final_wire": budget,
        }))


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None:
        if not math.isfinite(deadline):
            raise ValueError("invalid qualification deadline")
        if time.monotonic() >= deadline:
            raise TimeoutError("qualification deadline expired")


def _load_fit(publication: Publication, ref: Ref) -> dict[str, object]:
    if ref.codec != FIT_CODEC:
        raise Corrupt("unexpected qualification fit codec")
    raw = publication.load(ref)
    if len(raw) != ref.size or sha256(raw).hexdigest() != ref.sha256:
        raise Corrupt("qualification fit identity changed")
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or encode_record(value) != raw:
            raise ValueError("noncanonical fit")
    except (ValueError, RecursionError) as error:
        raise Corrupt("invalid qualification fit") from error
    return value


def _fit_ref(fit: dict[str, object], field: str, codec: str) -> Ref:
    raw = fit.get(field)
    if (not isinstance(raw, list) or len(raw) != 3 or raw[0] != codec
            or not isinstance(raw[1], str) or len(raw[1]) != 64
            or any(char not in "0123456789abcdef" for char in raw[1])
            or type(raw[2]) is not int or raw[2] < 0):
        raise Corrupt("invalid qualification input reference")
    return Ref(*raw)


@dataclass(frozen=True, slots=True)
class _Admission:
    scope: Ref
    guard_inputs: Ref
    guard: Artifact
    qualified: Qualified
    deadline: float | None


class _Qualification:
    def __init__(self, ctx: Context, live: Live, proofs: Proofs,
                 meter: Meter, publication: Publication) -> None:
        self._context = ctx
        self._live = live
        self._proofs = proofs
        self._meter = meter
        self._publication = publication
        self._certificates: OrderedDict[tuple[Basis, str, Ref, Artifact, MeterId], _Admission] = OrderedDict()
        self._counts: OrderedDict[tuple[MeterId, str], Count] = OrderedDict()
        self._active: _Admission | None = None

    def _check_guard(self, at: Basis, guard_inputs: Ref, expected: Artifact) -> None:
        if self._proofs.guard(at, "send", guard_inputs) != expected:
            raise Stale("qualification send guard changed")

    def _count(self, candidate: Candidate, deadline: float | None) -> Count:
        from kernel_request_diet import _tokenization_content

        try:
            body = json.loads(candidate.wire)
            if not isinstance(body, dict) or body.get("model") != candidate.meter.model:
                raise ValueError("request model differs from meter")
            content = _tokenization_content(body)
            projection = sha256(content.encode()).hexdigest()
        except (ValueError, RecursionError) as error:
            raise Corrupt("invalid qualification provider wire") from error
        key = (candidate.meter, projection)
        _check_deadline(deadline)
        count = self._counts.get(key)
        if count is None:
            count = self._meter.count(candidate.meter, content, deadline)
            _check_deadline(deadline)
            if (not isinstance(count, Count) or count.meter != candidate.meter
                    or count.input_sha256 != projection
                    or type(count.tokens) is not int or count.tokens < 0):
                raise Corrupt("exact meter returned a different identity or an invalid count")
            self._counts[key] = count
        self._counts.move_to_end(key)
        if len(self._counts) > _RESIDENT_CANDIDATES:
            self._counts.popitem(last=False)
        return count

    def qualify(self, at: Basis, candidate: Candidate, deadline: float | None) -> Qualified:
        self._active = None
        _check_deadline(deadline)
        if type(candidate.wire) is not bytes:
            raise Corrupt("qualification requires immutable provider bytes")
        wire_sha = sha256(candidate.wire).hexdigest()
        identity = (at, wire_sha, candidate.inputs, candidate.guard, candidate.meter)
        resident = self._certificates.get(identity)
        if resident is not None:
            with self._live.observing(resident.scope) as live:
                if live != at:
                    raise Stale("qualification basis changed")
            _check_deadline(resident.deadline)
            _check_deadline(deadline)
            if deadline is not None and (resident.deadline is None or deadline < resident.deadline):
                resident = _Admission(resident.scope, resident.guard_inputs, resident.guard,
                                      resident.qualified, deadline)
                self._certificates[identity] = resident
            self._certificates.move_to_end(identity)
            self._active = resident
            return resident.qualified

        fit = _load_fit(self._publication, candidate.inputs)
        scope = _fit_ref(fit, "scope", "turn/scope/1")
        guard_inputs = _fit_ref(fit, "guard_inputs", "turn/guard-inputs/1")
        if (fit.get("basis") != json.loads(encode_record(at))
                or fit.get("guard") != list(candidate.guard.value)):
            raise Stale("qualification fit and send guard belong to different observations")
        if fit.get("meter") != list(candidate.meter):
            raise Stale("qualification meter changed")
        receipt = fit.get("request_budget_final_wire")
        if (not isinstance(receipt, dict) or receipt.get("status") != "READY"
                or receipt.get("final_validation_failures") != []
                or not isinstance(fit.get("request_id"), str)
                or not fit["request_id"]):
            raise Unavailable("final fit receipt is unavailable")
        # The proof provider may acquire its own workspace pin. Verify that
        # proof first, then match its complete basis under the host fence.
        self._check_guard(at, guard_inputs, candidate.guard)
        with self._live.observing(scope) as live:
            if live != at:
                raise Stale("qualification basis changed")

        if receipt.get("measure_method") != "tokenize":
            raise Unavailable("qualification requires exact tokenizer evidence")
        if (receipt.get("body_sha256") != "sha256:" + wire_sha
                or receipt.get("request_bytes") != len(candidate.wire)):
            raise Corrupt("qualification fit receipt belongs to other bytes")
        byte_limit = receipt.get("max_request_bytes")
        token_limit = receipt.get("max_prompt_tokens")
        overhead = receipt.get("template_overhead_tokens")
        measured = receipt.get("measured_prompt_tokens")
        if (type(byte_limit) is not int or len(candidate.wire) > byte_limit
                or type(token_limit) is not int or token_limit < 0
                or type(overhead) is not int or overhead < 0 or type(measured) is not int):
            raise Corrupt("invalid qualification ceilings or overhead")

        count = self._count(candidate, deadline)
        prompt_tokens = count.tokens + overhead
        if prompt_tokens > token_limit or prompt_tokens != measured:
            raise Unavailable("exact count differs from final fit proof or exceeds its ceiling")
        value = encode_record({
            "context": [self._context.principal, self._context.run, self._context.attempt,
                        self._context.profile_digest],
            "basis": at, "request_id": fit["request_id"], "route": count.meter.route,
            "wire": [wire_sha, len(candidate.wire)], "guard": candidate.guard.value,
            "guard_deps": [[dep.key.family, dep.key.selector.hex(), dep.revision]
                           for dep in candidate.guard.deps],
            "fit": candidate.inputs, "count": count, "ceilings": [byte_limit, token_limit],
        })
        with self._publication.transaction():
            ref = self._publication.stage(CERTIFICATE_CODEC, value)
        _check_deadline(deadline)
        qualified = Qualified(at, candidate.wire, Artifact(ref, candidate.guard.deps), count)
        admission = _Admission(scope, guard_inputs, candidate.guard, qualified, deadline)
        self._certificates[identity] = admission
        self._active = admission
        if len(self._certificates) > _RESIDENT_CANDIDATES:
            self._certificates.popitem(last=False)
        return qualified

    def handoff(self, q: Qualified, submit: Callable[[bytes], None]) -> None:
        admission = self._active
        if admission is None or admission.qualified is not q:
            raise Stale("qualification is no longer the selected candidate")
        _check_deadline(admission.deadline)
        self._check_guard(q.basis, admission.guard_inputs, admission.guard)
        with self._live.observing(admission.scope) as live:
            if live != q.basis:
                raise Stale("qualification basis changed before dispatch")
            _check_deadline(admission.deadline)
            if self._active is not admission:
                raise Stale("qualification selection changed before dispatch")
            submit(q.wire)


def open_port(ctx: Context, *, live: Live, panels: Panels, proofs: Proofs,
              meter: Meter, publication: Publication) -> Qualification:
    """Open the run-local candidate cache. Panel assembly precedes qualification."""
    return _Qualification(ctx, live, proofs, meter, publication)
