"""Floor-first admission over the pure complete-candidate byte oracle.

The core's arithmetic is additive; the provider body's dependent sections are
not. Each core decision therefore prices complete minimum completions relative
to their smallest exact rendering. This includes escaping, shared carriers and
count/overlay growth without pretending that diagnostic marginals add up.
No live runtime, tokenizer service or journal is accessed by this adapter.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import math

if __package__:
    from . import kernel_working_set_allocator as core
    from .kernel_working_set_proof import CertificateError, verify_certificate
    from .kernel_working_set_recovery import RecoveryRouteUnavailable
    from .kernel_working_set_render import RenderedCandidate, propose_slots, render_assignment
    from .kernel_working_set_snapshot import PreparationSnapshot
    from .kernel_working_set_stop import MandatoryEnvelopeProof, ReasonCode
else:
    import kernel_working_set_allocator as core
    from kernel_working_set_proof import CertificateError, verify_certificate
    from kernel_working_set_recovery import RecoveryRouteUnavailable
    from kernel_working_set_render import RenderedCandidate, propose_slots, render_assignment
    from kernel_working_set_snapshot import PreparationSnapshot
    from kernel_working_set_stop import MandatoryEnvelopeProof, ReasonCode

MAX_MEASUREMENTS = 8


@dataclass(frozen=True)
class Measurement:
    prompt_tokens: int
    frame_tokens: int
    method: str
    tokenizer_identity: str = ""
    contract_valid: bool = True
    protected_capacity_valid: bool = True

    def __post_init__(self) -> None:
        if (type(self.prompt_tokens) is not int or self.prompt_tokens < 0
                or type(self.frame_tokens) is not int or self.frame_tokens < 0
                or not isinstance(self.method, str) or not self.method
                or not isinstance(self.tokenizer_identity, str)
                or type(self.contract_valid) is not bool
                or type(self.protected_capacity_valid) is not bool):
            raise ValueError("invalid complete-candidate token measurement")


@dataclass(frozen=True)
class Trial:
    assignment_digest: str
    body_sha256: str
    total_bytes: int
    measurement: Measurement
    violated_limits: tuple[str, ...]


@dataclass(frozen=True)
class PreparedCandidate:
    rendered: RenderedCandidate
    measurement: Measurement
    floor_bytes: int
    cap: int
    trials: tuple[Trial, ...]
    fits_without_degrade: bool

    @property
    def total_bytes(self) -> int:
        return len(self.rendered.wire)

    @property
    def wire(self) -> bytes:
        return self.rendered.wire

    @property
    def assignment_digest(self) -> str:
        return self.rendered.certificate.assignment_digest


@dataclass(frozen=True)
class Blocked:
    """Shadow diagnostic; only FLOOR_INFEASIBLE has a mandatory-byte proof.

    Exhaustion is bounded-search failure, never proof that all requests are
    impossible. This value does not suspend a runtime or change its live state.
    """
    reason: ReasonCode
    floor_bytes: int
    total_bytes: int
    cap: int
    section_breakdown: tuple[tuple[str, int], ...]
    trials: tuple[Trial, ...] = ()
    proof: MandatoryEnvelopeProof | None = None
    detail: str = ""
    assignment_digest: str = ""


Measure = Callable[[bytes], Measurement | tuple[int, int, str]]


def _best_render(snapshot: PreparationSnapshot, modes: Mapping[str, core.Mode]) -> RenderedCandidate:
    candidates = []
    errors = []
    for layout in snapshot.layouts:
        try:
            candidates.append(render_assignment(snapshot, modes, layout))
        except RecoveryRouteUnavailable:
            raise
        except ValueError as error:
            errors.append(error)
    if not candidates:
        raise ValueError("no applicable complete-candidate layout") from (errors[0] if errors else None)
    return min(candidates, key=lambda candidate: len(candidate.wire))


def _minimum_completion(snapshot: PreparationSnapshot, slots: tuple[core.Slot, ...]
                        ) -> tuple[dict[str, core.Mode], RenderedCandidate]:
    """The deterministic minimum policy completion, before fidelity decisions.

    Start from measured marginal minima. Re-price the whole body, including
    simultaneous equivalent carriers: two FULL witnesses may share bytes even
    when neither independent marginal would choose FULL. A coordinate descent
    then corrects count/index-dependent minima. This is the declared policy
    floor, not an exhaustive global optimization over every Cartesian choice.
    """
    modes = {slot.key: min(slot.allowed, key=lambda entry: (entry[1], -entry[0].rank))[0]
             for slot in slots}
    candidate = _best_render(snapshot, modes)
    # Removing the last omission can remove a shared notice/summary shell.
    # That simultaneous improvement is invisible to independent marginals.
    rich = {slot.key: slot.richest_first[0][0] for slot in slots}
    rich_candidate = _best_render(snapshot, rich)
    if len(rich_candidate.wire) < len(candidate.wire):
        modes, candidate = rich, rich_candidate
    groups: dict[tuple[object, ...], list[str]] = {}
    by_key = {slot.key: slot for slot in slots}
    for source in snapshot.sources:
        groups.setdefault((source.identity, source.page), []).append(source.key)
    decisions = [tuple(keys) for keys in groups.values() if len(keys) > 1]
    path_groups: dict[str, list[str]] = {}
    for source in snapshot.sources:
        path_groups.setdefault(source.identity.path, []).append(source.key)
    decisions.extend(tuple(keys) for keys in path_groups.values() if len(keys) > 1)
    decisions.extend((slot.key,) for slot in slots)
    # Strictly decreasing wire lengths prevent cycles. The finite pass bound
    # also bounds work for hostile snapshots with unusually coupled sections.
    for _pass in range(len(slots) + 1):
        changed = False
        for keys in decisions:
            common = set(mode for mode, _ in by_key[keys[0]].allowed)
            for key in keys[1:]:
                common.intersection_update(mode for mode, _ in by_key[key].allowed)
            for mode in sorted(common, key=lambda value: value.rank, reverse=True):
                trial_modes = {**modes, **dict.fromkeys(keys, mode)}
                trial = _best_render(snapshot, trial_modes)
                if len(trial.wire) < len(candidate.wire):
                    modes, candidate, changed = trial_modes, trial, True
        if not changed:
            break
    return modes, candidate


def _floor_check(slots: tuple[core.Slot, ...], minimum: Mapping[str, core.Mode],
                 floor: RenderedCandidate, cap: int) -> core.Assignment | core.Blocked:
    # Every minimum payload and its dependent bytes are already in the frozen
    # floor wire. Singleton domains carry zero additional bytes, avoiding any
    # double counting of shared carriers or aggregate sections.
    floor_slots = tuple(replace(slot, domain=((minimum[slot.key], 0),)) for slot in slots)
    return core.solve(floor_slots, fixed_bytes=len(floor.wire), cap=cap)


def _mandatory_lower_bound(snapshot: PreparationSnapshot, slots: tuple[core.Slot, ...]) -> bytes:
    """Serialize only fragments every legal layout must physically preserve.

    The coupled policy minimum is not a Cartesian impossibility proof. Omit
    mutable frames, tails and all optional carriers from this independent bound.
    Stable system/task rows, the native exchange and canonical F-only source
    rows survive both supported layouts byte for byte after normalization.
    """
    if __package__:
        from .kernel_working_set_render import _normalize
    else:
        from kernel_working_set_render import _normalize
    body = json.loads(snapshot.fixed_body)
    messages = [row for row in body["messages"] if row.get("role") in {
        "system", "developer", "assistant", "tool"}
        or (row.get("role") == "user" and str(row.get("content", "")).startswith("Task:"))]
    mandatory = {slot.key for slot in slots if tuple(mode for mode, _ in slot.allowed) == (core.FULL,)}
    pages = {row.get("content") for row in messages if row.get("role") == "tool"
             and isinstance(row.get("content"), str)}
    for source in sorted(snapshot.sources, key=lambda s: (s.admission_ordinal, s.key)):
        if source.key not in mandatory or source.page in pages:
            continue
        if source.page.startswith(("[read page]", "[Current source:", "[Working source:")):
            messages.append({"role": "user", "content": source.page})
            pages.add(source.page)
    body["messages"] = messages
    return json.dumps(_normalize(body, snapshot)).encode()


def _choose_slot(snapshot: PreparationSnapshot, slot: core.Slot,
                 completion: Mapping[str, core.Mode], cap: int,
                 *, carried_prefix: bool) -> tuple[core.Mode, RenderedCandidate] | None:
    candidates = {mode: _best_render(snapshot, {**completion, slot.key: mode})
                  for mode, _cost in slot.allowed}
    minimum_bytes = min(len(candidate.wire) for candidate in candidates.values())
    priced = replace(slot,
        domain=tuple((mode, len(candidate.wire) - minimum_bytes)
                     for mode, candidate in candidates.items()),
        priority_tier=(slot.priority_tier if carried_prefix else core.PriorityTier.OPTIONAL),
        previous_mode=(slot.previous_mode if carried_prefix else None))
    selected = core.solve((priced,), fixed_bytes=minimum_bytes, cap=cap)
    if isinstance(selected, core.Blocked):
        return None
    mode = selected.modes[slot.key]
    return mode, candidates[mode]


def _solve_bytes(snapshot: PreparationSnapshot, slots: tuple[core.Slot, ...],
                 minimum: Mapping[str, core.Mode], floor: RenderedCandidate,
                 cap: int) -> RenderedCandidate | None:
    if isinstance(_floor_check(slots, minimum, floor, cap), core.Blocked):
        return None
    modes = dict(minimum)
    candidate = floor
    fresh, protected, carried, optional = core.priority_groups(slots)
    for slot in (*fresh, *protected):
        choice = _choose_slot(snapshot, slot, modes, cap, carried_prefix=True)
        if choice is None:
            return None
        modes[slot.key], candidate = choice
    tail = list(optional)
    for index, slot in enumerate(carried):
        choice = _choose_slot(snapshot, slot, modes, cap, carried_prefix=True)
        if choice is None:
            return None
        if choice[0] != slot.previous_mode:
            tail.extend(carried[index:])
            break
        modes[slot.key], candidate = choice
    for slot in sorted(tail, key=core.optional_order_key):
        choice = _choose_slot(snapshot, slot, modes, cap, carried_prefix=False)
        if choice is None:
            return None
        modes[slot.key], candidate = choice
    return candidate


def _offline_measure(snapshot: PreparationSnapshot, wire: bytes) -> Measurement:
    from kernel_context import token_estimate

    ratio = snapshot.bytes_per_token
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("invalid frozen calibration ratio")
    body = json.loads(wire)
    # Preserve the legacy unconfigured frame counter's interpretation; this
    # estimate must not masquerade as a tokenizer's prompt measurement.
    return Measurement(math.ceil(len(wire) / ratio), token_estimate(body),
                       "calibrated_bytes", snapshot.tokenizer_identity)


def _violations(snapshot: PreparationSnapshot, candidate: RenderedCandidate,
                measured: Measurement) -> tuple[str, ...]:
    failed = []
    if len(candidate.wire) > snapshot.byte_cap:
        failed.append("serialized_bytes")
    if measured.method == "verified_byte_bound":
        failed.append("token_count_unavailable")
    if measured.prompt_tokens > snapshot.prompt_limit:
        failed.append("prompt_tokens")
    if snapshot.frame_limit is not None and measured.frame_tokens > snapshot.frame_limit:
        failed.append("frame_tokens")
    body = json.loads(candidate.wire)
    output_tokens = body.get("max_completion_tokens", body.get("max_tokens", 0))
    if type(output_tokens) is not int or output_tokens < 0:
        failed.append("output_allocation")
    elif (snapshot.context_limit is not None
          and measured.prompt_tokens + output_tokens + snapshot.output_reserve
              + snapshot.safety_reserve > snapshot.context_limit):
        failed.append("context_tokens")
    if not measured.protected_capacity_valid:
        failed.append("protected_capacity")
    return tuple(failed)


def prepare_candidate(snapshot: PreparationSnapshot, *, measure: Measure | None = None
                      ) -> PreparedCandidate | Blocked:
    """Prepare and prove a shadow candidate, with at most eight measurements.

    The last available measurement is reserved for the smallest policy body,
    even if the heuristic search cap has already excluded it. Tokenizer identity
    may change once; rebuilding shares this same budget and never dispatches.
    """
    trials: list[Trial] = []
    floor: RenderedCandidate | None = None

    def blocked(reason: ReasonCode, candidate: RenderedCandidate | None = None,
                *, detail: str = "", proof: MandatoryEnvelopeProof | None = None) -> Blocked:
        value = candidate or floor
        return Blocked(reason, len(floor.wire) if floor is not None else 0,
            len(value.wire) if value is not None else 0, snapshot.byte_cap,
            value.section_breakdown if value is not None else (), tuple(trials), proof, detail,
            value.certificate.assignment_digest if value is not None else "")

    try:
        slots = tuple(propose_slots(snapshot))
        minimum, floor = _minimum_completion(snapshot, slots)
        verify_certificate(floor.wire, floor.certificate)
        result = _floor_check(slots, minimum, floor, snapshot.byte_cap)
        if isinstance(result, core.Blocked):
            # Only a unique-domain floor or an independent immutable lower
            # bound can certify impossibility. Shared metadata can hide an
            # unsearched subset even after coordinate/group minimization.
            mandatory = (floor.wire if all(len(slot.allowed) == 1 for slot in slots)
                         else _mandatory_lower_bound(snapshot, slots))
            if len(mandatory) <= snapshot.byte_cap:
                return blocked(ReasonCode.BYTE_POLICY_DOMAIN_EXHAUSTED,
                               detail="policy minimum exceeds cap without an impossibility proof")
            proof = MandatoryEnvelopeProof(mandatory, snapshot.byte_cap,
                                           (("fixed", len(mandatory)),))
            proof.validate()
            return blocked(ReasonCode.FLOOR_INFEASIBLE, proof=proof)
        byte_cap = snapshot.byte_cap
        rejected: set[str] = set()
        floor_measured = False
        identity = snapshot.tokenizer_identity
        identity_restarts = 0
        rich = {slot.key: slot.richest_first[0][0] for slot in slots}
        rich_candidate = _best_render(snapshot, rich)

        for attempt in range(MAX_MEASUREMENTS):
            candidate = (floor if attempt == MAX_MEASUREMENTS - 1 else
                         _solve_bytes(snapshot, slots, minimum, floor, byte_cap))
            digest = hashlib.sha256(candidate.wire).hexdigest() if candidate is not None else ""
            if candidate is None or digest in rejected:
                if floor_measured:
                    break
                candidate = floor
                digest = hashlib.sha256(candidate.wire).hexdigest()
            verify_certificate(candidate.wire, candidate.certificate)
            raw = measure(candidate.wire) if measure is not None else _offline_measure(snapshot, candidate.wire)
            measured = raw if isinstance(raw, Measurement) else Measurement(*raw)
            failed = _violations(snapshot, candidate, measured)
            trials.append(Trial(candidate.certificate.assignment_digest, digest,
                                len(candidate.wire), measured, failed))
            if not measured.contract_valid:
                return blocked(ReasonCode.PREPARATION_CONTRACT_FAILURE, candidate,
                               detail="measurement/source contract failed")
            if measured.tokenizer_identity and identity and measured.tokenizer_identity != identity:
                identity_restarts += 1
                if identity_restarts > 1:
                    return blocked(ReasonCode.PREPARATION_CONTRACT_FAILURE, candidate,
                                   detail="tokenizer authority changed twice")
                identity = measured.tokenizer_identity
                snapshot = replace(snapshot, tokenizer_identity=identity)
                # Measurement authority is part of the frozen transaction.
                # Rebuild certificates even though the provider bytes do not
                # change when only the tokenizer identity changes.
                minimum, floor = _minimum_completion(snapshot, slots)
                verify_certificate(floor.wire, floor.certificate)
                rich_candidate = _best_render(snapshot, rich)
                rejected.clear()
                floor_measured = False
                byte_cap = snapshot.byte_cap
                continue
            identity = measured.tokenizer_identity or identity
            if identity and not snapshot.tokenizer_identity:
                snapshot = replace(snapshot, tokenizer_identity=identity)
                by_name = {mode.name: mode for slot in slots for mode, _cost in slot.allowed}
                selected = {key: by_name[name] for key, name in candidate.certificate.assignment}
                rebound = render_assignment(snapshot, selected, candidate.certificate.layout)
                if rebound.wire != candidate.wire:
                    return blocked(ReasonCode.PREPARATION_CONTRACT_FAILURE, candidate,
                                   detail="tokenizer binding changed provider bytes")
                candidate = rebound
                verify_certificate(candidate.wire, candidate.certificate)
                trials[-1] = replace(trials[-1], assignment_digest=candidate.certificate.assignment_digest)
                minimum, floor = _minimum_completion(snapshot, slots)
                rich_candidate = _best_render(snapshot, rich)
            floor_measured |= candidate.wire == floor.wire
            if not failed:
                return PreparedCandidate(candidate, measured, len(floor.wire), snapshot.byte_cap,
                                         tuple(trials), candidate.wire == rich_candidate.wire)
            rejected.add(digest)
            ratios = [1.0]
            if measured.prompt_tokens > snapshot.prompt_limit:
                ratios.append(snapshot.prompt_limit / measured.prompt_tokens)
            if snapshot.frame_limit is not None and measured.frame_tokens > snapshot.frame_limit:
                ratios.append(snapshot.frame_limit / measured.frame_tokens)
            if snapshot.context_limit is not None and measured.prompt_tokens:
                body = json.loads(candidate.wire)
                output_tokens = body.get("max_completion_tokens", body.get("max_tokens", 0))
                if type(output_tokens) is int:
                    available = max(0, snapshot.context_limit - output_tokens
                                    - snapshot.output_reserve - snapshot.safety_reserve)
                    if measured.prompt_tokens > available:
                        ratios.append(available / measured.prompt_tokens)
            byte_cap = min(byte_cap - 1, len(candidate.wire) - 1,
                           math.floor(len(candidate.wire) * min(ratios)))
        return blocked(ReasonCode.TOKEN_ADMISSION_SEARCH_EXHAUSTED,
                       detail="no candidate admitted within the shared measurement budget")
    except RecoveryRouteUnavailable:
        return blocked(ReasonCode.RECOVERY_ROUTE_UNAVAILABLE)
    except (CertificateError, ValueError, TypeError, KeyError) as error:
        # Diagnostics contain only an error type; a malformed provider body or
        # capture can contain secrets and must never be copied into the journal.
        return blocked(ReasonCode.PREPARATION_CONTRACT_FAILURE, detail=type(error).__name__)


allocate = prepare_candidate
