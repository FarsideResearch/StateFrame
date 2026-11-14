"""Pure complete-candidate oracle for the default-off working-set allocator.

Every evaluation starts from frozen JSON and captured text. No discovery, LRU
touch, journal publication or workspace read is permitted here. Marginal costs
are proposals; only the complete normalized transport wire is an admission cost.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    if __package__:
        from .kernel_working_set_snapshot import PreparationSnapshot, SourceSlot
    else:
        from kernel_working_set_snapshot import PreparationSnapshot, SourceSlot

if __package__:
    from . import kernel_working_set_allocator as core
    from .kernel_working_set_cache import frozen, source_frozen, trial_render, wire_bytes
    from .kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers
else:
    import kernel_working_set_allocator as core
    from kernel_working_set_cache import frozen, source_frozen, trial_render, wire_bytes
    from kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CarrierWitness:
    slot_id: str
    obligation_ids: tuple[str, ...]
    receipt_identity: ReceiptIdentity
    mode: str
    message_index: int
    content_start: int
    content_end: int
    carrier_sha256: str
    payload_start: int | None
    payload_end: int | None
    recovery_arguments: bytes


@dataclass(frozen=True)
class DeferredPlanWitness:
    slot_ids: tuple[str, ...]
    message_index: int
    content_start: int
    content_end: int
    carrier_sha256: str


def deferred_plan_slot(slot: SourceSlot) -> bool:
    """Only never-delivered plan proposals may share a path-level omission."""
    return (slot.plan_only and slot.state in {"PLAN", "CLOSURE"}
            and not slot.capture_ids and slot.previous_mode is None
            and not slot.recovery_pin)


@dataclass(frozen=True)
class AssignmentCertificate:
    # snapshot is a trusted host object, not deserialized from model prose.
    snapshot: PreparationSnapshot
    assignment: tuple[tuple[str, str], ...]
    layout: str
    snapshot_digest: str
    assignment_digest: str
    body_sha256: str
    witnesses: tuple[CarrierWitness, ...]
    deferred_plan_witness: DeferredPlanWitness | None = None


@dataclass(frozen=True)
class RenderedCandidate:
    wire: bytes
    certificate: AssignmentCertificate
    section_breakdown: tuple[tuple[str, int], ...]


@dataclass
class _WirePreview:
    """Exact trial bytes; construct witnesses only when a candidate is used."""
    wire: bytes
    snapshot: PreparationSnapshot
    assignment: tuple[tuple[str, core.Mode], ...]
    candidate: RenderedCandidate | None = None

    def materialize(self) -> RenderedCandidate:
        if self.candidate is None:
            candidate = render_assignment(self.snapshot, dict(self.assignment), "plain")
            if candidate.wire != self.wire:
                raise ValueError("trial bytes differ from complete candidate")
            self.candidate = candidate
        return self.candidate

    @property
    def certificate(self):
        return self.materialize().certificate

    @property
    def section_breakdown(self):
        return self.materialize().section_breakdown


@dataclass
class _ProjectedPreview:
    """A projected-layout trial carried over from a sibling snapshot.

    The bytes were fully rendered and carrier-reconciled against a snapshot
    that differs only in tokenizer identity, so they are exact for this one.
    Witnesses and the certificate are rendered against this snapshot on use.
    """
    wire: bytes
    snapshot: PreparationSnapshot
    assignment: tuple[tuple[str, core.Mode], ...]
    layout: str
    candidate: RenderedCandidate | None = None

    def materialize(self) -> RenderedCandidate:
        if self.candidate is None:
            candidate = render_assignment(self.snapshot, dict(self.assignment), self.layout)
            if candidate.wire != self.wire:
                raise ValueError("trial bytes differ from complete candidate")
            self.candidate = candidate
        return self.candidate

    @property
    def certificate(self):
        return self.materialize().certificate

    @property
    def section_breakdown(self):
        return self.materialize().section_breakdown


def materialize(candidate):
    return (candidate.materialize() if isinstance(candidate, (_WirePreview, _ProjectedPreview))
            else candidate)


def _rebind_preview(result, snapshot, modes, layout):
    """Carry a cached trial to a sibling snapshot of the same byte family."""
    bound = (result.snapshot if isinstance(result, (_WirePreview, _ProjectedPreview))
             else result.certificate.snapshot)
    if bound is snapshot:
        return result
    assignment = tuple(modes.items())
    if layout == "plain":
        return _WirePreview(result.wire, snapshot, assignment)
    return _ProjectedPreview(result.wire, snapshot, assignment, layout)


@source_frozen
def _slot_modes(snapshot: PreparationSnapshot, slot: SourceSlot) -> tuple[core.Mode, ...]:
    if slot.state == "DUE" or slot.recovery_pin:
        return (core.FULL,)
    if (slot.state in {"RETIRING", "DISCHARGED", "SUPERSEDED", "DEGRADED"}
            or slot.previous_mode in {"D", "O"}):
        return (core.DEFERRED,)
    if slot.identity.path in snapshot.protected_paths:
        return (core.FULL,)
    if __package__:
        from .kernel_working_set_recovery import RecoveryRouteUnavailable, validate_recovery
    else:
        from kernel_working_set_recovery import RecoveryRouteUnavailable, validate_recovery
    try:
        validate_recovery(snapshot, slot)
    except RecoveryRouteUnavailable:
        # An omission without a callable recovery route is outside the legal
        # domain. Its available FULL carrier may still fit the complete wire.
        return (core.FULL,)
    return (core.FULL, core.DEFERRED)


# The adapter uses this same domain authority; it must never infer modes from
# the cost of a notice or from model-authored source claims.
domain = _slot_modes


def _tier(snapshot: PreparationSnapshot, slot: SourceSlot) -> core.PriorityTier:
    if slot.state == "DUE":
        return core.PriorityTier.FRESH
    if slot.recovery_pin or slot.identity.path in snapshot.protected_paths:
        return core.PriorityTier.PROTECTED
    if slot.previous_mode is not None:
        return core.PriorityTier.CARRIED
    return core.PriorityTier.OPTIONAL


@frozen
def _assignment_domains(snapshot: PreparationSnapshot):
    keys = frozenset(s.key for s in snapshot.slots)
    if len(keys) != len(snapshot.slots):
        raise ValueError("duplicate frozen slot key")
    return keys, tuple((slot.key, _slot_modes(snapshot, slot)) for slot in snapshot.slots)


def _assignment(snapshot: PreparationSnapshot,
                assignment: Mapping[str, core.Mode] | core.Assignment) -> dict[str, core.Mode]:
    modes = assignment.modes if isinstance(assignment, core.Assignment) else assignment
    keys, domains = _assignment_domains(snapshot)
    if not isinstance(modes, Mapping) or set(modes) != keys:
        raise ValueError("assignment must select every frozen slot exactly once")
    result = {}
    for key, allowed in domains:
        mode = modes[key]
        if mode not in allowed:
            raise ValueError("assignment violates frozen slot domain")
        result[key] = mode
    return result


@source_frozen
def _recovery_arguments(snapshot: PreparationSnapshot, slot: SourceSlot) -> bytes:
    if __package__:
        from .kernel_working_set_recovery import validate_recovery
    else:
        from kernel_working_set_recovery import validate_recovery
    return json.dumps(validate_recovery(snapshot, slot),
                      sort_keys=True, separators=(",", ":")).encode()


@source_frozen
def _notice(snapshot: PreparationSnapshot, slot: SourceSlot) -> str:
    identity = slot.identity
    arguments = json.loads(_recovery_arguments(snapshot, slot))
    # Preserve positional/cursor recovery arguments, including lossy columns.
    call = "read(" + ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                               for key, value in sorted(arguments.items())) + ")"
    metadata = []
    if arguments.get("expected_revision") != identity.revision:
        metadata.append(f"revision={identity.revision}")
    if (arguments.get("start_line"), arguments.get("end_line")) != (
            identity.first_line, identity.last_line):
        metadata.append(f"lines={identity.first_line}-{identity.last_line}")
    metadata.append(f"digest={identity.payload_sha256}")
    return ("[Source page deferred]\n" + " ".join(metadata) + "\n"
            f"Retrieve current bytes with: {call}.")


@frozen
def _payload_interval(slot: SourceSlot) -> tuple[int, int]:
    payload = slot.payload.encode("utf-8")
    page = slot.page.encode("utf-8")
    # A capture page owns its payload suffix, never a matching fragment in a
    # path/header. This also handles blank and repeated source lines exactly.
    if slot.payload_start is None:
        if not page.endswith(payload):
            raise ValueError("capture page does not end in its exact payload")
        start = len(page) - len(payload)
    else:
        start = slot.payload_start
        if type(start) is not int or start < 0 or page[start:start + len(payload)] != payload:
            raise ValueError("frozen payload coordinates differ from captured page")
    if digest(payload) != slot.identity.payload_sha256:
        raise ValueError("captured payload hash differs from receipt")
    if line_digests_with_numbers(slot.payload, slot.identity.first_line) != (
            slot.line_digests, slot.line_numbers):
        raise ValueError("captured line evidence differs from receipt")
    return start, start + len(payload)


def _fixed_tool_carrier(messages, page: str) -> int | None:
    """Reuse exact read bytes even when the host appended an advisory line."""
    for index, message in enumerate(messages):
        content = message.get("content")
        if (message.get("role") == "tool" and isinstance(content, str)
                and (content == page or content.startswith(page + "\n"))):
            return index
    return None


@frozen
def snapshot_digest(snapshot: PreparationSnapshot) -> str:
    # Dataclasses are recursively frozen; represent all authority, including
    # presentation/carrier coordinates, rather than just the fixed envelope.
    from dataclasses import asdict
    def encode(value):
        if isinstance(value, bytes):
            return {"bytes": value.hex()}
        raise TypeError("unsupported preparation authority")
    return digest(json.dumps(asdict(snapshot), default=encode,
                             sort_keys=True, separators=(",", ":")).encode())


def _insertion_index(snapshot, messages) -> int:
    explicit = snapshot.insertion_index
    if explicit is not None:
        if type(explicit) is not int or not 0 <= explicit <= len(messages):
            raise ValueError("invalid frozen source insertion point")
        return explicit
    # Native exchanges must remain contiguous; the newest opaque user suffix
    # is kept immediately ahead of the whole newest exchange.
    assistant = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get("role") == "assistant"), None)
    end = len(messages) if assistant is None else assistant
    return next((i for i in range(end - 1, -1, -1)
                 if messages[i].get("role") == "user"
                 and not str(messages[i].get("content", "")).startswith("Task:")), end)


def _normalize(body, snapshot):
    if __package__:
        from .kernel_launch_identity_cost import enforce_provider_body, ReasoningProfile
        from .kernel_measurement import _without_private_message_keys
    else:
        from kernel_launch_identity_cost import enforce_provider_body, ReasoningProfile
        from kernel_measurement import _without_private_message_keys
    body = _without_private_message_keys(body)
    # The llama.cpp reasoning-on contract is exactly a deepcopy. Each render
    # owns its body and message dictionaries; nested fixed values are opaque
    # and read-only. Other profiles still run their full provider transform.
    result = (body if snapshot.dialect == "llama.cpp" and snapshot.reasoning_mode == "on"
              else enforce_provider_body(body,
                  profile=ReasoningProfile(snapshot.reasoning_mode, "snapshot"),
                  dialect=snapshot.dialect, environ={}))
    # Output is normalized before costing. Never mutate it after certification.
    output_key = "max_completion_tokens" if "max_completion_tokens" in result else "max_tokens"
    output = result.get(output_key)
    if type(output) is int and snapshot.context_limit is not None:
        available = (snapshot.context_limit - snapshot.prompt_limit
                     - snapshot.output_reserve - snapshot.safety_reserve)
        if available < 1:
            raise ValueError("frozen output/context reserve is infeasible")
        result[output_key] = min(output, available)
    return result


@frozen
def _fixed_body(snapshot: PreparationSnapshot):
    return json.loads(snapshot.fixed_body)


def _render_assignment(snapshot: PreparationSnapshot,
                       assignment: Mapping[str, core.Mode] | core.Assignment,
                       layout: str = "plain", *, wire_only: bool = False):
    """Render selected payloads, their dependent truth and provider bytes once.

    Slot order is admission order, independent of solver decision order. FULL
    carriers can share only exact identity+page material or a frozen native tool
    result whose exact page prefix equals the captured page. No path-only lookup.
    """
    modes = _assignment(snapshot, assignment)
    if layout not in snapshot.layouts or layout not in {"plain", "final_cache"}:
        raise ValueError("layout not admitted by snapshot")
    body = dict(_fixed_body(snapshot))
    messages = body.get("messages")
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError("frozen provider body requires messages")
    # Truth projection replaces content in these rows. Nested native call and
    # content-part objects remain opaque; provider/layout transforms copy them.
    messages = [dict(message) for message in messages]
    insertion = _insertion_index(snapshot, messages)
    generated = []
    locations = {}
    shared = {}
    categories = {}
    ordered = sorted(snapshot.slots, key=lambda s: (s.admission_ordinal, s.key))
    deferred_plan = []
    for slot in ordered:
        mode = modes[slot.key]
        if mode == core.FULL:
            payload_start, payload_end = _payload_interval(slot)
            key = (slot.identity, slot.page, slot.standalone_page, slot.standalone_payload_start)
            if key in shared:
                locations[slot.key] = shared[key]
                continue
            # Match the complete captured page at offset zero, never a path or
            # arbitrary substring. Advisory suffixes stay inside the witness's
            # whole-carrier hash; payload coordinates remain byte-exact.
            fixed = _fixed_tool_carrier(messages, slot.page)
            if fixed is not None:
                location = ("fixed", fixed, payload_start, payload_end)
            else:
                page = slot.page
                if slot.standalone_page is not None:
                    # Reuse the closure's existing user-role grammar without
                    # inventing an unpaired native tool result. The certificate
                    # still authenticates the original capture's exact payload.
                    page = slot.standalone_page
                    payload_start = slot.standalone_payload_start
                    payload_end = payload_start + len(slot.payload.encode())
                location = ("generated", len(generated), payload_start, payload_end)
                generated.append({"role": "user", "content": page})
                categories[len(generated) - 1] = "full_source"
            locations[slot.key] = shared[key] = location
        else:
            if not snapshot.recovery_allowed:
                raise ValueError("frozen tool/owner surface refuses source recovery")
            if deferred_plan_slot(slot):
                deferred_plan.append(slot.key)
                continue
            locations[slot.key] = ("generated", len(generated), None, None)
            generated.append({"role": "user", "content": _notice(snapshot, slot)})
            categories[len(generated) - 1] = "deferred_notices"

    full = [s for s in ordered if modes[s.key] == core.FULL]
    revisions = dict(snapshot.current_revisions)
    current = [s for s in full if revisions.get(s.identity.path) == s.identity.revision]
    if current:
        # Range-scoped facts: a second disjoint or historical capture of this
        # path receives no authority from the first one's FULL witness.
        facts = "[Host fact]\nCurrent source bytes attached to this request: " + "; ".join(
            f"{s.identity.path} lines {s.identity.first_line}-{s.identity.last_line} "
            f"revision {s.identity.revision}" for s in current)
        categories[len(generated)] = "host_facts"
        generated.append({"role": "user", "content": facts})
    deferred_paths = set(snapshot.deferred_paths)
    deferred_paths.update(s.identity.path for s in ordered if modes[s.key] != core.FULL)
    if deferred_paths:
        if __package__:
            from .kernel_request_source_window import _render_truthful_deferred_source_notice
        else:
            from kernel_request_source_window import _render_truthful_deferred_source_notice
        deferred_notice_index = len(generated)
        categories[len(generated)] = "deferred_notices"
        generated.append({"role": "user", "content": _render_truthful_deferred_source_notice(
            sorted(deferred_paths), delivered_source_ranges=tuple(
                (s.identity.path, s.identity.revision, s.identity.first_line, s.identity.last_line)
                for s in current))})
    # Counts are derived after deduplication and escape with the whole body.
    carriers = {(locations[s.key][0], locations[s.key][1]) for s in full}
    counted = set()
    chars = 0
    for slot in full:
        carrier = locations[slot.key][:2]
        if carrier not in counted:
            chars += len(slot.payload)
            counted.add(carrier)
    categories[len(generated)] = "coverage"
    generated.append({"role": "user", "content":
                      f"kept {len(carriers)} selected source windows; about {chars} chars of source fit"})
    body["messages"] = messages[:insertion] + generated + messages[insertion:]
    if __package__:
        from .kernel_working_set_facts import rebuild_delivery_truth
    else:
        from kernel_working_set_facts import rebuild_delivery_truth
    # Carry fixed-owner coordinates across insertion. Generated carriers and
    # opaque fixed messages never enter the truth parser, even by marker.
    truth_indexes = tuple(i + len(generated) if i >= insertion else i
                          for i in snapshot.truth_message_indexes)
    rebuild_delivery_truth(body, snapshot, full, message_indexes=truth_indexes)
    plain = _normalize(body, snapshot)
    if layout == "final_cache":
        if __package__:
            from .kernel_cache_layout_candidate import _checked_final_cache_projection
        else:
            from kernel_cache_layout_candidate import _checked_final_cache_projection
        body = _checked_final_cache_projection(plain)
    else:
        body = plain
    wire = wire_bytes(body, snapshot, layout)
    if wire_only:
        # Plain normalization preserves whole source rows and their indexes.
        # Validate snapshot serialization now, including malformed authority;
        # no trial may hide a contract error until after policy selection.
        snapshot_digest(snapshot)
        return wire
    witnesses = []
    sections = {"full_source": 0, "deferred_notices": 0, "host_facts": 0, "coverage": 0}
    physical = body["messages"]
    plain_messages = plain["messages"]
    # Cache projection preserves source/notice carriers as whole rows; derive
    # physical indexes by whole-row identity, never by source path or interval.
    physical_indexes = {}
    if layout == "plain":
        physical_indexes = dict(enumerate(range(len(plain_messages))))
    else:
        used = set()
        for index, message in enumerate(plain_messages):
            match = next((i for i, m in enumerate(physical) if i not in used and m == message), None)
            if match is not None:
                physical_indexes[index] = match
                used.add(match)
    classified = set()
    for generated_index, category in categories.items():
        index = physical_indexes.get(insertion + generated_index)
        if index is not None:
            sections[category] += len(json.dumps(physical[index]).encode())
            classified.add(index)
    for slot in ordered:
        if slot.key in deferred_plan:
            continue
        owner, index, payload_start, payload_end = locations[slot.key]
        semantic_index = (index + insertion if owner == "generated" else
                          index + len(generated) if index >= insertion else index)
        if semantic_index not in physical_indexes:
            raise ValueError("layout changed an owned source carrier")
        physical_index = physical_indexes[semantic_index]
        content = physical[physical_index].get("content")
        if not isinstance(content, str):
            raise ValueError("provider normalization changed source carrier")
        raw = content.encode()
        if owner == "fixed" and physical_index not in classified:
            sections["full_source"] += len(json.dumps(physical[physical_index]).encode())
            classified.add(physical_index)
        witnesses.append(CarrierWitness(slot.key, slot.obligation_ids, slot.identity,
            modes[slot.key].name, physical_index, 0, len(raw), digest(raw),
            payload_start, payload_end,
            slot.recovery_arguments if modes[slot.key] == core.FULL
            else _recovery_arguments(snapshot, slot)))
    sections["fixed_and_layout"] = len(wire) - sum(sections.values())
    selected = tuple(sorted((key, mode.name) for key, mode in modes.items()))
    snap_digest = snapshot_digest(snapshot)
    assignment_digest = digest(json.dumps([snap_digest, layout, selected], separators=(",", ":")).encode())
    aggregate = None
    if deferred_plan:
        index = physical_indexes.get(insertion + deferred_notice_index)
        if index is None:
            raise ValueError("layout changed the deferred plan carrier")
        raw = physical[index]["content"].encode()
        aggregate = DeferredPlanWitness(tuple(deferred_plan), index, 0, len(raw), digest(raw))
    certificate = AssignmentCertificate(snapshot, selected, layout, snap_digest,
        assignment_digest, digest(wire), tuple(witnesses), aggregate)
    return RenderedCandidate(wire, certificate, tuple(sections.items()))


def render_assignment(snapshot: PreparationSnapshot,
                      assignment: Mapping[str, core.Mode] | core.Assignment,
                      layout: str = "plain") -> RenderedCandidate:
    """Render the whole candidate and its carrier-bound certificate."""
    return _render_assignment(snapshot, assignment, layout)


def preview_assignment(snapshot, assignment, layout="plain"):
    """Price exact plain wires without building unused witness inventories.

    FULL payload evidence and omission routes are still validated by the body
    builder. Projected layouts require physical-carrier reconciliation even
    for trials and therefore retain the full renderer. Every selected preview
    is fully rendered and independently proved before measurement or exposure.
    """
    if layout != "plain":
        return render_assignment(snapshot, assignment, layout)
    wire = _render_assignment(snapshot, assignment, layout, wire_only=True)
    modes = assignment.modes if isinstance(assignment, core.Assignment) else assignment
    return _WirePreview(wire, snapshot, tuple(modes.items()))


preview_assignment.rebind = _rebind_preview


def propose_slots(snapshot: PreparationSnapshot) -> tuple[core.Slot, ...]:
    """Measure each legal mode against the same frozen completion.

    Costs include JSON escaping and dependent summaries. The conditional solver
    remeasures the complete assignment; adding these diagnostics is not proof.
    A nonnegative per-slot offset handles richer-but-cheaper representations.
    """
    baseline = {s.key: _slot_modes(snapshot, s)[-1] for s in snapshot.slots}
    result = []
    for slot in snapshot.slots:
        measured = [(mode, len(trial_render(snapshot, {**baseline, slot.key: mode},
                                           "plain", preview_assignment).wire))
                    for mode in _slot_modes(snapshot, slot)]
        offset = min(cost for _mode, cost in measured)
        result.append(core.Slot(slot.key, slot.kind, _tier(snapshot, slot),
            slot.admission_ordinal, slot.last_use_ordinal,
            tuple((mode, cost - offset) for mode, cost in measured),
            previous_mode={"F": core.FULL, "D": core.DEGRADED,
                           "O": core.DEFERRED}.get(slot.previous_mode)))
    return tuple(result)
