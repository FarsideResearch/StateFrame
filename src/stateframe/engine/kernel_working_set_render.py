"""Pure complete-candidate oracle for the default-off working-set shadow.

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
    from .kernel_read_receipts import ReceiptIdentity, line_digests_with_numbers
else:
    import kernel_working_set_allocator as core
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
class AssignmentCertificate:
    # snapshot is a trusted host object, not deserialized from model prose.
    snapshot: PreparationSnapshot
    assignment: tuple[tuple[str, str], ...]
    layout: str
    snapshot_digest: str
    assignment_digest: str
    body_sha256: str
    witnesses: tuple[CarrierWitness, ...]


@dataclass(frozen=True)
class RenderedCandidate:
    wire: bytes
    certificate: AssignmentCertificate
    section_breakdown: tuple[tuple[str, int], ...]


def _slot_modes(snapshot: PreparationSnapshot, slot: SourceSlot) -> tuple[core.Mode, ...]:
    if slot.state == "DUE" or slot.recovery_pin:
        return (core.FULL,)
    if slot.state in {"RETIRING", "DISCHARGED", "SUPERSEDED", "DEGRADED"}:
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
    return core.PriorityTier.OPTIONAL


def _assignment(snapshot: PreparationSnapshot,
                assignment: Mapping[str, core.Mode] | core.Assignment) -> dict[str, core.Mode]:
    modes = assignment.modes if isinstance(assignment, core.Assignment) else assignment
    if not isinstance(modes, Mapping) or set(modes) != {s.key for s in snapshot.slots}:
        raise ValueError("assignment must select every frozen slot exactly once")
    if len({s.key for s in snapshot.slots}) != len(snapshot.slots):
        raise ValueError("duplicate frozen slot key")
    result = {}
    for slot in snapshot.slots:
        mode = modes[slot.key]
        if mode not in _slot_modes(snapshot, slot):
            raise ValueError("assignment violates frozen slot domain")
        result[slot.key] = mode
    return result


def _recovery_arguments(snapshot: PreparationSnapshot, slot: SourceSlot) -> bytes:
    if __package__:
        from .kernel_working_set_recovery import validate_recovery
    else:
        from kernel_working_set_recovery import validate_recovery
    return json.dumps(validate_recovery(snapshot, slot),
                      sort_keys=True, separators=(",", ":")).encode()


def _notice(snapshot: PreparationSnapshot, slot: SourceSlot) -> str:
    identity = slot.identity
    arguments = json.loads(_recovery_arguments(snapshot, slot))
    # Preserve positional/cursor recovery arguments, including lossy columns.
    call = "read(" + ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                               for key, value in sorted(arguments.items())) + ")"
    return ("[Source page deferred]\n"
            f"path={identity.path} revision={identity.revision} "
            f"lines={identity.first_line}-{identity.last_line} digest={identity.payload_sha256}\n"
            f"Retrieve current bytes with: {call}.")


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
    result = enforce_provider_body(body, profile=ReasoningProfile(snapshot.reasoning_mode, "snapshot"),
                                   dialect=snapshot.dialect, environ={})
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


def render_assignment(snapshot: PreparationSnapshot,
                      assignment: Mapping[str, core.Mode] | core.Assignment,
                      layout: str = "plain") -> RenderedCandidate:
    """Render selected payloads, their dependent truth and provider bytes once.

    Slot order is admission order, independent of solver decision order. FULL
    carriers can share only exact identity+page material or a frozen native tool
    result whose whole page is equal to the captured page. No path-only lookup.
    """
    modes = _assignment(snapshot, assignment)
    if layout not in snapshot.layouts or layout not in {"plain", "final_cache"}:
        raise ValueError("layout not admitted by snapshot")
    body = json.loads(snapshot.fixed_body)
    messages = body.get("messages")
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        raise ValueError("frozen provider body requires messages")
    insertion = _insertion_index(snapshot, messages)
    generated = []
    locations = {}
    shared = {}
    categories = {}
    ordered = sorted(snapshot.slots, key=lambda s: (s.admission_ordinal, s.key))
    for slot in ordered:
        mode = modes[slot.key]
        if mode == core.FULL:
            payload_start, payload_end = _payload_interval(slot)
            key = (slot.identity, slot.page)
            if key in shared:
                locations[slot.key] = shared[key]
                continue
            # Only a whole captured tool page is a reusable fixed carrier.
            # User prose cannot mint ownership by including a source string.
            fixed = next((i for i, m in enumerate(messages)
                          if m.get("role") == "tool" and m.get("content") == slot.page), None)
            if fixed is not None:
                location = ("fixed", fixed, payload_start, payload_end)
            else:
                location = ("generated", len(generated), payload_start, payload_end)
                generated.append({"role": "user", "content": slot.page})
                categories[len(generated) - 1] = "full_source"
            locations[slot.key] = shared[key] = location
        else:
            if not snapshot.recovery_allowed:
                raise ValueError("frozen tool/owner surface refuses source recovery")
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
    rebuild_delivery_truth(body, snapshot, full)
    plain = _normalize(body, snapshot)
    if layout == "final_cache":
        if __package__:
            from .kernel_cache_layout_candidate import _checked_final_cache_projection
        else:
            from kernel_cache_layout_candidate import _checked_final_cache_projection
        body = _checked_final_cache_projection(plain)
    else:
        body = plain
    wire = json.dumps(body).encode()
    witnesses = []
    sections = {"full_source": 0, "deferred_notices": 0, "host_facts": 0, "coverage": 0}
    physical = body["messages"]
    plain_messages = plain["messages"]
    # Cache projection preserves source/notice carriers as whole rows; derive
    # physical indexes by whole-row identity, never by source path or interval.
    physical_indexes = {}
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
    certificate = AssignmentCertificate(snapshot, selected, layout, snap_digest,
                                        assignment_digest, digest(wire), tuple(witnesses))
    return RenderedCandidate(wire, certificate, tuple(sections.items()))


def propose_slots(snapshot: PreparationSnapshot) -> tuple[core.Slot, ...]:
    """Measure each legal mode against the same frozen completion.

    Costs include JSON escaping and dependent summaries. The conditional solver
    remeasures the complete assignment; adding these diagnostics is not proof.
    A nonnegative per-slot offset handles richer-but-cheaper representations.
    """
    baseline = {s.key: _slot_modes(snapshot, s)[-1] for s in snapshot.slots}
    result = []
    for slot in snapshot.slots:
        measured = [(mode, len(render_assignment(snapshot, {**baseline, slot.key: mode}, "plain").wire))
                    for mode in _slot_modes(snapshot, slot)]
        offset = min(cost for _mode, cost in measured)
        result.append(core.Slot(slot.key, slot.kind, _tier(snapshot, slot),
            slot.admission_ordinal, slot.last_use_ordinal,
            tuple((mode, cost - offset) for mode, cost in measured)))
    return tuple(result)
