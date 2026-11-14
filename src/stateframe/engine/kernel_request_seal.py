"""Immutable request inputs and the single provider authorization record."""
from __future__ import annotations

import kernel_frame_routes as _frame_routes

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import time

from kernel_frame_request_allocation import FrameRequestAllocation
from kernel_request_measurement import RequestMeasurement
from kernel_reason_log import emit_reason, emit_timeout


def freeze(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def log_seal_violation(message: str, detail: str | None = None) -> None:
    """Log a broken seal invariant once per run (logging only; the caller still raises).

    Shared by every site that raises RequestSealError. `message` must be a fixed string (its
    prefix before ": " is the cause_key); variable text goes in `detail`.
    """
    # cause_key: the message up to its ": <detail>" suffix, so it stays a small fixed set.
    # journal=False: request path, journal positions are identity-bearing.
    emit_reason("invariant", message.split(": ", 1)[0], "request_seal", "qualified request seal violated",
                f"{message}: {detail}" if detail else message, level="ERROR", journal=False)


_seal_violation = log_seal_violation


class RequestSealError(RuntimeError):
    harness_contract = True
    outcome = "HARNESS_CONTRACT"


@dataclass(frozen=True, slots=True)
class OwnerPresentation:
    """Owner-authorized presentation, independent of stripped semantic text."""
    contract_digest: str
    continuation_limits: str | None
    evidence_pin: str | None

    @classmethod
    def capture(cls, runtime: object, contract: Mapping[str, object], messages):
        from kernel_completion_owner import _contract_evidence_pin_value, _digest, _wire_continuation_limits

        limits = _wire_continuation_limits(messages)
        expected = contract.get("budget_wire_digest")
        if expected is not None and (limits is None or _digest(limits) != expected):
            _seal_violation("owner presentation limits differ from admitted contract")
            raise RequestSealError("owner presentation limits differ from admitted contract")
        pin = _contract_evidence_pin_value(runtime)
        if contract.get("evidence_pin_wire_digest") is not None and _digest([pin]) != contract["evidence_pin_wire_digest"]:
            _seal_violation("owner presentation evidence differs from admitted contract")
            raise RequestSealError("owner presentation evidence differs from admitted contract")
        return cls(hashlib.sha256(freeze(contract)).hexdigest(), limits, pin)

    def validate(self, contract: Mapping[str, object]) -> None:
        if hashlib.sha256(freeze(contract)).hexdigest() != self.contract_digest:
            _seal_violation("owner presentation belongs to another contract")
            raise RequestSealError("owner presentation belongs to another contract")

    def rebind(
            self, previous: Mapping[str, object], current: Mapping[str, object],
    ) -> OwnerPresentation:
        """Carry immutable owner prose across an authorized contract rebind."""
        from kernel_completion_owner import _digest

        self.validate(previous)
        limits_digest = current.get("budget_wire_digest")
        if previous.get("budget_wire_digest") is not None and limits_digest is None:
            _seal_violation("owner presentation limits removed from rebound contract")
            raise RequestSealError("owner presentation limits removed from rebound contract")
        if limits_digest is not None and (
                self.continuation_limits is None
                or _digest(self.continuation_limits) != limits_digest):
            _seal_violation("owner presentation limits differ from rebound contract")
            raise RequestSealError("owner presentation limits differ from rebound contract")
        pin_digest = current.get("evidence_pin_wire_digest")
        if previous.get("evidence_pin_wire_digest") is not None and pin_digest is None:
            _seal_violation("owner presentation evidence removed from rebound contract")
            raise RequestSealError("owner presentation evidence removed from rebound contract")
        if pin_digest is not None and _digest([self.evidence_pin]) != pin_digest:
            _seal_violation("owner presentation evidence differs from rebound contract")
            raise RequestSealError("owner presentation evidence differs from rebound contract")
        return OwnerPresentation(
            hashlib.sha256(freeze(current)).hexdigest(),
            self.continuation_limits, self.evidence_pin,
        )


@dataclass(frozen=True, slots=True)
class FrameSelectionProposal:
    """Detached selection inventory. This record grants no wire authority."""
    optional: bytes
    inventory: bytes
    limits: bytes

    @property
    def optional_heads(self):
        return json.loads(self.optional)

    @property
    def site_inventory(self):
        return json.loads(self.inventory)

    @property
    def capacity(self):
        return json.loads(self.limits)


@dataclass(frozen=True, slots=True)
class QualifiedRequest:
    request_id: str
    wire: bytes
    body_digest: str
    allocation: FrameRequestAllocation
    metadata: bytes
    binding: bytes
    input_digest: str
    owner_presentation: OwnerPresentation
    deadline: float | None
    lineage: str
    measurement: RequestMeasurement

    @property
    def body(self):
        return json.loads(self.wire)

    def validate(self, capture: object, runtime: object, body=None) -> None:
        from kernel_frame_allocation_validation import validate_allocation_authority
        from kernel_frame_optional_allocation import FrameAllocationReplanNeeded, identity, validate_optional_dependencies
        from kernel_request_reprepare import _current_binding
        from kernel_stateframe_framefit5 import _workspace_root

        if capture.request_id != self.request_id or runtime.current_request_id != self.request_id:
            _seal_violation("qualified request identity changed")
            raise RequestSealError("qualified request identity changed")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            emit_timeout("request_seal", what="qualified request", cause_key="deadline",
                         why="qualified request deadline expired", journal=False)
            raise TimeoutError("qualified request deadline expired")
        if hashlib.sha256(self.wire).hexdigest() != self.body_digest:
            _seal_violation("qualified request wire digest changed")
            raise RequestSealError("qualified request wire digest changed")
        candidate = self.body if body is None else body
        if json.dumps(candidate).encode() != self.wire:
            _seal_violation("qualified request body changed after seal")
            raise RequestSealError("qualified request body changed after seal")
        validate_optional_dependencies(self.allocation.optional_heads, root=_workspace_root(runtime))
        if freeze(_current_binding(runtime)) != self.binding:
            raise FrameAllocationReplanNeeded("allocation product revision changed")
        if self.allocation.outcome["body_digest"] != identity(candidate):
            _seal_violation("qualified allocation and body differ")
            raise RequestSealError("qualified allocation and body differ")
        if capture.frame_request_allocation is not self.allocation:
            _seal_violation("active allocation differs from qualified request")
            raise RequestSealError("active allocation differs from qualified request")
        authority = json.loads(self.metadata)
        for field, attribute in (("request_kind", "request_kind"), ("adapter", "_qualified_adapter")):
            if field in authority and authority[field] != getattr(capture, attribute, None):
                _seal_violation("qualified request classification changed: " + field)
                raise RequestSealError("qualified request classification changed: " + field)
        meter = (getattr(capture, "_request_token_meter", None)
                 if _frame_routes.ROUTES_ACTIVE and getattr(capture, "frame_route", None) is not None
                 else getattr(runtime, "_request_token_meter", None))
        if meter is None:
            meter = getattr(capture, "_request_token_meter", None)
        if meter is None:
            _seal_violation("qualified request has no measurement provider")
            raise RequestSealError("qualified request has no measurement provider")
        if getattr(capture, "_request_measurement", None) is not self.measurement:
            _seal_violation("qualified measurement authority changed")
            raise RequestSealError("qualified measurement authority changed")
        from kernel_request_diet import resolve_tokenize_url, tokenizer_contract
        if meter.provider_identity != tokenizer_contract(runtime, capture):
            _seal_violation("qualified tokenizer provider contract changed")
            raise RequestSealError("qualified tokenizer provider contract changed")
        endpoint = resolve_tokenize_url(
            capture=capture if _frame_routes.ROUTES_ACTIVE and getattr(capture, "frame_route", None) is not None else None,
            endpoint=getattr(capture, "_tokenize_endpoint", None))
        self.measurement.validate(
            candidate, authority["request_budget_final_wire"], request_id=self.request_id,
            logical_request_id=self.lineage,
            tokenizer_identity=meter.tokenizer_identity(candidate, endpoint=endpoint),
        )
        if (self.allocation.capacity["byte_ceiling"] != self.measurement.max_request_bytes
                or self.allocation.capacity["token_ceiling"] != self.measurement.max_prompt_tokens
                or self.allocation.capacity["measurement_identity"] != self.measurement.method):
            _seal_violation("qualified allocation and measurement limits differ")
            raise RequestSealError("qualified allocation and measurement limits differ")
        for name in ("request_action_contract", "_request_source_closure_override",
                     "request_source_final_delivery", "request_budget_final_wire",
                     "final_tool_surface", "framefit5_product_validation"):
            if freeze(getattr(capture, name, None)) != freeze(authority[name]):
                _seal_violation("qualified request authority changed: " + name)
                raise RequestSealError("qualified request authority changed: " + name)
        if (getattr(capture, "owner_presentation", None) != self.owner_presentation
                or freeze(getattr(capture, "release_precomposed_host_text", None))
                != freeze(authority["host_contribution"])):
            _seal_violation("qualified host presentation authority changed")
            raise RequestSealError("qualified host presentation authority changed")
        self.owner_presentation.validate(authority["request_action_contract"])
        from kernel_completion_owner import assert_request_action_contract
        assert_request_action_contract(runtime, candidate["messages"], candidate.get("tools", []),
            request_id=self.request_id, boundary="qualified_request",
            frozen_contract=authority["request_action_contract"], read_only=True)
        for name, expected in authority.get("runtime_authority", {}).items():
            if freeze(getattr(runtime, name, None)) != freeze(expected):
                _seal_violation("qualified runtime authority changed: " + name)
                raise RequestSealError("qualified runtime authority changed: " + name)
        validate_allocation_authority(self.allocation, runtime, candidate["messages"], candidate.get("tools", []))
