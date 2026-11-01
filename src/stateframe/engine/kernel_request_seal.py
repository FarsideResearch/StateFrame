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


def freeze(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


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
            raise RequestSealError("owner presentation limits differ from admitted contract")
        pin = _contract_evidence_pin_value(runtime)
        if contract.get("evidence_pin_wire_digest") is not None and _digest([pin]) != contract["evidence_pin_wire_digest"]:
            raise RequestSealError("owner presentation evidence differs from admitted contract")
        return cls(hashlib.sha256(freeze(contract)).hexdigest(), limits, pin)

    def validate(self, contract: Mapping[str, object]) -> None:
        if hashlib.sha256(freeze(contract)).hexdigest() != self.contract_digest:
            raise RequestSealError("owner presentation belongs to another contract")


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
            raise RequestSealError("qualified request identity changed")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("qualified request deadline expired")
        if hashlib.sha256(self.wire).hexdigest() != self.body_digest:
            raise RequestSealError("qualified request wire digest changed")
        candidate = self.body if body is None else body
        if json.dumps(candidate).encode() != self.wire:
            raise RequestSealError("qualified request body changed after seal")
        validate_optional_dependencies(self.allocation.optional_heads, root=_workspace_root(runtime))
        if freeze(_current_binding(runtime)) != self.binding:
            raise FrameAllocationReplanNeeded("allocation product revision changed")
        if self.allocation.outcome["body_digest"] != identity(candidate):
            raise RequestSealError("qualified allocation and body differ")
        if capture.frame_request_allocation is not self.allocation:
            raise RequestSealError("active allocation differs from qualified request")
        authority = json.loads(self.metadata)
        for field, attribute in (("request_kind", "request_kind"), ("adapter", "_qualified_adapter")):
            if field in authority and authority[field] != getattr(capture, attribute, None):
                raise RequestSealError("qualified request classification changed: " + field)
        meter = (getattr(capture, "_request_token_meter", None)
                 if _frame_routes.ROUTES_ACTIVE and getattr(capture, "frame_route", None) is not None
                 else getattr(runtime, "_request_token_meter", None))
        if meter is None:
            meter = getattr(capture, "_request_token_meter", None)
        if meter is None:
            raise RequestSealError("qualified request has no measurement provider")
        if getattr(capture, "_request_measurement", None) is not self.measurement:
            raise RequestSealError("qualified measurement authority changed")
        from kernel_request_diet import resolve_tokenize_url, tokenizer_contract
        if meter.provider_identity != tokenizer_contract(runtime, capture):
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
            raise RequestSealError("qualified allocation and measurement limits differ")
        for name in ("request_action_contract", "_request_source_closure_override",
                     "request_source_final_delivery", "request_budget_final_wire",
                     "final_tool_surface", "framefit5_product_validation"):
            if freeze(getattr(capture, name, None)) != freeze(authority[name]):
                raise RequestSealError("qualified request authority changed: " + name)
        if (getattr(capture, "owner_presentation", None) != self.owner_presentation
                or freeze(getattr(capture, "release_precomposed_host_text", None))
                != freeze(authority["host_contribution"])):
            raise RequestSealError("qualified host presentation authority changed")
        self.owner_presentation.validate(authority["request_action_contract"])
        from kernel_completion_owner import assert_request_action_contract
        assert_request_action_contract(runtime, candidate["messages"], candidate.get("tools", []),
            request_id=self.request_id, boundary="qualified_request",
            frozen_contract=authority["request_action_contract"], read_only=True)
        for name, expected in authority.get("runtime_authority", {}).items():
            if freeze(getattr(runtime, name, None)) != freeze(expected):
                raise RequestSealError("qualified runtime authority changed: " + name)
        validate_allocation_authority(self.allocation, runtime, candidate["messages"], candidate.get("tools", []))
