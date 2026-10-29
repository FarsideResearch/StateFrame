"""Immutable evidence produced by the meter, consumed by final reconciliation.

This value does not admit requests. The existing budget reconciler and request
seal own admission and validate the value against their final body and limits.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json

PROJECTION_VERSION = "provider-prompt-json/1"


def prompt_projection(body: Mapping[str, object]) -> str:
    # JSON preserves roles, message partitions, tools, call IDs/arguments and
    # provider template options. Output allocation and sampling are not prompt
    # text; their changes still require fresh body-bound admission evidence.
    fields = ("messages", "tools", "functions", "tool_choice", "function_call",
              "response_format", "chat_template", "chat_template_kwargs",
              "add_generation_prompt", "continue_final_message", "parallel_tool_calls",
              "reasoning_effort", "reasoning_mode", "reasoning_budget_tokens")
    return json.dumps({key: body[key] for key in fields if key in body},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    body_sha256: str
    request_bytes: int
    projection_sha256: str
    projection_version: str
    tokenizer_identity: str
    request_id: str
    logical_request_id: str
    prompt_tokens: int | None
    rejection_bound: int | None
    method: str
    output_tokens: int
    thinking_tokens: int
    safety_tokens: int
    template_overhead_tokens: int
    max_prompt_tokens: int
    max_request_bytes: int

    @classmethod
    def bind(cls, body: Mapping[str, object], *, tokenizer_identity: str,
             request_id: str, logical_request_id: str, tokens: int, method: str,
             template_overhead_tokens: int, max_prompt_tokens: int,
             max_request_bytes: int) -> RequestMeasurement:
        from kernel_request_budget import (
            ENDPOINT_OUTPUT_BUDGET_TOKENS, ENDPOINT_THINKING_BUDGET_TOKENS,
            ENDPOINT_SAFETY_HEADROOM_TOKENS,
        )
        wire = json.dumps(body).encode()
        output = body.get("max_tokens")
        if type(output) is not int or output <= 0:
            output = ENDPOINT_OUTPUT_BUDGET_TOKENS
        return cls(
            digest(wire), len(wire), digest(prompt_projection(body).encode()),
            PROJECTION_VERSION, tokenizer_identity, request_id, logical_request_id,
            None if method == "verified_byte_bound" else tokens,
            tokens if method == "verified_byte_bound" else None, method,
            output, ENDPOINT_THINKING_BUDGET_TOKENS, ENDPOINT_SAFETY_HEADROOM_TOKENS,
            template_overhead_tokens if method == "tokenize" else 0,
            max_prompt_tokens, max_request_bytes,
        )

    def matches(self, body: Mapping[str, object], *, tokenizer_identity: str,
                request_id: str, logical_request_id: str) -> bool:
        return (self.body_sha256 == digest(json.dumps(body).encode())
                and self.projection_sha256 == digest(prompt_projection(body).encode())
                and self.projection_version == PROJECTION_VERSION
                and self.tokenizer_identity == tokenizer_identity
                and self.request_id == request_id
                and self.logical_request_id == logical_request_id)

    def receipt_fields(self) -> dict[str, object]:
        return {"measurement": asdict(self), "measurement_body_sha256": self.body_sha256,
                "measured_prompt_tokens": self.prompt_tokens, "measure_method": self.method,
                "tokenizer_identity": self.tokenizer_identity,
                "template_overhead_tokens": self.template_overhead_tokens,
                "max_prompt_tokens": self.max_prompt_tokens, "max_request_bytes": self.max_request_bytes}

    def validate(self, body: Mapping[str, object], receipt: Mapping[str, object], *,
                 request_id: str, logical_request_id: str, tokenizer_identity: str) -> None:
        from kernel_request_seal import RequestSealError
        if not self.matches(body, tokenizer_identity=tokenizer_identity,
                            request_id=request_id, logical_request_id=logical_request_id):
            raise RequestSealError("measurement body/projection/provider identity changed")
        if (self.request_bytes != len(json.dumps(body).encode())
                or receipt.get("body_sha256") != self.body_sha256
                or receipt.get("request_bytes") != self.request_bytes
                or any(receipt.get(key) != value for key, value in self.receipt_fields().items())):
            raise RequestSealError("measurement and final budget receipt disagree")
        if (self.method not in {"tokenize", "calibrated_bytes"}
                or type(self.prompt_tokens) is not int or self.prompt_tokens < 0
                or self.prompt_tokens > self.max_prompt_tokens
                or self.request_bytes > self.max_request_bytes):
            raise RequestSealError("measurement does not establish request capacity")
        from kernel_request_budget import (
            ENDPOINT_CONTEXT_TOKENS, ENDPOINT_OUTPUT_BUDGET_TOKENS,
            ENDPOINT_SAFETY_HEADROOM_TOKENS, ENDPOINT_THINKING_BUDGET_TOKENS,
        )
        output = body.get("max_tokens")
        if type(output) is not int or output <= 0:
            output = ENDPOINT_OUTPUT_BUDGET_TOKENS
        if (output != self.output_tokens
                or self.thinking_tokens != ENDPOINT_THINKING_BUDGET_TOKENS
                or self.safety_tokens != ENDPOINT_SAFETY_HEADROOM_TOKENS
                or self.max_prompt_tokens > max(0, ENDPOINT_CONTEXT_TOKENS - output
                                               - self.thinking_tokens - self.safety_tokens)):
            raise RequestSealError("measurement output/thinking/safety reserves disagree")
