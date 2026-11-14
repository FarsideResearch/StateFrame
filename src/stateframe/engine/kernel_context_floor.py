"""Project an exhausted small-slot fit into a measured, actionable host stop."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING

from kernel_endpoint_size import (
    DEFAULT_CONTEXT_TOKENS, DEFAULT_PROMPT_BUDGET, ENDPOINT_SIZE, EndpointSize,
)


@dataclass(frozen=True)
class ContextTooSmall:
    configured_context_tokens: int
    minimum_context_tokens: int
    minimum_prompt_tokens: int
    measure_method: str

    def terminal_fields(self) -> dict[str, object]:
        return {
            "result": "RUN_INCOMPLETE", "reason_code": "CONTEXT_TOO_SMALL",
            "configured_context_tokens": self.configured_context_tokens,
            "minimum_context_tokens": self.minimum_context_tokens,
            "minimum_prompt_tokens": self.minimum_prompt_tokens,
            "minimum_measure_method": self.measure_method,
            "message": (f"model context {self.configured_context_tokens} is too small; "
                        f"StateFrame needs at least {self.minimum_context_tokens}"),
        }


def context_floor(detail: Mapping[str, object], *,
                  size: EndpointSize = ENDPOINT_SIZE) -> ContextTooSmall | None:
    """Use the final irreducible envelope, including the byte-admission floor.

    This is the context needed by this source-qualified minimum under the
    shipped proportional prompt policy. Estimated meters remain labelled as
    such. Explicitly smaller prompt/frame ceilings are independent constraints
    and must not be misreported as a model-slot failure.
    """
    if size.context_tokens >= DEFAULT_CONTEXT_TOKENS or size.budget_explicit:
        return None
    from kernel_request_budget import (
        ENDPOINT_REQUEST_BYTES_PER_TOKEN, ENDPOINT_VERIFIED_BYTES_PER_TOKEN,
    )

    def count(name: str) -> int:
        value = detail.get(name)
        return value if type(value) is int and value >= 0 else 0

    method = str(detail.get("measure_method", "unknown"))
    ratio = (ENDPOINT_VERIFIED_BYTES_PER_TOKEN
             if method in {"tokenize", "verified_byte_bound"}
             else ENDPOINT_REQUEST_BYTES_PER_TOKEN)
    wire_bytes = max(count("request_bytes"), count("protected_request_bytes"))
    byte_tokens = int((Decimal(wire_bytes) / Decimal(str(ratio))).to_integral_value(
        rounding=ROUND_CEILING))
    # Cache-layout headroom is optional: a larger slot can use the proven plain
    # minimum. It must not inflate the context recommendation for that minimum.
    prompt = max(count("prompt_tokens"), count("prompt_tokens_rejection_bound"), byte_tokens)
    if prompt <= size.prompt_budget:
        return None
    minimum = (prompt * DEFAULT_CONTEXT_TOKENS + DEFAULT_PROMPT_BUDGET - 1) // DEFAULT_PROMPT_BUDGET
    return ContextTooSmall(size.context_tokens, minimum, prompt, method)
