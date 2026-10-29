"""One run-scoped source of truth for the serving slot and prompt ceiling."""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

CONTEXT_ENV = "STATEFRAME_CONTEXT_TOKENS"
PROMPT_ENV = "STATEFRAME_PROMPT_BUDGET"
DEFAULT_CONTEXT_TOKENS = 65_536
DEFAULT_PROMPT_BUDGET = 33_344
DEFAULT_OUTPUT_TOKENS = 24_000
DEFAULT_THINKING_TOKENS = 4_096
DEFAULT_SAFETY_TOKENS = 4_096


@dataclass(frozen=True)
class EndpointSize:
    context_tokens: int
    prompt_budget: int
    output_tokens: int
    thinking_tokens: int
    safety_tokens: int
    configured: bool
    budget_explicit: bool = False

    def run_config(self) -> dict[str, int | bool]:
        return {"context_tokens": self.context_tokens,
                "prompt_budget": self.prompt_budget,
                "configured": self.configured}

    def startup_line(self) -> str | None:
        if not self.configured:
            return None
        source = ("budget set explicitly" if self.budget_explicit else
                  f"budget derived from slot at {DEFAULT_PROMPT_BUDGET}/{DEFAULT_CONTEXT_TOKENS}")
        return (f"Context: slot {self.context_tokens} tokens, prompt budget "
                f"{self.prompt_budget} tokens ({source})")


def _positive_integer(name: str, raw: object) -> int:
    try:
        value = int(raw) if isinstance(raw, str) and raw.isdecimal() else 0
    except ValueError:  # Python limits conversion of oversized decimal strings.
        value = 0
    if value < 1:
        supplied = (raw if len(raw) <= 80 else f"{raw[:80]}... ({len(raw)} chars)") \
            if isinstance(raw, str) else f"<{type(raw).__name__}>"
        raise ValueError(f"{name}={supplied} is invalid; minimum 1 token. "
                         "Set it to a positive integer or unset it.")
    return value


def resolve_endpoint_size(environment: Mapping[str, str] | None = None) -> EndpointSize:
    """Resolve once per run; unset preserves the historical request limits."""
    source = os.environ if environment is None else environment
    context_set = CONTEXT_ENV in source
    budget_set = PROMPT_ENV in source
    context = (_positive_integer(CONTEXT_ENV, source[CONTEXT_ENV]) if context_set
               else DEFAULT_CONTEXT_TOKENS)
    budget = (_positive_integer(PROMPT_ENV, source[PROMPT_ENV]) if budget_set
              else context * DEFAULT_PROMPT_BUDGET // DEFAULT_CONTEXT_TOKENS)
    output = context * DEFAULT_OUTPUT_TOKENS // DEFAULT_CONTEXT_TOKENS
    thinking = context * DEFAULT_THINKING_TOKENS // DEFAULT_CONTEXT_TOKENS
    safety = context * DEFAULT_SAFETY_TOKENS // DEFAULT_CONTEXT_TOKENS
    if context_set or budget_set:
        # The task-dependent protected minimum is measured on the assembled
        # candidate by FrameFit5; its selection ceiling is not that minimum.
        if budget < 1:
            raise ValueError(f"{CONTEXT_ENV}={context} derives a prompt budget below "
                             "the minimum 1 token. Increase it or unset it.")
        maximum = context - output - thinking - safety
        if budget > maximum:
            raise ValueError(f"{PROMPT_ENV}={budget} exceeds the maximum {maximum} "
                             f"tokens for {CONTEXT_ENV}={context} after reserving "
                             f"{output + thinking + safety} tokens. Set it to at most "
                             f"{maximum} or unset it.")
    return EndpointSize(context, budget, output, thinking, safety,
                        context_set or budget_set, budget_set)


ENDPOINT_SIZE = resolve_endpoint_size()
