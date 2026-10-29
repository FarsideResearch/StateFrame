"""Default-OFF admission policy for optional public review."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

DEADLINE_ENV = "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"


def _positive(name: str, default: float | None = None) -> float:
    raw = os.environ.get(name)
    if raw is None and default is None:
        raise ValueError(name + " is required")
    value = float(raw) if raw is not None else default
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


@dataclass(frozen=True)
class ReviewDeadline:
    parent_deadline: float
    call_seconds: float
    reserve_seconds: float
    probe_seconds: float = 20.0

    @classmethod
    def configured(cls):
        flag = os.environ.get("KERNEL_PUBLIC_REVIEW_DEADLINE", "0")
        if flag not in {"0", "1"}:
            raise ValueError("KERNEL_PUBLIC_REVIEW_DEADLINE must be 0 or 1")
        if flag == "0":
            return None
        from kernel_measurement import request_timeouts
        return cls(_positive(DEADLINE_ENV),
                   _positive("KERNEL_PUBLIC_REVIEW_CALL_SECONDS", request_timeouts()[1]),
                   _positive("KERNEL_PUBLIC_REVIEW_RETURN_RESERVE_SECONDS", 30.0))

    @property
    def deadline(self) -> float:
        return self.parent_deadline - self.reserve_seconds

    def admit(self, calls: int) -> dict:
        remaining = self.parent_deadline - time.monotonic()
        required = calls * self.call_seconds + self.probe_seconds + self.reserve_seconds
        return {"admitted": remaining >= required, "remaining_seconds": remaining,
                "required_seconds": required, "model_calls": calls,
                "call_seconds": self.call_seconds, "probe_seconds": self.probe_seconds,
                "return_reserve_seconds": self.reserve_seconds,
                "authority": "OPTIONAL_WORK_ADMISSION_ONLY"}

    def call_deadline(self) -> float:
        return min(self.deadline, time.monotonic() + self.call_seconds)
