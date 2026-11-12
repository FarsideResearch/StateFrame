"""Shared scaling for test-only wall and process wait timeouts."""
from __future__ import annotations

import math
import os


_TIMEOUT_SCALE_ENV = "SF_TEST_TIMEOUT_SCALE"


def timeout_scale() -> float:
    """Return the validated suite timeout multiplier captured at import."""
    return _SCALE


def _read_timeout_scale() -> float:
    try:
        scale = float(os.environ.get(_TIMEOUT_SCALE_ENV, "1.0"))
    except ValueError as error:
        raise ValueError(f"{_TIMEOUT_SCALE_ENV} must be a finite float greater than 0") from error
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{_TIMEOUT_SCALE_ENV} must be a finite float greater than 0")
    return scale


_SCALE = _read_timeout_scale()


def scaled(seconds: float) -> float:
    """Scale one test timeout while preserving its default duration."""
    return seconds * timeout_scale()
