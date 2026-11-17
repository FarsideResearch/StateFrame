"""Exact decimal arithmetic for the data workspace money paths.

A value is ``(coefficient, scale)`` meaning ``coefficient * 10**-scale``, held in Python ints so nothing rounds.
Floats are never accepted or produced. Parsing is strict: plain decimal notation only, with no exponents,
no NaN and no infinity. ``native_sum_safe`` guards any path that would hand these sums to a native int64 engine.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from fractions import Fraction

ExactDecimal = tuple[int, int]

MAX_SCALE = 38
NATIVE_INT64_MAX = 2**63 - 1
NATIVE_COUNT_MAX = 2**63

_DECIMAL_TEXT = re.compile(r"(?P<sign>[+-]?)(?P<whole>[0-9]+)(?:\.(?P<frac>[0-9]+))?")


def _check(value: ExactDecimal) -> None:
    """Reject non-int parts (bool included) and scales outside ``0..MAX_SCALE``."""
    coefficient, scale = value
    if type(coefficient) is not int or type(scale) is not int:
        raise TypeError("exact decimal parts must be int")
    if not 0 <= scale <= MAX_SCALE:
        raise ValueError("exact decimal scale out of range")


def parse_decimal(text: str) -> ExactDecimal:
    """Parse plain decimal text such as ``-12.50`` into ``(coefficient, scale)``."""
    match = _DECIMAL_TEXT.fullmatch(text)
    if match is None:
        raise ValueError("malformed decimal text")
    frac = match.group("frac") or ""
    if len(frac) > MAX_SCALE:
        raise ValueError("decimal scale out of range")
    magnitude = int(match.group("whole") + frac)
    coefficient = -magnitude if match.group("sign") == "-" else magnitude
    return (coefficient, len(frac))


def exact_sum(values: Sequence[ExactDecimal]) -> ExactDecimal:
    """Sum exactly, rescaling every term to the largest scale present. Empty sums to ``(0, 0)``."""
    if not values:
        return (0, 0)
    for value in values:
        _check(value)
    scale = max(value[1] for value in values)
    total = 0
    for coefficient, term_scale in values:
        total += coefficient * 10 ** (scale - term_scale)
    return (total, scale)


def native_sum_safe(max_abs_value: int, count: int) -> bool:
    """True when ``count`` values of magnitude up to ``max_abs_value`` fit a native int64 sum exactly."""
    if type(max_abs_value) is not int or type(count) is not int:
        raise TypeError("native_sum_safe takes ints")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > NATIVE_COUNT_MAX:
        return False
    return abs(max_abs_value) * count <= NATIVE_INT64_MAX


def exact_mean(values: Sequence[ExactDecimal]) -> Fraction:
    """Exact mean as a ``Fraction``, so non-terminating quotients stay rational."""
    if not values:
        raise ValueError("exact_mean of an empty sequence is undefined")
    coefficient, scale = exact_sum(values)
    return Fraction(coefficient, 10**scale) / len(values)


def format_decimal(value: ExactDecimal) -> str:
    """Render ``(coefficient, scale)`` as canonical plain decimal text, keeping trailing zeros."""
    _check(value)
    coefficient, scale = value
    sign = "-" if coefficient < 0 else ""
    digits = str(abs(coefficient))
    if scale == 0:
        return sign + digits
    digits = digits.rjust(scale + 1, "0")
    return f"{sign}{digits[:-scale]}.{digits[-scale:]}"
