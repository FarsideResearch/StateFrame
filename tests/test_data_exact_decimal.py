"""Data workspace exact money contract: decimal parsing, exact sums, native-sum safety, exact means, formatting.

Red tests for stage 2. The module under test (kernel_data_exact) is not implemented yet, so every test is a
strict xfail that fails on ImportError. Modules are imported inside each test so the file always collects.
"""
from __future__ import annotations

import fractions
import importlib

import pytest


_NATIVE_MAX = 2**63 - 1


# --- parse_decimal ----------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12.50", (1250, 2)),
        ("-0.5", (-5, 1)),
        ("7", (7, 0)),
    ],
)
def test_parse_decimal_returns_coefficient_and_scale(text, expected):
    mod = importlib.import_module("kernel_data_exact")
    assert mod.parse_decimal(text) == expected


@pytest.mark.parametrize("text", ["1e5", "1E-3", "1.5e+2", "", "1.2.3", "NaN", "inf"])
def test_parse_decimal_rejects_exponent_empty_malformed_and_non_finite(text):
    mod = importlib.import_module("kernel_data_exact")
    with pytest.raises(ValueError):
        mod.parse_decimal(text)


# --- exact_sum --------------------------------------------------------------

def test_exact_sum_has_no_float_drift():
    mod = importlib.import_module("kernel_data_exact")
    # 0.1 + 0.2 is 0.30000000000000004 in float; exact decimal sum must be 0.3.
    assert mod.exact_sum([(1, 1), (2, 1)]) == (3, 1)


def test_exact_sum_rescales_mixed_scales_to_max_scale():
    mod = importlib.import_module("kernel_data_exact")
    # 0.1 + 0.25 = 0.35
    assert mod.exact_sum([(1, 1), (25, 2)]) == (35, 2)


def test_exact_sum_handles_huge_integers_exactly():
    mod = importlib.import_module("kernel_data_exact")
    assert mod.exact_sum([(10**30, 0), (10**30, 0)]) == (2 * 10**30, 0)


def test_exact_sum_huge_value_mixed_with_fractional_part_stays_exact():
    mod = importlib.import_module("kernel_data_exact")
    # 10**30 + 0.000...1 (30 digits of scale): coefficient becomes 10**60 + 1 at scale 30.
    assert mod.exact_sum([(10**30, 0), (1, 30)]) == (10**60 + 1, 30)


# --- native_sum_safe --------------------------------------------------------

def test_native_sum_safe_accepts_exact_int64_boundary_single_value():
    mod = importlib.import_module("kernel_data_exact")
    assert mod.native_sum_safe(_NATIVE_MAX, 1) is True


def test_native_sum_safe_rejects_one_past_int64_boundary():
    mod = importlib.import_module("kernel_data_exact")
    assert mod.native_sum_safe(_NATIVE_MAX, 2) is False


def test_native_sum_safe_count_boundary_is_2_pow_63():
    mod = importlib.import_module("kernel_data_exact")
    assert mod.native_sum_safe(0, 2**63) is True
    assert mod.native_sum_safe(0, 2**63 + 1) is False


def test_native_sum_safe_product_just_under_and_over_limit():
    mod = importlib.import_module("kernel_data_exact")
    assert mod.native_sum_safe(1, _NATIVE_MAX) is True
    assert mod.native_sum_safe(2, 2**62) is False


# --- exact_mean -------------------------------------------------------------

def test_exact_mean_returns_exact_fraction():
    mod = importlib.import_module("kernel_data_exact")
    # (0.1 + 0.2) / 2 = 0.15 = 3/20
    result = mod.exact_mean([(1, 1), (2, 1)])
    assert isinstance(result, fractions.Fraction)
    assert result == fractions.Fraction(3, 20)


def test_exact_mean_keeps_non_terminating_quotients_exact():
    mod = importlib.import_module("kernel_data_exact")
    # mean of 1, 1, 2 is 4/3, a non-terminating binary/decimal quotient that must stay rational.
    assert mod.exact_mean([(1, 0), (1, 0), (2, 0)]) == fractions.Fraction(4, 3)


def test_exact_mean_of_empty_sequence_raises_value_error():
    mod = importlib.import_module("kernel_data_exact")
    with pytest.raises(ValueError):
        mod.exact_mean([])


# --- format_decimal ---------------------------------------------------------

@pytest.mark.parametrize("text", ["12.50", "-0.5", "0", "0.05", "-0.05", "100"])
def test_format_decimal_round_trips_canonical_strings(text):
    mod = importlib.import_module("kernel_data_exact")
    assert mod.format_decimal(mod.parse_decimal(text)) == text


def test_format_decimal_pads_fraction_with_leading_zeros():
    mod = importlib.import_module("kernel_data_exact")
    # coefficient 5 at scale 2 is 0.05, not 0.5 or 5
    assert mod.format_decimal((5, 2)) == "0.05"
