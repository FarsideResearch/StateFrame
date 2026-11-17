"""Blank lines around a verification selector are not a line break inside it."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel_v30_token_contracts import verification_selector_is_executable


@pytest.mark.parametrize("selector", [
    "\ntests/test_pricing.py\n",
    "\n\ntests/test_pricing.py\n\n",
    "  \ntests/test_pricing.py\r\n",
    ["\n", "tests/test_pricing.py\n"],
])
def test_blank_lines_around_valid_selector_are_accepted(selector):
    assert verification_selector_is_executable(SimpleNamespace(), selector) is True


@pytest.mark.parametrize("selector", [
    "true\nrm -rf x",
    "true\r\nrm -rf x",
    "true \n rm -rf x",
    "\ntrue\n\nrm -rf x\n",
    ["tests/test_pricing.py", "true\nrm -rf x"],
])
def test_embedded_line_break_is_rejected(selector):
    assert verification_selector_is_executable(SimpleNamespace(), selector) is False


def test_blank_lines_around_selector_are_checked_against_host_allowlist():
    runtime = SimpleNamespace(authorized_verification_selectors=["tests/test_pricing.py"])
    assert verification_selector_is_executable(runtime, "\ntests/test_pricing.py\n") is True
    assert verification_selector_is_executable(runtime, "\ntests/test_other.py\n") is False
