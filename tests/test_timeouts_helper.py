"""Validation and caching behavior for shared test timeout configuration."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_timeouts():
    path = Path(__file__).with_name("_timeouts.py")
    spec = importlib.util.spec_from_file_location("_timeouts_test_instance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeout_scale_defaults_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SF_TEST_TIMEOUT_SCALE", raising=False)
    timeouts = _load_timeouts()

    assert timeouts.timeout_scale() == 1.0
    assert timeouts.scaled(2.5) == 2.5


def test_timeout_scale_caches_value_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_TEST_TIMEOUT_SCALE", "2.5")
    timeouts = _load_timeouts()
    monkeypatch.delenv("SF_TEST_TIMEOUT_SCALE")

    assert timeouts.timeout_scale() == 2.5
    assert timeouts.scaled(2.0) == 5.0
    assert timeouts.scaled(2.5) == 6.25


@pytest.mark.parametrize("value", ("", "not-a-number", "0", "-1", "nan", "inf", "-inf"))
def test_timeout_scale_rejects_invalid_or_nonpositive_values(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    monkeypatch.setenv("SF_TEST_TIMEOUT_SCALE", value)

    with pytest.raises(ValueError, match="SF_TEST_TIMEOUT_SCALE"):
        _load_timeouts()
