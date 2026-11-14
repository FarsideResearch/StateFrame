"""Disk-backed scratch for tests that start a Linux sandbox boundary."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path


def sandbox_visible_temp_dir() -> tempfile.TemporaryDirectory[str]:
    """Keep executable test helpers visible when bwrap replaces /dev."""
    directory = "/tmp" if sys.platform.startswith("linux") else None
    return tempfile.TemporaryDirectory(prefix="stateframe-sandbox-test-", dir=directory)


def shadowed_by_bwrap_dev(path: Path) -> bool:
    return sys.platform.startswith("linux") and path.resolve().is_relative_to("/dev/shm")


def timeout_scale() -> float:
    """Scale test-only wall budgets without changing product deadlines."""
    raw = os.environ.get("SF_TEST_TIMEOUT_SCALE", "1")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError("SF_TEST_TIMEOUT_SCALE must be a finite positive number") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("SF_TEST_TIMEOUT_SCALE must be a finite positive number")
    return value
