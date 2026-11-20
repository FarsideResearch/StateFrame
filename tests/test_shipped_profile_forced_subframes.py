"""The shipped profile must continue to dispatch provider requests to child frames."""
from __future__ import annotations

from pathlib import Path

import pytest

from stateframe import profiles
from tests.test_strict_provider_mock_gate import _run_forced_subframes

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src" / "stateframe" / "profiles" / "r2hc3.json"


@pytest.mark.host_sandbox
def test_shipped_profile_forced_subframes_send_child_requests(tmp_path: Path) -> None:
    # Match runner._provider_environment's profile loading, without ambient shell overrides.
    environment = profiles.load(str(PROFILE))
    environment["STATEFRAME_WORKSPACE_OBJECTS"] = "on"
    home = tmp_path / "home"
    home.mkdir()
    environment["HOME"] = str(home)
    _run_forced_subframes(tmp_path, profile_environment=environment)
