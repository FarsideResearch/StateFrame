"""A routed recorder must not contaminate the next test in one pytest worker."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.host_sandbox
def test_routed_recorder_then_http_gzip_recorder_in_one_process():
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly"]
    if importlib.util.find_spec("xdist") is not None:
        command += ["-n", "0"]
    command += [
        "-q",
        "tests/test_routes_cli.py::test_three_resumes_skip_unproven_pending_through_chained_endpoint_and_model_moves",
        "tests/test_http_gzip.py::test_wire_recorder_normal_record_keeps_022_shape",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
