"""A DeepInfra-shaped provider must admit a multi-send source-window run."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_request_seal_wire_form import _run_scripted, assert_sealed_wire


@pytest.mark.host_sandbox
def test_strict_provider_six_sends_seal_provider_bytes(tmp_path: Path) -> None:
    steps: list[dict[str, object]] = [
        {"tool": "read", "arguments": {"path": path}}
        for path in ("app.py", "tests/test_value.py", "app.py", "tests/test_value.py")
    ] + [{"tool": "run_tests", "arguments": {"target": "tests/test_value.py"}},
         {"tool": "done", "arguments": {"summary": "VALUE is seven and tests pass."}}]
    process, run, script = _run_scripted(tmp_path, strict=True, steps=steps)
    output = process.stdout + process.stderr
    assert len(script.requests) >= 6, output
    assert (run / "tagged-source-window").is_file(), output
    assert "provider dialect: openai-strict" in output
    assert "measurement body/projection/provider identity changed" not in output
    assert process.returncode == 0, output
    script.assert_consumed()
    assert not script.strict_openai_errors
    assert_sealed_wire(run, minimum=6)
