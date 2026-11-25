"""The shipped profile must continue to dispatch provider requests to child frames."""
from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from stateframe import profiles
from tests import test_strict_provider_mock_gate as strict_gate
from tests.engine_run import start_monitored
from tests.fixtures.parallel.e2e_support import install_startup_hook

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src" / "stateframe" / "profiles" / "r2hc3.json"
_WATCHDOG_SECONDS = 90
_STARTUP_GATE_SOURCE = """from __future__ import annotations
import os
import time
from pathlib import Path

gate = os.environ.pop("SF_LOADWIRE_FIXTURE_GATE", None)
if gate:
    Path(f"{gate}.started").write_text("started", encoding="ascii")
    deadline = time.monotonic() + 90
    ready = Path(f"{gate}.ready")
    while not ready.is_file():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fixture readiness gate was not released")
        time.sleep(min(0.01, remaining))
"""


def _wait_for_file(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))
    return True


def _write_fixture_slowly(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.provisioning")
    with temporary.open("xb") as output:
        for offset in range(0, len(content), 16):
            output.write(content[offset:offset + 16])
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _request_bodies(
        endpoint: strict_gate._StrictParallelEndpoint,
) -> tuple[dict[str, tuple[bytes, ...]], tuple[tuple[str, str], ...]]:
    by_marker: dict[str, list[bytes]] = {}
    receipt_digests: list[tuple[str, str]] = []
    for item in endpoint.requests:
        raw = item.get("raw_request")
        assert isinstance(raw, bytes), f"endpoint did not retain request bytes for {item['marker']}"
        request = json.loads(raw)
        for message in request.get("messages", []):
            content = message.get("content")
            if not isinstance(content, str) or not content.startswith("[StateFrame]\n"):
                continue
            frame = json.loads(content.split("\n", 1)[1])
            for digest in _receipt_digests(frame):
                # Receipts are run-scoped; compare every other provider byte exactly.
                receipt_digests.append((str(item["marker"]), digest))
                raw = raw.replace(digest.encode("ascii"), b"<run-receipt-digest>")
        by_marker.setdefault(str(item["marker"]), []).append(raw)
    return ({marker: tuple(bodies) for marker, bodies in by_marker.items()},
            tuple(receipt_digests))


def _receipt_digests(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        direct = value.get("receipt_digest")
        nested = tuple(digest for item in value.values() for digest in _receipt_digests(item))
        if (isinstance(direct, str) and len(direct) == 64
                and all(character in "0123456789abcdef" for character in direct)):
            return (direct, *nested)
        return nested
    if isinstance(value, list):
        return tuple(digest for item in value for digest in _receipt_digests(item))
    return ()


def _run_with_delayed_fixture(
        command: Sequence[str], *, cwd: str | Path, env: dict[str, str],
        expected_seconds: float, run_dir: Path, check: bool,
        capture_output: bool, text: bool,
        endpoint: strict_gate._StrictParallelEndpoint,
        released_at: dict[str, float],
) -> subprocess.CompletedProcess[str]:
    assert check is False and capture_output and text
    repo = Path(command[command.index("--repo") + 1])
    fixture_paths = tuple(repo / name for name in (
        "left.py", "right.py", "tests/test_left.py", "tests/test_right.py",
    ))
    fixture = {path: path.read_bytes() for path in fixture_paths}
    for path in fixture_paths:
        path.unlink()

    gate = run_dir.parent / "fixture-gate"
    child_environment = dict(env)
    child_environment["SF_LOADWIRE_FIXTURE_GATE"] = str(gate)
    install_startup_hook(Path(command[0]), "loadwire_fixture_gate", _STARTUP_GATE_SOURCE)

    with start_monitored(
            command, cwd=cwd, env=child_environment, expected_seconds=expected_seconds,
            run_dir=run_dir, step="forced subframe CLI run",
    ) as process:
        assert _wait_for_file(Path(f"{gate}.started"), _WATCHDOG_SECONDS), (
            "engine did not reach the delayed fixture gate")
        for path, content in fixture.items():
            _write_fixture_slowly(path, content)
        Path(f"{gate}.ready").write_text("ready", encoding="ascii")

        for marker in ("left", "right"):
            assert endpoint.wait_for_requests(marker, timeout=_WATCHDOG_SECONDS), (
                f"{marker} did not reach its first provider request")
        for marker in ("left", "right"):
            released_at[marker] = time.monotonic()
            endpoint.release_subframe(marker)
        for marker in ("left", "right"):
            assert endpoint.wait_for_requests(marker, count=2, timeout=_WATCHDOG_SECONDS), (
                f"{marker} did not send its read result")
        returncode = process.wait(timeout=expected_seconds)
        stdout, stderr = process.stdout_text, process.stderr_text

    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _profile_environment() -> dict[str, str]:
    environment = profiles.load(str(PROFILE))
    environment["STATEFRAME_WORKSPACE_OBJECTS"] = "on"
    return environment


def _assert_held_child_order(
        records: Sequence[dict[str, object]], released_at: dict[str, float],
) -> None:
    parent = [item for item in records if item["marker"] == "parent"]
    assert parent, "the parent sent no provider request"
    child_requests = {
        marker: [item for item in records if item["marker"] == marker]
        for marker in ("left", "right")
    }
    assert all(len(requests) >= 2 for requests in child_requests.values())
    first_starts = [requests[0].get("started") for requests in child_requests.values()]
    assert all(isinstance(started, float) for started in first_starts)
    assert set(released_at) == {"left", "right"}
    assert max(started for started in first_starts if isinstance(started, float)) <= min(
        released_at.values()), "a child response escaped before both requests reached the barrier"
    for marker, requests in child_requests.items():
        second_start = requests[1].get("started")
        assert isinstance(second_start, float)
        assert released_at[marker] <= second_start, (
            "a child sent its next request before its held response was released")


def test_child_order_accepts_request_after_response_before_endpoint_bookkeeping() -> None:
    records: tuple[dict[str, object], ...] = (
        {"marker": "parent", "started": 0.0, "finished": 1.0},
        {"marker": "left", "started": 1.0, "finished": 10.0},
        {"marker": "right", "started": 2.0, "finished": 11.0},
        {"marker": "left", "started": 4.0, "finished": 5.0},
        {"marker": "right", "started": 4.5, "finished": 6.0},
    )

    _assert_held_child_order(records, {"left": 3.0, "right": 3.5})


def _subframe_evidence(root: Path) -> tuple[tuple[object, ...], ...]:
    result_paths = tuple((root / "run" / "parallel").glob("*/result.json"))
    assert len(result_paths) == 1, result_paths
    outcomes = json.loads(result_paths[0].read_bytes())["outcomes"]
    evidence = []
    for outcome in outcomes:
        usage = outcome["usage"]
        files = tuple(sorted(
            (item["kind"], item["path"], item["added_lines"], item["removed_lines"])
            for item in outcome["files"]
        ))
        evidence.append((outcome["unit_id"], outcome["status"], outcome["check"],
                         outcome["contract_ok"], files, usage["requests"],
                         usage["physical_sends"]))
    return tuple(sorted(evidence))


@pytest.mark.host_sandbox
def test_shipped_profile_forced_subframes_send_child_requests(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old_umask = os.umask(0o077)
    try:
        normal_root = tmp_path / "normal"
        delayed_root = tmp_path / "delayed"
        normal_root.mkdir()
        delayed_root.mkdir()
        endpoints: list[strict_gate._StrictParallelEndpoint] = []

        class CapturingEndpoint(strict_gate._StrictParallelEndpoint):
            def __init__(self, parent_steps: list[dict[str, object]],
                         subframe_steps: dict[str, list[dict[str, object]]]) -> None:
                super().__init__(parent_steps, subframe_steps)
                endpoints.append(self)

        monkeypatch.setattr(strict_gate, "_StrictParallelEndpoint", CapturingEndpoint)
        strict_gate._run_forced_subframes(
            normal_root, profile_environment=_profile_environment())
        normal_endpoint = endpoints[-1]

        class DelayedEndpoint(CapturingEndpoint):
            def __init__(self, parent_steps: list[dict[str, object]],
                         subframe_steps: dict[str, list[dict[str, object]]]) -> None:
                super().__init__(parent_steps, subframe_steps)
                self.hold_subframe_request = {"left": 1, "right": 1}

        monkeypatch.setattr(strict_gate, "_StrictParallelEndpoint", DelayedEndpoint)
        released_at: dict[str, float] = {}

        def delayed_run(
                command: Sequence[str], *, cwd: str | Path, env: dict[str, str],
                expected_seconds: float, run_dir: Path, check: bool,
                capture_output: bool, text: bool,
        ) -> subprocess.CompletedProcess[str]:
            return _run_with_delayed_fixture(
                command, cwd=cwd, env=env, expected_seconds=expected_seconds,
                run_dir=run_dir, check=check, capture_output=capture_output, text=text,
                endpoint=endpoints[-1], released_at=released_at)

        monkeypatch.setattr(strict_gate, "run_monitored", delayed_run)
        strict_gate._run_forced_subframes(
            delayed_root, profile_environment=_profile_environment())
        delayed_endpoint = endpoints[-1]
    finally:
        os.umask(old_umask)

    _assert_held_child_order(delayed_endpoint.requests, released_at)
    expected_evidence = (
        ("left", "COMMITTED", "PASS", True, (("modify", "left.py", 1, 1),), 2, 2),
        ("right", "COMMITTED", "PASS", True, (("modify", "right.py", 1, 1),), 2, 2),
    )
    assert _subframe_evidence(normal_root) == _subframe_evidence(delayed_root) == expected_evidence
    normal_requests, normal_digests = _request_bodies(normal_endpoint)
    delayed_requests, delayed_digests = _request_bodies(delayed_endpoint)
    assert normal_requests == delayed_requests, (
        "provider requests changed while fixture readiness was delayed")
    assert len(normal_digests) == len(delayed_digests)
    assert tuple(marker for marker, _ in normal_digests) == tuple(
        marker for marker, _ in delayed_digests)
    differing_digests = [
        (left, right) for left, right in zip(normal_digests, delayed_digests, strict=True)
        if left[1] != right[1]
    ]
    assert len(differing_digests) <= 1 and all(
        left[0] == right[0] == "parent" for left, right in differing_digests
    ), f"unexpected run-scoped receipt differences: {differing_digests!r}"
