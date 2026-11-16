"""Guard: a subframe worker under the inherited working-set allocator sends owned source bytes.

Workers used to pin ``STATEFRAME_WORKING_SET_ALLOCATOR=off`` into their recorded environment,
so a parent running the allocator spawned subframes whose requests carried zero owned source.
The worker now inherits the parent's setting.  This test runs the shipped profile with
``--parallel 2``, has the one subframe read every page of its owned file, and checks the
request the worker actually SENT (the wire the scripted endpoint received).

Every ``[read page]`` receipt on the wire must carry exactly the on-disk bytes of the span it
declares (``effective=lines:S-E@SC:EC``).  The read tool may clip a scripted window and continue
it by cursor, so the owned windows are checked as coverage: each one must be fully covered by
verified wire spans.

The allocator is enabled on the PARENT only; nothing pins or patches the worker's setting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

_TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TEST_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_TEST_ROOT / "tests" / "fixtures" / "parallel"))

from barrier_endpoint import BarrierEndpoint
from e2e_support import create_engine_venv, run_cli, tool_step


pytestmark = pytest.mark.host_sandbox

PATH = "owned_source.py"
UNIT = "owned_reader"
LINE_COUNT = 2100
# The shipped allocator pages at 1,000 lines; three windows cover the 2,100-line file.
WINDOWS = ((1, 1000), (1001, 2000), (2001, 2100))
# Scripted reads are cut into chunks that fit one read-tool page without a char clip (a 1,000-line
# read clips near 11.5K chars and the continuation is then not scripted). Coverage still targets WINDOWS.
READ_CHUNK_LINES = 500

# Wire header from the read tool (kernel_source_delivery._header). The next line is the
# "more content" / "complete" line, then the body, whose length is exactly ``chars``.
_READ_PAGE = re.compile(
    r"^\[read page\] path=(?P<path>\S+) revision=(?P<revision>sha256:[0-9a-f]{64}) "
    r"lines=(?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+) requested=\S+ "
    r"effective=lines:(?P<eff_start>\d+)-(?P<eff_end>\d+)@(?P<eff_start_col>\d+):(?P<eff_end_col>\d+) "
    r"clip=\S+ chars=(?P<chars>\d+) bytes=\d+ truncated=(?P<truncated>true|false) "
    r"decoded_with_replacement=(?:true|false)$",
    re.MULTILINE,
)


@pytest.fixture(scope="session")
def isolated_engine_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return create_engine_venv(tmp_path_factory.mktemp("allocator-guard-engine") / "venv")


def _source() -> str:
    return "".join(f"VALUE_{line:04d} = {line}\n" for line in range(LINE_COUNT))


def _unit() -> dict[str, object]:
    return {
        "id": UNIT,
        "goal": f"Read {PATH} completely and report what it contains.",
        "owns": [PATH],
        "creates": [],
        "exports": [],
        "consumes": [],
        "reads": [],
        "requirements": [],
        "check": {"tests": []},
    }


def _child_steps() -> list[dict[str, object]]:
    chunks: list[tuple[int, int]] = []
    for window_start, window_end in WINDOWS:
        for chunk_start in range(window_start, window_end + 1, READ_CHUNK_LINES):
            chunks.append((chunk_start, min(chunk_start + READ_CHUNK_LINES - 1, window_end)))
    steps = [tool_step("read", {"path": PATH, "start_line": start, "end_line": end})
             for start, end in chunks]
    steps.append(tool_step("done", {"summary": "The owned source was read in full."},
                           required=False))
    return steps


def _offset(lines: list[str], line: int, column: int) -> int:
    """Character offset of 1-based ``line`` plus ``column`` in the file; clamps at end of file."""
    if line > len(lines):
        return sum(len(item) for item in lines)
    return sum(len(item) for item in lines[:line - 1]) + min(column, len(lines[line - 1]))


def _window_span(lines: list[str], window: tuple[int, int]) -> tuple[int, int]:
    start, end = window
    return _offset(lines, start, 0), _offset(lines, end + 1, 0)


def _covered(target: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    """True when the union of ``spans`` contains the whole ``target`` interval."""
    cursor, goal = target[0], target[1]
    for start, end in sorted(spans):
        if start > cursor:
            break
        cursor = max(cursor, end)
        if cursor >= goal:
            return True
    return cursor >= goal


def _read_page_receipts(request: dict[str, object], lines: list[str]) -> list[dict[str, object]]:
    """Every ``[read page]`` receipt for the owned path in one sent request, verified against disk.

    Each receipt's wire body must equal the disk bytes of its declared effective span exactly.
    """
    messages = request.get("messages")
    assert isinstance(messages, list), "sent request has no messages"
    file_text = "".join(lines)
    found: list[dict[str, object]] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        for match in _READ_PAGE.finditer(content):
            if match.group("path") != PATH:
                continue
            eff_start, eff_end = int(match.group("eff_start")), int(match.group("eff_end"))
            start_col, end_col = int(match.group("eff_start_col")), int(match.group("eff_end_col"))
            chars = int(match.group("chars"))
            label = f"[read page] effective lines {eff_start}@{start_col}-{eff_end}@{end_col}"
            assert chars > 0, f"zero-source: {label} carried no owned bytes"
            more_at = match.end() + 1
            assert content[match.end():more_at] == "\n", f"{label} header is not followed by its page line"
            body_start = content.find("\n", more_at)
            assert body_start >= 0, f"{label} has no body after its page line"
            body_start += 1
            body = content[body_start:body_start + chars]
            assert len(body) == chars, f"{label} body is shorter than its declared chars={chars}"
            start_offset = _offset(lines, eff_start, start_col)
            end_offset = _offset(lines, eff_end, end_col)
            expected = file_text[start_offset:end_offset]
            assert _digest(body) == _digest(expected), (
                f"{label} wire bytes differ from disk\n"
                f"wire sha256={_digest(body)} disk sha256={_digest(expected)}")
            found.append({
                "revision": match.group("revision"),
                "total": int(match.group("total")),
                "start_offset": start_offset,
                "end_offset": end_offset,
                "label": label,
            })
    return found


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_worker_under_inherited_allocator_sends_owned_page_bytes(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / PATH).write_text(_source(), encoding="utf-8")
    on_disk = (repo / PATH).read_bytes()
    disk_lines = on_disk.decode("utf-8").splitlines(keepends=True)
    revision = "sha256:" + hashlib.sha256(on_disk).hexdigest()
    run_dir = tmp_path / "run"

    parent = [tool_step("parallel", {"subframes": [_unit()]}),
              tool_step("done", {"summary": "Owned source inspected by one subframe."})]
    endpoint = BarrierEndpoint(parent, {UNIT: _child_steps()}, slots=2)
    with endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Read the owned source with one subframe.", parallel="2",
            arguments=("--parallel-min-subframes", "1",
                       "--parallel-max-owned-bytes", "100000"),
            # Parent-only switch: the worker must pick it up by inheritance, never by a pin.
            extra_environment={"STATEFRAME_WORKING_SET_ALLOCATOR": "on"}, timeout=600,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    # Weak sanity only. The worker may send more requests than scripted steps (retries, cursor
    # continuations), so the exact count is not asserted; the page checks below are the real guard.
    worker_requests = [item["request"] for item in endpoint.requests if item["marker"] == UNIT]
    assert worker_requests, "worker sent no requests to the scripted endpoint"

    # Every receipt that reached the wire carries exactly the on-disk bytes of its span.
    verified_spans: list[tuple[int, int]] = []
    receipt_labels: list[str] = []
    for request in worker_requests:
        for receipt in _read_page_receipts(request, disk_lines):
            assert receipt["revision"] == revision, (
                f"{receipt['label']} names revision {receipt['revision']}, disk is {revision}")
            assert receipt["total"] == LINE_COUNT, (
                f"{receipt['label']} names total={receipt['total']}, file has {LINE_COUNT} lines")
            verified_spans.append((receipt["start_offset"], receipt["end_offset"]))
            receipt_labels.append(receipt["label"])

    # Each owned window (1-1000, 1001-2000, 2001-2100) must be fully covered by verified wire bytes.
    missing = [window for window in WINDOWS
               if not _covered(_window_span(disk_lines, window), verified_spans)]
    assert not missing, (
        f"owned source pages missing or zero-source in the worker's sent requests: "
        f"missing {missing}, verified receipts={len(verified_spans)}, "
        f"sent requests={len(worker_requests)}, verified receipt labels={receipt_labels}")
    assert "".join(disk_lines) == _source()

    # The recorded worker environment inherited the parent's switch unchanged (no runner pin).
    specs = list((run_dir / "parallel").glob(f"*/units/{UNIT}/spec.json"))
    assert len(specs) == 1, specs
    spec = json.loads(specs[0].read_text(encoding="utf-8"))
    assert spec["environment"].get("STATEFRAME_WORKING_SET_ALLOCATOR") == "on"
    assert spec.get("environment_sources", {}).get("STATEFRAME_WORKING_SET_ALLOCATOR") != (
        "runner:unit-allocator-pin")
