"""FU-070 checkpoint 1: document the allocator/subframe source-view failure."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest

_TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TEST_ROOT / "tests" / "fixtures"))
sys.path.insert(0, str(_TEST_ROOT / "tests" / "fixtures" / "parallel"))

from _journal import measurement_events
from barrier_endpoint import BarrierEndpoint
from e2e_support import (
    create_engine_venv, install_startup_hook, run_cli, subframe_directories, tool_step,
)


pytestmark = pytest.mark.host_sandbox


@pytest.fixture(scope="session")
def isolated_engine_python(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Reuse the existing real-child venv builder from the parallel E2E harness."""
    return create_engine_venv(tmp_path_factory.mktemp("fu070-parallel-engine") / "venv")

_ALLOCATOR_TEST_HOOK = '''
"""Test-only seam that re-enables the worker allocator after worker prep."""
import json
import os

if os.environ.get("FU070_ALLOCATOR_TEST") == "on":
    from stateframe import parallel_worker
    _prepare_worker = parallel_worker.prepare_worker

    def _prepare_with_allocator(*args, **kwargs):
        result = _prepare_worker(*args, **kwargs)
        # prepare_worker(spec, spec_path, run_dir, ...) records the worker spec at run_dir/spec.json.
        spec_path = args[2] / "spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["environment"]["STATEFRAME_WORKING_SET_ALLOCATOR"] = "on"
        spec["environment_sources"]["STATEFRAME_WORKING_SET_ALLOCATOR"] = "test:fu070-alloc"
        spec_path.write_text(json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\\n",
                             encoding="utf-8")
        return result

    parallel_worker.prepare_worker = _prepare_with_allocator
'''


def _unit(path: str) -> dict[str, object]:
    return {
        "id": "large_source",
        "goal": f"Inspect {path} and report the source context.",
        "owns": [path],
        "creates": [],
        "exports": [],
        "consumes": [],
        "reads": [],
        "requirements": [],
        "check": {"tests": []},
    }


def _prepare(tmp_path: Path, isolated_engine_python: Path, *, line_count: int = 2100,
             child_steps: list[dict[str, object]] | None = None,
             ) -> tuple[Path, BarrierEndpoint, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    path = "owned_source.py"
    # The checked-in allocator uses 1,000-line / 64,000-character page ceilings.
    # 2,100 short lines cross that line boundary twice (3 pages) while the whole file plus its
    # serialization overhead stays under the child byte cap, so a complete FILE view is feasible.
    source = "".join(f"VALUE_{line:04d} = {line}\n" for line in range(line_count))
    (repo / path).write_text(source, encoding="utf-8")
    run_dir = tmp_path / "run"
    parent = [tool_step("parallel", {"subframes": [_unit(path)]}),
              tool_step("done", {"summary": "Source view checkpoint recorded."})]
    if child_steps is None:
        child_steps = [tool_step("done", {"summary": "The full source was available."},
                                 required=False)]
    endpoint = BarrierEndpoint(parent, {"large_source": child_steps}, slots=2)
    install_startup_hook(isolated_engine_python, "fu070_allocator_test_hook", _ALLOCATOR_TEST_HOOK)
    return repo, endpoint, run_dir


def _full_read_steps() -> list[dict[str, object]]:
    """Read owned_source.py in 1,000-line windows until the 2,100-line file is covered."""
    windows = ((1, 1000), (1001, 2000), (2001, 2100))
    steps = [tool_step("read", {"path": "owned_source.py", "start_line": start, "end_line": end})
             for start, end in windows]
    steps.append(tool_step("done", {"summary": "The full source was read."}, required=False))
    return steps


def _run(tmp_path: Path, isolated_engine_python: Path, *,
         child_steps: list[dict[str, object]] | None = None):
    repo, endpoint, run_dir = _prepare(tmp_path, isolated_engine_python, child_steps=child_steps)
    with endpoint:
        completed = run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Inspect the owned source with one subframe.", parallel="2",
            arguments=("--parallel-min-subframes", "1",
                       "--parallel-max-owned-bytes", "100000"),
            extra_environment={"FU070_ALLOCATOR_TEST": "on"}, timeout=600,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    specs = list((run_dir / "parallel").glob("*/units/large_source/spec.json"))
    assert len(specs) == 1
    assert json.loads(specs[0].read_text(encoding="utf-8"))["environment"][
        "STATEFRAME_WORKING_SET_ALLOCATOR"] == "on"
    return run_dir, endpoint


def _refused_request(run_dir: Path) -> tuple[Path, dict[str, object]]:
    matches = list((run_dir / "parallel").glob("*/units/large_source/unit-refused-request.json"))
    assert len(matches) == 1, f"expected one refused subframe request, found {matches}"
    path = matches[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    return path, value


def _page_count(request: dict[str, object]) -> int:
    messages = request.get("messages")
    assert isinstance(messages, list)
    text = "\n".join(message.get("content", "") for message in messages
                     if isinstance(message, dict) and isinstance(message.get("content"), str))
    return len(re.findall(r"(?m)^\[read page\] path=owned_source\.py\b", text))


def _deferred_paths(request: dict[str, object]) -> list[str]:
    messages = request.get("messages")
    assert isinstance(messages, list)
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.startswith("[Deferred required source paths]"):
            return [line[2:] for line in content.splitlines()[1:] if line.startswith("- ")]
    return []


def _read_then_edit_steps() -> list[dict[str, object]]:
    """Read the 3 windows, then make one admitted edit (the ownerless early-done gate needs a changed file)."""
    edit = {"path": "owned_source.py", "old_str": "VALUE_0000 = 0",
            "new_str": "VALUE_0000 = 100"}
    return [*_full_read_steps()[:-1], tool_step("str_replace", edit),
            tool_step("done", {"summary": "The full source was read and edited."}, required=False)]


def test_deferred_first_request_is_sent_and_reads_fill_pages(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    run_dir, endpoint = _run(tmp_path, isolated_engine_python, child_steps=_read_then_edit_steps())
    assert any(item["marker"] == "large_source" for item in endpoint.requests)
    assert not list((run_dir / "parallel").glob("*/units/large_source/unit-refused-request.json"))
    results = list((run_dir / "parallel").glob("*/units/large_source/engine-result.json"))
    assert len(results) == 1
    result = json.loads(results[0].read_text(encoding="utf-8"))["result"]
    assert result["result"] != "SUBFRAME_CONTEXT_TOO_LARGE"
    # The reads must really complete: every scripted child step reaches the endpoint and
    # the child never blocks on an infeasible working-set floor.
    assert len([item for item in endpoint.requests if item["marker"] == "large_source"]) == 5
    assert "WORKING_SET_ALLOCATION_BLOCKED" not in json.dumps(result)
    assert result["result"] != "BLOCKED"
    subframe = subframe_directories(run_dir)["large_source"]
    refusals = [event for event in measurement_events(subframe)
                if event.get("kind") == "edit_admission_refused"]
    assert refusals == [], refusals
    source = "".join(f"VALUE_{line:04d} = {line}\n" for line in range(2100))
    expected = source.replace("VALUE_0000 = 0\n", "VALUE_0000 = 100\n", 1)
    assert (run_dir / "workspace/owned_source.py").read_text(encoding="utf-8") == expected


def test_edit_admitted_only_when_view_complete() -> None:
    import hashlib

    from stateframe.engine.kernel_edit_admission import admit_edit, issue_handles
    from stateframe.engine.kernel_source_view import SourceView, VerifiedSpan

    def sha(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    # Three pages of two lines each; page 2 is the one missing from the incomplete view.
    page_1, page_2, page_3 = b"one\ntwo\n", b"three\nfour\n", b"five\nsix\n"
    source = page_1 + page_2 + page_3
    span_1 = VerifiedSpan(1, 2, sha(page_1))
    span_2 = VerifiedSpan(3, 4, sha(page_2))
    span_3 = VerifiedSpan(5, 6, sha(page_3))
    incomplete = SourceView("owned_source.py", sha(source), 6, True, (span_1, span_3),
                            complete=False)
    complete = SourceView("owned_source.py", sha(source), 6, True, (span_1, span_2, span_3),
                          complete=True)

    assert issue_handles(incomplete, request_id="req-1") is None
    handle = issue_handles(complete, request_id="req-1")
    assert handle is not None and handle.mode == "FILE"
    admit_edit(handle, current_source=source, touched_lines=(3, 4))


def test_stale_revision_handle_refused_after_source_change() -> None:
    import hashlib

    from stateframe.engine.kernel_edit_admission import EditNotAdmitted, admit_edit, issue_handles
    from stateframe.engine.kernel_source_view import SourceView, VerifiedSpan

    def sha(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    source = b"a\nb\nc\n"
    view = SourceView("owned_source.py", sha(source), 3, True,
                      (VerifiedSpan(1, 3, sha(source)),), complete=True)
    handle = issue_handles(view, request_id="req-1")
    assert handle is not None
    with pytest.raises(EditNotAdmitted) as raised:
        admit_edit(handle, current_source=b"a\nB\nc\n", touched_lines=(2, 2))
    assert raised.value.reason == "stale_revision"


def test_edit_outside_verified_span_is_not_admitted() -> None:
    import hashlib

    from stateframe.engine.kernel_edit_admission import EditHandle, EditNotAdmitted, admit_edit
    from stateframe.engine.kernel_source_view import VerifiedSpan

    def sha(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    # Span 1-3 is verified against the current bytes; line 4 exists but is outside it.
    source = b"a\nb\nc\nd\ne\n"
    handle = EditHandle("request", "owned_source.py", sha(source),
                        (VerifiedSpan(1, 3, sha(b"a\nb\nc\n")),), "SPAN")
    with pytest.raises(EditNotAdmitted) as raised:
        admit_edit(handle, current_source=source, touched_lines=(4, 4))
    assert raised.value.reason == "outside_span"


def test_unadmitted_write_expires_handle() -> None:
    import hashlib

    from stateframe.engine.kernel_edit_admission import (
        EditNotAdmitted,
        admit_edit,
        expire_handles,
        issue_handles,
    )
    from stateframe.engine.kernel_source_view import SourceView, VerifiedSpan

    def sha(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def complete_view(path: str, source: bytes) -> SourceView:
        line_count = source.count(b"\n")
        return SourceView(path, sha(source), line_count, True,
                          (VerifiedSpan(1, line_count, sha(source)),), complete=True)

    a_source, b_source = b"a1\na2\n", b"b1\nb2\n"
    handle_a = issue_handles(complete_view("a.py", a_source), request_id="req-1")
    handle_b = issue_handles(complete_view("b.py", b_source), request_id="req-1")
    assert handle_a is not None and handle_b is not None

    # An unadmitted write to a.py drops its authority; b.py keeps its handle in order.
    remaining = expire_handles((handle_a, handle_b), touched_paths=["a.py"])
    assert remaining == (handle_b,)

    # Even a retained copy of the old a.py handle is refused once the bytes have changed.
    with pytest.raises(EditNotAdmitted) as raised:
        admit_edit(handle_a, current_source=b"a1\nchanged\n", touched_lines=(1, 1))
    assert raised.value.reason == "stale_revision"
    admit_edit(remaining[0], current_source=b_source, touched_lines=(1, 1))


def test_oversized_owned_file_refused_with_typed_capacity_and_zero_calls(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    # 50,000 lines (just under the 1 MiB work-copy file cap) cannot fit a 32k child context;
    # the owned-bytes ceiling is raised so the refusal under test is the child capacity
    # floor, not the admission byte cap. 16k is below what the parent run itself needs.
    repo, endpoint, run_dir = _prepare(tmp_path, isolated_engine_python, line_count=50_000)
    with endpoint:
        run_cli(
            isolated_engine_python, endpoint.url, repo, run_dir,
            "Inspect the owned source with one subframe.", parallel="2",
            arguments=("--parallel-min-subframes", "1",
                       "--parallel-max-owned-bytes", "10000000"),
            extra_environment={"FU070_ALLOCATOR_TEST": "on",
                               "STATEFRAME_CONTEXT_TOKENS": "32768"},
            timeout=600,
        )
    assert not any(item["marker"] == "large_source" for item in endpoint.requests)
    unit_files = list((run_dir / "parallel").glob("*/units/large_source/*.json"))
    assert unit_files, "expected typed evidence for the refused subframe"
    evidence = "\n".join(path.read_text(encoding="utf-8") for path in unit_files)
    assert "needs_source_capacity" in evidence or "CONTEXT_TOO_SMALL" in evidence
    assert "floor" in evidence and "band" in evidence


def _edit_first_then_read_steps() -> list[dict[str, object]]:
    """Edit first (the full-file first request admits it), read the 3 windows, then repeat the same edit.

    The repeat must fail normally because old_str no longer matches; its tool error is expected.
    """
    edit = {"path": "owned_source.py", "old_str": "VALUE_0000 = 0",
            "new_str": "VALUE_0000 = 100"}
    return [tool_step("str_replace", edit), *_full_read_steps()[:-1],
            tool_step("str_replace", edit),
            tool_step("done", {"summary": "The edit landed on the full-file first request."},
                      required=False)]


@pytest.mark.xfail(strict=True, reason="FU-070-ALLOC: post-edit child request fails frame-only preflight seal (crew-01)")
def test_edit_admitted_when_first_request_carries_full_file(
        tmp_path: Path, isolated_engine_python: Path) -> None:
    run_dir, endpoint = _run(tmp_path, isolated_engine_python,
                             child_steps=_edit_first_then_read_steps())
    subframe = subframe_directories(run_dir)["large_source"]
    refusals = [event for event in measurement_events(subframe)
                if event.get("kind") == "edit_admission_refused"]
    assert refusals == [], refusals
    source = "".join(f"VALUE_{line:04d} = {line}\n" for line in range(2100))
    expected = source.replace("VALUE_0000 = 0\n", "VALUE_0000 = 100\n", 1)
    assert (run_dir / "workspace/owned_source.py").read_text(encoding="utf-8") == expected
    # No replies or journal entries may carry the unverified-source refusal.
    child_requests = [item for item in endpoint.requests if item["marker"] == "large_source"]
    assert "SOURCE_NOT_VERIFIED" not in json.dumps(child_requests, default=str)
    for journal_file in subframe.rglob("*"):
        if journal_file.is_file():
            text = journal_file.read_text(encoding="utf-8", errors="replace")
            assert "SOURCE_NOT_VERIFIED" not in text, journal_file
