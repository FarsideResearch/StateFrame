"""Frame fail-open, render overflow and replan sites report one reason line per cause per run."""
import io
import re
from types import SimpleNamespace

import pytest

import kernel_reason_log as rl
import kernel_stateframe_frame_append as frame_append
import kernel_stateframe_page_log as page_log
import kernel_stateframe_runtime as runtime
import kernel_stateframe_tool_stable as tool_stable
from kernel_context import RenderOverflow

LINE = re.compile(r"^\[stateframe\] (WARN|ERROR) (?P<component>[a-z_]+): .+", re.M)


@pytest.fixture(autouse=True)
def stream(monkeypatch):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    out = io.StringIO()
    rl.configure(level="WARN", stream=out)
    yield out
    rl.reset_for_tests()


def _components(out):
    return [m.group("component") for m in LINE.finditer(out.getvalue())]


class _FakePending:
    def __init__(self, **fields):
        self.__dict__.update(fields)


# (module, pending class name, pending attr on capture, component)
FAIL_OPEN_SITES = [
    (page_log, "PageLogPending", page_log._PENDING_ATTR, "stateframe_page_log"),
    (tool_stable, "ToolSurfacePending", tool_stable._PENDING_ATTR, "stateframe_tool_stable"),
    (frame_append, "PendingCommit", "_frame_append_pending", "stateframe_frame_append"),
]


# per module: (first reason, second reason). frame_append takes caller-supplied reasons, so its
# cause_key is one fixed value and the reason only rides in the line.
REASONS = {"page_log": ("concurrent_page_log_state_advance", "page_log_state_commit_failed"),
           "tool_stable": ("concurrent_tool_state_advance", "tool_state_commit_failed"),
           "frame_append": ("any_caller_reason", "another_caller_reason")}


@pytest.mark.parametrize("module, class_name, attr, component", FAIL_OPEN_SITES,
                         ids=["page_log", "tool_stable", "frame_append"])
def test_fail_open_pending_logs_once_per_reason(stream, monkeypatch, module, class_name, attr, component):
    monkeypatch.setattr(module, class_name, _FakePending)
    fields = dict(request_id="req-1", step=3, mode="on", menu_mode="off", rebase_mode="off")
    prefix = component.removeprefix("stateframe_")
    first, second = REASONS[prefix]

    def capture():
        return SimpleNamespace(**{attr: _FakePending(**fields)})

    module.fail_open_pending(capture(), None, reason=first)
    assert _components(stream) == [component]
    assert first in stream.getvalue()

    # Same reason again in this run: counted, not printed.
    module.fail_open_pending(capture(), None, reason=first)
    assert _components(stream) == [component]

    module.fail_open_pending(capture(), None, reason=second)
    by_cause = rl.run_summary()["by_cause"]
    if prefix == "frame_append":
        assert _components(stream) == [component]
        assert by_cause["fallback:frame_append_failed_open"]["count"] == 3
    else:
        # A different (known) reason is its own cause and prints its own line.
        assert _components(stream) == [component, component]
        assert by_cause[f"fallback:{prefix}:{first}"]["count"] == 2
        assert by_cause[f"fallback:{prefix}:{second}"]["count"] == 1


def test_fail_open_unknown_reason_collapses_to_other(stream, monkeypatch):
    monkeypatch.setattr(page_log, "PageLogPending", _FakePending)
    for reason in ("r1", "r2", "r3"):
        cap = SimpleNamespace(**{page_log._PENDING_ATTR: _FakePending(request_id="q", step=1, mode="on")})
        page_log.fail_open_pending(cap, None, reason=reason)
    assert list(rl.run_summary()["by_cause"]) == ["fallback:page_log:other"]


def test_fail_open_pending_without_pending_state_is_silent(stream):
    page_log.fail_open_pending(SimpleNamespace(), None, reason="page_log_state_commit_failed")
    tool_stable.fail_open_pending(SimpleNamespace(), None, reason="tool_state_commit_failed")
    frame_append.fail_open_pending(SimpleNamespace(), None, reason="x")
    assert stream.getvalue() == ""


def test_render_overflow_one_line_per_run_and_names_the_first_group(stream):
    overflow = RenderOverflow("f4:state", 100, 250, "frame/state", "stateframe_f4_ceiling")
    runtime._log_render_overflow(overflow, None)
    runtime._log_render_overflow(overflow, None)
    other = RenderOverflow("tool_schema", 50, 90, "tool_schema", "tool_schema")
    runtime._log_render_overflow(other, None)

    # cause_key is fixed (group ids can be per-item); the first group is named, repeats count.
    lines = [line for line in stream.getvalue().splitlines() if line.startswith("[stateframe]")]
    assert len(lines) == 1 and "f4:state" in lines[0]
    assert lines[0].startswith("[stateframe] WARN stateframe_runtime:")
    assert rl.run_summary()["by_cause"]["refusal:render_overflow"]["count"] == 3
