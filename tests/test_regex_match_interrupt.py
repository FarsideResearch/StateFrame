"""Regression coverage for bounded regex matching in line based searches."""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import kernel_navigation_pages as navigation


_ROOT = Path(__file__).resolve().parents[1]
_ENGINE = _ROOT / "src" / "stateframe" / "engine"


def _run_guarded(script: str, *arguments: str, timeout: float = 5) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (_ENGINE.as_posix(), env.get("PYTHONPATH", "")) if part
    )
    env["TMPDIR"] = str(_ROOT / ".tmp")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    args = [sys.executable, "-c", script, *arguments]
    try:
        result = subprocess.run(args, cwd=_ROOT, env=env, capture_output=True, text=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail("guarded regex subprocess did not finish")
    assert result.returncode == 0, result.stderr
    return result


def _require_free_timer() -> tuple[int, int]:
    if not all(hasattr(signal, name) for name in ("SIGVTALRM", "ITIMER_VIRTUAL", "getitimer", "setitimer")):
        pytest.skip("CPU interval timers are unavailable")
    signum, timer = signal.SIGVTALRM, signal.ITIMER_VIRTUAL
    if signal.getsignal(signum) is not signal.SIG_DFL or any(signal.getitimer(timer)):
        pytest.skip("the test process already owns the CPU timer")
    return signum, timer


@pytest.mark.parametrize("pinned", (False, True), ids=("descriptor-walk", "pinned"))
def test_costly_line_is_undecided_and_search_continues(tmp_path: Path, pinned: bool) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    script = """
import os, sys
from pathlib import Path
import kernel_navigation_pages as navigation
navigation._LINE_CPU_LIMIT_SECONDS = 0.05
navigation._LINE_TIMER_TICK_SECONDS = 0.01
root = Path(sys.argv[1])
(root / 'a-costly.txt').write_text('a' * 50 + '!\\nOK on same file\\n')
(root / 'z-later.txt').write_text('OK in later file\\n')
if sys.argv[2] == 'pinned':
    from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
    from kernel_workspace_state import WorkspaceFence, WorkspaceState
    os.environ['STATEFRAME_WORKSPACE_OBJECTS'] = 'on'
    consumers = WorkspaceConsumers(WorkspaceState(WorkspaceFence.create(root, root.parent / 'fence')))
    bind_consumers(root, consumers)
else:
    os.environ['STATEFRAME_WORKSPACE_OBJECTS'] = 'off'
page = navigation.search_repository(root, r'(a+)+$|^OK')
print(page.text)
print(f"total={page.metadata['total']} complete={page.metadata['scan_complete']}")
empty = navigation.search_repository(root, r'(a+)+$')
print('NO-HITS-BEGIN')
print(empty.text)
print('NO-HITS-END')
"""
    result = _run_guarded(script, str(root), "pinned" if pinned else "descriptor")
    assert "1 line not decided" in result.stdout
    assert "a-costly.txt:2:OK on same file" in result.stdout
    assert "z-later.txt:1:OK in later file" in result.stdout
    assert "total=2 complete=False" in result.stdout
    no_hits = result.stdout.split("NO-HITS-BEGIN\n", 1)[1].split("\nNO-HITS-END", 1)[0]
    assert "1 line not decided" in no_hits
    assert "No hits" not in no_hits


@pytest.mark.parametrize("pinned", (False, True), ids=("descriptor-walk", "pinned"))
def test_many_slow_lines_share_one_call_budget(tmp_path: Path, pinned: bool) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    script = """
import os, sys, time
from pathlib import Path
import kernel_navigation_pages as navigation
navigation._CALL_CPU_LIMIT_SECONDS = 0.3
navigation._LINE_TIMER_TICK_SECONDS = 0.01
root = Path(sys.argv[1])
(root / 'a-fast.txt').write_text('OK before\\n')
(root / 'b-slow.txt').write_text(('a' * 21 + '!\\n') * 200)
(root / 'c-later.txt').write_text('OK after\\n')
if sys.argv[2] == 'pinned':
    from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
    from kernel_workspace_state import WorkspaceFence, WorkspaceState
    os.environ['STATEFRAME_WORKSPACE_OBJECTS'] = 'on'
    consumers = WorkspaceConsumers(WorkspaceState(WorkspaceFence.create(root, root.parent / 'fence')))
    bind_consumers(root, consumers)
else:
    os.environ['STATEFRAME_WORKSPACE_OBJECTS'] = 'off'
started = time.thread_time()
page = navigation.search_repository(root, r'(a+)+$|^OK')
print('cpu', time.thread_time() - started)
print(page.text)
print(sorted((k, v) for k, v in page.metadata.items() if k in (
    'total', 'scan_complete', 'coverage_limited', 'unsearched_files')))
clean = navigation.search_repository(root, '^OK')
print('CLEAN', clean.metadata['total'], clean.metadata['scan_complete'], 'coverage_limited' in clean.metadata)
"""
    out = _run_guarded(script, str(root), "pinned" if pinned else "descriptor", timeout=30).stdout
    assert "a-fast.txt:1:OK before" in out
    assert "c-later.txt" not in out
    assert "lines not decided." in out and "1 file not searched." in out
    assert "[('coverage_limited', True), ('scan_complete', False), ('total', 1), ('unsearched_files', 1)]" in out
    assert "CLEAN 2 True False" in out
    assert float(out.split()[1]) < 3


def test_unit_policy_call_budget_reports_files_not_searched(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    script = """
import sys
from pathlib import Path
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
import kernel_navigation_pages as navigation
from kernel_unit_policy import UnitPolicy
navigation._CALL_CPU_LIMIT_SECONDS = 0.3
navigation._LINE_TIMER_TICK_SECONDS = 0.01
root = Path(sys.argv[1])
names = ('src/a.py', 'src/b.py')
recipe = CheckRecipe((), (), 'none')
contract = UnitContract('a', 'fix a', names, (), (), (), (), (), recipe)
spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (), names, (), '', 0, 0, 'manifest', 'policy', ParallelLimits())
(root / 'src/a.py').write_text(('a' * 21 + '!\\n') * 200)
(root / 'src/b.py').write_text('OK\\n')
page = UnitPolicy(spec).search_visible(root, r'(a+)+$|^OK$')
print(page.text)
print(page.metadata['scan_complete'], page.metadata['coverage_limited'], page.metadata['unsearched_files'])
"""
    out = _run_guarded(script, str(root), timeout=30).stdout
    assert "lines not decided." in out and "1 file not searched." in out
    assert "False True 1" in out


@pytest.mark.parametrize("hit", (False, True))
def test_spent_budget_without_undecided_lines_reports_files_not_searched(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hit: bool) -> None:
    from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
    from kernel_unit_policy import UnitPolicy

    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    (tmp_path / "a.py").write_text("OK\n" if hit else "no\n")
    (tmp_path / "b.py").write_text("OK\n")
    matching = navigation._matching_lines

    def spend(lines, *args):
        yield from matching(lines, *args)
        navigation._LINE_SCOPE.state["spent"] = True

    monkeypatch.setattr(navigation, "_matching_lines", spend)
    monkeypatch.setitem(UnitPolicy.search_visible.__wrapped__.__globals__, "_matching_lines", spend)
    names = ("a.py", "b.py")
    contract = UnitContract("a", "fix a", names, (), (), (), (), (), CheckRecipe((), (), "none"))
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (), names, (), "", 0, 0,
                    "manifest", "policy", ParallelLimits())
    for page in (navigation.search_repository(tmp_path, "^OK", mode="regex"),
                 UnitPolicy(spec).search_visible(tmp_path, "^OK")):
        assert "1 file not searched.\nSlow-pattern CPU limit reached (30 s in this call)" in page.text
        assert "not decided" not in page.text
        assert "No hits" not in page.text and "no matches" not in page.text
        assert ("a.py:1:OK" in page.text) is hit
        assert page.metadata["scan_complete"] is False
        assert page.metadata["coverage_limited"] is True
        assert page.metadata["unsearched_files"] == 1
        assert page.metadata["total"] == int(hit)


def test_pinned_result_with_files_not_searched_is_not_cached(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
    from kernel_workspace_state import WorkspaceFence, WorkspaceState

    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text("OK\n")
    (root / "b.py").write_text("OK\n")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "on")
    bind_consumers(root, WorkspaceConsumers(WorkspaceState(
        WorkspaceFence.create(root, tmp_path / "fence"))))
    matching = navigation._matching_lines

    def spend(lines, *args):
        yield from matching(lines, *args)
        navigation._LINE_SCOPE.state["spent"] = True

    with monkeypatch.context() as patch:
        patch.setattr(navigation, "_matching_lines", spend)
        cut = navigation.search_repository(root, "^OK", mode="regex")
    assert (cut.metadata["total"], cut.metadata["unsearched_files"]) == (1, 1)
    full = navigation.search_repository(root, "^OK", mode="regex")
    assert full.metadata["total"] == 2 and full.metadata["scan_complete"] is True


def test_partial_pinned_file_and_result_are_not_cached(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    script = """
import os, sys
from pathlib import Path
import kernel_navigation_pages as navigation
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_state import WorkspaceFence, WorkspaceState
navigation._LINE_CPU_LIMIT_SECONDS = 0.05
navigation._LINE_TIMER_TICK_SECONDS = 0.01
root = Path(sys.argv[1])
(root / 'sample.txt').write_text('a' * 50 + '!\\nOK\\n')
os.environ['STATEFRAME_WORKSPACE_OBJECTS'] = 'on'
consumers = WorkspaceConsumers(WorkspaceState(WorkspaceFence.create(root, root.parent / 'fence')))
bind_consumers(root, consumers)
original = navigation._read_text
reads = {'count': 0}
def read(*args, **kwargs):
    reads['count'] += 1
    return original(*args, **kwargs)
navigation._read_text = read
for _ in range(2):
    page = navigation.search_repository(root, r'(a+)+$|^OK$')
cache = vars(consumers)['_navigation_search_cache']
print(page.text)
print(f'reads={reads["count"]} file_cache={len(cache.files.entries)} result_cache={len(cache.results.entries)}')
"""
    result = _run_guarded(script, str(root))
    assert "1 line not decided" in result.stdout
    assert "sample.txt:2:OK" in result.stdout
    assert "reads=2 file_cache=0 result_cache=0" in result.stdout


def test_unit_policy_reports_undecided_line_and_keeps_later_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    script = """
import sys
from pathlib import Path
from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
import kernel_navigation_pages as navigation
from kernel_unit_policy import UnitPolicy
navigation._LINE_CPU_LIMIT_SECONDS = 0.05
navigation._LINE_TIMER_TICK_SECONDS = 0.01
root = Path(sys.argv[1])
recipe = CheckRecipe((), (), 'none')
contract = UnitContract('a', 'fix a', ('src/sample.py',), (), (), (), (), (), recipe)
spec = UnitSpec('call', 'a', 'execution', 'seed', contract, (), (), ('src/sample.py',), (), '', 0, 0, 'manifest', 'policy', ParallelLimits())
(root / 'src/sample.py').write_text('a' * 50 + '!\\nOK\\n')
page = UnitPolicy(spec).search_visible(root, r'(a+)+$|^OK$')
print(page.text)
print(f"complete={page.metadata['scan_complete']}")
empty = UnitPolicy(spec).search_visible(root, r'(a+)+$')
print(empty.text)
"""
    result = _run_guarded(script, str(root))
    assert "src/sample.py:2:OK" in result.stdout
    assert "1 line not decided" in result.stdout
    assert "complete=False" in result.stdout
    assert "(no matches in visible source)" not in result.stdout


def test_import_and_regex_search_when_virtual_timer_api_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "sample.txt").write_text("OK\n")
    script = """
import signal, sys
from pathlib import Path
for name in ('setitimer', 'ITIMER_VIRTUAL'):
    if hasattr(signal, name): delattr(signal, name)
import kernel_navigation_pages as navigation
page = navigation.search_repository(Path(sys.argv[1]), '^OK$')
print(page.text)
"""
    result = _run_guarded(script, str(root))
    assert "sample.txt:1:OK" in result.stdout


def test_off_main_thread_search_does_not_arm_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_free_timer()
    monkeypatch.setattr(navigation, "_LINE_CPU_LIMIT_SECONDS", 0.001)
    monkeypatch.setattr(signal, "setitimer", lambda *_args: pytest.fail("worker armed process timer"))

    class SlowMatcher:
        def search(self, _line: str) -> bool:
            end = time.thread_time() + 0.03
            while time.thread_time() < end:
                pass
            return True

    @navigation._line_match_session
    def search() -> list[tuple[int, str]]:
        return list(navigation._matching_lines(["line"], SlowMatcher(), None))

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(search).result(timeout=3) == [(1, "line")]


def test_armed_loop_does_not_call_thread_time_or_copy_first_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, object] = {"armed": True, "flag": [0], "last_tick": None}
    calls = 0

    def thread_time() -> float:
        nonlocal calls
        calls += 1
        return 0.0

    class NoSlices(list[str]):
        def __getitem__(self, index):
            assert not isinstance(index, slice), "the first pass must not copy the lines"
            return super().__getitem__(index)

    monkeypatch.setattr(navigation, "_THREAD_TIME", thread_time)
    navigation._LINE_SCOPE.state = state
    try:
        assert list(navigation._matching_lines(NoSlices(["skip", "hit"]), re.compile("hit"), [0])) == [
            (2, "hit")
        ]
        assert calls == 0
        assert state["flag"] == [0]
    finally:
        del navigation._LINE_SCOPE.state


@pytest.mark.parametrize("current_match", (False, True))
def test_tick_between_search_and_yield_preserves_line_accounting(
        monkeypatch: pytest.MonkeyPatch, current_match: bool) -> None:
    state: dict[str, object] = {"armed": True, "flag": [0], "last_tick": None}
    cpu = [0.0]
    monkeypatch.setattr(navigation, "_LINE_CPU_LIMIT_SECONDS", 0.05)
    monkeypatch.setattr(navigation, "_THREAD_TIME", lambda: cpu[0])
    navigation._LINE_SCOPE.state = state

    class DelayedResult:
        checks = 0

        def __bool__(self) -> bool:
            self.checks += 1
            if self.checks == 1:
                navigation._line_tick(0, None)
            return current_match

    class Matcher:
        def search(self, line: str) -> bool | DelayedResult:
            if line == "slow":
                navigation._line_tick(0, None)
                cpu[0] = 0.06
                return DelayedResult()
            return line == "after"

    undecided = [0]
    try:
        assert list(navigation._matching_lines(["slow", "after"], Matcher(), undecided)) == (
            [(1, "slow"), (2, "after")] if current_match else [(2, "after")]
        )
        assert undecided == [0 if current_match else 1]
    finally:
        del navigation._LINE_SCOPE.state


def test_equal_line_numbers_in_distinct_file_passes_reset_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    state: dict[str, object] = {"armed": True, "flag": [0], "last_tick": None}
    cpu = [0.0]
    first_token: list[object] = []
    monkeypatch.setattr(navigation, "_THREAD_TIME", lambda: cpu[0])
    navigation._LINE_SCOPE.state = state

    class TickingMatcher:
        def search(self, line: str) -> bool:
            navigation._line_tick(0, None)
            if line == "first":
                first_token.append(state["last_tick"][0])
                cpu[0] = 100.0
                return False
            return True

    try:
        assert list(navigation._matching_lines(["first"], TickingMatcher(), [0])) == []
        assert list(navigation._matching_lines(["second"], TickingMatcher(), [0])) == [(1, "second")]
        assert state["last_tick"][0] is not first_token[0]
    finally:
        del navigation._LINE_SCOPE.state


def test_yielded_match_does_not_time_consumer_work(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_free_timer()
    monkeypatch.setattr(navigation, "_LINE_CPU_LIMIT_SECONDS", 0.02)
    monkeypatch.setattr(navigation, "_LINE_TIMER_TICK_SECONDS", 0.005)

    @navigation._line_match_session
    def consume() -> tuple[int, str]:
        undecided = [0]
        lines = navigation._matching_lines(["match"], re.compile("^match$"), undecided)
        result = next(lines)
        assert navigation._LINE_SCOPE.state["flag"] == [0]
        end = time.thread_time() + 0.08
        while time.thread_time() < end:
            pass
        assert navigation._LINE_SCOPE.state["flag"] == [0]
        assert undecided == [0]
        with pytest.raises(StopIteration):
            next(lines)
        return result

    assert consume() == (1, "match")


@pytest.mark.parametrize("error", (ValueError, KeyboardInterrupt))
def test_matching_flag_clears_after_exception(error: type[BaseException]) -> None:
    _require_free_timer()

    class Raises:
        def search(self, _line: str) -> None:
            assert navigation._LINE_SCOPE.state["flag"][0] == 1
            raise error("matcher failed")

    @navigation._line_match_session
    def search() -> None:
        try:
            list(navigation._matching_lines(["line"], Raises(), [0]))
        except BaseException:
            assert navigation._LINE_SCOPE.state["flag"] == [0]
            raise

    with pytest.raises(error, match="matcher failed"):
        search()
    assert signal.getsignal(signal.SIGVTALRM) is signal.SIG_DFL
    assert signal.getitimer(signal.ITIMER_VIRTUAL) == (0.0, 0.0)


def test_late_pending_tick_after_timer_stop_does_not_kill_child() -> None:
    if (os.name != "posix" or not hasattr(signal, "pthread_sigmask")
            or not all(hasattr(signal, name) for name in ("SIGVTALRM", "ITIMER_VIRTUAL"))):
        pytest.skip("pending-signal masking is unavailable")
    script = """
import os, signal
import kernel_navigation_pages as navigation
original = signal.setitimer
def stop_then_queue(which, seconds, interval=0.0):
    previous = original(which, seconds, interval)
    if which == signal.ITIMER_VIRTUAL and seconds == 0:
        blocked = {signal.SIGVTALRM}
        signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        os.kill(os.getpid(), signal.SIGVTALRM)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, blocked)
    return previous
signal.setitimer = stop_then_queue
@navigation._line_match_session
def search(): return 'finished'
assert search() == 'finished'
assert signal.getsignal(signal.SIGVTALRM) is signal.SIG_DFL
print('survived late tick')
"""
    result = _run_guarded(script)
    assert "survived late tick" in result.stdout


@pytest.mark.parametrize("occupied", ("handler", "timer"))
def test_existing_handler_or_timer_is_left_alone(occupied: str, tmp_path: Path) -> None:
    signum, timer = _require_free_timer()
    (tmp_path / "sample.txt").write_text("not a match\n")
    before_handler, before_timer = signal.getsignal(signum), signal.getitimer(timer)

    def occupied_handler(*_args: object) -> None:
        pass

    try:
        if occupied == "handler":
            signal.signal(signum, occupied_handler)
        else:
            signal.setitimer(timer, 60)
        active_handler, active_timer = signal.getsignal(signum), signal.getitimer(timer)
        assert navigation.search_repository(tmp_path, "^never-match-this$").metadata["scan_complete"]
        assert signal.getsignal(signum) is active_handler
        after_timer = signal.getitimer(timer)
        if occupied == "timer":
            assert after_timer[0] > 0 and after_timer[1] == active_timer[1]
        else:
            assert after_timer == active_timer
    finally:
        signal.setitimer(timer, 0)
        signal.signal(signum, before_handler)
        signal.setitimer(timer, *before_timer)


def test_literal_and_path_searches_do_not_arm_guard(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "sample.txt").write_text("ok\n")
    (root / "src/sample.py").write_text("ok\n")
    from kernel_parallel_records import CheckRecipe, ParallelLimits, UnitContract, UnitSpec
    from kernel_unit_policy import UnitPolicy

    monkeypatch.setattr(navigation.signal, "setitimer",
                        lambda *_args: pytest.fail("literal or path search armed CPU timer"), raising=False)
    assert navigation.search_repository(root, "ok", mode="literal").metadata["total"] == 2
    assert navigation.search_repository(root, "sample", mode="path").metadata["total"] == 2
    recipe = CheckRecipe((), (), "none")
    contract = UnitContract("a", "fix a", ("src/sample.py",), (), (), (), (), (), recipe)
    spec = UnitSpec("call", "a", "execution", "seed", contract, (), (),
                    ("src/sample.py",), (), "", 0, 0, "manifest", "policy", ParallelLimits())
    page = UnitPolicy(spec).search_visible(root, "ok", mode="literal")
    assert "src/sample.py:1:ok" in page.text
