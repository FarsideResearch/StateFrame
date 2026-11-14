"""Class guard: every kill / refusal / fail-open site must say why (kernel_reason_log).

Also pins the request-path byte-safety rule (journal=False) and the run-end problems summary.
"""
from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import kernel_reason_log as rl
from _reason_log_scanner import WRAPPERS, scan_reason_sites

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stateframe"
SCANNED_FLOOR = 80       # a broken detector must not pass by finding nothing
WITH_REASON_FLOOR = 70
WRAPPERS.add("log_seal_violation")

# "path:function" -> why this site is NOT expected to log itself.
ALLOWLIST = {
    "src/stateframe/engine/kernel_bounded_http.py:_terminate_child":
        "stdlib-only isolated child (test_installed_isolated_children); cannot import the helper; its parent reports the timeout",
    "src/stateframe/engine/kernel_hook_driver.py:_stop_group":
        "sandboxed hook driver, stdlib-only (test_process_signal_authority); the host hook wrapper logs the hook kill",
    "src/stateframe/engine/kernel_hook_driver.py:run":
        "sandboxed hook driver, stdlib-only; the host hook wrapper logs the hook kill",
    "src/stateframe/engine/kernel_exec_tracker.py:signal_launch_group":
        "low-level signaler; kill_launch_unit and the supervisors log the cause at the caller",
    "src/stateframe/engine/kernel_exec_tracker.py:signal_after_recheck":
        "low-level signaler; the caller that chose the kill logs it",
    "src/stateframe/engine/kernel_exec_tracker.py:signal":
        "low-level signaler on a verified launch identity; the caller that chose the kill logs it",
    "src/stateframe/engine/kernel_environment_state.py:_signal_group":
        "signaler; cleanup() owns the reason and journals environment_process_* events",
    "src/stateframe/runner.py:_stop_group":
        "post-exit cleanup helper; the wall-deadline caller (on_timeout) logs the kill with its reason",
}


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def test_repo_has_no_unlogged_sites():
    scan = scan_reason_sites(_sources(), root=ROOT, allowlist=ALLOWLIST)
    print(scan.receipt)
    assert scan.violations == (), "\n".join(scan.violations)
    assert scan.scanned >= SCANNED_FLOOR, scan.receipt
    assert scan.with_reason >= WITH_REASON_FLOOR, scan.receipt


def test_allowlist_entries_have_reasons_and_are_not_stale():
    assert all(reason.strip() for reason in ALLOWLIST.values())
    scan = scan_reason_sites(_sources(), root=ROOT, allowlist=ALLOWLIST)
    assert scan.stale_allowlist == (), f"allowlist entries that are no longer sites: {scan.stale_allowlist}"
    assert scan.allowlisted >= len(ALLOWLIST)  # one entry can cover several sites in a function


PLANTED = {
    "planted_kill.py": "import os, signal\ndef stop(p):\n    os.killpg(p.pid, signal.SIGKILL)\n",
    "planted_refusal.py": "def f():\n    raise ContextCapacityError('x')\n",
    "planted_failopen.py": "def fail_open_pending(capture, journal, *, reason):\n    capture.pending = None\n",
}
PLANTED_OK = {
    "ok_kill.py": ("import os, signal\ndef stop(p):\n    emit_kill('c', target='t', why='w')\n"
                   "    os.killpg(p.pid, signal.SIGKILL)\n"),
    "ok_refusal.py": ("def f():\n    emit_refusal('c', what='w', why='y')\n"
                      "    raise ContextCapacityError('x')\n"),
    "ok_failopen.py": ("def fail_open_pending(capture, journal, *, reason):\n"
                       "    emit_reason('fallback', 'k', 'c', 'w', reason)\n    capture.pending = None\n"),
}


def _write(tmp_path: Path, files: dict[str, str]) -> list[Path]:
    paths = []
    for name, text in files.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def test_planted_control_detector_fires_exactly_three_times(tmp_path):
    scan = scan_reason_sites(_write(tmp_path, PLANTED), root=tmp_path)
    assert scan.scanned == 3 and len(scan.violations) == 3, scan.violations
    kinds = sorted(v.rsplit("(", 1)[1] for v in scan.violations)
    assert kinds == ["drop)", "kill)", "refusal)"]


def test_planted_control_logged_copies_are_clean(tmp_path):
    scan = scan_reason_sites(_write(tmp_path, PLANTED_OK), root=tmp_path)
    assert scan.scanned == 3 and scan.violations == () and scan.with_reason == 3


def test_liveness_probe_and_signal_handler_are_not_sites(tmp_path):
    source = textwrap.dedent("""
        import os, signal
        def probe(pid):
            os.kill(pid, 0)
            os.killpg(pid, 0)
            signal.signal(signal.SIGTERM, lambda *a: None)
    """)
    scan = scan_reason_sites(_write(tmp_path, {"probe.py": source}), root=tmp_path)
    assert scan.scanned == 0


def test_allowlist_suppresses_only_the_named_function(tmp_path):
    paths = _write(tmp_path, PLANTED)
    scan = scan_reason_sites(paths, root=tmp_path, allowlist={"planted_kill.py:stop": "test"})
    assert scan.allowlisted == 1 and len(scan.violations) == 2


# ---- request-path byte safety -------------------------------------------------------------

REQUEST_PATH_MODULES = (
    "kernel_request_diet", "kernel_request_budget", "kernel_skills", "kernel_prompt_compact",
    "kernel_agent_schemas", "kernel_request_seal", "kernel_request_reprepare",
    "kernel_stateframe_frame_append", "kernel_stateframe_page_log", "kernel_stateframe_tool_stable",
    "kernel_stateframe_runtime", "kernel_measurement", "kernel_context_capacity",
)
EMITTERS_WITH_JOURNAL = {"emit_reason", "emit_refusal", "emit_timeout"}


def _journal_kw_false(call: ast.Call) -> bool:
    return any(k.arg == "journal" and isinstance(k.value, ast.Constant) and k.value.value is False
               for k in call.keywords)


@pytest.mark.parametrize("module", REQUEST_PATH_MODULES)
def test_request_path_log_calls_never_touch_the_journal(module):
    """A reason_logged event in the journal shifts identity-bearing positions (see
    kernel_request_diet._emit_fallback), so request-path sites pass journal=False."""
    tree = ast.parse((SRC / "engine" / f"{module}.py").read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in EMITTERS_WITH_JOURNAL):
            continue
        # Only the kernel_reason_log helpers take a component string first; skip look-alikes.
        if not (node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            if not any(k.arg in {"what", "why"} for k in node.keywords):
                continue
        if (node.func.id == "emit_reason" and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "kill"):
            continue
        if not _journal_kw_false(node):
            offenders.append(f"{module}:{node.lineno} {node.func.id}")
    assert offenders == []


class _Journal:
    def __init__(self):
        self.events = []

    def emit(self, kind, **fields):
        self.events.append((kind, fields))


@pytest.fixture()
def fresh(monkeypatch, tmp_path):
    monkeypatch.delenv(rl.LEVEL_ENV, raising=False)
    monkeypatch.delenv(rl.PROBLEMS_ENV, raising=False)
    rl.reset_for_tests()
    yield tmp_path
    rl.reset_for_tests()


def test_journal_false_skips_the_journal_but_keeps_stderr_and_problems_file(fresh):
    journal, out = _Journal(), io.StringIO()
    problems = fresh / "problems.jsonl"
    rl.configure(level="WARN", journal=journal, problems_path=problems, stream=out)
    assert rl.emit_reason("fallback", "a", "request_diet", "w", "y", journal=False) is True
    assert rl.emit_refusal("request_diet", what="w", why="y", cause_key="b", journal=False) is True
    assert rl.emit_timeout("request_diet", what="w", why="y", cause_key="c", journal=False) is True
    assert journal.events == []
    assert out.getvalue().count("[stateframe] WARN request_diet:") == 3
    assert len(problems.read_text().splitlines()) == 3
    # The default (no kwarg) still reaches the bound journal, so False is a real opt-out.
    rl.emit_reason("fallback", "d", "runner", "w", "y")
    assert [kind for kind, _ in journal.events] == ["reason_logged"]


def test_request_path_sites_leave_a_bound_journal_untouched(fresh):
    import kernel_request_diet as diet
    import kernel_request_seal as seal
    import kernel_skills as skills

    journal = _Journal()
    rl.configure(level="DEBUG", journal=journal, stream=io.StringIO())
    meter = diet.RequestTokenMeter(tokenizer=lambda _text: 1)
    meter._deadline = 0.0
    with pytest.raises(TimeoutError):
        meter.measure({"messages": [], "tools": []})
    seal.log_seal_violation("qualified request identity changed")
    skills.note_budget_omission([{"role": "user", "content": "[StateFrame]"}],
                                {"_request_budget_section": "skill_bodies",
                                 "content": "USER-PROVIDED SKILL INSTRUCTIONS: demo-skill (body)"})
    assert journal.events == []
    assert {"timeout:tokenizer_deadline", "invariant:qualified request identity changed",
            "eviction:skill_omitted"} <= set(rl.run_summary()["by_cause"])


# ---- bounded cause keys / levels ---------------------------------------------------------------

def test_seal_cause_key_is_bounded_by_message_prefix(fresh):
    import kernel_request_seal as seal
    rl.configure(level="ERROR", stream=io.StringIO())
    for field in ("a", "b", "c"):
        seal.log_seal_violation("qualified request classification changed: " + field)
    assert list(rl.run_summary()["by_cause"]) == ["invariant:qualified request classification changed"]


def test_diet_deferred_unknown_reason_collapses_to_other(fresh):
    import kernel_request_budget as budget
    rl.configure(level="WARN", stream=io.StringIO())
    for reason in ("r1", "r2", "required_evidence_removed"):
        budget._log_diet_deferred({"reason": reason}, stage="FREE")
    assert sorted(rl.run_summary()["by_cause"]) == [
        "refusal:diet_deferred:other", "refusal:diet_deferred:required_evidence_removed"]


def test_byte_fallback_without_tokenizer_is_info_not_warn(fresh):
    import kernel_request_diet as diet
    out = io.StringIO()
    rl.configure(level="WARN", stream=out)
    meter = diet.RequestTokenMeter(None)
    for _ in range(3):
        meter.measure({"messages": [{"role": "user", "content": "hi"}], "tools": []})
    assert out.getvalue() == ""  # a healthy byte-sized run prints nothing at the WARN default
    levels = {v["level"] for v in rl.run_summary()["by_cause"].values()}
    assert levels == {"INFO"}


def test_configured_tokenizer_failure_fallback_is_warn(fresh):
    import kernel_request_diet as diet
    out = io.StringIO()
    rl.configure(level="WARN", stream=out)

    def broken(_text):
        raise OSError("down")
    meter = diet.RequestTokenMeter(broken)
    meter.measure({"messages": [{"role": "user", "content": "hi"}], "tools": []})
    assert "[stateframe] WARN request_diet: tokenizer unavailable" in out.getvalue()


# ---- run-end flush and the problems summary / gate line, end to end -----------------------------

def test_runtime_finish_flushes_the_summary_once(fresh):
    source = (SRC / "engine" / "kernel_stateframe_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    finish = next(n for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef) and cls.name == "StateFrameRuntime"
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "finish")
    assert any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "flush_summary"
               for c in ast.walk(finish))
    journal, out = _Journal(), io.StringIO()
    rl.configure(level="WARN", stream=out)
    rl.emit_reason("fallback", "x", "c", "w", "y", journal=False)
    assert rl.flush_summary(journal) is not None
    assert rl.flush_summary(journal) is None  # idempotent: finish() and journal.close() may both call it
    assert [k for k, _ in journal.events] == ["reason_log_summary"]
    assert out.getvalue().count("[stateframe] problems:") == 1


GATE_GREP = r"grep -rE '^\[stateframe\] ERROR ' "


def _gate(path: Path) -> int:
    return subprocess.run(["bash", "-c", GATE_GREP + str(path)], capture_output=True).returncode


def test_problems_summary_and_gate_line_end_to_end(fresh):
    sys.path.insert(0, str(ROOT / "src"))
    from stateframe import runner

    run_dir = fresh / "run"
    run_dir.mkdir()
    problems = run_dir / "problems.jsonl"
    out = io.StringIO()
    rl.configure(level="WARN", problems_path=problems, stream=out)
    rl.emit_kill("exec_tracker", target="probe", why="wall exceeded", pid=1)           # ERROR
    rl.emit_refusal("request_diet", what="w", why="y", cause_key="k", journal=False)   # WARN
    rl.emit_refusal("request_diet", what="w", why="y", cause_key="k", journal=False)   # repeat: counted
    rl.flush_summary()
    expected = rl.run_summary()
    problems_block = runner._problems_summary(run_dir)
    assert problems_block["errors"] == 1 and problems_block["warnings"] == 1
    assert problems_block["by_cause"]["refusal:k"]["count"] == expected["by_cause"]["refusal:k"]["count"] == 2

    dirty = fresh / "stderr_dirty.txt"
    dirty.write_text(out.getvalue())
    assert "[stateframe] ERROR exec_tracker:" in dirty.read_text()
    assert _gate(dirty) == 0  # grep found the ERROR line (gate script turns a hit into exit 1)

    clean = fresh / "stderr_clean.txt"
    clean.write_text("[stateframe] WARN request_diet: w (y)\nordinary output\n")
    assert _gate(clean) == 1  # no ERROR line: grep misses, gate passes
    gate_hit = subprocess.run(
        ["bash", "-c", f"{GATE_GREP}{dirty} >/dev/null && echo GATE FAIL && exit 1 || true"],
        capture_output=True, text=True)
    assert gate_hit.returncode == 1 and "GATE FAIL" in gate_hit.stdout
    gate_clean = subprocess.run(
        ["bash", "-c", f"{GATE_GREP}{clean} >/dev/null && echo GATE FAIL && exit 1 || true"],
        capture_output=True, text=True)
    assert gate_clean.returncode == 0 and gate_clean.stdout == ""
    # The json gate: summary.json problems.errors drives the same verdict.
    summary = run_dir / "summary.json"
    summary.write_text(json.dumps({"problems": problems_block}))
    code = ("import json,sys;p=json.load(open(sys.argv[1])).get('problems',{});"
            "sys.exit(1 if p.get('errors') else 0)")
    assert subprocess.run([sys.executable, "-c", code, str(summary)]).returncode == 1
