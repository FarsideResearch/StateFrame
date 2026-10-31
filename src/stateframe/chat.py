"""Interactive session (preview): type tasks, watch each run live, keep building on the result.

Every task is an ordinary ``stateframe run`` on the session's working copy: frame-only, no chat
history. A follow-up carries ONE typed record of the session's CURRENT state (which files differ
from the session start, with hashes; the last outcome; which test files the model wrote), never
earlier task texts, transcripts or diffs. The user's repository is only written by ``/apply``,
which is checked against the session-start baseline and refuses to clobber concurrent edits.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import outcomes, workcopy
from .endpoint import EndpointError
from .runner import RunError, RunOptions, run, succeeded
from .task_row import TaskRowError

HELP = """commands:
  <text>      run a task on the session copy (follow-ups build on earlier results)
  /diff       session changes vs your repo
  /tests      last recorded test output (not re-run), with that attempt's outcome
  /undo       drop the last task's changes        /apply     copy session changes into your repo
  /history    tasks so far                        /quit      leave (your repo is untouched unless /apply)
Ctrl-C stops a running task; Ctrl-D or /quit leaves."""

FACTS_SCHEMA = "stateframe-session-facts/1"
EXPECTED_ERRORS = (RunError, EndpointError, TaskRowError, workcopy.WorkCopyError, OSError)


@dataclass
class TaskRecord:
    task: str
    result: str
    tests_passed: bool
    files: list[str]
    diff: str
    run_dir: str
    kept: bool
    attempt: int = 0
    runtime_outcome: str = "engine_error"
    check_outcome: str = "not_run"
    artifact: str = "none"
    error: str | None = None

    def outcome_label(self) -> str:
        return (f"runtime={self.runtime_outcome} checks={self.check_outcome} "
                f"artifact={self.artifact}")


@dataclass
class Session:
    repo: Path
    home: Path
    base: RunOptions
    history: list[TaskRecord] = field(default_factory=list)
    snapshots: list[Path] = field(default_factory=list)
    baseline: dict[str, str] = field(default_factory=dict)  # original's sha256 per file at session start
    attempts: int = 0
    _active_copy: Path | None = field(default=None, init=False, repr=False)
    _baseline_snapshot: Path | None = field(default=None, init=False, repr=False)

    @property
    def copy(self) -> Path:
        return self._active_copy or self.home / "repo"

    @property
    def start_snapshot(self) -> Path:
        return self._baseline_snapshot or self.home / "start"

    def start(self) -> None:
        self.home.mkdir(parents=True)
        work = workcopy.create(self.repo, self.copy)
        self.baseline = dict(work.baseline)
        shutil.copytree(self.copy, self.start_snapshot)  # immutable session-start state

    def facts(self) -> dict[str, object] | None:
        """Typed CURRENT state of the session copy (no earlier task texts, no diffs)."""
        kept = [r for r in self.history if r.kept]
        if not kept:
            return None
        start, now = _files(self.start_snapshot), _files(self.copy)
        changed = []
        for rel in _changed(self.start_snapshot, self.copy):
            status = "added" if rel not in start else "deleted" if rel not in now else "modified"
            entry = {"path": rel, "status": status}
            if rel in now:
                entry["sha256"] = hashlib.sha256(now[rel]).hexdigest()
            changed.append(entry)
        model_tests = sorted(rel for r in kept for rel in r.files
                             if rel.startswith("tests/") or Path(rel).name.startswith("test_"))
        last = kept[-1]
        return {
            "schema": FACTS_SCHEMA,
            "workspace": "already contains every change listed here (host-verified)",
            "files_changed_since_session_start": changed,
            "last_outcome": {"engine_result": last.result, "tests_after": "PASS" if last.tests_passed else "FAIL"},
            # Written by the model earlier in this session: not the user's tests.
            "tests_written_this_session": sorted(set(model_tests)),
        }

    def task_text(self, text: str) -> str:
        facts = self.facts()
        if facts is None:
            return text
        return f"{text}\n\nSESSION FACTS:\n{json.dumps(facts, indent=1, sort_keys=True)}"

    def run_task(self, text: str, printer=print) -> TaskRecord:
        # Reserve identity before facts/setup can fail. Undo never rewinds this counter.
        self.attempts += 1
        n = self.attempts
        run_dir = self.home / f"task-{n:02d}"
        record = TaskRecord(task=text, result="NOT_RUN", tests_passed=False, files=[], diff="",
                            run_dir=str(run_dir), kept=False, attempt=n)
        self.history.append(record)
        summary = {}
        try:
            options = replace(self.base, repo=self.copy, task=self.task_text(text),
                              run_dir=run_dir, in_place=False, shell_confirm=confirm_shell)
            summary = run(options, printer=printer)
            projection = outcomes.project(summary, cancelled=summary.get("runtime_outcome") == "cancelled",
                                          timed_out=summary.get("child_exit_code") == 124)
            record.runtime_outcome = str(projection["runtime_outcome"])
            record.check_outcome = str(projection["check_outcome"])
            record.artifact = str(projection["artifact"])
            record.result = str(summary.get("engine_result"))
            record.tests_passed = record.check_outcome == "pass"
            record.files = list(summary.get("files_changed") or [])
            record.diff = (run_dir / "final.diff").read_text()
            if record.files and succeeded(summary):
                self._promote(record)
        except (KeyboardInterrupt, *EXPECTED_ERRORS) as error:
            cancelled = isinstance(error, KeyboardInterrupt)
            # Preserve check/artifact evidence, but the attempt did not finish successfully.
            failed = outcomes.project({**summary, "child_exit_code": 130 if cancelled else 1},
                                      cancelled=cancelled)
            record.runtime_outcome = str(failed["runtime_outcome"])
            record.check_outcome = str(failed["check_outcome"])
            record.artifact = str(failed["artifact"])
            record.tests_passed = record.check_outcome == "pass"
            record.error = "task cancelled" if cancelled else f"task error: {_one_line(error)}"
        return record

    def _promote(self, record: TaskRecord) -> None:
        """Stage the entire next state before changing the active copy; retain both as artifacts."""
        staged = self.home / f"session-{record.attempt:02d}"
        shutil.copytree(self.copy, staged)
        baseline = {rel: hashlib.sha256(data).hexdigest() for rel, data in _files(self.copy).items()}
        workspace = Path(record.run_dir) / "workspace"
        changes = {}
        for rel in record.files:
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise workcopy.WorkCopyError(f"invalid candidate path: {rel!r}")
            source = workspace / rel
            if source.is_symlink() or any((workspace / p).is_symlink() for p in path.parents):
                raise workcopy.WorkCopyError(f"symlink in candidate path: {rel!r}")
            changes[rel] = source.read_bytes() if source.is_file() else None
        workcopy.guarded_apply(staged, changes, baseline, what="session promotion")
        self.snapshots.append(self.copy)
        self._active_copy = staged
        record.kept = True

    def undo(self) -> str:
        for i in range(len(self.history) - 1, -1, -1):
            if self.history[i].kept:
                snap = self.snapshots.pop()
                self._active_copy = snap
                self.history[i].kept = False
                record = self.history[i]
                return f"undid task {record.attempt or i + 1}: {record.task!r}"
        return "nothing to undo"

    def session_diff(self) -> str:
        """Session changes vs the session-start snapshot (never re-reads the live repository)."""
        return _tree_diff(self.start_snapshot, self.copy)

    def apply(self) -> list[str]:
        """Write session changes into the repository, refusing any file that changed there since
        the session started (the same guarded path as --in-place)."""
        changes = {rel: ((self.copy / rel).read_bytes() if (self.copy / rel).is_file() else None)
                   for rel in _changed(self.start_snapshot, self.copy)}
        if not changes:
            return []
        # Prepare the next baseline before writing the original. Failed staging leaves both intact.
        staged = Path(tempfile.mkdtemp(prefix="applied-", dir=self.home)) / "repo"
        shutil.copytree(self.copy, staged)
        applied = workcopy.guarded_apply(self.repo, changes, self.baseline, what="/apply")
        # The applied state is the new baseline: later /apply only writes later changes.
        self._baseline_snapshot = staged
        for rel in applied:
            data = changes[rel]
            if data is None:
                self.baseline.pop(rel, None)
            else:
                self.baseline[rel] = hashlib.sha256(data).hexdigest()
        return applied


def _files(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _one_line(error: BaseException) -> str:
    return " ".join(str(error).splitlines()) or type(error).__name__


def _changed(a: Path, b: Path) -> list[str]:
    fa, fb = _files(a), _files(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


def _tree_diff(a: Path, b: Path) -> str:
    import difflib
    out = []
    fa, fb = _files(a), _files(b)
    for rel in _changed(a, b):
        old = fa.get(rel, b"").decode("utf-8", "replace").splitlines(keepends=True)
        new = fb.get(rel, b"").decode("utf-8", "replace").splitlines(keepends=True)
        out.extend(difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}"))
    return "".join(out) or "(no session changes)"


def default_home(repo: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.home() / ".stateframe" / "sessions" / f"{repo.resolve().name}-{stamp}"


def repl(session: Session, read=input, printer=print, print_summary=None) -> int:
    session.start()
    printer(f"stateframe chat (preview) on {session.repo}  [session copy: {session.copy}]")
    printer(HELP)
    while True:
        try:
            line = read("\nstateframe> ").strip()
        except EOFError:
            printer("")
            break
        except KeyboardInterrupt:
            printer("\n(Ctrl-D or /quit to leave)")
            continue
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            printer(HELP)
        elif line == "/diff":
            printer(session.session_diff())
        elif line == "/undo":
            printer(session.undo())
        elif line == "/history":
            for i, r in enumerate(session.history, 1):
                printer(f"{r.attempt or i}. [{r.outcome_label()}"
                        f"{'' if r.kept else ', not kept'}] {r.task}")
            if not session.history:
                printer("(no tasks yet)")
        elif line == "/apply":
            try:
                changed = session.apply()
            except (KeyboardInterrupt, *EXPECTED_ERRORS) as error:
                printer(f"not applied: {_one_line(error)}")
                continue
            printer("applied to your repo: " + (", ".join(changed) if changed else "nothing"))
        elif line == "/tests":
            if not session.history:
                printer("(run a task first)")
            else:
                record = session.history[-1]
                printer(f"last recorded test output (not re-run); attempt {record.attempt}: "
                        f"{record.outcome_label()}")
                try:
                    printer(Path(record.run_dir, "tests-after.txt").read_text()[-2000:])
                except FileNotFoundError:
                    printer("(no recorded test output for this attempt)")
                except OSError as error:
                    printer(f"cannot read recorded test output: {_one_line(error)}")
        elif line.startswith("/"):
            printer(f"unknown command {line}; /help")
        else:
            record = session.run_task(line, printer=printer)
            if print_summary is not None and record.error is None:
                try:
                    print_summary(Path(record.run_dir) / "summary.json")
                except OSError as error:
                    printer(f"cannot read run summary: {_one_line(error)}")
            printer(f"=> attempt {record.attempt}: {record.error or record.result}; "
                    f"{record.outcome_label()}; "
                    f"{'kept on the session copy' if record.kept else 'session copy unchanged; attempt path ' + record.run_dir}"
                    f" (/diff, /undo, /apply)")
    printer(f"session dir {session.home} holds inspectable artifacts and is NOT resumable; "
            "your repo changes only through /apply")
    return 0


__all__ = ["Session", "repl", "default_home", "HELP"]


def confirm_shell(command: str) -> bool:
    """Confirm a shell request only while this chat owns an interactive terminal."""
    import sys
    if not sys.stdin.isatty():
        return False
    import select
    print(f"Run shell command {command!r}? [y/N] ", end="", flush=True)
    if not select.select([sys.stdin], [], [], 115)[0]:
        return False
    try:
        return input().strip().lower() in {"y", "yes"}
    except EOFError:
        return False
