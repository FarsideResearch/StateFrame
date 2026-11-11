"""``stateframe run``: prepare a run directory, drive the engine child, report the outcome."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import queue
import threading
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from . import cache_metrics, outer_checks, outcomes, preflight, profiles, routes, test_interpreter, workcopy
from .test_runners import (DECLARED_TEST_SOURCE, declared_test_evidence, discover_check,
                            lock_initial_failing_tests, parse_command, validate_patterns)
from .test_changes import inspect_test_changes
from . import resume as resume_module
from .endpoint import CHAT_SUFFIX, Endpoint, EndpointError, _is_loopback
from .task_row import EDIT_SCOPE_WIDEN_CAP, build_row, widened_scope_paths
from .engine.kernel_custom_tools import CustomToolError, load_tool_files
from .engine.kernel_mcp_tools import load_mcp_files, discover as discover_mcp
from .engine.kernel_tool_settings import ToolSettings
from . import instructions as project_instructions
from .skills import SkillError, measured_skills_digest, resolve_skills, skills_digest, verify_records
from .state_dirs import private_directory
from .request_records import RequestRecordsError, list_request_records
from .storage_gc import _fsync_directory, _write_json_atomic

from .engine.kernel_safe_paths import safe_home

if TYPE_CHECKING:
    from .engine.kernel_hooks import HookSet

PACKAGE = Path(__file__).resolve().parent
ENGINE_DIR = PACKAGE / "engine"
CHILD = PACKAGE / "_child.py"
LOG_PREFIX = "[sf] "
# Environment the engine child inherits from the parent; everything else is dropped.
PASS_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "USER", "LOGNAME", "SHELL",
            "SYSTEMROOT", "TERM")
# Documented user settings the engine child reads directly; copied into the child environment when set.
# LLAMA_CPP_TOKENIZE_URL: token counting against a co-located llama.cpp /tokenize (kernel_request_diet).
USER_ENV = ("LLAMA_CPP_TOKENIZE_URL",)
# The engine child's test interpreter (absolute path); the engine's test_python() helper reads exactly this key.
TEST_PYTHON_ENV = "STATEFRAME_TEST_PYTHON"
# Engine checkpoint switches the runner copies into the run's environment (kernel_checkpoint).
CHECKPOINT_ENV = {"STATEFRAME_CHECKPOINT": "save", "STATEFRAME_CHECKPOINT_EVERY": "1",
                  "STATEFRAME_CHECKPOINT_KEEP": "2",
                  "STATEFRAME_CHECKPOINT_MIN_FREE_BYTES": ""}


def checkpoint_env(profile_env=None):
    """Checkpoint switches for the run, or none when the engine has no checkpoint kernel.

    An engine refuses STATEFRAME settings it does not consume, so a product/engine mismatch must
    degrade to no checkpoints (resume then replays), never to a failed run.
    """
    if not (ENGINE_DIR / "kernel_checkpoint.py").is_file():
        return {}
    # Precedence: ambient environment, then the profile, then the release default.
    profile_env = profile_env or {}
    # An empty ambient value (`STATEFRAME_CHECKPOINT= stateframe run`) counts as unset, never as "off".
    return {key: os.environ.get(key, "").strip() or profile_env.get(key, default)
            for key, default in CHECKPOINT_ENV.items()}


def edit_scope_widened(spec: dict) -> bool:
    """Whether the run was started with opt-in edit scope widening."""
    value = str((spec.get("environment") or {}).get("STATEFRAME_EDIT_SCOPE_WIDEN", "")).strip().lower()
    return value == "on"


def _widen_admissions(run_dir: Path, baseline: dict[str, str]) -> list[str]:
    """Read the sole durable engine journal in admission order, failing closed."""
    from .engine.kernel_journal_store import journal_paths, open_journal

    journals = journal_paths(run_dir / "worker" / "measurement-runs")
    if len(journals) != 1:
        raise RunError("edit scope admission journal is missing or ambiguous")
    admitted: list[str] = []
    try:
        with open_journal(journals[0]) as source:
            for line in source:
                record = json.loads(line)
                if record.get("kind") != "scope_widen":
                    continue
                path, digest = record.get("path"), record.get("pre_edit_sha256")
                if (not isinstance(path, str) or not isinstance(digest, str)
                        or digest != baseline.get(path) or path in admitted
                        or len(admitted) >= EDIT_SCOPE_WIDEN_CAP):
                    raise RunError("invalid edit scope admission")
                admitted.append(path)
    except (OSError, UnicodeError, ValueError, AttributeError) as error:
        raise RunError(f"cannot verify edit scope admissions: {error}") from error
    return admitted


def _widen_hook_paths(spec: dict) -> frozenset[str]:
    hooks = spec.get("hooks")
    if not hooks:
        return frozenset()
    from .engine.kernel_hooks import HookSet
    pinned = {key: hooks[key] for key in ("files", "definitions", "digest", "bundle_manifest")}
    return HookSet.from_spec(pinned).protected_repo_paths(Path(spec["repo"]))


def _admitted_candidate_paths(run_dir: Path, spec: dict, work: workcopy.WorkCopy,
                              candidate: dict[str, bytes | None]) -> list[str]:
    row = spec["public_row"]
    if row.get("scoped_edits") is not True:
        return []
    declared = set(row["files"])
    changed = [path for path, data in candidate.items()
               if path in work.baseline and path not in declared
               and (data is None or hashlib.sha256(data).hexdigest() != work.baseline[path])]
    if not changed:
        return []
    admissions = _widen_admissions(run_dir, work.baseline)
    hook_paths = _widen_hook_paths(spec)
    allowed = widened_scope_paths(row, admissions, work.baseline, hook_paths=hook_paths)
    if allowed != admissions:
        raise RunError("edit scope journal contains an inadmissible path")
    if not set(changed) <= set(allowed):
        raise RunError("sealed candidate contains an unadmitted edit outside scope")
    for path in changed:
        with workcopy._parent_fd(work.copy, path) as parent:
            name = Path(path).name
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=parent)
            with os.fdopen(fd, "rb") as source:
                identity = os.fstat(source.fileno())
                if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                    raise RunError(f"widened candidate source is not an isolated file: {path}")
                current = source.read()
            live = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if ((live.st_dev, live.st_ino) != (identity.st_dev, identity.st_ino)
                    or hashlib.sha256(current).hexdigest() != work.baseline[path]):
                raise RunError(f"widened candidate source changed before host apply: {path}")
    return allowed


def _dropped_existing_candidate_paths(work: workcopy.WorkCopy,
                                      candidate: dict[str, bytes | None],
                                      declared: set[str]) -> list[str]:
    """Find changed baseline files that apply_files would silently omit."""
    return sorted(path for path, data in candidate.items()
                  if path in work.baseline and path not in declared
                  and (data is None or hashlib.sha256(data).hexdigest() != work.baseline[path]))


def checkpoint_resume_off(spec: dict) -> bool:
    """An explicit current or recorded opt-out blocks restore and all later saves."""
    current = os.environ.get("STATEFRAME_CHECKPOINT", "").strip().lower()
    recorded = str(spec.get("environment", {}).get("STATEFRAME_CHECKPOINT", "")).strip().lower()
    return current in {"off", "0"} or recorded in {"off", "0"}


class RunError(RuntimeError):
    pass


class HookContainmentRunError(RunError):
    """Hook containment failed after the run directory was created."""


class HookRefusedRunError(RunError):
    """A runner gating hook refused the prompt before the engine started."""

    def __init__(self, message: str, *, result: str):
        super().__init__(message)
        self.result = result


class RunDeadlineExceeded(RunError):
    """An absolute deadline, never a recoverable candidate materialization error."""


def _validate_timeout(timeout_s: float | None) -> None:
    """None means no time limit (the default); a user-set limit must be finite and positive."""
    if timeout_s is None:
        return
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RunError("--timeout must be finite and positive")


def _remaining(options: RunOptions) -> float:
    if options._deadline is not None:
        remaining = options._deadline - time.monotonic()
    elif options.timeout_s is not None:
        remaining = options.timeout_s
    else:
        return math.inf  # no time limit: the run goes until the engine finishes or Ctrl-C
    if remaining <= 0:
        raise RunDeadlineExceeded("run deadline exhausted; partial artifacts remain in the run directory")
    return remaining


# The engine stops its model loop at its own wall, then seals, stores and exports the submission.
# Killing the child at that same instant discarded a sealed, stored submission (one long run:
# done_submission_stored at 1775.7 s, killed during export). The hard kill therefore allows a
# finalization grace past the engine wall, still bounded by the absolute run deadline.
ENGINE_FINALIZE_GRACE_S = 120.0


def _kill_deadline(engine_deadline: float, run_deadline: float | None, budget_s: float) -> float:
    # Grace scales with the engine budget (10%, at most 120 s) so tiny limits stay tight.
    grace = min(ENGINE_FINALIZE_GRACE_S, 0.1 * max(0.0, budget_s))
    return min(engine_deadline + grace,
               run_deadline if run_deadline is not None else math.inf)


@contextmanager
def _wall_deadline(options: RunOptions):
    """Bound synchronous copy/finalization as well as subprocess phases on POSIX.

    CLI and chat execute on the main thread. Refuse callers where a synchronous
    filesystem operation cannot be interrupted instead of promising a false limit.
    """
    if options.timeout_s is None:
        yield  # no time limit: no process alarm at all
        return
    if threading.current_thread() is not threading.main_thread():
        raise RunError("run deadline requires the main thread")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RunError("run deadline cannot replace an existing process alarm")

    def expired(signum, frame):
        raise RunDeadlineExceeded("run deadline exhausted; partial artifacts remain in the run directory")

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, _remaining(options))
        yield
        _remaining(options)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class RunOptions:
    repo: Path
    task: str
    endpoint: Endpoint | None
    max_turns: int | None = None  # None: the profile's REACT_STEPS
    test_cmd: str | None = None
    check_cmd: str | None = None
    check_source: str | None = None
    runner_kind: str = "auto"
    protected_test_globs: list[str] = field(default_factory=list)
    protect_tests: str | None = None
    checks: list[str] = field(default_factory=list)
    edit_paths: list[str] = field(default_factory=list)
    target: str | None = None
    run_dir: Path | None = None
    in_place: bool = False
    profile: str = profiles.DEFAULT
    python: str = sys.executable  # the ENGINE's interpreter
    # The interpreter for the repository's tests (test_interpreter.choose). python_explicit marks a
    # user-passed --python, which keeps its old meaning of engine AND tests.
    test_python: str | None = None
    python_explicit: bool = False
    _test_python: str | None = field(default=None, init=False, repr=False)
    # The exact interpreter path whose contained startup probe already imported pytest this run.
    _pytest_proven: str | None = field(default=None, init=False, repr=False)
    keep_checkpoints: bool = False  # --keep-checkpoints / STATEFRAME_KEEP_CHECKPOINTS=1
    keep_run_env: bool = False  # --keep-run-env / STATEFRAME_KEEP_RUN_ENV=on
    allow_source_tree: bool = False
    mutations: str | None = None
    mutation_network: str = "none"
    mutation_domains: list[str] = field(default_factory=list)
    mutation_write: list[str] = field(default_factory=list)
    mutation_read: list[str] = field(default_factory=list)
    mutation_retain: str = "run"
    mutation_audit: str = "source"
    mutation_hooks: list[str] = field(default_factory=list)
    _context_tokens: int = field(default=65_536, init=False, repr=False)
    _prompt_budget: int = field(default=33_344, init=False, repr=False)
    _context_env: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _context_display: str = field(default="default", init=False, repr=False)
    slot: int | None = None
    timeout_s: float | None = None  # None: no time limit (the product default)
    quiet: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)
    _cache_slot: int | None = field(default=None, init=False, repr=False)
    _cache_slot_selection: str | None = field(default=None, init=False, repr=False)
    _cache_slot_reason: str | None = field(default=None, init=False, repr=False)
    _cache_slot_degraded: bool = field(default=False, init=False, repr=False)
    _capabilities: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    # Environment for the repository's tests (engine checks and --test-cmd), e.g. test-only
    # settings a project needs to import. Declared to the engine as host task env keys.
    test_env: dict[str, str] = field(default_factory=dict)
    pytest_plugins: str | None = None  # "installed": let pytest load installed plugins
    tool_settings: ToolSettings = field(default_factory=ToolSettings)
    skills: tuple[str, ...] = ()
    trust_skill_tools: bool = False
    # "auto": the repo's AGENTS.md, else CLAUDE.md; "off": none; anything else: that file.
    instructions: str = project_instructions.DEFAULT_MODE
    hooks_files: tuple[Path, ...] = ()
    powerup_requests: tuple[str, ...] = ()
    powerup_sets: tuple[str, ...] = ()
    no_powerups: bool = False
    no_project_hooks: bool = False
    hooks_repo: Path | None = None
    project_hooks: HookSet | None = field(default=None, repr=False)
    project_hooks_digest: str | None = field(default=None, repr=False)
    project_hooks_resolved: bool = field(default=False, repr=False)
    shell_confirm: Callable[[str], bool] | None = field(default=None, repr=False)
    _deadline: float | None = field(default=None, init=False, repr=False)
    route_cli: list[str] = field(default_factory=list)
    routes_file: Path | None = None
    _routes: routes.RouteTable | None = field(default=None, init=False, repr=False)
    _recorded_routes: object = field(default=None, init=False, repr=False)
    _recorded_route_default: dict[str, object] | None = field(default=None, init=False, repr=False)
    _provider_capabilities: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _provider_route_dialects: dict[str, dict[str, str]] = field(default_factory=dict,
                                                                  init=False, repr=False)


TEST_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
# Never accepted as test env: they would steer the engine itself, not the tests.
TEST_ENV_FORBIDDEN_PREFIXES = ("STATEFRAME_", "KERNEL_", "REACT_", "DS_", "PILOT_")


def validate_test_env(test_env: dict[str, str]) -> dict[str, str]:
    for key in test_env:
        if not TEST_ENV_KEY.match(key) or key.startswith(TEST_ENV_FORBIDDEN_PREFIXES):
            raise RunError(f"--test-env {key}: must be an UPPER_CASE name and not an engine setting")
    return dict(test_env)


def default_run_dir(repo: Path) -> Path:
    """Outside the repository, so a run never writes anything into the user's tree."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    home = safe_home()
    state_root = (home if home is not None else
                  Path(tempfile.gettempdir()) / f"stateframe-{getattr(os, 'getuid', lambda: 'unknown')()}")
    return state_root / ".stateframe" / "runs" / f"{repo.resolve().name}-{stamp}"


def run_test_cmd(command: str, cwd: Path, timeout_s: float = 900,
                 test_env: dict[str, str] | None = None, *, python: str = sys.executable,
                 pytest_plugins: str | None = None,
                 deadline: float | None = None, test_python: str | None = None,
                 runner_kind: str = "pytest", capture_output: bool = False) -> dict[str, object]:
    """Check a throwaway copy through the engine policy in a fresh interpreter.

    ``test_python``: the interpreter the command runs, handed to an engine that reads STATEFRAME_TEST_PYTHON so its
    sandbox admits it (None for an older engine, which would refuse the unknown setting)."""
    try:
        return outer_checks.run(command, cwd, python=python, timeout_s=timeout_s,
                                deadline=deadline, test_env=validate_test_env(test_env or {}),
                                pytest_plugins=pytest_plugins, test_python=test_python,
                                runner_kind=runner_kind, capture_output=capture_output)
    except (outer_checks.OuterCheckError, OSError) as error:
        raise RunError(str(error)) from error


# A cold interpreter importing pytest inside a fresh sandbox can take tens of seconds on a loaded
# machine (measured: 10 s was exceeded in a busy Linux container); stay bounded but tolerant.
STARTUP_PROBE_SECONDS = 60


def _startup_probe_seconds() -> float:
    name = "STATEFRAME_STARTUP_PROBE_SECONDS"
    raw = os.environ.get(name)
    if raw is None:
        return STARTUP_PROBE_SECONDS
    try:
        seconds = float(raw)
    except ValueError:
        seconds = 0.0
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 300:
        raise RunError(f"{name}={raw!r} must be a positive number of seconds at most 300 "
                       "(the retry limit is 600); set it or unset it")
    return seconds


def _probe_stopped_by_host(result: object) -> bool:
    """A wall timeout, or a SIGKILL our own cleanup delivered: load, never a broken interpreter."""
    if getattr(result, "timed_out", False):
        return True
    cleanup = getattr(result, "cleanup", None)
    return (getattr(result, "exit_code", None) in (137, -9)
            and int(getattr(cleanup, "killed", 0) or 0) > 0)


def _startup_probe(python: str, code: str, *, label: str) -> bytes:
    """Contain interpreter startup itself, before importing any project packages.

    A heavily loaded host can push a healthy interpreter past the wall limit, so a timeout (and
    only a timeout) is retried once with double the wall. A non-zero exit or an oversized output is
    a real failure and is reported at once.
    """
    from .engine import kernel_safe_probe
    from .engine.kernel_exec_session import ExecSessionError

    base_wall = _startup_probe_seconds()
    walls = (base_wall, 2 * base_wall)
    for wall in walls:
        try:
            with tempfile.TemporaryDirectory(prefix="stateframe-startup-") as directory:
                result = kernel_safe_probe.run_probe([python, "-I", "-B", "-c", code], Path(directory),
                                                     timeout=wall)
        except (OSError, ExecSessionError, ValueError) as error:
            raise RunError(f"{label} startup probe could not run safely: {error}") from error
        if not _probe_stopped_by_host(result):
            break
    else:
        raise RunError(f"{label} startup probe timed out under load: it did not finish within "
                       f"{walls[0]} s and then {walls[1]} s. The interpreter may be fine and the "
                       "host overloaded; retry when the machine is less busy")
    if result.output_truncated:
        raise RunError(f"{label} startup probe exceeded its 65536 byte output limit")
    if result.exit_code != 0:
        detail = result.output.decode("utf-8", "replace").strip()[-1000:]
        failure = ("cannot import pytest" if "pytest" in code else "could not start")
        install = "install pytest in that environment; " if "pytest" in code else ""
        raise RunError(f"{label} {failure} in the isolated startup probe; the interpreter is broken "
                       f"or incomplete (exit {result.exit_code}, not a timeout); {install}"
                       f"sandbox restrictions must permit interpreter startup: {detail or 'no diagnostic output'}")
    return result.output


def _check_python(python: str, *, require_pytest: bool = True) -> str:
    """Resolve --python to an absolute executable (relative to the caller's cwd, BEFORE any chdir)
    and check it can run the engine and the checks. Returns the absolute path."""
    try:
        python = preflight.resolve_interpreter(python, os.getcwd())
    except preflight.PreflightError as error:
        raise RunError(str(error)) from None
    imports = "import sys, json, pytest" if require_pytest else "import sys, json"
    output = _startup_probe(python, imports + "; print(json.dumps(list(sys.version_info[:3])))",
                            label=f"engine interpreter {python}")
    try:
        version = json.loads(output)
        if (not isinstance(version, list) or len(version) != 3
                or any(type(value) is not int or value < 0 for value in version)):
            raise ValueError("expected three non-negative version integers")
    except (ValueError, UnicodeError) as error:
        raise RunError(f"engine interpreter {python} startup probe returned an invalid Python version") from error
    try:
        preflight.check_python_version(version=tuple(version))
    except preflight.PreflightError as error:
        raise RunError(f"interpreter {python}: {error}") from None
    return python


_ENGINE_SWITCHES: dict[str, frozenset[str]] = {}


def engine_switches(engine_dir: Path = ENGINE_DIR) -> frozenset[str]:
    """The STATEFRAME_* switches the shipped engine reads, from the engine's own switch registry (loaded
    by file path; stdlib only). The engine refuses any STATEFRAME_* setting it has no reader for
    (HARNESS_CONTRACT), so the runner writes an optional key only to an engine that reads it."""
    key = str(engine_dir)
    if key not in _ENGINE_SWITCHES:
        import importlib.util
        switches: frozenset[str] = frozenset()
        source = engine_dir / "kernel_release_switch_registry.py"
        try:
            spec = importlib.util.spec_from_file_location("_sf_switch_registry", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            switches = frozenset(module.release_stateframe_switches(engine_dir)[0])
        except (OSError, ImportError, AttributeError, SyntaxError, ValueError):
            switches = frozenset()
        _ENGINE_SWITCHES[key] = switches
    return _ENGINE_SWITCHES[key]


def engine_reads_test_python(engine_dir: Path = ENGINE_DIR) -> bool:
    return TEST_PYTHON_ENV in engine_switches(engine_dir)


def _outer_test_python(options: "RunOptions") -> str | None:
    """The before/after checks' interpreter hand-off: only to an engine that reads it."""
    return _tests_python(options) if engine_reads_test_python() else None


# Context size: exactly these two names pass from the user's environment or the
# profile to the engine, ambient > profile > default, never a blanket STATEFRAME_* passthrough.
CONTEXT_TOKENS_ENV = "STATEFRAME_CONTEXT_TOKENS"
PROMPT_BUDGET_ENV = "STATEFRAME_PROMPT_BUDGET"
CONTEXT_ENV_NAMES = (CONTEXT_TOKENS_ENV, PROMPT_BUDGET_ENV)
DEFAULT_CONTEXT_TOKENS = 65_536
DEFAULT_PROMPT_BUDGET = 33_344


# Engine switches a user may set in the shell: exactly these names and values,
# ambient over the profile. Never a blanket STATEFRAME_* passthrough.
OPT_IN_SWITCHES = {
    "STATEFRAME_ENV_SEED": ("auto", "copy"),  # host-side run environment creation policy
    "STATEFRAME_ENV_SEED_MAX_GB": (),  # non-negative decimal GB; 0 disables seed eviction
    "STATEFRAME_KEEP_RUN_ENV": ("on", "off"),  # host-side finished-run retention policy
    "STATEFRAME_MUTATIONS": ("off", "prebuilt", "authored"),
    "STATEFRAME_CACHE_LAYOUT": ("on", "off"),
    "STATEFRAME_EVICTFIX": ("on", "legacy"),
    "STATEFRAME_MICROGRAPH": ("on", "off"),  # experimental; the engine defaults to off
    "STATEFRAME_STABLE_TOOL_HEAD": ("on", "off"),  # append-only tool list per run; on by default
    "STATEFRAME_SHELL_STATE": ("on", "off"),  # shell cwd and exported variables persist between calls
    "STATEFRAME_GIT_COMMIT": ("on", "off"),  # git_commit/git_branch tools; patch series per branch after the run
    "STATEFRAME_EDIT_SCOPE_WIDEN": ("on", "off"),  # scoped runs admit an edited product file outside the scope
    "STATEFRAME_REQUIREMENT_RECHECK": ("on", "off"),  # first done answered once with the task's requirement checklist
    "STATEFRAME_FINAL_CACHE_LAYOUT": ("on", "off"),  # cacheable final request layout: volatile authority last; on by default
    "STATEFRAME_PROVIDER": ("llama.cpp",),  # operator-confirmed llama.cpp dialect override
    "STATEFRAME_STABLE_WIRE": ("on", "off"),  # append-only source windows and a split stable frame
    "STATEFRAME_STABLE_WIRE_HEADROOM": (),  # retained, not-current source window tokens per epoch
    "STATEFRAME_QUEUE_WAIT_MAX_S": (),  # finite positive seconds; a profile may set it too
    "STATEFRAME_SEND_STALL_S": (),  # 10-600 seconds without request-body progress
    "STATEFRAME_TRANSPORT_MAX_ATTEMPTS": ("1", "2", "3", "4", "5"),
    "STATEFRAME_TRANSPORT_BACKOFF_S": (),  # finite seconds between 0 and 60
    "STATEFRAME_FL_CHECKPOINT_CAPTURE": ("on", "off"),  # opt-in settled-boundary observer capture
}
# Accepted aliases -> the engine's own value (EVICTFIX "off" is sent as "legacy").
OPT_IN_ALIASES = {"STATEFRAME_EVICTFIX": {"off": "legacy"}}
# These settings are read by the host before the engine child starts; they must not enter the engine spec.
HOST_ONLY_SWITCHES = frozenset({"STATEFRAME_ENV_SEED", "STATEFRAME_KEEP_RUN_ENV"})
# Legacy unless the user opts in, on every engine that reads the switch (read eviction fix: slower on long runs
# in this release, and no measured edit gain). Applied by the runner so the default never depends on the engine's own.
OPT_IN_RUNNER_DEFAULTS = {"STATEFRAME_EVICTFIX": "legacy"}


def resolve_opt_in_switches(engine_dir: Path = ENGINE_DIR) -> dict[str, str]:
    """The shell switches set by the user, validated and forwarded in the engine spec."""
    chosen = {}
    for name in OPT_IN_SWITCHES:
        raw = os.environ.get(name, "").strip()
        if not raw:
            default = OPT_IN_RUNNER_DEFAULTS.get(name)
            if default is not None and name in engine_switches(engine_dir):
                chosen[name] = default
            continue
        if name == "STATEFRAME_ENV_SEED_MAX_GB":
            # The host seed builder reports an invalid ambient value and uses the default.
            chosen[name] = raw if re.fullmatch(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", raw) else "10"
            continue
        if name == "STATEFRAME_STABLE_WIRE_HEADROOM":
            value = _positive_int(name, raw)
            if name not in engine_switches(engine_dir):
                raise RunError(f"{name} is set, but this engine does not support it; unset it to run")
            chosen[name] = str(value)
            continue
        if name in {"STATEFRAME_QUEUE_WAIT_MAX_S", "STATEFRAME_SEND_STALL_S",
                    "STATEFRAME_TRANSPORT_BACKOFF_S"}:
            try:
                seconds = float(raw)
            except ValueError as error:
                raise RunError(f"{name} must be finite and in its supported range") from error
            if name == "STATEFRAME_SEND_STALL_S":
                valid = 10 <= seconds <= 600
            elif name == "STATEFRAME_TRANSPORT_BACKOFF_S":
                valid = 0 <= seconds <= 60
            else:
                valid = seconds > 0
            if not math.isfinite(seconds) or not valid:
                raise RunError(f"{name} must be finite and in its supported range")
            if name not in engine_switches(engine_dir):
                raise RunError(f"{name} is set, but this engine does not support it; unset it to run")
            chosen[name] = raw
            continue
        value = raw.lower()
        if name == "STATEFRAME_KEEP_RUN_ENV" and value in {"1", "true", "yes"}:
            value = "on"
        elif name == "STATEFRAME_KEEP_RUN_ENV" and value in {"0", "false", "no"}:
            value = "off"
        value = OPT_IN_ALIASES.get(name, {}).get(value, value)
        allowed = OPT_IN_SWITCHES[name]
        if value not in allowed:
            raise RunError(f"{name}={raw!r} must be {' or '.join(allowed)}; unset it to keep the default")
        if name in HOST_ONLY_SWITCHES:
            continue
        if name not in engine_switches(engine_dir):
            raise RunError(f"{name} is set, but this engine does not support it; unset it to run")
        chosen[name] = value
    capture_enabled = chosen.get("STATEFRAME_FL_CHECKPOINT_CAPTURE") == "on"
    capture_limits = {
        "STATEFRAME_FL_CAPTURE_EVERY_K": 1000,
        "STATEFRAME_FL_CAPTURE_MAX_COUNT": 1000,
        "STATEFRAME_FL_CAPTURE_MAX_FILES": 10000,
        "STATEFRAME_FL_CAPTURE_MAX_BYTES": 16 * 1024 * 1024,
    }
    for name, maximum in capture_limits.items():
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        if not capture_enabled:
            raise RunError(f"{name} requires STATEFRAME_FL_CHECKPOINT_CAPTURE=on")
        try:
            value = int(raw)
        except ValueError:
            raise RunError(f"{name} must be an integer") from None
        if value <= 0 or value > maximum:
            raise RunError(f"{name} must be between 1 and {maximum}")
        # The child environment is allowlisted from this spec, not inherited.
        chosen[name] = str(value)
    return chosen


def mutation_spec_options(options: RunOptions | None, environment: dict[str, str]) -> dict | None:
    """Keep L0 out of the run spec; CLI wins over environment and profile."""
    level = (options.mutations if options is not None and options.mutations is not None
             else environment.get("STATEFRAME_MUTATIONS", "off"))
    if level not in ("off", "prebuilt", "authored"):
        raise RunError("--mutations must be off, prebuilt or authored")
    if level == "off":
        return None
    if options is None:
        return {"level": level, "network": "none", "domains": [], "write_globs": [],
                "read_globs": [], "retain": "run", "audit": "source", "shell_network": "none"}
    if options.mutation_network not in ("none", "public-web"):
        raise RunError("--mutation-network must be none or public-web")
    if options.mutation_retain not in ("run", "project", "user"):
        raise RunError("--mutation-retain must be run, project or user")
    if options.mutation_audit not in ("source", "digest"):
        raise RunError("--mutation-audit must be source or digest")
    for item in options.mutation_hooks:
        unit, separator, event = item.partition(":")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]{1,40}", unit) or event not in ("post_tool", "session_end"):
            raise RunError("--mutation-hook must be UNIT:post_tool or UNIT:session_end")
    return {"level": level, "network": options.mutation_network,
            "domains": list(options.mutation_domains),
            "write_globs": list(options.mutation_write),
            "read_globs": list(options.mutation_read),
            "retain": options.mutation_retain, "audit": options.mutation_audit,
            "shell_network": options.tool_settings.shell_network,
            **({"hooks": list(options.mutation_hooks)} if options.mutation_hooks else {})}


def mutation_builtin_names(spec: dict) -> frozenset[str]:
    options = spec.get("mutations")
    return (frozenset({"mutate"}) if isinstance(options, dict)
            and options.get("level") in {"prebuilt", "authored"} else frozenset())


def resolve_exec_rpc_env() -> dict[str, str]:
    """Pass an explicitly configured supervisor timeout to the isolated engine child."""
    name = "STATEFRAME_EXEC_RPC_SECONDS"
    raw = os.environ.get(name)
    if raw is None:
        return {}
    value = raw.strip()
    try:
        seconds = float(value)
    except ValueError:
        seconds = 0.0
    if not math.isfinite(seconds) or seconds <= 0:
        raise RunError(f"{name}={raw!r} must be a positive number of seconds; set it or unset it")
    return {name: value}


# Every STATEFRAME_* name the product reads from the user's shell. Any other STATEFRAME_* set there never
# reaches the engine (the child drops inherited ones), so the runner says so instead of ignoring it silently.
SHELL_SETTINGS = frozenset({
    "STATEFRAME_ENDPOINT", "STATEFRAME_MODEL", "STATEFRAME_ALLOW_REMOTE_ENDPOINT", "STATEFRAME_KEEP_CHECKPOINTS",
    "STATEFRAME_CONTEXT_TOKENS", "STATEFRAME_PROMPT_BUDGET", "STATEFRAME_ROUTES",
    "STATEFRAME_EXEC_RPC_SECONDS", "STATEFRAME_STARTUP_PROBE_SECONDS",
    *CHECKPOINT_ENV, *OPT_IN_SWITCHES,
})


def ignored_shell_settings(environ=None) -> list[str]:
    environ = os.environ if environ is None else environ
    return sorted(name for name in environ if name.startswith("STATEFRAME_") and name not in SHELL_SETTINGS
                  and not (name.startswith("STATEFRAME_ROUTE_") and name not in
                           {"STATEFRAME_ROUTE_SCOPE", "STATEFRAME_ROUTE_UNIT_CLASS", "STATEFRAME_ROUTE_SLOT"}))


def derived_prompt_budget(context_tokens: int) -> int:
    """The prompt budget for a slot when only the slot is set: the default ratio, rounded down."""
    return context_tokens * DEFAULT_PROMPT_BUDGET // DEFAULT_CONTEXT_TOKENS


def _positive_int(name: str, raw: str) -> int:
    try:
        value = int(raw.strip().replace(",", "").replace("_", ""))
    except ValueError:
        value = 0
    if value <= 0:
        raise RunError(f"{name}={raw!r} is not a positive whole number of tokens; set it to e.g. 65536 or unset it")
    return value


def _engine_endpoint_size(values: dict[str, str], engine_dir: Path = ENGINE_DIR):
    """The engine's resolve_endpoint_size() on exactly these values, or None when the engine has no resolver.
    Loaded by file path (stdlib only); its ValueError becomes the user-facing refusal."""
    source = engine_dir / "kernel_endpoint_size.py"
    if not source.is_file():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("_sf_endpoint_size", source)
    module = importlib.util.module_from_spec(spec)
    saved = {name: os.environ.pop(name) for name in CONTEXT_ENV_NAMES if name in os.environ}
    sys.modules[spec.name] = module  # dataclasses resolve their module through sys.modules while loading
    try:
        spec.loader.exec_module(module)  # its import-time default must not read the user's shell values
        return module.resolve_endpoint_size(values)
    except ValueError as error:
        raise RunError(str(error)) from None
    except Exception as error:  # an engine resolver that exists but cannot run must not fall back silently
        raise RunError(f"could not load the engine's context-size rules ({type(error).__name__}: {error}); "
                       f"unset {CONTEXT_TOKENS_ENV} and {PROMPT_BUDGET_ENV} to run with the default sizes") from None
    finally:
        sys.modules.pop(spec.name, None)
        os.environ.update(saved)


def resolve_context(profile_env: dict[str, str] | None = None,
                    engine_dir: Path = ENGINE_DIR) -> tuple[int, int, str, dict[str, str]]:
    """``(slot tokens, prompt budget, how the budget was chosen, env to hand the engine)``.

    With neither name set anywhere, the defaults apply and nothing is handed over (the engine's own
    defaults are the same values). A set name needs an engine that reads it."""
    profile_env = profile_env or {}
    raw = {name: os.environ.get(name, "").strip() or str(profile_env.get(name, "")).strip()
           for name in CONTEXT_ENV_NAMES}
    if not any(raw.values()):
        return DEFAULT_CONTEXT_TOKENS, DEFAULT_PROMPT_BUDGET, "default", {}
    # Supported only when the engine reads BOTH names (its context-size resolver). An older engine reads
    # STATEFRAME_CONTEXT_TOKENS with a different, frame-only meaning (kernel_model_profile), so handing it the
    # slot would change frame sizes without changing the prompt budget: refuse instead.
    unread = [name for name, value in raw.items() if value]
    if not set(CONTEXT_ENV_NAMES) <= engine_switches(engine_dir):
        raise RunError(f"{', '.join(unread)} is set, but this engine has a fixed context size "
                       f"({DEFAULT_CONTEXT_TOKENS:,}-token slot, {DEFAULT_PROMPT_BUDGET:,}-token prompt budget); "
                       "unset it to run with those sizes")
    engine_size = _engine_endpoint_size({name: value for name, value in raw.items() if value}, engine_dir)
    if engine_size is not None:
        # The engine's own resolver is the one rule (and its refusal text is what the user sees either way).
        slot, budget = engine_size.context_tokens, engine_size.prompt_budget
        how = ("budget set explicitly" if engine_size.budget_explicit
               else f"budget derived from slot at {DEFAULT_PROMPT_BUDGET}/{DEFAULT_CONTEXT_TOKENS}")
        handed = {name: value for name, value in raw.items() if value}
        return slot, budget, how, handed
    slot = _positive_int(CONTEXT_TOKENS_ENV, raw[CONTEXT_TOKENS_ENV]) if raw[CONTEXT_TOKENS_ENV] \
        else DEFAULT_CONTEXT_TOKENS
    if raw[PROMPT_BUDGET_ENV]:
        budget, how = _positive_int(PROMPT_BUDGET_ENV, raw[PROMPT_BUDGET_ENV]), "budget set explicitly"
        if budget >= slot:
            raise RunError(f"{PROMPT_BUDGET_ENV}={budget} is not smaller than the {slot:,}-token slot; the rest "
                           f"of the slot holds the model's reply. Lower it (default for this slot: "
                           f"{derived_prompt_budget(slot)}) or unset it")
    else:
        budget = derived_prompt_budget(slot)
        how = f"budget derived from slot at {DEFAULT_PROMPT_BUDGET}/{DEFAULT_CONTEXT_TOKENS}"
    handed = {name: str(value) for name, value in ((CONTEXT_TOKENS_ENV, slot), (PROMPT_BUDGET_ENV, budget))
              if raw[name]}
    return slot, budget, how, handed


def _tests_python(options: RunOptions) -> str:
    """The interpreter that runs the repository's tests (set by _select_test_python / resume)."""
    if options._test_python is None:
        raise RunError("internal: test interpreter was not selected before running checks")
    return options._test_python


def use_run_env_python(options: RunOptions, run_env: object) -> None:
    """Bind both host checks and the engine child to a created run environment.

    The tool-settings owner calls this after ``prepare`` and before ``_execute``;
    ``drive`` forwards the same selection as ``STATEFRAME_TEST_PYTHON``.
    """
    candidate = getattr(run_env, "python", None)
    if not isinstance(candidate, Path) or not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RunError("run environment has no executable Python interpreter")
    options._test_python = str(candidate)
    options._capabilities["test_python"] = str(candidate)


def _check_engine_python(options: RunOptions) -> None:
    """Check the engine interpreter; a pytest runner's probe also proves pytest imports from it."""
    pytest_runner = options.runner_kind != "generic"
    options.python = _check_python(options.python, require_pytest=pytest_runner)
    options._pytest_proven = options.python if pytest_runner else None


def _select_test_python(options: RunOptions, printer=print) -> str:
    """Choose and check the interpreter for the repository's tests; never StateFrame's own silently."""
    try:
        chosen, reason = test_interpreter.choose(options.repo, test_python=options.test_python,
                                                 python=options.python if options.python_explicit else None)
        python = preflight.resolve_interpreter(chosen, os.getcwd())
    except (test_interpreter.TestInterpreterError, preflight.PreflightError) as error:
        raise RunError(str(error)) from None
    # The engine probe ran "import pytest" from this exact path under the same contained capability,
    # so a second probe proves nothing new; a different path (even a symlink to the same binary,
    # with its own site-packages) is probed.
    if options.runner_kind != "generic" and python != options._pytest_proven:
        _startup_probe(python, "import pytest", label=f"test interpreter {python} ({reason})")
    from .engine.kernel_test_isolation import check_sandbox_exec_denial
    denial = check_sandbox_exec_denial(python)
    if denial:
        raise RunError(f"test interpreter {python} ({reason}): {denial}. Every check would fail "
                       "to start; check that the interpreter and its links are available and "
                       "executable inside the check sandbox, or report this as a StateFrame bug")
    printer(f"test interpreter: {python} ({reason})")
    if not engine_reads_test_python():
        # Honest scope: --test-cmd before/after use it; this engine's own checks do not yet.
        printer("  note: this engine runs its own in-run checks with StateFrame's interpreter; "
                "the before/after checks use the one above")
    options._capabilities["test_python"] = python
    options._capabilities["test_python_reason"] = reason
    return python


_CAPABILITY_PROBE = r'''
import contextlib
import importlib
import importlib.metadata
import io
import json

def check(module, distribution, attribute=None):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            loaded = importlib.import_module(module)
            if attribute is not None and not callable(getattr(loaded, attribute, None)):
                return {"enabled": False, "detail": distribution + " import API unavailable"}
            version = getattr(loaded, "__version__", None)
            if not isinstance(version, str):
                try:
                    version = importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:
                    version = "version unknown"
        return {"enabled": True, "detail": distribution + " " + version[:80]}
    except ImportError:
        return {"enabled": False, "detail": distribution + " missing"}
    except Exception as error:
        return {"enabled": False, "detail": distribution + " import failed: " + type(error).__name__}

schema = check("jsonschema", "jsonschema")
yaml = check("yaml", "PyYAML")
try:  # the product's per-language grammar packages, exposed under the pack's module name
    from stateframe import ts_grammars
    adapter = ts_grammars.install_as_language_pack()
except Exception:
    adapter = False
repo_map = check("tree_sitter_language_pack", "tree-sitter-language-pack", "get_parser")
if adapter:
    working = []
    for name in ts_grammars.available_languages():
        try:
            ts_grammars.get_parser(name).parse(b"x")
            working.append(name)
        except Exception:
            pass
    repo_map = ({"enabled": True, "detail": f"{len(working)} languages: " + ", ".join(working)} if working
                else {"enabled": False, "detail": "no grammar package installed"})
if not repo_map["enabled"]:
    # Only the audited individual grammar packages are ever used (ts_grammars blocks both grammar bundles).
    repo_map["detail"] += "; text-scan fallback available"
print(json.dumps({"schema_validation": schema, "yaml": yaml, "repo_map_ast": repo_map}))
'''


def _probe_capabilities(options: RunOptions) -> dict[str, object]:
    """Import-only receipt from the selected interpreter; never inspect the task tree."""
    labels = {"schema_validation": "schema validation", "yaml": "yaml", "repo_map_ast": "repo map AST"}
    try:
        # -P omits cwd from imports without hiding user-site packages that the
        # engine child can use. The allowlisted environment also drops PYTHONPATH.
        probe = subprocess.run([options.python, "-P", "-B", "-c", _CAPABILITY_PROBE],
                               cwd=PACKAGE, env={k: v for k, v in os.environ.items() if k in PASS_ENV},
                               capture_output=True, text=True, check=False,
                               timeout=min(10, _remaining(options)))
        if probe.returncode != 0:
            raise ValueError("probe exited unsuccessfully")
        receipt = json.loads(probe.stdout)
        if not isinstance(receipt, dict) or any(
            not isinstance(receipt.get(key), dict)
            or type(receipt[key].get("enabled")) is not bool
            or not isinstance(receipt[key].get("detail"), str)
            for key in labels
        ):
            raise ValueError("invalid capability receipt")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        receipt = {key: {"enabled": None, "detail": "probe unavailable: " + type(error).__name__}
                   for key in labels}
    parts = []
    for key, label in labels.items():
        item = receipt[key]
        state = "on" if item["enabled"] is True else "OFF" if item["enabled"] is False else "UNKNOWN"
        # Keep unexpected dependency metadata from injecting extra terminal lines.
        item["detail"] = " ".join(item["detail"].split())[:240]
        parts.append(f"{label} {state} ({item['detail']})")
    return {**receipt, "python": options.python, "display": "capabilities: " + ", ".join(parts)}


def engine_supports_unbounded_steps() -> bool:
    """True when the bundled engine implements a 0 step limit as "no step limit".

    Either signal counts: BUILD's STEPS_UNBOUNDED capability marker in the loop module, or the engine's own
    kernel_step_limits.parse_step_limit, verified by calling it (0 must map to None, not to zero fuel).
    """
    loop = ENGINE_DIR / "probes" / "react_repair.py"
    try:
        if "STEPS_UNBOUNDED" in loop.read_text(encoding="utf-8"):
            return True
    except OSError:
        pass
    limits = ENGINE_DIR / "kernel_step_limits.py"
    if not limits.is_file():
        return False
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("_sf_step_limits_probe", limits)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.parse_step_limit(0) is None and module.parse_step_limit("unbounded") is None
    except Exception:  # an unreadable or older module is not support
        return False


def profile_steps(profile: str) -> int:
    """The step limit is one switch: the profile's REACT_STEPS."""
    return int(profiles.load(profile)["REACT_STEPS"])


def _write_run_spec_atomic(run_dir: Path, spec: dict, *, mode: int | None = None) -> None:
    """Publish a spec atomically without changing its existing privacy mode."""
    path = run_dir / "spec.json"
    _write_json_atomic(path, spec, mode=mode)


def _write_hook_spec(run_dir: Path, spec: dict) -> None:
    """Publish hook state atomically after its durable tape records are complete."""
    path = run_dir / "spec.json"
    try:
        _write_run_spec_atomic(run_dir, spec, mode=0o600)
    except OSError as error:
        raise RunError(f"cannot publish hook state to {path}: {error}") from error


def _write_hook_terminal_summary(run_dir: Path, *, runtime_outcome: str, engine_result: str,
                                 refusal: str | None = None, containment_error: str | None = None,
                                 printer=print) -> str | None:
    """Persist an outcome that stops before the normal engine summary path."""
    from . import hook_host

    try:
        counts = hook_host.hook_record_summary(run_dir)
    except hook_host.HookRecordsError as error:
        containment_error = containment_error or str(error)
        counts = {"runs": 0, "denied": 0, "vetoes": 0, "errors": 1,
                  "last_error": containment_error}
        printer(f"hooks: records error: {json.dumps(str(error), ensure_ascii=False)}")
    if containment_error and counts["last_error"] != containment_error:
        counts["errors"] = int(counts["errors"]) + 1
        counts["last_error"] = containment_error
    summary = {
        "run_dir": str(run_dir),
        "runtime_outcome": "hook_containment" if containment_error else runtime_outcome,
        "engine_result": "HOOK_CONTAINMENT" if containment_error else engine_result,
        "hook_refusal": refusal,
        "hook_containment": containment_error,
        "check_outcome": "not_run",
        "artifact": "none",
        "exit_code": 2,
        "succeeded": False,
        "hooks": counts,
    }
    _write_json_atomic(run_dir / "summary.json", summary)
    printer(f"hooks: {counts['runs']} runs, {counts['denied']} denied, "
            f"{counts['vetoes']} vetoes, {counts['errors']} errors")
    return containment_error


def _finish_prompt_refusal(options: RunOptions, run_dir: Path, spec: dict,
                           refusal: str, printer) -> str | None:
    """Run terminal hooks and persist their counts after a pre-engine refusal."""
    from . import __version__, hook_host

    containment_error = None
    try:
        hook_host.run_session_end(
            run_dir,
            spec["hooks"],
            run_dir / "workspace",
            data={
                "outcome": "hook_refused",
                "stop_reason": "prompt_submit",
                "applied": False,
                "changed_paths": [],
                "run_dir": str(run_dir),
            },
            shell_network=options.tool_settings.shell_network,
            run_seed=str(spec["run_identity_seed"]),
            run_id=str(spec["run_identity_seed"]),
            version=__version__,
            printer=printer,
        )
    except (hook_host.HookContainmentError, hook_host.HookConfigError, OSError, ValueError) as error:
        containment_error = str(error)
        printer(f"hooks: containment: {json.dumps(containment_error, ensure_ascii=False)}")

    result = "HOOK_CONTAINMENT" if containment_error else (
        "HOOK_FAILED" if refusal.startswith("HOOK_FAILED:") else "HOOK_DENIED")
    containment_error = _write_hook_terminal_summary(
        run_dir,
        runtime_outcome="hook_containment" if containment_error else "hook_refused",
        engine_result=result,
        refusal=refusal,
        containment_error=containment_error,
        printer=printer,
    )
    return containment_error


def prepare(options: RunOptions, printer=print) -> tuple[Path, workcopy.WorkCopy, dict]:
    explicit_test_protection = options.protect_tests is not None
    try:
        options.runner_kind, test_argv = parse_command(options.test_cmd, options.runner_kind)
        protected_globs = validate_patterns(options.protected_test_globs)
        if options.runner_kind == "generic" and not test_argv and options.check_source != "none":
            raise ValueError("a generic runner requires --test-cmd")
        if options.runner_kind == "pytest" and protected_globs:
            raise ValueError("--protected-test-glob requires a generic runner")
    except ValueError as error:
        raise RunError(str(error)) from error
    _validate_timeout(options.timeout_s)
    _remaining(options)
    environment = profiles.load(options.profile)
    _apply_operator_reasoning_effort(environment)
    if options._routes is not None:
        routed_keys = {settings.get("api_key_env") for settings in options._routes.roles.values()}
        stored_names = set(environment) | set(options.extra_env) | set(options.test_env) | set(options._context_env)
        collision = sorted(name for name in stored_names if name in routed_keys)
        if collision:
            raise RunError(f"{collision[0]} is a routed api_key_env; remove that spec environment value")
    options.protect_tests = options.protect_tests or environment.get("STATEFRAME_PROTECT_TESTS", "all")
    if options.protect_tests not in {"task", "all", "off"}:
        raise RunError("--protect-tests must be task, all, or off")
    environment.pop("STATEFRAME_PROTECT_TESTS", None)
    if options.max_turns is None:
        options.max_turns = profile_steps(options.profile)
    if options.max_turns == 0:
        # 0 = no step limit, passed through unchanged as the engine's REACT_STEPS=0. Only an engine that
        # implements it may receive it: on an older engine 0 would mean zero turns, not unlimited.
        if not engine_supports_unbounded_steps():
            raise RunError("--max-turns 0 (no step limit) needs an engine with unbounded steps; "
                           "this build's engine does not support it")
    elif options.max_turns < 1 or options.max_turns > 500:
        raise RunError("--max-turns must be 0 (no limit) or between 1 and 500")
    if not ENGINE_DIR.is_dir():
        raise RunError("engine directory missing from the package; reinstall stateframe")
    from . import hook_host, powerups
    from .powerup_prepare import prepare_run_inputs
    user_tool_count = len(options.tool_settings.tools_files)
    try:
        powerup_selection = prepare_run_inputs(options, environment, printer)
    except (powerups.PowerupError, SkillError, CustomToolError, OSError, ValueError) as error:
        raise RunError(str(error)) from error
    try:
        selected_skills = resolve_skills(options.skills, trusted=options.trust_skill_tools)
    except SkillError as error:
        raise RunError(str(error)) from error
    if selected_skills and str(options.extra_env.get("STATEFRAME_AGENT_LOOP", environment.get("STATEFRAME_AGENT_LOOP", "full"))).lower() == "plain":
        raise RunError("--skill requires the composed loop; plain loop does not support skills")
    if selected_skills and str(options.extra_env.get("STATEFRAME_LEAN", environment.get("STATEFRAME_LEAN", "off"))).lower() == "on":
        raise RunError("--skill requires the composed loop; lean mode does not support skills")
    records = [skill.record(trusted=options.trust_skill_tools) for skill in selected_skills]
    measurements = None
    if selected_skills:
        # Admission is host-owned and recorded before the child builds its first request.
        sys.path.insert(0, str(ENGINE_DIR))
        try:
            from .engine.kernel_request_diet import RequestTokenMeter
            from .engine.kernel_skills import SkillRuntime, measure_admission
            meter = RequestTokenMeter(
                endpoint=options.endpoint.chat_url, template_overhead_tokens=0,
                auth_key=options.endpoint.api_key(), generation_endpoint=options.endpoint.chat_url,
            )
            measurements = measure_admission(selected_skills, records, meter)
            digest = measured_skills_digest(records, measurements)
            SkillRuntime(records, digest, options.endpoint.chat_url, measurements,
                         auth_key=options.endpoint.api_key(),
                         generation_endpoint=options.endpoint.chat_url)
        except SkillError as error:
            raise RunError(str(error)) from error
        finally:
            sys.path.remove(str(ENGINE_DIR))
    settings = options.tool_settings
    if options.trust_skill_tools:
        skill_files = tuple({"path": str(skill.path / "tools.json"), "digest": "pending"}
                            for skill in selected_skills if skill.tools_json_sha256)
        settings = replace(settings, tools_files=(*settings.tools_files[:user_tool_count],
                                                *skill_files, *settings.tools_files[user_tool_count:]))
    builtin_names = frozenset(("grep", "list_files", "read", "edit", "str_replace", "insert_before",
        "authored_check", "file_create", "run_tests", "read_check_receipt", "recall", "done",
        "shell", "job_wait", "job_output", "job_stop", "file_delete", "file_move", "git", "web_fetch", "git_commit", "git_branch", "skill_load"))
    mutation_environment = dict(environment)
    if "STATEFRAME_MUTATIONS" in os.environ:
        mutation_environment["STATEFRAME_MUTATIONS"] = os.environ["STATEFRAME_MUTATIONS"].strip().lower()
    mutation_environment.update(options.extra_env)
    mutation_options = mutation_spec_options(options, mutation_environment)
    builtin_names |= mutation_builtin_names({"mutations": mutation_options})
    try:
        custom = load_tool_files([Path(item["path"]) for item in settings.tools_files],
                                 builtin_names=builtin_names, shell_network=settings.shell_network)
        mcp_servers = load_mcp_files([Path(item["path"]) for item in settings.mcp_files],
                                     shell_network=settings.shell_network)
    except CustomToolError as error:
        raise RunError(str(error)) from error
    files = tuple({"path": str(Path(item["path"]).resolve()),
                   "digest": hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()}
                  for item in settings.tools_files)
    mcp_files = tuple({"path": str(Path(item["path"]).resolve()),
                       "digest": hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()}
                      for item in settings.mcp_files)
    settings = replace(settings, tools_files=files, mcp_files=mcp_files)
    options.tool_settings = settings
    hook_repo = (options.hooks_repo or options.repo).resolve()
    if options.project_hooks_resolved:
        project_hooks = hook_host.ProjectHookSelection(
            repo=hook_repo,
            hooks=options.project_hooks,
            digest=options.project_hooks_digest,
            message=None,
        )
    else:
        try:
            project_hooks = hook_host.resolve_project_hooks(
                hook_repo,
                shell_network=settings.shell_network,
                no_project_hooks=options.no_project_hooks,
            )
        except (hook_host.HookTrustError, hook_host.HookConfigError) as error:
            raise RunError(str(error)) from error
        options.project_hooks = project_hooks.hooks
        options.project_hooks_digest = project_hooks.digest
        options.project_hooks_resolved = True
        if project_hooks.message:
            printer(project_hooks.message)
    try:
        hooks = hook_host.load_hooks(
            options.hooks_files,
            repo=hook_repo,
            shell_network=settings.shell_network,
            project=project_hooks,
        )
    except (hook_host.HookTrustError, hook_host.HookConfigError) as error:
        raise RunError(str(error)) from error
    run = (options.run_dir or default_run_dir(options.repo)).resolve()
    if run.exists():
        raise RunError(f"run directory already exists: {run}")
    # Checked BEFORE anything is created: a run never writes into the user's repository.
    if run.is_relative_to(options.repo.resolve()):
        raise RunError(f"--run-dir must be outside the repository: {run}")
    if hooks is not None:
        try:
            hook_host.refuse_run_dir_under_roots(run, hooks)
        except hook_host.HookConfigError as error:
            raise RunError(str(error)) from error
    if options.run_dir is None:
        private_directory(run.parent.parent, "runs", run.name)
    else:
        run.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        private_directory(run)
    work = workcopy.create(options.repo, run / "workspace")
    instructions_record = _resolve_project_instructions(options, work.copy, environment)
    # Read-only reference copy of the starting state; the engine's finish path reads it, and
    # diffs are taken against it (never against the live original, which may change meanwhile).
    shutil.copytree(work.copy, run / "parent", symlinks=False)
    work.snapshot = run / "parent"
    row = build_row(work, task=options.task, checks=options.checks or None, target=options.target,
                     edit_paths=options.edit_paths or None, runner_kind=options.runner_kind,
                     test_command=test_argv, protected_test_globs=protected_globs,
                     protect_tests=options.protect_tests)
    if row.get("scoped_edits") and not options.target and not options.edit_paths:
        chosen = sorted(row["files"])
        printer(f"edit scope: {len(chosen)} file(s) chosen from the task and its checks "
                f"(pass --target/--edit to choose): {', '.join(chosen)}")
    if explicit_test_protection:
        row["test_wrong_wire"] = True
    if options.runner_kind == "generic":
        row["check_source"] = options.check_source
    try:
        mcp = discover_mcp(mcp_servers, work.copy, run,
                           shell_network=settings.shell_network,
                           host_authorization=settings.host_authorization is not None,
                           custom_schemas=custom.schemas(), endpoint=options.endpoint.chat_url,
                           auth_key=options.endpoint.api_key(),
                           generation_endpoint=options.endpoint.chat_url) if mcp_servers else None
    except (CustomToolError, OSError, ValueError) as error:
        raise RunError(f"MCP discovery failed: {error}") from error
    environment.update({
        "DS_URL": options.endpoint.chat_url,
        "DS_MODEL": options.endpoint.model,
        "REACT_STEPS": str(options.max_turns),
        "PYTHONHASHSEED": environment.get("PYTHONHASHSEED", "0"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if not (ENGINE_DIR / "kernel_checkpoint.py").is_file():
        # An engine without checkpoints refuses STATEFRAME_CHECKPOINT* settings it cannot read: drop the
        # profile's too (checkpoint_env() already returns none for it).
        for key in CHECKPOINT_ENV:
            environment.pop(key, None)
    environment.update(checkpoint_env(environment))
    # Repo-intel caches live in this attempt's worker dir, never in ~/.cache: a cache shared across
    # processes made resume skip parses the original attempt did, so the rebuilt journal and request
    # bytes diverged and replay went live early. resume archives worker/, so each attempt starts cold.
    environment["STATEFRAME_REPO_CACHE_DIR"] = str(run / "worker" / "repo-intel-cache")
    settings_file = run / "tool-settings.json"
    _write_json_atomic(settings_file, settings.as_dict(), mode=0o600)
    skills_file = run / "skill-settings.json"
    skills_record = {"schema": "stateframe-skill-settings/1", "skills": records,
                     "skills_digest": (measured_skills_digest(records, measurements)
                                       if measurements is not None else skills_digest(records)),
                     **({"measurements": measurements} if measurements is not None else {})}
    _write_json_atomic(skills_file, skills_record, mode=0o600)
    environment.update(options._context_env)
    environment.update(resolve_exec_rpc_env())
    opt_in = resolve_opt_in_switches()
    environment.update(opt_in)
    environment.update(options.extra_env)
    if mutation_options is not None:
        environment["STATEFRAME_MUTATIONS"] = mutation_options["level"]
    else:
        environment.pop("STATEFRAME_MUTATIONS", None)
    # Host authority wins over every caller-supplied environment overlay.
    environment["STATEFRAME_TOOL_SETTINGS_FILE"] = str(settings_file)
    environment["STATEFRAME_SKILL_SETTINGS_FILE"] = str(skills_file)
    environment.pop("SF_PROMPT_CACHE_PIN_DEGRADED_REASON", None)
    if options._cache_slot_degraded:
        # Prompt-cache reuse is optional unless an explicit slot was requested. With
        # no slot proof, disable it so secondary cache-enabled bodies also degrade.
        environment["STATEFRAME_PROMPT_CACHE"] = "off"
        environment["SF_PROMPT_CACHE_PIN_DEGRADED_REASON"] = (
            options._cache_slot_reason or "provider did not prove llama.cpp /slots"
        )
    # Slot selection is a runner-owned contract. A stale ambient value or an
    # extra environment entry must not pin a slot after discovery found none.
    environment.pop("REACT_SLOT_ID", None)
    selected_slot = options._cache_slot if options._cache_slot_selection is not None else options.slot
    if selected_slot is not None:
        environment["REACT_SLOT_ID"] = str(selected_slot)
    test_env = validate_test_env(options.test_env)
    if test_env:
        # The engine captures these at import as host task policy: the model cannot change them.
        environment.update(test_env)
        environment["STATEFRAME_TASK_ENV_KEYS"] = ",".join(sorted(test_env))
    if options.pytest_plugins:
        if options.pytest_plugins != "installed":
            raise RunError("--pytest-plugins only accepts 'installed'")
        environment["STATEFRAME_TASK_PYTEST_PLUGINS"] = "installed"
    environment_sources = {k: "profile:" + options.profile for k in environment}
    environment_sources["STATEFRAME_TOOL_SETTINGS_FILE"] = "runner:host-tool-settings"
    environment_sources["STATEFRAME_SKILL_SETTINGS_FILE"] = "runner:host-skill-settings"
    for name in opt_in:
        environment_sources[name] = ("ambient:" + name if os.environ.get(name, "").strip()
                                     else "runner:default-" + OPT_IN_RUNNER_DEFAULTS[name])
    if options._cache_slot_degraded:
        environment_sources["STATEFRAME_PROMPT_CACHE"] = "runner:optional-pin-degraded"
        environment_sources["SF_PROMPT_CACHE_PIN_DEGRADED_REASON"] = "runner:provider-capability-degradation"
    if selected_slot is not None:
        environment_sources["REACT_SLOT_ID"] = "runner:" + (options._cache_slot_selection or "explicit")
    provider_records = _provider_preflight_records(options)
    spec = {
        "schema": "stateframe-run/1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_dir": str(ENGINE_DIR),
        "engine_source": profiles.engine_source_summary(),
        "profile": options.profile,
        "capabilities": options._capabilities,
        **({"provider_capabilities": provider_records} if provider_records else {}),
        "chat_url": options.endpoint.chat_url,
        "model": options.endpoint.model,
        "api_key_env": options.endpoint.api_key_env,
        "max_turns": options.max_turns,
        "parent_snapshot": str(run / "parent"),
        "public_row": row,
        "tool_settings": settings.as_dict(),
        "custom_tools_digest": custom.digest,
        "custom_executables": {tool.name: tool.executable_digest for tool in custom.tools},
        "skills": records,
        "skills_digest": skills_record["skills_digest"],
        **({"project_instructions": instructions_record} if instructions_record is not None else {}),
        **({"skills_measurements": measurements} if measurements is not None else {}),
        "environment": environment,
        "environment_sources": environment_sources,
        "prompt_cache_slot": {
            "slot": selected_slot,
            "selection": (options._cache_slot_selection or
                          ("explicit" if options.slot is not None else "not_selected")),
            **({"degraded": True} if options._cache_slot_degraded else {}),
            "reason": (options._cache_slot_reason or
                       ("requested by --slot" if options.slot is not None else
                        "slot discovery has not run")),
        },
        "test_cmd": options.test_cmd,
        "check_cmd": options.check_cmd,
        "check_source": options.check_source,
        "runner_kind": options.runner_kind,
        "protected_test_globs": list(protected_globs),
        "protect_tests": options.protect_tests,
        # The engine's run-scoped ids derive from this (see _child.install_run_identity), so a resume can
        # rebuild the run's requests byte for byte.
        "run_identity_seed": secrets.token_hex(16),
        "repo": str(options.repo.resolve()),
        "in_place": options.in_place,
        **({"mutations": mutation_options} if mutation_options is not None else {}),
    }
    if powerup_selection.active or powerup_selection.skipped:
        spec["powerups"] = powerup_selection.record()
    if options._routes is not None:
        spec["routes"] = options._routes.public()
    if mcp is not None:
        spec.update(mcp_tools_digest=mcp.digest, mcp_omitted=list(mcp.omitted),
                    mcp_tools=[tool.name for tool in mcp.tools])
    if hooks is not None:
        try:
            manifest = hook_host.create_bundle(hooks, run / "hook-bundle")
            hooks_spec = hooks.as_spec(manifest)
            # Persist the pinned definitions before invoking any hook so a crash can resume
            # from the durable runner tape instead of silently skipping or repeating a hook.
            spec["hooks"] = hooks_spec
            _write_hook_spec(run, spec)
            printer(f"hooks: {len(hooks.definitions)} loaded from {len(hooks.files)} files")
            from . import __version__
            hooks_spec["start_context"] = hook_host.run_start_hooks(
                run,
                hooks_spec,
                run / "workspace",
                task=options.task,
                repo=hook_repo,
                shell_network=settings.shell_network,
                run_seed=spec["run_identity_seed"],
                run_id=spec["run_identity_seed"],
                version=__version__,
                printer=printer,
            )
            _write_hook_spec(run, spec)
        except hook_host.HookRunRefused as error:
            containment = _finish_prompt_refusal(options, run, spec, str(error), printer)
            if containment:
                raise HookContainmentRunError(f"HOOK_CONTAINMENT: {containment}") from error
            result = "HOOK_FAILED" if str(error).startswith("HOOK_FAILED:") else "HOOK_DENIED"
            raise HookRefusedRunError(f"hook refused run: {error}", result=result) from error
        except hook_host.HookContainmentError as error:
            detail = str(error)
            _write_hook_terminal_summary(
                run,
                runtime_outcome="hook_containment",
                engine_result="HOOK_CONTAINMENT",
                containment_error=detail,
                printer=printer,
            )
            raise HookContainmentRunError(f"HOOK_CONTAINMENT: {error}") from error
        except hook_host.HookConfigError as error:
            raise RunError(str(error)) from error
    if hooks is None:
        _write_json_atomic(run / "spec.json", spec)
    return run, work, spec


def _child_env(options: RunOptions) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in PASS_ENV}
    env.update({k: v for k, v in os.environ.items() if k in USER_ENV and v.strip()})
    if options.endpoint is not None and options.endpoint.api_key_env:
        env[options.endpoint.api_key_env] = options.endpoint.api_key()
    if options._routes is not None:
        for settings in options._routes.roles.values():
            name = settings.get("api_key_env")
            if isinstance(name, str):
                env[name] = os.environ[name]
    selected_slot = options._cache_slot if options._cache_slot_selection is not None else options.slot
    if selected_slot is not None:
        env["REACT_SLOT_ID"] = str(selected_slot)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _slot_id(entry: dict) -> int | None:
    slot_id = entry.get("id")
    if type(slot_id) is not int or slot_id < 0:
        slot_id = entry.get("id_slot")
    return slot_id if type(slot_id) is int and slot_id >= 0 else None


def _idle_slot(slots: object) -> tuple[int | None, bool]:
    """Return the lowest uniquely reported idle slot and whether status was usable."""
    if not isinstance(slots, list):
        return None, False
    by_id: dict[int, bool] = {}
    duplicates: set[int] = set()
    for entry in slots:
        if not isinstance(entry, dict):
            continue
        slot_id = _slot_id(entry)
        processing = entry.get("is_processing")
        if slot_id is None or type(processing) is not bool:
            continue
        if slot_id in by_id:
            duplicates.add(slot_id)
        else:
            by_id[slot_id] = processing
    usable = not slots or bool(by_id)
    idle = sorted(slot_id for slot_id, processing in by_id.items()
                   if not processing and slot_id not in duplicates)
    return (idle[0] if idle else None), usable


REQUIRED_SLOT_TOKENS = 65_536  # the default slot; a run uses options._context_tokens


def check_slot_context(options: RunOptions, printer=print) -> None:
    """Refuse a server whose context slots are smaller than the prompt budget needs.

    Only a server that reports per-slot ``n_ctx`` on ``/slots`` (llama.cpp) is checked; one that does not
    expose slots is not refused (the requirement stays in the banner and docs).
    """
    assert options.endpoint is not None
    capabilities = _provider_capabilities_for(
        options.endpoint.chat_url, options.endpoint.api_key(), options,
    )
    if not capabilities.slots:
        return
    endpoint = replace(options.endpoint, timeout_s=min(options.endpoint.timeout_s, _remaining(options)))
    try:
        slots = endpoint.get_slots()
    except (OSError, TimeoutError, EndpointError, ValueError):
        return
    sizes = [s["n_ctx"] for s in slots or [] if isinstance(s, dict) and type(s.get("n_ctx")) is int]
    need = options._context_tokens
    if sizes and min(sizes) < need:
        raise RunError(f"the server's context slot holds {min(sizes):,} tokens; StateFrame needs "
                       f"{need:,} per slot. Restart the server with a larger context "
                       f"(llama.cpp: --ctx-size = {need} x --parallel), or set {CONTEXT_TOKENS_ENV} to the "
                       "slot size your server has.")
    # With llama.cpp --kv-unified every slot reports the TOTAL cache as its n_ctx, so N slots that each
    # report exactly the per-run size may be sharing one cache of that size: concurrent runs then overflow.
    # /slots cannot tell the two layouts apart, so warn instead of refusing a correctly sized server.
    if len(sizes) > 1 and len(set(sizes)) == 1 and sizes[0] < need * len(sizes):
        printer(f"note: the server reports {len(sizes)} slots of {sizes[0]:,} tokens. That is fine if each slot has "
                f"its own cache. Only if the server uses --kv-unified is it undersized: then all slots share one "
                f"{sizes[0]:,}-token cache and runs at the same time can overflow it; use "
                f"--ctx-size = {need} x --parallel ({need * len(sizes):,} here).")


def _select_prompt_cache_slot(options: RunOptions, printer=print) -> None:
    """Use an explicit slot or pin one reported idle by a local llama.cpp server."""
    options._cache_slot = None
    options._cache_slot_selection = None
    options._cache_slot_reason = None
    options._cache_slot_degraded = False
    assert options.endpoint is not None
    from .engine.kernel_provider_dialect import require_feature
    capabilities = _provider_capabilities_for(
        options.endpoint.chat_url, options.endpoint.api_key(), options,
    )
    if options.slot is not None:
        if type(options.slot) is not int or options.slot < 0:
            raise RunError("--slot must be a nonnegative integer")
        require_feature(
            capabilities, "prompt-cache pinned slot", slots=True, slot_id=options.slot,
        )
        slot = options.slot
        selection = "explicit"
        reason = "requested by --slot"
    else:
        if not capabilities.slots:
            options._cache_slot_selection = "not_pinned"
            if capabilities.dialect == "llama.cpp":
                # A proven llama.cpp without slot status was unpinned before dialect
                # probing; no optional pin or prompt-cache mode has been degraded.
                options._cache_slot_reason = "server did not expose usable slot status"
                printer("cache: server does not expose slots; not pinned")
            else:
                options._cache_slot_degraded = True
                options._cache_slot_reason = (
                    "provider did not prove llama.cpp /slots; optional prompt-cache pin dropped"
                )
                printer("cache: prompt-cache pin dropped because provider did not prove llama.cpp /slots")
            return
        endpoint = replace(options.endpoint,
                           timeout_s=min(options.endpoint.timeout_s, _remaining(options)))
        try:
            slots = endpoint.get_slots()
        except (OSError, TimeoutError, EndpointError, ValueError):
            slots = None
        slot, usable = _idle_slot(slots)
        if slot is None:
            options._cache_slot_selection = "not_pinned"
            options._cache_slot_reason = (
                "server did not expose usable slot status" if slots is None or not usable
                else "server reported no idle slots")
            if slots is None or not usable:
                printer("cache: server does not expose slots; not pinned")
            else:
                printer("cache: no idle server slots; not pinned")
            return
        selection = "automatic_idle"
        reason = "lowest uniquely reported idle slot from loopback GET /slots"

    options._cache_slot = slot
    options._cache_slot_selection = selection
    options._cache_slot_reason = reason
    environment = profiles.load(options.profile)
    environment.update(options.extra_env)
    cache_mode = environment.get("STATEFRAME_PROMPT_CACHE", "off")
    cache_label = "on" if cache_mode == "on" else "off"
    printer(f"cache: pinned slot {slot} (cache_prompt {cache_label})")


def _route_defaults(options: RunOptions) -> dict[str, object]:
    assert options.endpoint is not None
    switches = profiles.load(options.profile)
    return {
        "endpoint": options.endpoint.base_url, "model": options.endpoint.model,
        "api_key_env": options.endpoint.api_key_env,
        "temperature": float(switches.get("REACT_TEMP", "1.0")),
        "max_tokens": int(switches.get("REACT_MAX_OUTPUT_TOKENS", "24000")),
        "timeout_s": float(switches.get("REACT_REQUEST_TIMEOUT_S", "1800")),
        "slot": options._cache_slot,
        "context_tokens": options._context_tokens,
        "allow_remote": not _is_loopback(options.endpoint.base_url),
    }


def _resolve_routes(options: RunOptions) -> routes.RouteTable:
    return routes.resolve(_route_defaults(options), cli=options.route_cli, file=options.routes_file,
                          profile=profiles.load_routes(options.profile), recorded=options._recorded_routes,
                          recorded_default=options._recorded_route_default)


def _routed_endpoint(settings: dict[str, object]) -> Endpoint:
    return Endpoint(str(settings["endpoint"]), str(settings["model"]),
                    settings.get("api_key_env") or None,
                    min(float(settings.get("timeout_s") or 10), 10),
                    bool(settings.get("allow_remote")))


def _probe_routes(options: RunOptions, printer=print, *, resume_roles: dict | None = None,
                  warnings: list[str] | None = None) -> None:
    options._provider_route_dialects.clear()
    table = _resolve_routes(options)
    if not table.active:
        options._routes = None
        return
    from .engine.kernel_launch_identity_cost import reasoning_profile
    from .engine.kernel_provider_dialect import origin_of, reasoning_controls_summary

    provider_environment = _provider_environment(options)
    selected_profile = reasoning_profile(environ=provider_environment)
    assert options.endpoint is not None
    from .engine.kernel_provider_dialect import require_routed_static_slot
    base_provider_capabilities = _provider_capabilities_for(
        options.endpoint.chat_url, options.endpoint.api_key(), options,
    )
    mismatched = [role for role, settings in table.roles.items()
                  if routes.uses_other_model(role, settings["model"], options.endpoint.model)
                  and table.sources[role].get("tokenizer", "default") == "default"]
    if os.environ.get("LLAMA_CPP_TOKENIZE_URL") and mismatched:
        raise RunError(f"LLAMA_CPP_TOKENIZE_URL is set while route {mismatched[0]} uses another model; set "
                       f"{mismatched[0]}.tokenizer or unset LLAMA_CPP_TOKENIZE_URL")
    checked: set[tuple[str, str, str | None]] = set()
    slots_by_endpoint: dict[str, object] = {}
    provider_routes: dict[str, dict[str, str]] = {}
    # Resolve default-endpoint auto pins before constructing probe endpoints. The
    # resolved table may be inactive, and Endpoint construction can print a warning.
    for role, settings in table.roles.items():
        if settings.get("slot") != "auto" or settings.get("endpoint") != options.endpoint.base_url:
            continue
        require_routed_static_slot(base_provider_capabilities)
        if options.endpoint.base_url not in slots_by_endpoint:
            slots_by_endpoint[options.endpoint.base_url] = options.endpoint.get_slots()
        recorded = (resume_roles or {}).get(role, {})
        selected = recorded.get("slot") if type(recorded.get("slot")) is int else None
        if selected is None:
            selected, _ = _idle_slot(slots_by_endpoint[options.endpoint.base_url])
        if selected is None:
            raise RunError(f"route {role}.slot = auto found no idle slot at {options.endpoint.base_url}; "
                           "choose a slot number or free a server slot")
        settings["slot"] = selected
    baseline = _route_defaults(options)
    if not routes.request_settings_changed(table, baseline, environ=os.environ):
        options._routes = None
        return
    def effective_unit_routes():
        overrides = table.unit_overrides()
        if not overrides:
            return
        classes = {"default", *(name[5:] for name in overrides if name.startswith("unit."))}
        for request_role in routes.ROLES:
            if request_role == "unit":
                continue
            for unit_class in sorted(classes):
                settings = table.resolve(request_role, unit_class=unit_class)
                sources = dict(table.sources[request_role])
                for name in ("unit", f"unit.{unit_class}"):
                    for route_field in overrides.get(name, ()):
                        sources[route_field] = table.sources[name][route_field]
                yield f"{request_role} (unit {unit_class})", settings, sources

    for role, settings, sources in chain(
            ((role, settings, table.sources[role]) for role, settings in table.roles.items()),
            effective_unit_routes()):
        endpoint = _routed_endpoint(settings)
        provider_capabilities = _provider_capabilities_for(
            endpoint.chat_url, endpoint.api_key(), options,
        )
        if (endpoint.base_url != options.endpoint.base_url
                and sources.get("slot") == "default"):
            # A pin discovered on the default server is not a pin on this one.
            settings["slot"] = None
        selected_slot = settings.get("slot")
        if selected_slot == "auto":
            require_routed_static_slot(provider_capabilities)
        elif type(selected_slot) is int:
            require_routed_static_slot(provider_capabilities, selected_slot)
        identity = (endpoint.base_url, endpoint.model, endpoint.api_key_env)
        if identity not in checked:
            try:
                probe = endpoint.probe()
            except EndpointError as error:
                raise RunError(f"route {role} {error}; change {role}.endpoint or {role}.model") from None
            if probe.get("model_found") is False:
                if not (role.partition(" (unit ")[0] == "decide2" and sources.get("model") == "default"):
                    raise RunError(f"route {role} endpoint does not serve model {endpoint.model!r}; "
                                   f"change {role}.model or {role}.endpoint")
            else:
                checked.add(identity)
        if endpoint.base_url not in slots_by_endpoint:
            slots_by_endpoint[endpoint.base_url] = (
                endpoint.get_slots() if provider_capabilities.slots else None
            )
        slots = slots_by_endpoint[endpoint.base_url]
        selected = settings.get("slot")
        if selected == "auto":
            recorded = (resume_roles or {}).get(role, {})
            selected = recorded.get("slot") if type(recorded.get("slot")) is int else None
            if selected is None:
                selected, _ = _idle_slot(slots)
            if selected is None:
                raise RunError(f"route {role}.slot = auto found no idle slot at {endpoint.base_url}; "
                               "choose a slot number or free a server slot")
            settings["slot"] = selected
            require_routed_static_slot(provider_capabilities, selected)
        if selected is not None and isinstance(slots, list) and slots and not any(
                isinstance(entry, dict) and _slot_id(entry) == selected
                for entry in slots):
            raise RunError(f"route {role}.slot {selected} is not reported by {endpoint.base_url}; "
                           "choose a slot this server reports or use slot = 'auto'")
        slot_entries = [entry for entry in slots or [] if isinstance(entry, dict) and
                        (selected is None or _slot_id(entry) == selected)]
        sizes = [entry["n_ctx"] for entry in slot_entries if type(entry.get("n_ctx")) is int]
        configured_context = int(settings.get("context_tokens") or options._context_tokens)
        if configured_context < options._context_tokens:
            raise RunError(f"route {role}.context_tokens={configured_context} is smaller than this run's "
                           f"{options._context_tokens:,}-token frame; raise it or lower STATEFRAME_CONTEXT_TOKENS")
        explicit = bool(options._context_env.get(CONTEXT_TOKENS_ENV))
        need = options._context_tokens if explicit else configured_context
        if sizes and (min(sizes) < need or explicit and any(size != need for size in sizes)):
            size = min(sizes)
            raise RunError(f"route {role} model {endpoint.model} has a {size:,}-token slot; this run's frames "
                           f"need {need:,} ({options._prompt_budget:,} prompt + reply). Give {role} a model "
                           f"with a {need:,}-token slot, or run with {CONTEXT_TOKENS_ENV}={size}")
        if not _is_loopback(endpoint.base_url):
            warning = (f"WARNING: route {role} uses remote endpoint {endpoint.base_url}; "
                       "source code will leave the machine")
            if warnings is None:
                print(warning, file=sys.stderr)
            else:
                warnings.append(warning)
        tokenizer = settings.get("tokenizer")
        if tokenizer:
            try:
                tokenizer_endpoint = Endpoint(str(tokenizer), endpoint.model, endpoint.api_key_env,
                                              endpoint.timeout_s, bool(settings.get("allow_remote")))
                result = tokenizer_endpoint.probe()
            except EndpointError as error:
                raise RunError(f"route {role}.tokenizer cannot be used: {error}; change {role}.tokenizer") from None
            if result.get("model_found") is not True:
                raise RunError(f"route {role}.tokenizer does not list model {endpoint.model!r}; "
                               f"change {role}.tokenizer to a server for that model")
        provider_routes[role] = {
            "origin": origin_of(endpoint.chat_url),
            "dialect": provider_capabilities.dialect,
            "reason": provider_capabilities.reason,
            "reasoning_controls": reasoning_controls_summary(
                provider_capabilities.dialect,
                reasoning_off=selected_profile.off,
                environ=provider_environment,
            ),
        }
    # An auto slot is an effective setting: hash the resolved table the engine receives.
    table = routes.with_effective_roles(routes.RouteTable(table.roles, table.sources, True, ""))
    options._routes = table
    options._provider_route_dialects.update(provider_routes)
    for role, settings in table.roles.items():
        from urllib.parse import urlsplit
        host = urlsplit(str(settings["endpoint"])).netloc
        locality = "local" if _is_loopback(str(settings["endpoint"])) else "remote"
        slot = f", slot {settings['slot']}" if settings.get("slot") is not None else ""
        inherited = all(source == "default" for source in table.sources[role].values())
        route_provider = provider_routes.get(role)
        if route_provider is None:
            continue
        if inherited:
            printer(f"route {role}: default ({settings['model']} at {host}, {locality}{slot}); "
                    f"provider {route_provider['dialect']} ({route_provider['reason']})")
        else:
            printer(f"route {role}: {settings['model']} at {host} ({locality}{slot}); "
                    f"provider {route_provider['dialect']} ({route_provider['reason']})")


def _provider_preflight_summary(options: RunOptions) -> str:
    from .engine.kernel_launch_identity_cost import reasoning_profile
    from .engine.kernel_provider_dialect import reasoning_controls_summary

    environment = _provider_environment(options)
    profile = reasoning_profile(environ=environment)
    default = _provider_capabilities_for(
        options.endpoint.chat_url, options.endpoint.api_key(), options,
    )
    default_controls = reasoning_controls_summary(
        default.dialect, reasoning_off=profile.off, environ=environment,
    )
    result = (f"provider dialect: {default.dialect} ({default.reason}); "
              f"{default_controls}")
    if options._provider_route_dialects:
        routes_summary = " | ".join(
            f"{role}@{record['origin']}={record['dialect']} ({record['reason']}); "
            f"{record['reasoning_controls']}"
            for role, record in sorted(options._provider_route_dialects.items())
        )
        result += "; routes: " + routes_summary
    return result


def _provider_environment(options: RunOptions) -> dict[str, str]:
    """Resolve the same operator/profile environment used by the engine."""
    result = dict(os.environ)
    result.update(profiles.load(options.profile))
    result.update(resolve_opt_in_switches())
    result.update(options.extra_env)
    return result


def _apply_operator_reasoning_effort(
    environment: dict[str, str], *, environ=None,
) -> None:
    """Preserve the shell's existing reasoning setting in the isolated run spec.

    The child purges inherited REACT_* variables before installing that spec.
    Profile values keep precedence; command-specific extra_env is applied later.
    """
    source = os.environ if environ is None else environ
    value = source.get("REACT_REASONING_EFFORT")
    if "REACT_REASONING_EFFORT" not in environment and isinstance(value, str) and value.strip():
        environment["REACT_REASONING_EFFORT"] = value


def _provider_capabilities_for(endpoint: str, key: str, options: RunOptions):
    from .engine.kernel_provider_dialect import capabilities_for

    return capabilities_for(
        endpoint, key, cache=options._provider_capabilities,
        environ=_provider_environment(options),
    )


def _provider_preflight_records(options: RunOptions) -> dict[str, dict[str, object]]:
    from .engine.kernel_launch_identity_cost import reasoning_profile
    from .engine.kernel_provider_dialect import reasoning_controls_summary

    environment = _provider_environment(options)
    selected = reasoning_profile(environ=environment)
    return {
        origin: {
            **capability.to_record(),
            "reasoning_controls": reasoning_controls_summary(
                capability.dialect, reasoning_off=selected.off, environ=environment,
            ),
        }
        for origin, capability in options._provider_capabilities.items()
        if isinstance(origin, str) and hasattr(capability, "to_record")
    }


def _print_run_banner(options: RunOptions, printer=print) -> None:
    """Print general run settings after the provider decision leads the output."""
    printer(_time_limit_line(options))
    printer(_step_limit_line(options))
    printer("per-check timeout: outer 900s; engine 300s; per model request: the profile's timeout")
    ignored = ignored_shell_settings()
    if ignored:
        printer(f"note: {', '.join(ignored)} {'is' if len(ignored) == 1 else 'are'} set in your shell but "
                "not passed to the engine (engine settings come from the profile; see docs/guide/running.md)")
    printer(f"prompt budget: {options._prompt_budget:,} tokens; {options._context_tokens:,}-token server slot required"
            + ("" if options._context_display == "default" else f" ({options._context_display})"))
    printer("endpoint policy: local-only by default; a loopback URL can still forward elsewhere")


# Plain-language CLI text for known engine refusals whose engine wording is consumed by engine code
# (so it cannot be rewritten at build time). The raw detail stays in engine.log.
USER_ERROR_TEXT = {
    "request preparation binding changed before send": (
        "a file in the working copy changed while a request was being prepared, so the request was "
        "refused rather than send stale content. Avoid editing the working copy during a run."),
}


def _render_event(event: dict) -> str | None:
    kind = event.get("kind")
    if kind == "provider_dialects":
        endpoints = event.get("endpoints")
        if not isinstance(endpoints, list):
            return None
        rows = []
        for item in endpoints:
            if not isinstance(item, dict):
                continue
            rows.append(
                f"{item.get('origin')}: {item.get('dialect')} ({item.get('reason')}); "
                f"{item.get('reasoning_controls', 'reasoning controls not recorded')}"
            )
        return "provider dialects: " + " | ".join(rows) if rows else None
    if kind == "engine_start":
        turns = event.get("turns")
        line = "engine started (no step limit)" if turns in (None, 0) else f"engine started (step limit {turns})"
        if event.get("resume_attempt") is not None:
            line += f", resume attempt {event['resume_attempt']}, replaying {event.get('replay_sends')} exchanges"
        return line
    if kind == "request":
        return (f"turn {event.get('turn')}: request {event.get('request')} "
                f"[{event.get('roles')}] menu={','.join(str(t) for t in event.get('tools') or [])}")
    if kind == "send":
        cached = event.get("cached_tokens")
        tokens = event.get("prompt_tokens")
        if (type(tokens) is int and tokens > 0 and type(cached) is int
                and 0 <= cached <= tokens):
            cache = f", cached share: {cached / tokens:.1%} ({cached}/{tokens}, normalized)"
        else:
            cache = ", cached share: not reported by server"
        actions = ", ".join(event.get("actions") or []) or "(no tool call)"
        if event.get("replayed") is True:
            return (f"   replaying {event.get('replay_position')}/{event.get('replay_total')}: recorded answer, "
                    f"no model call -> {actions}")
        return (f"   sent {event.get('request_bytes')} B, prompt {tokens} tok{cache}, "
                f"{event.get('seconds')}s -> {actions}")
    if kind == "tests":
        # Display only: the engine's summary text is also model-facing and matched by engine code, so it is
        # never rewritten at the source. The CLI shows a plain separator instead of the engine's em dash.
        return f"   tests: {str(event.get('summary')).replace(' — ', ': ')}"
    if kind == "replay_done":
        return f"replay complete ({event.get('replayed')} recorded exchanges); continuing live"
    if kind == "replay_diverged":
        return (f"replay stopped early at recorded exchange {event.get('tape_position')} of "
                f"{event.get('tape_total')}: the engine's rebuilt request differs from the recorded one, so the "
                f"rest is not a faithful replay; continuing LIVE from here, the model is asked again "
                f"(where they differ: resume-divergence.json)")
    if kind == "token_counting_unavailable":
        return "token counting unavailable: using conservative estimate"
    if kind == "send_error":
        return f"   send failed: {event.get('error')}"
    if kind == "engine_error":
        detail = str(event.get("detail") or "")
        for engine_text, plain in USER_ERROR_TEXT.items():
            if engine_text in detail:
                return f"stopped: {plain}"
        return f"engine error: {event.get('error')}: {detail}"
    if kind == "engine_end":
        return f"engine finished: {event.get('result')} in {event.get('seconds')}s"
    return None


def _warn_checkpoint_disabled(reason: str, warned: set[str]) -> None:
    if "checkpoint" not in warned:
        warned.add("checkpoint")
        detail = " ".join(str(reason).split())[:240] or "unknown reason"
        print(f"warning: checkpoints are off for this run: {detail}", file=sys.stderr, flush=True)


def _warn_checkpoint_event(event: dict, warned: set[str]) -> None:
    if event.get("kind") == "checkpoint_key_unusable":
        _warn_checkpoint_disabled(str(event.get("reason") or "checkpoint key is unusable"), warned)


def _warn_checkpoint_journal(run: Path, warned: set[str]) -> None:
    from .engine.kernel_journal_store import open_journal

    path = run / "checkpoints" / "events.jsonl"
    try:
        with open_journal(path) as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("event") == "checkpoint_disabled":
                    _warn_checkpoint_disabled(str(event.get("reason") or "saving failed"), warned)
                    return
    except OSError:
        pass


def drive(run: Path, options: RunOptions, printer=print) -> int:
    """Start the engine child and stream its progress lines."""
    _validate_timeout(options.timeout_s)
    remaining = _remaining(options)
    spec_path = run / "spec.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        environment = spec["environment"]
        sources = spec.setdefault("environment_sources", {})
        # No engine wall unless the user set a run limit: the engine reads an unset wall as no deadline.
        if math.isfinite(remaining):
            environment["STATEFRAME_RUN_WALL_SECONDS"] = str(remaining)
            sources["STATEFRAME_RUN_WALL_SECONDS"] = "wrapper:run-deadline"
        else:
            environment.pop("STATEFRAME_RUN_WALL_SECONDS", None)
            sources.pop("STATEFRAME_RUN_WALL_SECONDS", None)
        # Contract with the engine: the engine runs PROJECT tests with exactly this
        # interpreter, else its own sys.executable. Through the spec, because _child.install_environment
        # drops inherited STATEFRAME_* variables.
        if engine_reads_test_python():
            environment[TEST_PYTHON_ENV] = _tests_python(options)
            sources[TEST_PYTHON_ENV] = "runner:test-interpreter"
        else:
            environment.pop(TEST_PYTHON_ENV, None)
            sources.pop(TEST_PYTHON_ENV, None)
        # Keep request limits within the run budget. The engine separately bounds
        # silent, responsive-provider queue waits by STATEFRAME_QUEUE_WAIT_MAX_S.
        for key, default in (("REACT_REQUEST_TIMEOUT_S", 1800),
                             ("STATEFRAME_PER_REQUEST_TIMEOUT_S", 600)):
            limit = float(environment.get(key, default))
            _validate_timeout(limit)
            environment[key] = str(min(limit, remaining))
            if math.isfinite(remaining):
                sources[key] = "wrapper:run-deadline"
    engine_deadline = min(time.monotonic() + remaining,
                          options._deadline if options._deadline is not None else math.inf)
    if spec_path.exists():
        # react_repair reads this absolute deadline before constructing its
        # continuation budget, avoiding a fresh allowance after child startup.
        if math.isfinite(engine_deadline):
            environment["CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"] = str(engine_deadline)
            spec["environment_sources"]["CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"] = "wrapper:run-deadline"
        else:
            environment.pop("CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC", None)
        _write_run_spec_atomic(run, spec)
    # Keep the engine identity near the front of Linux argv: narrow `ps args`
    # output can truncate the absolute _child.py path even though the child is direct.
    # Start from the selected interpreter directory so Python retains venv discovery
    # and sys.executable while argv stays short; then restore the engine's normal cwd/path.
    if sys.platform == "linux":
        python_executable = Path(os.path.abspath(options.python))
        child_bootstrap = (
            "'stateframe._child.py';import os,runpy,sys;"
            f"sys.path[0]={str(CHILD.parent)!r};"
            f"sys.argv=[{str(CHILD)!r},{str(run)!r}];"
            f"os.chdir({str(run)!r});"
            f"runpy.run_path({str(CHILD)!r},run_name='__main__')"
        )
        command = [f"./{python_executable.name}", "-B", "-c", child_bootstrap,
                   "stateframe._child"]
        child_cwd = python_executable.parent
    else:
        command = [str(options.python), "-B", str(CHILD), str(run)]
        child_cwd = run
    log_path = run / "engine.log"
    checkpoint_warned: set[str] = set()
    if spec_path.exists() and environment.get("STATEFRAME_CHECKPOINT", "on").strip().lower() in {"off", "0"}:
        _warn_checkpoint_disabled("disabled by configuration", checkpoint_warned)
    # Checkpoints copy the live log path; resume archives the previous attempt before this append.
    try:
        with log_path.open("x", encoding="utf-8"):
            pass
    except FileExistsError:
        pass
    with log_path.open("a", encoding="utf-8") as log:
        _remaining(options)
        child_env = _child_env(options)
        if spec_path.exists():
            # Hash randomization is fixed at interpreter start: the child applying the spec's
            # PYTHONHASHSEED to os.environ itself would be too late to change its own set order.
            seed = json.loads(spec_path.read_text())["environment"].get("PYTHONHASHSEED")
            if seed is not None:
                child_env["PYTHONHASHSEED"] = str(seed)
        process = subprocess.Popen(command, executable=options.python, cwd=child_cwd, env=child_env,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)
        from .engine.kernel_exec_tracker import capture_launch_identity
        capture_launch_identity(process, session_leader=True)
        assert process.stdout is not None
        timed_out = threading.Event()

        def on_timeout() -> None:
            timed_out.set()
            _stop_group(process)

        # A silent, hung engine must still be stopped, so the limit is a timer, not a line check.
        kill_at = _kill_deadline(engine_deadline, options._deadline, remaining)
        watchdog = (threading.Timer(max(0, kill_at - time.monotonic()), on_timeout)
                    if math.isfinite(kill_at) else None)  # no limit set: no watchdog kill
        try:
            if watchdog is not None:
                watchdog.daemon = True
                watchdog.start()
            # A forked checkpoint writer can inherit the engine's stdout; reading to EOF would then wait for
            # the writer, not the engine. Read on a thread and stop at most OUTPUT_DRAIN_SECONDS after the
            # engine itself exits (the process group is stopped in `finally`).
            lines: queue.Queue = queue.Queue()
            reader = threading.Thread(target=_pump_lines, args=(process.stdout, lines), daemon=True)
            reader.start()
            drain_until = None
            from .engine.kernel_exec_tracker import child_exited_unreaped
            while True:
                from .shell_host import answer_pending
                answer_pending(run, options.shell_confirm)
                if drain_until is None and (child_exited_unreaped(process) or timed_out.is_set()):
                    drain_until = time.monotonic() + OUTPUT_DRAIN_SECONDS
                if drain_until is not None and timed_out.is_set() and time.monotonic() >= drain_until:
                    break
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    if drain_until is not None and time.monotonic() >= drain_until:
                        break
                    continue
                if line is None:  # EOF
                    break
                log.write(line)
                log.flush()
                if line.startswith(LOG_PREFIX):
                    try:
                        event = json.loads(line[len(LOG_PREFIX):])
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    _warn_checkpoint_event(event, checkpoint_warned)
                    rendered = _render_event(event)
                    if rendered and (not options.quiet or event.get("kind") in {
                            "token_counting_unavailable", "engine_error"}):
                        printer(rendered)
            exit_deadline = time.monotonic() + 60
            while not child_exited_unreaped(process) and time.monotonic() < exit_deadline:
                time.sleep(.02)
            _stop_group(process)
            code = process.returncode
            if code is None:
                raise RuntimeError("engine survived process-group cleanup")
            _warn_checkpoint_journal(run, checkpoint_warned)
            _print_queue_wait_warning(run, printer)
            if timed_out.is_set():
                printer("engine timed out: deadline exhausted; engine stopped")
                return 124
            return code
        finally:
            if watchdog is not None:
                watchdog.cancel()
            _stop_group(process)
            process.stdout.close()


OUTPUT_DRAIN_SECONDS = 10.0


def _queue_wait_warning(run: Path) -> str | None:
    """Summarize the longest durable queue wait, including failed sends."""
    from .engine.kernel_journal_store import journal_paths, open_journal

    longest = 0.0
    # A finished run seals each journal as events.jsonl.xz; journal_paths/open_journal read both layouts.
    for path in journal_paths(run / "worker" / "measurement-runs"):
        try:
            with open_journal(path) as source:
                for line in source:
                    record = json.loads(line)
                    if record.get("kind") == "provider_queue_wait":
                        seconds = record.get("seconds_waited")
                        if isinstance(seconds, (int, float)) and math.isfinite(seconds):
                            longest = max(longest, seconds)
        except (OSError, UnicodeError, ValueError):
            continue
    if longest > 0:
        return (f"warning: the model server queued requests for up to {longest:.1f} s; "
                "consider fewer parallel runs")
    return None


def _print_queue_wait_warning(run: Path, printer) -> None:
    warning = _queue_wait_warning(run)
    if warning is not None:
        printer(warning)


def _pump_lines(stream, lines: "queue.Queue") -> None:
    try:
        for line in stream:
            lines.put(line)
    except (OSError, ValueError):
        pass  # the stream was closed by cleanup
    finally:
        lines.put(None)


def keep_checkpoints_requested(flag: bool = False) -> bool:
    return flag or os.environ.get("STATEFRAME_KEEP_CHECKPOINTS", "").strip().lower() in {"1", "on", "true", "yes"}


def clean_finished_checkpoints(run_dir: Path, *, keep: bool = False) -> int | None:
    """Delete <run>/checkpoints/ once the run is finished for good (resume would refuse it anyway), after the
    journal and summary are written. Returns the bytes removed, or None when nothing was removed. Stopped,
    cancelled, killed and crashed runs keep their checkpoints: those are the runs resume needs."""
    root = run_dir / "checkpoints"
    if keep_checkpoints_requested(keep) or not root.is_dir() or not resume_module.run_finished_for_good(run_dir):
        return None
    size = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
        except OSError:
            pass
    from .storage_gc import remove_tree
    remove_tree(root)
    return None if root.exists() else size


def keep_run_env_requested(flag: bool = False, environ=None) -> bool:
    """Whether the finished run environment should be kept, matching checkpoint retention aliases."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get("STATEFRAME_KEEP_RUN_ENV", "")).strip().lower()
    if raw and raw not in {"on", "off", "1", "0", "true", "false", "yes", "no"}:
        raise RunError("STATEFRAME_KEEP_RUN_ENV must be on or off")
    return flag or raw in {"on", "1", "true", "yes"}


def clean_finished_run_env(run_dir: Path, *, keep: bool = False) -> int | None:
    """Write env-manifest.json and remove env/ only after resume can no longer use it."""
    if keep_run_env_requested(keep) or not resume_module.run_finished_for_good(run_dir):
        return None
    from .storage_gc import remove_run_env
    return remove_run_env(run_dir, require_finished=False)


def finish_run_storage(run_dir: Path, *, keep_checkpoints: bool = False,
                       keep_run_env: bool = False, printer=print) -> dict[str, int | None]:
    """Apply finish-time storage actions without allowing cleanup errors to change the result."""
    removed: dict[str, int | None] = {"checkpoints": None, "env": None}
    try:
        if not resume_module.run_finished_for_good(run_dir):
            return removed
    except Exception as error:
        printer(f"warning: finished-run storage was not checked: {type(error).__name__}: {error}")
        return removed
    try:
        removed["checkpoints"] = clean_finished_checkpoints(run_dir, keep=keep_checkpoints)
    except Exception as error:
        printer(f"warning: checkpoints were not removed: {type(error).__name__}: {error}")
    try:
        removed["env"] = clean_finished_run_env(run_dir, keep=keep_run_env)
    except Exception as error:
        printer(f"warning: run environment was not removed: {type(error).__name__}: {error}")
    try:
        from . import storage_gc
    except Exception as error:
        printer(f"warning: run storage helpers could not load: {type(error).__name__}: {error}")
        return removed
    for label, seal in (("measurement storage", storage_gc.seal_measurements),
                        ("preflight requests", storage_gc.seal_requests)):
        try:
            seal(run_dir, printer=printer)
        except Exception as error:
            printer(f"warning: {label} was not sealed: {type(error).__name__}: {error}")
    return removed


def _stop_group(process: subprocess.Popen) -> None:
    """Stop the engine and everything it started (pytest, HTTP helpers): its own session."""
    from .engine.kernel_exec_tracker import recorded_launch_identity, signal_launch_group

    try:
        signalled = signal_launch_group(process, recorded_launch_identity(process), signal.SIGKILL)
    except PermissionError:
        # macOS can return EPERM for the group (zombie members, or a sandboxed member we may not signal).
        # Cleanup runs in `finally` blocks and must never raise: an error here would replace the real
        # reason the run stopped (a --timeout run once reported "could not start engine"). Kill the leader.
        signalled = False
    if not signalled and process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _artifact_digest(files: dict[str, bytes | None]) -> str | None:
    """Use the engine's source-map hash, distinct from the host-check tree hash."""
    from .engine.kernel_done_is_submit_authority import DoneCaptureError, artifact_digest

    sources = {}
    for name, data in files.items():
        if data is None:
            sources[name] = None
        else:
            try:
                sources[name] = data.decode("utf-8")
            except UnicodeDecodeError:
                sources[name] = data
    try:
        return artifact_digest(sources)
    except DoneCaptureError:
        return None


def final_candidate(run: Path) -> dict[str, bytes | None] | None:
    """Export the immutable DoneSeal artifact; unsealed captures are partial only."""
    folder = run / "worker" / "candidates"
    captures = sorted(folder.glob("[0-9][0-9][0-9][0-9].json")) if folder.is_dir() else []
    terminal = outcomes.terminal_receipt(run)
    submission = terminal.get("submission")
    digest = outcomes.sealed_artifact_digest(terminal)
    if isinstance(submission, dict) and submission.get("stored") is True and digest is None:
        raise RunError("malformed submission seal; candidate remains partial")
    if digest is not None:
        # store_candidate owns this layout and canonical digest. Proposal numbers
        # do not identify a submitted artifact and must never select its export.
        root = run / "worker/results/react_done_submissions"
        artifacts = list(root.glob(f"*/{digest.removeprefix('sha256:')}.json"))
        if not artifacts and not captures:
            return None
        if len(artifacts) != 1:
            raise RunError("sealed candidate artifact missing or ambiguous; candidate remains partial")
        path = artifacts[0]
        if any((run / Path(*path.relative_to(run).parts[:index])).is_symlink()
               for index in range(1, len(path.relative_to(run).parts) + 1)):
            raise RunError("sealed candidate artifact contains a symlink")
        record = json.loads(path.read_text(encoding="utf-8"))
        writes = record.get("writes") if isinstance(record, dict) else None
        deletes = record.get("deletes") if isinstance(record, dict) else None
        schema = record.get("schema") if isinstance(record, dict) else None
        binary = record.get("binary_writes", {}) if isinstance(record, dict) else None
        expected_keys = ({"schema", "artifact_digest", "writes", "deletes", "binary_writes"}
                         if schema == "stateframe-done-artifact/3" else
                         {"schema", "artifact_digest", "writes", "deletes"})
        if (not isinstance(record, dict)
                or set(record) != expected_keys
                or schema not in {"stateframe-done-artifact/2", "stateframe-done-artifact/3"}
                or record.get("artifact_digest") != digest or not isinstance(writes, dict)
                or not isinstance(deletes, list) or not (writes or deletes or binary)
                or not isinstance(binary, dict)
                or (schema == "stateframe-done-artifact/3" and not binary)
                or not all(isinstance(k, str) and isinstance(v, str)
                           for k, v in writes.items())
                or not all(isinstance(k, str) and isinstance(v, str)
                           for k, v in binary.items())
                or not all(isinstance(path, str) for path in deletes)
                or deletes != sorted(set(deletes))
                or (set(writes) & set(deletes)) or (set(binary) & (set(writes) | set(deletes)))):
            raise RunError("malformed sealed candidate artifact")
        files = {name: source.encode("utf-8") for name, source in writes.items()}
        try:
            files.update({name: base64.b64decode(source, validate=True)
                          for name, source in binary.items()})
        except (ValueError, binascii.Error) as error:
            raise RunError("malformed sealed candidate binary write") from error
        files.update({name: None for name in deletes})
        if _artifact_digest(files) != digest:
            raise RunError("candidate digest does not match submission seal; candidate remains partial")
        return files
    if not captures:
        return None
    record = json.loads(captures[-1].read_text())
    files = record.get("files_base64") if isinstance(record, dict) else None
    if not isinstance(files, dict) or not all(isinstance(k, str) and isinstance(v, str)
                                             for k, v in files.items()):
        raise RunError("malformed candidate capture")
    deletes = record.get("deleted_files", [])
    if not isinstance(deletes, list) or not all(isinstance(path, str) for path in deletes):
        raise RunError("malformed candidate capture deletions")
    if set(files) & set(deletes):
        raise RunError("candidate capture writes and deletes overlap")
    return {**{name: base64.b64decode(data, validate=True) for name, data in files.items()},
            **{name: None for name in deletes}}


def _read_send_metrics(run: Path, sends: list[dict]) -> list[dict[str, int | None]]:
    """Normalize each captured response, with a compatibility path for older runs."""
    wire_dir = run / "wire"
    metrics = []
    for fallback_index, send in enumerate(sends, start=1):
        send_index = send.get("send")
        if type(send_index) is not int or send_index < 1:
            send_index = fallback_index
        response_path = wire_dir / f"{send_index:04d}.response.json"
        if response_path.is_file():
            try:
                response = json.loads(response_path.read_text())
            except (OSError, UnicodeError, json.JSONDecodeError):
                response = {}
            response = response if isinstance(response, dict) else {}
            metrics.append(cache_metrics.normalize(response.get("usage"), response.get("timings")))
            continue
        # Older run folders kept usage.prompt_tokens (a total) and the cache
        # counter in sends.jsonl, but did not retain raw response timings.
        metrics.append(cache_metrics.normalize(
            {"prompt_tokens": send.get("prompt_tokens"),
             "prompt_tokens_details": {"cached_tokens": send.get("cached_tokens")}}, {}))
    return metrics


def _known_total(metrics: list[dict[str, int | None]], key: str) -> int | None:
    values = [item[key] for item in metrics]
    if not values or any(type(value) is not int for value in values):
        return None
    return sum(values)


def _cache_share_line(summary: dict[str, object]) -> str:
    total = summary.get("prompt_tokens_sent")
    cached = summary.get("cached_tokens")
    share = summary.get("cached_share")
    coverage = summary.get("cache_coverage") or {}
    suffix = (f"; coverage {coverage.get('n_sends_with_both', 0)}/"
              f"{coverage.get('n_sends', 0)} sends")
    if type(total) is not int or type(cached) is not int or type(share) not in (float, int):
        return "cached share: unknown" + suffix
    return f"cached share: {share:.1%} ({cached}/{total}, normalized)" + suffix


def _effective_route_prices(route_spec: dict, role: str, route: dict) -> dict:
    prices = dict(route_spec["roles"].get(role, {}))
    if route.get("scope") == "unit":
        unit_class = route.get("unit_class")
        for name in ("unit", f"unit.{unit_class}" if isinstance(unit_class, str) else ""):
            unit = route_spec["roles"].get(name, {})
            prices.update({key: unit[key] for key in route_spec.get("unit_overrides", {}).get(name, ())
                           if key in unit})
    return prices


def _route_costs(run: Path, route_spec: dict, sends: list[dict],
                 metrics: list[dict[str, int | None]]) -> dict[str, object]:
    """Price recorded sends only; missing usage stays unknown rather than becoming zero."""
    rates = route_spec["roles"]
    rows: dict[str, dict[str, object]] = {
        role: {"requests": 0, "escalated": 0, "prompt_tokens": 0, "cached_tokens": 0,
               "completion_tokens": 0, "estimated_cost": None}
        for role in rates}
    incomplete: set[str] = set()
    priced: dict[str, float] = {role: 0.0 for role in rates}
    priced_as_edit = 0.0
    edit_incomplete = False
    keys = ("cost_input_per_mtok", "cost_cached_input_per_mtok", "cost_output_per_mtok")
    request_roles: dict[str, str] = {}
    escalated_from: list[tuple[str, int]] = []
    unattributed_sends = max(0, len(sends) - len(metrics))
    for fallback, (send, usage) in enumerate(zip(sends, metrics), start=1):
        index = send.get("send") if type(send.get("send")) is int else fallback
        path = run / "wire" / f"{index:04d}.route.json"
        if not path.is_file():
            unattributed_sends += 1
            continue
        try:
            route = json.loads(path.read_text())
        except (OSError, UnicodeError, ValueError):
            unattributed_sends += 1
            continue
        role = route.get("role") if isinstance(route, dict) else None
        if role not in rows:
            unattributed_sends += 1
            continue
        row = rows[role]
        row["requests"] += 1
        request_id = send.get("request_id")
        if isinstance(request_id, str):
            request_roles[request_id] = role
        source = route.get("escalated_from")
        if source:
            escalated_from.append((str(source), fallback))
        completion = send.get("completion_tokens")
        if type(completion) is not int:
            response_path = run / "wire" / f"{index:04d}.response.json"
            if response_path.is_file():
                try:
                    response = json.loads(response_path.read_text())
                    completion = (response.get("usage") or {}).get("completion_tokens")
                except (OSError, UnicodeError, ValueError, AttributeError):
                    completion = None
        if type(completion) is not int and isinstance(send.get("usage"), dict):
            completion = send["usage"].get("completion_tokens")
        token_counts = {"prompt_tokens": usage["total"], "cached_tokens": usage["cached"],
                        "completion_tokens": completion}
        for key, value in token_counts.items():
            if type(value) is int and value >= 0:
                if row[key] is not None:
                    row[key] += value
            else:
                # One unreported count makes the role's column unknown, never a silent undercount.
                row[key] = None
                incomplete.add(role)
        valid_usage = (all(type(token_counts[key]) is int and token_counts[key] >= 0 for key in token_counts)
                       and token_counts["cached_tokens"] <= token_counts["prompt_tokens"])
        for hypothetical, prices in ((False, _effective_route_prices(route_spec, role, route)),
                                     (True, _effective_route_prices(route_spec, "edit", route))):
            if valid_usage and all(type(prices.get(key)) in (int, float) for key in keys):
                cost = ((token_counts["prompt_tokens"] - token_counts["cached_tokens"]) * prices[keys[0]] +
                        token_counts["cached_tokens"] * prices[keys[1]] +
                        token_counts["completion_tokens"] * prices[keys[2]]) / 1_000_000
                if hypothetical:
                    priced_as_edit += cost
                else:
                    priced[role] += cost
            elif hypothetical:
                edit_incomplete = True
            else:
                incomplete.add(role)
        if not valid_usage:
            incomplete.add(role)
    for source, fallback in escalated_from:
        role = request_roles.get(source)
        if role is None and fallback > 1:
            previous = sends[fallback - 2]
            prior_index = previous.get("send") if type(previous.get("send")) is int else fallback - 1
            prior_path = run / "wire" / f"{prior_index:04d}.route.json"
            if prior_path.is_file():
                try:
                    role = json.loads(prior_path.read_text()).get("role")
                except (OSError, UnicodeError, ValueError, AttributeError):
                    pass
        if role in rows:
            rows[role]["escalated"] += 1
    for role, row in rows.items():
        if role not in incomplete and (row["requests"] or
                                      all(type(rates[role].get(key)) in (int, float) for key in keys)):
            row["estimated_cost"] = round(priced[role], 8)
    total = {key: (None if any(row[key] is None for row in rows.values())
                   else sum(int(row[key]) for row in rows.values()))
             for key in ("requests", "escalated", "prompt_tokens", "cached_tokens", "completion_tokens")}
    total["requests"] += unattributed_sends
    if unattributed_sends:
        for key in ("escalated", "prompt_tokens", "cached_tokens", "completion_tokens"):
            total[key] = None
    costs = [row["estimated_cost"] for row in rows.values() if row["requests"]]
    total["estimated_cost"] = (round(sum(costs), 8)
                               if not unattributed_sends and all(cost is not None for cost in costs) else None)
    as_edit = round(priced_as_edit, 8) if total["estimated_cost"] is not None and not edit_incomplete else None
    return {"roles": rows, "total": total, "unattributed_sends": unattributed_sends,
            "all_at_edit_rates": as_edit,
            "estimated_savings": round(as_edit - total["estimated_cost"], 8)
            if as_edit is not None and total["estimated_cost"] is not None else None}


def summarize(run: Path, started: float) -> dict[str, object]:
    sends = []
    path = run / "sends.jsonl"
    if path.exists():
        sends = [record for line in path.read_text().splitlines() if line.strip()
                 for record in [json.loads(line)] if isinstance(record, dict)]
    metrics = _read_send_metrics(run, sends)
    pairs = [(item["total"], item["cached"]) for item in metrics
             if type(item["total"]) is int and type(item["cached"]) is int
             and 0 <= item["cached"] <= item["total"]]
    total_prompt = _known_total(metrics, "total")
    cached_prompt = _known_total(metrics, "cached")
    uncached_prompt = _known_total(metrics, "uncached")
    try:
        requests = len(list_request_records(run))
    except (OSError, RequestRecordsError):
        requests = None
    slot_info: dict[str, object] = {"slot": None, "selection": "not_recorded",
                                   "reason": "run did not record slot selection"}
    capabilities: dict[str, object] = {}
    provider_capabilities: dict[str, object] = {}
    spec: dict = {}
    spec_path = run / "spec.json"
    if spec_path.is_file():
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            spec = {}
        if isinstance(spec, dict) and isinstance(spec.get("prompt_cache_slot"), dict):
            slot_info = spec["prompt_cache_slot"]
        if isinstance(spec, dict) and isinstance(spec.get("capabilities"), dict):
            capabilities = spec["capabilities"]
        if isinstance(spec, dict) and isinstance(spec.get("provider_capabilities"), dict):
            provider_capabilities = spec["provider_capabilities"]
    paired_total = sum(total for total, _ in pairs)
    paired_cached = sum(cached for _, cached in pairs)
    summary: dict[str, object] = {
        "requests": requests,
        "provider_sends": len(sends),
        # Sends answered from an earlier attempt's records by ``stateframe resume`` (whole-run totals
        # below still count them once: they were sent to the model by that earlier attempt).
        "replayed_sends": sum(1 for s in sends if s.get("replayed") is True),
        "request_bytes_sent": sum(int(s.get("request_bytes") or 0) for s in sends),
        "max_request_bytes": max((int(s.get("request_bytes") or 0) for s in sends), default=0),
        "prompt_tokens_sent": total_prompt,
        "uncached_prompt_tokens": uncached_prompt,
        "completion_tokens": sum(int(s.get("completion_tokens") or 0) for s in sends) or None,
        "cached_tokens": cached_prompt,
        "cached_share": (round(paired_cached / paired_total, 4)
                         if pairs and len(pairs) == len(sends) and paired_total > 0 else None),
        "cache_coverage": {"n_sends_with_both": len(pairs), "n_sends": len(sends)},
        "capabilities": capabilities,
        "provider_capabilities": provider_capabilities,
        "prompt_cache_slot": slot_info,
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    summary["cached_share_display"] = _cache_share_line(summary)
    if isinstance(spec.get("routes"), dict) and isinstance(spec["routes"].get("roles"), dict):
        summary["route_costs"] = _route_costs(run, spec["routes"], sends, metrics)
    result = outcomes.terminal_receipt(run)
    summary["engine_result"] = result.get("result")
    summary["engine_terminal"] = {key: result[key] for key in (
        "result", "reason_code", "error", "error_type", "error_message", "failure_origin", "behaviour",
        "done_facts", "submission", "labels", "claims")
        if key in result}
    if isinstance(result.get("wrong_test_evidence"), dict):
        summary["wrong_test_evidence"] = result["wrong_test_evidence"]
    limitations = result.get("limitations")
    if isinstance(limitations, list):
        selected = [item for item in limitations if isinstance(item, str) and item]
        if selected:
            summary["limitations"] = selected
    if isinstance(result.get("authored_check_receipt_id"), str):
        summary["authored_check_receipt_id"] = result["authored_check_receipt_id"]
    for key in ("steps", "edits", "tests"):
        if key in result:
            summary[key] = result[key]
    return summary


def succeeded(summary: dict[str, object]) -> bool:
    """Use the same evidence and exit policy for CLI, in-place, and chat promotion."""
    return outcomes.exit_code(summary) == 0


def candidate_digest(root: Path) -> str:
    """Bind host checks to the materialized candidate, including unchanged context files."""
    # Tool litter the checks themselves write (.pytest_cache, __pycache__) is not candidate content.
    manifest = {}
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if (not p.is_file() or p.is_symlink()
                or any(workcopy._ignored(part, True) for part in rel.parts[:-1])
                or workcopy._ignored(rel.name, False)):
            continue
        manifest[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _time_limit_line(options: RunOptions) -> str:
    if options.timeout_s is None:
        return "time limit: none (stop any time with Ctrl-C)"
    return (f"time limit: {options.timeout_s:g}s total (--timeout), including copy, checks and "
            "finalization; stop earlier with Ctrl-C")


def _apply_resumed_time_limit(options: RunOptions, resume: dict, started: float,
                              timeout_s: float | None, printer) -> None:
    """Keep the run's own --timeout across a resume.

    The engine gets the time that was left at the run's newest recorded request, started once
    its replay ends (replaying recorded steps spends none of it). The runner's outer limit is
    the run's whole allowance from now, so a slow replay cannot outlive what the run was given.
    A --timeout passed to resume replaces all of this.
    """
    seconds_left = resume.get("seconds_left")
    if timeout_s is not None:
        resume.pop("seconds_left", None)
        resume.pop("time_allowance_seconds", None)
        return
    if not isinstance(seconds_left, (int, float)) or isinstance(seconds_left, bool):
        return
    allowance = float(resume.get("time_allowance_seconds") or seconds_left)
    options.timeout_s = max(allowance, float(seconds_left))
    options._deadline = started + options.timeout_s
    printer(f"time limit: this run's --timeout continues with {float(seconds_left):.0f}s left, "
            "counted from the end of replay; stop earlier with Ctrl-C")


def step_limit_restart_warning(old: int, new: int) -> str:
    """Resume with a different step limit: the limit is part of request 1, so nothing recorded can be reused."""
    def name(turns: int) -> str:
        return "none" if turns == 0 else f"{turns} model turns"
    return (f"WARNING: the step limit changed ({name(old)} -> {name(new)}). The limit is part of the first "
            "request, so the checkpoint and the recorded requests cannot be reused: this run restarts live "
            "from request 1 (the earlier attempt's files are kept under attempts/).")


def _step_limit_line(options: RunOptions) -> str:
    try:
        steps = options.max_turns if options.max_turns is not None else profile_steps(options.profile)
    except Exception:  # an unreadable profile is reported by the run itself; never fail the banner
        return "step limit: unknown (profile unreadable)"
    source = "--max-turns" if options.max_turns is not None else "profile"
    if steps == 0:
        return f"step limit: none ({source}); stop any time with Ctrl-C"
    return f"step limit: {steps} model turns ({source}); stop any time with Ctrl-C"


def run(options: RunOptions, printer=print) -> dict[str, object]:
    if sys.platform != "linux":
        if options.mutations is not None:
            level = options.mutations
        elif "STATEFRAME_MUTATIONS" in options.extra_env:
            level = options.extra_env["STATEFRAME_MUTATIONS"]
        elif "STATEFRAME_MUTATIONS" in os.environ:
            level = os.environ["STATEFRAME_MUTATIONS"]
        else:
            level = profiles.load(options.profile).get("STATEFRAME_MUTATIONS", "off")
        requested = str(level).strip().lower() != "off" or bool(options.mutation_hooks)
        if requested:
            from .mutate_host import mutation_platform_error

            refusal = mutation_platform_error(sys.platform, requested=True)
            if refusal is not None:
                raise RunError(refusal)
    _validate_timeout(options.timeout_s)
    _startup_probe_seconds()
    started = time.monotonic()
    options._deadline = started + options.timeout_s if options.timeout_s is not None else None
    try:
        with _wall_deadline(options):
            options._context_tokens, options._prompt_budget, how, options._context_env = resolve_context(
                profiles.load(options.profile))
            options._context_display = how
            with _sigterm_as_interrupt():
                return _run(options, started, printer)
    finally:
        options._deadline = None


def _compare_resume_routes(recorded: object, current: routes.RouteTable | None,
                           cli: list[str], *, old_default: tuple[str, str] | None = None,
                           new_default: tuple[str, str] | None = None,
                           unresolved_auto: bool = False) -> list[dict[str, object]]:
    allowed: set[tuple[str, str]] = set()
    for entry in cli:
        left = entry.partition("=")[0]
        role, _, field = left.rpartition(".")
        if field not in ("endpoint", "model") or not routes.valid_role(role):
            raise resume_module.ResumeRefused(
                f"--route {entry!r} can only change ROLE.endpoint or ROLE.model; remove it to resume")
        allowed.add((role, field))
    if recorded is None:
        if cli:
            role, _, field = cli[0].partition("=")[0].rpartition(".")
            raise resume_module.ResumeRefused(
                f"this run started without per-step models; --route {role}.{field} is set now; remove it to resume")
        if current is None or not current.active:
            return []
        if current.first_change is None:
            raise resume_module.ResumeRefused("this run started without per-step models; remove routes to resume")
        role, field = current.first_change
        env_name = f"STATEFRAME_ROUTE_{role.upper().replace('.', '_')}_{field.upper()}"
        raise resume_module.ResumeRefused(
            f"this run started without per-step models; route {role}.{field} is set now "
            f"({env_name} or a routes file); remove it to resume")
    old_roles = recorded.get("roles", {}) if isinstance(recorded, dict) else {}
    new_roles = current.roles if current is not None else {}
    old_masks = recorded.get("unit_overrides", {}) if isinstance(recorded, dict) else {}
    new_masks = current.unit_overrides() if current is not None else {}
    for role in sorted(set(old_masks) | set(new_masks)):
        before_fields = set(old_masks.get(role, ()))
        after_fields = set(new_masks.get(role, ()))
        for field in sorted(before_fields ^ after_fields):
            raise resume_module.ResumeRefused(
                f"route {role}.{field} changed (unit override: "
                f"run: {field in before_fields}, now: {field in after_fields}); unset it to resume")
    changes: list[dict[str, object]] = []
    for role in sorted(set(old_roles) | set(new_roles)):
        old = old_roles.get(role, {})
        new = new_roles.get(role, {})
        for field in sorted(set(old) | set(new)):
            before, after = old.get(field), new.get(field)
            if before == after:
                continue
            if unresolved_auto and field == "slot" and after == "auto" and type(before) is int:
                continue
            inherited_move = (old_default is not None and new_default is not None and
                              field in ("endpoint", "model") and
                              before == old_default[0 if field == "endpoint" else 1] and
                              after == new_default[0 if field == "endpoint" else 1])
            if (role, field) in allowed or inherited_move:
                changes.append({"role": role, "field": field, "from": before, "to": after})
                continue
            raise resume_module.ResumeRefused(
                f"route {role}.{field} changed (run: {before}, now: {after}); unset it to resume")
    return changes


def _compose_resume_route_changes(spec: dict, changes: list[dict[str, object]]) -> None:
    """Keep moves needed by carried tape, then apply this resume's moves in order."""
    if not changes or spec["resume"].get("restart_live"):
        return
    prior = spec["resume"].get("route_changes", [])
    if prior[-len(changes):] != changes:
        spec["resume"]["route_changes"] = [*prior, *changes]


def resume_run(run_dir: Path, *, endpoint: Endpoint | None = None, max_turns: int | None = None,
                timeout_s: float | None = None, quiet: bool = False, model: str | None = None,
                keep_checkpoints: bool = False, keep_run_env: bool = False, replay_external: bool = False,
                route_cli: list[str] | None = None, routes_file: Path | None = None,
                printer=print) -> dict[str, object]:
    """Continue an interrupted run in place (see ``stateframe.resume``).

    ``model`` continues under another model NAME the endpoint serves (the same weights listed differently,
    e.g. after moving servers); without it the run's own model name is required.
    """
    if sys.platform != "linux":
        try:
            saved_spec = json.loads((Path(run_dir) / "spec.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved_spec = {}
        if isinstance(saved_spec, dict):
            from .mutate_host import mutation_platform_error, spec_requests_mutations

            refusal = mutation_platform_error(
                sys.platform, requested=spec_requests_mutations(saved_spec))
            if refusal is not None:
                raise resume_module.ResumeRefused(refusal)
    _validate_timeout(timeout_s)
    started = time.monotonic()
    run_dir = run_dir.resolve()
    state = resume_module.assess(run_dir, max_turns=max_turns, model=model,
                                 replay_external=replay_external)
    spec: dict = state["spec"]  # type: ignore[assignment]
    if "powerups" in spec:
        from . import powerups
        try:
            powerups.verify_records(spec["powerups"])
        except powerups.PowerupError as error:
            raise resume_module.ResumeRefused(str(error)) from error
    try:
        verify_records(spec.get("skills", []), spec.get("skills_digest", skills_digest([])),
                       spec.get("skills_measurements"))
    except SkillError as error:
        raise resume_module.ResumeRefused(str(error)) from error
    if "project_instructions" in spec:
        try:
            project_instructions.verify_unchanged(spec["project_instructions"])
        except project_instructions.InstructionsError as error:
            raise resume_module.ResumeRefused(str(error)) from error
    start_context_missing = "hooks" in spec and "start_context" not in spec["hooks"]
    if start_context_missing:
        from .hook_host import rebuild_start_context
        try:
            context = rebuild_start_context(run_dir, spec["hooks"],
                                            run_seed=str(spec["run_identity_seed"]))
        except ValueError as error:
            raise resume_module.ResumeRefused(str(error)) from error
        if context is not None:
            spec["hooks"]["start_context"] = context
            _write_hook_spec(run_dir, spec)
    if "tool_settings" in spec:
        try:
            settings = ToolSettings.from_dict(spec["tool_settings"])
        except ValueError as error:
            raise resume_module.ResumeRefused(f"recorded tool settings are invalid: {error}") from error
        builtin_names = frozenset(("grep", "list_files", "read", "edit", "str_replace", "insert_before",
            "authored_check", "file_create", "run_tests", "read_check_receipt", "recall", "done",
            "shell", "job_wait", "job_output", "job_stop", "file_delete", "file_move", "git", "web_fetch", "git_commit", "git_branch", "skill_load"))
        builtin_names |= mutation_builtin_names(spec)
        for item in (*settings.tools_files, *settings.mcp_files):
            path = Path(item["path"])
            try:
                unchanged = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item["digest"]
            except OSError:
                unchanged = False
            if not unchanged:
                raise resume_module.ResumeRefused(f"custom tool definition changed: {path}")
        try:
            custom = load_tool_files([Path(item["path"]) for item in settings.tools_files],
                                     builtin_names=builtin_names, shell_network=settings.shell_network)
            mcp_servers = load_mcp_files([Path(item["path"]) for item in settings.mcp_files],
                                         shell_network=settings.shell_network)
        except CustomToolError as error:
            raise resume_module.ResumeRefused(f"custom tool changed: {error}") from error
        if custom.digest != spec.get("custom_tools_digest"):
            raise resume_module.ResumeRefused("custom tool executable or definition changed")
        if mcp_servers:
            external = [server.name for server in mcp_servers.servers if server.replay == "external"]
            if external and not replay_external:
                raise resume_module.ResumeRefused(
                    "MCP discovery requires --replay-external for servers: " + ", ".join(external))
            try:
                mcp = discover_mcp(mcp_servers, run_dir / "workspace", run_dir,
                                   shell_network=settings.shell_network,
                                   host_authorization=settings.host_authorization is not None,
                                   custom_schemas=custom.schemas(), endpoint=spec.get("chat_url"),
                                   auth_key=(os.environ.get(spec.get("api_key_env"), "")
                                             if isinstance(spec.get("api_key_env"), str) else None),
                                   generation_endpoint=spec.get("chat_url"))
            except (CustomToolError, OSError, ValueError) as error:
                raise resume_module.ResumeRefused(f"MCP discovery changed or failed: {error}") from error
            if mcp.digest != spec.get("mcp_tools_digest") or [t.name for t in mcp.tools] != spec.get("mcp_tools") or list(mcp.omitted) != spec.get("mcp_omitted"):
                raise resume_module.ResumeRefused("MCP tool list or schema changed")
        settings = replace(settings, replay_external=replay_external)
    elif replay_external:
        raise resume_module.ResumeRefused("this run has no tool tape; --replay-external cannot apply")
    environment = spec["environment"]
    keys = [k for k in str(environment.get("STATEFRAME_TASK_ENV_KEYS", "")).split(",") if k]
    base = spec["chat_url"].removesuffix(CHAT_SUFFIX)
    wanted = model or spec["model"]
    if endpoint is None:
        recorded_key = spec.get("api_key_env")
        if recorded_key and recorded_key not in os.environ:
            raise RunError(f"this run was started with --api-key-env {recorded_key}, and {recorded_key} is not "
                           "set now; set it again, or pass --api-key-env NAME to use another variable")
        endpoint = Endpoint.from_args(base, wanted, recorded_key)
    elif endpoint.model != wanted:
        raise RunError(f"this run used model {spec['model']!r}; resume continues it with the same model "
                       "(pass --model NAME to continue under the name your endpoint serves)")
    swap = state.get("model_swap")
    options = RunOptions(
        repo=Path(spec["repo"]), task=str(spec["public_row"].get("msg", "")), endpoint=endpoint,
        max_turns=int(state["max_turns"]), test_cmd=spec.get("test_cmd"), run_dir=run_dir,
        check_cmd=spec.get("check_cmd"), check_source=spec.get("check_source"),
        runner_kind=str(spec.get("runner_kind") or "auto"),
        protected_test_globs=list(spec.get("protected_test_globs") or ()),
        protect_tests=spec.get("protect_tests", "all"),
        in_place=bool(spec.get("in_place")), profile=str(spec.get("profile") or profiles.DEFAULT),
        python=str((spec.get("capabilities") or {}).get("python") or sys.executable),
        timeout_s=timeout_s, quiet=quiet, test_env={k: environment[k] for k in keys if k in environment},
         pytest_plugins=environment.get("STATEFRAME_TASK_PYTEST_PLUGINS"),
         keep_checkpoints=keep_checkpoints, keep_run_env=keep_run_env,
         allow_source_tree=bool(spec.get("allow_source_tree", False)))
    options.route_cli = route_cli or []
    options.routes_file = routes_file
    if "tool_settings" in spec:
        options.tool_settings = settings
    options._deadline = started + timeout_s if timeout_s is not None else None
    try:
        with _wall_deadline(options), _sigterm_as_interrupt():
            # A run that had a --timeout prints the time it has left once its attempt is archived.
            if timeout_s is not None or resume_module.replay_tape.recorded_countdown_window(
                    run_dir / "wire") is None:
                printer(_time_limit_line(options))
            printer(_step_limit_line(options))
            options.endpoint.check_locality()
            options.python = _check_python(options.python,
                                           require_pytest=options.runner_kind != "generic")
            options._capabilities = dict(spec.get("capabilities") or {})
            started_env = spec.get("environment") or {}
            options._context_tokens = int(started_env.get(CONTEXT_TOKENS_ENV) or DEFAULT_CONTEXT_TOKENS)
            options._prompt_budget = int(started_env.get(PROMPT_BUDGET_ENV)
                                         or derived_prompt_budget(options._context_tokens))
            for name in CONTEXT_ENV_NAMES:
                now = os.environ.get(name, "").strip()
                if now and now != str(started_env.get(name, "")):
                    raise RunError(f"{name}={now} differs from the run's start ({started_env.get(name) or 'unset'}); "
                                   "a resumed run keeps the context size it started with. Unset it to resume")
            # Resume keeps the run's recorded test interpreter (older runs: the engine's).
            options._test_python = str(options._capabilities.get("test_python") or options.python)
            probe = replace(options.endpoint,
                            timeout_s=min(options.endpoint.timeout_s, _remaining(options))).probe()
            if probe.get("model_found") is False:
                raise RunError(f"endpoint does not serve model {options.endpoint.model!r}; it lists: "
                               + ", ".join(probe.get("served_models") or [])
                               + ("" if model else " (if it is the same model under another name, "
                                  "pass --model NAME)"))
            check_slot_context(options, printer)
            slot = spec.get("prompt_cache_slot") or {}
            if isinstance(slot.get("slot"), int):
                options._cache_slot, options._cache_slot_selection = slot["slot"], "resumed"
            old_default = (base, str(spec["model"]))
            new_default = (options.endpoint.base_url, options.endpoint.model)
            options._recorded_routes = spec.get("routes")
            options._recorded_route_default = _route_defaults(options)
            options._recorded_route_default.update(endpoint=base, model=str(spec["model"]),
                                                   allow_remote=not _is_loopback(base))
            try:
                current_routes = _resolve_routes(options)
                _compare_resume_routes(spec.get("routes"), current_routes, options.route_cli,
                                       old_default=old_default, new_default=new_default, unresolved_auto=True)
                _probe_routes(options, printer, resume_roles=(spec.get("routes") or {}).get("roles"))
            except routes.RecordedRouteError as error:
                raise resume_module.ResumeRefused(str(error)) from None
            route_changes = _compare_resume_routes(spec.get("routes"), options._routes, options.route_cli,
                                                   old_default=old_default, new_default=new_default)
            lock = resume_module.hold_run_lock(run_dir)  # noqa: F841 - held until this resume returns
            if "tool_settings" in spec:
                _write_json_atomic(run_dir / "tool-settings.json", settings.as_dict(), mode=0o600)
                spec["tool_settings"] = settings.as_dict()
                _write_run_spec_atomic(run_dir, spec)
            # A checkpoint holds the engine's loop limit, endpoint and model name as they were; a resume
            # that changes any of them rebuilds the run by replay instead.
            same_limits = (int(state["max_turns"]) == int(spec["max_turns"])
                           and endpoint.chat_url == spec["chat_url"] and not isinstance(swap, dict)
                           and not route_changes)
            checkpoint_off = checkpoint_resume_off(spec)
            hooks_configured = "hooks" in spec
            routes_configured = "routes" in spec or options._routes is not None
            # Widened edit scope lives in engine memory, not in a checkpoint; replay rebuilds it.
            widen_configured = edit_scope_widened(spec)
            # Shell-enabled runs restore too: a checkpoint copies the run directory (workspace, run
            # environment, shell home and baseline), saves the execution session unstarted, and is
            # refused while a job is live; recorded exchanges after it replay through the tool tape.
            checkpoint_eligible = same_limits and not checkpoint_off
            checkpoint_skip_reason = ("mutations_configured" if "mutations" in spec else
                                      "hooks_configured" if hooks_configured else
                                      "routes_configured" if routes_configured else
                                      "edit_scope_widen" if widen_configured else None)
            checkpoint_skipped = checkpoint_eligible and checkpoint_skip_reason is not None
            if checkpoint_off:
                spec["environment"]["STATEFRAME_CHECKPOINT"] = "off"
            if int(state["max_turns"]) != int(spec["max_turns"]):
                printer(step_limit_restart_warning(int(spec["max_turns"]), int(state["max_turns"])))
            previous = resume_module.archive_attempt(
                state, ENGINE_DIR, use_checkpoint=checkpoint_eligible and not checkpoint_skipped,
                replay_external=replay_external)
            spec = _resume_spec(run_dir, endpoint)
            if checkpoint_skipped:
                spec["resume"]["checkpoint"] = {"status": "skipped", "reason": checkpoint_skip_reason}
                events = run_dir / "checkpoints" / "events.jsonl"
                events.parent.mkdir(parents=True, exist_ok=True)
                with events.open("a") as stream:
                    stream.write(json.dumps({"ts": round(time.time(), 3), "event": "checkpoint_skipped",
                                             "reason": checkpoint_skip_reason}) + "\n")
            if options._routes is not None:
                spec["routes"] = options._routes.public()
            _compose_resume_route_changes(spec, route_changes)
            _apply_resumed_time_limit(options, spec["resume"], started, timeout_s, printer)
            _write_run_spec_atomic(run_dir, spec)
            # The cache slot stays the one the run started with: it is part of the request bytes.
            slot = spec.get("prompt_cache_slot") or {}
            if isinstance(slot.get("slot"), int):
                options._cache_slot, options._cache_slot_selection = slot["slot"], "resumed"
            printer(f"resuming run {run_dir} (attempt {spec['resume']['attempt']}; previous attempt "
                    f"stopped with {state['previous_status'] or 'no engine result'}, files kept in {previous})")
            if spec["resume"].get("pending_check", {}).get("status") == "skipped":
                printer("resume: the in-flight request of an earlier resume could not be proven; "
                        "resuming by full replay without that check")
            if isinstance(swap, dict):
                printer(f"MODEL NAME SWAP: this run was recorded with model {', '.join(map(repr, swap['recorded']))}"
                        f" and continues with {swap['served']!r}, the name the endpoint serves. This is only "
                        "correct if it is the SAME model. Replay compares each request with its model field "
                        "set back to the recorded name; nothing else is changed (recorded in spec.json, "
                        "summary.json and any resume-divergence.json)")
            checkpoint = spec["resume"].get("checkpoint")
            if spec["resume"].get("restart_live"):
                printer("restart live: the changed step limit starts at request 1; previous model and tool "
                        "exchanges remain archived")
            elif checkpoint and checkpoint.get("status") != "skipped":
                printer(f"checkpoint: engine state restored as of send {checkpoint['sends']} (step "
                        f"{checkpoint['step']}); {checkpoint['tail']} later recorded exchange(s) are replayed "
                        "and checked byte for byte, then the run continues live")
            else:
                if checkpoint_skipped:
                    if checkpoint_skip_reason == "mutations_configured":
                        printer("checkpoint: not restored because mutations are configured for this run; "
                                "replaying the recorded exchanges instead")
                    elif checkpoint_skip_reason == "hooks_configured":
                        printer("checkpoint: not restored because hooks are configured for this run; "
                                "replaying the recorded exchanges instead")
                    elif checkpoint_skip_reason == "edit_scope_widen":
                        printer("checkpoint: not restored because edit scope widening is on for this run; "
                                "replaying the recorded exchanges instead")
                    else:
                        printer("checkpoint: not restored because per-step models are configured for this run; "
                                "replaying the recorded exchanges instead")
                printer(f"replay: {state['tape_count']} recorded model exchanges are answered from the run's "
                        "records, then the run continues live")
            from .shell_host import provision
            provision(options, run_dir, spec)
            work = resume_module.fresh_workcopy(run_dir, options.repo)
            if start_context_missing and "start_context" not in spec["hooks"]:
                from . import __version__, hook_host
                try:
                    spec["hooks"]["start_context"] = hook_host.resume_start_hooks(
                        run_dir,
                        spec["hooks"],
                        work.copy,
                        task=options.task,
                        repo=options.repo,
                        shell_network=options.tool_settings.shell_network,
                        run_seed=str(spec["run_identity_seed"]),
                        run_id=str(spec["run_identity_seed"]),
                        version=__version__,
                        printer=printer,
                    )
                    _write_hook_spec(run_dir, spec)
                except hook_host.HookRunRefused as error:
                    raise resume_module.ResumeRefused(f"hook refused run: {error}") from error
                except hook_host.HookContainmentError as error:
                    raise RunError(f"HOOK_CONTAINMENT: {error}") from error
                except (hook_host.HookConfigError, OSError, ValueError) as error:
                    raise resume_module.ResumeRefused(str(error)) from error
            summary = _execute(options, started, printer, run_dir, work, spec)
            refused = (summary.get("checkpoint_refused") or _checkpoint_refusal(run_dir, summary.get("child_exit_code"))
                       if checkpoint and checkpoint.get("status") != "skipped" else None)
            if refused is None:
                return summary
            # Fail open: a checkpoint that cannot be restored faithfully is never continued from. Keep the
            # refused attempt's files and replay the interrupted attempt from the start instead.
            printer(f"checkpoint restore refused ({refused}); falling back to replaying the run")
            state["attempt"] = int(state["attempt"]) + 1
            resume_module.archive_attempt(state, ENGINE_DIR, use_checkpoint=False, tape_from=previous,
                                          replay_external=replay_external)
            spec = _resume_spec(run_dir, endpoint)
            if options._routes is not None:
                spec["routes"] = options._routes.public()
            _compose_resume_route_changes(spec, route_changes)
            spec["resume"]["checkpoint_refused"] = refused
            _write_run_spec_atomic(run_dir, spec)
            from .shell_host import provision
            provision(options, run_dir, spec)
            work = resume_module.fresh_workcopy(run_dir, options.repo)
            return _execute(options, started, printer, run_dir, work, spec)
    finally:
        options._deadline = None


def _resume_spec(run_dir: Path, endpoint: Endpoint) -> dict:
    spec = json.loads((run_dir / "spec.json").read_text())
    if endpoint.chat_url != spec["chat_url"]:
        spec["chat_url"] = endpoint.chat_url
        spec["environment"]["DS_URL"] = endpoint.chat_url
        _write_run_spec_atomic(run_dir, spec)
    return spec


# The engine child's exit status for "checkpoint restore refused" (agreed with the engine in
# logs/team_build/evictfix_par_notes/LINUXCKPT_requests.md): the diagnostic file can fail to be written.
CHECKPOINT_REFUSED_EXIT = 75


def _checkpoint_refusal(run_dir: Path, child_exit: object = None) -> str | None:
    """Why this attempt refused its checkpoint restore (engine or replay check), or None.

    Refusal paths write ``checkpoint-refused.json`` before anything is sent live, and the engine also exits
    with CHECKPOINT_REFUSED_EXIT, so a refusal whose file could not be written still falls back to replay.
    """
    path = run_dir / "checkpoint-refused.json"
    try:
        present = path.is_file()
    except OSError:
        present = False
    if not present:
        if child_exit == CHECKPOINT_REFUSED_EXIT:
            return f"engine refused the checkpoint (exit status {CHECKPOINT_REFUSED_EXIT}; no refusal record)"
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return "checkpoint refusal record unreadable"
    if not isinstance(record, dict):
        return "checkpoint refusal record malformed"
    return f"{record.get('error')}: {record.get('reason') or record.get('detail') or ''}".strip()


def _resolve_project_instructions(options: RunOptions, workspace: Path, environment: dict,
                                  printer=print) -> dict[str, object] | None:
    """Bounded, digest-pinned AGENTS.md / CLAUDE.md for this run (None: no slot, bytes unchanged)."""
    mode = options.instructions
    explicit = None if mode in ("auto", "off") else Path(mode)
    loop = str(options.extra_env.get("STATEFRAME_AGENT_LOOP", environment.get("STATEFRAME_AGENT_LOOP", "full"))).lower()
    lean = str(options.extra_env.get("STATEFRAME_LEAN", environment.get("STATEFRAME_LEAN", "off"))).lower() == "on"
    try:
        record = project_instructions.resolve(
            options.repo, explicit, discover=(mode == "auto"),
            # The working copy applies the sandbox read policy: an excluded file is never shown.
            readable=lambda relative: (workspace / relative).is_file())
    except project_instructions.InstructionsError as error:
        raise RunError(str(error)) from error
    if record is not None and (loop == "plain" or lean):
        if explicit is not None:
            raise RunError("--instructions requires the composed loop")
        if not options.quiet:
            printer(f"project instructions {record['source']} skipped: this loop does not support them")
        return None
    if record is not None and not options.quiet:
        printer(f"project instructions: {record['source']} ({record['shown_bytes']} of {record['bytes']} bytes"
                + (", truncated" if record["truncated"] else "") + ")")
    return record


def _session_end_is_final(summary: dict[str, object]) -> bool:
    # session_end belongs to the logical run's terminal attempt. Incomplete and cancelled attempts
    # may be resumed, and replay must not collide with a second session_end event id.
    terminal = summary.get("engine_terminal") or {}
    result = terminal.get("result") if isinstance(terminal, dict) else None
    return isinstance(result, str) and (result in {"SUBMITTED", "HOOK_BLOCKED", "CONTINUATION_BLOCKED"}
                                        | outcomes.SEAL_FAILURES)


def _run(options: RunOptions, started: float, printer=print) -> dict[str, object]:
    if options.endpoint is None:
        raise RunError("an endpoint is required")
    options.endpoint.check_locality()
    try:
        if options.test_cmd and options.check_cmd:
            raise ValueError("pass either --test-cmd or --check-cmd, not both")
        if options.check_cmd:
            options.test_cmd = options.check_cmd
            options.runner_kind = "generic"
            options.check_source = "command"
        elif options.test_cmd:
            options.check_source = DECLARED_TEST_SOURCE
        else:
            detected, source, has_tests = discover_check(options.repo)
            if detected:
                options.test_cmd = detected
                options.runner_kind = "generic"
                options.check_source = source
            elif not has_tests:
                options.runner_kind = "generic"
                options.check_source = "none"
            else:
                options.check_source = DECLARED_TEST_SOURCE
        options.runner_kind, command_argv = parse_command(options.test_cmd, options.runner_kind)
        if options.runner_kind == "generic" and not command_argv and options.check_source != "none":
            raise ValueError("a generic runner requires --test-cmd")
    except ValueError as error:
        raise RunError(str(error)) from error
    _check_engine_python(options)
    options._provider_capabilities.clear()
    options._provider_route_dialects.clear()
    options._capabilities = _probe_capabilities(options)
    probe = replace(options.endpoint, timeout_s=min(options.endpoint.timeout_s, _remaining(options))).probe()
    if probe.get("model_found") is False:
        raise RunError(f"endpoint does not serve model {options.endpoint.model!r}; it lists: "
                       + ", ".join(probe.get("served_models") or []))
    preflight_messages: list[str] = []
    preflight_warnings: list[str] = []
    check_slot_context(options, preflight_messages.append)
    _select_prompt_cache_slot(options, preflight_messages.append)
    _probe_routes(options, preflight_messages.append, warnings=preflight_warnings)
    printer(f"{options._capabilities['display']}; {_provider_preflight_summary(options)}")
    _print_run_banner(options, printer)
    for message in preflight_messages:
        printer(message)
    for warning in preflight_warnings:
        print(warning, file=sys.stderr)
    options._test_python = _select_test_python(options, printer)
    run_dir, work, spec = prepare(options, printer=printer)
    from .shell_host import provision
    provision(options, run_dir, spec)
    printer(f"run directory: {run_dir}")
    lock = resume_module.hold_run_lock(run_dir)  # noqa: F841 - held until this run returns
    return _execute(options, started, printer, run_dir, work, spec)


class Terminated(KeyboardInterrupt):
    """SIGTERM: stop like Ctrl-C (engine group stopped, summary written), so the run can be resumed."""


@contextmanager
def _sigterm_as_interrupt():
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    received = []

    def terminated(signum, frame):
        received.append(signum)
        raise Terminated()

    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
    if received:
        # The run is stopped and its summary written; now end the process, as a plain `kill` asks.
        # (Chat would otherwise treat it as Ctrl-C on one task and keep running.)
        raise SystemExit(130)


def _resume_record(run_dir: Path, resume: dict, summary: dict) -> dict[str, object]:
    """What this resume actually did: how much of the recorded run it replayed, and whether faithfully."""
    tape = len(resume.get("tape") or [])
    replayed = int(summary.get("replayed_sends") or 0)
    divergence = None
    path = run_dir / "resume-divergence.json"
    if path.is_file():
        try:
            divergence = json.loads(path.read_text())
        except (OSError, ValueError):
            divergence = {"error": "resume-divergence.json unreadable"}
    return {"attempt": resume.get("attempt"), "previous_attempt": resume.get("previous_attempt"),
            "recorded_exchanges": tape, "replayed": replayed,
            # Faithful = every recorded exchange was replayed against a byte-matching rebuilt request.
            "faithful": divergence is None and replayed == tape,
            "went_live_at_recorded_exchange": None if divergence is None else divergence.get("tape_position"),
            "divergence": divergence, "model_swap": resume.get("model_swap")}


def _execute(options: RunOptions, started: float, printer, run_dir: Path, work: workcopy.WorkCopy,
             spec: dict) -> dict[str, object]:
    def atomic_bytes(path: Path, data: bytes) -> None:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.",
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    printer(f"working copy: {work.copy} ({len(work.baseline)} files; original untouched"
            f"{' until the end (--in-place)' if options.in_place else ''})")
    if options.runner_kind != "generic":
        printer(f"starting file hint: {spec['public_row']['entry']['path']}; "
                f"checks: {', '.join(spec['public_row']['public_check_paths'])}")
    else:
        printer(f"starting file hint: {spec['public_row']['entry']['path']}; "
                f"check: {options.test_cmd}")
    test_cmd = options.test_cmd or ("python -m pytest -q " + " ".join(
        shlex.quote(p) for p in spec["public_row"]["public_check_paths"])) \
        if options.runner_kind != "generic" else options.test_cmd
    before = ({"command": None, "exit_code": None, "passed": None, "tail": "no check available"}
              if test_cmd is None else run_test_cmd(
                  test_cmd.replace("python ", shlex.quote(_tests_python(options)) + " ", 1)
                          if test_cmd.startswith("python ") else test_cmd, work.copy,
                          test_env=options.test_env, python=_tests_python(options),
                          pytest_plugins=options.pytest_plugins, deadline=options._deadline,
                          timeout_s=min(900, _remaining(options)), test_python=_outer_test_python(options),
                           runner_kind=options.runner_kind,
                           capture_output=options.protect_tests in {"task", "off"}))
    printer(f"before: {test_cmd or '(none)'} -> "
            f"{'PASS' if before['passed'] else 'FAIL' if before['passed'] is False else 'NOT RUN'} "
            f"(exit {before['exit_code']})")
    if options.protect_tests == "task" and before["passed"] is False:
        row = spec["public_row"]
        lock_initial_failing_tests(row, before)
        _write_json_atomic(run_dir / "spec.json", spec)
    if options.protect_tests in {"task", "off"} or options.runner_kind == "generic":
        from .test_runners import failing_test_nodes, is_test_file, judging_test_selection
        row = spec["public_row"]
        check_argv = parse_command(test_cmd, options.runner_kind)[1] if test_cmd else ()
        test_paths = tuple(row.get("test_file_paths") or
                           sorted(path for path in work.baseline if is_test_file(path)))
        failed_output = (str(before.get("output") or before.get("tail") or "")
                         if before.get("passed") is False else "")
        selection = judging_test_selection(test_paths, check_argv, failed_output=failed_output)
        row["test_wrong_failed_nodes"] = failing_test_nodes(test_paths, failed_output)
        for index, arg in enumerate(check_argv):
            if arg in {"-k", "-m"} and index + 1 < len(check_argv):
                row[f"test_wrong_{'keyword' if arg == '-k' else 'mark'}"] = check_argv[index + 1]
        original = work.snapshot or work.original
        row["test_wrong_selectors"] = selection
        row["test_wrong_sources"] = {
            path: (original / path).read_text(encoding="utf-8", errors="replace")
            for path in selection}
        row["test_wrong_hashes"] = {path: work.baseline[path] for path in selection}
        _write_json_atomic(run_dir / "spec.json", spec)
    cancelled = False
    _remaining(options)
    try:
        child_exit = drive(run_dir, options, printer)
    except KeyboardInterrupt:
        child_exit, cancelled = 130, True
    except OSError as error:
        raise RunError(f"could not start engine: {error}") from error
    checkpoint = (spec.get("resume") or {}).get("checkpoint")
    if checkpoint and checkpoint.get("status") != "skipped" and not cancelled:
        refused = _checkpoint_refusal(run_dir, child_exit)
        if refused is not None:
            # A refused checkpoint restore sent nothing live and produced no result: hand straight back to the
            # caller's fall-back-to-replay. Diffing or checking this attempt first could fail (10-28: a missing
            # start snapshot raised in the diff) and then the fallback never ran.
            return {"checkpoint_refused": refused, "child_exit_code": child_exit, "run_dir": str(run_dir)}
    terminal = outcomes.terminal_receipt(run_dir)
    cancelled = cancelled or child_exit in (130, -2) or terminal.get("result") in ("CANCELLED", "CANCELED")
    declared = set(spec["public_row"]["files"])
    candidate, written, candidate_error = None, [], None
    dropped_paths: list[str] = []
    artifact_digest = None
    try:
        candidate = final_candidate(run_dir)
        if candidate is not None:
            artifact_digest = _artifact_digest(candidate)
            if edit_scope_widened(spec):
                declared.update(_admitted_candidate_paths(run_dir, spec, work, candidate))
            dropped_paths = _dropped_existing_candidate_paths(work, candidate, declared)
            if dropped_paths:
                candidate_error = "sealed candidate changed existing files outside edit scope"
                printer("warning: candidate changes outside edit scope were dropped: "
                        + ", ".join(dropped_paths))
            else:
                written = workcopy.apply_files(work, candidate, declared=declared)
    except RunDeadlineExceeded:
        raise
    except (RunError, workcopy.WorkCopyError, OSError, ValueError) as error:
        candidate_error = str(error)
    digest = candidate_digest(work.copy) if candidate is not None and not candidate_error else None
    after = {"command": test_cmd, "exit_code": None, "passed": None, "tail": "checks not run"}
    if (digest and not cancelled and test_cmd is not None
            and terminal.get("result") != "TEST_REPORTED_WRONG"):
        try:
            after = run_test_cmd(test_cmd.replace("python ", shlex.quote(_tests_python(options)) + " ", 1)
                                 if test_cmd.startswith("python ") else test_cmd, work.copy,
                                 test_env=options.test_env, python=_tests_python(options),
                                 pytest_plugins=options.pytest_plugins, deadline=options._deadline,
                                 timeout_s=min(900, _remaining(options)),
                                  test_python=_outer_test_python(options), runner_kind=options.runner_kind,
                                  capture_output=True)
            if candidate_digest(work.copy) == digest:
                after["candidate_sha256"] = digest
            else:
                candidate_error = "candidate changed while host checks ran; checks are stale"
        except KeyboardInterrupt:
            cancelled = True
    diff = workcopy.unified_diff(work)
    atomic_bytes(run_dir / "final.diff", diff.encode("utf-8"))
    commits = None
    if (run_dir / "commits.jsonl").is_file():
        from .commit_export import CommitExportError, export_commits
        try:
            commits = export_commits(run_dir)
        except (CommitExportError, OSError, ValueError, subprocess.TimeoutExpired) as error:
            commits = {"error": str(error)}
    summary = summarize(run_dir, started)
    queue_wait_warning = _queue_wait_warning(run_dir)
    if queue_wait_warning is not None:
        summary["provider_queue_wait_warning"] = queue_wait_warning
    printer(str(summary["cached_share_display"]))
    if commits is not None:
        summary["commits"] = commits
        if "error" in commits:
            printer(f"commits: export failed: {commits['error']}")
        else:
            printer(f"commits: {commits['count']} in {commits['dir']} "
                    f"(apply a branch with: git am {commits['dir']}/<branch>/*.patch)")
    summary.update({
        "run_dir": str(run_dir),
        "child_exit_code": child_exit,
        "candidate_captured": candidate is not None,
        "candidate_sha256": digest,
        "candidate_artifact_digest": artifact_digest,
        "candidate_error": candidate_error,
        "candidate_paths_dropped": dropped_paths,
        "candidate_files_written": written,
        "files_changed": sorted(workcopy.changed_files(work)),
        "tests_before": {k: before[k] for k in ("command", "exit_code", "passed")},
        "tests_after": {k: after[k] for k in ("command", "exit_code", "passed", "candidate_sha256")
                        if k in after},
        "runner_kind": spec.get("runner_kind", "pytest"),
        "check_source": options.check_source,
        **({"test_lock_note": spec["public_row"]["test_lock_note"]}
           if "test_lock_note" in spec["public_row"] else {}),
        "protect_tests": spec.get("protect_tests", "all"),
        "endpoint": options.endpoint.public(),
        "provider_capabilities": _provider_preflight_records(options),
        "profile": options.profile,
        "engine_source_hash": spec["engine_source"].get("source_hash"),
        "engine_shipped_hash": spec["engine_source"].get("shipped_hash"),
    })
    if options.protect_tests in {"task", "off"}:
        test_changes = inspect_test_changes(work)
        test_paths = {item["path"] for item in test_changes}
        summary["test_files_changed"] = test_changes
        summary["product_files_changed"] = sorted(set(summary["files_changed"]) - test_paths)
        summary["test_weakening_findings"] = [finding for item in test_changes
                                               for finding in item["weakening"]]
    if isinstance(spec.get("resume"), dict):
        summary["resume"] = _resume_record(run_dir, spec["resume"], summary)
    atomic_bytes(run_dir / "tests-after.txt", after["tail"].encode("utf-8"))
    summary.update(outcomes.project(
        summary, cancelled=cancelled, timed_out=child_exit == 124,
        check_output=str(after.get("output") or after.get("tail") or ""),
        check_repo=work.copy,
        declared_test_command=declared_test_evidence(
            options.test_cmd, options.check_cmd, options.check_source, options.runner_kind)))
    summary["exit_code"] = outcomes.exit_code(summary)
    summary["succeeded"] = succeeded(summary)
    if options.in_place and diff:
        if summary["succeeded"]:
            try:
                summary["applied_in_place"] = workcopy.apply_in_place(work)
            except RunDeadlineExceeded as error:
                raise RunDeadlineExceeded(
                    f"{error}; --in-place may be partially applied; inspect the original and final.diff") from error
            except workcopy.WorkCopyError as error:
                summary["apply_error"] = str(error)
                summary["exit_code"] = 2
                summary["succeeded"] = False
            else:
                try:
                    _write_json_atomic(run_dir / "apply-receipt.json", {
                        "applied_files": summary["applied_in_place"],
                    })
                except OSError as error:
                    raise RunError(
                        f"in-place changes were applied, but the durable receipt could not be published: "
                        f"{error}") from error
        else:
            summary["applied_in_place"] = []
            summary["in_place_skipped"] = (
                f"candidate not promoted: runtime={summary['runtime_outcome']}, "
                f"checks={summary['check_outcome']}, artifact={summary['artifact']}")
    if "hooks" in spec:
        from . import __version__, hook_host
        containment_error = None
        if (summary.get("runtime_outcome") != "hook_containment"
                and _session_end_is_final(summary)):
            terminal = summary.get("engine_terminal") or {}
            stop_reason = terminal.get("reason_code") or terminal.get("result") or "unknown"
            end_data = {
                "outcome": str(summary.get("runtime_outcome") or "unknown"),
                "stop_reason": str(stop_reason),
                "applied": bool(summary.get("applied_in_place")),
                "changed_paths": list(summary.get("files_changed") or []),
                "run_dir": str(run_dir),
            }
            try:
                hook_host.run_session_end(
                    run_dir,
                    spec["hooks"],
                    work.copy,
                    data=end_data,
                    shell_network=options.tool_settings.shell_network,
                    run_seed=str(spec["run_identity_seed"]),
                    run_id=str(spec["run_identity_seed"]),
                    version=__version__,
                    printer=printer,
                )
            except (hook_host.HookContainmentError, hook_host.HookConfigError, OSError, ValueError) as error:
                containment_error = str(error)
                printer(f"hooks: containment: {json.dumps(containment_error, ensure_ascii=False)}")
                summary["runtime_outcome"] = "hook_containment"
                summary["hook_containment"] = containment_error
                summary["exit_code"] = 2
                summary["succeeded"] = False
        if summary.get("runtime_outcome") == "hook_containment":
            terminal = summary.get("engine_terminal") or {}
            containment_error = str(summary.get("hook_containment") or terminal.get("error_message")
                                    or terminal.get("reason") or terminal.get("reason_code")
                                    or "hook execution containment failed")
            summary["hook_containment"] = containment_error
        try:
            hook_summary = hook_host.hook_record_summary(run_dir)
        except hook_host.HookRecordsError as error:
            containment_error = containment_error or str(error)
            summary["runtime_outcome"] = "hook_containment"
            summary["hook_containment"] = containment_error
            summary["exit_code"] = 2
            summary["succeeded"] = False
            hook_summary = {"runs": 0, "denied": 0, "vetoes": 0, "errors": 1,
                            "last_error": containment_error}
            printer(f"hooks: records error: {json.dumps(str(error), ensure_ascii=False)}")
        if containment_error and hook_summary["last_error"] != containment_error:
            hook_summary["errors"] = int(hook_summary["errors"]) + 1
            hook_summary["last_error"] = containment_error
        summary["hooks"] = hook_summary
        printer(f"hooks: {hook_summary['runs']} runs, {hook_summary['denied']} denied, "
                f"{hook_summary['vetoes']} vetoes, {hook_summary['errors']} errors")
    _write_json_atomic(run_dir / "summary.json", summary)
    removed = finish_run_storage(run_dir, keep_checkpoints=options.keep_checkpoints,
                                 keep_run_env=options.keep_run_env, printer=printer)
    if removed["checkpoints"] is not None:
        printer(f"Removed checkpoints ({removed['checkpoints'] / 1e6:,.0f} MB); run finished. "
                "Use --keep-checkpoints to keep them.")
    if removed["env"] is not None:
        printer(f"Removed run env ({removed['env'] / 1e6:,.0f} MB); run finished. "
                "Use --keep-run-env to keep it.")
    if (summary["runtime_outcome"] in ("cancelled", "engine_error", "missing_result", "budget_stop")
            and not summary.get("engine_error_reason")):  # a seal failure would only repeat on resume
        limit = " --max-turns <more>" if summary["runtime_outcome"] == "budget_stop" else ""
        printer(f"stopped before finishing; continue it with: stateframe resume {run_dir}{limit}")
    return summary
