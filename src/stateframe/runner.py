"""``stateframe run``: prepare a run directory, drive the engine child, report the outcome."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from . import cache_metrics, outer_checks, outcomes, preflight, profiles, test_interpreter, workcopy
from . import resume as resume_module
from .endpoint import CHAT_SUFFIX, Endpoint, EndpointError
from .task_row import build_row

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
CHECKPOINT_ENV = {"STATEFRAME_CHECKPOINT": "save", "STATEFRAME_CHECKPOINT_EVERY": "15",
                  "STATEFRAME_CHECKPOINT_KEEP": "2"}


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


class RunError(RuntimeError):
    pass


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
    keep_checkpoints: bool = False  # --keep-checkpoints / STATEFRAME_KEEP_CHECKPOINTS=1
    _context_tokens: int = field(default=65_536, init=False, repr=False)
    _prompt_budget: int = field(default=33_344, init=False, repr=False)
    _context_env: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    slot: int | None = None
    timeout_s: float | None = None  # None: no time limit (the product default)
    quiet: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)
    _cache_slot: int | None = field(default=None, init=False, repr=False)
    _cache_slot_selection: str | None = field(default=None, init=False, repr=False)
    _cache_slot_reason: str | None = field(default=None, init=False, repr=False)
    _capabilities: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    # Environment for the repository's tests (engine checks and --test-cmd), e.g. test-only
    # settings a project needs to import. Declared to the engine as host task env keys.
    test_env: dict[str, str] = field(default_factory=dict)
    pytest_plugins: str | None = None  # "installed": let pytest load installed plugins
    _deadline: float | None = field(default=None, init=False, repr=False)


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
    return Path.home() / ".stateframe" / "runs" / f"{repo.resolve().name}-{stamp}"


def run_test_cmd(command: str, cwd: Path, timeout_s: float = 900,
                 test_env: dict[str, str] | None = None, *, python: str = sys.executable,
                 pytest_plugins: str | None = None,
                 deadline: float | None = None, test_python: str | None = None) -> dict[str, object]:
    """Check a throwaway copy through the engine policy in a fresh interpreter.

    ``test_python``: the interpreter the command runs, handed to an engine that reads STATEFRAME_TEST_PYTHON so its
    sandbox admits it (None for an older engine, which would refuse the unknown setting)."""
    try:
        return outer_checks.run(command, cwd, python=python, timeout_s=timeout_s,
                                deadline=deadline, test_env=validate_test_env(test_env or {}),
                                pytest_plugins=pytest_plugins, test_python=test_python)
    except (outer_checks.OuterCheckError, OSError) as error:
        raise RunError(str(error)) from error


def _check_python(python: str) -> str:
    """Resolve --python to an absolute executable (relative to the caller's cwd, BEFORE any chdir)
    and check it can run the engine and the checks. Returns the absolute path."""
    try:
        python = preflight.resolve_interpreter(python, os.getcwd())
    except preflight.PreflightError as error:
        raise RunError(str(error)) from None
    probe = subprocess.run([python, "-c", "import sys, pytest; print(sys.version_info[:3])"],
                           capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        raise RunError(f"interpreter {python} cannot import pytest; install pytest there or pass "
                       "--python with one that can (the engine runs the checks with it)")
    major, minor, micro = (int(x) for x in probe.stdout.strip().strip("()").split(","))
    try:
        preflight.check_python_version(version=(major, minor, micro))
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


# Opt-in engine switches a user may set in the shell: exactly these names and values,
# ambient over the profile; unset keeps the profile's value. Never a blanket STATEFRAME_* passthrough.
OPT_IN_SWITCHES = {
    "STATEFRAME_CACHE_LAYOUT": ("on", "off"),
    "STATEFRAME_SLOT_AFFINITY": ("on", "off"),
    "STATEFRAME_TOOL_ORDER": ("stable", "legacy"),
    "STATEFRAME_EVICTFIX": ("on", "legacy"),
    "STATEFRAME_MICROGRAPH": ("on", "off"),  # experimental; the engine defaults to off
}
# Accepted aliases -> the engine's own value (EVICTFIX "off" is sent as "legacy").
OPT_IN_ALIASES = {"STATEFRAME_EVICTFIX": {"off": "legacy"}}
# Legacy unless the user opts in, on every engine that reads the switch (read eviction fix: slower on long runs
# in this release, and no measured edit gain). Applied by the runner so the default never depends on the engine's own.
OPT_IN_RUNNER_DEFAULTS = {"STATEFRAME_EVICTFIX": "legacy"}


def resolve_opt_in_switches(engine_dir: Path = ENGINE_DIR) -> dict[str, str]:
    """The opt-in switches set in the user's environment, validated, for the engine's spec."""
    chosen = {}
    for name in OPT_IN_SWITCHES:
        raw = os.environ.get(name, "").strip()
        if not raw:
            default = OPT_IN_RUNNER_DEFAULTS.get(name)
            if default is not None and name in engine_switches(engine_dir):
                chosen[name] = default
            continue
        value = raw.lower()
        value = OPT_IN_ALIASES.get(name, {}).get(value, value)
        allowed = OPT_IN_SWITCHES[name]
        if value not in allowed:
            raise RunError(f"{name}={raw!r} must be {' or '.join(allowed)}; unset it to keep the default")
        if name not in engine_switches(engine_dir):
            raise RunError(f"{name} is set, but this engine does not support it; unset it to run")
        chosen[name] = value
    return chosen


# Every STATEFRAME_* name the product reads from the user's shell. Any other STATEFRAME_* set there never
# reaches the engine (the child drops inherited ones), so the runner says so instead of ignoring it silently.
SHELL_SETTINGS = frozenset({
    "STATEFRAME_ENDPOINT", "STATEFRAME_MODEL", "STATEFRAME_ALLOW_REMOTE_ENDPOINT", "STATEFRAME_KEEP_CHECKPOINTS",
    "STATEFRAME_CONTEXT_TOKENS", "STATEFRAME_PROMPT_BUDGET", *CHECKPOINT_ENV, *OPT_IN_SWITCHES,
})


def ignored_shell_settings(environ=None) -> list[str]:
    environ = os.environ if environ is None else environ
    return sorted(name for name in environ if name.startswith("STATEFRAME_") and name not in SHELL_SETTINGS)


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


def _select_test_python(options: RunOptions, printer=print) -> str:
    """Choose and check the interpreter for the repository's tests; never StateFrame's own silently."""
    try:
        chosen, reason = test_interpreter.choose(options.repo, test_python=options.test_python,
                                                 python=options.python if options.python_explicit else None)
        python = preflight.resolve_interpreter(chosen, os.getcwd())
    except (test_interpreter.TestInterpreterError, preflight.PreflightError) as error:
        raise RunError(str(error)) from None
    probe = subprocess.run([python, "-c", "import pytest"], capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        raise RunError(f"test interpreter {python} ({reason}) cannot import pytest; install pytest in "
                       "your project's environment or pass --test-python with one that has it")
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


def prepare(options: RunOptions) -> tuple[Path, workcopy.WorkCopy, dict]:
    _validate_timeout(options.timeout_s)
    _remaining(options)
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
    run = (options.run_dir or default_run_dir(options.repo)).resolve()
    if run.exists():
        raise RunError(f"run directory already exists: {run}")
    # Checked BEFORE anything is created: a run never writes into the user's repository.
    if run.is_relative_to(options.repo.resolve()):
        raise RunError(f"--run-dir must be outside the repository: {run}")
    run.mkdir(parents=True)
    work = workcopy.create(options.repo, run / "workspace")
    # Read-only reference copy of the starting state; the engine's finish path reads it, and
    # diffs are taken against it (never against the live original, which may change meanwhile).
    shutil.copytree(work.copy, run / "parent", symlinks=False)
    work.snapshot = run / "parent"
    row = build_row(work, task=options.task, checks=options.checks or None, target=options.target,
                    edit_paths=options.edit_paths or None)
    environment = profiles.load(options.profile)
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
    environment.update(options._context_env)
    opt_in = resolve_opt_in_switches()
    environment.update(opt_in)
    environment.update(options.extra_env)
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
    for name in opt_in:
        environment_sources[name] = ("ambient:" + name if os.environ.get(name, "").strip()
                                     else "runner:default-" + OPT_IN_RUNNER_DEFAULTS[name])
    if selected_slot is not None:
        environment_sources["REACT_SLOT_ID"] = "runner:" + (options._cache_slot_selection or "explicit")
    spec = {
        "schema": "stateframe-run/1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_dir": str(ENGINE_DIR),
        "engine_source": profiles.engine_source_summary(),
        "profile": options.profile,
        "capabilities": options._capabilities,
        "chat_url": options.endpoint.chat_url,
        "model": options.endpoint.model,
        "api_key_env": options.endpoint.api_key_env,
        "max_turns": options.max_turns,
        "parent_snapshot": str(run / "parent"),
        "public_row": row,
        "environment": environment,
        "environment_sources": environment_sources,
        "prompt_cache_slot": {
            "slot": selected_slot,
            "selection": (options._cache_slot_selection or
                          ("explicit" if options.slot is not None else "not_selected")),
            "reason": (options._cache_slot_reason or
                       ("requested by --slot" if options.slot is not None else
                        "slot discovery has not run")),
        },
        "test_cmd": options.test_cmd,
        # The engine's run-scoped ids derive from this (see _child.install_run_identity), so a resume can
        # rebuild the run's requests byte for byte.
        "run_identity_seed": secrets.token_hex(16),
        "repo": str(options.repo.resolve()),
        "in_place": options.in_place,
    }
    (run / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    return run, work, spec


def _child_env(options: RunOptions) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in PASS_ENV}
    env.update({k: v for k, v in os.environ.items() if k in USER_ENV and v.strip()})
    if options.endpoint is not None and options.endpoint.api_key_env:
        env[options.endpoint.api_key_env] = options.endpoint.api_key()
    selected_slot = options._cache_slot if options._cache_slot_selection is not None else options.slot
    if selected_slot is not None:
        env["REACT_SLOT_ID"] = str(selected_slot)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _idle_slot(slots: object) -> tuple[int | None, bool]:
    """Return the lowest uniquely reported idle slot and whether status was usable."""
    if not isinstance(slots, list):
        return None, False
    by_id: dict[int, bool] = {}
    duplicates: set[int] = set()
    for entry in slots:
        if not isinstance(entry, dict):
            continue
        slot_id = entry.get("id")
        if type(slot_id) is not int:
            slot_id = entry.get("id_slot")
        processing = entry.get("is_processing")
        if type(slot_id) is not int or slot_id < 0 or type(processing) is not bool:
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
    if options.slot is not None:
        if type(options.slot) is not int or options.slot < 0:
            raise RunError("--slot must be a nonnegative integer")
        slot = options.slot
        selection = "explicit"
        reason = "requested by --slot"
    else:
        assert options.endpoint is not None
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


# Plain-language CLI text for known engine refusals whose engine wording is consumed by engine code
# (so it cannot be rewritten at build time). The raw detail stays in engine.log.
USER_ERROR_TEXT = {
    "request preparation binding changed before send": (
        "a file in the working copy changed while a request was being prepared, so the request was "
        "refused rather than send stale content. Avoid editing the working copy during a run."),
}


def _render_event(event: dict) -> str | None:
    kind = event.get("kind")
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
        # Per-request timeouts stay (a hung server must not hang the run), never wider than a run limit.
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
        spec_path.write_text(json.dumps(spec, indent=1, sort_keys=True))
    command = [options.python, "-B", str(CHILD), str(run)]
    log_path = run / "engine.log"
    with log_path.open("w") as log:
        _remaining(options)
        child_env = _child_env(options)
        if spec_path.exists():
            # Hash randomization is fixed at interpreter start: the child applying the spec's
            # PYTHONHASHSEED to os.environ itself would be too late to change its own set order.
            seed = json.loads(spec_path.read_text())["environment"].get("PYTHONHASHSEED")
            if seed is not None:
                child_env["PYTHONHASHSEED"] = str(seed)
        process = subprocess.Popen(command, cwd=run, env=child_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   start_new_session=True)
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
            while True:
                if drain_until is None and process.poll() is not None:
                    drain_until = time.monotonic() + OUTPUT_DRAIN_SECONDS
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
                    rendered = _render_event(event)
                    if rendered and (not options.quiet or event.get("kind") == "token_counting_unavailable"):
                        printer(rendered)
            code = process.wait(timeout=60)
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
    if keep_checkpoints_requested(keep) or not root.is_dir() or not resume_module.finished_for_good(run_dir):
        return None
    size = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
        except OSError:
            pass
    shutil.rmtree(root, ignore_errors=True)
    return None if root.exists() else size


def _stop_group(process: subprocess.Popen) -> None:
    """Stop the engine and everything it started (pytest, HTTP helpers): its own session."""
    # Kill even when the leader already exited: descendants can still hold pipes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # macOS can return EPERM for the group (zombie members, or a sandboxed member we may not signal).
        # Cleanup runs in `finally` blocks and must never raise: an error here would replace the real
        # reason the run stopped (a --timeout run once reported "could not start engine"). Kill the leader.
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _artifact_digest(files: dict[str, bytes]) -> str | None:
    """Use the engine's source-map hash, distinct from the host-check tree hash."""
    from .engine.kernel_done_is_submit_authority import artifact_digest

    try:
        sources = {name: data.decode("utf-8") for name, data in files.items()}
    except UnicodeError:
        return None  # A binary partial capture cannot establish a textual DoneSeal.
    return artifact_digest(sources)


def final_candidate(run: Path) -> dict[str, bytes] | None:
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
        sources = record.get("sources") if isinstance(record, dict) else None
        if (not isinstance(record, dict) or record.get("schema") != "stateframe-done-artifact/1"
                or record.get("artifact_digest") != digest or not isinstance(sources, dict)
                or not sources or not all(isinstance(k, str) and isinstance(v, str)
                                          for k, v in sources.items())):
            raise RunError("malformed sealed candidate artifact")
        files = {name: source.encode("utf-8") for name, source in sources.items()}
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
    return {name: base64.b64decode(data, validate=True) for name, data in files.items()}


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
    requests = len(list((run / "requests").glob("*.json"))) if (run / "requests").is_dir() else 0
    slot_info: dict[str, object] = {"slot": None, "selection": "not_recorded",
                                   "reason": "run did not record slot selection"}
    capabilities: dict[str, object] = {}
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
        "prompt_cache_slot": slot_info,
        "wall_seconds": round(time.monotonic() - started, 1),
    }
    summary["cached_share_display"] = _cache_share_line(summary)
    result = outcomes.terminal_receipt(run)
    summary["engine_result"] = result.get("result")
    summary["engine_terminal"] = {key: result[key] for key in (
        "result", "reason_code", "error", "error_type", "error_message", "failure_origin", "behaviour",
        "done_facts", "submission", "labels", "claims")
        if key in result}
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
    _validate_timeout(options.timeout_s)
    started = time.monotonic()
    options._deadline = started + options.timeout_s if options.timeout_s is not None else None
    try:
        with _wall_deadline(options):
            printer(_time_limit_line(options))
            printer(_step_limit_line(options))
            printer("per-check timeout: outer 900s; engine 300s; per model request: the profile's timeout")
            ignored = ignored_shell_settings()
            if ignored:
                printer(f"note: {', '.join(ignored)} {'is' if len(ignored) == 1 else 'are'} set in your shell but "
                        "not passed to the engine (engine settings come from the profile; see docs/guide/running.md)")
            options._context_tokens, options._prompt_budget, how, options._context_env = resolve_context(
                profiles.load(options.profile))
            printer(f"prompt budget: {options._prompt_budget:,} tokens; {options._context_tokens:,}-token server slot required"
                    + ("" if how == "default" else f" ({how})"))
            printer("endpoint policy: local-only by default; a loopback URL can still forward elsewhere")
            with _sigterm_as_interrupt():
                return _run(options, started, printer)
    finally:
        options._deadline = None


def resume_run(run_dir: Path, *, endpoint: Endpoint | None = None, max_turns: int | None = None,
               timeout_s: float | None = None, quiet: bool = False, model: str | None = None,
               keep_checkpoints: bool = False, printer=print) -> dict[str, object]:
    """Continue an interrupted run in place (see ``stateframe.resume``).

    ``model`` continues under another model NAME the endpoint serves (the same weights listed differently,
    e.g. after moving servers); without it the run's own model name is required.
    """
    _validate_timeout(timeout_s)
    started = time.monotonic()
    run_dir = run_dir.resolve()
    state = resume_module.assess(run_dir, max_turns=max_turns, model=model)
    spec: dict = state["spec"]  # type: ignore[assignment]
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
        in_place=bool(spec.get("in_place")), profile=str(spec.get("profile") or profiles.DEFAULT),
        python=str((spec.get("capabilities") or {}).get("python") or sys.executable),
        timeout_s=timeout_s, quiet=quiet, test_env={k: environment[k] for k in keys if k in environment},
        pytest_plugins=environment.get("STATEFRAME_TASK_PYTEST_PLUGINS"), keep_checkpoints=keep_checkpoints)
    options._deadline = started + timeout_s if timeout_s is not None else None
    try:
        with _wall_deadline(options), _sigterm_as_interrupt():
            printer(_time_limit_line(options))
            printer(_step_limit_line(options))
            options.endpoint.check_locality()
            options.python = _check_python(options.python)
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
            lock = resume_module.hold_run_lock(run_dir)  # noqa: F841 - held until this resume returns
            # A checkpoint holds the engine's loop limit, endpoint and model name as they were; a resume
            # that changes any of them rebuilds the run by replay instead.
            same_limits = (int(state["max_turns"]) == int(spec["max_turns"])
                           and endpoint.chat_url == spec["chat_url"] and not isinstance(swap, dict))
            if int(state["max_turns"]) != int(spec["max_turns"]):
                printer(step_limit_restart_warning(int(spec["max_turns"]), int(state["max_turns"])))
            previous = resume_module.archive_attempt(state, ENGINE_DIR, use_checkpoint=same_limits)
            spec = _resume_spec(run_dir, endpoint)
            # The cache slot stays the one the run started with: it is part of the request bytes.
            slot = spec.get("prompt_cache_slot") or {}
            if isinstance(slot.get("slot"), int):
                options._cache_slot, options._cache_slot_selection = slot["slot"], "resumed"
            printer(f"resuming run {run_dir} (attempt {spec['resume']['attempt']}; previous attempt "
                    f"stopped with {state['previous_status'] or 'no engine result'}, files kept in {previous})")
            if isinstance(swap, dict):
                printer(f"MODEL NAME SWAP: this run was recorded with model {', '.join(map(repr, swap['recorded']))}"
                        f" and continues with {swap['served']!r}, the name the endpoint serves. This is only "
                        "correct if it is the SAME model. Replay compares each request with its model field "
                        "set back to the recorded name; nothing else is changed (recorded in spec.json, "
                        "summary.json and any resume-divergence.json)")
            checkpoint = spec["resume"].get("checkpoint")
            if checkpoint:
                printer(f"checkpoint: engine state restored as of send {checkpoint['sends']} (step "
                        f"{checkpoint['step']}); {checkpoint['tail']} later recorded exchange(s) are replayed "
                        "and checked byte for byte, then the run continues live")
            else:
                printer(f"replay: {state['tape_count']} recorded model exchanges are answered from the run's "
                        "records, then the run continues live")
            work = resume_module.fresh_workcopy(run_dir, options.repo)
            summary = _execute(options, started, printer, run_dir, work, spec)
            refused = (summary.get("checkpoint_refused") or _checkpoint_refusal(run_dir, summary.get("child_exit_code"))
                       if checkpoint else None)
            if refused is None:
                return summary
            # Fail open: a checkpoint that cannot be restored faithfully is never continued from. Keep the
            # refused attempt's files and replay the interrupted attempt from the start instead.
            printer(f"checkpoint restore refused ({refused}); falling back to replaying the run")
            state["attempt"] = int(state["attempt"]) + 1
            resume_module.archive_attempt(state, ENGINE_DIR, use_checkpoint=False, tape_from=previous)
            spec = _resume_spec(run_dir, endpoint)
            spec["resume"]["checkpoint_refused"] = refused
            (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
            work = resume_module.fresh_workcopy(run_dir, options.repo)
            return _execute(options, started, printer, run_dir, work, spec)
    finally:
        options._deadline = None


def _resume_spec(run_dir: Path, endpoint: Endpoint) -> dict:
    spec = json.loads((run_dir / "spec.json").read_text())
    if endpoint.chat_url != spec["chat_url"]:
        spec["chat_url"] = endpoint.chat_url
        spec["environment"]["DS_URL"] = endpoint.chat_url
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
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


def _run(options: RunOptions, started: float, printer=print) -> dict[str, object]:
    if options.endpoint is None:
        raise RunError("an endpoint is required")
    options.endpoint.check_locality()
    options.python = _check_python(options.python)
    options._capabilities = _probe_capabilities(options)
    printer(str(options._capabilities["display"]))
    options._test_python = _select_test_python(options, printer)
    probe = replace(options.endpoint, timeout_s=min(options.endpoint.timeout_s, _remaining(options))).probe()
    if probe.get("model_found") is False:
        raise RunError(f"endpoint does not serve model {options.endpoint.model!r}; it lists: "
                       + ", ".join(probe.get("served_models") or []))
    check_slot_context(options, printer)
    _select_prompt_cache_slot(options, printer)
    run_dir, work, spec = prepare(options)
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
    printer(f"working copy: {work.copy} ({len(work.baseline)} files; original untouched"
            f"{' until the end (--in-place)' if options.in_place else ''})")
    printer(f"starting file hint: {spec['public_row']['entry']['path']}; "
            f"checks: {', '.join(spec['public_row']['public_check_paths'])}")
    test_cmd = options.test_cmd or ("python -m pytest -q " + " ".join(
        shlex.quote(p) for p in spec["public_row"]["public_check_paths"]))
    before = run_test_cmd(test_cmd.replace("python ", shlex.quote(_tests_python(options)) + " ", 1)
                          if test_cmd.startswith("python ") else test_cmd, work.copy,
                          test_env=options.test_env, python=_tests_python(options),
                          pytest_plugins=options.pytest_plugins, deadline=options._deadline,
                          timeout_s=min(900, _remaining(options)), test_python=_outer_test_python(options))
    printer(f"before: {test_cmd} -> {'PASS' if before['passed'] else 'FAIL'} (exit {before['exit_code']})")
    cancelled = False
    _remaining(options)
    try:
        child_exit = drive(run_dir, options, printer)
    except KeyboardInterrupt:
        child_exit, cancelled = 130, True
    except OSError as error:
        raise RunError(f"could not start engine: {error}") from error
    if (spec.get("resume") or {}).get("checkpoint") and not cancelled:
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
    artifact_digest = None
    try:
        candidate = final_candidate(run_dir)
        if candidate is not None:
            artifact_digest = _artifact_digest(candidate)
            written = workcopy.apply_files(work, candidate, declared=declared)
    except RunDeadlineExceeded:
        raise
    except (RunError, workcopy.WorkCopyError, OSError, ValueError) as error:
        candidate_error = str(error)
    digest = candidate_digest(work.copy) if candidate is not None and not candidate_error else None
    after = {"command": test_cmd, "exit_code": None, "passed": None, "tail": "checks not run"}
    if digest and not cancelled:
        try:
            after = run_test_cmd(test_cmd.replace("python ", shlex.quote(_tests_python(options)) + " ", 1)
                                 if test_cmd.startswith("python ") else test_cmd, work.copy,
                                 test_env=options.test_env, python=_tests_python(options),
                                 pytest_plugins=options.pytest_plugins, deadline=options._deadline,
                                 timeout_s=min(900, _remaining(options)),
                                 test_python=_outer_test_python(options))
            if candidate_digest(work.copy) == digest:
                after["candidate_sha256"] = digest
            else:
                candidate_error = "candidate changed while host checks ran; checks are stale"
        except KeyboardInterrupt:
            cancelled = True
    diff = workcopy.unified_diff(work)
    (run_dir / "final.diff").write_text(diff)
    summary = summarize(run_dir, started)
    printer(str(summary["cached_share_display"]))
    summary.update({
        "run_dir": str(run_dir),
        "child_exit_code": child_exit,
        "candidate_captured": candidate is not None,
        "candidate_sha256": digest,
        "candidate_artifact_digest": artifact_digest,
        "candidate_error": candidate_error,
        "candidate_files_written": written,
        "files_changed": sorted(workcopy.changed_files(work)),
        "tests_before": {k: before[k] for k in ("command", "exit_code", "passed")},
        "tests_after": {k: after[k] for k in ("command", "exit_code", "passed", "candidate_sha256")
                        if k in after},
        "endpoint": options.endpoint.public(),
        "profile": options.profile,
        "engine_source_hash": spec["engine_source"].get("source_hash"),
        "engine_shipped_hash": spec["engine_source"].get("shipped_hash"),
    })
    if isinstance(spec.get("resume"), dict):
        summary["resume"] = _resume_record(run_dir, spec["resume"], summary)
    (run_dir / "tests-after.txt").write_text(after["tail"])
    summary.update(outcomes.project(summary, cancelled=cancelled, timed_out=child_exit == 124))
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
            summary["applied_in_place"] = []
            summary["in_place_skipped"] = (
                f"candidate not promoted: runtime={summary['runtime_outcome']}, "
                f"checks={summary['check_outcome']}, artifact={summary['artifact']}")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    removed = clean_finished_checkpoints(run_dir, keep=options.keep_checkpoints)
    if removed is not None:
        printer(f"Removed checkpoints ({removed / 1e6:,.0f} MB); run finished. Use --keep-checkpoints to keep them.")
    if (summary["runtime_outcome"] in ("cancelled", "engine_error", "missing_result", "budget_stop")
            and not summary.get("engine_error_reason")):  # a seal failure would only repeat on resume
        limit = " --max-turns <more>" if summary["runtime_outcome"] == "budget_stop" else ""
        printer(f"stopped before finishing; continue it with: stateframe resume {run_dir}{limit}")
    return summary
