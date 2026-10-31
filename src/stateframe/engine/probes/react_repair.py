"""Iterative repair loop: reason, act, observe, and revise.

Tasks run in isolated repository copies. The worker can search and read source,
edit definitions, execute tests, and use observations on subsequent steps.
Task budgets bound execution; completion requires evidence.
"""
from __future__ import annotations

import ast
import copy
import concurrent.futures as cf
from contextlib import nullcontext
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

from kernel_shell_tool import SHELL_SCHEMA

if __package__:
    from .decoy_tools import (
        build_tool_surface,
        decoy_response,
        decoy_names,
        decoy_kind as selected_decoy_kind,
        is_decoy_name,
        is_near_duplicate_name,
    )
    from .regime_controls import (
        approx_tokens,
        nonnegative_env_int,
        pad_system_prompt,
        render_neutral_preamble,
    )
else:  # Direct historical script entrypoint.
    from decoy_tools import (
        build_tool_surface,
        decoy_response,
        decoy_names,
        decoy_kind as selected_decoy_kind,
        is_decoy_name,
        is_near_duplicate_name,
    )
    from regime_controls import (
        approx_tokens,
        nonnegative_env_int,
        pad_system_prompt,
        render_neutral_preamble,
    )

POC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POC))
sys.path.insert(0, str(POC / "probes"))
import kernel_py39_compat  # noqa: E402,F401
from build_test_path import resolve_build_test_path  # noqa: E402
from kernel_tool_paths import (  # noqa: E402
    AmbiguousToolPath, receipt_target_path, resolve_tool_path, unique_tool_match,
)
from kernel_verdict import BLOCKED  # noqa: E402
from kernel_tool_reply import tool_reply_for_call as _tool_reply_for_call  # noqa: E402
from kernel_silent_off_organs import (  # noqa: E402
    expectation_subject as _expectation_subject,
    observe_expectation_suite as _observe_expectation_suite,
    wire_composed_cognition as _wire_composed_cognition,
    wire_corpus_fixes as _wire_corpus_fixes,
    wire_expectation_ledger as _wire_expectation_ledger,
    PendingWiringJournal as _PendingWiringJournal,
)
from sampler_config import apply_to_request as _apply_sampler_fields  # noqa: E402
from kernel_done_acceptance import done_accepted  # noqa: E402
import kernel_workspace_truth as _workspace_truth  # noqa: E402

CORPUS = POC / "results.SNAPSHOT-20260920" / (os.environ.get("BENCH_CORPUS") or "real_tasks_v3.jsonl")
API = os.environ.get("DS_URL", "")
# DeepSeek V4 Flash is the model we actually care about — it is the current agentic/coding
# model and the one whose capability the whole "harness, not model" thesis is measured against.
# This defaulted to V3.1 (a year-old general model) purely because I typed it first, and every
# ReAct run tonight silently used it. Flash is the default now so a stale model can never be
# picked up by accident again.
MODEL = os.environ.get("DS_MODEL", "")
KEY = os.environ.get("DEEPINFRA_API_KEY", "")
from kernel_step_limits import (  # noqa: E402
    DEFAULT_STEPS, limit_text, parse_step_limit, reached, remaining, step_guidance, step_indices,
)

MAX_STEPS = parse_step_limit(os.environ.get("REACT_STEPS", str(DEFAULT_STEPS)))
STEPS_UNBOUNDED = True  # product capability marker: REACT_STEPS=0 / MAX_STEPS=0 means unbounded
# CPU budget for one public check, aligned with its 300 s wall timeout (recorded bound; the
# policy default of 30 s RLIMIT_CPU SIGKILLed a 37 s check in batch 843).  Bounded by the
# execution policy's MAX_WALL_SECONDS.
PUBLIC_CHECK_CPU_SECONDS = float(os.environ.get("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", "300"))
# Module-level so a persistence wrapper can diversify retries: attempt 1 stays at 0 for
# comparability, later attempts resample (an identical deterministic trajectory teaches
# nothing). See probes/persist_repair.py.
TEMPERATURE = float(os.environ.get("REACT_TEMP", "0"))
EDIT_HARDENING_ENV = "STATEFRAME_TOKENLAB_EDIT_HARDENING"
EDIT_NORMALIZATION_ENV = "STATEFRAME_EDIT_NORMALIZATION"
try:
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].lstrip("-").isdigit() else 6
except (TypeError, ValueError):
    # This module is imported by offline contract tests as well as run as a
    # script.  A unittest module name is not a task-count argument.
    N = 6
OUT = POC / "results" / "react_repair.jsonl"


class _GreenLockRefused(Exception):
    """Short-circuit a file_create before its existing write path runs."""


def _reported_progress_counts(scope: Mapping[str, object]) -> tuple[int, int]:
    """Prefer the live host fact reader while preserving the primary error path."""
    host_truth_enabled = scope.get("_host_truth_progress_enabled") is True
    reader = scope.get("_progress_facts")
    if callable(reader):
        try:
            facts = reader()
        except Exception:                                             # noqa: BLE001
            facts = None
        if facts is not None:
            edits = getattr(facts, "edits", None)
            checks = getattr(facts, "checks", None)
            if type(edits) is int and type(checks) is int:
                return edits, checks
    if host_truth_enabled:
        # Preserve the original exception while refusing to label legacy guesses as
        # host evidence when the fact reducer itself could not produce a snapshot.
        return 0, 0
    edits = scope.get("edits", 0)
    checks = scope.get("tests", 0)
    return (
        edits if type(edits) is int and edits >= 0 else 0,
        checks if type(checks) is int and checks >= 0 else 0,
    )


def _finish_boundary_recovery_terminal(
        measurement: object | None, outcome: dict[str, object], messages: object, *,
        finish_transcript: Callable[[object], object],
        attach: Callable[[dict[str, object], object | None], dict[str, object]],
) -> dict[str, object]:
    """Finish an already candidate-captured boundary stop without worker_exception."""
    if measurement is not None:
        measurement.system_outcome = "run_incomplete"
        measurement.journal.emit(
            "boundary_recovery_terminal", reason_code="boundary_unrecoverable",
            step=outcome.get("steps"),
            candidate_export_attempted=True, failure_origin="harness", dispatch=False,
        )
        finish_transcript(messages)
        measurement.finish(outcome, messages=messages)
    return attach(outcome, measurement)


def _edit_hardening_enabled() -> bool:
    return os.environ.get(EDIT_HARDENING_ENV, "off").strip().lower() == "on"


def _nonpy_edit_policy_enabled() -> bool:
    from kernel_tokenlab_nonpy_edits import enabled
    return enabled()


def _nonpy_edit_applies(path: pathlib.Path) -> bool:
    from kernel_tokenlab_nonpy_edits import applies
    return applies(path)


def _pytest_exit4_target_meaning(root: pathlib.Path, target: object, *,
                                journal: object = None) -> str | None:
    raw_path = pathlib.Path(str(target).split("::", 1)[0])
    resolved = raw_path if raw_path.is_absolute() else root / raw_path
    if resolved.exists():
        return None
    if _workspace_truth.enabled():
        fact = _workspace_truth.exists(root, str(raw_path), journal=journal)
        return (_workspace_truth.render_fact(fact)
                + " Re-check the target with `grep` or `read`.")
    return "the path or node id does not exist — re-check it with `grep` or `read`."


def _validate_v21_candidate(path: pathlib.Path, source: str,
                            operation: str) -> tuple[str | None, str | None]:
    """Return ``(rejection, warning)`` for an enabled candidate."""
    if not _nonpy_edit_policy_enabled():
        return None, None
    from kernel_tokenlab_nonpy_edits import (
        NonPythonValidationError,
        emit,
        file_type,
        validate,
        validate_python,
    )
    try:
        if file_type(path) == "python":
            validate_python(source, path)
            return None, None
        warning = validate(source, path)
    except SyntaxError as error:
        reason = f"syntax error at line {error.lineno}: {error.msg}"
        emit(path, operation=operation, outcome="REJECTED", reason=reason)
        return f"{operation} REJECTED: {reason}. Nothing was changed.", None
    except NonPythonValidationError as error:
        reason = str(error)
        emit(path, operation=operation, outcome="REJECTED", reason=reason)
        return f"{operation} REJECTED: {reason}. Nothing was changed.", None
    return None, warning


def _edit_normalization_enabled() -> bool:
    return os.environ.get(EDIT_NORMALIZATION_ENV, "off").strip().lower() == "on"


def _normalize_candidate_for_publication(
    path: pathlib.Path,
    candidate: bytes,
    *,
    mode: str,
    root: pathlib.Path | None,
    target: str | None = None,
    target_ref: object = None,
    expected_base_digest: str | None = None,
    composed_candidate: bytes | None = None,
) -> tuple[bytes | None, str | None]:
    """Run mechanical normalization immediately before the existing guards/effect."""
    if not _edit_normalization_enabled():
        # A definition payload is only the replacement fragment. Its caller
        # already composed the full file; disabling normalization must retain it.
        return composed_candidate if composed_candidate is not None else candidate, None
    from kernel_edit_normalization import AmbiguousScope, Rejected, normalize_edit

    try:
        base = path.read_bytes()
        relative = path.resolve().relative_to((root or path.parent).resolve()).as_posix()
    except (OSError, ValueError) as error:
        return None, f"edit REJECTED: normalization could not bind {path.name}: {type(error).__name__}"
    def ref_value(name: str, default: object = None) -> object:
        if isinstance(target_ref, Mapping):
            return target_ref.get(name, default)
        return getattr(target_ref, name, default) if target_ref is not None else default

    target_payload: dict[str, object] = {}
    if target:
        pieces = [part for part in str(target).split(".") if part]
        target_payload["name"] = pieces[-1] if pieces else str(target)
        if len(pieces) > 1:
            target_payload["owner"] = ".".join(pieces[:-1])
    if target_ref is not None:
        for key in (
            "site_id", "path", "start", "end", "node_kind", "kind", "owner", "name",
            "signature", "authority_ref", "source_digest", "file_revision",
            "permitted_operations", "boundary",
        ):
            value = ref_value(key)
            if value is not None:
                target_payload[key] = value
    reference_authority = str(ref_value("authority_ref", "") or "")
    authority_ref = reference_authority or "executor-edit-authority"
    reference_revision = str(ref_value("file_revision", "") or "")
    base_revision = reference_revision or ("source:" + hashlib.sha256(base).hexdigest())
    reference_kind = str(ref_value("node_kind", ref_value("kind", "")) or "").casefold()
    supports_definition = mode == "complete-definition" and reference_kind not in {"module", "file", "whole-file"}
    payload: dict[str, object] = {
        "mode": mode,
        "path": relative,
        "candidate": candidate,
        "expected_base_digest": expected_base_digest or str(
            ref_value("source_digest", "") or "sha256:" + hashlib.sha256(base).hexdigest()
        ),
        "base_revision": base_revision,
        "authority_ref": authority_ref,
        "supports_definition": supports_definition,
    }
    if target_payload:
        payload["target"] = target_payload
    try:
        from kernel_revision_bound_edit import EditAuthority
        authority = EditAuthority(
            root=(root or path.parent),
            sources={relative: base},
            base_revision=base_revision,
            authority_ref=authority_ref,
            allowed_paths=(relative,),
        )
    except (ImportError, OSError, TypeError, ValueError) as error:
        return None, f"edit REJECTED: normalization authority unavailable ({type(error).__name__})"
    result = normalize_edit(
        authority, {"path": relative, "bytes": base, "base_revision": base_revision}, payload,
        target_site=target_ref,
    )
    if hasattr(result, "candidate_bytes"):
        return result.candidate_bytes, None
    if isinstance(result, AmbiguousScope):
        reason = result.reason
    elif isinstance(result, Rejected):
        reason = result.reason
    else:
        reason = "normalization returned an unknown result"
    return None, f"edit REJECTED: faithful normalization: {reason}"


def _emit_edit_hardening(**fields: object) -> None:
    if not _edit_hardening_enabled():
        return
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        runtime = ACTIVE_RUNTIME.get()
    except (ImportError, LookupError):
        runtime = None
    journal = getattr(runtime, "journal", None)
    emit = getattr(journal, "emit", None)
    if callable(emit):
        emit("tokenlab_v17_edit_hardening", producer="probes.react_repair",
             event_schema="stateframe-v17-edit-hardening/1", dispatch=False, **fields)


def _release_noread_after_applied_mutation(
    runtime: object,
    *,
    request_id: object,
    tool_name: object,
    observation: str,
    batch_requested: bool = False,
    batch_status: object = None,
    batch_changed: bool = False,
    single_str_replace_applied: bool = False,
) -> bool:
    """Release noread after any applied source mutation, including test-file edits."""
    name = str(tool_name or "").strip()
    first_line = observation.split("\n", 1)[0].lower()
    applied = (
        (name == "edit" and "applied" in first_line)
        or (name == "str_replace" and (
            single_str_replace_applied
            or (batch_requested and batch_status == "APPLIED" and batch_changed)
        ))
    )
    if not applied:
        return False
    from kernel_rejection_template_restrict import release_noread_after_mutation

    return release_noread_after_mutation(
        runtime, request_id=request_id, tool_name=name)


def _task_public_brief(row: object) -> str:
    """Return the user-facing brief used for explicit destructive-edit exceptions."""
    if not isinstance(row, Mapping):
        return ""
    for key in ("_bridge_ordinary_brief", "msg", "brief", "symptom"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


_LOOP_BREAKING_COMMANDS = frozenset({
    "edit", "str_replace", "file_create", "insert_before", "replace_module", "write",
    "test", "run_tests", "authored_check", "done", "work", "abstain",
})


def _repeat_query_surgery_eligible(
    line: str, *, interventions: bool, worst_repeat: int,
    step: int, last_surgery_step: int,
) -> bool:
    """Excise looping QUERIES only.

    Reads stay dispatchable so exact-range reuse can settle them, and the loop-breaking
    actions themselves (edit/test/done...) are never intercepted; benchmark r2j
    had its only edit suppressed as REPEAT_SUPPRESSED because a READ count reached 18.
    """
    command = line.split(maxsplit=1)[0] if line.strip() else ""
    return (interventions and worst_repeat >= 10 and command != "read"
            and command not in _LOOP_BREAKING_COMMANDS
            and step - last_surgery_step > 10)


def _same_query_suppression_eligible(line: str, last_line: str | None) -> bool:
    """Suppress only repeated grep queries; reads may request distinct source ranges."""
    parts = line.split(maxsplit=1)
    return line == last_line and bool(parts) and parts[0] == "grep"


def _append_structure_fact_observation(measurement, bolt, observation: str,
                                       kind: str) -> str:
    """Expose the already-attached typed facts in a history tool result."""
    if getattr(measurement, "structure_facts_mode", "off") != "on":
        return observation
    receipts = getattr(bolt, "_receipts", ())
    receipt = next((row for row in reversed(receipts or ())
                    if isinstance(row, Mapping) and row.get("kind") == kind), None)
    facts = receipt.get("structure_facts") if isinstance(receipt, Mapping) else None
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes)) or not facts:
        return observation
    from kernel_empirical_fixes import structure_facts_observation
    rendered = structure_facts_observation(facts)
    return observation + ("\n" if observation else "") + rendered if rendered else observation

REPOS = {
    name: pathlib.Path(path)
    for name, key in (("aha", "STATEFRAME_AHA_REPO"),
                      ("mc", "STATEFRAME_MC_REPO"),
                      ("build", "STATEFRAME_BUILD_REPO"))
    if (path := os.environ.get(key))
}

SYSTEM = """You are repairing a bug in a large production Python codebase.

ORIENTATION — where you are and how paths work. The first version of this prompt omitted all
of it, and the model burned 29 and 36 steps making ZERO edits: it could not resolve a path, so
it grepped forever and never reached a state where editing was possible. An agent with no map
is not an agent.

  * You are working inside a private throwaway COPY of the repository. Nothing you do can
    affect the real codebase, so edit freely and test often.
  * Your base directory is the repository root. EVERY path you give is relative to it, with no
    leading slash. `{REL}` is a valid path exactly as written.
  * The file you almost certainly need is `{REL}`, and the function is `{FN}`.
  * `grep` searches the whole tree and returns paths already relative to your base directory,
    so you can paste one straight into `read` or `edit`.

You work in STEPS. Each step you output EXACTLY ONE command and nothing else:

  grep <pattern>          search the codebase
  read <path> [line]      read 120 lines of a file
  edit <path>             then a ```python block with the COMPLETE corrected function or class
  test [path]             run the repository's tests (defaults to tests touching this function)
  done                    stop; your last edit is your answer

HOW TO WORK. You have {STEPS} steps. Investigating is cheap but it is not progress: you learn
nothing about whether you are right until you EDIT and TEST. So form a hypothesis early, make
the edit, run the tests, and read what they say. A failed test is information, not defeat —
it is the only real feedback available to you, and you cannot get it without acting first.

  * Do not spend more than about 8 steps investigating before your first edit. A wrong edit
    that fails a test teaches you more than four more greps.
  * After every edit, run `test`.
  * You may edit and re-test as many times as you like. Iterate.
  * `done` with no edit made is a wasted run: you will have submitted the original bug.

### Operating rules
(Ported from Mission Control's production coding agent, `agent-runner/app/coding_executor.py`
:DEFAULT_CODING_SYSTEM_PROMPT — these are battle-tested on a real fleet and this probe had
none of them.)

  * Read the file you are about to edit. Never edit blind.
  * Do NOT re-read a file just to confirm an edit landed. The edit command tells you whether
    it applied, and a replacement that does not parse is REJECTED rather than written, so a
    confirming read teaches you nothing and costs a step.
  * Investigate failures before declaring done. A red test with a traceback names the line.
  * If you find yourself repeating the same action, STOP and change approach — say what is
    blocking you and try a different command. Repetition is the clearest sign you are stuck.
  * Never make a failing test pass by weakening it. A test encodes a requirement; if you
    believe the test itself is wrong, say so explicitly rather than editing around it.
  * Write code that reads like the surrounding code — match its naming and idiom. Change only
    what the bug requires.
  * Do not end on an intention. If your final step would be "next I would try X", do X now."""


_REG_CACHE: dict[str, object] = {}


def _registry_for(root: pathlib.Path):
    """Build the symbol index once per sandbox and reuse it.

    Indexing a 2.5M-LOC tree takes seconds; doing it per failed grep would make the bolt-on
    cost more than the six wasted steps it exists to prevent.
    """
    key = str(root)
    if key not in _REG_CACHE:
        from symbol_registry import SymbolRegistry
        _REG_CACHE[key] = SymbolRegistry().build(root)
    return _REG_CACHE[key]


def fn_src(src, name):
    if "." in name:
        from kernel_edit_affordance import enabled, qualified_source
        if enabled():
            return qualified_source(src, name)
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        # ClassDef included for build tasks whose unit is a class (E382: TokenBucket
        # SKIPped because fn_src only knew functions).
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and n.name == name:
            return ast.get_source_segment(src, n)
    return None


def _build_test_path(row: Mapping[str, object], fn: str | None = None) -> str | None:
    if row.get("repo") != "build" or isinstance(row.get("files"), Mapping):
        return None
    return resolve_build_test_path(row, fn)


def _materialize_build_test(root: pathlib.Path, row: Mapping[str, object],
                            fn: str) -> str | None:
    relative = _build_test_path(row, fn)
    if relative is None:
        return None
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(row["test_src"])
    legacy = root / "tests" / f"test_{fn.strip('_')}_build.py"
    if (legacy != destination and legacy.is_file()
            and legacy.resolve() != destination.resolve()):
        legacy.unlink()
    return relative


def _public_check_ids(row: Mapping[str, object], fn: str) -> list[str]:
    build_path = _build_test_path(row, fn)
    if build_path:
        return [build_path]
    from converge import fail_to_pass_ids
    return fail_to_pass_ids(row) or []


def norm(src):
    try:
        t = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(t):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                node.body = b[1:] or [ast.Pass()]
    return ast.dump(t)


@dataclass(frozen=True)
class FileSet:
    """Normalized complete-source map for an H10 multi-file repair row."""

    present: bool
    parent: Mapping[str, str]
    fixed: Mapping[str, str]
    changed: Mapping[str, tuple[str, ...]]
    entry_path: str | None = None
    entry_fn: str | None = None
    public_tests: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row) -> "FileSet":
        raw_files = row.get("files")
        if raw_files is None:
            return cls(False, {}, {}, {})
        if isinstance(raw_files, Mapping):
            file_items = raw_files.items()
        elif isinstance(raw_files, Sequence) and not isinstance(raw_files, (str, bytes)):
            file_items = []
            for item in raw_files:
                if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                    raise ValueError("multi-file row file list items need path and source")
                file_items.append((item["path"], item))
        else:
            raise ValueError("multi-file row files must be a mapping or public file list")
        from build_tasks.multifile_row import normalize_repo_path

        parent: dict[str, str] = {}
        fixed: dict[str, str] = {}
        public_tests: dict[str, str] = {}
        raw_checks = row.get("public_check_paths")
        declared_checks = (set(raw_checks or ())
                           if isinstance(raw_checks, Sequence)
                           and not isinstance(raw_checks, (str, bytes)) else set())
        for raw_path, payload in file_items:
            path = normalize_repo_path(raw_path, field="row.files path")
            if path in parent:
                raise ValueError(f"duplicate normalized multi-file path: {path}")
            if not isinstance(payload, Mapping):
                raise ValueError(f"row.files[{path}] must be a mapping")
            parent_src = payload.get("parent_src", payload.get("source", payload.get("content")))
            fixed_src = payload.get("fixed_src", parent_src)
            if not isinstance(parent_src, str) or not isinstance(fixed_src, str):
                raise ValueError(f"row.files[{path}] needs parent_src and optional fixed_src")
            if path.startswith("tests/") or path in declared_checks:
                public_tests[path] = parent_src
                continue
            parent[path] = parent_src
            fixed[path] = fixed_src

        changed: dict[str, tuple[str, ...]] = {}
        raw_changed = row.get("changed_functions", {})
        if isinstance(raw_changed, Mapping):
            for raw_path, names in raw_changed.items():
                path = normalize_repo_path(raw_path, field="row.changed_functions path")
                if isinstance(names, str):
                    names = (names,)
                if isinstance(names, Sequence):
                    changed[path] = tuple(str(name) for name in names if str(name))

        entry_path = entry_fn = None
        entry = row.get("entry")
        if not isinstance(entry, Mapping) and isinstance(row.get("working_path"), str):
            entry = {"path": row["working_path"]}
        if not isinstance(entry, Mapping) and isinstance(row.get("path"), str):
            entry = {"path": row["path"]}
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
            entry_path = normalize_repo_path(entry["path"], field="row.entry.path")
            entry_fn = (str(entry["fn"]) if isinstance(entry.get("fn"), str)
                        and entry.get("fn") else None)
            if entry_path not in parent:
                raise ValueError(f"row.entry.path is not declared in files: {entry_path}")

        # Separate public-test declarations are authoritative when tests are not
        # included in the product ``files`` mapping.
        try:
            from kernel_public_evidence import public_test_sources
            public_tests.update(public_test_sources(row))
        except ImportError:
            # Legacy rows may only provide test_src; the single-file path keeps
            # its historical materialization route below.
            pass
        return cls(True, parent, fixed, changed, entry_path, entry_fn, public_tests)

    @property
    def is_multi_file(self) -> bool:
        return self.present

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(sorted(self.parent))

    @property
    def primary_path(self) -> str | None:
        if self.entry_path in self.parent:
            return self.entry_path
        return None

    @property
    def primary_fn(self) -> str:
        if self.entry_fn:
            return self.entry_fn.split(".")[-1]
        # An honest ticket can intentionally omit its target.  An AST symbol
        # from the first sorted file is not localization evidence and can name
        # a symbol that only exists in the hidden fix.  Keep the planner in its
        # targetless state until the model supplies a path or symbol.
        return ""


def _source_snapshot(root: pathlib.Path, paths: Sequence[str]) -> dict[str, str | None]:
    """Read declared files without turning a missing target into a harness crash."""
    snapshot: dict[str, str | None] = {}
    for path in paths:
        target = root / path
        snapshot[path] = target.read_text(errors="ignore") if target.is_file() else None
    return snapshot


def _authored_check_snapshots(
        root: pathlib.Path, runtime: object,
) -> tuple[dict[str, object], dict[str, object], str | None]:
    """Materialize only hash-verified parent/current files for an authored check."""
    try:
        current = runtime.binding() if callable(getattr(runtime, "binding", None)) else {}
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        current = {}
    parent = getattr(runtime, "_row_role_parent_binding", None)
    if not isinstance(current, Mapping) or not isinstance(parent, Mapping):
        return {"revision": "", "files": {}}, {"revision": "", "files": {}}, "revision_binding_unavailable"
    current_files: dict[str, object] = {}
    current_hashes = {
        str(path): str(value) for key in ("product_files", "harness_files")
        for path, value in (current.get(key, {}) if isinstance(current.get(key), Mapping) else {}).items()
    }
    parent_hashes = {
        str(path): str(value) for key in ("product_files", "harness_files")
        for path, value in (parent.get(key, {}) if isinstance(parent.get(key), Mapping) else {}).items()
    }
    total_bytes = 0
    for relative, expected in current_hashes.items():
        try:
            target = (root / relative).resolve()
            target.relative_to(root.resolve())
            source = target.read_bytes()
        except (OSError, ValueError):
            continue
        if len(source) > 8 * 1024 * 1024 or total_bytes + len(source) > 64 * 1024 * 1024:
            continue
        digest = "sha256:" + hashlib.sha256(source).hexdigest()
        if expected.startswith("sha256:") and digest != expected:
            continue
        current_files[relative] = source
        total_bytes += len(source)
    parent_sources: dict[str, str] = {}
    task_row = getattr(getattr(runtime, "owner", None), "task_row", {})
    declared = task_row.get("files", {}) if isinstance(task_row, Mapping) else {}
    if isinstance(declared, Mapping):
        for relative, payload in declared.items():
            if isinstance(payload, Mapping) and isinstance(payload.get("parent_src"), str):
                parent_sources[str(relative)] = payload["parent_src"]
    parent_files: dict[str, bytes] = {}
    for relative, expected in parent_hashes.items():
        source = parent_sources.get(relative)
        if source is None and relative in current_files and str(expected) == \
                "sha256:" + hashlib.sha256(current_files[relative]).hexdigest():
            source = current_files[relative]
        if source is None:
            continue
        raw = source.encode("utf-8") if isinstance(source, str) else bytes(source)
        if str(expected).startswith("sha256:") and \
                "sha256:" + hashlib.sha256(raw).hexdigest() != str(expected):
            continue
        parent_files[relative] = raw
    parent_revision = str(parent.get("product_revision") or "").strip()
    candidate_revision = str(current.get("product_revision") or "").strip()
    parent_missing = sorted(set(parent_hashes) - set(parent_files))
    current_missing = sorted(set(current_hashes) - set(current_files))
    def manifest(files: Mapping[str, object]) -> list[dict[str, object]]:
        return [{
            "path": relative,
            "sha256": "sha256:" + hashlib.sha256(
                source if isinstance(source, bytes) else bytes(source)
            ).hexdigest(),
            "bytes": len(source),
        } for relative, source in sorted(files.items())]

    if parent_missing or current_missing:
        missing = ",".join((*parent_missing, *current_missing))
        return {"revision": parent_revision, "files": parent_files,
                "manifest": manifest(parent_files), "manifest_complete": False}, \
            {"revision": candidate_revision, "files": current_files,
             "manifest": manifest(current_files), "manifest_complete": False}, \
            "snapshot_incomplete:" + missing
    if not parent_revision or not candidate_revision:
        return {"revision": parent_revision, "files": parent_files,
                "manifest": manifest(parent_files), "manifest_complete": False}, \
            {"revision": candidate_revision, "files": current_files,
             "manifest": manifest(current_files), "manifest_complete": False}, \
            "revision_binding_unavailable"
    parent_manifest = manifest(parent_files)
    current_manifest = manifest(current_files)
    if parent_revision == candidate_revision:
        return {"revision": parent_revision, "files": parent_files,
                "manifest": parent_manifest, "manifest_complete": True,
                "binding_stable": True}, \
            {"revision": candidate_revision, "files": current_files,
             "manifest": current_manifest, "manifest_complete": True,
             "binding_stable": True}, None
    return {"revision": parent_revision, "files": parent_files,
            "manifest": parent_manifest, "manifest_complete": True}, \
        {"revision": candidate_revision, "files": current_files,
         "manifest": current_manifest, "manifest_complete": True}, None


def _bound_edit_refusal(root: pathlib.Path, path: pathlib.Path,
                        target_ref: object = None) -> str | None:
    """Validate an optional structured repair exact replacement target before writing."""
    from kernel_bound_edit import enabled as bound_edit_enabled, validate_target
    if not bound_edit_enabled():
        return None
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        relative = path.as_posix().replace("\\", "/").lstrip("./")
    try:
        raw = path.read_bytes() if path.is_file() else b""
        source = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        return f"bound edit target rejected: cannot read current source ({type(error).__name__})"
    revision = "sha256:" + hashlib.sha256(raw).hexdigest()
    return validate_target(target_ref, path=relative, revision=revision, source=source)


def _capture_done_candidate_after_exception(
        measurement: object, context: object) -> object:
    """Capture the latest scoped source snapshot if an exception follows done."""
    from collections.abc import Mapping
    from kernel_done_is_submit_authority import (
        submit_done, wait_for_pending_mutations,
    )

    values = context if isinstance(context, Mapping) else {}
    data = values.get("d")
    data = data if isinstance(data, Mapping) else {}
    root = values.get("root")
    target = values.get("target")
    fileset = values.get("fileset")
    wait_for_pending_mutations(measurement)
    current_sources = {}
    snapshot = globals().get("_source_snapshot")
    product_paths = globals().get("_product_source_paths")
    if isinstance(root, pathlib.Path) and callable(snapshot):
        if bool(values.get("multifile")) and fileset is not None and callable(product_paths):
            current_sources = snapshot(root, product_paths(root, fileset, data))
        elif isinstance(target, pathlib.Path):
            try:
                relative = target.relative_to(root).as_posix()
                current_sources = snapshot(root, (relative,)) if target.is_file() else {}
            except (OSError, ValueError):
                current_sources = {}
    if not current_sources:
        fallback_sources = values.get("final_sources")
        current_sources = fallback_sources if isinstance(fallback_sources, Mapping) else {}
    source_map = {
        str(path): source for path, source in current_sources.items()
        if isinstance(path, str) and (isinstance(source, str) or source is None)
    } if isinstance(current_sources, Mapping) else {}
    parent_sources: dict[str, str] = {}
    parent = getattr(fileset, "parent", None)
    if isinstance(parent, Mapping):
        parent_sources = {
            str(path): source for path, source in parent.items()
            if isinstance(path, str) and isinstance(source, str)
        }
    elif isinstance(root, pathlib.Path) and isinstance(target, pathlib.Path):
        try:
            relative = target.relative_to(root).as_posix()
        except ValueError:
            relative = ""
        parent_source = data.get("parent_src")
        if relative and isinstance(parent_source, str):
            parent_sources[relative] = parent_source
    task_id = data.get("task_id") or data.get("id") or values.get("fn") or "candidate"
    intent = getattr(measurement, "_done_submit_pending", None)
    source = getattr(intent, "source", "model")
    return submit_done(
        measurement,
        source=source,
        destination=POC / "results" / "react_done_submissions",
        task_id=task_id,
        sources=source_map,
        parent_sources=parent_sources,
        out_of_scope=bool(
            data.get("out_of_scope")
            or getattr(measurement, "_candidate_out_of_scope", False)
        ),
        certification="not_evaluated",
    )


def _apply_revision_bound_insertion(
        path: pathlib.Path, new_str: str, target_ref: object, *,
        root: pathlib.Path, protected_paths: Sequence[object] = (),
        check_refs: Sequence[object] = (), public_brief: object = None) -> str:
    """Apply one host-issued module-boundary insertion without fuzzy matching."""
    if not isinstance(new_str, str) or not new_str.strip():
        return "insert_before REJECTED: new_str must contain authored source"
    if protected_test_path(path, protected_paths):
        return "insert_before REJECTED: protected public test paths are immutable"
    try:
        from kernel_revision_bound_edit import EditAuthority, EditPlan, EditPlanError, compose_candidate
        if hasattr(target_ref, "to_dict"):
            target_ref = target_ref.to_dict()
        if not isinstance(target_ref, Mapping):
            raise EditPlanError("TARGET_REF", "insert_before requires a host-issued target_ref")
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        current = path.read_bytes()
        authority_ref = str(target_ref.get("authority_ref") or "").strip()
        file_revision = str(target_ref.get("file_revision") or "").strip()
        if not authority_ref or not file_revision:
            raise EditPlanError("TARGET_REF", "target_ref is missing authority or file revision")
        authority = EditAuthority(
            root=root, sources={relative: current}, base_revision=file_revision,
            authority_ref=authority_ref, allowed_paths=(relative,),
        )
        plan = EditPlan.from_mapping({
            "base_revision": file_revision,
            "authority_ref": authority_ref,
            "inventory_ref": str(target_ref.get("source_receipt") or ""),
            "check_refs": [str(item) for item in check_refs if str(item).strip()],
            "edits": [{"op": "insert_before", "target_ref": dict(target_ref), "new": new_str}],
        })
        candidate = compose_candidate(authority, plan)[relative]
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        code = getattr(error, "code", type(error).__name__)
        return f"insert_before REJECTED [{code}]: {str(error)[:240]}"
    try:
        candidate_text = candidate.decode("utf-8")
        ast.parse(candidate_text, filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        return f"insert_before REJECTED: candidate source is invalid ({type(error).__name__})"
    from kernel_destructive_edit_guard import guard_write
    shrink_refusal = guard_write(path, candidate_text, root=root, public_brief=public_brief)
    if shrink_refusal is not None:
        return shrink_refusal
    from kernel_green_lock import guard_product_mutation, note_applied_product_edit
    refusal = guard_product_mutation(path)
    if refusal is not None:
        return "insert_before REJECTED: " + refusal
    applied, shadow = _effect_write(path, candidate_text, operation="insert_before")
    if not applied:
        return shadow or "insert_before SHADOW: effect withheld"
    note_applied_product_edit(path)
    return _post_edit_observation(path, "insert_before applied.")


def _source_map_digest(sources: Mapping[str, str | None]) -> str:
    payload = [(path, hashlib.sha256(source.encode("utf-8")).hexdigest()
                if isinstance(source, str) else None)
               for path, source in sorted(sources.items())]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _missing_declared_paths(root: pathlib.Path, fileset: FileSet, *,
                            journal: object = None) -> tuple[str, ...]:
    if _workspace_truth.enabled():
        return tuple(
            path for path in fileset.paths
            if _workspace_truth.exists(root, path, journal=journal).status == "missing"
        )
    return tuple(path for path in fileset.paths if not (root / path).is_file())


def _product_roots(row: Mapping[str, object]) -> tuple[str, ...]:
    """Return the public product roots used for dynamic candidate registration."""
    raw = row.get("allowed_product_paths")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        roots = tuple(str(path).strip().strip("/") for path in raw if str(path).strip())
        if roots:
            return roots
    return ("app",)


def _is_product_path(relative: str, row: Mapping[str, object]) -> bool:
    """Keep product snapshots out of tests, probes, scratch, and cache trees."""
    from stateframe.workcopy import _ignored
    try:
        from build_tasks.multifile_row import normalize_repo_path
        path = normalize_repo_path(relative, field="product snapshot path")
    except (ImportError, ValueError):
        return False
    parts = pathlib.PurePosixPath(path).parts
    if any(part in {"__pycache__", ".pytest_cache", ".git", ".test-tmp",
                   "tests", "test", "probes", "scratch", "tmp", "private",
                   "hidden_tests"} for part in parts):
        return False
    if pathlib.PurePosixPath(path).suffix in {".pyc", ".pyo"}:
        return False
    if any(_ignored(part, True) for part in parts[:-1]) or _ignored(parts[-1], False):
        return False
    if (parts[-1].startswith("test_") or parts[-1].endswith("_test.py")
            or parts[-1] == "conftest.py"):
        return False
    # Ordinary unscoped user repositories admit new product roots as well as
    # files present at startup. Exclusions and test guards above still apply.
    if row.get("tier") == "USER-REPO" and row.get("scoped_edits") is False:
        return True
    return any(path == root or path.startswith(root + "/") for root in _product_roots(row))


# The run loop imports kernel_green_lock.is_product_path under the same local name, so tool
# handlers inside it must call the row-aware check through this alias.
_row_is_product_path = _is_product_path


def _product_source_paths(root: pathlib.Path, fileset: FileSet,
                          row: Mapping[str, object]) -> tuple[str, ...]:
    """Merge declared product files with product files created during the run."""
    # FileSet may still carry a legacy declared path that is not a product
    # source (for example a probe or scratch file).  The final candidate is
    # explicitly a product-tree manifest, so apply the same boundary to both
    # initially declared and dynamically discovered files.
    paths = {path for path in fileset.paths if _is_product_path(path, row)}
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        relative = candidate.relative_to(root).as_posix()
        if _is_product_path(relative, row):
            paths.add(relative)
    return tuple(sorted(paths))


def _product_files_changed_count(
        root: pathlib.Path | None,
        fileset: FileSet,
        row: Mapping[str, object],
        *,
        target: pathlib.Path | None,
        current_sources: Mapping[str, str | None] | None = None,
        initial_sources: Mapping[str, str | None] | None = None,
) -> int:
    """Count net product-file changes against the run's initial product tree.

    ``initial_sources`` (snapshot taken before the first model turn) is the authority;
    the row's declared parent text is only a fallback (e.g. resumed runs).
    """
    if (not isinstance(root, pathlib.Path) or not isinstance(fileset, FileSet)
            or not isinstance(row, Mapping)):
        return 0
    try:
        if fileset.is_multi_file:
            parent_sources = (dict(initial_sources) if isinstance(initial_sources, Mapping)
                              else dict(fileset.parent))
            if isinstance(current_sources, Mapping):
                final_sources = dict(current_sources)
            else:
                snapshot_paths = set(initial_sources) if isinstance(initial_sources, Mapping) else set()
                snapshot_paths.update(fileset.paths)
                final_sources = _source_snapshot(root, tuple(sorted(snapshot_paths)))
        else:
            if not isinstance(target, pathlib.Path):
                return 0
            relative = target.relative_to(root).as_posix()
            if isinstance(initial_sources, Mapping) and relative in initial_sources:
                parent_sources = {relative: initial_sources[relative]}
            else:
                parent_source = row.get("parent_src")
                if not isinstance(parent_source, str):
                    return 0
                parent_sources = {relative: parent_source}
            if isinstance(current_sources, Mapping) and relative in current_sources:
                final_sources = {relative: current_sources[relative]}
            else:
                final_sources = _source_snapshot(root, (relative,))
    except (OSError, TypeError, ValueError):
        return 0

    paths = set(parent_sources) | set(final_sources)
    return sum(
        1 for path in paths
        if _is_product_path(path, row)
        and final_sources.get(path) != parent_sources.get(path)
    )


_FINAL_PRODUCT_HOST_CRASH_RESULTS = frozenset({
    "ERROR", "HOST_ERROR", "HOST_CRASH", "HOST_CRASHED",
    "HARNESS_ERROR", "HARNESS_TRANSPORT", "HARNESS_CONTRACT",
})


def _is_final_product_host_crash_result(result: str) -> bool:
    """Recognize terminal worker outcomes Stage 6 settles as operational errors."""
    normalized = result.strip().upper()
    return (normalized in _FINAL_PRODUCT_HOST_CRASH_RESULTS
            or normalized.startswith("ERR:"))


def _terminal_finalization_deferred(measurement: object) -> bool:
    """Return whether the finalization defer is still the run's latest request boundary."""
    journal = getattr(measurement, "journal", None)
    cached_records = getattr(journal, "cached_records", None)
    if not callable(cached_records):
        return False
    try:
        records = cached_records()
    except Exception:  # noqa: BLE001 - terminal bookkeeping must not replace the run outcome
        return False

    last_deferred = -1
    last_request = -1
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        kind = record.get("kind")
        if kind == "completion_owner_finalization_deferred":
            last_deferred = index
        elif kind == "request_submitted":
            last_request = index
    return last_deferred > last_request


def _record_final_product_no_candidate(
        measurement: object,
        *,
        result: str,
        product_files_changed: int,
        manifest_written: bool,
        manifest_write_failed: bool = False,
        host_crash: bool = False,
) -> bool:
    """Record one terminal run that settled without its final product manifest."""
    if manifest_written or getattr(
            measurement, "_final_product_no_candidate_emitted", False):
        return False

    changed = product_files_changed if type(product_files_changed) is int else 0
    changed = max(0, changed)
    finalization_deferred = _terminal_finalization_deferred(measurement)
    reason = ("finalization_deferred" if finalization_deferred else
              "manifest_write_failed" if manifest_write_failed else
              "host_stopped" if host_crash else
              "no_product_edit" if changed == 0 else "not_completed")
    # Mark before the durable append so a finalizer retry cannot duplicate a
    # record if the journal reports a post-fsync observer error.
    setattr(measurement, "_final_product_no_candidate_emitted", True)
    # 10-08 merge amendments (SEER/OPERATOR2): salvage_gradable tells RUNREAL to grade the final snapshot like the
    # TIMEOUT path when edits exist (a correct-but-unaccepted candidate is never lost); every field stays a short
    # scalar so the event is never blob-externalized (OPERATOR2's scanner reads raw journal lines).
    # The post-finish recorder journals while the writer is open, else stderr + post_finish_facts.jsonl; never raises.
    _record_post_finish_fact(
        measurement, "final_product_no_candidate",
        reason=reason,
        result=str(result)[:200],
        product_files_changed=changed,
        salvage_gradable=changed > 0,
        fact_owner="HOST",
        producer="react_repair",
        dispatch=False,
    )
    if finalization_deferred:
        _record_post_finish_fact(
            measurement, "terminal_outcome_recorded",
            path="finalization_deferred", reason=reason,
        )
    return True


def _materialize_public_tests(root: pathlib.Path, fileset: FileSet,
                              row: Mapping[str, object]) -> tuple[str, ...]:
    """Write declared public tests independently of the product source map."""
    from kernel_live_projection import is_harness_path

    sources = dict(fileset.public_tests)
    if not sources:
        return ()
    written = []
    for path, source in sorted(sources.items()):
        destination, normalized = _multi_path(root, path)
        if not is_harness_path(pathlib.Path(normalized)):
            raise ValueError(f"public test must identify a declared test file: {path}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
        written.append(normalized)
    return tuple(written)


def _staged_phases_enabled() -> bool:
    return os.environ.get("LONG_HORIZON_STAGED_PHASES", "off").strip().lower() in {
        "on", "1", "true", "yes",
    }


def _visible_staged_row(row: Mapping[str, object]) -> dict[str, object]:
    """Keep unreleased phase payloads out of StateFrame/task-row projections."""
    return {key: value for key, value in row.items() if key != "staged_phases"}


def _materialize_staged_phase(root: pathlib.Path, phase: Mapping[str, object]) -> tuple[str, ...]:
    """Materialize only the public files for one phase, after its gate is green."""
    from kernel_live_projection import is_harness_path

    raw_tests = phase.get("public_tests")
    if not isinstance(raw_tests, Sequence) or isinstance(raw_tests, (str, bytes)):
        raise ValueError("staged phase public_tests must be a sequence")
    written: list[str] = []
    for item in raw_tests:
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
            raise ValueError("staged phase public test needs path and source")
        source = item.get("source", item.get("parent_src"))
        if not isinstance(source, str):
            raise ValueError(f"staged public test has no source: {item['path']}")
        destination, normalized = _multi_path(root, item["path"])
        if not is_harness_path(pathlib.Path(normalized)):
            raise ValueError(f"staged public test must identify a declared test file: {normalized}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding="utf-8")
        written.append(normalized)
    return tuple(written)


def _staged_ticket_block(phase: int, ticket: str) -> str:
    return (
        f"\n\nFOLLOW-UP TICKET — PHASE {phase} "
        f"(released after phase {phase - 1} was accepted):\n{ticket}"
    )


def _append_staged_ticket(*, d: dict, msgs: list[dict], baseline_messages: list[dict],
                          information: object, phase: int, ticket: str) -> str:
    """Weld one released ticket into every history/frame brief carrier."""
    block = _staged_ticket_block(phase, ticket)
    seen_messages: set[int] = set()
    for carrier in (msgs, baseline_messages):
        if len(carrier) > 1 and isinstance(carrier[1], Mapping):
            if id(carrier[1]) in seen_messages:
                continue
            seen_messages.add(id(carrier[1]))
            carrier[1]["content"] = str(carrier[1].get("content") or "") + block
    for key in ("msg", "_bridge_ordinary_brief"):
        original = d.get(key)
        if not isinstance(original, str):
            original = str(d.get("msg") or "") if key == "_bridge_ordinary_brief" else ""
        d[key] = original + block
    information.ordinary_brief = str(getattr(information, "ordinary_brief", "")) + block
    return block


def _target_gone_lines(paths: Sequence[str], *, root: pathlib.Path | None = None,
                       journal: object = None) -> str:
    if not _workspace_truth.enabled():
        return "\n".join(f"TARGET_GONE path={path}" for path in paths)
    lines: list[str] = []
    for path in paths:
        fact = _workspace_truth.exists(root, path, journal=journal)
        if fact.status == "missing":
            lines.append(f"TARGET_GONE path={path}; {_workspace_truth.render_fact(fact)}")
        else:
            lines.append(f"TARGET_STATUS path={path}; {_workspace_truth.render_fact(fact)}")
    return "\n".join(lines)


def _ast_by_file(candidate: Mapping[str, str | None], fixed: Mapping[str, str]) -> dict:
    """Return per-file AST diagnostics; grading remains behavioural."""
    def detail(source: str | None) -> dict:
        normalized = norm(source) if isinstance(source, str) else None
        return {
            "parseable": normalized is not None,
            "digest": (hashlib.sha256(normalized.encode()).hexdigest()
                        if normalized is not None else None),
        }

    diagnostics = {}
    for path in sorted(set(candidate) | set(fixed)):
        candidate_detail = detail(candidate.get(path))
        fixed_detail = detail(fixed.get(path))
        diagnostics[path] = {
            "candidate": candidate_detail,
            "gold": fixed_detail,
            "match": (candidate_detail["parseable"] and fixed_detail["parseable"]
                      and candidate_detail["digest"] == fixed_detail["digest"]),
        }
    return diagnostics


def _persist_source_map(destination: pathlib.Path,
                        sources: Mapping[str, str | None]) -> pathlib.Path:
    """Persist complete final sources plus an explicit map for artifact consumers."""
    destination.mkdir(parents=True, exist_ok=True)
    for path, source in sorted(sources.items()):
        if not isinstance(source, str):
            continue
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    map_path = destination / "source_map.json"
    map_path.write_text(json.dumps({"files": dict(sorted(sources.items()))},
                                   indent=1, sort_keys=True))
    return map_path


_RECEIPT_FAILURE_TAIL_LIMIT = 1200
_FAILED_SELECTOR_LINE = re.compile(
    r"(?m)^\s*FAILED\s+(?P<selector>[^\s]+\.py::[^\s]+)"
)
_PYTEST_SEPARATOR_LINE = re.compile(
    r"^\s*(?:(?:[_=]\s*){3,}|[_=]{3,}\s+.+?\s+[_=]{3,})\s*$"
)
_PYTEST_FRAME_LINE = re.compile(
    r'^(?:File "[^\"]+", line \d+|[^\s]+\.py:\d+(?::\d+)?(?:\s|:|$))'
)


def _strip_pytest_separators(lines: Sequence[str]) -> list[str]:
    return [line for line in lines
            if not _PYTEST_SEPARATOR_LINE.match(line)]


def _pytest_failure_excerpt(lines: Sequence[str]) -> str:
    """Keep assertion evidence and one source frame, bounded for the model."""
    signal: list[str] = []
    last_frame = None
    for line in lines:
        stripped = line.lstrip()
        if "/_pytest/" in line or "site-packages" in line:
            continue
        if _PYTEST_FRAME_LINE.match(stripped):
            last_frame = line
        if (stripped.startswith(("E ", "> ", "FAILED "))
                or "assert" in stripped):
            signal.append(line)
    if last_frame is not None and last_frame not in signal:
        signal.append(last_frame)
    return "\n".join(signal[-40:] if signal else list(lines)[-40:])


def _bounded_failure_tail(failure_tail: str) -> str:
    """Keep recent output plus compact explicit pytest failure node IDs."""
    text = str(failure_tail or "")
    if len(text) <= _RECEIPT_FAILURE_TAIL_LIMIT:
        return text
    selectors: list[str] = []
    for match in _FAILED_SELECTOR_LINE.finditer(text):
        selector = match.group("selector")
        if selector not in selectors:
            selectors.append(selector)
    markers = "\n".join(f"FAILED {selector}" for selector in selectors)
    if len(markers) >= _RECEIPT_FAILURE_TAIL_LIMIT:
        return markers[-_RECEIPT_FAILURE_TAIL_LIMIT:]
    context_limit = _RECEIPT_FAILURE_TAIL_LIMIT - len(markers) - 1
    return text[-context_limit:] + "\n" + markers


def _attach_failure_tail(bolt, failure_tail: str) -> None:
    """Attach the revision-bound check handle instead of a second clipped output."""
    receipts = getattr(bolt, "_receipts", None)
    if receipts:
        measurement = getattr(bolt, "measurement", None)
        latest = getattr(measurement, "_latest_check_receipt", None)
        if isinstance(latest, Mapping):
            receipts[-1].update({
                "check_output_receipt": latest.get("full_output_handle"),
                "check_output_omission_counts": {
                    "stdout_bytes": latest.get("stdout_bytes", 0),
                    "stderr_bytes": latest.get("stderr_bytes", 0),
                    "stdout_omitted_bytes": latest.get("stdout_omitted_bytes", 0),
                    "stderr_omitted_bytes": latest.get("stderr_omitted_bytes", 0),
                },
            })
        else:
            receipts[-1]["failure_tail"] = _bounded_failure_tail(failure_tail)
        if measurement is not None:
            measurement.amend_receipt(receipts[-1])


def _annotate_receipt(bolt, *, paths: Sequence[str], scope: Sequence[str],
                      before: Mapping[str, str | None],
                      after: Mapping[str, str | None],
                      selectors: Sequence[str], failure_tail: str) -> None:
    """Attach bounded multi-file evidence to the receipt just emitted by KernelBolt."""
    receipts = getattr(bolt, "_receipts", None)
    if not receipts:
        return
    receipts[-1].update({
        "paths": list(paths),
        "scope": list(scope),
        "before_digest": _source_map_digest(before),
        "after_digest": _source_map_digest(after),
        "selectors": list(selectors),
    })
    measurement = getattr(bolt, "measurement", None)
    latest = getattr(measurement, "_latest_check_receipt", None)
    if isinstance(latest, Mapping):
        receipts[-1].update({
            "check_output_receipt": latest.get("full_output_handle"),
            "check_output_omission_counts": {
                "stdout_bytes": latest.get("stdout_bytes", 0),
                "stderr_bytes": latest.get("stderr_bytes", 0),
                "stdout_omitted_bytes": latest.get("stdout_omitted_bytes", 0),
                "stderr_omitted_bytes": latest.get("stderr_omitted_bytes", 0),
            },
        })
    else:
        receipts[-1]["failure_tail"] = _bounded_failure_tail(failure_tail)
    if measurement is not None:
        measurement.amend_receipt(receipts[-1])


def _record_rejected_edit_receipt(bolt, *, root: pathlib.Path, path: pathlib.Path,
                                  paths: Sequence[str], scope: Sequence[str],
                                  operation: str, target: str, dialect: str,
                                  old: str, new: str, detail: str,
                                  multifile: bool = False) -> None:
    """Record a refused edit with action identity and unchanged source evidence."""
    from kernel_action_identity import canonical_edit
    before = _source_snapshot(root, paths)
    attempt = canonical_edit(bolt, path, dialect=dialect, old=old, new=new)
    bolt.receipt("edit", operation, target, "REJECTED", detail,
                 attempt=attempt, target_path=receipt_target_path(root, path))
    receipts = getattr(bolt, "_receipts", None)
    if receipts:
        receipts[-1].update({
            "paths": list(paths),
            "scope": list(scope),
            "before_digest": _source_map_digest(before),
            "after_digest": _source_map_digest(_source_snapshot(root, paths)),
        })
        if getattr(bolt, "measurement", None) is not None:
            bolt.measurement.amend_receipt(receipts[-1])
    if multifile:
        _annotate_receipt(
            bolt, paths=paths, scope=scope, before=before,
            after=_source_snapshot(root, paths),
            selectors=tuple(getattr(bolt, "f2p_ids", ())), failure_tail=detail)


def _selector_error_repair_note(bolt, selector: str, turn: int) -> str | None:
    """Return one recovery note when a selector file was edited in this run."""
    selector_raw = str(selector or "").split("::", 1)[0].strip("`'\"")
    fallback_files = getattr(bolt, "_selector_error_fallback_files", None)
    if fallback_files is None:
        fallback_files = set()
        bolt._selector_error_fallback_files = fallback_files
    root = pathlib.Path(getattr(bolt, "root", "."))

    def normalize(candidate: object) -> str:
        raw = str(candidate or "").strip().strip("`'\"")
        if not raw:
            return ""
        path = pathlib.Path(raw)
        if path.is_absolute():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                return path.as_posix()
        return raw.lstrip("./")

    selector_file = normalize(selector_raw)
    if not selector_file or selector_file in fallback_files:
        return None

    for receipt in getattr(bolt, "_receipts", ()):
        if not isinstance(receipt, Mapping) or receipt.get("kind") != "edit":
            continue
        if receipt.get("result_class") != "APPLIED":
            continue
        candidates = receipt.get("scope") or ()
        if not candidates:
            material = str(receipt.get("material") or "")
            candidates = (material.split("|", 1)[0],)
        if selector_file not in {normalize(candidate) for candidate in candidates}:
            continue
        fallback_files.add(selector_file)
        edited_turn = receipt.get("turn")
        if edited_turn is None:
            edited_turn = turn
        return (f"[kernel] SELECTOR_ERROR on {selector_file} which YOU edited at turn "
                f"{edited_turn}: re-read the current file with `read` and fix its import, "
                "then re-run. Do not overwrite it from a parent revision. Product edits stay.")
    return None


def _multi_path(root: pathlib.Path, raw: str) -> tuple[pathlib.Path, str]:
    """Resolve a multi-file tool path exactly; never guess by basename."""
    from build_tasks.multifile_row import normalize_repo_path
    normalized = normalize_repo_path(raw, field="sandbox path")
    target = root / normalized
    if root.resolve() not in target.resolve().parents:
        raise ValueError(f"sandbox path escapes checkout: {raw}")
    return target, normalized


def _edit_path_label(path: pathlib.Path, root: pathlib.Path | None = None) -> str:
    """Render the path carried by a typed edit rejection."""
    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    return path.as_posix()


def _typed_edit_rejection(code: str, path: pathlib.Path, *,
                          root: pathlib.Path | None = None,
                          hint: str = "", symbol: str | None = None,
                          definitions: Sequence[Mapping[str, object]] | None = None) -> str:
    """Return the JSON edit-rejection envelope shown to the model and frame.

    The executor is the authority for the target path.  Keeping the alternative in the
    same envelope means a rejected shortcut cannot strand a multi-file repair with only
    an unhelpful prose explanation.
    """
    path_text = _edit_path_label(path, root)
    normalized = path_text.replace("\\", "/").lstrip("./")
    protected = (normalized.startswith(("test/", "tests/"))
                 or "/tests/" in "/" + normalized
                 or normalized.startswith("test_")
                 or "/test_" in "/" + normalized
                 or normalized.endswith("/conftest.py"))
    if protected or str(code) in {"NOT_EDITABLE", "OUTSIDE_WORKSPACE", "DIRECTORY"}:
        alternative = {
            "tool": "read",
            "path": path_text,
            "hint": ("This path is protected or unavailable for writing; inspect it and "
                     "repair the authorized product path instead.")[:600],
        }
    elif str(code) == "PATH_NOT_FOUND":
        alternative = {
            "tool": "file_create",
            "path": path_text,
            "hint": ("Use file_create only when this missing path is intentionally new; "
                     "otherwise use read/list to locate the existing file.")[:600],
        }
    else:
        alternative = {
            "tool": "str_replace",
            "path": path_text,
            "hint": (str(hint).strip() or
                     "Read the current definition and use an exact text replacement here.")[:600],
        }
    payload: dict[str, object] = {
        "alternatives": [alternative],
        "code": str(code),
        "path": path_text,
        "status": "REJECTED",
    }
    if symbol:
        payload["symbol"] = str(symbol)
    if definitions is not None:
        payload["definitions"] = [dict(item) for item in definitions]
    if (str(code) == "NO_SUCH_DEFINITION" and symbol and not protected
            and path.is_file() and not pathlib.PurePath(path_text).is_absolute()):
        try:
            from kernel_revision_bound_edit import insertion_affordance
            source = path.read_bytes()
            insertion = insertion_affordance(
                path=normalized,
                source=source,
                symbol=str(symbol),
                source_receipt="source:" + hashlib.sha256(source).hexdigest(),
                authority_ref="repair-edit-authority",
                allowed_paths=(normalized,),
            )
        except (OSError, UnicodeError, ValueError, TypeError):
            insertion = None
        if isinstance(insertion, Mapping) and insertion.get("status") == "AVAILABLE":
            payload["insertion_affordance"] = insertion
            payload["alternatives"].insert(0, {
                "tool": "insert_before",
                "path": path_text,
                "target_ref": insertion.get("target_ref"),
                "write_authority": insertion.get("write_authority"),
                "hint": ("Use the returned opaque target_ref with the bounded batch "
                         "insert_before operation; do not replace the module."),
            })
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _is_typed_edit_rejection(observation: object) -> bool:
    if not isinstance(observation, str):
        return False
    try:
        payload = json.loads(observation)
    except (TypeError, ValueError):
        return False
    return (isinstance(payload, Mapping) and payload.get("status") == "REJECTED"
            and isinstance(payload.get("code"), str)
            and isinstance(payload.get("path"), str)
            and isinstance(payload.get("alternatives"), list))


def _canonical_typed_edit_rejection(observation: object) -> str | None:
    """Recover the JSON envelope if an advisory was appended after it."""
    if not isinstance(observation, str):
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(observation.lstrip())
    except (TypeError, ValueError):
        return None
    if (isinstance(payload, Mapping) and payload.get("status") == "REJECTED"
            and isinstance(payload.get("code"), str)
            and isinstance(payload.get("path"), str)
            and isinstance(payload.get("alternatives"), list)):
        return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return None


def _create_path_contained(root: pathlib.Path, rel: str) -> bool:
    """True only for a non-empty path that resolves inside the workspace root.

    Bughunt r2AE #3: single-file file_create had no containment check. This is the
    same rule reads use (resolve, then is_relative_to the resolved root), including
    symlinked-directory escapes, but without read-side basename recovery.
    """
    if not rel:
        return False
    try:
        return (root / rel).resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError):
        return False


def _read_edit_source(path: pathlib.Path, verb: str) -> tuple[str, str | None]:
    """Return (text, None) for an editable UTF-8 file, else ("", typed refusal).

    Edit applicators must never raise on file content (bughunt r2AE #1, #4): a
    non-UTF-8 byte or a NUL byte is a property of the existing file, not of the
    model's edit, so the refusal says so and nothing is changed.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        return "", f"{verb} REJECTED: cannot read {path.name} ({type(error).__name__}). Nothing was changed."
    if b"\x00" in raw:
        return "", (f"{verb} REJECTED: {path.name} contains NUL bytes, so it is not an editable "
                    "text source (it was already unparseable before this edit). Nothing was changed; "
                    "leave it unchanged or replace it with a whole-file rewrite if the task requires it.")
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as error:
        return "", (f"{verb} REJECTED: {path.name} is not valid UTF-8 (byte {error.start}). "
                    "Nothing was changed; leave it unchanged or rewrite the file as valid UTF-8 "
                    "if the task requires it.")


def _read_path_is_deferred_required(measurement: object, path: str) -> bool:
    """True when the live request told the model this path's source is missing.

    The read budget must never refuse a path the request itself lists as required
    but not included.
    Resolves the runtime the same way the request budget does.
    """
    wanted = str(path or "").replace("\\", "/").strip().lstrip("./")
    if not wanted:
        return False
    candidates: list[object] = []
    try:
        from kernel_pilot import ACTIVE_RUNTIME
        active = ACTIVE_RUNTIME.get()
        if active is not None:
            candidates.append(active)
    except (ImportError, LookupError):
        pass
    for name in ("presentation_runtime", "completion_owner_runtime"):
        runtime = getattr(measurement, name, None)
        if runtime is not None:
            candidates.append(runtime)
    try:
        from kernel_completion_owner import _pending_deferred_source_paths
    except ImportError:
        return False
    for runtime in candidates:
        try:
            deferred = _pending_deferred_source_paths(runtime)
        except (AttributeError, TypeError, ValueError):
            continue
        if any(str(item).lstrip("./") == wanted for item in deferred):
            return True
    return False


def _edit_observation_is_rejection(observation: object) -> bool:
    return (_canonical_typed_edit_rejection(observation) is not None
            or isinstance(observation, str) and observation.startswith((
                "edit failed", "edit REJECTED", "edit NO-OP",
                "str_replace NO-OP", "str_replace REJECTED",
                "REJECTED_SHRINK ", "REJECTED_DESTRUCTIVE_COMMAND ")))


def _normalize_edit_rejection(observation: str, path: pathlib.Path, *,
                              root: pathlib.Path | None = None,
                              symbol: str | None = None) -> str:
    """Convert legacy executor prose into the typed wire result.

    Lower-level guards remain readable for their direct unit tests.  Once a result crosses
    the live edit executor, however, every rejection has the same machine-readable shape.
    """
    if observation.startswith(("REJECTED_SHRINK ", "REJECTED_DESTRUCTIVE_COMMAND ")):
        return observation
    if _is_typed_edit_rejection(observation) or "NO-OP" in observation:
        return observation
    lowered = observation.lower()
    if not ("rejected" in lowered or observation.startswith("edit failed")):
        return observation
    if "no function named" in lowered or "no ast target" in lowered:
        code = "NO_SUCH_DEFINITION"
    elif ("does not exist" in lowered or "filenotfounderror" in lowered
          or "no such file" in lowered):
        code = "PATH_NOT_FOUND"
    elif ("protected" in lowered or "pre-existing test" in lowered
          or "outside the workspace" in lowered or "not editable" in lowered
          or "is a directory" in lowered or "not a file" in lowered):
        code = "NOT_EDITABLE"
    elif "whole-file edit would delete" in lowered:
        code = "NO_CLOBBER"
    elif "retry gate" in lowered:
        code = "RETRY_GATE"
    elif ("syntax" in lowered or "parse" in lowered
          or "function_source" in lowered):
        code = "INVALID_FUNCTION_SOURCE"
    else:
        code = "EDIT_REJECTED"
    return _typed_edit_rejection(code, path, root=root, hint=observation, symbol=symbol)


def _is_alias_assignment(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and isinstance(statement.value, (ast.Name, ast.Attribute))
    )


def _source_span(source: str, node: ast.AST) -> tuple[int, int]:
    from kernel_frame_edit_integrity import definition_span, enabled as integrity_enabled
    from kernel_inflight_fixes import enabled as inflight_fixes_enabled

    inflight_fix = inflight_fixes_enabled()
    if (integrity_enabled() and not inflight_fix
            and isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))):
        return definition_span(source, node)
    lines = source.splitlines(keepends=True)
    if not inflight_fix:
        start_line = max(0, int(getattr(node, "lineno", 1)) - 1)
        end_line = max(start_line, int(getattr(node, "end_lineno", start_line + 1)) - 1)
        start = sum(len(line) for line in lines[:start_line]) + int(getattr(node, "col_offset", 0))
        end_col = int(getattr(node, "end_col_offset", len(lines[end_line])))
        end = sum(len(line) for line in lines[:end_line]) + end_col
        return start, end
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def source_index(line_number: int, byte_column: int) -> int:
        line_index = max(0, line_number - 1)
        line_start = line_offsets[line_index]
        # Python AST columns are UTF-8 byte offsets, but ``source`` is a decoded
        # string.  Convert the byte prefix before slicing it as characters.
        line = lines[line_index]
        char_column = len(line.encode("utf-8")[:max(0, byte_column)].decode("utf-8"))
        return line_start + char_column

    decorators = tuple(getattr(node, "decorator_list", ()))
    if integrity_enabled() and isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        start_line = min(int(getattr(item, "lineno", 1)) for item in (node, *decorators))
        start_column = min(int(getattr(item, "col_offset", 0)) for item in (node, *decorators))
    else:
        start_line = int(getattr(node, "lineno", 1))
        start_column = int(getattr(node, "col_offset", 0))
    end_line = int(getattr(node, "end_lineno", start_line))
    end_column = int(getattr(node, "end_col_offset", len(lines[end_line - 1])))
    start = source_index(start_line, start_column)
    end = source_index(end_line, end_column)
    return start, end


def _multi_definition_payload(source: str) -> tuple[ast.Module, list[ast.AST], str] | None:
    """Return a parsed multi-definition/module payload under the boundary switch."""
    if (os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() != "on"
            and not _edit_hardening_enabled()):
        return None
    replacement = textwrap.dedent(source)
    try:
        tree = ast.parse(replacement)
    except SyntaxError:
        return None
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    has_module_prefix = any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))
        for node in tree.body
    )
    if len(definitions) <= 1 and not (definitions and has_module_prefix):
        return None
    for index, statement in enumerate(tree.body):
        allowed = isinstance(statement, (ast.Import, ast.ImportFrom,
                                         ast.ClassDef, ast.FunctionDef,
                                         ast.AsyncFunctionDef))
        docstring = (index == 0 and isinstance(statement, ast.Expr)
                     and isinstance(statement.value, ast.Constant)
                     and isinstance(statement.value.value, str))
        if not (allowed or docstring or _is_alias_assignment(statement)):
            return tree, definitions, (
                "unsupported top-level statement "
                f"`{ast.unparse(statement)}`; use str_replace instead"
            )
    return tree, definitions, ""


def _insert_new_definitions(current: str, payload_source: str,
                            definitions: Sequence[ast.AST]) -> str:
    """Insert new top-level definitions before their first top-level use."""
    if not definitions:
        return current
    payload_source = textwrap.dedent(payload_source)
    rendered = [ast.get_source_segment(payload_source, node) or ast.unparse(node)
                for node in definitions]
    block = "\n\n".join(textwrap.dedent(item).strip("\n") for item in rendered)
    tree = ast.parse(current)
    first_use: int | None = None
    names = {str(getattr(node, "name", "")) for node in definitions}
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and statement.name in names:
            continue
        loads = {
            node.id for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if loads & names:
            first_use = _source_span(current, statement)[0]
            break
    if first_use is None:
        return current.rstrip() + "\n\n" + block + "\n"
    prefix = current[:first_use]
    separator = "" if prefix.endswith("\n\n") else "\n\n"
    return prefix + separator + block + "\n\n" + current[first_use:]


def _apply_multi_definition_edit(path: pathlib.Path, current: str, replacement: str,
                                  *, root: pathlib.Path | None = None,
                                  public_brief: object = None) -> str | None:
    parsed = _multi_definition_payload(replacement)
    if parsed is None:
        return None
    if _edit_normalization_enabled():
        return _typed_edit_rejection(
            "MULTI_DEFINITION_SCOPE", path, root=root,
            hint=("normalization requires a complete file for multiple intended hunks "
                  "or one parser-resolved complete definition; no definitions were spliced"),
        )
    payload_tree, payload_defs, payload_error = parsed
    current_tree = ast.parse(current, filename=str(path))
    current_defs = {
        node.name: node for node in current_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    names = [node.name for node in payload_defs]
    missing = [name for name in names if name not in current_defs]
    hardening = _edit_hardening_enabled()
    # Duplicate definitions would edit the same span twice with stale offsets.
    # Reject them before applying edits to avoid invalid syntax.
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates and not payload_error:
        payload_error = f"payload defines {duplicates[0]} more than once; submit each definition exactly once"
    if payload_error or (missing and not hardening):
        reason = payload_error or f"no current definition named {missing[0]}"
        details = [{"name": name, "status": "UNAPPLIED",
                    "reason": reason if name in missing or payload_error else "blocked"}
                   for name in names]
        return _typed_edit_rejection(
            "MULTI_DEFINITION_EDIT", path, root=root,
            hint=("multi-definition edit was not applied atomically: " + reason),
            definitions=details,
        )

    operations: list[tuple[int, int, str]] = []
    for node in payload_defs:
        if node.name not in current_defs:
            continue
        old_node = current_defs[node.name]
        start, end = _source_span(current, old_node)
        payload_source = textwrap.dedent(replacement)
        if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
            from kernel_inflight_fixes import enabled as inflight_fixes_enabled
            if inflight_fixes_enabled():
                payload_start, payload_end = _source_span(payload_source, node)
                segment = payload_source[payload_start:payload_end]
            else:
                from kernel_frame_edit_integrity import definition_source
                segment = definition_source(payload_source, node)
        else:
            segment = ast.get_source_segment(payload_source, node)
        operations.append((start, end, textwrap.dedent(segment or ast.unparse(node))))
    candidate = current
    for start, end, segment in sorted(operations, reverse=True):
        candidate = candidate[:start] + segment + candidate[end:]
    if missing:
        candidate = _insert_new_definitions(candidate, replacement,
                                            [node for node in payload_defs if node.name in missing])
        if hardening:
            _emit_edit_hardening(path=str(path), target="<multi-definition>",
                                 dedented=True, class_scope_resolved=False,
                                 multi_definition_add=True, added_definitions=missing)

    try:
        candidate_tree = ast.parse(candidate, filename=str(path))
    except SyntaxError as error:   # r45 in-flight fix 2: typed refusal, never an exception out of the edit tool
        return _typed_edit_rejection(
            "MULTI_DEFINITION_EDIT", path, root=root,
            hint=f"multi-definition result does not parse: {error.msg}; nothing was written",
            definitions=[{"name": name, "status": "UNAPPLIED", "reason": str(error)} for name in names],
        )
    existing_imports = {ast.dump(node, include_attributes=False)
                        for node in candidate_tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom))}
    missing_imports = [node for node in payload_tree.body
                      if isinstance(node, (ast.Import, ast.ImportFrom))
                      and ast.dump(node, include_attributes=False) not in existing_imports]
    if missing_imports:
        import_text = "\n".join(ast.unparse(node) for node in missing_imports) + "\n\n"
        insertion = 0
        body = candidate_tree.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            insertion = _source_span(candidate, body[0])[1]
            while insertion < len(candidate) and candidate[insertion] in "\r\n":
                insertion += 1
        candidate = candidate[:insertion] + import_text + candidate[insertion:]

    # Preserve alias assignments bundled with a multi-definition payload.  A
    # missing alias is safe to append only after every replacement above has
    # already been proven applicable.
    aliases = [node for node in payload_tree.body if _is_alias_assignment(node)]
    if aliases:
        existing_aliases = {
            node.targets[0].id: node for node in ast.parse(candidate).body
            if _is_alias_assignment(node)
        }
        alias_ops: list[tuple[int, int, str]] = []
        append_text: list[str] = []
        for alias in aliases:
            alias_name = alias.targets[0].id
            rendered = ast.unparse(alias)
            if alias_name in existing_aliases:
                start, end = _source_span(candidate, existing_aliases[alias_name])
                alias_ops.append((start, end, rendered))
            else:
                append_text.append(rendered)
        for start, end, rendered in sorted(alias_ops, reverse=True):
            candidate = candidate[:start] + rendered + candidate[end:]
        if append_text:
            candidate = candidate.rstrip() + "\n\n" + "\n".join(append_text) + "\n"

    try:
        ast.parse(candidate, filename=str(path))
    except SyntaxError as error:
        details = [{"name": name, "status": "UNAPPLIED", "reason": str(error)}
                   for name in names]
        return _typed_edit_rejection(
            "MULTI_DEFINITION_EDIT", path, root=root,
            hint=f"multi-definition result does not parse: {error.msg}",
            definitions=details,
        )
    if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
        from kernel_frame_edit_integrity import duplicate_decorator_rejection
        rejection = duplicate_decorator_rejection(current, candidate, file=path.name)
        if rejection is not None:
            return rejection
    if candidate.strip() == current.strip():
        return "edit NO-OP: your replacement is IDENTICAL to the current source"
    from kernel_destructive_edit_guard import guard_write
    shrink_refusal = guard_write(
        path, candidate, root=root, public_brief=public_brief,
    )
    if shrink_refusal is not None:
        return shrink_refusal
    from kernel_green_lock import guard_product_mutation, note_applied_product_edit
    refusal = guard_product_mutation(path)
    if refusal is not None:
        return "edit REJECTED: " + refusal
    applied, shadow = _effect_write(path, candidate, encoding="utf-8")
    if not applied:
        return shadow or "edit SHADOW: effect withheld"
    note_applied_product_edit(path)
    return _post_edit_observation(path, "edit applied.")


def _top_level_definition_name(function_source: str) -> tuple[str | None, str | None]:
    """Extract the one top-level definition an ``edit`` payload names."""
    replacement = textwrap.dedent(function_source)
    try:
        tree = ast.parse(replacement)
    except SyntaxError as exc:
        return None, f"function_source does not parse ({exc.msg} line {exc.lineno})"
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    boundary_on = (os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on"
                   or _edit_hardening_enabled())
    if boundary_on and definitions:
        parsed_multi = _multi_definition_payload(replacement)
        if parsed_multi is not None:
            _multi_tree, _multi_defs, multi_error = parsed_multi
            if multi_error:
                return None, f"function_source {multi_error}"
            return definitions[0].name, None
    if len(definitions) != 1 or not tree.body \
            or not isinstance(tree.body[0], (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        names = ", ".join(node.name for node in definitions) or "no top-level definition"
        return None, f"function_source must contain exactly one top-level definition; found {names}"
    if len(tree.body) > 1:
        if not boundary_on:
            return None, ("function_source must contain exactly one top-level definition; "
                          f"found {definitions[0].name}")
        for statement in tree.body[1:]:
            valid_alias = _is_alias_assignment(statement)
            if not valid_alias:
                try:
                    rendered = ast.unparse(statement)
                except (AttributeError, ValueError):
                    rendered = type(statement).__name__
                return None, (
                    "function_source has unsupported trailing top-level statement "
                    f"`{rendered}`; use str_replace instead"
                )
    return definitions[0].name, None


def _top_level_definitions_lost(old_source: str, new_source: str) -> list[str]:
    """Top-level def/class names present in old_source but absent from new_source."""
    def names(source: str) -> set[str] | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        return {node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    before, after = names(old_source), names(new_source)
    if before is None or after is None:
        return []  # unparseable old file: nothing reliable to protect; new syntax is checked by the caller
    return sorted(before - after)


def _replace_file(path: pathlib.Path, new_source: str, *, root: pathlib.Path | None = None,
                  public_brief: object = None, journal: object = None) -> str:
    """Replace one complete declared file when no function entry was supplied."""
    from kernel_edit_guard import edit_guard_mode
    edit_guard_mode()
    if not path.exists():
        if _workspace_truth.enabled():
            fact = _workspace_truth.exists(root, str(path), journal=journal)
            return "edit REJECTED: " + _workspace_truth.render_fact(fact)
        return f"edit REJECTED: {path} does not exist"
    if path.is_dir():
        return f"edit REJECTED: {path} is a directory"
    if _nonpy_edit_applies(path):
        candidate = new_source.encode("utf-8")
        if path.read_bytes() == candidate:
            return "edit NO-OP: replacement is identical"
        candidate, normalization_refusal = _normalize_candidate_for_publication(
            path, candidate, mode="complete-file", root=root,
        )
        if normalization_refusal is not None:
            return normalization_refusal
        assert candidate is not None
        rejection, warning = _validate_v21_candidate(path, new_source, "edit")
        if rejection is not None:
            return rejection
        return _publish_nonpy_candidate(
            path, candidate, operation="edit", root=root,
            public_brief=public_brief, warning=warning,
        )
    try:
        ast.parse(new_source, filename=str(path))
    except SyntaxError as exc:
        return f"edit REJECTED: syntax error at line {exc.lineno}"
    try:
        old_source = path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return "edit REJECTED: source is not valid UTF-8; use an exact byte-preserving edit"
    if old_source == new_source:
        return "edit NO-OP: replacement is identical"
    normalized_candidate, normalization_refusal = _normalize_candidate_for_publication(
        path, new_source.encode("utf-8"), mode="complete-file", root=root,
    )
    if normalization_refusal is not None:
        return normalization_refusal
    assert normalized_candidate is not None
    try:
        new_source = normalized_candidate.decode("utf-8")
    except UnicodeDecodeError:
        return "edit REJECTED: normalized candidate is not valid UTF-8"
    if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
        from kernel_frame_edit_integrity import duplicate_decorator_rejection
        rejection = duplicate_decorator_rejection(old_source, new_source, file=path.name)
        if rejection is not None:
            return rejection
    lost = _top_level_definitions_lost(old_source, new_source)
    if lost:
        # Whole-file mode writes exactly what it is given. A fragment (e.g. one method)
        # would silently delete every other definition in the file (09-24: a TokenBucket
        # method edit wiped the class, __init__ and _Bucket). Deleting definitions must be
        # explicit, so refuse and name exactly what would be lost.
        return ("edit REJECTED: whole-file edit would delete existing top-level "
                f"definitions {lost} from {path.name}. `edit` here replaces the ENTIRE file: "
                "include every definition you want to keep, or use str_replace for a smaller "
                "change (including an explicit deletion).")
    from kernel_destructive_edit_guard import guard_write
    shrink_refusal = guard_write(
        path, new_source, root=root, public_brief=public_brief,
    )
    if shrink_refusal is not None:
        return shrink_refusal
    from kernel_green_lock import guard_product_mutation, note_applied_product_edit
    refusal = guard_product_mutation(path)
    if refusal is not None:
        return "edit REJECTED: " + refusal
    applied, shadow = _effect_write(path, new_source, encoding="utf-8")
    if not applied:
        return shadow or "edit SHADOW: effect withheld"
    note_applied_product_edit(path)
    return _post_edit_observation(path, "edit applied.")


_SYSTEM_TRIM_DETAIL: list = [None]  # last trim_system detail, journaled by emit_system_trim


def emit_system_trim(journal: object) -> bool:
    """own event of STATEFRAME_TOKENLAB_SYSTEM_TRIM (once per run, at the first journal)."""
    detail, _SYSTEM_TRIM_DETAIL[0] = _SYSTEM_TRIM_DETAIL[0], None
    if detail is None or not hasattr(journal, "emit"):
        return False
    journal.emit("system_trim_applied", producer="kernel_tokenlab_v13", dispatch=False, **detail)
    return True


def _prepared_tool_contract(prepared: object, fallback: object) -> list[dict]:
    """Return the exact final tool schemas used for this prepared request."""
    tools = getattr(prepared, "tools", None)
    if tools is None:
        tools = fallback
    if not isinstance(tools, (list, tuple)):
        return []
    return [copy.deepcopy(dict(tool)) for tool in tools if isinstance(tool, Mapping)]


def _contract_errors(tools: object, tool_name: str, arguments: object,
                     *, arguments_parse_error: bool = False) -> list[str]:
    """Validate against the same final menu that was presented to the model."""
    from kernel_completion_owner import validate_tool_arguments as owner_validate_tool_arguments
    errors = list(owner_validate_tool_arguments(tools, tool_name, arguments))
    names = [
        str(tool.get("function", {}).get("name", ""))
        for tool in tools
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    ] if isinstance(tools, (list, tuple)) else []
    if tool_name not in set(names) and not any(
            error.startswith(f"tool '{tool_name}' is not offered") for error in errors):
        offered = ", ".join(name for name in names if name) or "none"
        errors.insert(
            0,
            f"tool '{tool_name}' is not offered for the current state; "
            f"choose one of: {offered}",
        )
    if arguments_parse_error:
        errors.insert(0, "arguments are not a JSON object")
    return errors


def _render_contract_rejection(tool_name: str, errors: Sequence[str], *, binding_current: bool) -> str:
    """Render actionable, bounded contract feedback for the next model turn."""
    details = list(errors)
    if not details and not binding_current:
        details.append("workspace changed since the proposal's state")
    rendered = "[step-contract] REJECTED before dispatch: " + "; ".join(details)
    # Soak C4 anchor recovery (default off): edits refused before dispatch (bad/ungrounded
    # refs, unoffered edit tool) are the LAB10 failure shape; the executor tail never sees them.
    if tool_name in {"edit", "str_replace", "file_create", "insert_before"}:
        from kernel_anchor_recovery import annotate_rejection
        rendered = annotate_rejection(rendered, rejected=True)
    return rendered


def _post_edit_observation(path: pathlib.Path, result: str) -> str:
    """Append guarded host facts after a successful mutation, without changing off-mode output."""
    if "applied." not in result:
        return result
    try:
        from kernel_edit_guard import host_observed_findings
        findings = host_observed_findings(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, UnicodeError):
        return result
    return result if not findings else result + "\n" + "\n".join(findings)


def _render_multifile_list(fileset: FileSet, *, limit: int = 1400) -> str:
    lines = ["FILE LIST (declared source files):"]
    for path in fileset.paths:
        names = ", ".join(fileset.changed.get(path, ()))
        lines.append(f"- {path}" + (f" [{names}]" if names else ""))
    rendered = "\n".join(lines)
    if len(rendered) > limit:
        rendered = rendered[:limit].rsplit("\n", 1)[0] + "\n- ... (FILE LIST BOUNDED)"
    return rendered


# ── NATIVE TOOL-CALLING (E285) ────────────────────────────────────────────────
#
# Native tool calls deliver structured arguments and avoid ambiguities in
# free-form command parsing, including quoting and truncated code fences.
#
# Ported from MC's agent-runner/app/openai_client.py: the DeepSeek DSML quirk — Flash
# occasionally emits tool calls as literal <｜DSML｜...> markup inside content with an empty
# tool_calls array (known upstream bug). MC's parser grammar is reproduced verbatim.


def _list_files_command(args: Mapping[str, object]) -> str:
    """Serialize optional listing args without shifting depth/limit into scope."""
    import shlex

    line = "list_files"
    path = args.get("path")
    depth = args.get("depth")
    limit = args.get("limit")
    if path:
        line += " " + shlex.quote(str(path))
    if depth or limit:
        if not path:
            line += " ."
        line += " " + str(depth or 2)
        if limit:
            line += " " + str(limit)
    return line


TOOLS = [
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search authorized repository text or paths. Results are pageable; "
                       "the default mode preserves regex semantics.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "regex or literal to search"},
            "path": {"type": "string", "description": "optional file or dir to scope"},
            "mode": {"type": "string", "enum": ["regex", "literal", "path"],
                     "description": "regex by default; literal or path when explicit"},
            "file_glob": {"type": "string", "description": "optional file-name/path glob"},
            "glob": {"type": "string", "description": "alias for file_glob"},
            "page_size": {"type": "integer", "description": "bounded rows per page"},
            "cursor": {"type": "string", "description": "continuation from a prior page"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List a bounded workspace tree with repository-relative paths, "
                       "entry types, and byte sizes. It is read-only; path is the optional "
                       "directory scope, depth is capped at 4, and limit caps returned rows.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "optional repository directory scope"},
            "depth": {"type": "integer", "minimum": 1, "maximum": 4,
                      "description": "tree depth; 1 lists direct children"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 240,
                      "description": "maximum returned entries"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read a file or directory page. Source results include exact range, "
                       "revision, requested/effective range, clipping reason, exact page bytes, "
                       "and a revision-bound next cursor. Use an inclusive start_line/end_line "
                       "range, or resume a cursor with max_lines/max_chars.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "workspace-relative file or directory path"},
            "line": {"type": "integer", "minimum": 1,
                     "description": "1-based start_line alias; default 1. Use with end_line for an inclusive range."},
            "start_line": {"type": "integer", "minimum": 1,
                           "description": "1-based inclusive range start; default 1."},
            "end_line": {"type": "integer", "minimum": 1,
                         "description": "1-based inclusive range end. Honors the full requested range up to the 1000-line host cap; omit max_lines."},
            "symbol": {"type": "string", "description": "qualified symbol name to read; do not combine with line coordinates or cursor"},
            "column": {"type": "integer", "minimum": 0,
                       "description": "0-based starting column for a long-line read; do not combine with cursor"},
            "max_lines": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 120,
                          "description": "Lines per cursor/open page; default 120, range 1-1000. For an inclusive end_line range, omit this field."},
            "max_chars": {"type": "integer", "minimum": 512, "maximum": 64000, "default": 12000,
                          "description": "Character budget per page; default 12000, range 512-64000. A clip returns a revision-bound next cursor."},
            "page_size": {"type": "integer", "minimum": 1, "maximum": 240,
                          "description": "Maximum directory rows per page; default 60, maximum 240."},
            "cursor": {"type": "string",
                       "description": "Revision-bound continuation from a prior source page; resumes at its exact line and column. Use with max_lines/max_chars only."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit",
        "description": "Replace one COMPLETE top-level function or class in a file. "
                       "function_source must start with `def`/`async def`/`class` and parse.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "function_source": {"type": "string"}},
            "required": ["path", "function_source"]}}},
    {"type": "function", "function": {
        "name": "str_replace",
        "description": "Surgical edit: replace ONE exact occurrence of old_str with "
                       "new_str in a file. old_str must match exactly once (include "
                       "enough surrounding lines to be unique). Preferred over `edit` "
                       "for small changes inside large functions.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"}},
            "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {
        "name": "insert_before",
        "description": "Apply one bounded revision-bound insertion at a host-issued source "
                       "boundary. Use the exact target_ref returned by an edit refusal; "
                       "the operation never fuzzy-matches or replaces a module.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "target_ref": {"type": "object"},
            "new_str": {"type": "string"},
            "check_refs": {"type": "array", "items": {"type": "string"}},
        }, "required": ["path", "target_ref", "new_str"]}}},
    {"type": "function", "function": {
        "name": "authored_check",
        "description": "Create and execute one separate host-admitted authored check "
                       "against immutable parent and candidate revisions. The check must "
                       "name its public requirement quote, selector, source, and scratch "
                       "write authority; generic probes remain observation-only.",
        "parameters": {"type": "object", "properties": {
            "check_id": {"type": "string"},
            "requirement_id": {"type": "string"},
            "clause": {"type": "string"},
            "requirement_quote": {"type": "string"},
            "path": {"type": "string"},
            "selector": {"type": "string"},
            "language": {"type": "string"},
            "source": {"type": "string"},
            "environment": {"type": "object"},
            "observation_marker": {"type": "string"},
            "write_authority": {"type": "object"},
            "authorized_paths": {"type": "array", "items": {"type": "string"}},
            "protected_paths": {"type": "array", "items": {"type": "string"}},
            "public_text": {"type": "string"},
            "source_unit_handle": {"type": "string", "description": "opaque host-issued source-unit handle"},
        }, "required": ["check_id", "requirement_id", "clause", "requirement_quote",
                         "path", "selector", "source", "write_authority"]}}},
    {"type": "function", "function": {
        "name": "file_create",
        "description": "Create a NEW file with the given content (parents created). "
                       "Refuses to overwrite an existing file — use str_replace/edit "
                       "for those.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "file_delete",
        "description": "Delete one existing product file. Refuses protected tests, directories, symlinks, and paths outside the editable scope.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "why": {"type": "string"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "file_move",
        "description": "Move one existing product file to a new path, creating destination directories. Refuses collisions and unauthorized paths.",
        "parameters": {"type": "object", "properties": {
            "source": {"type": "string"}, "destination": {"type": "string"},
            "why": {"type": "string"}},
            "required": ["source", "destination"]}}},
    {"type": "function", "function": {
        "name": "run_tests",
        "description": "Run pytest. target may be a file, file::node, or a bare "
                       "test function name.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "read_check_receipt",
        "description": "Retrieve lossless public check stdout, stderr, or assertion context "
                       "from a revision-bound receipt handle. Pages are bounded; use next_offset.",
        "parameters": {"type": "object", "properties": {
            "handle": {"type": "string"},
            "stream": {"type": "string", "enum": [
                "combined", "stdout", "stderr", "assertion_context"]},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 12000}},
            "required": ["handle"]}}},
    {"type": "function", "function": {
        "name": "recall",
        "description": "Read-only recall of any indexed run item by id. The item is included in the next frame; the compact index is always available.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string", "description": "exact id from the RECALL INDEX"}},
            "required": ["id"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "done",
        "description": "Declare the fix complete and verified.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"},
            "no_change_paths": {
                "type": "array", "maxItems": 128,
                "description": "For a declared source path that needs no edit, give its exact path and a concise reason.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 512},
                        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
                    },
                    "required": ["path", "reason"],
                    "additionalProperties": False,
                },
            },
        }},
    }},
]

_JQ_REPLACE_MODULE_TOOL = {"type": "function", "function": {
    "name": "replace_module",
    "description": "Replace one complete Python module after a host revision check. "
                   "Provide expected_revision and either source or a retained candidate_id. "
                   "The host validates scope, protected files, syntax, and publishes atomically.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"},
        "expected_revision": {"type": "string"},
        "source": {"type": "string"},
        "candidate_id": {"type": "string"}},
        "required": ["path", "expected_revision"]}}}

def _nonpy_neutralize_tool_surface(tools: Sequence[Mapping[str, object]]) -> list:
    """Make user-facing action descriptions describe changes, not only repairs."""
    updated = copy.deepcopy(list(tools))
    descriptions = {
        "edit": ("Replace one complete existing file or top-level definition. For an "
                  "exact text change use str_replace; for a new file use file_create."),
        "file_create": ("Create a new file with the given content (parents created). "
                        "Refuses to overwrite an existing file; use str_replace or edit "
                        "for an existing file."),
        "run_tests": ("Run the repository's configured checks. target may be a file, "
                      "file::node, or a named check."),
        "done": "Declare the requested change complete and verified.",
        "replace_module": ("Replace one complete existing source file after a host revision "
                            "check. The host validates the file type and publishes atomically."),
    }
    for tool in updated:
        if not isinstance(tool, Mapping):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if name in descriptions:
                # Keep the single EditAuthority prefix intact for the release
                # runner's authority contract assertions.
                old = str(function.get("description") or "")
                try:
                    from kernel_edit_affordance import AUTHORITY_EDIT_DESCRIPTION as _auth
                except ImportError:
                    _auth = ""
                if _auth and old.startswith(_auth):
                    function["description"] = _auth + " " + descriptions[name]
                else:
                    function["description"] = descriptions[name]
    return updated


def _nonpy_neutralize_system(system: str) -> str:
    """Remove repair/Python-only framing from the enabled creation prompt."""
    replacements = {
        "You are repairing a bug in a large production Python codebase.":
            "You are making a requested change in a large production codebase.",
        "edit <path>             then a ```python block with the COMPLETE corrected function or class":
            "edit <path>             then provide complete replacement source or file content",
        "test [path]             run the repository's tests (defaults to tests touching this function)":
            "test [path]             run the repository's configured checks for this change",
        "a replacement that does not parse is REJECTED rather than written":
            "a replacement that violates its file-type gate is REJECTED rather than written",
        "`done` with no edit made is a wasted run: you will have submitted the original bug.":
            "`done` with no change made is a wasted run: you will have submitted the original state.",
    }
    for old, new in replacements.items():
        system = system.replace(old, new)
    return system


_PROBE_TOOL = {"type": "function", "function": {
    "name": "probe",
    "description": "Run one short Python snippet against the current workspace to "
                   "discriminate an uncovered brief obligation. The expected field is "
                   "your unverified interpretation; subprocess output is host-observed.",
    "parameters": {"type": "object", "properties": {
        "obligation": {"type": "string", "description": "brief obligation id, such as C1"},
        "code": {"type": "string", "description": "short Python snippet calling the target"},
        "expected": {"type": "string", "description": "your expectation from the brief, not source"}},
        "required": ["obligation", "code"]}}}

TOOLS.append(SHELL_SCHEMA)
from kernel_agent_schemas import BUILTIN_SCHEMAS
TOOLS.extend(BUILTIN_SCHEMAS)

# Tool calls accept an optional `why`: one declared line
# of intent per action. The model narrates its plan as it works (a commitment device that
# measurably keeps small models on track), the receipt tape records intent alongside
# outcome, and intent-vs-outcome divergence becomes snitch-architecture input. Optional,
# never enforced — an honest field, not a gate.
for _t in TOOLS:
    _t["function"]["parameters"].setdefault("properties", {})["why"] = {
        "type": "string",
        "description": "one line: what you expect this action to show or achieve, and "
                       "how it serves your current plan"}

_PROBE_TOOL["function"]["parameters"].setdefault("properties", {})["why"] = {
    "type": "string",
    "description": "one line: what you expect this action to show or achieve, and "
                   "how it serves your current plan"}
# _PROBE_TOOL is added per-request only when STATEFRAME_PROBE_GATE=on (never mutate TOOLS).

_OBSERVATION_PROBE_TOOL = {"type": "function", "function": {
    "name": "probe",
    "description": "Run one bounded, observation-only Python snippet in a fresh "
                   "read-only workspace snapshot. The host returns a typed status "
                   "and bounded output; this never verifies or completes the task.",
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python snippet, at most 8 KiB"},
        "question": {"type": "string", "description": "optional question for this observation"},
        "why": {"type": "string", "description": "one line purpose of the observation"}},
        "required": ["code"]}}}
# This surface is selected only by STATEFRAME_PROBE=on.  It is intentionally a
# distinct object from the older completion-challenge probe above.

_WHERE_USED_FILE_LIST_TOOL = {"type": "function", "function": {
    "name": "file_list",
    "description": "List the host-observed repository files before choosing a change. "
                   "The optional path scopes the inventory and may contain spaces.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "optional repository path or directory"},
        "limit": {"type": "integer", "description": "maximum rows (default 200)"},
        "why": {"type": "string", "description": "one line purpose of the inventory"}},
        "required": []}}}
_DEPENDENCY_FACT_TOOL = {"type": "function", "function": {
    "name": "dependency_fact",
    "description": "Characterize one named attribute or function from an installed "
                   "dependency in a bounded, read-only sandbox. The returned shape is "
                   "a HOST_OBSERVED characterization keyed by package and version; it "
                   "is descriptive only and never proves an application contract.",
    "parameters": {"type": "object", "properties": {
        "package": {"type": "string", "description": "installed distribution/import name"},
        "module": {"type": "string", "description": "optional import module; defaults to package"},
        "target": {"type": "string", "description": "dotted attribute or function name"},
        "args": {"type": "array", "description": "JSON-shaped positional call arguments"},
        "kwargs": {"type": "object", "description": "JSON-shaped keyword call arguments"},
        "call": {"type": "boolean", "description": "call the target when callable; defaults true"},
        "question": {"type": "string", "description": "one narrow characterization question"},
         "why": {"type": "string", "description": "one line purpose of the observation"}},
         "required": ["package", "target"]}}}


def _coordinated_change_tool_enabled() -> bool:
    return os.environ.get(
        "STATEFRAME_TOKENLAB_ATOMIC_MULTI_FILE_APPLY", "off").strip().lower() == "on"


def _owner_coordinate_read_surface(selected: list[dict]) -> list[dict]:
    """Expose revision-pinned range reads only while coordinates are enabled."""
    from kernel_owner_first import coordinates_enabled
    if not coordinates_enabled():
        return selected
    result = copy.deepcopy(selected)
    for tool in result:
        function = tool.get("function", {})
        if function.get("name") != "read":
            continue
        parameters = function.get("parameters", {})
        properties = parameters.get("properties", {})
        properties["expected_revision"] = {
            "type": "string",
            "description": "Optional exact sha256 source revision; stale coordinates are rejected.",
        }
        break
    return result


def _tools_for_probe_modes(*, probe_gate_on: bool,
                           observation_probe_on: bool,
                           dependency_fact_on: bool = False) -> list[dict]:
    """Select the per-request probe surface without changing the OFF base list."""
    from kernel_shell_tool import service as shell_service
    shell = shell_service()
    base_tools = [tool for tool in TOOLS if tool["function"]["name"] != "shell"
                  or (shell is not None and shell.settings.shell != "off")]
    from kernel_agent_tools import project_menu
    base_tools = project_menu(base_tools)
    boundary_aliases = os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on"
    probe_modes = ["python"]
    if not observation_probe_on:
        probe_modes.append("legacy")
    else:
        if os.environ.get("STATEFRAME_BEHAVIOR_WITNESS", "off").strip().lower() == "on":
            probe_modes.extend(["behavior_witness", "clause_discriminator_propose"])
        if os.environ.get("STATEFRAME_CLAUSE_REVIEW", "off").strip().lower() == "on":
            if "clause_discriminator_propose" not in probe_modes:
                probe_modes.append("clause_discriminator_propose")
            probe_modes.extend(["clause_review_propose", "clause_review_assess"])
        if dependency_fact_on:
            probe_modes.append("dependency_fact")
    if observation_probe_on:
        observation_tool = _OBSERVATION_PROBE_TOOL
        behavior_witness_on = os.environ.get(
            "STATEFRAME_BEHAVIOR_WITNESS", "off").strip().lower() == "on"
        clause_review_on = os.environ.get(
            "STATEFRAME_CLAUSE_REVIEW", "off").strip().lower() == "on"
        typed_witness_fields = (
            "clause_id", "entrypoint", "input", "expected", "rationale")
        if behavior_witness_on or clause_review_on:
            observation_tool = copy.deepcopy(observation_tool)
        if behavior_witness_on:
            observation_tool["function"]["parameters"]["properties"].update({
                "clause_id": {"type": "string", "description": "current public clause id"},
                "entrypoint": {"type": "string", "description": "public-source callable, including an internal callable in a public module"},
                "source_unit_handle": {"type": "string", "description": "opaque host-issued source-unit handle"},
                "input": {"type": ["object", "array", "string", "number", "boolean", "null"], "description": "JSON-shaped frozen fixture"},
                "expected": {"type": ["object", "array", "string", "number", "boolean", "null"], "description": "immutable expected observation"},
                "rationale": {"type": "string", "description": "public-contract reason for this case"},
            })
            observation_tool["function"]["parameters"]["required"] = list(typed_witness_fields)
            observation_tool["function"]["description"] = (
                "Propose one language-neutral public behavior witness as a compact typed object; "
                "the host derives provenance, admits it, and runs the bounded probe. "
                "Required fields: " + ", ".join(typed_witness_fields) + ".")
        if clause_review_on:
            observation_tool["function"]["description"] = (
                "Use mode clause_discriminator_propose to submit one typed "
                "{clause_ref, entrypoint, input, expected_relation} case, or use "
                "the clause review modes for the existing bounded packet flow; the "
                "host owns execution, provenance, comparison, and bounds.")
        selected = [tool for tool in base_tools if tool is not _PROBE_TOOL] + [observation_tool]
    else:
        selected = ([tool for tool in base_tools if tool is not _PROBE_TOOL]
                    + ([_PROBE_TOOL] if probe_gate_on else []))
    from kernel_where_used import mode as where_used_mode
    if where_used_mode() == "on":
        selected = selected + [copy.deepcopy(_WHERE_USED_FILE_LIST_TOOL)]
    from kernel_repo_intel import tool_schemas as repo_intel_tool_schemas
    selected.extend(repo_intel_tool_schemas())
    if dependency_fact_on:
        selected.append(_DEPENDENCY_FACT_TOOL)
    if _coordinated_change_tool_enabled():
        from kernel_coordinated_change_ledger import augment_str_replace_tool
        selected = augment_str_replace_tool(selected)
    from kernel_decisive_factor import tool_surface as decisive_factor_tool_surface
    selected = decisive_factor_tool_surface(selected)
    if not boundary_aliases:
        from kernel_job_watch import tool_surface as job_tool_surface
        selected = job_tool_surface(selected)
        from kernel_completion_owner import canonicalize_tool_surface
        return canonicalize_tool_surface(
            _owner_coordinate_read_surface(selected), probe_modes=probe_modes)
    selected = copy.deepcopy(selected)
    for tool in selected:
        function = tool.get("function", {})
        if function.get("name") == "edit":
            function["description"] = (
                "Replace one COMPLETE top-level function or class. With "
                "STATEFRAME_BOUNDARY_CHECKS=on, trailing module-level alias assignments "
                "such as `legacy_name = new_name` are accepted in function_source; "
                "other trailing statements are rejected with a typed str_replace alternative."
            )
    from kernel_job_watch import tool_surface as job_tool_surface
    selected = job_tool_surface(selected)
    from kernel_completion_owner import canonicalize_tool_surface
    return canonicalize_tool_surface(
        _owner_coordinate_read_surface(selected), probe_modes=probe_modes)

_DSML_PIPE = "｜"  # full-width vertical line
_DSML_INVOKE_RE = re.compile(
    rf'<{_DSML_PIPE}DSML{_DSML_PIPE}invoke\s+name="([^"]*)"[^>]*>([\s\S]*?)'
    rf"</{_DSML_PIPE}DSML{_DSML_PIPE}invoke>", re.IGNORECASE)
_DSML_PARAM_RE = re.compile(
    rf'<{_DSML_PIPE}DSML{_DSML_PIPE}parameter\s+name="([^"]*)"'
    rf'(?:\s+string="(true|false)")?[^>]*>([\s\S]*?)'
    rf"</{_DSML_PIPE}DSML{_DSML_PIPE}parameter>", re.IGNORECASE)


def _parse_r1_tool_calls(content: str) -> list:
    """E360 — DeepSeek-R1's NATIVE dialect (measured live, watchdog s01-s07): ASCII
    `<|tool_calls_begin|>` markers wrapping `{"function": "<name>", ...}` objects, with
    the real args sometimes split into a following ```json fence. Merge every JSON
    object in the reply that carries a "function" key (fenced or bare); later objects'
    keys fill earlier gaps. Never raises; [] when nothing matches."""
    _TOOLS = ("grep", "read", "edit", "str_replace", "insert_before", "authored_check", "file_create", "file_delete", "file_move", "shell",
               "replace_module", "run_tests", "reproduce", "test", "probe", "done",
               "where_defined", "who_calls", "references")
    # Fix round (rv_figure_it_out_REPORT.md finding 3): the three figure_it_out
    # operator names must only be recognized -- and so only turned into native
    # calls from this parser -- when STATEFRAME_FIGURE_IT_OUT is actually on.
    # Unconditionally including them meant an identical TRACE-shaped JSON reply
    # parsed as [] on baseline but as a real tool call in this lane, even with
    # the switch absent/off.
    try:
        from kernel_figure_it_out import enabled as _figure_it_out_enabled
        if _figure_it_out_enabled():
            _TOOLS += ("BIND_QUANTITY", "CONTRACT_CASES", "TRACE_BEHAVIOR")
    except (ImportError, ValueError):
        pass
    if os.environ.get("STATEFRAME_GET_SURE", "off").strip().lower() == "on":
        _TOOLS += ("predict",)
    if os.environ.get("STATEFRAME_DEPENDENCY_FACT", "off") == "on":
        _TOOLS += ("dependency_fact",)
    if os.environ.get("STATEFRAME_DECISIVE_FACTOR", "off") == "on":
        _TOOLS += ("decide",)
    try:
        from kernel_job_watch import enabled as job_watch_enabled
        if job_watch_enabled():
            _TOOLS += ("launch_job", "job_status", "job_output", "job_kill")
            from kernel_frame_diet import JOB_TOOL_NAME, enabled as frame_diet_enabled
            if frame_diet_enabled():
                _TOOLS += (JOB_TOOL_NAME,)
    except (ImportError, ValueError):
        pass
    from kernel_where_used import mode as where_used_mode
    if where_used_mode() == "on":
        _TOOLS += ("file_list",)
    try:
        # Depth-aware JSON object extraction (11d rule: never regex-count braces).
        objs = []
        i = 0
        while i < len(content):
            if content[i] == "{":
                depth, j, instr, esc = 0, i, False, False
                while j < len(content):
                    c = content[j]
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == '"':
                        instr = not instr
                    elif not instr:
                        if c == "{":
                            depth += 1
                        elif c == "}":
                            depth -= 1
                            if depth == 0:
                                break
                    j += 1
                if depth == 0 and j < len(content):
                    try:
                        o = json.loads(content[i:j + 1])
                        if isinstance(o, dict):
                            objs.append(o)
                    except (TypeError, ValueError):
                        pass
                    i = j + 1
                    continue
            i += 1
        name, args = None, {}
        for o in objs:
            fn = o.get("function")
            if isinstance(fn, dict):                 # nested OpenAI-ish variant
                nm = fn.get("name")
                if isinstance(nm, str) and name is None:
                    name = nm
                    inner = fn.get("parameters") or fn.get("arguments") or {}
                    if isinstance(inner, dict):
                        args.update(inner)
            elif isinstance(fn, str) and name is None:
                name = fn
                args.update({k: v for k, v in o.items() if k != "function"})
            elif name is not None and "function" not in o and "type" not in o:
                for k, v in o.items():
                    args.setdefault(k, v)
        if name is None:
            # Bare-word variant: a line that IS a tool name, args in a json object after.
            bm = re.search(r"(?:^|\n)\s*(" + "|".join(_TOOLS) + r")\s*\n", content)
            if bm:
                name = bm.group(1)
                for o in objs:
                    if "function" not in o:
                        for k, v in o.items():
                            args.setdefault(k, v)
        if not name or name not in _TOOLS:
            return []
        return [{"id": "r1_0", "function":
                 {"name": name, "arguments": json.dumps(args)}}]
    except Exception:                                                # noqa: BLE001
        return []


def parse_dsml_tool_calls(content: str) -> list:
    """MC's grammar, verbatim semantics: never raises, [] on truncated/absent markup."""
    if not content:
        return []
    if f"{_DSML_PIPE}DSML{_DSML_PIPE}" not in content:
        return _parse_r1_tool_calls(content)                         # E360 fallback
    try:
        out = []
        for i, m in enumerate(_DSML_INVOKE_RE.finditer(content)):
            name, body = m.group(1), m.group(2)
            if not name:
                continue
            args = {}
            for pm in _DSML_PARAM_RE.finditer(body):
                pname, is_str, raw = pm.group(1), pm.group(2), pm.group(3)
                if not pname:
                    continue
                if is_str == "false":
                    try:
                        args[pname] = json.loads(raw.strip())
                    except (TypeError, ValueError):
                        args[pname] = raw
                else:
                    args[pname] = raw
            out.append({"id": f"dsml_{i}", "function":
                        {"name": name, "arguments": json.dumps(args)}})
        return out
    except Exception:                                            # noqa: BLE001
        return []


def _normalize_probe_tool_args(args: Mapping[str, object]) -> dict[str, object]:
    """Preserve every typed probe field needed by downstream witness dispatch."""
    dispatch_variant = str(args.get("_dispatch_variant") or "")
    return {
        "obligation": str(args.get("obligation") or ""),
        "clause_id": str(args.get("clause_id") or ""),
        "clause_ref": args.get("clause_ref"),
        "source_revision": args.get("source_revision"),
        "check_identity": args.get("check_identity"),
        "oracle_identity": args.get("oracle_identity"),
        "entrypoint": str(args.get("entrypoint") or ""),
        "source_unit_handle": str(args.get("source_unit_handle") or ""),
        "input": args.get("input"),
        "code": str(args.get("code") or ""),
        "expected": args.get("expected"),
        "expected_relation": args.get("expected_relation"),
        "rationale": str(args.get("rationale") or ""),
        "question": str(args.get("question") or ""),
        "mode": str(args.get("mode") or dispatch_variant or "python"),
        "package": args.get("package"),
        "module": args.get("module"),
        "target": args.get("target"),
        "args": args.get("args"),
        "kwargs": args.get("kwargs"),
        "projection": args.get("projection"),
        "episode_id": args.get("episode_id"),
        "schema": args.get("schema"),
        "clause_ids": args.get("clause_ids"),
        "clauses": args.get("clauses"),
        "packet": args.get("packet"),
        "assessment": args.get("assessment"),
        "max_cases": args.get("max_cases"),
        "max_rounds": args.get("max_rounds"),
        "token_contract": args.get("token_contract"),
        "call": args.get("call", True),
    }


def _render_semantic_facts(ev) -> str | None:
    """E365e — render a SemanticEvidence record as MEASURED FACTS for the model.
    CONTRADICTED evidence (source violates the required contract) is the strongest
    possible localization signal and the consumer's missing-evidence advisory
    deliberately doesn't cover it. Facts only: predicate + per-requirement states.
    Never a suggestion, never a patch."""
    if ev is None:
        return None
    try:
        status = getattr(getattr(ev, "status", None), "value", "")
        if status not in ("contradicted", "unknown"):
            return None
        bad = []
        for req in getattr(ev, "requirements", ()) or ():
            state = getattr(getattr(req, "state", None), "value", "?")
            code = getattr(getattr(req, "code", None), "value", "?")
            if state != "true":
                # E368: include WHERE — the evidence carries exact source spans.
                spans = ""
                prov = getattr(req, "provenance", ()) or ()
                if prov:
                    p0 = prov[0]
                    a, b = getattr(p0, "start_line", None), getattr(p0, "end_line", None)
                    if a:
                        spans = f" (target fn lines {a}-{b})"
                bad.append(f"{code}={state}{spans}")
        if not bad:
            return None
        pred = getattr(getattr(ev, "predicate", None), "expression", "") or ""
        subj = getattr(ev, "subject", None)
        where = f"{getattr(subj, 'path', '?')}:{getattr(subj, 'symbol', '?')}"
        # E371: surface the producers' CONCRETE mismatch witnesses ("contract says
        # ACCEPT" provenance lines where source disagrees) — test vectors, not patches.
        cases = []
        for req in getattr(ev, "requirements", ()) or ():
            for p in getattr(req, "provenance", ()) or ():
                fact = str(getattr(p, "fact", "") or "")
                _fu = fact.upper()
                if ("CONTRACT SAYS ACCEPT" in _fu and "REJECTED" in _fu) or \
                   ("CONTRACT SAYS REJECT" in _fu and "ACCEPTED" in _fu) or \
                   ("=UNFILTERED" in _fu and "=FILTERED" in _fu):
                    cases.append(fact[:160])
        case_txt = ("\n  Concrete mismatches (each is a witness your fix must flip): "
                    + " | ".join(cases[:3])) if cases else ""
        return ("[kernel] MEASURED CONTRACT FACTS for " + where +
                (f" — required behaviour: {pred}." if pred else ".") +
                " Static analysis of the CURRENT source finds: " + "; ".join(bad[:4]) +
                "." + case_txt +
                "\n  These are measurements, not suggestions — your fix must make the "
                "false ones true.")
    except Exception:                                                # noqa: BLE001
        return None


_HARNESS_TERMINAL_OUTCOMES = frozenset({"HARNESS_TRANSPORT", "HARNESS_CONTRACT"})
_CONNECTION_FAILURE_NAMES = ("RemoteDisconnected", "ConnectionReset", "ConnectionRefused",
                             "URLError", "IncompleteRead")


def _is_connection_failure(exc: BaseException) -> bool:
    """Only a connection-level failure may wait on /health (audit #4c).

    An ``HTTPError`` is a response from a live server; awaiting /health for it
    returns instantly and re-sends the same bytes.  Classify it instead.
    """
    if isinstance(exc, urllib.error.HTTPError):
        _mark_http_contract(exc)
        return False
    return any(name in type(exc).__name__ for name in _CONNECTION_FAILURE_NAMES)


def _mark_http_contract(exc: BaseException) -> None:
    """Tag an uncaptured ``HTTPError`` with its transport class (audit #4b)."""
    if not isinstance(exc, urllib.error.HTTPError) or getattr(exc, "transport_class", None):
        return
    from kernel_bounded_http import transport_class
    from kernel_measurement import _server_error_from
    server_error = _server_error_from(exc, KEY or "")
    exc.server_error = server_error
    exc.transport_class = transport_class(exc.code, server_error, type(exc).__name__)
    if exc.transport_class == "contract":
        exc.harness_contract = True
        exc.outcome = "HARNESS_CONTRACT"
        exc.failure_origin = "harness"


def _await_endpoint(max_wait: int = 300) -> bool:
    """E353 (the E338 sibling): after a connection-class failure, WAIT for the LLM
    endpoint to come back (pod OOM restarts take ~1-3 min) instead of burning the
    lane as ERR. Bounded; True when healthy again."""
    base = API.rsplit("/v1/", 1)[0]
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=4) as r:
                if b"ok" in r.read():
                    return True
        except Exception:                                            # noqa: BLE001
            pass
        time.sleep(10)
    return False


def _live_action_references(measurement, bolt, fileset, target=None) -> dict[str, list[object]]:
    """Collect host-owned selectors for the next action schema, without model parsing."""
    def add(result: list[object], value: object) -> None:
        if isinstance(value, (str, int, float, bool)) and value not in result:
            result.append(value)

    paths: list[object] = []
    for path in getattr(fileset, "paths", ()):
        add(paths, path)
    try:
        relative = target.resolve().relative_to(pathlib.Path(measurement.owner.root).resolve()) \
            if target is not None else None
        if relative is not None:
            add(paths, relative.as_posix())
    except (AttributeError, OSError, ValueError):
        pass
    try:
        binding = measurement.binding()
        if isinstance(binding, Mapping):
            for field_name in ("product_files", "harness_files"):
                files = binding.get(field_name)
                if isinstance(files, Mapping):
                    for path in files:
                        add(paths, path)
    except Exception:  # noqa: BLE001 - selector telemetry must not stop a request
        pass

    selectors: list[object] = []
    for selector in getattr(bolt, "f2p_ids", ()) or ():
        add(selectors, selector)
    for selector in getattr(fileset, "public_tests", {}) or {}:
        add(selectors, selector)
    ledger = getattr(measurement, "step_ledger", None)
    records = getattr(ledger, "records", ()) if ledger is not None else ()
    step_numbers: list[object] = []
    receipt_ids: list[object] = []
    for record in records or ():
        if isinstance(record, Mapping):
            add(step_numbers, record.get("step"))
            arguments = record.get("arguments")
            if isinstance(arguments, Mapping) and record.get("tool") == "run_tests":
                add(selectors, arguments.get("target"))
            for receipt_id in record.get("receipt_ids", ()) or ():
                add(receipt_ids, receipt_id)
            add(receipt_ids, record.get("step_record_id"))
    for receipt in getattr(bolt, "_receipts", ()) or ():
        if isinstance(receipt, Mapping):
            add(receipt_ids, receipt.get("receipt_id"))
            add(step_numbers, receipt.get("turn"))

    work_ids: list[object] = []
    decision_state = getattr(measurement, "decision_state", None)
    for collection_name in ("items", "notes"):
        collection = getattr(decision_state, collection_name, {}) if decision_state is not None else {}
        if isinstance(collection, Mapping):
            for work_id in collection:
                add(work_ids, work_id)
    try:
        work_record = getattr(measurement, "work_record", None)
        if work_record is not None:
            from kernel_work_record import open_work
            for work_id in open_work(work_record):
                add(work_ids, work_id)
    except Exception:  # noqa: BLE001 - reference telemetry must not stop a request
        pass
    references: list[object] = []
    for row in getattr(measurement, "retained_evidence", ()) or ():
        if isinstance(row, Mapping):
            add(references, row.get("reference"))
    return {"paths": paths, "test_selectors": selectors,
            "receipt_ids": receipt_ids, "references": references,
            "work_ids": work_ids, "step_numbers": step_numbers}


def ask_tools(messages, mt=4000, temp=None, tools=None, known_refs=None):
    """Tool-call turn: returns (content, tool_calls) — tool_calls in wire shape.

    Same transport hardening as ask(): finish_reason retry-doubling, 420s wall clock per
    attempt in a worker thread, transport exceptions attempt-scoped with backoff.
    """
    import concurrent.futures as _cf
    from kernel_measurement import (
        ACTIVE_REQUEST, ProviderRequestTimeoutError, activate_turn_timing,
        is_provider_request_timeout_error,
        output_token_limits, per_request_timeout_seconds,
        request_timeout_budget_seconds, request_timeouts,
        skip_capped_length_retry_enabled, timeout_retry_frozen_enabled,
        turn_timing_step,
    )
    capture = ACTIVE_REQUEST.get()
    from kernel_pilot import ACTIVE_RUNTIME
    from kernel_request_kind import require_capture
    require_capture(capture, ACTIVE_RUNTIME.get())
    from kernel_wire_string_hygiene import record_replacements, sanitize_messages, sanitize_value
    messages, replacements = sanitize_messages(messages)
    record_replacements(
        getattr(capture, "journal", None), replacements,
        step=getattr(capture, "step", None),
        request_id=getattr(capture, "request_id", None),
    )
    tools, replacements = sanitize_value(tools)
    record_replacements(
        getattr(capture, "journal", None), replacements,
        step=getattr(capture, "step", None),
        request_id=getattr(capture, "request_id", None),
    )
    socket_timeout, wall_timeout = request_timeouts()
    timeout_retry_active = per_request_timeout_seconds() is not None
    frozen_timeout_retry_active = (
        capture is not None and timeout_retry_active and timeout_retry_frozen_enabled())
    timeout_retry_attempt: int | None = None
    skip_capped_length_retry = skip_capped_length_retry_enabled()
    mt, output_ceiling = output_token_limits(mt)
    from kernel_step_contract import action_schema_enabled
    action_schema_mode = action_schema_enabled()
    from kernel_response_truncation import enabled as _truncation_typed
    typed_truncation = _truncation_typed()
    truncations = 0
    if typed_truncation and capture is not None:
        from kernel_response_truncation import note_checked as _truncation_checked
        _truncation_checked(getattr(capture, "journal", None))

    def _one_call(body, call_timeout):
        if capture is not None:
            if frozen_timeout_retry_active:
                return capture.send_with_frozen_timeout_retry(body, API, KEY)
            if timeout_retry_active:
                return capture.send(body, API, KEY, transport_attempt_limit=1,
                                    journal_timeouts=False)
            return capture.send(body, API, KEY)
        req = urllib.request.Request(
            API, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=call_timeout) as r:
            return json.loads(r.read())["choices"][0]

    if tools is None:
        from kernel_probe_gate import probe_gate_mode
        from kernel_probe import observation_probe_mode
        tools = _tools_for_probe_modes(
            probe_gate_on=probe_gate_mode() == "on",
            observation_probe_on=observation_probe_mode() == "on")
    if capture is not None:
        from kernel_tool_schema_diet import note_applied as note_schema_diet_applied
        note_schema_diet_applied(getattr(capture, "journal", None), tools)
    from kernel_native_tools import enabled as native_batches_enabled
    native_batches_mode = native_batches_enabled()
    if action_schema_mode and native_batches_mode:
        raise ValueError("STATEFRAME_ACTION_SCHEMA cannot use native tool batches")
    self_plan_thinking_body = None
    if (capture is not None
            and getattr(capture, "self_plan_wants_plan", False)
            and getattr(capture, "stateframe_think_prefill_mode", "off") == "self_plan"):
        from kernel_measurement import RequestCapture
        from kernel_self_plan_prefill import (
            build_pair_requests, extract_plan, journal_plan, pair_cache_enabled,
            self_plan_id_slot,
        )
        base_body = {
            "model": MODEL, "messages": list(messages), "max_tokens": mt,
            "temperature": TEMPERATURE if temp is None else temp, "tools": tools,
        }
        _apply_sampler_fields(base_body)
        from sampler_config import journal_sent as _journal_sampler_sent
        _journal_sampler_sent(capture, base_body, phase="self_plan")
        from kernel_step_contract import configure_reasoning_budget
        configure_reasoning_budget(base_body)
        slot = self_plan_id_slot()
        pair_cache = pair_cache_enabled()
        plan_body, _unused, _metadata = build_pair_requests(
            base_body, capture.self_plan_frame, "1. Check the current StateFrame.",
            id_slot=slot, cache=pair_cache)
        plan_capture = RequestCapture(
            capture.journal,
            capture.step,
            capture.snapshot_id,
            endpoint_profile=getattr(capture, "endpoint_profile", None),
        )
        from kernel_request_kind import bind_auxiliary
        bind_auxiliary(plan_capture, "self_plan")
        # Retain the shared prompt schema for cache-prefix accounting while
        # disabling workspace action generation in the planning side call.
        plan_body["tool_choice"] = "none"
        plan_choice = plan_capture.send(plan_body, API, KEY)
        plan = extract_plan(plan_choice)
        _unused_plan_body, self_plan_thinking_body, metadata = build_pair_requests(
            base_body, capture.self_plan_frame, plan, id_slot=slot, cache=pair_cache)
        raw_usage = plan_capture.accounting.get("raw_usage", {})
        raw_usage = raw_usage if isinstance(raw_usage, Mapping) else {}
        plan_tokens = raw_usage.get("completion_tokens")
        if not isinstance(plan_tokens, int) or plan_tokens < 0:
            plan_tokens = len(plan.split())
        capture.self_plan_plan_text = plan
        capture.self_plan_plan_tokens = plan_tokens
        capture.self_plan_plan_request_id = plan_capture.request_id
        capture.self_plan_plan_accounting = dict(plan_capture.accounting)
        capture.self_plan_pair_cache = pair_cache
        capture.stateframe_think_prefill_text = str(metadata["prefill_text"])
        capture.stateframe_think_prefill_tokens = int(metadata["prefill_tokens"])
        capture.stateframe_think_prefill_mode = "self_plan"
        capture.journal.emit(
            "think_prefill", request_id=capture.request_id,
            prefill_text=capture.stateframe_think_prefill_text,
            prefill_mode="self_plan",
            generated_token_exclusion="prompt_prefill",
            prefill_token_count=capture.stateframe_think_prefill_tokens,
            serialization="assistant_reasoning_content_continuation",
            continue_final_message=True,
            prefill_region_not_model_reasoning=True,
        )
        journal_plan(
            capture.journal, request_id=capture.request_id,
            plan_request_id=plan_capture.request_id, plan=plan,
            plan_tokens=plan_tokens, metadata={**metadata, "id_slot": slot,
                                                "cache_prompt": pair_cache,
                                                "plan_cache_n": plan_capture.accounting.get("cache_n"),
                                                "plan_prompt_n": plan_capture.accounting.get("prompt_n")})
    from kernel_pilot import POLICY, charge
    for attempt in range(POLICY.transport_attempts + int(timeout_retry_active)):
        if (timeout_retry_active and attempt >= POLICY.transport_attempts
                and timeout_retry_attempt != attempt):
            break
        charge("model_calls", "transport submission including internal retry")
        if self_plan_thinking_body is not None:
            body = copy.deepcopy(self_plan_thinking_body)
            body["max_tokens"] = mt
            body["temperature"] = TEMPERATURE if temp is None else temp
        else:
            body = {"model": MODEL, "messages": messages, "max_tokens": mt,
                    "temperature": TEMPERATURE if temp is None else temp,
                    "tools": tools}
            # STATEFRAME_TOKENLAB_CACHE_STABLE: the provider sees the run's fixed tool list; the
            # dispatch menu (`tools` / `_request_tools`) stays the authority for what may execute.
            _layout_tools = getattr(capture, "tokenlab_layout_tools", None) if capture is not None else None
            if (isinstance(_layout_tools, list) and _layout_tools) or (
                    isinstance(_layout_tools, list)
                    and os.environ.get("STATEFRAME_TOKENLAB_MENU_TOOL_LIST", "off").strip().lower() == "on"):
                body["tools"] = copy.deepcopy(_layout_tools)
            request_contract = getattr(capture, "request_action_contract", None) if capture is not None else None
            if isinstance(request_contract, Mapping):
                admitted_tools = request_contract.get("admitted_tool_schemas")
                if isinstance(admitted_tools, list):
                    body["tools"] = copy.deepcopy(admitted_tools)
        _apply_sampler_fields(body)
        from sampler_config import journal_sent as _journal_sampler_sent
        _journal_sampler_sent(capture, body, phase="turn", attempt=attempt)
        if (capture is not None
                and getattr(capture, "stateframe_think_prefill_mode", "off") in {"on", "self_plan"}
                and getattr(capture, "stateframe_think_prefill_text", "")):
            # llama.cpp's MiniCPM5 continuation branch renders the last
            # assistant reasoning_content inside an open <think>; this flag
            # is intentionally sent only by the opt-in true-prefill arm.
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False
        from kernel_step_contract import configure_reasoning_budget
        configure_reasoning_budget(body)
        if native_batches_mode and not action_schema_mode:
            body.update(tool_choice='auto', parallel_tool_calls=True)
        if capture is not None:
            from kernel_v43g_guard import forced_done_tool_choice, terminal_choice_mode
            if terminal_choice_mode() == "on":
                from kernel_completion_owner import is_reserved_request
                from kernel_pilot import ACTIVE_RUNTIME
                if is_reserved_request(
                        ACTIVE_RUNTIME.get(), body.get("messages"), body.get("tools")):
                    forced_choice = forced_done_tool_choice(body.get("tools"))
                    if forced_choice is not None:
                        body["tool_choice"] = forced_choice
        if "REACT_SEED" in os.environ:
            body["seed"] = int(os.environ["REACT_SEED"])
        if "REACT_SLOT_ID" in os.environ and self_plan_thinking_body is None:
            slot = int(os.environ["REACT_SLOT_ID"])
            if slot < 0:
                raise ValueError("REACT_SLOT_ID must be nonnegative")
            from kernel_prompt_cache import configure_pinned_slot
            configure_pinned_slot(body, slot)
        from kernel_step_contract import (configure, configure_action_envelope,
                                          decode, enabled as contract_enabled)
        with turn_timing_step("contracts"):
            if action_schema_mode:
                configure_action_envelope(body, tools, known=known_refs)
            elif contract_enabled():
                if capture is None:
                    raise ValueError("step contracts require a captured StateFrame binding")
                configure(body, tools, capture.snapshot_id)
        _ex = _cf.ThreadPoolExecutor(max_workers=1)
        fut = _ex.submit(_one_call, body, socket_timeout)
        try:
            future_timeout = (
                max(request_timeout_budget_seconds(), wall_timeout * 2)
                if frozen_timeout_retry_active else wall_timeout
            )
            choice = fut.result(timeout=future_timeout)
            if capture is not None:
                activate_turn_timing(capture.journal)
            if capture is not None and "_kernel_response_event_id" in choice:
                capture.selected_response_event_id = choice["_kernel_response_event_id"]
        except _cf.TimeoutError:
            _ex.shutdown(wait=False)
            from kernel_request_kind import reject_ambiguous_timeout
            reject_ambiguous_timeout(fut, capture=capture)
            if timeout_retry_active:
                if frozen_timeout_retry_active:
                    _record_provider_request_timeout(
                        capture, attempt=attempt + 1, timeout_seconds=future_timeout,
                        endpoint=API, retry_scheduled=False)
                    raise ProviderRequestTimeoutError(
                        attempts=attempt + 1, error_type="TimeoutError",
                        http_status=None, relay_error=None)
                retry_scheduled = timeout_retry_attempt is None
                _record_provider_request_timeout(
                    capture, attempt=attempt + 1, timeout_seconds=wall_timeout,
                    endpoint=API, retry_scheduled=retry_scheduled)
                if retry_scheduled:
                    timeout_retry_attempt = attempt + 1
                    continue
                raise ProviderRequestTimeoutError(
                    attempts=attempt + 1, error_type="TimeoutError",
                    http_status=None, relay_error=None)
            from kernel_pilot import ACTIVE_RUNTIME
            _runtime = ACTIVE_RUNTIME.get()
            if _runtime is not None and capture is not None:
                _runtime.harness_fault("transport_timeout", capture.request_id)
            if attempt == POLICY.transport_attempts - 1:
                raise TimeoutError("LLM call exceeded configured wall-clock allowance")
            continue
        except Exception as _te:                                 # noqa: BLE001
            _ex.shutdown(wait=False)
            from kernel_frame_request_allocation import FrameRequestAllocationError
            from kernel_frame_optional_allocation import FrameAllocationReplanNeeded
            from kernel_frame_allocation_capture import FrameFit5ImpossibleMinimum
            from kernel_completion_owner import RequestActionContractError
            from kernel_request_reprepare import RequestPreparationContractError
            from kernel_request_seal import RequestSealError
            from kernel_stateframe_framefit5 import FrameFit5Error
            from kernel_tool_schema_lint import ToolSchemaContractError
            from kernel_wire_truth_gate import WireTruthBlocked
            if isinstance(_te, (FrameRequestAllocationError, FrameAllocationReplanNeeded,
                                FrameFit5ImpossibleMinimum, RequestActionContractError,
                                RequestPreparationContractError, RequestSealError,
                                FrameFit5Error, ToolSchemaContractError, WireTruthBlocked)):
                raise
            if timeout_retry_active and is_provider_request_timeout_error(_te):
                if frozen_timeout_retry_active:
                    raise _te
                retry_scheduled = timeout_retry_attempt is None
                _record_provider_request_timeout(
                    capture, attempt=attempt + 1, timeout_seconds=wall_timeout,
                    endpoint=API, retry_scheduled=retry_scheduled)
                if retry_scheduled:
                    timeout_retry_attempt = attempt + 1
                    continue
                raise ProviderRequestTimeoutError(
                    attempts=attempt + 1, error_type="TimeoutError",
                    http_status=None, relay_error=None) from _te
            # Audit 2026-10-05 #1/#4: a harness contract error (schema the server
            # cannot compile, 4xx, context overflow) is terminal: no retry, no
            # /health polling, never booked against the model.
            _mark_http_contract(_te)
            if (getattr(_te, "boundary_recovery_stop", False)
                    or getattr(_te, "harness_contract", False)
                    or getattr(_te, "outcome", None) in _HARNESS_TERMINAL_OUTCOMES):
                raise
            if (getattr(_te, "long_horizon_terminal", False)
                    or getattr(_te, "long_horizon_reroutable", False)):
                raise
            if timeout_retry_active and timeout_retry_attempt == attempt:
                raise
            # E353: connection-class failures mean the SERVER died (pod OOM) — wait
            # for recovery instead of spending attempts on a dead endpoint.  An
            # HTTPError carries a real status from a LIVE server: never await it.
            if _is_connection_failure(_te):
                print(f"    [kernel-wiring] endpoint failure ({type(_te).__name__}) — "
                      f"awaiting recovery")
                if _await_endpoint():
                    continue                     # recovered: retry, attempt not wasted
            if attempt == POLICY.transport_attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
            continue
        _ex.shutdown(wait=False)
        m = choice["message"]
        raw_content = m.get("content") if isinstance(m, Mapping) else None
        content = raw_content.strip() if isinstance(raw_content, str) else ""
        reasoning_content = m.get("reasoning_content") if isinstance(m, Mapping) else ""
        if choice.get("finish_reason") == "length":
            if action_schema_mode:
                from kernel_step_contract import record_action_envelope_metrics
                record_action_envelope_metrics(
                    capture.journal if capture is not None else None,
                    step=capture.step if capture is not None else 0,
                    menu=tools, content=content, reasoning_content=reasoning_content,
                    outcome="length")
            next_max_tokens = min(mt * 2, output_ceiling)
            retry_available = attempt + 1 < POLICY.transport_attempts
            ceiling_exhausted = next_max_tokens <= mt
            if typed_truncation and ceiling_exhausted:
                # r45 rule 11d (kernel_response_truncation): at the ceiling a truncated completion is never parsed;
                # journal it and end typed on the FIRST truncation (a seeded identical retry repeats the loop).
                from kernel_response_truncation import EVENT, MODEL_NOTE, RESULT_CLASS
                truncations += 1
                if capture is not None:
                    capture.journal.emit(
                        EVENT, request_id=capture.request_id, step=capture.step, attempt=attempt,
                        truncation=truncations, max_tokens=mt, result_class=RESULT_CLASS,
                        retry_scheduled=False, dispatch=False)
                return MODEL_NOTE, []
            if skip_capped_length_retry and ceiling_exhausted:
                if capture is not None:
                    capture.journal.emit(
                        'model_retry', request_id=capture.request_id,
                        reason='response_length_limit_exhausted_ceiling', attempt=attempt,
                        previous_max_tokens=mt, retry_scheduled=False,
                        next_max_tokens=None, tool_dispatched=False)
                break
            if retry_available:
                from kernel_request_kind import reject_changed_body_retry
                reject_changed_body_retry(capture)
            if capture is not None:
                capture.journal.emit(
                    'model_retry', request_id=capture.request_id,
                    reason='response_length_limit', attempt=attempt,
                    previous_max_tokens=mt,
                    retry_scheduled=retry_available,
                    next_max_tokens=(next_max_tokens if retry_available else None),
                    tool_dispatched=False)
            mt = next_max_tokens
            continue
        if action_schema_mode:
            from kernel_step_contract import (decode_action_envelope,
                                              record_action_envelope_metrics)
            try:
                with turn_timing_step("contracts"):
                    call_id = ("action_" + capture.request_id
                               if capture is not None else "action_0")
                    calls = decode_action_envelope(
                        content, tools, known=known_refs, call_id=call_id)
            except (ValueError, TypeError) as error:
                if capture is not None:
                    capture.journal.emit(
                        "action_envelope_rejected", request_id=capture.request_id,
                        step=capture.step, reason=str(error), dispatch=False,
                        typed_rejection=True)
                record_action_envelope_metrics(
                    capture.journal if capture is not None else None,
                    step=capture.step if capture is not None else 0,
                    menu=tools, content=content, reasoning_content=reasoning_content,
                    schema_rejected=True, outcome="schema_rejected")
                return '[action-schema] Rejected proposal: ' + str(error), []
            arguments = json.loads(calls[0]["function"]["arguments"])
            record_action_envelope_metrics(
                capture.journal if capture is not None else None,
                step=capture.step if capture is not None else 0,
                menu=tools, content=content, reasoning_content=reasoning_content,
                action=calls[0]["function"]["name"], arguments=arguments)
            return "", calls
        if contract_enabled():
            try:
                with turn_timing_step("contracts"):
                    return "", decode(content, tools, capture.snapshot_id)
            except (ValueError, TypeError) as error:
                capture.journal.emit("step_contract_rejected", request_id=capture.request_id,
                                     reason=str(error), dispatch=False)
                return '[step-contract] Rejected proposal: ' + str(error), []
        calls = m.get("tool_calls") or []
        if not calls:
            calls = parse_dsml_tool_calls(content or (m.get("reasoning_content") or ""))
        if calls:
            from kernel_decision_cache import enabled as decision_cache_enabled
            if decision_cache_enabled() and isinstance(m, Mapping) and "memory" in m:
                calls = [copy.deepcopy(call) for call in calls]
                calls[0]["memory"] = copy.deepcopy(m.get("memory"))
        return content, calls
    if action_schema_mode:
        return '[action-schema] No complete action within the response limit.', []
    if os.environ.get('REACT_STEP_CONTRACT', '0') == '1':
        return '[step-contract] No complete action within the response limit. Choose one action and keep reasoning brief.', []
    return "", []


def _record_provider_request_timeout(capture, *, attempt: int, timeout_seconds: float,
                                    endpoint: str, retry_scheduled: bool) -> None:
    from kernel_measurement import journal_provider_request_timeout
    from kernel_pilot import ACTIVE_RUNTIME

    runtime = ACTIVE_RUNTIME.get()
    journal = getattr(capture, "journal", None) if capture is not None else None
    if journal is None:
        journal = getattr(runtime, "journal", None)
    request_id = getattr(capture, "request_id", None) if capture is not None else None
    if request_id is None:
        request_id = getattr(runtime, "request_id", None)
    step = getattr(capture, "step", None) if capture is not None else None
    if type(step) is not int:
        step = getattr(runtime, "current_step", None)
    if type(step) is not int:
        step = None
    journal_provider_request_timeout(
        journal, request_id=request_id, step=step,
        timeout_seconds=timeout_seconds, attempt=attempt,
        endpoint=endpoint, retry_scheduled=retry_scheduled)


def _send_turtle_request(measurement, body, endpoint, key):
    """Submit Turtle through a fresh capture so it is a separate model request."""
    from kernel_measurement import RequestCapture
    capture = RequestCapture(measurement.journal, getattr(measurement.owner, "current_step", 0),
                             measurement.snapshot_id,
                             endpoint_profile=getattr(measurement, "endpoint_profile", None))
    from kernel_request_kind import bind_auxiliary
    bind_auxiliary(capture, "turtle")
    return capture.send(body, endpoint, key), {
        "request_id": capture.request_id,
        "response_event_id": capture.response_event_id,
        "accounting": dict(capture.accounting),
    }


def _dispatch_decoy_call(bolt, messages: list[dict[str, object]], *,
                         tool_call_id: str, name: str,
                         arguments: Mapping[str, object],
                         decoy_kind: str | None = None) -> dict[str, object]:
    """Answer a decoy without touching the sandbox and leave receipt evidence."""
    result = decoy_response(name, arguments, tool_call_id, decoy_kind)
    bolt.receipt("tool", f"decoy {name}", json.dumps(dict(arguments), sort_keys=True),
                 "WRONG_TOOL", str(result["message"]))
    receipt_list = getattr(bolt, "_receipts", None) or getattr(bolt, "receipts", None)
    if decoy_kind == "near_duplicate" and receipt_list:
        receipt = receipt_list[-1]
        if isinstance(receipt, dict):
            receipt["decoy_kind"] = decoy_kind
            receipt["sibling_of"] = result.get("sibling_of")
    messages.append(_tool_reply_for_call(messages, tool_call_id, json.dumps(result, sort_keys=True)))
    return result


def ask(messages, mt=4000, temp=None):
    from kernel_measurement import (
        ACTIVE_REQUEST, ProviderRequestTimeoutError, activate_turn_timing,
        is_provider_request_timeout_error, per_request_timeout_seconds,
        request_timeout_budget_seconds, request_timeouts,
        skip_capped_length_retry_enabled, timeout_retry_frozen_enabled,
    )
    capture = ACTIVE_REQUEST.get()
    from kernel_pilot import ACTIVE_RUNTIME
    from kernel_request_kind import require_capture
    require_capture(capture, ACTIVE_RUNTIME.get())
    socket_timeout, wall_timeout = request_timeouts()
    timeout_retry_active = per_request_timeout_seconds() is not None
    frozen_timeout_retry_active = (
        capture is not None and timeout_retry_active and timeout_retry_frozen_enabled())
    timeout_retry_attempt: int | None = None
    skip_capped_length_retry = skip_capped_length_retry_enabled()
    # DEFECT #36 — our own rule 11d, unapplied in our own harness. mt was 1200 and
    # finish_reason was never inspected: any complete-function edit larger than ~1200 tokens
    # truncated BEFORE its closing ``` fence, the block regex matched nothing, and the loop
    # nudged "edit needs a ```python block" forever (measured: 18+ consecutive rejections on
    # _ack_decision, a ~100-line function). finish_reason=='length' is a FAILURE — never
    # parse a truncated completion; retry once with a doubled budget.
    def _one_call(body, call_timeout):
        if capture is not None:
            if frozen_timeout_retry_active:
                return capture.send_with_frozen_timeout_retry(body, API, KEY)
            if timeout_retry_active:
                return capture.send(body, API, KEY, transport_attempt_limit=1,
                                    journal_timeouts=False)
            return capture.send(body, API, KEY)
        req = urllib.request.Request(
            API, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req, timeout=call_timeout) as r:
            return json.loads(r.read())["choices"][0]

    from kernel_pilot import POLICY, charge
    for attempt in range(POLICY.transport_attempts + int(timeout_retry_active)):
        if (timeout_retry_active and attempt >= POLICY.transport_attempts
                and timeout_retry_attempt != attempt):
            break
        charge("model_calls", "transport submission including internal retry")
        body = {"model": MODEL, "messages": messages, "max_tokens": mt,
                "temperature": TEMPERATURE if temp is None else temp}
        _apply_sampler_fields(body)
        if (capture is not None
                and getattr(capture, "stateframe_think_prefill_mode", "off") == "on"
                and getattr(capture, "stateframe_think_prefill_text", "")):
            # Keep the no-tools path on the same llama.cpp chat-template
            # continuation contract as ask_tools.
            body["continue_final_message"] = True
            body["add_generation_prompt"] = False
        if "REACT_SEED" in os.environ:
            body["seed"] = int(os.environ["REACT_SEED"])
        # WALL-CLOCK BOUND (defect #38). urlopen's timeout is PER SOCKET OPERATION — a
        # half-dead connection that dribbles a byte every few minutes never trips it.
        # Measured live (v15): one request wedged in SSL read for 20+ minutes; the stack
        # dump showed poll() inside _ssl read with the process at 0% CPU. Run the call in a
        # worker thread with a hard overall deadline (rule #54's pattern); an abandoned
        # thread holds its socket until process exit, which a short-lived bench worker can
        # afford. A wall-clock timeout counts as a failed attempt and retries.
        import concurrent.futures as _cf
        # NO `with` BLOCK: the executor's __exit__ JOINS its threads, so a hung call would
        # hang the shutdown too — the exact wait we are escaping. shutdown(wait=False)
        # abandons the wedged thread instead.
        _ex = _cf.ThreadPoolExecutor(max_workers=1)
        fut = _ex.submit(_one_call, body, socket_timeout)
        try:
            future_timeout = (
                max(request_timeout_budget_seconds(), wall_timeout * 2)
                if frozen_timeout_retry_active else wall_timeout
            )
            choice = fut.result(timeout=future_timeout)
            if capture is not None:
                activate_turn_timing(capture.journal)
        except _cf.TimeoutError:
            _ex.shutdown(wait=False)
            from kernel_request_kind import reject_ambiguous_timeout
            reject_ambiguous_timeout(fut, capture=capture)
            if timeout_retry_active:
                if frozen_timeout_retry_active:
                    _record_provider_request_timeout(
                        capture, attempt=attempt + 1, timeout_seconds=future_timeout,
                        endpoint=API, retry_scheduled=False)
                    raise ProviderRequestTimeoutError(
                        attempts=attempt + 1, error_type="TimeoutError",
                        http_status=None, relay_error=None)
                retry_scheduled = timeout_retry_attempt is None
                _record_provider_request_timeout(
                    capture, attempt=attempt + 1, timeout_seconds=wall_timeout,
                    endpoint=API, retry_scheduled=retry_scheduled)
                if retry_scheduled:
                    timeout_retry_attempt = attempt + 1
                    continue
                raise ProviderRequestTimeoutError(
                    attempts=attempt + 1, error_type="TimeoutError",
                    http_status=None, relay_error=None)
            from kernel_pilot import ACTIVE_RUNTIME
            _runtime = ACTIVE_RUNTIME.get()
            if _runtime is not None and capture is not None:
                _runtime.harness_fault("transport_timeout", capture.request_id)
            if attempt == POLICY.transport_attempts - 1:
                raise TimeoutError("LLM call exceeded configured wall-clock allowance")
            continue
        except Exception as _te:                                  # noqa: BLE001
            _ex.shutdown(wait=False)
            if timeout_retry_active and is_provider_request_timeout_error(_te):
                if frozen_timeout_retry_active:
                    raise _te
                retry_scheduled = timeout_retry_attempt is None
                _record_provider_request_timeout(
                    capture, attempt=attempt + 1, timeout_seconds=wall_timeout,
                    endpoint=API, retry_scheduled=retry_scheduled)
                if retry_scheduled:
                    timeout_retry_attempt = attempt + 1
                    continue
                raise ProviderRequestTimeoutError(
                    attempts=attempt + 1, error_type="TimeoutError",
                    http_status=None, relay_error=None) from _te
            _mark_http_contract(_te)
            if (getattr(_te, "boundary_recovery_stop", False)
                    or getattr(_te, "harness_contract", False)
                    or getattr(_te, "outcome", None) in _HARNESS_TERMINAL_OUTCOMES):
                raise
            if (getattr(_te, "long_horizon_terminal", False)
                    or getattr(_te, "long_horizon_reroutable", False)):
                raise
            if frozen_timeout_retry_active:
                raise
            if timeout_retry_active and timeout_retry_attempt == attempt:
                raise
            # TRANSPORT ERRORS RETRY LIKE TIMEOUTS (v21: a raw SSLError at step 181 killed
            # the whole task as ERR — 3 hours of work charged to a flaky socket). Anything
            # the HTTP layer throws is attempt-scoped, not task-scoped.
            if attempt == POLICY.transport_attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
            continue
        _ex.shutdown(wait=False)
        m = choice["message"]
        text = (m.get("content") or m.get("reasoning_content") or "").strip()
        if choice.get("finish_reason") != "length":
            return text
        next_max_tokens = min(mt * 2, 16000)
        if (attempt + 1 < POLICY.transport_attempts
                and not (skip_capped_length_retry and next_max_tokens <= mt)):
            from kernel_request_kind import reject_changed_body_retry
            reject_changed_body_retry(capture)
        if skip_capped_length_retry:
            retry_available = attempt + 1 < POLICY.transport_attempts
            ceiling_exhausted = next_max_tokens <= mt
            if capture is not None:
                capture.journal.emit(
                    'model_retry', request_id=capture.request_id,
                    reason=('response_length_limit_exhausted_ceiling'
                            if ceiling_exhausted else 'response_length_limit'),
                    attempt=attempt, previous_max_tokens=mt,
                    retry_scheduled=retry_available and not ceiling_exhausted,
                    next_max_tokens=(next_max_tokens
                                     if retry_available and not ceiling_exhausted else None),
                    tool_dispatched=False)
            if ceiling_exhausted or not retry_available:
                return ""
        mt = next_max_tokens
    return text  # three truncations in a row: hand back what we have, the loop's nudge names it



def protected_test_path(path: pathlib.Path, baseline_tests: set[str]) -> bool:
    """Protect the original test resource through path and inode aliases."""
    canonical = path.resolve()
    for name in baseline_tests:
        protected = pathlib.Path(name)
        if canonical == protected.resolve():
            return True
        if path.exists() and protected.exists() and path.samefile(protected):
            return True
    return False


def _publish_nonpy_candidate(path: pathlib.Path, candidate: bytes, *, operation: str,
                             root: pathlib.Path | None, public_brief: object = None,
                             warning: str | None = None, prepare=None) -> str:
    """Apply an enabled non-Python candidate while preserving its exact bytes."""
    from kernel_tokenlab_nonpy_edits import emit
    candidate_text = candidate.decode("utf-8", errors="replace")
    from kernel_destructive_edit_guard import guard_write
    shrink_refusal = guard_write(
        path, candidate_text, root=root, public_brief=public_brief,
    )
    if shrink_refusal is not None:
        emit(path, operation=operation, outcome="REJECTED", reason=shrink_refusal)
        return shrink_refusal
    from kernel_green_lock import guard_product_mutation, note_applied_product_edit
    refusal = guard_product_mutation(path)
    if refusal is not None:
        result = f"{operation} REJECTED: {refusal}"
        emit(path, operation=operation, outcome="REJECTED", reason=refusal)
        return result
    applied, shadow = _effect_write_bytes(
        path, candidate, operation=operation, prepare=prepare,
    )
    if not applied:
        result = shadow or f"{operation} SHADOW: effect withheld"
        emit(path, operation=operation, outcome="SHADOW", reason=result)
        return result
    note_applied_product_edit(path)
    result = f"{operation} applied."
    if warning:
        result += "\n" + warning
    emit(path, operation=operation, outcome="APPLIED", warning=warning)
    return _post_edit_observation(path, result)


def _str_replace_nonpy_edit(path: pathlib.Path, old_str: str, new_str: str, *,
                            root: pathlib.Path | None = None,
                            public_brief: object = None) -> str:
    """Perform v21 exact-byte replacement plus the enabled file-type gate."""
    from kernel_tokenlab_nonpy_edits import emit, validate
    try:
        source = path.read_bytes()
    except OSError as error:
        return f"str_replace REJECTED: could not read {path.name}: {type(error).__name__}."
    old_bytes = old_str.encode("utf-8")
    new_bytes = new_str.encode("utf-8")
    count = source.count(old_bytes)
    if count == 0:
        if os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on":
            try:
                source_text = source.decode("utf-8")
            except UnicodeDecodeError:
                source_text = ""
            if source_text:
                from kernel_empirical_fixes import normalized_str_replace_candidate
                normalized_source, normalized_note = normalized_str_replace_candidate(
                    source_text, old_str, new_str)
                if normalized_note and normalized_source is None:
                    emit(path, operation="str_replace", outcome="REJECTED",
                         reason=normalized_note)
                    return normalized_note
                if normalized_source is not None:
                    candidate = normalized_source.encode("utf-8")
                    try:
                        warning = validate(normalized_source, path)
                    except ValueError as error:
                        reason = str(error)
                        emit(path, operation="str_replace", outcome="REJECTED", reason=reason)
                        return (f"str_replace REJECTED: {reason}. Nothing was changed.")
                    return _publish_nonpy_candidate(
                        path, candidate, operation="str_replace", root=root,
                        public_brief=public_brief, warning=warning,
                    )
        emit(path, operation="str_replace", outcome="REJECTED", reason="old_str not found")
        return ("str_replace REJECTED: old_str not found. Copy the exact current text "
                "(check whitespace) — `read` the region first.")
    if count > 1:
        emit(path, operation="str_replace", outcome="REJECTED",
             reason=f"old_str matches {count} places")
        return (f"str_replace REJECTED: old_str matches {count} places. Include more "
                f"surrounding lines so it is unique.")
    candidate = source.replace(old_bytes, new_bytes, 1)
    candidate_text = candidate.decode("utf-8", errors="replace")
    try:
        warning = validate(candidate_text, path)
    except ValueError as error:
        reason = str(error)
        emit(path, operation="str_replace", outcome="REJECTED", reason=reason)
        return f"str_replace REJECTED: {reason}. Nothing was changed."
    return _publish_nonpy_candidate(
        path, candidate, operation="str_replace", root=root,
        public_brief=public_brief, warning=warning,
    )


def str_replace_edit(path: pathlib.Path, old_str: str, new_str: str, *,
                     target_ref: object = None, root: pathlib.Path | None = None,
                     public_brief: object = None, journal: object = None) -> str:
    """MC coding.py's _code_str_replace contract, transplanted:
    exactly-one match required; whole-file syntax check; revert on breakage."""
    from kernel_edit_guard import edit_guard_mode
    edit_guard_mode()
    if root is not None:
        refusal = _bound_edit_refusal(root, path, target_ref)
        if refusal is not None:
            return "str_replace REJECTED: " + refusal
    if not path.exists():
        if _workspace_truth.enabled():
            fact = _workspace_truth.exists(root, str(path), journal=journal)
            return "str_replace REJECTED: " + _workspace_truth.render_fact(fact)
        return f"str_replace REJECTED: {path.name} does not exist."
    if path.is_dir():
        return f"str_replace REJECTED: {path.name} is a directory, not a file."
    if not old_str:
        # Bughunt r2AE #5: an empty/missing old_str matched "every position" and produced
        # a false "matches N places" repair loop. It is a schema error, not an ambiguity.
        return ("str_replace REJECTED: old_str is required and must be non-empty: copy the "
                "exact current text to replace. Nothing was changed.")
    if old_str == new_str:
        # ack s57 (E330 lineage): a no-op means the model believes the CURRENT text is
        # already the fix — redirect that belief instead of just rejecting the edit.
        return ("str_replace NO-OP: old_str and new_str are identical — nothing would "
                "change. You just re-emitted the CURRENT code, which means you believe "
                "it is already correct. If so, the defect is NOT on this line: check "
                "the CALLEE's signature/behaviour (does it use every kwarg passed?), a "
                "different call site, or the code path that runs when this one doesn't.")
    if _nonpy_edit_applies(path):
        return _str_replace_nonpy_edit(
            path, old_str, new_str, root=root, public_brief=public_brief,
        )
    # Bughunt r2AE #1: strict read_text() raised UnicodeDecodeError out of the tool
    # handler and ended the run. Decode explicitly and return a typed rejection.
    src, decode_refusal = _read_edit_source(path, "str_replace")
    if decode_refusal is not None:
        return decode_refusal
    n = src.count(old_str)
    if n == 0:
        if os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on":
            from kernel_empirical_fixes import normalized_str_replace_candidate
            normalized_source, normalized_note = normalized_str_replace_candidate(
                src, old_str, new_str)
            if normalized_note and normalized_source is None:
                return normalized_note
            if normalized_source is not None:
                new_src = normalized_source
                try:
                    ast.parse(new_src)
                except SyntaxError as e:
                    return (f"str_replace REJECTED: the file would not parse afterwards "
                            f"({e.msg} line {e.lineno}). Nothing was changed.")
                from kernel_destructive_edit_guard import guard_write
                shrink_refusal = guard_write(
                    path, new_src, root=root, public_brief=public_brief,
                )
                if shrink_refusal is not None:
                    return shrink_refusal
                from kernel_green_lock import guard_product_mutation, note_applied_product_edit
                refusal = guard_product_mutation(path)
                if refusal is not None:
                    return "str_replace REJECTED: " + refusal
                applied, shadow = _effect_write(path, new_src)
                if not applied:
                    return shadow or "str_replace SHADOW: effect withheld"
                note_applied_product_edit(path)
                return _post_edit_observation(
                    path, "str_replace applied. [normalized leading-indent match]")
        return ("str_replace REJECTED: old_str not found. Copy the exact current text "
                "(check whitespace) — `read` the region first.")
    if n > 1:
        return (f"str_replace REJECTED: old_str matches {n} places. Include more "
                f"surrounding lines so it is unique.")
    new_src = src.replace(old_str, new_str, 1)
    try:
        ast.parse(new_src)
    except SyntaxError as e:
        return (f"str_replace REJECTED: the file would not parse afterwards "
                f"({e.msg} line {e.lineno}). Nothing was changed.")
    from kernel_destructive_edit_guard import guard_write
    shrink_refusal = guard_write(
        path, new_src, root=root, public_brief=public_brief,
    )
    if shrink_refusal is not None:
        return shrink_refusal
    from kernel_green_lock import guard_product_mutation, note_applied_product_edit
    refusal = guard_product_mutation(path)
    if refusal is not None:
        return "str_replace REJECTED: " + refusal
    applied, shadow = _effect_write(path, new_src)
    if not applied:
        return shadow or "str_replace SHADOW: effect withheld"
    note_applied_product_edit(path)
    return _post_edit_observation(path, "str_replace applied.")


def replace_module(path: pathlib.Path, expected_revision: str, source: str | None = None,
                   candidate_id: str | None = None, *, root: pathlib.Path | None = None,
                   protected_paths=(), public_brief: object = None) -> str:
    """Public wrapper for the JobQueue completion revision-checked complete-file write."""
    from kernel_jq_finish import replace_module as _replace_module

    def dispatch():
        result = _replace_module(path, expected_revision, source=source,
                                 candidate_id=candidate_id, root=root,
                                 protected_paths=protected_paths,
                                 public_brief=public_brief)
        if " applied." in result:
            from kernel_green_lock import note_applied_product_edit
            note_applied_product_edit(path)
        return result

    if not _effect_boundary_enabled():
        return dispatch()
    before = path.read_text(errors="ignore") if path.exists() else "<absent>"
    _applied, value, rendered = _effect_application(
        "edit",
        {"path": str(path), "before": before, "after": source or candidate_id or "<candidate>"},
        dispatch,
        shadow_value=None,
        before_snapshot=lambda: {"path": str(path), "content": before},
        effect_applied=lambda result: isinstance(result, str) and " applied." in result,
        after_snapshot=lambda: {
            "path": str(path),
            "content": path.read_text(errors="ignore") if path.exists() else "<absent>",
        },
    )
    if _applied:
        return str(value)
    return str(value or rendered or "edit SHADOW: effect withheld")


def replace_fn(path: pathlib.Path, name: str, new_src: str,
               *, multi_file: bool | None = None, target_ref: object = None,
               root: pathlib.Path | None = None, public_brief: object = None,
               journal: object = None) -> str:
    """Swap one top-level function or class's source in place."""
    from kernel_edit_guard import edit_guard_mode
    edit_guard_mode()
    hardening = _edit_hardening_enabled()
    if hardening:
        original_source = new_src
        new_src = textwrap.dedent(new_src)
        _emit_edit_hardening(path=str(path), target=str(name), dedented=(new_src != original_source),
                             class_scope_resolved=False, multi_definition_add=False)
    if multi_file is None:
        multi_file = "." in name
    if path.is_dir():
        return f"edit REJECTED: {path.name} is a directory, not a file."
    if _nonpy_edit_applies(path):
        return _replace_file(path, new_src, root=root, public_brief=public_brief,
                             journal=journal)
    try:
        txt = path.read_bytes().decode("utf-8")
        tree = ast.parse(txt)
    except Exception as e:                                      # noqa: BLE001
        return f"edit failed: cannot parse {path.name}: {type(e).__name__}"
    if multi_file and "." not in name:
        # In a declared multi-file edit the submitted definition is the target binding.
        # Keep qualified method edits on their explicit class.method route, but do not let
        # the task's entry label redirect a top-level definition to another file's symbol.
        payload_name, _payload_error = _top_level_definition_name(new_src)
        if payload_name is not None:
            name = payload_name
    if hardening:
        from kernel_empirical_fixes import unique_method_target
        top_level_names = {node.name for node in tree.body
                           if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                                ast.AsyncFunctionDef))}
        qualified, target_error = (unique_method_target(txt, name)
                                   if name not in top_level_names else (None, None))
        if target_error is not None:
            return target_error
        if qualified is not None and qualified != name:
            name = qualified
            multi_file = True
            _emit_edit_hardening(path=str(path), target=str(name), dedented=True,
                                 class_scope_resolved=True, multi_definition_add=False)
        else:
            try:
                payload_tree = ast.parse(textwrap.dedent(new_src))
                payload_defs = [node for node in payload_tree.body
                                if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                                      ast.AsyncFunctionDef))]
                target_definitions = dict(_qualified_definitions_for_repair(tree))
                target = target_definitions.get(name)
                if (isinstance(target, ast.ClassDef) and len(payload_defs) == 1
                        and isinstance(payload_defs[0], (ast.FunctionDef, ast.AsyncFunctionDef))):
                    qualified, target_error = unique_method_target(txt, payload_defs[0].name)
                    if target_error is not None:
                        return target_error
                    if qualified is not None:
                        name = qualified
                        multi_file = True
                        _emit_edit_hardening(path=str(path), target=str(name), dedented=True,
                                             class_scope_resolved=True, multi_definition_add=False)
            except SyntaxError:
                pass
    if root is not None:
        refusal = _bound_edit_refusal(root, path, target_ref)
        if refusal is not None:
            return "edit REJECTED: " + refusal
    from kernel_jq_class_edit import enabled as jq_class_edit_enabled, prepare_class_edit
    if jq_class_edit_enabled():
        class_plan = prepare_class_edit(
            txt, name, new_src, path=path, root=root,
        )
        if class_plan.handled:
            if class_plan.rejection is not None:
                return class_plan.rejection
            candidate = class_plan.candidate
            if candidate is None:
                return _typed_edit_rejection(
                    "INVALID_CLASS_RESULT", path, root=root,
                    hint=f"class edit for {name} did not produce a candidate",
                    symbol=name,
                )
            normalized_candidate, normalization_refusal = _normalize_candidate_for_publication(
                path, new_src.encode("utf-8"), mode="complete-definition", root=root,
                target=name, target_ref=target_ref,
                composed_candidate=candidate.encode("utf-8"),
            )
            if normalization_refusal is not None:
                return normalization_refusal
            assert normalized_candidate is not None
            try:
                candidate = normalized_candidate.decode("utf-8")
            except UnicodeDecodeError:
                return "edit REJECTED: normalized candidate is not valid UTF-8"
            if candidate.strip() == txt.strip():
                return "edit NO-OP: your replacement is IDENTICAL to the current source"
            from kernel_destructive_edit_guard import guard_write
            shrink_refusal = guard_write(
                path, candidate, root=root, public_brief=public_brief,
            )
            if shrink_refusal is not None:
                return shrink_refusal
            from kernel_green_lock import guard_product_mutation, note_applied_product_edit
            refusal = guard_product_mutation(path)
            if refusal is not None:
                return "edit REJECTED: " + refusal
            applied, shadow = _effect_write(path, candidate, operation="edit", encoding="utf-8")
            if not applied:
                return shadow or "edit SHADOW: effect withheld"
            note_applied_product_edit(path)
            # r44 staging (attribution): the class-edit path is only reachable with STATEFRAME_JQ_CLASS_EDIT on;
            # journal that it acted.  Best-effort, never changes the tool result.
            try:
                from kernel_pilot import ACTIVE_RUNTIME
                _jq_journal = getattr(ACTIVE_RUNTIME.get(), "journal", None)
            except (ImportError, LookupError):
                _jq_journal = None
            if _jq_journal is not None and callable(getattr(_jq_journal, "emit", None)):
                try:
                    _jq_journal.emit("jq_class_edit_applied", producer="kernel_jq_class_edit",
                                     path=str(path), symbol=str(name), dispatch=False)
                except (TypeError, ValueError, OSError):
                    pass   # attribution only: the edit is already applied and reported
            return _post_edit_observation(path, "edit applied.")
    if (os.environ.get("STATEFRAME_BOUNDARY_CHECKS", "off").lower() == "on"
            or _edit_hardening_enabled()):
        multi_result = _apply_multi_definition_edit(
            path, txt, new_src, root=root, public_brief=public_brief,
        )
        if multi_result is not None:
            return multi_result
    if os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on":
        from kernel_empirical_fixes import unique_method_target
        top_level_names = {node.name for node in tree.body
                           if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                                ast.AsyncFunctionDef))}
        qualified, target_error = (unique_method_target(txt, name)
                                   if "." not in name and name not in top_level_names
                                   else (name, None))
        if target_error is not None:
            return target_error
        if qualified is not None and qualified != name:
            name = qualified
            multi_file = True
        else:
            # Some native edit payloads carry a method body while the task row
            # names the owning class.  Resolve the payload's sole method before
            # the legacy class splice can turn that body into a nested class.
            try:
                payload_tree = ast.parse(textwrap.dedent(new_src))
                payload_defs = [node for node in payload_tree.body
                                if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                                     ast.AsyncFunctionDef))]
                target_definitions = dict(_qualified_definitions_for_repair(tree))
                target_is_class = isinstance(
                    target_definitions.get(name), ast.ClassDef)
                if (target_is_class and len(payload_defs) == 1 and isinstance(
                        payload_defs[0], (ast.FunctionDef, ast.AsyncFunctionDef))):
                    qualified, target_error = unique_method_target(
                        txt, payload_defs[0].name)
                    if target_error is not None:
                        return target_error
                    if qualified is not None:
                        name = qualified
                        multi_file = True
                elif (target_is_class and len(payload_defs) >= 1
                      and any(isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom))
                              for node in payload_tree.body[:1])
                      and any(isinstance(node, ast.ClassDef) and node.name == name
                              for node in payload_defs)):
                    # A complete module payload is valid for the explicit
                    # whole-file-shaped class mistake; route it through the
                    # no-clobber file contract instead of replacing the class.
                    return _replace_file(
                        path, textwrap.dedent(new_src), root=root,
                        public_brief=public_brief, journal=journal,
                    )
            except SyntaxError:
                pass
    if os.environ.get("STATEFRAME_JQ_FINISH", "off") == "on":
        # Native edit carries the replacement payload but historically only the
        # task symbol was passed as ``name``.  For a complete class payload the
        # AST definition is the authoritative scoped target (Job, JobQueue, or
        # QueueStats); never splice it into the task class by accident.
        try:
            payload_tree = ast.parse(textwrap.dedent(new_src))
            payload_defs = [node for node in payload_tree.body
                            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
            source_names = {qualified for qualified, _node in
                            _qualified_definitions_for_repair(tree)}
            if len(payload_defs) == 1 and isinstance(payload_defs[0], ast.ClassDef):
                payload_name = payload_defs[0].name
                if payload_name in source_names and payload_name != name:
                    name = payload_name
                    multi_file = False
        except SyntaxError:
            pass
    # JobQueue completion uses one direct span operation for the enabled JobQueue finish
    # path.  This keeps class and qualified-method edits on the same semantics
    # and prevents a class payload from ever falling through to a file splice.
    if os.environ.get("STATEFRAME_JQ_FINISH", "off") == "on":
        from kernel_jq_finish import (_replace_qualified_source,
                                      validate_edit_payload, validate_source)
        replacement = textwrap.dedent(new_src)
        rejection = validate_edit_payload(txt, name, replacement)
        if rejection is not None:
            # A target may be absent only after an earlier rejected operation;
            # keep the ordinary diagnostic in that case.
            if name not in {qualified for qualified, _node in
                            _qualified_definitions_for_repair(tree)}:
                return _typed_edit_rejection(
                    "NO_SUCH_DEFINITION", path, root=root,
                    hint=f"no function named {name} in {path.name}", symbol=name)
            return rejection
        candidate = _replace_qualified_source(txt, tree, name, replacement)
        if candidate is None:
            return _typed_edit_rejection(
                "NO_SUCH_DEFINITION", path, root=root,
                hint=f"no function named {name} in {path.name}", symbol=name)
        normalized_candidate, normalization_refusal = _normalize_candidate_for_publication(
            path, replacement.encode("utf-8"), mode="complete-definition", root=root,
            target=name, target_ref=target_ref,
            composed_candidate=candidate.encode("utf-8"),
        )
        if normalization_refusal is not None:
            return normalization_refusal
        assert normalized_candidate is not None
        try:
            candidate = normalized_candidate.decode("utf-8")
        except UnicodeDecodeError:
            return "edit REJECTED: normalized candidate is not valid UTF-8"
        try:
            validate_source(candidate, filename=str(path))
        except ValueError as exc:
            return f"edit REJECTED: {exc}"
        if candidate.strip() == txt.strip():
            return "edit NO-OP: your replacement is IDENTICAL to the current source"
        from kernel_destructive_edit_guard import guard_write
        shrink_refusal = guard_write(
            path, candidate, root=root, public_brief=public_brief,
        )
        if shrink_refusal is not None:
            return shrink_refusal
        from kernel_green_lock import guard_product_mutation, note_applied_product_edit
        refusal = guard_product_mutation(path)
        if refusal is not None:
            return "edit REJECTED: " + refusal
        applied, shadow = _effect_write(path, candidate, encoding="utf-8")
        if not applied:
            return shadow or "edit SHADOW: effect withheld"
        note_applied_product_edit(path)
        return _post_edit_observation(path, "edit applied.")
    if not multi_file and "." not in name:
        from kernel_edit_guard import edit_rejection
        rejection = edit_rejection(txt, name, new_src)
        if rejection is not None:
            return rejection
    _qualified: dict[int, str] = {}

    def collect_qualified(node: ast.AST, prefix: tuple[str, ...] = ()) -> None:
        current = prefix
        if isinstance(node, ast.ClassDef):
            current = (*prefix, node.name)
            _qualified[id(node)] = ".".join(current)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current = (*prefix, node.name)
            _qualified[id(node)] = ".".join(current)
        for child in ast.iter_child_nodes(node):
            collect_qualified(child, current)

    if multi_file:
        collect_qualified(tree)
    for n in ast.walk(tree):
        is_function = isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        is_class = isinstance(n, ast.ClassDef)
        is_top_level = getattr(n, "col_offset", 1) == 0
        target_match = ((is_function or is_class) and
                        ((not multi_file and is_top_level and n.name == name) or
                         (multi_file and _qualified.get(id(n)) == name)))
        if target_match:
            target_kind = "class" if is_class else "function"
            if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
                from kernel_frame_edit_integrity import definition_source
                old = definition_source(txt, n)
            else:
                old = ast.get_source_segment(txt, n)
            if not old:
                if target_kind == "class":
                    return "edit failed: could not locate the class body"
                return "edit failed: could not locate the function body"
            replacement = textwrap.dedent(new_src) if multi_file else new_src
            if multi_file and not replacement.strip():
                return (f"edit REJECTED: the replacement for `{name}` is empty. "
                        "Send the COMPLETE function in the ```python block.")
            try:
                ast.parse(replacement)
            except SyntaxError as e:
                # NAME THE CONTRACT IN THE REJECTION (v18, s35-39: five rejected fragments in
                # a row — "unexpected indent line 2" means the block began mid-function, and
                # the bare parse error never told the model what shape was expected).
                if target_kind != "class":
                    return (f"edit REJECTED: your replacement does not parse ({e.msg} line "
                            f"{e.lineno}). The ```python block must be the COMPLETE function — "
                            f"first line `def {name}(...)` (or `async def`), plain ASCII code "
                            f"only, no prose or arrows inside the block.")
                return (f"edit REJECTED: your replacement does not parse ({e.msg} line "
                        f"{e.lineno}). The ```python block must be the COMPLETE class — "
                        f"first line `class {name}`; plain ASCII code "
                        f"only, no prose or arrows inside the block.")
            # E307 — THE NO-OP EDIT CONTRACT: the 4B "fixed"
            # require_isolated_f2_database by resubmitting the UNCHANGED function; the tool
            # said "edit applied", the count incremented, and the model concluded success.
            # An edit that changes nothing is not an edit — say so, deterministically.
            # AST source segments begin at ``def`` (without the owning class's
            # indentation).  Reapply that indentation to every replacement line
            # after the first so qualified class-method edits remain parseable.
            base_indent = " " * getattr(n, "col_offset", 0) if multi_file else ""
            if base_indent:
                _replacement_lines = replacement.splitlines()
                replacement = (_replacement_lines[0] + "\n" +
                               "\n".join(base_indent + line
                                          if line.strip() else line
                                          for line in _replacement_lines[1:]))
            if os.environ.get("STATEFRAME_JQ_FINISH", "off") == "on":
                try:
                    from kernel_jq_finish import validate_edit_payload
                    rejection = validate_edit_payload(txt, name, replacement)
                except (ImportError, TypeError, ValueError) as exc:
                    rejection = f"edit REJECTED: validator unavailable ({type(exc).__name__}: {exc})"
                if rejection is not None:
                    return rejection
            if replacement.strip() == (old or "").strip():
                return ("edit NO-OP: your replacement is IDENTICAL to the current source — "
                        "nothing changed. The bug is still present; the fix requires a "
                        "DIFFERENT expression somewhere in this function.")
            candidate = txt.replace(old, replacement, 1)
            normalized_candidate, normalization_refusal = _normalize_candidate_for_publication(
                path, replacement.encode("utf-8"), mode="complete-definition", root=root,
                target=name, target_ref=target_ref,
                composed_candidate=candidate.encode("utf-8"),
            )
            if normalization_refusal is not None:
                return normalization_refusal
            assert normalized_candidate is not None
            try:
                candidate = normalized_candidate.decode("utf-8")
            except UnicodeDecodeError:
                return "edit REJECTED: normalized candidate is not valid UTF-8"
            # VALIDATE THE RESULT, NOT JUST THE PART (E290's cause): a function that parses
            # alone can still break the FILE when spliced (indentation context, `old`
            # matching inside a string). Never write a file that does not parse.
            try:
                ast.parse(candidate)
            except SyntaxError as e:
                return (f"edit REJECTED: the file would not parse after this replacement "
                        f"({e.msg} line {e.lineno}). Nothing was changed — check the "
                        f"indentation of your function relative to the file.")
            from kernel_destructive_edit_guard import guard_write
            shrink_refusal = guard_write(
                path, candidate, root=root, public_brief=public_brief,
            )
            if shrink_refusal is not None:
                return shrink_refusal
            if os.environ.get("STATEFRAME_FRAME_EDIT_INTEGRITY", "off") == "on":
                from kernel_frame_edit_integrity import duplicate_decorator_rejection
                rejection = duplicate_decorator_rejection(txt, candidate, file=path.name)
                if rejection is not None:
                    return rejection
            from kernel_green_lock import guard_product_mutation, note_applied_product_edit
            refusal = guard_product_mutation(path)
            if refusal is not None:
                return "edit REJECTED: " + refusal
            applied, shadow = _effect_write(path, candidate)
            if not applied:
                return shadow or "edit SHADOW: effect withheld"
            note_applied_product_edit(path)
            return _post_edit_observation(path, "edit applied.")
    return _typed_edit_rejection(
        "NO_SUCH_DEFINITION", path, root=root,
        hint=f"no function named {name} in {path.name}", symbol=name)


def _qualified_definitions_for_repair(tree: ast.Module):
    found = []
    def visit(body, prefix=()):
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*prefix, node.name))
                found.append((qualified, node))
                visit(node.body, (*prefix, node.name))
    visit(tree.body)
    return found


def _db_alive(port: int = 55432, timeout: float = 1.5) -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_db() -> bool:
    """E338 per-invocation liveness guard: the shared mission-control-db has died
    mid-round TWICE (E314, then again at 01:25Z under external stack restarts),
    converting real model progress into contaminated verdicts. Probe before every
    pytest; on failure attempt ONE bench-local recovery (the exact manual action
    performed both times) and wait bounded for recovery."""
    if _db_alive():
        return True
    print("    [kernel-wiring] mission-control-db unreachable — attempting recovery")
    try:
        subprocess.run(["docker", "start", "mission-control-db"],
                       capture_output=True, timeout=30)
    except Exception:                                                # noqa: BLE001
        pass
    for _ in range(45):
        time.sleep(2)
        if _db_alive():
            print("    [kernel-wiring] mission-control-db recovered")
            return True
    return False


def _effect_boundary_enabled() -> bool:
    """Cheap off-path check; do not import or snapshot when the arm is disabled."""
    return os.environ.get("STATEFRAME_EFFECT_BOUNDARY", "off").strip().lower() not in {
        "", "off", "0", "false", "no",
    }


def _effect_boundary_mode() -> str:
    """Return this lane's mode without importing its adapter on the OFF path."""
    if not _effect_boundary_enabled():
        return "off"
    from kernel_effect_application import mode
    return mode().value


def _effect_boundary_shadow_completion(runtime: object, rendered: str | None):
    """Build a typed non-sealing result when SHADOW withholds accepted done."""
    from kernel_verdict import CompletionDecision, REFUSED
    detail = rendered or "[EFFECT_BOUNDARY] submit SHADOW: completion withheld"
    return CompletionDecision(
        REFUSED,
        runtime._completion_basis(),
        reason_codes=("EFFECT_BOUNDARY_SHADOW",),
        detail=detail,
        artifact_preserved=True,
    )


def _effect_boundary_done_only(runtime: object) -> bool:
    """Whether the current host menu has no executable escape besides done."""
    menu = tuple(str(item) for item in (getattr(runtime, "completion_menu", ()) or ()))
    return bool(menu) and set(menu) == {"done"}


def _effect_application(operation: str, snapshot: object, dispatch, *,
                        owner: object | None = None, shadow_value: object = None,
                        effect_applied=None, before_snapshot=None, after_snapshot=None):
    """Call one real runner effect through R-012 while preserving the old off path."""
    if not _effect_boundary_enabled():
        return True, dispatch(), None
    from kernel_effect_application import apply_effect
    from kernel_pilot import ACTIVE_RUNTIME
    result = apply_effect(
        owner=owner if owner is not None else ACTIVE_RUNTIME.get(),
        operation=operation, snapshot=snapshot,
        evaluate=lambda frozen: frozen, dispatch=dispatch,
        before_snapshot=before_snapshot, after_snapshot=after_snapshot,
        shadow_value=shadow_value,
        effect_applied=effect_applied,
    )
    return result.executed, result.value, result.rendered


def _effect_write(path: pathlib.Path, source: str, *, operation: str = "edit",
                  encoding: str | None = None, prepare=None) -> tuple[bool, str | None]:
    """Publish a candidate source, returning ``(applied, shadow_observation)``."""
    def write():
        if prepare is not None:
            prepare()
        if encoding is None:
            return path.write_text(source)
        return path.write_text(source, encoding=encoding)

    if not _effect_boundary_enabled():
        write()
        return True, None
    before = path.read_text(errors="ignore") if path.exists() else "<absent>"
    applied, _value, rendered = _effect_application(
        operation,
        {"path": str(path), "before": before, "after": source},
        write,
        shadow_value=None,
        before_snapshot=lambda: {"path": str(path), "content": before},
        after_snapshot=lambda: {
            "path": str(path),
            "content": path.read_text(errors="ignore") if path.exists() else "<absent>",
        },
    )
    return applied, rendered


def _effect_write_bytes(path: pathlib.Path, source: bytes, *, operation: str = "edit",
                        prepare=None) -> tuple[bool, str | None]:
    """Publish an opaque candidate without newline or encoding normalization."""
    def write():
        if prepare is not None:
            prepare()
        return path.write_bytes(source)

    if not _effect_boundary_enabled():
        write()
        return True, None
    before = path.read_bytes() if path.exists() else b"<absent>"
    applied, _value, rendered = _effect_application(
        operation,
        {"path": str(path), "before": before.decode("utf-8", errors="replace"),
         "after": source.decode("utf-8", errors="replace")},
        write,
        shadow_value=None,
        before_snapshot=lambda: {
            "path": str(path), "content": before.decode("utf-8", errors="replace"),
        },
        after_snapshot=lambda: {
            "path": str(path),
            "content": (path.read_bytes().decode("utf-8", errors="replace")
                        if path.exists() else "<absent>"),
        },
    )
    return applied, rendered


def _effect_test_subprocess(command, *, root: pathlib.Path, targets, env, runner=None):
    """Run one test child through the caller-supplied central execution policy.

    r45 merge: ``runner`` is the caller's check runner (staging's supervised-executor
    aware ``_run_check``); the effect boundary wraps it rather than replacing it.
    A missing runner is a refusal, never permission to start a raw child.
    """
    def dispatch():
        if runner is None:
            raise RuntimeError("central execution policy runner is required")
        return runner(command)
    if not _effect_boundary_enabled():
        return dispatch(), None
    _applied, value, rendered = _effect_application(
        "test",
        {"command": tuple(str(item) for item in command),
         "targets": tuple(str(item) for item in targets)},
        dispatch,
        shadow_value=None,
    )
    return (value if _applied else None), (None if _applied else rendered)


def _compose_test_command(command: Sequence[object], targets: Sequence[object]) -> tuple[str, ...]:
    """Append selected test targets without depending on list/tuple command shape."""
    return tuple(str(item) for item in (*command, *targets))


def _discover_test_targets(root: pathlib.Path, needle: str) -> list[str]:
    """Discover tests without executing a model-controlled grep command."""
    found: list[str] = []
    try:
        paths = sorted(path for path in root.rglob("test_*.py")
                       if path.is_file() and not path.is_symlink())
    except (OSError, RuntimeError):
        return found
    for path in paths[:2048]:
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            if needle in path.read_text(encoding="utf-8", errors="replace"):
                found.append(str(path))
        except (OSError, UnicodeError):
            continue
    return found


def _nearest_pytest_config_args(root, targets) -> list[str]:
    """Return ["-c", cfg, "--rootdir", dir] for the nearest project pytest config of the targets.

    Real repositories keep pytest config in a subproject (e.g. backend/pytest.ini sets
    pythonpath); running from the repo root without it makes conftest imports fail
    (pytest exit 4). 10-03: every AHA bug-replay check died this way. This mirrors
    `cd backend && pytest`: walk up from each target to `root`, first config wins.
    Returns [] when the repo root already has a config, targets disagree, or none found.
    """
    root = pathlib.Path(root).resolve()

    def _has_pytest_config(directory: pathlib.Path) -> pathlib.Path | None:
        ini = directory / "pytest.ini"
        if ini.is_file():
            return ini
        pyproject = directory / "pyproject.toml"
        try:
            if pyproject.is_file() and "[tool.pytest.ini_options]" in pyproject.read_text(
                    encoding="utf-8", errors="replace"):
                return pyproject
        except OSError:
            return None
        return None

    if _has_pytest_config(root) is not None:
        return []
    found: set[pathlib.Path] = set()
    for target in targets or ():
        rel = str(target).split("::", 1)[0]
        if not rel:
            continue
        current = (root / rel).resolve().parent
        try:
            current.relative_to(root)
        except ValueError:
            continue
        while current != root:
            config = _has_pytest_config(current)
            if config is not None:
                found.add(config)
                break
            current = current.parent
    if len(found) != 1:
        return []
    config = found.pop()
    return ["-c", str(config), "--rootdir", str(config.parent)]


def _classify_runner_failure(blob: str, fn: str = "", rel_hint: str | None = None) -> str:
    """Classify runner output without letting an empty target match every line."""
    text = str(blob or "")
    lower = text.lower()
    if (re.search(r"\bUNSUPPORTED(?:_EXECUTION|_ADAPTER)?\b", text, re.IGNORECASE)
            or "no execution adapter" in lower
            or "unsupported language" in lower):
        return "UNSUPPORTED"
    ours = ("conftest" in text or "import_plugin" in text or "_pytest/config" in text)
    theirs = bool(str(fn or "").strip() and str(fn) in text) \
        or bool(str(rel_hint or "").strip() and str(rel_hint) in text)
    import_failure = "ImportError" in text or "ModuleNotFoundError" in text
    if import_failure and ours and not theirs:
        return "ENVIRONMENT"
    if import_failure and theirs:
        return "PRODUCT"
    if "pytest" in lower and ("INTERNALERROR" in text or "usage:" in lower):
        return "RUNNER"
    if "no tests ran" in lower or "no tests collected" in lower:
        return "SMOKE"
    return "UNKNOWN"


# A host execution-contract refusal (sandbox/canary/policy) is an INFRA fact, never a
# test verdict.  Measured batch 834: the canary RuntimeError escaped run_tests, the
# worker re-raised without a journal row, and finish() closed the step with "".
HOST_REFUSED_PREFIX = "HOST_REFUSED: host could not run tests"


class _HostExecutionRefused(Exception):
    """Typed carrier for a refusal raised by the central execution policy runner."""

    def __init__(self, error: BaseException):
        super().__init__(f"{type(error).__name__}: {error}")
        self.error = error


def _host_refusal_summary(detail: object) -> str:
    """Return the bounded, language-neutral model-facing refusal card."""
    reason = " ".join(str(detail or "").split()).strip(" .")
    if not reason:
        reason = "the host execution policy refused the request"
    reason = reason[:240].rstrip(" .")
    return (f"{HOST_REFUSED_PREFIX}: {reason}. No test ran and no verdict is "
            "available.")


def _journal_records(journal: object) -> tuple[Mapping[str, object], ...]:
    """Read journal events from in-memory fixtures or the real append-only journal."""
    if journal is None:
        return ()
    events = getattr(journal, "events", None)
    if callable(events):
        try:
            events = events()
        except Exception:                                      # noqa: BLE001
            events = None
    if isinstance(events, Mapping):
        events = events.values()
    if events is not None and not isinstance(events, (str, bytes)):
        try:
            return tuple(event for event in events if isinstance(event, Mapping))
        except TypeError:
            pass
    records = getattr(journal, "records", None)
    if callable(records):
        try:
            return tuple(event for event in records() if isinstance(event, Mapping))
        except Exception:                                      # noqa: BLE001
            pass
    path = getattr(journal, "path", None)
    if path is None:
        return ()
    loaded: list[Mapping[str, object]] = []
    try:
        with pathlib.Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except (TypeError, ValueError, UnicodeError):
                    continue
                if isinstance(event, Mapping):
                    loaded.append(event)
    except (OSError, TypeError, ValueError):
        return ()
    return tuple(loaded)


def _host_refusal_visible_observation(journal: object, *, tool: object,
                                      result: object, observation: object = "") -> str:
    """Recover a refused test's explanation when action feedback was lost."""
    visible = str(observation or "")
    if visible.strip() or str(tool or "").strip().lower() not in {
            "test", "run_tests", "pytest"}:
        return visible
    if str(result or "").strip().upper() != "HARNESS_ERROR":
        return visible
    refusal = next(
        (event for event in reversed(_journal_records(journal))
         if event.get("kind") == "run_tests_host_refused"),
        None,
    )
    if refusal is None:
        return visible
    detail = next(
        (refusal.get(key) for key in ("detail", "reason", "message", "error_detail", "error_type")
         if refusal.get(key)),
        "the host execution policy refused the request",
    )
    return _host_refusal_summary(detail)


def _append_host_refusal_observation(runtime: object, receipts: object,
                                     messages: list[dict]) -> str | None:
    """Put the refusal card on the model wire before a failed action is closed."""
    if not isinstance(messages, list):
        return None
    receipt_rows = (receipts if isinstance(receipts, Sequence)
                    and not isinstance(receipts, (str, bytes)) else ())
    latest_test = next(
        (receipt for receipt in reversed(tuple(receipt_rows))
         if isinstance(receipt, Mapping)
         and str(receipt.get("kind", "")).strip().lower() in {"test", "run_tests", "pytest"}),
        None,
    )
    if latest_test is None:
        return None
    action = getattr(runtime, "current_action", None)
    action_kind = action.get("kind") if isinstance(action, Mapping) else None
    if action_kind and str(action_kind).strip().lower() not in {"test", "run_tests", "pytest"}:
        return None
    card = _host_refusal_visible_observation(
        getattr(runtime, "journal", None),
        tool=latest_test.get("kind"),
        result=latest_test.get("result_class", latest_test.get("result")),
        observation="",
    )
    if not card:
        return None
    if any(str(message.get("content") or "") == card
           for message in messages if isinstance(message, Mapping)):
        return card
    tool_call_id = getattr(runtime, "current_tool_call_id", None)
    message: dict[str, object] = _tool_reply_for_call(
        messages, tool_call_id, card, no_id_role="user",
        legacy_key_order=("role", "content", "tool_call_id"),
    )
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    messages.append(message)
    return card


def _host_refusal_observation(runtime, refusal: "_HostExecutionRefused", *, selector: str,
                              binding: object, execution_id: object = None) -> str:
    """Render, receipt and journal one host refusal; never empty, never a verdict."""
    detail = str(refusal) or type(refusal.error).__name__
    text = _host_refusal_summary(detail)
    if runtime is None:
        return text
    journal = getattr(runtime, "journal", None)
    if callable(getattr(journal, "emit", None)):
        try:
            journal.emit("run_tests_host_refused", producer="react_repair.run_tests",
                         error_type=type(refusal.error).__name__, detail=detail,
                         selector=selector[:300], dispatch=False)
        except Exception as emit_error:                              # noqa: BLE001
            print(f"    [kernel-wiring] run_tests_host_refused emit failed: "
                  f"{type(emit_error).__name__}")
    try:
        from kernel_check_truth_adapter import record_native_observation
        record_native_observation(
            runtime, selector=selector, status="HOST_REFUSED", stderr=detail,
            assertion_context=text, outcome="ENVIRONMENT", failure_phase="HOST",
            reached_observation=False,
            execution_id=str(execution_id or ""),
            revision=(binding.get("product_revision") if isinstance(binding, Mapping) else None),
            sequence=getattr(getattr(runtime, "owner", None), "current_step", None))
    except Exception as receipt_error:                               # noqa: BLE001
        return text + f"\n[check receipt unavailable: {type(receipt_error).__name__}]"
    return text


def _run_tests_impl(root: pathlib.Path, fn: str, explicit: str | None,
                    rel_hint: str | None = None, row=None) -> str:
    """rel_hint is the file under repair — used to tell OUR import failures from the agent's."""
    _profile = None
    if os.environ.get("STATEFRAME_TEST_RUN_SPANS", "off").strip().lower() in {"shadow", "on"}:
        from kernel_test_run_spans import current_profile
        _profile = current_profile()

    def _profile_phase(name: str, overlap_group: str | None = None):
        if _profile is None:
            return nullcontext()
        return _profile.span(name, overlap_group=overlap_group)

    _selector_span = _profile.start_span("selector_resolution") if _profile else None
    if explicit:
        from kernel_public_evidence import requested_check_targets
        requested = requested_check_targets(root, explicit)
        if not requested:
            if _profile:
                _profile.end_span(_selector_span)
            return "pytest exit 4 — invalid test target list; provide file paths or selectors, not options."
        for target in requested:
            base = str(target).split("::", 1)[0].strip("`'\"").lstrip("./")
            # A path-like selector must resolve before it reaches pytest.  Bare
            # names remain valid -k expressions, but a guessed public test file
            # is a host refusal, not product-red evidence and must not become
            # the next repair loop.
            path_like = (
                "/" in base or "\\" in base or base.startswith("tests/")
                or base.endswith(".py")
            )
            if path_like:
                try:
                    candidate = (root / base).resolve()
                    root_resolved = root.resolve()
                    if not candidate.is_relative_to(root_resolved) or not candidate.is_file():
                        from kernel_nav_basics import missing_path_message
                        if _profile:
                            _profile.end_span(_selector_span)
                        return (
                            "[selector] REFUSED_PREDISPATCH — public selector does not "
                            f"resolve to an existing repository test: {target}; "
                            f"{missing_path_message(root, base, label='selector')}"
                        )
                except (OSError, RuntimeError, ValueError):
                    from kernel_nav_basics import missing_path_message
                    if _profile:
                        _profile.end_span(_selector_span)
                    return (
                        "[selector] REFUSED_PREDISPATCH — public selector could not be "
                        f"resolved: {target}; {missing_path_message(root, base, label='selector')}"
                    )
        if len(requested) > 1:
            if _profile:
                _profile.end_span(_selector_span)
                _profile.suppress_event = True
                from kernel_test_run_spans import independent_profile_scope
                with independent_profile_scope():
                    return run_tests_for_selectors(root, fn, requested, rel_hint, row=row)[0]
            return run_tests_for_selectors(root, fn, requested, rel_hint, row=row)[0]
        explicit = requested[0]
    if _profile:
        _profile.end_span(_selector_span)
    swebench = bool(row and row.get("source_dataset") and row.get("base_commit"))
    _db_ok = True
    if not swebench and not (row and row.get("_skip_db")):
        _db_span = _profile.start_span("db_liveness_recovery") if _profile else None
        try:
            _db_ok = _ensure_db()
        finally:
            if _profile:
                _profile.end_span(_db_span)
    if not _db_ok:
        # INFRA card without spending the model's epistemic budget on a lie.
        return ("pytest NOT RUN — the test database is unreachable (environment "
                "failure, NOT evidence about your patch). The harness attempted "
                "recovery and failed; try again in a few steps.")
    _target_span = _profile.start_span("selector_target_resolution") if _profile else None
    if explicit:
        targets = [explicit]
    else:
        targets = _discover_test_targets(root, fn)
    if not targets:
        if _profile:
            _profile.end_span(_target_span)
        return ("no test file in this repository calls that function — there is no test "
                "oracle for this bug, so verify your fix by reading the code carefully.")
    # A node id the agent GUESSED (`::TestFoo`) yields pytest exit 4, which reads as failure.
    # Measured live: a bare `test` and a guessed `::TestBuildChatBlock` both came back as
    # usage errors and the agent then went looking for a Makefile. If the selector does not
    # resolve, fall back to the whole file rather than handing back a meaningless error.
    kexpr = None
    if explicit:
        base = explicit.split("::", 1)[0].strip("`'\"").lstrip("./")
        # A BARE TEST NAME is a `-k` request, not a path (defect #34, caught at s01 of the
        # first honest graded task: the agent asked for
        # `test test_bulk_ack_graduation_decision_applies_grant` — a name it had just read in
        # the test file — and the resolver treated it as a filename, found nothing, and
        # answered "path does not exist". Meeting the dialect: name-shaped selectors run as
        # `pytest -k <name>` against the discovered/known test files.
        if ("/" not in base and "::" not in explicit and base.startswith("test_")
                and not base.endswith(".py")):
            kexpr = base
            # Retarget to the file(s) that actually DEFINE this test — targets currently
            # holds the bare name, which pytest would treat as a missing path.
            out_targets = _discover_test_targets(root, f"def {base}")
            try:
                selected = unique_tool_match(
                    root, base, [pathlib.Path(f) for f in out_targets])
            except AmbiguousToolPath as error:
                if _profile:
                    _profile.end_span(_target_span)
                return f"pytest exit 4 — {error}"
            found = [str(selected)] if selected is not None else []
            if found:
                targets = found
            else:
                from kernel_nav_basics import missing_path_message
                if _profile:
                    _profile.end_span(_target_span)
                return (f"no test named `{base}` is defined anywhere in this repository — "
                        f"re-check the name with `grep 'def {base}'`. "
                        f"{missing_path_message(root, base, label='selector')}")
        elif not (root / base).exists():
            # Resolve ANY unresolved test path, not just node ids. Measured live: `test
            # tests/test_memory_generation_edge_cases.py` returned exit 4 because the file
            # sits at a different depth in the sandbox than the agent assumed. Answering a
            # correct intent with a usage error sends it hunting for a Makefile.
            try:
                selected = resolve_tool_path(root, base)
            except AmbiguousToolPath as error:
                if _profile:
                    _profile.end_span(_target_span)
                return f"pytest exit 4 — {error}"
            if selected.exists():
                suffix = explicit[len(explicit.split("::", 1)[0]):]
                targets = [str(selected) + suffix]
    if _profile:
        _profile.end_span(_target_span)
    _config_span = _profile.start_span("config_command_preparation") if _profile else None
    try:
        # THE MC TEST ORACLE NEEDS ITS DEV POSTGRES. `mission-control-db` (pgvector on 55432,
        # The local development database had been unavailable, so every
        # MC test failed with `psycopg2.OperationalError: connection refused` and we read those
        # as bug-related failures. That single stopped container is why the loop had no oracle.
        #
        # Point at an ISOLATED database (rule #190: never share a test DB between agents; MC's
        # test DB accumulates and other lanes use their own). MC's conftest defaults to
        # `mission_control_test` and never drops a schema, so dev data is untouched either way.
        env = dict(os.environ)
        # A FRESH DATABASE PER PYTEST INVOCATION. grading audit asked "is the parent-
        # control run genuinely independent?" and the answer was NO, measured: the parent run
        # (-x, stops at the fail-to-pass assertion) leaves ledger rows behind, and the gold
        # run then fails on protocol state it inherited — a sha-era test failing against the
        # sha-era tree purely from OUR reuse. MC's conftest never drops schemas (rule #190,
        # the accumulating test DB) but it DOES auto-create any database name it is handed,
        # so a unique name per run buys clean state for the cost of schema creation.
        # E279 — THE FRESH-DB-PER-INVOCATION FIX WAS A DISK BOMB. A uuid per pytest run
        # created 118 full MC schemas in one night and filled the Docker volume until
        # postgres could not write its own pid file ("No space left on device") — my E265
        # isolation fix industrialised the rule-#190 accumulation it was avoiding. One DB
        # per TASK is the right grain: the E265 pollution was parent-control-vs-gold WITHIN
        # a task's grading legs, which target-file rewrites between legs already reset;
        # cross-invocation reuse inside one task is the iteration loop working as intended.
        # REACT_DB_NAME is set per task by the caller; converge sweeps react_* leftovers at
        # startup so even a crashed run leaks at most a handful.
        import uuid as _uuid
        _dbname = os.environ.get("REACT_DB_NAME") or (
            f"mission_control_test_react_{_uuid.uuid4().hex[:8]}")
        # E387b: the AHA sandbox conftest imports Settings, which REQUIRES database_url +
        # jwt_secret at import time — absent env means pytest exit 4 on EVERY invocation
        # (both preflight legs died identically, masquerading as "no usable oracle").
        # Harmless defaults only fill MISSING vars; suites that read them get a real-shaped
        # local value, never production anything.
        # E387c (v2): corpus-fix replacement oracles are dependency-light BY MANDATE, but
        # blanket --noconftest broke MC oracles whose fixtures legitimately live in
        # conftest (measured: _collect_metrics "fixture not found" on both legs). The
        # precise rule: run normally; ONLY when the repo's conftest itself fails to
        # IMPORT (exit 4, AHA's Settings/DB-at-import world) retry oracle targets
        # without conftests — the conftest was unusable anyway, so nothing is lost.
        from kernel_test_python import test_python
        executable = (
            row.get("python")
            if isinstance(row, dict) and isinstance(row.get("python"), str)
            and row.get("python")
            else test_python()
        )
        _cmd = ([executable, "-m", "pytest", "-x", "-q", "--no-header"]
                + (["-k", kexpr] if kexpr else []) + ["-p", "no:cacheprovider"]
                + _nearest_pytest_config_args(root, targets))
        from kernel_step_contract import enabled as contract_enabled
        if isinstance(row, dict) and row.get('_skip_db') and (
                contract_enabled() or os.environ.get("STATEFRAME_ACTION_SCHEMA", "off") == "on"):
            # Standalone contract tasks use installed pytest, fixed configuration
            # and no candidate-supplied collection hooks.
            _cmd = ([executable, '-I', '-m', 'pytest', '--noconftest', '-c', '/dev/null',
                     '--rootdir', str(root), '-o', 'pythonpath=' + str(root)] + _cmd[3:])
        from kernel_test_artifact_policy import (
            changed_write_set, prepare_pytest_command, snapshot_workspace,
        )
        _cmd, env, _test_artifact_policy = prepare_pytest_command(_cmd, root, environment=env)
        _profiled_command = False
        if _profile:
            try:
                from kernel_test_run_spans import profiled_pytest_command
                _profiled = profiled_pytest_command(_cmd, _profile.execution_id)
                if _profiled is not None:
                    _cmd = list(_profiled)
                    _profiled_command = True
                    _profile.mark("pytest_profile_command_ready")
                else:
                    _profile.mark("pytest_profile_command_unsupported")
            except Exception:  # optional instrumentation must leave today's command runnable
                _profile.mark("pytest_profile_command_failed_open")
        if _profile:
            _profile.end_span(_config_span)
        _charge_span = _profile.start_span("test_subprocess_charge") if _profile else None
        from kernel_pilot import ACTIVE_RUNTIME, charge
        charge("tool_work", "test subprocess")
        if _profile:
            _profile.end_span(_charge_span)
        _runtime = ACTIVE_RUNTIME.get()
        if _profile:
            _profile.bind_runtime(_runtime)
            _profile.mark("execution_identity_prepared")
        _binding_span = _profile.start_span("pre_test_binding") if _profile else None
        _test_binding = _runtime.binding() if _runtime else {}
        if _profile:
            _profile.end_span(_binding_span)
        _execution_id = None
        if _runtime is not None:
            from kernel_check_truth_adapter import begin_check_execution
            _execution_id = begin_check_execution(
                _runtime,
                selector=explicit or ", ".join(str(target) for target in targets),
                sequence=getattr(getattr(_runtime, "owner", None), "current_step", None),
                binding=_test_binding,
                selector_scope=tuple(str(target) for target in targets),
                execution_id=_profile.execution_id if _profile else None,
            )
        if _profile and _execution_id:
            _profile.execution_id = str(_execution_id)
            _profile.mark("canonical_execution_id_bound")
        _snapshot_span = _profile.start_span("pre_test_snapshot") if _profile else None
        _test_write_snapshot = snapshot_workspace(root)
        if _profile:
            _profile.end_span(_snapshot_span)
        _check_output_truncated: bool | None = None

        def _timeout_stream(value):
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value if isinstance(value, str) else ""

        def _shadow_observation(shadow_text):
            rendered = str(shadow_text)
            if _runtime is None:
                return rendered
            from kernel_check_receipts import compact_observation
            from kernel_check_truth_adapter import record_native_observation

            with _profile_phase("first_receipt"):
                shadow_receipt = record_native_observation(
                    _runtime,
                    selector=explicit or ", ".join(str(target) for target in targets),
                    status="NOT_RUN",
                    assertion_context=rendered,
                    execution_id=str(_execution_id or ""),
                    failure_phase=None,
                    revision=(_test_binding.get("product_revision")
                              if isinstance(_test_binding, Mapping) else None),
                    sequence=getattr(getattr(_runtime, "owner", None), "current_step", None),
                    output_truncated=None,
                )
            return rendered + "\n" + compact_observation(shadow_receipt)

        def _run_check(command):
            nonlocal _check_output_truncated
            _check_output_truncated = None
            from kernel_execution_policy import run_sandboxed
            from kernel_pilot import ParentBudgetExhausted
            try:
                # The CPU budget must match the wall budget: the policy default (30 s RLIMIT_CPU,
                # hard) SIGKILLed a 37 s public check mid-run (rc 137, batch 843) while this call
                # allowed 300 s of wall time.  Recorded bound: PUBLIC_CHECK_CPU_SECONDS.
                result = run_sandboxed(
                    command,
                    root,
                    env=env,
                    timeout=300,
                    cpu_seconds=PUBLIC_CHECK_CPU_SECONDS,
                    output_limit=256 * 1024,
                    **({"observe_pytest": True} if _profiled_command else {}),
                )
            except ParentBudgetExhausted:
                raise
            except RuntimeError as refusal:  # ExecutionPolicyError, stage6 canary, ...
                raise _HostExecutionRefused(refusal) from refusal
            reported_truncation = getattr(result, "output_truncated", None)
            _check_output_truncated = (
                reported_truncation if type(reported_truncation) is bool else None
            )
            if _profile:
                try:
                    _profile.record_sandbox_result(
                        getattr(result, "execution_trace", {}),
                        getattr(result, "profile_payload", None),
                        stdout=getattr(result.completed, "stdout", ""),
                        stderr=getattr(result.completed, "stderr", ""),
                        returncode=getattr(result.completed, "returncode", None),
                    )
                except Exception:  # observation failures never change the check result
                    pass
            if result.timed_out:
                raise subprocess.TimeoutExpired(command, 300, output=result.completed.stdout,
                                                stderr=result.completed.stderr)
            return result.completed

        _check_output_truncated = None
        p, _shadow = _effect_test_subprocess(
            _compose_test_command(_cmd, targets), root=root, targets=targets, env=env,
            runner=_run_check)
        if _shadow is not None:
            return _shadow_observation(_shadow)
        _is_oracle = any(str(t).endswith("_oracle.py") or "_oracle.py::" in str(t)
                         for t in targets)
        if (p.returncode == 4 and _is_oracle
                and "ImportError while loading conftest" in (p.stdout + p.stderr)):
            charge("tool_work", "test subprocess conftest retry")
            _check_output_truncated = None
            p, _shadow = _effect_test_subprocess(
                _compose_test_command((*_cmd, "--noconftest"), targets), root=root,
                targets=targets, env=env, runner=_run_check)
            if _shadow is not None:
                return _shadow_observation(_shadow)
    except _HostExecutionRefused as host_refusal:
        return _host_refusal_observation(
            _runtime, host_refusal,
            selector=explicit or ", ".join(str(target) for target in targets),
            binding=_test_binding, execution_id=_execution_id)
    except subprocess.TimeoutExpired as timeout_error:
        try:
            timeout_s = float(getattr(timeout_error, "timeout", 300) or 300)
        except (TypeError, ValueError):
            timeout_s = 300.0
        timeout_label = f"{timeout_s:g}"
        if _runtime is not None and callable(getattr(_runtime, "note_long_run_signal", None)):
            _runtime.note_long_run_signal("check_timeout", timeout_s=timeout_s)
        if _runtime is not None:
            from kernel_public_evidence import check_selectors
            _timeout_observation = check_selectors(root, list(targets), kexpr)
            with _profile_phase("post_snapshot_integrity"):
                _timeout_write_set = changed_write_set(_test_write_snapshot, snapshot_workspace(root))
            with _profile_phase("readiness_binding_enrichment"):
                _runtime.test_observation(
                    _timeout_observation, 1, _test_binding,
                    observation=str(_timeout_observation) + "\npytest exit 1 — TIMED OUT",
                    run_write_set=_timeout_write_set, command=_cmd,
                    test_environment=env,
                )
        # E403: carry a parseable exit marker — a HANGING implementation is a FAILING
        # implementation. Without "pytest exit" in this string the grading tail mapped a
        # hang to no_test_oracle (next_cron_match: infinite scan graded as a missing
        # oracle instead of tests_fail — misleading label, wrong bookkeeping).
        timeout_context = (
            f"pytest exit 1 — TIMED OUT after {timeout_label}s. The configured check "
            "did not establish a result within its time budget; code, harness, and "
            "environment are not distinguished by this observation.")
        if _runtime is not None:
            from kernel_check_receipts import compact_observation
            from kernel_check_truth_adapter import record_native_observation
            with _profile_phase("first_receipt"):
                timeout_receipt = record_native_observation(
                    _runtime,
                    selector=explicit or ", ".join(str(target) for target in targets),
                    status="TIMEOUT",
                    stdout=_timeout_stream(timeout_error.output),
                    stderr=_timeout_stream(timeout_error.stderr),
                    assertion_context=timeout_context,
                    execution_id=str(_execution_id or ""),
                    exit_code=124,
                    failure_phase="TIMEOUT",
                    revision=(_test_binding.get("product_revision")
                               if isinstance(_test_binding, Mapping) else None),
                    sequence=getattr(getattr(_runtime, "owner", None), "current_step", None),
                    output_truncated=_check_output_truncated,
                )
            return timeout_context + "\n" + compact_observation(timeout_receipt)
        return timeout_context + (
            " Preserve the partial output, keep completion unresolved, and use an offered "
            "diagnostic or verification action before attributing a product defect.")
    if _runtime is not None:
        from kernel_public_evidence import check_selectors
        _test_observation = check_selectors(root, list(targets), kexpr)
        with _profile_phase("post_snapshot_integrity"):
            _test_write_set = changed_write_set(_test_write_snapshot, snapshot_workspace(root))
        with _profile_phase("readiness_binding_enrichment"):
            _runtime.test_observation(
                _test_observation, p.returncode, _test_binding,
                observation=p.stdout + "\n" + p.stderr,
                run_write_set=_test_write_set, command=_cmd, test_environment=env,
            )
        _repro_gate = getattr(_runtime, "repro_first_gate", None)
        _repro_context = getattr(_runtime, "_repro_first_observation_context", None)
        if (__import__("kernel_repro_first").repro_first_gate_mode() == "on"
                and _repro_gate is not None
                and isinstance(_repro_context, Mapping)
                and _repro_context.get("observe_in_runner", True)):
            from kernel_repro_first_coding import observe_pytest
            _probe_id = str(_repro_context.get("probe_id") or rel_hint or fn or "pytest")
            observe_pytest(
                _repro_gate,
                "pytest exit %d\n%s\n%s" % (p.returncode, p.stdout, p.stderr),
                command=str(_repro_context.get("command") or ("pytest:" + _probe_id)),
                probe_id=_probe_id,
                root=root,
                exit_code=p.returncode,
            )
    tail = _strip_pytest_separators((p.stdout + p.stderr).strip().splitlines())
    # INTERPRET THE EXIT CODE. A bare "pytest exit 4" is unreadable feedback and the agent
    # treats it as a failing test — but 4 is a USAGE error (bad path / bad node id) and 5 is
    # "no tests collected". Our own rule #209: a pytest result with no test count is not a
    # verdict. Seen live: the agent got `exit 4` on a node id it guessed, and had no way to
    # know its command was malformed rather than its patch wrong. Telling it which one is
    # true costs nothing and saves the rest of the run.
    meaning = {
        # Signal exits (128+N / negative) are HOST outcomes, never a verdict on the patch.
        **{code: ("CHECK PROCESS KILLED BY THE HOST SANDBOX (signal %d) — a resource limit or "
                  "timeout, NOT a test result; nothing about your patch was observed. Re-run the "
                  "same check once; if it is killed again, narrow the selector." % (code - 128))
           for code in (130, 134, 137, 143, 152, 153)},
        124: "CHECK TIMED OUT (host wall limit) — not a test result; re-run or narrow the selector.",
        0: "ALL TESTS PASSED.",
        1: "TESTS FAILED — the traceback below names the assertion; your patch is wrong or "
           "incomplete.",
        2: "INTERRUPTED / COLLECTION ERROR — a test file or import is broken, usually "
           "TEST PLUMBING, not your product patch. Read the error, fix the plumbing "
           "ONCE if it is yours, otherwise pick a selector that collects. Do NOT treat "
           "this as evidence about your fix (E410).",
        3: "INTERNAL PYTEST ERROR — environment, not your patch. Re-run once; if it "
           "repeats, use a different selector (E410).",
        4: "USAGE ERROR. This says NOTHING about your patch.",
        5: "NO TESTS COLLECTED — the file exists but contains no tests matching that "
           "selector. This says NOTHING about your patch.",
    }.get(p.returncode, f"pytest exited {p.returncode}.")

    # DISTINGUISH "BAD PATH" FROM "THIS SUITE CANNOT RUN HERE" (exit 4 covers both).
    #
    # Measured live: the agent ran the SAME valid test file three times and was told each time
    # that the path did not exist. The path was fine — the AHA repo's conftest.py fails to
    # import in our sandbox, so nothing can collect. Telling an agent to fix a path that was
    # never wrong is the same lie class as the grep and read defects: it sends it hunting for
    # a problem that is ours, not its.
    blob = "\n".join(tail)
    # A COLLECTION FAILURE CAN EXIT 1, NOT ONLY 3/4. Measured: the agent was told "TESTS
    # FAILED — your patch is wrong or incomplete" while the output was a pytest plugin
    # ImportError and not one test had run. Detect the harness-level failure by its CONTENT at
    # ANY exit code; the code alone is not sufficient evidence.
    # AN IMPORT ERROR IN THE FILE THE AGENT JUST PATCHED IS THE AGENT'S FAULT.
    #
    # The first version of this check exonerated the patch for ANY ImportError, which is far
    # too generous: measured live, `article_publishable_projection` got a real exit-1 failure,
    # edited, and then got an exit-4 that we labelled "your patch is NOT implicated" — while
    # the likely cause was its own edit breaking the module import. Telling an agent its broken
    # patch is fine is worse than telling it nothing.
    #
    # Ours = the failure is in conftest or pytest's plugin machinery. Theirs = the traceback
    # names the file under repair.
    _failure_class = _classify_runner_failure(blob, fn, rel_hint)
    _harness_err = _failure_class == "ENVIRONMENT"
    if _harness_err:
        first = next((line for line in blob.splitlines()
                      if "Error" in line or "conftest" in line), "")
        meaning = ("THIS REPOSITORY'S TEST SUITE CANNOT RUN IN THIS SANDBOX (its conftest or a "
                   "pytest plugin fails to import). Your path was fine and your patch is NOT "
                   "implicated. There is NO test oracle for this bug — do not retry `test`; "
                   f"verify your fix by reading the code. [{first.strip()[:120]}]")
    elif (p.returncode == 2
          and os.environ.get("STATEFRAME_TOKENLAB_TRUTHFUL_EXIT2", "off").strip().lower() == "on"):
        from kernel_tokenlab_exit2 import classify as classify_exit2, emit as emit_exit2
        product_error = classify_exit2(
            blob, root=root, targets=targets, fn=fn, rel_hint=rel_hint)
        if product_error is not None:
            meaning = str(product_error["label"])
            emit_exit2(_runtime, product_error)
    elif p.returncode == 4 and targets:
        # RESOLVE RELATIVE TO THE SANDBOX ROOT, not the process cwd (defect #33). The old
        # check ran Path("tests/x.py").exists() from local-swarm-poc, so every relative path
        # "did not exist" — the agent was told its correct path was wrong while the REAL
        # exit-4 cause stayed hidden, and it went off editing conftest.py to fix a phantom.
        _missing_target_meaning = _pytest_exit4_target_meaning(
            root, targets[0], journal=getattr(_runtime, "journal", None),
        )
        if _missing_target_meaning is not None:
            meaning = _missing_target_meaning
        else:
            # The file exists and pytest still exited 4: show the agent the actual reason
            # instead of a guess. The last error line is the truth we have.
            _err = next((line for line in reversed(blob.splitlines())
                         if "Error" in line or "error" in line), "")
            meaning = (f"the file exists but pytest could not run it "
                       f"[{_err.strip()[:140]}]. This says NOTHING about your patch.")
    # SHOW THE ASSERTION, NOT PYTEST'S GUTS. The last 25 lines of a failure are usually
    # `_pytest/config/__init__.py` frames — measured: the agent received 25 lines of pytest
    # internals and nothing about its own code. Prefer the lines that actually name the
    # failure (FAILED/assert/E-prefixed/short summary), and only fall back to the raw tail.
    # KEEP LOCATION FRAMES WHEN THE ERROR IS AN IMPORT-CHAIN FAILURE (defect #35). The
    # E-lines-only filter once reduced a conftest import crash to a bare
    # `NameError: name 'item' is not defined` with no file or line — the agent (and we)
    # then chased a phantom bug at the wrong line. A NameError/ImportError without its
    # `File "...", line N` frame is not evidence, it is a rumour.
    body = _pytest_failure_excerpt(tail)
    receipt = None
    if _runtime is not None:
        from kernel_check_receipts import compact_observation
        from kernel_check_truth_adapter import record_native_observation
        receipt_status = {
            0: "PASS",
            1: "FAIL",
            5: "NO_TESTS",
        }.get(p.returncode, "ERROR")
        receipt_phase = (
            "IMPORT" if _failure_class in {"ENVIRONMENT", "PRODUCT"}
            else "COLLECTION" if p.returncode in {2, 3, 4}
            else None
        )
        receipt_reached = (
            False if receipt_phase is not None
            else True if p.returncode == 1
            else None
        )
        receipt_revision = (_test_binding.get("product_revision")
                            if isinstance(_test_binding, Mapping) else None)
        with _profile_phase("first_receipt"):
            receipt = record_native_observation(
                _runtime,
                selector=explicit or ", ".join(str(target) for target in targets),
                status=receipt_status,
                stdout=p.stdout,
                stderr=p.stderr,
                assertion_context=body,
                execution_id=str(_execution_id or ""),
                exit_code=p.returncode,
                reached_observation=receipt_reached,
                failure_phase=receipt_phase,
                revision=receipt_revision,
                sequence=getattr(getattr(_runtime, "owner", None), "current_step", None),
                output_truncated=_check_output_truncated,
            )
        result = "pytest exit %d — %s\n%s" % (
            p.returncode, meaning, compact_observation(receipt))
    else:
        result = "pytest exit %d — %s\n%s" % (p.returncode, meaning, body)
    if (p.returncode == 0 and _runtime is not None
            and getattr(_runtime, "stateframe_first_green_disclosure_mode", "off") == "on"):
        from kernel_first_green_disclosure import decorate_test_result
        with _profile_phase("final_result_enrichment"):
            result = decorate_test_result(_runtime, result)
    return result


def run_tests(root: pathlib.Path, fn: str, explicit: str | None,
              rel_hint: str | None = None, row=None) -> str:
    """Keep test profiling advisory and attach one scope to each host invocation."""
    if os.environ.get("STATEFRAME_TEST_RUN_SPANS", "off").strip().lower() not in {"shadow", "on"}:
        return _run_tests_impl(root, fn, explicit, rel_hint, row=row)
    from kernel_test_run_spans import profile_scope
    with profile_scope() as profile:
        if profile is None:
            return _run_tests_impl(root, fn, explicit, rel_hint, row=row)
        profile.mark("run_tests_entered")
        with profile.span("run_tests_body"):
            return _run_tests_impl(root, fn, explicit, rel_hint, row=row)


def _selector_exit_code(observation: str) -> int | None:
    match = re.search(r"pytest exit (\d+)", observation)
    return int(match.group(1)) if match else None


def provider_dispatch_allowed(prepared: object) -> bool:
    """The runner's only dispatch gate for a prepared request.

    RENDER_OVERFLOW is a recoverable presentation degradation: the runtime has either
    compacted the view or supplied an explicit omission notice; keep the request alive so a
    red test can be repaired on the next model turn.  Every other overflow kind (budget
    faults, HYBRID_CONTEXT_OVERFLOW) is terminal and is never sent."""
    overflow = getattr(prepared, "overflow", None)
    return not overflow or overflow.get("kind") == "RENDER_OVERFLOW"


def _record_prepared_provider_not_dispatched(
        runtime: object, prepared: object, *, provider_request: bool,
        reason_code: str) -> None:
    if not provider_request:
        return
    capture = getattr(prepared, "capture", None)
    request_id = getattr(capture, "request_id", None)
    from kernel_completion_owner import record_provider_request_not_dispatched
    record_provider_request_not_dispatched(
        runtime, request_id=request_id, reason_code=reason_code)


def run_tests_for_selectors(root: pathlib.Path, fn: str, selectors: Sequence[str],
                            rel_hint: str | None = None, row=None) -> tuple[str, tuple[int, ...]]:
    """Run every logical oracle selector while preserving the legacy run_tests API."""
    observations: list[str] = []
    codes: list[int] = []
    for selector in selectors:
        observation = run_tests(root, fn, selector, rel_hint, row=row)
        code = _selector_exit_code(observation)
        observations.append(f"[{selector}]\n{observation}")
        codes.append(code if code is not None else -1)
    if not codes:
        return "no multi-file selectors supplied", ()
    refused = [obs for obs in observations if obs.split("\n", 1)[-1].startswith(HOST_REFUSED_PREFIX)]
    if refused:
        # No aggregate exit: a refused selector never ran, so no verdict exists.
        return (refused[0].split("\n", 1)[-1].splitlines()[0] + "\n" +
                "\n\n".join(observations)), tuple(codes)
    if all(code == 0 for code in codes):
        aggregate = 0
    elif any(code == 1 for code in codes):
        aggregate = 1
    else:
        aggregate = next(code for code in codes if code != 0)
    return (f"pytest exit {aggregate} — multi-selector oracle ({len(codes)} selectors)\n" +
            "\n\n".join(observations)), tuple(codes)


def _fork_a_revalidate_candidate(workspace: pathlib.Path, fn: str,
                                 selectors: Sequence[str], rel_hint: str | None,
                                 row) -> dict[str, object]:
    """Re-run the public selector oracle inside the parent-owned recheck copy."""
    observation, codes = run_tests_for_selectors(
        workspace, fn, tuple(str(selector) for selector in selectors), rel_hint, row=row)
    return {
        "passed": bool(codes) and all(code == 0 for code in codes),
        "exit_codes": list(codes),
        "observation": observation,
    }


def _fork_a_admit_completed_candidate(measurement, workers: Sequence[object], *,
                                      fn: str, selectors: Sequence[str],
                                      rel_hint: str | None, row) -> object | None:
    """Give the parent exactly one completed worker and a public recheck callback."""
    candidate = next(
        (worker for worker in workers if getattr(worker, "status", None) == "completed"),
        None,
    )
    if candidate is None:
        return None

    def revalidate(worker, workspace):
        result = _fork_a_revalidate_candidate(
            workspace, fn, selectors, rel_hint, row)
        return {**result, "parent_revision": worker.parent_revision}

    return measurement.fork_a_admit(candidate.worker_id, revalidate=revalidate)


def run_observation_probe(root: pathlib.Path, code: str, question: str | None = None,
                          *, row=None, runner=None):
    """Run the observation probe through its host-owned runner."""
    from kernel_probe import ProbeRunner
    if runner is None:
        task_id = (row.get("task_id") or row.get("id") or "") if isinstance(row, Mapping) else ""
        runner = ProbeRunner(
            root, task_id=str(task_id),
            task_deadline=(row.get("task_deadline") if isinstance(row, Mapping) else None),
            finish_reserve_seconds=(row.get("finish_reserve_seconds", 0)
                                    if isinstance(row, Mapping) else 0),
            executable=(row.get("python") if isinstance(row, Mapping)
                        and isinstance(row.get("python"), str) else None),
        )
    return runner.run(code, question)


def run_probe(root: pathlib.Path, code: str, *, row=None, question: str | None = None,
              runner=None) -> str:
    """Run one bounded snippet in the same workspace subprocess lane as tests.

    The snippet is an observation mechanism only.  It does not parse product
    source or derive an expected value; the caller supplies the model's claim
    separately and the kernel records that claim as MODEL_INTERPRETATION.
    """
    # The former OFF fallback executed model code directly with inherited host
    # privileges. It is intentionally removed from the supported artifact;
    # callers must opt into the sandboxed observation probe runner.
    from kernel_probe import observation_probe_mode, render_probe_receipt
    if observation_probe_mode() != "on":
        return (
            "probe UNAVAILABLE — legacy unsandboxed probe disabled; "
            "enable STATEFRAME_PROBE=on for the fail-closed sandboxed runner"
        )
    return render_probe_receipt(
        run_observation_probe(root, code, question, row=row, runner=runner))


def inworld_obligation_gate(
    task: Mapping[str, object],
    oracle_ids: Sequence[str],
    receipts: Sequence[object],
    *,
    unit_plan: object | None = None,
) -> tuple[bool, str]:
    """Check only observable oracle obligations; held-out IDs are never blockers."""
    if not oracle_ids:
        return True, ""
    try:
        from kernel_obligations import (
            compile_obligation_graph,
            done_gate_refusal,
            update_obligation_graph,
        )

        graph = compile_obligation_graph(
            task,
            oracle_ids=oracle_ids,
            unit_plan=unit_plan,
        )
        graph = update_obligation_graph(graph, receipts)
        if not graph.complete_enough():
            return False, "[kernel-gate] " + done_gate_refusal(graph)[:900]
    except Exception:
        # Structured dispatch must not allow completion when this gate is unavailable;
        # retain the legacy fail-open behavior only when both structured modes are off.
        if (os.environ.get('REACT_STEP_CONTRACT', '0') == '1'
                or os.environ.get('REACT_NATIVE_TOOL_BATCHES', '0') == '1'):
            return False, '[kernel-gate] Completion unverified: obligation computation failed.'
        return True, ""
    return True, ""


def _memory_identity_fields(row: Mapping[str, object]) -> dict[str, object]:
    """STATEFRAME_MEMORY_ISOLATION=on: persist the explicit task identity in the
    transcript so a later same-task run can prove provenance.  Off: {} (byte-identical)."""
    from kernel_memory_isolation import enabled, explicit_task_identity
    if not enabled():
        return {}
    task_id = explicit_task_identity(row)
    if task_id is None:
        return {}
    fields: dict[str, object] = {"task_id": task_id}
    if isinstance(row.get("repo"), str) and row["repo"].strip():
        fields["repo"] = row["repo"]
    return fields


def _exception_location(error: BaseException) -> dict[str, object]:
    """Bounded innermost frame + traceback tail for a swallowed harness-terminal error.

    In benchmark, HARNESS_CONTRACT returned an outcome instead of
    raising, so the worker receipt said last_frame=<unavailable> and traceback="".
    """
    import traceback as _traceback
    try:
        frames = _traceback.extract_tb(error.__traceback__)
        last = frames[-1] if frames else None
        text = "".join(_traceback.format_exception(type(error), error, error.__traceback__))
    except Exception:  # noqa: BLE001 - location capture must never replace the outcome
        return {"last_frame": {"file": "<unavailable>", "function": "<unavailable>", "line": 0},
                "traceback_tail": "", "traceback_truncated": False}
    return {
        "last_frame": {"file": last.filename if last else "<unavailable>",
                       "function": last.name if last else "<unavailable>",
                       "line": int(last.lineno or 0) if last else 0},
        "error_message": str(error)[:500],
        "traceback_tail": text[-4000:],
        "traceback_truncated": len(text) > 4000,
    }


def _record_post_finish_fact(measurement: object, kind: str, **fields: object) -> str:
    """Record a host fact that may arrive AFTER measurement.finish() closed the journal.

    10-08 (L126e 928/929, both real solves lost): the clean-finish manifest block runs after finish(), so its
    failure emit hit the closed writer, JournalError masked the manifest error, and a PASS became WORKER_EXCEPTION.
    Never raises. Journal when the writer is still open; otherwise stderr + <journal dir>/post_finish_facts.jsonl.
    Returns where the fact went ("journal" | "sidecar" | "stderr").
    """
    try:
        line = json.dumps({"kind": kind, **fields}, sort_keys=True, default=str)
    except Exception as error:  # noqa: BLE001 - preserve a small typed fallback if a field is malformed
        line = json.dumps({
            "kind": str(kind)[:120],
            "fact_serialization_failed": True,
            "error_type": type(error).__name__,
        }, sort_keys=True)
    try:
        print(f"[post-finish] {line}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - stderr may be unavailable in a broken host process
        pass
    try:
        journal = getattr(measurement, "journal", None)
    except Exception:  # noqa: BLE001
        journal = None
    # Try the journal first (an open writer, or any journal-like object); a closed writer raises and falls back.
    try:
        emit = getattr(journal, "emit", None)
    except Exception:  # noqa: BLE001
        emit = None
    if callable(emit):
        try:
            emit(kind, **fields)
            return "journal"
        except Exception as error:  # noqa: BLE001 - this fact must never cost the verdict
            emit_error = f"{type(error).__name__}: {str(error)[:160]}"
    else:
        emit_error = "no journal writer"
    try:
        journal_path = getattr(journal, "path")
        sidecar = pathlib.Path(journal_path).parent / "post_finish_facts.jsonl"
        with sidecar.open("a") as handle:
            handle.write(line + "\n")
        destination = "sidecar"
    except Exception:  # noqa: BLE001
        destination = "stderr"
    # Expected after finish() closes the writer: the fact is NOT lost, it lives in the sidecar.
    try:
        print(f"[post-finish] {kind} recorded to {destination} "
              f"(journal unavailable after finish: {emit_error})", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001
        pass
    return destination


def _attach_env_receipt(result: Mapping[str, object], measurement: object) -> dict[str, object]:
    """Carry the host receipt into the harness row without changing OFF rows."""
    result_payload = dict(result)
    if getattr(measurement, "_deferred_finalization_no_changes_classified", False):
        result_payload["result"] = "NO_CHANGES"
    if getattr(measurement, "stateframe_delivery_contract_mode", "off") == "on":
        try:
            manifest = measurement.delivery_manifest_payload()
        except (AttributeError, TypeError, ValueError):
            manifest = None
        if manifest is not None:
            result_payload["delivery_manifest"] = manifest
    receipt = getattr(measurement, "env_preflight_receipt", None)
    if receipt is None:
        return result_payload
    payload = receipt.as_dict() if hasattr(receipt, "as_dict") else receipt
    return {**result_payload, "env_receipt": payload}


def _red_confrontation_target(symbol: object, path: object) -> tuple[str, str] | None:
    if not isinstance(symbol, str) or not symbol.strip():
        return None
    if not isinstance(path, str) or not path.strip():
        return None
    return path.strip(), symbol.strip()


def work(d, *, resume=None):
    # Workers and scripted tapes may assign MAX_STEPS after import.
    step_cap = parse_step_limit(MAX_STEPS)
    # V29 real-repository lane: an explicit typed row enters the canonical
    # StateFrame runtime path without being coerced into the synthetic build
    # corpus or its gold/source fields.  The ordinary mapping path below stays
    # byte-compatible for existing probes and release diagnostics.
    from kernel_arbitrary_repo_agent import ArbitraryRepoRow, run as run_arbitrary_repo
    if isinstance(d, ArbitraryRepoRow):
        return run_arbitrary_repo(d, resume=resume)
    _edit_nturn_run_started = time.monotonic()
    _entry_ticket_text = str(d.get("msg") or "")
    from kernel_native_tools import NativeToolBatch, validate_configuration
    from kernel_step_contract import (
        action_schema_enabled,
        entry_file_only_enabled,
        emit_claim_rule_rendered,
        edit_action_applied,
        edit_nturn_enabled,
        edit_nturn_required_seconds,
        edit_nturn_verification_reserve_seconds,
        emit_edit_nturn_rendered,
        emit_edit_nturn_scope_turns_rendered,
        emit_edit_nturn_skipped_budget,
        append_edit_nturn_instruction,
        parse_model_plan_site_records,
        start_edit_nturn_episode,
        is_edit_nturn_no_change_reply,
        is_edit_action_line,
        EDIT_NTURN_RUN_BUDGET_SECONDS,
        prompt_guidance,
        render_blank_entry_template,
        render_claim_rule_first_turn,
        system_prompt_variant,
    )
    _system_prompt_variant = system_prompt_variant()
    _edit_nturn_enabled = edit_nturn_enabled()
    entry_file_only_enabled()
    action_schema_mode = action_schema_enabled()
    native_tool_batches = validate_configuration()
    native_batch = NativeToolBatch() if native_tool_batches else None

    def finish_native_batch_transcript(messages=None):
        if native_batch is None:
            return
        native_batch.finish()
        if isinstance(messages, list):
            native_batch.append_unexecuted_tool_results(messages)

    from kernel_intervention_policy import interventions_enabled
    interventions = interventions_enabled()
    from kernel_information_policy import InformationPolicy
    # Keep controller-only grading material in a private local reference.  The
    # projected row below is the only row allowed to reach worker/runtime state.
    _grading_row = d
    _grading_fixed_src = d.get("fixed_src")
    fixture_initial_src = d.pop("_fixture_initial_src", None)
    if fixture_initial_src is not None and not isinstance(fixture_initial_src, str):
        raise ValueError("fixture initial product source must be text")
    information = InformationPolicy(d)  # Validate before any sandbox, test or model call.
    if information.strict:
        from kernel_test_isolation import backend
        if d.get("repo") != "build":
            raise ValueError("restricted information policies currently require a pure build task; database/network tasks need an isolated service environment")
        information.isolation_backend = backend()  # Fail before inference without an OS boundary.
    from kernel_continuation_restart import enabled as restart_enabled, validate_profile
    if restart_enabled() or resume is not None:
        validate_profile(information)
        if d.get("repo") != "build" or not d.get("_skip_db"):
            raise ValueError("worker restart currently requires an isolated database-free build task")
    from kernel_continuation_runtime import enabled as continuation_enabled
    continuation_active = continuation_enabled()
    if continuation_active:
        from kernel_candidate_restart_reconcile import enabled as candidate_reconcile_enabled
        candidate_reconcile_enabled()
    d = information.worker_row(d)
    _real04_target_metadata = ""
    if os.environ.get("KERNEL_REAL04_TARGET_METADATA", "off").strip().lower() != "off":
        from kernel_real04_bugs import preserve_target_metadata, target_metadata_block
        d = preserve_target_metadata(d, _grading_row)
        _real04_target_metadata = target_metadata_block(d)
    _staged_enabled = _staged_phases_enabled()
    _staged_remaining = (
        [copy.deepcopy(phase) for phase in d.get("staged_phases", ())]
        if _staged_enabled and isinstance(d.get("staged_phases"), Sequence)
        and not isinstance(d.get("staged_phases"), (str, bytes)) else [])
    _current_staged_phase = 1
    fileset = FileSet.from_row(d)
    multifile = fileset.is_multi_file
    if multifile:
        primary_path = fileset.primary_path
        fn = fileset.primary_fn
        source_repo = d.get("repo")
        # Keep the worker's downstream contracts populated from an explicit public
        # entry when present.  A multi-file row with no public entry is still a
        # valid workspace task; it gets a repository layout instead of a guessed
        # first-file target.
        d = {**d,
             "fn": fn,
             "rel": primary_path or "",
             "path": primary_path or "",
             "sha": str(d.get("sha") or "multifile"),
             "repo": d.get("repo") or "build",
             "parent_src": fileset.parent.get(primary_path, "") if primary_path else "",
             "fixed_src": fileset.fixed.get(primary_path, "") if primary_path else "",
             "test_src": d.get("test_src") or "",
             "_skip_db": bool(d.get("_skip_db") or source_repo in (None, "build")),
             }
        parent = gold = None
    else:
        fn = d["fn"]
        parent = fn_src(d["parent_src"], fn)
        gold = fn_src(_grading_fixed_src, fn) if isinstance(_grading_fixed_src, str) else ""
        if not parent or not gold:
            return dict(fn=fn, result="SKIP", steps=0, edits=0, tests=0)
    # E279: ONE database per task for the agent's whole iteration loop (see run_tests).
    # Grading legs override this below — E265's pollution was BETWEEN grading legs, and
    # they still get fresh names.
    import uuid as _uuid
    os.environ.setdefault("REACT_RUN_ID", _uuid.uuid4().hex[:6])
    os.environ["REACT_DB_NAME"] = (
        f"mission_control_test_react_{os.environ['REACT_RUN_ID']}_{fn[:20].lower()}")

    src_repo = REPOS.get(d.get("repo") or "aha")
    if src_repo is None and d.get("repo") != "build" and not (d.get("source_dataset") and d.get("base_commit")):
        raise RuntimeError("source repository is not configured; set STATEFRAME_AHA_REPO or STATEFRAME_MC_REPO")
    continuation_workspace = None
    if continuation_active:
        from kernel_continuation_workspace import ContinuationWorkspace
        from kernel_measurement import digest
        continuation_workspace = (ContinuationWorkspace.create(
            POC / "continuation-workspaces", task_digest=digest(d)) if resume is None else
            resume.open_workspace(POC / "continuation-workspaces", task_digest=digest(d)))
        sbox = continuation_workspace.directory
    else:
        sbox = pathlib.Path(tempfile.mkdtemp(prefix="react_"))
    measurement = None
    _final_product_manifest_written = False
    _final_product_manifest_write_failed = False
    try:
        if resume is not None:
            from kernel_continuation_restart import execution_paths
            root, target = execution_paths(continuation_workspace)
        else:
            root = None
            target = None
            if d.get("source_dataset") and d.get("base_commit"):
                from converge import _swebench_path, _swebench_workspace, _write_swebench_runner

                root = sbox / "repo"
                shutil.copytree(
                    _swebench_workspace(d) / "base",
                    root,
                    ignore=shutil.ignore_patterns("__pycache__", ".git", ".*"),
                )
                target = _swebench_path(root, d["path"], "source")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(d["parent_src"], encoding="utf-8")
                _write_swebench_runner(root, d)
            # E380 BUILD TASKS: synthetic greenfield world (must mirror converge._sandbox).
            # The spec tests are visible by design — build-to-spec is TDD; grading still
            # restores the pristine spec (overlay_fix_tests build branch = anti-weakening).
            elif d.get("repo") == "build" and not multifile:
                _broot = sbox / "repo"
                _btarget = _broot / "app" / d["rel"]
                _btarget.parent.mkdir(parents=True, exist_ok=True)
                for _pkg in (_btarget.parent, _broot / "app"):
                    if _pkg.is_dir():
                        (_pkg / "__init__.py").touch()
                _btarget.write_text(d["parent_src"])
                _btdir = _broot / "tests"
                _btdir.mkdir(parents=True, exist_ok=True)
                _materialize_build_test(_broot, d, fn)
            # COHERENT SNAPSHOT. The old sandbox
            # was TODAY'S repo with ONE file reverted to the bug era — "do not mix the current
            # repository with only one historical target file; that can create impossible
            # cross-module contracts." The task's era and the repo's era disagreed on every
            # neighbouring module and on the tests themselves. Autopilot never has this problem:
            # it works on a coherent checkout.
            #
            # For MC tasks, materialise the PARENT COMMIT of the fix via `git archive` — every
            # module, and the tests, from the same moment the bug existed. Falls back to the
            # copy-current approach if the sha does not resolve (and for AHA, whose git root is
            # the whole lab-repo tree).
            root = (sbox / "repo") if d.get("repo") == "build" else root
            if (d.get("repo") == "mc") and d.get("sha") and \
                    os.environ.get("COHERENT_SNAPSHOT", "1") == "1":
                mc_root = src_repo.parent                       # mission-control/ (the git root)
                (sbox / "repo").mkdir()
                sha = str(d["sha"])
                if not re.fullmatch(r"[0-9a-fA-F]{7,64}", sha):
                    print(f"  [snapshot] archive refused invalid commit id for {fn}", flush=True)
                    ar = None
                else:
                    archive = subprocess.run(
                        ["git", "-C", str(mc_root), "archive", f"{sha}^",
                         "backend", "test_support"],
                        capture_output=True, timeout=180, check=False)
                    if archive.returncode == 0:
                        ar = subprocess.run(
                            ["tar", "-x", "-C", str(sbox / "repo")],
                            input=archive.stdout, capture_output=True,
                            timeout=180, check=False)
                    else:
                        ar = archive
                if ar is not None and ar.returncode != 0:
                    # Say WHY out loud (v19: a silent archive failure sent override_mr_checks
                    # into the era-incoherent fallback and then a crash; #208's lesson — an
                    # enumeration/branch decision made on invisible output is unverified).
                    print(f"  [snapshot] archive of {d['sha'][:12]}^ FAILED for {fn}: "
                          f"{(ar.stderr or b'').decode(errors='replace').strip()[:160]}", flush=True)
                if ar is not None and ar.returncode == 0 and (sbox / "repo" / "backend").is_dir():
                    root = sbox / "repo" / "backend"
                    # test_support must sit next to backend's cwd for pytest's plugin import
                    ts = sbox / "repo" / "test_support"
                    if ts.is_dir() and not (root / "test_support").exists():
                        shutil.copytree(ts, root / "test_support")
            if root is None:
                # DEFECT #42 (v19 override_mr_checks, ERR:FileExistsError): the archive branch
                # pre-creates sbox/repo; when the archive fails (sha not in this checkout's
                # history) the fallback copytree collided with that empty dir and the TASK died
                # as a harness error. Same class as the v7 converge crash — clear the stump.
                shutil.rmtree(sbox / "repo", ignore_errors=True)
                shutil.copytree(src_repo, sbox / "repo",
                                ignore=shutil.ignore_patterns("__pycache__", ".git", ".*"))
                root = sbox / "repo"
            # THE SANDBOX MUST BE SELF-CONTAINED OR THERE IS NO TEST ORACLE AT ALL.
            #
            # Measured: MC tests collect fine in the real repo (20 tests) and FAIL in our copy with
            #   ImportError: plugin "test_support.guard_pytest": No module named 'test_support'
            # because `test_support/` lives at mission-control/, one level ABOVE the backend/
            # directory we copy. pytest loads it as a plugin before collecting anything, so every
            # test run in the sandbox died at plugin import — and the agent was told "TESTS FAILED"
            # or "suite cannot run" for a defect that was entirely ours.
            #
            # This is the feedback channel the whole act-observe-revise design depends on. Without
            # it the loop is one-shot with extra steps: measured on `article_publishable_projection`,
            # which applied FOUR sensible edits and learned nothing from any of them.
            for helper in ("test_support",) if src_repo is not None else ():
                src_helper = src_repo.parent / helper
                if src_helper.is_dir() and not (root / helper).exists():
                    shutil.copytree(src_helper, root / helper,
                                    ignore=shutil.ignore_patterns("__pycache__", ".*"))
            # RESOLVE THE CORPUS PATH PROPERLY. `rel` is APP-relative
            # (e.g. "services/autopilot/lifecycle.py") while the sandbox root is the BACKEND dir,
            # so `root/rel` matched 0 of 21 tasks and every one fell through to a basename rglob.
            # For four tasks that picked an unrelated module — `services/autopilot/lifecycle.py`
            # resolved to `app/services/sign_seal/lifecycle.py` — so we planted a historical file
            # over the wrong code and graded the agent on it. Found by trap audit.
            #
            # Try app-relative FIRST, then root-relative, and only then a basename search; a
            # basename search that is ambiguous is now a hard failure rather than a silent guess.
            if target is None and multifile and fileset.primary_path:
                target = root / fileset.primary_path
            if target is None:
                for cand in (root / "app" / d["rel"], root / d["rel"]):
                    if cand.exists():
                        target = cand
                        break
            if target is None and not multifile:
                hits = list(root.rglob(pathlib.Path(d["rel"]).name))
                if len(hits) == 1:
                    target = hits[0]
                elif len(hits) > 1:
                    return dict(fn=fn, result="HARNESS_INVALID", steps=0, edits=0, tests=0,
                                behaviour=f"ambiguous path: {len(hits)} files named "
                                          f"{pathlib.Path(d['rel']).name}")
            if target is None and not multifile:
                return dict(fn=fn, result="NO_TARGET", steps=0, edits=0, tests=0)
            # PLANT THE WHOLE HISTORICAL FILE, not just the function.
            #
            # This used to patch the function only IF it was already present in the live file.
            # The repository has moved on since the corpus was harvested, so on 4 of 21 tasks the
            # target function NO LONGER EXISTS in the current file — the plant silently did
            # nothing, the agent worked on modern code that does not contain the bug, and the
            # shipped patch itself could never grade as PASS. Those four were unwinnable BY
            # CONSTRUCTION and have been silently capping the bench at 17/21 while every score was
            # reported out of 21.
            #
            # `unified_gym` already does it correctly (it writes `t["buggy"]`, the entire parent
            # file). Matching that: write the parent commit's file verbatim, which is the state the
            # bug actually existed in. Verified by benchmark-integrity replay --
            # feeding the harness the known-good patch and checking it scores.
            if multifile:
                # The legacy setup above establishes the same sandbox shape, but the row's
                # complete workspace is authoritative for every declared path.
                root.mkdir(parents=True, exist_ok=True)
                for path, source in fileset.parent.items():
                    destination = root / path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(source, encoding="utf-8")
                target = (root / fileset.primary_path) if fileset.primary_path else None
                _materialize_public_tests(root, fileset, d)
            else:
                target.write_text(d["parent_src"])

        # SUBSTITUTE THE ORIENTATION. Without this the model receives the literal text
        # "{REL}" and "{FN}" — the placeholders were there and never filled, which is exactly
        # the class of defect that made the first run edit nothing.
        # THE BRIEF. Our corpus symptom is a machine-scrubbed keyword string
        # ("Something is going wrong around blocking, parent, work and promotion...") that
        # measurably does not point at the defect — Flash edits lines with 0-6% overlap with
        # gold. When the repository contains a test that FAILS on this buggy code, that
        # failure is a real, non-leaky bug report: it exists independently of the patch and is
        # the same artefact MC's autopilot receives when it succeeds. Prefer it.
        #
        # FAIR_BRIEF=0 disables, so the keyword arm stays reproducible for comparison.
        brief = d.get("symptom") or d.get("msg")
        _initial_repro_probe = None
        # GROUNDED BRIEF (port plan, item 1 — the single biggest measured lever: real
        # commit messages moved symptom/fix overlap from 2/21 to ~20/21 in the 4B campaign
        # retrospective). The scrubbed keyword string measurably does not identify the defect
        # (0-6% edit overlap with gold). The commit SUBJECT LINE describes the PROBLEM —
        # "fix(ledger): a THIRD human-verdict path dropped its evidence — and the test was
        # masking it" — which is exactly the task brief autopilot receives. Subject only; the
        # body is excluded because it can narrate the patch itself.
        if os.environ.get("GROUNDED_BRIEF", "1") == "1" and d.get("sha") \
                and src_repo is not None and d.get("repo") != "build":
            # (build tasks have a synthetic sha and no git world — their brief IS the spec)
            git_root = (src_repo.parent if d.get("repo") == "mc"
                        else src_repo.parent)               # lab-repo root for aha
            gm = subprocess.run(["git", "-C", str(git_root), "log", "--format=%s",
                                 "-1", d["sha"]], capture_output=True, text=True, timeout=30)
            if gm.returncode == 0 and gm.stdout.strip():
                brief = gm.stdout.strip()
        if os.environ.get("FAIR_BRIEF", "1") == "1":
            _probe = run_tests(root, fn, None, d["rel"], row=d)
            _initial_repro_probe = _probe
            # NEVER put an infrastructure error in the bug report. Measured TWICE now: first
            # the dead mission-control-db fed "connection refused" as the brief (E262), then a
            # transient "server closed the connection unexpectedly" did the same in the graded
            # v9 run — the agent spent 200 steps chasing a database outage as if it were the
            # bug. A brief containing infra text is worse than the keyword salad.
            _infra = ("OperationalError" in _probe or "connection" in _probe.lower()
                      or "ImportError" in _probe or "conftest" in _probe)
            if "pytest exit 1" in _probe and not _infra:
                _lines = [line for line in _probe.splitlines()
                          if line.startswith(("E ", "assert", "FAILED", "> "))][:14]
                if _lines:
                    brief = ("A test in this repository FAILS against this code. This is the "
                             "failure:\n" + "\n".join(_lines))
        # ORACLE-SYMPTOM BRIEF (E277). Measured in v17: with only the commit subject,
        # article_publishable_projection latched onto a PASSING (non-discriminating) test,
        # concluded nothing was broken, and made zero edits in 200 steps — the fix-shipped
        # f2p test is absent from its world by design, so the agent had no red signal at
        # all. The preflight already captured the f2p test's FAILURE LINES against this very
        # code; that is the symptom a real bug report carries (and exactly what MC's
        # autopilot receives). It quotes the assertion, never the patch; the test file
        # itself stays out of the agent's world.
        if "_bridge_ordinary_brief" in d:
            brief = str(d["_bridge_ordinary_brief"])
        information.ordinary_brief = str(brief or "")
        if d.get("_oracle_symptom"):
            # E480 (autopsy_desk5): this REPLACED the brief, silently discarding the
            # task msg — every invariant checklist we taught (E464/E473/E475) never
            # reached the model; only test-failure legends survived. The task's own
            # spec is the contract: keep it, then add the symptom.
            _task_msg = str(d.get("_bridge_ordinary_brief", d.get("msg")) or "").strip()
            if multifile:
                brief = (((_task_msg + "\n\n") if _task_msg else "") +
                         "A regression test (not present in your checkout) FAILS against this "
                         "code with:\n" + str(d["_oracle_symptom"])[:1200] + "\n\n" +
                         _render_multifile_list(fileset) +
                         "\nReproduce the broken behaviour from the code itself, fix "
                         f"`{fn}`, and verify with the repository's existing tests.")
            else:
                brief = (((_task_msg + "\n\n") if _task_msg else "") +
                         "A regression test (not present in your checkout) FAILS against this "
                         "code with:\n" + str(d["_oracle_symptom"])[:1200] +
                         "\nReproduce the broken behaviour from the code itself, fix "
                         f"`{fn}`, and verify with the repository's existing tests.")
            # E309b — target_guard fast path: enumerate the suspect
            # function's OWN guard/comparison expressions, deterministically, from the
            # parent source. Pure ground truth the model could read itself — this only
            # saves localization (require_isol failed 4/4 editing AROUND the guard it never
            # looked at). No direction, no suggestion, never the patch.
            try:
                _ptree = ast.parse(d["parent_src"])
                _guards = []
                for _nd in ast.walk(_ptree):
                    if isinstance(_nd, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and _nd.name == fn:
                        for _sub in ast.walk(_nd):
                            if isinstance(_sub, (ast.Compare, ast.BoolOp)) \
                                    and hasattr(_sub, "lineno"):
                                _src = ast.get_source_segment(d["parent_src"], _sub)
                                if _src and len(_src) < 120:
                                    _guards.append(f"  L{_sub.lineno}: {_src}")
                        break
                _seen_g = list(dict.fromkeys(_guards))[:8]
                if _seen_g:
                    brief += ("\nFor orientation, the suspect function currently contains "
                              "these condition expressions:\n" + "\n".join(_seen_g))
            except Exception:                                    # noqa: BLE001
                pass

        # Behavioral guidance occupies approximately 90 tokens of the
        # invariants appended to the system prompt. Not cosplay ("you are an elite
        # engineer"), invariants ("evidence outranks intuition"). A/B via
        # REACT_CONSTITUTION: unset/1 = on (default after A/B), 0 = off.
        # REGIME RIG (off by default): keep both controls local to each run so tests can
        # monkeypatch the environment without re-importing this hot module.  With zero values
        # the exact seven-tool object and exact system message below are preserved.
        _decoy_count = nonnegative_env_int("REACT_DECOY_TOOLS")
        _decoy_kind = selected_decoy_kind()
        from kernel_lean import PLAIN_SYSTEM_PROMPT, enabled as _lean_enabled
        _lean_on = _lean_enabled()
        _preamble_requested = (
            0 if _lean_on else nonnegative_env_int("REACT_PREAMBLE_TOKENS")
        )
        from kernel_probe_gate import probe_gate_mode
        _probe_gate_on = probe_gate_mode() == "on"
        from kernel_probe import observation_probe_mode
        _observation_probe_on = observation_probe_mode() == "on"
        from kernel_dependency_probe import dependency_fact_mode
        _dependency_fact_on = dependency_fact_mode() == "on"
        _base_tools = _tools_for_probe_modes(
            probe_gate_on=_probe_gate_on,
            observation_probe_on=_observation_probe_on,
            dependency_fact_on=_dependency_fact_on)
        from kernel_work_record import tool_surface as work_record_tool_surface
        from kernel_obligation_assess import mode as _host_closure_mode
        from kernel_sweep_fixes import enabled as _sweep_fixes_enabled
        _sweep_fixes_on = _sweep_fixes_enabled()
        _base_tools = work_record_tool_surface(
            _base_tools, intent_only=_host_closure_mode() == "on")
        # replace_module is advertised only on its own opt-in switch: the 2B called it 0/~12
        # times, yet the extra tool text measurably perturbed the trajectory (autopsy jq110).
        if (os.environ.get("STATEFRAME_JQ_FINISH", "off") == "on"
                and os.environ.get("STATEFRAME_JQ_REPLACE_MODULE", "off") == "on"):
            _base_tools = _base_tools + [_JQ_REPLACE_MODULE_TOOL]
        tool_surface = build_tool_surface(_base_tools, _decoy_count, kind=_decoy_kind)
        from kernel_decision_state import tool_surface as decision_tool_surface
        if not (_sweep_fixes_on and _host_closure_mode() == "on"):
            tool_surface = decision_tool_surface(tool_surface)
        else:
            tool_surface = [tool for tool in tool_surface
                            if tool.get("function", {}).get("name") not in {
                                "work_update", "note_update"}]
        from kernel_retained_context import tool_surface as retained_tool_surface
        tool_surface = retained_tool_surface(tool_surface)
        from kernel_edit_affordance import (contract as edit_contract,
                                            measure_product_sizes, tool_surface as edit_surface)
        from kernel_green_lock import is_product_path as _is_product_path
        _declared_product_paths = tuple(path for path in fileset.paths
                                        if _is_product_path(path))
        _product_sizes = (measure_product_sizes(root, _declared_product_paths)
                          if multifile and not (fileset.entry_path and fileset.entry_fn)
                          else None)
        _edit_contract = edit_contract(multi_file=multifile,
                                       entry_path=fileset.entry_path, entry_fn=fileset.entry_fn,
                                       target_fn=fn,
                                       product_paths=_declared_product_paths,
                                       product_sizes=_product_sizes)
        _definition_mode = bool(_edit_contract and
                                _edit_contract.get('mode') == 'declared_product_files')
        tool_surface = edit_surface(tool_surface, _edit_contract,
                                     authority=_edit_contract is not None)
        from kernel_decision_cache import tool_surface as decision_cache_tool_surface
        tool_surface = decision_cache_tool_surface(tool_surface)
        if os.environ.get("STATEFRAME_GET_SURE", "off").strip().lower() == "on":
            from kernel_stateframe_get_sure import tool_surface as get_sure_tool_surface
            tool_surface = get_sure_tool_surface(tool_surface)
        if _nonpy_edit_policy_enabled():
            tool_surface = _nonpy_neutralize_tool_surface(tool_surface)
        active_decoy_names = frozenset(decoy_names(_decoy_count, _decoy_kind))
        operational_prompt = os.environ.get('STATEFRAME_VIEW', 'existing') == 'operational'
        _entry_workspace_root = src_repo.parent if src_repo is not None else root
        from kernel_entry_authority import entry_file_from_ticket
        _entry_ticket_file = entry_file_from_ticket(
            _entry_ticket_text, workspace_root=_entry_workspace_root)
        _CONSTITUTION = (
            "\n\nBehavioral invariants:\n"
            "- Evidence outranks intuition; an assumption is not evidence.\n"
            "- When evidence contradicts your hypothesis, revise the hypothesis — "
            "do not defend it.\n"
            "- Root causes outrank symptoms.\n"
            "- Check the small details that could invalidate your conclusion; "
            "skip the ones that cannot.\n"
            "- Prefer the smallest complete change that addresses the cause.\n"
            "- Construct conclusions that would survive an attempt to disprove them.\n"
            "- Be decisive when evidence is sufficient; be skeptical when it is not."
        ) if (_system_prompt_variant == "v1"
              and os.environ.get("REACT_CONSTITUTION", "1") != "0") else ""
        _system_template = SYSTEM
        if _system_prompt_variant in ("v2", "v3") and not operational_prompt:
            _system_template = _system_template.replace(
                "  * The file you almost certainly need is `{REL}`, and the function is `{FN}`.",
                "  * The entry point is `{FN}` in `{REL}`. It is where the call chain starts, "
                "not the only file you may need to change.")
            _system_template = (
                _system_template.split("HOW TO WORK. You have {STEPS} steps.", 1)[0]
                + "You have {STEPS} steps.\n\n" + prompt_guidance(_system_prompt_variant))
        if not str(fn or "").strip():
            _system_template = render_blank_entry_template(
                _system_template,
                _entry_ticket_file,
                ticket_entry_file=_entry_ticket_file is not None,
            )
        _base_system = (_system_template.replace("{REL}", d["rel"]).replace("{FN}", fn)
                        .replace("You have {STEPS} steps.", step_guidance(step_cap))
                        .replace("{STEPS}", limit_text(step_cap)) + _CONSTITUTION)
        if _probe_gate_on:
            _base_system += (
                "\nProbe gate: when the working view lists an uncovered brief obligation, "
                "use the `probe` tool with its C-number, a short Python snippet calling "
                "the target, and an expectation copied from the brief. Host output is "
                "observed evidence; your expectation is only MODEL_INTERPRETATION.")
        if _observation_probe_on:
            _base_system += (
                "\nObservation probe: `probe` runs a short Python snippet in a fresh, "
                "read-only current-workspace snapshot. It is exploratory only: its "
                "status and output are host observations and never complete the task.")
        if _dependency_fact_on:
            _base_system += (
                "\nDependency fact: when a traceback crosses site-packages, use "
                "`dependency_fact` for one narrow characterization of the installed "
                "package/version and target. The result is descriptive HOST_OBSERVED "
                "evidence, not a normative compatibility claim or completion signal.")
        if operational_prompt and target is not None:
            from kernel_entry_authority import operating_instructions_for_ticket
            _base_system = operating_instructions_for_ticket(
                _entry_ticket_text,
                step_cap,
                workspace_root=_entry_workspace_root,
            ) + _CONSTITUTION
            if native_tool_batches:
                from kernel_native_tools import instructions as native_instructions
                _base_system = native_instructions(_base_system)
        from kernel_step_contract import INSTRUCTIONS, enabled as contract_enabled
        if action_schema_mode:
            _base_system += (
                "\nAction envelope: return exactly one visible JSON object with action and args. "
                "Choose action from the live menu in the request schema. Do not emit tool calls, "
                "Markdown, XML, or commands in reasoning/content. For done, claim must be one "
                "of complete, partial, or blocked and evidence_refs must contain only listed "
                "host references; summary is display text only.")
        elif contract_enabled():
            _base_system += "\n" + INSTRUCTIONS + "\nAvailable tools:\n" + json.dumps(tool_surface)
        from kernel_decision_state import instructions as decision_instructions
        if not (_sweep_fixes_on and _host_closure_mode() == "on"):
            _base_system = decision_instructions(_base_system)
        from kernel_edit_affordance import instructions as edit_instructions
        _base_system = edit_instructions(
            _base_system, _edit_contract, authority=_edit_contract is not None)
        from kernel_work_record import instructions as work_record_instructions
        _base_system = work_record_instructions(
            _base_system, intent_only=_host_closure_mode() == "on")
        from kernel_decision_cache import enabled as decision_cache_enabled
        if decision_cache_enabled():
            _base_system += (
                "\nOptional decision memory rides on this action. In the visible action envelope, "
                "use the optional top-level memory object only for ADOPT, CHALLENGE, or "
                "SUPERSEDE typed drafts. It is never required. Cite only receipt references "
                "present in the current StateFrame; malformed memory is ignored by the host."
            )
        from kernel_read_fixes import budget_unit_mode as _configured_budget_unit
        _configured_budget_mode = _configured_budget_unit()
        from kernel_tool_surface import budget_guidance_enabled
        if (_configured_budget_mode == "response"
                and budget_guidance_enabled()):
            _base_system += (
                "\nBudget accounting: one model response consumes one decision-budget unit. "
                f"All calls returned in one native batch share that one unit; the host still "
                f"{('does not cap total tool calls' if step_cap is None else f'caps total tool calls at {step_cap * 4}')}. Tool calls in the batch are not "
                "additional model-response budget units."
            )
        from kernel_tokenlab_v13 import system_trim_enabled as _system_trim_on, trim_system as _trim_system
        if _system_trim_on():
            # : final tool/schema policy decides which guidance is phantom.
            _surface_names = {
                str(tool.get("function", {}).get("name", ""))
                for tool in tool_surface if isinstance(tool, Mapping)
            }
            from kernel_decision_cache import enabled as _decision_cache_on
            _base_system, _trim_detail = _trim_system(
                _base_system,
                capabilities={
                    "work": bool(_surface_names.intersection({"work", "work_update"})),
                    "memory_sidecar": (_decision_cache_on()
                                        and os.environ.get("STATEFRAME_TOKENLAB_SIDECAR_OFF", "off").strip().lower() != "on"),
                    "boot_read": os.environ.get("STATEFRAME_BOOT_READ", "off").strip().lower() == "on",
                    "edit_authority": isinstance(_edit_contract, Mapping)
                    and isinstance(_edit_contract.get("edit_authority"), Mapping),
                },
            )
            _SYSTEM_TRIM_DETAIL[0] = _trim_detail
        if _nonpy_edit_policy_enabled():
            _base_system = _nonpy_neutralize_system(_base_system)
        _system = pad_system_prompt(_base_system, _preamble_requested)
        _actual_preamble = approx_tokens(render_neutral_preamble(_preamble_requested))
        regime = {
            "tag": os.environ.get("REACT_TAG", "") or "(base)",
            "real_tool_count": len(tool_surface) - _decoy_count,
            "tool_count": len(tool_surface),
            "decoy_kind": _decoy_kind,
            "decoy_count": _decoy_count,
            "decoy_names": list(decoy_names(_decoy_count, _decoy_kind)),
            "preamble_requested_tokens": _preamble_requested,
            "preamble_tokens": _actual_preamble,
        }
        _orientation = (_render_multifile_list(fileset)
                        if multifile and target is None else
                        f"The suspect function is `{fn}` in `{d['rel']}`.")
        if _real04_target_metadata:
            _orientation += "\n\n" + _real04_target_metadata
        try:
            from kernel_nav_basics import append_intake_orientation
            from kernel_repo_intel import intake_orientation
            _nav_selectors = tuple(_public_check_ids(d, fn))
            _target_relative = (target.relative_to(root).as_posix()
                                if target is not None else str(d.get("rel", "")))
            _intake_orientation = intake_orientation(
                root, target_path=_target_relative, symbol=fn, selectors=_nav_selectors)
        except (ImportError, OSError, TypeError, ValueError):
            _intake_orientation = ""
        msgs = [{"role": "system", "content": _system},
                {"role": "user", "content":
                    f"Bug report: {brief}\n\n{_orientation}\n\n"
                    f"{_intake_orientation}\nBegin." if _intake_orientation else
                    f"Bug report: {brief}\n\n{_orientation}\nBegin."}]
        if operational_prompt and target is not None:
            target_relative = target.relative_to(root).as_posix()
            if _edit_contract is not None:
                label = json.dumps({"path": target_relative, "function": fn}, sort_keys=True)
                msgs[1]['content'] = f"Task: {brief}\n\nentry_point: {label}."
            else:
                msgs[1]['content'] = (
                    f"Task: {brief}\n\nTarget: `{fn}` in `{target_relative}`.")
            if _real04_target_metadata:
                msgs[1]['content'] += "\n\n" + _real04_target_metadata
            if _intake_orientation:
                msgs = append_intake_orientation(msgs, _intake_orientation)
        if _lean_on:
            msgs = [
                {"role": "system", "content": PLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {brief}"},
            ]
            _claim_rule_rendered = False
        else:
            _claim_rule_rendered = render_claim_rule_first_turn(msgs, _entry_ticket_text)
        if _lean_on:
            from kernel_lean import BoundedToolMessages
            msgs = BoundedToolMessages(msgs)
        baseline_messages = [dict(message) for message in msgs]
        edits = tests = 0
        pending_edit_path = None      # file awaiting a code block (two-message edit form)
        unparsed_streak = 0           # consecutive no-command replies; 5 aborts as HARNESS_SUSPECT
        harness_suspect = None        # set by the dialect abort; graded tail still runs (#41)
        # IN-RUN TEMPERATURE UNSTICK (defect #37, second half). At temp 0 an identical
        # command answered identically is a deterministic fixed point — once entered, no
        # step budget escapes it. The kernel now varies its replay text (first half); if a
        # command STILL repeats 6+ times consecutively, bump temperature for the rest of the
        # attempt. This is the pilot's temp ladder applied in-run, before the budget burns.
        last_line, same_streak, run_temp = None, 0, None
        edits_since_test = 0          # E306 skeptic input: churn = edits with no verification
        # E310 — BASELINE TEST MANIFEST (design review, measured across all 47
        # transcripts: 0 of 6 winning runs ever edited an existing test file; losing runs
        # applied 37 such edits, 26 after red). Editing a PRE-EXISTING test is pure waste
        # here (grading overlays pristine tests) — reject outright. Tests the agent
        # CREATES this run (file_create) stay editable: that is legitimate repro work.
        baseline_tests = {str(p) for p in root.rglob("test_*.py")} \
            | {str(p) for p in root.rglob("conftest.py")}
        from kernel_shell_tool import service as shell_service
        _shell = shell_service()
        if _shell is not None:
            _shell.bind(root, d, baseline_tests)
            os.environ["REACT_ENVIRONMENT_REVISION"] = _shell.identity()
        from kernel_agent_tools import service as agent_tools_service
        _agent_tools = agent_tools_service()
        if _agent_tools is not None:
            _agent_tools.bind()
        created_this_run: set = set()
        action_anchor = len(msgs)     # conversation index just after the last real ACTION
        last_surgery_step = -99       # cooldown so back-to-back surgeries cannot thrash
        # KERNEL BOLT-ON. The agent DRIVES; these cylinders only answer deterministically,
        # remember what it cannot hold, or refuse a provably premature exit. None of them
        # chooses a fix, picks a file, or writes code — that is the multiply-not-replace line.
        # Baseline to diff against (v8, same six tasks, bolt-on ABSENT): 1/6, with
        # steps/edits/tests recorded per task.
        from kernel_bolt import KernelBolt
        bolt = KernelBolt(root, fn, d["sha"][:8])
        bolt._selector_error_fallback_files = set()
        _green_nag_runtime = SimpleNamespace(
            owner=SimpleNamespace(root=root, task_row=d),
        )
        # Cache observations about repository structure for subsequent steps.
        # invalidation is safe: re-observation at the new revision supersedes.
        try:
            from detreason.world_knowledge import WorldKnowledgeStore
            bolt.world = WorldKnowledgeStore(sandbox_id=d["sha"][:8])
        except Exception as _e:                                      # noqa: BLE001
            bolt.world = None
            print(f"    [kernel-wiring] world cache OFF: {type(_e).__name__}: {_e}")
        bolt.product_edits = 0        # E307: the done-gate's test-only-edits evidence
        bolt.target_path = d["rel"]   # E313: get_sure orientation target
        # E317 test-navigation oracle: names of the discriminating regression nodes and
        # which of them exist in THIS world (three-worlds: f2p tests are often held out).
        # Two SEPARATE fallbacks (#113: best-effort hooks log their failures — the
        # one-big-try version silently killed E317/E322 for a whole run): the name list
        # is cheap and must survive even when the in-world scan fails.
        try:
            _f2p = _public_check_ids(d, fn)
            if information.strict:
                from kernel_public_evidence import public_check_files
                _f2p = public_check_files(root, _f2p)
            bolt.f2p_names = tuple({i.split("::")[-1] for i in _f2p})
            bolt.f2p_ids = tuple(_f2p)                               # E448 done-gate
            bolt.oracle_file_ids = tuple(sorted({i.split("::", 1)[0] for i in _f2p}))
        except Exception as _e:                                      # noqa: BLE001
            bolt.f2p_names = ()
            bolt.f2p_ids = ()
            bolt.oracle_file_ids = ()
            print(f"    [kernel-wiring] f2p name resolution failed (nav oracle OFF): "
                  f"{type(_e).__name__}: {_e}")
        bolt.oracle_preflight = {} if information.strict else d.get("_oracle_preflight", {})
        bolt.f2p_inworld = ()
        bolt.oracle_selector_counts = {}
        if bolt.f2p_names:
            try:
                _inworld = set()
                for _oracle_id in bolt.f2p_ids:
                    _oracle_file = _oracle_id.split("::", 1)[0]
                    if "::" not in _oracle_id and (root / _oracle_file).is_file():
                        _inworld.add(_oracle_id)
                        _oracle_tree = ast.parse(
                            (root / _oracle_file).read_text(errors="replace"),
                            filename=_oracle_file,
                        )
                        bolt.oracle_selector_counts[_oracle_file] = len({
                            _node.name
                            for _node in ast.walk(_oracle_tree)
                            if isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and _node.name.startswith("test_")
                        })
                for _tf in root.rglob("test_*.py"):
                    _txt = _tf.read_text(errors="replace")
                    for _n in bolt.f2p_names:
                        if not _n.endswith(".py") and f"def {_n}" in _txt:
                            _inworld.add(f"{_tf.relative_to(root)}::{_n}")
                bolt.f2p_inworld = tuple(_inworld)
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] in-world f2p scan failed (held-out msgs "
                      f"only): {type(_e).__name__}: {_e}")
        bolt._rec("gate", "nav_armed",
                  f"names={len(bolt.f2p_names)} inworld={len(bolt.f2p_inworld)}")
        # E335 (measured on ack, commit 252c12de: "the test was masking it"): every
        # in-world f2p copy is by construction the PRE-FIX version — the corpus mints
        # fix commits that ship their tests, so the grading copy is always stronger.
        # Tell the model up front which contract is authoritative.
        if bolt.f2p_inworld:
            _e335_note = ("[kernel] NOTE: this checkout's copy of the regression test ("
                          + "; ".join(sorted(bolt.f2p_inworld)[:2]) +
                          ") predates the fix and may be WEAKER than the version used "
                          "for grading — a green on it is necessary but not sufficient. "
                          "The authoritative contract is the failure description in the "
                          "bug report above: make the code satisfy THAT.")
            if os.environ.get('STATEFRAME_VIEW', 'existing') == 'operational':
                _e335_note = ('Completion requires satisfying the public requirements and passing '
                              'the declared public checks. Those checks may not cover every '
                              'requirement; review the current implementation before done. '
                              'Private grading is unavailable to the solver.')
            # E367: welded into the brief so the trim can never remove it.
            if len(msgs) > 1 and msgs[1].get("role") == "user":
                msgs[1]["content"] = str(msgs[1]["content"]) + "\n\n" + _e335_note
            else:
                msgs.append({"role": "user", "content": _e335_note})
        # E365: semantic bridge (the four presolve producers; degrades to None cleanly).
        try:
            from kernel_semantics import SemanticBridge
            bolt.sem = SemanticBridge(d["sha"][:8], trace_cb=bolt._rec,
                                      organ_out=bolt.organ_out)
        except Exception as _e:                                      # noqa: BLE001
            bolt.sem = None
            print(f"    [kernel-wiring] semantic bridge OFF: {type(_e).__name__}: {_e}")
        # E350: vitals EKG (behavioral early warning; degrades to None cleanly).
        try:
            from kernel_vitals import VitalsEKG
            bolt.ekg = VitalsEKG()
        except Exception as _e:                                      # noqa: BLE001
            bolt.ekg = None
            print(f"    [kernel-wiring] vitals EKG OFF: {type(_e).__name__}: {_e}")
        # E342: hypothesis tracker (kill-list/frontier; degrades to None cleanly).
        try:
            from kernel_hypotheses import HypothesisTracker
            bolt.hyp = HypothesisTracker(d["sha"][:8], trace_cb=bolt._rec)
        except Exception as _e:                                      # noqa: BLE001
            bolt.hyp = None
            print(f"    [kernel-wiring] hypothesis tracker OFF: {type(_e).__name__}: {_e}")
        # E393: nit-picker (contract-atom attention; degrades to None cleanly).
        try:
            from kernel_nitpicker import NitpickerState
            bolt.nit = NitpickerState(task_id=d["sha"][:8], target_symbol=fn)
        except Exception as _e:                                      # noqa: BLE001
            bolt.nit = None
            print(f"    [kernel-wiring] nit-picker OFF: {type(_e).__name__}: {_e}")
        # E415: close-look organ (anchor-stability -> microscope dossier).
        try:
            from kernel_close_look import CloseLookKernel
            bolt.closelook = CloseLookKernel(task_id=d["sha"][:8])
        except Exception as _e:                                      # noqa: BLE001
            bolt.closelook = None
            print(f"    [kernel-wiring] close-look OFF: {type(_e).__name__}: {_e}")
        # E428: pivot organ (tiers + pre-graph + loop ban). The fallback order is
        # declared UP FRONT; only a new failing fingerprint moves us along an edge,
        # and a (approach, fingerprint) pair already burned is BANNED (no loops).
        try:
            from kernel_pivot import ApproachNode, ObjectiveTier, PivotGraph, PivotOrgan, PivotTier
            _pg = PivotGraph((
                ApproachNode("inside", f"fix the defect inside {fn} itself",
                             ObjectiveTier.PRIMARY_ORACLE, PivotTier.T1, ("contract",)),
                ApproachNode("contract", f"fix the target-side contract for {fn}",
                             ObjectiveTier.PRIMARY_ORACLE, PivotTier.T1, ("reread",)),
                ApproachNode("reread", "re-read the failing evidence + probe with a focused test",
                             ObjectiveTier.UNIT_GATE, PivotTier.T2),
            ), roots={ObjectiveTier.PRIMARY_ORACLE: "inside"})
            bolt.pivot = PivotOrgan(_pg)
        except Exception as _e:                                      # noqa: BLE001
            bolt.pivot = None
            print(f"    [kernel-wiring] pivot OFF: {type(_e).__name__}: {_e}")
        # E429: ignored-advice ledger — organ_out records deliveries; each action
        # settles them (followed / ignored / overtaken); twice-ignored escalates once.
        try:
            from kernel_advice_ledger import AdviceLedger
            bolt.advice = AdviceLedger()
            bolt._advice_escalated_seen = set()
        except Exception as _e:                                      # noqa: BLE001
            bolt.advice = None
            print(f"    [kernel-wiring] advice-ledger OFF: {type(_e).__name__}: {_e}")
        # E444: idle governor — the first ENFORCED organ (autopsy_sw2: 145 idle turns
        # while every advisory organ fired and was ignored). Warn at 8 idle
        # non-product actions, GOVERN at 14: reads/greps are refused until one
        # product action lands. Edits and tests are never denied.
        try:
            from kernel_gate_switches import idle_governor_enabled
            from kernel_idle_governor import IdleGovernor
            # STATEFRAME_IDLE_GOVERNOR=off disables the read budget for ablations;
            # an invalid value raises (caught below would hide it, so check first).
            bolt.idlegov = IdleGovernor() if idle_governor_enabled() else None
        except ValueError:
            raise
        except Exception as _e:                                      # noqa: BLE001
            bolt.idlegov = None
            print(f"    [kernel-wiring] idle-governor OFF: {type(_e).__name__}: {_e}")
        bolt.information_policy = information
        bolt.task_row = ({key: value for key, value in d.items()
                          if key not in {'fixed_src', 'test_patch', '_oracle_preflight',
                                         'FAIL_TO_PASS', 'PASS_TO_PASS', 'fail_to_pass', 'pass_to_pass'}}
                         if information.strict else d)
        if _staged_enabled:
            bolt.task_row = _visible_staged_row(bolt.task_row)
        # E467: micro-reflection gate (feast #9/#10) — deterministic triggers +
        # prediction-vs-reality; signals only, never steering.
        try:
            from kernel_reflection import ReflectionGate
            bolt.reflect = ReflectionGate()
        except Exception as _e:                                      # noqa: BLE001
            bolt.reflect = None
            print(f"    [kernel-wiring] reflection OFF: {type(_e).__name__}: {_e}")
        # E491: changed-condition retry gate (gap-map rank 3).
        try:
            from kernel_retry_gate import RetryGate
            bolt.retrygate = RetryGate()
        except Exception as _e:                                      # noqa: BLE001
            bolt.retrygate = None
            print(f"    [kernel-wiring] retry-gate OFF: {type(_e).__name__}: {_e}")
        # WAVE 3 / D3: pre-dispatch retry gate. Defaults OFF (KERNEL_MODE unset) --
        # current baseline is unchanged (retry_gate above remains the only, post-hoc,
        # detector). Set KERNEL_MODE="retry_predispatch:ENFORCE" to block dispatch.
        try:
            from kernel_wave3 import PreDispatchGate
            bolt.predispatch_gate = (PreDispatchGate(RetryGate())
                                     if bolt.retrygate is not None else None)
        except Exception as _e:                                      # noqa: BLE001
            bolt.predispatch_gate = None
            print(f"    [kernel-wiring] predispatch-gate OFF: {type(_e).__name__}: {_e}")
        # E484: cognitive governor — doses organ chatter by measured difficulty.
        try:
            from kernel_cog_governor import CognitiveGovernor
            bolt.cogov = CognitiveGovernor()
        except Exception as _e:                                      # noqa: BLE001
            bolt.cogov = None
            print(f"    [kernel-wiring] cog-governor OFF: {type(_e).__name__}: {_e}")
        # E525: persistent obligation graph (in-world oracle ids only, E513) so the
        # green->red REGRESSION tripwire can watch every test receipt, not just done.
        try:
            from kernel_obligations import compile_obligation_graph as _cog
            _inw = [i for i in (getattr(bolt, "f2p_ids", ()) or ())
                    if (root / str(i).split("::")[0]).exists()]
            bolt.obgraph = _cog(d, oracle_ids=_inw)
        except Exception as _e:                                      # noqa: BLE001
            bolt.obgraph = None
            print(f"    [kernel-wiring] obligation graph OFF: {type(_e).__name__}: {_e}")
        # E470: dependency-invalidation cascade (gap-map rank 2, "make for reasoning")
        # — an edit dirties every belief whose evidence predates it; dirty beliefs get
        # ONE bounded re-observe notice (max 2/turn, deduped).
        try:
            from kernel_invalidation import LedgerIndex
            bolt.inval = LedgerIndex()
        except Exception as _e:                                      # noqa: BLE001
            bolt.inval = None
            print(f"    [kernel-wiring] invalidation OFF: {type(_e).__name__}: {_e}")
        # E456: measure-before-infer — an edit whose stated
        # rationale asserts an UNMEASURED workspace fact gets the ONE cheap probe
        # that would ground it, advisory-only.
        try:
            from kernel_groundtruth import MeasureBeforeInfer
            bolt.gt = MeasureBeforeInfer()
        except Exception as _e:                                      # noqa: BLE001
            bolt.gt = None
            print(f"    [kernel-wiring] groundtruth-B OFF: {type(_e).__name__}: {_e}")
        # E417: expectation ledger (every count the agent sees
        # becomes a tripwire; deltas demand investigation).
        # r2b: organs are wired BEFORE StateFrameRuntime exists (measurement is None here); buffer their state events
        # and flush them into the run journal right after the runtime is constructed (930/931 startup crash).
        _wiring_journal = _PendingWiringJournal()
        _expectation_wiring = _wire_expectation_ledger(_wiring_journal)
        bolt.expect = _expectation_wiring.value
        if _expectation_wiring.effective == "failed":
            print("    [kernel-wiring] expectation ledger OFF: "
                  f"{_expectation_wiring.error_type}")
        # E423: composed cognition loop (fold->hypotheses->replan->selector->card).
        # Generic per-task assembly: scope from task metadata; acceptance atoms from
        # the oracle's f2p nodes; seed hypotheses from the target seam (the model's
        # own theories join via bolt.note as the run progresses); guard-anchor probes
        # from the target file/fn (deterministic candidates only, per the design).
        _composed_wiring = _wire_composed_cognition(d, fn, bolt, _wiring_journal)
        bolt.composed = _composed_wiring.value
        if _composed_wiring.effective == "failed":
            print("    [kernel-wiring] composed cognition OFF: "
                  f"{_composed_wiring.error_type}")
        # E377: corpus_fixes overlay — when a task's in-world test is PROVEN misleading
        # (E373: watchdog's rewards the wrong edit) a verified replacement (two-sided:
        # parent FAIL / gold PASS) lands in the sandbox as an honest in-world oracle.
        # Grading still uses the ORIGINAL hidden overlay — guidance fixed, bar unchanged.
        _corpus_wiring = _wire_corpus_fixes(POC, root, d, resume, _wiring_journal)
        _n_fix = int(_corpus_wiring.value or 0) if isinstance(_corpus_wiring.value, int) else 0
        if _corpus_wiring.effective == "failed":
            print("    [kernel-wiring] corpus_fixes overlay skipped: "
                  f"{_corpus_wiring.error_type}")
        elif _corpus_wiring.reason == "not_shipped":
            print("    [kernel-wiring] corpus_fixes overlay off: not_shipped")
        if _n_fix:
            bolt._rec("gate", "corpus_fix_applied", f"files={_n_fix}")
            # refresh E310 baseline + register the honest oracle's nodes as
            # RELEVANT (else E317/E322 would fight the model for using it)
            baseline_tests |= {str(p) for p in root.rglob("test_*.py")}
            import re as _re2
            for _of in (pathlib.Path("corpus_fixes") / fn).glob("test_*.py") \
                    if (pathlib.Path("corpus_fixes") / fn).is_dir() else []:
                _names = _re2.findall(r"^def (test_\w+)", _of.read_text(), _re2.M)
                bolt.f2p_names = tuple(set(bolt.f2p_names) | set(_names))
                _sb = next(iter(root.rglob(_of.name)), None)
                if _sb:
                    bolt.f2p_inworld = tuple(set(bolt.f2p_inworld) | {
                        f"{_sb.relative_to(root)}::{n}" for n in _names})
        # E365d: the bridge's evidence is SOURCE-derived — firing it only on an in-world
        # red left it dormant on held-out-oracle tasks (measured: 0 firings in a full
        # rss run). Derive ONCE at task start from the parent source + a metadata
        # trigger, and put the evidence-gap advisory straight into the brief.
        if getattr(bolt, "sem", None):
            try:
                _pmap0 = {"fetch_rss_feed": "rss",
                          "_watchdog_auto_pause_trigger": "watchdog",
                          "require_isolated_f2_database": "db",
                          "_ack_decision": "ack"}
                _src0 = (root / d["rel"]).read_text(errors="replace") \
                    if (root / d["rel"]).is_file() else ""
                _trigger0 = [{"kind": "test", "intent": "route", "fingerprint": "boot",
                              "result_class": "RED", "why": "", "detail": "boot-trigger",
                              "turn": 0, "workspace_revision": 0, "ts": 0,
                              "anchor": f"{d['rel']}:{d.get('changed_line', 1)}:{fn}"}]
                _ev0 = bolt.sem.derive(_trigger0, _src0, producer=_pmap0.get(fn))
                _line0 = _render_semantic_facts(_ev0)
                if _line0:
                    _line0 = bolt.organ_out("semantic", _line0)
                if _line0:
                    # E367: weld the facts INTO the brief (msgs[1]) — the trim protects
                    # only msgs[:2], and a facts message appended after it was measured
                    # being trimmed away mid-run (require lost it by s21).
                    if len(msgs) > 1 and msgs[1].get("role") == "user":
                        msgs[1]["content"] = str(msgs[1]["content"]) + "\n\n" + _line0
                    else:
                        msgs.append({"role": "user", "content": _line0})
                    bolt._rec("semantic", "boot_facts_emitted", _line0[:120])
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] boot semantic derive failed: {type(_e).__name__}")
        # E388: GENERIC fact compiler (contract-directed abstract interpreter,
        # with family-agnostic DECISION+EFFECT facts for tasks the four
        # legacy producers do not cover. Same E367 weld; FACT_COMPILER=0 is the A/B
        # control switch. Parent-src only (no gold), deterministic, CPU-cheap.
        if os.environ.get("FACT_COMPILER", "1") == "1" and fn not in _pmap0:
            try:
                from kernel_semantic_ir import AllowedTaskView
                from kernel_fact_compiler import (compile_contract_facts,
                                                  render_contract_facts)
                _gf = compile_contract_facts(AllowedTaskView.from_result_row(d))
                # E394 (adv delta autopsy): default 16 cells buried the signal in
                # "unrelated helper trivia" and helped freeze a run — attention is
                # the point, so the weld carries only the highest-signal few.
                # E398 (advS 200st/0-edit freeze): UNKNOWN/UNAVAILABLE cells NEVER
                # reach the brief — five welded "nothing is provable" blocks put the
                # model into 200 turns of pure reading. Lazy-UNKNOWN law (E369/E370)
                # applies to the WELD, not just the producer: few, PROVEN, or nothing.
                try:
                    from kernel_semantic_ir import SemanticState as _SState
                    import dataclasses as _dc
                    _keep = tuple(c for c in _gf.cells
                                  if c.state not in (_SState.UNKNOWN,))
                    _gf = _dc.replace(_gf, cells=_keep) if _keep != _gf.cells else _gf
                except Exception:                                    # noqa: BLE001
                    pass
                _gline = render_contract_facts(_gf, max_cells=5) if _gf.cells else None
                if _gline:
                    _gline = bolt.organ_out("fact_compiler", _gline)
                if _gline:
                    if len(msgs) > 1 and msgs[1].get("role") == "user":
                        msgs[1]["content"] = str(msgs[1]["content"]) + "\n\n" + _gline
                    else:
                        msgs.append({"role": "user", "content": _gline})
                    bolt._rec("semantic", "generic_facts_emitted", _gline[:120])
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] fact compiler OFF: {type(_e).__name__}: {_e}")
        # E408: CROSS-RUN FAILURE MEMORY — amnesia between
        # attempts is over. Prior attempts of THIS (fn, sha) distill into a bounded
        # "PRIOR ATTEMPTS" capsule (tried theories, failed legs, terminal shapes,
        # unexplored space) welded into the brief. Agent-world sources only (gold-leak
        # proven in its tests); attention-budgeted; empty when no priors.
        if not information.strict and os.environ.get("CROSSRUN_MEMORY", "1") == "1":
            try:
                from kernel_memory_isolation import enabled as _memory_isolation_enabled
                if _memory_isolation_enabled():
                    # Typed envelope only; the StateFrame runtime binds it to its
                    # own namespace and renders it.  Nothing is welded here.
                    from kernel_memory_isolation import (
                        build_prior_carry_envelope, worker_identity_from_row,
                    )
                    _cap = None
                    _carry_diag: dict = {}
                    _env = build_prior_carry_envelope(
                        str(POC / "results"), fn, d.get("sha", ""), 10**6,
                        row=d, worker_id=worker_identity_from_row(d), max_tokens=800,
                        diagnostics=_carry_diag)
                    if _carry_diag.get("filtered"):
                        bolt.memory_carry_filtered = dict(_carry_diag)  # r44 M6: journaled by the runtime
                    if _env and bolt.organ_out("crossrun", _env["records"][0]["value"]):
                        bolt.memory_carry_envelope = _env
                        bolt._rec("crossrun", "prior_capsule_enveloped",
                                  str(_env["source"]["task_id"])[:120])
                else:
                    from detreason.attempt_memory import render_prior_capsule
                    _cap = render_prior_capsule(
                        str(POC / "results"), fn, d.get("sha", ""),
                        current_attempt=10**6, max_tokens=800)
                if _cap:
                    _cap = bolt.organ_out("crossrun", _cap)
                if _cap:
                    if len(msgs) > 1 and msgs[1].get("role") == "user":
                        msgs[1]["content"] = str(msgs[1]["content"]) + "\n\n" + _cap
                    else:
                        msgs.append({"role": "user", "content": _cap})
                    bolt._rec("crossrun", "prior_capsule_welded", _cap[:120])
            except Exception as _e:                                  # noqa: BLE001
                print(f"    [kernel-wiring] cross-run memory OFF: {type(_e).__name__}: {_e}")
        # E426: UNIT PLANNER (the horizon unlock, E425) — multi-unit build tasks get a
        # derived sequence (Job -> JobQueue -> QueueStats) with per-unit test gates and
        # the handoff rule welded into the brief; single-unit tasks emit nothing (E394).
        if d.get("repo") == "build":
            try:
                from unit_planner import derive_unit_plan, render_plan_block, joint_handoffs_enabled
                _uplan = derive_unit_plan(d, joint_handoffs=joint_handoffs_enabled())
                bolt.unit_plan = _uplan
                if len(getattr(_uplan, "units", ())) > 1:
                    _pblock = bolt.organ_out("unit_plan", render_plan_block(_uplan))
                    if _pblock:
                        if len(msgs) > 1 and msgs[1].get("role") == "user":
                            msgs[1]["content"] = str(msgs[1]["content"]) + "\n\n" + _pblock
                        else:
                            msgs.append({"role": "user", "content": _pblock})
                        bolt._rec("unit_plan", "plan_welded",
                                  f"units={len(_uplan.units)}")
            except Exception as _e:                                  # noqa: BLE001
                bolt.unit_plan = None
                print(f"    [kernel-wiring] unit planner OFF: {type(_e).__name__}: {_e}")
        # E334: per-task receipt tape (append-only JSONL, telemetry-only).
        try:
            _rdir = POC / "results" / ("receipts" + os.environ.get("REACT_TAG", ""))
            _rdir.mkdir(parents=True, exist_ok=True)
            bolt.receipts_path = str(_rdir / f"{fn}.jsonl")
        except OSError as _e:
            print(f"    [kernel-wiring] receipt tape unavailable: {_e}")
        # E346 boot fingerprint — AFTER every organ init so it cannot lie (and the
        # E351 lesson: it crashed the whole wave by reading bolt.hyp before init;
        # the observability line must never be able to kill the run it observes).
        try:
            bolt._rec("gate", "boot_fingerprint",
                      f"shadow={sorted(s for s in os.environ.get('KERNEL_SHADOW', '').split(',') if s.strip()) or '[]'} "
                      f"hyp={'on' if getattr(bolt, 'hyp', None) else 'OFF'} "
                      f"ekg={'on' if getattr(bolt, 'ekg', None) else 'OFF'} "
                      f"nav={'on' if bolt.f2p_names else 'OFF'} "
                      f"receipts={'on' if getattr(bolt, 'receipts_path', None) else 'OFF'}")
        except Exception as _e:                                      # noqa: BLE001
            print(f"    [kernel-wiring] fingerprint failed: {type(_e).__name__}")
        toolcall_mode = (os.environ.get("REACT_TOOLCALL", "1") == "1"
                         or action_schema_mode)
        tool_calls_seen = 0
        from kernel_read_fixes import ContinuationLimits
        from kernel_measurement import output_token_limits as _output_token_limits
        _budget_mode = _configured_budget_mode
        _, _output_ceiling = _output_token_limits()
        _sandbox_deadline = None
        for _deadline_source in (
                d.get("task_deadline"),
                d.get("deadline_monotonic"),
                d.get("attempt_deadline"),
                os.environ.get("CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC")):
            if _deadline_source is None:
                continue
            try:
                _sandbox_deadline = float(_deadline_source)
            except (TypeError, ValueError):
                _sandbox_deadline = None
            if _sandbox_deadline is not None:
                break
        if _sandbox_deadline is None:
            # External execution deadlines must agree with the budget shown to the model.
            # "sandbox_seconds_remaining=unbounded". The live env states the run wall (relative seconds); the
            # deadline is measured from this runtime's start, which slightly overstates nothing (worker start
            # precedes it).
            try:
                _wall = float(os.environ.get("STATEFRAME_RUN_WALL_SECONDS", "") or "nan")
            except ValueError:
                _wall = float("nan")
            if _wall == _wall and _wall > 0:
                _sandbox_deadline = time.monotonic() + _wall
        _budget = ContinuationLimits(
            step_cap,
            mode=_budget_mode,
            arm="shared",
            automatic_check_limit=step_cap,
            terminal_reserve_limit=2,
            output_reservation_tokens=_output_ceiling,
            sandbox_deadline_monotonic=_sandbox_deadline,
        )
        decoy_call_names: list[str] = []
        schema_violation_details: list[dict[str, object]] = []
        unknown_tool_names: list[str] = []
        first_edit_step = None
        wrong_tool_before_first_edit = False
        if not interventions:
            msgs[:] = baseline_messages
        from kernel_stateframe_runtime import StateFrameRuntime
        from kernel_measurement import ACTIVE_REQUEST
        _working_backwards = d.get("_working_backwards")
        if isinstance(_working_backwards, Mapping):
            from kernel_working_backwards import run_and_attach_packet
            run_and_attach_packet(
                bolt,
                _working_backwards.get("tasks", (d,)),
                _working_backwards.get("worker"),
                analysis=_working_backwards.get("analysis"),
            )
        # (benchmark r2j: 4 "changed" files with zero applied edits): net product
        # change is measured against the tree the run actually started from, not the
        # row's declared parent text (which can differ byte-wise from the materialized repo).
        _initial_product_sources = None
        if resume is None and isinstance(root, pathlib.Path):
            try:
                if multifile:
                    _initial_paths = _product_source_paths(root, fileset, d)
                elif isinstance(target, pathlib.Path):
                    _initial_relative = target.relative_to(root).as_posix()
                    _initial_paths = (_initial_relative,)
                else:
                    _initial_paths = ()
                if _initial_paths:
                    _initial_product_sources = _source_snapshot(root, _initial_paths)
            except (OSError, TypeError, ValueError):
                _initial_product_sources = None
        measurement = StateFrameRuntime(
            bolt, output=POC / "measurement-runs", target=target, model=MODEL,
            tools=tool_surface if toolcall_mode else None,
            temperature=TEMPERATURE, source_resolver=fn_src,
            step_limit=_budget.tool_attempt_limit,
            multi_file=multifile, edit_contract=_edit_contract, workspace=continuation_workspace, resume=resume)
        if isinstance(fixture_initial_src, str) and isinstance(target, pathlib.Path):
            # Seed only after parent capture to reproduce a host/source divergence
            # that has no corresponding StepLedger edit in the current run.
            target.write_text(fixture_initial_src, encoding="utf-8")
            refresh_view = getattr(measurement, "current_view", None)
            if callable(refresh_view):
                refresh_view("external")
            invalidate_binding = getattr(measurement, "_invalidate_binding_cache", None)
            if callable(invalidate_binding):
                invalidate_binding()
            measurement.prepared_binding = measurement.binding()
        _wiring_journal.flush(measurement.journal)
        _edit_nturn_session = None
        _edit_nturn_request_active = False
        _edit_nturn_model_sites: list[object] = []
        _edit_nturn_model_plan_record_id = ""
        _edit_nturn_verification_reserve = 0.0
        _edit_nturn_deadline = None
        if _edit_nturn_enabled:
            _edit_nturn_verification_reserve = edit_nturn_verification_reserve_seconds(d)
            _edit_nturn_deadline = (
                _edit_nturn_run_started + EDIT_NTURN_RUN_BUDGET_SECONDS)
            if _sandbox_deadline is not None:
                _edit_nturn_deadline = min(_edit_nturn_deadline, _sandbox_deadline)
        if _claim_rule_rendered:
            emit_claim_rule_rendered(measurement.journal)
        _budget.journal = measurement.journal
        measurement.decision_budget = _budget
        measurement.continuation_limits = _budget
        def _decide2_provider(body, *, side_call_contract=None):
            """Use the existing captured transport for host-owned DECIDE2 calls."""
            from collections.abc import Mapping as _Mapping
            from kernel_measurement import RequestCapture
            from kernel_pilot import charge

            charge("model_calls", "DECIDE2 host decision checkpoint")
            capture = RequestCapture(
                measurement.journal,
                getattr(measurement.owner, "current_step", 0),
                measurement.snapshot_id,
                call_purpose="decide2",
                endpoint_profile=getattr(measurement, "endpoint_profile", None),
            )
            from kernel_request_kind import bind_auxiliary
            bind_auxiliary(capture, "decide2")
            choice = capture.send(dict(body), API, KEY, side_call_contract=side_call_contract)
            if isinstance(choice, _Mapping):
                message = choice.get("message")
                if isinstance(message, _Mapping):
                    return message.get("content") or message.get("reasoning_content") or choice
            return choice

        measurement.decide2_provider = _decide2_provider
        try:
            from kernel_v46_mechanism_wiring import (
                sanitized_error_message as _v46_sanitized_error_message,
                task_start as _v46_task_start,
            )
            _v46_task_start(measurement, d, pathlib.Path(root))
        except Exception as _error:  # noqa: BLE001 - mechanism adapters are fail-open
            measurement.journal.emit(
                "v46_mechanism_wiring_error",
                producer="kernel_v46_mechanism_wiring",
                phase="task_start",
                status="error",
                error_type=type(_error).__name__,
                message=_v46_sanitized_error_message(_error),
                dispatch=False,
            )
        measurement.long_horizon_phase = _current_staged_phase
        if (os.environ.get("LONG_HORIZON_TELEMETRY", "off").strip().lower()
                in {"on", "1", "true", "yes"}):
            from long_horizon_telemetry import AppendOnlyLogicalDepth
            measurement.long_horizon_depth_tracker = AppendOnlyLogicalDepth()
            measurement.long_horizon_depth_tracker.observe(msgs)
        if (step_cap is None or os.environ.get("STATEFRAME_TOOL_CALL_LOOP_BREAKER", "off").strip().lower()
                in {"on", "1", "true", "yes"}):
            from kernel_tool_call_loop_breaker import ToolCallLoopBreaker
            measurement._tool_call_loop_breaker = ToolCallLoopBreaker.from_environment()
        if _budget_mode == "response":
            measurement.journal.emit("budget_configured", **_budget.snapshot(),  # journal-safe: test_budget_snapshot_has_no_reserved_keys
                                     history_and_frame_shared=True,
                                     internal_step_limit=_budget.tool_call_cap,
                                     count_location="probes/react_repair.py before ask_tools/ask",
                                     tool_call_location="probes/react_repair.py after native selection",
                                     continuation_limits=_budget.continuation_snapshot())
        from kernel_stall_observation import observe_stall_in_place
        from kernel_lean import (
            enabled as lean_enabled, history_compaction_enabled,
            should_deliver_zero_progress_advisory,
        )
        from kernel_progress_facts import (
            derive_host_progress_facts, host_truth_progress_enabled,
        )
        _host_truth_progress_enabled = host_truth_progress_enabled()
        _lean_run = lean_enabled()
        # HISTORY_COMPACTION is deliberately a history-arm-only presentation hook.  The
        # enforced frame arm already chooses a bounded model view; giving it a second,
        # model-authored transcript would change the frame experiment and make the arms
        # incomparable.  The OFF branch does not import or call any compaction code.
        history_compactor = None
        _history_switch = os.environ.get("REACT_HISTORY_COMPACTION", "off").strip().lower()
        if _lean_run and _history_switch in {"on", "1", "true", "yes"}:
            measurement.journal.emit(
                "lean_history_compaction_suppressed", producer="kernel_lean",
                request_id=getattr(measurement, "current_request_id", None),
                switch="REACT_HISTORY_COMPACTION", reason="append_only_history",
                dispatch=False,
            )
        if history_compaction_enabled():
            from kernel_effect_boundary import AXES
            _history_arm = not AXES[measurement.modes["render_selection"]][2]
            if _history_arm:
                from history_compaction import (HistoryCompactor,
                                                summary_request)
                from kernel_measurement import RequestCapture

                def _summarize_history(summary_prompt):
                    _capture = RequestCapture(
                        measurement.journal,
                        getattr(measurement.owner, "current_step", 0),
                        measurement.snapshot_id,
                        endpoint_profile=getattr(measurement, "endpoint_profile", None),
                    )
                    _capture.long_horizon_request_role = "summary"
                    from kernel_request_kind import bind_auxiliary
                    bind_auxiliary(_capture, "summary")
                    _bind = getattr(measurement, "_bind_long_horizon_capture", None)
                    if callable(_bind):
                        _bind(_capture, presentation="history")
                    # The attempt deadline is a LONG_HORIZON_PROTOCOL-only behavior;
                    # protocol off keeps the pre-lane call exactly (fix round, review #4).
                    if os.environ.get("LONG_HORIZON_PROTOCOL", "off").strip().lower() in {
                            "on", "1", "true", "yes"}:
                        _summary_deadline = None
                        _raw_summary_deadline = os.environ.get(
                            "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC")
                        if _raw_summary_deadline is not None:
                            try:
                                _summary_deadline = float(_raw_summary_deadline)
                            except (TypeError, ValueError):
                                _summary_deadline = None
                        _choice = _capture.send(
                            summary_request(summary_prompt, model=MODEL), API, KEY,
                            deadline=_summary_deadline)
                    else:
                        _choice = _capture.send(
                            summary_request(summary_prompt, model=MODEL), API, KEY)
                    measurement.journal.emit(
                        "history_compaction_accounting",
                        request_id=_capture.request_id,
                        response_event_id=_capture.response_event_id,
                        accounting=copy.deepcopy(_capture.accounting),
                        counted_in_run_totals=True,
                    )
                    _usage = _capture.accounting.get("raw_usage", {})
                    _completion_tokens = (
                        _usage.get("completion_tokens")
                        if isinstance(_usage, Mapping)
                        and type(_usage.get("completion_tokens")) is int
                        and _usage.get("completion_tokens") >= 0
                        else None
                    )
                    _result = {"choice": _choice,
                               "request_id": _capture.request_id}
                    if _completion_tokens is not None:
                        _result["summary_tokens"] = _completion_tokens
                    return _result

                history_compactor = HistoryCompactor.from_environment(
                    summarize=_summarize_history,
                    charge=lambda category, reason: measurement.parent.charge(
                        category, reason=reason),
                    journal=measurement.journal,
                )
        if __import__("kernel_repro_first").repro_first_gate_mode() != "off":
            from kernel_repro_first_coding import attach_repro_first_gate
            _repro_ticket = dict(d)
            if isinstance(_grading_row, Mapping):
                for _ticket_key in (
                        "ticket_type", "issue_type", "task_type", "category", "kind", "type",
                        "is_bug", "is_regression"):
                    if _ticket_key in _grading_row:
                        _repro_ticket[_ticket_key] = copy.deepcopy(_grading_row[_ticket_key])
            attach_repro_first_gate(
                measurement, ticket=_repro_ticket, root=root,
                # Public preflight is context only; REPROFIRST receipts may
                # come only from the model-authored scratch action.
                initial_probe=None, discipline=True)

        def _run_repro_check(source: str, scratch_rel: str):
            """Run transient model source through the exact public sandbox runner."""
            from kernel_repro_first_runtime import SCRATCH_DIR, is_scratch_path
            scratch = (root / scratch_rel).resolve()
            root_resolved = root.resolve()
            try:
                scratch.relative_to(root_resolved)
            except ValueError as error:
                raise ValueError("reproduction scratch path escaped workspace") from error
            if not is_scratch_path(scratch.relative_to(root_resolved).as_posix()):
                raise ValueError("reproduction runner received a non-scratch path")
            scratch.parent.mkdir(parents=True, exist_ok=True)
            if scratch.exists():
                raise FileExistsError("reproduction scratch path collision")
            scratch.write_text(source, encoding="utf-8")
            source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
            prior_context = getattr(measurement, "_repro_first_observation_context", None)
            measurement._repro_first_observation_context = {
                "command": "pytest:" + SCRATCH_DIR + "/reproduction.py",
                "probe_id": "repro-first:" + source_digest,
                # The adapter records exactly once after the sandbox returns;
                # this context only prevents ordinary public checks from
                # entering the gate.
                "observe_in_runner": False,
            }
            # ``requested_check_targets`` strips leading ``./`` and dots from
            # selectors.  Route the hidden scratch path through a temporary
            # non-hidden parent so the public selector resolves to the actual
            # file without weakening the scratch-path boundary.
            selector_parent = root / SCRATCH_DIR.lstrip(".")
            selector_parent.mkdir(parents=True, exist_ok=True)
            scratch_selector = (
                f"{SCRATCH_DIR.lstrip('.')}/../"
                f"{scratch.relative_to(root_resolved).as_posix()}"
            )
            try:
                observation = run_tests(
                    root, fn, scratch_selector,
                    d["rel"], row=d)
                match = re.search(r"pytest\s+exit\s+(-?\d+)", observation or "", re.I)
                return {
                    "observation": observation,
                    "exit_code": int(match.group(1)) if match else None,
                }
            finally:
                try:
                    selector_parent.rmdir()
                except OSError:
                    pass
                measurement._repro_first_observation_context = prior_context
                try:
                    scratch.unlink()
                finally:
                    parent = scratch.parent
                    while parent != root_resolved and parent.name == SCRATCH_DIR:
                        try:
                            parent.rmdir()
                        except OSError:
                            break
                        parent = parent.parent

        # IDEALPATH S3 uses the same isolated runner as the legacy REPROFIRST
        # adapter; only the source producer changes (typed host compiler).
        measurement.configure_idealpath_typed_check_runner(_run_repro_check)

        def _reproduce_feedback(arguments: Mapping[str, object], call_id: str | None) -> None:
            """Dispatch REPROFIRST before generic action gates can strand its typed result."""
            from kernel_repro_first_runtime import execute_reproduction, render_result
            if not isinstance(arguments, Mapping):
                result = {
                    "status": "REJECTED",
                    "code": "REPRO_SOURCE_REQUIRED",
                    "detail": "use the structured reproduce tool with complete Python source",
                }
            else:
                result = execute_reproduction(
                    measurement, root=root, arguments=arguments, runner=_run_repro_check)
            latest_probe = getattr(getattr(measurement, "repro_first_gate", None), "_latest_probe", None)
            if latest_probe is not None:
                result = {
                    **result,
                    "probe": {
                        "status": getattr(latest_probe, "status", None),
                        "outcome": getattr(latest_probe, "outcome", None),
                        "failure_phase": getattr(latest_probe, "failure_phase", None),
                        "exit_code": getattr(latest_probe, "exit_code", None),
                        "output": str(getattr(latest_probe, "output", ""))[:500],
                    },
                }
            measurement.journal.emit(
                "idealpath_reproduction_result",
                producer="kernel_idealpath",
                result=result,
                dispatch=False,
            )
            observation = render_result(result)
            bolt.receipt(
                "reproduce", str(result.get("probe_id") or "ticket_reproduction"),
                str(result.get("scratch_path") or ""),
                str(result.get("code") or result.get("status") or "REJECTED"),
                detail=str(result.get("detail") or "")[:240],
            )
            measurement.observe("reproduce", observation)
            msgs.append(_tool_reply_for_call(msgs, call_id, observation)
                        if call_id else {"role": "user", "content": observation})

        system_prompt_audited = False

        def _audit_system_prompt(messages=None):
            nonlocal system_prompt_audited
            if system_prompt_audited:
                return
            system_text = "\n".join(
                str(message.get("content"))
                for message in (messages or ())
                if message.get("role") == "system"
                and isinstance(message.get("content"), str))
            if not system_text:
                system_text = _system
            measurement.journal.emit(
                "system_prompt_variant",
                variant=_system_prompt_variant,
                sha256=hashlib.sha256(system_text.encode()).hexdigest(),
            )
            system_prompt_audited = True

        observation_probe_runner = None
        dependency_fact_runner = None
        if _observation_probe_on:
            from kernel_probe import ProbeRunner
            _deadline_value = None
            for _deadline_key in (
                    "task_deadline", "deadline_monotonic", "attempt_deadline",
                    "deadline", "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"):
                _candidate_deadline = d.get(_deadline_key)
                if _candidate_deadline is not None:
                    try:
                        _deadline_value = float(_candidate_deadline)
                    except (TypeError, ValueError):
                        _deadline_value = None
                    if _deadline_value is not None:
                        break
            if _deadline_value is None:
                try:
                    _deadline_value = float(os.environ["CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"])
                except (KeyError, TypeError, ValueError):
                    _deadline_value = None
            _reserve_value = 0.0
            for _reserve_key in ("finish_reserve_seconds", "probe_finish_reserve_seconds"):
                if d.get(_reserve_key) is not None:
                    try:
                        _reserve_value = max(0.0, float(d[_reserve_key]))
                    except (TypeError, ValueError):
                        pass
                    break
            observation_probe_runner = ProbeRunner(
                root, task_id=str(d.get("task_id") or d.get("id") or fn),
                task_deadline=_deadline_value,
                finish_reserve_seconds=_reserve_value,
                executable=d.get("python") if isinstance(d.get("python"), str) else None)
            measurement.observation_probe_runner = observation_probe_runner
        if _dependency_fact_on:
            from kernel_dependency_probe import DependencyFactRunner
            _dependency_deadline = None
            for _deadline_key in (
                    "task_deadline", "deadline_monotonic", "attempt_deadline",
                    "deadline", "CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"):
                _candidate_deadline = d.get(_deadline_key)
                if _candidate_deadline is not None:
                    try:
                        _dependency_deadline = float(_candidate_deadline)
                    except (TypeError, ValueError):
                        _dependency_deadline = None
                    if _dependency_deadline is not None:
                        break
            if _dependency_deadline is None:
                try:
                    _dependency_deadline = float(
                        os.environ["CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC"])
                except (KeyError, TypeError, ValueError):
                    _dependency_deadline = None
            _dependency_reserve = 0.0
            for _reserve_key in ("finish_reserve_seconds", "probe_finish_reserve_seconds"):
                if d.get(_reserve_key) is not None:
                    try:
                        _dependency_reserve = max(0.0, float(d[_reserve_key]))
                    except (TypeError, ValueError):
                        pass
                    break
            dependency_fact_runner = DependencyFactRunner(
                root, task_id=str(d.get("task_id") or d.get("id") or fn),
                task_deadline=_dependency_deadline,
                finish_reserve_seconds=_dependency_reserve,
                executable=d.get("python") if isinstance(d.get("python"), str) else None)
            measurement.dependency_fact_runner = dependency_fact_runner
        measurement.native_batch = native_batch
        if (measurement.auto_verify_mode == "on"
                or getattr(measurement, "stateframe_green_lock_repair_mode", "off") == "on"
                or getattr(measurement, "stateframe_get_sure_mode", "off") == "on"):
            def _auto_public_verification(selectors, _root=root, _fn=fn, _rel=d["rel"], _row=d):
                if not _budget.charge_automatic_check(
                        selector=", ".join(str(item) for item in selectors)):
                    return {"observation": "AUTOMATIC_CHECK_NOT_FUNDED",
                            "exit_code": 5,
                            "exit_codes": (5,),
                            "target": ", ".join(str(item) for item in selectors)}
                from kernel_check_truth_adapter import automatic_result_identity, receipt_ids
                _before_check_ids = receipt_ids(measurement)
                observation, codes = run_tests_for_selectors(
                    _root, _fn, tuple(selectors), _rel, row=_row)
                exit_code = (0 if codes and all(code == 0 for code in codes)
                             else 1 if any(code == 1 for code in codes)
                             else next((code for code in codes if code != 0), 5))
                # W2: the result references the canonical receipts this run
                # recorded; consumers never re-derive a verdict from exit_code.
                return {"observation": observation, "exit_code": exit_code,
                        "exit_codes": tuple(codes), "target": ", ".join(selectors),
                        **automatic_result_identity(measurement, _before_check_ids)}
            measurement.configure_auto_verify(_auto_public_verification)
        if getattr(measurement, "stateframe_get_sure_mode", "off") == "on":
            def _get_sure_public_prediction(scratch_root, selectors,
                                            _fn=fn, _rel=d["rel"], _row=d):
                selected = tuple(str(item) for item in selectors if str(item).strip())
                if not selected:
                    selected = tuple(str(item) for item in getattr(bolt, "f2p_ids", ())
                                     if str(item).strip())
                if not _budget.charge_automatic_check(selector=", ".join(selected)):
                    return {"observation": "AUTOMATIC_CHECK_NOT_FUNDED",
                            "exit_code": 5,
                            "exit_codes": (5,),
                            "target": ", ".join(selected)}
                observation, codes = run_tests_for_selectors(
                    pathlib.Path(scratch_root), _fn, selected, _rel, row=_row)
                exit_code = (0 if codes and all(code == 0 for code in codes)
                             else 1 if any(code == 1 for code in codes)
                             else next((code for code in codes if code != 0), 5))
                return {"observation": observation, "exit_code": exit_code,
                        "exit_codes": tuple(codes), "target": ", ".join(selected)}
            measurement.configure_get_sure_public_runner(_get_sure_public_prediction)
        if getattr(measurement, "stateframe_turtle_mode", "off") == "on":
            measurement.turtle_sender = lambda body, _m=measurement: _send_turtle_request(
                _m, body, API, KEY)
        if resume is not None and native_batch is not None:
            from kernel_native_restart import restore_batch
            restore_batch(measurement)
        if resume is not None:
            _restored = measurement.resume_worker_state
            edits, tests = _restored["edits"], _restored["tests"]
            unparsed_streak, harness_suspect = _restored["unparsed_streak"], _restored["harness_suspect"]
            last_line, same_streak, run_temp = (_restored[key] for key in ("last_line", "same_streak", "run_temp"))
            edits_since_test = _restored["edits_since_test"]
            baseline_tests, created_this_run = set(_restored["baseline_tests"]), set(_restored["created_this_run"])
            last_surgery_step, tool_calls_seen = _restored["last_surgery_step"], _restored["tool_calls_seen"]
            _budget.restore_usage(measurement.journal.records(), tool_calls_used=int(tool_calls_seen))
            decoy_call_names = _restored["decoy_call_names"]
            schema_violation_details = _restored["schema_violation_details"]
            unknown_tool_names = _restored["unknown_tool_names"]
            first_edit_step = _restored["first_edit_step"]
            wrong_tool_before_first_edit = _restored["wrong_tool_before_first_edit"]
        step = getattr(measurement, "resume_step", 0) - 1
        _restored_advisories = (
            _restored.get("zero_progress_advised_checkpoints", ())
            if resume is not None and isinstance(_restored, Mapping) else ()
        )
        _zero_progress_advised_checkpoints: set[int] = {
            item for item in _restored_advisories
            if type(item) is int and item in {50, 100, 150}
        } if _host_truth_progress_enabled and isinstance(
            _restored_advisories, (list, tuple, set, frozenset)) else set()

        def _progress_facts():
            finish_result = getattr(measurement, "_last_finish_result", None)
            if not isinstance(finish_result, Mapping):
                finish_result = {}
            check_output = getattr(measurement, "_check_output_receipts", {})
            check_receipts = (tuple(check_output.values())
                              if isinstance(check_output, Mapping) else check_output)
            resume_state = getattr(measurement, "resume_worker_state", None)
            prior = (resume_state.get("host_progress_facts")
                     if isinstance(resume_state, Mapping) else None)
            return derive_host_progress_facts(
                edit_receipts=getattr(bolt, "_receipts", ()),
                effect_receipts=getattr(measurement, "_effect_application_receipts", ()),
                delivery_manifest=finish_result.get("delivery_manifest"),
                done_facts=finish_result.get("done_facts"),
                product_edits=getattr(bolt, "product_edits", 0),
                file_create_results=created_this_run,
                test_receipts=getattr(measurement, "test_receipts", ()),
                host_check_receipts=getattr(measurement, "host_check_receipts", ()),
                check_receipts=check_receipts,
                owner_receipts=getattr(bolt, "_receipts", ()),
                legacy_edits=edits,
                legacy_checks=tests,
                legacy_product_edits=getattr(bolt, "product_edits", 0),
                legacy_first_edit_step=first_edit_step,
                legacy_edits_since_check=edits_since_test,
                prior_facts=prior,
                enabled=_host_truth_progress_enabled,
            )

        def _reported_steps() -> int:
            return (int(_budget.steps_used) if _budget_mode == "response"
                    else step + 1)

        def _remaining_budget_units() -> int | None:
            return (remaining(_budget.limit, _budget.responses_used)
                    if _budget_mode == "response"
                    else remaining(step_cap, step + 1))

        measurement._long_horizon_capacity_terminal_emitted = False

        def _capacity_details(refusal: object) -> tuple[dict, int | None, int | None]:
            capacity = getattr(refusal, "capacity", None)
            capacity = dict(capacity) if isinstance(capacity, Mapping) else {}
            prompt_tokens = getattr(refusal, "prompt_tokens", None)
            context_limit = getattr(refusal, "context_limit", None)
            if prompt_tokens is None:
                prompt_tokens = capacity.get("prompt_tokens")
            if context_limit is None:
                context_limit = capacity.get("context_limit")
            return capacity, prompt_tokens, context_limit

        def _emit_capacity_terminal(refusal: object, capture: object) -> None:
            if getattr(measurement, "_long_horizon_capacity_terminal_emitted", False):
                return
            capacity, _prompt_tokens, _context_limit = _capacity_details(refusal)
            measurement.journal.emit(
                "terminal_result",
                request_id=getattr(capture, "request_id", None),
                terminal_result="CONTEXT_OVERFLOW",
                failure_class="context_overflow",
                assigned=True,
                capacity=capacity,
            )
            measurement._long_horizon_capacity_terminal_emitted = True

        def _terminal_capacity_refusal(refusal: object):
            from capability_capacity_preflight import CapacityRefusal
            capacity, prompt_tokens, context_limit = _capacity_details(refusal)
            return CapacityRefusal(
                str(refusal), terminal=True, prompt_tokens=prompt_tokens,
                context_limit=context_limit, capacity=capacity,
                request_id=getattr(refusal, "request_id", None), reroutable=False)

        def _force_capacity_reroute() -> str | None:
            """Force the active presentation mechanism once after exact refusal."""
            nonlocal _request_tools, _action_known_refs, _uncapped_messages
            if history_compactor is not None:
                forced = history_compactor.maybe_compact(
                    msgs, tools=_request_tools,
                    original_prefix=baseline_messages[:2], turn_index=step, force=True)
                if forced.applied:
                    msgs[:] = [copy.deepcopy(dict(message))
                               for message in forced.messages]
                    _uncapped_messages = copy.deepcopy(msgs)
                    _request_tools = (measurement.tool_menu(
                        tool_surface, _uncapped_messages) if toolcall_mode else None)
                    if (native_batch is not None and _request_tools is not None
                            and os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on"):
                        _request_tools = native_batch.menu_for_pending(_request_tools)
                    _action_known_refs = (
                        _live_action_references(measurement, bolt, fileset, target)
                        if action_schema_mode else None)
                    measurement.history_summary_ordinal += 1
                    if getattr(measurement, "long_horizon_depth_tracker", None) is not None:
                        measurement.long_horizon_depth_tracker.rebase(msgs)
                    return "history_force_summary"
                return None
            if getattr(measurement, "stateframe_hybrid_mode", "off") == "on":
                if not getattr(measurement, "_hybrid_crossed", False):
                    measurement._force_hybrid_crossing = True
                    return "hybrid_force_crossing"
                # Already crossed: the exact count says the latched frame's log
                # grew past capacity before the chars/4 rollover guard noticed.
                # One fresh epoch (snapshot + empty log) is the hybrid's own
                # compaction; only if that still does not fit is it terminal.
                measurement._force_frame_epoch = True
                return "hybrid_force_epoch"
            if (getattr(measurement, "stateframe_frame_epoch_mode", "off") == "on"
                    or getattr(measurement, "stateframe_f4_mode", "off") == "on"):
                measurement._force_frame_epoch = True
                return "frame_force_epoch"
            return None

        from kernel_stub_gate import StubGate
        _stub_gate = StubGate()
        _loop_limit = _budget.tool_attempt_limit
        from kernel_tool_call_loop_breaker import UnboundedProgressWatch, stop_unbounded_no_progress
        _progress_watch = UnboundedProgressWatch() if step_cap is None else None
        if (_progress_watch is not None and resume is not None
                and "unbounded_progress_watch" in _restored):
            _progress_watch = UnboundedProgressWatch.from_snapshot(
                _restored["unbounded_progress_watch"])

        def _finish_edit_nturn() -> None:
            nonlocal _edit_nturn_request_active
            if _edit_nturn_session is None:
                return
            _edit_nturn_session.finish()
            _edit_nturn_request_active = False
            if not _edit_nturn_session.event_emitted:
                emit_edit_nturn_rendered(
                    measurement.journal,
                    n_sites=_edit_nturn_session.n_sites,
                    turns_used=_edit_nturn_session.turns_used,
                    edit_wall_seconds=time.monotonic() - _edit_nturn_session.started_at,
                    site_source=_edit_nturn_session.site_source,
                )
                _edit_nturn_session.event_emitted = True
            if (_edit_nturn_session.scope_turns_enabled
                    and not _edit_nturn_session.scope_event_emitted):
                emit_edit_nturn_scope_turns_rendered(
                    measurement.journal, _edit_nturn_session)

        def _queue_edit_nturn_extra(current_step: int) -> None:
            if _edit_nturn_session is None or _edit_nturn_session.finished:
                return
            if _edit_nturn_session.turns_remaining <= 0:
                _finish_edit_nturn()
                return
            remaining = max(0.0, float(_edit_nturn_deadline) - time.monotonic())
            required = edit_nturn_required_seconds(_edit_nturn_verification_reserve)
            reason = None
            if remaining < required:
                reason = "wall_clock"
            elif reached(_loop_limit, current_step + 1):
                reason = "step_budget"
            elif not _budget.can_start_response():
                reason = "response_budget"
            elif not _budget.can_start_tool_call():
                reason = "tool_budget"
            if reason is not None:
                emit_edit_nturn_skipped_budget(
                    measurement.journal,
                    n_sites=_edit_nturn_session.n_sites,
                    turns_used=_edit_nturn_session.turns_used,
                    remaining_seconds=remaining,
                    required_seconds=required,
                    reason=reason,
                )
                _finish_edit_nturn()
                return
            instruction = _edit_nturn_session.queue_extra_turn(
                remaining, _edit_nturn_verification_reserve)
            if instruction is None:
                _finish_edit_nturn()
                return
            append_edit_nturn_instruction(
                msgs,
                _edit_nturn_session.continuation_instruction(
                    current_revision=int(getattr(bolt, "workspace_revision", 0) or 0),
                    public_checks_green=_edit_nturn_public_checks_green(),
                ),
            )

        def _edit_nturn_public_checks_green() -> bool:
            if _host_truth_progress_enabled:
                progress = _progress_facts()
                return (
                    progress.checks > 0
                    and progress.latest_check_passed
                    and progress.edits_since_check == 0
                )
            revision = int(getattr(bolt, "workspace_revision", 0) or 0)
            if bolt.last_test_exit != 0 or edits_since_test != 0:
                return False
            for receipt in reversed(getattr(bolt, "_receipts", ())):
                if not isinstance(receipt, Mapping) or receipt.get("kind") != "test":
                    continue
                return (
                    receipt.get("workspace_revision") == revision
                    and receipt.get("result_class") in {
                        "GREEN", "PASS", "PASS_CASES", "TESTS_PASSED",
                    }
                )
            return False

        def _edit_nturn_record_changed_owner(
            model_action_args: object,
            current_step: int,
        ) -> None:
            session = _edit_nturn_session
            if session is None or not session.scope_sites:
                return
            revision = int(getattr(bolt, "workspace_revision", 0) or 0)
            receipt = next((item for item in reversed(getattr(bolt, "_receipts", ()))
                            if isinstance(item, Mapping)
                            and item.get("kind") == "edit"
                            and item.get("result_class") == "APPLIED"
                            and item.get("turn") == current_step), None)
            if (receipt is None
                    or int(receipt.get("workspace_revision", -2)) + 1 != revision):
                return
            path = str(receipt.get("target_path") or "").replace("\\", "/").lstrip("./")
            arguments = model_action_args if isinstance(model_action_args, Mapping) else {}
            owner_values: set[str] = set()
            source_values: list[str] = []
            for key in ("owner", "label", "symbol", "name", "function",
                        "function_name", "target_symbol"):
                value = arguments.get(key)
                if isinstance(value, str) and value.strip():
                    owner_values.add(value.strip())
            target_ref = arguments.get("target_ref")
            if isinstance(target_ref, Mapping):
                for key in ("owner", "label", "symbol", "name"):
                    value = target_ref.get(key)
                    if isinstance(value, str) and value.strip():
                        owner_values.add(value.strip())
            for key in ("new_str", "new", "function_source", "source", "content"):
                value = arguments.get(key)
                if isinstance(value, str):
                    source_values.append(value)
            for site in session.scope_sites:
                if path != site.path:
                    continue
                aliases = {site.owner, site.symbol} - {""}
                explicit_match = bool(aliases & owner_values)
                source_match = bool(site.symbol and any(re.search(
                    rf"\b(?:def|class)\s+{re.escape(site.symbol)}\b", source)
                    for source in source_values))
                if explicit_match or source_match:
                    session.record_changed_owner(site.owner, revision)
                    if site.symbol:
                        session.record_changed_owner(site.symbol, revision)

        def _remember_edit_nturn_model_plan(content: object) -> None:
            nonlocal _edit_nturn_model_plan_record_id
            if not _edit_nturn_enabled:
                return
            capture = getattr(_prepared, "capture", None)
            # Scope labels come only from the model-visible public response, never its
            # private reasoning channel or host/scorer metadata.
            discovered = parse_model_plan_site_records(content)
            if discovered:
                known = {
                    (getattr(site, "path", ""), getattr(site, "owner", "").casefold())
                    for site in _edit_nturn_model_sites
                }
                for site in discovered:
                    key = (site.path, site.owner.casefold())
                    if key not in known:
                        _edit_nturn_model_sites.append(site)
                        known.add(key)
                _edit_nturn_model_plan_record_id = str(
                    getattr(capture, "request_id", "") or "")

        def _edit_nturn_after_action(line_value: object, observation: object,
                                    current_step: int,
                                    model_action_args: object = None,
                                    scope_receipt: object = None) -> None:
            nonlocal _edit_nturn_session, _edit_nturn_request_active
            if not _edit_nturn_enabled:
                return
            applied = edit_action_applied(line_value, observation)
            if _edit_nturn_request_active:
                _edit_nturn_request_active = False
                if _edit_nturn_session is not None:
                    if applied:
                        _edit_nturn_record_changed_owner(model_action_args, current_step)
                    _edit_nturn_session.record_edit_result(applied=applied)
                    if applied:
                        _queue_edit_nturn_extra(current_step)
                    else:
                        _finish_edit_nturn()
                return
            if _edit_nturn_session is not None or not applied:
                return
            localization = getattr(measurement, "_last_localize_result", None)
            capture = getattr(_prepared, "capture", None)
            request_id = getattr(capture, "request_id", "")
            action_args = (model_action_args if isinstance(model_action_args, Mapping)
                           else {})
            scope_evidence = (scope_receipt or action_args.get("scope_receipt"))
            if scope_evidence is None:
                scope_evidence = tuple(
                    receipt for receipt in getattr(bolt, "_receipts", ())
                    if isinstance(receipt, Mapping)
                    and receipt.get("kind") in {"edit", "read", "grep", "list_files", "localize"}
                    and receipt.get("result_class") not in {
                        "FAILED", "REFUSED", "REJECTED", "NO_OP", "UNKNOWN_TOOL",
                    }
                    and receipt.get("scope") is not None
                    and receipt.get("scope") not in ("diagnostic", "workspace", "full")
                )
            _edit_nturn_session, _resolution = start_edit_nturn_episode(
                True,
                measurement.journal,
                repository_root=root,
                model_plan_sites=tuple(_edit_nturn_model_sites),
                model_plan_record_id=_edit_nturn_model_plan_record_id,
                localization_result=localization,
                scope_receipt=scope_evidence,
                model_action_targets=(
                    action_args if action_args.get("path") is not None else None
                ),
                str_replace_targets=(
                    action_args if action_args.get("edits") is not None
                    or action_args.get("old_str") is not None else None
                ),
                live_config=d,
                record_id=request_id,
            )
            if _edit_nturn_session is not None:
                _edit_nturn_record_changed_owner(model_action_args, current_step)
                _queue_edit_nturn_extra(current_step)

        def _continue_after_heuristic_terminal(trigger: str) -> bool:
            from kernel_terminal_policy import (
                enabled as hard_budget_only_enabled,
                queue_advisory,
            )
            if (step_cap is None and trigger in {
                    "tool_call_loop", "postgreen_stall", "response_reuse_stall",
                    "postgreen_no_progress"}):
                return False
            if not hard_budget_only_enabled(measurement):
                return False
            queue_advisory(measurement, trigger)
            return True

        # True save/load resume (kernel_checkpoint): restore before the loop, snapshot at each boundary.
        import kernel_checkpoint as _kernel_checkpoint
        _kernel_checkpoint.restore_point(sys._getframe(), POC)
        for step in step_indices(step + 1, _loop_limit):
            _kernel_checkpoint.save_point(sys._getframe(), step, POC)
            _kernel_checkpoint.freeze_warm_heap(step)
            if (_progress_watch is not None
                    and stop_unbounded_no_progress(measurement, _progress_watch, boundary=step)):
                break
            if native_batch is not None:
                native_batch.append_unexecuted_tool_results(msgs)
            if _edit_nturn_request_active:
                _finish_edit_nturn()
            _edit_nturn_request_active = False
            plain_observation = None
            if d.get("_bridge_capture"):
                from kernel_bridge_evidence import capture_candidate
                capture_candidate(d["_bridge_capture"], root, d, step)
            bolt.current_step = step        # E331: turn-stamps every kernel trace event
            if (_budget_mode == "response"
                    and not _budget.can_start_tool_call()):
                if native_batch is not None:
                    native_batch.discard_pending("tool_call_cap")
                measurement.system_outcome = "tool_call_cap"
                break
            if measurement.system_outcome in {'invalidate_attempt', 'quarantine_harness'}:
                measurement.journal.emit('supervision_stop', reason=measurement.system_outcome, step=step)
                break
            if measurement.system_outcome == "STOPPED_INCOMPLETE":
                if _continue_after_heuristic_terminal("generic_terminal"):
                    measurement.system_outcome = None
                else:
                    # Legacy mode retains the repair-block cap behavior.
                    measurement.journal.emit('supervision_stop', reason=measurement.system_outcome, step=step)
                    break
            # E399: checkpoint decisions use host receipts and can only add one
            # advisory per checkpoint. The model still gets this turn and hard budgets
            # remain the only run-ending limits.
            _checkpoint_progress = _progress_facts()
            if (should_deliver_zero_progress_advisory(
                    step,
                    (_checkpoint_progress.product_edits if _host_truth_progress_enabled
                     else _checkpoint_progress.edits),
                    _checkpoint_progress.checks,
                    interventions=interventions, lean_mode=_lean_run)
                    and step not in _zero_progress_advised_checkpoints):
                _zero_progress_advised_checkpoints.add(step)
                _advisory_delivered = False
                if getattr(bolt, "composed", None) is not None:
                    try:
                        _fadv = bolt.composed.advance(
                            {"kind": "gate", "intent": "freeze_checkpoint",
                             "result_class": "FROZEN", "fingerprint": f"freeze{step}",
                             "turn": step, "workspace_revision": 0, "ts": 0,
                             "why": "no host-recorded edit or check by checkpoint"})
                        # Deliver one compact next action, never the full cognition card.
                        _sel = getattr(_fadv, "selection", None)
                        _aid = (getattr(_sel, "selected_action_id", None)
                                or getattr(_sel, "baseline_action_id", None))
                        if _aid:
                            bolt._rec("composed", "zero_progress_advisory", f"step {step}")
                            msgs.append({"role": "user", "content":
                                         "[kernel] You have taken ZERO actions in "
                                         f"{step} steps of pure reading. Stop "
                                         "deliberating. Concrete next action: "
                                         f"{_aid} — re-read the target function "
                                         "against the oracle's demanded values, "
                                         "then make ONE edit and run the tests."})
                            _advisory_delivered = True
                    except Exception:                                # noqa: BLE001
                        pass
                if not _advisory_delivered:
                    bolt._rec("gate", "zero_progress_advisory", f"step {step}")
                    msgs.append({"role": "user", "content":
                                 "[kernel] You have taken ZERO actions in "
                                 f"{step} steps of pure reading. Pick one concrete "
                                 "product edit or run the relevant tests now. "
                                 "Continue with the task; this checkpoint does not stop the run."})
                print(f"    [checkpoint] {fn}: no host-recorded edit/check by step {step} — advisory delivered")
            line = None
            # Components support three states: on, off, and passthrough.
            # KERNEL_SHADOW="e317,e320" runs those organs fully but SUPPRESSES their
            # model-facing output, recording would_fire trace events instead. Build
            # everything, dry-run safely, and check post-hoc whether a shadowed organ
            # would have caught what a failed run missed.
            tool_edit_src = None          # structured edit source (toolcall mode; no fences)
            tool_str_args = None          # (old_str, new_str) for the str_replace organ
            tool_insert_args = None       # revision-bound insert_before arguments
            tool_authored_check_args = None
            tool_batch_args = None        # ordered str_replace edits; one parent receipt
            tool_replace_module_args = None  # (expected_revision, source, candidate_id)
            _pointer_read_path = None     # successful, authorized source delivery only
            _pointer_read_delivery = None
            _rejection_template_source = None
            _rejection_template_path = None
            tool_target_ref = None        # optional STATEFRAME_BOUND_EDIT target
            tool_probe_args = None        # structured probe arguments (gate or observation mode)
            tool_repro_args = None        # model-authored scratch reproduction arguments
            tool_predict_args = None      # host-owned scratch prediction arguments
            tool_dependency_fact_args = None  # installed-dependency characterization request
            tool_work_args = None         # typed host-owned work-record operations
            tool_decisive_factor_args = None  # advisory approach/factor/disqualifier record
            _done_summary = ""              # legacy text mode has no native args object
            _done_claim = None            # typed action-envelope completion claim
            _done_evidence_refs = None    # typed action-envelope host references
            _done_no_change_paths = None
            _decisive_factor_advisory = ""
            _args = {}                   # text mode has no native argument payload
            tool_call_id = None           # set -> observation goes back as role=tool
            observe_stall_in_place(
                measurement, msgs, str(brief or ""), enabled=not _lean_on,
            )
            _uncapped_messages = copy.deepcopy(msgs)
            # E298 graft #3 — HISTORY TRIM for small context windows. 4bA died
            # ERR:HTTPError at s139 (31 edits in context): the local llama-server rejects
            # oversized prompts, an error Flash's 128k window never surfaces. Keep the
            # system prompt + brief and the most recent exchanges, cutting at an assistant
            # boundary so no tool_call is orphaned (#87b's sanitize rule).
            if (history_compactor is None
                    and not measurement.replaces_legacy_trim
                    and not _lean_on
                    and len(msgs) > int(os.environ.get("REACT_TRIM", "90"))):
                _keep_from = None
                for _i in range(max(3, len(msgs) - int(int(os.environ.get("REACT_TRIM", "90")) * 0.75)), len(msgs)):
                    if msgs[_i].get("role") == "assistant":
                        _keep_from = _i
                        break
                if _keep_from and _keep_from > 2:
                    dropped = _keep_from - 2
                    # E316 (StateFrame shadow, writer half): the trim is the ONLY lossy
                    # point in the transcript — persist the dropped exchanges append-only
                    # so the offline projector can reconstruct the FULL run. Never read
                    # back into the loop, never shown to the model.
                    try:
                        _edir = POC / "results" / ("react_events"
                                                   + os.environ.get("REACT_TAG", ""))
                        _edir.mkdir(parents=True, exist_ok=True)
                        with (_edir / f"{fn}.jsonl").open("a") as _ef:
                            for _dm in msgs[2:_keep_from]:
                                _ef.write(json.dumps({"kind": "trimmed", "step": step,
                                                      "msg": _dm}) + "\n")
                    except OSError:
                        pass  # shadow persistence must never break the live loop
                    # E407 consumer: the trim message carries the world-cache digest —
                    # what the model OBSERVED (test lists, signatures) survives the trim
                    # without re-reading files. Fresh-only, bounded (attention law E394).
                    # E426: the trim message also carries the current-unit line so the
                    # plan survives every trim (the JobQueue drift failure class).
                    _up_line = ""
                    if getattr(bolt, "unit_plan", None) is not None and \
                            len(getattr(bolt.unit_plan, "units", ())) > 1:
                        try:
                            from unit_planner import render_current_unit_line
                            _up_line = "\n" + render_current_unit_line(bolt.unit_plan)
                        except Exception:                            # noqa: BLE001
                            _up_line = ""
                    _wk_digest = ""
                    if getattr(bolt, "world", None) is not None:
                        try:
                            _wk_digest = bolt.world.render_rehydration_digest(
                                unit_scope={"target_path": (target.relative_to(root).as_posix()
                                                              if target is not None else ""),
                                            "target_symbol": fn},
                                revision=max(getattr(bolt, "workspace_revision", 0), bolt.world.current_revision),
                                max_tokens=220) or ""
                            if _wk_digest:
                                _wk_digest = "\n" + _wk_digest
                        except Exception:                            # noqa: BLE001
                            _wk_digest = ""
                    _trim_progress = _progress_facts()
                    trim_summary = (f"{bolt.plan_summary(_trim_progress.edits, _trim_progress.checks)}){_up_line}{_wk_digest}"
                                    if interventions else "history omitted)")
                    msgs[:] = msgs[:2] + [
                        {"role": "user", "content":
                         f"[kernel] ({dropped} earlier exchanges trimmed to fit context. "
                         f"{trim_summary}"}
                    ] + msgs[_keep_from:]
            _request_tools = measurement.tool_menu(tool_surface, _uncapped_messages) if toolcall_mode else None
            if (native_batch is not None and _request_tools is not None
                    and os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on"):
                _request_tools = native_batch.menu_for_pending(_request_tools)
            _action_known_refs = (_live_action_references(measurement, bolt, fileset, target)
                                 if action_schema_mode else None)
            if getattr(measurement, "stateframe_turtle_mode", "off") == "on":
                _turtle_context = measurement.turtle_pre_request(endpoint=API, key=KEY)
                if _turtle_context:
                    _turtle_message = {"role": "user", "content": _turtle_context}
                    msgs.append(_turtle_message)
                    _uncapped_messages.append(copy.deepcopy(_turtle_message))
            if history_compactor is not None:
                _compaction = history_compactor.maybe_compact(
                    msgs,
                    tools=_request_tools,
                    original_prefix=baseline_messages[:2],
                    turn_index=step,
                )
                if _compaction.applied:
                    msgs[:] = [copy.deepcopy(dict(message))
                               for message in _compaction.messages]
                    measurement.history_summary_ordinal += 1
                    if getattr(measurement, "long_horizon_depth_tracker", None) is not None:
                        measurement.long_horizon_depth_tracker.rebase(msgs)
                    # The runtime's uncapped/restart copy must follow the same
                    # model-authored summary, or prepare() would reintroduce the
                    # dropped turns on the next request.
                    _uncapped_messages = copy.deepcopy(msgs)
                    _request_tools = (measurement.tool_menu(
                        tool_surface, _uncapped_messages) if toolcall_mode else None)
                    if (native_batch is not None and _request_tools is not None
                            and os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on"):
                        _request_tools = native_batch.menu_for_pending(_request_tools)
            if getattr(measurement, "long_horizon_depth_tracker", None) is not None:
                measurement.long_horizon_depth_tracker.observe(msgs)
            if measurement.restart_enabled:
                from kernel_continuation_restart import checkpoint
                _checkpoint_state = {
                    "edits": edits, "tests": tests, "unparsed_streak": unparsed_streak,
                    "harness_suspect": harness_suspect, "last_line": last_line,
                    "same_streak": same_streak, "run_temp": run_temp,
                    "edits_since_test": edits_since_test, "baseline_tests": baseline_tests,
                    "created_this_run": created_this_run, "last_surgery_step": last_surgery_step,
                    "tool_calls_seen": tool_calls_seen, "decoy_call_names": decoy_call_names,
                    "schema_violation_details": schema_violation_details,
                    "unknown_tool_names": unknown_tool_names, "first_edit_step": first_edit_step,
                    "wrong_tool_before_first_edit": wrong_tool_before_first_edit}
                if _host_truth_progress_enabled:
                    _checkpoint_progress = _progress_facts()
                    _checkpoint_state.update({
                        "edits": _checkpoint_progress.edits,
                        "tests": _checkpoint_progress.checks,
                        "edits_since_test": _checkpoint_progress.edits_since_check,
                        "first_edit_step": _checkpoint_progress.first_edit_step,
                        "host_progress_facts": _checkpoint_progress.to_record(),
                        "zero_progress_advised_checkpoints": tuple(
                            sorted(_zero_progress_advised_checkpoints)),
                    })
                if _progress_watch is not None:
                    _checkpoint_state["unbounded_progress_watch"] = _progress_watch.snapshot()
                checkpoint(measurement, _uncapped_messages, _checkpoint_state)
            _provider_request = not (native_batch is not None and native_batch.pending)
            _green_then_exhaust_boundary = (
                _provider_request
                and _budget_mode == "response"
                and _budget.exhausted
                and getattr(measurement, "stateframe_green_then_exhaust_mode", "off") == "on"
            )
            if _green_then_exhaust_boundary:
                from kernel_green_then_exhaust import prepare_notice
                measurement._green_then_exhaust_prepare_notice = prepare_notice(measurement)
            else:
                measurement._green_then_exhaust_prepare_notice = None
            _prepare_host_inputs = getattr(ask_tools, "prepare_request_inputs", None)
            _request_host_text_preparer = getattr(
                ask_tools, "prepare_request_host_text", None,
            )
            if not callable(_request_host_text_preparer):
                _request_host_text_preparer = None
            if _provider_request and callable(_prepare_host_inputs):
                msgs = _prepare_host_inputs(msgs)
                _uncapped_messages = _prepare_host_inputs(_uncapped_messages)
            if _provider_request and _request_host_text_preparer is not None:
                _prepared = measurement.prepare(
                    msgs, uncapped=_uncapped_messages,
                    tools=_request_tools, provider_request=True,
                    request_host_text_preparer=_request_host_text_preparer,
                )
            else:
                _prepared = measurement.prepare(
                    msgs, uncapped=_uncapped_messages,
                    tools=_request_tools, provider_request=_provider_request,
                )
            # Validate the exact final contract that was prepared for this
            # request.  Owner/menu projections may narrow a fresh list during
            # prepare; reusing the pre-prepare list was the museum L71 bug,
            # where offered edit calls were rejected as unavailable.
            request_contract = getattr(measurement, "_completion_owner_request_contract", None)
            if isinstance(request_contract, Mapping):
                admitted_tools = request_contract.get("admitted_tool_schemas")
                if isinstance(admitted_tools, list):
                    _request_tools = [copy.deepcopy(dict(tool)) for tool in admitted_tools
                                      if isinstance(tool, Mapping)]
                else:
                    _request_tools = _prepared_tool_contract(_prepared, _request_tools)
            else:
                _request_tools = _prepared_tool_contract(_prepared, _request_tools)
            if (isinstance(request_contract, Mapping)
                    and request_contract.get("host_stop")):
                measurement.journal.emit(
                    "provider_boundary_host_stop",
                    producer="kernel_completion_owner",
                    event_schema="stateframe-request-action-contract/1",
                    request_id=_prepared.capture.request_id,
                    route="host-stop",
                    reason="settled before provider dispatch",
                    dispatch=False,
                )
                if _continue_after_heuristic_terminal("completion_owner_exhausted"):
                    from kernel_terminal_policy import (
                        deliver_pending_advisories,
                        terminal_outcome_permitted,
                    )
                    if not terminal_outcome_permitted(
                            getattr(measurement, "system_outcome", None), measurement):
                        _record_prepared_provider_not_dispatched(
                            measurement, _prepared, provider_request=_provider_request,
                            reason_code="completion_owner_advisory")
                        deliver_pending_advisories(
                            measurement, _prepared.messages, msgs)
                        continue
                else:
                    _record_prepared_provider_not_dispatched(
                        measurement, _prepared, provider_request=_provider_request,
                        reason_code="completion_owner_host_stop")
                    measurement.system_outcome = str(
                        getattr(measurement, "system_outcome", None)
                        or "COMPLETION_OWNER_EXHAUSTED")
                    break
            _green_then_exhaust_boundary_active = (
                _green_then_exhaust_boundary
                and bool(getattr(measurement, "_green_then_exhaust_rendered", False))
            )
            measurement._green_then_exhaust_boundary_active = _green_then_exhaust_boundary_active
            _audit_system_prompt(_prepared.messages)
            from kernel_tool_call_loop_breaker import terminal_preemption_allowed
            if terminal_preemption_allowed(measurement):
                from kernel_completion_owner import request_pending as owner_request_pending
                if not owner_request_pending(measurement):
                    if not _continue_after_heuristic_terminal("tool_call_loop"):
                        _record_prepared_provider_not_dispatched(
                            measurement, _prepared, provider_request=_provider_request,
                            reason_code="tool_call_loop_breaker")
                        measurement.system_outcome = "tool_call_loop_breaker"
                        break
            if not provider_dispatch_allowed(_prepared):
                _record_prepared_provider_not_dispatched(
                    measurement, _prepared, provider_request=_provider_request,
                    reason_code="prepared_overflow")
                break
            _completion_owner_on = (
                getattr(measurement, "stateframe_completion_owner_mode", "off") == "on")
            if _completion_owner_on:
                from kernel_completion_owner import (
                    before_provider_request as owner_boundary,
                    charge_response as owner_charge_response,
                )
                if not owner_boundary(measurement, provider_request=_provider_request):
                    _record_prepared_provider_not_dispatched(
                        measurement, _prepared, provider_request=_provider_request,
                        reason_code="completion_owner_boundary")
                    break
            _response_charged = True if not _provider_request else (
                owner_charge_response(measurement, _budget, step=step)
                if _completion_owner_on else _budget.charge_response(step=step)
            )
            if _provider_request and not _response_charged:
                _record_prepared_provider_not_dispatched(
                    measurement, _prepared, provider_request=_provider_request,
                    reason_code="response_budget_unavailable")
                if _green_then_exhaust_boundary_active:
                    from kernel_green_then_exhaust import submit_if_ready
                    _green_then_exhaust_decision = submit_if_ready(measurement)
                    if getattr(_green_then_exhaust_decision, "accepted", False):
                        break
                measurement.system_outcome = "response_budget_exhausted"
                break
            _capture_token = measurement.activate(_prepared)

            def _provider_request_once():
                nonlocal _prepared, _capture_token, _request_tools, _action_known_refs
                rerouted = False
                reroute_action = None
                while True:
                    try:
                        if toolcall_mode:
                            if native_batch is not None:
                                return native_batch.next(
                                    ask_tools, list(_prepared.messages), temp=run_temp,
                                    tools=_request_tools, known_refs=_action_known_refs)
                            return ask_tools(
                                list(_prepared.messages), temp=run_temp,
                                tools=_request_tools, known_refs=_action_known_refs)
                        return ask(list(_prepared.messages), temp=run_temp)
                    except Exception as error:                       # noqa: BLE001
                        from capability_capacity_preflight import CapacityRefusal
                        if (not isinstance(error, CapacityRefusal)
                                or not getattr(error, "long_horizon_reroutable", False)):
                            raise
                        protocol_on = os.environ.get(
                            "LONG_HORIZON_PROTOCOL", "off").strip().lower() in {
                                "on", "1", "true", "yes"}
                        no_compaction = os.environ.get(
                            "LONG_HORIZON_NO_COMPACTION", "off").strip().lower() in {
                                "on", "1", "true", "yes"}
                        if not protocol_on or no_compaction:
                            raise
                        capacity, prompt_tokens, context_limit = _capacity_details(error)
                        refusal_request_id = getattr(error, "request_id", None) or getattr(
                            _prepared.capture, "request_id", None)
                        if rerouted:
                            measurement.journal.emit(
                                "capacity_reroute",
                                request_id=refusal_request_id,
                                exact_prompt_tokens=prompt_tokens,
                                context_limit=context_limit,
                                action=reroute_action or "unknown",
                                outcome="second_refusal",
                            )
                            _emit_capacity_terminal(error, _prepared.capture)
                            raise _terminal_capacity_refusal(error) from error
                        rerouted = True
                        reroute_action = _force_capacity_reroute()
                        if reroute_action is None:
                            measurement.journal.emit(
                                "capacity_reroute",
                                request_id=refusal_request_id,
                                exact_prompt_tokens=prompt_tokens,
                                context_limit=context_limit,
                                action="none",
                                outcome="no_reroute_possible",
                            )
                            _emit_capacity_terminal(error, _prepared.capture)
                            raise _terminal_capacity_refusal(error) from error
                        if _capture_token is not None:
                            ACTIVE_REQUEST.reset(_capture_token)
                            _capture_token = None
                        if _provider_request and _request_host_text_preparer is not None:
                            _prepared = measurement.prepare(
                                msgs, uncapped=_uncapped_messages,
                                tools=_request_tools, provider_request=True,
                                request_host_text_preparer=_request_host_text_preparer,
                            )
                        else:
                            _prepared = measurement.prepare(
                                msgs, uncapped=_uncapped_messages,
                                tools=_request_tools, provider_request=_provider_request,
                            )
                        _audit_system_prompt(_prepared.messages)
                        if not provider_dispatch_allowed(_prepared):
                            measurement.journal.emit(
                                "capacity_reroute",
                                request_id=refusal_request_id,
                                exact_prompt_tokens=prompt_tokens,
                                context_limit=context_limit,
                                action=reroute_action,
                                outcome="reroute_preparation_overflow",
                            )
                            _emit_capacity_terminal(error, _prepared.capture)
                            raise _terminal_capacity_refusal(error) from error
                        _capture_token = measurement.activate(_prepared)
                        measurement.journal.emit(
                            "capacity_reroute",
                            request_id=refusal_request_id,
                            exact_prompt_tokens=prompt_tokens,
                            context_limit=context_limit,
                            action=reroute_action,
                            outcome="rerouted",
                        )

            try:
                if toolcall_mode:
                    content, calls = _provider_request_once()
                    _remember_edit_nturn_model_plan(content)
                    if _edit_nturn_session is not None and _edit_nturn_session.pending:
                        _edit_nturn_session.request_started()
                        _edit_nturn_request_active = True
                    if calls:
                        # ONE tool per turn: record only the call we answer — every
                        # tool_call in an assistant message MUST get a tool reply
                        # (#87b's sanitize lesson) and the model re-issues the rest.
                        c0 = calls[0]
                        _fn = c0.get("function") or {}
                        _name = _fn.get("name") or ""
                        from kernel_measurement import start_turn_timing_step
                        start_turn_timing_step("tool_execution")
                        if (_edit_nturn_request_active
                                and _name not in {"edit", "file_create", "file_delete", "file_move", "shell", "str_replace",
                                                  "insert_before", "replace_module"}):
                            _finish_edit_nturn()
                        try:
                            _args = json.loads(_fn.get("arguments") or "{}")
                            if not isinstance(_args, dict):
                                _args = {}
                                _args_parse_error = True
                            else:
                                _args_parse_error = False
                        except (TypeError, ValueError):
                            _args = {}
                            _args_parse_error = True
                        if getattr(measurement, "stateframe_completion_owner_mode", "off") == "on":
                            from kernel_completion_owner import note_model_action
                            note_model_action(measurement, _fn.get("name") or "")
                        _memory_sidecar = None
                        _has_memory_sidecar = False
                        from kernel_decision_cache import (
                            enabled as decision_cache_enabled,
                            split_action_arguments,
                        )
                        if decision_cache_enabled():
                            _raw_args = _args
                            _args, _args_memory = split_action_arguments(_args, call=c0)
                            _has_memory_sidecar = (
                                (isinstance(c0, Mapping) and "memory" in c0)
                                or (isinstance(_raw_args, Mapping) and "memory" in _raw_args)
                            )
                            _memory_sidecar = (c0.get("memory") if isinstance(c0, Mapping)
                                               and "memory" in c0 else _args_memory)
                        _continuity_delta = None
                        from kernel_decision_continuity import (
                            enabled as decision_continuity_enabled,
                            split_action_arguments as split_continuity_arguments,
                        )
                        if decision_continuity_enabled():
                            _args, _continuity_delta = split_continuity_arguments(
                                _args, call=c0)
                        _working_note = None
                        from kernel_working_notebook import (
                            enabled as working_notes_enabled,
                            split_action_arguments as split_working_note_arguments,
                        )
                        if working_notes_enabled():
                            _args, _working_note = split_working_note_arguments(_args, call=c0)
                        _lean_read_page_line_limit = None
                        _lean_read_page_byte_limit = None
                        if _lean_on:
                            from kernel_lean import (
                                MAX_READ_PAGE_LINES, MAX_TOOL_RESULT_BYTES,
                                bound_plain_tool_arguments,
                            )
                            _args = bound_plain_tool_arguments(_name, _args)
                            if _name == "read":
                                _lean_read_page_line_limit = MAX_READ_PAGE_LINES
                                _lean_read_page_byte_limit = MAX_TOOL_RESULT_BYTES
                        if _name == "read":
                            from kernel_frame_tool_result_budget import read_page_byte_limit
                            _lean_read_page_byte_limit = read_page_byte_limit(
                                msgs, _lean_read_page_byte_limit)
                        if getattr(measurement, "stateframe_completion_diagnosis_mode", "off") == "on":
                            # Keep the action that crossed this request's completion
                            # boundary available to the host-only diagnosis record.
                            measurement.current_action = {
                                "kind": _name, "arguments": copy.deepcopy(_args)}
                        if getattr(measurement, "intent_carry_mode", "off") == "on":
                            measurement.record_model_decision(
                                reasoning=getattr(_prepared.capture,
                                                  "response_reasoning_content", None),
                                content=content, action={"tool": _name,
                                                         "arguments": _args}, step=step)
                        tool_call_id = c0.get("id") or f"call_{step}"
                        measurement.current_tool_call_id = tool_call_id
                        _grounding_feedback = measurement.start_action(
                            _name, _args, charge_dispatch=not contract_enabled())
                        # start_action snapshots the call identity into the settled
                        # action record before clearing its transient receipt field.
                        measurement.current_tool_call_id = tool_call_id
                        tool_calls_seen += 1
                        if not _budget.register_tool_call(step=step, tool=_name):
                            measurement.system_outcome = "tool_call_cap"
                            if native_batch is not None:
                                native_batch.skip_inflight("tool_call_cap")
                            break
                        # A host refusal still closes a real model action.  Keep
                        # its bounded model patch independent from dispatch so
                        # the next turn can explain why the attempted action was
                        # rejected rather than losing the model's hypothesis.
                        if _continuity_delta is not None:
                            measurement.ingest_decision_continuity_delta(_continuity_delta)
                        bolt.last_why = str(_args.get("why") or "")[:160]
                        _schema_errors = _contract_errors(
                            _request_tools, _name, _args,
                            arguments_parse_error=_args_parse_error)
                        _dispatch_variant = None
                        if not _args_parse_error:
                            from kernel_completion_owner import (
                                RequestActionContractError,
                                dispatch_tool_call,
                            )
                            try:
                                _validated_dispatch = dispatch_tool_call(
                                    _request_tools, _name, _args)
                            except RequestActionContractError as _dispatch_error:
                                if not _schema_errors:
                                    _schema_errors = (str(_dispatch_error),)
                            else:
                                _dispatch_variant = _validated_dispatch.variant
                        if _schema_errors:
                            schema_violation_details.append({
                                "tool": _name,
                                "turn": step,
                                "errors": _schema_errors[:6],
                            })
                        msgs.append({"role": "assistant", "content": content or None,
                                     "tool_calls": [{"id": tool_call_id,
                                                      "type": "function",
                                                       "function": {
                                                           "name": _name,
                                                           "arguments": _fn.get("arguments")
                                                           or "{}"}}]})
                        if _agent_tools is not None and _name == "git" and not _agent_tools.offered("git"):
                            _obs = "REFUSED NOT_A_GIT_REPO: source is not a Git repository or Git is disabled"
                            bolt.receipt("git", "git", "unavailable", "REFUSED", _obs)
                            measurement.observe("git", _obs)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _obs))
                            continue
                        if _name == "shell" and (_shell is None or _shell.settings.shell == "off"):
                            _obs = "REFUSED SHELL_OFF: shell is disabled by --shell off"
                            bolt.receipt("shell", "shell", "off", "REFUSED", _obs)
                            measurement.observe("shell", _obs)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _obs))
                            continue
                        if (_schema_errors and _agent_tools is not None and _name in _agent_tools.names):
                            _code = ("BAD_ARGUMENTS" if _name in _agent_tools.custom else
                                     "BAD_CONDITION" if _name.startswith("job_") else
                                     "BAD_ARGUMENT" if _name == "git" else "WEB_HTTP_ERROR")
                            _obs = f"REFUSED {_code}: " + "; ".join(_schema_errors[:3])
                            bolt.receipt(_name, _name, "arguments", "REFUSED", _obs)
                            measurement.observe(_name, _obs)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _obs))
                            continue
                        _active_menu_capture = getattr(measurement, "_active_capture", None)
                        _menu_request_id = getattr(_active_menu_capture, "request_id", None)
                        from kernel_completion_owner import request_menu_action_admissible
                        if not request_menu_action_admissible(
                                measurement, _name, _menu_request_id):
                            _menu_refusal = (
                                "[kernel-menu] This action is not available in the "
                                "current request. Choose one of the offered actions.")
                            measurement.journal.emit(
                                "owner_menu_dispatch_refused",
                                producer="kernel_completion_owner",
                                request_id=_menu_request_id,
                                tool_name=_name,
                                dispatch=False,
                            )
                            measurement.observe(_name, _menu_refusal)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _menu_refusal))
                            continue
                        # IDEALPATH is the typed owner for its own actions.  Route
                        # those calls before generic binding/legacy-contract gates;
                        # the reducer validates the complete host identity and
                        # stage schema itself.
                        _idealpath_admission = getattr(measurement, "idealpath_admission", None)
                        _idealpath_feedback = (
                            _idealpath_admission(_name, _args)
                            if callable(_idealpath_admission) else None
                        )
                        if _idealpath_feedback is not None:
                            measurement.observe(_name, _idealpath_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _idealpath_feedback))
                            continue
                        _idealpath_dispatch = getattr(measurement, "idealpath_operation", None)
                        _idealpath_feedback = (
                            _idealpath_dispatch(_name, _args)
                            if callable(_idealpath_dispatch) else None
                        )
                        if _idealpath_feedback is not None:
                            measurement.observe(_name, _idealpath_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _idealpath_feedback))
                            continue
                        if _name == "reproduce":
                            _reproduce_feedback(_args, tool_call_id)
                            continue
                        if _name == "restore_candidate":
                            from kernel_candidate_preservation import restore_candidate_for_runtime
                            _restore_feedback = restore_candidate_for_runtime(
                                measurement,
                                _args.get("checkpoint_id", ""),
                                _args.get("expected_current_revision", ""),
                            )
                            _restore_text = json.dumps(
                                _restore_feedback, ensure_ascii=False, sort_keys=True, default=str)
                            measurement.observe(_name, _restore_text)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _restore_text))
                            continue
                        if _grounding_feedback is not None:
                            # Claim grounding is a host pre-dispatch refusal.  Keep it on
                            # the normal typed edit receipt rail so the complete JSON
                            # envelope, including closest real paths, survives the model
                            # observation and journal amendment paths.
                            bolt.receipt(
                                "edit", f"{_name} claim grounding",
                                str(_args.get("path") or _args.get("target_ref") or _name),
                                "REJECTED", _grounding_feedback,
                                target_path=str(_args.get("path") or "") or None,
                            )
                            measurement.observe(_name, _grounding_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _grounding_feedback))
                            continue
                        _binding_current = measurement.dispatch_binding_current()
                        if (action_schema_mode or contract_enabled() or native_tool_batches) and (
                                _schema_errors or not _binding_current):
                            _reject_text = _render_contract_rejection(
                                _name, _schema_errors,
                                binding_current=_binding_current)
                            if (_name == "read" and not str(_args.get("path") or "").strip()
                                    and os.environ.get("STATEFRAME_TOKENLAB_EMPTY_READ_PATHS", "off").strip().lower() == "on"):
                                # request fitting (L1c): answer `read {}` with the readable paths, once per turn
                                # shape, so the next call can name a real file (or act on the frame).
                                _readable = sorted(str(_p) for _p in fileset.paths if (root / _p).is_file())[:24]
                                _reject_text += ("\nread needs a `path`. Readable files: " + ", ".join(_readable)
                                                 + ". Files whose content is already in the frame need no re-read;"
                                                   " next: read one file by path, edit, run_tests or done.")
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _reject_text))
                            continue
                        if _args.get("path") and not information.check_path(root, str(_args["path"])):
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, "[information-policy] Path outside workspace denied."))
                            continue
                        # Q6: an identical call already answered by [REPEATED ZERO] is refused.
                        from kernel_repeated_zero import refusal as _rz_refusal
                        _rz_feedback = _rz_refusal(measurement, _name, _args)
                        if _rz_feedback is not None:
                            bolt.receipt("gate", _name, _name, "REPEATED_ZERO_SUPPRESSED", _rz_feedback)
                            measurement.observe(_name, _rz_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _rz_feedback))
                            continue
                        _series_feedback = measurement.series_restriction(_name, _args)
                        if _series_feedback is not None:
                            # Q8: a navigation lookup answers first; the gate text is an advisory.
                            from kernel_navigation_receipt import answer_despite_gate as _nav_answer
                            _series_feedback = _nav_answer(
                                _name, _args, _series_feedback,
                                getattr(measurement, "repo_intel_operation", None)) or _series_feedback
                            measurement.observe(_name, _series_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _series_feedback))
                            continue
                        # r46: STATEFRAME_STUB_SWEEP_DISPATCH refuses the exact redundant read at
                        # dispatch (default off; kernel_stub_sweep_cut.dispatch_refusal).
                        _stub_sweep_dispatch_fn = getattr(
                            measurement, "stub_sweep_dispatch_refusal", None)
                        _stub_sweep_dispatch_feedback = (
                            _stub_sweep_dispatch_fn(_name, _args)
                            if callable(_stub_sweep_dispatch_fn) else None)
                        if _stub_sweep_dispatch_feedback is not None:
                            measurement.observe(_name, _stub_sweep_dispatch_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _stub_sweep_dispatch_feedback))
                            continue
                        # Terminal green-lock refusal MUST run before continuation authorization:
                        # refusing after authorize left an open effect, stranded the continuation,
                        # and every later action was REFUSED_PREDISPATCH (batch 136 crawl_a_one:
                        # 37 blocked reads, done never reached).
                        from kernel_green_lock import guard_action as _green_guard_action
                        # A check receipt is a read-only evidence lookup.  It
                        # cannot change product state or completion status, so
                        # the completion/green-lock veto must not intercept a
                        # valid handle before the dedicated dispatch branch.
                        _green_refusal = (
                            None if _name in {
                                "read_check_receipt", "recall", "authored_check", "insert_before",
                            }
                            else _green_guard_action(_name, _args if isinstance(_args, dict) else {})
                        )
                        if _green_refusal is not None:
                            # Q8: never substitute a completion/green-lock refusal for a
                            # where_defined/who_calls/references answer (read-only lookup).
                            from kernel_navigation_receipt import answer_despite_gate as _nav_answer
                            _nav_feedback = _nav_answer(
                                _name, _args, _green_refusal,
                                getattr(measurement, "repo_intel_operation", None))
                            if _nav_feedback is not None:
                                measurement.observe(_name, _nav_feedback)
                                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _nav_feedback))
                                continue
                            bolt.receipt("gate", _name, _name, "REFUSED_PREDISPATCH", _green_refusal)
                            measurement.observe(_name, _green_refusal)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _green_refusal))
                            continue
                        # decide is a host-owned advisory record with no workspace effect.
                        # Like predict it is served BEFORE continuation authorization, so
                        # KERNEL_CONTINUATION_EXECUTIVE=1 cannot refuse it for lacking an
                        # execution receipt contract or strand an open effect.
                        if _name == "decide":
                            _decisive_factor_feedback = measurement.decisive_factor_operation(
                                _name, tool_decisive_factor_args or _args)
                            if _decisive_factor_feedback is not None:
                                measurement.observe(_name, _decisive_factor_feedback)
                                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _decisive_factor_feedback))
                                continue
                        _hstate_plus_predict = False
                        if _name == "predict":
                            try:
                                from kernel_stateframe_hstate_plus import history_arm
                                _hstate_plus_predict = history_arm(measurement)
                            except (ImportError, AttributeError, TypeError, ValueError):
                                _hstate_plus_predict = False
                        if (_name == "predict"
                                and (getattr(measurement, "stateframe_get_sure_mode", "off") == "on"
                                     or _hstate_plus_predict)):
                            from kernel_stateframe_get_sure import predict_runtime
                            _prediction = predict_runtime(
                                measurement, tool_predict_args or _args)
                            _prediction_text = json.dumps(
                                _prediction, ensure_ascii=False, sort_keys=True, default=str)
                            bolt.receipt(
                                "predict", "candidate prediction",
                                str((tool_predict_args or _args).get("path", "")),
                                str(_prediction.get("prediction_status", "PREDICTED")),
                                detail=_prediction_text[:240])
                            measurement.observe("predict", _prediction_text)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _prediction_text))
                            continue
                        if _name == "recall":
                            _recall_feedback = measurement.recall_operation(_name, _args)
                            if _recall_feedback is not None:
                                measurement.observe(_name, _recall_feedback)
                                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _recall_feedback))
                                continue
                        _continuation_feedback = measurement.continuation_before_dispatch(_name, _args)
                        if _continuation_feedback is not None:
                            from kernel_navigation_receipt import answer_despite_gate as _nav_answer
                            _nav_feedback = _nav_answer(
                                _name, _args, _continuation_feedback,
                                getattr(measurement, "repo_intel_operation", None))
                            if _nav_feedback is not None:
                                from kernel_navigation_receipt import record_execution_receipt
                                record_execution_receipt(bolt, _name, _args, _nav_feedback)
                                _continuation_feedback = _nav_feedback
                            measurement.observe(_name, _continuation_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _continuation_feedback))
                            continue
                        if action_schema_mode or contract_enabled():
                            measurement.accept_action()
                        if _has_memory_sidecar:
                            measurement.ingest_decision_sidecar(_memory_sidecar)
                        _state_feedback = measurement.retained_operation(_name, _args)
                        if _state_feedback is None:
                            _state_feedback = measurement.state_operation(_name, _args)
                        if _state_feedback is not None:
                            measurement.observe(_name, _state_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _state_feedback))
                            continue
                        _repo_intel_dispatch = getattr(measurement, "repo_intel_operation", None)
                        _repo_intel_feedback = (
                            _repo_intel_dispatch(_name, _args)
                            if callable(_repo_intel_dispatch) else None)
                        if _repo_intel_feedback is not None:
                            from kernel_navigation_receipt import compose as _nav_compose
                            from kernel_navigation_receipt import enabled as _nav_enabled
                            from kernel_navigation_receipt import is_navigation as _nav_is
                            if _nav_is(_name) and _nav_enabled():
                                _repo_intel_feedback = _nav_compose(_repo_intel_feedback)
                                # Q6: a repeated identical NOT FOUND changes the route.
                                from kernel_repeated_zero import observe as _rz_observe
                                _repo_intel_feedback = _rz_observe(
                                    measurement, _name, _args, _repo_intel_feedback)
                            if _nav_is(_name):
                                from kernel_navigation_receipt import record_execution_receipt
                                record_execution_receipt(
                                    bolt, _name, _args, _repo_intel_feedback)
                            measurement.observe(_name, _repo_intel_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _repo_intel_feedback))
                            continue
                        if _working_note is not None:
                            measurement.stage_working_note(
                                _working_note, tool=_name, arguments=_args)
                        _work_feedback = measurement.work_operation(_name, _args)
                        if _work_feedback is not None:
                            measurement.observe(_name, _work_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _work_feedback))
                            continue
                        _job_dispatch = getattr(measurement, "job_operation", None)
                        _job_feedback = (_job_dispatch(_name, _args)
                                         if callable(_job_dispatch) else None)
                        if _job_feedback is not None:
                            measurement.observe(_name, _job_feedback)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _job_feedback))
                            continue
                        if getattr(measurement, "figure_it_out_enabled", False):
                            from kernel_figure_it_out import is_operator
                            if is_operator(_name):
                                _figure_feedback = measurement.figure_it_out_operation(
                                    _name, _args)
                                if _figure_feedback is not None:
                                    try:
                                        _figure_result = json.loads(_figure_feedback)
                                    except (TypeError, ValueError):
                                        _figure_result = {}
                                    _figure_class = str(
                                        _figure_result.get("status", "HOST_RESULT"))
                                    bolt.receipt(
                                        "operator", f"figure_it_out {_name}", _name,
                                        _figure_class, detail=_figure_feedback[:240])
                                    measurement.observe(_name, _figure_feedback)
                                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _figure_feedback))
                                    continue
                        if _name in active_decoy_names and is_decoy_name(_name):
                            decoy_call_names.append(_name)
                            _first_edit_seen = (
                                _progress_facts().first_edit_step is not None
                                if _host_truth_progress_enabled else first_edit_step is not None)
                            if not _first_edit_seen:
                                wrong_tool_before_first_edit = True
                            _dispatch_decoy_call(
                                bolt, msgs, tool_call_id=tool_call_id, name=_name,
                                arguments=_args, decoy_kind=_decoy_kind,
                            )
                            continue
                        from kernel_tool_surface import runner_dispatchable as _runner_dispatchable
                        if _name == "grep" and os.environ.get("STATEFRAME_TOKENLAB_GREP_QUOTE", "off").strip().lower() == "on":
                            # request fitting (lane 4 L1b): the native pattern is ONE argv item; quote it so a
                            # space in the pattern never becomes a bogus scope at the shlex.split below.
                            import shlex as _grep_shlex
                            line = ("grep -- " + _grep_shlex.quote(str(_args.get("pattern") or "")) +
                                    ((" " + _grep_shlex.quote(str(_args["path"]))) if _args.get("path")
                                     else "")).strip()
                        elif _name == "grep":
                            line = ("grep " + str(_args.get("pattern") or "") +
                                    ((" " + str(_args["path"])) if _args.get("path")
                                     else "")).strip()
                        elif _name == "list_files":
                            line = _list_files_command(_args)
                        elif _name == "file_list":
                            import shlex as _file_list_shlex
                            line = "file_list"
                            if _args.get("path"):
                                line += " " + _file_list_shlex.quote(str(_args["path"]))
                            if _args.get("limit"):
                                line += " " + str(_args["limit"])
                        elif _name == "read":
                            line = ("read " + str(_args.get("path") or "") +
                                    ((" " + str(_args["line"])) if _args.get("line")
                                     else "")).strip()
                        elif _name == "edit":
                            line = "edit " + str(_args.get("path") or "")
                            tool_edit_src = str(_args.get("function_source") or "")
                            tool_target_ref = _args.get("target_ref")
                        elif _name == "shell" or (_agent_tools is not None and _name in _agent_tools.names):
                            from kernel_shell_tool import ShellCleanupFailed, ShellExecutionFailed, publish_effects
                            if _name == "shell" and (_shell is None or _shell.settings.shell == "off"):
                                _obs = "REFUSED SHELL_OFF: shell is disabled by --shell off"
                                bolt.receipt("shell", "shell", "off", "REFUSED", _obs)
                            else:
                                _reply = (_shell.execute(_args, tool_call_id) if _name == "shell" else
                                          _agent_tools.execute(_name, _args, tool_call_id))
                                _obs = _reply.observation
                                try:
                                    _rebound = publish_effects(_reply, measurement, bolt, created_this_run, name=_name)
                                except Exception as _error:
                                    raise ShellExecutionFailed(
                                        f"shell receipt publication failed: {type(_error).__name__}: {_error}") from _error
                                if _rebound is not None:
                                    target = _rebound
                                if _reply.applied:
                                    edits += 1
                                    edits_since_test += 1
                                    if first_edit_step is None:
                                        first_edit_step = step
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _obs))
                            measurement.observe(_name, _obs)
                            _edit_nturn_after_action(_name, _obs, step, model_action_args=_args)
                            if _shell is not None and _reply.fatal:
                                raise ShellCleanupFailed(_obs)
                            continue
                        elif _name in {"file_delete", "file_move"}:
                            from kernel_file_ops import FileOperationRefused, FileOperationReplay, file_delete, file_move
                            from kernel_tool_surface import new_files_forbidden

                            _source_arg = _args.get("path") if _name == "file_delete" else _args.get("source")
                            _dest_arg = _args.get("destination") if _name == "file_move" else None
                            _paths = ()
                            _file_replay = FileOperationReplay(
                                root, _name, _args, tool_call_id,
                                authority={"editable": sorted(fileset.paths),
                                           "created": sorted(str(pathlib.Path(p).relative_to(root))
                                                             for p in created_this_run),
                                           "protected": sorted(str(pathlib.Path(p).relative_to(root))
                                                               for p in baseline_tests),
                                           "new_files_forbidden": new_files_forbidden(measurement),
                                           "allowed_product_paths": d.get("allowed_product_paths"),
                                           "new_file_roots": d.get("new_file_roots")},
                                environment_identity=_shell.identity() if _shell is not None else "disabled")
                            try:
                                def _protected(path):
                                    return protected_test_path(path, baseline_tests)

                                def _editable(relative):
                                    return (relative in fileset.paths
                                            or str(root / relative) in created_this_run)
                                if _name == "file_delete":
                                    _effect = file_delete(root, _source_arg, protected=_protected,
                                                          editable=_editable)
                                else:
                                    _effect = file_move(
                                        root, _source_arg, _dest_arg, protected=_protected,
                                        editable=_editable,
                                        new_file_allowed=lambda relative: (
                                            not new_files_forbidden(measurement)
                                            and _row_is_product_path(relative, d)),
                                    )
                                _paths = _effect.changed_paths
                                _obs = f"{_name} applied. {', '.join(_paths)}"
                                from kernel_post_edit_refresh import rebind_after_file_operation
                                _rebound_target = rebind_after_file_operation(
                                    measurement, _effect.source, _effect.destination)
                                if _rebound_target is not None:
                                    target = _rebound_target
                                if _name == "file_move":
                                    created_this_run.discard(str(root / _effect.source))
                                    created_this_run.add(str(root / _effect.destination))
                                else:
                                    created_this_run.discard(str(root / _effect.source))
                                if first_edit_step is None:
                                    first_edit_step = step
                                edits += 1
                                _result = "APPLIED"
                            except FileOperationRefused as _error:
                                _obs = _error.reply()
                                _result = "REFUSED"
                            except OSError as _error:
                                _obs = f"{_name} FAILED: {type(_error).__name__}: {_error}"
                                _result = "FAILED"
                            _obs = _file_replay.complete(_result, _obs)
                            bolt.receipt(
                                "edit", f"{_name} {_source_arg}",
                                f"{_source_arg}|{_dest_arg or ''}", _result, _obs,
                                target_path=(_effect.destination if _name == "file_move"
                                             else _paths[0]) if _paths else None,
                                changed_paths=_paths,
                            )
                            _obs = _append_structure_fact_observation(measurement, bolt, _obs, "edit")
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _obs))
                            _edit_nturn_after_action(
                                f"{_name} {_source_arg}", _obs, step, model_action_args=_args)
                            continue
                        elif _name == "file_create":
                            # E298 graft #1 (coding.py _code_file_create's contract): both
                            # 4B lanes burned 25+ steps retrying edits on files they were
                            # trying to INVENT. New files are legitimate (a repro test, a
                            # helper); grading is immune (pristine-overlay + two-sided
                            # oracle), so let them.
                            _create_rel = str(_args.get("path") or "").strip().lstrip("/")
                            _create_error = None
                            if multifile:
                                try:
                                    _fc, _create_rel = _multi_path(root, _create_rel)
                                except ValueError as _e:
                                    _fc = root / "__invalid_path__"
                                    _create_error = str(_e)
                            else:
                                _fc = root / _create_rel
                                # Bughunt r2AE #3: single-file mode had no containment check,
                                # so "../x.py" wrote outside the checkout with an APPLIED
                                # receipt. Same containment rule (and text) as reads; no
                                # basename recovery, which could redirect a create.
                                if not _create_path_contained(root, _create_rel):
                                    _fc = root / "__invalid_path__"
                                    _create_error = (
                                        f"Path {str(_args.get('path') or '')!r} is outside the "
                                        "workspace. Use a repository-relative path.")
                            _create_before = (_source_snapshot(root, fileset.paths)
                                              if multifile else {})
                            tool_call_id_fc = tool_call_id
                            if _create_error:
                                _obs = f"file_create REJECTED: {_create_error}"
                            elif _fc.exists():
                                if protected_test_path(_fc, baseline_tests):
                                    _obs = (f"file_create REFUSED: {_fc.name} is a protected "
                                            "pre-existing test path; repair the product or "
                                            "create a distinct authorized repro path.")
                                else:
                                    _obs = (f"file_create REFUSED: {_fc.name} already exists — "
                                            f"use str_replace or edit.")
                            else:
                                try:
                                    _content = str(_args.get("content") or "")
                                    _v21_rejection, _v21_warning = _validate_v21_candidate(
                                        _fc, _content, "file_create")
                                    if _v21_rejection is not None:
                                        _obs = _v21_rejection
                                    elif _nonpy_edit_applies(_fc):
                                        _v21_published = _publish_nonpy_candidate(
                                            _fc, _content.encode("utf-8"),
                                            operation="file_create", root=root,
                                            public_brief=_task_public_brief(d),
                                            warning=_v21_warning,
                                            prepare=lambda: _fc.parent.mkdir(
                                                parents=True, exist_ok=True),
                                        )
                                        _obs = (_v21_published.replace(
                                            "file_create applied.",
                                            f"created {_args.get('path')}", 1)
                                            if "file_create applied." in _v21_published
                                            else _v21_published)
                                    elif os.environ.get("STATEFRAME_JQ_FINISH", "off") == "on":
                                        from kernel_jq_finish import validate_and_publish
                                        def _publish_created():
                                            return validate_and_publish(
                                                _fc, _content, root=root,
                                                protected_paths=baseline_tests,
                                                prefix="file_create", require_new=True,
                                                public_brief=_task_public_brief(d))

                                        if _effect_boundary_enabled():
                                            _fc_before = _fc.read_text(errors="ignore") \
                                                if _fc.exists() else "<absent>"
                                            _fc_applied, _published, _fc_rendered = _effect_application(
                                                "file_create",
                                                {"path": str(_fc), "before": _fc_before,
                                                 "after": _content},
                                                _publish_created,
                                                before_snapshot=lambda: {
                                                    "path": str(_fc),
                                                    "content": _fc_before,
                                                },
                                                effect_applied=lambda result: (
                                                    isinstance(result, str)
                                                    and " applied." in result),
                                                after_snapshot=lambda: {
                                                    "path": str(_fc),
                                                    "content": (_fc.read_text(errors="ignore")
                                                                if _fc.exists() else "<absent>"),
                                                },
                                            )
                                            _obs = (f"created {_args.get('path')}"
                                                    if _fc_applied else
                                                    (_published or _fc_rendered or
                                                     "file_create SHADOW: effect withheld"))
                                        else:
                                            _published = _publish_created()
                                            _obs = (f"created {_args.get('path')}"
                                                    if " applied." in _published else _published)
                                    else:
                                        from kernel_destructive_edit_guard import guard_write
                                        _shrink_refusal = guard_write(
                                            _fc, _content, root=root,
                                            public_brief=_task_public_brief(d))
                                        if _shrink_refusal is not None:
                                            _obs = _shrink_refusal
                                            raise _GreenLockRefused
                                        from kernel_green_lock import guard_product_mutation
                                        _green_refusal = guard_product_mutation(_fc)
                                        if _green_refusal is not None:
                                            _obs = "file_create REJECTED: " + _green_refusal
                                            raise _GreenLockRefused
                                        _fc_applied, _fc_rendered = _effect_write(
                                            _fc, _content, operation="file_create",
                                            prepare=lambda: _fc.parent.mkdir(
                                                parents=True, exist_ok=True))
                                        _obs = (f"created {_args.get('path')}"
                                                if _fc_applied else
                                                (_fc_rendered or
                                                 "file_create SHADOW: effect withheld"))
                                    if _obs.startswith("created "):
                                        from kernel_green_lock import note_applied_product_edit
                                        note_applied_product_edit(_fc)
                                        from kernel_rejection_template_restrict import (
                                            release_after_product_mutation,
                                        )
                                        _active_capture = getattr(
                                            measurement, "_active_capture", None)
                                        release_after_product_mutation(
                                            measurement,
                                            request_id=getattr(
                                                _active_capture, "request_id", None),
                                            tool_name="file_create",
                                        )
                                    if first_edit_step is None:
                                        if _obs.startswith("created "):
                                            first_edit_step = step
                                    # E344 (vitals audit blocker #1): file_create was
                                    # the one edit path with no E334 receipt.
                                    bolt.receipt(
                                        "edit", f"file_create {_args.get('path')}",
                                        f"{_args.get('path')}|"
                                        f"{str(_args.get('content'))[:400]}",
                                        "APPLIED" if _obs.startswith("created ") else "REJECTED",
                                        (_obs[:240] if _nonpy_edit_policy_enabled()
                                         else _obs.splitlines()[0][:100]),
                                        target_path=receipt_target_path(root, _fc))
                                    if multifile:
                                        _annotate_receipt(
                                            bolt, paths=fileset.paths, scope=(_create_rel,),
                                            before=_create_before,
                                            after=_source_snapshot(root, fileset.paths),
                                            selectors=tuple(getattr(bolt, "f2p_ids", ())),
                                            failure_tail=_obs)
                                    if _obs.startswith("created "):
                                        created_this_run.add(str(_fc))
                                except _GreenLockRefused:
                                    pass
                                except OSError as _e:
                                    _obs = f"file_create failed: {_e}"
                            if measurement.projector_kind == 'rich' and not _obs.startswith('created '):
                                bolt.receipt('edit', f'file_create {_create_rel}', _create_rel,
                                             'FAILED' if _obs.startswith('file_create failed:') else 'REFUSED',
                                             _obs)
                            _obs = _append_structure_fact_observation(measurement, bolt, _obs, "edit")
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id_fc, _obs))
                            with (POC / "results" /
                                  f"react_live{os.environ.get('REACT_TAG', '')}.log"
                                  ).open("a") as live:
                                live.write(f"[{fn[:26]:28s} s{step + 1:02d}] "
                                           f"{('file_create ' + str(_args.get('path')))[:72]:72s}"
                                           f" -> {_obs[:88]}\n")
                            _edit_nturn_after_action(
                                "file_create " + _create_rel, _obs, step,
                                model_action_args=_args,
                            )
                            continue
                        elif _name == "str_replace":
                            # Rides the `edit` rail (kernel treats it as the action it is:
                            # ledger cleared, edits counted) with its own applicator.
                            if isinstance(_args.get("edits"), list):
                                line = "str_replace"
                                tool_batch_args = dict(_args)
                            else:
                                line = "edit " + str(_args.get("path") or "")
                                tool_str_args = (str(_args.get("old_str") or ""),
                                                 str(_args.get("new_str") or ""))
                                tool_target_ref = _args.get("target_ref")
                        elif _name == "insert_before":
                            line = "edit " + str(_args.get("path") or "")
                            tool_insert_args = {
                                "target_ref": _args.get("target_ref"),
                                "new_str": str(_args.get("new_str") or ""),
                                "check_refs": _args.get("check_refs", ()),
                            }
                            tool_str_args = ("", tool_insert_args["new_str"])
                            tool_target_ref = tool_insert_args["target_ref"]
                        elif _name == "authored_check":
                            line = "authored_check"
                            tool_authored_check_args = dict(_args)
                        elif _name == "replace_module":
                            line = "edit " + str(_args.get("path") or "")
                            tool_replace_module_args = (
                                str(_args.get("expected_revision") or ""),
                                (_args.get("source") if isinstance(_args.get("source"), str) else None),
                                (_args.get("candidate_id") if isinstance(_args.get("candidate_id"), str) else None),
                            )
                        elif _name == "run_tests":
                            line = ("test " + str(_args.get("target") or "")).strip()
                        elif _name == "reproduce":
                            line = "reproduce"
                            tool_repro_args = dict(_args)
                        elif _name == "read_check_receipt":
                            from kernel_check_receipts import observe_check_output
                            _receipt_text = observe_check_output(
                                measurement, bolt,
                                _args.get("handle"),
                                stream=str(_args.get("stream") or "combined"),
                                offset=_args.get("offset", 0),
                                limit=_args.get("limit", 6000),
                            )
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _receipt_text))
                            continue
                        elif _name == "predict":
                            line = "predict"
                            tool_predict_args = dict(_args)
                        elif _name == "probe":
                            line = "probe"
                            tool_probe_args = _normalize_probe_tool_args({
                                **_args,
                                "_dispatch_variant": _dispatch_variant,
                            })
                        elif _name == "dependency_fact":
                            line = "dependency_fact"
                            tool_dependency_fact_args = dict(_args)
                        elif (_name == "dispute_check"
                              and _runner_dispatchable(_name, runtime=measurement)):
                            line = "dispute_check"
                        elif (_name in {"read_ticket", "dispute"}
                              and _runner_dispatchable(_name, runtime=measurement)):
                            line = _name
                        elif _name == "work":
                            line = "work"
                            tool_work_args = _args
                        elif (_name == "decide"
                              and os.environ.get("STATEFRAME_DECISIVE_FACTOR", "off").strip().lower()
                              == "on"):
                            line = "decide"
                            tool_decisive_factor_args = dict(_args)
                        elif _name == "done":
                            line = "done"
                            # E333: carry the model's claim to the gate.
                            _done_summary = str(_args.get("summary") or "")
                            _done_no_change_paths = _args.get("no_change_paths")
                            if action_schema_mode:
                                _done_claim = _args.get("claim")
                                _done_evidence_refs = _args.get("evidence_refs")
                            # (E337 note: `why` capture happens after the chain below.)
                        else:
                            line = None
                            tool_call_id_unknown = tool_call_id
                            unknown_tool_names.append(_name)
                            _first_edit_seen = (
                                _progress_facts().first_edit_step is not None
                                if _host_truth_progress_enabled else first_edit_step is not None)
                            if not _first_edit_seen:
                                wrong_tool_before_first_edit = True
                            bolt.receipt("tool", f"unknown {_name}",
                                         _fn.get("arguments") or "{}", "UNKNOWN_TOOL",
                                         "unknown tool call; recovery requested")
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id_unknown, f"Unknown tool `{_name}`. Use "
                                              + "/".join(dict.fromkeys(
                                                  str(item.get("function", {}).get("name", ""))
                                                  for item in _request_tools
                                                  if isinstance(item, Mapping)
                                                  and isinstance(item.get("function"), Mapping)
                                                  and item.get("function", {}).get("name")))
                                              + "."))
                            continue
                        # E337: declared intent (any tool) rides into the receipt tape.
                        bolt.last_why = str(_args.get("why") or "")[:160]
                        reply = content or ""
                    else:
                        reply = content
                        from kernel_response_truncation import TRUNCATED_ACTION, is_truncated
                        if is_truncated(reply):
                            if _edit_nturn_request_active:
                                _finish_edit_nturn()
                            # r45 rule 11d: typed truncation -- no proposal, no invalid edit; the model gets the note.
                            measurement.start_action(TRUNCATED_ACTION, {}, charge_dispatch=False)
                            measurement.observe(TRUNCATED_ACTION, reply)
                            msgs.append({"role": "user", "content": reply})
                            continue
                        if (_edit_nturn_request_active
                                and is_edit_nturn_no_change_reply(reply)):
                            msgs.append({"role": "assistant", "content": reply})
                            _edit_nturn_after_action(None, reply, step)
                            continue
                        if action_schema_mode or contract_enabled() or native_tool_batches:
                            measurement.start_action('invalid_proposal', {}, charge_dispatch=False)
                        msgs.append({"role": "assistant", "content": reply})
                        if action_schema_mode or contract_enabled() or native_tool_batches:
                            if native_tool_batches:
                                measurement.journal.emit('native_response_rejected',
                                    reason='no valid native tool-call batch', dispatch=False)
                            msgs.append({'role': 'user', 'content':
                                'No tool was executed. Return a valid tool call; prose commands are not executed.'
                                if native_tool_batches else reply or
                                '[step-contract] No complete action was returned. Choose one tool.'})
                            if _edit_nturn_request_active:
                                _finish_edit_nturn()
                            continue
                else:
                    reply = _provider_request_once()
                    _remember_edit_nturn_model_plan(reply)
                    if _edit_nturn_session is not None and _edit_nturn_session.pending:
                        _edit_nturn_session.request_started()
                        _edit_nturn_request_active = True
                    if (_edit_nturn_request_active
                            and is_edit_nturn_no_change_reply(reply)):
                        msgs.append({"role": "assistant", "content": reply})
                        _edit_nturn_after_action(None, reply, step)
                        continue
                    if getattr(measurement, "stateframe_completion_owner_mode", "off") == "on":
                        from kernel_completion_owner import note_model_action
                        note_model_action(measurement, "")
                    measurement.start_action("text", {})
                    msgs.append({"role": "assistant", "content": reply})
            except Exception as e:                              # noqa: BLE001
                _finish_edit_nturn()
                if getattr(e, "boundary_recovery_stop", False):
                    raise
                if _prepared.capture.transport_errors:
                    measurement.harness_fault("transport_exception", _prepared.capture.request_id)
                # HARNESS_TRANSPORT or HARNESS_CONTRACT (audit 2026-10-05 #1): the run
                # ends typed ONCE; no native_response_rejected / invalid_proposal.
                _harness_outcome = getattr(e, "outcome", None)
                if getattr(e, "harness_contract", False):
                    _harness_outcome = "HARNESS_CONTRACT"
                transport_terminal = _harness_outcome in _HARNESS_TERMINAL_OUTCOMES
                if transport_terminal:
                    measurement.system_outcome = _harness_outcome
                from kernel_pilot import ParentBudgetExhausted
                if isinstance(e, ParentBudgetExhausted):
                    measurement.system_outcome = "attempt_budget_exhausted"
                    if action_schema_mode or contract_enabled():
                        raise
                if (action_schema_mode or contract_enabled()) and not transport_terminal:
                    raise
                if transport_terminal:
                    _location = _exception_location(e)
                    print(_location["traceback_tail"], file=sys.stderr, flush=True)
                    _reported_edits, _reported_checks = _reported_progress_counts(locals())
                    outcome = {
                        "result": _harness_outcome, "steps": _reported_steps(),
                        "edits": _reported_edits, "tests": _reported_checks,
                        "behaviour": "unverified",
                        "error_type": type(e).__name__, "failure_origin": "harness",
                        **_location,
                    }
                    measurement.journal.emit(
                        "worker_exception", error_type=type(e).__name__,
                        outcome=_harness_outcome, failure_origin="harness",
                        step=step, **_location)
                    finish_native_batch_transcript(msgs)
                    measurement.finish(outcome, messages=msgs)
                    return _attach_env_receipt(outcome, measurement)
                # r38: never lose the cause of an ERR:* outcome (smoke 295 ran blind).
                import traceback as _err_tb
                _err_trace = _err_tb.format_exc()[-6000:]
                print(_err_trace, flush=True)
                try:
                    measurement.journal.emit(
                        "run_exception", error_type=type(e).__name__,
                        error_detail=str(e)[:600], traceback=_err_trace, dispatch=False)
                except Exception:                                  # noqa: BLE001
                    pass
                finish_native_batch_transcript(msgs)
                _reported_edits, _reported_checks = _reported_progress_counts(locals())
                measurement.finish({"result": f"ERR:{type(e).__name__}",
                                    "error_detail": str(e)[:600] or (_err_trace.strip().splitlines() or [""])[-1][:600],
                                    "steps": _reported_steps(),
                                    "edits": _reported_edits,
                                    "tests": _reported_checks}, messages=msgs)
                return _attach_env_receipt(
                    dict(fn=fn, result=f"ERR:{type(e).__name__}", steps=_reported_steps(),
                         edits=_reported_edits, tests=_reported_checks), measurement)
            finally:
                if _capture_token is not None:
                    ACTIVE_REQUEST.reset(_capture_token)
            # FIND THE COMMAND ANYWHERE IN THE REPLY, not just on line 1.
            #
            # The first version read `splitlines()[0]` and models reason out loud before
            # acting — "Now let me read the suspect function:" then the command on the next
            # line. Measured from the transcripts: first-line tokens parsed as commands
            # included "I'll", "Now", "Let", "I", "Perfect!" — roughly 40% of turns were
            # answered with "Unrecognised command" and thrown away. `edit` appeared in the
            # transcript while the run reported edits=0: THE MODEL ACTED AND THE HARNESS
            # DELETED IT. edits=0 was never a refusal to commit, it was our bug.
            #
            # Scan every line and take the first one that starts with a real verb. Prose is
            # ignored rather than punished, which is how every production coding agent parses.
            _verbs = ("grep ", "read ", "edit ", "test", "reproduce", "probe", "done")
            if os.environ.get("STATEFRAME_GET_SURE", "off").strip().lower() == "on":
                _verbs += ("predict",)
            if _dependency_fact_on:
                _verbs += ("dependency_fact",)
            if os.environ.get("STATEFRAME_DECISIVE_FACTOR", "off").strip().lower() == "on":
                _verbs += ("decide",)
            _destructive_command_probe = None
            if os.environ.get("STATEFRAME_EDIT_SHRINK_GUARD", "off").strip().lower() == "on":
                from kernel_destructive_edit_guard import command_receipt as _destructive_command_probe
            from kernel_where_used import mode as where_used_mode
            if where_used_mode() == "on":
                _verbs += ("file_list",)
            if line is None:                       # toolcall mode may have set it already
                for _ln in (reply or "").strip().splitlines():
                    _s = _ln.strip().lstrip("`").strip()
                    if (_s.startswith(_verbs)
                            or (_destructive_command_probe is not None
                                and _destructive_command_probe(_s, root=root) is not None)):
                        line = _s
                        break
            if line is None:
                # ACCEPT THE MODEL'S XML DIALECT. Measured in the graded v9 run: Flash-0731
                # answered `<read>routers/helm_human_work.py</read>` — a perfectly clear
                # command in a tool-call idiom it was trained on — and the parser nudged it
                # 200 CONSECUTIVE TIMES (402-message transcript, zero commands executed).
                # Same defect class as ```py vs ```python: the model doing the right thing in
                # a dialect the harness refuses. Meet the dialect.
                _command_names = "grep|read|edit|test|reproduce|probe|done"
                if os.environ.get("STATEFRAME_GET_SURE", "off").strip().lower() == "on":
                    _command_names += "|predict"
                if _dependency_fact_on:
                    _command_names += "|dependency_fact"
                if os.environ.get("STATEFRAME_DECISIVE_FACTOR", "off").strip().lower() == "on":
                    _command_names += "|decide"
                _xm = re.search(
                    rf"<({_command_names})>\s*(.*?)\s*</\1>", reply, re.S)
                if _xm:
                    line = f"{_xm.group(1)} {_xm.group(2)}".strip()
                else:
                    # DEFECT #39 — the SINGLE-TAG dialect: `<read routers/agent_tasks.py>`,
                    # argument inside the brackets, no closing tag. Measured in v17: two
                    # tasks died at step 5 (instant HARNESS_SUSPECT) because five clear
                    # commands in this shape were all answered "No command found". Third
                    # dialect miss of the campaign (```py, <verb>arg</verb>, now this) —
                    # same rule every time: meet the dialect, never punish clarity.
                    _xm2 = re.search(
                        rf"<({_command_names})\s+([^<>\n]+?)\s*/?>", reply)
                    if _xm2:
                        line = f"{_xm2.group(1)} {_xm2.group(2)}".strip()

            if (_edit_nturn_request_active and line is not None
                    and not is_edit_action_line(line)):
                _finish_edit_nturn()
            if (not toolcall_mode and line is not None
                    and getattr(measurement, "intent_carry_mode", "off") == "on"):
                measurement.record_model_decision(
                    reasoning=getattr(_prepared.capture, "response_reasoning_content", None),
                    content=reply, action=line, step=step)

            # A raw git mutation is not one of the coding dialects accepted by
            # the normal parser. When the audit is enabled, surface it as a
            # command long enough for the host denial below to fire; the off
            # arm leaves the parser's existing behavior untouched.
            if (line is None and
                    os.environ.get("STATEFRAME_PREPUBLISH_DIFF_AUDIT", "off").strip().lower()
                    == "on"):
                from kernel_prepublish_diff_audit import command_rejection as _prepublish_command_rejection
                for _ln in (reply or "").strip().splitlines():
                    _candidate = _ln.strip().lstrip("`").strip()
                    if _candidate and _prepublish_command_rejection(_candidate) is not None:
                        line = _candidate
                        break

            if line is None:
                if _edit_nturn_request_active:
                    _edit_nturn_after_action(None, reply, step)
                # NEVER DEFAULT TO `done`. This defaulted to "done" and BROKE THE LOOP, so any
                # reply without a recognised verb silently ended the run. Measured from the
                # transcripts: 3 of 6 tasks ended at 9-11 steps of a 36-step budget having
                # never issued `done` — MY PARSER killed them, and I reported it as the model
                # "quitting early". One of them had just been asked for a ```python block,
                # supplied it in the next message, and had the entire edit discarded.
                #
                # A reply with no command is a prompting miss, not a decision to stop. Say so
                # and let the agent continue; ending a run is a decision only the agent or the
                # step budget gets to make.
                blocks = re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", reply, re.S)
                if blocks and pending_edit_path:
                    # A bare code block right after an edit prompt IS the edit.
                    _edit_before = (_source_snapshot(root, fileset.paths)
                                    if multifile else {})
                    from kernel_action_identity import canonical_edit
                    _pending_path = pending_edit_path
                    _pending_rel = str(_pending_path.relative_to(root))
                    _pending_attempt = canonical_edit(bolt, _pending_path, dialect="replace_fn",
                                                      old="", new=blocks[-1].strip())
                    _pending_gate = getattr(bolt, "predispatch_gate", None)
                    _pending_outcome = (_pending_gate.evaluate({**_pending_attempt, "kind": "edit", "turn": step})
                                        if _pending_gate else None)
                    _pending_blocked = bool(_pending_outcome and _pending_outcome.blocks)
                    if _pending_blocked:
                        obs = "edit REJECTED (pre-dispatch retry gate): repeated unchanged failure"
                    elif (_bound_refusal := _bound_edit_refusal(
                            root, pending_edit_path, tool_target_ref)) is not None:
                        obs = "edit REJECTED: " + _bound_refusal
                    elif multifile:
                        _pending_rel = str(pending_edit_path.relative_to(root))
                        if _pending_rel not in fileset.paths:
                            obs = _typed_edit_rejection(
                                "NOT_EDITABLE", pending_edit_path, root=root,
                                hint=(f"{_pending_rel} is not listed in the EditAuthority's "
                                      "editable_paths"))
                        elif protected_test_path(pending_edit_path, baseline_tests):
                            obs = _typed_edit_rejection(
                                "NOT_EDITABLE", pending_edit_path, root=root,
                                hint="pre-existing test files are read-only")
                        elif (fileset.entry_path and fileset.entry_fn) or _definition_mode:
                            _pending_name, _pending_error = _top_level_definition_name(
                                blocks[-1].strip())
                            if _pending_error is not None:
                                obs = _typed_edit_rejection(
                                    "INVALID_FUNCTION_SOURCE", pending_edit_path, root=root,
                                    hint=_pending_error)
                            else:
                                obs = replace_fn(pending_edit_path, _pending_name,
                                                  blocks[-1].strip(), multi_file=True,
                                                  target_ref=tool_target_ref, root=root,
                                                  public_brief=_task_public_brief(d),
                                                  journal=measurement.journal)
                        else:
                            obs = _replace_file(
                                pending_edit_path, blocks[-1].strip(), root=root,
                                public_brief=_task_public_brief(d),
                                journal=measurement.journal)
                    else:
                        obs = replace_fn(pending_edit_path, fn, blocks[-1].strip(),
                                         target_ref=tool_target_ref, root=root,
                                         public_brief=_task_public_brief(d),
                                         journal=measurement.journal)
                    obs = _normalize_edit_rejection(
                        obs, pending_edit_path, root=root,
                        symbol=(fileset.entry_fn if multifile and fileset.entry_fn else fn))
                    if first_edit_step is None and "applied." in obs:
                        first_edit_step = step
                    edits += 1
                    pending_edit_path = None
                    if multifile:
                        _pending_receipt_kind = "edit"
                        bolt.receipt(_pending_receipt_kind, f"replace_fn {_pending_rel}",
                                     f"{_pending_rel}|{blocks[-1][:400]}",
                                     "REFUSED_PREDISPATCH" if _pending_blocked else
                                     "APPLIED" if "applied." in obs else
                                     "REJECTED" if _is_typed_edit_rejection(obs)
                                     or "REJECTED" in obs else "FAILED",
                                     obs, attempt=_pending_attempt,
                                     target_path=receipt_target_path(root, _pending_path))
                        _annotate_receipt(
                            bolt, paths=fileset.paths, scope=(_pending_rel,),
                            before=_edit_before,
                            after=_source_snapshot(root, fileset.paths),
                            selectors=tuple(getattr(bolt, "f2p_ids", ())),
                            failure_tail=obs)
                    else:
                        # This legacy continuation had no post-hoc organ receipt. Keep
                        # baseline organs unchanged while recording actual D3 execution.
                        _pending_event = measurement.journal.emit("action_receipt", receipt={
                            "kind": "edit", "attempt": _pending_attempt, "turn": step,
                            "target_path": receipt_target_path(root, _pending_path),
                            "result_class": "REFUSED_PREDISPATCH" if _pending_blocked else
                            "APPLIED" if "applied." in obs else "FAILED", "detail": obs})
                        _pending_record = {**_pending_event["receipt"], "receipt_id": _pending_event["event_id"]}
                        if _pending_gate:
                            _pending_gate.observe(_pending_record)
                        measurement.completed_receipt(_pending_record, blocks[-1].strip())
                    obs = _append_structure_fact_observation(measurement, bolt, obs, "edit")
                    measurement.observe("edit " + _pending_rel, obs)
                    msgs.append({"role": "user", "content": obs})
                    _edit_nturn_after_action("edit " + _pending_rel, obs, step)
                    continue
                unparsed_streak += 1
                # MID-RUN HARNESS-SUSPECT ABORT (E267). The XML-dialect doom loop nudged the
                # agent 200 CONSECUTIVE times — a whole budget burned against a parser
                # refusal. If five nudges in a row change nothing, the mismatch is ours, not
                # the model's; abort the ATTEMPT (not the task) as HARNESS_SUSPECT so the
                # pilot retries instead of the budget grinding to zero. The sidekick already
                # counts these post-mortem; this is the same judgement made in time to matter.
                if unparsed_streak >= 5:
                    # DEFECT #41: this abort used to RETURN, skipping the grading tail — and
                    # a task that had gone red -> edit -> GREEN (v19 _watchdog, s19) had its
                    # correct tree thrown away because the model wrote five verbless wrap-ups
                    # after finishing. An abort ends the CONVERSATION, never the evidence:
                    # break to the grading tail; edits==0 keeps the old semantics (nothing to
                    # grade) via the not_attempted branch there.
                    if _continue_after_heuristic_terminal("unparsed_replies"):
                        from kernel_terminal_policy import deliver_pending_advisories
                        deliver_pending_advisories(measurement, msgs)
                        continue
                    harness_suspect = (f"{unparsed_streak} consecutive unparsed replies — "
                                       f"dialect mismatch; do not charge the model")
                    break
                msgs.append({"role": "user", "content":
                             "No command found in that reply. Respond with exactly one of: "
                             "grep <pattern> / read <path> [line] / edit <path> (followed by "
                             "a ```python block) / test [path] / done."})
                continue

            # A parsed command breaks the streak — only CONSECUTIVE refusals mean dialect
            # mismatch; occasional prose-only replies are normal model behaviour.
            unparsed_streak = 0
            # FIXED-POINT DETECTION (defect #37): identical QUERY 6x in a row -> raise the
            # temperature for the rest of the attempt. Queries only — repeated edits/tests
            # are iteration (the payload differs) and must never be treated as stalling.
            # ALTERNATION DEFEATS ADJACENCY (defect #37, fourth half). Measured v16
            # article_publishable_projection: a grep/read/grep/read two-cycle reached kernel
            # repeat #76 while `line == last_line` was false on every single step — the
            # consecutive-identity detector saw streak 0 forever. The kernel's ledger counts
            # repeats regardless of adjacency, so consult IT: any single query re-issued 10+
            # times is a loop, whatever rhythm it drums.
            _worst_repeat = max(bolt._repeat_counts.values(), default=0)
            if _repeat_query_surgery_eligible(
                    line, interventions=interventions, worst_repeat=_worst_repeat,
                    step=step, last_surgery_step=last_surgery_step):
                del msgs[action_anchor:]
                msgs.append({"role": "user", "content":
                             f"[kernel] You have been cycling the same queries (one re-issued "
                             f"{_worst_repeat} times); those exchanges were removed from the "
                             f"conversation. Their answers will not change until you edit "
                             f"something. State: {_progress_facts().edits} edit(s), "
                             f"{_progress_facts().checks} test run(s). Take a "
                             f"DIFFERENT action — `edit` `{d['rel']}` or `test`."})
                bolt._repeat_counts.clear()
                last_surgery_step = step
                same_streak, last_line = 0, None
                if measurement.projector_kind == 'rich':
                    bolt.receipt(line.split()[0], line, line, 'REPEAT_SUPPRESSED',
                                 detail=msgs[-1]['content'])
                    measurement.observe(line, msgs[-1]['content'])
                continue
            if _same_query_suppression_eligible(line, last_line):
                if same_streak == 0:
                    streak_anchor = len(msgs) - 1   # before this repeat's assistant msg
                same_streak += 1
                from kernel_pilot import POLICY
                if interventions:
                    run_temp = POLICY.repeat_temperature(same_streak, run_temp)
                # CONTEXT SURGERY (defect #37, third half — measured: the escalating replay
                # AND the temp bump both failed; v14 hit repeat #47). Every repeated
                # exchange APPENDED to the conversation makes repetition more likely — the
                # context itself becomes the fuel, and by the time the temperature rises the
                # conditioning has already saturated. So stop feeding the loop back to
                # itself: excise the repeated exchanges from the conversation and replace
                # them with one stateful reframe. The model keeps everything it learned
                # BEFORE the loop; the loop itself never happened, context-wise.
                if interventions and same_streak >= 8:
                    del msgs[streak_anchor:]
                    msgs.append({"role": "user", "content":
                                 f"[kernel] You repeated `{line}` {same_streak} times; those "
                                 f"exchanges have been removed from the conversation. The "
                                 f"answer has not changed and will not change until you edit "
                                 f"something. State so far: {_progress_facts().edits} edit(s), "
                                 f"{_progress_facts().checks} test run(s). Take a DIFFERENT action now — `read` an unread "
                                 f"file, `edit` `{d['rel']}`, or `test`."})
                    same_streak = 0
                    last_line = None
                    if measurement.projector_kind == 'rich':
                        bolt.receipt(line.split()[0], line, line, 'REPEAT_SUPPRESSED',
                                     detail=msgs[-1]['content'])
                        measurement.observe(line, msgs[-1]['content'])
                    continue
            else:
                same_streak = 0
            last_line = line

            # DEFECT #43 — MASHED VERBS. v20: `test read routers/helm_human_work.py` repeated
            # 20+ times, each executed as pytest-on-a-router (exit 5) and never counted by the
            # query-loop defenses because `test` is an action. When the first verb's argument
            # BEGINS with another verb, the inner command is the intent — rewrite before
            # dispatch and note the interpretation in the observation.
            # Outer verb `test` ONLY: `grep test ...` legitimately searches for the word
            # "test", but pytest can never sensibly target `read <path>`/`grep <pat>`.
            _mash = line.split(" ", 2)
            if len(_mash) >= 2 and _mash[0] == "test" and _mash[1] in ("read", "grep", "edit"):
                line = line.split(" ", 1)[1].strip()
            from kernel_destructive_edit_guard import command_receipt
            _destructive_refusal = command_receipt(line, root=root)
            if _destructive_refusal is not None:
                bolt.receipt(
                    line.split()[0] if line.split() else "command",
                    line, line, "REJECTED", detail=_destructive_refusal)
                measurement.observe(line, _destructive_refusal)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _destructive_refusal) if tool_call_id else
                            {"role": "user", "content": _destructive_refusal})
                continue
            # Shell-invocation targets get the same deterministic answer at repeat 1 —
            # require_isol s30-s35 cycled SIX textually-different spellings of the same
            # futile idea (`test python -c ...`, `test bash -c ...`), which per-line
            # repeat counting can never see. Answer the INTENT: pytest is the runner.
            # E347 (deploop occurrence-2): an exact-identical futile command already
            # answered with a card gets a one-line suppression, not the card again —
            # re-feeding the full card back is loop fuel (E328's own lesson).
            if not hasattr(bolt, "_futile_answered"):
                bolt._futile_answered = set()
            _shellish = line.split(" ", 1)[1].strip() if line.startswith("test ") else ""
            if _shellish.split(" ")[0] in ("python", "python3", "bash", "sh") or \
                    " -c " in _shellish[:40]:
                if measurement.projector_kind == 'rich':
                    bolt.receipt('test', line, line, 'UNSUPPORTED_ACTION', 'Only pytest selectors are supported')
                if line in bolt._futile_answered:
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, "[kernel] Same futile command again — the "
                                            "answer has not changed: test <path>.")
                                if tool_call_id else
                                {"role": "user", "content": "[kernel] Same futile "
                                 "command; answer unchanged: test <path>."})
                    continue
                bolt._futile_answered.add(line)
                _tf_hint = next((w for w in _shellish.split() if w.endswith(".py")), None)
                _hint = (f" To run your file with pytest: test {_tf_hint}"
                         if _tf_hint else "")
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, "[kernel] Shell/python invocation is not a tool "
                             "here — pytest is the ONLY runner, and `test <path>` is how "
                             "to invoke it." + _hint)
                            if tool_call_id else
                            {"role": "user", "content": "[kernel] pytest is the only "
                             "runner: use `test <path>`." + _hint})
                continue
            # Package-install requests get a deterministic answer instead of an exit-4
            # loop (measured: `test pip install feedparser` repeated s29->s32+).
            if "pip install" in line or line.startswith(("pip ", "test pip")):
                if measurement.projector_kind == 'rich':
                    bolt.receipt('test', line, line, 'UNSUPPORTED_ACTION', 'Package installation is unavailable')
                if line in bolt._futile_answered:
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, "[kernel] Same futile command again — deps "
                                            "are preinstalled; run: test <path>.")
                                if tool_call_id else
                                {"role": "user", "content": "[kernel] Same futile "
                                 "command; deps preinstalled. Run: test <path>."})
                    continue
                bolt._futile_answered.add(line)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, "[kernel] Package installation is not a tool "
                             "here and is not needed — the environment's dependencies "
                             "are managed by the harness (feedparser and tiktoken are "
                             "installed). Run your test directly: test <path>.")
                            if tool_call_id else
                            {"role": "user", "content": "[kernel] pip is unavailable; "
                             "dependencies are preinstalled. Run: test <path>."})
                continue

            if not information.command_allowed(root, line):
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, "[information-policy] Path outside workspace denied.")
                            if tool_call_id else {"role": "user", "content":
                            "[information-policy] Path outside workspace denied."})
                continue
            from kernel_prepublish_diff_audit import command_rejection as _prepublish_command_rejection
            _prepublish_command_refusal = _prepublish_command_rejection(line)
            if _prepublish_command_refusal is not None:
                measurement.journal.emit(
                    "prepublish_git_command_denied",
                    producer="kernel_prepublish_diff_audit",
                    command=" ".join(line.split())[:300],
                    reason=_prepublish_command_refusal,
                    dispatch=False,
                )
                bolt.receipt("command", line, line, "REJECTED",
                             detail=_prepublish_command_refusal)
                measurement.observe(line, _prepublish_command_refusal)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _prepublish_command_refusal)
                            if tool_call_id else {"role": "user", "content":
                            _prepublish_command_refusal})
                continue
            from kernel_green_lock import guard_action
            _green_action_kind = (_name if toolcall_mode and isinstance(_name, str) and _name
                                  else line.split()[0])
            # Tool-call mode is guarded BEFORE continuation authorization (above); guarding
            # again here would consume the repeat-to-override on the same call.
            _green_action_refusal = (
                None if toolcall_mode or line == "authored_check" else guard_action(
                    _green_action_kind, {"command": " ".join(line.split())})
            )
            if _green_action_refusal is not None:
                measurement.observe(line, _green_action_refusal)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _green_action_refusal) if tool_call_id else
                            {"role": "user", "content": _green_action_refusal})
                continue
            if not toolcall_mode:
                measurement.current_action = {"kind": line.split()[0],
                                              "arguments": {"command": " ".join(line.split())}}
                _legacy_grounding_name = line.split()[0]
                _legacy_grounding_args = {}
                _legacy_tail = line.split(None, 1)[1] if len(line.split(None, 1)) == 2 else ""
                if _legacy_grounding_name in {"read", "edit", "grep", "test"}:
                    _legacy_grounding_args["path" if _legacy_grounding_name != "test" else "target"] = (
                        _legacy_tail.split(None, 1)[0] if _legacy_tail else "")
                elif _legacy_grounding_name == "done":
                    _legacy_grounding_args["summary"] = _legacy_tail
                _legacy_grounding_feedback = measurement.claim_grounding_before_dispatch(
                    _legacy_grounding_name, _legacy_grounding_args)
                if _legacy_grounding_feedback is not None:
                    bolt.receipt(
                        "edit", f"{_legacy_grounding_name} claim grounding",
                        str(_legacy_grounding_args.get("path") or _legacy_grounding_name),
                        "REJECTED", _legacy_grounding_feedback,
                        target_path=str(_legacy_grounding_args.get("path") or "") or None,
                    )
                    measurement.observe(_legacy_grounding_name, _legacy_grounding_feedback)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _legacy_grounding_feedback) if tool_call_id else
                                {"role": "user", "content": _legacy_grounding_feedback})
                    continue
            if line in {"read_ticket", "dispute"}:
                from kernel_tool_surface import tool_allowed as _tool_surface_allowed
                if not _tool_surface_allowed(line, runtime=measurement):
                    _surface_feedback = f"Tool `{line}` is not available in this frame."
                    bolt.receipt("tool", f"unavailable {line}",
                                 json.dumps(_args, sort_keys=True), "UNKNOWN_TOOL",
                                 _surface_feedback)
                    measurement.observe(line, _surface_feedback)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _surface_feedback) if tool_call_id else
                                {"role": "user", "content": _surface_feedback})
                    continue
                _ledger_action = getattr(measurement, line)
                _ledger_result = _ledger_action(_args if isinstance(_args, Mapping) else {})
                bolt.receipt("test", line, json.dumps(_args, sort_keys=True),
                             "LEDGER_RESPONSE", detail=_ledger_result[:240])
                measurement.observe(line, _ledger_result)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _ledger_result) if tool_call_id else
                            {"role": "user", "content": _ledger_result})
                continue
            if line == "dispute_check":
                _dispute = measurement.dispute_check(_args if isinstance(_args, Mapping) else {})
                bolt.receipt("test", "dispute_check", json.dumps(_args, sort_keys=True),
                             "DISPUTE_RETIRED" if '"status": "RETIRED"' in _dispute else
                             "DISPUTE_REFUSED", detail=_dispute[:240])
                measurement.observe(line, _dispute)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _dispute) if tool_call_id else
                            {"role": "user", "content": _dispute})
                continue
            if line == "decide" or line.startswith("decide "):
                if tool_decisive_factor_args is None:
                    raw_decision = line[len("decide"):].strip()
                    try:
                        parsed_decision = json.loads(raw_decision) if raw_decision else {}
                    except (TypeError, ValueError):
                        parsed_decision = {}
                    if isinstance(parsed_decision, Mapping):
                        tool_decisive_factor_args = dict(parsed_decision)
                _decision_feedback = measurement.decisive_factor_operation(
                    "decide", tool_decisive_factor_args or {})
                if _decision_feedback is None:
                    _decision_feedback = "decide unavailable; completion not blocked."
                bolt.receipt(
                    "decision", "decide", json.dumps(tool_decisive_factor_args or {}, sort_keys=True),
                    "RECORDED" if "factor=" in _decision_feedback else "ADVISORY",
                    detail=_decision_feedback[:240],
                )
                measurement.observe("decide", _decision_feedback)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _decision_feedback) if tool_call_id else
                            {"role": "user", "content": _decision_feedback})
                continue
            _hstate_plus_predict = False
            if line == "predict" or line.startswith("predict "):
                try:
                    from kernel_stateframe_hstate_plus import history_arm
                    _hstate_plus_predict = history_arm(measurement)
                except (ImportError, AttributeError, TypeError, ValueError):
                    _hstate_plus_predict = False
            if (line == "predict" or line.startswith("predict ")) \
                    and (getattr(measurement, "stateframe_get_sure_mode", "off") == "on"
                         or _hstate_plus_predict):
                if tool_predict_args is None:
                    raw_predict = line[len("predict"):].strip()
                    try:
                        parsed_predict = json.loads(raw_predict) if raw_predict else {}
                    except (TypeError, ValueError):
                        parsed_predict = None
                    if isinstance(parsed_predict, Mapping):
                        tool_predict_args = dict(parsed_predict)
                from kernel_stateframe_get_sure import predict_runtime
                _prediction = predict_runtime(measurement, tool_predict_args or _args)
                _prediction_text = json.dumps(
                    _prediction, ensure_ascii=False, sort_keys=True, default=str)
                bolt.receipt(
                    "predict", "candidate prediction",
                    str((tool_predict_args or _args).get("path", "")),
                    str(_prediction.get("prediction_status", "PREDICTED")),
                    detail=_prediction_text[:240])
                measurement.observe("predict", _prediction_text)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, _prediction_text) if tool_call_id else
                            {"role": "user", "content": _prediction_text})
                continue
            if (line == "probe" or line.startswith("probe ")) \
                    and isinstance(tool_probe_args, Mapping) \
                    and tool_probe_args.get("mode") == "dependency_fact":
                tool_dependency_fact_args = dict(tool_probe_args)
                line = "dependency_fact"
            if line == "dependency_fact" or line.startswith("dependency_fact "):
                if not _dependency_fact_on:
                    obs = "dependency_fact unavailable — STATEFRAME_DEPENDENCY_FACT is off"
                else:
                    if tool_dependency_fact_args is None:
                        raw_dependency = line[len("dependency_fact"):].strip()
                        try:
                            parsed_dependency = json.loads(raw_dependency)
                        except (TypeError, ValueError):
                            parsed_dependency = None
                        if isinstance(parsed_dependency, Mapping):
                            tool_dependency_fact_args = dict(parsed_dependency)
                    if not isinstance(tool_dependency_fact_args, Mapping):
                        from kernel_dependency_probe import (
                            DependencyFactResult, DependencyFactStatus,
                            render_dependency_fact,
                        )
                        _dependency_result = DependencyFactResult(
                            status=DependencyFactStatus.INVALID,
                            reason="arguments_required",
                        )
                        measurement.record_dependency_fact(_dependency_result)
                        obs = render_dependency_fact(_dependency_result)
                    else:
                        from kernel_dependency_probe import render_dependency_fact, run_dependency_fact
                        _dependency_args = tool_dependency_fact_args
                        _dependency_package = _dependency_args.get("package")
                        _dependency_target = _dependency_args.get("target")
                        if dependency_fact_runner is None:
                            _dependency_result = run_dependency_fact(
                                root, _dependency_package, _dependency_target,
                                module=_dependency_args.get("module"),
                                args=_dependency_args.get("args"),
                                kwargs=_dependency_args.get("kwargs"),
                                call=_dependency_args.get("call", True) is not False,
                                question=_dependency_args.get("question"), row=d)
                        else:
                            _dependency_result = dependency_fact_runner.run(
                                _dependency_package, _dependency_target,
                                module=_dependency_args.get("module"),
                                args=_dependency_args.get("args"),
                                kwargs=_dependency_args.get("kwargs"),
                                call=_dependency_args.get("call", True) is not False,
                                question=_dependency_args.get("question"))
                        measurement.record_dependency_fact(_dependency_result)
                        _dependency_rendered = render_dependency_fact(_dependency_result)
                        _dependency_material = (
                            f"{_dependency_package}:{_dependency_target}")
                        bolt.receipt(
                            "dependency_fact", _dependency_material,
                            json.dumps(dict(_dependency_args), sort_keys=True, default=str),
                            _dependency_result.status.value,
                            detail=_dependency_rendered[:240])
                        obs = _dependency_rendered
                measurement.observe("dependency_fact", obs)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else
                            {"role": "user", "content": obs})
                continue
            if line == "authored_check" or line.startswith("authored_check "):
                if tool_authored_check_args is None:
                    raw_authored = line[len("authored_check"):].strip()
                    try:
                        parsed_authored = json.loads(raw_authored) if raw_authored else {}
                    except (TypeError, ValueError):
                        parsed_authored = None
                    if isinstance(parsed_authored, Mapping):
                        tool_authored_check_args = dict(parsed_authored)
                authored_args = tool_authored_check_args
                parent_snapshot = {"revision": ""}
                candidate_snapshot = {"revision": ""}
                if not isinstance(authored_args, Mapping):
                    authored_result: dict[str, object] = {
                        "status": "UNRUNNABLE", "result": "UNRUNNABLE",
                        "certifies_clause": False, "reason": "authored_check_arguments_required",
                    }
                else:
                    parent_snapshot, candidate_snapshot, snapshot_error = _authored_check_snapshots(
                        root, measurement)
                    if snapshot_error is not None:
                        authored_result = {
                            "status": "UNRUNNABLE", "result": "UNRUNNABLE",
                            "certifies_clause": False, "reason": snapshot_error,
                            "parent_revision": parent_snapshot.get("revision"),
                            "candidate_revision": candidate_snapshot.get("revision"),
                        }
                    else:
                        authored_proposal = dict(authored_args)
                        authored_proposal["public_text"] = (
                            authored_proposal.get("public_text") or _task_public_brief(d))
                        authored_proposal["parent_revision"] = parent_snapshot["revision"]
                        authored_proposal["candidate_revision"] = candidate_snapshot["revision"]
                        protected = []
                        for protected_path in baseline_tests:
                            try:
                                protected.append(pathlib.Path(protected_path).resolve()
                                                 .relative_to(root.resolve()).as_posix())
                            except (OSError, ValueError):
                                continue
                        extra_protected = authored_proposal.get("protected_paths", ())
                        if isinstance(extra_protected, Sequence) and not isinstance(
                                extra_protected, (str, bytes, bytearray)):
                            protected.extend(str(item) for item in extra_protected)
                        authored_proposal["protected_paths"] = tuple(dict.fromkeys(protected))
                        from kernel_clause_discriminator import execute_authored_check_proposal
                        authored_result = execute_authored_check_proposal(
                            measurement, authored_proposal,
                            parent_snapshot=parent_snapshot,
                            candidate_snapshot=candidate_snapshot,
                        )
                obs = json.dumps(authored_result, sort_keys=True, ensure_ascii=False, default=str)
                authored_status = str(authored_result.get("status") or authored_result.get("result") or "UNKNOWN")
                bolt.receipt(
                    "test", "authored_check",
                    json.dumps({key: authored_args.get(key) for key in (
                        "check_id", "requirement_id", "path", "selector", "language",
                    )} if isinstance(authored_args, Mapping) else {}, sort_keys=True, default=str),
                    authored_status, detail=obs[:240],
                )
                if getattr(bolt, "_receipts", None):
                    authored_receipt = bolt._receipts[-1]
                    authored_receipt.update({
                        "authored_check": True,
                        "receipt_id": authored_result.get("receipt_id"),
                        "check_id": authored_result.get(
                            "check_id", authored_args.get("check_id")
                            if isinstance(authored_args, Mapping) else None),
                        "clause_id": authored_result.get(
                            "clause_id", authored_result.get(
                                "clause_ref", authored_args.get("requirement_id")
                                if isinstance(authored_args, Mapping) else None)),
                        "selector": authored_result.get(
                            "selector", authored_args.get("selector")
                            if isinstance(authored_args, Mapping) else None),
                        "check_hash": authored_result.get("check_hash"),
                        "content_sha256": authored_result.get("content_sha256"),
                        "environment_digest": authored_result.get("environment_digest"),
                        "write_authority": authored_result.get("write_authority") or (
                            authored_args.get("write_authority")
                            if isinstance(authored_args, Mapping) else None),
                        "requirement_quote": authored_result.get("requirement_quote") or (
                            authored_args.get("requirement_quote")
                            if isinstance(authored_args, Mapping) else None),
                        "observation_only": authored_result.get("observation_only") is True,
                        "result": authored_result.get("result"),
                        "provenance": authored_result.get("provenance"),
                        "parent_revision": authored_result.get(
                            "parent_revision", parent_snapshot.get("revision")),
                        "candidate_revision": authored_result.get(
                            "candidate_revision", candidate_snapshot.get("revision")),
                        "certifies_clause": authored_result.get("certifies_clause") is True,
                    })
                    measurement.amend_receipt(authored_receipt)
                tests += 1
                edits_since_test = 0
                measurement.observe("authored_check", obs)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else
                            {"role": "user", "content": obs})
                tool_authored_check_args = None
                continue
            if line == "probe" or line.startswith("probe "):
                if _observation_probe_on:
                    if tool_probe_args is None:
                        raw_probe = line[5:].strip()
                        try:
                            parsed_probe = json.loads(raw_probe)
                        except (TypeError, ValueError):
                            parsed_probe = None
                        if isinstance(parsed_probe, Mapping):
                            tool_probe_args = _normalize_probe_tool_args(parsed_probe)
                    if not isinstance(tool_probe_args, Mapping):
                        obs = "probe NOT_RUN receipt=none reason=code_required"
                    else:
                        from kernel_probe import render_probe_receipt
                        _probe_code = str(tool_probe_args.get("code") or "")
                        _probe_question = str(tool_probe_args.get("question") or "") or None
                        if observation_probe_runner is None:
                            obs = "probe RUNNER_ERROR receipt=none reason=runner_unavailable"
                        else:
                            _behavior_on = os.environ.get(
                                "STATEFRAME_BEHAVIOR_WITNESS", "off").strip().lower() == "on"
                            _clause_review_on = str(
                                getattr(measurement, "stateframe_clause_review_mode", "")
                                or os.environ.get("STATEFRAME_CLAUSE_REVIEW", "off")
                            ).strip().lower() == "on"
                            _review_mode = str(tool_probe_args.get("mode") or "python")
                            if _review_mode == "clause_discriminator_propose":
                                from kernel_clause_discriminator import execute_proposal
                                _discriminator_result = execute_proposal(
                                    measurement,
                                    {
                                        "clause_ref": tool_probe_args.get("clause_ref"),
                                        "entrypoint": tool_probe_args.get("entrypoint"),
                                        "source_unit_handle": tool_probe_args.get("source_unit_handle"),
                                        "input": tool_probe_args.get("input"),
                                        "expected_relation": tool_probe_args.get("expected_relation"),
                                        "token_count": tool_probe_args.get("token_count"),
                                    },
                                    observation_probe_runner,
                                )
                                obs = json.dumps(
                                    dict(_discriminator_result), sort_keys=True, default=str)
                                bolt.receipt(
                                    "probe", str(tool_probe_args.get("clause_ref") or "clause_discriminator"),
                                    obs, str(_discriminator_result.get("status") or "HOST_OBSERVED"),
                                    detail=obs[:240])
                            elif _clause_review_on and _review_mode in {
                                    "clause_review_propose", "clause_review_assess"}:
                                from kernel_clause_review import (
                                    PROPOSE_MODE, assess_packet, execute_packet,
                                    render_result,
                                )
                                if _review_mode == PROPOSE_MODE:
                                    _review_result = execute_packet(
                                        measurement, tool_probe_args.get("packet"),
                                        observation_probe_runner)
                                else:
                                    _review_result = assess_packet(
                                        measurement, tool_probe_args.get("assessment"))
                                obs = render_result(_review_result)
                                bolt.receipt(
                                    "probe", str(tool_probe_args.get("episode_id") or "clause_review"),
                                    json.dumps(dict(_review_result), sort_keys=True, default=str),
                                    str(_review_result.get("status") or "HOST_OBSERVED"),
                                    detail=obs[:240])
                            elif _behavior_on and _review_mode == "behavior_witness":
                                from kernel_behavior_witness import (
                                    admit_proposal, context_for_runtime, execute_admitted,
                                    probe_source, record_refused_probe,
                                    selected_host_obligation,
                                )
                                from kernel_clause_discriminator import settle_witness_execution
                                _clause_id = str(tool_probe_args.get("clause_id") or "")
                                _brief, _interface, _clause = context_for_runtime(
                                    measurement, d, _clause_id)
                                _host_obligation = selected_host_obligation(
                                    measurement, _clause_id, _clause)
                                if _host_obligation is not None:
                                    # The host row is authoritative for the clause
                                    # namespace. The AST index remains authoritative
                                    # for callable visibility and call shape.
                                    _clause = str(_host_obligation.get("clause") or _clause)
                                _proposal = {
                                    "clause_id": _clause_id,
                                    "entrypoint": str(tool_probe_args.get("entrypoint") or ""),
                                    "source_unit_handle": str(tool_probe_args.get("source_unit_handle") or ""),
                                    "input": tool_probe_args.get("input"),
                                    "expected": tool_probe_args.get("expected"),
                                    "rationale": str(tool_probe_args.get("rationale") or ""),
                                    "args": tool_probe_args.get("args"),
                                    "kwargs": tool_probe_args.get("kwargs"),
                                    "projection": tool_probe_args.get("projection"),
                                }
                                _admission = admit_proposal(
                                    _proposal, clause_id=_clause_id, clause=_clause,
                                    brief=_brief, interface=_interface,
                                    revision=measurement.binding(), journal=measurement.journal,
                                    obligation=_host_obligation,
                                    open_obligation=_host_obligation,
                                    open_obligations=tuple(
                                        getattr(measurement, "host_closure_assessments", ()) or ()),
                                    changed_surface=(
                                        _host_obligation.get("changed_surface")
                                        if isinstance(_host_obligation, Mapping) else None),
                                )
                                if not _admission.get("admitted"):
                                    _unrunnable = record_refused_probe(
                                        measurement, _proposal,
                                        reason=str(_admission.get("reason") or "admission_rejected"),
                                        clause_id=_clause_id, clause=_clause, bolt=bolt)
                                    obs = ("behavior witness UNRUNNABLE "
                                           + json.dumps(_unrunnable, sort_keys=True, default=str))
                                else:
                                    _bound = execute_admitted(
                                        measurement, _admission, observation_probe_runner,
                                        question=_probe_question)
                                    _record_idealpath_probe = getattr(
                                        measurement, "idealpath_record_probe_observation", None)
                                    if callable(_record_idealpath_probe):
                                        _record_idealpath_probe(
                                            (str(_clause_id or _probe_question or "probe"),),
                                            _bound.get("receipt_status", _bound.get("result", "UNKNOWN")),
                                            json.dumps(_bound, sort_keys=True, default=str),
                                        )
                                    settle_witness_execution(measurement, _admission, _bound)
                                    _probe_material = _clause_id or _probe_question or "probe"
                                    bolt.receipt(
                                        "probe", _probe_material,
                                        probe_source(_admission),
                                        _bound.get("receipt_status", _bound.get("result", "UNKNOWN")),
                                        detail=json.dumps(_bound, sort_keys=True, default=str)[:150])
                                    obs = ("behavior witness " + str(_bound["result"]) +
                                           " " + json.dumps({
                                               "clause_id": _bound["clause_id"],
                                               "input": _bound["input"],
                                               "expected": _bound["expected"],
                                               "observed": _bound["observed"],
                                               "repair_required": _bound["result"] != "PASS",
                                           }, sort_keys=True, default=str))
                            else:
                                _observation_receipt = observation_probe_runner.run(
                                    _probe_code, _probe_question)
                                measurement.record_observation_probe(_observation_receipt)
                                _record_idealpath_probe = getattr(
                                    measurement, "idealpath_record_probe_observation", None)
                                if callable(_record_idealpath_probe):
                                    _record_idealpath_probe(
                                        (str(_probe_question or "probe"),),
                                        _observation_receipt.status,
                                        render_probe_receipt(_observation_receipt),
                                    )
                                _probe_material = _probe_question or "probe"
                                bolt.receipt(
                                    "probe", _probe_material, _probe_code,
                                    _observation_receipt.status.value,
                                    detail=render_probe_receipt(_observation_receipt)[:150])
                                obs = render_probe_receipt(_observation_receipt)
                elif not _probe_gate_on:
                    obs = "probe unavailable — STATEFRAME_PROBE_GATE is off"
                else:
                    if tool_probe_args is None:
                        raw_probe = line[5:].strip()
                        try:
                            parsed_probe = json.loads(raw_probe)
                        except (TypeError, ValueError):
                            parsed_probe = None
                        if isinstance(parsed_probe, Mapping):
                            tool_probe_args = {
                                "obligation": str(parsed_probe.get("obligation") or ""),
                                "code": str(parsed_probe.get("code") or ""),
                                "expected": str(parsed_probe.get("expected") or ""),
                                "question": str(parsed_probe.get("question") or ""),
                            }
                    if not isinstance(tool_probe_args, Mapping):
                        obs = ("probe REJECTED — use JSON with obligation, code, and "
                               "optional expected fields")
                    else:
                        _probe_id = str(tool_probe_args.get("obligation") or "")
                        _probe_code = str(tool_probe_args.get("code") or "")
                        _probe_expected = str(tool_probe_args.get("expected") or "")
                        _probe_raw = run_probe(root, _probe_code, row=d)
                        if _probe_raw.startswith("probe REJECTED"):
                            obs = _probe_raw
                        else:
                            _probe_record = measurement.record_probe(
                                _probe_id, _probe_raw, _probe_expected)
                            _record_idealpath_probe = getattr(
                                measurement, "idealpath_record_probe_observation", None)
                            if callable(_record_idealpath_probe):
                                _record_idealpath_probe(
                                    (str(_probe_id or "probe"),),
                                    "FAIL" if any(marker in _probe_raw for marker in (
                                        "AssertionError", "ImportError", "ModuleNotFoundError",
                                    )) else "PASS",
                                    _probe_raw,
                                )
                            if _probe_record.get("accepted"):
                                bolt.receipt(
                                    "probe", _probe_id, _probe_code,
                                    "HOST_OBSERVED",
                                    detail=_probe_raw[:240])  # expectation stays MODEL_INTERPRETATION in the gate record
                                obs = (_probe_raw +
                                       "\n[kernel] probe recorded: HOST_OBSERVED; "
                                       "the supplied expectation remains MODEL_INTERPRETATION")
                            else:
                                obs = (_probe_raw + "\nprobe not recorded — "
                                       + str(_probe_record.get("reason", "rejected")))
                measurement.observe("probe", obs)
                msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else
                            {"role": "user", "content": obs})
                continue
            if line == "reproduce" or line.startswith("reproduce "):
                if tool_repro_args is None:
                    raw_repro = line[len("reproduce"):].strip()
                    try:
                        parsed_repro = json.loads(raw_repro)
                    except (TypeError, ValueError):
                        parsed_repro = None
                    tool_repro_args = dict(parsed_repro) if isinstance(parsed_repro, Mapping) else None
                _idealpath_dispatch = getattr(measurement, "idealpath_operation", None)
                _idealpath_feedback = (
                    _idealpath_dispatch("reproduce", tool_repro_args or {})
                    if callable(_idealpath_dispatch) else None
                )
                if _idealpath_feedback is not None:
                    measurement.observe("reproduce", _idealpath_feedback)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _idealpath_feedback) if tool_call_id else
                                {"role": "user", "content": _idealpath_feedback})
                    continue
                _reproduce_feedback(tool_repro_args or {}, tool_call_id)
                continue
            if line.startswith('test'):
                _public_probe = measurement.public_probe_operation(line[4:].strip())
                if _public_probe is not None:
                    from kernel_probe_validity_gate import public_probe_result_class
                    bolt.receipt('test', line, line,
                                 public_probe_result_class(_public_probe), detail=_public_probe)
                    measurement.observe(line, _public_probe)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _public_probe)
                                if tool_call_id else {'role': 'user', 'content': _public_probe})
                    continue
                reused = measurement.redundant_verification(line[4:].strip())
                if reused:
                    reused = measurement.green_lock.decorate_test_result(reused)
                    if getattr(measurement, "stateframe_first_green_disclosure_mode", "off") == "on":
                        from kernel_first_green_disclosure import decorate_test_result
                        reused = decorate_test_result(measurement, reused)
                    _reuse_result_class = 'REUSED_VERIFICATION'
                    if getattr(measurement, "stateframe_green_coherence_mode", "off") == "on":
                        from kernel_terminal_surface_unify import refusal_result_class
                        _reuse_result_class = (refusal_result_class(reused)
                                               or _reuse_result_class)
                    bolt.receipt('test', line, line, _reuse_result_class, detail=reused)
                    import kernel_inflight_fixes as _inflight
                    if _inflight.enabled():   # r45 in-flight fix 3: the one reuse predicate's writer
                        _inflight.note_reused_revision(measurement, _inflight.verification_key(measurement),
                                                       line[4:].strip())
                        _inflight.emit(measurement, "reused_verification", stage="applicable",
                                       selector=line[4:].strip())
                    measurement.observe(line, reused)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, reused)
                                 if tool_call_id else {'role': 'user', 'content': reused})
                    reused_cleared = bool(getattr(
                        measurement, "_post_green_reused_verification_cleared", False))
                    if reused_cleared:
                        # A trusted current-green receipt is real clearing
                        # evidence for the pending edit.  It is progress, not
                        # a no-progress loop, so the next model action must be
                        # allowed to request done.
                        try:
                            from kernel_postgreen_no_progress import note_progress
                            note_progress(measurement)
                        except (ImportError, AttributeError, TypeError, ValueError):
                            pass
                        disposition = getattr(measurement, "v2_disposition", None)
                        if disposition is not None:
                            getattr(disposition, "repeated_requests", {}).clear()
                            disposition.terminal = None
                        if getattr(measurement, "system_outcome", None) in {
                                "stalled", "post_green_stall"}:
                            measurement.system_outcome = None
                        continue
                    if getattr(measurement, "system_outcome", None) in {
                            "stalled", "post_green_stall"}:
                        if getattr(measurement, "system_outcome", None) == "post_green_stall":
                            measurement.journal.emit(
                                "post_green_stall_terminal", terminal="POST_GREEN_STALL",
                                reason="two refused unchanged-green re-checks")
                        else:
                            measurement.journal.emit("v2_terminal", terminal="STALLED",
                                                     reason="third identical settled verification")
                        if _inflight.enabled():
                            # r45 in-flight fix 3: a green, submit-ready stall submits instead of ending unsubmitted.
                            from kernel_green_then_exhaust import submit_if_ready
                            _stall_decision = submit_if_ready(measurement, boundary="post_green_stall")
                            _inflight.emit(measurement, "post_green_stall_submit",
                                           stage=("acted" if getattr(_stall_decision, "accepted", False)
                                                  else "evaluated"))
                            if getattr(_stall_decision, "accepted", False):
                                break
                        if _continue_after_heuristic_terminal(
                                "postgreen_stall" if getattr(measurement, "system_outcome", None)
                                == "post_green_stall" else "response_reuse_stall"):
                            continue
                        break
                    continue
            if line.startswith("done"):
                _jobs_stopped = (_agent_tools.stop_jobs(str(tool_call_id or f"done-{step}"))
                                 if _agent_tools is not None else "")
                if _jobs_stopped:
                    measurement.observe("job_stop", _jobs_stopped)
                from kernel_done_is_submit_authority import (
                    enabled as done_submit_enabled,
                    ownerless_early_done_refusal,
                    request_submission,
                    wait_for_pending_mutations,
                )
                if done_submit_enabled(measurement):
                    refusal = ownerless_early_done_refusal(
                        measurement,
                        available_tools=getattr(
                            measurement, "_stateframe_current_request_tool_names", (),
                        ),
                    )
                    if refusal is not None:
                        observation = str(refusal["detail"]) + ("\n" + _jobs_stopped if _jobs_stopped else "")
                        bolt.receipt(
                            "done", line, "current candidate", "NEEDS_WORK",
                            detail=observation,
                        )
                        measurement.observe(line, observation)
                        msgs.append(_tool_reply_for_call(
                            msgs, tool_call_id, observation, no_id_role="user",
                        ))
                        continue
                    request_submission(measurement, source="model")
                    pending_mutations = wait_for_pending_mutations(measurement)
                    measurement._done_submit_pending_mutation_count = len(pending_mutations)
                    bolt.receipt(
                        "done", line, "current candidate",
                        "SUBMIT_PENDING", detail="model done routed to host artifact capture",
                    )
                    observation = "Submission requested; capturing the current candidate." + ("\n" + _jobs_stopped if _jobs_stopped else "")
                    measurement.observe(line, observation)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, observation, no_id_role="user"))
                    break
                _decisive_factor_advisory = measurement.decisive_factor_done_check()
                if not _done_summary:
                    _done_summary = line[4:].strip()
                # Staged long-horizon work releases exactly one public phase at
                # a time.  This is deliberately before every completion gate:
                # `done` is a phase acceptance request while withheld phases
                # remain, not a submission request.
                if _staged_enabled and _staged_remaining:
                    _next_phase = _staged_remaining[0]
                    _release_phase = int(
                        _next_phase.get("phase", _current_staged_phase + 1))
                    _released_selectors = tuple(
                        str(selector) for selector in
                        (_next_phase.get("public_check_paths") or ())
                        if str(selector).strip())
                    _released_tests = _next_phase.get("public_tests") or ()
                    _phase_observation = ""
                    try:
                        _phase_observation, _phase_codes = run_tests_for_selectors(
                            root, fn,
                            tuple(str(selector) for selector in getattr(
                                bolt, "f2p_ids", ()) if str(selector).strip()),
                            d["rel"], row=d)
                    except Exception as _phase_error:                  # noqa: BLE001
                        _phase_observation = (
                            f"phase gate host error: {type(_phase_error).__name__}: "
                            f"{_phase_error}")
                        _phase_codes = ()
                    _phase_green = bool(_phase_codes) and all(
                        code == 0 for code in _phase_codes)
                    _phase_gate_exit = (
                        0 if _phase_green else
                        1 if any(code == 1 for code in _phase_codes) else
                        next((code for code in _phase_codes if code != 0), 5))
                    if not _phase_green:
                        _phase_refusal = (
                            f"[staged phases] PHASE {_current_staged_phase} accepted "
                            f"checks are not green; phase {_release_phase} remains withheld.\n"
                            f"Failing check receipt:\n{_phase_observation}")
                        measurement.journal.emit(
                            "phase_release_refused", phase=_release_phase, step=step,
                            gate_exit_code=_phase_gate_exit)
                        bolt.receipt(
                            "done", "staged phase gate", line, "PHASE_RELEASE_REFUSED",
                            _phase_refusal)
                        measurement.observe("done", _phase_refusal)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, _phase_refusal) if tool_call_id else
                                    {"role": "user", "content": _phase_refusal})
                        continue
                    if (not isinstance(_released_tests, Sequence)
                            or isinstance(_released_tests, (str, bytes))):
                        raise ValueError("staged phase public_tests must be a sequence")
                    _written_phase_tests = _materialize_staged_phase(root, _next_phase)
                    _append_staged_ticket(
                        d=d, msgs=msgs, baseline_messages=baseline_messages,
                        information=information, phase=_release_phase,
                        ticket=str(_next_phase.get("ticket") or ""))
                    _staged_remaining.pop(0)
                    _current_staged_phase = _release_phase
                    d["staged_phases"] = copy.deepcopy(_staged_remaining)
                    _public_tests = d.get("public_tests")
                    if not isinstance(_public_tests, list):
                        _public_tests = []
                    _public_tests.extend(copy.deepcopy(list(_released_tests)))
                    d["public_tests"] = _public_tests
                    _public_paths = d.get("public_check_paths")
                    if not isinstance(_public_paths, list):
                        _public_paths = list(_public_paths or ())
                    for _selector in _released_selectors:
                        if _selector not in _public_paths:
                            _public_paths.append(_selector)
                    d["public_check_paths"] = _public_paths
                    try:
                        fileset.public_tests.update({
                            str(item["path"]): str(item.get("source", item.get("parent_src", "")))
                            for item in _released_tests if isinstance(item, Mapping)
                        })
                    except AttributeError:
                        pass
                    _selectors = list(getattr(bolt, "f2p_ids", ()) or ())
                    _selectors_to_add = _released_selectors or _written_phase_tests
                    for _selector in _selectors_to_add:
                        if _selector not in _selectors:
                            _selectors.append(_selector)
                    bolt.f2p_ids = tuple(_selectors)
                    bolt.f2p_names = tuple({
                        str(selector).split("::")[-1] for selector in _selectors})
                    bolt.oracle_file_ids = tuple(sorted({
                        str(selector).split("::", 1)[0] for selector in _selectors}))
                    bolt.f2p_inworld = tuple(sorted(set(getattr(
                        bolt, "f2p_inworld", ()) or ()) | set(_written_phase_tests)))
                    baseline_tests.update(str(root / path) for path in _written_phase_tests)
                    _visible_row = _visible_staged_row(d)
                    if information.strict:
                        _visible_row = {
                            key: value for key, value in _visible_row.items()
                            if key not in {'fixed_src', 'test_patch', '_oracle_preflight',
                                           'FAIL_TO_PASS', 'PASS_TO_PASS', 'fail_to_pass',
                                           'pass_to_pass'}
                        }
                    bolt.task_row = _visible_row
                    if getattr(measurement, "probe_gate", None) is not None:
                        measurement.probe_gate.configure(
                            brief=information.ordinary_brief,
                            public_selectors=bolt.f2p_ids,
                            public_sources=d.get("public_tests"))
                    measurement.long_horizon_phase = _current_staged_phase
                    if getattr(measurement, "long_horizon_depth_tracker", None) is not None:
                        measurement.long_horizon_depth_tracker.rebase(msgs)
                        measurement.long_horizon_depth_tracker.add_message({
                            "role": "user",
                            "content": _staged_ticket_block(
                                _release_phase, str(_next_phase.get("ticket") or "")),
                        })
                    _brief_after = str(msgs[1].get("content") or "")
                    measurement.journal.emit(
                        "phase_released", phase=_release_phase, step=step,
                        released_selectors=list(_released_selectors or _written_phase_tests),
                        gate_exit_code=_phase_gate_exit,
                        brief_sha256_after=hashlib.sha256(
                            _brief_after.encode()).hexdigest())
                    _phase_tool_message = (
                        f"Phase {_current_staged_phase - 1} accepted. Follow-up ticket "
                        "has been appended to the task brief:\n" +
                        str(_next_phase.get("ticket") or ""))
                    bolt.receipt(
                        "done", "staged phase gate", line, "PHASE_RELEASED",
                        _phase_tool_message[:1200])
                    measurement.observe("done", _phase_tool_message)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _phase_tool_message) if tool_call_id else
                                {"role": "user", "content": _phase_tool_message})
                    continue
                # STUB GATE (R-147): host-observed leftover NotImplementedError in the target.
                _stub_ok, _stub_why = True, ""
                try:  # never let the gate crash a run; failure to read = no challenge
                    if isinstance(target, pathlib.Path) and target.is_file():
                        try:
                            _stub_rel = target.relative_to(root).as_posix()
                        except (ValueError, TypeError):
                            _stub_rel = target.name
                        _stub_ok, _stub_why = _stub_gate.evaluate(
                            _stub_rel, target.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    _stub_ok, _stub_why = True, ""
                # v2 admits a stub only through the brief-grounded, executed smoke
                # review below. Keep the legacy StubGate untouched for off/on, but
                # do not let its independent soft challenge consume v2's one refusal.
                if getattr(measurement, "disposition_v2", False):
                    _stub_ok, _stub_why = True, ""
                if not _stub_ok:
                    _stub_decision = measurement.completion_decision(
                        admissible=False, detail=_stub_why,
                        reason_codes=("STUB_GATE",), path=(d.get("rel") if isinstance(d, Mapping) else None))
                    _stub_why = _stub_decision.detail
                    bolt.receipt("done", "stub gate", line, "CHALLENGED", detail=_stub_why)
                    measurement.observe("done", _stub_why)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _stub_why) if tool_call_id else
                                {"role": "user", "content": _stub_why})
                    if _stub_decision.outcome == BLOCKED:
                        if _continue_after_heuristic_terminal("done_stub_gate"):
                            continue
                        measurement.system_outcome = BLOCKED.lower()
                        break
                    continue
                # N14 (STATEFRAME_SHIP_TREE_SMOKE): a touched module that fails a fresh
                # import is a completion obligation; the menu stays intact for the repair.
                _smoke_gate = getattr(measurement, "ship_tree_done_gate", None)
                _smoke_ok, _smoke_why = _smoke_gate() if _smoke_gate is not None else (True, "")
                _smoke_decision = None
                if not _smoke_ok:
                    _smoke_decision = measurement.completion_decision(
                        admissible=False, detail=_smoke_why,
                        reason_codes=("SHIP_TREE_SMOKE",),
                        path=(d.get("rel") if isinstance(d, Mapping) else None))
                    if _smoke_decision.accepted:
                        # r44 fix DONE (consult item 2): consume the SHARED decision.  A
                        # witnessed smoke FAIL is never waived (kernel_done_challenge_once.
                        # witnessed_failure_bound); should any policy still accept, the runner
                        # must not render a refusal beside an accepted submission -- journal
                        # it and let the ordinary done path reconcile the accepted decision.
                        measurement.journal.emit(
                            "ship_tree_smoke_decision_accepted", producer="react_repair",
                            reason_codes=list(_smoke_decision.reason_codes), dispatch=False)
                        _smoke_ok = True
                if not _smoke_ok:
                    _smoke_why = _smoke_decision.detail or _smoke_why
                    bolt.receipt("done", "ship-tree smoke", line,
                                 "STOPPED_INCOMPLETE" if "STOPPED_INCOMPLETE"
                                 in tuple(_smoke_decision.reason_codes or ()) else "CHALLENGED",
                                 detail=_smoke_why)
                    measurement.observe("done", _smoke_why)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _smoke_why) if tool_call_id else
                                {"role": "user", "content": _smoke_why})
                    if _smoke_decision.outcome == BLOCKED:
                        if _continue_after_heuristic_terminal("done_ship_tree_smoke"):
                            continue
                        measurement.system_outcome = BLOCKED.lower()
                        break
                    continue
                _probe_gate_ok, _probe_gate_why = measurement.probe_done()
                if not _probe_gate_ok:
                    _probe_decision = measurement.completion_decision(
                        admissible=False, detail=_probe_gate_why,
                        reason_codes=("PROBE_GATE",), selectors=measurement.required_public_selectors())
                    _probe_gate_why = _probe_decision.detail
                    bolt.receipt("done", "probe gate", line,
                                 "CHALLENGED", detail=_probe_gate_why)
                    measurement.observe("done", _probe_gate_why)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _probe_gate_why) if tool_call_id else
                                {"role": "user", "content": _probe_gate_why})
                    if _probe_decision.outcome == BLOCKED:
                        if _continue_after_heuristic_terminal("done_probe_gate"):
                            continue
                        measurement.system_outcome = BLOCKED.lower()
                        break
                    continue
                # Re-audit the CURRENT source at every done (bounded inside the gate); a
                # refuse-once gate accepted half-repairs (batch 121 JobQueue-924083).
                _jq_gate_kwargs = {}
                if (getattr(measurement, "jq_finish_enabled", False) and target is not None
                        and getattr(measurement, "jq_audit", None) is not None):
                    try:
                        _jq_gate_row = d if isinstance(d, Mapping) else {}
                        _jq_gate_kwargs = {
                            "source": target.read_text(encoding="utf-8"),
                            "path": target.resolve().relative_to(root.resolve()).as_posix(),
                            "brief": next((str(_jq_gate_row.get(key)) for key in (
                                "_bridge_ordinary_brief", "msg", "brief", "symptom")
                                if isinstance(_jq_gate_row.get(key), str)
                                and _jq_gate_row.get(key).strip()), ""),
                        }
                    except (OSError, UnicodeError, ValueError):
                        _jq_gate_kwargs = {}
                _jq_gate_ok, _jq_gate_why = measurement.jobqueue_audit_done_gate(**_jq_gate_kwargs)
                if not _jq_gate_ok:
                    _jq_decision = measurement.completion_decision(
                        admissible=False, detail=_jq_gate_why,
                        reason_codes=("CONTRACT_AUDIT",), path=(d.get("rel") if isinstance(d, Mapping) else None))
                    _jq_gate_why = _jq_decision.detail
                    bolt.receipt("done", "contract audit", line,
                                 "CHALLENGED", detail=_jq_gate_why)
                    measurement.observe("done", _jq_gate_why)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _jq_gate_why) if tool_call_id else
                                {"role": "user", "content": _jq_gate_why})
                    if _jq_decision.outcome == BLOCKED:
                        if _continue_after_heuristic_terminal("done_contract_audit"):
                            continue
                        measurement.system_outcome = BLOCKED.lower()
                        break
                    continue
                if getattr(measurement, "stateframe_turtle_mode", "off") == "on":
                    _turtle_trigger = measurement.turtle_trigger(event="done")
                    if _turtle_trigger is not None:
                        _turtle_result = measurement.turtle_review(
                            trigger=_turtle_trigger, endpoint=API, key=KEY)
                        if _turtle_result.get("verdict") == "DEFECT":
                            _turtle_message = measurement.turtle_message(_turtle_result)
                            _turtle_decision = measurement.completion_decision(
                                admissible=False, detail=_turtle_message,
                                reason_codes=("TURTLE_DEFECT",),
                                selectors=measurement.required_public_selectors())
                            _turtle_message = _turtle_decision.detail
                            bolt.receipt("done", "turtle grounded defect", line,
                                         "TURTLE_ADVISORY", detail=_turtle_message)
                            measurement.observe("done", _turtle_message)
                            msgs.append(_tool_reply_for_call(msgs, tool_call_id, _turtle_message) if tool_call_id else
                                        {"role": "user", "content": _turtle_message})
                            if _turtle_decision.outcome == BLOCKED:
                                if _continue_after_heuristic_terminal("done_turtle_defect"):
                                    continue
                                measurement.system_outcome = BLOCKED.lower()
                                break
                            continue
                _public_review = measurement.public_review_completion(
                    d, _remaining_budget_units(), API, KEY)
                if _public_review is not None:
                    # Fix (post-review r44, finding 1): PROBE_INCONCLUSIVE is
                    # a diagnostic card only; the internal design note is not evidence.
                    # validity-20260923.md" requires it never become
                    # completion evidence, but it must NOT itself refuse an
                    # otherwise-admissible `done` either. Surface it and fall
                    # through to the ordinary completion checks below,
                    # instead of forcing a PUBLIC_REVIEW refusal + another
                    # turn the way a real (PROBE_RED) review still does.
                    _review_probe_result = getattr(
                        getattr(measurement, 'public_review', None),
                        'last_exposed_result', None)
                    if _review_probe_result == 'PROBE_INCONCLUSIVE':
                        bolt.receipt('done', 'public review diagnostic (non-blocking)', line,
                                     'REVIEW_DIAGNOSTIC_DELIVERED', detail=_public_review)
                        measurement.observe('done', _public_review)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, _public_review))
                        _public_review = None
                if _public_review is not None:
                    _review_decision = measurement.completion_decision(
                        admissible=False, detail=_public_review,
                        reason_codes=("PUBLIC_REVIEW",),
                        selectors=measurement.required_public_selectors())
                    _public_review = _review_decision.detail
                    bolt.receipt('done', 'public review stop proposal', line,
                                 'REVIEW_ADVISORY_DELIVERED', detail=_public_review)
                    measurement.observe('done', _public_review)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _public_review))
                    if _review_decision.outcome == BLOCKED:
                        if _continue_after_heuristic_terminal("done_public_review"):
                            continue
                        measurement.system_outcome = BLOCKED.lower()
                        break
                    continue
                _sidekick_challenge = measurement.completion_challenge(
                    _remaining_budget_units())
                # CYLINDERS 3 + 5: is stopping legal? Refuse `done` with zero edits (it would
                # submit the original bug) and challenge `done` straight after a red test.
                # The kernel is not choosing the fix — it is refusing an exit the evidence
                # does not support, and it always says why.
                _done_progress = _progress_facts()
                bolt.edits = _done_progress.edits
                if _host_truth_progress_enabled:
                    bolt.product_edits = _done_progress.product_edits
                # E333: give the gate the model's own claim so a concrete diagnosis
                # with zero edits gets quoted back as an action demand.
                bolt.last_done_summary = line[4:].strip() or \
                    (locals().get("_done_summary") or "")
                if action_schema_mode:
                    # Keep typed completion data beside the legacy display summary. The
                    # completion lane consumes these fields; summary remains presentation
                    # text and is not the source of the action-envelope claim.
                    bolt.last_done_claim = _done_claim
                    bolt.last_done_evidence_refs = list(_done_evidence_refs or [])
                _completion_reason_codes = []
                ok_finish, why = bolt.may_finish()
                _repro_gate = getattr(measurement, "repro_first_gate", None)
                if (__import__("kernel_repro_first").repro_first_gate_mode() == "on"
                        and _repro_gate is not None):
                    from kernel_repro_first_coding import file_unreproduced
                    if _repro_gate.receipt is None:
                        file_unreproduced(
                            _repro_gate, _done_summary or line[4:].strip())
                    from kernel_repro_first_runtime import complete as complete_repro_first
                    _repro_decision = complete_repro_first(
                        measurement,
                        root=root,
                        public_suite_green=(
                            _done_progress.checks > 0
                            and _done_progress.latest_check_passed
                            and _done_progress.edits_since_check == 0
                            if _host_truth_progress_enabled
                            else bolt.last_test_exit == 0 and edits_since_test == 0
                        ),
                    )
                    if not _repro_decision.accepted:
                        ok_finish = False
                        why = "[repro-first] " + _repro_decision.detail
                        _completion_reason_codes.append("REPRO_FIRST_GATE")
                if not ok_finish:
                    _completion_reason_codes.append("BOLT_GATE")
                if measurement.continuation is not None:
                    _completion_finding = None
                    _finding_method = getattr(measurement.continuation, "completion_finding", None)
                    if callable(_finding_method):
                        _completion_finding = _finding_method()
                    if isinstance(_completion_finding, Mapping):
                        _continuation_blocker = str(_completion_finding.get("detail") or "")
                        if _continuation_blocker:
                            ok_finish, why = False, _continuation_blocker
                            _completion_reason_codes.append(
                                str(_completion_finding.get("code") or "CONTINUATION_BLOCKER"))
                    else:
                        _continuation_blocker = measurement.continuation.completion_blocker()
                        if _continuation_blocker:
                            ok_finish, why = False, _continuation_blocker
                            _completion_reason_codes.append("CONTINUATION_BLOCKER")
                _work_blocker = measurement.declared_work_blocker()
                if _work_blocker:
                    ok_finish, why = False, _work_blocker
                    _completion_reason_codes.append("DECLARED_WORK")
                if _sidekick_challenge:
                    ok_finish, why = False, _sidekick_challenge
                    _completion_reason_codes.append("SIDEKICK_CHALLENGE")
                # The question engine contributes a typed done-time clause
                # fact to the same reducer call below. It never returns or
                # vetoes independently of CompletionReducer.
                _question_assessment = measurement.question_done_assessment(
                    claim=_done_claim if action_schema_mode else None,
                    evidence_refs=_done_evidence_refs if action_schema_mode else (),
                    summary=_done_summary)
                _unresolved_questions = ()
                if _question_assessment is not None:
                    _unresolved_questions = tuple(
                        str(item) for item in _question_assessment.get(
                            "unresolved_questions", ()) if str(item).strip())
                # E448: a done claim is measured against the ORACLE, not the model's
                # narrowed selector (stamp3: fixed one bug, dismissed the second
                # failing oracle test as "pre-existing" and got ACCEPTED). When the
                # oracle tests exist in-world, run them; red REFUTES the claim with
                # the actual failing tail. Held-out oracles skip (can't run).
                if (interventions and ok_finish and information.feedback != "FINAL_ONLY"
                        and getattr(bolt, "f2p_ids", ()) and measurement.continuation is None):
                    try:
                        if multifile:
                            _available = [i for i in bolt.f2p_ids
                                          if (root / i.split("::")[0]).exists()]
                            _oobs, _ocodes = run_tests_for_selectors(
                                root, fn, _available, d["rel"], row=d)
                            _oracle_green = bool(_ocodes) and all(
                                code == 0 for code in _ocodes)
                            _oracle_targets_present = bool(_available)
                        else:
                            _osel = " ".join(
                                i for i in bolt.f2p_ids
                                if (root / i.split("::")[0]).exists())
                            _oobs = (run_tests(root, fn, _osel, d["rel"], row=d)
                                     if _osel else "")
                            _oracle_green = "pytest exit 0" in _oobs
                            _oracle_targets_present = bool(_osel)
                        if _oracle_targets_present:
                            if not _oracle_green:
                                ok_finish = False
                                _completion_reason_codes.append("ORACLE_TESTS_RED")
                                from kernel_wire_identity import public_wire_text
                                why = ("[kernel-gate] DONE REFUSED: the ORACLE tests "
                                       "still fail — every one of them must pass, "
                                       "'pre-existing' is not an accepted excuse. "
                                       "Failing tail:\n" + public_wire_text(
                                           _oobs[-900:], measurement.journal))
                                why = information.done_feedback(why)
                    except Exception:                                # noqa: BLE001
                        ok_finish = False
                        _completion_reason_codes.append("ORACLE_UNAVAILABLE")
                        why = information.done_feedback("[grader] UNAVAILABLE: completion unverified.")
                # E492: completeness governor (feast #7) — an ACCEPTED done is also
                # checked against the obligation graph (every oracle id / unit gate):
                # UNKNOWN or MISSING obligations refuse with the exact open list, so
                # "pre-existing"/"unrelated" claims meet a named blocker, not a vibe.
                if interventions and ok_finish and information.feedback != "FINAL_ONLY":
                    # E513: only IN-WORLD oracle ids are obligations the agent can
                    # observe; held-out f2p tests (absent by design in repair lanes)
                    # are graded by the E448 rerun + final grading, never by this gate.
                    _inworld = [i for i in (getattr(bolt, "f2p_ids", ()) or ())
                                if (root / str(i).split("::")[0]).exists()]
                    _obligations_ok, _obligation_why = inworld_obligation_gate(
                        d,
                        _inworld,
                        getattr(bolt, "_receipts", ()),
                        unit_plan=getattr(bolt, "unit_plan", None),
                    )
                    if not _obligations_ok:
                        ok_finish = False
                        _completion_reason_codes.append("INWORLD_OBLIGATION")
                        why = information.done_feedback(_obligation_why)
                _v2_terminal = False
                _v2_decision = measurement.v2_done_decision(
                    legacy_ok=ok_finish, legacy_detail=why)
                if _v2_decision is not None:
                    if _v2_decision.action == "refuse":
                        ok_finish = False
                        _completion_reason_codes.append("V2_DISPOSITION")
                        why = _v2_decision.detail
                    elif _v2_decision.action == "terminal":
                        ok_finish = False
                        _completion_reason_codes.append("V2_TERMINAL")
                        _v2_terminal = True
                        why = ("[disposition-v2] INCOMPLETE: the same unresolved witnessed "
                               "blocker remained after one recoverable refusal; this run "
                               "is not submitted.")
                # Atomic completion boundary. Every preceding gate contributes
                # to this one immutable assessment; no component is allowed to
                # veto after the reducer seals it. Continuation commit is a
                # reducer callback, so a commit failure is a typed BLOCKED
                # outcome rather than a late second refusal.
                _empirical_repair_action = None
                if (os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on"
                        and not ok_finish):
                    from kernel_empirical_fixes import enrich_refusal
                    _refusal = enrich_refusal(
                        why, getattr(bolt, "_receipts", ()),
                        selectors=measurement.required_public_selectors(),
                        path=(d.get("rel") if isinstance(d, Mapping) else None),
                        menu=measurement.completion_menu,
                    )
                    why = _refusal["detail"]
                    _empirical_repair_action = _refusal.get("repair_action")
                _completion_verdict = None
                _completion_commit = None
                if ok_finish and measurement.continuation is not None:
                    try:
                        _completion_verdict = bolt.completion_verdict()
                        if getattr(measurement, "stateframe_submit_idempotent_mode",
                                   "off") == "on":
                            # Exactly-once: the continuation journal is the keyed
                            # destination (kernel_submission_idempotency).
                            _completion_commit = measurement.continuation.keyed_destination(
                                _completion_verdict)
                        else:
                            def _commit_completion():
                                return measurement.continuation.accept_done(_completion_verdict)
                            _completion_commit = _commit_completion
                    except Exception as _completion_error:             # noqa: BLE001
                        ok_finish = False
                        why = ("[continuation] completion certification could not be "
                               f"prepared: {type(_completion_error).__name__}: "
                               f"{_completion_error}")
                _completion_kwargs = {
                    "admissible": ok_finish,
                    "detail": why if not ok_finish else "",
                    "reason_codes": tuple(dict.fromkeys(_completion_reason_codes))
                    if _completion_reason_codes else
                    (("EXTERNAL_DONE_GATE",) if not ok_finish else ()),
                    "selectors": measurement.required_public_selectors(),
                    "path": (d.get("rel") if isinstance(d, Mapping) else None),
                    "repair_action": _empirical_repair_action,
                    "unresolved_questions": _unresolved_questions,
                    "claims": {"summary": _done_summary, "claim": _done_claim,
                               "evidence_refs": list(_done_evidence_refs or [])},
                    "no_change_paths": _done_no_change_paths,
                    "commit": _completion_commit,
                }

                def _dispatch_completion():
                    # This is the real acceptance operation.  When the boundary
                    # is enabled it is invoked only by ENFORCE, so continuation
                    # commit and reducer sealing cannot happen before admission.
                    return measurement.completion_decision(**_completion_kwargs)

                _submit_shadow = False
                if not _effect_boundary_enabled():
                    # Preserve the legacy OFF path exactly: no snapshot, import,
                    # receipt, render, or altered completion decision.
                    _completion_decision = _dispatch_completion()
                else:
                    _submit_snapshot = {
                        "task_id": str(d.get("task_id") or fn),
                        "step": step,
                        "admissible": ok_finish,
                        "reason_codes": _completion_kwargs["reason_codes"],
                        "has_commit": _completion_commit is not None,
                    }

                    def _completion_state():
                        return {
                            "sealed": bool(getattr(
                                getattr(measurement, "completion_reducer", None),
                                "_sealed", False)),
                            "completion_accepted": bool(
                                done_accepted(measurement)),
                            "continuation_completed": bool(getattr(
                                getattr(measurement, "continuation", None),
                                "completed", False)),
                        }

                    _submit_applied, _submit_value, _submit_rendered = _effect_application(
                        "submit",
                        _submit_snapshot,
                        _dispatch_completion,
                        owner=measurement,
                        before_snapshot=_completion_state,
                        effect_applied=lambda decision: bool(
                            getattr(decision, "accepted", False)),
                        after_snapshot=_completion_state,
                    )
                    if _submit_value is None:
                        _submit_shadow = _effect_boundary_mode() == "shadow"
                        _completion_decision = _effect_boundary_shadow_completion(
                            measurement, _submit_rendered)
                    else:
                        _completion_decision = _submit_value
                ok_finish = _completion_decision.accepted
                if not ok_finish:
                    why = _completion_decision.detail
                if _v2_terminal:
                    # r44 review F2: the terminal is read off the SHARED stop decision
                    # (V2_TERMINAL is protected from the challenge arbiter), so an accepted
                    # decision can never be labelled INCOMPLETE or vice versa.
                    _v2_terminal = not bool(_completion_decision.accepted)
                if _v2_terminal:
                    if not _continue_after_heuristic_terminal("v2_terminal"):
                        measurement.system_outcome = "incomplete"
                elif _completion_decision.outcome == BLOCKED:
                    if not _continue_after_heuristic_terminal("blocked"):
                        measurement.system_outcome = BLOCKED.lower()
                # E334: done receipt — stop-control vs certification stays separable.
                _done_receipt_detail = (why[:150] if not ok_finish and why else
                                        f"edits={_done_progress.edits} last_exit={bolt.last_test_exit}")
                if ok_finish and _unresolved_questions:
                    from kernel_questions import render_unresolved_questions
                    _done_receipt_detail += "\n" + render_unresolved_questions(
                        _unresolved_questions)
                if _decisive_factor_advisory:
                    _done_receipt_detail += "\n" + _decisive_factor_advisory[:360]
                if _jobs_stopped:
                    _done_receipt_detail += "\n" + _jobs_stopped
                bolt.receipt("done", bolt.last_done_summary[:100] or "done",
                             bolt.last_done_summary,
                             "ACCEPTED" if ok_finish else "CHALLENGED",
                             _done_receipt_detail,
                             why=why if not ok_finish else None)
                if action_schema_mode and getattr(bolt, "_receipts", None):
                    _typed_done_receipt = bolt._receipts[-1]
                    _typed_done_receipt["claim"] = _done_claim
                    _typed_done_receipt["evidence_refs"] = list(_done_evidence_refs or [])
                    measurement.amend_receipt(_typed_done_receipt)
                if multifile:
                    _done_paths = _product_source_paths(root, fileset, d)
                    _done_snapshot = _source_snapshot(root, _done_paths)
                    _annotate_receipt(
                        bolt, paths=_done_paths, scope=_done_paths,
                        before=_done_snapshot, after=_done_snapshot,
                        selectors=tuple(getattr(bolt, "f2p_ids", ())),
                        failure_tail=why if not ok_finish else "")
                if _jobs_stopped:
                    why = (why + "\n" if why else "") + _jobs_stopped
                if not ok_finish:
                    measurement.observe("done", why)
                    if (getattr(measurement, "portfolio", None) is not None
                            and bolt.last_test_exit == 0
                            and isinstance(getattr(measurement, "host_closure_payload", None), Mapping)
                            and str(measurement.host_closure_payload.get("mode", "")).upper() == "REPAIR"):
                        def _portfolio_model_request(_context):
                            from kernel_request_kind import reject_unprepared_branch
                            reject_unprepared_branch(measurement)
                            _branch_messages = copy.deepcopy(
                                _context.request.get("messages") or msgs)
                            _branch_messages.append({
                                "role": "user",
                                "content": ("[portfolio] Independent repair branch; use the "
                                             "current host witness and make one fresh repair "
                                             "attempt in your isolated workspace."),
                            })
                            _content, _calls = ask_tools(
                                _branch_messages, temp=_context.temperature,
                                tools=_context.request.get("tools") or _request_tools)
                            return {"content": _content, "tool_calls": _calls}

                        measurement.portfolio_fork(
                            refusal={"verified_progress": True,
                                     "host_payload": measurement.host_closure_payload},
                            request={"messages": copy.deepcopy(msgs),
                                     "tools": copy.deepcopy(_request_tools or [])},
                            model_request=_portfolio_model_request)
                    if (getattr(measurement, "fork_a", None) is not None
                            and bolt.last_test_exit == 0
                            and isinstance(getattr(measurement, "host_closure_payload", None), Mapping)
                            and str(measurement.host_closure_payload.get("mode", "")).upper() == "REPAIR"):
                        def _fork_a_model_request(_worker):
                            from kernel_request_kind import reject_unprepared_branch
                            reject_unprepared_branch(measurement)
                            _branch_messages = copy.deepcopy(
                                _worker.request.get("messages") or msgs)
                            _branch_messages.append({
                                "role": "user",
                                "content": ("[fork-a] Checkpoint-derived worker; use the "
                                             "current host witness and make one isolated repair "
                                             "attempt. Results are admitted by the parent only."),
                            })
                            _content, _calls = ask_tools(
                                _branch_messages,
                                temp=(0.3 if _worker.worker_index == 0 else 1.0),
                                tools=_worker.request.get("tools") or _request_tools)
                            return {"content": _content, "tool_calls": _calls}

                        _fork_a_workers = measurement.fork_a_run(
                            refusal={"verified_progress": True,
                                     "host_payload": measurement.host_closure_payload},
                            request={"messages": copy.deepcopy(msgs),
                                     "tools": copy.deepcopy(_request_tools or [])},
                            model_request=_fork_a_model_request)
                        _fork_a_admit_completed_candidate(
                            measurement, _fork_a_workers, fn=fn,
                            selectors=tuple(getattr(bolt, "f2p_ids", ())),
                            rel_hint=d.get("rel"), row=d)
                    # The opt-in Sidekick schema owns this shared challenge bucket. Once
                    # its run-wide cap is spent, do not let the legacy generic completion
                    # refusal charge the same category again on a later done.
                    if (not _sidekick_challenge and not (
                            getattr(measurement, "stateframe_challenge_schema_mode", "off") == "on"
                            and measurement.parent.used["challenges"] >=
                            measurement.parent.limits["challenges"])):
                        measurement.parent.charge("challenges", reason="existing kernel completion challenge")
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, why) if tool_call_id
                                else {"role": "user", "content": why})
                    with (POC / "results" / f"react_live{os.environ.get('REACT_TAG', '')}.log").open("a") as live:
                        live.write(f"[{fn[:26]:28s} s{step + 1:02d}] "
                                   f"{'done REFUSED by kernel':72s} -> {why[:88]}\n")
                    if _submit_shadow and _effect_boundary_done_only(measurement):
                        if _continue_after_heuristic_terminal("effect_boundary_shadow"):
                            continue
                        measurement.system_outcome = "effect_boundary_shadow"
                        break
                    if _v2_terminal:
                        if _continue_after_heuristic_terminal("v2_terminal"):
                            continue
                        break
                    continue
                if getattr(measurement, "portfolio", None) is not None:
                    measurement.portfolio_preserve_candidate(
                        public_green=bolt.last_test_exit == 0,
                        public_suite=tuple(getattr(measurement, "test_receipts", ())),
                        reason=f"accepted-done:step-{step}")
                    measurement.portfolio_select()
                # CompletionVerdict remains a receipt/certification projection;
                # the reducer above has already committed any continuation
                # transition before emitting ACCEPTED.
                if measurement.continuation is None:
                    try:
                        bolt.completion_verdict()
                    except Exception as _e:                              # noqa: BLE001
                        print(f"    [kernel-wiring] completion_verdict failed: {_e}")
                break
            elif line == "list_files" or line.startswith("list_files "):
                import shlex
                try:
                    argv = shlex.split(line)
                except ValueError:
                    argv = line.split()
                scope = argv[1] if len(argv) > 1 else None
                depth = argv[2] if len(argv) > 2 else 2
                limit = argv[3] if len(argv) > 3 else 80
                resolved_scope = None
                try:
                    from kernel_nav_basics import render_file_tree, resolve_scope
                    resolved_scope = resolve_scope(root, scope, journal=measurement.journal)
                    if not information.check_path(root, str(resolved_scope)):
                        raise ValueError("[information-policy] Path outside workspace denied.")
                    obs = render_file_tree(root, resolved_scope, depth=depth, limit=limit)
                    outcome = "HOST_OBSERVED"
                except (OSError, TypeError, ValueError) as error:
                    obs = f"list_files REJECTED: {error}"
                    outcome = "REJECTED"
                bolt.receipt("list_files", line, str(scope or "."), outcome,
                             detail=obs[:240], target_path=receipt_target_path(root, resolved_scope)
                             if resolved_scope is not None and outcome == "HOST_OBSERVED" else None)
            elif line == "file_list" or line.startswith("file_list "):
                import shlex
                try:
                    argv = shlex.split(line)
                except ValueError:
                    argv = line.split()
                scope = argv[1] if len(argv) > 1 else None
                limit = 200
                if len(argv) > 2 and argv[2].isdigit():
                    limit = max(1, min(1000, int(argv[2])))
                if scope:
                    try:
                        resolved_scope = resolve_tool_path(root, scope)
                    except AmbiguousToolPath as error:
                        obs = str(error)
                        bolt.receipt("file_list", line, line, "REJECTED", detail=obs)
                        measurement.observe(line, obs)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else
                                    {"role": "user", "content": obs})
                        continue
                    if not information.check_path(root, str(resolved_scope)):
                        obs = "[information-policy] Path outside workspace denied."
                        bolt.receipt("file_list", line, line, "REJECTED", detail=obs)
                        measurement.observe(line, obs)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else
                                    {"role": "user", "content": obs})
                        continue
                    scope = str(resolved_scope)
                from kernel_where_used import render_file_list
                obs = render_file_list(root, scope, limit=limit)
                bolt.receipt("file_list", line, line, "HOST_OBSERVED", detail=f"n={len(obs.splitlines())}")
            elif line.startswith("grep "):
                pat = line[5:].strip()
                # STRIP SHELL QUOTING THE AGENT REASONABLY WROTE.
                #
                # Unwrap quoted search patterns before literal matching to avoid
                # incorrectly reporting valid symbols as absent.
                #
                # Also drop leading flags: a human-style `grep -n "x" path/` is a perfectly
                # sensible thing for a coding agent to type, and answering it with silence
                # teaches it that the symbol is absent.
                # PARSE `grep [flags] <pattern> [path]` THE WAY A SHELL WOULD.
                #
                # Two measured failures, both ours, both looking like the model being wrong:
                #   grep "def blocking_run_status"          -> searched WITH the quote chars
                #   grep ensure_born_minimal app/x/y.py     -> used "pattern path" as ONE regex
                # The second is nastier: a bare `grep X` found the symbol while `grep X path`
                # returned nothing, which reads exactly like the file not containing it.
                # Honour the path instead of ignoring it — scoping a search is a correct thing
                # for an agent to do and it should be rewarded, not silently broken.
                import shlex
                try:
                    argv = shlex.split(pat)                 # handles quoting properly
                except ValueError:
                    argv = pat.split()
                # DROP FLAGS *AND THEIR VALUES*. `-A 20 -B 5` is two flags with numeric
                # arguments; stripping only the dashed tokens leaves "20" as the first
                # non-flag word and it becomes the search PATTERN. Measured live: the agent
                # ran `grep -A 20 -B 5 "gate.*approve" tests/x.py` and got conftest.py hits,
                # because we searched for "20". A context-flag is a perfectly normal thing for
                # a coding agent to type.
                _valued = {"-A", "-B", "-C", "-m", "-e", "--include", "--exclude"}
                cleaned, skip_next = [], False
                _end_of_flags = False
                for a in argv:
                    if skip_next:
                        skip_next = False
                        continue
                    if _end_of_flags:
                        cleaned.append(a)
                        continue
                    if a == "--" and os.environ.get("STATEFRAME_TOKENLAB_GREP_QUOTE", "off").strip().lower() == "on":
                        _end_of_flags = True
                        continue
                    if a in _valued:
                        skip_next = True
                        continue
                    if a.startswith("-"):
                        continue
                    cleaned.append(a)
                argv = cleaned
                # Structured native calls must keep pattern/path as their exact
                # JSON values; reparsing them through the legacy shell dialect
                # would split a literal containing spaces.
                if (_name == "grep" and isinstance(_args, Mapping)
                        and "pattern" in _args):
                    pat = str(_args.get("pattern") or "")
                    argv = [pat]
                    if _args.get("path"):
                        argv.append(str(_args["path"]))
                else:
                    pat = argv[0] if argv else pat
                scope = str(root)
                if len(argv) > 1:
                    try:
                        scope = str(resolve_tool_path(root, argv[1]))
                    except AmbiguousToolPath as error:
                        obs = str(error)
                        bolt.receipt("grep", line, line, "REJECTED", detail=obs)
                        measurement.observe(line, obs)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs)
                                    if tool_call_id else {"role": "user", "content": obs})
                        continue
                if not information.check_path(root, scope):
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, "[information-policy] Path outside workspace denied.")
                                if tool_call_id else {"role": "user", "content":
                                "[information-policy] Path outside workspace denied."})
                    continue
                _grep_args = _args if isinstance(_args, Mapping) else {}
                _grep_error = None
                _grep_page = None
                try:
                    from kernel_model_policy import resolve_model_policy
                    from kernel_navigation_pages import search_repository
                    _grep_page = search_repository(
                        root, pat, scope,
                        mode=_grep_args.get("mode"),
                        file_glob=(_grep_args.get("file_glob") or _grep_args.get("glob") or None),
                        cursor=_grep_args.get("cursor"),
                        requested_page_size=_grep_args.get("page_size"),
                        policy=resolve_model_policy(MODEL),
                        journal=measurement.journal,
                    )
                except (OSError, TypeError, ValueError) as error:
                    _grep_error = str(error)
                _grep_metadata = (dict(_grep_page.metadata) if _grep_page is not None else {})
                _grep_total = int(_grep_metadata.get("total", 0))
                _grep_shown = int(_grep_metadata.get("shown", 0))
                _grep_complete = bool(_grep_metadata.get("scan_complete", True))
                bolt.last_grep_hits = None if _grep_error is not None else _grep_total
                if _grep_error is None:
                    bolt.sure_observe({"kind": "grep", "query": pat[:80],
                                       "hit_count": _grep_total,
                                       "shown": _grep_shown})
                # E346: receipts carry the same complete-page metadata the model receives.
                bolt.receipt(
                    "grep", pat[:80], pat,
                    "ERROR" if _grep_error is not None else
                    "PARTIAL" if not _grep_complete else ("HITS" if _grep_total else "NO_HITS"),
                    json.dumps(_grep_metadata, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"))[:150]
                    if _grep_error is None else _grep_error,
                    result_set_digest=_grep_metadata.get("result_set_digest"))
                if multifile:
                    _grep_snapshot = _source_snapshot(root, fileset.paths)
                    _annotate_receipt(
                        bolt, paths=fileset.paths, scope=(scope,),
                        before=_grep_snapshot, after=_grep_snapshot,
                        selectors=tuple(getattr(bolt, "f2p_ids", ())),
                        failure_tail="")
                if _grep_error is not None:
                    obs = f"Search failed: {_grep_error}. Correct the query or scope and retry."
                elif _grep_page is not None:
                    _grep_symbol_fact = None
                    if _grep_total == 0:
                        # The symbol registry remains a useful Python-only aid, but it no
                        # longer defines repository search completeness or suppresses pages.
                        try:
                            reg = _registry_for(root)
                            token = re.sub(r"^(def|class|async def)\s+", "", pat).strip()
                            token = re.split(r"[^\w]", token)[0] if token else ""
                            if token and interventions:
                                _grep_symbol_fact = (token, reg.answer(token))
                        except Exception:                            # noqa: BLE001
                            pass
                    from kernel_owner_first import coordinates_enabled
                    if coordinates_enabled():
                        from kernel_navigation_pages import render_grep_result
                        obs = render_grep_result(
                            _grep_page,
                            symbol=_grep_symbol_fact[0] if _grep_symbol_fact else None,
                            symbol_result=_grep_symbol_fact[1] if _grep_symbol_fact else None,
                            workspace_root=root,
                            journal=measurement.journal,
                        )
                    else:
                        obs = _grep_page.text
                        if _grep_symbol_fact:
                            obs += "\n(no grep matches)\n" + _grep_symbol_fact[1]
                    # Q6: the second identical zero grep at one revision changes the route.
                    from kernel_repeated_zero import observe as _rz_observe
                    obs = _rz_observe(
                        measurement, "grep",
                        _grep_args if _grep_args else {"pattern": pat, "path": argv[1] if len(argv) > 1 else ""},
                        obs, zero=(_grep_total == 0 and _grep_complete))
                else:
                    obs = "Search failed: no page was produced. Retry the query or scope."
            elif line.startswith("read "):
                _native_read = (_name == "read" and isinstance(_args, Mapping)
                                and "path" in _args)
                if _native_read:
                    raw = str(_args.get("path") or "")
                    _requested_start = _args.get("line", _args.get("start_line", ""))
                    start = max(0, int(_requested_start) - 1) \
                        if str(_requested_start).isdigit() else 0
                    parts = [raw]
                else:
                    parts = line[5:].split()
                    raw = parts[0].strip("`'\"")  # markdown backticks / quotes
                    start = max(0, int(parts[1]) - 1) if len(parts) > 1 and parts[1].isdigit() else 0
                # ACCEPT `path:line`, WHICH IS THE FORMAT OUR OWN GREP EMITS.
                # grep returns "app/foo.py:81: <code>", so an agent naturally copies
                # "app/foo.py:81" straight into read. Refusing the shape we just handed it is
                # a self-inflicted trap — measured in the tool audit as one of two remaining
                # read failures. Use the line number as the start position, which is what the
                # agent meant.
                if ":" in raw and (multifile or not pathlib.Path(raw).exists()):
                    head, _, tail = raw.rpartition(":")
                    if tail.isdigit() and head:
                        raw, start = head, max(0, int(tail) - 1)
                parts = [raw] + parts[1:]
                if multifile:
                    try:
                        p, resolved_path = _multi_path(root, raw)
                    except ValueError:
                        p, resolved_path = root / "__invalid_path__", raw
                else:
                    resolved_path = raw
                    try:
                        p = resolve_tool_path(root, raw)
                    except AmbiguousToolPath as error:
                        obs = str(error)
                        bolt.receipt("read", line, line, "REJECTED", detail=obs)
                        measurement.observe(line, obs)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs)
                                    if tool_call_id else {"role": "user", "content": obs})
                        continue
                if not information.check_path(root, str(p)):
                    _denied = "[information-policy] Path outside workspace denied."
                    bolt.receipt("read", f"{raw}@{start + 1}", f"{raw}|{start}",
                                 "DENIED", detail=_denied[:150],
                                 target_path=str(p))
                    try:
                        measurement.observe(line, _denied)
                    except (AttributeError, TypeError):
                        pass
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, _denied)
                                if tool_call_id else {"role": "user", "content":
                                _denied})
                    continue
                _read_before = (_source_snapshot(root, fileset.paths)
                                if multifile else {})
                # The executor owns file caching, exact-range reuse, and its
                # revision-bound receipt. Keep one read dispatch path so cached
                # and fresh reads close the same native continuation effect.
                from kernel_read_exec import dispatch_live_read
                from kernel_model_policy import resolve_model_policy
                (_read_result, obs, _read_bytes,
                 _read_delivery) = dispatch_live_read(
                    root, p, raw, parts[0], start, args=_args,
                    measurement=measurement, step=step, bolt=bolt,
                    policy=resolve_model_policy(MODEL),
                    resolved_path=resolved_path, visible_messages=msgs,
                    page_line_limit=_lean_read_page_line_limit,
                    page_byte_limit=_lean_read_page_byte_limit)
                if _read_result == "OK":
                    try:
                        _pointer_read_path = p.resolve().relative_to(root.resolve()).as_posix()
                        _pointer_read_delivery = _read_delivery
                    except (OSError, ValueError):
                        _pointer_read_path = None
                        _pointer_read_delivery = None
                if _read_result == "MISSING":
                    # E451b: the TARGET file going missing is an emergency, not a
                    # shrug — jq3 destroyed its own jobq.py and wandered 60+ turns.
                    if interventions and not multifile and (raw.endswith(d["rel"]) or d["rel"].endswith(raw)):
                        if _workspace_truth.enabled():
                            fact = _workspace_truth.exists(
                                root, d["rel"], journal=getattr(measurement, "journal", None),
                            )
                            obs += "\n[kernel] " + _workspace_truth.render_fact(fact)
                            if fact.status == "missing":
                                obs += (" If complete authorized content is available, use "
                                        "`file_create` now; otherwise report the capability "
                                        "blocker. Do not use `edit` to recreate a missing file.")
                            elif fact.status == "exists":
                                obs += " The read returned MISSING; retry the read before editing."
                            else:
                                obs += " The read returned MISSING, but current status is unknown; retry the read."
                        else:
                            obs += _workspace_truth.render_legacy_target_gone_notice(d["rel"])
                if multifile:
                    _annotate_receipt(
                        bolt, paths=fileset.paths, scope=(resolved_path,),
                        before=_read_before, after=_read_before,
                        selectors=tuple(getattr(bolt, "f2p_ids", ())),
                        failure_tail="")
                # E407: every successful source read feeds the world cache (test lists,
                # signatures survive trims/compaction — deterministic extraction only).
                if (getattr(bolt, "world", None) is not None and _read_result == "OK"
                        and _read_bytes is not None and p.suffix == ".py"):
                    try:
                        bolt.world.observe_source(
                            path=str(p.relative_to(root)), content=_read_bytes.decode("utf-8", errors="ignore"),
                            revision=max(getattr(bolt, "workspace_revision", 0), bolt.world.current_revision),
                            evidence_ref=f"read@{step}")
                    except Exception as error:                       # noqa: BLE001
                        measurement.journal.emit('world_observation_error', operation='read',
                                                 error_type=type(error).__name__)
                if multifile:
                    _missing_at_read = _missing_declared_paths(
                        root, fileset, journal=measurement.journal,
                    )
                    if interventions and _missing_at_read:
                        obs += "\n[kernel] " + _target_gone_lines(
                            _missing_at_read, root=root, journal=measurement.journal,
                        )
                # dispatch_live_read applies repeat-read notes to this tool result.
            elif line == "str_replace" and tool_batch_args is not None:
                from kernel_coordinated_change_ledger import (
                    APPLY_EVENT, APPLY_SCHEMA, BatchFailure, BatchResult,
                    atomic_apply, tree_revision,
                )
                _batch_guard_failure = None
                _batch_edits = tool_batch_args.get("edits", [])
                if isinstance(_batch_edits, list):
                    for _batch_index, _batch_edit in enumerate(_batch_edits):
                        if not isinstance(_batch_edit, Mapping):
                            continue
                        _batch_raw_path = str(_batch_edit.get("path") or "")
                        try:
                            _batch_path = resolve_tool_path(root, _batch_raw_path)
                            if protected_test_path(_batch_path, baseline_tests):
                                _batch_reason = "protected public test paths are immutable"
                            else:
                                _batch_reason = _bound_edit_refusal(
                                    root, _batch_path, _batch_edit.get("target_ref"))
                        except (AmbiguousToolPath, OSError, ValueError) as _batch_error:
                            _batch_reason = str(_batch_error)
                        if _batch_reason:
                            _batch_guard_failure = BatchFailure(
                                _batch_index, _batch_raw_path, "BOUND_REFUSAL",
                                str(_batch_reason))
                            break
                if _batch_guard_failure is not None:
                    _batch_result = BatchResult(
                        "REJECTED", "batch:edit-hardening", tree_revision(root),
                        failures=(_batch_guard_failure,))
                    measurement.journal.emit(
                        APPLY_EVENT, producer="kernel_coordinated_change_ledger",
                        event_schema=APPLY_SCHEMA, dispatch=False, action="preflight",
                        status="REJECTED", batch_id=_batch_result.batch_id,
                        failures=[_batch_guard_failure.to_dict()])
                else:
                    _batch_result = atomic_apply(
                        root, tool_batch_args, journal=measurement.journal,
                        protected_paths=baseline_tests)
                obs = json.dumps(_batch_result.to_dict(), sort_keys=True,
                                 ensure_ascii=False)
                _batch_status = _batch_result.status
                bolt.receipt(
                    "edit", "str_replace batch", json.dumps(tool_batch_args, sort_keys=True),
                    "APPLIED" if _batch_status == "APPLIED" else
                    "STAGED" if _batch_status == "STAGED" else "REJECTED", obs[:240])
                if getattr(bolt, "_receipts", None):
                    _batch_receipt = bolt._receipts[-1]
                    _batch_receipt.update({
                        "change_set_ref": _batch_result.change_set_ref or _batch_result.batch_id,
                        "batch_id": _batch_result.batch_id,
                        "changed_paths": list(_batch_result.changed_paths),
                        "item_statuses": list(_batch_result.item_statuses),
                        "revision": _batch_result.revision,
                        "status": _batch_status,
                    })
                    if _batch_status == "APPLIED":
                        try:
                            from kernel_coordinated_change_ledger import record_runtime_batch_receipt
                            record_runtime_batch_receipt(measurement, _batch_receipt)
                        except (ImportError, AttributeError, TypeError, ValueError, OSError):
                            pass
                    measurement.amend_receipt(_batch_receipt)
                if _batch_status == "APPLIED":
                    if first_edit_step is None:
                        first_edit_step = step
                    edits += len(_batch_result.changed_paths)
            elif line.startswith("edit ") and tool_replace_module_args:
                _raw = line[5:].strip().strip("`'\"")
                if multifile:
                    try:
                        p, resolved_path = _multi_path(root, _raw)
                    except ValueError:
                        p, resolved_path = root / "__invalid_path__", _raw
                else:
                    resolved_path = _raw
                    try:
                        p = resolve_tool_path(root, _raw)
                    except AmbiguousToolPath as error:
                        obs = _typed_edit_rejection(
                            "NOT_EDITABLE", root / _raw, root=root, hint=str(error))
                        bolt.receipt("edit", line, line, "REJECTED", detail=obs)
                        measurement.observe(line, obs)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else
                                    {"role": "user", "content": obs})
                        continue
                if protected_test_path(p, baseline_tests):
                    obs = _typed_edit_rejection(
                        "NOT_EDITABLE", p, root=root,
                        hint="replace_module cannot edit a protected pre-existing test file")
                else:
                    expected, source, candidate_id = tool_replace_module_args
                    obs = replace_module(
                        p, expected, source=source, candidate_id=candidate_id,
                        root=root, protected_paths=baseline_tests,
                        public_brief=_task_public_brief(d))
                obs = _normalize_edit_rejection(obs, p, root=root)
                bolt.receipt(
                    "edit", f"replace_module {p.name}",
                    f"{p}|{tool_replace_module_args[0]}|{tool_replace_module_args[2] or 'source'}",
                    "APPLIED" if " applied." in obs else
                    "NO_OP" if " NO-OP" in obs else "REJECTED",
                    obs, target_path=receipt_target_path(root, p))
                if multifile:
                    _annotate_receipt(
                        bolt, paths=fileset.paths, scope=(resolved_path,),
                        before=_source_snapshot(root, fileset.paths),
                        after=_source_snapshot(root, fileset.paths),
                        selectors=tuple(getattr(bolt, "f2p_ids", ())), failure_tail=obs)
                if " applied." in obs:
                    if first_edit_step is None:
                        first_edit_step = step
                    edits += 1
            elif line.startswith("edit ") and tool_str_args:
                _raw = line[5:].strip().strip("`'\"")
                if multifile:
                    try:
                        p, resolved_path = _multi_path(root, _raw)
                    except ValueError:
                        p, resolved_path = root / "__invalid_path__", _raw
                else:
                    resolved_path = _raw
                    try:
                        p = resolve_tool_path(root, _raw)
                    except AmbiguousToolPath as error:
                        obs = _typed_edit_rejection(
                            "NOT_EDITABLE", root / _raw, root=root, hint=str(error))
                        bolt.receipt("edit", line, line, "REJECTED", detail=obs)
                        measurement.observe(line, obs)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs)
                                    if tool_call_id else {"role": "user", "content": obs})
                        continue
                if tool_insert_args is not None:
                    _insert_before = _source_snapshot(root, fileset.paths) if multifile else {}
                    if protected_test_path(p, baseline_tests):
                        obs = _typed_edit_rejection(
                            "NOT_EDITABLE", p, root=root,
                            hint="insert_before cannot modify a protected pre-existing test file")
                    else:
                        obs = _apply_revision_bound_insertion(
                            p, str(tool_insert_args.get("new_str") or ""),
                            tool_insert_args.get("target_ref"), root=root,
                            protected_paths=baseline_tests,
                            check_refs=tool_insert_args.get("check_refs", ()),
                            public_brief=_task_public_brief(d))
                    from kernel_action_identity import canonical_edit
                    _insert_attempt = canonical_edit(
                        bolt, p, dialect="insert_before", old="",
                        new=str(tool_insert_args.get("new_str") or ""))
                    bolt.receipt(
                        "edit", f"insert_before {p.name}",
                        json.dumps({"path": _raw, "target_ref": tool_insert_args.get("target_ref"),
                                    "check_refs": tool_insert_args.get("check_refs", ())},
                                   sort_keys=True, default=str),
                        "APPLIED" if " applied." in obs else "REJECTED", obs,
                        attempt=_insert_attempt,
                        target_path=receipt_target_path(root, p))
                    if getattr(bolt, "_receipts", None):
                        _insert_receipt = bolt._receipts[-1]
                        _insert_ref = tool_insert_args.get("target_ref")
                        _insert_ref_map = (_insert_ref if isinstance(_insert_ref, Mapping)
                                           else {})
                        _insert_receipt.update({
                            "edit_operation": "insert_before",
                            "target_ref": _insert_ref_map,
                            "source_receipt": _insert_ref_map.get("source_receipt"),
                            "write_authority": _insert_ref_map.get("authority_ref"),
                            "check_refs": [str(item) for item in
                                           (tool_insert_args.get("check_refs") or ())
                                           if str(item).strip()],
                        })
                        measurement.amend_receipt(_insert_receipt)
                    if " applied." in obs:
                        if first_edit_step is None:
                            first_edit_step = step
                        edits += 1
                        from kernel_rejection_template_restrict import release_after_product_mutation
                        _active_capture = getattr(measurement, "_active_capture", None)
                        release_after_product_mutation(
                            measurement,
                            request_id=getattr(_active_capture, "request_id", None),
                            tool_name="insert_before",
                        )
                    if multifile:
                        _annotate_receipt(
                            bolt, paths=fileset.paths, scope=(resolved_path,),
                            before=_insert_before,
                            after=_source_snapshot(root, fileset.paths),
                            selectors=tuple(getattr(bolt, "f2p_ids", ())),
                            failure_tail=obs)
                    measurement.observe(line, obs)
                    msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs) if tool_call_id else
                                {"role": "user", "content": obs})
                    tool_insert_args = None
                    tool_str_args = None
                    continue
                # E310: pre-existing test files are read-only (see baseline_tests rationale)
                if protected_test_path(p, baseline_tests):
                    obs = _typed_edit_rejection(
                        "NOT_EDITABLE", p, root=root,
                        hint=("this is a PRE-EXISTING test file; make the PRODUCT code satisfy "
                              "the original test contract"))
                    _receipt_paths = fileset.paths or (_edit_path_label(p, root),)
                    _record_rejected_edit_receipt(
                        bolt, root=root, path=p, paths=_receipt_paths,
                        scope=(resolved_path,), operation=f"str_replace {p.name}",
                        target=f"{p}|{tool_str_args[0][:200]}|{tool_str_args[1][:200]}",
                        dialect="str_replace", old=tool_str_args[0], new=tool_str_args[1],
                        detail=obs, multifile=multifile)
                else:
                    # WAVE 3 / D3: pre-dispatch retry gate -- evaluated AFTER action
                    # parsing (path/old_str/new_str are known above), BEFORE the
                    # editing tool executes below. Defaults OFF (see kernel wiring
                    # above): under KERNEL_MODE=retry_predispatch:ENFORCE, a DENY
                    # verdict blocks dispatch entirely -- str_replace_edit() is never
                    # called for that turn.
                    _pdg = getattr(bolt, "predispatch_gate", None)
                    _predispatch_blocked = False
                    _bound_reason = _bound_edit_refusal(root, p, tool_target_ref)
                    from kernel_action_identity import canonical_edit
                    _edit_attempt = canonical_edit(
                        bolt, p, dialect="str_replace", old=tool_str_args[0],
                        new=tool_str_args[1])
                    if _bound_reason is not None:
                        obs = "str_replace REJECTED: " + _bound_reason
                    elif _pdg is not None:
                        _pd_rec = {**_edit_attempt, "kind": "edit", "turn": step}
                        _pd_outcome = _pdg.evaluate(_pd_rec)
                        if _pd_outcome.blocks:
                            _predispatch_blocked = True
                    _edit_before = (_source_snapshot(root, fileset.paths)
                                    if multifile else {})
                    if _bound_reason is not None:
                        pass
                    elif _predispatch_blocked:
                        # WIRE CONNECTED: the editing tool is never invoked -- the
                        # forbidden action does not reach the executor. "REJECTED" in
                        # the text keeps this classified consistently with every other
                        # non-applied edit outcome downstream (repeat-cycling checks,
                        # E334 receipt bucketing below).
                        obs = ("edit REJECTED (pre-dispatch retry gate): " +
                              (_pd_outcome.decision.message or
                               "retry denied before dispatch."))
                    else:
                        if os.environ.get("STATEFRAME_JQ_FINISH", "off") == "on":
                            from kernel_jq_finish import validated_str_replace
                            def _validated_str_replace():
                                return validated_str_replace(
                                    p, *tool_str_args, root=root,
                                    protected_paths=baseline_tests,
                                    public_brief=_task_public_brief(d),
                                    journal=measurement.journal)

                            if _effect_boundary_enabled():
                                _str_before = p.read_text(errors="ignore") \
                                    if p.exists() else "<absent>"
                                _str_applied, _str_value, _str_rendered = _effect_application(
                                    "edit",
                                     {"path": str(p), "before": _str_before,
                                      "old": tool_str_args[0], "new": tool_str_args[1]},
                                    _validated_str_replace,
                                    before_snapshot=lambda: {
                                        "path": str(p), "content": _str_before,
                                    },
                                    effect_applied=lambda result: (
                                        isinstance(result, str)
                                        and " applied." in result),
                                    after_snapshot=lambda: {
                                        "path": str(p),
                                        "content": (p.read_text(errors="ignore")
                                                    if p.exists() else "<absent>"),
                                    },
                                )
                                obs = (str(_str_value) if _str_applied else
                                       (str(_str_value) if _str_value else _str_rendered or
                                        "str_replace SHADOW: effect withheld"))
                            else:
                                obs = _validated_str_replace()
                            if " applied." in obs:
                                from kernel_green_lock import note_applied_product_edit
                                note_applied_product_edit(p)
                        else:
                            obs = str_replace_edit(p, *tool_str_args,
                                                   target_ref=tool_target_ref, root=root,
                                                   public_brief=_task_public_brief(d),
                                                   journal=measurement.journal)
                    obs = _normalize_edit_rejection(obs, p, root=root)
                    if (_pdg is not None and _bound_reason is None
                            and _pd_outcome.advice and not _predispatch_blocked):
                        obs += "\n[kernel-retry] " + _pd_outcome.advice
                    # E334: typed edit receipt (payload-fingerprinted).
                    bolt.receipt(
                        "edit", f"str_replace {p.name}",
                        f"{p}|{tool_str_args[0][:200]}|{tool_str_args[1][:200]}",
                        "REFUSED_PREDISPATCH" if _predispatch_blocked else
                        "APPLIED" if "applied." in obs else
                        "NO_OP" if "NO-OP" in obs else
                        "REJECTED" if "REJECTED" in obs else "FAILED",
                        obs, attempt=_edit_attempt,
                        target_path=receipt_target_path(root, p))
                    if multifile:
                        _annotate_receipt(
                            bolt, paths=fileset.paths, scope=(resolved_path,),
                            before=_edit_before,
                            after=_source_snapshot(root, fileset.paths),
                            selectors=tuple(getattr(bolt, "f2p_ids", ())),
                            failure_tail=obs)
                    # E407: an APPLIED edit re-observes at the NEW revision — fresh-only
                    # rendering supersedes stale facts (over-refresh safe).
                    if "applied." in obs and getattr(bolt, "world", None) is not None:
                        try:
                            bolt.world.observe_source(
                                path=str(p.relative_to(root)), content=p.read_text(),
                                revision=max(getattr(bolt, "workspace_revision", 0), bolt.world.current_revision),
                                evidence_ref=f"edit@{step}")
                        except Exception as error:                   # noqa: BLE001
                            measurement.journal.emit('world_observation_error', operation='edit',
                                                     error_type=type(error).__name__)
                    # Relate the applied edit to the failing-test anchor.
                    if "applied." in obs:
                        _rw = bolt.organ_out("edit_relevance", bolt.edit_relevance(
                            str(p.relative_to(root)) if p.is_absolute() else str(p)))
                        if _rw:
                            obs = obs + "\n" + _rw
                        # E325: editing a SELF-CREATED test after it went red taints it.
                        if str(p) in created_this_run:
                            # B2 (effect boundary): routed through authorize() so
                            # KERNEL_MODE's effects axis actually gates the taint
                            # mutation, not just this text (kernel_bolt.py:686 bug).
                            _tw = bolt.authorize(
                                "self_oracle", version="B2-2026-09-20",
                                decision_fn=lambda _p=p: bolt.evaluate_created_test_edit(
                                    str(_p.relative_to(root))))
                            if _tw:
                                obs = obs + "\n" + _tw
                if "applied." in obs:
                    if first_edit_step is None and "applied." in obs:
                        first_edit_step = step
                    edits += 1
            elif line.startswith("edit "):
                # Toolcall mode delivers the source STRUCTURED — no fence regex at all.
                blocks = ([tool_edit_src] if tool_edit_src else
                          re.findall(r"```(?:python|py)?[^\n]*\n(.*?)```", reply, re.S))
                _raw = line[5:].strip().strip("`'\"")
                if multifile:
                    try:
                        p, resolved_path = _multi_path(root, _raw)
                    except ValueError:
                        p, resolved_path = root / "__invalid_path__", _raw
                else:
                    resolved_path = _raw
                    try:
                        p = resolve_tool_path(root, _raw)
                    except AmbiguousToolPath as error:
                        obs = _typed_edit_rejection(
                            "NOT_EDITABLE", root / _raw, root=root, hint=str(error))
                        bolt.receipt("edit", line, line, "REJECTED", detail=obs)
                        measurement.observe(line, obs)
                        msgs.append(_tool_reply_for_call(msgs, tool_call_id, obs)
                                    if tool_call_id else {"role": "user", "content": obs})
                        continue
                _rejection_template_source = (
                    blocks[-1] if blocks and isinstance(blocks[-1], str) else None)
                _rejection_template_path = p
                if not blocks:
                    # REMEMBER WHICH FILE IS AWAITING A BLOCK. Measured: the agent sent
                    # `edit <path>`, was told a block was needed, supplied the block in the
                    # very next message — and the parser found no verb in a pure code block,
                    # defaulted to `done`, and discarded a complete function. Holding the
                    # pending path makes the two-message form work, which is how models
                    # naturally write an edit.
                    pending_edit_path = p
                    obs = ("edit needs a ```python block with the complete function or class. "
                           + ("Call the `edit` tool next with a nonempty `function_source` "
                              "for " if (native_tool_batches or action_schema_mode
                                           or contract_enabled()) else
                              "Send just the block next and it will be applied to ")
                           + f"{_raw}.")
                elif protected_test_path(p, baseline_tests):
                    obs = _typed_edit_rejection(
                        "NOT_EDITABLE", p, root=root,
                        hint=("this is a PRE-EXISTING test file; make the PRODUCT code satisfy "
                              "the original test contract"))
                    from kernel_action_identity import canonical_edit
                    _edit_attempt2 = canonical_edit(
                        bolt, p, dialect="replace_fn", old="", new=blocks[-1].strip())
                    _edit_before = (_source_snapshot(root, fileset.paths)
                                    if multifile else {})
                    bolt.receipt(
                        "edit", f"replace_fn {p.name}",
                        f"{p}|{blocks[-1][:400]}", "REJECTED", obs,
                        attempt=_edit_attempt2,
                        target_path=receipt_target_path(root, p))
                    if multifile:
                        _annotate_receipt(
                            bolt, paths=fileset.paths, scope=(resolved_path,),
                            before=_edit_before,
                            after=_source_snapshot(root, fileset.paths),
                            selectors=tuple(getattr(bolt, "f2p_ids", ())),
                            failure_tail=obs)
                else:
                    # WAVE 3 / D3, ADVERSARIAL-REVIEW fix: this is the SECOND live edit
                    # -application path (fence/whole-function-replace, taken whenever
                    # `tool_str_args` is unset -- i.e. REACT_TOOLCALL=0 text mode, which
                    # a weak local model with unreliable structured tool-calling is
                    # exactly the case likely to use). The str_replace branch above wires
                    # the same gate; this branch did not, so under
                    # KERNEL_MODE=retry_predispatch:ENFORCE a forbidden repeat edit could
                    # still reach replace_fn()/_replace_file() unguarded through here.
                    # Mirrors the str_replace call site's pattern exactly: defaults OFF
                    # (no-op when KERNEL_MODE is unset), fails to SHADOW, never blocks
                    # unless a DENY verdict is actually returned.
                    _pdg2 = getattr(bolt, "predispatch_gate", None)
                    _predispatch_blocked2 = False
                    _bound_reason2 = _bound_edit_refusal(root, p, tool_target_ref)
                    from kernel_action_identity import canonical_edit
                    _edit_attempt2 = canonical_edit(
                        bolt, p, dialect="replace_fn", old="", new=blocks[-1].strip())
                    if _bound_reason2 is not None:
                        obs = _typed_edit_rejection(
                            "BOUND_TARGET", p, root=root, hint=_bound_reason2)
                    elif _pdg2 is not None:
                        _pd_rec2 = {**_edit_attempt2, "kind": "edit", "turn": step}
                        _pd_outcome2 = _pdg2.evaluate(_pd_rec2)
                        if _pd_outcome2.blocks:
                            _predispatch_blocked2 = True
                    _edit_before = (_source_snapshot(root, fileset.paths)
                                    if multifile else {})
                    if _bound_reason2 is not None:
                        pass
                    elif _predispatch_blocked2:
                        # WIRE CONNECTED (second path): same classification as the
                        # str_replace branch's pre-dispatch refusal.
                        obs = _typed_edit_rejection(
                            "RETRY_GATE", p, root=root,
                            hint=(_pd_outcome2.decision.message or
                                  "retry denied before dispatch."))
                    elif multifile:
                        if resolved_path not in fileset.paths:
                            obs = _typed_edit_rejection(
                                "NOT_EDITABLE", p, root=root,
                                hint=(f"{resolved_path} is not listed in the EditAuthority's "
                                      "editable_paths"))
                        elif (fileset.entry_path and fileset.entry_fn) or _definition_mode:
                            _definition_name, _definition_error = _top_level_definition_name(
                                blocks[-1].strip())
                            if _definition_error is not None:
                                obs = _typed_edit_rejection(
                                    "INVALID_FUNCTION_SOURCE", p, root=root,
                                    hint=_definition_error)
                            else:
                                obs = replace_fn(p, _definition_name, blocks[-1].strip(),
                                                  multi_file=True,
                                                  target_ref=tool_target_ref, root=root,
                                                  public_brief=_task_public_brief(d),
                                                  journal=measurement.journal)
                        else:
                            obs = _replace_file(
                                p, blocks[-1].strip(), root=root,
                                public_brief=_task_public_brief(d),
                                journal=measurement.journal)
                    else:
                        obs = replace_fn(p, fn, blocks[-1].strip(),
                                         target_ref=tool_target_ref, root=root,
                                         public_brief=_task_public_brief(d),
                                         journal=measurement.journal)
                    obs = _normalize_edit_rejection(
                        obs, p, root=root,
                        symbol=(fileset.entry_fn if multifile and fileset.entry_fn else fn))
                    if (_pdg2 is not None and _bound_reason2 is None
                            and _pd_outcome2.advice and not _predispatch_blocked2):
                        obs += "\n[kernel-retry] " + _pd_outcome2.advice
                    if first_edit_step is None and "applied." in obs:
                        first_edit_step = step
                    edits += 1
                    pending_edit_path = None
                    # E334: typed edit receipt (payload-fingerprinted).
                    bolt.receipt("edit", f"replace_fn {p.name}",
                                 f"{p}|{blocks[-1][:400]}",
                                 "REFUSED_PREDISPATCH" if _predispatch_blocked2 else
                                 "APPLIED" if "applied." in obs else
                                 "NO_OP" if "NO-OP" in obs else
                                 "REJECTED" if "REJECTED" in obs else "FAILED",
                                 obs, attempt=_edit_attempt2,
                                 target_path=receipt_target_path(root, p))
                    if multifile:
                        _annotate_receipt(
                            bolt, paths=fileset.paths, scope=(resolved_path,),
                            before=_edit_before,
                            after=_source_snapshot(root, fileset.paths),
                            selectors=tuple(getattr(bolt, "f2p_ids", ())),
                            failure_tail=obs)
                    # Relate the applied edit to the failing-test anchor.
                    if "applied." in obs:
                        _rw = bolt.organ_out("edit_relevance", bolt.edit_relevance(
                            str(p.relative_to(root)) if p.is_absolute() else str(p)))
                        if _rw:
                            obs = obs + "\n" + _rw
                        # E325: editing a SELF-CREATED test after it went red taints it.
                        if str(p) in created_this_run:
                            # B2 (effect boundary): routed through authorize() so
                            # KERNEL_MODE's effects axis actually gates the taint
                            # mutation, not just this text (kernel_bolt.py:686 bug).
                            _tw = bolt.authorize(
                                "self_oracle", version="B2-2026-09-20",
                                decision_fn=lambda _p=p: bolt.evaluate_created_test_edit(
                                    str(_p.relative_to(root))))
                            if _tw:
                                obs = obs + "\n" + _tw
            elif line.startswith("test"):
                rest = line[4:].strip()
                _test_before = (_source_snapshot(root, fileset.paths)
                                if multifile else {})
                from kernel_public_evidence import requested_check_targets
                _test_selectors = (tuple(requested_check_targets(root, rest)) if rest else
                                   tuple(getattr(bolt, "f2p_ids", ())))
                if multifile and _test_selectors:
                    obs, _ = run_tests_for_selectors(
                        root, fn, _test_selectors, d["rel"], row=d)
                else:
                    obs = run_tests(root, fn, rest or None, d["rel"], row=d)
                plain_observation = obs
                if os.environ.get("STATEFRAME_EMPIRICAL_FIXES", "off") == "on":
                    from kernel_empirical_fixes import flag_brief_traceback_lines
                    _brief_for_line_check = next(
                        (str(d.get(key)) for key in (
                            "_bridge_ordinary_brief", "msg", "brief", "symptom")
                         if isinstance(d.get(key), str) and d.get(key).strip()), "")
                    _line_mismatches = flag_brief_traceback_lines(
                        _brief_for_line_check, root)
                    if _line_mismatches:
                        _mismatch_text = (
                            "[kernel-harness] TRACEBACK_LINE_MISMATCH "
                            + json.dumps(_line_mismatches[:5], sort_keys=True,
                                         separators=(",", ":")))
                        obs += "\n" + _mismatch_text
                        measurement.journal.emit(
                            "harness_traceback_line_mismatch",
                            findings=_line_mismatches[:5],
                            rest=rest or "bare-run",
                            dispatch=False,
                        )
                plain_observation = obs
                tests += 1
                from kernel_wire_identity import public_wire_text
                obs = public_wire_text(obs, measurement.journal)
                # E409: the exit parse + traceback signal were
                # computed ~50 lines BELOW the receipt, so every test receipt carried the
                # PREVIOUS run's verdict and anchor (stale-by-one). Hoisted here so the
                # receipt, hypothesis-open, and nit-picker all see THIS run's truth.
                _m_exit0 = re.search(r"pytest exit (\d+)", obs)
                bolt.last_test_exit = int(_m_exit0.group(1)) if _m_exit0 else None
                if bolt.last_test_exit is not None:
                    from kernel_rejection_template_restrict import release_after_test_run
                    _active_capture = getattr(measurement, "_active_capture", None)
                    release_after_test_run(
                        measurement,
                        request_id=getattr(_active_capture, "request_id", None),
                        tool_name=("run_tests" if locals().get("_name") == "run_tests"
                                   else "test"),
                    )
                _sig = None
                if bolt.last_test_exit == 1:
                    try:
                        import tracebacks as _tb0
                        if not hasattr(bolt, "_tb_index"):
                            bolt._tb_index = _tb0.repo_index_from_paths(
                                [str(p.relative_to(root)) for p in root.rglob("*.py")])
                        _sig = _tb0.traceback_signal(obs, repo_index=bolt._tb_index)
                    except Exception:                                # noqa: BLE001
                        _sig = None
                # E428: feed the pivot organ THIS run's verdict; a fired advisory
                # (new edge, or loop-ban) is delivered same-turn via the pending slot.
                _host_refused = obs.startswith(HOST_REFUSED_PREFIX)
                if getattr(bolt, "pivot", None) is not None and not _host_refused:
                    try:
                        _pvo = bolt.pivot.observe({
                            "kind": "test",
                            "fingerprint": _sig or f"exit:{bolt.last_test_exit}",
                            "result_class": ("GREEN" if bolt.last_test_exit == 0
                                             else "PRODUCT_RED"),
                            "turn": step,
                        })
                        if _pvo.advisory is not None:
                            bolt._pivot_pending = bolt.organ_out(
                                "pivot", _pvo.advisory.render())
                    except Exception:                                # noqa: BLE001
                        pass
                # E417: every test run's COUNTS become expectations; deltas on the same
                # enumeration render the "you saw N, now M — investigate" card (the
                # consistent accounting for the expectation ledger).
                if getattr(bolt, "expect", None) is not None and not _host_refused:
                    try:
                        from expectation_ledger import parse_pytest_result
                        _tsr = parse_pytest_result(
                            obs, exit_code=bolt.last_test_exit or 0,
                            scope=(rest or "full"), selected=[rest] if rest else [])
                        _eres = _observe_expectation_suite(
                            bolt.expect, _tsr, journal=measurement.journal,
                            environment="react",
                            subject=_expectation_subject(fn, d),
                            selector=rest or "full", turn=step,
                            workspace_revision=getattr(bolt, "workspace_revision", 0))
                        _adv = [r.advisory for r in _eres if r.advisory]
                        if _adv:
                            _eline = bolt.organ_out("expectation",
                                                    "[kernel-expect] " + " | ".join(_adv[:2]))
                            if _eline:
                                obs = obs + "\n" + _eline
                                bolt._rec("expectation", "delta_advisory",
                                          _adv[0][:100])
                    except Exception as _e:                          # noqa: BLE001
                        print(f"    [kernel-wiring] expectation observe failed: "
                              f"{type(_e).__name__}")
                # E325: remember which test file just went red (self-oracle taint input).
                if "TESTS FAILED" in obs and rest:
                    bolt.note_test_red(rest.split("::")[0])
                # E334: typed test receipt (+E342: anchored so the hypothesis graph can
                # relate it; unanchored reds are inert there by design).
                _anch = (locals().get("_sig") or {}).get("primary_anchor") \
                    if bolt.last_test_exit == 1 else None
                # E416: a pure assertion red carries NO traceback anchor (measured:
                # collect-5, 3 REDs all NO-ANCHOR -> close-look could never trigger on
                # its own target class). Fallback: the region under repair IS the anchor
                # when the traceback is silent — honest and stable across reds.
                if _anch is None and bolt.last_test_exit == 1:
                    # E419: anchor to the fn's REAL def line (the row's changed_line can
                    # be junk — collect's was 1, making the dossier region line 1-1).
                    _fline = d.get("changed_line") or 1
                    try:
                        _m = re.search(rf"^\s*def {re.escape(fn)}\b",
                                       target.read_text(), re.M)
                        if _m:
                            _fline = target.read_text()[:_m.start()].count("\n") + 1
                    except Exception:                                # noqa: BLE001
                        pass
                    _anch = f"{d['rel']}:{_fline}:{fn}"
                # E342: a fresh red anchor opens a kernel-derived hypothesis BEFORE the
                # receipt lands, so the receipt's evidence has a subject to relate to.
                if _anch and getattr(bolt, "hyp", None):
                    try:
                        bolt.hyp.open_from_anchor(
                            _anch, (locals().get("_sig") or {}).get("family") or "other")
                    except Exception as _e:                          # noqa: BLE001
                        print(f"    [kernel-wiring] open_from_anchor: {type(_e).__name__}")
                from kernel_continuation_evidence import classify_run_test_exit
                # Continuation effects need the typed exit vocabulary even when the
                # separate host-closure arm is off.  The old fallback collapsed every
                # non-zero/unknown subprocess result to RED, so exit 5 and an
                # unstructured runner result could strand the effect as if a product
                # check had actually executed and failed.  Keep the legacy producer
                # for the non-continuation history arm, but never use it for a live
                # continuation producer.
                host_registry = (
                    getattr(measurement, "stateframe_host_closure_mode", "off") == "on"
                    or getattr(measurement, "continuation", None) is not None)
                if _host_refused:
                    run_test_result_class = "HARNESS_ERROR"  # host refusal: no verdict
                elif obs.startswith("[selector] REFUSED_PREDISPATCH"):
                    run_test_result_class = "REFUSED_PREDISPATCH"
                else:
                    run_test_result_class = (classify_run_test_exit(bolt.last_test_exit)
                                             if host_registry
                                             else "GREEN" if bolt.last_test_exit == 0 else "RED")
                if (os.environ.get("KERNEL_REAL04_UNKNOWN_TEST_REPAIR", "off").strip().lower()
                        != "off" and host_registry and not _host_refused):
                    from kernel_real04_bugs import classify_real04_run_test_result
                    run_test_result_class = classify_real04_run_test_result(
                        obs, bolt.last_test_exit, default=run_test_result_class)
                bolt.receipt("test", rest or "bare-run", rest or "bare",
                             run_test_result_class,
                             anchor=_anch, test_output=obs)
                if getattr(measurement, "structure_facts_mode", "off") == "on":
                    measurement.record_structure_test_facts(
                        obs, preferred_paths=(str(d.get("rel", "")),))
                if multifile:
                    _annotate_receipt(
                        bolt, paths=fileset.paths, scope=fileset.paths,
                        before=_test_before,
                        after=_source_snapshot(root, fileset.paths),
                        selectors=_test_selectors,
                        failure_tail=obs)
                else:
                    _attach_failure_tail(bolt, obs)
                if bolt.last_test_exit == 4:
                    _selector_note = _selector_error_repair_note(
                        bolt, rest, step)
                    if _selector_note and interventions:
                        msgs.append({"role": "user", "content": _selector_note})
                # E525 (after the receipt lands — never stale-by-one): fold this test into the obligation graph; a SATISFIED->MISSING
                # transition renders ONE bounded "REGRESSION: X was passing at turn N,
                # your edit at M broke it" line (the E464/E510 class, mechanically).
                if getattr(bolt, "obgraph", None) is not None:
                    try:
                        from kernel_obligations import (update_obligation_graph as _uog,
                                                        render_regression_notice as _rrn,
                                                        pending_regression_notices as _prn)
                        bolt.obgraph = _uog(bolt.obgraph, list(getattr(bolt, "_receipts", ()))[-1:])
                        _rrn(bolt.obgraph)
                        for _nl in _prn():
                            _ol = bolt.organ_out("obligation_regression", "[kernel-regress] " + _nl[:300])
                            if _ol:
                                bolt._obreg_pending = ((getattr(bolt, "_obreg_pending", None) or "") +
                                                       ("\n" if getattr(bolt, "_obreg_pending", None) else "") + _ol)
                    except Exception:                                # noqa: BLE001
                        pass
                # E407: test outcomes are world knowledge too (selector -> verdict).
                if getattr(bolt, "world", None) is not None and rest:
                    try:
                        bolt.world.observe_test_selector(
                            selector=rest,
                            result_class={0: "GREEN", 1: "RED"}.get(
                                bolt.last_test_exit, "UNKNOWN"),
                            revision=getattr(bolt, "workspace_revision", 0),
                            evidence_ref=f"test@{step}")
                    except Exception:                                # noqa: BLE001
                        pass
                # Deploop-audit occurrence-1 card: exit 5 = NOTHING COLLECTED — an
                # unchanged retry is futile by construction, say so immediately.
                if bolt.last_test_exit == 5 and rest and \
                        rest not in getattr(bolt, "_no_oracle_seen", set()):
                    if not hasattr(bolt, "_no_oracle_seen"):
                        bolt._no_oracle_seen = set()
                    bolt._no_oracle_seen.add(rest)
                    obs += ("\n[kernel] exit 5 = that selector COLLECTED NOTHING — "
                            "retrying it unchanged can never succeed. Find the real "
                            "node first (grep the test name) or pick another target.")
                # JobQueue completion: the first current public green automatically runs the
                # small frozen public-brief audit.  It is host evidence, not a model
                # proposal, and is deliberately shared by history and frame arms.
                if (bolt.last_test_exit == 0 and target is not None
                        and getattr(measurement, "jq_finish_enabled", False)):
                    try:
                        _jq_row = d if isinstance(d, Mapping) else {}
                        _jq_brief = next((str(_jq_row.get(key)) for key in (
                            "_bridge_ordinary_brief", "msg", "brief", "symptom")
                            if isinstance(_jq_row.get(key), str) and _jq_row.get(key).strip()), "")
                        _jq_path = target.resolve().relative_to(root.resolve()).as_posix()
                        _jq_text = measurement.jobqueue_contract_audit(
                            source=target.read_text(encoding="utf-8"),
                            brief=_jq_brief, path=_jq_path)
                        if _jq_text:
                            obs += "\n" + _jq_text
                            _jq_result = "PASS_CASES" if measurement.jq_audit.get("status") == "PASS_CASES" else "FAIL_WITH_WITNESS"
                            bolt.receipt("contract_audit", "public contract audit", _jq_path,
                                         _jq_result, _jq_text,
                                         target_path=receipt_target_path(root, target))
                            if getattr(bolt, "_receipts", None):
                                bolt._receipts[-1]["current_revision"] = (
                                    "sha256:" + hashlib.sha256(
                                        target.read_text(encoding="utf-8").encode()).hexdigest())
                    except (OSError, UnicodeError, ValueError) as _jq_error:
                        obs += f"\n[contract-audit] audit unavailable: {type(_jq_error).__name__}: {_jq_error}"
            else:
                obs = ("Unrecognised command. Use exactly one of: grep, read, edit, test, done.")

            if line.startswith("edit "):
                obs = _append_structure_fact_observation(measurement, bolt, obs, "edit")
            elif line.startswith("test"):
                obs = _append_structure_fact_observation(measurement, bolt, obs, "test")
            if plain_observation is None:
                plain_observation = obs

            # CYLINDERS 1 + 2: ground-truth a dead grep, and hand back the previous answer to
            # a repeated QUERY (never to a repeated edit/test — those are actions whose
            # payload changes, and blocking them would kill the iteration the loop exists for).
            if line.startswith("test"):
                # E409: exit + _sig now computed at the TOP of the test handler (stale-
                # by-one receipt fix); this block only augments obs from the hoisted _sig.
                # E312: TRACEBACK ANCHOR ADAPTER (~2-4 expected
                # conversions): tracebacks.traceback_signal was a fully built, never-wired
                # seam. Every red run now carries its deterministic primary anchor
                # (file:line:function) + failure family — structured ground truth derived
                # from the traceback the model already sees, so it stops editing the wrong
                # layer. Never suggests the fix.
                if bolt.last_test_exit == 1 and locals().get("_sig"):
                    try:
                        if _sig.get("primary_anchor"):
                            obs += (f"\n[kernel] the failure anchors at "
                                    f"{_sig['primary_anchor']}"
                                    + (f" (failure family: {_sig['family']})"
                                       if _sig.get("family") not in (None, "unknown")
                                       else ""))
                        # E329: multi-hop reds get the measured call
                        # path — the fix belongs at the broken CONTRACT's hop.
                        _cp = bolt.organ_out(
                            "call_path", bolt.call_path_card(_sig.get("anchors") or []))
                        if _cp:
                            obs += "\n" + _cp
                    except Exception:                            # noqa: BLE001
                        pass                  # diagnosis is an aid; never break the loop
                # E313: structured test record for the orientation round
                bolt.sure_observe({"kind": "test", "step": step,
                                   "exit": bolt.last_test_exit,
                                   "verdict": ("pass" if bolt.last_test_exit == 0 else
                                               "red" if bolt.last_test_exit == 1 else
                                               "no_verdict"),
                                   "primary_anchor": (locals().get("_sig") or {}).get(
                                       "primary_anchor") if bolt.last_test_exit == 1
                                   else None})
                # FIRE CONDITION (E313b, continuous): every red/no-verdict test or deep
                # repeat streak offers a get_sure round; the bolt dedups by anchor
                # signature + cooldown so only NEW uncertainty actually fires.
                if bolt.last_test_exit in (1, 2, 3, 4, 5) or \
                        max(bolt._repeat_counts.values(), default=0) >= 10:
                    _sure_line = bolt.organ_out("get_sure", bolt.sure_orient(step=step))
                    if _sure_line:
                        obs = obs + "\n" + _sure_line
                # E317: a green run that selected none of the relevant regression nodes
                # is named as unrelated — navigation, never answer material.
                # B2 (effect boundary): routed through authorize() -- test_relevance
                # mutates _last_relevant_match/_last_green_irrelevant/
                # _self_oracle_tainted as a side effect of being called; those effects
                # must honor KERNEL_MODE too, not just the advisory text.
                _nav_req = (locals().get("rest") or line or "")
                _nav_exit = bolt.last_test_exit
                _nav_line = bolt.authorize(
                    "navigation", version="B2-2026-09-20",
                    decision_fn=lambda: bolt.evaluate_test_relevance(_nav_req, _nav_exit))
                if _nav_line:
                    obs = obs + "\n" + _nav_line
                # E318: an environmental red is inadmissible as patch evidence.
                _adm_line = bolt.organ_out(
                    "admissibility", bolt.result_admissibility(obs, bolt.last_test_exit))
                if _adm_line:
                    obs = obs + "\n" + _adm_line
                # E365: on a red, the semantic bridge derives typed evidence from the
                # receipt tape + current target source and (if the frontier wants it)
                # emits ONE bounded evidence-gap advisory. Never a patch.
                if bolt.last_test_exit == 1 and getattr(bolt, "sem", None):
                    try:
                        _src_now = (root / d["rel"]).read_text(errors="replace") \
                            if (root / d["rel"]).is_file() else ""
                        # E365b: route EXPLICITLY from the task fn — the auto-router
                        # needs structured anchors our string receipts don't carry
                        # (measured: 6 ARMED events, zero derives in a full rss run).
                        _pmap = {"fetch_rss_feed": "rss",
                                 "_watchdog_auto_pause_trigger": "watchdog",
                                 "require_isolated_f2_database": "db",
                                 "_ack_decision": "ack"}
                        _rcpts = list(getattr(bolt, "_receipts", []))
                        # E365c: producers trigger on an ANCHORED red. When the live red
                        # didn't anchor (common), synthesize the trigger from the task's
                        # KNOWN target (row metadata, not invention).
                        if not any(r.get("anchor") for r in _rcpts):
                            _rcpts.append({"kind": "test", "intent": "route",
                                           "fingerprint": "route", "result_class": "RED",
                                           "why": "", "detail": "synthetic-trigger",
                                           "turn": step, "workspace_revision":
                                               getattr(bolt, "workspace_revision", 0),
                                           "ts": 0,
                                           "anchor": f"{d['rel']}:"
                                                     f"{d.get('changed_line', 1)}:{fn}"})
                        _evs = bolt.sem.derive(_rcpts, _src_now,
                                               producer=_pmap.get(fn))
                        if _evs is not None:
                            _front = bolt.hyp.frontier() if getattr(bolt, "hyp", None) else []
                            _adv = bolt.sem.consume(_front, _evs)
                            if isinstance(_adv, str) and _adv:
                                obs = obs + "\n" + _adv
                    except Exception as _e:                          # noqa: BLE001
                        print(f"    [kernel-wiring] semantic derive failed: {type(_e).__name__}")
                # E320: exact guard text at a product-code anchor.
                if bolt.last_test_exit == 1:
                    bolt.note_witness_values(obs)   # E340: harvest concrete failure values
                    _tg_line = bolt.organ_out("guard_tracer", bolt.guard_at_anchor())
                    if _tg_line:
                        obs = obs + "\n" + _tg_line
                    # E323: once, on first red — the target's live contract.
                    _cc_line = bolt.organ_out("contract_card", bolt.contract_card())
                    if _cc_line:
                        obs = obs + "\n" + _cc_line
                # #43 second half: a repeated test that produced NO VERDICT (exit 4/5 — bad
                # selector, nothing collected) is not iteration, it is a loop the query
                # defenses cannot see. Feed it to the surgery ledger. Real red runs (exit 1)
                # stay exempt — repeating those is red-green work.
                if bolt.last_test_exit in (4, 5):
                    bolt._repeat_counts[line] = bolt._repeat_counts.get(line, 1) + 1
            # E298 graft #2: an edit that FAILS identically is not iteration either —
            # 4bB retried `edit <missing file>` 25+ times, invisible to every defense
            # because actions are exempt. Count failing edits toward the surgery ledger;
            # successful edits stay exempt (that is the iteration the loop exists for).
            if line.split()[0] == "edit" and _edit_observation_is_rejection(obs):
                bolt._repeat_counts[line] = bolt._repeat_counts.get(line, 1) + 1
            # E444: idle governor at the ONE delivery chokepoint. It records each
            # decision; default-on V keeps an already executed successful source page
            # intact, while failed or unexecuted collection actions can still be refused.
            # Edits/tests feed observe_receipt so product work resets the idle counter.
            _igv = getattr(bolt, "idlegov", None)
            if _igv is not None:
                try:
                    from kernel_idle_governor import (
                        GovernorDecision, post_execution_observation,
                    )
                    from kernel_gate_switches import deliver_requested_bytes_enabled
                    _atoks = line.split()
                    _akind = _atoks[0] if _atoks else ""
                    _deliver_requested = deliver_requested_bytes_enabled()
                    _successful_source_delivery = False
                    if _akind in ("read", "grep", "list"):
                        _gpath = _atoks[1].strip("`'\"") if len(_atoks) > 1 else ""
                        _gaction = {"kind": _akind, "path": _gpath, "turn": step}
                        if _akind == "read":
                            # Budget only re-reads of already-delivered unchanged bytes:
                            # the digest of THIS read's output is the novelty key, and a
                            # path the request lists as required-but-missing is exempt.
                            if _deliver_requested:
                                _successful_source_delivery = (
                                    _read_result == "OK"
                                    and _read_delivery.get("delivery_status") == "DELIVERED"
                                )
                                _requested_byte_count = _read_delivery.get(
                                    "requested_bytes_delivered", 0)
                                _has_requested_bytes = (
                                    type(_requested_byte_count) is int
                                    and _requested_byte_count > 0
                                )
                                if _successful_source_delivery and _has_requested_bytes:
                                    _gaction["content_digest"] = str(
                                        _read_delivery.get("requested_bytes_sha256") or "")
                                else:
                                    # Empty pages and proof-backed stubs add no
                                    # source-byte novelty to the read budget.
                                    _gaction["path"] = ""
                            else:
                                _gaction["content_digest"] = hashlib.sha256(
                                    str(obs).encode("utf-8", "replace")).hexdigest()[:32]
                            _gaction["exempt"] = _read_path_is_deferred_required(
                                measurement, _gpath)
                        elif _akind == "grep" and _deliver_requested:
                            _successful_source_delivery = (
                                _grep_error is None and _grep_page is not None)
                        _gd = _igv.should_allow(_gaction)
                        _gmsg = getattr(getattr(_igv, "_last_receipt", None),
                                        "message", "") or ""
                        if (_akind == "read" and _gd != GovernorDecision.DENY
                                and getattr(_igv, "governed", False)):
                            # Live proof that a governed budget still admitted a first
                            # read / changed bytes / required-missing path.
                            try:
                                measurement.journal.emit(
                                    "read_budget_admitted", producer="kernel_idle_governor",
                                    step=step, path=_gpath, dispatch=False,
                                    reason=str(getattr(getattr(_igv, "_last_receipt", None),
                                                       "event", "")),
                                    exempt=bool(_gaction.get("exempt")))
                            except Exception:  # noqa: BLE001 - observability only
                                pass
                        if _gd == GovernorDecision.DENY:
                            obs = post_execution_observation(
                                obs, _gd, _gmsg,
                                source_bytes_delivered=(
                                    _deliver_requested and _successful_source_delivery),
                            )
                        elif _gmsg:
                            _gl = bolt.organ_out("idle_governor",
                                                 "[kernel-governor] " + _gmsg)
                            if _gl:
                                obs = obs + "\n" + _gl
                    elif _akind == "edit":
                        _eres = ("NO_OP" if _edit_observation_is_rejection(obs)
                            else "GREEN")
                        _igv.observe_receipt({"kind": "edit", "turn": step,
                                              "result_class": _eres})
                    elif _akind == "test":
                        _igv.observe_receipt({
                            "kind": "test", "turn": step,
                            "result_class": ("GREEN" if bolt.last_test_exit == 0
                                             else "RED")})
                except Exception:                                    # noqa: BLE001
                    pass
            # E467: reflection gate — deterministic prediction capture. A successful
            # edit implicitly predicts the next test goes green; a red next test is a
            # PredictionMiss the trigger set hears. Feed every action for triggers.
            _rfl = getattr(bolt, "reflect", None)
            if _rfl is not None:
                try:
                    _rtoks = line.split()
                    _rkind = _rtoks[0] if _rtoks else ""
                    if _rkind == "edit" and not _edit_observation_is_rejection(obs):
                        _rfl.record_expectation("edit", "next-test", "green")
                    elif _rkind == "test":
                        _rmiss = _rfl.compare_outcome("next-test", obs[:240])
                        if _rmiss is not None:
                            _rl = bolt.organ_out(
                                "reflection",
                                "[kernel-reflect] Your last edit predicted green; "
                                "the test is still red. Before the next edit: is the "
                                "hypothesis wrong, or the edit incomplete? State "
                                "which, with the evidence line.")
                            if _rl:
                                obs = obs + "\n" + _rl
                    _rfl.inspect({"kind": _rkind, "intent": line[:200],
                                  "result_class": ("RED" if _rkind == "test"
                                                   and bolt.last_test_exit
                                                   else "OK"),
                                  "fingerprint": str(locals().get("_sig") or ""),
                                  "turn": step})
                except Exception:                                    # noqa: BLE001
                    pass
            # E456: organ B — receipts become measurements; edit rationales are
            # checked for unmeasured assertions (advisory names the ONE probe).
            _gtb = getattr(bolt, "gt", None)
            if _gtb is not None:
                try:
                    _gtoks = line.split()
                    _gkind = _gtoks[0] if _gtoks else ""
                    _grec = {"kind": _gkind, "turn": step, "intent": line[:200],
                             "detail": obs[:300],
                             "path": (_gtoks[1].strip("`'\"") if len(_gtoks) > 1
                                      else ""),
                             "workspace_revision": getattr(bolt,
                                                            "workspace_revision", 0)}
                    if measurement.view_kind == 'operational':
                        # An edit tool name is not a claim that the product defines
                        # that symbol. Keep legacy tape inference out of live frames.
                        _grec['assertion_policy'] = 'explicit_only'
                    _gtb.observe_receipt(_grec)
                    if _gkind == "edit":
                        for _mr in _gtb.check_action(_grec):
                            if _mr.challenge and _mr.advisory:
                                _gl = bolt.organ_out(
                                    "groundtruth_b",
                                    "[kernel-measure] " + _mr.advisory[:220] +
                                    (f" Probe: `{_mr.probe}`" if _mr.probe else ""))
                                if _gl:
                                    obs = obs + "\n" + _gl
                                break
                except Exception:                                    # noqa: BLE001
                    pass
            obs = bolt.observe(line, obs, step=step)
            # E415: a dossier opened by this turn's receipt reaches the model NOW.
            _clp = getattr(bolt, "_closelook_pending", None)
            if _clp:
                obs = obs + "\n" + _clp
                bolt._closelook_pending = None
            # E423: same-turn delivery for a fired composed-cognition card.
            _cop = getattr(bolt, "_composed_pending", None)
            if _cop:
                obs = obs + "\n" + _cop
                bolt._composed_pending = None
            # E428: pivot advisory (next unvisited edge / loop-ban) same-turn.
            _pvp = getattr(bolt, "_pivot_pending", None)
            if _pvp:
                obs = obs + "\n" + _pvp
                bolt._pivot_pending = None
            # E525: regression tripwire notices, same-turn.
            _obp = getattr(bolt, "_obreg_pending", None)
            if _obp:
                obs = obs + "\n" + _obp
                bolt._obreg_pending = None
            # E491: retry-gate warning/refusal, same-turn.
            _rtp = getattr(bolt, "_retry_pending", None)
            if _rtp:
                obs = obs + "\n" + _rtp
                bolt._retry_pending = None
            # E470: dirty-belief notices from the invalidation cascade, same-turn.
            _edit_verb = line.split(maxsplit=1)[0] if line.strip() else ""
            _ivp = bolt.take_invalidation_notice(_edit_verb)
            if _ivp:
                obs = obs + "\n" + _ivp
            # E429: settle open advice rows against this action; deliver any fresh
            # SECOND NOTICE (2-strike cap lives in the ledger, never a third).
            _adl = getattr(bolt, "advice", None)
            if _adl is not None:
                try:
                    _act = {"intent": line, "turn": step,
                            "workspace_revision": getattr(bolt, "workspace_revision", 0) or 0}
                    _adl.observe_action(step, _act)
                    _adl.tick(step, world_revision=getattr(bolt, "workspace_revision", 0) or 0)
                    for _fp, _msg in getattr(_adl, "_escalation_messages", {}).items():
                        if _fp not in bolt._advice_escalated_seen:
                            bolt._advice_escalated_seen.add(_fp)
                            _nl = bolt.organ_out("advice_ledger", _msg)
                            if _nl:
                                obs = obs + "\n" + _nl
                except Exception:                                    # noqa: BLE001
                    pass

            # E306 — the two advisory nudges wiring audit called for. Bounded (one per
            # generation/streak in the kernel), advisory (append to the OBSERVATION, never
            # block), and structural (green-after-edit / edits-without-verification).
            if line.startswith("test") and bolt.last_test_exit == 0:
                edits_since_test = 0
                bolt._last_green_step = step
                bolt._last_green_revision = None
                try:
                    from kernel_remaining_sites import (
                        current_workspace_revision,
                        green_nag_enabled,
                    )
                    if green_nag_enabled():
                        bolt._last_green_revision = current_workspace_revision(
                            _green_nag_runtime)
                except Exception:  # noqa: BLE001 - advisory revision capture is fail-closed
                    pass
                _fn_nudge = bolt.finish_nudge(_progress_facts().edits)
                if _fn_nudge:
                    obs = obs + "\n" + _fn_nudge
            elif line.startswith("test"):
                edits_since_test = 0
                bolt._last_green_revision = None
                # E391: a fresh RED re-arms the post-green freeze (red->green->edit
                # cycles each get their own one challenge).
                bolt._green_gen = getattr(bolt, "_green_gen", 0) + 1
                if bolt.last_test_exit == 0:
                    bolt._last_green_step = step
                # On a failing test, extract contract atoms (order/bounds/
                # polarity/window/default) from the failing output + test source and
                # confront: "the test demands X, your code does Y at <span>". Bounded
                # (<=3 atoms, fingerprint-deduped inside the organ), advisory.
                _red_target = _red_confrontation_target(fn, d["rel"])
                if (getattr(bolt, "nit", None) is not None
                        and bolt.last_test_exit not in (0, None)
                        and _red_target is not None):
                    try:
                        from kernel_nitpicker import NitEvidence, TestRunEvidence
                        _tsel = line.split(None, 1)[1].strip() if " " in line else ""
                        _tpath0 = _tsel.split("::", 1)[0]
                        _tfile = root / _tpath0 if _tpath0.endswith(".py") else None
                        _nev = NitEvidence(
                            task_id=d["sha"][:8], target_file=_red_target[0],
                            target_symbol=_red_target[1],
                            target_source=(target.read_text() if target.exists() else ""),
                            test_run=TestRunEvidence(
                                output=obs, exit_code=bolt.last_test_exit or 1,
                                test_path=_tpath0, test_line=1, nodeid=_tsel),
                            test_source=(_tfile.read_text()
                                         if _tfile is not None and _tfile.exists() else ""),
                            test_path=_tpath0)
                        _conf = bolt.nit.observe_red(_nev)
                        _nline = _conf.render_red() if _conf is not None else None
                        if _nline:
                            _nline = bolt.organ_out("nitpicker", _nline)
                        if _nline:
                            obs = obs + "\n" + _nline
                            bolt._rec("nitpicker", "red_confrontation",
                                      f"atoms={_conf.count}")
                        else:
                            # E393 snitch: a silent no-fire must be distinguishable
                            # from a broken organ (#181c) — record WHY nothing rendered.
                            bolt._rec("nitpicker", "red_no_atoms",
                                      f"conf={'none' if _conf is None else _conf.count}")
                    except Exception as _e:                          # noqa: BLE001
                        print(f"    [kernel-wiring] nitpicker red-confront failed: "
                              f"{type(_e).__name__}: {_e}")
            elif line.split()[0] == "edit" and not _edit_observation_is_rejection(obs):
                edits_since_test += 1
                _last_green_step = getattr(bolt, "_last_green_step", None)
                _green_preceded_edit = (
                    bolt.last_test_exit == 0
                    and type(_last_green_step) is int
                    and _last_green_step < step
                )
                _gf_nudge = bolt.green_freeze_nudge(
                    green_before_edit=_green_preceded_edit,
                )
                _green_nag_opted_in = (
                    os.environ.get("STATEFRAME_GREEN_NAG_SITES", "off").strip().lower() == "on"
                )
                _green_nag_gate_available = False
                try:
                    from kernel_remaining_sites import (
                        decide_green_nag,
                        green_nag_enabled,
                        record_green_nag_decision,
                    )
                    _green_nag_opted_in = green_nag_enabled()
                    _green_nag_gate_available = True
                except Exception:  # noqa: BLE001 - opt-in uncertainty is handled below
                    pass
                if _green_nag_opted_in and not _green_nag_gate_available:
                    _pending_postgreen_nudge = None
                elif not _green_nag_opted_in:
                    _pending_postgreen_nudge = _gf_nudge
                else:
                    try:
                        _green_nag_decision = decide_green_nag(
                            _gf_nudge,
                            _green_nag_runtime,
                            getattr(bolt, "_last_green_revision", None),
                        )
                        record_green_nag_decision(bolt, _green_nag_decision)
                        _pending_postgreen_nudge = (
                            None if _green_nag_decision.inline
                            else _green_nag_decision.message
                        )
                        if _green_nag_decision.inline and _green_nag_decision.message:
                            obs = obs + "\n" + _green_nag_decision.message
                    except Exception:  # noqa: BLE001 - opted-in uncertainty must not nag
                        _pending_postgreen_nudge = None
                _ch_nudge = bolt.churn_nudge(_progress_facts().edits_since_check)
                if _ch_nudge:
                    obs = obs + "\n" + _ch_nudge

            # CYLINDER 4 was ENGAGED AND NEVER FIRED — `bolt.note()` existed only in the
            # selftest, so the hypothesis graph reported "engaged" on every run while doing
            # nothing. That is the exact dead-module state the report was built to expose, and
            # Review identified the missing input. Feed it the two things the model
            # cannot carry across 36 steps: the reasoning it states before acting, and every
            # edit outcome. This is memory ON BEHALF of the agent, not a decision for it.
            _prose = next((line.strip() for line in reply.strip().splitlines()
                           if len(line.strip()) > 40
                           and not line.strip().startswith(("`", "#"))), "")
            if _prose:
                bolt.note("hypothesis", _prose)
            if line.startswith("edit ") and obs.startswith("edit applied"):
                bolt.note("evidence", f"patched {line[5:].strip()} at step {step + 1}")
            # KEEP bolt.edits IN SYNC EVERY STEP, not only at `done`.
            # A run that exhausts its budget without ever issuing `done` reported ZERO edits in
            # the kernel report, so the graveyard read would show a task that patched five
            # times as one that never acted — exactly the wrong conclusion, and the same shape
            # as the edits=0 story that sent me chasing model behaviour for an hour.
            _turn_progress = _progress_facts()
            bolt.edits = _turn_progress.edits
            if _host_truth_progress_enabled:
                bolt.product_edits = _turn_progress.product_edits

            # E350: one EKG check per turn — a degraded trajectory earns ONE bounded
            # advisory (kernel_vitals owns thresholds/escalation; shadowable organ).
            if getattr(bolt, "ekg", None) is not None:
                try:
                    _ekg_line = bolt.ekg.check(organ_out=bolt.organ_out)
                    if _ekg_line:
                        obs = obs + "\n" + _ekg_line
                except Exception as _e:                              # noqa: BLE001
                    print(f"    [kernel-wiring] ekg.check failed: {type(_e).__name__}")

            if not interventions:
                obs = plain_observation
            if line.startswith("test") and bolt.last_test_exit == 0:
                obs = measurement.green_lock.decorate_test_result(obs)
                if getattr(measurement, "stateframe_first_green_disclosure_mode", "off") == "on":
                    from kernel_first_green_disclosure import decorate_test_result
                    obs = decorate_test_result(measurement, obs)
            # An ACTION (edit/test) advances the anchor: cycle-surgery only ever excises the
            # query chatter AFTER the last action, so no edit or test feedback is ever lost.
            # E308 — TEST-EDIT-AFTER-RED warning: after a red
            # run the 4B edited the TEST fixture to match its code, then trusted a
            # "1 passed, 8 deselected" green. Grading restores pristine test files, so a
            # test edit can never make the task pass — state that FACT once, bounded.
            # (Mirrors the real-world rule: never fix the test to fix the bug.)
            if (interventions and line.split()[0] == "edit" and ("tests/" in line or "/test_" in line)
                    and bolt.last_test_exit in (1, 2, 3)
                    and "applied" in obs.split("\n")[0]
                    and not bolt._challenged.get("test_edit_after_red")):
                bolt._challenged["test_edit_after_red"] = True
                bolt._rec("skeptic", "test_edit_after_red", line[:60])
                obs = (obs + "\n[kernel] You edited a TEST file after a red run. Grading "
                       "restores the ORIGINAL test files, so changing tests cannot make "
                       "this task pass — the PRODUCT code must satisfy the original test's "
                       "contract. Your test edit stays for local exploration only.")
            if line.startswith("edit "):
                _typed_observation = _canonical_typed_edit_rejection(obs)
                if _typed_observation is not None:
                    obs = _typed_observation
            # Soak C4 anchor recovery (default off): note on a rejected edit result.
            if _edit_verb in {"edit", "str_replace", "file_create", "insert_before"}:
                from kernel_anchor_recovery import annotate_rejection as _anchor_note
                obs = _anchor_note(obs, rejected=(
                    _edit_observation_is_rejection(obs)
                    or (_edit_verb == "str_replace"
                        and locals().get("_batch_status") not in (None, "APPLIED"))))
            from kernel_read_pointer import (
                INLINE_FINAL_INSTRUCTION as _read_pointer_inline_instruction,
                INLINE_MARKER as _read_pointer_inline_marker,
                all_task_definitions_read as _all_read_pointer_definitions_read,
                edit_menu_should_arm as _read_pointer_edit_menu_should_arm,
                observe_tool_result as _read_pointer_note,
            )
            _read_pointer_task_text = (
                d.get("_bridge_ordinary_brief") or d.get("msg")
                or d.get("symptom") or ""
            )
            obs = _read_pointer_note(
                measurement, obs, root=root,
                task_text=_read_pointer_task_text,
                path_allowed=lambda candidate: information.check_path(root, str(candidate)),
                read_path=_pointer_read_path,
                read_delivery=_pointer_read_delivery,
            )
            _read_pointer_inline_delivered = (
                _read_pointer_inline_marker in obs
                and _read_pointer_inline_instruction in obs
            )
            _read_pointer_definitions_read = _all_read_pointer_definitions_read(
                measurement, _read_pointer_task_text
            )
            if _read_pointer_edit_menu_should_arm(
                    inline_delivered=_read_pointer_inline_delivered,
                    source_definitions_read=_read_pointer_definitions_read):
                from kernel_rejection_template_restrict import arm_read_pointer_menu
                _active_capture = getattr(measurement, "_active_capture", None)
                arm_read_pointer_menu(
                    measurement,
                    source_request_id=getattr(_active_capture, "request_id", None),
                )
            if _edit_verb == "edit":
                _whole_edit_rejected = _edit_observation_is_rejection(obs)
                def _rejection_path_allowed(candidate: pathlib.Path) -> bool:
                    return (
                        information.check_path(root, str(candidate))
                        and not protected_test_path(candidate, baseline_tests)
                    )
                from kernel_rejection_template import (
                    annotate_rejection as _rejection_template_note,
                )
                from kernel_rejection_template_restrict import (
                    arm_on_rejected_edit,
                    rejected_definition_edit_eligible,
                )
                _restricted_definition_rejected = rejected_definition_edit_eligible(
                    rejected=_whole_edit_rejected,
                    root=root,
                    path=_rejection_template_path,
                    proposed=_rejection_template_source,
                    path_allowed=_rejection_path_allowed,
                )
                obs = _rejection_template_note(
                    obs,
                    rejected=_whole_edit_rejected,
                    root=root,
                    path=_rejection_template_path,
                    proposed=_rejection_template_source,
                    path_allowed=_rejection_path_allowed,
                )
                _active_capture = getattr(measurement, "_active_capture", None)
                arm_on_rejected_edit(
                    measurement,
                    rejected=_restricted_definition_rejected,
                    source_request_id=getattr(_active_capture, "request_id", None),
                )
            if (_name == "str_replace" and tool_batch_args is None
                    and "old_str not found" in obs):
                _replace_args = locals().get("tool_str_args")
                _replace_action = locals().get("_args")
                if (isinstance(_replace_args, tuple) and len(_replace_args) == 2
                        and isinstance(_replace_args[0], str) and _replace_args[0]
                        and isinstance(_replace_action, Mapping)):
                    from kernel_edit_rejection_fresh_text import annotate_rejection
                    obs = annotate_rejection(
                        obs,
                        root=root,
                        path=str(_replace_action.get("path") or ""),
                        old_text=_replace_args[0],
                    )
            # Fallthrough dispatch branches deliver once here. Only branches
            # that exit early may observe/append their own result before exiting.
            if line.split() and line.split()[0] == "test":
                from kernel_wire_identity import public_wire_text
                obs = public_wire_text(obs, measurement.journal)
            measurement.observe(line, obs)
            delivered_obs = obs
            if line.split() and line.split()[0] == "test":
                from kernel_stateframe_framefit5 import has_framefit5_state
                if has_framefit5_state(msgs):
                    from kernel_frame_tool_result_budget import bound_frame_tool_result
                    delivered_obs = bound_frame_tool_result(obs, msgs, kind="test")
            # Read/search helpers have already applied their shared bounded delivery
            # contract.  A second unconditional character slice here made receipts and
            # model-visible bytes disagree, especially for long lines and later pages.
            msgs.append(_tool_reply_for_call(msgs, tool_call_id, delivered_obs) if tool_call_id
                        else {"role": "user", "content": delivered_obs})
            _postgreen_advisory = locals().get("_pending_postgreen_nudge")
            if isinstance(_postgreen_advisory, str) and _postgreen_advisory:
                msgs.append({"role": "user", "content": _postgreen_advisory})
                _pending_postgreen_nudge = None
            if line.split()[0] in ("edit", "test"):
                action_anchor = len(msgs)
            # E307 completion_evidence input: distinguish PRODUCT edits from test edits so
            # the done-gate can refuse "changed only tests" completion claims.
            _product_edit_applied = (
                line.split()[0] == "edit"
                and "applied" in obs.split("\n")[0]
                and "tests/" not in line and "/test_" not in line
                and not line.split()[-1].startswith("test_"))
            _batch_product_mutation_applied = False
            if _edit_verb == "str_replace" and tool_batch_args is not None:
                from kernel_rejection_template_restrict import (
                    applied_product_batch_mutation as _applied_product_batch_mutation,
                )
                _batch_result = locals().get("_batch_result")
                _batch_product_mutation_applied = _applied_product_batch_mutation(
                    locals().get("_batch_status"),
                    getattr(_batch_result, "changed_paths", ()),
                )
            _single_product_str_replace_applied = (
                _edit_verb == "str_replace"
                and tool_batch_args is None
                and "applied." in obs.split("\n")[0]
            )
            if _product_edit_applied:
                bolt.product_edits += 1
            _batch_result_for_noread = locals().get("_batch_result")
            _active_capture = getattr(measurement, "_active_capture", None)
            _release_noread_after_applied_mutation(
                measurement,
                request_id=getattr(_active_capture, "request_id", None),
                tool_name=_edit_verb,
                observation=obs,
                batch_requested=tool_batch_args is not None,
                batch_status=locals().get("_batch_status"),
                batch_changed=bool(getattr(
                    _batch_result_for_noread, "changed_paths", ())),
                single_str_replace_applied=_single_product_str_replace_applied,
            )
            if (_product_edit_applied or _batch_product_mutation_applied
                    or _single_product_str_replace_applied):
                from kernel_rejection_template_restrict import (
                    release_after_product_mutation as _release_rejection_template_restriction,
                )
                _active_capture = getattr(measurement, "_active_capture", None)
                _current_action = getattr(measurement, "current_action", None)
                _product_tool_name = (
                    str(_current_action.get("kind") or "")
                    if isinstance(_current_action, Mapping) else ""
                ) or _edit_verb
                _release_rejection_template_restriction(
                    measurement,
                    request_id=getattr(_active_capture, "request_id", None),
                    tool_name=_product_tool_name,
                )
            _edit_nturn_after_action(
                line, obs, step,
                model_action_args=(
                    _args if toolcall_mode and isinstance(_args, Mapping) else None
                ),
            )
            if (line.startswith("test") and bolt.last_test_exit == 0
                    and getattr(measurement, "portfolio", None) is not None):
                measurement.portfolio_preserve_candidate(
                    public_green=True,
                    public_suite=tuple(getattr(measurement, "test_receipts", ())),
                    reason=f"public-green:step-{step}")

            # Flush each step with a task tag so concurrent activity can be inspected
            # promptly and parser failures remain visible.
            with (POC / "results" / f"react_live{os.environ.get('REACT_TAG', '')}.log").open("a") as live:
                live.write(f"[{fn[:26]:28s} s{step + 1:02d}] {line[:70]:72s} "
                           f"-> {obs[:90].splitlines()[0] if obs.strip() else '(empty)'}\n")
                live.flush()

        _finish_edit_nturn()
        if (_budget_mode == "response" and _budget.tool_call_cap is not None
                and _budget.tool_calls_used >= _budget.tool_call_cap
                and measurement.system_outcome is None
                and not done_accepted(measurement)):
            # The range() boundary can be reached immediately after the last
            # allowed call, so there is no next iteration to report the cap.
            _budget.can_start_tool_call()
            measurement.system_outcome = "tool_call_cap"
        if getattr(measurement, "system_outcome", None) == "env_broken":
            _reported_edits, _reported_checks = _reported_progress_counts(locals())
            outcome = {"fn": fn, "result": "ENV_BROKEN", "steps": 0,
                       "edits": _reported_edits, "tests": _reported_checks,
                       "behaviour": "unverified"}
            finish_native_batch_transcript(msgs)
            measurement.finish(outcome, messages=msgs)
            return _attach_env_receipt(outcome, measurement)

        # E451: the model can delete/move the target file mid-run (jq3 did) — a
        # missing file at grade time is a FAILED attempt, never a crashed row.
        source_map_path = None
        target_gone_paths = (_missing_declared_paths(
            root, fileset, journal=measurement.journal,
        ) if multifile else ())
        if multifile and target_gone_paths:
            _gone_snapshot = _source_snapshot(root, fileset.paths)
            _target_gone = _target_gone_lines(
                target_gone_paths, root=root, journal=measurement.journal,
            )
            bolt.receipt("gate", "target_gone", _target_gone,
                         "TARGET_GONE", _target_gone)
            _annotate_receipt(
                bolt, paths=fileset.paths, scope=target_gone_paths,
                before=_gone_snapshot, after=_gone_snapshot,
                selectors=tuple(getattr(bolt, "f2p_ids", ())),
                failure_tail=_target_gone)
        # An explicit closure reserve must fail honestly when the hard cap is
        # reached before the verification+done hand-off.  Keep the candidate
        # tree and emit a typed outcome; do not turn an affordable artifact
        # into a crash or a silent incomplete row.
        if (getattr(measurement, "closure_reserve", 0)
                and not done_accepted(measurement)
                and measurement.system_outcome is None
                and ((_budget_mode == "response" and _budget.exhausted)
                     or (_budget_mode != "response" and reached(step_cap, step + 1)))):
            measurement.bounded_completion(
                detail=("closure reserve became unaffordable at the step cap; "
                        "the current artifact was preserved"))
        # b1426 (r2EW, benchmark): the completion reducer ACCEPTED and SEALED done (seq 2311) and set
        # completion_accepted, but this check read only the legacy v2_disposition flag, so an accepted done
        # was relabelled "step limit without accepted done" / step_budget_exhausted and the terminal
        # downgraded it to NOT_COMPLETED. Either acceptance authority counts as an accepted done.
        if (getattr(measurement, "disposition_v2", False)
                and not done_accepted(measurement)
                and measurement.system_outcome is None):
            # Exhausting the action loop without an accepted v2 done is the
            # existing runner's non-submission shape, made explicit.
            from kernel_terminal_policy import enabled as hard_budget_only_enabled
            measurement.system_outcome = (
                "run_incomplete" if step_cap is None else
                "step_budget_exhausted" if hard_budget_only_enabled(measurement)
                else "incomplete")
            measurement.journal.emit("v2_terminal", terminal="INCOMPLETE",
                                     reason=("action loop ended without accepted done" if step_cap is None
                                             else "step limit without accepted done"))
        measurement.flush_public_state(msgs, _reported_steps())
        if (action_schema_mode or contract_enabled()
                or measurement.continuation is not None):
            # Terminal evaluation is separate from the solver's public evidence.
            from kernel_pilot import ACTIVE_PARENT, ACTIVE_RUNTIME
            ACTIVE_PARENT.set(None)
            ACTIVE_RUNTIME.set(None)
        if d.get("_bridge_capture"):
            from kernel_bridge_evidence import capture_candidate
            # Candidate files are indexed by LOOP step inside the loop (line ~3832). In budget-unit=response mode
            # _reported_steps() counts a different quantity and could reuse an already-captured index with
            # different contents (r35d full smoke: "captured twice with different contents"). The grader takes the
            # highest index as final, so the final capture must be strictly past every loop capture.
            from kernel_bridge_evidence import final_capture_index
            try:
                capture_candidate(d["_bridge_capture"], root, d,
                                  final_capture_index(locals().get("step", -1), _reported_steps()))
            except Exception as capture_error:  # noqa: BLE001 - the immutable done store below is authoritative
                from kernel_done_is_submit_authority import submission_pending
                if not submission_pending(measurement):
                    raise
                measurement.journal.emit(
                    "done_auxiliary_capture_failed",
                    producer="kernel_done_is_submit_authority",
                    error_type=type(capture_error).__name__,
                    dispatch=False,
                )
        if continuation_workspace is not None:
            if target is None:
                raise ValueError("continuation requires a declared public working path")
            _execution_root = root
            _target_relative = target.relative_to(root)
            root = continuation_workspace.grading_copy(root)
            target = root / _target_relative
            measurement.journal.emit("continuation_grading_isolated",
                                     execution_root=str(_execution_root), grading_root=str(root))
        if multifile:
            # The initial FileSet remains the public contract, but the final
            # candidate is the product tree as it exists now.  This registers
            # file_create (and equivalent in-tree writes) without admitting
            # tests, probes, scratch, or cache artifacts.
            final_paths = _product_source_paths(root, fileset, d)
            final_sources = _source_snapshot(root, final_paths)
            primary_source = final_sources.get(fileset.primary_path)
            final = (fn_src(primary_source, fn)
                     if isinstance(primary_source, str) else "") or ""
            ast_by_file = _ast_by_file(final_sources, fileset.fixed)
            ok = (not target_gone_paths
                  and all(ast_by_file[path]["match"] for path in fileset.paths
                          if path in ast_by_file))
            # E498: multi-file consumers receive the complete final source map, not a
            # misleading one-file snapshot.
            try:
                _task_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                                    str(d.get("task_id") or fn))[:120]
                source_map_path = _persist_source_map(
                    POC / "results" / f"react_final{os.environ.get('REACT_TAG', '')}"
                    / _task_slug,
                    final_sources)
            except Exception:                                            # noqa: BLE001
                pass
        else:
            final_sources = {}
            ast_by_file = {}
            if target.exists():
                _final_target_relative = target.relative_to(root).as_posix()
                final_sources = _source_snapshot(root, (_final_target_relative,))
                final = fn_src(final_sources.get(_final_target_relative) or "", fn)
            else:
                final = ""
            final = final or ""
            # E498: persist the FINAL target source verbatim (results/react_final<TAG>/<path>)
            # — the phase runner snapshots passing artifacts from this, never by replaying
            # transcript edits (which silently turned a real PASS into FAILED).
            try:
                if target.exists():
                    # E604: seed2-derived rows carry `rel` but no `path`; keying on `path`
                    # alone silently persisted NOTHING for that whole corpus (the KeyError was
                    # swallowed below). Fall back to `rel`, and never skip silently.
                    _rel_key = d.get("path") or d.get("rel")
                    if not _rel_key:
                        raise KeyError("row has neither path nor rel")
                    _ff = (POC / "results" /
                           f"react_final{os.environ.get('REACT_TAG', '')}" / _rel_key)
                    _ff.parent.mkdir(parents=True, exist_ok=True)
                    _ff.write_text(target.read_text(errors="ignore"))
            except Exception as _e:                                      # noqa: BLE001
                print(f"    [kernel-wiring] final-source persist SKIPPED: "
                      f"{type(_e).__name__}: {_e}", flush=True)
            nf, ng = norm(final), norm(gold)
            ok = nf is not None and nf == ng

        from kernel_done_is_submit_authority import submission_pending
        if submission_pending(measurement):
            from kernel_done_is_submit_authority import submit_done
            try:
                current_sources = {
                    str(path): source for path, source in final_sources.items()
                    if isinstance(path, str) and (isinstance(source, str) or source is None)
                }
                if multifile:
                    parent_sources = dict(fileset.parent)
                else:
                    parent_path = target.relative_to(root).as_posix()
                    parent_sources = {parent_path: str(d.get("parent_src") or "")}
                intent = getattr(measurement, "_done_submit_pending", None)
                submission_source = getattr(intent, "source", "model")
                submit_done(
                    measurement,
                    source=submission_source,
                    destination=POC / "results" / "react_done_submissions",
                    task_id=d.get("task_id") or d.get("id") or fn,
                    sources=current_sources,
                    parent_sources=parent_sources,
                    out_of_scope=bool(
                        d.get("out_of_scope")
                        or getattr(measurement, "_candidate_out_of_scope", False)
                    ),
                    certification="not_evaluated",
                )
                behaviour = "not_evaluated"
            except Exception as capture_error:  # any capture/storage failure is the sole done error
                measurement._done_capture_error_class = "DONE_CAPTURE_ERROR"
                measurement._done_capture_error_type = type(capture_error).__name__
                measurement._done_submit_pending = None
                measurement.system_outcome = "error"
                measurement.journal.emit(
                    "done_capture_error",
                    producer="kernel_done_is_submit_authority",
                    error_type=type(capture_error).__name__,
                    dispatch=False,
                )

        # BEHAVIOURAL GRADE.
        #
        # Exact normalised-AST equality rejects 20 of 21 behaviourally-equivalent patches —
        # gold with ONE local variable renamed fails. So every score we have reported is a
        # FLOOR by an unknown margin, and a real repair written in a different shape is
        # indistinguishable from no repair at all.
        #
        # The repository ships its own tests. When AST says fail, ask the tests instead, and
        # make the question DISCRIMINATING: the test must PASS with the candidate AND FAIL
        # with the unpatched parent. A test that passes on the buggy code proves nothing about
        # the fix — it just does not cover the bug (measured: such tests exist in this corpus).
        # Both halves are required, which is what stops this from being a softer bar.
        _v2_not_submitted = (getattr(measurement, "disposition_v2", False)
                             and measurement.system_outcome in {"incomplete", "stalled"})
        if _v2_not_submitted:
            ok = False
        behaviour = "not_attempted"
        if multifile and target_gone_paths:
            behaviour = _target_gone_lines(
                target_gone_paths, root=root, journal=measurement.journal,
            )
        # E598: ALWAYS grade, including exact-AST matches. The old `not ok and` skipped the
        # behavioural grade whenever the target function matched gold, leaving accepted wins
        # labelled not_attempted (8 of them in one night) and blind to collateral damage
        # elsewhere in the file (fn_src compares the function only — E557's gettext_noop).
        # An AST match stays a diagnostic in `result`; `behaviour` is the only verdict.
        from kernel_done_is_submit_authority import done_sealed
        if (final and not target_gone_paths and not _v2_not_submitted
                and done_sealed(measurement) is None
                and not getattr(measurement, "_done_capture_error_class", None)):
            try:
                # GRADE AGAINST PRISTINE TESTS (anti-gaming control: an assertion-
                # weakening patch must be REJECTED). The agent can legitimately edit test
                # files while investigating in its sandbox — but grading against ITS version
                # of the tests means "gut the assert" counts as resolution, which is the exact
                # test-passing behavior that bypasses the contract. Restore every test file from
                # the real repo before the graded run; only the agent's PRODUCT-code edits
                # remain. (tests/ is restored wholesale; the target under repair is not in
                # tests/ on any bench task.)
                # SWE-bench-style grading world: the agent worked in the parent era (fix-added
                # tests absent — no leak possible); grading overlays the FIX COMMIT'S test
                # files as the hidden fail-to-pass oracle. This also IS the anti-weakening
                # control: overlaying replaces any test the agent edited. (The earlier version
                # restored TODAY'S tests, which are era-incoherent with the parent sandbox.)
                _sel = None
                if ((d.get("repo") == "mc" and d.get("sha"))
                        or d.get("repo") == "build"
                        or (d.get("source_dataset") and d.get("base_commit"))):
                    if d.get("repo") == "build" and not multifile:
                        _materialize_build_test(root, d, fn)
                    else:
                        from converge import overlay_fix_tests
                        overlay_fix_tests(root, _grading_row)
                    # GRADE WITH THE PREFLIGHT'S SELECTOR (E288). With explicit=None,
                    # run_tests greps for tests that CALL the function by name — minted
                    # tasks' f2p tests exercise routes/behaviour and never name the
                    # function, so grading found zero targets and returned no_test_oracle
                    # for tasks the preflight had verified ORACLE OK (measured:
                    # content_article_artifact, create_attention_event). Preflight and
                    # grading must ask the SAME question.
                    _ids = _public_check_ids(d, fn)
                    _sel = tuple(_ids) if multifile else (_ids[0] if _ids else None)
                # E265 + E279: the two grading legs each get a FRESH database — leg-to-leg
                # pollution (parent-control ledger rows poisoning gold) was measured; the
                # agent-loop DB is per-task and must not bleed in here either.
                _rid = os.environ.get("REACT_RUN_ID", "r0")
                os.environ["REACT_DB_NAME"] = (
                    f"mission_control_test_react_{_rid}_{fn[:18].lower()}_g")
                if multifile:
                    after, _after_codes = run_tests_for_selectors(
                        root, fn, _sel, d["rel"], row=d)
                    after_green = bool(_after_codes) and all(
                        code == 0 for code in _after_codes)
                else:
                    after = run_tests(root, fn, _sel, d["rel"], row=d)
                    after_green = "pytest exit 0" in after
                if after_green:
                    # Keep candidate-only modules out of the parent control
                    # leg, then restore the exact candidate tree afterwards.
                    # Otherwise a newly created importable product module can
                    # make the unpatched parent pass or disappear at delivery.
                    _candidate_sources = dict(final_sources)
                    _candidate_paths = set(_candidate_sources)
                    if multifile:
                        for _path, _source in fileset.parent.items():
                            _target = root / _path
                            _target.parent.mkdir(parents=True, exist_ok=True)
                            _target.write_text(_source, encoding="utf-8")
                        for _path in sorted(_candidate_paths - set(fileset.paths)):
                            if _is_product_path(_path, d):
                                _candidate = root / _path
                                if _candidate.is_file() or _candidate.is_symlink():
                                    _candidate.unlink()
                    else:
                        # keep the exact candidate bytes; the control leg
                        # previously left the grading root holding the unpatched parent,
                        # so post-finish facts/salvage saw "no product edit".
                        _single_candidate = (target.read_bytes()
                                             if target.is_file() else None)
                        target.write_text(d["parent_src"])          # unpatched control
                    try:
                        os.environ["REACT_DB_NAME"] = (
                            f"mission_control_test_react_{_rid}_{fn[:18].lower()}_c")
                        if multifile:
                            before, _before_codes = run_tests_for_selectors(
                                root, fn, _sel, d["rel"], row=d)
                            before_green = bool(_before_codes) and all(
                                code == 0 for code in _before_codes)
                        else:
                            before = run_tests(root, fn, _sel, d["rel"], row=d)
                            before_green = "pytest exit 0" in before
                    finally:
                        if multifile:
                            for _path, _source in _candidate_sources.items():
                                if isinstance(_source, str):
                                    _candidate = root / _path
                                    _candidate.parent.mkdir(parents=True, exist_ok=True)
                                    _candidate.write_text(_source, encoding="utf-8")
                        elif _single_candidate is not None:
                            target.write_bytes(_single_candidate)
                    behaviour = ("PASS_behavioural" if not before_green
                                 else "test_does_not_cover_bug")
                    # Keep the AST result as the acceptance result.  The
                    # behavioural leg is still authoritative evidence and is
                    # reported separately, but a synthetic/public oracle must
                    # not turn a nonexact candidate into an accepted patch.
                elif "pytest exit" in after:
                    behaviour = "tests_fail"
                else:
                    behaviour = "no_test_oracle"
            except Exception as e:                                   # noqa: BLE001
                behaviour = f"error:{type(e).__name__}"
        # Liveness coverage check: verify components exist, are wired, and are enabled.
        # task-end only, from recorded evidence — DARK organs (triggered, never fired,
        # never legitimately suppressed) get named instead of rotting silently.
        _liveness = None
        try:
            from kernel_liveness import Evidence, run_census
            _receipt_path = getattr(bolt, "receipts_path", None)
            _organ_outputs = tuple(
                event
                for organ in (getattr(bolt, "nit", None), getattr(bolt, "ekg", None))
                for event in getattr(organ, "trace", ()))
            _liveness = run_census(
                Evidence.from_records(
                    receipts=tuple(getattr(bolt, "_receipts", ())),
                    traces=tuple(bolt.trace),
                    organ_outputs=_organ_outputs,
                    run_id=getattr(bolt, "task_id", fn),
                    complete=_receipt_path is not None,
                    gaps=() if _receipt_path is not None
                         else ("receipt tape unavailable",)),
                run_id=getattr(bolt, "task_id", fn),
                receipt_path=_receipt_path)
            bolt.trace.append({
                "cylinder": "liveness",
                "event": ("CENSUS_COMPLETE" if _liveness.status != "ERROR"
                          else "CENSUS_ERROR"),
                "detail": f"digest={_liveness.digest} status={_liveness.status}",
                "turn": getattr(bolt, "current_step", None)})
        except Exception as _e:                                      # noqa: BLE001
            print(f"    [kernel-wiring] liveness coverage check failed: {type(_e).__name__}")
        # PERSIST THE TRANSCRIPT. The first run reported edits=0 on two tasks and I could not
        # say WHY, because only counts were recorded — the same "saved the score, lost the
        # evidence" defect banked as E240 this morning. The command/observation pairs are the
        # only way to tell "the model refused to act" from "every path it tried was invalid".
        _done_submit_fields = {}
        from kernel_done_is_submit_authority import final_result, terminal_from_seal
        _done_terminal = terminal_from_seal(measurement)
        _finish_result = final_result(measurement, "PASS" if ok else "fail")
        if _done_terminal is not None:
            _done_submit_fields = {
                "submission": _done_terminal["done_seal"],
                "labels": _done_terminal["labels"],
            }
        elif getattr(measurement, "_done_capture_error_class", None):
            _done_submit_fields = {
                "submission_error": getattr(measurement, "_done_capture_error_class"),
            }
        tdir = POC / "results" / ("react_transcripts" + os.environ.get("REACT_TAG", ""))
        tdir.mkdir(exist_ok=True)
        _multi_result = ({
            "task_id": d.get("task_id"),
            "shape": "multi_file",
            "paths": list(_product_source_paths(root, fileset, d)),
            "target_gone_paths": list(target_gone_paths),
            "ast_by_file": ast_by_file,
            "source_map_path": str(source_map_path) if source_map_path else None,
            "final_source_paths": [path for path, source in final_sources.items()
                                   if isinstance(source, str)],
        } if multifile else {})
        _finish_progress = _progress_facts()
        (tdir / f"{fn}.json").write_text(json.dumps(
            {"fn": fn, "result": _finish_result, "steps": _reported_steps(),
              "edits": _finish_progress.edits, "tests": _finish_progress.checks, "messages": msgs,
             "tag": regime["tag"], "regime": regime,
             "tool_calls": tool_calls_seen,
             "decoy_calls": len(decoy_call_names),
             "decoy_call_names": decoy_call_names,
             "sibling_confusion_calls": sum(
                 _decoy_kind == "near_duplicate" and is_near_duplicate_name(name)
                 for name in decoy_call_names
             ),
             "schema_violations": len(schema_violation_details),
             "schema_violation_details": schema_violation_details,
             "unknown_tool_calls": len(unknown_tool_names),
             "unknown_tool_names": unknown_tool_names,
             "wrong_tool_before_first_edit": wrong_tool_before_first_edit,
             # The graveyard read: which cylinders engaged, which fired, which errored. An
             # ENGAGED-BUT-NEVER-FIRED cylinder is as dead as one that failed to start, and
             # only this record tells them apart — the blindness that let 47 modules sit dark.
              "kernel": {"status": bolt.status, "trace": list(bolt.trace),
                         "repeats_blocked": bolt.repeats}, **_done_submit_fields,
              **_multi_result,
              **_memory_identity_fields(d)}, indent=1))
        with (POC / "results" / f"react_live{os.environ.get('REACT_TAG', '')}.log").open("a") as live:
            live.write(bolt.report() + "\n")
        # #41: a dialect abort with a graded WIN is still a win — the evidence outranks the
        # conversation's ending. Without a win it stays HARNESS_SUSPECT (uncharged).
        finish_native_batch_transcript(msgs)
        measurement.finish({"result": _finish_result, "steps": _reported_steps(),
                            "edits": _finish_progress.edits,
                            "tests": _finish_progress.checks,
                            "behaviour": behaviour,
                            "completion": (measurement._last_completion_decision.to_dict()
                                            if measurement._last_completion_decision is not None else None)})
        _finish_progress = _progress_facts()
        if (d.get("_bridge_capture")
                and done_accepted(measurement)
                and measurement.system_outcome is None
                and _finish_result == "PASS"):
            # 10-08: a manifest-write failure must NEVER turn a won run into an ERROR. It is journaled as a
            # host fact (final_product_manifest_failed) and the PASS stands; RUNREAL's clean-finish path then falls
            # back to the timeout-salvage export so the candidate is still graded (never silently dropped).
            try:
                from kernel_final_product_manifest import emit_stage6_clean_finish_manifest

                capture_value = d["_bridge_capture"]
                if not isinstance(capture_value, (str, os.PathLike)):
                    raise RuntimeError("clean Stage 6 finish has no candidate capture directory")
                run_dir = pathlib.Path(capture_value).resolve().parent.parent
                from tools.sciledger.final_product import FINAL_PRODUCT_MANIFEST_NAME

                _manifest_destination = run_dir / FINAL_PRODUCT_MANIFEST_NAME
                _manifest_preexisting = (_manifest_destination.exists()
                                         or _manifest_destination.is_symlink())
                task_id = str(d.get("task_id") or d.get("id") or fn)
                parent_snapshot = REPOS.get("build")
                if not isinstance(parent_snapshot, pathlib.Path):
                    raise RuntimeError("clean Stage 6 finish has no parent snapshot")
                manifest_path = emit_stage6_clean_finish_manifest(
                    clean_finish=True,
                    run_dir=run_dir,
                    parent_root=parent_snapshot,
                    final_root=root,
                    live_manifest_path=pathlib.Path("/stage/live-manifest.json"),
                    journal_path=pathlib.Path(measurement.journal.path),
                    solver_run_id=str(measurement.journal.run_id),
                    task_id=task_id,
                    candidate_dir=pathlib.Path(capture_value),
                    event_sink=lambda kind, **fields: _record_post_finish_fact(measurement, kind, **fields),
                )
                _final_product_manifest_written = manifest_path is not None
            except Exception as _manifest_error:  # noqa: BLE001 - host bookkeeping must not flip the verdict
                _final_product_manifest_write_failed = True
                if (isinstance(locals().get("_manifest_destination"), pathlib.Path)
                        and not locals().get("_manifest_preexisting", False)
                        and _manifest_destination.is_file()
                        and not _manifest_destination.is_symlink()):
                    _final_product_manifest_written = True
                _record_post_finish_fact(
                    measurement, "final_product_manifest_failed", producer="react_repair",
                    fact_owner="HOST", dispatch=False, error_type=type(_manifest_error).__name__,
                    message=str(_manifest_error)[:400], fallback="timeout_salvage_export")
        _done_facts_fields = {}
        if getattr(measurement, "stateframe_host_done_facts_mode", "off") == "on":
            _done_facts_fields = {
                "done_facts": measurement.done_facts_payload(),
                "claims": measurement.done_facts_claims(),
            }
        if measurement.system_outcome:
            return _attach_env_receipt(
                dict(fn=fn, result=measurement.system_outcome.upper(), steps=_reported_steps(),
                     edits=_finish_progress.edits, tests=_finish_progress.checks,
                     behaviour=behaviour, **_done_facts_fields,
                      **_done_submit_fields, **_multi_result), measurement)
        if (_done_terminal is None and harness_suspect and not ok
                and behaviour != "PASS_behavioural"):
            return _attach_env_receipt(
                dict(fn=fn, result="HARNESS_SUSPECT", steps=_reported_steps(),
                     edits=_finish_progress.edits, tests=_finish_progress.checks,
                     behaviour=harness_suspect, **_done_facts_fields,
                     **_done_submit_fields, **_multi_result), measurement)
        return _attach_env_receipt(
            dict(fn=fn, result=_finish_result, steps=_reported_steps(),
                 edits=_finish_progress.edits, tests=_finish_progress.checks,
                 behaviour=behaviour, **_done_facts_fields,
                 **_done_submit_fields, **_multi_result), measurement)
    except Exception as exc:
        if resume is not None and measurement is None:
            raise  # Recovery refusal must remain visible; do not finalize the original run.
        if measurement is not None:
            from kernel_done_is_submit_authority import (
                submission_pending, terminal_from_seal,
            )
            done_terminal = terminal_from_seal(measurement)
            capture_failed = bool(getattr(measurement, "_done_capture_error_class", None))
            if done_terminal is None and submission_pending(measurement):
                try:
                    _capture_done_candidate_after_exception(measurement, locals())
                    done_terminal = terminal_from_seal(measurement)
                except Exception as capture_error:
                    capture_failed = True
                    measurement._done_capture_error_class = "DONE_CAPTURE_ERROR"
                    measurement._done_capture_error_type = type(capture_error).__name__
                    measurement._done_submit_pending = None
                    measurement.system_outcome = "error"
                    measurement.journal.emit(
                        "done_capture_error",
                        producer="kernel_done_is_submit_authority",
                        error_type=type(capture_error).__name__,
                        dispatch=False,
                    )
            if done_terminal is not None or capture_failed:
                progress = _reported_progress_counts(locals())
                report_steps = locals().get("_reported_steps")
                if done_terminal is not None:
                    outcome = {
                        "fn": fn,
                        "result": "SUBMITTED",
                        "steps": report_steps() if callable(report_steps) else locals().get("step", -1),
                        "edits": progress[0],
                        "tests": progress[1],
                        "behaviour": "SUBMITTED",
                        "accepted": True,
                        "submission": done_terminal.get("done_seal"),
                        "labels": done_terminal.get("labels", {}),
                    }
                    measurement.system_outcome = "submitted"
                    measurement.journal.emit(
                        "done_submission_recovered_after_exception",
                        producer="kernel_done_is_submit_authority",
                        error_type=type(exc).__name__,
                        artifact_digest=done_terminal.get("artifact_digest"),
                        dispatch=False,
                    )
                else:
                    outcome = {
                        "fn": fn,
                        "result": "ERROR",
                        "steps": locals().get("step", -1),
                        "edits": progress[0],
                        "tests": progress[1],
                        "behaviour": "unverified",
                        "error_class": "DONE_CAPTURE_ERROR",
                        "error_type": getattr(
                            measurement, "_done_capture_error_type", type(exc).__name__),
                    }
                    measurement.system_outcome = "error"
                try:
                    finish_native_batch_transcript(locals().get("msgs"))
                except Exception as transcript_error:
                    measurement.journal.emit(
                        "done_submission_transcript_error",
                        error_type=type(transcript_error).__name__,
                        dispatch=False,
                    )
                try:
                    measurement.finish(outcome, messages=locals().get("msgs"))
                except Exception as finish_error:
                    measurement.journal.emit(
                        "done_submission_finish_error",
                        error_type=type(finish_error).__name__,
                        dispatch=False,
                    )
                return _attach_env_receipt(outcome, measurement)
            try:
                _append_host_refusal_observation(
                    measurement, getattr(bolt, "_receipts", ()), msgs)
            except Exception:                                      # noqa: BLE001
                pass  # A recovery card must never replace the original terminal error.
        _harness_outcome = ("HARNESS_CONTRACT" if getattr(exc, "harness_contract", False)
                            else getattr(exc, "outcome", None))
        if (not getattr(exc, "boundary_recovery_stop", False)
                and _harness_outcome in _HARNESS_TERMINAL_OUTCOMES):
            _location = _exception_location(exc)
            print(_location["traceback_tail"], file=sys.stderr, flush=True)
            outcome = dict(
                fn=fn, result=_harness_outcome, steps=locals().get("step", -1),
                edits=_reported_progress_counts(locals())[0],
                tests=_reported_progress_counts(locals())[1],
                behaviour="unverified", error_type=type(exc).__name__,
                failure_origin="harness", **_location)
            if measurement is not None:
                measurement.system_outcome = _harness_outcome
                measurement.journal.emit(
                    "worker_exception", error_type=type(exc).__name__,
                    outcome=_harness_outcome, failure_origin="harness",
                    step=locals().get("step", -1), **_location)
                measurement.finish(outcome, messages=locals().get("msgs"))
            return _attach_env_receipt(outcome, measurement)
        from kernel_pilot import ParentBudgetExhausted
        _reported_steps_fn = locals().get("_reported_steps")
        from kernel_bridge_evidence import final_capture_index
        _exception_steps = final_capture_index(
            locals().get("step", -1), _reported_steps_fn() if callable(_reported_steps_fn) else -1)
        if measurement is not None and d.get('_bridge_capture'):
            try:
                from kernel_bridge_evidence import capture_candidate
                capture_candidate(d['_bridge_capture'], root, d, _exception_steps)
            except Exception as capture_error:
                measurement.journal.emit('terminal_candidate_capture_failed',
                                         error_type=type(capture_error).__name__)
        _reported_edits, _reported_checks = _reported_progress_counts(locals())
        if getattr(exc, "boundary_recovery_stop", False):
            outcome = dict(
                fn=fn, result="RUN_INCOMPLETE", reason_code="boundary_unrecoverable",
                steps=_exception_steps,
                edits=_reported_edits, tests=_reported_checks, behaviour="unverified",
            )
            _messages = locals().get("msgs")
            return _finish_boundary_recovery_terminal(
                measurement, outcome, _messages,
                finish_transcript=finish_native_batch_transcript,
                attach=_attach_env_receipt,
            )
        if not isinstance(exc, ParentBudgetExhausted):
            if os.environ.get('REACT_STEP_CONTRACT', '0') != '1':
                # Batch 834: this re-raise left NO journal row while finish() closed the
                # in-flight step with an empty observation.  Record the cause first.
                if measurement is not None:
                    try:
                        measurement.journal.emit(
                            'run_exception', error_type=type(exc).__name__,
                            error_detail=str(exc)[:600], step=locals().get('step', -1),
                            dispatch=False)
                    except Exception:                                # noqa: BLE001
                        pass
                _final_product_host_crash = True
                raise
            outcome = dict(fn=fn, result='HARNESS_ERROR', steps=_exception_steps,
                           edits=_reported_edits, tests=_reported_checks,
                           behaviour='unverified', error_type=type(exc).__name__)
            if measurement is not None:
                measurement.system_outcome = measurement.system_outcome or 'harness_error'
                measurement.journal.emit('worker_exception', error_type=type(exc).__name__,
                                         step=locals().get('step', -1))
                _messages = locals().get('msgs')
                finish_native_batch_transcript(_messages)
                measurement.finish(outcome, messages=_messages)
            return _attach_env_receipt(outcome, measurement)
        outcome = dict(fn=fn, result="ATTEMPT_BUDGET_EXHAUSTED",
                       steps=_exception_steps,
                       edits=_reported_edits, tests=_reported_checks,
                       behaviour="unverified", budget_category=str(exc))
        if measurement is not None:
            measurement.system_outcome = "attempt_budget_exhausted"
            _messages = locals().get('msgs')
            finish_native_batch_transcript(_messages)
            measurement.finish(outcome, messages=_messages)
        return _attach_env_receipt(outcome, measurement)
    finally:
        try:
            _messages = locals().get('msgs')
            try:
                finish_native_batch_transcript(_messages)
                if measurement is not None:
                    if 'system_prompt_audited' in locals():
                        _audit_system_prompt()
                    measurement.finish({"result": "terminated_without_final_grade"}, messages=_messages)
            finally:
                if measurement is not None:
                    _active_exception = sys.exc_info()[0]
                    try:
                        _finish_record = getattr(measurement, "_last_finish_result", None)
                        _terminal_result = (_finish_record.get("result")
                                            if isinstance(_finish_record, Mapping) else None)
                        if not isinstance(_terminal_result, str):
                            _terminal_result = locals().get("_finish_result")
                        if not isinstance(_terminal_result, str):
                            _terminal_result = str(getattr(measurement, "system_outcome", None) or "unknown")
                        _final_sources = locals().get("final_sources")
                        _changed_files = _product_files_changed_count(
                            locals().get("root"),
                            locals().get("fileset"),
                            locals().get("d", {}),
                            target=locals().get("target"),
                            current_sources=(_final_sources if isinstance(_final_sources, Mapping)
                                             else None),
                            initial_sources=locals().get("_initial_product_sources"),
                        )
                    except Exception:  # noqa: BLE001 - outcome recording must survive finish bookkeeping errors
                        _terminal_result = str(locals().get("_finish_result") or "HOST_ERROR")
                        _changed_files = 0
                        _active_exception = _active_exception or RuntimeError("terminal bookkeeping failed")
                    _host_crash = bool(locals().get("_final_product_host_crash", False))
                    if _active_exception is not None:
                        _host_crash = True
                    if _is_final_product_host_crash_result(_terminal_result):
                        _host_crash = True
                    try:
                        _record_final_product_no_candidate(
                            measurement,
                            result=_terminal_result,
                            product_files_changed=_changed_files,
                            manifest_written=bool(locals().get("_final_product_manifest_written", False)),
                            manifest_write_failed=bool(
                                locals().get("_final_product_manifest_write_failed", False)),
                            host_crash=_host_crash,
                        )
                    except Exception as _terminal_fact_error:  # noqa: BLE001 - never replace the worker outcome
                        _record_post_finish_fact(
                            measurement, "terminal_outcome_recording_failed",
                            producer="react_repair", fact_owner="HOST", dispatch=False,
                            error_type=type(_terminal_fact_error).__name__,
                        )
        finally:
            if continuation_workspace is None:
                shutil.rmtree(sbox, ignore_errors=True)
            else:
                try:
                    if resume is None or measurement is not None:
                        continuation_workspace.record_stop(measurement.journal if measurement is not None else None)
                finally:
                    continuation_workspace.close()


def main() -> int:
    if not KEY:
        print("DEEPINFRA_API_KEY unset — refusing to report a vacuous zero.")
        return 2
    tasks = [json.loads(line) for line in CORPUS.read_text().splitlines()
             if line.strip()][:N]

    # WINNABILITY PREFLIGHT (FINDINGS E256). Four of 21 tasks were unwinnable BY CONSTRUCTION
    # — the shipped patch itself could not grade as PASS — and they silently capped the bench
    # at 17 while every score was reported out of 21. That is a corrupted DENOMINATOR, which
    # is worse than a wrong result because it survives every rerun and every model change.
    #
    # So verify the ruler before spending a single model call: does the gold function extract,
    # and does it differ from the parent? A task failing this cannot be scored and must be
    # named, never silently included. Cheap (pure AST, no repo copy, no network).
    unwinnable = []
    for t in tasks:
        _fs = FileSet.from_row(t)
        if _fs.is_multi_file:
            _label = str(t.get("task_id") or t.get("instance_id") or "multi_file")
            _parent_norm = {path: norm(source) for path, source in _fs.parent.items()}
            _fixed_norm = {path: norm(source) for path, source in _fs.fixed.items()}
            if not _fs.paths or any(value is None for value in _parent_norm.values()) \
                    or any(value is None for value in _fixed_norm.values()):
                unwinnable.append((_label, "one or more declared files do not parse"))
            elif _parent_norm == _fixed_norm:
                unwinnable.append((_label, "gold workspace is identical to parent"))
            continue
        g = fn_src(t["fixed_src"], t["fn"])
        p = fn_src(t["parent_src"], t["fn"])
        if not g or not p:
            unwinnable.append((t["fn"], "gold or parent function does not extract"))
        elif norm(g) == norm(p):
            unwinnable.append((t["fn"], "gold is identical to parent — nothing to repair"))
    if unwinnable:
        print("!! PREFLIGHT: these tasks cannot be scored and are EXCLUDED from the "
              "denominator:")
        for fn_, why in unwinnable:
            print(f"     {fn_:34s} {why}")
        bad = {f for f, _ in unwinnable}
        tasks = [t for t in tasks
                 if (str(t.get("task_id") or t.get("instance_id") or "multi_file")
                     if isinstance(t.get("files"), Mapping) else t["fn"]) not in bad]
        print()

    print(f"ReAct REPAIR — {MODEL}, {len(tasks)} real bugs, {limit_text(parse_step_limit(MAX_STEPS))} steps each\n")
    rows = []
    with OUT.open("a") as sink, cf.ThreadPoolExecutor(max_workers=4) as ex:
        for r in ex.map(work, tasks):
            rows.append(r)
            sink.write(json.dumps(r) + "\n")
            sink.flush()                     # per line: an interrupted run keeps everything
            print(f"  {r['fn'][:34]:36s} {r['result']:8s} "
                  f"steps={r['steps']:<3} edits={r['edits']} tests={r['tests']}", flush=True)

    p = sum(1 for r in rows if r["result"] == "PASS")
    g = sum(1 for r in rows if r["result"] in ("PASS", "fail"))
    used = [r["steps"] for r in rows if r["steps"]]
    # TWO GRADES, REPORTED SEPARATELY AND NEVER MERGED.
    # AST = "reproduced the shipped implementation exactly" — a strict, legacy diagnostic.
    # BEHAVIOURAL = "the repository own test passes with this patch AND fails without it" —
    # a discriminating check, which is the only kind worth anything (a test that passes on the
    # buggy code proves nothing). Merging them would let the softer grade quietly inflate the
    # headline, so both are printed with their own denominators.
    beh = sum(1 for r in rows if r.get("behaviour") == "PASS_behavioural")
    uncovered = sum(1 for r in rows if r.get("behaviour") == "test_does_not_cover_bug")
    notest = sum(1 for r in rows if r.get("behaviour") == "no_test_oracle")
    print(f"\n  ReAct SCORE : {p}/{g}   (exact-AST)")
    if beh or uncovered or notest:
        print(f"  BEHAVIOURAL : +{beh} additional task(s) where the repo test passes with the "
              f"patch and FAILS without it")
        print(f"                {uncovered} candidate(s) passed a test that also passes on "
              f"the BUGGY code (proves nothing)")
        print(f"                {notest} task(s) have no test oracle at all")
        print(f"  COMBINED    : {p + beh}/{g}  (AST-exact OR behaviourally verified)")
    print("  one-shot kernel : 1/21     naked one-shot : 2/21")
    print(f"  steps used: median {sorted(used)[len(used)//2] if used else 0}, max "
          f"{max(used) if used else 0} of {limit_text(parse_step_limit(MAX_STEPS))}")
    print(f"  runs that ran tests: {sum(1 for r in rows if r['tests'])}/{len(rows)}")
    print("\n  READ: if ReAct >> 2/21 the missing ingredient was ITERATION, not filtering,")
    print("  and the kernel should be rebuilt around a loop that can act and observe. If it")
    print("  does not move, iteration is not the lever either and the ceiling is elsewhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
