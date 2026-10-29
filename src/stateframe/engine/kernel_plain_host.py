"""Boundary adapter for the plain-loop engine: tools, workspace, checks, provider, export.

Executes the baseline's seven tools against StateFrame's isolated worker workspace
and serialized public-check service, records truthful host-side receipts, and
exports the declared-file candidate through ``kernel_bridge_evidence.capture_candidate``
(resolved at call time, so the Stage 6 candidate contract applies) after every
product edit and on every exit. Model-facing tool text matches the baseline,
except that real sandbox paths are shown under the /workspace alias the system
prompt promises.

Never imports the full StateFrame runtime (react_repair, KernelBolt, owner, F4).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

from kernel_plain_contracts import (
    ENGINE_ID, ROOT_ALIAS, CheckReceipt, EditAuthority, EditReceipt, PlainContractError, PlainJournal,
    ReadReceipt, ReceiptBook, file_revision, sha256_bytes, validate_schema_agreement,
)
from kernel_plain_hooks import HookSettings, PlainHooks
from kernel_plain_http import AbortiveHTTPTransport
from kernel_plain_loop import (
    OUT_CAP, TOOLS, Limits, ProviderHTTPError, ProviderTransportError, run_loop,
)
from kernel_test_python import test_python

CHECK_TIMEOUT_S = 300
GREP_TIMEOUT_S = 60
EDITABLE_NOTICE = "Only these existing files may be changed (new files and other edits are discarded): "
_PYTEST_CONFIG_NAMES = ("pytest.ini", "pyproject.toml", "setup.cfg")
_TEST_DIRS = frozenset({"test", "tests"})

# Offline replays/tests may install a materializer: (root, row) -> None, copying the public
# parent snapshot.  None means: use the Stage 6 adapter's hook when present, else row files only.
WORKSPACE_MATERIALIZER: Callable[[Path, Mapping[str, object]], None] | None = None
# Offline replays install a stub check runner whose receipts are marked synthetic (never green).
CHECK_RUNNER: "CheckRunner | None" = None
# The pristine parent tree for the final-product manifest when WORKSPACE_MATERIALIZER is injected.
PARENT_SNAPSHOT: Path | None = None


def parent_snapshot_root(workspace_source: str) -> Path | None:
    """Pristine parent tree: injected, or the Stage 6 adapter's mounted snapshot (REPOS['build'])."""
    if workspace_source == "injected":
        return PARENT_SNAPSHOT
    if workspace_source == "stage6":
        repair = sys.modules.get("probes.react_repair") or sys.modules.get("react_repair")
        snapshot = getattr(repair, "REPOS", {}).get("build")
        return snapshot if isinstance(snapshot, Path) else None
    return None


@dataclass(frozen=True)
class CheckOutcome:
    exit_code: int | None
    output: str
    timed_out: bool = False
    synthetic: bool = False
    error: str = ""


CheckRunner = Callable[[Sequence[str], Path, Mapping[str, str]], CheckOutcome]


def _active_test_run_profile():
    if os.environ.get("STATEFRAME_TEST_RUN_SPANS", "off").strip().lower() not in {"shadow", "on"}:
        return None
    from kernel_test_run_spans import current_profile
    return current_profile()


class PlainHost:
    """The seven baseline tools over one sandbox root."""

    SUPPORTED = frozenset({"list_dir", "read_file", "grep", "str_replace", "write_file", "run_tests"})

    def __init__(self, authority: EditAuthority, *, check_runner: CheckRunner, journal, hooks: PlainHooks,
                 on_product_edit: Callable[[], None] = lambda: None):
        self.authority, self.check_runner, self.journal, self.hooks = authority, check_runner, journal, hooks
        self.on_product_edit = on_product_edit
        self.book = ReceiptBook()
        validate_schema_agreement(TOOLS, set(self.SUPPORTED))

    @property
    def root(self) -> Path:
        return self.authority.root

    def alias(self, text: str) -> str:
        return text.replace(str(self.root), ROOT_ALIAS)

    def execute(self, name: str, args: dict) -> str:
        if not isinstance(args, dict):
            raise TypeError("tool arguments must be a JSON object")
        handler = getattr(self, "_" + name, None) if name in self.SUPPORTED else None
        if handler is None:
            return f"error: unknown tool {name}"
        return handler(args)

    def _list_dir(self, a: dict) -> str:
        p = self.authority.resolve(a.get("path", "."))
        return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()))

    def _read_file(self, a: dict) -> str:
        path = self.authority.resolve(a["path"])
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        s = max(1, int(a.get("start") or 1))
        e = min(len(lines), int(a.get("end") or len(lines)))
        relative = path.relative_to(self.root).as_posix()
        ranged = bool(a.get("start") or a.get("end"))
        ws, we = self.hooks.widen(relative, text, s, e, ranged=ranged) if e >= s else (s, e)
        out = "\n".join(f"{i}\t{lines[i - 1]}" for i in range(ws, we + 1)) or "(empty)"
        self.book.reads += 1
        self.journal.receipt("read_receipt", ReadReceipt(
            path=relative, revision=file_revision(path), requested=(a.get("start"), a.get("end")),
            returned=(ws, we), total_lines=len(lines), widened=(ws, we) != (s, e), chars=len(out),
            truncated_by_cap=len(out) > OUT_CAP))
        return out

    def _grep(self, a: dict) -> str:
        raw = a.get("path") or "."
        target = self.authority.resolve(raw)
        arg = str(target) if str(raw).startswith("/") else (target.relative_to(self.root).as_posix()
                                                          if target != self.root else ".")
        try:
            p = subprocess.run(["grep", "-rnE", "--exclude-dir=__pycache__", "--exclude-dir=node_modules",
                                a["pattern"], arg], cwd=self.root, capture_output=True, text=True,
                               errors="replace", timeout=GREP_TIMEOUT_S)
            return self.alias(f"exit={p.returncode}\n{p.stdout}{p.stderr}")
        except subprocess.TimeoutExpired:
            return f"TIMEOUT after {GREP_TIMEOUT_S}s"

    def _str_replace(self, a: dict) -> str:
        scope, path, relative = self.authority.classify(a["path"])
        if scope == "refused":
            return self._refused("str_replace", relative)
        old_rev = file_revision(path)
        text = path.read_text(encoding="utf-8")
        n = text.count(a["old"])
        if n != 1:
            self._edit_receipt("str_replace", relative, scope, False, old_rev, old_rev, n,
                               "anchor must occur exactly once")
            return f"error: old string found {n} times (must be exactly 1)"
        path.write_text(text.replace(a["old"], a["new"], 1), encoding="utf-8")
        result = self._applied("str_replace", relative, scope, old_rev, path, n)
        if scope == "product":
            fact = self.hooks.sibling_fact(relative, a["old"], a["new"], self._declared_sources(),
                                           self.authority.product_revision())
            if fact:
                self.journal.write("hook_applied", hook="sibling_sites", fact=fact)
                return result + "\n" + fact
        return result

    def _write_file(self, a: dict) -> str:
        scope, path, relative = self.authority.classify(a["path"])
        if scope == "refused":
            return self._refused("write_file", relative)
        old_rev = file_revision(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(a["content"], encoding="utf-8")
        return self._applied("write_file", relative, scope, old_rev, path, None)

    def _run_tests(self, a: dict) -> str:
        if os.environ.get("STATEFRAME_TEST_RUN_SPANS", "off").strip().lower() not in {"shadow", "on"}:
            return self._run_tests_impl(a, None)
        from kernel_test_run_spans import profile_scope
        with profile_scope(runtime=self.journal) as profile:
            return self._run_tests_impl(a, profile)

    def _run_tests_impl(self, a: dict, profile) -> str:
        from kernel_wire_identity import public_wire_text

        def visible(output: str) -> str:
            return public_wire_text(self.alias(output), self.journal)

        selector_span = profile.start_span("selector_resolution") if profile else None
        tokens = [_unalias(token) for token in shlex.split(a.get("args", ""))]
        selectors = tuple(t for t in tokens if _is_test_selector(self.root, t))
        if profile:
            profile.end_span(selector_span)
        config_span = profile.start_span("config_command_preparation") if profile else None
        config = [] if any(t in {"-c", "--config-file"} or t.startswith("--config-file=") for t in tokens) \
            else _nearest_config_args(self.root, selectors)
        argv = [test_python(), "-m", "pytest", "-p", "no:cacheprovider", *config, *tokens]
        if profile:
            profile.end_span(config_span)
        binding_span = profile.start_span("pre_test_binding") if profile else None
        revision = self.authority.product_revision()
        if profile:
            profile.end_span(binding_span)
        try:
            if profile:
                with profile.span("test_child_and_readiness"):
                    outcome = self.check_runner(argv, self.root, {})
            else:
                outcome = self.check_runner(argv, self.root, {})
        except PlainContractError:
            raise
        except Exception as error:  # an execution refusal is an error observation, never green
            outcome = CheckOutcome(exit_code=None, output=f"error: test execution refused: {error}",
                                   error=f"{type(error).__name__}: {error}"[:300])
        self.book.checks.append(receipt := CheckReceipt(
            argv=tuple(argv[1:]), selectors=selectors, product_revision=revision,
            exit_code=outcome.exit_code, timed_out=outcome.timed_out, synthetic=outcome.synthetic,
            output_sha256=sha256_bytes(outcome.output.encode()), output_chars=len(outcome.output),
            error=outcome.error))
        if profile:
            with profile.span("first_receipt"):
                self.journal.write("check_receipt", receipt=asdict(receipt),
                                   execution_id=profile.execution_id)
        else:
            self.journal.receipt("check_receipt", receipt)
        if profile:
            with profile.span("final_result_enrichment"):
                if outcome.timed_out:
                    return f"TIMEOUT after {CHECK_TIMEOUT_S}s"
                if outcome.exit_code is None:
                    return visible(outcome.output)
                return visible(f"exit={outcome.exit_code}\n{outcome.output}")
        if outcome.timed_out:
            return f"TIMEOUT after {CHECK_TIMEOUT_S}s"
        if outcome.exit_code is None:
            return visible(outcome.output)
        return visible(f"exit={outcome.exit_code}\n{outcome.output}")

    def _refused(self, tool: str, relative: str) -> str:
        self.book.refused_edits += 1
        self._edit_receipt(tool, relative, "refused", False, "unchanged", "unchanged", None, "outside edit authority")
        return self.authority.refusal(relative)

    def _applied(self, tool: str, relative: str, scope: str, old_rev: str, path: Path, n: int | None) -> str:
        new_rev = file_revision(path)
        if scope == "product":
            self.book.product_edits += 1
        else:
            self.book.scratch_edits += 1
            self.authority.scratch_created.add(relative)
        self._edit_receipt(tool, relative, scope, old_rev != new_rev, old_rev, new_rev, n,
                           "" if old_rev != new_rev else "content unchanged")
        if scope == "product":
            self.on_product_edit()
        return "ok"

    def _edit_receipt(self, tool, relative, scope, applied, old_rev, new_rev, n, reason) -> None:
        self.journal.receipt("edit_receipt", EditReceipt(
            tool=tool, path=relative, scope=scope, applied=applied, old_revision=old_rev,
            new_revision=new_rev, product_revision=self.authority.product_revision(),
            occurrences=n, reason=reason))

    def _declared_sources(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {}
        for relative in self.authority.declared:
            path = self.root / relative
            out[relative] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
        return out


def _unalias(token: str) -> str:
    if token == ROOT_ALIAS:
        return "."
    return token[len(ROOT_ALIAS) + 1:] if token.startswith(ROOT_ALIAS + "/") else token


def _is_test_selector(root: Path, token: str) -> bool:
    base = token.split("::", 1)[0]
    if token.startswith("-") or not base.endswith(".py"):
        return False
    parts = Path(base).parts
    return (root / base).is_file() and any(part in _TEST_DIRS for part in parts[:-1])


def _nearest_config_args(root: Path, selectors: Sequence[str]) -> list[str]:
    """``-c <relative config>`` for the one project config enclosing every selector."""
    found = set()
    for selector in selectors:
        current = (root / selector.split("::", 1)[0]).resolve().parent
        while current == root or root in current.parents:
            hit = next((current / name for name in _PYTEST_CONFIG_NAMES if (current / name).is_file()
                        and (name != "pyproject.toml" or "[tool.pytest" in (current / name).read_text(errors="replace"))
                        and (name != "setup.cfg" or "[tool:pytest]" in (current / name).read_text(errors="replace"))),
                       None)
            if hit is not None:
                found.add(hit.relative_to(root).as_posix())
                break
            if current == root:
                break
            current = current.parent
    return ["-c", found.pop()] if len(found) == 1 else []


def capability_preflight_argv(root: Path, row: Mapping[str, object]) -> list[str] | None:
    """A collect-only run of the first declared public check, or None when there is none.

    Stage 6 runs its child-capability canary lazily, only inside a declared-public-selector
    pytest command, while every check leaves a test-lock row. A plain run whose model never
    names a public test file (batch 1076) therefore had lock rows but no canary receipt and
    settled ERROR/EVIDENCE_INCOMPLETE, dropping a salvage-gradable candidate.
    """
    paths = row.get("public_check_paths")
    selector = next((p for p in paths if isinstance(p, str) and (root / p).is_file()), None) \
        if isinstance(paths, list) else None
    if selector is None:
        return None
    return [test_python(), "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q",
            *_nearest_config_args(root, [selector]), selector]


def sandboxed_check_runner(lock_path: Path, audit_path: Path, *,
                           preflight_argv: Sequence[str] | None = None,
                           on_preflight: Callable[[dict], None] = lambda receipt: None) -> CheckRunner:
    """Public checks through the frozen execution policy, serialized by the study lock.

    Before the first check, a host-only capability preflight (never shown to the model)
    proves the execution sandbox, so every run with test executions carries its canary.
    """
    preflight_pending = preflight_argv is not None

    def run(argv: Sequence[str], root: Path, _extra: Mapping[str, str]) -> CheckOutcome:
        nonlocal preflight_pending
        profile = _active_test_run_profile()
        if preflight_pending:
            preflight_pending = False
            try:
                if profile:
                    with profile.span("capability_canary"):
                        outcome = execute(preflight_argv, root)
                else:
                    outcome = execute(preflight_argv, root)
                canary_receipt = {"argv": list(preflight_argv[1:]), "exit_code": outcome.exit_code,
                                  "timed_out": outcome.timed_out}
                if profile:
                    canary_receipt["execution_id"] = profile.execution_id
                on_preflight(canary_receipt)
            except Exception as error:  # the model's own check below surfaces the same refusal
                refusal = {"argv": list(preflight_argv[1:]),
                           "error": f"{type(error).__name__}: {error}"[:300]}
                if profile:
                    refusal["execution_id"] = profile.execution_id
                on_preflight(refusal)
        if profile:
            with profile.span("requested_test_subprocess"):
                return execute(argv, root)
        return execute(argv, root)

    def execute(argv: Sequence[str], root: Path) -> CheckOutcome:
        from capability_study_support import test_process_lock
        from kernel_execution_policy import run_sandboxed
        from kernel_test_artifact_policy import prepare_pytest_command
        profile = _active_test_run_profile()
        config_span = profile.start_span("config_command_preparation") if profile else None
        command, env, _policy = prepare_pytest_command(list(argv), root, environment=dict(os.environ))
        profiled_command = False
        if profile:
            try:
                from kernel_test_run_spans import profiled_pytest_command
                wrapped = profiled_pytest_command(command, profile.execution_id)
                if wrapped is not None:
                    command = wrapped
                    profiled_command = True
            except Exception:  # telemetry setup failure keeps the current command intact
                profile.mark("pytest_profile_command_failed_open")
        if profile:
            profile.end_span(config_span)
            profile.mark("test_lock_requested")
            wait_span = profile.start_span("test_lock_wait", overlap_group="shared_lock")
        held_span = None
        try:
            with test_process_lock(lock_path, audit_path):
                if profile:
                    profile.end_span(wait_span)
                    profile.mark("test_lock_acquired")
                    held_span = profile.start_span("test_lock_hold", overlap_group="shared_lock")
                execution_options = {}
                if profiled_command:
                    execution_options["observe_pytest"] = True
                result = run_sandboxed(
                    list(command), root, env=env, timeout=CHECK_TIMEOUT_S,
                    cpu_seconds=float(os.environ.get("STATEFRAME_PUBLIC_CHECK_CPU_SECONDS", "300")),
                    output_limit=256 * 1024, **execution_options,
                )
                if profile:
                    try:
                        profile.record_sandbox_result(
                            result.execution_trace, result.profile_payload,
                            stdout=result.completed.stdout, stderr=result.completed.stderr,
                            returncode=result.completed.returncode,
                        )
                    except Exception:  # profiler data is advisory
                        pass
        finally:
            if profile and held_span is not None:
                profile.end_span(held_span)
                profile.mark("test_lock_released")
        stdout, stderr = result.completed.stdout or "", result.completed.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        note = "\n[output truncated by the host]" if result.output_truncated else ""
        return CheckOutcome(exit_code=None if result.timed_out else result.completed.returncode,
                            output=stdout + stderr + note, timed_out=bool(result.timed_out))

    return run


GZIP_ENV = "STATEFRAME_PLAIN_GZIP"
GZIP_MIN_BYTES = 32 * 1024


def encode_request_body(body: dict, environ: Mapping[str, str] | None = None) -> tuple[bytes, dict]:
    """Serialize the request; gzip it when large (the lab relay inflates ``Content-Encoding: gzip``).

    10-13: the container -> socat sidecar -> host.docker.internal path truncates request bodies at
    a fixed ~149,326 bytes (relay_short_body expected=162765 received=149326, 5x identical), so a
    transcript past that size could never be sent. gzip keeps the wire well under the cap.
    ``STATEFRAME_PLAIN_GZIP=off`` restores identity bodies for endpoints without the relay.
    """
    raw = json.dumps(body).encode()
    mode = str((environ if environ is not None else os.environ).get(GZIP_ENV, "on")).strip().lower()
    if mode not in {"on", "off"}:
        raise ValueError(f"{GZIP_ENV} must be 'on' or 'off'")
    # End-to-end integrity header over the IDENTITY body: the lab relay verifies it after any
    # inflate and refuses a mismatch, localizing byte corruption (10-13 benchmark 1076: a NUL inside
    # our own "tool_call_id" key reached llama-server at byte ~161 KB). Other servers ignore it.
    integrity = {"X-Request-Body-SHA256": hashlib.sha256(raw).hexdigest()}
    if mode == "on" and len(raw) >= GZIP_MIN_BYTES:
        import gzip
        return gzip.compress(raw, compresslevel=6, mtime=0), {"Content-Encoding": "gzip", **integrity}
    return raw, integrity


class _PlainHTTPProvider:
    def __init__(self, url: str, key: str) -> None:
        self.url = url
        self.key = key
        self.transport = AbortiveHTTPTransport()

    def __call__(self, body: dict, timeout: float) -> Mapping[str, object]:
        data, encoding = encode_request_body(body)
        request = urllib.request.Request(self.url, data=data, headers={
            "Content-Type": "application/json", **encoding,
            **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        tracker_id = self.transport.begin()
        completed = False
        try:
            try:
                with self.transport.opener(tracker_id).open(request, timeout=timeout) as response:
                    payload = response.read()
                    self.transport.finish(tracker_id)
                completed = True
                return json.loads(payload)
            except urllib.error.HTTPError as error:
                try:
                    error_body = error.read().decode(errors="replace")
                except (TimeoutError, urllib.error.URLError, ConnectionError, OSError,
                        http.client.HTTPException) as transport_error:
                    reason = getattr(transport_error, "reason", None)
                    timed_out = (isinstance(transport_error, TimeoutError)
                                 or isinstance(reason, TimeoutError))
                    raise ProviderTransportError(
                        f"{type(transport_error).__name__}: {transport_error}"[:200],
                        timed_out=timed_out) from transport_error
                raise ProviderHTTPError(error.code, error_body) from error
            # HTTPException covers IncompleteRead (a truncated body after a relay drop).
            except (TimeoutError, urllib.error.URLError, ConnectionError, OSError,
                    http.client.HTTPException) as error:
                reason = getattr(error, "reason", None)
                timed_out = isinstance(error, TimeoutError) or isinstance(reason, TimeoutError)
                raise ProviderTransportError(
                    f"{type(error).__name__}: {error}"[:200], timed_out=timed_out) from error
        finally:
            if not completed:
                self.transport.abort(tracker_id)
            self.transport.release(tracker_id)

    def abort(self) -> None:
        self.transport.abort_active()


def http_provider(url: str, key: str) -> Callable[[dict, float], Mapping[str, object]]:
    return _PlainHTTPProvider(url, key)


def limits_from_env(environ: Mapping[str, str]) -> Limits:
    def number(key: str, default: str, kind=float):
        raw = str(environ.get(key, default) or default)
        try:
            return kind(raw)
        except ValueError as error:
            raise ValueError(f"{key} must be numeric") from error

    sampler = {}
    for key, field, kind in (("REACT_TOP_P", "top_p", float), ("REACT_TOP_K", "top_k", int),
                             ("REACT_MIN_P", "min_p", float)):
        if environ.get(key) not in (None, ""):
            sampler[field] = number(key, "0", kind)
    wall = number("STATEFRAME_PLAIN_WALL_S", "1800")
    deadline = environ.get("CAPABILITY_ATTEMPT_DEADLINE_MONOTONIC")
    if deadline:  # never outlive the supervisor; leave a minute to export
        wall = max(1.0, min(wall, float(deadline) - time.monotonic() - 60))
    return Limits(model=str(environ.get("DS_MODEL", "")), wall_s=wall,
                  max_requests=number("STATEFRAME_PLAIN_MAX_REQUESTS", "150", int),
                  request_timeout_s=number("REACT_REQUEST_TIMEOUT_S", "600"),
                  temperature=number("REACT_TEMP", "1.0"),
                  max_tokens=number("REACT_MAX_OUTPUT_TOKENS", "24000", int),
                  reasoning_budget=number("STATEFRAME_PLAIN_REASONING_BUDGET", "0", int),
                  sampler=sampler)


def materialize_workspace(root: Path, row: Mapping[str, object]) -> str:
    """Declared files first, then the parent snapshot (Stage 6 hook or injected) wins."""
    files = row.get("files")
    if not isinstance(files, Mapping) or not files:
        raise PlainContractError("plain engine requires a multi-file public row with declared files")
    for relative, payload in files.items():
        destination = (root / relative).resolve()
        if root.resolve() not in destination.parents:
            raise PlainContractError(f"declared path escapes the workspace: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(payload["parent_src"]), encoding="utf-8")
    public_tests = row.get("public_tests") if isinstance(row.get("public_tests"), Mapping) else {}
    if WORKSPACE_MATERIALIZER is not None:
        WORKSPACE_MATERIALIZER(root, row)
        source = "injected"
    else:
        repair = sys.modules.get("probes.react_repair") or sys.modules.get("react_repair")
        hook = getattr(repair, "_materialize_public_tests", None)
        # Only the Stage 6 adapter's repo-root materializer copies the full parent snapshot;
        # the frozen generic one needs the full runtime's FileSet and is never called here.
        if callable(hook) and getattr(hook, "__name__", "") == "materialize_repo_root_tests":
            hook(root, SimpleNamespace(public_tests=dict(public_tests)), dict(row))
            return "stage6"
        source = "row"
    for relative, text in public_tests.items():
        destination = (root / relative).resolve()
        if root.resolve() not in destination.parents:
            raise PlainContractError(f"public test escapes the workspace: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(str(text), encoding="utf-8")
    return source


def run_plain(row: Mapping[str, object], output: Path, *,
              provider: Callable[[dict, float], Mapping[str, object]] | None = None,
              check_runner: CheckRunner | None = None,
              before_request: Callable[[dict], None] | None = None,
              environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """One plain episode inside ``output`` (the worker directory); returns the outcome mapping."""
    env = dict(os.environ if environ is None else environ)
    limits = limits_from_env(env)
    settings = HookSettings.from_env(env)
    journal = PlainJournal(output / "plain")
    capture_dest = row.get("_bridge_capture")
    root = Path(tempfile.mkdtemp(prefix="plain_", dir=str(output / "scratch"))) / "repo"
    root.mkdir()
    workspace_source = materialize_workspace(root, row)
    authority = EditAuthority(root, list(row["files"]))
    hooks = PlainHooks(settings, authority.declared, journal, out_cap=OUT_CAP)
    exports = {"count": 0, "failures": 0}

    def export(reason: str) -> None:
        if not capture_dest:
            return
        import kernel_bridge_evidence  # resolved at call time: Stage 6 replaces capture_candidate
        exports["count"] += 1
        try:
            kernel_bridge_evidence.capture_candidate(str(capture_dest), root, dict(row), exports["count"])
        except Exception as error:
            exports["failures"] += 1
            journal.write("candidate_export_failed", reason=reason, index=exports["count"],
                          error=f"{type(error).__name__}: {error}"[:300])
            raise PlainContractError(f"candidate export failed: {error}") from error
        journal.write("candidate_exported", reason=reason, index=exports["count"],
                      product_revision=authority.product_revision())

    if check_runner is None:
        check_runner = CHECK_RUNNER
    if check_runner is None:
        lock =Path(env.get("CAPABILITY_TEST_LOCK", str(output / "study-pytest.lock")))
        check_runner = sandboxed_check_runner(
            lock, output.parent / "test-lock.jsonl", preflight_argv=capability_preflight_argv(root, row),
            on_preflight=lambda receipt: journal.write("capability_preflight", **receipt))
    host = PlainHost(authority, check_runner=check_runner, journal=journal, hooks=hooks,
                     on_product_edit=lambda: export("product_edit"))
    if provider is None:
        url = env.get("DS_URL", "")
        if not url:
            raise PlainContractError("plain engine requires DS_URL")
        provider = http_provider(url, env.get("DS_API_KEY") or env.get("DEEPINFRA_API_KEY", ""))
    task_text = str(row.get("msg") or "") + "\n\n" + EDITABLE_NOTICE + ", ".join(authority.declared)
    parent_revision = authority.product_revision()
    journal.save("engine.json", {"engine": ENGINE_ID, "hooks": settings.receipt(), "limits": limits.__dict__,
                                 "declared_files": list(authority.declared), "workspace_source": workspace_source,
                                 "parent_revision": parent_revision,
                                 "task_sha256": hashlib.sha256(task_text.encode()).hexdigest()})
    loop = None
    error: BaseException | None = None
    try:
        loop = run_loop(task_text, provider=provider, execute=host.execute, limits=limits, journal=journal,
                        prepare=lambda m: hooks.prepare(m, product_edits=host.book.product_edits),
                        before_request=before_request)
    except BaseException as caught:  # preserve the candidate first, then surface the failure
        error = caught
        journal.write("run_exception", error_type=type(caught).__name__, error_detail=str(caught)[:500])
    finally:
        try:
            export("final")
        except PlainContractError as export_error:
            error = error or export_error
    current = authority.product_revision()
    completion = host.book.completion(current)
    outcome = {
        "engine": ENGINE_ID, "result": (loop.end if loop else "host_error").upper(),
        "end": loop.end if loop else "host_error", "steps": loop.requests if loop else 0,
        "tool_calls": loop.tool_calls if loop else 0, "edits": host.book.product_edits,
        "scratch_edits": host.book.scratch_edits, "refused_edits": host.book.refused_edits,
        "tests": len(host.book.checks), "transport_retries": loop.transport_retries if loop else 0,
        "prompt_tokens": loop.prompt_tokens if loop else 0,
        "completion_tokens": loop.completion_tokens if loop else 0, "wall_s": loop.wall_s if loop else 0.0,
        "summary": loop.summary if loop else "", "completion": completion,
        "parent_revision": parent_revision, "candidate_revision": current,
        "product_changed": current != parent_revision, "candidates_exported": exports["count"],
        "parent_status": "UNCOVERED",
        "candidate_status": "OBSERVED_PASS" if completion["status"] == "VERIFIED_PUBLIC_GREEN" else "UNCOVERED",
        "hooks": settings.receipt(),
    }
    if error is None:  # a host exception settles as a worker exception instead
        from kernel_plain_settlement import settle_stage6
        changed = sum(1 for relative, payload in row["files"].items()
                      if not (root / relative).is_file()
                      or (root / relative).read_text(encoding="utf-8", errors="replace") != payload["parent_src"])
        outcome["settlement"] = settle_stage6(
            outcome=outcome, row=row, final_root=root, parent_root=parent_snapshot_root(workspace_source),
            product_files_changed=changed, journal=journal)
    journal.save("outcome.json", outcome)
    if error is not None:
        raise error
    return outcome


__all__ = ["CheckOutcome", "EDITABLE_NOTICE", "PlainHost", "WORKSPACE_MATERIALIZER", "http_provider",
           "limits_from_env", "materialize_workspace", "run_plain", "sandboxed_check_runner"]
