"""Foreground shell: capability admission, transactional effects and durable replay."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import time
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from kernel_effects import (
    BaselineStore, ContentStore, EffectError, Manifest, effect_policy_for_run, manifest_digest,
    reconcile, scan,
)
from kernel_exec_capability import current_path_authority_fingerprint, shell_capability
from kernel_exec_session import ExecSession, ExecSessionError
from kernel_run_env import RunEnv, RunEnvError, env_identity
from kernel_shell_policy import ShellPolicyRefused, admit_shell, normalize_output_delivery
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import (ToolReplayRefused, ToolTapeError, replay_decision, replay_mode_active,
                              tape_for_run, verify_intent, verify_reexecution)

from kernel_safe_paths import container_active, safe_home

BACKEND_CONTAINER = "container"


def active() -> SimpleNamespace:
    """Expose the selected backend without loading the boundary module."""
    return SimpleNamespace(backend=BACKEND_CONTAINER if container_active() else "platform")


SHELL_SCHEMA = {"type": "function", "function": {
    "name": "shell", "description": "Run a bounded foreground shell command. Output is not source-read or check evidence.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
        "background": {"type": "boolean"}, "why": {"type": "string"}},
        "required": ["command"]}}}


def workspace_identity(manifest: Manifest) -> str:
    """Tape identity follows workcopy scope, without weakening effect inspection."""
    from stateframe.workcopy import _ignored

    authoritative = {}
    for relative, entry in manifest.items():
        parts = PurePosixPath(relative).parts
        if (any(_ignored(part, True) for part in parts[:-1])
                or _ignored(parts[-1], entry.kind == "dir")):
            continue
        authoritative[relative] = entry
    return manifest_digest(authoritative)


def attach_shell_verification(sequence: list[dict], call_id: str,
                              assistant: dict, message: dict) -> bool:
    """Keep a shell result and its automatic check in one newest exchange.

    Frame-only requests retain one complete batch. A separate synthetic batch
    would hide the command's output and effects before the model could see them.
    """
    def is_effect_tool(name):
        from kernel_agent_tools import service
        host = service()
        return name == "shell" or (host is not None and name in host.custom
                                   and host.custom[name].writes == "workspace")

    for index in range(len(sequence) - 1, -1, -1):
        parent = sequence[index]
        calls = parent.get("tool_calls")
        if parent.get("role") != "assistant" or not isinstance(calls, list):
            continue
        if not any(call.get("id") == call_id and
                   is_effect_tool(call.get("function", {}).get("name")) for call in calls):
            continue
        end = index + 1
        while end < len(sequence) and sequence[end].get("role") == "tool":
            end += 1
        results = sequence[index + 1:end]
        if ({item.get("tool_call_id") for item in results}
                != {call.get("id") for call in calls} or len(results) != len(calls)):
            return False
        parent = copy.deepcopy(parent)
        parent["tool_calls"].extend(copy.deepcopy(assistant["tool_calls"]))
        sequence[index] = parent
        sequence.insert(end, copy.deepcopy(message))
        return True
    return False


def bounded_summary(text: str) -> str:
    data = text.encode("utf-8", errors="backslashreplace")  # a restored name may not be valid UTF-8
    if len(data) <= 4096:
        return data.decode("utf-8")
    return data[:4096].decode("utf-8", errors="ignore") + f" [... {len(data) - 4096} bytes omitted ...]"


_ENV_IDENTITY_EPOCH_SWITCH = "STATEFRAME_SHELL_ENV_IDENTITY_EPOCH"
_NON_MUTATING_SHELL_BUILTINS = frozenset({":", "true", "false", "pwd", "echo", "printf"})
_SHELL_CONTROL_CHARACTERS = frozenset(";&|<>`$\n\r")


def _known_non_mutating_command(command: str) -> bool:
    """Recognize only simple builtins that cannot dispatch or write another command."""
    if any(character in _SHELL_CONTROL_CHARACTERS for character in command):
        return False
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    return not words or words[0] in _NON_MUTATING_SHELL_BUILTINS


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def _effects_touch_environment(effects, workspace: Path, environment: RunEnv) -> bool:
    """Catch recorded writes into the run env, its site packages, or its interpreter paths."""
    if not effects.applied:
        return False
    roots = [environment.root, environment.bin_dir, environment.python]
    site_manifest = environment.root.parent / "env-sites.json"
    roots.append(site_manifest)
    try:
        roots.extend(environment.root.glob("lib/python*/site-packages"))
        if site_manifest.is_file():
            sites = json.loads(site_manifest.read_text())
            if not isinstance(sites, list) or any(not isinstance(path, str) for path in sites):
                return True
            roots.extend(Path(path) for path in sites)
    except (OSError, UnicodeError, json.JSONDecodeError):
        # If env identity inputs are unreadable, do not assume they are untouched.
        return True
    try:
        resolved_roots = [root.resolve(strict=False) for root in roots]
    except (OSError, RuntimeError):
        return True
    for effect in effects.applied:
        path = Path(effect.path)
        candidate = path if path.is_absolute() else workspace / path
        try:
            candidate = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            # A path we cannot resolve is not safe to treat as unrelated to the env.
            return True
        if any(_paths_overlap(candidate, root) for root in resolved_roots):
            return True
    return False


def _command_may_change_environment(command: str, custom, effects,
                                    workspace: Path, environment: RunEnv) -> bool:
    # Custom tools, package installs, scripts, and unfamiliar shell expressions are unknown
    # mutations. Only the simple builtin allowlist can keep the current epoch alive.
    if custom is not None or not _known_non_mutating_command(command):
        return True
    return _effects_touch_environment(effects, workspace, environment)


def _container_tampered_protected_tests(baseline: Manifest, current: Manifest,
                                        paths) -> tuple[str, ...]:
    return tuple(sorted(path for path in paths
                        if path not in baseline or path not in current
                        or baseline[path].digest != current[path].digest))


def _container_test_tamper_notice(paths: tuple[str, ...]) -> str:
    return "DETECTED PROTECTED_TEST_TAMPERED: post-command hash changed for " + ", ".join(paths)


class ShellExecutionFailed(BaseException):
    """Fatal shell boundary failure; ordinary tool-error recovery must not swallow it."""


class ShellCleanupFailed(ShellExecutionFailed):
    """A command could not be made quiescent; no subsequent action is safe."""


@dataclass(frozen=True)
class ShellReply:
    observation: str
    applied: tuple = ()
    environment_changed: bool = False
    environment_identity: str = "disabled"
    fatal: bool = False
    exit_code: int | None = None
    memory_limit_mechanism: str | None = None
    memory_limit_killed: bool = False
    cgroup_leaf_pattern: str | None = None


def request_confirmation(run_dir: Path, call_id: str, command: str, *, timeout: float = 120) -> bool:
    request = run_dir / "shell-ask.json"
    temporary = run_dir / "shell-ask.pending"
    temporary.write_text(json.dumps({"call_id": call_id, "command": command}))
    temporary.replace(request)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            answer = run_dir / "shell-answer.json"
            if answer.exists():
                value = json.loads(answer.read_text())
                if value.get("call_id") == call_id:
                    return value.get("approved") is True
            time.sleep(0.05)
        return False
    finally:
        request.unlink(missing_ok=True)


class ShellTool:
    def __init__(self, run_dir: Path, settings: ToolSettings, spec: dict, session: ExecSession):
        self.run_dir, self.settings, self.spec, self.session = run_dir, settings, spec, session
        value = spec.get("run_env")
        self.environment = (RunEnv(Path(value["root"]), Path(value["root"]) / "bin/python",
                                   Path(value["root"]) / "bin", value["env"], value["kind"])
                            if value else None)
        self._environment_identity_epoch_enabled = (
            os.environ.get(_ENV_IDENTITY_EPOCH_SWITCH, "on").strip().lower() != "off")
        self._environment_identity_epoch = 0
        self._environment_identity_cache: tuple[int, str] | None = None
        self.root = None
        self.workspace_effects = None
        self.protected_test_paths: set[str] = set()
        self._pending_baseline_root: Path | None = None
        self._load_effect_policy = None
        self._post_manifest_reuse_enabled = True
        self._post_manifest: Manifest | None = None
        self._post_manifest_key: tuple[str, int, int, str | None] | None = None
        self._path_snapshot = spec.get("shell_path_snapshot")
        # STATEFRAME_SHELL_STATE: cwd and exported variables carried between foreground calls.
        import kernel_shell_state
        self.shell_state = kernel_shell_state.ShellState() if kernel_shell_state.enabled() else None

    def _clear_post_manifest(self) -> None:
        self._post_manifest = None
        self._post_manifest_key = None

    def invalidate_manifest_cache(self) -> None:
        """Discard a reusable shell baseline before a native workspace write."""
        self._clear_post_manifest()

    def _manifest_scan_key(self) -> tuple[str, int, int, str | None]:
        if self.root is None:
            raise EffectError("shell workspace is not bound")
        info = self.root.stat(follow_symlinks=False)
        excluded = str(self.environment.root) if self.environment is not None else None
        return str(self.root), info.st_dev, info.st_ino, excluded

    def _cached_post_manifest(self, capability) -> Manifest | None:
        if (self._post_manifest is None or self.workspace_effects is not None
                or not self._post_manifest_reuse_enabled or not capability.sandboxed):
            return None
        if replay_mode_active():
            self._clear_post_manifest()
            return None
        try:
            current_key = self._manifest_scan_key()
        except OSError:
            self._clear_post_manifest()
            return None
        if current_key != self._post_manifest_key:
            self._clear_post_manifest()
            return None
        return self._post_manifest

    def _remember_post_manifest(self, effects, capability) -> None:
        # Native edit paths invalidate this cache; only a settled, unreverted state is reusable.
        if (not self._post_manifest_reuse_enabled or self.workspace_effects is not None
                or not capability.sandboxed or replay_mode_active()
                or effects.restored):
            self._clear_post_manifest()
            return
        try:
            key = self._manifest_scan_key()
        except OSError:
            self._clear_post_manifest()
            return
        self._post_manifest = effects.manifest
        self._post_manifest_key = key

    def bind(self, workspace: Path, row: dict, baseline_tests: set[str], *, policy=None) -> None:
        if self.root is not None:
            raise RuntimeError("shell workspace is already bound")
        self.root = workspace.resolve(strict=True)
        self.protected_test_paths = set()
        for name in baseline_tests:
            path = Path(name)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.root)
                except ValueError:
                    continue
            if not path.is_absolute() and ".." not in path.parts:
                self.protected_test_paths.add(path.as_posix())
        self._clear_post_manifest()
        home = self.run_dir / "shell-home"
        home.mkdir(mode=0o700, exist_ok=True)
        env_root = self.environment.root if self.environment else self.run_dir / "shell-cache"
        env_root.mkdir(mode=0o700, exist_ok=True)
        writable = self.settings.shell != "read-only"
        cap = shell_capability(workspace=self.root, run_env_dir=env_root, scratch_home=home,
            network=self.settings.shell_network, writable=writable,
            extra_read_roots=tuple(Path(p) for p in self.spec.get("shell_read_roots", ())),
            path_dirs=tuple(Path(p) for p in self.spec.get("shell_path_dirs", ())),
            **({"path_snapshot": self._path_snapshot} if self._path_snapshot is not None else {}))
        environment = dict(cap.env)
        if self.environment:
            environment.update({key: value for key, value in self.environment.env.items() if key != "PATH"})
        if not writable:
            # Inspection resolves trusted host tools, never executables installed by a prior command.
            environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        self.capability = replace(cap, env=environment,
            network=(self.settings.shell_network if writable or active().backend == BACKEND_CONTAINER
                     else "none"),
            read_roots=(*cap.read_roots, env_root),
            write_roots=cap.write_roots if writable else (home,),
            sandboxed=self.settings.host_authorization is None)
        # The session envelope admits explicitly trusted custom capabilities;
        # each actual command still receives its narrower per-tool capability.
        self.session.configure(replace(self.capability, network=self.settings.shell_network))
        from sys import modules
        module = modules.get("kernel_workspace_consumers")
        consumers = module.consumers_for(self.root) if module is not None else None
        if consumers is None:
            import os
            if (os.environ.get("STATEFRAME_WORKSPACE_OBJECTS") == "on"
                    and self.capability.sandboxed):
                from kernel_workspace_consumers import bind_run_consumers
                consumers = bind_run_consumers(
                    self.run_dir, self.root, parent=Path(self.spec.get("parent_snapshot", self.root)),
                    excluded=self.environment.root if self.environment else None,
                    defer_parent=True,
                )
        self.workspace_consumers = consumers
        self.workspace_effects = consumers.shell if consumers is not None else None
        if self.workspace_effects is None:
            self.store = ContentStore(self.run_dir)
            baseline_root = Path(self.spec.get("parent_snapshot", self.root)).resolve(strict=True)
            self.baseline = scan(baseline_root, exclude=lambda _: False)
            self.baseline_store = BaselineStore(self.run_dir / "shell-baseline", self.baseline)
            if (self.capability.sandboxed and not baseline_root.is_relative_to(self.root)
                    and not self.root.is_relative_to(baseline_root)):
                # The parent snapshot is outside the command workspace. Keep its
                # manifest for policy and capture its bytes before execution.
                self._pending_baseline_root = baseline_root
            else:
                self.baseline_store.publish(baseline_root)
        elif self.workspace_effects.load_parent is not None and policy is None:
            paths = self.workspace_effects.parent_paths
            if paths:
                # Freeze collection and inode policy while the startup workspace
                # is still intact. The policy builder only uses the manifest keys.
                policy = effect_policy_for_run(
                    self.root, row, baseline_tests, baseline_manifest=dict.fromkeys(paths))
                policy = replace(policy, baseline_manifest=None)
            self._load_effect_policy = lambda: self._prepare_effect_policy(row, baseline_tests, policy)
            return
        else:
            self.baseline = self.workspace_effects.baseline()
        self.policy = (replace(policy, baseline_manifest=self.baseline) if policy is not None else
                       effect_policy_for_run(self.root, row, baseline_tests,
                                             baseline_manifest=self.baseline))

    @property
    def capability(self):
        self._validate_path_authority()
        return self._capability

    @capability.setter
    def capability(self, value):
        self._capability = value

    def _validate_path_authority(self) -> None:
        # All command purposes derive their capabilities here. Check the host's
        # frozen inputs before using them, including files reached by shebangs.
        snapshot = self._path_snapshot
        if snapshot is not None:
            try:
                current = current_path_authority_fingerprint(snapshot["inputs"])
            except OSError as error:
                raise ShellExecutionFailed(f"shell PATH authority changed: {error}") from error
            if current != snapshot["fingerprint"]:
                raise ShellExecutionFailed("shell PATH authority changed")

    def _prepare_effect_policy(self, row, baseline_tests, policy):
        self.baseline = self.workspace_effects.baseline()
        self.policy = (replace(policy, baseline_manifest=self.baseline) if policy is not None else
                       effect_policy_for_run(self.root, row, baseline_tests,
                                             baseline_manifest=self.baseline))
        self._load_effect_policy = None

    def _scan(self):
        if self.workspace_effects is not None:
            return self.workspace_effects.scan()
        return scan(self.root, exclude=lambda p: bool(self.environment and
                    (self.root / p).is_relative_to(self.environment.root)))

    def identity(self) -> str:
        if self.environment:
            if not self._environment_identity_epoch_enabled:
                return env_identity(self.environment)
            cached = self._environment_identity_cache
            if cached is not None and cached[0] == self._environment_identity_epoch:
                return cached[1]
            identity = env_identity(self.environment)
            self._environment_identity_cache = (self._environment_identity_epoch, identity)
            return identity
        selected = self.spec.get("capabilities", {}).get("test_python")
        if selected is None:
            return "disabled"
        python = Path(selected)
        metadata = python.stat()
        # Replay/cache identity follows selection, not the ephemeral shim path.
        value = ("selected-python-aliases-1", str(python), str(python.resolve(strict=True)),
                 metadata.st_dev, metadata.st_ino, metadata.st_size,
                 metadata.st_mtime_ns, metadata.st_ctime_ns)
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()

    def invalidate_environment_identity(self) -> None:
        """Start a fresh identity epoch after an operation that may mutate the run env."""
        self._environment_identity_epoch += 1
        self._environment_identity_cache = None

    def confirm(self, call_id: str, command: str, intent: dict, *, replay: bool) -> bool:
        tape = tape_for_run()
        decision_id = call_id + ":confirmation"
        decision_intent = {**intent, "tool": "shell_confirmation", "args": {"command": command},
                           "replay_class": "observation"}
        if replay:
            decision = tape.lookup(decision_id)
            reason = replay_decision(decision, replay_external_allowed=False).refuse_reason
            if not reason:
                reason = verify_intent(decision, **decision_intent)
            if reason:
                raise ToolReplayRefused(reason)
            if decision.observation not in {"approved", "declined"}:
                raise ToolReplayRefused("invalid recorded shell confirmation")
            return decision.observation == "approved"
        tape.intent(call_id=decision_id, **decision_intent)
        approved = request_confirmation(self.run_dir, call_id, command)
        tape.complete(call_id=decision_id, exit_code=None,
                      observation="approved" if approved else "declined",
                      post_manifest_digest=intent["pre_manifest_digest"],
                      env_identity=intent["env_identity"])
        return approved

    def execute(self, args: dict, call_id: str) -> ShellReply:
        try:
            from kernel_read_receipts import enabled
            if enabled():
                from kernel_shell_read_receipts import execute_with_read_audit
                return execute_with_read_audit(self, args, call_id)
            return self._execute(args, call_id)
        except (EffectError, RunEnvError, ToolTapeError, OSError) as error:
            self._clear_post_manifest()
            raise ShellExecutionFailed(f"{type(error).__name__}: {error}") from error

    def _execute(self, args: dict, call_id: str, *, custom=None) -> ShellReply:
        mcp_server = None
        if custom is None:
            try:
                admission = admit_shell(args.get("command"), self.settings)
            except ShellPolicyRefused as error:
                return ShellReply(f"REFUSED {error.code}: {error}")
            if args.get("background"):
                from kernel_agent_tools import service
                return service().background(args, call_id, admission)
            timeout = args.get("timeout_seconds", 120)
            if type(timeout) is not int or not 1 <= timeout <= 600:
                return ShellReply("REFUSED DENIED_PROGRAM: timeout_seconds must be an integer from 1 to 600")
            command, name = args["command"], "shell"
            argv = ["/bin/sh", "-c", command]
            cap = replace(self.capability, limits=replace(self.capability.limits, wall_seconds=timeout))
            if self.shell_state is not None:
                import kernel_shell_state
                state_dir = self.run_dir / "shell-home" / ".sf-shell-state"
                state_dir.mkdir(mode=0o700, exist_ok=True)
                if not self.shell_state.directory(self.root).is_dir():
                    # A previous command removed its directory: start this one at the workspace root.
                    self.shell_state = replace(self.shell_state, cwd=".")
                state_prefix = state_dir / kernel_shell_state._safe(call_id)
                argv = ["/bin/sh", "-c", kernel_shell_state.WRAPPER, "sf-shell",
                        str(self.shell_state.directory(self.root)), str(state_prefix), command]
                cap = replace(cap, env=self.shell_state.environment(cap.env))
                state_args = {**args, "_shell_state": self.shell_state.as_dict()}
            requires_confirmation = admission.requires_confirmation
            replay_class = "external" if cap.network == "full" or not cap.sandboxed else "local"
            if self.settings.shell == "read-only":
                replay_class = "observation"
        else:
            from kernel_custom_tools import CustomToolError
            mcp_server = getattr(custom, "server", None)
            # MCP owns its declared arguments. Only command tools receive an engine-added rationale.
            tool_args = args if mcp_server is not None else {k: v for k, v in args.items() if k != "why"}
            try:
                argv = custom.build_argv(tool_args)
            except CustomToolError as error:
                return ShellReply(f"REFUSED BAD_ARGUMENTS: {error}")
            try:
                custom.verify_executable()
            except CustomToolError as error:
                return ShellReply(f"REFUSED TOOL_CHANGED: {error}")
            name, command = custom.name, custom.name
            from kernel_exec_capability import executable_read_roots
            extra_roots = ()
            if mcp_server is not None:
                import sys
                extra_roots = (*extra_roots, *executable_read_roots(Path(sys.executable)),
                               *(Path(arg).resolve() for arg in mcp_server.argv[1:]
                                 if Path(arg).is_absolute() and Path(arg).is_file()))
            cap = replace(self.capability, purpose="custom_tool", network=custom.network,
                          sandboxed=True if mcp_server is not None else self.capability.sandboxed,
                          workspace_writable=custom.writes == "workspace",
                          read_roots=(*self.capability.read_roots, *executable_read_roots(custom.executable),
                                      *extra_roots),
                          limits=replace(self.capability.limits, wall_seconds=custom.timeout_seconds,
                                         output_bytes=4 << 20 if mcp_server is not None else self.capability.limits.output_bytes))
            requires_confirmation = False
            replay_class = custom.replay
        workspace_effects = self.workspace_effects
        if workspace_effects is None:
            if self._pending_baseline_root is not None:
                self.baseline_store.publish(self._pending_baseline_root)
                self._pending_baseline_root = None
            before = self._cached_post_manifest(cap)
            if before is None:
                before = self._scan()
                self.store.capture(self.root, before)
            before_tree = None
        else:
            before_tree, before = workspace_effects.before(call_id)
        identity = self.identity()
        # Runtime workspace names vary on replay. Bind the same semantic authority,
        # preserving every external read root and all resource/network settings.
        encoded = json.dumps(cap.as_dict(), sort_keys=True)
        for path, label in ((str(self.root), "<workspace>"), (str(self.run_dir), "<run>")):
            encoded = encoded.replace(path, label)
        stateful = custom is None and self.shell_state is not None
        intent = dict(tool=name, args=state_args if stateful else args, replay_class=replay_class,
                      capability_digest=hashlib.sha256(encoded.encode()).hexdigest(),
                      definition_digest=custom.definition_digest if custom else None, env_identity=identity,
                      pre_manifest_digest=workspace_identity(before))
        tape = tape_for_run()
        record = tape.lookup(call_id) if replay_mode_active() else None
        if replay_mode_active():
            approved = (self.confirm(call_id, command, intent, replay=True)
                        if requires_confirmation else True)
            if (record is not None and record.completed
                    and record.observation.startswith("REFUSED ASK_DECLINED:")):
                reason = verify_intent(record, **intent)
                if reason:
                    raise ToolReplayRefused(reason)
                if approved:
                    raise ToolReplayRefused("shell decline contradicts its confirmation record")
                return ShellReply(record.observation)
            if not approved:
                raise ToolReplayRefused("shell execution contradicts its declined confirmation")
            decision = replay_decision(record, replay_external_allowed=self.settings.replay_external)
            reason = decision.refuse_reason
            if not reason:
                reason = verify_intent(record, **intent)
            if reason:
                raise ToolReplayRefused(reason)
            if decision.serve_taped and not decision.reexecute:
                if stateful:
                    import kernel_shell_state
                    replayed = kernel_shell_state.load(self.run_dir, call_id)
                    if replayed is None:
                        raise ToolReplayRefused("shell state for a recorded call is missing")
                    self.shell_state = replayed
                return ShellReply(record.observation)
        else:
            tape.intent(call_id=call_id, **intent)
        if record is None and requires_confirmation and not self.confirm(call_id, command, intent, replay=False):
            observation = "REFUSED ASK_DECLINED: user declined shell command"
            tape.complete(call_id=call_id, exit_code=None, observation=observation,
                          post_manifest_digest=workspace_identity(before), env_identity=identity)
            return ShellReply(observation)
        try:
            if custom is not None:
                try:
                    custom.verify_executable()
                except CustomToolError as error:
                    observation = f"REFUSED TOOL_CHANGED: {error}"
                    if record is not None:
                        raise ToolReplayRefused(observation) from error
                    tape.complete(call_id=call_id, exit_code=None, observation=observation,
                                  post_manifest_digest=workspace_identity(before), env_identity=identity)
                    return ShellReply(observation)
            if self._load_effect_policy is not None:
                self._load_effect_policy()
            stdin = custom.request(tool_args) if custom is not None and mcp_server is not None else None
            execution = workspace_effects.execution(cap) if workspace_effects else nullcontext()
            with execution:
                if mcp_server is not None:
                    from kernel_mcp_driver import exchange
                    result = exchange(json.loads(stdin), cap, self.root)
                else:
                    result = self.session.run(argv, cap, cwd=self.root)
        except (ExecSessionError, OSError) as error:
            if "requires " in str(error) and "no unsandboxed fallback" in str(error):
                observation = f"REFUSED SANDBOX_UNAVAILABLE: {error}"
                completion = dict(exit_code=None, post_manifest_digest=workspace_identity(before),
                                  env_identity=identity)
                if record is not None:
                    reason = verify_reexecution(record, **completion)
                    if reason:
                        raise ToolReplayRefused(reason)
                    observation = record.observation
                else:
                    tape.complete(call_id=call_id, observation=observation, **completion)
                return ShellReply(observation)
            # A failed supervisor RPC may follow a partial execution. Keep its intent
            # incomplete, stop the run and require reconciliation on resume.
            self._clear_post_manifest()
            return ShellReply(f"REFUSED CLEANUP_FAILED: {error}", fatal=True)
        with tempfile.NamedTemporaryFile(prefix="shell-output-", suffix=".bin",
                                         dir=self.run_dir, delete=False) as raw:
            raw.write(result.output)
            output_path = Path(raw.name)
        output_path.with_suffix(".json").write_text(json.dumps({
            "call_id": call_id, "total_bytes": result.output_total_bytes,
            "retained_bytes": len(result.output), "truncated": result.output_truncated}))
        if not result.cleanup.ok:
            self._clear_post_manifest()
            return ShellReply(f"REFUSED CLEANUP_FAILED: {result.cleanup.detail}", fatal=True)
        container_backend = active().backend == BACKEND_CONTAINER
        if container_backend:
            post_manifest = self._scan()
            tampered = _container_tampered_protected_tests(
                self.baseline, post_manifest, self.protected_test_paths)
        else:
            post_manifest = None
            tampered = ()
        if workspace_effects is None:
            manifest = post_manifest if container_backend else self._scan()
            effects = reconcile(self.root, before, manifest, self.policy,
                                self.store, self.baseline_store)
        else:
            effects = workspace_effects.settle(before_tree, before, self.policy, call_id)
        if (self.environment and _command_may_change_environment(
                command, custom, effects, self.root, self.environment)):
            self.invalidate_environment_identity()
            after_identity = self.identity()
        else:
            after_identity = identity
        delivery_bytes = result.output
        mcp_refusal = None
        if custom is not None and mcp_server is not None and not result.timed_out:
            from kernel_mcp_tools import result_text
            try:
                if result.output_truncated or result.exit_code != 0:
                    raise ValueError("driver envelope exceeds output bound")
                envelope = json.loads(result.output)
                if not isinstance(envelope, dict) or type(envelope.get("ok")) is not bool:
                    raise ValueError("invalid driver envelope")
                if not envelope["ok"]:
                    code = envelope.get("code")
                    if code == "TIMEOUT":
                        mcp_refusal = "REFUSED TIMEOUT: "
                    else:
                        mcp_refusal = "REFUSED MCP_PROTOCOL_ERROR: "
                    delivery_bytes = str(envelope.get("error", "server protocol failed"))[:500].encode()
                else:
                    payload = envelope.get("result")
                    if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
                        raise ValueError("invalid tools/call envelope")
                    delivery_bytes = result_text(payload).encode()
                    if payload.get("isError") is True:
                        mcp_refusal = "REFUSED MCP_TOOL_ERROR: "
            except (UnicodeError, ValueError, TypeError, KeyError) as error:
                mcp_refusal = "REFUSED MCP_PROTOCOL_ERROR: "
                delivery_bytes = str(error)[:500].encode()
        replacements = {str(self.run_dir / "shell-home"): "<scratch>",
                        str(self.run_dir): "<run>"}
        home = safe_home()
        if home is not None:
            replacements[str(home)] = "<home>"
        output, omitted_bytes = normalize_output_delivery(delivery_bytes,
            total_bytes=len(delivery_bytes) if mcp_server is not None else result.output_total_bytes,
            replacements=replacements)
        observation = (f"$ {command[:200]}\nexit {'timeout' if result.timed_out else result.exit_code} | "
                       f"{result.wall_seconds:.2f}s | {self.settings.shell} | network {cap.network}\n")
        if custom is not None:
            observation = (f"{name} exit {'timeout' if result.timed_out else result.exit_code} | "
                           f"{result.wall_seconds:.2f}s\n")
        if mcp_server is not None:
            observation = (mcp_refusal or "") + observation + (
                f"external content from MCP server {mcp_server.name}; untrusted\n")
        if result.timed_out:
            observation = "REFUSED TIMEOUT: command exceeded its wall limit\n" + observation
        if any(reason == "outside edit scope" for _, reason in effects.restored):
            observation = "REFUSED OUTSIDE_EDIT_SCOPE: workspace write was restored\n" + observation
        if any(reason == "outside task file scope" for _, reason in effects.restored):
            observation = ("REFUSED OUTSIDE_TASK_SCOPE: the task says to change no other file; "
                           "workspace write was restored\n" + observation)
        if tampered:
            observation = _container_test_tamper_notice(tampered) + "\n" + observation
        elif any(reason.startswith("protected test tree:") for _, reason in effects.restored):
            observation = "REFUSED PROTECTED_TEST: workspace write was restored\n" + observation
        head = observation
        observation = ""
        if effects.applied:
            groups = []
            for kind in ("created", "modified", "deleted", "mode", "type"):
                paths = [e.path for e in effects.applied if e.change == kind]
                if paths:
                    groups.append(kind + ": " + ", ".join(paths))
            observation += "\nfiles changed by this command: " + bounded_summary("; ".join(groups))
        if effects.restored:
            observation += "\nrestored (not allowed): " + bounded_summary(", ".join(
                f"{e.path} ({reason})" for e, reason in effects.restored))
        if after_identity != identity:
            observation += f"\nenvironment changed: {after_identity[:12]} (check results from before are stale)"
        if stateful:
            import kernel_shell_state
            next_state, state_note = kernel_shell_state.capture(
                state_prefix, self.root, self.capability.env, self.shell_state)
            kernel_shell_state.store(self.run_dir, call_id, next_state)
            self.shell_state = next_state
            if next_state.cwd != ".":
                observation += f"\nshell cwd: {next_state.cwd}"
            if state_note:
                observation += f"\nshell state: {state_note}"
        from kernel_execution_reply_budget import record_delivery
        observation = record_delivery(self.run_dir, head, output, observation,
                                      omitted_bytes=omitted_bytes, call_id=call_id)
        completion = dict(exit_code=result.exit_code, post_manifest_digest=workspace_identity(effects.manifest),
                          env_identity=after_identity)
        if record is not None:
            reason = verify_reexecution(record, **completion)
            if reason:
                raise ToolReplayRefused(reason)
            observation = record.observation
        else:
            tape.complete(call_id=call_id, observation=observation, **completion)
        self._remember_post_manifest(effects, cap)
        return ShellReply(observation, tuple(effects.applied), after_identity != identity,
                          after_identity, exit_code=result.exit_code,
                          memory_limit_mechanism=getattr(
                              result, "memory_limit_mechanism", None),
                          memory_limit_killed=(
                              result.cleanup.detail == "aggregate resident-memory limit exceeded"),
                          cgroup_leaf_pattern=getattr(
                              result, "cgroup_leaf_pattern", None))


_active: ShellTool | None = None


def invalidate_active_environment_identity() -> None:
    """Discard any cached environment identity restored from a checkpoint."""
    if _active is not None:
        _active.invalidate_environment_identity()


def configure(run_dir: Path, settings: ToolSettings, spec: dict, session: ExecSession) -> None:
    global _active
    _active = ShellTool(run_dir, settings, spec, session)


def service() -> ShellTool | None:
    return _active


def publish_host_effects(runtime, bolt, created: set[str], *, changed, created_paths,
                         deleted=(), name="parallel") -> Path | None:
    """Publish committed host mutations through the same receipt and proof path as shell."""
    from kernel_post_edit_refresh import rebind_after_file_operation
    from kernel_green_lock import note_applied_product_edit

    paths = tuple(changed)
    created_paths = frozenset(created_paths)
    deleted = frozenset(deleted)
    target = runtime.target
    lock = getattr(runtime, "green_lock", None)
    cache_binding = bool(paths) and (
        (lock is not None and getattr(runtime, "stateframe_post_green_work_menu_mode", "off") == "on")
        or (getattr(lock, "mode", None) == "terminal"
            and getattr(lock, "_terminal_fallback_revision", None) is not None)
    )
    binding = None
    if cache_binding:
        try:
            binding = runtime.binding()
        except (AttributeError, LookupError, OSError, TypeError, UnicodeError, ValueError):
            cache_binding = False
    for relative in paths:
        path = Path(bolt.root) / relative
        if relative in deleted:
            created.discard(str(path))
            target = rebind_after_file_operation(runtime, relative, None)
        elif relative in created_paths:
            created.add(str(path))
        # The last notice decides what is pending, so it reads the binding again: a file may have
        # changed since the first one.
        if cache_binding and relative != paths[-1]:
            note_applied_product_edit(relative, binding=binding)
        else:
            note_applied_product_edit(relative)
    if paths:
        bolt.product_edits += 1
    if paths:
        bolt.last_test_exit = None
        bolt._last_green_revision = None
    if name:
        observation = f"subframe commit applied: {', '.join(paths)}" if paths else "subframe commit made no file changes"
        bolt.receipt("shell_effect" if paths else name, name, observation,
                     "APPLIED" if paths else "NO_OP", observation,
                     changed_paths=paths,
                     effects=tuple({"path": path, "change": "delete" if path in deleted else
                                    "create" if path in created_paths else "modify"} for path in paths))
    return target


def publish_effects(reply: ShellReply, runtime, bolt, created: set[str], *, name="shell") -> Path | None:
    """Publish an opaque observation and retire proof for every admitted mutation."""
    paths = tuple(effect.path for effect in reply.applied)
    if reply.memory_limit_mechanism:
        runtime.journal.emit(
            "shell_memory_enforcement",
            producer="kernel_shell_tool",
            memory_limit_mechanism=reply.memory_limit_mechanism,
            **({"cgroup_leaf_pattern": reply.cgroup_leaf_pattern}
               if isinstance(reply.cgroup_leaf_pattern, str) else {}),
            **({
                "memory_limit_killed": True,
                "memory_limit_reason": "aggregate resident-memory limit exceeded",
            } if reply.memory_limit_killed else {}),
            dispatch=False,
        )
    target = publish_host_effects(runtime, bolt, created, changed=paths,
                                  created_paths=tuple(effect.path for effect in reply.applied
                                                      if effect.after is not None and effect.after.kind == "file"
                                                      and effect.before is None),
                                  deleted=tuple(effect.path for effect in reply.applied if effect.after is None),
                                  name="")
    if reply.environment_changed:
        bolt.last_test_exit = None
        bolt._last_green_revision = None
    if reply.environment_changed:
        runtime.journal.manifest["environment_revision"] = reply.environment_identity
        runtime._invalidate_binding_cache()
        runtime.journal.emit("shell_environment_changed", identity=reply.environment_identity,
                             detail="Installed for this run; update dependency declarations to keep installs.")
    result = "APPLIED" if paths or reply.environment_changed else (
        "REFUSED" if reply.observation.startswith("REFUSED ") else "NO_OP")
    bolt.receipt("shell_effect" if paths or reply.environment_changed else name,
                 name, reply.observation, result, reply.observation,
                 changed_paths=paths,
                 effects=tuple({"path": e.path, "change": e.change} for e in reply.applied),
                 exit_code=(reply.exit_code if name == "shell" and
                            getattr(runtime, "continuation", None) is not None else None))
    return target
