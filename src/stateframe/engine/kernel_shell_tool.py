"""Foreground shell: capability admission, transactional effects and durable replay."""
from __future__ import annotations

import copy
import hashlib
import json
import time
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from kernel_effects import ContentStore, EffectError, Manifest, effect_policy_for_run, manifest_digest, reconcile, scan
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession, ExecSessionError
from kernel_run_env import RunEnv, RunEnvError, env_identity
from kernel_shell_policy import ShellPolicyRefused, admit_shell, normalize_output_delivery
from kernel_tool_settings import ToolSettings
from kernel_tool_tape import (ToolReplayRefused, ToolTapeError, replay_decision, replay_mode_active,
                              tape_for_run, verify_intent, verify_reexecution)

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
    data = text.encode("utf-8")
    if len(data) <= 4096:
        return text
    return data[:4096].decode("utf-8", errors="ignore") + f" [... {len(data) - 4096} bytes omitted ...]"


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
        self.root = None

    def bind(self, workspace: Path, row: dict, baseline_tests: set[str], *, policy=None) -> None:
        if self.root is not None:
            raise RuntimeError("shell workspace is already bound")
        self.root = workspace.resolve(strict=True)
        home = self.run_dir / "shell-home"
        home.mkdir(mode=0o700, exist_ok=True)
        env_root = self.environment.root if self.environment else self.run_dir / "shell-cache"
        env_root.mkdir(mode=0o700, exist_ok=True)
        writable = self.settings.shell != "read-only"
        cap = shell_capability(workspace=self.root, run_env_dir=env_root, scratch_home=home,
            network=self.settings.shell_network, writable=writable,
            extra_read_roots=tuple(Path(p) for p in self.spec.get("shell_read_roots", ())),
            path_dirs=tuple(Path(p) for p in self.spec.get("shell_path_dirs", ())))
        environment = dict(cap.env)
        if self.environment:
            environment.update({key: value for key, value in self.environment.env.items() if key != "PATH"})
        if not writable:
            # Inspection resolves trusted host tools, never executables installed by a prior command.
            environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        self.capability = replace(cap, env=environment,
            network=self.settings.shell_network if writable else "none",
            read_roots=(*cap.read_roots, env_root),
            write_roots=cap.write_roots if writable else (home,),
            sandboxed=self.settings.host_authorization is None)
        # The session envelope admits explicitly trusted custom capabilities;
        # each actual command still receives its narrower per-tool capability.
        self.session.configure(replace(self.capability, network=self.settings.shell_network))
        self.store = ContentStore(self.run_dir)
        self.baseline_store = ContentStore(self.run_dir / "shell-baseline")
        baseline_root = Path(self.spec.get("parent_snapshot", self.root))
        self.baseline = scan(baseline_root, exclude=lambda _: False)
        self.baseline_store.capture(baseline_root, self.baseline)
        self.policy = (replace(policy, baseline_manifest=self.baseline) if policy is not None else
                       effect_policy_for_run(self.root, row, baseline_tests,
                                             baseline_manifest=self.baseline))

    def _scan(self):
        return scan(self.root, exclude=lambda p: bool(self.environment and
                    (self.root / p).is_relative_to(self.environment.root)))

    def identity(self) -> str:
        return env_identity(self.environment) if self.environment else "disabled"

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
            return self._execute(args, call_id)
        except (EffectError, RunEnvError, ToolTapeError, OSError) as error:
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
            requires_confirmation = admission.requires_confirmation
            replay_class = "external" if cap.network == "full" or not cap.sandboxed else "local"
            if self.settings.shell == "read-only":
                replay_class = "observation"
        else:
            from kernel_custom_tools import CustomToolError
            try:
                argv = custom.build_argv({k: v for k, v in args.items() if k != "why"})
            except CustomToolError as error:
                return ShellReply(f"REFUSED BAD_ARGUMENTS: {error}")
            try:
                custom.verify_executable()
            except CustomToolError as error:
                return ShellReply(f"REFUSED TOOL_CHANGED: {error}")
            name, command = custom.name, custom.name
            from kernel_exec_capability import executable_read_roots
            mcp_server = getattr(custom, "server", None)
            extra_roots = (executable_read_roots(Path(__file__).with_name("kernel_mcp_driver.py"))
                           if mcp_server is not None else ())
            if mcp_server is not None:
                import sys
                extra_roots = (*extra_roots, *executable_read_roots(Path(sys.executable)),
                               *(Path(arg).resolve() for arg in mcp_server.argv[1:]
                                 if Path(arg).is_absolute() and Path(arg).is_file()))
            cap = replace(self.capability, purpose="custom_tool", network=custom.network,
                          workspace_writable=custom.writes == "workspace",
                          read_roots=(*self.capability.read_roots, *executable_read_roots(custom.executable),
                                      *extra_roots),
                          limits=replace(self.capability.limits, wall_seconds=custom.timeout_seconds,
                                         output_bytes=4 << 20 if mcp_server is not None else self.capability.limits.output_bytes))
            requires_confirmation = False
            replay_class = custom.replay
        before = self._scan()
        self.store.capture(self.root, before)
        identity = self.identity()
        # Runtime workspace names vary on replay. Bind the same semantic authority,
        # preserving every external read root and all resource/network settings.
        encoded = json.dumps(cap.as_dict(), sort_keys=True)
        for path, label in ((str(self.root), "<workspace>"), (str(self.run_dir), "<run>")):
            encoded = encoded.replace(path, label)
        intent = dict(tool=name, args=args, replay_class=replay_class,
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
            stdin = custom.request({k: v for k, v in args.items() if k != "why"}) if custom is not None and mcp_server is not None else None
            result = (self.session.run(argv, cap, stdin=stdin, cwd=self.root) if stdin is not None
                      else self.session.run(argv, cap, cwd=self.root))
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
            return ShellReply(f"REFUSED CLEANUP_FAILED: {error}", fatal=True)
        with tempfile.NamedTemporaryFile(prefix="shell-output-", suffix=".bin",
                                         dir=self.run_dir, delete=False) as raw:
            raw.write(result.output)
            output_path = Path(raw.name)
        output_path.with_suffix(".json").write_text(json.dumps({
            "call_id": call_id, "total_bytes": result.output_total_bytes,
            "retained_bytes": len(result.output), "truncated": result.output_truncated}))
        if not result.cleanup.ok:
            return ShellReply(f"REFUSED CLEANUP_FAILED: {result.cleanup.detail}", fatal=True)
        effects = reconcile(self.root, before, self._scan(), self.policy,
                            self.store, self.baseline_store)
        after_identity = self.identity()
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
        output, omitted_bytes = normalize_output_delivery(delivery_bytes,
            total_bytes=len(delivery_bytes) if mcp_server is not None else result.output_total_bytes,
            replacements={str(self.run_dir / "shell-home"): "<scratch>",
                          str(self.run_dir): "<run>", str(Path.home()): "<home>"})
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
        return ShellReply(observation, tuple(effects.applied), after_identity != identity, after_identity)


_active: ShellTool | None = None


def configure(run_dir: Path, settings: ToolSettings, spec: dict, session: ExecSession) -> None:
    global _active
    _active = ShellTool(run_dir, settings, spec, session)


def service() -> ShellTool | None:
    return _active


def publish_effects(reply: ShellReply, runtime, bolt, created: set[str], *, name="shell") -> Path | None:
    """Publish an opaque observation and retire proof for every admitted mutation."""
    from kernel_post_edit_refresh import rebind_after_file_operation
    from kernel_green_lock import note_applied_product_edit

    paths = tuple(effect.path for effect in reply.applied)
    target = runtime.target
    for effect in reply.applied:
        path = Path(bolt.root) / effect.path
        if effect.after is None:
            created.discard(str(path))
            target = rebind_after_file_operation(runtime, effect.path, None)
        elif effect.after.kind == "file" and effect.before is None:
            created.add(str(path))
        note_applied_product_edit(path)
    if paths:
        bolt.product_edits += 1
    if paths or reply.environment_changed:
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
                 effects=tuple({"path": e.path, "change": e.change} for e in reply.applied))
    return target
