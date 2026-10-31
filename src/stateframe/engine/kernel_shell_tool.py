"""Foreground shell: capability admission, transactional effects and durable replay."""
from __future__ import annotations

import hashlib
import json
import time
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from kernel_effects import ContentStore, EffectError, effect_policy_for_run, manifest_digest, reconcile, scan
from kernel_exec_capability import shell_capability
from kernel_exec_session import ExecSession, ExecSessionError
from kernel_run_env import RunEnv, RunEnvError, env_identity
from kernel_shell_policy import ShellPolicyRefused, admit_shell, normalize_output
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
            network=self.settings.shell_network if writable else "none", writable=writable,
            extra_read_roots=tuple(Path(p) for p in self.spec.get("shell_read_roots", ())),
            path_dirs=tuple(Path(p) for p in self.spec.get("shell_path_dirs", ())))
        environment = dict(cap.env)
        if self.environment:
            environment.update({key: value for key, value in self.environment.env.items() if key != "PATH"})
        if not writable:
            # Inspection resolves trusted host tools, never executables installed by a prior command.
            environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        self.capability = replace(cap, env=environment,
            read_roots=(*cap.read_roots, env_root),
            write_roots=cap.write_roots if writable else (home,),
            sandboxed=self.settings.host_authorization is None)
        self.session.configure(self.capability)
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

    def _execute(self, args: dict, call_id: str) -> ShellReply:
        try:
            admission = admit_shell(args.get("command"), self.settings)
        except ShellPolicyRefused as error:
            return ShellReply(f"REFUSED {error.code}: {error}")
        if args.get("background"):
            return ShellReply("REFUSED JOB_LIMIT: background jobs are unavailable in this foreground tool")
        timeout = args.get("timeout_seconds", 120)
        if type(timeout) is not int or not 1 <= timeout <= 600:
            return ShellReply("REFUSED DENIED_PROGRAM: timeout_seconds must be an integer from 1 to 600")
        command = args["command"]
        cap = replace(self.capability, limits=replace(self.capability.limits, wall_seconds=timeout))
        before = self._scan()
        self.store.capture(self.root, before)
        identity = self.identity()
        replay_class = "external" if cap.network == "full" or not cap.sandboxed else "local"
        # Runtime workspace names vary on replay. Bind the same semantic authority,
        # preserving every external read root and all resource/network settings.
        encoded = json.dumps(cap.as_dict(), sort_keys=True)
        for path, label in ((str(self.root), "<workspace>"), (str(self.run_dir), "<run>")):
            encoded = encoded.replace(path, label)
        intent = dict(tool="shell", args=args, replay_class=replay_class,
                      capability_digest=hashlib.sha256(encoded.encode()).hexdigest(),
                      definition_digest=None, env_identity=identity,
                      pre_manifest_digest=manifest_digest(before))
        tape = tape_for_run()
        record = tape.lookup(call_id) if replay_mode_active() else None
        if replay_mode_active():
            approved = (self.confirm(call_id, command, intent, replay=True)
                        if admission.requires_confirmation else True)
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
        else:
            tape.intent(call_id=call_id, **intent)
        if record is None and admission.requires_confirmation and not self.confirm(call_id, command, intent, replay=False):
            observation = "REFUSED ASK_DECLINED: user declined shell command"
            tape.complete(call_id=call_id, exit_code=None, observation=observation,
                          post_manifest_digest=manifest_digest(before), env_identity=identity)
            return ShellReply(observation)
        try:
            result = self.session.run(["/bin/sh", "-c", command], cap, cwd=self.root)
        except (ExecSessionError, OSError) as error:
            if "requires " in str(error) and "no unsandboxed fallback" in str(error):
                observation = f"REFUSED SANDBOX_UNAVAILABLE: {error}"
                completion = dict(exit_code=None, post_manifest_digest=manifest_digest(before),
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
        output = normalize_output(result.output, total_bytes=result.output_total_bytes,
            replacements={str(self.run_dir / "shell-home"): "<scratch>",
                          str(self.run_dir): "<run>", str(Path.home()): "<home>"})
        observation = (f"$ {command[:200]}\nexit {'timeout' if result.timed_out else result.exit_code} | "
                       f"{result.wall_seconds:.2f}s | {self.settings.shell} | network {cap.network}\n{output}")
        if result.timed_out:
            observation = "REFUSED TIMEOUT: command exceeded its wall limit\n" + observation
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
        completion = dict(exit_code=result.exit_code, post_manifest_digest=manifest_digest(effects.manifest),
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


def publish_effects(reply: ShellReply, runtime, bolt, created: set[str]) -> Path | None:
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
    bolt.receipt("shell_effect" if paths or reply.environment_changed else "shell",
                 "shell", reply.observation, result, reply.observation,
                 changed_paths=paths,
                 effects=tuple({"path": e.path, "change": e.change} for e in reply.applied))
    return target
