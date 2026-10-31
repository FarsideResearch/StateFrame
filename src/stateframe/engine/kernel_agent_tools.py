"""Per-run tool registration and adapters for jobs, Git, web and custom commands."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from dataclasses import replace

from kernel_exec_session import ExecSessionError
from kernel_agent_jobs import AgentJobs
from kernel_agent_schemas import BUILTIN_SCHEMAS, JOB_NAMES, OBSERVATION_NAMES
from kernel_git_view import GitView, excluded_path, handle_git_call
from kernel_shell_tool import ShellReply, ShellCleanupFailed, workspace_identity
from kernel_tool_tape import (replay_mode_active, tape_for_run, replay_decision,
                              verify_intent, ToolReplayRefused)
from kernel_web_fetch import WebPolicy, fetch


class AgentTools:
    def __init__(self, shell, custom, mcp_servers=(), *, resume_active=False):
        from kernel_completion_owner import TOOL_SPECS
        from kernel_lean import _KNOWN_TOOL_NAMES
        from kernel_custom_tools import CustomToolError
        reserved = set(TOOL_SPECS) | set(_KNOWN_TOOL_NAMES)
        collisions = sorted(tool.name for tool in custom.tools if tool.name in reserved)
        if collisions:
            raise CustomToolError("custom tools duplicate built-in capabilities: " + ", ".join(collisions))
        self.shell = shell
        self.custom = {tool.name: tool for tool in custom.tools}
        self.mcp_servers = tuple(mcp_servers)
        self.resume_active = resume_active
        self.custom_schemas = copy.deepcopy(custom.schemas())
        for schema in self.custom_schemas:
            schema["function"]["parameters"]["properties"].setdefault("why", {"type": "string"})
        wall = os.environ.get("STATEFRAME_RUN_WALL_SECONDS")
        self.deadline = time.monotonic() + float(wall) if wall else float("inf")
        self.jobs = AgentJobs(shell, self.deadline)
        self.git = None
        source = shell.spec.get("repo")
        if shell.settings.git_tool and source:
            metadata = shell.run_dir / "git-view-metadata.json"
            if metadata.exists():
                pinned = json.loads(metadata.read_text())
                directory = Path(pinned["git_dir"]).resolve(strict=True)
                if not directory.is_relative_to(shell.run_dir.resolve()):
                    raise ValueError("pinned Git view is outside this run")
                if (directory / "HEAD").read_text().strip() != pinned["head_oid"]:
                    raise ValueError("pinned Git HEAD differs from its run record")
                self.git = GitView(git_dir=directory, workspace=shell.run_dir / "workspace",
                    baseline=shell.run_dir / "parent", head_oid=pinned["head_oid"],
                    object_dirs=tuple(Path(p) for p in pinned["object_dirs"]), secret_filter=excluded_path)
            else:
                directory = Path(tempfile.mkdtemp(prefix="git-attempt-", dir=shell.run_dir))
                self.git = GitView.build(Path(source), directory, secret_filter=excluded_path)
                if self.git is not None:
                    with metadata.open("x") as record:
                        json.dump({"head_oid": self.git.head_oid, "git_dir": str(self.git.git_dir),
                                   "object_dirs": [str(p) for p in self.git.object_dirs]}, record, sort_keys=True)
        self.names = OBSERVATION_NAMES | self.custom.keys()
        from kernel_agent_schemas import measure_schemas
        self.schema_budget = measure_schemas(self.custom_schemas, endpoint=shell.spec.get("chat_url"))
        with tempfile.NamedTemporaryFile(mode="w", prefix="tool-schema-budget-", suffix=".json",
                                         dir=shell.run_dir, delete=False) as report:
            json.dump(self.schema_budget, report, sort_keys=True)
            self.schema_budget_path = Path(report.name)

    def bind(self):
        if self.mcp_servers:
            from kernel_mcp_tools import discover
            from kernel_custom_tools import CustomToolError
            if self.resume_active and not self.shell.settings.replay_external:
                external = [server.name for server in self.mcp_servers if server.replay == "external"]
                if external:
                    raise ToolReplayRefused(
                        "MCP discovery requires --replay-external for servers: " + ", ".join(external))
            found = discover(self.mcp_servers, self.shell.root, self.shell.run_dir,
                             shell_network=self.shell.settings.shell_network,
                             host_authorization=self.shell.settings.host_authorization is not None,
                             custom_schemas=self.custom_schemas, endpoint=self.shell.spec.get("chat_url"))
            if (found.digest != self.shell.spec.get("mcp_tools_digest")
                    or [t.name for t in found.tools] != self.shell.spec.get("mcp_tools")
                    or list(found.omitted) != self.shell.spec.get("mcp_omitted")):
                raise CustomToolError("MCP tool list or schema changed since host validation")
            self.custom.update({tool.name: tool for tool in found.tools})
            self.custom_schemas.extend(copy.deepcopy(found.schemas()))
            self.names = OBSERVATION_NAMES | self.custom.keys()
            from kernel_agent_schemas import measure_schemas
            self.schema_budget = measure_schemas(self.custom_schemas, endpoint=self.shell.spec.get("chat_url"))
            self.schema_budget_path.write_text(json.dumps(self.schema_budget, sort_keys=True))
        if self.git is not None:
            self.git.workspace = self.shell.root
            self.git.baseline = Path(self.shell.spec.get("parent_snapshot", self.shell.root))

    def offered(self, name):
        if name in JOB_NAMES:
            return bool(self.jobs.jobs)
        if name == "git":
            return self.git is not None and self.shell.settings.git_tool
        if name == "web_fetch":
            return self.shell.settings.web
        return name in self.custom

    def schemas(self):
        return [copy.deepcopy(s) for s in (*BUILTIN_SCHEMAS, *self.custom_schemas)
                if self.offered(s["function"]["name"])]

    def observation(self, name, args, call_id, operation, *, definition=None, capability=None):
        before = workspace_identity(self.shell._scan())
        identity = self.shell.identity()
        authority = {"settings": self.shell.settings.as_dict(), "purpose": name}
        authority["settings"].pop("replay_external", None)
        if capability is None and name != "web_fetch":
            capability = replace(self.shell.capability, purpose="job", workspace_writable=False, write_roots=())
        if capability is not None:
            authority["capability"] = capability.as_dict()
        encoded = json.dumps(authority, sort_keys=True)
        replacements = [(str(self.shell.root), "<workspace>"), (str(self.shell.run_dir), "<run>")]
        if self.git is not None:
            replacements.insert(0, (str(self.git.git_dir), "<git-view>"))
        for path, label in replacements:
            encoded = encoded.replace(path, label)
        intent = dict(tool=name, args=args, replay_class="observation",
            capability_digest=hashlib.sha256(encoded.encode()).hexdigest(),
            definition_digest=definition, env_identity=identity, pre_manifest_digest=before)
        tape = tape_for_run()
        if replay_mode_active():
            record = tape.lookup(call_id)
            reason = replay_decision(record, replay_external_allowed=False).refuse_reason
            if not reason:
                reason = verify_intent(record, **intent)
            if reason:
                raise ToolReplayRefused(reason)
            return record.observation
        tape.intent(call_id=call_id, **intent)
        try:
            result = operation()
        except ExecSessionError as error:
            if "requires " in str(error) and "no unsandboxed fallback" in str(error):
                result = f"REFUSED SANDBOX_UNAVAILABLE: {error}"
            else:
                raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {error}") from error
        tape.complete(call_id=call_id, exit_code=None, observation=result,
                      post_manifest_digest=before, env_identity=identity)
        return result

    def background(self, args, call_id, admission):
        try:
            return self._background(args, call_id, admission)
        except ShellCleanupFailed as error:
            return ShellReply(str(error), fatal=True)

    def _background(self, args, call_id, admission):
        def launch():
            if admission.requires_confirmation:
                from kernel_shell_tool import request_confirmation
                if not request_confirmation(self.shell.run_dir, call_id, args["command"]):
                    return "REFUSED ASK_DECLINED: user declined shell command"
            return self.jobs.launch(args)
        timeout = args.get("timeout_seconds", 600)
        if type(timeout) is not int or not 1 <= timeout <= 600:
            return ShellReply("REFUSED DENIED_PROGRAM: timeout_seconds must be an integer from 1 to 600")
        capability = self.jobs.capability(f"j{len(self.jobs.jobs)+1}", timeout)
        observation = self.observation("shell", args, call_id, launch, capability=capability)
        if replay_mode_active():
            self.jobs.replay_launch(args, observation)
        return ShellReply(observation)

    def snapshot(self):
        def observe(sequence, build):
            value = self.observation("job_snapshot", {"sequence": sequence}, f"job-snapshot-{sequence}",
                                     lambda: json.dumps(build(), sort_keys=True))
            snapshot = json.loads(value)
            if replay_mode_active():
                for row in snapshot["jobs"]:
                    job = self.jobs.jobs.get(row["id"])
                    if job is not None and job.handle is None and row["state"] != "running":
                        job.state, job.exit_code = row["state"], row["exit_code"]
            return snapshot
        return self.jobs.snapshot(observe)

    def stop_jobs(self, call_id):
        if not self.jobs.jobs:
            return ""
        result = self.observation("job_stop_all", {}, call_id + ":stop-jobs", self.jobs.stop_all)
        if replay_mode_active():
            for job in self.jobs.jobs.values():
                if job.handle is None and job.state == "lost on restart":
                    job.state = "stopped"
        return result

    def execute(self, name, args, call_id):
        try:
            return self._execute(name, args, call_id)
        except ShellCleanupFailed as error:
            return ShellReply(str(error), fatal=True)

    def _execute(self, name, args, call_id):
        if name in self.custom:
            from kernel_effects import EffectError
            from kernel_run_env import RunEnvError
            from kernel_tool_tape import ToolTapeError
            from kernel_shell_tool import ShellExecutionFailed
            try:
                return self.shell._execute(args, call_id, custom=self.custom[name])
            except (EffectError, RunEnvError, ToolTapeError, OSError) as error:
                raise ShellExecutionFailed(f"{type(error).__name__}: {error}") from error
        if name == "git" and not self.offered(name):
            return ShellReply("REFUSED NOT_A_GIT_REPO: Git is disabled or the source is not a Git repository")
        if name == "web_fetch" and not self.offered(name):
            return ShellReply("REFUSED WEB_HTTP_ERROR: web_fetch is disabled; enable --web")
        def run():
            if name in JOB_NAMES:
                return self.jobs.execute(name, args)
            if name == "git":
                return self._git(args)
            if name == "web_fetch":
                return self._web(args)
            raise ValueError(f"unregistered host tool: {name}")
        observation = self.observation(name, args, call_id, run,
                          definition=self.git.head_oid if name == "git" and self.git else None,
                          capability=self._git_capability() if name == "git" else None)
        if replay_mode_active() and name == "job_stop" and args.get("job") in self.jobs.jobs:
            self.jobs.jobs[args["job"]].state = "stopped"
        return ShellReply(observation)

    def _git_capability(self):
        roots = (*self.shell.capability.read_roots, *self.git.object_dirs, self.git.git_dir)
        return replace(self.shell.capability, purpose="git", workspace_writable=False,
                       network="none", write_roots=(), read_roots=tuple(set(roots)),
                       env={**self.shell.capability.env, **self.git.env()})

    def _git(self, args):
        params = {k: v for k, v in args.items() if k != "why"}
        action = params.get("action")
        if action == "show" and isinstance(params.get("rev"), str) and ":" in params["rev"]:
            if "path" in params:
                return "REFUSED BAD_ARGUMENT: supply path once"
            params["rev"], params["path"] = params["rev"].split(":", 1)
        if action == "diff" and "path" in params:
            params["paths"] = [params.pop("path")]
        def runner(argv, env, roots):
            cap = self._git_capability()
            result = self.shell.session.run(argv, cap, cwd=self.shell.root)
            if not result.cleanup.ok:
                raise ShellCleanupFailed(f"REFUSED CLEANUP_FAILED: {result.cleanup.detail}")
            return result.exit_code, result.output
        text, meta = handle_git_call(self.git, runner, params)
        if meta["ok"]:
            from kernel_shell_policy import normalize_output
            return normalize_output(text.encode()) or "No changes."
        detail = meta.get("detail", text)
        if "excluded" in detail:
            code = "EXCLUDED_PATH"
        elif "revision" in detail or "unknown revision" in detail or "ambiguous argument" in detail or "invalid object name" in detail:
            code = "BAD_REVISION"
        elif meta.get("refusal") == "object_unavailable":
            code = "MISSING_OBJECT"
        else:
            code = "BAD_ARGUMENT"
        return f"REFUSED {code}: {detail}"

    def _web(self, args):
        size = args.get("max_chars", 12000)
        if not isinstance(args.get("url"), str) or type(size) is not int or not 1 <= size <= 12000:
            return "REFUSED WEB_HTTP_ERROR: url must be text and max_chars an integer from 1 to 12000"
        policy = WebPolicy(tuple(self.shell.settings.web_allow), max_chars=size,
                           deadline=max(0.001, min(20, self.deadline-time.monotonic())))
        result = fetch(args["url"], policy)
        if result.ok and result.status is not None and 200 <= result.status < 300:
            return result.text
        reason = result.refusal or f"HTTP status {result.status}"
        lower = reason.lower()
        if any(word in lower for word in ("private", "non-public", "non-global", "address")):
            code = "WEB_REFUSED_ADDRESS"
        elif "scheme" in lower or "http or https" in lower:
            code = "WEB_REFUSED_SCHEME"
        elif lower.startswith("port ") or "invalid url: port" in lower:
            code = "WEB_REFUSED_PORT"
        elif "domain" in lower or "web allow list" in lower:
            code = "WEB_DOMAIN_NOT_ALLOWED"
        elif any(word in lower for word in ("limit", "too large", "exceeds")):
            code = "WEB_TOO_LARGE"
        elif any(word in lower for word in ("content type", "charset", "encoding")):
            code = "WEB_CONTENT_TYPE"
        elif "timeout" in lower or "timed out" in lower or "deadline" in lower:
            code = "WEB_TIMEOUT"
        else:
            code = "WEB_HTTP_ERROR"
        return f"REFUSED {code}: {reason}"


_active: AgentTools | None = None


def configure(shell, custom, mcp_servers=(), *, resume_active=False):
    global _active
    candidate = AgentTools(shell, custom, mcp_servers, resume_active=resume_active)
    # Publish only after all definitions and the source view validate. Dynamic
    # dispatch and schema consumers resolve this same per-run object.
    _active = candidate


def service():
    return _active


def is_registered(name):
    return _active is not None and name in _active.names


def project_menu(tools):
    if _active is None:
        return [tool for tool in tools if tool["function"]["name"] not in OBSERVATION_NAMES]
    selected = [tool for tool in tools if tool["function"]["name"] not in _active.names]
    return [*selected, *_active.schemas()]
