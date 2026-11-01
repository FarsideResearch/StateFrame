"""Child-side hook state at action, request, and submission boundaries."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from kernel_effects import effect_policy_for_run
from kernel_hook_exec import HookRunner
from kernel_hooks import PanelEntry, event_id, render_panel, sanitize

_ACTIVE = None


def active_wiring():
    if _ACTIVE is None:
        raise RuntimeError("hook checkpoint restored without an active runner")
    return _ACTIVE


class HookBlockedError(RuntimeError):
    """A submission was vetoed where no model action can recover it."""


@dataclass(frozen=True)
class DispatchOutcome:
    status: str
    executed: bool
    exit_code: int | None
    applied_paths: tuple[str, ...] | None
    deleted_paths: tuple[str, ...] = ()


def _product_version() -> str:
    """Read the adjacent product version without importing its package in the child."""
    source = Path(__file__).resolve().parent.parent / "__init__.py"
    for statement in ast.parse(source.read_text(encoding="utf-8")).body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "__version__"
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            return statement.value.value
    raise RuntimeError("product version is unavailable for hook payloads")


class HookWiring:
    def __init__(self, run_dir, hooks, spec, *, replay):
        global _ACTIVE
        self.run_dir = Path(run_dir)
        self.hooks = hooks
        self.spec = spec
        self.replay = replay
        self.runner = None
        self.runtime = None
        self.protected_paths = hooks.protected_repo_paths(Path(spec["repo"]))
        self.state = {
            "stop_count": 0,
            "block_counts": {},
            "pending_notes": {},
            "pending_ids": {},
            "consumed_ids": [],
            "gate": None,
        }
        raw_context = spec["hooks"].get("start_context", ())
        if (
            not isinstance(raw_context, (list, tuple))
            or len(raw_context) > 16
            or any(
                type(entry) is not dict
                or set(entry) != set(PanelEntry.__dataclass_fields__)
                or any(type(value) is not str for value in entry.values())
                or len(entry["text"].encode("utf-8")) > 2048
                for entry in raw_context
            )
        ):
            raise ValueError("invalid pinned hook start context")
        self.start_context = tuple(PanelEntry(**entry) for entry in raw_context)
        start_ids = []
        for entry in self.start_context:
            match = next(
                (
                    index
                    for index, hook in enumerate(hooks.definitions)
                    if hook.name == entry.hook and hook.event == entry.event
                ),
                None,
            )
            if (
                match is None
                or entry.kind != "start"
                or entry.event not in {"session_start", "prompt_submit"}
            ):
                raise ValueError("hook start context has no matching start hook")
            start_ids.append(event_id(spec["run_identity_seed"], entry.event, match))
        self.start_ids = tuple(start_ids)
        self._actions = {}
        self._last_panel = None
        self._panel_admission = None
        _ACTIVE = self

    def __reduce__(self):
        # Checkpoints restore the live child runner and refill only its serializable state.
        return active_wiring, ()

    def bind(self, root, row, baseline_tests, shell, runtime):
        if self.runner is not None:
            raise RuntimeError("hook workspace is already bound")
        policy = (
            shell.policy
            if shell is not None and getattr(shell, "root", None) is not None
            else effect_policy_for_run(
                root, row, baseline_tests, hook_paths=self.protected_paths
            )
        )
        self.runner = HookRunner(
            self.run_dir,
            self.hooks,
            bundle_dir=self.run_dir / "hook-bundle",
            workspace=root,
            shell_network=self.spec["tool_settings"]["shell_network"],
            run_seed=self.spec["run_identity_seed"],
            run_id=runtime.journal.run_id,
            version=_product_version(),
            replay=self.replay,
            effect_policy=policy,
            protected_paths=self.protected_paths,
        )
        self.runtime = runtime
        runtime._hook_wiring = self
        runtime.journal.emit(
            "user_hook_config",
            digest=self.hooks.digest,
            count=len(self.hooks.definitions),
            dispatch=False,
        )

    def _record_runs(self, result):
        for run in result.runs:
            self.runtime.journal.emit(
                "user_hook_run",
                hook_event_id=run.event_id,
                hook=run.hook,
                hook_event=run.event,
                decision=run.outcome.decision,
                error_code=run.outcome.error_code,
                replayed=run.replayed,
                skipped=run.skipped,
                dispatch=False,
            )

    def protected(self, path):
        if self.runner is None:
            raise RuntimeError("hook workspace is not bound")
        try:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.runner.workspace / candidate
            relative = (
                candidate.absolute().relative_to(self.runner.workspace).as_posix()
            )
        except ValueError:
            return False
        if relative in self.protected_paths:
            return True
        try:
            resolved = candidate.resolve().relative_to(self.runner.workspace).as_posix()
        except (OSError, ValueError):
            return False
        if resolved in self.protected_paths:
            return True
        try:
            info = os.lstat(candidate)
        except (FileNotFoundError, OSError):
            return False
        if not stat.S_ISREG(info.st_mode):
            return False
        for name in self.protected_paths:
            try:
                protected = os.lstat(self.runner.workspace / name)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(protected.st_mode) and (info.st_dev, info.st_ino) == (
                protected.st_dev,
                protected.st_ino,
            ):
                return True
        return False

    def admitted(self, name, args, shell, agent_tools):
        if name == "shell" and shell is not None:
            from kernel_shell_policy import ShellPolicyRefused, admit_shell

            try:
                admit_shell(args.get("command"), shell.settings)
            except ShellPolicyRefused:
                return False
            timeout = args.get("timeout_seconds", 120)
            return type(timeout) is int and 1 <= timeout <= 600
        custom = (
            agent_tools.custom.get(name)
            if agent_tools is not None and name in agent_tools.custom
            else None
        )
        if custom is not None:
            from kernel_custom_tools import CustomToolError

            # Admit with the arguments dispatch will use: MCP owns its declared arguments (including any
            # `why`), command tools get the engine-added rationale removed. A mismatch would send an
            # MCP call down the not-admitted route, which executes without the pre_tool hook.
            tool_args = (args if getattr(custom, "server", None) is not None
                         else {key: value for key, value in args.items() if key != "why"})
            try:
                custom.build_argv(tool_args)
                custom.verify_executable()
            except CustomToolError:
                return False
        return True

    def pre_action(self, name, args, call_id):
        if call_id in self._actions:
            return self._actions[call_id]["refusal"]
        if self.runner is None:
            raise RuntimeError("hook workspace is not bound")
        paths = tuple(
            str(args[key])
            for key in ("path", "source", "destination")
            if isinstance(args.get(key), str)
        )
        result = self.runner.run_event(
            "pre_tool",
            boundary=f"pre_tool:{call_id}",
            data={"tool": name, "call_id": call_id, "args": args, "paths": paths},
            match={"tool": name, "arg_paths": paths},
        )
        self._record_runs(result)
        refusal = (
            f"REFUSED {result.deny_code}: {result.deny_hook}: {result.reason}"
            if result.decision == "deny"
            else None
        )
        self._actions[call_id] = {
            "name": name,
            "args": args,
            "refusal": refusal,
        }
        return refusal

    def dispatch_outcome(self, name, call_id, runtime) -> DispatchOutcome:
        action = self._actions.get(call_id)
        if action is None:
            return DispatchOutcome("refused", False, None, ())
        if action["refusal"]:
            return DispatchOutcome("denied", False, None, ())

        pending = getattr(getattr(runtime, "step_ledger", None), "pending", None)
        receipts = ()
        if (
            isinstance(pending, Mapping)
            and pending.get("tool_call_id") == call_id
            and isinstance(pending.get("receipts"), list)
        ):
            receipts = tuple(
                receipt for receipt in pending["receipts"]
                if isinstance(receipt, Mapping)
            )

        result_classes = tuple(
            str(receipt.get("result_class", receipt.get("status", ""))).upper()
            for receipt in receipts
        )
        status = "failed" if any(
            value in {
                "FAILED", "FAIL", "RED", "ERROR", "HARNESS_ERROR",
                "HARNESS_CONTRACT", "HARNESS_TRANSPORT", "TIMEOUT",
                "UNRUNNABLE", "ENVIRONMENT_BLOCKED", "EXECUTION_FAILED",
            }
            or value.endswith(("_FAILED", "_ERROR"))
            for value in result_classes
        ) else "refused" if any(
            value.startswith(("REFUSED", "REJECTED", "UNKNOWN_TOOL", "NOT_EDITABLE"))
            for value in result_classes
        ) else "ok"

        exit_code = None
        if (
            name == "shell"
            or name.startswith("mcp_")
            or name in getattr(getattr(self, "agent_tools", None), "names", ())
        ):
            from kernel_tool_tape import tape_for_run

            recorded = tape_for_run().lookup(call_id)
            if recorded is not None and recorded.completed:
                exit_code = recorded.exit_code
                if exit_code not in (None, 0) and status == "ok":
                    status = "failed"

        recorded_paths = action.get("applied_paths")
        recorded_deleted = action.get("deleted_paths", frozenset())
        if recorded_paths is not None:
            applied_paths = tuple(sorted(recorded_paths))
            deleted_paths = tuple(sorted(recorded_deleted))
        else:
            applied: set[str] = set()
            deleted: set[str] = set()
            for receipt in receipts:
                result_class = str(
                    receipt.get("result_class", receipt.get("status", ""))
                ).upper()
                if result_class != "APPLIED":
                    continue
                target_path = receipt.get("target_path")
                if isinstance(target_path, str) and target_path:
                    applied.add(self._relative_effect_path(target_path))
                for field in ("changed_paths", "changed_files"):
                    values = receipt.get(field, ())
                    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
                        continue
                    for value in values:
                        path = value.get("path") if isinstance(value, Mapping) else value
                        if isinstance(path, str) and path:
                            applied.add(self._relative_effect_path(path))
                effects = receipt.get("effects", ())
                if isinstance(effects, (list, tuple)):
                    for effect in effects:
                        if not isinstance(effect, Mapping):
                            continue
                        path = effect.get("path")
                        if isinstance(path, str) and path:
                            relative = self._relative_effect_path(path)
                            applied.add(relative)
                            if effect.get("after") is None:
                                deleted.add(relative)
            if applied:
                applied_paths = tuple(sorted(applied))
                deleted_paths = tuple(sorted(deleted))
            elif status in {"refused", "failed"} or not self._may_write_workspace(name):
                applied_paths, deleted_paths = (), ()
            else:
                applied_paths, deleted_paths = None, ()

        return DispatchOutcome(status, status != "refused", exit_code,
                               applied_paths, deleted_paths)

    def _relative_effect_path(self, raw_path: str) -> str:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve(strict=False).relative_to(
                    self.runner.workspace.resolve()
                )
            except (OSError, ValueError) as error:
                raise ValueError("applied hook effect path is outside the workspace") from error
        relative = candidate.as_posix()
        if relative in {"", "."} or relative.startswith("../"):
            raise ValueError("applied hook effect path is outside the workspace")
        return relative

    def _may_write_workspace(self, name: str) -> bool:
        return (
            name in {
                "edit", "str_replace", "insert_before", "replace_module",
                "file_create", "file_delete", "file_move", "shell",
            }
            or name in getattr(getattr(self, "agent_tools", None), "names", ())
        )

    def finalize_action(
        self, name, args, call_id, reply, runtime, bolt, created,
        *, outcome: DispatchOutcome,
    ):
        if self.runner is None:
            raise RuntimeError("hook workspace is not bound")
        self._actions.pop(call_id, None)
        changed_paths = outcome.applied_paths or ()
        result = self.runner.run_event(
            "post_tool",
            boundary=f"post_tool:{call_id}",
            data={
                "tool": name,
                "call_id": call_id,
                "args": args,
                "status": outcome.status,
                "executed": outcome.executed,
                "exit_code": outcome.exit_code,
                "output": reply,
                "changed_paths": changed_paths,
                "deleted_paths": outcome.deleted_paths,
            },
            match={
                "tool": name,
                "changed_paths": changed_paths,
                "deleted_paths": outcome.deleted_paths,
                "applied_paths_available": outcome.applied_paths is not None,
            },
            label=f"after {name}, call {call_id}",
        )
        self._record_runs(result)
        for entry in result.entries:
            self.state["pending_notes"][entry.hook] = vars(entry)
            matching = next(
                (
                    run.event_id
                    for run in reversed(result.runs)
                    if run.hook == entry.hook
                ),
                None,
            )
            if matching is not None:
                self.state["pending_ids"][entry.hook] = matching
        for writer in result.writes:
            from kernel_shell_tool import ShellReply, publish_effects

            receipt = ShellReply(
                f"{writer.hook} changed {', '.join(effect.path for effect in writer.authorized)}",
                tuple(writer.reconciliation.applied),
            )
            publish_effects(receipt, runtime, bolt, created, name=f"hook:{writer.hook}")
            parts = []
            if writer.authorized:
                parts.append(
                    f"{writer.hook} changed "
                    + ", ".join(effect.path for effect in writer.authorized)
                )
            if writer.restored:
                parts.append(
                    f"restored {len(writer.restored)} unauthorized hook effects"
                )
            if parts:
                previous = self.state["pending_notes"].get(writer.hook)
                text = sanitize(
                    (previous["text"] + "\n" if previous else "") + "\n".join(parts),
                    limit=2048,
                ).text
                self.state["pending_notes"][writer.hook] = vars(
                    PanelEntry(
                        "note",
                        writer.hook,
                        "post_tool",
                        f"after {name}, call {call_id}",
                        text,
                    )
                )
                matching = next(
                    (
                        run.event_id
                        for run in reversed(result.runs)
                        if run.hook == writer.hook
                    ),
                    None,
                )
                if matching is not None:
                    self.state["pending_ids"][writer.hook] = matching
        return result

    def note_applied(self, call_id, paths, *, deleted_paths=()):
        action = self._actions.get(call_id)
        if action is not None:
            action["applied_paths"] = frozenset(
                set(action.get("applied_paths", ()))
                | {self._relative_effect_path(path) for path in paths}
            )
            action["deleted_paths"] = frozenset(
                set(action.get("deleted_paths", ()))
                | {self._relative_effect_path(path) for path in deleted_paths}
            )

    def panel(self, runtime=None):
        entries = (
            *self.start_context,
            *(PanelEntry(**entry) for entry in self.state["pending_notes"].values()),
        )
        if not entries:
            return None
        meter = getattr(runtime, "_request_token_meter", None)

        def count(content):
            if meter is not None:
                try:
                    measured = meter.count_text_exact(content)
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                    measured = None
                if isinstance(measured, int) and measured >= 0:
                    return measured
            return len(content.encode("utf-8"))

        self._last_panel = render_panel(entries, room_tokens=None, count_tokens=count)
        return self._last_panel.message

    def sent(self, wire=None, request_id=None):
        admission = self._panel_admission
        if wire is not None and (self.start_ids or self.state["pending_ids"]):
            try:
                body = json.loads(wire)
                messages = body.get("messages", ())
            except (TypeError, ValueError, RecursionError):
                body, messages = None, ()
            identity = admission.get("identity") if isinstance(admission, Mapping) else None
            admitted = 0
            if (
                isinstance(identity, Mapping)
                and admission.get("request_id") == request_id
                and isinstance(messages, list)
                and type(identity.get("index")) is int
                and 0 <= identity["index"] < len(messages)
            ):
                message = messages[identity["index"]]
                content = message.get("content") if isinstance(message, dict) else None
                if (
                    isinstance(content, str)
                    and message.get("role") == identity.get("role")
                    and hashlib.sha256(content.encode("utf-8")).hexdigest()
                    == identity.get("content_sha256")
                ):
                    admitted = len(content.encode("utf-8"))
            requested = int(admission.get("requested_bytes", 0)) if isinstance(
                admission, Mapping
            ) else 0
            ids = (*self.start_ids, *self.state["pending_ids"].values())
            self.runner.record_delivery(
                ids,
                request_id=request_id,
                admitted_bytes=admitted,
                omitted_bytes=max(0, requested - admitted),
                dropped=admitted == 0,
            )
            self.state["consumed_ids"].extend(self.state["pending_ids"].values())
        self._panel_admission = None
        self.state["pending_notes"].clear()
        self.state["pending_ids"].clear()
        self._last_panel = None

    def admit_panel(self, body, wire, request_id):
        """Carry the typed panel's exact identity to the physical send boundary."""
        if type(request_id) is not str or not request_id:
            raise ValueError("hook panel admission requires a request id")
        messages = body.get("messages") if isinstance(body, Mapping) else None
        typed = [
            (index, message)
            for index, message in enumerate(messages if isinstance(messages, list) else ())
            if isinstance(message, Mapping)
            and message.get("_request_budget_section") == "user_hooks"
        ]
        identity = None
        if len(typed) > 1:
            raise ValueError("request contains multiple typed user hook panels")
        if typed:
            index, message = typed[0]
            content = message.get("content")
            role = message.get("role")
            if not isinstance(content, str) or not isinstance(role, str):
                raise ValueError("typed user hook panel has invalid content")
            identity = {
                "index": index,
                "role": role,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        elif (
            isinstance(self._panel_admission, Mapping)
            and self._panel_admission.get("request_id") == request_id
            and isinstance(self._panel_admission.get("identity"), Mapping)
        ):
            carried = self._panel_admission["identity"]
            wire_messages = json.loads(wire).get("messages", ())
            index = carried.get("index")
            if isinstance(wire_messages, list) and type(index) is int and 0 <= index < len(wire_messages):
                message = wire_messages[index]
                content = message.get("content") if isinstance(message, Mapping) else None
                if (
                    isinstance(content, str)
                    and isinstance(message, Mapping)
                    and message.get("role") == carried.get("role")
                    and hashlib.sha256(content.encode("utf-8")).hexdigest()
                    == carried.get("content_sha256")
                ):
                    identity = dict(carried)
        requested = (
            self._last_panel.delivered_bytes + self._last_panel.omitted_bytes
            if self._last_panel is not None else 0
        )
        self._panel_admission = {
            "request_id": request_id,
            "identity": identity,
            "requested_bytes": requested,
        }
        return identity

    def stop(
        self, runtime, *, source, call_id=None, candidate_digest=None, changed_paths=()
    ):
        if self.runner is None:
            raise RuntimeError("hook workspace is not bound")
        self.state["stop_count"] += 1
        number = self.state["stop_count"]
        blocks = self.state["block_counts"]
        max_blocks = min(
            (hook.max_blocks for hook in self.hooks.for_event("stop")), default=3
        )
        latest = getattr(runtime, "_latest_check_receipt", None)
        checks = (
            {
                key: latest[key]
                for key in (
                    "receipt_id",
                    "status",
                    "outcome",
                    "revision",
                    "exit_code",
                    "execution_complete",
                    "termination_reason",
                    "counts",
                )
                if key in latest
            }
            if isinstance(latest, Mapping)
            else {}
        )
        result = self.runner.run_event(
            "stop",
            boundary=f"stop:{number}",
            data={
                "call_id": call_id,
                "source": source,
                "candidate_digest": candidate_digest,
                "changed_paths": changed_paths,
                "checks": checks,
                "blocks_so_far": sum(blocks.values()),
                "max_blocks": max_blocks,
            },
        )
        self._record_runs(result)
        self.state["gate"] = {
            "source": source,
            "digest": candidate_digest,
            "decision": result.decision,
            "number": number,
        }
        if result.decision == "deny":
            blocks[result.deny_hook] = blocks.get(result.deny_hook, 0) + 1
        limit = next(
            (
                hook.max_blocks
                for hook in self.hooks.for_event("stop")
                if hook.name == result.deny_hook
            ),
            max_blocks,
        )
        return result, blocks.get(result.deny_hook, 0) > limit

    def close(self):
        global _ACTIVE
        if self.runner is not None:
            self.runner.close()
        if _ACTIVE is self:
            _ACTIVE = None
