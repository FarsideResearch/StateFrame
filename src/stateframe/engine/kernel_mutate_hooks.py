"""Observing mutation attachments with bounded, non-gating trigger receipts."""
from __future__ import annotations

import json
import hashlib
import re
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class HookResult:
    enabled: bool
    timed_out: bool
    cleanup: object | None
    output_bytes: int
    text: str
    veto: bool = False
    recursion_depth: int = 0


_FACTS = frozenset({"tool", "call_id", "status", "executed", "exit_code", "steps"})


def _safe_facts(facts: dict) -> dict[str, str | int | bool]:
    """Pass only bounded host-owned scalar metadata, never a tool observation."""
    result: dict[str, str | int | bool] = {}
    for key in _FACTS:
        value = facts.get(key)
        if key in {"tool", "call_id"} and isinstance(value, str) and re.fullmatch(r"[a-z0-9_-]{1,64}", value):
            result[key] = value
        elif key == "status" and isinstance(value, str) and value in {"ok", "failed", "refused", "applied"}:
            result[key] = value
        elif key == "executed" and type(value) is bool:
            result[key] = value
        elif key in {"exit_code", "steps"} and type(value) is int and -1000 <= value <= 1000:
            result[key] = value
    return result


class MutationHooks:
    def __init__(self, runtime):
        self.runtime = runtime
        self.attached: dict[tuple[str, str], dict] = {}
        self.active = False

    def attach(self, unit: dict, event: str) -> None:
        if event not in ("post_tool", "session_end"):
            raise ValueError("mutation hooks are observing post_tool or session_end only")
        if not isinstance(unit, dict) or not isinstance(unit.get("name"), str):
            raise ValueError("invalid mutation hook unit")
        requires = unit.get("requires", {})
        if requires.get("network") or requires.get("writes"):
            raise ValueError("mutation hooks must be offline and read-only")
        source = unit.get("source")
        files = unit.get("files")
        if source is not None and not isinstance(source, str):
            raise ValueError("invalid mutation hook source")
        if files is not None and (not isinstance(files, dict)
                                  or not all(isinstance(name, str) and isinstance(data, bytes)
                                             for name, data in files.items())):
            raise ValueError("invalid mutation hook bundle")
        name = unit["name"]
        if any(row.get("kind") == "HOOK_DISABLED" and row.get("name") == name
               for row in self.runtime.journal.events()):
            for key in list(self.attached):
                if key[0] == name:
                    del self.attached[key]
            return
        self.attached[(name, event)] = dict(unit)
        if not any(row.get("kind") == "HOOK_ATTACHED" and row.get("name") == name
                   and row.get("event") == event for row in self.runtime.journal.events()):
            self.runtime.journal.append("HOOK_ATTACHED", name=name, event=event)

    def failures(self, name: str) -> int:
        return sum(row.get("kind") == "HOOK_FAILURE" and row.get("name") == name
                   for row in self.runtime.journal.events())

    def disable(self, name: str, reason: str) -> None:
        if not any(row.get("kind") == "HOOK_DISABLED" and row.get("name") == name
                   for row in self.runtime.journal.events()):
            self.runtime.journal.append("HOOK_DISABLED", name=name, reason=reason)
        for key in list(self.attached):
            if key[0] == name:
                del self.attached[key]

    def refuse_platform(self, name: str, event: str, trigger: str | None, reason: str) -> None:
        rows = self.runtime.journal.events()
        has_intent = trigger is not None and any(
            row.get("kind") == "HOOK_TRIGGER_INTENT" and row.get("invocation_id") == trigger
            for row in rows)
        if has_intent:
            already_refused = any(row.get("kind") == "HOOK_REFUSED"
                                  and row.get("invocation_id") == trigger for row in rows)
        else:
            already_refused = any(row.get("kind") == "HOOK_REFUSED" and row.get("name") == name
                                  and row.get("event") == event for row in rows)
        if not already_refused:
            self.runtime.journal.append("HOOK_REFUSED", invocation_id=trigger,
                                        name=name, event=event,
                                        code="MUTATION_PLATFORM_UNQUALIFIED", reason=reason[:600])

    def invoke(self, event: str, facts: dict) -> HookResult:
        if __package__:
            from .kernel_mutate_runtime import MutationUnavailable
        else:
            from kernel_mutate_runtime import MutationUnavailable
        if event not in ("post_tool", "session_end"):
            raise ValueError("mutation hook event is not observing")
        if self.active or not self.attached:
            return HookResult(False, False, None, 0, "")
        if not isinstance(facts, dict):
            facts = {}
        from kernel_tool_tape import replay_mode_active, tape_for_run
        if replay_mode_active():
            for (name, attached_event) in self.attached:
                if attached_event != event:
                    continue
                completed = [row for row in self.runtime.journal.events()
                             if row.get("kind") == "HOOK_TRIGGER_COMPLETE"
                             and row.get("name") == name and row.get("event") == event
                             and row.get("parent_turn_id") == facts.get("call_id")]
                if completed:
                    text = str(completed[-1].get("text", ""))[:512]
                    if self.runtime.run_dir is not None:
                        from kernel_tool_tape import ToolReplayRefused

                        record = tape_for_run().lookup(completed[-1]["invocation_id"])
                        if record is None or not record.completed or record.observation != text:
                            raise ToolReplayRefused("mutation hook tape differs from trigger evidence")
                    self.runtime.hook_lines.append(text)
            return HookResult(False, False, None, 0, "")
        result = HookResult(False, False, None, 0, "")
        for (name, attached_event), unit in list(self.attached.items()):
            if attached_event != event:
                continue
            boundary = facts.get("call_id")
            trigger = ("h" + hashlib.sha256(
                f"{name}:{event}:{boundary}".encode()).hexdigest()[:30]
                if isinstance(boundary, str) and boundary else "h" + uuid.uuid4().hex)
            platform_error = getattr(self.runtime, "_platform_error", None)
            if platform_error is not None:
                self.refuse_platform(name, event, trigger, platform_error)
                self.disable(name, "platform_unqualified")
                continue
            if self.failures(name) >= 3:
                self.disable(name, "failure_limit")
                continue
            spent = sum(float(row.get("duration_s", 0)) for row in self.runtime.journal.events()
                        if row.get("kind") in ("HOOK_TRIGGER_COMPLETE", "HOOK_FAILURE"))
            remaining = max(0.0, 10.0 - spent)
            if remaining <= 0:
                self.disable(name, "run_wall_budget")
                continue
            if any(row.get("kind") == "HOOK_TRIGGER_INTENT" and row.get("invocation_id") == trigger
                   for row in self.runtime.journal.events()):
                continue
            redacted = _safe_facts(facts)
            try:
                json.dumps(facts, allow_nan=False)
            except (TypeError, ValueError):
                redacted = {}
                invalid_facts = True
            else:
                invalid_facts = False
            self.runtime.journal.append("HOOK_TRIGGER_INTENT", invocation_id=trigger,
                                        name=name, event=event, parent_turn_id=boundary,
                                        facts=redacted)
            tape = tape_for_run() if self.runtime.run_dir is not None else None
            if tape is not None:
                tape.intent(call_id=trigger, tool="mutation_hook", args={"name": name, "event": event},
                            replay_class="observation", capability_digest="mutation-hook/1",
                            definition_digest=None, env_identity="mutation-runtime/1",
                            pre_manifest_digest="mutation-hook/1")
            self.active = True
            started = time.monotonic()
            try:
                if invalid_facts:
                    raise ValueError("invalid hook facts")
                result = self._run(unit, trigger, redacted, limit_s=min(1.0, remaining))
                if result.timed_out:
                    self.runtime.retire_hook_executor()
                if result.timed_out or not result.enabled:
                    raise ValueError("mutation hook failed or timed out")
                self.runtime.journal.append("HOOK_TRIGGER_COMPLETE", invocation_id=trigger,
                                            name=name, event=event, parent_turn_id=boundary,
                                            text=result.text,
                                            duration_s=time.monotonic() - started)
            except MutationUnavailable as error:
                self.refuse_platform(name, event, trigger, str(error))
                self.disable(name, "platform_unqualified")
                result = HookResult(False, False, None, 0, "")
            except Exception as error:
                # Containment failures inherit BaseException and remain fail-closed.
                self.runtime.journal.append("HOOK_FAILURE", invocation_id=trigger,
                                            name=name, event=event, reason=type(error).__name__,
                                            duration_s=time.monotonic() - started)
                if self.failures(name) >= 3:
                    self.disable(name, "failure_limit")
                result = HookResult(False, result.timed_out, result.cleanup, 0, "")
            finally:
                self.active = False
            if tape is not None:
                tape.complete(call_id=trigger, exit_code=0, observation=result.text,
                              post_manifest_digest="mutation-hook/1", env_identity="mutation-runtime/1")
            if result.text:
                self.runtime.hook_lines.append(result.text)
                self.runtime.hook_lines[:] = self.runtime.hook_lines[-4:]
        return result

    def _run(self, unit: dict, trigger: str, facts: dict, *, limit_s: float = 1.0) -> HookResult:
        from kernel_mutate_build import stage_unit
        from kernel_mutate_exec import (LaunchSpec, MutationLimits, MutationContainmentError,
                                        _hook_timing)
        from kernel_mutate_wire import build_stdin_doc, parse_stdout

        source = unit.get("source")
        files = unit.get("files")
        if not isinstance(source, str) and not isinstance(files, dict):
            raise ValueError("mutation hook source is unavailable")
        manifest = {key: value for key, value in unit.items() if key not in ("source", "files")}
        if not manifest.get("smoke"):
            manifest["smoke"] = [{"input": {}, "expect": {}}]
        _hook_timing("executor start requested")
        executor = self.runtime._ensure_executor()
        started = time.monotonic()
        bundle = files if isinstance(files, dict) else {
            manifest["entry"]: source.encode("utf-8", "strict")}
        staged = stage_unit(bundle, manifest, executor.root)
        deadline = started + limit_s
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return HookResult(False, True, None, 0, "")
        launch = uuid.uuid4().hex
        document = build_stdin_doc(wire=1, call=trigger, launch=launch, phase=0,
                                   role="payload", input=facts, inputs=[], responses=[],
                                   grants={"network": False, "writes": False}, max_fetch=0)
        spec = LaunchSpec("payload", trigger, launch, staged.dir, {}, {}, document,
                          MutationLimits(wall_s=remaining, cpu_s=min(1.0, remaining),
                                         mem_mb=128, output_bytes=4096), budget_after_ready=True,
                          trigger_deadline=deadline)
        try:
            outcome = self.runtime._launch(spec)
        except BaseException:
            # The launch's stop_all only quiesces jobs. Retire the executor as well so
            # startup failures cannot strand its ledger, control socket or runtime root.
            self.runtime.retire_hook_executor()
            raise
        if not outcome.cleanup.ok:
            raise MutationContainmentError("mutation hook cleanup is unverified")
        if outcome.timed_out or outcome.stdout_overflow or outcome.exit_code != 0:
            return HookResult(False, outcome.timed_out, outcome.cleanup, 0, "")
        frame = parse_stdout(outcome.stdout, call=trigger, output_schema=manifest["output_schema"])
        if frame.kind != "result":
            return HookResult(False, False, outcome.cleanup, 0, "")
        encoded = json.dumps(frame.result, sort_keys=True, ensure_ascii=False).encode("utf-8")[:512]
        text = encoded.decode("utf-8", "ignore")
        return HookResult(True, False, outcome.cleanup, len(text.encode()), text)
