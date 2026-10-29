"""V30-only compatibility overlay for audited advice and dead-end contracts.

The V30 replay gate permits behavior at the runtime overlay boundary while
requiring unchanged v23-parity modules to remain byte-identical. These small
wrappers preserve each module's authority and observations, changing only the
model-facing projection.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    import kernel_completion_policy
    import kernel_edit_affordance
    import kernel_empirical_fixes
    import kernel_prepublish_diff_audit
    import kernel_probe
    import kernel_rejected_candidate
    import kernel_stateframe_f4
    import kernel_stateframe_receipts
    try:  # lab-only accounting tool; absent from frozen releases
        import scripts.compare_release as _compare_release
    except ImportError:
        _compare_release = None

    original_render = kernel_completion_policy.render

    def render(runtime: object) -> str:
        text = original_render(runtime)
        return text.replace(
            "done now: submits uncertified",
            "done now: host admission is not established; completion remains unresolved",
        )

    kernel_completion_policy.render = render

    original_apply_frame = kernel_edit_affordance.apply_frame

    def apply_frame(frame: dict, scope: dict | None, path: str, *, authority: bool = False) -> None:
        original_apply_frame(frame, scope, path, authority=authority)
        if not authority or not isinstance(scope, Mapping):
            return
        edit_scope = frame.get("action_prerequisites", {}).get("edit_scope")
        if not isinstance(edit_scope, dict):
            return
        edit_scope.setdefault("mode", scope.get("mode", ""))
        if "tool_matrix" in scope:
            edit_scope.setdefault("tool_matrix", scope["tool_matrix"])

    kernel_edit_affordance.apply_frame = apply_frame

    original_method_target = kernel_empirical_fixes.unique_method_target

    def unique_method_target(source: str, name: str) -> tuple[str | None, str | None]:
        target, error = original_method_target(source, name)
        if error:
            error = error.replace(
                "use the qualified class.method target.",
                "use an exact str_replace with the class context, or replace the complete class.",
            )
        return target, error

    kernel_empirical_fixes.unique_method_target = unique_method_target
    kernel_prepublish_diff_audit._GIT_REASON = (
        "git state mutation is host-denied; only the controller may use its typed restore path"
    )

    original_probe_render = kernel_probe.render_probe_receipt

    def render_probe_receipt(receipt: object) -> str:
        rendered = original_probe_render(receipt)
        if getattr(receipt, "reason", "") == "task_probe_limit":
            rendered += (
                "\nHOST_CAPABILITY: task-scoped probe allocation is exhausted; later proposals "
                "do not replenish it. Use an offered observation or verification action, or "
                "preserve the blocked condition."
            )
        return rendered

    kernel_probe.render_probe_receipt = render_probe_receipt

    def bounded_literal(text: str, room: int, reference: str) -> str:
        handle = (
            f"\n[... draft truncated; recall with "
            f"recall_generated_artifact(reference={reference}, offset=..., limit=...)]\n"
        )
        if len(text) <= room:
            return text
        available = max(0, room - len(handle))
        head = available // 2
        tail = available - head
        return text[:head] + handle + (text[-tail:] if tail else "")

    kernel_rejected_candidate._bounded_literal = bounded_literal

    original_project = kernel_stateframe_f4.project_f4_state

    def project_f4_state(*args: object, **kwargs: object) -> dict[str, object]:
        result = original_project(*args, **kwargs)
        frame = args[0] if args else kwargs.get("frame")
        if isinstance(frame, Mapping):
            prerequisites = frame.get("action_prerequisites")
            edit_scope = prerequisites.get("edit_scope") if isinstance(prerequisites, Mapping) else None
            if isinstance(edit_scope, Mapping) and edit_scope:
                result.setdefault("state", {})["edit_authority"] = {
                    key: edit_scope[key]
                    for key in ("mode", "editable_paths", "shortcuts", "tool_matrix")
                    if key in edit_scope
                }
        return result

    kernel_stateframe_f4.project_f4_state = project_f4_state

    original_handoff = kernel_stateframe_f4._handoff_text

    def handoff_text(*args: object, **kwargs: object) -> str:
        return original_handoff(*args, **kwargs).replace(
            "supply the missing decision slot and choose the next discriminating action",
            "No optional decision slot was supplied; choose the next offered action from the current evidence",
        )

    kernel_stateframe_f4._handoff_text = handoff_text

    original_investigation = kernel_stateframe_receipts._investigation_lines

    def investigation_lines(frame: Mapping[str, object], row: Mapping[str, object] | None):
        lines, advisories = original_investigation(frame, row)
        if os.environ.get("STATEFRAME_STALE_PROBE_INSPECT", "off") != "off":
            lines = [line for line in lines if "request one by receipt id for inspection" not in line]
            probe_rows = kernel_stateframe_receipts._probe_rows(frame)
            for index, probe in enumerate(probe_rows):
                prior = probe_rows[index - 1] if index else None
                status = kernel_stateframe_receipts.classify_probe_exit(probe, prior=prior)
                lines.append(
                    "probe exit: "
                    + status
                    + "; test="
                    + str(probe.get("test", probe.get("selector", probe.get("obligation", "UNKNOWN"))))
                )
        return lines, advisories

    kernel_stateframe_receipts._investigation_lines = investigation_lines

    if _compare_release is None:
        return
    original_perf = _compare_release.perf_from_events

    def perf_from_events(events: object, **kwargs: object):
        if (not isinstance(events, Iterable)
                or isinstance(events, (str, bytes, bytearray, Mapping))):
            return original_perf(events, **kwargs)
        seen: set[str] = set()
        unique = []
        for event in events:
            if not isinstance(event, Mapping):
                unique.append(event)
                continue
            response = event.get("response")
            response = response if isinstance(response, Mapping) else {}
            key = next((str(value) for value in (
                response.get("id"), event.get("response_id"), event.get("request_id")
            ) if value not in (None, "")), "")
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            unique.append(event)
        return original_perf(unique, **kwargs)

    _compare_release.perf_from_events = perf_from_events


__all__ = ["install"]
