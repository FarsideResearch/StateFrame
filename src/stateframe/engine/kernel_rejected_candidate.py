"""Bounded recovery of rejected model-authored edits; never execution authority."""
from __future__ import annotations

import copy
import difflib
import os

from kernel_measurement import digest

_RENDER_ROOM = 40000
_CONTEXT_RADIUS = 32
_EDIT_TOOLS = frozenset({"edit", "str_replace", "file_create"})


def enabled() -> bool:
    value = os.environ.get("KERNEL_REJECTED_CANDIDATE_RECOVERY", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_REJECTED_CANDIDATE_RECOVERY must be 0 or 1")
    return value == "1"


class RejectedCandidate:
    def __init__(self):
        self.candidate: dict | None = None
        self.binding: dict | None = None
        self.reason = "no_candidate"

    def clear(self, reason: str) -> None:
        self.candidate = None
        self.binding = None
        self.reason = reason

    @staticmethod
    def _line_column(text: str, offset: int) -> tuple[int, int]:
        line = text.count("\n", 0, offset) + 1
        line_start = text.rfind("\n", 0, offset) + 1
        return line, offset - line_start + 1

    @staticmethod
    def _context(text: str, offset: int) -> str:
        start = max(0, offset - _CONTEXT_RADIUS)
        end = min(len(text), offset + _CONTEXT_RADIUS)
        return text[start:end]

    @classmethod
    def _first_mismatch(cls, old: str, current: str) -> dict:
        """Locate the first non-equal opcode in the current-file coordinate space."""
        matcher = difflib.SequenceMatcher(None, old, current, autojunk=False)
        opcode = next((item for item in matcher.get_opcodes() if item[0] != "equal"), None)
        if opcode is None:
            offset_old = len(old)
            offset_current = len(current)
            tag = "missing_old_str"
        else:
            tag, offset_old, _old_end, offset_current, _current_end = opcode
        line, column = cls._line_column(current, offset_current)
        return {
            "status": "OLD_STR_MISMATCH",
            "reason": "old_str not found in current file",
            "kind": tag,
            "line": line,
            "column": column,
            "old_context": cls._context(old, offset_old),
            "current_context": cls._context(current, offset_current),
        }

    def close(self, pending: dict | None, record: dict,
              *, current_source: str | None = None) -> None:
        tool = record.get("tool")
        if tool not in _EDIT_TOOLS:
            return
        self.clear("superseded_by_edit_attempt")
        if record.get("result") != "REJECTED":
            return
        if not pending or pending.get("tool") != tool or pending.get("step") != record.get("step"):
            self.reason = "pending_mismatch"
            return
        args = pending.get("arguments")
        if not isinstance(args, dict) or digest(args) != record.get("arguments_hash"):
            self.reason = "arguments_mismatch"
            return
        receipts = pending.get("receipts", [])
        if not any(r.get("receipt_id") in record.get("receipt_ids", [])
                   and r.get("result_class") == "REJECTED" for r in receipts):
            self.reason = "missing_rejection_receipt"
            return
        before, after = record.get("before_binding"), record.get("after_binding")
        if not isinstance(after, dict) or before != after or pending.get("before") != before:
            self.reason = "changed_binding"
            return
        path = args.get("path")
        revisions = after.get("product_files", {})
        revision = revisions.get(path) if isinstance(revisions, dict) and isinstance(path, str) else None
        if tool == "file_create" and revision is None:
            revision = "ABSENT"
        if not isinstance(revision, str) or not revision:
            self.reason = "unbound_product_path"
            return
        required = (("old_str", "new_str") if tool == "str_replace" else
                    ("function_source",) if tool == "edit" else
                    ("content",) if isinstance(args.get("content"), str) else ("new_str",))
        if not all(isinstance(args.get(k), str) and args[k] for k in required):
            self.reason = "missing_candidate_text"
            return
        artifact_id = digest(args)
        artifact_reference = "stateframe://artifacts/rejected/" + artifact_id
        candidate = {
            "status": "UNVERIFIED_MODEL_PROPOSAL_NOT_APPLIED",
            "authority": "MODEL_GENERATED_UNVERIFIED",
            "instruction": "Prior rejected edit, not current source or proof. Correct it through ordinary edit tools if useful.",
            "tool": tool, "arguments": copy.deepcopy(args), "path": path,
            "revision": revision, "step": record["step"],
            "candidate_digest": artifact_id,
            "reference": artifact_reference,
            "artifact_reference": artifact_reference,
            "feedback": str(record.get("observation", ""))[:2000],
        }
        if (tool == "str_replace" and isinstance(current_source, str)
                and args["old_str"] not in current_source):
            candidate["mismatch"] = self._first_mismatch(args["old_str"], current_source)
        self.candidate = candidate
        self.binding = copy.deepcopy(after)
        self.reason = "available"

    def context(self, binding: dict, step: int) -> dict | None:
        if self.candidate is None:
            return None
        if binding != self.binding:
            self.clear("stale_binding")
            return None
        if step - self.candidate["step"] > 4:
            self.clear("expired")
            return None
        return copy.deepcopy(self.candidate)


def render_candidate(candidate: dict, mode: str = "json") -> str:
    """Present exact proposal text, bounded only at the model-facing render seam."""
    from kernel_measurement import canonical_bytes
    if mode not in {"json", "blocks", "current_blocks"}:
        raise ValueError("unsupported candidate presentation")
    heading = "[Rejected candidate; NOT APPLIED]\n"
    projected = _bounded_candidate(candidate)
    if mode == "json":
        return heading + canonical_bytes(projected).decode()
    projected = copy.deepcopy(projected)
    arguments = projected["arguments"]
    names = (("old_str", "new_str") if candidate["tool"] == "str_replace" else
             ("function_source",) if candidate["tool"] == "edit" else
             ("content",) if isinstance(arguments.get("content"), str) else ("new_str",))
    texts = []
    for name in names:
        text = arguments.pop(name)
        if not isinstance(text, str):
            raise ValueError("candidate code must be exact text")
        texts.append((name, text))
    projected["argument_texts"] = {name: {"characters": len(text), "representation": "literal_text"}
                                    for name, text in texts}
    blocks = [heading + canonical_bytes(projected).decode()]
    for name, text in texts:
        blocks.append(f"[Candidate argument: {name}; {len(text)} characters]\n"
                      + text + f"\n[End candidate argument: {name}]")
    rendered = "\n".join(blocks)
    if len(rendered) <= _RENDER_ROOM:
        return rendered
    # The metadata and old_str are useful context; preserve the requested draft
    # as a head/tail literal when the block presentation itself is larger.
    target = "new_str" if candidate["tool"] == "str_replace" else (
        "function_source" if candidate["tool"] == "edit" else
        "content" if isinstance(candidate["arguments"].get("content"), str) else "new_str")
    original = candidate["arguments"].get(target, "")
    base = "\n".join(blocks).replace(texts[[name for name, _ in texts].index(target)][1], "")
    allowed = max(0, _RENDER_ROOM - len(base))
    bounded = _bounded_literal(original, allowed, candidate.get("artifact_reference", candidate["reference"]))
    for index, (name, _text) in enumerate(texts):
        if name == target:
            texts[index] = (name, bounded)
    blocks = [heading + canonical_bytes(projected).decode()]
    for name, text in texts:
        blocks.append(f"[Candidate argument: {name}; {len(text)} characters]\n"
                      + text + f"\n[End candidate argument: {name}]")
    return "\n".join(blocks)[:_RENDER_ROOM]


def _bounded_literal(text: str, room: int, reference: str) -> str:
    handle = f"\n[... draft truncated; recall with read {reference}]\n"
    if len(text) <= room:
        return text
    available = max(0, room - len(handle))
    head = available // 2
    tail = available - head
    return text[:head] + handle + (text[-tail:] if tail else "")


def _bounded_candidate(candidate: dict) -> dict:
    from kernel_measurement import canonical_bytes
    full = copy.deepcopy(candidate)
    if len(canonical_bytes(full).decode()) <= _RENDER_ROOM:
        return full
    projected = copy.deepcopy(candidate)
    arguments = projected.get("arguments", {})
    target = ("new_str" if candidate.get("tool") == "str_replace" else
              "function_source" if candidate.get("tool") == "edit" else
              "content" if isinstance(arguments.get("content"), str) else "new_str")
    original = arguments.get(target)
    if not isinstance(original, str):
        return projected
    arguments[target] = _bounded_literal(
        original, _RENDER_ROOM // 2,
        candidate.get("artifact_reference", candidate.get("reference", "")))
    while len(canonical_bytes(projected).decode()) > _RENDER_ROOM and len(arguments[target]) > 64:
        arguments[target] = _bounded_literal(
            original, max(64, len(arguments[target]) - 1024),
            candidate.get("artifact_reference", candidate.get("reference", "")))
    return projected
