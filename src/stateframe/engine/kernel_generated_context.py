"""Opt-in live generated-proposal retrieval, distinct from observed source evidence."""
from __future__ import annotations

import copy
import json
import os

from kernel_generated_artifacts import GeneratedArtifactIndex
from kernel_measurement import digest


def _dedup_presentation_enabled() -> bool:
    from kernel_stateframe_dedup import enabled
    return enabled()


def enabled() -> bool:
    value = os.environ.get("KERNEL_GENERATED_ARTIFACTS", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_GENERATED_ARTIFACTS must be 0 or 1")
    return value == "1"


def statement_enabled() -> bool:
    value = os.environ.get("KERNEL_ASSISTANT_STATEMENT", "0")
    if value not in {"0", "1"}:
        raise ValueError("KERNEL_ASSISTANT_STATEMENT must be 0 or 1")
    if value == "1" and not enabled():
        raise ValueError("assistant statement requires generated artifact retrieval")
    return value == "1"


def tool_surface(tools: list[dict]) -> list[dict]:
    if not enabled() or any(t["function"]["name"] == "recall_generated_artifact" for t in tools):
        return tools
    return [*tools, {"type": "function", "function": {
        "name": "recall_generated_artifact",
        "description": "Retrieve your previous generated proposal without rewriting or searching. "
                       "Use a listed generated reference, optionally field=new_str or function_source for literal code. "
                       "Use reference=catalog to page older references. Proposals are unverified, not current source. "
                       "Returned content is carried into the next frame; follow next_offset for more.",
        "parameters": {"type": "object", "properties": {
            "reference": {"type": "string"}, "field": {"type": "string"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 8000}},
            "required": ["reference"], "additionalProperties": False}}}]


class GeneratedContext:
    def __init__(self, journal):
        self.journal = journal
        self.index = GeneratedArtifactIndex.replay(journal.path, run_id=journal.run_id)
        self.requested: dict | None = None
        self.selection: dict | None = None
        journal.generated_artifacts = self.index

    @staticmethod
    def _select(index, args: dict) -> dict:
        reference = args["reference"]
        if reference == "catalog":
            if args.get("field") or "limit" in args:
                raise ValueError("catalog takes only reference and offset")
            offset = args.get("offset", 0)
            # Tool recall remains the recorded generated-artifact API.  The
            # dedup switch changes only the automatic presentation catalog;
            # recall selection bytes and its journal receipt stay unchanged.
            return index.catalog(offset)
        return index.recall(reference, field=args.get("field", ""),
                            offset=args.get("offset", 0), limit=args.get("limit", 4000))

    def checkpoint_snapshot(self) -> dict:
        return {"selection": copy.deepcopy(self.selection)}

    def restore_snapshot(self, saved: dict, records: list[dict], boundary: int) -> None:
        """Rebuild the exact requested view from trusted pre-checkpoint events."""
        if self.requested is not None or self.selection is not None:
            raise ValueError("generated holder was used before restoration")
        if not isinstance(saved, dict) or set(saved) != {"selection"}:
            raise ValueError("invalid generated checkpoint fields")
        if type(boundary) is not int or boundary < 1:
            raise ValueError("invalid generated checkpoint boundary")
        prefix = [event for event in records if event["sequence"] <= boundary]
        validated = GeneratedArtifactIndex(self.journal.run_id)
        for record in prefix:
            validated.observe(record)
        if validated.sequence != boundary:
            raise ValueError("generated checkpoint prefix is incomplete")
        recalls = [event for event in prefix if event["kind"] == "generated_artifact_recalled"]
        selection = saved["selection"]
        if selection is None:
            if recalls:
                raise ValueError("generated checkpoint lost its requested selection")
            return
        if (not isinstance(selection, dict)
                or set(selection) != {"event_id", "request", "result_hash"} or not recalls):
            raise ValueError("invalid generated selection descriptor")
        event = recalls[-1]
        if (selection["event_id"] != event["event_id"]
                or selection["request"] != event.get("selection")
                or selection["result_hash"] != event.get("result_hash")):
            raise ValueError("generated selection differs from its recall event")
        index = GeneratedArtifactIndex(self.journal.run_id)
        for record in prefix:
            if record["sequence"] >= event["sequence"]:
                break
            index.observe(record)
        requested = self._select(index, selection["request"])
        if digest(requested) != selection["result_hash"]:
            raise ValueError("generated selection differs from historical proposal bytes")
        self.requested = requested
        self.selection = copy.deepcopy(selection)

    def recall(self, args: dict) -> str:
        try:
            result = self._select(self.index, args)
        except (KeyError, ValueError, TypeError):
            return "Generated recall unavailable: use a listed reference and valid field/range, or reference=catalog."
        reference = args["reference"]
        request = {"reference": reference, "offset": args.get("offset", 0)}
        if reference != "catalog":
            request.update(field=args.get("field", ""), limit=args.get("limit", 4000))
        result_hash = digest(result)
        event = self.journal.emit("generated_artifact_recalled", reference=reference,
                                  field=args.get("field", ""), offset=args.get("offset", 0),
                                  next_offset=result.get("next_offset"), selection=request,
                                  result_hash=result_hash)
        self.requested = copy.deepcopy(result)
        self.selection = {"event_id": event["event_id"], "request": request, "result_hash": result_hash}
        return self.render(result)

    @staticmethod
    def render(result: dict) -> str:
        metadata = {key: value for key, value in result.items() if key != "content"}
        text = "[Generated artifact; UNVERIFIED proposal]\n" + json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        if "content" in result:
            text += "\n[Literal generated text]\n" + result["content"] + "\n[End generated text]"
        return text

    def automatic_statement(self) -> str:
        if self.journal.manifest.get("assistant_statement") is not True:
            return ""
        item = self.index.latest_assistant_statement()
        if item is None:
            return ""
        full = len(item["text"]) <= 1600
        card = {
            "reference": item["reference"],
            "response_ref": item["response_event_id"],
            "authority": "MODEL_GENERATED_UNVERIFIED",
            "applicability": "Historical model text; not reassessed against current evidence.",
            "completeness": item["completeness"],
            "basis": "Support and dependencies have not been inferred from this passage.",
            "representation": "full" if full else "reference_only",
            "original_characters": len(item["text"]),
            "recall": {"tool": "recall_generated_artifact",
                       "arguments": {"reference": item["reference"]}},
        }
        if full:
            card["text"] = item["text"]
        else:
            card["omission_reason"] = "intact_statement_exceeds_budget"
        return "[Automatic historical model passage]\n" + json.dumps(card, ensure_ascii=False)

    def context(self) -> str:
        catalog = (self.index.typed_catalog() if _dedup_presentation_enabled()
                   else self.index.catalog())
        text = "[Generated artifact catalog; proposals are not observed source]\n" + json.dumps(
            catalog, ensure_ascii=False,
            separators=(",", ":") if _dedup_presentation_enabled() else None)
        automatic = self.automatic_statement()
        if automatic:
            text += "\n" + automatic
        if self.requested is not None:
            # Refresh lifecycle metadata, keeping exactly the requested historical bytes.
            result = copy.deepcopy(self.requested)
            reference = result.get("reference")
            if reference is not None:
                result.update(self.index.metadata(reference))
            text += "\n" + self.render(result)
        return text
