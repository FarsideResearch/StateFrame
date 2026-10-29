"""Derived, explicitly scoped index of model proposals already in the audit journal."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


_MAX_DIGEST_DEPTH = 100_000
_MAX_DIGEST_NODES = 1_000_000


def _digest(value: object) -> str:
    """Hash canonical JSON with an explicit stack so deep audit rows stay safe."""
    output: list[str] = []
    active: set[int] = set()
    stack: list[tuple[str, object, int]] = [("value", value, 0)]
    nodes = 0
    while stack:
        operation, current, depth = stack.pop()
        if operation == "leave":
            active.remove(int(current))
            continue
        if operation == "raw":
            output.append(str(current))
            continue
        nodes += 1
        if nodes > _MAX_DIGEST_NODES:
            raise ValueError("generated event exceeds digest node limit")
        if depth > _MAX_DIGEST_DEPTH:
            raise ValueError("generated event exceeds digest depth limit")
        if isinstance(current, dict):
            identity = id(current)
            if identity in active:
                raise ValueError("generated event contains a circular mapping")
            if any(not isinstance(key, str) for key in current):
                raise TypeError("generated event mapping keys must be strings")
            active.add(identity)
            output.append("{")
            items = sorted(current.items())
            tokens: list[tuple[str, object, int]] = []
            for index, (key, item) in enumerate(items):
                if index:
                    tokens.append(("raw", ",", 0))
                tokens.extend((
                    ("raw", json.dumps(key, ensure_ascii=False), 0),
                    ("raw", ":", 0),
                    ("value", item, depth + 1),
                ))
            stack.append(("leave", identity, 0))
            stack.append(("raw", "}", 0))
            stack.extend(reversed(tokens))
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in active:
                raise ValueError("generated event contains a circular sequence")
            active.add(identity)
            output.append("[")
            tokens = []
            for index, item in enumerate(current):
                if index:
                    tokens.append(("raw", ",", 0))
                tokens.append(("value", item, depth + 1))
            stack.append(("leave", identity, 0))
            stack.append(("raw", "]", 0))
            stack.extend(reversed(tokens))
        else:
            output.append(json.dumps(current, separators=(",", ":"), ensure_ascii=False))
    return hashlib.sha256("".join(output).encode()).hexdigest()


class GeneratedArtifactIndex:
    """No filesystem source reads, execution authority, or cross-run implicit recall."""

    # The existing catalog is already a four-item page.  Keep that bound for
    # the compact presentation: the batch-166 evidence contains four delivered
    # entries per request while the retained index grows from 5 to 19 items.
    TYPED_CATALOG_LIMIT = 4

    def __init__(self, run_id: str):
        if not isinstance(run_id, str) or not run_id or ":" in run_id:
            raise ValueError("explicit valid origin run required")
        self.run_id = run_id
        self.records: dict[str, dict] = {}
        self._events: dict[int, str] = {}
        self._batch: dict[tuple[str, int], str] = {}
        self._aliases: dict[str, str] = {}
        self.sequence = 0
        self._responses: dict[str, str] = {}
        self._response_calls: dict[str, list | None] = {}
        self._unobserved_count = 0
        self._first_unobserved: dict[str, object] | None = None
        self._last_unobserved: dict[str, object] | None = None
        self._last_unobserved_sequence: int | None = None

    @property
    def completeness(self) -> str:
        return "INCOMPLETE" if self._unobserved_count else "COMPLETE"

    def _record_unobserved(self, *, sequence: object, kind: object,
                           error_type: object) -> None:
        if type(sequence) is not int or sequence < 1:
            sequence = None
        summary = {
            "sequence": sequence,
            "kind": str(kind)[:96],
            "error_type": str(error_type)[:80],
        }
        self._unobserved_count += 1
        if self._first_unobserved is None:
            self._first_unobserved = summary
        self._last_unobserved = summary
        self._last_unobserved_sequence = sequence

    def mark_unobserved(self, event: dict, *, error_type: str, error: str = "") -> None:
        """Advance past one failed observer delivery and retain bounded degradation."""
        sequence = event.get("sequence")
        if (event.get("schema") != "stateframe-audit/1" or event.get("run_id") != self.run_id
                or type(sequence) is not int or sequence != self.sequence + 1
                or event.get("event_id") != f"{self.run_id}:{sequence}"):
            raise ValueError("invalid unobserved journal identity or sequence")
        try:
            event_digest = _digest(event)
        except (TypeError, ValueError, RecursionError):
            event_digest = "UNOBSERVED:" + str(error_type)[:80]
        self._events[sequence] = event_digest
        self.sequence = sequence
        self._record_unobserved(
            sequence=sequence, kind=event.get("kind"), error_type=error_type)

    def _add(self, event: dict, suffix: str, text: str, **metadata: object) -> str:
        reference = f"generated:{self.run_id}:{event['sequence']}:{suffix}"
        self.records[reference] = {
            "reference": reference, "origin_run_id": self.run_id,
            "origin_event_id": event["event_id"], "origin_request_id": event.get("request_id"),
            "authority": "MODEL_GENERATED_UNVERIFIED", "text": text,
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "execution_status": "NO_EXECUTION_RECEIPT", **metadata,
        }
        return reference

    def observe(self, event: dict) -> None:
        seq = event.get("sequence")
        if (event.get("schema") != "stateframe-audit/1" or event.get("run_id") != self.run_id
                or type(seq) is not int or seq < 1 or event.get("event_id") != f"{self.run_id}:{seq}"):
            raise ValueError("invalid journal identity or scope")
        digest = _digest(event)
        if seq <= self.sequence:
            if self._events.get(seq) != digest:
                raise ValueError("conflicting replay event")
            return
        if seq != self.sequence + 1:
            raise ValueError("journal gap or out-of-order event")
        kind = event.get("kind")
        if kind == "journal_observer_failed" and event.get("subscriber") == "generated_artifacts":
            failed_sequence = event.get("event_sequence")
            if failed_sequence != self._last_unobserved_sequence:
                self._record_unobserved(
                    sequence=failed_sequence,
                    kind=event.get("event_kind"),
                    error_type=event.get("error_type"),
                )
        if seq == 1 and kind != "run_started":
            raise ValueError("journal must begin with run_started")
        if kind == "response_received":
            self._response(event)
        elif kind == "rejected_candidate_artifact":
            reference = event.get("reference")
            text = event.get("text")
            artifact_id = event.get("artifact_id")
            if (not isinstance(reference, str) or not reference.startswith("stateframe://artifacts/rejected/")
                    or not isinstance(text, str) or not isinstance(artifact_id, str)):
                raise ValueError("malformed rejected candidate artifact")
            generated_reference = self._add(
                event, "rejected:" + artifact_id, text,
                artifact_kind="rejected_candidate", tool=event.get("tool"),
                public_reference=reference, field=event.get("field", "new_str"))
            self._aliases[reference] = generated_reference
        elif kind == "native_batch_received":
            self._received(event)
        elif isinstance(kind, str) and kind.startswith("native_batch_"):
            self._lifecycle(event)
        self._events[seq] = digest
        self.sequence = seq

    def _response(self, event: dict) -> None:
        self._responses[event["event_id"]] = event.get("request_id")
        response = event.get("response")
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if not isinstance(choices, list):
            return
        for index, choice in enumerate(choices):
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                continue
            message = choice["message"]
            reason = choice.get("finish_reason")
            reason = reason if isinstance(reason, str) else ""
            completeness = ("PROVIDER_FINALIZED" if reason in {"stop", "tool_calls", "function_call"}
                            else "PARTIAL" if reason == "length" else "COMPLETENESS_UNKNOWN")
            common = {"response_event_id": event["event_id"], "choice_index": index,
                      "transport_attempt_id": event.get("transport_attempt_id"),
                      "completeness": completeness, "finish_reason": reason[:64]}
            # Provider reasoning/internal metadata is intentionally not a recall surface.
            if isinstance(message.get("content"), str) and message["content"]:
                self._add(event, f"choice{index}:assistant", message["content"],
                          artifact_kind="assistant_content", **common)
            calls = message.get("tool_calls") or []
            if index == 0:
                valid = (isinstance(calls, list) and bool(calls) and all(
                    isinstance(c, dict) and isinstance(c.get("function"), dict)
                    for c in calls))
                self._response_calls[event["event_id"]] = (
                    [(c["function"].get("name"), c["function"].get("arguments")) for c in calls]
                    if valid else None)
            if not isinstance(calls, list):
                continue
            for call_index, call in enumerate(calls):
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    self._add(event, f"choice{index}:call{call_index}", function["arguments"],
                              artifact_kind="provider_tool_arguments",
                              tool=function.get("name") if isinstance(function.get("name"), str) else None,
                              batch_index=call_index, **common)

    def _received(self, event: dict) -> None:
        calls, origin = event.get("calls"), event.get("request_id")
        if not isinstance(calls, list) or not isinstance(origin, str):
            raise ValueError("malformed native batch")
        for index, call in enumerate(calls):
            function = call.get("function", {}) if isinstance(call, dict) else {}
            if (not isinstance(function, dict) or not isinstance(function.get("arguments"), str)
                    or not isinstance(function.get("name"), str)):
                raise ValueError("malformed native call")
            if (origin, index) in self._batch:
                raise ValueError("ambiguous native batch origin")
        response_id = event.get("origin_response_event_id")
        if response_id is None:
            raise ValueError("missing exact response origin")
        if response_id is not None:
            expected = f"{self.run_id}:"
            if not isinstance(response_id, str) or not response_id.startswith(expected):
                raise ValueError("foreign response origin")
            response_seq = response_id[len(expected):]
            if (not response_seq.isdigit() or int(response_seq) not in self._events
                    or self._responses.get(response_id) != origin):
                raise ValueError("unknown response origin")
        expected = self._response_calls.get(response_id)
        actual = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
        if expected is not None and expected != actual:
            raise ValueError("native proposal differs from linked response")
        for index, call in enumerate(calls):
            function = call["function"]
            ref = self._add(event, f"native{index}", function["arguments"],
                            artifact_kind="native_tool_arguments", tool=function["name"], batch_index=index,
                            response_event_id=response_id, completeness="VALIDATED_NATIVE_ENVELOPE",
                            execution_call_id=f"native_{origin}_{index}",
                            execution_authority="HOST_AUDIT_LIFECYCLE",
                            origin_snapshot_id=event.get("origin_snapshot_id"))
            self._batch[(origin, index)] = ref

    def _lifecycle(self, event: dict) -> None:
        kind, origin = event.get("kind"), event.get("origin_request_id")
        if kind == "native_batch_unexecuted":
            start = event.get("next_index")
            calls = event.get("calls")
            if type(start) is not int or start < 0 or not isinstance(calls, list):
                raise ValueError("malformed unexecuted receipt")
            indexes = list(range(start, start + len(calls)))
            state = "NOT_EXECUTED"
        elif kind in {"native_batch_action_selected", "native_batch_action_closed", "native_batch_action_unresolved"}:
            index = event.get("batch_index")
            if type(index) is not int or index < 0:
                raise ValueError("invalid exact batch index")
            indexes = [index]
            state = {"native_batch_action_selected": "SELECTED_OUTCOME_UNKNOWN",
                     "native_batch_action_closed": "CLOSED",
                     "native_batch_action_unresolved": "OUTCOME_UNKNOWN"}[kind]
        else:
            return
        for position, index in enumerate(indexes):
            ref = self._batch.get((origin, index))
            if ref is None:
                raise ValueError("lifecycle without native proposal")
            item = self.records[ref]
            if event.get("origin_response_event_id") != item["response_event_id"]:
                raise ValueError("lifecycle response origin mismatch")
            if ("origin_snapshot_id" in event
                    and event["origin_snapshot_id"] != item.get("origin_snapshot_id")):
                raise ValueError("lifecycle snapshot origin mismatch")
            expected_state = "SELECTED_OUTCOME_UNKNOWN" if state in {"CLOSED", "OUTCOME_UNKNOWN"} else "NO_EXECUTION_RECEIPT"
            if item["execution_status"] != expected_state:
                raise ValueError("conflicting lifecycle transition")
            call = (event.get("tool_call") if state == "SELECTED_OUTCOME_UNKNOWN" else
                    event.get("call") if state == "OUTCOME_UNKNOWN" else
                    calls[position] if state == "NOT_EXECUTED" else None)
            if state == "CLOSED":
                if event.get("tool_call_id") != item["execution_call_id"]:
                    raise ValueError("closed call identity mismatch")
            else:
                if not isinstance(call, dict) or call.get("id") != item["execution_call_id"]:
                    raise ValueError("native call identity mismatch")
                function = call.get("function")
                if (not isinstance(function, dict) or function.get("name") != item["tool"]
                        or function.get("arguments") != item["text"]):
                    raise ValueError("native call content mismatch")
        for index in indexes:
            item = self.records[self._batch[(origin, index)]]
            item["execution_status"] = state
            item["execution_event_id"] = event["event_id"]
            if state == "CLOSED":
                value = event.get("result")
                item["execution_receipt"] = {
                    "authority": "HOST_OBSERVED_EXECUTION_CLOSE",
                    "result_class": value[:80] if isinstance(value, str) else "UNKNOWN",
                    "step_record_id": str(event.get("step_record_id", ""))[:128],
                    "does_not_certify_candidate_correctness": True,
                }

    def latest_assistant_statement(self) -> dict | None:
        """Return choice-zero public prose from the latest received response only.

        Empty responses intentionally clear the automatic selection. Historical
        prose remains in the catalog for explicit recall. Journal replay rebuilds
        response insertion order; no separate persistence or truth state is added.
        """
        latest = next(reversed(self._responses), None)
        if latest is None:
            return None
        matches = [record for record in self.records.values()
                   if record.get("response_event_id") == latest
                   and record.get("artifact_kind") == "assistant_content"
                   and record.get("choice_index") == 0]
        if len(matches) > 1:
            raise ValueError("ambiguous assistant statement identity")
        if not matches or not matches[0]["text"].strip():
            return None
        return copy.deepcopy(matches[0])

    def metadata(self, reference: str) -> dict:
        reference = self._aliases.get(reference, reference)
        item = self.records[reference]
        result = {key: copy.deepcopy(value) for key, value in item.items() if key != "text"}
        if isinstance(result.get("tool"), str):
            result["tool"] = result["tool"][:128]
        result["characters"] = len(item["text"])
        result["applicability"] = "NOT_REVALIDATED_FOR_CURRENT_WORKSPACE"
        try:
            arguments = json.loads(item["text"])
        except (ValueError, TypeError):
            arguments = None
        if isinstance(arguments, dict):
            fields = [(key, len(value)) for key, value in arguments.items() if isinstance(value, str)]
            result["text_fields"] = dict((key, size) for key, size in fields[:16] if len(key) <= 128)
            result["text_field_count"] = len(fields)
        return result

    def catalog(self, offset: int = 0) -> dict:
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid catalog offset")
        refs = list(reversed(self.records))
        chosen = refs[offset:offset+4]
        return {"authority": "MODEL_GENERATED_UNVERIFIED", "total": len(refs),
                "completeness": self.completeness,
                "unobserved": {"count": self._unobserved_count,
                               "first": copy.deepcopy(self._first_unobserved),
                               "last": copy.deepcopy(self._last_unobserved)},
                "items": [self.metadata(ref) for ref in chosen],
                "next_offset": offset+len(chosen) if offset+len(chosen) < len(refs) else None}

    @staticmethod
    def _typed_claim(item: dict) -> str:
        """Return a deterministic shape claim, never generated artifact text."""
        def field(value: object) -> str:
            # Claims are identifiers and dimensions, not a passage to be
            # paraphrased. Escape line breaks so the claim remains one line.
            return str(value).replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")

        text = item.get("text", "")
        characters = len(text) if isinstance(text, str) else 0
        parts = ["kind=" + field(item.get("artifact_kind") or "UNKNOWN")]
        if item.get("tool") is not None:
            parts.append("tool=" + field(item["tool"]))
        if item.get("field") is not None:
            parts.append("field=" + field(item["field"]))
        if item.get("batch_index") is not None:
            parts.append("batch_index=" + field(item["batch_index"]))
        parts.append("characters=" + str(characters))
        return "; ".join(parts)

    def typed_catalog(self, offset: int = 0) -> dict:
        """Return a bounded reference/claim/status index without free-text bodies."""
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid catalog offset")
        refs = list(reversed(self.records))
        chosen = refs[offset:offset + self.TYPED_CATALOG_LIMIT]
        items = []
        for reference in chosen:
            item = self.records[reference]
            items.append({
                "reference": reference,
                "claim": self._typed_claim(item),
                "status": item.get("execution_status") or "UNKNOWN",
            })
        next_offset = offset + len(chosen)
        return {
            "schema": "generated-artifact-index/1",
            "authority": "MODEL_GENERATED_UNVERIFIED",
            "total": len(refs),
            "completeness": self.completeness,
            "unobserved": {"count": self._unobserved_count,
                           "first": copy.deepcopy(self._first_unobserved),
                           "last": copy.deepcopy(self._last_unobserved)},
            "items": items,
            "next_offset": next_offset if next_offset < len(refs) else None,
        }

    def recall(self, reference: str, *, field: str = "", offset: int = 0, limit: int = 4000) -> dict:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8000:
            raise ValueError("invalid bounded character range")
        reference = self._aliases.get(reference, reference)
        if reference not in self.records:
            raise ValueError("unknown reference in explicit origin scope")
        item = self.records[reference]
        text = item["text"]
        if field:
            arguments = json.loads(text)
            if not isinstance(arguments, dict) or not isinstance(arguments.get(field), str):
                raise ValueError("field is not retained text")
            text = arguments[field]
        if offset > len(text):
            raise ValueError("offset past artifact end")
        selected = text[offset:offset+limit]
        return {**self.metadata(reference), "field": field, "offset": offset,
                "field_characters": len(text), "selected_characters": len(selected), "content": selected,
                "next_offset": offset+len(selected) if offset+len(selected) < len(text) else None,
                "range_complete": offset == 0 and len(selected) == len(text)}

    @classmethod
    def replay(cls, path: Path, *, run_id: str, cutoff: int | None = None,
               expected_sha256: str | None = None):
        if cutoff is not None and (type(cutoff) is not int or cutoff < 0):
            raise ValueError("invalid event cutoff")
        index = cls(run_id)
        hasher = hashlib.sha256()
        from kernel_measurement import resolve_journal_blob_refs

        blob_cache: dict[str, object] = {}
        with path.open("rb") as handle:
            for line in handle:
                hasher.update(line)
                if cutoff is not None and index.sequence >= cutoff:
                    if expected_sha256 is None:
                        break
                    continue
                if not line.endswith(b"\n"):
                    raise ValueError("torn journal tail; explicit earlier cutoff required")
                record = json.loads(line)
                index.observe(resolve_journal_blob_refs(record, path, blob_cache))
        if expected_sha256 is not None and hasher.hexdigest() != expected_sha256:
            raise ValueError("journal digest differs from evidence anchor")
        if cutoff is not None and index.sequence != cutoff:
            raise ValueError("cutoff exceeds available journal")
        return index
