"""Immutable inputs for a detached, replayable complete-candidate transaction.

Only snapshot construction may hydrate blobs. Rendering never consults a live
runtime, workspace, environment, journal, clock or residency index.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

if __package__:
    from .kernel_read_receipts import ReceiptIdentity, DecodedSpan, line_digests_with_numbers
else:
    from kernel_read_receipts import ReceiptIdentity, DecodedSpan, line_digests_with_numbers


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SnapshotConstructionError(ValueError):
    """Typed incomplete-authority diagnostic, never a live suspension."""
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class SourceSlot:
    key: str
    obligation_ids: tuple[str, ...]
    identity: ReceiptIdentity
    payload: str
    page: str
    line_digests: tuple[str, ...]
    line_numbers: tuple[int, ...]
    recovery_arguments: bytes
    admission_ordinal: int
    last_use_ordinal: int
    state: str
    recovery_pin: bool = False
    kind: str = "product"
    # Each carrier contains exactly one captured interval; never merge pages.
    capture_ids: tuple[str, ...] = ()
    payload_start: int | None = None
    previous_mode: str | None = None
    # An identical closure interval can supply its existing user-role grammar
    # when the captured native tool result is no longer in the newest exchange.
    standalone_page: str | None = None
    standalone_payload_start: int | None = None
    # Plan-only proposals were absent from the source panels at freeze time.
    # Their omission can share a path notice; capture debt never can.
    plan_only: bool = False

    def __post_init__(self) -> None:
        for name in ("obligation_ids", "capture_ids", "line_digests", "line_numbers"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise ValueError("source slot sequence must be finite")
            object.__setattr__(self, name, tuple(value))
        if (not isinstance(self.key, str) or not self.key or not self.obligation_ids
                or not all(isinstance(k, str) and k for k in self.obligation_ids)
                or not isinstance(self.identity, ReceiptIdentity)
                or not isinstance(self.payload, str) or not isinstance(self.page, str)
                or self.payload not in self.page
                or not isinstance(self.recovery_arguments, bytes)
                or type(self.admission_ordinal) is not int or self.admission_ordinal < 1
                or type(self.last_use_ordinal) is not int or self.last_use_ordinal < 0
                or self.previous_mode not in {None, "F", "D", "O"}
                or type(self.recovery_pin) is not bool
                or type(self.plan_only) is not bool
                or self.state not in {"DUE", "RESIDENT", "DEGRADED", "RETIRING",
                                      "DISCHARGED", "SUPERSEDED", "CLOSURE", "PLAN", "OPTIONAL"}):
            raise ValueError("source slot lacks immutable source authority")
        if digest(self.payload.encode()) != self.identity.payload_sha256:
            raise ValueError("capture payload digest differs")
        if line_digests_with_numbers(self.payload, self.identity.first_line) != (
                self.line_digests, self.line_numbers):
            raise ValueError("capture line digests differ")
        if not isinstance(json.loads(self.recovery_arguments), dict):
            raise ValueError("invalid recovery arguments")
        if self.payload_start is not None:
            raw, payload = self.page.encode(), self.payload.encode()
            if (type(self.payload_start) is not int or self.payload_start < 0
                    or raw[self.payload_start:self.payload_start + len(payload)] != payload):
                raise ValueError("source payload interval differs")
        if self.standalone_page is None:
            if self.standalone_payload_start is not None:
                raise ValueError("standalone payload lacks a carrier")
        elif (not isinstance(self.standalone_page, str)
              or type(self.standalone_payload_start) is not int
              or self.standalone_payload_start < 0
              or self.standalone_page.encode()[self.standalone_payload_start:
                  self.standalone_payload_start + len(self.payload.encode())] != self.payload.encode()):
            raise ValueError("standalone payload interval differs")


@dataclass(frozen=True)
class PreparationSnapshot:
    fixed_body: bytes
    sources: tuple[SourceSlot, ...]
    request_id: str
    due_ordinal: int
    byte_cap: int
    protected_paths: tuple[str, ...] = ()
    current_revisions: tuple[tuple[str, str], ...] = ()
    deferred_paths: tuple[str, ...] = ()
    prompt_limit: int = 33_344
    frame_limit: int | None = None
    context_limit: int | None = None
    output_tokens: int | None = None
    layouts: tuple[Literal["plain", "final_cache"], ...] = ("plain",)
    # Position in fixed messages before the opaque newest user suffix/exchange.
    insertion_index: int | None = None
    dialect: str = "llama.cpp"
    reasoning_mode: str = "on"
    bytes_per_token: float = 2.46
    output_reserve: int = 0
    safety_reserve: int = 0
    tokenizer_identity: str = ""
    recovery_allowed: bool = True
    recovery_schema: bytes = b""
    recovery_owner_allowed: bool = True
    # Fixed-message coordinates, explicitly owned by assignment-derived truth.
    # Other fixed messages remain opaque regardless of host-looking contents.
    truth_message_indexes: tuple[int, ...] | None = None
    # Complete native grep pages have no continuation footer to carry this.
    grep_result_digests: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    @property
    def slots(self) -> tuple[SourceSlot, ...]:
        return self.sources

    @property
    def logical_ordinal(self) -> int:
        return self.due_ordinal

    def __post_init__(self) -> None:
        for name in ("sources", "protected_paths", "deferred_paths", "layouts"):
            value = getattr(self, name)
            if not isinstance(value, (list, tuple)):
                raise ValueError("snapshot sequence must be finite")
            object.__setattr__(self, name, tuple(value))
        if not isinstance(self.current_revisions, (list, tuple)):
            raise ValueError("current revisions must be finite")
        object.__setattr__(self, "current_revisions", tuple(tuple(row) for row in self.current_revisions))
        if not isinstance(self.grep_result_digests, (list, tuple)):
            raise TypeError("grep result-set bindings must be a list or tuple")
        grep_digests = tuple(tuple(row) if isinstance(row, (list, tuple)) else ()
                             for row in self.grep_result_digests)
        if (any(len(row) != 2 or not isinstance(row[0], str) or not row[0]
                or not isinstance(row[1], str) or not re.fullmatch(r"[0-9a-f]{24}", row[1])
                for row in grep_digests) or len(dict(grep_digests)) != len(grep_digests)):
            raise ValueError("invalid grep result-set bindings")
        object.__setattr__(self, "grep_result_digests", grep_digests)
        if not isinstance(self.fixed_body, bytes):
            raise ValueError("snapshot provider body must be immutable bytes")
        body = json.loads(self.fixed_body)
        if (not isinstance(body, dict) or not isinstance(body.get("messages"), list)
                or not all(isinstance(row, dict) for row in body["messages"])
                or not all(isinstance(slot, SourceSlot) for slot in self.sources)
                or not isinstance(self.request_id, str) or not self.request_id
                or type(self.byte_cap) is not int or self.byte_cap < 0
                or type(self.due_ordinal) is not int or self.due_ordinal < 1
                or type(self.prompt_limit) is not int or self.prompt_limit < 0
                or len({slot.key for slot in self.sources}) != len(self.sources)
                or not self.layouts or not set(self.layouts) <= {"plain", "final_cache"}
                or self.dialect not in {"llama.cpp", "openai-strict", "openai-compatible"}
                or self.reasoning_mode not in {"on", "off", "passthrough"}
                or not isinstance(self.recovery_schema, bytes)
                or type(self.recovery_allowed) is not bool
                or type(self.recovery_owner_allowed) is not bool):
            raise ValueError("invalid preparation snapshot")
        indexes = self.truth_message_indexes
        if indexes is None:
            # Direct snapshot callers have no captured suffix ownership. Only
            # the canonical StateFrame position is unambiguous in that case.
            indexes = _canonical_truth_indexes(body["messages"])
        if (not isinstance(indexes, (tuple, list)) or len(set(indexes)) != len(indexes)
                or any(type(i) is not int or not 0 <= i < len(body["messages"]) for i in indexes)):
            raise ValueError("invalid delivery-truth ownership")
        object.__setattr__(self, "truth_message_indexes", tuple(indexes))
        for name in ("frame_limit", "context_limit", "output_tokens", "insertion_index"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid snapshot optional limit")
        for name in ("output_reserve", "safety_reserve"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("invalid snapshot reserve")
        if (not isinstance(self.bytes_per_token, (int, float))
                or isinstance(self.bytes_per_token, bool)
                or not 0 < self.bytes_per_token < float("inf")):
            raise ValueError("invalid frozen byte calibration")
        if (any(len(row) != 2 or not all(isinstance(v, str) and v for v in row)
                for row in self.current_revisions)
                or len(dict(self.current_revisions)) != len(self.current_revisions)):
            raise ValueError("invalid current revision bindings")

    @property
    def snapshot_digest(self) -> str:
        extra = f"grep_result_digests={self.grep_result_digests!r}" if self.grep_result_digests else ""
        return digest((repr(self) + extra).encode())

    @property
    def ledger_digest(self) -> str:
        return digest(json.dumps([asdict(s) for s in self.sources],
                    default=lambda value: value.decode(), sort_keys=True).encode())


def _canonical_truth_indexes(messages):
    """The host StateFrame follows the leading system messages and task."""
    # Snapshot-only host projections can contain just the StateFrame. Live
    # suffix ownership is still checked before admitting this coordinate.
    if (len(messages) == 1 and messages[0].get("role") == "user"
            and str(messages[0].get("content", "")).startswith("[StateFrame]\n")):
        return (0,)
    task = next((i for i, m in enumerate(messages) if m.get("role") != "system"), len(messages))
    state = task + 1
    if (state < len(messages) and messages[task].get("role") == "user"
            and str(messages[task].get("content", "")).startswith("Task:")
            and messages[state].get("role") == "user"
            and str(messages[state].get("content", "")).startswith("[StateFrame]\n")):
        return (state,)
    return ()


def _merge_source(sources, proposed):
    """Keep read lifecycle authority while retaining identical closure grammar."""
    for index, slot in enumerate(sources):
        if (slot.identity.path == proposed.identity.path
                and slot.identity.revision == proposed.identity.revision
                and slot.identity.first_line == proposed.identity.first_line
                and slot.identity.last_line == proposed.identity.last_line
                and slot.payload == proposed.payload
                and (slot.identity.lossy_span is None or
                     slot.identity.lossy_span.presentation_version == "model-source-window/1")):
            if slot.capture_ids:
                sources[index] = replace(slot, standalone_page=proposed.page,
                                         standalone_payload_start=proposed.payload_start)
            return
    sources.append(proposed)


def _check_output_paths(output: str, runtime: object, editable: set[str]) -> set[str]:
    """Map diagnostic locations to safe, editable workspace identities."""
    from kernel_request_source_window import _safe_framefit_path

    root_value = getattr(getattr(runtime, "owner", None), "root", None)
    root = None
    if root_value is not None:
        try:
            root = Path(root_value).resolve()
        except (OSError, TypeError, ValueError):
            return set()
    paths = set()
    # Keep quoted traceback names whole; never recover a relative suffix from
    # an unsafe absolute, traversal or Windows-drive diagnostic location.
    locations = re.finditer(
        r'File "([^"\r\n]+)", line [0-9]+|'
        r'(?<!\S)((?:[A-Za-z]:)?[\w./\\-]+\.[\w]+):[0-9]+', output)
    for match in locations:
        value = (match.group(1) or match.group(2)).replace("\\", "/")
        try:
            if value.startswith("/"):
                if root is None:
                    continue
                value = Path(value).relative_to(root).as_posix()
            path = _safe_framefit_path(value)
            if path is None:
                continue
            if root is not None:
                path = _safe_framefit_path((root / path).resolve().relative_to(root).as_posix())
        except (OSError, ValueError):
            continue
        if path is not None and (not editable or path in editable):
            paths.add(path)
    return paths


def _freeze_grep_digests(exchange, owner) -> tuple[tuple[str, str], ...]:
    """Match host page receipts to native result rows before detaching."""
    if not exchange:
        return ()
    grep_calls = {call["id"] for call in exchange[0]["tool_calls"]
                  if isinstance(call.get("function"), dict) and call["function"].get("name") == "grep"}
    pending = {}
    for row in exchange[1:]:
        call_id, content = row["tool_call_id"], row.get("content")
        if call_id not in grep_calls or row.get("name", "grep") != "grep" or not isinstance(content, str):
            continue
        header = content.partition("\n")[0]
        if (header.startswith("[search page] shown=") and " complete=true" in header
                and not header.startswith(("[search page] shown=0 ", "[search page] shown=1 "))):
            pending[call_id] = content
    if not pending:
        return ()
    bindings = []
    for receipt in reversed(getattr(owner, "_receipts", ())):
        page = receipt.get("grep_page") if isinstance(receipt, dict) and receipt.get("kind") == "grep" else None
        if not isinstance(page, dict):
            continue
        call_id = page.get("provider_call_id")
        if not isinstance(call_id, str) or call_id not in pending:
            continue
        content = pending.pop(call_id)
        chars, result = page.get("chars"), receipt.get("result_set_digest")
        if (type(chars) is int and 0 < chars <= len(content)
                and (chars == len(content) or content[chars] == "\n")
                and isinstance(result, str) and re.fullmatch(r"[0-9a-f]{24}", result)
                and digest(content[:chars].encode()) == page.get("sha256")):
            bindings.append((call_id, result))
        if not pending:
            break
    return tuple(sorted(bindings))


def freeze_preparation(body, runtime, *, request_id: str, byte_cap: int,
                       prompt_limit: int = 33_344, frame_limit: int | None = None,
                       layouts=("plain",), dialect="llama.cpp",
                       reasoning_mode="on") -> PreparationSnapshot:
    """Detach authoritative captures and canonical closure windows once.

    Existing source parsers identify host-owned windows. User/tool prose is
    opaque and never classified by a path substring. Missing blobs fail closed.
    """
    from kernel_measurement import _read_journal_blob
    from kernel_read_reuse_visibility import editable_read_paths
    from kernel_request_source_window import (parse_model_source_window, parse_window,
                                              _model_window_segments, _window_segments)
    from kernel_stateframe_framefit5 import (_validated_exchange_and_volatile_tail,
                                             FrameFit5Error)

    frozen = json.loads(json.dumps(body))
    from kernel_working_set_snapshot_compat import snapshot_semantic_messages
    messages = snapshot_semantic_messages(frozen["messages"])
    sources = []
    state = getattr(runtime, "working_set_state", None)
    # An index past the live limit is complete (no capture is ever dropped);
    # its overshoot is retired by the O carriers the allocator must select.
    ledger = getattr(runtime, "read_capture_ledger", None)
    captures = {c.capture_id: c for c in ledger.all()} if ledger is not None else {}
    for obligation in state.active_obligations() if state is not None else ():
        owned = [captures[key] for key in obligation.capture_ids]
        if not owned or any(c.identity != obligation.identity for c in owned):
            raise ValueError("obligation capture identity differs")
        capture = owned[-1]
        payload = json.loads(_read_journal_blob(runtime.journal.path, capture.payload_blob_ref))
        page = json.loads(_read_journal_blob(runtime.journal.path, capture.original_page_blob_ref))
        recovery = json.loads(capture.recovery_arguments)
        # Canonical identity is the resolved authority for an admitted alias,
        # including absolute workspace paths; keep original coordinates intact.
        recovery["path"] = obligation.identity.path
        sources.append(SourceSlot(obligation.obligation_id, (obligation.obligation_id,),
            obligation.identity, payload, page, capture.line_digests, capture.line_numbers,
            json.dumps(recovery).encode(), obligation.admission_ordinal,
            obligation.last_use_ordinal, obligation.state, obligation.recovery_pin,
            "tests" if obligation.identity.path.startswith("tests/") else "product",
            tuple(c.capture_id for c in owned),
            payload_start=(len(page.encode()) - len(payload.encode())
                           if page.endswith(payload) else page.encode().find(payload.encode())),
            previous_mode=obligation.previous_mode))

    # Only assembler-owned user windows before the suffix can be optional.
    try:
        exchange, boundary, _tail = _validated_exchange_and_volatile_tail(messages)
    except (ValueError, FrameFit5Error):
        exchange = ()
        default_boundary = len(messages)
        if messages and str(messages[-1].get("content", "")).startswith("[WIRE VOLATILE TAIL]"):
            default_boundary -= 1
        boundary = next((i for i, m in enumerate(messages)
                         if m.get("role") in {"assistant", "tool"}), default_boundary)
    suffix = list(getattr(getattr(runtime, "_active_capture", None),
                          "_framefit5_newest_user_suffix", ()))
    suffix_start = boundary - len(suffix)
    if suffix and (suffix_start < 0 or messages[suffix_start:boundary] != suffix):
        raise SnapshotConstructionError("SOURCE_SNAPSHOT_UNSTABLE")
    fixed = []
    truth_indexes = []
    owned_truth = {i for i in _canonical_truth_indexes(messages)
                   if i < (suffix_start if suffix else boundary)}
    # Tail ownership is positional, after the newest exchange. A suffix can
    # quote the same marker and must not hide or acquire the real tail owner.
    if (messages and len(messages) - 1 >= boundary
            and messages[-1].get("role") == "user"
            and str(messages[-1].get("content", "")).startswith("[WIRE VOLATILE TAIL]")):
        owned_truth.add(len(messages) - 1)
    revisions = {}
    # Typed current revision observations survive even when their source body
    # was not selected into the legacy source panel.
    owner_actions = None
    pending_site_paths = set()
    for index in sorted(owned_truth):
        message = messages[index]
        content = str(message.get("content", ""))
        if not content.startswith("[StateFrame]\n"):
            continue
        frame = json.loads(content.split("\n", 1)[1]).get("state", {}).get("framefit5", {})
        for fact in frame.get("read_facts", ()):
            if isinstance(fact, dict) and isinstance(fact.get("current_revision"), str):
                revisions[fact.get("path")] = fact["current_revision"]
        sites = list(frame.get("open_sites", ()))
        if isinstance(frame.get("next_site"), dict):
            sites.append(frame["next_site"])
        pending_site_paths.update(site["path"] for site in sites
            if isinstance(site, dict) and isinstance(site.get("path"), str))
        actions = frame.get("owner_protocol", {}).get("admitted_actions")
        if isinstance(actions, list):
            owner_actions = tuple(actions)
    plan = getattr(runtime, "_request_source_closure", None)
    deferred = set(plan.get("deferred_paths", ())) if isinstance(plan, dict) else set()
    insertion = None
    for index, message in enumerate(messages):
        content = message.get("content")
        if suffix_start <= index < boundary and suffix:
            fixed.append(message)
            continue  # opaque user ownership outranks every host/source grammar
        metadata = ((parse_model_source_window(content) or parse_window(content))
                    if message.get("role") == "user" and index < boundary else None)
        if isinstance(metadata, dict) and metadata.get("ranges"):
            path, revision = metadata["path"], metadata["revision"]
            if path in revisions and revisions[path] != revision:
                raise ValueError("source window differs from current revision observation")
            revisions[path] = revision
            # Parse owned bodies using the existing grammar, excluding its end
            # marker and restoring the exact trailing-newline declaration.
            segments = (_model_window_segments(content, metadata)
                        if parse_model_source_window(content) is not None
                        else _window_segments(content, metadata))
            if not segments:
                raise ValueError("source window lacks exact interval carriers")
            for descriptor, payload in segments:
                first, last = descriptor["start_line"], descriptor["end_line"]
                if descriptor.get("trailing_newline") is False and payload.endswith("\n"):
                    payload = payload[:-1]
                # These are decoded source-window coordinates. A closure body
                # alone cannot authenticate its offset in the raw whole file.
                last_column = len(payload.splitlines(keepends=True)[-1]) if payload else 0
                identity = ReceiptIdentity(path, revision, first, last, None, None,
                    digest(payload.encode()), DecodedSpan(0, last_column, "model-source-window/1"))
                key = "closure:" + digest((path + revision + str(first) + str(last) + payload).encode())
                from kernel_working_set_windows import render_frozen_span
                page, payload_start = render_frozen_span(metadata, descriptor, payload)
                digests, numbers = line_digests_with_numbers(payload, first)
                ordinal = max((s.admission_ordinal for s in sources), default=0) + 1
                _merge_source(sources, SourceSlot(key, (key,), identity, payload, page, digests, numbers,
                    json.dumps({"path": path, "start_line": first, "end_line": last}).encode(),
                    ordinal, 0, "CLOSURE", kind="tests" if path.startswith("tests/") else "context",
                    payload_start=payload_start))
            continue
        if index < boundary and message.get("role") == "user" and isinstance(content, str):
            if content.startswith("[Deferred required source paths]"):
                label = "Required source paths are not attached; read one only if your edit needs it:"
                if label in content:
                    deferred.update(line[2:] for line in content.split(label, 1)[1].splitlines() if line.startswith("- "))
                continue
            if content.startswith(("[Host fact]\n", "[Source page deferred]\n",
                                   "[Source coverage]\n")) or content.startswith("kept "):
                continue
        if index == boundary:
            # Source follows StateFrame and precedes the opaque user suffix/exchange.
            insertion = next((i + 1 for i in truth_indexes
                if str(fixed[i].get("content", "")).startswith("[StateFrame]\n")), len(fixed))
        if index in owned_truth:
            truth_indexes.append(len(fixed))
        fixed.append(message)
    if insertion is None:
        insertion = next((i + 1 for i in truth_indexes
            if str(fixed[i].get("content", "")).startswith("[StateFrame]\n")), len(fixed))
    from kernel_working_set_windows import frozen_plan_windows, frozen_embedded_windows
    # Include capture duplicates as presentation proposals. _merge_source keeps
    # the retained obligation's mode/domain; a closure cannot promote it.
    proposed, plan_revisions = frozen_plan_windows(runtime,
        tuple(s for s in sources if not s.capture_ids), revisions)
    for proposal in proposed:
        _merge_source(sources, replace(proposal,
            admission_ordinal=max((s.admission_ordinal for s in sources), default=0) + 1))
    revisions.update(plan_revisions)
    if getattr(runtime, "owner", None) is not None:
        embedded, embedded_revisions = frozen_embedded_windows(runtime,
            [messages[i] for i in sorted(owned_truth)], tuple(sources), revisions)
        sources.extend(embedded)
        revisions.update(embedded_revisions)
    frozen["messages"] = fixed
    # Protection follows the pending edit surface, rather than every file the
    # workspace-wide edit authority would permit modifying.
    from kernel_request_source_window import _safe_framefit_path
    editable = editable_read_paths(runtime)
    protected = {path for path in pending_site_paths
                 if _safe_framefit_path(path) == path and (not editable or path in editable)}
    batch = getattr(runtime, "native_batch", None)
    for call in getattr(batch, "pending", ()):
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or function.get("name") not in {
                "edit", "str_replace", "insert_before", "file_create", "file_delete", "file_move"}:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                continue
        if not isinstance(arguments, dict):
            continue
        for key in ("path", "source_path", "destination_path"):
            path = _safe_framefit_path(arguments.get(key))
            if path is not None and (not editable or path in editable):
                protected.add(path)
    # Check receipts keep bounded, private stdout/stderr in insertion order.
    # Protect traceback paths from the newest failing check, not older failures.
    private = getattr(runtime, "_check_output_receipts", {})
    if isinstance(private, dict):
        for check in reversed(tuple(private.values())):
            if isinstance(check, dict) and check.get("exit_code") not in {None, 0}:
                output = str(check.get("stdout", "")) + "\n" + str(check.get("stderr", ""))
                protected.update(_check_output_paths(output, runtime, editable))
                break
    return PreparationSnapshot(json.dumps(frozen).encode(), tuple(sources), request_id,
        state.next_due_ordinal if state is not None else 1, byte_cap,
        tuple(sorted(protected)), tuple(sorted(revisions.items())), tuple(sorted(deferred)),
        prompt_limit, frame_limit, layouts=tuple(layouts), insertion_index=insertion,
        dialect=dialect, reasoning_mode=reasoning_mode,
        truth_message_indexes=tuple(truth_indexes),
        grep_result_digests=_freeze_grep_digests(exchange, getattr(runtime, "owner", None)),
        recovery_allowed=any(isinstance(t, dict) and isinstance(t.get("function"), dict)
            and t["function"].get("name") == "read" for t in frozen.get("tools", ())),
        recovery_owner_allowed=owner_actions is not None and "read" in owner_actions,
        recovery_schema=next((json.dumps(t["function"].get("parameters", {})).encode()
            for t in frozen.get("tools", ()) if isinstance(t, dict)
            and isinstance(t.get("function"), dict) and t["function"].get("name") == "read"), b""))
