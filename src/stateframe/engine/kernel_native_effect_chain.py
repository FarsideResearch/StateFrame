"""Historical source replay for ordered native mutation contracts.

Only journal-bound bytes enter this planner. It never reads the current
workspace to manufacture source visibility or authorize an unexplained change.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import PurePosixPath
import textwrap

from kernel_continuation_menu_authority import EDIT_TOOLS
from kernel_continuation_state import ContinuationError
from kernel_native_prefix import argument_hash, delivered_source, require
from kernel_native_prefix_delivery import _acquired_source, _anchor_delivered


def revision(source: str) -> str:
    return "sha256:" + hashlib.sha256(source.encode()).hexdigest()


def canonical_path(path: object) -> str:
    require(isinstance(path, str) and bool(path) and path != "." and "\\" not in path
            and "\x00" not in path and str(PurePosixPath(path)) == path
            and not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts,
            "noncanonical target")
    return path


def edit_authority_mode(origin: dict, recorded_mode: object = None) -> str:
    """The EditAuthority mode the executor honoured for `edit` in the proposing frame.

    New batches journal the host contract alongside their origin binding. Older
    bindings may carry it directly. Tool prose and model arguments are not authority.
    """
    if isinstance(recorded_mode, str):
        return recorded_mode.replace("-", "_").lower()
    authority = origin.get("edit_authority")
    if isinstance(authority, dict) and isinstance(authority.get("mode"), str):
        return authority["mode"].replace("-", "_").lower()
    return "definition"


def whole_file_change(path: str, source: str, args: dict) -> tuple[int, int, str]:
    """Mirror the executor's whole-file replacement: function_source becomes the file, byte-exact."""
    replacement = args.get("function_source")
    require(isinstance(replacement, str), "invalid whole-file edit")
    if PurePosixPath(path).suffix == ".py":
        try:
            ast.parse(replacement)
        except SyntaxError as error:
            raise ContinuationError("native prefix: invalid whole-file source") from error
    return 0, len(source), replacement


def definition_change(path: str, source: str, args: dict) -> tuple[int, int, str]:
    """Use the executor's complete-definition normalizer, preserving exterior bytes."""
    from kernel_edit_normalization import NormalizedEdit, normalize_edit

    replacement = args.get("function_source")
    require(isinstance(replacement, str) and bool(replacement.strip()), "empty definition edit")
    replacement = replacement.strip()
    if PurePosixPath(path).suffix != ".py":
        return 0, len(source), replacement
    replacement = textwrap.dedent(replacement)
    try:
        definitions = [node for node in ast.parse(replacement).body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    except SyntaxError as error:
        raise ContinuationError("native prefix: invalid definition source") from error
    require(len(definitions) == 1, "definition edit requires one unambiguous target")
    ref = args.get("target_ref")
    target = dict(ref) if isinstance(ref, dict) else {"name": definitions[0].name}
    normalized = normalize_edit(
        {"sources": {path: source.encode()}, "allowed_paths": [path]}, source.encode(),
        {"path": path, "mode": "complete-definition", "candidate": replacement.encode(),
         "supports_definition": True, "target": target, "expected_base_digest": revision(source)})
    require(isinstance(normalized, NormalizedEdit), "definition edit target is stale or ambiguous")
    span = normalized.preservation_proof["target_span"]
    raw = source.encode()
    start, end = len(raw[:span["start"]].decode()), len(raw[:span["end"]].decode())
    return start, end, replacement


def insertion_change(path: str, source: str, args: dict) -> tuple[int, int, str]:
    """Validate the same exact revision/boundary reference as compose_candidate."""
    from kernel_revision_bound_edit import SiteReference, bind_site

    replacement, value = args.get("new_str"), args.get("target_ref")
    require(isinstance(replacement, str) and bool(replacement.strip()) and isinstance(value, dict),
            "invalid revision-bound insertion")
    try:
        ref = SiteReference.from_mapping(value)
        require(ref.path == path and ref.file_revision == revision(source)
                and ref.source_digest == revision(source) and ref.is_boundary
                and "insert_before" in ref.permitted_operations, "stale insertion boundary")
        expected = bind_site(path=path, file_revision=revision(source), source=source,
            start=ref.start, end=ref.end, boundary=ref.boundary, node_kind=ref.node_kind,
            source_receipt=ref.source_receipt, permitted_operations=ref.permitted_operations,
            authority_ref=ref.authority_ref, display_line=ref.display_line,
            attached_digest=ref.attached_digest, dependency_digest=ref.dependency_digest)
        require(ref == expected, "insertion reference differs from bound source")
        start = len(source.encode()[:ref.start].decode())
    except (TypeError, ValueError, UnicodeError) as error:
        raise ContinuationError("native prefix: invalid or stale insertion reference") from error
    return start, start, replacement


def historical_page_anchor(messages: list[dict], path: str, source: str,
                           bound: str, anchor: str) -> bool:
    """Qualify whole-line read pages against the same bound source as windows."""
    from kernel_request_source_window import _READ_PAGE_HEADER, _read_page_source_range

    if not anchor:
        return False
    for message in messages:
        content = message.get("content")
        if message.get("role") not in {"user", "tool"} or not isinstance(content, str):
            continue
        header = _READ_PAGE_HEADER.fullmatch(content.partition("\n")[0].removesuffix("\r"))
        if header is None or header["revision"] != bound:
            continue
        verified = _read_page_source_range(content, {path: source}, {path: bound})
        if verified is not None and verified[0] == path:
            first, last = verified[1]
            if anchor in "".join(source.splitlines(keepends=True)[first - 1:last]):
                return True
    return False


class OrderedSources:
    def __init__(self, events, prepared, payload, origin_binding, *, final_delivery,
                 source_evidence=None, edit_mode: object = None):
        self.events, self.prepared, self.payload = events, prepared, payload
        self.source_evidence = source_evidence
        self.origin = origin_binding
        self.edit_mode = edit_authority_mode(origin_binding, edit_mode)
        self.final_delivery = final_delivery
        self.texts: dict[str, str] = {}
        self.original: dict[str, str] = {}
        self.visible: set[str] = set()
        self.regions: dict[str, list[tuple[int, int, int]]] = {}
        self.seen: set[tuple[str, str]] = set()

    def source(self, path: str) -> str:
        if path not in self.texts:
            bound = self.origin.get("product_files", {}).get(path)
            require(isinstance(bound, str), "target is not a bound product file")
            try:
                source = delivered_source(self.payload.get("messages", []), path, bound)
                self.visible.add(path)
            except ContinuationError:
                if not self.final_delivery:
                    raise
                events = (self.source_evidence.acquired_source_events(
                    path, bound, self.prepared["sequence"])
                    if self.source_evidence is not None else self.events)
                source = _acquired_source(events, path, bound, self.prepared["sequence"])
            self.texts[path] = self.original[path] = source
        return self.texts[path]

    def historical_anchor(self, path: str, anchor: str) -> bool:
        """Accepted, immutable past wires retain authority after source eviction."""
        from kernel_working_set_state import enabled
        from kernel_native_prefix import one
        from kernel_native_prefix_delivery import submitted_payload, validate_prepared_delivery

        if not enabled() or self.source_evidence is None:
            return False
        evidence = self.source_evidence
        bound, source = self.origin["product_files"][path], self.original[path]
        receipts = evidence.events_of_kinds(frozenset({"working_set_response"}))
        for receipt in reversed(receipts):
            if (receipt.get("disposition") != "accepted"
                    or not isinstance(receipt.get("body_sha256"), str)
                    or receipt["sequence"] >= self.prepared["sequence"]):
                continue
            try:
                response = evidence.event(receipt.get("response_id"), "response_received")
                submitted = evidence.event(response.get("transport_attempt_id"), "request_submitted")
                require(submitted["sequence"] < response["sequence"] < receipt["sequence"]
                        and receipt.get("request_id") == response.get("request_id")
                        == submitted.get("request_id")
                        and submitted.get("payload_hash") == "sha256:" + receipt.get("body_sha256", ""),
                        "historical exposure identity differs")
                payload = submitted_payload(evidence, submitted)
                prepared = one([event for event in evidence.native_validation_events(receipt["request_id"])
                                if event["kind"] == "request_prepared"],
                               "missing or duplicate historical prepared request")
                require(validate_prepared_delivery(evidence.final_delivery_events(submitted),
                        prepared, submitted, payload), "historical final delivery is missing")
            except ContinuationError:
                continue  # An incomplete/rejected carrier cannot grant authority.
            messages = payload.get("messages", [])
            if (_anchor_delivered(messages, path, source, bound, anchor)
                    or historical_page_anchor(messages, path, source, bound, anchor)):
                return True
            try:
                delivered_source(messages, path, bound)
            except ContinuationError:
                continue
            if anchor and anchor in source:
                return True
        return False

    def apply(self, tool: str, args: dict, number: int, *, pending: bool) -> str | None:
        if tool not in EDIT_TOOLS:
            return None
        path = canonical_path(args.get("path"))
        fingerprint = (tool, argument_hash(args))
        require(fingerprint not in self.seen,
                f"call {number + 1} not applied: repeated identical call")
        self.seen.add(fingerprint)
        if tool == "file_create":
            require(path not in self.origin.get("product_files", {})
                    and path not in self.origin.get("harness_files", {}) and path not in self.texts,
                    f"call {number + 1} not applied: file already exists")
            content = args.get("content")
            require(isinstance(content, str), "invalid create content")
            self.texts[path] = content
            # The response authors these bytes; no historical read is claimed.
            self.visible.add(path)
            return path
        source = self.source(path)
        if tool == "str_replace":
            old, replacement = args.get("old_str"), args.get("new_str")
            require(isinstance(old, str) and bool(old) and isinstance(replacement, str),
                    "invalid original exact-edit precondition")
            require(source.count(old) == 1,
                    f"call {number + 1} not applied: old_str is no longer unique; re-issue against current text")
            start, end = source.index(old), source.index(old) + len(old)
        elif tool == "edit" and self.edit_mode == "whole_file":
            start, end, replacement = whole_file_change(path, source, args)
        elif tool == "edit":
            start, end, replacement = definition_change(path, source, args)
        else:
            start, end, replacement = insertion_change(path, source, args)
        if pending and path not in self.visible:
            anchor = source[start:end] if start != end else source[max(0, start-32):start+32]
            require(_anchor_delivered(self.payload.get("messages", []), path, self.original[path],
                    self.origin["product_files"][path], anchor)
                    or self.historical_anchor(path, anchor), "pending exact-edit anchor was not delivered")
        regions = self.regions.setdefault(path, [])
        for prior, left, right in regions:
            overlap = (start < right and end > left) if start != end and left != right else (
                left < start < right if start == end else start < left < end)
            require(not overlap, f"call {number + 1} not applied: overlaps call {prior + 1}; re-issue against current text")
        delta = len(replacement) - (end - start)
        self.regions[path] = [(prior, left+delta if end <= left else left,
                              right+delta if end <= left else right) for prior, left, right in regions]
        self.regions[path].append((number, start, start + len(replacement)))
        self.texts[path] = source[:start] + replacement + source[end:]
        return path
