"""Host-only StateFrame v2 S1 component map and open-anomaly carry.

S1 deliberately has no model-facing input.  It reads the brief, current file bytes,
AST bindings, shared structural detectors, and action/test receipts.  The small
rendered sidecar is appended after arm-specific selection so it is refreshed for
both provider arms and cannot be evicted by history selection.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
import re
from collections.abc import Mapping, Sequence

MODE_ENV = "STATEFRAME_V2_S1"
MAX_TOKENS = 450
_NAME = r"[A-Za-z_]\w*"
_CLASS_RE = re.compile(r"\bclass\s+(" + _NAME + r")")
_CALL_RE = re.compile(r"\b(" + _NAME + r")\s*(?=\()")
# A period in prose is not an attribute access.  Requiring an identifier on
# both sides and no whitespace after the dot keeps spans such as ``Job. urgent``
# out of the structural map while retaining real ``queue.dead_letter`` code.
_DOT_RE = re.compile(r"\b(" + _NAME + r")\.((?:" + _NAME + r"))\b")
_WORD_RE = re.compile(r"\b(" + _NAME + r")\b")
_EXPLICIT_METHOD_RE = re.compile(r"\b(?:def|function|method)\s+(" + _NAME + r")")
_FIELD_RE = re.compile(r"\b(?:field|fields|attribute|attributes)\s+([^.;]+)", re.I)
_STOP_WORDS = {
    "and", "or", "the", "with", "returns", "return", "counts", "count", "value",
    "values", "jobs", "items", "records", "record", "objects", "object", "a", "an",
    "on", "to", "of", "from", "as", "is", "in", "at", "by", "until", "then",
}
_GOOD_RESULTS = {"OK", "GREEN", "PASSED", "PASS", "SUCCESS", "REUSED_VERIFICATION", "APPLIED"}
_EDIT_KINDS = {"edit", "str_replace", "file_create"}


def stateframe_v2_s1_mode() -> str:
    """Return the strict S1 switch; the absent switch is the byte-preserving OFF arm."""
    value = os.environ.get(MODE_ENV, "off")
    if value not in {"off", "on"}:
        raise ValueError(f"{MODE_ENV} must be off or on")
    return value


def enabled() -> bool:
    return stateframe_v2_s1_mode() == "on"


def _compact(value: object, limit: int = 96) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _brief_text(brief: object) -> str:
    return brief if isinstance(brief, str) else ""


def _brief_names(brief: str) -> dict[str | None, dict[str, list[str]]]:
    """Use the existing conservative brief parser when it is available."""
    try:
        from kernel_completion_disposition import brief_named_elements
        parsed = brief_named_elements(brief)
        return parsed if isinstance(parsed, dict) else {}
    except (ImportError, AttributeError, TypeError):
        return {}


def _span_for(segment: str, offset: int, name: str, *, dotted: bool = False) -> tuple[int, int] | None:
    pattern = (r"\.\s*" + re.escape(name) + r"\b") if dotted else r"\b" + re.escape(name) + r"\b"
    match = re.search(pattern, segment)
    if match is None:
        return None
    return offset + match.start(), offset + match.end()


def _declared_field_names(text: str) -> list[str]:
    """Read field names at prose level, excluding type/English words in parens."""
    names: list[str] = []
    depth = 0
    expect_name = True
    index = 0
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
            index += 1
            continue
        if char == ")":
            depth = max(0, depth - 1)
            index += 1
            continue
        if depth == 0 and char in ".;":
            break
        if depth == 0 and char == ",":
            expect_name = True
            index += 1
            continue
        match = re.match(_NAME, text[index:]) if depth == 0 else None
        if match is not None:
            word = match.group(0)
            lower = word.lower()
            if expect_name and lower not in _STOP_WORDS:
                names.append(word)
                expect_name = False
            elif lower in {"and", "or"}:
                expect_name = True
            index += len(word)
            continue
        index += 1
    return list(dict.fromkeys(names))


def _brief_requirements(brief: str, path: str) -> list[dict]:
    """Capture exact brief spans, without interpreting prose as implementation."""
    parsed = _brief_names(brief)
    requirements: list[dict] = []
    class_matches = list(_CLASS_RE.finditer(brief))
    seen: set[tuple[str, str, str | None]] = set()

    def add(kind: str, owner: str | None, name: str, span: tuple[int, int] | None) -> None:
        key = (kind, name, owner)
        if span is None or key in seen:
            return
        seen.add(key)
        start, end = span
        requirements.append({
            "kind": kind, "owner": owner, "name": name,
            "brief_span": {"start": start, "end": end, "text": brief[start:end]},
            "path": path,
        })

    for index, class_match in enumerate(class_matches):
        owner = class_match.group(1)
        segment_end = class_matches[index + 1].start() if index + 1 < len(class_matches) else len(brief)
        segment_start = class_match.start()
        segment = brief[segment_start:segment_end]
        add("class", None, owner, class_match.span())
        values = parsed.get(owner, {})
        methods = list(values.get("methods", ())) if isinstance(values, Mapping) else []
        methods.extend(match.group(1) for match in _EXPLICIT_METHOD_RE.finditer(segment))
        methods.extend(match.group(1) for match in _CALL_RE.finditer(segment))
        for name in methods:
            if name != owner and name.lower() not in _STOP_WORDS:
                add("function", owner, name, _span_for(segment, segment_start, name))

        attributes = list(values.get("attributes", ())) if isinstance(values, Mapping) else []
        attributes.extend(match.group(2) for match in _DOT_RE.finditer(segment))
        for field_match in _FIELD_RE.finditer(segment):
            attributes.extend(_declared_field_names(field_match.group(1)))
        for name in attributes:
            span = _span_for(segment, segment_start, name, dotted=True) or _span_for(
                segment, segment_start, name)
            add("attribute", owner, name, span)

    module_values = parsed.get(None, {})
    if isinstance(module_values, Mapping):
        for name in module_values.get("methods", ()):
            add("function", None, name, _span_for(brief, 0, name))
        for name in module_values.get("attributes", ()):
            add("attribute", None, name, _span_for(brief, 0, name, dotted=True)
                or _span_for(brief, 0, name))

    kind_order = {"class": 0, "function": 1, "attribute": 2}
    requirements.sort(key=lambda row: (row["brief_span"]["start"], kind_order[row["kind"]],
                                       row["owner"] or "", row["name"]))
    for row in requirements:
        identity = "\0".join((row["path"], row["kind"], row["owner"] or "", row["name"],
                               str(row["brief_span"]["start"]), str(row["brief_span"]["end"])))
        row["id"] = "C-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    return requirements


def _binding_nodes(source: str) -> tuple[dict[str, dict], set[str], set[str]]:
    """Return live AST bindings, duplicate qualified names, and live stub names."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}, set(), set()
    from kernel_stub_gate import duplicate_definitions, unimplemented_functions

    all_bindings: dict[str, list[dict]] = {}

    def add(name: str, node: ast.AST, kind: str) -> None:
        all_bindings.setdefault(name, []).append({
            "qualified": name, "kind": kind, "line": int(getattr(node, "lineno", 0)),
            "column": int(getattr(node, "col_offset", 0)),
        })

    def visit(body: Sequence[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                qualified = prefix + node.name
                add(qualified, node, "class")
                visit(node.body, qualified + ".")
                for child in ast.walk(node):
                    if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                        if child.value.id in {"self", node.name}:
                            add(qualified + "." + child.attr, child, "attribute")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = prefix + node.name
                add(qualified, node, "function")
                visit(node.body, qualified + ".")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        add(prefix + target.id, target, "attribute")

    visit(tree.body)
    live = {name: rows[-1] for name, rows in all_bindings.items()}
    for name, row in live.items():
        row = dict(row)
        row["count"] = len(all_bindings[name])
        live[name] = row
    duplicates = set(duplicate_definitions(source))
    stubs = set(unimplemented_functions(source))
    return live, duplicates, stubs


def _revision_values(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, Mapping):
        result: set[str] = set()
        for nested in value.values():
            result.update(_revision_values(nested))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: set[str] = set()
        for nested in value:
            result.update(_revision_values(nested))
        return result
    return set()


def _receipt_id(receipt: Mapping[str, object]) -> str:
    value = receipt.get("receipt_id", receipt.get("event_id", ""))
    return _compact(value, 48) or "UNLOCATED_RECEIPT"


def _current_recheck(receipt: Mapping[str, object], revision: str) -> bool:
    """Accept only a receipt that explicitly binds its observation to current bytes."""
    values = _revision_values(receipt)
    raw_revision = revision.removeprefix("sha256:")
    if revision in values or raw_revision in values:
        return True
    for key in ("current_bytes_sha256", "current_revision", "after_revision", "product_revision"):
        value = receipt.get(key)
        if value in {revision, raw_revision}:
            return True
    return receipt.get("current_bytes_recheck") is True and receipt.get("status") in _GOOD_RESULTS


def _evidence_receipts(receipts: Sequence[object], revision: str) -> list[str]:
    refs: list[str] = []
    for item in receipts:
        if not isinstance(item, Mapping):
            continue
        result = str(item.get("result_class", item.get("result", item.get("status", "")))).upper()
        kind = str(item.get("kind", item.get("tool", ""))).lower()
        if result not in _GOOD_RESULTS and not (kind in {"test", "run_tests", "probe"}
                                                and item.get("exit_code") == 0):
            continue
        if _current_recheck(item, revision):
            refs.append(_receipt_id(item))
    return list(dict.fromkeys(refs))


def _path_matches(path: str, receipt: Mapping[str, object]) -> bool:
    target = receipt.get("target_path", receipt.get("path"))
    return target is None or str(target).lstrip("./") == path.lstrip("./")


def _resolution_receipts(receipts: Sequence[object], revision: str, path: str) -> list[str]:
    return [_receipt_id(item) for item in receipts if isinstance(item, Mapping)
            and _path_matches(path, item) and _current_recheck(item, revision)
            and str(item.get("kind", item.get("tool", ""))).lower()
            in {"read", "test", "run_tests", "probe", "edit", "str_replace", "file_create"}]


class S1Store:
    """Run-local deterministic store.  Open rows are never closed by disappearance alone."""

    def __init__(self) -> None:
        self.components: list[dict] = []
        self.anomalies: dict[str, dict] = {}
        self.revision: str | None = None
        self.path: str | None = None

    def _anomaly(self, *, kind: str, symbol: str, path: str, class_name: str,
                 revision: str, receipt_refs: Sequence[str] = ()) -> dict:
        identity = "\0".join((kind, path, symbol))
        anomaly_id = "A-" + hashlib.sha256(identity.encode()).hexdigest()[:12]
        return {"id": anomaly_id, "kind": kind, "symbol": symbol, "path": path,
                "class": class_name, "status": "OPEN", "first_revision": revision,
                "last_revision": revision, "receipt_refs": list(dict.fromkeys(receipt_refs))}

    def _observe(self, finding: dict) -> None:
        row = self.anomalies.get(finding["id"])
        if row is None or row.get("status") == "CLOSED":
            self.anomalies[finding["id"]] = finding
            return
        row["last_revision"] = finding["last_revision"]
        row["receipt_refs"] = list(dict.fromkeys(row.get("receipt_refs", [])
                                                  + finding.get("receipt_refs", [])))

    def update(self, *, brief: str, sources: Mapping[str, str], path: str,
               receipts: Sequence[object] = (), host_findings: Sequence[object] = ()) -> dict:
        source = sources.get(path, "")
        revision = "sha256:" + hashlib.sha256(source.encode()).hexdigest()
        self.path, self.revision = path, revision
        from kernel_edit_guard import host_observed_findings
        requirements = _brief_requirements(brief, path)
        bindings: dict[str, dict] = {}
        duplicate_keys: set[str] = set()
        stub_keys: set[str] = set()
        source_revisions: dict[str, str] = {}
        shared_findings: dict[str, list[str]] = {}
        for source_path, text in sorted(sources.items()):
            live, duplicate_names, stub_names = _binding_nodes(text)
            shared_findings[source_path] = host_observed_findings(text, force=True)
            bindings.update({f"{source_path}:{name}": {**row, "path": source_path}
                             for name, row in live.items()})
            duplicate_keys.update(
                f"{source_path}:{name}" for name in duplicate_names
                if any(f"{name} defined" in finding for finding in shared_findings[source_path]))
            stub_keys.update(
                f"{source_path}:{name}" for name in stub_names
                if any(f"live {name} " in finding for finding in shared_findings[source_path]))
            source_revisions[source_path] = "sha256:" + hashlib.sha256(text.encode()).hexdigest()

        evidence_refs = _evidence_receipts(receipts, revision)
        components: list[dict] = []
        for requirement in requirements:
            owner = requirement["owner"]
            qualified = requirement["name"] if owner is None else owner + "." + requirement["name"]
            key = f"{path}:{qualified}"
            binding = bindings.get(key)
            owner_key = f"{path}:{owner}" if owner else ""
            is_duplicate = key in duplicate_keys or owner_key in duplicate_keys
            is_stub = key in stub_keys
            if binding is None:
                state = "unlocated"
            elif is_stub or is_duplicate:
                state = "witnessed_defect"
            elif evidence_refs:
                state = "evidence_available"
            else:
                state = "present"
            components.append({**requirement, "state": state,
                               "binding": dict(binding) if binding is not None else None,
                               "receipt_refs": list(evidence_refs)})
        self.components = sorted(components, key=lambda row: (
            row["path"], row["brief_span"]["start"], row["kind"], row["name"]))

        for source_path, text in sorted(sources.items()):
            source_revision = source_revisions[source_path]
            live, duplicate_names, stub_names = _binding_nodes(text)
            for name in sorted(duplicate_names):
                if f"{source_path}:{name}" not in duplicate_keys:
                    continue
                self._observe(self._anomaly(kind="duplicate_binding", symbol=name, path=source_path,
                                            class_name="STRUCTURAL", revision=source_revision))
            for name in sorted(stub_names):
                if f"{source_path}:{name}" not in stub_keys:
                    continue
                self._observe(self._anomaly(kind="live_stub", symbol=name, path=source_path,
                                            class_name="STRUCTURAL", revision=source_revision))
            for requirement in self.components:
                if requirement["path"] != source_path or requirement["state"] != "unlocated":
                    continue
                symbol = (requirement["owner"] + "." if requirement["owner"] else "") + requirement["name"]
                self._observe(self._anomaly(kind="missing_public_symbol", symbol=symbol, path=source_path,
                                            class_name="STRUCTURAL", revision=source_revision))

        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                continue
            kind = str(receipt.get("kind", receipt.get("tool", ""))).lower()
            result = str(receipt.get("result_class", receipt.get("result", ""))).upper()
            detail = str(receipt.get("detail", "")).upper()
            if kind in _EDIT_KINDS and (result in {"REJECTED", "NOT_APPLIED", "REFUSED"}
                                        or "REJECTED" in detail or "NOT_APPLIED" in detail):
                target = str(receipt.get("target_path", receipt.get("path", path)))
                self._observe(self._anomaly(kind="edit_rejection", symbol=kind, path=target,
                                            class_name="STRUCTURAL", revision=revision,
                                            receipt_refs=[_receipt_id(receipt)]))

        # Existing smoke checks are host execution observations.  They are kept
        # distinct from the AST-only structural absence above.
        execution_keys: set[tuple[str, str]] = set()
        for raw in host_findings:
            finding = str(raw)
            match = re.search(
                r"brief-named\s+(?:class|method|attribute)\s+"
                r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", finding)
            if match is None:
                continue
            finding_path = path
            path_match = re.search(r" at ([^:]+):", finding)
            if path_match is not None:
                finding_path = path_match.group(1)
            execution_keys.add((finding_path, match.group(1)))
            self._observe(self._anomaly(kind="public_symbol_execution", symbol=match.group(1),
                                        path=finding_path, class_name="EXECUTION_WITNESSED",
                                        revision=revision))

        for finding in self.anomalies.values():
            if finding.get("status") != "OPEN":
                continue
            active = finding["path"] in sources and (
                (f"{finding['path']}:{finding['symbol']}" in duplicate_keys) or
                (f"{finding['path']}:{finding['symbol']}" in stub_keys) or
                (finding["kind"] == "public_symbol_execution"
                 and (finding["path"], finding["symbol"]) in execution_keys) or
                any(row["path"] == finding["path"] and row["state"] == "unlocated"
                    and ((row["owner"] + "." if row["owner"] else "") + row["name"] == finding["symbol"])
                    for row in self.components))
            if active:
                continue
            refs = _resolution_receipts(receipts, source_revisions.get(finding["path"], revision),
                                        finding["path"])
            if refs:
                finding["status"] = "CLOSED"
                finding["closed_receipt_refs"] = refs
                finding["last_revision"] = source_revisions.get(finding["path"], revision)

        return self.snapshot()

    def snapshot(self) -> dict:
        return {"schema": "stateframe-v2-s1/1", "authority": "HOST_CAPTURED",
                "components": [dict(row) for row in self.components],
                "anomalies": [dict(row) for row in sorted(self.anomalies.values(), key=lambda row: row["id"])
                               if row.get("status") == "OPEN"]}

    def render(self) -> str:
        snapshot = self.snapshot()
        lines = ["[StateFrame v2 S1 | AUTHORITY: HOST_CAPTURED]",
                 "MAP | AUTHORITY: BRIEF_SPANS + AST_BINDINGS"]
        for row in snapshot["components"]:
            span = row["brief_span"]
            binding = row.get("binding")
            binding_text = "-" if binding is None else (
                f"{binding.get('path', row['path'])}:{binding.get('line', 0)}:{binding.get('column', 0)}")
            refs = ",".join(row.get("receipt_refs", ()))
            receipt_text = f" r={refs}" if refs else ""
            lines.append(f"{row['id']} {row['kind']} {row['name']} "
                         f"b={span['start']}:{span['end']}:{_compact(span['text'], 34)!r} "
                         f"a={binding_text} s={row['state']}{receipt_text}")
        lines.append("ANOMALY_CARRY | AUTHORITY: HOST_OBSERVED | OPEN")
        for row in snapshot["anomalies"]:
            refs = ",".join(row.get("receipt_refs", ()))
            receipt_text = f" r={refs}" if refs else ""
            lines.append(f"{row['id']} kind={row['kind']} symbol={row['symbol']} "
                         f"class={row['class']}{receipt_text}")
        lines.append("MODEL_TEXT_AUTHORITY=NONE; facts")
        return "\n".join(lines)


def sidecar_message(store: S1Store) -> dict:
    return {"role": "user", "content": store.render()}


def append_sidecar(messages: Sequence[dict], store: object, *, provider_request: bool = True) -> list[dict]:
    """Compose the final arm payload boundary used by StateFrameRuntime."""
    result = [copy.deepcopy(message) for message in messages]
    if provider_request and store is not None:
        result.append(sidecar_message(store))
    return result


__all__ = ["MAX_TOKENS", "MODE_ENV", "S1Store", "append_sidecar", "enabled",
           "sidecar_message", "stateframe_v2_s1_mode"]
