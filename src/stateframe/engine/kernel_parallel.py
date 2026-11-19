"""Parent-owned subframe coordination and durable publication."""
from __future__ import annotations

import dataclasses
import ast
import base64
import binascii
import copy
import errno
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import sys
import tempfile
import time
from typing import Callable, Mapping, Protocol, Sequence

if __package__:
    from .kernel_fanout_continuation import WRITER_DETAIL_FIELDS, WRITER_SCOPE_FIELDS, WRITER_UNIT_FIELDS
    from .kernel_parallel_admission import DirectorySnapshot, _definitions, _safe_path, admit, rescan_proposal, scan_coupling
    from .kernel_parallel_contracts import _module_map, project_interface, verify_contracts
    from .kernel_parallel_projection import project_unit
    from .kernel_parallel_budget import BudgetStore
    from .kernel_parallel_records import (
        AdmissionResult, CanonicalSignature, CheckReceipt, CheckRecipe, EngineBinding, ExecutionState,
        ConsumeSpec, ExportSpec, FileDelta, HostAnswer, HostRequest, ParallelLimits, ParallelMeasured,
        ParallelRequest, ParallelResult, Proposal, ProposalEntry, Refusal, RequirementClause, TxnRecord, UnitContract, UnitOutcome,
        UnitUsage, MAX_PARALLEL_UNITS, canonical_json, format_parallel_limit,
    )
    from .kernel_parallel_txn import TxnStore, TxnStoreError, manifest_digest, pre_manifest_blob
    from .kernel_tool_tape import ToolReplayRefused, tape_for_run
    from .kernel_journal_store import journal_paths, open_journal
    from .kernel_exec_tracker import ProcessTable, _boot_id, load_tolerant_seconds
else:
    from kernel_fanout_continuation import WRITER_DETAIL_FIELDS, WRITER_SCOPE_FIELDS, WRITER_UNIT_FIELDS
    from kernel_parallel_admission import DirectorySnapshot, _definitions, _safe_path, admit, rescan_proposal, scan_coupling
    from kernel_parallel_contracts import _module_map, project_interface, verify_contracts
    from kernel_parallel_projection import project_unit
    from kernel_parallel_budget import BudgetStore
    from kernel_parallel_records import (
        AdmissionResult, CanonicalSignature, CheckReceipt, CheckRecipe, EngineBinding, ExecutionState,
        ConsumeSpec, ExportSpec, FileDelta, HostAnswer, HostRequest, ParallelLimits, ParallelMeasured,
        ParallelRequest, ParallelResult, Proposal, ProposalEntry, Refusal, RequirementClause, TxnRecord, UnitContract, UnitOutcome,
        UnitUsage, MAX_PARALLEL_UNITS, canonical_json, format_parallel_limit,
    )
    from kernel_parallel_txn import TxnStore, TxnStoreError, manifest_digest, pre_manifest_blob
    from kernel_tool_tape import ToolReplayRefused, tape_for_run
    from kernel_journal_store import journal_paths, open_journal
    from kernel_exec_tracker import ProcessTable, _boot_id, load_tolerant_seconds


class ParallelError(RuntimeError):
    """A subframe call could not be admitted or completed safely."""


class ParallelInputError(ParallelError):
    """A model-authored argument is invalid; execution has not begun."""

    def __init__(self, unit_id: str, field: str, detail: str, hint: str):
        self.unit_id = unit_id
        self.field = field
        self.detail = detail
        self.hint = hint
        super().__init__(f"subframe {unit_id} {field}: {detail}; fix: {hint}")


class ParallelReceiptMissing(ParallelError):
    """A parallel exchange has no durable receipt; the fold records a typed refusal instead of crashing."""


_REQUIRED_PROPERTY = re.compile(r"'([A-Za-z_][A-Za-z0-9_]{0,63})' is a required property")


def schema_input_error(args: Mapping, errors: Sequence[str]) -> ParallelInputError:
    """Translate a tool-schema rejection into the typed refusal the model can act on."""
    subframes = args.get("subframes") if isinstance(args, Mapping) else None
    units = subframes if isinstance(subframes, list) else []
    for error in errors:
        match = _REQUIRED_PROPERTY.search(str(error))
        if match is None:
            continue
        field = match.group(1)
        unit_id = "all"
        for index, unit in enumerate(units):
            if isinstance(unit, Mapping) and field not in unit:
                identity = unit.get("id")
                unit_id = identity if isinstance(identity, str) and identity.strip() else str(index)
                break
        return ParallelInputError(unit_id, field, "is required by the parallel schema",
                                  f'add "{field}": ... and resend the whole parallel call')
    detail = "; ".join(str(error) for error in errors[:3]) or "does not match the parallel schema"
    return ParallelInputError("all", "subframes", detail[:300],
                              "fix the subframe fields to match the parallel schema and resend the whole call")


class RecoveryRequired(BaseException):
    """A durable decision needs replay before the parent can take another action."""


TINY_PLAN_OWNED_BYTES = 4096
TINY_PLAN_ADVISORY = "This subframe plan is tiny; serial work may be cheaper."
_HOST_ACK_SCHEMA = "stateframe-parallel-ack/1"


@lru_cache(maxsize=8)
def _host_runtime_id(pid: int) -> str:
    try:
        identity = ProcessTable().identity(pid)
        boot_id = _boot_id()
    except (OSError, RuntimeError, ValueError) as error:
        raise ParallelError("subframe host runtime identity is unavailable") from error
    if identity is None or identity.pid != pid:
        raise ParallelError("subframe host runtime identity is unavailable")
    return hashlib.sha256(canonical_json({"pid": pid, "start": list(identity.start),
                                          "boot_id": boot_id})).hexdigest()


def _tiny_plan(request: ParallelRequest, snapshot: DirectorySnapshot | _StartSnapshot) -> bool:
    """Recognize one-file, one-check plans solely for a cost advisory."""
    return bool(request.units) and all(
        len(unit.owns) == 1 and not unit.creates and not unit.reads
        and len(unit.check.source_paths) == 1 and len(unit.check.selectors) == 1
        and len(snapshot.read_bytes(unit.owns[0])) <= TINY_PLAN_OWNED_BYTES
        for unit in request.units)


_SIGNATURE = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["function", "async_function", "class", "method", "constant"]},
    "params": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "kind": {"type": "string",
        "enum": ["posonly", "pos", "varpos", "kwonly", "varkw"]},
        "has_default": {"type": "boolean"}, "annotation": {"type": ["string", "null"]}},
        "required": ["name", "kind", "has_default", "annotation"]}},
    "returns": {"type": ["string", "null"]},
    "decorators": {"type": "array", "items": {"type": "string"}}},
    "required": ["kind", "params", "returns", "decorators"]}
_UNIT = {"type": "object", "properties": {
    "id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,23}$"},
    "goal": {"type": "string", "maxLength": 300},
    "outcome": {"type": "string", "maxLength": 300},
    "owns": {"type": "array", "description":
             "Existing files this subframe may edit. Every file the subframe changes must be listed here or in creates.",
             "items": {"type": "string"}},
    "creates": {"type": "array", "description":
                "NEW files this subframe adds. Only paths that do not exist yet; an existing file goes in owns, never creates.",
                "items": {"type": "string"}},
    "exports": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "path": {"type": "string"},
        "symbol": {"type": "string"}, "signature": _SIGNATURE},
        "required": ["id", "path", "symbol"]}},
    "consumes": {"type": "array", "items": {"type": "object", "description":
        "Use export_id for another subframe's export, path and symbol for an existing definition, "
        "or all three when they identify the same export.", "properties": {
        "export_id": {"type": "string"}, "path": {"type": "string"},
        "symbol": {"type": "string"}}}},
    "reads": {"type": "array", "description": "Existing supporting source files this subframe may read; omit owned files.",
              "items": {"type": "string"}},
    "requirements": {"type": "array", "maxItems": 8,
                     "description": "Acceptance conditions (up to 300 characters each), or exact parent requirement IDs.",
                     "items": {"type": "string", "maxLength": 300}},
    "check": {"description":
              "the test selector that proves this subframe (e.g. tests/adapters/test_csv.py).",
              "oneOf": [
                  {"type": "string"},
                  {"type": "array", "items": {"type": "string"}},
                  {"type": "object", "properties": {"tests": {
                      "type": "array", "items": {"type": "string"}}}, "required": ["tests"]},
              ]},
    "model_class": {"type": "string"}},
    "required": ["id", "goal", "owns"]}
PARALLEL_TOOL = {"type": "function", "function": {
    "name": "parallel",
    "description": "For each subframe send {id, goal, owns}; optional consumes (symbol or path) "
                    "and check. The host fills checks, reads, exports, dependencies, requirements "
                    "and limits. Full contracts with outcome and creates remain valid. "
                    "Use when disjoint files each need a read, edit and check cycle.\n"
                    "Use parallel as soon as you know the task has 2+ independent parts: each part edits its own files and has its own test selector.\n"
                    "Each subframe starts with its owned files and its test already loaded, so do not read those files yourself before dispatching.\n"
                    "Typical flow: read the task and the test layout, then call parallel. Edit serially only when parts share a file or depend on each other.\n"
                    "\n"
                    "Valid:   {\"id\":\"csv\",\"goal\":\"Update the CSV adapters\",\"owns\":[\"src/adapters4/csv_reader.py\",\"src/adapters4/csv_writer.py\"],\"creates\":[],\"check\":\"tests/adapters/test_csv.py\"}\n"
                    "Invalid: {\"id\":\"csv\",\"creates\":[\"src/adapters4/csv_reader.py\"]}  (file already exists: put it in owns)\n"
                    "Subframes must not share an owned or created path.",
    "parameters": {"type": "object", "properties": {"subframes": {"type": "array", "items": _UNIT}},
                    "required": ["subframes"]}}}


class HostClientProtocol(Protocol):
    def submit(self, op: str, scope: str, payload: dict) -> str: ...
    def poll(self, result_id: str) -> HostAnswer: ...
    def cancel(self, scope: str) -> None: ...


class CompositionCheckFailed(ParallelError):
    """A composition check ran and did not pass; the unit outcome keeps that status, not NOT_RUN."""

    def __init__(self, check_status: str, message: str) -> None:
        super().__init__(message)
        self.check_status = check_status


_FINAL_CHECK_RECEIPT = "final-check-receipt.json"


def _atomic_json(path: Path, value: object, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json(value)
    if path.is_symlink():
        raise ParallelError(f"subframe artifact path is a symlink: {path}")
    if path.exists() and not replace:
        if path.read_bytes() != data:
            raise ParallelError(f"immutable subframe artifact conflicts: {path}")
        return
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".parallel-")
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if replace:
            os.replace(name, path)
        else:
            try:
                os.link(name, path)
            except FileExistsError:
                if path.is_symlink() or path.read_bytes() != data:
                    raise ParallelError(f"immutable subframe artifact conflicts: {path}") from None
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class HostClient:
    """One parent attempt's immutable file channel to the host supervisor."""

    def __init__(self, call_root: Path, attempt: int):
        self.call_root = Path(call_root)
        self.attempt = attempt
        self.seq = 0
        self.channel = self.call_root / f"attempt-{attempt}"
        (self.channel / "rq").mkdir(parents=True, exist_ok=True)
        (self.channel / "an").mkdir(parents=True, exist_ok=True)
        self.runtime_id = _host_runtime_id(os.getppid())

    def submit(self, op: str, scope: str, payload: dict) -> str:
        self.seq += 1
        request = HostRequest(self.attempt, self.call_root.name, self.seq, op, scope, payload,
                              hashlib.sha256(canonical_json(payload)).hexdigest())
        path = self.channel / "rq" / f"{self.seq:03d}.json"
        _atomic_json(path, request.to_json())
        return str(self.channel / "an" / f"{self.seq:03d}.json")

    def poll(self, result_id: str) -> HostAnswer:
        path = Path(result_id)
        if (not path.is_relative_to(self.call_root) or path.is_symlink() or
                not path.resolve(strict=False).is_relative_to(self.call_root.resolve(strict=True))):
            raise ParallelError("subframe host answer path escaped its call")
        if not path.is_file():
            return HostAnswer(self.attempt, self.call_root.name, 0, "pending", "PENDING", result_id, {})
        raw = path.read_bytes()
        data = json.loads(raw)
        if path.name.endswith(".result.json"):
            if set(data) != {"status", "payload"} or data["status"] not in {"OK", "ERROR"}:
                raise ParallelError("subframe seal result is invalid")
            if not self._has_matching_ack(path, raw):
                return HostAnswer(self.attempt, self.call_root.name, 0, "pending", "PENDING", result_id, {})
            return HostAnswer(self.attempt, self.call_root.name, 0, "seal", data["status"], None, data["payload"])
        answer = HostAnswer.from_json(data)
        if answer.attempt != self.attempt or answer.call_id != self.call_root.name:
            raise ParallelError("subframe host answer has a foreign identity")
        if not self._has_matching_ack(path, raw):
            return HostAnswer(self.attempt, self.call_root.name, 0, "pending", "PENDING", result_id, {})
        return answer

    @staticmethod
    def _ack_path(path: Path) -> Path:
        return path.with_name(f"{path.stem}.ack.json")

    def _has_matching_ack(self, path: Path, raw: bytes) -> bool:
        """Only a supervisor ACK for these bytes and this live attempt runtime is ready."""
        runtime_id = getattr(self, "runtime_id", None)
        if not isinstance(runtime_id, str) or not runtime_id:
            return False
        ack_path = self._ack_path(path)
        try:
            if ack_path.is_symlink() or not ack_path.is_file():
                return False
            ack_bytes = ack_path.read_bytes()
            if len(ack_bytes) > 4096:
                return False
            evidence = json.loads(ack_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        return (
            isinstance(evidence, dict)
            and set(evidence) == {"schema", "attempt", "call_id", "runtime", "sha256"}
            and evidence.get("schema") == _HOST_ACK_SCHEMA
            and type(evidence.get("attempt")) is int
            and evidence.get("attempt") == self.attempt
            and evidence.get("call_id") == self.call_root.name
            and evidence.get("runtime") == runtime_id
            and evidence.get("sha256") == hashlib.sha256(raw).hexdigest()
        )

    def cancel(self, scope: str) -> None:
        self.submit("cancel", scope, {"unit_id": scope, "reason": "parent subframe call cancelled"})


def _host_answer(host: HostClientProtocol, op: str, scope: str, payload: dict,
                 deadline: float | None) -> HostAnswer:
    response_seconds = load_tolerant_seconds("STATEFRAME_PARALLEL_HOST_RESPONSE_SECONDS")
    response_deadline = time.monotonic() + response_seconds
    ref = host.submit(op, scope, payload)
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise ParallelError("subframe parent deadline exceeded")
        if time.monotonic() >= response_deadline:
            raise ParallelError(f"subframe host response for {op} {scope} timed out after "
                                f"{response_seconds:g} seconds; inspect the host supervisor, "
                                "or raise STATEFRAME_PARALLEL_HOST_RESPONSE_SECONDS")
        answer = host.poll(ref)
        if answer.status == "ERROR":
            raise ParallelError(f"subframe host {op} failed: {answer.payload.get('error', 'unknown error')}")
        if answer.status == "OK":
            return answer
        if answer.status == "PENDING" and answer.result_id and op == "seal":
            ref = answer.result_id
        elif answer.status == "PENDING" and answer.op == op and op == "launch":
            return answer
        time.sleep(0.2)


def _planner_id(text: str) -> str:
    return "planner:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _planner_clauses(args: Mapping, requirements: tuple[RequirementClause, ...]) -> tuple[RequirementClause, ...]:
    known = {clause.id for clause in requirements}
    clauses = {clause.id: clause for clause in requirements}
    for item in args["subframes"]:
        for text in item.get("requirements", []):
            if text in known:
                continue
            identifier = _planner_id(text)
            previous = clauses.get(identifier)
            if previous is not None and previous.text != text:
                raise ParallelInputError(item["id"], "requirements", "planner ID collision",
                                         "use a different acceptance sentence")
            clauses[identifier] = RequirementClause(identifier, text, "planner", ())
    return tuple(clauses[key] for key in sorted(clauses))


def _source_symbols(snapshot: DirectorySnapshot, path: str) -> tuple[str, ...]:
    if not path.endswith(".py") or not _safe_path(path, snapshot.root, create=False):
        return ()
    try:
        source = snapshot.read_bytes(path)
        if len(source) > 24576:
            return ()
        symbols = _definitions(path, source)
    except OSError:
        return ()
    return tuple(sorted(symbols or ()))


def _existing_export_match(snapshot: DirectorySnapshot, export_id: str) -> tuple[str, str] | None:
    """Resolve an export-style name only when file and definition are both unique."""
    stem = export_id.removesuffix("_export")
    if not stem or stem == export_id:
        return None
    paths = list(itertools.islice(snapshot.list_files(), 2001))
    if len(paths) > 2000:
        return None
    matches = [(path, symbols[0]) for path in paths if Path(path).stem == stem
               if len(symbols := _source_symbols(snapshot, path)) == 1]
    if len(matches) != 1:
        return None
    return matches[0]


def _coupling_symbols(hit, producer: dict, consumer: dict, snapshot: DirectorySnapshot,
                      modules: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Resolve a scan hit to the existing public definitions the consumer needs."""
    paths = {path for field in ("owns", "creates") for path in producer.get(field, ())
             if isinstance(path, str) and snapshot.exists(path)}
    evidence = hit.evidence[0]
    references: set[tuple[str, str]] = set()
    for importer_path in (path for field in ("owns", "creates")
                          for path in consumer.get(field, ()) if snapshot.exists(path)):
        importer = snapshot.read_bytes(importer_path)
        if not importer_path.endswith(".py"):
            continue
        try:
            tree = ast.parse(importer.decode("utf-8"))
        except (SyntaxError, UnicodeError) as error:
            raise ParallelInputError(hit.dst_unit, "consumes", f"cannot parse {importer_path}: {error}",
                                     "declare the producer export and consume explicitly") from error
        local = _definitions(importer_path, importer) or set()
        imported = {alias.asname or alias.name.split(".")[0] for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        loaded = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
                  and isinstance(node.ctx, ast.Load)} - local - imported
        for name in loaded:
            owners = [path for path in paths
                      if name in (_definitions(path, snapshot.read_bytes(path)) or set())]
            if len(owners) > 1:
                raise ParallelInputError(hit.dst_unit, "consumes",
                                         f"ambiguous definition {name} in {', '.join(sorted(owners))}",
                                         "declare an exact producer export and consume")
            references.update((path, name) for path in owners)
        current = next((name for name, path in modules.items() if path == importer_path), "")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base_parts = (current if importer_path.endswith("/__init__.py")
                              else current.rpartition(".")[0]).split(".")
                base = ".".join(base_parts[:len(base_parts) - node.level + 1]) if node.level else ""
                module = ".".join(part for part in (base, node.module or "") if part)
                path = modules.get(module)
                if path in paths:
                    for alias in node.names:
                        names = ((_definitions(path, snapshot.read_bytes(path)) or set())
                                 if alias.name == "*" else (alias.name,))
                        references.update((path, name) for name in names if not name.startswith("_"))
                else:
                    for alias in node.names:
                        imported_path = modules.get(f"{module}.{alias.name}")
                        if imported_path in paths:
                            names = _definitions(imported_path, snapshot.read_bytes(imported_path)) or set()
                            references.update((imported_path, name) for name in names
                                              if not name.startswith("_"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for module, path in modules.items():
                        if path in paths and (module == alias.name or module.startswith(alias.name + ".")):
                            names = _definitions(path, snapshot.read_bytes(path)) or set()
                            references.update((path, name) for name in names if not name.startswith("_"))
    if not references:
        raise ParallelInputError(hit.dst_unit, "consumes",
                                 f"cannot project imported interface from {hit.src_unit} at "
                                 f"{evidence.path}:{evidence.line}",
                                 "declare the producer export and exact consume, or combine the subframes")
    return tuple(sorted(references))


def _test_imports(snapshot: DirectorySnapshot, modules: dict[str, str],
                  limits: ParallelLimits) -> dict[str, dict[str, set[str]]]:
    """Index direct imports in bounded test source without executing repository code."""
    result: dict[str, dict[str, set[str]]] = {}
    paths = [path for path in snapshot.list_files()
             if path.startswith("tests/") and path.endswith(".py")
             and Path(path).name.startswith("test_")]
    if len(paths) > limits.scan_max_files:
        raise ParallelInputError("all", "check", "test scan exceeds file limit",
                                 "provide check.tests and exports explicitly for each subframe")
    scanned = 0
    for path in paths:
        if not _safe_path(path, snapshot.root, create=False):
            continue
        data = snapshot.read_bytes(path)
        scanned += len(data)
        if scanned > limits.scan_max_bytes:
            raise ParallelInputError("all", "check", "test scan exceeds byte limit",
                                     "provide check.tests and exports explicitly for each subframe")
        try:
            tree = ast.parse(data.decode("utf-8"), filename=path)
        except (SyntaxError, UnicodeError) as error:
            raise ParallelInputError("all", "check", f"cannot scan {path}: {error}",
                                     "provide check.tests and exports explicitly for each subframe") from error
        imports: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                owner = modules.get(node.module or "")
                if owner:
                    names = {alias.name for alias in node.names}
                    if "*" in names:
                        definitions = _definitions(owner, snapshot.read_bytes(owner)) or set()
                        names = {name for name in definitions if not name.startswith("_")}
                    imports.setdefault(owner, set()).update(names)
                for alias in node.names:
                    child = modules.get(f"{node.module}.{alias.name}")
                    if child:
                        definitions = _definitions(child, snapshot.read_bytes(child)) or set()
                        imports.setdefault(child, set()).update(
                            name for name in definitions if not name.startswith("_"))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    owner = modules.get(alias.name)
                    if owner:
                        definitions = _definitions(owner, snapshot.read_bytes(owner)) or set()
                        imports.setdefault(owner, set()).update(
                            name for name in definitions if not name.startswith("_"))
        result[path] = imports
    return result


def _expand_terse(units: list, snapshot: DirectorySnapshot, limits: ParallelLimits,
                  record: Callable[..., None]) -> None:
    if not any(isinstance(unit, dict) and "goal" in unit for unit in units):
        return
    needs_test_scan = any(isinstance(unit, dict) and "goal" in unit
                          and ("check" not in unit or "exports" not in unit)
                          for unit in units)
    tests = _test_imports(snapshot, _module_map(snapshot), limits) if needs_test_scan else {}
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or "goal" not in unit:
            continue
        unit_id = unit.get("id") if isinstance(unit.get("id"), str) else str(index)
        goal = unit["goal"]
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 300:
            raise ParallelInputError(unit_id, "goal", "expected nonempty text within 300 characters",
                                     'send {"id": "...", "goal": "...", "owns": ["..."]}')
        if "outcome" not in unit:
            unit["outcome"] = goal
            record(unit_id, "outcome", "expanded_field", "GOAL_SENTENCE", source="goal")
        if "requirements" not in unit:
            unit["requirements"] = [goal]
            record(unit_id, "requirements", "expanded_field", "GOAL_PLANNER_CLAUSE", source="goal")
        if "creates" not in unit:
            unit["creates"] = []
            record(unit_id, "creates", "expanded_field", "OPTIONAL_FIELD_DEFAULTED", source="default")
        owned = unit.get("owns")
        if not isinstance(owned, list) or any(not isinstance(path, str) for path in owned):
            continue  # The ordinary parser reports the exact owns error.
        if "check" not in unit:
            selected: set[str] = set()
            imported = False
            for owned_path in owned:
                candidates = sorted(path for path, imports in tests.items() if owned_path in imports)
                if len(candidates) > 1:
                    fragment = json.dumps({"check": {"tests": [candidates[0]]}})
                    raise ParallelInputError(unit_id, "check",
                                             f"ambiguous checks for {owned_path}: {', '.join(candidates)}",
                                             f"resend this subframe with {fragment}")
                if candidates:
                    selected.add(candidates[0])
                    imported = True
                else:
                    fallback = f"tests/test_{Path(owned_path).stem}.py"
                    if fallback in tests:
                        selected.add(fallback)
            unit["check"] = {"tests": sorted(selected)}
            record(unit_id, "check", "expanded_field",
                   "STATIC_TEST_IMPORT" if imported else
                   "MATCHING_TEST_NAME" if selected else "NO_MATCHING_CHECK", source="tests")
        if "reads" not in unit:
            check = unit.get("check")
            selectors = check.get("tests") if isinstance(check, dict) else None
            sources = ({test.split("::", 1)[0] for test in selectors if isinstance(test, str)}
                       if isinstance(selectors, list) else set())
            unit["reads"] = sorted(sources - set(owned))
            record(unit_id, "reads", "expanded_field", "CHECK_SOURCE_PATHS", source="check")
        if "exports" not in unit:
            symbols: set[tuple[str, str]] = set()
            check = unit.get("check")
            selectors = check.get("tests") if isinstance(check, dict) else None
            for selector in selectors if isinstance(selectors, list) else ():
                if not isinstance(selector, str):
                    continue
                imports = tests.get(selector.split("::", 1)[0], {})
                for owned_path in owned:
                    for symbol in imports.get(owned_path, ()):
                        if symbol != "*" and not symbol.startswith("_"):
                            symbols.add((owned_path, symbol))
            unit["exports"] = [{"id": "auto_" + hashlib.sha256(
                f"{unit_id}:{path}:{symbol}".encode()).hexdigest()[:16],
                "path": path, "symbol": symbol} for path, symbol in sorted(symbols)]
            record(unit_id, "exports", "expanded_field", "TEST_IMPORTED_SYMBOLS", source="tests")


def _normalize_check(value: object, unit_id: str) -> dict[str, list[str]]:
    """Normalize the public check shorthands before completion or direct parsing."""
    if isinstance(value, str):
        tests = [value]
    elif isinstance(value, list):
        tests = value
    elif isinstance(value, dict) and set(value) == {"tests"}:
        tests = value["tests"]
    else:
        raise ParallelInputError(unit_id, "check.tests", "expected a selector, selector array, or check object with tests only",
                                 'send {"tests": []} for no check')
    if not isinstance(tests, list) or any(not isinstance(test, str) for test in tests):
        raise ParallelInputError(unit_id, "check.tests", "expected an array of strings",
                                 "send check.tests as a string array")
    if any(not test.strip() for test in tests):
        raise ParallelInputError(unit_id, "check.tests", "empty selector",
                                 "provide pytest file paths or node IDs")
    return {"tests": tests}


def _complete_arguments(args: Mapping, snapshot: DirectorySnapshot,
                        limits: ParallelLimits | None = None) -> tuple[dict, tuple[dict, ...]]:
    """Complete only references determined by the current plan and admitted source tree."""
    if not isinstance(args, Mapping) or not isinstance(args.get("subframes"), list):
        return dict(args) if isinstance(args, Mapping) else args, ()
    completed = copy.deepcopy(dict(args))
    units = completed["subframes"]
    actions: list[dict] = []

    for index, unit in enumerate(units):
        if isinstance(unit, dict) and "check" in unit:
            unit_id = unit.get("id") if isinstance(unit.get("id"), str) else str(index)
            unit["check"] = _normalize_check(unit["check"], unit_id)

    def record(unit_id: str, field: str, action: str, reason: str,
               producer: str | None = None, source: str | None = None) -> None:
        event = {"subframe": unit_id, "field": field, "action": action,
                 "producer": producer, "reason_code": reason}
        if action == "expanded_field":
            event["source"] = source
        actions.append(event)

    def declared_paths(unit: dict) -> tuple[str, ...]:
        return tuple(path for field in ("owns", "creates")
                     for path in (unit.get(field) if isinstance(unit.get(field), list) else ())
                     if isinstance(path, str))

    _expand_terse(units, snapshot, limits or ParallelLimits(), record)

    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            continue
        unit_id = unit.get("id") if isinstance(unit.get("id"), str) else str(index)
        requirements = unit.get("requirements")
        if ("goal" not in unit and "outcome" not in unit
                and isinstance(requirements, list) and requirements
                and all(isinstance(text, str) and text.strip() for text in requirements)):
            unit["outcome"] = requirements[0]
            record(unit_id, "outcome", "expanded_field", "REQUIREMENT_SENTENCE",
                   source="requirements[0]")
        for field in ("creates", "consumes", "reads", "exports", "requirements", "check",
                      "model_class"):
            if field not in unit:
                unit[field] = ({"tests": []} if field == "check" else
                               "default" if field == "model_class" else [])
                if "goal" in unit or field == "creates":
                    record(unit_id, field, "expanded_field", "OPTIONAL_FIELD_DEFAULTED", source="default")
                else:
                    record(unit_id, field, "defaulted_optional", "OPTIONAL_FIELD_DEFAULTED")
        for field in ("owns", "creates", "reads"):
            values = unit.get(field)
            if not isinstance(values, list):
                continue
            unique = list(dict.fromkeys(values)) if all(isinstance(value, str) for value in values) else values
            if len(unique) != len(values):
                unit[field] = unique
                record(unit_id, field, "deduplicated_path", "DUPLICATE_PATH")
        if isinstance(unit.get("reads"), list) and all(isinstance(path, str) for path in unit["reads"]):
            owned = set(declared_paths(unit))
            reads = [path for path in unit["reads"] if path not in owned]
            if len(reads) != len(unit["reads"]):
                unit["reads"] = reads
                record(unit_id, "reads", "deduplicated_path", "REDUNDANT_OWNED_READ")
        check = unit.get("check")
        if isinstance(check, dict) and isinstance(check.get("tests"), list) and all(
                isinstance(value, str) for value in check["tests"]):
            tests = list(dict.fromkeys(check["tests"]))
            if len(tests) != len(check["tests"]):
                check["tests"] = tests
                record(unit_id, "check.tests", "deduplicated_path", "DUPLICATE_PATH")
        if isinstance(unit.get("exports"), list):
            for number, export in enumerate(unit["exports"]):
                if isinstance(export, dict) and "signature" not in export:
                    record(unit_id, f"exports[{number}].signature", "inferred_signature",
                           "SIGNATURE_INFERRED" if isinstance(export.get("path"), str)
                           and _safe_path(export["path"], snapshot.root, create=False)
                           else "SIGNATURE_UNVERIFIED")
        if isinstance(unit.get("requirements"), list):
            for number, requirement in enumerate(unit["requirements"]):
                if isinstance(requirement, str) and not requirement.startswith("req-"):
                    record(unit_id, f"requirements[{number}]", "planner_clause", "SENTENCE_REQUIREMENT")

    declared: list[tuple[dict, dict]] = []
    for unit in units:
        if isinstance(unit, dict) and isinstance(unit.get("exports"), list):
            declared.extend((unit, export) for export in unit["exports"] if isinstance(export, dict))

    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or not isinstance(unit.get("consumes"), list):
            continue
        unit_id = unit.get("id") if isinstance(unit.get("id"), str) else str(index)
        for number, consume in enumerate(unit["consumes"]):
            if not isinstance(consume, dict) or not consume or any(
                    key not in {"export_id", "path", "symbol"} or not isinstance(value, str) or not value
                    for key, value in consume.items()):
                continue
            field = f"consumes[{number}]"
            candidates = [(producer, export) for producer, export in declared
                          if producer is not unit and
                          all(export.get(key if key != "export_id" else "id") == value
                              for key, value in consume.items())]
            if not candidates and "export_id" not in consume:
                for producer in units:
                    if not isinstance(producer, dict) or producer is unit:
                        continue
                    for path in declared_paths(producer):
                        if "path" in consume and consume["path"] != path:
                            continue
                        if "path" in consume and "symbol" in consume:
                            candidates.append((producer, {"path": path, "symbol": consume["symbol"]}))
                            continue
                        symbols = _source_symbols(snapshot, path)
                        for symbol in symbols:
                            if "symbol" not in consume or consume["symbol"] == symbol:
                                candidates.append((producer, {"path": path, "symbol": symbol}))
            if not candidates and "export_id" not in consume and len(consume) == 1:
                paths = ([consume["path"]] if "path" in consume else
                         list(itertools.islice(snapshot.list_files(), 2001)))
                owned_paths = {path for producer in units if isinstance(producer, dict)
                               for path in declared_paths(producer)}
                matches = [(path, symbol) for path in paths if path not in owned_paths
                           for symbol in _source_symbols(snapshot, path)
                           if "symbol" not in consume or consume["symbol"] == symbol]
                if len(paths) <= 2000 and len(matches) == 1:
                    path, symbol = matches[0]
                    unit["consumes"][number] = {"path": path, "symbol": symbol}
                    record(unit_id, field, "resolved_existing", "UNIQUE_EXISTING_DEFINITION")
                    continue
                if len(matches) > 1:
                    options = " or ".join(json.dumps({"path": path, "symbol": symbol})
                                          for path, symbol in matches[:8])
                    raise ParallelInputError(unit_id, field, "ambiguous existing definitions",
                                             f"choose one exact consume: {options}")
            if not candidates and set(consume) == {"export_id"}:
                existing = _existing_export_match(snapshot, consume["export_id"])
                if existing is not None:
                    path, symbol = existing
                    candidates = [(producer, {"path": path, "symbol": symbol}) for producer in units
                                  if isinstance(producer, dict) and producer is not unit
                                  and path in declared_paths(producer)]
                    if not candidates:
                        unit["consumes"][number] = {"path": path, "symbol": symbol}
                        record(unit_id, field, "resolved_existing", "UNIQUE_EXISTING_DEFINITION")
                        continue
            if not candidates and "export_id" in consume and any(
                    export.get("id") == consume["export_id"] for _, export in declared):
                # Admission owns the typed mismatch refusal for contradictory declarations.
                continue
            # An explicit path+symbol in an unowned file remains an existing-definition consume.
            if not candidates and set(consume) == {"path", "symbol"}:
                continue
            if not candidates:
                existing = (_existing_export_match(snapshot, consume["export_id"])
                            if "export_id" in consume else None)
                fragment = (json.dumps({"path": existing[0], "symbol": existing[1]})
                            if existing is not None else None)
                if fragment is None:
                    fragment = '{"consumes": []}'
                raise ParallelInputError(unit_id, field,
                                         f"no subframe export or existing definition matches {consume}",
                                         f"send {fragment} if it names an existing definition; "
                                         'otherwise send {"consumes": []} if no dependency exists, '
                                         "or declare the missing producer export in subframes")
            if len(candidates) > 1:
                options = sorted('{"export_id": "' + str(export.get("id")) + '", "path": "' +
                                 str(export.get("path")) + '", "symbol": "' + str(export.get("symbol")) + '"}'
                                 for _, export in candidates)
                raise ParallelInputError(unit_id, field,
                                         "ambiguous producers " + ", ".join(sorted({
                                             str(producer.get("id")) + "/" + str(export.get("id"))
                                             for producer, export in candidates})),
                                         "choose one exact consume: " + " or ".join(options))
            producer, export = candidates[0]
            if "id" not in export:
                basis = f'{producer.get("id")}:{export["path"]}:{export["symbol"]}'
                export = {**export, "id": "auto_" + hashlib.sha256(basis.encode()).hexdigest()[:16]}
                producer["exports"].append(export)
                declared.append((producer, export))
                record(str(producer.get("id")), "exports", "synthesized_export", "OWNED_SYMBOL_EXPORTED")
                if "goal" in producer:
                    record(str(producer.get("id")), "exports", "expanded_field",
                           "OWNED_SYMBOL_EXPORTED", source="consumes")
            normalized = {"export_id": export["id"], "path": export["path"], "symbol": export["symbol"]}
            if consume != normalized:
                unit["consumes"][number] = normalized
            record(unit_id, field, "linked_consume", "UNIQUE_PRODUCER", str(producer.get("id")))
            if "goal" in unit:
                record(unit_id, "consumes", "expanded_field", "UNIQUE_PRODUCER",
                       str(producer.get("id")), source="resolved_reference")
        unique_consumes = list(dict.fromkeys(json.dumps(value, sort_keys=True)
                                              for value in unit["consumes"]))
        if len(unique_consumes) != len(unit["consumes"]):
            unit["consumes"] = [json.loads(value) for value in unique_consumes]
            record(unit_id, "consumes", "deduplicated_consume", "DUPLICATE_CONSUME")

    # The scan runs before admission so its unique, projectable references can
    # become explicit contract edges. Ambiguous ownership remains a refusal.
    request = _request(completed, "coupling-completion", "1")
    hits, _ = scan_coupling(request.units, snapshot, limits or ParallelLimits())
    by_id = {unit["id"]: unit for unit in units}
    modules = _module_map(snapshot)
    for hit in hits:
        producer = by_id[hit.src_unit]
        consumer = by_id[hit.dst_unit]
        for path, symbol in _coupling_symbols(hit, producer, consumer, snapshot, modules):
            claimants = sorted(unit["id"] for unit in units if path in declared_paths(unit))
            if claimants != [hit.src_unit]:
                raise ParallelInputError(hit.dst_unit, "consumes",
                                         f"ambiguous owners of {path}: {', '.join(claimants)}",
                                         "fix owns so the imported file has one owner")
            projection = project_interface(snapshot, path, symbol, None)
            if not projection.line or projection.signature is None:
                raise ParallelInputError(hit.dst_unit, "consumes",
                                         f"cannot project {path}::{symbol} from {hit.src_unit}",
                                         "declare a projectable producer export and exact consume, or combine the subframes")
            matches = [export for export in producer["exports"]
                       if export.get("path") == path and export.get("symbol") == symbol]
            if len(matches) > 1:
                raise ParallelInputError(hit.dst_unit, "consumes",
                                         f"multiple exports for {path}::{symbol} from {hit.src_unit}",
                                         "keep one producer export")
            if matches:
                export = matches[0]
            else:
                basis = f'{hit.src_unit}:{path}:{symbol}'
                export = {"id": "auto_" + hashlib.sha256(basis.encode()).hexdigest()[:16],
                          "path": path, "symbol": symbol}
                producer["exports"].append(export)
                record(hit.src_unit, "exports", "synthesized_export", "OWNED_SYMBOL_EXPORTED")
                if "goal" in producer:
                    record(hit.src_unit, "exports", "expanded_field", "OWNED_SYMBOL_EXPORTED",
                           source="coupling_scan")
            consume = {"export_id": export["id"], "path": path, "symbol": symbol}
            if consume not in consumer["consumes"]:
                consumer["consumes"].append(consume)
                record(hit.dst_unit, "consumes", "linked_coupling", "UNIQUE_COUPLING_OWNER",
                       hit.src_unit)
                if "goal" in consumer:
                    record(hit.dst_unit, "consumes", "expanded_field", "UNIQUE_COUPLING_OWNER",
                           hit.src_unit, source="coupling_scan")
    if [unit.get("id") for unit in units if isinstance(unit, dict)] != sorted(
            unit.get("id") for unit in units if isinstance(unit, dict) and isinstance(unit.get("id"), str)):
        record("all", "subframes", "derived_dag", "ORDER_DERIVED")
    return completed, tuple(actions)


def _request(args: Mapping, call_id: str, mode: str, *,
             requirements: tuple[RequirementClause, ...] = ()) -> ParallelRequest:
    if not isinstance(args, Mapping) or set(args) != {"subframes"} or not isinstance(args["subframes"], list):
        raise ParallelInputError("all", "subframes", "expected a subframes array with no other top-level fields",
                                 "send {\"subframes\": [...]} with two or more subframes")
    known = {clause.id for clause in requirements}
    contracts = []

    def strings(value: object, unit_id: str, field: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(part, str) for part in value):
            hint = ('add "owns": ["relative/path.py"]'
                    if field == "owns" and value is None else f"send {field} as a string array")
            raise ParallelInputError(unit_id, field, "expected an array of strings",
                                     hint)
        return tuple(value)

    allowed = {"id", "goal", "outcome", "owns", "creates", "exports", "consumes", "reads",
               "requirements", "check", "model_class"}
    for index, item in enumerate(args["subframes"]):
        if not isinstance(item, dict):
            raise ParallelInputError(str(index), f"subframes[{index}]", "expected an object",
                                     "provide id, outcome, owns and creates")
        unit_id = item.get("id") if isinstance(item.get("id"), str) else str(index)
        extra = set(item) - allowed
        if extra:
            field = sorted(extra)[0]
            raise ParallelInputError(unit_id, field, "unknown field", "remove the field")
        if not isinstance(item.get("id"), str) or not item["id"].strip():
            hint = f'add "id": "subframe_{index}"' if "id" not in item else "provide id as text"
            raise ParallelInputError(unit_id, "id", "expected nonempty text", hint)
        # goal is expanded into outcome before this check, so a missing outcome here means neither was sent.
        if "goal" not in item and "outcome" not in item:
            raise ParallelInputError(unit_id, "goal", "expected nonempty text",
                                     'add "goal": "<one sentence>"')
        if not isinstance(item.get("outcome"), str) or not item["outcome"].strip():
            raise ParallelInputError(unit_id, "outcome", "expected nonempty text", "provide outcome as text")
        owns = strings(item.get("owns"), unit_id, "owns")
        creates = strings(item.get("creates"), unit_id, "creates")
        reads = strings(item.get("reads", []), unit_id, "reads")
        requirement_text = strings(item.get("requirements", []), unit_id, "requirements")
        if len(requirement_text) > 8:
            raise ParallelInputError(unit_id, "requirements", "more than 8 acceptance conditions",
                                     "keep at most 8")
        for number, text in enumerate(requirement_text):
            if not text.strip() or len(text) > 300:
                raise ParallelInputError(unit_id, f"requirements[{number}]", "empty or over 300 characters",
                                         "write a nonempty condition within 300 characters")
        exports_raw = item.get("exports", [])
        consumes_raw = item.get("consumes", [])
        if not isinstance(exports_raw, list):
            raise ParallelInputError(unit_id, "exports", "expected an array", "use [] if no exports are needed")
        if not isinstance(consumes_raw, list):
            raise ParallelInputError(unit_id, "consumes", "expected an array", "use [] if no consumes are needed")
        exports = []
        for number, value in enumerate(exports_raw):
            field = f"exports[{number}]"
            if not isinstance(value, dict):
                raise ParallelInputError(unit_id, field, "expected an object", "provide id, path and symbol")
            extra = set(value) - {"id", "path", "symbol", "signature"}
            if extra:
                raise ParallelInputError(unit_id, f"{field}.{sorted(extra)[0]}", "unknown field", "remove the field")
            for key in ("id", "path", "symbol"):
                if not isinstance(value.get(key), str) or not value[key]:
                    raise ParallelInputError(unit_id, f"{field}.{key}", "expected nonempty text", f"provide {key}")
            try:
                signature = (CanonicalSignature.from_json(value["signature"])
                             if value.get("signature") is not None else None)
            except ValueError as error:
                raise ParallelInputError(unit_id, f"{field}.signature", str(error),
                                         "provide a canonical signature object") from error
            exports.append(ExportSpec(value["id"], value["path"], value["symbol"], signature))
        consumes = []
        for number, value in enumerate(consumes_raw):
            field = f"consumes[{number}]"
            if not isinstance(value, dict):
                raise ParallelInputError(unit_id, field, "expected an object", "provide export_id or path and symbol")
            extra = set(value) - {"export_id", "path", "symbol"}
            if extra:
                raise ParallelInputError(unit_id, f"{field}.{sorted(extra)[0]}", "unknown field", "remove the field")
            if any(not isinstance(part, str) or not part for part in value.values()):
                raise ParallelInputError(unit_id, field, "expected nonempty text values",
                                         "provide export_id or path and symbol")
            try:
                consumes.append(ConsumeSpec(value.get("export_id"), value.get("path"), value.get("symbol")))
            except ValueError as error:
                raise ParallelInputError(unit_id, field, str(error),
                                         "provide export_id or path and symbol") from error
        tests = tuple(_normalize_check(item.get("check", {"tests": []}), unit_id)["tests"])
        model_class = item.get("model_class", "default")
        if not isinstance(model_class, str) or not model_class:
            raise ParallelInputError(unit_id, "model_class", "expected nonempty text", "provide a model class")
        recipe = CheckRecipe(tuple(sorted({value.split("::", 1)[0] for value in tests})),
                             tests, "pytest" if tests else "none")
        ids = tuple(text if text in known else _planner_id(text) for text in requirement_text)
        contracts.append(UnitContract(item["id"], item["outcome"], owns, creates, tuple(exports),
                                      tuple(consumes), reads, ids, recipe, model_class))
    return ParallelRequest(call_id, tuple(contracts), mode)


def _bind_implicit_signatures(request: ParallelRequest,
                              snapshot: DirectorySnapshot) -> ParallelRequest:
    """Freeze existing Python interfaces from the admitted tree, not a later proposal."""
    units = []
    for unit in request.units:
        exports = []
        for export in unit.exports:
            signature = export.signature
            if signature is None and export.path.endswith(".py") and snapshot.exists(export.path):
                signature = project_interface(snapshot, export.path, export.symbol, export.id).signature
            exports.append(dataclasses.replace(export, signature=signature))
        units.append(dataclasses.replace(unit, exports=tuple(exports)))
    return dataclasses.replace(request, units=tuple(units))


@dataclasses.dataclass
class ParallelContext:
    runtime: object
    bolt: object
    root: Path
    created: set[str]
    parent_seed: str
    attempt: int
    requirements: tuple
    limits: ParallelLimits
    mode: str
    protected: Callable[[str], bool]
    deadline: float | None = None


_service: tuple[Path, dict] | None = None


def install(run: Path, spec: dict) -> None:
    global _service
    if _service is not None:
        raise ParallelError("subframe service is already installed")
    if spec.get("parallel_mode") == "off" or "unit_spec" in spec:
        raise ParallelError("subframe service cannot be installed in this run")
    _service = (Path(run), spec)


def available() -> bool:
    return _service is not None


PARALLEL_ACTION_SUMMARY_BYTES = 3072
# 8 KiB for result/header metadata plus 64 bytes per unit at the configured 1,024-unit maximum.
PARALLEL_DISPOSITION_HEADER_BYTES = 8 * 1024
PARALLEL_DISPOSITION_UNIT_BYTES = 64
MAX_PARALLEL_DISPOSITION_BYTES = (
    PARALLEL_DISPOSITION_HEADER_BYTES + PARALLEL_DISPOSITION_UNIT_BYTES * MAX_PARALLEL_UNITS
)
PARALLEL_DISPOSITION_SCHEMA = "stateframe-parallel-disposition/1"
_RECEIPT_PAGE_BYTES = 4096
_RECEIPT_HANDLE = re.compile(r"subframe:([A-Za-z0-9][A-Za-z0-9_-]{0,63}):([1-9][0-9]{0,4}):([0-9a-f]{64})\Z")
# The 16 MiB durable result can appear twice in a paged refusal (instruction and observation).
_MAX_RECEIPT_PAGES = 32768


def _receipt_handle_parts(handle: object) -> tuple[str, int, str] | None:
    if not isinstance(handle, str):
        return None
    match = _RECEIPT_HANDLE.fullmatch(handle)
    if match is None:
        return None
    number = int(match[2])
    if number > _MAX_RECEIPT_PAGES:
        return None
    return match[1], number, match[3]


def _open_call_dir(run: Path, call_id: str, *, create: bool) -> int:
    """Traverse every artifact ancestor relative to the trusted run descriptor."""
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in ("parallel", call_id):
            if create:
                try:
                    os.mkdir(component, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError:
        os.close(descriptor)
        raise


def _open_receipt_dir(run: Path, call_id: str, *, create: bool) -> int:
    descriptor = _open_call_dir(run, call_id, create=create)
    try:
        if create:
            try:
                os.mkdir("receipt", dir_fd=descriptor)
            except FileExistsError:
                pass
        child = os.open("receipt", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor)
        return child
    finally:
        os.close(descriptor)


def _read_receipt_file(descriptor: int, name: str, limit: int) -> bytes:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ParallelError("subframe receipt artifact is not a regular file")
        if info.st_size > limit:
            raise ParallelError("subframe receipt artifact exceeds the read limit")
        with os.fdopen(fd, "rb", closefd=False) as source:
            data = source.read(limit + 1)
        if len(data) > limit:
            raise ParallelError("subframe receipt artifact exceeds the read limit")
        return data
    finally:
        os.close(fd)


def _write_receipt_file(descriptor: int, name: str, data: bytes) -> bool:
    """Keep immutable cache entries; report a conflicting name to the caller."""
    try:
        existing = _read_receipt_file(descriptor, name, 16 << 20)
    except FileNotFoundError:
        existing = None
    except ParallelError:
        return False
    except OSError as error:
        if error.errno not in {errno.ELOOP, errno.EISDIR}:
            raise
        return False
    if existing is not None:
        return existing == data
    temporary = f".receipt-{secrets.token_hex(16)}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=descriptor)
    try:
        with os.fdopen(fd, "wb", closefd=False) as output:
            output.write(data)
            output.flush()
            os.fsync(fd)
        try:
            os.link(temporary, name, src_dir_fd=descriptor, dst_dir_fd=descriptor,
                    follow_symlinks=False)
        except FileExistsError:
            try:
                return _read_receipt_file(descriptor, name, 16 << 20) == data
            except ParallelError:
                return False
            except OSError as error:
                if error.errno not in {errno.ELOOP, errno.EISDIR}:
                    raise
                return False
        os.fsync(descriptor)
        return True
    finally:
        os.close(fd)
        os.unlink(temporary, dir_fd=descriptor)


def _write_receipt_alternate(descriptor: int, prefix: str, data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    for suffix in ("", *(f"-{number}" for number in range(1, 1025))):
        if _write_receipt_file(descriptor, f"{prefix}-{digest}{suffix}.json", data):
            return
    raise ParallelError("subframe receipt cache names are exhausted")


def receipt_handle_available(handle: object) -> bool:
    return _service is not None and _receipt_handle_parts(handle) is not None


def _receipt_page_payloads(result: ParallelResult, rows: list[dict[str, object]]) -> list[bytes]:
    """Derive the complete authenticated chain from the durable result."""
    pages: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for row in rows:
        candidate = [*current, row]
        candidate_size = len(canonical_json({
            "schema": "stateframe-subframe-receipt-page/1", "call_id": result.call_id,
            "page": len(pages) + 1, "rows": candidate, "next_sha256": "0" * 64,
        }))
        if current and candidate_size > _RECEIPT_PAGE_BYTES:
            pages.append(current)
            current = [row]
        else:
            current = candidate
    if current:
        pages.append(current)
    if not pages:
        pages = [[]]
    if len(pages) > _MAX_RECEIPT_PAGES:
        raise ParallelError("subframe receipt exceeds the page count limit")
    encoded_pages = [b""] * len(pages)
    next_digest: str | None = None
    for index in range(len(pages) - 1, -1, -1):
        number = index + 1
        payload = {"schema": "stateframe-subframe-receipt-page/1", "call_id": result.call_id,
                   "page": number, "rows": pages[index], "next_sha256": next_digest}
        encoded = canonical_json(payload)
        if len(encoded) > (16 << 20):
            raise ParallelError("subframe receipt page exceeds the artifact read limit")
        encoded_pages[index] = encoded
        next_digest = hashlib.sha256(encoded).hexdigest()
    return encoded_pages


def _receipt_pages(result: ParallelResult, rows: list[dict[str, object]]) -> str:
    """Derive the handle first; cache materialization has no authority over it."""
    if _service is None:
        raise ParallelError("subframe result service is unavailable")
    encoded_pages = _receipt_page_payloads(result, rows)
    digests = [hashlib.sha256(page).hexdigest() for page in encoded_pages]
    journal = canonical_json({
        "schema": "stateframe-subframe-receipt-journal/1", "call_id": result.call_id,
        "result_sha256": hashlib.sha256(result.observation.encode()).hexdigest(),
        "pages": digests,
    })
    handle = f"subframe:{result.call_id}:1:{digests[0]}"
    try:
        descriptor = _open_receipt_dir(_service[0], result.call_id, create=True)
        try:
            for number, encoded in enumerate(encoded_pages, 1):
                name = f"page-{number:04d}.json"
                if not _write_receipt_file(descriptor, name, encoded):
                    _write_receipt_alternate(descriptor, f"page-{number:04d}", encoded)
            if not _write_receipt_file(descriptor, "journal.json", journal):
                _write_receipt_alternate(descriptor, "journal", journal)
        finally:
            os.close(descriptor)
    except (OSError, ParallelError):
        # Pages are authenticated by the durable result, so an unsafe or unwritable
        # derived cache only disables caching. Never follow a conflicting cache path.
        pass
    return handle


def read_subframe_receipt(handle: object) -> str:
    """Read one authenticated receipt page through the existing read-only receipt tool."""
    if _service is None:
        raise ParallelError("subframe receipt handle is unavailable")
    parts = _receipt_handle_parts(handle)
    if parts is None:
        raise ParallelError("subframe receipt handle is invalid")
    call_id, number, expected = parts
    result, request = action_receipt_source(call_id)
    rows = _receipt_rows(result, request)
    encoded_pages = _receipt_page_payloads(result, rows)
    if number > len(encoded_pages):
        raise ParallelError("subframe receipt handle is invalid")
    encoded = encoded_pages[number - 1]
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ParallelError("subframe receipt artifact failed identity or digest validation")
    _receipt_pages(result, rows)
    page = json.loads(encoded)
    next_handle = (f"subframe:{call_id}:{number + 1}:{page['next_sha256']}"
                   if page["next_sha256"] is not None else None)
    return json.dumps({**page, "next_handle": next_handle}, sort_keys=True, separators=(",", ":"))


def action_receipt_source(call_id: str) -> tuple[ParallelResult, ParallelRequest | None]:
    """Read the durable structured result and expanded plan for an F4 action receipt."""
    if _service is None:
        raise ParallelError("subframe result service is unavailable")
    run, _ = _service
    safe_id = (call_id if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", call_id)
               else "call_" + hashlib.sha256(call_id.encode()).hexdigest()[:40])
    try:
        descriptor = _open_call_dir(run, safe_id, create=False)
        try:
            result_bytes = _read_receipt_file(descriptor, "result.json", 16 << 20)
            try:
                intent_bytes = _read_receipt_file(descriptor, "intent.json", 16 << 20)
            except FileNotFoundError:
                intent_bytes = None
        finally:
            os.close(descriptor)
        result = ParallelResult.from_json(json.loads(result_bytes))
        if intent_bytes is None:
            if (result.admission.status == "REFUSED"
                    and result.admission.refusal is not None
                    and result.admission.refusal.reason == "INVALID_ARGUMENTS"):
                if result.call_id != safe_id:
                    raise ParallelError("subframe result has a different call identity")
                return result, None
            raise ParallelError("subframe expanded plan is missing")
        intent = json.loads(intent_bytes)
        if not isinstance(intent, dict) or intent.get("call_id") != safe_id:
            raise ParallelError("subframe expanded plan has a different call identity")
        if intent.get("schema") == "stateframe-parallel-intent/1":
            request = ParallelRequest.from_json(intent["request"])
            admission = AdmissionResult.from_json(intent["admission"])
            if request.call_id != safe_id or admission != result.admission:
                raise ParallelError("subframe result differs from the expanded plan")
        elif intent.get("schema") == "stateframe-parallel-input-refusal/1":
            request = None
            if result.admission.status != "REFUSED" or result.admission.refusal is None:
                raise ParallelError("subframe input refusal has no refusal result")
        else:
            raise ParallelError("subframe expanded plan schema is unsupported")
    except FileNotFoundError as error:
        raise ParallelReceiptMissing("subframe result or expanded plan is missing") from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ParallelError("subframe receipt path has an unsafe ancestor") from error
        raise RecoveryRequired(f"RECOVERY_REQUIRED: subframe result storage failed: {error}") from error
    except (ValueError, KeyError, TypeError) as error:
        raise ParallelError(f"subframe result or expanded plan is invalid: {error}") from error
    if result.call_id != safe_id:
        raise ParallelError("subframe result has a different call identity")
    return result, request


def _receipt_rows(result: ParallelResult, request: ParallelRequest | None) -> list[dict[str, object]]:
    if result.admission.status == "REFUSED":
        refusal = result.admission.refusal
        if refusal is None:
            raise ParallelError("subframe refusal has no structured reason")
        instruction = "fix: " + _receipt_refusal_hint(refusal, request)
        chunks = [("instruction", instruction), ("observation", result.observation)]
        return [{"id": "refusal", "outcome": "REFUSED", "reason_code": refusal.reason,
                 "section": section, "part": offset // 2000 + 1, "text": content[offset:offset + 2000]}
                for section, content in chunks for offset in range(0, len(content), 2000)]
    if request is None or request.call_id != result.call_id:
        raise ParallelError("subframe result has no expanded plan")
    by_id = {item.unit_id: item for item in result.outcomes}
    if len(by_id) != len(result.outcomes) or set(by_id) != {unit.id for unit in request.units}:
        raise ParallelError("subframe result does not cover every planned subframe")
    failed = [unit for unit in request.units if by_id[unit.id].status not in
              {"COMMITTED", "COMMITTED_UNVERIFIED", "CANCELLED"}]
    cancelled = [unit for unit in request.units if by_id[unit.id].status == "CANCELLED"]
    unverified = [unit for unit in request.units if by_id[unit.id].status == "COMMITTED_UNVERIFIED"]
    committed = [unit for unit in request.units if by_id[unit.id].status == "COMMITTED"]
    rows = []
    for unit in (*failed, *cancelled, *unverified, *committed):
        item = by_id[unit.id]
        consumes = [consume.export_id or f"{consume.path}::{consume.symbol}"
                    for consume in unit.consumes]
        expanded = (f"Plan {unit.id}: owns={','.join(unit.owns) or '-'}; "
                    f"check={','.join(unit.check.selectors) or 'NOT_RUN'}; "
                    f"consumes={','.join(consumes) or '-'}; status={item.status}; "
                    f"check_status={item.check}")
        if item.reason or item.detail:
            expanded += f"; reason={_render_reason(item.reason or item.status, item.detail)}"
        reason = item.reason or "-"
        reason_code = reason if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason) else item.status
        rows.append({"id": unit.id, "outcome": item.status, "check_status": item.check,
                     "reason_code": reason_code, "detail": item.detail, "owns": list(unit.owns),
                     "check": list(unit.check.selectors), "consumes": consumes, "expanded": expanded})
    return rows


def _receipt_refusal_hint(refusal: Refusal, request: ParallelRequest | None) -> str:
    """Use structured refusal data, never delimiters in untrusted observation text."""
    if refusal.reason == "INVALID_ARGUMENTS":
        if refusal.hint is not None:
            return refusal.hint
        # Completed records from before structured hints keep the original detail.
        _, separator, hint = refusal.detail.rpartition("; fix: ")
        if separator:
            return hint.splitlines()[0]
    if request is not None:
        return _refusal_location(refusal, request)[2]
    return "resend a corrected subframe plan"


def action_receipt_summary(result: ParallelResult, request: ParallelRequest | None) -> tuple[str, int]:
    """Keep compact decisions on F4 and seal the full detail for paged retrieval."""

    def reason_code(item: UnitOutcome) -> str:
        reason = item.reason or "-"
        return reason if re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", reason) else item.status

    if result.admission.status == "REFUSED":
        summary = result.observation
        if len(summary.encode("utf-8")) <= PARALLEL_ACTION_SUMMARY_BYTES:
            return summary, 0
        refusal = result.admission.refusal
        if refusal is None:
            raise ParallelError("subframe refusal has no structured reason")
        instruction = _receipt_refusal_hint(refusal, request)
        if request is None:
            location = (refusal.detail if refusal.hint is not None else
                        refusal.detail.rpartition("; fix: ")[0] or refusal.detail)
        else:
            unit_id, field, _ = _refusal_location(refusal, request)
            location = f"subframe {unit_id}; field {field}"
        handle = _receipt_pages(result, _receipt_rows(result, request))
        if len(location.encode()) > 240:
            location = (location[:120] + "... sha256=" +
                        hashlib.sha256(location.encode()).hexdigest())
        reference = f'Details: read_check_receipt(handle="{handle}")'
        summary = (f"REFUSED {_subframe_reason(refusal.reason)}; {location}; fix: {instruction}\n"
                   f"{reference}")
        if len(summary.encode()) > PARALLEL_ACTION_SUMMARY_BYTES:
            summary = (f"REFUSED {_subframe_reason(refusal.reason)}; {location}; "
                       f"fix: read page 1 for the exact resend instruction; "
                       f"follow next_handle if it spans pages\n{reference}")
        return summary, 0
    if request is None or request.call_id != result.call_id:
        raise ParallelError("subframe result has no expanded plan")
    by_id = {item.unit_id: item for item in result.outcomes}
    if len(by_id) != len(result.outcomes) or set(by_id) != {unit.id for unit in request.units}:
        raise ParallelError("subframe result does not cover every planned subframe")
    failed = [unit for unit in request.units if by_id[unit.id].status not in
              {"COMMITTED", "COMMITTED_UNVERIFIED", "CANCELLED"}]
    cancelled = [unit for unit in request.units if by_id[unit.id].status == "CANCELLED"]
    unverified = [unit for unit in request.units if by_id[unit.id].status == "COMMITTED_UNVERIFIED"]
    committed = [unit for unit in request.units if by_id[unit.id].status == "COMMITTED"]
    status = ("all committed" if not failed and not cancelled else
              f"{len(failed)} failed; {len(cancelled)} cancelled")
    if unverified:
        status += f"; {len(unverified)} unverified"
    invalid = [item.unit_id for item in result.outcomes if item.check == "INVALID"]
    if invalid:
        status += f"; {len(invalid)} check INVALID (not a pass)"
    header = (f"Subframe results: {status}; final_check={result.final_check}; "
              f"integration={result.integration}")
    if not any(item.files for item in result.outcomes):
        header += "\nNo files changed by any subframe."

    def row(unit: UnitContract) -> str:
        item = by_id[unit.id]
        consumes = ",".join(consume.export_id or f"{consume.path}::{consume.symbol}"
                            for consume in unit.consumes) or "-"
        line = (f"Plan {unit.id}: owns={','.join(unit.owns) or '-'}; "
                 f"check={','.join(unit.check.selectors) or 'NOT_RUN'}; "
                 f"consumes={consumes}; status={item.status}; check_status={item.check}")
        if item.reason or item.detail:
            line += f"; reason={_render_reason(item.reason or item.status, item.detail)}"
        return line

    ordered = [*failed, *cancelled, *unverified, *committed]
    counts = (f"Counts: committed={len(committed)}; unverified={len(unverified)}; "
              f"failed={len(failed)}; cancelled={len(cancelled)}")
    complete = "\n".join([header, counts, *(row(unit) for unit in ordered)])
    if len(complete.encode("utf-8")) <= PARALLEL_ACTION_SUMMARY_BYTES:
        return complete, 0
    handle = _receipt_pages(result, _receipt_rows(result, request))
    reference = f'Details: read_check_receipt(handle="{handle}")'
    required = [*failed, *cancelled, *unverified]
    lines = [header, counts, *(f"Plan {unit.id}: status={by_id[unit.id].status}; "
               f"check_status={by_id[unit.id].check}; reason_code={reason_code(by_id[unit.id])}"
               for unit in required)]

    def fits(candidate: list[str], remaining: int) -> bool:
        tail = ([f"{remaining} more committed"] if remaining else []) + [reference]
        return len("\n".join([*candidate, *tail]).encode("utf-8")) <= PARALLEL_ACTION_SUMMARY_BYTES

    if not fits(lines, len(committed)):
        summary = "\n".join([header, counts, reference])
        if len(summary.encode("utf-8")) > PARALLEL_ACTION_SUMMARY_BYTES:
            raise ParallelError("subframe receipt index exceeds the F4 action receipt budget")
        return summary, len(committed)
    for index, unit in enumerate(required, 2):
        candidate = [*lines]
        candidate[index] = row(unit)
        if fits(candidate, len(committed)):
            lines = candidate
    kept = 0
    for unit in committed:
        candidate = [*lines, row(unit)]
        remaining = len(committed) - kept - 1
        if not fits(candidate, remaining):
            break
        lines.append(row(unit))
        kept += 1
    omitted = len(committed) - kept
    if omitted:
        lines.append(f"{omitted} more committed")
    lines.append(reference)
    summary = "\n".join(lines)
    if len(summary.encode("utf-8")) > PARALLEL_ACTION_SUMMARY_BYTES:
        raise ParallelError("subframe receipt summary exceeds the F4 action receipt budget")
    return summary, omitted


_PARALLEL_DISPOSITION_ID_FIELDS = WRITER_UNIT_FIELDS
_PARALLEL_DISPOSITION_DETAIL_FIELDS = WRITER_DETAIL_FIELDS
_PARALLEL_DISPOSITION_SCOPE_FIELDS = WRITER_SCOPE_FIELDS
_DISPOSITION_SCOPE_MAX_PATHS = 8


def _record_unit_scope(
        scope_by_id: dict[str, list[object]], unit: UnitContract, public_text: Callable[[object], str],
) -> bool:
    """Store the owned paths and check selectors a continuation pin needs; return True when clipped."""
    truncated = False

    def bounded(values: Sequence[str]) -> list[str]:
        nonlocal truncated
        truncated |= len(values) > _DISPOSITION_SCOPE_MAX_PATHS
        kept = []
        for value in values[:_DISPOSITION_SCOPE_MAX_PATHS]:
            text, was_truncated = _bounded_disposition_text(public_text(value), 128)
            truncated |= was_truncated
            kept.append(text)
        return kept

    scope_by_id[unit.id] = [unit.id, bounded(unit.owns), bounded(unit.check.selectors)]
    return truncated


def _bounded_disposition_text(value: object, byte_limit: int) -> tuple[str, bool]:
    """Return deterministic UTF-8 text with a digest when a field needs clipping."""
    text = value if isinstance(value, str) else str(value)
    raw = text.encode("utf-8", "replace")
    safe = raw.decode("utf-8")
    if len(raw) <= byte_limit:
        return safe, safe != text
    marker = "…#" + hashlib.sha256(raw).hexdigest()[:12]
    marker_bytes = marker.encode("utf-8")
    prefix_bytes = max(0, byte_limit - len(marker_bytes))
    prefix = raw[:prefix_bytes].decode("utf-8", "ignore")
    return prefix + marker, True


def _disposition_wire_bytes(value: Mapping[str, object]) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


def render_parallel_disposition(
        result: ParallelResult,
        request: ParallelRequest | None,
        *,
        journal: object = None,
) -> dict[str, object]:
    """Build the protected, size-bounded frame item for the latest fan-out result."""
    if __package__:
        from .kernel_wire_identity import public_wire_text
    else:
        from kernel_wire_identity import public_wire_text

    def public_text(value: object) -> str:
        return public_wire_text(value if isinstance(value, str) else str(value), journal)

    if request is not None and request.call_id != result.call_id:
        raise ParallelError("subframe disposition has a different plan identity")
    unit_contracts = request.units if request is not None else result.admission.units
    if len(unit_contracts) > MAX_PARALLEL_UNITS:
        raise ParallelError("subframe disposition exceeds the configured unit limit")

    summary, omitted_committed = action_receipt_summary(result, request)
    summary, data_truncated = _bounded_disposition_text(
        public_text(summary), PARALLEL_ACTION_SUMMARY_BYTES,
    )
    final_check, was_truncated = _bounded_disposition_text(public_text(result.final_check), 64)
    data_truncated |= was_truncated
    integration, was_truncated = _bounded_disposition_text(public_text(result.integration), 64)
    data_truncated |= was_truncated
    receipt_section: dict[str, object] | None = None
    if result.final_check_receipt is not None:
        receipt = result.final_check_receipt
        receipt_status, was_truncated = _bounded_disposition_text(public_text(receipt.status), 64)
        data_truncated |= was_truncated
        receipt_reason, was_truncated = _bounded_disposition_text(
            public_text(receipt.invalid_reason or ""), 192)
        data_truncated |= was_truncated
        receipt_section = {"status": receipt_status, "receipt_digest": receipt.digest(),
                           "invalid_reason": receipt_reason}
    refusal = result.admission.refusal
    admission: dict[str, object] = {"status": result.admission.status}
    if refusal is not None:
        reason, was_truncated = _bounded_disposition_text(public_text(refusal.reason), 64)
        data_truncated |= was_truncated
        detail, was_truncated = _bounded_disposition_text(public_text(refusal.detail), 512)
        data_truncated |= was_truncated
        hint, was_truncated = _bounded_disposition_text(public_text(refusal.hint or ""), 192)
        data_truncated |= was_truncated
        paths = []
        for path in refusal.paths[:8]:
            bounded_path, was_truncated = _bounded_disposition_text(public_text(path), 128)
            paths.append(bounded_path)
            data_truncated |= was_truncated
        admission.update(reason=reason, detail=detail, paths=paths, hint=hint or None)
        if len(refusal.paths) > len(paths):
            admission["omitted_paths"] = len(refusal.paths) - len(paths)
            data_truncated = True

    status_by_id: dict[str, str] = {}
    detail_by_id: dict[str, list[object]] = {}
    scope_by_id: dict[str, list[object]] = {}
    if result.admission.status == "REFUSED":
        for unit in unit_contracts:
            reason = refusal.reason if refusal is not None else "ADMISSION_REFUSED"
            bounded_reason, was_truncated = _bounded_disposition_text(public_text(reason), 192)
            data_truncated |= was_truncated
            status_by_id[unit.id] = "NOT_STARTED"
            detail_by_id[unit.id] = [
                unit.id, bounded_reason, [], None, "NOT_RUN", 0,
            ]
            data_truncated |= _record_unit_scope(scope_by_id, unit, public_text)
        ordered_ids = [unit.id for unit in unit_contracts]
    else:
        if request is None:
            raise ParallelError("subframe disposition has no expanded plan")
        by_id = {item.unit_id: item for item in result.outcomes}
        if len(by_id) != len(result.outcomes) or set(by_id) != {unit.id for unit in request.units}:
            raise ParallelError("subframe result does not cover every planned subframe")
        for unit in request.units:
            outcome = by_id[unit.id]
            status_by_id[unit.id] = outcome.status
            reason: str | None = None
            if outcome.reason or outcome.detail:
                reason, was_truncated = _bounded_disposition_text(
                    public_text(_render_reason(outcome.reason or outcome.status, outcome.detail)), 192,
                )
                data_truncated |= was_truncated
            changed_files = []
            for delta in sorted(outcome.files, key=lambda item: (item.path, item.kind)):
                path, was_truncated = _bounded_disposition_text(public_text(delta.path), 128)
                data_truncated |= was_truncated
                changed_files.append([path, delta.kind, delta.added_lines, delta.removed_lines])
            detail_by_id[unit.id] = [
                unit.id, reason, changed_files, outcome.contract_ok, outcome.check,
                outcome.usage.requests,
            ]
            data_truncated |= _record_unit_scope(scope_by_id, unit, public_text)
        failed = [unit.id for unit in request.units if status_by_id[unit.id] not in {
            "COMMITTED", "COMMITTED_UNVERIFIED", "CANCELLED",
        }]
        cancelled = [unit.id for unit in request.units if status_by_id[unit.id] == "CANCELLED"]
        unverified = [unit.id for unit in request.units
                      if status_by_id[unit.id] == "COMMITTED_UNVERIFIED"]
        committed = [unit.id for unit in request.units if status_by_id[unit.id] == "COMMITTED"]
        ordered_ids = [*failed, *cancelled, *unverified, *committed]

    identity_rows = [[unit_id, status_by_id[unit_id]] for unit_id in
                     (unit.id for unit in unit_contracts)]
    detail_rows = [detail_by_id[unit_id] for unit_id in ordered_ids]
    status_counts: dict[str, int] = {}
    for _unit_id, status in identity_rows:
        status_counts[status] = status_counts.get(status, 0) + 1
    section: dict[str, object] = {
        "result_summary": summary,
        "omitted_committed": omitted_committed,
        "admission": admission,
        "final_check": final_check,
        "integration": integration,
        # Additive: the parent final check's receipt; None when no check ran.
        "final_check_receipt": receipt_section,
        "unit_fields": list(_PARALLEL_DISPOSITION_ID_FIELDS),
        "unit_details_fields": list(_PARALLEL_DISPOSITION_DETAIL_FIELDS),
        "units": [],
        "unit_details": [],
        # Additive: owned paths and check selectors for continuation pins; older readers ignore it.
        "unit_scope_fields": list(_PARALLEL_DISPOSITION_SCOPE_FIELDS),
        "unit_scopes": [],
    }
    item: dict[str, object] = {
        "schema": PARALLEL_DISPOSITION_SCHEMA,
        "kind": "parallel_disposition",
        "call_id": result.call_id,
        # Sections let later renderers add a continuation block without changing this item.
        "sections": {"disposition": section},
        "truncation": {},
    }

    def update_truncation(omitted_details: int) -> None:
        omitted_units = len(identity_rows) - len(section["units"])
        omitted_status_counts = {
            status: count for status, count in sorted(status_counts.items()) if count
        }
        item["truncation"] = {
            "truncated": bool(data_truncated or omitted_units or omitted_details),
            "omitted_units": omitted_units,
            "omitted_details": omitted_details,
            "omitted_status_counts": omitted_status_counts,
        }

    section["units"] = []
    section["unit_details"] = []
    update_truncation(len(detail_rows))
    if _disposition_wire_bytes(item) > MAX_PARALLEL_DISPOSITION_BYTES:
        raise ParallelError("subframe disposition metadata exceeds its byte budget")

    for row in identity_rows:
        section["units"].append(row)
        status_counts[row[1]] -= 1
        update_truncation(len(detail_rows))
        if _disposition_wire_bytes(item) > MAX_PARALLEL_DISPOSITION_BYTES:
            section["units"].pop()
            status_counts[row[1]] += 1
            data_truncated = True
            update_truncation(len(detail_rows))

    retained_details = 0
    visible_ids = {row[0] for row in section["units"]}
    for row in detail_rows:
        if row[0] not in visible_ids:
            data_truncated = True
            continue
        section["unit_details"].append(row)
        update_truncation(len(detail_rows) - retained_details - 1)
        if _disposition_wire_bytes(item) > MAX_PARALLEL_DISPOSITION_BYTES:
            section["unit_details"].pop()
            data_truncated = True
            update_truncation(len(detail_rows) - retained_details)
        else:
            retained_details += 1

    # Scopes come last so a tight budget drops pins before it drops committed facts.
    for row in section["unit_details"]:
        scope = scope_by_id.get(str(row[0]))
        if scope is None:
            data_truncated = True
            continue
        section["unit_scopes"].append(scope)
        if _disposition_wire_bytes(item) > MAX_PARALLEL_DISPOSITION_BYTES:
            section["unit_scopes"].pop()
            data_truncated = True
            break

    update_truncation(len(detail_rows) - retained_details)
    if _disposition_wire_bytes(item) > MAX_PARALLEL_DISPOSITION_BYTES:
        raise ParallelError("subframe disposition exceeds its byte budget")
    return item


def writable_workspace() -> bool:
    if _service is None:
        return False
    run, spec = _service
    parent = Path(str(spec.get("parent_snapshot", "")))
    return parent == run / "parent" and parent.is_dir() and os.access(parent, os.W_OK)


def _requirements(runtime: object) -> tuple[RequirementClause, ...]:
    ledger = getattr(runtime, "requirements_ledger", None)
    if ledger is None and getattr(runtime, "stateframe_requirements_ledger_mode", "off") == "on":
        ledger = runtime._requirements_ledger_for(revision=runtime.binding())
    if ledger is None:
        return ()
    if not isinstance(ledger, Mapping):
        raise ParallelError("parent requirements ledger is not an object")
    rows = ledger.get("requirements")
    bindings = ledger.get("bindings", ())
    if not isinstance(rows, list) or not isinstance(bindings, list):
        raise ParallelError("parent requirements ledger has invalid requirements or bindings")
    if any(not isinstance(row, Mapping) or not isinstance(row.get("id"), str) for row in bindings):
        raise ParallelError("parent requirements ledger contains an invalid binding")
    by_id = {row["id"]: row for row in bindings}
    clauses = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or not isinstance(row.get("exact_clause"), str):
            raise ParallelError("parent requirements ledger contains an invalid clause")
        if row["id"] in seen:
            raise ParallelError(f"parent requirements ledger repeats clause {row['id']}")
        seen.add(row["id"])
        contract = row.get("contract", {})
        if not isinstance(contract, Mapping):
            raise ParallelError(f"parent requirement {row['id']} has an invalid contract")
        binding_ids = contract.get("subject_binding_ids", ())
        if (not isinstance(binding_ids, (tuple, list)) or
                any(not isinstance(binding_id, str) for binding_id in binding_ids)):
            raise ParallelError(f"parent requirement {row['id']} has invalid binding references")
        refs = set()
        for binding_id in binding_ids:
            binding = by_id.get(binding_id)
            if binding is None:
                if binding_id == "B-UNRESOLVED":
                    continue
                raise ParallelError(f"parent requirement {row['id']} references unknown binding {binding_id}")
            locator = binding.get("locator")
            if isinstance(locator, Mapping):
                path, symbol = locator.get("path"), locator.get("symbol")
                if isinstance(path, str) and isinstance(symbol, str):
                    refs.add(f"{path}::{symbol}")
                elif isinstance(path, str):
                    refs.add(path)
            elif isinstance(locator, str) and locator:
                refs.add(locator)
        try:
            clauses.append(RequirementClause(row["id"], row["exact_clause"], "intersect", tuple(sorted(refs))))
        except ValueError as error:
            raise ParallelError(f"parent requirement {row['id']} has invalid structured references: {error}") from error
    return tuple(sorted(clauses, key=lambda clause: clause.id))


def context(runtime, bolt, root: Path, created: set[str], *,
            protected: Callable[[str], bool], deadline: float | None = None) -> ParallelContext:
    if _service is None:
        raise ParallelError("subframe service is not installed")
    run, spec = _service
    limits = ParallelLimits.from_json(spec["parallel_limits"])
    resume = spec.get("resume")
    attempt = int(resume.get("attempt", 1)) if isinstance(resume, dict) else 1
    return ParallelContext(runtime, bolt, Path(root), created, str(spec["run_identity_seed"]),
                           attempt, _requirements(runtime), limits, str(spec["parallel_mode"]), protected, deadline)


def _binding(root: Path, runtime) -> EngineBinding:
    if __package__:
        from .kernel_live_projection import workspace_binding
    else:
        from kernel_live_projection import workspace_binding
    active = runtime.binding()
    projection = workspace_binding(root, str(active.get("environment_revision", "")))
    return EngineBinding(str(projection["product_revision"]), str(projection["test_harness_revision"]))


class _StartSnapshot:
    """A read-only copy whose authenticated manifest retains source modes."""

    def __init__(self, root: Path, generation: int):
        self.root = root
        self.generation = generation
        if root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
            raise ParallelError("subframe start snapshot root is not read-only")
        path = root / "snapshot-manifest.json"
        if path.is_symlink() or not path.is_file():
            raise ParallelError("subframe start snapshot manifest is missing")
        try:
            manifest = json.loads(path.read_bytes())
        except (ValueError, OSError) as error:
            raise ParallelError(f"subframe start snapshot manifest is invalid: {error}") from error
        if (not isinstance(manifest, dict) or set(manifest) !=
                {"schema", "files", "directories", "source_manifest_digest"} or
                manifest["schema"] != "stateframe-snapshot-manifest/1" or
                not isinstance(manifest["files"], list) or
                not isinstance(manifest["directories"], list)):
            raise ParallelError("subframe start snapshot manifest has invalid fields")
        self.snapshot_manifest_digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
        self.manifest_digest = manifest["source_manifest_digest"]
        rows = manifest["files"]
        expected = []
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"path", "kind", "mode", "sha256"}
                    or row["kind"] != "file" or not isinstance(row["path"], str)
                    or type(row["mode"]) is not int or not isinstance(row["sha256"], str)):
                raise ParallelError("subframe start snapshot has an invalid file entry")
            relative = Path(row["path"])
            if (relative.is_absolute() or ".." in relative.parts or
                    relative.as_posix() != row["path"] or row["path"] == "snapshot-manifest.json"):
                raise ParallelError("subframe start snapshot has an unsafe file entry")
            file = root / relative
            if (file.is_symlink() or not file.is_file() or stat.S_IMODE(file.stat().st_mode) != 0o444
                    or hashlib.sha256(file.read_bytes()).hexdigest() != row["sha256"]):
                raise ParallelError(f"subframe start snapshot file differs: {row['path']}")
            expected.append(row["path"])
        actual_paths = list(root.rglob("*"))
        if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in actual_paths):
            raise ParallelError("subframe start snapshot contains an unsafe entry")
        if expected != sorted(set(expected)) or set(expected) != {
                path.relative_to(root).as_posix() for path in actual_paths
                if path.is_file() and path != root / "snapshot-manifest.json"}:
            raise ParallelError("subframe start snapshot file inventory differs")
        source_digest = hashlib.sha256(b"".join(canonical_json(row) + b"\n" for row in rows)).hexdigest()
        if source_digest != self.manifest_digest:
            raise ParallelError("subframe start snapshot source digest differs")
        self._files = expected
        directory_names = []
        for row in manifest["directories"]:
            if (not isinstance(row, dict) or set(row) != {"path", "mode"}
                    or not isinstance(row["path"], str) or type(row["mode"]) is not int):
                raise ParallelError("subframe start snapshot has an invalid directory entry")
            directory = root / row["path"]
            if (directory.is_symlink() or not directory.is_dir() or
                    stat.S_IMODE(directory.stat().st_mode) != 0o555):
                raise ParallelError(f"subframe start snapshot directory differs: {row['path']}")
            directory_names.append(row["path"])
        if directory_names != sorted(set(directory_names)) or set(directory_names) != {
                path.relative_to(root).as_posix() for path in actual_paths if path.is_dir()}:
            raise ParallelError("subframe start snapshot directory inventory differs")

    def read_bytes(self, path: str) -> bytes:
        if path not in self._files:
            raise ParallelError(f"subframe start snapshot does not contain {path}")
        return (self.root / path).read_bytes()

    def exists(self, path: str) -> bool:
        return path in self._files

    def list_files(self) -> list[str]:
        return list(self._files)


def _snapshot(root: Path, call_root: Path, generation: int) -> _StartSnapshot:
    destination = call_root / "snapshots" / f"g{generation}"
    source = DirectorySnapshot(root, generation)
    if destination.is_dir():
        existing = _StartSnapshot(destination, generation)
        if existing.manifest_digest != source.manifest_digest:
            raise ParallelError("existing subframe start snapshot differs from authoritative generation")
        return existing
    if destination.exists() or destination.is_symlink():
        raise ParallelError("subframe start snapshot path is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".g{generation}-{os.getpid()}-{secrets.token_hex(8)}"
    shutil.copytree(root, temporary, symlinks=True)
    copied = DirectorySnapshot(temporary, generation)
    if copied.manifest_digest != source.manifest_digest:
        raise ParallelError("subframe start snapshot copy changed during capture")
    rows = [json.loads(line) for line in pre_manifest_blob(temporary).splitlines()]
    directories = [{"path": path.relative_to(temporary).as_posix(),
                    "mode": stat.S_IMODE(path.stat().st_mode)}
                   for path in sorted(temporary.rglob("*")) if path.is_dir()]
    manifest = {"schema": "stateframe-snapshot-manifest/1", "files": rows,
                "directories": directories, "source_manifest_digest": source.manifest_digest}
    _atomic_json(temporary / "snapshot-manifest.json", manifest)
    for path in sorted(temporary.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    temporary.chmod(0o555)
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return _StartSnapshot(destination, generation)


def _proposal(path: Path, expected_digest: str, unit: UnitContract,
              execution_id: str, incarnation: int) -> tuple[Proposal, dict[str, bytes]]:
    if path.is_symlink() or path.name != "proposal.json":
        raise ParallelError("subframe proposal path is unsafe")
    value = json.loads(path.read_bytes())
    proposal = Proposal.from_json(value)
    if (proposal.digest() != expected_digest or proposal.unit_id != unit.id
            or proposal.execution_id != execution_id or proposal.incarnation != incarnation
            or proposal.contract_digest != unit.digest()):
        raise ParallelError("subframe sealed proposal identity differs from its launch")
    allowed = set(unit.owns) | set(unit.creates)
    if any(entry.path not in allowed or entry.kind != ("create" if entry.path in unit.creates else "modify")
           for entry in proposal.entries):
        raise ParallelError("SCOPE_VIOLATION: subframe proposal exceeds declared ownership")
    blobs = {}
    for entry in proposal.entries:
        blob = path.parent / "blobs" / entry.after_sha256
        if blob.is_symlink() or not blob.is_file():
            raise ParallelError(f"sealed subframe blob is missing: {entry.path}")
        data = blob.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.after_sha256:
            raise ParallelError(f"sealed subframe blob digest differs: {entry.path}")
        blobs[entry.after_sha256] = data
    if not set(unit.creates).issubset({entry.path for entry in proposal.entries}):
        raise ParallelError("SCOPE_VIOLATION: subframe omitted a required created file")
    return proposal, blobs


def _compose(root: Path, entries: tuple[ProposalEntry, ...], blobs: Mapping[str, bytes],
             destination: Path) -> DirectorySnapshot:
    shutil.copytree(root, destination, symlinks=True)
    for entry in entries:
        path = destination / entry.path
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                    if parent.is_relative_to(destination) and parent != destination):
            raise ParallelError(f"unsafe subframe composition path: {entry.path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if entry.kind == "create" and path.exists():
            raise ParallelError(f"CREATE_EXISTS: {entry.path}")
        if entry.kind == "modify" and (not path.is_file() or
            hashlib.sha256(path.read_bytes()).hexdigest() != entry.before_sha256):
            raise ParallelError(f"STALE_BASE: {entry.path}")
        path.write_bytes(blobs[entry.after_sha256])
        path.chmod(entry.mode)
    return DirectorySnapshot(destination, 0)


def _initial_frame_tokens(run_dir: Path) -> int | None:
    attempts = run_dir / "attempts"
    roots = (sorted(attempts.glob("[0-9]*/worker/measurement-runs"),
                    key=lambda path: int(path.parents[1].name)) if attempts.is_dir() else [])
    roots.append(run_dir / "worker" / "measurement-runs")
    for root in roots:
        if root.is_symlink():
            raise ParallelError(f"subframe frame-token root is a symlink: {root}")
        # journal_paths silently skips symlinked run directories; this guard refuses them instead.
        if root.is_dir() and any(child.is_symlink() for child in root.iterdir()):
            raise ParallelError(f"subframe frame-token journal directory is a symlink under {root}")
        # Through the journal shim, so a sealed (.jsonl.xz) journal is read like a live one.
        for journal in journal_paths(root):
            if journal.is_symlink():
                raise ParallelError(f"subframe frame-token journal is a symlink: {journal}")
            with open_journal(journal) as source:
                for number, line in enumerate(source, 1):
                    try:
                        event = json.loads(line)
                    except ValueError as error:
                        raise ParallelError(f"invalid subframe frame-token journal at {journal}:{number}") from error
                    if not isinstance(event, dict):
                        raise ParallelError(f"invalid subframe frame-token journal at {journal}:{number}")
                    if event.get("kind") == "unit_frame_tokens":
                        tokens = event.get("tokens")
                        if type(tokens) is not int or tokens < 0:
                            raise ParallelError(f"invalid subframe frame-token count at {journal}:{number}")
                        return tokens
    return None


def _usage(run_dir: Path) -> UnitUsage:
    records_by_send: dict[int, dict] = {}
    attempts = run_dir / "attempts"
    paths = (sorted(attempts.glob("[0-9]*/sends.jsonl"),
                    key=lambda path: int(path.parent.name)) if attempts.is_dir() else [])
    paths.append(run_dir / "sends.jsonl")
    for sends in paths:
        if not sends.is_file() or sends.is_symlink():
            continue
        for line in sends.read_text().splitlines():
            item = json.loads(line)
            index = item.get("send")
            if type(index) is not int or index < 1:
                raise ParallelError(f"invalid subframe send record in {sends}")
            records_by_send[index] = item
    records = list(records_by_send.values())
    ledger = run_dir / "sends-ledger.jsonl"
    physical = 0
    if ledger.is_file():
        physical = sum(1 for line in ledger.read_text().splitlines()
                       if json.loads(line).get("state") == "AUTHORIZED")
    frame_tokens = _initial_frame_tokens(run_dir)
    if records and frame_tokens is None:
        raise ParallelError(f"subframe request records lack an initial frame-token measurement: {run_dir}")
    return UnitUsage(len(records), physical,
                     sum(int(item.get("prompt_tokens") or 0) for item in records),
                     sum(int(item.get("completion_tokens") or 0) for item in records),
                     frame_tokens)


def _empty_usage() -> UnitUsage:
    return UnitUsage(0, 0, 0, 0, None)


def _file_deltas(entries: tuple[ProposalEntry, ...], root: Path, blobs: Mapping[str, bytes]) -> tuple[FileDelta, ...]:
    result = []
    for entry in entries:
        before = ((root / entry.path).read_text(encoding="utf-8", errors="replace").splitlines()
                  if entry.kind == "modify" else [])
        after = blobs[entry.after_sha256].decode("utf-8", errors="replace").splitlines()
        import difflib
        changes = list(difflib.ndiff(before, after))
        result.append(FileDelta(entry.path, sum(line.startswith("+ ") for line in changes),
                                sum(line.startswith("- ") for line in changes), entry.kind))
    return tuple(result)


def _check(spec, candidate: Path, evidence_dir: Path, *, test_python: str):
    if __package__:
        from .kernel_unit_policy import (UnitPolicy, _reap_check_token, _terminate_check_token, new_check_token, reap_check_group,
                                         run_check_command, run_frozen_check)
        from .kernel_execution_policy import ExecutionPolicyError, check_cpu_backstop_seconds, run_sandboxed
    else:
        from kernel_unit_policy import (UnitPolicy, _reap_check_token, _terminate_check_token, new_check_token, reap_check_group,
                                        run_check_command, run_frozen_check)
        from kernel_execution_policy import ExecutionPolicyError, check_cpu_backstop_seconds, run_sandboxed

    from kernel_check_watchdog import check_watchdog_config

    def execute(clone: Path, selector: str) -> str:
        if spec.contract.check.command_kind == "run_check_command":
            return run_check_command(clone, selector)
        token = new_check_token()
        try:
            completed = run_sandboxed([test_python, "-m", "pytest", selector], clone,
                                      timeout=None, check_watchdog=check_watchdog_config(),
                                      cpu_seconds=check_cpu_backstop_seconds(), output_limit=256 * 1024,
                                      internal_env={"STATEFRAME_CHECK_TOKEN": token},
                                      pre_reap_cleanup=lambda: _terminate_check_token(token))
        except (ExecutionPolicyError, OSError) as error:
            _reap_check_token(token)
            return f"pytest exit 4: {type(error).__name__}: {error}"
        if getattr(completed, "cleanup_finding", None) is not None:
            from kernel_unit_policy import UnitCheckCleanupError
            raise UnitCheckCleanupError(completed.cleanup_finding)
        reap_check_group(completed.pid, token=token)
        execution_stop = getattr(completed, "execution_stop", None)
        if execution_stop is not None:
            from kernel_check_watchdog import IncompleteCheck
            return IncompleteCheck("check incomplete: " + str(execution_stop["cause"])
                                   + "\n" + completed.completed.stdout + completed.completed.stderr,
                                   execution_stop, exit_code=completed.completed.returncode)
        return (f"pytest exit {completed.completed.returncode}\n"
                f"{completed.completed.stdout}{completed.completed.stderr}")

    return run_frozen_check(UnitPolicy(spec), candidate, execute=execute, evidence_dir=evidence_dir)


def _refusal_location(refusal: Refusal, request: ParallelRequest) -> tuple[str, str, str]:
    if refusal.reason == "UNIT_CONTEXT_TOO_LARGE" and "source_lower_bound_tokens=" in refusal.detail:
        for unit in request.units:
            if any(path in (*unit.owns, *unit.reads, *unit.check.source_paths)
                   for path in refusal.paths):
                return (unit.id, "owns/reads/check.tests",
                        "split this subframe's source scope or raise --parallel-subframe-tokens")
    if refusal.reason == "TOO_MANY_UNITS":
        return "all", "subframes", "remove subframes until the named maximum is met, or raise that maximum"
    if refusal.reason == "TOO_FEW_UNITS":
        return "all", "subframes", "add subframes until the named minimum is met, or lower that minimum"
    if refusal.reason == "RESOURCES_UNAVAILABLE":
        return "all", "subframes", "reduce the request reservation or increase the available parent turns"
    if refusal.reason == "DEPENDENCY_CYCLE":
        return ",".join(refusal.members) or "all", "consumes", "remove the declared dependency cycle"
    if refusal.reason == "OWNERSHIP_OVERLAP":
        return ",".join(refusal.members) or "all", "owns/creates", "give each path one owner"
    if refusal.reason == "MISSING_SIGNATURE":
        for unit in request.units:
            for number, export in enumerate(unit.exports):
                if export.id == refusal.detail:
                    return unit.id, f"exports[{number}].signature", "supply the canonical Python signature"
    if refusal.reason == "UNRESOLVED_CONSUME":
        for unit in request.units:
            for number, consume in enumerate(unit.consumes):
                if ((consume.export_id and consume.export_id in refusal.detail)
                        or (consume.path and consume.path in refusal.paths)):
                    export = next((export for producer in request.units for export in producer.exports
                                   if export.id == consume.export_id), None)
                    if export is not None:
                        corrected = {"export_id": export.id, "path": export.path,
                                     "symbol": export.symbol}
                        return unit.id, f"consumes[{number}]", f"send {json.dumps(corrected)}"
                    return unit.id, f"consumes[{number}]", "name an existing definition or declare its producer"
    if refusal.reason in {"AMBIGUOUS_CONSUME", "UNSUPPORTED_CONTRACT"}:
        for unit in request.units:
            for number, consume in enumerate(unit.consumes):
                if ((consume.export_id and consume.export_id in refusal.detail)
                        or (consume.path and consume.path in refusal.paths)):
                    return unit.id, f"consumes[{number}]", "identify a supported, unambiguous definition"
    if refusal.reason in {"EXPORT_OUTSIDE_OWNERSHIP", "DUPLICATE_EXPORT"}:
        for unit in request.units:
            for number, export in enumerate(unit.exports):
                if export.id == refusal.detail:
                    return unit.id, f"exports[{number}]", "declare a unique export in an owned path"
    for unit in request.units:
        for path in refusal.paths:
            for field in ("owns", "creates", "reads"):
                if path in getattr(unit, field):
                    return unit.id, field, "correct this subframe's path declaration"
            if path in unit.check.source_paths:
                return unit.id, "check.tests", "choose an allowed test selector"
    if refusal.reason == "UNSAFE_PATH" and "duplicate subframe id" in refusal.detail:
        seen = set()
        for unit in request.units:
            if unit.id in seen:
                return unit.id, "id", "give every subframe a unique id"
            seen.add(unit.id)
    for unit in request.units:
        if unit.id in refusal.detail:
            return unit.id, "id/outcome", "correct this subframe contract"
    return "all", "subframes", "adjust the subframe contracts and retry"


def _expanded_plan_lines(request: ParallelRequest | None) -> list[str]:
    if request is None:
        return []
    lines = []
    for unit in request.units:
        consumes = ",".join(consume.export_id or
                            f"{consume.path or ''}::{consume.symbol or ''}"
                            for consume in unit.consumes) or "-"
        lines.append(f"Plan {unit.id}: owns={','.join(unit.owns) or '-'}; "
                     f"check={','.join(unit.check.selectors) or 'NOT_RUN'}; consumes={consumes}")
    return lines


def _salvage_line(call_id: str, record: Mapping[str, object]) -> str:
    """One parent-visible line for a subframe that stopped with unsubmitted edits."""
    files = ", ".join(str(path) for path in record.get("files") or ()) or "not determined"
    if record.get("adoption") == "ADOPTED":
        verdict = f"its edits passed the commit gates and were adopted (txn {record.get('txn_id')})"
    else:
        reason = " ".join(str(record.get("reason") or "").split())[:300]
        verdict = f"its edits are kept but not adopted ({reason})"
    return (f"Subframe {record.get('unit')} salvage: stopped {record.get('terminal')} before submitting; "
            f"{verdict}; files: {files}; bundle parallel/{call_id}/{record.get('path')} "
            f"sha256={record.get('sha256')}")


def _observation(admission: AdmissionResult, outcomes: tuple[UnitOutcome, ...],
                 final_check: str, integration: str, measured: ParallelMeasured,
                 request: ParallelRequest | None = None, *, tiny_plan: bool = False,
                 salvage: Mapping[str, Mapping[str, object]] | None = None,
                 call_id: str = "") -> str:
    if admission.refusal is not None:
        refusal = admission.refusal
        evidence = "; ".join(f"{item.path}:{item.line} {item.text}" for item in refusal.evidence)
        message = (f"REFUSED {_render_reason(refusal.reason, refusal.detail)}"
                   + (f"; {evidence}" if evidence else ""))
        if request is not None:
            unit_id, field, hint = _refusal_location(refusal, request)
            message += f"; subframe {unit_id}; field {field}; fix: {hint}"
            plan_lines = _expanded_plan_lines(request)
            if plan_lines:
                message += "\n" + "\n".join(plan_lines)
        if admission.adjustments:
            rows = [f"{item.action} {item.unit_id} {item.path}: {item.reason}"
                    for item in admission.adjustments]
            message += "\nPlan adjustments: " + "; ".join(rows)
        return message
    lines = ["Subframe results:",
             f"subframe_request_cap={admission.limits.unit_max_requests}"]
    for item in admission.adjustments:
        lines.append(f"Plan adjustment {item.action} {item.unit_id} {item.path}: {item.reason}")
    lines.extend(_expanded_plan_lines(request))
    for item in outcomes:
        files = ", ".join(f"{delta.path} (+{delta.added_lines}/-{delta.removed_lines})"
                          for delta in item.files) or "no file changes"
        lines.append(f"Subframe {item.unit_id}: {item.status}; {files}; contract={item.contract_ok}; "
                      f"check={item.check}; requests={item.usage.requests}" +
                      (f"; reason={_render_reason(item.reason, item.detail)}" if item.reason else ""))
        if salvage and item.unit_id in salvage:
            lines.append(_salvage_line(call_id, salvage[item.unit_id]))
    invalid = [item.unit_id for item in outcomes if item.check == "INVALID"]
    if invalid:
        lines.append(f"Check INVALID (not a pass) for: {', '.join(invalid)}")
    if outcomes and not any(item.files for item in outcomes):
        lines.append("No files changed by any subframe.")
    lines.append(f"Final check: {final_check}; integration: {integration}")
    if tiny_plan:
        lines.append(TINY_PLAN_ADVISORY)
    # Scheduling measurements stay in ParallelResult, outside provider-visible text.
    return "\n".join(lines)


def _subframe_reason(reason: str) -> str:
    return {
        "TOO_MANY_UNITS": "TOO_MANY_SUBFRAMES",
        "TOO_FEW_UNITS": "TOO_FEW_SUBFRAMES",
        "UNIT_CONTEXT_TOO_LARGE": "SUBFRAME_CONTEXT_TOO_LARGE",
    }.get(reason, reason)


def _render_reason(reason: str, detail: str | None) -> str:
    code = _subframe_reason(reason)
    text = " ".join((detail or "").split())
    if text in {reason, code}:
        text = ""
    for prefix in (reason + ": ", code + ": "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return code + (f": {text}" if text else "")


def _publish_txn(store: TxnStore, record: TxnRecord, context: ParallelContext) -> None:
    current = manifest_digest(context.root)
    if current == record.pre_manifest_digest and _binding(context.root, context.runtime) != record.pre_binding:
        raise RecoveryRequired("RECOVERY_REQUIRED: subframe transaction pre-binding differs")
    # DECIDED can precede a crash after any individual file replacement. The
    # transaction store authenticates every unchanged and changed path before it
    # completes the materialization; the partial manifest is not a refusal.
    store.materialize(record.txn_id, context.root)
    if __package__:
        from .kernel_shell_tool import publish_host_effects
    else:
        from kernel_shell_tool import publish_host_effects
    changed = tuple(entry.path for entry in record.entries)
    publish_host_effects(context.runtime, context.bolt, context.created, changed=changed,
                         created_paths=tuple(entry.path for entry in record.entries if entry.kind == "create"),
                         name="parallel")
    context.runtime._invalidate_binding_cache()
    if (manifest_digest(context.root) != record.post_manifest_digest or
            _binding(context.root, context.runtime) != record.post_binding):
        raise RecoveryRequired("RECOVERY_REQUIRED: subframe publication binding differs")
    store.mark_published(record.txn_id, f"parallel:{record.call_id}:{record.txn_id}")
    context.runtime.journal.emit("parallel_txn", txn_id=record.txn_id, unit=record.unit_id,
                                 disposition="PUBLISHED", pre_generation=record.pre_generation,
                                 post_generation=record.post_generation)


def _latest_records(store: TxnStore):
    latest = {}
    for record, disposition in store.records():
        latest[record.txn_id] = record, disposition
    return tuple(latest[number] for number in sorted(latest))


def _replay_txns(store: TxnStore, context: ParallelContext) -> tuple[int, set[str]]:
    generation = 0
    committed = set()
    for record, _ in _latest_records(store):
        if record.pre_generation != generation:
            raise ToolReplayRefused("subframe transaction generation chain diverged")
        _publish_txn(store, record, context)
        generation = record.post_generation
        committed.add(record.unit_id)
    return generation, committed


def _parent_environment_digest() -> str:
    """Parent-side environment digest. Must equal the worker-side value so PASS receipts reuse."""
    if __package__:
        from .kernel_unit_policy import _check_environment_digest
    else:
        from kernel_unit_policy import _check_environment_digest
    return _check_environment_digest()


def _candidate_check(spec, candidate: Path, call_root: Path, context: ParallelContext,
                     worker_receipt):
    if spec.contract.check.command_kind == "none" or not spec.contract.check.selectors:
        return None
    if __package__:
        from .kernel_effects import scan
        from .kernel_unit_policy import _manifest_digest
    else:
        from kernel_effects import scan
        from kernel_unit_policy import _manifest_digest

    def ignored(path: str) -> bool:
        return any(part in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
                            ".git", ".venv", "venv"} for part in Path(path).parts)

    source = scan(candidate, exclude=ignored)
    product_digest = _manifest_digest(source)
    harness_digest = _manifest_digest(source, set(spec.contract.check.source_paths))
    environment_digest = _parent_environment_digest()
    if (worker_receipt is not None and worker_receipt.status == "PASS"
            and worker_receipt.recipe_digest == spec.contract.check.digest()
            and worker_receipt.product_manifest_digest == product_digest
            and worker_receipt.harness_manifest_digest == harness_digest
            and worker_receipt.environment_digest == environment_digest
            and worker_receipt.policy_digest == spec.policy_digest):
        return worker_receipt
    test_python = os.environ.get("STATEFRAME_TEST_PYTHON", sys.executable)
    return _check(spec, candidate, call_root / "checks" / spec.unit_id, test_python=test_python)


def _accept_proposal(context: ParallelContext, request: ParallelRequest, admission: AdmissionResult,
                     spec, proposal: Proposal, blobs: dict[str, bytes], call_root: Path,
                     store: TxnStore, generation: int, usage: UnitUsage,
                     wall: float, queue_wait: float) -> tuple[UnitOutcome, int, tuple]:
    unit = spec.contract
    for entry in proposal.entries:
        path = context.root / entry.path
        if entry.kind == "modify":
            if (not path.is_file() or path.is_symlink() or
                    hashlib.sha256(path.read_bytes()).hexdigest() != entry.before_sha256):
                raise ParallelError(f"STALE_BASE: {entry.path}")
        elif path.exists() or path.is_symlink():
            raise ParallelError(f"CREATE_EXISTS: {entry.path}")
    with tempfile.TemporaryDirectory(prefix="stateframe-parallel-composition-") as temporary:
        composed_root = Path(temporary) / "candidate"
        composed = _compose(context.root, proposal.entries, blobs, composed_root)
        rescan = rescan_proposal(request.units, admission, composed, unit.id)
        if rescan.refusal is not None:
            raise ParallelError(f"{rescan.refusal.reason}: {rescan.refusal.detail}")
        _, current_pairs = scan_coupling(request.units, composed, context.limits)
        report = verify_contracts(request.units, composed, unit_id=unit.id)
        if not report.ok:
            _record_refused_contracts(call_root, unit.id, report)
            failures = [verdict.reason for verdict in report.verdicts if verdict.status == "FAIL"]
            failures.extend("UNRESOLVED_IMPORT" for item in report.imports if item.status == "UNRESOLVED")
            raise ParallelError(", ".join(failures) or "subframe contract verification failed")
        receipt = _candidate_check(spec, composed_root, call_root, context, proposal.check_receipt)
        check_status = receipt.status if receipt is not None else "NOT_RUN"
        if receipt is not None and receipt.status != "PASS":
            raise CompositionCheckFailed(
                receipt.status, f"COMPOSITION_CHECK_FAILED: {receipt.status}: {receipt.invalid_reason or ''}")
        if any(item.status == "UNKNOWN" for item in report.verdicts) and check_status != "PASS":
            _record_refused_contracts(call_root, unit.id, report)
            raise ParallelError("UNVERIFIABLE_CONTRACT: subframe has no passing composition check")
        pre_blob = pre_manifest_blob(context.root)
        pre_digest = hashlib.sha256(pre_blob).hexdigest()
        if pre_digest != manifest_digest(context.root):
            raise ParallelError("subframe pre-manifest algorithms disagree")
        pre_binding = _binding(context.root, context.runtime)
        post_binding = _binding(composed_root, context.runtime)
        post_digest = manifest_digest(composed_root)
        files = _file_deltas(proposal.entries, context.root, blobs)
        txn_id = len(_latest_records(store)) + 1
        record = TxnRecord(txn_id, request.call_id, unit.id, spec.execution_id,
                           generation, generation + 1, proposal.entries, pre_digest, post_digest,
                           pre_binding, post_binding, report.digest(),
                           receipt.digest() if receipt is not None else None, usage)
        try:
            store.decide(record, {**blobs, pre_digest: pre_blob})
        except (TxnStoreError, OSError) as error:
            if any(existing.txn_id == txn_id for existing, _ in _latest_records(store)):
                raise RecoveryRequired(f"RECOVERY_REQUIRED: subframe decision {txn_id}: {error}") from error
            raise ParallelError(f"subframe transaction decision failed: {error}") from error
    context.runtime.journal.emit("parallel_txn", txn_id=record.txn_id, unit=unit.id,
                                 disposition="DECIDED", pre_generation=generation,
                                 post_generation=generation + 1)
    try:
        _publish_txn(store, record, context)
    except (OSError, ValueError, RuntimeError) as error:
        raise RecoveryRequired(f"RECOVERY_REQUIRED: subframe transaction {txn_id}: {error}") from error
    outcome = UnitOutcome(unit.id, "COMMITTED" if check_status == "PASS" else "COMMITTED_UNVERIFIED",
                          None, None, txn_id, files, True, check_status, usage, wall, queue_wait)
    return outcome, generation + 1, tuple(current_pairs)


def _result(context: ParallelContext, call_root: Path, request: ParallelRequest,
            admission: AdmissionResult, outcomes: Mapping[str, UnitOutcome],
            store: TxnStore, unknown_pairs: tuple, started: float, peak: int,
            final_check: str, integration: str, *,
            final_check_receipt: CheckReceipt | None = None) -> ParallelResult:
    ordered = tuple(outcomes[unit.id] for unit in request.units if unit.id in outcomes)
    elapsed = round(max(0.0, time.monotonic() - started), 3)
    serial = round(sum(item.wall_seconds for item in ordered), 3)
    measured = ParallelMeasured(elapsed, serial,
                                round(serial / elapsed, 3) if elapsed else 0.0, peak,
                                round(sum(item.queue_wait_seconds for item in ordered), 3))
    tiny_plan = (admission.status == "ADMITTED"
                 and _tiny_plan(request, _StartSnapshot(call_root / "snapshots" / "g0", 0)))
    observation = _observation(admission, ordered, final_check, integration, measured,
                               request, tiny_plan=tiny_plan,
                               salvage=_salvage_adoptions(call_root, request), call_id=request.call_id)
    result = ParallelResult(request.call_id, admission, ordered,
                            tuple(record.txn_id for record, _ in _latest_records(store)),
                            final_check, unknown_pairs, integration, observation, measured,
                            final_check_receipt=final_check_receipt)
    _emit_late_parallel_degrade(context.runtime, ordered, len(request.units))
    _atomic_json(call_root / "result.json", result.to_json())
    return result


def _peak_interval_overlap(intervals: list[tuple[float, float]]) -> int:
    """Count half-open worker intervals; exits at a timestamp precede starts."""
    events = []
    for start, finish in intervals:
        if (not math.isfinite(start) or not math.isfinite(finish) or finish < start):
            raise ParallelError("subframe timing interval is invalid")
        if finish > start:
            events.extend(((start, 1), (finish, -1)))
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    return peak


def _peak_active_subframes(call_root: Path) -> int:
    intervals: dict[tuple[str, int], tuple[float, float]] = {}
    directory = call_root / "intervals"
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            if path.is_symlink():
                raise ParallelError("subframe timing record is a symlink")
            try:
                row = json.loads(path.read_bytes())
            except (OSError, ValueError) as error:
                raise ParallelError(f"subframe timing record is invalid: {error}") from error
            if not isinstance(row, dict):
                raise ParallelError("subframe timing record is not an object")
            start, finish = row.get("running_since"), row.get("finished_epoch")
            unit_id, separator, incarnation_text = path.stem.rpartition("-")
            if (not separator or not incarnation_text.isdecimal()
                    or type(start) not in (int, float) or type(finish) not in (int, float)):
                raise ParallelError("subframe timing record has invalid interval fields")
            intervals[unit_id, int(incarnation_text)] = (float(start), float(finish))
    host_directory = call_root / "host" / "units"
    if host_directory.is_dir():
        for path in sorted(host_directory.glob("*/[0-9]*.json")):
            if path.is_symlink():
                raise ParallelError("subframe host timing record is a symlink")
            try:
                row = json.loads(path.read_bytes())
            except (OSError, ValueError) as error:
                raise ParallelError(f"subframe host timing record is invalid: {error}") from error
            if not isinstance(row, dict):
                raise ParallelError("subframe host timing record is not an object")
            start, finish = row.get("started_epoch"), row.get("finished_epoch")
            if start is None or finish is None:
                continue
            if type(start) not in (int, float) or type(finish) not in (int, float):
                raise ParallelError("subframe host timing record has invalid interval fields")
            key = path.parent.name, int(path.stem)
            interval = float(start), float(finish)
            if key in intervals and intervals[key] != interval:
                raise ParallelError("subframe timing record differs from host evidence")
            intervals[key] = interval
    return _peak_interval_overlap(list(intervals.values()))


def _parent_steps(context: ParallelContext) -> int:
    """Parent turns spent so far, including the turn that issued this call.

    The ReAct loop stamps ``bolt.current_step`` with its 0-based loop index each turn; the
    measurement runtime does not carry a step, so reading only the runtime always yielded 0."""
    stamped = getattr(context.bolt, "current_step", None)
    if type(stamped) is int and stamped >= 0:
        return stamped + 1
    for source in (getattr(context.runtime, "owner", None), context.runtime):
        step = getattr(source, "current_step", None)
        if type(step) is int and step > 0:
            return step
    return 0


def _record_refused_contracts(call_root: Path, unit_id: str, report: object) -> None:
    """Keep a refused subframe's verdicts; final-contracts.json lists surviving commits only."""
    _atomic_json(call_root / "refused-contracts" / f"{unit_id}.json", report.to_json(),
                 replace=True)


def _final_check_receipt(call_root: Path) -> CheckReceipt | None:
    """The parent final check's receipt written by _final_check; None when no check ran."""
    path = call_root / _FINAL_CHECK_RECEIPT
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ParallelError("final subframe check receipt is not a regular file")
    try:
        return CheckReceipt.from_json(json.loads(path.read_bytes()))
    except (OSError, ValueError) as error:
        raise ParallelError(f"final subframe check receipt is invalid: {error}") from error


def _final_check(context: ParallelContext, request: ParallelRequest, call_root: Path,
                 unknown_pairs: tuple, surviving: set[str] | None = None) -> tuple[str, str]:
    if _service is None:
        raise ParallelError("subframe service is not installed")
    _, parent_spec = _service
    # A receipt from an earlier evaluation of this call must not outlive this one.
    stale = call_root / _FINAL_CHECK_RECEIPT
    if stale.is_symlink() or stale.is_file():
        stale.unlink()
    surviving = surviving if surviving is not None else {unit.id for unit in request.units}
    snapshot = DirectorySnapshot(context.root, 0)
    contracts = {unit.id: verify_contracts(request.units, snapshot, unit_id=unit.id)
                 for unit in request.units if unit.id in surviving}
    _atomic_json(call_root / "final-contracts.json",
                 {unit_id: report.to_json() for unit_id, report in sorted(contracts.items())})
    contracts_ok = all(report.ok for report in contracts.values())
    if not surviving:
        return "NOT_RUN", "UNVERIFIED" if unknown_pairs else "NOT_RUN"
    test_cmd = parent_spec.get("test_cmd")
    selectors = tuple(dict.fromkeys(selector for unit in request.units
                                    for selector in unit.check.selectors))
    if isinstance(test_cmd, str) and test_cmd.strip():
        recipe = CheckRecipe((), (test_cmd,), "run_check_command")
    elif selectors:
        recipe = CheckRecipe(tuple(sorted({selector.split("::", 1)[0] for selector in selectors})),
                             selectors, "pytest")
    else:
        return "NOT_RUN", "FAILED" if not contracts_ok else "UNVERIFIED" if unknown_pairs else "NOT_RUN"
    first = next(unit for unit in request.units if unit.id in surviving)
    first_spec_path = call_root / "units" / first.id / "unit-spec.json"
    if not first_spec_path.is_file():
        raise ParallelError(f"final subframe check lacks an authenticated spec for {first.id}")
    if __package__:
        from .kernel_parallel_records import UnitSpec
    else:
        from kernel_parallel_records import UnitSpec
    first_spec = UnitSpec.from_json(json.loads(first_spec_path.read_bytes()))
    all_paths = tuple(sorted({path for unit in request.units for path in
                              (*unit.owns, *unit.creates, *unit.reads, *unit.check.source_paths)}))
    final_spec = dataclasses.replace(first_spec,
                                     contract=dataclasses.replace(first_spec.contract, check=recipe),
                                     visible_paths=all_paths)
    test_python = os.environ.get("STATEFRAME_TEST_PYTHON", sys.executable)
    receipt = _check(final_spec, context.root, call_root / "checks" / "final", test_python=test_python)
    _atomic_json(call_root / _FINAL_CHECK_RECEIPT, receipt.to_json(), replace=True)
    if not contracts_ok:
        return receipt.status, "FAILED"
    if receipt.status == "PASS":
        return "PASS", "CHECKED_WITH_UNKNOWNS" if unknown_pairs else "VERIFIED"
    return receipt.status, "FAILED" if receipt.status == "FAIL" else "UNVERIFIED"


def _resource_admission(admission: AdmissionResult, context: ParallelContext,
                        count: int) -> AdmissionResult:
    if _service is None:
        raise ParallelError("subframe service is not installed")
    run, spec = _service
    allowance = spec.get("max_turns")
    budget = BudgetStore(run, limit=allowance if type(allowance) is int and allowance > 0 else None,
                         parent_reserve=context.limits.parent_reserve_requests)
    parent_steps = _parent_steps(context)
    if type(allowance) is int and allowance > 0 and \
            allowance < parent_steps + context.limits.parent_reserve_requests:
        refusal = Refusal(
            "RESOURCES_UNAVAILABLE",
            f"{format_parallel_limit('unit_max_requests', context.limits.unit_max_requests)}, "
            f"{format_parallel_limit('parent_reserve_requests', context.limits.parent_reserve_requests)}; "
            f"required={count * context.limits.unit_max_requests}, free=0", (), (), ())
        return dataclasses.replace(admission, status="REFUSED", refusal=refusal)
    budget.recover(parent_steps_spent=parent_steps)
    free = json.loads(budget.path.read_bytes())["free"]
    explicit = spec.get("parallel_subframe_max_requests_explicit", False)
    if type(explicit) is not bool:
        raise ParallelError("subframe request-cap source must be a boolean")
    explicit = explicit or context.limits.unit_max_requests != ParallelLimits().unit_max_requests
    cap = context.limits.unit_max_requests
    if not explicit and free is not None:
        cap = min(cap, free // count)
        if cap < 3:
            refusal = Refusal(
                "RESOURCES_UNAVAILABLE",
                f"subframe_request_cap={cap} is below minimum 3; "
                f"free={free}, subframes={count}; "
                f"{format_parallel_limit('parent_reserve_requests', context.limits.parent_reserve_requests)}",
                (), (), ())
            return dataclasses.replace(admission, status="REFUSED", refusal=refusal)
    reservations = {f"pending_{index}": cap for index in range(count)}
    if not budget.can_reserve(reservations, parent_steps_spent=parent_steps):
        required = count * cap
        refusal = Refusal(
            "RESOURCES_UNAVAILABLE",
            f"{format_parallel_limit('unit_max_requests', cap)}, "
            f"{format_parallel_limit('parent_reserve_requests', context.limits.parent_reserve_requests)}; "
            f"required={required}, free={free}", (), (), ())
        return dataclasses.replace(admission, status="REFUSED", refusal=refusal)
    return dataclasses.replace(admission, limits=dataclasses.replace(admission.limits,
                                                                     unit_max_requests=cap))


def _emit_parallel_degrade(runtime: object, admission: AdmissionResult,
                           subframes: tuple[str, ...]) -> None:
    """Make a capacity fallback visible even when no subframe was launched."""
    refusal = admission.refusal
    if refusal is None or refusal.reason not in {
            "UNIT_CONTEXT_TOO_LARGE", "RESOURCES_UNAVAILABLE"}:
        return
    runtime.journal.emit("parallel_degraded", reason=refusal.reason,
                         detail=refusal.detail, subframes=list(subframes))


def _emit_late_parallel_degrade(runtime: object, outcomes: tuple[UnitOutcome, ...],
                                planned_count: int) -> None:
    if not outcomes or len(outcomes) != planned_count or not all(
            item.status == "REFUSED" and item.reason in {
                "SUBFRAME_CONTEXT_TOO_LARGE", "RESOURCES_UNAVAILABLE"}
            for item in outcomes):
        return
    runtime.journal.emit(
        "parallel_degraded",
        reason=",".join(sorted({str(item.reason) for item in outcomes})),
        detail="; ".join(f"{item.unit_id}: {item.detail or item.reason}" for item in outcomes),
        subframes=[item.unit_id for item in outcomes])


def _declared_dependencies(admission: AdmissionResult) -> dict[str, set[str]]:
    dependencies = {unit_id: set() for unit_id in admission.order}
    for edge in admission.edges:
        dependencies[edge.dst_unit].add(edge.src_unit)
    return dependencies


_SALVAGE_MAX_BYTES = 16 << 20
_SALVAGE_JOURNAL_FILES = 64
_SALVAGE_JOURNAL_PATH_CHARS = 200


def _read_salvage_file(path: Path) -> bytes | None:
    """Read a regular file of at most _SALVAGE_MAX_BYTES without following a symlink.

    The open refuses a symlink itself (O_NOFOLLOW), so a path swapped after an
    earlier check cannot redirect the read. Returns None for a symlink, a
    non-regular file or an oversized one; other OS errors propagate.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError as error:
        if error.errno == errno.ELOOP:
            return None
        raise
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _SALVAGE_MAX_BYTES:
            return None
        raw = handle.read(_SALVAGE_MAX_BYTES + 1)
    return raw if len(raw) <= _SALVAGE_MAX_BYTES else None


def _salvage_bundle_number(name: str) -> int:
    return int(name[:-len(".json")])


def _salvage_changed_files(bundle: Mapping, base: Path | None, oldest: Mapping | None) -> list[str]:
    """Paths a candidate bundle changed, judged against the bytes the subframe started from.

    The bundle's patch covers its single target file only, so it cannot be the
    edit signal by itself. With the start snapshot the comparison is exact; without
    it the oldest bundle of the same subframe stands in for the starting bytes.
    """
    hashes = bundle.get("file_hashes")
    hashes = hashes if isinstance(hashes, Mapping) else {}
    earlier = oldest.get("file_hashes") if isinstance(oldest, Mapping) else None
    changed = set()
    for name, digest in hashes.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            continue
        if base is not None:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            try:
                current = _read_salvage_file(base / relative)
            except OSError:
                current = None  # absent or unreadable in the start snapshot: the bundle changed it
            if current is None or hashlib.sha256(current).hexdigest() != digest:
                changed.add(name)
        elif isinstance(earlier, Mapping) and earlier.get(name) != digest:
            changed.add(name)
    deleted = bundle.get("deleted_files")
    if isinstance(deleted, list):
        changed.update(name for name in deleted if isinstance(name, str))
    patch, target = bundle.get("patch"), bundle.get("target")
    if isinstance(patch, str) and patch.strip() and isinstance(target, str) and target:
        changed.add(target)
    return sorted(changed)


def preserve_unit_candidate(call_root: Path, unit_id: str, *,
                            start_generation: int | None = None) -> dict[str, object] | None:
    """Keep a subframe's newest readable candidate bundle in the parent call root.

    The child captures its workspace edits as worker/candidates/NNNN.json before every
    request. When its work does not reach the parent (it ended without a submission,
    or its submission was not accepted), that bundle is the only record of its edits.
    Copy it, byte for byte, to parallel/<call>/salvage/<unit>.json so the parent keeps
    it regardless of unit-directory cleanup. A bundle cut short by a killed worker is
    skipped for the previous one. Returns the receipt, or None when the unit left no
    edit. Never raises: salvage must not change the unit's terminal outcome.
    """
    try:
        candidates = call_root / "units" / unit_id / "worker" / "candidates"
        if candidates.is_symlink() or not candidates.is_dir():
            return None
        names = sorted((path.name for path in candidates.iterdir()
                        if re.fullmatch(r"\d{4,}\.json", path.name)), key=_salvage_bundle_number)
        if not names:
            return None
        chosen = None
        for name in reversed(names):
            raw = _read_salvage_file(candidates / name)
            if raw is None:
                return {"status": "REFUSED", "reason": "candidate bundle is a symlink or too large"}
            try:
                bundle = json.loads(raw)
            except (ValueError, RecursionError):
                continue  # a capture interrupted mid-write; the previous one is still whole
            if isinstance(bundle, Mapping):
                chosen = name, raw, bundle
                break
        if chosen is None:
            return {"status": "FAILED", "reason": "no readable candidate bundle"}
        name, raw, bundle = chosen
        base = oldest = None
        if start_generation is not None:
            base = call_root / "snapshots" / f"g{start_generation}"
            if base.is_symlink() or not base.is_dir():
                base = None
        if base is None and name != names[0]:
            first = _read_salvage_file(candidates / names[0])
            if first is not None:
                try:
                    oldest = json.loads(first)
                except (ValueError, RecursionError):
                    oldest = None
        changed = _salvage_changed_files(bundle, base, oldest)
        if not changed:
            return None
        target = call_root / "salvage" / f"{unit_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=f".{unit_id}.", suffix=".tmp")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
        return {"status": "PRESERVED", "path": str(target.relative_to(call_root)),
                "source": name, "sha256": hashlib.sha256(raw).hexdigest(),
                "files": [path[:_SALVAGE_JOURNAL_PATH_CHARS] for path in changed[:_SALVAGE_JOURNAL_FILES]],
                "files_total": len(changed)}
    except Exception as error:  # noqa: BLE001 - bookkeeping; the journal gets the type, never host paths
        return {"status": "FAILED", "reason": type(error).__name__}


def _journal_salvage(context: ParallelContext, unit_id: str, terminal: str,
                     salvage: Mapping[str, object], **fields: object) -> None:
    try:
        context.runtime.journal.emit("parallel_unit_salvaged", unit=unit_id, terminal=terminal,
                                     **salvage, **fields)
    except Exception:  # noqa: BLE001 - a journal failure must not change the unit's outcome
        pass


SALVAGE_ADOPT_SWITCH = "STATEFRAME_PARALLEL_SALVAGE_ADOPT"
SALVAGE_ADOPTION_SCHEMA = "stateframe-parallel-salvage-adoption/1"
# The child's candidate capture never records these directories.
_SALVAGE_UNCAPTURED_PARTS = frozenset({"__pycache__", ".pytest_cache", ".test-tmp"})


def salvage_adopt_enabled() -> bool:
    """Whether the parent offers a stopped subframe's salvaged edits to the commit gates."""
    value = os.environ.get(SALVAGE_ADOPT_SWITCH, "off")
    if value not in {"on", "off"}:
        raise ParallelError(f"{SALVAGE_ADOPT_SWITCH} must be on or off")
    return value == "on"


def _salvage_proposal(call_root: Path, unit: UnitContract, spec, incarnation: int,
                      receipt: Mapping[str, object]) -> tuple[Proposal, dict[str, bytes]]:
    """Turn a preserved candidate bundle into a proposal against the subframe's start snapshot.

    The bundle holds the subframe's declared files and every file that differs from
    its starting bytes; a file it does not list is unchanged. Entries are derived
    here, by the parent, from bytes the parent authenticated: the start snapshot the
    subframe was launched on and the bundle digest recorded when it was preserved.
    The result carries no check receipt, so acceptance reruns the frozen check itself.
    """
    path = call_root / "salvage" / f"{unit.id}.json"
    if receipt.get("status") != "PRESERVED":
        raise ParallelError(f"SALVAGE_INVALID: candidate bundle was not preserved: {receipt.get('reason')}")
    try:
        raw = _read_salvage_file(path)
    except OSError:
        raw = None
    if raw is None:
        raise ParallelError("SALVAGE_INVALID: candidate bundle is missing, a symlink or too large")
    if hashlib.sha256(raw).hexdigest() != receipt.get("sha256"):
        raise ParallelError("SALVAGE_INVALID: candidate bundle differs from its preserved digest")
    try:
        bundle = json.loads(raw)
    except ValueError as error:
        raise ParallelError(f"SALVAGE_INVALID: candidate bundle is not JSON: {error}") from error
    encoded = bundle.get("files_base64") if isinstance(bundle, Mapping) else None
    if not isinstance(encoded, Mapping) or not all(
            isinstance(name, str) and isinstance(data, str) for name, data in encoded.items()):
        raise ParallelError("SALVAGE_INVALID: candidate bundle has no file capture")
    if bundle.get("deleted_files"):
        raise ParallelError("SALVAGE_UNSUPPORTED: subframe proposals do not support deleted files: "
                            f"{[str(name)[:120] for name in list(bundle['deleted_files'])[:6]]}")
    snapshot = _StartSnapshot(call_root / "snapshots" / f"g{spec.start_generation}", spec.start_generation)
    if snapshot.manifest_digest != spec.start_manifest_digest:
        raise ParallelError("SALVAGE_INVALID: start snapshot differs from the subframe launch")
    rows = json.loads((snapshot.root / "snapshot-manifest.json").read_bytes())["files"]
    before = {row["path"]: (row["sha256"], row["mode"]) for row in rows}

    def captured(name: str) -> bool:
        return not _SALVAGE_UNCAPTURED_PARTS.intersection(Path(name).parts)

    files: dict[str, bytes] = {}
    for name, data in encoded.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name or not name:
            raise ParallelError(f"SALVAGE_INVALID: candidate bundle has an unsafe path: {name[:120]}")
        try:
            files[name] = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ParallelError(f"SALVAGE_INVALID: candidate bundle bytes are malformed: {name}") from error
    after = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    changed = sorted(name for name, digest in after.items()
                     if captured(name) and (name not in before or before[name][0] != digest))
    allowed = set(unit.owns) | set(unit.creates)
    outside = [name for name in changed if name not in allowed]
    if outside:
        raise ParallelError(f"SCOPE_VIOLATION: salvaged subframe edits exceed declared ownership: {outside[:6]}")
    if not changed:
        raise ParallelError("SALVAGE_EMPTY: candidate bundle holds no change to the start snapshot")
    missing = sorted(set(unit.creates) - set(changed))
    if missing:
        raise ParallelError(f"SALVAGE_INCOMPLETE: subframe stopped before creating declared files: {missing[:6]}")
    entries = []
    for name in changed:
        if name in unit.creates:
            entries.append(ProposalEntry(name, "create", None, after[name], 0o644))
        elif name in before:
            entries.append(ProposalEntry(name, "modify", before[name][0], after[name], before[name][1]))
        else:
            raise ParallelError(f"SCOPE_VIOLATION: salvaged subframe created an undeclared file: {name}")
    proposal = Proposal(spec.call_id, unit.id, spec.execution_id, incarnation, tuple(entries),
                        unit.digest(), None)
    return proposal, {entry.after_sha256: files[entry.path] for entry in entries}


def _record_salvage_adoption(call_root: Path, unit_id: str, terminal: str,
                             receipt: Mapping[str, object], files: Sequence[str],
                             outcome: UnitOutcome | None, reason: str | None) -> dict[str, object]:
    record = {"schema": SALVAGE_ADOPTION_SCHEMA, "unit": unit_id, "terminal": terminal,
              "path": receipt.get("path"), "sha256": receipt.get("sha256"), "files": list(files),
              "adoption": "ADOPTED" if outcome is not None else "REFUSED",
              "txn_id": outcome.txn_id if outcome is not None else None,
              "reason": None if outcome is not None else " ".join(str(reason or "").split())[:300]}
    try:
        _atomic_json(call_root / "salvage" / f"{unit_id}.adoption.json", record, replace=True)
    except (ParallelError, OSError):
        pass  # the journal still carries the verdict; bookkeeping never changes an outcome
    return record


def _salvage_adoptions(call_root: Path, request: ParallelRequest) -> dict[str, dict[str, object]]:
    """Adoption records the parent wrote for this call; none exist while the switch is off."""
    folder = call_root / "salvage"
    result = {}
    if folder.is_symlink() or not folder.is_dir():
        return result
    for unit in request.units:
        path = folder / f"{unit.id}.adoption.json"
        try:
            raw = _read_salvage_file(path)
            if raw is None:
                continue
            record = json.loads(raw)
        except (OSError, ValueError):
            continue
        if (isinstance(record, dict) and record.get("schema") == SALVAGE_ADOPTION_SCHEMA
                and record.get("unit") == unit.id):
            result[unit.id] = record
    return result


def _failure_outcome(unit_id: str, reason: str, detail: str, usage: UnitUsage,
                     wall: float, queue_wait: float, *, check: str = "NOT_RUN") -> UnitOutcome:
    status = reason if reason in {"BLOCKED", "BUDGET_EXHAUSTED", "ENGINE_ERROR", "TIMED_OUT", "CANCELLED"} else "REFUSED"
    return UnitOutcome(unit_id, status, reason, detail[:300], None, (), None,
                       check, usage, round(wall, 3), round(queue_wait, 3))


def _blocked_outcome(call_root: Path, unit_id: str, usage: UnitUsage,
                     wall: float, queue_wait: float) -> UnitOutcome:
    """Retain the typed model blocker after the host reports its terminal code."""
    path = call_root / "units" / unit_id / "engine-result.json"
    try:
        if path.is_symlink() or path.stat().st_size > (1 << 20):
            raise ParallelError("subframe blocked receipt is a symlink or exceeds the read limit")
        envelope = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise ParallelError(f"subframe blocked receipt is unavailable: {error}") from error
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict) or result.get("result") != "SUBFRAME_BLOCKED":
        raise ParallelError("subframe blocked receipt disagrees with the host terminal")
    code, detail = result.get("reason_code"), result.get("detail")
    if (code not in {"NEEDS_UNOWNED_FILE", "INTERFACE_WRONG", "CONTRACT_IMPOSSIBLE",
                     "CHECK_WRONG", "OTHER"} or not isinstance(detail, str)
            or len(detail) > 300):
        raise ParallelError("subframe blocked receipt has an invalid reason or detail")
    return UnitOutcome(unit_id, "BLOCKED", code, detail, None, (), None,
                       "NOT_RUN", usage, round(wall, 3), round(queue_wait, 3))


def _record_start(call_root: Path, intent: dict, spec, snapshot: _StartSnapshot) -> None:
    execution = ExecutionState(spec.unit_id, spec.execution_id,
                               str(call_root / "units" / spec.unit_id), spec.digest(), 1, None)
    current = [item for item in intent["executions"] if item["unit_id"] != spec.unit_id]
    intent["executions"] = [*current, execution.to_json()]
    intent["snapshot_manifests"][f"g{spec.start_generation}"] = snapshot.snapshot_manifest_digest
    _atomic_json(call_root / "intent.json", intent, replace=True)


def _record_outcome(call_root: Path, outcome: UnitOutcome, budget: BudgetStore,
                    parent_steps_spent: int) -> None:
    path = call_root / "outcomes.jsonl"
    if path.is_symlink():
        raise ParallelError("subframe outcomes log is a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as output:
        output.write(canonical_json(outcome.to_json()) + b"\n")
        output.flush()
        os.fsync(output.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    budget.terminal(call_root.name, outcome.unit_id, parent_steps_spent=parent_steps_spent)


def _load_outcomes(call_root: Path) -> dict[str, UnitOutcome]:
    path = call_root / "outcomes.jsonl"
    if path.is_symlink():
        raise ToolReplayRefused("subframe outcomes log is a symlink")
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ToolReplayRefused("subframe outcomes log ends with an incomplete record")
    result = {}
    for number, line in enumerate(data.splitlines(), 1):
        try:
            outcome = UnitOutcome.from_json(json.loads(line))
        except (ValueError, TypeError) as error:
            raise ToolReplayRefused(f"subframe outcomes log line {number} is invalid: {error}") from error
        if outcome.unit_id in result:
            raise ToolReplayRefused(f"subframe outcomes log repeats terminal subframe {outcome.unit_id}")
        result[outcome.unit_id] = outcome
    return result


def _same_input_refusal(recorded: ParallelResult, current: ParallelResult,
                        error: ParallelInputError) -> bool:
    if recorded == current:
        return True
    old = recorded.admission.refusal
    if (old is None or old.reason != "INVALID_ARGUMENTS" or old.hint is not None
            or old.detail != str(error)):
        return False
    legacy = dataclasses.replace(current, admission=dataclasses.replace(
        current.admission, refusal=old))
    return recorded == legacy


def _input_refusal(context: ParallelContext, args: Mapping, error: ParallelInputError,
                   call_id: str, raw_call_id: str, call_root: Path) -> ParallelResult:
    tape = tape_for_run()
    record = tape.lookup(raw_call_id)
    intent = {"schema": "stateframe-parallel-input-refusal/1", "arguments": args}
    current_manifest = manifest_digest(context.root)
    refusal = Refusal("INVALID_ARGUMENTS", f"subframe {error.unit_id} {error.field}: {error.detail}",
                      (), (), (), error.hint)
    admission = AdmissionResult("REFUSED", refusal, (), (), (), 0, current_manifest, context.limits)
    measured = ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0)
    observation = f"REFUSED INVALID_ARGUMENTS: {error}"
    result = ParallelResult(call_id, admission, (), (), "NOT_RUN", (), "NOT_RUN", observation, measured)
    environment = str(context.runtime.binding().get("environment_revision") or "disabled")
    if record is not None:
        if (record.tool != "parallel" or record.replay_class != "local"
                or record.args != intent or record.pre_manifest_digest != current_manifest
                or record.capability_digest != hashlib.sha256(b"stateframe-subframe-input/1").hexdigest()
                or record.definition_digest != hashlib.sha256(canonical_json(args)).hexdigest()
                or record.env_identity != environment):
            raise ToolReplayRefused("subframe input refusal differs from its recorded call")
        if record.completed:
            if record.post_manifest_digest != current_manifest or record.post_env_identity != environment:
                raise ToolReplayRefused("subframe input refusal changed the workspace")
            result_path = call_root / "result.json"
            if result_path.is_symlink() or not result_path.is_file():
                raise ToolReplayRefused("subframe input refusal result is missing")
            try:
                recorded = ParallelResult.from_json(json.loads(result_path.read_bytes()))
            except (OSError, ValueError) as recorded_error:
                raise ToolReplayRefused(f"subframe input refusal result is invalid: {recorded_error}") from recorded_error
            if not _same_input_refusal(recorded, result, error) or record.observation != observation:
                raise ToolReplayRefused("subframe input refusal observation changed")
            return recorded
        recovery = _service[1].get("parallel_recovery") if _service is not None else None
        if (not isinstance(recovery, dict)
                or recovery.get("schema") != "stateframe-parallel-input-refusal/1"
                or recovery.get("call_id") != raw_call_id
                or recovery.get("arguments") != args):
            raise ToolReplayRefused("pending subframe input refusal lacks validated recovery")
    if record is None:
        tape.intent(call_id=raw_call_id, tool="parallel", args=intent, replay_class="local",
                    capability_digest=hashlib.sha256(b"stateframe-subframe-input/1").hexdigest(),
                    definition_digest=hashlib.sha256(canonical_json(args)).hexdigest(),
                    env_identity=environment, pre_manifest_digest=current_manifest)
    if call_root.is_symlink():
        raise ToolReplayRefused("subframe input refusal artifact root is a symlink")
    call_root.mkdir(parents=True, exist_ok=True)
    result_path = call_root / "result.json"
    if record is not None and result_path.exists():
        if result_path.is_symlink():
            raise ToolReplayRefused("pending subframe input refusal result is a symlink")
        try:
            recorded = ParallelResult.from_json(json.loads(result_path.read_bytes()))
        except (OSError, ValueError) as recorded_error:
            raise ToolReplayRefused(f"pending subframe input refusal result is invalid: {recorded_error}") from recorded_error
        if not _same_input_refusal(recorded, result, error):
            raise ToolReplayRefused("pending subframe input refusal result changed")
        result = recorded
    else:
        _atomic_json(result_path, result.to_json())
    tape.complete(call_id=raw_call_id, exit_code=0, observation=observation,
                  post_manifest_digest=current_manifest, env_identity=environment)
    return result


def _recorded_input_refusal(context: ParallelContext, args: Mapping, record,
                            raw_call_id: str, call_id: str, call_root: Path) -> ParallelResult:
    """Replay a sealed refusal, including historical hint wording, with no effects."""
    current_manifest = manifest_digest(context.root)
    environment = str(context.runtime.binding().get("environment_revision") or "disabled")
    if (record.tool != "parallel" or record.replay_class != "local"
            or record.args != {"schema": "stateframe-parallel-input-refusal/1", "arguments": args}
            or record.pre_manifest_digest != current_manifest
            or record.env_identity != environment
            or record.capability_digest != hashlib.sha256(b"stateframe-subframe-input/1").hexdigest()
            or record.definition_digest != hashlib.sha256(canonical_json(args)).hexdigest()):
        raise ToolReplayRefused("subframe input refusal differs from its recorded call")
    if record.completed:
        if record.post_manifest_digest != current_manifest or record.post_env_identity != environment:
            raise ToolReplayRefused("subframe input refusal changed the workspace")
    else:
        recovery = _service[1].get("parallel_recovery") if _service is not None else None
        if (not isinstance(recovery, dict)
                or recovery.get("schema") != "stateframe-parallel-input-refusal/1"
                or recovery.get("call_id") != raw_call_id
                or recovery.get("arguments") != args):
            raise ToolReplayRefused("pending subframe input refusal lacks validated recovery")
    if call_root.is_symlink():
        raise ToolReplayRefused("subframe input refusal artifact root is a symlink")
    path = call_root / "result.json"
    if path.is_symlink() or not path.is_file():
        raise ToolReplayRefused("subframe input refusal result is missing")
    try:
        result = ParallelResult.from_json(json.loads(path.read_bytes()))
    except (OSError, ValueError) as error:
        raise ToolReplayRefused(f"subframe input refusal result is invalid: {error}") from error
    if (result.call_id != call_id or result.admission.status != "REFUSED"
            or result.admission.refusal is None
            or result.admission.refusal.reason != "INVALID_ARGUMENTS"
            or result.admission.manifest_digest != current_manifest
            or result.admission.limits != context.limits
            or result.admission.edges or result.admission.pairs or result.admission.order
            or result.outcomes or result.txn_order
            or result.final_check != "NOT_RUN" or result.integration != "NOT_RUN"
            or not result.observation.startswith("REFUSED INVALID_ARGUMENTS: ")
            or (record.completed and result.observation != record.observation)):
        raise ToolReplayRefused("subframe input refusal observation changed")
    if not record.completed:
        tape_for_run().complete(call_id=raw_call_id, exit_code=0, observation=result.observation,
                                post_manifest_digest=current_manifest, env_identity=environment)
    return result


def execute_parallel(context: ParallelContext, args: Mapping, *, host: HostClientProtocol,
                     store_root: Path, schema_errors: Sequence[str] = ()) -> ParallelResult:
    """Admit, run, validate and publish one parent-owned subframe call.

    schema_errors carries a rejection from the advertised tool schema. It is
    sealed as the same typed refusal receipt as any other input refusal, so
    every parallel exchange leaves a durable receipt for the frame fold.
    """
    if _service is None:
        raise ParallelError("subframe service is not installed")
    run, parent_spec = _service
    allowance = parent_spec.get("max_turns")
    budget = BudgetStore(run, limit=allowance if type(allowance) is int and allowance > 0 else None,
                         parent_reserve=context.limits.parent_reserve_requests)
    parent_steps = _parent_steps(context)
    raw_call_id = str(getattr(context.runtime, "current_tool_call_id", ""))
    if not raw_call_id:
        raise ParallelError("subframe tool call lacks a recorded identity")
    call_id = (raw_call_id if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", raw_call_id)
               else "call_" + hashlib.sha256(raw_call_id.encode()).hexdigest()[:40])
    call_root = Path(store_root)
    if call_root != run / "parallel" / call_id:
        raise ParallelError("subframe artifact root differs from its run-scoped call identity")
    recorded = tape_for_run().lookup(raw_call_id)
    if (recorded is not None and isinstance(recorded.args, dict)
            and recorded.args.get("schema") == "stateframe-parallel-input-refusal/1"):
        if recorded.completed or (call_root / "result.json").exists():
            return _recorded_input_refusal(context, args, recorded, raw_call_id, call_id, call_root)
    pre_snapshot = DirectorySnapshot(context.root, 0)
    if (recorded is not None and isinstance(recorded.args, dict)
            and recorded.args.get("schema") == "stateframe-parallel-input-refusal/1"
            and not recorded.completed and isinstance(args, Mapping)
            and set(args) == {"subframes"} and isinstance(args["subframes"], list)
            and args["subframes"] and isinstance(args["subframes"][0], dict)):
        first = args["subframes"][0]
        if ("goal" not in first and "outcome" not in first
                and set(first) <= set(_UNIT["properties"])
                and isinstance(first.get("id"), str) and first["id"].strip()):
            # This was the first parser refusal before requirement-based completion existed.
            historical = ParallelInputError(first["id"], "outcome", "expected nonempty text",
                                            "provide outcome as text")
            return _input_refusal(context, args, historical, call_id, raw_call_id, call_root)
    if (recorded is not None and isinstance(recorded.args, dict)
            and recorded.args.get("schema") == "stateframe-parallel-input-refusal/1"
            and isinstance(args, Mapping) and isinstance(args.get("subframes"), list)
            and any(isinstance(unit, dict) and "goal" in unit for unit in args["subframes"])):
        # An old pending intent may have stopped before its refusal artifact was sealed.
        try:
            _test_imports(pre_snapshot, _module_map(pre_snapshot), context.limits)
        except ParallelInputError as error:
            return _input_refusal(context, args, error, call_id, raw_call_id, call_root)
        for index, unit in enumerate(args["subframes"]):
            if not isinstance(unit, dict) or "goal" not in unit:
                continue
            goal = unit["goal"]
            if not isinstance(goal, str) or not goal.strip() or len(goal) > 300:
                break
            if "outcome" in unit and unit["outcome"] != goal:
                unit_id = unit.get("id") if isinstance(unit.get("id"), str) else str(index)
                error = ParallelInputError(unit_id, "outcome", "conflicts with goal",
                                           "make outcome equal goal or omit outcome")
                return _input_refusal(context, args, error, call_id, raw_call_id, call_root)
    if schema_errors:
        return _input_refusal(context, args, schema_input_error(args, schema_errors),
                              call_id, raw_call_id, call_root)
    try:
        completed_args, plan_actions = _complete_arguments(args, pre_snapshot, context.limits)
        request = _request(completed_args, call_id, context.mode, requirements=context.requirements)
        planner_requirements = _planner_clauses(completed_args, context.requirements)
    except ParallelInputError as error:
        return _input_refusal(context, args, error, call_id, raw_call_id, call_root)
    context = dataclasses.replace(context, requirements=planner_requirements)
    if recorded is None:
        if __package__:
            from .kernel_parallel_meter import (_parent_meter_calibration,
                                                _parent_subframe_cap_calibration)
        else:
            from kernel_parallel_meter import (_parent_meter_calibration,
                                               _parent_subframe_cap_calibration)
        calibration = _parent_meter_calibration(context.runtime)
        cap_calibration = _parent_subframe_cap_calibration(context.runtime)
    else:
        calibration = recorded.args.get("meter_calibration") if isinstance(recorded.args, dict) else None
        if calibration is not None and not isinstance(calibration, dict):
            raise ToolReplayRefused("recorded subframe calibration snapshot is invalid")
        cap_calibration = (recorded.args.get("subframe_cap_calibration")
                           if isinstance(recorded.args, dict) else None)
        if cap_calibration is not None and not isinstance(cap_calibration, dict):
            raise ToolReplayRefused("recorded subframe cap profile is invalid")
    pre_snapshot.meter_calibration = calibration
    pre_admission = admit(request, pre_snapshot, context.requirements, context.limits,
                          protected=context.protected, case_insensitive=sys.platform == "darwin")
    if pre_admission.status == "ADMITTED":
        request = dataclasses.replace(request, units=pre_admission.units)
        request = _bind_implicit_signatures(request, pre_snapshot)
        pre_admission = dataclasses.replace(pre_admission, units=request.units)
    tape = tape_for_run()
    record = tape.lookup(raw_call_id)
    if record is not None and record.tool != "parallel":
        raise ToolReplayRefused("subframe call id conflicts with a different recorded tool")
    if record is not None and record.completed:
        if record.replay_class != "local" or not isinstance(record.args, dict):
            raise ToolReplayRefused("completed subframe tape has an invalid replay class or intent")
        saved_request = ParallelRequest.from_json(record.args["request"])
        if saved_request != request:
            raise ToolReplayRefused("subframe request changed before completed replay")
        path = call_root / "result.json"
        if path.is_symlink() or not path.is_file():
            raise ToolReplayRefused("completed subframe result is missing")
        result = ParallelResult.from_json(json.loads(path.read_bytes()))
        if result.call_id != call_id or result.observation != record.observation:
            raise ToolReplayRefused("completed subframe observation differs from its tape")
        store = TxnStore(call_root / "txn")
        generation, _ = _replay_txns(store, context)
        if (result.txn_order != tuple(item.txn_id for item, _ in _latest_records(store))
                or generation != len(result.txn_order)
                or manifest_digest(context.root) != record.post_manifest_digest):
            raise ToolReplayRefused("completed subframe replay produced a different workspace")
        return result

    recovery = parent_spec.get("parallel_recovery") if record is not None else None
    if record is not None and (not isinstance(recovery, dict) or recovery.get("call_id") != call_id):
        raise ToolReplayRefused("pending subframe intent lacks a validated recovery descriptor")
    generation = 0
    committed: set[str] = set()
    store = TxnStore(call_root / "txn")
    if record is None:
        admission = pre_admission
        if admission.status == "ADMITTED":
            admission = _resource_admission(admission, context, len(request.units))
        initial_snapshot = _snapshot(context.root, call_root, generation) if admission.status == "ADMITTED" else None
        intent = {"schema": "stateframe-parallel-intent/1", "call_id": call_id,
                   "request": request.to_json(), "admission": admission.to_json(),
                   "meter_calibration": calibration,
                   "subframe_cap_calibration": cap_calibration,
                   "requirements_digest": hashlib.sha256(canonical_json(
                       [clause.to_json() for clause in context.requirements])).hexdigest(),
                  "reservations": {unit.id: admission.limits.unit_max_requests for unit in request.units},
                  "generation": admission.generation, "mode": context.mode,
                   "policy_digest": hashlib.sha256(b"stateframe-subframe-policy/1").hexdigest(),
                   "snapshot_manifests": ({"g0": initial_snapshot.snapshot_manifest_digest}
                                          if initial_snapshot is not None else {}),
                   "executions": []}
        pre_digest = manifest_digest(context.root)
        environment = str(context.runtime.binding().get("environment_revision") or "disabled")
        tape.intent(call_id=raw_call_id, tool="parallel", args=intent, replay_class="local",
                    capability_digest=intent["policy_digest"], definition_digest=request.digest(),
                    env_identity=environment, pre_manifest_digest=pre_digest)
        call_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(call_root / "intent.json", intent)
        for action in plan_actions:
            context.runtime.journal.emit("parallel_plan_completed", **action)
        if admission.status == "ADMITTED":
            for unit in request.units:
                context.runtime.journal.emit(
                    "parallel_plan_completed", subframe=unit.id, field="request_cap",
                    action="expanded_field", producer=None, source="admission_limits",
                    reason_code="RESOLVED_REQUEST_CAP", value=admission.limits.unit_max_requests)
        if admission.status == "ADMITTED" and _tiny_plan(request, pre_snapshot):
            context.runtime.journal.emit(
                "parallel_plan_completed", action="tiny_plan_advisory",
                reason_code="TINY_PLAN", detail=TINY_PLAN_ADVISORY)
        if admission.status == "ADMITTED" and not budget.reserve(
                call_id, intent["reservations"], parent_steps_spent=parent_steps):
            raise ParallelError("subframe reservation changed after admission")
        context.runtime.journal.emit("parallel_admission", status=admission.status,
                                     reason=admission.refusal.reason if admission.refusal else None,
                                     subframe_request_cap=(admission.limits.unit_max_requests
                                                           if admission.status == "ADMITTED" else None),
                                     units=[unit.id for unit in request.units],
                                     edges=[edge.to_json() for edge in admission.edges],
                                     pairs=[pair.to_json() for pair in admission.pairs],
                                     adjustments=[item.to_json() for item in admission.adjustments])
        _emit_parallel_degrade(context.runtime, admission,
                               tuple(unit.id for unit in request.units))
    else:
        intent = record.args
        if (not isinstance(intent, dict) or intent.get("schema") != "stateframe-parallel-intent/1"
                or ParallelRequest.from_json(intent["request"]) != request):
            raise ToolReplayRefused("pending subframe intent differs from the requested call")
        admission = AdmissionResult.from_json(intent["admission"])
        if admission.status == "ADMITTED":
            budget.recover(parent_steps_spent=parent_steps)
            if not budget.reserve(call_id, intent["reservations"],
                                  parent_steps_spent=parent_steps):
                raise ToolReplayRefused("subframe recovery cannot restore the parent reservation")
        expected_requirements = hashlib.sha256(canonical_json(
            [clause.to_json() for clause in context.requirements])).hexdigest()
        if intent.get("requirements_digest") != expected_requirements:
            raise ToolReplayRefused("subframe parent requirements ledger changed during recovery")
        if not store.records() and manifest_digest(context.root) != record.pre_manifest_digest:
            raise ToolReplayRefused("subframe recovery pre-manifest differs from recorded tape intent")
        generation, committed = _replay_txns(store, context)
        completed_result = call_root / "result.json"
        if completed_result.is_file() and not completed_result.is_symlink():
            result = ParallelResult.from_json(json.loads(completed_result.read_bytes()))
            if (result.call_id != call_id or result.admission != admission or
                    result.txn_order != tuple(item.txn_id for item, _ in _latest_records(store))):
                raise ToolReplayRefused("durable subframe result differs from its recovered transactions")
            tape.complete(call_id=raw_call_id, exit_code=0, observation=result.observation,
                          post_manifest_digest=manifest_digest(context.root),
                          env_identity=str(context.runtime.binding().get("environment_revision") or "disabled"))
            return result
        latest_intent = call_root / "intent.json"
        if latest_intent.is_file():
            on_disk = json.loads(latest_intent.read_bytes())
            if {key: value for key, value in on_disk.items() if key not in {"executions", "snapshot_manifests"}} != {
                    key: value for key, value in intent.items() if key not in {"executions", "snapshot_manifests"}}:
                raise ToolReplayRefused("subframe recovery intent changed on disk")
            if on_disk.get("snapshot_manifests", {}).get("g0") != intent.get("snapshot_manifests", {}).get("g0"):
                raise ToolReplayRefused("subframe initial snapshot manifest changed during recovery")
            intent = on_disk
    if admission.status != "ADMITTED":
        measured = ParallelMeasured(0.0, 0.0, 0.0, 0, 0.0)
        observation = _observation(admission, (), "NOT_RUN", "NOT_RUN", measured, request)
        result = ParallelResult(call_id, admission, (), (), "NOT_RUN", (), "NOT_RUN", observation, measured)
        _atomic_json(call_root / "result.json", result.to_json())
        tape.complete(call_id=raw_call_id, exit_code=0, observation=observation,
                      post_manifest_digest=manifest_digest(context.root),
                      env_identity=str(context.runtime.binding().get("environment_revision") or "disabled"))
        return result

    adopt_salvage = salvage_adopt_enabled()
    started = time.monotonic()
    host = host or HostClient(call_root, context.attempt)
    dependencies = _declared_dependencies(admission)
    units = {unit.id: unit for unit in request.units}
    outcomes: dict[str, UnitOutcome] = _load_outcomes(call_root) if record is not None else {}
    active: dict[str, dict] = {}
    unknown = {(pair.a, pair.b): pair for pair in admission.pairs if pair.status == "UNKNOWN"}
    for previous, _ in _latest_records(store):
        recovered = UnitOutcome(previous.unit_id, "COMMITTED" if previous.check_receipt_digest else
                                "COMMITTED_UNVERIFIED", None, None, previous.txn_id, (), True,
                                "PASS" if previous.check_receipt_digest else "NOT_RUN",
                                previous.usage, 0.0, 0.0)
        logged = outcomes.get(previous.unit_id)
        if logged is None:
            _record_outcome(call_root, recovered, budget, parent_steps)
            outcomes[previous.unit_id] = recovered
        elif logged.txn_id != previous.txn_id or logged.status != recovered.status:
            raise ToolReplayRefused(f"subframe outcome for {previous.unit_id} differs from transaction")
    for unit_id, outcome in outcomes.items():
        if unit_id not in units or (outcome.txn_id is not None and unit_id not in committed):
            raise ToolReplayRefused(f"subframe outcome for {unit_id} has no matching execution")

    while len(outcomes) < len(request.units):
        if context.deadline is not None and time.monotonic() >= context.deadline:
            for unit_id in active:
                host.cancel(unit_id)
            raise RecoveryRequired("RECOVERY_REQUIRED: subframe parent deadline exceeded")
        for unit_id in admission.order:
            if unit_id in outcomes or unit_id in active:
                continue
            failed = [parent for parent in dependencies[unit_id]
                      if parent in outcomes and parent not in committed]
            if failed:
                producer = sorted(failed)[0]
                outcome = UnitOutcome(unit_id, "CANCELLED", "PRODUCER_FAILED",
                                      producer, None, (), None, "NOT_RUN", _empty_usage(), 0.0, 0.0)
                _record_outcome(call_root, outcome, budget, parent_steps)
                outcomes[unit_id] = outcome
                context.runtime.journal.emit("parallel_unit_terminal", unit=unit_id,
                                             terminal="CANCELLED", reason="PRODUCER_FAILED")
                continue
            if not dependencies[unit_id].issubset(committed):
                continue
            existing = call_root / "units" / unit_id / "unit-spec.json"
            if existing.is_file():
                if __package__:
                    from .kernel_parallel_records import UnitSpec
                else:
                    from kernel_parallel_records import UnitSpec
                try:
                    spec = UnitSpec.from_json(json.loads(existing.read_bytes()))
                except (ValueError, OSError) as error:
                    raise ToolReplayRefused(f"subframe recovered spec is invalid for {unit_id}: {error}") from error
                snapshot_path = call_root / "snapshots" / f"g{spec.start_generation}"
                try:
                    snapshot = _StartSnapshot(snapshot_path, spec.start_generation)
                except (ParallelError, OSError) as error:
                    raise ToolReplayRefused(f"subframe recovered start snapshot is invalid for {unit_id}: {error}") from error
                expected_manifest = (record.pre_manifest_digest if spec.start_generation == 0 else
                                     next((txn.post_manifest_digest for txn, _ in _latest_records(store)
                                           if txn.post_generation == spec.start_generation), None))
                if (spec.start_generation > generation or snapshot.manifest_digest != spec.start_manifest_digest
                        or snapshot.manifest_digest != expected_manifest
                        or intent["snapshot_manifests"].get(f"g{spec.start_generation}") !=
                        snapshot.snapshot_manifest_digest):
                    raise ToolReplayRefused("subframe recovered start snapshot differs from its spec")
                try:
                    execution = next((ExecutionState.from_json(item) for item in intent["executions"]
                                      if item.get("unit_id") == unit_id), None)
                except (ValueError, TypeError, AttributeError) as error:
                    raise ToolReplayRefused(f"subframe recovered execution is invalid for {unit_id}: {error}") from error
                if execution is None or execution.execution_id != spec.execution_id or \
                        execution.spec_digest != spec.digest() or execution.run_dir != str(existing.parent):
                    raise ToolReplayRefused(f"subframe recovered execution identity or spec digest differs for {unit_id}")
                try:
                    expected_spec = project_unit(admission, request, unit_id, snapshot,
                                                 parent_seed=context.parent_seed,
                                                 policy_digest=intent["policy_digest"],
                                                 failure_projection="", requirements=context.requirements)
                except ValueError as error:
                    raise ToolReplayRefused(f"subframe recovered projection is invalid for {unit_id}: {error}") from error
                expected_spec = dataclasses.replace(expected_spec,
                                                    meter_calibration=spec.meter_calibration,
                                                    subframe_cap_calibration=spec.subframe_cap_calibration)
                if expected_spec != spec:
                    raise ToolReplayRefused(f"subframe recovered projection differs for {unit_id}")
                records = list((call_root / "host" / "units" / unit_id).glob("[0-9]*.json"))
                incarnation = max((int(path.stem) for path in records if path.stem.isdecimal()), default=0) + 1
            else:
                start_snapshot = _snapshot(context.root, call_root, generation)
                snapshot = start_snapshot
                snapshot_path = start_snapshot.root
                spec = project_unit(admission, request, unit_id, start_snapshot,
                                    parent_seed=context.parent_seed, policy_digest=intent["policy_digest"],
                                    failure_projection="", requirements=context.requirements)
                if __package__:
                    from .kernel_parallel_meter import _parent_meter_calibration
                else:
                    from kernel_parallel_meter import _parent_meter_calibration
                spec = dataclasses.replace(spec, meter_calibration=_parent_meter_calibration(context.runtime))
                spec = dataclasses.replace(spec, subframe_cap_calibration=cap_calibration)
                _record_start(call_root, intent, spec, start_snapshot)
                existing.parent.mkdir(parents=True, exist_ok=True)
                _atomic_json(existing, spec.to_json())
                incarnation = 1
            sealed_path = call_root / "proposals" / unit_id / "proposal.json"
            if record is not None and sealed_path.exists():
                if sealed_path.is_symlink() or incarnation <= 1:
                    raise ToolReplayRefused("recovered subframe proposal lacks a prior worker incarnation")
                sealed = _host_answer(host, "seal", unit_id,
                                      {"unit_id": unit_id, "incarnation": incarnation - 1},
                                      context.deadline).payload
                proposal, blobs = _proposal(Path(str(sealed["proposal_path"])),
                                            str(sealed["proposal_digest"]), units[unit_id],
                                            spec.execution_id, incarnation - 1)
                usage = _usage(call_root / "units" / unit_id)
                try:
                    outcome, generation, pairs = _accept_proposal(
                        context, request, admission, spec, proposal, blobs, call_root,
                        store, generation, usage, 0.0, 0.0)
                except ParallelError as error:
                    reason, _, detail = str(error).partition(": ")
                    outcome = _failure_outcome(unit_id, reason, detail or str(error),
                                               usage, 0.0, 0.0)
                else:
                    committed.add(unit_id)
                    for pair in pairs:
                        if pair.status == "UNKNOWN":
                            unknown[pair.a, pair.b] = pair
                _record_outcome(call_root, outcome, budget, parent_steps)
                outcomes[unit_id] = outcome
                context.runtime.journal.emit("parallel_unit_terminal", unit=unit_id,
                                             terminal=outcome.status, reason=outcome.reason)
                continue
            payload = {"unit_id": unit_id, "execution_id": spec.execution_id,
                       "spec_path": str(existing), "spec_digest": spec.digest(),
                       "run_dir": str(existing.parent), "incarnation": incarnation,
                       "start_snapshot_path": str(snapshot_path),
                       "start_manifest_digest": spec.start_manifest_digest,
                       "snapshot_manifest_digest": snapshot.snapshot_manifest_digest}
            try:
                _host_answer(host, "launch", unit_id, payload, context.deadline)
            except ParallelError as error:
                host.cancel(unit_id)
                outcome = _failure_outcome(unit_id, "ENGINE_ERROR", str(error),
                                           _empty_usage(), 0.0, 0.0)
                _record_outcome(call_root, outcome, budget, parent_steps)
                outcomes[unit_id] = outcome
                continue
            active[unit_id] = {"spec": spec, "incarnation": incarnation,
                               "launched": time.monotonic(), "running": None,
                               "started_emitted": False}
        if not active and len(outcomes) < len(request.units):
            raise ParallelError("subframe dependency scheduler has no runnable or active subframes")
        exited = []
        for unit_id in sorted(tuple(active)):
            info = active[unit_id]
            try:
                status = _host_answer(host, "status", unit_id, {"unit_id": unit_id}, context.deadline).payload
            except ParallelError as error:
                host.cancel(unit_id)
                outcome = _failure_outcome(unit_id, "ENGINE_ERROR", str(error),
                                           _usage(call_root / "units" / unit_id),
                                           time.monotonic() - info["launched"], 0.0)
                _record_outcome(call_root, outcome, budget, parent_steps)
                outcomes[unit_id] = outcome
                del active[unit_id]
                continue
            running_since = status.get("running_since")
            finished_epoch = status.get("finished_epoch")
            slot = status.get("slot")
            slot_wait = status.get("slot_wait_seconds")
            if (running_since is not None and
                    (not isinstance(running_since, (int, float)) or isinstance(running_since, bool)
                     or not math.isfinite(running_since))):
                raise ParallelError(f"subframe host status has invalid running_since for {unit_id}")
            if (not isinstance(slot_wait, (int, float)) or isinstance(slot_wait, bool)
                    or not math.isfinite(slot_wait) or slot_wait < 0):
                raise ParallelError(f"subframe host status has invalid slot_wait_seconds for {unit_id}")
            if running_since is not None and (type(slot) is not int or slot < 0):
                raise ParallelError(f"subframe host status has invalid leased slot for {unit_id}")
            if finished_epoch is not None and (type(finished_epoch) not in (int, float)
                                               or not math.isfinite(finished_epoch)):
                raise ParallelError(f"subframe host status has invalid finished_epoch for {unit_id}")
            info["running"] = running_since
            info["slot_wait"] = float(slot_wait)
            if running_since is not None and not info["started_emitted"]:
                context.runtime.journal.emit("parallel_unit_started", unit=unit_id,
                                             execution_id=info["spec"].execution_id,
                                             start_generation=info["spec"].start_generation,
                                             slot=slot)
                info["started_emitted"] = True
            if status.get("state") == "RUNNING":
                if running_since is None:
                    raise ParallelError(f"subframe host RUNNING status lacks running_since for {unit_id}")
            if status.get("state") != "EXITED":
                continue
            exited_seq = status.get("exited_seq")
            if type(exited_seq) is not int or exited_seq < 1:
                raise ParallelError(f"subframe host EXITED status lacks valid exited_seq for {unit_id}")
            if running_since is not None and (finished_epoch is None or finished_epoch < running_since):
                raise ParallelError(f"subframe host EXITED status lacks valid finish time for {unit_id}")
            exited.append((exited_seq, unit_id, status))
        if exited and len({item[0] for item in exited}) != len(exited):
            raise ParallelError("subframe host returned duplicate exited_seq values")
        for _, unit_id, status in sorted(exited):
            info = active[unit_id]
            terminal = status.get("terminal")
            wall = (status["finished_epoch"] - info["running"]) if info["running"] is not None else 0.0
            if info["running"] is not None:
                _atomic_json(call_root / "intervals" / f"{unit_id}-{info['incarnation']}.json",
                             {"running_since": info["running"],
                              "finished_epoch": status["finished_epoch"]})
            queue_wait = info["slot_wait"]
            usage = _usage(call_root / "units" / unit_id)
            if terminal == "SUBMITTED":
                try:
                    sealed = _host_answer(host, "seal", unit_id,
                                          {"unit_id": unit_id, "incarnation": info["incarnation"]},
                                          context.deadline).payload
                    proposal, blobs = _proposal(Path(str(sealed["proposal_path"])),
                                                str(sealed["proposal_digest"]), units[unit_id],
                                                info["spec"].execution_id, info["incarnation"])
                    outcome, generation, pairs = _accept_proposal(context, request, admission,
                                                                  info["spec"], proposal, blobs,
                                                                  call_root, store, generation, usage,
                                                                  wall, queue_wait)
                    for pair in pairs:
                        if pair.status == "UNKNOWN":
                            unknown[pair.a, pair.b] = pair
                    committed.add(unit_id)
                except (ParallelError, OSError, ValueError, RuntimeError) as error:
                    reason, _, detail = str(error).partition(": ")
                    check = error.check_status if isinstance(error, CompositionCheckFailed) else "NOT_RUN"
                    outcome = _failure_outcome(unit_id, reason, detail or str(error), usage,
                                               wall, queue_wait, check=check)
                    # A submission the parent did not accept leaves its edits in the same
                    # private directory; keep them. It already failed the gates, so it is
                    # never offered to them a second time.
                    salvage = preserve_unit_candidate(call_root, unit_id,
                                                      start_generation=info["spec"].start_generation)
                    if salvage is not None:
                        _journal_salvage(context, unit_id, "SUBMITTED", salvage)
            else:
                if terminal == "SUBFRAME_BLOCKED":
                    try:
                        outcome = _blocked_outcome(call_root, unit_id, usage, wall, queue_wait)
                    except ParallelError as error:
                        outcome = _failure_outcome(unit_id, "ENGINE_ERROR", str(error), usage,
                                                   wall, queue_wait)
                else:
                    reason = str(terminal or "ENGINE_ERROR")
                    detail = reason
                    if reason == "SUBFRAME_CONTEXT_TOO_LARGE" and usage.initial_frame_tokens is not None:
                        detail = (f"first_frame_tokens={usage.initial_frame_tokens}; "
                                  f"cap={info['spec'].limits.unit_request_tokens}")
                    outcome = _failure_outcome(unit_id, reason, detail, usage, wall, queue_wait)
                # An unsubmitted unit's edits live only in its candidate bundle.
                salvage = preserve_unit_candidate(call_root, unit_id,
                                                  start_generation=info["spec"].start_generation)
                if salvage is not None and adopt_salvage:
                    # The salvaged edits earn publication only through the gates a
                    # submitted proposal passes; a refusal leaves the unit's outcome as is.
                    stopped = str(terminal or "ENGINE_ERROR")
                    adopted, refusal = None, None
                    salvage_files = tuple(salvage.get("files") or ())
                    try:
                        proposal, blobs = _salvage_proposal(call_root, units[unit_id], info["spec"],
                                                            info["incarnation"], salvage)
                        salvage_files = tuple(entry.path for entry in proposal.entries)
                        adopted, generation, pairs = _accept_proposal(
                            context, request, admission, info["spec"], proposal, blobs,
                            call_root, store, generation, usage, wall, queue_wait)
                    except (ParallelError, OSError, ValueError, RuntimeError) as error:
                        refusal = str(error) or type(error).__name__
                    else:
                        for pair in pairs:
                            if pair.status == "UNKNOWN":
                                unknown[pair.a, pair.b] = pair
                        committed.add(unit_id)
                        outcome = adopted
                    adoption = _record_salvage_adoption(call_root, unit_id, stopped, salvage,
                                                        salvage_files, adopted, refusal)
                    _journal_salvage(context, unit_id, stopped, salvage,
                                     adoption=adoption["adoption"], adoption_reason=adoption["reason"],
                                     adopted_files=adoption["files"])
                elif salvage is not None:
                    _journal_salvage(context, unit_id, str(terminal or "ENGINE_ERROR"), salvage)
            _record_outcome(call_root, outcome, budget, parent_steps)
            outcomes[unit_id] = outcome
            context.runtime.journal.emit("parallel_unit_terminal", unit=unit_id,
                                         terminal=outcome.status, reason=outcome.reason)
            del active[unit_id]
        if active:
            time.sleep(0.2)

    unknown_pairs = tuple(unknown[key] for key in sorted(unknown))
    final_check, integration = _final_check(context, request, call_root, unknown_pairs, committed)
    final_receipt = _final_check_receipt(call_root)
    if any(outcome.status not in {"COMMITTED", "COMMITTED_UNVERIFIED"}
           for outcome in outcomes.values()):
        integration = "FAILED"
    elif integration == "NOT_RUN" and final_check == "NOT_RUN" and outcomes:
        # Committed work with no check that ran cannot be called integration-NOT_RUN.
        integration = "UNVERIFIED"
    context.runtime.journal.emit("parallel_final", final_check=final_check,
                                 integration=integration,
                                 unknown_pairs=[pair.to_json() for pair in unknown_pairs])
    result = _result(context, call_root, request, admission, outcomes, store, unknown_pairs,
                      started, _peak_active_subframes(call_root), final_check, integration,
                      final_check_receipt=final_receipt)
    tape.complete(call_id=raw_call_id, exit_code=0, observation=result.observation,
                  post_manifest_digest=manifest_digest(context.root),
                  env_identity=str(context.runtime.binding().get("environment_revision") or "disabled"))
    return result
