"""Bounded public context projection for model-authored executable checks."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "stateframe-authored-check-context/1"
AUTHORED_CHECK_CONTEXT_ATTR = "_authored_check_context"
AUTHORED_CHECK_SCRATCH_AUTHORITY_ATTR = "_authored_check_scratch_authority"
_FORBIDDEN_PARTS = frozenset({
    ".git", "backups", "evaluator", "evaluators", "hidden_tests", "reference_fix",
    "reports", "journals", "operator", "belt_queue", "measurement-runs",
})
_IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)|import\s+([\w.]+))", re.MULTILINE)


class ContextError(ValueError):
    """Public-context projection refusal."""


def read_runtime_authored_check_context(runtime: object) -> Mapping[str, object]:
    """Read the host context through the same attribute written below."""
    value = getattr(runtime, AUTHORED_CHECK_CONTEXT_ATTR, None)
    return value if isinstance(value, Mapping) else {}


def write_runtime_authored_check_context(
        runtime: object, bundle: object, ledger: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Mint run-scoped scratch authority from host bundle and ledger facts."""
    if not isinstance(ledger, Mapping):
        return None
    existing = read_runtime_authored_check_context(runtime)
    existing_authority = existing.get("scratch_authority")
    if (existing.get("producer") != "host_bundle_requirements_ledger"
            and str(existing.get("public_text", "") or "").strip()
            and isinstance(existing_authority, Mapping)
            and _text(existing_authority.get("authority_ref"))):
        return existing
    requirements = _seq(ledger.get("requirements"))
    if not requirements or bundle is None:
        return None

    owner = getattr(runtime, "owner", None)
    root_value = getattr(owner, "root", getattr(runtime, "root", None))
    if root_value is None or not str(root_value).strip():
        return None
    try:
        workspace = str(Path(str(root_value)).resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None

    ticket_sources = [
        item for item in _seq(ledger.get("sources"))
        if isinstance(item, Mapping) and str(item.get("kind", "")) == "ticket"
    ]
    ticket_source = ticket_sources[0] if len(ticket_sources) == 1 else {}
    clauses: list[str] = []
    ticket_text = _text(ticket_source.get("text")) if isinstance(ticket_source, Mapping) else ""
    if ticket_text:
        clauses.append(ticket_text)
    for requirement in requirements:
        if isinstance(requirement, Mapping):
            clause = _text(requirement.get("exact_clause"))
            if clause and clause not in clauses:
                clauses.append(clause)
    specs = getattr(bundle, "specs", ())
    for spec in _seq(specs):
        clause_value = spec.get("clause", "") if isinstance(spec, Mapping) else getattr(
            spec, "clause", "")
        clause = _text(clause_value)
        if clause and clause not in clauses:
            clauses.append(clause)
    public_text = "\n".join(clauses)
    if not public_text.strip():
        return None

    binding = getattr(runtime, "binding", None)
    try:
        bound = binding() if callable(binding) else binding
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        bound = None
    candidate = None
    if isinstance(bound, Mapping):
        candidate = next((bound.get(key) for key in (
            "product_revision", "current_revision", "revision")
                          if bound.get(key) not in (None, "")), None)
    elif bound not in (None, ""):
        candidate = bound
    if candidate in (None, ""):
        candidate = next((getattr(runtime, key, None) for key in (
            "product_revision", "current_revision", "revision")
                          if getattr(runtime, key, None) not in (None, "")), None)
    if candidate in (None, ""):
        candidate = ledger.get("current_revision")
    candidate_revision = _text(candidate) or "unknown"
    parent_revision = _text(getattr(runtime, "parent_revision", None)) or candidate_revision

    task_id = _text(getattr(bundle, "task_id", "")) or _text(
        ledger.get("ticket_source_id")) or _text(getattr(owner, "task_id", "")) or "host-run"
    run_identity = _digest({
        "workspace": workspace,
        "task_id": task_id,
        "ticket_source_id": ledger.get("ticket_source_id", ""),
        "ticket_sha256": ticket_source.get("sha256", "") if isinstance(ticket_source, Mapping) else "",
        "revision": candidate_revision,
    })
    path = f"scratch/authored-check/{run_identity[7:23]}/check.py"
    authority = {
        "schema": "stateframe-authored-check-scratch-authority/1",
        "authority_ref": _digest({"workspace": workspace, "run": run_identity, "path": path}),
        "scope": "scratch_test_path",
        "workspace_root": workspace,
        "run_identity": run_identity,
        "allowed_paths": [path],
    }
    source_id = _text(ticket_source.get("id")) if isinstance(ticket_source, Mapping) else ""
    source_digest = _text(ticket_source.get("sha256")) if isinstance(ticket_source, Mapping) else ""
    context: dict[str, object] = {
        "schema": SCHEMA,
        "producer": "host_bundle_requirements_ledger",
        "task_id": task_id,
        "run_identity": run_identity,
        "public_text": public_text,
        "parent_revision": parent_revision,
        "candidate_revision": candidate_revision,
        "scratch_test_path": path,
        "authorized_paths": [path],
        "scratch_authority": authority,
        "source_unit_handle": _digest({"source_id": source_id, "source_digest": source_digest}),
    }
    if isinstance(ticket_source, Mapping) and ticket_source:
        context["public_source_span"] = {
            "source_unit_handle": context["source_unit_handle"],
            "path": _text(ticket_source.get("path")),
            "source_id": source_id,
            "source_sha256": source_digest,
            "text": ticket_text,
        }
    setattr(runtime, AUTHORED_CHECK_SCRATCH_AUTHORITY_ATTR, authority)
    setattr(runtime, AUTHORED_CHECK_CONTEXT_ATTR, context)
    ticket_span: object = ticket_text
    ticket_path = _text(ticket_source.get("path")) if isinstance(ticket_source, Mapping) else ""
    if ticket_path:
        ticket_span = {"path": ticket_path, "text": ticket_text,
                       "provenance": "host_requirements_ledger"}
    public_context = build_public_check_context({
        "ticket_span": ticket_span,
        "runner": {
            "runner_id": "stateframe-authored-check",
            "cwd": workspace,
            "selector": path,
            "scratch_test_path": path,
        },
    })
    setattr(runtime, "_authored_check_public_context", public_context)
    return context


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else str(value or "").strip()


def _digest(value: object) -> str:
    if isinstance(value, bytes):
        data = value
    else:
        data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _seq(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _path(value: object) -> str:
    raw = _text(value).replace("\\", "/")
    parts = raw.split("/")
    if not raw or raw.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise ContextError(f"invalid public context path: {raw!r}")
    if any(part.casefold() in _FORBIDDEN_PARTS for part in parts):
        raise ContextError("public context path is outside the solver allowlist")
    return raw


def _span(path: str, source: str, *, start: int = 1, end: int | None = None,
          kind: str, provenance: str) -> dict[str, object]:
    lines = source.splitlines()
    start = max(1, start)
    end = max(start, end if end is not None else len(lines))
    text = "\n".join(lines[start - 1:end])
    return {
        "path": path, "start_line": start, "end_line": end,
        "text": text, "sha256": _digest(text.encode("utf-8")),
        "kind": kind, "provenance": provenance,
    }


def _files(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for raw_path, source in value.items():
        try:
            result[_path(raw_path)] = source.decode("utf-8") if isinstance(source, bytes) else str(source)
        except (ContextError, UnicodeDecodeError):
            continue
    return result


@dataclass(frozen=True)
class PublicCheckContext:
    schema: str
    ticket_span: Mapping[str, object]
    target: Mapping[str, object] | None
    imports: tuple[Mapping[str, object], ...]
    fixtures: tuple[Mapping[str, object], ...]
    dependencies: tuple[Mapping[str, object], ...]
    runner: Mapping[str, object]
    missing: tuple[Mapping[str, object], ...]
    omitted: tuple[Mapping[str, object], ...]
    manifest_digest: str
    provenance: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "ticket_span": dict(self.ticket_span),
            "target": dict(self.target) if isinstance(self.target, Mapping) else None,
            "imports": [dict(item) for item in self.imports],
            "fixtures": [dict(item) for item in self.fixtures],
            "dependencies": [dict(item) for item in self.dependencies],
            "runner": dict(self.runner),
            "missing": [dict(item) for item in self.missing],
            "omitted": [dict(item) for item in self.omitted],
            "manifest_digest": self.manifest_digest,
            "provenance": dict(self.provenance),
        }

    def wire_projection(self) -> dict[str, object]:
        """Lossless bounded delivery package; no expected answer is generated."""
        return self.as_dict()


def _target_record(target: object, files: Mapping[str, str], missing: list[dict[str, object]]) -> dict[str, object] | None:
    if not isinstance(target, Mapping):
        missing.append({"kind": "target", "reason": "target_declaration_unavailable"})
        return None
    try:
        path = _path(target.get("path"))
    except ContextError:
        missing.append({"kind": "target", "reason": "target_path_unavailable"})
        return None
    source = files.get(path)
    if source is None:
        missing.append({"kind": "target", "path": path, "reason": "target_source_unavailable"})
        return None
    symbol = _text(target.get("symbol", target.get("name")))
    if not symbol:
        missing.append({"kind": "target", "path": path,
                        "reason": "callable_identity_unavailable"})
        return None
    start = target.get("start_line", 1)
    end = target.get("end_line")
    start = start if type(start) is int else 1
    end = end if type(end) is int else None
    span = _span(path, source, start=start, end=end, kind="target",
                 provenance=_text(target.get("provenance")) or "host_symbol_closure")
    declaration = target.get("declaration")
    if not isinstance(declaration, str) or not declaration.strip():
        declaration = span["text"]
    span["symbol"] = symbol
    span["declaration"] = declaration
    type_declaration = target.get("type_declaration")
    if isinstance(type_declaration, str) and type_declaration.strip():
        span["type_declaration"] = type_declaration
    span["language"] = _text(target.get("language")) or path.rsplit(".", 1)[-1]
    return span


def _import_spans(target: Mapping[str, object] | None, files: Mapping[str, str],
                  missing: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    if not isinstance(target, Mapping):
        return ()
    path = _text(target.get("path"))
    # Imports belong to the complete host source unit.  The target span may be
    # only the callable declaration and therefore cannot be used as the import
    # closure source.
    source = files.get(path, _text(target.get("text", target.get("declaration"))))
    imports: list[dict[str, object]] = []
    for match in _IMPORT_RE.finditer(source):
        module = _text(match.group(1) or match.group(2))
        if not module:
            continue
        line = source[:match.start()].count("\n") + 1
        line_start = source.rfind("\n", 0, match.start()) + 1
        line_end = source.find("\n", match.end())
        line_end = len(source) if line_end < 0 else line_end
        import_text = source[line_start:line_end]
        resolved = next((candidate for candidate in files if candidate.replace("/", ".").rsplit(".", 1)[0] == module), None)
        item = {
            "kind": "target_import", "module": module, "path": path,
            "resolved_path": resolved or "", "line": line,
            "start_line": line, "end_line": line, "text": import_text,
            "sha256": _digest(import_text.encode("utf-8")),
            "source_sha256": _digest(source.encode("utf-8")),
            "provenance": "target_import",
        }
        imports.append(item)
        if resolved is None:
            missing.append({"kind": "import", "module": module, "reason": "import_source_unavailable"})
    return tuple(imports)


def _fixture_spans(value: object, files: Mapping[str, str], missing: list[dict[str, object]],
                   omitted: list[dict[str, object]], limit: int) -> tuple[dict[str, object], ...]:
    fixtures: list[dict[str, object]] = []
    for raw in _seq(value):
        if len(fixtures) >= limit:
            omitted.append({"kind": "fixture", "reason": "bounded_fixture_limit"})
            break
        if not isinstance(raw, Mapping):
            continue
        try:
            path = _path(raw.get("path"))
        except ContextError:
            missing.append({"kind": "fixture", "reason": "fixture_path_unavailable"})
            continue
        source = files.get(path)
        if source is None:
            missing.append({"kind": "fixture", "path": path, "reason": "fixture_source_unavailable"})
            continue
        fixture = _span(path, source, start=raw.get("start_line", 1) if type(raw.get("start_line", 1)) is int else 1,
                        end=raw.get("end_line") if type(raw.get("end_line")) is int else None,
                        kind="fixture", provenance=_text(raw.get("provenance")) or "host_fixture_closure")
        fixture["selector"] = _text(raw.get("selector"))
        fixture["dependencies"] = list(_text(item) for item in _seq(raw.get("dependencies")) if _text(item))
        fixtures.append(fixture)
    return tuple(fixtures)


def build_public_check_context(request: Mapping[str, object], *, max_spans: int = 12,
                               max_bytes: int = 24_000) -> PublicCheckContext:
    """Project only public, host-discovered evidence needed to write a check."""
    files = _files(request.get("public_files", request.get("files", {})))
    missing: list[dict[str, object]] = []
    omitted: list[dict[str, object]] = []
    ticket = request.get("ticket_span", request.get("ticket"))
    if isinstance(ticket, Mapping):
        ticket_span = dict(ticket)
        if "path" in ticket_span:
            try:
                ticket_span["path"] = _path(ticket_span["path"])
            except ContextError:
                missing.append({"kind": "ticket", "reason": "ticket_path_unavailable"})
        ticket_span.setdefault("provenance", "host_ticket")
        ticket_span.setdefault("sha256", _digest(ticket_span.get("text", "")))
    else:
        text = _text(ticket)
        ticket_span = {"text": text, "sha256": _digest(text), "provenance": "host_ticket"}
        if not text:
            missing.append({"kind": "ticket", "reason": "ticket_text_unavailable"})
    target_candidates = _seq(request.get("targets"))
    if target_candidates:
        names = [_text(item.get("symbol", item.get("name"))) for item in target_candidates
                 if isinstance(item, Mapping)]
        duplicates = {name for name in names if name and names.count(name) > 1}
        if duplicates:
            missing.append({"kind": "target", "reason": "ambiguous_symbol_collision",
                            "symbols": sorted(duplicates)})
            target = None
        else:
            target = _target_record(target_candidates[0], files, missing)
    else:
        target = _target_record(request.get("target", request.get("discovered_target")), files, missing)
    imports = _import_spans(target, files, missing)
    fixtures = _fixture_spans(request.get("fixtures", request.get("fixture_plan", ())), files, missing,
                               omitted, max(1, max_spans // 3))
    dependencies: list[dict[str, object]] = []
    dependency_map: dict[object, object] = {}
    requested_dependencies = request.get("dependencies", {})
    if isinstance(requested_dependencies, Mapping):
        dependency_map.update(requested_dependencies)
    for fixture in fixtures:
        for dependency in _seq(fixture.get("dependencies")):
            dependency_path = _text(dependency)
            if dependency_path and dependency_path not in dependency_map:
                dependency_map[dependency_path] = files.get(dependency_path)
    if isinstance(dependency_map, Mapping):
        for raw_path, raw_source in dependency_map.items():
            try:
                path = _path(raw_path)
            except ContextError:
                missing.append({"kind": "dependency", "reason": "dependency_path_unavailable"})
                continue
            source = files.get(path, raw_source if isinstance(raw_source, str) else None)
            if source is None:
                missing.append({"kind": "dependency", "path": path, "reason": "dependency_source_unavailable"})
                continue
            dependencies.append(_span(path, source, kind="dependency", provenance="host_import_closure"))
    runner_raw = request.get("runner", request.get("native_runner", {}))
    runner = dict(runner_raw) if isinstance(runner_raw, Mapping) else {}
    if not _text(runner.get("runner_id")):
        missing.append({"kind": "runner", "reason": "runner_unavailable"})
    if not _text(runner.get("cwd")):
        runner["cwd"] = "."
    spans: list[Mapping[str, object]] = [*imports, *fixtures, *dependencies]
    if target is not None:
        spans.insert(0, target)
    selected: list[Mapping[str, object]] = []
    total = len(json.dumps(ticket_span, ensure_ascii=False).encode("utf-8"))
    for item in spans:
        encoded = len(json.dumps(item, ensure_ascii=False, default=str).encode("utf-8"))
        if len(selected) >= max_spans or total + encoded > max_bytes:
            omitted.append({"kind": str(item.get("kind", "source")),
                            "path": _text(item.get("path")), "reason": "bounded_context_limit"})
            continue
        selected.append(item)
        total += encoded
    selected_imports = tuple(item for item in selected if item.get("kind") == "target_import")
    selected_fixtures = tuple(item for item in selected if item.get("kind") == "fixture")
    selected_dependencies = tuple(item for item in selected if item.get("kind") == "dependency")
    manifest_digest = _digest({"ticket": ticket_span, "spans": selected, "runner": runner})
    provenance = {"source": "public_allowlist", "manifest_digest": manifest_digest,
                  "max_spans": max_spans, "max_bytes": max_bytes}
    return PublicCheckContext(
        schema=SCHEMA, ticket_span=ticket_span, target=target,
        imports=selected_imports, fixtures=selected_fixtures,
        dependencies=selected_dependencies, runner=runner,
        missing=tuple(missing), omitted=tuple(omitted),
        manifest_digest=manifest_digest, provenance=provenance,
    )


def public_check_context(request: Mapping[str, object], **kwargs: object) -> dict[str, object]:
    return build_public_check_context(request, **kwargs).as_dict()


def serialize_public_check_context(context: PublicCheckContext | Mapping[str, object]) -> bytes:
    value = context.as_dict() if isinstance(context, PublicCheckContext) else dict(context)
    if value.get("schema") != SCHEMA:
        raise ContextError("public check context schema is not host-authored")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def deliver_public_check_context(messages: list[dict[str, object]],
                                 context: PublicCheckContext | Mapping[str, object]) -> dict[str, object]:
    """Place the bounded projection in the existing protected evidence stream.

    This is deliberately a normal request message, not a sidecar or steering
    instruction.  Re-delivery replaces the same digest-bearing message so a
    reused capture cannot accumulate stale context copies.
    """
    if not isinstance(messages, list):
        raise TypeError("request messages must be a list")
    wire = serialize_public_check_context(context)
    digest = _digest(wire)
    marker = "[Public authored-check context]\n"
    messages[:] = [
        item for item in messages
        if not (isinstance(item, Mapping)
                and item.get("authored_check_context") is True)
    ]
    messages.append({
        "role": "user",
        "section": "stage_evidence",
        "authored_check_context": True,
        "context_digest": digest,
        "content": marker + wire.decode("utf-8"),
    })
    return {"schema": SCHEMA, "context_digest": digest, "wire_bytes": len(wire),
            "delivered": True}


__all__ = [
    "AUTHORED_CHECK_CONTEXT_ATTR", "AUTHORED_CHECK_SCRATCH_AUTHORITY_ATTR",
    "ContextError", "PublicCheckContext", "SCHEMA", "build_public_check_context",
    "deliver_public_check_context", "public_check_context",
    "read_runtime_authored_check_context", "serialize_public_check_context",
    "write_runtime_authored_check_context",
]
