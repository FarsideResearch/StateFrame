"""Pure deterministic source extractors for the world-knowledge cache.

The extractor deliberately returns observations, not conclusions.  In particular,
``static_references_symbol`` means that a call *may* exercise a symbol; only an
explicit runtime witness could justify a stronger coverage fact.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

EXTRACTOR_NAME = "python_ast"
EXTRACTOR_VERSION = "1"
LANGUAGE_NEUTRAL_EXTRACTOR = "language_neutral"
LANGUAGE_NEUTRAL_VERSION = "1"
MAX_OBSERVED_LINES = 10_000_000
_PYTHON_SUFFIXES = frozenset({".py", ".pyi", ".pyw"})
_DATA_SUFFIXES = frozenset({
    ".csv", ".json", ".jsonl", ".ndjson", ".toml", ".tsv", ".xml", ".yaml", ".yml",
})


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def normalize_path(path: str) -> str:
    """Normalize a sandbox-relative path and reject paths escaping the sandbox."""

    raw = str(path).replace("\\", "/").strip()
    if (
        not raw
        or raw.startswith("/")
        or (len(raw) >= 2 and raw[1] == ":")
    ):
        raise ValueError("path must be a non-empty sandbox-relative path")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("path must not escape the sandbox")
    return "/".join(parts)


def source_digest(content: str) -> str:
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def bounded_line_count(content: str) -> int:
    """Count line boundaries without allocating a whole-file line list."""
    if not content:
        return 0
    boundaries = content.count("\n") + content.count("\r") - content.count("\r\n")
    count = boundaries if content.endswith(("\n", "\r")) else boundaries + 1
    return min(MAX_OBSERVED_LINES, count)


def source_kind(path: str) -> str:
    """Classify source without pretending every language is Python."""
    suffix = PurePosixPath(normalize_path(path)).suffix.lower()
    if suffix in _PYTHON_SUFFIXES:
        return "python"
    if suffix in _DATA_SUFFIXES:
        return "data"
    return "text"


def source_extractor_version(path: str) -> str:
    """Return the stable version identity for every extractor used on a path."""
    versions = {f"{EXTRACTOR_NAME}/{EXTRACTOR_VERSION}"}
    if source_kind(path) != "python":
        versions.add(f"{LANGUAGE_NEUTRAL_EXTRACTOR}/{LANGUAGE_NEUTRAL_VERSION}")
    return ",".join(sorted(versions))


def subject_ref(path: str, subject: str) -> str:
    return f"env://{normalize_path(path)}#{subject}"


@dataclass(frozen=True)
class ExtractedFact:
    """One AST/receipt observation before store-level provenance is attached."""

    path: str
    predicate: str
    subject_ref: str
    value: Mapping[str, object]
    spans: tuple[tuple[int, int], ...] = ()
    dependencies: tuple[str, ...] = ()
    dependencies_complete: bool = True
    extractor: str = f"{EXTRACTOR_NAME}/{EXTRACTOR_VERSION}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_path(self.path))
        object.__setattr__(self, "predicate", str(self.predicate).strip())
        object.__setattr__(self, "subject_ref", str(self.subject_ref).strip())
        object.__setattr__(self, "value", dict(self.value))
        object.__setattr__(
            self,
            "spans",
            tuple((int(start), int(end)) for start, end in self.spans),
        )
        deps = {normalize_dependency(dep) for dep in self.dependencies if str(dep).strip()}
        deps.add(self.path)
        object.__setattr__(self, "dependencies", tuple(sorted(deps)))

    @property
    def value_digest(self) -> str:
        return _digest(self.value)


def normalize_dependency(dependency: str) -> str:
    """Normalize a dependency hint without rejecting external module names."""

    raw = str(dependency).replace("\\", "/").strip()
    if raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise ValueError("dependency must be sandbox-relative or a module name")
    if any(part == ".." for part in raw.split("/")):
        raise ValueError("dependency must not escape the sandbox")
    return raw.strip("./") if raw else raw


def _span(node: ast.AST) -> tuple[int, int]:
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    return (start, max(start, end))


def _expression(node: ast.AST | None, source: str) -> str:
    if node is None:
        return ""
    segment = ast.get_source_segment(source, node)
    if segment:
        return segment.strip()
    try:
        return ast.unparse(node).strip()
    except (AttributeError, TypeError):
        return ""


def _annotation(node: ast.AST | None, source: str) -> str:
    text = _expression(node, source)
    return f": {text}" if text else ""


def _arg_text(arg: ast.arg, default: ast.AST | None, source: str) -> str:
    text = arg.arg + _annotation(arg.annotation, source)
    if default is not None:
        text += " = " + _expression(default, source)
    return text


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> str:
    """Render all argument markers, annotations, defaults, and return type."""

    args = node.args
    positional = [*args.posonlyargs, *args.args]
    default_start = len(positional) - len(args.defaults)
    rendered = [
        _arg_text(
            arg,
            args.defaults[index - default_start] if index >= default_start else None,
            source,
        )
        for index, arg in enumerate(positional)
    ]
    if args.posonlyargs:
        rendered.insert(len(args.posonlyargs), "/")
    if args.vararg is not None:
        rendered.append("*" + _arg_text(args.vararg, None, source))
    elif args.kwonlyargs:
        rendered.append("*")
    rendered.extend(
        _arg_text(arg, default, source)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True)
    )
    if args.kwarg is not None:
        rendered.append("**" + _arg_text(args.kwarg, None, source))
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    result = f"{prefix} {node.name}({', '.join(rendered)})"
    return_annotation = _annotation(node.returns, source)
    return result + (f" -> {return_annotation[2:]}" if return_annotation else "")


def _class_signature(node: ast.ClassDef, source: str) -> str:
    bases = [_expression(base, source) for base in node.bases]
    keywords = [
        f"{keyword.arg}={_expression(keyword.value, source)}"
        if keyword.arg
        else "**" + _expression(keyword.value, source)
        for keyword in node.keywords
    ]
    args = ", ".join(item for item in (*bases, *keywords) if item)
    return f"class {node.name}" + (f"({args})" if args else "")


def _callee(node: ast.AST) -> tuple[str, str] | None:
    if isinstance(node, ast.Name):
        return (node.id, node.id)
    if not isinstance(node, ast.Attribute):
        return None
    pieces: list[str] = [node.attr]
    value: ast.AST = node.value
    while isinstance(value, ast.Attribute):
        pieces.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        pieces.append(value.id)
        pieces.reverse()
        return (pieces[-1], ".".join(pieces))
    return (node.attr, node.attr)


def _module_candidates(path: str, module: str, level: int = 0) -> tuple[str, ...]:
    module = module.strip()
    if not module:
        return ()
    module_parts = tuple(part for part in module.split(".") if part)
    if level:
        base = PurePosixPath(path).parent
        for _ in range(max(level - 1, 0)):
            base = base.parent
        stem_parts = (*base.parts, *module_parts)
    else:
        stem_parts = tuple(module_parts)
    stem = "/".join(stem_parts)
    return (f"{stem}.py", f"{stem}/__init__.py")


class _PythonExtractor(ast.NodeVisitor):
    def __init__(self, path: str, source: str) -> None:
        self.path = path
        self.source = source
        self.facts: list[ExtractedFact] = []
        self.stack: list[str] = []
        self.function_nodes: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        self.function_calls: dict[str, set[str]] = {}
        self.test_functions: set[str] = set()
        self.imported_symbols: dict[str, str] = {}
        self.imported_qualified_symbols: dict[str, str] = {}
        self.import_dependencies: set[str] = {path}
        self.unknown_dependencies = False

    @property
    def enclosing(self) -> str:
        return ".".join(self.stack)

    def add(
        self,
        predicate: str,
        subject: str,
        value: Mapping[str, object],
        node: ast.AST | None = None,
        *,
        spans: tuple[tuple[int, int], ...] | None = None,
        dependencies: Sequence[str] = (),
        dependencies_complete: bool = True,
    ) -> None:
        self.facts.append(
            ExtractedFact(
                path=self.path,
                predicate=predicate,
                subject_ref=subject_ref(self.path, subject),
                value=value,
                spans=spans if spans is not None else (_span(node),) if node else (),
                dependencies=tuple(dependencies),
                dependencies_complete=dependencies_complete,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = ".".join((*self.stack, node.name))
        self.add(
            "symbol_defined",
            qualname,
            {
                "name": node.name,
                "qualname": qualname,
                "kind": "class",
                "signature": _class_signature(node, self.source),
            },
            node,
        )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        qualname = ".".join((*self.stack, node.name))
        kind = "method" if self.stack else "function"
        signature = function_signature(node, self.source)
        self.function_nodes[qualname] = node
        self.function_calls.setdefault(qualname, set())
        self.add(
            "symbol_defined",
            qualname,
            {
                "name": node.name,
                "qualname": qualname,
                "kind": kind,
                "async": isinstance(node, ast.AsyncFunctionDef),
                "signature": signature,
            },
            node,
        )
        if node.name.startswith("test"):
            self.test_functions.add(qualname)
            self.add(
                "test_defined",
                qualname,
                {
                    "name": node.name,
                    "qualname": qualname,
                    "kind": kind,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "node_id": f"{self.path}::{qualname}",
                },
                node,
            )
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Import(self, node: ast.Import) -> None:
        for index, alias in enumerate(node.names):
            module = alias.name
            alias_name = alias.asname or module.split(".")[0]
            self.imported_symbols[alias_name] = module
            self.imported_qualified_symbols[alias_name] = module
            self.add(
                "import_edge",
                f"import:{index}:{module}:{alias_name}",
                {
                    "kind": "import",
                    "module": module,
                    "name": module,
                    "alias": alias.asname,
                },
                node,
            )
            self.import_dependencies.update(_module_candidates(self.path, module))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        candidates = _module_candidates(self.path, node.module or "", node.level)
        self.import_dependencies.update(candidates)
        if node.level and not candidates:
            # ``from . import name`` cannot be resolved without the package
            # manifest; unknown dependency state is safer than a false-complete
            # cross-file relationship.
            self.unknown_dependencies = True
        for index, alias in enumerate(node.names):
            name = alias.name
            alias_name = alias.asname or name
            self.imported_symbols[alias_name] = name
            qualified_name = f"{module}.{name}".strip(".")
            self.imported_qualified_symbols[alias_name] = qualified_name
            self.add(
                "import_edge",
                f"from:{index}:{module}:{name}:{alias_name}",
                {
                    "kind": "from",
                    "module": module,
                    "name": name,
                    "alias": alias.asname,
                },
                node,
            )

    def _add_guard(self, node: ast.AST, kind: str, predicate: ast.AST) -> None:
        text = _expression(predicate, self.source)
        normalized = ast.dump(predicate, annotate_fields=True, include_attributes=False)
        self.add(
            "guard_predicate",
            f"guard:{kind}:{getattr(predicate, 'lineno', 1)}:{getattr(predicate, 'col_offset', 0)}",
            {
                "kind": kind,
                "predicate": text,
                "normalized_predicate": normalized,
                "source": text,
                "enclosing_symbol": self.enclosing or None,
            },
            predicate,
        )

    def visit_If(self, node: ast.If) -> None:
        self._add_guard(node, "if", node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._add_guard(node, "while", node.test)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self._add_guard(node, "assert", node.test)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _callee(node.func)
        if callee is not None:
            raw_symbol, qualified = callee
            symbol = self.imported_symbols.get(raw_symbol, raw_symbol)
            qualified = self.imported_qualified_symbols.get(raw_symbol, qualified)
            enclosing = self.enclosing or None
            if enclosing:
                self.function_calls.setdefault(enclosing, set()).add(symbol)
            if symbol in {"__import__", "import_module"}:
                self.unknown_dependencies = True
            self.add(
                "static_references_symbol",
                f"reference:{symbol}:{getattr(node, 'lineno', 1)}:{getattr(node, 'col_offset', 0)}",
                {
                    "symbol": symbol,
                    "qualified_name": qualified,
                    "enclosing_symbol": enclosing,
                    "may_exercise": True,
                    "runtime_coverage": False,
                },
                node.func,
                dependencies=tuple(self.import_dependencies),
                dependencies_complete=not self.unknown_dependencies,
            )
        self.generic_visit(node)

    def transitive_test_references(self) -> None:
        """Add conservative same-file test-to-helper references for query answers."""

        for test_name in sorted(self.test_functions):
            seen_functions: set[str] = set()
            frontier = sorted(self.function_calls.get(test_name, ()))
            while frontier:
                called = frontier.pop(0)
                if called in seen_functions:
                    continue
                seen_functions.add(called)
                matches = sorted(
                    name for name in self.function_nodes if name.rsplit(".", 1)[-1] == called
                )
                for match in matches:
                    frontier.extend(sorted(self.function_calls.get(match, ())))
            direct = {
                fact.value.get("symbol")
                for fact in self.facts
                if fact.predicate == "static_references_symbol"
                and fact.value.get("enclosing_symbol") == test_name
            }
            node = self.function_nodes[test_name]
            for symbol in sorted(seen_functions - direct):
                self.add(
                    "static_references_symbol",
                    f"test-reference:{test_name}:{symbol}",
                    {
                        "symbol": symbol,
                        "qualified_name": symbol,
                        "enclosing_symbol": test_name,
                        "may_exercise": True,
                        "runtime_coverage": False,
                        "transitive": True,
                    },
                    spans=(_span(node),),
                    dependencies=tuple(self.import_dependencies),
                    dependencies_complete=not self.unknown_dependencies,
                )


def _file_digest_fact(
    path: str, content: str, *, content_digest: str | None = None,
    kind: str = "python",
) -> ExtractedFact:
    end = max(1, bounded_line_count(content))
    return ExtractedFact(
        path=path,
        predicate="file_digest",
        subject_ref=subject_ref(path, "file"),
        value={
            "digest": content_digest or source_digest(content),
            "line_count": bounded_line_count(content),
            **({"source_kind": kind} if kind != "python" else {}),
        },
        spans=((1, end),),
        dependencies=(path,),
    )


def extract_source_facts(
    path: str, content: str, *, content_digest: str | None = None
) -> tuple[ExtractedFact, ...]:
    """Extract deterministic facts from one Python source file.

    Syntax failure yields a file digest and a deterministic ``parse_error`` fact;
    it never becomes a confident statement about symbols that were not parsed.
    """

    normalized = normalize_path(path)
    if not isinstance(content, str):
        raise TypeError("content must be text")
    kind = source_kind(normalized)
    facts: list[ExtractedFact] = [_file_digest_fact(
        normalized, content, content_digest=content_digest, kind=kind
    )]
    if kind != "python":
        facts.append(
            ExtractedFact(
                path=normalized,
                predicate="parser_unsupported",
                subject_ref=subject_ref(normalized, "parser"),
                value={
                    "source_kind": kind,
                    "status": "UNKNOWN",
                    "reason": "no deterministic parser is registered for this source kind",
                },
                spans=((1, max(1, bounded_line_count(content))),),
                dependencies=(normalized,),
                dependencies_complete=False,
                extractor=f"{LANGUAGE_NEUTRAL_EXTRACTOR}/{LANGUAGE_NEUTRAL_VERSION}",
            )
        )
        return tuple(sorted(facts, key=lambda fact: (fact.predicate, fact.subject_ref)))
    try:
        tree = ast.parse(content, filename=normalized)
    except SyntaxError as exc:
        facts.append(
            ExtractedFact(
                path=normalized,
                predicate="parse_error",
                subject_ref=subject_ref(normalized, "parse"),
                value={
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "line": exc.lineno or 1,
                    "offset": exc.offset or 0,
                },
                spans=((exc.lineno or 1, exc.lineno or 1),),
                dependencies=(normalized,),
            )
        )
        return tuple(sorted(facts, key=lambda fact: (fact.predicate, fact.subject_ref)))

    extractor = _PythonExtractor(normalized, content)
    extractor.visit(tree)
    extractor.transitive_test_references()
    all_dependencies = tuple(sorted(extractor.import_dependencies))
    facts.extend(
        replace(
            fact,
            dependencies=all_dependencies,
            dependencies_complete=not extractor.unknown_dependencies,
        )
        if fact.predicate == "static_references_symbol"
        else fact
        for fact in extractor.facts
    )
    return tuple(
        sorted(
            facts,
            key=lambda fact: (fact.predicate, fact.subject_ref, fact.value_digest),
        )
    )


def extract_test_selector(selector: str, result_class: str, *, path: str | None = None) -> ExtractedFact:
    """Turn an explicit test invocation into an exact oracle fact."""

    normalized_selector = str(selector).strip()
    if not normalized_selector:
        raise ValueError("selector is required")
    result = str(result_class).strip().upper() or "UNKNOWN"
    base, *nodes = normalized_selector.split("::")
    selector_path = normalize_path(path or base) if (path or base) else "<selector>"
    resolved = [node for node in nodes if node]
    return ExtractedFact(
        path=selector_path,
        predicate="oracle_selector",
        subject_ref="oracle://" + _digest(normalized_selector),
        value={
            "selector": normalized_selector,
            "result_class": result,
            "resolved_test_nodes": resolved,
        },
        spans=(),
        dependencies=((path or base,) if (path or base) else ()),
        # A selector result is coupled to the exercised product/environment, but
        # this API carries no complete dependency witness.  Fail closed after
        # any later edit until a fresh selector receipt is observed.
        dependencies_complete=False,
        extractor="explicit_selector/1",
    )


def extract_manifest_facts(files: Mapping[str, str]) -> tuple[ExtractedFact, ...]:
    """Build deterministic file and manifest digest facts for a source snapshot."""

    normalized = {normalize_path(path): content for path, content in files.items()}
    file_facts = [_file_digest_fact(path, content) for path, content in normalized.items()]
    manifest_digest = _digest(
        [(path, source_digest(normalized[path])) for path in sorted(normalized)]
    )
    file_facts.append(
        ExtractedFact(
            path="<manifest>",
            predicate="manifest_digest",
            subject_ref="env://<manifest>#manifest",
            value={"digest": manifest_digest, "paths": sorted(normalized)},
            spans=(),
            dependencies=tuple(sorted(normalized)),
        )
    )
    return tuple(sorted(file_facts, key=lambda fact: (fact.predicate, fact.subject_ref)))


__all__ = [
    "EXTRACTOR_NAME",
    "EXTRACTOR_VERSION",
    "ExtractedFact",
    "MAX_OBSERVED_LINES",
    "bounded_line_count",
    "extract_manifest_facts",
    "extract_source_facts",
    "extract_test_selector",
    "function_signature",
    "normalize_path",
    "source_digest",
    "source_extractor_version",
    "source_kind",
    "subject_ref",
]
