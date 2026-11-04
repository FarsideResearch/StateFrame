"""Catch reads of undocumented stdlib base-class state before Python upgrades do."""

import ast
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STDLIB_BASE_MODULES = frozenset({
    "urllib.request", "http.client", "http.server", "ssl", "socket", "socketserver",
})
# Every exception needs a cross-version review. Unmatched entries fail the test.
ALLOWLIST: dict[tuple[str, str, str], str] = {
}

Hit = tuple[str, str, str, int]  # relative path, class, attribute, line


def _imports(tree: ast.Module) -> dict[str, str]:
    """Resolve module-level stdlib import spellings without importing product code."""
    names: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                # `import http.client` binds `http`; `import http.client as hc` binds `hc`.
                names[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            for alias in statement.names:
                if alias.name != "*":
                    names[alias.asname or alias.name] = f"{statement.module}.{alias.name}"
    return names


def _dotted(node: ast.expr, imports: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return imports.get(node.id)
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value, imports)
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__") and len(name) > 1


def _receiver(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id in {"self", "cls"}) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "super" and not node.args and not node.keywords
    )


class _Attributes(ast.NodeVisitor):
    """Inspect one class body; a nested class owns its own attribute accesses."""

    def __init__(self) -> None:
        self.writes: set[str] = set()
        self.reads: list[ast.Attribute] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # A bare annotation does not initialize a private instance attribute.
        if node.value is None:
            self.visit(node.annotation)
        else:
            self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _private(node.attr) and _receiver(node.value):
            if isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name):
                self.writes.add(node.attr)
            elif isinstance(node.ctx, ast.Load):
                self.reads.append(node)
        self.generic_visit(node)


def _bound_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _bound_names(item)}
    return set()


def _class_names(node: ast.ClassDef, attributes: _Attributes) -> set[str]:
    """Collect names our class defines, including writes in any of its methods."""
    names: set[str] = set()
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(statement.name)
        elif isinstance(statement, ast.Assign):
            names.update(name for target in statement.targets
                         for name in _bound_names(target))
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)) and (
            not isinstance(statement, ast.AnnAssign) or statement.value is not None
        ):
            names.update(_bound_names(statement.target))
        attributes.visit(statement)
    return names | attributes.writes


def scan_source(source: str, path: str) -> list[Hit]:
    """Find private reads inherited from the selected stdlib modules."""
    tree = ast.parse(source, filename=path)
    imports = _imports(tree)
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    # Ambiguous repeated class names cannot establish a trustworthy local parent.
    counts = Counter(node.name for node in classes)
    local = {node.name: node for node in classes if counts[node.name] == 1}
    collected: dict[ast.ClassDef, _Attributes] = {}
    own: dict[ast.ClassDef, set[str]] = {}
    for node in classes:
        attrs = _Attributes()
        own[node] = _class_names(node, attrs)
        collected[node] = attrs

    def ancestors(node: ast.ClassDef, seen: set[ast.ClassDef]) -> tuple[bool, set[str]]:
        if node in seen:
            return False, set()
        seen = seen | {node}
        stdlib = False
        names = set(own[node])
        for base in node.bases:
            dotted = _dotted(base, imports)
            if dotted and dotted.rpartition(".")[0] in STDLIB_BASE_MODULES:
                stdlib = True
            elif isinstance(base, ast.Name) and base.id in local:
                inherited, inherited_names = ancestors(local[base.id], seen)
                stdlib |= inherited
                names.update(inherited_names)
        return stdlib, names

    hits: list[Hit] = []
    for node in classes:
        stdlib, defined = ancestors(node, set())
        if stdlib:
            hits.extend((path, node.name, read.attr, read.lineno)
                        for read in collected[node].reads if read.attr not in defined)
    return sorted(hits, key=lambda hit: (hit[0], hit[3], hit[1], hit[2]))


def scan_tree(root: Path, relative_to: Path) -> list[Hit]:
    hits: list[Hit] = []
    for file in sorted(root.rglob("*.py")):
        hits.extend(scan_source(file.read_text(encoding="utf-8"),
                                file.relative_to(relative_to).as_posix()))
    return hits


def test_planted_controls() -> None:
    fixture = ROOT / "tests/fixtures/stdlib_private_attrs"
    hits = scan_tree(fixture, fixture)
    assert [(path, cls, attr) for path, cls, attr, _ in hits] == [
        ("planted.py", "BadHandler", "_check_hostname"),
        ("planted.py", "DerivedHandler", "_check_hostname"),
        ("planted.py", "TunnelConnection", "_tunnel_host"),
        ("planted.py", "ImportedHandler", "_check_hostname"),
        ("planted.py", "LeakyClassAttr", "_missing"),
        ("planted.py", "ClientAlias", "_missing"),
    ]


def test_original_abortive_handler_regression() -> None:
    source = """import urllib.request
class _AbortiveHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        factory = object
        return self.do_open(factory, request, context=self._context, check_hostname=self._check_hostname)
"""
    assert [(cls, attr) for _, cls, attr, _ in scan_source(source, "regression.py")] == [
        ("_AbortiveHTTPSHandler", "_check_hostname"),
        ("_AbortiveHTTPSHandler", "_context"),
    ]


def test_no_unreviewed_stdlib_private_attribute_reads() -> None:
    hits = scan_tree(ROOT / "src", ROOT)
    actual = {(path, cls, attr) for path, cls, attr, _ in hits}
    stale = sorted(ALLOWLIST.keys() - actual)
    unexpected = [hit for hit in hits if hit[:3] not in ALLOWLIST]
    assert not stale, f"Stale stdlib private-attribute allowlist entries: {stale}"
    assert not unexpected, "Unreviewed stdlib private-attribute reads:\n" + "\n".join(
        f"{path}:{line} {cls}.{attr}" for path, cls, attr, line in unexpected
    )
