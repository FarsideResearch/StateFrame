"""AST guard for unframed path/content hashing; run with a src directory."""
from __future__ import annotations

import ast
import sys
from pathlib import Path


PATH_WORDS = ("path", "relative", "name", "filename", "dependency", " p", "f:", "d:")
CONTENT_WORDS = ("read_bytes", "guarded_read", "source", "raw", "content", "files[",
                 "files.get", "os.readlink", "item.read_bytes", "hashlib.sha256(raw)",
                 "contents")


def _hash_constructor(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "hashlib"
            and (node.func.attr.startswith("sha") or node.func.attr == "new"))


def _empty_hash_constructor(node: ast.AST) -> bool:
    if not _hash_constructor(node):
        return False
    return not node.args or (node.func.attr == "new" and len(node.args) == 1)


def _payload_arguments(node: ast.Call) -> list[ast.AST]:
    return list(node.args[1:] if node.func.attr == "new" else node.args)


def _kind(node: ast.AST) -> set[str]:
    source = ast.unparse(node).lower()
    result = set()
    if any(word in source for word in PATH_WORDS) or source.startswith(("p.encode", "f'{", 'f"{')):
        result.add("path")
    if any(word in source for word in CONTENT_WORDS) or source.startswith(("files[", "h.update(files[")):
        result.add("content")
    return result


def _concatenation(node: ast.AST) -> bool:
    return (isinstance(node, (ast.JoinedStr, ast.BinOp))
            or any(isinstance(child, (ast.JoinedStr, ast.BinOp))
                   or isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                   and child.func.attr == "join" for child in ast.walk(node)))


def _local_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Nested functions own their own hashes, not their parent's findings."""
    nodes = []
    pending = list(fn.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        nodes.append(node)
        pending.extend(ast.iter_child_nodes(node))
    return nodes


def scan_source(source: str) -> tuple[int, set[str]]:
    tree = ast.parse(source)
    eligible = 0
    violations: set[str] = set()
    for fn in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        local = _local_nodes(fn)
        constructors = {target.id for node in local
                        if isinstance(node, (ast.Assign, ast.AnnAssign))
                        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
                        if isinstance(target, ast.Name) and _empty_hash_constructor(node.value)}
        for variable in constructors:
            updates = [node for node in local
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == "update" and isinstance(node.func.value, ast.Name)
                       and node.func.value.id == variable and node.args]
            if updates:
                eligible += 1
                kinds = set().union(*(_kind(node.args[0]) for node in updates))
                expressions = [ast.unparse(node.args[0]).lower() for node in updates]
                fixed_file_hash = any("hashlib.sha256(path.read_bytes())" in expr for expr in expressions)
                # NUL separators are NOT framing: content may contain NUL (stage-2 class).
                if kinds == {"path", "content"} and not fixed_file_hash:
                    violations.add(fn.name)
        for call in (node for node in local if _hash_constructor(node) and _payload_arguments(node)):
            eligible += 1
            payload = _payload_arguments(call)[0]
            expression = ast.unparse(payload)
            if (_concatenation(payload) and _kind(payload) == {"path", "content"}
                    and "\\x00" not in expression and "\\0" not in expression):
                violations.add(fn.name)
    return eligible, violations


def scan_tree(src: Path) -> tuple[int, int, set[str]]:
    count = eligible = 0
    violations: set[str] = set()
    for path in sorted((src / "stateframe").rglob("*.py")):
        if path.name == "kernel_tree_digest.py":
            continue
        count += 1
        site_count, functions = scan_source(path.read_text(encoding="utf-8"))
        eligible += site_count
        violations.update(f"{path.relative_to(src).as_posix()}:{name}" for name in functions)
    return count, eligible, violations


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: digest_class_guard.py <src dir>")
    files, eligible, violations = scan_tree(Path(sys.argv[1]))
    print(f"scanned={files} eligible={eligible} violated={len(violations)}")
    for site in sorted(violations):
        print(site)


if __name__ == "__main__":
    main()
