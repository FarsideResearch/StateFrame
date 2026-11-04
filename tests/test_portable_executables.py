"""Keep executable fixtures runnable on both supported host platforms."""

import ast
import re
from pathlib import Path


TESTS = Path(__file__).resolve().parent
# These are OS-provided at these paths on both macOS and Linux.
PORTABLE = frozenset({"/bin/sh", "/bin/echo", "/bin/cat", "/bin/sleep",
                      "/bin/ps", "/usr/bin/env", "/usr/bin/getconf"})
# Intentional missing-program fixture: validates rejection before launch.
DELIBERATELY_MISSING = {
    "/missing/program": "invalid executable regression",
    "/nonexistent/interpreter": "missing shebang interpreter regression",
    "/no/such/check-program": "missing gate check regression",
    "/does/not/exist": "missing launcher executable regression",
    "/definitely/not/a/hook/executable": "missing hook executable regression",
    "/not/an/executable": "invalid tool executable regression",
    "/no/such/executable": "invalid MCP executable regression",
}
SHEBANG = re.compile(r"(?m)^#!\s*(/[^\s]+)")
PROGRAM = re.compile(r"^(/(?:bin|sbin|usr/bin|usr/sbin|missing)/[\w.\-]+)(?:\s|$)")
ABSOLUTE = re.compile(r"^(/[^\s]+)(?:\s|$)")


def _platform_condition(expr: ast.AST) -> bool | str:
    """Evaluate only simple pytest skip conditions for a simulated Darwin host."""
    if isinstance(expr, ast.Constant):
        return expr.value
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        if (expr.value.id, expr.attr) == ("sys", "platform"):
            return "darwin"
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return not _platform_condition(expr.operand)
    if isinstance(expr, ast.BoolOp):
        values = [_platform_condition(value) for value in expr.values]
        return all(values) if isinstance(expr.op, ast.And) else any(values)
    if isinstance(expr, ast.Compare) and len(expr.ops) == len(expr.comparators) == 1:
        left = _platform_condition(expr.left)
        right = _platform_condition(expr.comparators[0])
        if isinstance(expr.ops[0], ast.Eq):
            return left == right
        if isinstance(expr.ops[0], ast.NotEq):
            return left != right
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "startswith" and len(expr.args) == 1):
        return str(_platform_condition(expr.func.value)).startswith(_platform_condition(expr.args[0]))
    raise ValueError("condition is not a statically known platform check")


def _skipif_on_darwin(node: ast.AST) -> bool:
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(_skipif_on_darwin(item) for item in node.elts)
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr == "skipif" and node.args:
        condition = node.args[0]
        if not any(isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                   and (child.value.id, child.attr) == ("sys", "platform")
                   for child in ast.walk(condition)):
            return False
        try:
            return _platform_condition(condition) is True
        except ValueError:
            return False
    return False


def _early_linux_skip(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for stmt in node.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue  # docstring
        if not isinstance(stmt, ast.If):
            return False
        if not any(isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                   and (child.value.id, child.attr) == ("sys", "platform")
                   for child in ast.walk(stmt.test)):
            return False
        try:
            condition = _platform_condition(stmt.test)
        except ValueError:
            return False
        if condition and any(isinstance(body_stmt, ast.Expr)
                             and isinstance(body_stmt.value, ast.Call)
                             and isinstance(body_stmt.value.func, ast.Attribute)
                             and isinstance(body_stmt.value.func.value, ast.Name)
                             and (body_stmt.value.func.value.id, body_stmt.value.func.attr)
                             == ("pytest", "skip")
                             for body_stmt in stmt.body):
            return True
        return False
    return False


def _module_skipped(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if (isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name)
                and stmt.target.id == "pytestmark" and _skipif_on_darwin(stmt.value)):
            return True
        if (isinstance(stmt, (ast.Assign, ast.AnnAssign))
                and any(isinstance(target, ast.Name) and target.id == "pytestmark"
                        for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]))
                and _skipif_on_darwin(stmt.value)):
            return True
    return False


def scan(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    if _module_skipped(tree):
        return []
    violations = []

    def visit(node: ast.AST, parents: tuple[ast.AST, ...] = ()) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if any(_skipif_on_darwin(decorator) for decorator in node.decorator_list):
                return
            if _early_linux_skip(node):
                return
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            paths = [match.group(1) for match in SHEBANG.finditer(value)]
            parent = parents[-1] if parents else None
            is_first = (isinstance(parent, ast.List)
                        and bool(parent.elts) and parent.elts[0] is node)
            is_path_fixture = (isinstance(parent, ast.Call)
                               and isinstance(parent.func, ast.Name) and parent.func.id == "Path")
            owner = parents[-2] if len(parents) > 1 else None
            launch_list = False
            if is_first and isinstance(owner, ast.Call):
                func = owner.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else None)
                launch_list = name in {"launch", "invoke", "run", "Popen", "call",
                                       "check_call", "check_output", "run_sandboxed",
                                       "capability_command", "isolated_command", "_prepared"}
            configured_argv = (is_first and isinstance(owner, ast.Dict)
                               and any(value is parent and isinstance(key, ast.Constant)
                                       and key.value == "argv"
                                       for key, value in zip(owner.keys, owner.values)))
            parameterized_argv = (is_first and isinstance(owner, ast.Tuple)
                                  and len(owner.elts) == 2 and owner.elts[1] is parent
                                  and isinstance(owner.elts[0], ast.Constant)
                                  and owner.elts[0].value == "argv")
            slash = chr(47)
            # A standalone path is also an executable argument in probe/config fixtures.
            is_bare = not is_path_fixture and bool(PROGRAM.match(value)) and (
                is_first or value.startswith((f"{slash}bin/true", f"{slash}usr/bin/true",
                                              f"{slash}usr/bin/python3"))
                or isinstance(parent, ast.Call))
            if is_bare:
                paths.append(PROGRAM.match(value).group(1))
            if not is_path_fixture and (launch_list or configured_argv or parameterized_argv):
                match = ABSOLUTE.match(value)
                if match:
                    paths.append(match.group(1))
            for path in set(paths) - PORTABLE - DELIBERATELY_MISSING.keys():
                violations.append(f"{filename}:{node.lineno}: {path}")
        for child in ast.iter_child_nodes(node):
            visit(child, (*parents, node))

    visit(tree)
    return violations


def test_scanner_controls():
    slash = chr(47)
    source = (f'import sys, pytest\n'
              f'def test_bad():\n    run(["{slash}bin/true"])\n')
    source += f"    probe('{slash}usr/bin/python3')\n"
    source += (f'@pytest.mark.skipif(sys.platform == "darwin", reason="Linux only")\n'
               f'def test_linux():\n    run(["{slash}bin/true"])\n')
    found = scan(source, "control.py")
    assert len(found) == 2, found
    assert any(f"{slash}bin/true" in item for item in found)
    assert any(f"{slash}usr/bin/python3" in item for item in found)


def test_skip_scopes_and_path_fixtures():
    slash = chr(47)
    path = f"{slash}bin/true"
    class_skip = (f'import sys, pytest\n'
                  f'@pytest.mark.skipif(sys.platform == "darwin", reason="Linux only")\n'
                  f'class TestLinux:\n    def test_case(self):\n        run(["{path}"])\n')
    assert scan(class_skip, "class.py") == []
    module_skip = (f'import sys, pytest\n'
                   f'pytestmark = [pytest.mark.skipif(not sys.platform.startswith("linux"), '
                   f'reason="Linux only")]\nrun(["{path}"])\n')
    assert scan(module_skip, "module.py") == []
    early_skip = (f'import sys, pytest\n'
                  f'def test_case():\n'
                  f'    if not sys.platform.startswith("linux"):\n'
                  f'        pytest.skip("Linux only")\n    run(["{path}"])\n')
    assert scan(early_skip, "early.py") == []
    generic_skip = (f'import pytest\n@pytest.mark.skipif(True, reason="unrelated")\n'
                    f'def test_case():\n    run(["{path}"])\n')
    assert scan(generic_skip, "generic.py") == [f"generic.py:4: {path}"]
    assert scan(f'from pathlib import Path\np = Path("{slash}usr/bin/python3")\n',
                "fixture.py") == []


def test_launch_lists_and_deliberate_missing_programs():
    slash = chr(47)
    source = f'def test_case():\n    launch(["{slash}opt/custom-tool"])\n'
    assert scan(source, "launch.py") == [f"launch.py:2: {slash}opt/custom-tool"]
    source = f'def test_case():\n    launch(["{slash}does/not/exist"])\n'
    assert scan(source, "negative.py") == []


def test_no_nonportable_executable_literals():
    violations = [item for path in sorted(TESTS.rglob("*.py"))
                  for item in scan(path.read_text(encoding="utf-8"), str(path.relative_to(TESTS)))]
    assert not violations, "Nonportable executable literals:\n" + "\n".join(violations)


EXAMPLES = TESTS.parent / "examples" / "powerups"
HOOK_EXAMPLES = TESTS.parent / "examples" / "hooks"


def scan_example_argv(text: str, filename: str) -> list[str]:
    """Report absolute, nonportable argv[0] values in a shipped powerup JSON file."""
    import json

    value = json.loads(text)
    found = []
    for key in ("tools", "hooks", "servers"):
        for item in value.get(key, []) if isinstance(value, dict) else []:
            argv = item.get("argv") if isinstance(item, dict) else None
            if isinstance(argv, list) and argv and isinstance(argv[0], str):
                program = argv[0]
                if program.startswith("/") and program not in PORTABLE:
                    found.append(f"{filename}: {program}")
    return found


def scan_example_programs(examples: Path) -> list[str]:
    """Report nonportable executables in example configs and Python shebangs."""
    violations = [item for path in sorted(examples.rglob("*.json"))
                  for item in scan_example_argv(path.read_text(encoding="utf-8"),
                                                str(path.relative_to(examples)))]
    for path in sorted(examples.rglob("powerup.toml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r'\s*default\s*=\s*"(/[^\s"]+)', line)
            if match and match.group(1) not in PORTABLE:
                violations.append(f"{path.relative_to(examples)}: {match.group(1)}")
    for path in sorted(examples.rglob("*.py")):
        with path.open(encoding="utf-8") as script:
            match = SHEBANG.match(script.readline())
        if match and match.group(1) not in PORTABLE:
            violations.append(f"{path.relative_to(examples)}: {match.group(1)}")
    return violations


def scan_hook_scripts(examples: Path) -> list[str]:
    violations = []
    for path in sorted(examples.glob("*.py")):
        with path.open("rb") as script:
            if script.readline(4096).startswith(b"#!"):
                violations.append(f"{path.name}: shebang")
        if path.stat().st_mode & 0o111:
            violations.append(f"{path.name}: executable mode")
    return violations


def test_example_scanner_flags_planted_absolute_python(tmp_path):
    example = tmp_path / "fake-powerup"
    example.mkdir()
    hooks = example / "hooks.json"
    script = example / "script.py"
    hooks.write_text('{"hooks": [{"argv": ["/usr/bin/python3", "script.py"]}]}',
                     encoding="utf-8")
    script.write_text("#!" + "/usr/bin/" + 'python3\nprint("hello")\n',
                      encoding="utf-8")
    assert scan_example_programs(tmp_path) == [
        "fake-powerup/hooks.json: /usr/bin/python3",
        "fake-powerup/script.py: /usr/bin/python3",
    ]

    hooks.write_text('{"hooks": [{"argv": ["python3", "script.py"]}]}',
                     encoding="utf-8")
    script.write_text('#!/usr/bin/env python3\nprint("hello")\n', encoding="utf-8")
    assert scan_example_programs(tmp_path) == []

    hook_example = tmp_path / "fake-hooks"
    hook_example.mkdir()
    (hook_example / "hooks.json").write_text(
        '{"hooks": [{"argv": ["{python}", "script.py"]}]}',
        encoding="utf-8",
    )
    (hook_example / "script.py").write_text(
        "#!" + "/usr/bin/" + "python3\nprint('hello')\n", encoding="utf-8"
    )
    assert scan_example_programs(tmp_path) == [
        "fake-hooks/script.py: /usr/bin/python3",
    ]
    assert scan_hook_scripts(hook_example) == ["script.py: shebang"]


def test_shipped_powerups_use_portable_programs():
    violations = scan_example_programs(EXAMPLES)
    assert not violations, "Nonportable programs in shipped powerups:\n" + "\n".join(violations)


def test_shipped_hooks_use_python_token_without_shebangs_or_executable_modes():
    import json

    config = json.loads((HOOK_EXAMPLES / "hooks.json").read_text(encoding="utf-8"))
    assert all(hook["argv"][0] == "{python}" for hook in config["hooks"])
    scripts = sorted(HOOK_EXAMPLES.glob("*.py"))
    assert scripts
    violations = scan_hook_scripts(HOOK_EXAMPLES)
    assert not violations, "Nonportable Python hook scripts:\n" + "\n".join(violations)
