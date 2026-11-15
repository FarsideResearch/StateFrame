"""Pinned Git oracles must survive a clone containing only HEAD's history."""
import ast
import itertools
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_UNKNOWN = "{unresolved}"


def git_show_references(source):
    """Find literal, constant, default and parametrized revisions without importing tests.

    An unknown filename in an f-string is fine; an unknown revision fails closed.
    Index-only reads have no revision and do not depend on historical objects.
    """
    tree = ast.parse(source)
    bindings = {}

    def bind(name, value):
        bindings.setdefault(name, []).append(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bind(target.id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bind(node.target.id, node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = (*node.args.posonlyargs, *node.args.args)
            for arg, default in zip(reversed(args), reversed(node.args.defaults)):
                bind(arg.arg, default)
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None:
                    bind(arg.arg, default)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "parametrize" and len(node.args) >= 2
              and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)
              and isinstance(node.args[1], (ast.List, ast.Tuple))):
            names = [name.strip() for name in node.args[0].value.split(",")]
            for row in node.args[1].elts:
                if isinstance(row, ast.Call):
                    values = row.args  # pytest.param(..., marks=..., id=...)
                elif len(names) == 1:
                    values = [row]
                elif isinstance(row, (ast.List, ast.Tuple)):
                    values = row.elts
                else:
                    continue
                for name, value in zip(names, values):
                    bind(name, value)

    def strings(node, seen=frozenset()):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name) and node.id not in seen:
            return set().union(*(strings(value, seen | {node.id})
                                 for value in bindings.get(node.id, ()))) or {_UNKNOWN}
        if isinstance(node, ast.JoinedStr):
            parts = [strings(part.value if isinstance(part, ast.FormattedValue) else part, seen)
                     for part in node.values]
            return {"".join(parts) for parts in itertools.product(*parts)}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return {left + right for left in strings(node.left, seen)
                    for right in strings(node.right, seen)}
        return {_UNKNOWN}

    revisions = set()

    def record(value):
        revision = value.split(":", 1)[0]
        assert _UNKNOWN not in revision, f"unresolved Git oracle revision: {value}"
        if revision:
            revisions.add(revision)

    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 3:
            words = [item.value if isinstance(item, ast.Constant) else None for item in node.elts]
            if words[0] == "git" and "show" in words[1:-1]:
                for value in strings(node.elts[words.index("show") + 1]):
                    record(value)
        elif (isinstance(node, ast.Call) and node.args
              and isinstance(node.func, ast.Attribute)
              and node.func.attr in {"run", "check_output", "check_call", "call", "Popen"}):
            for command in strings(node.args[0]):
                if command.startswith("git "):
                    words = shlex.split(command)
                    if "show" in words[1:-1]:
                        record(words[words.index("show") + 1])
    return revisions


def literal_parity_references(source):
    """Cover the working-set wrappers that materialize or delegate pinned trees.

    Keep malformed literals too: a concatenated short/full SHA must fail the
    ancestry check rather than disappear from the audit.
    """
    revisions = set()

    def record(argument):
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            revisions.add(argument.value)

    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            if node.func.id == "pinned_tree":
                index, keyword = 0, "commit"
            elif node.func.id == "test_off_provider_bytes_match_cutover_base":
                index, keyword = 2, "reference"
            else:
                continue
            if len(node.args) > index:
                record(node.args[index])
            for argument in node.keywords:
                if argument.arg == keyword:
                    record(argument.value)
        elif (isinstance(node.func, ast.Attribute) and node.func.attr == "parametrize"
              and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant)
              and node.args[0].value == "reference"
              and isinstance(node.args[1], (ast.List, ast.Tuple))):
            for row in node.args[1].elts:
                record(row.args[0] if isinstance(row, ast.Call) and row.args else row)
    return revisions


def test_git_show_oracles_are_ancestors_of_head():
    if not (ROOT / ".git").exists():
        pytest.skip("source archive has no Git history to audit")
    references = {}
    for path in sorted((ROOT / "tests").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        revisions = git_show_references(source)
        if path.name.startswith(("test_working_set_", "test_read_receipts")):
            revisions.update(literal_parity_references(source))
        for revision in revisions:
            references.setdefault(revision, []).append(str(path.relative_to(ROOT)))
    assert references, "no pinned Git oracles found; check the scanner"
    failures = []
    for revision, paths in sorted(references.items()):
        result = subprocess.run(["git", "merge-base", "--is-ancestor", revision, "HEAD"],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        if result.returncode:
            failures.append(f"{revision}: {', '.join(paths)} ({result.stderr.strip()})")
    assert not failures, "Git oracles outside HEAD history:\n" + "\n".join(failures)


@pytest.mark.parametrize("source,expected", [
    ('subprocess.run(["git", "show", "abc1234:src/a.py"])', {"abc1234"}),
    ('subprocess.run(["git", "show", f"abc1234:src/{filename}.py"])', {"abc1234"}),
    (('BASELINE = "abc1234"\n'
     '@pytest.mark.parametrize("baseline", [BASELINE, "def5678"])\n'
     'def test_oracle(baseline=BASELINE):\n'
     '    subprocess.run(["git", "show", f"{baseline}:src/{filename}.py"])'),
     {"abc1234", "def5678"}),
    (('BASELINE: str = "abc1234"\n'
     'subprocess.run(["git", "show", BASELINE + ":src/a.py"])'), {"abc1234"}),
    (('@pytest.mark.parametrize("baseline,other", [pytest.param("abc1234", 1, id="old")])\n'
     'def test_oracle(baseline, other):\n'
     '    subprocess.run(["git", "show", f"{baseline}:src/a.py"])'), {"abc1234"}),
    ('subprocess.run("git show abc1234:src/a.py", shell=True)', {"abc1234"}),
    ('subprocess.run(["git", "show", "v0.2.1:src/a.py"])', {"v0.2.1"}),
    ('subprocess.run(["git", "show", ":src/a.py"])', set()),
    ('sha = "abc1234"\nvalues = ["show", sha]\n# git show def5678:ignored', set()),
])
def test_git_show_reference_discovery(source, expected):
    assert git_show_references(source) == expected


def test_git_show_reference_discovery_refuses_unresolved_revision():
    with pytest.raises(AssertionError, match="unresolved Git oracle revision"):
        git_show_references('subprocess.run(["git", "show", f"{baseline}:src/a.py"])')


@pytest.mark.parametrize("source,expected", [
    ('pinned_tree("abc1234", factory)', {"abc1234"}),
    ('pinned_tree(commit="abc1234", tmp_path_factory=factory)', {"abc1234"}),
    ('test_off_provider_bytes_match_cutover_base(tmp, factory, "abc1234")', {"abc1234"}),
    ('test_off_provider_bytes_match_cutover_base(tmp, factory, reference="abc1234")',
     {"abc1234"}),
    ('@pytest.mark.parametrize("reference", ["abc1234", pytest.param("def5678", id="base")])\n'
     'def test_parity(reference): pass', {"abc1234", "def5678"}),
    ('pinned_tree("' + "a" * 71 + '", factory)', {"a" * 71}),
    ('pinned_tree(reference, factory)', set()),
    ('# pinned_tree("abc1234", factory)\nvalue = "def5678"', set()),
])
def test_literal_parity_reference_discovery(source, expected):
    assert literal_parity_references(source) == expected


@pytest.mark.parametrize("oracle", ["git_show", "pinned_tree", "parity_helper", "parameter"])
@pytest.mark.parametrize("reference,rejected", [
    ("parent", False), ("sibling", True), ("missing", True), ("malformed", True),
])
def test_ancestry_guard_rejects_even_present_nonancestor_objects(tmp_path, monkeypatch,
                                                               reference, rejected, oracle):
    def git(*args, data=None):
        return subprocess.check_output([
            "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
            cwd=tmp_path, input=data, stderr=subprocess.PIPE).decode().strip()

    git("init", "--initial-branch=main")
    tree = git("mktree", data=b"")
    parent = git("commit-tree", tree, data=b"parent\n")
    head = git("commit-tree", tree, "-p", parent, data=b"head\n")
    sibling = git("commit-tree", tree, "-p", parent, data=b"sibling\n")
    git("update-ref", "refs/heads/main", head)
    revision = {"parent": parent, "sibling": sibling, "missing": "0" * 40,
                "malformed": "0" * 71}[reference]
    if reference in {"parent", "sibling"}:
        git("cat-file", "-e", revision)
    tests = tmp_path / "tests"
    tests.mkdir()
    source = {
        "git_show": f'subprocess.run(["git", "show", "{revision}:oracle.py"])\n',
        "pinned_tree": f'pinned_tree("{revision}", factory)\n',
        "parity_helper": f'test_off_provider_bytes_match_cutover_base(tmp, factory, "{revision}")\n',
        "parameter": f'@pytest.mark.parametrize("reference", ["{revision}"])\n'
                     'def test_parity(reference): pass\n',
    }[oracle]
    (tests / "test_working_set_oracle.py").write_text(source)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    if rejected:
        with pytest.raises(AssertionError, match=revision):
            test_git_show_oracles_are_ancestors_of_head()
    else:
        test_git_show_oracles_are_ancestors_of_head()
