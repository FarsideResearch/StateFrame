"""Every executable Python module in release artifacts needs an explicit review."""

from __future__ import annotations

import ast
from itertools import product
from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile

import pytest


# Paths are relative to the wheel root or the sdist's project root. A main
# block is allowed only for the purpose recorded here; batch drivers live in
# maintainer task driver and are deliberately absent from both release artifacts.
ALLOWLIST = {
    "scripts/build_wheelhouse.py": "sdist installer dependency build tool",
    "stateframe/_child.py": "isolated engine child process",
    "stateframe/cli.py": "registered stateframe command",
    "stateframe/engine/blame_channel.py": "diagnostic command",
    "stateframe/engine/capability_release_runner.py": "release capability runner",
    "stateframe/engine/chains.py": "seam manifest diagnostic",
    "stateframe/engine/detreason/diagnosis.py": "operator diagnosis command",
    "stateframe/engine/detreason/plan_synth.py": "operator plan diagnostic",
    "stateframe/engine/detreason/priors.py": "operator prior diagnostic",
    "stateframe/engine/detreason/skeptic.py": "manual audit driver",
    "stateframe/engine/experiment_protocol.py": "experiment protocol command",
    "stateframe/engine/fixer.py": "direct fixer command",
    "stateframe/engine/kernel_bolt.py": "kernel self-test",
    "stateframe/engine/kernel_bounded_http.py": "HTTP and probe child process",
    "stateframe/engine/kernel_exec_supervisor.py": "execution supervisor child process",
    "stateframe/engine/kernel_hook_driver.py": "sandbox hook child process",
    "stateframe/engine/kernel_liveness.py": "liveness diagnostic",
    "stateframe/engine/kernel_release_switch_registry.py": "release registry maintenance command",
    "stateframe/engine/kernel_stateframe_thinking_router.py": "operator routing command",
    "stateframe/engine/probes/apply_corpus_fixes.py": "explicit corpus repair command",
    "stateframe/engine/prompt_ledger.py": "ledger diagnostic command",
    "stateframe/engine/protected_tree_guard.py": "protected tree check command",
    "stateframe/engine/sf_launch.py": "sandbox launch child process",
    "stateframe/engine/sf_mutate_boot.py": "sandbox mutation boot gate and unit runner child process",
    "stateframe/engine/sf_mutate_fetch.py": "bounded mutation HTTP broker child process",
    "stateframe/engine/sidekick.py": "sidekick self-test",
    "stateframe/engine/symbol_registry.py": "symbol registry command",
    "stateframe/outer_checks.py": "outer check helper process",
    "stateframe/scripted_endpoint.py": "offline example endpoint command",
}


# Process entry points must retain their launch sites, not merely an import or
# a digest/staging reference. Operator commands above are launched externally.
# The bounded HTTP and outer-check helpers relaunch their own file by __file__.
PROCESS_LAUNCH_SITES = {
    "stateframe/_child.py": (
        "stateframe/runner.py", "stateframe._child", "subprocess.Popen"),
    "stateframe/engine/kernel_bounded_http.py": (
        "stateframe/engine/kernel_bounded_http.py",
        "--http-child", "subprocess.Popen"),
    "stateframe/engine/kernel_exec_supervisor.py": (
        "stateframe/engine/kernel_exec_session.py",
        "kernel_exec_supervisor.py", "subprocess.Popen"),
    "stateframe/engine/kernel_hook_driver.py": (
        "stateframe/engine/kernel_hook_exec.py",
        "kernel_hook_driver.py", "session.run"),
    "stateframe/engine/sf_launch.py": (
        "stateframe/engine/kernel_exec_supervisor.py",
        "/sf/boot/sf_launch.py", "subprocess.Popen"),
    "stateframe/engine/sf_mutate_boot.py": (
        "stateframe/engine/kernel_mutate_exec.py",
        "/sf/boot/sf_mutate_boot.py", "self._session._spawn"),
    "stateframe/engine/sf_mutate_fetch.py": (
        "stateframe/engine/sf_mutate_boot.py",
        "sf_mutate_fetch.py", "runpy.run_path"),
    "stateframe/outer_checks.py": (
        "stateframe/outer_checks.py",
        "--helper", "subprocess.Popen"),
}


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return (
        len(test.ops) == len(test.comparators) == 1
        and isinstance(test.ops[0], ast.Eq)
        and (
            isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        )
    )


def unlisted_mains(sources: dict[str, bytes]) -> list[str]:
    violations = []
    for name, source in sorted(sources.items()):
        tree = ast.parse(source, filename=name)
        if any(_is_main_guard(node) for node in ast.walk(tree)) and name not in ALLOWLIST:
            violations.append(name)
    return violations


def _artifact_sources(wheel: Path, sdist: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    with zipfile.ZipFile(wheel) as archive:
        wheel_sources = {
            name: archive.read(name)
            for name in archive.namelist()
            if name.endswith(".py")
        }
    with tarfile.open(sdist, "r:gz") as archive:
        root = archive.getnames()[0].split("/", 1)[0] + "/"
        sdist_sources = {
            name.removeprefix(root).removeprefix("src/"): archive.extractfile(name).read()
            for name in archive.getnames()
            if name.endswith(".py")
        }
    return wheel_sources, sdist_sources


def missing_launch_sites(
    sources: dict[str, bytes], sites: dict[str, tuple[str, str, str]],
) -> list[str]:
    """Require the reviewed entry point in the launch's script position.

    This is a deliberately small, fail-closed source check, not a Python evaluator.
    Follow local bindings and the path/argv builders used by the reviewed sites;
    do not count arbitrary function arguments, unrelated scopes or launch options.
    """
    missing = []
    for name, (caller, target, launch) in sites.items():
        tree = ast.parse(sources.get(caller, b""), filename=caller)
        targets = {node.value for node in ast.walk(tree)
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        if target not in targets or not _LaunchBindings(launch, name).check(
            tree.body, {"__file__": {caller}},
        ):
            missing.append(name)
    return missing


# Unknown scalars/argv entries remain opaque; unsupported expressions cannot
# manufacture launch evidence. Sets retain alternatives at control-flow joins.
_Value = str | tuple[str | None, ...] | None
_Bindings = dict[str, set[_Value]]
_PYTHON = "<python interpreter>"


class _LaunchBindings:
    def __init__(self, launch: str, entry: str) -> None:
        self.launch = launch
        self.entry = entry

    def values(self, node: ast.AST, bindings: _Bindings) -> set[_Value]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if ast.unparse(node) in {"sys.executable", "options.python"}:
            return {_PYTHON}
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            return {str(PurePosixPath(value).parent) if isinstance(value, str) else None
                    for value in self.values(node.value, bindings)}
        if isinstance(node, (ast.Name, ast.Attribute)):
            return bindings.get(ast.unparse(node), {None})
        if isinstance(node, (ast.List, ast.Tuple)):
            sequences: set[_Value] = {()}
            for item in node.elts:
                expanded: set[_Value] = set()
                starred = isinstance(item, ast.Starred)
                for sequence, value in product(sequences, self.values(
                    item.value if starred else item, bindings,
                )):
                    assert isinstance(sequence, tuple)
                    suffix = (value if isinstance(value, tuple) else (None,)) if starred else (
                        value if isinstance(value, str) else None,
                    )
                    expanded.add(sequence + suffix)
                if len(expanded) > 64:
                    return {None}
                sequences = expanded
            return sequences
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
            result: set[_Value] = set()
            for left, right in product(self.values(node.left, bindings), self.values(node.right, bindings)):
                if isinstance(node.op, ast.Add) and isinstance(left, tuple) and isinstance(right, tuple):
                    result.add(left + right)
                elif isinstance(left, str) and isinstance(right, str):
                    result.add(str(PurePosixPath(left) / right) if isinstance(node.op, ast.Div)
                               else left + right)
                else:
                    result.add(None)
            return result
        if isinstance(node, ast.IfExp):
            return self.values(node.body, bindings) | self.values(node.orelse, bindings)
        if isinstance(node, ast.Call):
            if ast.unparse(node.func) in {"str", "Path", "os.path.realpath"} and len(node.args) == 1:
                return self.values(node.args[0], bindings)
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "with_name" and len(node.args) == 1:
                    return {str(PurePosixPath(base).with_name(name))
                            if isinstance(base, str) and isinstance(name, str) else None
                            for base, name in product(self.values(node.func.value, bindings),
                                                      self.values(node.args[0], bindings))}
                if node.func.attr == "resolve" and not node.args:
                    return self.values(node.func.value, bindings)
        return {None}

    def launches(self, value: _Value) -> bool:
        def is_entry(path: str | None) -> bool:
            return isinstance(path, str) and PurePosixPath(path).name == PurePosixPath(self.entry).name

        if self.launch == "runpy.run_path":
            return isinstance(value, str) and is_entry(value)
        if not isinstance(value, tuple):
            return False
        for index, token in enumerate(value):
            if token not in {_PYTHON, "python", "python3"}:
                continue
            # A wrapped Python command must follow the wrapper's '--' delimiter.
            if index and value[index - 1] != "--":
                continue
            tail = list(value[index + 1:])
            while tail and tail[0] in {"-I", "-B", "-S", "-u"}:
                tail.pop(0)
            if tail and is_entry(tail[0]):
                return True
            if len(tail) >= 2 and tail[0] == "-m" and isinstance(tail[1], str):
                if tail[1] == self.entry.removesuffix(".py").replace("/", "."):
                    return True
        return False

    @staticmethod
    def replace_aliases(bindings: _Bindings, old: set[_Value], new: set[_Value]) -> None:
        for key, values in list(bindings.items()):
            if values & old:
                bindings[key] = (values - old) | new

    def check(self, statements: list[ast.stmt], bindings: _Bindings) -> bool:
        found = False
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                local = bindings.copy()
                # A local binding shadows an outer binding throughout the function,
                # including before its assignment. Nested bodies have their own scope.
                pending: list[ast.AST] = [*statement.body, statement.args]
                while pending:
                    node = pending.pop()
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        local.pop(node.name, None)
                        continue
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        local.pop(node.id, None)
                    if isinstance(node, ast.arg):
                        local.pop(node.arg, None)
                        if node.arg == "python":
                            # outer_checks receives the selected test interpreter.
                            local[node.arg] = {_PYTHON}
                    pending.extend(ast.iter_child_nodes(node))
                found |= self.check(statement.body, local)
                continue
            if isinstance(statement, ast.ClassDef):
                local = bindings.copy()
                # Only constructor-assigned instance paths are shared between methods.
                for method in statement.body:
                    if isinstance(method, ast.FunctionDef) and method.name == "__init__":
                        instance = bindings.copy()
                        self.check(method.body, instance)
                        local.update({key: value for key, value in instance.items()
                                      if key.startswith("self.")})
                found |= self.check(statement.body, local)
                continue
            # Inspect this statement's expressions, excluding nested statement bodies.
            pending = list(ast.iter_child_nodes(statement))
            while pending:
                node = pending.pop()
                if isinstance(node, (ast.stmt, ast.Lambda)):
                    continue
                if (isinstance(node, ast.Call) and ast.unparse(node.func) == self.launch):
                    argument = node.args[0] if node.args else next(
                        (kw.value for kw in node.keywords if kw.arg in {"args", "path_name"}),
                        None,
                    )
                    if argument is not None and any(
                        self.launches(value) for value in self.values(argument, bindings)
                    ):
                        found = True
                pending.extend(ast.iter_child_nodes(node))
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = self.values(statement.value, bindings) if statement.value else {None}
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, (ast.Tuple, ast.List)):
                        for item in target.elts:
                            bindings[ast.unparse(item)] = {None}
                        # Reviewed sandbox builder returns (wrapped argv, environment).
                        # It preserves argv in both bwrap and sandbox-exec modes.
                        if (isinstance(statement.value, ast.Call)
                                and ast.unparse(statement.value.func) == "capability_command"
                                and statement.value.args and len(target.elts) == 2):
                            bindings[ast.unparse(target.elts[0])] = self.values(
                                statement.value.args[0], bindings)
                    else:
                        if isinstance(target, ast.Subscript):
                            key = ast.unparse(target.value)
                            old = bindings.get(key, {None})
                            updated: set[_Value] = set()
                            try:
                                index = ast.literal_eval(target.slice)
                            except (ValueError, TypeError):
                                index = None
                            for sequence, item in product(old, value):
                                if (isinstance(sequence, tuple) and type(index) is int
                                        and -len(sequence) <= index < len(sequence)):
                                    items = list(sequence)
                                    items[index] = item if isinstance(item, str) else None
                                    updated.add(tuple(items))
                                else:
                                    updated.add(None)
                            self.replace_aliases(bindings, old, updated)
                        else:
                            bindings[ast.unparse(target)] = value
            elif isinstance(statement, ast.AugAssign):
                key = ast.unparse(statement.target)
                bindings[key] = self.values(ast.BinOp(
                    left=statement.target, op=statement.op, right=statement.value,
                ), bindings)
            else:
                if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                        and isinstance(statement.value.func, ast.Attribute)):
                    # Unknown in-place argv edits cannot preserve a launch witness.
                    receiver = ast.unparse(statement.value.func.value)
                    old = {value for value in bindings.get(receiver, set())
                           if isinstance(value, tuple)}
                    self.replace_aliases(bindings, old, {None})
                # Explore branches separately, then retain possible reaching values.
                branches = []
                if isinstance(statement, (ast.With, ast.AsyncWith)):
                    found |= self.check(statement.body, bindings)
                    continue
                for field in ("body", "orelse", "finalbody"):
                    body = getattr(statement, field, None)
                    if isinstance(body, list) and body:
                        branch = bindings.copy()
                        found |= self.check(body, branch)
                        branches.append(branch)
                for handler in getattr(statement, "handlers", []):
                    branch = bindings.copy()
                    found |= self.check(handler.body, branch)
                    branches.append(branch)
                if branches:
                    if isinstance(statement, ast.If) and not statement.orelse:
                        branches.append(bindings.copy())
                    for key in set().union(*(branch.keys() for branch in branches)):
                        bindings[key] = set().union(*(branch.get(key, set()) for branch in branches))
        return found


def test_launch_sites_require_live_code_not_comments_or_imports() -> None:
    sites = {"child.py": ("caller.py", "child.py", "subprocess.Popen")}
    for source in (
        b"",  # absent caller
        b"import child\n",
        b"# subprocess.Popen(['python', 'child.py'])\n",
        b"path = 'child.py'\n",  # staging/digest reference without a launch
        b"subprocess.Popen(['python', 'other.py'])\n",
    ):
        assert missing_launch_sites({"caller.py": source}, sites) == ["child.py"]
    assert missing_launch_sites({}, sites) == ["child.py"]
    assert missing_launch_sites(
        {"caller.py": b"subprocess.Popen(['python', 'child.py'])\n"}, sites,
    ) == []


def test_launch_sites_reject_unrelated_target_and_launch() -> None:
    sites = {"child.py": ("caller.py", "child.py", "subprocess.Popen")}
    source = b"path = 'child.py'; subprocess.Popen(['python', 'other.py'])"
    assert missing_launch_sites({"caller.py": source}, sites) == ["child.py"]


@pytest.mark.parametrize("source", [
    "path = 'child.py'; argv = ['python', path]; subprocess.Popen(argv)",
    "path = 'child.py'; subprocess.Popen(args=['python', path])",
    "path = Path(__file__).with_name('child.py').resolve(); runpy.run_path(str(path))",
    "argv = ['python', 'child.py']; argv = [*wrapper, '--', *argv]; subprocess.Popen(argv)",
    "class Runner:\n"
    " def __init__(self):\n  self.path = 'child.py'\n"
    " def run(self):\n  subprocess.Popen(['python', self.path])\n",
])
def test_launch_sites_follow_bound_arguments(source: str) -> None:
    launch = "runpy.run_path" if "runpy.run_path" in source else "subprocess.Popen"
    sites = {"child.py": ("caller.py", "child.py", launch)}
    assert missing_launch_sites({"caller.py": source.encode()}, sites) == []


@pytest.mark.parametrize("source", [
    "path = 'child.py'; path = 'other.py'; subprocess.Popen(['python', path])",
    "subprocess.Popen(['python', path]); path = 'child.py'",
    "subprocess.Popen(['python', 'other.py'], cwd='child.py')",
    "subprocess.Popen(['python', 'other.py', 'child.py'])",
    "subprocess.Popen(['python', '-c', 'pass', 'child.py'])",
    "subprocess.Popen(['node', 'child.py'])",
    "subprocess.Popen(['python', 'other.py'] + ['python', 'child.py'])",
    "path = 'child.py'; subprocess.Popen(replace(path))",
    "argv = ['python', 'child.py']; argv[1] = 'other.py'; subprocess.Popen(argv)",
    "argv = ['python', 'child.py']; alias = argv; alias.clear(); subprocess.Popen(argv)",
    "def unrelated():\n path = 'child.py'\n"
    "def launch():\n subprocess.Popen(['python', path])\n",
    "path = 'child.py'\ndef launch(path):\n subprocess.Popen(['python', path])\n",
    "path = 'child.py'\ndef launch():\n"
    " subprocess.Popen(['python', path])\n path = 'other.py'\n",
    "path = 'child.py'\nif flag:\n path = 'other.py'\nelse:\n path = 'another.py'\n"
    "subprocess.Popen(['python', path])\n",
    "class Runner:\n"
    " def __init__(self):\n  path = 'child.py'\n"
    " def run(self):\n  subprocess.Popen(['python', path])\n",
])
def test_launch_sites_reject_disconnected_bindings(source: str) -> None:
    sites = {"child.py": ("caller.py", "child.py", "subprocess.Popen")}
    assert missing_launch_sites({"caller.py": source.encode()}, sites) == ["child.py"]


@pytest.mark.parametrize("name", PROCESS_LAUNCH_SITES)
@pytest.mark.parametrize("target_as_data", [False, True])
def test_every_process_launch_requires_its_target_argument(name: str, target_as_data: bool) -> None:
    caller, target, launch = PROCESS_LAUNCH_SITES[name]
    repo = Path(__file__).resolve().parents[1]
    tree = ast.parse((repo / "src" / caller).read_bytes(), filename=caller)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and ast.unparse(node.func) == launch]
    assert calls, f"missing reviewed launch call: {caller} {launch}"
    assert missing_launch_sites(
        {caller: ast.unparse(tree).encode()}, {name: PROCESS_LAUNCH_SITES[name]},
    ) == []
    for call in calls:
        assert call.args, f"review command argument binding for {caller} {launch}"
        argv = ["python", "other.py"]
        if target_as_data:
            argv.extend([target, PurePosixPath(name).name])
        call.args[0] = ast.List(elts=[ast.Constant(item) for item in argv], ctx=ast.Load())
    # Leave target evidence present, just as in the reviewer's counterexample.
    tree.body.append(ast.Expr(value=ast.Constant(target)))
    assert missing_launch_sites(
        {caller: ast.unparse(ast.fix_missing_locations(tree)).encode()},
        {name: PROCESS_LAUNCH_SITES[name]},
    ) == [name]


def test_allowlisted_mains_retain_entry_points_and_process_launch_sites() -> None:
    repo = Path(__file__).resolve().parents[1]
    sources = {
        path.relative_to(repo / "src").as_posix(): path.read_bytes()
        for path in (repo / "src" / "stateframe").rglob("*.py")
    }
    sources.update({name: (repo / name).read_bytes()
                    for name in ALLOWLIST if name.startswith("scripts/")})
    unlisted = unlisted_mains(sources)
    assert not unlisted, f"unreviewed main blocks: {unlisted}"
    stale = [name for name in ALLOWLIST if name not in sources or not any(
        _is_main_guard(node) for node in ast.walk(ast.parse(sources[name], filename=name)))]
    assert not stale, f"allowlisted entry points missing their main block: {stale}"

    process_entries = {name for name, purpose in ALLOWLIST.items() if purpose.endswith("process")}
    assert process_entries == set(PROCESS_LAUNCH_SITES), "child processes need reviewed launch sites"
    missing = missing_launch_sites(sources, PROCESS_LAUNCH_SITES)
    assert not missing, f"allowlisted child processes missing launch sites: {missing}"


def test_shipped_main_allowlist(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "_build"))
    import sf_backend

    wheel = tmp_path / sf_backend.build_wheel(str(tmp_path))
    sdist = tmp_path / sf_backend.build_sdist(str(tmp_path))
    wheel_sources, sdist_sources = _artifact_sources(wheel, sdist)
    assert wheel_sources and sdist_sources
    wheel_violations = unlisted_mains(wheel_sources)
    sdist_violations = unlisted_mains(sdist_sources)
    assert not wheel_violations, f"wheel has unreviewed main blocks: {wheel_violations}"
    assert not sdist_violations, f"sdist has unreviewed main blocks: {sdist_violations}"
