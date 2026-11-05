"""Frozen release guard against the shared lab database and Docker side effects."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shlex
import socket
import subprocess


_FORBIDDEN_PORT = 55432
_FORBIDDEN_PORT_TEXT = "55432"
_LAB_CONTAINER = "mission-control-db"
_DOCKER = "docker"


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal_argv0(node: ast.AST) -> str | None:
    """Read argv[0] only when it is statically present in a literal expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            words = shlex.split(node.value)
        except ValueError:
            return None
        return words[0] if words else None
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        return _literal_argv0(node.elts[0])
    if isinstance(node, ast.Call) and _dotted_name(node.func) == "shlex.split" and node.args:
        return _literal_argv0(node.args[0])
    return None


def _is_docker_program(program: object) -> bool:
    try:
        return os.path.basename(os.fspath(program)) == _DOCKER
    except TypeError:
        return False


def _program_candidates(call: ast.Call) -> list[ast.AST]:
    candidates: list[ast.AST] = []
    if call.args:
        candidates.append(call.args[0])

    # execv/execvp receive both a program and an argv array; inspect both.
    function_name = _dotted_name(call.func).rsplit(".", 1)[-1]
    if function_name in {"execv", "execve", "execvp", "execvpe"} and len(call.args) > 1:
        candidates.append(call.args[1])

    # argv is sometimes supplied by keyword. Check literal sequences at any argument
    # position as well as conventional command/program keyword names.
    for value in (*call.args, *(keyword.value for keyword in call.keywords)):
        if isinstance(value, (ast.List, ast.Tuple)):
            candidates.append(value)
        elif (isinstance(value, ast.Call)
              and _dotted_name(value.func) == "shlex.split"):
            candidates.append(value)
    for keyword in call.keywords:
        if keyword.arg in {"args", "argv", "cmd", "command", "executable", "file", "path"}:
            candidates.append(keyword.value)
    return candidates


def _is_preserved_namespace_default(
    path: Path, node: ast.Constant, parents: dict[ast.AST, ast.AST],
) -> bool:
    """Allow the one source-only default in the protected semantics module."""
    if path.parts[-3:] != ("stateframe", "engine", "kernel_semantics.py"):
        return False
    if type(node.value) is not int or node.value != _FORBIDDEN_PORT:
        return False
    expression = parents.get(node)
    if not (isinstance(expression, ast.BoolOp) and isinstance(expression.op, ast.Or)
            and len(expression.values) == 2 and expression.values[1] is node):
        return False
    source = expression.values[0]
    if not (isinstance(source, ast.Attribute) and source.attr == "port"
            and isinstance(source.value, ast.Name) and source.value.id == "analysis"):
        return False
    assignment = parents.get(expression)
    if not (isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and assignment.targets[0].id == "actual_port"):
        return False
    method = parents.get(assignment)
    owner = parents.get(method)
    return (isinstance(method, ast.FunctionDef) and method.name == "_derive"
            and isinstance(owner, ast.ClassDef) and owner.name == "NamespaceMatrixProducer")


def _scan_src(source_root: Path) -> list[str]:
    """Return every forbidden literal or Docker program call with source locations."""
    findings: set[str] = set()
    display_root = source_root.parent
    for path in sorted(source_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            line = getattr(error, "lineno", None) or 1
            findings.add(f"{path.relative_to(display_root)}:{line}: cannot scan Python source: {error}")
            continue

        relative_path = path.relative_to(display_root)
        parents = {child: parent for parent in ast.walk(tree)
                   for child in ast.iter_child_nodes(parent)}
        preserved_defaults = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                value = node.value
                if type(value) is int and value == _FORBIDDEN_PORT:
                    if _is_preserved_namespace_default(relative_path, node, parents):
                        preserved_defaults += 1
                    else:
                        findings.add(
                            f"{relative_path}:{node.lineno}: forbidden port literal {_FORBIDDEN_PORT}"
                        )
                elif isinstance(value, str):
                    if _FORBIDDEN_PORT_TEXT in value:
                        findings.add(
                            f"{relative_path}:{node.lineno}: forbidden port text {_FORBIDDEN_PORT_TEXT!r}"
                        )
                    if _LAB_CONTAINER in value:
                        findings.add(
                            f"{relative_path}:{node.lineno}: forbidden container reference {_LAB_CONTAINER!r}"
                        )
            elif isinstance(node, ast.Call):
                if any(_is_docker_program(_literal_argv0(candidate))
                       for candidate in _program_candidates(node)):
                    findings.add(
                        f"{relative_path}:{node.lineno}: Docker program/argv[0] literal {_DOCKER!r}"
                    )
        if preserved_defaults > 1:
            findings.add(f"{relative_path}: duplicate protected namespace defaults")
    return sorted(findings)


def test_src_has_no_lab_database_or_docker_literals():
    source_root = Path(__file__).resolve().parents[1] / "src"
    findings = _scan_src(source_root)
    assert not findings, "forbidden lab database/container references in src/:\n" + "\n".join(
        f"- {finding}" for finding in findings
    )


def test_protected_namespace_default_does_not_hide_another_port_literal(tmp_path: Path):
    path = tmp_path / "src/stateframe/engine/kernel_semantics.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "class NamespaceMatrixProducer:\n"
        "    def _derive(self, analysis):\n"
        "        actual_port = analysis.port or 55432\n"
        "        return actual_port\n"
        "OTHER_PORT = 55432\n",
        encoding="utf-8",
    )
    findings = _scan_src(tmp_path / "src")
    assert len(findings) == 1
    assert "forbidden port literal" in findings[0]


def _argv0(command: object) -> object | None:
    if isinstance(command, str):
        try:
            words = shlex.split(command)
        except ValueError:
            return None
        return words[0] if words else None
    if isinstance(command, (list, tuple)) and command:
        candidate = command[0]
        try:
            return os.fspath(candidate)
        except TypeError:
            return candidate
    return None


def _target_port(address: object) -> int | None:
    if isinstance(address, (list, tuple)) and len(address) > 1:
        try:
            return int(address[1])
        except (TypeError, ValueError):
            return None
    if isinstance(address, str):
        try:
            return int(address.rsplit(":", 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def test_real_selector_runner_never_probes_lab_db_or_spawns_docker(
    monkeypatch, tmp_path: Path,
):
    import kernel_execution_policy
    from stateframe.engine.probes import react_repair

    # A valid selector reaches the real row=None runner path. That path calls the
    # real _ensure_db today; all external effects are intercepted below.
    root = tmp_path / "repo"
    root.mkdir()
    (root / "test_guard_target.py").write_text(
        "def test_guard_target():\n    assert True\n", encoding="utf-8"
    )

    socket_attempts: list[tuple[str, object]] = []
    process_attempts: list[tuple[str, object, object | None]] = []

    def refuse_create_connection(address, *args, **kwargs):
        socket_attempts.append(("socket.create_connection", address))
        raise OSError("connection blocked by no-lab-db test")

    def refuse_socket_connect(_sock, address):
        socket_attempts.append(("socket.socket.connect", address))
        raise OSError("connection blocked by no-lab-db test")

    def refuse_process_api(api_name: str, command: object):
        program = _argv0(command)
        process_attempts.append((api_name, command, program))
        raise OSError("process launch blocked by no-lab-db test")

    def refuse_run(command=None, *args, **kwargs):
        if command is None:
            command = kwargs.get("args")
        return refuse_process_api("subprocess.run", command)

    def refuse_popen(command=None, *args, **kwargs):
        if command is None:
            command = kwargs.get("args")
        return refuse_process_api("subprocess.Popen", command)

    def refuse_execvp(program, argv, *args, **kwargs):
        process_attempts.append(("os.execvp", argv, program or _argv0(argv)))
        raise OSError("process launch blocked by no-lab-db test")

    monkeypatch.setattr(socket, "create_connection", refuse_create_connection)
    monkeypatch.setattr(socket.socket, "connect", refuse_socket_connect)
    monkeypatch.setattr(subprocess, "run", refuse_run)
    monkeypatch.setattr(subprocess, "Popen", refuse_popen)
    monkeypatch.setattr(os, "execvp", refuse_execvp)
    # Today's _ensure_db polls 45 times after its refused recovery attempt.
    monkeypatch.setattr(react_repair.time, "sleep", lambda *_args, **_kwargs: None)

    # Keep a future compliant path from running the temporary pytest target. This
    # seam is reached after the real _ensure_db call; the socket and process APIs
    # above still record and refuse any direct side effect.
    def refuse_sandboxed_check(*args, **kwargs):
        raise RuntimeError("sandboxed pytest blocked by no-lab-db test")

    monkeypatch.setattr(kernel_execution_policy, "run_sandboxed", refuse_sandboxed_check)

    react_repair.run_tests_for_selectors(
        root, "guard_target", ("test_guard_target.py",), row=None
    )

    forbidden_socket_attempts = [
        address for _api, address in socket_attempts
        if _target_port(address) == _FORBIDDEN_PORT
    ]
    docker_process_attempts = [
        (api, command, program)
        for api, command, program in process_attempts
        if _is_docker_program(program) or _is_docker_program(_argv0(command))
    ]
    violations = []
    if forbidden_socket_attempts:
        violations.append(
            f"port {_FORBIDDEN_PORT} targets={len(forbidden_socket_attempts)}, "
            f"first={forbidden_socket_attempts[0]!r}"
        )
    if docker_process_attempts:
        violations.append(f"Docker process attempts={docker_process_attempts!r}")
    assert not violations, "runtime side-effect guard failed: " + "; ".join(violations)


def test_built_wheel_and_sdist_have_no_lab_drivers_or_side_effects(tmp_path: Path):
    """Inspect the actual artifacts, not just the source tree."""
    import sys
    import tarfile
    import zipfile

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "_build"))
    import sf_backend

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    wheel = artifacts / sf_backend.build_wheel(str(artifacts))
    sdist = artifacts / sf_backend.build_sdist(str(artifacts))
    with zipfile.ZipFile(wheel) as archive:
        wheel_sources = {name: archive.read(name) for name in archive.namelist()
                         if name.endswith(".py")}
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_sources = {name: archive.extractfile(name).read()
                         for name in archive.getnames()
                         if name.endswith(".py") and "/src/stateframe/" in name}
    assert wheel_sources and sdist_sources
    banned = {"converge.py", "pilot.py", "noncoding_harness.py"}
    for artifact_name, sources in (("wheel", wheel_sources), ("sdist", sdist_sources)):
        for name, content in sources.items():
            parts = Path(name).parts
            assert not ("engine" in parts and parts[-1] in banned), (artifact_name, name)
            source = content.decode("utf-8")
            tree = ast.parse(source, filename=name)
            parents = {child: parent for parent in ast.walk(tree)
                       for child in ast.iter_child_nodes(parent)}
            preserved_defaults = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant):
                    if _FORBIDDEN_PORT_TEXT in str(node.value):
                        assert _is_preserved_namespace_default(Path(name), node, parents), (
                            artifact_name, name, node.lineno)
                        preserved_defaults += 1
                    assert _LAB_CONTAINER not in str(node.value), (artifact_name, name, node.lineno)
                if isinstance(node, ast.Call):
                    assert not any(_is_docker_program(_literal_argv0(candidate))
                                   for candidate in _program_candidates(node)), (artifact_name, name, node.lineno)
            if parts[-1] == "react_repair.py":
                assert not any(isinstance(node, ast.If)
                               and isinstance(node.test, ast.Compare)
                               and isinstance(node.test.left, ast.Name)
                               and node.test.left.id == "__name__"
                               and any(isinstance(value, ast.Constant) and value.value == "__main__"
                                       for value in node.test.comparators)
                               for node in tree.body), (artifact_name, name)
            assert preserved_defaults <= 1, (artifact_name, name, "duplicate namespace defaults")
