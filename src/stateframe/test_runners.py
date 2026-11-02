"""Runner classification and protected test resources for repository checks."""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
from pathlib import Path, PurePosixPath

from .workcopy import IGNORE_DIRS


PYTEST_PATTERNS = ("test_*.py", "conftest.py")
DECLARED_TEST_SOURCE = "tests"
GENERIC_PATTERNS = (
    "test_*.py", "*_test.py", "conftest.py", "*.test.*", "*.spec.*",
    "*_test.go", "*.test.sh", "test*.sh", "check*.sh",
    "tests/**", "test/**", "__tests__/**",
    "package.json", "Makefile", "makefile", "GNUmakefile", "Cargo.toml",
    "vitest.config.*", "jest.config.*", "playwright.config.*",
)
TEST_PATTERNS = ("test_*.py", "*_test.py", "conftest.py", "*.test.*", "*.spec.*",
                 "*_test.go", "*.test.sh", "test*.sh", "check*.sh", "tests/**", "test/**", "__tests__/**")


def is_test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return any(fnmatch.fnmatchcase(path, pattern) if "/" in pattern else
               fnmatch.fnmatchcase(name, pattern) for pattern in TEST_PATTERNS)


def judging_test_paths(paths: tuple[str, ...], command: tuple[str, ...],
                       checks: tuple[str, ...]) -> tuple[str, ...]:
    """Only explicit path operands name judging files; package names never do."""
    named = set(checks)
    for operand in command[1:]:
        candidate = operand.split("::", 1)[0].removeprefix("./")
        if candidate in paths and is_test_file(candidate):
            named.add(candidate)
        elif candidate.rstrip("/") in {"tests", "test", "__tests__"}:
            named.update(path for path in paths if path.startswith(candidate.rstrip("/") + "/"))
    return tuple(sorted(named & set(paths)))


def judging_test_selection(paths: tuple[str, ...], command: tuple[str, ...],
                           *, failed_output: str = "") -> dict[str, list[str]]:
    """Select judging files and explicit pytest node names for wrong-test evidence."""
    selected = set(judging_test_paths(paths, command, ()))
    nodes: dict[str, list[str]] = {}
    for arg in command[1:]:
        path, *parts = arg.removeprefix("./").split("::")
        if path in selected and parts:
            nodes.setdefault(path, []).append("::".join(parts))
    executable = PurePosixPath(command[0]).name.lower() if command else ""
    discovery = (executable in {"pytest", "py.test", "jest", "vitest"}
                 or (re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", executable)
                     and command[1:3] in {("-m", "pytest"), ("-m", "unittest")})
                 or (executable == "node" and "--test" in command)
                 or (executable in {"npm", "yarn", "pnpm", "bun"}
                     and (command[1:2] == ("test",) or command[1:3] == ("run", "test")))
                 or (executable == "npx" and command[1:2] in {("jest",), ("vitest",)})
                 or (executable in {"go", "cargo"} and command[1:2] == ("test",)))
    if not selected and discovery:
        selected.update(paths)  # these runners can discover tests without a file operand
    for path, failures in failing_test_nodes(paths, failed_output).items():
        if path not in selected:
            selected.add(path)
            nodes[path] = failures
    return {path: nodes.get(path, []) for path in sorted(selected)}


def failing_test_nodes(paths: tuple[str, ...], output: str) -> dict[str, list[str]]:
    return {path: sorted(set(matches)) for path in paths if (matches := re.findall(
        rf"(?m)^.*\b(?:FAILED|ERROR|FAIL)\b.*(?:\./)?{re.escape(path)}::([\w:\[\].-]+)", output))}


_ANSI = re.compile(r"\x1b\[[\d;]*m")
_PYTEST_RESULT = re.compile(
    r"^\s*(?:=+\s*)?(?P<counts>\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|"
    r"deselected|warnings?|reruns?)(?:, \d+ (?:passed|failed|errors?|skipped|xfailed|"
    r"xpassed|deselected|warnings?|reruns?))*) in \S+(?:\s*=+)?\s*$")
_COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?|reruns?)\b")
_PYTEST_NONRUN_FLAGS = frozenset({"--collect-only", "--co", "--version", "-V", "--help", "-h"})


def _runner_for_command(kind: str, command: tuple[str, ...], repo: Path | None) -> str | None:
    if not command:
        return None
    executable = PurePosixPath(command[0]).name.lower()
    args = command[1:]
    if executable == "env":
        offset = 0
        while offset < len(args) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", args[offset]):
            offset += 1
        return _runner_for_command(kind, args[offset:], repo)
    if executable in {"uv", "poetry", "pipenv"} and args[:1] == ("run",):
        return _runner_for_command(kind, args[1:], repo)
    if kind == "pytest":
        module_args = args
        while module_args[:1] in {("-B",), ("-I",), ("-s",), ("-u",), ("-E",)}:
            module_args = module_args[1:]
        if executable in {"pytest", "py.test"} or (
                re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", executable)
                and module_args[:2] == ("-m", "pytest")):
            return "pytest" if not any(arg in _PYTEST_NONRUN_FLAGS or
                                        arg.startswith("--collect-only=") for arg in args) else None
        return None
    if kind != "generic":
        return None
    if executable in {"npm", "yarn", "pnpm"} and (args[:1] == ("test",) or
            args[:2] == ("run", "test")):
        if repo is None:
            return None
        try:
            package = json.loads((repo / "package.json").read_text(encoding="utf-8"))
            script = package["scripts"]["test"]
            if not isinstance(script, str):
                return None
            resolved = tuple(shlex.split(script))
        except (OSError, UnicodeError, ValueError, KeyError, TypeError):
            return None
        # A package script is a shell program. Only a direct runner invocation
        # can establish which program emitted the receipt.
        if any(re.search(r"[;&|<>`$\n]", arg) for arg in resolved):
            return None
        if resolved and PurePosixPath(resolved[0]).name.lower() in {"npm", "yarn", "pnpm"}:
            return None
        return _runner_for_command("generic", resolved, repo)
    if executable in {"node", "node.exe"} and "--test" in args:
        return "node"
    if executable in {"jest", "vitest"}:
        return "package"
    if executable == "npx" and args[:1] in {("jest",), ("vitest",)}:
        return "package"
    if executable == "go" and args[:1] == ("test",):
        return "go"
    if executable == "cargo" and args[:1] == ("test",):
        return "cargo"
    return None


def _parsed_passes(runner: str, output: str) -> int:
    lines = [line.strip() for line in _ANSI.sub("", output).splitlines() if line.strip()]
    if not lines:
        return 0
    if runner == "pytest":
        match = _PYTEST_RESULT.fullmatch(lines[-1])
        if match is None:
            return 0
        counts = {name: int(count) for count, name in _COUNT.findall(match.group("counts"))}
        return counts.get("passed", 0) if not any(counts.get(name, 0) for name in
                                                   ("failed", "error", "errors")) else 0
    if runner == "node":
        footer = []
        for line in reversed(lines):
            if re.fullmatch(r"[#ℹ]\s*duration_ms\s+\S+", line):
                continue
            if re.fullmatch(r"[#ℹ]\s*(?:tests|suites|pass|fail|error|errors|cancelled|skipped|todo)\s+\d+", line):
                footer.append(line)
                continue
            break
        counts = {name: int(count) for line in reversed(footer)
                  for name, count in re.findall(r"^[#ℹ]\s*(\w+)\s+(\d+)$", line)}
        return counts.get("pass", 0) if "fail" in counts and not any(
            counts.get(name, 0) for name in ("fail", "error", "errors", "cancelled")) else 0
    if runner == "package":
        tail = []
        for line in reversed(lines):
            if re.match(r"^(?:Tests?\b|Test Files\b|Test Suites\b|Snapshots\b|Time\b|"
                        r"Start at\b|Duration\b|Ran all test suites\b)", line):
                tail.append(line)
                continue
            break
        test_lines = [line for line in tail if re.match(r"^Tests(?::|\s)", line)]
        if not test_lines:
            return 0
        counts = {name: int(count) for count, name in _COUNT.findall(test_lines[0])}
        return counts.get("passed", 0) if not any(counts.get(name, 0) for name in
                                                   ("failed", "error", "errors")) else 0
    if runner == "go":
        if not re.fullmatch(r"ok\s+\S+(?:\s+.*)?", lines[-1]) or any(
                re.match(r"(?:--- FAIL:|FAIL\b)", line) for line in lines):
            return 0
        if len(lines) < 3 or lines[-2] != "PASS":
            return 0
        start = max((index for index, line in enumerate(lines[:-2]) if re.match(
            r"(?:ok\s|FAIL\b|\?\s)", line)), default=-1) + 1
        package_lines = lines[start:-2]
        started = {match.group(1) for line in package_lines if (match := re.fullmatch(
            r"=== RUN\s+(\S+)", line))}
        return sum(bool(match and match.group(1) in started) for line in package_lines
                   if (match := re.fullmatch(r"--- PASS: (\S+)(?:\s+.*)?", line)))
    if runner == "cargo":
        match = re.fullmatch(r"test result: ok\. (\d+) passed; (\d+) failed; "
                             r"(\d+) ignored; (\d+) measured; (\d+) filtered out; finished in \S+", lines[-1])
        if match is None:
            return 0
        return int(match.group(1)) if int(match.group(2)) == 0 else 0
    return 0


def command_runs_tests(kind: str, command: tuple[str, ...], paths: set[str],
                       *, output: str = "", repo: Path | None = None) -> bool:
    """Report whether a recognized runner emitted a positive, failure-free receipt."""
    del paths
    runner = _runner_for_command(kind, command, repo)
    return bool(runner and _parsed_passes(runner, output) >= 1)


def declared_test_evidence(test_cmd: str | None, check_cmd: str | None,
                           source: str | None, kind: str) -> bool:
    """An implicit pytest check is the declared default when tests exist."""
    return bool(source == DECLARED_TEST_SOURCE and not check_cmd
                and (test_cmd or kind == "pytest"))


def verification_evidence_for_tests(source: str | None, kind: str, command: tuple[str, ...],
                                    output: str, *, passed: bool, declared: bool,
                                    repo: Path | None = None) -> tuple[str | None, str | None, bool]:
    """Sole authority for a final test source and its human verification label."""
    evidence = bool(source == "tests" and declared and passed and
                    command_runs_tests(kind, command, set(), output=output, repo=repo))
    if evidence:
        return "tests", "verified by tests", True
    return ("command" if source == "tests" else source), None, False


def initial_failing_tests(paths: tuple[str, ...], output: str) -> tuple[str, ...]:
    """Resolve test paths printed by the starting check."""
    return tuple(sorted(path for path in paths if path in output or f"./{path}" in output))


def lock_initial_failing_tests(row: dict[str, object], before: dict[str, object]) -> None:
    """Protect every attributable failure, or every test when attribution is incomplete."""
    tests = tuple(row.get("test_file_paths", ()))
    locked = set(row.get("protected_test_paths", ()))
    output = str(before.get("output") or before.get("tail") or "")
    failed = initial_failing_tests(tests, output)
    if before.get("output_truncated") or not failed:
        failed = tests
    locked.update(failed)
    row["protected_test_paths"] = sorted(locked)
    for path in locked & set(tests):
        row["files"].pop(path, None)


def repository_paths(repo: Path) -> set[str]:
    """List source files without following links or generated dependency trees."""
    files = []
    for root, dirs, names in os.walk(repo, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in IGNORE_DIRS
                         and not (Path(root) / name).is_symlink())
        files.extend((Path(root) / name).relative_to(repo).as_posix()
                     for name in names if not (Path(root) / name).is_symlink())
    return set(files)


def discover_check(repo: Path) -> tuple[str | None, str, bool]:
    """Return a conservative build check only when the repository has no tests.

    The final boolean says whether any test file exists. Discovery never runs a
    project command and never infers a test script from a build script.
    """
    paths = repository_paths(repo)
    has_tests = any(is_test_file(path) for path in paths)
    if has_tests:
        return None, "tests", True
    if any(name in paths for name in ("pyproject.toml", "setup.cfg", "setup.py")):
        sources = sorted(path for path in paths if path.endswith(".py") and not is_test_file(path))
        if sources:
            return "python -m compileall -q " + " ".join(shlex.quote(path) for path in sources), "build", False
    if "package.json" in paths:
        try:
            package = json.loads((repo / "package.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError(f"cannot inspect package.json scripts: {error}") from error
        scripts = package.get("scripts") if isinstance(package, dict) else None
        if isinstance(scripts, dict):
            manager = ("pnpm" if "pnpm-lock.yaml" in paths else
                       "yarn" if "yarn.lock" in paths else
                       "bun" if "bun.lock" in paths or "bun.lockb" in paths else "npm")
            for name in ("typecheck", "build", "lint"):
                if isinstance(scripts.get(name), str) and scripts[name].strip():
                    return f"{manager} run {name}", "build", False
    if "go.mod" in paths:
        return "go build ./...", "build", False
    if "Cargo.toml" in paths:
        return "cargo check", "build", False
    for name in ("Makefile", "makefile", "GNUmakefile"):
        if name in paths:
            try:
                source = (repo / name).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                raise ValueError(f"cannot inspect {name} targets: {error}") from error
            for target in ("check", "build"):
                if re.search(rf"(?m)^{target}\s*:(?![=])", source):
                    return f"make -f {shlex.quote(name)} {target}", "build", False
    return None, "none", False


def parse_command(command: str | None, kind: str = "auto") -> tuple[str, tuple[str, ...]]:
    """Return a fixed runner kind and argv; a shell is never inserted."""
    if kind not in {"auto", "pytest", "generic"}:
        raise ValueError("--runner-kind must be auto, pytest, or generic")
    if command is None:
        return ("pytest" if kind == "auto" else kind), ()
    try:
        argv = tuple(shlex.split(command))
    except ValueError as error:
        raise ValueError(f"--test-cmd could not be parsed: {error}") from error
    if not argv:
        raise ValueError("--test-cmd must contain an executable")
    if any("\x00" in arg for arg in argv):
        raise ValueError("--test-cmd contains a NUL byte")
    if kind == "auto":
        effective = argv
        if PurePosixPath(effective[0]).name.lower() == "env":
            offset = 1
            while offset < len(effective) and re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*=.*", effective[offset]):
                offset += 1
            if offset < len(effective):
                effective = effective[offset:]
        if (len(effective) >= 3 and PurePosixPath(effective[0]).name.lower()
                in {"uv", "poetry", "pipenv"} and effective[1] == "run"):
            effective = effective[2:]
        executable = PurePosixPath(effective[0]).name.lower()
        python_module = (re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", executable)
                         is not None and any(effective[index:index + 2] == ("-m", "pytest")
                                             for index in range(1, len(effective) - 1)))
        kind = ("pytest" if executable in {"pytest", "py.test"} or python_module
                else "generic")
    return kind, argv


def validate_patterns(patterns: list[str]) -> tuple[str, ...]:
    result = []
    for pattern in patterns:
        value = pattern.replace("\\", "/").strip()
        parts = PurePosixPath(value).parts
        if not value or value.startswith("/") or ".." in parts or value.startswith("~"):
            raise ValueError(f"--protected-test-glob must be relative to the repository: {pattern}")
        if value not in result:
            result.append(value)
    return tuple(result)


def protected_paths(paths: list[str] | tuple[str, ...], *, kind: str,
                    extra_patterns: tuple[str, ...] = (),
                    command: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Match original files only; explicit command scripts and test configs stay immutable."""
    patterns = (*(PYTEST_PATTERNS if kind == "pytest" else GENERIC_PATTERNS),
                *extra_patterns)
    available = set(paths)
    protected = set()
    for path in paths:
        name = PurePosixPath(path).name
        if any((fnmatch.fnmatchcase(path, pattern) if "/" in pattern
                else fnmatch.fnmatchcase(name, pattern)) for pattern in patterns):
            protected.add(path)
    if kind == "generic":
        if command:
            executable = command[0].removeprefix("./")
            if executable in available:
                protected.add(executable)
            if len(command) > 1 and PurePosixPath(command[0]).name in {"sh", "bash"}:
                script = command[1].removeprefix("./")
                if script in available:
                    protected.add(script)
    return tuple(sorted(protected))
