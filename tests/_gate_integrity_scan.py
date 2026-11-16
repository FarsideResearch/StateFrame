"""Gate and harness integrity scans: collection census, parity selection, shell exit-status lint.

Used by tests/test_regfam_f8.py. The class this guards: a gate that reports PASS without having
checked what it claims (a test file never collected, a parity test never selected by the release
matrix, a pipeline whose exit status is the exit status of ``tail``, a count of zero read as success).

Also a small CLI so gate scripts outside this repository can reuse the same rules:
    python3 tests/_gate_integrity_scan.py shell-lint <file-or-dir>...   # exit 1 on findings
    python3 tests/_gate_integrity_scan.py parity-files                  # parity test files, one per line
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------------------
# Collection census
# --------------------------------------------------------------------------------------------------

# Directories of recorded or fixture repositories: their test_*.py files are data that StateFrame runs
# inside an engine run, never tests of this repository. Reviewed; each pattern must still match a file.
FIXTURE_DATA_PATTERNS: dict[str, str] = {
    "tests/fixtures/*": "recorded repository copies used as engine inputs",
    "tests/e2e/*/*": "fixture repositories driven by the top-level e2e tests",
    "tests/acceptance/fixtures/*": "fixture repositories driven by the acceptance tests",
    "tests/replay_corpus/_repos/*": "recorded repositories replayed by the corpus tests",
}

PARITY_TOKENS = ("off_wire", "byte_identical", "parity")


def is_fixture_data(relative: str) -> bool:
    return any(fnmatch(relative, pattern) for pattern in FIXTURE_DATA_PATTERNS)


def test_files_on_disk(root: Path) -> list[str]:
    """Every tests/**/test_*.py below ``root`` (tracked or not), as root-relative POSIX paths."""
    found = []
    for path in (root / "tests").rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        found.append(path.relative_to(root).as_posix())
    return sorted(found)


@dataclass(frozen=True)
class Census:
    returncode: int
    items: tuple[dict, ...]
    reports: tuple[dict, ...]
    output: str

    @property
    def files(self) -> set[str]:
        return {item["file"] for item in self.items}


def collect(root: Path, timeout: float) -> Census:
    """Run the default suite collection (``pytest --collect-only tests``) in a fresh process."""
    with tempfile.TemporaryDirectory(prefix="sf-census-") as scratch:
        target = Path(scratch) / "census.json"
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(root / "src"), str(Path(__file__).resolve().parent)]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "SF_COLLECTION_CENSUS": str(target),
        }
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
             "-p", "_collection_census_plugin", "--rootdir", str(root), "tests"],
            cwd=root, env=environment, capture_output=True, text=True, timeout=timeout, check=False,
        )
        output = (result.stdout + result.stderr)[-6000:]
        if not target.is_file():
            raise AssertionError(f"collection wrote no census (pytest exit {result.returncode}):\n{output}")
        data = json.loads(target.read_text(encoding="utf-8"))
    return Census(result.returncode, tuple(data["items"]), tuple(data["reports"]), output)


def collection_problems(root: Path, census: Census) -> list[str]:
    """Files that should run but yielded nothing, fixture files that leaked in, and collect errors."""
    problems: list[str] = []
    if census.returncode != 0:
        problems.append(f"collection exited {census.returncode} (0 required)")
    if not census.items:
        problems.append("collection produced 0 items")
    errors = [report for report in census.reports if report["outcome"] == "failed"]
    for report in errors:
        problems.append(f"collection error in {report['nodeid']}: {report['detail'].splitlines()[-1:]}")
    skipped_files = {report["nodeid"].split("::", 1)[0] for report in census.reports
                     if report["outcome"] == "skipped"}
    collected = census.files
    on_disk = test_files_on_disk(root)
    missing = [path for path in on_disk
               if not is_fixture_data(path) and path not in collected and path not in skipped_files]
    if missing:
        problems.append(f"{len(missing)} test file(s) never collected by `pytest tests`: {missing}")
    leaked = sorted({path for path in collected if is_fixture_data(path)})
    if leaked:
        problems.append(f"fixture-data files were collected as tests: {leaked}")
    stale = [pattern for pattern in FIXTURE_DATA_PATTERNS
             if not any(fnmatch(path, pattern) for path in on_disk)]
    if stale:
        problems.append(f"fixture-data patterns that match no test file (remove them): {stale}")
    return problems


# --------------------------------------------------------------------------------------------------
# Parity selection
# --------------------------------------------------------------------------------------------------

def _base_nodeid(nodeid: str) -> str:
    return nodeid.split("[", 1)[0]


def is_parity_item(item: dict) -> bool:
    text = " ".join([item["nodeid"].lower(), *(name.lower() for name in item["markers"])])
    return any(token in text for token in PARITY_TOKENS)


def parity_tests(census: Census) -> list[str]:
    """Parity tests by base node id (parametrized cases folded into their function)."""
    return sorted({_base_nodeid(item["nodeid"]) for item in census.items if is_parity_item(item)})


def release_matrix_selection(root: Path) -> set[str]:
    """Base node ids the reviewed release matrix (tests/must_run.json) selects."""
    manifest = json.loads((root / "tests" / "must_run.json").read_text(encoding="utf-8"))
    selected: set[str] = set()
    for cell in manifest["cells"]:
        for nodeid in (cell["nodeid"], *cell.get("item_nodeids", ())):
            selected.add(_base_nodeid(nodeid))
    return selected


def is_selected(nodeid: str, selection: set[str]) -> bool:
    file_part = nodeid.split("::", 1)[0]
    return nodeid in selection or file_part in selection


# --------------------------------------------------------------------------------------------------
# Shell lint
# --------------------------------------------------------------------------------------------------

RULE_PIPE_STATUS = "pipe-status-lost"     # `cmd | tail` decides an exit status without pipefail
RULE_ZERO_IS_PASS = "zero-counted-as-pass"  # pytest exit 5 or an empty comparison read as success

_SINK = re.compile(r"(?<![|])\|(?![|])\s*(tail|head|tee)\b")
_PIPEFAIL_ON = re.compile(r"\bset\s+(-[A-Za-z]*o\s+pipefail\b|-o\s+pipefail\b)")
_PIPEFAIL_OFF = re.compile(r"\bset\s+\+o\s+pipefail\b")
_ERREXIT_ON = re.compile(r"\bset\s+(-[A-Za-z]*e[A-Za-z]*\b|-o\s+errexit\b)")
_CONDITION = re.compile(r"^\s*(?:then\s+|do\s+|else\s+)?(if|elif|while|until|!)\s")
_HEREDOC = re.compile(r"(?<!<)<<-?(?!<)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_PYTEST_FIVE = re.compile(
    r"(\$\?|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)\s*(-eq|==|=)\s*['\"]?5\b|^\s*\(?\s*(0\|5|5\|0)\s*\)")
_FAILURE_WORDS = re.compile(r"\b(fail|FAIL|die|false|exit\s+[1-9]|return\s+[1-9])")
_VERDICT = re.compile(r"\b(echo|printf)\b[^\n]*\b(PASS|IDENTICAL|MATCH(ES)?|SAME)\b")
_COMPARE = re.compile(r"\b(cmp|diff|comm)\b")
# A count or size actually tested (printing `wc -l` next to the verdict is not a guard; ``$? -eq 0`` is not one).
_NONEMPTY_GUARD = re.compile(r"\[\[?\s+-s\s|\btest\s+-s\b|-(gt|ne)\s+0\b|-ge\s+1\b|-(lt|le)\s+1\b|-(eq|le)\s+0\b")
_EXIT_STATUS_TEST = re.compile(r"\$\?\s+-(eq|ne)\s+0\b")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    text: str
    why: str

    def key(self) -> tuple[str, str, str]:
        return (self.path, self.rule, self.text)

    def render(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.why}: {self.text}"


def _mask_line(line: str, in_single: bool) -> tuple[str, bool]:
    """Blank single-quoted text and drop comments; returns the masked line and the open-quote state."""
    out = []
    in_double = False
    previous = " "
    for char in line:
        if in_single:
            if char == "'":
                in_single = False
                out.append(char)
            else:
                out.append(" ")
        elif char == "'" and not in_double:
            in_single = True
            out.append(char)
        elif char == '"' and previous != "\\":
            in_double = not in_double
            out.append(char)
        elif char == "#" and not in_double and previous in " \t;":
            break
        else:
            out.append(char)
        previous = char
    return "".join(out), in_single


def logical_lines(text: str) -> list[tuple[int, str]]:
    """(first line number, masked logical line): continuations joined, comments and heredoc bodies removed."""
    lines = text.splitlines()
    result: list[tuple[int, str]] = []
    index = 0
    in_single = False
    pending: list[str] = []
    start = 1
    while index < len(lines):
        raw = lines[index]
        if not pending:
            start = index + 1
        masked, in_single = _mask_line(raw, in_single)
        # The delimiter may be quoted ('EOF'), which the mask blanks: read it from the raw line.
        heredoc = _HEREDOC.search(raw) if "<<" in masked else None
        if heredoc:
            delimiter = heredoc.group(2)
            body = index + 1
            while body < len(lines) and lines[body].strip() != delimiter:
                body += 1
            pending.append(masked)
            result.append((start, " ".join(pending)))
            pending = []
            index = body + 1
            continue
        if masked.rstrip().endswith("\\") or in_single:
            pending.append(masked.rstrip().rstrip("\\"))
        else:
            pending.append(masked)
            result.append((start, " ".join(pending)))
            pending = []
        index += 1
    if pending:
        result.append((start, " ".join(pending)))
    return [(number, line) for number, line in result if line.strip()]


def _operator_after(text: str) -> str:
    match = re.search(r"\|\||&&|;|\)|`", text)
    return match.group(0) if match else ""


def shell_findings(text: str, path: str, *, embedded: bool = False) -> list[Finding]:
    """Lint one shell script (or one snippet passed to ``sh -c``)."""
    findings: list[Finding] = []
    statements = logical_lines(text)
    pipefail = False
    errexit = bool(re.match(r"#!\S*\s+-[A-Za-z]*e", text))
    mentions_pytest = "pytest" in text
    has_guard = bool(_NONEMPTY_GUARD.search(_EXIT_STATUS_TEST.sub(" ", text)))
    for position, (number, line) in enumerate(statements):
        following = statements[position + 1][1] if position + 1 < len(statements) else ""
        if _PIPEFAIL_OFF.search(line):
            pipefail = False
        if _PIPEFAIL_ON.search(line):
            pipefail = True
        if _ERREXIT_ON.search(line):
            errexit = True
        stripped = line.strip()
        for match in _SINK.finditer(line):
            if pipefail or "PIPESTATUS" in line or "PIPESTATUS" in following:
                continue
            reason = None
            if _CONDITION.match(line):
                reason = "pipeline is a condition"
            elif _operator_after(line[match.end():]) in ("&&", "||"):
                reason = "pipeline status feeds && / ||"
            elif "$?" in line[match.end():] or "$?" in following:
                reason = "pipeline status is read through $?"
            elif errexit:
                reason = "set -e decides on the last stage"
            elif position == len(statements) - 1:
                reason = "pipeline status is the " + ("snippet" if embedded else "script") + " exit status"
            if reason:
                findings.append(Finding(path, number, RULE_PIPE_STATUS, stripped,
                                        f"`| {match.group(1)}` hides the producer's exit code ({reason})"))
                break
        if mentions_pytest and _PYTEST_FIVE.search(line) and not _FAILURE_WORDS.search(line):
            findings.append(Finding(path, number, RULE_ZERO_IS_PASS, stripped,
                                    "pytest exit 5 (nothing collected) accepted"))
        if not has_guard and _COMPARE.search(line) and _VERDICT.search(line):
            findings.append(Finding(path, number, RULE_ZERO_IS_PASS, stripped,
                                    "comparison verdict with no non-empty input check (0 vs 0 passes)"))
    return findings


# Python that shells out ---------------------------------------------------------------------------

_SHELL_CALLS = {"run", "Popen", "call", "check_call", "check_output", "system", "popen",
                "getoutput", "getstatusoutput"}
_SHELLS = {"sh", "bash", "zsh", "dash"}


def _string_value(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(part.value if isinstance(part, ast.Constant) else "X" for part in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _string_value(node.left, constants), _string_value(node.right, constants)
        if left is not None or right is not None:
            return (left or "X") + (right or "X")
        return None
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _module_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for statement in tree.body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            value = _string_value(statement.value, constants)
            if value is not None:
                constants[statement.targets[0].id] = value
    return constants


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def python_shell_snippets(source: str) -> list[tuple[int, str]]:
    """(line, shell text) for each subprocess / os.system call that runs a shell over a known string."""
    tree = ast.parse(source)
    constants = _module_constants(tree)
    snippets: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node.func) not in _SHELL_CALLS or not node.args:
            continue
        first = node.args[0]
        shell_kw = any(keyword.arg == "shell" and isinstance(keyword.value, ast.Constant)
                       and keyword.value.value is True for keyword in node.keywords)
        if shell_kw or _call_name(node.func) in {"system", "popen", "getoutput", "getstatusoutput"}:
            text = _string_value(first, constants)
            if text is not None:
                snippets.append((node.lineno, text))
            continue
        if isinstance(first, (ast.List, ast.Tuple)) and len(first.elts) >= 3:
            program = _string_value(first.elts[0], constants)
            if program is None or PurePosixPath(program).name not in _SHELLS:
                continue
            for index, element in enumerate(first.elts[1:-1], start=1):
                flag = _string_value(element, constants)
                if flag and flag.startswith("-") and flag.endswith("c"):
                    text = _string_value(first.elts[index + 1], constants)
                    if text is not None:
                        snippets.append((node.lineno, text))
                    break
    return snippets


def _zero_exit_container(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return False
    values = set()
    for element in node.elts:
        if isinstance(element, ast.Constant):
            values.add(element.value)
        elif isinstance(element, ast.Attribute):
            values.add(element.attr)
    return bool(values & {0, "OK"}) and bool(values & {5, "NO_TESTS_COLLECTED"})


_SHELL_HINT = re.compile(r"shell\s*=\s*True|['\"]-[A-Za-z]*c['\"]|\b(system|popen|getoutput|getstatusoutput)\(")
_EXIT_FIVE_HINT = re.compile(r"\b5\b|NO_TESTS_COLLECTED")


def python_findings(source: str, path: str) -> list[Finding]:
    findings: list[Finding] = []
    # Parsing every engine module costs ~35 s; only files that can hold a finding are parsed.
    shells_out = bool(_SHELL_HINT.search(source))
    exit_five = "pytest" in source and bool(_EXIT_FIVE_HINT.search(source))
    if not (shells_out or exit_five):
        return findings
    lines = source.splitlines()
    for number, snippet in python_shell_snippets(source) if shells_out else ():
        for finding in shell_findings(snippet, path, embedded=True):
            findings.append(Finding(path, number, finding.rule, finding.text, finding.why))
    if exit_five:
        for node in ast.walk(ast.parse(source)):
            if (isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops)
                    and any(_zero_exit_container(c) for c in node.comparators)):
                findings.append(Finding(path, node.lineno, RULE_ZERO_IS_PASS, lines[node.lineno - 1].strip(),
                                        "exit 0 and exit 5 (nothing collected) accepted together"))
    return findings


# Tree walk -----------------------------------------------------------------------------------------

_SKIP_DIRS = {".git", "__pycache__", "node_modules", "wheelhouse", "_build", ".venv", "venv"}


def _is_shell_file(path: Path) -> bool:
    """*.sh / *.bash, or an extensionless file whose shebang names a shell (``#!/usr/bin/env bash``)."""
    if path.suffix in {".sh", ".bash"}:
        return True
    if path.suffix:
        return False
    try:
        with path.open("rb") as handle:
            first = handle.readline(200)
    except OSError:
        return False
    words = first[2:].decode("latin-1").split() if first.startswith(b"#!") else []
    names = [PurePosixPath(word).name for word in words if not word.startswith("-")]
    return bool(names) and (names[0] in _SHELLS or (names[0] == "env" and names[1:2] and names[1] in _SHELLS))


def _is_embedded_repository(relative: str) -> bool:
    """Recorded repositories under tests/replay_corpus (``repo`` / ``_repos``) are data, as in its conftest."""
    parts = PurePosixPath(relative).parts
    if "replay_corpus" not in parts:
        return False
    return any(part in {"repo", "_repos"} for part in parts[parts.index("replay_corpus") + 1:])


def iter_scan_files(root: Path) -> list[Path]:
    files = []
    for directory, subdirectories, names in os.walk(root):
        subdirectories[:] = sorted(name for name in subdirectories if name not in _SKIP_DIRS)
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if is_fixture_data(relative) or _is_embedded_repository(relative):
                continue
            if path.suffix == ".py" or _is_shell_file(path):
                files.append(path)
    return files


def scan_path(path: Path, display: str) -> list[Finding]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    if path.suffix == ".py":
        try:
            return python_findings(text, display)
        except SyntaxError:
            return []
    return shell_findings(text, display)


def scan_tree(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in iter_scan_files(root):
        findings.extend(scan_path(path, path.relative_to(root).as_posix()))
    return findings


def main(argv: list[str]) -> int:
    if argv[:1] == ["parity-files"]:
        from _timeouts import scaled  # noqa: PLC0415 -- the CLI runs from the repository's tests/
        for path in sorted({nodeid.split("::", 1)[0] for nodeid in parity_tests(collect(ROOT, scaled(300)))}):
            print(path)
        return 0
    if argv[:1] == ["shell-lint"] and len(argv) > 1:
        findings: list[Finding] = []
        for raw in argv[1:]:
            target = Path(raw)
            paths = iter_scan_files(target) if target.is_dir() else [target]
            for path in paths:
                findings.extend(scan_path(path, str(path)))
        for finding in findings:
            print(finding.render())
        return 1 if findings else 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
