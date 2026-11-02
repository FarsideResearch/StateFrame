"""Final test certification and structural bypass regressions."""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from stateframe.test_runners import declared_test_evidence, verification_evidence_for_tests


ROOT = Path(__file__).resolve().parents[1]
FAKE_RECEIPTS = (
    "5 passed in 0.01s\n",
    "Tests: 3 passed, 3 total\n",
    "Tests 3 passed (3)\n",
    "# pass 2\n# fail 0\n",
    "ok  example.com/x 0.1s\n",
    "test result: ok. 4 passed; 0 failed\n",
)


def _violations(source: str) -> list[int]:
    """Find direct writes of a certified source or label in product syntax."""
    tree = ast.parse(source)
    found = []

    def target_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            return node.slice.value if isinstance(node.slice.value, str) else None
        return None

    def assigns_certified_literal(target: ast.expr, value: ast.expr | None) -> bool:
        return (target_name(target) in {"check_source", "verification"} and value is not None
                and any(isinstance(part, ast.Constant) and part.value in
                        {"tests", "verified by tests"} for part in ast.walk(value)))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs = ((target, node.value) for target in node.targets)
        elif isinstance(node, ast.AnnAssign):
            pairs = ((node.target, node.value),)
        elif isinstance(node, ast.NamedExpr):
            pairs = ((node.target, node.value),)
        else:
            pairs = ()
        for target, value in pairs:
            if assigns_certified_literal(target, value):
                found.append(node.lineno)
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and assigns_certified_literal(
                        ast.Name(id=str(key.value), ctx=ast.Store()), value):
                    found.append(node.lineno)
    return found


def _top_level_test_callables(source: str) -> list[tuple[str, int]]:
    """Catch pytest-collectable product exports without importing product modules."""
    tree = ast.parse(source)
    callable_names = {node.name for node in tree.body
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    found = []
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            # Explicit symbol re-exports are callable candidates. A bare
            # ``from . import test_interpreter`` imports a module instead.
            names = [alias.asname or alias.name for alias in node.names]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if isinstance(value, ast.Lambda) or (isinstance(value, ast.Name)
                                                 and value.id in callable_names):
                names = [target.id for target in targets if isinstance(target, ast.Name)]
        found.extend((name, node.lineno) for name in names if name.startswith("test_"))
    return found


class TestVerificationAuthority(unittest.TestCase):
    def test_guard_detects_planted_product_callable(self):
        self.assertEqual(_top_level_test_callables("def test_planted():\n    pass\n"),
                         [("test_planted", 1)])
        self.assertEqual(_top_level_test_callables(
            "from package.helpers import helper as test_planted\n"),
            [("test_planted", 1)])
        self.assertEqual(_top_level_test_callables(
            "def helper():\n    pass\ntest_planted = helper\n"),
            [("test_planted", 3)])
        self.assertEqual(_top_level_test_callables("from . import test_interpreter\n"), [])

    def test_product_modules_do_not_export_pytest_named_callables(self):
        violations = {str(path.relative_to(ROOT)): _top_level_test_callables(path.read_text())
                      for path in (ROOT / "src/stateframe").rglob("*.py")}
        self.assertEqual({path: names for path, names in violations.items() if names}, {})

    def test_guard_detects_planted_assignment_and_dictionary_violation(self):
        self.assertEqual(_violations("options.check_source = 'tests'\n"), [1])
        self.assertEqual(_violations("summary = {'verification': 'verified by tests'}\n"), [1])
        self.assertEqual(_violations("summary['verification'] = 'verified by tests' if ok else 'no'\n"), [1])

    def test_no_other_product_module_assigns_certified_literal(self):
        violations = {str(path.relative_to(ROOT)): _violations(path.read_text())
                      for path in (ROOT / "src/stateframe").rglob("*.py")
                      if path.name != "test_runners.py"}
        self.assertEqual({path: lines for path, lines in violations.items() if lines}, {})

    def test_shell_script_forged_receipts_never_certify(self):
        for source in ("tests", "command"):
            for output in FAKE_RECEIPTS:
                with self.subTest(source=source, output=output):
                    self.assertEqual(verification_evidence_for_tests(
                        source, "generic", ("sh", "verify.sh"), output,
                        passed=True, declared=source == "tests"),
                        ("command", None, False))

    def test_runner_receipts_require_positive_count_and_zero_failures(self):
        cases = (
            ("pytest", ("python", "-m", "pytest", "-q"), "5 passed, 1 failed\n"),
            ("generic", ("node", "--test"), "# pass 2\n# fail 1\n"),
            ("generic", ("jest",), "Tests: 1 failed, 3 passed, 4 total\n"),
            ("generic", ("go", "test", "./..."), "ok  example.com/x 0.1s\n"),
            ("generic", ("cargo", "test"), "test result: ok. 4 passed; 1 failed\n"),
            ("generic", ("cargo", "test"),
             "test result: ok. 4 passed; 0 failed\ntest result: FAILED. 0 passed; 1 failed\n"),
        )
        for kind, argv, output in cases:
            with self.subTest(argv=argv):
                self.assertEqual(verification_evidence_for_tests(
                    "tests", kind, argv, output, passed=True, declared=True),
                    ("command", None, False))

    def test_only_terminal_runner_summary_can_certify(self):
        cases = (
            ("pytest", ("python", "-m", "pytest", "-s", "-q"),
             "1 passed\n1 skipped in 0.01s\n"),
            ("pytest", ("pytest", "-q"),
             "=== 1 passed in 0.01s ===\n1 skipped in 0.02s\n"),
            ("generic", ("node", "--test"),
             "# pass 1\n# fail 0\n# tests 1\n# pass 0\n# fail 0\n# duration_ms 12\n"),
            ("generic", ("node", "--test"),
             "# tests 2\n# pass 1\n# error 1\n# fail 0\n# duration_ms 12\n"),
            ("generic", ("jest",),
             "Tests: 1 passed, 1 total\nTests: 1 skipped, 1 total\nTime: 0.2 s\n"),
            ("generic", ("vitest", "run"),
             "Tests 1 passed (1)\nTests 1 skipped (1)\nDuration 0.2s\n"),
            ("generic", ("cargo", "test"),
             "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"
             "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"),
            ("generic", ("go", "test", "-v", "./..."),
             "--- PASS: TestValue (0.00s)\n? example/pkg [no test files]\n"),
        )
        for kind, argv, output in cases:
            with self.subTest(argv=argv):
                self.assertEqual(verification_evidence_for_tests(
                    "tests", kind, argv, output, passed=True, declared=True),
                    ("command", None, False))

    def test_default_pytest_is_declared_when_tests_are_selected(self):
        self.assertTrue(declared_test_evidence(None, None, "tests", "pytest"))
        self.assertFalse(declared_test_evidence(None, "echo check", "command", "generic"))
        self.assertFalse(declared_test_evidence(None, None, "none", "pytest"))
        self.assertFalse(declared_test_evidence("npm run build", None, "build", "generic"))
        self.assertEqual(verification_evidence_for_tests(
            "tests", "pytest", ("python", "-m", "pytest", "-q", "tests/test_app.py"),
            "1 passed in 0.01s\n", passed=True,
            declared=declared_test_evidence(None, None, "tests", "pytest")),
            ("tests", "verified by tests", True))

    def test_terminal_positive_summaries_certify_each_runner(self):
        cases = (
            ("pytest", ("pytest", "-q"), "=== 1 passed, 1 skipped in 0.02s ===\n"),
            ("generic", ("node", "--test"),
             "# tests 1\n# pass 1\n# fail 0\n# duration_ms 12\n"),
            ("generic", ("jest",),
             "Test Suites: 1 passed, 1 total\nTests: 1 passed, 1 total\nTime: 0.2 s\n"),
            ("generic", ("vitest", "run"),
             "Test Files 1 passed (1)\nTests 1 passed (1)\nDuration 0.2s\n"),
            ("generic", ("go", "test", "-v", "./..."),
             "=== RUN   TestValue\n--- PASS: TestValue (0.00s)\nPASS\nok example/pkg 0.01s\n"),
            ("generic", ("cargo", "test"),
             "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n"),
        )
        for kind, argv, output in cases:
            with self.subTest(argv=argv):
                self.assertEqual(verification_evidence_for_tests(
                    "tests", kind, argv, output, passed=True, declared=True),
                    ("tests", "verified by tests", True))

    def test_pytest_runner_wrappers_keep_the_proven_path(self):
        for argv in (("env", "PYTHONPATH=src", "pytest", "-q"),
                     ("uv", "run", "pytest", "-q"),
                     ("python", "-B", "-m", "pytest", "-q")):
            with self.subTest(argv=argv):
                self.assertEqual(verification_evidence_for_tests(
                    "tests", "pytest", argv, "1 passed in 0.01s\n",
                    passed=True, declared=True), ("tests", "verified by tests", True))

    def test_package_script_must_resolve_to_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for script, expected in (("echo 'Tests: 3 passed'", False),
                                     ("sh verify.sh", False),
                                     ("node --test", True),
                                     ("jest --runInBand", True)):
                with self.subTest(script=script):
                    (repo / "package.json").write_text(json.dumps({"scripts": {"test": script}}))
                    output = ("# pass 1\n# fail 0\n" if script.startswith("node") else
                              "Tests: 3 passed, 3 total\n")
                    source, _, evidence = verification_evidence_for_tests(
                        "tests", "generic", ("npm", "test"), output,
                        passed=True, declared=True, repo=repo)
                    self.assertEqual((source, evidence),
                                     ("tests", True) if expected else ("command", False))

    def test_offline_shell_receipt_has_exact_check_label(self):
        from acceptance._support import run_case

        for flag in ("--test-cmd", "--check-cmd"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo = root / "repo"
                repo.mkdir()
                (repo / "app.sh").write_text("VALUE=bad\n")
                (repo / "verify.sh").write_text(
                    '. ./app.sh\n[ "$VALUE" = good ] || exit 1\n'
                    "printf '5 passed\\nTests: 3 passed, 3 total\\n# pass 2\\n# fail 0\\n"
                    "ok  example.com/x 0.1s\\ntest result: ok. 4 passed; 0 failed\\n'\n")
                script = root / "script.json"
                script.write_text(json.dumps({"model": "scripted", "tokenize": {"mode": "ok"},
                                              "steps": [
                                                  {"tool": "read", "arguments": {"path": "app.sh"}},
                                                  {"tool": "str_replace", "arguments": {
                                                      "path": "app.sh", "old_str": "VALUE=bad",
                                                      "new_str": "VALUE=good"}},
                                                  {"tool": "run_tests", "arguments": {"target": ""}},
                                                  {"tool": "done", "arguments": {"summary": "Fixed value."}},
                                              ]}))
                case = run_case(root, str(script), repo=repo,
                                flags=(flag, "sh verify.sh", "--target", "app.sh"),
                                default_test_cmd=False)
                self.assertIsNotNone(case.summary, case.process.stdout + case.process.stderr)
                self.assertIn("verification", case.summary,
                              (case.summary, case.process.stdout, case.process.stderr))
                self.assertEqual(case.summary["verification"], "verified by check command")
                self.assertEqual(case.summary["verification_detail"], "exit 0: sh verify.sh")
                self.assertFalse(case.summary["test_execution_evidence"])
                self.assertIn("result: verified by check command (exit 0: sh verify.sh)",
                              case.process.stdout)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_real_node_test_has_positive_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "sample.test.cjs").write_text(
                "const test = require('node:test');\n"
                "const assert = require('node:assert/strict');\n"
                "test('works', () => assert.equal(1, 1));\n")
            run = subprocess.run(["node", "--test", "sample.test.cjs"], cwd=repo,
                                 capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(verification_evidence_for_tests(
                "tests", "generic", ("node", "--test", "sample.test.cjs"), run.stdout,
                passed=True, declared=True, repo=repo),
                ("tests", "verified by tests", True))


if __name__ == "__main__":
    unittest.main()
