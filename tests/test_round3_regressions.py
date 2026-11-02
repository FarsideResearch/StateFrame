"""Focused, stdlib-runnable regressions for runner evidence and editable tests."""
from __future__ import annotations

import tempfile
import unittest
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path

from stateframe import outcomes
from stateframe.test_changes import _python_findings
from stateframe.test_runners import command_runs_tests, judging_test_selection


class RoundThreeRegressions(unittest.TestCase):
    def test_receipt_matches_runner_and_declared_source(self):
        from stateframe.test_runners import verification_evidence_for_tests

        cases = [
            ("build", "generic", ("npm", "run", "build"), "Tests: 1 passed, 1 total", "build", False),
            ("command", "generic", ("sh", "verify.sh"), "ok example/pkg 0.01s", "command", False),
            ("tests", "generic", ("node", "--test"), "ok example/pkg 0.01s", "command", False),
            ("tests", "generic", ("vitest", "run"), "# pass 1", "command", False),
            ("tests", "generic", ("vitest", "run"), "Tests 1 passed (1)", "tests", True),
            ("tests", "generic", ("go", "test", "-v", "./..."),
             "=== RUN TestValue\n--- PASS: TestValue (0.00s)\nPASS\nok example/pkg 0.01s",
             "tests", True),
        ]
        for source, kind, argv, output, expected_source, evidence in cases:
            with self.subTest(argv=argv, source=source):
                actual_source, _, actual_evidence = verification_evidence_for_tests(
                    source, kind, argv, output, passed=True, declared=source == "tests")
                self.assertEqual((actual_source, actual_evidence), (expected_source, evidence))
                if source == "tests":
                    self.assertEqual(command_runs_tests(kind, argv, set(), output=output), evidence)
        self.assertFalse(command_runs_tests("pytest", ("sh", "verify.sh"), set(),
                                            output="1 passed in 0.01s"))

    def test_final_summary_and_result_line_preserve_check_provenance(self):
        from stateframe.cli import _print_summary
        from stateframe.test_runners import verification_evidence_for_tests

        for declared, argv, output, expected in (
                ("build", ("npm", "run", "build"), "Tests: 1 passed, 1 total", "verified by build only"),
                ("command", ("sh", "verify.sh"), "ok example/pkg 0.01s", "verified by check command")):
            with self.subTest(declared=declared), tempfile.TemporaryDirectory() as temp:
                source, _, evidence = verification_evidence_for_tests(
                    declared, "generic", argv, output, passed=True, declared=declared == "tests")
                summary = {**_passing_summary(), "check_source": source,
                           "test_execution_evidence": evidence,
                           "tests_after": {**_passing_summary()["tests_after"],
                                           "command": " ".join(argv)}}
                summary.update(outcomes.project(summary, check_output=output,
                                                declared_test_command=declared == "tests"))
                self.assertEqual(summary["verification"], expected)
                Path(temp, "final.diff").write_text("")
                summary.update(run_dir=temp, succeeded=True,
                               tests_before={"passed": False},
                               tests_after={"passed": True, "command": " ".join(argv)})
                printed = io.StringIO()
                with redirect_stdout(printed):
                    _print_summary(summary)
                self.assertIn(f"result: {expected} (exit 0: {' '.join(argv)})", printed.getvalue())
                self.assertIn("=== checks ===", printed.getvalue())
                self.assertNotIn("verified by tests", printed.getvalue())

    def test_unittest_methods_and_weakening_qualify_result(self):
        original = ("import unittest\nclass Base(unittest.TestCase):\n    pass\n"
                    "class AccountTests(Base):\n"
                    "    def test_balance(self):\n        self.assertEqual(balance(), 10)\n")
        changes = [
            ("", "assertion removed"),
            ("self.assertTrue(balance())", "comparison loosened"),
            ("self.assertIsNotNone(balance())", "comparison loosened"),
            ("self.assertEqual(balance(), balance())", "assertion made tautological"),
            ("self.skipTest('later')\n        self.assertEqual(balance(), 10)",
             "skip/xfail/only/todo marker added"),
        ]
        for replacement, reason in changes:
            with self.subTest(replacement=replacement):
                changed = original.replace("self.assertEqual(balance(), 10)", replacement or "pass")
                findings = _python_findings("tests/test_account.py", original, changed)
                self.assertIn(reason, {item["reason"] for item in findings})
                self.assertIn("tests/test_account.py::AccountTests::test_balance",
                              {item["test"] for item in findings})
                self.assertEqual(outcomes.project({**_passing_summary(),
                                                  "test_weakening_findings": findings,
                                                  "tests_after": {**_passing_summary()["tests_after"],
                                                                  "command": "node --test"}},
                                                 check_output="# pass 1\n# fail 0\n",
                                                 declared_test_command=True)["verification"].split(", but ")[0],
                                  "verified by tests")
                self.assertIn("test file weakened", outcomes.project({**_passing_summary(),
                              "test_weakening_findings": findings,
                              "tests_after": {**_passing_summary()["tests_after"], "command": "node --test"}},
                              check_output="# pass 1\n# fail 0\n",
                              declared_test_command=True)["verification"])
        for marker in ("@unittest.skip('later')", "@unittest.expectedFailure"):
            changed = original.replace("class AccountTests(Base):", marker + "\nclass AccountTests(Base):")
            self.assertIn("skip/xfail/only/todo marker added",
                          {item["reason"] for item in _python_findings("tests/test_account.py", original, changed)})
        camel = original.replace("test_balance", "testBalance")
        self.assertIn("assertion removed", {item["reason"] for item in _python_findings(
            "tests/test_account.py", camel, camel.replace("self.assertEqual(balance(), 10)", "pass"))})

    def test_unconditional_early_exit_qualifies_unchanged_assertion(self):
        cases = (
            ("def test_value():\n    assert value == 1\n",
             "def test_value():\n    return\n    assert value == 1\n"),
            ("def test_value():\n    assert value == 1\n",
             "def test_value():\n    if True:\n        return\n    assert value == 1\n"),
            ("import unittest\nclass ValueTests(unittest.TestCase):\n"
             "    def test_value(self):\n        self.assertEqual(value, 1)\n",
             "import unittest\nclass ValueTests(unittest.TestCase):\n"
             "    def test_value(self):\n        raise RuntimeError('stop')\n"
             "        self.assertEqual(value, 1)\n"),
        )
        for before, after in cases:
            with self.subTest(after=after):
                findings = _python_findings("tests/test_value.py", before, after)
                self.assertIn("assertion made unreachable", {item["reason"] for item in findings})
                summary = {**_passing_summary(), "test_weakening_findings": findings,
                           "tests_after": {**_passing_summary()["tests_after"],
                                           "command": "node --test"}}
                self.assertIn("assertion made unreachable; heuristic",
                              outcomes.project(summary, check_output="# pass 1\n# fail 0\n",
                                               declared_test_command=True)["verification"])
        before = "def test_value(flag):\n    if flag:\n        return\n    assert value == 1\n"
        self.assertNotIn("assertion made unreachable", {item["reason"] for item in
                         _python_findings("tests/test_value.py", before, before)})

    def test_wrong_requires_original_unchanged_judging_named_body(self):
        from stateframe.engine.probes.test_wrong_evidence import validate_test_wrong

        original = ("def test_selected():\n    assert value == 1\n\n"
                    "def test_other():\n    assert value == 2\n")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tests").mkdir()
            selected = root / "tests/test_a.py"
            selected.write_text(original)
            (root / "tests/test_b.py").write_text("def test_other():\n    assert value == 3\n")
            row = {"test_wrong_sources": {"tests/test_a.py": original},
                   "test_wrong_hashes": {"tests/test_a.py": hashlib.sha256(
                       original.encode()).hexdigest()},
                   "test_wrong_selectors": {"tests/test_a.py": ["test_selected"]}}
            evidence = {"path": "tests/test_a.py", "test": "test_selected",
                        "assertion": "assert value == 1", "why": "wrong", "observed": "failed"}
            self.assertIsNone(validate_test_wrong(evidence, row, root))
            self.assertIn("judging", validate_test_wrong({**evidence, "path": "tests/test_b.py"}, row, root))
            self.assertIn("named test", validate_test_wrong({**evidence, "test": "test_other"}, row, root))
            self.assertIn("named test", validate_test_wrong({**evidence, "assertion": "assert value == 2"}, row, root))
            self.assertIn("named test", validate_test_wrong({**evidence, "assertion": "assert value == 3"}, row, root))
            self.assertIn("-k", validate_test_wrong(evidence, {**row, "test_wrong_keyword": "other"}, root))
            self.assertIsNone(validate_test_wrong(evidence, {**row, "test_wrong_keyword": "other",
                                                        "test_wrong_failed_nodes": {
                                                            "tests/test_a.py": ["test_selected"]}}, root))
            selected.write_text(original + "\n# changed\n")
            self.assertIn("modified", validate_test_wrong(evidence, row, root))
            selected.write_text(original.replace("assert value == 1", "assert value == 4"))
            self.assertIn("modified", validate_test_wrong({**evidence, "assertion": "assert value == 4"}, row, root))

    def test_wrong_judging_selection_excludes_unselected_files(self):
        paths = ("tests/test_a.py", "tests/test_b.py")
        self.assertEqual(judging_test_selection(
            paths, ("python", "-m", "pytest", "-q", "tests/test_a.py::test_selected")),
            {"tests/test_a.py": ["test_selected"]})
        self.assertEqual(judging_test_selection(paths, ("sh", "verify.sh")), {})
        self.assertEqual(judging_test_selection(
            ("tests/check.sh", "tests/other.sh"), ("sh", "tests/check.sh")),
            {"tests/check.sh": []})
        self.assertEqual(judging_test_selection(
            paths, ("python", "-m", "pytest", "-q", "tests/test_a.py"),
            failed_output="FAILED tests/test_b.py::test_other - AssertionError"),
            {"tests/test_a.py": [], "tests/test_b.py": ["test_other"]})

    def test_wrong_evidence_survives_strict_worker_projection(self):
        from stateframe.engine.kernel_information_policy import InformationPolicy, VERSION

        row = {"tier": "USER-REPO", "test_wrong_selectors": {"tests/test_a.py": ["test_selected"]},
               "test_wrong_sources": {"tests/test_a.py": "def test_selected():\n    assert False\n"},
               "test_wrong_hashes": {"tests/test_a.py": "abc"},
               "test_wrong_failed_nodes": {"tests/test_a.py": ["test_selected"]},
               "test_wrong_keyword": "selected", "test_wrong_mark": "slow",
               "information_policy": {"information_policy_version": VERSION,
                                      "initial_oracle_symptom": "OFF",
                                      "live_grader_feedback": "DIAGNOSTIC",
                                      "target_disclosure": "FILE_AND_SYMBOL"}}
        projected = InformationPolicy(row).worker_row(row)
        for key in ("test_wrong_selectors", "test_wrong_sources", "test_wrong_hashes",
                    "test_wrong_failed_nodes", "test_wrong_keyword", "test_wrong_mark"):
            self.assertEqual(projected[key], row[key])


def _passing_summary():
    digest = "sha256:" + "a" * 64
    return {"runner_kind": "generic", "check_source": "tests", "test_execution_evidence": True,
            "child_exit_code": 0, "engine_terminal": {"result": "SUBMITTED", "submission": {
                "stored": True, "seal_id": digest, "artifact_digest": digest}},
            "candidate_artifact_digest": digest, "candidate_captured": True, "candidate_sha256": digest,
            "tests_after": {"passed": True, "exit_code": 0, "candidate_sha256": digest}}
