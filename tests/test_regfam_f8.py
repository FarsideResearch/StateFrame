"""Gate and harness integrity: a gate must account for what it ran, not only report a verdict.

PRERUN wiring: trigger when tests/conftest.py, any tests/**/conftest.py, tests/must_run.json, a new or renamed
tests/**/test_*.py, any *.sh, or a .py that shells out (subprocess shell=True / ["bash", "-c", ...]) changes.
Command: PYTHONPATH=src python3 -m pytest -q -p no:cacheprovider tests/test_regfam_f8.py
Cost: one `pytest --collect-only tests` child (~1 min) plus a source scan (~10 s); model-free, deterministic.
Out-of-repo gate scripts can reuse the lint (python3 tests/_gate_integrity_scan.py shell-lint <dir>) and the
run verdict (python3 tests/_gate_verdict.py -- <pytest args>: PASS only when every mandatory case PASSED).

Past instances this class covers: tests/e2e/test_*.py hidden from every full run by the "e2e/*" collect glob;
the release guards stage never selecting OFF byte-parity tests; an identity gate printing IDENTICAL for two
empty request lists; a release script reading the exit status of `| tail -1`.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import _gate_integrity_scan as scan
import _gate_verdict as verdict
from _timeouts import scaled

ROOT = Path(__file__).resolve().parents[1]

# OWED TO THE RELEASE MATRIX: files whose parity tests (node id or marker contains off_wire / byte_identical /
# parity) tests/must_run.json does not select, with the per-file count of such tests (parametrized cases
# folded). The release owner made every listed parity test mandatory except the ones below, which were red,
# skipping, or unwitnessable on this tree; a mandatory set must never start red. The check passes only while
# the uncovered set equals this list exactly, so a new uncovered parity test, a removed one, or one newly
# selected by the matrix all fail until the list is updated deliberately.
PENDING_RELEASE_SELECTION: dict[str, int] = {
    # failed on the release-owner selection run
    "tests/acceptance/test_routes_host.py": 1,  # test_inactive_routes_keep_requests_byte_identical
    "tests/e2e/test_e2e_parallel_f4_receipt.py": 1,  # [partial] and [paged_refusal] failed
    "tests/e2e/test_e2e_parallel_recovery.py": 2,  # both tests failed (5 of 5 crash points, checkpoint case)
    "tests/test_env_seed_request_parity.py": 1,  # all 3 example cases failed
    "tests/test_fu070_offparity_gates.py": 6,  # pending release-owner selection
    "tests/test_hook_wiring_parity.py": 1,  # test_no_hooks_real_shaped_llama_keeps_automatic_pin_and_release_bytes
    "tests/test_ledger_pin_delivery.py": 1,  # pending release-owner selection
    "tests/test_parallel_worker_prep.py": 1,  # test_default_parallel_off_wire_and_public_artifacts_match_merged_main
    "tests/test_working_set_off_parity_base.py": 1,  # test_off_provider_bytes_match_cutover_base, both bases
    "tests/test_working_set_render_off_parity.py": 1,  # test_off_wire_bytes_match_1de34aa4, both bases
    # always skip in a gate run
    "tests/test_mutate_parity.py": 1,  # needs SF_TEST_PARITY_BASELINE, which the must-run gate rejects as retired
    "tests/test_mutation_adversarial_parity.py": 2,  # needs SF_TEST_PARITY_BASELINE (retired, rejected by the gate)
    "tests/test_stable_wire_parity.py": 1,  # pinned commit 37f0ac81 is not in this repository's objects
    # pass only through pytest.raises: no passing assert statement for an assertion_match witness
    "tests/test_journal_replay_publication.py": 2,  # external_effect / parent_meter refusal type change controls
    "tests/test_llama_wire_seal_parity.py": 1,  # test_wire_guard_rejects_planted_changes
    "tests/test_native_prefix_whole_file_edit.py": 1,  # test_whole_file_executor_proof_parity
    "tests/test_source_window_length_framing.py": 1,  # pending release-owner selection
    "tests/test_unit_coalesce.py": 3,  # missing addition history, history errors, rejects pinned oracles
}

# Reviewed shell-lint findings: (path, rule, statement text) -> why it is not a false pass.
REVIEWED_SHELL_FINDINGS: dict[tuple[str, str, str], str] = {
    ("src/stateframe/engine/kernel_runner_receipts.py", scan.RULE_ZERO_IS_PASS,
     'status: ReceiptStatus = "NO_TESTS" if exit_code in {0, 5} else "FAIL"'):
        "only reached when total == 0; labels the run NO_TESTS, never PASS",
    ("src/stateframe/engine/unit_gate_evidence.py", scan.RULE_ZERO_IS_PASS,
     "valid = bool(runs) and all(run.complete and run.exit_code in (0, 1, 5) for run in runs)"):
        "marks evidence well-formed, not passed; collected_nodes() returns None for 0 nodes",
}


@pytest.fixture(scope="module")
def census() -> scan.Census:
    return scan.collect(ROOT, scaled(300))


# --- (a) collection completeness -------------------------------------------------------------------

def test_every_test_file_is_collected_by_the_default_run(census):
    problems = scan.collection_problems(ROOT, census)
    assert not problems, "\n".join(problems) + "\n--- pytest output tail ---\n" + census.output[-1500:]


def test_release_matrix_cells_name_collected_tests(census):
    collected = {scan._base_nodeid(item["nodeid"]) for item in census.items}
    collected_ids = {item["nodeid"] for item in census.items}
    manifest = json.loads((ROOT / "tests" / "must_run.json").read_text(encoding="utf-8"))
    dangling = sorted(
        nodeid
        for cell in manifest["cells"]
        for nodeid in (cell["nodeid"], *cell.get("item_nodeids", ()))
        if nodeid not in collected_ids and nodeid not in collected
    )
    assert manifest["cells"], "release matrix has 0 cells"
    assert not dangling, f"release matrix cells select nothing (renamed or removed tests): {dangling}"


def test_wire_identity_tests_outside_the_release_matrix_are_exactly_the_pending_list(census):
    parity = scan.parity_tests(census)
    assert parity, "parity census found 0 tests: the token match or the collection is broken"
    selection = scan.release_matrix_selection(ROOT)
    uncovered: dict[str, int] = {}
    for nodeid in parity:
        if not scan.is_selected(nodeid, selection):
            path = nodeid.split("::", 1)[0]
            uncovered[path] = uncovered.get(path, 0) + 1
    new = {path: count for path, count in uncovered.items() if PENDING_RELEASE_SELECTION.get(path) != count}
    stale = {path: count for path, count in PENDING_RELEASE_SELECTION.items() if uncovered.get(path) != count}
    assert not new and not stale, (
        "parity tests outside tests/must_run.json no longer match PENDING_RELEASE_SELECTION.\n"
        f"uncovered now (file: count) not in the list: {new}\n"
        f"listed but gone, selected, or with another count: {stale}\n"
        "A new uncovered parity test needs a must_run cell; update the list only with the release owner's OK.")


# --- (b) shell lint --------------------------------------------------------------------------------

def test_shell_lint_over_the_tree_has_only_reviewed_findings():
    findings = scan.scan_tree(ROOT)
    keys = {finding.key() for finding in findings}
    new = [finding.render() for finding in findings if finding.key() not in REVIEWED_SHELL_FINDINGS]
    stale = sorted(key for key in REVIEWED_SHELL_FINDINGS if key not in keys)
    assert not new, "gate-integrity lint findings (fix, or review into REVIEWED_SHELL_FINDINGS):\n" + "\n".join(new)
    assert not stale, f"REVIEWED_SHELL_FINDINGS entries that no longer match: {stale}"
    assert all(reason.strip() for reason in REVIEWED_SHELL_FINDINGS.values())


# --- planted controls: each rule fires on the real past shape and stays quiet on the fix ----------

def _rules(text: str, **kwargs) -> list[str]:
    return [finding.rule for finding in scan.shell_findings(textwrap.dedent(text), "x.sh", **kwargs)]


@pytest.mark.parametrize("script", [
    "out=$(python3 mkrel.py | tail -1)\nrc=$?\n",                     # mkrel: status read from tail
    "if pytest -q tests | tail -5; then echo ok; fi\n",
    "pytest -q tests | tail -40 && echo GREEN\n",
    "set -e\npytest -q tests | tee log\necho done\n",
    "#!/bin/sh\npytest -q tests 2>&1 | tail -40\n",                 # script exit status = tail's
    "while ! grep -q ready log | head -1; do sleep 1; done\n",
])
def test_pipe_status_rule_fires(script):
    assert scan.RULE_PIPE_STATUS in _rules(script)


@pytest.mark.parametrize("script", [
    "set -euo pipefail\npytest -q tests | tail -40\n",
    "set -o pipefail\nif pytest | tail -5; then :; fi\n",
    "pytest -q | tail -5\nrc=${PIPESTATUS[0]}\nexit $rc\n",
    "ls | head -3\necho listed\n",                                     # not an exit decision
    "# pytest | tail && echo PASS\necho fine\n",                       # comment
    "awk '{print $1 | \"sort\"}' f && echo ok\necho fine\n",          # single-quoted pipe is awk text
    "cat <<'EOF' > f.py\nx = a | head\nok = y | tail && z\nEOF\necho fine\n",  # heredoc body is data
])
def test_pipe_status_rule_quiet(script):
    assert scan.RULE_PIPE_STATUS not in _rules(script)


def test_pipefail_switched_off_again_is_seen():
    assert scan.RULE_PIPE_STATUS in _rules("set -o pipefail\nset +o pipefail\npytest | tail -1 && echo ok\n")


@pytest.mark.parametrize("script, fires", [
    # ident_pair before 11-05: two empty request lists diff equal and print IDENTICAL.
    ('diff -q "$a.sha" "$b.sha" >/dev/null && echo "IDENTICAL requests=$(wc -l < "$b.sha")" || echo DIFFERS\n',
     True),
    ('[ -s "$a.sha" ] || { echo VACUOUS; exit 3; }\ncmp -s "$a" "$b" && echo IDENTICAL\n', False),
    ('n=$(wc -l < "$a")\n[ "$n" -gt 0 ] || exit 3\ncmp -s "$a" "$b" && echo PASS\n', False),
    ('pytest -q tests\nrc=$?\n[ $rc -eq 5 ] && rc=0\nexit $rc\n', True),     # 0 collected mapped to pass
    ('pytest -q tests\nrc=$?\n[ $rc -eq 5 ] && fail "0 tests collected"\nexit $rc\n', False),
    ('pytest -q tests\ncase $? in\n  0|5) echo PASS ;;\nesac\n', True),
])
def test_zero_counted_as_pass_rule(script, fires):
    assert (scan.RULE_ZERO_IS_PASS in _rules(script)) is fires


def test_python_shell_snippets_are_linted():
    source = textwrap.dedent('''
        import os, subprocess
        GATE = "pytest -q tests | tail -1"
        subprocess.run(GATE, shell=True, check=True)
        subprocess.run(["bash", "-lc", "pytest -q | tail -3 && echo ok"])
        subprocess.run(["/bin/sh", "-c", f"set -o pipefail; pytest {1} | tail -3"], check=True)
        os.system("echo hi | head -1")
        subprocess.run(["grep", "-c", "x | tail"])
        ok = result.returncode in (0, 5)  # pytest gate
        label = "NO_TESTS" if code == 5 else "FAIL"
    ''')
    findings = scan.python_findings(source, "gate.py")
    lines = sorted((finding.line, finding.rule) for finding in findings)
    assert lines == [
        (4, scan.RULE_PIPE_STATUS),
        (5, scan.RULE_PIPE_STATUS),
        (7, scan.RULE_PIPE_STATUS),
        (9, scan.RULE_ZERO_IS_PASS),
    ]


def test_extensionless_shell_scripts_are_scanned(tmp_path):
    gate = tmp_path / "rungate"
    gate.write_text("#!/usr/bin/env bash\npytest -q | tail -1\n", encoding="utf-8")
    data = tmp_path / "README"
    data.write_text("pytest | tail && echo ok\n", encoding="utf-8")
    assert scan.iter_scan_files(tmp_path) == [gate]
    assert [finding.rule for finding in scan.scan_tree(tmp_path)] == [scan.RULE_PIPE_STATUS]


def _planted_suite(root: Path, ignore_glob: str) -> None:
    tests = root / "tests"
    # Distinct basenames: pytest's default import mode refuses two same-named modules outside packages.
    for index, directory in enumerate(("e2e/repo", "fixtures/rec", "acceptance/fixtures/r", "replay_corpus/_repos/r")):
        (tests / directory).mkdir(parents=True)
        (tests / directory / f"test_data{index}.py").write_text("def test_data():\n    pass\n", encoding="utf-8")
    (tests / "conftest.py").write_text(f"collect_ignore_glob = [{ignore_glob!r}]\n", encoding="utf-8")
    (tests / "test_top.py").write_text("def test_top_parity():\n    pass\n", encoding="utf-8")
    (tests / "e2e" / "test_cli.py").write_text("def test_cli():\n    pass\n", encoding="utf-8")


def test_collection_control_catches_the_old_e2e_glob(tmp_path):
    _planted_suite(tmp_path, "e2e/*")
    problems = scan.collection_problems(tmp_path, scan.collect(tmp_path, scaled(120)))
    assert any("never collected" in problem and "tests/e2e/test_cli.py" in problem for problem in problems)


def test_collection_control_catches_leaked_fixture_tests(tmp_path):
    _planted_suite(tmp_path, "e2e/*/*")
    problems = scan.collection_problems(tmp_path, scan.collect(tmp_path, scaled(120)))
    leaked = [problem for problem in problems if "fixture-data files were collected" in problem]
    assert leaked and "tests/fixtures/rec/test_data1.py" in leaked[0]


def test_collection_control_is_green_on_the_fixed_glob(tmp_path):
    _planted_suite(tmp_path, "e2e/*/*")
    (tmp_path / "tests" / "conftest.py").write_text(
        'collect_ignore_glob = ["e2e/*/*", "fixtures/*", "acceptance/fixtures/*", "replay_corpus/_repos/*"]\n',
        encoding="utf-8")
    census = scan.collect(tmp_path, scaled(120))
    assert scan.collection_problems(tmp_path, census) == []
    assert scan.parity_tests(census) == ["tests/test_top.py::test_top_parity"]


def test_collection_control_reports_import_errors(tmp_path):
    _planted_suite(tmp_path, "e2e/*/*")
    (tmp_path / "tests" / "conftest.py").write_text(
        'collect_ignore_glob = ["e2e/*/*", "fixtures/*", "acceptance/fixtures/*", "replay_corpus/_repos/*"]\n',
        encoding="utf-8")
    (tmp_path / "tests" / "test_broken.py").write_text("import not_a_module_here\n", encoding="utf-8")
    problems = scan.collection_problems(tmp_path, scan.collect(tmp_path, scaled(120)))
    assert any("collection error in tests/test_broken.py" in problem for problem in problems)
    assert any("collection exited" in problem for problem in problems)


def test_release_selection_matches_file_and_function_cells():
    selection = {"tests/a.py::test_off_wire", "tests/b_parity.py"}
    assert scan.is_selected("tests/a.py::test_off_wire", selection)
    assert scan.is_selected("tests/b_parity.py::test_anything", selection)
    assert not scan.is_selected("tests/a.py::test_byte_identical", selection)
    item = {"nodeid": "tests/c.py::test_x[on]", "markers": ["byte_identical"]}
    assert scan.is_parity_item(item)
    assert not scan.is_parity_item({"nodeid": "tests/c.py::test_x", "markers": ["slow"]})


# --- (a, upgrade) mandatory cases must PASS: skip / xfail / missing is INCOMPLETE, never PASS --------

def _finished(outcomes: dict[str, str], reports: tuple = (), exitstatus: int = 0) -> dict:
    return {"phase": "finished", "exitstatus": exitstatus, "outcomes": outcomes, "reports": list(reports)}


@pytest.mark.parametrize("outcomes, mandatory, expected", [
    ({"t.py::a": "passed", "t.py::b": "skipped"}, ["t.py::a"], verdict.PASS),
    ({"t.py::a": "skipped"}, ["t.py::a"], verdict.INCOMPLETE),
    ({"t.py::a": "xfailed"}, ["t.py::a"], verdict.INCOMPLETE),
    ({"t.py::a": "xpassed"}, ["t.py::a"], verdict.INCOMPLETE),
    ({"t.py::b": "passed"}, ["t.py::a"], verdict.INCOMPLETE),                      # mandatory case missing
    ({"t.py::a[x]": "passed", "t.py::a[y]": "skipped"}, ["t.py::a"], verdict.INCOMPLETE),
    ({"t.py::a[x]": "passed", "t.py::a[y]": "passed"}, ["t.py::a"], verdict.PASS),
    ({"t.py::a": "passed"}, [], verdict.INCOMPLETE),                                # 0 mandatory cases
    ({}, ["t.py::a"], verdict.INCOMPLETE),                                          # 0 tests ran
    ({"t.py::a": "passed", "t.py::b": "failed"}, ["t.py::a"], verdict.PRODUCT_FAIL),
])
def test_verdict_counts_only_passed_mandatory_cases(outcomes, mandatory, expected):
    assert verdict.classify(_finished(outcomes), mandatory).outcome == expected


def test_verdict_separates_environment_and_harness_failures():
    import_error = {"nodeid": "tests/t.py", "outcome": "failed", "detail": "ModuleNotFoundError: No module named 'x'"}
    syntax_error = {"nodeid": "tests/t.py", "outcome": "failed", "detail": "SyntaxError: invalid syntax"}
    assert verdict.classify(_finished({}, (import_error,), 2), ["t.py::a"]).outcome == verdict.INVALID_ENV
    assert verdict.classify(_finished({}, (syntax_error,), 2), ["t.py::a"]).outcome == verdict.PRODUCT_FAIL
    assert verdict.classify(None, ["t.py::a"]).outcome == verdict.HARNESS_FAIL
    assert verdict.classify({"phase": "collected"}, ["t.py::a"]).outcome == verdict.HARNESS_FAIL
    assert verdict.classify(_finished({"t.py::a": "passed"}, (), 4), ["t.py::a"]).outcome == verdict.HARNESS_FAIL
    conftest_failure = "ImportError while loading conftest '/r/tests/conftest.py'."
    assert verdict.classify(None, ["t.py::a"], conftest_failure).outcome == verdict.INVALID_ENV


def test_gate_run_end_to_end_reports_a_skipped_mandatory_case_as_incomplete(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_gate.py").write_text(
        "import pytest\n"
        "def test_ok():\n    pass\n"
        "def test_skipped():\n    pytest.skip('no baseline here')\n"
        "@pytest.mark.xfail(strict=False, reason='known')\n"
        "def test_xfail():\n    assert False\n",
        encoding="utf-8")
    run = lambda mandatory: verdict.run_gate(tmp_path, ["-q", "tests"], mandatory, scaled(120))  # noqa: E731
    assert run(["tests/test_gate.py::test_ok"]).outcome == verdict.PASS
    skipped = run(["tests/test_gate.py::test_ok", "tests/test_gate.py::test_skipped"])
    assert skipped.outcome == verdict.INCOMPLETE and "skipped: tests/test_gate.py::test_skipped" in skipped.render()
    xfailed = run(["tests/test_gate.py::test_xfail"])
    assert xfailed.outcome == verdict.INCOMPLETE and "xfailed" in xfailed.render()
    (tests / "test_broken.py").write_text("import not_a_module_here\n", encoding="utf-8")
    assert run(["tests/test_gate.py::test_ok"]).outcome == verdict.INVALID_ENV
    assert verdict.run_gate(tmp_path, ["--no-such-option"], ["x"], scaled(60)).outcome == verdict.HARNESS_FAIL


def test_release_matrix_mandatory_set_is_not_empty_for_either_platform():
    for platform in ("linux", "darwin"):
        assert verdict.mandatory_nodeids(ROOT, platform), f"no mandatory release cases for {platform}"
