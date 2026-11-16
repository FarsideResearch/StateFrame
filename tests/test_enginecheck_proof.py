"""Suite and red-proof integrity: a run passes only when every required node was collected, executed and passed.

Each case builds a tiny throwaway pytest project in tmp_path and runs it through enginecheck.proof.
The real repo suite is never executed here.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from enginecheck import proof as proof_module
from enginecheck.proof import SuiteReport, red_proof, run_nodes


def _project(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "conftest.py").write_text("")
    for name, body in files.items():
        (root / name).write_text(textwrap.dedent(body))
    return root


def _report(
    *,
    required: tuple[str, ...] = (),
    collected: tuple[str, ...] = (),
    executed: tuple[str, ...] = (),
    passed: tuple[str, ...] = (),
    exit_code: int = 0,
) -> SuiteReport:
    return SuiteReport(
        required=required,
        collected=collected,
        executed=executed,
        passed=passed,
        failed={},
        skipped={},
        errors={},
        exit_code=exit_code,
    )


# Pure logic, no subprocess.

def test_empty_required_is_never_ok():
    assert not _report().ok


def test_bare_id_with_no_collected_params_is_not_ok():
    report = _report(required=("t.py::test_p",), exit_code=0)
    assert not report.ok
    assert report.missing == ("t.py::test_p",)


# Suite runs.

def test_all_required_passing_is_ok(tmp_path: Path):
    root = _project(tmp_path, {
        "test_ok.py": """
            def test_a():
                assert True

            def test_b():
                assert True
        """,
    })
    report = run_nodes(["test_ok.py::test_a", "test_ok.py::test_b"], cwd=root)
    assert report.ok
    assert report.summary().startswith("SUITE OK required=2 collected=2 executed=2 passed=2")


def test_required_id_never_collected_is_missing(tmp_path: Path):
    root = _project(tmp_path, {
        "test_ok.py": """
            def test_a():
                assert True
        """,
    })
    report = run_nodes(["test_ok.py::test_a", "test_ok.py::test_aa"], cwd=root)
    assert not report.ok
    assert "test_ok.py::test_aa" in report.missing
    assert report.summary().startswith("SUITE NOT_OK")


def test_syntax_error_in_one_file_fails_the_suite(tmp_path: Path):
    root = _project(tmp_path, {
        "test_good.py": """
            def test_g():
                assert True
        """,
        "test_bad.py": """
            def test_b(:
                assert True
        """,
    })
    report = run_nodes(["test_good.py::test_g", "test_bad.py::test_b"], cwd=root)
    assert not report.ok
    assert "test_bad.py" in report.errors
    assert "test_bad.py::test_b" in report.missing


def test_class_scoped_required_id_is_matched(tmp_path: Path):
    root = _project(tmp_path, {
        "test_cls.py": """
            class TestGroup:
                def test_m(self):
                    assert True
        """,
    })
    report = run_nodes(["test_cls.py::TestGroup::test_m"], cwd=root)
    assert report.ok
    assert report.collected == ("test_cls.py::TestGroup::test_m",)


def test_skipped_required_test_fails_the_suite(tmp_path: Path):
    root = _project(tmp_path, {
        "test_s.py": """
            import pytest

            @pytest.mark.skip(reason="not ready yet")
            def test_s():
                assert True

            def test_p():
                assert True
        """,
    })
    report = run_nodes(["test_s.py::test_s", "test_s.py::test_p"], cwd=root)
    assert not report.ok
    assert "test_s.py::test_s" in report.skipped
    assert report.passed == ("test_s.py::test_p",)


def test_bare_parametrized_id_matches_every_param(tmp_path: Path):
    root = _project(tmp_path, {
        "test_p.py": """
            import pytest

            @pytest.mark.parametrize("n", [1, 2, 3])
            def test_p(n):
                assert n > 0
        """,
    })
    report = run_nodes(["test_p.py::test_p"], cwd=root)
    assert report.ok
    assert len(report.collected) == 3


def test_bare_parametrized_id_with_empty_param_set_is_not_ok(tmp_path: Path):
    root = _project(tmp_path, {
        "test_p.py": """
            import pytest

            @pytest.mark.parametrize("n", [])
            def test_p(n):
                assert n > 0
        """,
    })
    report = run_nodes(["test_p.py::test_p"], cwd=root)
    assert not report.ok
    assert any(key.startswith("test_p.py::test_p") for key in report.skipped)


def test_collection_error_detail_keeps_exception_name_intact(tmp_path: Path):
    root = _project(tmp_path, {
        "test_eof.py": """
            raise EOFError("boom")
        """,
    })
    report = run_nodes(["test_eof.py::test_x"], cwd=root)
    assert not report.ok
    assert report.errors["test_eof.py"].startswith("EOFError")


def test_class_id_matches_every_method(tmp_path: Path):
    root = _project(tmp_path, {
        "t.py": """
            class TestGroup:
                def test_a(self):
                    assert True

                def test_b(self):
                    assert True
        """,
    })
    report = run_nodes(["t.py::TestGroup"], cwd=root)
    assert report.ok
    assert len(report.collected) == 2


def test_class_id_with_no_methods_is_not_ok(tmp_path: Path):
    root = _project(tmp_path, {
        "t.py": """
            class TestGroup:
                pass
        """,
    })
    report = run_nodes(["t.py::TestGroup"], cwd=root)
    assert not report.ok


def test_non_strict_xpass_counts_as_passed(tmp_path: Path):
    root = _project(tmp_path, {
        "test_x.py": """
            import pytest

            @pytest.mark.xfail
            def test_x():
                assert True
        """,
    })
    report = run_nodes(["test_x.py::test_x"], cwd=root)
    assert report.ok


def test_timeout_reports_124_and_is_not_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _project(tmp_path, {
        "test_slow.py": """
            import time

            def test_s():
                time.sleep(60)
        """,
    })
    monkeypatch.setattr(proof_module, "_RUN_TIMEOUT_SECONDS", 1)
    report = run_nodes(["test_slow.py::test_s"], cwd=root)
    assert report.exit_code == 124
    assert "<timeout>" in report.errors
    assert not report.ok


def test_run_nodes_rejects_an_empty_node_list(tmp_path: Path):
    with pytest.raises(ValueError):
        run_nodes([], cwd=tmp_path)


def test_summary_never_claims_all_tests_passed(tmp_path: Path):
    passing = _project(tmp_path / "pass", {
        "test_ok.py": """
            def test_a():
                assert True
        """,
    })
    broken = _project(tmp_path / "broken", {
        "test_bad.py": """
            def test_b(:
                assert True
        """,
    })
    ok_summary = run_nodes(["test_ok.py::test_a"], cwd=passing).summary()
    bad_summary = run_nodes(["test_bad.py::test_b"], cwd=broken).summary()
    assert "ALL TESTS PASSED" not in ok_summary
    assert "ALL TESTS PASSED" not in bad_summary
    assert "ALL TESTS PASSED" not in _report().summary()


# Red proofs.

def test_red_proof_strict_type_rejects_expect_appearing_only_in_message(tmp_path: Path):
    root = _project(tmp_path, {
        "test_fail.py": """
            import pytest

            def test_f():
                pytest.fail("AssertionError happened")
        """,
    })
    proof = red_proof("test_fail.py::test_f", cwd=root, expect="AssertionError")
    assert proof.status == "failed"
    assert proof.intended is False


def test_red_proof_strict_type_accepts_real_assertion(tmp_path: Path):
    root = _project(tmp_path, {
        "test_real.py": """
            def test_r():
                assert 1 == 2
        """,
    })
    proof = red_proof("test_real.py::test_r", cwd=root, expect="AssertionError")
    assert proof.intended is True


def test_red_proof_not_intended_for_name_error_in_test_body(tmp_path: Path):
    root = _project(tmp_path, {
        "test_nm.py": """
            def test_nm():
                assert undefined_name_for_proof_test == 1
        """,
    })
    proof = red_proof("test_nm.py::test_nm", cwd=root, expect="NameError")
    assert proof.status == "failed"
    assert proof.intended is False


def test_red_proof_not_intended_for_import_error_inside_test(tmp_path: Path):
    root = _project(tmp_path, {
        "test_lazy.py": """
            def test_lazy():
                import enginecheck_proof_missing_module_xyz
                assert True
        """,
    })
    proof = red_proof("test_lazy.py::test_lazy", cwd=root, expect="ModuleNotFoundError")
    assert proof.status == "failed"
    assert proof.intended is False

def test_red_proof_intended_on_matching_assertion(tmp_path: Path):
    root = _project(tmp_path, {
        "test_red.py": """
            def test_red():
                assert 1 == 2, "expected_marker_ mismatch"
        """,
    })
    proof = red_proof("test_red.py::test_red", cwd=root, expect="expected_marker_")
    assert proof.status == "failed"
    assert proof.intended is True


def test_red_proof_not_intended_when_import_fails_at_module_level(tmp_path: Path):
    root = _project(tmp_path, {
        "test_imp.py": """
            import enginecheck_proof_no_such_module_xyz

            def test_x():
                assert True
        """,
    })
    proof = red_proof("test_imp.py::test_x", cwd=root, expect="ModuleNotFoundError")
    assert proof.status == "errored"
    assert proof.intended is False


def test_red_proof_not_intended_when_test_passes(tmp_path: Path):
    root = _project(tmp_path, {
        "test_pass.py": """
            def test_ok():
                assert True
        """,
    })
    proof = red_proof("test_pass.py::test_ok", cwd=root, expect="AssertionError")
    assert proof.status == "passed"
    assert proof.intended is False


def test_red_proof_not_intended_when_message_lacks_expect(tmp_path: Path):
    root = _project(tmp_path, {
        "test_wrong.py": """
            def test_wrong():
                assert 1 == 2
        """,
    })
    proof = red_proof("test_wrong.py::test_wrong", cwd=root, expect="expected_marker_")
    assert proof.status == "failed"
    assert proof.intended is False
