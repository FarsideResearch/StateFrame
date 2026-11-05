"""Independent inventory and planted controls for the release-test gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_FILE = Path(__file__).with_name("must_run_plugin.py")
PLUGIN_DIR = PLUGIN_FILE.parent
_QUALIFYING_REQUIRED_ASSERTION = 'assert observed == "expected"'
_QUALIFYING_REQUIRED_SOURCE = (
    "def test_required():\n"
    '    observed = "expected"\n'
    f"    {_QUALIFYING_REQUIRED_ASSERTION}\n"
)
RELEASE_COMMIT = "33c6c511611cc249b90b47930905a8e60648c05c"
RELEASE_TREE = "59bef5821e45d506309bbb5efb5ed3fb59e52279"

EXPECTED_ENTRIES = (
    ("tests/acceptance/test_provider_queue_wait.py::test_first_queued_send_submits_once_and_run_succeeds", "requires:host_sandbox"),
    ("tests/acceptance/test_routes_host.py::test_public_cli_routes_first_plan_send_to_second_endpoint", "requires:host_sandbox"),
    ("tests/e2e/test_request_tokenizer_auth_cli.py::test_public_cli_uses_bound_key_for_every_tokenizer_measurement", "requires:host_sandbox"),
    ("tests/e2e/test_request_tokenizer_auth_cli.py::test_public_cli_wrong_key_reports_unauthorized_tokenizer_fallback", "requires:host_sandbox"),
    ("tests/test_allocation_replay_corpus.py::test_recorded_allocation_refusal_reaches_submitted", "requires:serial_lane"),
    ("tests/test_allocation_replay_corpus.py::test_planted_broken_deferral_is_reported_as_a_corpus_violation", "requires:serial_lane"),
    ("tests/test_check_sandbox_probe.py::test_probe_accepts_venv_python_without_host_selection", "requires:host_sandbox"),
    ("tests/test_check_sandbox_probe.py::test_isolated_command_default_keeps_selected_interpreter_and_environment", "requires:host_sandbox"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_CACHE_LAYOUT]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_EDIT_SCOPE_WIDEN]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_ENV_SEED_MAX_GB]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_EVICTFIX]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_FINAL_CACHE_LAYOUT]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_FL_CHECKPOINT_CAPTURE]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_GIT_COMMIT]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_MICROGRAPH]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_QUEUE_WAIT_MAX_S]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_REQUIREMENT_RECHECK]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_SEND_STALL_S]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_SHELL_STATE]", "all"),
    ("tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_STABLE_TOOL_HEAD]", "all"),
    ("tests/test_exec_capability.py::test_sandbox_binary_resolves_without_path", "requires:host_sandbox"),
    ("tests/test_exec_launch.py::test_macos_launcher_requires_memory_supervision", "darwin"),
    ("tests/test_hook_wiring_parity.py::test_no_hooks_raw_requests_and_journal_match_release", "requires:parity_baseline"),
    ("tests/test_hook_wiring_parity.py::test_no_hooks_regular_stateframe_file_matches_release", "requires:parity_baseline"),
    ("tests/test_hook_wiring_parity.py::test_no_hooks_resume_rebuilt_requests_match_release", "requires:parity_baseline"),
    ("tests/test_http_gzip.py::test_default_profile_cli_run_sends_large_request_without_gzip", "requires:host_sandbox"),
    ("tests/test_no_mcp_release_comparison.py::test_no_mcp_requests_match_release_0_2_3", "requires:parity_baseline"),
    ("tests/test_powerups_digest.py::test_unreadable_entry_fails_closed", "requires:filesystem_modes"),
    ("tests/test_protected_collection_e2e.py::test_model_create_collection_file_is_refused[tests_init]", "requires:host_sandbox"),
    ("tests/test_protected_collection_e2e.py::test_model_create_collection_file_is_refused[tests_conftest]", "requires:host_sandbox"),
    ("tests/test_protected_collection_e2e.py::test_model_create_collection_file_is_refused[root_conftest]", "requires:host_sandbox"),
    ("tests/test_protected_collection_e2e.py::test_product_create_allowed_under_protection", "requires:host_sandbox"),
    ("tests/test_protected_collection_e2e.py::test_off_keeps_collection_create_allowed", "requires:host_sandbox"),
    ("tests/test_protected_collection_e2e.py::test_shell_collection_create_is_restored_and_refused", "requires:host_sandbox"),
    ("tests/test_uv_venv_sandbox.py::test_isolated_command_builds_with_missing_home_and_unknown_uid", "linux_requires:bwrap"),
    ("tests/test_uv_venv_sandbox.py::test_linux_binds_the_uv_install_and_its_link_hop", "linux_requires:bwrap"),
    ("tests/test_uv_venv_sandbox.py::test_linux_check_command_with_home_removed_under_unknown_uid", "linux_requires:unknown_uid_bwrap"),
    ("tests/test_uv_venv_sandbox.py::test_live_uv_venv_runs_under_the_check_sandbox", "darwin_requires:uv_sandbox"),
    ("tests/test_must_run_gate.py::test_must_run_manifest_matches_reviewed_inventory", "all"),
    ("tests/test_must_run_gate.py::test_planted_controls_reject_unqualified_runs", "all"),
    ("tests/test_must_run_gate.py::test_optional_dependency_present_absent_and_touch_controls", "all"),
    ("tests/test_must_run_gate.py::test_aggregator_rejects_missing_and_stale_job_evidence", "all"),
)

REQUIRED_PROVISIONS = ("host_sandbox", "parity_baseline")
EXPECTED_JOBS = (
    ("darwin", "darwin", True, REQUIRED_PROVISIONS,
     ("bwrap", "filesystem_modes", "sandbox_exec", "serial_lane", "uv_sandbox")),
    ("linux", "linux", True, REQUIRED_PROVISIONS,
     ("bwrap", "filesystem_modes", "serial_lane", "unknown_uid_bwrap")),
)
# The 0.2.3 pin is for an explicit cross-version comparison; immediate parity follows SF_GATE_BASE.
EXPECTED_BASELINES = (
    ("immediate", {"source": "SF_GATE_BASE"}),
    ("release_0_2_3", {"commit": RELEASE_COMMIT, "tree_digest": RELEASE_TREE}),
)
EXPECTED_OPTIONAL_DEPENDENCIES = (
    ("src/stateframe/engine/kernel_shell_tool.py", "host-command:jq", "jq",
     ("tests/acceptance/test_additional_edges.py::test_u13_data_work_with_jq",), ("jq is not installed",)),
    ("src/stateframe/outer_checks.py", "host-command:node", "node",
     ("tests/acceptance/test_generic_checks.py::test_node_test_runner_fixes_bug",), ("node is not installed",)),
    ("src/stateframe/outer_checks.py", "host-command:npm", "npm",
     ("tests/acceptance/test_generic_checks.py::test_npm_script_without_tests_reports_check_command",), ("npm is not installed",)),
    ("src/stateframe/outer_checks.py", "host-command:go", "go",
     ("tests/acceptance/test_generic_checks.py::test_go_test_runner_fixes_bug",), ("go is not installed",)),
    ("src/stateframe/outer_checks.py", "host-command:node+npm", "node,npm",
     ("tests/acceptance/test_generic_checks.py::test_no_tests_package_build_is_verified_by_build_only",),
     ("node and npm are required for package build discovery",)),
    ("src/stateframe/outer_checks.py", "host-command:npm", "npm",
     ("tests/acceptance/test_generic_checks.py::test_build_printing_test_count_keeps_build_result",),
     ("npm is required for package build discovery",)),
    ("src/stateframe/engine/kernel_exec_session.py", "host-command:node", "node",
     ("tests/acceptance/test_shell_workflows.py::test_u3_node_runner_when_installed",), ("node is not installed",)),
    ("src/stateframe/engine/kernel_bounded_http.py", "host-command:openssl", "openssl",
     ("tests/test_bounded_http_https.py::*",), ("openssl CLI is required for local HTTPS certificate tests",)),
    ("src/stateframe/engine/kernel_test_isolation.py", "host-sandbox", "platform_sandbox",
     ("tests/test_exec_capability.py::test_sandbox_binary_resolves_without_path",), ("not installed on this host",)),
    ("src/stateframe/engine/kernel_git_commit.py", "host-command:git", "git",
     ("tests/test_git_commit_tool.py::test_export_writes_a_series_per_branch_that_git_am_applies",),
     ("git is not installed",)),
    ("src/stateframe/engine/kernel_hook_exec.py", "host-command:make", "/usr/bin/make",
     ("tests/test_hook_exec.py::test_macos_developer_commands_run_as_hooks_in_real_sandbox[make]",), ("is absent",)),
    ("src/stateframe/engine/kernel_hook_exec.py", "host-command:git", "/usr/bin/git",
     ("tests/test_hook_exec.py::test_macos_developer_commands_run_as_hooks_in_real_sandbox[git]",), ("is absent",)),
    ("src/stateframe/test_runners.py", "host-command:node", "node",
     ("tests/test_test_verification_authority.py::TestVerificationAuthority::test_real_node_test_has_positive_receipt",),
     ("node is not installed",)),
    ("src/stateframe/engine/kernel_test_isolation.py", "host-sandbox:bwrap", "bwrap",
     ("tests/test_uv_venv_sandbox.py::test_linux_binds_the_uv_install_and_its_link_hop",
      "tests/test_uv_venv_sandbox.py::test_isolated_command_builds_with_missing_home_and_unknown_uid",
      "tests/test_uv_venv_sandbox.py::test_linux_check_command_with_home_removed_under_unknown_uid"),
     ("bwrap is not installed",)),
    ("src/stateframe/engine/kernel_test_isolation.py", "host-sandbox:unknown-uid", "unknown_uid_bwrap",
     ("tests/test_uv_venv_sandbox.py::test_linux_check_command_with_home_removed_under_unknown_uid",),
     ("creating an unknown uid would require root", "current non-root uid has a passwd entry")),
    ("src/stateframe/engine/kernel_test_isolation.py", "host-sandbox:uv_sandbox", "uv_sandbox",
     ("tests/test_uv_venv_sandbox.py::test_live_uv_venv_runs_under_the_check_sandbox",),
     ("needs macOS with uv and a uv-managed Python",)),
    ("src/stateframe/powerup_manifest.py", "host-capability:filesystem-modes", "filesystem_modes",
     ("tests/test_powerups_digest.py::test_unreadable_entry_fails_closed",),
     ("running with privileges that ignore directory modes",)),
    ("src/stateframe/engine/mc_repo_intel_vendor/", "repomap", "python:tree_sitter,python:tree_sitter_python,python:tree_sitter_typescript,python:tree_sitter_go",
     ("tests/test_repo_map_ast.py::test_ast_path_is_taken_for_python_typescript_and_go",
      "tests/test_repo_map_ast.py::test_ast_repo_map_renders_parser_signatures",
      "tests/test_repo_map_ast.py::test_ast_and_fallback_differ_only_where_expected",
      "tests/test_repo_map_ast.py::test_each_mode_is_deterministic[ast]",
      "tests/test_repo_map_ast.py::test_repo_map_lists_go_files",
      "tests/test_repo_map_ast.py::test_ast_repo_map_lists_other_supported_languages",
      "tests/test_repo_map_ast.py::test_repo_map_header_does_not_claim_tree_sitter_when_every_parse_failed",
      "tests/test_repo_map_ast.py::test_repo_map_header_counts_one_failed_parser",
      "tests/test_repo_map_ast.py::test_ast_repo_map_keeps_async",
      "tests/test_repo_map_ast.py::test_mixed_map_reports_actual_fallback_count",
      "tests/test_repo_map_ast.py::test_orientation_journals_the_modes_actually_used",
      "tests/test_repo_map_ast.py::test_pure_python_and_js_repo_map_is_unchanged[ast]"),
     ("repo map AST needs the repomap extra (requirements-repomap.txt)",)),
    ("src/stateframe/ts_grammars.py", "repomap", "python:tree_sitter,python:tree_sitter_python,python:tree_sitter_typescript,python:tree_sitter_go",
     ("tests/test_repo_map_ast.py::test_ast_path_is_taken_for_python_typescript_and_go",
      "tests/test_repo_map_ast.py::test_ast_repo_map_renders_parser_signatures",
       "tests/test_repo_map_ast.py::test_ast_and_fallback_differ_only_where_expected",
      "tests/test_repo_map_ast.py::test_each_mode_is_deterministic[ast]",
      "tests/test_repo_map_ast.py::test_repo_map_lists_go_files",
      "tests/test_repo_map_ast.py::test_ast_repo_map_lists_other_supported_languages",
      "tests/test_repo_map_ast.py::test_repo_map_header_does_not_claim_tree_sitter_when_every_parse_failed",
      "tests/test_repo_map_ast.py::test_repo_map_header_counts_one_failed_parser",
      "tests/test_repo_map_ast.py::test_ast_repo_map_keeps_async",
      "tests/test_repo_map_ast.py::test_mixed_map_reports_actual_fallback_count",
      "tests/test_repo_map_ast.py::test_orientation_journals_the_modes_actually_used",
      "tests/test_repo_map_ast.py::test_pure_python_and_js_repo_map_is_unchanged[ast]"),
     ("repo map AST needs the repomap extra (requirements-repomap.txt)",)),
)

_PLATFORM_REASONS = {
    "tests/test_exec_launch.py::test_macos_launcher_requires_memory_supervision": "macOS only",
    "tests/test_uv_venv_sandbox.py::test_isolated_command_builds_with_missing_home_and_unknown_uid": "requires Linux bwrap behavior",
    "tests/test_uv_venv_sandbox.py::test_linux_binds_the_uv_install_and_its_link_hop": "requires Linux bwrap behavior",
    "tests/test_uv_venv_sandbox.py::test_linux_check_command_with_home_removed_under_unknown_uid": "requires Linux uid/passwd behavior",
    "tests/test_uv_venv_sandbox.py::test_live_uv_venv_runs_under_the_check_sandbox": "needs macOS with uv and a uv-managed Python",
}


EXPECTED_ASSERTIONS = {
    "tests/acceptance/test_provider_queue_wait.py::test_first_queued_send_submits_once_and_run_succeeds":
        'assert summary["succeeded"] is True',
    "tests/acceptance/test_routes_host.py::test_public_cli_routes_first_plan_send_to_second_endpoint":
        'assert first["model"] == "cheap"',
    "tests/e2e/test_request_tokenizer_auth_cli.py::test_public_cli_uses_bound_key_for_every_tokenizer_measurement":
        "assert script.tokenize_authorized and all(script.tokenize_authorized)",
    "tests/e2e/test_request_tokenizer_auth_cli.py::test_public_cli_wrong_key_reports_unauthorized_tokenizer_fallback":
        "assert script.tokenize_authorized and not any(script.tokenize_authorized)",
    "tests/test_allocation_replay_corpus.py::test_recorded_allocation_refusal_reaches_submitted":
        "assert violation is None",
    "tests/test_allocation_replay_corpus.py::test_planted_broken_deferral_is_reported_as_a_corpus_violation":
        "assert any(error in outcome.evidence for error in ALLOCATION_ERRORS)",
    "tests/test_check_sandbox_probe.py::test_probe_accepts_venv_python_without_host_selection":
        "assert isolation.check_sandbox_exec_denial(str(python)) is None",
    "tests/test_check_sandbox_probe.py::test_isolated_command_default_keeps_selected_interpreter_and_environment":
        'assert default_env["HOME"] == str(tmp_path / ".test-tmp")',
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_CACHE_LAYOUT]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_EDIT_SCOPE_WIDEN]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_ENV_SEED_MAX_GB]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_EVICTFIX]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_FINAL_CACHE_LAYOUT]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_FL_CHECKPOINT_CAPTURE]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_GIT_COMMIT]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_MICROGRAPH]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_QUEUE_WAIT_MAX_S]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_REQUIREMENT_RECHECK]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_SEND_STALL_S]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_SHELL_STATE]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_documented_switches.py::test_shell_setting_reaches_the_engine_spec[STATEFRAME_STABLE_TOOL_HEAD]":
        "assert runner.resolve_opt_in_switches()[name] == value",
    "tests/test_exec_capability.py::test_sandbox_binary_resolves_without_path":
        "assert resolved == expected or os.path.realpath(resolved) == os.path.realpath(expected)",
    "tests/test_exec_launch.py::test_macos_launcher_requires_memory_supervision":
        "assert result.returncode == 125",
    "tests/test_http_gzip.py::test_default_profile_cli_run_sends_large_request_without_gzip":
        "assert all(encoding is None for encoding, _ in seen)",
    "tests/test_powerups_digest.py::test_unreadable_entry_fails_closed":
        'assert "unreadable" in str(error.value)',
    "tests/test_protected_collection_e2e.py::test_model_create_collection_file_is_refused[tests_init]":
        'assert "REFUSED PROTECTED_TEST" in requests',
    "tests/test_protected_collection_e2e.py::test_model_create_collection_file_is_refused[tests_conftest]":
        'assert "REFUSED PROTECTED_TEST" in requests',
    "tests/test_protected_collection_e2e.py::test_model_create_collection_file_is_refused[root_conftest]":
        'assert "REFUSED PROTECTED_TEST" in requests',
    "tests/test_protected_collection_e2e.py::test_product_create_allowed_under_protection":
        'assert (run / "workspace/pkg/new_module.py").read_text() == "VALUE = 2 "',
    "tests/test_protected_collection_e2e.py::test_off_keeps_collection_create_allowed":
        'assert "REFUSED PROTECTED_TEST" not in requests',
    "tests/test_protected_collection_e2e.py::test_shell_collection_create_is_restored_and_refused":
        'assert "restored (not allowed): tests/conftest.py" in requests',
    "tests/test_uv_venv_sandbox.py::test_isolated_command_builds_with_missing_home_and_unknown_uid":
        'assert child_env["HOME"] == str(tmp_path / ".test-tmp")',
    "tests/test_uv_venv_sandbox.py::test_linux_binds_the_uv_install_and_its_link_hop":
        'assert ["--ro-bind", str(real), str(link_dir)] in binds',
    "tests/test_uv_venv_sandbox.py::test_linux_check_command_with_home_removed_under_unknown_uid":
        "assert result.returncode == 0",
    "tests/test_uv_venv_sandbox.py::test_live_uv_venv_runs_under_the_check_sandbox":
        'assert (result.returncode, result.stdout.strip()) == (0, "ok")',
    "tests/test_must_run_gate.py::test_must_run_manifest_matches_reviewed_inventory":
        "assert expected_cells and len(expected_cells) == len(actual_cells) and cell_violated == 0",
    "tests/test_must_run_gate.py::test_planted_controls_reject_unqualified_runs":
        "assert expected_control_count == detected and expected_control_count - detected == 0",
    "tests/test_must_run_gate.py::test_optional_dependency_present_absent_and_touch_controls":
        "assert expected_control_count == detected and expected_control_count - detected == 0",
    "tests/test_must_run_gate.py::test_aggregator_rejects_missing_and_stale_job_evidence":
        "assert detected_controls == expected_control_count and expected_control_count - detected_controls == 0",
}

REPLAY_CASE_IDS = (
    "clean-meter-22file-r1", "econ-flat-r1-off", "econ-flat-r1-p1", "econ-flat-r1-p2",
    "econ-flat-r2-p1", "econ-flat-r2-p2", "econ-flat-sub-1", "econ-rel10-full-r1-off",
    "econ-rel10-full-r1-p1", "econ-rel10-full-r1-p2", "econ-rel10-full-sub-6",
    "econ-relb8-full-r1-off", "econ-relb8-full-r1-p1", "econ-relb8-full-r1-p2",
    "econ-relb8-full-sub-1", "factory-a-econ-feature-r1", "factory-a-econ-feature-r3",
    "factory-a-econ-flat-r1", "factory-a-econ-flat-r2", "factory-a-econ-flat-r3",
    "factory-b-econ-feature-r1", "factory-b-econ-feature-r2", "factory-b-econ-flat-r1",
    "factory-b-econ-flat-r2", "factory-b-econ-flat-r3", "legacy-b8-off", "legacy-b8-p1",
    "legacy-b8-p2", "legacy-guard22", "legacy-v037-slot0", "presend-full",
    "presend-p3-nosvc", "presend-p4-nosvc", "presend-p4-svc", "presend-p5-nosvc",
)


def _expected_cell(nodeid: str, scope: str) -> dict[str, object]:
    linux_state: object = "RUN"
    darwin_state: object = "RUN"
    if scope == "darwin" or scope.startswith("darwin_"):
        linux_state = {"disposition": "NA_PLATFORM", "platform": "darwin",
                       "reason": _PLATFORM_REASONS[nodeid]}
    if scope == "linux" or scope.startswith("linux_"):
        darwin_state = {"disposition": "NA_PLATFORM", "platform": "linux",
                        "reason": _PLATFORM_REASONS[nodeid]}

    base_scope = scope.split(":", 1)[1] if scope.startswith("requires:") else ""
    if scope.startswith("linux_requires:"):
        base_scope = scope.split(":", 1)[1]
    elif scope.startswith("darwin_requires:"):
        base_scope = scope.split(":", 1)[1]
    prerequisites = {
        "host_sandbox": ["host_sandbox"],
        "parity_baseline": ["parity_baseline"],
        "serial_lane": ["host_sandbox", "serial_lane"],
        "filesystem_modes": ["filesystem_modes"],
        "bwrap": ["bwrap"],
        "unknown_uid_bwrap": ["bwrap", "unknown_uid_bwrap"],
        "uv_sandbox": ["uv_sandbox"],
    }.get(base_scope, [])
    optional = [
        name for name in prerequisites
        if name in {"filesystem_modes", "bwrap", "serial_lane", "unknown_uid_bwrap", "uv_sandbox"}
    ]
    baseline = None
    if nodeid.startswith("tests/test_hook_wiring_parity.py::"):
        baseline = "immediate"
    elif nodeid == "tests/test_no_mcp_release_comparison.py::test_no_mcp_requests_match_release_0_2_3":
        baseline = "release_0_2_3"
    item_nodeids = [nodeid]
    if nodeid == "tests/test_allocation_replay_corpus.py::test_recorded_allocation_refusal_reaches_submitted":
        item_nodeids = [f"{nodeid}[{case_id}]" for case_id in REPLAY_CASE_IDS]
    witness_schema = "parity_trace" if baseline else "assertion_match"
    return {
        "scenario_id": nodeid.removeprefix("tests/").replace(".py::", "_").replace("::", "_")
        .replace("[", "_").replace("]", "").replace("/", "_").replace("-", "_").lower(),
        "nodeid": nodeid,
        "item_nodeids": item_nodeids,
        "jobs": {"linux": linux_state, "darwin": darwin_state},
        "prerequisites": prerequisites,
        "optional_prerequisites": optional,
        "baseline": baseline,
        "witness_schema": witness_schema,
        "witness_assertion": None if baseline else EXPECTED_ASSERTIONS[nodeid],
    }


def _git(project: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(project), *arguments],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )


def _make_project(
    directory: Path,
    *,
    required_source: str = _QUALIFYING_REQUIRED_SOURCE,
    required_provisions: tuple[str, ...] = ("serial_lane",),
    conditional_provisions: tuple[str, ...] = (),
    cell_disposition: object = "RUN",
    baseline_name: str | None = None,
    optional_dependencies: tuple[dict[str, object], ...] = (),
    optional_source: str | None = None,
    cell_prerequisites: tuple[str, ...] | None = None,
    optional_prerequisites: tuple[str, ...] = (),
    load_serial_lane_hooks: bool = False,
) -> tuple[Path, str]:
    project = directory / "project"
    tests_dir = project / "tests"
    tests_dir.mkdir(parents=True)
    current_platform = "linux" if sys.platform.startswith("linux") else "darwin"
    prerequisites = list(cell_prerequisites) if cell_prerequisites is not None else (
        ["parity_baseline"] if baseline_name == "immediate" else []
    )
    manifest = {
        "schema": 3,
        "jobs": [{
            "id": "local",
            "platform": current_platform,
            "required": True,
            "required_provisions": list(required_provisions),
            "conditional_provisions": list(conditional_provisions),
        }],
        "baselines": {
            "immediate": {"source": "SF_GATE_BASE"},
            "release_0_2_3": {"commit": RELEASE_COMMIT, "tree_digest": RELEASE_TREE},
        },
        "cells": [{
            "scenario_id": "required_test",
            "nodeid": "test_required.py::test_required",
            "jobs": {"local": cell_disposition},
            "prerequisites": prerequisites,
            "optional_prerequisites": list(optional_prerequisites),
            "baseline": baseline_name,
            "item_nodeids": ["test_required.py::test_required"],
            "witness_schema": "parity_trace" if baseline_name is not None else "assertion_match",
            "witness_assertion": None if baseline_name is not None else _QUALIFYING_REQUIRED_ASSERTION,
        }],
        "optional_dependencies": list(optional_dependencies),
    }
    (tests_dir / "must_run.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    (project / "test_required.py").write_text(required_source, encoding="utf-8")
    (project / "test_other.py").write_text("def test_other():\n    assert True\n", encoding="utf-8")
    (project / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    plugin_list = "('must_run_plugin', 'xdist.plugin')" if load_serial_lane_hooks else "('must_run_plugin',)"
    conftest = (
        "import os, sys\n"
        "sys.path.insert(0, os.environ['MUST_RUN_PLUGIN_DIR'])\n"
        f"pytest_plugins = {plugin_list}\n"
    )
    if load_serial_lane_hooks:
        conftest += (
            "import pytest, serial_lane_plugin\n"
            "def pytest_configure(config):\n"
            "    serial_lane_plugin.configure_serial_lane(config)\n"
            "def pytest_collection_modifyitems(items, config):\n"
            "    serial_lane_plugin.collect_serial_lane_items(items, config)\n"
            "@pytest.hookimpl(hookwrapper=True)\n"
            "def pytest_runtest_makereport(item, call):\n"
            "    outcome = yield\n"
            "    serial_lane_plugin.record_call(item, outcome.get_result())\n"
            "def pytest_runtest_logreport(report):\n"
            "    serial_lane_plugin.record_logreport(report)\n"
        )
    (project / "conftest.py").write_text(conftest, encoding="utf-8")
    if optional_source is not None:
        (project / "test_optional.py").write_text(optional_source, encoding="utf-8")
    module = project / "src" / "stateframe" / "worker.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    _git(project, "init", "--quiet")
    _git(project, "add", "-A")
    _git(project, "-c", "user.name=Gate Control", "-c", "user.email=gate@example.invalid",
         "commit", "--quiet", "-m", "base")
    (project / "baseline-marker").write_text("parent commit\n", encoding="utf-8")
    _git(project, "add", "baseline-marker")
    _git(project, "-c", "user.name=Gate Control", "-c", "user.email=gate@example.invalid",
         "commit", "--quiet", "-m", "immediate parent")
    (project / "candidate-marker").write_text("candidate commit\n", encoding="utf-8")
    _git(project, "add", "candidate-marker")
    _git(project, "-c", "user.name=Gate Control", "-c", "user.email=gate@example.invalid",
         "commit", "--quiet", "-m", "candidate")
    base = _git(project, "rev-parse", "HEAD^1").stdout.strip()
    return project, base


def _prepare(project: Path, directory: Path) -> tuple[Path, Path]:
    ticket_dir = directory / "tickets"
    evidence_dir = directory / "evidence"
    result = subprocess.run(
        [sys.executable, str(PLUGIN_FILE), "prepare", "--candidate-root", str(project),
         "--manifest", str(project / "tests" / "must_run.json"),
         "--ticket-dir", str(ticket_dir), "--evidence-dir", str(evidence_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return ticket_dir, evidence_dir


def _run_shard(
    project: Path,
    base: str,
    ticket_dir: Path,
    evidence_dir: Path,
    shard_id: str,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    for name in (
        "SF_GATE_MANIFEST", "SF_TEST_BASELINE_IMMEDIATE", "SF_TEST_PARITY_BASELINE",
        "SF_SERIAL_LANE_LEDGER", "PYTEST_ADDOPTS",
    ):
        environment.pop(name, None)
    environment.update({
        "MUST_RUN_PLUGIN_DIR": str(PLUGIN_DIR),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": os.pathsep.join(
            value for value in (str(PLUGIN_DIR), environment.get("PYTHONPATH", "")) if value
        ),
        "SF_GATE_MUST_RUN": "1",
        "SF_GATE_JOB_ID": "local",
        "SF_GATE_PROVISIONS": "serial_lane" if shard_id == "serial_lane" else "",
        "SF_GATE_BASE": base,
        "SF_GATE_TICKET": str(ticket_dir / "local.json"),
        "SF_GATE_EVIDENCE": str(evidence_dir / f"local.{shard_id}.json"),
        "SF_GATE_SHARD": shard_id,
    })
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short"]
    if shard_id == "xdist":
        command.extend(["-n", "1"])
    command.extend(["-o", "enable_assertion_pass_hook=true"])
    return subprocess.run(
        command, cwd=project, env=environment, capture_output=True, text=True, timeout=45, check=False
    )


def _run_control(
    directory: Path,
    *,
    required_source: str = _QUALIFYING_REQUIRED_SOURCE,
    mode: str = "1",
    ignore_required: bool = False,
    deselect_required: bool = False,
    job_id: str | None = "local",
    required_provisions: tuple[str, ...] = ("serial_lane",),
    claimed_provisions: tuple[str, ...] = ("serial_lane",),
    conditional_provisions: tuple[str, ...] = (),
    cell_disposition: object = "RUN",
    baseline_name: str | None = None,
    baseline_mismatch: bool = False,
    wrong_baseline_ancestor: bool = False,
    extra_environment: dict[str, str] | None = None,
    stale_evidence: bool = False,
) -> subprocess.CompletedProcess[str]:
    project, base = _make_project(
        directory,
        required_source=required_source,
        required_provisions=required_provisions,
        conditional_provisions=conditional_provisions,
        cell_disposition=cell_disposition,
        baseline_name=baseline_name,
    )
    baseline_checkout = None
    base_override = base
    baseline_reference = base
    if wrong_baseline_ancestor:
        if baseline_name != "immediate":
            raise ValueError("an ancestor baseline control must name the immediate baseline")
        baseline_reference = _git(project, "rev-parse", "HEAD^^").stdout.strip()
        base_override = baseline_reference
    if baseline_mismatch or wrong_baseline_ancestor:
        if baseline_name != "immediate":
            raise ValueError("an immediate baseline control must name the immediate baseline")
        baseline_checkout = directory / "immediate-baseline"
        _git(project, "worktree", "add", "--quiet", "--detach", str(baseline_checkout), baseline_reference)
    if baseline_mismatch:
        (baseline_checkout / "baseline-marker").write_text("different commit\n", encoding="utf-8")
        _git(baseline_checkout, "add", "baseline-marker")
        _git(baseline_checkout, "-c", "user.name=Gate Control", "-c", "user.email=gate@example.invalid",
             "commit", "--quiet", "-m", "different immediate baseline")
    if ignore_required:
        with (project / "conftest.py").open("a", encoding="utf-8") as handle:
            handle.write("collect_ignore = ['test_required.py']\n")
        _git(project, "add", "conftest.py")
        _git(project, "-c", "user.name=Gate Control", "-c", "user.email=gate@example.invalid",
             "commit", "--quiet", "-m", "ignore required")
    ticket_dir, evidence_dir = _prepare(project, directory / "gate-state")
    ticket = ticket_dir / "local.json"
    evidence = evidence_dir / "local.json"
    if stale_evidence:
        evidence.write_text("{}\n", encoding="utf-8")
    environment = dict(os.environ)
    for name in ("SF_GATE_MANIFEST", "SF_TEST_BASELINE_IMMEDIATE", "SF_TEST_PARITY_BASELINE",
                 "SF_SERIAL_LANE_LEDGER", "PYTEST_ADDOPTS"):
        environment.pop(name, None)
    environment.update({
        "MUST_RUN_PLUGIN_DIR": str(PLUGIN_DIR),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "SF_GATE_MUST_RUN": mode,
        "SF_GATE_PROVISIONS": ",".join(claimed_provisions),
        "SF_GATE_BASE": base_override,
        "SF_GATE_TICKET": str(ticket),
        "SF_GATE_EVIDENCE": str(evidence),
        "PYTHONPATH": os.pathsep.join(
            value for value in (str(PLUGIN_DIR), environment.get("PYTHONPATH", "")) if value
        ),
    })
    if baseline_checkout is not None:
        environment["SF_TEST_BASELINE_IMMEDIATE"] = str(baseline_checkout)
    if job_id is None:
        environment.pop("SF_GATE_JOB_ID", None)
    else:
        environment["SF_GATE_JOB_ID"] = job_id
    if extra_environment:
        environment.update(extra_environment)
    command = [sys.executable, "-m", "pytest", "-q", "--tb=short",
               "-o", "enable_assertion_pass_hook=true"]
    if deselect_required:
        command.extend(["-k", "other"])
    return subprocess.run(command, cwd=project, env=environment, capture_output=True,
                          text=True, timeout=45, check=False)


def _control_evidence(
    project: Path,
    ticket: dict[str, object],
    scenario: dict[str, object],
    *,
    shard_id: str = "serial_lane",
) -> dict[str, object]:
    now = time.time_ns()
    scenario_id = scenario["scenario_id"]
    nodeid = scenario["nodeid"]
    if shard_id == "xdist":
        item = {
            "nodeid": nodeid,
            "reports": {"setup": ["skipped"], "teardown": ["passed"]},
            "skip_reasons": ["serial_lane: planted defer"],
        }
        expected_cell_ids: list[str] = []
        scanned_cell_ids: list[str] = []
        cells: list[dict[str, object]] = []
        deferred_cell_ids = [scenario_id]
        deferred_cells = [{
            "scenario_id": scenario_id,
            "nodeid": nodeid,
            "item_nodeids": [nodeid],
            "status": "deferred",
            "items": [item],
        }]
    else:
        item = {
            "nodeid": nodeid,
            "reports": {"setup": "passed", "call": "passed", "teardown": "passed"},
            "assertion_passes": 1,
            "assertion_witness": True,
        }
        expected_cell_ids = [scenario_id]
        scanned_cell_ids = [scenario_id]
        cells = [{
            "scenario_id": scenario_id,
            "nodeid": nodeid,
            "item_nodeids": [nodeid],
            "status": "qualified",
            "baseline": scenario["baseline"],
            "witness_schema": scenario["witness_schema"],
            "items": [item],
        }]
        deferred_cell_ids = []
        deferred_cells = []
    return {
        "schema": 2,
        "job_id": ticket["job_id"],
        "shard_id": shard_id,
        "run_nonce": ticket["run_nonce"],
        "candidate_digest": ticket["candidate_digest"],
        "manifest_digest": ticket["manifest_digest"],
        "started_at_ns": ticket["issued_at_ns"] + 1,
        "finished_at_ns": max(now, ticket["issued_at_ns"] + 2),
        "expected_cell_ids": expected_cell_ids,
        "scanned_cell_ids": scanned_cell_ids,
        "deferred_cell_ids": deferred_cell_ids,
        "cells": cells,
        "deferred_cells": deferred_cells,
        "optional_skips": [],
        "violations": {},
    }


def test_must_run_manifest_matches_reviewed_inventory():
    manifest = json.loads((ROOT / "tests" / "must_run.json").read_text(encoding="utf-8"))
    expected_cells = tuple(sorted(
        json.dumps(_expected_cell(nodeid, scope), sort_keys=True) for nodeid, scope in EXPECTED_ENTRIES
    ))
    actual_cells = tuple(sorted(json.dumps(cell, sort_keys=True) for cell in manifest["cells"]))
    cell_duplicates = len(actual_cells) - len(set(actual_cells))
    cell_violated = len(set(actual_cells) ^ set(expected_cells)) + cell_duplicates
    print(
        f"GUARD must_run_manifest: expected={len(expected_cells)} scanned={len(actual_cells)} "
        f"violated={cell_violated}"
    )
    assert expected_cells and len(expected_cells) == len(actual_cells) and cell_violated == 0, (
        f"scenario inventory differs: {sorted(set(actual_cells) ^ set(expected_cells))}; "
        f"duplicate rows={cell_duplicates}"
    )

    expected_jobs = tuple(sorted(EXPECTED_JOBS))
    actual_jobs = tuple(sorted((
        job["id"], job["platform"], job["required"], tuple(job["required_provisions"]),
        tuple(job["conditional_provisions"]),
    ) for job in manifest["jobs"]))
    job_violated = len(set(actual_jobs) ^ set(expected_jobs)) + len(actual_jobs) - len(set(actual_jobs))
    print(f"GUARD must_run_jobs: expected={len(expected_jobs)} scanned={len(actual_jobs)} violated={job_violated}")
    assert expected_jobs and len(expected_jobs) == len(actual_jobs) and job_violated == 0

    actual_baselines = tuple(sorted(
        (name, json.dumps(value, sort_keys=True)) for name, value in manifest["baselines"].items()
    ))
    expected_baselines = tuple(sorted(
        (name, json.dumps(value, sort_keys=True)) for name, value in EXPECTED_BASELINES
    ))
    baseline_violated = len(set(actual_baselines) ^ set(expected_baselines))
    print(
        f"GUARD must_run_baselines: expected={len(expected_baselines)} "
        f"scanned={len(actual_baselines)} violated={baseline_violated}"
    )
    assert expected_baselines and len(expected_baselines) == len(actual_baselines) and baseline_violated == 0

    expected_optional = tuple(sorted(EXPECTED_OPTIONAL_DEPENDENCIES))
    actual_optional = tuple(sorted((
        entry["module"], entry["extra"], entry["tool"], tuple(entry["tests"]),
        tuple(entry["absence_reasons"]),
    ) for entry in manifest["optional_dependencies"]))
    optional_violated = len(set(actual_optional) ^ set(expected_optional))
    print(
        f"GUARD optional_dependency_map: expected={len(expected_optional)} "
        f"scanned={len(actual_optional)} violated={optional_violated}"
    )
    assert expected_optional and len(expected_optional) == len(actual_optional) and optional_violated == 0


def test_planted_controls_reject_unqualified_runs(tmp_path):
    current_platform = "linux" if sys.platform.startswith("linux") else "darwin"
    other_platform = "darwin" if current_platform == "linux" else "linux"
    external_manifest = tmp_path / "unreviewed-must-run.json"
    external_manifest.write_text("{}\n", encoding="utf-8")
    platform_disposition = {
        "disposition": "NA_PLATFORM",
        "platform": other_platform,
        "reason": f"requires {other_platform}",
    }
    controls = (
        ("skip", "import pytest\ndef test_required():\n    pytest.skip('planted skip')\n",
         {}, 1, "required node skipped"),
        ("missing", "def test_required():\n    assert True\n",
         {"ignore_required": True}, 1, "required item was deselected or produced no run report"),
        ("deselected", "def test_required():\n    assert True\n",
         {"deselect_required": True}, 1, "deselected"),
        ("xfail", "import pytest\n@pytest.mark.xfail(reason='planted', strict=False)\n"
         "def test_required():\n    assert False\n", {}, 1, "XFAIL"),
        ("xpass", "import pytest\n@pytest.mark.xfail(reason='planted', strict=False)\n"
         "def test_required():\n    assert True\n", {}, 1, "XPASS"),
        ("early_return_after_assertion", "def test_required():\n    assert True\n    return\n", {}, 1,
         "scenario-specific assertion witness is missing"),
        ("collection_error", "def test_required(:\n    assert True\n", {}, 1, "collection error"),
        ("missing_job", "def test_required():\n    assert True\n",
         {"job_id": None}, 1, "SF_GATE_JOB_ID is required"),
        ("stale_evidence", "def test_required():\n    assert True\n",
         {"stale_evidence": True}, 1, "refusing pre-existing evidence"),
        ("missing_provision", "def test_required():\n    assert True\n",
         {"required_provisions": ("host_sandbox", "serial_lane"),
          "claimed_provisions": ("serial_lane",)}, 1, "required provision was not claimed"),
        ("missing_required_tool", "def test_required():\n    assert True\n",
         {"required_tool_absent": True}, 1, "provision:"),
        ("missing_baseline", "def test_required():\n    assert True\n",
         {"required_provisions": ("parity_baseline", "serial_lane"),
          "claimed_provisions": ("parity_baseline", "serial_lane")}, 1, "SF_TEST_BASELINE_IMMEDIATE is not set"),
        ("wrong_baseline_head", "def test_required():\n    assert True\n",
         {"required_provisions": ("parity_baseline", "serial_lane"),
          "claimed_provisions": ("parity_baseline", "serial_lane"),
          "baseline_name": "immediate", "baseline_mismatch": True},
         1, "immediate baseline HEAD does not match SF_GATE_BASE"),
        ("wrong_valid_ancestor", "def test_required():\n    assert True\n",
         {"required_provisions": ("parity_baseline", "serial_lane"),
          "claimed_provisions": ("parity_baseline", "serial_lane"),
          "baseline_name": "immediate", "wrong_baseline_ancestor": True},
         1, "SF_GATE_BASE must equal candidate HEAD^"),
        ("external_manifest", "def test_required():\n    assert True\n",
         {"extra_environment": {"SF_GATE_MANIFEST": str(external_manifest)}}, 4,
         "must-run manifest must be the in-repository tests/must_run.json"),
        ("platform_na", "import sys, pytest\n"
         f"@pytest.mark.skipif(sys.platform != {other_platform!r}, reason='requires {other_platform}')\n"
         "def test_required():\n    assert True\n",
         {"cell_disposition": platform_disposition}, 0, "platform_na=1"),
        ("platform_call_skip", "import pytest\ndef test_required():\n"
         f"    pytest.skip('requires {other_platform}')\n",
         {"cell_disposition": platform_disposition}, 0, "platform_na=1"),
        ("platform_executed", "def test_required():\n    assert True\n",
         {"cell_disposition": platform_disposition}, 1,
         "platform-not-applicable test call did not skip"),
        ("passing", _QUALIFYING_REQUIRED_SOURCE, {}, 0, "qualified=1"),
        ("ordinary_skip", "import pytest\ndef test_required():\n    pytest.skip('ordinary')\n",
         {"mode": "0"}, 0, "mode=off"),
    )
    detected = 0
    for name, source, options, expected_status, evidence in controls:
        values = dict(options)
        if values.pop("stale_evidence", False):
            result = _run_control(tmp_path / name, required_source=source, stale_evidence=True)
        elif values.pop("required_tool_absent", False):
            missing_tool = "sandbox_exec" if current_platform == "linux" else "bwrap"
            result = _run_control(
                tmp_path / name, required_source=source,
                required_provisions=(missing_tool, "serial_lane"),
                claimed_provisions=(missing_tool, "serial_lane"),
            )
        else:
            result = _run_control(tmp_path / name, required_source=source, **values)
        output = result.stdout + result.stderr
        assert result.returncode == expected_status, f"{name}: {output}"
        assert evidence in output, f"{name}: {output}"
        detected += 1
    expected_control_count = 20
    control_count = len(controls)
    print(
        f"GUARD must_run_controls: expected={expected_control_count} scanned={detected} "
        f"violated={expected_control_count - detected}"
    )
    print(f"controls expected={expected_control_count} detected={detected}")
    assert control_count == expected_control_count
    assert expected_control_count == detected and expected_control_count - detected == 0


def test_optional_dependency_present_absent_and_touch_controls(tmp_path):
    optional_map = (
        {
            "module": "src/stateframe/worker.py",
            "extra": "fixture-tool",
            "tool": "gate-fixture",
            "tests": ["test_optional.py::test_optional"],
            "absence_reasons": ["gate fixture tool is absent"],
        },
        {
            "module": "src/stateframe/worker.py",
            "extra": "unrelated-fixture-tool",
            "tool": "other-fixture",
            "tests": ["test_optional.py::test_optional"],
            "absence_reasons": ["other fixture tool is absent"],
        },
    )
    optional_source = (
        "import shutil, pytest\n"
        "def test_optional():\n"
        "    if shutil.which('gate-fixture') is None:\n"
        "        pytest.skip('gate fixture tool is absent')\n"
        "    assert shutil.which('gate-fixture') is not None\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool = bin_dir / "gate-fixture"
    fake_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_tool.chmod(0o755)
    controls = (
        ("present", False, (str(bin_dir), os.environ.get("PATH", "")), 0, "optional_skipped=0", False),
        ("absent_untouched", False, (os.environ.get("PATH", ""),), 0,
         "optional_na=1 optional_skipped=1 optional_touched_violations=0", False),
        ("absent_touched", True, (os.environ.get("PATH", ""),), 1,
         "optional_na=1 optional_skipped=1 optional_touched_violations=1", False),
        ("unmapped", False, (os.environ.get("PATH", ""),), 1,
         "optional dependency skip has no reviewed module/extra/test mapping", True),
    )
    detected = 0
    for name, touch_module, path_parts, expected_status, evidence, unmapped in controls:
        directory = tmp_path / name
        project, base = _make_project(
            directory,
            optional_dependencies=optional_map,
            optional_source=optional_source,
        )
        if touch_module:
            module = project / "src" / "stateframe" / "worker.py"
            module.write_text("VALUE = 2\n", encoding="utf-8")
            _git(project, "add", "src/stateframe/worker.py")
            _git(project, "-c", "user.name=Gate Control", "-c", "user.email=gate@example.invalid",
                 "commit", "--quiet", "-m", "touch optional module")
            base = _git(project, "rev-parse", "HEAD^1").stdout.strip()
        if unmapped:
            (project / "test_unclassified.py").write_text(
                "import pytest\n"
                "def test_unclassified():\n"
                "    pytest.skip('unclassified extra (missing optional package)')\n",
                encoding="utf-8",
            )
        ticket_dir, evidence_dir = _prepare(project, directory / "gate-state")
        environment = dict(os.environ)
        for key in ("SF_GATE_MANIFEST", "SF_TEST_BASELINE_IMMEDIATE", "SF_TEST_PARITY_BASELINE", "PYTEST_ADDOPTS"):
            environment.pop(key, None)
        environment.update({
            "MUST_RUN_PLUGIN_DIR": str(PLUGIN_DIR),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": os.pathsep.join(value for value in (str(PLUGIN_DIR), environment.get("PYTHONPATH", "")) if value),
            "SF_GATE_MUST_RUN": "1",
            "SF_GATE_JOB_ID": "local",
            "SF_GATE_PROVISIONS": "serial_lane",
            "SF_GATE_BASE": base,
            "SF_GATE_TICKET": str(ticket_dir / "local.json"),
            "SF_GATE_EVIDENCE": str(evidence_dir / "local.json"),
            "PATH": os.pathsep.join(value for value in path_parts if value),
        })
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short", "-o", "enable_assertion_pass_hook=true"],
            cwd=project, env=environment, capture_output=True, text=True, timeout=45, check=False,
        )
        output = result.stdout + result.stderr
        assert result.returncode == expected_status, f"{name}: {output}"
        assert evidence in output, f"{name}: {output}"
        detected += 1
    expected_control_count = 4
    print(
        f"GUARD optional_skip_controls: expected={expected_control_count} scanned={detected} "
        f"violated={expected_control_count - detected}"
    )
    print(f"controls expected={expected_control_count} detected={detected}")
    assert len(controls) == expected_control_count
    assert expected_control_count == detected and expected_control_count - detected == 0


def test_aggregator_rejects_missing_and_stale_job_evidence(tmp_path):
    project = tmp_path / "project"
    tests_dir = project / "tests"
    tests_dir.mkdir(parents=True)
    jobs = [
        {"id": "linux", "platform": "linux", "required": True,
         "required_provisions": [], "conditional_provisions": ["filesystem_modes", "serial_lane"]},
        {"id": "darwin", "platform": "darwin", "required": True,
         "required_provisions": [], "conditional_provisions": ["filesystem_modes", "serial_lane"]},
    ]
    cell = {
        "scenario_id": "gate_inventory_control",
        "nodeid": "test_required.py::test_required",
        "jobs": {"linux": "RUN", "darwin": "RUN"},
        "prerequisites": ["filesystem_modes", "serial_lane"],
        "optional_prerequisites": ["filesystem_modes", "serial_lane"],
        "baseline": None,
        "item_nodeids": ["test_required.py::test_required"],
        "witness_schema": "assertion_match",
        "witness_assertion": "assert result == \"complete\"",
    }
    manifest_path = tests_dir / "must_run.json"
    manifest_path.write_text(json.dumps({
        "schema": 3,
        "jobs": jobs,
        "baselines": {
            "immediate": {"source": "SF_GATE_BASE"},
        },
        "cells": [cell],
        "optional_dependencies": [{
            "module": "src/stateframe/worker.py",
            "extra": "filesystem-mode-check",
            "tool": "filesystem_modes",
            "tests": ["test_required.py::test_required"],
            "absence_reasons": ["running with privileges that ignore directory modes"],
        }],
    }, sort_keys=True), encoding="utf-8")
    (project / "test_required.py").write_text(
        'def test_required():\n    result = "complete"\n    assert result == "complete"\n', encoding="utf-8"
    )
    (project / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n", encoding="utf-8")
    _git(project, "init", "--quiet")
    _git(project, "add", "-A")
    _git(project, "-c", "user.name=Gate Control", "-c", "user.email=gate@example.invalid",
         "commit", "--quiet", "-m", "base")

    ticket_dir = tmp_path / "tickets"
    evidence_dir = tmp_path / "evidence"
    prepared = subprocess.run(
        [sys.executable, str(PLUGIN_FILE), "prepare", "--candidate-root", str(project),
         "--manifest", str(manifest_path), "--ticket-dir", str(ticket_dir),
         "--evidence-dir", str(evidence_dir)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    aggregate_command = [
        sys.executable, str(PLUGIN_FILE), "aggregate", "--candidate-root", str(project),
        "--manifest", str(manifest_path), "--ticket-dir", str(ticket_dir),
        "--evidence-dir", str(evidence_dir),
    ]
    missing = subprocess.run(aggregate_command, capture_output=True, text=True, timeout=30, check=False)
    assert missing.returncode == 1
    assert "expected=2 scanned=0 violated=2" in missing.stdout
    detected_controls = 1

    tickets = {job_id: json.loads((ticket_dir / f"{job_id}.json").read_text(encoding="utf-8"))
               for job_id in ("linux", "darwin")}
    for job_id, ticket in tickets.items():
        evidence = _control_evidence(project, ticket, cell, shard_id="xdist")
        serial_evidence = _control_evidence(project, ticket, cell)
        if job_id == "darwin":
            evidence["run_nonce"] = "0" * 32
        (evidence_dir / f"{job_id}.xdist.json").write_text(json.dumps(evidence), encoding="utf-8")
        (evidence_dir / f"{job_id}.serial_lane.json").write_text(
            json.dumps(serial_evidence), encoding="utf-8"
        )
    stale = subprocess.run(aggregate_command, capture_output=True, text=True, timeout=30, check=False)
    assert stale.returncode == 1
    assert "stale or wrong-run evidence" in stale.stdout
    detected_controls += 1

    ticket = tickets["darwin"]
    valid = _control_evidence(project, ticket, cell, shard_id="xdist")
    serial_valid = _control_evidence(project, ticket, cell)
    (evidence_dir / "darwin.xdist.json").write_text(json.dumps(valid), encoding="utf-8")
    (evidence_dir / "darwin.serial_lane.json").write_text(json.dumps(serial_valid), encoding="utf-8")
    complete = subprocess.run(aggregate_command, capture_output=True, text=True, timeout=30, check=False)
    assert complete.returncode == 0, complete.stdout + complete.stderr
    assert "expected=2 scanned=2 violated=0" in complete.stdout
    detected_controls += 1

    optional_na_rows = []
    for job_id, ticket in tickets.items():
        evidence = _control_evidence(project, ticket, cell)
        row = evidence["cells"][0]
        row.update({
            "status": "prerequisite_na",
            "reason": "running with privileges that ignore directory modes",
            "missing_prerequisites": ["filesystem_modes"],
        })
        row["items"][0]["reports"] = {"setup": ["skipped"]}
        row["items"][0]["skip_reasons"] = ["running with privileges that ignore directory modes"]
        row["items"][0].pop("assertion_passes")
        row["items"][0].pop("assertion_witness")
        optional_na_rows.append((job_id, evidence))
    (evidence_dir / "darwin.serial_lane.json").write_text(
        json.dumps(optional_na_rows[1][1]), encoding="utf-8"
    )
    rejected_optional_na = subprocess.run(
        aggregate_command, capture_output=True, text=True, timeout=30, check=False
    )
    assert rejected_optional_na.returncode == 1
    assert "invalid prerequisite skip evidence" in rejected_optional_na.stdout
    detected_controls += 1

    for job_id, evidence in optional_na_rows:
        evidence["optional_skips"] = [{
            "nodeid": cell["nodeid"],
            "module": "src/stateframe/worker.py",
            "extra": "filesystem-mode-check",
            "tool": "filesystem_modes",
            "reason": "running with privileges that ignore directory modes",
            "validated_absence": True,
        }]
        (evidence_dir / f"{job_id}.serial_lane.json").write_text(json.dumps(evidence), encoding="utf-8")
    optional_na = subprocess.run(aggregate_command, capture_output=True, text=True, timeout=30, check=False)
    assert optional_na.returncode == 0, optional_na.stdout + optional_na.stderr
    assert "expected=2 scanned=2 violated=0" in optional_na.stdout
    detected_controls += 1
    optional_na_rows[1][1]["cells"][0]["items"][0]["reports"] = {
        "setup": ["skipped"],
        "call": ["failed"],
    }
    (evidence_dir / "darwin.serial_lane.json").write_text(
        json.dumps(optional_na_rows[1][1]), encoding="utf-8"
    )
    invalid_skip_report = subprocess.run(
        aggregate_command, capture_output=True, text=True, timeout=30, check=False
    )
    assert invalid_skip_report.returncode == 1
    assert "invalid prerequisite skip evidence" in invalid_skip_report.stdout
    detected_controls += 1

    shard_directory = tmp_path / "live-shard-control"
    shard_project, shard_base = _make_project(
        shard_directory,
        required_source=(
            "import pytest\n"
            "@pytest.mark.serial_lane\n"
            "def test_required():\n"
            '    observed = "expected"\n'
            '    assert observed == "expected"\n'
        ),
        required_provisions=(),
        conditional_provisions=("serial_lane",),
        cell_prerequisites=("serial_lane",),
        optional_prerequisites=("serial_lane",),
        load_serial_lane_hooks=True,
    )
    shard_tickets, shard_evidence = _prepare(shard_project, shard_directory / "gate-state")
    xdist_run = _run_shard(shard_project, shard_base, shard_tickets, shard_evidence, "xdist")
    assert xdist_run.returncode == 0, xdist_run.stdout + xdist_run.stderr
    assert "shard=xdist" in xdist_run.stdout
    serial_run = _run_shard(shard_project, shard_base, shard_tickets, shard_evidence, "serial_lane")
    assert serial_run.returncode == 0, serial_run.stdout + serial_run.stderr
    assert "qualified=1" in serial_run.stdout
    shard_aggregate = subprocess.run(
        [sys.executable, str(PLUGIN_FILE), "aggregate", "--candidate-root", str(shard_project),
         "--manifest", str(shard_project / "tests" / "must_run.json"),
         "--ticket-dir", str(shard_tickets), "--evidence-dir", str(shard_evidence)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert shard_aggregate.returncode == 0, shard_aggregate.stdout + shard_aggregate.stderr
    assert "expected=1 scanned=1 violated=0" in shard_aggregate.stdout
    detected_controls += 1

    expected_control_count = 7
    print(
        f"GUARD must_run_aggregator_controls: expected={expected_control_count} "
        f"scanned={detected_controls} violated={expected_control_count - detected_controls}"
    )
    assert detected_controls == expected_control_count and expected_control_count - detected_controls == 0
    assert "expected=2 scanned=2 violated=0" in complete.stdout
