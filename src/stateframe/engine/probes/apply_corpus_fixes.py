"""Apply audited corpus replacement oracles to a throwaway sandbox.

The normal convergence code remains unchanged. A future caller can run the
existing fix-era overlay first, then call ``apply_corpus_fixes`` and use
``replacement_selector`` for the replacement whole-file oracle.

CLI usage::

    python3 probes/apply_corpus_fixes.py \
        --root /tmp/backend-sandbox \
        --task-row results/gap_watchdog_auto_pause_trigger.jsonl

``--task-row`` must contain one JSON task object; a one-row ``.jsonl`` file
from ``results/`` is valid. The destination is a caller-owned throwaway
sandbox, never the working tree.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


POC = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CorpusFix:
    function: str
    sha: str
    source_relpath: str
    destination_relpath: str
    selector: str


CORPUS_FIXES: tuple[CorpusFix, ...] = (
    CorpusFix(
        function="_collect_metrics",
        sha="bc8c32655ae985930105430a76e732f4330b9440",
        source_relpath="corpus_fixes/_collect_metrics/test_collect_metrics_oracle.py",
        destination_relpath="tests/test_collect_metrics_oracle.py",
        selector="tests/test_collect_metrics_oracle.py",
    ),
    CorpusFix(
        function="_advance_inner",
        sha="6061439d91d60153021fe4f1b77bf04e832b3e0d",
        source_relpath="corpus_fixes/_advance_inner/test_advance_inner_oracle.py",
        destination_relpath="tests/test_advance_inner_oracle.py",
        selector="tests/test_advance_inner_oracle.py",
    ),
    CorpusFix(
        function="_watchdog_auto_pause_trigger",
        sha="46c342dc01a6822fef3fd33aaf4028c1f1e7e183",
        source_relpath=(
            "corpus_fixes/_watchdog_auto_pause_trigger/"
            "test_watchdog_auto_pause_trigger_oracle.py"
        ),
        destination_relpath="tests/test_watchdog_auto_pause_trigger_oracle.py",
        selector="tests/test_watchdog_auto_pause_trigger_oracle.py",
    ),
    CorpusFix(
        function="override_mr_checks",
        sha="7787d96f68ebc5aae13ce049b201ae2a8adbcd52",
        source_relpath=(
            "corpus_fixes/override_mr_checks/"
            "test_override_mr_checks_oracle.py"
        ),
        destination_relpath="tests/test_override_mr_checks_oracle.py",
        selector="tests/test_override_mr_checks_oracle.py",
    ),
    CorpusFix(
        function="run_diverge_focus",
        sha="efe31188d0c76ebac0866d6ebec5518ed37b1374",
        source_relpath=(
            "corpus_fixes/run_diverge_focus/"
            "test_run_diverge_focus_oracle.py"
        ),
        destination_relpath="tests/test_run_diverge_focus_oracle.py",
        selector="tests/test_run_diverge_focus_oracle.py",
    ),
    CorpusFix(
        function="_supporting_data_param_digest",
        sha="91ef71d807bf7c1290c36d2c6145cd4e6e275120",
        source_relpath=(
            "corpus_fixes/_supporting_data_param_digest/"
            "test_supporting_data_param_digest_oracle.py"
        ),
        destination_relpath="tests/test_supporting_data_param_digest_oracle.py",
        selector="tests/test_supporting_data_param_digest_oracle.py",
    ),
    CorpusFix(
        function="normalize_generated_content",
        sha="2b5d3f3a61385e328b022159ac1b8a7b333d77c7",
        source_relpath=(
            "corpus_fixes/normalize_generated_content/"
            "test_normalize_generated_content_oracle.py"
        ),
        destination_relpath="tests/test_normalize_generated_content_oracle.py",
        selector="tests/test_normalize_generated_content_oracle.py",
    ),
    CorpusFix(
        function="score_tool_receipt",
        sha="133b184c3c1147912a5015ef8deb2465a2405eb3",
        source_relpath=(
            "corpus_fixes/score_tool_receipt/"
            "test_score_tool_receipt_oracle.py"
        ),
        destination_relpath="tests/test_score_tool_receipt_oracle.py",
        selector="tests/test_score_tool_receipt_oracle.py",
    ),
    CorpusFix(
        function="_build_chat_block",
        sha="ea639fc143042b9aab503bc226f9f71c53b6979c",
        source_relpath=(
            "corpus_fixes/_build_chat_block/"
            "test_build_chat_block_oracle.py"
        ),
        destination_relpath="tests/test_build_chat_block_oracle.py",
        selector="tests/test_build_chat_block_oracle.py",
    ),
    CorpusFix(
        function="create_and_dispatch_agent_task",
        sha="09523282e038064a1d221aef7d387ab244ec53a2",
        source_relpath=(
            "corpus_fixes/create_and_dispatch_agent_task/"
            "test_create_and_dispatch_agent_task_oracle.py"
        ),
        destination_relpath="tests/test_create_and_dispatch_agent_task_oracle.py",
        selector="tests/test_create_and_dispatch_agent_task_oracle.py",
    ),
    CorpusFix(
        function="runner_schedule_reinit",
        sha="5cfbe87d72c8921df483518d105fceb037ad64bd",
        source_relpath=(
            "corpus_fixes/runner_schedule_reinit/"
            "test_runner_schedule_reinit_oracle.py"
        ),
        destination_relpath="tests/test_runner_schedule_reinit_oracle.py",
        selector="tests/test_runner_schedule_reinit_oracle.py",
    ),
    CorpusFix(
        function="_summarize_packet_for_prompt",
        sha="0171e7da1858f4766a41425bf6d3596593c059e2",
        source_relpath=(
            "corpus_fixes/_summarize_packet_for_prompt/"
            "test_summarize_packet_for_prompt_oracle.py"
        ),
        destination_relpath="tests/test_summarize_packet_for_prompt_oracle.py",
        selector="tests/test_summarize_packet_for_prompt_oracle.py",
    ),
    CorpusFix(
        function="create_post",
        sha="fbf8d8e5b3e4bc18b4ea55064141a2fb14e3b66c",
        source_relpath=(
            "corpus_fixes/create_post/"
            "test_create_post_oracle.py"
        ),
        destination_relpath="tests/test_create_post_oracle.py",
        selector="tests/test_create_post_oracle.py",
    ),
    CorpusFix(
        function="seed_secret_bindings",
        sha="0a4926602812d1369e311f10172e5404dae1f79f",
        source_relpath=(
            "corpus_fixes/seed_secret_bindings/"
            "test_seed_secret_bindings_oracle.py"
        ),
        destination_relpath="tests/test_seed_secret_bindings_oracle.py",
        selector="tests/test_seed_secret_bindings_oracle.py",
    ),
    CorpusFix(
        function="_advance_merge_with_claim",
        sha="eae0a3282e1df904bbee0d48c0caf59c54785365",
        source_relpath=(
            "corpus_fixes/_advance_merge_with_claim/"
            "test_advance_merge_with_claim_oracle.py"
        ),
        destination_relpath="tests/test_advance_merge_with_claim_oracle.py",
        selector="tests/test_advance_merge_with_claim_oracle.py",
    ),
    CorpusFix(
        function="create_attention_event",
        sha="d042f48c35193c2da48c3e6239d0553a56dfbd5d",
        source_relpath=(
            "corpus_fixes/create_attention_event/"
            "test_create_attention_event_oracle.py"
        ),
        destination_relpath="tests/test_create_attention_event_oracle.py",
        selector="tests/test_create_attention_event_oracle.py",
    ),
    CorpusFix(
        function="_apply_deliverable_verification_policy",
        sha="a2683b9580f6597b6e61ff338d85b9326827a46c",
        source_relpath=(
            "corpus_fixes/_apply_deliverable_verification_policy/"
            "test_apply_deliverable_verification_policy_oracle.py"
        ),
        destination_relpath=(
            "tests/test_apply_deliverable_verification_policy_oracle.py"
        ),
        selector="tests/test_apply_deliverable_verification_policy_oracle.py",
    ),
    CorpusFix(
        function="resolve_due_forecasts",
        sha="ed46dd1c69858e80071201ce1f450a34ef130697",
        source_relpath=(
            "corpus_fixes/resolve_due_forecasts/"
            "test_resolve_due_forecasts_oracle.py"
        ),
        destination_relpath="tests/test_resolve_due_forecasts_oracle.py",
        selector="tests/test_resolve_due_forecasts_oracle.py",
    ),
    CorpusFix(
        function="verify_container",
        sha="96b6c11644820392c686a9b84ad6375f92443cc7",
        source_relpath=(
            "corpus_fixes/verify_container/"
            "test_verify_container_oracle.py"
        ),
        destination_relpath="tests/test_verify_container_oracle.py",
        selector="tests/test_verify_container_oracle.py",
    ),
    CorpusFix(
        function="_running_commit",
        sha="b1877bb1eef2dab35b5bd2456c11ee9d57868b20",
        source_relpath=(
            "corpus_fixes/_running_commit/"
            "test_running_commit_oracle.py"
        ),
        destination_relpath="tests/test_running_commit_oracle.py",
        selector="tests/test_running_commit_oracle.py",
    ),
    CorpusFix(
        function="update_experiment",
        sha="fb1d75cbeb18110639dfdffeeba140786922150b",
        source_relpath=(
            "corpus_fixes/update_experiment/"
            "test_update_experiment_oracle.py"
        ),
        destination_relpath="tests/test_update_experiment_oracle.py",
        selector="tests/test_update_experiment_oracle.py",
    ),
    CorpusFix(
        function="_collect_prior_outputs",
        sha="f03e0f212a49a3b6396e5c0b6ac75d856570ac5d",
        source_relpath=(
            "corpus_fixes/_collect_prior_outputs/"
            "test_collect_prior_outputs_oracle.py"
        ),
        destination_relpath="tests/test_collect_prior_outputs_oracle.py",
        selector="tests/test_collect_prior_outputs_oracle.py",
    ),
    CorpusFix(
        function="trigger_page_embed",
        sha="2359949c94fcdf1d70df117ca8b4fa54b6a2105b",
        source_relpath=(
            "corpus_fixes/trigger_page_embed/"
            "test_trigger_page_embed_oracle.py"
        ),
        destination_relpath="tests/test_trigger_page_embed_oracle.py",
        selector="tests/test_trigger_page_embed_oracle.py",
    ),
    CorpusFix(
        function="execute_plan",
        sha="02f19c8626bbfb2955c02f947b3f6b82eab09ce1",
        source_relpath=(
            "corpus_fixes/execute_plan/"
            "test_execute_plan_oracle.py"
        ),
        destination_relpath="tests/test_execute_plan_oracle.py",
        selector="tests/test_execute_plan_oracle.py",
    ),
    CorpusFix(
        function="handle_ci_sync_account_costs",
        sha="aa8f8490f0a9df5d40dd67742d59c6658ce644b2",
        source_relpath=(
            "corpus_fixes/handle_ci_sync_account_costs/"
            "test_handle_ci_sync_account_costs_oracle.py"
        ),
        destination_relpath="tests/test_handle_ci_sync_account_costs_oracle.py",
        selector="tests/test_handle_ci_sync_account_costs_oracle.py",
    ),
    CorpusFix(
        function="trigger_account_cost_sync",
        sha="90217bb59dde4c3dd39edfcd4269632f41fa4092",
        source_relpath=(
            "corpus_fixes/trigger_account_cost_sync/"
            "test_trigger_account_cost_sync_oracle.py"
        ),
        destination_relpath="tests/test_trigger_account_cost_sync_oracle.py",
        selector="tests/test_trigger_account_cost_sync_oracle.py",
    ),
    CorpusFix(
        function="_mark_regressed",
        sha="a9af54041907d184cf90aca62a57cc758dce2b1c",
        source_relpath=(
            "corpus_fixes/_mark_regressed/"
            "test_mark_regressed_oracle.py"
        ),
        destination_relpath="tests/test_mark_regressed_oracle.py",
        selector="tests/test_mark_regressed_oracle.py",
    ),
    CorpusFix(
        function="simulate_unattended",
        sha="c3d052462a7c7a0906531fb6cff57093683cb694",
        source_relpath=(
            "corpus_fixes/simulate_unattended/"
            "test_simulate_unattended_oracle.py"
        ),
        destination_relpath="tests/test_simulate_unattended_oracle.py",
        selector="tests/test_simulate_unattended_oracle.py",
    ),
    CorpusFix(
        function="create_review_session",
        sha="4aa427e91cb340c1d91c45f2073093ef22f546c5",
        source_relpath=(
            "corpus_fixes/create_review_session/"
            "test_create_review_session_oracle.py"
        ),
        destination_relpath="tests/test_create_review_session_oracle.py",
        selector="tests/test_create_review_session_oracle.py",
    ),
    CorpusFix(
        function="_escalate_unrescuable_tasks",
        sha="25af585010e2ee9b62a39d420e24acbc61496036",
        source_relpath=(
            "corpus_fixes/_escalate_unrescuable_tasks/"
            "test_escalate_unrescuable_tasks_oracle.py"
        ),
        destination_relpath="tests/test_escalate_unrescuable_tasks_oracle.py",
        selector="tests/test_escalate_unrescuable_tasks_oracle.py",
    ),
    CorpusFix(
        function="run_probe_pack",
        sha="bdb1bfb48a1189eeb5874aac9609675e65fcb8e9",
        source_relpath=(
            "corpus_fixes/run_probe_pack/"
            "test_run_probe_pack_oracle.py"
        ),
        destination_relpath="tests/test_run_probe_pack_oracle.py",
        selector="tests/test_run_probe_pack_oracle.py",
    ),
    CorpusFix(
        function="_tool_resolve_decision",
        sha="a91283b0f53da402db2a13ab9ae9bce63b09ac29",
        source_relpath=(
            "corpus_fixes/_tool_resolve_decision/"
            "test_tool_resolve_decision_oracle.py"
        ),
        destination_relpath="tests/test_tool_resolve_decision_oracle.py",
        selector="tests/test_tool_resolve_decision_oracle.py",
    ),
    CorpusFix(
        function="_build_domain_brief",
        sha="461710e66bee707bb3f879bc4cc765f9ee708f6f",
        source_relpath=(
            "corpus_fixes/_build_domain_brief/"
            "test_build_domain_brief_oracle.py"
        ),
        destination_relpath="tests/test_build_domain_brief_oracle.py",
        selector="tests/test_build_domain_brief_oracle.py",
    ),
    CorpusFix(
        function="load_run_recipe",
        sha="e13f15813425e98f6e95d260a45e6930f1767f5b",
        source_relpath=(
            "corpus_fixes/load_run_recipe/"
            "test_load_run_recipe_oracle.py"
        ),
        destination_relpath="tests/test_load_run_recipe_oracle.py",
        selector="tests/test_load_run_recipe_oracle.py",
    ),
    CorpusFix(
        function="sync_workspace",
        sha="45384668cd367f337d27b4991324fcb907e57568",
        source_relpath=(
            "corpus_fixes/sync_workspace/"
            "test_sync_workspace_oracle.py"
        ),
        destination_relpath="tests/test_sync_workspace_oracle.py",
        selector="tests/test_sync_workspace_oracle.py",
    ),
    CorpusFix(
        function="_dispatch_stage_turn",
        sha="167c24ea57dd017b5b8513971124b68cf894e097",
        source_relpath=(
            "corpus_fixes/_dispatch_stage_turn/"
            "test_dispatch_stage_turn_oracle.py"
        ),
        destination_relpath="tests/test_dispatch_stage_turn_oracle.py",
        selector="tests/test_dispatch_stage_turn_oracle.py",
    ),
    CorpusFix(
        function="_run_list_out",
        sha="8a54b9f1accdee9daad04600472255e3aa930a17",
        source_relpath=(
            "corpus_fixes/_run_list_out/"
            "test_run_list_out_oracle.py"
        ),
        destination_relpath="tests/test_run_list_out_oracle.py",
        selector="tests/test_run_list_out_oracle.py",
    ),
    CorpusFix(
        function="_serve_dulwich",
        sha="26d27494d178790899e624be428a00a2897fbdad",
        source_relpath=(
            "corpus_fixes/_serve_dulwich/"
            "test_serve_dulwich_oracle.py"
        ),
        destination_relpath="tests/test_serve_dulwich_oracle.py",
        selector="tests/test_serve_dulwich_oracle.py",
    ),
    CorpusFix(
        function="advance",
        sha="36d50b1e1649249af14bcbc24734a474048d08af",
        source_relpath=(
            "corpus_fixes/advance/"
            "test_advance_oracle.py"
        ),
        destination_relpath="tests/test_advance_oracle.py",
        selector="tests/test_advance_oracle.py",
    ),
    CorpusFix(
        function="validate",
        sha="db1c79e1544ae7afe727e84e625a42694be5b7b3",
        source_relpath=(
            "corpus_fixes/validate/"
            "test_validate_oracle.py"
        ),
        destination_relpath="tests/test_validate_oracle.py",
        selector="tests/test_validate_oracle.py",
    ),
    CorpusFix(
        function="score_unfaithful_reasoning",
        sha="0e3101c6d35a2b9f65383b1ba65e62da5e9e9d5a",
        source_relpath=(
            "corpus_fixes/score_unfaithful_reasoning/"
            "test_score_unfaithful_reasoning_oracle.py"
        ),
        destination_relpath="tests/test_score_unfaithful_reasoning_oracle.py",
        selector="tests/test_score_unfaithful_reasoning_oracle.py",
    ),
    CorpusFix(
        function="_dependency_gate",
        sha="707d1067e20f3acf86d016beb3063ff0a66ef276",
        source_relpath=(
            "corpus_fixes/_dependency_gate/"
            "test_dependency_gate_oracle.py"
        ),
        destination_relpath="tests/test_dependency_gate_oracle.py",
        selector="tests/test_dependency_gate_oracle.py",
    ),
    CorpusFix(
        function="create_proposed_run",
        sha="edc1bb1f4b06700a8fe192b98c570a5264defbd6",
        source_relpath=(
            "corpus_fixes/create_proposed_run/"
            "test_create_proposed_run_oracle.py"
        ),
        destination_relpath="tests/test_create_proposed_run_oracle.py",
        selector="tests/test_create_proposed_run_oracle.py",
    ),
    CorpusFix(
        function="_find_mr",
        sha="471566a4b2363bfc699f007e96f3eb19613803e7",
        source_relpath=(
            "corpus_fixes/_find_mr/"
            "test_find_mr_oracle.py"
        ),
        destination_relpath="tests/test_find_mr_oracle.py",
        selector="tests/test_find_mr_oracle.py",
    ),
    CorpusFix(
        function="record_deployment",
        sha="f515352efb41440990c179b8abb826f473f09c03",
        source_relpath=(
            "corpus_fixes/record_deployment/"
            "test_record_deployment_oracle.py"
        ),
        destination_relpath="tests/test_record_deployment_oracle.py",
        selector="tests/test_record_deployment_oracle.py",
    ),
    CorpusFix(
        function="_run_did_work",
        sha="567dce1581fcc05b7d522ffcfac881c29cdd40c3",
        source_relpath=(
            "corpus_fixes/_run_did_work/"
            "test_run_did_work_oracle.py"
        ),
        destination_relpath="tests/test_run_did_work_oracle.py",
        selector="tests/test_run_did_work_oracle.py",
    ),
    CorpusFix(
        function="_create_run_session",
        sha="9f8a25de76fdeb640b14d94dba7ce55079717594",
        source_relpath=(
            "corpus_fixes/_create_run_session/"
            "test_create_run_session_oracle.py"
        ),
        destination_relpath="tests/test_create_run_session_oracle.py",
        selector="tests/test_create_run_session_oracle.py",
    ),
    CorpusFix(
        function="seed_data_science_team",
        sha="93311766555c78cdf0d4d828bda2f95ec9a8e57f",
        source_relpath=(
            "corpus_fixes/seed_data_science_team/"
            "test_seed_data_science_team_oracle.py"
        ),
        destination_relpath="tests/test_seed_data_science_team_oracle.py",
        selector="tests/test_seed_data_science_team_oracle.py",
    ),
    CorpusFix(
        function="seed_creative_team",
        sha="978d4c73d8d24207f8e949d0d81e82d8c22fbb5f",
        source_relpath=(
            "corpus_fixes/seed_creative_team/"
            "test_seed_creative_team_oracle.py"
        ),
        destination_relpath="tests/test_seed_creative_team_oracle.py",
        selector="tests/test_seed_creative_team_oracle.py",
    ),
    CorpusFix(
        function="_env_guidance",
        sha="725282e300639060b28018f9204b92827b2a543c",
        source_relpath=(
            "corpus_fixes/_env_guidance/"
            "test_env_guidance_oracle.py"
        ),
        destination_relpath="tests/test_env_guidance_oracle.py",
        selector="tests/test_env_guidance_oracle.py",
    ),
    CorpusFix(
        function="_compose_test",
        sha="c63265b3890028fbd85c3e772929e4f43c5de12b",
        source_relpath=(
            "corpus_fixes/_compose_test/"
            "test_compose_test_oracle.py"
        ),
        destination_relpath="tests/test_compose_test_oracle.py",
        selector="tests/test_compose_test_oracle.py",
    ),
    CorpusFix(
        function="_resolve_workspace",
        sha="c273277ba68c87082440017feba658f194767013",
        source_relpath=(
            "corpus_fixes/_resolve_workspace/"
            "test_resolve_workspace_oracle.py"
        ),
        destination_relpath="tests/test_resolve_workspace_oracle.py",
        selector="tests/test_resolve_workspace_oracle.py",
    ),
    CorpusFix(
        function="_retry_note",
        sha="513672219dfd9ebf6e3f5e47483a6a0529b16d4a",
        source_relpath=(
            "corpus_fixes/_retry_note/"
            "test_retry_note_oracle.py"
        ),
        destination_relpath="tests/test_retry_note_oracle.py",
        selector="tests/test_retry_note_oracle.py",
    ),
)


def corpus_fix_for(task_row: Mapping[str, object]) -> CorpusFix | None:
    function = task_row.get("fn")
    sha = task_row.get("sha")
    if not isinstance(function, str) or not isinstance(sha, str):
        return None
    return next(
        (
            fix
            for fix in CORPUS_FIXES
            if fix.function == function and fix.sha == sha
        ),
        None,
    )


def apply_corpus_fixes(root: Path, task_row: Mapping[str, object]) -> int:
    """Copy the exact replacement oracle for ``task_row`` into ``root``.

    Returns the number of files copied, matching ``converge.overlay_fix_tests``.
    The destination is always under the caller-owned throwaway sandbox.
    """
    fix = corpus_fix_for(task_row)
    if fix is None:
        return 0

    source = POC / fix.source_relpath
    if not source.is_file():
        raise FileNotFoundError(f"corpus replacement is missing: {source}")

    destination = Path(root) / fix.destination_relpath
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return 1


def replacement_selector(task_row: Mapping[str, object]) -> str | None:
    """Return a selector rooted at the ConvergeRunner backend sandbox."""
    fix = corpus_fix_for(task_row)
    return fix.selector if fix is not None else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="sandbox backend root")
    parser.add_argument(
        "--task-row",
        type=Path,
        required=True,
        help="JSON file containing one task object (a one-row JSONL is valid)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    task_row = json.loads(args.task_row.read_text())
    if not isinstance(task_row, dict):
        raise SystemExit("task row must be a JSON object")
    count = apply_corpus_fixes(args.root, task_row)
    selector = replacement_selector(task_row)
    print(f"applied={count} selector={selector or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
