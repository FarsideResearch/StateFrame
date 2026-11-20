# Startup schema deadline review, round 4

T1: N/A: switch-registry.json changes only sources_sha256; the switch list is identical to 9a75da09 and the series introduces no STATEFRAME_* variables or feature switch. signed: crew-14
T2: N/A: no added or changed feature switch has an OFF arm; registry content differs only in the source digest. signed: crew-14
T3: tests/test_schema_metering_deadline.py::test_startup_fallback_is_journaled_and_replayed_byte_for_byte
T6: tests/test_schema_metering_interruptions.py::test_archive_interruption_preserves_schema_record tests/test_schema_metering_interruptions.py::test_live_restart_interrupted_after_tool_tape_move_keeps_external_authorization tests/test_mcp_integration.py::test_resume_keeps_mcp_admission_when_tokenizer_counts_change tests/test_frame_routes_replay.py::test_retry_cannot_drop_the_pending_request_after_missing_startup_counts
T8: tests/test_schema_metering_deadline.py::test_replay_refuses_a_journal_without_startup_counts tests/test_schema_metering_deadline.py::test_bind_bounds_schema_metering_and_keeps_fast_counts
T9: tests/test_mcp_integration.py::test_resume_keeps_mcp_admission_when_tokenizer_counts_change RED: d87dc0b tests/test_schema_metering_interruptions.py::test_live_restart_interrupted_after_tool_tape_move_keeps_external_authorization RED: 0a7dfc9
T10: tests/test_schema_metering_deadline.py::test_parallel_binds_share_one_tokenizer_with_bounded_workers tests/test_schema_metering_deadline.py::test_stuck_address_resolution_does_not_hold_interpreter_open
T12: tests/test_schema_metering_deadline.py::test_startup_fallback_is_journaled_and_replayed_byte_for_byte tests/test_mcp_integration.py::test_resume_keeps_mcp_admission_when_tokenizer_counts_change tests/test_schema_metering_interruptions.py::test_live_restart_interrupted_after_tool_tape_move_keeps_external_authorization

## Interruption inventory

Every archive boundary has a separate parameter in test_archive_interruption_preserves_schema_record:

- Archive directory creation and spec copy.
- Each ATTEMPT_FILES move: resume-pending-verified.json, worker, wire, requests, sends.jsonl, engine.log, engine-result.json, child-traceback.txt, summary.json, final.diff, tests-after.txt, resume-divergence.json, workspace, checks, checkpoint-refused.json, env, env-sites.json, shell-home, shell-cache, shell-ask.json, shell-answer.json, hooks.
- Checkpoint archival and replacement of the live spec.

The external-operation regression covers the additional live-restart tool-tape move. The MCP regression checks discovery after workspace archival and after spec publication.

Existing startup tests exercise interruption before any send or journal, pending-request replay, missing startup counts, repeated retry before any send, partial worker and wire archives, checkpoint bypass, and exact replay of the journaled fallback record.

The parallel-bind matrix now covers warm and cold capability caches against a shared trickling tokenizer: body, headers, chunk framing, partial exact results, and staggered starts. The blocked-resolution subprocess test verifies interpreter exit.

## Validation

Release source verification: regen_engine_source.py --check reports stale=False. No engine source was edited in this round.

The requested manifest destination outside this checkout was read-only. This file is the writable delivery copy.
