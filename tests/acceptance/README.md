# Offline agent-tool acceptance suite

Run fixture and contract checks with the machine-wide gate:

```bash
bash /Users/marida/Repos/farside/sf-work/pytest_gate.sh /Users/marida/Repos/farside/lanes/acceptance tests/acceptance/test_fixture_integrity.py -q
```

Run the end-to-end suite on a host that allows local sockets and `sandbox-exec` or `bwrap`, after checking that engine `run_tests` uses the declared non-pytest check command. The current engine's `run_tests` starts pytest, which would overlap the outer pytest process and violate the machine-wide single-pytest rule; the integrator must resolve that before running this command:

```bash
bash /Users/marida/Repos/farside/sf-work/pytest_gate.sh /Users/marida/Repos/farside/lanes/acceptance tests/acceptance -m host_sandbox -q
```

Every end-to-end test launches the public `stateframe run` CLI against a scripted local endpoint. It inspects the recorded model requests, reply text, working copy, summary, final diff, and original repository where relevant. `force_endpoint.py` uses the same scripted endpoint handler but forces a stale or hallucinated tool call so the engine's off-menu refusal can be tested. The normal endpoint rejects such a call before sending it. `pause_endpoint.py` holds the second model reply to make SIGKILL and resume deterministic after a completed shell step.

The fixture repositories under `fixtures/` are excluded from outer pytest collection. The `host_sandbox` marker is registered by `tests/conftest.py`.

`E22` uses `python -m pytest` *inside* the agent shell. Running that inside pytest would start a second pytest process, violating the machine-wide safety rule. Its scripted scenario is provided in `fixtures/scripts/pytest_completion.script.json` and must be run as a standalone end-to-end command when no pytest process is running:

```bash
run_dir=$(mktemp -d /tmp/sf-e22.XXXXXX)
bash /Users/marida/Repos/farside/sf-work/lanes/e2e.sh /Users/marida/Repos/farside/lanes/acceptance tests/acceptance/fixtures/scripts/pytest_completion.script.json tests/acceptance/fixtures/simple_repo 'Verify shell pytest does not complete the task' --run-dir "$run_dir" --test-python /Users/marida/Repos/farside/sf-work/venv/bin/python --test-cmd 'python -m unittest discover -s tests -q' --shell standard
```

| Test | Acceptance row |
|---|---|
| `test_shell_workflows.py::test_u1_*` | U1, read-only refusal |
| `test_shell_workflows.py::test_u2_*` | U2 |
| `test_shell_workflows.py::test_u3_u4_*`, `test_u3_node_*` | U3, U4 |
| `test_shell_workflows.py::test_u5_*` | U5, E17 |
| `test_shell_workflows.py::test_u6_*` | U6, E23 environment visibility |
| `test_shell_workflows.py::test_u7_*` | U7, U8, E18 |
| `test_shell_workflows.py::test_u10_*` | U10 with shell effects |
| `test_git_web.py::test_u9_*` | U9, secret history, non-Git refusal |
| `test_git_web.py::test_u11_*` | U11, E26 private address |
| `test_jobs_custom.py::test_u12_*` | U12, jobs panel, cleanup |
| `test_jobs_custom.py::test_u14_*` | U14, schema, argv, effects, BAD_ARGUMENTS |
| `test_jobs_custom.py::test_u15_*` | U15 |
| `test_effects_policy.py::test_e1_e2_*` | E1, E2 command smuggling |
| `test_effects_policy.py::test_e7_*` | E7 |
| `test_effects_policy.py::test_e8_*` | E8 |
| `test_effects_policy.py::test_e9_*` | E9 |
| `test_effects_policy.py::test_e10_*` | E10 |
| `test_effects_policy.py::test_e11_e20_*` | E11, E20 |
| `test_effects_policy.py::test_e12_*` | E12 bounded child tree |
| `test_effects_policy.py::test_e13_*` | E13 |
| `test_effects_policy.py::test_e14_*` | E14 |
| `test_additional_edges.py::test_u13_*` | U13 (`jq`, when installed) |
| `test_additional_edges.py::test_e3_e5_*` | E3, E5 |
| `test_additional_edges.py::test_e4_*` | E4 |
| `test_additional_edges.py::test_e6_*` | E6 |
| `test_additional_edges.py::test_e15_*` | E15 |
| `test_additional_edges.py::test_e26_*` | E26 direct private addresses |
| `test_additional_edges.py::test_e30_*` | E30 |
| `test_resume_completion.py::test_e16_*` | E16, local replay and next request |
| `test_resume_completion.py::test_e19_*` | E19, volatile output replay |
| `test_resume_completion.py::test_e23_*` | E23, external replay refusal |
| `test_resume_completion.py::test_e21_*` | E21 |
| `fixtures/scripts/pytest_completion.script.json` standalone | E22 |
| `test_resume_completion.py::test_e24_*` | E24 |
| `test_fixture_integrity.py::test_e25_*` | E25, old run spec refusal |
| `test_jobs_custom.py::test_e29_*` | E29 |
| `test_resume_completion.py::test_all_tool_schemas_*` | Frame budget with shell, Git, web, custom tools |

The catalog's LIVE network install and public web fetch cases require an external service and are outside this offline suite. E27 and E28 need a reachable public test server: the required SSRF policy refuses local fixture servers before their response bodies can be observed. E31 needs the host suite on both macOS and Linux. E32 needs a host with sandbox tooling intentionally unavailable. The prompt-budget test requires the integrated summary to report `max_request_tokens`.
