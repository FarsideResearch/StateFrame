"""Exercise the default working set across three provider turns in a large workspace."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import pytest
from _journal import measurement_events
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from engine_run import run_monitored

from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]

TASK = """Add per-workspace retry ceilings to Harbor Ops, our multi-tenant automation platform.

Team routines currently retry transient delivery failures against the same global ceiling.
Operations staff need to be able to use the workspace policy already carried on each dispatch
request. Dispatch must forward the request's workspace_retry_limit to retry_delay instead of
quietly dropping that field. The workspace setting is optional, and 0 is a valid value that
disables retries entirely. An omitted value must continue using the global limit. Keep the
existing behavior for permanent failures, the backoff factor, and the maximum delay.

1. Inspect `harbor_ops/dispatch/jobs/dispatcher.py` for the request-to-policy call site and
   fallback handling. Read `harbor_ops/policies/retry/limits.py` to confirm the argument,
   counting convention, and how exhaustion is represented. Read
   `harbor_ops/operations/routines/scheduler.py` to confirm that
   unattended routines share the dispatch entry point and still handle its return value.
2. Search the platform packages for workspace_retry_limit. The workspace API, admin previews,
   agent tool adapters, queue consumers, and routine planners already carry it in their
   contracts. Verify that the narrow dispatch fix covers every delivery that uses the shared
   entry point. Keep those callers and their request shape intact. Old queued messages with
   no explicit setting should keep working while the change rolls out.
3. Apply the change at the policy call in `harbor_ops/dispatch/jobs/dispatcher.py`. Keep
   retry calculation in the policy package. Preserve the distinction between an exhausted
   retry budget and a delay of zero. Preserve tenant identity on the request and keep logging
   and scheduling contracts stable. No additional storage, worker jobs, or API endpoints
   are required. Keep the patch focused on the dispatch path.
4. Run tests/test_workspace_retry.py. It must cover omission, an explicit ceiling of zero,
   exhaustion at the configured boundary, a transient failure below that boundary, permanent
   failure, and an unattended routine. Leave a brief summary with the check outcome.

Acceptance: a workspace with a ceiling of one still gets a retry after its first attempt
and stops after its second, while a workspace with no override still gets the default three
retries. A disabled workspace must never enqueue a follow-up. The scheduler should consume
the same optional delay returned by manual dispatch and should leave successful and
permanent-failure handling alone.
"""

AREAS = ("agents", "approvals", "audit", "billing", "catalog", "content", "dispatch", "events",
         "identity", "integrations", "knowledge", "notifications", "operations", "policies",
         "projects", "queues", "search", "teams", "tools", "workspaces")
LAYERS = ("api", "services", "workers")
MODULES = ("adapters", "contracts", "models", "routing", "validation")
DISPATCHER = "harbor_ops/dispatch/jobs/dispatcher.py"
RETRY_POLICY = "harbor_ops/policies/retry/limits.py"
SCHEDULER = "harbor_ops/operations/routines/scheduler.py"
LARGE_FILES = (DISPATCHER, RETRY_POLICY, SCHEDULER)
RETRY_FIELD = "workspace_retry_limit"
OLD_CALL = "return retry_delay(attempt, transient=transient)"
NEW_CALL = "return retry_delay(attempt, transient=transient, workspace_retry_limit=workspace_retry_limit)"


def _write(repo: Path, path: str, source: str) -> None:
    file = repo / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(source, encoding="utf-8")


def _policy_rules(rng: random.Random, names: list[str]) -> str:
    rules = []
    for name in names:
        lease = rng.randrange(15, 90)
        quota = rng.randrange(4, 40)
        rules.append(f'\n\ndef {name}(active_jobs: int, lease_seconds: int, *, paused: bool) -> bool:\n'
                     f'    """Check capacity and lease requirements for {name.replace("_", " ")}."""\n'
                     '    if paused or active_jobs < 0:\n'
                     '        return False\n'
                     f'    return active_jobs < {quota} and lease_seconds >= {lease}\n')
    return "".join(rules)


def _make_repo(repo: Path) -> None:
    rng = random.Random(90157)
    for area in AREAS:
        for layer in LAYERS:
            for module in MODULES:
                path = f"harbor_ops/{area}/{layer}/{module}.py"
                source = f'"""{module.capitalize()} for {area} {layer}."""\n'
                if module == "contracts":
                    source += ('\n\nclass DeliveryOptions:\n'
                               '    workspace_id: str\n'
                               '    workspace_retry_limit: int | None = None\n')
                source += _policy_rules(rng, [f"allow_{area}_{layer}_{module}_{i}" for i in range(2)])
                _write(repo, path, source)

    sources = {
        DISPATCHER: '"""Dispatch jobs and route transient failures through the retry policy."""\n'
        'from harbor_ops.policies.retry.limits import retry_delay\n'
        '\n\ndef dispatch(attempt: int, *, transient: bool, workspace_retry_limit: int | None = None) -> int | None:\n'
        f'    {OLD_CALL}\n',
        RETRY_POLICY: '"""Shared backoff and exhaustion policy for workspace jobs."""\n'
        'DEFAULT_RETRY_LIMIT = 3\nMAX_RETRY_DELAY = 60\n'
        '\n\ndef retry_delay(attempt: int, *, transient: bool, workspace_retry_limit: int | None = None) -> int | None:\n'
        '    limit = DEFAULT_RETRY_LIMIT if workspace_retry_limit is None else workspace_retry_limit\n'
        '    if not transient or attempt > limit:\n'
        '        return None\n'
        '    return min(2 ** attempt, MAX_RETRY_DELAY)\n',
        SCHEDULER: '"""Schedule delivery attempts for unattended team routines."""\n'
        'from harbor_ops.dispatch.jobs.dispatcher import dispatch\n'
        '\n\ndef retry_routine(attempt: int, *, transient: bool, workspace_retry_limit: int | None = None) -> int | None:\n'
        '    return dispatch(attempt, transient=transient, workspace_retry_limit=workspace_retry_limit)\n',
    }
    for path, source in sources.items():
        component = Path(path).stem
        names = [f"allow_{component}_{layer}_{module}_{i}"
                 for layer in LAYERS for module in MODULES for i in range(22)]
        _write(repo, path, source + _policy_rules(rng, names))

    _write(repo, "tests/test_workspace_retry.py", """\
from harbor_ops.dispatch.jobs.dispatcher import dispatch
from harbor_ops.operations.routines.scheduler import retry_routine


def test_workspace_retry():
    assert dispatch(3, transient=True) == 8
    assert dispatch(4, transient=True) is None
    assert dispatch(1, transient=True, workspace_retry_limit=0) is None
    assert dispatch(1, transient=True, workspace_retry_limit=1) == 2
    assert dispatch(2, transient=True, workspace_retry_limit=1) is None
    assert dispatch(1, transient=False, workspace_retry_limit=5) is None
    assert retry_routine(2, transient=True, workspace_retry_limit=1) is None
""")


def _script() -> Script:
    # The initial closure includes the three entry points. Read beyond its
    # existing policy and scheduler windows to exercise fresh paging within
    # the native batch's fixed allowance.
    reads = [{"tool": "read", "arguments": {
        "path": path, "start_line": first_line, "end_line": last_line,
    }} for path, first_line, last_line in ((DISPATCHER, 1, 6), (RETRY_POLICY, 150, 151),
                                         (SCHEDULER, 150, 151))]
    return Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {**reads[0], "also": reads[1:]},
        {"tool": "grep", "arguments": {
            "path": "harbor_ops", "pattern": RETRY_FIELD, "mode": "literal", "page_size": 80,
        }},
        {"tool": "str_replace", "arguments": {
            "path": DISPATCHER, "old_str": OLD_CALL, "new_str": NEW_CALL,
        }, "also": [
            {"tool": "run_tests", "arguments": {"target": "tests/test_workspace_retry.py"}},
            {"tool": "done", "arguments": {"summary": "Forwarded the workspace retry limit."}},
        ]},
    ]})


@pytest.mark.host_sandbox
def test_large_repo_default_working_set_completes_three_turns(tmp_path):
    repo = tmp_path / "repo"
    _make_repo(repo)
    sources = list((repo / "harbor_ops").rglob("*.py"))
    assert len(sources) >= 300
    assert all(len((repo / path).read_text().splitlines()) > 2_000 for path in LARGE_FILES)
    assert sum(RETRY_FIELD in path.read_text() for path in sources) >= 60

    run_dir = tmp_path / "run"
    script = _script()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment["STATEFRAME_HOME"] = isolated_stateframe_home()
    environment.pop("LLAMA_CPP_TOKENIZE_URL", None)
    environment["PYTHONPATH"] = str(ROOT / "src")
    server = serve(script)
    try:
        # The test host has taken about 200 s. With a 360 s base ceiling, a
        # heavily loaded local run was still finalizing at 1,414 s. Allow 480 s
        # at base load and let the 160 s base idle window bound stalls.
        result = run_monitored(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", TASK,
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--run-dir", str(run_dir), "--test-python", sys.executable, "--no-run-env", "--quiet"],
            env=environment, run_dir=run_dir, expected_seconds=scaled(160))
    finally:
        server.shutdown()
        server.server_close()
    events = measurement_events(run_dir)
    wires = sorted((run_dir / "wire").glob("*.request.json"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    terminal = summary.get("engine_terminal")
    diag = (f"CLI exit {result.returncode}; received {len(script.requests)} provider requests; "
            f"wire records {[path.name for path in wires]}; terminal={terminal}\n{result.stdout}{result.stderr}")

    main_requests = [request for request in script.requests if request.get("parallel_tool_calls") is True]
    assert main_requests, f"Request 1 did not reach the provider. {diag}"
    initial_contents = [str(message.get("content") or "") for message in main_requests[0]["messages"]]
    for signature in ("def dispatch(", "def retry_delay(", "def retry_routine("):
        assert any(signature in content for content in initial_contents), (signature, diag)
    forbidden = ("FLOOR_INFEASIBLE", "WORKING_SET_ALLOCATION_BLOCKED", "RequestSealError")
    failures = [(event.get("sequence"), event.get("kind"), marker) for event in events
                for marker in forbidden if marker in json.dumps(event)]
    assert not failures and not any(marker in diag for marker in forbidden), (failures, diag)
    assert "frame render overflow" not in (run_dir / "engine.log").read_text(), diag
    partial_closures = [(event.get("sequence"), event["closure_receipt"]) for event in events
                        if event.get("kind") == "request_source_closure"
                        and re.search(r"\b\w+=PARTIAL\b", event["closure_receipt"])]
    assert not partial_closures, (partial_closures, diag)

    assert len(main_requests) == 3, diag
    script.assert_consumed()
    read_paths = {event["identity"]["path"] for event in events
                  if event.get("kind") == "read_result" and event.get("status") in {"OK", "PAGE"}}
    assert set(LARGE_FILES) <= read_paths, (read_paths, diag)
    assert any(event.get("kind") == "action_receipt" and event["receipt"]["kind"] == "grep"
               and event["receipt"]["result_class"] == "HITS" for event in events), diag
    assert result.returncode == 0 and summary["succeeded"], diag
    assert summary["check_outcome"] == "pass", diag
    assert summary["files_changed"] == [DISPATCHER], diag

    budgets = {event["body_sha256"]: event["measurement"] for event in events
               if event.get("kind") == "request_budget_final_wire"
               and event.get("status") == "READY" and isinstance(event.get("measurement"), dict)}
    assert len(wires) == len(script.requests), diag
    for path, request in zip(wires, script.requests, strict=True):
        wire = path.read_bytes()
        assert json.loads(wire) == request, path.name
        digest = "sha256:" + hashlib.sha256(wire).hexdigest()
        assert digest in budgets, f"No ready budget receipt for request {path.name} ({digest})"
        budget = budgets[digest]
        assert len(wire) == budget["request_bytes"] <= budget["max_request_bytes"], budget
        assert 0 < budget["prompt_tokens"] <= budget["max_prompt_tokens"], budget
