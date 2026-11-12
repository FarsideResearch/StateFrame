"""Raw no-hooks parity against this candidate's immediate base release.

SF_TEST_BASELINE_IMMEDIATE may name an explicit checkout. Otherwise use the in-tree merge base with
origin/main when it is a strict ancestor; without that ref, use the newest ancestor release commit
(excluding HEAD) whose subject identifies a StateFrame version or release. Both sides explicitly
disable the stable tool head and v0.4.0 final cache layout, which changes wire order: this
parity is about hooks, not tool lists or cache layout.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib import error, request
from urllib.parse import urlsplit

import pytest

from _timeouts import scaled
from acceptance._support import assert_tree_imports, tree_python

from _journal import measurement_events
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, serve
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from tools.compare_no_mcp_requests import (
    DERIVED_RESULT_FIELDS, MOVED_ID_FIELDS, changed_id_chars, id_length_deltas,
    planted_body_change_is_caught, request_id_handle_mismatches,
    release_request_schema_alignment, release_wire_id_drift,
    unbacked_result_hash_changes, wire_id_drift)
from tools.immediate_baseline import baseline_checkout


def _assert_journal_kinds_match_release(baseline_kinds, current_kinds):
    """Allow one extra of each event only when that kind is absent from the baseline."""
    trimmed = list(current_kinds)
    done_kind = "done_facts_finalized"
    baseline_done = baseline_kinds.count(done_kind)
    current_done = trimmed.count(done_kind)
    assert current_done == baseline_done or (baseline_done == 0 and current_done == 1), (
        done_kind, baseline_done, current_done)
    if current_done > baseline_done:
        del trimmed[len(trimmed) - 1 - trimmed[::-1].index(done_kind)]
    dialect_kind = "provider_dialect"
    baseline_dialect = baseline_kinds.count(dialect_kind)
    current_dialect = trimmed.count(dialect_kind)
    assert current_dialect == baseline_dialect or (
        baseline_dialect == 0 and current_dialect == 1
    ), (dialect_kind, baseline_dialect, current_dialect)
    if current_dialect > baseline_dialect:
        del trimmed[len(trimmed) - 1 - trimmed[::-1].index(dialect_kind)]
    assert trimmed == baseline_kinds


def test_journal_kind_parity_rejects_unexpected_extra_kind():
    baseline = ["run_started", "request_sent", "prompt_cache_accounting", "run_done"]
    allowed = ["run_started", "provider_dialect", "request_sent",
               "prompt_cache_accounting", "done_facts_finalized", "run_done"]
    _assert_journal_kinds_match_release(baseline, allowed)
    _assert_journal_kinds_match_release(
        [*baseline[:1], "provider_dialect", *baseline[1:]], allowed)
    _assert_journal_kinds_match_release(baseline, [kind for kind in allowed
                                                   if kind != "done_facts_finalized"])
    baseline_with_done = [*baseline[:-1], "done_facts_finalized", baseline[-1]]
    _assert_journal_kinds_match_release(baseline_with_done, allowed)
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(baseline_with_done,
                                            [*allowed[:-1], "done_facts_finalized", allowed[-1]])
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(baseline_with_done,
                                            [kind for kind in allowed if kind != "done_facts_finalized"])
    baseline_with_dialect = [*baseline[:1], "provider_dialect", *baseline[1:]]
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(baseline_with_dialect,
                                            [*allowed, "provider_dialect"])
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(
            baseline_with_dialect,
            [kind for kind in allowed if kind != "provider_dialect"])
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(baseline, [*allowed, "provider_dialect"])
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(baseline, [*allowed, "unexpected_kind"])
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(baseline, [*allowed, "provider_feature_degraded"])
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release([*baseline, "unexpected_kind"], allowed)
    with pytest.raises(AssertionError):
        _assert_journal_kinds_match_release(baseline, [kind for kind in allowed
                                                        if kind != "prompt_cache_accounting"])


def _assert_spec_keys_match_release(baseline_keys, current_keys):
    # The fixture proves llama.cpp via /props, so the candidate records that proof in spec.json.
    expected = baseline_keys | RUNNER_SPEC_FIELDS
    if "provider_capabilities" not in baseline_keys:
        expected.add("provider_capabilities")
    assert current_keys == expected


def test_spec_key_parity_rejects_unexpected_extra_key():
    baseline = {"schema", "chat_url"}
    allowed = baseline | RUNNER_SPEC_FIELDS | {"provider_capabilities"}
    _assert_spec_keys_match_release(baseline, allowed)
    baseline_with_capabilities = baseline | {"provider_capabilities"}
    _assert_spec_keys_match_release(
        baseline_with_capabilities, baseline_with_capabilities | RUNNER_SPEC_FIELDS)
    with pytest.raises(AssertionError):
        _assert_spec_keys_match_release(
            baseline_with_capabilities,
            (baseline_with_capabilities | RUNNER_SPEC_FIELDS) - {"provider_capabilities"})
    with pytest.raises(AssertionError):
        _assert_spec_keys_match_release(baseline, baseline | RUNNER_SPEC_FIELDS)
    with pytest.raises(AssertionError):
        _assert_spec_keys_match_release(baseline, allowed | {"unexpected_spec_key"})


def _assert_proven_fixture_capabilities(run: Path) -> None:
    spec = json.loads((run / "spec.json").read_text())
    endpoint = urlsplit(spec["chat_url"])
    origin = f"{endpoint.scheme}://{endpoint.netloc}"
    assert spec["provider_capabilities"] == {
        origin: {
            "dialect": "llama.cpp",
            "slots": False,
            "slot_ids": [],
            "reason": "proven by /props",
            "props_status": 200,
            "slots_status": 404,
            "reasoning_controls": "llama.cpp reasoning controls unchanged",
        },
    }


def _assert_unpinned_without_degradation(run: Path) -> None:
    slot = json.loads((run / "spec.json").read_text())["prompt_cache_slot"]
    assert slot["selection"] == "not_pinned" and slot["slot"] is None
    assert slot.get("degraded") is not True


def _assert_automatic_idle_pin(run: Path, requests: list[bytes]) -> None:
    spec = json.loads((run / "spec.json").read_text())
    assert spec["prompt_cache_slot"]["selection"] == "automatic_idle"
    assert spec["prompt_cache_slot"]["slot"] == 0
    assert spec["prompt_cache_slot"].get("degraded") is not True
    assert spec["environment"]["REACT_SLOT_ID"] == "0"
    assert requests
    assert all(json.loads(body)["id_slot"] == 0 and json.loads(body)["cache_prompt"] is True
               for body in requests)


def _assert_parity_endpoint_shape(origin: str, slots: list[int] | None) -> None:
    with request.urlopen(origin + "/props", timeout=scaled(2)) as response:
        props = json.load(response)
    assert props["default_generation_settings"]["n_ctx"] == 65536
    assert props["model_path"] == "scripted-endpoint"
    if slots is None:
        with pytest.raises(error.HTTPError) as missing:
            request.urlopen(origin + "/slots", timeout=scaled(2))
        assert missing.value.code == 404
    else:
        with request.urlopen(origin + "/slots", timeout=scaled(2)) as response:
            actual = json.load(response)
        assert actual == [{"id": index, "n_ctx": size, "is_processing": False}
                          for index, size in enumerate(slots)]


def _is_size_count(key, value) -> bool:
    return (isinstance(key, str) and type(value) is int
            and any(word in key for word in ("bytes", "tokens", "chars")))


def _size_count_problems(old, new, delta, path="", changed_chars=0) -> list[str]:
    """Walk two payloads together.

    A byte/char count may move only with the request's ID-length change `delta`: same
    direction, never by more. A token count follows the IDs' characters, not just their
    length, so it may move either way, but never by more than `changed_chars` (the characters
    in IDs whose value changed). Everything else is left for the masked byte comparison.
    """
    problems = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in old.keys() & new.keys():
            if _is_size_count(key, old[key]) and _is_size_count(key, new[key]):
                change = new[key] - old[key]
                if "tokens" in key:
                    bad = abs(change) > changed_chars
                else:
                    bad = change != 0 and (change * delta < 0 or abs(change) > abs(delta))
                if bad:
                    problems.append(f"{path}.{key}: {old[key]} -> {new[key]} with ID-length change "
                                    f"{delta}, {changed_chars} changed ID characters")
            else:
                problems += _size_count_problems(old[key], new[key], delta, f"{path}.{key}",
                                                 changed_chars)
    elif isinstance(old, list) and isinstance(new, list):
        for index, (left, right) in enumerate(zip(old, new)):
            problems += _size_count_problems(left, right, delta, f"{path}[{index}]", changed_chars)
    return problems


def _without_size_counts(value):
    if isinstance(value, dict):
        return {key: _without_size_counts(item) for key, item in value.items()
                if not _is_size_count(key, item)}
    if isinstance(value, list):
        return [_without_size_counts(item) for item in value]
    return value


def _assert_diet_payloads_match_release(baseline_payloads, current_payloads,
                                        baseline_requests, current_requests):
    """request_diet_applied payloads name their request and record its sizes.

    Only the named IDs may move. Every request body's size change must be exactly its
    wire-ID length change (older releases suffixed IDs with ':<sequence>'). Size, token and
    char counts in the payload may move only for such a request, in the same direction and
    by no more than that change; with no ID-length change they must be equal.
    """
    aligned_requests = release_request_schema_alignment(
        list(baseline_requests), list(current_requests))
    deltas, problems = id_length_deltas(aligned_requests, list(current_requests))
    assert not problems, problems
    changed = changed_id_chars(list(baseline_requests), list(current_requests))
    assert len(baseline_payloads) == len(current_payloads)
    for index, (old, new) in enumerate(zip(baseline_payloads, current_payloads)):
        delta = deltas[index] if index < len(deltas) else 0
        chars = changed[index] if index < len(changed) else 0
        old = copy.deepcopy(old)
        if index < len(aligned_requests):
            schema_bytes = len(aligned_requests[index]) - len(baseline_requests[index])
            if schema_bytes:
                from kernel_context import token_estimate

                old_tools = json.loads(baseline_requests[index])["tools"]
                new_tools = json.loads(current_requests[index])["tools"]
                schema_tokens = token_estimate(new_tools) - token_estimate(old_tools)
                for field in ("request_bytes_before", "request_bytes_after"):
                    old[field] += schema_bytes
                old_sections = {row["section"]: row for row in old["sections"]}
                new_sections = {row["section"]: row for row in new["sections"]}
                for field in ("before_bytes", "after_bytes"):
                    old_sections["schemas"][field] += schema_bytes
                    assert old_sections["schemas"][field] == new_sections["schemas"][field]
                for field in ("before_tokens", "after_tokens"):
                    old_sections["schemas"][field] += schema_tokens
                    assert old_sections["schemas"][field] == new_sections["schemas"][field]
                for field in ("request_tokens_before", "request_tokens_after"):
                    observed = new[field] - old[field]
                    # Whole-request proxy rounding can differ by one token after
                    # the exact schema-section delta; ID changes retain their bound.
                    assert abs(observed - schema_tokens) <= chars + 1
                    old[field] = new[field]
        problems = _size_count_problems(old, new, delta, f"payload[{index}]", chars)
        assert not problems, problems
    encode = lambda rows: json.dumps(_without_size_counts(rows), sort_keys=True).encode()  # noqa: E731
    problems, kinds = wire_id_drift([encode(baseline_payloads)], [encode(current_payloads)])
    assert not problems, problems
    assert set(kinds) <= MOVED_ID_FIELDS, kinds


def test_release_diet_counts_allow_only_verified_schema_effects():
    from kernel_context import token_estimate
    from tools.compare_no_mcp_requests import _OLD_BATCH_STR_REPLACE, _SCALAR_STR_REPLACE

    def request(tool):
        return json.dumps({"tools": [tool], "messages": [{"role": "user", "content": "task"}]},
                          separators=(",", ":")).encode()

    baseline_request = request(_OLD_BATCH_STR_REPLACE)
    current_request = request(_SCALAR_STR_REPLACE)
    byte_change = len(current_request) - len(baseline_request)
    token_change = token_estimate([_SCALAR_STR_REPLACE]) - token_estimate([_OLD_BATCH_STR_REPLACE])
    baseline = {"kind": "request_diet_applied", "request_bytes_before": 1000,
                "request_bytes_after": 1000, "request_tokens_before": 300,
                "request_tokens_after": 300, "sections": [
                    {"section": "schemas", "before_bytes": 500, "after_bytes": 500,
                     "before_tokens": 125, "after_tokens": 125},
                    {"section": "task", "before_bytes": 50, "after_bytes": 50,
                     "before_tokens": 13, "after_tokens": 13}]}
    current = copy.deepcopy(baseline)
    for field in ("request_bytes_before", "request_bytes_after"):
        current[field] += byte_change
    for field in ("request_tokens_before", "request_tokens_after"):
        current[field] += token_change + 1  # Whole-request proxy rounding.
    for field in ("before_bytes", "after_bytes"):
        current["sections"][0][field] += byte_change
    for field in ("before_tokens", "after_tokens"):
        current["sections"][0][field] += token_change
    _assert_diet_payloads_match_release([baseline], [current],
                                        [baseline_request], [current_request])
    wrong_section = copy.deepcopy(current)
    wrong_section["sections"][1]["after_bytes"] += 1
    with pytest.raises(AssertionError):
        _assert_diet_payloads_match_release([baseline], [wrong_section],
                                            [baseline_request], [current_request])
    wrong_token_count = copy.deepcopy(current)
    wrong_token_count["request_tokens_after"] += 2
    with pytest.raises(AssertionError):
        _assert_diet_payloads_match_release([baseline], [wrong_token_count],
                                            [baseline_request], [current_request])


def _assert_only_wire_ids_moved(baseline_requests, current_requests, current_run=None,
                                baseline_run=None):
    """The receipt-determinism change (11-03) moved request_id and the receipt IDs that are
    derived like it. Everything else must still be byte-identical to the base release."""
    problems, kinds = release_wire_id_drift(list(baseline_requests), list(current_requests))
    assert not problems, problems
    assert set(kinds) <= MOVED_ID_FIELDS | DERIVED_RESULT_FIELDS, kinds
    # The comparison itself still fails on any other one-byte change.
    assert planted_body_change_is_caught(current_requests[0])
    if current_run is not None:
        assert request_id_handle_mismatches(current_run) == []
    if baseline_run is not None:
        assert current_run is not None
        assert unbacked_result_hash_changes(baseline_run, current_run) == []


CURRENT = Path(__file__).resolve().parents[1]
RUNNER_SPEC_FIELDS = {"check_cmd", "check_source", "protect_tests",
                      "protected_test_globs", "runner_kind"}
pytestmark = pytest.mark.host_sandbox


def _write_launchers(launcher: Path, wrapper: Path) -> None:
    launcher.write_text(
        "import secrets, subprocess, sys\n"
        "from stateframe import cli\n"
        "wrapper, trace, *args = sys.argv[1:]\n"
        "if args and args[0] == 'run':\n"
        "    original_hex = secrets.token_hex\n"
        "    used = False\n"
        "    def fixed_hex(size=None):\n"
        "        global used\n"
        "        if size == 16 and not used:\n"
        "            used = True\n"
        "            return 'a' * 32\n"
        "        return original_hex(size)\n"
        "    secrets.token_hex = fixed_hex\n"
        "original_popen = subprocess.Popen\n"
        "def traced_popen(command, *positional, **keywords):\n"
        "    if len(command) == 5 and list(command[1:3]) == ['-B', '-c'] and command[4] == 'stateframe._child':\n"
        "        import ast, os, re\n"
        "        child_path, run_path = ast.literal_eval(re.search(r'sys\\.argv=(\\[.*?\\]);', command[3]).group(1))\n"
        "        command = [os.path.join(keywords.get('cwd') or '.', command[0]), '-B', child_path, run_path]\n"
        "        keywords['cwd'] = run_path\n"
        "    if len(command) >= 4 and str(command[2]).endswith('_child.py'):\n"
        "        command = [command[0], command[1], wrapper, command[2], command[3], trace]\n"
        "    return original_popen(command, *positional, **keywords)\n"
        "subprocess.Popen = traced_popen\n"
        "raise SystemExit(cli.main(args))\n"
    )
    wrapper.write_text(
        "import json, os, runpy, sys\n"
        "child, run, trace = sys.argv[1:]\n"
        "sys.path.insert(0, os.path.dirname(child))\n"
        "sys.argv = [child, run]\n"
        "try:\n"
        "    runpy.run_path(child, run_name='__main__')\n"
        "finally:\n"
        "    with open(trace, 'w') as target:\n"
        "        json.dump(sorted(sys.modules), target)\n"
    )


def _run_checkout(
    checkout: Path, repo: Path, run: Path, launcher: Path, wrapper: Path, trace: Path,
    *, stable_tool_head: str | None, final_cache_layout: str | None,
    slots: list[int] | None = None,
):
    python = tree_python(checkout / "src",
                         run.parent / f"{run.name}-{checkout.name}-python")
    assert_tree_imports(python, checkout / "src", run.parent)
    document = json.loads((CURRENT / "examples/discount_bug.script.json").read_text())
    if slots is not None:
        document["slots"] = slots
    script = Script(document)
    server = serve(script)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    environment.pop("STATEFRAME_FINAL_CACHE_LAYOUT", None)
    if final_cache_layout is not None:
        environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = final_cache_layout
    try:
        _assert_parity_endpoint_shape(f"http://127.0.0.1:{server.server_port}", slots)
        command = [
            python,
            str(launcher),
            str(wrapper),
            str(trace),
            "run",
            "--repo",
            str(repo),
            "--task",
            "Fix the pricing code so the tests pass.",
            "--endpoint",
            f"http://127.0.0.1:{server.server_port}/v1",
            "--model",
            "scripted",
            "--run-dir",
            str(run),
            "--test-python",
            sys.executable,
            "--max-turns",
            "16",
            "--quiet",
        ]
        result = subprocess.run(
            command, env=environment, capture_output=True, text=True, timeout=scaled(240)
        )
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
        requests = [
            path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
        ]
        assert requests
        events = measurement_events(run)
        kinds = [event["kind"] for event in events]
        # Hash-chain fields and tokenizer measurement/endpoint identity vary
        # between runs because each scripted server uses a fresh port. The
        # allowed journal additions also shift sequence numbers.
        diet_payloads = [
            {key: value for key, value in event.items()
             if key not in {"timestamp_ns", "previous_hash", "record_hash",
                            "run_id", "task_id", "event_id", "sequence",
                            "tokenize_latency_ms", "tokenize_route",
                            "tokenizer_identity"}}
            for event in events if event["kind"] == "request_diet_applied"
        ]
        tape = (
            (run / "tool-tape.jsonl").read_bytes()
            if (run / "tool-tape.jsonl").exists()
            else None
        )
        return (
            requests,
            kinds,
            tape,
            set(json.loads((run / "spec.json").read_text())),
            json.loads(trace.read_text()),
            diet_payloads,
        )
    finally:
        server.shutdown()
        server.server_close()


def test_no_hooks_raw_requests_and_journal_match_release(tmp_path, baseline_checkout, must_run_witness):
    assert (baseline_checkout / "src/stateframe/runner.py").is_file()
    repo = tmp_path / "repo"
    source = CURRENT / "examples/discount_bug"
    import shutil

    shutil.copytree(source, repo)
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _run_checkout(
        baseline_checkout, repo, run, launcher, wrapper, tmp_path / "baseline_modules.json",
        stable_tool_head="off", final_cache_layout="off",
    )
    run.rename(tmp_path / "baseline_run")
    _assert_unpinned_without_degradation(tmp_path / "baseline_run")
    current = _run_checkout(
        CURRENT, repo, run, launcher, wrapper, tmp_path / "current_modules.json",
        stable_tool_head="off", final_cache_layout="off",
    )
    _assert_unpinned_without_degradation(run)
    _assert_only_wire_ids_moved(baseline[0], current[0], run, tmp_path / "baseline_run")
    _assert_journal_kinds_match_release(baseline[1], current[1])
    assert current[2] == baseline[2]
    _assert_spec_keys_match_release(baseline[3], current[3])
    _assert_proven_fixture_capabilities(run)
    assert current[5], "fixture did not exercise request-diet journal payloads"
    _assert_diet_payloads_match_release(baseline[5], current[5], baseline[0], current[0])
    assert "hooks" not in current[3]
    assert not {
        "kernel_hooks",
        "kernel_hook_exec",
        "kernel_hook_driver",
        "kernel_hook_wiring",
    } & set(current[4])
    assert not any(
        (run / name).exists() for name in ("hooks", "hook-bundle", "hook-tape.jsonl")
    )
    must_run_witness.record_parity(
        baseline_trace=baseline[4],
        candidate_trace=current[4],
        expected_trace_prefixes=("stateframe",),
        comparison_completed=True,
    )


def test_no_hooks_regular_stateframe_file_matches_release(tmp_path, baseline_checkout, must_run_witness):
    repo = tmp_path / "repo"
    import shutil

    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    (repo / ".stateframe").write_text("ordinary repository data", encoding="utf-8")
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _run_checkout(baseline_checkout, repo, run, launcher, wrapper, tmp_path / "baseline.json",
                             stable_tool_head="off", final_cache_layout="off")
    run.rename(tmp_path / "baseline_run")
    _assert_unpinned_without_degradation(tmp_path / "baseline_run")
    current = _run_checkout(CURRENT, repo, run, launcher, wrapper, tmp_path / "current.json",
                            stable_tool_head="off", final_cache_layout="off")
    _assert_unpinned_without_degradation(run)
    _assert_only_wire_ids_moved(baseline[0], current[0], run, tmp_path / "baseline_run")
    _assert_journal_kinds_match_release(baseline[1], current[1])
    assert current[2] == baseline[2]
    _assert_spec_keys_match_release(baseline[3], current[3])
    _assert_proven_fixture_capabilities(run)
    _assert_diet_payloads_match_release(baseline[5], current[5], baseline[0], current[0])
    must_run_witness.record_parity(
        baseline_trace=baseline[4],
        candidate_trace=current[4],
        expected_trace_prefixes=("stateframe",),
        comparison_completed=True,
    )


def test_no_hooks_real_shaped_llama_keeps_automatic_pin_and_release_bytes(
    tmp_path, baseline_checkout,
):
    """An idle /slots response pins the default profile on both release and candidate."""
    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _run_checkout(
        baseline_checkout, repo, run, launcher, wrapper, tmp_path / "baseline.json",
        stable_tool_head="off", final_cache_layout="off", slots=[65536],
    )
    _assert_automatic_idle_pin(run, baseline[0])
    run.rename(tmp_path / "baseline_run")
    current = _run_checkout(
        CURRENT, repo, run, launcher, wrapper, tmp_path / "current.json",
        stable_tool_head="off", final_cache_layout="off", slots=[65536],
    )
    _assert_automatic_idle_pin(run, current[0])
    _assert_only_wire_ids_moved(baseline[0], current[0], run, tmp_path / "baseline_run")
    _assert_journal_kinds_match_release(baseline[1], current[1])
    assert "provider_feature_degraded" not in current[1]
    _assert_diet_payloads_match_release(baseline[5], current[5], baseline[0], current[0])


def _resume_checkout(checkout: Path, repo: Path, run: Path, launcher: Path,
                     wrapper: Path, trace: Path, *,
                     stable_tool_head: str | None,
                     final_cache_layout: str | None) -> tuple[list[bytes], list[bytes], list[str]]:
    python = tree_python(checkout / "src",
                         run.parent / f"{run.name}-{checkout.name}-python")
    assert_tree_imports(python, checkout / "src", run.parent)
    class InterruptAfterTwoActions(Script):
        def __init__(self):
            super().__init__({"model": "scripted", "steps": [
                {"tool": "read", "required": True, "arguments": {"path": "shop/pricing.py"}},
                {"tool": "str_replace", "required": True, "arguments": {
                    "path": "shop/pricing.py",
                    "old_str": "    return round(price * percent / 100, 2)",
                    "new_str": "    return round(price * (100 - percent) / 100, 2)",
                }},
                {"tool": "run_tests", "required": True,
                 "arguments": {"target": "tests/test_pricing.py"}},
                {"tool": "done", "required": True, "arguments": {"summary": "Checks finished."}},
            ], "tokenize": {"mode": "ok"}})
            self.waiting = threading.Event()
            self.release = threading.Event()

        def next_call(self, offered):
            if self.position == 2 and not self.release.is_set():
                self.waiting.set()
                if not self.release.wait(scaled(60)):
                    raise TimeoutError("interrupted request was not released")
                raise ScriptStepUnavailable("request interrupted before a response")
            return super().next_call(offered)

    script = InterruptAfterTwoActions()
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    environment.pop("STATEFRAME_FINAL_CACHE_LAYOUT", None)
    if final_cache_layout is not None:
        environment["STATEFRAME_FINAL_CACHE_LAYOUT"] = final_cache_layout
    command = [python, str(launcher), str(wrapper), str(trace), "run",
               "--repo", str(repo), "--task", "Fix the pricing code so the tests pass.",
               "--endpoint", endpoint, "--model", "scripted", "--run-dir", str(run),
               "--test-python", sys.executable, "--max-turns", "16", "--quiet"]
    process = None
    try:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        assert script.waiting.wait(scaled(600)), "first attempt did not reach the third request"
        original = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))
                    if path.with_name(path.name.replace(".request.", ".response.")).exists()]
        assert len(original) == 2
        children = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
        assert len(children) == 1, children
        child = int(children[0])
        identity = record_identity(child, launched_after=0, command_fragment="_child.py")
        signal_run_verified(run, child, identity, signal.SIGKILL,
                            deadline=time.monotonic() + scaled(120), what="hook wiring parity")
        script.release.set()
        process.communicate(timeout=scaled(90))
        resumed_trace = trace.with_name(f"{trace.stem}-resume.json")
        resumed = subprocess.run(
            [python, str(launcher), str(wrapper), str(resumed_trace), "resume", str(run),
             "--endpoint", endpoint],
            env=environment,
            capture_output=True,
            text=True,
            timeout=scaled(240),
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        script.assert_consumed()
        rebuilt = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
        assert rebuilt[:2] == original
        assert resumed_trace.is_file(), "resume did not launch a traced engine child"
        trace_modules = json.loads(resumed_trace.read_text(encoding="utf-8"))
        assert isinstance(trace_modules, list) and all(isinstance(item, str) for item in trace_modules), (
            "resumed child did not write a valid module trace"
        )
        return original, rebuilt, trace_modules
    finally:
        script.release.set()
        if process is not None and process.poll() is None:
            # Failure cleanup may precede the first checkpoint.
            signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                             what="hook parity cleanup", require_checkpoint_saved=False)
            process.communicate(timeout=scaled(10))
        server.shutdown()
        server.server_close()


def test_no_hooks_resume_rebuilt_requests_match_release(tmp_path, baseline_checkout, must_run_witness):
    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(CURRENT / "examples/discount_bug", repo)
    launcher = tmp_path / "launcher.py"
    wrapper = tmp_path / "child_wrapper.py"
    _write_launchers(launcher, wrapper)
    run = tmp_path / "run"
    baseline = _resume_checkout(baseline_checkout, repo, run, launcher, wrapper, tmp_path / "baseline.json",
                                stable_tool_head="off", final_cache_layout="off")
    run.rename(tmp_path / "baseline_run")
    current = _resume_checkout(CURRENT, repo, run, launcher, wrapper, tmp_path / "current.json",
                               stable_tool_head="off", final_cache_layout="off")
    # current/baseline are (original, rebuilt) request lists; resume parity inside one
    # version stays byte-exact (asserted in _resume_checkout), across versions only IDs move.
    _assert_only_wire_ids_moved(baseline[0], current[0])
    _assert_only_wire_ids_moved(baseline[1], current[1], run, tmp_path / "baseline_run")
    must_run_witness.record_parity(
        baseline_trace=baseline[2],
        candidate_trace=current[2],
        expected_trace_prefixes=("stateframe",),
        comparison_completed=True,
    )
