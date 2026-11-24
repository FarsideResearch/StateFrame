"""The request seal binds the provider's serialized body, not private routing tags."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import sys
import subprocess
import tempfile
import time
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events
from engine_run import run_monitored, sandbox_test_python
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from kernel_final_wire_allocation import _finalize_provider_request_body
from kernel_provider_dialect import ProviderCapabilities
from kernel_request_measurement import RequestMeasurement
from kernel_request_seal import RequestSealError
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]


def _clone_local_test_python(tmp_path: Path) -> Path:
    """Give the child a sandboxable Python without writing beside the host venv."""
    venv_root = tmp_path / "sandbox-test-python"
    venv.EnvBuilder(system_site_packages=True).create(venv_root)
    return venv_root / "bin" / "python"


def _strict_body(body: dict[str, object]) -> dict[str, object]:
    capture = SimpleNamespace(frame_route=None, journal=SimpleNamespace(manifest={}))
    return _finalize_provider_request_body(
        capture, body, ProviderCapabilities("openai-strict"),
    )


def _measurement(body: dict[str, object]) -> tuple[RequestMeasurement, dict[str, object]]:
    wire = json.dumps(_strict_body(body)).encode()
    measured = RequestMeasurement.bind(
        body, tokenizer_identity="strict-test", request_id="r", logical_request_id="r",
        tokens=1, method="tokenize", template_overhead_tokens=0,
        max_prompt_tokens=33_344, max_request_bytes=100_000, wire=wire,
    )
    receipt = {"body_sha256": measured.body_sha256, "request_bytes": measured.request_bytes,
               **measured.receipt_fields()}
    return measured, receipt


def _validate(measured: RequestMeasurement, receipt: dict[str, object], body: dict[str, object]) -> None:
    measured.validate(body, receipt, request_id="r", logical_request_id="r",
                      tokenizer_identity="strict-test")


def test_private_section_change_does_not_invalidate_wire_measurement() -> None:
    body = {"model": "strict-mock", "messages": [
        {"role": "user", "content": "[Model source window]", "_request_budget_section": "stage_evidence"}],
        "tools": [], "max_tokens": 100}
    measured, receipt = _measurement(body)
    changed = copy.deepcopy(body)
    changed["messages"][0]["_request_budget_section"] = "other"  # type: ignore[index]
    _validate(measured, receipt, _strict_body(changed))
    del changed["messages"][0]["_request_budget_section"]  # type: ignore[index]
    _validate(measured, receipt, _strict_body(changed))


@pytest.mark.parametrize("field,value", [("content", "changed"), ("role", "system"),
                                         ("name", "changed")])
def test_visible_message_change_still_breaks_seal(field: str, value: str) -> None:
    body = {"model": "strict-mock", "messages": [
        {"role": "user", "content": "source", "_request_budget_section": "stage_evidence"}],
        "tools": [], "max_tokens": 100}
    measured, receipt = _measurement(body)
    changed = _strict_body(body)
    changed["messages"][0][field] = value  # type: ignore[index]
    with pytest.raises(RequestSealError, match="measurement body/projection/provider identity changed"):
        _validate(measured, receipt, changed)


def _run_scripted(tmp_path: Path, *, strict: bool, steps: list[dict[str, object]],
                  inject_allocation_replan: bool = False,
                  source_revision_changes: int = 0,
                  repeat_same_revision: int = 0,
                  unreadable_source_revisions: bool = False,
                  prepare_source_revision_changes: int = 0,
                  repeat_same_revision_prepare: int = 0,
                  write_source_during_tests: bool = False,
                  revision_comparison_mutation: str | None = None,
                  kill_after_source_reprepared: bool = False,
                  resources_out: dict[str, object] | None = None,
                  allocator: str = "off") -> tuple[object, Path, Script]:
    source_root = ROOT / "src"
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    (repo / "tests").mkdir()
    test_source = "from app import VALUE\ndef test_value(): assert VALUE == 7\n"
    if write_source_during_tests:
        test_source = (
            "from pathlib import Path\n"
            "from app import VALUE\n\n"
            "def test_value():\n"
            "    app = Path(__file__).parents[1] / 'app.py'\n"
            "    source = app.read_text(encoding='utf-8')\n"
            "    count = sum(line.startswith('# run_tests source revision ') "
            "for line in source.splitlines()) + 1\n"
            "    app.write_text(source + f'# run_tests source revision {count}\\n', "
            "encoding='utf-8')\n"
            "    assert VALUE == 7\n"
        )
    (repo / "tests/test_value.py").write_text(test_source, encoding="utf-8")
    script = Script({"model": "strict-mock" if strict else "scripted",
                     "strict_openai": strict, "steps": steps})
    server = serve(script)
    run = tmp_path / "run"
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment.update({"PYTHONPATH": os.pathsep.join((str(source_root),
                                                         str(source_root / "stateframe/engine"))),
                        "SF_SEAL_TEST_SOURCE": str(source_root),
                        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                        "STATEFRAME_HOME": str(tmp_path / "state-home")})
    if allocator == "off":
        environment["STATEFRAME_WORKING_SET_ALLOCATOR"] = "off"
    # The isolated startup probe replaces /dev, so a wrapper under a
    # /dev/shm-backed pytest basetemp is invisible inside its mount namespace.
    wrapper_root = Path(os.environ.get("TMPDIR") or tmp_path)
    wrapper_root.mkdir(parents=True, exist_ok=True)
    wrapper_directory = tempfile.TemporaryDirectory(
        prefix="stateframe-test-python-", dir=str(wrapper_root),
    )
    wrapper = Path(wrapper_directory.name) / "python-with-private-source-tag"
    bootstrap = "\n".join((
        "import json, sys",
        "from pathlib import Path",
        "child, run = map(Path, sys.argv[1:3])",
        f"write_source_during_tests = {write_source_during_tests!r}",
        f"kill_after_source_reprepared = {kill_after_source_reprepared!r}",
        "sys.path.insert(0, str(child.parent.parent))",
        "import stateframe._child as host",
        "host.install_environment(json.loads((run / 'spec.json').read_text()))",
        "sys.path.insert(0, str(child.parent / 'engine'))",
        f"revision_comparison_mutation = {revision_comparison_mutation!r}",
        "if revision_comparison_mutation is not None:",
        "    import kernel_request_reprepare as reprepare",
        "    reprepare.same_source_revision_report = lambda previous, current: revision_comparison_mutation == 'always_true'",
        "import kernel_final_wire_allocation as allocation",
        "def mark(body, request_id):",
        "    for message in body.get('messages', []):",
        "        content = str(message.get('content', ''))",
        "        if content.startswith(('[Current source:', '[Model source window]', '[read page]')):",
        "            message['_request_budget_section'] = 'stage_evidence'",
        "            (run / 'tagged-source-window').write_text(request_id)",
        "            break",
        *((
            # ON validators are read-only after certification. Inject the same
            # private classification before the provider normalizer certifies it.
            "original = allocation._finalize_provider_request_body",
            "def tagged(capture, body, capabilities):",
            "    mark(body, capture.request_id)",
            "    return original(capture, body, capabilities)",
            "allocation._finalize_provider_request_body = tagged",
        ) if allocator == "on" else (
            "original = allocation._prepare_qualified_hook_body",
            "def tagged(context, body, request_id):",
            "    result = original(context, body, request_id)",
            "    mark(result, request_id)",
            "    return result",
            "allocation._prepare_qualified_hook_body = tagged",
        )),
        *((
            "import hashlib",
            "import kernel_frame_allocation_validation as validation",
            "from kernel_frame_optional_allocation import FrameAllocationReplanNeeded",
            "validate = validation.validate_allocation_authority",
            f"source_revision_changes = {source_revision_changes}",
            f"repeat_same_revision = {repeat_same_revision}",
            f"write_source_during_tests = {write_source_during_tests!r}",
            f"inject_allocation_replan = {inject_allocation_replan!r}",
            "def has_prior_provider_send(runtime):",
            "    active = getattr(runtime, '_active_capture', None)",
            "    active_request_id = getattr(active, 'request_id', None)",
            "    for journal_path in run.rglob('events.jsonl'):",
            "        for line in journal_path.read_text(encoding='utf-8').splitlines():",
            "            try:",
            "                row = json.loads(line)",
            "                if (row.get('kind') == 'request_submitted' and",
            "                        row.get('request_id') != active_request_id): return True",
            "            except (TypeError, ValueError):",
            "                continue",
            "    return False",
            "def replan_after_charge(allocation, runtime, messages, tools=None, **kwargs):",
            "    budget = getattr(runtime, 'continuation_limits', None)",
            "    if budget is not None and budget.continuation_model_decisions_used:",
            "        if inject_allocation_replan and not (run / 'injected-replan').exists():",
            "            (run / 'injected-replan').write_text(str(budget.continuation_model_decisions_used))",
            "            raise FrameAllocationReplanNeeded('allocation ledger authority changed')",
            "        observations_path = run / 'source-replan-observations.json'",
            "        observations = json.loads(observations_path.read_text()) if observations_path.exists() else []",
            "        source_plan = allocation.source_layout.get('source_plan', {})",
            "        observation = {",
            "            'revisions': source_plan.get('revisions', {}),",
            "            'allocation_digest': allocation.digest,",
            "            'frame_digest': hashlib.sha256(json.dumps(messages, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),",
            "        }",
            "        if not observations or observations[-1]['revisions'] != observation['revisions']:",
            "            observations.append(observation)",
            "        observations_path.write_text(json.dumps(observations, sort_keys=True))",
            "        edit_count_path = run / 'source-revision-edit-count'",
            "        edit_count = int(edit_count_path.read_text()) if edit_count_path.exists() else 0",
            "        if edit_count < source_revision_changes and has_prior_provider_send(runtime):",
            "            edit_count += 1",
            "            if write_source_during_tests:",
            "                from expectation_ledger import run_typed_suite",
            "                result = run_typed_suite(Path(runtime.owner.root), scope='target', selectors=('tests/test_value.py',))",
            "                if getattr(result.status, 'value', result.status) != 'PASS':",
            "                    raise AssertionError('pre-send run_tests source writer did not pass')",
            "            else:",
            "                (Path(runtime.owner.root) / 'app.py').write_text(f'VALUE = 7\\n# source revision {edit_count}\\n', encoding='utf-8')",
            "            edit_count_path.write_text(str(edit_count))",
            "            try:",
            "                return validate(allocation, runtime, messages, tools, **kwargs)",
            "            except FrameAllocationReplanNeeded:",
            "                raise",
            "            raise AssertionError('planned app.py revision change was not detected')",
            "        repeat_count_path = run / 'same-revision-report-count'",
            "        repeat_count = int(repeat_count_path.read_text()) if repeat_count_path.exists() else 0",
            "        if repeat_count < repeat_same_revision:",
            "            if repeat_count:",
            "                observations.append(observation)",
            "                observations_path.write_text(json.dumps(observations, sort_keys=True))",
            "            repeat_count += 1",
            "            repeat_count_path.write_text(str(repeat_count))",
            "            raise FrameAllocationReplanNeeded('allocation ledger authority changed')",
            "        observations_path.write_text(json.dumps(observations, sort_keys=True))",
            "    return validate(allocation, runtime, messages, tools, **kwargs)",
            "validation.validate_allocation_authority = replan_after_charge",
        ) if (inject_allocation_replan or source_revision_changes or repeat_same_revision) else ()),
        *((
            "import kernel_request_reprepare as unreadable_reprepare",
            "unreadable_reprepare.prepared_source_revision_set = lambda *_args, **_kwargs: None",
        ) if unreadable_source_revisions else ()),
        *((
            "import kernel_frame_source_snapshot as source_snapshot",
            "from kernel_frame_optional_allocation import FrameAllocationReplanNeeded",
            f"prepare_source_revision_changes = {prepare_source_revision_changes}",
            f"repeat_same_revision_prepare = {repeat_same_revision_prepare}",
            "original_frame_source_snapshot = source_snapshot.frame_source_snapshot",
            "def replan_during_prepare(runtime, plan):",
            "    revisions = plan.get('revisions', {}) if isinstance(plan, dict) else {}",
            "    if 'app.py' in revisions:",
            "        observations_path = run / 'prepare-replan-observations.json'",
            "        observations = json.loads(observations_path.read_text()) if observations_path.exists() else []",
            "        if not observations or observations[-1] != revisions:",
            "            observations.append(revisions)",
            "        observations_path.write_text(json.dumps(observations, sort_keys=True))",
            "        change_count_path = run / 'prepare-source-revision-edit-count'",
            "        change_count = int(change_count_path.read_text()) if change_count_path.exists() else 0",
            "        if change_count < prepare_source_revision_changes:",
            "            change_count += 1",
            "            if write_source_during_tests:",
            "                from expectation_ledger import run_typed_suite",
            "                result = run_typed_suite(Path(runtime.owner.root), scope='target', selectors=('tests/test_value.py',))",
            "                if getattr(result.status, 'value', result.status) != 'PASS':",
            "                    raise AssertionError('prepare-phase run_tests source writer did not pass')",
            "            else:",
            "                (Path(runtime.owner.root) / 'app.py').write_text(f'VALUE = 7\\n# prepared source revision {change_count}\\n', encoding='utf-8')",
            "            change_count_path.write_text(str(change_count))",
            "            try:",
            "                return original_frame_source_snapshot(runtime, plan)",
            "            except FrameAllocationReplanNeeded:",
            "                raise",
            "            raise AssertionError('planned app.py preparation revision change was not detected')",
            "        repeat_count_path = run / 'prepare-same-revision-count'",
            "        repeat_count = int(repeat_count_path.read_text()) if repeat_count_path.exists() else 0",
            "        if repeat_count < repeat_same_revision_prepare:",
            "            if repeat_count:",
            "                observations.append(revisions)",
            "                observations_path.write_text(json.dumps(observations, sort_keys=True))",
            "            repeat_count += 1",
            "            repeat_count_path.write_text(str(repeat_count))",
            "            raise FrameAllocationReplanNeeded('source proof snapshot revision changed')",
            "        observations_path.write_text(json.dumps(observations, sort_keys=True))",
            "    return original_frame_source_snapshot(runtime, plan)",
            "source_snapshot.frame_source_snapshot = replan_during_prepare",
        ) if (prepare_source_revision_changes or repeat_same_revision_prepare) else ()),
        *((
            "import kernel_measurement as measurement",
            "original_recover_source_revision = measurement.RequestCapture._recover_source_revision_replan",
            "def interrupt_after_source_reprepared(self, error, state):",
            "    original_emit = self.journal.emit",
            "    def interrupting_emit(kind, **fields):",
            "        result = original_emit(kind, **fields)",
            "        if kill_after_source_reprepared and kind == 'source_revision_reprepared':",
            "            (run / 'killed-after-source-reprepared').write_text(self.request_id)",
            "        return result",
            "    self.journal.emit = interrupting_emit",
            "    try:",
            "        return original_recover_source_revision(self, error, state)",
            "    finally:",
            "        self.journal.emit = original_emit",
            "measurement.RequestCapture._recover_source_revision_replan = interrupt_after_source_reprepared",
        ) if kill_after_source_reprepared else ()),
        "raise SystemExit(host.main([str(child), str(run)]))",
    ))
    try:
        with sandbox_test_python(_clone_local_test_python(tmp_path)) as test_python:
            wrapper.write_text(
                f"#!{test_python}\n"
                "import os,sys\n"
                f"real_python={test_python!r}\n"
                "args=sys.argv[1:]\n"
                "if len(args) >= 3 and args[0] == '-B' and os.path.basename(args[1]) == '_child.py':\n"
                f"    os.execv(real_python, [real_python, '-B', '-c', {bootstrap!r}, *args[1:]])\n"
                # Linux launches the child as `-B -c <bootstrap> stateframe._child` with the
                # child and run paths inside the bootstrap's sys.argv literal.
                "if len(args) == 4 and args[:2] == ['-B', '-c'] and args[3] == 'stateframe._child':\n"
                "    import ast,re\n"
                "    child,run=ast.literal_eval(re.search(r'sys\\.argv=(\\[.*?\\]);',args[2]).group(1))\n"
                "    os.chdir(run)\n"
                f"    os.execv(real_python, [real_python, '-B', '-c', {bootstrap!r}, child, run])\n"
                "os.execv(real_python, [real_python, *args])\n", encoding="utf-8")
            wrapper.chmod(0o755)
            command = [
                sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
                "--task", "Read app.py and tests/test_value.py; run the test and report VALUE.",
                "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                "--model", script.model, "--run-dir", str(run), "--max-turns",
                "32" if (inject_allocation_replan or source_revision_changes
                          or repeat_same_revision or prepare_source_revision_changes
                          or repeat_same_revision_prepare) else "12",
                "--test-python", test_python, "--python", str(wrapper), "--quiet",
            ]
            if kill_after_source_reprepared:
                launched_after = time.time()
                process = subprocess.Popen(
                    command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, start_new_session=True,
                )
                marker = run / "killed-after-source-reprepared"
                deadline = time.monotonic() + 240
                while not marker.exists() and process.poll() is None:
                    if time.monotonic() >= deadline:
                        signal_run_owned(
                            run, process, signal.SIGKILL, deadline=time.monotonic() + 30,
                            what="source replan interruption timeout", group=True,
                        )
                        break
                    time.sleep(0.02)
                if marker.is_file() and process.poll() is None:
                    engine_pid = int((run / ".engine.lock").read_text(encoding="utf-8"))
                    engine_identity = record_identity(
                        engine_pid, launched_after=launched_after,
                        command_fragment="stateframe._child",
                    )
                    if "stateframe._child" not in engine_identity.command and "_child.py" not in engine_identity.command:
                        raise AssertionError("run engine lock did not identify its engine child")
                    signal_run_owned(
                        run, process, signal.SIGKILL, deadline=time.monotonic() + 180,
                        what="source replan interruption", group=True,
                    )
                    try:
                        signal_run_verified(
                            run, engine_pid, engine_identity, signal.SIGKILL,
                            deadline=time.monotonic() + 30, what="source replan engine child",
                        )
                    except ProcessLookupError:
                        pass
                stdout, stderr = process.communicate(timeout=10)
                process = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            else:
                process = run_monitored(
                    command, cwd=ROOT, env=environment, capture_output=True, text=True,
                    expected_seconds=240, run_dir=run, check=False,
                )
            if resources_out is not None:
                resources_out.update({
                    "server": server,
                    "wrapper_directory": wrapper_directory,
                    "wrapper": wrapper,
                    "test_python": test_python,
                    "environment": environment,
                })
            return process, run, script
    finally:
        if resources_out is None or not resources_out:
            try:
                server.shutdown()
            finally:
                try:
                    server.server_close()
                finally:
                    wrapper_directory.cleanup()


def assert_sealed_wire(run: Path, *, minimum: int) -> None:
    requests = sorted((run / "wire").glob("*.request.json"))
    assert len(requests) >= minimum
    rows = [row for row in measurement_events(run)
            if row.get("kind") == "request_budget_final_wire" and row.get("status") == "READY"]
    digests = [row["body_sha256"] for row in rows if "body_sha256" in row]
    for request in requests:
        wire = request.read_bytes()
        assert "sha256:" + hashlib.sha256(wire).hexdigest() in digests
        assert all(not key.startswith("_") for message in json.loads(wire)["messages"]
                   for key in message)


@pytest.mark.host_sandbox
@pytest.mark.parametrize("allocator", ["off", "on"])
def test_hosted_stage_evidence_sends_seal_exact_wire(tmp_path: Path, allocator: str) -> None:
    steps: list[dict[str, object]] = [
        {"tool": "read", "arguments": {"path": path}}
        for path in ("app.py", "tests/test_value.py", "app.py", "tests/test_value.py")
    ] + [{"tool": "run_tests", "arguments": {"target": "tests/test_value.py"}},
         {"tool": "done", "arguments": {"summary": "VALUE is seven and tests pass."}}]
    process, run, script = _run_scripted(tmp_path, strict=True, steps=steps, allocator=allocator)
    output = process.stdout + process.stderr
    traceback = run / "child-traceback.txt"
    if traceback.is_file():
        output += traceback.read_text(encoding="utf-8")
    assert len(script.requests) >= 4, output
    assert (run / "tagged-source-window").is_file(), output
    assert "measurement body/projection/provider identity changed" not in output
    assert process.returncode == 0, output
    spec = json.loads((run / "spec.json").read_text())
    assert spec["environment"]["STATEFRAME_WORKING_SET_ALLOCATOR"] == allocator
    script.assert_consumed()
    assert_sealed_wire(run, minimum=4)
