"""Provider-boundary checks for F094 comparison captions."""
from __future__ import annotations

import gzip
import io
import json
import os
import signal
import subprocess
import sys
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread

import pytest
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from fixtures.process_signals import (
    record_identity,
    signal_run_owned,
    signal_run_verified,
)
from tools.pinned_tree import pinned_tree
from tools.sandbox_temp import sandbox_visible_temp_dir

from stateframe.replay_tape import first_difference
from stateframe.scripted_endpoint import Script, make_handler
from acceptance._support import run_case

ROOT = Path(__file__).resolve().parents[1]
BASE = "29765b39820fd39800fd7292f52fb5bcb81e6069"
SWITCH = "STATEFRAME_R200_F094_WRITE_FIGURES_SENTENCE"
FIGURE_SENTENCE_GUIDANCE = (
    "For figure comparisons, first write one source-backed sentence naming both operands and the "
    "baseline in your public response or figure specification. Then use the mandated form, or "
    "choose a chart or table if none is mandated. Omit unsupported claims."
)

_CLI = """
import os
import secrets
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from stateframe import cli, runner

assert runner.__file__ == sys.argv[1]
runner.secrets = SimpleNamespace(
    token_hex=lambda size: 'a' * 32 if size == 16 else secrets.token_hex(size),
)

def interpreter_probe(python, code, *, label, on_clean_success=None, probe_args=None):
    # The parity fixture runs a trusted interpreter and leaves the nested startup sandbox
    # out of the comparison. Both trees still execute their own interpreter probe code.
    assert os.path.samefile(python, sys.executable), label
    args = probe_args if probe_args is not None else ("-I", "-B", "-c", code)
    with tempfile.TemporaryDirectory() as cwd:
        output = subprocess.check_output([python, *args], cwd=cwd,
                                         timeout=2 * runner._startup_probe_seconds())
    if on_clean_success is not None:
        on_clean_success()
    return output

runner._startup_probe = interpreter_probe
raise SystemExit(cli.main(sys.argv[2:]))
"""


def _run(checkout: Path, repo: Path, run: Path, switch: str | None,
         document: dict[str, object], *, skills: tuple[str, ...] = ()) -> tuple[tuple[bytes, ...], tuple[bytes, ...]]:
    script = Script(document)
    requests: list[bytes] = []
    pending, interrupted = Event(), Event()
    handler = make_handler(script)

    class RecordingHandler(handler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return super().do_POST()
            stream = self.rfile
            raw = stream.read(int(self.headers["Content-Length"]))
            wire = gzip.decompress(raw) if self.headers.get("Content-Encoding") == "gzip" else raw
            requests.append(wire)
            if len(requests) == 3:
                # Withhold the reply until the engine has stopped with this send pending.
                pending.set()
                interrupted.wait(scaled(90))
                return
            try:
                self.rfile = io.BytesIO(raw)
                return super().do_POST()
            finally:
                self.rfile = stream

    environment = {key: value for key, value in os.environ.items() if not key.startswith("STATEFRAME_")}
    environment.update(PYTHONPATH=str(checkout / "src"), STATEFRAME_HOME=isolated_stateframe_home(),
                       STATEFRAME_PROVIDER="llama.cpp")
    if switch is not None:
        environment[SWITCH] = switch

    with ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"

        def command(*args: str) -> list[str]:
            return [sys.executable, "-B", "-c", _CLI, str(checkout / "src/stateframe/runner.py"),
                    *args, "--endpoint", endpoint, "--quiet"]

        first = None
        try:
            deadline = time.monotonic() + scaled(240)
            launched = time.time()
            first = subprocess.Popen(
                command("run", "--repo", str(repo), "--run-dir", str(run), "--model", "scripted",
                        "--test-python", sys.executable, "--shell", "off",
                        "--task", "Write the comparison from metrics.csv to figure.txt.",
                        *(argument for skill in skills for argument in ("--skill", skill))),
                env=environment, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            while not pending.wait(0.2):
                if first.poll() is not None:
                    stdout, stderr = first.communicate()
                    pytest.fail(stdout + stderr)
                assert time.monotonic() < deadline, "third provider request did not arrive"
            original = tuple(requests)
            children = subprocess.check_output(["pgrep", "-P", str(first.pid)], text=True).split()
            assert len(children) == 1, children
            child = int(children[0])
            identity = record_identity(child, launched_after=launched, command_fragment="_child.py")
            signal_run_verified(run, child, identity, signal.SIGKILL,
                                deadline=time.monotonic() + scaled(60), what="F094 parity interruption")
            interrupted.set()
            first.communicate(timeout=scaled(30))
            resumed = subprocess.run(command("resume", str(run)), env=environment, cwd=repo,
                                     capture_output=True, text=True, timeout=scaled(240), check=False)
            assert resumed.returncode == 0, resumed.stdout + resumed.stderr
            script.assert_consumed()
            rebuilt = tuple(requests[len(original):])
            assert rebuilt, "resume never reached the provider"
            assert rebuilt[0] == original[-1], first_difference(original[-1], rebuilt[0])
            return original, rebuilt
        finally:
            try:
                if first is not None and first.poll() is None:
                    signal_run_owned(run, first, signal.SIGKILL, deadline=time.monotonic(),
                                     what="F094 parity cleanup", require_checkpoint_saved=False)
                    first.communicate(timeout=scaled(10))
            finally:
                interrupted.set()
                server.shutdown()
                thread.join()


@pytest.mark.host_sandbox
def test_f094_write_figures_sentence_off_request_bytes_match_pinned_099(tmp_path_factory):
    pinned = pinned_tree(BASE, tmp_path_factory)
    document = {"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "required": True, "arguments": {"path": "metrics.csv"}},
        {"tool": "skill_load", "required": True, "arguments": {"name": "comparison-caption"}},
        {"tool": "file_create", "required": True,
         "arguments": {"path": "figure.txt", "content": "Revenue rose from 40 to 52.\n"}},
        {"tool": "done", "required": True, "arguments": {"summary": "Wrote the comparison."}},
    ]}
    with sandbox_visible_temp_dir() as directory:
        work = Path(directory)
        repo = work / "repo"
        run = work / "run"
        repo.mkdir()
        (repo / "metrics.csv").write_text("period,value\nbaseline,40\ncurrent,52\n", encoding="utf-8")
        skill = repo / "skills" / "comparison-caption"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: comparison-caption\ndescription: Write a comparison sentence before selecting a form.\n---\n"
            "Use source evidence to name operands and the baseline before choosing a chart or table.\n",
            encoding="utf-8",
        )
        selected = ("./skills/comparison-caption",)
        baseline = _run(pinned, repo, run, None, document, skills=selected)
        assert b"comparison-caption" in baseline[0][0]
        assert b"USER-PROVIDED SKILL INSTRUCTIONS: comparison-caption" in baseline[0][-1]
        run.rename(work / "baseline")
        for name, setting in (("unset", None), ("off", "off"), ("false", "false")):
            current = _run(ROOT, repo, run, setting, document, skills=selected)
            for expected_attempt, actual_attempt in zip(baseline, current, strict=True):
                for expected, actual in zip(expected_attempt, actual_attempt, strict=True):
                    assert actual == expected, (name, first_difference(expected, actual))
            run.rename(work / name)


def test_f094_comparison_caption_skill_resolves_from_requested_repo_path(tmp_path, monkeypatch):
    from stateframe import skills

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)

    with skills.bundled_comparison_caption_selection(True):
        selected = skills.resolve_skills(("./skills/comparison-caption",), trusted=False)

    assert len(selected) == 1
    assert selected[0].name == "comparison-caption"
    assert selected[0].description == "Write a source-backed comparison caption before committing to a figure form."
    assert "baseline" in selected[0].body.lower()
    with pytest.raises(skills.SkillError, match="skill directory missing or linked"):
        skills.resolve_skills(("./skills/comparison-caption",), trusted=False)


def _caption_profile(tmp_path):
    from stateframe import profiles

    profile = tmp_path / "caption-profile.json"
    profile.write_text(json.dumps({"schema": "stateframe-product-profile/1",
                                   "switches": {**profiles.load(profiles.DEFAULT), SWITCH: "on"}}),
                       encoding="utf-8")
    return profile


@pytest.mark.parametrize("setting", [None, "false"])
def test_f094_disabled_missing_caption_path_uses_original_resolver(tmp_path, monkeypatch, setting):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    script = tmp_path / "caption.script.json"
    script.write_text(json.dumps({"model": "scripted", "steps": [
        {"tool": "done", "arguments": {"summary": "No change needed."}},
    ]}), encoding="utf-8")

    profile_flags = ("--profile", str(_caption_profile(tmp_path))) if setting == "false" else ()
    case = run_case(tmp_path, str(script), repo=repo, task="Create report.txt.",
                    flags=("--skill", "./skills/comparison-caption", "--shell", "off", *profile_flags),
                    default_test_cmd=False, env_overrides={} if setting is None else {SWITCH: setting})

    assert "skill directory missing or linked: skills/comparison-caption" in case.process.stderr
    assert case.process.returncode != 0
    assert not case.requests


def test_f094_catalog_guidance_is_owned_by_order_allowed_skill_module():
    from stateframe.engine import kernel_skills as runtime

    kernel_skills = ROOT / "src/stateframe/engine/kernel_skills.py"
    source = kernel_skills.read_text(encoding="utf-8")
    skill_source = (ROOT / "src/stateframe/skills.py").read_text(encoding="utf-8")

    assert SWITCH in source
    assert runtime._FIGURE_SENTENCE_GUIDANCE == FIGURE_SENTENCE_GUIDANCE
    assert "_BUNDLED_COMPARISON_CAPTION_SOURCE" in skill_source


@pytest.fixture
def caption_runtime(tmp_path, monkeypatch):
    from stateframe import skills
    from stateframe.engine.kernel_skills import SkillRuntime

    monkeypatch.chdir(tmp_path)
    with skills.bundled_comparison_caption_selection(True):
        selected = skills.resolve_skills(("./skills/comparison-caption",), trusted=False)
    records = [skill.record(trusted=False) for skill in selected]
    return SkillRuntime(records, skills.skills_digest(records), None, write_figures_sentence=True)


def test_f094_catalog_requires_visible_caption_before_renderer(caption_runtime):
    assert not caption_runtime.messages()
    (line,) = caption_runtime.catalog()["catalog"]
    caption = "in your public response or figure specification."
    renderer = "Then use the mandated form"
    assert caption in line
    assert renderer in line
    assert line.index(caption) < line.index(renderer)


@pytest.mark.parametrize("delivery", ["catalog", "body"])
def test_f094_guidance_honors_mandated_renderer(caption_runtime, delivery):
    if delivery == "catalog":
        (text,) = caption_runtime.catalog()["catalog"]
        assert "use the mandated form, or choose a chart or table if none is mandated" in text
    else:
        assert caption_runtime.load("comparison-caption") == "skill comparison-caption loaded"
        (message,) = caption_runtime.messages()
        assert message["role"] == "user"
        assert "Honor any form mandated by the brief." in message["content"]
        assert "If no form is mandated, choose the chart or table" in message["content"]


@pytest.fixture
def boundary_skill_paths(tmp_path):
    names = ("comparison-caption", *(f"skill-{index:02d}" for index in range(10)))
    for index, name in enumerate(names):
        directory = tmp_path / name
        directory.mkdir()
        description = "Valid comparison skill " + "x" * (220 if index < 5 else 219)
        (directory / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\nFollow the task and cite evidence.\n",
            encoding="utf-8",
        )
    return tuple(str(tmp_path / name) for name in names)


def test_f094_optional_catalog_guidance_does_not_reject_valid_skills(boundary_skill_paths):
    from stateframe import runner
    from stateframe.engine.kernel_request_diet import RequestTokenMeter
    from stateframe.engine.kernel_skills import SkillRuntime, measure_admission
    from stateframe.skills import measured_skills_digest, resolve_skills

    selected = resolve_skills(boundary_skill_paths, trusted=False)
    records = [skill.record(trusted=False) for skill in selected]
    meter = RequestTokenMeter(endpoint=None, template_overhead_tokens=0, disabled=True)

    baseline = measure_admission(selected, records, meter, write_figures_sentence=False)
    assert baseline["catalog"] == {"tokens": 1166, "method": "calibrated_bytes"}
    augmented = measure_admission(selected, records, meter, write_figures_sentence=True)
    assert augmented["catalog"] == {"tokens": 1268, "method": "calibrated_bytes"}

    admitted, effective = runner._admit_skill_settings(
        selected, records, None, meter, write_figures_sentence=True,
    )

    assert effective is False
    assert admitted == baseline
    runtime = SkillRuntime(records, measured_skills_digest(records, admitted), None, admitted,
                           write_figures_sentence=effective)
    assert sum(FIGURE_SENTENCE_GUIDANCE in line for line in runtime.catalog_lines) == 0


def test_f094_profile_catalog_fallback_reaches_provider(tmp_path, boundary_skill_paths):
    from stateframe.skills import catalog_lines, resolve_skills

    profile = _caption_profile(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    script = tmp_path / "caption.script.json"
    script.write_text(json.dumps({"model": "scripted", "steps": [
        {"tool": "file_create", "arguments": {"path": "report.txt", "content": "Comparison complete.\n"}},
        {"tool": "done", "arguments": {"summary": "Wrote the report."}},
    ]}), encoding="utf-8")

    case = run_case(tmp_path, str(script), repo=repo, task="Create report.txt.", default_test_cmd=False,
                    flags=("--profile", str(profile), "--shell", "off",
                           *(arg for path in boundary_skill_paths for arg in ("--skill", path))))

    assert case.process.returncode == 0, case.process.stdout + case.process.stderr
    assert case.requests, "profile launch never reached the provider"
    spec = json.loads((case.run_dir / "spec.json").read_text(encoding="utf-8"))
    assert spec["skills_measurements"]["catalog"] == {"tokens": 1166, "method": "calibrated_bytes"}
    assert spec["environment"][SWITCH] == "off"
    assert spec["environment_sources"][SWITCH] == "runner:skill-catalog-admission"
    selected = resolve_skills(boundary_skill_paths, trusted=False)
    assert _request_state(case.requests[0])["framefit5"]["skills"]["catalog"] == list(
        catalog_lines(selected, spec["skills"])
    )


def _request_state(request):
    frame = next(message["content"] for message in request["messages"]
                 if str(message.get("content", "")).startswith("[StateFrame]\n"))
    return json.loads(frame.split("\n", 1)[1])["state"]


def test_f094_write_figures_sentence_switch_defaults_off(monkeypatch):
    from stateframe import runner
    from stateframe.engine import kernel_release_switch_registry as registry

    monkeypatch.delenv(SWITCH, raising=False)
    assert runner.OPT_IN_SWITCHES[SWITCH] == ("off", "on")
    assert SWITCH not in runner.resolve_opt_in_switches()

    monkeypatch.setenv(SWITCH, "false")
    assert SWITCH not in runner.resolve_opt_in_switches()
    assert SWITCH in registry.release_stateframe_switches()[0]


@pytest.mark.parametrize("enabled_by", ["ambient", "profile"])
def test_f094_write_figures_sentence_on_path_is_bounded_and_idempotent(tmp_path, monkeypatch, enabled_by):
    from stateframe import runner
    from stateframe import skills as skill_library

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "metrics.csv").write_text("period,revenue_usd\nbaseline,40\ncurrent,52\n", encoding="utf-8")
    script_path = tmp_path / "comparison-caption.script.json"
    task = "Create a figure specification comparing baseline and current revenue in metrics.csv."
    renderer = "bar"
    if enabled_by == "profile":
        task += " Use a table as the renderer."
        renderer = "table"
    artifact = json.dumps({
        "caption": "Revenue rose from $40 to $52, a 30% increase (metrics.csv).",
        "renderer": renderer,
    }, indent=2) + "\n"
    script_path.write_text(json.dumps({
        "model": "scripted",
        "tokenize": {"mode": "ok"},
        "steps": [
            {"tool": "read", "arguments": {"path": "metrics.csv"}},
            {"tool": "skill_load", "arguments": {"name": "comparison-caption"}},
            {"tool": "skill_load", "arguments": {"name": "comparison-caption"}},
            {"tool": "file_create", "arguments": {"path": "figure.json", "content": artifact}},
            {"tool": "done", "arguments": {"summary": "Created a figure with a sourced comparison caption."}},
        ],
    }), encoding="utf-8")

    monkeypatch.chdir(repo)
    flags = ("--skill", "./skills/comparison-caption", "--shell", "off")
    env_overrides = {SWITCH: "on"} if enabled_by == "ambient" else {}
    if enabled_by == "ambient":
        monkeypatch.setenv(SWITCH, "on")
        assert runner.resolve_opt_in_switches()[SWITCH] == "on"
    else:
        # A project powerup state file triggers the trusted-skill lookup in powerup preflight.
        config = repo / ".stateframe"
        config.mkdir()
        (config / "powerups.json").write_text(json.dumps({"schema": "stateframe-powerup-state/1",
                                                         "global": {}, "profiles": {}}), encoding="utf-8")
        flags += ("--profile", str(_caption_profile(tmp_path)), "--trust-skill-tools")
    case = run_case(tmp_path, str(script_path), repo=repo, task=task, flags=flags,
                    default_test_cmd=False, env_overrides=env_overrides)
    assert case.process.returncode == 0, case.process.stdout + case.process.stderr
    assert case.summary is not None and case.summary["succeeded"] is True
    assert case.summary["candidate_files_written"] == ["figure.json"]

    first = case.requests[0]
    edit_request = next(
        request for request in case.requests
        if any(message.get("role") == "tool"
               and str(message.get("content", "")) == "skill comparison-caption already loaded"
               for message in request["messages"])
    )
    first_catalog = _request_state(first)["framefit5"]["skills"]["catalog"]
    edit_catalog = _request_state(edit_request)["framefit5"]["skills"]["catalog"]
    for catalog in (first_catalog, edit_catalog):
        matching = [line for line in catalog if str(line).startswith(
            "comparison-caption: Write a source-backed comparison caption before committing to a figure form.")]
        assert len(matching) == 1
        assert sum(FIGURE_SENTENCE_GUIDANCE in str(line) for line in catalog) == 1

    edit_messages = [str(message.get("content", "")) for message in edit_request["messages"]]
    (body,) = [text for text in edit_messages
               if text.startswith("USER-PROVIDED SKILL INSTRUCTIONS: comparison-caption ")]
    assert "in your public response or figure specification before committing to a renderer" in body
    assert "Honor any form mandated by the brief." in body
    assert any("period,revenue_usd" in text and "baseline,40" in text and "current,52" in text
               for text in edit_messages)
    created = (case.run_dir / "workspace" / "figure.json").read_text(encoding="utf-8")
    assert created == artifact
    assert created.index('"caption"') < created.index(f'"renderer": "{renderer}"')
    spec = json.loads((case.run_dir / "spec.json").read_text(encoding="utf-8"))
    records = spec["skills"]
    assert isinstance(records, list)
    assert len(records) == 1 and records[0]["name"] == "comparison-caption"
    with skill_library.bundled_comparison_caption_selection(True):
        bundled_skill = skill_library.resolve_skills(("./skills/comparison-caption",), trusted=False)[0]
    assert records[0]["skill_md_sha256"] == bundled_skill.skill_md_sha256
    assert len(bundled_skill.body.encode("utf-8")) <= 32 * 1024
    assert len(FIGURE_SENTENCE_GUIDANCE.encode("utf-8")) <= 256
