from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

from stateframe import resume, runner, shell_host
from stateframe.endpoint import Endpoint


def _die_after_archive_spec_copy(run_value: str, base_url: str, model: str) -> None:
    run = Path(run_value)
    endpoint = Endpoint.from_args(base_url, model, None)
    runner._check_engine_python = lambda *_args, **_kwargs: None
    runner.Endpoint.check_locality = lambda *_args, **_kwargs: None
    runner._provider_model = lambda *_args, **_kwargs: {"model_found": True}
    runner.check_slot_context = lambda *_args, **_kwargs: None
    runner._probe_routes = lambda *_args, **_kwargs: None
    runner._execute = lambda *_args, **_kwargs: {"result": "INTERRUPTED"}
    shell_host.provision = lambda *_args, **_kwargs: None

    real_copy = resume.shutil.copy2

    def stop_after_archive_spec_copy(source, target, *args, **kwargs):
        result = real_copy(source, target, *args, **kwargs)
        if Path(target) == run / "attempts" / "1" / "spec.json":
            os._exit(86)
        return result

    resume.shutil.copy2 = stop_after_archive_spec_copy
    runner.resume_run(run, endpoint=endpoint, printer=lambda *_: None)
    os._exit(0)


def test_public_resume_finishes_archive_after_process_dies_before_first_move(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_check_python", lambda python, **_kwargs: python)
    monkeypatch.setattr(runner, "_check_engine_python", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.Endpoint, "check_locality", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_provider_model", lambda *_args, **_kwargs: {"model_found": True})
    monkeypatch.setattr(runner, "check_slot_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_probe_routes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(shell_host, "provision", lambda *_args, **_kwargs: None)

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    run = tmp_path / "run"
    endpoint = Endpoint.from_args("http://127.0.0.1:9/v1", "scripted", None)
    runner.prepare(runner.RunOptions(repo=repo, task="Inspect app.py", endpoint=endpoint,
                                    test_cmd="true", check_source="none", run_dir=run),
                   printer=lambda *_: None)

    wire = run / "wire"
    wire.mkdir(exist_ok=True)
    request = b'{"messages": [], "model": "scripted"}'
    response = b'{"choices": [{"message": {"content": "done"}}]}'
    (wire / "0001.request.json").write_bytes(request)
    (wire / "0001.response.json").write_bytes(response)
    (run / "sends.jsonl").write_bytes(b"")

    executions = []

    def finish_resumed_attempt(*_args):
        executions.append("executed")
        return {"result": "INTERRUPTED"}

    monkeypatch.setattr(runner, "_execute", finish_resumed_attempt)

    child = multiprocessing.get_context("spawn").Process(
        target=_die_after_archive_spec_copy, args=(str(run), endpoint.base_url, endpoint.model))
    child.start()
    child.join()
    assert child.exitcode == 86

    archive = run / "attempts" / "1"
    assert (archive / "spec.json").is_file()
    assert not (archive / "worker").exists()
    assert not (archive / "wire").exists()
    assert (wire / "0001.request.json").read_bytes() == request
    assert (wire / "0001.response.json").read_bytes() == response

    result = runner.resume_run(run, endpoint=endpoint, printer=lambda *_: None)

    assert result == {"result": "INTERRUPTED"}
    assert executions == ["executed"]
    assert (archive / "wire" / "0001.request.json").read_bytes() == request
    assert (archive / "wire" / "0001.response.json").read_bytes() == response
    assert (archive / "sends.jsonl").read_bytes() == b""
    assert not (run / "attempts" / "2").exists()
    assert json.loads((run / "spec.json").read_text(encoding="utf-8"))["resume"]["attempt"] == 2


def test_archive_with_a_linked_spec_is_not_adopted(tmp_path):
    run = tmp_path / "run"
    folder = run / "attempts" / "1"
    folder.mkdir(parents=True)
    spec = {"run_identity_seed": "seed"}
    (run / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (folder / "spec.json").symlink_to(run / "spec.json")

    assert resume._interrupted_archive(run, spec) is None

    (folder / "spec.json").unlink()
    (folder / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    assert resume._interrupted_archive(run, spec) == folder


def test_linked_spec_folder_beside_a_real_one_is_skipped(tmp_path):
    run = tmp_path / "run"
    real, linked = run / "attempts" / "1", run / "attempts" / "2"
    real.mkdir(parents=True)
    linked.mkdir()
    spec = {"run_identity_seed": "seed"}
    (real / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (linked / "spec.json").symlink_to(real / "spec.json")

    assert resume._interrupted_archive(run, spec) == real
