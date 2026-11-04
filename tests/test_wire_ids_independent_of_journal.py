"""Prove internal journal noise cannot move provider-visible request identities."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from tests._journal import measurement_events


# Reuse the acceptance tests' child-launch wrapper seam. The wrapper executes
# _child.py in the actual engine process, then installs its journal hook only
# after install_environment() and install_run_identity() have run.
_ENGINE_CHILD_WRAPPER = r'''
import importlib
import json
import os
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

child, run, mode, marker = sys.argv[1:]
child_path = Path(child).resolve()
run_path = Path(run).resolve()
sys.path.insert(0, str(child_path.parent))
spec = json.loads((run_path / "spec.json").read_text())
engine_path = Path(spec["engine_dir"]).resolve()
sys.path.insert(0, str(engine_path))
marker_path = Path(marker) if marker else None
wrapper_status_path = (marker_path.with_suffix(".wrapper.json")
                       if marker_path is not None else None)
stage_path = (marker_path.with_suffix(".stages.jsonl")
              if marker_path is not None else None)


def write_status(path, stage, **fields):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stage": stage, **fields}, sort_keys=True))


def write_stage(stage, **fields):
    if stage_path is None:
        return
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    with stage_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, **fields}, sort_keys=True) + "\n")


write_status(wrapper_status_path, "wrapper_started", pid=os.getpid(),
             cwd=os.getcwd(), engine_path=str(engine_path))
write_stage("wrapper_started", pid=os.getpid(), cwd=os.getcwd(),
            engine_path=str(engine_path))
sys.argv[:] = [str(child_path), str(run_path)]
namespace = runpy.run_path(str(child_path), run_name="stateframe_engine_child")
sys.argv[:] = [str(child_path), str(run_path)]
main = namespace["main"]
main_globals = main.__globals__
write_stage(
    "child_module_loaded", pid=os.getpid(), child_file=str(child_path),
    engine_module=main_globals.get("__name__"),
    returned_namespace_is_main_globals=(namespace is main_globals),
)

original_install_run_identity = main_globals["install_run_identity"]
patchers = []


def _patch(target, name, replacement):
    patcher = patch.object(target, name, replacement)
    patcher.start()
    patchers.append(patcher)


def _install_child_hook(seed):
    write_stage("install_run_identity_entered", mode=mode)
    original_install_run_identity(seed)
    if mode == "off":
        write_status(wrapper_status_path, "hook_installed", mode=mode,
                     modules=[], writer_types=[])
        write_stage("hook_installed", mode=mode, modules=[], writer_types=[])
        return

    installed_modules = []
    installed_writer_types = []
    if mode in {"sequence", "sequence_noise"}:
        wire_identity = importlib.import_module("kernel_wire_identity")

        def sequence_request_wire_id(journal, step, content, *, purpose=None, ordinal=1):
            del step, content, purpose, ordinal
            sequence = getattr(journal, "_sequence", 0)
            return f"{int(sequence):020x}"[-20:]

        _patch(wire_identity, "request_wire_id", sequence_request_wire_id)
        installed_modules.append(wire_identity.__name__)

    if mode in {"noise", "sequence_noise"}:
        verdict_identity = importlib.import_module("kernel_verdict_identity")
        original_prepare = verdict_identity.prepare_journal_event

        def prepare_without_noise_identity(journal, kind, fields):
            if kind == "wire_id_guard_test_noise":
                return dict(fields), False
            return original_prepare(journal, kind, fields)

        _patch(verdict_identity, "prepare_journal_event", prepare_without_noise_identity)
        installed_modules.append(verdict_identity.__name__)

        measurement = importlib.import_module("kernel_measurement")
        continuation = importlib.import_module("kernel_continuation_journal")

        def install_journal_hook(journal_type):
            original_emit = journal_type.emit

            def emit_with_one_noise_event(self, kind, **fields):
                result = original_emit(self, kind, **fields)
                if (kind == "request_submitted"
                        and not getattr(self, "_test_wire_id_noise_planted", False)):
                    self._test_wire_id_noise_planted = True
                    row = original_emit(
                        self, "wire_id_guard_test_noise",
                        producer="test_wire_ids_independent_of_journal",
                        dispatch=False,
                    )
                    writer_module = type(self).__module__
                    write_status(
                        marker_path, "planted",
                        pid=os.getpid(),
                        event_sequence=row.get("sequence"),
                        event_kind=row.get("kind"),
                        trigger_kind=kind,
                        trigger_request_id=fields.get("request_id"),
                        writer_module=writer_module,
                        writer_class=type(self).__qualname__,
                        writer_file=getattr(sys.modules.get(writer_module), "__file__", None),
                    )
                    write_stage(
                        "event_planted", pid=os.getpid(),
                        event_sequence=row.get("sequence"),
                        event_kind=row.get("kind"), trigger_kind=kind,
                        writer_module=writer_module,
                        writer_class=type(self).__qualname__,
                    )
                return result

            _patch(journal_type, "emit", emit_with_one_noise_event)
            installed_writer_types.append(
                f"{journal_type.__module__}.{journal_type.__qualname__}")

        # The active profile uses ContinuationJournal. Patch the base journal too so
        # this remains an engine-child hook if a fixture selects the legacy writer.
        install_journal_hook(measurement.AuditJournal)
        install_journal_hook(continuation.ContinuationJournal)
        installed_modules.extend((measurement.__name__, continuation.__name__))

    write_status(
        wrapper_status_path, "hook_installed",
        modules=installed_modules, writer_types=installed_writer_types,
    )
    write_stage(
        "hook_installed", mode=mode,
        modules=installed_modules, writer_types=installed_writer_types,
    )


def run_engine_child(argv):
    write_stage("engine_reached", pid=os.getpid(), child_file=str(child_path))
    return main(argv)


try:
    with patch.dict(main_globals, {"install_run_identity": _install_child_hook}):
        status = run_engine_child([str(child_path), str(run_path)])
finally:
    for patcher in reversed(patchers):
        patcher.stop()

raise SystemExit(status)
'''

_SCRIPT = {
    "model": "scripted",
    "steps": [
        {"tool": "read", "required": True,
         "arguments": {"path": "README.md"}},
        {"tool": "run_tests", "required": True,
         "arguments": {"target": "tests/test_wire_id_fixture.py"}},
        {"tool": "done", "required": True,
         "arguments": {"summary": "Read the README and ran the fixture test."}},
    ],
    # The scripted endpoint answers every request-tokenizer call as well as each
    # model send, so this test never needs a live model or external tokenizer.
    "tokenize": {"mode": "ok"},
}


def _request_files(run_dir: Path) -> dict[str, bytes]:
    from stateframe.request_records import list_request_records, read_request_record

    files = {
        f"requests/{name}": read_request_record(run_dir, name)
        for name in list_request_records(run_dir)
    }
    files.update({
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in sorted(run_dir.rglob("*.request.json"))
    })
    assert any(name.startswith("requests/") for name in files), "run wrote no request records"
    assert any(name.startswith("wire/") for name in files), "run wrote no wire request files"
    return files


def _run_scripted(monkeypatch, tmp_path: Path, name: str, mode: str,
                  child_hook: dict[str, str]) -> tuple[Path, dict[str, bytes], Path | None]:
    from stateframe import cli
    from stateframe.scripted_endpoint import Script, serve

    case = tmp_path / name
    case.mkdir()
    repo = case / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "README.md").write_text("Wire IDs must ignore journal noise.\n")
    (repo / "tests" / "test_wire_id_fixture.py").write_text(
        "import unittest\n"
        "from pathlib import Path\n\n"
        "class WireIdFixtureTest(unittest.TestCase):\n"
        "    def test_readme_exists(self):\n"
        "        self.assertIn('journal noise', Path('README.md').read_text())\n"
    )
    run_dir = case / "run"
    marker = run_dir / "wire-id-test-hook.json"

    child_hook["mode"] = mode
    child_hook["marker"] = str(marker) if mode in {"noise", "sequence_noise"} else ""
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "stateframe-home"))

    script = Script(json.loads(json.dumps(_SCRIPT)))
    endpoint = serve(script)
    arguments = [
        "run", "--repo", str(repo), "--task", "Read the README and validate the fixture.",
        "--endpoint", f"http://127.0.0.1:{endpoint.server_port}/v1",
        "--model", "scripted", "--run-dir", str(run_dir), "--quiet", "--json",
        "--max-turns", "8", "--runner-kind", "generic",
        "--test-cmd", "python -m unittest discover -s tests -q",
        "--test-python", sys.executable,
    ]
    try:
        cli.main(arguments)
    finally:
        child_hook["mode"] = "off"
        child_hook["marker"] = ""
        endpoint.shutdown()
        endpoint.server_close()
    script.assert_consumed()
    return run_dir, _request_files(run_dir), marker if marker.exists() else None


def _assert_one_event_before_request_two(
        run_dir: Path, marker_path: Path | None) -> None:
    wrapper_status_path = run_dir / "wire-id-test-hook.wrapper.json"
    launch_status_path = run_dir / "wire-id-test-hook.launch.json"
    stage_path = run_dir / "wire-id-test-hook.stages.jsonl"
    diagnostics = {
        "launch": json.loads(launch_status_path.read_text())
        if launch_status_path.is_file() else None,
        "wrapper": json.loads(wrapper_status_path.read_text())
        if wrapper_status_path.is_file() else None,
        "stages": [json.loads(line) for line in stage_path.read_text().splitlines()]
        if stage_path.is_file() else [],
    }
    assert marker_path is not None, (
        "engine-child journal hook did not write its marker; "
        f"launch/wrapper stages: {json.dumps(diagnostics, sort_keys=True)}"
    )
    marker = json.loads(marker_path.read_text())
    assert marker.get("stage") == "planted", marker
    assert diagnostics["launch"] is not None, "parent did not intercept the engine-child launch"
    assert diagnostics["launch"].get("stage") == "popen_intercepted"
    assert diagnostics["wrapper"] is not None
    assert diagnostics["wrapper"].get("stage") == "hook_installed"
    assert [record.get("stage") for record in diagnostics["stages"]] == [
        "wrapper_started", "child_module_loaded", "engine_reached",
        "install_run_identity_entered", "hook_installed", "event_planted",
    ], diagnostics["stages"]
    assert diagnostics["stages"][0].get("pid") == diagnostics["stages"][2].get("pid")
    assert diagnostics["stages"][2].get("pid") == marker.get("pid")
    assert marker.get("writer_class") in {"AuditJournal", "ContinuationJournal"}
    assert marker.get("event_kind") == "wire_id_guard_test_noise"
    assert marker.get("writer_module") in {
        "kernel_measurement", "kernel_continuation_journal",
    }
    writer_type = f"{marker['writer_module']}.{marker['writer_class']}"
    assert writer_type in diagnostics["wrapper"].get("writer_types", [])
    assert isinstance(marker.get("writer_file"), str)
    engine_dir = Path(json.loads((run_dir / "spec.json").read_text())["engine_dir"]).resolve()
    assert Path(marker["writer_file"]).resolve().parent == engine_dir
    request_records = {
        name: json.loads(raw)
        for name, raw in _request_files(run_dir).items()
        if name.startswith("requests/")
    }
    request_names = sorted(request_records)
    assert len(request_names) >= 2, "fixture did not build request 2"
    first_id = request_records[request_names[0]].get("request_id")
    second_id = request_records[request_names[1]].get("request_id")
    assert marker["trigger_kind"] == "request_submitted"
    assert marker["trigger_request_id"] == first_id

    rows = measurement_events(run_dir)
    noise = [row for row in rows if row.get("kind") == "wire_id_guard_test_noise"]
    assert len(noise) == 1, f"expected one injected journal event, found {len(noise)}"
    noise_row = noise[0]
    assert noise_row.get("sequence") == marker["event_sequence"]
    submitted = [row for row in rows if row.get("kind") == "request_submitted"]
    assert submitted and submitted[0]["sequence"] < noise_row["sequence"]
    second_budget_rows = [
        row for row in rows
        if row.get("kind") == "request_budget_final_wire"
        and row.get("request_id") == second_id
    ]
    assert second_budget_rows, "request 2 did not emit its final-wire budget row"
    assert noise_row["sequence"] < min(row["sequence"] for row in second_budget_rows)


def _assert_request_files_identical(before: dict[str, bytes], after: dict[str, bytes]) -> None:
    if before.keys() != after.keys():
        raise AssertionError(
            f"request file names differ: {sorted(before)} != {sorted(after)}"
        )
    changed = sorted(name for name in before if before[name] != after[name])
    assert not changed, f"request files differ byte-for-byte: {changed}"


@pytest.mark.host_sandbox
def test_wire_ids_ignore_one_internal_journal_event_before_request_two(
        tmp_path: Path, monkeypatch) -> None:
    from stateframe import runner

    wrapper = tmp_path / "engine_child_wrapper.py"
    wrapper.write_text(_ENGINE_CHILD_WRAPPER)
    child_hook = {"mode": "off", "marker": ""}
    original_popen = runner.subprocess.Popen

    def wrapped_popen(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) >= 4
                and Path(str(command[2])).name == "_child.py"):
            marker = child_hook["marker"]
            if marker:
                launch_status = Path(marker).with_suffix(".launch.json")
                launch_status.parent.mkdir(parents=True, exist_ok=True)
                launch_status.write_text(json.dumps({
                    "stage": "popen_intercepted",
                    "interpreter": str(command[0]),
                    "flag": str(command[1]),
                    "child": str(command[2]),
                    "run_dir": str(command[3]),
                    "wrapper": str(wrapper),
                }, sort_keys=True))
            command = [command[0], command[1], str(wrapper), str(command[2]),
                       str(command[3]), child_hook["mode"], child_hook["marker"]]
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "Popen", wrapped_popen)
    monkeypatch.setattr(
        runner.secrets, "token_hex",
        lambda n=16: "0123456789abcdef0123456789abcdef"[: 2 * n],
    )

    _, normal_files, _ = _run_scripted(
        monkeypatch, tmp_path, "normal", "off", child_hook)
    noisy_run, noisy_files, noisy_marker = _run_scripted(
        monkeypatch, tmp_path, "with-noise", "noise", child_hook)
    _assert_one_event_before_request_two(noisy_run, noisy_marker)
    _assert_request_files_identical(normal_files, noisy_files)

    # Sensitivity control: force request IDs back to the old global-sequence
    # shape in the isolated child. The same one-event injection must then move
    # later requests, proving the byte comparison detects this bug class.
    _, sequence_files, _ = _run_scripted(
        monkeypatch, tmp_path, "sequence-control", "sequence", child_hook)
    sequence_noise_run, sequence_noise_files, sequence_noise_marker = _run_scripted(
        monkeypatch, tmp_path, "sequence-control-with-noise", "sequence_noise", child_hook)
    _assert_one_event_before_request_two(sequence_noise_run, sequence_noise_marker)
    with pytest.raises(AssertionError, match="request files differ byte-for-byte"):
        _assert_request_files_identical(sequence_files, sequence_noise_files)
