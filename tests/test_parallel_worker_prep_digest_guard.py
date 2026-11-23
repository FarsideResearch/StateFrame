import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests import test_parallel_worker_prep as parity
from tests.test_parallel_worker_prep import _assert_wire_bytes

_OFF_REQUEST = (
    Path(__file__).resolve().parent
    / "fixtures/parallel/off_wire_main_b520a637/0001.request.json"
)


def test_identical_off_request_replay_passes_digest_guard():
    expected = _OFF_REQUEST.read_bytes()

    _assert_wire_bytes(expected, expected, send=1, name="initial")


def test_off_request_content_change_fails_digest_guard():
    expected = _OFF_REQUEST.read_bytes()
    changed = expected.replace(b"Fix the pricing code", b"Fix the pricing task", 1)

    assert changed != expected
    with pytest.raises(AssertionError, match="first differing byte"):
        _assert_wire_bytes(expected, changed, send=1, name="initial")


@pytest.mark.parametrize(
    ("original", "changed"),
    [
        ('Preserve {"edit_receipt":"0123456789abcdef0123"} verbatim.',
         'Preserve {"edit_receipt":"fedcba9876543210fedc"} verbatim.'),
        ("Preserve `exit 0 | 1.28s | standard` verbatim.",
         "Preserve `exit 0 | 0.64s | standard` verbatim."),
    ],
    ids=["task-receipt-literal", "task-duration-literal"],
)
def test_release_comparison_rejects_off_task_changes(tmp_path, original, changed):
    baseline, candidate = (tmp_path / name for name in ("baseline", "candidate"))
    for run, task in ((baseline, original), (candidate, changed)):
        wire = run / "wire"
        wire.mkdir(parents=True)
        (wire / "0001.request.json").write_text(json.dumps({
            "messages": [{"role": "user", "content": task}],
        }))

    with pytest.raises(AssertionError, match=r"component messages\[0\]\.content"):
        parity._assert_wire_parity(baseline, "http://127.0.0.1:30001/v1",
                                   candidate, "http://127.0.0.1:30002/v1")


def test_off_grep_does_not_suggest_a_definition_deleted_before_first_lookup(tmp_path, monkeypatch):
    from kernel_bolt import KernelBolt

    assert "eager_symbols" not in inspect.signature(KernelBolt).parameters

    monkeypatch.setenv("KERNEL_INTERVENTIONS", "1")
    monkeypatch.setenv("STATEFRAME_WORKSPACE_OBJECTS", "off")
    monkeypatch.setenv("STATEFRAME_TURN_REPO_INTEL", "off")
    monkeypatch.setenv("STATEFRAME_TURN_INDEXED_READS", "off")
    monkeypatch.setenv("STATEFRAME_REPEAT_GATE", "on")
    source = tmp_path / "answer.py"
    source.write_text("def removed(): pass\n")
    bolt = KernelBolt(tmp_path, "", "lookup")
    assert bolt.trace[0] == {
        "cylinder": "liveness", "event": "ORGAN_DISARMED",
        "detail": "organ=bolt.symbol status=deferred until first unresolved grep", "turn": None,
    }
    source.write_text("def replacement(): pass\n")

    assert bolt.observe("grep removed", "(no matches)") == "(no matches)"
    assert "`replacement` EXISTS" in bolt.observe("grep replacement", "(no matches)")


def test_default_off_comparison_exercises_the_candidate_profile(tmp_path, monkeypatch):
    from stateframe import profiles

    # Plant a change outside the cache switches that the historical wire case pins.
    switch = "STATEFRAME_REPEAT_GATE"
    shipped = json.loads((profiles.HERE / "r2hc3.json").read_text())
    assert shipped["switches"][switch] == "on"
    shipped["switches"][switch] = "off"
    candidate_profiles = tmp_path / "candidate-profiles"
    candidate_profiles.mkdir()
    (candidate_profiles / "r2hc3.json").write_text(json.dumps(shipped))
    monkeypatch.setattr(profiles, "HERE", candidate_profiles)

    class CandidateProfileChecked(Exception):
        pass

    def check_selected_profile(_python, _script, _repo, run, *, parallel_off, profile, **_kwargs):
        selected = profiles.load(profile)[switch]
        if run.name == "new-run":
            assert parallel_off
            assert selected == profiles.load(profiles.DEFAULT)[switch] == "off", (
                "both release sides use the historical profile and hide the changed candidate setting"
            )
            raise CandidateProfileChecked
        assert not parallel_off
        assert selected == "on"
        return run, "http://127.0.0.1:30001/v1"

    isolate_release_test(monkeypatch, check_selected_profile)

    with pytest.raises(CandidateProfileChecked):
        parity.test_default_parallel_off_wire_and_public_artifacts_match_merged_main(tmp_path)


def isolate_release_test(monkeypatch, run_engine):
    original_run = subprocess.run

    def arrange_release(root, release):
        examples = release / "examples"
        shutil.copytree(root / "examples/discount_bug", examples / "discount_bug")
        shutil.copy2(root / "examples/discount_bug.script.json", examples)
        profile = release / "src/stateframe/profiles/r2hc3.json"
        profile.parent.mkdir(parents=True)
        profile.write_text(json.dumps({
            "schema": "stateframe-product-profile/1",
            "switches": {"PYTHONHASHSEED": "0", "STATEFRAME_REPEAT_GATE": "on"},
        }))

    def setup(command, **kwargs):
        if command[1:3] == ["-m", "venv"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        return original_run(command, **kwargs)

    monkeypatch.setattr(parity, "_archive_off_release", arrange_release)
    monkeypatch.setattr(subprocess, "run", setup)
    monkeypatch.setattr(parity, "_install_release_package", lambda *_args: None)
    monkeypatch.setattr(parity, "_directory_effect_record", lambda *_args: {
        "exit_code": 0,
        "post_manifest_digest": "stable",
        "observation": "files changed by this command: created: generated",
    })
    monkeypatch.setattr(parity, "_read_child_environment",
                        lambda *_args, **_kwargs: {"PYTHONHASHSEED": "0"})
    monkeypatch.setattr(parity, "_run_scripted_engine", run_engine)


def test_release_parity_checks_wire_before_reading_artifacts(tmp_path, monkeypatch):
    def write_wire(_python, _script, _repo, run, **_kwargs):
        task = "Changed task" if run.name == "new-run" else "Original task"
        (run / "wire").mkdir(parents=True)
        (run / "wire/0001.request.json").write_text(json.dumps({
            "messages": [{"role": "user", "content": task}],
        }))
        return run, "http://127.0.0.1:30001/v1"

    isolate_release_test(monkeypatch, write_wire)

    with pytest.raises(AssertionError, match=r"component messages\[0\]\.content") as error:
        parity.test_default_parallel_off_wire_and_public_artifacts_match_merged_main(tmp_path)
    assert "component messages[0].content" in str(error.value)


def test_default_off_comparison_detects_changed_profile_provenance(tmp_path, monkeypatch):
    def write_run(_python, _script, _repo, run, *, profile, **_kwargs):
        candidate = run.name == "new-run"
        (run / "wire").mkdir(parents=True)
        (run / "wire/0001.request.json").write_text('{"messages":[]}')
        spec = {
            "profile": profile,
            "environment": {"PYTHONHASHSEED": "0"},
            "environment_sources": {
                "PYTHONHASHSEED": "profile:r2hc3" if candidate else f"profile:{profile}",
            },
        }
        (run / "spec.json").write_text(json.dumps(spec))
        (run / "summary.json").write_text("{}")
        artifact = {"result": {"result": "SUBMITTED"}, "seconds": 0.5, "exit_status": 0}
        (run / "engine-result.json").write_text(json.dumps(artifact))
        return run, "http://127.0.0.1:30001/v1"

    isolate_release_test(monkeypatch, write_run)

    with pytest.raises(AssertionError, match="default-off run spec values differ"):
        parity.test_default_parallel_off_wire_and_public_artifacts_match_merged_main(tmp_path)


def test_default_off_spec_rejects_capabilities_bound_to_the_wrong_loopback_port(tmp_path):
    capability = {"dialect": "llama.cpp", "props_status": 200}
    released = {"provider_capabilities": {"http://127.0.0.1:30001": capability}}
    candidate = {"provider_capabilities": {"http://127.0.0.1:39999": capability}}

    with pytest.raises(AssertionError, match="default-off run spec values differ"):
        parity._assert_default_off_spec_parity(
            released, candidate, tmp_path / "released", "http://127.0.0.1:30001/v1",
            tmp_path / "candidate", "http://127.0.0.1:30002/v1",
        )


def test_default_off_result_rejects_a_corrupt_delivery_manifest_digest(tmp_path):
    from kernel_delivery_manifest import build_manifest

    manifest = build_manifest(phase="submitted", files_changed=[], claims=[], receipts=[],
                              open_items=[], payload={})
    corrupt = {**manifest, "manifest_id": "sha256:" + "0" * 64}
    assert manifest["manifest_id"] != corrupt["manifest_id"]
    released = {"result": {"delivery_manifest": manifest}, "seconds": 0.5, "exit_status": 0}
    candidate = {**released, "result": {"delivery_manifest": corrupt}}

    parity._assert_default_off_result_parity(
        released, released, tmp_path / "released", "http://127.0.0.1:30001/v1",
        tmp_path / "candidate", "http://127.0.0.1:30002/v1",
    )
    with pytest.raises(AssertionError, match="default-off engine results differ"):
        parity._assert_default_off_result_parity(
            released, candidate, tmp_path / "released", "http://127.0.0.1:30001/v1",
            tmp_path / "candidate", "http://127.0.0.1:30002/v1",
        )
    with pytest.raises(AssertionError, match="invalid delivery manifest digest"):
        parity._assert_default_off_result_parity(
            candidate, candidate, tmp_path / "released", "http://127.0.0.1:30001/v1",
            tmp_path / "candidate", "http://127.0.0.1:30002/v1",
        )
