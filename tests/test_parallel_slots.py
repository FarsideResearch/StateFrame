import json
from pathlib import Path

import pytest

from stateframe import profiles
from stateframe.cli import build_parser
from stateframe.endpoint import CHAT_SUFFIX, Endpoint
from stateframe.runner import RunError, RunOptions, auto_parallel_concurrency, prepare
import stateframe.parallel_resources as resources


def _repo(path: Path) -> Path:
    (path / "src").mkdir(parents=True)
    (path / "src" / "sample.py").write_text("VALUE = 1\n")
    (path / "tests").mkdir()
    (path / "tests" / "test_sample.py").write_text(
        "from src.sample import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n")
    return path


def _options(tmp_path: Path, *, run_name: str = "run", profile: str = "r2hc3",
             slots: str | None = None, mode: str = "auto") -> RunOptions:
    repo = _repo(tmp_path / f"repo-{run_name}")
    return RunOptions(
        repo=repo,
        task="inspect the sample",
        endpoint=Endpoint.from_args("http://127.0.0.1:8080/v1", "local-model", None),
        run_dir=tmp_path / run_name,
        max_turns=1,
        profile=profile,
        parallel_mode=mode,
        parallel_slots=slots,
    )


def _profile_with_slots(tmp_path: Path, value: str) -> Path:
    switches = profiles.load("r2hc3")
    switches["STATEFRAME_PARALLEL_SLOTS"] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": switches}))
    return path


@pytest.mark.parametrize("command", ["run", "chat"])
def test_run_and_chat_expose_subframe_slot_pool_help(command, capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as result:
        parser.parse_args([command, "--help"])

    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--parallel-slots LIST" in help_text
    assert "subframe" in help_text


@pytest.mark.parametrize("command", ["run", "chat"])
def test_run_and_chat_parse_the_slot_pool_value(command, tmp_path):
    args = [command, "--repo", str(tmp_path / "repo"), "--endpoint", "http://127.0.0.1:8080/v1",
            "--model", "local-model", "--parallel-slots", "5,2"]
    if command == "run":
        args.extend(["--task", "inspect the sample"])

    parsed = build_parser().parse_args(args)

    assert parsed.parallel_slots == "5,2"


def test_profile_slot_pool_is_normalized_into_run_and_worker_specs(tmp_path):
    profile = _profile_with_slots(tmp_path, "5, 2")
    options = _options(tmp_path, profile=str(profile))

    run_dir, _, spec = prepare(options)
    recorded = json.loads((run_dir / "spec.json").read_text())

    assert spec["parallel_slots"] == [5, 2]
    assert recorded["parallel_slots"] == [5, 2]
    assert "STATEFRAME_PARALLEL_SLOTS" not in recorded["environment"]


def test_cli_slot_pool_overrides_profile_and_unset_pool_adds_no_spec_key(tmp_path):
    profile = _profile_with_slots(tmp_path, "3,4")
    options = _options(tmp_path, run_name="cli", profile=str(profile), slots="7, 1")

    run_dir, _, spec = prepare(options)

    assert spec["parallel_slots"] == [7, 1]
    assert json.loads((run_dir / "spec.json").read_text())["parallel_slots"] == [7, 1]

    default_options = _options(tmp_path, run_name="default")
    default_dir, _, default_spec = prepare(default_options)
    default_record = json.loads((default_dir / "spec.json").read_text())
    assert "parallel_slots" not in default_spec
    assert "parallel_slots" not in default_record
    assert "STATEFRAME_PARALLEL_SLOTS" not in default_record["environment"]


@pytest.mark.parametrize("value", ["", "2,,5", "-1,2", "2,02", "２"])
def test_run_start_rejects_invalid_cli_slot_pools_before_creating_run(tmp_path, value):
    options = _options(tmp_path, slots=value)

    with pytest.raises(RunError, match="--parallel-slots"):
        prepare(options)

    assert not options.run_dir.exists()


def test_run_start_rejects_invalid_profile_slot_pool_with_both_setting_names(tmp_path):
    profile = _profile_with_slots(tmp_path, "2,2")
    options = _options(tmp_path, profile=str(profile))

    with pytest.raises(RunError) as result:
        prepare(options)

    assert "STATEFRAME_PARALLEL_SLOTS" in str(result.value)
    assert "--parallel-slots" in str(result.value)
    assert not options.run_dir.exists()


def test_explicit_parallelism_cannot_exceed_pool_size(tmp_path):
    options = _options(tmp_path, slots="2,5", mode="3")

    with pytest.raises(RunError, match="--parallel-slots pool size 2"):
        prepare(options)

    assert not options.run_dir.exists()


def test_auto_concurrency_counts_only_free_slots_in_pool_across_unique_endpoints(monkeypatch):
    primary = f"http://127.0.0.1:8080/v1{CHAT_SUFFIX}"
    routed = "http://127.0.0.2:8080/v1"
    seen = []

    def report(endpoint):
        seen.append(endpoint)
        if endpoint == "http://127.0.0.1:8080/v1":
            return [2, 5, 9]
        if endpoint == routed:
            return [5]
        raise AssertionError(f"unexpected endpoint {endpoint}")

    monkeypatch.setattr(resources, "idle_slots", report)
    spec = {
        "chat_url": primary,
        "parallel_slots": [5, 2],
        "routes": {
            "primary-again": {"endpoint": primary},
            "routed": {"endpoint": routed},
        },
    }

    assert auto_parallel_concurrency(spec, 10) == 3
    assert seen == ["http://127.0.0.1:8080/v1", routed]

    seen.clear()
    unpooled = {key: value for key, value in spec.items() if key != "parallel_slots"}
    assert auto_parallel_concurrency(unpooled, 10) == 4
    assert seen == ["http://127.0.0.1:8080/v1", routed]


def test_auto_concurrency_uses_pool_size_when_slots_are_unsupported(monkeypatch):
    monkeypatch.setattr(resources, "idle_slots", lambda _endpoint: None)

    assert auto_parallel_concurrency({"chat_url": "http://127.0.0.1:8080/v1", "parallel_slots": [8, 3]}, 4) == 2
    assert auto_parallel_concurrency({"chat_url": "http://127.0.0.1:8080/v1"}, 3) == 3
