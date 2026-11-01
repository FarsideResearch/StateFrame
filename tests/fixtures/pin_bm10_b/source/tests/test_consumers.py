import importlib

import pytest


def _batch_api():
    return importlib.import_module("pkg.batch")


@pytest.mark.feature
def test_service_worker_and_cli_use_the_package_runner(monkeypatch):
    batch = _batch_api()
    from pkg import cli, service, worker

    real_runner = batch.run_batch
    calls = []

    def recording_runner(items, work, *, stop_on_error=False):
        calls.append(tuple(items))
        return real_runner(items, work, stop_on_error=stop_on_error)

    monkeypatch.setattr(batch.runner, "run_batch", recording_runner)
    assert service.run_batch([1], lambda value: value + 1)[0].value == 2
    assert worker.Worker().process_batch([2], lambda value: value + 1)[0].value == 3
    output = cli.run_batch([3], lambda value: value + 1)
    assert output == [{"index": 0, "item": 3, "ok": True, "value": 4, "error": None}]
    assert calls == [(1,), (2,), (3,)]


@pytest.mark.feature
def test_completed_package_imports_from_a_fresh_interpreter():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from pkg import BatchResult, run_batch; assert run_batch([1], str)[0].value == '1'"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
