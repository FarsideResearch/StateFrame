"""Answer metrics must follow durable ACKs on the real asynchronous channel lane."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from _timeouts import scaled
import stateframe.parallel_host as host
from stateframe.parallel_channel import ChannelWrites
from stateframe.parallel_journal import Journal
from test_parallel_host import _request, _write_request


def _wait(predicate):
    deadline = time.monotonic() + scaled(5)
    while not predicate():
        assert time.monotonic() < deadline, "channel helper did not reach the requested boundary"
        time.sleep(0.01)


@contextmanager
def _gated_channel(tmp_path, monkeypatch):
    """Real helper process + real fsyncs, independently gated before publication and ACK."""
    supervisor = host.ParallelSupervisor(tmp_path / "run")
    rq = supervisor.parallel_root / "call/attempt-1/rq"
    request = rq / "001.json"
    answer = rq.parent / "an/001.json"
    _write_request(request, _request(1, "call", 1, "status", {"unit_id": "a"}))
    arrived = time.time() - 3.0
    os.utime(request, (arrived, arrived))
    publish = tmp_path / "publish"
    acknowledge = tmp_path / "acknowledge"
    entered = tmp_path / "entered"
    visible = tmp_path / "visible"
    shim = tmp_path / "gated_channel_helper.py"
    shim.write_text(
        "import sys, time\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(Path(host.__file__).resolve().parents[1])!r})\n"
        "from stateframe import parallel_host as host, parallel_journal as journal\n"
        "original = host._write_json\n"
        "def wait(path):\n"
        "    while not Path(path).exists(): time.sleep(0.005)\n"
        "def write(path, value, **kwargs):\n"
        f"    if str(path) == {str(answer)!r}:\n"
        f"        Path({str(entered)!r}).touch()\n"
        f"        wait({str(publish)!r})\n"
        "        result = original(path, value, **kwargs)\n"
        f"        Path({str(visible)!r}).touch()\n"
        f"        wait({str(acknowledge)!r})\n"
        "        return result\n"
        "    return original(path, value, **kwargs)\n"
        "host._write_json = write\njournal.serve()\n"
    )
    processes = []

    def spawn(journal):
        journal.epoch += 1
        process = subprocess.Popen([sys.executable, "-u", str(shim)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        processes.append(process)
        return process

    monkeypatch.setattr(Journal, "_spawn", spawn)
    writer = supervisor._channel_writes = ChannelWrites()
    supervisor._channel_thread = threading.current_thread()
    try:
        def ready():
            supervisor._service_channels()
            return entered.exists()

        _wait(ready)
        assert answer in writer._pending
        assert supervisor._answer_latency.snapshot()["count"] == 0
        yield supervisor, writer, request, answer, publish, acknowledge, visible
    finally:
        publish.touch()
        acknowledge.touch()
        writer.close()
        writer.journal._thread.join(scaled(5))
        assert not writer.journal._thread.is_alive()
        for process in processes:
            assert process.wait(timeout=scaled(5)) == 0
            process.stdout.close()


def test_answer_published_between_passes_is_measured_once_at_ack(tmp_path, monkeypatch):
    with _gated_channel(tmp_path, monkeypatch) as lane:
        supervisor, writer, request, answer, publish, acknowledge, visible = lane
        # The preceding pass sampled before publication. Both publication and its ACK
        # now arrive while no channel pass is running, exactly the formerly lost case.
        publish.touch()
        acknowledge.touch()
        _wait(lambda: writer._pending[answer].ticket.done.is_set())
        assert visible.exists() and answer.is_file()
        supervisor._service_channels()
        measured = supervisor._answer_latency.snapshot()
        assert measured["count"] == 1
        assert measured["max"] >= 3.0
        # A replay and further drains must never count the same publication again.
        supervisor.serve(request.parent, answer.parent)
        for _ in range(3):
            supervisor._service_channels()
        assert supervisor._answer_latency.snapshot() == measured
        metrics = json.loads((supervisor.parallel_root / "host-metrics.json").read_text())
        assert metrics["answer_latency_seconds"] == measured


def test_visible_answer_waits_for_ack_and_includes_durability_delay(tmp_path, monkeypatch):
    with _gated_channel(tmp_path, monkeypatch) as lane:
        supervisor, writer, request, answer, publish, acknowledge, visible = lane
        original_serve = supervisor.serve

        def publish_during_pass(rq, an):
            publish.touch()
            _wait(visible.exists)
            assert answer.is_file() and not writer._pending[answer].ticket.done.is_set()
            return original_serve(rq, an)

        monkeypatch.setattr(supervisor, "serve", publish_during_pass)
        # Publish between the pass's arrival snapshot and its completion sample,
        # while holding the helper's ACK beyond the end of that pass.
        supervisor._service_channels()
        assert supervisor._answer_latency.snapshot()["count"] == 0
        monkeypatch.setattr(supervisor, "serve", original_serve)
        # Changing/removing request metadata after acceptance must not replace the
        # retained arrival time, nor can answer mtime stand in for ACK completion.
        pre_ack_latency = answer.stat().st_mtime - request.stat().st_mtime
        request.unlink()
        time.sleep(0.3)
        acknowledge.touch()
        _wait(lambda: writer._pending[answer].ticket.done.is_set())
        supervisor._service_channels()
        measured = supervisor._answer_latency.snapshot()
        assert measured["count"] == 1
        assert measured["max"] >= pre_ack_latency + 0.3
        for _ in range(3):
            supervisor._service_channels()
        assert supervisor._answer_latency.snapshot() == measured
