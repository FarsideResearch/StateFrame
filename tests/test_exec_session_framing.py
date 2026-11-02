"""Execution supervisor control replies stay matched after timeouts and partial reads."""
import json
import socket
import threading

import pytest

import kernel_exec_session
from kernel_exec_session import ExecSession, ExecSessionError
from kernel_exec_supervisor import Supervisor
from stateframe import runner
from stateframe.endpoint import Endpoint


@pytest.fixture
def control(tmp_path):
    session = ExecSession(tmp_path)
    client, server = socket.socketpair()
    client.settimeout(0.2)
    server.settimeout(2)
    session._control = client
    yield session, server
    client.close()
    server.close()


def _request(peer):
    data = bytearray()
    while b"\n" not in data:
        data.extend(peer.recv(65536))
    return json.loads(data.partition(b"\n")[0])


def _reply(peer, rid, **fields):
    peer.sendall(json.dumps({"rid": rid, **fields}).encode() + b"\n")


def test_late_reply_after_timeout_does_not_poison_next_rpc(control):
    session, peer = control
    release_first = threading.Event()

    def supervisor():
        first = _request(peer)
        assert release_first.wait(2)
        _reply(peer, first["rid"], value="late")
        second = _request(peer)
        _reply(peer, second["rid"], value="current")

    thread = threading.Thread(target=supervisor)
    thread.start()
    try:
        session._control.settimeout(0.03)
        with pytest.raises(ExecSessionError, match="execution supervisor did not answer within"):
            session._rpc({"op": "first"})
        release_first.set()
        session._control.settimeout(1)
        assert session._rpc({"op": "second"})["value"] == "current"
    finally:
        release_first.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_two_replies_in_one_recv_are_separate(control):
    session, peer = control
    peer.sendall(b'{"rid":0,"value":"stale"}\n{"rid":1,"value":"right"}\n')
    assert session._rpc({"op": "one"})["value"] == "right"


def test_buffered_next_reply_survives_between_calls(control):
    session, peer = control
    peer.sendall(b'{"rid":1,"value":"first"}\n{"rid":2,"value":"second"}\n')
    assert session._rpc({"op": "one"})["value"] == "first"
    assert session._rpc({"op": "two"})["value"] == "second"


def test_reply_split_across_recv_chunks(control):
    session, peer = control
    original = session._control

    class ChunkedSocket:
        def sendall(self, payload):
            original.sendall(payload)

        def recv(self, _size):
            return original.recv(3)

    session._control = ChunkedSocket()
    _reply(peer, 1, value="split")
    assert session._rpc({"op": "one"})["value"] == "split"


def test_stale_rid_is_discarded_before_current_reply(control):
    session, peer = control
    peer.sendall(b'{"rid":-1,"value":"stale"}\n{"rid":1,"value":"right"}\n')
    assert session._rpc({"op": "one"})["value"] == "right"


def test_matching_error_reply_raises(control):
    session, peer = control
    _reply(peer, 1, error="supervisor rejected request")
    with pytest.raises(ExecSessionError, match="supervisor rejected request"):
        session._rpc({"op": "bad"})


def test_supervisor_exit_and_socket_failure_keep_distinct_errors(control):
    session, _ = control

    class EofSocket:
        def sendall(self, _payload):
            pass

        def recv(self, _size):
            return b""

    class BrokenSocket:
        def sendall(self, _payload):
            raise OSError("broken socket")

    session._control = EofSocket()
    with pytest.raises(ExecSessionError, match="execution supervisor exited"):
        session._rpc({"op": "one"})
    session._control = BrokenSocket()
    with pytest.raises(ExecSessionError, match="execution supervisor communication failed"):
        session._rpc({"op": "two"})


def test_receive_buffer_keeps_six_mib_bound(control):
    session, _ = control

    class OversizedSocket:
        def sendall(self, _payload):
            pass

        def recv(self, _size):
            return b"x" * ((6 << 20) + 1)

    session._control = OversizedSocket()
    with pytest.raises(ExecSessionError, match="execution supervisor response exceeds bound"):
        session._rpc({"op": "one"})


def test_supervisor_echoes_rid_on_success_and_error():
    supervisor = object.__new__(Supervisor)
    supervisor.dispatch = lambda request: {"seen": request}
    assert supervisor.reply({"op": "all", "rid": 4}) == {"seen": {"op": "all"}, "rid": 4}
    assert supervisor.reply({"op": "all"}) == {"seen": {"op": "all"}}
    supervisor.dispatch = lambda request: (_ for _ in ()).throw(ValueError("bad op"))
    assert supervisor.reply({"op": "bad", "rid": 5}) == {"error": "ValueError: bad op", "rid": 5}


def test_supervisor_queues_every_request_in_one_recv_chunk():
    supervisor = object.__new__(Supervisor)
    supervisor.dispatch = lambda request: {"op": request["op"]}
    supervisor.incoming = bytearray(b'{"op":"first","rid":1}\n{"op":"second","rid":2}\n')
    supervisor.outgoing = bytearray()
    supervisor._queue_replies()
    assert [json.loads(line) for line in supervisor.outgoing.splitlines()] == [
        {"op": "first", "rid": 1}, {"op": "second", "rid": 2}]
    assert not supervisor.incoming


@pytest.mark.parametrize("base,expected", [(None, 60), ("2.5", 10), ("0", 60), ("abc", 60),
                                            ("nan", 60), ("inf", 60)])
def test_rpc_timeout_scales_with_load_and_valid_base(monkeypatch, base, expected):
    if base is None:
        monkeypatch.delenv("STATEFRAME_EXEC_RPC_SECONDS", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_EXEC_RPC_SECONDS", base)
    monkeypatch.setattr("kernel_exec_session.os.getloadavg", lambda: (16, 1, 1))
    monkeypatch.setattr("kernel_exec_session.os.cpu_count", lambda: 4)
    assert kernel_exec_session.rpc_timeout_seconds() == expected


def test_rpc_timeout_uses_base_when_load_unavailable(monkeypatch):
    monkeypatch.delenv("STATEFRAME_EXEC_RPC_SECONDS", raising=False)
    monkeypatch.setattr("kernel_exec_session.os.getloadavg", lambda: (_ for _ in ()).throw(OSError()))
    assert kernel_exec_session.rpc_timeout_seconds() == 15


def test_rpc_timeout_load_factor_is_capped(monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXEC_RPC_SECONDS", "3")
    monkeypatch.setattr("kernel_exec_session.os.getloadavg", lambda: (1000, 1, 1))
    monkeypatch.setattr("kernel_exec_session.os.cpu_count", lambda: 4)
    assert kernel_exec_session.rpc_timeout_seconds() == 24


def test_timeout_message_names_seconds_and_host_load(control, monkeypatch):
    session, _ = control
    session._control.settimeout(0.01)
    monkeypatch.setattr("kernel_exec_session.os.getloadavg", lambda: (8, 1, 1))
    monkeypatch.setattr("kernel_exec_session.os.cpu_count", lambda: 4)
    with pytest.raises(ExecSessionError, match=r"within 0.01 s \(host load 8 on 4 CPUs\).*STATEFRAME_EXEC_RPC_SECONDS"):
        session._rpc({"op": "unanswered"})


@pytest.mark.parametrize("bad", ["", "0", "-1", "nope", "nan", "inf"])
def test_runner_refuses_invalid_rpc_timeout(monkeypatch, bad):
    monkeypatch.setenv("STATEFRAME_EXEC_RPC_SECONDS", bad)
    with pytest.raises(runner.RunError, match="STATEFRAME_EXEC_RPC_SECONDS.*positive"):
        runner.resolve_exec_rpc_env()


def test_runner_prepare_refuses_invalid_rpc_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_EXEC_RPC_SECONDS", "0")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 1\n")
    (repo / "test_app.py").write_text("def test_app(): assert True\n")
    options = runner.RunOptions(repo=repo, task="Inspect app", endpoint=Endpoint("http://127.0.0.1:8080/v1", "base"),
                                run_dir=tmp_path / "run", target="app.py")
    with pytest.raises(runner.RunError, match="STATEFRAME_EXEC_RPC_SECONDS.*positive"):
        runner.prepare(options, printer=lambda *_: None)


@pytest.mark.parametrize("base", [None, "2.5"])
def test_runner_spec_only_has_rpc_timeout_when_shell_sets_it(tmp_path, monkeypatch, base):
    if base is None:
        monkeypatch.delenv("STATEFRAME_EXEC_RPC_SECONDS", raising=False)
    else:
        monkeypatch.setenv("STATEFRAME_EXEC_RPC_SECONDS", base)
    assert "STATEFRAME_EXEC_RPC_SECONDS" not in runner.ignored_shell_settings()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("pass\n")
    (repo / "test_app.py").write_text("def test_app(): assert True\n")
    options = runner.RunOptions(repo=repo, task="Inspect app", endpoint=Endpoint("http://127.0.0.1:8080/v1", "base"),
                                run_dir=tmp_path / "run")
    _, _, spec = runner.prepare(options, printer=lambda *_: None)
    assert ("STATEFRAME_EXEC_RPC_SECONDS" in spec["environment"]) is (base is not None)
    assert spec["environment"].get("STATEFRAME_EXEC_RPC_SECONDS") == base
