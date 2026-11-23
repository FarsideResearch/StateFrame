"""CLI coverage of the repeated-tail policy against a local OpenAI-shaped server."""
from __future__ import annotations

import gzip
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread

from _journal import measurement_events
from stateframe import cli
from stateframe.scripted_endpoint import Script, make_handler


def _tool_call(name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": f"call-{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _reply(
    number: int,
    *,
    finish_reason: str,
    content: str = "",
    tools: tuple[tuple[str, dict[str, object]], ...] = (),
) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tools:
        message["tool_calls"] = [_tool_call(*tool) for tool in tools]
    return {
        "id": f"chatcmpl-{number}", "object": "chat.completion", "model": "guard-fake",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 80, "total_tokens": 100},
    }


class _GuardEndpoint:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.requests: list[dict[str, object]] = []
        self.lock = Lock()
        base_handler = make_handler(Script({"model": "guard-fake", "steps": []}))
        endpoint = self

        class Handler(base_handler):
            def do_POST(self) -> None:  # noqa: N802
                if self.path.rstrip("/") != "/v1/chat/completions":
                    super().do_POST()
                    return
                wire = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.headers.get("Content-Encoding") == "gzip":
                    wire = gzip.decompress(wire)
                with endpoint.lock:
                    endpoint.requests.append(json.loads(wire))
                    number = len(endpoint.requests)
                self._send(200, endpoint.reply(number))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = Thread(target=self.server.serve_forever, daemon=True)

    def reply(self, number: int) -> dict[str, object]:
        if self.mode == "normal":
            if number == 1:
                rows = "\n".join(
                    f"| item-{index:03d} | ready | shared queue processing record {index:03d} |"
                    for index in range(70)
                )
                return _reply(number, finish_reason="tool_calls", content=rows,
                              tools=(
                                  ("str_replace", {"path": "app.py", "old_str": "VALUE = 6",
                                                   "new_str": "VALUE = 7"}),
                                  ("done", {"summary": "app.py now defines VALUE as seven."}),
                              ))
            return _reply(number, finish_reason="tool_calls",
                          tools=(("done", {"summary": "app.py now defines VALUE as seven."}),))

        if number == 1:
            loop = "the answer has stopped adding information and repeats " * 24
            return _reply(number, finish_reason="length", content=loop)
        if number == 2:
            return _reply(number, finish_reason="tool_calls", tools=(
                ("str_replace", {"path": "app.py", "old_str": "VALUE = 6", "new_str": "VALUE = 7"}),
                ("done", {"summary": "app.py now defines VALUE as seven."}),
            ))
        return _reply(number, finish_reason="tool_calls",
                      tools=(("done", {"summary": "app.py now defines VALUE as seven."}),))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def __enter__(self) -> _GuardEndpoint:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def _run_cli(mode: str, tmp_path: Path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("VALUE = 6\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("STATEFRAME_HOME", str(home / ".stateframe"))
    monkeypatch.setenv("STATEFRAME_DEGENERATION_GUARD", "on")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    with _GuardEndpoint(mode) as endpoint:
        status = cli.main([
            "run", "--repo", str(repo), "--task", "Change app.py VALUE from six to seven.",
            "--endpoint", endpoint.url, "--model", "guard-fake", "--run-dir", str(run_dir),
            "--max-turns", "4", "--shell", "off", "--target", "app.py", "--edit", "app.py",
            "--test-python", sys.executable, "--check-cmd", f"{sys.executable} app.py",
            "--no-project-hooks", "--no-instructions",
            "--quiet", "--json",
        ])
        requests = tuple(endpoint.requests)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    summary_path = run_dir / "summary.json"
    assert summary_path.is_file(), output
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return status, summary, output, requests


def test_cli_keeps_normal_repetitive_tool_call_without_resending(
        tmp_path: Path, monkeypatch, capsys) -> None:
    status, summary, output, requests = _run_cli("normal", tmp_path, monkeypatch, capsys)
    assert status == 0, output
    assert summary["succeeded"] is True, summary
    assert len(requests) == 2, summary
    assert any(message.get("role") == "tool" for message in requests[1]["messages"])
    assert summary["engine_result"] == "SUBMITTED", summary
    assert summary["files_changed"] == ["app.py"], summary
    assert not [event for event in measurement_events(tmp_path / "run")
                if event.get("kind") == "degeneration_detected"]


def test_cli_resends_length_capped_repeated_tail_once(
        tmp_path: Path, monkeypatch, capsys) -> None:
    status, summary, output, requests = _run_cli("runaway", tmp_path, monkeypatch, capsys)
    assert status == 0, output
    assert summary["succeeded"] is True, summary
    assert len(requests) == 3, summary
    assert requests[0]["messages"] == requests[1]["messages"]
    assert any(message.get("role") == "tool" for message in requests[2]["messages"])
    assert summary["engine_result"] == "SUBMITTED", summary
    assert summary["files_changed"] == ["app.py"], summary
    events = [event for event in measurement_events(tmp_path / "run")
              if event.get("kind") == "degeneration_detected"]
    assert len(events) == 1, events
    assert events[0]["finish_reason"] == "length"
    assert events[0]["source"] == "content"
    assert events[0]["discarded_response"] is True
