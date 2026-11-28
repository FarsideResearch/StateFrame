"""Scripted provider-boundary coverage for the dashboard action cue."""
from __future__ import annotations

import hashlib
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from textwrap import dedent

from stateframe.runner import final_candidate
from stateframe.scripted_endpoint import Script, make_handler
from tools.pinned_tree import pinned_tree


ROOT = Path(__file__).resolve().parents[1]
BASE = "fe31666bd5e3a8d1e109c5f3cd7638ab176adcbe"
SWITCH = "STATEFRAME_TASK_HIERARCHY"
PATH = "dashboard.html"
CONTROL = "Review incidents"
DESTINATION = "/incidents"
ACTION = f'Review current incidents using the "{CONTROL}" link to {DESTINATION}.'
BRIEF = (
    f"Refine the operations dashboard in {PATH}.\n"
    f"Immediate user action: {ACTION}\n"
    "Keep the incident count and accessible status heading."
)
SOURCE = (
    '<main aria-labelledby="status">\n'
    '  <h1 id="status">Operations status</h1>\n'
    '  <p>3 open incidents</p>\n'
    f'  <a href="{DESTINATION}">{CONTROL}</a>\n'
    '</main>\n'
)
PROMOTED_DASHBOARD = dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Operations status</title>
      <style>
        .primary-action {
          display: inline-block;
          padding: 0.5em 1em;
          background-color: ButtonText;
          color: ButtonFace;
          font-weight: 700;
        }
      </style>
    </head>
    <body>
      <main aria-labelledby="status">
        <h1 id="status">Operations status</h1>
        <p>3 open incidents</p>
        <a class="primary-action" href="/incidents">Review incidents</a>
      </main>
    </body>
    </html>
""")
CUE_LABEL = "[Design focus]"
LAUNCH = dedent("""\
    import os
    from pathlib import Path
    import secrets
    import sys
    from types import SimpleNamespace
    from urllib.request import ProxyHandler, Request, build_opener
    from stateframe import _child, cli, runner
    from stateframe.endpoint import Endpoint

    runner.secrets = SimpleNamespace(
        token_hex=lambda size: "0123456789abcdef" * 2 if size == 16 else secrets.token_hex(size),
    )

    # Invoke the real request path in a disposable process without shell provisioning.
    args = cli.build_parser().parse_args(sys.argv[1:])
    endpoint = Endpoint.from_args(args.endpoint, args.model, args.api_key_env)
    options = cli._run_options(args, endpoint)
    options.check_source = "none"
    run, _, spec = runner.prepare(options, printer=lambda *_: None)
    name = _child.RUNNER_HOME_IDENTITY_ENV
    os.environ[name] = spec["environment"][name]
    _child.watch_parent = lambda: None
    # Record the real wire and journal files without flushing disposable data to disk.
    os.fsync = lambda _fd: None
    os.chdir(run)

    # Send the sealed bytes to the loopback oracle without a curl subprocess.
    sys.path.insert(0, str(Path(_child.__file__).parent / "engine"))
    import kernel_bounded_http

    opener = build_opener(ProxyHandler({}))

    def send(wire, endpoint, _key, **kwargs):
        request = Request(endpoint, data=wire, headers={"Content-Type": "application/json"})
        with opener.open(request, timeout=30) as response:
            return response.read()

    kernel_bounded_http.request_bytes = send
    raise SystemExit(_child.main([_child.__file__, str(run)]))
""")


def _provider_requests(tree, work, setting=None, *, brief=BRIEF):
    repo = work / "repo"
    repo.mkdir(parents=True)
    (repo / PATH).write_text(SOURCE, encoding="utf-8")
    run = work / "run"
    script = Script({
        "tokenize": {"mode": "ok"},
        "steps": [
            {"tool": "str_replace", "arguments": {
                "path": PATH, "old_str": SOURCE, "new_str": PROMOTED_DASHBOARD,
            }, "required": True},
            {"tool": "done", "arguments": {"summary": "Reviewed the dashboard."}, "required": True},
        ],
    })
    requests = []

    class RecordingHandler(make_handler(script)):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append(raw)
                self.rfile = BytesIO(raw)
            super().do_POST()

    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("STATEFRAME_", "SF_", "REACT_", "LLAMA_CPP_"))
    }
    env.update(
        STATEFRAME_HOME=str(work / "home"),
        STATEFRAME_PROVIDER="llama.cpp",
        PYTHONPATH=str(tree / "src"),
        PYTHONHASHSEED="0",
        PYTHONDONTWRITEBYTECODE="1",
    )
    if setting is not None:
        env[SWITCH] = setting

    with ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run(
                [sys.executable, "-c", LAUNCH, "run", "--repo", str(repo), "--task", brief,
                 "--target", PATH, "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                 "--model", "scripted", "--run-dir", str(run), "--test-python", sys.executable,
                 "--max-turns", "2", "--runner-kind", "generic", "--shell", "off",
                 "--no-git-tool", "--quiet"],
                cwd=tree, env=env, capture_output=True, text=True, timeout=240, check=False,
            )
        finally:
            server.shutdown()
            thread.join()
    assert result.returncode == 0, result.stdout + result.stderr
    script.assert_consumed()
    assert len(requests) == 2
    recorded = sorted((run / "wire").glob("*.request.json"))
    assert requests == [path.read_bytes() for path in recorded]
    sends = [json.loads(line) for line in (run / "sends.jsonl").read_text().splitlines()]
    assert [send["send"] for send in sends] == [1, 2]
    assert [send["request_sha256"] for send in sends] == [
        hashlib.sha256(raw).hexdigest() for raw in requests
    ]
    spec = json.loads((run / "spec.json").read_text())
    assert spec["environment"].get(SWITCH) == ("off" if setting == "false" else setting)
    return requests


def _user_messages(raw):
    return [message["content"] for message in json.loads(raw)["messages"]
            if message["role"] == "user"]


def test_f002_dashboard_brief_names_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory):
    base = pinned_tree(BASE, tmp_path_factory)
    expected = _provider_requests(base, tmp_path / "base")
    assert _provider_requests(ROOT, tmp_path / "unset") == expected
    assert _provider_requests(ROOT, tmp_path / "false", "false") == expected


def test_f002_dashboard_brief_names_switch_defaults_off(tmp_path):
    for raw in _provider_requests(ROOT, tmp_path / "default"):
        messages = _user_messages(raw)
        assert any(BRIEF in message for message in messages)
        assert all(CUE_LABEL not in message for message in messages)


def test_f002_dashboard_brief_names_on_path_is_bounded_and_idempotent(tmp_path):
    enabled = _provider_requests(ROOT, tmp_path / "enabled", "on")

    first_messages = _user_messages(enabled[0])
    cue, = [message for message in first_messages if message.startswith(CUE_LABEL + "\n")]
    _, selected_action, instruction = cue.splitlines()
    assert selected_action == ACTION
    assert instruction == (
        "Give the existing entry point for this action the strongest visual emphasis "
        "and preserve its destination."
    )
    task, = [message for message in first_messages if BRIEF in message]
    assert task.count(ACTION) == 1
    assert len(cue.encode()) - len(ACTION.encode()) <= 128
    assert sum(message.count(ACTION) for message in first_messages) == 2
    assert all(raw.count(CUE_LABEL.encode()) <= 1 for raw in enabled)
    assert CUE_LABEL.encode() not in enabled[1]
    assert any(BRIEF in message for message in _user_messages(enabled[1]))

    # Bind the action to the actual source window for this same send.
    link = f'<a href="{DESTINATION}">{CONTROL}</a>'
    source_window, = [message for message in first_messages
                      if message.startswith(f"[Current source: {PATH}]\n") and link in message]
    source_info = json.loads(source_window.splitlines()[2])
    assert source_info["revision"] == "sha256:" + hashlib.sha256(SOURCE.encode()).hexdigest()

    run = tmp_path / "enabled" / "run"
    reply = json.loads((run / "wire" / "0001.response.json").read_bytes())
    edit, = reply["choices"][0]["message"]["tool_calls"]
    assert edit["function"]["name"] == "str_replace"
    arguments = json.loads(edit["function"]["arguments"])
    assert arguments["path"] == PATH
    assert arguments["old_str"] == SOURCE

    candidate = final_candidate(run)
    assert candidate is not None
    dashboard = candidate[PATH]
    assert dashboard is not None
    primary_link = f'<a class="primary-action" href="{DESTINATION}">{CONTROL}</a>'
    assert dashboard.count(b'class="primary-action"') == 1
    assert primary_link.encode() in dashboard
    assert dashboard == arguments["new_str"].encode() == PROMOTED_DASHBOARD.encode()
    post_edit_revisions = {
        json.loads(message.splitlines()[2])["revision"]
        for message in _user_messages(enabled[1])
        if message.startswith(f"[Current source: {PATH}]\n") and primary_link in message
    }
    assert post_edit_revisions == {"sha256:" + hashlib.sha256(dashboard).hexdigest()}

    assert _provider_requests(ROOT, tmp_path / "repeat", "on") == enabled
    assert final_candidate(tmp_path / "repeat" / "run") == candidate
    oversized = BRIEF.replace(ACTION, ACTION + "x" * 161)
    for raw in _provider_requests(ROOT, tmp_path / "oversized", "on", brief=oversized):
        assert all(CUE_LABEL not in message for message in _user_messages(raw))
