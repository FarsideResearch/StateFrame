import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

from kernel_hooks import parse_outcome

DRIVER = (
    Path(__file__).resolve().parents[1] / "src/stateframe/engine/kernel_hook_driver.py"
)


def invoke(argv, *, payload=b"{}", timeout=2):
    request = {
        "argv": argv,
        "payload_b64": base64.b64encode(payload).decode(),
        "timeout": timeout,
        "stream_cap": 32768,
    }
    process = subprocess.run(
        [sys.executable, "-I", str(DRIVER)],
        input=json.dumps(request).encode(),
        capture_output=True,
        timeout=timeout + 5,
        check=True,
    )
    assert process.stderr == b""
    return json.loads(process.stdout)


@pytest.mark.parametrize("code,expected", [(0, "continue"), (2, "deny"), (1, "error")])
def test_exit_status_and_separate_streams(code, expected):
    script = (
        "import sys;print('hello');print('reason',file=sys.stderr);sys.exit(%d)" % code
    )
    envelope = invoke([sys.executable, "-c", script])
    assert base64.b64decode(envelope["stdout_b64"]) == b"hello\n"
    assert base64.b64decode(envelope["stderr_b64"]) == b"reason\n"
    assert parse_outcome("pre_tool", envelope).decision == expected


def test_json_text_protocol_and_garbage():
    for output, expected in (
        ('{"decision":"deny","reason":"no"}', "deny"),
        ("{broken", "error"),
        ("ordinary text", "continue"),
    ):
        envelope = invoke([sys.executable, "-c", f"print({output!r})"])
        assert parse_outcome("pre_tool", envelope).decision == expected


def test_stream_flood_retains_prefix_and_counts():
    envelope = invoke(
        [
            sys.executable,
            "-c",
            "import os;os.write(1,b'x'*40000);os.write(2,b'y'*41000)",
        ]
    )
    assert envelope["stdout_total"] == 40000 and envelope["stderr_total"] == 41000
    assert len(base64.b64decode(envelope["stdout_b64"])) == 32768
    assert len(base64.b64decode(envelope["stderr_b64"])) == 32768
    assert parse_outcome("post_tool", envelope).omitted_bytes >= 7232
    envelope = invoke([sys.executable, "-c", "import os;os.write(1,b'{' + b'x'*40000)"])
    assert parse_outcome("post_tool", envelope).error_code == "protocol"
    binary = invoke(
        [sys.executable, "-c", "import os;os.write(1,b'\\x00\\xff\\x1b[31m')"]
    )
    text = parse_outcome("post_tool", binary).context
    assert "\x00" not in text and "\x1b" not in text


def test_timeout_kills_descendant_holding_pipe():
    script = "import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(9)']);sys.exit(0)"
    envelope = invoke([sys.executable, "-c", script], timeout=1)
    assert envelope["timed_out"] is True
    assert envelope["cleanup_ok"] is True


def test_signal_and_maximum_timeout():
    signaled = invoke(
        [sys.executable, "-c", "import os,signal;os.kill(os.getpid(),signal.SIGTERM)"]
    )
    assert signaled["signal"] == 15
    assert parse_outcome("pre_tool", signaled).error_code == "exit"
    request = {
        "argv": [sys.executable, "-c", "print('ok')"],
        "payload_b64": "e30=",
        "timeout": 590,
        "stream_cap": 32768,
    }
    result = subprocess.run(
        [sys.executable, "-I", str(DRIVER)],
        input=json.dumps(request).encode(),
        capture_output=True,
        timeout=5,
        check=True,
    )
    assert json.loads(result.stdout)["exit_code"] == 0


def test_missing_envelope_is_containment():
    assert parse_outcome("pre_tool", None).decision == "containment"
    assert (
        parse_outcome("pre_tool", {"schema": "stateframe-hook-envelope/1"}).decision
        == "containment"
    )
    damaged = invoke([sys.executable, "-c", "pass"])
    damaged["stdout_b64"] = "%%%"
    assert parse_outcome("pre_tool", damaged).decision == "containment"


def test_large_payload_and_argv_transport():
    script = "import sys;print(len(sys.stdin.buffer.read()),len(sys.argv)-1)"
    envelope = invoke(
        [sys.executable, "-c", script] + ["x" * 1000] * 63, payload=b"p" * (256 << 10)
    )
    assert base64.b64decode(envelope["stdout_b64"]) == b"262144 63\n"


def test_launch_error_is_enveloped():
    envelope = invoke(["/definitely/not/a/hook/executable"])
    assert envelope["launch_error"]
    assert envelope["cleanup_ok"] is True
    assert parse_outcome("pre_tool", envelope).decision == "error"
