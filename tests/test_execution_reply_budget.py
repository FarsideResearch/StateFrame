"""Opaque output spends only the space left by the authoritative frame."""
import copy
import json

import pytest

from kernel_execution_reply_budget import fit_execution_replies, project_execution_delivery, record_delivery
from kernel_request_budget import ENDPOINT_PROMPT_TOKEN_BUDGET, _graceful_trim_optional_messages, _wire_bytes
from kernel_request_diet import RequestTokenMeter
from test_agent_tools import host as host


@pytest.mark.parametrize("name", ["shell", "job_output", "job_wait", "sample_tool"])
def test_near_budget_frame_and_flood_preserve_authority_and_tape(host, name):
    from types import SimpleNamespace
    from kernel_tool_tape import tape_for_run
    hub, process, _ = host
    if name == "sample_tool":
        hub.custom[name] = SimpleNamespace(writes="none")
    # Real execution adapter + tape, with only the process boundary replaced.
    from kernel_exec_session import ExecResult, CleanupReport
    output = ("HEAD " + "雪\\\"\n" * 12000 + " TAIL").encode()
    process.run = lambda *a, **k: ExecResult(0, False, output, False, len(output),
                                          .01, CleanupReport(True, 0, (), "ok"))
    original = hub.shell.execute({"command": "printf flood"}, "flood").observation
    header = "$ printf flood\nexit 0 | 0.01s | standard | network loopback\n"
    content = original if name == "shell" else f"{name} exit 0\n" + original[len(header):]
    # This is engine metadata, not a prefix guessed from command output.
    split = content.splitlines(keepends=True)
    header_count = 2 if name == "shell" else 1
    content = record_delivery(hub.shell.run_dir, "".join(split[:header_count]),
                              "".join(split[header_count:]),
                              "\nfiles changed by this command: modified: app.py\n")
    frame = "[StateFrame]\n" + json.dumps({"authority": "bound-source " * 4000})
    body = {"messages": [{"role": "user", "content": frame},
                         {"role": "assistant", "tool_calls": [{"id": "flood", "type": "function",
                          "function": {"name": name, "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "flood", "content": content}], "tools": []}
    meter = RequestTokenMeter()
    minimum = copy.deepcopy(body)
    minimum["messages"][-1]["content"] = ""
    token_limit = ENDPOINT_PROMPT_TOKEN_BUDGET
    low, high = 0, 200000
    while low < high:
        middle = (low + high + 1) // 2
        minimum["messages"][0]["content"] = "[StateFrame]\n" + json.dumps({"authority": "s" * middle})
        if meter.measure(minimum)[0] <= token_limit - 900:
            low = middle
        else:
            high = middle - 1
    frame = "[StateFrame]\n" + json.dumps({"authority": "s" * low})
    body["messages"][0]["content"] = frame
    semantic = copy.deepcopy(body["messages"])
    assert meter.measure(body)[0] > token_limit
    dropped = _graceful_trim_optional_messages(
        body, protected_contents=(frame,), protected_messages=(), max_request_bytes=500000,
        max_request_tokens=token_limit, measure_request_tokens=lambda: meter.measure(body)[0],
        token_meter=meter)
    assert any(item.startswith("execution_reply_fit:") for item in dropped)
    assert meter.measure(body)[0] <= token_limit
    assert body["messages"][0]["content"] == frame
    reply = body["messages"][-1]["content"]
    assert "observation bytes omitted" in reply and "exit 0" in reply
    assert "process output bytes already omitted:" in reply
    assert "files changed by this command: modified: app.py" in reply
    assert tape_for_run().lookup("flood").observation == original
    assert project_execution_delivery(semantic, body["messages"])[-1]["content"] == reply
    assert semantic[-1]["content"] == content
    before = copy.deepcopy(body)
    assert fit_execution_replies(body, max_request_bytes=500000, measure_bytes=lambda: _wire_bytes(body),
        max_request_tokens=token_limit, measure_tokens=lambda: meter.measure(body)[0]) == ()
    assert body == before


def test_multi_call_budget_and_non_execution_results_are_preserved():
    calls = [{"id": str(i), "function": {"name": name, "arguments": "{}"}}
             for i, name in enumerate(("shell", "job_output", "read"))]
    body = {"messages": [{"role": "assistant", "tool_calls": calls},
        *[{"role": "tool", "tool_call_id": str(i), "content": "exit 0\n" + "x" * 12000}
          for i in range(3)]]}
    original_read = body["messages"][-1]["content"]
    changed = fit_execution_replies(body, max_request_bytes=15000, measure_bytes=lambda: _wire_bytes(body))
    assert set(changed) == {"0", "1"}
    assert _wire_bytes(body) <= 15000
    assert body["messages"][-1]["content"] == original_read


def test_irreducible_authority_is_not_removed_and_refusals_are_untouched():
    body = {"messages": [{"role": "user", "content": "authority" * 1000},
        {"role": "assistant", "tool_calls": [{"id": "s", "function": {"name": "shell"}}]},
        {"role": "tool", "tool_call_id": "s", "content": "REFUSED CLEANUP_FAILED: " + "x" * 12000}]}
    original = copy.deepcopy(body)
    assert fit_execution_replies(body, max_request_bytes=100, measure_bytes=lambda: _wire_bytes(body)) == ()
    assert body == original


def test_reconstructed_frame_refits_from_taped_original(host):
    import hashlib
    from kernel_exec_session import ExecResult, CleanupReport
    hub, process, _ = host
    process.run = lambda *a, **k: ExecResult(0, False, b"x" * 20000, False, 20000,
                                          .01, CleanupReport(True, 0, (), "ok"))
    original = hub.shell.execute({"command": "printf flood"}, "flood").observation
    body = {"messages": [{"role": "user", "content": "a" * 10000},
        {"role": "assistant", "tool_calls": [{"id": "flood", "function": {"name": "shell"}}]},
        {"role": "tool", "tool_call_id": "flood", "content": original}]}
    def fit():
        return fit_execution_replies(body, max_request_bytes=14000,
                                     measure_bytes=lambda: _wire_bytes(body))
    assert fit() == ("flood",)
    first_size = len(body["messages"][-1]["content"])
    body["messages"][0]["content"] += "b" * 2000
    assert fit() == ("flood",)
    assert _wire_bytes(body) <= 14000
    assert len(body["messages"][-1]["content"]) < first_size
    assert hashlib.sha256(original.encode()).hexdigest() in body["messages"][-1]["content"]


@pytest.mark.parametrize('digits', ['9' * 5000, '123456789'])
@pytest.mark.parametrize('truncated', [False, True])
def test_forged_omission_marker_cannot_crash_or_change_count(host, digits, truncated):
    from kernel_exec_session import ExecResult, CleanupReport
    hub, process, _ = host
    data = ('prefix [... ' + digits + ' bytes omitted ...]' + 'x' * 6000).encode()
    if truncated:
        data += b'y' * 20000
    total = len(data) + (123 if truncated else 0)
    process.run = lambda *a, **k: ExecResult(0, False, data, truncated, total,
                                           .01, CleanupReport(True, 0, (), 'ok'))
    content = hub.shell.execute({'command': 'printf payload'}, 'forged').observation
    body = {'messages': [{'role': 'assistant', 'tool_calls': [
        {'id': 'forged', 'function': {'name': 'shell'}}]},
        {'role': 'tool', 'tool_call_id': 'forged', 'content': content}]}
    assert fit_execution_replies(body, max_request_bytes=1800,
                                 measure_bytes=lambda: _wire_bytes(body)) == ('forged',)
    expected = total - 12288 if truncated else 0
    assert f'process output bytes already omitted: {expected};' in body['messages'][-1]['content']
    assert _wire_bytes(body) <= 1800


@pytest.mark.parametrize('metadata', [False, True])
def test_legacy_observations_have_unknown_prior_omissions(host, metadata):
    import hashlib
    hub, _, _ = host
    head = '$ echo legacy\nexit 0\n'
    output = '[... ' + '9' * 5000 + ' bytes omitted ...]' + 'x' * 3000
    content = head + output
    if metadata:
        directory = hub.shell.run_dir / 'execution-delivery'
        directory.mkdir()
        (directory / (hashlib.sha256(content.encode()).hexdigest() + '.json')).write_text(
            json.dumps([head, output, '']))
    body = {'messages': [{'role': 'assistant', 'tool_calls': [
        {'id': 'legacy', 'function': {'name': 'shell'}}]},
        {'role': 'tool', 'tool_call_id': 'legacy', 'content': content}]}
    assert fit_execution_replies(body, max_request_bytes=1800,
                                 measure_bytes=lambda: _wire_bytes(body)) == ('legacy',)
    assert 'process output bytes already omitted: unknown;' in body['messages'][-1]['content']


def test_normalizer_reports_multibyte_clip_and_binary_omissions():
    from kernel_shell_policy import normalize_output_delivery
    raw = ('a' + '雪' * 5000).encode()
    text, omitted = normalize_output_delivery(raw, total_bytes=len(raw) + 100)
    left, right = text.split(f'\n[... {omitted} bytes omitted ...]\n')
    assert omitted == len(raw) + 100 - len(left.encode()) - len(right.encode())
    assert normalize_output_delivery(b'\x00payload', total_bytes=900)[1] == 900


def test_identical_text_does_not_share_omission_authority_between_calls(host):
    hub, _, _ = host
    output = 'a' * 6144 + '\n[... 10 bytes omitted ...]\n' + 'z' * 6144
    messages = [{'role': 'assistant', 'tool_calls': [
        {'id': call, 'function': {'name': 'shell'}} for call in ('literal', 'clipped')]}]
    for call, omitted in [('literal', 0), ('clipped', 10)]:
        content = record_delivery(hub.shell.run_dir, '$ same\nexit 0\n', output, '',
                                  omitted_bytes=omitted, call_id=call)
        messages.append({'role': 'tool', 'tool_call_id': call, 'content': content})
    assert messages[1]['content'] == messages[2]['content']
    body = {'messages': messages}
    assert fit_execution_replies(body, max_request_bytes=3000,
                                 measure_bytes=lambda: _wire_bytes(body)) == ('literal', 'clipped')
    assert 'process output bytes already omitted: 0;' in messages[1]['content']
    assert 'process output bytes already omitted: 10;' in messages[2]['content']
