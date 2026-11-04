import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events, measurement_journal_paths
from acceptance._support import budget
from test_shell_e2e import command, environment, fixture_repo, run_script, step
from kernel_completion_owner import REFUSAL_REASON_CODES
from probes.react_repair import _terminal_block_skips_final_grading
from stateframe.scripted_endpoint import Script, serve


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox
REFUSAL_MARKER = re.compile(
    r"\b(?:REJECTED|(?:HOST_)?REFUSED[A-Z0-9_-]*|NOT[\s_]+EXECUTED|NO TOOL WAS EXECUTED)\b"
    r"|^\s*\[continuation\]",
    re.IGNORECASE | re.MULTILINE,
)
NAMED_BACKTICK_ACTION = re.compile(
    r"\b(?:use|try|call|retry\s+with)\s+(?:the\s+)?"
    r"`([a-z_][a-z0-9_]*)`",
    re.IGNORECASE,
)
NAMED_BARE_ACTION = re.compile(
    r"\b(?:use|try|call|retry\s+with)\s+"
    r"(?!the\b|a\b|an\b)([a-z_][a-z0-9_]*)",
    re.IGNORECASE,
)
TOOL_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")
NEXT_ACTION_FIELDS = (
    "alternatives", "repair_action", "resolver", "next_action",
    "next_actions", "suggested_action", "suggested_tools",
)
PLAIN_TEXT_REFUSAL_PRODUCERS = {
    "[continuation] Dispatch refused: stale binding":
        ("kernel_continuation_runtime.py", "[continuation] Dispatch refused:",
         "Continuation authorization returns a plain-text dispatch veto."),
    "HOST_REFUSED: host could not run tests":
        ("probes/react_repair.py", "HOST_REFUSED: host could not run tests",
         "The test host reports a refused subprocess as plain text."),
    "[selector] REFUSED_PREDISPATCH: invalid selector":
        ("probes/react_repair.py", "[selector] REFUSED_PREDISPATCH",
         "Selector validation refuses before dispatch in plain text."),
    "No tool was executed. Return a valid tool call.":
        ("probes/react_repair.py", "No tool was executed.",
         "The no-tool-call branch has no structured tool receipt."),
}


@dataclass(frozen=True)
class RefusalReply:
    reply_request: str
    offered_request: str | None
    tool_call_id: str | None
    offered_tool: str | None
    offered_menu: frozenset[str]
    actions: tuple[str, ...]
    menu: frozenset[str]
    text: str
    content: object


def _reply_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_reply_text(item) for item in value)
    if isinstance(value, dict):
        return "\n".join(
            _reply_text(value[key])
            for key in ("text", "content", "message", "detail", "error")
            if key in value
        )
    return ""


def _structured_objects(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _structured_objects(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _structured_objects(nested)
    elif isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return
        if isinstance(decoded, (dict, list)):
            yield from _structured_objects(decoded)


def _is_refusal(content):
    for item in _structured_objects(content):
        for field in ("kind", "status", "outcome", "result", "result_class"):
            value = item.get(field)
            if isinstance(value, str) and (
                    value.strip().upper().startswith(("REFUSED", "REJECTED"))
                    or value.strip().upper() in {"UNKNOWN_TOOL", "UNSUPPORTED_ACTION"}
                    or value.strip().upper().replace(" ", "_") == "NOT_EXECUTED"):
                return True
        codes = [item.get("code"), item.get("reason_code")]
        reason_codes = item.get("reason_codes")
        if isinstance(reason_codes, (list, tuple)):
            codes.extend(reason_codes)
        terminal_or_action_codes = {
            "NO_CLOBBER", "UNRESOLVED_EFFECT", "CONTINUATION_BLOCKED", "HOOK_DENIED"}
        if any(isinstance(code, str) and code.strip().upper() in terminal_or_action_codes
               for code in codes):
            return True
        if (item.get("accepted") is False or item.get("admitted") is False
                or str(item.get("state", "")).upper() in {"BLOCKED", "REFUSED"}):
            if any(isinstance(code, str) and code.strip().upper() in REFUSAL_REASON_CODES
                   for code in codes):
                return True
    return bool(REFUSAL_MARKER.search(_reply_text(content)))


def _named_actions(content, known_tool_names):
    text = _reply_text(content)
    candidates = [match.group(1) for match in NAMED_BACKTICK_ACTION.finditer(text)]
    candidates.extend(match.group(1) for match in NAMED_BARE_ACTION.finditer(text))
    for item in _structured_objects(content):
        for key in NEXT_ACTION_FIELDS:
            value = item.get(key)
            values = value if isinstance(value, list) else [value]
            for candidate in values:
                if isinstance(candidate, str):
                    name = candidate
                elif isinstance(candidate, dict):
                    name = candidate.get("tool") or candidate.get("name")
                else:
                    name = None
                if isinstance(name, str):
                    candidates.append(name)
    known = {
        name.lower() for name in known_tool_names
        if isinstance(name, str) and TOOL_NAME.fullmatch(name.lower())
    }
    actions = []
    for candidate in candidates:
        name = candidate.lower()
        if TOOL_NAME.fullmatch(name) and name in known:
            actions.append(name)
    return tuple(dict.fromkeys(actions))


def _terminal_status(content):
    text = _reply_text(content).lower()
    if "terminal status" in text or "run will end with blocked status" in text:
        return True
    return any(
        any(isinstance(item.get(field), str)
            and item[field].strip().upper() == "CONTINUATION_BLOCKED"
            for field in ("kind", "code", "reason_code", "status", "result"))
        for item in _structured_objects(content)
    )


def _tool_menu(request):
    names = set()
    for tool in request.get("tools", ()):
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


def _tool_calls(message):
    calls = message.get("tool_calls", ())
    for call in calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        call_id = call.get("id") or call.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            yield call_id, name if isinstance(name, str) else None


def _known_tool_names(request_rows):
    names = set()
    for request_path, request in request_rows:
        names.update(_tool_menu(request))
        for message in request.get("messages", ()):
            if not isinstance(message, dict):
                continue
            names.update(name for _, name in _tool_calls(message) if name)
            if message.get("role") == "tool":
                name = message.get("name") or message.get("tool_name")
                if isinstance(name, str) and name:
                    names.add(name)
        response_path = request_path.with_name(
            request_path.name.removesuffix(".request.json") + ".response.json"
        )
        if not response_path.is_file():
            continue
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in _structured_objects(response):
            names.update(name for _, name in _tool_calls(item) if name)
            legacy_call = item.get("function_call")
            if isinstance(legacy_call, dict):
                name = legacy_call.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
    return frozenset(names)


def recorded_refusal_replies(run_dir):
    """Pair each host reply with its call ID, offering request, and next menu."""
    request_paths = sorted(
        (Path(run_dir) / "wire").glob("[0-9][0-9][0-9][0-9].request.json")
    )
    request_rows = [
        (request_path, json.loads(request_path.read_text(encoding="utf-8")))
        for request_path in request_paths
    ]
    known_tool_names = _known_tool_names(request_rows)
    offers = {}
    replies = []
    for index, (request_path, request) in enumerate(request_rows):
        menu = _tool_menu(request)
        messages = request.get("messages", ())
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "assistant":
                offered_request = request_rows[index - 1][0].name if index else None
                offered_menu = _tool_menu(request_rows[index - 1][1]) if index else frozenset()
                for call_id, name in _tool_calls(message):
                    offers[call_id] = (offered_request, name, offered_menu)
                continue
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not _is_refusal(content):
                continue
            call_id = message.get("tool_call_id")
            offer = offers.get(call_id) if isinstance(call_id, str) else None
            offered_request, offered_tool, offered_menu = (
                offer if offer is not None else (None, None, frozenset()))
            replies.append(RefusalReply(
                reply_request=request_path.name,
                offered_request=offered_request,
                tool_call_id=call_id if isinstance(call_id, str) else None,
                offered_tool=offered_tool,
                offered_menu=offered_menu,
                actions=_named_actions(content, known_tool_names | menu),
                menu=menu,
                text=_reply_text(content),
                content=content,
            ))
    return replies


def recorded_refusal_violations(run_dir):
    """Return uncorrelated refusals, missing actions, and actions absent from the next menu."""
    violations = []
    result_path = Path(run_dir) / "engine-result.json"
    try:
        run_result = json.loads(result_path.read_text(encoding="utf-8"))["result"]["result"]
    except (OSError, ValueError, KeyError, TypeError):
        run_result = None
    for reply in recorded_refusal_replies(run_dir):
        detail = (reply.reply_request, reply.offered_request, reply.tool_call_id)
        if reply.offered_request is None or reply.tool_call_id is None:
            violations.append((*detail, None, "tool reply has no correlated offered call"))
        elif reply.offered_tool not in reply.offered_menu:
            violations.append((*detail, reply.offered_tool,
                               "called action was not on its offering request menu"))
        terminal_claim = _terminal_status(reply.content)
        if terminal_claim and run_result != "CONTINUATION_BLOCKED":
            violations.append((*detail, None, "terminal claim does not match run result"))
        if not reply.actions and not terminal_claim:
            violations.append((*detail, None, "missing next action or terminal status"))
        for action in reply.actions:
            if action not in reply.menu:
                violations.append((*detail, action, "named action is not on the request menu"))
    return violations


def test_refusal_terminal_claim_requires_matching_run_result(tmp_path):
    run = tmp_path / "run"
    wire = run / "wire"
    wire.mkdir(parents=True)
    (wire / "0001.request.json").write_text(json.dumps({
        "messages": [{"role": "user", "content": "Inspect."}],
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }))
    (wire / "0002.request.json").write_text(json.dumps({
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-read", "type": "function",
                "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-read",
             "content": {"code": "CONTINUATION_BLOCKED"}},
        ],
        "tools": [{"type": "function", "function": {"name": "done"}}],
    }))
    (run / "engine-result.json").write_text(json.dumps({"result": {"result": "BLOCKED"}}))
    assert recorded_refusal_violations(run) == [
        ("0002.request.json", "0001.request.json", "call-read", None,
         "terminal claim does not match run result"),
    ]
    (run / "engine-result.json").write_text(json.dumps({
        "result": {"result": "CONTINUATION_BLOCKED"}}))
    assert recorded_refusal_violations(run) == []


def test_recorded_wire_helper_follows_three_request_tool_call_chronology():
    fixture = ROOT / "tests/fixtures/refusal_live_menu"

    assert len(list((fixture / "wire").glob("*.request.json"))) == 3
    replies = recorded_refusal_replies(fixture)
    assert [
        (reply.reply_request, reply.offered_request, reply.tool_call_id, reply.offered_tool)
        for reply in replies
    ] == [
        ("0002.request.json", "0001.request.json", "call-bad", "edit"),
        ("0002.request.json", "0001.request.json", "call-good", "shell"),
    ]
    assert recorded_refusal_violations(fixture) == [
        ("0002.request.json", "0001.request.json", "call-bad", "shell",
         "named action is not on the request menu"),
    ]


def test_refusal_helper_recognizes_host_structured_markers():
    assert _is_refusal({"status": "REJECTED"})
    assert _is_refusal({"status": "REFUSED_PREDISPATCH"})
    assert _is_refusal("Tool NOT EXECUTED: edit was denied.")
    assert _is_refusal({"status": "NOT_EXECUTED"})
    assert _is_refusal("[continuation] Unresolved effect remains open.")
    assert _is_refusal({"code": "NO_CLOBBER"})


def test_refusal_helper_uses_typed_host_tool_outcome():
    assert _is_refusal({"tool_outcome": {"kind": "REFUSED", "code": "hook_denied"}})
    assert _is_refusal({"result_class": "REFUSED_PREDISPATCH"})
    assert _is_refusal({"admitted": False, "reason_code": "CAPABILITY_DENIED"})
    assert not _is_refusal({"code": "UNKNOWN", "detail": "Evidence still pending"})


def _uncovered_refusal_variants(reason_codes, result_classes):
    missing_codes = {code for code in reason_codes
                     if not _is_refusal({"accepted": False, "reason_code": code})}
    missing_classes = {code for code in result_classes
                       if not _is_refusal({"result_class": code})}
    return missing_codes, missing_classes


def _assert_refusal_coverage(reason_codes, result_classes):
    assert _uncovered_refusal_variants(reason_codes, result_classes) == (set(), set())


def test_refusal_checker_covers_source_producers_and_owner_reason_codes():
    """Keep receipt classes and refusal codes coupled to their host producers."""
    source_root = ROOT / "src/stateframe/engine"
    sources = [path.read_text(encoding="utf-8") for path in source_root.rglob("*.py")]
    # The reply classifier names receipt classes that represent refusal even
    # without a REFUSED prefix. Every source REFUSED variant is also sampled.
    reply_source = (source_root / "kernel_tool_reply.py").read_text(encoding="utf-8")
    receipt_sets = re.findall(r"if result_class in \{([^}]+)\}", reply_source)
    refusal_sets = [values for values in receipt_sets if '"REFUSED"' in values]
    assert len(refusal_sets) == 1
    refusal_classes = set(re.findall(r'"([A-Z_]+)"', refusal_sets[0]))
    refusal_classes.update(re.findall(r"\bREFUSED(?:_[A-Z0-9_]+)?\b", "\n".join(sources)))
    assert {"REFUSED_PREDISPATCH", "REFUSED_SERIES", "REFUSED_UNCHANGED_GREEN",
            "REJECTED", "UNKNOWN_TOOL", "UNSUPPORTED_ACTION"} <= refusal_classes
    assert REFUSAL_REASON_CODES
    _assert_refusal_coverage(REFUSAL_REASON_CODES, refusal_classes)


def test_refusal_coverage_guard_rejects_planted_code_and_result_class():
    with pytest.raises(AssertionError, match="REFUSED_PLANTED_POLICY"):
        _assert_refusal_coverage({*REFUSAL_REASON_CODES, "REFUSED_PLANTED_POLICY"}, set())
    with pytest.raises(AssertionError, match="DENIED_PLANTED_RESULT_CLASS"):
        _assert_refusal_coverage(REFUSAL_REASON_CODES, {"DENIED_PLANTED_RESULT_CLASS"})


def test_plain_text_host_refusal_producers_remain_recognized():
    assert len(PLAIN_TEXT_REFUSAL_PRODUCERS) == 4
    source_root = ROOT / "src/stateframe/engine"
    for reply, (path, marker, reason) in PLAIN_TEXT_REFUSAL_PRODUCERS.items():
        assert reason
        assert marker in (source_root / path).read_text(encoding="utf-8")
        assert _is_refusal(reply), reply


@pytest.mark.parametrize("outcome,continuation_blocked,skip", [
    ("continuation_blocked", False, True),
    ("blocked", True, False),
    ("blocked", False, False),
    ("hook_blocked", False, False),
    ("hook_containment", False, False),
    (None, False, False),
])
def test_terminal_block_final_grading_policy(outcome, continuation_blocked, skip):
    runtime = SimpleNamespace(system_outcome=outcome,
                              continuation=SimpleNamespace(terminal_blocked=continuation_blocked))
    assert _terminal_block_skips_final_grading(runtime) is skip


@pytest.fixture(scope="module")
def recorded_blocked_run(tmp_path_factory):
    """A scripted caller records a no-clobber refusal, then a net-zero create."""
    tmp_path = tmp_path_factory.mktemp("recorded-blocked")
    repo = fixture_repo(tmp_path)
    (repo / "pkg/value.py").write_text(
        "VALUE = 1\n\ndef keep_value():\n    return VALUE\n\n"
        "def remove_value():\n    return VALUE + 1\n")
    config = tmp_path / "hooks.json"
    config.write_text(json.dumps({"schema": "stateframe-hooks/1", "hooks": [
        {"name": "undo-create", "event": "post_tool", "tools": ["file_create"],
         "changed": True, "writes": "changed", "argv": [sys.executable, "-c",
                  "import pathlib,sys;pathlib.Path(sys.argv[1]).unlink()", "{paths}"]},
    ]}))
    script = Script({"model": "scripted", "steps": [
        step("edit", path="pkg/value.py",
             function_source="def keep_value():\n    return VALUE + 2\n"),
        step("file_create", path="pkg/generated.py", content="VALUE = 1\n"),
    ], "tokenize": {"mode": "ok"}})
    server = serve(script)
    run = tmp_path / "run"
    try:
        try:
            result = subprocess.run(
                command(repo, run, f"http://127.0.0.1:{server.server_port}/v1",
                        flags=("--hooks", str(config), "--max-turns", "4")),
                env=environment(), capture_output=True, text=True,
                timeout=budget("recorded_blocked_run"))
        except subprocess.TimeoutExpired as error:
            requests = len(list((run / "wire").glob("*.request.json")))
            pytest.fail(f"recorded blocked run exceeded measured timeout: "
                        f"{requests} provider requests in {budget('recorded_blocked_run')} seconds; "
                        f"{error}")
        assert (run / "engine-result.json").is_file(), result.stdout + result.stderr
        script.assert_consumed()
        return run
    finally:
        server.shutdown()
        server.server_close()


def test_default_policy_blocked_run_stops_before_another_request(recorded_blocked_run):
    run = recorded_blocked_run
    result = json.loads((run / "engine-result.json").read_text())["result"]
    assert result["result"] == "CONTINUATION_BLOCKED"
    assert result["behaviour"] == "not_attempted"
    assert len(list((run / "wire").glob("*.request.json"))) == 2


def test_recorded_blocked_run_uses_host_refusal_codes(recorded_blocked_run):
    run = recorded_blocked_run
    assert json.loads((run / "engine-result.json").read_text())["result"]["result"] == (
        "CONTINUATION_BLOCKED")
    assert measurement_journal_paths(run)
    events = measurement_events(run)
    receipts = [event["receipt"] for event in events if event.get("kind") == "action_receipt"]
    denied = [receipt for receipt in receipts
              if receipt.get("result_class") == "REJECTED"
              and "NO_CLOBBER" in str(receipt.get("detail", ""))]
    assert denied and all(_is_refusal(receipt) for receipt in denied)
    assert recorded_refusal_replies(run)
    assert recorded_refusal_violations(run) == []


def test_named_actions_only_accept_known_tool_names():
    known_tools = frozenset({"shell", "read"})

    assert _named_actions("try again", known_tools) == ()
    assert _named_actions("try shell", known_tools) == ("shell",)


def test_run_tool_names_include_response_calls_and_ignore_non_tool_words(tmp_path):
    wire = tmp_path / "run" / "wire"
    wire.mkdir(parents=True)
    request_path = wire / "0001.request.json"
    request_path.write_text(json.dumps({
        "messages": [{"role": "user", "content": "Inspect the package."}],
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }))
    (wire / "0001.response.json").write_text(json.dumps({
        "choices": [{"message": {"role": "assistant", "tool_calls": [
            {"id": "call-shell", "type": "function",
             "function": {"name": "shell", "arguments": "{}"}},
        ]}}],
    }))
    request_rows = [(request_path, json.loads(request_path.read_text()))]

    known_tools = _known_tool_names(request_rows)

    assert known_tools == frozenset({"read", "shell"})
    assert _named_actions("try again, then call shell", known_tools) == ("shell",)


def test_recorded_wire_helper_uses_latest_offer_for_reused_call_id(tmp_path):
    wire = tmp_path / "run" / "wire"
    wire.mkdir(parents=True)
    (wire / "0001.request.json").write_text(json.dumps({
        "messages": [{"role": "user", "content": "Edit the package."}],
        "tools": [
            {"type": "function", "function": {"name": "edit"}},
            {"type": "function", "function": {"name": "shell"}},
        ],
    }))
    (wire / "0002.request.json").write_text(json.dumps({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "call-reused", "type": "function",
                 "function": {"name": "edit", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call-reused",
             "content": "{\"status\":\"REJECTED\"}"},
        ],
        "tools": [{"type": "function", "function": {"name": "shell"}}],
    }))
    (wire / "0003.request.json").write_text(json.dumps({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "call-reused", "type": "function",
                 "function": {"name": "shell", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call-reused",
             "content": "{\"status\":\"REFUSED_POLICY\"}"},
        ],
        "tools": [{"type": "function", "function": {"name": "done"}}],
    }))

    replies = recorded_refusal_replies(tmp_path / "run")

    assert [
        (reply.reply_request, reply.offered_request, reply.tool_call_id, reply.offered_tool)
        for reply in replies
    ] == [
        ("0002.request.json", "0001.request.json", "call-reused", "edit"),
        ("0003.request.json", "0002.request.json", "call-reused", "shell"),
    ]


def test_recorded_wire_helper_checks_call_against_its_offering_request(tmp_path):
    wire = tmp_path / "run" / "wire"
    wire.mkdir(parents=True)
    (wire / "0001.request.json").write_text(json.dumps({
        "messages": [{"role": "user", "content": "Edit."}],
        "tools": [{"type": "function", "function": {"name": "edit"}}],
    }))
    (wire / "0002.request.json").write_text(json.dumps({
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-wrong", "type": "function",
                "function": {"name": "shell", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call-wrong",
             "content": "{\"status\":\"REJECTED\",\"next_action\":\"read\"}"},
        ],
        "tools": [{"type": "function", "function": {"name": "read"}}],
    }))
    assert recorded_refusal_violations(tmp_path / "run") == [
        ("0002.request.json", "0001.request.json", "call-wrong", "shell",
         "called action was not on its offering request menu"),
    ]


def test_recorded_wire_helper_flags_the_old_shell_blocker(tmp_path):
    wire = tmp_path / "run" / "wire"
    wire.mkdir(parents=True)
    (wire / "0001.request.json").write_text(json.dumps({
        "messages": [{"role": "user", "content": "Create the package."}],
        "tools": [{"type": "function", "function": {"name": "shell"}}],
    }))
    (wire / "0002.request.json").write_text(json.dumps({
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"id": "call-shell-effect", "type": "function",
                 "function": {"name": "shell", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call-shell-effect",
             "content": "[continuation] Unresolved effect effect-1 (tool=shell, "
                        "status=unknown) has no typed repair action on the live menu; "
                        "completion is BLOCKED."},
        ],
        "tools": [{"type": "function", "function": {"name": "done"}}],
    }))

    reply = recorded_refusal_replies(tmp_path / "run")[0]
    assert reply.offered_tool == "shell"
    assert recorded_refusal_violations(tmp_path / "run") == [
        ("0002.request.json", "0001.request.json", "call-shell-effect", None,
         "missing next action or terminal status"),
    ]


def test_real_no_clobber_refusal_names_action_in_rendered_next_menu(tmp_path):
    def prepare(repo):
        (repo / "pkg/value.py").write_text(
            "VALUE = 1\n\n"
            "def keep_value():\n    return VALUE\n\n"
            "def remove_value():\n    return VALUE + 1\n"
        )

    run, requests = run_script(tmp_path, [
        step("read", path="pkg/value.py"),
        step("edit", path="pkg/value.py",
             function_source="def keep_value():\n    return VALUE + 2\n"),
        step("run_tests", target="tests/test_value.py"),
        step("done", summary="Checked the refused whole-file edit."),
    ], prepare=prepare)

    assert "whole-file edit would delete existing top-level definitions" in requests
    refusals = recorded_refusal_replies(run)
    no_clobber = [
        reply for reply in refusals
        if "NO_CLOBBER" in json.dumps(reply.content).upper()
        or "whole-file edit would delete" in reply.text.lower()
    ]
    assert no_clobber
    assert any(reply.offered_tool == "edit" for reply in no_clobber)
    assert all(reply.offered_tool in reply.offered_menu for reply in no_clobber)
    assert recorded_refusal_violations(run) == []
    assert json.loads((run / "engine-result.json").read_text())["result"]["result"] == "SUBMITTED"


# Host fences that can still refuse with no reachable way out. Listed, not hidden: each entry names its owner.
# When a fence gets a live repair action (or a terminal status), delete its entry; a NEW unlisted fence fails.
KNOWN_DEAD_END_FENCES = {}
DEAD_END_FENCE_PATTERN = re.compile(r"no typed repair|no \w+ action on the live menu|nothing on the live menu")


def test_known_dead_end_fences_are_listed_and_still_present():
    found = set()
    for path in sorted((ROOT / "src/stateframe").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in DEAD_END_FENCE_PATTERN.finditer(text):
            found.add((path.relative_to(ROOT).as_posix(), match.group(0)))
    assert found <= set(KNOWN_DEAD_END_FENCES), f"new dead-end fence(s) not listed: {sorted(found - set(KNOWN_DEAD_END_FENCES))}"
    assert set(KNOWN_DEAD_END_FENCES) <= found, (
        f"listed fence(s) no longer in source, remove them from KNOWN_DEAD_END_FENCES: "
        f"{sorted(set(KNOWN_DEAD_END_FENCES) - found)}")
