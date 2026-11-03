import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from test_shell_e2e import run_script, step


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.host_sandbox
REFUSAL_MARKER = re.compile(
    r"\b(?:REJECTED|REFUSED[A-Z0-9_-]*|NOT[\s_]+EXECUTED)\b"
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


@dataclass(frozen=True)
class RefusalReply:
    reply_request: str
    offered_request: str | None
    tool_call_id: str | None
    offered_tool: str | None
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
    if REFUSAL_MARKER.search(_reply_text(content)):
        return True
    for item in _structured_objects(content):
        status = item.get("status")
        if isinstance(status, str):
            normalized = status.strip().upper()
            if (normalized == "REJECTED" or normalized.startswith("REFUSED")
                    or normalized.replace("_", " ") == "NOT EXECUTED"):
                return True
    return False


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
        isinstance(item.get("code"), str)
        and item["code"].strip().upper() == "CONTINUATION_BLOCKED"
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
    """Pair host refusals with offered calls and the menu for the next model turn."""
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
                for call_id, name in _tool_calls(message):
                    offers[call_id] = (offered_request, name)
                continue
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not _is_refusal(content):
                continue
            call_id = message.get("tool_call_id")
            offer = offers.get(call_id) if isinstance(call_id, str) else None
            offered_request, offered_tool = offer if offer is not None else (None, None)
            replies.append(RefusalReply(
                reply_request=request_path.name,
                offered_request=offered_request,
                tool_call_id=call_id if isinstance(call_id, str) else None,
                offered_tool=offered_tool,
                actions=_named_actions(content, known_tool_names | menu),
                menu=menu,
                text=_reply_text(content),
                content=content,
            ))
    return replies


def recorded_refusal_violations(run_dir):
    """Return uncorrelated refusals, missing actions, and actions absent from the next menu."""
    violations = []
    for reply in recorded_refusal_replies(run_dir):
        detail = (reply.reply_request, reply.offered_request, reply.tool_call_id)
        if reply.offered_request is None or reply.tool_call_id is None:
            violations.append((*detail, None, "tool reply has no correlated offered call"))
        if not reply.actions and not _terminal_status(reply.content):
            violations.append((*detail, None, "missing next action or terminal status"))
        for action in reply.actions:
            if action not in reply.menu:
                violations.append((*detail, action, "named action is not on the request menu"))
    return violations


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
    assert not _is_refusal({"code": "NO_CLOBBER"})


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
    assert recorded_refusal_violations(run) == []


# Host fences that can still refuse with no reachable way out. Listed, not hidden: each entry names its owner.
# When a fence gets a live repair action (or a terminal status), delete its entry; a NEW unlisted fence fails.
KNOWN_DEAD_END_FENCES = {
    ("src/stateframe/engine/kernel_continuation_runtime.py", "no typed repair"): (
        "unresolved effect with no repair action blocks every non-done tool before dispatch; the directory-only "
        "shell case is fixed (kernel_continuation_evidence.py), other unknown effects remain. Owner: work-queue "
        "item 'no unrecoverable continuation fence' (spec: Part B review findings 1-5)."),
}
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
