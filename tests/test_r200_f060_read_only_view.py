"""Provider and filesystem contracts for the opt-in memo review."""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
from _stateframe_home import isolated_stateframe_home
from _timeouts import scaled
from tools.pinned_tree import pinned_tree

from stateframe.scripted_endpoint import Script, make_handler

ROOT = Path(__file__).resolve().parents[1]
BASE = "29765b39820fd39800fd7292f52fb5bcb81e6069"
SWITCH = "STATEFRAME_R200_F060_READ_ONLY_VIEW"
TOOL = "section_openers"
MEMO = """# Scope
The evidence covers
returning customers. The sample excludes new customers.

```md
## Ignored heading
This is fenced code.
```

## Criterion
The trial uses retention as the criterion.

# Proposal
The proposal applies to all customers.
"""
MONTH_BOUNDARY_CASES = [
    pytest.param(
        "We excluded Jan. UK applications submitted after the deadline.",
        "We excluded Jan. UK applications submitted after the deadline.",
        id="continuation",
    ),
    pytest.param("We excluded Jan. UK applications closed after the deadline.", None, id="ambiguous"),
    pytest.param(
        "The policy to date is Jan. UK applications shall be assessed under the original criteria.",
        None,
        id="policy-to-date",
    ),
    pytest.param(
        "The deadline was Jan. UK applicants can request an extension.",
        "The deadline was Jan.",
        id="deadline",
    ),
    pytest.param(
        "The application deadline was Jan. UK applicants can request an extension.",
        "The application deadline was Jan.",
        id="application-deadline",
    ),
]


def _script(steps):
    return {"model": "scripted", "steps": steps, "tokenize": {"mode": "ok"}}


@contextmanager
def _provider(steps, *, interrupt=False):
    script = Script(_script(steps))
    requests = []

    class RawHandler(make_handler(script)):
        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return super().do_POST()
            stream = self.rfile
            raw = stream.read(int(self.headers["Content-Length"]))
            requests.append(raw)
            if interrupt and len(requests) in {2, 4}:
                self._send(503, {"error": {"message": "Scripted transport interruption."}})
                return
            self.rfile = io.BytesIO(raw)
            try:
                super().do_POST()
            finally:
                self.rfile = stream

    with ThreadingHTTPServer(("127.0.0.1", 0), RawHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield script, requests, f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()
            thread.join()


@pytest.fixture
def memo_run(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "memo.md").write_text(MEMO, encoding="utf-8")
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    launcher = tmp_path / "launch.py"
    launcher.write_text(
        "import secrets, sys\n"
        "from stateframe import cli\n"
        "original = secrets.token_hex\n"
        "def fixed(size=None):\n"
        "    if size == 16:\n"
        "        secrets.token_hex = original\n"
        "        return 'a' * 32\n"
        "    return original(size)\n"
        "secrets.token_hex = fixed\n"
        "raise SystemExit(cli.main(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    run = tmp_path / "run"

    def cli(checkout, value, endpoint, *args, allocator="on"):
        env = {key: val for key, val in os.environ.items() if not key.startswith("STATEFRAME_")}
        env.update(PYTHONPATH=str(checkout / "src"), STATEFRAME_HOME=isolated_stateframe_home(),
                   STATEFRAME_PROVIDER="llama.cpp", STATEFRAME_WORKING_SET_ALLOCATOR=allocator,
                   STATEFRAME_CHECKPOINT="off", STATEFRAME_TRANSPORT_MAX_ATTEMPTS="1")
        if value is not None:
            env[SWITCH] = value
        return subprocess.run(
            [sys.executable, str(launcher), *args, "--endpoint", endpoint, "--quiet"],
            env=env, capture_output=True, text=True, timeout=scaled(240), check=False,
        )

    return repo, run, cli


def _run_args(repo, run, turns):
    return ("run", "--repo", str(repo), "--run-dir", str(run), "--model", "scripted",
            "--task", "Review the finished memo.md for the first unsupported transition.",
            "--test-python", sys.executable, "--no-run-env", "--shell", "off", "--no-git-tool",
            "--max-turns", str(turns))


def _call(tool, **args):
    return {"tool": tool, "arguments": args}


def _diagnostic(result, run):
    return result.stdout + result.stderr + "".join(
        log.read_text()[-8000:] for name in ("engine.log", "child-traceback.txt")
        if (log := run / name).is_file()
    )


@pytest.mark.host_sandbox
@pytest.mark.parametrize("allocator", ["on", "off"])
def test_f060_read_only_view_off_request_bytes_match_pinned_099(
        tmp_path, tmp_path_factory, memo_run, allocator):
    base = pinned_tree(BASE, tmp_path_factory)
    repo, run, cli = memo_run
    steps = [_call("read", path="memo.md", line=1), _call("read", path="memo.md", line=5)]
    with _provider(steps, interrupt=True) as (script, requests, endpoint):
        traces = []
        for label, checkout, value in (("base", base, None), ("unset", ROOT, None),
                                        ("false", ROOT, "false")):
            first = cli(checkout, value, endpoint, *_run_args(repo, run, 4), allocator=allocator)
            assert len(requests) == 2, _diagnostic(first, run)
            second = cli(checkout, value, endpoint, "resume", str(run),
                         allocator=allocator)
            assert len(requests) == 4, _diagnostic(second, run)
            replay = json.loads((run / "spec.json").read_bytes())["resume"]
            assert len(replay["tape"]) == 1 and not replay.get("restart_live")
            assert not (run / "resume-divergence.json").exists()
            assert requests[1] == requests[2]
            script.assert_consumed()
            assert all(TOOL.encode() not in raw for raw in requests)
            traces.append(tuple(requests))
            run.rename(tmp_path / label)
            requests.clear()
            script.__init__(_script(steps))
    for current in traces[1:]:
        assert current == traces[0]


def test_f060_read_only_view_switch_defaults_off(monkeypatch):
    from stateframe import runner

    monkeypatch.delenv(SWITCH, raising=False)
    assert SWITCH in runner.OPT_IN_SWITCHES
    assert SWITCH not in runner.resolve_opt_in_switches()
    for value, expected in (("on", "on"), ("off", "off"), ("false", "off")):
        monkeypatch.setenv(SWITCH, value)
        assert runner.resolve_opt_in_switches()[SWITCH] == expected


def test_f060_release_provenance_matches_current_manifest():
    engine = ROOT / "src" / "stateframe" / "engine"
    manifest_digest = hashlib.sha256((engine / "release-manifest.json").read_bytes()).hexdigest()
    provenance = json.loads((engine / "ENGINE_SOURCE.json").read_bytes())

    assert provenance["files"]["release-manifest.json"] == manifest_digest

    # The top-level identifiers describe the engine's source release.
    source_tree = provenance["source_tree"]
    assert re.fullmatch(r"[0-9a-f]{64}", source_tree)
    assert provenance["release_id"] == provenance["release_manifest"] == source_tree


def test_f060_marks_empty_ambiguous_and_omitted_sections(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = "# Parent\n## Child\nDr. Lee studied returning customers.\n# End\nEnrollment expanded."
    memo = tmp_path / "memo.md"
    memo.write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))
    assert [row["status"] for row in view["sections"]] == ["empty", "ambiguous", "quoted"]
    assert [row["section_end_line"] for row in view["sections"]] == [1, 3, 5]
    assert not view["coverage"]["complete"]
    assert "before identifying the first" in view["review"]
    assert "repair" not in view["review"].lower()

    extra = "".join(f"\n## Section {index}\nEvidence covers segment {index}.\n" for index in range(100))
    memo.write_text(source + extra, encoding="utf-8")
    result = _section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT)
    bounded = json.loads(result)
    assert len(result.encode()) <= _OPENER_OUTPUT_LIMIT
    assert bounded["coverage"]["sections"] == 103
    assert bounded["coverage"]["shown"] == len(bounded["sections"])
    assert bounded["coverage"]["omitted"] == 103 - len(bounded["sections"]) > 0
    assert not bounded["coverage"]["complete"]
    assert bounded["sha256"] == hashlib.sha256((source + extra).encode()).hexdigest()


def test_f060_refuses_unsafe_or_unsupported_sources(tmp_path):
    from kernel_agent_tools import (
        _OPENER_INPUT_LIMIT,
        _OPENER_OUTPUT_LIMIT,
        _section_opener_view,
    )

    folder = tmp_path / "folder"
    folder.mkdir()
    memo = folder / "memo.md"
    memo.write_text(MEMO, encoding="utf-8")
    (tmp_path / "link").symlink_to(folder, target_is_directory=True)
    (tmp_path / "link.md").symlink_to(memo)
    os.link(memo, tmp_path / "hard.md")
    os.mkfifo(tmp_path / "pipe.md")
    (tmp_path / "large.md").write_bytes(b"x" * (_OPENER_INPUT_LIMIT + 1))
    (tmp_path / "binary.md").write_bytes(b"\xff")
    for path in (str(memo), "../memo.md", "link/memo.md", "link.md", "hard.md", "pipe.md",
                 "large.md", "binary.md", "folder"):
        assert _section_opener_view(tmp_path, path, _OPENER_OUTPUT_LIMIT).startswith("REFUSED MEMO_SOURCE:")


def test_f060_declines_view_when_latest_request_has_no_spare_capacity(tmp_path):
    from kernel_agent_tools import _section_opener_limit, _section_opener_view
    from kernel_pilot import ACTIVE_RUNTIME

    measurement = SimpleNamespace(prompt_tokens=32_900, max_prompt_tokens=33_344,
                                  request_bytes=30_000, max_request_bytes=100_000)
    runtime = SimpleNamespace(frame_ceiling=28_000, _active_capture=SimpleNamespace(
        qualified_request=SimpleNamespace(measurement=measurement)))
    token = ACTIVE_RUNTIME.set(runtime)
    try:
        limit = _section_opener_limit({"chat_url": "http://localhost/v1/chat/completions"})
    finally:
        ACTIVE_RUNTIME.reset(token)
    assert limit == 0
    assert _section_opener_view(tmp_path, "memo.md", limit).startswith("REFUSED MEMO_VIEW_CAPACITY:")


@pytest.mark.host_sandbox
def test_f060_read_only_view_on_path_is_bounded_and_idempotent(memo_run):
    repo, run, cli = memo_run
    steps = [_call(TOOL, path="memo.md"), _call(TOOL, path="memo.md"),
             _call("read", path="memo.md", line=1)]
    with _provider(steps) as (script, requests, endpoint):
        result = cli(ROOT, "on", endpoint, *_run_args(repo, run, 3))
        assert len(requests) >= 3, _diagnostic(result, run)
        script.assert_consumed()
        payloads = [json.loads(raw) for raw in requests[:3]]

    for payload in payloads:
        assert sum(tool["function"]["name"] == TOOL for tool in payload["tools"]) == 1
    assert b'"coverage"' not in requests[0]
    replies = []
    for payload in payloads[1:]:
        results = [json.loads(msg["content"]) for msg in payload["messages"]
                   if msg["role"] == "tool" and '"coverage":' in msg["content"]]
        assert len(results) == 1, payload["messages"]
        replies.extend(results)
    assert replies[0] == replies[1]
    view = replies[0]
    assert view["sha256"] == hashlib.sha256(MEMO.encode()).hexdigest()
    assert view["coverage"] == {"complete": True, "sections": 3, "shown": 3, "omitted": 0,
                                "empty": 0, "ambiguous": 0, "unsupported_structure": False}
    assert [(row["sentence"], row["source_lines"]) for row in view["sections"]] == [
        ("The evidence covers returning customers.", [2, 3]),
        ("The trial uses retention as the criterion.", [11, 11]),
        ("The proposal applies to all customers.", [14, 14]),
    ]
    assert "first verified gap" in view["review"] and "native read" in view["review"]
    assert (repo / "memo.md").read_text(encoding="utf-8") == MEMO


@pytest.mark.host_sandbox
@pytest.mark.parametrize(("opener", "sentence"), MONTH_BOUNDARY_CASES)
def test_f060_month_boundary_preserves_evidence_on_next_request(memo_run, opener, sentence):
    repo, run, cli = memo_run
    source = f"# Criterion\nThe deadline was January 31.\n\n# Population\n{opener}\n"
    (repo / "memo.md").write_text(source, encoding="utf-8")
    steps = [_call(TOOL, path="memo.md"), _call("read", path="memo.md", line=5)]
    with _provider(steps) as (script, requests, endpoint):
        result = cli(ROOT, "on", endpoint, *_run_args(repo, run, 2))
        assert len(requests) >= 2, _diagnostic(result, run)
        script.assert_consumed()

    payload = json.loads(requests[1])
    (view,) = [json.loads(msg["content"]) for msg in payload["messages"]
               if msg["role"] == "tool" and '"coverage":' in msg["content"]]
    assert view["sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert [row["heading"] for row in view["sections"]] == ["Criterion", "Population"]
    assert view["sections"][0]["source_lines"] == [2, 2]
    row = view["sections"][1]
    assert (row["heading_line"], row["section_end_line"]) == (4, 5)
    assert row["status"] == ("ambiguous" if sentence is None else "quoted")
    if sentence is not None:
        assert row["sentence"] == sentence
        assert row["source_lines"] == [5, 5]
        assert "first verified gap" in view["review"]
    else:
        assert "sentence" not in row
        assert "Incomplete view. Use native read" in view["review"]
    assert view["coverage"] == {
        "complete": sentence is not None, "sections": 2, "shown": 2, "omitted": 0,
        "empty": 0, "ambiguous": int(sentence is None), "unsupported_structure": False,
    }


def test_f060_month_abbreviations_do_not_truncate_the_opening_sentence(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = "# Memo\nWe sampled customers in Jan. and Feb. before broadening eligibility.\n"
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert view["sections"][0]["sentence"] == (
        "We sampled customers in Jan. and Feb. before broadening eligibility."
    )
    assert view["coverage"]["complete"]


def test_f060_month_abbreviation_before_acronym_does_not_truncate_claim(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = "# Memo\nWe compared the Jan. UK cohort with returning customers.\n"
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert view["sections"][0]["sentence"] == (
        "We compared the Jan. UK cohort with returning customers."
    )
    assert view["coverage"]["complete"]


def test_f060_month_abbreviation_before_new_sentence_is_a_boundary(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = "# Memo\nThe pilot ended in Jan. A separate test enrolled new customers.\n"
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert view["sections"][0]["sentence"] == "The pilot ended in Jan."
    assert view["sections"][0]["source_lines"] == [2, 2]
    assert view["coverage"]["complete"]


def test_f060_month_abbreviation_before_acronym_subject_verb_keeps_sentence_boundary(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = "# Memo\nThe pilot ended in Jan. UK enrollment opened in February.\n"
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert view["sections"][0]["sentence"] == "The pilot ended in Jan."
    assert view["sections"][0]["source_lines"] == [2, 2]
    assert view["coverage"]["complete"]


@pytest.mark.parametrize(
    ("opener", "sentence"),
    [
        *MONTH_BOUNDARY_CASES,
        pytest.param(
            "The date is Jan. UK applications shall be assessed under the original criteria.",
            "The date is Jan.",
            id="calendar-subject",
        ),
        pytest.param("The focus was Jan. UK waste can designs.", None, id="compound-noun"),
    ],
)
def test_f060_abbreviated_month_keeps_sentence_boundary(tmp_path, opener, sentence):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    (tmp_path / "memo.md").write_text(f"# Deadline\n{opener}\n", encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    row = view["sections"][0]
    assert view["coverage"]["complete"] == (sentence is not None)
    if sentence is None:
        assert row["status"] == "ambiguous"
        assert "sentence" not in row
    else:
        assert row["status"] == "quoted"
        assert row["sentence"] == sentence
        assert row["source_lines"] == [2, 2]


def test_f060_month_acronym_continuation_allows_wrapped_lines(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = "# Memo\nWe compared the Jan. UK\ncohort with returning customers.\n"
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert view["sections"][0]["sentence"] == (
        "We compared the Jan. UK cohort with returning customers."
    )
    assert view["sections"][0]["source_lines"] == [2, 3]
    assert view["coverage"]["complete"]


def test_f060_list_item_fences_do_not_create_sections(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = (
        "# Memo\n"
        "The memo includes two examples.\n\n"
        "- ```md\n"
        "  ## Hidden in first example\n"
        "  This is fenced content.\n"
        "  ```\n"
        "- ```md\n"
        "  ## Hidden in second example\n"
        "  This is also fenced content.\n"
        "  ```\n"
    )
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert [row["heading"] for row in view["sections"]] == ["Memo"]
    assert view["coverage"]["unsupported_structure"]
    assert not view["coverage"]["complete"]


def test_f060_two_digit_list_fence_closes_before_following_top_level_section(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = (
        "# Memo\n"
        "The memo includes a numbered example.\n\n"
        "10. ```md\n"
        "    ## Hidden example section\n"
        "    This is fenced content.\n"
        "    ```\n\n"
        "## Proposal\n"
        "The proposal applies to returning customers.\n"
    )
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert [row["heading"] for row in view["sections"]] == ["Memo", "Proposal"]
    assert view["coverage"]["sections"] == 2
    assert view["coverage"]["omitted"] == 0
    assert view["coverage"]["unsupported_structure"]
    assert not view["coverage"]["complete"]


def test_f060_table_without_leading_pipes_is_not_mistaken_for_opening_prose(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = (
        "# Findings\n"
        "Population | Summary\n"
        "---------- | -------------------\n"
        "Returning customers. | Eligibility expanded.\n"
        "\n"
        "The review covered new customers after eligibility expanded.\n"
    )
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    view = json.loads(_section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT))

    assert view["sections"][0]["sentence"] == (
        "The review covered new customers after eligibility expanded."
    )
    assert view["sections"][0]["source_lines"] == [6, 6]
    assert view["coverage"]["unsupported_structure"]
    assert not view["coverage"]["complete"]


def test_f060_twenty_one_sections_remain_a_bounded_json_view(tmp_path):
    from kernel_agent_tools import _OPENER_OUTPUT_LIMIT, _section_opener_view

    source = "".join(
        f"## Section {index:02d}\n"
        f"We expanded cohort eligibility for group {index:02d}.\n\n"
        for index in range(1, 22)
    )
    (tmp_path / "memo.md").write_text(source, encoding="utf-8")
    result = _section_opener_view(tmp_path, "memo.md", _OPENER_OUTPUT_LIMIT)

    assert len(result.encode("utf-8")) <= _OPENER_OUTPUT_LIMIT
    view = json.loads(result)
    assert view["sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert view["coverage"]["sections"] == 21
    assert view["coverage"]["shown"] == len(view["sections"])
    assert view["coverage"]["omitted"] == 21 - len(view["sections"])
    assert view["coverage"]["omitted"] > 0
    assert not view["coverage"]["complete"]
    assert "Incomplete view" in view["review"]
