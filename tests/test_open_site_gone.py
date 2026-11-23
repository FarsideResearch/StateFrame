"""Invariant for FRAMEFIT5 after its most recent open edit site disappears."""
from __future__ import annotations

import json
import hashlib
import os
import random
import sys
from pathlib import Path

import pytest

from _stateframe_home import isolated_stateframe_home
from engine_run import run_monitored
from stateframe.scripted_endpoint import Script, serve
from kernel_file_ops import file_delete, file_move
from kernel_stateframe_action_ledger import ActionLedgerError, build_framefit_action_context


ROOT = Path(__file__).resolve().parents[1]
SITE = "app/site.py"
OTHER = "app/other.py"
STALE_LINE = "SITE_VALUE = 'before'\n"
GONE_NOTICES = tuple({"status": "UNVERIFIABLE", "reason": reason} for reason in
                     ("open_edit_site_gone", "missing_exact_current_site"))
MATRIX = (
    "deleted", "renamed", "directory", "symlink_inside", "symlink_outside",
    "emptied", "truncated", "chmod_000", "non_utf8", "blank_start", "blank_end",
    "blank_whole", "u2028", "neighbour_70kb",
)


def _step(tool: str, **arguments: object) -> dict[str, object]:
    return {"tool": tool, "required": True, "arguments": arguments}


def _read_messages(path: str, snapshot: str, start: int, end: int) -> list[dict[str, object]]:
    lines = snapshot.splitlines(keepends=True)
    assert 1 <= start <= end <= len(lines)
    visible = "".join(lines[start - 1:end])
    revision = "sha256:" + hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
    call_id = "open-site-holdout-read"
    result = (
        f"[read page] path={path} revision={revision} lines={start}-{end}/{len(lines)} "
        f"chars={len(visible)} bytes={len(visible.encode('utf-8'))} "
        "clip=max_lines truncated=true\ncurrent source\n"
        f"{visible}"
    )
    return [
        {"role": "assistant", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": path})},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _source_for(case: str) -> tuple[str, int, int]:
    if case == "blank_start":
        return " \nSITE_VALUE = 'before'\n", 1, 1
    if case == "blank_end":
        return "SITE_VALUE = 'before'\n \n", 2, 2
    if case == "blank_whole":
        return "\n\n", 1, 2
    if case == "u2028":
        return "PREFIX = 1\u2028SITE_VALUE = 'before'\n", 2, 2
    if case == "neighbour_70kb":
        return "N = '" + ("x" * 70_000) + "'\n" + STALE_LINE, 2, 2
    return STALE_LINE + "NEIGHBOUR = 1\n", 1, 1


def _write(root: Path, path: str, content: bytes | str) -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return target


def _change(root: Path, case: str, *, host_root: Path) -> None:
    target = root / SITE
    if case == "deleted":
        file_delete(root, SITE, protected=lambda _path: False, editable=lambda _path: True)
    elif case == "renamed":
        file_move(root, SITE, "app/renamed.py", protected=lambda _path: False,
                  editable=lambda _path: True, new_file_allowed=lambda _path: True)
    elif case == "directory":
        target.unlink()
        target.mkdir()
    elif case == "symlink_inside":
        target.unlink()
        target.symlink_to(root / OTHER)
    elif case == "symlink_outside":
        outside = _write(host_root, "outside.py", "OUTSIDE = True\n")
        target.unlink()
        target.symlink_to(outside)
    elif case == "emptied":
        target.write_bytes(b"")
    elif case == "truncated":
        target.write_bytes(b"SITE_VALUE =")
    elif case == "chmod_000":
        if os.geteuid() == 0:
            pytest.skip("chmod 000 is unreadable only when the test is not running as root")
        target.chmod(0)
    elif case == "non_utf8":
        target.write_bytes(b"\xff\xfe\n")
    elif case in {"blank_start", "blank_end", "blank_whole"}:
        target.write_text("CHANGED = True\n", encoding="utf-8")
    elif case == "u2028":
        target.write_text("PREFIX = 2\u2028REPLACED = True\n", encoding="utf-8")
    elif case == "neighbour_70kb":
        target.write_text("N = '" + ("z" * 70_000) + "'\nREPLACED = True\n",
                          encoding="utf-8")
    else:
        raise AssertionError(f"unknown open-site mutation: {case}")


def _assert_gone_context(context: dict[str, object], *, stale_line: str) -> None:
    """A gone site is one typed notice and contributes no source excerpt to the request."""
    assert context.get("site_context") in GONE_NOTICES
    assert context.get("open_edit_site") is None
    serialized = json.dumps(context, ensure_ascii=False)
    assert stale_line not in serialized


def _build_context_or_fail(root: Path, messages: list[dict[str, object]]) -> dict[str, object]:
    try:
        return build_framefit_action_context(messages, workspace_root=root)
    except (ActionLedgerError, OSError, RuntimeError, ValueError) as error:
        pytest.fail(f"next request was not built after the open site changed: {error}")


def _scripted_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app/other.py").write_text("OTHER_VALUE = 2\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests/test_ok.py").write_text(
        "from app.other import OTHER_VALUE\n"
        "def test_ok():\n    assert OTHER_VALUE == 3\n", encoding="utf-8")
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        _step("file_create", path="app/gone.py", content="VALUE = 1\n"),
        _step("read", path="app/gone.py"),
        _step("file_delete", path="app/gone.py"),
        _step("read", path="app/other.py"),
        _step("str_replace", path="app/other.py", old_str="OTHER_VALUE = 2",
              new_str="OTHER_VALUE = 3"),
        _step("run_tests", target="tests/test_ok.py"),
        _step("done", summary="Removed the temporary file and corrected the other value."),
    ]})
    server = serve(script)
    run = tmp_path / "run"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("STATEFRAME_")}
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"),
                                        str(ROOT / "src/stateframe/engine"))),
        "STATEFRAME_HOME": isolated_stateframe_home(),
    })
    url = f"http://127.0.0.1:{server.server_port}/v1"
    return repo, script, server, run, environment, url


def _cli_command(repo: Path, run: Path, url: str, *, max_turns: int = 8,
                 keep_checkpoints: bool = False) -> list[str]:
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", "Create and inspect app/gone.py, remove it, then fix and verify app/other.py.",
        "--target", "app/other.py", "--endpoint", url, "--model", "scripted",
        "--run-dir", str(run), "--test-python", sys.executable,
        "--max-turns", str(max_turns), "--quiet",
    ]
    if keep_checkpoints:
        command.append("--keep-checkpoints")
    return command


def _scripted_cli(tmp_path: Path) -> tuple[Path, object, str, Script]:
    repo, script, server, run, environment, url = _scripted_fixture(tmp_path)
    try:
        result = run_monitored(_cli_command(repo, run, url), env=environment,
                               capture_output=True, text=True,
                               expected_seconds=240, run_dir=run)
    finally:
        server.shutdown()
        server.server_close()
    log = (run / "engine.log").read_text(errors="replace") if (run / "engine.log").exists() else ""
    return run, result, log, script


def _scripted_chat(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo, script, server, _run, environment, url = _scripted_fixture(tmp_path)
    monkeypatch.setenv("STATEFRAME_HOME", environment["STATEFRAME_HOME"])
    monkeypatch.setenv("PYTHONPATH", environment["PYTHONPATH"])
    try:
        from stateframe import chat, runner
        from stateframe.endpoint import Endpoint

        base = runner.RunOptions(
            repo=repo, task="", endpoint=Endpoint.from_args(url, "scripted", None),
            max_turns=8, test_cmd="true", checks=["tests/test_ok.py"],
            target=OTHER, python=sys.executable, test_python=sys.executable,
        )
        session = chat.Session(repo=repo, home=tmp_path / "session", base=base)
        session.start(printer=lambda _message: None)
        record = session.run_task(
            "Create and inspect app/gone.py, remove it, then fix and verify app/other.py.",
            printer=lambda _message: None,
        )
    finally:
        server.shutdown()
        server.server_close()
    return Path(record.run_dir), record, script


def _scripted_resume(tmp_path: Path):
    repo, script, server, run, environment, url = _scripted_fixture(tmp_path)
    try:
        first = run_monitored(
            _cli_command(repo, run, url, max_turns=4, keep_checkpoints=True),
            env=environment, capture_output=True, text=True, expected_seconds=240,
            run_dir=run,
        )
        assert script.position in {3, 4}, "initial attempt passed the intended recovery point"
        assert (run / "checkpoints/LATEST").is_file(), "the deletion point had no saved checkpoint"
        resume_turns = 8
        resume_command = [
            sys.executable, "-m", "stateframe", "resume", str(run),
            "--endpoint", url, "--model", "scripted", "--max-turns", str(resume_turns),
            "--keep-checkpoints", "--quiet",
        ]
        resumed = run_monitored(
            resume_command, env=environment, capture_output=True, text=True,
            expected_seconds=240, run_dir=run,
        )
    finally:
        server.shutdown()
        server.server_close()
    return run, first, resumed, script


def _wire_documents(run: Path) -> list[tuple[Path, dict[str, object]]]:
    rows = []
    for path in run.rglob("*.request.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            rows.append((path, value))
    return rows


@pytest.mark.parametrize("mode", ("cli", "chat", "resume"))
def test_deleted_open_site_does_not_block_next_read_or_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    if mode == "cli":
        run, result, log, script = _scripted_cli(tmp_path)
        exit_code = result.returncode
        diagnostic = log + result.stdout + result.stderr
    elif mode == "chat":
        run, record, script = _scripted_chat(tmp_path, monkeypatch)
        exit_code = 0 if record.runtime_outcome == "completed" else 1
        log = (run / "engine.log").read_text(errors="replace") if (run / "engine.log").exists() else ""
        diagnostic = log + "\n" + str(record.error or record.runtime_outcome)
    else:
        run, first, result, script = _scripted_resume(tmp_path)
        exit_code = result.returncode
        diagnostic = first.stdout + first.stderr + result.stdout + result.stderr

    assert exit_code == 0, diagnostic[-4000:]
    script.assert_consumed()
    rows = _wire_documents(run)
    wire_text = [(path, json.dumps(request, ensure_ascii=False)) for path, request in rows]
    later = [(path, text) for path, text in wire_text if path.name[:4].isdigit() and int(path.name[:4]) >= 4]
    notices = [path for path, text in later
               if "open_edit_site_gone" in text or "latest_file_action_has_no_exact_current_site" in text]
    assert notices, [path for path, _text in wire_text]
    assert not any("[Current source: app/gone.py]" in text for _path, text in later)
    assert any("OTHER_VALUE = 2" in text for _path, text in wire_text)
    assert any("OTHER_VALUE = 3" in text for _path, text in wire_text)
    assert (run / "workspace/app/other.py").read_text(encoding="utf-8") == "OTHER_VALUE = 3\n"


@pytest.mark.parametrize("case", MATRIX, ids=MATRIX)
def test_open_site_change_builds_a_gone_notice_then_recovers_on_another_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    # CLI, chat, and resumed runs converge on this shared request-builder seam;
    # the scripted deletion case below separately covers all three entrypoints.
    monkeypatch.delenv("STATEFRAME_FRAMEFIT5_ACTION_CONTEXT", raising=False)
    root = tmp_path / "workspace"
    source, start, end = _source_for(case)
    _write(root, SITE, source)
    _write(root, OTHER, "OTHER_VALUE = 2\n")
    visible = "".join(source.splitlines(keepends=True)[start - 1:end])
    messages = [{"role": "user", "content": "Inspect the current source."}]
    messages.extend(_read_messages(SITE, source, start, end))

    _change(root, case, host_root=tmp_path)

    context = _build_context_or_fail(root, messages)
    _assert_gone_context(context, stale_line=visible)

    other_read = [{"role": "user", "content": "Continue with the other source."}]
    other_read.extend(_read_messages(OTHER, "OTHER_VALUE = 2\n", 1, 1))
    recovered = _build_context_or_fail(root, other_read)
    site = recovered.get("open_edit_site")
    assert isinstance(site, dict)
    assert site["path"] == OTHER
    assert site["current_bytes"] == "OTHER_VALUE = 2\n"


@pytest.mark.parametrize("seed", range(20), ids=lambda seed: f"seed-{seed}")
def test_seeded_random_changes_and_valid_reads_keep_request_construction_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int,
) -> None:
    monkeypatch.delenv("STATEFRAME_FRAMEFIT5_ACTION_CONTEXT", raising=False)
    rng = random.Random(seed)
    cases = tuple(case for case in MATRIX if case != "chmod_000" or os.geteuid() != 0)
    root = tmp_path / "workspace"
    _write(root, OTHER, "OTHER_VALUE = 2\n")
    rounds = rng.randint(1, 4)

    for index in range(rounds):
        target = root / SITE
        if target.is_symlink():
            target.unlink()
        elif target.is_file():
            target.chmod(0o600)
            target.unlink()
        elif target.is_dir():
            target.rmdir()
        case = rng.choice(cases)
        source, start, end = _source_for(case)
        _write(root, SITE, source)
        visible = "".join(source.splitlines(keepends=True)[start - 1:end])
        messages = [{"role": "user", "content": f"Seed {seed}, site {index}."}]
        messages.extend(_read_messages(SITE, source, start, end))

        _change(root, case, host_root=tmp_path)
        context = _build_context_or_fail(root, messages)
        _assert_gone_context(context, stale_line=visible)

        other_text = f"OTHER_VALUE = {seed * 10 + index}\n"
        _write(root, OTHER, other_text)
        other_read = [{"role": "user", "content": "Read a valid file after the site notice."}]
        other_read.extend(_read_messages(OTHER, other_text, 1, 1))
        recovered = _build_context_or_fail(root, other_read)
        site = recovered.get("open_edit_site")
        assert isinstance(site, dict)
        assert site["path"] == OTHER
        assert site["current_bytes"] == other_text
