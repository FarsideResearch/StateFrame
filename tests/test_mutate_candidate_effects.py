"""Finish-time candidate apply keeps committed mutation edits.

The DoneSeal candidate is captured by the engine and never sees mutation effects, so a file a
writer mutation committed still carries its preimage there. Applying it at finish used to
silently revert the committed edit (a create survived only because the candidate lacks it).
The unit tests drive runner._keep_committed_mutation_effects against a real mutation journal;
the end-to-end tests run a scripted writer with and without --in-place.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from stateframe import runner
from stateframe.engine.kernel_mutate_records import Journal

SRC = Path(__file__).resolve().parents[1] / "src"
PRE = b"alpha\nbeta\n"
POST = PRE + b"gamma rolled forward\n"


def _sha(data: bytes | None) -> str | None:
    return None if data is None else hashlib.sha256(data).hexdigest()


def _commit(run: Path, path: str, pre: bytes | None, post: bytes | None, *, call: str = "c1",
            kind: str = "EFFECT_COMMITTED") -> Journal:
    journal = Journal(run)
    item = {"path": path, "pre_sha256": _sha(pre), "post_sha256": _sha(post),
            "pre_blob": journal.put_blob(pre) if pre is not None else None,
            "post_blob": journal.put_blob(post) if post is not None else None}
    journal.append(kind, call=call, files=[item])
    return journal


def _conflicts(run: Path) -> list[dict]:
    return [row for row in Journal(run).events() if row.get("kind") == runner.MUTATION_CANDIDATE_CONFLICT]


def test_candidate_preimage_keeps_committed_postimage(tmp_path):
    _commit(tmp_path, "notes.txt", PRE, POST)
    candidate = {"notes.txt": PRE, "reader.py": b"x\n"}
    applied, kept = runner._keep_committed_mutation_effects(tmp_path, candidate)
    assert applied == {"notes.txt": POST, "reader.py": b"x\n"}
    assert kept == {"notes.txt"}
    assert candidate["notes.txt"] == PRE  # the sealed candidate itself is not mutated
    assert not _conflicts(tmp_path)


def test_candidate_postimage_is_a_noop(tmp_path):
    _commit(tmp_path, "notes.txt", PRE, POST)
    applied, kept = runner._keep_committed_mutation_effects(tmp_path, {"notes.txt": POST})
    assert applied == {"notes.txt": POST} and kept == set()
    assert not _conflicts(tmp_path)


def test_candidate_edit_of_committed_file_is_a_typed_conflict_and_keeps_postimage(tmp_path):
    _commit(tmp_path, "notes.txt", PRE, POST)
    model = b"the model rewrote this\n"
    applied, kept = runner._keep_committed_mutation_effects(tmp_path, {"notes.txt": model})
    assert applied == {"notes.txt": POST} and kept == {"notes.txt"}
    conflicts = _conflicts(tmp_path)
    assert len(conflicts) == 1, conflicts
    assert conflicts[0]["path"] == "notes.txt"
    assert conflicts[0]["candidate_sha256"] == _sha(model)
    assert conflicts[0]["post_sha256"] == _sha(POST)


def test_candidate_deleting_committed_file_is_a_conflict(tmp_path):
    _commit(tmp_path, "notes.txt", PRE, POST)
    applied, _kept = runner._keep_committed_mutation_effects(tmp_path, {"notes.txt": None})
    assert applied == {"notes.txt": POST}
    assert [row["candidate_sha256"] for row in _conflicts(tmp_path)] == [None]


def test_committed_deletion_is_kept_over_candidate_preimage(tmp_path):
    _commit(tmp_path, "old.txt", PRE, None)
    applied, kept = runner._keep_committed_mutation_effects(tmp_path, {"old.txt": PRE})
    assert applied == {"old.txt": None} and kept == {"old.txt"}
    assert not _conflicts(tmp_path)


def test_committed_create_absent_from_candidate_is_untouched(tmp_path):
    _commit(tmp_path, "reports/new.md", None, POST)
    applied, kept = runner._keep_committed_mutation_effects(tmp_path, {"reader.py": b"x\n"})
    assert applied == {"reader.py": b"x\n"} and kept == set()


def test_chained_commits_accept_any_earlier_state_and_keep_the_last(tmp_path):
    middle = PRE + b"middle\n"
    _commit(tmp_path, "notes.txt", PRE, middle, call="c1")
    _commit(tmp_path, "notes.txt", middle, POST, call="c2")
    for stale in (PRE, middle):
        applied, _kept = runner._keep_committed_mutation_effects(tmp_path, {"notes.txt": stale})
        assert applied == {"notes.txt": POST}
    assert not _conflicts(tmp_path)


def test_effect_whose_last_row_is_not_committed_is_ignored(tmp_path):
    journal = _commit(tmp_path, "notes.txt", PRE, POST)
    journal.append("EFFECT_PREPARED", call="c2", files=[{"path": "notes.txt", "pre_sha256": _sha(POST),
                                                          "post_sha256": _sha(PRE), "pre_blob": _sha(POST),
                                                          "post_blob": _sha(PRE)}])
    applied, kept = runner._keep_committed_mutation_effects(tmp_path, {"notes.txt": PRE})
    assert applied == {"notes.txt": PRE} and kept == set()


def test_tampered_postimage_evidence_refuses(tmp_path):
    journal = _commit(tmp_path, "notes.txt", PRE, POST)
    blob = journal.evidence / _sha(POST)
    os.chmod(blob, 0o600)
    blob.write_bytes(b"tampered\n")
    with pytest.raises(runner.RunError, match="postimage evidence is invalid"):
        runner._keep_committed_mutation_effects(tmp_path, {"notes.txt": PRE})


def test_no_mutation_journal_is_inert_and_creates_nothing(tmp_path):
    candidate = {"notes.txt": PRE}
    applied, kept = runner._keep_committed_mutation_effects(tmp_path, candidate)
    assert applied is candidate and kept == set()
    assert not (tmp_path / "mutations").exists()
    assert list(tmp_path.iterdir()) == []


# --- End to end: a scripted writer replaces an existing file, then the run calls done. -----------

WRITER_MANIFEST = {
    "schema": "stateframe-mutation-unit/1", "name": "notes_writer",
    "revision": 1, "parent": None, "entry": "main.py", "summary": "Append a line to notes.txt",
    "kind": "reader", "input_schema": {"type": "object"},
    "output_schema": {"type": "object", "properties": {"lines": {"type": "integer"}},
                      "required": ["lines"], "additionalProperties": False},
    "requires": {"network": False, "writes": True},
    "smoke": [{"input": {"mode": "smoke"}, "expect": {"lines": 0}}],
    "limits": {"wall_s": 25, "cpu_s": 5, "mem_mb": 256}}
WRITER_SOURCE = ("import hashlib\n"
                 "def handle(input, ctx):\n"
                 "    if input.get('mode') == 'smoke': return {'lines': 0}\n"
                 "    text = ctx.input_text(input['input_id'])\n"
                 "    ctx.propose_edit('replace', 'notes.txt', text + 'gamma rolled forward\\n',\n"
                 "                     base_sha256=hashlib.sha256(text.encode()).hexdigest())\n"
                 "    return {'lines': text.count('\\n') + 1}\n")


def _step(op: str, ident: str | None = None, args: dict | None = None) -> dict:
    fields: dict[str, object] = {"op": op}
    if ident is not None:
        fields["id"] = ident
    if args is not None:
        fields["args"] = json.dumps(args)
    return {"tool": "mutate", "required": True, "arguments": fields}


def _writer_run(tmp_path: Path, *extra: str) -> tuple[Path, Path, dict]:
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_report.py").write_text("def test_report(): assert True\n")
    (repo / "reader.py").write_text("def read_report(): return None\n")
    (repo / "notes.txt").write_bytes(PRE)
    (repo / "test_notes.py").write_text("from pathlib import Path\n\ndef test_notes():\n"
                                        f"    assert Path('notes.txt').read_bytes() == {POST!r}\n")
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        _step("enter", args={"goal": "Append a line to notes.txt", "needs": {"writes": True}}),
        _step("submit", args={"manifest": WRITER_MANIFEST, "files": {"main.py": WRITER_SOURCE}}),
        _step("use", "notes_writer", {"input": {"input_id": "in0"}, "inputs": ["notes.txt"]}),
        {"tool": "done", "required": True, "arguments": {"summary": "Notes written."}},
    ]})
    server = serve(script)
    home = tmp_path / "home"
    home.mkdir()
    run = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(SRC), "STATEFRAME_MUTATIONS": "authored",
                   "HOME": str(home), "STATEFRAME_HOME": str(home / ".stateframe")}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", "Append to notes.",
             "--target", "reader.py", "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
             "--model", "scripted", "--run-dir", str(run), "--quiet", "--max-turns", "20",
             "--test-python", sys.executable, "--mutation-write", "notes.txt",
             "--check", "test_notes.py", *extra],
            env=environment, capture_output=True, text=True, timeout=420)
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    committed = [row for row in Journal(run).events() if row.get("kind") == "EFFECT_COMMITTED"]
    assert len(committed) == 1, committed
    return repo, run, json.loads((run / "summary.json").read_text())


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation launches require the Linux sandbox")
def test_committed_replace_of_existing_file_survives_finish(tmp_path):
    repo, run, summary = _writer_run(tmp_path)
    assert (run / "workspace" / "notes.txt").read_bytes() == POST
    assert (repo / "notes.txt").read_bytes() == PRE  # not in place: the original is untouched
    assert summary["check_outcome"] == "pass", summary.get("check_outcome")
    assert not _conflicts(run)


@pytest.mark.host_sandbox
@pytest.mark.skipif(sys.platform != "linux", reason="mutation launches require the Linux sandbox")
def test_committed_replace_of_existing_file_survives_finish_in_place(tmp_path):
    repo, run, summary = _writer_run(tmp_path, "--in-place")
    assert (run / "workspace" / "notes.txt").read_bytes() == POST
    assert (repo / "notes.txt").read_bytes() == POST
    assert summary["check_outcome"] == "pass", summary.get("check_outcome")
    assert not _conflicts(run)
