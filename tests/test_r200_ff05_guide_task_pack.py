"""FF05 guide-task treatments: real request assembly and ticket-bound export scoring."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.pinned_tree import pinned_tree

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "fe31666bd5e3a8d1e109c5f3cd7638ab176adcbe"
RUN_REFERENCES = ("INC-PRICE-731", "INC-PRICE-984")
_UNSET = object()
_STEPS = [
    {"tool": "read", "required": True,
     "arguments": {"path": "docs/quickstart.md"}},
    {"tool": "done", "required": True,
     "arguments": {"summary": "The handover identifies the matching run and send 2."}},
]


def _reset_fixture_repo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "docs").mkdir(exist_ok=True)
    (repo / "docs/quickstart.md").write_text(
        "# Quickstart\n\nRun the launch command.\n", encoding="utf-8")
    return hashlib.sha256((repo / "docs/quickstart.md").read_bytes()).hexdigest()


def _task(reference: str) -> str:
    return (
        f"Support ticket {reference} requests the exact send-2 request from its pricing run. "
        "Read docs/quickstart.md and report how a newcomer should locate and export that exact "
        "send from the matching one of two runs. Do not modify files."
    )


def _remove_run(path: Path) -> None:
    if not path.exists():
        return
    path.chmod(0o700)
    for directory, children, _files in os.walk(path):
        os.chmod(directory, 0o700)
        for child in children:
            child_path = Path(directory) / child
            if not child_path.is_symlink():
                child_path.chmod(0o700)
    shutil.rmtree(path)


def _environment(source_tree: Path, *, home: Path, switch: object) -> dict[str, str]:
    environment = {
        str(key): str(value) for key, value in os.environ.items()
        if not str(key).startswith("STATEFRAME_")
    }
    environment["PYTHONPATH"] = str(source_tree / "src")
    environment["STATEFRAME_HOME"] = str(home)
    if switch is not _UNSET:
        environment["STATEFRAME_GUIDE_TASK_PACK"] = str(switch)
    return environment


def _run_scripted(source_tree: Path, fixture_root: Path, index: int, reference: str,
                  *, switch: object) -> tuple[list[bytes], str]:
    from stateframe.scripted_endpoint import Script, serve

    repo = fixture_root / f"repo-{index}"
    run = fixture_root / f"run-{index}"
    home = fixture_root / "stateframe-home"
    _remove_run(run)
    source_revision = _reset_fixture_repo(repo)
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"},
                     "steps": [dict(step) for step in _STEPS]})
    server = serve(script)
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    command = [
        sys.executable, "-m", "stateframe", "run", "--repo", str(repo),
        "--task", _task(reference), "--endpoint", endpoint, "--model", "scripted",
        "--run-dir", str(run), "--test-python", sys.executable,
        "--max-turns", "8", "--quiet",
    ]
    try:
        result = subprocess.run(
            command, cwd=repo,
            env=_environment(source_tree, home=home, switch=switch),
            capture_output=True, text=True, timeout=300, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
        if switch is not _UNSET:
            spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
            assert spec["environment"].get("STATEFRAME_GUIDE_TASK_PACK") == str(switch)
        request_paths = sorted(
            (run / "wire").glob("*.request.json"),
            key=lambda path: int(path.name.split(".", 1)[0]),
        )
        assert len(request_paths) == 4, [path.name for path in request_paths]
        return [path.read_bytes() for path in request_paths], source_revision
    finally:
        server.shutdown()
        server.server_close()


def _run_pair(source_tree: Path, fixture_root: Path,
              switches: tuple[object, object]) -> tuple[dict[str, list[bytes]], dict[str, str]]:
    requests: dict[str, list[bytes]] = {}
    revisions: dict[str, str] = {}
    for index, (reference, switch) in enumerate(zip(RUN_REFERENCES, switches), start=1):
        requests[reference], revisions[reference] = _run_scripted(
            source_tree, fixture_root, index, reference, switch=switch)
    return requests, revisions


def _messages(raw: bytes) -> list[dict[str, object]]:
    document = json.loads(raw)
    assert isinstance(document, dict)
    messages = document.get("messages")
    assert isinstance(messages, list)
    return messages


def _task_content(raw: bytes) -> str:
    return "\n".join(
        str(message.get("content")) for message in _messages(raw)
        if message.get("role") == "user" and isinstance(message.get("content"), str)
    )


def _inspect_frame_export(source_tree: Path, fixture_root: Path, index: int) -> bytes:
    repo = fixture_root / f"repo-{index}"
    run = fixture_root / f"run-{index}"
    result = subprocess.run(
        [sys.executable, "-m", "stateframe", "inspect", str(run), "--send", "2", "--frame"],
        cwd=repo,
        env=_environment(source_tree, home=fixture_root / "stateframe-home", switch="both"),
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.encode("utf-8")


def test_ff05_guide_task_pack_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory):
    baseline = pinned_tree(BASE_REVISION, tmp_path_factory)
    fixture_root = tmp_path / "off-parity"
    # The pinned release predates this switch, so keep it unset there and
    # compare against both absent and explicit-false candidates.
    expected, _ = _run_pair(baseline, fixture_root, (_UNSET, _UNSET))
    actual, _ = _run_pair(ROOT, fixture_root, (_UNSET, "off"))

    assert set(actual) == set(expected) == set(RUN_REFERENCES)
    for reference in RUN_REFERENCES:
        assert len(expected[reference]) == len(actual[reference]) == 4
        # Compare every complete, provider-bound body byte-for-byte (not parsed JSON).
        assert actual[reference] == expected[reference]


def test_ff05_guide_task_pack_switch_defaults_off(monkeypatch):
    from kernel_guide_task_pack import (
        GuideTaskPackSettings, MODE_ENV, assemble_request_cues, settings,
    )

    monkeypatch.delenv(MODE_ENV, raising=False)
    assert settings() == GuideTaskPackSettings()
    from stateframe.runner import OPT_IN_SWITCHES, resolve_opt_in_switches
    assert OPT_IN_SWITCHES[MODE_ENV] == (
        "off", "neither", "ordering", "result_check", "both")
    assert MODE_ENV not in resolve_opt_in_switches()
    baseline = [{"role": "user", "content": "Task: inspect send 2 for INC-PRICE-731"}]
    prepared, uncapped = assemble_request_cues(baseline, baseline, settings())
    assert prepared is baseline and uncapped is baseline
    assert prepared[0]["content"] == "Task: inspect send 2 for INC-PRICE-731"


def test_ff05_export_identity_uses_original_task_not_attached_receipts():
    from kernel_guide_task_pack import score_export

    target, decoy = RUN_REFERENCES
    target_bytes = json.dumps({
        "model": "scripted",
        "messages": [
            {"role": "user", "content": _task(target)},
            {"role": "user", "content": f"Attached launch receipt for {decoy}."},
        ],
    }, separators=(",", ":")).encode()
    decoy_bytes = json.dumps({
        "model": "scripted",
        "messages": [{"role": "user", "content": _task(decoy)}],
    }, separators=(",", ":")).encode()
    score = score_export(
        requested_run=target, requested_send=2, exported_bytes=target_bytes,
        exported_send=2, run_sends={target: {2: target_bytes}, decoy: {2: decoy_bytes}},
    )
    assert score.accepted and score.identity_match and score.exact_bytes_match


def test_ff05_export_identity_requires_complete_incident_reference():
    from kernel_guide_task_pack import score_export

    target, decoy = "INC-PRICE-7310", "INC-PRICE-731"
    target_bytes = json.dumps({
        "model": "scripted",
        "messages": [{"role": "user", "content": _task(target)}],
    }, separators=(",", ":")).encode()
    decoy_bytes = json.dumps({
        "model": "scripted",
        "messages": [{"role": "user", "content": _task(decoy)}],
    }, separators=(",", ":")).encode()

    score = score_export(
        requested_run=target, requested_send=2, exported_bytes=target_bytes,
        exported_send=2, run_sends={target: {2: target_bytes}, decoy: {2: decoy_bytes}},
    )

    assert score.accepted and score.identity_match and score.exact_bytes_match
    assert score.observed_run == target


def test_ff05_export_identity_accepts_punctuation_and_unicode_boundaries():
    from kernel_guide_task_pack import score_export

    target = RUN_REFERENCES[0]
    scores = []
    for task in (
        "Support ticket INC-PRICE-731--please export",
        "チケットINC-PRICE-731の送信2",
    ):
        exact_request = json.dumps({
            "model": "scripted",
            "messages": [{"role": "user", "content": task}],
        }, separators=(",", ":")).encode()
        score = score_export(
            requested_run=target, requested_send=2,
            exported_bytes=exact_request, exported_send=2,
            run_sends={target: {2: exact_request}},
        )
        scores.append(score)

    assert all(score.exact_bytes_match for score in scores)
    assert all(score.accepted and score.identity_match for score in scores), scores
    assert all(score.observed_run == target for score in scores), scores


def test_ff05_export_identity_rejects_single_hyphen_reference_extension():
    from kernel_guide_task_pack import score_export

    target = RUN_REFERENCES[0]
    exact_request = json.dumps({
        "model": "scripted",
        "messages": [{
            "role": "user",
            "content": "Ticket INC-PRICE-731-ALT requests send 2.",
        }],
    }, separators=(",", ":")).encode()
    score = score_export(
        requested_run=target, requested_send=2,
        exported_bytes=exact_request, exported_send=2,
        run_sends={target: {2: exact_request}},
    )

    assert score.exact_bytes_match
    assert not score.accepted and not score.identity_match
    assert score.observed_run is None


def test_ff05_export_task_text_keeps_marker_like_user_data():
    from kernel_guide_task_pack import _request_task_text

    task = "[StateFrame] User supplied text for support ticket INC-PRICE-731."
    document = {"messages": [
        {"role": "user", "content": task},
        {"role": "user", "content": "Attached receipt for INC-PRICE-984."},
    ]}

    assert _request_task_text(document) == task


def test_ff05_frame_payload_uses_structural_frame_row():
    from kernel_guide_task_pack import _request_frame_payload

    document = {"messages": [
        {"role": "system", "content": "Rules"},
        {"role": "user", "content": "[StateFrame] User supplied task text."},
        {"role": "user", "content": '[StateFrame]\n{"frame":"actual"}'},
    ]}

    assert _request_frame_payload(document) == {"frame": "actual"}


def test_ff05_guide_task_pack_on_path_is_bounded_and_idempotent(tmp_path, monkeypatch):
    from kernel_guide_task_pack import (
        MAX_EXPORT_BYTES, MAX_RUNS, MAX_SENDS_PER_RUN, PACK_MARKER,
        GuideTaskPackSettings, assemble_request_cues, score_export, settings,
    )

    cells = {
        "neither": (False, False, "immediately after", "readable JSON"),
        "ordering": (True, False, "before the guide's first use", "readable JSON"),
        "result_check": (False, True, "immediately after", "compare it with the support ticket"),
        "both": (True, True, "before the guide's first use", "compare it with the support ticket"),
    }
    all_sends: dict[str, dict[int, bytes]] = {}
    cell_requests = {}
    for cell, (ordering, result_check, order_cue, check_cue) in cells.items():
        config = settings(cell)
        assert config == GuideTaskPackSettings(True, ordering, result_check)
        # Applying the real runtime assembler twice must leave one identical cue pack.
        seed = [{"role": "user", "content": f"Task: {_task(RUN_REFERENCES[0])}"}]
        once, _ = assemble_request_cues(seed, seed, config)
        twice, _ = assemble_request_cues(once, once, config)
        assert twice == once
        assert twice[0]["content"].count(PACK_MARKER) == 1
        assert order_cue in twice[0]["content"]
        assert check_cue in twice[0]["content"]
        assert ("If they differ" in twice[0]["content"]) is result_check

    fixture_roots = {cell: tmp_path / cell for cell in cells}
    with ThreadPoolExecutor(max_workers=len(cells)) as pool:
        futures = {
            pool.submit(_run_pair, ROOT, fixture_roots[cell], (cell, cell)): cell
            for cell in cells
        }
        cell_revisions = {}
        for future in as_completed(futures):
            cell = futures[future]
            request_pair, source_revisions = future.result()
            cell_revisions[cell] = source_revisions
            ordering, result_check, order_cue, check_cue = cells[cell]
            assert source_revisions[RUN_REFERENCES[0]] == source_revisions[RUN_REFERENCES[1]]
            for reference in RUN_REFERENCES:
                assert source_revisions[reference]
                bodies = request_pair[reference]
                assert len(bodies) == 4
                assert len(bodies) <= MAX_SENDS_PER_RUN
                for send, raw in enumerate(bodies, start=1):
                    content = _task_content(raw)
                    assert content.count(PACK_MARKER) == 1, (reference, send, content[:600])
                    assert reference in content
                    assert order_cue in content
                    assert check_cue in content
                    assert ("If they differ" in content) is result_check
                    assert ("before the guide's first use" in content) is ordering
            cell_requests[cell] = request_pair
    assert set(cell_requests) == set(cells), "all four cells must reach the scripted provider"
    source_revisions = cell_revisions["both"]
    assert all(
        revisions[RUN_REFERENCES[0]] == source_revisions[RUN_REFERENCES[0]]
        for revisions in cell_revisions.values()
    )
    for reference, bodies in cell_requests["both"].items():
        all_sends[reference] = {send: raw for send, raw in enumerate(bodies, start=1)}
    assert len(all_sends) == 2 and len(all_sends) <= MAX_RUNS

    target, decoy = RUN_REFERENCES
    target_send = cell_requests["both"][target][1]
    decoy_send = cell_requests["both"][decoy][1]
    request_sends = {reference: sends for reference, sends in all_sends.items()}
    frame_export = _inspect_frame_export(ROOT, fixture_roots["both"], 1)
    frame_document = json.loads(frame_export)
    assert isinstance(frame_document, dict) and "task" not in frame_document
    frame_score = score_export(
        requested_run=target, requested_send=2, exported_bytes=frame_export,
        exported_send=2, run_sends=request_sends,
    )
    assert frame_score.identity_match and frame_score.send_match, frame_score
    assert not frame_score.wire_format_match and not frame_score.exact_bytes_match
    assert not frame_score.accepted

    correct = score_export(
        requested_run=target, requested_send=2, exported_bytes=target_send,
        exported_send=2, run_sends=request_sends,
    )
    assert correct.accepted and correct.identity_match and correct.send_match
    assert correct.exact_bytes_match and correct.wire_format_match
    scored_cases = {(correct.requested_run, correct.requested_send)}

    # A decoy's self-consistent digest is not ticket identity or requested bytes.
    decoy_digest = hashlib.sha256(decoy_send).hexdigest()
    wrong_run = score_export(
        requested_run=target, requested_send=2, exported_bytes=decoy_send,
        exported_send=2, run_sends=request_sends,
        selected_summary_digest=decoy_digest,
    )
    assert wrong_run.wire_format_match and wrong_run.send_match
    assert wrong_run.selected_summary_digest_match
    assert not wrong_run.identity_match and not wrong_run.accepted

    reverse_ticket = score_export(
        requested_run=decoy, requested_send=2, exported_bytes=decoy_send,
        exported_send=2, run_sends=request_sends,
    )
    assert reverse_ticket.accepted and reverse_ticket.identity_match
    scored_cases.add((reverse_ticket.requested_run, reverse_ticket.requested_send))
    reverse_wrong_run = score_export(
        requested_run=decoy, requested_send=2, exported_bytes=target_send,
        exported_send=2, run_sends=request_sends,
    )
    assert not reverse_wrong_run.identity_match and not reverse_wrong_run.accepted

    target_send1 = cell_requests["both"][target][0]
    target_send1_score = score_export(
        requested_run=target, requested_send=1, exported_bytes=target_send1,
        exported_send=1, run_sends=request_sends,
    )
    assert target_send1_score.accepted and target_send1_score.send_match
    scored_cases.add((target_send1_score.requested_run, target_send1_score.requested_send))
    assert {(target, 2), (decoy, 2), (target, 1)} <= scored_cases
    wrong_send = score_export(
        requested_run=target, requested_send=2, exported_bytes=target_send1,
        exported_send=2, run_sends=request_sends,
    )
    assert target_send1 != target_send
    assert wrong_send.identity_match and wrong_send.send_match and wrong_send.wire_format_match
    assert not wrong_send.exact_bytes_match and not wrong_send.accepted

    # Package imports must score frame exports in a clean process too. Pytest's
    # conftest adds the flat engine directory to sys.path, which the runtime
    # consumer does not necessarily have.
    package_fixture = tmp_path / "package-frame-score.json"
    package_fixture.write_text(json.dumps({
        "frame": frame_export.decode("utf-8"),
        "target_send": target_send.decode("utf-8"),
        "decoy_send": decoy_send.decode("utf-8"),
    }), encoding="utf-8")
    package_probe = r"""
import json, sys
from pathlib import Path
from stateframe.engine.kernel_guide_task_pack import score_export

fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = "INC-PRICE-731"
decoy = "INC-PRICE-984"
score = score_export(
    requested_run=target, requested_send=2,
    exported_bytes=fixture["frame"].encode("utf-8"), exported_send=2,
    run_sends={target: {2: fixture["target_send"].encode("utf-8")},
               decoy: {2: fixture["decoy_send"].encode("utf-8")}},
)
assert score.identity_match and score.send_match, score
assert not score.wire_format_match and not score.exact_bytes_match and not score.accepted, score
"""
    clean_env = {
        str(key): str(value) for key, value in os.environ.items()
        if not str(key).startswith("STATEFRAME_") and str(key) != "PYTHONPATH"
    }
    clean_env["PYTHONPATH"] = str(ROOT / "src")
    package_result = subprocess.run(
        [sys.executable, "-c", package_probe, str(package_fixture)],
        cwd=tmp_path, env=clean_env, capture_output=True, text=True,
        timeout=60, check=False,
    )
    assert package_result.returncode == 0, package_result.stdout + package_result.stderr

    # The real inspect --frame export has independent run identity but is not the raw request.
    wrong_format = score_export(
        requested_run=target, requested_send=2, exported_bytes=frame_export,
        exported_send=2, run_sends=request_sends,
    )
    assert wrong_format.identity_match and wrong_format.send_match
    assert not wrong_format.wire_format_match
    assert not wrong_format.exact_bytes_match and not wrong_format.accepted

    oversized = score_export(
        requested_run=target, requested_send=2,
        exported_bytes=b"x" * (MAX_EXPORT_BYTES + 1), exported_send=2,
        run_sends=request_sends,
    )
    assert not oversized.accepted and oversized.reason == "export_too_large_or_not_bytes"

    # Each scored result is bound to the request body at the requested send.
    assert hashlib.sha256(target_send).hexdigest() == hashlib.sha256(
        cell_requests["both"][target][1]).hexdigest()
    assert source_revisions[target] == source_revisions[decoy]
