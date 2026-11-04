"""Compare raw no-server request files against the 0.2.3 source tree that test_no_mcp_release_comparison pins.

Run with --baseline /path/to/33c6c51 (0.2.3) --candidate /path/to/this/tree.
Both runs use the same temporary repository and run-directory path in sequence.
The candidate pins the settings whose defaults changed after 0.2.3 (pytest runner, all tests
protected, stable tool head off), so the request files are expected to be byte-identical;
pass --current-defaults to see how today's defaults differ instead.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
from pathlib import Path
import select
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from stateframe.request_records import list_request_records, read_request_record


def source_env(tree: Path, stable_tool_head: str | None, *,
               source_on_pythonpath: bool = True) -> dict[str, str]:
    environment = dict(os.environ)
    if source_on_pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join((str(tree / "src"),
            str(tree / "src/stateframe/engine")))
    else:
        environment.pop("PYTHONPATH", None)
    environment.pop("STATEFRAME_STABLE_TOOL_HEAD", None)
    if stable_tool_head is not None:
        environment["STATEFRAME_STABLE_TOOL_HEAD"] = stable_tool_head
    return environment


# Wire IDs are exactly WIRE_ID_HEX_LENGTH (20) hex chars; check receipts use 24.
# The receipt-determinism change (11-03) derives them from task + step + content
# instead of the journal position, so they deliberately differ from releases cut
# before it. Older releases also suffixed some IDs with the global journal sequence
# ("check:<id>:1067"); the change removes that suffix, so it is masked with the ID.
# A tool result's result_sha256 hashes that result's text, wire IDs included. The
# text itself is in the same request and compared after masking, so its hash is
# masked too. Release comparisons mask ONLY these tokens; every other byte must match.
_WIRE_ID_TOKEN = re.compile(
    rb"(?<![0-9a-f])(?:[0-9a-f]{20}|[0-9a-f]{24})(?![0-9a-f])(?::[0-9]+)?")
_RESULT_SHA = re.compile(rb'(result_sha256\\*":\s*\\*")([0-9a-f]{64})')
# historical_applied_edits.sha256 digests the older applied-edit rows pruned from the frame.
# Each row carries an edit_receipt ID, and each row was sent (and compared after masking) in an
# earlier request while it was among the last three edits. Only this one digest is masked; its
# count stays compared.
_HISTORY_SHA = re.compile(
    rb'(historical_applied_edits\\*":\s*\{\\*"count\\*":\s*[0-9]+,\s*\\*"sha256\\*":\s*\\*")([0-9a-f]{64})')


def mask_wire_ids(body: bytes) -> bytes:
    """Replace wire-ID tokens so a release comparison checks everything else exactly."""
    body = _HISTORY_SHA.sub(rb"\1<applied-edit-history-sha256>", body)
    return _WIRE_ID_TOKEN.sub(b"<wire-id>", _RESULT_SHA.sub(rb"\1<result-sha256>", body))


def result_hashes(run_dir: Path) -> tuple[list[str], set[str]]:
    """The result_sha256 values a run sent, in order, and the hashes of its sent message texts."""
    sent, texts = [], set()
    for path in sorted(run_dir.rglob("*.request.json")):
        raw = path.read_bytes()
        sent += [match.group(2).decode() for match in _RESULT_SHA.finditer(raw)]
        for message in json.loads(raw).get("messages", []):
            if isinstance(message.get("content"), str):
                texts.add(hashlib.sha256(message["content"].encode("utf-8")).hexdigest())
    return sent, texts


def unbacked_result_hash_changes(before_run: Path, after_run: Path) -> list[str]:
    """Changed result_sha256 values whose hashed text was not sent, so masking them could hide a difference."""
    before_hashes, before_texts = result_hashes(before_run)
    after_hashes, after_texts = result_hashes(after_run)
    if len(before_hashes) != len(after_hashes):
        return [f"{len(before_hashes)} result hashes before, {len(after_hashes)} after"]
    return [f"{before} -> {after}" for before, after in zip(before_hashes, after_hashes)
            if before != after and (before not in before_texts or after not in after_texts)]


# The field a differing wire ID sits in, named from the text just before it.
_ID_FIELD_KINDS = (
    (re.compile(rb"wire-truth:$"), "wire_truth_handle(request_id)"),
    (re.compile(rb"edit_receipt\\*\":\s*\\*\"$"), "edit_receipt"),
    (re.compile(rb"check-receipt:$"), "check_receipt"),
    (re.compile(rb"check-execution:$"), "check_execution"),
    (re.compile(rb"receipt_id\\*\":\s*\\*\"$"), "receipt_id"),
    # native_<request_id>_<n>: tool-call IDs embed the request that made the call.
    (re.compile(rb"native_$"), "native_call_id(request_id)"),
    (re.compile(rb"request_id\\*\":\s*\\*\"$"), "request_id"),
    (re.compile(rb"\"check:$"), "check_finding_id"),
    (re.compile(rb"check_execution_ref: $"), "check_execution_ref"),
    (re.compile(rb"expectation-receipt:$"), "expectation_receipt"),
    (re.compile(rb"\"receipt:$"), "evidence_receipt"),
    # A continuation workspace directory name, seen by the model inside test tracebacks.
    (re.compile(rb"continuation-workspaces/$"), "continuation_workspace_dir"),
)


# Every field the receipt-determinism change (11-03) is allowed to move. Release parity tests
# fail on a changed ID in any other field, and on any change outside an ID.
MOVED_ID_FIELDS = frozenset({
    "wire_truth_handle(request_id)", "request_id", "native_call_id(request_id)",
    "edit_receipt", "check_receipt", "check_execution", "check_execution_ref",
    "check_finding_id", "receipt_id", "result_sha256", "historical_applied_edits_sha256",
    "expectation_receipt", "continuation_workspace_dir", "evidence_receipt",
})


def _id_field_kind(body: bytes, start: int) -> str:
    before = body[max(0, start - 40):start]
    return next((name for pattern, name in _ID_FIELD_KINDS if pattern.search(before)), "other_wire_id")


def wire_id_drift(before: list[bytes], after: list[bytes]) -> tuple[list[str], dict[str, int]]:
    """Problems if two request-body lists differ in anything but wire IDs, plus which ID fields changed.

    Bodies must be byte-identical after mask_wire_ids. Each changed token is named by the field
    it sits in, so a report can say exactly what moved.
    """
    problems: list[str] = []
    kinds: dict[str, int] = {}
    if len(before) != len(after):
        return [f"{len(before)} requests before, {len(after)} after"], kinds
    for index, (old, new) in enumerate(zip(before, after)):
        if mask_wire_ids(old) != mask_wire_ids(new):
            masked_old, masked_new = mask_wire_ids(old), mask_wire_ids(new)
            offset = next((i for i, (x, y) in enumerate(zip(masked_old, masked_new)) if x != y),
                          min(len(masked_old), len(masked_new)))
            problems.append(f"request {index}: non-ID bytes differ at masked offset {offset}: "
                            f"{masked_old[max(0, offset - 60):offset + 40]!r} -> "
                            f"{masked_new[max(0, offset - 60):offset + 40]!r}")
            continue
        for old_token, new_token in zip(_WIRE_ID_TOKEN.finditer(old), _WIRE_ID_TOKEN.finditer(new)):
            if old_token.group() != new_token.group():
                kind = _id_field_kind(new, new_token.start())
                kinds[kind] = kinds.get(kind, 0) + 1
        for old_hash, new_hash in zip(_RESULT_SHA.finditer(old), _RESULT_SHA.finditer(new)):
            if old_hash.group(2) != new_hash.group(2):
                kinds["result_sha256"] = kinds.get("result_sha256", 0) + 1
        for old_hash, new_hash in zip(_HISTORY_SHA.finditer(old), _HISTORY_SHA.finditer(new)):
            if old_hash.group(2) != new_hash.group(2):
                kinds["historical_applied_edits_sha256"] = kinds.get("historical_applied_edits_sha256", 0) + 1
    return problems, kinds


def changed_id_chars(before: list[bytes], after: list[bytes]) -> list[int]:
    """Per request, how many characters sit in wire-ID tokens whose value changed. A token count
    can follow the ID's characters, not only its length, so this bounds token movement."""
    return [sum(len(n.group()) for o, n in zip(_WIRE_ID_TOKEN.finditer(old), _WIRE_ID_TOKEN.finditer(new))
                if o.group() != n.group())
            for old, new in zip(before, after)]


def id_length_deltas(before: list[bytes], after: list[bytes]) -> tuple[list[int], list[str]]:
    """Per request, how much the wire-ID tokens changed in length, and any body whose size
    change is NOT fully explained by them (older releases suffixed some IDs with ':<sequence>')."""
    deltas, problems = [], []
    for index, (old, new) in enumerate(zip(before, after)):
        pairs = zip(_WIRE_ID_TOKEN.finditer(old), _WIRE_ID_TOKEN.finditer(new))
        delta = sum(len(n.group()) - len(o.group()) for o, n in pairs)
        deltas.append(delta)
        if len(new) - len(old) != delta:
            problems.append(f"request {index}: size changed by {len(new) - len(old)}, ID tokens by {delta}")
    return deltas, problems


_WIRE_TRUTH_HANDLE = re.compile(rb"wire-truth:([0-9a-f]{20}):")


def request_id_handle_mismatches(run_dir: Path) -> list[str]:
    """Each wire-truth handle a request sends must name that request's own recorded request_id."""
    # Request records may be loose files or sealed into requests.pack.xz; read both forms.
    from stateframe.request_records import list_request_records, read_request_record

    problems = []
    names = list_request_records(run_dir)
    if not names:
        return [f"{run_dir}: no request records, so nothing was checked"]
    for name in names:
        request_id = json.loads(read_request_record(run_dir, name)).get("request_id")
        body = (run_dir / "wire" / f"{Path(name).stem}.request.json").read_bytes()
        handles = {match.group(1).decode() for match in _WIRE_TRUTH_HANDLE.finditer(body)}
        if handles - {request_id}:
            problems.append(f"{name}: request_id {request_id} but handles {sorted(handles)}")
    return problems


def planted_body_change_is_caught(body: bytes) -> bool:
    """Flip one letter outside every wire ID in a real body; wire_id_drift must report each one.

    Planted at the first eligible byte, the middle of the body, and the first byte after a
    masked ID, so a mask that swallowed a neighbouring field would show up.
    """
    masked_spans = [match.span() for match in _WIRE_ID_TOKEN.finditer(body)]
    masked_spans += [match.span(2) for match in _RESULT_SHA.finditer(body)]
    masked_spans += [match.span(2) for match in _HISTORY_SHA.finditer(body)]
    eligible = [position for position, value in enumerate(body)
                if chr(value).isalpha()
                and not any(start <= position < end for start, end in masked_spans)]
    if not eligible or not masked_spans:
        return False
    first_id_end = min(end for _start, end in masked_spans)
    targets = {eligible[0], eligible[len(eligible) // 2],
               next((p for p in eligible if p >= first_id_end), eligible[-1])}
    for position in targets:
        planted = body[:position] + bytes([body[position] ^ 0x20]) + body[position + 1:]
        if not wire_id_drift([body], [planted])[0]:
            return False
    return True


def write_launcher(path: Path) -> None:
    path.write_text("import runpy, secrets\n"
                    "_sequence = 0\n"
                    "def fixed_hex(size=32):\n"
                    "    global _sequence\n"
                    "    _sequence += 1\n"
                    "    return format(_sequence, 'x').zfill(size * 2)[-size * 2:]\n"
                    "secrets.token_hex = fixed_hex\n"
                    "runpy.run_module('stateframe', run_name='__main__')\n")


def capture(tree: Path, python: str, launcher: Path, script: Path,
            repo: Path, run_dir: Path, *, legacy_mode: bool = False,
            stable_tool_head: str | None,
            source_on_pythonpath: bool = True,
            test_python: str | None = None) -> bytes:
    # stable_tool_head has no default (see tools/changed_defaults.py): every caller states its arm.
    environment = source_env(tree, stable_tool_head=stable_tool_head,
                             source_on_pythonpath=source_on_pythonpath)
    # Each capture gets its own empty home. The current tree reads enabled powerups and user skills from
    # STATEFRAME_HOME and old releases do not, so a home inherited from the test session (where an earlier test
    # may have left a powerup enabled) changed only one side's request bytes (11-04, 0.2.8 parity in full runs).
    environment["STATEFRAME_HOME"] = tempfile.mkdtemp(prefix=f"{run_dir.name}-home-", dir=run_dir.parent)
    endpoint = subprocess.Popen([python, "-m", "stateframe.scripted_endpoint",
        "--script", str(script), "--port", "0"], cwd=tree, env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert endpoint.stdout is not None
        ready, _, _ = select.select([endpoint.stdout], [], [], 15)
        if not ready:
            raise RuntimeError("scripted endpoint did not start")
        url = endpoint.stdout.readline().strip()
        if not url.startswith("http://127.0.0.1:"):
            raise RuntimeError(f"unexpected scripted endpoint URL: {url!r}")
        command = [python, str(launcher), "run", "--repo", str(repo),
            "--task", "Fix the pricing code so the tests pass.", "--endpoint", url,
            "--model", "scripted", "--run-dir", str(run_dir),
            "--test-python", test_python or python,
            "--test-cmd", "python -m unittest discover -s tests -q", "--max-turns", "32",
            "--no-run-env", "--quiet",
            *(("--runner-kind", "pytest", "--protect-tests", "all") if legacy_mode else ())]
        result = subprocess.run(command, cwd=tree, env=environment, capture_output=True,
                                text=True, timeout=300, check=False)
        # The comparison is about request bytes, so a run whose checks fail still counts;
        # only a run that never wrote its summary or requests is a capture failure.
        names = list_request_records(run_dir)
        if not names or not (run_dir / "summary.json").is_file():
            raise RuntimeError(f"{tree}: run produced no requests or summary ({result.returncode}):\n"
                               f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        return b"\n".join(read_request_record(run_dir, name) for name in names)
    finally:
        endpoint.terminate()
        try:
            endpoint.wait(timeout=5)
        except subprocess.TimeoutExpired:
            endpoint.kill()
            endpoint.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="untouched source tree of 0.2.3 (33c6c51)")
    parser.add_argument("--candidate", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter that runs the example's tests (each tree runs under its own)")
    parser.add_argument("--current-defaults", action="store_true",
                        help="run the candidate with today's defaults instead of the pinned 0.2.3 settings")
    args = parser.parse_args()
    baseline = args.baseline.resolve(strict=True)
    candidate = args.candidate.resolve(strict=True)
    script = candidate / "examples/discount_bug.script.json"
    # The engine child drops PYTHONPATH, so a shared interpreter would import whichever stateframe it has
    # installed. Give each tree its own interpreter and prove what it imports before comparing bytes.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from acceptance._support import assert_tree_imports, tree_python
    with tempfile.TemporaryDirectory(prefix="sf-no-mcp-diff-") as temporary:
        root = Path(temporary)
        repo = root / "repo"
        shutil.copytree(candidate / "examples/discount_bug", repo)
        launcher = root / "fixed_seed_run.py"
        write_launcher(launcher)
        run_dir = root / "run"
        pythons = {}
        for label, tree in (("baseline", baseline), ("candidate", candidate)):
            pythons[label] = tree_python(tree / "src", root / f"{label}-python")
            assert_tree_imports(pythons[label], tree / "src", root)
        previous = capture(baseline, pythons["baseline"], launcher, script, repo, run_dir,
                           stable_tool_head=None, source_on_pythonpath=False, test_python=args.python)
        run_dir.rename(root / "baseline-run")
        current = capture(candidate, pythons["candidate"], launcher, script, repo, run_dir,
                          legacy_mode=not args.current_defaults,
                          stable_tool_head=None if args.current_defaults else "off",
                          source_on_pythonpath=False, test_python=args.python)
        for label, body in (("baseline", previous), ("candidate", current)):
            print(f"{label}: {len(body)} bytes sha256={hashlib.sha256(body).hexdigest()}")
        if previous == current:
            print("Model request files are byte-identical.")
            return 0
        before = previous.decode(errors="replace").splitlines(keepends=True)
        after = current.decode(errors="replace").splitlines(keepends=True)
        sys.stdout.writelines(difflib.unified_diff(before, after, fromfile="baseline",
                                                   tofile="candidate", n=3))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
