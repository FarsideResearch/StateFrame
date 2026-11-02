"""Run a bounded, agent-written Python check against no-tests parent and candidate."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from kernel_authored_check import admit_authored_check
from kernel_authored_check_lifecycle import pair_executions
from kernel_execution_policy import run_sandboxed


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True,
                                                  default=str).encode()).hexdigest()


def _run_side(snapshot: Mapping[str, object], path: str, source: bytes,
              identity: str, marker: str, side: str, parent: Path) -> dict[str, object]:
    revision = str(snapshot.get("revision") or "")
    files = snapshot.get("files")
    if not revision or not isinstance(files, Mapping):
        return {"check_identity": identity, "side": side, "revision": revision,
                "status": "UNRUNNABLE", "reason": "snapshot_unavailable"}
    try:
        with tempfile.TemporaryDirectory(prefix="authored-check-", dir=parent.resolve()) as directory:
            root = Path(directory).resolve()
            for relative, content in files.items():
                if not isinstance(relative, str) or not isinstance(content, bytes):
                    return {"check_identity": identity, "side": side, "revision": revision,
                            "status": "UNRUNNABLE", "reason": "snapshot_invalid"}
                target = root / relative
                if not target.resolve().is_relative_to(root.resolve()):
                    return {"check_identity": identity, "side": side, "revision": revision,
                            "status": "UNRUNNABLE", "reason": "snapshot_path_invalid"}
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            check = root / path
            check.parent.mkdir(parents=True, exist_ok=True)
            check.write_bytes(source)
            bootstrap = root / "_stateframe_check_main.py"
            bootstrap.write_text("import runpy\nrunpy.run_path(" + repr(str(check)) +
                                 ", run_name='__main__')\n", encoding="utf-8")
            result = run_sandboxed((sys.executable, str(bootstrap)), root,
                                   generic_check=True,
                                   timeout=120, cpu_seconds=120, output_limit=65536)
    except (OSError, RuntimeError, ValueError) as error:
        return {"check_identity": identity, "side": side, "revision": revision,
                "status": "UNRUNNABLE", "reason": f"{type(error).__name__}: {error}"}
    output = result.completed.stdout + result.completed.stderr
    reached = marker in output
    status = ("UNRUNNABLE" if result.timed_out or result.completed.returncode in {126, 127}
              else "PASS" if result.completed.returncode == 0 and reached
              else "BEHAVIOR_FAIL" if result.completed.returncode != 0 and reached
              else "UNKNOWN")
    return {"check_identity": identity, "side": side, "revision": revision,
            "status": status, "reached_target": reached,
            "integrity": not result.timed_out, "binding_stable": True,
            "reason": "timeout" if result.timed_out else "marker_not_reached" if not reached else "",
            "exit_code": result.completed.returncode, "output_tail": output[-4096:]}


def run_agent_check(proposal: Mapping[str, object], *, task: str,
                    parent_snapshot: Mapping[str, object],
                    candidate_snapshot: Mapping[str, object], root: Path) -> dict[str, object]:
    """Require an admitted scratch check that fails on the parent and passes now."""
    path = proposal.get("path")
    if not isinstance(path, str) or not re.fullmatch(r"scratch/authored-check/[A-Za-z0-9_-]+\.py", path):
        return {"status": "REJECTED", "reason": "path must be scratch/authored-check/<name>.py",
                "certifies_clause": False}
    for snapshot in (parent_snapshot, candidate_snapshot):
        files = snapshot.get("files")
        if isinstance(files, Mapping) and (path in files or "_stateframe_check_main.py" in files):
            return {"status": "REJECTED", "reason": "authored check path collides with repository file",
                    "certifies_clause": False}
    authority = {"authority_ref": _digest({"root": str(root), "path": path}),
                 "allowed_paths": [path]}
    candidate = dict(proposal)
    candidate["write_authority"] = authority
    admission = admit_authored_check(
        candidate, public_text=task,
        parent_revision=str(parent_snapshot.get("revision") or ""),
        candidate_revision=str(candidate_snapshot.get("revision") or ""),
        authorized_paths=(path,), protected_paths=())
    if admission.get("admitted") is not True:
        return {"status": "REJECTED", "admission": admission, "certifies_clause": False}
    artifact = admission["artifact"]
    marker = str(artifact["observation_marker"])
    source = artifact["source"]
    if not isinstance(source, bytes):
        source = source.encode("utf-8")
    identity = str(admission["check_hash"])
    before = _run_side(parent_snapshot, path, source, identity, marker, "parent", root.parent)
    after = _run_side(candidate_snapshot, path, source, identity, marker, "candidate", root.parent)
    result = pair_executions(before, after, clause_id=str(admission["clause_id"]))
    result["route"] = "NO_TESTS_AUTHORED_CHECK"
    result["parent_output_tail"] = before.get("output_tail", "")
    result["candidate_output_tail"] = after.get("output_tail", "")
    return result
