"""Private, opt-in candidate snapshots for predeclared budget curves."""
from __future__ import annotations

import difflib
import base64
import hashlib
import json
from pathlib import Path

from kernel_information_policy import text_hash


def capture_candidate(destination: str, root: Path, row: dict, proposals: int) -> None:
    if not destination:
        return
    folder = Path(destination).resolve()
    poc = Path(__file__).parent.resolve()
    if folder.is_relative_to(root.resolve()) or any(
            folder.is_relative_to((poc / name).resolve())
            for name in ("results", "results.SNAPSHOT-20260920")):
        raise ValueError("candidate evidence must remain outside worker and protected trees")
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {"__pycache__", ".pytest_cache", ".test-tmp"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("candidate evidence cannot replay symlinks")
        if path.is_file():
            files[str(relative)] = path.read_bytes()
    sources = {name: data.decode("utf-8", errors="replace")
               for name, data in files.items() if name.endswith(".py")}
    if row.get("files") is not None:
        from build_tasks.multifile_row import normalize_repo_path
        target = normalize_repo_path(row["path"], field="candidate multi-file target")
        if target not in row["files"]:
            raise ValueError("candidate target must be a declared multi-file source")
    else:
        target = "app/" + row["rel"]
    candidate = sources.get(target)
    record = {"proposals": proposals, "sources": sources, "target": target,
              "files_base64": {name: base64.b64encode(data).decode("ascii") for name, data in files.items()},
              "file_hashes": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
              "target_missing": candidate is None,
              "source_hashes": {key: text_hash(value) for key, value in sources.items()},
              "patch": "".join(difflib.unified_diff(
                  row["parent_src"].splitlines(keepends=True),
                  (candidate or "").splitlines(keepends=True),
                  fromfile=target + ".parent", tofile=target))}
    path = folder / f"{proposals:04d}.json"
    # One snapshot at each proposal boundary. Repeated boundary writes must agree.
    encoded = json.dumps(record, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError("candidate boundary was captured twice with different contents")
        return
    with path.open("x") as handle:
        handle.write(encoded)


def final_capture_index(loop_step: int, reported_steps: int) -> int:
    """Index for the final/exception candidate capture: strictly past every in-loop capture (keyed by loop step).

    In budget-unit=response mode the reported step count is a different quantity and could reuse an already
    captured index with different contents (r35d: 'candidate boundary was captured twice with different contents').
    """
    return max(int(loop_step) + 1, int(reported_steps))
