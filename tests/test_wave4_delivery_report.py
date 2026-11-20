"""Check startup measurement reports and their command exit status."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from _timeouts import scaled


OBSERVER = Path(__file__).resolve().parents[1] / "scripts" / "measure_send1_gap.py"
MARKER = b'{"send":1}'
WIRE = b'{"messages":[]}'
PUBLISH = """
from pathlib import Path
import sys

run = Path(sys.argv[1])
if sys.argv[2] != 'missing':
    (run / 'requests').mkdir(parents=True)
    (run / 'requests/0001.json').write_bytes(bytes.fromhex(sys.argv[3]))
(run / 'wire').mkdir(parents=True)
(run / 'wire/0001.request.json').write_bytes(bytes.fromhex(sys.argv[4]))
"""


@pytest.mark.parametrize(
    "case, expected_status, marker_equal, wire_equal",
    [
        pytest.param("matching", 0, True, True, id="matching"),
        pytest.param("different-wire", 1, True, False, id="different-wire"),
        pytest.param("missing-marker", 1, None, True, id="missing-marker"),
        pytest.param("wrong-digest", 1, True, True, id="wrong-digest"),
    ],
)
def test_observer_reports_exact_pairs_and_preserves_exit_status(
        tmp_path, case, expected_status, marker_equal, wire_equal):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("requires bash to check pipeline exit status")

    plan = []
    for label in ("baseline", "candidate"):
        run = tmp_path / f"{label} result"
        marker_state = "missing" if case == "missing-marker" and label == "candidate" else "present"
        wire = b'{"messages":[1]}' if case == "different-wire" and label == "candidate" else WIRE
        command = [sys.executable, "-I", "-c", PUBLISH, str(run), marker_state, MARKER.hex(), wire.hex()]
        plan.append({"label": label, "state": "cold", "run_dir": str(run), "command": command})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    report_path = tmp_path / "report.json"
    marker_digest = hashlib.sha256(MARKER).hexdigest()
    expected_digest = "0" * 64 if case == "wrong-digest" else marker_digest
    result = subprocess.run(
        [bash, "--noprofile", "--norc", "-o", "pipefail", "-c", '"${@:2}" | tee "$1"', "bash",
         str(report_path), sys.executable, "-I", "-B", str(OBSERVER), "--plan", str(plan_path),
         "--period", "0.01", "--timeout", "10", "--expected-marker-sha256", expected_digest],
        cwd=tmp_path, capture_output=True, timeout=scaled(30),
    )
    assert result.returncode == expected_status, result.stderr.decode()
    assert result.stdout == report_path.read_bytes()
    report = json.loads(result.stdout)
    assert [(row["label"], row["run_dir"]) for row in report["runs"]] == [
        (entry["label"], entry["run_dir"]) for entry in plan
    ]
    baseline, candidate = report["runs"]
    assert baseline["request_marker"]["sha256"] == marker_digest
    assert baseline["status"] == "complete"
    assert candidate["status"] == ("early_exit" if marker_equal is None else "complete")
    pair, = report["pairs"]
    assert pair["baseline_run"] == baseline["run_dir"]
    assert pair["candidate_run"] == candidate["run_dir"]
    assert pair["request_marker_equal"] is marker_equal
    assert pair["transport_body_equal"] is wire_equal
