"""Class killer: a run that SUCCEEDS must not die at the finish line on evidence digests.

0.5.9 P0: with the default r2hc3 profile and a short task that one edit satisfies, the
engine auto-finished on the first green host check and crashed with
``TypeError: Object of type CheckReceipt is not JSON serializable``
(kernel_measurement.canonical_bytes <- pass_gates_vector evidence digest of
``host_checks`` <- kernel_done_readiness_authority.decision_for_runtime <- finish).

Root cause: ``StateFrameRuntime.host_check_receipts`` / ``host_repair_check_receipts``
hold ``CheckReceipt`` dataclass instances (the executor's native row type) while the
pass-gates evidence digest, the DECIDE2 check-evidence identity and the done-readiness
evidence digest fed those rows straight into ``json.dumps``.  The 10-06 fix only
normalized the smoke result.  Every one of those digests now goes through
``gate_evidence_identity`` (typed: ``to_dict`` for receipt rows, field mapping for other
dataclasses, never ``default=str``).

(a) e2e: the real CLI + scripted endpoint, finish on the first green check -> graded success.
(b) property: every runtime evidence collection that reaches ``canonical_bytes`` from the
    pass-gates / done-readiness / DECIDE2 path is JSON-native after normalization, and the
    normalization is byte-identical for rows that were already JSON-native (replay parity).
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest

from _timeouts import scaled
from engine_run import run_monitored
from stateframe.profiles import load as load_profile
from stateframe.scripted_endpoint import Script, serve

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "stateframe" / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

DEATH_MARKER = "is not JSON serializable"
MODELS_EDIT = ('"""Domain model."""', '"""Domain model (books)."""')
# One edit satisfies this task, so the engine may finish on the first green check.
TASK = ("Task: Change the module docstring of app/models.py to exactly "
        '"""Domain model (books)."""' + ". Keep the existing tests passing.")

FILES = {
    "app/__init__.py": '"""Books package."""\n',
    "app/models.py": '"""Domain model."""\nfrom dataclasses import dataclass\n\n\n@dataclass\nclass Book:\n    id: int\n    name: str\n',
    "app/api.py": (
        '"""JSON-style handlers for books."""\nfrom dataclasses import asdict\nfrom app.models import Book\n\n'
        "_ITEMS: dict[int, Book] = {}\n\n\ndef reset():\n    _ITEMS.clear()\n\n\n"
        "def create(payload: dict) -> dict:\n    item = Book(id=len(_ITEMS) + 1, name=payload[\"name\"])\n"
        "    _ITEMS[item.id] = item\n    return asdict(item)\n\n\n"
        "def get(item_id: int) -> dict:\n    item = _ITEMS.get(item_id)\n    if item is None:\n"
        "        return {\"error\": \"not_found\"}\n    return asdict(item)\n"),
    "tests/test_base.py": (
        "from app import api\n\n\ndef setup_function():\n    api.reset()\n\n\n"
        "def test_create_get():\n    a = api.create({\"name\": \"first\"})\n"
        "    assert a[\"id\"] == 1 and a[\"name\"] == \"first\"\n    assert api.get(99) == {\"error\": \"not_found\"}\n"),
}
STEPS = (
    {"tool": "read", "arguments": {"path": "app/models.py"}},
    {"tool": "str_replace", "arguments": {"path": "app/models.py", "old_str": MODELS_EDIT[0],
                                          "new_str": MODELS_EDIT[1]}},
    {"tool": "done", "arguments": {}},
)


def _environment() -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("STATEFRAME_", "REACT_"))}
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ENGINE)))
    return environment


def _profile(tmp_path: Path) -> Path:
    switches = load_profile("r2hc3")
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"schema": "stateframe-product-profile/1", "switches": switches}))
    return path


@pytest.mark.timeout(scaled(600))
def test_finish_on_first_green_check_completes_with_graded_success(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for relative, text in FILES.items():
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        (repo / relative).write_text(text, encoding="utf-8")
    run_dir = tmp_path / "run"
    document = {"model": "scripted", "steps": [dict(step) for step in STEPS],
                "tokenize": {"mode": "ok"}, "slots": [65536]}
    server = serve(Script(document))
    try:
        command = [sys.executable, "-m", "stateframe", "run", "--repo", str(repo), "--task", TASK,
                   "--target", "app/models.py", "--endpoint", f"http://127.0.0.1:{server.server_port}/v1",
                   "--model", "scripted", "--run-dir", str(run_dir), "--test-python", sys.executable,
                   "--max-turns", "8", "--in-place", "--shell", "off",
                   "--profile", str(_profile(tmp_path)), "--quiet"]
        result = run_monitored(command, env=_environment(), run_dir=run_dir, check=False,
                               expected_seconds=scaled(300), step="finish-on-green")
    finally:
        server.shutdown()
        server.server_close()
    output = result.stdout + result.stderr
    assert DEATH_MARKER not in output, output[-4000:]
    assert "Traceback" not in output, output[-4000:]
    assert result.returncode == 0, output[-4000:]
    summary_path = run_dir / "summary.json"
    assert summary_path.is_file(), output[-4000:]
    summary = json.loads(summary_path.read_text())
    assert (repo / "app/models.py").read_text().startswith(MODELS_EDIT[1])
    assert summary.get("engine_result") == "SUBMITTED", summary
    assert summary.get("check_outcome") == "pass", summary
    assert summary.get("artifact") == "saved", summary
    done_facts = (summary.get("engine_terminal") or {}).get("done_facts") or {}
    assert done_facts.get("accepted") is True, done_facts
    assert done_facts.get("rejection_reasons") == [], done_facts


# --------------------------------------------------------------------------- property


def _receipt(**overrides: object):
    from kernel_obligation_assess import CheckReceipt, PASS
    fields = dict(check_id="c1", status=PASS, snapshot="sha256:s", check="sha256:c",
                  fixture="sha256:f", environment="sha256:e", observed={"rc": 0},
                  expected={"rc": 0}, reason="ok", receipt_id="r1", instance_key="k1",
                  selection=("a",), contract_cell_hash="h", execution="EXECUTED",
                  trace_refs=("t",))
    fields.update(overrides)
    return CheckReceipt(**fields)


def test_gate_evidence_identity_is_json_native_for_every_receipt_collection() -> None:
    """Every evidence axis that can hold executor rows digests without default=str."""
    from kernel_measurement import canonical_bytes
    from kernel_stateframe_runtime import gate_evidence_identity

    receipt = _receipt(identity={"schema": "verdict-identity/1", "host": "x"})
    axes = {
        "host_checks": (receipt,),
        "host_repairs": [receipt, receipt],
        "tests": [{"name": "pytest", "passed": True}],
        "nested": {"rows": ((receipt,),), "scalar": 1},
    }
    identity = gate_evidence_identity(axes)
    encoded = canonical_bytes(identity)  # must not raise
    decoded = json.loads(encoded)
    assert decoded["host_checks"][0]["check_id"] == "c1"
    assert decoded["host_checks"][0]["schema"] == "stateframe-host-check-receipt/1"
    assert decoded["host_repairs"][0] == decoded["host_repairs"][1]
    # The receipt row is its journaled to_dict() form, so digests agree with the journal.
    assert decoded["host_checks"][0] == json.loads(json.dumps(receipt.to_dict()))
    # Deterministic across equal receipts built separately.
    assert canonical_bytes(gate_evidence_identity((_receipt(),))) == canonical_bytes(
        gate_evidence_identity((_receipt(),)))


def test_gate_evidence_identity_is_byte_identical_for_json_native_rows() -> None:
    """Replay / resume parity: rows that were already JSON-native keep their exact digest."""
    from kernel_measurement import canonical_bytes
    from kernel_stateframe_runtime import gate_evidence_identity

    native = {
        "gate_revisions": {"invalidation": "", "recovery": "r1"},
        "tests": [{"name": "pytest", "passed": True, "rows": [1, 2.5, None, False]}],
        "host_checks": (),
        "host_repairs": [],
        "catalog": None,
        "scope": {"ok": True},
        "available_actions": ("done", "read"),
        "witness_rows": [{"a": {"b": [{"c": "d"}]}}],
    }
    assert canonical_bytes(gate_evidence_identity(native)) == canonical_bytes(native)


def test_raw_receipt_rows_reproduce_the_0_5_9_finish_line_class() -> None:
    """The exact pre-fix failure: a CheckReceipt in host_checks fed straight to digest()."""
    from kernel_measurement import digest
    from kernel_stateframe_runtime import gate_evidence_identity

    axes = {"host_checks": (_receipt(),), "host_repairs": [_receipt(check_id="c2")]}
    with pytest.raises(TypeError, match="CheckReceipt is not JSON serializable"):
        digest(axes)
    assert digest(gate_evidence_identity(axes)).startswith("sha256:")


def test_pass_gates_evidence_digest_accepts_receipt_rows_on_a_runtime() -> None:
    """pass_gates_vector's evidence digest (the finish-line caller) survives receipt rows."""
    from types import SimpleNamespace
    from kernel_stateframe_runtime import StateFrameRuntime

    seen: dict[str, object] = {}

    class Runtime:
        host_check_receipts = [_receipt()]
        host_repair_check_receipts = [_receipt(check_id="repair")]
        test_receipts = [{"name": "pytest", "passed": True}]

        def binding(self):
            return {"product_revision": "sha256:abc"}

        def _catalog_sync(self):
            return None

        def catalog_completion_blocker(self):
            seen["reached_make"] = True
            raise RuntimeError("stop after the evidence digest")

    with pytest.raises(RuntimeError, match="stop after the evidence digest"):
        StateFrameRuntime.pass_gates_vector(Runtime(), ({"function": {"name": "done"}},))
    assert seen.get("reached_make") is True


def test_model_facing_check_evidence_digests_receipt_rows() -> None:
    """DECIDE2's per-request dependency identity must not die on host receipt rows either."""
    from types import SimpleNamespace
    from kernel_stateframe_runtime import _model_facing_check_evidence

    runtime = SimpleNamespace(
        test_receipts=[{"name": "pytest", "passed": True}],
        host_check_receipts=[_receipt(identity={"schema": "verdict-identity/1", "host": "x"})],
        _scope_lock_last_result={"ok": True},
    )
    evidence = _model_facing_check_evidence(runtime)
    assert evidence["dependency_identity"].startswith("sha256:")
    rows = evidence["public_evidence"]
    assert rows[0] == {"name": "pytest", "passed": True}
    assert isinstance(rows[1], dict) and rows[1]["check_id"] == "c1"
    json.dumps(rows)  # fully JSON-native for the model-facing frame
    # Parity: without host rows the identity is unchanged from the raw digest.
    from kernel_measurement import digest
    bare = SimpleNamespace(test_receipts=[{"name": "pytest"}], host_check_receipts=(),
                           _scope_lock_last_result=None)
    assert _model_facing_check_evidence(bare)["dependency_identity"] == digest(
        {"test_receipts": [{"name": "pytest"}], "host_checks": (), "scope": None})


def test_gate_evidence_identity_refuses_opaque_objects() -> None:
    """No repr fallback: an unknown host object must fail loudly, not hash its address."""
    from kernel_stateframe_runtime import gate_evidence_identity

    class Opaque:
        pass

    with pytest.raises(TypeError, match="Opaque"):
        gate_evidence_identity({"smoke": Opaque()})


def test_runtime_evidence_collections_are_receipt_rows_or_json_native() -> None:
    """Static guard: every digest() on the pass-gates / DECIDE2 evidence path is normalized.

    These are the two producers that hold executor receipt rows; a future axis added raw
    to either digest would re-open the finish-line class."""
    import ast
    source = (ENGINE / "kernel_stateframe_runtime.py").read_text()
    tree = ast.parse(source)
    guarded = {"pass_gates_vector", "_model_facing_check_evidence"}
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in guarded]
    assert {node.name for node in functions} == guarded, functions
    digest_calls = [node for function in functions for node in ast.walk(function)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "digest"]
    assert len(digest_calls) == 2, [call.lineno for call in digest_calls]
    for call in digest_calls:
        argument = call.args[0] if call.args else None
        assert isinstance(argument, ast.Call) and isinstance(argument.func, ast.Name) \
            and argument.func.id == "gate_evidence_identity", (
                f"line {call.lineno}: digest() argument must be wrapped in gate_evidence_identity")
