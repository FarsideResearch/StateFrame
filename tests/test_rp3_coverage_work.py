"""Work counts and compatibility at the two coverage preparation seams."""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pytest

import kernel_false_completion as false_completion
import kernel_named_layer_coverage as layers
import kernel_workspace_tree as workspace_tree
from kernel_stateframe_runtime import StateFrameRuntime
from kernel_workspace_consumers import WorkspaceConsumers, bind_consumers
from kernel_workspace_state import WorkspaceFence, WorkspaceState


_SWITCHES = (
    "STATEFRAME_WORKSPACE_OBJECTS",
    "STATEFRAME_TURN_ATOMIC_PUBLICATION",
    "STATEFRAME_TURN_BOUNDED_QUALIFICATION",
    "STATEFRAME_TURN_DELIVERY_PROOFS",
    "STATEFRAME_TURN_GENERATION_OVERLAP",
    "STATEFRAME_TURN_INDEXED_READS",
    "STATEFRAME_TURN_MINIMAL_LAUNCH",
    "STATEFRAME_TURN_PANELS",
    "STATEFRAME_TURN_REPO_INTEL",
    "STATEFRAME_TURN_TIMING",
)
_TICKET = ("Update CLI `cli.run`, API `api.serve`, storage `store.save`, and "
           "validation `validate`. Do not change config `settings.load`.")


def _runtime(root, ticket=_TICKET):
    events = []
    runtime = StateFrameRuntime.__new__(StateFrameRuntime)
    runtime.owner = SimpleNamespace(root=root, task_row={"ticket_text": ticket})
    runtime.journal = SimpleNamespace(
        emit=lambda kind, **fields: events.append({"kind": kind, **fields}),
        records=lambda: events.copy(),
    )
    runtime.stateframe_layer_coverage_mode = "on"
    runtime.stateframe_coverage_reminders_mode = "on"
    runtime.stateframe_false_completion_mode = "on"
    runtime._workspace_snapshot_cache = {}
    runtime.test_receipts = []
    runtime.behavior_witness_records = []
    binding = {"product_revision": "start", "code_hash": "start"}
    runtime.binding = lambda: binding.copy()
    return runtime, events, binding


def _admit(runtime, *, resume=None):
    runtime._initialize_layer_coverage(resume=resume)
    runtime._initialize_false_completion(resume=resume)


def _reports(runtime):
    return runtime.layer_coverage_report(), runtime.false_completion_report()


def _switches(monkeypatch, mode):
    for name in _SWITCHES:
        monkeypatch.setenv(name, mode)


def _bind(root, private):
    state = WorkspaceState(WorkspaceFence.create(root, private))
    consumers = WorkspaceConsumers(state)
    bind_consumers(root, consumers)
    return consumers


def _instrument(monkeypatch, root, state):
    counts = Counter()
    snapshot = layers.snapshot_workspace
    defines = layers._defines
    relative = layers._relative
    read_bytes = Path.read_bytes
    read_text = Path.read_text
    pin = state.pin
    tree_diff = workspace_tree.diff

    def count_snapshot(*args):
        counts["legacy_sweeps"] += 1
        return snapshot(*args)

    def count_defines(*args):
        counts["legacy_defines"] += 1
        return defines(*args)

    def count_relative(*args):
        counts["path_resolutions"] += 1
        return relative(*args)

    def count_read(fn, path, *args, **kwargs):
        if path.is_relative_to(root):
            counts["source_reads"] += 1
        return fn(path, *args, **kwargs)

    def count_diff(old, new, work=None):
        work = workspace_tree.TreeWork() if work is None else work
        for change in tree_diff(old, new, work):
            counts["projection_paths"] += 1
            yield change
        counts["tree_pages"] += work.pages_visited

    @contextmanager
    def count_pin(*args, **kwargs):
        with pin(*args, **kwargs) as observation:
            counts["full_observations"] += bool(state.work.directories_scanned)
            counts["workspace_reads"] += state.work.content_reads
            yield observation

    monkeypatch.setattr(layers, "snapshot_workspace", count_snapshot)
    monkeypatch.setattr(layers, "_defines", count_defines)
    monkeypatch.setattr(layers, "_relative", count_relative)
    monkeypatch.setattr(Path, "read_bytes", lambda path: count_read(read_bytes, path))
    monkeypatch.setattr(
        Path, "read_text", lambda path, *args, **kwargs: count_read(read_text, path, *args, **kwargs))
    monkeypatch.setattr(state, "pin", count_pin)
    monkeypatch.setattr(workspace_tree, "diff", count_diff)
    if hasattr(layers, "_blob_definitions"):
        definitions = layers._blob_definitions

        def count_definitions(data):
            counts["definition_scans"] += 1
            return definitions(data)

        monkeypatch.setattr(layers, "_blob_definitions", count_definitions)
    return counts


def test_legacy_coverage_shares_one_file_scan_and_refreshes_on_add(tmp_path, monkeypatch):
    _switches(monkeypatch, "off")
    monkeypatch.setenv("STATEFRAME_HOST_STEP_BUDGET2_CACHE", "off")
    (tmp_path / "cli.py").write_text("def run(): pass\n")
    (tmp_path / "settings.py").write_text("def load(): pass\n")
    ticket = "Update CLI. Do not change config. Do not change docs."
    runtime, events, _ = _runtime(tmp_path, ticket)
    scan = layers.CoverageFileCache._scan
    scans = []

    def counted_scan(root):
        scans.append(root)
        return scan(root)

    def unexpected_resolution(*args):
        pytest.fail("Regular files should use their directory prefix")

    monkeypatch.setattr(layers.CoverageFileCache, "_scan", staticmethod(counted_scan))
    monkeypatch.setattr(layers, "_relative", unexpected_resolution)

    _admit(runtime)
    named, _ = _reports(runtime)
    assert scans == [tmp_path]
    assert events[0]["start_snapshot"] == events[1]["start_snapshot"]
    assert named["rows"][0]["files"] == ["cli.py"]
    assert [row["target_files"] for row in named["review_reminders"]] == [["settings.py"], []]

    (tmp_path / "README.md").write_text("Documentation\n")
    named, _ = _reports(runtime)
    assert scans == [tmp_path, tmp_path]
    assert named["review_reminders"][1]["target_files"] == ["README.md"]


def test_coverage_prepare_work_at_repository_scale(tmp_path, monkeypatch):
    _switches(monkeypatch, "on")
    root = tmp_path / "workspace"
    root.mkdir()
    for index in range(4495):
        (root / f"module_{index:04}.py").write_text(f"VALUE = {index}\n")
    for path, name in (("cli", "run"), ("api", "serve"), ("store", "save"),
                       ("guards", "validate"), ("settings", "load")):
        (root / f"{path}.py").write_text(f"def {name}():\n    return 1\n")
    (tmp_path / "outside.py").write_text("def validate():\n    return -1\n")
    (root / "escaped.py").symlink_to(tmp_path / "outside.py")

    consumers = _bind(root, tmp_path / "fence")
    counts = _instrument(monkeypatch, root, consumers.state)
    runtime, events, binding = _runtime(root)
    measurements = {}

    def measure(label, operation):
        counts.clear()
        start = perf_counter()
        result = operation()
        measurements[label] = (perf_counter() - start, counts.copy())
        return result

    measure("admission", lambda: _admit(runtime))
    assert [event["kind"] for event in events] == [
        "named_layer_coverage_started", "false_completion_started",
    ]
    assert len(events[0]["start_snapshot"]) == 4500
    assert events[0]["start_snapshot"] == events[1]["start_snapshot"]
    first = measure("first_prepare", lambda: _reports(runtime))
    assert [row["layer"] for row in first[0]["rows"]] == ["CLI", "API", "storage", "validation"]
    assert first[1]["low_coverage"] is True

    def prepare_twice():
        assert _reports(runtime) == first
        assert _reports(runtime) == first

    measure("two_unchanged_prepares", prepare_twice)
    with consumers.state.begin_write(("guards.py",)):
        (root / "guards.py").write_text("def validate():\n    return 2\n")
    binding.update(product_revision="edited", code_hash="edited")
    edited = measure("one_edit", lambda: _reports(runtime))
    assert edited[0]["rows"][-1]["diff_from_start"] is True

    runtime.owner.task_row["public_tests"] = {"test_guards.py": "assert validate() == 2\n"}
    amended = measure("test_amendment", lambda: _reports(runtime))
    assert amended[0]["rows"] == edited[0]["rows"]
    assert amended[1]["assertion_count"] == 1
    runtime.test_receipts = [{"clause": "Do not change config `settings.load`.",
                              "status": "PASS", "exit_code": 0}]
    checked = measure("public_check_amendment", lambda: _reports(runtime))
    assert checked[1] == amended[1]
    assert checked[0]["review_reminders"][0]["clause_check_recorded"] is True

    for label, (seconds, work) in measurements.items():
        print(f"{label}: {seconds:.6f}s {dict(work)}")
    admission = measurements["admission"][1]
    assert admission["legacy_sweeps"] + admission["full_observations"] <= 1, measurements
    for label in ("two_unchanged_prepares", "test_amendment", "public_check_amendment"):
        assert not any(measurements[label][1].values()), measurements
    edit = measurements["one_edit"][1]
    assert not any(edit[key] for key in ("legacy_sweeps", "full_observations", "legacy_defines"))
    assert edit["source_reads"] + edit["workspace_reads"] <= 3, measurements
    assert edit["path_resolutions"] <= 4, measurements
    assert edit["projection_paths"] <= 4 and edit["tree_pages"] <= 64, measurements
    assert edit["definition_scans"] <= 1, measurements
    assert measurements["two_unchanged_prepares"][0] < 2, measurements


@pytest.mark.parametrize("linked", [False, True])
def test_coverage_reports_match_legacy_and_resume_separate_events(tmp_path, monkeypatch, linked):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "cli.py").write_text("async def run():\r\n    pass\r\n")
    (root / "api.py").write_text("class\n\ndef serve():\n    pass\n")
    (root / "guards.py").write_text("def validate():\n    pass\n")
    (root / "settings.py").write_text("load: object\n")
    (root / "scrap.txt").write_bytes(b"\xff")
    if linked:
        (tmp_path / "outside.py").write_text("def validate():\n    pass\n")
        (root / "escape.py").symlink_to(tmp_path / "outside.py")
        (root / "alias.py").symlink_to("guards.py")
    consumers = _bind(root, tmp_path / "fence")

    _switches(monkeypatch, "off")
    legacy, legacy_events, legacy_binding = _runtime(root)
    _admit(legacy)
    baseline = _reports(legacy)
    base_assessment = {"reason_codes": ("other_evidence",), "detail": "Existing evidence"}
    baseline_verdict = false_completion.merge_precommit(legacy, base_assessment, admissible=True)
    _switches(monkeypatch, "on")
    current, current_events, current_binding = _runtime(root)
    _admit(current)
    assert _reports(current) == baseline
    assert false_completion.merge_precommit(
        current, base_assessment, admissible=True) == baseline_verdict
    assert current_events == legacy_events
    assert "escape.py" not in current_events[0]["start_snapshot"]
    assert baseline[0]["rows"][-1]["confidence"] == "CHECK_INCONCLUSIVE"

    with consumers.state.begin_write(("cli.py", "scrap.txt")):
        (root / "cli.py").write_text("def run():\n    return 1\n")
        (root / "scrap.txt").unlink()
    for binding in (legacy_binding, current_binding):
        binding.update(product_revision="edited", code_hash="edited")
    _switches(monkeypatch, "off")
    legacy_edited = _reports(legacy)
    _switches(monkeypatch, "on")
    assert _reports(current) == legacy_edited
    assert legacy_edited[0]["rows"][0]["diff_from_start"] is True

    resumed, events, binding = _runtime(root)
    events.extend(current_events)
    binding.update(current_binding)
    _admit(resumed, resume=object())
    assert events == current_events
    assert _reports(resumed) == legacy_edited


def test_coverage_preserves_start_path_normalization_after_fallback(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    ambiguous = root / "pkg\\cli.py"
    source = "def run():\n    pass\n"
    ambiguous.write_text(source)
    consumers = _bind(root, tmp_path / "fence")
    ticket = "Update the CLI in `pkg/cli.py`."
    results = []
    for mode in ("off", "on"):
        _switches(monkeypatch, mode)
        runtime, _, binding = _runtime(root, ticket)
        _admit(runtime)
        with consumers.state.begin_write(("pkg", ambiguous.name)):
            ambiguous.unlink()
            (root / "pkg").mkdir()
            (root / "pkg" / "cli.py").write_text(source)
        binding.update(product_revision="edited", code_hash="edited")
        results.append(_reports(runtime))
        with consumers.state.begin_write(("pkg", ambiguous.name)):
            (root / "pkg" / "cli.py").unlink()
            (root / "pkg").rmdir()
            ambiguous.write_text(source)
    assert results[0] == results[1]
    assert results[1][0]["rows"][0]["diff_from_start"] is False
