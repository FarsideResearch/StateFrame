"""Mutation authoring projection and physical-send accounting."""

from types import SimpleNamespace

import pytest


def test_authoring_tool_intersection_keeps_only_mutate_and_readers():
    from kernel_mutate_projection import authoring_tools

    tools = [{"function": {"name": name}} for name in
             ("done", "mutate", "read", "list_files", "grep", "shell")]
    selected = authoring_tools(tools)
    assert [tool["function"]["name"] for tool in selected] == [
        "mutate", "read", "list_files", "grep"]


def test_authoring_surface_keeps_only_live_gate_resolvers():
    from kernel_mutate_projection import authoring_dispatch_allowed, authoring_tools

    tools = [{"function": {"name": name}} for name in
             ("done", "edit", "mutate", "read", "run_tests", "read_check_receipt", "shell")]
    runtime = SimpleNamespace(pass_gates_vector=lambda _tools: SimpleNamespace(
        recoveries=({"tool": "run_tests", "arguments": {}},
                    {"tool": "read_check_receipt", "arguments": {}})))
    selected = authoring_tools(tools, runtime)
    names = [tool["function"]["name"] for tool in selected]
    assert names == ["mutate", "read", "run_tests", "read_check_receipt"]
    assert authoring_dispatch_allowed("run_tests", runtime, selected)
    assert authoring_dispatch_allowed("read_check_receipt", runtime, selected)
    for name in ("edit", "done", "shell", "file_create"):
        assert not authoring_dispatch_allowed(name, runtime, tools)
    assert not authoring_dispatch_allowed("run_tests", SimpleNamespace(
        pass_gates_vector=lambda _tools: SimpleNamespace(recoveries=())), tools)


def test_authoring_view_is_bounded_and_contains_goal():
    from kernel_mutate_projection import authoring_view
    from kernel_context import token_estimate

    runtime = SimpleNamespace(episode={"id": "e1", "goal": "parse report",
                                       "sends_left": 2, "draft": {"main.py": "x" * 10000}})
    text = authoring_view(runtime)
    assert "parse report" in text
    assert "e1" in text
    assert len(text) < 900
    assert token_estimate({"role": "system", "content": text}) <= 120


def test_authoring_view_hard_limit_with_full_draft():
    from kernel_mutate_projection import authoring_view
    from kernel_context import token_estimate

    runtime = SimpleNamespace(episode={"id": "e1", "goal": "g" * 512,
                                       "diagnostic": "d" * 512, "sends_left": 1,
                                       "draft": {"a.py": "a" * 10000, "b.py": "b" * 10000}})
    text = authoring_view(runtime)
    assert token_estimate({"role": "system", "content": text}) <= 120


def test_observing_hook_panel_keeps_bounded_512_byte_result():
    from kernel_mutate_projection import panel

    runtime = SimpleNamespace(level="authored", ready={}, last="none",
                              hook_lines=["x" * 512])
    text = panel(runtime)
    assert "x" * 512 in text
    assert len(text.encode()) < 800


def test_committed_tool_bridge_invokes_observer_and_watches_once_per_boundary():
    from kernel_mutate_projection import observe_committed_tool

    calls = []
    runtime = SimpleNamespace(
        last="", invoke_hook=lambda **kwargs: calls.append(("hook", kwargs)),
        tick=lambda parent, index: calls.append(("tick", parent, index)) or ["watch changed"])
    measurement = SimpleNamespace(_active_capture=SimpleNamespace(request_id="turn-1"))
    observe_committed_tool(runtime, measurement, "read src/a.py", 2, continue_turn=True)
    assert calls == [("hook", {"event": "post_tool", "facts": {
        "tool": "read", "status": "observed", "call_id": "turn-1"}}),
        ("tick", "turn-1", 2)]
    assert runtime.last == "watch changed"
    calls.clear()
    observe_committed_tool(runtime, measurement, "mutate", 3, continue_turn=False)
    assert calls == []


def test_native_tool_reply_bridge_covers_branches_without_measurement_observe():
    from kernel_mutate_projection import observe_tool_reply

    calls = []
    runtime = SimpleNamespace(last="", invoke_hook=lambda **kwargs: calls.append(kwargs),
                              tick=lambda *_args: [])
    measurement = SimpleNamespace(_active_capture=SimpleNamespace(request_id="turn-2"))
    messages = [{"role": "assistant", "tool_calls": [{"id": "call-2", "function": {
        "name": "file_create", "arguments": "{}"}}]}]
    seen = set()
    observe_tool_reply(runtime, measurement, messages, "call-2", 3, seen=seen,
                       continue_turn=True)
    observe_tool_reply(runtime, measurement, messages, "call-2", 3, seen=seen,
                       continue_turn=True)
    assert seen == {"call-2"}
    assert calls == [{"event": "post_tool", "facts": {
        "tool": "file_create", "status": "observed", "call_id": "call-2"}}]


def test_mutation_edit_receipts_invalidate_native_read_and_check_state():
    from kernel_mutate_projection import publish_mutation_edit_receipts

    calls = []
    runtime = SimpleNamespace(committed_edit_paths=lambda _call: ("src/a.py", "src/b.py"))
    measurement = SimpleNamespace(_note_applied_mutation=lambda paths, _outcome:
                                  calls.append(("invalidate", paths)))
    bolt = SimpleNamespace(receipt=lambda *args, **kwargs:
                           calls.append(("receipt", args, kwargs)))
    paths = publish_mutation_edit_receipts(runtime, "call-1", bolt, measurement)
    assert paths == ("src/a.py", "src/b.py")
    assert calls[0] == ("invalidate", paths)
    assert [entry[2]["target_path"] for entry in calls[1:]] == list(paths)
    assert all(entry[2]["changed_paths"] == paths for entry in calls[1:])


def test_send_reservation_is_idempotent_across_same_attempt(tmp_path):
    from kernel_mutate_projection import reserve_send

    journal = SimpleNamespace(rows=[], append=lambda *a, **k: journal.rows.append((a, k)))
    runtime = SimpleNamespace(episode={"id": "e1", "sends_left": 4},
                              mutation_journal=journal)
    assert reserve_send(runtime, "request-1", 1, 0)
    assert reserve_send(runtime, "request-1", 1, 0)
    assert len(journal.rows) == 1
    assert runtime.episode["sends_left"] == 3


def test_replayed_send_uses_durable_reservation_without_append():
    from kernel_mutate_projection import reserve_send

    journal = SimpleNamespace(events=lambda: [{"kind": "send_reserved", "episode": "e1",
                                                "request_id": "r1", "attempt": 1,
                                                "ordinal": 1, "key": "e1:r1:1:1"}],
                              append=lambda *_a, **_k: pytest.fail("replay appended a reservation"))
    runtime = SimpleNamespace(episode={"id": "e1", "sends_left": 4}, journal=journal)
    assert reserve_send(runtime, "r1", 2, 1, replay=True)
    assert runtime.episode["sends_left"] == 3
    assert reserve_send(runtime, "r1", 2, 1, replay=True)
    assert runtime.episode["sends_left"] == 3


def test_authoring_surface_refuses_broad_callable_contract():
    from kernel_mutate_projection import require_authoring_surface

    capture = SimpleNamespace(request_action_contract={
        "callable_actions": ["mutate", "shell"]})
    with pytest.raises(RuntimeError, match="MUTATION_FRAME"):
        require_authoring_surface(capture)
    capture.request_action_contract = {"callable_actions": ["mutate", "read"]}
    require_authoring_surface(capture)


def test_stale_effect_call_is_refused_in_episode():
    from kernel_mutate_projection import authoring_dispatch_allowed

    assert authoring_dispatch_allowed("read")
    assert not authoring_dispatch_allowed("shell")


def test_open_episode_pins_edit_route(monkeypatch):
    import kernel_frame_routes
    import kernel_mutate_runtime

    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    monkeypatch.setattr(kernel_mutate_runtime, "service",
                        lambda: SimpleNamespace(episode={"id": "e1"}))
    assert kernel_frame_routes.role_for_request(request_kind="decide2") == "edit"


def test_view_expires_last_authoring_send_before_next_transport(tmp_path, monkeypatch):
    import kernel_mutate_runtime
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime
    from kernel_stateframe_runtime import StateFrameRuntime

    runtime = MutationRuntime(level="authored", journal=Journal(tmp_path))
    runtime.episode = {"id": "episode-1", "sends_left": 0}
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "authored")
    monkeypatch.setattr(kernel_mutate_runtime, "service", lambda: runtime)
    seen = []
    owner = SimpleNamespace(stateframe_f4_mode="on", _f4_view=lambda prefix, *_a, **_k:
                            seen.append(prefix) or ("parent", {}),
                            _record_recall_render_result=lambda *_a: None)
    assert StateFrameRuntime._view(owner, [], {}, "", "", [], provider_request=False) == ("parent", {})
    assert runtime.episode is None
    assert "FAILED authoring sends exhausted" in runtime.last
    assert "FAILED authoring sends exhausted" in seen[0][-1]["content"]


def _mutation_proposal_runtime(repo, *, network="none", write_globs=(), read_globs=(),
                               catalog_root=None, ordinary_read_globs=None):
    from pathlib import Path
    from kernel_mutate_policy import MutationPolicy
    from kernel_mutate_records import Journal
    from kernel_mutate_runtime import MutationRuntime

    package_catalog = Path(__file__).resolve().parents[1] / "src/stateframe/mutations_prebuilt"
    return MutationRuntime(
        level="prebuilt", journal=Journal(repo), repo=repo,
        policy=MutationPolicy(level="prebuilt", network=network, write_globs=write_globs,
                              read_globs=read_globs),
        catalog_root=catalog_root or package_catalog,
        ordinary_read_globs=ordinary_read_globs)


def _host_args(frame, unit):
    import json

    host = next(line for line in frame.splitlines() if line.startswith("Host:"))
    return json.loads(host.split(f"mutate use {unit} ", 1)[1].split(" or decline", 1)[0])


def test_proposal_call_shape_is_engine_accepted_and_binds_trigger_file_for_each_reader(
        tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path
    import sys
    import kernel_mutate_build
    import kernel_mutate_qualify
    from kernel_mutate_projection import panel
    from kernel_mutate_wire import validate_schema

    monkeypatch.setattr(sys, "platform", "linux")
    catalog_root = Path(__file__).resolve().parents[1] / "src/stateframe/mutations_prebuilt"
    catalog = json.loads((catalog_root / "manifest.json").read_text(encoding="utf-8"))
    manifests = {name: json.loads((catalog_root / name / "unit.json").read_text(encoding="utf-8"))
                 for name in catalog["units"]}
    proposed = 0
    proposed_units = set()
    reader_names = {name for name, manifest in manifests.items() if manifest.get("kind") == "reader"}
    assert reader_names == {
        "archive_list", "file_facts", "html_extract", "snapshot_diff",
        "sqlite_peek", "structured_peek", "text_grep_struct",
    }
    for unit, manifest in sorted(manifests.items()):
        if manifest.get("kind") != "reader" or not manifest.get("formats"):
            continue
        required = manifest["input_schema"].get("required", [])
        assert sum(isinstance(key, str) and key.endswith("_id") for key in required) == 1, unit
        unique_formats = [candidate for candidate in manifest["formats"]
                          if sum(candidate in row.get("formats", ())
                                 for row in manifests.values()
                                 if row.get("kind") == "reader") == 1]
        assert unique_formats, f"{unit} has no unique declared format"
        format_name = unique_formats[0]
        workspace = tmp_path / unit
        workspace.mkdir()
        filename = f"input.{format_name}"
        (workspace / filename).write_bytes(b"sample")
        original_digest = hashlib.sha256((workspace / filename).read_bytes()).hexdigest()
        runtime = _mutation_proposal_runtime(workspace)
        executor_root = tmp_path / f"executor-{unit}"
        (executor_root / "in").mkdir(parents=True)
        runtime.executor = SimpleNamespace(root=executor_root)
        monkeypatch.setattr(kernel_mutate_build, "admit", lambda staged, *_a, **_k:
                            SimpleNamespace(admission_digest="a" * 64))
        monkeypatch.setattr(kernel_mutate_qualify, "_binding", lambda: {
            "closure_digest": "closure", "sdk_digest": "sdk"})
        launches = []

        class ReachedLaunch(Exception):
            pass

        def stop_at_launch(spec):
            # _use reaches LaunchSpec only after envelope, schema, and input-binding checks.
            launches.append(spec)
            raise ReachedLaunch

        runtime._launch = stop_at_launch
        task = f"Inspect {filename}"
        proposal = runtime.propose_prebuilt(task)
        assert proposal is not None and proposal["unit"] == unit
        frame = panel(runtime, task=task)
        args = _host_args(frame, unit)
        assert args["inputs"] == [filename]
        assert args["input"]["input_id"] == "in0"
        validate_schema(args["input"], manifest["input_schema"])
        if "format" in manifest["input_schema"].get("properties", {}):
            assert args["input"]["format"] == format_name
        else:
            assert "format" not in args["input"]
        description = manifest["summary"].splitlines()[0].strip()[:160]
        if not description.endswith("."):
            description += "."
        assert frame.splitlines() == [
            f"ready: {unit} — {description}",
            f"ready inputs: in0={filename}",
            f"Host: {unit} fits this task. Call mutate use {unit} "
            f"{json.dumps(args, ensure_ascii=True, separators=(',', ':'))} or decline (say why).",
        ]
        with pytest.raises(ReachedLaunch):
            runtime.dispatch({"op": "use", "id": unit,
                              "args": json.dumps(args, ensure_ascii=True, separators=(",", ":"))},
                             f"proposal-engine-use-{unit}")
        assert len(launches) == 1
        stdin_doc = json.loads(launches[0].stdin_doc)
        expected_input = dict(args["input"])
        if unit == "sqlite_peek":
            expected_input["wal_present"] = False
        assert stdin_doc["input"] == expected_input
        assert stdin_doc["inputs"] == [{"id": "in0", "path": "/in/in0",
                                        "bytes": len(b"sample"), "name": filename}]
        assert launches[0].inputs["in0"].read_bytes() == b"sample"
        assert hashlib.sha256((workspace / filename).read_bytes()).hexdigest() == original_digest
        proposed += 1
        proposed_units.add(unit)
    assert proposed_units == {
        "archive_list", "file_facts", "sqlite_peek", "structured_peek", "text_grep_struct"
    }
    assert proposed == len(proposed_units)


def _write_proposal_catalog(root, names, *, network, writes=False):
    import copy
    import json
    from pathlib import Path
    from kernel_mutate_build import source_digest

    source_catalog = Path(__file__).resolve().parents[1] / "src/stateframe/mutations_prebuilt"
    source_dir = source_catalog / "structured_peek"
    source_manifest = json.loads((source_dir / "unit.json").read_text(encoding="utf-8"))
    files = {path.name: path.read_bytes() for path in source_dir.glob("*.py")}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    units = {}
    for name in names:
        manifest = copy.deepcopy(source_manifest)
        manifest["name"] = name
        manifest["requires"] = {"network": network, "writes": writes}
        unit_dir = root / name
        unit_dir.mkdir()
        (unit_dir / "main.py").write_bytes(files["main.py"])
        (unit_dir / "unit.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        units[name] = source_digest(files, manifest)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "stateframe-mutation-prebuilt/1", "units": units
    }, indent=2) + "\n", encoding="utf-8")
    return root


def test_proposal_requires_admitted_extension_grants_and_unambiguous_reader(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sales.csv").write_text("day,units\nMon,1\n", encoding="utf-8")
    (workspace / "notes.txt").write_text("notes\n", encoding="utf-8")
    network_catalog = _write_proposal_catalog(
        tmp_path / "network_catalog", ("network_reader",), network=True)

    no_grant = _mutation_proposal_runtime(
        workspace, network="none", catalog_root=network_catalog)
    assert no_grant.propose_prebuilt("Summarize sales.csv") is None
    with_grant = _mutation_proposal_runtime(
        workspace, network="public-web", catalog_root=network_catalog)
    proposal = with_grant.propose_prebuilt("Summarize sales.csv")
    assert proposal is not None and proposal["unit"] == "network_reader"
    assert with_grant.propose_prebuilt("Summarize notes.txt") is None
    assert with_grant.propose_prebuilt("Summarize missing.csv") is None
    assert with_grant.propose_prebuilt("Summarize an unmentioned file") is None
    read_scope_blocked = _mutation_proposal_runtime(
        workspace, network="public-web", read_globs=("notes.txt",), catalog_root=network_catalog)
    assert read_scope_blocked.propose_prebuilt("Summarize sales.csv") is None

    ambiguous_catalog = _write_proposal_catalog(
        tmp_path / "ambiguous_catalog", ("reader_one", "reader_two"), network=False)
    ambiguous = _mutation_proposal_runtime(workspace, catalog_root=ambiguous_catalog)
    assert ambiguous.propose_prebuilt("Summarize sales.csv") is None

    write_catalog = _write_proposal_catalog(
        tmp_path / "write_catalog", ("writer_reader",), network=False, writes=True)
    no_write_grant = _mutation_proposal_runtime(workspace, catalog_root=write_catalog)
    assert no_write_grant.propose_prebuilt("Summarize sales.csv") is None
    with_write_grant = _mutation_proposal_runtime(
        workspace, catalog_root=write_catalog, write_globs=("sales.csv",))
    proposal = with_write_grant.propose_prebuilt("Summarize sales.csv")
    assert proposal is not None and proposal["unit"] == "writer_reader"


def test_proposal_journal_event_is_small_and_idempotent(tmp_path):
    (tmp_path / "sales.csv").write_text("day,units\nMon,1\n", encoding="utf-8")
    runtime = _mutation_proposal_runtime(tmp_path)
    task = "Summarize sales.csv"
    first = runtime.propose_prebuilt(task)
    second = runtime.propose_prebuilt(task)
    assert first is not None and first["unit"] == "structured_peek"
    assert second is not None and second["unit"] == "structured_peek"
    events = [row for row in runtime.journal.events() if row.get("kind") == "prebuilt_proposed"]
    assert events == [{"kind": "prebuilt_proposed", "unit": "structured_peek",
                       "trigger_file": "sales.csv", "input_id": "in0"}]


def test_mutation_panel_without_trigger_keeps_original_bytes(tmp_path):
    from kernel_mutate_projection import panel

    runtime = _mutation_proposal_runtime(tmp_path)
    rendered = panel(runtime, task="Explain the test setup")
    assert rendered == panel(runtime)
    assert rendered == "MUTATE mutations: prebuilt | ready: none | last: none"


def test_stateframe_view_passes_the_current_task_to_the_proposal_panel(tmp_path, monkeypatch):
    import kernel_mutate_runtime
    from kernel_mutate_projection import panel
    from kernel_stateframe_runtime import StateFrameRuntime

    (tmp_path / "sales.csv").write_text("day,units\nMon,1\n", encoding="utf-8")
    runtime = _mutation_proposal_runtime(tmp_path)
    captured = []
    owner = SimpleNamespace(
        owner=SimpleNamespace(task_row={"msg": "Inspect sales.csv"}),
        stateframe_f4_mode="on",
        _f4_view=lambda prefix, *_a, **_k: captured.extend(prefix) or ("parent", {}),
        _record_recall_render_result=lambda *_a: None,
    )
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "prebuilt")
    monkeypatch.setattr(kernel_mutate_runtime, "service", lambda: runtime)

    assert StateFrameRuntime._view(owner, [], {}, "", "", [], provider_request=False) == ("parent", {})
    rendered = captured[-1]["content"]
    assert rendered == panel(runtime, task="Inspect sales.csv")
    assert "Host: structured_peek fits this task." in rendered


def test_stateframe_view_uses_current_step_text_to_trigger_proposal(tmp_path, monkeypatch):
    import kernel_mutate_runtime
    from kernel_stateframe_runtime import StateFrameRuntime

    (tmp_path / "sales.csv").write_text("day,units\nMon,1\n", encoding="utf-8")
    runtime = _mutation_proposal_runtime(tmp_path)
    captured = []
    owner = SimpleNamespace(
        owner=SimpleNamespace(task_row={"msg": "Analyze the latest sales"}),
        stateframe_f4_mode="on",
        last_user_ask=lambda: {"content": "Inspect sales.csv and summarize its structure"},
        _f4_view=lambda prefix, *_a, **_k: captured.extend(prefix) or ("parent", {}),
        _record_recall_render_result=lambda *_a: None,
    )
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "prebuilt")
    monkeypatch.setattr(kernel_mutate_runtime, "service", lambda: runtime)

    assert StateFrameRuntime._view(owner, [], {}, "", "", [], provider_request=False) == ("parent", {})
    rendered = captured[-1]["content"]
    assert "ready: structured_peek" in rendered
    assert "ready inputs: in0=sales.csv" in rendered


def test_mutations_off_keeps_request_bytes_identical_with_matching_input(tmp_path, monkeypatch):
    import kernel_mutate_runtime
    from kernel_measurement import canonical_bytes
    from kernel_stateframe_runtime import StateFrameRuntime

    (tmp_path / "sales.csv").write_text("day,units\nMon,1\n", encoding="utf-8")
    runtime = _mutation_proposal_runtime(tmp_path)
    prefix = [{"role": "system", "content": "stable prefix"}]
    frame = {"state": "unchanged"}
    owner = SimpleNamespace(
        stateframe_f4_mode="on",
        _f4_view=lambda actual, value, *_a, **_k: (
            canonical_bytes({"prefix": actual, "frame": value}), {}),
        _record_recall_render_result=lambda *_a: None,
    )
    expected = (canonical_bytes({"prefix": prefix, "frame": frame}), {})
    monkeypatch.setenv("STATEFRAME_MUTATIONS", "off")
    monkeypatch.setattr(kernel_mutate_runtime, "service", lambda: runtime)

    actual = StateFrameRuntime._view(owner, prefix, frame, "", "", [], provider_request=False)
    assert actual == expected
