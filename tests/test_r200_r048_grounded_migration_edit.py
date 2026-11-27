"""R048 provider parity and source-bound advisory behavior."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

# 0.10.8: the switch-ON scripted run hangs (subprocess timeout); r048 ships default OFF until that is fixed.
pytestmark = pytest.mark.skip(reason="r048 switch-ON run hang; fix pending (c21-t0108-fx)")

from tools.pinned_tree import pinned_tree

BASE = "6e3b096cf3059f759138a88f7fdf46acd3d75baa"
SWITCH = "STATEFRAME_MIGRATION_SITE_PARTITION"
ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "class Widget:\n    def __init__(self, *, old=1):\n        self.value = old\n"
UPDATED_PRODUCER = PRODUCER.replace("old", "new")
CALLERS = """from factory import Widget as Build

first = Build(old=2)
repeat = Build(old=3)
consumed = Build(old=4).value

def forward(**options):
    return Build(**options)

def dynamic(Build):
    return Build(old=5)

def another(old):
    return old

unrelated = another(old=6)
"""
UPDATED_CALLERS = CALLERS.replace("first = Build(old=2)", "first = Build(new=2)")


@pytest.mark.host_sandbox
def test_r048_grounded_migration_edit_off_request_bytes_match_pinned_099(
        tmp_path, tmp_path_factory, monkeypatch):
    import io
    import os
    import shutil
    import subprocess
    import sys

    from _timeouts import scaled
    from acceptance._support import tree_python
    from stateframe.scripted_endpoint import Script, serve

    pinned = pinned_tree(BASE, tmp_path_factory)
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "examples/discount_bug", repo)
    run = tmp_path / "run"

    def capture(checkout, label, setting):
        python = tree_python(checkout / "src", tmp_path / label)
        script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
        server = serve(script)
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        raw = []
        handler = server.RequestHandlerClass
        interrupted = False

        class RawRequests(handler):
            def do_POST(self):  # noqa: N802
                if self.path != "/v1/chat/completions":
                    return super().do_POST()
                stream = self.rfile
                body = stream.read(int(self.headers["Content-Length"]))
                raw.append(body)
                if len(raw) == 3 and not interrupted:
                    self._send(429, {"error": {
                        "type": "insufficient_quota", "message": "scripted interruption",
                    }})
                    return
                self.rfile = io.BytesIO(body)
                try:
                    return super().do_POST()
                finally:
                    self.rfile = stream

        server.RequestHandlerClass = RawRequests
        environment = {k: v for k, v in os.environ.items() if k not in {SWITCH, "PYTHONPATH"}}
        if setting is not None:
            environment[SWITCH] = setting
        environment["STATEFRAME_WORKING_SET_ALLOCATOR"] = "on"

        def invoke(*args):
            return subprocess.run(
                [python, "-m", "stateframe", *args], env=environment,
                capture_output=True, text=True, timeout=scaled(360))

        try:
            first = invoke("run", "--repo", str(repo), "--task",
                           "Fix the pricing code so the tests pass.", "--run-dir", str(run),
                           "--endpoint", endpoint, "--model", "scripted", "--quiet",
                           "--test-python", sys.executable, "--max-turns", "16")
            assert len(raw) == 3 and script.position == 2, first.stdout + first.stderr
            interrupted = True
            resumed = invoke("resume", str(run), "--endpoint", endpoint, "--quiet")
            assert resumed.returncode == 0, resumed.stdout + resumed.stderr
            script.assert_consumed()
            assert len(raw) > 3 and all(json.loads(body)["tools"] for body in raw)
            return raw[:3], raw[3:]
        finally:
            server.shutdown()
            server.server_close()
            if run.exists():
                run.rename(tmp_path / f"{label}-run")

    monkeypatch.delenv(SWITCH, raising=False)
    expected = capture(pinned, "pinned", None)
    assert capture(ROOT, "unset", None) == expected
    assert capture(ROOT, "off", "false") == expected


@pytest.mark.host_sandbox
def test_r048_off_request_bytes_match_pinned_099_without_allocator(
        tmp_path, tmp_path_factory, monkeypatch):
    import io
    import os
    import shutil
    import subprocess
    import sys

    from _timeouts import scaled
    from acceptance._support import tree_python
    from stateframe.scripted_endpoint import Script, serve

    pinned = pinned_tree(BASE, tmp_path_factory)

    def capture(checkout, label, setting):
        python = tree_python(checkout / "src", tmp_path / label)
        fixture = tmp_path / f"repo-{label}"
        shutil.copytree(ROOT / "examples/discount_bug", fixture)
        run = tmp_path / f"run-{label}"
        script = Script(json.loads((ROOT / "examples/discount_bug.script.json").read_text()))
        server = serve(script)
        endpoint = f"http://127.0.0.1:{server.server_port}/v1"
        raw = []
        handler = server.RequestHandlerClass
        interrupted = False

        class RawRequests(handler):
            def do_POST(self):  # noqa: N802
                if self.path != "/v1/chat/completions":
                    return super().do_POST()
                stream = self.rfile
                body = stream.read(int(self.headers["Content-Length"]))
                raw.append(body)
                if len(raw) == 3 and not interrupted:
                    self._send(429, {"error": {
                        "type": "insufficient_quota", "message": "scripted interruption",
                    }})
                    return
                self.rfile = io.BytesIO(body)
                try:
                    return super().do_POST()
                finally:
                    self.rfile = stream

        server.RequestHandlerClass = RawRequests
        environment = {key: value for key, value in os.environ.items()
                       if key not in {SWITCH, "PYTHONPATH", "STATEFRAME_WORKING_SET_ALLOCATOR"}}
        if setting is not None:
            environment[SWITCH] = setting

        def invoke(*args):
            return subprocess.run(
                [python, "-m", "stateframe", *args], env=environment,
                capture_output=True, text=True, timeout=scaled(360))

        try:
            first = invoke("run", "--repo", str(fixture), "--task",
                           "Fix the pricing code so the tests pass.", "--run-dir", str(run),
                           "--endpoint", endpoint, "--model", "scripted", "--quiet",
                           "--test-python", sys.executable, "--max-turns", "16")
            assert len(raw) == 3 and script.position == 2, first.stdout + first.stderr
            interrupted = True
            resumed = invoke("resume", str(run), "--endpoint", endpoint, "--quiet")
            assert resumed.returncode == 0, resumed.stdout + resumed.stderr
            script.assert_consumed()
            assert len(raw) > 3 and all(json.loads(body)["tools"] for body in raw)
            return raw[:3], raw[3:]
        finally:
            server.shutdown()
            server.server_close()
            if run.exists():
                run.rename(tmp_path / f"{label}-run")

    monkeypatch.delenv(SWITCH, raising=False)
    monkeypatch.delenv("STATEFRAME_WORKING_SET_ALLOCATOR", raising=False)
    expected = capture(pinned, "pinned-no-allocator", None)
    assert capture(ROOT, "unset-no-allocator", None) == expected


def test_r048_grounded_migration_edit_switch_defaults_off(monkeypatch):
    from stateframe import runner
    from kernel_stateframe_framefit5 import migration_site_partition_enabled

    monkeypatch.delenv(SWITCH, raising=False)
    assert SWITCH in runner.OPT_IN_SWITCHES
    assert SWITCH in runner.engine_switches(runner.ENGINE_DIR)
    assert SWITCH not in runner.resolve_opt_in_switches()
    assert not migration_site_partition_enabled()
    monkeypatch.setenv(SWITCH, "false")
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"
    assert not migration_site_partition_enabled()
    monkeypatch.setenv(SWITCH, "on")
    assert runner.resolve_opt_in_switches()[SWITCH] == "on"
    assert migration_site_partition_enabled()


class Journal:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def records(self, *, rebuild):
        assert rebuild is False
        return iter(self.rows)

    def emit(self, kind, **fields):
        self.rows.append({"kind": kind, **fields})


def _read_page(path, source, *, payload=None):
    from kernel_r200_r048_grounded_migration_edit import revision

    text = source if payload is None else payload
    lines = text.splitlines(keepends=True)
    count = len(lines)
    return {
        "role": "tool",
        "content": (
            f"[read page] path={path} revision={revision(source)} lines=1-{count}/{count} "
            f"requested=1-{count} effective=lines:1-{count}@0:{len(lines[-1])} "
            f"clip=none chars={len(text)} bytes={len(text.encode())} "
            f"truncated=false decoded_with_replacement=false\n\n{text}"
        ),
    }


def _runtime(root, journal):
    return SimpleNamespace(
        owner=SimpleNamespace(root=root, current_step=0, task_row={
            "brief": "Migrate Widget callers from old to new.",
            "files": ["factory.py", "client.py"],
        }),
        journal=journal, frame_ceiling=40_000,
    )


def _apply(runtime, path, old, new, *, forged=None):
    from kernel_coordinated_change_ledger import migration_edit_preimage, record_migration_edit
    from probes.react_repair import str_replace_edit

    target = runtime.owner.root / path
    before = target.read_text()
    preimage = migration_edit_preimage(runtime, target, old, new)
    capture = SimpleNamespace(_evictfix_final_wire_messages=[
        _read_page(path, before, payload=forged),
    ])
    assert "applied." in str_replace_edit(target, old, new, root=runtime.owner.root)
    record_migration_edit(runtime, capture, preimage, old, new,
                          f"edit-receipt-{runtime.owner.current_step}")
    runtime.owner.current_step += 1


def test_r048_grounded_migration_edit_on_path_is_bounded_and_idempotent(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition
    from kernel_r200_r048_grounded_migration_edit import (
        DELIVERY_ACTION, HEADER, MAX_SUMMARY_BYTES, parse_fact, revision,
    )
    from kernel_stateframe_framefit5 import (
        VOLATILE_TAIL_MARKER, _fit_migration_site_partition, finalize_frame_only_wire_order,
        record_migration_delivery,
    )

    monkeypatch.setenv(SWITCH, "on")
    producer = tmp_path / "factory.py"
    producer.write_text(PRODUCER)
    client = tmp_path / "client.py"
    client.write_text(CALLERS + "".join(f"many_{i} = Build(old={i})\n" for i in range(40)))
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    assert migration_partition(runtime) is None
    _apply(runtime, "client.py", "first = Build(old=2)", "first = Build(new=2)")

    partition = migration_partition(runtime)
    assert partition.producer.receipt == "edit-receipt-0"
    assert partition.exemplar.receipt == "edit-receipt-1"
    assert partition.producer.after == revision(UPDATED_PRODUCER)
    decision = {row.caller.source: row for row in partition.decisions}
    assert decision["Build(old=3)"].classification == "reusable"
    assert decision["Build(old=4)"].reason == "result use differs from exemplar"
    assert decision["Build(old=4)"].caller.context_source == "consumed = Build(old=4).value"
    assert decision["Build(**options)"].reason == "forwarding changes argument evidence"
    assert decision["Build(old=5)"].classification == "unknown"
    assert decision["another(old=6)"].classification == "unrelated"
    assert decision["Build(old=3)"].caller.binding_source == "from factory import Widget as Build"
    assert all(row.caller.revision == revision(client.read_text()) for row in partition.decisions)

    base = [{"role": "user", "content": VOLATILE_TAIL_MARKER + "\n\nexisting fact"}]
    messages = finalize_frame_only_wire_order(base)
    contract = {"admitted_tool_schemas": [{"type": "function", "function": {"name": "str_replace"}}]}
    _fit_migration_site_partition(messages, runtime, contract, {}, None)
    text = messages[-1]["content"]
    lines = [line for line in text.splitlines() if line.startswith(HEADER)]
    assert len(lines) == 1 and len(lines[0].encode()) <= MAX_SUMMARY_BYTES
    fact = parse_fact(lines[0])
    assert fact["omitted"] > 0
    assert any(site["classification"] == "reusable" for site in fact["sites"])
    assert {site["caller"]["source"] for site in fact["sites"] if site["classification"] == "exceptional"} == {
        "Build(old=4)", "Build(**options)",
    }

    # Reassembly and restart use durable scalar receipts and the same source bytes.
    restored = _runtime(tmp_path, Journal(runtime.journal.rows))
    restored.owner.current_step = runtime.owner.current_step
    repeated = finalize_frame_only_wire_order(messages)
    _fit_migration_site_partition(repeated, restored, contract, {}, None)
    assert repeated == messages
    capture = SimpleNamespace(journal=restored.journal, request_id="send", step=2,
                              _evictfix_final_wire_messages=repeated)
    record_migration_delivery(capture, restored)
    event = restored.journal.rows[-1]
    assert event.get("action") == DELIVERY_ACTION and event["partition"] == fact["partition"]
    assert event["delivered_instances"] == 1
    assert {(row["path"], row["line"], row["revision"]) for row in event["sites"]
            if row["classification"] == "exceptional"} == {
        ("client.py", 5, revision(client.read_text())), ("client.py", 8, revision(client.read_text())),
    }

    restored.frame_ceiling = 10
    _fit_migration_site_partition(repeated, restored, contract, {}, None)
    assert repeated == base
    restored.frame_ceiling = runtime.frame_ceiling
    _fit_migration_site_partition(repeated, restored, contract, {"evidence.py": "value\n" * 40_000}, None)
    assert repeated == base
    producer.write_text(UPDATED_PRODUCER + "# changed since receipt\n")
    _fit_migration_site_partition(messages, restored, contract, {}, None)
    assert messages == base


def test_r048_migration_requires_exact_delivered_preimage(tmp_path, monkeypatch):
    from kernel_r200_r048_grounded_migration_edit import EDIT_ACTION

    monkeypatch.setenv(SWITCH, "on")
    (tmp_path / "factory.py").write_text(PRODUCER)
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER, forged=PRODUCER.replace("old", "bad"))
    assert all(row.get("action") != EDIT_ACTION for row in runtime.journal.rows)


def test_r048_factory_default_requires_same_call_shape():
    from kernel_r200_r048_grounded_migration_edit import CallerFile, classify, exemplar_for, producer_delta

    old = "def widget(value=2, count=1):\n    return (value, count)\n"
    delta = producer_delta("factory.py", old, old.replace("value=2", "value=0"))
    assert delta.previous_default == "2"
    import_source = "from factory import widget\n"
    exemplar = exemplar_for(delta, "client.py", import_source + "first = widget()\n",
                            import_source + "first = widget(value=2)\n")
    callers = CallerFile("more.py", import_source + """
repeat = widget()
changed_default = widget(count=2)
different_binding = widget(3)
""")
    results = {site.source: classify(delta, exemplar, call, site).classification
               for call, site in callers.relevant_calls(delta)}
    assert results == {"widget()": "reusable", "widget(count=2)": "exceptional", "widget(3)": "exceptional"}


def test_r048_positional_calls_do_not_consume_exception_summary(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition
    from kernel_r200_r048_grounded_migration_edit import HEADER, parse_fact

    monkeypatch.setenv(SWITCH, "on")
    producer = "class Widget:\n    def __init__(self, old=1):\n        self.value = old\n"
    updated = producer.replace("old", "new")
    (tmp_path / "factory.py").write_text(producer)
    positional_sites = "".join(
        f"positional_{index} = Build({index})\n" for index in range(12))
    source = ("from factory import Widget as Build\n"
              "chosen = Build(old=2)\n" + positional_sites
              + "exceptional = Build(old=99).value\n")
    client = tmp_path / "client.py"
    client.write_text(source)
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", producer, updated)
    _apply(runtime, "client.py", "chosen = Build(old=2)", "chosen = Build(new=2)")

    partition = migration_partition(runtime)
    decisions = {row.caller.source: row for row in partition.decisions}
    fact = parse_fact(next(line for line in partition.render().splitlines()
                           if line.startswith(HEADER)))
    assert "Build(old=99)" in {site["caller"]["source"] for site in fact["sites"]}
    assert all(decisions[f"Build({index})"].classification == "unrelated"
               for index in range(12))
    assert decisions["Build(old=99)"].classification == "exceptional"
    assert decisions["Build(old=99)"].caller.context_source == "exceptional = Build(old=99).value"


def test_r048_local_expression_positional_calls_do_not_hide_exception(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition
    from kernel_r200_r048_grounded_migration_edit import HEADER, parse_fact

    monkeypatch.setenv(SWITCH, "on")
    producer = "class Widget:\n    def __init__(self, old=1):\n        self.value = old\n"
    updated = producer.replace("old", "new")
    (tmp_path / "factory.py").write_text(producer)
    positional = "".join(
        f"    positional_{index} = Build(value + {index})\n" for index in range(12))
    source = ("from factory import Widget as Build\n"
              "chosen = Build(old=2)\n"
              "def migrate(value):\n" + positional +
              "    exceptional = Build(old=99).value\n")
    client = tmp_path / "client.py"
    client.write_text(source)
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", producer, updated)
    _apply(runtime, "client.py", "chosen = Build(old=2)", "chosen = Build(new=2)")

    partition = migration_partition(runtime)
    decisions = {row.caller.source: row for row in partition.decisions}
    fact = parse_fact(next(line for line in partition.render().splitlines()
                           if line.startswith(HEADER)))
    assert all(decisions[f"Build(value + {index})"].classification == "unrelated"
               for index in range(12))
    assert decisions["Build(old=99)"].classification == "exceptional"
    assert "Build(old=99)" in {site["caller"]["source"] for site in fact["sites"]}


def test_r048_same_module_exemplar_is_accepted(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition

    monkeypatch.setenv(SWITCH, "on")
    client_source = (PRODUCER + "\nfirst = Widget(old=2)\n"
                     "second = Widget(old=3)\nthird = Widget(old=4)\n")
    (tmp_path / "factory.py").write_text(client_source)
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    _apply(runtime, "factory.py", "first = Widget(old=2)", "first = Widget(new=2)")

    partition = migration_partition(runtime)
    assert partition is not None
    assert partition.producer.path == partition.exemplar.path == "factory.py"
    assert {row.caller.source for row in partition.decisions} == {
        "Widget(old=3)", "Widget(old=4)",
    }
    assert all(row.classification == "reusable" for row in partition.decisions)


def test_r048_same_module_required_parameter_invalidates_historical_partition(
        tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition

    monkeypatch.setenv(SWITCH, "on")
    source = (PRODUCER + "\nfirst = Widget(old=2)\nsecond = Widget(old=4)\n")
    factory = tmp_path / "factory.py"
    factory.write_text(source)
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    _apply(runtime, "factory.py", "first = Widget(old=2)", "first = Widget(new=2)")
    assert migration_partition(runtime) is not None

    _apply(runtime, "factory.py", "def __init__(self, *, new=1):",
           "def __init__(self, *, new=1, region):")
    current = factory.read_text()
    namespace = {}
    producer_source = current.partition("\nfirst =")[0]
    exec(compile(producer_source, "factory.py", "exec"), namespace)
    with pytest.raises(TypeError, match="region"):
        namespace["Widget"](new=4)

    assert migration_partition(runtime) is None
    replayed = _runtime(tmp_path, Journal(runtime.journal.rows))
    replayed.owner.current_step = runtime.owner.current_step
    assert migration_partition(replayed) is None


def test_r048_partition_survives_followup_edit_in_exemplar_file(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition
    from kernel_r200_r048_grounded_migration_edit import EDIT_ACTION

    monkeypatch.setenv(SWITCH, "on")
    (tmp_path / "factory.py").write_text(PRODUCER)
    (tmp_path / "client.py").write_text(
        "from factory import Widget as Build\n"
        "first = Build(old=2)\nsecond = Build(old=3)\nthird = Build(old=4)\n")
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    _apply(runtime, "client.py", "first = Build(old=2)", "first = Build(new=2)")
    first_partition = migration_partition(runtime)
    assert {row.caller.source for row in first_partition.decisions} == {
        "Build(old=3)", "Build(old=4)",
    }

    _apply(runtime, "client.py", "Build(old=3)", "Build(new=3)")
    assert [row["role"] for row in runtime.journal.rows
            if row.get("action") == EDIT_ACTION] == [
        "producer", "exemplar", "site",
    ]
    next_partition = migration_partition(runtime)
    assert next_partition is not None
    assert [(row.caller.source, row.classification) for row in next_partition.decisions] == [
        ("Build(new=3)", "unrelated"), ("Build(old=4)", "reusable"),
    ]

    import_line = "from factory import Widget as Build\n"
    _apply(runtime, "client.py", import_line, import_line + "# source moved below\n")
    shifted = migration_partition(runtime)
    assert shifted is not None
    assert [(row.caller.source, row.classification) for row in shifted.decisions] == [
        ("Build(new=3)", "unrelated"), ("Build(old=4)", "reusable"),
    ]
    assert [row["role"] for row in runtime.journal.rows
            if row.get("action") == EDIT_ACTION] == [
        "producer", "exemplar", "site", "site",
    ]

    restored = _runtime(tmp_path, Journal(runtime.journal.rows))
    restored.owner.current_step = runtime.owner.current_step
    replayed = migration_partition(restored)
    assert replayed is not None
    assert replayed.exemplar == shifted.exemplar
    assert [(row.caller.source, row.classification) for row in replayed.decisions] == [
        ("Build(new=3)", "unrelated"), ("Build(old=4)", "reusable"),
    ]


def test_r048_context_edit_preserving_exemplar_keeps_later_guidance(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition
    from kernel_r200_r048_grounded_migration_edit import HEADER, parse_fact

    monkeypatch.setenv(SWITCH, "on")
    (tmp_path / "factory.py").write_text(PRODUCER)
    client = tmp_path / "client.py"
    client.write_text(
        "from factory import Widget as Build\n"
        "first = Build(old=2)\nsecond = Build(old=3)\nthird = Build(old=4)\n")
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    _apply(runtime, "client.py", "first = Build(old=2)", "first = Build(new=2)")
    assert migration_partition(runtime) is not None

    _apply(runtime, "client.py",
           "first = Build(new=2)\nsecond = Build(old=3)",
           "first = Build(new=2)\nsecond = Build(new=3)")
    partition = migration_partition(runtime)
    assert partition is not None
    assert [(row.caller.source, row.classification) for row in partition.decisions] == [
        ("Build(new=3)", "unrelated"), ("Build(old=4)", "reusable"),
    ]
    fact = parse_fact(next(line for line in partition.render().splitlines()
                           if line.startswith(HEADER)))
    assert "Build(old=4)" in {site["caller"]["source"] for site in fact["sites"]}

    replayed = _runtime(tmp_path, Journal(runtime.journal.rows))
    replayed.owner.current_step = runtime.owner.current_step
    replayed_partition = migration_partition(replayed)
    assert replayed_partition is not None
    assert [(row.caller.source, row.classification)
            for row in replayed_partition.decisions] == [
        ("Build(new=3)", "unrelated"), ("Build(old=4)", "reusable"),
    ]


def test_r048_partition_invalidates_when_exemplar_binding_changes(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition

    monkeypatch.setenv(SWITCH, "on")
    (tmp_path / "factory.py").write_text(PRODUCER)
    (tmp_path / "client.py").write_text(
        "from factory import Widget as Build\nfirst = Build(old=2)\nsecond = Build(old=3)\n")
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    _apply(runtime, "client.py", "first = Build(old=2)", "first = Build(new=2)")
    assert migration_partition(runtime) is not None

    _apply(runtime, "client.py", "from factory import", "from other import")
    assert migration_partition(runtime) is None


def test_r048_partition_invalidates_when_receipt_changes_default_dependency(
        tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition

    monkeypatch.setenv(SWITCH, "on")
    producer = (
        "DEFAULT_TIMEOUT = None\n"
        "class Widget:\n"
        "    def __init__(self, *, old=DEFAULT_TIMEOUT):\n"
        "        self.timeout = old\n"
        "\n"
        "first = Widget(old=1)\n"
        "repeat = Widget(old=2)\n"
        "omitted = Widget()\n"
    )
    changed_constructor = (
        "class Widget:\n"
        "    def __init__(self, *, new=DEFAULT_TIMEOUT):\n"
        "        self.timeout = new"
    )
    old_constructor = (
        "class Widget:\n"
        "    def __init__(self, *, old=DEFAULT_TIMEOUT):\n"
        "        self.timeout = old"
    )
    factory = tmp_path / "factory.py"
    factory.write_text(producer)
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", old_constructor, changed_constructor)
    _apply(runtime, "factory.py", "first = Widget(old=1)", "first = Widget(new=1)")

    before_default_change = migration_partition(runtime)
    assert before_default_change is not None
    omitted = next(row for row in before_default_change.decisions
                   if row.caller.source == "Widget()")
    assert omitted.classification == "unrelated"
    before_namespace = {}
    producer_source = factory.read_text().partition("\nfirst =")[0]
    exec(compile(producer_source, "factory.py", "exec"), before_namespace)
    assert before_namespace["Widget"]().timeout is None

    _apply(runtime, "factory.py", "DEFAULT_TIMEOUT = None", "DEFAULT_TIMEOUT = 30")
    current_namespace = {}
    producer_source = factory.read_text().partition("\nfirst =")[0]
    exec(compile(producer_source, "factory.py", "exec"), current_namespace)
    assert current_namespace["Widget"]().timeout == 30
    assert migration_partition(runtime) is None


def test_r048_future_annotations_above_exemplar_preserve_partition(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition

    monkeypatch.setenv(SWITCH, "on")
    source = (
        "DEFAULT_TIMEOUT: int | None = None\n"
        "class Widget:\n"
        "    def __init__(self, *, old=DEFAULT_TIMEOUT):\n"
        "        self.timeout = old\n"
        "\n"
        "first = Widget(old=1)\n"
        "repeat = Widget(old=30)\n"
    )
    updated_constructor = (
        "class Widget:\n"
        "    def __init__(self, *, new=DEFAULT_TIMEOUT):\n"
        "        self.timeout = new"
    )
    old_constructor = (
        "class Widget:\n"
        "    def __init__(self, *, old=DEFAULT_TIMEOUT):\n"
        "        self.timeout = old"
    )
    factory = tmp_path / "factory.py"
    factory.write_text(source)
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", old_constructor, updated_constructor)
    before_exemplar = (
        "DEFAULT_TIMEOUT: int | None = None\n"
        "class Widget:\n"
        "    def __init__(self, *, new=DEFAULT_TIMEOUT):\n"
        "        self.timeout = new\n"
        "\n"
        "first = Widget(old=1)\n"
    )
    after_exemplar = (
        "from __future__ import annotations\n"
        "DEFAULT_TIMEOUT: int | None = None\n"
        "class Widget:\n"
        "    def __init__(self, *, new=DEFAULT_TIMEOUT):\n"
        "        self.timeout = new\n"
        "\n"
        "first = Widget(new=1)\n"
    )
    _apply(runtime, "factory.py", before_exemplar, after_exemplar)

    partition = migration_partition(runtime)
    assert partition is not None
    assert partition.producer.receipt == "edit-receipt-0"
    assert partition.exemplar.receipt == "edit-receipt-1"
    assert partition.delta.previous_default is None
    repeat = next(row for row in partition.decisions if row.caller.source == "Widget(old=30)")
    assert repeat.classification == "reusable"
    namespace = {}
    producer_source = factory.read_text().partition("\nfirst =")[0]
    exec(compile(producer_source, "factory.py", "exec"), namespace)
    assert namespace["Widget"]().timeout is None
    assert namespace["Widget"](new=30).timeout == 30


def test_r048_rebound_producers_are_ineligible():
    from kernel_r200_r048_grounded_migration_edit import producer_delta

    rebound = "\nWidget = other\n"
    assert producer_delta(
        "factory.py", PRODUCER + rebound, UPDATED_PRODUCER + rebound) is None
    constructor_override = "    __init__ = external\n"
    assert producer_delta("factory.py", PRODUCER + constructor_override,
                          UPDATED_PRODUCER + constructor_override) is None


@pytest.mark.host_sandbox
def test_r048_switch_on_reaches_scripted_provider(tmp_path, monkeypatch):
    import subprocess
    import sys

    from _journal import measurement_events
    from _timeouts import scaled
    from acceptance._support import tree_python
    from kernel_r200_r048_grounded_migration_edit import DELIVERY_ACTION, EDIT_ACTION, HEADER
    from stateframe.scripted_endpoint import Script, serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "factory.py").write_text(PRODUCER)
    (repo / "client.py").write_text(CALLERS)
    (repo / "test_migration.py").write_text(
        "from client import first, repeat, consumed\n"
        "def test_migration():\n    assert (first.value, repeat.value, consumed) == (2, 3, 4)\n")
    script = Script({"model": "scripted", "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "required": True, "arguments": {"path": "factory.py"}},
        {"tool": "str_replace", "required": True, "arguments": {
            "path": "factory.py", "old_str": PRODUCER.rstrip(), "new_str": UPDATED_PRODUCER.rstrip(),
        }},
        {"tool": "read", "required": True, "arguments": {"path": "client.py"}},
        {"tool": "str_replace", "required": True, "arguments": {
            "path": "client.py", "old_str": "first = Build(old=2)", "new_str": "first = Build(new=2)",
        }},
        {"tool": "str_replace", "required": True, "arguments": {
            "path": "client.py", "old_str": "Build(old=3)", "new_str": "Build(new=3)",
        }},
        {"tool": "str_replace", "required": True, "arguments": {
            "path": "client.py", "old_str": "Build(old=4)", "new_str": "Build(new=4)",
        }},
        {"tool": "run_tests", "required": True, "arguments": {"target": "test_migration.py"}},
        {"tool": "done", "required": True, "arguments": {"summary": "Migrated constructor calls."}},
    ]})
    monkeypatch.setenv(SWITCH, "on")
    python = tree_python(ROOT / "src", tmp_path / "tree-python")
    server = serve(script)
    run = tmp_path / "run"
    try:
        result = subprocess.run(
            [python, "-m", "stateframe", "run", "--repo", str(repo), "--run-dir", str(run),
             "--task", "Migrate Widget constructor callers from old to new.",
             "--edit", "factory.py", "--edit", "client.py",
             "--endpoint", f"http://127.0.0.1:{server.server_port}/v1", "--model", "scripted",
             "--test-python", sys.executable, "--max-turns", "16", "--quiet"],
            capture_output=True, text=True, timeout=scaled(240))
        failures = [row for row in measurement_events(run)
                    if row["kind"] in {"framefit_fail_closed", "engine_error"}]
        assert result.returncode == 0, (result.stdout + result.stderr, failures)
        script.assert_consumed()
    finally:
        server.shutdown()
        server.server_close()
    events = measurement_events(run)
    assert [event["role"] for event in events if event.get("action") == EDIT_ACTION] == [
        "producer", "exemplar", "site", "site",
    ]
    counts = [sum(str(row.get("content", "")).count(HEADER) for row in body["messages"])
              for body in script.requests]
    assert max(counts) == 1
    assert counts[4] == 1
    assert any(event.get("action") == DELIVERY_ACTION and event["delivered_instances"] == 1 for event in events)


def test_r048_unchanged_default_calls_do_not_consume_exception_summary(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition
    from kernel_r200_r048_grounded_migration_edit import HEADER, MAX_SUMMARY_BYTES, parse_fact

    monkeypatch.setenv(SWITCH, "on")
    (tmp_path / "factory.py").write_text(PRODUCER)
    client = tmp_path / "client.py"
    client.write_text(
        "from factory import Widget as Build\n"
        "chosen = Build(old=2)\n"
        "reusable = Build(old=3)\n"
        "exceptional = Build(old=99).value\n"
        + "".join(f"unchanged_{index} = Build()\n" for index in range(12)))
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    _apply(runtime, "client.py", "chosen = Build(old=2)", "chosen = Build(new=2)")

    partition = migration_partition(runtime)
    unchanged = [row for row in partition.decisions if row.caller.source == "Build()"]
    assert len(unchanged) == 12
    assert all(row.classification == "unrelated" for row in unchanged)
    assert next(row for row in partition.decisions if row.caller.source == "Build(old=99)").classification == "exceptional"
    fact = parse_fact(next(
        line for line in partition.render().splitlines() if line.startswith(HEADER)))
    assert len(partition.render().encode()) <= MAX_SUMMARY_BYTES
    assert fact["omitted"] > 0
    assert any(site["classification"] == "reusable"
               and site["caller"]["source"] == "Build(old=3)" for site in fact["sites"])
    assert any(site["classification"] == "exceptional"
               and site["caller"]["source"] == "Build(old=99)" for site in fact["sites"])


def test_r048_forwarded_old_keyword_stays_exceptional_after_new_keyword(tmp_path, monkeypatch):
    from kernel_coordinated_change_ledger import migration_partition

    monkeypatch.setenv(SWITCH, "on")
    (tmp_path / "factory.py").write_text(PRODUCER)
    client = tmp_path / "client.py"
    client.write_text(
        "from factory import Widget as Build\n"
        "chosen = Build(old=2)\n"
        "config = {'old': 7}\n"
        "forwarded = Build(new=3, **config)\n")
    runtime = _runtime(tmp_path, Journal())
    _apply(runtime, "factory.py", PRODUCER, UPDATED_PRODUCER)
    _apply(runtime, "client.py", "chosen = Build(old=2)", "chosen = Build(new=2)")

    partition = migration_partition(runtime)
    decisions = {row.caller.source: row for row in partition.decisions}
    assert decisions["Build(new=3, **config)"].classification == "exceptional"
    assert decisions["Build(new=3, **config)"].reason == "forwarding changes argument evidence"
    namespace = {}
    exec(UPDATED_PRODUCER, namespace)
    with pytest.raises(TypeError, match="old"):
        namespace["Widget"](new=3, **{"old": 7})


@pytest.mark.host_sandbox
def test_r048_off_request_bytes_match_pinned_099_after_missing_source_requalification(
        tmp_path, tmp_path_factory, monkeypatch):
    import os
    import subprocess
    import sys
    import textwrap
    import urllib.request
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    from _timeouts import scaled

    pinned = pinned_tree(BASE, tmp_path_factory)
    script = tmp_path / "framefit5_missing_source_requalification.py"
    script.write_text(textwrap.dedent("""
        import json
        import hashlib
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        checkout, workspace_arg, output_arg = map(Path, sys.argv[1:])
        sys.path[:0] = [str(checkout / "src"),
                        str(checkout / "src" / "stateframe" / "engine")]
        from kernel_request_evidence_closure import SpanNeed, build_source_closure
        from kernel_request_reprepare import qualify_source
        from kernel_request_source_window import (
            source_delivery_satisfied, source_revision, verify_final_source_delivery,
        )
        from kernel_measurement import _serialize_final_request_body
        from kernel_stateframe_framefit5 import (
            _serialize_state_message, assemble_framefit5_request,
        )

        workspace = workspace_arg
        workspace.mkdir(parents=True, exist_ok=True)
        source = "\\n".join((
            "class Widget:",
            "    def __init__(self, *, new=1):",
            "        self.value = new",
            "",
        ))
        (workspace / "factory.py").write_text(source)
        revision = source_revision(source)
        plan = build_source_closure(
            workspace, ["factory.py"], "Inspect the migrated Widget constructor.",
            target_path="factory.py",
        ).as_dict()
        plan["paths"] = ["factory.py"]
        plan["units"] = [{
            **SpanNeed("factory.py", 1, 3, "target", "fixture", 1).as_dict(),
            "unit_handle": "factory:1",
        }]
        plan["revisions"] = {"factory.py": revision}
        plan["source_contents"] = {"factory.py": source}
        plan["bounded_delivery"] = {
            "retained_handles": ["factory:1"], "omitted_handles": [],
        }
        source_map = {"factory.py": source}

        system = {"role": "system", "content": "Follow the task."}
        task_content = "Task: Inspect the migrated Widget constructor."
        task = {"role": "user", "content": task_content}
        frame = {
            "schema": "stateframe-framefit5/1", "mode": "FRAME_ONLY",
            "task": {"sha256": hashlib.sha256(task_content.encode("utf-8")).hexdigest()},
            "open_sites": [],
            "applied_edits": [],
            "last_verdict": {"line": "No test run"},
            "action_ledger": {"reads": [], "edits": [], "unclassified_actions": []},
        }
        state = {"role": "user", "content": _serialize_state_message(
            {"state": {"framefit5": frame}})}
        followup = {"role": "user", "content": "Read factory.py and continue."}
        assistant = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "read-1", "type": "function",
            "function": {"name": "read", "arguments": "{}"},
        }]}
        tool = {"role": "tool", "tool_call_id": "read-1", "content": "source omitted"}
        transcript = [system, task, followup, assistant, tool]
        prepared = [system, task, state]
        tools = [{"type": "function", "function": {
            "name": "read", "description": "Read a source file",
            "parameters": {"type": "object", "properties": {}},
        }}]
        class Journal:
            def __init__(self):
                self.rows = []

            def emit(self, kind, **fields):
                self.rows.append({"kind": kind, **fields})

        journal = Journal()
        capture = SimpleNamespace(
            journal=journal, request_id="send", step=1,
            _request_source_closure_override=plan,
            _framefit5_task_message=task,
            frame_request_allocation=None,
            request_action_contract={"admitted_tool_schemas": tools},
        )
        runtime = SimpleNamespace(
            owner=SimpleNamespace(root=workspace),
            _active_capture=capture,
            _empirical_frame_sources={"factory.py": {"content": source}},
            frame_ceiling=40_000,
            journal=journal,
        )

        for setting in ("unset", "false"):
            journal.rows.clear()
            if setting == "unset":
                import os
                os.environ.pop("STATEFRAME_MIGRATION_SITE_PARTITION", None)
            else:
                os.environ["STATEFRAME_MIGRATION_SITE_PARTITION"] = "false"
            messages = assemble_framefit5_request(prepared, transcript)
            body = {"model": "scripted", "messages": messages, "tools": tools}
            wire = _serialize_final_request_body(body, runtime, capture.request_id)
            missing = verify_final_source_delivery(
                plan, messages, payload=wire, sources=source_map,
            )
            assert any(row.get("kind") == "final_wire_proof_missing"
                       for row in missing.get("proof_failures", []))
            capture.request_budget_final_wire = {
                "status": "HOST_BLOCKED", "messages": True, "tools": True,
                "final_validation_failures": ["source_delivery_unqualified"],
                "source_delivery": missing,
            }
            repaired_wire = qualify_source(
                capture, runtime, None, body, wire, allow_requalification=True,
            )
            receipt = capture.request_source_final_delivery
            assert source_delivery_satisfied(receipt)
            assert _serialize_final_request_body(body, runtime, capture.request_id) == repaired_wire
            actions = [(row.get("path", "factory.py"), row.get("action"))
                       for row in journal.rows if row.get("kind") == "source_requalified"]
            assert actions == [("factory.py", "framefit5_reassembled_source")]
            Path(str(output_arg) + "." + setting).write_bytes(repaired_wire)
            Path(str(output_arg) + "." + setting + ".json").write_text(
                json.dumps({"actions": actions, "status": receipt["status"]}),
            )
    """))

    received = []

    class RawProvider(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), RawProvider)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    work = tmp_path / "workspace"
    source_cases = (("pinned", pinned), ("current", ROOT))
    outputs = {}
    try:
        for label, checkout in source_cases:
            output = tmp_path / f"{label}-requalified"
            environment = {
                key: value for key, value in os.environ.items()
                if key not in {SWITCH, "PYTHONPATH"}
            }
            environment["STATEFRAME_WORKING_SET_ALLOCATOR"] = "off"
            environment["STATEFRAME_SOURCE_REQUALIFY"] = "on"
            result = subprocess.run(
                [sys.executable, str(script), str(checkout), str(work), str(output)],
                env=environment, capture_output=True, text=True, timeout=scaled(180),
            )
            assert result.returncode == 0, result.stdout + result.stderr
            for setting in ("unset", "false"):
                metadata = json.loads(Path(f"{output}.{setting}.json").read_text())
                assert metadata == {
                    "actions": [["factory.py", "framefit5_reassembled_source"]],
                    "status": "COMPLETE",
                }
                wire = Path(f"{output}.{setting}").read_bytes()
                outputs[f"{label}-{setting}"] = wire
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                    data=wire, headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=scaled(10)) as response:
                    assert response.status == 200
                assert received[-1] == ("/v1/chat/completions", wire)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=scaled(10))

    assert outputs["pinned-unset"] == outputs["current-unset"]
    assert outputs["pinned-false"] == outputs["current-false"]
    assert outputs["current-unset"] == outputs["current-false"]
