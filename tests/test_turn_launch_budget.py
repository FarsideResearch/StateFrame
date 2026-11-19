"""The launch port's critical path, with workspace admission and actual HTTP submission."""

from __future__ import annotations

import ast
import json
import threading
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
from time import monotonic
from urllib.request import Request, urlopen

import pytest

from kernel_workspace_index import WorkspaceIndex
from symbol_registry import SymbolRegistry
from test_workspace_state_oracle import scripted_provider

try:
    import kernel_turn_launch as launch
    from kernel_turn_contracts import Context, Ref, Snapshot, Unavailable
except ModuleNotFoundError as error:
    if error.name != "kernel_turn_launch":
        raise
    launch = None


FILES = 5_000
TURNS = 200
POINT = "group_099/module_49.py"
PROMPT = "Inspect the implementation and preserve its public behavior."
READ = {"tool": "read", "arguments": {"path": POINT}}
SYSTEM = {"role": "system", "content": "Use only the tools authorized for this request."}
INSTRUCTIONS = {"role": "user", "content": "Keep the public interface stable."}
READ_SCHEMA = {
    "type": "function",
    "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class _Publication:
    def __init__(self):
        self.objects = {}

    @contextmanager
    def transaction(self):
        yield

    def stage(self, codec, data):
        ref = Ref(codec, sha256(data).hexdigest(), len(data))
        self.objects[ref] = data
        return ref

    def load(self, ref):
        return self.objects[Ref(*ref)]


class _Workspace:
    def __init__(self, source, publication):
        self.source = source
        self.publication = publication
        self.counts = Counter()
        self.pins = 0

    def admit(self, source, policy):
        assert source == self.source
        index = WorkspaceIndex(source)
        tree = index.refresh()
        self.counts["admission_scans"] += 1
        self.counts["admission_reads"] += index.work.content_reads
        if not any(source.iterdir()) or json.loads(self.publication.load(policy)).get("reject"):
            raise Unavailable("fixture admission refused")
        admission = self.publication.stage("turn/admission/1", _json({"root": tree.digest.hex()}))
        self.snapshot = Snapshot(tree.digest.hex(), admission, 0)
        return self.snapshot

    @contextmanager
    def retain(self, root):
        assert root == self.snapshot.root
        self.pins += 1
        try:
            yield self.snapshot
        finally:
            self.pins -= 1

    def read(self, at, path):
        assert self.pins > 0 and at == self.snapshot
        self.counts["point_reads"] += 1
        return (self.source / path.decode()).read_bytes()


class _Facts:
    def __init__(self, workspace, persisted):
        self.workspace = workspace
        self.persisted = persisted
        self.counts = Counter()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()

    def build(self, parent, stop):
        assert self.workspace.pins > 0 and parent == self.workspace.snapshot
        self.counts["build_calls"] += 1
        self.entered.set()
        try:
            if self.persisted.exists() and self.persisted.read_text() == parent.root:
                self.counts["warm_hits"] += 1
                return
            # No parser or result can become available until the first real point read.
            deadline = monotonic() + 6
            while not stop() and not self.release.wait(0.01):
                if monotonic() >= deadline:
                    return  # Mutation backstop.
            if not stop():
                self.persisted.write_text(parent.root)
        finally:
            self.exited.set()


def _repo(root):
    for group in range(100):
        directory = root / f"group_{group:03d}"
        directory.mkdir(parents=True)
        for leaf in range(50):
            body = f"VALUE = {group * 50 + leaf}\n"
            body += "".join(f"def handle_{i}(value):\n    return value + {i}\n" for i in range(16))
            (directory / f"module_{leaf:02d}.py").write_text(body)


def _warmup_input(history):
    head = sha256(history.read_bytes()).hexdigest()
    return {
        "head": [TURNS, head],
        "messages": [SYSTEM, INSTRUCTIONS, {"role": "user", "content": PROMPT}],
        "msg": PROMPT,
        "scope": {"entry": {"path": "group_000/module_00.py"}, "allowed_product_paths": ["group_000"]},
        "preflight": {"passed": False, "exit_code": 1},
        "tools": [READ_SCHEMA, {"type": "function", "function": {"name": "done"}}],
        "authority": sha256(b"test-authority").hexdigest(),
        "step": TURNS,
    }


def _submit(url, messages, tools):
    body = {"model": "scripted", "messages": messages, "tools": tools}
    with urlopen(Request(url + "/chat/completions", _json(body),
                         headers={"Content-Type": "application/json"}), timeout=5) as response:
        return json.load(response)["choices"][0]["message"]["tool_calls"][0]["function"]


def _first_wire(url, publication, artifact):
    manifest = json.loads(publication.load(artifact.value))
    fragments = {name: json.loads(publication.load(ref)) for name, ref, _ in manifest["slots"]}
    messages = fragments["messages"] + [
        {"role": "user", "content": _json(fragments["scope"]).decode()},
        {"role": "user", "content": _json(fragments["preflight"]).decode()},
    ]
    assert manifest["layout"] == [
        ["messages", "bootstrap"], ["scope", "bootstrap"], ["preflight", "bootstrap"], ["tools", "bootstrap"]
    ]
    return _submit(url, messages, fragments["tools"])


def test_cold_first_send_with_ingestion_paused(tmp_path, monkeypatch):
    source = tmp_path / "repo"
    _repo(source)
    history = tmp_path / "history.jsonl"
    history.write_bytes(b"\n".join(_json({"turn": i, "private": "old exchange"}) for i in range(TURNS)))
    inputs = _warmup_input(history)
    counts = Counter()
    original_parse = ast.parse
    original_open = Path.open
    foreground = threading.get_ident()

    def parse(*args, **kwargs):
        if threading.get_ident() == foreground:
            counts["foreground_parses"] += 1
        return original_parse(*args, **kwargs)

    def open_path(path, *args, **kwargs):
        if path == history:
            counts["history_reads"] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", parse)
    monkeypatch.setattr(Path, "open", open_path)
    with scripted_provider([READ, READ]) as (url, trace):
        started = monotonic()
        if launch is None:
            # KernelBolt._boot performs this real legacy build before any model request.
            registry = SymbolRegistry().build(source)
            _submit(url, inputs["messages"], [READ_SCHEMA])
            elapsed = (trace.requests[0][0] / 1e9) - started
            assert counts["foreground_parses"] == registry.files_scanned == 0 and elapsed < 5, (
                f"legacy first submission {elapsed:.3f}s; "
                f"foreground_parses={counts['foreground_parses']} (budget=0), "
                f"registry_source_reads={registry.files_scanned} (budget=0)"
            )

        publication = _Publication()
        workspace = _Workspace(source, publication)
        facts = _Facts(workspace, tmp_path / "facts-root")
        ctx = Context("principal", "run", "cold", tmp_path, "profile")
        port = launch.open_port(ctx, workspace=workspace, facts=facts, publication=publication)
        policy = publication.stage("turn/admission-policy/1", _json({"max_files": FILES}))
        task = publication.stage("turn/task/1", _json(inputs))
        with publication.transaction():
            parent, frame = port.bootstrap(source, policy, task)
            with port.start_build(parent):
                call = _first_wire(url, publication, frame)
                elapsed = (trace.requests[0][0] / 1e9) - started
                assert facts.entered.wait(1)
                assert not facts.exited.is_set(), (
                    f"ingestion completed before demand while release gate was closed; first HTTP {elapsed:.3f}s"
                )
                assert call["name"] == "read"
                source_bytes = workspace.read(parent, json.loads(call["arguments"])["path"].encode())
                useful = monotonic() - started
                assert source_bytes.startswith(b"VALUE = 4999\n")
                assert workspace.counts == {
                    "admission_scans": 1, "admission_reads": FILES, "point_reads": 1,
                }
                assert counts["foreground_parses"] == counts["history_reads"] == 0
                facts.release.set()
                assert facts.exited.wait(2)
        assert workspace.pins == 0
        assert facts.counts == {"build_calls": 1}

        warm_facts = _Facts(workspace, facts.persisted)
        warm = launch.open_port(ctx._replace(attempt="warm"), workspace=workspace, facts=warm_facts,
                                publication=publication)
        started_warm = monotonic()
        with publication.transaction():
            warm_parent, warm_frame = warm.bootstrap(source, policy, task)
            with warm.start_build(warm_parent):
                _first_wire(url, publication, warm_frame)
                assert warm_facts.exited.wait(1)
        warm_elapsed = (trace.requests[1][0] / 1e9) - started_warm
        assert trace.requests[0][1] == trace.requests[1][1]
        assert sha256(trace.requests[0][1]).hexdigest() == "b263875190cc3dbdb1899702a8f82b245f090612022acc0f7b6c021fa71aca44"
        assert b"old exchange" not in trace.requests[0][1]
        assert warm_facts.counts == {"build_calls": 1, "warm_hits": 1}
        assert workspace.pins == 0
        trace.script.assert_consumed()
        print(f"cold HTTP {elapsed:.3f}s, point result {useful:.3f}s, warm HTTP {warm_elapsed:.3f}s; "
              f"admission reads per launch {FILES}, demand reads 1, foreground parses 0, history reads 0")
        assert max(elapsed, useful, warm_elapsed) < 5, (elapsed, useful, warm_elapsed)


@pytest.mark.skipif(launch is None, reason="requires launch port after RED budget fails")
def test_invalid_admission_cannot_start_optional_build(tmp_path):
    source = tmp_path / "empty"
    source.mkdir()
    publication = _Publication()
    workspace = _Workspace(source, publication)
    facts = _Facts(workspace, tmp_path / "facts-root")
    ctx = Context("principal", "run", "attempt", tmp_path, "profile")
    port = launch.open_port(ctx, workspace=workspace, facts=facts, publication=publication)
    policy = publication.stage("turn/admission-policy/1", b"{}")
    history = tmp_path / "history"
    history.touch()
    task = publication.stage("turn/task/1", _json(_warmup_input(history)))
    with pytest.raises(Unavailable, match="admission refused"):
        port.bootstrap(source, policy, task)
    assert facts.counts == {}
    assert workspace.pins == 0
    assert not any(ref.codec == "turn/slot-manifest/1" for ref in publication.objects)


@pytest.mark.skipif(launch is None, reason="requires launch port after RED budget fails")
def test_child_retains_parent_and_cancels_build_when_first_action_fails(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from stateframe._child import _turn_launch_binding

    source = tmp_path / "repo"
    source.mkdir()
    (source / "answer.py").write_text("VALUE = 1\n")
    publication = _Publication()
    workspace = _Workspace(source, publication)
    facts = _Facts(workspace, tmp_path / "facts-root")
    probe = object()
    observed = []
    host = ModuleType("kernel_turn_host")

    @contextmanager
    def open_run(ctx, *, live):
        assert live is probe
        assert ctx.private_dir == tmp_path
        assert ctx.run == "run-id" and ctx.attempt == "0"
        observed.append("open")
        try:
            yield SimpleNamespace(
                publication=publication,
                launch=launch.open_port(ctx, workspace=workspace, facts=facts, publication=publication),
            )
        finally:
            assert workspace.pins == 0
            assert facts.exited.is_set()
            observed.append("closed")

    host.open_run = open_run
    monkeypatch.setitem(sys.modules, "kernel_turn_host", host)
    spec = {
        "environment": {},
        "parent_snapshot": str(source),
        "profile": "product",
        "run_identity_seed": "run-id",
        "public_row": {"msg": PROMPT, "entry": {"path": "answer.py"}, "allowed_product_paths": ["answer.py"]},
        "turn_launch": {"baseline": {"passed": False, "exit_code": 1}},
    }
    opener, require_admitted = _turn_launch_binding(spec, tmp_path, restoring=False)
    with pytest.raises(Unavailable, match="no active admitted parent"):
        require_admitted()
    history = tmp_path / "history"
    history.touch()
    with pytest.raises(RuntimeError, match="first action"):
        with opener(probe) as binding:
            with pytest.raises(Unavailable, match="no active admitted parent"):
                require_admitted()
            with publication.transaction():
                policy = publication.stage("turn/admission-policy/1", b"{}")
                parent, frame = binding.bootstrap(policy, _warmup_input(history))
                manifest = json.loads(publication.load(frame.value))
                slots = {name: json.loads(publication.load(ref)) for name, ref, _ in manifest["slots"]}
                assert slots["scope"]["entry"]["path"] == "answer.py"
                assert slots["preflight"]["baseline"] == spec["turn_launch"]["baseline"]
                assert slots["tools"] == [READ_SCHEMA]
            require_admitted()
            assert facts.entered.wait(1)
            assert workspace.read(parent, b"answer.py") == b"VALUE = 1\n"
            raise RuntimeError("first action failed")
    assert observed == ["open", "closed"]
    assert facts.counts == {"build_calls": 1}
    assert not facts.persisted.exists()
    with pytest.raises(Unavailable, match="no active admitted parent"):
        require_admitted()


@pytest.mark.skipif(launch is None, reason="requires launch port after RED budget fails")
@pytest.mark.parametrize("worker_starts", [True, False])
def test_facts_failure_keeps_admitted_point_read_available(tmp_path, caplog, monkeypatch, worker_starts):
    source = tmp_path / "repo"
    source.mkdir()
    (source / "answer.py").write_text("VALUE = 1\n")
    publication = _Publication()
    workspace = _Workspace(source, publication)
    entered = threading.Event()

    class FailingFacts:
        def build(self, parent, stop):
            entered.set()
            raise OSError("private failure details")

    if not worker_starts:
        def unable_to_start(self):
            raise RuntimeError("private failure details")

        monkeypatch.setattr(launch.Thread, "start", unable_to_start)
    ctx = Context("principal", "run", "attempt", tmp_path, "profile")
    port = launch.open_port(ctx, workspace=workspace, facts=FailingFacts(), publication=publication)
    history = tmp_path / "history"
    history.touch()
    policy = publication.stage("turn/admission-policy/1", b"{}")
    task = publication.stage("turn/task/1", _json(_warmup_input(history)))
    with publication.transaction():
        parent, _ = port.bootstrap(source, policy, task)
    with port.start_build(parent):
        if worker_starts:
            assert entered.wait(1)
        assert workspace.read(parent, b"answer.py") == b"VALUE = 1\n"
        with pytest.raises(RuntimeError, match="unused admitted parent"):
            with port.start_build(parent):
                pytest.fail("launched another build")
    assert workspace.pins == 0
    error_type = "OSError" if worker_starts else "RuntimeError"
    assert f"Background fact build failed ({error_type})" in caplog.text
    assert "private failure details" not in caplog.text
