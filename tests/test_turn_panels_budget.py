"""Work and byte parity at the committed journal-to-panel boundary."""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import pytest

from kernel_stateframe_envelope import serialize_stateframe_payload
from kernel_stateframe_runtime import StateFrameRuntime


TURN_COUNT = 200
FILE_COUNT = 5000
PORT_AVAILABLE = importlib.util.find_spec("kernel_turn_panels") is not None


def _wire(value):
    return json.dumps(value).encode()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class History(list):
    """Count exactly the rows visited by the release's _frame receipt loop."""

    def __init__(self, counts):
        self.counts = counts
        super().__init__({
            "kind": "run_tests" if index == 5 else "read",
            "intent": "inspect",
            "result_class": "PASS",
            "detail": 'résumé “東京” \\line\n𝄞',
            "turn": index + 1,
            "attempt_id": None,
        } for index in range(TURN_COUNT))

    def __iter__(self):
        for row in super().__iter__():
            self.counts["prefix_rows"] += 1
            yield row


class DiskPublication:
    """Use persisted bytes for every load, also across fresh port instances."""

    def __init__(self, root):
        self.root = root
        root.mkdir(exist_ok=True)
        self.records = []
        self.suffix_reads = 0

    def transaction(self):
        return nullcontext()

    def stage(self, codec, data):
        from kernel_turn_contracts import Ref

        digest = _sha(data)
        target = self.root / digest
        if not target.exists():
            target.write_bytes(data)
        return Ref(codec, digest, len(data))

    def load(self, ref):
        from kernel_turn_contracts import Corrupt

        try:
            data = (self.root / ref.sha256).read_bytes()
        except FileNotFoundError as exc:
            raise Corrupt("missing object") from exc
        if len(data) != ref.size or _sha(data) != ref.sha256:
            raise Corrupt("damaged object")
        return data

    def suffix(self, after, through):
        for index in range(after.seq, through.seq):
            self.suffix_reads += 1
            yield self.records[index]


class Fixture:
    layout = [["instructions", "full"], ["source", "full"], ["receipts", "full"]]

    def __init__(self, path):
        self.counts = Counter()
        self.workspace = path / "work"
        self.workspace.mkdir()
        for index in range(FILE_COUNT):
            (self.workspace / f"file_{index:04}.py").write_text(
                f"VALUE = {index}  # café\n", encoding="utf-8")
        self.target = self.workspace / "file_0000.py"
        self.history = History(self.counts)
        self.objects = DiskPublication(path / "objects")

    def read_source(self):
        self.counts["source_reads"] += 1
        return self.target.read_text(encoding="utf-8")

    def instructions(self):
        self.counts["instructions_projections"] += 1
        return {"role": "system", "content": 'Keep "résumé" intact.\n'}

    def source(self, working=None):
        self.counts["source_projections"] += 1
        return {"role": "user", "content": self.read_source() if working is None else working}

    def runtime(self):
        return SimpleNamespace(
            owner=SimpleNamespace(_receipts=self.history, fn="", task_id="panel-test",
                                  current_step=TURN_COUNT),
            target=self.target, _target_source=self.read_source,
            _target_relative=lambda: self.target.name,
            _refresh_working_notes=lambda: None, parent=SimpleNamespace(charge=lambda *a, **k: None),
            source_resolver=lambda source, fn: source, projector_kind="rich",
            journal=SimpleNamespace(manifest={"environment_revision": "env"},
                                    emit=lambda *a, **k: None),
            observations=[], catalog_unknowns_projection=dict, history_findings=None,
            work_record=None, decisive_factor=None, unified_mode="off", boot_packet=None,
            disposition_mode="off", working_view_mode="source", jq_finish_enabled=False,
            jq_class_edit_enabled=False, stateframe_v2_s1=None, thinking=None,
            disposition_enabled=False, live_projection=None, probe_receipts=[],
            observation_probe_enabled=False, step_ledger=None, empirical_fixes_mode="off",
            decision_state=None, retained=None, _bind_quantity_finding=None,
            active_tool_restriction=None, public_proof_id="include",
            green_lock=SimpleNamespace(sync=lambda frame: False), _run_questions=lambda *a: None,
            frame_series=None, stateframe_pattern_cards_mode="off", decision_cache=None,
            _micrograph_frame=lambda frame: frame,
        )

    def uncached(self):
        frame, working, _ = StateFrameRuntime._frame(self.runtime())
        state = self.state(frame["latest_receipts"], frame["receipt_count"])
        self.counts["unchanged_encodes"] += 2
        self.counts["changed_encodes"] += 1
        return _wire([
            self.instructions(), self.source(working),
            {"role": "user", "content": serialize_stateframe_payload(state)},
        ])

    @staticmethod
    def state(latest, count):
        return {"schema": "stateframe-f4/1", "state": {"latest_receipts": latest,
                                                       "receipt_count": count}}

    def amend(self):
        self.history[5]["detail"] = 'amended "café" \\new\n𝄞'


class PointFacts:
    """Fixture implementation of the pinned, scoped Fact lookups in the turn ABI."""

    def __init__(self, fixture, panels, contracts):
        self.fixture = fixture
        self.panels = panels
        self.contracts = contracts
        self.receipt = contracts.Key("receipt", b'"turn-6"')
        self.revisions = {self.receipt: "first"}
        for name, variant in fixture.layout:
            self.revisions[panels.slot_key(name, variant)] = "first"
        self.revisions[panels.CATALOG_KEY] = "first"

    def lookup(self, at, key, cursor, limit):
        assert cursor is None
        assert limit <= 256
        fixture = self.fixture
        fixture.counts["fact_lookups"] += 1
        dependencies = (self.contracts.Dep(key, self.revisions[key]),)
        if key == self.panels.CATALOG_KEY:
            rows = fixture.layout
        else:
            _, name, _ = json.loads(key.selector)
            if name == "receipts":
                latest = {}
                for index in (5, TURN_COUNT - 1):
                    row = fixture.history[index]
                    fixture.counts["point_rows"] += 1
                    latest[row["kind"]] = {
                        field: row.get(field) for field in
                        ("kind", "intent", "result_class", "detail", "turn", "attempt_id")
                    }
                rows = [["stateframe-message", self.panels.ordered_value(
                    fixture.state(latest, len(fixture.history)))]]
                dependencies += (self.contracts.Dep(self.receipt, self.revisions[self.receipt]),)
            else:
                value = getattr(fixture, name)()
                rows = [["json", self.panels.ordered_value(value)]]
        ref = fixture.objects.stage("turn/fact-page/1", _canonical(rows))
        return self.contracts.Page(self.contracts.Artifact(ref, dependencies), None)

    def current(self, at, deps):
        self.fixture.counts["dependency_comparisons"] += len(deps)
        return all(self.revisions.get(dep.key) == dep.revision for dep in deps)

    def build(self, parent, stop):
        raise AssertionError("a panel must not build the repository index")


def _open(fixture):
    import kernel_turn_contracts as contracts
    import kernel_turn_panels as panels

    facts = PointFacts(fixture, panels, contracts)
    ctx = contracts.Context("principal", "run", "attempt", fixture.objects.root, "profile")
    return panels, contracts, facts, ctx


def _basis(contracts, turn, *, epoch=0, authority="authority"):
    return contracts.Basis("parent", "live", epoch,
                           contracts.Head(turn, _sha(str(turn).encode())), turn, authority)


def _event(contracts, fixture, before, after, *keys):
    payload = fixture.objects.stage("turn/event/1", _canonical({"kind": "receipt_amended"}))
    return contracts.Event(before.journal, after.journal, "receipt_amended", keys, payload)


def test_amended_receipt_reencodes_only_dependent_slots(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFRAME_FRAME_EDIT_INTEGRITY", "off")
    monkeypatch.setenv("STATEFRAME_TURN_PANELS", "on")
    fixture = Fixture(tmp_path)
    if PORT_AVAILABLE:
        panels, contracts, facts, ctx = _open(fixture)
        previous = _basis(contracts, TURN_COUNT)
        current = _basis(contracts, TURN_COUNT + 1)
        # Historical records are already committed; the panel consumer starts
        # at a verified fact snapshot and receives only the newly committed suffix.
        fixture.objects.records.extend([None] * TURN_COUNT)
        layout = fixture.objects.stage("turn/layout/1", _canonical(fixture.layout))
        port = panels.open_port(ctx, facts=facts, publication=fixture.objects)
        import kernel_turn_reducers as reducers

        encode = reducers.encode_fragment

        def counted_encode(encoding, value):
            fixture.counts["changed_encodes" if encoding == "stateframe-message"
                           else "unchanged_encodes"] += 1
            return encode(encoding, value)

        monkeypatch.setattr(reducers, "encode_fragment", counted_encode)
        start = perf_counter()
        with fixture.objects.transaction():
            manifest = port.fold(None, previous, iter(()))
            first = port.render(manifest, layout)
        cold = perf_counter() - start
        assert first == fixture.uncached()
        # Reopen the port, with no in-process slot cache or old receipt reads.
        port = panels.open_port(ctx, facts=facts, publication=fixture.objects)
    else:
        start = perf_counter()
        first = fixture.uncached()
        cold = perf_counter() - start

    fixture.amend()
    fixture.counts.clear()
    start = perf_counter()
    if PORT_AVAILABLE:
        facts.revisions[facts.receipt] = "amended"
        fixture.objects.records.append(_event(contracts, fixture, previous, current, facts.receipt))
        suffix = fixture.objects.suffix(previous.journal, current.journal)
        with fixture.objects.transaction():
            amended = port.fold(manifest, current, suffix)
            result = port.render(amended, layout)
    else:
        result = fixture.uncached()
    elapsed = perf_counter() - start
    work = {key: fixture.counts[key] for key in (
        "prefix_rows", "source_reads", "instructions_projections", "source_projections",
        "unchanged_encodes", "changed_encodes",
    )}
    print(f"panel fixture cold={cold:.6f}s amended={elapsed:.6f}s work={work} "
          f"lookups={fixture.counts['fact_lookups']} "
          f"point_rows={fixture.counts['point_rows']} "
          f"comparisons={fixture.counts['dependency_comparisons']} "
          f"suffix_reads={fixture.objects.suffix_reads}")
    assert result != first
    assert result == fixture.uncached()
    assert work == {
        "prefix_rows": 0, "source_reads": 0, "instructions_projections": 0,
        "source_projections": 0, "unchanged_encodes": 0, "changed_encodes": 1,
    }
    assert elapsed < 0.70
    if PORT_AVAILABLE:
        assert cold < 0.45
        assert fixture.objects.suffix_reads == 1
        assert fixture.counts["fact_lookups"] == 1
        assert fixture.counts["point_rows"] == 2
        assert fixture.counts["dependency_comparisons"] == 1


@pytest.fixture
def active_panels(monkeypatch, tmp_path):
    if not PORT_AVAILABLE:
        pytest.skip("turn panel port is absent on the release baseline")
    monkeypatch.setenv("STATEFRAME_TURN_PANELS", "on")
    monkeypatch.setenv("STATEFRAME_FRAME_EDIT_INTEGRITY", "off")
    return Fixture(tmp_path)


def test_clean_turns_reuse_fragments_and_read_only_the_committed_suffix(active_panels, monkeypatch):
    import kernel_turn_reducers as reducers

    fixture = active_panels
    panels, contracts, facts, ctx = _open(fixture)
    before = _basis(contracts, 0)
    port = panels.open_port(ctx, facts=facts, publication=fixture.objects)
    layout = fixture.objects.stage("turn/layout/1", _canonical(fixture.layout))
    start = perf_counter()
    with fixture.objects.transaction():
        manifest = port.fold(None, before, iter(()))
        expected = port.render(manifest, layout)
    cold = perf_counter() - start
    assert expected == fixture.uncached()

    def reject_encode(*args):
        pytest.fail("an unchanged fragment was encoded again")

    monkeypatch.setattr(reducers, "encode_fragment", reject_encode)
    fixture.counts.clear()
    elapsed = []
    for step in range(1, TURN_COUNT + 1):
        after = _basis(contracts, step)
        fixture.objects.records.append(_event(
            contracts, fixture, before, after, contracts.Key("step", b'"current"')))
        suffix = fixture.objects.suffix(before.journal, after.journal)
        start = perf_counter()
        with fixture.objects.transaction():
            manifest = port.fold(manifest, after, suffix)
            wire = port.render(manifest, layout)
        elapsed.append(perf_counter() - start)
        assert wire == expected
        before = after

    assert fixture.objects.suffix_reads == TURN_COUNT
    assert not fixture.counts
    restart = ctx._replace(attempt="restart")
    port = panels.open_port(restart, facts=facts, publication=fixture.objects)
    start = perf_counter()
    with fixture.objects.transaction():
        restored = panels.restore_artifact(restart, publication=fixture.objects, ref=manifest.value)
        resumed = port.fold(restored, before, iter(()))
        assert port.render(resumed, layout) == expected
    warm = perf_counter() - start
    print(f"panel fixture cold={cold:.6f}s 200 turns max={max(elapsed):.6f}s "
          f"warm restart={warm:.6f}s suffix_reads={fixture.objects.suffix_reads} "
          f"restart_comparisons={fixture.counts['dependency_comparisons']}")
    assert cold < 0.45
    assert max(elapsed) < 0.70
    assert warm < 0.40
    assert fixture.counts == {"dependency_comparisons": 5}


def test_suffix_conflicts_and_corruption_require_a_verified_rebuild(active_panels):
    fixture = active_panels
    panels, contracts, facts, ctx = _open(fixture)
    port = panels.open_port(ctx, facts=facts, publication=fixture.objects)
    before = _basis(contracts, TURN_COUNT)
    after = _basis(contracts, TURN_COUNT + 1)
    selection = [["receipts", "full"], ["instructions", "full"]]
    layout = fixture.objects.stage("turn/layout/1", _canonical(selection))
    event = _event(contracts, fixture, before, after, facts.receipt)
    with fixture.objects.transaction():
        previous = port.fold(None, before, iter(()))
        committed = port.fold(previous, after, iter((event, event)))
        assert port.fold(committed, after, iter((event,))) == committed
        for suffix in ((), (event._replace(before=_basis(contracts, 0).journal),),
                       (event._replace(after=after.journal._replace(digest="fork")),)):
            with pytest.raises(contracts.Corrupt):
                port.fold(previous, after, iter(suffix))
        with pytest.raises(contracts.Corrupt):
            port.fold(committed, after, iter((event._replace(kind="different"),)))

        other_run = panels.open_port(ctx._replace(run="another-run"), facts=facts,
                                     publication=fixture.objects)
        with pytest.raises(contracts.Corrupt):
            other_run.render(committed, layout)
        expected = port.render(committed, layout)
        reference_rows = json.loads(fixture.uncached())
        assert expected == _wire([reference_rows[2], reference_rows[0]])
        raw = json.loads(fixture.objects.load(committed.value))
        target = fixture.objects.root / raw["slots"][0][1][1]
        target.write_bytes(b"damaged")
        with pytest.raises(contracts.Corrupt, match="damaged object"):
            port.render(committed, layout)
        target.unlink()
        rebuilt = port.fold(None, after, iter(()))
        assert port.render(rebuilt, layout) == expected


def test_off_switch_decoder():
    if not PORT_AVAILABLE:
        pytest.skip("turn panel port is absent on the release baseline")
    import kernel_turn_panels as panels

    assert not panels.enabled({})
    assert not panels.enabled({"STATEFRAME_TURN_PANELS": "true"})
    assert panels.enabled({"STATEFRAME_TURN_PANELS": "on"})


def test_empty_catalog_cannot_publish_and_absent_slot_can_appear(active_panels, monkeypatch):
    fixture = active_panels
    panels, contracts, facts, ctx = _open(fixture)
    at = _basis(contracts, TURN_COUNT)
    optional = panels.slot_key("optional", "full")
    policy = contracts.Key("policy", b'"effective"')
    facts.revisions.update({optional: "absent", policy: "first"})
    lookup = facts.lookup
    values = {panels.CATALOG_KEY: [], optional: []}

    def scoped_lookup(at, key, cursor, limit):
        if key not in values:
            return lookup(at, key, cursor, limit)
        ref = fixture.objects.stage("turn/fact-page/1", _canonical(values[key]))
        deps = (contracts.Dep(key, facts.revisions[key]),)
        if key == optional:
            deps += (contracts.Dep(policy, facts.revisions[policy]),)
        return contracts.Page(contracts.Artifact(ref, deps), None)

    monkeypatch.setattr(facts, "lookup", scoped_lookup)
    port = panels.open_port(ctx, facts=facts, publication=fixture.objects)
    layout = fixture.objects.stage("turn/layout/1", _canonical([["optional", "full"]]))
    with fixture.objects.transaction():
        with pytest.raises(contracts.Unavailable, match="catalog"):
            port.fold(None, at, iter(()))
        values[panels.CATALOG_KEY] = [["optional", "full"]]
        before = port.fold(None, at, iter(()))
        with pytest.raises(contracts.Stale, match="absent"):
            port.render(before, layout)
        value = {"role": "user", "content": None, "object": {"z": "𝄞", "a": []}}
        values[optional] = [["json", panels.ordered_value(value)]]
        facts.revisions.update({optional: "present", policy: "second"})
        changed = port.fold(before, at._replace(authority="new"), iter(()))
        assert port.render(changed, layout) == _wire([value])
        values[optional] = [["json", panels.ordered_value({"role": "user"})]]
        facts.revisions[optional] = "third"
        changed = port.fold(changed, at._replace(authority="newer"), iter(()))
        assert port.render(changed, layout) == _wire([{"role": "user"}])
