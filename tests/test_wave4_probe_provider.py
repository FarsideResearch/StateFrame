"""Provider admission uses live commitments and live automatic slot occupancy."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module, util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import pytest

from _child_env import child_env

from stateframe import endpoint as endpoint_module
from stateframe.endpoint import Endpoint, EndpointError, _facts_digest
from stateframe.engine import kernel_provider_dialect as dialect


def _probe_store(home, monkeypatch=None):
    """Use the shared store when present, or a file-backed contract fixture."""
    name = "stateframe.send1_probe_cache"
    # Keep structural counts independent of process scheduling. The separate age
    # tests move the stored timestamps and the live slot clock explicitly.
    clock = SimpleNamespace(monotonic=lambda: 1000., monotonic_ns=lambda: 10**12,
                            time=lambda: 1_700_000_000.)
    if monkeypatch is None:
        endpoint_module.time = clock
        endpoint_module._PROVIDER_EPOCH_TIMEOUT = 5.
    else:
        monkeypatch.setattr(endpoint_module, "time", clock)
        monkeypatch.setattr(endpoint_module, "_PROVIDER_EPOCH_TIMEOUT", 5.)
    if util.find_spec(name) is not None:
        from stateframe.engine.kernel_request_auth_identity import protected_home_identity

        protected_home_identity(initialize=True)
        module = import_module(name)
        if monkeypatch is None:
            module.time = clock
        else:
            monkeypatch.setattr(module, "time", clock)
        return module
    module = ModuleType(name)
    module.enabled = lambda: os.environ.get("STATEFRAME_SEND1_WAVE4") == "on"
    module.host_fingerprint = lambda: Path("/proc/sys/kernel/random/boot_id").read_text().strip()

    def lookup(kind, fingerprint, *, max_age):
        path = home / (kind + "-" + fingerprint + ".json")
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def publish(kind, fingerprint, facts):
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        (home / (kind + "-" + fingerprint + ".json")).write_text(json.dumps(facts))

    module.lookup, module.publish = lookup, publish
    if monkeypatch is None:
        sys.modules[name] = module
    else:
        monkeypatch.setitem(sys.modules, name, module)
    return module


class Provider:
    instance = "first"
    configuration = "config-1"
    model_revision = "weights-1"
    advertise = True
    native = True
    available = True
    slot_status = 200
    require_auth = False
    models = ("alpha", "beta")
    epoch_redirect = None

    def __init__(self):
        self.calls = Counter()
        self.wire = []
        self.slots = [
            {"id": 1, "n_ctx": 65536, "is_processing": False},
            {"id": 3, "n_ctx": 65536, "is_processing": False},
        ]

    def commitment(self, selected):
        facts = {}
        if selected in self.models:
            facts["models"] = _facts_digest({
                "reachable": True, "models_listed": True, "model_found": True,
                "served_models": list(self.models),
            })
        if self.native and self.slot_status == 200:
            facts["capabilities"] = _facts_digest(dialect.ProviderCapabilities(
                "llama.cpp", True, frozenset(row["id"] for row in self.slots),
                "proven by /props", 200, 200,
            ).to_record())
            facts["slots"] = _facts_digest([
                {"id": row["id"], "n_ctx": row["n_ctx"]} for row in self.slots
            ])
        elif not self.native:
            facts["capabilities"] = _facts_digest(dialect.ProviderCapabilities(
                reason="openai-strict because /props was 404 and /slots was unproven",
                props_status=404, slots_status=404,
            ).to_record())
        return {
            "version": 1, "instance": self.instance, "configuration": self.configuration,
            "model_revision": self.model_revision, "facts": facts,
        }

    @contextmanager
    def listen(self, port=0):
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, status, value):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlsplit(self.path)
                path = parsed.path
                provider.calls[path] += 1
                if not provider.available:
                    self.reply(503, {})
                elif (provider.require_auth
                      and self.headers.get("Authorization") not in {"Bearer first-key", "Bearer second-key"}):
                    self.reply(401, {})
                elif path.endswith("/provider_epoch") and provider.advertise:
                    if provider.epoch_redirect is not None:
                        self.send_response(302)
                        self.send_header("Location", provider.epoch_redirect)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    query = parse_qs(parsed.query)
                    selected, = query["model"]
                    nonce, = query["nonce"]
                    self.reply(200, {"nonce": nonce, **provider.commitment(selected)})
                elif path.endswith("/models"):
                    self.reply(200, {"data": [{"id": model} for model in provider.models]})
                elif path == "/props" and provider.native:
                    self.reply(200, {"default_generation_settings": {"n_ctx": 65536}})
                elif path == "/slots" and provider.native:
                    self.reply(provider.slot_status, provider.slots)
                else:
                    self.reply(404, {})

            def do_POST(self):
                provider.wire.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.reply(200, {})

        with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
            thread.start()
            try:
                yield Endpoint(f"http://127.0.0.1:{server.server_port}/v1", "alpha", timeout_s=5.)
            finally:
                server.shutdown()
                thread.join()


@pytest.fixture
def store(tmp_path, monkeypatch):
    if not Path("/proc/self/ns/time").is_symlink():
        pytest.skip("persistent provider proofs require a Linux time namespace")
    monkeypatch.setenv("STATEFRAME_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", "on")
    return _probe_store(tmp_path / "home", monkeypatch)


def preflight(endpoint, route="default"):
    epoch = endpoint.provider_epoch(route)
    model = endpoint.probe(epoch=epoch)
    caps = dialect.capabilities_for(endpoint.chat_url, endpoint.api_key(), cache={}, environ={}, epoch=epoch)
    return model, caps, epoch


def test_provider_commitment_refuses_redirects(store):
    provider = Provider()
    provider.epoch_redirect = "/v1/models"
    with provider.listen() as endpoint:
        assert endpoint.provider_epoch("default") is None
    assert provider.calls == Counter({"/v1/provider_epoch": 1})


def _subprocess_preflight(base, model, fixed_slot):
    from stateframe.runner import _idle_slot

    _probe_store(Path(os.environ["STATEFRAME_HOME"]))
    endpoint = Endpoint(base, model)
    model_facts, caps, epoch = preflight(endpoint)
    slots = endpoint.get_slots(epoch=epoch, static=fixed_slot)
    assert model_facts["model_found"] is True
    assert min(row["n_ctx"] for row in slots) == 65536
    selected = 1 if fixed_slot else _idle_slot(slots)[0]
    body = {"model": model, "messages": [{"role": "user", "content": "probe"}], "id_slot": selected}
    dialect.require_selected_request_features(caps, body, environ={})
    request = Request(endpoint.chat_url, data=json.dumps(body, separators=(",", ":")).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=1) as response:
        assert response.status == 200


def test_cross_process_static_work_and_wire_parity(store):
    provider = Provider()
    with provider.listen() as endpoint:
        args = [sys.executable, __file__, endpoint.base_url, endpoint.model, "fixed"]
        pairs = [
            ("", Counter({"/v1/models": 1, "/props": 1, "/slots": 2})),
            ("off", Counter({"/v1/models": 1, "/props": 1, "/slots": 2})),
            ("on", Counter({"/v1/provider_epoch": 1, "/v1/models": 1, "/props": 1, "/slots": 1})),
            ("on", Counter({"/v1/provider_epoch": 1})),
        ]
        for selection, work in pairs:
            provider.calls.clear()
            env = child_env()
            if selection:
                env["STATEFRAME_SEND1_WAVE4"] = selection
            else:
                env.pop("STATEFRAME_SEND1_WAVE4", None)
            subprocess.run(args, env=env, check=True, timeout=15)
            assert provider.calls == work
        assert len(provider.wire) == len(pairs)
        assert len(set(provider.wire)) == 1

        provider.calls.clear()
        provider.slots[0]["is_processing"] = True
        subprocess.run([*args[:-1], "automatic"], env=child_env(), check=True, timeout=15)
        assert provider.calls == Counter({"/v1/provider_epoch": 1, "/slots": 1})
        assert json.loads(provider.wire[-1])["id_slot"] == 3


def test_runner_preflight_shares_proof_and_reads_automatic_occupancy(store, tmp_path, monkeypatch):
    from stateframe import runner

    class Prepared(Exception):
        pass

    selected = []

    def prepare(options, **_kwargs):
        selected.append(options._cache_slot)
        raise Prepared

    monkeypatch.setattr(runner, "_check_engine_python", lambda _options: None)
    monkeypatch.setattr(runner, "_probe_capabilities", lambda _options: {"display": "fixture"})
    monkeypatch.setattr(runner, "_select_test_python", lambda *_args: "")
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "time", endpoint_module.time)
    provider = Provider()
    with provider.listen() as endpoint:
        cases = [
            ("off", 1, Counter({"/v1/models": 1, "/props": 1, "/slots": 2})),
            ("on", 1, Counter({"/v1/provider_epoch": 1, "/v1/models": 1, "/props": 1, "/slots": 1})),
            ("on", 1, Counter({"/v1/provider_epoch": 1})),
            ("on", None, Counter({"/v1/provider_epoch": 1, "/slots": 1})),
        ]
        for pilot, fixed_slot, calls in cases:
            monkeypatch.setenv("STATEFRAME_SEND1_WAVE4", pilot)
            provider.calls.clear()
            provider.slots[0]["is_processing"] = fixed_slot is None
            options = runner.RunOptions(repo=tmp_path, task="check", endpoint=endpoint, slot=fixed_slot)
            with pytest.raises(Prepared):
                runner._run(options, time.monotonic(), lambda _line: None)
            assert provider.calls == calls
    assert selected == [1, 1, 1, 3]


def test_routed_runner_keeps_automatic_slot_occupancy_live(store, tmp_path):
    from stateframe import runner

    base, alternate = Provider(), Provider()
    with base.listen() as default, alternate.listen() as routed:
        for expected in (1, 3):
            alternate.calls.clear()
            alternate.slots[0]["is_processing"] = expected == 3
            options = runner.RunOptions(
                repo=tmp_path, task="check", endpoint=default,
                route_cli=[f"read.endpoint={routed.base_url}", "read.model=beta", "read.slot=auto"],
            )
            options._send1_wave4 = True
            runner._probe_routes(options, lambda _line: None)
            route = options._routes.roles["read"]
            assert route["model"] == "beta"
            assert route["slot"] == expected
            assert alternate.calls["/v1/provider_epoch"] == 1
        assert alternate.calls == Counter({"/v1/provider_epoch": 1, "/slots": 1})


def test_live_occupancy_and_changed_static_slots(store):
    from stateframe.runner import _idle_slot

    provider = Provider()
    with provider.listen() as endpoint:
        assert preflight(endpoint)[1].slot_ids == {1, 3}
        provider.calls.clear()
        provider.slots[0]["is_processing"] = True
        _, caps, epoch = preflight(endpoint)
        static = endpoint.get_slots(epoch=epoch, static=True)
        assert _idle_slot(static) == (None, False)
        assert _idle_slot(endpoint.get_slots(epoch=epoch)) == (3, True)
        assert caps.slot_ids == {1, 3}
        assert provider.calls == Counter({"/v1/provider_epoch": 1, "/slots": 1})
        run_cache = {dialect.origin_of(endpoint.base_url): caps}

        # A fresh slot response can contradict an earlier live commitment.
        provider.slots.pop()
        provider.slots[0]["n_ctx"] = 32768
        assert endpoint.get_slots(epoch=epoch) == provider.slots
        assert epoch.invalidated
        capabilities = dialect.capabilities_for(endpoint.chat_url, "", cache=run_cache, environ={}, epoch=epoch)
        assert capabilities.slot_ids == {1}
        with pytest.raises(dialect.ProviderCapabilityError):
            dialect.require_routed_static_slot(capabilities, 3)

        # Static digest changes force revalidation even with an unchanged epoch label.
        provider.calls.clear()
        model, capabilities, epoch = preflight(endpoint)
        assert model["model_found"] and capabilities.slot_ids == {1}
        assert endpoint.get_slots(epoch=epoch, static=True) == [{"id": 1, "n_ctx": 32768}]
        assert provider.calls == Counter({
            "/v1/provider_epoch": 1, "/v1/models": 1, "/props": 1, "/slots": 1,
        })


def test_preflight_binding_and_model_mismatch(store, monkeypatch):
    provider = Provider()
    provider.require_auth = True
    monkeypatch.setenv("PROBE_KEY", "first-key")
    with provider.listen() as base:
        endpoint = replace(base, api_key_env="PROBE_KEY")
        preflight(endpoint)
        provider.calls.clear()
        preflight(endpoint)
        assert provider.calls == Counter({"/v1/provider_epoch": 1})

        for candidate, route in [
            (replace(endpoint, base_url=endpoint.base_url + "/other"), "default"),
            (replace(endpoint, model="beta"), "default"),
            (endpoint, "edit"),
        ]:
            provider.calls.clear()
            assert preflight(candidate, route)[0]["model_found"] is True
            assert provider.calls[urlsplit(candidate.base_url).path + "/models"] == 1
            assert provider.calls["/props"] == 1

        monkeypatch.setenv("PROBE_KEY", "second-key")
        provider.calls.clear()
        assert preflight(endpoint)[0]["model_found"] is True
        assert provider.calls["/v1/models"] == provider.calls["/props"] == 1
        for field in ("model_revision", "configuration"):
            setattr(provider, field, "changed")
            provider.calls.clear()
            preflight(endpoint)
            assert provider.calls["/v1/models"] == provider.calls["/props"] == 1
        provider.models = ("beta",)
        model, _, epoch = preflight(endpoint)
        assert model["model_found"] is False and epoch.invalidated


def test_replacement_dialect_and_missing_instance(store):
    old = Provider()
    with old.listen() as endpoint:
        port = urlsplit(endpoint.base_url).port
        assert preflight(endpoint)[1].dialect == "llama.cpp"
    replacement = Provider()
    replacement.instance = "second"
    with replacement.listen(port) as endpoint:
        assert preflight(endpoint)[1].dialect == "llama.cpp"
        assert replacement.calls["/props"] == 1
        replacement.native = False
        replacement.calls.clear()
        assert preflight(endpoint)[1].dialect == "openai-strict"
        assert replacement.calls["/props"] == 1
        replacement.calls.clear()
        assert preflight(endpoint)[1].dialect == "openai-strict"
        assert replacement.calls == Counter({"/v1/provider_epoch": 1})

        replacement.instance = ""
        replacement.calls.clear()
        model, caps, epoch = preflight(endpoint)
        assert epoch is None and model["model_found"] and caps.dialect == "openai-strict"
        assert replacement.calls["/v1/models"] == replacement.calls["/props"] == 1
        replacement.advertise = False
        assert endpoint.provider_epoch("default") is None


def test_expired_corrupt_unavailable_and_refused_facts(store, monkeypatch):
    provider = Provider()
    with provider.listen() as endpoint:
        preflight(endpoint)
        lookup = store.lookup

        def stale_or_corrupt(kind, fingerprint, *, max_age):
            entry = lookup(kind, fingerprint, max_age=max_age)
            if entry and kind == "provider-models":
                entry["observed_ns"] -= int(max_age * 1e9)
            elif entry:
                entry["facts"] = {"invalid": True}
            return entry

        monkeypatch.setattr(store, "lookup", stale_or_corrupt)
        provider.calls.clear()
        assert preflight(endpoint)[1].slot_ids == {1, 3}
        assert provider.calls["/v1/models"] == provider.calls["/props"] == 1

        def unavailable(*_args, **_kwargs):
            raise PermissionError("fixture cache is inaccessible")

        monkeypatch.setattr(store, "lookup", unavailable)
        provider.calls.clear()
        assert preflight(endpoint)[1].slot_ids == {1, 3}
        assert provider.calls["/v1/models"] == provider.calls["/props"] == 1
        monkeypatch.setattr(store, "lookup", lookup)

        epoch = preflight(endpoint)[2]
        epoch.observed_at -= 1.
        provider.calls.clear()
        assert endpoint.get_slots(epoch=epoch, static=True) == provider.slots
        assert provider.calls == Counter({"/slots": 1})

        epoch = preflight(endpoint)[2]
        provider.slot_status = 403
        assert endpoint.get_slots(epoch=epoch) is None
        assert epoch.invalidated
        assert not preflight(endpoint)[1].slots

        provider.available = False
        assert endpoint.provider_epoch("default") is None
        with pytest.raises(EndpointError, match="HTTP 503"):
            endpoint.probe()
        monkeypatch.setattr(dialect, "PROBE_RETRY_DELAYS", (0, 0, 0))
        caps = dialect.capabilities_for(endpoint.chat_url, "", cache={}, environ={})
        warning = dialect.probe_degradation_warning(caps, endpoint.base_url)
        assert warning and "4 attempt(s)" in warning


def test_slot_observation_rejects_future_stale_and_wrong_models(monkeypatch):
    endpoint = Endpoint("http://127.0.0.1:1234/v1", "alpha")
    observation = Endpoint.SlotObservation([], 100., endpoint.base_url, "", endpoint.model)
    monkeypatch.setattr(time, "monotonic", lambda: 100.5)
    assert observation.current_slots(endpoint) == []
    assert observation.current_slots(replace(endpoint, model="beta")) is None
    monkeypatch.setattr(time, "monotonic", lambda: 101.)
    assert observation.current_slots(endpoint) is None
    monkeypatch.setattr(time, "monotonic", lambda: 99.)
    assert observation.current_slots(endpoint) is None


if __name__ == "__main__":
    _subprocess_preflight(sys.argv[1], sys.argv[2], sys.argv[3] == "fixed")
