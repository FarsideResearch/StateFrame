"""Provider-wire parity and evidence-bound selection for the opt-in TypeScript slot."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from _journal import measurement_events
from _timeouts import scaled
from acceptance._support import assert_tree_imports, tree_python
from fixtures.process_signals import record_identity, signal_run_owned, signal_run_verified
from stateframe import runner
from stateframe.scripted_endpoint import Script, ScriptStepUnavailable, make_handler
from test_hook_wiring_parity import _write_launchers
from tools.pinned_tree import pinned_tree

BASE = "29765b39820fd39800fd7292f52fb5bcb81e6069"
CURRENT = Path(__file__).resolve().parents[1]
SWITCH = "STATEFRAME_PUBLIC_IDENTIFIER_SLOT"
OLD = "export const answer = 0;"
DRAFT = 'export const answer = enabled ? ops.invented("é", 7) : "invented";'
RESOLVED = DRAFT.replace("ops.invented", "ops.lookup")
SOURCE = 'import * as ops from "fixture-ops";\nconst enabled = true;\n' + OLD + "\n"
TASK = "Look up the answer using the installed module when enabled."

_TYPESCRIPT_API_FIXTURE = r'''// Offline TypeScript compiler-API fixture. It reads declarations and the
// draft source; the resolver still runs its normal sandboxed analysis script.
const fs = require("node:fs");
const path = require("node:path");
const exportsByModule = new Map();
function sourceFile(file, text) {
    return {kind: "SourceFile", fileName: file, text, getSourceFile() { return this; },
        getText(node) { return text.slice(node.start, node.end); }};
}
function identifier(source, text, start) {
    return {kind: "Identifier", text, start, end: start + text.length,
        getStart() { return this.start; }, getEnd() { return this.end; },
        getText() { return source.text.slice(this.start, this.end); }};
}
function parseCalls(source) {
    const calls = [];
    const pattern = /\b([A-Za-z_$][\w$]*)\.([A-Za-z_$][\w$]*)\s*\(([^()]*)\)/g;
    for (const match of source.text.matchAll(pattern)) {
        const start = match.index, receiver = identifier(source, match[1], start);
        const name = identifier(source, match[2], start + match[1].length + 1);
        const property = {kind: "PropertyAccessExpression", expression: receiver, name,
            start, end: start + match[0].length, children: [receiver, name],
            getText() { return source.text.slice(this.start, this.end); }};
        calls.push({kind: "CallExpression", expression: property,
            arguments: match[3].trim() ? match[3].split(",") : [], start,
            end: start + match[0].length, children: [property],
            getText() { return source.text.slice(this.start, this.end); }});
    }
    return calls;
}
function parseModule(file, text) {
    const moduleSource = sourceFile(file, text), exported = [];
    const patterns = [
        /export\s+(?:declare\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*:\s*([^;]+);/g,
        /export\s+function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)/g,
        /(?:exports|module\.exports)\.([A-Za-z_$][\w$]*)\s*=\s*function(?:\s+[A-Za-z_$][\w$]*)?\s*\(([^)]*)\)/g,
    ];
    for (const pattern of patterns) for (const match of text.matchAll(pattern)) {
        let symbol = exported.find(item => item.name === match[1]);
        if (!symbol) {
            const declaration = {getSourceFile() { return moduleSource; }};
            symbol = {name: match[1], getName() { return this.name; }, flags: 1,
                declarations: [declaration], valueDeclaration: declaration,
                callable: true, signatures: []};
            exported.push(symbol);
        }
        const parameters = match[2] && match[2].trim()
            ? match[2].split(",").map(parameter => {
                const colon = parameter.indexOf(":");
                const type = colon < 0 ? "any" : parameter.slice(colon + 1)
                    .replace(/\s*=.*$/, "").trim();
                return {name: (colon < 0 ? parameter : parameter.slice(0, colon))
                    .trim().replace(/\?$/, ""), type};
            }) : [];
        symbol.signatures.push({parameters, returnType: match[3] || "unknown"});
    }
    return {source: moduleSource, exported};
}
function argumentType(argument) {
    const value = argument.trim();
    if (/^["'`]/.test(value)) return "string";
    if (/^(?:true|false)$/.test(value)) return "boolean";
    if (/^-?(?:\d+\.?\d*|\.\d+)$/.test(value)) return "number";
    return "unknown";
}
function applicable(signature, call) {
    return signature.parameters.length === call.arguments.length &&
        signature.parameters.every((parameter, index) => {
            const actual = argumentType(call.arguments[index]);
            const expected = parameter.type.split("|").map(type => type.trim());
            return actual === "unknown" || expected.includes("any") ||
                expected.includes("unknown") || expected.includes(actual);
        });
}
module.exports = {
    sys: {getDirectories(file) { return fs.readdirSync(file, {withFileTypes: true})
        .filter(entry => entry.isDirectory()).map(entry => entry.name); }},
    ModuleKind: {NodeNext: 1}, SymbolFlags: {Alias: 2, Value: 1},
    TypeFlags: {Any: 1, Unknown: 2}, DiagnosticCategory: {Error: 1},
    getPreEmitDiagnostics(program) {
        return program.getSourceFiles().flatMap(file => program.getSemanticDiagnostics(file));
    },
    flattenDiagnosticMessageText(message) { return String(message); },
    findConfigFile() { return undefined; }, getDefaultLibFileName() { return "lib.d.ts"; },
    createCompilerHost() { return {readFile(file) {
        try { return fs.readFileSync(file, "utf8"); } catch (_) { return undefined; }
    }}; },
    isCallExpression(node) { return node.kind === "CallExpression"; },
    isPropertyAccessExpression(node) { return node.kind === "PropertyAccessExpression"; },
    isIdentifier(node) { return node.kind === "Identifier"; },
    isNamespaceImport(node) { return node.kind === "NamespaceImport"; },
    isStringLiteral(node) { return node.kind === "StringLiteral"; },
    isSourceFile(node) { return node.kind === "SourceFile"; },
    getModeForUsageLocation(source) {
        try {
            return JSON.parse(fs.readFileSync(path.join(path.dirname(source.fileName), "package.json"), "utf8"))
                .type === "module" ? "import" : "require";
        } catch (_) { return "require"; }
    },
    forEachChild(node, visit) { for (const child of node.children || []) visit(child); },
    createProgram(files, _options, host) {
        const rootFile = files[0], source = sourceFile(rootFile, host.readFile(rootFile));
        const calls = parseCalls(source);
        source.children = calls;
        const importMatch = source.text.match(/import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+["']([^"']+)["']/);
        let moduleSource = null, moduleSymbol = null, binding = null, names = new Map();
        if (/\.(?:mjs|cjs|js)$/.test(rootFile) && rootFile.includes(`${path.sep}node_modules${path.sep}`)) {
            const parsed = parseModule(rootFile, source.text);
            moduleSource = parsed.source;
            names = new Map(parsed.exported.map(item => [item.name, item]));
            moduleSymbol = {declarations: [moduleSource], exports: new Set()};
            exportsByModule.set(moduleSymbol, parsed.exported);
        } else if (importMatch) {
            const specifier = {kind: "StringLiteral", text: importMatch[2]};
            const importDeclaration = {moduleSpecifier: specifier};
            binding = {kind: "NamespaceImport", parent: {isTypeOnly: false, parent: importDeclaration}};
            const moduleFile = path.join(process.cwd(), "node_modules", importMatch[2], "index.d.ts");
            const parsed = parseModule(moduleFile, host.readFile(moduleFile));
            moduleSource = parsed.source;
            names = new Map(parsed.exported.map(item => [item.name, item]));
            moduleSymbol = {declarations: [moduleSource], exports: new Set()};
            exportsByModule.set(moduleSymbol, parsed.exported);
        }
        const checker = {
            getSymbolAtLocation(node) {
                if (node && node.kind === "StringLiteral") return moduleSymbol;
                if (node && node.kind === "SourceFile" && node.fileName === rootFile) return moduleSymbol;
                if (node && node.text === (importMatch && importMatch[1])) return {declarations: [binding]};
                return undefined;
            },
            getTypeAtLocation() { return {namespace: true}; },
            getPropertyOfType(_namespace, name) { return names.get(name); },
            getExportsOfModule(symbol) { return exportsByModule.get(symbol) || []; },
            getAliasedSymbol(symbol) { return symbol; },
            getResolvedSignature(call) {
                const symbol = names.get(call.expression.name.text);
                const selected = symbol && symbol.signatures.find(signature => applicable(signature, call));
                return selected && {returnType: selected.returnType,
                    getParameters() { return selected.parameters; }};
            },
            getReturnTypeOfSignature(signature) { return {text: signature.returnType}; },
            typeToString(type) { return type.text || type.type || "unknown"; },
            getTypeOfSymbolAtLocation(symbol) {
                if (symbol && symbol.type) return {text: symbol.type};
                return {getCallSignatures() { return symbol.callable ? [true] : []; }};
            },
        };
        return {
            getSourceFile(file) { return file === rootFile ? source : undefined; },
            getSyntacticDiagnostics() { return []; }, getOptionsDiagnostics() { return []; },
            getSemanticDiagnostics(file) {
                if (file !== source) return [];
                return calls.flatMap(call => {
                    const symbol = names.get(call.expression.name.text);
                    if (!symbol) return [{code: 2339, start: call.expression.name.start,
                        category: 1, messageText: "Property does not exist"}];
                    if (!symbol.signatures.some(signature => applicable(signature, call)))
                        return [{code: 2345, start: call.expression.name.start,
                            category: 1, messageText: "No overload matches the call"}];
                    return [];
                });
            },
            getTypeChecker() { return checker; },
            getSourceFiles() { return moduleSource ? [source, moduleSource] : [source]; },
            isSourceFileDefaultLibrary() { return false; },
        };
    },
};
'''


def _f012_typescript_fixture():
    original = '            const moduleFile = path.join(process.cwd(), "node_modules", importMatch[2], "index.d.ts");'
    subpath_aware = r'''            const parts = importMatch[2].split("/");
            const packageParts = importMatch[2].startsWith("@") ? 2 : 1;
            const packageName = parts.slice(0, packageParts).join("/");
            const subpath = parts.slice(packageParts).join("/");
            const declaration = subpath ? subpath.replace(/(?:\.js)?$/, ".d.ts") : "index.d.ts";
            const moduleFile = path.join(process.cwd(), "node_modules", packageName, declaration);'''
    assert original in _TYPESCRIPT_API_FIXTURE
    return _TYPESCRIPT_API_FIXTURE.replace(original, subpath_aware, 1)


def _script():
    return Script({"slots": [65536], "tokenize": {"mode": "ok"}, "steps": [
        {"tool": "read", "required": True, "arguments": {"path": "caller.ts"}},
        {"tool": "str_replace", "required": True, "arguments": {
            "path": "caller.ts", "old_str": OLD, "new_str": DRAFT}},
        {"tool": "run_tests", "required": True, "arguments": {"target": "test_caller.py"}},
        {"tool": "done", "required": True, "arguments": {"summary": "Updated the caller."}},
    ]})


def _capture_handler(script, received, chose=None, choice="lookup"):
    base = make_handler(script)

    class Handler(base):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return super().do_POST()
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(raw)
            self.rfile = io.BytesIO(raw)
            if chose is not None:
                body = json.loads(raw)
                schema = body.get("response_format", {}).get("json_schema", {})
                if schema.get("name") == "public_identifier_slot":
                    chose.append(body)
                    self._send(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": json.dumps({
                            "name": choice(body) if callable(choice) else choice}),
                    }}], "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108}})
                    return None
            return super().do_POST()

    return Handler


def _repo(root, expected):
    root.mkdir()
    (root / "caller.ts").write_text(SOURCE, encoding="utf-8")
    (root / "test_caller.py").write_text(
        "from pathlib import Path\n\n"
        "def test_caller():\n"
        f"    assert {expected!r} in Path('caller.ts').read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )


def _env(switch):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop(SWITCH, None)
    environment["STATEFRAME_HTTP_GZIP"] = "off"
    if switch is not None:
        environment[SWITCH] = switch
    return environment


def _launch(python, launcher, wrapper, run, trace, endpoint, repo):
    return [python, str(launcher), str(wrapper), str(trace), "run",
            "--repo", str(repo), "--task", TASK, "--endpoint", endpoint, "--model", "scripted",
            "--run-dir", str(run), "--test-python", sys.executable, "--max-turns", "16",
            "--no-run-env", "--quiet"]


@pytest.mark.host_sandbox
def test_f012_draft_caller_freely_off_request_bytes_match_pinned_099(tmp_path, tmp_path_factory):
    baseline = pinned_tree(BASE, tmp_path_factory)
    repo, run = tmp_path / "repo", tmp_path / "run"
    _repo(repo, DRAFT)
    launcher, wrapper = tmp_path / "launcher.py", tmp_path / "wrapper.py"
    _write_launchers(launcher, wrapper)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _capture_handler(_script(), []))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"

    def run_with_resume(checkout, switch, label):
        python = tree_python(checkout / "src", tmp_path / f"{label}-python")
        assert_tree_imports(python, checkout / "src", Path(checkout.anchor))
        script = _script()
        requests = []
        at_third_send, interrupted = threading.Event(), threading.Event()
        next_call = script.next_call

        def pause_once(offered):
            if script.position == 2 and not interrupted.is_set():
                at_third_send.set()
                assert interrupted.wait(scaled(120))
                raise ScriptStepUnavailable("attempt interrupted")
            return next_call(offered)

        script.next_call = pause_once
        server.RequestHandlerClass = _capture_handler(script, requests)
        trace = tmp_path / f"{label}-modules.json"
        environment = _env(switch)
        process = subprocess.Popen(_launch(python, launcher, wrapper, run, trace, endpoint, repo),
                                   env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert at_third_send.wait(scaled(600)), (run / "engine.log").read_text()
            child, = subprocess.check_output(["pgrep", "-P", str(process.pid)], text=True).split()
            child = int(child)
            identity = record_identity(child, launched_after=0, command_fragment="_child.py")
            signal_run_verified(run, child, identity, signal.SIGKILL,
                                deadline=time.monotonic() + scaled(120), what="identifier slot parity")
            interrupted.set()
            process.communicate(timeout=scaled(90))
            original = list(requests)
            result = subprocess.run(
                [python, str(launcher), str(wrapper), str(trace), "resume", str(run), "--endpoint", endpoint],
                env=environment, capture_output=True, text=True, timeout=scaled(240))
            assert result.returncode == 0, result.stdout + result.stderr
            script.assert_consumed()
            assert len(original) == 3 and len(requests) == 5
            assert json.loads(original[1])["tools"]
            assert (run / "workspace/caller.ts").read_bytes() == SOURCE.replace(OLD, DRAFT).encode()
            assert "kernel_r200_f012_draft_caller_freely" not in json.loads(trace.read_text())
            # Replayed exchanges are intercepted before the provider. Check those
            # rebuilt bytes against the provider's original completed exchanges too.
            rebuilt = [path.read_bytes() for path in sorted((run / "wire").glob("*.request.json"))]
            assert rebuilt[:2] == original[:2]
            run.rename(tmp_path / label)
            return requests, rebuilt
        finally:
            interrupted.set()
            if process.poll() is None:
                signal_run_owned(run, process, signal.SIGKILL, deadline=time.monotonic(),
                                 what="identifier slot parity cleanup", require_checkpoint_saved=False)
                process.communicate(timeout=scaled(10))

    try:
        reference = run_with_resume(baseline, None, "reference")
        assert run_with_resume(CURRENT, None, "default") == reference
        assert run_with_resume(CURRENT, "false", "explicit-off") == reference
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_f012_draft_caller_freely_switch_defaults_off(monkeypatch):
    from stateframe import profiles

    monkeypatch.delenv(SWITCH, raising=False)
    assert SWITCH not in runner.resolve_opt_in_switches()
    assert SWITCH not in profiles.load("r2hc3")
    assert SWITCH in runner.OPT_IN_SWITCHES
    assert SWITCH in runner.engine_switches()
    monkeypatch.setenv(SWITCH, "false")
    assert runner.resolve_opt_in_switches()[SWITCH] == "off"
    monkeypatch.setenv(SWITCH, "on")
    assert runner.resolve_opt_in_switches()[SWITCH] == "on"


def test_f012_observation_interruption_resumes_without_new_run(tmp_path, monkeypatch):
    import kernel_r200_f012_draft_caller_freely as slot
    import kernel_tool_tape as tool_tape
    from kernel_measurement import AuditJournal, RequestCapture
    from stateframe import profiles, resume

    root = tmp_path / "workspace"
    _repo(root, RESOLVED)
    path = root / "caller.ts"
    journal = AuditJournal(tmp_path, "fixture", {})
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root, workspace_revision=3),
                              journal=journal, snapshot_id="fixture")
    tape = tool_tape.ToolTape(tmp_path)
    monkeypatch.setattr(tool_tape, "_active_tape", tape)
    start = DRAFT.encode().index(b"invented")
    evidence = slot.Evidence.parse({
        "module": "fixture-ops", "names": ["lookup", "search"],
        "runtime_condition": "import", "runtime_entry": "runtime/index.mjs",
        "spelling": "invented",
        "call": 'ops.invented("é", 7)', "start": start, "end": start + len("invented"),
        "revision": "a" * 64,
    }, DRAFT.encode())
    assert evidence is not None
    monkeypatch.setattr(slot, "_probe", lambda *args: evidence)
    sends = []

    def send(_capture, _body, _endpoint, _key):
        sends.append(True)
        return {"finish_reason": "stop", "message": {"content": '{"name":"lookup"}'}}

    def interrupt_after_observation(*_args, **_kwargs):
        raise KeyboardInterrupt("simulated interruption after durable observation")

    def refuse_split_write(**_kwargs):
        raise AssertionError("feature observations must not expose a split intent/completion")

    monkeypatch.setattr(tape, "intent", refuse_split_write)
    monkeypatch.setattr(tape, "complete", refuse_split_write)
    monkeypatch.setattr(RequestCapture, "send", interrupt_after_observation)

    (tmp_path / "parent").mkdir()
    (tmp_path / "wire").mkdir()
    (tmp_path / "sends.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "spec.json").write_text(json.dumps({
        "schema": "stateframe-run/1", "environment": {}, "public_row": {},
        "parent_snapshot": str(tmp_path / "parent"), "chat_url": "http://127.0.0.1/v1",
        "model": "scripted", "max_turns": 8, "repo": str(root),
        "engine_source": {"shipped_hash": profiles.engine_source_summary()["shipped_hash"]},
    }), encoding="utf-8")

    def resolve():
        return slot.resolve_edit(runtime, path, OLD, DRAFT, step=8, call_id="interrupted",
                                 task=TASK, endpoint="fixture", key="", model="scripted")

    with pytest.raises(KeyboardInterrupt, match="after durable observation"):
        resolve()
    assert len(tape.pending()) == 0
    assert next(iter(tape._records.values())).completed
    assert sends == []
    assert path.read_bytes() == SOURCE.encode()

    # The observation is repeatable and already complete, so resume can accept
    # it before the resolver revalidates the current compiler evidence.
    assert resume.assess(tmp_path)["run"] == tmp_path.resolve()

    # Resume rebuilds the append-only tape after the process interruption.
    resumed_tape = tool_tape.ToolTape(tmp_path)
    assert len(resumed_tape.pending()) == 0
    monkeypatch.setattr(tool_tape, "_active_tape", resumed_tape)
    monkeypatch.setattr(RequestCapture, "send", send)
    resolved, stale = resolve()
    assert (resolved, stale) == (RESOLVED, False)
    assert len(resumed_tape.pending()) == 0
    assert len(sends) == 1


def test_f012_probe_mounts_engine_python_runtime(tmp_path, monkeypatch):
    import kernel_exec_capability as capability
    import kernel_exec_session as exec_session
    import kernel_r200_f012_draft_caller_freely as slot

    workspace, repo, run = (tmp_path / name for name in ("workspace", "repo", "run"))
    workspace.mkdir()
    run.mkdir()
    compiler = repo / "node_modules/typescript/lib/typescript.js"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("module.exports = {};", encoding="utf-8")
    (run / "spec.json").write_text(json.dumps({"repo": str(repo)}), encoding="utf-8")
    node = tmp_path / "node"
    node.write_text("test node", encoding="utf-8")
    runtime_root = tmp_path / "uv-python-runtime"
    runtime_root.mkdir()
    observed = {}

    class Session:
        containment_verified = True

        def __init__(self, _directory):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run(self, argv, cap, *, stdin):
            observed.update(argv=argv, cap=cap, inputs=json.loads(stdin))
            return SimpleNamespace(exit_code=0, timed_out=False, output_truncated=False,
                                   output="null", cleanup=SimpleNamespace(ok=True))

    monkeypatch.setattr(slot.shutil, "which", lambda _name: str(node))
    monkeypatch.setattr(capability, "system_runtime_read_roots", lambda: ())
    monkeypatch.setattr(capability, "executable_read_roots", lambda _path: ())
    python_roots_calls = []

    def engine_python_roots(path):
        python_roots_calls.append(Path(path))
        return (runtime_root,)

    monkeypatch.setattr(capability, "python_runtime_read_roots", engine_python_roots)
    monkeypatch.setattr(exec_session, "ExecSession", Session)
    assert slot._probe(workspace, run, {"path": "caller.ts", "source": "a" * 64,
                                        "start": 0, "end": 1, "draft": "x"}) is None
    assert python_roots_calls == [Path(sys.executable)]
    assert runtime_root in observed["cap"].read_roots
    assert observed["argv"][:2] == [str(node), "-e"]


@pytest.mark.host_sandbox
def test_f012_draft_caller_freely_on_path_is_bounded_and_idempotent(tmp_path, monkeypatch):
    import kernel_r200_f012_draft_caller_freely as slot
    import kernel_tool_tape as tool_tape
    from kernel_action_identity import canonical_edit
    from kernel_measurement import AuditJournal, RequestCapture
    from kernel_request_kind import (
        RequestKindError, classify, guard_physical_send, validate_auxiliary_response,
    )

    real_probe = slot._probe
    real_send = RequestCapture.send
    root = tmp_path / "workspace"
    _repo(root, RESOLVED)
    path = root / "caller.ts"
    journal = AuditJournal(tmp_path, "fixture", {})
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root, workspace_revision=3),
                              journal=journal, snapshot_id="fixture")
    tape = tool_tape.ToolTape(tmp_path)
    monkeypatch.setattr(tool_tape, "_active_tape", tape)
    assert not slot.requires_full_replay()
    start = DRAFT.encode().index(b"invented")
    payload = {"module": "fixture-ops", "names": ["lookup", "search"],
               "runtime_condition": "import", "runtime_entry": "runtime/index.mjs",
               "spelling": "invented",
               "call": 'ops.invented("é", 7)', "start": start, "end": start + len("invented"),
               "revision": "a" * 64}
    evidence = slot.Evidence.parse(payload, DRAFT.encode())
    assert evidence is not None
    monkeypatch.setattr(slot, "_probe", lambda *args: evidence)
    sends = []
    answer = '{"name":"lookup"}'

    def send(capture, body, endpoint, key):
        assert capture.request_kind == slot.PURPOSE
        assert classify(capture, runtime, body) is False
        assert classify(capture, runtime, body) is False
        with pytest.raises(RequestKindError):
            guard_physical_send(capture, runtime, {**body, "messages": body["messages"][:1]}, b"")
        with pytest.raises(RequestKindError):
            classify(capture, runtime, {**body, "tools": [{"type": "function"}]})
        assert body["max_tokens"] <= 96
        assert body["response_format"]["json_schema"]["schema"]["properties"]["name"]["enum"] == [
            "lookup", "search", None]
        prompt = json.loads(body["messages"][1]["content"])
        assert prompt["draft"] == DRAFT
        assert prompt["call"] == payload["call"]
        assert prompt["callable_exports"] == ["lookup", "search"]
        assert prompt["span"] == [start, start + len("invented")]
        sends.append(prompt)
        reply = {"finish_reason": "stop", "message": {"content": answer}}
        with pytest.raises(RequestKindError):
            validate_auxiliary_response(capture, {
                **reply, "message": {**reply["message"], "tool_calls": [{"function": {"name": "shell"}}]}})
        validate_auxiliary_response(capture, reply)
        return reply

    monkeypatch.setattr(RequestCapture, "send", send)

    def resolve(call_id, task=TASK):
        return slot.resolve_edit(runtime, path, OLD, DRAFT, step=8, call_id=call_id,
                                 task=task, endpoint="fixture", key="", model="scripted")

    assert resolve("selected") == (RESOLVED, False)
    assert resolve("selected") == (RESOLVED, False)
    assert slot.requires_full_replay()
    assert len(sends) == 1
    records = [row for row in map(json.loads, journal.path.read_text().splitlines())
               if row["kind"] == slot.PURPOSE]
    receipt, = records
    assert receipt["selected_name"] == "lookup"
    assert receipt["status"] == "selected"
    assert receipt["proposal_digest"] == canonical_edit(
        runtime.owner, path, dialect="str_replace", old=OLD, new=DRAFT)["edit_digest"]
    assert receipt["resolved_edit_digest"] == canonical_edit(
        runtime.owner, path, dialect="str_replace", old=OLD, new=RESOLVED)["edit_digest"]
    for field in ("exact_set_digest", "source_revision", "proposal_digest"):
        assert receipt[field] == sends[0][field]
    assert receipt["exact_set_digest"] == "sha256:" + hashlib.sha256(b'["lookup","search"]').hexdigest()
    assert path.read_bytes() == SOURCE.encode()

    answer = '{"name":null}'
    assert resolve("abstain") == (DRAFT, False)
    answer = '{"name":"lookup","name":"search"}'
    assert resolve("duplicate-json-field") == (DRAFT, False)
    answer = '{"name":"invented"}'
    assert resolve("outside-exact-set") == (DRAFT, False)
    assert resolve("oversize", task="x" * 8192) == (DRAFT, False)
    assert len(sends) == 4
    assert slot.Evidence.parse({**payload, "start": start + 1}, DRAFT.encode()) is None
    assert slot.Evidence.parse({**payload, "names": [f"symbol{i:02}" for i in range(49)]},
                               DRAFT.encode()) is None

    # A dependency change during decode invalidates the pending edit. An older
    # completed tape record also rejects drift before replay can reuse the answer.
    changed = slot.Evidence.parse({**payload, "revision": "b" * 64}, DRAFT.encode())
    revisions = iter([evidence, changed])
    monkeypatch.setattr(slot, "_probe", lambda *args: next(revisions))
    answer = '{"name":"lookup"}'
    assert resolve("revision-changed") == (DRAFT, True)
    monkeypatch.setattr(slot, "_probe", lambda *args: changed)
    with pytest.raises(tool_tape.ToolReplayRefused, match="evidence changed"):
        resolve("selected")
    monkeypatch.setattr(slot, "_probe", lambda *args: None)
    with pytest.raises(tool_tape.ToolReplayRefused, match="no longer available"):
        resolve("selected")
    assert len(sends) == 5
    receipts = {row["parent_tool_call_id"]: row for row in map(json.loads, journal.path.read_text().splitlines())
                if row["kind"] == slot.PURPOSE}
    assert receipts["abstain"]["status"] == "abstained"
    assert receipts["abstain"]["selected_name"] is None
    assert receipts["revision-changed"]["status"] == "stale_evidence"

    # Keep the prescribed check on the actual resolver and edit dispatcher too.
    monkeypatch.setattr(slot, "_probe", real_probe)
    monkeypatch.setattr(RequestCapture, "send", real_send)
    _exercise_installed_typescript_provider_and_dispatch(tmp_path / "dispatch")


def _exercise_installed_typescript_provider_and_dispatch(tmp_path):
    compiler = os.environ.get("F012_TEST_TYPESCRIPT")
    node_available = shutil.which("node") is not None
    tmp_path.mkdir(parents=True, exist_ok=True)
    if not node_available:
        _exercise_no_node_fallback(tmp_path)
        return
    repo, run = tmp_path / "repo", tmp_path / "run"
    _repo(repo, RESOLVED)
    (repo / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    deps = repo / "node_modules"
    ops = deps / "fixture-ops"
    ops.mkdir(parents=True)
    if compiler is None:
        typescript = deps / "typescript/lib/typescript.js"
        typescript.parent.mkdir(parents=True)
        typescript.write_text(_TYPESCRIPT_API_FIXTURE, encoding="utf-8")
    else:
        shutil.copytree(compiler, deps / "typescript")
    (ops / "package.json").write_text(json.dumps({
        "name": "fixture-ops", "version": "0.1.0", "types": "index.d.ts",
        "exports": {".": {"types": "./index.d.ts", "import": "./runtime/index.mjs",
                            "require": "./runtime/index.cjs", "default": "./runtime/index.mjs"}},
    }), encoding="utf-8")
    (ops / "runtime").mkdir()
    (ops / "runtime/index.mjs").write_text(
        'export function lookup(key, limit) { return arguments.length === 1 ? `bytes:${key}` : `text:${key}:${limit}`; }\n'
        'export function search(key, limit) { return `search:${key}:${limit}`; }\n', encoding="utf-8")
    (ops / "runtime/index.cjs").write_text(
        'exports.lookup = function lookup(key, limit) { return `text:${key}:${limit}`; };\n'
        'exports.search = function search(key, limit) { return `search:${key}:${limit}`; };\n', encoding="utf-8")
    if compiler is None:
        (ops / "index.d.ts").write_text(
            "export declare function lookup(key: string): Uint8Array;\n"
            "export declare function lookup(key: string, limit: number): string;\n"
            "export declare function search(key: string, limit: number): string;\n"
            "export declare const version: string;\n", encoding="utf-8")
    else:
        (ops / "index.d.ts").write_text(
            "export declare function lookup(key: string): Uint8Array;\n"
            "export declare function lookup(key: string, limit: number): string;\n"
            'export {find as search} from "./implementation";\n'
            'export type * from "./types";\n'
            "export declare const version: string;\n"
            "declare function internal(): void;\n"
            "export type {internal};\n", encoding="utf-8")
        (ops / "implementation.d.ts").write_text(
            "export declare function find(key: string, limit: number): string;\n", encoding="utf-8")
        (ops / "types.d.ts").write_text(
            "export declare function erased(): void;\n", encoding="utf-8")
    launcher, wrapper, trace = (tmp_path / name for name in ("launcher.py", "wrapper.py", "trace.json"))
    _write_launchers(launcher, wrapper)
    python = sys.executable
    environment = _env("on")
    environment["PYTHONPATH"] = str(CURRENT / "src")
    script, requests, selections = _script(), [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _capture_handler(script, requests, selections))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            _launch(python, launcher, wrapper, run, trace,
                    f"http://127.0.0.1:{server.server_port}/v1", repo),
            env=environment, capture_output=True, text=True, timeout=scaled(240))
        assert result.returncode == 0, result.stdout + result.stderr
        script.assert_consumed()
        selected, = selections
        assert selected["response_format"]["json_schema"]["schema"]["properties"]["name"]["enum"] == [
            "lookup", "search", None]
        assert not selected.get("tools")
        prompt = json.loads(selected["messages"][1]["content"])
        assert prompt["runtime_condition"] == "import"
        assert prompt["runtime_entry"] == "runtime/index.mjs"
        assert (run / "workspace/caller.ts").read_bytes() == SOURCE.replace(OLD, RESOLVED).encode()
        assert _hidden_overload_result(shutil.which("node"), ops / "runtime/index.mjs", "lookup") == [
            "text:é:7", "bytes:é"]
        receipt, = (event for event in measurement_events(run)
                    if event["kind"] == "public_identifier_slot")
        assert receipt["selected_name"] == "lookup"
        assert receipt["runtime_condition"] == "import"
        assert receipt["runtime_entry"] == "runtime/index.mjs"
        assert receipt["source_revision"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    _exercise_heldout_package_versions(tmp_path / "heldout", compiler)


def _hidden_overload_result(node, entry, name):
    script = (
        "const {pathToFileURL}=await import('node:url'); "
        "const api=await import(pathToFileURL(process.argv[1]).href); "
        "const fn=api[process.argv[2]]; "
        "console.log(JSON.stringify([fn('é',7),fn('é')]));"
    )
    result = subprocess.run([node, "--input-type=module", "-e", script, str(entry), name],
                            capture_output=True, text=True, check=False, timeout=scaled(30))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _node_resolved_package_entry(node, repo, specifier, condition):
    script = (
        "const {fileURLToPath} = await import('node:url'); "
        "const {resolve} = await import('node:path'); "
        "const loaded = await import(process.argv[1]); "
        "const entry = loaded.__f012_entry ?? loaded.default?.__f012_entry; "
        "if (typeof entry !== 'string') throw Error('fixture entry marker missing'); "
        "console.log(entry.startsWith('file:') ? fileURLToPath(entry) : resolve(entry));"
        if condition == "import" else
        "console.log(require.resolve(process.argv[1]));"
    )
    command = ([node, "--input-type=module", "-e", script, specifier]
               if condition == "import" else [node, "-e", script, specifier])
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True,
                            check=False, timeout=scaled(30))
    assert result.returncode == 0, result.stdout + result.stderr
    return Path(result.stdout.strip()).resolve()


def _node_without_import_meta_resolve(tmp_path, node):
    wrapper = tmp_path / "node18-without-import-meta-resolve"
    wrapper.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        f"node = {json.dumps(node)}\n"
        "arguments = sys.argv[1:]\n"
        "if '-e' in arguments:\n"
        "    index = arguments.index('-e') + 1\n"
        "    arguments[index] = 'import.meta.resolve = undefined;\\n' + arguments[index]\n"
        "raise SystemExit(subprocess.run([node, *arguments], check=False).returncode)\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return str(wrapper)


def test_f012_node_entry_helper_handles_node_18_without_import_meta_resolve(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("package entry compatibility requires Node.js")
    repo = tmp_path / "repo"
    package = repo / "node_modules/fixture-ops"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({
        "name": "fixture-ops", "exports": {".": {"import": "./index.mjs"}},
    }), encoding="utf-8")
    (package / "index.mjs").write_text(
        "export const __f012_entry = import.meta.url;\n"
        "export function lookup() { return 'lookup'; }\n", encoding="utf-8")
    compatible_node = _node_without_import_meta_resolve(tmp_path, node)
    assert _node_resolved_package_entry(
        compatible_node, repo, "fixture-ops", "import"
    ) == (package / "index.mjs").resolve()


def _node_has_module_sync(node):
    result = subprocess.run([node, "-p", "String(process.features?.require_module === true)"],
                            capture_output=True, text=True, check=False, timeout=scaled(30))
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip() == "true"


def _exercise_heldout_package_versions(tmp_path, compiler):
    import kernel_r200_f012_draft_caller_freely as slot
    import kernel_tool_tape as tool_tape
    from kernel_measurement import AuditJournal, RequestCapture

    node = shutil.which("node")
    if node is None:
        return None
    tmp_path.mkdir(parents=True, exist_ok=True)
    results = []
    for version, export_name, runtime_names, condition in (
            ("1.4.2", "lookup", ["lookup", "search"], "import"),
            ("2.1.0", "search", ["search"], "import"),
            ("3.0.0", "lookup", ["lookup"], "require")):
        case = tmp_path / version
        case.mkdir()
        repo, run = case / "repo", case / "run"
        _repo(repo, DRAFT.replace("invented", export_name))
        (repo / "package.json").write_text(
            '{"type":"module"}' if condition == "import" else "{}", encoding="utf-8")
        deps = repo / "node_modules"
        ops = deps / "fixture-ops"
        runtime = ops / "runtime"
        runtime.mkdir(parents=True)
        typescript = deps / "typescript/lib/typescript.js"
        if compiler is None:
            typescript.parent.mkdir(parents=True)
            typescript.write_text(_f012_typescript_fixture(), encoding="utf-8")
        else:
            shutil.copytree(compiler, deps / "typescript")
        (ops / "package.json").write_text(json.dumps({
            "name": "fixture-ops", "version": version,
            "exports": {".": {
                "types": "./index.d.ts",
                "import": {"types": "./index.d.ts", "default": "./runtime/import.mjs"},
                "require": {"types": "./index.d.ts", "default": "./runtime/require.cjs"},
            }},
        }), encoding="utf-8")
        if version == "1.4.2":
            declarations = (
                "export declare function lookup(key: string): Uint8Array;\n"
                "export declare function lookup(key: string, limit: number): string;\n"
                "export declare function search(key: string, limit: number): string;\n")
            esm = (
                'export const __f012_entry = import.meta.url;\n'
                'export function lookup(key, limit) { return arguments.length === 1 ? `bytes:v1:${key}` : `text:v1:${key}:${limit}`; }\n'
                'export function search(key, limit) { return `wrong:v1:${key}:${limit}`; }\n')
        else:
            declarations = (
                "export declare function lookup(key: string, limit: number): string;\n"
                "export declare function search(key: string): Uint8Array;\n"
                "export declare function search(key: string, limit: number): string;\n")
            esm = (
                'export const __f012_entry = import.meta.url;\n'
                'export function search(key, limit) { return arguments.length === 1 ? `bytes:v2:${key}` : `text:v2:${key}:${limit}`; }\n')
        (ops / "index.d.ts").write_text(declarations, encoding="utf-8")
        (runtime / "import.mjs").write_text(esm, encoding="utf-8")
        (runtime / "require.cjs").write_text(
            f'exports.lookup = function lookup(key, limit) {{ return arguments.length === 1 ? `bytes:v3:${{key}}` : `text:v3:${{key}}:${{limit}}`; }};\n',
            encoding="utf-8")
        run.mkdir(parents=True)
        (run / "spec.json").write_text(json.dumps({"repo": str(repo)}), encoding="utf-8")
        expected_entry = "runtime/import.mjs" if condition == "import" else "runtime/require.cjs"
        expected_behavior = ("text:v1:é:7" if version == "1.4.2"
                             else "text:v2:é:7" if version == "2.1.0"
                             else "text:v3:é:7")

        path = repo / "caller.ts"
        journal = AuditJournal(case, f"heldout-{version}", {})
        runtime_context = SimpleNamespace(
            owner=SimpleNamespace(root=repo, workspace_revision=3),
            journal=journal, snapshot_id=f"heldout-{version}")
        tape = tool_tape.ToolTape(run)
        previous_tape = tool_tape._active_tape
        previous_send = RequestCapture.send
        observed = []

        def scripted_send(capture, body, endpoint, key):
            prompt = json.loads(body["messages"][1]["content"])
            enum = body["response_format"]["json_schema"]["schema"]["properties"]["name"]["enum"]
            assert prompt["callable_exports"] == runtime_names
            assert enum == [*runtime_names, None]
            observed.append(prompt)
            return {"finish_reason": "stop", "message": {
                "content": json.dumps({"name": export_name})}}

        tool_tape._active_tape = tape
        RequestCapture.send = scripted_send
        try:
            resolved, stale = slot.resolve_edit(
                runtime_context, path, OLD, DRAFT, step=8, call_id=f"heldout-{version}",
                task=TASK, endpoint="fixture", key="", model="scripted")
        finally:
            tool_tape._active_tape = previous_tape
            RequestCapture.send = previous_send
        assert not stale
        assert resolved == DRAFT.replace("ops.invented", f"ops.{export_name}"), observed
        assert len(observed) == 1
        prompt = observed[0]
        assert prompt["runtime_condition"] == condition
        assert prompt["runtime_entry"] == expected_entry
        assert prompt["callable_exports"] == runtime_names
        assert _node_resolved_package_entry(
            node, repo, "fixture-ops", condition) == (ops / expected_entry).resolve()

        # Run the source produced by the edit through the package condition
        # used by the draft. This executes its actual callee and argument list.
        edited_source = SOURCE.replace(OLD, resolved)
        path.write_text(edited_source, encoding="utf-8")
        call_match = re.search(r"\bops\.([A-Za-z_$][A-Za-z0-9_$]*)\(([^()]*)\)", edited_source)
        assert call_match is not None and call_match.group(1) == export_name
        resolved_call = call_match.group(0)
        overload = _typescript_resolved_overload(node, repo, path, export_name)
        assert overload == {"parameter_count": 2, "parameter_types": ["string", "number"],
                            "return_type": "string"}
        behavior = _run_edited_caller(node, repo, condition)
        assert behavior == expected_behavior
        results.append({
            "version": version, "selected_name": export_name,
            "runtime_condition": prompt["runtime_condition"],
            "runtime_entry": prompt["runtime_entry"], "resolved_call": resolved_call,
            "overload": overload, "overload_result": behavior,
        })
    return results


def _typescript_resolved_overload(node, repo, caller, name):
    script = r'''const fs = require("node:fs"), ts = require(process.argv[1]);
const file = process.argv[2], name = process.argv[3];
const options = {module: ts.ModuleKind.NodeNext,
  moduleResolution: ts.ModuleResolutionKind?.NodeNext,
  target: ts.ScriptTarget?.ES2022, noEmit: true, strict: true};
const host = ts.createCompilerHost(options);
if (typeof ts.getPreEmitDiagnostics !== "function")
  host.readFile = path => { try { return fs.readFileSync(path, "utf8"); } catch (_) { return undefined; } };
const program = ts.createProgram([file], options, host);
if (typeof ts.getPreEmitDiagnostics === "function") {
  const errors = ts.getPreEmitDiagnostics(program).filter(item =>
    item.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw Error(ts.flattenDiagnosticMessageText(errors[0].messageText, "\n"));
}
const source = program.getSourceFile(file), checker = program.getTypeChecker();
let selected;
function visit(node) {
  if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) &&
      node.expression.expression.getText(source) === "ops" && node.expression.name.text === name)
    selected = checker.getResolvedSignature(node);
  ts.forEachChild(node, visit);
}
visit(source);
if (!selected) throw Error("edited caller has no resolved overload");
const returnType = typeof checker.typeToString === "function"
  ? checker.typeToString(checker.getReturnTypeOfSignature(selected)) : selected.returnType;
const parameterTypes = selected.getParameters().map(parameter =>
  typeof checker.typeToString === "function"
    ? checker.typeToString(checker.getTypeOfSymbolAtLocation(parameter, source))
    : parameter.type);
process.stdout.write(JSON.stringify({parameter_count: selected.getParameters().length,
  parameter_types: parameterTypes, return_type: returnType}));
'''
    typescript = repo / "node_modules/typescript/lib/typescript.js"
    result = subprocess.run([node, "-e", script, str(typescript), str(caller), name],
                            cwd=repo, capture_output=True, text=True, check=False,
                            timeout=scaled(60))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _run_edited_caller(node, repo, condition):
    source = (repo / "caller.ts").read_text(encoding="utf-8")
    if condition == "import":
        runner = repo / "edited-caller.mjs"
        executable = source.replace("export const answer =", "const answer =", 1)
    else:
        runner = repo / "edited-caller.cjs"
        executable = source.replace(
            'import * as ops from "fixture-ops";',
            'const ops = require("fixture-ops");', 1)
        executable = executable.replace("export const answer =", "const answer =", 1)
    executable += "\nprocess.stdout.write(JSON.stringify(answer));\n"
    runner.write_text(executable, encoding="utf-8")
    result = subprocess.run([node, str(runner)], cwd=repo, capture_output=True,
                            text=True, check=False, timeout=scaled(30))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def _exercise_no_node_fallback(tmp_path):
    import kernel_r200_f012_draft_caller_freely as slot
    import kernel_tool_tape as tool_tape
    from kernel_measurement import AuditJournal

    root = tmp_path / "workspace"
    _repo(root, DRAFT)
    path = root / "caller.ts"
    journal = AuditJournal(tmp_path, "no-node", {})
    runtime = SimpleNamespace(owner=SimpleNamespace(root=root, workspace_revision=3),
                              journal=journal, snapshot_id="no-node")
    tape = tool_tape.ToolTape(tmp_path)
    prior_tape = tool_tape._active_tape
    tool_tape._active_tape = tape
    try:
        resolved, stale = slot.resolve_edit(
            runtime, path, OLD, DRAFT, step=1, call_id="no-node",
            task=TASK, endpoint="fixture", key="", model="scripted")
        assert (resolved, stale) == (DRAFT, False)
        assert tape.records() == ()
        assert not any(row["kind"] == slot.PURPOSE
                       for row in map(json.loads, journal.path.read_text().splitlines()))
    finally:
        tool_tape._active_tape = prior_tape


@pytest.mark.host_sandbox
def test_f012_installed_typescript_provider_and_dispatch(tmp_path):
    _exercise_installed_typescript_provider_and_dispatch(tmp_path)


def _probe_fixture(repo, run, *, specifier, package, declarations, runtime_files,
                   caller_type="module"):
    import kernel_r200_f012_draft_caller_freely as slot

    repo.mkdir(parents=True)
    run.mkdir(parents=True)
    source = SOURCE.replace('"fixture-ops"', json.dumps(specifier))
    (repo / "caller.ts").write_text(source, encoding="utf-8")
    (repo / "package.json").write_text(json.dumps({"type": caller_type}), encoding="utf-8")
    deps = repo / "node_modules"
    typescript = deps / "typescript/lib/typescript.js"
    typescript.parent.mkdir(parents=True)
    typescript.write_text(_f012_typescript_fixture(), encoding="utf-8")
    package_name = specifier.split("/")[0]
    package_root = deps / package_name
    package_root.mkdir(parents=True)
    (package_root / "package.json").write_text(json.dumps(package), encoding="utf-8")
    for name, text in declarations.items():
        target = package_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for name, text in runtime_files.items():
        target = package_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (run / "spec.json").write_text(json.dumps({"repo": str(repo)}), encoding="utf-8")
    raw = source.encode()
    start = source.index(OLD)
    evidence = slot._probe(repo, run, {
        "path": "caller.ts", "source": hashlib.sha256(raw).hexdigest(),
        "start": start, "end": start + len(OLD), "draft": DRAFT,
    })
    return evidence


def test_f012_legacy_subpath_runtime_entry_uses_the_imported_subpath(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("the resolver requires Node.js")
    evidence = _probe_fixture(
        tmp_path / "repo", tmp_path / "run", specifier="fixture-ops/codec.js",
        package={"name": "fixture-ops", "main": "./runtime/index.js"},
        declarations={
            "codec.d.ts": "export declare function codecOnly(key: string, limit: number): string;\n",
            "index.d.ts": "export declare function rootOnly(key: string, limit: number): string;\n",
        },
        runtime_files={
            "runtime/index.js": "exports.rootOnly = function rootOnly() {};\n",
            "codec.js": (
                "exports.__f012_entry = __filename;\n"
                "exports.codecOnly = function codecOnly() {};\n"),
        },
    )
    assert evidence is not None
    assert evidence.runtime_entry == "codec.js"
    assert list(evidence.names) == ["codecOnly"]
    assert _node_resolved_package_entry(
        shutil.which("node"), tmp_path / "repo", "fixture-ops/codec.js", "import"
    ) == (tmp_path / "repo/node_modules/fixture-ops/codec.js").resolve()


def test_f012_commonjs_extensionless_subpath_resolves_node_entry(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("the resolver requires Node.js")
    evidence = _probe_fixture(
        tmp_path / "repo", tmp_path / "run", specifier="fixture-ops/codec",
        package={"name": "fixture-ops", "main": "./runtime/index.js"},
        declarations={
            "codec.d.ts": "export declare function codecOnly(key: string, limit: number): string;\n",
            "index.d.ts": "export declare function rootOnly(key: string, limit: number): string;\n",
        },
        runtime_files={
            "runtime/index.js": "exports.rootOnly = function rootOnly() {};\n",
            "codec.js": "exports.codecOnly = function codecOnly() {};\n",
            "codec/index.js": "exports.directoryOnly = function directoryOnly() {};\n",
        },
        caller_type="commonjs",
    )
    assert evidence is not None
    assert evidence.runtime_condition == "require"
    assert evidence.runtime_entry == "codec.js"
    assert list(evidence.names) == ["codecOnly"]
    assert _node_resolved_package_entry(
        node, tmp_path / "repo", "fixture-ops/codec", "require"
    ) == (tmp_path / "repo/node_modules/fixture-ops/codec.js").resolve()


def test_f012_runtime_entry_includes_node_module_sync_condition(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("the resolver requires Node.js")
    if not _node_has_module_sync(node):
        pytest.skip("this Node.js build does not enable the module-sync condition")
    evidence = _probe_fixture(
        tmp_path / "repo", tmp_path / "run", specifier="fixture-ops",
        package={"name": "fixture-ops", "exports": {".": {
            "types": "./index.d.ts",
            "module-sync": "./runtime/modern.mjs",
            "import": "./runtime/old.mjs",
            "require": "./runtime/require.cjs",
        }}},
        declarations={
            "index.d.ts": (
                "export declare function modern(key: string, limit: number): string;\n"
                "export declare function legacy(key: string, limit: number): string;\n"
            ),
        },
        runtime_files={
            "runtime/modern.mjs": (
                "export const __f012_entry = import.meta.url;\n"
                "export function modern() {}\n"),
            "runtime/old.mjs": "export function legacy() {}\n",
            "runtime/require.cjs": "exports.legacy = function legacy() {};\n",
        },
    )
    assert evidence is not None
    assert evidence.runtime_entry == "runtime/modern.mjs"
    assert list(evidence.names) == ["modern"]
    assert _node_resolved_package_entry(
        node, tmp_path / "repo", "fixture-ops", "import"
    ) == (tmp_path / "repo/node_modules/fixture-ops/runtime/modern.mjs").resolve()


def test_f012_heldout_versions_resolve_and_execute_the_edited_overload(tmp_path):
    results = _exercise_heldout_package_versions(
        tmp_path, os.environ.get("F012_TEST_TYPESCRIPT"))
    if results is None:
        pytest.skip("held-out package execution requires Node.js")
    assert results == [
        {"version": "1.4.2", "selected_name": "lookup", "runtime_condition": "import",
         "runtime_entry": "runtime/import.mjs", "resolved_call": 'ops.lookup("é", 7)',
         "overload": {"parameter_count": 2, "parameter_types": ["string", "number"],
                      "return_type": "string"},
         "overload_result": "text:v1:é:7"},
        {"version": "2.1.0", "selected_name": "search", "runtime_condition": "import",
         "runtime_entry": "runtime/import.mjs", "resolved_call": 'ops.search("é", 7)',
         "overload": {"parameter_count": 2, "parameter_types": ["string", "number"],
                      "return_type": "string"},
         "overload_result": "text:v2:é:7"},
        {"version": "3.0.0", "selected_name": "lookup", "runtime_condition": "require",
         "runtime_entry": "runtime/require.cjs", "resolved_call": 'ops.lookup("é", 7)',
         "overload": {"parameter_count": 2, "parameter_types": ["string", "number"],
                      "return_type": "string"},
         "overload_result": "text:v3:é:7"},
    ]


def test_f012_overload_check_rejects_wrong_argument_type(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("the TypeScript fixture check requires Node.js")
    repo = tmp_path / "repo"
    package = repo / "node_modules/fixture-ops"
    typescript = repo / "node_modules/typescript/lib/typescript.js"
    package.mkdir(parents=True)
    compiler = os.environ.get("F012_TEST_TYPESCRIPT")
    if compiler is None:
        typescript.parent.mkdir(parents=True)
        typescript.write_text(_f012_typescript_fixture(), encoding="utf-8")
    else:
        shutil.copytree(compiler, repo / "node_modules/typescript")
    (repo / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    (package / "index.d.ts").write_text(
        "export declare function lookup(key: string): Uint8Array;\n"
        "export declare function lookup(key: string, limit: boolean): string;\n",
        encoding="utf-8",
    )
    caller = repo / "caller.ts"
    caller.write_text(SOURCE.replace(OLD, RESOLVED), encoding="utf-8")

    with pytest.raises(AssertionError, match="No overload"):
        _typescript_resolved_overload(node, repo, caller, "lookup")


def test_f012_heldout_helper_handles_missing_node(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert _exercise_heldout_package_versions(tmp_path, None) is None
