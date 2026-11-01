[![StateFrame: persistent agents, real progress](docs/assets/stateframe-banner.webp)](https://farsideresearch.com/open-source/stateframe)

# StateFrame

**Preview release.**

StateFrame is an Open Agent Engine for long-running agents, built and measured on software engineering work. It never sends chat history. Every request is built fresh from the system instructions, the tools, the task, a typed state frame that holds what the task needs right now, and the newest exchange. Earlier turns stay on your machine, in the run's own records.

## Why StateFrame

Each point says whether it is **measured** (we have the number) or **by design** (it follows from how requests are built; no benchmark yet).

- **The prompt does not grow with the task.** Every request fits a fixed prompt budget (33,344 tokens in this release), however many turns came before; that holds whatever budget size is chosen. *Measured:* across a 300-request run the largest prompt was 32,159 tokens.
- **No history compaction.** There is no chat history to summarize, so nothing is lost to repeated summaries. The engine never summarizes away its authoritative state; to stay in budget it only trims lower-priority detail, and it always keeps a ledger of what was read. *Tested so far to 300 requests replayed and about 100 live;* it is designed to hold beyond that, and the evidence points that way.
- **Bigger models on the same hardware.** A 65,536-token server slot is enough, so memory a long-context cache would take can go to model weights. *By design.*
- **More agents per GPU.** Each agent's cache is capped by its fixed slot, so its memory stays flat however long the task runs. *By design;* 4 at once is measured below, and how many fit at most is not benchmarked yet.
- **Steady prompt size every turn.** The model reads about the same amount at turn 300 as at turn 3, so prefill does not creep up. *By design.* Engine time per turn also stays about flat on the build this release is cut from (same engine code, comments cleaned): on a 55-turn replay, median time per turn went from 15.2 s to 16.9 s, and on a 60-request live run the engine's own time between responses did not trend up (about -4 ms per request). Runs much longer than that are still being measured (see KNOWN_LIMITATIONS.md).
- **Behavior is enforced by the engine, not left to the model.** The engine owns the tool menu, the guards, the completion checks, the test gates and exactly what the model sees each turn, so the model cannot step outside those rails. Engine settings live in a versioned profile file you can supply with `--profile`. *By design.*
- **Weak spots get fixed in the frame, not by buying a bigger model.** The engine keeps the bookkeeping (facts, receipts, what was read and tested) in typed state instead of the model's memory, so when a model stumbles you give the frame the missing fact or guard. *Early evidence:* in our tests one such fix took a recurring finishing failure from 21 of 24 runs correct to 24 of 24.
- **Work that isn't code.** *Early evidence:* tasks that produce a report, a data file, or a plan document, each checked by an automatic test, passed 11 of 11 live runs (5/5, 3/3, 3/3) on the build just before this release. These were short tasks (3-4 model requests each); longer non-coding work hasn't been measured yet. Recipe: [Tasks whose output is files](#tasks-whose-output-is-files) and `examples/brief_to_report`.
- **Several agents on one GPU.** *Measured:* with the default settings, 4 sessions ran at once on one 48 GB GPU (an RTX 4090 48 GB variant) serving Qwen3.6-35B-A3B (a mixture-of-experts model, 35B total, about 3B active, 4-bit): 4 of 4 passed and GPU memory held at about 26.4 GB the whole time, because the model and its shared 262,144-token cache are allocated up front. Sharing has a cost: each session ran about 2 times longer than alone, while total throughput was about 1.9 times one-at-a-time. Measured on short tasks; a stock 24 GB card was not tested.
- **Any model can take any step.** Each request is self-contained (task, typed state, newest exchange), so nothing ties a run to one model's history. *By design.*
- **Identical inputs, identical requests.** Request ids are not random and volatile test output is normalized. *Measured on the build just before this release:* two runs on the same input sent byte-identical requests on 40 of 40 requests, apart from three clock fields; in live runs an edit-receipt ID can also differ between otherwise identical runs. The same check on the build this release is cut from: 40 of 40.
- **Crash-proof resume.** `stateframe resume` continues after Ctrl-C, `kill -9`, a crash or a lost server: it replays the recorded run without calling the model and checks the next request byte for byte before going live. *Measured on this release:* a killed long run resumed with its next request byte-identical.
- **Steady memory on long runs.** Run history lives in the run directory on disk, not in memory. *Measured (short runs, release-candidate build):* the engine's memory footprint held at about 1.7 GB from request 10 to request 60, growing about 1 MB per request; the engine before this fix grew about 10.5 MB per request. The 1.7 GB is a fixed base for repository analysis and grows with repository size. Longer runs are being measured.
- **You can see exactly what was sent.** Every request is recorded, with typed state, receipts and ledgers, and `stateframe inspect` shows the frame, the messages or the exact bytes of any request. *By design.*
- **Local first.** It talks to any OpenAI-compatible server on your machine, such as llama.cpp, uses the server's prompt cache, and only allows loopback endpoints unless you opt in. *By design.*

Details: [ARCHITECTURE.md](ARCHITECTURE.md), [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## Coming next

Upcoming, not in this release:

- Choosing which model handles each kind of step, for example a small model for reading and a strong model for writing code.
- Published agents-per-GPU benchmark (how many concurrent agents fit on one card).
- Measurements over longer runs than the ranges tested so far.
- Faster resume from checkpoints (on this release resume uses the full replay).
- Smaller checkpoints.
- Fully deterministic edit-receipt IDs.

## Quickstart


```bash
./install.sh
```

The installer links the command into `~/.local/bin`. If `stateframe` is not found afterwards, add that folder to your `PATH` (the installer prints the line):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

```bash
export STATEFRAME_ENDPOINT=http://127.0.0.1:8080/v1
export STATEFRAME_MODEL=my-local-model
stateframe run --repo examples/discount_bug --task "Discounts are charged wrong at checkout: a 25% discount on an 80.00 order should cost 60.00, and a 100% discount should be free. Fix the pricing code so the tests pass."
```

An excerpt from a real run on a local model (lines with local paths and the start-up banner left out):

```text
prompt budget: 33,344 tokens; 65,536-token server slot required
before: python -m pytest -q tests/test_pricing.py -> FAIL (exit 1)
   sent 27794 B, prompt 7298 tok, ... 12.547s -> read(shop/pricing.py), read(tests/test_pricing.py)
   sent 37520 B, prompt 10983 tok, ... 42.455s -> str_replace(shop/pricing.py)
   tests: pytest exit 0: ALL TESTS PASSED.
   sent 33213 B, prompt 10233 tok, ... 33.871s -> done
engine finished: SUBMITTED in 100.7s

=== final diff ===
-    return round(price * percent / 100, 2)
+    return round(price * (100 - percent) / 100, 2)

=== run summary ===
               outcome: runtime=completed checks=pass artifact=saved
        model requests: 3
             wall time: 102.4 s
         files changed: shop/pricing.py
```

```bash
stateframe chat --repo path/to/your-repo
```

## Skills

Install a reusable instruction bundle with `stateframe skill install path/to/skill`, then select it with `--skill NAME` on `run` or `chat`. The run shows a small catalog and loads the body only when it calls `skill_load`. Optional commands need `--trust-skill-tools`. The [skills guide](docs/guide/skills.md) covers the format, limits and resume rules.

## Project instructions

Add `--instructions` to `run` or `chat` to give the agent your repository's own `AGENTS.md` (or `CLAUDE.md` when there is no `AGENTS.md`), or `--instructions=FILE` to use a specific file. The text is added to the end of StateFrame's system prompt in every request, framed as untrusted repository content: it can describe conventions and commands, but it cannot add tools, widen the sandbox or override engine rules. It is off by default because a live comparison on the bundled examples showed no gain (see [known limitations](KNOWN_LIMITATIONS.md)). Details in [Running](docs/guide/running.md#project-instructions).

## What the agent can do

- **Find and read:** search the code, list files, read files, and recall what it found earlier in the run.
- **Use selected skills:** see a bounded catalog of explicitly selected instruction bundles and load one by name. See [Skills](docs/guide/skills.md).
- **Change code:** make a targeted edit, replace an exact string, insert code at a spot, create, delete, or move a file, or rewrite a whole module.
- **Run commands:** foreground shell commands and read-only background jobs with bounded, budget-aware replies, per-run dependencies, protected-test restoration and replay verification. See [agent tools](docs/guide/tools.md).
- **Inspect and organize:** read sanitized Git history, fetch public documentation when enabled, and delete or move files within the task's edit scope. Explicitly trusted command tools can be supplied with `--tools`; local MCP stdio servers can be supplied with `--mcp`.
- **Check its work:** run your tests, try code in a sandboxed probe, write and run its own checks, and read the check results.
- **Customize run boundaries:** use sandboxed hooks to add start context, guard admitted calls (including `skill_load` and MCP tools), inspect tool results, gate submissions and report final outcomes. See the [hooks guide](docs/guide/hooks.md).
- **Finish:** a run counts as done only after its checks pass. The engine makes that call, not the model.
- **Around that:** a map of your repository's code (12 languages, plus structure for HTML, CSS, JSON, TOML and YAML), an interactive chat with diff, undo and apply, resume after a stop, and inspect to see exactly what the model was sent.

**Not yet:**

- MCP supports explicit stdio tool servers. HTTP transport, resources, prompts, sampling and persistent sessions are not supported. A new engine-owned tool still needs code (see [docs/guide/extending.md](docs/guide/extending.md)).
- No persistent background jobs across resume. A stopped run records their observations but does not relaunch them.
- No executable-bit changes in the sealed result. Such command effects are refused and restored; executable files can be removed with `file_delete` but cannot be moved with `file_move`.

**Live checks on 0.2.0** (0.2.1 changes only what the model is shown about checks and edit targets; single runs unless noted, so treat them as smoke tests, not success rates):

- Qwen3.6-35B-A3B (4-bit, llama.cpp, one 48 GB GPU): the HTTP transport task in a production Python codebase was graded a correct candidate. The 7-file feature submitted with all of its public tests passing and 11 of 18 independent edge-case tests; the previous release, run on the same GPU the same day, reached 17 of 18 (the two runs made different edits from the second request on). The 8-file batch refactor ended in a no-progress stop, as the previous release did on the same task the same day (see the 0.2.2 results under "Multi-file source and repeated reads" in KNOWN_LIMITATIONS.md).
- DeepSeek V4 Flash: the 8-file batch refactor passed all 14 tests (public and hidden) in 2 of 2 runs, in 9 and 12 requests.
- The largest prompt in any of these runs was 31,834 tokens, inside the 33,344-token budget, with no FrameFit refusals.

## Tools and safety

The [agent tools guide](docs/guide/tools.md) lists the tool arguments, command policies, sandbox boundaries and replay rules. Shell commands are sandboxed by default; user hooks always run in the sandbox. Web fetch is opt-in, and host shell execution requires an explicit flag that prints a warning. Shell and workspace-writing custom commands are reconciled within limits of 16 MiB per file, 256 MiB in aggregate and 20,000 entries. The shipped product profile uses the composed loop; lean mode is not supported. The plain loop only supports foreground shell from this tool set.

## What works today


- **Install:** one command. Dependencies are pinned and hash-checked from PyPI, with an offline fallback.
- **Demos on this build:** a small bug fix, building a multi-file static site from a short spec, and an interactive chat session with diff, undo and apply all pass.
- **Real repositories** (measured on the preview engine before the final build; re-measure on the final engine pending): 13 of 14 recorded real tasks complete, with no crashes. The 14th is a long task that stops with a clear refusal once its task state no longer fits the prompt budget (see KNOWN_LIMITATIONS.md). A real multi-file bug fix in a production Python codebase was solved live with a local model and passed hidden tests. The refusal on some large files from the previous build is fixed.
- **A bigger task** (measured on the preview engine before the final build; the 0.2.0 live check of the same task is under "Live checks on 0.2.0" above): with a local model, a 7-file feature passed all of its protected tests and 17 of 18 independent edge-case tests. On that preview build, changes of around 10 files or more could stop with a refusal (see KNOWN_LIMITATIONS.md).
- **Source in view on multi-file work:** each request offers current product code before repeated-read and navigation bookkeeping. A coverage line for every editable product file tells you which lines are shown and whether they changed since the last read when that comparison is available. Files of at most 60 lines and 2 KiB may be shown whole when the budget allows. When a current failing check has a usable source location, code bookmarks keep its test code and related product code in view. *Measured:* an offline rebuild of one recorded request showed all 11 product files whole, up from 1 of 11, while request size fell from 85,664 to 77,017 bytes, below the 82,026-byte ceiling. In two live reruns of a multi-file task that had stopped on every previously observed build, both runs completed and were submitted with all 14 tests passing in 2 to 4 requests per run. This is an early two-run result, not a general success rate. See KNOWN_LIMITATIONS.md.
- **Bounded requests:** every request is system instructions, tools, the task, typed task state and the newest exchange, inside a 33,344-token prompt budget on a 65,536-token server slot. No chat history is resent.
- **No time or step limit:** a run continues until it is done or you stop it with Ctrl-C. `--max-turns N` caps the model turns and `--timeout <seconds>` caps the time, if you want either.
- **Resume:** `stateframe resume <run-dir>` continues a run stopped by Ctrl-C, `kill`, `kill -9`, a crash or power loss, or a lost model server. It replays the recorded model responses (no model calls) up to where the run stopped, checks that the next request it rebuilds matches the recorded one byte for byte, then continues live in the same run directory; the stopped attempt's files are kept under `<run-dir>/attempts/`. A resume that is itself killed during its catch-up resumes again. *Measured on this release:* a killed long run resumed this way with its next request byte-identical and then continued live. Catch-up takes time in proportion to the run's length (on an earlier preview engine, a run killed at request 118 caught up in about 37 min; the original took about 2 h). Resume refuses with a reason when the run already finished, its records are damaged, or a different engine version is installed. If the server now lists the same model under a different name, add `--model <name it lists>`; replay still checks every recorded request byte for byte, with only the model name swapped. The engine can also save checkpoints for a faster restore, but they are off by default in this release because a checkpoint restore did not rebuild requests faithfully in our tests; see "Checkpoint settings" below. On the previous engine, where checkpoint restore was faithful, a run killed at request 150 restored in 31 s.
- **Prompt caching:** the server's prompt cache is used; reuse is modest (see KNOWN_LIMITATIONS.md).
- **Speed** (measured on the preview engine before the final build; re-measure on the final engine pending): on one 48 GB GPU (RTX A6000 class) in one live run, model generation took about 89% of the wall time, and the engine added about 19 seconds per turn. Measured on one machine.
- **Safety:** work happens on a copy, the agent cannot edit protected tests, changes are applied only when you ask, and the task ledger is schema-validated.

## Checkpoint settings

| Variable | Default | Meaning |
|---|---|---|
| `STATEFRAME_CHECKPOINT` | `off` | `on`/`save` (or `1`) turns checkpoints on; `off` (or `0`) turns them off. |
| `STATEFRAME_CHECKPOINT_EVERY` | `15` | Save at most once every N engine steps (about 2.8 steps per model request, so 15 is about one save per 5 requests). Lower saves more often (less to replay on resume, more host time); `1` saves after every request. |
| `STATEFRAME_CHECKPOINT_KEEP` | `2` | How many checkpoints to keep; older ones are deleted after each save. |

Checkpoints are off by default in this release: in our tests a checkpoint restore did not rebuild the next
request faithfully, so resume refused it and replayed anyway. Turning them on (`STATEFRAME_CHECKPOINT=save
stateframe run ...`) is safe, because resume checks every restore and falls back to the full replay, but it
costs disk: each checkpoint holds about 430-460 MB of engine state on a large repository (about 1.1 GB with
the two newest kept, briefly about 1.7 GB while a new one is written). When a run with checkpoints finishes
for good, StateFrame removes them; `--keep-checkpoints` (or `STATEFRAME_KEEP_CHECKPOINTS=1`) keeps them.
Checkpoints are signed with a per-install key in `~/.stateframe/checkpoint.key` (a private file); a checkpoint
that was edited, copied from another machine, or written by another engine build is never loaded.

Precedence: an environment variable wins over a profile setting, which wins over the default. An engine
without checkpoint support ignores these settings and resume replays.

## Token counting

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_CPP_TOKENIZE_URL` | the model endpoint | Base URL of a llama.cpp server whose `/tokenize` the engine uses to count prompt tokens. Point it at a server on the same machine as `stateframe` when the model endpoint is remote, so each count skips a network round trip. It must serve the same model (same tokenizer). |

Check it before a run: send the same text to `/tokenize` on both servers and compare the token ids; they must be identical (QUICKSTART.md has the commands). Measured: a count took about 0.65 s against a remote server over our tunnel and about 0.08 s against a tokenizer on the same machine.

## Requirements

- macOS on Apple silicon (sandbox-exec, built in) or Linux x86_64 (tested on Ubuntu 22.04; needs bubblewrap: `sudo apt install bubblewrap`). Python 3.11+. Docker is not supported.
- Free disk: a few GB per long run. Measured on a large repository: a run directory uses about 3 MB per request (about 550-750 MB for a 209-request run), plus checkpoints while the run is live if you turn them on (see Checkpoint settings).
- Python 3.11 or newer, with `venv` and `ensurepip`. Python 3.14.1 is not supported: a required dependency (networkx 3.6.1) excludes that one release; any other 3.14.x works.
- A local OpenAI-compatible chat server with tool calling, configured with a context slot of 65,536 tokens. Size the server's cache RAM to your machine; to use a different slot size, see [Context size](docs/guide/running.md#context-size). These are the sizes we built and measured with: a single 48 GB GPU (RTX A6000 class) running Qwen3.6-35B-A3B, a mixture-of-experts model (35B total, about 3B active per token), 4-bit UD-Q4_K_XL, under llama.cpp, with one server sharing a 262,144-token cache across 4 slots (`-np 4 -c 262144 --kv-unified`), so 65,536 tokens per run. A stock 24 GB card was not tested. They are the defaults; both can be changed for bigger or smaller servers (see [Context size](docs/guide/running.md#context-size)).
- A model that handles tool calls well. The results above were measured with Qwen3.6-35B-A3B, a mixture-of-experts model (35B total, about 3B active per token), 4-bit UD-Q4_K_XL, under llama.cpp on a single 48 GB GPU (RTX A6000 class); other models and smaller GPUs are untested.

## Configuration

- `--endpoint` and `--model` (or `STATEFRAME_ENDPOINT` and `STATEFRAME_MODEL`) choose the server and model. Only loopback addresses are allowed by default; a remote endpoint needs `STATEFRAME_ALLOW_REMOTE_ENDPOINT=1` and prints a warning that your code will leave the machine.
- The engine checks its work with pytest test files from your repository (`--check <path>` names them; by default every `test_*.py` / `*_test.py`). A repository needs at least one. `--test-cmd` only changes the outer command StateFrame runs before and after to report pass or fail; it does not replace the engine's pytest checks.
- Your tests run with your project's Python: the repository's `.venv/`, `venv/` or `env/`, else `python3` on PATH. `--test-python PATH` picks another; it needs pytest and your project's test dependencies. `--python` sets one interpreter for both the engine and your tests.
- `--in-place` applies a passing result to your repository. Without it, nothing in your repository changes; the result stays in the run directory.
- `--hooks FILE` adds a trusted hook definition file to `run` or `chat`; it can be repeated. A project file at `.stateframe/hooks.json` runs only after you trust its exact contents with `stateframe hooks trust --repo PATH`. Use `--no-project-hooks` to skip that file for one run. See [Hooks](docs/guide/hooks.md).
- There is no time limit by default. Stop any run with Ctrl-C, or pass `--timeout <seconds>`.
- There is no step limit by default. `--max-turns N` stops a run after N model turns.
- `STATEFRAME_EVICTFIX=on` (opt-in, off by default) keeps more recent reads in the frame instead of evicting them. It is slower on long runs in this release, and we have not measured an edit gain from it. `legacy` (or `off`) is the default.
- In a profile's switch map, `STATEFRAME_CODE_BOOKMARKS` is on by default; setting it off restores the 0.2.1 behavior byte for byte. `STATEFRAME_READ_LOCKOUT` is off by default. When enabled, it can hide source-reading routes for up to two requests after proved unchanged reads, if a current failing check, editable bookmarked source, and edit and check routes are available. A run can still stop first when the no-progress watch is one step from its limit. Set these in a profile, not as shell variables; see [Profiles and switches](docs/guide/running.md#profiles-and-switches).
- `STATEFRAME_MICROGRAPH=on` is experimental and off by default. When on, the engine keeps a small, deterministic graph of facts taken from its own tool receipts (files, failing tests, error locations) and uses it to track which hypotheses about the bug have been checked; the model's free text is never parsed into it. We make no quality claims for it yet. `off` is the default; any other value is refused at start.

## Real repositories

In a repository with more than 200 source files, say which files the change may touch: `--target` names the starting file and each `--edit` adds another editable file. The whole repository stays available as read-only context. Without them, StateFrame stops and asks for them.

```bash
stateframe run --repo ~/code/shop --task "Discounts above 50% are rejected; allow up to 100%." \
  --target shop/pricing.py --edit shop/checkout.py --check tests/test_pricing.py
```

## Tasks whose output is files

StateFrame handles file-based tasks whose acceptance is a pytest file over the output. The output does not have to be code: a report, a JSON or CSV summary, or documents in a folder all work the same way, as long as a test file can check them. On the build just before this release, 11 of 11 short live runs of report, data-file and plan-document tasks passed their acceptance tests (see "Why StateFrame"); one file-creation demo also passes (`examples/tiny_site`). `examples/brief_to_report` shows the recipe for a written report: the agent reads `input/brief.md` and `input/tickets.csv` and writes `report/summary.json` and `report/report.md`. `tests/test_report.py` recomputes every number from the CSV and checks the headings.

Recipe:

1. Put the inputs in the repository and say in the task which files to write.
2. Write the acceptance as a pytest file over the output files. Check what can be checked exactly: the files exist, the JSON has the right shape, every number matches a value recomputed from the input, and the required headings and labels are present. The test cannot judge how well the prose reads.
3. Run it with `--check tests/<your test file>.py`. The run fails before and passes after, as with code.

If the test environment can import StateFrame, `stateframe.checks` provides strict checks for these files: `schema_validity`, `closed_set_membership`, `citation_span_grounding` and `deterministic_recomputation`. Each returns a pass/fail receipt with evidence, and `checks.require(...)` turns a failing receipt into a test failure. These checks have not been measured as task acceptance on StateFrame runs yet.

## Repo map

The engine gives the model a short map of the repository's functions and classes. The default install adds tree-sitter grammars, each from its own MIT-licensed package. The engine reads 12 languages with them: Python, JavaScript, TypeScript, TSX, Go, Rust, Java, Ruby, PHP, C, C++ and C#. For HTML, CSS, JSON, TOML and YAML files it reads structure instead (element ids and headings, selectors and at-rules, keys, tables), from five more grammar packages, so the installer reports 17 languages. Other languages fall back to a simpler text scan. `./install.sh --no-repomap` skips the grammar packages.

## Known limitations

See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md).

## License

StateFrame is licensed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only), or under a separate commercial license from Farside Research LLC (see COMMERCIAL-LICENSE.md). Third-party dependencies keep their own licenses; see THIRD_PARTY_LICENSES.md.

If you modify StateFrame and let others use it over a network, the AGPL requires offering them its source; a commercial license is the alternative.

## Contributing

We'd genuinely love your help. Bug reports, small fixes, docs, new benchmarks, or a wild idea about how agents should keep state: all welcome. Open an issue or a pull request and we'll take a look. First-time contributors are very welcome too.

Code contributions need a one-time CLA signature (it takes a minute, and you keep your copyright). Details are in [CONTRIBUTING.md](CONTRIBUTING.md).

Originally created by [Michael Arida](https://www.linkedin.com/in/michael-arida/).
