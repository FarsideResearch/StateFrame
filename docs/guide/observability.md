# Observability

The optional exporter sends historical traces from finished runs over OTLP/HTTP. It is off by
default and runs only when you invoke `stateframe-otel`. The supported commands are `export`,
`status`, and `purge`.

## Send a finished run

Start an OTLP/HTTP receiver. The default address is `http://127.0.0.1:4318`.

```bash
stateframe-otel export "$RUN" --trust-source
stateframe-otel status "$RUN" --json
```

Use `--trust-source` on the first export of a run whose records you trust. The run directory
must be owner-only and free of unsafe parent permissions. A run with an active native lock is
refused. Use `--stopped` after verifying that an interrupted run has stopped when it has no
final summary. The exporter reads surviving records without changing the run directory.

## Trace coverage

| Span | Record |
|---|---|
| Run and attempt contexts | `spec.json` and archived attempt specs. A resume creates another attempt context in the same trace. Chat tasks have separate run directories and traces. |
| Model calls | Request and response records in native journals, including worker journals. A send log can provide a fallback span only when its record has a usable timestamp; native send records have no timestamp, so they can reconcile counts and add usage details but cannot recover a missing model-call span. Send logs also identify locally replayed responses. Recorded elapsed time is exported as `stateframe.duration_ns`. |
| Actions and checks | Journal action and check receipts, tool tape completions, and baseline and final check results recorded in a summary. Tool and check kinds use fixed built-in enums; unrecognized values become `other`. A failed check has outcome `fail`. |
| Parallel work | Host worker records and the workers' native journals. Each worker's calls descend from the parent attempt. |
| Attempt outcomes | The run summary or engine result. A stopped run with no summary can have a point at its last recorded evidence. |

A point span marks a recorded publication or another recorded boundary in that attempt. If an
action or check records only a duration, the point carries `stateframe.duration_s`. A model call
can carry an elapsed time even when its journal timestamps do not locate the measured interval.
The exporter cannot infer missing bounds or elapsed time. Native process lifetime, individual
frame construction and token-count timings, and operations without surviving evidence are outside
this coverage. Oversized, corrupt, ambiguous, or send-count-mismatched records make coverage
incomplete.

## Exported values and receiver settings

Every exported value is a number, boolean, fixed enum, strict version, or keyed identity hash.
Model names are replaced with installation-keyed hashes. Provider names use a fixed list and
unrecognized names become `other`. Span, event, resource, and attribute names come from the
source code. Task text, prompts, responses, reasoning, provider envelopes, source paths and
bodies, tool names, check commands, endpoint values, credentials, and environment values are
not exported.

`--include-cost true` (or `OTEL_STATEFRAME_INCLUDE_COST=true`) remains the supported cost detail opt-in. It exports only a numeric cost
amount: a nonnegative integer in microunits, paired with a fixed `reported` or `estimated` kind.
No currency code or free-form cost text is exported. An estimate is available when both rates and
valid token usage are present. Set `price_per_million_input_tokens` and
`price_per_million_output_tokens` together in `[detail]` to estimate cost. Cost detail is off by
default. The model hash, fixed provider, and strict engine version are part of the base trace.
The provider is taken from a closed list; runs do not record a provider name, so it is `other`.

The generic `OTEL_EXPORTER_OTLP_ENDPOINT` receives a `/v1/traces` suffix.
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` selects the complete URL. Endpoints cannot contain a query,
fragment, or user information. Plain HTTP accepts only loopback IP literals; use HTTPS for other
destinations and for configured headers or TLS files. Header values remain operator configuration
and are never copied into trace records. Export refuses headers that contain application
credentials.

Use `--protocol http/json` for JSON or the default `http/protobuf`. Use `--compression gzip` to
compress requests and `--timeout-ms` (also `--max-export-time-milliseconds`) for a per-batch
deadline in milliseconds (100 to 30,000, default 10,000). Transport settings also accept generic
and trace-specific OTLP environment forms for `PROTOCOL`, `COMPRESSION`, `TIMEOUT`, `HEADERS`,
`CERTIFICATE`, `CLIENT_CERTIFICATE`, and `CLIENT_KEY`.

`--config FILE` takes an absolute owner-only TOML file outside the repository, run directory,
and tool grants. The supported tables are `[otlp]`, `[traces]`, `[sampling]`, and `[detail]`.
CLI, trace-specific environment, generic environment, trace TOML, generic TOML, and defaults
determine settings in that order. Requests use direct HTTP connections without proxies or
redirects. `OTEL_SDK_DISABLED=true` refuses export. `--traces none` also disables it.
Sampling defaults to `always_on`. Change it with `--trace-sampler` and
`--trace-sampling-ratio` (also `--sample` and `--sample-ratio`), or
`OTEL_TRACES_SAMPLER` and `OTEL_TRACES_SAMPLER_ARG`.

## Delivery and exit status

Export uses these stable exit codes.

| Code | Meaning |
|---|---|
| `0` | Coverage is complete and all selected spans are acknowledged, already delivered, or sampled out. |
| `1` | A diagnostic reports incomplete record coverage or that delivery failed. |
| `2` | A diagnostic reports refused input or configuration. |

After a receiver outage, repeat export with the same receiver settings. The private local store
retains queued traces and deduplicates acknowledged spans. A lost acknowledgement can cause
redelivery with the same IDs. Partial receiver acceptance is recorded after the local settlement
is durable; a crash before settlement can redeliver the whole batch, including items the receiver
already accepted. Once settlement is durable, repeat exports do not replay that ambiguous batch.
Receiver acknowledgement establishes acceptance by that receiver.

IDs remain stable across repeat exports, moves, copies, and resume archival with the same
installation key. `status DIR` reports the same trace ID and the local delivery counts.
`purge --yes` clears the local outbox and acknowledgement ledger after exports stop. It retains
the installation key. The next export after purge requires `--trust-source` and can send
previously acknowledged spans again.

The [trace dashboard](../assets/observability-dashboard.json) shows attempt outcomes when
your trace backend accepts the exported spans.
