# guardrails

`guardrails` is a transparent proxy for OpenAI-compatible chat-completions
servers. It is designed for local model servers such as LM Studio, where models
often produce tool calls in inconsistent formats or omit required arguments.

The proxy sits between your OpenAI-compatible client and backend. Plain chat
requests pass through unchanged. Tool-enabled, non-streaming requests are checked
and repaired before the response reaches the client.

## What It Does

- Forwards non-tool and streaming requests without rewriting the request or
  response body.
- Normalizes valid tool calls into OpenAI's `tool_calls` shape.
- Recovers tool calls from model text formats such as Qwen, Qwen-Coder, Hermes,
  Llama, Mistral, LiquidAI LFM2 / LFM2.5 (Pythonic or JSON calls wrapped in
  `<|tool_call_start|>` / `<|tool_call_end|>`), fenced JSON, and bare JSON.
- Repairs almost-JSON tool calls as a fallback when strict parsing fails:
  single-quoted strings, unquoted keys, literal newlines inside strings,
  trailing commas, and braces/brackets clipped by truncation.
- Validates tool names and JSON-object arguments against the request's declared
  tools.
- Checks required JSON-schema argument fields, preventing calls such as `Edit`
  without a required `filePath`.
- Coerces obviously-mistyped scalar arguments to the declared schema type (for
  example a stringified `"3"` for an `integer` field), repairing them in place
  instead of spending a retry.
- Repairs argument keys that name a declared property in a different casing or
  separator style (for example `file_path` for a schema's `filePath`), but only
  to fill a missing required field and only when the match is unambiguous.
- Retries invalid tool calls with a corrective nudge, then falls back safely
  instead of forwarding invalid tool calls to the client.
- Optionally injects a synthetic `respond` tool so models can return a final text
  answer through the same tool-call path.

## Request Flow

```text
OpenAI client -> guardrail proxy -> LM Studio or another OpenAI-compatible server
```

For requests without tools, or requests with `stream: true`, guardrails forwards
bytes directly.

For tool-enabled, non-streaming requests, guardrails runs this loop:

```text
backend response -> decode -> rescue -> validate -> retry or return
```

Valid native tool calls pass through unchanged. Rescued tool calls are re-emitted
in canonical OpenAI format. Invalid calls are retried up to the configured retry
budget.

## Install And Run

Start your OpenAI-compatible backend first. For LM Studio, the default local URL
is usually `http://127.0.0.1:1234`.

Run the proxy from the repository root:

```bash
cargo run -p guardrail -- \
  --listen 127.0.0.1:8080 \
  --backend http://127.0.0.1:1234
```

Then point your client at:

```text
http://127.0.0.1:8080/v1
```

## Configuration

Every option is available as both a CLI flag and an environment variable.

| CLI flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--listen` | `GUARDRAIL_LISTEN` | `127.0.0.1:8080` | Proxy listen address. |
| `--backend` | `GUARDRAIL_BACKEND` | `http://127.0.0.1:1234` | Backend base URL. |
| `--connect-timeout-secs` | `GUARDRAIL_CONNECT_TIMEOUT_SECS` | `10` | Backend connection timeout. |
| `--read-timeout-secs` | `GUARDRAIL_READ_TIMEOUT_SECS` | `300` | Maximum idle gap while reading backend responses. |
| `--max-retries` | `GUARDRAIL_MAX_RETRIES` | `2` | Maximum corrective retries per request. Set to `0` to disable retries while keeping the other repairs. |
| `--metrics-db` | `GUARDRAIL_METRICS_DB` | `~/.guardrails/guardrails.sql` | Path to the SQLite database that holds the failure-metrics table. One row is recorded per guarded request. |

Rescue, the synthetic `respond` tool, and the deterministic argument repairs
are always on. The only knob is the retry budget:

```bash
cargo run -p guardrail -- --max-retries 0
```

## Failure Metrics

Metrics are always on. The proxy records one row per guarded request to the
`outcomes` table in `~/.guardrails/guardrails.sql` (override with `--metrics-db`
or `GUARDRAIL_METRICS_DB`). The database is a general SQLite file — `outcomes`
is created with `CREATE TABLE IF NOT EXISTS`, so other tables can live alongside
it. Recording happens on a background writer thread, so it never blocks the
proxy's response path, and the database uses WAL mode so you can query it while
the proxy runs.

Each row captures the request's `model`, the terminal `outcome`, an
`error_category` (for unfixed errors), the rescue `parser`, the offending
`tool_name`, the number of `retries`, whether the guardrails `fixed` it, and a
redacted `detail` snippet for triage.

Outcomes:

| `outcome` | Meaning | `fixed` |
| --- | --- | --- |
| `native_valid` | Valid native tool call, forwarded unchanged. | 1 |
| `rescued` | Recovered from model text by a rescue parser. | 1 |
| `repaired` | Made valid by deterministic argument repair. | 1 |
| `recovered_after_retry` | Invalid, then valid after corrective retries. | 1 |
| `respond_intercept` | Synthetic `respond` tool carried the final text. | 1 |
| `fallback_unfixed` | Retries exhausted, still invalid — the errors to triage. | 0 |
| `passthrough_no_calls` | Model returned plain text, no tool call to check. | 1 |
| `non_json` | Backend response was not JSON; forwarded unverified. | 1 |

Error categories (on `fallback_unfixed`): `unknown_tool`, `bad_arguments`,
`missing_argument`, `wrong_type`.

### Viewing stats

The `stats` subcommand reads the database and prints a text report in a
**total → tool calls → errors** hierarchy per model: every guarded request
(`total`), how many were a real tool call (`tool calls`), how many of those the
guardrails could not fix (`errors`), the success rate over tool calls, the full
outcome breakdown, and the triage list of unfixed errors (with a redacted
argument snippet):

```bash
cargo run -p guardrail -- stats
```

```text
Requests by model
=================

qwen2.5-7b
  total: 168  |  tool calls: 142  |  succeeded: 137  |  errors: 5  |  success rate: 96.5%
    native_valid           110
    rescued                 18
    repaired                 9
    fallback_unfixed         5
    respond_intercept       14
    passthrough_no_calls    12

Unfixed errors (triage list)
============================

  [3x] qwen2.5-7b / missing_argument / Edit
        The arguments for tool "Edit" were missing required field "filePath". … | args: {"oldString":"a","newString":"b"}
```

`total` counts every guarded request, so it includes plain-text answers
(`passthrough_no_calls`) and final answers routed through the synthetic
`respond` tool (`respond_intercept`). Neither is a real tool call, so both are
excluded from `tool calls` and from the success rate.

### Querying directly

The metrics also answer the usual questions with plain SQL. A *tool call* is one
of `native_valid`, `rescued`, `repaired`, `recovered_after_retry`, or
`fallback_unfixed` (the same set the report uses):

```sql
-- Total requests, and of those the real tool calls, per model.
SELECT model,
       COUNT(*) AS total,
       SUM(outcome IN ('native_valid','rescued','repaired',
                       'recovered_after_retry','fallback_unfixed')) AS tool_calls
FROM outcomes
GROUP BY model;

-- Success vs. error proportion over tool calls, per model.
SELECT model,
       SUM(outcome != 'fallback_unfixed') AS succeeded,
       SUM(outcome = 'fallback_unfixed')  AS errors,
       1.0 * SUM(outcome != 'fallback_unfixed') / COUNT(*) AS success_rate
FROM outcomes
WHERE outcome IN ('native_valid','rescued','repaired',
                  'recovered_after_retry','fallback_unfixed')
GROUP BY model;

-- Errors the guardrails could not fix, by category — the triage list.
SELECT model, error_category, tool_name, detail, COUNT(*) AS n
FROM outcomes
WHERE fixed = 0
GROUP BY model, error_category, tool_name, detail
ORDER BY n DESC;
```

The sink is abstracted behind a `Recorder` trait, so an OpenTelemetry / OTLP
exporter can be added later as a second implementation without changing the
guardrail loop.

## Logging

Logs use `tracing` and default to:

```text
guardrail=info,warn
```

Override logging with `RUST_LOG`:

```bash
RUST_LOG=guardrail=debug cargo run -p guardrail
```

## Tests

Run the full test suite from the repository root:

```bash
cargo test -p guardrail
```

The integration tests cover byte-for-byte passthrough, response inspection,
rescue parsing, validation, retry behavior, and safe fallback for invalid tool
calls.

## Project Layout

```text
guardrail/src/application/  HTTP proxy and guardrail loop
guardrail/src/connector/    Backend HTTP forwarding
guardrail/src/domain/       Decode, rescue, validate, retry, and respond logic
guardrail/tests/            End-to-end proxy tests
```
