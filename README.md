# guardrails

`guardrails` is a transparent proxy for OpenAI-compatible servers, speaking both
the chat-completions and the Responses API. It is designed for local model servers such as LM Studio, where models
often produce tool calls in inconsistent formats or omit required arguments.

The proxy sits between your OpenAI-compatible client and backend. Plain chat
requests pass through unchanged. Tool-enabled, non-streaming requests are checked
and repaired before the response reaches the client.

## What It Does

- Forwards requests that declare no tools — including streamed ones — without
  rewriting the request or response body.
- Guards tool-enabled requests even when the client asked to stream: the upstream
  call is buffered so the response can be checked, then the guarded result is
  re-emitted to the client as SSE (the `stream: true` contract is preserved; a
  tool-calling turn just no longer streams token by token).
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

## Both APIs

`POST /v1/chat/completions` and `POST /v1/responses` are both guarded, with the
same repairs. The two protocols differ in shape — Responses carries tool calls
as `output[]` items typed `function_call` rather than `choices[].message.tool_calls`,
declares tools flat rather than nested under `function`, and streams typed events
instead of deltas — so the translation happens at the edges. The guardrails in
between (rescue, validation, argument repair, retries, the synthetic `respond`
tool) are shared, and every repair described below applies to both.

A request is guarded on whichever API it arrives on; the proxy does not convert
between them.

## Request Flow

```text
OpenAI client -> guardrail proxy -> LM Studio or another OpenAI-compatible server
```

Requests that declare no tools are forwarded bytes-for-bytes, streamed or not —
there is no tool call to check. This holds on `/v1/responses` too.

On the Responses path, text is held back the moment it starts to look like a
tool call written as prose (a `<tool_call>` marker, a fence, a leading
`{"name":`). Forwarding it and then emitting the rescued call would show the
client raw call syntax and then contradict it. Text that turns out to be
ordinary prose is released when the stream ends.

Tool-enabled requests run this loop, whether or not the client asked to stream:

```text
backend response -> decode -> rescue -> validate -> retry or return
```

Valid native tool calls pass through unchanged. Rescued tool calls are re-emitted
in canonical OpenAI format. Invalid calls are retried up to the configured retry
budget. When the client asked to stream, the upstream call is sent in buffered
(non-streaming) mode so the whole response can be inspected, and the guarded
result is re-emitted as a single SSE chunk followed by `[DONE]`.

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

### Proxying several backends

Repeat `--backend` as `NAME=URL` to put more than one server behind the same
port. Requests route by the model they name:

```bash
cargo run -p guardrail -- \
  --backend lmstudio=http://127.0.0.1:1234 \
  --backend other=http://127.0.0.1:5678
```

At startup the proxy asks each backend which models it serves and builds the
routing table from the answers, so the model list is never hand-maintained. A
backend that is unreachable at startup keeps its place — a local server is often
started after the proxy — it simply claims no models until the next restart.

Three rules make routing predictable:

- **The first backend is the default.** It serves any model no other backend
  claims, including models that discovery missed and requests that name none.
- **The first to claim a model id wins.** When two backends both serve `gpt-4o`,
  the one listed first gets it, and the other is logged rather than silently
  preferred.
- **Names must be unique**, and every backend after the first must be named, so
  its requests can be told apart in the metrics.

A single bare `--backend URL` still behaves exactly as before; it is named
`default`.

### Choosing which models are exposed

Configuration lives in `~/.guardrails/config.json`. CLI flags seed it on first
run; after that the file is the source of truth, so a change made through the
management API is not undone by whatever flags the launcher passes next time.
It is plain JSON and meant to be hand-editable.

The management API (on the admin port) reads and changes it at runtime — no
restart:

| Method & path | Does |
| --- | --- |
| `GET /providers` | Every provider, its discovered models, and which are exposed. |
| `POST /providers` | Add a provider (`name`, `base_url`, optional `unversioned`). |
| `PATCH /providers/{name}` | Set per-model exposure, `enabled`, or `expose_by_default`. |
| `DELETE /providers/{name}` | Remove a provider. |

```bash
# Hide one model.
curl -X PATCH http://127.0.0.1:8081/providers/remote-a \
  -d '{"models": {"tiny-draft": false}}'

# Or expose nothing except what you pick, for a server with a large catalogue.
curl -X PATCH http://127.0.0.1:8081/providers/remote-b \
  -d '{"expose_by_default": false, "models": {"qwen3-72b": true}}'
```

A hidden model is **not served**, not merely unlisted: it disappears from
`GET /v1/models` and a request naming it gets `404`, so what the proxy
advertises and what it will do agree. Exposure decisions are stored per model,
so a model that vanishes when a backend restarts keeps whatever you chose for
it.

### GitHub Copilot

`--copilot` adds a `copilot` provider backed by a Copilot subscription. Unlike a
plain backend it needs an OAuth credential, six client-identity headers GitHub
gates access on, and its routes live at the root rather than under `/v1` — all
supplied by [`gh-copilot-rs`](https://github.com/ArtemisMucaj/gh-copilot-rs). The
proxy's own surface stays `/v1/...`, so clients need not know the difference.

Authorize once, through the admin server's device flow:

```bash
guardrail --backend lmstudio=http://127.0.0.1:1234 \
  --copilot --admin-listen 127.0.0.1:8081

# Returns a user_code and verification_uri to open in a browser.
curl -X POST http://127.0.0.1:8081/copilot/login
# Poll until it reads {"status":"authorized"}.
curl http://127.0.0.1:8081/copilot/login
```

The token is stored at `~/.guardrails/copilot-token`, created `0600` so it is
not world-readable, and reused on later runs. Restart the proxy after
authorizing for the provider to pick it up. Until a token exists the proxy still
runs — the other providers work, and Copilot claims no models.

A client's own `Authorization` never displaces the Copilot credential: the
provider reserves that header along with the six gating ones. Without this an
OpenAI-compatible client sending `Bearer no-key`, as many do by default, would
have its placeholder forwarded to GitHub and every request would fail as a `401`
that reads like an expired token.

**Security.** The admin server is unauthenticated, and with `--copilot` it can
both start a login and front a paid subscription. The proxy therefore refuses to
start if `--admin-listen` is not a loopback address. Loopback keeps other hosts
out, but not other processes on this machine: anything running locally can use
the credential through the proxy. That is an acceptable trade on a single-user
development machine, and not on a shared one.

`GET /v1/models` returns the union across providers, each entry tagged with the
`provider` that serves it, so a client can name any routable model. An id served
by more than one provider is listed once, under the provider routing sends it to.
A provider that cannot be reached is skipped rather than emptying the list; if
none can be reached the endpoint answers `502` rather than claiming the proxy
serves no models. With a single backend the response is forwarded untouched, so
the byte-for-byte passthrough is preserved.

The prebuilt macOS release binary is signed with a Developer ID and notarized
by Apple, so it runs without a Gatekeeper exception.

## Configuration

Every option is available as both a CLI flag and an environment variable.

| CLI flag | Environment variable | Default | Description |
| --- | --- | --- | --- |
| `--listen` | `GUARDRAIL_LISTEN` | `127.0.0.1:8080` | Proxy listen address. |
| `--admin-listen` | `GUARDRAIL_ADMIN_LISTEN` | _(disabled)_ | Address for the read-only admin HTTP server (stats, health, info), on a separate port. Disabled unless set. |
| `--backend` | `GUARDRAIL_BACKEND` | `http://127.0.0.1:1234` | An OpenAI-compatible backend, as `URL` or `NAME=URL`. Repeat to proxy several; the environment variable takes a comma-separated list. |
| `--connect-timeout-secs` | `GUARDRAIL_CONNECT_TIMEOUT_SECS` | `10` | Backend connection timeout. |
| `--read-timeout-secs` | `GUARDRAIL_READ_TIMEOUT_SECS` | `300` | Maximum idle gap while reading backend responses. |
| `--max-retries` | `GUARDRAIL_MAX_RETRIES` | `2` | Maximum corrective retries per request. Set to `0` to disable retries while keeping the other repairs. |
| `--match-conversations` | `GUARDRAIL_MATCH_CONVERSATIONS` | `false` | Reconstruct conversations from Chat Completions traffic by matching resent message prefixes, so token metrics count a resent transcript once. Off by default: it is the only thing that makes the metrics path read message content (to hash it — no text is stored), and the grouping is approximate. |
| `--copilot` | `GUARDRAIL_COPILOT` | `false` | Proxy GitHub Copilot models, using a Copilot subscription. |
| `--copilot-base-url` | `GUARDRAIL_COPILOT_BASE_URL` | `https://api.githubcopilot.com` | Copilot API base URL. Override for an enterprise deployment. |

Rescue, the synthetic `respond` tool, and the deterministic argument repairs
are always on. The only knob is the retry budget:

```bash
cargo run -p guardrail -- --max-retries 0
```

## Failure Metrics

Metrics are always on. The proxy records one row per chat-completions request
to the `outcomes` table in `~/.guardrails/guardrails.sql`: tool-enabled requests
(streamed or not) get their terminal guarded outcome, while requests that declare
no tools — which have no tool call to check — are recorded as passthroughs, so
the report reflects all chat traffic instead of being empty for clients that only
ever stream. The database is a general
SQLite file — `outcomes` is created with `CREATE TABLE IF NOT EXISTS`, so other
tables can live alongside it. A database written before per-provider
stats existed lacks the `provider` column; the proxy recreates the `outcomes`
table on first run and logs that it did, discarding the previous request
history rather than leaving metrics disabled. Other tables are left alone. Recording happens on a background writer thread, so
it never blocks the proxy's response path, and the database uses WAL mode so you
can query it while the proxy runs.

Each row captures the serving `provider`, the request's `model`, the terminal `outcome`, an
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
| `retries_exhausted` | Retries exhausted, still invalid — the errors to triage. | 0 |
| `write_refused` | Write-only tool called on an existing file — model told to read first. | 0 |
| `passthrough_no_calls` | Model returned plain text, no tool call to check. | 1 |
| `streamed_passthrough` | Streaming request with no tools, forwarded live (nothing to guard). | 1 |
| `non_tool_passthrough` | Non-streaming request with no tools, forwarded unguarded. | 1 |
| `non_json` | Backend response was not JSON; forwarded unverified. | 1 |

Error categories (on `retries_exhausted`): `unknown_tool`, `bad_arguments`,
`missing_argument`, `wrong_type`.

### Viewing stats

The `stats` subcommand reads the database and prints a text report in a
**total → tool calls → errors** hierarchy per provider and model: every guarded request
(`total`), how many were a real tool call (`tool calls`), how many of those the
guardrails could not fix (`errors`), the success rate over tool calls, the full
outcome breakdown, and the triage list of errors (with a redacted argument
snippet):

```bash
cargo run -p guardrail -- stats
```

```text
Requests by provider and model
==============================

lmstudio / qwen2.5-7b
  total: 168  |  tool calls: 142  |  succeeded: 137  |  errors: 5  |  success rate: 96.5%
    native_valid           110
    rescued                 18
    repaired                 9
    retries_exhausted         5
    respond_intercept       14
    passthrough_no_calls    12
  tokens billed: 1284300  |  prompt: 1201400 (388200 new, 813200 cached)  |  completion: 82900
  cache hit rate: 67.7%  |  backend calls per request: 1.09  |  measured over 168 of 168 requests
  prompt per request:     min 412 | p50 3180 | p90 24100 | p99 96400 | max 131000
  completion per request: min 3 | p50 240 | p90 1420 | p99 3100 | max 4096

Errors (triage list)
====================

  [3x] lmstudio / qwen2.5-7b / missing_argument / Edit
        The arguments for tool "Edit" were missing required field "filePath". … | args: {"oldString":"a","newString":"b"}
```

`total` counts every chat-completions request the proxy saw, so it includes
plain-text answers (`passthrough_no_calls`), final answers routed through the
synthetic `respond` tool (`respond_intercept`), and the streaming / non-tool
requests forwarded unguarded (`streamed_passthrough`, `non_tool_passthrough`).
None of these is a real tool call, so all are excluded from `tool calls` and
from the success rate.

The sink is abstracted behind a `Recorder` trait, so an OpenTelemetry / OTLP
exporter can be added later as a second implementation without changing the
guardrail loop.

#### Token figures

Tokens are recorded per request, summed over **every** backend attempt it made —
a corrective retry is a second billed call, and `backend calls per request` is
the multiplier the guardrails add to the bill. Requests whose backend reported
no usage are left out of the totals rather than averaged in as zeroes, which is
what `measured over N of M requests` reports.

Prompt tokens **do not add up across a conversation**. Every chat turn resends
the whole transcript, so turn 5's prompt contains turns 1–4, and summing them
counts shared prefixes once per later turn — growth quadratic in turn count, not
a count of distinct tokens. The sum is therefore labelled *billed*: it is a
faithful answer to "what did the provider charge" and a wrong answer to "how
many tokens did this traffic contain". Only completion tokens are generated once
and so sum cleanly.

Two figures address that, and they answer different questions:

- **`distinct tokens: N over K conversation(s)`** counts a resent prefix once, by
  taking the largest prompt per conversation rather than the sum. It needs a
  conversation key, which only the stateful Responses API supplies (via
  `previous_response_id`). On Chat Completions the line is **absent** rather than
  repeating the inflated sum under a better name — see [#46].
- **`prompt/completion per request`** is the spread across single requests, so it
  needs no conversation key and is reported for all traffic. It is what
  distinguishes a workload where every request is 3k tokens from one where most
  are 500 and a handful are 130k; the average alone describes neither. The
  percentiles are nearest-rank, so every figure is a token count some request
  actually reported rather than an interpolated point between two of them.

For any grouping the report does not do — by hour, by outcome, or by a session
key only the client knows — `GET /requests` serves the underlying per-request
rows.

#### Grouping Chat Completions (`--match-conversations`)

`distinct tokens` is absent on Chat Completions because that API supplies no
conversation key. `--match-conversations` reconstructs one, so the line appears
for that traffic too:

```text
  distinct tokens: ~660 over ~1 conversation(s)

  ~ conversations inferred from message prefixes
```

The `~` marks a figure that rests on inferred grouping; the footnote is printed
once per report rather than repeated on every line.

The mechanism is **prefix containment**, not a fingerprint of the opening. Turn
N's `messages[]` *is* turn N−1's array plus the new entries — a property of how a
stateless API works, not a guess about what conversations look like — so a turn
is matched to its predecessor by asking whether the predecessor's messages are a
prefix of this one's. Where several turns qualify, the most recent wins; a
candidate more than two hours old is refused, on the grounds that a shared prefix
across a long gap is more likely two sessions that opened alike than one exchange
resumed.

**Message text is never stored.** What is written is a rolling hash per prefix
length (`h₁ = H(m₁)`, `h₂ = H(h₁ ‖ m₂)`, …), so containment reduces to comparing
digests and nothing is reconstructible from the database. The hash is non-
cryptographic by design: a collision merges two conversations in a local metrics
report, which is a failure the heuristic already admits, and nothing of
consequence rests on it.

It is nonetheless **off by default**, because enabling it is the one thing that
makes the metrics path read message content at all — to hash it — where it
otherwise never touches the body of a guarded request.

The grouping is approximate, and both the report (`~`) and the `/stats` JSON
(`inferred_conversations: true`) say so. Known limits:

- A **regenerated or edited turn** sends an array that diverges rather than
  extends, so it reads as a new conversation.
- Two clients **replaying an identical transcript** are indistinguishable.
- **Parallel turns** on one conversation form a tree rather than a chain. That is
  handled correctly — a conversation contributes its largest prompt regardless of
  branching — and the siblings remain separable in the raw rows, since the second
  request typically reports more `cached_tokens` than the first, having read the
  cache the first populated.

The Responses API is unaffected either way: it names its own edges, so its
deduplication is exact and is never marked approximate.

[#46]: https://github.com/ArtemisMucaj/guardrails/issues/46

### Admin HTTP server

For programmatic access — a desktop app, a dashboard, a health check — the same
stats are available over HTTP from a dedicated admin server on a **separate
port** from the proxy. It is opt-in: pass `--admin-listen` to enable it. The metrics routes are read-only; the
management and login routes change configuration, so the server is no longer
`GET`-only. A login
response carries the user code and verification URL, never the token.

```bash
cargo run -p guardrail -- \
  --listen 127.0.0.1:8080 \
  --admin-listen 127.0.0.1:8081 \
  --backend http://127.0.0.1:1234
```

Keeping it on its own port means the model-facing proxy port only ever speaks
the OpenAI protocol, and the admin surface can be bound, firewalled, or exposed
to a UI independently. Bind it to a loopback address; the metrics are not
authenticated.

| Method & path | Returns |
| --- | --- |
| `GET /healthz` | `{"status":"ok"}` — a liveness probe. The server only runs while the proxy is up, so a reachable `/healthz` is the connected signal. |
| `GET /info` | The running proxy's `version`, `providers` (each as `name=scheme://host[:port]`, in routing order — never credentials or query), `proxy_listen`, `admin_listen`, `max_retries`, and `database` path. |
| `GET /stats` | The full metrics rollup as JSON (see below). |
| `GET /requests` | The individual recorded requests, newest first, so a consumer can group them itself. `?limit=` (default 1000), clamped to `[1, 10000]`. |
| `GET /providers` | Providers, their discovered models, and exposure. |
| `POST /providers` | Add a provider. |
| `PATCH /providers/{name}` | Change exposure for one provider. |
| `DELETE /providers/{name}` | Remove a provider. |
| `GET /copilot/login` | Current device-flow status. Only present with `--copilot`. |
| `POST /copilot/login` | Start (or restart) the device flow; returns the `user_code` and `verification_uri`. Only present with `--copilot`. |
| `GET /` | Lists the available endpoints. |

`GET /stats` reads the guardrails database on each request — the same source the
`stats` subcommand reads — so the response is always current and the admin
server holds no in-memory counters that could drift from the proxy. Because the
database runs in WAL mode, these reads never block the proxy's writes.

```jsonc
{
  "per_model": [
    {
      "provider": "lmstudio",
      "model": "qwen2.5-7b",
      "total": 168,
      "tool_calls": 142,
      "succeeded": 137,
      "errors": 5,
      "success_rate": 0.965,        // null when the model made no tool call
      "by_outcome": [
        { "outcome": "native_valid", "count": 110 },
        { "outcome": "rescued", "count": 18 }
      ],
      // null when no request reported usage, so a consumer shows "not
      // measured" rather than a confident zero.
      "usage": {
        "prompt_tokens": 1201400,   // billed: NOT additive across a conversation
        "completion_tokens": 82900, // generated once, so this sums cleanly
        "billed_tokens": 1284300,
        "cached_tokens": 813200,
        "uncached_prompt_tokens": 388200,
        "cache_hit_rate": 0.677,
        "billed_calls": 183,        // backend calls, retries included
        "requests": 168,
        "calls_per_request": 1.089,
        // Resent prefixes counted once. null on Chat Completions, which
        // carries no conversation key — never the inflated sum renamed.
        "distinct_prompt_tokens": null,
        "distinct_tokens": null,
        "conversations": null,
        // true when the edges above were inferred from message prefixes
        // (--match-conversations) rather than supplied by the API.
        "inferred_conversations": false,
        // Spread across single requests: needs no conversation key, so it is
        // populated for all traffic. Percentiles are nearest-rank.
        "prompt_distribution": {
          "count": 168, "min": 412, "p50": 3180,
          "p90": 24100, "p99": 96400, "max": 131000
        },
        "completion_distribution": {
          "count": 168, "min": 3, "p50": 240,
          "p90": 1420, "p99": 3100, "max": 4096
        }
      }
    }
  ],
  "errors": [
    {
      "provider": "lmstudio",
      "model": "qwen2.5-7b",
      "error_category": "missing_argument",
      "tool_name": "Edit",
      "detail": "… | args: {\"oldString\":\"a\"}",   // argument values redacted
      "count": 3
    }
  ]
}
```

The `detail` snippet is redacted the same way as in the CLI report: argument
values are reduced to type/size tags, never stored or served verbatim, so the
endpoint is safe to surface in a UI.

`GET /requests` serves the rows behind that rollup, newest first, for the
groupings it does not do — a histogram with its own buckets, a breakdown by
hour, or a per-conversation total keyed on a session id only the client knows.
Only requests that reported usage appear, matching the population every token
figure in `/stats` is computed over. No message content is stored or served;
these are counts.

```jsonc
{
  "count": 2,      // rows returned; equals `limit` when more were available
  "limit": 1000,   // the limit actually applied, after clamping
  "requests": [
    {
      "ts": "2026-08-22T14:03:11Z",
      "provider": "lmstudio",
      "model": "qwen2.5-7b",
      "outcome": "native_valid",
      "prompt_tokens": 3180,
      "completion_tokens": 240,
      "cached_tokens": 2100,
      "billed_calls": 1,        // backend calls this request made, retries included
      "response_id": null,      // null on Chat Completions: no conversation key
      "parent_id": null         // the turn this one continues, when known
    }
  ]
}
```

On Responses traffic `response_id` and `parent_id` are populated, so the rows
can be walked into chains — which is exactly what `distinct_prompt_tokens` does
server-side. On Chat Completions they are `null`, and grouping is the consumer's
to do with whatever key it has.

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
guardrail/src/admin/        Read-only admin HTTP server (stats, health, info)
guardrail/src/connector/    Backend HTTP forwarding
guardrail/src/copilot/      GitHub Copilot provider and device-flow login
guardrail/src/domain/       Decode, rescue, validate, retry, respond, provider routing,
                            the Responses API translation, and conversation matching
guardrail/tests/            End-to-end proxy tests
```
