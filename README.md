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
| `--rescue` | `GUARDRAIL_RESCUE` | `true` | Recover tool calls embedded in text. |
| `--respond` | `GUARDRAIL_RESPOND` | `true` | Inject and unwrap the synthetic `respond` tool. |
| `--retry` | `GUARDRAIL_RETRY` | `true` | Retry invalid tool calls with a corrective nudge. |
| `--max-retries` | `GUARDRAIL_MAX_RETRIES` | `2` | Maximum corrective retries per request. |

Example with all guardrails disabled:

```bash
cargo run -p guardrail -- \
  --rescue false \
  --respond false \
  --retry false
```

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
