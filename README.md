# guardrail

A transparent **OpenAI chat-completions proxy** that sits in front of an
OpenAI-compatible backend (LM Studio first) and applies small-model tool-call
reliability guardrails in the wire path. Ships as a Jarvis sidecar: a single
static binary the menu bar app launches and supervises alongside the jarvis MCP
proxy.

## Where it fits

Jarvis and guardrail operate at two different edges of the same agent and do not
call each other at runtime:

```
agent harness ──(MCP)──────────► jarvis ──(MCP)──► backend MCP servers
agent harness ──(OpenAI HTTP)──► guardrail ──(HTTP)──► LM Studio
```

Jarvis shrinks the tool surface the model sees to 3 synthetic tools; guardrail
makes the calls to those tools reliable on small models. Complementary, not
coupled.

## Status

Requests without tools (and any streamed request) are forwarded **verbatim**,
byte-for-byte, including the response stream — the `model` field is never
rewritten. Tool-enabled, non-streamed requests run the **active guardrail loop**:
inject `respond`, call the backend, decode → rescue → validate, retry with a
nudge on failure, unwrap `respond` to text. Native valid calls pass through
unchanged; only repaired responses are rewritten. Every guardrail is on by
default and individually toggleable (`--rescue/--respond/--retry false`), so the
proxy degrades to a zero-overhead passthrough.

Milestones:

1. ✅ Transparent passthrough (failure-isolation).
2. ✅ Typed+passthrough serde model — round-trip fidelity.
3. ✅ Validation + canonical re-emit.
4. ✅ Rescue parsing — Mistral, rehearsal (`name[ARGS]{}`), Qwen-Coder
   (`<function=…><parameter=…>`), Qwen `<tool_call>`, Hermes, Llama, fenced JSON,
   and `{tool,args}`/`{name,arguments}` JSON (forge's full set + native formats).
5. ✅ Synthetic `respond` tool + strip-to-text.
6. ✅ Retry loop + ErrorTracker with fallback-to-last-text.
7. ✅ Per-guardrail config toggles (observability tracing throughout).
8. _(deferred)_ optimistic streaming; more backends; Anthropic wire format.

## Run

```bash
# from the repo root (or drop --manifest-path when run inside rust/)
cargo run --manifest-path rust/Cargo.toml -p guardrail -- \
  --listen 127.0.0.1:8080 --backend http://127.0.0.1:1234
```

Config is also available via env: `GUARDRAIL_LISTEN`, `GUARDRAIL_BACKEND`,
`GUARDRAIL_CONNECT_TIMEOUT_SECS`, `GUARDRAIL_READ_TIMEOUT_SECS`, and the guardrail
toggles `GUARDRAIL_RESCUE`, `GUARDRAIL_RESPOND`, `GUARDRAIL_RETRY`,
`GUARDRAIL_MAX_RETRIES`. Logging via `RUST_LOG` (default `guardrail=info,warn`).

Point your OpenAI-compatible client's base URL at `http://127.0.0.1:8080/v1`.

## Test

```bash
cargo test          # from the rust/ workspace root
```

`tests/passthrough.rs` stands up a wiremock backend and asserts status, headers,
request body, and a streamed SSE body all round-trip unchanged — the Milestone 1
acceptance suite.

## Build a release binary

```bash
bash scripts/build_guardrail_binary.sh        # macOS → app Resources/
bash scripts/build_guardrail_binary_linux.sh  # Linux → dist/
```
