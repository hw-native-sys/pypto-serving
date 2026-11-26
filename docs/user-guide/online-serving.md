# Online Serving

Start `pypto-serving` without `--prompt` to load the model in workers and expose its HTTP API. The server supports completions and chat completions for the [supported models](../models/index.md).

For opt-in multi-node prefill/decode deployment, see [PD disaggregation](pd-disaggregation.md).

## Start a Server

This Qwen3-14B example serves one NPU. For other checkpoints and device topologies, follow the model deployment guide.

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

Wait for `Application startup complete`. The startup log reports the served model, platform, device groups, parallelism, request limits, and endpoints. Check health from another shell:

```bash
curl --noproxy '*' http://127.0.0.1:8899/health
```

## Send Requests

Generate a completion from a text prompt:

```bash
curl --noproxy '*' http://127.0.0.1:8899/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Huawei is","max_tokens":32,"temperature":0.0}'
```

Generate a chat response with the model's chat formatter:

```bash
curl --noproxy '*' http://127.0.0.1:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":32}'
```

`chat_template_kwargs` is passed to the model formatter, so supported settings such as Qwen thinking mode can be selected per request. The [HTTP API Reference](../api-reference/http.md#chat-fields) covers token limits and EOS controls; [Tool Calling](tool-calling.md) describes function calls for models with a registered parser.

## AgentInfer Admission

Add `--agentinfer-mode enforce` to enable adaptive agent admission in PyPTO
Serving. The policy uses native KV cache observations to decide when to retain
a session and when to resume queued work.
It supports one serving replica with prefix caching: Qwen, standard DeepSeek V4
MTP K=1, or DeepSeek V4 DSpark K=0/K=7. PD deployments, multiple replicas, and
DeepSeek V4.1 are not supported. The default mode is `disabled`.

Optional policy bounds are provided with `--agentinfer-config`, for example:

```bash
pypto-serving --model /path/to/Qwen3-14B --agentinfer-mode enforce \
  --agentinfer-config '{"resume_capacity_ratio":0.85,"force_resume_timeout_min_seconds":15,"force_resume_timeout_max_seconds":15}'
```

The force-resume bounds must satisfy `0 <= min <= max <= 300` seconds. A request
still held by AgentInfer after 300 seconds moves to the native scheduler, where
it may wait for capacity before generation starts.

Requests with a stable agent identity share a Program across model calls.
Requests without an identity use native admission. Completions and Chat can use
session headers; Chat also accepts `agent_hint` or canonical
`vllm_xargs.agentic_context`:

| Request metadata | Program identity |
| --- | --- |
| `X-Claude-Code-Session-Id`, optionally `X-Claude-Code-Agent-Id` | Session and lead or subagent ID. |
| `X-OpenCode-Session-Id` or `X-OpenCode-Session` | Session and lead ID. |
| `X-DeepSeek-Harness-Session-Id` | Session and lead ID; auxiliary compaction requests are excluded. |
| `Session-Id`, optionally `Thread-Id` and `X-Codex-Parent-Thread-Id` | Session and lead or thread ID, with parent lineage. |
| `X-Session-Id` | Session and lead ID. |
| `vllm_xargs.agentic_context` | Canonical Program identity with optional task and parent fields. |
| `agent_hint.session_id`, optionally `parent_session_id` | Explicit Program identity and parent lineage. |

Canonical context takes precedence over `agent_hint`, which takes precedence
over headers. For example, `X-Session-Id: run-123` identifies Program
`run-123:lead`, while `agent_hint.session_id: run-123` uses Program `run-123`.
These mappings apply to the supported OpenAI-compatible endpoints.

Text-action clients explicitly confirm task completion after verifying it.
Send the Program ID and the Chat response `id` to `POST /v1/agent/lifecycle`:

```json
{"session_id":"run-123","request_id":"<chat response id>","lifecycle":"terminal"}
```

The endpoint returns `{"released":true}` when it releases the matching Program.
Use `lifecycle: "continue"` to report another round. Delayed completion signals
cannot release a newer round in the same Program. Chat clients can opt into
automatic stop/tool-call lifecycle handling with
`agent_hint.lifecycle_mode: "chat"`; a length-limited response does not confirm
task completion. The default lifecycle mode is `explicit`.

## Stream Output

Set `stream: true` on either generation endpoint to receive Server-Sent Events:

```bash
curl --noproxy '*' http://127.0.0.1:8899/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Huawei is","max_tokens":32,"stream":true}'
```

Append text deltas as they arrive. The final usage event has an empty `choices` list and authoritative token counts; `data: [DONE]` closes the stream. For all request fields, response shapes, finish reasons, and streaming errors, see the [HTTP API Reference](../api-reference/http.md).

## Observe and Stop

`/metrics` exposes Prometheus text and `/metrics/json` returns engine metrics as JSON. [Profiling](../developer-guide/profile.md#profile-http-serving) describes how to capture internal traces when the server starts with `--profile`. Stop the server with the normal process signal for your environment; graceful shutdown attempts to flush and merge active profile fragments.
