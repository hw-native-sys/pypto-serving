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

## Interactive Coding Frontend

The optional [mini-swe-agent HTTP frontend](../../examples/mini_swe_agent/README.md)
connects to the Chat API for coding tasks and file-based conversation. It
supports command confirmation, context budgeting, execution records, and optional
completion checks. Install its dependencies in a separate client environment;
the frontend does not start or stop the server.
