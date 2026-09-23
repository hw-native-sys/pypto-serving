# Online Serving

Online serving starts `pypto-serving`, loads the model in worker processes, and exposes an OpenAI-compatible HTTP API subset.

## Start a Qwen Server

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

The startup log prints the model name, platform, device groups, parallelism, request limits, scheduler token limit, and enabled endpoints. Wait for `Application startup complete` before sending traffic.

## Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | `GET` | Return server health. |
| `/v1/models` | `GET` | Return the served model name. |
| `/v1/completions` | `POST` | Generate text from a prompt. |
| `/v1/chat/completions` | `POST` | Apply the tokenizer chat template and generate a response. |

## Health and Models

```bash
curl --noproxy "*" http://127.0.0.1:8899/health
curl --noproxy "*" http://127.0.0.1:8899/v1/models
```

`/health` returns `{"status":"ok"}`. `/v1/models` returns the served model name, using `--served-model-name` when it is set.

## Completion Request

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":32,"temperature":0.0}'
```

Completions accept `model`, `prompt`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, and `stream`.

## Chat Request

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":32}'
```

Chat completions accept `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `stream`, `reasoning_effort`, `include_reasoning`, and `chat_template_kwargs`. DeepSeek V4 also supports the function-tool fields described below.

The server converts chat messages to a prompt with the tokenizer's `apply_chat_template` method. `chat_template_kwargs` is forwarded to the tokenizer, which allows model-specific controls such as Qwen thinking-mode settings when the tokenizer supports them.

## DeepSeek V4 Function Tools

Tool calling is selected by the model tokenizer; no extra launcher flag or vLLM dependency is required. Serving encodes tool definitions and parses model output. **The client executes tools**, then sends the results in a new chat request.

Send this request to a **DeepSeek V4** server, not the Qwen server in the examples above. Set `DEEPSEEK_BASE_URL` to that server's host and port (8000 is the default serving port).

```bash
DEEPSEEK_BASE_URL=http://127.0.0.1:8000  # Replace with your DeepSeek V4 endpoint.
curl --noproxy "*" "$DEEPSEEK_BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in London?"}],
    "tools": [{"type": "function", "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }}],
    "tool_choice": "auto",
    "max_tokens": 512
  }'
```

`auto` is the default when non-empty `tools` are supplied. The model can answer normally or return `message.tool_calls`, with each call containing `id`, `type: "function"`, and `function: {name, arguments}`. `arguments` is a **JSON string**, not a JSON object. `content` can be null; `reasoning`, when enabled, remains separate from both content and tools.

Without schema constraints, the parser returns a model-generated function name even if that name is absent from this request's `tools`, matching vLLM's default DeepSeek V4 behavior. Serving does not provide built-in functions such as `read_file`; the client decides which calls it can execute. Check the returned name against the client's available tools before executing it.

For a successful call, the client should validate the function name and arguments against its schema before execution. Append the returned assistant message and a tool result that references the same call ID:

```json
[
  {"role": "user", "content": "What is the weather in London?"},
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
      "id": "call_example",
      "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}
    }]
  },
  {"role": "tool", "tool_call_id": "call_example", "content": "Sunny, 18 degrees Celsius"}
]
```

Send that history to the same chat endpoint, including `tools` again if another tool call is allowed. Preserve the returned assistant `reasoning` when present and use consistent thinking settings across the round trip. Results of multiple calls are encoded in the original call order, even if the client returns them out of order.

Supported controls and limits:

- `tool_choice: "none"` suppresses tool-call output. Recognized tool blocks are consumed when tools are supplied; ordinary no-tools chat keeps its existing parser behavior.
- Multiple calls are supported. With constraints enabled, `parallel_tool_calls: false` limits generation to one call. Without constraints it exposes only the first parsed call. Serving never executes tools.
- `tool_choice: "required"` and named function choices constrain the call structure. `strict: true` also constrains parameter schemas, including in `auto` mode. Explicit `strict: false` leaves parameter schemas relaxed unless the server override below is enabled. Non-function tool types are rejected during request validation.
- Tools on a model without a registered tool parser are rejected. DSML formatting stays in the DeepSeek implementation, not the HTTP server or scheduler.
- Tool-history argument values cannot contain the reserved `</｜DSML｜parameter>` delimiter, including inside nested JSON values. Tool-result content cannot contain `</tool_result>`. These inputs return HTTP 400 before generation rather than breaking the history encoding.
- The parser preserves DSML parameter types without schema-based coercion or guessed JSON repairs. A length-truncated call can have incomplete arguments: do not execute it as a successful call.
- This feature applies to `/v1/chat/completions`; `/v1/completions` remains an unparsed text API.

### Server-side Tool Schema Constraints

Add `--enforce-tool-schema` to the server command to constrain all declared tools,
including clients that send `strict: false`. The client continues to send ordinary
OpenAI-compatible `tools`; no client-specific parameter aliases or system prompt
are needed. `tool_choice: auto` still permits a normal text answer and EOS.
Constraints do not require the model to call a tool for every task or guarantee
that a valid shell command performs the intended operation.

Install the updated serving runtime dependencies (`xgrammar>=0.2.7`). Strict
function parameters must be object schemas with declared `properties` and
`required` fields. Undeclared parameters are excluded. Nested values use
XGrammar's JSON Schema support; unsupported root composition keywords return
HTTP 400 before streaming starts. Plain string values retain DSML's raw-string
encoding; other values, including constrained strings, use its JSON encoding.

The DSpark runner applies packed token masks before NPU greedy sampling and
speculative acceptance. Draft previews roll back grammar state; only committed
tokens advance it. Constrained requests wait for their preceding decode result
before the next mask is prepared, reducing decode overlap. Other speculative
executors currently reject constrained requests. CPU-logit sampling supports
constraints without speculation.

This change also updates the DSpark prefill/decode kernel arguments. Update the
`pypto-lib` kernel sources together with serving. These kernels use new compile
cache slot names to avoid loading binaries with the previous argument layout;
the first launch recompiles them.

## Streaming

Set `stream: true` on a completion or chat completion request to receive Server-Sent Events:

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":32,"stream":true}'
```

Each event is emitted as `data: {...}`. The stream ends with:

```text
data: [DONE]
```

Accumulate `choices[0].text` for completions and `choices[0].delta.content` for chat completions. The final usage event has an empty `choices` list and authoritative token counts.

For tool-enabled chat, collect `delta.tool_calls` separately, keyed by `index`. The first delta for a call supplies its `id`, `type`, and `function.name`; concatenate subsequent `function.arguments` fragments for that index. Arguments can be incomplete JSON until the call finishes. String parameters stream before their closing delimiter; non-string parameters are emitted once their JSON value is complete. Accumulate `delta.reasoning` separately when present.

Invalid tool configuration is rejected before the stream starts. A model-output parsing error after SSE headers sends `data: {"error": {"message": "...", "type": "invalid_model_output", "code": 400}}`, followed by `[DONE]`, instead of a successful tool-call finish. The HTTP status is already 200 in that case; clients must inspect stream error events.

## Responses

Non-streaming responses include one choice and usage counts when the request finishes. Finish reasons are normalized to:

| Value | Meaning |
| --- | --- |
| `stop` | The model produced EOS or a stop string matched. |
| `length` | The request reached `max_tokens` or model length. |
| `aborted` | The request was aborted. |
| `error` | The engine reported a failure. |
| `tool_calls` | Tool-enabled chat ended normally with complete calls. |

`length`, `aborted`, and `error` are not overwritten by `tool_calls`. A normal end inside an incomplete tool block is a request-local parsing error, not a successful call.

Scheduler and engine rejections are returned as HTTP 400 with:

```json
{"object":"error","message":"..."}
```

## Shutdown

Stop the server with the normal process signal for your environment. On a graceful shutdown, the server attempts to stop active profile recorders and merge available profile fragments when profiling is enabled.
