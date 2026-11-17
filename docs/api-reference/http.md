# HTTP API

`pypto-serving` exposes the endpoints below in server mode. The API implements a subset of OpenAI-style completions and chat completions; use [Online Serving](../user-guide/online-serving.md) for a first request and [Models](../models/index.md) for model-specific support.

## Endpoints

| Method | Path | Result |
| --- | --- | --- |
| `GET` | `/health` | `{"status":"ok"}` |
| `GET` | `/v1/models` | A list with the served model ID, using `--served-model-name` when set. |
| `POST` | `/v1/completions` | Text generation from `prompt`. |
| `POST` | `/v1/chat/completions` | Chat generation after applying the model's message encoding. |
| `GET` | `/metrics` | Prometheus text metrics when the engine exposes metrics. |
| `GET` | `/metrics/json` | The same engine's metrics snapshot as JSON, when available. |

`POST /start_profile` and `POST /stop_profile` are available only when the server starts with `--profile`. Their behavior is documented in [Profiling](../developer-guide/profile.md#profile-http-serving).

## Generation Fields

Both generation endpoints accept these JSON fields. Omitted sampling fields inherit the server's `--generate-config` defaults; explicit request values override them. The installed [CLI Reference](../cli-reference/pypto-serving.md#generation-controls) describes that configuration.

| Field | Type | Behavior |
| --- | --- | --- |
| `model` | string | Optional response model name; when omitted, the served model ID is used. |
| `max_tokens` | integer or null | Maximum new tokens. |
| `temperature` | number or null | Sampling temperature; model-specific restrictions apply. |
| `top_p` | number or null | Nucleus sampling cutoff. |
| `top_k` | integer or null | Top-k sampling cutoff. |
| `seed` | integer or null | Sampling seed. |
| `stop` | array of strings or null | Stop strings. `[]` clears server defaults. |
| `stream` | boolean | Return Server-Sent Events when `true`; defaults to `false`. |

`model` is used as the response label; it does not select another loaded checkpoint. `/v1/completions` also takes `prompt` (string; defaults to an empty string). It always sets `ignore_eos` internally. `/v1/chat/completions` requires `messages` and uses the model's chat EOS behavior unless overridden below.

## Chat Fields

Each `messages` item has a `role` and optional `content`, `reasoning`, `tool_calls`, and `tool_call_id`. Text `content` is a string or null; multimodal message parts are not in this request schema.

| Field | Type | Behavior |
| --- | --- | --- |
| `messages` | array | Required chat history. The model must support its roles and formatting. |
| `max_completion_tokens` | integer or null | Maximum new tokens. A non-null value takes precedence over `max_tokens`; otherwise the non-null `max_tokens` or server default applies. |
| `min_tokens` | non-negative integer or null | Minimum generated tokens before EOS can be sampled. Must not exceed the selected maximum; see model-specific limits below. |
| `ignore_eos` | boolean or null | Override the server's EOS setting for this chat request. |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max` | Model-specific thinking control. |
| `include_reasoning` | boolean | Include parsed reasoning in the response; defaults to `true`. |
| `chat_template_kwargs` | object or null | Passed to the model tokenizer/chat formatter, for example Qwen's `enable_thinking`. |
| `tools` | array or null | Function definitions for models with a tool parser. |
| `tool_choice` | string, object, or null | Select tool behavior where supported. |
| `parallel_tool_calls` | boolean | Allow multiple returned calls; defaults to `true`. |

Both token-limit fields must be positive when set. `min_tokens` also masks EOS when `ignore_eos` is true; implementations can differ in how they enforce that mask. See the [DeepSeek V4 decoding limits](../models/deepseek-v4/index.md#token-limit-behavior) for its execution paths.

Tool definitions use `{"type":"function","function":{"name":"...","parameters":{...}}}`; `description` and `strict` are optional function fields. Tool-call history uses `tool_calls` on an assistant message and a matching `tool_call_id` on each tool result. See [Tool Calling](../user-guide/tool-calling.md) for the client workflow and [DSpark Tool Constraints](../models/deepseek-v4/tool-constraints.md) for the currently supported structural generation path. Models without a registered tool parser reject tool requests.

## Responses and Streaming

Non-streaming generation returns `id`, `object`, `created`, `model`, one entry in `choices`, and `usage` after completion. A text choice has `text`; a chat choice has an assistant `message`. Usage contains `prompt_tokens`, `completion_tokens`, and `total_tokens`. Finish reasons are `stop`, `length`, `aborted`, `error`, or `tool_calls` when a complete tool call ends normally. A length-truncated tool call is not a successful invocation.

For example, a completed text request has this shape:

```json
{
  "id": "cmpl-example",
  "object": "text_completion",
  "created": 0,
  "model": "served-model-name",
  "choices": [{"index": 0, "text": "...", "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
}
```

`created` is a Unix timestamp; the values above only illustrate the response fields. Chat responses use `object: "chat.completion"` and `choices[0].message` instead of `text`.

With `stream: true`, each event is `data: {...}` followed by a blank line. Text completions append `choices[0].text`; chat completions append `choices[0].delta.content` and, when present, `delta.reasoning`. Each stream ends with a usage event whose `choices` is empty, then `data: [DONE]`. Tool-call deltas and their assembly rules are in [Tool Calling](../user-guide/tool-calling.md#choice-and-streaming).

## Errors

Request validation failures, unsupported model options, and scheduler/engine rejections can return HTTP 400. Scheduler/engine rejections use `{"object":"error","message":"..."}`; schema validation errors use the framework's validation response. A chat tool-output parsing failure after streaming headers sends `data: {"error":{"message":"...","type":"invalid_model_output","code":400}}` followed by `[DONE]`. The HTTP status has already been sent as 200 in that case, so streaming clients must inspect events for errors.
