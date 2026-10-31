# Tool Calling

The chat API accepts function definitions in `tools` and returns calls in `message.tool_calls`. The client validates and executes a returned call, then sends the result in a new chat request. Serving does not execute tools or provide built-in functions.

Tool support depends on the loaded model's output parser. [DeepSeek V4](../models/deepseek-v4/index.md) and [Qwen3-14B](../models/qwen3-14b.md) support function tools. The [model matrix](../models/index.md) lists supported paths, and the [HTTP API](../api-reference/http.md#chat-fields) defines the request fields.

## Request a Function Call

Send this request to a server running a model with tool support, such as DeepSeek V4 or Qwen3-14B:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
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

`auto` is the default when non-empty `tools` are supplied. The model may answer directly or return `message.tool_calls`. Each call has an `id`, `type: "function"`, and `function: {name, arguments}`. `arguments` is a JSON string, not an object. Assistant `content` can be null when a call is returned; parsed `reasoning` remains separate.

Qwen3 uses its checkpoint chat template for tool definitions and emits `<tool_call>` JSON blocks. Serving returns only complete calls as structured `tool_calls`; tool-free Qwen3 chat retains its normal text output. Qwen3 tool selection is unconstrained: `required`, a named choice, and `strict: true` are rejected before generation. The full prompt, including tool definitions, must fit the configured model context limit.

Validate the function name and decoded arguments against the client's allowed tools before execution. In unconstrained mode, a model may produce a name that was not declared in the request or arguments that do not match its schema. A length-truncated call may have incomplete arguments and must not be executed.

## Return the Tool Result

Append the returned assistant message and a tool result with the same call ID, then send the history to `/v1/chat/completions`:

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

Include `tools` again if another call is allowed. Preserve assistant `reasoning` when present and keep thinking settings consistent across the round trip. The server validates that tool results reference preceding assistant call IDs.

## Choice and Streaming

`tool_choice: "none"` suppresses tool-call output. `auto` allows a direct answer or a call. `required` and a named function choice request structural constraints, whose availability depends on the model; see [DSpark Tool Constraints](../models/deepseek-v4/tool-constraints.md) for the currently supported constrained path. `parallel_tool_calls: false` exposes only the first parsed call. These controls never cause the server to execute a function.

For `stream: true`, collect `choices[0].delta.tool_calls` by `index`. The first delta supplies the call ID, type, and function name; concatenate later `function.arguments` fragments for that index. The fragments may be incomplete JSON until the call finishes. Accumulate `delta.reasoning` separately when present.

Invalid tool configuration is rejected before streaming starts. If output parsing fails after SSE headers, the stream sends an `invalid_model_output` error event followed by `[DONE]`; the HTTP status is already 200. See [HTTP API errors](../api-reference/http.md#errors). Tool parsing applies to `/v1/chat/completions`; `/v1/completions` remains unparsed text.
