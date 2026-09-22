# PD Integration Checks

These checks use an already running Router, Prefill node and Decode node.
They do not install dependencies, launch processes, connect through SSH, or
change device state. Deployment instructions are in
[Disaggregated Serving](../../../docs/user-guide/disaggregated-serving.md).

## Chat Streaming

```bash
python tests/integration/pd/check_chat_stream.py \
  --router-url http://router.example.internal:8110 \
  --prefill-url http://prefill.example.internal:8111 \
  --decode-url http://decode.example.internal:8112 \
  --model dsv4-flash-dspark-w8a8 \
  --max-tokens 128 \
  --output /path/to/new/chat-result.json
```

The check validates SSE termination, request identity, a terminal usage
record, assistant text, the Router's token digest, and released P/D
reservations. Run it on an otherwise idle deployment because its final
capacity checks cover the whole node. The output path must be new.

## Decode Prefix Reuse

Use `runtime.prefix_cache_mode="d_only"` and send the same prompt of at least
128 tokens twice. Capture P and D `/internal/pd/metrics` before either
request, after the cold request, and after the repeated request. Store each
pair as one JSON object with `prefill` and `decode` keys holding the respective
endpoint responses.

```bash
python tests/integration/pd/check_prefix_reuse.py \
  --before /path/to/before.json \
  --after-cold /path/to/after-cold.json \
  --after-hit /path/to/after-hit.json
```

The DSpark check requires one cold reservation, one hit reservation, at least
128 hit tokens, and fewer transferred bytes for the repeated request.
Transferred bytes must remain nonzero: final request-local state is required
even on a full KV hit.

For `independent` mode, also inspect `prefix.p_hit_tokens` and exercise
different P/D hit lengths. The metrics check alone does not establish output
correctness; compare generated token IDs with the same non-PD K7 request,
and separately cover cancellation and uncertain transfer cleanup.
