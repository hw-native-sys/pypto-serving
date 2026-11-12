# DeepSeek V4 DSpark Deployment

DSpark serves the converted DeepSeek V4 Flash W8A8 checkpoint with TP4 groups. It supports target-only greedy decoding (`K=0`) and a seven-token speculative chain (`K=7`). The 16-device configuration below uses `--dp 4 --ep 16 --tp 4`; an eight-device configuration uses `--dp 2 --ep 8 --tp 4`. Each configuration needs exactly as many device IDs as its `--ep` value.

## Prepare the Checkpoint

Use the same [checkpoint conversion](mtp.md#checkpoint-conversion) as the eight-device MTP path. Set `--block-size 32` for DSpark. The CLI validates the topology and caps `--max-num-seqs` at 64 requests per TP group (256 on 16 devices, 128 on eight). `--max-model-len` cannot exceed 1,048,576 or the checkpoint's `max_position_embeddings`. DSpark currently supports greedy sampling only, so use `temperature: 0`.

## Start the Server

```bash
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --served-model-name dsv4-flash-w8a8 \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --dp 4 --ep 16 --tp 4 \
  --block-size 32 --max-model-len 1024 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 128 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":0}' \
  --no-enable-prefix-caching \
  --ring-heap 2147483648,2147483648,4294967296,8589934592 \
  --port 8000
```

Set `num_speculative_tokens` to `7` to enable the DSpark speculative chain; other positive values are rejected. K=7 requires the checkpoint's three MTP layers and matching `dspark_target_layer_ids`, `dspark_noise_token_id`, and `dspark_markov_rank` metadata. `--enable-prefix-caching` is supported for both K=0 and K=7. Cached prefixes are aligned to 128 source tokens; K=7 keeps at least 128 trailing tokens for drafter reconstruction, so a prompt needs at least 256 tokens for its first possible hit. K=0 needs at least 129 tokens. See [Tool Calling](../../user-guide/tool-calling.md) for the client workflow and [Constrained Tool Deployment](tool-constraints.md) for the K=7 structural setup.

The checkpoint cadence is controlled by `--prefix-cache-retention-interval`
(default 8192), independently of the Prefill chunk. Positive intervals must
align to 128 source tokens; zero keeps replay/shared boundaries only. Prefill
balances the remaining cache budget between history and checkpoint windows
after reserving live/in-flight pages and one maximum-context history. Decode
retains its history-first budget policy. Checkpoints remain safely evictable.
Explicit `--ring-heap` values apply to all DSpark profiles and worker prewarm,
including speculative bootstrap; available capacity still requires validation
at the intended context and batch sizes.

## Verify

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-flash-w8a8","prompt":"Huawei is","max_tokens":32,"temperature":0}'
```

For the cache layout, kernel dispatch, and drafter state behind these options, see [DSpark Internals](dspark-internals.md).
