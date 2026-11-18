# DeepSeek V4 DSpark Deployment

DSpark serves the converted DeepSeek V4 Flash W8A8 checkpoint with TP4 groups. It supports target-only greedy decoding (`K=0`) and speculative decoding with `K=5` or `K=7`. The 16-device configuration below uses `--dp 4 --ep 16 --tp 4`; an eight-device configuration uses `--dp 2 --ep 8 --tp 4`. Each configuration needs exactly as many device IDs as its `--ep` value.

## Prepare the Checkpoint

Use the same [checkpoint conversion](mtp.md#checkpoint-conversion) as the eight-device MTP path. Set `--block-size 32` for DSpark. The CLI validates the topology and caps `--max-num-seqs` at 16 times the device count (256 on 16 devices, 128 on eight). `--max-model-len` cannot exceed 1,048,576 or the checkpoint's `max_position_embeddings`. DSpark currently supports greedy sampling only, so use `temperature: 0`.

Serving configures Lib's Decode group batch before importing kernels: B=16 at
TP1 and B=64 at TP4. The verification width is S=K+1: six positions for K5,
eight for K7. Decode and LM-head use 16*S rows per owner; bootstrap retains
the full 128-token history independently. Sampling keeps its physical
eight-column stride. Independent LM-head TP shards the vocabulary and does
not divide the owner capacity again.
Target-only execution uses the eight-position target ABI.

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

Set `num_speculative_tokens` to `5` or `7` to enable the DSpark speculative chain. K is fixed when the instance starts. Both use the checkpoint's three MTP layers and matching `dspark_target_layer_ids`, `dspark_noise_token_id`, and `dspark_markov_rank` metadata. `--enable-prefix-caching` supports target-only and speculative decoding. Cached prefixes are aligned to 128 source tokens; speculative decoding keeps at least 128 trailing tokens for drafter reconstruction, so a prompt needs at least 256 tokens for its first possible hit. K=0 needs at least 129 tokens. See [Tool Calling](../../user-guide/tool-calling.md) for the client workflow and [Constrained Tool Deployment](tool-constraints.md) for the structural setup.

The checkpoint cadence is controlled by `--prefix-cache-retention-interval`
(default 8192), independently of the Prefill chunk. Positive intervals must
align to 128 source tokens; zero keeps replay/shared boundaries only. Prefill
balances the remaining cache budget between history and checkpoint windows
after reserving live/in-flight pages and one maximum-context history. Decode
retains its history-first budget policy. Checkpoints remain safely evictable.
Explicit `--ring-heap` values apply to all DSpark profiles and worker prewarm,
including speculative bootstrap; available capacity still requires validation
at the intended context and batch sizes.

## Expert Load Balancing

Add these options to a DSpark server using TP4/EP8 or TP4/EP16:

```bash
--enable-eplb \
--eplb-config '{"num_redundant_experts":16,"max_migration_bytes":17179869184}'
```

EPLB preserves logical expert routing and places expert replicas in fixed physical
slots. The logical expert count plus `num_redundant_experts` must divide evenly
across EP ranks. Redundant slots consume additional device memory. Set
`max_migration_bytes` to an explicit positive per-rebalance transfer budget;
the example permits up to 16 GiB and is not a recommended memory sizing value.

By default, EPLB collects decode loads over a 128-step window and considers
rebalancing every 1024 collected steps, with at most one layer per rebalance
and a minimum relative gain of 0.05. Optional JSON fields include `window_size`,
`step_interval`, `load_collection_phase` (`prefill` or `decode`), `policy`
(`temporal_greedy` or `cumulative_greedy`), `max_layers_per_rebalance`, and
`min_relative_gain`. Drafter balancing is unsupported. PD roles independently select
prefill or decode loads; D also supports TP1 with independent head TP4; see [PD EPLB](../../user-guide/pd-disaggregation.md#expert-load-balancing).

Maintenance waits at most `maintenance_timeout_seconds` (default 1 second) for
compute outputs and conflicting DMA to settle before postponing a candidate.
`cooldown_seconds` defaults to 30; `max_sample_age_seconds` defaults to 60.
A measured pause-cost estimate is compared with predicted savings over
`amortization_steps` (default 1024), with `maintenance_cost_margin` (default 1.25).
The estimate uses dispatch wall time and load-balance gain as proxies, not a
guarantee of kernel or end-to-end acceleration. Set `max_rebalances` to a positive
count to learn that many placements and then freeze; zero (default) is unlimited.

Serving and pypto-lib must both include the matching EPLB ABI. Maintenance drains
in-flight outputs before migrating expert weights and scales, then publishes the
new placement. A failed installation stops subsequent compute on that executor.

## Verify

```bash
curl --noproxy '*' http://127.0.0.1:8000/health
curl --noproxy '*' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-flash-w8a8","prompt":"Huawei is","max_tokens":32,"temperature":0}'
```

For the cache layout, kernel dispatch, and drafter state behind these options, see [DSpark Internals](dspark-internals.md).
