# DeepSeek V4 MTP Internals

This page covers the eight-rank MTP runner. For the command line and checkpoint requirements, see the [MTP deployment guide](mtp.md); DSpark has a separate [implementation page](dspark-internals.md).

## Implementation Entry Points

| Layer | Interface and responsibility |
| --- | --- |
| Model integration | `pypto_serving/model/integrations/deepseek.py`: `DeepSeekV4Integration.resolve()` enforces the eight-rank DP8/EP8/TP1 contract and derives cache groups and kernel prefill limits. |
| Executor | `pypto_serving/model/deepseek/npu_executor.py`: `DeepSeekV4PyptoExecutor._compile_model()` loads and compiles the PyPTO callables; `_create_runner()` supplies the model runner. |
| Cache contract | `pypto_serving/model/deepseek/npu_runner.py`: `build_deepseek_v4_cache_group_specs()` describes the seven main-model groups; `DeepSeekV4ModelRunner.init_kv_cache()` allocates their device pools after weights are resident. |
| Worker | `pypto_serving/serving/server/serving_worker.py`: `_make_prefill_batch()` and `_make_decode_batch()` carry scheduler-owned `block_ids_by_group` into the runner; `_batch_prefill()` calls `finalize_prefill()` after sampling the first output token. |
| Runner | `DeepSeekV4ModelRunner.run_prefill()` packs and dispatches a chunk; `run_decode()` selects AR or MTP; `release_finished_requests()` frees per-request runner state. |

`ParallelConfig` selects overlapped placement in the model integration, so attention DP and MoE EP reuse the same physical ranks. `ReplicaEngineCore` still owns one scheduler, cache manager, and worker for this model group; DP8 does not create eight independent HTTP serving replicas.

## Dynamic Prefill Extent

The main prefill kernel supports a dynamic request extent up to 8192 tokens and walks it internally in 128-token tiles. AR and MTP use the same main-prefill request limit; the effective dispatch extent is the minimum of 8192, `--max-num-batched-tokens`, `--long-prefill-token-threshold`, and `--max-model-len`.

AR and MTP submit each main-prefill chunk once, with its backing extent padded to the next 128-token tile. An 8191-token prompt therefore uses one 8192-row main-prefill dispatch when those configured limits permit it, rather than 64 separate serving dispatches. The 128-token width is an internal kernel tile, not a serving chunk restriction.

The main kernel returns each owner's final 128 valid pre-HC rows, which the standalone fixed-width MTP prefill kernel uses to rebuild its 127-row KV window before retaining the final row for sampling.

## Cache Layout

DeepSeek V4 uses grouped cache pools instead of the generic KV cache tensor layout. The seven main-model KV/state pools are allocated during runner preflight as rank-sharded worker-resident tensors.

`DeepSeekV4Integration.resolve()` builds the `KVCacheGroupSpec` entries, and `resolve_launch_options()` places them in `RuntimeConfig.kv_cache_groups`. `ReplicaEngineCore` validates them through `SchedulerConfig.validate_cache_groups()` and initializes `KvCacheManager` without a fixed page count. On worker startup, `DeepSeekV4ModelRunner.init_kv_cache()` materializes weights and persistent buffers, computes a capacity from remaining NPU memory, then returns the `ori` block count. The engine scales the other groups from the same admitted slot count. Changing a group's block size or partitioning therefore requires coordinated changes in the model integration, scheduler metadata, and kernel argument binding.

Prefill and decode pass the same device handles and address them with scheduler-owned group block IDs. `DeepSeekV4CacheMetadataBuilder` converts those IDs and token positions into the block tables and slot mappings consumed by the PyPTO callables. There is no prefill CPU snapshot or cache handoff. Reassigned pages are cleared with targeted host-to-device copies before their new owner writes them.

## Prepacked Sidecar Contract

On startup, the DeepSeek V4 loader samples the optional sidecar's Linux page-cache residency before opening it, then validates a hot sidecar against the checkpoint-file and deployment fingerprint. A hot, valid sidecar is memory mapped as the final layout instead of repacking every hidden layer. A cold, missing, or stale sidecar falls back to the original checkpoint path, avoiding a cold 323 GiB page-fault stream on the weight-upload path.

The sidecar layout follows the order produced by `DEEPSEEK_V4_LAYER_RULES`; its metadata records a name-to-offset map built from that order. Reordering the rule table invalidates existing sidecars. The fingerprint covers the config, weight map, and every source shard's size and modification time, so startup can detect a stale sidecar. Rebuild with `pypto-prepack-deepseek-v4 --force` after replacing checkpoint shards or changing the packed rank layout.

## MTP State

MTP prefill context, draft token, recurrent hidden state, and acceptance counters are owned by request ID. MTP prefill and decode share one worker-resident cache, but each request addresses it with scheduler-owned rank-local `ori` block IDs.

The scheduler reserves all speculative positions before dispatch, including when a draft sequence crosses a 128-token page boundary.

The worker calls `DeepSeekV4ModelRunner.finalize_prefill()` only after a terminal prefill chunk has produced its first sampled token. The runner uses that token to initialize the draft and persistent device state, so the first decode step can consume already-initialized state. `run_decode()` dispatches to `_run_autoregressive_decode()` or `_run_mtp_decode()` according to the compiled variant; the latter can prepare, dispatch, and reclaim asynchronously through `prepare_decode()`, `dispatch_prepared_decode()`, and `reclaim_prepared_decode()`.

Before the first decode is prepared, each request owns a stable rank-local device-state slot and reuse generation. Terminal prefill fills that reserved slot with the committed tail token, next draft token, tail position, and committed count.

The fused decode kernel uses `(rank, slot, generation)` to build the next `[tail, draft]` input rows and sequence metadata before main decode, then updates the same slot after MTP verification. Host output processing mirrors the state for scheduling and statistics, but is not an input dependency of the next steady-state decode.

Generation matching prevents a stale queued step from updating a slot after preemption and reuse.
