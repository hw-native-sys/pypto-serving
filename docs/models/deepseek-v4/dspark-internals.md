# DeepSeek V4 DSpark Internals

This page describes the DSpark runner and kernel contracts. For checkpoint preparation, supported flags, and an HTTP smoke test, see [DSpark Deployment](dspark.md).

## Implementation Entry Points

| Layer | Interface and responsibility |
| --- | --- |
| CLI | `cli/main.py` selects `PyptoDeepSeekV4DSparkExecutor` in `_executor_cls_for_model_family`, validates TP4/EP topology in `_validate_dspark_topology`, and builds seven cache groups in `_build_runtime_config`. |
| Executor | `model/deepseek_dspark/npu_executor.py`: `DeepSeekV4DSparkPyptoExecutor._compile_model()` imports and compiles the DSpark prefill/decode programs; `_create_runner()` constructs the runner. |
| Runner | `model/deepseek_dspark/npu_runner.py`: `DSparkModelRunner.init_kv_cache()` allocates post-weight pools; `prepare_prefill_inputs()` and `prepare_decode_inputs()` lower scheduler batches into the frozen kernel ABI. |
| Worker | `serving/server/serving_worker.py`: `_make_prefill_batch()` and `_make_decode_batch()` pass group block IDs and cache partitions; `_batch_prefill()` calls `finalize_prefill()` for terminal chunks. |
| Weights | `model/deepseek_dspark/weight_loader.py`: `DSparkWeightStore` packs target slabs and, for K=7, the three-layer drafter bank. |

The CLI accepts two or four TP4 groups (`--ep 8` or `16`, with `--dp 2` or `4`). The 16-rank deployment discussed below has four scheduler cache partitions. Physical block IDs are local to each partition and replicated across its four TP ranks.

## Prefix caching

The HCA cache layout changes the prefill/decode kernel ABI. On the first
launch after upgrading, set `PYPTO_CACHE=0` (or unset it) to regenerate both
graphs.

The seven target cache families use the shared grouped prefix-cache manager.
Lookup selects the TP partition with the longest reusable prefix, sharing
full compressed-history pages and the immutable rolling tail. The allocator
detaches rolling write destinations before continuing a shared prefix.
`build_dspark_cache_group_specs()` defines these groups; `DSparkModelRunner.init_kv_cache()`
allocates their replicated device pools after weights are materialized. The scheduler
hands each batch `block_ids_by_group` and `cache_partitions`, which
`DSparkCacheMetadataBuilder` lowers into the tables and slot mappings used by the kernels.
Rolling pool capacities include both the configured prefill chunk and the
historical window, because prefill writes its whole chunk before attention
reads the prefix. The pool budget follows the active chunk limit; the CSA
lookup requires only its eight-token historical tail. The fused decode state
descriptors use the same physical ring lengths as the scheduler, so device-side
modulo stays aligned after wraparound, including after a prefix-cache hit.

Once a step completes, the scheduler releases raw-KV and compressor-state
pages older than their confirmed sliding windows. Expired entries become `-1`
without compacting the logical table or changing its ring period. Allocation
fills only the retained window and new write positions; full-history compressed
pools are unchanged. For a 1024-token prompt, raw KV ownership drops from 32
pages to four after prefill, plus pages needed for subsequent writes. Queued
async snapshots keep their own references until completion, and speculative
reservations never advance the reclamation boundary. Released prefix pages
remain reusable through their hashes until the allocator evicts them.

The drafter's private caches are rebuilt rather than shared. K=7 sets
`speculative_prefix_cache_replay_tokens=128` in the runtime contract; the
scheduler caps the hit before rounding down to the grouped page alignment.
Suffix prefill may span several chunks. Its last 128 hidden rows and absolute
positions seed a fresh drafter lease through the normal prefill-finalization
path. The runner rejects an incomplete tail instead of reading stale private
KV from a previous lease. A prefix leaving fewer than 128 suffix tokens falls
back to an earlier aligned hit or a cold prefill.

The checkpoint's `max_position_embeddings` must cover the requested Serving limit.

## How the deployment maps onto the kernels

`DeepSeekV4DSparkPyptoExecutor._compile_model()` loads the PyPTO programs from
`pypto-lib/models/deepseek_v4_flash_dspark`. `DSparkModelRunner.prepare_prefill_inputs()`
and `prepare_decode_inputs()` translate `PrefillBatch` and `DecodeBatch` into
kernel metadata; `run_prefill()` and `run_decode()` dispatch them. The worker
calls `finalize_prefill()` for terminal chunks so a K=7 request has drafter
state before decode. These interfaces are the handoff points to inspect when
changing the kernel ABI.

- **Cache partitions = TP groups.** The scheduler sees 4 partitions. Every
  rank of a group holds an identical replicated pool: prefill writes the same
  rows on all four ranks, and decode rebuilds the group's whole KV stream
  from the gathered token rows each step (pypto-lib#1079).
- **Long-context capacity is logical, not fully resident.** Ratio-128 history
  grows to 8192 single-row pages per request at 1M; ratio-4 and index
  history also grow to 8192 pages. Raw KV and HCA/CSA state remain rolling
  physical pools. Startup memory admission still needs room for one complete
  logical-capacity request plus scratch pages.
- **Block tables are staged at the kernels' frozen depths** (CSA compressed
  and indexer tables 8192 entries, HCA compress-state table 131072, CSA
  compress-state tables 524288, -1 past
  each request's span). The generated orchestration reshapes these tables
  with the frozen depths baked in, so a shallower table asserts on device and
  surfaces as an opaque AICore 507901; the depths therefore track the kernel
  constants (`IDX_MAX_BLOCKS`, `CMP_MAX_BLOCKS`,
  `COMPRESS_STATE_MAX_BLOCKS`), never the serving context ceiling.
- **Prefill** can pack multiple requests per TP group, within 64 requests and
  8192 total chunk tokens per group (256 requests per dispatch across four
  groups). These
  are admission capacities, not a claim that every occupancy is hardware
  validated. The worker splits dispatches at either limit; configured
  `--max-num-seqs`, token budgets, and available cache can impose lower limits.
  Request boundaries use `query_start_loc`; each request has separate block
  tables, absolute positions, and a last-token output row on the group leader.
  The physical token extent is the largest packed group length rounded up to
  TP4 alignment, not the per-request context limit. Padding has zero inputs,
  synthetic positions, and `-1` cache mappings. Request metadata backing
  capacity follows `--max-num-seqs` within the kernel limits: the global
  request capacity is `min(max_num_seqs, 256)` and each group's metadata
  capacity is `min(max_num_seqs, 64)`. The per-group capacity is not divided
  by four, so uneven groups can share a dispatch. Worker admission follows
  these compiled capacities; dispatch descriptors bind
  only the live request-axis extent, with repeated terminal boundaries for
  groups containing fewer requests.
  Packed prefill is enabled automatically; no environment toggle is needed.
  Actual dispatch sizes follow scheduled requests and per-group token budgets,
  without waiting for a full batch. A 32-request offline run with ragged
  33-512 token prompts, 128-token chunked prefill, and K=7 speculation
  completed end-to-end with correct token accounting at eight requests per
  group. Intermittent EP16 prefill stalls reproduce without packed prefill
  (single-request controls and the kernel-side golden fixture fail the same
  way), so they are not attributed to request packing; they are tracked in
  pypto-lib#1213.
- **Decode** always runs the full 512-row group tile (16 requests x 8 rows
  per rank). Row 0 of each request carries the committed token and is the
  only accepted row in this milestone (one token per step); rows 1-7 carry
  the DSpark noise token and their compressed-boundary writes are masked so
  uncommitted rows never publish cache entries.
- **Weights** load through the standard lazy store (no prepack sidecar). The
  decode weight bank stacks all layers with the HC function matrices padded
  to 32 storage rows; the unpadded prefill slabs are derived from the same
  data. The output projection is TP-sharded (2 of 8 groups per rank) and
  regathered on device.

## Constrained Decode ABI

`serving/constraints/provider.py` owns the request-local XGrammar state. The worker
passes it with terminal prefill and decode batches; `DSparkModelRunner._stage_prefill_grammar()`
and `_stage_decode_grammar()` copy allowed-token masks into the buffers defined in
`model/deepseek_dspark/task_args.py`. The prefill and decode positional ABIs both
include `grammar_mask`; fused K7 decode also includes `valid_draft_counts`.
Serving and PyPTO-Lib must therefore be upgraded together.

Ordinary requests reuse immutable all-allowed masks and default draft counts.
The mask's row-mode header occupies segment 0's last padding word: `-1` selects
ordinary argmax, while `0` enables masking. The header is packed by Serving,
not supplied by XGrammar. Constrained requests reuse request-local host bitmask
storage. Draft validation and mask generation share one speculative traversal,
then roll back; only committed output advances the grammar. With asynchronous
scheduling, acceptance-independent metadata may prepare early, but mask
planning waits for that request's prior output to commit. Mutable slot ownership
lasts through output reclaim, and masks still use the full-tensor runtime upload
path rather than partial host-to-device updates.

## Ring heaps

Split target decode uses `(1, 1, 1, 4) GiB` at the four scope depths for the frozen
1M layout. Its deepest scope retains partial-attention output, normalization,
and stream tensors; a 1 GiB heap is insufficient. Markov uses a separate 1 GiB
profile, and the drafter uses `(4, 4, 4, 4) GiB`. Fused K=7 decode includes
the drafter and retains the upstream `(4, 4, 4, 4) GiB` profile.

`PYPTO_DSPARK_DECODE_RING_HEAP` overrides only split target decode. It accepts one
byte count (broadcast to all four scope depths) or four comma-separated byte counts.
PyPTO `RunConfig` validates the sizes: nonzero entries must be powers of two
and at least 1024; zero leaves that scope at its runtime default. To retain the default profile, use
`1073741824,1073741824,1073741824,4294967296`. A scalar `4294967296` instead
allocates 4 GiB at every depth. The legacy scalar `1073741824` leaves the deepest
scope at 1 GiB. Smaller overrides require separate device validation; they do
not change the table ABI.

The default DSpark ring heap is prefill's rebalanced profile,
`(2, 2, 4, 8) GiB` per scope depth (pypto-lib#1073); the example command pins
it explicitly. `PTO2_RING_*` environment variables are dead -- sizing flows
through the per-dispatch `RunConfig` only (pypto-lib#1075).

## Weight-bank caveat

The 16-card kernel-harness runs validated `prefill_fwd.py` and `decode_fwd.py`
at this topology; the decode witness ran with a one-layer reusable weight
bank, so serving's `--weight-bank-size 43` (all-layer resident banks) was
first exercised end-to-end by this integration.

## Accuracy checks

```bash
PYPTO_DSV4_DSPARK_MODEL_DIR=/data/models/dsv4-flash-0731-dspark-w8a8 \
TASK_DEVICE=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
python -m pytest tests/test_deepseek_dspark_accuracy.py -q
```

`tests/test_deepseek_dspark_accuracy.py` starts the HTTP server on 16 borrowed
devices, then checks greedy generation with two short cases mirroring the MTP guard's
64-token prompt / 128-token gate (the same Palace Museum prompt, so both
variants gate the same request shape): the target-only `K=0` contract and the
`K=7` speculative chain. Two additional prefix-cache cases compare cold and
cached completions at K=0 and K=7 and require an observable 128-token cache
hit. Run these alone with `-k prefix-cache`.

Greedy is the only sampling mode -- the kernels expose
device greedy sampling with no temperature ABI, so requests with
`temperature > 0` fail with an explicit error.

Unit guards (no devices needed): `tests/unit/model/deepseek_dspark/` covers
the cache topology contract, the ABI order parity against the pypto-lib
signatures (prefill/decode/drafter/markov), the import-context isolation, host
metadata lowering parity against the pypto-lib reference helpers,
prefill/decode staging assembly, the drafter lease and staging contracts,
linear-chain acceptance semantics, and the weight shard policy (synthetic
checkpoints; the production checkpoint validates opt-in through
`PYPTO_DSV4_DSPARK_MODEL_DIR`).

## Speculative decoding (K=7)

Select with `--speculative-config '{"method": "dspark",
"num_speculative_tokens": 7}'`; `0` keeps the target-only path. K is fixed at
`DSPARK_QUERY_WIDTH = 7` -- the decode tile is exactly one committed row plus
seven draft rows.

The runner owns the request-to-lease map in `_drafter_states`.
`_reserve_drafter_state()` and `release_finished_requests()` manage its lifetime;
`_seed_prefill_wave()` establishes the initial context after terminal prefill.
For decode, `_prepare_decode_plan()` and `_stage_device_prepared_decode()` build
the target inputs; `_accept_decode_outputs_on_device()` and
`_accept_decode_outputs_on_host()` apply the accepted chain. Keep the scheduler's
committed lengths as the source of truth when changing any of these steps.

**Programs.** `l3_dspark_drafter` (59 args at the pinned pypto-lib) and
`l3_distributed_markov_sample` (12 args) compile alongside the target
programs only when `num_speculative_tokens > 0`; the K=0 path loads, requires,
allocates, and compiles nothing drafter-related. The drafter dispatches under
its own `(4 GiB,)*4` ring profile, markov under the 1 GiB decode profile.

**Weights.** `DSparkWeightStore.load_drafter_weights` packs the checkpoint's
`mtp.0/1/2` modules plus three replicated heads (`mtp.0.main_proj/main_norm`,
`mtp.2.norm`, `mtp.2.markov_head.markov_w1/w2`, `mtp.2.confidence_head`) and
the target hash layers' `tid2eid` (INT32) into the kernel's flattened
three-layer banks: the checkpoint's `[out, in]` projections transpose
(`wq_a`/`wq_b`/`wkv`), the router gate and confidence head cast to FP32, the
o-projection TP-shards (`wo_a` groups / `wo_b` columns), and the routed
experts EP-shard per rank. All banks upload inside `_ensure_l3_shared_buffers`,
so the KV-capacity free-memory snapshot sees them.

**Drafter caches.** The drafter's SWA pools are runner-private
(`[16, 3, 512, 32, 1, 512]` BF16, one full group replica per rank) and never
scheduler-visible. Each live request holds a stable group-local lease in
`[0, 64)`: six ring blocks per draft layer
(`lease*6 + (logical + 7*layer) % 6`), sized so a 128-deep window plus the
seven query rows never aliases itself at any alignment. Blocks 384-389 hold a
shared read-only zero-initialized filler history; filler batch rows publish
nothing (`-1` slots, token 0, `num_sampled = 0`). Leases free on completion,
abort, and preemption; a re-admitted re-prefill starts a fresh incarnation.

**Packed-prefill seeding.** Backbone hidden rows are extracted by each
request's packed interval, including intervals crossing TP rank bands. Each
request retains its own last 128 prompt rows. The drafter seed ABI holds one
context/lease per group, so terminal-prefill requests seed in waves containing
at most one request per group; a later request cannot overwrite another
request's seed context. This does not fix the known inactive-group decode
`TENSOR_WAIT_TIMEOUT` seen when an active batch shrinks from four to three.

**Verify staging (eager publish).** Speculative decode stages the pending
drafts into verify rows 1..7 and publishes all eight rows of the window (raw
KV, recurrent states, and compression boundaries; `kv_seq_lens = start + 8`).
Rejection self-heals without rollback: every stale position a truncated
acceptance leaves behind falls inside the next dispatch's eight-row window
and is rewritten with correct tokens before any read, because reads stay
bounded by the committed lengths. When the window cannot fit under the
position ceiling the request falls back per-step to the single-anchor K=0
publication path (no clamped duplicate positions ever carry real slots), and
its next drafter query is skipped. Terminal truncation (EOS, stop, max
tokens) remains scheduler-authoritative, including a fully accepted step
ending mid-list.

**Acceptance.** The device greedy sampler emits one prediction per logit row;
the host runs linear-chain acceptance (longest matching draft prefix plus the
target's own prediction at the stop, 1..8 tokens per step) and returns
variable-length `accepted_token_ids`, which the scheduler already consumes.
Acceptance changes speed, never the served contract: the e2e guard asserts
token accounting only, plus the runner's `DSpark speculation progress` log
line as proof the chain really dispatched. Per-request counters (verify
steps, proposed/matched drafts, accepted tokens, fallback steps, mean
accepted length before scheduler truncation) are exposed through
`DSparkModelRunner.dspark_speculation_summary()`; mean accepted length is a
bring-up effectiveness goal, not a coherence or latency claim.

**Seeding.** `run_prefill` captures a rolling 128-row prompt tail from the
prefill tap's rank-owned rows on every chunk; the worker's terminal-prefill
hook (`finalize_prefill`, which knows prompt completion from its cached
length) validates the completed subset and dispatches the drafter in prefill
mode (`num_sampled = 0`, the first sampled token as `next_prefill_tokens`,
anchor at the prompt end) followed by markov, before the first decode step.
