# DeepSeek V4 DSpark NPU Serving Dev Notes

Serving notes for the DSpark target kernels
(`pypto-lib/models/deepseek_v4_flash_dspark`): the 16-card TP4/DP4/EP16
deployment of the DeepSeek-V4-Flash W8A8 checkpoint with the DSpark decode
tile (64 requests x 8 rows per TP group), block size 32, and the paged
full-context prefill tables.

The current milestone serves the **target model only** -- prefill, decode, and
greedy generation without speculation. The DSpark drafter chain is a
subsequent milestone; see "Drafter roadmap" below.

## Topology and selection

DSpark serves the same W8A8 checkpoint format as the MTP variant
(`docs/cli-reference/deepseek-v4-conversion.md` describes the conversion). Select it with the
speculative-config method:

```bash
python -m pypto_serving.cli \
  --model /data/models/dsv4-flash-0731-dspark-w8a8 \
  --backend npu --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --dp 4 --ep 16 --tp 4 \
  --block-size 32 --max-model-len 1024 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 --long-prefill-token-threshold 128 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":0}' \
  --no-enable-prefix-caching \
  --ring-heap 2147483648,2147483648,4294967296,8589934592 \
  --port 8000
```

Validated constraints (enforced at startup):

- Exactly 16 devices with `--dp 4 --ep 16 --tp 4`. The 16 ranks form 4 TP
  groups; `moe.py` rescales `n_routed_experts` by `EP/16`, so other EP values
  compile a wrong expert view.
- `--block-size 32` (the DSpark page size; the MTP variant uses 128).
- `--max-num-seqs` at most 256 (64 requests per TP group).
- `--max-model-len` at most 16384: the decode cache tables cap context at
  16K until pypto-lib#905 extends them. Prefill itself is 1M-capable
  (pypto-lib#1073), so longer prompts chunk through the 8192-token dispatch
  bound whenever decode catches up.
- Prefix caching is forced off for now.

## How the deployment maps onto the kernels

- **Cache partitions = TP groups.** The scheduler sees 4 partitions. Every
  rank of a group holds an identical replicated pool: prefill writes the same
  rows on all four ranks, and decode rebuilds the group's whole KV stream
  from the gathered token rows each step (pypto-lib#1079).
- **Block tables are staged at the kernels' frozen depths** (CSA compressed
  and indexer tables 8192 entries, HCA compress-state table 131072, CSA
  compress-state tables 524288, -1 past
  each request's span). The generated orchestration reshapes these tables
  with the frozen depths baked in, so a shallower table asserts on device and
  surfaces as an opaque AICore 507901; the depths therefore track the kernel
  constants (`IDX_MAX_BLOCKS`, `CMP_MAX_BLOCKS`,
  `COMPRESS_STATE_MAX_BLOCKS`), never the serving context ceiling.
- **Prefill** runs one request per TP group per dispatch at the fixed
  8192-token physical extent; logical rows are padded per the pypto-lib#1069
  contract (zero-padded inputs, `-1` cache mappings, logical-only logit
  rows). A group leader publishes one greedy-sampled token.
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

## Ring heaps

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
devices, then checks greedy generation with two cases mirroring the MTP guard's
64-token prompt / 128-token gate (the same Palace Museum prompt, so both
variants gate the same request shape): the target-only `K=0` contract and the
`K=7` speculative chain. Greedy is the only sampling mode -- the kernels expose
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
