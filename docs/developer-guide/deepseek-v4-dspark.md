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
devices, then checks greedy generation with one case mirroring the MTP guard's
64-token prompt / 128-token gate (the same Palace Museum prompt, so both
variants gate the same request shape). Greedy is the only
sampling mode in this milestone -- the kernels expose device greedy sampling
with no temperature ABI, so requests with `temperature > 0` fail with an
explicit error.

Unit guards (no devices needed): `tests/unit/model/deepseek_dspark/` covers
the cache topology contract, the ABI order parity against the pypto-lib
signatures, the import-context isolation, host metadata lowering parity
against the pypto-lib reference helpers, prefill/decode staging assembly, and
the weight shard policy.

## Drafter roadmap

The subsequent milestone adds the DSpark speculative chain:

1. pypto-lib#1078 lands the `dspark_target_hidden` output tap on the target
   programs (layers 40/41/42); bump the submodule pin.
2. Compile `l3_dspark_drafter` (53 args) and `l3_distributed_markov_sample`
   (12 args); drafter weights ship in the checkpoint under `mtp.0/1/2`.
3. Runner: drafter-private rank-local SWA pools, prefill-completion seeding,
   per-step verify -> accept -> draft -> markov chain, K=7 scheduler
   reservations with per-rank batch padding to `(4, 8, 12, 16)`.
4. Acceptance changes speed, never text: the e2e guard keeps asserting
   equality with this milestone's greedy output plus a mean accepted length
   above one.
