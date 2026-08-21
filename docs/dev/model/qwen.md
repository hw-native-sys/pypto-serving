# Qwen NPU Serving Dev Notes

These commands are for Qwen3 serving checks on the shared Ascend development
machines that provide `task-submit`. Use the README commands for environment-neutral usage.

## Single-Device Serving

```bash
task-submit --device auto --max-time 1200 --run \
  "pypto-serving \
    --model /data/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --devices {} \
    --max-model-len 512 \
    --port 8899"
```

## Weight Staging

Layer weights are described in `model/qwen/weight_spec.py` and staged by the
shared pipeline in `model/common/weights/`; see
[Weight staging](weight-staging.md) for the rule kinds and the invariants.

The loader reads **metadata only** for the per-layer weights: each layer is read,
written into its slab slice and dropped before the next one, which keeps the
staging peak at roughly one layer per worker instead of a second copy of the
model. The globals stay eager because `Executor.lookup_embeddings` reads
`embed_tokens` at request time.

Three details specific to this model: layers stack on **axis 0** (there is no
rank axis, unlike DeepSeek V4), projections are stored transposed so every one
carries `transpose=True`, and the slabs are allocated in **shared memory**
because the upload reads them from a forked child. A checkpoint without QK norms
is a supported variant — the rules default those gammas to ones, not zeros.

## DP=2 Serving

```bash
task-submit --device auto --device-num 2 --max-time 1800 --run \
  "pypto-serving \
    --model /data/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --devices {} \
    --dp 2 \
    --tp 1 \
    --max-model-len 512 \
    --port 8899"
```
