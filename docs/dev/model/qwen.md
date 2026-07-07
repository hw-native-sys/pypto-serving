# Qwen NPU Serving Dev Notes

These commands are for Qwen3 serving checks on the shared Ascend development
machines that provide `task-submit` and `$TASK_DEVICE`. Use the README commands
for environment-neutral usage.

## Single-Device Serving

When launching through `task-submit`, keep the payload in single quotes so
`$TASK_DEVICE` expands inside the allocated task:

```bash
task-submit --device auto --max-time 1200 --run \
  'python -m python.cli.main \
    --model /data/linyifan/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --devices "$TASK_DEVICE" \
    --max-model-len 512 \
    --max-new-tokens 16 \
    --port 8899'
```

## Concurrent Serving Ring Settings

Single-request HTTP serving does not require larger PTO2 ring settings. For
concurrent Qwen NPU serving, use topology-specific values.

Single-replica concurrent serving:

```bash
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=1048576
export PTO2_RING_DEP_POOL=1048576
```

DP=2+ concurrent serving:

```bash
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072
```

Using `PTO2_RING_TASK_WINDOW=1048576` and `PTO2_RING_DEP_POOL=1048576` with a
4 GiB heap on DP=2+ can reserve about 19 GiB of runtime arena per replica and
fail with `rtMalloc failed: 207001`.

## DP=2 Serving

```bash
task-submit --device auto --device-num 2 --max-time 1800 --run \
  'export PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072; \
  python -m python.cli.main \
    --model /data/linyifan/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --devices "$TASK_DEVICE" \
    --dp 2 \
    --tp 1 \
    --max-model-len 512 \
    --max-new-tokens 16 \
    --port 8899'
```

Without the ring settings above, multi-request serving may return HTTP 200 while
generating no tokens and logging worker runtime failures such as `rtMalloc
failed: 207001`, `507018`, or `507046`.
