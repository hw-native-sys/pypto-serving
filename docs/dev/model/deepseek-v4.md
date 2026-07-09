# DeepSeek V4 NPU Serving Dev Notes

These commands are for DeepSeek V4 Flash W8A8 serving checks on shared Ascend
development machines with `task-submit`. Run them from the pypto-serving
checkout.

## 8-Device TP Serving

Use the quantized checkpoint under `/data/models/dsv4-flash-w8a8` and run with
TP=8 on devices 8-15:

```bash
task-submit --device 8,9,10,11,12,13,14,15 --max-time 0 --timeout 0 --ptoas 0.46 --run "PYPTO_RUNTIME_LOG=error PTO2_RING_DEP_POOL=131072 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_HEAP=2147483648 PTO2_OP_EXECUTE_TIMEOUT_US=400000000 PTO2_STREAM_SYNC_TIMEOUT_MS=440000 PTO2_SCHEDULER_TIMEOUT_MS=320000 SERVING_WORKER_STEP_TIMEOUT=1800 python python/cli/main.py --model /data/models/dsv4-flash-w8a8 --served-model-name dsv4-flash-w8a8 --backend npu --platform a2a3 --devices 8,9,10,11,12,13,14,15 --dp 1 --tp 8 --block-size 128 --max-model-len 260 --max-num-seqs 4 --max-num-batched-tokens 512 --long-prefill-token-threshold 2048 --no-enable-prefix-caching --port 8225 --show-startup-logs"
```

## Completion Check

Check server health first:

```bash
curl --noproxy "*" http://127.0.0.1:8225/health
```

Then send a deterministic completion request:

```bash
curl --noproxy "*" -s http://127.0.0.1:8225/v1/completions -H "Content-Type: application/json" -d '{"model":"dsv4-flash-w8a8","prompt":"Huawei is","max_tokens":25,"temperature":0.0}'
```
