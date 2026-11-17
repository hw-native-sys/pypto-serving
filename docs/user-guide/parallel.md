# Parallelism and Scaling

PyPTO Serving supports replica and overlapped placement. The model executor selects its supported mode and validates the requested topology. Single-device serving remains the default where the model supports it.

## Replica Placement

In replica placement, data parallelism creates independent serving replicas. Each replica owns one tensor-parallel device group, so `dp * tp` device IDs are needed. Qwen uses this mode:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --devices 0,1 \
  --dp 2 \
  --tp 1 \
  --max-model-len 512 \
  --port 8899
```

The server routes requests across replicas using `least_pending_tokens`.

## Offline Generate Runs

Offline generation uses `pypto-serving --prompt`, the same engine as HTTP serving. For one tensor-parallel worker group:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1 \
  --tp 2 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens": 16}'
```

Repeat `--prompt` to schedule multiple prompts. In replica placement, `--devices`, `--tp`, and `--dp` behave as they do in serving: prompts are routed across replicas with the same least-pending-tokens policy.

## Overlapped Placement

In overlapped placement, parallel axes can share physical ranks within one model worker group. The required device count and supported `--dp`, `--ep`, and `--tp` values are model-specific. For example, DeepSeek V4 MTP reuses eight ranks for attention DP and MoE EP. See its [model guide](../models/deepseek-v4/mtp.md#8-device-dpep-serving) for the complete command and constraints.
