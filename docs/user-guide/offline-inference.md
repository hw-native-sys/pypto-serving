# Offline Inference

Offline inference runs `pypto-serving --prompt` without opening an HTTP port. It uses the same scheduler, worker, and model executor as HTTP serving. Use it for model validation, kernel checks, or generating from several prompts in one run.

## Run a Workload

Add `--prompt` to a model's deployment command to run offline. Repeat it to submit multiple requests through the same engine:

```text
--prompt 'Huawei is' --prompt 'Ascend is'
```

Start with the [Qwen3-14B](../models/qwen3-14b.md#offline-generation) or [DeepSeek V4 MTP](../models/deepseek-v4/mtp.md#8-device-offline-generation) example. DSpark uses a different [eight- or 16-device topology](../models/deepseek-v4/dspark.md); its server command also accepts `--prompt` for offline generation. Model-specific checkpoint, block-size, and speculative decoding requirements remain the same in either mode.

Use `--generate-config` for output length and sampling defaults. The [CLI Reference](../cli-reference/pypto-serving.md#generation-controls) lists its fields; `--profile --profile-output /path/to/profile` captures the generation window as described in [Profiling](../developer-guide/profile.md#profile-offline-generation).

## Output

Offline generate mode prints generated text, token IDs, finish reason, and a throughput summary. For startup failures, follow [Troubleshooting](troubleshooting.md).
