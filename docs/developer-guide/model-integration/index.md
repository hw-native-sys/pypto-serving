# Model Integration

Adding a new model family to PyPTO Serving requires implementing several components that bridge the gap between a Hugging Face-style checkpoint on disk and the NPU kernels that execute prefill and decode. This guide walks through each step.

## Prerequisites

Before starting, you should be familiar with:

- The model's architecture (number of layers, hidden size, attention heads, etc.)
- The model's checkpoint format (safetensors, config.json structure)
- Basic PyPTO concepts: L3 callables, `DistributedWorker`, `TaskArgs`
- Whether the model needs NPU kernel sources in `pypto-lib/`

## Integration Steps

The integration follows a linear pipeline. Each step is documented in its own page:

| Step | Page | What you produce |
|------|------|------------------|
| 1 | [Family Detection](family-detection.md) | A `ModelFormatLoader` that detects and loads your checkpoint |
| 2 | [Weight Spec](weight-spec.md) | A rule table mapping checkpoint tensors to kernel weight names |
| 3 | [Executor & Runner](executor-runner.md) | NPU executor and runner for prefill and decode |
| 4 | [Testing](testing.md) | Unit tests and NPU validation |

## Reference Implementations

The codebase has two reference implementations that demonstrate the range of complexity:

- **Qwen3-14B** (`pypto_serving/model/qwen/`): Single-device, simple weight layout, 3 kernel modules, no speculative decoding. The best starting point for a new single-rank model.

- **DeepSeek V4 Flash W8A8** (`pypto_serving/model/deepseek/`): Eight-rank overlapped DP/EP, compressed KV cache, MTP speculative decoding, 4 kernel modules. The reference for distributed models.

## Additional Resources

- [Architecture](../architecture.md) — overview of the serving stack
- [Weight Staging](../weight-staging.md) — deep dive into the weight staging pipeline internals
- [DeepSeek V4 Runtime](../deepseek-v4-runtime.md) — model-specific runtime details

## Checklist

- [ ] Model family detection: `detect_model_family()` or `ModelFormatLoader`
- [ ] Model loading and tokenizer handling
- [ ] `RuntimeConfig` and KV cache requirements
- [ ] Weight spec rules under `pypto_serving/model/<family>/weight_spec.py`
- [ ] NPU executor: `_compile_model()` and `_create_runner()`
- [ ] NPU runner: `run_prefill()`, `run_decode()`, `init_kv_cache()`
- [ ] PyPTO kernel sources in `pypto-lib/` or existing kernels
- [ ] Task args: kernel argument contracts
- [ ] Offline generation example commands
- [ ] HTTP serving topology validation in the CLI path
- [ ] Unit tests for config and scheduler behavior
- [ ] NPU validation or accuracy checks
- [ ] Model documentation under `docs/user-guide/`