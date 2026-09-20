# Model Support Matrix

PyPTO Serving supports the model families below on Ascend NPU backends. The matrix lists public serving configurations; development-only kernel harnesses are not included.

## Models

| Model family | Checkpoint layout | Devices | Parallelism | Offline | HTTP serving | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-14B | Hugging Face style checkpoint | One or more Ascend NPUs | Single device, TP offline, DP online replicas | Supported | Supported | Uses the Qwen model loader, tokenizer, NPU executor, and PyPTO kernels. |
| DeepSeek V4 Flash W8A8 | Converted compressed-tensors checkpoint | Exactly eight Ascend NPUs | Overlapped attention DP=8 and MoE EP=8, TP=1 | Supported | Supported | Requires `--dp 8 --ep 8 --tp 1` and `--block-size 128`. |
| DeepSeek V4 Flash DSpark W8A8 | Converted compressed-tensors checkpoint | Exactly 16 Ascend NPUs | DSpark target kernels with DP=4, EP=16, TP=4 | Supported | Supported | Select with `--speculative-config '{"method":"dspark","num_speculative_tokens":0}'`; requires `--block-size 32` and greedy generation. |

See [Qwen3-14B](qwen.md), [DeepSeek V4](deepseek-v4.md), and [DeepSeek V4 DSpark](../developer-guide/deepseek-v4-dspark.md) for command examples and topology details.

## Serving Features

| Feature | Qwen3-14B | DeepSeek V4 Flash W8A8 | DeepSeek V4 Flash DSpark W8A8 | Notes |
| --- | --- | --- | --- | --- |
| Text generation | Supported | Supported | Supported | Offline and HTTP paths share the serving engine. |
| `/v1/completions` | Supported | Supported | Supported | |
| `/v1/chat/completions` | Supported | Supported | Supported | DeepSeek V4 uses model-specific message encoding. |
| Streaming | Supported | Supported | Supported | Completion and chat streams use Server-Sent Events. |
| Usage counts | Supported | Supported | Supported | Non-streaming responses and terminal stream chunks include token counts. |
| Continuous batching | Supported | Supported | Supported | Scheduler behavior is controlled by runtime capacity settings. |
| Paged KV cache | Supported | Supported | Supported | DeepSeek V4 uses grouped, model-specific cache pools. |
| Chunked prefill | Supported | Supported | Supported | Enabled by default; controlled by CLI flags. |
| Prefix caching | Supported | Limited | Not supported | Qwen enables it by default. Routine DeepSeek V4 serving should disable it; DSpark forces it off. |
| MTP speculative decoding | Not supported | Supported | Not supported | Use `--speculative-config '{"method":"mtp","num_speculative_tokens":K}'` for the non-DSpark DeepSeek V4 path. |
| General speculative decoding | Not supported | Not supported | Not supported | Draft-model, n-gram, EAGLE, and suffix speculation are not exposed. |
| Logprobs and prompt logprobs | Not supported | Not supported | Not supported | Request schemas do not expose logprob fields. |
| Beam search and best-of | Not supported | Not supported | Not supported | Qwen and DeepSeek V4 expose greedy/sampling controls; DSpark is greedy-only. |
| Structured outputs | Not supported | Not supported | Not supported | JSON schema, grammar, and guided decoding are not exposed. |
| Tool calling | Not supported | Not supported | Not supported | Tool schemas and tool-call response parsing are not exposed. |
| Multimodal inputs | Not supported | Not supported | Not supported | Chat content is string-only. |
| Embeddings and pooling | Not supported | Not supported | Not supported | `/v1/embeddings` is not exposed. |
| LoRA adapters | Not supported | Not supported | Not supported | Runtime adapter loading is not exposed. |
| General quantization matrix | Not supported | Limited | Limited | DeepSeek V4 uses the validated W8A8 conversion path only. |

## Hardware Scope

The public runtime backend is Ascend NPU with PyPTO kernels. `--backend npu` is currently the only accepted backend; CPU and GPU fallback backends are not exposed by the `pypto-serving` CLI.

## Adding a Model

New models need more than a Hugging Face `config.json`. A model integration must add or reuse model-family detection, checkpoint loading, weight-spec rules, runner and executor code, PyPTO kernels, tests, and user documentation. See the [Model Integration](../developer-guide/model-integration/index.md) guide.
