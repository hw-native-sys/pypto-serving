# Models

Choose the checkpoint and device topology before following a deployment command. The rows describe supported paths, not interchangeable CLI presets.

| Path | Checkpoint | Devices and placement | Decode | HTTP tools |
| --- | --- | --- | --- | --- |
| [Qwen3-14B](qwen3-14b.md) | Hugging Face style checkpoint | Single device; TP for offline runs; DP replicas for HTTP serving | Standard generation | Function tools without structural constraints |
| [DeepSeek V4 MTP](deepseek-v4/mtp.md) | Converted Flash W8A8 | 8 devices, overlapped DP8/EP8/TP1 | Autoregressive or MTP speculative | Function tools without structural constraints |
| [DeepSeek V4 DSpark](deepseek-v4/dspark.md) | Converted Flash W8A8 | 8 devices, DP2/EP8/TP4; or 16 devices, DP4/EP16/TP4 | Greedy target-only K=0 or speculative K=7 | Function tools; [structural constraints](deepseek-v4/tool-constraints.md) on K=7 |

All three paths support offline generation through `--prompt` and HTTP completions/chat completions. DSpark currently requires greedy sampling; prefix caching and request limits depend on the path and speculative mode. See each deployment guide for those limits, [Offline Inference](../user-guide/offline-inference.md) and [Tool Calling](../user-guide/tool-calling.md) for common workflows, and the [CLI Reference](../cli-reference/pypto-serving.md) for shared arguments.
