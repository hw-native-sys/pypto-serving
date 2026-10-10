# CLI Reference

PyPTO Serving exposes command-line tools for serving, offline generation, DeepSeek V4 checkpoint preparation, and runtime integration.

## Installed Commands

| Command | Purpose |
| --- | --- |
| [`pypto-serving`](pypto-serving.md) | Start the OpenAI-compatible HTTP server, or run offline generation when `--prompt` is set. |
| [`pypto-serving-router`](pypto-serving-router.md) | Route requests across several serving replicas, keeping each conversation on the replica holding its KV cache. |
| [`pypto-prepack-deepseek-v4`](pypto-prepack-deepseek-v4.md) | Build the optional DeepSeek V4 rank-stacked weight sidecar used by the serving loader. |

## Repository Utilities

| Command | Purpose |
| --- | --- |
| [`python scripts/convert_deepseek_v4_to_w8a8.py`](deepseek-v4-conversion.md) | Convert the DeepSeek V4 Flash source checkpoint variant validated by this repository to the W8A8 layout expected by PyPTO Serving. |

## Workflow Guides

Use the CLI reference for argument meanings and defaults. Use the workflow guides for complete serving runs:

- [Offline Inference](../user-guide/offline-inference.md)
- [Online Serving](../user-guide/online-serving.md)
- [Parallelism and Scaling](../user-guide/parallel.md)
- [Multi-Node Routing](../user-guide/multi-node-routing.md)
- [Profiling](../user-guide/profile.md)
