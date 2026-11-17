# PyPTO Serving

PyPTO Serving is a local inference stack for running selected large language models with PyPTO kernels on Ascend NPUs. It provides offline generation tools, a small OpenAI-compatible HTTP server, model-specific NPU executors, and profiling hooks for understanding host and kernel time.

The project is intentionally focused. The current external surface is for Qwen3-14B and DeepSeek V4 Flash W8A8 inference on Ascend NPU environments.

## Features

- Ascend NPU backend with PyPTO kernel execution.
- Offline generation for model validation and local inference.
- OpenAI-compatible HTTP API subset for completions, chat completions, model listing, health checks, and server-side streaming.
- Qwen3-14B single-device serving, tensor-parallel offline execution, and data-parallel online replicas.
- DeepSeek V4 Flash W8A8 MTP on eight devices or DSpark on eight or 16 devices.
- Continuous batching, paged KV cache management, chunked prefill, prefix caching for supported models, and DeepSeek V4 MTP speculative decoding.
- Chrome Trace Event Format profiling across the HTTP API, scheduler, engine, worker, executor, and NPU dispatch paths.

## Start Here

- [Installation](get-started/installation.md): clone, initialize submodules, prepare the runtime, install, and verify the CLI.
- [Quickstart](get-started/quickstart.md): run Qwen3-14B offline and start the HTTP server.
- [Online Serving](user-guide/online-serving.md): start a server and send requests.
- [Tool Calling](user-guide/tool-calling.md): use function tools with supported models.
- [Models](models/index.md): choose a supported model and deployment path.
- [HTTP API](api-reference/http.md): check request fields, responses, and errors.
- [CLI Reference](cli-reference/pypto-serving.md): check command-line arguments and defaults.
- [Troubleshooting](user-guide/troubleshooting.md): diagnose setup and serving failures.

## Documentation Map

- **User Guide** explains how to install, run inference, serve HTTP traffic, scale across devices, and troubleshoot failures.
- **Models** contains checkpoint requirements, deployment commands, and model-specific runtime details.
- **Reference** documents the HTTP API and serving command.
- **Developer Guide** explains the serving architecture, model integration, profiling, runtime internals, and contribution workflow.
