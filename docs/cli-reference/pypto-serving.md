# `pypto-serving`

`pypto-serving` is the main command for offline generation and HTTP serving. It starts the OpenAI-compatible server by default. Passing one or more `--prompt` arguments switches to offline generate mode and exits after the scheduled requests finish.

## Offline Generate Mode

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --generate-config '{"max_new_tokens":32,"temperature":0.0}'
```

Repeat `--prompt` to schedule multiple offline requests through the same serving engine.

## HTTP Server Mode

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512
```

## Model, Backend, and Device Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model PATH` | Required | Model directory. |
| `--served-model-name NAME` | Directory name | Model name returned by the API. |
| `--backend npu` | `npu` | Inference backend. `npu` is the only supported backend. |
| `--platform NAME` | `a2a3` | Target NPU platform. |
| `--device ID` | `0` | Single default device ID. |
| `--devices LIST` | unset | Comma-separated device IDs for multi-device placement. |
| `--dtype DTYPE` | `bfloat16` | Weight data type. |
| `--kv-cache-dtype DTYPE` | `bfloat16` | KV cache data type. `auto` follows `--dtype`. |
| `--use-compile-cache` / `--no-use-compile-cache` | inherit PyPTO policy (off by default) | Enable/disable validated persistent JIT reuse. |
| `--show-startup-logs` | off | Show model loading and kernel compilation logs. |

### Startup and Build Cache

The first NPU run may compile kernels and assemble device binaries. Serving uses
PyPTO's persistent JIT cache for Qwen, DeepSeek V4 MTP and DSpark. Install a PyPTO
revision containing [#2723](https://github.com/hw-native-sys/pypto/pull/2723).

```bash
PYPTO_CACHE_DIR=/path/to/shared-jit-cache pypto-serving \
  --model /path/to/Qwen3-14B --platform a2a3 --use-compile-cache
```

With neither flag, PyPTO resolves `RunConfig.cache_config`, process defaults from
`pypto.configure_cache()`, then `PYPTO_CACHE`, `PYPTO_CACHE_DIR` and
`PYPTO_CACHE_READONLY`. Process defaults must be configured in the worker process;
environment variables reach spawned workers. `PYPTO_CACHE=1` enables reuse without
a CLI flag. The default artifact root is `~/.cache/pypto/jit`.
`--no-use-compile-cache` explicitly disables persistence. An explicit enable flag
uses the configured root and read-only environment settings unless a Python
caller supplied a complete `RunConfig.cache_config` policy.

Leave `PYPTO_PROG_BUILD_DIR` unset for cache reuse. PyPTO treats it as an explicit
output/rebuild request, so setting it intentionally bypasses caching; serving
only adds a per-worker suffix when the user explicitly sets it. Private cache
misses/runtime output use PyPTO's own unique build directories, separate from
`PYPTO_CACHE_DIR`. Existing named build slots are not imported or trusted; they
can remain on disk while the new cache is populated. Serving no longer
needs manual cache clearing after source, specialization, platform or toolchain
changes: PyPTO checks those inputs before reuse. Toolchain installations must
remain unchanged for the lifetime of a worker; restart workers after upgrades.

Ordinary worker initialization publishes GENERATED and then READY artifacts as
it prepares the binaries. No serving `load()`/`store()` or separate warmup call
is required. `compile()` alone does not promise READY; use a JIT function's
`warmup()` API when preparing complete binaries without launching a worker.
Explicit diagnostic/output requests retain PyPTO's cache bypass behavior.

`PYPTO_CACHE_READONLY=1` allows reuse without cache writes. A miss, invalid entry,
unsupported input or unavailable storage falls back to private compilation.
For a known artifact key, storage fallback coalesces concurrent requests within
one process. Private compilation results are not shared across processes.
Cache writers must be trusted. Different device placements/configurations can
produce different keys, so sharing a directory does not guarantee hits across ranks.

Cache hits still pay toolchain discovery/content-inventory costs in each new
process; the measured PyPTO baseline was about 12 seconds on one installation.
Cross-process amortization is tracked in
[#2732](https://github.com/hw-native-sys/pypto/issues/2732). In the worker process,
`pypto.cache_stats()` reports stage hits/builds and `last_bypass_reason`;
`python -m pypto.jit stat --root /path/to/shared-jit-cache` inspects disk entries.
Use `--show-startup-logs` to see compilation progress.

## Parallelism Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--data-parallel-size`, `--dp` | `1` | Data-parallel size. |
| `--tensor-parallel-size`, `--tp` | `1` | Tensor-parallel group size. |
| `--expert-parallel-size`, `--ep` | `1` | Expert-parallel size for supported overlapped placement. |
| `--data-parallel-routing` | `least_pending_tokens` | DP request routing policy. |

For Qwen-style replica placement, the number of device IDs must equal `dp * tp`. DeepSeek V4 uses overlapped placement and requires exactly eight devices with `--dp 8 --ep 8 --tp 1`.

## Runtime Capacity Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--max-model-len` | `1024` | Maximum prompt plus generated token length. |
| `--block-size` | `128` | KV cache block size. |
| `--npu-memory-utilization` | `0.90` | Fraction of NPU memory available to the server. |
| `--max-num-seqs` | `16` | Maximum concurrent requests in serving mode. |
| `--max-num-batched-tokens` | `4096` | Maximum scheduled tokens per iteration. |
| `--long-prefill-token-threshold` | `2048` | Chunked-prefill threshold in serving mode. |
| `--ring-dep-pool` | runtime default | Simpler ring dependency-edge pool capacity. A single integer broadcasts to all scope-depth rings; a comma-separated four-integer list sizes rings 0..3, with `0` leaving that ring at its default. |
| `--ring-task-window` | runtime default | Simpler ring task-slot window capacity. Accepts the same single integer or four-entry list form as `--ring-dep-pool`. |
| `--ring-heap` | runtime default | Simpler per-ring output-heap size in bytes. Accepts the same single integer or four-entry list form as `--ring-dep-pool`. |

### Capacity Semantics

Continuous batching keeps multiple requests active across engine iterations. Requests move through waiting, prefill, decode, and finished states while the scheduler allocates KV cache pages and dispatches work that fits the configured request and token limits.

`--max-num-seqs` bounds the number of active requests. `--max-num-batched-tokens` bounds the scheduled tokens in one engine iteration. Paged KV cache capacity is determined by `--max-model-len`, `--block-size`, active request count, cache dtype, and available NPU memory.

Standard models use the generic paged KV cache layout. DeepSeek V4 uses model-specific grouped cache pools that match its decode layout and compressed-state requirements; see [DeepSeek V4](../user-guide/deepseek-v4.md) for the topology-specific command lines.

The ring options size Simpler runtime queues and output heap capacity. A single integer applies to every scope-depth ring; a four-entry list controls rings 0 through 3, with `0` preserving that ring's runtime default.

## Serving Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--host` | `0.0.0.0` | HTTP bind host. |
| `--port` | `8000` | HTTP bind port. |

## Generation Controls

Server defaults and offline generate mode use `--generate-config`. HTTP request fields override the server defaults.

| Field or option | Meaning |
| --- | --- |
| `max_tokens` / `max_new_tokens` | Maximum generated tokens. |
| `temperature` | Sampling temperature. |
| `top_p` | Nucleus sampling cutoff. |
| `top_k` | Top-k sampling cutoff. |
| `stop` | Stop strings. |
| `stream` | Stream text deltas. |
| `ignore_eos` | Generate-mode EOS handling. |

The HTTP completion path ignores EOS for completion requests and uses standard generation behavior for chat requests.

## Feature Flags

| Argument | Default | Description |
| --- | --- | --- |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | enabled | Enable or disable prefix caching for supported paths. |
| `--enable-chunked-prefill` / `--no-enable-chunked-prefill` | enabled | Enable or disable chunked prefill. |
| `--speculative-config JSON` | unset | DeepSeek V4 MTP config. |
| `--num-speculative-tokens K` | unset | Deprecated DeepSeek V4 MTP alias. |

### Feature Flag Semantics

Chunked prefill breaks long prompts into scheduler-visible chunks when a prompt exceeds `--long-prefill-token-threshold`. `--no-enable-chunked-prefill` disables that scheduler path.

Prefix caching reuses KV cache state for repeated prompt prefixes when the model path supports it. It is enabled by default for Qwen serving. DeepSeek V4 command examples disable it unless prefix-cache behavior is being validated.

`--speculative-config '{"method":"mtp","num_speculative_tokens":K}'` enables DeepSeek V4 MTP speculative decoding. The deprecated `--num-speculative-tokens` flag remains as a compatibility alias. Model-specific MTP layout and batch-size constraints are documented in [DeepSeek V4](../user-guide/deepseek-v4.md#8-device-dpep-serving).

## Profiling Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--profile` | off | Enable `/start_profile` and `/stop_profile`, or profile the offline generation window. |
| `--profile-output PATH` | `./profile_out` | Profile output directory or JSON path. |
| `--profile-level LEVELS` | `e2e,kernel` | Comma-separated profile levels. |

## Help Output

Use the installed command's help output as the source of truth for the exact arguments available in the active package:

```bash
pypto-serving --help
```
