# JIT Cache

Supported model kernels use PyPTO's persistent JIT cache. Serving does not implement a separate kernel cache or expose a cache CLI flag; its worker processes inherit the PyPTO cache environment. Install a PyPTO revision containing [#2723](https://github.com/hw-native-sys/pypto/pull/2723).

## Configure Reuse

PyPTO resolves `RunConfig.cache_config`, process defaults from `pypto.configure_cache()`, then the `PYPTO_CACHE*` environment variables. `PYPTO_CACHE=1` enables reuse; `PYPTO_CACHE=0` disables it. `PYPTO_CACHE_DIR` selects the shared artifact root (default `~/.cache/pypto/jit`) without enabling caching by itself. `PYPTO_CACHE_READONLY=1` allows lookup without writes.

```bash
PYPTO_CACHE=1 PYPTO_CACHE_DIR=/path/to/shared-jit-cache pypto-serving \
  --model /path/to/Qwen3-14B --platform a2a3
```

Leave `PYPTO_PROG_BUILD_DIR` unset for reuse. PyPTO treats it as an explicit output/rebuild request that bypasses caching; serving adds a per-worker suffix only when the user explicitly sets it. Private cache misses and runtime output use PyPTO's own unique build directories, separate from `PYPTO_CACHE_DIR`.

## Build and Lookup Behavior

Worker initialization publishes GENERATED and then READY artifacts as it prepares binaries. No separate serving `load()`/`store()` or warmup call is required. `compile()` alone does not promise READY; use a JIT function's `warmup()` API when preparing complete binaries without launching a worker. Explicit diagnostic/output requests retain PyPTO's cache bypass behavior.

Existing named build slots are not imported or trusted. PyPTO checks source, specialization, platform, and toolchain identities before reuse, so a changed input builds a new artifact rather than requiring manual cache clearing. Restart workers after a toolchain upgrade; an installation must stay unchanged during a worker's lifetime.

With `PYPTO_CACHE_READONLY=1`, a miss, invalid entry, unsupported input, or unavailable storage falls back to private compilation. For a known artifact key, storage fallback coalesces concurrent requests within one process; private compilation results are not shared across processes. Cache writers must be trusted. Different device placements or configurations can produce different keys, so one shared directory does not guarantee cross-rank hits.

## Inspect a Slow Start

Cache hits still pay toolchain discovery and content-inventory costs in a new process; one measured PyPTO baseline was about 12 seconds. Cross-process amortization is tracked in [#2732](https://github.com/hw-native-sys/pypto/issues/2732). In a worker process, `pypto.cache_stats()` reports stage hits/builds and `last_bypass_reason`. To inspect disk entries, run:

```bash
python -m pypto.jit stat --root /path/to/shared-jit-cache
```

Use `--show-startup-logs` to see loading and compilation progress. [Troubleshooting](../user-guide/troubleshooting.md#first-start-is-slow-or-cache-reuse-is-missing) covers the first checks for a deployment.
