# DeepSeek V4

PyPTO Serving supports DeepSeek V4 Flash through a converted W8A8 compressed-tensors checkpoint and a fixed eight-device NPU topology. The model uses overlapped attention data parallelism and MoE expert parallelism: the same eight physical ranks are attention DP=8 and MoE EP=8 ranks.

## Requirements

- A converted W8A8 compressed-tensors checkpoint.
- Exactly eight NPU device IDs.
- `--dp 8 --ep 8 --tp 1` for HTTP serving.
- `--block-size 128`.
- `--long-prefill-token-threshold 2048` or another explicit long-prompt dispatch limit up to the DeepSeek V4 8192-token main-prefill ceiling.

See [Checkpoint Conversion](#checkpoint-conversion) before starting a serving run with a source checkpoint that has not been converted to the PyPTO W8A8 layout.

## Checkpoint Conversion

PyPTO Serving expects a DeepSeek V4 W8A8 compressed-tensors checkpoint. The DeepSeek V4 Flash source checkpoint variant validated by this repository mixes FP8 weights with packed MXFP4 expert weights, so it must be converted before serving.

The conversion can run on CPU and does not require `torch_npu`. The source and output directories must be different, and the host must have enough free disk space for both copies.

Run the repository conversion utility documented in [DeepSeek V4 Conversion](../cli-reference/deepseek-v4-conversion.md). The converter writes one safetensors shard at a time using atomic replacement and supports resumable conversion after an interrupted run.

A successful run prints `Conversion complete` and leaves a converted `config.json`, `model.safetensors.index.json`, safetensors shards, and a `.pypto-w8a8-conversion.json` marker in the output directory.

After conversion, [Prepacked Weights](#prepacked-weights) describes the optional sidecar that reduces repeated startup work.

## 8-Device Offline Generation

The offline entry uses the same scheduler, worker process, rank-partitioned cache pools, and MTP acceptance path as HTTP serving, without opening a port.

```bash
PYPTO_RUNTIME_LOG=error \
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 \
  --ep 8 \
  --tp 1 \
  --block-size 128 \
  --max-model-len 512 \
  --long-prefill-token-threshold 2048 \
  --ring-dep-pool 131072 \
  --ring-task-window 131072 \
  --ring-heap 2147483648 \
  --generate-config '{"max_new_tokens": 20}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Repeat `--prompt` to exercise continuous batching, or add `--profile --profile-output /path/to/profile` to capture only the generation window after model initialization.

## 8-Device DP/EP Serving

Use the converted W8A8 checkpoint and run with overlapped attention DP=8 and MoE EP=8. Both parallel axes use the same eight physical ranks, so this is one model replica rather than eight independent serving replicas:

```bash
PYPTO_RUNTIME_LOG=error \
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --served-model-name dsv4-flash-w8a8 \
  --backend npu \
  --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 \
  --ep 8 \
  --tp 1 \
  --block-size 128 \
  --max-model-len 512 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 512 \
  --long-prefill-token-threshold 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --no-enable-prefix-caching \
  --ring-dep-pool 131072 \
  --ring-task-window 131072 \
  --ring-heap 2147483648 \
  --port 8225 \
  --show-startup-logs
```

Each NPU runs one prefill row at a time, so DP=8 admits up to eight prefill requests in one global step. The vLLM-style `--speculative-config` selects `method="mtp"`; `num_speculative_tokens` is the maximum number of draft tokens, and any positive value enables MTP. The 16-row MTP decode tile uses B8S2 for K=1, B4S4 for K=2-3, and B2S8 for K>=4. K values larger than seven are supported through repeated target verification chunks. Set `--max-num-seqs` no higher than 64, 32, or 16, respectively. Non-MTP decode retains B8S1T8. The deprecated `--num-speculative-tokens K` flag remains as a compatibility alias.

The server applies DeepSeek V4's model-specific message encoding for chat requests because the checkpoint does not ship a Jinja chat template. Chat mode is the default:

```bash
curl --noproxy "*" http://127.0.0.1:8225/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is 1+1?"}],"max_tokens":32}'
```

Pass `"reasoning_effort":"high"` or `"chat_template_kwargs":{"enable_thinking":true}` to use the model's thinking prompt. An explicit `enable_thinking` value takes priority when both are given. The current API supports string content with system, user, developer, assistant, and `latest_reminder` roles; tool calls and multimodal content are not yet exposed by the PyPTO request schema. Generation defaults to greedy sampling (`temperature=0`).

The main prefill kernel accepts a dynamic request extent up to 8192 tokens and walks it internally in 128-token tiles. The effective dispatch extent is the minimum of 8192, `--max-num-batched-tokens`, `--long-prefill-token-threshold`, and `--max-model-len`. An 8191-token prompt can therefore use one 8192-row main-prefill dispatch when those configured limits permit it, instead of 64 serving dispatches.

For repeated launches, set `PYPTO_PROG_BUILD_DIR` to a persistent directory and add `--use-compile-cache`. The first launch populates a device-specific worker subdirectory after executable assembly. Later launches reuse the compiled programs without fingerprint validation, so use the same model configuration, assigned devices, and kernel sources, and clear the directory after any change.

For MTP K greater than 1, prefix caching is disabled automatically. For routine DeepSeek V4 serving, pass `--no-enable-prefix-caching` unless you are explicitly validating prefix-cache behavior.

## Mooncake External Prefix Cache

External prefix caching is DeepSeek V4-only and extends the local HBM prefix
cache with Mooncake DRAM and optional SSD storage. PyPTO owns the seven-group
checkpoint format, stable cache keys, scheduler state, and page lifetime
fences. Mooncake owns object placement, transfer, DRAM eviction, and SSD
restore. Transfers use the multi-buffer API directly from the Simpler chip
children that own the NPU tensors; there is no host KV staging path.

The scheduler binds a zero-memory Mooncake control client to the first configured
NPU so it can query committed manifests. Only chip-child clients contribute
segments, register KV allocations, and execute data transfers.

Build and install Mooncake with Store, Ascend transport, and SSD offload support.
For a single-host evaluation, start the master with its embedded HTTP metadata
service and SSD control plane:

```bash
mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  --enable_offload=true \
  --offload_on_evict=true
```

Create the SSD root before starting serving. Each chip child automatically uses
`rank_<physical-device-id>` below this root, so no two embedded real clients
write the same bucket files.

```bash
mkdir -p /nvme/mooncake/pypto-serving
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend
export MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=1
export MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=2199023255552
export MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=67108864
```

`bucket_storage_backend` scans existing SSD metadata when a real client starts,
which permits a serving process restart to re-register valid disk objects with
the still-running master. Setting the bucket key limit to one makes each
immutable cache object durable without waiting for the default 500-key bucket
to fill; use a larger value only when delayed persistence is acceptable. Do not
use Mooncake's legacy `--root_fs_dir` together with this client-owned SSD path.

Size the local SSD staging buffer for the largest single checkpoint object; the
64 MiB example is sufficient for the current DeepSeek V4 layout. Mooncake keeps
departed clients visible until their lease expires, so a controlled serving
restart must wait for the old client to disappear from the master before
restoring its SSD objects (approximately 45 seconds with the default settings).

Copy and edit
`examples/model/deepseek_v4/mooncake_external_cache.json`. The model and
tokenizer revisions must be immutable identifiers for the exact checkpoint;
they are part of the cache namespace. `global_segment_size` and
`local_buffer_size` are per chip child, not totals across all eight devices.
Use `protocol: "ascend"` for the direct NPU data path. Set
`enable_ssd_offload` to `false` to use Mooncake DRAM only.

Set `ascend_buffer_pool: "4:8"` when the NPU environment cannot export
cross-process shareable memory, including VBS virtual devices. This enables
Mooncake's intermediate-buffer transfer mode; use `"0:0"` or omit the option
on systems where Ascend direct transfer is supported and preferred. The
Mooncake Ascend transport must be able to read `/etc/hccn.conf` in every
serving process and chip child.

Mooncake 0.3.12 does not reconstruct tenant-prefixed bucket indexes correctly
after a process restart, so use `tenant_id: "default"` when SSD offload is
enabled with that release. PyPTO's stable object namespace still separates
incompatible model and deployment revisions.

External prefix caching supports autoregressive decode and the fused one-draft
MTP path (`num_speculative_tokens=1`). Deeper MTP
(`num_speculative_tokens>1`) disables DeepSeek prefix caching and is rejected
when external caching is configured.

Add this option to the normal eight-device DeepSeek command:

```bash
--external-prefix-cache-config examples/model/deepseek_v4/mooncake_external_cache.json
```

Local HBM lookup always runs first. A committed Mooncake manifest is considered
only when it extends the local hit and meets the minimum-token threshold. During
an external load, the request waits in `WAITING_FOR_REMOTE_KV` and its target
pages remain pinned. A missing object, transfer error, cancellation, or timeout
discards the entire checkpoint and schedules cold prefill. Because Mooncake
0.3.12 cannot cancel an in-flight Store operation, pages touched by a timed-out
DMA remain quarantined until its terminal result arrives; cold prefill uses
different pages. A transfer timeout also disables new external-cache operations
for that serving process so repeated backend failures cannot consume more HBM.

`save_timeout_ms` bounds a save before cancellation is requested.
`max_pending_saves` and `max_pending_save_blocks` bound the number and total
rank-local pages pinned by queued or active saves. A newer checkpoint replaces
an older save from the same request if the older save has not reached the
worker; checkpoints are dropped while either bound is full. `failure_policy` is
`cold_miss` by default; use `fail_startup` when an unavailable Mooncake client
must prevent service startup.

PyPTO logs lookup outcomes and load/save token counts, bytes, latency,
throughput, fallbacks, waiting requests, and pinned HBM blocks. Mooncake's
master metrics distinguish memory and SSD cache hits:

```bash
curl --noproxy "*" http://127.0.0.1:9003/metrics/summary
curl --noproxy "*" http://127.0.0.1:9003/metrics
```

For an SSD-path correctness check, warm a long prefix, apply enough cache
pressure for Mooncake to evict it from DRAM, then submit the same prefix again.
Compare generated token IDs against a run with external caching disabled and
confirm that Mooncake's SSD-hit counter increases while PyPTO reports an
external load instead of full prefill.

## Weight Staging

The 49 per-layer weights are described declaratively in `pypto_serving/model/deepseek/weight_spec.py` and evaluated by the shared pipeline documented in [Weight Staging](../developer-guide/weight-staging.md). DeepSeek V4 stages serially on purpose: packing one layer allocates roughly 8 GB of intermediates, so overlapping layers increases peak memory instead of hiding latency.

DeepSeek V4 layers form three slab groups: `fwd` over all layers, `csa` over the `compress_ratio==4` layers, and `hca` over the `compress_ratio==128` layers. Each group stores its layers contiguously in first-appearance order.

See [Prepacked Weights](#prepacked-weights) for the optional sidecar that reduces repeated startup work.

## Prepacked Weights

DeepSeek V4 hidden-layer weights can be converted once into the rank-stacked host layout consumed by the serving runner. The optional sidecar reduces repeated startup work on later launches.

### Build the Sidecar

Use the [`pypto-prepack-deepseek-v4`](../cli-reference/pypto-prepack-deepseek-v4.md) CLI after converting a DeepSeek V4 Flash checkpoint to the W8A8 layout. The default sidecar path is `pypto-deepseek-v4-stacked-r8.safetensors` beside the checkpoint.

### Runtime Behavior

On startup, the DeepSeek V4 loader samples the sidecar's Linux page-cache residency before opening it, then validates a hot sidecar against the checkpoint-file and deployment fingerprint. A hot, valid sidecar is memory mapped as the final layout instead of repacking every hidden layer.

A cold, missing, or stale sidecar falls back to the original checkpoint path, avoiding a cold 323 GiB page-fault stream on the weight-upload path. Rebuild with `--force` after replacing checkpoint shards or changing the packed rank layout.

### Layout Contract

The sidecar layout follows the order produced by `DEEPSEEK_V4_LAYER_RULES`, and its metadata records a name-to-offset map built from that order. Reordering the rule table invalidates already-written sidecars.

The fingerprint covers the config, weight map, and each source shard's size and modification time, which lets startup detect a stale sidecar instead of silently using the wrong layout.

## Completion Check

Check server health first:

```bash
curl --noproxy "*" http://127.0.0.1:8225/health
```

Then send a deterministic completion request:

```bash
curl --noproxy "*" -s http://127.0.0.1:8225/v1/completions -H "Content-Type: application/json" -d '{"model":"dsv4-flash-w8a8","prompt":"Huawei is","max_tokens":25,"temperature":0.0}'
```
