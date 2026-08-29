# DeepSeek V4 NPU Serving Dev Notes

These commands are for DeepSeek V4 Flash W8A8 serving checks on shared Ascend
development machines with `task-submit`. Run them from the pypto-serving
checkout.

## Prepare the W8A8 Checkpoint

PyPTO serving expects a compressed-tensors W8A8 checkpoint. The released
[`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
checkpoint instead mixes FP8 weights with packed MXFP4 expert weights, so it
must be converted before serving. The conversion can run on CPU and does not
require `torch_npu`, but the source and output checkpoints must use different
directories. Make sure the machine has enough free disk space for both copies.

Install the download and safetensors dependencies, then verify that the active
PyPTO environment already provides PyTorch. Use the PyTorch build appropriate
for that environment instead of replacing it with a generic wheel.

```bash
python -m pip install --upgrade huggingface_hub safetensors
python -c "import torch, safetensors; print(torch.__version__)"
```

Download the original Hybrid FP8/MXFP4 checkpoint. If it is already available
from another official mirror, use that snapshot directory as `--input-dir`
instead.

```bash
hf download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir /data/models/DeepSeek-V4-Flash
```

Validate the source checkpoint and print the conversion plan without writing
any output:

```bash
python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /data/models/DeepSeek-V4-Flash \
  --output-dir /data/models/dsv4-flash-w8a8 \
  --dry-run
```

Run the conversion:

```bash
python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /data/models/DeepSeek-V4-Flash \
  --output-dir /data/models/dsv4-flash-w8a8
```

The converter writes one safetensors shard at a time using atomic replacement.
If the process is interrupted, rerun the same command with `--resume`:

```bash
python scripts/convert_deepseek_v4_to_w8a8.py \
  --input-dir /data/models/DeepSeek-V4-Flash \
  --output-dir /data/models/dsv4-flash-w8a8 \
  --resume
```

A successful run prints `Conversion complete` and leaves a converted
`config.json`, `model.safetensors.index.json`, safetensors shards, and a
`.pypto-w8a8-conversion.json` marker in the output directory. The resulting
index records the total tensor payload size. The directory can be passed
directly to `pypto-serving` as shown below.

## 8-Device Offline Generation

The offline entry uses the same scheduler, worker process, rank-partitioned
cache pools, and MTP acceptance path as HTTP serving, without opening a port:

```bash
task-submit --device 8,9,10,11,12,13,14,15 --max-time 0 --timeout 0 --ptoas 0.48 --run "PYPTO_RUNTIME_LOG=error PTO2_RING_DEP_POOL=131072 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_HEAP=2147483648 PTO2_OP_EXECUTE_TIMEOUT_US=400000000 PTO2_STREAM_SYNC_TIMEOUT_MS=440000 PTO2_SCHEDULER_TIMEOUT_MS=320000 SERVING_WORKER_STEP_TIMEOUT=1800 pypto-serving --model /data/models/dsv4-flash-w8a8 --prompt 'Huawei is' --platform a2a3 --devices 8,9,10,11,12,13,14,15 --dp 8 --ep 8 --max-model-len 512 --generate-config '{\"max_new_tokens\": 20}' --num-speculative-tokens 1"
```

Repeat `--prompt` to exercise continuous batching, or add
`--profile --profile-output /path/to/profile` to capture only the generation
window after model initialization.

## 8-Device DP/EP Serving

Use the quantized checkpoint under `/data/models/dsv4-flash-w8a8` and run with
overlapped attention DP=8 and MoE EP=8 on devices 8-15. Both parallel axes use
the same eight physical ranks, so this is one model replica rather than eight
independent serving replicas:

```bash
task-submit --device 8,9,10,11,12,13,14,15 --max-time 0 --timeout 0 --ptoas 0.48 --run "PYPTO_RUNTIME_LOG=error PTO2_RING_DEP_POOL=131072 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_HEAP=2147483648 PTO2_OP_EXECUTE_TIMEOUT_US=400000000 PTO2_STREAM_SYNC_TIMEOUT_MS=440000 PTO2_SCHEDULER_TIMEOUT_MS=320000 SERVING_WORKER_STEP_TIMEOUT=1800 pypto-serving --model /data/models/dsv4-flash-w8a8 --served-model-name dsv4-flash-w8a8 --backend npu --platform a2a3 --devices 8,9,10,11,12,13,14,15 --dp 8 --ep 8 --tp 1 --block-size 128 --max-model-len 512 --max-num-seqs 32 --max-num-batched-tokens 512 --long-prefill-token-threshold 2048 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' --no-enable-prefix-caching --port 8225 --show-startup-logs"
```

Each NPU runs one prefill row at a time, so DP=8 admits up to eight prefill
requests in one global step. The vLLM-style `--speculative-config` selects
`method="mtp"`; `num_speculative_tokens` is the maximum number of draft tokens,
and any positive value enables MTP. The
16-row MTP decode tile uses B8S2 for K=1, B4S4 for K=2-3, and B2S8 for
K>=4. K values larger than seven are supported through repeated target
verification chunks. Set `--max-num-seqs` no higher than 64, 32, or 16,
respectively. Non-MTP decode retains B8S1T8. The deprecated
`--num-speculative-tokens K` and `--enable-mtp`
flags remain compatibility aliases; `--enable-mtp` selects K=1.

The main prefill kernel supports a dynamic request extent up to 8192 tokens and
walks it internally in 128-token tiles. AR and MTP use the same main-prefill
request limit; the effective dispatch extent is the minimum of 8192,
`--max-num-batched-tokens`, `--long-prefill-token-threshold`, and
`--max-model-len`. AR and MTP submit each main-prefill chunk once, with its
backing extent padded to the next 128-token tile. Thus an 8191-token prompt uses
one 8192-row main-prefill dispatch when those configured limits permit it,
rather than 64 separate serving dispatches. The 128-token width is an internal
kernel tile, not a serving chunk restriction. The main kernel returns each
owner's final 128 valid pre-HC rows, which the standalone fixed-width MTP prefill
kernel uses to rebuild its 127-row KV window before retaining the final row for
sampling.

For repeated launches, set `PYPTO_PROG_BUILD_DIR` to a persistent directory and
add `--use-compile-cache`. The first launch populates a device-specific worker
subdirectory after executable assembly. Later launches reuse the compiled
programs without fingerprint validation, so use the same model configuration,
assigned devices, and kernel sources, and clear the directory after any change.

MTP prefill context, draft token, recurrent hidden state, and acceptance
counters are owned by request ID. MTP prefill and decode share one
worker-resident cache, but each request addresses it with the scheduler-owned
rank-local `ori` block IDs.
The scheduler reserves all K speculative positions before dispatch, including
when a draft sequence crosses a 128-token page boundary.

Before the first decode is prepared, each request owns a stable rank-local
device-state slot and reuse generation. Terminal prefill fills that reserved
slot with the committed tail token, next draft token, tail position, and
committed count. The fused decode kernel
uses `(rank, slot, generation)` to build the next `[tail, draft]` input rows and
sequence metadata before main decode, then updates the same slot after MTP
verification. Host output processing mirrors the state for scheduling and
statistics, but is not an input dependency of the next steady-state decode.
Generation matching prevents a stale queued step from updating a slot after
preemption and reuse.

The seven main-model KV/state pools are allocated during runner preflight as
rank-sharded worker-resident tensors. Prefill and decode pass the same device
handles and address them with scheduler-owned group block IDs; there is no
prefill CPU snapshot or cache handoff. Reassigned pages are cleared with
targeted host-to-device copies before their new owner writes them.

## Weight Staging

The 49 per-layer weights are described declaratively in
`model/deepseek/weight_spec.py` and evaluated by the shared pipeline in
`model/common/weights/`; see [Weight staging](weight-staging.md) for the rule
kinds, the slab geometry and the invariants. Two things specific to this model:
staging is **serial** on purpose — packing one layer allocates ~8 GB of
intermediates, so overlapping layers multiplies the peak rather than hiding
latency — and the layers form three slab groups (`fwd` over all layers, `csa`
over the `compress_ratio==4` ones, `hca` over the `==128` ones), each holding its
layers contiguously in first-appearance order.

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
on systems where Ascend direct transfer is supported and preferred.
The Mooncake Ascend transport must be able to read `/etc/hccn.conf` in every
serving process and chip child.
Mooncake 0.3.12 does not reconstruct tenant-prefixed bucket indexes correctly
after a process restart, so use `tenant_id: "default"` when SSD offload is
enabled with that release. PyPTO's stable object namespace still separates
incompatible model and deployment revisions.

External prefix caching supports autoregressive decode and the fused one-draft
MTP path (`num_speculative_tokens=1`). Deeper MTP (`num_speculative_tokens>1`)
still disables DeepSeek prefix caching and is rejected when external caching is
configured.

Add these options to the normal eight-device DeepSeek command:

```bash
--external-prefix-cache-backend mooncake \
--external-prefix-cache-config examples/model/deepseek_v4/mooncake_external_cache.json \
--external-prefix-cache-min-tokens 1024 \
--external-prefix-cache-load-timeout-ms 30000 \
--external-prefix-cache-transfer-concurrency 2
```

Local HBM lookup always runs first. A committed Mooncake manifest is considered
only when it extends the local hit and meets the minimum-token threshold. During
an external load, the request waits in `WAITING_FOR_REMOTE_KV` and its target
pages remain pinned. A missing object, transfer error, cancellation, or timeout
discards the entire checkpoint and schedules cold prefill. `failure_policy` is
`cold_miss` by default; use `fail_startup` when an unavailable Mooncake client
must prevent service startup.

PyPTO logs lookup outcomes and load/save token counts, bytes, latency,
throughput, fallbacks, waiting requests, and pinned HBM blocks. Mooncake's
master metrics distinguish memory and SSD cache hits. Inspect the summary or
Prometheus endpoint on the default admin port:

```bash
curl --noproxy "*" http://127.0.0.1:9003/metrics/summary
curl --noproxy "*" http://127.0.0.1:9003/metrics
```

For an SSD-path correctness check, warm a long prefix, apply enough cache
pressure for Mooncake to evict it from DRAM, then submit the same prefix again.
Compare generated token IDs against a run with external caching disabled and
confirm that Mooncake's SSD-hit counter increases while PyPTO reports an
external load instead of full prefill.

## Optional Prepacked Weights

The 43 hidden layers can be converted once into the final rank-stacked Host
layout:

```bash
pypto-prepack-deepseek-v4 /data/models/dsv4-flash-w8a8
```

The command atomically writes
`pypto-deepseek-v4-stacked-r8.safetensors` beside the checkpoint. Subsequent
starts sample its Linux page-cache residency before opening it. A hot sidecar is
validated against the checkpoint-file and deployment fingerprint, then
memory-mapped as the final layout instead of repacking every hidden layer. A
cold, missing, or stale sidecar uses the original checkpoint path, avoiding a
cold 323 GiB page-fault stream on the weight-upload path. Rebuild with `--force`
after replacing checkpoint shards or changing the packed rank layout.

The sidecar's layout is the order the rule table produces, and its metadata
records a name-to-offset map built from that order — so reordering
`DEEPSEEK_V4_LAYER_RULES` invalidates every sidecar already written. The
fingerprint covers the config, the weight map and each source shard's size and
mtime, which is what makes a stale sidecar detectable rather than silently
wrong.

## Completion Check

Check server health first:

```bash
curl --noproxy "*" http://127.0.0.1:8225/health
```

Then send a deterministic completion request:

```bash
curl --noproxy "*" -s http://127.0.0.1:8225/v1/completions -H "Content-Type: application/json" -d '{"model":"dsv4-flash-w8a8","prompt":"Huawei is","max_tokens":25,"temperature":0.0}'
```
