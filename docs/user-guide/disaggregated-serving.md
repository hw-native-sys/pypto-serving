# Disaggregated Serving

Disaggregated serving runs Prefill and Decode in separate model processes.
An external Router accepts completion and chat requests, selects compatible
P/D nodes, and streams the Decode response back to the client. Model data moves
directly between the nodes; it does not pass through the Router.

The current model adapter supports DeepSeek V4 Flash DSpark K7 with greedy
sampling. Prefill runs the target model, while Decode owns its local drafter.
The supported transfer provider is Mooncake AscendDirect over RoCE. Data is
sent after each Prefill chunk, not after each model layer.

## Requirements

- A working non-disaggregated DSpark K7 installation on each model node, with
  matching model weights, kernel layouts, and parallelism.
- The current validated layout uses 16 devices per node, with DP=4, EP=16,
  TP=4 and a 32-token block size.
- Mooncake's Python module and native libraries available in each node's
  activated runtime environment.
- PyPTO/Simpler owner-local service support: `chip_service_factories`,
  `retain_service_tensor`, and `release_service_tensor`. Memory registration
  must run inside the process that owns the device allocation. These are
  dependency requirements, not runtime patches applied by Serving.
- Reachable node HTTP endpoints, the P/D control endpoint, and device RoCE
  interfaces. Set `HCCL_INTRA_ROCE_ENABLE=1` on both model nodes so the transfer
  provider can coexist with the model's compute communication domain.

## Shared Configuration

Copy [the example configuration](../../examples/pd/config.json) and replace
the two hostnames. Router, Prefill and Decode must load the same document:

```json
{
  "runtime": {
    "prefill": [{"host": "prefill.example.internal", "port": 8111}],
    "decode": [{"host": "decode.example.internal", "port": 8112}]
  }
}
```

Model loading and compute options remain regular `pypto-serving` arguments.
The model adapter derives the PD contract and buffer layout from the loaded
runtime; they are not duplicated in this JSON document.

Optional runtime settings include:

| Setting | Default | Purpose |
| --- | --- | --- |
| `provider` | `mooncake` | Select the supported transfer implementation. |
| `policy` | `round_robin` | Select compatible P/D nodes. |
| `prefix_cache_mode` | `disabled` | Use `d_only` for Decode reuse or `independent` for separate P/D reuse. |
| `run_id` | Derived from the endpoints | Share an explicit deployment identity when needed. |
| `enable_chunk_overlap` | `true` | Overlap closed-page transfer with later Prefill work. |

An endpoint can also specify `node_id`, `control_host`, `control_port` and
`transfer_hostname`. The default control port is 29831. When the device RoCE
address differs from the HTTP hostname, set `transfer_hostname` to the address
required by Mooncake on that node.

To configure multiple nodes, add endpoints to either array, give each one a
unique `node_id`, and pass `--pd-node-id` to its model process. The Router
checks their runtime contracts and selects a compatible pair using the
configured policy. Each model/service deployment uses its own Router and
configuration.

The current data plane retains one peer registration per model process.
The routing interfaces support multiple candidates, but concurrent multi-peer
handoffs require a peer-indexed session/registry pool and are not yet validated.

## Start the Three Processes

Activate the normal Serving environment on each node first. These commands
run in the foreground; process placement and supervision belong to your
deployment system.

On each model node, define the same model options, adjusting the model path
and device IDs for that node:

```bash
model_args=(
  --model /path/to/dsv4-flash-dspark-w8a8
  --served-model-name dsv4-flash-dspark-w8a8
  --backend npu --platform a2a3
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  --dp 4 --ep 16 --tp 4
  --block-size 32 --max-model-len 293888
  --max-num-seqs 8 --max-num-batched-tokens 8192
  --long-prefill-token-threshold 128
  --generate-config '{"max_new_tokens":131072}'
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}'
  --ring-heap 2147483648,2147483648,4294967296,8589934592
)
```

Start Prefill on its assigned node:

```bash
HCCL_INTRA_ROCE_ENABLE=1 pypto-serving "${model_args[@]}" \
  --pd-role prefill --pd-config examples/pd/config.json \
  --host 0.0.0.0 --port 8111
```

Start Decode on its assigned node:

```bash
HCCL_INTRA_ROCE_ENABLE=1 pypto-serving "${model_args[@]}" \
  --pd-role decode --pd-config examples/pd/config.json \
  --host 0.0.0.0 --port 8112
```

After both nodes report healthy, start the CPU-only Router in its own process:

```bash
pypto-pd-router --pd-config examples/pd/config.json --host 0.0.0.0 --port 8110
```

The shared prefix-cache mode is authoritative for PD nodes; a separate
`--enable-prefix-caching` flag does not override it. To run ordinary combined
Prefill/Decode serving, omit the PD options and use the normal model command.

## Send Requests and Inspect State

Clients use the Router's OpenAI-compatible endpoints:

```bash
curl http://router.example.internal:8110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-flash-dspark-w8a8","messages":[{"role":"user","content":"Describe the Forbidden City."}],"max_tokens":128,"temperature":0.0,"stream":true}'
```

`/v1/completions` is also supported. P/D nodes expose internal control APIs
instead of public generation endpoints.

| Endpoint | Process | Contents |
| --- | --- | --- |
| `/health` | Router, P, D | Readiness and recovery gate. |
| `/metrics` | Router | Request outcomes and routing metrics. |
| `/recovery` | Router | Recovery state and generation. |
| `/internal/pd/metrics` | P, D | Prefix hits, transfer progress and request cleanup. |
| `/internal/pd/capacity` | P, D | Active requests, reservations and quarantined allocations. |

Default logs and state live under
`serving_log/pd_disag/<run_id>/<role>/`, relative to the launch directory.
Model-node directories additionally include the resolved node ID. Optional
`observability.enabled` and `observability.root` control diagnostic output;
the lifecycle journal remains available when diagnostics are disabled.

## Prefix Reuse and Recovery

Prefix matching and publication belong to Serving's `KvCacheManager`. P's hit
length determines its computation start; D's independent hit determines the
network suffix. Reservations retain a read-only shared prefix and separately
own writable suffix/state pages. Even a full KV hit must transfer the final
request-local compressed-attention state before Decode can start.

P and D also choose cache partitions independently. The model adapter maps
the actual source partition to D's reserved partition once per handoff. For
the validated TP=4 layout, P ranks 4–7 can send to D ranks 8–11, pairing equal
TP slots. Source and destination block tables, owners and leases remain
separate; the immutable mapping is included in the manifest hash and checked
again when completion is recorded. No Router round trip or user-supplied rank
map is required. A partition change during a handoff is rejected.

All chunks use asynchronous worker submission and Host completion polling.
Mooncake's native call is synchronous inside an owner-local progress thread;
it does not wait in the model worker's command loop. The overlap setting only
controls whether the next Prefill chunk may compute before the current transfer
finishes. Overlap is enabled by default: one chunk transfer per request may overlap the next
compute step. Only closed pages are sent early, source snapshots stay pinned,
and rolling pages detach before reuse. Final request-local state is transferred
after the final Prefill step. Decode starts only after all required writes and
commit complete; its normal asynchronous scheduling pipeline is unchanged.

Set `runtime.enable_chunk_overlap` to `false` to serialize each request's
Prefill chunks with its transfers, for example when comparing traces. Worker
submission and completion polling remain asynchronous in either mode; this
switch does not select a blocking native call in the model execution lane.
Multi-chunk K7 regressions verified matching output token digests with overlap
enabled and disabled; the enabled run also exercised cross-partition handoffs.
Host traces confirmed overlapping
native transfer calls and later Prefill dispatch intervals. These are runtime
timeline measurements, not a hardware-stream utilization or bandwidth guarantee.

The D-local drafter lease is reserved on the Host. Clearing a reused lease
and publishing its initial device state happen once on the device execution
lane, after older run handles settle. This initial-admission boundary is not
a per-token synchronization or a reason to disable asynchronous Decode.

Deterministic failures release allocations only when no writer can access
them. An uncertain transfer quarantines writable allocations and closes the
affected admission path. Healthy shared prefix pages remain separate from
that quarantine.

Automatic process restart is not enabled by default. Recovery is performed
manually or by an external supervisor, which must confirm that the old owner
processes have exited before starting a new arena. Use one fresh `run_id`
shared by Router/P/D for a replacement deployment; reusing an old journal
does not establish that the old memory is safe to reuse.

## Code Organization and Extension Points

```text
pypto_serving/
├── router/                   # Public API, node directory, routing and recovery
├── serving/pd/               # Configuration, handoff protocol and lifecycle
│   ├── integration.py        # Optional application/worker composition
│   ├── adapter.py            # Model adapter interface and registry
│   ├── contracts.py          # Model and runtime layout contracts
│   ├── service.py            # Prefill/Decode control flow
│   ├── worker.py             # Owner registration, transfer jobs and cleanup
│   ├── http_api.py           # Node control contracts and route installation
│   ├── handoff.py            # Node-local handoff records
│   ├── connector.py          # Decode reservation and commit
│   ├── planner.py            # Chunk/suffix transfer plans
│   └── observability.py      # Startup logs and metrics
├── transfer/                 # Backend contract, Mooncake and owner-local memory
├── serving/memory/
│   └── reservation.py        # Allocator-owned external-fill cache leases
└── model/deepseek_dspark/
    └── pd_adapter.py         # Eight-region layout and K7 continuation semantics
```

Ordinary serving does not import or instantiate PD components. The CLI loads
the optional composition root only when a P/D role is selected. That root
binds public request preparation, scheduling and worker capabilities; it
does not replace the normal engine loop. Both local and remotely prefilled
requests use the same output, parser, stop and cancellation paths.

The worker's generic ordered control envelope carries opaque bytes. PD
operation decoding and transfer state live in `serving/pd/worker.py`; the
model adapter supplies resident-cache geometry and Decode-local state
initialization through explicit callbacks. The regular runner owns model
computation and device allocations. Cache reservations use the existing
allocator and are created only when requested; protocol manifests never
enter the allocator.

For another model, implement and register a model adapter with its cache
components, legal transfer boundaries, worker factory and Decode continuation.
Register it through the lazy model adapter factory in `model/__init__.py`. Keep physical
cache matching and publication in `KvCacheManager`. For another transfer
backend, implement the provider completion contract and owner-local
registration, then connect it to provider selection. Placement policies
implement `RoutePolicy` in `router/policy.py`; they do not manage physical
cache pages.

Regression tests mirror these responsibilities under `tests/unit/router/`,
`tests/unit/serving/pd/`, and `tests/unit/transfer/`. See
[PD integration checks](../../tests/integration/pd/README.md) for checks
against running services.
