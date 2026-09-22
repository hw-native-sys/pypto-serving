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
| `enable_chunk_overlap` | `false` | Allow closed-page transfer to overlap later Prefill work. |

An endpoint can also specify `node_id`, `control_host`, `control_port` and
`transfer_hostname`. The default control port is 29831. When the device RoCE
address differs from the HTTP hostname, set `transfer_hostname` to the address
required by Mooncake on that node.

To configure multiple nodes, add endpoints to either array, give each one a
unique `node_id`, and pass `--pd-node-id` to its model process. The Router
checks their runtime contracts and selects a compatible pair using the
configured policy. Each model/service deployment uses its own Router and
configuration.

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
│   ├── adapter.py            # Model adapter interface and registry
│   ├── contracts.py          # Model and runtime layout contracts
│   ├── service.py            # Prefill/Decode control flow
│   ├── handoff.py            # Node-local handoff records
│   ├── connector.py          # Decode reservation and commit
│   ├── planner.py            # Chunk/suffix transfer plans
│   └── observability.py      # Startup logs and metrics
├── transfer/                 # Backend contract, Mooncake and owner-local memory
└── model/deepseek_dspark/
    └── pd_adapter.py         # Eight-region layout and K7 continuation semantics
```

For another model, implement and register a model adapter with its cache
components, legal transfer boundaries and Decode continuation. Keep physical
cache matching and publication in `KvCacheManager`. For another transfer
backend, implement the provider completion contract and owner-local
registration, then connect it to provider selection. Placement policies
implement `RoutePolicy` in `router/policy.py`; they do not manage physical
cache pages.

Regression tests mirror these responsibilities under `tests/unit/router/`,
`tests/unit/serving/pd/`, and `tests/unit/transfer/`. See
[PD integration checks](../../tests/integration/pd/README.md) for checks
against running services.
