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
| `run_id` | Derived from the endpoints | Share an explicit deployment identity when needed. |
| `enable_chunk_overlap` | `true` | Overlap closed-page transfer with later Prefill work. |

An endpoint can also specify `node_id`, `control_host`, `control_port` and
`transfer_hostname`. The default control port is 29831. When the device RoCE
address differs from the HTTP hostname, set `transfer_hostname` to the address
required by Mooncake on that node.

Prefix caching is enabled by default and follows each model node's ordinary
`--enable-prefix-caching` / `--no-enable-prefix-caching` option. It is not a
shared JSON setting: all four P/D combinations are valid. P hits reduce local
computation; D hits independently reduce transferred bytes. Final request
state is still transferred even when the KV prefix is a full hit.

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
  --block-size 32 --max-model-len 655360
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

Each model process uses its own ordinary prefix-caching option; no shared
cache-mode field is needed. The example context capacity includes a 512k-token
prompt and 128k-token generation budget; it is not a maximum-length validation
claim. To run ordinary combined Prefill/Decode serving, omit the PD options
and use the normal model command.

## Send Requests and Inspect State

Clients use the Router's OpenAI-compatible endpoints:

```bash
curl http://router.example.internal:8110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-flash-dspark-w8a8","messages":[{"role":"user","content":"Describe the Forbidden City."}],"max_tokens":128,"temperature":0.0,"stream":true}'
```

`/v1/completions` is also supported. Router sends the original request to
both nodes in parallel through the same completion/chat endpoints. It adds
one typed `pd_transfer` context containing the request/attempt identity and
assigned peers. Clients do not supply this field. A PD node rejects requests
without it; it does not silently run combined Prefill/Decode.

P uses Serving's normal preparation, tokenization and constraint preflight
once. Normalized tokens, parser/constraint specifications and prefix-match
metadata travel to D over the existing node control session. D pairs them
with its HTTP request, reserves cache and returns writable placement to P.
Physical blocks, reservations and native memory addresses never pass through
Router. The original output budget is preserved; P does not rewrite
`max_tokens` to one. D does not template or tokenize the request again.

For DeepSeek V4, the Router accepts `tools`, `tool_choice`,
`parallel_tool_calls`, and assistant/tool history. Structured tool-call deltas
and final calls travel from D to the Router alongside text and reasoning.
Strict `auto`, `required`, and a named function choice are also supported.
These requests require matching Serving/Lib grammar ABIs and XGrammar on both
model nodes. Ordinary non-strict `auto` remains unconstrained.
P prepares each generation attempt once, including constraint preflight.
D returns HTTP headers only after accepting that normalized continuation and
reserving its cache. Router can then establish SSE without waiting for P's
source-pin cleanup, and supervises P's confirmation alongside D's output.
Preparation failures remain HTTP errors. Execution or parser failures after
headers are reported in the stream. Handoff admission and terminal publication
still require P's matching READY result; a client disconnect is not a native
retirement proof.

Protocol version 12 has device coverage for completion/chat, constrained tools,
cancellation and cold/hot reuse with all four independent P/D prefix-cache
settings. The parallel source/output boundary is covered by cancellation after
the first public output, concurrent requests and recovery of normal admission.
Rank/component completion and commit ordering are audited separately from
public responses; these checks are not device-byte checksums or a complete
native fault-injection matrix.
This is not a claim of identical token sequences between repeated inferences
or validation of every topology and maximum length. Router and
nodes must use the same protocol; older nodes are explicitly rejected.
The Router finalizes handoff state and cleanup before publishing the terminal
output; intermediate outputs continue to stream immediately.

Decode uses the same JSON/SSE formatter as ordinary Serving. Router forwards
that public response without a second model parser or response formatter.
SSE frames carry content/reasoning/tool-call deltas followed by finish, usage
and `[DONE]`; they do not carry cumulative text or token histories.
Non-streaming JSON contains the full answer and preserves the optional token
digest header. Completion deltas include the final detokenization flush;
already-published text cannot be rewritten. There is no private NDJSON path
or compatibility mode for the previous five-stage HTTP generation protocol.
Execution and connection errors after SSE headers produce an explicit error
event followed by `[DONE]`; client cancellation still propagates to cleanup
and is not converted into a successful terminal response.

| Endpoint | Process | Contents |
| --- | --- | --- |
| `/health` | Router, P, D | Readiness and recovery gate. |
| `/v1/models` | Router, P, D | Served model identity. |
| `/internal/pd/descriptor` | P, D | Node identity, protocol and model/provider contract. |
| `/metrics` | Router | Request outcomes and routing metrics. |
| `/recovery` | Router | Recovery state and generation. |
| `/internal/pd/metrics` | P, D | Prefix hits, transfer progress and request cleanup. |
| `/internal/pd/capacity` | P, D | Active requests, reservations and quarantined allocations. |
| `/internal/pd/query`, `/internal/pd/abort` | P, D | Correlated status and safe cancellation; not generation stages. |

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
Active handoffs share one bounded completion poller per worker; a poll batches
their job IDs without waiting for slower jobs to finish. Completion observation uses
an independent, bounded local pipe, so queries do
not fence or queue behind Prefill computation. Submission, source readiness
and resource retirement remain on the ordered worker lane. The progress reader
only publishes terminal facts; it never accesses device tensors or releases
source pins or destination reservations. It is created only for PD Prefill and
closes with that worker; ordinary serving creates no PD channel or thread.
Owner-local transfer IPC uses
versioned MsgPack tasks with a per-task region table. Segments reference that
table while retaining the exact owner generation, lease and byte bounds.
Each chunk carries one complete bundle of rank/component completion facts,
including required zero-byte units. D validates the whole bundle, persists
it with one journal sync, and then acknowledges it; commit still requires all
completion units and atomic groups.
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

## Admission and Shutdown

The Router bounds total client streams separately from Prefill/transfer
handoffs. A matching P READY reply releases the handoff slot after P has
retired its source pins; the client-stream slot remains held through Decode
and cancellation cleanup. D reservation remains the authority for actual KV
capacity. This allows bounded preparation of more P work while D generates,
without treating a completed transfer as permission to free D's allocation.

For normal shutdown, stop accepting Router requests and allow active streams
to drain, then stop both model-node processes. Either node can initiate peer
drain, and simultaneous node shutdown is supported by the control protocol:
both peers stop admission and drain, P unregisters its transfer memory, D
unregisters after P's release notification, and D acknowledges completion.
Both native engines remain alive until this exchange completes; only then
may their owner processes and compute runtime stop.

Lost peer connections are not proof of safe release. A timeout or unregister
failure reports incomplete retirement with the retained owner identity and
stage; it must be handled by the external supervisor. The PD application does
not continue into compute-engine shutdown after such a failure. Do not treat
an HTTP process exiting as proof that its device owners were released.

Router and both nodes must be upgraded together. The control protocol and
owner IPC reject incompatible versions rather than using an old-wire fallback.

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
│   ├── http_api.py           # Standard generation routes and operational API
│   ├── generation.py         # Typed context and original-request binding
│   ├── protocol.py           # Transport-neutral identity/normalized handoff
│   ├── handoff.py            # Node-local handoff records
│   ├── connector.py          # Decode reservation and commit
│   ├── planner.py            # Chunk/suffix transfer plans
│   └── observability.py      # Startup logs and metrics
├── transfer/                 # Backend contract, Mooncake and owner-local memory
├── serving/server/
│   └── responses.py          # Shared normal token-output JSON/SSE formatter
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
