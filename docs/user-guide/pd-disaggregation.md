# Prefill/decode disaggregation

PD is opt-in. Without `--pd-role`, Serving does not load PD or Mooncake, create transfer services, or read the PD configuration. Prefix caching remains enabled by default in both modes.

## Requirements

The initial adapter supports DeepSeek V4 Flash DSpark K7, greedy sampling, identical model weights and compatible cache formats on P and D. Use Serving's pinned PyPTO-Lib, which includes the independent-head support from [Lib PR #1542](https://github.com/hw-native-sys/pypto-lib/pull/1542).

Use matching PyPTO/Simpler source builds with owner-local service and allocation-retain hooks; these hooks are not upstream APIs yet. PyPTO must expose `chip_service_factories`, `retain_service_tensor`, and `release_service_tensor`, backed by Simpler's owner-local allocation pinning. Its `RunConfig` must support explicit `backend_type` and `execution_mode` fields. These are required interfaces, not a fallback to an older runtime.

Control protocol version 2 separates the common replicated-page data contract from each node's local rank groups. All participants must use the same protocol version; older schemas are rejected. Every destination cache replica selects a complete source replica from the request's Prefill partition. Page/ring formats must remain compatible: this is replica mapping, not a general sharded-layout conversion.

Follow the SDK's normal source-build instructions with its matching runtime pin and required PTOAS version. The tested integration uses PTOAS 0.67. Keep runtime revision checks enabled; a binary copied from another checkout is not a matching source installation.

Model nodes require Linux Python with `os.pidfd_open` and a working Mooncake Ascend installation. Startup checks these dependencies before loading weights. Mooncake registers all regions inside their real allocation owners before the first transfer; adding regions after a peer connection is established is not supported. Transfers are device-to-device without Host staging. Deploy on a trusted network: internal control endpoints are not authenticated.

Install `python -m pip install -e '.[pd]'` in each environment. The Router only needs this extra, not PyPTO, model weights, or NPUs. Optional constrained tools use the ordinary [XGrammar deployment requirements](../models/deepseek-v4/tool-constraints.md).

## Configure and start

Copy [the minimal configuration](https://github.com/hw-native-sys/pypto-serving/blob/main/examples/pd.json), replace the example IPs, and use the same configuration on all processes. Each service/model has its own Router and node pools. Add replica entries with either role for multiple nodes; the policy defaults to round-robin over compatible healthy pairs. With multiple entries in one role, select the node using `--pd-node-id host:port` (or an explicit entry `name`).

On **each** model host, start one process with its appropriate role and port:

```bash
pypto-serving \
  --model /path/to/dsv4-flash-dspark-w8a8 --served-model-name dsv4 \
  --backend npu --platform a2a3 \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --dp 4 --ep 16 --tp 4 --block-size 32 --max-num-seqs 8 \
  --max-model-len 655360 --max-num-batched-tokens 8192 \
  --long-prefill-token-threshold 128 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":7}' \
  --ring-heap 1073741824,1073741824,1073741824,1073741824 \
  --generate-config '{"max_new_tokens":131072,"temperature":0}' \
  --pd-config examples/pd.json --pd-role prefill --host 0.0.0.0 --port 8000
```

Use `--pd-role decode` on D; both launchers specify K7, while P internally runs target-only prefill. The context limit covers **prompt plus output**: 655360 allows a 524288-token prompt plus 131072 generated tokens, subject to actual device memory. Configuring this limit is not a full-length validation.

Each node chooses its own attention `--tp`, with `--dp = --ep / --tp`.
The vocabulary head uses independent `--lm-head-tp` (default 4, retaining NZ).
For example, keep P at `--dp 4 --tp 4 --ep 16` and set D to
`--dp 16 --tp 1 --ep 16 --lm-head-tp 4`; no PD JSON mode changes are needed.
Without PD, Prefill and Decode share the same local topology.
Smaller attention TP does not multiply batch capacity: every rank still owns
16 decode request slots, distinct from the padded 64-request group ABI.

Real-weight K7 validation covers ordinary DP4TP4, homogeneous PD DP4TP4,
and asymmetric P DP4TP4 to D DP16TP1 with head TP4 and prefix caching enabled.
It includes cache/state replay, rolling boundaries, cancellation and both
streaming and non-streaming APIs. Repeat compilation and hardware validation
after changes to the SDK or the model kernels; these results do not validate
other parallel layouts or full-length context stress.

Each node independently accepts `--no-enable-prefix-caching`; all four P/D combinations are supported. No prefix mode belongs in JSON. P cache hits reduce computation; D hits reduce network traffic, independently.

The common Serving Router uses `role: "prefill"` and `role: "decode"` in its `replicas` table. Entries without a role remain ordinary complete replicas. PD requires both roles and uses statically launched nodes; mixing complete replicas, PD nodes or launchable `hosts` is rejected. The Router never starts or stops these external P/D processes.

Start the external Router, then wait for `/health` to report `ok`:

```bash
pypto-serving-router --replicas examples/pd.json --host 0.0.0.0 --port 8110
curl http://127.0.0.1:8110/health
curl -N http://127.0.0.1:8110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Tell me about the Forbidden City."}],"temperature":0,"max_tokens":128,"stream":true}'
```

Completion/chat, streaming, tools and optional constraints reuse Serving's ordinary API. Invalid preparation fails before SSE headers. Later execution failures emit an explicit stream error; clients must inspect events even after HTTP 200.

HTTP, control (HTTP port + 10000 by default), and Mooncake's dynamic RPC/RoCE endpoints must be reachable. Optional entry fields are `control_port` and `transfer_hostname` for a distinct data-plane address. Optional fields in the top-level `pd` section are `provider` (currently `mooncake`), `policy` (currently `round_robin`), `capacity` (8 per node), `handoff_timeout` (300 seconds), `request_timeout` (3600 seconds), and `run_id`. Timeouts do not authorize reuse of possibly active memory.

`handoff_timeout` bounds individual control, transfer, and stop-confirmation operations. Waiting for the full Prefill before Decode uses `request_timeout`, so a long, progressing Prefill is not mistaken for a stalled transfer. Non-streaming JSON response headers also wait under `request_timeout`, because they follow generation; streaming setup headers retain `handoff_timeout`. Expiry still follows the normal abort and lease-retirement rules.

Logs default to `serving_log/pd_disag/<automatic-run-id>/{router,p,d}/`; node files have stable hashed names. Set a shared `pd.run_id` to group processes under one tag. Boot identities are still regenerated at every startup. Optional `observability` fields are `enabled` and `root`. Detailed profiling uses Serving's existing profiler, not a second trace system.

### Optional Decode Step Metrics

Set `PYPTO_SERVING_DECODE_METRICS_DIR` before starting each model server to collect lightweight worker JSONL records. Use a fresh directory per server run. This is disabled by default and does not require profiling.

Each record contains one decode batch's Host dispatch-to-result-consumption duration and per-request raw accepted-token counts, including any bonus token and before scheduler output-length clipping. Compute mean acceptance length as total accepted tokens divided by total request-verifies; count batch duration once, not once per request. The duration includes asynchronous completion waits and is not device kernel time.

Prepared synchronous and asynchronous paths exclude explicit input preparation. The `sync_unprepared` path may include preparation inside the executor; analyze it separately. Records also identify commands that schedule both prefill and decode. Compare like-for-like paths and batch sizes, and enable the same collection settings on all deployments being compared. The sink records no prompt or tensor data.

## Execution, extension and shutdown

The common Router refreshes PD descriptors through its existing background health monitor. It becomes ready only when a compatible healthy P/D pair exists. Ordinary session-affine routing is unchanged; PD retains round-robin pair selection without session pinning. The PD coordinator fixes a pair per request and sends two ordinary generation POSTs concurrently. P prepares input once; D reserves shared read-only prefix and writable suffix/state. After each confirmed prefill chunk, P pins source pages and submits a rank-mapped transfer. Owner-side communication and an independent progress channel overlap subsequent computation. D validates all required regions, commits through CacheManager, accepts the grammar seed, and adopts the request into ordinary K7 decode. Final request-local state is transferred even on a complete D prefix hit; output does not wait for P's source cleanup.

DSpark sizes each cache group from the tightest post-resident rank budget using
`--npu-memory-utilization` (default `0.90`), including filler/alignment, live rings,
in-flight replacement pages and bounded checkpoint retention. Ring periods and
kernel table limits do not change. Maximum context is a per-request limit, not
a promise that every concurrent request can fill it. D reserves only received
prompt pages and final state; its logical output limit does not preallocate future
Decode pages. Ordinary scheduling grows those tables per step. With no in-flight
work able to free capacity, an exhausted external request fails explicitly; it
never falls back to local Prefill recomputation.

P also captures the last 128 prompt tokens' target hidden states, without loading or running the drafter. Target-only P skips capture dispatches outside that window; a packed batch keeps the existing all-rank collective if any request needs capture. Ordinary K7 prefill is unchanged. The final asynchronous transfer includes this request-local bootstrap region, even on a complete D KV hit. D reconstructs absolute positions from the agreed prompt length and uses its own lease generation, then reuses ordinary terminal-prefill bootstrap before its first verification. This initializes D's local drafter with prompt history rather than an empty window. The fixed 128-row hidden region is 3 MiB per destination rank: 12 MiB for a TP4 cache group or 3 MiB for TP1, including short prompts. Bootstrap runs once at admission; steady-state decode keeps its existing asynchronous pipeline. The optional profiler span `pd.dspark.bootstrap` isolates this initialization cost.

`pd/adapter.py` owns model-specific regions, prefix rules, mapping and decode bootstrap; register another executor in `ADAPTERS`. `pd/transfer.py` defines the owner-side provider contract; register another implementation in `config.PROVIDERS`. `router/routing.py` provides node pools and routing policies; `pd/coordinator.py` filters compatible pairs and coordinates their execution inside the common Router. The Router never manages physical cache; providers never implement model semantics. New adapters/providers require their own layout, lifecycle and native validation; changing JSON alone cannot add them.

Terminal output waits for request retirement, including the local state-release acknowledgement, before the Router can admit a replacement. Normal Decode completion reuses Serving's FIFO worker-release ACK without an extra control barrier. Initial-token termination and cancellation without that ACK still explicitly discard reserved state. HTTP cancellation does not cancel the owned retirement task. A confirmed abort for a request that was never admitted consumes no capacity and authorizes no writes.

To shut down normally, stop admitting requests at the Router, let active requests finish, then stop P before D. Confirm sender registrations and pins have retired before freeing receiver memory. Deterministic errors reclaim normally; uncertain native completion gates the affected node and retains its writable suffix/state and bootstrap slot, not healthy shared prefix. Recovery is manual/external: no automatic restart, transfer replay, or backend fallback. Before write authorization, a cached placement without receive-suffix capacity can be released and retried once through normal cold allocation; this is not a transfer retry. Do not equate a closed HTTP connection with stopped DMA.
