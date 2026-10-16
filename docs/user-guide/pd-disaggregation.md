# Prefill/decode disaggregation

PD is opt-in. Without `--pd-role`, Serving does not load PD or Mooncake, create transfer services, or read the PD configuration. Prefix caching remains enabled by default in both modes.

## Requirements

The initial adapter supports DeepSeek V4 Flash DSpark K7, greedy sampling, identical model weights and compatible cache layouts/topologies on P and D. Use Serving's pinned PyPTO-Lib. The tested SDK commits are PyPTO `e9c2239ec6d350a676343bae308f1ee70addcdde` and its Simpler pin `196810112eec6741923d1cb51d670541afa017a7`, with PTOAS 0.65. PyPTO must expose `chip_service_factories`, `retain_service_tensor`, and `release_service_tensor`, backed by Simpler's owner-local allocation pinning. These are required runtime interfaces, not a fallback to an older runtime.

Model nodes require Linux Python with `os.pidfd_open` and a working Mooncake Ascend installation. Startup checks these dependencies before loading weights. Mooncake registers all regions inside their real allocation owners before the first transfer; adding regions after a peer connection is established is not supported. Transfers are device-to-device without Host staging. Deploy on a trusted network: internal control endpoints are not authenticated.

Install `python -m pip install -e '.[pd]'` in each environment. The Router only needs this extra, not PyPTO, model weights, or NPUs. Optional constrained tools use the ordinary [XGrammar deployment requirements](deepseek-v4-dspark-tool-constraints.md).

## Configure and start

Copy [the minimal configuration](../../examples/pd.json), replace the example IPs, and use the same configuration on all processes. Each service/model has its own Router and node pools. Add entries to either list for multiple nodes; the policy defaults to round-robin over compatible healthy pairs. With multiple entries in one role, select the node using `--pd-node-id host:port` (or an explicit entry `node_id`).

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
  --ring-heap 2147483648,2147483648,4294967296,8589934592 \
  --generate-config '{"max_new_tokens":131072,"temperature":0}' \
  --pd-config examples/pd.json --pd-role prefill --host 0.0.0.0 --port 8000
```

Use `--pd-role decode` on D; both launchers specify K7, while P internally runs target-only prefill. The context limit covers **prompt plus output**: 655360 allows a 524288-token prompt plus 131072 generated tokens, subject to actual device memory. Configuring this limit is not a full-length validation.

Each node independently accepts `--no-enable-prefix-caching`; all four P/D combinations are supported. No prefix mode belongs in JSON. P cache hits reduce computation; D hits reduce network traffic, independently.

Start the external Router, then wait for `/health` to report `ok`:

```bash
pypto-pd-router --pd-config examples/pd.json --host 0.0.0.0 --port 8110
curl http://127.0.0.1:8110/health
curl -N http://127.0.0.1:8110/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Tell me about the Forbidden City."}],"temperature":0,"max_tokens":128,"stream":true}'
```

Completion/chat, streaming, tools and optional constraints reuse Serving's ordinary API. Invalid preparation fails before SSE headers. Later execution failures emit an explicit stream error; clients must inspect events even after HTTP 200.

HTTP, control (HTTP port + 10000 by default), and Mooncake's dynamic RPC/RoCE endpoints must be reachable. Optional entry fields are `control_port` and `transfer_hostname` for a distinct data-plane address. Optional runtime fields are `provider` (currently `mooncake`), `policy` (currently `round_robin`), `capacity` (8 per node), `handoff_timeout` (300 seconds), `request_timeout` (3600 seconds), and `run_id`. Timeouts do not authorize reuse of possibly active memory.

Logs default to `serving_log/pd_disag/<automatic-run-id>/{router,p,d}/`; node files have stable hashed names. Set a shared `runtime.run_id` to group processes under one tag. Boot identities are still regenerated at every startup. Optional `observability` fields are `enabled` and `root`. Detailed profiling uses Serving's existing profiler, not a second trace system.

## Execution, extension and shutdown

The Router fixes a P/D pair per request and sends two ordinary generation POSTs concurrently. P prepares input once; D reserves shared read-only prefix and writable suffix/state. After each confirmed prefill chunk, P pins source pages and submits a rank-mapped transfer. Owner-side communication and an independent progress channel overlap subsequent computation. D validates all required regions, commits through CacheManager, accepts the grammar seed, and adopts the request into ordinary K7 decode. Final request-local state is transferred even on a complete D prefix hit; output does not wait for P's source cleanup.

`pd/adapter.py` owns model-specific regions, prefix rules, mapping and decode bootstrap; register another executor in `ADAPTERS`. `pd/transfer.py` defines the owner-side provider contract; register another implementation in `config.PROVIDERS`. `pd/routing.py` owns pair selection. The Router never manages physical cache; providers never implement model semantics. New adapters/providers require their own layout, lifecycle and native validation; changing JSON alone cannot add them.

To shut down normally, stop admitting requests at the Router, let active requests finish, then stop P before D. Confirm sender registrations and pins have retired before freeing receiver memory. Deterministic errors reclaim normally; uncertain native completion gates the affected node and quarantines writable suffix/state, not healthy shared prefix. Recovery is manual/external: no automatic restart, replay, retry, or fallback. Do not equate a closed HTTP connection with stopped DMA.
