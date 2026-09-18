# Multi-Node Routing

`pypto-serving-router` spreads OpenAI-compatible requests across several independent serving replicas, keeping each conversation on the replica that already holds its KV cache.

A replica is one `pypto-serving` process owning one engine core. Replicas are addressed by URL, so a replica on the router's own machine and a replica on another host are treated identically.

## Why a Router

A DeepSeek V4 replica occupies all eight devices of a node, so a second replica needs a second node. Scaling out is the only way to add DeepSeek capacity, and it needs no change to the model, the parallel placement, or the runtime: the replicas do not talk to each other.

For Qwen, where a replica is one card, the router is throughput scaling across as many cards or hosts as are available.

## Session Affinity

The prefix cache is local to a replica. Turn N of a conversation publishes its prompt blocks after the worker confirms them, and turn N's answer enters the cache as part of turn N+1's prompt. A turn that lands on the same replica re-prefills only the new text; a turn that lands elsewhere pays a full prefill.

The router therefore pins a conversation to a replica:

- A request with no session id starts a new session; the router picks the least loaded replica and mints an id.
- Every response carries the id in an `X-Session-Id` header, on streaming and non-streaming responses alike.
- Sending that id back on the next turn routes it to the same replica.

The id may be supplied either as an `X-Session-Id` request header or as a top-level `session_id` field in the JSON body. The header wins when both are present.

Affinity is a preference, not a guarantee. Cached blocks are evicted under memory pressure, and a pinned replica that falls too far behind loses the session to a less loaded one — see `--affinity-slack`. Pins expire after `--session-ttl` of inactivity; expiry releases the pin only, since KV blocks are keyed by content and reclaimed by each replica's own cache.

Session affinity buys nothing when prefix caching is off, which is the case for DeepSeek V4 with more than one speculative token.

## Start the Replicas

Start one process per replica, each with its own devices and port. For Qwen, one card per replica:

```bash
pypto-serving --model /path/to/Qwen3-14B --devices 0 --port 8001 &
pypto-serving --model /path/to/Qwen3-14B --devices 1 --port 8002 &
```

For DeepSeek V4, one eight-card replica per node, identical commands on each host:

```bash
pypto-serving \
  --model /path/to/dsv4-flash-w8a8 \
  --devices 0,1,2,3,4,5,6,7 \
  --dp 8 --ep 8 --tp 1 \
  --block-size 128 \
  --port 8001
```

Give each replica process its own `PYPTO_PROG_BUILD_DIR`. It is both the PyPTO build directory and the kernel compile cache, and replicas that share a networked home directory will otherwise collide on it, costing a cold compile on startup.

## Start the Router

Describe the replicas in a JSON file:

```json
{
  "replicas": [
    {"name": "node0", "host": "127.0.0.1", "port": 8001},
    {"name": "node1", "host": "10.0.0.2", "port": 8001}
  ]
}
```

`name` is optional and defaults to `host:port`.

```bash
pypto-serving-router --replicas replicas.json --port 8000
```

The router needs no NPU, no model, and no tokenizer: it forwards request bodies verbatim and never tokenizes. It can run on its own host, or alongside a replica on a host that also serves.

## Use It

The first turn arrives without a session id:

```bash
curl -i http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-14B", "messages": [{"role": "user", "content": "Hello"}]}'
```

Read `X-Session-Id` from the response headers, then send it on every later turn of the same conversation:

```bash
curl -i http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-Id: 9f2c1b7e4a3d5061b8c2e7f04a1d9b63' \
  -d '{"model": "Qwen3-14B", "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "Say more"}]}'
```

As with any OpenAI-compatible server, the client still sends the whole transcript each turn. The session id decides only which replica receives it.

## Readiness and Failover

The router polls each replica's `/health` every `--health-interval` seconds. A replica is taken out of rotation after two consecutive failures and returns on the first success. Sessions pinned to an unroutable replica fall back to a live one and re-pin there.

`/health` on a replica reports serving readiness, not just that the process is answering: it returns 503 when the worker process has exited or the engine loop has stopped scheduling. A replica still loading its model refuses the connection outright, because the server binds its socket only after the engine has started.

The router's own `/health` reports the state of the whole table:

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "ok",
  "replicas_ready": 2,
  "replicas_total": 2,
  "sessions": 14,
  "routed": 512,
  "affinity_hits": 498,
  "rejected": 0,
  "replicas": [
    {"name": "node0", "url": "http://127.0.0.1:8001", "ready": true, "outstanding": 3}
  ]
}
```

`affinity_hits` against `routed` is the cheapest check that pinning is working.

## Tuning

| Flag | Default | Effect |
| --- | --- | --- |
| `--session-ttl` | `600` | Seconds of inactivity before a conversation loses its pin. |
| `--affinity-slack` | `8` | Extra outstanding requests a pinned replica may carry before a session is re-routed. `0` disables affinity; a large value makes it strict. |
| `--health-interval` | `5` | Seconds between replica health probes. |
| `--request-timeout` | `3600` | Overall timeout for one proxied request. |
| `--connect-timeout` | `5` | Connect timeout per replica. |

Raise `--affinity-slack` when prefills are expensive relative to queueing — long transcripts, few replicas. Lower it when replicas are easily saturated and a wait costs more than a re-prefill.

## Limits

- The router does not retry. A replica that fails mid-stream ends that request; the client resends.
- Load is counted as outstanding requests, not tokens, because the router never tokenizes.
- The replica list is fixed at launch. Adding a replica means restarting the router.
