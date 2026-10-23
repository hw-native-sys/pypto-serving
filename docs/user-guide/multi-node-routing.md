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

## Two Ways to Get Replicas

Replicas can already exist, or the router can start them.

- **`replicas`** in the config are endpoints that are already running. The router routes to them and never stops them.
- **`hosts`** declare machines the router may launch on — one slot per declared device. A replica launched from a slot belongs to the router, which stops it on an orderly shutdown.

Both can appear in the same file. The sections below cover starting replicas by hand; [Launching Replicas](#launching-replicas) covers letting the router do it.

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

`name` is optional and defaults to `host:port`. `scheme` is optional and defaults to `http`; set it to `https` when the replicas are not on a trusted network, since request bodies and generated text cross this hop in the clear otherwise.

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

## Launching Replicas

Declare the machines and the devices the router may use:

```json
{
  "hosts": [
    {
      "name": "node01",
      "devices": [4],
      "port_base": 8001,
      "model": "/models/Qwen3-14B",
      "served_model_name": "qwen",
      "serve_args": ["--max-num-seqs", "16", "--block-size", "128", "--max-model-len", "1024"],
      "env": {"PYPTO_PROG_BUILD_DIR": "/scratch/build_d{device}"},
      "log_dir": "/scratch/router-logs"
    },
    {
      "name": "node02",
      "ssh": "user@node02",
      "identity_file": "~/.ssh/id_ed25519",
      "devices": [0, 1, 2, 3],
      "port_base": 8001,
      "model": "/weights/Qwen3-14B",
      "served_model_name": "qwen",
      "launch_wrapper": ["task-submit", "--device", "{device}", "--max-time", "0", "--run"]
    }
  ]
}
```

A host with no `ssh` is the machine the router is on. `identity_file` is a path — key material never goes in the config. `{device}`, `{devices}` and `{port}` expand per slot.

`launch_wrapper` is optional. Present, the serving command is handed to it as a single argument, which is what a broker such as `task-submit` expects. Absent, the command runs directly — deployments without a broker are exactly why this is configuration rather than something built in.

Ports are derived as `port_base + device`, so two replicas on one host can never collide.

### How many replicas fit on a host

`devices` says which cards the router may use; `devices_per_replica` says how many of them one replica needs. The host's capacity is the quotient:

```text
capacity = len(devices) // devices_per_replica
```

`devices_per_replica` is a property of the model and its kernels, not something the router can measure:

| Model | Devices per replica | Enforced by |
| --- | --- | --- |
| Qwen3-14B | 1, or the `--tp` group size | `ParallelConfig.worker_group_size` |
| DeepSeek V4 | exactly 8, with `--dp 8 --ep 8 --tp 1` | serving refuses any other count at startup |
| DeepSeek V4 DSpark | exactly 16 | `DSPARK_RANKS` |

For DeepSeek those counts are compiled into the kernels — the expert-parallel width is not a tuning knob — so an eight-card node holds **eight** Qwen replicas or **one** DeepSeek V4:

```json
{
  "name": "node01",
  "devices": [0, 1, 2, 3, 4, 5, 6, 7],
  "devices_per_replica": 8,
  "model": "/models/dsv4-flash-w8a8",
  "serve_args": ["--dp", "8", "--ep", "8", "--tp", "1", "--block-size", "128"]
}
```

It is declared rather than inferred because deriving it means reading a checkpoint on a machine the router has no reason to be able to see. A wrong value is caught on the replica at startup, where the model's topology is validated and the error names the count it wanted. A `devices` list that is not a whole multiple of `devices_per_replica` is refused when the config is read — a leftover card is a typo or a misread model, not a partial replica.

Memory is not the binding constraint: KV sizing claims about 90% of whatever is free once the weights are loaded, so the first replica on a card effectively takes it.

### Capacity

The declared devices are the ceiling. The Qwen config above has five slots: one on `node01`, four on `node02`.

```bash
pypto-serving-router --replicas fleet.json --port 8000 --initial-replicas 1
```

`--initial-replicas` fills slots in config order, local machine first, so `3` here starts one replica on `node01` and spills two onto `node02`. It defaults to 1 when the file declares hosts, and launches concurrently — three replicas cost one model load's wall clock, not three.

`--max-replicas` lowers the ceiling below the declared pool. On a shared cluster that is what stops the router taking every free card.

### Growing and shrinking at runtime

```bash
curl -X POST http://localhost:8000/replicas        # 202, or 409 when the pool is full
curl -X DELETE http://localhost:8000/replicas/node02-d1
curl -s http://localhost:8000/replicas             # the fleet, and what is left
```

`POST` answers `202`, not `200`: the replica is starting, and cannot serve for the minutes its model takes to load. It is registered immediately but not routable, and the health poller promotes it when it answers. A replica that never becomes ready within `--launch-timeout` (default 600s) is stopped and its device released, so a failed launch cannot hold a card for the life of the router.

**Set `--launch-timeout` for the slowest model you actually run.** Qwen3-14B reports ready in about 130s with a warm kernel cache; DeepSeek V4 on eight cards took 370s in a measured run on this stack, comfortably inside the 600s default. That figure is not fixed, though — an earlier measurement of DeepSeek on a different checkpoint and an older stack ran to 1198s. Time your own launch and leave headroom, because the cost of setting it too low is the router stopping a replica that was loading normally.

`DELETE` drains first — the replica stops taking new sessions, its pinned sessions are released so they re-route on their next turn, and only once its in-flight requests finish (or `--drain-timeout`, default 300s, expires) is it stopped.

A slot only returns to the pool once the stop is confirmed: the stop command has to succeed *and* the replica's address has to go silent within the grace period. *Any* HTTP answer counts as still there, `503` included — serving answers `/health` with `503` whenever its engine is not ready, so a replica whose worker died with its HTTP server still up is alive and still holding its cards. If either check fails, the replica is out of rotation but its devices are in an unknown state — silence does not prove the serving process and its device workers have exited either — so the slot stays reserved, `DELETE` answers `500`, and the slot is listed under `capacity.unconfirmed` with the reason `stop-unconfirmed`. Freeing it is a manual decision: the alternative is launching onto cards another instance may still hold.

### When a replica dies

A replica that was serving and then stops answering is retired and replaced on the same device group, with no operator involved. Nothing used to do this: the health poller took it out of rotation and there it stayed, unroutable, holding its cards, reported exactly like a replica still loading its model.

The sequence, with the defaults: one or two failed probes take it out of rotation within a few seconds, so traffic stops reaching it almost at once; `--replace-after` (30s, floored at two health intervals -- so with `--health-interval` above 15s that floor is what applies) then decides that unroutable means dead rather than briefly unwell; the replica is drained and stopped like any other, and only once that stop is **confirmed** is another launched on its slot. A stop that cannot be confirmed is never launched onto — the slot stays reserved under `capacity.unconfirmed`, because putting a second process on cards the first may still hold is worse than running short.

A slot that keeps losing replicas is given up on: three in a row, with the wait between attempts doubling from 10s, and then the slot is left free and listed under `capacity.abandoned`. The router will not use it again by itself; `POST /replicas` or a restart will, once whatever is wrong with it is fixed.

`/replicas` and `/health` carry a `phase` per replica — `starting`, `ready`, `draining` or `failed` — and `/health` counts the failed ones in `replicas_failed`. Before this, a dead replica and one three minutes into a model load were the same JSON.

Measured on two cards with Qwen3-14B at the default settings: a replica killed outright was out of rotation in 7s, the router began replacing it 52s after the death, and the fleet was back to full strength 199s after it, with the surviving replica serving both users throughout — 288 turns, no failures.

Concurrent `DELETE`s for the same replica share one retirement rather than running one each, and the cleanup is keyed on the instance that was launched, not on the slot name — the name is reused by whatever replica comes next.

Startup output goes to `<log_dir>/<slot>.log` on the machine that runs the replica. A launch is detached, so that file is the only place a failed model load can be diagnosed.

### Security

`/replicas` starts processes on other machines. Set `--admin-token` and send `Authorization: Bearer <token>`; without one, anyone who can reach the port can launch and stop replicas. The router has no other authentication and belongs behind a trusted boundary.

### If the router restarts

An orderly shutdown (SIGTERM or SIGINT) drains and stops every replica the router launched. Statically listed replicas are left alone — only what it launched, it stops.

A crash runs no shutdown hook, so those replicas keep serving. Pass `--state-file` and the router records what it started; on the next start it takes each recorded device group back **before** probing anything, and only then asks what is on it:

| what answers | what happens |
|---|---|
| `/health` is `200` | adopted, routable immediately |
| anything else answers | adopted unroutable; the health poller promotes it once its model is up |
| nothing answers | **not** adopted, and the slot stays reserved as `unreachable-at-startup` |

That turns a router restart into a round trip rather than a fresh model load per replica. The last row is the one that matters on a shared cluster: a replica that was still loading when the router died answers nothing yet — uvicorn binds only once the engine is up — and neither does a healthy one behind a blip in the network, so a slot is never handed to a new launch on the strength of a probe that failed. Unresolved reservations are written to the state file too, so a stop that could not be confirmed is still reserved after a restart.

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
| `--initial-replicas` | `1` with hosts, else `0` | Replicas to launch at startup. |
| `--max-replicas` | declared pool | Cap launched replicas below the declared devices. |
| `--launch-timeout` | `600` | Seconds a launch may take to report ready before it is stopped. |
| `--drain-timeout` | `300` | Seconds to wait for in-flight requests before stopping a replica. |
| `--state-file` | none | Where to record launched replicas so a restart can adopt them. |
| `--admin-token` | none | Require a bearer token on `/replicas`. |

Raise `--affinity-slack` when prefills are expensive relative to queueing — long transcripts, few replicas. Lower it when replicas are easily saturated and a wait costs more than a re-prefill.

## Limits

- The router does not retry. A replica that fails mid-stream ends that request; the client resends.
- Load is counted as outstanding requests, not tokens, because the router never tokenizes.
- Growth is explicit: something has to call `POST /replicas`. The router does not decide on its own when more compute is needed.
- The pool is fixed at launch. Adding a *host* means restarting the router; adding a replica within the declared pool does not.
- Session pins are capped (100,000 by default) and evicted least-recently-used, so a flood of distinct session ids costs pins rather than memory.
