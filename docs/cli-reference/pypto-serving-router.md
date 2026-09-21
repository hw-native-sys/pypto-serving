# `pypto-serving-router`

`pypto-serving-router` is an OpenAI-compatible reverse proxy over a fixed set of `pypto-serving` replicas. It keeps each conversation on the replica that holds its KV cache, and takes unreachable replicas out of rotation.

It requires no NPU, no model directory, and no tokenizer: request bodies are forwarded verbatim.

For the deployment walkthrough, see [Multi-Node Routing](../user-guide/multi-node-routing.md).

## Usage

```bash
pypto-serving-router --replicas replicas.json --port 8000
```

## Replica File

A JSON file with `replicas` (endpoints that already exist) and/or `hosts` (machines the router may launch on). At least one of the two must be present.

```json
{
  "replicas": [
    {"name": "node0", "host": "127.0.0.1", "port": 8001},
    {"name": "node1", "host": "10.0.0.2", "port": 8001}
  ]
}
```

### `replicas` — already running, never stopped by the router

| Field | Default | Description |
| --- | --- | --- |
| `host` | Required | Hostname or address of the replica. |
| `port` | Required | Port the replica's HTTP server listens on. |
| `name` | `host:port` | Label used in logs and on `/health`. |
| `scheme` | `http` | `http` or `https`. Request bodies and generated text cross this hop in the clear under `http`; set `https` when the replicas are not on a trusted network and TLS is terminated in front of them. |

### `hosts` — machines the router may launch on

One slot per declared device; a replica launched from a slot is owned by the router.

| Field | Default | Description |
| --- | --- | --- |
| `devices` | Required | Device ids the router may use. Declared, not discovered. |
| `devices_per_replica` | `1` | How many of those cards one replica needs — 1 for Qwen3-14B, 8 for DeepSeek V4, 16 for DSpark, where the count is compiled into the kernels. The host's capacity is `len(devices) // devices_per_replica`; a list that is not a whole multiple is refused. |
| `model` | Required | Model path **on that machine**. |
| `name` | `host<N>` | Label; slots are named `<name>-d<device>`. |
| `ssh` | none | ssh destination, e.g. `user@host`. Omitted means the machine the router runs on. |
| `identity_file` | none | Path to a private key. A path only — key material never appears in the config. |
| `port_base` | `8001` | A replica listens on `port_base +` the first device of its group. |
| `served_model_name` | checkpoint name | Should match across hosts, since the fleet serves one model. |
| `serve_args` | `[]` | Extra `pypto-serving` flags, passed through verbatim. |
| `env` | `{}` | Environment for the launched process. `{device}` (the group's first), `{devices}` (the whole comma-separated group) and `{port}` expand. |
| `launch_wrapper` | `[]` | Prefix such as `["task-submit", "--device", "{devices}", "--run"]`; the serving command is appended as one argument. Absent, the command runs bare. |
| `stop_command` | `pkill` by port | How to stop a replica. `{device}`, `{devices}` and `{port}` expand. |
| `python` | `python3` | Interpreter on that machine. |
| `workdir` | none | Directory to run from. |
| `log_dir` | `/tmp/pypto-serving-router` | Where a launched replica's startup output goes. A launch is detached, so this is the only record of a failed model load. |

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--replicas PATH` | Required | JSON file listing the replicas. |
| `--host HOST` | `0.0.0.0` | Host to bind the router. |
| `--port PORT` | `8000` | Port for the router. |
| `--session-ttl SECONDS` | `600` | Inactivity before a conversation loses its replica pin. Expiry drops the pin only; KV blocks are reclaimed by each replica's own cache. |
| `--affinity-slack N` | `8` | Extra outstanding requests a pinned replica may carry, relative to the least loaded one, before a session is re-routed. `0` disables affinity; a large value makes it strict. |
| `--health-interval SECONDS` | `5` | Interval between replica health probes. |
| `--request-timeout SECONDS` | `3600` | Overall timeout for one proxied request. There is no read timeout: decode is slow by design. |
| `--connect-timeout SECONDS` | `5` | Connect timeout per replica. |
| `--initial-replicas N` | `1` with hosts, else `0` | Replicas to launch at startup, filling slots in config order, local machine first. Refused at startup if it exceeds the pool. |
| `--max-replicas N` | declared pool | Cap launched replicas below the declared devices. |
| `--launch-timeout SECONDS` | `600` | How long a launch may take to report ready before it is stopped and its devices released. Measured loads: Qwen3-14B ~130s warm, DeepSeek V4 ~370s on eight cards. Load times move with the checkpoint and the stack, so time your own and leave headroom — too low means the router stops a replica that was loading normally. |
| `--drain-timeout SECONDS` | `300` | How long to wait for in-flight requests before stopping a replica. |
| `--state-file PATH` | none | Records launched replicas so a restarted router can adopt them instead of paying for a fresh model load. |
| `--admin-token TOKEN` | none | Require `Authorization: Bearer TOKEN` on `/replicas`. |

## Endpoints

| Endpoint | Behaviour |
| --- | --- |
| `POST /v1/completions` | Routed to a replica by session; response relayed unmodified. |
| `POST /v1/chat/completions` | Same, including streaming responses. |
| `GET /v1/models` | Answered by any routable replica. |
| `GET /health` | The router's own view of the replica table. 503 when no replica is routable. |
| `GET /replicas` | The fleet, with each replica's ready/draining/owned state, and the capacity left. |
| `POST /replicas` | Launch one replica on the next free declared device. `202` once started, `409` when the pool is full, `502` when the launch command failed. |
| `DELETE /replicas/{name}` | Drain then stop a replica the router launched. `409` for one it did not launch, `404` for an unknown name. |

## Session Identity

| | |
| --- | --- |
| Request | `X-Session-Id` header, else a top-level `session_id` in the JSON body. The header wins. |
| Response | `X-Session-Id` header, always, on streaming and non-streaming responses. |
| Absent | A new session id is minted and returned. |
| Unusable | An id longer than 128 characters, or outside `[A-Za-z0-9._:-]`, is replaced by a fresh one rather than rejected: it only decides routing, and it is echoed into a response header. |

A `session_id` left in the request body is ignored by the replica, which rejects no unknown fields.

## Exit Status

| Code | Meaning |
| --- | --- |
| `0` | The router shut down normally. |
| `1` | The replica file is missing, unreadable, or invalid — including an `--initial-replicas` above the declared pool. |
| `2` | Invalid command-line arguments (from `argparse`, before the replica file is read). |
