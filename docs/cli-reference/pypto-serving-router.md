# `pypto-serving-router`

`pypto-serving-router` is an OpenAI-compatible reverse proxy over a fixed set of `pypto-serving` replicas. It keeps each conversation on the replica that holds its KV cache, and takes unreachable replicas out of rotation.

It requires no NPU, no model directory, and no tokenizer: request bodies are forwarded verbatim.

For the deployment walkthrough, see [Multi-Node Routing](../user-guide/multi-node-routing.md).

## Usage

```bash
pypto-serving-router --replicas replicas.json --port 8000
```

## Replica File

A JSON file listing the replicas. Either `{"replicas": [...]}` or a bare list is accepted.

```json
{
  "replicas": [
    {"name": "node0", "host": "127.0.0.1", "port": 8001},
    {"name": "node1", "host": "10.0.0.2", "port": 8001}
  ]
}
```

| Field | Default | Description |
| --- | --- | --- |
| `host` | Required | Hostname or address of the replica. |
| `port` | Required | Port the replica's HTTP server listens on. |
| `name` | `host:port` | Label used in logs and on `/health`. |
| `scheme` | `http` | `http` or `https`. Request bodies and generated text cross this hop in the clear under `http`; set `https` when the replicas are not on a trusted network and TLS is terminated in front of them. |

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

## Endpoints

| Endpoint | Behaviour |
| --- | --- |
| `POST /v1/completions` | Routed to a replica by session; response relayed unmodified. |
| `POST /v1/chat/completions` | Same, including streaming responses. |
| `GET /v1/models` | Answered by any routable replica. |
| `GET /health` | The router's own view of the replica table. 503 when no replica is routable. |

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
| `1` | The replica file is missing, unreadable, or invalid. |
| `2` | Invalid command-line arguments (from `argparse`, before the replica file is read). |
