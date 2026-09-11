# Platform Architecture

## What the platform layer is

The platform layer is the **host-side control plane** for a distributed PyPTO Serving deployment. It manages:

- Instance lifecycle (start, stop, health)
- Deployment configuration distribution
- Host-side channel creation and teardown
- Module initialization and service scheduling

It is not a model execution layer. Model kernels, tensor movement, KV cache, batching, and token scheduling belong to Model Support and are executed through Simpler on each device. The platform exists so that Model Support does not have to manage MPI ranks, HiCR channels, or distributed lifecycle directly.

## Instance roles

Every MPI rank in a deployment plays one of four roles, determined by its instance ID and the deployment configuration:

| Role | Count | Responsibility |
|---|---|---|
| **Deployer** | 1 (root) | Reads deployment config, broadcasts it to all other ranks, then transitions to a coordinator or replica role |
| **Coordinator** | one per partition | Routes jobs to replicas, accumulates outputs, fires completion callbacks |
| **Replica** | one or more per partition | Executes model computation via Simpler on a dedicated NPU device |
| **Request Manager** | 1 | Entry point for client requests; maps to the partition owning the user-interface edge |

Device assignment for replicas: `deviceId = instanceId - numPartitions`. Each replica rank owns exactly one NPU.

## Module system

All platform behaviour is expressed as `serving::modules::Module` instances owned by the `serving::system::Engine`. The engine drives a fixed lifecycle:

```
initialize() → run() → [service loop] → terminate() → await() → finalize()
```

Modules can register a periodic background service (via `taskr::Service`) that runs inside the service loop between the `run()` and `terminate()` phases. The engine coordinates the lifecycle of all instances over RPC, so every rank progresses through the same phases in lockstep.

### Modules in this PR

| Module | Purpose |
|---|---|
| `broadcastDeployment` | Deployer sends deployment JSON to all workers over RPC at initialize time |
| `channelController` | Desired-vs-actual reconciliation loop; creates and tears down HiCR SPSC channels |
| `service` | Wraps `taskr::Runtime`; owns and drives background `taskr::Service` instances |

### Deferred modules (next PRs)

| Module | Purpose |
|---|---|
| `channelDispatcher` | Polls subscribed input channels; dispatches messages to registered handlers |
| `taskScheduler` | Registers named `taskr::Task` instances; drives taskr through the module lifecycle |
| `roles::coordinator` | Job queue and replica dispatch; completion callback when all outputs are gathered |
| `roles::replica` | Receives coordinator input, invokes `processFc`, returns outputs |
| `heartbeat` | Periodic health check between coordinators and replicas |

## Channel model

All channels in this PR are **host-side HiCR SPSC channels** carrying variable-size payloads and fixed-size metadata. They are used for control traffic and small tensors.

In-device tensor channels — where the hot path (prefill/decode token data) moves directly between NPU devices without staging through host memory — are **not implemented here**. They are the next major milestone and unblock the TP/PP tensor data path.

## What is deliberately out of scope

- Per-token scheduling (hot path)
- KV cache management
- Batching and sampling policy
- Python bindings
- `RuntimePlan` update protocol and watch/subscribe API
- Dynamic scaling, drain, and fault recovery
- Topology-aware placement
