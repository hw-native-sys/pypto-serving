# Model Support Interface

This document describes what the Model Support layer needs to know and control when integrating with the platform.

## What the platform tells you

At startup, after `broadcastDeployment` completes, every rank knows:

- **Its role**: coordinator, replica, or request manager — determined by instance ID and the deployment config
- **Its device index** (replicas only): `deviceId = instanceId - numPartitions`
- **The full deployment graph**: partitions, tasks, edges, and which edges carry user-interface traffic

Model Support should use this information to initialize Simpler on replica ranks before the engine starts.

## What you provide

### 1. Compiled Simpler artifacts

Replica ranks need the five runtime library paths in `RuntimeBinaries`. The platform does not locate or validate them. Pass them via command-line arguments or configuration alongside the deployment JSON.

### 2. A `processFc` per task function name

For each task declared in the deployment configuration, Model Support registers a function:

```cpp
std::function<void(serving::modules::roles::TaskContext &context)>
```

This is the only point where Model Support code runs during execution. Everything else — job routing, input aggregation, output forwarding, channel lifecycle — is handled by the platform.

### 3. Request/response edge names

The deployment config declares a `RequestManager` with an input edge name and an output edge name. Model Support must ensure:

- The input edge name matches the edge from which user requests arrive
- The output edge name matches the edge to which results are written
- The task that reads the input edge does not have any other inter-partition inputs
- The task that writes the output edge does not have any other inter-partition outputs

These constraints are verified by `Deployment::verify()` at startup.

## What you do not own

| Concern | Owner |
|---|---|
| MPI rank management | Platform (via HiCR `InstanceManager`) |
| Channel creation and teardown | Platform (`channelController` module) |
| Deployment broadcast | Platform (`broadcastDeployment` module) |
| Job routing from coordinator to replica | Platform (`coordinator::Module`) |
| Input aggregation and output forwarding | Platform (`replica::Module`) |
| Engine lifecycle (init/run/terminate) | Platform (`Engine`) |
| Replica health monitoring | Platform (heartbeat, future) |

## RuntimePlan — future interface

The current PR does not yet expose a `RuntimePlan`. A future milestone will provide a versioned, observable snapshot of what the platform has actually instantiated:

- Which replicas are live, draining, or unavailable
- Which channels exist and what their endpoints are
- Safe `active → draining → removed` state transitions before any channel is deleted

Model Support will consume this object to make routing and scheduling decisions (KV-locality-aware replica selection, drain-aware request assignment, channel handle lookup) without duplicating platform resource state. The same object will be queryable by operators and higher-level control loops for observability.

Until `RuntimePlan` is implemented, Model Support should treat the deployment configuration as the ground truth of what is running.

## Tensor channel hot path — future

All channels are currently host-side. For the prefill/decode token data path, in-device tensor channels will carry tensor payloads directly between NPU devices without staging through host memory. When implemented:

- Replica `processFc` will receive device-memory `LocalMemorySlot` handles rather than host buffers
- `toDevice` / `toHost` staging in the process function becomes unnecessary
- The `processFc` signature does not change

Model Support should design the process function to be forward-compatible: check whether the incoming `LocalMemorySlot` is a host or device buffer and branch accordingly.
