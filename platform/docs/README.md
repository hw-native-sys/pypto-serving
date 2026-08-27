# Platform Documentation

The `platform/` subtree contains the first-party C++ platform-management layer for PyPTO Serving. It is separate from the Python model-serving path and is intended to manage distributed-system bootstrap, deployment metadata, channel lifecycle, module services, and instance lifecycle.

This layer follows the design split described in GitHub issues #32 and #13:

- The platform starts and manages the distributed system.
- Model support keeps ownership of LLM-specific behavior such as batching, KV cache policy, token scheduling, sampling, and PyPTO/Simpler execution.
- Platform code should not sit in the per-token execution hot path.
- Host-side tasks should orchestrate, supervise, and exchange control metadata.
- Device-side runtime instances should perform model execution and tensor movement when the model data path is integrated.

## Source Layout

```text
platform/
  include/modules/                  module interfaces and platform modules
  include/system/                   engine lifecycle and system utilities
  include/system/channels/          HiCR-backed channel primitives
  examples/                         executable examples for modules and engine lifecycle
  extern/                           vendored dependencies; not documented here
  build/                            generated build output; not documented here
```

## Documentation

- [Architecture](architecture.md): instance roles, module system, channel model, and scope.
- [Model Support Quick-Start](model-support-quickstart.md): wire your inference kernel into the coordinator→replica path, understand the `processFc` interface, and benchmark framework overhead.
- [Simpler Integration](simpler-integration.md): how replica ranks launch Simpler, the `processFc` interface, and the path to in-device tensor channels.
- [Model Support Interface](model-support-interface.md): what Model Support owns, provides, and will consume from the platform.
- [System](system/README.md): engine lifecycle and cross-instance start/stop control.
- [Channels](system/channels.md): payload, coordination, metadata, input, output, and message primitives.
- [Modules](modules/README.md): configuration, service, channel controller, and broadcast deployment.

## Runtime Shape

The current platform runtime is built around `serving::system::Engine`. The engine owns a set of `serving::modules::Module` instances, initializes them, starts them across instances through RPC, waits for termination, and finalizes them.

This PR covers the following building blocks:

- Engine lifecycle: cross-instance start/stop over RPC (`serving::system::Engine`).
- Module base interface: initialize/run/terminate/await/finalize lifecycle with optional `taskr::Service` (`serving::modules::Module`).
- Channel primitives: `Input`, `Output`, `Message`, and `MessageTypeRegistry` for host-side control traffic.
- Configuration types: `Deployment`, `Partition`, `Task`, `Edge`, `Replica`, `RequestManager` with JSON serialization and verification.
- `broadcastDeployment`: deployer-to-worker deployment config distribution over RPC.
- `channelController`: desired-vs-actual reconciliation loop for host-side SPSC channels.
- `service`: wraps `taskr::Runtime` for cooperative background services.

Dynamic scaling, fault recovery, `channelDispatcher`, `taskScheduler`, executor roles, heartbeat, `RuntimePlan` update protocol, in-device tensor channels, and Python bindings are deferred to follow-up PRs.
