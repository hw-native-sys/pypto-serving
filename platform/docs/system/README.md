# System Layer

Source paths:

- `include/system/engine.hpp`
- `include/system/channels/messageTypeRegistry.hpp`
- `include/system/channels/`

The system layer provides the runtime shell for platform modules. It owns module lifecycle, cross-instance start/stop RPCs, and message type registration.

## Main Type

`serving::system::Engine` owns:

- A `HiCR::InstanceManager`.
- A `HiCR::ComputeManager`.
- A `HiCR::frontend::RPCEngine`.
- Current and deployer instance ids.
- Registered platform modules.
- A `channels::MessageTypeRegistry`.

## Engine Responsibilities

The engine provides:

- `addModule(name, module)` for unique named module registration.
- `initialize()` to initialize all registered modules.
- `run()` to start the deployer locally and start workers through RPC.
- `await()` to process stop RPCs while running, then terminate, await, and finalize modules.
- `terminate()` to stop workers and the deployer.
- `createInstance()` as a thin instance-manager wrapper.
- `getMessageTypeRegistry()` for message type registration.

## Cross-Instance Control

The engine registers two RPC targets:

- `__SERVING_SYSTEM_START_RPC_NAME`
- `__SERVING_SYSTEM_STOP_RPC_NAME`

The deployer instance sends start and stop RPCs to workers. Workers call `rpcEngine->listen()` in `run()` and wait for the deployer to start them.

## Relation To Platform Design

The engine gives the platform layer ownership during bootstrap, matching issues #32 and #13. It starts all platform modules before request execution is handed to model support and provides a place for passive management services to continue running.

## Current Limitations

- Engine stop is controlled by an atomic flag and RPC polling; it does not define failure recovery.
- Dynamic instance creation is only exposed as a pass-through and is not integrated with deployment state.
