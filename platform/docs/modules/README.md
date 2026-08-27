# Modules

Source paths:

- `include/modules/module.hpp`
- `include/modules/subscription.hpp`
- `include/modules/service/module.hpp`
- `include/modules/channelController/module.hpp`
- `include/modules/broadcastDeployment/module.hpp`

The modules layer defines reusable platform components that can be installed into `serving::system::Engine`. Modules use a common lifecycle and may optionally expose a periodic TaskR service.

## Base Module

`serving::modules::Module` is the abstract base class for all platform modules.

Main responsibilities:

- Provide a uniform lifecycle: `initialize()`, `run()`, `terminate()`, `await()`, and `finalize()`.
- Optionally create a `taskr::Service` when constructed with a positive interval.
- Let the engine discover and register periodic services through `hasService()` and `getService()`.

The base module does not define deployment semantics itself. It only defines how a platform component participates in the runtime lifecycle.

## Lifecycle

The engine calls module methods in this order:

1. `initialize()` before the instance is marked running.
2. `run()` when the deployer starts itself or sends start RPCs to workers.
3. `terminate()` after stop is requested.
4. `await()` after termination starts.
5. `finalize()` after module work has stopped.

Periodic modules implement their work in the protected `service()` method. The base constructor wraps that method in a `taskr::Service` using the requested interval.

## Subscription

`serving::modules::Subscription` binds a message type, an input channel, and a callback:

- Message type: `channels::Message::messageType_t`.
- Edge: `std::shared_ptr<channels::Input>`.
- Handler: `std::function<void(const std::shared_ptr<channels::Input>, const channels::Message &)>`.

This type is a small ownership wrapper for message-driven modules. It does not poll or dispatch by itself.

## Relation To Platform Design

The module interface supports the issue #32 goal that platform management should be a side service, not a model-execution abstraction. Modules can bootstrap deployment state, reconcile channels, or run control services without placing platform calls in the token-level model path.

## Future work
- More modules are coming