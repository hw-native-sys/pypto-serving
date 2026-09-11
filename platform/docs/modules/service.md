# Service Module

Source path: `include/modules/service/module.hpp`

The service module adapts TaskR services into the platform module lifecycle. It lets the platform run one or more periodic control services even when there are no normal TaskR tasks.

## Main Type

`serving::modules::service::Module` derives from `serving::modules::Module` and owns a shared `taskr::Runtime`.

It maintains a map from service name to non-owning `taskr::Service *`.

## Behavior

`addService(name, service)` registers a service with the module. Names must be unique.

During `initialize()`:

- The TaskR runtime is configured with `setFinishOnLastTask(false)` so service-only workloads do not finish immediately.
- Registered services are added to the TaskR runtime.
- The TaskR runtime is initialized.

During `run()`:

- The TaskR runtime starts running.

During `terminate()`:

- The TaskR runtime is switched back to `setFinishOnLastTask(true)` so it can exit.

During `await()` and `finalize()`:

- The module waits for TaskR completion, finalizes the runtime, and clears the service registry.

## Example

`examples/modules/service/service.cpp` creates a simple periodic `helloWorld` service, registers it with `service::Module`, installs the module into `serving::system::Engine`, and terminates from the root instance after a short delay.

## Relation To Platform Design

This module supports the passive platform-management role from issues #32 and #13. After deployment bootstrap, periodic services such as heartbeat, monitoring, channel reconciliation, or future scaling decisions can run through TaskR without becoming part of model execution.