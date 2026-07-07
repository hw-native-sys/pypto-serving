# Engine

Source path: `include/system/engine.hpp`

`serving::system::Engine` coordinates platform module lifecycle on each instance and uses RPC to start or stop worker instances from the deployer.

## Lifecycle

`initialize()`:

- Marks the engine as not running.
- Calls `initialize()` on every registered module.

`run()`:

- If running on the deployer, sends start RPCs to workers and starts itself.
- If running on a worker, listens for the start RPC.

`await()`:

- While running, polls for RPCs and executes them.
- After stopping, calls `terminate()`, `await()`, and `finalize()` on every module.

`terminate()`:

- If running on the deployer, sends stop RPCs to workers and stops itself.

## Module Contract

Modules are registered by unique name with `addModule()`. Duplicate names are rejected.

The engine does not inspect module type. Each module is responsible for implementing its own lifecycle semantics.

## Example

`examples/system/engine/engine.cpp` creates the HiCR managers, RPC engine, and serving engine, then runs the minimal lifecycle without installing any modules.

## Current Limitations

- No dependency graph exists between modules.
- No restart or replacement flow exists for failed modules or failed instances.
